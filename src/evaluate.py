"""
Evaluation script (kidney-stone classification + kidney/stone segmentation).
Put this file next to train.py, dataset.py and model.py, then run:
    python evaluate.py
    python evaluate.py --splits val test --batch_size 8

Expected folder layout (same as train.py):
    data/<split>/images/{Stone,Non-Stone}/*.png
    data/<split>/masks/{Stone,Non-Stone}/*.png
Splits whose folders do not exist are skipped.
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import KidneyStoneDataset  # noqa: E402
from model import MGMTNet              # noqa: E402

CLASS_NAMES = ["Non-Stone", "Stone"]

# Same loss weights as train.py (used to reproduce the total loss)
ALPHA, BETA, GAMMA = 1.0, 1.5, 0.8


# ---------------------------------------------------------------------------
# Same loss as train.py, so loss values are comparable with training logs
# ---------------------------------------------------------------------------
class CombinedSegLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        inter = (probs * t).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + t.sum() + self.smooth)
        return bce_loss + dice_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    model = MGMTNet(in_channels=3, num_classes=2)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state:
                state = state[key]
                break
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def build_loader(root, split, batch_size, size, workers, pin):
    img_dir = os.path.join(root, split, "images")
    msk_dir = os.path.join(root, split, "masks")
    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        print(f"[skip] '{split}' split not found ({img_dir})")
        return None, None
    ds = KidneyStoneDataset(img_dir, msk_dir, target_size=size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=pin)
    return ds, loader


def dice_iou(pred, gt, eps=1e-7):
    """pred, gt: (B,1,H,W) binary. Returns per-image Dice and IoU."""
    p = pred.flatten(1).float()
    g = gt.flatten(1).float()
    inter = (p * g).sum(1)
    ps, gs = p.sum(1), g.sum(1)
    dice = (2 * inter + eps) / (ps + gs + eps)
    iou = (inter + eps) / (ps + gs - inter + eps)
    return dice.cpu().numpy(), iou.cpu().numpy()


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_split(model, loader, device, thresh, n_vis, desc):
    seg_crit, cls_crit = CombinedSegLoss().to(device), nn.CrossEntropyLoss().to(device)
    use_amp = device.type == "cuda"  # same as train.py

    sums = {"total": 0.0, "kidney": 0.0, "stone": 0.0, "cls": 0.0}
    labels, preds, probs = [], [], []
    seg = {o: {"dice": [], "iou": [], "gt_pos": []} for o in ("kidney", "stone")}
    vis = {"stone": [], "non_stone": []}

    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device)
        km = batch["kidney_mask"].to(device)
        sm = batch["stone_mask"].to(device)
        y = batch["label"].to(device)
        bs = x.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            lk = seg_crit(out["kidney_logits"], km)
            ls = seg_crit(out["stone_logits"], sm)
            lc = cls_crit(out["cls_logits"], y)
            total = ALPHA * lk + BETA * ls + GAMMA * lc

        sums["total"] += total.item() * bs
        sums["kidney"] += lk.item() * bs
        sums["stone"] += ls.item() * bs
        sums["cls"] += lc.item() * bs

        # Classification
        p = torch.softmax(out["cls_logits"].float(), 1)[:, 1]
        labels += y.cpu().tolist()
        preds += out["cls_logits"].argmax(1).cpu().tolist()
        probs += p.cpu().tolist()

        # Segmentation
        pk = torch.sigmoid(out["kidney_logits"].float()) > thresh
        ps = torch.sigmoid(out["stone_logits"].float()) > thresh
        for name, pm, gt in (("kidney", pk, km), ("stone", ps, sm)):
            d, i = dice_iou(pm, gt > 0.5)
            seg[name]["dice"] += d.tolist()
            seg[name]["iou"] += i.tolist()
            seg[name]["gt_pos"] += (gt.flatten(1).sum(1) > 0).cpu().tolist()

        # Keep a few samples for visual overlays
        for b in range(bs):
            key = "stone" if y[b].item() == 1 else "non_stone"
            if len(vis[key]) < max(n_vis // 2, 1):
                vis[key].append(dict(
                    img=x[b, 0].cpu().numpy(),
                    gk=km[b, 0].cpu().numpy(), gs=sm[b, 0].cpu().numpy(),
                    pk=pk[b, 0].cpu().numpy(), ps=ps[b, 0].cpu().numpy(),
                    label=int(y[b]), pred=int(preds[-bs + b]),
                    prob=float(probs[-bs + b])))

    n = len(labels)
    return dict(
        n=n,
        losses={k: v / n for k, v in sums.items()},
        labels=np.array(labels), preds=np.array(preds), probs=np.array(probs),
        seg={o: {k: np.array(v) for k, v in d.items()} for o, d in seg.items()},
        vis=vis["stone"] + vis["non_stone"],
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(name, r, out_dir):
    y, p, pr = r["labels"], r["preds"], r["probs"]
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "split": name,
        "n_images": int(r["n"]),
        "accuracy_%": 100.0 * (tp + tn) / max(cm.sum(), 1),
        "sensitivity_recall_stone": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision_stone": tp / max(tp + fp, 1),
        "f1_stone": 2 * tp / max(2 * tp + fp + fn, 1),
        "roc_auc": float(roc_auc_score(y, pr)) if len(np.unique(y)) > 1 else float("nan"),
        "loss_total": r["losses"]["total"],
        "loss_kidney": r["losses"]["kidney"],
        "loss_stone": r["losses"]["stone"],
        "loss_cls": r["losses"]["cls"],
    }
    for organ in ("kidney", "stone"):
        d, i, pos = (r["seg"][organ][k] for k in ("dice", "iou", "gt_pos"))
        pos = pos.astype(bool)
        metrics[f"{organ}_dice_all"] = float(d.mean())
        metrics[f"{organ}_iou_all"] = float(i.mean())
        metrics[f"{organ}_dice_gt_positive"] = float(d[pos].mean()) if pos.any() else float("nan")
        metrics[f"{organ}_iou_gt_positive"] = float(i[pos].mean()) if pos.any() else float("nan")

    print(f"\n{'=' * 4} {name.upper()} ({r['n']} images) {'=' * 4}")
    print(f"Loss  total {metrics['loss_total']:.4f} | kidney {metrics['loss_kidney']:.3f} "
          f"| stone {metrics['loss_stone']:.3f} | cls {metrics['loss_cls']:.3f}")
    print(f"Accuracy {metrics['accuracy_%']:.2f}% | Sensitivity {metrics['sensitivity_recall_stone']:.4f} "
          f"| Specificity {metrics['specificity']:.4f} | AUC {metrics['roc_auc']:.4f}")
    print(classification_report(y, p, labels=[0, 1], target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    for organ in ("kidney", "stone"):
        print(f"{organ.capitalize():6s} seg | Dice all {metrics[f'{organ}_dice_all']:.4f} "
              f"| Dice GT-positive {metrics[f'{organ}_dice_gt_positive']:.4f} "
              f"| IoU all {metrics[f'{organ}_iou_all']:.4f} "
              f"| IoU GT-positive {metrics[f'{organ}_iou_gt_positive']:.4f}")

    # --- confusion matrix figure ---
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {name}")
    for (a, b), v in np.ndenumerate(cm):
        ax.text(b, a, int(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"), dpi=150)
    plt.close(fig)

    # --- ROC curve ---
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, pr)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC - {name}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"roc_{name}.png"), dpi=150)
        plt.close(fig)

    return metrics


def overlay(img, k, s):
    rgb = np.stack([img] * 3, -1).astype(np.float32)
    rgb[k > 0] = 0.65 * rgb[k > 0] + 0.35 * np.array([0, 1, 0])   # kidney = green
    rgb[s > 0] = 0.35 * rgb[s > 0] + 0.65 * np.array([1, 0, 0])   # stone  = red
    return np.clip(rgb, 0, 1)


def save_overlays(name, samples, out_dir):
    if not samples:
        return
    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)), squeeze=False)
    for r_i, s in enumerate(samples):
        axes[r_i, 0].imshow(s["img"], cmap="gray")
        axes[r_i, 0].set_title(f"Input | true: {CLASS_NAMES[s['label']]}", fontsize=9)
        axes[r_i, 1].imshow(overlay(s["img"], s["gk"], s["gs"]))
        axes[r_i, 1].set_title("Ground truth (green=kidney, red=stone)", fontsize=9)
        axes[r_i, 2].imshow(overlay(s["img"], s["pk"], s["ps"]))
        axes[r_i, 2].set_title(f"Prediction: {CLASS_NAMES[s['pred']]} (p={s['prob']:.2f})", fontsize=9)
        for a in axes[r_i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"segmentation_examples_{name}.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model_best.pth")
    ap.add_argument("--data_root", default="./Last_data/data")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=16)   # same as training
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num_vis", type=int, default=8)
    ap.add_argument("--out_dir", default="eval_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    print(f"Device: {device}")
    if not is_cuda:
        torch.set_num_threads(8)

    model = load_model(args.ckpt, device)
    print(f"Loaded weights from {args.ckpt}")

    all_metrics = []
    for split in args.splits:
        ds, loader = build_loader(args.data_root, split, args.batch_size, (512, 512),
                                  4 if is_cuda else 0, is_cuda)
        if loader is None:
            continue
        r = run_split(model, loader, device, args.threshold, args.num_vis, f"Eval {split}")
        all_metrics.append(summarize(split, r, args.out_dir))
        save_overlays(split, r["vis"], args.out_dir)

        # Per-image predictions (useful for finding the misclassified slices)
        df = pd.DataFrame({
            "image": ds.image_paths,
            "true_label": r["labels"],
            "pred_label": r["preds"],
            "stone_prob": r["probs"],
            "kidney_dice": r["seg"]["kidney"]["dice"],
            "stone_dice": r["seg"]["stone"]["dice"],
            "correct": r["labels"] == r["preds"],
        })
        df.to_csv(os.path.join(args.out_dir, f"predictions_{split}.csv"), index=False)

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAll results saved in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
