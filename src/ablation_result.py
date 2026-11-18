"""
Test-time component-contribution analysis for MGMTNet (no retraining).

Components that can be switched off at inference:
  cnn_bottleneck     - CNN bottleneck branch's contribution to the fusion  (b_fused = b_cnn + b_trans)
  transformer        - Transformer bottleneck branch's contribution to the fusion
  skip_e1..skip_e4   - each encoder->decoder skip connection (both decoders)
  skips              - all four skip connections together (group, used for Shapley)
  kidney_decoder     - kidney decoder output (mask forced to background)
  stone_decoder      - stone decoder output (mask forced to background)
  spatial_attention  - mask-guided spatial modulation  f' = b_fused * (1 + A_sp)
  se_attention       - squeeze-and-excitation channel attention

Ablation modes:
  zero : component output replaced by zeros (spatial attention -> 0, SE weights -> 1,
         i.e. the identity, decoders -> constant background logit)
  mean : component output replaced by its mean over a calibration split (default: val)

Outputs (in --out):
  ablation_results.csv        per-component metrics + drop from baseline
  shapley_contributions.csv   Shapley values (if --shapley)
  fig_ablation_drop_*.png/pdf grouped bars of the drop per task
  fig_shapley.png/pdf         Shapley contribution bars (if --shapley)
  table_ablation_<mode>.tex   booktabs LaTeX table with drops, 95% bootstrap CI/p in the CSV
  table_shapley.tex           LaTeX Shapley table (if --shapley)

Usage:
  python ablation_MGMTNet.py --root data --ckpt checkpoints/model_best.pth \
         --model_module model --split test --calib_split val --size 512 \
         --device cpu --shapley
"""
import argparse
import importlib
import itertools
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EPS = 1e-7
BG_LOGIT = -10.0  # logit used to force a decoder output to "background"

SINGLE_COMPONENTS = [
    "cnn_bottleneck", "transformer",
    "skip_e1", "skip_e2", "skip_e3", "skip_e4",
    "kidney_decoder", "stone_decoder",
    "spatial_attention", "se_attention",
]
SHAPLEY_COMPONENTS = ["cnn_bottleneck", "transformer", "skips",
                      "spatial_attention", "se_attention"]
METRICS = ["kidney_dice", "kidney_iou", "stone_dice", "stone_iou",
           "cls_acc", "cls_f1", "cls_auc"]


# --------------------------------------------------------------------------- data
class SliceDataset(Dataset):
    """data/<split>/images/{Stone,Non-Stone}/x.png  +  data/<split>/masks/{...}/x.png
    mask values: 0 background, 1 kidney, 2 stone. Label from folder (Stone=1)."""

    def __init__(self, root, split, size):
        self.size = size
        self.items = []
        for cls, label in (("Non-Stone", 0), ("Stone", 1)):
            img_dir = Path(root) / split / "images" / cls
            msk_dir = Path(root) / split / "masks" / cls
            masks = {p.stem: p for p in msk_dir.glob("*")}
            for p in sorted(img_dir.glob("*")):
                if p.stem in masks:
                    self.items.append((p, masks[p.stem], label))
        if not self.items:
            raise RuntimeError(f"No image/mask pairs found under {Path(root) / split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp, y = self.items[i]
        s = self.size
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = np.repeat(img[None], 3, axis=0)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (s, s), interpolation=cv2.INTER_NEAREST)
        kidney = (m > 0).astype(np.float32)[None]
        stone = ((m == 2) | (m > 128)).astype(np.float32)[None]
        return (torch.from_numpy(img), torch.from_numpy(kidney),
                torch.from_numpy(stone), torch.tensor(y, dtype=torch.long))


# --------------------------------------------------------------------------- model
def load_model(args, device):
    mod = importlib.import_module(args.model_module)
    model = getattr(mod, args.model_class)(in_channels=3, num_classes=2)
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    return model.to(device).eval()


def forward_ablate(m, x, off=frozenset(), mode="zero", stats=None, record=None):
    """Re-implementation of MGMTNet.forward with switchable components.
    With off=() it is numerically identical to m(x)."""
    off = set(off)
    if "skips" in off:
        off |= {"skip_e1", "skip_e2", "skip_e3", "skip_e4"}

    def rec(key, t):
        if record is None:
            return
        s = t.sum((0, 2, 3), keepdim=True).double()
        n = t.shape[0] * t.shape[2] * t.shape[3]
        old = record.get(key)
        record[key] = (s, n) if old is None else (old[0] + s, old[1] + n)

    def repl(key, t, zero_val=0.0):
        if key not in off:
            return t
        if mode == "mean":
            return stats[key].to(device=t.device, dtype=t.dtype).expand_as(t)
        return torch.full_like(t, zero_val)

    # encoder
    x1 = m.e1(x)
    x2 = m.e2(m.pool(x1))
    x3 = m.e3(m.pool(x2))
    x4 = m.e4(m.pool(x3))
    for k, t in (("skip_e1", x1), ("skip_e2", x2), ("skip_e3", x3), ("skip_e4", x4)):
        rec(k, t)
    s1, s2, s3, s4 = (repl("skip_e1", x1), repl("skip_e2", x2),
                      repl("skip_e3", x3), repl("skip_e4", x4))

    # dual bottleneck + residual fusion
    b_cnn = m.cnn_bot(m.pool(x4))
    b_trans = m.trans_bot(b_cnn)
    rec("cnn_bottleneck", b_cnn)
    rec("transformer", b_trans)
    b_fused = repl("cnn_bottleneck", b_cnn) + repl("transformer", b_trans)

    # kidney decoder
    d = m.dk4(b_fused, s4)
    d = m.dk3(d, s3)
    d = m.dk2(d, s2)
    d = m.dk1(d, s1)
    k_logits = m.out_k(d)
    if "kidney_decoder" in off:
        k_logits = torch.full_like(k_logits, BG_LOGIT)
    m_k = torch.sigmoid(k_logits)

    # stone decoder
    d = m.ds4(b_fused, s4)
    d = m.ds3(d, s3)
    d = m.ds2(d, s2)
    d = m.ds1(d, s1)
    s_logits = m.out_s(d)
    if "stone_decoder" in off:
        s_logits = torch.full_like(s_logits, BG_LOGIT)
    m_s = torch.sigmoid(s_logits)

    # mask-guided spatial attention
    masks = torch.cat([m_k, m_s], dim=1)
    masks_down = F.interpolate(masks, size=b_fused.shape[2:], mode="bilinear", align_corners=False)
    a_sp = m.g_mask(masks_down)
    rec("spatial_attention", a_sp)
    a_sp = repl("spatial_attention", a_sp, zero_val=0.0)  # 0 -> f' = b_fused
    f_prime = b_fused * (1.0 + a_sp)

    # SE channel attention
    se_pool = m.gap(f_prime).view(f_prime.size(0), -1)
    w = torch.sigmoid(m.se_fc2(F.relu(m.se_fc1(se_pool)))).unsqueeze(-1).unsqueeze(-1)
    rec("se_attention", w)
    w = repl("se_attention", w, zero_val=1.0)  # 1 -> identity
    f2 = f_prime * w

    cls_feat = m.gap(f2).view(f2.size(0), -1)
    cls_logits = m.classifier(cls_feat)
    return {"kidney_logits": k_logits, "stone_logits": s_logits, "cls_logits": cls_logits}


@torch.no_grad()
def compute_stats(model, loader, device):
    record = {}
    for x, *_ in loader:
        forward_ablate(model, x.to(device), record=record)
    return {k: (s / n).float() for k, (s, n) in record.items()}


# --------------------------------------------------------------------------- metrics
def dice_iou(pred, gt):
    inter = (pred * gt).sum((1, 2, 3))
    p, g = pred.sum((1, 2, 3)), gt.sum((1, 2, 3))
    dice = (2 * inter + EPS) / (p + g + EPS)
    iou = (inter + EPS) / (p + g - inter + EPS)
    return dice, iou


@torch.no_grad()
def evaluate_raw(model, loader, device, off=frozenset(), mode="zero", stats=None):
    """Per-slice results (needed for bootstrap confidence intervals)."""
    kd, ki, sd, si, ys, preds, probs = [], [], [], [], [], [], []
    for x, km, sm, y in loader:
        x, km, sm = x.to(device), km.to(device), sm.to(device)
        out = forward_ablate(model, x, off, mode, stats)
        kp = (out["kidney_logits"] > 0).float()
        sp = (out["stone_logits"] > 0).float()
        d, i = dice_iou(kp, km)
        kd.append(d.cpu()); ki.append(i.cpu())
        d, i = dice_iou(sp, sm)
        has_stone = sm.sum((1, 2, 3)) > 0  # stone metrics on stone-positive slices only
        nan = torch.full_like(d, float("nan"))
        sd.append(torch.where(has_stone, d, nan).cpu())
        si.append(torch.where(has_stone, i, nan).cpu())
        pr = torch.softmax(out["cls_logits"], dim=1)
        probs.append(pr[:, 1].cpu()); preds.append(pr.argmax(1).cpu()); ys.append(y)
    return {"kd": torch.cat(kd).numpy(), "ki": torch.cat(ki).numpy(),
            "sd": torch.cat(sd).numpy(), "si": torch.cat(si).numpy(),
            "y": torch.cat(ys).numpy(), "pred": torch.cat(preds).numpy(),
            "prob": torch.cat(probs).numpy()}


def metrics_from_samples(S, idx=None):
    if idx is not None:
        S = {k: v[idx] for k, v in S.items()}
    try:
        auc = roc_auc_score(S["y"], S["prob"])
    except ValueError:
        auc = float("nan")
    return {
        "kidney_dice": float(np.nanmean(S["kd"])), "kidney_iou": float(np.nanmean(S["ki"])),
        "stone_dice": float(np.nanmean(S["sd"])), "stone_iou": float(np.nanmean(S["si"])),
        "cls_acc": accuracy_score(S["y"], S["pred"]),
        "cls_f1": f1_score(S["y"], S["pred"], zero_division=0), "cls_auc": auc,
    }


def evaluate(model, loader, device, off=frozenset(), mode="zero", stats=None):
    return metrics_from_samples(evaluate_raw(model, loader, device, off, mode, stats))


def bootstrap_ci(S, n_boot, seed=42):
    """95% percentile CI of each metric, resampling test slices."""
    rng = np.random.default_rng(seed)
    n = len(S["y"])
    vals = {k: [] for k in METRICS}
    for _ in range(n_boot):
        m = metrics_from_samples(S, rng.integers(0, n, n))
        for k in METRICS:
            vals[k].append(m[k])
    return {k: tuple(np.nanpercentile(vals[k], [2.5, 97.5])) for k in METRICS}


def bootstrap_drop(S_base, S_abl, n_boot, seed=42):
    """Paired bootstrap of (baseline - ablated): 95% CI and two-sided p-value."""
    rng = np.random.default_rng(seed)
    n = len(S_base["y"])
    drops = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mb, ma = metrics_from_samples(S_base, idx), metrics_from_samples(S_abl, idx)
        for k in METRICS:
            drops[k].append(mb[k] - ma[k])
    out = {}
    for k in METRICS:
        d = np.array(drops[k]); d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out[k] = (lo, hi, p)
    return out


# --------------------------------------------------------------------------- analyses
def single_ablation(model, loader, device, modes, stats, n_boot):
    rows = []
    S_base = evaluate_raw(model, loader, device)
    base = metrics_from_samples(S_base)
    ci = bootstrap_ci(S_base, n_boot)
    row = {"mode": "-", "component": "baseline", **base}
    for k in METRICS:
        row[f"ci_lo_{k}"], row[f"ci_hi_{k}"] = ci[k]
    rows.append(row)
    print("baseline:", {k: round(v, 4) for k, v in base.items()})
    for mode in modes:
        for comp in SINGLE_COMPONENTS:
            S = evaluate_raw(model, loader, device, {comp}, mode, stats)
            res = metrics_from_samples(S)
            bd = bootstrap_drop(S_base, S, n_boot)
            row = {"mode": mode, "component": f"no_{comp}", **res}
            for k in METRICS:
                row[f"drop_{k}"] = base[k] - res[k]
                row[f"reldrop_{k}_%"] = 100 * (base[k] - res[k]) / max(abs(base[k]), EPS)
                row[f"drop_ci_lo_{k}"], row[f"drop_ci_hi_{k}"], row[f"p_{k}"] = bd[k]
            rows.append(row)
            print(f"[{mode}] no_{comp}:", {k: round(res[k], 4) for k in METRICS})
    return pd.DataFrame(rows), base


def shapley(model, loader, device, mode, stats):
    comps = SHAPLEY_COMPONENTS
    n = len(comps)
    cache = {}

    def v(on):
        key = frozenset(on)
        if key not in cache:
            off = set(comps) - set(key)
            cache[key] = evaluate(model, loader, device, off, mode, stats)
            print(f"  coalition ON={sorted(key)} done ({len(cache)}/{2 ** n})")
        return cache[key]

    phi = {c: {k: 0.0 for k in METRICS} for c in comps}
    for c in comps:
        others = [o for o in comps if o != c]
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                with_c, without_c = v(set(S) | {c}), v(set(S))
                for k in METRICS:
                    phi[c][k] += w * (with_c[k] - without_c[k])
    rows = [{"component": c, "mode": mode, **phi[c]} for c in comps]
    df = pd.DataFrame(rows)
    full, empty = v(set(comps)), v(set())
    check = {k: full[k] - empty[k] for k in METRICS}
    return df, check


# --------------------------------------------------------------------------- plots
def save_fig(fig, base):
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_drops(df, mode, out):
    d = df[df["mode"] == mode].copy()
    d["component"] = d["component"].str.replace("no_", "", regex=False)
    cols = [("drop_kidney_dice", "Kidney Dice"), ("drop_stone_dice", "Stone Dice"),
            ("drop_cls_acc", "Classification acc."), ("drop_cls_auc", "Classification AUC")]
    x = np.arange(len(d))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (c, lab) in enumerate(cols):
        ax.bar(x + (i - 1.5) * w, d[c].values, w, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(d["component"], rotation=35, ha="right")
    ax.set_ylabel("Drop from baseline (absolute)")
    ax.set_title(f"Test-time component ablation ({mode} replacement)")
    ax.legend(frameon=False)
    save_fig(fig, str(Path(out) / f"fig_ablation_drop_{mode}"))


def plot_shapley(df, out):
    cols = [("kidney_dice", "Kidney Dice"), ("stone_dice", "Stone Dice"),
            ("cls_acc", "Classification acc."), ("cls_auc", "Classification AUC")]
    x = np.arange(len(df))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (c, lab) in enumerate(cols):
        ax.bar(x + (i - 1.5) * w, df[c].values, w, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["component"], rotation=25, ha="right")
    ax.set_ylabel("Shapley contribution")
    ax.set_title("Shapley-value contribution of each component")
    ax.legend(frameon=False)
    save_fig(fig, str(Path(out) / "fig_shapley"))


# --------------------------------------------------------------------------- LaTeX
PRETTY = {
    "baseline": "Full model (MGMTNet)",
    "no_cnn_bottleneck": "w/o CNN bottleneck branch", "no_transformer": "w/o Transformer bottleneck",
    "no_skip_e1": "w/o skip $E_1$", "no_skip_e2": "w/o skip $E_2$",
    "no_skip_e3": "w/o skip $E_3$", "no_skip_e4": "w/o skip $E_4$",
    "no_kidney_decoder": "w/o kidney decoder", "no_stone_decoder": "w/o stone decoder",
    "no_spatial_attention": "w/o mask-guided spatial attention", "no_se_attention": "w/o SE channel attention",
    "cnn_bottleneck": "CNN bottleneck branch", "transformer": "Transformer bottleneck",
    "skips": "Skip connections", "spatial_attention": "Mask-guided spatial attention",
    "se_attention": "SE channel attention",
}


def write_latex_ablation(df, mode, out):
    cols = [("kidney_dice", "Kidney Dice"), ("stone_dice", "Stone Dice"),
            ("cls_acc", "Acc."), ("cls_auc", "AUC")]
    d = pd.concat([df[df["component"] == "baseline"], df[df["mode"] == mode]])
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Test-time component ablation (" + mode + r" replacement) on the held-out test set. "
         r"Values are in \%; the number in parentheses is the change from the full model. "
         r"$^\dagger$ marks a significant drop (paired bootstrap over test slices, $p<0.05$).}",
         r"\label{tab:ablation_" + mode + "}",
         r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Configuration & " + " & ".join(l for _, l in cols) + r" \\", r"\midrule"]
    for _, r in d.iterrows():
        cells = []
        for k, _ in cols:
            if r["component"] == "baseline":
                cells.append(f"{100 * r[k]:.2f}")
            else:
                delta = f"{-100 * r['drop_' + k]:+.2f}".replace("-", "$-$")
                dag = r"$^\dagger$" if (r["p_" + k] < 0.05 and r["drop_" + k] > 0) else ""
                cells.append(f"{100 * r[k]:.2f} ({delta}){dag}")
        L.append(PRETTY.get(r["component"], r["component"]) + " & " + " & ".join(cells) + r" \\")
        if r["component"] == "baseline":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (Path(out) / f"table_ablation_{mode}.tex").write_text("\n".join(L), encoding="utf-8")


def write_latex_shapley(sdf, mode, out):
    cols = [("kidney_dice", "Kidney Dice"), ("stone_dice", "Stone Dice"),
            ("cls_acc", "Acc."), ("cls_auc", "AUC")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Shapley contribution of each component (" + mode + r" replacement), in percentage points.}",
         r"\label{tab:shapley}", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Component & " + " & ".join(l for _, l in cols) + r" \\", r"\midrule"]
    for _, r in sdf.iterrows():
        L.append(PRETTY.get(r["component"], r["component"]) + " & "
                 + " & ".join(f"{100 * r[k]:.2f}" for k, _ in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (Path(out) / "table_shapley.tex").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--ckpt", default="checkpoints/model_best.pth")
    ap.add_argument("--model_module", default="model", help="python module that defines the model class")
    ap.add_argument("--model_class", default="MGMTNet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib_split", default="val", help="split used for mean-ablation statistics")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modes", nargs="+", default=["zero", "mean"], choices=["zero", "mean"])
    ap.add_argument("--shapley", action="store_true")
    ap.add_argument("--shapley_mode", default="zero", choices=["zero", "mean"])
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for CIs / p-values")
    ap.add_argument("--out", default="ablation_results_v2")
    args = ap.parse_args()




    device = torch.device(args.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = load_model(args, device)

    test_loader = DataLoader(SliceDataset(args.root, args.split, args.size),
                             batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # sanity check: ablation forward == original forward when nothing is switched off
    x = next(iter(test_loader))[0][:2].to(device)
    with torch.no_grad():
        ref, mine = model(x), forward_ablate(model, x)
    for k in ("kidney_logits", "stone_logits", "cls_logits"):
        assert torch.allclose(ref[k], mine[k], atol=1e-4), f"forward mismatch in {k}"
    print("sanity check passed: ablation forward matches model.forward")

    stats = None
    if "mean" in args.modes or (args.shapley and args.shapley_mode == "mean"):
        calib = DataLoader(SliceDataset(args.root, args.calib_split, args.size),
                           batch_size=args.batch, shuffle=False, num_workers=args.workers)
        stats = compute_stats(model, calib, device)

    df, base = single_ablation(model, test_loader, device, args.modes, stats, args.boot)
    df.to_csv(Path(args.out) / "ablation_results.csv", index=False)
    for mode in args.modes:
        plot_drops(df, mode, args.out)
        write_latex_ablation(df, mode, args.out)

    if args.shapley:
        print(f"Shapley ({2 ** len(SHAPLEY_COMPONENTS)} coalitions) ...")
        sdf, check = shapley(model, test_loader, device, args.shapley_mode, stats)
        sdf.to_csv(Path(args.out) / "shapley_contributions.csv", index=False)
        plot_shapley(sdf, args.out)
        write_latex_shapley(sdf, args.shapley_mode, args.out)
        print("efficiency check (sum of Shapley values == full - empty):")
        for k in METRICS:
            print(f"  {k}: sum={sdf[k].sum():.4f}  full-empty={check[k]:.4f}")

    print(f"Done. Results in {args.out}/")


if __name__ == "__main__":
    main()