r"""
Regenerate Fig. 11 (classification explanations) and Fig. 12 (segmentation explanations)
and report WHICH SPLIT each TP/TN/FN/FP slice comes from.

Put this file in the same folder as model.py (class MGMTNet).

Install:  python -m pip install grad-cam matplotlib pandas opencv-python
Step 1:   python make_fig11_fig12.py --scan-only        # predictions + errors per split
Step 2:   python make_fig11_fig12.py                    # build the two figures
Option:   python make_fig11_fig12.py --local "F:\path\to\local_cohort"   # add the independent cohort
"""
import argparse, random
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from model import MGMTNet
# ----------------------------- CONFIG -----------------------------
CKPT = r"F:\Paper_Production_#1(Exp_1)\checkpoints\model_best.pth"
BASE = Path(r"F:\Paper_Production_#1(Exp_1)\Last_data\data")
SPLITS = {                                   # layout: <split>/images/{Stone,Non-Stone} and <split>/masks/{Stone,Non-Stone}
    "test": BASE / "test",
    "val":  BASE / "val",                    # skipped automatically if missing
}
IMG_NAMES   = ["images", "image", "Images", "Image"]
MASK_NAMES  = ["masks", "mask", "Masks", "Mask"]
CLASS_NAMES = {"Non-Stone": 0, "Stone": 1}   # change if your folders are spelled differently
OUT = Path("figs"); OUT.mkdir(exist_ok=True)
SEED, SIZE, THR = 0, 512, 0.5
# Fig. 12 layer choice: last decoder block of each stream (32 ch, full resolution).
# If "deepest" in your paper meant the block next to the bottleneck, use "dk4"/"ds4" instead.
SEG_LAYER_K, SEG_LAYER_S = "dk1", "ds1"
# ------------------------------------------------------------------

class MGMTNetCAM(MGMTNet):
    """Identical weights/maths to MGMTNet; only adds identity layers so CAM methods can hook F'' and A_sp."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.feat_out = nn.Identity()
        self.asp_out = nn.Identity()

    def forward(self, x):
        x1 = self.e1(x); x2 = self.e2(self.pool(x1)); x3 = self.e3(self.pool(x2)); x4 = self.e4(self.pool(x3))
        b_cnn = self.cnn_bot(self.pool(x4)); b_fused = b_cnn + self.trans_bot(b_cnn)
        dk4 = self.dk4(b_fused, x4); dk3 = self.dk3(dk4, x3); dk2 = self.dk2(dk3, x2); dk1 = self.dk1(dk2, x1)
        k_logits = self.out_k(dk1); m_k = torch.sigmoid(k_logits)
        ds4 = self.ds4(b_fused, x4); ds3 = self.ds3(ds4, x3); ds2 = self.ds2(ds3, x2); ds1 = self.ds1(ds2, x1)
        s_logits = self.out_s(ds1); m_s = torch.sigmoid(s_logits)
        masks = torch.cat([m_k, m_s], dim=1)
        masks_down = F.interpolate(masks, size=b_fused.shape[2:], mode="bilinear", align_corners=False)
        a_sp = self.asp_out(self.g_mask(masks_down))
        f_prime = b_fused * (1.0 + a_sp)
        se_pool = self.gap(f_prime).view(f_prime.size(0), -1)
        se_w = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se_pool)))).unsqueeze(-1).unsqueeze(-1)
        f2 = self.feat_out(f_prime * se_w)
        cls_logits = self.classifier(self.gap(f2).view(f2.size(0), -1))
        return {"kidney_logits": k_logits, "stone_logits": s_logits, "cls_logits": cls_logits,
                "m_k": m_k, "m_s": m_s, "a_sp": a_sp}


class ClsWrap(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)["cls_logits"]


class SegWrap(nn.Module):
    def __init__(self, m, key): super().__init__(); self.m, self.key = m, key
    def forward(self, x): return self.m(x)[self.key]


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck
    if isinstance(ck, dict):
        for k in ("model_state_dict", "state_dict", "model", "net"):
            if k in ck and isinstance(ck[k], dict): sd = ck[k]; break
        print("checkpoint keys:", [k for k in ck.keys() if not isinstance(ck[k], dict)][:12])
        for k in ("epoch", "best_epoch", "val_acc", "best_val_acc"):
            if k in ck: print(f"  {k} = {ck[k]}")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m = MGMTNetCAM()
    m.load_state_dict(sd, strict=True)          # fails loudly if this is not the full multi-task model
    print(f"loaded OK, parameters = {sum(p.numel() for p in m.parameters())/1e6:.2f} M (paper: 23.52 M)")
    return m.eval()


def load_img(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return g, torch.from_numpy(np.stack([g, g, g]))[None]


def first_existing(root, names):
    for n in names:
        if (Path(root) / n).is_dir(): return Path(root) / n
    return None


def find_mask(mask_root, cls_name, img_path):
    if mask_root is None: return None
    for ext in (".png", ".PNG", ".jpg", ".jpeg", ".tif", ".tiff"):
        c = mask_root / cls_name / (img_path.stem + ext)
        if c.exists(): return str(c)
    for pat in (f"{img_path.stem}*", f"*{img_path.stem}*"):          # e.g. name_mask.png, mask_name.png
        hits = sorted((mask_root / cls_name).glob(pat))
        if len(hits) == 1: return str(hits[0])
    return None


def load_mask(path):
    m = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return (m >= 1).astype(np.uint8), (m == 2).astype(np.uint8)   # kidney (incl. stone), stone


def scan(model):
    rows = []
    for split, root in SPLITS.items():
        root = Path(root); img_root, mask_root = first_existing(root, IMG_NAMES), first_existing(root, MASK_NAMES)
        if img_root is None: print(f"[skip] {split}: no images folder under {root}"); continue
        for cls_name, y in CLASS_NAMES.items():
            files = sorted(f for e in ("*.png", "*.jpg", "*.jpeg") for f in (img_root / cls_name).glob(e))
            print(f"{split}/{cls_name}: {len(files)} images")
            for f in files:
                _, x = load_img(f)
                with torch.no_grad(): p = torch.softmax(model(x)["cls_logits"], 1)[0, 1].item()
                rows.append(dict(split=split, file=str(f), mask=find_mask(mask_root, cls_name, f), y=y, p=p, pred=int(p >= THR)))
    df = pd.DataFrame(rows)
    if df.empty: raise SystemExit("No images found. Check BASE, IMG_NAMES and CLASS_NAMES.")
    df.to_csv(OUT / "all_predictions.csv", index=False)
    print("\nAccuracy per split:\n", df.assign(ok=df.pred == df.y).groupby("split").ok.mean().round(4))
    print("\nLowest P(stone) among stone slices / highest P(stone) among non-stone slices, per split")
    print("(manuscript, test set: 0.577 / 0.450):")
    for sp, g in df.groupby("split"):
        print(f"  {sp}: {g[g.y == 1].p.min():.3f} / {g[g.y == 0].p.max():.3f}")
    print("\nErrors per split:\n", df[df.pred != df.y].groupby("split").size())
    print("\nCandidate FN (stone, P 0.30-0.40):\n", df[(df.y == 1) & df.p.between(0.30, 0.40)][["split", "file", "p"]])
    print("\nCandidate FP (non-stone, P 0.55-0.61):\n", df[(df.y == 0) & df.p.between(0.55, 0.61)][["split", "file", "p"]])
    return df


def pick(df, tptn_split, seed=SEED):
    rnd = random.Random(seed)          # seed=None -> different images every run
    d0 = df[df.split == tptn_split] if tptn_split in set(df.split) else df
    tp, tn = d0[(d0.y == 1) & (d0.pred == 1)], d0[(d0.y == 0) & (d0.pred == 0)]
    fn, fp = df[(df.y == 1) & (df.pred == 0)], df[(df.y == 0) & (df.pred == 1)]
    if tp.empty or tn.empty: raise SystemExit("No correctly classified TP/TN slices found: check the checkpoint and paths.")
    sel = {"TP": tp.iloc[rnd.randrange(len(tp))], "TN": tn.iloc[rnd.randrange(len(tn))]}
    if len(fn): sel["FN"] = fn.sort_values("p").iloc[0]                    # most confident error
    if len(fp): sel["FP"] = fp.sort_values("p", ascending=False).iloc[0]
    for k in ("FN", "FP"):
        if k not in sel: print(f"[warn] no {k} in the scanned splits (expected on the test set: 0 errors)")
    return sel


def draw_gt(ax, row):
    if isinstance(row.mask, str):
        k, s = load_mask(row.mask)
        ax.contour(k, levels=[0.5], colors="cyan", linewidths=0.8)
        if s.any(): ax.contour(s, levels=[0.5], colors="white", linewidths=0.8)


ANON = False
def disp(g):
    """Display-only copy with the margins that usually carry burned-in patient/hospital text blanked.
    Check visually that no anatomy is removed; the model always sees the original image."""
    if not ANON: return g
    g = g.copy(); H, W = g.shape
    g[: int(.18 * H), :] = 0                      # top text band
    g[int(.88 * H):, : int(.22 * W)] = 0          # bottom-left text
    g[int(.88 * H):, int(.80 * W):] = 0           # bottom-right text
    g[:, : int(.05 * W)] = 0; g[:, int(.93 * W):] = 0   # orientation marker / scale bar
    return g


def overlay(g, cam):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    return show_cam_on_image(np.stack([disp(g)] * 3, -1), cv2.resize(cam, (SIZE, SIZE)), use_rgb=True)


def main(scan_only, local, ckpt, tptn, anon=False, seed=SEED):
    global CKPT, ANON
    ANON = anon
    if ckpt: CKPT = ckpt
    if local: SPLITS["local"] = Path(local)
    model = load_model(); df = scan(model)
    if scan_only: return
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget
    sel = pick(df, tptn, seed); cls_model = ClsWrap(model).eval()
    methods = {"Grad-CAM": GradCAM, "Grad-CAM++": GradCAMPlusPlus, "LayerCAM": LayerCAM}

    fig, ax = plt.subplots(len(sel), 5, figsize=(12, 2.6 * len(sel)), squeeze=False)
    for r, (tag, row) in enumerate(sel.items()):
        g, x = load_img(row.file)
        ax[r, 0].imshow(disp(g), cmap="gray"); draw_gt(ax[r, 0], row)
        ax[r, 0].set_title(f"{tag}  P={row.p:.2f} [{row.split}]", fontsize=8)
        for c, (name, M) in enumerate(methods.items(), start=1):
            with M(model=cls_model, target_layers=[model.feat_out]) as cam:
                m = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
            ax[r, c].imshow(overlay(g, m)); ax[r, c].set_title(name, fontsize=8)
        with torch.no_grad(): asp = model(x)["a_sp"][0, 0].numpy()
        ax[r, 4].imshow(overlay(g, (asp - asp.min()) / (np.ptp(asp) + 1e-8))); ax[r, 4].set_title("A_sp", fontsize=8)
        for a in ax[r]: a.axis("off")
    plt.tight_layout(h_pad=1.5); plt.savefig(OUT / "fig11_classification_explanations.png", dpi=600); plt.close()

    fig, ax = plt.subplots(len(sel), 3, figsize=(8, 2.6 * len(sel)), squeeze=False)
    for r, (tag, row) in enumerate(sel.items()):
        g, x = load_img(row.file)
        ax[r, 0].imshow(disp(g), cmap="gray"); draw_gt(ax[r, 0], row); ax[r, 0].set_title(f"{tag} P={row.p:.2f} [{row.split}]", fontsize=8)
        for c, (key, lname) in enumerate((("stone_logits", SEG_LAYER_S), ("kidney_logits", SEG_LAYER_K)), start=1):
            sw = SegWrap(model, key).eval()
            with torch.no_grad(): pm = (torch.sigmoid(sw(x))[0, 0] > 0.5).numpy().astype(np.float32)
            with GradCAM(model=sw, target_layers=[getattr(model, lname)]) as cam:
                m = cam(input_tensor=x, targets=[SemanticSegmentationTarget(0, pm)])[0]
            ax[r, c].imshow(overlay(g, m)); ax[r, c].set_title(f"Seg-Grad-CAM {key.split('_')[0]}", fontsize=8)
        for a in ax[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "fig12_segmentation_explanations.png", dpi=600); plt.close()

    pd.DataFrame(sel).T.to_csv(OUT / "selected_slices.csv")
    print("\nSelected slices (check the 'split' column):\n", pd.DataFrame(sel).T[["split", "file", "y", "p"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--local", default=None, help="folder of the independent cohort (images/ and masks/ subfolders)")
    ap.add_argument("--ckpt", default=None, help="override the checkpoint path")
    ap.add_argument("--tptn-split", default="test", help="split used to draw the TP and TN examples")
    ap.add_argument("--anonymise", action="store_true", help="blank margins with burned-in text in the DISPLAYED images")
    ap.add_argument("--seed", type=int, default=SEED, help="seed for choosing the TP/TN slices")
    ap.add_argument("--random", action="store_true", help="ignore the seed and pick different slices every run")
    a = ap.parse_args()
    main(a.scan_only, a.local, a.ckpt, a.tptn_split, a.anonymise, None if a.random else a.seed)
