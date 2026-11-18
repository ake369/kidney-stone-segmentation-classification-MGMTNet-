#!/usr/bin/env python
"""
XAI for MGMTNet (multi-task: kidney mask, stone mask, stone / non-stone classification)
=========================================================================================
Self-contained: only needs torch, numpy, opencv-python, matplotlib (scikit-learn optional, for t-SNE).
Nothing in your training code or checkpoint is modified.

What it produces (all in --out, default XAI_visuals/), PNG @300 dpi + vector PDF:
  Fig_A_classification_explanations   rows = TP / TN / FP / FN cases
        columns = CT | GT | Grad-CAM | Grad-CAM++ | LayerCAM | Integrated Gradients | Occlusion | mask-guided attention
  Fig_B_segmentation_explanations     Seg-Grad-CAM for the stone and kidney decoders
  Fig_C_quantitative_evaluation       deletion / insertion curves, energy-in-mask, pointing game
  Fig_D_sanity_check                  cascading weight-randomisation test (Adebayo et al., NeurIPS 2018)
  Fig_E_feature_embedding             t-SNE (or PCA) of the pooled classification feature
  table_xai_metrics.csv / .tex        mean, SD and 95% bootstrap CI for every method / metric
  per_image_metrics.csv               raw per-image numbers

Typical use (CPU):
  python xai_MGMTNet.py --ckpt checkpoints/model_best.pth --test_root data/test --out XAI_visuals
Run only parts:  --run figs,metrics,sanity,tsne
"""
import os
import glob
import csv
import time
import copy
import argparse
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =====================================================================================
# 1. MODEL (verbatim copy of your definition)
# =====================================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransformerBottleneck(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        transformed = self.transformer(tokens)
        out = transformed.transpose(1, 2).view(B, C, H, W)
        return out



class MGMTNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.e1 = ConvBlock(in_channels, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.e4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.cnn_bot = ConvBlock(256, 512)
        self.trans_bot = TransformerBottleneck(embed_dim=512, num_heads=8, num_layers=4)
        self.dk4 = DecoderBlock(512, 256, 256)
        self.dk3 = DecoderBlock(256, 128, 128)
        self.dk2 = DecoderBlock(128, 64, 64)
        self.dk1 = DecoderBlock(64, 32, 32)
        self.out_k = nn.Conv2d(32, 1, kernel_size=1)
        self.ds4 = DecoderBlock(512, 256, 256)
        self.ds3 = DecoderBlock(256, 128, 128)
        self.ds2 = DecoderBlock(128, 64, 64)
        self.ds1 = DecoderBlock(64, 32, 32)
        self.out_s = nn.Conv2d(32, 1, kernel_size=1)
        self.g_mask = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid())
        self.se_fc1 = nn.Linear(512, 32)
        self.se_fc2 = nn.Linear(32, 512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        b_cnn = self.cnn_bot(self.pool(x4))
        b_trans = self.trans_bot(b_cnn)
        b_fused = b_cnn + b_trans
        dk4 = self.dk4(b_fused, x4)
        dk3 = self.dk3(dk4, x3)
        dk2 = self.dk2(dk3, x2)
        dk1 = self.dk1(dk2, x1)
        k_logits = self.out_k(dk1)
        m_k = torch.sigmoid(k_logits)
        ds4 = self.ds4(b_fused, x4)
        ds3 = self.ds3(ds4, x3)
        ds2 = self.ds2(ds3, x2)
        ds1 = self.ds1(ds2, x1)
        s_logits = self.out_s(ds1)
        m_s = torch.sigmoid(s_logits)
        masks = torch.cat([m_k, m_s], dim=1)
        masks_down = F.interpolate(masks, size=b_fused.shape[2:], mode="bilinear", align_corners=False)
        a_sp = self.g_mask(masks_down)
        f_prime = b_fused * (1.0 + a_sp)
        se_pool = self.gap(f_prime).view(f_prime.size(0), -1)
        se_weights = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se_pool)))).unsqueeze(-1).unsqueeze(-1)
        f_double_prime = f_prime * se_weights
        cls_feat = self.gap(f_double_prime).view(f_double_prime.size(0), -1)
        cls_logits = self.classifier(cls_feat)
        return {"kidney_logits": k_logits, "stone_logits": s_logits, "cls_logits": cls_logits,
                "m_k": m_k, "m_s": m_s}


def forward_ex(model, x):
    """Identical computation to MGMTNet.forward, but also returns the intermediate tensors
    needed for explanations (final pre-GAP feature map, decoder features, attention map)."""
    x1 = model.e1(x)
    x2 = model.e2(model.pool(x1))
    x3 = model.e3(model.pool(x2))
    x4 = model.e4(model.pool(x3))
    b_cnn = model.cnn_bot(model.pool(x4))
    b_trans = model.trans_bot(b_cnn)
    b_fused = b_cnn + b_trans
    dk4 = model.dk4(b_fused, x4)
    dk3 = model.dk3(dk4, x3)
    dk2 = model.dk2(dk3, x2)
    dk1 = model.dk1(dk2, x1)
    k_logits = model.out_k(dk1)
    m_k = torch.sigmoid(k_logits)
    ds4 = model.ds4(b_fused, x4)
    ds3 = model.ds3(ds4, x3)
    ds2 = model.ds2(ds3, x2)
    ds1 = model.ds1(ds2, x1)
    s_logits = model.out_s(ds1)
    m_s = torch.sigmoid(s_logits)
    masks = torch.cat([m_k, m_s], dim=1)
    masks_down = F.interpolate(masks, size=b_fused.shape[2:], mode="bilinear", align_corners=False)
    a_sp = model.g_mask(masks_down)
    f_prime = b_fused * (1.0 + a_sp)
    se_pool = model.gap(f_prime).view(f_prime.size(0), -1)
    se_w = torch.sigmoid(model.se_fc2(F.relu(model.se_fc1(se_pool)))).unsqueeze(-1).unsqueeze(-1)
    f_final = f_prime * se_w
    cls_feat = model.gap(f_final).view(f_final.size(0), -1)
    cls_logits = model.classifier(cls_feat)
    return {"cls_logits": cls_logits, "kidney_logits": k_logits, "stone_logits": s_logits,
            "m_k": m_k, "m_s": m_s, "a_sp": a_sp, "b_fused": b_fused, "f_final": f_final,
            "dk4": dk4, "ds4": ds4, "cls_feat": cls_feat}


# =====================================================================================
# 2. DATA (same preprocessing as your KidneyStoneDataset: gray/255, 3-channel, no normalisation)
# =====================================================================================
def discover_test_items(root):
    """Last_data/data/test/images/{Stone,Non-Stone}/*.png  +  Last_data/data/test/masks/{Stone,Non-Stone}/*.png"""
    mask_lookup = {}
    for folder in ["Stone", "Non-Stone"]:
        for m in glob.glob(os.path.join(root, "masks", folder, "*.png")):
            mask_lookup[os.path.splitext(os.path.basename(m))[0]] = m
    items = []
    for folder, label in [("Stone", 1), ("Non-Stone", 0)]:
        for p in sorted(glob.glob(os.path.join(root, "images", folder, "*.png"))):
            base = os.path.splitext(os.path.basename(p))[0]
            if base in mask_lookup:
                items.append({"img": p, "msk": mask_lookup[base], "label": label, "name": f"{folder}/{base}"})
    if not items:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")
    return items


def load_item(it, size):
    img = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(it["msk"], cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        raise FileNotFoundError(it["img"])
    if img.shape != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (size, size), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    kidney = (msk > 0).astype(np.uint8)
    stone = ((msk == 2) | (msk > 128)).astype(np.uint8)
    return img, kidney, stone


def to_tensor(img, device):
    return torch.from_numpy(img)[None, None].repeat(1, 3, 1, 1).to(device)


# =====================================================================================
# 3. ATTRIBUTION METHODS
# =====================================================================================
def _norm01(a):
    a = a - a.min()
    m = a.max()
    return a / m if m > 1e-12 else np.zeros_like(a)


def _upsample_norm(cam_t, size):
    cam = F.interpolate(cam_t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return _norm01(cam)


def cam_gradcam(A, G):
    w = G.mean(dim=(2, 3), keepdim=True)
    return F.relu((w * A).sum(1, keepdim=True))


def cam_gradcampp(A, G):
    g2, g3 = G ** 2, G ** 3
    denom = 2 * g2 + A.sum(dim=(2, 3), keepdim=True) * g3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha = g2 / denom
    w = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)
    return F.relu((w * A).sum(1, keepdim=True))


def cam_layercam(A, G):
    return F.relu((F.relu(G) * A).sum(1, keepdim=True))


def cls_cams(model, x, target=1):
    """Grad-CAM / Grad-CAM++ / LayerCAM on the final pre-GAP feature map (B,512,H/16,W/16)
    for the logit of `target` (1 = stone). Returns dict of normalised (H,W) maps + class probs."""
    size = x.shape[-1]
    model.eval()
    with torch.enable_grad():
        out = forward_ex(model, x)
        A = out["f_final"]
        score = out["cls_logits"][0, target]
        G = torch.autograd.grad(score, A)[0]
    A, G = A.detach(), G.detach()
    probs = F.softmax(out["cls_logits"].detach(), dim=1)[0].cpu().numpy()
    cams = {"Grad-CAM": _upsample_norm(cam_gradcam(A, G), size),
            "Grad-CAM++": _upsample_norm(cam_gradcampp(A, G), size),
            "LayerCAM": _upsample_norm(cam_layercam(A, G), size)}
    return cams, probs


def seg_gradcam(model, x, which="stone", thr=0.5):
    """Seg-Grad-CAM (Vinogradova et al. 2020). Score = sum of logits over the predicted mask.
    Layer: deepest decoder stage (ds4 / dk4, 256 ch). NOTE: the last decoder stage is deliberately not used,
    because the 1x1 output conv is linear and Grad-CAM there collapses to the logit map itself."""
    size = x.shape[-1]
    model.eval()
    with torch.enable_grad():
        out = forward_ex(model, x)
        logits = out[f"{which}_logits"]
        A = out["ds4" if which == "stone" else "dk4"]
        sel = (torch.sigmoid(logits) > thr).float()
        fallback = False
        if sel.sum() == 0:  # nothing predicted -> explain the 0.1% most 'stone-like' pixels
            k = max(1, int(0.001 * logits.numel()))
            tv = torch.topk(logits.flatten(), k).values[-1]
            sel = (logits >= tv).float()
            fallback = True
        score = (logits * sel).sum()
        G = torch.autograd.grad(score, A)[0]
    cam = _upsample_norm(cam_gradcam(A.detach(), G.detach()), size)
    return cam, fallback


def integrated_gradients(model, x, target=1, steps=24, bs=4):
    model.eval()
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    for chunk in alphas.split(bs):
        xs = (base + chunk.view(-1, 1, 1, 1).to(x.device) * (x - base)).requires_grad_(True)
        with torch.enable_grad():
            score = model(xs)["cls_logits"][:, target].sum()
            g = torch.autograd.grad(score, xs)[0]
        total += g.sum(0, keepdim=True)
    ig = ((x - base) * total / steps).sum(1)[0].clamp(min=0).cpu().numpy()  # positive evidence only
    ig = cv2.GaussianBlur(ig, (0, 0), 4)
    ig = np.clip(ig / (np.percentile(ig, 99.5) + 1e-12), 0, 1)
    return ig


@torch.no_grad()
def occlusion_map(model, x, target=1, patch=64, stride=48, bs=8):
    model.eval()
    H = x.shape[-1]
    p0 = F.softmax(model(x)["cls_logits"], 1)[0, target].item()
    pos = list(range(0, H - patch + 1, stride))
    if pos[-1] != H - patch:
        pos.append(H - patch)
    coords = [(i, j) for i in pos for j in pos]
    sens = np.zeros((H, H), np.float32)
    cnt = np.zeros((H, H), np.float32)
    for s in range(0, len(coords), bs):
        chunk = coords[s:s + bs]
        xs = x.repeat(len(chunk), 1, 1, 1).clone()
        for k, (i, j) in enumerate(chunk):
            xs[k, :, i:i + patch, j:j + patch] = 0.0
        p = F.softmax(model(xs)["cls_logits"], 1)[:, target].cpu().numpy()
        for k, (i, j) in enumerate(chunk):
            sens[i:i + patch, j:j + patch] += p0 - p[k]
            cnt[i:i + patch, j:j + patch] += 1
    sens = np.clip(sens / np.maximum(cnt, 1), 0, None)
    return _norm01(sens)


def random_map(rng, size, grid):
    r = rng.random((grid, grid)).astype(np.float32)
    return _norm01(cv2.resize(r, (size, size), interpolation=cv2.INTER_LINEAR))


# =====================================================================================
# 4. QUANTITATIVE METRICS
# =====================================================================================
_trapz = getattr(np, "trapezoid", None) or np.trapz


def loc_metrics(cam, gt_stone, gt_kidney, tol):
    tot = cam.sum() + 1e-12
    eim = float(cam[gt_stone > 0].sum() / tot)
    area = float((gt_stone > 0).mean())
    kid = float(cam[gt_kidney > 0].sum() / tot)
    pk = np.unravel_index(int(cam.argmax()), cam.shape)
    strict = bool(gt_stone[pk] > 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    tolerant = bool(cv2.dilate(gt_stone, k)[pk] > 0)
    return {"energy_in_stone": eim, "energy_ratio": eim / max(area, 1e-12), "energy_in_kidney": kid,
            "pointing_strict": float(strict), "pointing_tol": float(tolerant), "stone_area_frac": area}


@torch.no_grad()
def _probs(model, batch, target, bs):
    out = []
    for s in range(0, batch.shape[0], bs):
        out.append(F.softmax(model(batch[s:s + bs])["cls_logits"], 1)[:, target].cpu().numpy())
    return np.concatenate(out)


def deletion_insertion(model, img, sal, fracs, target=1, bs=4, device="cpu", rng=None):
    """Deletion: remove most-salient pixels first (replace by 0). Insertion: reveal most-salient first onto a blurred copy.
    AUC computed over fraction of pixels in [0,1] on a non-uniform grid (dense at small fractions)."""
    H = img.shape[0]
    n = H * H
    jitter = (rng.random(n) * 1e-9) if rng is not None else 0.0
    order = np.argsort(-(sal.ravel() + jitter), kind="stable")
    blur = cv2.GaussianBlur(img, (0, 0), 10)
    dels, inss = [], []
    for f in fracs:
        m = np.zeros(n, np.float32)
        m[order[:int(round(f * n))]] = 1.0
        m = m.reshape(H, H)
        dels.append(img * (1 - m))
        inss.append(blur * (1 - m) + img * m)
    stack = np.stack(dels + inss)[:, None].repeat(3, 1)
    p = _probs(model, torch.from_numpy(stack).to(device), target, bs)
    pd, pi = p[:len(fracs)], p[len(fracs):]
    return float(_trapz(pd, fracs)), float(_trapz(pi, fracs)), pd, pi


# =====================================================================================
# 5. UTILITIES: stats, plotting
# =====================================================================================
def mean_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bm = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, np.percentile(bm, 2.5), np.percentile(bm, 97.5)


def rankdata(a):
    a = np.asarray(a).ravel()
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    c = np.cumsum(cnt)
    return ((c - cnt + 1 + c) / 2.0)[inv]


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8})


def save_fig(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png/.pdf")


def show_overlay(ax, img, cam=None, stone=None, kidney=None, cmap="jet", alpha=0.45):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    if cam is not None:
        ax.imshow(cam, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    if kidney is not None and kidney.any():
        ax.contour(kidney, levels=[0.5], colors="cyan", linewidths=0.6)
    if stone is not None and stone.any():
        ax.contour(stone, levels=[0.5], colors="white", linewidths=0.9)
    ax.axis("off")


def add_colorbar(fig, cmap, label="Relative importance"):
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=7)


# =====================================================================================
# 6. PIPELINE STEPS
# =====================================================================================
def build_model(ckpt, device):
    model = MGMTNet(in_channels=3, num_classes=2)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_all(model, items, size, device, bs=4):
    probs, feats = [], []
    for s in range(0, len(items), bs):
        xs = []
        for it in items[s:s + bs]:
            img, _, _ = load_item(it, size)
            xs.append(img)
        x = torch.from_numpy(np.stack(xs))[:, None].repeat(1, 3, 1, 1).to(device)
        out = forward_ex(model, x)
        probs.append(F.softmax(out["cls_logits"], 1)[:, 1].cpu().numpy())
        feats.append(out["cls_feat"].cpu().numpy())
        print(f"\r  predicting {min(s + bs, len(items))}/{len(items)}", end="")
    print()
    return np.concatenate(probs), np.concatenate(feats)


def select_cases(items, probs, per_group, seed, include):
    rng = np.random.default_rng(seed)
    lab = np.array([it["label"] for it in items])
    pred = (probs >= 0.5).astype(int)
    groups = {"TP": np.where((lab == 1) & (pred == 1))[0], "TN": np.where((lab == 0) & (pred == 0))[0],
              "FN": np.where((lab == 1) & (pred == 0))[0], "FP": np.where((lab == 0) & (pred == 1))[0]}
    chosen = []
    for g in ["TP", "TN", "FN", "FP"]:
        idx = groups[g]
        if len(idx) == 0:
            print(f"  (no {g} cases in the test set)")
            continue
        if g in ("TP", "TN"):  # random, seeded -> no cherry-picking
            pick = rng.choice(idx, size=min(per_group, len(idx)), replace=False)
        else:  # errors: most confident mistakes first
            pick = idx[np.argsort(-np.abs(probs[idx] - 0.5))][:per_group]
        chosen += [(g, int(i)) for i in pick]
    if include:
        names = {it["name"]: i for i, it in enumerate(items)}
        for n in include:
            hit = [k for k in names if k.endswith(n) or n in k]
            for k in hit[:1]:
                i = names[k]
                g = ("TP" if pred[i] == 1 else "FN") if lab[i] == 1 else ("FP" if pred[i] == 1 else "TN")
                chosen.append((g, i))
    return chosen


def run_figs(model, items, probs, args, device):
    print("[figs] classification + segmentation explanations")
    cases = select_cases(items, probs, args.per_group, args.seed, args.include)
    res = []
    for g, i in cases:
        t0 = time.time()
        it = items[i]
        img, kid, stone = load_item(it, args.size)
        x = to_tensor(img, device)
        cams, pr = cls_cams(model, x, 1)
        with torch.no_grad():
            out = forward_ex(model, x)
        a_sp = F.interpolate(out["a_sp"], size=(args.size, args.size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        r = {"group": g, "name": it["name"], "img": img, "kid": kid, "stone": stone, "p": float(pr[1]),
             "cams": cams, "a_sp": _norm01(a_sp),
             "m_s": out["m_s"][0, 0].cpu().numpy(), "m_k": out["m_k"][0, 0].cpu().numpy()}
        r["seg_stone"], r["fb_s"] = seg_gradcam(model, x, "stone")
        r["seg_kid"], r["fb_k"] = seg_gradcam(model, x, "kidney")
        if not args.no_slow:
            r["cams"]["Integrated Gradients"] = integrated_gradients(model, x, 1, args.ig_steps)
            r["cams"]["Occlusion"] = occlusion_map(model, x, 1, args.occ_patch, args.occ_stride)
        res.append(r)
        print(f"  {g} {it['name']}  P(stone)={pr[1]:.3f}  ({time.time() - t0:.0f}s)")

    if not res:
        return
    # ---------------- Fig A ----------------
    methods = list(res[0]["cams"].keys())
    cols = ["CT + GT"] + methods + ["Mask-guided\nattention"]
    fig, axes = plt.subplots(len(res), len(cols), figsize=(1.9 * len(cols), 1.9 * len(res)), squeeze=False)
    for r_i, r in enumerate(res):
        show_overlay(axes[r_i, 0], r["img"], None, r["stone"], r["kid"])
        axes[r_i, 0].text(-0.04, 0.5, f"{r['group']}\nP(stone)={r['p']:.2f}", transform=axes[r_i, 0].transAxes,
                          ha="right", va="center", fontsize=8, fontweight="bold")
        for c_i, m in enumerate(methods, start=1):
            show_overlay(axes[r_i, c_i], r["img"], r["cams"][m], r["stone"], None, args.cmap)
        show_overlay(axes[r_i, len(cols) - 1], r["img"], r["a_sp"], r["stone"], None, args.cmap)
    for c_i, c in enumerate(cols):
        axes[0, c_i].set_title(c, fontsize=9)
    plt.subplots_adjust(left=0.08, right=0.92, wspace=0.03, hspace=0.03)
    add_colorbar(fig, args.cmap)
    save_fig(fig, args.out, "Fig_A_classification_explanations")

    # ---------------- Fig B ----------------
    cols = ["CT + GT", "GT stone", "Predicted stone\nprobability", "Seg-Grad-CAM\n(stone decoder)",
            "GT kidney", "Predicted kidney\nprobability", "Seg-Grad-CAM\n(kidney decoder)"]
    fig, axes = plt.subplots(len(res), len(cols), figsize=(1.9 * len(cols), 1.9 * len(res)), squeeze=False)
    for r_i, r in enumerate(res):
        show_overlay(axes[r_i, 0], r["img"], None, r["stone"], r["kid"])
        axes[r_i, 0].text(-0.04, 0.5, r["group"], transform=axes[r_i, 0].transAxes, ha="right", va="center",
                          fontsize=8, fontweight="bold")
        axes[r_i, 1].imshow(r["stone"], cmap="gray"); axes[r_i, 1].axis("off")
        show_overlay(axes[r_i, 2], r["img"], r["m_s"], r["stone"], None, "magma", 0.55)
        show_overlay(axes[r_i, 3], r["img"], r["seg_stone"], r["stone"], None, args.cmap)
        axes[r_i, 4].imshow(r["kid"], cmap="gray"); axes[r_i, 4].axis("off")
        show_overlay(axes[r_i, 5], r["img"], r["m_k"], None, r["kid"], "magma", 0.55)
        show_overlay(axes[r_i, 6], r["img"], r["seg_kid"], None, r["kid"], args.cmap)
        if r["fb_s"]:
            axes[r_i, 3].text(0.02, 0.02, "no stone predicted:\ntop-0.1% pixels explained", transform=axes[r_i, 3].transAxes,
                              color="yellow", fontsize=5, va="bottom")
    for c_i, c in enumerate(cols):
        axes[0, c_i].set_title(c, fontsize=9)
    plt.subplots_adjust(left=0.07, right=0.92, wspace=0.03, hspace=0.03)
    add_colorbar(fig, args.cmap)
    save_fig(fig, args.out, "Fig_B_segmentation_explanations")


def run_metrics(model, items, probs, args, device):
    print("[metrics] localisation + faithfulness")
    rng = np.random.default_rng(args.seed)
    lab = np.array([it["label"] for it in items])
    tp = np.where((lab == 1) & (probs >= 0.5))[0]
    if args.n_eval > 0 and len(tp) > args.n_eval:
        tp = rng.choice(tp, size=args.n_eval, replace=False)
    fracs = np.array(sorted(set([float(f) for f in args.faith_fracs.split(",")] + [0.0, 1.0])))
    rows, curves = [], {}
    for n, i in enumerate(tp):
        it = items[i]
        img, kid, stone = load_item(it, args.size)
        if stone.sum() == 0:
            print(f"  skip {it['name']}: empty GT stone mask (re-annotate!)")
            continue
        t0 = time.time()
        x = to_tensor(img, device)
        cams, _ = cls_cams(model, x, 1)
        cams["Random"] = random_map(rng, args.size, args.size // 16)
        if args.eval_slow:
            cams["Integrated Gradients"] = integrated_gradients(model, x, 1, args.ig_steps)
            cams["Occlusion"] = occlusion_map(model, x, 1, args.occ_patch, args.occ_stride)
        for m, cam in cams.items():
            lm = loc_metrics(cam, stone, kid, args.tol)
            da, ia, pd, pi = deletion_insertion(model, img, cam, fracs, 1, args.batch, device, rng)
            rows.append({"image": it["name"], "method": m, **lm, "deletion_auc": da, "insertion_auc": ia})
            curves.setdefault(m, {"del": [], "ins": []})
            curves[m]["del"].append(pd)
            curves[m]["ins"].append(pi)
        print(f"  [{n + 1}/{len(tp)}] {it['name']} ({time.time() - t0:.0f}s)")

    with open(os.path.join(args.out, "per_image_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = [("energy_in_stone", "Energy in stone"), ("energy_ratio", "Energy ratio (x chance)"),
               ("energy_in_kidney", "Energy in kidney"), ("pointing_strict", "Pointing game"),
               ("pointing_tol", f"Pointing game (+{args.tol}px)"), ("deletion_auc", "Deletion AUC (down)"),
               ("insertion_auc", "Insertion AUC (up)")]
    methods = list(curves.keys())
    summary = []
    for m in methods:
        rr = [r for r in rows if r["method"] == m]
        line = {"Method": m, "N": len(rr)}
        for key, lab_ in metrics:
            mu, sd, lo, hi = mean_ci([r[key] for r in rr])
            line[lab_] = f"{mu:.3f} ± {sd:.3f} [{lo:.3f}, {hi:.3f}]"
        summary.append(line)
    with open(os.path.join(args.out, "table_xai_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, "table_xai_metrics.tex"), "w") as f:
        keys = [lab_ for _, lab_ in metrics]
        f.write("\\begin{tabular}{l" + "c" * len(keys) + "}\n\\toprule\nMethod & " + " & ".join(keys) + " \\\\\n\\midrule\n")
        for s in summary:
            cells = [s[k].split(" [")[0].replace("±", "$\\pm$") for k in keys]
            f.write(s["Method"] + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  saved table_xai_metrics.csv/.tex and per_image_metrics.csv")

    # save curves first so a plotting problem can never lose hours of computation
    np.savez(os.path.join(args.out, "faithfulness_curves.npz"), fracs=fracs,
             **{f"{m.replace(' ', '_')}__{k}": np.stack(v) for m in curves for k, v in curves[m].items()})

    # ---------------- Fig C ----------------
    colors = {"Grad-CAM": "#1f77b4", "Grad-CAM++": "#ff7f0e", "LayerCAM": "#2ca02c",
              "Integrated Gradients": "#9467bd", "Occlusion": "#8c564b", "Random": "#7f7f7f"}
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    for kind, a, ttl in [("del", ax[0, 0], "(a) Deletion"), ("ins", ax[0, 1], "(b) Insertion")]:
        for m in methods:
            arr = np.stack(curves[m][kind])
            mu, sd = arr.mean(0), arr.std(0)
            auc = float(_trapz(mu, fracs))
            a.plot(fracs, mu, color=colors.get(m), lw=1.8, ls="--" if m == "Random" else "-", label=f"{m} (AUC {auc:.3f})")
            a.fill_between(fracs, mu - sd, mu + sd, color=colors.get(m), alpha=0.12)
        a.set_xlabel("Fraction of pixels " + ("removed" if kind == "del" else "inserted"))
        a.set_ylabel("P(stone)")
        a.set_title(ttl + (" (lower is better)" if kind == "del" else " (higher is better)"))
        a.set_xscale("symlog", linthresh=0.05)
        a.legend(fontsize=7, frameon=False)
        a.grid(alpha=0.25)
    data = [[r["energy_ratio"] for r in rows if r["method"] == m] for m in methods]
    tl = [m.replace(" ", "\n") for m in methods]
    try:
        bp = ax[1, 0].boxplot(data, tick_labels=tl, showfliers=False, patch_artist=True)
    except TypeError:  # older matplotlib
        bp = ax[1, 0].boxplot(data, labels=tl, showfliers=False, patch_artist=True)
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(colors.get(m)); patch.set_alpha(0.5)
    ax[1, 0].axhline(1.0, color="k", ls=":", lw=1)
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_ylabel("Energy in GT stone / chance level")
    ax[1, 0].set_title("(c) Localisation (dotted = chance)")
    xs = np.arange(len(methods))
    ps = [np.mean([r["pointing_strict"] for r in rows if r["method"] == m]) for m in methods]
    pt = [np.mean([r["pointing_tol"] for r in rows if r["method"] == m]) for m in methods]
    ax[1, 1].bar(xs - 0.2, ps, 0.4, label="strict", color="#4c72b0")
    ax[1, 1].bar(xs + 0.2, pt, 0.4, label=f"tolerance {args.tol}px", color="#dd8452")
    ax[1, 1].set_xticks(xs); ax[1, 1].set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=8)
    ax[1, 1].set_ylim(0, 1.05); ax[1, 1].set_ylabel("Pointing-game accuracy")
    ax[1, 1].set_title("(d) Peak of the map inside the stone"); ax[1, 1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    save_fig(fig, args.out, "Fig_C_quantitative_evaluation")


def run_sanity(model, items, probs, args, device):
    print("[sanity] cascading weight randomisation")
    rng = np.random.default_rng(args.seed + 1)
    lab = np.array([it["label"] for it in items])
    tp = np.where((lab == 1) & (probs >= 0.5))[0]
    tp = rng.choice(tp, size=min(args.n_sanity, len(tp)), replace=False)
    stages = [("classifier", ["classifier"]), ("SE block", ["se_fc1", "se_fc2"]), ("mask attn", ["g_mask"]),
              ("seg decoders", ["dk4", "dk3", "dk2", "dk1", "out_k", "ds4", "ds3", "ds2", "ds1", "out_s"]),
              ("transformer", ["trans_bot"]), ("CNN bottleneck", ["cnn_bot"]), ("enc e4", ["e4"]),
              ("enc e3", ["e3"]), ("enc e2", ["e2"]), ("enc e1", ["e1"])]
    data = []
    for i in tp:
        img, _, stone = load_item(items[i], args.size)
        data.append((img, stone, to_tensor(img, device)))
    ref = [cls_cams(model, d[2], 1)[0]["Grad-CAM"] for d in data]
    maps = [[r] for r in ref]
    corr = []
    rnd = copy.deepcopy(model).eval()
    torch.manual_seed(args.seed)
    for name, mods in stages:
        for mn in mods:
            for m in getattr(rnd, mn).modules():
                if isinstance(m, nn.MultiheadAttention):
                    m._reset_parameters()
                elif hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        row = []
        for k, d in enumerate(data):
            c = cls_cams(rnd, d[2], 1)[0]["Grad-CAM"]
            maps[k].append(c)
            row.append(spearman(cv2.resize(c, (128, 128)), cv2.resize(ref[k], (128, 128))))
        corr.append(row)
        print(f"  randomised up to '{name}': mean Spearman = {np.mean(row):.3f}")
    corr = np.array(corr)  # stages x images
    labels = ["Original"] + [s[0] for s in stages]
    fig = plt.figure(figsize=(1.6 * len(labels), 5.4))
    gs = fig.add_gridspec(2, len(labels), height_ratios=[1, 1.15], hspace=0.25, wspace=0.03)
    for c in range(len(labels)):
        ax = fig.add_subplot(gs[0, c])
        show_overlay(ax, data[0][0], maps[0][c], data[0][1], None, args.cmap)
        ax.set_title(labels[c], fontsize=7, rotation=0)
    ax = fig.add_subplot(gs[1, :])
    xs = np.arange(1, len(stages) + 1)
    for k in range(corr.shape[1]):
        ax.plot(xs, corr[:, k], color="grey", alpha=0.35, lw=0.8)
    ax.errorbar(xs, corr.mean(1), yerr=corr.std(1), color="crimson", marker="o", capsize=3, lw=1.8,
                label=f"mean ± SD (n={corr.shape[1]})")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in stages], rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ with original Grad-CAM")
    ax.set_xlabel("Cumulative randomisation (top → bottom of network)")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    save_fig(fig, args.out, "Fig_D_sanity_check")
    np.savetxt(os.path.join(args.out, "sanity_check_spearman.csv"), corr, delimiter=",",
               header=",".join(f"img{k}" for k in range(corr.shape[1])), comments="")


def run_tsne(items, probs, feats, args):
    print("[tsne] feature embedding")
    lab = np.array([it["label"] for it in items])
    pred = (probs >= 0.5).astype(int)
    f = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    try:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=min(30, max(5, len(f) // 4)), init="pca", random_state=args.seed).fit_transform(f)
        name = "t-SNE"
    except Exception:
        u, s, vt = np.linalg.svd(f, full_matrices=False)
        emb = u[:, :2] * s[:2]
        name = "PCA"
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for l, c, n in [(0, "#4c72b0", "Non-stone"), (1, "#c44e52", "Stone")]:
        ok = (lab == l) & (pred == lab)
        ax.scatter(emb[ok, 0], emb[ok, 1], s=14, c=c, alpha=0.7, label=f"{n} (correct)")
        bad = (lab == l) & (pred != lab)
        ax.scatter(emb[bad, 0], emb[bad, 1], s=40, facecolors="none", edgecolors=c, linewidths=1.4, label=f"{n} (misclassified)")
    ax.set_xlabel(f"{name} 1"); ax.set_ylabel(f"{name} 2"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("Pooled classification feature (test set)")
    save_fig(fig, args.out, "Fig_E_feature_embedding")


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model_best.pth")
    ap.add_argument("--test_root", default="Last_data/data/test")
    ap.add_argument("--out", default="XAI_visuals")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--run", default="figs,metrics,sanity,tsne")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--per_group", type=int, default=2, help="cases per TP/TN/FN/FP group in Fig A/B")
    ap.add_argument("--include", default="", help="comma-separated image names to force into Fig A/B")
    ap.add_argument("--no_slow", action="store_true", help="skip Integrated Gradients + Occlusion in Fig A")
    ap.add_argument("--eval_slow", action="store_true", help="also evaluate IG + Occlusion quantitatively (slow)")
    ap.add_argument("--n_eval", type=int, default=30, help="TP images for metrics (0 = all)")
    ap.add_argument("--n_sanity", type=int, default=5)
    ap.add_argument("--faith_fracs", default="0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.75")
    ap.add_argument("--tol", type=int, default=16, help="pointing-game tolerance in pixels (1 CAM cell = 16 px)")
    ap.add_argument("--ig_steps", type=int, default=24)
    ap.add_argument("--occ_patch", type=int, default=64)
    ap.add_argument("--occ_stride", type=int, default=48)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    args.include = [s.strip() for s in args.include.split(",") if s.strip()]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    setup_style()
    device = torch.device(args.device)

    model = build_model(args.ckpt, device)
    for p in model.parameters():  # keep requires_grad True (needed so attention layers use the differentiable path)
        p.requires_grad_(True)
    items = discover_test_items(args.test_root)
    print(f"Test set: {len(items)} pairs  (stone={sum(i['label'] for i in items)})")

    # check that the explanation forward == original forward
    img, _, _ = load_item(items[0], args.size)
    x = to_tensor(img, device)
    with torch.no_grad():
        a = model(x)["cls_logits"]
        b = forward_ex(model, x)["cls_logits"]
    assert torch.allclose(a, b, atol=1e-4), "forward_ex deviates from model.forward!"

    cache = os.path.join(args.out, "cache_predictions.npz")
    names = np.array([i["name"] for i in items])
    if os.path.exists(cache) and np.array_equal(np.load(cache, allow_pickle=True)["names"], names):
        c = np.load(cache, allow_pickle=True)
        probs, feats = c["probs"], c["feats"]
        print("Loaded cached predictions.")
    else:
        probs, feats = predict_all(model, items, args.size, device)
        np.savez(cache, names=names, probs=probs, feats=feats)
    lab = np.array([i["label"] for i in items])
    acc = ((probs >= 0.5).astype(int) == lab).mean()
    print(f"Test accuracy (threshold 0.5): {acc:.4f}")

    todo = [s.strip() for s in args.run.split(",")]
    if "figs" in todo:
        run_figs(model, items, probs, args, device)
    if "metrics" in todo:
        run_metrics(model, items, probs, args, device)
    if "sanity" in todo:
        run_sanity(model, items, probs, args, device)
    if "tsne" in todo:
        run_tsne(items, probs, feats, args)
    print("Done ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
