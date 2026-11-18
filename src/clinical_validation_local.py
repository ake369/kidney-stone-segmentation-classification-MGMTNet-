"""
clinical_validation_local.py
Clinical-endpoint validation of the trained model on a local dataset, in PIXELS (no spacing file).

Dataset:  E:\\Original_local_Dataset  with "Stone" and "Non-stone" classes, files named
          P01_STPH_01  ->  patient P01, hospital STPH, slice 01.
          Folder layout is auto-detected (see collect()); masks are optional.

Endpoints
  1) Presence/absence from the predicted stone mask (slice + patient level, patient-cluster bootstrap CIs)
     vs the folder label, next to the classifier head.
  2) Size agreement (needs masks): stone max diameter and kidney long axis in px
     (Pearson, Spearman, ICC(2,1), Lin CCC, Bland-Altman, MAE, relative error).

Threshold T (stone-area px) is taken from the development VALIDATION set (VAL_ROOT) and frozen.

Usage:
  python clinical_validation_local.py
  python clinical_validation_local.py --skip_inference     # re-use saved features
  python clinical_validation_local.py --T 120              # force a frozen threshold
"""
import os, re, sys, json, argparse, warnings
import numpy as np, pandas as pd, cv2
from scipy import stats
from skimage import measure
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================ SETTINGS ============================
DATA_ROOT    = r"E:\Original_local_Dataset"
VAL_ROOT     = r"data/val"                    # development validation set (images/{Stone,Non-Stone}, masks/...)
CHECKPOINT   = r"checkpoints/model_best.pth"
OUT_DIR      = "clinical_validation_local"

MODEL_MODULE = "model"                        # python file (without .py) that DEFINES the model class
MODEL_CLASS  = "MGMTNet"
MODEL_KWARGS = {}                             # constructor args, same as in training

IMG          = 512
SEED         = 42
BOOT_B       = 2000
MIN_STONE_PX = 5                              # ignore stone components smaller than this
MIN_KIDNEY_PX = 200
# =================================================================

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_WORDS = ("mask", "label", "seg", "annot", "ground")


# ------------------------- dataset discovery -----------------------
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def class_of(parts):
    """Stone=1, Non-stone=0, from any folder name in the relative path."""
    for p in parts:
        n = _norm(p)
        if "stone" not in n:
            continue
        prefix = n.replace("stone", "")
        return 0 if re.search(r"(non|no|without|neg)", prefix) else 1
    return None


def parse_name(stem):
    """P01_STPH_01 -> ('P01_STPH', 1). Falls back to the whole stem."""
    m = re.match(r"^(P\d+)_([A-Za-z]+)_(\d+)", stem, re.I)
    if m:
        return f"{m.group(1).upper()}_{m.group(2).upper()}", int(m.group(3))
    return stem, -1


def collect(root):
    """Walk root; pair image/mask files by (class, stem). Works for
         root/{Stone,Non-stone}/{images,masks}/*   root/{images,masks}/{Stone,Non-stone}/*
         root/{Stone,Non-stone}/*  (images only, or masks named *_mask)"""
    recs = {}
    for dp, _, fns in os.walk(root):
        parts = os.path.relpath(dp, root).split(os.sep)
        for fn in fns:
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in IMG_EXT:
                continue
            label = class_of(parts)
            if label is None:
                continue
            is_mask = any(w in p.lower() for p in parts for w in MASK_WORDS) or \
                bool(re.search(r"[_-](mask|seg|label)$", stem, re.I))
            key = re.sub(r"[_-](mask|seg|label)$", "", stem, flags=re.I)
            r = recs.setdefault((label, key), dict(label=label, stem=key))
            r["mask" if is_mask else "image"] = os.path.join(dp, fn)
    out = [r for r in recs.values() if "image" in r]
    return sorted(out, key=lambda r: (parse_name(r["stem"])[0], parse_name(r["stem"])[1], r["stem"]))


def summarize(recs, name):
    n_img = len(recs)
    n_mask = sum("mask" in r for r in recs)
    pats = {parse_name(r["stem"])[0] for r in recs}
    print(f"[{name}] {n_img} images ({sum(r['label'] == 1 for r in recs)} Stone, "
          f"{sum(r['label'] == 0 for r in recs)} Non-stone), {n_mask} with masks, {len(pats)} patients")
    if n_img == 0:
        sys.exit(f"No images found under {name}. Check the path and folder names (Stone / Non-stone).")
    if any(parse_name(r["stem"])[1] < 0 for r in recs):
        warnings.warn("Some file names do not look like P01_STPH_01; their whole name is used as patient ID.")


# ----------------------------- shapes ------------------------------
def shape_features(mask, min_px):
    """total area (px) and max Feret diameter (px) of the largest kept component."""
    lab = measure.label(mask.astype(np.uint8))
    props = [r for r in measure.regionprops(lab) if r.area >= min_px]
    if not props:
        return 0.0, np.nan
    area = float(sum(r.area for r in props))
    big = max(props, key=lambda r: r.area)
    return area, float(big.feret_diameter_max)


# --------------------------- inference -----------------------------
def load_model(ckpt):
    import importlib, torch
    for p in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        mod = importlib.import_module(MODEL_MODULE)
        net = getattr(mod, MODEL_CLASS)(**MODEL_KWARGS)
    except Exception as e:
        sys.exit(f"Could not build {MODEL_CLASS} from '{MODEL_MODULE}.py': {e}\n"
                 f"-> set MODEL_MODULE / MODEL_CLASS / MODEL_KWARGS at the top of this script, and run it from "
                 f"the folder that contains that file.")
    sd = torch.load(ckpt, map_location="cpu")
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    return net.eval()


def split_outputs(out):
    """-> kidney logits (B,1,H,W), stone logits (B,1,H,W), class prob(stone) (float)."""
    import torch
    k = s = c = None
    if isinstance(out, dict):
        for name, v in out.items():
            n = name.lower()
            if "kid" in n: k = v
            elif "stone" in n or n in ("stn", "st"): s = v
            elif "cls" in n or "class" in n: c = v
    else:
        ts = [t for t in (out if isinstance(out, (list, tuple)) else [out]) if torch.is_tensor(t)]
        segs = [t for t in ts if t.dim() == 4]
        cls_ = [t for t in ts if t.dim() == 2]
        if len(segs) == 1 and segs[0].shape[1] >= 2:
            k, s = segs[0][:, :1], segs[0][:, 1:2]
        elif len(segs) >= 2:
            k, s = segs[0], segs[1]
        c = cls_[0] if cls_ else None
    if k is None or s is None or c is None:
        sys.exit("Could not interpret the model output. Edit split_outputs() to return "
                 "(kidney_logits, stone_logits, class_logits).")
    prob = torch.sigmoid(c).flatten()[0] if c.shape[-1] == 1 else torch.softmax(c, 1)[0, 1]
    return k, s, float(prob)


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(path)
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(path)
    if m.ndim == 3:
        m = m[..., 0]
    m = cv2.resize(m, (IMG, IMG), interpolation=cv2.INTER_NEAREST)
    return m > 0, (m == 2) | (m > 128)                       # same rules as dataset.py


def run_items(net, recs, tag):
    import torch
    rows, seen_vals = [], set()
    for i, r in enumerate(recs, 1):
        img = read_image(r["image"])
        x = torch.from_numpy(img)[None, None].repeat(1, 3, 1, 1)
        with torch.no_grad():
            k, s, prob = split_outputs(net(x))
        pk = torch.sigmoid(k)[0, 0].numpy() > 0.5
        ps = torch.sigmoid(s)[0, 0].numpy() > 0.5
        pa, pdm = shape_features(ps, MIN_STONE_PX)
        pka, pkl = shape_features(pk, MIN_KIDNEY_PX)
        ra = rd = rka = rkl = np.nan
        if "mask" in r:
            gk, gs = read_mask(r["mask"])
            ra, rd = shape_features(gs, MIN_STONE_PX)
            rka, rkl = shape_features(gk, MIN_KIDNEY_PX)
            if len(seen_vals) < 12:
                seen_vals.update(np.unique(cv2.imread(r["mask"], cv2.IMREAD_UNCHANGED)).tolist())
        pid, sl = parse_name(r["stem"])
        rows.append(dict(split=tag, patient=pid, slice=sl, file=os.path.basename(r["image"]), label=r["label"],
                         cls_prob=prob, ref_stone_area=ra, ref_stone_diam=rd, pred_stone_area=pa,
                         pred_stone_diam=pdm, ref_kid_area=rka, ref_kid_len=rkl, pred_kid_area=pka,
                         pred_kid_len=pkl))
        if i % 25 == 0 or i == len(recs):
            print(f"  {tag}: {i}/{len(recs)}", flush=True)
    if seen_vals:
        print(f"  [{tag}] mask pixel values seen (first masks): {sorted(seen_vals)[:12]}  "
              f"(expected 0/1/2 or 0/255; stone = 2 or >128)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"features_{tag}.csv"), index=False)
    return df


# ------------------------- statistics helpers ----------------------
def cluster_boot(df, fn, B=BOOT_B, seed=SEED):
    """Resample PATIENTS with replacement; fn(df)->dict of scalars. Returns {key:(lo,hi)}."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("patient")}
    pats = np.array(list(groups))
    res = []
    for _ in range(B):
        d = pd.concat([groups[p] for p in rng.choice(pats, len(pats), replace=True)], ignore_index=True)
        try:
            res.append(fn(d))
        except Exception:
            pass
    if not res:
        return {}
    r = pd.DataFrame(res)
    return {c: tuple(np.nanpercentile(r[c].astype(float), [2.5, 97.5])) for c in r.columns}


def presence_metrics(d, thr, score="score"):
    y = d["ref_pos"].values.astype(bool)
    s = d[score].values.astype(float)
    p = s >= thr
    tp, tn = (y & p).sum(), (~y & ~p).sum()
    fp, fn_ = (~y & p).sum(), (y & ~p).sum()
    div = lambda a, b: a / b if b else np.nan
    try:
        auc = roc_auc_score(y, s)
    except Exception:
        auc = np.nan
    return dict(sensitivity=div(tp, tp + fn_), specificity=div(tn, tn + fp), PPV=div(tp, tp + fp),
                NPV=div(tn, tn + fn_), accuracy=div(tp + tn, len(y)), AUC=auc)


def icc21(a, b):
    X = np.c_[a, b]
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def ccc(a, b):
    return 2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)


def fisher_ci(r, n):
    if n < 4:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))


def agreement(ref, pred):
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    n = len(ref)
    diff = pred - ref
    bias, sd = diff.mean(), diff.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    se_b, se_l = sd / np.sqrt(n), sd * np.sqrt(3 / n)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    r, rho = stats.pearsonr(ref, pred)[0], stats.spearmanr(ref, pred)[0]
    boot = cluster_boot(pd.DataFrame(dict(patient=np.arange(n), a=ref, b=pred)),
                        lambda d: dict(icc=icc21(d.a.values, d.b.values), ccc=ccc(d.a.values, d.b.values)))
    nan2 = (np.nan, np.nan)
    rel = np.abs(diff) / ref
    return {"n (patients)": n, "unit": "px",
            "Pearson r": r, "Pearson 95% CI": fisher_ci(r, n),
            "Spearman rho": rho, "Spearman 95% CI": fisher_ci(rho, n),
            "ICC(2,1)": icc21(ref, pred), "ICC 95% CI": boot.get("icc", nan2),
            "Lin CCC": ccc(ref, pred), "CCC 95% CI": boot.get("ccc", nan2),
            "Bias (model-ref)": bias, "Bias 95% CI": (bias - t * se_b, bias + t * se_b),
            "LoA lower": lo, "LoA lower 95% CI": (lo - t * se_l, lo + t * se_l),
            "LoA upper": hi, "LoA upper 95% CI": (hi - t * se_l, hi + t * se_l),
            "MAE": np.abs(diff).mean(), "Median abs error": np.median(np.abs(diff)),
            "Mean relative error %": rel.mean() * 100,
            "Within 10%": (rel <= 0.10).mean(), "Within 20%": (rel <= 0.20).mean()}


# ------------------------------ plots ------------------------------
def _save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_agreement(ref, pred, title, stem):
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    mean, diff = (ref + pred) / 2, pred - ref
    bias, sd = diff.mean(), diff.std(ddof=1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    lim = [0, max(ref.max(), pred.max()) * 1.05]
    ax[0].scatter(ref, pred, s=22, alpha=.8, edgecolor="k", linewidth=.4)
    ax[0].plot(lim, lim, "k--", lw=1, label="identity")
    ax[0].set(xlim=lim, ylim=lim, xlabel="Reference (px)", ylabel="Model (px)", title="Scatter")
    ax[0].legend(frameon=False)
    ax[1].scatter(mean, diff, s=22, alpha=.8, edgecolor="k", linewidth=.4)
    for v, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
        ax[1].axhline(v, color="r", ls=ls, lw=1)
        ax[1].text(ax[1].get_xlim()[1], v, f" {v:+.1f}", va="center", fontsize=8, color="r")
    ax[1].set(xlabel="Mean of model and reference (px)", ylabel="Model - reference (px)", title="Bland-Altman")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, stem)


def plot_roc(d, stem):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for col, name in (("score", "Mask-derived (stone area)"), ("cls_prob", "Classifier head")):
        if d[col].nunique() > 1 and d["ref_pos"].nunique() > 1:
            fpr, tpr, _ = roc_curve(d["ref_pos"], d[col])
            ax.plot(fpr, tpr, label=f"{name}, AUC={roc_auc_score(d['ref_pos'], d[col]):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title="Slice-level ROC")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, stem)


def fmt(v):
    if isinstance(v, tuple):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)


# ------------------------------ main -------------------------------
def youden_threshold(df):
    if df["label"].nunique() < 2:
        return None
    fpr, tpr, thr = roc_curve(df["label"] == 1, df["pred_stone_area"])
    return max(float(thr[np.argmax(tpr - fpr)]), MIN_STONE_PX)


def main(a):
    os.makedirs(OUT_DIR, exist_ok=True)
    feat_path = os.path.join(OUT_DIR, "features_local.csv")
    thr_path = os.path.join(OUT_DIR, "threshold.json")

    if a.skip_inference:
        df = pd.read_csv(feat_path)
        T = json.load(open(thr_path))["T"]
    else:
        recs = collect(a.root)
        summarize(recs, "local dataset")
        net = load_model(a.ckpt)
        T, T_src = a.T, "given on the command line"
        if T is None and os.path.isdir(a.val_root):
            vrecs = collect(a.val_root)
            summarize(vrecs, "validation set")
            T = youden_threshold(run_items(net, vrecs, "val"))
            T_src = f"Youden's J on the validation set ({a.val_root})"
        df = run_items(net, recs, "local")
        if T is None:
            T = youden_threshold(df)
            T_src = "Youden's J on THIS dataset (optimistic! supply --val_root or --T for a frozen threshold)"
            warnings.warn("No validation set found; threshold was tuned on the evaluation data.")
        json.dump({"T": T, "source": T_src}, open(thr_path, "w"))
        print(f"Frozen stone-area threshold T = {T:.0f} px  [{T_src}]")

    df["ref_pos"] = df["label"] == 1
    df["score"] = df["pred_stone_area"]
    has_masks = df["ref_stone_area"].notna().any()
    print(f"\n{len(df)} slices, {df['patient'].nunique()} patients, masks available: {has_masks}")
    if has_masks:
        st = df[df.ref_pos & df.ref_stone_area.notna()]
        n_ok = int((st.ref_stone_area > 0).sum())
        print(f"Stone-folder slices whose mask contains stone pixels: {n_ok}/{len(st)}")
        if st.shape[0] and n_ok == 0:
            warnings.warn("No stone pixels found in any Stone mask - check the mask encoding "
                          "(expects stone = 2 or >128).")

    # ---- 1) presence / absence ----
    rows = []

    def add(level, method, d, thr_, score):
        m = presence_metrics(d, thr_, score)
        ci = cluster_boot(d, lambda x: presence_metrics(x, thr_, score))
        for k, v in m.items():
            lo, hi = ci.get(k, (np.nan, np.nan))
            rows.append(dict(level=level, method=method, metric=k, value=v, ci_lo=lo, ci_hi=hi))

    pat = df.groupby("patient").agg(ref_pos=("ref_pos", "any"), score=("score", "max"),
                                    cls_prob=("cls_prob", "max")).reset_index()
    add("slice", "mask-derived", df, T, "score")
    add("slice", "classifier head", df, 0.5, "cls_prob")
    add("patient", "mask-derived", pat, T, "score")
    add("patient", "classifier head", pat, 0.5, "cls_prob")
    pres = pd.DataFrame(rows)
    pres.to_csv(os.path.join(OUT_DIR, "table_presence.csv"), index=False)
    print("\nPresence/absence (point estimates; CIs in table_presence.csv)")
    print(pres.pivot_table(index=["level", "method"], columns="metric", values="value").round(3).to_string())
    plot_roc(df, "fig_roc_slice")

    err = df[(df.score >= T) != df.ref_pos][["patient", "file", "label", "pred_stone_area", "cls_prob"]]
    err.to_csv(os.path.join(OUT_DIR, "misclassified_slices_mask_derived.csv"), index=False)

    # ---- 2) size agreement (one slice per patient, both positive) ----
    if has_masks:
        ag = {}
        s = df[(df.ref_stone_diam > 0) & (df.pred_stone_area >= T)].dropna(subset=["pred_stone_diam"])
        if s["patient"].nunique() >= 5:
            s = s.loc[s.groupby("patient")["ref_stone_diam"].idxmax()]
            ag["Stone max diameter"] = agreement(s.ref_stone_diam, s.pred_stone_diam)
            plot_agreement(s.ref_stone_diam, s.pred_stone_diam, "Stone maximum diameter", "fig_agreement_stone")
        else:
            print("Fewer than 5 patients with a detected stone: stone agreement skipped.")
        k = df[(df.ref_kid_len > 0) & (df.pred_kid_len > 0)]
        if k["patient"].nunique() >= 5:
            k = k.loc[k.groupby("patient")["ref_kid_area"].idxmax()]
            ag["Kidney long axis"] = agreement(k.ref_kid_len, k.pred_kid_len)
            plot_agreement(k.ref_kid_len, k.pred_kid_len, "Kidney long-axis length", "fig_agreement_kidney")
        if ag:
            tab = pd.DataFrame({q: {x: fmt(y) for x, y in v.items()} for q, v in ag.items()})
            tab.to_csv(os.path.join(OUT_DIR, "table_agreement.csv"))
            print("\nSize agreement (pixels)")
            print(tab.to_string())
    else:
        print("\nNo masks found -> size agreement skipped (presence/absence only).")
    print(f"\nDone. Everything is in ./{OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--val_root", default=VAL_ROOT)
    ap.add_argument("--ckpt", default=CHECKPOINT)
    ap.add_argument("--T", type=float, default=None, help="frozen stone-area threshold in px")
    ap.add_argument("--skip_inference", action="store_true")
    main(ap.parse_args())
