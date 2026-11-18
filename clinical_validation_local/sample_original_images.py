"""
Randomly pick one CT image from each of four different patients and
arrange them in a row for a manuscript figure (PNG, 600 dpi).

Expects the dataset's original-image naming:
    PatientID_Hospital_Gender_StoneStatus_ImageNumber   e.g. P189_RA_F_S_I10.jpg
(augmented files have an extra suffix, e.g. P189_RA_F_S_I10_EL.jpg, and are skipped)

Usage:
    python make_ct_figure.py
Requires: matplotlib, pillow
"""
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------- CONFIG ---------------------------------
DATASET_DIR = r"E:\Original_local_Dataset"   # raw string (r"...") avoids the backslash warning; folder with the original images (searched recursively)
N_IMAGES = 4
BALANCE_CLASSES = True            # True: 2 stone + 2 non-stone patients; False: fully random
SEED = 42                         # fix for a reproducible pick; set None for a new pick each run

OUTPUT_FILE = "ct_examples.png"
DPI = 600
FIG_WIDTH_IN = 7.2                # ~double-column width (inches)
CROP_BOX = None                   # (left, upper, right, lower) pixels, to remove viewer borders/text
GRAYSCALE = True
FONT_SIZE = 10
SHOW_CLASS_LABEL = True           # caption "(a) Stone" instead of just "(a)"
# ----------------------------------------------------------------------

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
STATUS_NAMES = {"S": "Stone", "NS": "Non-stone"}
AUG_CODES = {"CR", "DO", "EL", "FH", "FV", "GB", "GN", "LC", "MUL", "PE", "PA", "ROT"}


def parse_name(path):
    """Return (patient_id, status) for an original image, else None.

    Tolerant of naming variations: finds the patient token (P + digits) and the
    status token (S / NS) anywhere in the file name, and falls back to the
    parent folder name (e.g. 'stone' / 'non-stone') for the status.
    """
    parts = path.stem.upper().split("_")
    if parts[-1] in AUG_CODES:               # augmented copy -> skip
        return None

    patient_id = next((p for p in parts if p.startswith("P") and p[1:].isdigit()), None)
    if patient_id is None:
        return None

    status = next((p for p in parts if p in STATUS_NAMES), None)
    if status is None:
        folder = path.parent.name.lower()
        if "non" in folder or "normal" in folder:
            status = "NS"
        elif "stone" in folder:
            status = "S"
    if status is None:
        return None
    return patient_id, status


def collect_by_patient(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DATASET_DIR does not exist: {root}")

    by_patient = defaultdict(list)   # patient_id -> [(path, status), ...]
    all_images = [p for p in root.rglob("*") if p.suffix.lower() in EXTS]
    print(f"Found {len(all_images)} image files under {root}")

    for p in all_images:
        parsed = parse_name(p)
        if parsed:
            by_patient[parsed[0]].append((p, parsed[1]))

    if not by_patient:
        print("None matched the expected naming. First few file names found:")
        for p in all_images[:8]:
            print("  ", p.relative_to(root))
        print("Send me a few of these names and I will adapt parse_name().")
    return by_patient


def pick_images(by_patient, n, balance, rng):
    """One random image from each of n different patients."""
    if len(by_patient) < n:
        raise ValueError(f"Only {len(by_patient)} patients found, need {n}.")

    if balance:
        # Group patients by the status of their images (use the first image's status)
        groups = {"S": [], "NS": []}
        for pid, items in by_patient.items():
            groups[items[0][1]].append(pid)
        half = n // 2
        if len(groups["S"]) < half or len(groups["NS"]) < n - half:
            raise ValueError("Not enough patients in each class for balancing.")
        chosen = rng.sample(groups["S"], half) + rng.sample(groups["NS"], n - half)
        rng.shuffle(chosen)
    else:
        chosen = rng.sample(list(by_patient), n)

    return [(pid, *rng.choice(by_patient[pid])) for pid in chosen]  # (pid, path, status)


def load_image(path):
    img = Image.open(path)
    if CROP_BOX:
        img = img.crop(CROP_BOX)
    return img.convert("L") if GRAYSCALE else img.convert("RGB")


def main():
    rng = random.Random(SEED)
    by_patient = collect_by_patient(DATASET_DIR)
    picks = pick_images(by_patient, N_IMAGES, BALANCE_CLASSES, rng)

    print("Selected images (keep this for your records):")
    for i, (pid, path, status) in enumerate(picks):
        print(f"  ({chr(ord('a') + i)}) {pid}  {status:>2}  {path.name}")

    images = [load_image(p) for _, p, _ in picks]
    w, h = images[0].size
    fig_height = FIG_WIDTH_IN / N_IMAGES * (h / w) + 0.35

    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, N_IMAGES, figsize=(FIG_WIDTH_IN, fig_height))

    for i, (ax, img, (_, _, status)) in enumerate(zip(axes, images, picks)):
        ax.imshow(img, cmap="gray" if GRAYSCALE else None)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        caption = f"({chr(ord('a') + i)})"
        if SHOW_CLASS_LABEL:
            caption += f" {STATUS_NAMES[status]}"
        ax.set_xlabel(caption, fontsize=FONT_SIZE, labelpad=4)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12, wspace=0.04)
    fig.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight")
    print(f"Saved {OUTPUT_FILE} at {DPI} dpi")


if __name__ == "__main__":
    main()