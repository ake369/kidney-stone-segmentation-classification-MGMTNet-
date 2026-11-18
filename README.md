# MGMTNet: Segmentation-Guided Multi-Task Kidney Stone Analysis on CT

Official code for the manuscript **"A Multi-Task Hybrid CNN–Transformer Framework for Segmentation-Guided Kidney Disease Classification from CT Images"** (under review).

MGMTNet jointly segments the kidney, segments the stone, and classifies each axial CT slice as stone or non-stone. The predicted kidney and stone masks condition the classifier through mask-guided attention, so the diagnosis is tied to the network's own estimate of the anatomy.

> **Data are not distributed.** This repository contains code, configuration, split lists and a trained checkpoint. It contains no CT images, no masks and no data from the independent local cohort (see [Data availability](#data-availability)).

## Highlights

- **One network, three outputs:** kidney mask, stone mask and slice-level stone/non-stone prediction (23.52 M parameters).
- **Segmentation-guided classification:** the predicted masks are resized to the bottleneck resolution and gate the bottleneck features before classification.
- **Compact hybrid backbone:** a CNN encoder trained from scratch with a 4-layer Transformer bottleneck fused residually.
- **Quantitative explainability:** Grad-CAM, Grad-CAM++ and LayerCAM evaluated for localisation and faithfulness against a random-map baseline.
- **Independent local validation:** evaluation on a separate single-hospital cohort without retraining.

## Model

| Component | Description |
|---|---|
| Input | 3 × 512 × 512 (grayscale slice scaled to [0, 1] and replicated over three channels) |
| Encoder | Four double-convolution stages (32, 64, 128, 256 channels) plus a 512-channel CNN bottleneck at 1/16 resolution |
| Transformer bottleneck | 4 layers, 8 heads, 512-d tokens, feed-forward dimension 2048 (post-norm, ReLU), fused with the CNN bottleneck by residual addition |
| Decoders | Two structurally identical, independently parameterised decoders (kidney, stone) with four skip connections |
| Mask-guided attention | Predicted kidney and stone probability maps are resized to the bottleneck, passed through a 3×3 → 1×1 convolutional gate, and applied as a residual spatial gate, followed by squeeze-and-excitation channel attention (reduction ratio 16) |
| Classification head | Global average pooling, then a two-layer fully connected head with batch normalisation and dropout 0.3 |
| Outputs | `kidney_logits`, `stone_logits`, `cls_logits` (plus `m_k`, `m_s`) |

The model definition is in `src/model.py`.

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── configs/default.yaml        # hyper-parameters, seeds, paths
├── src/
│   ├── model.py                # MGMTNet
│   ├── dataset.py              # data loading and preprocessing
│   ├── train.py                # joint training (loss weights, schedule)
│   ├── evaluate.py             # segmentation/classification metrics, bootstrap CIs
│   └── explain.py              # Grad-CAM family and Seg-Grad-CAM figures
├── splits/splits.csv           # patient-level split (public-dataset filenames only)
├── tests/test_model.py         # smoke tests that run without any data
└── results/expected_results.json
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/          # checks the model builds with 23.52 M parameters
```

All experiments in the paper were run on CPU only (no GPU).

## Data

**Not distributed.** To reproduce the setup, obtain the public axial CT kidney stone dataset from its original source (Abdalla et al., *Data in Brief*, 2025; see the manuscript for the full reference and check its licence). Our pixel-level annotations and the independent local cohort are not released.

Expected layout:

```
<data-root>/
├── train/ {images,masks}/{Stone,Non-Stone}/
├── val/   {images,masks}/{Stone,Non-Stone}/
└── test/  {images,masks}/{Stone,Non-Stone}/
```

- Images and masks share the same file name stem.
- Mask labels: `0` background, `1` kidney, `2` stone. The kidney target is the union of labels 1 and 2 (it includes the stone); the stone target is label 2.
- Splits are made at the **patient level** (70/15/15), so no patient contributes slices to more than one subset. The exact file lists are in `splits/splits.csv`.
- Augmentation is applied to the training subset only (flip, small affine, brightness/contrast, Gaussian noise, Gaussian blur).

## Training

```bash
python src/train.py --config configs/default.yaml --data-root <data-root> --seed 0
```

| Setting | Value |
|---|---|
| Optimiser | AdamW (learning rate 1e-3, weight decay 1e-4) |
| Schedule | Linear warm-up followed by cosine decay, 50 epochs |
| Batch size | 16 |
| Loss | BCE + soft Dice for each decoder, cross-entropy for classification, task weights (α, β, γ) = (1.0, 1.5, 0.8) |
| Model selection | Highest validation classification accuracy |

Exact values are in `configs/default.yaml`.

## Evaluation

```bash
python src/evaluate.py --ckpt checkpoints/mgmtnet_best.pth --data-root <data-root> --split test
```

- Kidney Dice and IoU are computed on all test slices; stone Dice and IoU are computed on stone-positive slices only.
- Classification metrics (accuracy, sensitivity, specificity, PPV, NPV, F1, AUC) use a threshold of 0.5.
- Confidence intervals for the independent cohort use a patient-level cluster bootstrap (2,000 resamples).

### Independent local validation

The released code reproduces the protocol, but not the data. The checkpoint is applied without retraining or recalibration. Two decision rules are evaluated from the same forward pass:

1. **Mask-derived rule:** a slice is stone-positive when the predicted stone area is at least *T* = 15 pixels at 512 × 512 resolution (fixed on the development validation set by maximising Youden's *J*).
2. **Classifier rule:** a slice is stone-positive when the stone-class probability is at least 0.5.

## Explainability figures

```bash
# list predictions and errors per split
python src/explain.py --scan-only --ckpt checkpoints/mgmtnet_best.pth --data-root <data-root>

# generate the classification and segmentation explanation figures
python src/explain.py --ckpt checkpoints/mgmtnet_best.pth --data-root <data-root> --anonymise
```

The figures show Grad-CAM, Grad-CAM++ and LayerCAM maps for the stone class (computed on the last feature map before pooling), the mask-guided attention map, and Seg-Grad-CAM for each decoder. The `--anonymise` flag blanks the image margins that may carry burned-in text in the **displayed** images only; the model always receives the original image.

## Pretrained checkpoint

The trained checkpoint is available on the [Releases](../../releases) page (tag `v1.0`).

- File: `mgmtnet_best.pth`
- SHA-256: `<hash>`
- Selected by highest validation classification accuracy (epoch `<n>`).
- Trained on the public-derived training split only; no local-cohort data were used for training, model selection or threshold selection.

## Expected results

Values reported in the manuscript (patient-level held-out test set, 168 slices: 83 stone, 85 non-stone):

| Metric | Value |
|---|---|
| Kidney Dice | 84.2% |
| Stone Dice (stone-positive slices) | 81.8% |
| Classification accuracy | 100% (95% CI 97.8–100) |
| AUC | 1.000 |
| Parameters | 23.52 M |

Independent local cohort (382 slices, 32 patients):

| Rule | Sensitivity | Specificity | AUC |
|---|---|---|---|
| Mask-derived (*T* = 15 px) | 99.0% | 97.8% | 0.994 |
| Classifier (p ≥ 0.5) | 96.0% | 97.3% | 0.995 |

These numbers come from one training run and one split; interpret small differences between methods with caution. Full tables, baselines, ablations and confidence intervals are in the paper. `results/expected_results.json` lists the values used in the paper's tables.

## Reproducibility notes

- Seeds are set for `random`, `numpy` and `torch` (default seed 0).
- Training and inference were run on CPU; inference takes about 33 ms per slice at batch size 1.
- Because the images and masks are not distributed, the tables cannot be regenerated from this repository alone. The architecture, the loss and training configuration, the split lists, the checkpoint and the evaluation code are provided so that the setup can be reproduced on the public images with your own annotations.

## Limitations

The model processes 2D axial slices; it was trained on a single public source and validated on one independent single-hospital cohort. It is a research prototype and not a medical device.

## Data availability

The source images are publicly available from the original dataset. The pixel-level annotations and the independent local cohort are not publicly available because of [patient privacy / institutional policy].

## Ethics

The independent local cohort was analysed retrospectively with institutional ethics approval [details withheld for blind review]. All images were anonymised before analysis.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{mgmtnet2026,
  title   = {A Multi-Task Hybrid CNN--Transformer Framework for Segmentation-Guided Kidney Disease Classification from CT Images},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The licences of the source images and of any annotations are separate and are governed by their original terms.

## Contact
Anonymised during review.