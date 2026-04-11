# CAST at SignEval 2026

**Channel-Aware Spatial Transfer Learning with Pseudo-Image Radar for Sign Language Recognition**

> MSLR Workshop @ CVPR 2026 · Denver, Colorado, USA · June 3–4, 2026

---

## Overview

**CAST** is a dual-stream architecture for radar-only sign language recognition that uses **no RGB or depth data** — only raw 60 GHz radar Range-Time Maps (RTM).

The three design principles encoded in the acronym:

| Letter | Component | What it does |
|---|---|---|
| **C**hannel-**A**ware | Cross-Antenna Self-Attention (CASA) | Learns inter-antenna importance weights before the backbone |
| **S**patial **T**ransfer | ImageNet-pretrained CNNs as backbones | Transfers spatial feature detectors to radar pseudo-images |
| Pseudo-Image Radar | CVD computation | Converts RTM slow-time axis to a Cadence-Velocity Diagram image |

### Architecture

```
RTM₁ RTM₂ RTM₃  ──►  CASA  ──►  EfficientNetV2-S  ──►  proj ──►  Q ─┐
                                                                       ├─►  Cross-Attn  ──►  gate  ──►  head  ──►  logits
CVD₁ CVD₂ CVD₃  ──►  CASA  ──►  ConvNeXt-Tiny      ──►  proj ──► K,V─┘
(pseudo-image)
```

- **RTM stream** — three RX-antenna Range-Time Maps stacked as a 3-channel image → EfficientNetV2-S
- **CVD stream** — per-antenna Cadence-Velocity Diagrams (Doppler spectra via rFFT) rendered as a 3-channel pseudo-image → ConvNeXt-Tiny
- **Fusion** — asymmetric cross-attention: RTM as query, CVD as key/value, plus a learned gated residual

---

## Results

| Model | CV Top-1 Acc | Setting |
|---|---|---|
| Single-backbone baseline (5-fold) | 77.2 ± 1.3% | EfficientNetV2-S, RTM only |
| CAST (5-fold CV) | **80.5 ± 0.9%** | Full model, CASA 4-head |
| CAST (single 90/10 run) | 81.73% | Kaggle leaderboard |
| EfficientNetV2-S + ConvNeXt-Tiny Ensemble (10-fold) | 84.88% | Kaggle leaderboard |

> **Note:** Kaggle leaderboard scores reflect different compute budgets (single run vs. 10-fold ensemble) and are not directly comparable.

### Ablation (5-fold CV)

| Variant | Top-1 Acc | Delta |
|---|---|---|
| RTM only — no CVD stream | 79.6% | baseline |
| + CVD (no dB→linear step) | 80.3% | +0.7 pp |
| + CVD (full pseudo-image pipeline) | **80.5%** | **+0.9 pp** |
| No CASA | 79.8% | −0.7 pp |
| CASA 1-head | 80.2% | −0.3 pp |
| CASA 4-heads | **80.5%** | — |

---

## Repository Structure

```
CAST/
├── README.md
├── LICENSE                         MIT
├── requirements.txt
├── .gitignore
│
├── src/                            Importable Python package
│   ├── models/
│   │   ├── casa.py                 Cross-Antenna Self-Attention (CASA)
│   │   ├── fusion.py               Asymmetric Cross-Attention Fusion
│   │   └── cast_model.py           CASTModel — full dual-stream model
│   ├── data/
│   │   ├── cvd.py                  Pseudo-CVD computation (dB→BH window→rFFT)
│   │   ├── augmentations.py        Physics-aware radar augmentations
│   │   └── dataset.py              DualStreamRTMDataset + TestDualStreamDataset
│   └── utils/
│       └── training.py             EMA, MixUp, CutMix, cosine LR, validate
│
├── configs/
│   └── cast.yaml                   All hyperparameters
│
└── notebooks/                      Kaggle-runnable, self-contained
    ├── 01_CAST_training_val0.8173.ipynb              Paper model — CAST (81.73%)
    ├── 02_ScoreMaximizer_V2_val0.8390.ipynb          V2 mega-ensemble (83.90%)
    ├── 03_BestModel_KFold_Ensemble_val0.8488.ipynb   Best result (84.88%)
    └── 04_Inference_TestSet_Combine.ipynb            Inference + submission merge
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tested with: Python 3.10 · PyTorch 2.2 · CUDA 12.1 · timm 0.9.16

### Dataset

MultiMeDaLIS is available through the Kaggle competition:

> [CVPR MSLR 2026 Track 2](https://www.kaggle.com/competitions/cvpr-mslr-2026-track-2)

Expected layout:

```
cvpr-mslr-2026-track-2/
├── train/
│   ├── 0_CLASSNAME/
│   │   └── SAMPLE_{id}/
│   │       ├── SAMPLE{id}_RTM1.npy   # Antenna 1, shape (T, R), float32, dB
│   │       ├── SAMPLE{id}_RTM2.npy
│   │       └── SAMPLE{id}_RTM3.npy
│   └── ... (126 classes total)
├── val/
│   └── SAMPLE_{id}/
└── test/
    └── SAMPLE_{id}/
```

---

## Reproducing Results

### Option A — Kaggle Notebooks (recommended)

Upload a notebook from `notebooks/` to a Kaggle P100/T4 session with the competition dataset attached and run cells top-to-bottom.

| Notebook | What it trains | Expected score |
|---|---|---|
| `01_CAST_training_val0.8173.ipynb` | CAST (paper model) | ~81.7% |
| `02_ScoreMaximizer_V2_val0.8390.ipynb` | V2 diverse-backbone ensemble | ~83.9% |
| `03_BestModel_KFold_Ensemble_val0.8488.ipynb` | 10-fold ensemble (best result) | ~84.9% |
| `04_Inference_TestSet_Combine.ipynb` | Merge val + test predictions | — |

### Option B — Python Package

```python
import torch
from src.models import CASTModel

model = CASTModel(
    num_classes=126,
    feat_dim=512,
    rtm_name="tf_efficientnetv2_s.in21k_ft_in1k",
    cvd_name="convnext_tiny.fb_in22k_ft_in1k",
    pretrained=True,
)

rtm_img = torch.randn(1, 3, 224, 224)   # 3-antenna RTM pseudo-image
cvd_img = torch.randn(1, 3, 224, 224)   # 3-antenna CVD pseudo-image

logits = model(rtm_img, cvd_img)         # (1, 126)

# With auxiliary heads during training
logits, logits_rtm, logits_cvd = model(rtm_img, cvd_img, return_aux=True)
```

### CVD Pseudo-Image Computation

```python
import numpy as np
from src.data import compute_cvd

rtm_db = np.load("SAMPLE_42/SAMPLE42_RTM1.npy").astype(np.float32)  # (T, R), dB
cvd_db = compute_cvd(rtm_db, n_fft=128)                              # (R, 64)
```

### Dataset Loading

```python
from torch.utils.data import DataLoader
from src.data import DualStreamRTMDataset

# sample_list: [(sample_dir_path, label_int), ...]
train_ds = DualStreamRTMDataset(sample_list, img_size=224, max_T=48, augment=True)
loader   = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4)

for rtm_imgs, cvd_imgs, labels in loader:
    # rtm_imgs: (B, 3, 224, 224)
    # cvd_imgs: (B, 3, 224, 224)
    # labels:   (B,)
    pass
```

---

## Method Details

### Pseudo-Image Radar (CVD computation)

```
RTM (dB) → dB→linear → Blackman-Harris window (slow-time)
         → zero-pad to n_fft → rFFT → |·| → drop DC → 20·log10 → CVD (dB)
```

Each CVD image has shape `(R, n_fft/2)` — range bins × Doppler bins — mirroring the RTM spatial layout. Applied identically to all three antenna channels, then stacked into a 3-channel pseudo-image for the CVD backbone.

### Channel-Aware: CASA (~1.5 K parameters)

A lightweight pre-backbone module applied to both streams independently. For each RX-antenna channel, a shared convolutional stem extracts a 16-d spatial embedding. Multi-head self-attention across the three antenna tokens produces per-channel scalar gates in (0, 1), which re-weight the input channels before the backbone.

### Training Recipe

| Component | Setting |
|---|---|
| Optimiser | AdamW — lr 3×10⁻⁴, weight decay 0.05 |
| Schedule | 5-epoch linear warmup + cosine annealing (70 epochs total) |
| Augmentation | MixUp (α=0.4) + CutMix (α=1.0) + physics augmentations |
| Physics aug | Time warp · magnitude warp · simulated multipath · antenna dropout |
| Regularisation | Label smoothing 0.1, Dropout 0.3, DropPath 0.2 |
| EMA | decay=0.9995 |
| SWA | starts at epoch 56/70, lr=1×10⁻⁵ |
| Checkpointing | Top-5 validation checkpoints ensembled at inference |
| TTA | 5-view test-time augmentation |
| Auxiliary loss | L = L_main + 0.3 × (L_rtm + L_cvd) |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{cast2026,
  title     = {{CAST} at {SignEval} 2026: Channel-Aware Spatial Transfer Learning
               with Pseudo-Image Radar for Sign Language Recognition},
  author    = {[Authors]},
  booktitle = {MSLR Workshop @ CVPR 2026},
  year      = {2026},
}
```

Please also cite the MultiMeDaLIS dataset:

```bibtex
@inproceedings{mineo2024sign,
  title     = {Sign Language Recognition for Patient-Doctor Communication:
               A Multimedia/Multimodal Dataset},
  author    = {Mineo, Raffaele and Caligiore, Gaia and Spampinato, Concetto
               and Fontana, Sabina and Palazzo, Simone and Ragonese, Egidio},
  booktitle = {IEEE 8th Forum on Research and Technologies for Society
               and Industry Innovation (RTSI)},
  pages     = {202--207},
  year      = {2024},
}

@inproceedings{mineo2025radar,
  title     = {Radar-Based Imaging for Sign Language Recognition
               in Medical Communication},
  author    = {Mineo, Raffaele and Caligiore, Gaia and others},
  booktitle = {MICCAI},
  year      = {2025},
  organization = {Springer},
}
```

---

## Acknowledgements

Competition hosted by Raffaele Mineo et al. (University of Catania).  
Supervised by Dr. Md. Milon Islam, Khulna University of Engineering & Technology.
