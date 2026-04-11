# RadarCVD-Net

**Dual-Stream Cadence Velocity Diagram Fusion with Spatial Attention for Radar-Only Sign Language Recognition**

> MSLR Workshop @ CVPR 2026 · Denver, Colorado, USA · June 3–4, 2026

---

## Overview

RadarCVD-Net is a dual-stream convolutional-transformer architecture for sign language recognition using **only 60 GHz radar data** (no RGB or depth).  The model simultaneously processes:

- **RTM stream** — three-antenna Range-Time Maps rendered as a 3-channel image, processed by EfficientNetV2-S
- **CVD stream** — per-antenna Cadence-Velocity Diagrams (Doppler spectra) rendered as a 3-channel image, processed by ConvNeXt-Tiny

Both streams pass through a lightweight **Cross-Antenna Self-Attention (CASA)** module before their backbone, and are fused via **asymmetric cross-attention** (RTM as query, CVD as key/value) with a learned gated residual.

```
RTM₁ RTM₂ RTM₃  ──►  CASA  ──►  EfficientNetV2-S  ──►  proj  ──►  Q ─┐
                                                                        ├─► Asymmetric Cross-Attn ──► gate ──► head ──► logits
CVD₁ CVD₂ CVD₃  ──►  CASA  ──►  ConvNeXt-Tiny      ──►  proj  ──► K,V─┘
```

---

## Results

| Model | CV Top-1 Acc | Notes |
|---|---|---|
| Single-backbone baseline (5-fold) | 77.2 ± 1.3% | EfficientNetV2-S, RTM only |
| RadarCVD-Net (5-fold CV) | **80.5 ± 0.9%** | Full model, CASA 4-head |
| RadarCVD-Net (single 90/10 run) | 81.73% | Kaggle leaderboard |
| EfficientNetV2-S + ConvNeXt-Tiny Ensemble (10-fold) | 84.88% | Kaggle leaderboard |

> **Note:** Kaggle leaderboard scores are not directly comparable across models due to different compute budgets (single run vs. 10-fold ensemble).

### Ablation (5-fold CV)

| Variant | Top-1 Acc |
|---|---|
| RTM only (no CVD) | 79.6% |
| + CVD stream (no linearisation) | 80.3% |
| + CVD stream (with linearisation) | **80.5%** |
| No CASA | 79.8% |
| CASA 1-head | 80.2% |
| CASA 4-heads | **80.5%** |

---

## Repository Structure

```
RadarCVD-Net/
├── README.md
├── LICENSE                         MIT
├── requirements.txt
├── .gitignore
│
├── src/                            Importable Python package
│   ├── models/
│   │   ├── casa.py                 Cross-Antenna Self-Attention
│   │   ├── fusion.py               Asymmetric Cross-Attention Fusion
│   │   └── radarcvd_net.py         Full RadarCVDNet model
│   ├── data/
│   │   ├── cvd.py                  Pseudo-CVD computation
│   │   ├── augmentations.py        Physics-aware radar augmentations
│   │   └── dataset.py              DualStreamRTMDataset + TestDualStreamDataset
│   └── utils/
│       └── training.py             EMA, MixUp, CutMix, cosine LR, validate
│
├── configs/
│   └── radarcvd_net.yaml           Full hyperparameter config
│
└── notebooks/                      Kaggle-runnable notebooks
    ├── 01_RadarCVD_Net_training_val0.8173.ipynb     Paper pipeline (81.73%)
    ├── 02_ScoreMaximizer_V2_val0.8390.ipynb         V2 mega-ensemble (83.90%)
    ├── 03_BestModel_KFold_Ensemble_val0.8488.ipynb  Best result (84.88%)
    └── 04_Inference_TestSet_Combine.ipynb           Inference + submission merge
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Tested with: Python 3.10, PyTorch 2.2, CUDA 12.1, timm 0.9.16.

### Dataset

The **MultiMeDaLIS** dataset is available through the competition page:

> [Kaggle — CVPR MSLR 2026 Track 2](https://www.kaggle.com/competitions/cvpr-mslr-2026-track-2)

Expected directory layout after download:

```
cvpr-mslr-2026-track-2/
├── train/
│   ├── 0_CLASSNAME/
│   │   └── SAMPLE_{id}/
│   │       ├── SAMPLE{id}_RTM1.npy
│   │       ├── SAMPLE{id}_RTM2.npy
│   │       └── SAMPLE{id}_RTM3.npy
│   └── ... (126 classes)
├── val/
│   └── SAMPLE_{id}/
└── test/
    └── SAMPLE_{id}/
```

---

## Reproducing Results

### Option A — Kaggle Notebooks (recommended)

The `notebooks/` directory contains self-contained Kaggle notebooks that auto-detect the dataset path.  Run them in order on a Kaggle P100/T4 GPU:

| Notebook | Purpose | Expected Score |
|---|---|---|
| `01_RadarCVD_Net_training_val0.8173.ipynb` | Train RadarCVD-Net (paper model) | ~81.7% |
| `02_ScoreMaximizer_V2_val0.8390.ipynb` | V2 pipeline with diverse backbones | ~83.9% |
| `03_BestModel_KFold_Ensemble_val0.8488.ipynb` | 10-fold ensemble (best result) | ~84.9% |
| `04_Inference_TestSet_Combine.ipynb` | Merge val + test predictions | — |

### Option B — Python Package

```python
import torch
from src.models import RadarCVDNet

model = RadarCVDNet(
    num_classes=126,
    feat_dim=512,
    rtm_name="tf_efficientnetv2_s.in21k_ft_in1k",
    cvd_name="convnext_tiny.fb_in22k_ft_in1k",
    pretrained=True,
)

# Forward pass
rtm_img = torch.randn(1, 3, 224, 224)   # 3-antenna RTM image
cvd_img = torch.randn(1, 3, 224, 224)   # 3-antenna CVD image

logits = model(rtm_img, cvd_img)         # (1, 126)

# With auxiliary loss during training
logits, logits_rtm, logits_cvd = model(rtm_img, cvd_img, return_aux=True)
```

### CVD Computation

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

# sample_list: [(sample_dir, label), ...]
train_ds = DualStreamRTMDataset(sample_list, img_size=224, max_T=48, augment=True)
loader   = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4)

for rtm_imgs, cvd_imgs, labels in loader:
    # rtm_imgs: (B, 3, 224, 224)
    # cvd_imgs: (B, 3, 224, 224)
    # labels:   (B,)
    pass
```

---

## Architecture Details

### Cross-Antenna Self-Attention (CASA)

A lightweight (~1.5 K parameter) pre-backbone module that learns inter-antenna importance weights via multi-head self-attention over spatial embeddings of each RX channel.  Applied independently to RTM and CVD inputs.

### CVD Computation Pipeline

```
RTM (dB) → dB→linear → Blackman-Harris window → zero-pad to n_fft
         → rFFT (slow-time axis) → |·| → drop DC → 20·log10 → CVD (dB)
```

The CVD captures Doppler (velocity) information encoded in the slow-time dimension of the RTM, complementing the range-time structure.

### Fusion

Asymmetric cross-attention with RTM features as queries and CVD features as keys/values, followed by a position-wise FFN and a gated residual that blends the attended representation with the original RTM features.

### Training Recipe

| Component | Setting |
|---|---|
| Optimiser | AdamW, lr=3e-4, wd=0.05 |
| Schedule | 5-epoch warmup + cosine annealing |
| Augmentation | MixUp (α=0.4) + CutMix (α=1.0) + physics augmentations |
| Regularisation | Label smoothing 0.1, Dropout 0.3, DropPath 0.2 |
| EMA | decay=0.9995 |
| SWA | starts at epoch 56/70, lr=1e-5 |
| Checkpointing | Top-5 val checkpoints ensembled at inference |
| TTA | 5-view test-time augmentation |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{radarcvdnet2026,
  title     = {Dual-Stream Cadence Velocity Diagram Fusion with Spatial Attention
               for Radar-Only Sign Language Recognition},
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
