"""
Training Utilities

Covers:
  - cosine_lr        : warmup + cosine annealing LR schedule
  - mixup_dual       : MixUp applied synchronously to RTM + CVD batches
  - cutmix_dual      : CutMix applied synchronously to RTM + CVD batches
  - ModelEMA         : exponential moving average of model weights
  - validate_dual    : evaluation loop for the dual-stream model
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from ..data.augmentations import rand_bbox


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def cosine_lr(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_frac: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup followed by cosine annealing.

    The LR decays from its peak value down to `min_frac × peak`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
        Number of warmup epochs.
    total_epochs : int
        Total number of training epochs.
    min_frac : float
        Minimum LR as a fraction of the base LR.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    """
    def _fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_frac + 0.5 * (1.0 - min_frac) * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _fn)


# ---------------------------------------------------------------------------
# Mixed-sample augmentations (dual-stream versions)
# ---------------------------------------------------------------------------

def mixup_dual(
    rtm: torch.Tensor,
    cvd: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
):
    """
    MixUp applied identically to RTM and CVD batches.

    Returns
    -------
    rtm_mix, cvd_mix, labels_a, labels_b, lam
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(rtm.size(0), device=rtm.device)
    rtm_mix = lam * rtm + (1.0 - lam) * rtm[idx]
    cvd_mix = lam * cvd + (1.0 - lam) * cvd[idx]
    return rtm_mix, cvd_mix, labels, labels[idx], lam


def cutmix_dual(
    rtm: torch.Tensor,
    cvd: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
):
    """
    CutMix applied identically to RTM and CVD batches.

    A random rectangular patch from a second sample is pasted into each
    sample at the same spatial location for both streams, preserving the
    RTM-CVD spatial correspondence.

    Returns
    -------
    rtm_mix, cvd_mix, labels_a, labels_b, lam_adjusted
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(rtm.size(0), device=rtm.device)
    _, _, H, W = rtm.shape
    y1, y2, x1, x2 = rand_bbox(H, W, lam)

    rtm_mix = rtm.clone()
    cvd_mix = cvd.clone()
    rtm_mix[:, :, y1:y2, x1:x2] = rtm[idx, :, y1:y2, x1:x2]
    cvd_mix[:, :, y1:y2, x1:x2] = cvd[idx, :, y1:y2, x1:x2]

    # Re-compute lam from actual box area
    lam_adj = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return rtm_mix, cvd_mix, labels, labels[idx], lam_adj


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the model parameters updated as:
        ema_param = decay * ema_param + (1 - decay) * model_param

    The EMA model typically achieves slightly better accuracy than the last
    checkpoint without any additional training cost.

    Parameters
    ----------
    model : nn.Module
    decay : float
        EMA decay factor (0.9995 works well for ~70 epoch training).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, m_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)
        # Keep BN running stats in sync
        for ema_b, m_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(m_b.data)

    def state_dict(self):
        return self.ema.state_dict()


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_dual(
    model: nn.Module,
    loader,
    device: str = "cuda",
    use_amp: bool = True,
) -> float:
    """
    Compute Top-1 accuracy on a dual-stream validation loader.

    Parameters
    ----------
    model : nn.Module
        RadarCVDNet (or any model accepting (rtm, cvd) inputs).
    loader : DataLoader
        Yields (rtm_imgs, cvd_imgs, labels).
    device : str
    use_amp : bool
        Use automatic mixed precision.

    Returns
    -------
    accuracy : float   (0–100)
    """
    model.eval()
    correct = total = 0
    for rtm_imgs, cvd_imgs, labels in loader:
        rtm_imgs = rtm_imgs.to(device, non_blocking=True)
        cvd_imgs = cvd_imgs.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(rtm_imgs, cvd_imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / max(total, 1)
