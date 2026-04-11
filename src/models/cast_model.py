"""
RadarCVD-Net

Dual-stream architecture for radar-only sign language recognition:

  RTM stream  — EfficientNetV2-S (3 × 224 × 224 range-time map image)
  CVD stream  — ConvNeXt-Tiny   (3 × 224 × 224 cadence-velocity diagram)

Both streams are pre-processed by a Cross-Antenna Self-Attention (CASA)
module before entering their backbone.  Backbone features are projected to a
shared feat_dim, then fused via asymmetric cross-attention (RTM as query,
CVD as key/value) with a learned gated residual.

During training the model optionally returns two auxiliary logits (one per
stream) for a three-way auxiliary loss:

  loss = loss_main + 0.3 * (loss_aux_rtm + loss_aux_cvd)

Reference
---------
"Dual-Stream Cadence Velocity Diagram Fusion with Spatial Attention
for Radar-Only Sign Language Recognition"
MSLR Workshop @ CVPR 2026
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm") from e

from .casa import CrossAntennaSelfAttention
from .fusion import AsymmetricCrossAttentionFusion


class RadarCVDNet(nn.Module):
    """
    Dual-stream radar SLR model.

    Args
    ----
    num_classes : int
        Number of gesture classes (126 for CVPR MSLR 2026 Track 2).
    feat_dim : int
        Projection dimension shared by both backbone heads (default 512).
    rtm_name : str
        timm model name for the RTM backbone.
    cvd_name : str
        timm model name for the CVD backbone.
    pretrained : bool
        Load ImageNet-21k → 1k pretrained weights (recommended).
    drop_rate : float
        Stochastic depth / dropout for both backbones.
    drop_path_rate : float
        Drop-path rate for both backbones.
    casa_heads : int
        Number of MHA heads inside CASA (1 is sufficient, 4 gives +0.3 pp).
    """

    def __init__(
        self,
        num_classes: int = 126,
        feat_dim: int = 512,
        rtm_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        cvd_name: str = "convnext_tiny.fb_in22k_ft_in1k",
        pretrained: bool = True,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        casa_heads: int = 1,
    ) -> None:
        super().__init__()

        # --- Pre-backbone attention (CASA) ---
        self.casa_rtm = CrossAntennaSelfAttention(in_channels=3, num_heads=casa_heads)
        self.casa_cvd = CrossAntennaSelfAttention(in_channels=3, num_heads=casa_heads)

        # --- RTM backbone ---
        self.rtm_backbone = timm.create_model(
            rtm_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        rtm_feat_dim = self.rtm_backbone.num_features

        # --- CVD backbone ---
        self.cvd_backbone = timm.create_model(
            cvd_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        cvd_feat_dim = self.cvd_backbone.num_features

        # Enable gradient checkpointing if available (saves VRAM)
        for bb in (self.rtm_backbone, self.cvd_backbone):
            if hasattr(bb, "set_grad_checkpointing"):
                bb.set_grad_checkpointing(True)

        # --- Projection heads ---
        self.rtm_proj = nn.Linear(rtm_feat_dim, feat_dim)
        self.cvd_proj = nn.Linear(cvd_feat_dim, feat_dim)

        # --- Fusion ---
        self.fusion = AsymmetricCrossAttentionFusion(feat_dim, num_heads=8)

        # --- Classification heads ---
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )
        # Auxiliary heads (used during training with auxiliary loss)
        self.head_rtm = nn.Linear(feat_dim, num_classes)
        self.head_cvd = nn.Linear(feat_dim, num_classes)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(
            f"RadarCVD-Net | {n_params:.1f}M params | "
            f"RTM: {rtm_name} | CVD: {cvd_name}"
        )

    def forward(
        self,
        rtm_img: torch.Tensor,
        cvd_img: torch.Tensor,
        return_aux: bool = False,
    ):
        """
        Args
        ----
        rtm_img : (B, 3, H, W)  — stacked antenna RTM image
        cvd_img : (B, 3, H, W)  — stacked antenna CVD image
        return_aux : bool
            If True, also return per-stream logits for auxiliary loss.

        Returns
        -------
        logits : (B, num_classes)
        [logits_rtm, logits_cvd] : optional, each (B, num_classes)
        """
        rtm_attended = self.casa_rtm(rtm_img)
        cvd_attended = self.casa_cvd(cvd_img)

        rtm_feat = self.rtm_proj(self.rtm_backbone(rtm_attended))
        cvd_feat = self.cvd_proj(self.cvd_backbone(cvd_attended))

        fused  = self.fusion(rtm_feat, cvd_feat)
        logits = self.head(fused)

        if return_aux:
            return logits, self.head_rtm(rtm_feat), self.head_cvd(cvd_feat)
        return logits
