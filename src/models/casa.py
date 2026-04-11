"""
Cross-Antenna Self-Attention (CASA)

Lightweight pre-backbone attention module that weights radar antenna channels
based on pairwise inter-antenna interactions. Operates on 3-channel RTM or CVD
tensors (one channel per RX antenna) before they enter the backbone.

Parameter count: ~1.5 K  (deliberately tiny — acts as a channel gate only)

Reference
---------
"CAST at SignEval 2026: Channel-Aware Spatial Transfer Learning with
Pseudo-Image Radar for Sign Language Recognition"
MSLR Workshop @ CVPR 2026
"""

import torch
import torch.nn as nn


class CrossAntennaSelfAttention(nn.Module):
    """
    Assign learned importance weights to each RX-antenna channel via
    multi-head self-attention over per-antenna spatial embeddings.

    Args
    ----
    in_channels : int
        Number of antenna channels (default 3 for RTM1/RTM2/RTM3).
    num_heads : int
        Number of attention heads in the inner MHA layer.
    """

    def __init__(self, in_channels: int = 3, num_heads: int = 1) -> None:
        super().__init__()

        # Lightweight spatial embedding: one shared extractor per channel
        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                 # -> (B, 16)
        )

        self.attn = nn.MultiheadAttention(16, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(16)

        # Per-channel scalar gate: 16-dim token -> scalar weight in (0, 1)
        self.channel_gate = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, C, H, W)  — C antenna channels

        Returns
        -------
        x_attended : (B, C, H, W)  — channels re-weighted by attention
        """
        B, C, H, W = x.shape

        # Extract a 16-d token for every antenna channel
        tokens = torch.stack(
            [self.feat_extract(x[:, i : i + 1, :, :]) for i in range(C)],
            dim=1,
        )  # (B, C, 16)

        # Self-attention over antenna dimension
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = self.norm(attn_out + tokens)           # residual + norm

        # Compute per-channel gate weights
        weights = torch.stack(
            [self.channel_gate(attn_out[:, i, :]) for i in range(C)],
            dim=1,
        ).unsqueeze(-1)                                   # (B, C, 1, 1)

        return x * weights
