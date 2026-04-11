"""
Asymmetric Cross-Attention Fusion

Fuses RTM and CVD feature vectors using cross-attention where:
  - RTM features act as *queries*  (primary stream)
  - CVD features act as *keys and values* (auxiliary stream)

A gated residual connection blends the attended representation with the
original RTM feature, so the model can fall back to RTM-only when CVD
information is uninformative.

Reference
---------
"Dual-Stream Cadence Velocity Diagram Fusion with Spatial Attention
for Radar-Only Sign Language Recognition"
MSLR Workshop @ CVPR 2026
"""

import torch
import torch.nn as nn


class AsymmetricCrossAttentionFusion(nn.Module):
    """
    Asymmetric cross-attention fusion block.

    Args
    ----
    feat_dim : int
        Common projection dimension for both RTM and CVD features.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability on attention weights and FFN.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            feat_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(dropout),
        )

        # Gated residual: learns how much of the fused vs. original RTM to keep
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        rtm_feat: torch.Tensor,
        cvd_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        rtm_feat : (B, D)  — RTM backbone features (queries)
        cvd_feat : (B, D)  — CVD backbone features (keys & values)

        Returns
        -------
        fused : (B, D)
        """
        # Unsqueeze to sequence length 1 for MHA API: (B, 1, D)
        q  = rtm_feat.unsqueeze(1)
        kv = cvd_feat.unsqueeze(1)

        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = self.norm1(attn_out + q).squeeze(1)   # residual, (B, D)

        ffn_out  = self.ffn(attn_out)
        enhanced = self.norm2(ffn_out + attn_out)

        # Gated blend between fused and original RTM
        g = self.gate(torch.cat([rtm_feat, enhanced], dim=1))
        fused = g * enhanced + (1.0 - g) * rtm_feat

        return fused
