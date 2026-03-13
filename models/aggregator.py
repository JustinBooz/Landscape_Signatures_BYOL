"""
Aggregator Head for Landscape Signature Extraction.

Architecture:
    GeM Pooling -> MLP Projector -> [MLP Predictor (online only)]

The final output is an L2-normalised 1D Landscape Signature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """
    2-layer MLP with BatchNorm and ReLU used as Projector or Predictor head.

    Architecture: Linear -> BN -> ReLU -> Linear
    This matches the MLP head from the canonical BYOL implementation.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            # LayerNorm is batch-independent, mandatory for micro-batches
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AggregatorHead(nn.Module):
    """
    Full aggregation stack applied on top of the CLS token from the encoder.

    online=True  → Projector → Predictor   (Online Network branch)
    online=False → Projector                (Target Network branch)

    Args:
        embed_dim:   Input dimensionality (DINOv3 output patch dim, e.g. 1024 for ViT-L)
        hidden_dim:  MLP hidden dim (default 2048)
        out_dim:     Final signature / projection dimension (default 1024)
        online:      Whether to append the Predictor MLP (True for Online, False for Target)
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 2048,
        out_dim: int = 1024,
        online: bool = True,
    ):
        super().__init__()
        self.projector = MLPHead(embed_dim, hidden_dim, out_dim)
        self.online = online
        if online:
            self.predictor = MLPHead(out_dim, hidden_dim, out_dim)

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_features: [B, D] — The CLS token from the ViT encoder
        Returns:
            L2-normalised vector [B, out_dim]
        """
        # Bypass pooling, go straight to projection
        x = self.projector(cls_features)        # [B, out_dim]
        if self.online:
            x = self.predictor(x)               # [B, out_dim]  (online only)
            
        return F.normalize(x, dim=-1, p=2)      # [B, out_dim]
