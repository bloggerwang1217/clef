"""
Deformable Bridge (Lightweight)
================================

Multi-scale feature projection module.
Connects Swin V2 encoder stages to the autoregressive decoder.

Responsibilities:
- Project each Swin stage to unified d_model dimension
- Add learnable level embeddings to distinguish scales
- Compute valid_ratios for padding handling

NOTE: NO self-attention here! The decoder's ClefAttention handles
cross-scale attention via deformable sampling across all levels.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class DeformableBridge(nn.Module):
    """Multi-scale Deformable Bridge (Lightweight).

    Simply projects and concatenates multi-scale Swin features.
    No self-attention - that's handled by the decoder.

    Input: F1, F2, F3, F4 from Swin V2 (different spatial resolutions)
    Output: Unified multi-scale features for decoder cross-attention
    """

    def __init__(
        self,
        swin_dims: List[int] = [96, 192, 384, 768],
        d_model: int = 512,
        n_heads: int = 8,  # Unused, kept for API compatibility
        n_levels: int = 4,
        n_points_freq: int = 2,  # Unused
        n_points_time: int = 2,  # Unused
        freq_offset_scale: float = 0.15,  # Unused
        time_offset_scale: float = 0.15,  # Unused
        n_layers: int = 2,  # Unused - no self-attention layers!
        ff_dim: int = 2048,  # Unused
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels

        # Input projections for each level
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
            )
            for dim in swin_dims
        ])

        # Level embedding (distinguish different scales)
        self.level_embed = nn.Parameter(torch.zeros(n_levels, d_model))
        nn.init.normal_(self.level_embed, std=0.02)

        # Optional: dropout after projection
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: List[torch.Tensor],  # [F1, F2, F3, F4] from Swin
        spatial_shapes_list: List[Tuple[int, int]],  # Pre-computed spatial shapes
        valid_ratios: Optional[torch.Tensor] = None,  # [B, L, 2] pre-computed valid ratios
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and concatenate multi-scale features.

        Args:
            features: List of 4 Swin stage outputs, each [B, N_patches, C]
                where N_patches varies per level
            spatial_shapes_list: List of (H, W) tuples for each level
            valid_ratios: Pre-computed valid ratios [B, L, 2] where 2 = (time, freq)

        Returns:
            output: Projected features [B, N_total, D]
            spatial_shapes: [L, 2] each level's (H, W)
            level_start_index: [L] start index for each level
            valid_ratios: [B, L, 2] valid ratios for each level
        """
        B = features[0].shape[0]
        device = features[0].device

        src_flatten = []

        for lvl, feat in enumerate(features):
            # feat: [B, N_patches, C] from Swin hidden_states
            # Project to d_model
            feat = self.input_projs[lvl](feat)  # [B, N_patches, D]

            # Add level embedding
            feat = feat + self.level_embed[lvl].view(1, 1, -1)

            # Apply dropout
            feat = self.dropout(feat)

            src_flatten.append(feat)

        # Concatenate all levels
        output = torch.cat(src_flatten, dim=1)  # [B, N_total, D]

        # Build spatial_shapes tensor from provided list
        spatial_shapes = torch.tensor(
            spatial_shapes_list, dtype=torch.long, device=device
        )

        level_start_index = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])

        # Use provided valid_ratios or default to all ones
        if valid_ratios is None:
            valid_ratios = torch.ones(B, self.n_levels, 2, device=device)

        return output, spatial_shapes, level_start_index, valid_ratios
