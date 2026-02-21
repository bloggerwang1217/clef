"""
MultiScaleBridge
================

Multi-scale feature projection module.
Connects Swin V2 encoder stages to the autoregressive decoder.

Responsibilities:
- Project each Swin stage to unified d_model dimension
- Add learnable level embeddings to distinguish scales
- Add spatial positional embeddings (freq: sinusoidal, time: sinusoidal)
- Compute valid_ratios for padding handling

NOTE: NO self-attention here! The decoder handles cross-attention
across all concatenated multi-scale levels.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _sinusoidal_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Generate sinusoidal positional encoding.

    Args:
        length: Sequence length
        d_model: Embedding dimension

    Returns:
        [length, d_model] sinusoidal encoding
    """
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(length, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float)
        * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MultiScaleBridge(nn.Module):
    """Multi-scale Bridge.

    Projects and concatenates multi-scale features from Swin stages.
    No self-attention - that's handled by the decoder.

    Input: List of features from different sources (Flow, Swin stages)
    Output: Unified multi-scale features for decoder cross-attention

    When use_flow=True, Level 0 is HarmonizingFlow output (H=1, W=T),
    followed by Swin stages as Levels 1-4.
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
        mel_height: int = 128,
        input_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels

        # Input projections for each level
        # input_dims overrides swin_dims when Flow is included
        proj_dims = input_dims if input_dims is not None else swin_dims
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
            )
            for dim in proj_dims
        ])

        # Level embedding (distinguish different scales)
        self.level_embed = nn.Parameter(torch.zeros(n_levels, d_model))
        nn.init.normal_(self.level_embed, std=0.02)

        # Spatial positional embeddings: fixed sinusoidal (Vaswani et al. 2017)
        # Swin V2 relative position bias does NOT encode absolute position
        # identity — silent input produces identical features at all positions.
        # Sinusoidal PE gives each position a unique, structured encoding
        # (adjacent positions similar, distant positions different).
        #
        # Scale factor controls initial impact relative to feature norm (~22):
        #   scale=1.0 → norm~16 (72% of feature, too disruptive)
        #   scale=0.1 → norm~1.6 (7% of feature, visible but gentle)
        self.spatial_pe_scale = 0.1

        # Freq PE: computed on-the-fly per level (variable H: 1 for Flow, 32/16/8/4 for Swin)
        # Time PE: also sinusoidal, computed on-the-fly (variable length)

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
            # feat: [B, N_patches, C] from Swin hidden_states or Flow
            # Project to d_model
            feat = self.input_projs[lvl](feat)  # [B, N_patches, D]

            # Reshape to 2D for spatial embeddings
            H, W = spatial_shapes_list[lvl]
            feat = feat.view(B, H, W, self.d_model)

            # Add frequency positional embedding (sinusoidal, computed on-the-fly)
            # H varies per level: 1 for Flow, 32/16/8/4 for Swin stages
            freq_pe = _sinusoidal_encoding(H, self.d_model, device)
            feat = feat + freq_pe.view(1, H, 1, self.d_model) * self.spatial_pe_scale

            # Add time positional embedding (sinusoidal)
            time_pe = _sinusoidal_encoding(W, self.d_model, device)
            feat = feat + time_pe.view(1, 1, W, self.d_model) * self.spatial_pe_scale

            # Flatten back to [B, H*W, D]
            feat = feat.reshape(B, H * W, self.d_model)

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
