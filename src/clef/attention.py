"""
CLEF Attention: Content-aware Learned-prior Event Focusing Attention
====================================================================

The core innovation of Clef architecture for audio-to-score transcription.

Naming origin:
- Content-aware: Predict attention region from decoder hidden state
- Learned-prior: freq_prior / time_prior learn "where to look"
- Event: Focus on musical events (notes, chords)
- Focusing: Sparse sampling, only look at important positions

Design rationale (based on cognitive science):
- Human brain's stream segregation (Bregman, 1990) is content-dependent
- freq_prior + time_prior do "coarse localization"
- offset only needs small-range "local detail"
- Square sampling (2x2) is sufficient since prior already selected the region

Relationship with Stripe-Transformer / hFT-Transformer:
- They use full attention to separate freq/time processing
- We use learned spatial priors + sparse sampling
- Lower complexity, comparable effectiveness (for piano/solo tasks)

Generality:
- Music: Time x Frequency
- Image: X x Y x Scale
- Extensible to any multi-dimensional focusing problem

Design:
- Separate n_points_freq / n_points_time for flexibility (but typically equal)
- Separate freq_offset_scale / time_offset_scale (but typically equal for square)
- Decoupled level-point attention: level selection (softmax over L) and point
  selection (softmax over K per level) use separate gradient paths, with
  level bias initialization for scale-axis specialization (analogous to
  freq_group for frequency-axis specialization)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FluxAttention(nn.Module):
    """CLEF Attention with content-aware spatial priors and square sampling.

    Key design choices:
    - freq_prior: Predict "high or low frequency" from hidden state
    - time_prior: Predict "which time point" from position embedding
    - Square sampling 2x2: prior locates, offset refines locally
    - Both offset_scale params default to same value (true square)

    Piano application:
    - Predicting right-hand melody -> freq_prior outputs high value
    - Predicting left-hand harmony -> freq_prior outputs low value
    """

    def __init__(
        self,
        d_model: int = 512,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points_freq: int = 2,        # Frequency direction: local detail
        n_points_time: int = 2,        # Time direction: local detail
        freq_offset_scale: float = 1.0,  # ±1 pixel in feature map space
        time_offset_scale: float = 1.0,  # ±1 pixel in feature map space
    ):
        super().__init__()

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points_freq = n_points_freq
        self.n_points_time = n_points_time
        self.n_points = n_points_freq * n_points_time  # Total sampling points
        self.freq_offset_scale = freq_offset_scale
        self.time_offset_scale = time_offset_scale

        self.head_dim = d_model // n_heads

        # Separate prediction of time and freq offsets
        self.time_offset_proj = nn.Linear(
            d_model,
            n_heads * n_levels * n_points_time
        )
        self.freq_offset_proj = nn.Linear(
            d_model,
            n_heads * n_levels * n_points_freq
        )

        # Decoupled level-point attention (inspired by auditory cortex):
        # Level selection and point selection use separate softmax, so level
        # specialization gets its own gradient path independent of point weights.
        # Analogous to freq_group for frequency axis — this provides inductive
        # bias for scale (resolution) axis.
        self.level_weights = nn.Linear(d_model, n_heads * n_levels)
        self.point_weights = nn.Linear(d_model, n_heads * n_levels * self.n_points)

        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize with square grid and level-specialization bias.

        Offset initialization (offset_scale=1.0, Deformable DETR style):
        - Bias = ±0.5 pixel in feature map space (within tanh linear region)
        - After spatial normalization: Level 0 (H=32) -> ±0.5/32 = ±0.016
          Level 3 (H=4) -> ±0.5/4 = ±0.125. Actual 2x2 grid, not degenerate.

        Level bias initialization (inspired by freq_group success):
        - Each pair of heads has a soft preference for one level
        - Analogous to auditory cortex: different regions handle different
          temporal integration windows (fine detail vs structure)
        - Head 0,1 -> Level 0 (+1.0 bias), Head 2,3 -> Level 1, etc.
        - softmax(+1.0, 0, 0, 0) ≈ (0.45, 0.18, 0.18, 0.18): soft, learnable
        """
        # === Time offset initialization ===
        nn.init.constant_(self.time_offset_proj.weight, 0.)

        # Bias: ±0.5 pixel spread (atanh(0.5/scale) puts it in tanh linear region)
        time_init = torch.linspace(-0.5, 0.5, self.n_points_time)

        time_bias = time_init.view(1, 1, self.n_points_time)
        time_bias = time_bias.expand(self.n_heads, self.n_levels, -1)

        with torch.no_grad():
            self.time_offset_proj.bias.copy_(time_bias.flatten())

        # === Freq offset initialization ===
        nn.init.constant_(self.freq_offset_proj.weight, 0.)

        freq_init = torch.linspace(-0.5, 0.5, self.n_points_freq)

        freq_bias = freq_init.view(1, 1, self.n_points_freq)
        freq_bias = freq_bias.expand(self.n_heads, self.n_levels, -1)

        with torch.no_grad():
            self.freq_offset_proj.bias.copy_(freq_bias.flatten())

        # === Level weights: soft level-specialization bias ===
        # Each pair of heads prefers a different level (independent of freq_group).
        # Frequency axis (bass/soprano) and scale axis (detail/structure) are
        # orthogonal — a bass head may need fine detail (Level 0) or broad
        # context (Level 3) depending on the musical content.
        nn.init.constant_(self.level_weights.weight, 0.)

        level_bias = torch.zeros(self.n_heads, self.n_levels)
        heads_per_level = max(1, self.n_heads // self.n_levels)
        for lvl in range(self.n_levels):
            start_head = lvl * heads_per_level
            end_head = min(start_head + heads_per_level, self.n_heads)
            level_bias[start_head:end_head, lvl] = 1.0

        with torch.no_grad():
            self.level_weights.bias.copy_(level_bias.flatten())

        # === Point weights: uniform initialization ===
        nn.init.constant_(self.point_weights.weight, 0.)
        nn.init.constant_(self.point_weights.bias, 0.)

        # === Projection layers: Xavier ===
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def compute_value_cache(
        self,
        value: torch.Tensor,              # [B, N_v, D]
        spatial_shapes: torch.Tensor,     # [L, 2]
        level_start_index: torch.Tensor,  # [L]
    ) -> List[torch.Tensor]:
        """Pre-compute projected and reshaped values per level for caching.

        Call once per chunk, reuse across all decoding steps.

        Returns:
            List of [B*H, D_head, H_l, W_l] tensors, one per level.
        """
        B, N_v, _ = value.shape
        value = self.value_proj(value)  # [B, N_v, D]
        value = value.view(B, N_v, self.n_heads, self.head_dim)  # [B, N_v, H, D_head]

        level_values = []
        for lid in range(self.n_levels):
            H_l, W_l = spatial_shapes[lid].tolist()
            start = level_start_index[lid].item()
            end = start + H_l * W_l
            value_l = value[:, start:end, :, :]  # [B, H_l*W_l, H, D_head]
            value_l = value_l.permute(0, 2, 3, 1)  # [B, H, D_head, H_l*W_l]
            value_l = value_l.reshape(B * self.n_heads, self.head_dim, H_l, W_l)
            level_values.append(value_l)

        return level_values

    def forward(
        self,
        query: torch.Tensor,              # [B, N_q, D]
        reference_points: torch.Tensor,   # [B, N_q, H, L, 2] or [B, N_q, L, 2]
        value: torch.Tensor,              # [B, N_v, D]
        spatial_shapes: torch.Tensor,     # [L, 2] each level's (H, W)
        level_start_index: torch.Tensor,  # [L] start index in value for each level
        valid_ratios: Optional[torch.Tensor] = None,  # [B, L, 2] padding valid ratios
        value_cache: Optional[List[torch.Tensor]] = None,  # from compute_value_cache
    ) -> torch.Tensor:
        """Forward pass with content-aware deformable attention.

        Args:
            query: Decoder query embeddings [B, N_q, D]
            reference_points: Normalized reference coordinates, either:
                - [B, N_q, H, L, 2] per-head (from decoder with n_freq_groups > 1)
                - [B, N_q, L, 2] shared across heads (from encoder self-attention)
                where 2 = (time, freq) in [0, 1]
            value: Multi-scale encoder features [B, N_v, D]
            spatial_shapes: Spatial shape (H, W) for each level [L, 2]
            level_start_index: Start index in value for each level [L]
            valid_ratios: Valid ratio for padding handling [B, L, 2]
            value_cache: Pre-computed per-level value maps (skips value_proj)

        Returns:
            Output features [B, N_q, D]
        """
        B, N_q, _ = query.shape
        Kt = self.n_points_time
        Kf = self.n_points_freq

        # === 1. Predict offsets ===

        # Time offset: [B, N_q, H, L, Kt]
        time_offset = self.time_offset_proj(query)  # [B, N_q, H*L*Kt]
        time_offset = time_offset.view(B, N_q, self.n_heads, self.n_levels, Kt)
        time_offset = time_offset.tanh() * self.time_offset_scale

        # Freq offset: [B, N_q, H, L, Kf]
        freq_offset = self.freq_offset_proj(query)  # [B, N_q, H*L*Kf]
        freq_offset = freq_offset.view(B, N_q, self.n_heads, self.n_levels, Kf)
        freq_offset = freq_offset.tanh() * self.freq_offset_scale

        # === 2. Compose into 2D sampling grid ===

        # Expand for outer product
        time_grid = time_offset.unsqueeze(-1)  # [B, N_q, H, L, Kt, 1]
        freq_grid = freq_offset.unsqueeze(-2)  # [B, N_q, H, L, 1, Kf]

        # Create sampling grid [B, N_q, H, L, Kt, Kf, 2]
        # Coordinate order: (time, freq) corresponds to (x, y)
        sampling_offsets = torch.stack([
            time_grid.expand(-1, -1, -1, -1, -1, Kf),
            freq_grid.expand(-1, -1, -1, -1, Kt, -1),
        ], dim=-1)

        # Flatten to [B, N_q, H, L, n_points, 2]
        sampling_offsets = sampling_offsets.flatten(-3, -2)

        # === 3. Compute sampling locations ===

        # Offset normalization: divide by each level's spatial size
        offset_normalizer = torch.stack([
            spatial_shapes[..., 1].float(),  # W (time)
            spatial_shapes[..., 0].float(),  # H (freq)
        ], dim=-1)  # [L, 2]

        # Handle both per-head [B, N_q, H, L, 2] and shared [B, N_q, L, 2] reference_points
        if reference_points.dim() == 4:
            # Shared: [B, N_q, L, 2] -> [B, N_q, 1, L, 1, 2]
            ref_expanded = reference_points[:, :, None, :, None, :]
        else:
            # Per-head: [B, N_q, H, L, 2] -> [B, N_q, H, L, 1, 2]
            ref_expanded = reference_points.unsqueeze(-2)

        # sampling_offsets: [B, N_q, H, L, K, 2]
        # offset_normalizer: [L, 2] -> [1, 1, 1, L, 1, 2]
        normalized_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        sampling_locations = ref_expanded + normalized_offsets  # [B, N_q, H, L, K, 2]

        # Constrain within valid range
        if valid_ratios is not None:
            # valid_ratios: [B, L, 2] -> [B, 1, 1, L, 1, 2]
            vr = valid_ratios[:, None, None, :, None, :]
            sampling_locations = sampling_locations * vr

        # Clamp to [0, 1]
        sampling_locations = sampling_locations.clamp(0, 1)

        # === 4. Decoupled level-point attention weights ===
        # Level selection: softmax over L (which resolution?)
        # Point selection: softmax over K per level (where within that level?)
        # Final weight = level_weight[l] * point_weight[l,k]
        # This gives level selection its own gradient path.

        lw = self.level_weights(query)  # [B, N_q, H*L]
        lw = lw.view(B, N_q, self.n_heads, self.n_levels)
        lw = F.softmax(lw, dim=-1)  # [B, N_q, H, L]

        pw = self.point_weights(query)  # [B, N_q, H*L*K]
        pw = pw.view(B, N_q, self.n_heads, self.n_levels, self.n_points)
        pw = F.softmax(pw, dim=-1)  # [B, N_q, H, L, K]

        # Combine: [B, N_q, H, L, K]
        attention_weights = lw.unsqueeze(-1) * pw

        # === 5. Deformable Attention computation ===

        if value_cache is not None:
            # Use pre-computed per-level value maps (skip value_proj + reshape)
            output = self._deformable_attention_core_cached(
                value_cache, sampling_locations, attention_weights
            )
        else:
            _, N_v, _ = value.shape
            value = self.value_proj(value)
            value = value.view(B, N_v, self.n_heads, self.head_dim)
            output = self._deformable_attention_core(
                value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights
            )

        output = self.output_proj(output)

        return output

    def _deformable_attention_core(
        self,
        value: torch.Tensor,              # [B, N_v, H, D_head]
        spatial_shapes: torch.Tensor,     # [L, 2]
        level_start_index: torch.Tensor,  # [L]
        sampling_locations: torch.Tensor,  # [B, N_q, H, L, K, 2]
        attention_weights: torch.Tensor,  # [B, N_q, H, L, K]
    ) -> torch.Tensor:
        """Pure PyTorch implementation (can be replaced with CUDA kernel).

        Uses F.grid_sample for bilinear interpolation at sampling locations.
        """
        B, N_q, H, L, K, _ = sampling_locations.shape
        D_head = value.shape[-1]

        # Convert sampling_locations to grid_sample format [-1, 1]
        sampling_grids = 2 * sampling_locations - 1

        # Accumulate weighted sum directly (no stack, saves 768 MB peak memory)
        # Old: stack 6 levels -> [B, H, D_head, N_q, K, L] -> weighted sum
        # New: for each level, compute weighted contribution and accumulate
        output = torch.zeros(B, N_q, H, D_head, device=value.device, dtype=value.dtype)

        for lid in range(L):
            H_l, W_l = spatial_shapes[lid].tolist()
            start = level_start_index[lid].item()
            end = start + H_l * W_l

            # Extract this level's values
            value_l = value[:, start:end, :, :]  # [B, H_l*W_l, H, D_head]
            value_l = value_l.permute(0, 2, 3, 1)  # [B, H, D_head, H_l*W_l]
            value_l = value_l.reshape(B * H, D_head, H_l, W_l)

            # Extract this level's sampling grid
            # grid_sample expects (x, y) where x is width (time), y is height (freq)
            grid_l = sampling_grids[:, :, :, lid, :, :]  # [B, N_q, H, K, 2]
            grid_l = grid_l.permute(0, 2, 1, 3, 4)  # [B, H, N_q, K, 2]
            grid_l = grid_l.reshape(B * H, N_q, K, 2)

            # Bilinear sampling
            sampled = F.grid_sample(
                value_l,
                grid_l,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # [B*H, D_head, N_q, K]

            sampled = sampled.view(B, H, D_head, N_q, K)  # [B, H, D_head, N_q, K]

            # Get this level's attention weights: [B, N_q, H, K]
            attn_l = attention_weights[:, :, :, lid, :]  # [B, N_q, H, K]

            # Weighted sum over K points for this level
            # sampled: [B, H, D_head, N_q, K] -> permute to [B, N_q, H, D_head, K]
            # attn_l: [B, N_q, H, K] -> [B, N_q, H, 1, K]
            sampled = sampled.permute(0, 3, 1, 2, 4)  # [B, N_q, H, D_head, K]
            attn_l = attn_l.unsqueeze(3)  # [B, N_q, H, 1, K]

            # Accumulate: [B, N_q, H, D_head]
            output = output + (sampled * attn_l).sum(dim=-1)

        # Reshape: [B, N_q, D]
        output = output.reshape(B, N_q, -1)

        return output

    def _deformable_attention_core_cached(
        self,
        level_values: List[torch.Tensor],  # pre-computed [B*H, D_head, H_l, W_l] per level
        sampling_locations: torch.Tensor,   # [B, N_q, H, L, K, 2]
        attention_weights: torch.Tensor,    # [B, N_q, H, L, K]
    ) -> torch.Tensor:
        """Deformable attention using pre-computed value cache.

        Skips value_proj and per-level reshape (already done in compute_value_cache).
        Uses accumulation instead of stack to save memory.
        """
        B, N_q, H, L, K, _ = sampling_locations.shape
        D_head = level_values[0].shape[1]

        # Convert to grid_sample format [-1, 1]
        sampling_grids = 2 * sampling_locations - 1

        # Accumulate weighted sum directly (no stack, saves memory)
        output = torch.zeros(B, N_q, H, D_head, device=sampling_locations.device, dtype=level_values[0].dtype)

        for lid in range(L):
            value_l = level_values[lid]  # [B*H, D_head, H_l, W_l]

            grid_l = sampling_grids[:, :, :, lid, :, :]  # [B, N_q, H, K, 2]
            grid_l = grid_l.permute(0, 2, 1, 3, 4)  # [B, H, N_q, K, 2]
            grid_l = grid_l.reshape(B * H, N_q, K, 2)

            sampled = F.grid_sample(
                value_l, grid_l,
                mode='bilinear', padding_mode='zeros', align_corners=False,
            )  # [B*H, D_head, N_q, K]

            sampled = sampled.view(B, H, D_head, N_q, K)  # [B, H, D_head, N_q, K]

            # Get this level's attention weights and accumulate
            attn_l = attention_weights[:, :, :, lid, :]  # [B, N_q, H, K]
            sampled = sampled.permute(0, 3, 1, 2, 4)  # [B, N_q, H, D_head, K]
            attn_l = attn_l.unsqueeze(3)  # [B, N_q, H, 1, K]
            output = output + (sampled * attn_l).sum(dim=-1)

        output = output.reshape(B, N_q, -1)

        return output


class DeformableEncoderLayer(nn.Module):
    """Encoder layer with deformable self-attention for multi-scale feature fusion."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points_freq: int = 2,
        n_points_time: int = 2,
        freq_offset_scale: float = 1.0,
        time_offset_scale: float = 1.0,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = FluxAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points_freq=n_points_freq,
            n_points_time=n_points_time,
            freq_offset_scale=freq_offset_scale,
            time_offset_scale=time_offset_scale,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,               # [B, N, D]
        reference_points: torch.Tensor,  # [B, N, L, 2]
        spatial_shapes: torch.Tensor,    # [L, 2]
        level_start_index: torch.Tensor,  # [L]
        valid_ratios: torch.Tensor,      # [B, L, 2]
    ) -> torch.Tensor:
        # Self-attention
        src2 = self.self_attn(
            src, reference_points, src,
            spatial_shapes, level_start_index, valid_ratios
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src = src + self.ffn(src)
        src = self.norm2(src)

        return src
