"""
Deformable Decoder
==================

Transformer decoder with FluxAttention cross-attention.

Key innovation: Content-Dependent Reference Points
- time_prior: Predict "which time point to look at" from positional embedding
- freq_prior: Predict "high or low frequency region" from hidden state
- This corresponds to stream tracking in human auditory perception (Bregman, 1990)

Piano application:
- When predicting right-hand melody -> freq_prior outputs high value
- When predicting left-hand harmony -> freq_prior outputs low value
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FluxAttention


class DeformableDecoderLayer(nn.Module):
    """Decoder Layer with Content-Aware Deformable Cross-Attention.

    Structure:
    1. Causal Self-Attention (standard)
    2. FluxAttention with content-dependent reference points
    3. FFN

    The content-dependent reference points are the key innovation:
    - time_prior(pos_embed) -> "where in time to look"
    - freq_prior(hidden_state) -> "which frequency region to focus on"
    """

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
        use_time_prior: bool = True,
        use_freq_prior: bool = True,
        n_freq_groups: int = 1,
        refine_range: float = 0.1,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.use_time_prior = use_time_prior
        self.use_freq_prior = use_freq_prior
        self.n_freq_groups = n_freq_groups
        self.refine_range = refine_range

        # 1. Causal Self-Attention (use F.scaled_dot_product_attention for Flash Attention)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.self_attn_dropout = dropout
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. FluxAttention (Cross-attention) with gain control
        # Problem: CA output norm (~3) << residual norm (~22), so CA gets drowned
        # after residual addition + LayerNorm. Training actively erases any good
        # initialization (level bias, offset grid) because gradient is too weak.
        #
        # Solution (CaiT LayerScale):
        # ca_gamma: per-channel learnable scale, init=1.0 (following CaiT).
        # No LayerNorm before gamma — raw CA output norm (~11 with proper init)
        # naturally reflects feature quality: bad CA → small norm → little impact,
        # good CA → large norm → decoder can use it. LayerNorm would force all
        # outputs to ~22.6 regardless of quality, amplifying noise at init.
        self.cross_attn = FluxAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points_freq=n_points_freq,
            n_points_time=n_points_time,
            freq_offset_scale=freq_offset_scale,
            time_offset_scale=time_offset_scale,
        )
        self.ca_gamma = nn.Parameter(torch.ones(d_model) * 0.5)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Content-Dependent Reference Points (key innovation!)
        if use_time_prior:
            # Predict time location from position embedding
            self.time_prior = nn.Linear(d_model, 1)
        else:
            self.time_prior = None

        if use_freq_prior:
            # Predict frequency region from hidden state (content-dependent!)
            # n_freq_groups > 1: per-head freq_prior for multi-stream tracking
            # e.g. n_freq_groups=4, n_heads=8 -> each group covers 2 heads
            # Analogous to cochlear tonotopic organization: multiple parallel
            # frequency trackers, each attending to different register
            assert n_heads % n_freq_groups == 0, \
                f"n_heads ({n_heads}) must be divisible by n_freq_groups ({n_freq_groups})"
            self.heads_per_group = n_heads // n_freq_groups
            self.freq_prior = nn.Linear(d_model, n_freq_groups)
        else:
            self.freq_prior = None
            self.heads_per_group = n_heads

        # Content-based refinement (+/-10%)
        self.reference_refine = nn.Linear(d_model, 2)

        self._init_reference_predictors()

    def _init_reference_predictors(self):
        """Initialize reference point predictors.

        Key insight: gain=0.001 causes reference points to be stuck at
        sigmoid(0)=0.5 (feature map center). The gradient signal from the
        language model prior is too weak to push them out. Instead:

        - freq_prior: Spread bias across frequency axis so each head group
          starts at a different register (bass/tenor/alto/soprano).
          Weight gain=0.1 gives W@x std ≈ 0.14, enough for content-dependent
          modulation within each band while keeping sigmoid gradient healthy.
        - reference_refine: std=0.01 (was 0.001) for visible initial offsets.
        - time_prior: Prefer fixed linspace (use_time_prior=False) because
          position embeddings (std=0.02) are too small to produce a 0→1 ramp
          through a linear layer.
        """
        if self.time_prior is not None:
            nn.init.xavier_uniform_(self.time_prior.weight, gain=0.1)
            nn.init.constant_(self.time_prior.bias, 0.)

        if self.freq_prior is not None:
            # Weight: gain=0.1 for content-dependent modulation (W@x std ≈ 0.14)
            nn.init.xavier_uniform_(self.freq_prior.weight, gain=0.1)

            # Bias: spread across frequency axis
            # sigmoid(bias) -> target freq position
            #   -1.386 -> 0.2 (bass)
            #   -0.405 -> 0.4 (tenor)
            #   +0.405 -> 0.6 (alto)
            #   +1.386 -> 0.8 (soprano)
            import math
            n_groups = self.freq_prior.bias.shape[0]
            with torch.no_grad():
                for g in range(n_groups):
                    # Evenly spaced targets in (0, 1), e.g. [0.2, 0.4, 0.6, 0.8]
                    target = (g + 1) / (n_groups + 1)
                    self.freq_prior.bias[g] = math.log(target / (1 - target))

        # Refinement: std=0.01 for meaningful initial offsets
        nn.init.normal_(self.reference_refine.weight, std=0.01)
        nn.init.constant_(self.reference_refine.bias, 0.)

    def forward(
        self,
        tgt: torch.Tensor,           # [B, S, D]
        memory: torch.Tensor,        # [B, N_total, D]
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,  # [B, L, 2]
        tgt_pos: Optional[torch.Tensor] = None,  # [B, S, D] or [1, S, D]
        past_kv: Optional[tuple] = None,  # (k_cache, v_cache) each [B, H, S_prev, D_head]
        use_cache: bool = False,
        value_cache: Optional[List[torch.Tensor]] = None,  # pre-computed cross-attn values
    ):
        """Forward pass with content-dependent reference points.

        Args:
            tgt: Decoder input embeddings [B, S, D]
            memory: Encoder output features [B, N_total, D]
            spatial_shapes: Spatial shape per level [L, 2]
            level_start_index: Start index per level [L]
            valid_ratios: Valid ratios for padding [B, L, 2]
            tgt_pos: Position embeddings [B, S, D] or [1, S, D]
            past_kv: Cached K, V from previous steps (inference only)
            use_cache: Whether to return updated KV cache
            value_cache: Pre-computed cross-attention value maps per level

        Returns:
            tgt: Updated decoder features [B, S, D]
            new_kv: (only if use_cache=True) updated (k, v) cache
        """
        B, S, D = tgt.shape

        # Add position embedding for self-attention
        if tgt_pos is not None:
            q_input = tgt + tgt_pos
        else:
            q_input = tgt

        # 1. Self-Attention (with optional KV-cache for incremental decoding)
        qkv = self.self_attn_qkv(q_input)  # [B, S, 3*D]
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_head]
        q_sa, k_sa, v_sa = qkv[0], qkv[1], qkv[2]

        if past_kv is not None:
            # Incremental: append new K, V to cache
            k_sa = torch.cat([past_kv[0], k_sa], dim=2)  # [B, H, S_prev+S, D_head]
            v_sa = torch.cat([past_kv[1], v_sa], dim=2)

        new_kv = (k_sa, v_sa) if use_cache else None

        # is_causal only valid for full-sequence mode (no cache, S > 1)
        use_causal = past_kv is None and S > 1

        attn_out = F.scaled_dot_product_attention(
            q_sa, k_sa, v_sa,
            dropout_p=self.self_attn_dropout if self.training else 0.0,
            is_causal=use_causal,
        )  # [B, H, S, D_head]

        # Reshape back
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, -1)  # [B, S, D]
        tgt2 = self.self_attn_out(attn_out)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Content-Dependent Reference Points (key innovation!)
        reference_points = self._compute_reference_points(tgt, tgt_pos, B, S)

        # 3. FluxAttention (Cross-attention) with gain control
        tgt2 = self.cross_attn(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios,
            value_cache=value_cache,
        )
        tgt2 = self.ca_gamma * tgt2
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 4. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, new_kv
        return tgt

    def _compute_reference_points(
        self,
        tgt: torch.Tensor,      # [B, S, D]
        tgt_pos: Optional[torch.Tensor],  # [B, S, D] or [1, S, D]
        B: int,
        S: int,
    ) -> torch.Tensor:
        """Compute content-dependent reference points (per-head).

        This is the key innovation of FluxAttention:
        - time_base: Predicted from position embedding (sequential progression)
        - freq_base: Predicted from hidden state (content-dependent!)

        Per-head freq_prior (n_freq_groups > 1):
        - Each group of heads can focus on a different frequency register
        - Analogous to cochlear tonotopic organization
        - e.g. Group 0 -> bass, Group 1 -> tenor, Group 2 -> alto, Group 3 -> soprano
        - Enables simultaneous multi-stream tracking (superhuman!)

        Returns:
            reference_points: [B, S, H, L, 2] where 2 = (time, freq)
        """
        device = tgt.device
        H = self.n_heads

        # Time prior: from position embedding (shared across heads)
        if self.time_prior is not None and tgt_pos is not None:
            time_base = self.time_prior(tgt_pos).sigmoid()  # [1, S, 1] or [B, S, 1]
            # Expand to batch size if needed
            if time_base.shape[0] == 1 and B > 1:
                time_base = time_base.expand(B, -1, -1)
        else:
            # Default: linear progression through time
            time_base = torch.linspace(0, 1, S, device=device)
            time_base = time_base.view(1, S, 1).expand(B, -1, -1)

        # time_base: [B, S, 1] -> expand to [B, S, H, 1]
        time_base = time_base.unsqueeze(2).expand(-1, -1, H, -1)  # [B, S, H, 1]

        # Freq prior: from hidden state (CONTENT-DEPENDENT!)
        if self.freq_prior is not None:
            # n_freq_groups=1: [B, S, 1] (original shared behavior)
            # n_freq_groups=4: [B, S, 4] (per-group different freq regions)
            freq_groups = self.freq_prior(tgt).sigmoid()  # [B, S, n_freq_groups]

            # Expand groups to heads: repeat_interleave
            # e.g. groups [g0, g1, g2, g3] -> heads [g0, g0, g1, g1, g2, g2, g3, g3]
            freq_base = freq_groups.repeat_interleave(
                self.heads_per_group, dim=-1
            )  # [B, S, H]
            freq_base = freq_base.unsqueeze(-1)  # [B, S, H, 1]
        else:
            # Default: center of frequency axis
            freq_base = torch.full((B, S, H, 1), 0.5, device=device)

        # Combine into base reference: [B, S, H, 2]
        base_ref = torch.cat([time_base, freq_base], dim=-1)  # [B, S, H, 2]

        # Content-based refinement (+/-10%), shared across heads
        refine = self.reference_refine(tgt).tanh() * self.refine_range  # [B, S, 2]
        refine = refine.unsqueeze(2)  # [B, S, 1, 2] -> broadcast to H

        # Final reference point
        reference_points = (base_ref + refine).clamp(0, 1)  # [B, S, H, 2]

        # Expand to all levels (shared reference across levels)
        reference_points = reference_points.unsqueeze(3).expand(
            -1, -1, -1, self.n_levels, -1
        )  # [B, S, H, L, 2]

        return reference_points


class ClefDecoder(nn.Module):
    """Full decoder stack with FluxAttention layers."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        n_levels: int = 4,
        n_points_freq: int = 2,
        n_points_time: int = 2,
        freq_offset_scale: float = 1.0,
        time_offset_scale: float = 1.0,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_time_prior: bool = True,
        use_freq_prior: bool = True,
        n_freq_groups: int = 1,
        refine_range: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_levels=n_levels,
                n_points_freq=n_points_freq,
                n_points_time=n_points_time,
                freq_offset_scale=freq_offset_scale,
                time_offset_scale=time_offset_scale,
                ff_dim=ff_dim,
                dropout=dropout,
                use_time_prior=use_time_prior,
                use_freq_prior=use_freq_prior,
                n_freq_groups=n_freq_groups,
                refine_range=refine_range,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        tgt_pos: Optional[torch.Tensor] = None,
        past_kv_list: Optional[list] = None,
        use_cache: bool = False,
        value_cache_list: Optional[list] = None,
    ):
        """Forward through all decoder layers.

        Args:
            past_kv_list: List of (k_cache, v_cache) per layer (inference only)
            use_cache: Whether to return updated KV caches
            value_cache_list: List of cross-attn value caches per layer

        Returns:
            output: Decoder output [B, S, D]
            new_kv_list: (only if use_cache=True) list of (k, v) per layer
        """
        output = tgt
        new_kv_list = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kv_list[i] if past_kv_list else None
            layer_value_cache = value_cache_list[i] if value_cache_list else None

            if use_cache:
                output, new_kv = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    tgt_pos=tgt_pos,
                    past_kv=layer_past_kv,
                    use_cache=True,
                    value_cache=layer_value_cache,
                )
                new_kv_list.append(new_kv)
            else:
                output = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    tgt_pos=tgt_pos,
                )

        output = self.norm(output)

        if use_cache:
            return output, new_kv_list
        return output
