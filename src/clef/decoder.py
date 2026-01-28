"""
Deformable Decoder
==================

Transformer decoder with ClefAttention cross-attention.

Key innovation: Content-Dependent Reference Points
- time_prior: Predict "which time point to look at" from positional embedding
- freq_prior: Predict "high or low frequency region" from hidden state
- This corresponds to stream tracking in human auditory perception (Bregman, 1990)

Piano application:
- When predicting right-hand melody -> freq_prior outputs high value
- When predicting left-hand harmony -> freq_prior outputs low value
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ClefAttention


class DeformableDecoderLayer(nn.Module):
    """Decoder Layer with Content-Aware Deformable Cross-Attention.

    Structure:
    1. Causal Self-Attention (standard)
    2. ClefAttention with content-dependent reference points
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
        freq_offset_scale: float = 0.15,
        time_offset_scale: float = 0.15,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_time_prior: bool = True,
        use_freq_prior: bool = True,
        refine_range: float = 0.1,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.use_time_prior = use_time_prior
        self.use_freq_prior = use_freq_prior
        self.refine_range = refine_range

        # 1. Causal Self-Attention (use F.scaled_dot_product_attention for Flash Attention)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.self_attn_dropout = dropout
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. ClefAttention (Cross-attention)
        self.cross_attn = ClefAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points_freq=n_points_freq,
            n_points_time=n_points_time,
            freq_offset_scale=freq_offset_scale,
            time_offset_scale=time_offset_scale,
        )
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
            self.freq_prior = nn.Linear(d_model, 1)
        else:
            self.freq_prior = None

        # Content-based refinement (+/-10%)
        self.reference_refine = nn.Linear(d_model, 2)

        self._init_reference_predictors()

    def _init_reference_predictors(self):
        """Initialize reference point predictors.

        IMPORTANT: Cannot use zero weights!
        - Zero weights -> constant output -> zero gradients -> no learning
        - Use small Xavier init for weights (enables gradient flow)
        - Use zero bias (centers output at sigmoid(0)=0.5 or tanh(0)=0)
        """
        if self.time_prior is not None:
            # Small Xavier init - prevents gradient explosion through grid_sample
            nn.init.xavier_uniform_(self.time_prior.weight, gain=0.001)
            nn.init.constant_(self.time_prior.bias, 0.)  # sigmoid(0) = 0.5

        if self.freq_prior is not None:
            # Small Xavier init - prevents gradient explosion through grid_sample
            nn.init.xavier_uniform_(self.freq_prior.weight, gain=0.001)
            nn.init.constant_(self.freq_prior.bias, 0.)  # sigmoid(0) = 0.5

        # Refinement: very small init to prevent gradient explosion through grid_sample
        # grid_sample gradients w.r.t. sampling locations can be very large
        nn.init.normal_(self.reference_refine.weight, std=0.001)
        nn.init.constant_(self.reference_refine.bias, 0.)  # tanh(0) = 0

    def forward(
        self,
        tgt: torch.Tensor,           # [B, S, D]
        memory: torch.Tensor,        # [B, N_total, D]
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,  # [B, L, 2]
        tgt_pos: Optional[torch.Tensor] = None,  # [B, S, D] or [1, S, D]
    ) -> torch.Tensor:
        """Forward pass with content-dependent reference points.

        Uses is_causal=True for Flash Attention (PyTorch 2.0+).

        Args:
            tgt: Decoder input embeddings [B, S, D]
            memory: Encoder output features [B, N_total, D]
            spatial_shapes: Spatial shape per level [L, 2]
            level_start_index: Start index per level [L]
            valid_ratios: Valid ratios for padding [B, L, 2]
            tgt_pos: Position embeddings [B, S, D] or [1, S, D]

        Returns:
            Updated decoder features [B, S, D]
        """
        B, S, D = tgt.shape

        # Add position embedding for self-attention
        if tgt_pos is not None:
            q = k = tgt + tgt_pos
        else:
            q = k = tgt

        # 1. Causal Self-Attention with Flash Attention (PyTorch 2.0+)
        # Compute Q, K, V
        qkv = self.self_attn_qkv(q)  # [B, S, 3*D]
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_head]
        q_sa, k_sa, v_sa = qkv[0], qkv[1], qkv[2]

        # Flash Attention with is_causal=True (O(S) memory instead of O(S²))
        attn_out = F.scaled_dot_product_attention(
            q_sa, k_sa, v_sa,
            dropout_p=self.self_attn_dropout if self.training else 0.0,
            is_causal=True,
        )  # [B, H, S, D_head]

        # Reshape back
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, -1)  # [B, S, D]
        tgt2 = self.self_attn_out(attn_out)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Content-Dependent Reference Points (key innovation!)
        reference_points = self._compute_reference_points(tgt, tgt_pos, B, S)

        # 3. ClefAttention (Cross-attention)
        tgt2 = self.cross_attn(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 4. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        return tgt

    def _compute_reference_points(
        self,
        tgt: torch.Tensor,      # [B, S, D]
        tgt_pos: Optional[torch.Tensor],  # [B, S, D] or [1, S, D]
        B: int,
        S: int,
    ) -> torch.Tensor:
        """Compute content-dependent reference points.

        This is the key innovation of ClefAttention:
        - time_base: Predicted from position embedding (sequential progression)
        - freq_base: Predicted from hidden state (content-dependent!)

        Human analogy:
        - Position tells us "roughly when" in the piece we are
        - Content tells us "which voice/register" we're focusing on

        Returns:
            reference_points: [B, S, L, 2] where 2 = (time, freq)
        """
        device = tgt.device

        # Time prior: from position embedding
        if self.time_prior is not None and tgt_pos is not None:
            time_base = self.time_prior(tgt_pos).sigmoid()  # [1, S, 1] or [B, S, 1]
            # Expand to batch size if needed
            if time_base.shape[0] == 1 and B > 1:
                time_base = time_base.expand(B, -1, -1)
        else:
            # Default: linear progression through time
            time_base = torch.linspace(0, 1, S, device=device)
            time_base = time_base.view(1, S, 1).expand(B, -1, -1)

        # Freq prior: from hidden state (CONTENT-DEPENDENT!)
        if self.freq_prior is not None:
            freq_base = self.freq_prior(tgt).sigmoid()  # [B, S, 1]
        else:
            # Default: center of frequency axis
            freq_base = torch.full((B, S, 1), 0.5, device=device)

        # Combine into base reference
        base_ref = torch.cat([time_base, freq_base], dim=-1)  # [B, S, 2]

        # Content-based refinement (+/-10%)
        refine = self.reference_refine(tgt).tanh() * self.refine_range  # [B, S, 2]

        # Final reference point
        reference_points = (base_ref + refine).clamp(0, 1)  # [B, S, 2]

        # Expand to all levels (shared reference across levels)
        reference_points = reference_points[:, :, None, :].expand(
            -1, -1, self.n_levels, -1
        )  # [B, S, L, 2]

        return reference_points


class ClefDecoder(nn.Module):
    """Full decoder stack with ClefAttention layers."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        n_levels: int = 4,
        n_points_freq: int = 2,
        n_points_time: int = 2,
        freq_offset_scale: float = 0.15,
        time_offset_scale: float = 0.15,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_time_prior: bool = True,
        use_freq_prior: bool = True,
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
    ) -> torch.Tensor:
        """Forward through all decoder layers.

        Uses is_causal=True for Flash Attention (PyTorch 2.0+).
        No explicit causal mask needed.
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output, memory,
                spatial_shapes, level_start_index, valid_ratios,
                tgt_pos=tgt_pos,
            )

        output = self.norm(output)
        return output
