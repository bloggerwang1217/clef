"""
Clef Decoder
============

Hybrid decoder: Mamba2 + Self-Attention + Cross-Attention.

Layer pattern (clef_piano_base):
  L0: Perceiver  — pass-through (audio_latent disabled, reserved for future)
  L1: full_ca    — SA + Full MHA on S2+S3, bootstraps temporal grounding
  L2: mamba_only — sequential writing
  L3: window_ca  — SA + Window CA on S0+S1 (beat/pitch)
  L4: mamba_only — sequential writing
  L5: window_ca  — SA + Window CA on L0+L1 (onset/pitch precision)
  L6: mamba_only — sequential writing

Key design:
- L1 full attention on S2+S3: tgt implicitly encodes audio temporal position
  through content-matched features, providing window center for L3/L5.
- L2/L4/L6 Mamba: sequential writing, collapses chord notes to same time.
- WindowCrossAttention (L3, L5): dense window sampling guided by L1 CoM.
"""

from typing import List, Optional, Tuple, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import WindowCrossAttention


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    RoPE encodes position information by rotating Q/K vectors in 2D subspaces.
    Benefits over learned absolute position embeddings:
    - No max_seq_len limit (position is computed on-the-fly)
    - Better length generalization
    - O(1) memory (no stored embedding table)

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        """
        Args:
            head_dim: Dimension per attention head (must be even)
            base: Base for geometric frequency spacing (default 10000 from paper)
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base

        # Precompute inverse frequencies: 1 / (base^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Cache for cos/sin tables (expanded on demand)
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Expand cos/sin cache if needed."""
        # Check seq_len, device, and dtype to avoid stale cache issues in DDP
        if (seq_len <= self._cache_seq_len
            and self._cos_cache is not None
            and self._cos_cache.device == device
            and self._cos_cache.dtype == dtype):
            return

        # Compute position indices
        t = torch.arange(seq_len, device=device, dtype=dtype)
        # Outer product: [seq_len] x [head_dim/2] -> [seq_len, head_dim/2]
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=dtype))
        # Duplicate for (cos, sin) pairs: [seq_len, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()
        self._cache_seq_len = seq_len

    def forward(
        self,
        q: torch.Tensor,  # [B, H, S, D_head]
        k: torch.Tensor,  # [B, H, S, D_head]
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to Q and K.

        Args:
            q: Query tensor [B, H, S, D_head]
            k: Key tensor [B, H, S, D_head]
            position_offset: Starting position (for incremental decoding with KV-cache)

        Returns:
            q_rotated: Q with position encoding [B, H, S, D_head]
            k_rotated: K with position encoding [B, H, S, D_head]
        """
        B, H, S, D = q.shape
        total_len = position_offset + S

        # Expand cache if needed
        self._update_cache(total_len, q.device, q.dtype)

        # Slice cache for current positions
        cos = self._cos_cache[position_offset:total_len]  # [S, D]
        sin = self._sin_cache[position_offset:total_len]  # [S, D]

        # Broadcast to [1, 1, S, D] for batch and heads
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims: [x1, x2] -> [-x2, x1]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)



class PerceiverLayer(nn.Module):
    """Perceiver IO-style beat tracker: builds global audio_latent from S2+S3.

    Role: Encode-only — does NOT decode tokens.
    - Builds beat_latent [B, M, D] by compressing S2+S3 into M=512 latent slots
    - Passes tgt through unchanged
    - audio_latent is consumed by time_prior.global_ca in subsequent SA+CA layers

    S2 (0.64s): energy, pitch centroid, chord density
    S3 (1.28s): pitch register contour, phrase boundary

    Perceiver IO (Jaegle et al., 2021):
    - Encode: latent × S2 → latent × S3 (2 rounds, shared refine_ca)
    """

    def __init__(self, d_model=512, ff_dim=2048, dropout=0.1,
                 n_latents=512, n_refine_rounds=2,
                 n_heads=8,
                 **kwargs):  # absorbs n_levels, use_time_prior, etc. from base_kwargs
        super().__init__()
        # Parameters preserved for potential future re-activation.
        # Computation is currently disabled in forward() — subsequent layers do not
        # consume audio_latent. Re-enable when time_prior is wired back in.

    def forward(
        self,
        tgt: torch.Tensor,              # [B, S, D] — passed through unchanged
        memory: torch.Tensor,           # [B, N_total, D] — unused (audio_latent disabled)
        spatial_shapes: torch.Tensor,   # unused
        level_start_index: torch.Tensor,  # unused
        valid_ratios: torch.Tensor,     # unused
        tgt_pos=None,                   # unused
        past_state=None,                # unused
        use_cache: bool = False,
        value_cache=None,               # unused
        audio_latent=None,              # unused
        window_center=None,             # unused
    ):
        """tgt passes through unchanged.

        Audio latent computation is disabled — subsequent layers (full_ca, window_ca)
        do not consume audio_latent. Re-enable when time_prior is wired back in.
        """
        if use_cache:
            return tgt, None
        return tgt


class MambaOnlyLayer(nn.Module):
    """Mamba-only decoder layer: Mamba + FFN (no CA).

    Used for sequential writing without audio re-injection.
    Assumes audio context was already injected by earlier CA layer.

    Inherits nn.Module directly (not DecoderLayerBase) to avoid creating
    unused cross_attn, time_prior, freq_prior, reference_refine components
    (~3M params per layer that would never receive gradients, causing DDP hang).
    """

    def __init__(self, d_model=512, ff_dim=2048, dropout=0.1,
                 d_state=128, d_conv=4, expand=2, **kwargs):
        # kwargs absorbs unused base_kwargs (n_heads, n_levels, etc.)
        super().__init__()
        from mamba_ssm import Mamba2

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self._mamba_layer_idx_set = False
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        tgt_pos: Optional[torch.Tensor] = None,
        past_state=None,
        use_cache: bool = False,
        value_cache: Optional[List[torch.Tensor]] = None,
        audio_latent=None,   # Unused (no time_prior)
        window_center=None,  # Unused (no CA)
    ):
        """Forward pass: Mamba + FFN only (skip CA)."""
        B, S, D = tgt.shape

        if past_state is not None:
            inference_params = past_state[0]
        else:
            inference_params = None

        # Mamba (no position embedding - intrinsic via recurrence)
        tgt2 = self.mamba(tgt, inference_params=inference_params)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Skip CA

        # FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, (inference_params, None)
        return tgt


class SAFullCALayer(nn.Module):
    """SA + Full Cross-Attention decoder layer (no Deformable Attention).

    L1 role: globally attend to S2+S3 to bootstrap temporal grounding.
    After attending to S2+S3, tgt encodes audio temporal position through
    content-matched audio features, making L3's time_prior much easier to train.

    L2 Mamba then collapses chord notes (same onset) to the same time position
    and advances across onsets, providing temporally-enriched context to L3+.

    Architecture: SA (causal, RoPE) → Full MHA on active_ca_levels → FFN
    No time_prior, no reference points, no FluxAttention.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        active_ca_levels: Optional[List[int]] = None,
        n_levels: int = 6,  # total encoder levels (for memory slicing)
        **kwargs,  # absorbs unused base_kwargs (n_points_*, use_time_prior, etc.)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.sa_head_dim = d_model // n_heads
        self.active_ca_levels = active_ca_levels
        self.n_levels = n_levels
        self.self_attn_dropout = dropout

        # Causal Self-Attention with RoPE
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.rope = RotaryPositionalEmbedding(self.sa_head_dim)

        # Full CA on active_ca_levels (e.g., S2+S3 for global harmonic context)
        # Separate Q/KV projections so output uses F.sdpa (FlashAttention)
        # while CoM is computed with no_grad head-averaged attention.
        self.ca_q_proj  = nn.Linear(d_model, d_model)
        self.ca_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.ca_out_proj = nn.Linear(d_model, d_model)
        self.ca_attn_dropout = dropout
        self.ca_head_dim = d_model // n_heads
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        tgt_pos=None,
        past_state=None,
        use_cache: bool = False,
        value_cache=None,      # ignored (no FluxCA)
        audio_latent=None,     # ignored (no time_prior in this layer)
        window_center=None,    # ignored (this layer produces window_center, not consumes)
    ):
        """SA + Full MHA on active encoder levels + FFN."""
        B, S, D = tgt.shape
        past_kv = past_state[0] if past_state is not None else None

        # 1. Causal Self-Attention with RoPE
        qkv = self.self_attn_qkv(tgt)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.sa_head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE position offset
        if isinstance(tgt_pos, int):
            position_offset = tgt_pos
        elif past_kv is not None:
            position_offset = past_kv[0].shape[2]
        else:
            position_offset = 0

        q, k = self.rope(q, k, position_offset=position_offset)

        # Concatenate with KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None
        use_causal = past_kv is None and S > 1

        sa_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.self_attn_dropout if self.training else 0.0,
            is_causal=use_causal,
        )
        sa_out = sa_out.permute(0, 2, 1, 3).reshape(B, S, -1)
        tgt2 = self.self_attn_out(sa_out)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Full MHA on active encoder levels (e.g., S2+S3)
        n_enc_levels = level_start_index.shape[0]
        if self.active_ca_levels is not None and len(self.active_ca_levels) > 0:
            slices = []
            for lvl in self.active_ca_levels:
                start = level_start_index[lvl].item()
                end = (level_start_index[lvl + 1].item()
                       if lvl + 1 < n_enc_levels else memory.shape[1])
                slices.append(memory[:, start:end, :])
            memory_ca = torch.cat(slices, dim=1)  # [B, N_active, D]
        else:
            memory_ca = memory

        # Full CA output via F.sdpa (FlashAttention path — no attention weights returned).
        N_kv = memory_ca.shape[1]
        q_ca = self.ca_q_proj(tgt).reshape(B, S, self.n_heads, self.ca_head_dim).permute(0, 2, 1, 3)
        kv = self.ca_kv_proj(memory_ca).reshape(B, N_kv, 2, self.n_heads, self.ca_head_dim)
        k_ca = kv[:, :, 0].permute(0, 2, 1, 3)   # [B, H, N_kv, D_head]
        v_ca = kv[:, :, 1].permute(0, 2, 1, 3)
        ca_sdpa = F.scaled_dot_product_attention(
            q_ca, k_ca, v_ca,
            dropout_p=self.ca_attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        tgt2 = self.ca_out_proj(ca_sdpa.permute(0, 2, 1, 3).reshape(B, S, D))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Compute attention weights for CoM (no_grad — CoM gradient path is broken by
        # floor().long() in gather; main gradient comes through the output path above).
        with torch.no_grad():
            q_avg = q_ca.mean(dim=1)                            # [B, S, D_head]
            k_avg = k_ca.mean(dim=1)                            # [B, N_kv, D_head]
            scores_avg = torch.bmm(q_avg, k_avg.transpose(1, 2)) / math.sqrt(self.ca_head_dim)
            attn_weights = scores_avg.softmax(dim=-1)           # [B, S, N_kv]

        # 3. Compute window center = attention center-of-mass over encoder positions.
        # For each active level, each token h*W_l + w has:
        #   time_coord = w / (W_l - 1),  freq_coord = h / (H_l - 1)  (both in [0, 1])
        # Weighted average over all N_active positions gives (time_center, freq_center).
        if self.active_ca_levels is not None and attn_weights is not None:
            time_coords_list = []
            freq_coords_list = []
            for lvl in self.active_ca_levels:
                H_l, W_l = spatial_shapes[lvl].tolist()
                w_range = torch.arange(W_l, device=tgt.device, dtype=tgt.dtype)
                h_range = torch.arange(H_l, device=tgt.device, dtype=tgt.dtype)
                w_norm = w_range / max(W_l - 1, 1)  # [W_l] in [0, 1]
                h_norm = h_range / max(H_l - 1, 1)  # [H_l] in [0, 1]
                # H_l x W_l grid, then flatten
                w_grid = w_norm.unsqueeze(0).expand(H_l, -1).reshape(-1)  # [H_l*W_l]
                h_grid = h_norm.unsqueeze(1).expand(-1, W_l).reshape(-1)  # [H_l*W_l]
                time_coords_list.append(w_grid)
                freq_coords_list.append(h_grid)
            time_coords = torch.cat(time_coords_list)   # [N_active]
            freq_coords = torch.cat(freq_coords_list)   # [N_active]
            # attn_weights: [B, S, N_active] — weighted average of positions
            time_center = (attn_weights * time_coords).sum(-1, keepdim=True)  # [B, S, 1]
            freq_center = (attn_weights * freq_coords).sum(-1, keepdim=True)  # [B, S, 1]
        else:
            # Fallback: proportional linspace
            positions = torch.arange(S, device=tgt.device, dtype=tgt.dtype)
            time_center = (positions / max(S - 1, 1)).view(1, S, 1).expand(B, -1, -1)
            freq_center = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)
        self._cached_window_center = (time_center, freq_center)

        # 4. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, (new_kv, None)
        return tgt


class SAWindowCALayer(nn.Module):
    """SA + Window Cross-Attention decoder layer.

    Replaces FluxAttention (deformable sparse sampling) with dense window attention.
    Window center (time, freq) is the center-of-mass of SAFullCALayer's attention
    weights over S2+S3 — a free byproduct requiring no additional parameters.

    Benefits over SADecoderLayer (FluxAttention):
    - Robust: center error < window_half is fully tolerated
    - Dense: all window tokens contribute gradient (not 4 sparse points)
    - No time_prior/freq_prior needed (center is free from L1)
    - Standard QK attention weights (not factored level/point weights)

    Architecture: SA (causal, RoPE) → WindowCrossAttention → FFN
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        active_ca_levels: Optional[List[int]] = None,
        n_levels: int = 6,
        window_time_frames: Union[int, List[int]] = 16,
        window_freq_bins: Union[int, List[int]] = 8,
        window_seq_chunk_size: int = 128,
        **kwargs,  # absorbs unused base_kwargs (n_points_*, use_time_prior, etc.)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.sa_head_dim = d_model // n_heads
        self.active_ca_levels = active_ca_levels
        self.n_levels = n_levels
        self.self_attn_dropout = dropout

        # Causal Self-Attention with RoPE (identical to SAFullCALayer)
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.rope = RotaryPositionalEmbedding(self.sa_head_dim)

        # Window Cross-Attention (replaces FluxAttention)
        self.window_ca = WindowCrossAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            window_time=window_time_frames,
            window_freq=window_freq_bins,
            seq_chunk_size=window_seq_chunk_size,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        tgt_pos=None,
        past_state=None,
        use_cache: bool = False,
        value_cache=None,     # kv_cache from WindowCrossAttention.compute_kv_cache
        audio_latent=None,    # unused
        window_center=None,   # (time_center [B,S,1], freq_center [B,S,1]) from L1
    ):
        """SA + Window CA + FFN.

        window_center is (time_center, freq_center) from SAFullCALayer's
        attention weight center-of-mass. Falls back to proportional linspace
        if not provided (e.g., first few warmup steps).
        """
        B, S, D = tgt.shape
        past_kv = past_state[0] if past_state is not None else None

        # 1. Causal Self-Attention with RoPE
        qkv = self.self_attn_qkv(tgt)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.sa_head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if isinstance(tgt_pos, int):
            position_offset = tgt_pos
        elif past_kv is not None:
            position_offset = past_kv[0].shape[2]
        else:
            position_offset = 0

        q, k = self.rope(q, k, position_offset=position_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None
        use_causal = past_kv is None and S > 1

        sa_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.self_attn_dropout if self.training else 0.0,
            is_causal=use_causal,
        )
        sa_out = sa_out.permute(0, 2, 1, 3).reshape(B, S, -1)
        tgt = tgt + self.dropout1(self.self_attn_out(sa_out))
        tgt = self.norm1(tgt)

        # 2. Window Cross-Attention
        if window_center is not None:
            time_center, freq_center = window_center
        else:
            # Fallback: proportional linspace (no L1 byproduct available)
            positions = torch.arange(S, device=tgt.device, dtype=tgt.dtype)
            time_center = (positions / max(S - 1, 1)).view(1, S, 1).expand(B, -1, -1)
            freq_center = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)

        tgt2 = self.window_ca(
            query=tgt,
            time_center=time_center,
            freq_center=freq_center,
            value=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            active_levels=self.active_ca_levels,
            kv_cache=value_cache,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, (new_kv, None)
        return tgt


class ClefDecoder(nn.Module):
    """Hybrid decoder: Mamba2 + Self-Attention + Cross-Attention.

    Layer pattern (clef_piano_base):
      L0: Perceiver  — pass-through (audio_latent disabled, reserved for future)
      L1: full_ca    — SA + Full MHA on S2+S3, bootstraps temporal grounding
      L2: mamba_only — sequential writing
      L3: window_ca  — SA + Window CA on S0+S1 (beat/pitch)
      L4: mamba_only — sequential writing
      L5: window_ca  — SA + Window CA on L0+L1 (onset/pitch precision)
      L6: mamba_only — sequential writing
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_levels: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        # Window cross-attention (for 'window_ca' layer type)
        window_time_frames: Union[int, List[int]] = 16,
        window_freq_bins: Union[int, List[int]] = 8,
        window_seq_chunk_size: int = 128,
        decoder_layer_types: Optional[List[str]] = None,
        decoder_layer_ca_levels: Optional[List] = None,  # per-layer active CA level list
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        use_rope: bool = True,  # RoPE for SA layers
        n_layers: int = 6,     # legacy fallback (ignored if decoder_layer_types is set)
        **kwargs,              # absorb deprecated FluxAttention params (n_points_*, *_scale, etc.)
    ):
        super().__init__()

        if decoder_layer_types is None:
            decoder_layer_types = ['perceiver', 'full_ca', 'mamba_only',
                                   'window_ca', 'mamba_only', 'window_ca', 'mamba_only']

        self.layer_types = decoder_layer_types
        self.use_rope = use_rope

        # Shared kwargs passed to all layer constructors (unused keys absorbed by **kwargs)
        base_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            ff_dim=ff_dim,
            dropout=dropout,
            rope_base=rope_base,
        )

        self.layers = nn.ModuleList()
        for i, lt in enumerate(decoder_layer_types):
            ca_levels = decoder_layer_ca_levels[i] if decoder_layer_ca_levels else None
            if lt == 'perceiver':
                self.layers.append(PerceiverLayer(**base_kwargs))
            elif lt == 'mamba_only':
                self.layers.append(MambaOnlyLayer(
                    d_state=d_state, d_conv=d_conv, expand=expand,
                    **base_kwargs,
                ))
            elif lt == 'full_ca':
                self.layers.append(SAFullCALayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    active_ca_levels=ca_levels,
                    n_levels=n_levels,
                ))
            elif lt == 'window_ca':
                self.layers.append(SAWindowCALayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    active_ca_levels=ca_levels,
                    n_levels=n_levels,
                    window_time_frames=window_time_frames,
                    window_freq_bins=window_freq_bins,
                    window_seq_chunk_size=window_seq_chunk_size,
                ))
            else:
                raise ValueError(f"Unknown decoder layer type: {lt!r}")

        # Assign layer_idx to Mamba layers (needed for InferenceParams state indexing)
        mamba_idx = 0
        for layer in self.layers:
            if isinstance(layer, MambaOnlyLayer):
                layer.mamba.layer_idx = mamba_idx
                mamba_idx += 1

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        tgt_pos: Optional[torch.Tensor] = None,
        past_states: Optional[list] = None,
        use_cache: bool = False,
        value_cache_list: Optional[list] = None,
    ):
        """Forward through all decoder layers.

        Args:
            past_states: List of per-layer state tuples (layer_state, time_prior_params).
                layer_state: KV tuple for SA, InferenceParams for Mamba
            use_cache: Whether to return updated states
            value_cache_list: List of cross-attn value caches per layer

        Returns:
            Without use_cache: output [B, S, D]
            With use_cache: (output [B, S, D], new_states list)
        """
        output = tgt
        new_states = [] if use_cache else None

        # audio_latent: disabled — PerceiverLayer no longer computes it,
        # and subsequent layers (full_ca, window_ca) do not consume it.
        # Re-enable when time_prior is wired back in.

        # window_center: (time_center [B,S,1], freq_center [B,S,1]) from SAFullCALayer (L1).
        # Center-of-mass of L1's attention weights over S2+S3 encoder positions.
        # Passed to SAWindowCALayer (L3, L5) as window center — no time_prior needed.
        window_center = None

        for i, layer in enumerate(self.layers):
            layer_state = past_states[i] if past_states else None
            layer_value_cache = value_cache_list[i] if value_cache_list else None

            if use_cache:
                output, new_state = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    tgt_pos=tgt_pos,
                    past_state=layer_state,
                    use_cache=True,
                    value_cache=layer_value_cache,
                    window_center=window_center,
                )
                new_states.append(new_state)

                if isinstance(layer, SAFullCALayer):
                    window_center = layer._cached_window_center
            else:
                output = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    tgt_pos=tgt_pos,
                    window_center=window_center,
                )

                if isinstance(layer, SAFullCALayer):
                    window_center = layer._cached_window_center

        output = self.norm(output)

        if use_cache:
            return output, new_states
        return output
