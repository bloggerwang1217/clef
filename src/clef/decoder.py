"""
Deformable Decoder
==================

Jamba-style hybrid decoder: Mamba2 + Sparse Self-Attention.

Layer pattern (default): [Mamba, Mamba, SA, Mamba, Mamba, SA]
- Mamba layers: sequential writing, beat counting via linear recurrence
- SA layers: constraint verification, retrieval, copy detection

Shared across all layers:
- FluxAttention cross-attention with content-dependent reference points
- FFN

Key innovation: Content-Dependent Reference Points
- time_prior: sinusoidal PE -> Linear -> sigmoid.
  Provides absolute time position in [0, 1] for Deformable Attention.
  Sinusoidal PE gives stable position signal; Linear learns average
  position-to-time mapping. reference_refine handles per-sample variation.
- freq_prior: Predict "high or low frequency region" from hidden state.
  Pitch -> frequency is a universal constant -> learns well.
- Corresponds to stream tracking in human auditory perception (Bregman, 1990)
"""

from typing import List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FluxAttention


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


class DecoderLayerBase(nn.Module):
    """Base decoder layer: FluxCA + FFN + reference points.

    Subclasses implement _sequence_forward() for SA or Mamba.

    Structure:
    1. Sequence model (SA or Mamba) — subclass-defined
    2. FluxAttention with content-dependent reference points
    3. FFN
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

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.use_time_prior = use_time_prior
        self.use_freq_prior = use_freq_prior
        self.n_freq_groups = n_freq_groups
        self.refine_range = refine_range

        # Post-sequence-model norm
        self.norm1 = nn.LayerNorm(d_model)

        # FluxAttention (Cross-attention)
        self.cross_attn = FluxAttention(
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

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Content-Dependent Reference Points
        # Time prior: sinusoidal PE → Linear → sigmoid
        # Sinusoidal PE provides stable position signal (not learned, like RoPE).
        # Linear learns average position-to-time mapping across training data.
        # reference_refine (±10%) handles per-sample variation.
        self.use_time_prior = use_time_prior
        if use_time_prior:
            pe_dim = 64  # small sinusoidal encoding for position
            self.time_pe_dim = pe_dim
            self.time_prior = nn.Linear(pe_dim, 1)
        else:
            self.time_prior = None

        if use_freq_prior:
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

        time_prior: Sinusoidal PE → Linear → sigmoid.
        Initialized so that sigmoid output ≈ linspace(0, 1, S).
        Linear weight is zero-init, bias=0 → sigmoid(0)=0.5 as starting point.
        The sinusoidal PE provides position signal; Linear learns the mapping.

        freq_prior: Spread bias across frequency axis so each head group
        starts at a different register (bass/tenor/alto/soprano).

        reference_refine: ±10% content-dependent adjustment.
        """
        if self.time_prior is not None:
            nn.init.xavier_uniform_(self.time_prior.weight, gain=0.1)
            nn.init.constant_(self.time_prior.bias, 0.0)

        if self.freq_prior is not None:
            nn.init.xavier_uniform_(self.freq_prior.weight, gain=0.1)

            import math
            n_groups = self.freq_prior.bias.shape[0]
            with torch.no_grad():
                for g in range(n_groups):
                    target = (g + 1) / (n_groups + 1)
                    self.freq_prior.bias[g] = math.log(target / (1 - target))

        nn.init.normal_(self.reference_refine.weight, std=0.01)
        nn.init.constant_(self.reference_refine.bias, 0.)

    def _sequence_forward(self, tgt, tgt_pos, past_state, use_cache):
        """Override in subclass. Returns (output, new_state)."""
        raise NotImplementedError

    def forward(
        self,
        tgt: torch.Tensor,           # [B, S, D]
        memory: torch.Tensor,        # [B, N_total, D]
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,  # [B, L, 2]
        tgt_pos: Optional[torch.Tensor] = None,  # [B, S, D] or [1, S, D]
        past_state=None,
        use_cache: bool = False,
        value_cache: Optional[List[torch.Tensor]] = None,
    ):
        """Forward pass with content-dependent reference points.

        Args:
            tgt: Decoder input embeddings [B, S, D]
            memory: Encoder output features [B, N_total, D]
            spatial_shapes: Spatial shape per level [L, 2]
            level_start_index: Start index per level [L]
            valid_ratios: Valid ratios for padding [B, L, 2]
            tgt_pos: Position embeddings [B, S, D] or [1, S, D]
            past_state: Layer-specific state (KV tuple for SA, InferenceParams for Mamba)
            use_cache: Whether to return updated state
            value_cache: Pre-computed cross-attention value maps per level

        Returns:
            tgt: Updated decoder features [B, S, D]
            new_state: (only if use_cache=True) updated state
        """
        B, S, D = tgt.shape

        # 1. Sequence model (SA or Mamba)
        tgt, new_state = self._sequence_forward(tgt, tgt_pos, past_state, use_cache)
        tgt = self.norm1(tgt)

        # 2. Content-Dependent Reference Points
        reference_points = self._compute_reference_points(tgt, tgt_pos, B, S)

        # 3. FluxAttention (Cross-attention)
        tgt2 = self.cross_attn(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios,
            value_cache=value_cache,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 4. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, new_state
        return tgt

    @staticmethod
    def _sinusoidal_pe(positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding.

        Args:
            positions: [S] or [B, S] position indices (int or float)
            dim: encoding dimension (must be even)

        Returns:
            pe: [..., dim] sinusoidal encoding
        """
        half = dim // 2
        freq = torch.exp(
            -torch.arange(half, device=positions.device, dtype=torch.float32)
            * (math.log(10000.0) / half)
        )  # [half]
        # positions: [...] → [..., 1] * [half] → [..., half]
        angles = positions.unsqueeze(-1).float() * freq
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [..., dim]

    def _compute_reference_points(
        self,
        tgt: torch.Tensor,      # [B, S, D]
        tgt_pos,  # [B, S, D], [1, S, D], or int (RoPE offset)
        B: int,
        S: int,
    ) -> torch.Tensor:
        """Compute content-dependent reference points (per-head).

        Time coordinate: sinusoidal PE → Linear → sigmoid.
        Each token independently predicts an absolute time position in [0, 1].
        Sinusoidal PE provides stable position signal; Linear learns the
        average position-to-time mapping. Train/inference identical.

        For inference (step-by-step), tgt_pos is an int offset, so
        positions = [offset, offset+1, ..., offset+S-1].

        Returns:
            reference_points: [B, S, H, L, 2] where 2 = (time, freq)
        """
        device = tgt.device
        dtype = tgt.dtype
        H = self.n_heads

        # Time prior: sinusoidal PE → Linear → sigmoid
        if self.time_prior is not None:
            # Build position indices
            if isinstance(tgt_pos, int):
                # RoPE mode: tgt_pos is offset (int)
                positions = torch.arange(tgt_pos, tgt_pos + S, device=device)
            else:
                # Legacy mode: use sequential indices
                positions = torch.arange(S, device=device)
            pe = self._sinusoidal_pe(positions, self.time_pe_dim)  # [S, pe_dim]
            time_base = self.time_prior(pe).sigmoid()  # [S, 1]
            time_base = time_base.unsqueeze(0).expand(B, -1, -1)  # [B, S, 1]
        else:
            # Fallback: linear interpolation
            time_base = torch.linspace(0, 1, S, device=device, dtype=dtype)
            time_base = time_base.view(1, S, 1).expand(B, -1, -1)

        time_base = time_base.unsqueeze(2).expand(-1, -1, H, -1)  # [B, S, H, 1]

        # Freq prior: from hidden state (CONTENT-DEPENDENT!)
        if self.freq_prior is not None:
            freq_groups = self.freq_prior(tgt).sigmoid()  # [B, S, n_freq_groups]
            freq_base = freq_groups.repeat_interleave(
                self.heads_per_group, dim=-1
            )  # [B, S, H]
            freq_base = freq_base.unsqueeze(-1)  # [B, S, H, 1]
        else:
            freq_base = torch.full((B, S, H, 1), 0.5, device=device)

        base_ref = torch.cat([time_base, freq_base], dim=-1)  # [B, S, H, 2]

        # Content-based refinement (+/-10%), shared across heads
        refine = self.reference_refine(tgt.detach()).tanh() * self.refine_range
        refine = refine.unsqueeze(2)  # [B, S, 1, 2] -> broadcast to H

        reference_points = (base_ref + refine).clamp(0, 1)  # [B, S, H, 2]

        # Expand to all levels
        reference_points = reference_points.unsqueeze(3).expand(
            -1, -1, -1, self.n_levels, -1
        )  # [B, S, H, L, 2]

        return reference_points


class SADecoderLayer(DecoderLayerBase):
    """Self-Attention decoder layer with RoPE.

    Causal self-attention for constraint verification and retrieval:
    - Look back at all tokens in the bar to verify beat sum
    - Detect repeated sections (copy-paste patterns)
    - Global attention over full sequence (no recurrence window limit)

    Uses Rotary Position Embedding (RoPE) instead of additive position embedding.
    """

    def __init__(self, use_rope: bool = True, **kwargs):
        super().__init__(**kwargs)
        d_model = kwargs['d_model']
        n_heads = kwargs['n_heads']
        dropout = kwargs['dropout']

        self.sa_n_heads = n_heads
        self.sa_head_dim = d_model // n_heads
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.self_attn_dropout = dropout
        self.dropout1 = nn.Dropout(dropout)

        # RoPE for position encoding (no max_seq_len limit)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.sa_head_dim)

    def _sequence_forward(self, tgt, tgt_pos, past_kv, use_cache):
        """Causal self-attention with RoPE and optional KV-cache.

        Args:
            tgt: Input tensor [B, S, D]
            tgt_pos: Position info - either embedding [B, S, D] or offset int (for RoPE)
            past_kv: Cached (K, V) from previous steps
            use_cache: Whether to return updated KV cache
        """
        B, S, D = tgt.shape

        # Compute Q, K, V (without adding position embedding - RoPE does rotation)
        if self.use_rope:
            qkv = self.self_attn_qkv(tgt)  # [B, S, 3*D]
        else:
            # Legacy: additive position embedding
            q_input = tgt + tgt_pos if tgt_pos is not None else tgt
            qkv = self.self_attn_qkv(q_input)

        qkv = qkv.reshape(B, S, 3, self.sa_n_heads, self.sa_head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_head]
        q_sa, k_sa, v_sa = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K
        if self.use_rope:
            # Get position offset from tgt_pos (int) or past_kv length
            if isinstance(tgt_pos, int):
                position_offset = tgt_pos
            elif past_kv is not None:
                position_offset = past_kv[0].shape[2]  # K cache length
            else:
                position_offset = 0

            q_sa, k_sa = self.rope(q_sa, k_sa, position_offset=position_offset)

        # Concatenate with cached K, V
        if past_kv is not None:
            k_sa = torch.cat([past_kv[0], k_sa], dim=2)
            v_sa = torch.cat([past_kv[1], v_sa], dim=2)

        new_kv = (k_sa, v_sa) if use_cache else None

        # is_causal only valid for full-sequence mode (no cache, S > 1)
        use_causal = past_kv is None and S > 1

        attn_out = F.scaled_dot_product_attention(
            q_sa, k_sa, v_sa,
            dropout_p=self.self_attn_dropout if self.training else 0.0,
            is_causal=use_causal,
        )  # [B, H, S, D_head]

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, -1)
        tgt2 = self.self_attn_out(attn_out)
        tgt = tgt + self.dropout1(tgt2)
        return tgt, new_kv


class MambaDecoderLayer(DecoderLayerBase):
    """Mamba2 decoder layer.

    Linear recurrence for sequential writing and beat counting:
    - h = A*h + B*x: A = exp(-Delta), Delta -> 0 means A -> 1 (exact hold)
    - Barline resets state (Delta large -> A -> 0)
    - Duration tokens decrement beat counter via B*x
    - Non-duration tokens pass through (B*x -> 0)
    - O(N log N) parallel scan during training, O(1) per step at inference
    """

    def __init__(self, d_state=128, d_conv=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        from mamba_ssm import Mamba2

        # layer_idx is set later by ClefDecoder after all layers are created.
        # It's needed for InferenceParams to index per-layer state at inference.
        self.mamba = Mamba2(
            d_model=kwargs['d_model'],
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self._mamba_layer_idx_set = False
        self.dropout1 = nn.Dropout(kwargs['dropout'])

    def _sequence_forward(self, tgt, tgt_pos, inference_params, use_cache):
        """Mamba2 forward. No position embedding needed (intrinsic via recurrence).

        tgt_pos is NOT added to input (position is encoded in recurrence state).
        tgt_pos is still used by _compute_reference_points in base class.

        inference_params is a shared InferenceParams object across all Mamba layers;
        mamba-ssm manages per-layer state internally via dict keyed by layer id.
        """
        tgt2 = self.mamba(tgt, inference_params=inference_params)
        tgt = tgt + self.dropout1(tgt2)
        # Return the same shared inference_params object
        return tgt, inference_params


# Keep backward compat alias
DeformableDecoderLayer = SADecoderLayer


class ClefDecoder(nn.Module):
    """Jamba-style hybrid decoder: Mamba2 + Sparse Self-Attention.

    Default layer pattern: [Mamba, Mamba, SA, Mamba, Mamba, SA]
    - 4 Mamba layers: sequential writing, beat counting
    - 2 SA layers: constraint verification, retrieval
    - All layers share: FluxCA + FFN + reference points
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
        # Jamba-specific
        decoder_layer_types: Optional[List[str]] = None,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        # Position encoding
        use_rope: bool = True,  # RoPE for SA layers (no max_seq_len limit)
        # Legacy (ignored if decoder_layer_types is set)
        n_layers: int = 6,
    ):
        super().__init__()

        if decoder_layer_types is None:
            # Legacy: all SA layers (backward compat with old checkpoints)
            decoder_layer_types = ['sa'] * n_layers

        self.layer_types = decoder_layer_types
        self.use_rope = use_rope

        # Shared kwargs for base class
        base_kwargs = dict(
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

        self.layers = nn.ModuleList()
        for lt in decoder_layer_types:
            if lt == 'mamba':
                self.layers.append(MambaDecoderLayer(
                    d_state=d_state, d_conv=d_conv, expand=expand,
                    **base_kwargs,
                ))
            elif lt == 'sa':
                self.layers.append(SADecoderLayer(use_rope=use_rope, **base_kwargs))
            else:
                raise ValueError(f"Unknown decoder layer type: {lt}")

        # Assign layer_idx to Mamba layers (needed for InferenceParams state indexing)
        mamba_idx = 0
        for layer in self.layers:
            if isinstance(layer, MambaDecoderLayer):
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
            past_states: List of per-layer state (KV tuple for SA, InferenceParams for Mamba)
            use_cache: Whether to return updated states
            value_cache_list: List of cross-attn value caches per layer

        Returns:
            output: Decoder output [B, S, D]
            new_states: (only if use_cache=True) list of states per layer
        """
        output = tgt
        new_states = [] if use_cache else None

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
                )
                new_states.append(new_state)
            else:
                output = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    tgt_pos=tgt_pos,
                )

        output = self.norm(output)

        if use_cache:
            return output, new_states
        return output
