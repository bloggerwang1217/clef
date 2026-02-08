"""
Deformable Decoder
==================

Jamba-style hybrid decoder: Mamba2 + Sparse Self-Attention.

Layer pattern (default): [Mamba, Mamba, SA, Mamba, Mamba, SA]
- Mamba layers: sequential writing, beat counting via linear recurrence
- SA layers: constraint verification, retrieval, copy detection

Shared across all layers:
- FluxAttention cross-attention with content-dependent reference points
- Predictive coding gate (surprise-driven information flow)
- FFN

Key innovation: Content-Dependent Reference Points
- time_prior: Predict "which time point to look at" from positional embedding
- freq_prior: Predict "high or low frequency region" from hidden state
- This corresponds to stream tracking in human auditory perception (Bregman, 1990)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FluxAttention


class DecoderLayerBase(nn.Module):
    """Base decoder layer: FluxCA + PC gate + FFN + reference points.

    Subclasses implement _sequence_forward() for SA or Mamba.

    Structure:
    1. Sequence model (SA or Mamba) — subclass-defined
    2. FluxAttention with content-dependent reference points + PC gate
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

        # FluxAttention (Cross-attention) with predictive coding gate
        self.cross_attn = FluxAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points_freq=n_points_freq,
            n_points_time=n_points_time,
            freq_offset_scale=freq_offset_scale,
            time_offset_scale=time_offset_scale,
        )
        # Predictive coding gate (Rao & Ballard 1999):
        # Only prediction error (surprise) passes through, gain can amplify.
        self.ca_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.ca_gain = nn.Linear(d_model, 1)
        # Auxiliary loss storage (training only, cleared after collection)
        self._last_prediction = None
        self._last_ca_output = None
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
        if use_time_prior:
            self.time_prior = nn.Linear(d_model, 1)
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

        Key insight: gain=0.001 causes reference points to be stuck at
        sigmoid(0)=0.5 (feature map center). The gradient signal from the
        language model prior is too weak to push them out. Instead:

        - freq_prior: Spread bias across frequency axis so each head group
          starts at a different register (bass/tenor/alto/soprano).
          Weight gain=0.1 gives W@x std ~ 0.14, enough for content-dependent
          modulation within each band while keeping sigmoid gradient healthy.
        - reference_refine: std=0.01 (was 0.001) for visible initial offsets.
        - time_prior: Prefer fixed linspace (use_time_prior=False) because
          position embeddings (std=0.02) are too small to produce a 0->1 ramp
          through a linear layer.
        """
        if self.time_prior is not None:
            nn.init.xavier_uniform_(self.time_prior.weight, gain=0.1)
            nn.init.constant_(self.time_prior.bias, 0.)

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

        # 3. FluxAttention (Cross-attention) with PC gate
        tgt2 = self.cross_attn(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios,
            value_cache=value_cache,
        )
        # Predictive coding: only surprise (prediction error) passes through
        prediction = self.ca_predictor(tgt.detach())       # [B, S, D]
        surprise = tgt2 - prediction.detach()               # [B, S, D]
        gain = F.softplus(self.ca_gain(tgt.detach()))       # [B, S, 1]
        tgt = tgt + gain * self.dropout2(surprise)
        # Store for auxiliary MSE loss (training only)
        if self.training:
            self._last_prediction = prediction
            self._last_ca_output = tgt2.detach()
        tgt = self.norm2(tgt)

        # 4. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, new_state
        return tgt

    def _compute_reference_points(
        self,
        tgt: torch.Tensor,      # [B, S, D]
        tgt_pos: Optional[torch.Tensor],  # [B, S, D] or [1, S, D]
        B: int,
        S: int,
    ) -> torch.Tensor:
        """Compute content-dependent reference points (per-head).

        Returns:
            reference_points: [B, S, H, L, 2] where 2 = (time, freq)
        """
        device = tgt.device
        H = self.n_heads

        # Time prior: from position embedding (shared across heads)
        if self.time_prior is not None and tgt_pos is not None:
            time_base = self.time_prior(tgt_pos).sigmoid()
            if time_base.shape[0] == 1 and B > 1:
                time_base = time_base.expand(B, -1, -1)
        else:
            time_base = torch.linspace(0, 1, S, device=device)
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
    """Self-Attention decoder layer.

    Causal self-attention for constraint verification and retrieval:
    - Look back at all tokens in the bar to verify beat sum
    - Detect repeated sections (copy-paste patterns)
    - Global attention over full sequence (no recurrence window limit)
    """

    def __init__(self, **kwargs):
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

    def _sequence_forward(self, tgt, tgt_pos, past_kv, use_cache):
        """Causal self-attention with optional KV-cache."""
        B, S, D = tgt.shape

        # SA needs explicit position embedding
        if tgt_pos is not None:
            q_input = tgt + tgt_pos
        else:
            q_input = tgt

        qkv = self.self_attn_qkv(q_input)  # [B, S, 3*D]
        qkv = qkv.reshape(B, S, 3, self.sa_n_heads, self.sa_head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_head]
        q_sa, k_sa, v_sa = qkv[0], qkv[1], qkv[2]

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
    - All layers share: FluxCA + PC gate + FFN + reference points
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
        # Legacy (ignored if decoder_layer_types is set)
        n_layers: int = 6,
    ):
        super().__init__()

        if decoder_layer_types is None:
            # Legacy: all SA layers (backward compat with old checkpoints)
            decoder_layer_types = ['sa'] * n_layers

        self.layer_types = decoder_layer_types

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
                self.layers.append(SADecoderLayer(**base_kwargs))
            else:
                raise ValueError(f"Unknown decoder layer type: {lt}")

        # Assign layer_idx to Mamba layers (needed for InferenceParams state indexing)
        mamba_idx = 0
        for layer in self.layers:
            if isinstance(layer, MambaDecoderLayer):
                layer.mamba.layer_idx = mamba_idx
                mamba_idx += 1

        self.norm = nn.LayerNorm(d_model)

    def collect_pred_loss(self) -> Optional[torch.Tensor]:
        """Collect predictor MSE loss from all layers (predictive_coding mode).

        Returns average MSE across layers, or None if not in PC mode.
        Clears stored predictions to prevent stale data.
        """
        losses = []
        for layer in self.layers:
            if getattr(layer, '_last_prediction', None) is not None:
                losses.append(
                    F.mse_loss(layer._last_prediction, layer._last_ca_output)
                )
                layer._last_prediction = None
                layer._last_ca_output = None
        if losses:
            return torch.stack(losses).mean()
        return None

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
