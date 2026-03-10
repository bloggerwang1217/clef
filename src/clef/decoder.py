"""
Clef Decoder
============

Hybrid decoder: Mamba2 + Self-Attention + Cross-Attention.

Layer pattern (clef_piano_base, 6 layers):
  L1: sa_window_ca  — SA + Bar Full-Attn on onset_1d + Window CA on S2+S3
  L2: mamba_only    — sequential writing
  L3: sa_window_ca  — SA + Window CA on S0+S1  (t from L1 DSNT, f from L1 DSNT)
  L4: mamba_only    — sequential writing
  L5: sa_window_ca  — SA + Window CA on L0+L1  (t from L3 DSNT, f from L3 DSNT)
  L6: mamba_only    — sequential writing

Key design:
- <bar> tokens in L1 do full attention over onset_1d (Octopus F-pooled, [B,T_octopus,C]).
  bar_center (carry-forward of bar full-attn com_t) is the initial t-center for L1 Window CA.
  This is Zeng 2024's bar_hidden state adapted to flat AR decoding.
  Positional bias (-scale * time_pos) ensures initial CoM near 0 (not 0.5), respecting BarGRU causality.
- Coarse-to-fine cascade (both t and f): L1 DSNT → L3 center → L3 DSNT → L5 center.
  L1 is the only layer that uses bar_center directly; L3/L5 use the refined DSNT com_t.
 - guidance_bounds supervises time_center (note tokens + <bar> tokens) to fall within measure range.
   For <bar> tokens, NoteGRU time_center is overridden to BarGRU com_t, so hinge loss
   still anchors bar timing while also supervising note-level time_center.
"""

from typing import List, Optional, Tuple, Union

import math

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .attention import WindowCrossAttention
from .cif import CIFModule, compute_acoustic_target_lengths, build_cif_alignment


def _carry_forward_h_bar(
    h_bar_scatter: torch.Tensor,  # [B, S, D_bar] — non-zero only at <bar> positions
    bar_mask: torch.Tensor,       # [B, S] bool
) -> torch.Tensor:
    """Propagate h_bar at <bar> positions forward so each position gets the h_bar
    from the most recent bar processed up to that point.

    Same logic as carry-forward bar center but for D_bar-dim vectors.
    Positions before the first <bar> get zeros.
    """
    B, S, D = h_bar_scatter.shape
    bar_count = bar_mask.long().cumsum(dim=1)  # [B, S]
    max_n = int(bar_count.max().item())
    if max_n == 0:
        return torch.zeros_like(h_bar_scatter)

    table = torch.zeros(B, max_n + 1, D, device=h_bar_scatter.device, dtype=h_bar_scatter.dtype)
    bar_positions = bar_mask.nonzero(as_tuple=False)
    if bar_positions.shape[0] > 0:
        b_idx = bar_positions[:, 0]
        s_idx = bar_positions[:, 1]
        k_idx = bar_count[b_idx, s_idx]
        table[b_idx, k_idx] = h_bar_scatter[b_idx, s_idx]

    idx = bar_count.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(table, 1, idx)


# =============================================================================
# BarMamba — Simplified bar-level temporal localizer
# =============================================================================

class BarMamba(nn.Module):
    """Bar-level DSNT attention on BiMamba encoder memory.

    Architecture (two-stage):

    Stage 1 — Bar Summary Attention (Perceiver-style, local within each bar):
        At each <bar_N> position, attend ONLY to the note embeddings of bar N-1.
        - Query  : embed(<bar_N>) + BarPE(N)   — who am I?
        - Key/Val: embed(notes of bar N-1) + BarPE(N-1)   — what happened before me?
        - Mask   : block-diagonal, so <bar_N> cannot see any other bar's notes.
        Result: summary_query[N] — unique, bar-isolated representation.

    Stage 2 — DSNT Attention on BiMamba memory:
        Use summary_query to attend BiMamba level 2 → com_t ∈ [0, 1].

    Bar PE: sinusoidal on bar_index (integer), NOT on token position.
    This is an explicit hierarchical PE: unlike RoPE (which operates on token pos),
    BarPE encodes *musical structure*, letting the model compare bars globally.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        active_ca_level: int = 2,  # BiMamba output level (shape [1, W])
        n_levels: int = 6,         # kept for API compatibility
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.active_ca_level = active_ca_level

        # Stage 1: Bar Summary Attention projections
        self.bar_q_proj  = nn.Linear(d_model, d_model, bias=False)  # <bar> → Q
        self.bar_k_proj  = nn.Linear(d_model, d_model, bias=False)  # notes → K
        self.bar_v_proj  = nn.Linear(d_model, d_model, bias=False)  # notes → V
        self.bar_out     = nn.Linear(d_model, d_model, bias=False)  # summary → D

        # Stage 2: DSNT attention on BiMamba memory
        self.query_proj    = nn.Linear(d_model, d_model, bias=False)
        self.memory_k_proj = nn.Linear(d_model, d_model, bias=False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _sinusoidal_pe(self, position: torch.Tensor, scale: int = 1) -> torch.Tensor:
        """Sinusoidal PE for arbitrary integer positions.

        Args:
            position: [N] integer or float positions
            scale: multiply position by this before applying PE (for time-norm → frame idx)
        Returns: [N, d_model]
        """
        half_d = self.d_model // 2
        device = position.device
        pos = position.float() * scale
        dim = torch.arange(half_d, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (dim / half_d))
        angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0)   # [N, half_d]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [N, d_model]

    def _bar_pe(self, bar_indices: torch.Tensor) -> torch.Tensor:
        """Bar Positional Encoding: sinusoidal on bar INDEX (not token position).

        bar_indices: [S] int — the bar number each token belongs to (0-based).
        Returns: [S, d_model]
        """
        return self._sinusoidal_pe(bar_indices.float(), scale=1)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        y: torch.Tensor,                 # [B, S, D]  — raw token embeddings (tgt)
        memory: torch.Tensor,            # [B, N_mem, D] — full flattened encoder memory
        spatial_shapes: torch.Tensor,    # [n_levels, 2]
        level_start_index: torch.Tensor, # [n_levels]
        bar_mask: torch.Tensor,          # [B, S] bool — True at <bar> positions
        input_ids: Optional[torch.Tensor] = None,  # [B, S] int — for bar index derivation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (com_t_all [B, S, 1], summary_embed_dense [B, S, D]).

        com_t_all: DSNT time center at <bar> positions (zeros elsewhere).
        summary_embed_dense: Stage 1 bar summary embed forward-filled to all tokens in
            each complete bar (bar_index < N_b). Tokens in the current bar (bar_index ==
            N_b, still being decoded) are left as zeros so the caller can substitute tgt.
        """
        B, S, D = y.shape
        device, dtype = y.device, y.dtype

        # ── Compute bar_index per token: token at pos s belongs to bar k
        #    bar_index[s] = number of <bar> tokens at positions ≤ s (0-based).
        #    First group (before any <bar>) is bar 0.
        bar_index = bar_mask.long().cumsum(dim=1)          # [B, S], 0-indexed bar number
        # <bar> token at pos s gets the index it just opened (already incremented).
        # Notes before it should belong to the PREVIOUS bar.
        # After cumsum: bar_index[s] for a note = number of <bar>'s seen so far.
        # For a <bar> at position s: bar_index[s] is incremented AT s, so it == bar N.
        # Notes following that <bar> until the next <bar> also have bar_index == N.
        # This is exactly the "you belong to bar N" assignment we want for notes.

        # ── Compute Bar PE for the full sequence ─────────────────────────────
        # shape: [S, D] for a single bar_index sequence.
        # We compute per-sample since bar_index differs per batch element.
        y_with_bar_pe = y.clone()
        for b in range(B):
            pe = self._bar_pe(bar_index[b])                # [S, D]
            y_with_bar_pe[b] = y[b] + pe.to(dtype)

        # ── Stage 1: Bar Summary Attention ───────────────────────────────────
        # For each <bar_N>, attend to bar N's OWN notes.
        # summary_queries[i] = bar N's complete context.
        # com_t is stored at <bar_{N+1}> (one-bar delay): bar N's summary guides bar N+1.
        # In inference: run Perceiver after bar N is fully generated (when <bar_{N+1}> appears).

        com_t_all = torch.zeros(B, S, 1, device=device, dtype=dtype)
        summary_embed_dense = torch.zeros(B, S, D, device=device, dtype=dtype)
        bar_positions = bar_mask.nonzero(as_tuple=False)   # [N_bars_total, 2]

        if bar_positions.shape[0] == 0:
            return com_t_all, summary_embed_dense

        # ── Stage 2 setup: BiMamba memory ────────────────────────────────────
        lvl = self.active_ca_level
        n_levels = spatial_shapes.shape[0]
        start = level_start_index[lvl].item()
        end = (level_start_index[lvl + 1].item()
               if lvl < n_levels - 1 else memory.shape[1])
        H_l = int(spatial_shapes[lvl, 0].item())
        W_l = int(spatial_shapes[lvl, 1].item())
        assert H_l == 1, (
            f"BarMamba expects a 1D (time-only) level, but level {lvl} has H={H_l}."
        )
        memory_lvl = memory[:, start:end]                  # [B, W_l, D]
        time_norm = torch.arange(W_l, device=device, dtype=torch.float32) / max(W_l - 1, 1)
        time_pe   = self._sinusoidal_pe(time_norm, scale=W_l)  # [W_l, D]
        K_mem     = self.memory_k_proj(memory_lvl) + time_pe.unsqueeze(0).to(dtype)  # [B, W_l, D]

        H, d_h = self.n_heads, self.head_dim

        # Process per batch element (bar structure differs per sample)
        for b in range(B):
            b_bars = (bar_positions[:, 0] == b).nonzero(as_tuple=False).squeeze(1)
            if b_bars.shape[0] == 0:
                continue

            bar_seq_positions = bar_positions[b_bars, 1]  # [N_b] seq positions of <bar>
            N_b = bar_seq_positions.shape[0]
            bi  = bar_index[b]                             # [S] bar index per token

            # Collect summary queries for all bars in this sample
            summary_queries = torch.zeros(N_b, D, device=device, dtype=dtype)

            # <sos> id=1: never a "previous bar note", exclude so the leading
            # <bar> (pos 1 in the new token format) correctly falls back to
            # using its own embedding rather than trivially attending to <sos>.
            b_ids = input_ids[b] if input_ids is not None else None

            for i, bar_pos in enumerate(bar_seq_positions.tolist()):
                bar_n = bi[bar_pos].item()  # this <bar> opened bar N

                # Notes of bar N itself: tokens with bar_index == bar_n, not a <bar>,
                # and not <sos>/<pad>. In training (teacher-forcing) these are available;
                # in inference they exist once bar N is fully generated (caller ensures
                # BarMamba is invoked only after <bar_{N+1}> appears).
                own_note_mask = (bi == bar_n) & (~bar_mask[b])
                if b_ids is not None:
                    own_note_mask = own_note_mask & (b_ids > 1)

                if own_note_mask.any():
                    # <bar_N> queries its own bar's complete notes.
                    q_bar = self.bar_q_proj(y_with_bar_pe[b, bar_pos].unsqueeze(0))  # [1, D]

                    own_notes = y_with_bar_pe[b, own_note_mask]     # [n_notes, D]
                    K_note = self.bar_k_proj(own_notes)               # [n_notes, D]
                    V_note = self.bar_v_proj(own_notes)               # [n_notes, D]

                    # Multi-head attention
                    n_notes = own_notes.shape[0]
                    q_h = q_bar.view(1, H, d_h).permute(1, 0, 2)       # [H, 1, d_h]
                    k_h = K_note.view(n_notes, H, d_h).permute(1, 0, 2)# [H, n, d_h]
                    v_h = V_note.view(n_notes, H, d_h).permute(1, 0, 2)# [H, n, d_h]
                    sc  = torch.bmm(q_h, k_h.transpose(1, 2)) / math.sqrt(d_h)  # [H, 1, n]
                    aw  = F.softmax(sc, dim=-1)                          # [H, 1, n]
                    ctx = torch.bmm(aw, v_h)                             # [H, 1, d_h]
                    ctx = ctx.permute(1, 0, 2).reshape(1, D)             # [1, D]
                    summary_queries[i] = self.bar_out(ctx).squeeze(0)
                else:
                    # Empty bar (no notes): use <bar_N>'s own embedding as fallback.
                    summary_queries[i] = y_with_bar_pe[b, bar_pos]

            # ── Stage 2: DSNT on BiMamba memory ──────────────────────────────
            Q_proj = self.query_proj(summary_queries)           # [N_b, D]
            Q_h    = Q_proj.view(N_b, H, d_h)                  # [N_b, H, d_h]

            K_b    = K_mem[b].unsqueeze(0).expand(N_b, -1, -1)             # [N_b, W_l, D]
            K_b_h  = K_b.view(N_b, W_l, H, d_h).permute(0, 2, 1, 3)      # [N_b, H, W_l, d_h]

            scores = torch.einsum('nhd,nhtd->nht', Q_h, K_b_h) / math.sqrt(d_h)  # [N_b, H, W_l]
            attn_w = F.softmax(scores, dim=-1).mean(1)                             # [N_b, W_l]

            com_t_b = (attn_w * time_norm).sum(-1, keepdim=True)                  # [N_b, 1]
            # One-bar delay: bar N's com_t goes to <bar_{N+1}> position.
            # bar_seq_positions[0] (<bar_1>) stays zero (no prior summary yet).
            # The last bar's com_t is discarded (no <bar_{N+1}> to store it).
            if N_b > 1:
                com_t_all[b, bar_seq_positions[1:]] = com_t_b[:-1].to(dtype)

            # ── Build summary_embed_dense for this batch item ─────────────────
            # summary_queries[i] = bar N's complete notes summary (N = bar_n at bar_pos_i).
            # Inject into bar N+1's tokens so they use it as CA query.
            # Bar 1's tokens (bi==1) have no prior summary → stay zero → caller uses raw tgt.
            bar_ns = [bi[p].item() for p in bar_seq_positions.tolist()]
            for i in range(N_b - 1):  # skip last bar (no bar N+1 to inject into)
                next_bar_n = bar_ns[i] + 1
                tok_mask_b = (bi == next_bar_n)
                if tok_mask_b.any():
                    summary_embed_dense[b, tok_mask_b] = summary_queries[i]

        return com_t_all, summary_embed_dense  # [B, S, 1], [B, S, D]




def compute_curriculum_mask(
    bar_mask: torch.Tensor, # [B, S] bool — True at <bar> positions
    visible_bars: int,      # how many trailing bars are shown in full
) -> torch.Tensor:
    """Return a [B, S] bool mask — True for positions that should be zeroed out.

    All token positions belonging to bars BEFORE the visible window are masked.
    The <bar> tokens themselves are also masked (they're part of the compressed bar).
    visible_bars=0 → everything masked; visible_bars≥total_bars → nothing masked.
    """
    B, S = bar_mask.shape
    curriculum_mask = torch.zeros(B, S, dtype=torch.bool, device=bar_mask.device)
    for b in range(B):
        bar_pos = bar_mask[b].nonzero(as_tuple=False).squeeze(1).tolist()
        n_bars = len(bar_pos)
        if n_bars <= visible_bars:
            continue  # all bars visible — nothing to compress
        n_compress = n_bars - visible_bars
        # Mask everything up to and including the last compressed <bar> token
        first_visible_start = bar_pos[n_compress]   # first <bar> of visible window
        curriculum_mask[b, :first_visible_start] = True
    return curriculum_mask





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



class MambaOnlyLayer(nn.Module):
    """Mamba-only decoder layer: pure Mamba block (no CA, no FFN).

    Used for sequential writing without audio re-injection.
    Assumes audio context was already injected by an earlier CIF/CA layer.
    No FFN: Mamba's expand/project provides equivalent non-linearity internally
    (same rationale as MambaFullCALayer and Zamba/RWKV-7 design conventions).

    Inherits nn.Module directly (not DecoderLayerBase) to avoid creating
    unused cross_attn, time_prior, freq_prior, reference_refine components
    (~3M params per layer that would never receive gradients, causing DDP hang).
    """

    def __init__(self, d_model=512, ff_dim=2048, dropout=0.1,
                 d_state=128, d_conv=4, expand=2, **kwargs):
        # kwargs absorbs unused base_kwargs (n_heads, n_levels, ff_dim, etc.)
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
        audio_latent=None,        # Unused (no time_prior)
        window_center=None,       # Unused (no CA)
        time_center_in=None,      # Unused (no CA) — added for Global NoteGRU
        freq_center_in=None,      # Unused (no CA) — added for Global NoteGRU
        guidance_bounds=None,     # Unused
        onset_1d=None,            # Unused (no bar attention)
        input_ids=None,           # Unused (no bar attention)
        bar_mask=None,            # Unused (no bar attention)
        com_t_all=None,           # Unused (no bar attention)
        h_bar_final=None,         # Unused (no bar attention)
        h_bar_carried=None,       # Unused (no bar attention)
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

        if use_cache:
            return tgt, (None, inference_params)
        return tgt


class MambaFullCALayer(nn.Module):
    """Mamba + Full Cross-Attention decoder layer.
    
    Architecture (Zamba inspired):
      1. Mamba Block (compresses history, provides temporal/structural context)
      2. Full Cross-Attention (queries Audio based on Mamba context)
      3. No FFN (Mamba internally provides non-linear expand/proj FFN equivalents)
      
    This fundamentally prevents "Lazy Decoder" (Attention Collapse) because Mamba 
    has a compressed hidden state and cannot perfectly copy historical tokens.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        active_ca_levels: Optional[List[int]] = None,
        n_levels: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ca_head_dim = d_model // n_heads
        self.active_ca_levels = active_ca_levels if active_ca_levels is not None else list(range(n_levels))
        self.ca_attn_dropout = dropout

        # 1. Mamba2 (processes token history → produces context-rich state y)
        # Mirrors Zeng's GRU: y_t encodes everything seen up to t-1.
        self.norm1 = nn.LayerNorm(d_model)  # pre-norm before Mamba
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout_mamba = nn.Dropout(dropout)

        # 2. Cross-Attention (Q = y = Mamba output, K/V = Audio)
        # Mirrors Zeng's:  attn_weights = self.attn(hidden, encoder_outputs)
        #                  context      = bmm(attn_weights, encoder_outputs)
        self.norm2 = nn.LayerNorm(d_model)  # pre-norm before CA (applied to y)
        self.ca_q_proj  = nn.Linear(d_model, d_model)
        self.ca_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.ca_out_proj = nn.Linear(d_model, d_model)
        self.dropout_ca  = nn.Dropout(dropout)

        # Sinusoidal time PE for CA keys (absolute audio-frame coordinate)
        class SinusoidalPositionEmbedding(nn.Module):
            def __init__(self, d_model: int):
                super().__init__()
                self.d_model = d_model

            def forward(self, time_norm: torch.Tensor) -> torch.Tensor:
                N = time_norm.shape[0]
                half_d = self.d_model // 2
                device = time_norm.device
                position = time_norm.float() * N
                dim = torch.arange(half_d, device=device, dtype=torch.float32)
                inv_freq = 1.0 / (10000.0 ** (dim / half_d))
                angles = position.unsqueeze(1) * inv_freq.unsqueeze(0)
                return torch.cat([angles.sin(), angles.cos()], dim=-1)

        self._sinusoidal_time_pe = SinusoidalPositionEmbedding(d_model)

        # 3. Sidechain Compressor Fusion
        # Mirrors Zeng: out = Linear(cat[gru_out, context])
        # Ours:         fused = ducked_y + ca_out
        self.render_proj     = nn.Linear(d_model, d_model)
        self.compressor_temp = nn.Parameter(torch.tensor(2.659))



    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        past_state: Optional[Tuple] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple]] | torch.Tensor:
        B, S, D = tgt.shape
        mamba_state = past_state[1] if past_state is not None else None

        # Optional separate query path for curriculum learning:
        # When curriculum zeroes tgt for compressed bars, tgt_query provides a
        # meaningful bar_embedding for those positions so CA Q is not degenerate.
        tgt_query: Optional[torch.Tensor] = kwargs.get('tgt_query', None)

        # CIF bypass: if acoustic_emb_dense [B, S, D] is provided, skip full CA
        # and use it directly as ca_out. This is the CIF sidechain path.
        acoustic_emb_dense: Optional[torch.Tensor] = kwargs.get('acoustic_emb_dense', None)

        # ── Step 1: Mamba2  (≡ Zeng: GRU hidden state) ──────────────────────────
        # y encodes the full causal token history, just like GRU hidden.
        tgt_normed = self.norm1(tgt)
        mamba_out = self.mamba(tgt_normed, inference_params=mamba_state)
        y = tgt + self.dropout_mamba(mamba_out)   # y = history-aware state

        # ── Step 2: Cross-Attention (or CIF bypass) ───────────────────────────────
        if acoustic_emb_dense is not None:
            # CIF path: use pre-computed per-token acoustic embedding as ca_out.
            # Audio (acoustic_emb_dense) is the immutable base; full CA is skipped.
            # Gradient flows: acoustic_emb_dense ← CIF ← encoder_1d ← BiMamba.
            ca_out = self.dropout_ca(acoustic_emb_dense)
        else:
            # Standard full-CA path (base model compatibility).
            # Q carries full history; K carries physical audio-frame time.
            # When curriculum is active, tgt_query substitutes bar_embed for compressed bars.
            if tgt_query is not None:
                y_for_q = tgt_query + self.dropout_mamba(self.mamba(self.norm1(tgt_query),
                                                                   inference_params=mamba_state))
                y_normed = self.norm2(y_for_q)
            else:
                y_normed = self.norm2(y)

            memory_ca = []
            time_norm_list = []
            for lvl in self.active_ca_levels:
                start = level_start_index[lvl].item()
                end   = (level_start_index[lvl + 1].item()
                         if lvl < spatial_shapes.shape[0] - 1
                         else memory.shape[1])
                memory_ca.append(memory[:, start:end])
                H_l = int(spatial_shapes[lvl, 0].item())
                W_l = int(spatial_shapes[lvl, 1].item())
                w_norm = torch.arange(W_l, device=tgt.device, dtype=torch.float32) / max(W_l - 1, 1)
                time_norm_list.append(w_norm.unsqueeze(0).expand(H_l, -1).reshape(-1))

            memory_ca = torch.cat(memory_ca, dim=1)   # [B, N_kv, D]
            time_norm  = torch.cat(time_norm_list)     # [N_kv]
            N_kv = memory_ca.shape[1]

            q_ca = self.ca_q_proj(y_normed).reshape(B, S, self.n_heads, self.ca_head_dim).permute(0, 2, 1, 3).contiguous()
            kv   = self.ca_kv_proj(memory_ca).reshape(B, N_kv, 2, self.n_heads, self.ca_head_dim)
            k_ca = kv[:, :, 0].permute(0, 2, 1, 3).contiguous()
            v_ca = kv[:, :, 1].permute(0, 2, 1, 3).contiguous()

            # Sinusoidal PE on K: "I am audio frame at physical time τ"
            time_pe   = self._sinusoidal_time_pe(time_norm)
            time_pe_k = (time_pe.view(N_kv, self.n_heads, self.ca_head_dim)
                                 .permute(1, 0, 2).unsqueeze(0).to(k_ca.dtype))
            k_ca = k_ca + time_pe_k

            ca_sdpa = F.scaled_dot_product_attention(
                q_ca, k_ca, v_ca,
                dropout_p=self.ca_attn_dropout if self.training else 0.0,
                is_causal=False,
            )
            ca_out = self.ca_out_proj(ca_sdpa.permute(0, 2, 1, 3).reshape(B, S, D))
            ca_out = self.dropout_ca(ca_out)

        # ── Step 3: Sidechain Compressor Fusion  (≡ Zeng: cat[gru_out, context]) ──
        # fused = ducked_y + ca_out
        # ∂fused/∂ca_out = 1 always → Audio gradient path is never broken.
        y_rendered      = self.render_proj(y)
        cosine_sim      = F.cosine_similarity(y_rendered, ca_out, dim=-1)
        compressor_gate = torch.sigmoid(self.compressor_temp * cosine_sim)
        ducked_y        = y_rendered * compressor_gate.unsqueeze(-1)
        fused           = ducked_y + ca_out   # Audio (ca_out) is the immutable base

        if use_cache:
            return fused, (None, mamba_state)
        return fused


class MambaCIFLayer(nn.Module):
    """Mamba + Cross-Attention over CIF acoustic freq-token sequences.

    Replaces MambaFullCALayer's full CA with a targeted CA over the small
    set of per-fire acoustic tokens ([B, S, L_freq, D]) from CIFModule.

    L_freq = P (pitch bins from PitchSA) + H_freq (Swin freq patches)
    Typical: 128 + 16 = 144 tokens per fire slot.

    Architecture:
      1. Mamba2 Block  (compresses token history → y)
      2. Cross-Attention (Q = y, K/V = acoustic_tokens at ptr position)
         This is the key polyphonic retrieval step:
         Q = "I just generated C4", K/V has 128 pitch tokens + 16 Swin tokens.
         The decoder attends to the specific note it needs to predict next.
      3. Compressor Fusion (like MambaFullCALayer: fused = ducked_y + ca_out)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,  # absorbs unused base_kwargs
    ):
        super().__init__()
        from mamba_ssm import Mamba2

        # 1. Mamba2
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self._mamba_layer_idx_set = False
        self.dropout_mamba = nn.Dropout(dropout)

        # 2. Cross-Attention: Q = Mamba output, K/V = freq token sequence
        self.norm2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.ca_head_dim = d_model // n_heads
        self.ca_q_proj  = nn.Linear(d_model, d_model)
        self.ca_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.ca_out_proj = nn.Linear(d_model, d_model)
        self.dropout_ca  = nn.Dropout(dropout)

        # 3. Concat Fusion (Zeng-style: cat[y, ca_out] → Linear)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        past_state=None,
        use_cache: bool = False,
        **kwargs,
    ):
        B, S, D = tgt.shape
        mamba_state = past_state[1] if past_state is not None else None

        # acoustic_tokens_dense: [B, S, L_freq, D_model]  per-token freq-token sequences
        acoustic_tokens_dense = kwargs.get('acoustic_emb_dense', None)

        # Step 1: Mamba (history → y)
        tgt_normed = self.norm1(tgt)
        mamba_out = self.mamba(tgt_normed, inference_params=mamba_state)
        y = tgt + self.dropout_mamba(mamba_out)

        # Step 2: Cross-Attention over freq tokens (polyphonic retrieval)
        if acoustic_tokens_dense is not None and acoustic_tokens_dense.dim() == 4:
            # acoustic_tokens_dense: [B, S, L, D]
            B_, S_, L, D_ac = acoustic_tokens_dense.shape
            assert D_ac == D, f"CA kv dim {D_ac} != d_model {D}"

            # Flatten batch × token dim for efficient batch CA
            # Q: [B*S, 1, D], K/V: [B*S, L, D]
            y_normed = self.norm2(y)
            Q_flat = self.ca_q_proj(y_normed).view(B * S, 1, D)        # [B*S, 1, D]
            kv_flat = self.ca_kv_proj(
                acoustic_tokens_dense.view(B * S, L, D_ac)
            )  # [B*S, L, 2D]
            K_flat, V_flat = kv_flat.chunk(2, dim=-1)                   # [B*S, L, D] each

            # Reshape for multi-head: [B*S, H, 1/L, d_h]
            H, dh = self.n_heads, self.ca_head_dim
            Q_h = Q_flat.view(B*S, 1, H, dh).permute(0, 2, 1, 3)       # [B*S, H, 1, dh]
            K_h = K_flat.view(B*S, L, H, dh).permute(0, 2, 1, 3)       # [B*S, H, L, dh]
            V_h = V_flat.view(B*S, L, H, dh).permute(0, 2, 1, 3)       # [B*S, H, L, dh]

            ca_sdpa = F.scaled_dot_product_attention(
                Q_h, K_h, V_h,
                dropout_p=self.dropout_ca.p if self.training else 0.0,
                is_causal=False,
            )  # [B*S, H, 1, dh]
            ca_out = self.ca_out_proj(
                ca_sdpa.permute(0, 2, 1, 3).reshape(B * S, 1, D)
                .view(B, S, D)
            )  # [B, S, D]
            ca_out = self.dropout_ca(ca_out)

        elif acoustic_tokens_dense is not None and acoustic_tokens_dense.dim() == 3:
            # Backward compat: legacy [B, S, D] single vector (old Zeng concat path)
            ca_out = self.dropout_ca(acoustic_tokens_dense)
        else:
            # Fallback: no acoustic info
            ca_out = torch.zeros_like(y)

        # Step 3: Concat Fusion (Zeng-style)
        fused = self.fusion_proj(torch.cat([y, ca_out], dim=-1))  # [B, S, D]

        if use_cache:
            return fused, (None, mamba_state)
        return fused


class MambaSALayer(nn.Module):
    """Mamba2 + Windowed CA (CIF calibration) + Self-Attention decoder layer.

    Flow:
      1. Mamba(tgt) → y  [B, S, D]  — compress token history
      2. last_token = y[:, -1:, :]  [B, 1, D]
      3. Windowed CA: Q=last_token, KV=summary_tokens[ptr-K : ptr+K]
         → attn_weights [B, 1, (2K+1)*M]
         → CoM offset → actual_ptr  (hard, not differentiable, that's fine)
         → ca_out [B, 1, D]  (residual: ca_out = last_token + wca_out)
      4. Gather full audio tokens at actual_ptr → [B, L, D]
      5. concat [ca_out, full_audio] → [B, 1+L, D]
      6. Self-Attention (Flash SDPA) — ca_out attends freq tokens, freq tokens互看
      7. FFN
      8. Output: SA_out[:, 0:1, :] → [B, 1, D]

    Key design choices:
      - CA calibrates the hard CIF pointer using learned attention (robust to ±K misfire)
      - residual on ca_out ensures last_token representation is preserved for SA
      - SA input is [ca_out, full_audio]: last_token (via ca_out)互看 audio freq tokens
      - CoM ptr correction has no gradient — decoder loss backprops through ca_out → CA weights
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        cif_window_k: int = 5,           # ±K fire slots for windowed CA
        cif_summary_m: int = 8,          # M summary tokens per fire slot (from Perceiver)
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cif_window_k = cif_window_k
        self.cif_summary_m = cif_summary_m

        # 1. Mamba2 (processes token history)
        self.norm1 = nn.LayerNorm(d_model)
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout_mamba = nn.Dropout(dropout)

        # 2. Windowed CA for CIF pointer calibration
        self.norm_wca = nn.LayerNorm(d_model)
        self.wca_q_proj = nn.Linear(d_model, d_model)
        self.wca_k_proj = nn.Linear(d_model, d_model)
        self.wca_v_proj = nn.Linear(d_model, d_model)
        self.wca_out    = nn.Linear(d_model, d_model)
        self.dropout_wca = nn.Dropout(dropout)

        # 3. Self-Attention (Flash)
        self.norm2 = nn.LayerNorm(d_model)
        self.sa_qkv = nn.Linear(d_model, 3 * d_model)
        self.sa_out = nn.Linear(d_model, d_model)
        self.dropout_sa = nn.Dropout(dropout)

        # 4. FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        past_state=None,
        use_cache: bool = False,
        **kwargs,
    ):
        B, S, D = tgt.shape
        mamba_state = past_state[1] if past_state is not None else None

        # full_tokens:    [B, N, L, D] — complete freq tokens per fire slot (for SA)
        # summary_tokens: [B, N, M, D] — Perceiver summary per fire slot (for Windowed CA)
        # Computed lazily via cif_lazy() to avoid holding [B, N, L, D] in memory
        # across the full decoder forward. cif_lazy is a zero-arg callable.
        full_tokens    = kwargs.get('acoustic_emb_full', None)
        summary_tokens = kwargs.get('acoustic_emb_summary', None)
        cif_lazy       = kwargs.get('cif_lazy', None)
        cif_ptr        = kwargs.get('cif_ptr', None)

        # Resolve lazy CIF if eager tensors were not provided
        if full_tokens is None and summary_tokens is None and cif_lazy is not None:
            full_tokens, summary_tokens = cif_lazy()
        # cif_ptr may be:
        #   [B]    during inference (single new token, pointer to current fire slot)
        #   [B, S] during training  (per-token pointer, from build_cif_alignment)

        # Step 1: Mamba (history → y)
        # Mamba needs all S tokens for causal compression.
        tgt_normed = self.norm1(tgt)
        mamba_out = self.mamba(tgt_normed, inference_params=mamba_state)
        y = tgt + self.dropout_mamba(mamba_out)

        # Steps 2-6 operate on the LAST token only, in both training and inference.
        # Training: Mamba sees full sequence; only last token queries audio.
        #   output = cat(y[:, :-1, :], last_out)  →  [B, S, D]
        # Inference: S=1 by definition, same path.
        last_token = y[:, -1:, :]  # [B, 1, D]

        # Step 2: Windowed CA — last token queries Perceiver summary around cif_ptr
        if summary_tokens is not None and cif_ptr is not None:
            N = summary_tokens.shape[1]
            K = self.cif_window_k
            M = self.cif_summary_m
            device = tgt.device
            offsets = torch.arange(-K, K + 1, device=device)  # [2K+1]

            # cif_ptr: [B, S] during training (take last col), [B] during inference
            ptr = cif_ptr[:, -1] if cif_ptr.dim() == 2 else cif_ptr  # [B]

            indices = ptr.unsqueeze(1) + offsets.unsqueeze(0)          # [B, 2K+1]
            pad_mask = (indices < 0) | (indices >= N)                  # [B, 2K+1]
            indices_clamped = indices.clamp(0, N - 1)                  # [B, 2K+1]
            idx_exp = indices_clamped.unsqueeze(-1).unsqueeze(-1).expand(B, 2*K+1, M, D)
            window = summary_tokens.gather(1, idx_exp)                 # [B, 2K+1, M, D]
            window = window.masked_fill(pad_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            window_flat = window.reshape(B, (2*K+1)*M, D)              # [B, (2K+1)*M, D]

            last_normed = self.norm_wca(last_token)
            q = self.wca_q_proj(last_normed)                           # [B, 1, D]
            k = self.wca_k_proj(window_flat)                           # [B, (2K+1)*M, D]
            v = self.wca_v_proj(window_flat)                           # [B, (2K+1)*M, D]

            def split_heads(x, seq):
                return x.reshape(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
            q_h = split_heads(q, 1)
            k_h = split_heads(k, (2*K+1)*M)
            v_h = split_heads(v, (2*K+1)*M)

            scale = self.head_dim ** -0.5
            attn_logits = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # [B, H, 1, (2K+1)*M]
            attn_weights = attn_logits.softmax(dim=-1)
            wca_out_h = torch.matmul(attn_weights, v_h)                      # [B, H, 1, head_dim]
            wca_out = wca_out_h.transpose(1, 2).reshape(B, 1, D)
            wca_out = self.wca_out(wca_out)
            wca_out = self.dropout_wca(wca_out)
            ca_out = last_token + wca_out                                     # [B, 1, D]

            # CoM → actual_ptr (used during inference for next-step pointer update)
            attn_avg = attn_weights.mean(dim=1).squeeze(1)                   # [B, (2K+1)*M]
            slot_weights = attn_avg.reshape(B, 2*K+1, M).sum(dim=-1)        # [B, 2K+1]
            com_offset = (slot_weights * offsets.float()).sum(dim=-1)        # [B]
            actual_ptr = (ptr + com_offset.round().long()).clamp(0, N - 1)   # [B]
        else:
            ca_out = last_token                                               # [B, 1, D]
            ptr = cif_ptr[:, -1] if (cif_ptr is not None and cif_ptr.dim() == 2) else cif_ptr
            actual_ptr = ptr

        # Step 3: Gather full audio tokens at actual_ptr, build SA input
        if full_tokens is not None:
            L = full_tokens.shape[2]
            ptr_clamped = actual_ptr.clamp(max=full_tokens.shape[1] - 1)    # [B]
            ptr_exp = ptr_clamped.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, 1, L, D)
            audio_at_ptr = full_tokens.gather(1, ptr_exp).squeeze(1)        # [B, L, D]
            sa_input = torch.cat([ca_out, audio_at_ptr], dim=1)             # [B, 1+L, D]
        else:
            sa_input = ca_out                                                # [B, 1, D]

        # Step 4: Self-Attention over [ca_out, full_audio] — last token attends to its audio slice
        sa_input_normed = self.norm2(sa_input)
        qkv = self.sa_qkv(sa_input_normed)
        q_sa, k_sa, v_sa = qkv.chunk(3, dim=-1)
        sa_out = F.scaled_dot_product_attention(
            q_sa, k_sa, v_sa,
            dropout_p=self.dropout_sa.p if self.training else 0.0,
            is_causal=False,
        )
        sa_out = self.sa_out(sa_out)
        sa_out = self.dropout_sa(sa_out)

        # Take only the ca_out position (index 0) after SA
        sa_out_0 = sa_out[:, :1, :]  # [B, 1, D]

        # Step 5: FFN
        ffn_out = self.ffn(self.norm3(sa_out_0))  # [B, 1, D]

        # last_out: combine ca_out + SA residual + FFN
        last_out = ca_out + sa_out_0 + ffn_out    # [B, 1, D]

        # Output: prepend Mamba pass-through for positions 0..S-2, append last_out
        if S > 1:
            output = torch.cat([y[:, :-1, :], last_out], dim=1)  # [B, S, D]
        else:
            output = last_out                                      # [B, 1, D]

        if use_cache:
            return output, (None, mamba_state)
        return output


def _monotonic_attn_alpha(
    p: torch.Tensor,             # [B, S, T]  selection probs p_{i,j} = σ(energy)
    alpha_init: torch.Tensor,    # [B, T]     α_{0,j} = 1[j=0]
) -> torch.Tensor:
    """Compute expected attention weights via monotonic recurrence (Raffel et al. 2017 eq.11-14).

    Structure:
      - Outer loop over i (output steps): SEQUENTIAL — α_{i,j} depends on α_{i-1,j}.
      - Inner computation over j (memory positions): PARALLEL via cumprod + cumsum.

    For each output step i, the recurrence is:
        q_{i,j} = (1-p_{i,j-1}) * q_{i,j-1} + α_{i-1,j}
        α_{i,j} = p_{i,j} * q_{i,j}

    Closed-form parallel-over-j solution:
        cp[j]  = prod_{l=0}^{j-1}(1-p[l])   ← exclusive cumprod of decay factors
        q[j]   = cp[j] * cumsum(α_prev / cp)[j]
        α[j]   = p[j] * q[j]

    Numerical stability: cp stays in (0,1] (no overflow); division uses safe eps clamp.
    When cp[j] → 0 (far past positions), α_prev[j] is also ~0 (no mass there), so error is negligible.

    Complexity: O(S) sequential outer steps, each O(T) parallel GPU ops = O(S * T) total.

    Returns:
        alpha: [B, S, T]
    """
    B, S, T = p.shape
    device, dtype = p.device, p.dtype

    alpha_prev = alpha_init                                                   # [B, T]
    alphas = []

    for i in range(S):
        p_i = p[:, i, :]                                                      # [B, T]

        # Exclusive cumprod of decay factors: cp[j] = prod_{l=0}^{j-1}(1-p_i[l])
        # cp[0] = 1.0, cp[j] = (1-p[0])*(1-p[1])*...*(1-p[j-1])
        one_minus_p = (1.0 - p_i).clamp(min=1e-8)                            # [B, T]
        cp = F.pad(
            torch.cumprod(one_minus_p[:, :-1], dim=-1),
            (1, 0), value=1.0
        )                                                                      # [B, T]

        # q[j] = cp[j] * cumsum(α_prev / cp)[j]
        safe_cp = cp.clamp(min=1e-8)                                          # avoid 0-division
        cs      = torch.cumsum(alpha_prev / safe_cp, dim=-1)                  # [B, T]
        q_i     = cp * cs                                                     # [B, T]

        alpha_i    = p_i * q_i                                                # [B, T]
        alphas.append(alpha_i)
        alpha_prev = alpha_i

    return torch.stack(alphas, dim=1)                                         # [B, S, T]


class MambaMonoAttnLayer(nn.Module):
    """Mamba + Monotonic Cross-Attention decoder layer.

    Implements the core CLEF decoder:

      Step 1: y = Mamba(tgt)                        [B, S, D]  — token history
      Step 2: e_{i,j} = a(y_i, h_j) + logit_bias_j              — energy + onset prior
              p_{i,j} = σ(e_{i,j})                 [B, S, T]
              α_{i,j} via recurrence (Raffel 2017)  [B, S, T]  — expected alignment
              c_i = Σ_j α_{i,j} h_j                [B, S, D]  — audio context
      Step 3: W_fuse(cat([y, c])) → output          [B, S, D]

    Gradient path: CE → c_i → α_{i,j} → p_{i,j} → onset_logit_bias → BiMamba
    Training: soft α (differentiable expected value)
    Inference: re-uses soft path (hard pointer optimization deferred)
    """

    def __init__(
        self,
        d_model: int = 384,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 6,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Step 1: Mamba (token history accumulation)
        self.norm_mamba = nn.LayerNorm(d_model)
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout_mamba = nn.Dropout(dropout)

        # Step 2: Monotonic cross-attention projections
        # Q from y_i, K from h_j (memory), multi-head dot-product
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # Scalar offset r (Raffel 2017 eq.16): added to energy before sigmoid.
        # r: fallback scalar offset used when onset_logit_bias is unavailable.
        # When onset_logit_bias is provided, it serves as the inductive bias instead.
        self.r = nn.Parameter(torch.tensor(-4.0))
        self.dropout_attn = nn.Dropout(dropout)

        # Step 3: Fusion W_fuse(cat([y, c])) → D
        self.fuse_proj = nn.Linear(2 * d_model, d_model)
        self.dropout_fuse = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,                              # [B, S, D]
        memory: torch.Tensor,                           # [B, T, D]  BiMamba output h_j
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        past_state=None,
        use_cache: bool = False,
        onset_logit_bias: Optional[torch.Tensor] = None,  # [B, T]  from OnsetDetector
        **kwargs,
    ) -> torch.Tensor:
        B, S, D = tgt.shape
        T = memory.shape[1]
        device = tgt.device
        mamba_state = past_state[1] if past_state is not None else None

        # ── Step 1: Mamba (token history) ─────────────────────────────────────
        tgt_normed = self.norm_mamba(tgt)
        mamba_out  = self.mamba(tgt_normed, inference_params=mamba_state)
        y = tgt + self.dropout_mamba(mamba_out)                              # [B, S, D]

        # ── Step 2: Monotonic Cross-Attention ─────────────────────────────────
        # Q from y, K from memory h, multi-head scaled dot-product
        H, dh = self.n_heads, self.head_dim
        Q = self.q_proj(y).reshape(B, S, H, dh).permute(0, 2, 1, 3)        # [B, H, S, dh]
        K = self.k_proj(memory).reshape(B, T, H, dh).permute(0, 2, 1, 3)   # [B, H, T, dh]

        # Energy: e_{i,j} = Q_i · K_j / sqrt(dh) + onset_logit_bias_j + r
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (dh ** 0.5)         # [B, H, S, T]
        energy = energy.mean(dim=1)                                          # [B, S, T]

        if onset_logit_bias is not None:
            # onset_logit_bias [B, T] acts as frame-varying inductive bias (replaces r)
            energy = energy + onset_logit_bias.unsqueeze(1)                  # [B, S, T]
        else:
            energy = energy + self.r                                         # [B, S, T]

        p = torch.sigmoid(energy)                                            # [B, S, T]

        # Monotonic attention recurrence: α_{i,j} via Raffel 2017 parallel scan
        alpha_init = torch.zeros(B, T, device=device, dtype=tgt.dtype)
        alpha_init[:, 0] = 1.0                                               # start at j=0
        alpha = _monotonic_attn_alpha(p, alpha_init)                         # [B, S, T]

        # Context vector: c_i = Σ_j α_{i,j} h_j
        c = torch.bmm(alpha, memory)                                         # [B, S, D]
        c = self.dropout_attn(c)

        # ── Step 3: Fusion ────────────────────────────────────────────────────
        fused = self.fuse_proj(torch.cat([y, c], dim=-1))                   # [B, S, D]
        output = self.dropout_fuse(fused)

        if use_cache:
            return output, (None, mamba_state)
        return output

    def decode_step(
        self,
        tgt: torch.Tensor,               # [B, S, D]  partial sequence so far
        memory: torch.Tensor,            # [B, T, D]  encoder memory
        onset_logit_bias: torch.Tensor,  # [B, T]     onset logit bias
        ptr: torch.Tensor,               # [B]        current monotonic pointer (LongTensor)
        tau: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hard monotonic decoding step for autoregressive inference.

        Scans forward from ptr until p_{i,j} > tau, returns c_i = h[t_i].
        Advances the pointer to the new position.

        Returns:
            output:  [B, D]  fused representation for last token
            new_ptr: [B]     updated pointer positions
        """
        B, S, D = tgt.shape
        T = memory.shape[1]
        device = tgt.device

        # Step 1: Mamba on full partial sequence
        tgt_normed = self.norm_mamba(tgt)
        mamba_out  = self.mamba(tgt_normed)
        y = tgt + self.dropout_mamba(mamba_out)
        y_last = y[:, -1:, :]                                       # [B, 1, D]

        # Step 2: Hard monotonic attention — scan from ptr
        H, dh = self.n_heads, self.head_dim
        Q = self.q_proj(y_last).reshape(B, 1, H, dh).permute(0, 2, 1, 3)   # [B, H, 1, dh]
        K = self.k_proj(memory).reshape(B, T, H, dh).permute(0, 2, 1, 3)   # [B, H, T, dh]

        energy = torch.matmul(Q, K.transpose(-2, -1)).squeeze(2) / (dh ** 0.5)  # [B, H, T]
        energy = energy.mean(dim=1) + onset_logit_bias                           # [B, T]  (r not used; onset_logit_bias is the inductive bias)
        p_full = torch.sigmoid(energy)                                            # [B, T]

        # For each batch element, scan forward from its current ptr
        new_ptr = ptr.clone()
        c_list  = []
        for b in range(B):
            p_b   = p_full[b]                        # [T]
            start = ptr[b].item()
            # Find first position >= start where p > tau
            fired = (p_b[start:] > tau).nonzero(as_tuple=False)
            if fired.numel() > 0:
                t_i = start + fired[0].item()
            else:
                t_i = T - 1                          # fallback: last frame
            new_ptr[b] = t_i
            c_list.append(memory[b, t_i])            # [D]

        c = torch.stack(c_list, dim=0).unsqueeze(1)  # [B, 1, D]

        # Step 3: Fusion
        fused = self.fuse_proj(torch.cat([y_last, c], dim=-1)).squeeze(1)   # [B, D]
        return fused, new_ptr


class MambaWindowCALayer(nn.Module):
    """Mamba2 + Window Cross-Attention decoder layer.

    Combines the best of MambaFullCALayer and SAWindowCALayer:
    - Mamba2 compresses token history (replaces SA as Q-source).
    - WindowCrossAttention provides coarse-to-fine DSNT cascade (replaces full SDPA).
    - Compressor Fusion gates the Mamba2 pathway by audio similarity.
    - No SA, no FFN (Mamba2's expand/project provides equivalent non-linearity).

    State format: (None, mamba_state) — no SA KV cache, only Mamba2 InferenceParams.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        active_ca_levels: Optional[List[int]] = None,
        n_levels: int = 6,
        # Window CA params (same as SAWindowCALayer)
        window_time_frames: Union[int, List[int]] = 16,
        window_freq_bins: Union[int, List[int]] = 8,
        window_seq_chunk_size: int = 128,
        full_freq: bool = False,
        full_freq_levels: Optional[List[int]] = None,
        cascade_com: bool = False,
        window_ca_use_checkpoint: bool = True,
        exp_decay_lambda: float = 0.0,        # >0: soft exp decay mask on window CA scores
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.active_ca_levels = active_ca_levels if active_ca_levels is not None else list(range(n_levels))

        # 1. Mamba2 (processes token history → produces context-rich state y)
        self.norm1 = nn.LayerNorm(d_model)
        from mamba_ssm import Mamba2
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout_mamba = nn.Dropout(dropout)

        # 2. Window Cross-Attention (Q = Mamba2 output)
        self.norm2 = nn.LayerNorm(d_model)
        self.window_ca = WindowCrossAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            window_time=window_time_frames,
            window_freq=window_freq_bins,
            seq_chunk_size=window_seq_chunk_size,
            full_freq=full_freq,
            full_freq_levels=full_freq_levels,
            cascade_com=cascade_com,
            use_checkpoint=window_ca_use_checkpoint,
            exp_decay_lambda=exp_decay_lambda,
        )
        self.dropout_ca = nn.Dropout(dropout)

        # 3. Compressor Fusion (same as MambaFullCALayer)
        self.render_proj = nn.Linear(d_model, d_model)
        self.compressor_temp = nn.Parameter(torch.tensor(2.659))

        # Caches read by ClefDecoder after forward
        self._cached_com_t_wca: Optional[torch.Tensor] = None
        self._cached_com_f: Optional[torch.Tensor] = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        past_state: Optional[Tuple] = None,
        use_cache: bool = False,
        value_cache=None,
        time_center_in=None,   # [B, S, 1] cascade from previous window_ca layer
        freq_center_in=None,   # [B, S, 1] cascade from previous window_ca layer
        **kwargs,
    ):
        """Mamba → Window CA (Q = Mamba output) → Compressor Fusion."""
        B, S, D = tgt.shape
        mamba_state = past_state[1] if past_state is not None else None

        # Step 1: Mamba2
        tgt_normed = self.norm1(tgt)
        mamba_out = self.mamba(tgt_normed, inference_params=mamba_state)
        y = tgt + self.dropout_mamba(mamba_out)

        # Step 2: Window Cross-Attention (Q = y from Mamba output)
        if freq_center_in is not None:
            freq_center = freq_center_in
        else:
            freq_center = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)

        if time_center_in is not None:
            time_center = time_center_in
        else:
            time_center = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)

        y_normed = self.norm2(y)
        ca_out, com_t, com_f = self.window_ca(
            query=y_normed,
            time_center=time_center,
            freq_center=freq_center,
            value=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            active_levels=self.active_ca_levels,
            kv_cache=value_cache,
        )
        self._cached_com_t_wca = com_t   # [B, S, 1]
        self._cached_com_f = com_f       # [B, S, 1]
        ca_out = self.dropout_ca(ca_out)

        # Step 3: Compressor Fusion (same as MambaFullCALayer)
        y_rendered = self.render_proj(y)
        cosine_sim = F.cosine_similarity(y_rendered, ca_out, dim=-1)
        compressor_gate = torch.sigmoid(self.compressor_temp * cosine_sim)
        ducked_y = y_rendered * compressor_gate.unsqueeze(-1)
        fused = ducked_y + ca_out   # Audio (ca_out) is the immutable base

        if use_cache:
            return fused, (None, mamba_state)
        return fused


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
        # Sinusoidal time PE is added to K so Q can learn temporal coordinates.
        self.ca_q_proj  = nn.Linear(d_model, d_model)
        self.ca_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.ca_out_proj = nn.Linear(d_model, d_model)
        self.ca_attn_dropout = dropout
        self.ca_head_dim = d_model // n_heads
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Direct time predictor for window center + guidance loss.
        # Replaces softmax-based CoM (which has gradient O(1/N_kv) = 8e-6 for N_kv=15k).
        # sigmoid(Linear(tgt)) has gradient O(1) — independent of N_kv.
        # Applied to tgt after SA (has sequential context via RoPE) before CA.
        # Zero-bias init → sigmoid(0) = 0.5 initially (center of chunk).
        self.time_predictor = nn.Linear(d_model, 1)
        nn.init.constant_(self.time_predictor.bias, 0.0)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def _sinusoidal_time_pe(time_norm: torch.Tensor, d_model: int) -> torch.Tensor:
        """Fixed sinusoidal position encoding for encoder KV positions.

        Treats normalized time as a scaled index, giving Q a consistent
        coordinate system to match against K across diverse audio pieces.
        Added to K only (not V): V stays as pure audio content.

        Args:
            time_norm: [N_kv] normalized time in [0, 1]
            d_model: total model dim (split evenly into sin/cos halves)

        Returns:
            pe [N_kv, d_model]
        """
        N = time_norm.shape[0]
        half_d = d_model // 2
        device = time_norm.device
        # Scale to [0, N_kv] so adjacent positions have distinct encodings
        position = time_norm.float() * N                         # [N_kv]
        dim = torch.arange(half_d, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (dim / half_d))            # [half_d]
        angles = position.unsqueeze(1) * inv_freq.unsqueeze(0)  # [N_kv, half_d]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [N_kv, d_model]

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
        value_cache=None,         # ignored (no FluxCA)
        audio_latent=None,        # ignored (no time_prior in this layer)
        window_center=None,       # ignored (this layer produces window_center, not consumes)
        guidance_bounds=None,     # [B, S, 2] (norm_start, norm_end) per token, or None
        onset_1d=None,            # unused
        input_ids=None,           # unused
        time_center_in=None,      # ignored (for compatibility with decoder calling convention)
        freq_center_in=None,      # ignored
        bar_mask=None,            # ignored
        com_t_all=None,           # ignored
        h_bar_final=None,         # ignored
        h_bar_carried=None,       # ignored
        tf_ratio=None,            # ignored
        pred_embs=None,           # ignored
    ):
        """SA + Full MHA on active encoder levels + FFN.

        Sinusoidal time PE is added to K (not V) so Q can learn to match
        temporal coordinates consistently across pieces.

        Guidance loss: hinge on sigmoid(Linear(tgt_after_ca)).
        Loss = relu(lo - t_pred) + relu(t_pred - hi), zero when t_pred
        is within the correct measure range [lo, hi].
        Gradient O(1), independent of N_kv.
        Placed AFTER Full CA so the gradient reaches CA Q/K/V projections
        and S2+S3 encoder features — prevents lazy-CA where SA bar-counting
        bypasses audio attention (gradient: loss → CA out → Q,K,V → SA).
        """
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

        N_kv = memory_ca.shape[1]

        # Compute time_norm [0,1] per encoder position (one value per level, own scale).
        # Used solely for sinusoidal time PE on K.
        if self.active_ca_levels is not None and len(self.active_ca_levels) > 0:
            time_norm_list = []
            for lvl in self.active_ca_levels:
                H_l = int(spatial_shapes[lvl, 0].item())
                W_l = int(spatial_shapes[lvl, 1].item())
                w_norm = (torch.arange(W_l, device=tgt.device, dtype=torch.float32)
                          / max(W_l - 1, 1))
                w_grid = w_norm.unsqueeze(0).expand(H_l, -1).reshape(-1)  # [H_l*W_l]
                time_norm_list.append(w_grid)
            time_norm = torch.cat(time_norm_list)  # [N_kv]
            has_coords = True
        else:
            time_norm = None
            has_coords = False

        # CA projections
        q_ca = self.ca_q_proj(tgt).reshape(B, S, self.n_heads, self.ca_head_dim).permute(0, 2, 1, 3)
        kv   = self.ca_kv_proj(memory_ca).reshape(B, N_kv, 2, self.n_heads, self.ca_head_dim)
        k_ca = kv[:, :, 0].permute(0, 2, 1, 3)   # [B, H, N_kv, D_head]
        v_ca = kv[:, :, 1].permute(0, 2, 1, 3)

        # Add sinusoidal time PE to K only (not V).
        # K_time = K_content + PE(time_norm) — gives Q a fixed temporal coordinate
        # to match against, consistent across audio pieces.
        if has_coords:
            time_pe = self._sinusoidal_time_pe(time_norm, D)   # [N_kv, D]
            # Reshape to [N_kv, H, D_head] then broadcast [1, H, N_kv, D_head]
            time_pe_k = (time_pe
                         .view(N_kv, self.n_heads, self.ca_head_dim)
                         .permute(1, 0, 2)      # [H, N_kv, D_head]
                         .unsqueeze(0)           # [1, H, N_kv, D_head]
                         .to(k_ca.dtype))
            k_ca = k_ca + time_pe_k

        # Full CA via F.sdpa (FlashAttention — no attention weights stored)
        ca_sdpa = F.scaled_dot_product_attention(
            q_ca, k_ca, v_ca,
            dropout_p=self.ca_attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        tgt2 = self.ca_out_proj(ca_sdpa.permute(0, 2, 1, 3).reshape(B, S, D))
        ca_out_dropped = self.dropout2(tgt2)   # save CA output (same dropout mask for both paths)
        tgt_sa = tgt                            # save tgt_SA before residual add
        tgt = tgt + ca_out_dropped
        tgt = self.norm2(tgt)

        # 2a. Direct time predictor (window center + guidance loss source).
        # Placed AFTER Full CA so guidance gradient flows through CA.
        #
        # SA-residual detach (Option B): gradient from guidance ONLY flows through
        # CA_out, not through the SA residual shortcut.
        # Without detach: guidance could train SA bar-counting alone and let CA be lazy.
        # With detach: CA Q/K/V projections + S2+S3 K content are the only gradient
        # path → CA is forced to be responsible for temporal prediction.
        #
        # The actual tgt (with both paths) is still used for decoding + CE loss.
        tgt_for_guidance = self.norm2(tgt_sa.detach() + ca_out_dropped)
        time_pred = torch.sigmoid(self.time_predictor(tgt_for_guidance))  # [B, S, 1]

        if self.training and guidance_bounds is not None:
            lo = guidance_bounds[:, :, 0]  # [B, S]
            hi = guidance_bounds[:, :, 1]  # [B, S]
            valid = lo >= 0               # structural tokens have sentinel -1
            if valid.any():
                com = time_pred.squeeze(-1)  # [B, S]
                loss_hinge = F.relu(lo - com) + F.relu(com - hi)
                self._cached_guidance_loss = loss_hinge[valid].mean()
            else:
                self._cached_guidance_loss = None
        else:
            self._cached_guidance_loss = None

        # 3. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, (new_kv, None)
        return tgt


class SAWindowCALayer(nn.Module):
    """SA + (optional Bar Full-Attention) + Window Cross-Attention decoder layer.

    L1 (bar_token_id is set):
      1. Causal SA with RoPE.
      2. Bar Full-Attention: only <bar> tokens attend to onset_1d [B, T_octopus, C].
         → com_t per <bar> (audio-grounded).
      3. Window CA on designated audio levels, centered at (time_center_in, freq_center).
      4. FFN.

    L3, L5 (bar_token_id is None):
      1. Causal SA with RoPE.
      2. Window CA centered at (time_center_in, freq_center).
      3. FFN.

    Optional guidance: if guidance_bounds is provided (not None), hinge loss is applied
    to com_t at <bar> positions. Default: off (guidance_loss_weight=0).
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
        bar_token_id: Optional[int] = None,   # set for L1 only
        onset_1d_channels: int = 32,          # C from Octopus F-pooled
        full_freq: bool = False,              # if True, all levels use full_freq
        full_freq_levels: Optional[List[int]] = None,  # per-level full_freq (overrides full_freq)
        cascade_com: bool = False,            # cascade CoM between levels (coarse→fine)
        window_ca_use_checkpoint: bool = True,  # gradient checkpoint per seq-chunk in WindowCA
        exp_decay_lambda: float = 0.0,        # >0: soft exp decay mask on window CA scores

        **kwargs,  # absorbs unused base_kwargs (n_points_*, use_time_prior, etc.)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.sa_head_dim = d_model // n_heads
        self.active_ca_levels = active_ca_levels
        self.n_levels = n_levels
        self.self_attn_dropout = dropout
        self.bar_token_id = bar_token_id

        # Causal Self-Attention with RoPE
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.rope = RotaryPositionalEmbedding(self.sa_head_dim)


        # Window Cross-Attention
        self.window_ca = WindowCrossAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            window_time=window_time_frames,
            window_freq=window_freq_bins,
            seq_chunk_size=window_seq_chunk_size,
            full_freq=full_freq,
            full_freq_levels=full_freq_levels,
            cascade_com=cascade_com,
            use_checkpoint=window_ca_use_checkpoint,
            exp_decay_lambda=exp_decay_lambda,
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

        # Compressor Fusion: SA history gates itself by cosine similarity with CA output.
        # Matches MambaWindowCALayer.render_proj / compressor_temp.
        self.render_proj = nn.Linear(d_model, d_model)
        self.compressor_temp = nn.Parameter(torch.tensor(2.659))

        # Caches read by ClefDecoder after forward.
        # _cached_com_f: [B, S, 1] com_f from this layer's WindowCA (passed to next layer as freq_center_in).
        # _cached_com_t_wca: [B, S, 1] com_t from this layer's WindowCA (time cascade to next layer).
        # _cached_guidance_loss: optional hinge loss on bar_center at <bar> positions.
        self._cached_com_f: Optional[torch.Tensor] = None
        self._cached_com_t_wca: Optional[torch.Tensor] = None
        self._cached_guidance_loss: Optional[torch.Tensor] = None

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
        value_cache=None,
        time_center_in=None,    # [B, S, 1] dense com_t from BarMamba (forward-filled)
        freq_center_in=None,    # [B, S, 1] com_f from previous layer's WindowCA (L3/L5 only)
        guidance_bounds=None,   # [B, S, 2] optional CoM hinge supervision
        bar_mask: Optional[torch.Tensor] = None,   # [B, S] bool, precomputed in ClefDecoder
        # Legacy kwargs passed by ClefDecoder (ignored here)
        audio_latent=None,
    ):
        """SA + Window CA + FFN."""
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

        # 2. Determine bar info for guidance and caching.
        self._cached_guidance_loss = None

        # Freq cascade: L1 starts at 0.5 (no prior); L3 uses L1's com_f; L5 uses L3's com_f.
        # com_f is the frequency center-of-mass from the previous layer's WindowCA attention
        # weights — audio-grounded, meaningful only when time is already aligned (NoteGRU handles that).
        if freq_center_in is not None:
            freq_center = freq_center_in  # [B, S, 1] from previous layer
        else:
            freq_center = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)

        if time_center_in is None:
            time_center = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)
        else:
            time_center = time_center_in

        # Time center guidance loss: CoM hinge on bar + note tokens.
        # For each supervised token (bar or note), the time_center should land within
        # the correct measure range [lo, hi] (normalized 0-1).
        # loss = relu(lo - time_center) + relu(time_center - hi)  →  0 inside interval, linear outside.
        # Structural tokens (sentinel -1) are excluded via lo >= 0.
        if self.training and guidance_bounds is not None:
            lo = guidance_bounds[:, :, 0]  # [B, S] measure start (normalized)
            hi = guidance_bounds[:, :, 1]  # [B, S] measure end (normalized)
            # Supervise all tokens with valid bounds (bar + note tokens)
            valid = lo >= 0  # [B, S]
            if valid.any():
                time_center_flat = time_center[:, :, 0]  # [B, S] scalar time_center per token
                loss_hinge = (
                    F.relu(lo[valid] - time_center_flat[valid]) +
                    F.relu(time_center_flat[valid] - hi[valid])
                )  # [N_valid]
                self._cached_guidance_loss = loss_hinge.mean()
            else:
                self._cached_guidance_loss = None
        else:
            self._cached_guidance_loss = None
        self._cached_com_f = None
        self._cached_com_t_wca = None

        # 4. Window Cross-Attention (Q = SA output, saved in tgt before WCA)
        sa_normed = tgt  # [B, S, D] — pre-WCA SA output (used as Compressor input)
        tgt2, _com_t_wca, com_f_wca = self.window_ca(
            query=sa_normed,
            time_center=time_center,
            freq_center=freq_center,
            value=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            active_levels=self.active_ca_levels,
            kv_cache=value_cache,
        )
        # Cache com_f for ClefDecoder to pass to the next sa_window_ca layer
        self._cached_com_f = com_f_wca  # [B, S, 1]
        self._cached_com_t_wca = _com_t_wca  # [B, S, 1]
        ca_out = self.dropout2(tgt2)

        # Compressor Fusion: SA history gates itself by cosine similarity with audio CA output.
        # Audio (ca_out) is the immutable base; SA (render_proj) is the compressor.
        sa_rendered = self.render_proj(sa_normed)
        cosine_sim = F.cosine_similarity(sa_rendered, ca_out, dim=-1)  # [B, S]
        compressor_gate = torch.sigmoid(self.compressor_temp * cosine_sim)
        ducked_sa = sa_rendered * compressor_gate.unsqueeze(-1)
        tgt = self.norm2(ducked_sa + ca_out)

        # 5. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, (new_kv, None)
        return tgt


class ClefDecoder(nn.Module):
    """Hybrid decoder: Mamba2 + Self-Attention + Cross-Attention.

    Layer pattern (clef_piano_base, 6 layers):
      L1: sa_window_ca - SA + Window CA on S2+S3
      L2: mamba_only   - sequential writing
      L3: sa_window_ca - SA + Window CA on S0+S1
      L4: mamba_only   - sequential writing
      L5: sa_window_ca - SA + Window CA on L0+L1
      L6: mamba_only   - sequential writing

    Temporal responsibility:
      BarMamba computes com_t at <bar> positions using BiMamba memory.
      com_t is forward-filled to all notes within the bar.
      Time cascade: L1 uses this dense com_t; L3 uses L1.com_t_wca; L5 uses L3.com_t_wca.
      (Direct cascade without blending: each layer refines the previous layer's output.)
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
        bar_token_id: Optional[int] = None,   # set for first window_ca (L1 BarGRU) or mamba_full_ca
        onset_1d_channels: int = 32,          # C from Octopus F-pooled for bar attention
        decoder_layer_full_freq: Optional[List] = None,  # per-layer: True/False or List[int] of full-freq level IDs
        decoder_layer_cascade_com: Optional[List[bool]] = None,  # per-layer cascade CoM flag
        window_ca_use_checkpoint: bool = True,  # passed to all SAWindowCALayer instances
        # Bar-GRU redesign params (forwarded to SAWindowCALayer)
        bar_gru_hidden_size: int = 256,
        bar_gru_input_dropout: float = 0.1,
        # Curriculum Learning for mamba_full_ca path
        curriculum_warmup_steps: int = 0,     # 0 = disabled; >0 = bar-progressive window expansion
        # Exponential decay soft mask for window CA (sa_window_ca and mamba_window_ca only)
        window_exp_decay_lambda: float = 0.0,
        # CIF (Continuous Integrate-and-Fire) — replaces BarMamba for tiny model
        use_cif: bool = False,                   # if True, build CIFModule instead of BarMamba
        cif_fire_signal_dim: int = 128,          # BiMamba output dim (for α prediction)
        cif_acoustic_dim: int = 192,             # Swin S0 output dim (pitch content)
        cif_acoustic_s1_dim: int = 384,          # Swin S1 output dim (beat content)
        cif_d_model: int = 512,                  # Final acoustic embedding dim (decoder d_model)
        cif_threshold: float = 0.5,              # CIF probability threshold
        cif_conv_kernel: int = 3,                # depthwise conv kernel for weight predictor
        cif_hidden_dim: int = 128,               # hidden dim for weight predictor Dense layer
        cif_target_fires: int = 128,             # Target fire count (avg structural tokens per chunk)
        cif_encoder_len: int = 3000,             # Encoder length (used for weight_proj bias init)
        cif_window_k: int = 5,                   # ±K fire slots for windowed CA in MambaSALayer
        cif_perceiver_m: int = 8,                # M Perceiver summary tokens per fire slot
    ):
        super().__init__()

        if decoder_layer_types is None:
            decoder_layer_types = ['sa_window_ca', 'mamba_only',
                                   'sa_window_ca', 'mamba_only', 'sa_window_ca', 'mamba_only']

        self.layer_types = decoder_layer_types
        self.bar_token_id = bar_token_id
        self.use_rope = use_rope
        self.bar_gru_hidden_size = bar_gru_hidden_size
        self.curriculum_warmup_steps = curriculum_warmup_steps

        # Shared kwargs passed to all layer constructors (unused keys absorbed by **kwargs)
        base_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            ff_dim=ff_dim,
            dropout=dropout,
            rope_base=rope_base,
        )

        # Build shared BarMamba or CIFModule depending on use_cif flag.
        # CIF replaces BarMamba for the tiny model (mamba_full_ca path).
        # Base model uses BarMamba (sa_window_ca path, bar_token_id still needed for window center).
        self.cif = None
        self.bar_mamba = None
        if bar_token_id is not None:
            if use_cif:
                self.cif = CIFModule(
                    fire_signal_dim=cif_fire_signal_dim,   # Octopus onset_1d dim (32)
                    swin_dim=cif_acoustic_dim,              # Swin output dim per token (192)
                    swin_s1_dim=cif_acoustic_dim,           # Swin S1 output dim (same as S0, 192)
                    d_model=cif_d_model,
                    threshold=cif_threshold,
                    conv_kernel=cif_conv_kernel,
                    cif_hidden_dim=cif_hidden_dim,
                    target_fires=cif_target_fires,
                    encoder_len=cif_encoder_len,
                    perceiver_m=cif_perceiver_m,
                )
            else:
                self.bar_mamba = BarMamba(
                    d_model=d_model,
                    n_heads=n_heads,
                    active_ca_level=2,   # BiMamba output level
                    n_levels=n_levels,
                )

        # The first window_ca layer gets bar_token_id assigned (legacy mapping)
        bar_ca_assigned = False

        self.layers = nn.ModuleList()
        for i, lt in enumerate(decoder_layer_types):
            ca_levels = decoder_layer_ca_levels[i] if decoder_layer_ca_levels else None
            if lt == 'mamba_only':
                self.layers.append(MambaOnlyLayer(
                    d_state=d_state, d_conv=d_conv, expand=expand,
                    **base_kwargs,
                ))
            elif lt == 'mamba_cif':
                self.layers.append(MambaCIFLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                ))
            elif lt == 'mamba_sa':
                self.layers.append(MambaSALayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    cif_window_k=cif_window_k,
                    cif_summary_m=cif_perceiver_m,
                ))
            elif lt == 'mamba_mono_attn':
                self.layers.append(MambaMonoAttnLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    n_heads=n_heads,
                    dropout=dropout,
                ))
            elif lt == 'mamba_full_ca':
                self.layers.append(MambaFullCALayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    n_heads=n_heads,
                    dropout=dropout,
                    active_ca_levels=ca_levels,
                    n_levels=n_levels,
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
            elif lt == 'sa_window_ca':
                # First sa_window_ca: assign bar_token_id for bar full-attention
                layer_bar_token_id = None
                if not bar_ca_assigned and bar_token_id is not None:
                    layer_bar_token_id = bar_token_id
                    bar_ca_assigned = True
                # decoder_layer_full_freq entry can be:
                #   True / False  → global full_freq flag
                #   List[int]     → per-level full_freq (e.g. [3] = only S1 full-freq)
                raw_ff = (
                    decoder_layer_full_freq[i]
                    if decoder_layer_full_freq is not None and i < len(decoder_layer_full_freq)
                    else False
                )
                if isinstance(raw_ff, (list, tuple)):
                    layer_full_freq = False
                    layer_full_freq_levels = list(raw_ff)
                else:
                    layer_full_freq = bool(raw_ff) if raw_ff is not None else False
                    layer_full_freq_levels = None
                layer_cascade_com = (
                    bool(decoder_layer_cascade_com[i])
                    if decoder_layer_cascade_com is not None and i < len(decoder_layer_cascade_com)
                    and decoder_layer_cascade_com[i] is not None
                    else False
                )
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
                    bar_token_id=layer_bar_token_id,
                    onset_1d_channels=onset_1d_channels,
                    full_freq=layer_full_freq,
                    full_freq_levels=layer_full_freq_levels,
                    cascade_com=layer_cascade_com,
                    window_ca_use_checkpoint=window_ca_use_checkpoint,
                    bar_gru_hidden_size=bar_gru_hidden_size,
                    bar_gru_input_dropout=bar_gru_input_dropout,
                    exp_decay_lambda=window_exp_decay_lambda,
                ))
            elif lt == 'mamba_window_ca':
                # Same full_freq / cascade_com parsing as window_ca
                raw_ff = (
                    decoder_layer_full_freq[i]
                    if decoder_layer_full_freq is not None and i < len(decoder_layer_full_freq)
                    else False
                )
                if isinstance(raw_ff, (list, tuple)):
                    layer_full_freq = False
                    layer_full_freq_levels = list(raw_ff)
                else:
                    layer_full_freq = bool(raw_ff) if raw_ff is not None else False
                    layer_full_freq_levels = None
                layer_cascade_com = (
                    bool(decoder_layer_cascade_com[i])
                    if decoder_layer_cascade_com is not None and i < len(decoder_layer_cascade_com)
                    and decoder_layer_cascade_com[i] is not None
                    else False
                )
                self.layers.append(MambaWindowCALayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    n_heads=n_heads,
                    dropout=dropout,
                    active_ca_levels=ca_levels,
                    n_levels=n_levels,
                    window_time_frames=window_time_frames,
                    window_freq_bins=window_freq_bins,
                    window_seq_chunk_size=window_seq_chunk_size,
                    full_freq=layer_full_freq,
                    full_freq_levels=layer_full_freq_levels,
                    cascade_com=layer_cascade_com,
                    window_ca_use_checkpoint=window_ca_use_checkpoint,
                    exp_decay_lambda=window_exp_decay_lambda,
                ))
            else:
                raise ValueError(f"Unknown decoder layer type: {lt!r}")

        # Assign layer_idx to Mamba layers (needed for InferenceParams state indexing)
        mamba_idx = 0
        for layer in self.layers:
            if isinstance(layer, (MambaOnlyLayer, MambaCIFLayer, MambaSALayer, MambaFullCALayer, MambaWindowCALayer)):
                layer.mamba.layer_idx = mamba_idx
                mamba_idx += 1

        self.norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = False  # enabled via enable_gradient_checkpointing()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        past_states: Optional[list] = None,
        use_cache: bool = False,
        value_cache_list: Optional[list] = None,
        guidance_bounds: Optional[torch.Tensor] = None,
        onset_1d: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        tf_ratio: float = 1.0,
        pred_embs: Optional[torch.Tensor] = None,
        p_onset: Optional[torch.Tensor] = None,          # [B, T]  onset logit bias @12.5fps
    ):
        """Forward through all decoder layers.

         Temporal routing (Global NoteGRU):
           - L1 (SAWindowCALayer with bar_token_id):
               BarGRU runs note_gru + bar_gru + FullAttn → h_bar + com_t.
           - NoteGRU reads before token (shared time_center), writes after token.
           - L1→L3→L5 time cascade: blend NoteGRU time_center with com_t_wca.

        Args:
            past_states: Per-layer state tuples (kv for SA, InferenceParams for Mamba).
            use_cache: Whether to return updated states for incremental decoding.
            value_cache_list: Pre-computed KV caches for WindowCrossAttention.
            guidance_bounds: [B, S, 2] per-token (lo, hi) time bounds.
            onset_1d: [B, T_octopus, C] Octopus F-pooled features for BarGRU.
            input_ids: [B, S] int token IDs for detecting <bar> tokens.
            tf_ratio: Scheduled sampling ratio (1.0 = full teacher forcing).
            pred_embs: [B, S, D] predicted token embeddings for TF.

        Returns:
            Without use_cache: output [B, S, D]
            With use_cache: (output [B, S, D], new_states list)
            Guidance loss cached on self._cached_guidance_loss after forward.
        """
        B, S, D = tgt.shape
        output = tgt
        new_states = [] if use_cache else None
        self._cached_guidance_loss = None


        # Derive bar_mask from input_ids
        bar_mask = None
        if input_ids is not None and self.bar_token_id is not None:
            bar_mask = (input_ids == self.bar_token_id)
        if bar_mask is None:
            bar_mask = torch.zeros(B, S, dtype=torch.bool, device=output.device)


        # ── CIF or BarMamba temporal alignment ──────────────────────────────
        dense_com_t = None
        summary_embed_dense = None
        acoustic_embs_dense   = None  # [B, S, L, D] for CIF path (non-MambaSALayer layers)
        acoustic_embs_full    = None  # [B, N, L, D] full freq tokens per fire slot
        acoustic_embs_summary = None  # [B, N, M, D] Perceiver summary per fire slot
        cif_lazy              = None  # callable () -> (full, summary) for MambaSALayer lazy eval
        self._cif_quantity_loss = None
        self._cif_sum_alpha = None
        self._cif_target = None

        # True if any layer type uses BarMamba/CIF's output (full CA or window CA)
        has_bar_ca = any(isinstance(l, (MambaCIFLayer, MambaFullCALayer, MambaWindowCALayer)) for l in self.layers)

        if self.cif is not None and memory is not None:
            # ── CIF path (tiny model) ─────────────────────────────────────────
            # Polyphonic design:
            #   fire_signal (onset_1d [B,T,32]): pure onset for α prediction
            #   acoustic_src (Swin 2D [B,H,W,192]): freq-slice tokens
            #   pitch_tokens (PitchSA): 128 pitch-bin tokens per fire slot

            if fire_signal is None or acoustic_src is None:
                raise ValueError("CIF requires both fire_signal and acoustic_src")

            target_lengths = None
            if input_ids is not None:
                target_lengths = compute_acoustic_target_lengths(input_ids)  # [B]

            # Step 1: compute α + fire_frames (no learnable token layers)
            fire_frames, alpha, qty_loss = self.cif.compute_fires(
                fire_signal=fire_signal,
                input_lengths=None,
                target_lengths=target_lengths,
            )

            # Step 2: PitchSA at fire positions (optional)
            pitch_tokens = None
            if pitch_sa_module is not None and flow_feat is not None:
                pitch_tokens = pitch_sa_module(flow_feat, fire_frames)  # [B, N, P, d_pitch]

            has_mamba_sa = any(isinstance(l, MambaSALayer) for l in self.layers)
            needs_dense  = any(isinstance(l, (MambaFullCALayer, MambaCIFLayer)) for l in self.layers)

            if needs_dense:
                # Step 3a: Non-MambaSALayer path — compute now and scatter to [B, S, L, D]
                acoustic_embs_full, acoustic_embs_summary = self.cif(
                    fire_signal=fire_signal,
                    swin_2d=acoustic_src,
                    fire_frames=fire_frames,
                    pitch_tokens=pitch_tokens,
                    swin_2d_s1=acoustic_src_s1,
                )
                acoustic_embs_dense = self.cif.align_to_seq_len(acoustic_embs_full, S, input_ids=input_ids)
            elif has_mamba_sa:
                # Step 3b: MambaSALayer path — defer CIF assembly until inside the layer.
                # Capture references in a closure; tensors are computed on first call and
                # the closure is not retained after the layer returns, so [B, N, L, D]
                # lives only while MambaSALayer.forward() is on the call stack.
                _cif_ref      = self.cif
                _fire_signal  = fire_signal
                _fire_frames  = fire_frames
                _pitch_tokens = pitch_tokens
                _acoustic_src = acoustic_src
                _acoustic_s1  = acoustic_src_s1
                def cif_lazy():
                    return _cif_ref(
                        fire_signal=_fire_signal,
                        swin_2d=_acoustic_src,
                        fire_frames=_fire_frames,
                        pitch_tokens=_pitch_tokens,
                        swin_2d_s1=_acoustic_s1,
                    )

            self._cif_quantity_loss = qty_loss
            self._cif_sum_alpha    = alpha.sum(dim=1).mean().item()
            self._cif_target       = target_lengths.mean().item() if target_lengths is not None else None


        elif (hasattr(self, 'bar_mamba')
                and self.bar_mamba is not None
                and memory is not None
                and bar_mask is not None
                and bar_mask.any()):

            # ── BarMamba path (base model, original behavior) ─────────────────
            sparse_com_t, summary_embed_dense = self.bar_mamba(
                y=tgt,
                memory=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                bar_mask=bar_mask,
                input_ids=input_ids,
            )

            # Forward-fill com_t (step-and-hold) within each bar
            dense_com_t = sparse_com_t.clone()
            for b in range(B):
                b_mask = bar_mask[b]
                if b_mask.any():
                    bar_idx = b_mask.nonzero(as_tuple=False).squeeze(1).tolist()
                    bar_ends = bar_idx[1:] + [S]
                    for start, end in zip(bar_idx, bar_ends):
                        val = sparse_com_t[b, start, 0]
                        dense_com_t[b, start:end, 0] = val

        # ── Build tgt_query (BarMamba path only) ─────────────────────────────
        # CIF path: tgt_query is not used (MambaFullCALayer uses acoustic_emb_dense instead)
        bar_index_per_tok = bar_mask.long().cumsum(dim=1)  # [B, S]

        if summary_embed_dense is not None and bar_mask.any():
            tgt_query = tgt.clone()
            for b in range(B):
                n_bars_b = int(bar_mask[b].float().sum().item())
                complete_mask = (bar_index_per_tok[b] >= 2) & (bar_index_per_tok[b] < n_bars_b)
                if complete_mask.any():
                    tgt_query[b, complete_mask] = summary_embed_dense[b, complete_mask]
        else:
            tgt_query = output  # fallback

        # ── Curriculum: embedding masking (no sequence length change) ────────
        # Compressed bars → their token embeddings are zeroed out.
        # ClefPianoTiny reads self._curriculum_mask to mask out corresponding labels.
        self._curriculum_mask = None  # reset each forward
        if (has_bar_ca
                and self.curriculum_warmup_steps > 0
                and bar_mask is not None
                and self.training):
            total_bars = int(bar_mask.float().sum(-1).max().item())
            global_step = getattr(self, '_curriculum_step', 0)
            steps_per_bar = max(self.curriculum_warmup_steps / max(total_bars, 1), 1)
            visible_bars = min(int(global_step / steps_per_bar) + 1, total_bars)
            curr_mask = compute_curriculum_mask(bar_mask, visible_bars)  # [B, S]
            if curr_mask.any():
                self._curriculum_mask = curr_mask
                # Zero-out Mamba decode path for compressed bars
                output = output.clone()
                output[curr_mask] = 0.0

                # ── Q-path scheduled sampling for compressed bars ─────────────
                # tgt_query already has summary_embed_dense for complete bars (GT).
                # For not-teacher-force: override with pred_embs (model's own prediction).
                # Applied per bar (same TF choice for all tokens in the same bar).
                if bar_mask.any() and tgt_query is not output:
                    for b in range(B):
                        compressed_bars = (
                            bar_mask[b] & curr_mask[b]
                        ).nonzero(as_tuple=False).squeeze(1)
                        if compressed_bars.numel() == 0:
                            continue
                        for bar_pos in compressed_bars.tolist():
                            bar_n = bar_index_per_tok[b, bar_pos].item()
                            tok_mask = (bar_index_per_tok[b] == bar_n)
                            teacher_force = (torch.rand(1, device=tgt.device).item() < tf_ratio)
                            if not teacher_force and pred_embs is not None:
                                tgt_query[b, tok_mask] = pred_embs[b, tok_mask].to(tgt.dtype)
                            # teacher_force: tgt_query already has summary_embed_dense (GT)


        # com_t cascade: L1's WindowCA com_t → L3 time_center; L3's → L5 time_center.
        last_com_t: Optional[torch.Tensor] = None  # [B, S, 1]
        # com_f cascade: L1's WindowCA com_f → L3 freq_center; L3's → L5 freq_center.
        last_com_f: Optional[torch.Tensor] = None  # [B, S, 1]

        # CIF pointer: current fire slot index for MambaSALayer windowed CA.
        # Shape [B] LongTensor for inference (starts at 0);
        # Shape [B, S] LongTensor for training (per-token alignment from build_cif_alignment).
        if self.cif is not None and input_ids is not None:
            cif_ptr = build_cif_alignment(input_ids)  # [B, S]
        else:
            cif_ptr = torch.zeros(B, dtype=torch.long, device=output.device)  # [B]

        for i, layer in enumerate(self.layers):
            layer_state = past_states[i] if past_states else None
            layer_value_cache = value_cache_list[i] if value_cache_list else None

            # Determine what to pass to this layer.
            layer_window_center = None
            layer_freq_center_in = None
            layer_guidance_bounds = None

            if isinstance(layer, SAWindowCALayer):
                if layer.bar_token_id is not None:
                    # L1: Start with dense_com_t from BarMamba
                    layer_window_center = dense_com_t
                    layer_guidance_bounds = guidance_bounds
                    # L1 freq_center starts at 0.5 (no prior)
                else:
                    # L3, L5: use cascaded time/freq centers from previous window_ca layers.
                    layer_freq_center_in = last_com_f
                    # Time cascade: L3/L5 directly use previous layer's com_t_wca (audio-grounded refinement).
                    layer_window_center = last_com_t

            elif isinstance(layer, MambaWindowCALayer):
                # Use BarMamba's dense_com_t if this is the first CA layer (last_com_t is None)
                # Otherwise cascade from the previous CA layer.
                layer_freq_center_in = last_com_f
                layer_window_center = last_com_t if last_com_t is not None else dense_com_t

            if use_cache:
                layer_kwargs = dict(
                    tgt_pos=tgt_pos,
                    past_state=layer_state,
                    use_cache=True,
                    value_cache=layer_value_cache,
                    time_center_in=layer_window_center,
                    freq_center_in=layer_freq_center_in,
                    guidance_bounds=layer_guidance_bounds,
                    bar_mask=bar_mask,
                )
                if isinstance(layer, (MambaFullCALayer, MambaCIFLayer, MambaSALayer)):
                    layer_kwargs['tgt_query'] = tgt_query
                    if acoustic_embs_dense is not None:
                        layer_kwargs['acoustic_emb_dense'] = acoustic_embs_dense
                    if isinstance(layer, MambaSALayer):
                        layer_kwargs['cif_ptr']   = cif_ptr
                        layer_kwargs['cif_lazy']  = cif_lazy
                output, new_state = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    **layer_kwargs,
                )
                new_states.append(new_state)
            elif self.training and self.gradient_checkpointing:
                _guidance = layer_guidance_bounds
                _freq_in = layer_freq_center_in
                _is_window = isinstance(layer, SAWindowCALayer)
                _is_cif_ca = isinstance(layer, (MambaFullCALayer, MambaCIFLayer, MambaSALayer))
                _is_mamba_sa = isinstance(layer, MambaSALayer)
                _tgt_query = tgt_query if _is_cif_ca else None
                _acoustic = acoustic_embs_dense if _is_cif_ca else None
                _cif_ptr = cif_ptr if _is_mamba_sa else None
                _cif_lazy = cif_lazy if _is_mamba_sa else None
                def _layer_fn(out, mem, gb, time_in, freq_in, tq, acou, cp,
                              _l=layer, _sp=spatial_shapes, _lsi=level_start_index,
                              _vr=valid_ratios, _tp=tgt_pos, _iw=_is_window,
                              _bm=bar_mask, _imfc=_is_cif_ca, _ims=_is_mamba_sa,
                              _cl=_cif_lazy):
                    kwargs = dict(
                        tgt_pos=_tp,
                        time_center_in=time_in,
                        freq_center_in=freq_in,
                        guidance_bounds=gb,
                        bar_mask=_bm,
                    )
                    if _imfc:
                        kwargs['tgt_query'] = tq
                        if acou is not None:
                            kwargs['acoustic_emb_dense'] = acou
                        if cp is not None:
                            kwargs['cif_ptr'] = cp
                        if _ims and _cl is not None:
                            kwargs['cif_lazy'] = _cl
                    return _l(out, mem, _sp, _lsi, _vr, **kwargs)
                output = grad_checkpoint(_layer_fn, output, memory,
                                        _guidance, layer_window_center, _freq_in,
                                        _tgt_query, _acoustic, _cif_ptr,
                                        use_reentrant=False)
            else:
                layer_kwargs = dict(
                    time_center_in=layer_window_center,
                    freq_center_in=layer_freq_center_in,
                    guidance_bounds=layer_guidance_bounds,
                    bar_mask=bar_mask,
                )
                if isinstance(layer, MambaMonoAttnLayer):
                    layer_kwargs['onset_logit_bias'] = p_onset
                elif isinstance(layer, (MambaFullCALayer, MambaCIFLayer, MambaSALayer)):
                    layer_kwargs['tgt_query'] = tgt_query
                    if acoustic_embs_dense is not None:
                        layer_kwargs['acoustic_emb_dense'] = acoustic_embs_dense
                    if isinstance(layer, MambaSALayer):
                        layer_kwargs['cif_ptr']  = cif_ptr
                        layer_kwargs['cif_lazy'] = cif_lazy
                output = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    **layer_kwargs,
                )

            # Update temporal state after each layer.
            if isinstance(layer, SAWindowCALayer):
                # Always update last_com_t/f from every window_ca layer (L1 → L3 → L5 cascade).
                if layer._cached_com_t_wca is not None:
                    last_com_t = layer._cached_com_t_wca  # [B, S, 1]
                if layer._cached_com_f is not None:
                    last_com_f = layer._cached_com_f  # [B, S, 1]
                if layer.bar_token_id is not None:
                    self._cached_guidance_loss = layer._cached_guidance_loss

            elif isinstance(layer, MambaWindowCALayer):
                # Same cascade as SAWindowCALayer (no guidance loss for this layer type).
                if layer._cached_com_t_wca is not None:
                    last_com_t = layer._cached_com_t_wca  # [B, S, 1]
                if layer._cached_com_f is not None:
                    last_com_f = layer._cached_com_f  # [B, S, 1]

            elif isinstance(layer, SAFullCALayer):
                # Legacy support if full_ca layers still exist in config
                self._cached_guidance_loss = layer._cached_guidance_loss

        output = self.norm(output)


        if use_cache:
            return output, new_states
        return output
