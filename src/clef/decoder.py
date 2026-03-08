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

        # Skip CA

        # FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

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
    
    Uses Mamba1 (not Mamba2) to support BPTT sequential mode, where Mamba is called
    with seqlen=1 per step. Mamba2's causal_conv1d kernel has a stride alignment
    constraint that fails at seqlen=1 with typical d_model values.
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
        use_gru: bool = False,       # use GRUCell instead of Mamba.step() for sequential mode
        tbptt_chunk_size: int = 256, # detach hidden state every N steps (0 = full BPTT)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ca_head_dim = d_model // n_heads
        self.active_ca_levels = active_ca_levels if active_ca_levels is not None else list(range(n_levels))
        self.ca_attn_dropout = dropout
        self.use_gru = use_gru
        self.tbptt_chunk_size = tbptt_chunk_size

        if use_gru:
            # GRUCell: input = cat(token, ca_out) = 2*d_model, hidden = d_model
            # Fully differentiable — all parameters receive gradients.
            self.gru_cell = nn.GRUCell(input_size=2 * d_model, hidden_size=d_model)
            self.dropout_gru = nn.Dropout(dropout)
            # No Mamba, norm1, dropout_mamba, or fusion_proj — unused in GRU path.
        else:
            # 1. Mamba (processes token history → produces context-rich state y)
            # Mirrors Zeng's GRU: y_t encodes everything seen up to t-1.
            # NOTE: Mamba1.step() does NOT support autograd through its SSM internals
            # (causal_conv1d_update is a CUDA kernel without backward). Only out_proj
            # receives gradients. Use use_gru=True for a fully-differentiable alternative.
            self.norm1 = nn.LayerNorm(d_model)
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.dropout_mamba = nn.Dropout(dropout)
            # Fusion projection: cat([token_t, ca_t]) [2D] → [D] → Mamba input
            self.fusion_proj = nn.Linear(2 * d_model, d_model)

        # 2. Cross-Attention (Q = y = Mamba/GRU output, K/V = Audio)
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

        # 4. Output projection: cat([h_t, ca_t]) [2D] → [D]
        # Keeps layer output shape consistent ([B, S, D]) so ClefDecoder.norm works.
        self.out_proj = nn.Linear(2 * d_model, d_model)



    def _precompute_kv(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        device: torch.device,
    ):
        """Precompute K, V and time PE from encoder memory (reused every step in sequential mode)."""
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
            w_norm = torch.arange(W_l, device=device, dtype=torch.float32) / max(W_l - 1, 1)
            time_norm_list.append(w_norm.unsqueeze(0).expand(H_l, -1).reshape(-1))

        memory_ca = torch.cat(memory_ca, dim=1)   # [B, N_kv, D]
        time_norm  = torch.cat(time_norm_list)     # [N_kv]
        N_kv = memory_ca.shape[1]
        B = memory.shape[0]

        kv = self.ca_kv_proj(memory_ca).reshape(B, N_kv, 2, self.n_heads, self.ca_head_dim)
        k_ca = kv[:, :, 0].permute(0, 2, 1, 3).contiguous()  # [B, H, N_kv, D_head]
        v_ca = kv[:, :, 1].permute(0, 2, 1, 3).contiguous()  # [B, H, N_kv, D_head]

        time_pe   = self._sinusoidal_time_pe(time_norm)
        time_pe_k = (time_pe.view(N_kv, self.n_heads, self.ca_head_dim)
                             .permute(1, 0, 2).unsqueeze(0).to(k_ca.dtype))
        k_ca = k_ca + time_pe_k

        return k_ca, v_ca  # [B, H, N_kv, D_head] each

    def _sdpa(self, query: torch.Tensor, k_ca: torch.Tensor, v_ca: torch.Tensor) -> torch.Tensor:
        """Single SDPA call given pre-projected query [B, S, D] and precomputed K, V."""
        B, S, D = query.shape
        q = self.ca_q_proj(self.norm2(query))
        q = q.reshape(B, S, self.n_heads, self.ca_head_dim).permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(
            q, k_ca, v_ca,
            dropout_p=self.ca_attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = self.ca_out_proj(out.permute(0, 2, 1, 3).reshape(B, S, D))
        return self.dropout_ca(out)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        past_state: Optional[Tuple] = None,
        use_cache: bool = False,
        encoder_hidden: Optional[torch.Tensor] = None,  # [B, 1, D] for sequential mode
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple]] | torch.Tensor:
        B, S, D = tgt.shape
        mamba_state = past_state[1] if past_state is not None else None

        # ── Sequential CA mode (Zeng-style, full BPTT) ────────────────────────
        # Enabled when encoder_hidden is provided.
        # Each step: ca_t = CA(h_{t-1}, encoder), h_t = Mamba(token_t + ca_t, state)
        # Mamba state accumulates across steps (like Zeng's GRU hidden state).
        # Gradient chain: loss_t → h_t → ca_t → prev_out → h_{t-1} → ... (full BPTT)
        #
        # Uses Mamba1.step() directly (no InferenceParams) so that:
        # (a) conv_state / ssm_state dtype always matches the input (BF16-safe), and
        # (b) gradient flows through the state tensors without any .copy_() detach.
        if encoder_hidden is not None:
            k_ca, v_ca = self._precompute_kv(memory, spatial_shapes, level_start_index, tgt.device)
            outputs = []

            if self.use_gru:
                # ── GRU sequential mode (fully differentiable) ────────────────
                # All GRUCell parameters receive gradients.
                # TBPTT: detach hidden state every tbptt_chunk_size steps to
                # bound gradient chain length (0 = full BPTT, not recommended
                # for sequences > 512 tokens due to vanishing gradients).
                h = encoder_hidden.squeeze(1)  # [B, D]

                for t in range(S):
                    if self.tbptt_chunk_size > 0 and t > 0 and t % self.tbptt_chunk_size == 0:
                        h = h.detach()

                    # CA: query = previous hidden state (audio-conditioned)
                    ca_t = self._sdpa(h.unsqueeze(1), k_ca, v_ca)  # [B, 1, D]

                    # GRU step: input = cat(token_t, ca_t)
                    gru_input = torch.cat([tgt[:, t, :], ca_t.squeeze(1)], dim=-1)  # [B, 2D]
                    h = self.gru_cell(gru_input, h)                                  # [B, D]
                    h = self.dropout_gru(h)

                    outputs.append(torch.cat([h.unsqueeze(1), ca_t], dim=-1))  # [B, 1, 2D]

            else:
                # ── Mamba sequential mode (broken gradients — kept for reference) ──
                # NOTE: Mamba1.step() has no backward for SSM internals. Only out_proj
                # receives gradients. Use use_gru=True for hypothesis validation.
                d_inner = self.mamba.d_model * self.mamba.expand
                conv_state: Optional[torch.Tensor] = None
                ssm_state:  Optional[torch.Tensor] = None
                prev_out = encoder_hidden  # [B, 1, D]

                for t in range(S):
                    token_t = tgt[:, t:t+1, :]
                    ca_t = self._sdpa(prev_out, k_ca, v_ca)

                    fused_input = torch.cat([token_t, ca_t], dim=-1)
                    proj = self.fusion_proj(fused_input)

                    mamba_input = self.norm1(proj)
                    if conv_state is None:
                        compute_dtype = (
                            torch.get_autocast_gpu_dtype()
                            if torch.is_autocast_enabled()
                            else mamba_input.dtype
                        )
                        conv_state = torch.zeros(B, d_inner, self.mamba.d_conv,  device=tgt.device, dtype=compute_dtype)
                        ssm_state  = torch.zeros(B, d_inner, self.mamba.d_state, device=tgt.device, dtype=compute_dtype)
                    mamba_out_t, conv_state, ssm_state = self.mamba.step(mamba_input, conv_state, ssm_state)
                    h_t = proj + self.dropout_mamba(mamba_out_t)

                    outputs.append(torch.cat([h_t, ca_t], dim=-1))
                    prev_out = h_t

            fused = torch.cat(outputs, dim=1)            # [B, S, 2D]
            out   = self.out_proj(fused)                  # [B, S, D]
            if use_cache:
                return out, (None, mamba_state)
            return out


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
        # GRU sequential mode for mamba_full_ca layers
        use_gru_sequential: bool = False,     # replace Mamba.step() with GRUCell (fully differentiable)
        tbptt_chunk_size: int = 256,          # detach GRU hidden every N steps (0 = full BPTT)
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
        self.use_gru_sequential = use_gru_sequential
        self.tbptt_chunk_size = tbptt_chunk_size

        # Shared kwargs passed to all layer constructors (unused keys absorbed by **kwargs)
        base_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            ff_dim=ff_dim,
            dropout=dropout,
            rope_base=rope_base,
        )

        # Build shared BarMamba if bar_token_id is present.
        # active_ca_level: use level 0 of the first CA layer (matches MambaFullCALayer).
        # Derived from decoder_layer_ca_levels so it stays in sync with the decoder config.
        if bar_token_id is not None:
            bar_mamba_level = 0
            if decoder_layer_ca_levels:
                for ca_lvls in decoder_layer_ca_levels:
                    if ca_lvls:
                        bar_mamba_level = ca_lvls[0]
                        break
            self.bar_mamba = BarMamba(
                d_model=d_model,
                n_heads=n_heads,
                active_ca_level=bar_mamba_level,
                n_levels=n_levels,
            )
        else:
            self.bar_mamba = None

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
                    use_gru=use_gru_sequential,
                    tbptt_chunk_size=tbptt_chunk_size,
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
        # MambaFullCALayer with use_gru=True has no self.mamba — skip those.
        mamba_idx = 0
        for layer in self.layers:
            if isinstance(layer, MambaFullCALayer) and layer.use_gru:
                continue
            if isinstance(layer, (MambaOnlyLayer, MambaFullCALayer, MambaWindowCALayer)):
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
        tgt_pos: Optional[torch.Tensor] = None,
        past_states: Optional[list] = None,
        use_cache: bool = False,
        value_cache_list: Optional[list] = None,
        guidance_bounds: Optional[torch.Tensor] = None,  # [B, S, 2] or None
        onset_1d: Optional[torch.Tensor] = None,         # [B, T_octopus, C] for bar attention
        input_ids: Optional[torch.Tensor] = None,        # [B, S] int token IDs
        tf_ratio: float = 1.0,                            # teacher-forcing ratio for note_gru
        pred_embs: Optional[torch.Tensor] = None,        # [B, S, D] predicted embeddings for TF
        encoder_hidden: Optional[torch.Tensor] = None,   # [B, 1, D] BiMamba final hidden for sequential CA
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


        # ── BarMamba (Universal Bar-Level Temporal Localizer) ────────────────
        # BarMamba computes com_t via DSNT attention on encoder memory at <bar> positions.
        # We then forward-fill com_t to all notes within the bar so they share the same
        # temporal center for Window Cross-Attention.
        # BarMamba also returns summary_embed_dense: Stage 1 bar summary embed for each
        # complete bar (bar_index < N_b), used as tgt_query for MambaFullCA.
        dense_com_t = None
        summary_embed_dense = None
        # True if any layer type uses BarMamba's com_t (full CA or window CA)
        has_bar_ca = any(isinstance(l, (MambaFullCALayer, MambaWindowCALayer)) for l in self.layers)

        if (hasattr(self, 'bar_mamba')
                and self.bar_mamba is not None
                and memory is not None
                and bar_mask is not None
                and bar_mask.any()):

            # 1. Compute sparse com_t [B, S, 1] and summary_embed_dense [B, S, D]
            sparse_com_t, summary_embed_dense = self.bar_mamba(
                y=tgt,
                memory=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                bar_mask=bar_mask,
                input_ids=input_ids,
            )

            # 2. Forward-fill com_t (step-and-hold) within each bar
            B, S = bar_mask.shape
            dense_com_t = sparse_com_t.clone()
            for b in range(B):
                b_mask = bar_mask[b]
                if b_mask.any():
                    # indices of <bar> tokens
                    bar_idx = b_mask.nonzero(as_tuple=False).squeeze(1).tolist()
                    bar_starts = bar_idx
                    bar_ends = bar_idx[1:] + [S]
                    
                    for start, end in zip(bar_starts, bar_ends):
                        # The <bar> token's com_t
                        val = sparse_com_t[b, start, 0]
                        # Fill subsequent non-bar tokens with this value
                        dense_com_t[b, start:end, 0] = val

        # ── Build tgt_query ──────────────────────────────────────────────────
        # bar_index_per_tok[b, s] = number of <bar> tokens seen up to and including s.
        # Complete bars: bar_index < N_b (N_b = total <bar> count for this sample).
        # Current bar:   bar_index == N_b (still being decoded → use raw sequential tgt).
        bar_index_per_tok = bar_mask.long().cumsum(dim=1)  # [B, S]

        if summary_embed_dense is not None and bar_mask.any():
            # For bars with a prior-bar summary (bar_index >= 2): CA query = summary embed.
            # Bar 1 (bar_index == 1) has no preceding summary → raw tgt.
            # Current bar (bar_index == N_b) still being decoded → raw tgt.
            tgt_query = tgt.clone()
            for b in range(B):
                n_bars_b = int(bar_mask[b].float().sum().item())  # N_b
                # summary_embed_dense is non-zero only at bar_index 2..N_b (bar 2 onwards).
                # bar 1 notes get no summary replacement; current bar also stays raw.
                complete_mask = (bar_index_per_tok[b] >= 2) & (bar_index_per_tok[b] < n_bars_b)
                if complete_mask.any():
                    tgt_query[b, complete_mask] = summary_embed_dense[b, complete_mask]
        else:
            tgt_query = output  # fallback: Mamba decode path (original behavior)

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
                if isinstance(layer, MambaFullCALayer):
                    layer_kwargs['tgt_query'] = tgt_query
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
                _is_mamba_full_ca = isinstance(layer, MambaFullCALayer)
                _tgt_query = tgt_query if _is_mamba_full_ca else None
                def _layer_fn(out, mem, gb, time_in, freq_in, tq,
                              _l=layer, _sp=spatial_shapes, _lsi=level_start_index,
                              _vr=valid_ratios, _tp=tgt_pos, _iw=_is_window,
                              _bm=bar_mask, _imfc=_is_mamba_full_ca):
                    kwargs = dict(
                        tgt_pos=_tp,
                        time_center_in=time_in,
                        freq_center_in=freq_in,
                        guidance_bounds=gb,
                        bar_mask=_bm,
                    )
                    if _imfc:
                        kwargs['tgt_query'] = tq
                    return _l(out, mem, _sp, _lsi, _vr, **kwargs)
                output = grad_checkpoint(_layer_fn, output, memory,
                                        _guidance, layer_window_center, _freq_in, _tgt_query,
                                        use_reentrant=False)
            else:
                layer_kwargs = dict(
                    tgt_pos=tgt_pos,
                    time_center_in=layer_window_center,
                    freq_center_in=layer_freq_center_in,
                    guidance_bounds=layer_guidance_bounds,
                    bar_mask=bar_mask,
                )
                if isinstance(layer, MambaFullCALayer):
                    layer_kwargs['tgt_query'] = tgt_query
                    layer_kwargs['encoder_hidden'] = encoder_hidden
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
