"""
Clef Decoder
============

Hybrid decoder: Mamba2 + Self-Attention + Cross-Attention.

Layer pattern (clef_piano_base, 6 layers):
  L1: window_ca  — SA + Bar Full-Attn on onset_1d + Window CA on S2+S3
  L2: mamba_only — sequential writing
  L3: window_ca  — SA + Window CA on S0+S1  (t from L1 DSNT, f from L1 DSNT)
  L4: mamba_only — sequential writing
  L5: window_ca  — SA + Window CA on L0+L1  (t from L3 DSNT, f from L3 DSNT)
  L6: mamba_only — sequential writing

Key design:
- <bar> tokens in L1 do full attention over onset_1d (Octopus F-pooled, [B,T_octopus,C]).
  bar_center (carry-forward of bar full-attn com_t) is the initial t-center for L1 Window CA.
  This is Zeng 2024's bar_hidden state adapted to flat AR decoding.
  Positional bias (-scale * time_pos) ensures initial CoM near 0 (not 0.5), respecting BarGRU causality.
- Coarse-to-fine cascade (both t and f): L1 DSNT → L3 center → L3 DSNT → L5 center.
  L1 is the only layer that uses bar_center directly; L3/L5 use the refined DSNT com_t.
- guidance_bounds supervises time_center (note tokens + <bar> tokens) to fall within measure range.
  For <bar> tokens, this directly supervises BarGRU com_t after the override in NoteGRU.
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


class BarGRU(nn.Module):
    """Hierarchical bar-level GRU for cross-bar time tracking (Zeng-inspired).

    Architecture:
      1. note_gru: encodes all tokens within a bar into a bar_summary vector.
      2. bar_gru:  updates across bars using bar_summary → h_bar.
      3. query_proj: maps h_bar to an attention query.
      4. FullAttn(query, onset_1d): audio-grounded bar center (com_t).

    Training: bar segments are processed with PackedSequence for parallelism.
    Scheduled sampling: each bar independently uses GT or predicted tokens
    (bar-level TF), and within a bar each token independently (note-level TF).

    Cache:
      _cached_com_t: [B, S, 1] — com_t at each <bar> position (zeros elsewhere).
      _cached_h_bar: [1, B, D_bar] — last bar_gru hidden state.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_bar: int = 256,
        onset_1d_channels: int = 32,
        input_dropout: float = 0.1,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_bar = d_bar
        self.n_heads = n_heads
        self.head_dim = d_bar // n_heads

        # note_gru: encode intra-bar token sequence → bar summary
        self.note_gru = nn.GRU(d_model, d_bar, batch_first=True)
        # bar_gru: update across bars
        self.bar_gru = nn.GRU(d_bar, d_bar, batch_first=True)
        # query projection: h_bar → attention query for onset_1d FullAttn
        self.query_proj = nn.Linear(d_bar, d_bar)
        # onset_1d projection: C → d_bar (K only; attention output is com_t, not V-aggregated)
        self.onset_k_proj = nn.Linear(onset_1d_channels, d_bar)

        # Positional bias: encourage attention toward earlier positions (causal prior)
        # Initial CoM should be near 0 (not 0.5) since BarGRU is sequential
        # scale=2.5 gives a mild early bias; learnable so model can adjust if needed
        self.pos_bias_scale = nn.Parameter(torch.tensor(2.5))

        self.input_dropout = nn.Dropout(input_dropout)

        # Cached state (set during training forward; updated incrementally during inference)
        self._cached_com_t: Optional[torch.Tensor] = None
        self._cached_h_bar: Optional[torch.Tensor] = None

    @staticmethod
    def _sinusoidal_pe(T: int, C: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sinusoidal positional encoding for onset_1d tokens [T, C]."""
        half = C // 2
        pos = torch.arange(T, device=device, dtype=dtype)
        div = torch.exp(
            torch.arange(0, half, device=device, dtype=dtype)
            * (-math.log(10000.0) / half)
        )
        pe = torch.zeros(T, C, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div.unsqueeze(0))
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div.unsqueeze(0))[:, :C - half]
        return pe

    def forward(
        self,
        token_embs: torch.Tensor,    # [B, S, D] full sequence embedding
        onset_1d: torch.Tensor,       # [B, T_oct, C_onset]
        bar_mask: torch.Tensor,       # [B, S] bool — True at <bar> positions
        tf_ratio: float = 1.0,        # teacher-forcing ratio for note_gru (1.0 = all GT)
        pred_embs: Optional[torch.Tensor] = None,  # [B, S, D] predicted embeddings for TF
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-bar com_t, final bar_gru hidden state, and per-position carried h_bar.

        Returns:
            com_t_all:    [B, S, 1]      — com_t at <bar> positions, 0 elsewhere
            h_bar_final:  [1, B, D_bar]  — final bar_gru hidden state (last bar)
            h_bar_carried:[B, S, D_bar]  — carry-forward of h_bar at each position;
                                           h_bar_carried[b, s] = h_bar after most recent bar ≤ s.
                                           Used by NoteGRU to reset from the *correct* bar's h_bar,
                                           avoiding future-leakage during teacher-forced training.
        """
        B, S, D = token_embs.shape
        device = token_embs.device
        dtype = token_embs.dtype

        # Pool onset_1d 8x → ~160ms resolution
        pool = 8
        onset_pooled = F.avg_pool1d(
            onset_1d.permute(0, 2, 1),
            kernel_size=pool, stride=pool, ceil_mode=True,
        ).permute(0, 2, 1)  # [B, T_oct//8, C_onset]
        T_oct = onset_pooled.shape[1]
        C_onset = onset_pooled.shape[2]

        # Add sinusoidal PE to onset_pooled (as K input)
        pe = self._sinusoidal_pe(T_oct, C_onset, device, dtype)
        onset_pooled_pe = onset_pooled + pe.unsqueeze(0)  # [B, T_oct, C_onset]

        # Project K for FullAttn (attention output is com_t, not V-aggregated)
        K = self.onset_k_proj(onset_pooled_pe)  # [B, T_oct, D_bar]

        bar_positions = bar_mask.nonzero(as_tuple=False)  # [N_bars_total, 2]

        com_t_all = torch.zeros(B, S, 1, device=device, dtype=dtype)
        h_bar_final = torch.zeros(1, B, self.d_bar, device=device, dtype=dtype)
        h_bar_scatter = torch.zeros(B, S, self.d_bar, device=device, dtype=dtype)

        if bar_positions.shape[0] == 0:
            self._cached_com_t = com_t_all
            self._cached_h_bar = h_bar_final
            return com_t_all, h_bar_final, h_bar_scatter

        # --- Fast path: tf_ratio == 1.0 (no scheduled sampling) ---
        # Use PackedSequence to process all bar segments across all batches in one GRU call.
        if tf_ratio >= 1.0 or pred_embs is None:
            self._forward_fast(
                token_embs, K, bar_mask, B, S, T_oct, device, dtype,
                com_t_all, h_bar_final, h_bar_scatter,
            )
        else:
            # Slow path: per-batch per-bar loop (needed for TF mixing)
            self._forward_slow(
                token_embs, pred_embs, K, bar_mask, tf_ratio, B, S, T_oct, device, dtype,
                com_t_all, h_bar_final, h_bar_scatter,
            )

        # Carry h_bar forward: each position gets the h_bar after the most recent bar.
        h_bar_carried = _carry_forward_h_bar(h_bar_scatter, bar_mask)  # [B, S, D_bar]

        self._cached_com_t = com_t_all
        self._cached_h_bar = h_bar_final
        return com_t_all, h_bar_final, h_bar_carried

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_com_t_batched(
        self,
        queries: torch.Tensor,   # [N, D_bar] — one query per bar
        K: torch.Tensor,         # [B, T_oct, D_bar]
        bar_b_idx: torch.Tensor, # [N] — which batch each bar belongs to
        T_oct: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute com_t for N bars in parallel using batched multi-head attention.

        Returns com_t: [N, 1]
        """
        N = queries.shape[0]
        H = self.n_heads
        d_h = self.head_dim

        # Project queries
        Q_proj = self.query_proj(queries)  # [N, D_bar]
        Q = Q_proj.view(N, H, d_h)         # [N, H, d_h]

        # Gather K for each bar's batch: [N, T_oct, D_bar]
        K_N = K[bar_b_idx]                         # [N, T_oct, D_bar]
        K_N = K_N.view(N, T_oct, H, d_h).permute(0, 2, 1, 3)  # [N, H, T_oct, d_h]

        # Scaled dot-product: [N, H, T_oct]
        scores = torch.einsum('nhd,nhtd->nht', Q, K_N) / math.sqrt(d_h)

        # Positional bias: encourage earlier positions
        time_pos = torch.linspace(0, 1, T_oct, device=device, dtype=dtype)
        pos_bias = -self.pos_bias_scale * time_pos  # [T_oct]
        scores = scores + pos_bias.unsqueeze(0).unsqueeze(0)  # broadcast [N, H, T_oct]

        attn_w = F.softmax(scores, dim=-1)  # [N, H, T_oct]
        # com_t: weighted mean of time positions
        com_t = (attn_w.mean(1) * time_pos).sum(-1, keepdim=True)  # [N, 1]
        return com_t

    def _forward_fast(
        self,
        token_embs: torch.Tensor,  # [B, S, D]
        K: torch.Tensor,           # [B, T_oct, D_bar]
        bar_mask: torch.Tensor,    # [B, S] bool
        B: int, S: int, T_oct: int,
        device: torch.device, dtype: torch.dtype,
        com_t_all: torch.Tensor,        # [B, S, 1] — written in-place
        h_bar_final: torch.Tensor,      # [1, B, D_bar] — written in-place
        h_bar_scatter: torch.Tensor,    # [B, S, D_bar] — written in-place
    ):
        """Fast path: no TF mixing. PackedSequence over all segments at once."""
        # Build per-batch bar info
        # bar_positions_b[b] = sorted list of bar token positions for batch b
        per_batch_bar_pos: List[torch.Tensor] = []
        for b in range(B):
            per_batch_bar_pos.append(bar_mask[b].nonzero(as_tuple=False).squeeze(1))  # [N_b]

        # --- Step 1: Encode all bar segments with note_gru (PackedSequence) ---
        # Each bar segment: tokens from (prev_bar+1) to (bar_pos) inclusive
        # We want the final hidden state of note_gru for each segment.
        # Collect all segments across all batches.
        all_segs: List[torch.Tensor] = []   # each: [seg_len, D]
        seg_lengths: List[int] = []
        seg_b_idx: List[int] = []           # which batch each segment belongs to
        seg_bar_idx: List[int] = []         # which bar within batch

        # Also need per-batch bar info for bar_gru
        per_batch_n_bars: List[int] = []
        per_batch_bar_pos_lists: List[List[int]] = []

        for b in range(B):
            bar_pos_b = per_batch_bar_pos[b]  # [N_b]
            N_b = bar_pos_b.shape[0]
            per_batch_n_bars.append(N_b)

            bar_pos_list = bar_pos_b.tolist()
            per_batch_bar_pos_lists.append(bar_pos_list)

            if N_b == 0:
                continue

            bar_starts = [0] + [int(p) + 1 for p in bar_pos_list[:-1]]
            bar_ends   = [int(p) for p in bar_pos_list]  # inclusive end = bar token pos

            for bar_i, (seg_s, seg_e) in enumerate(zip(bar_starts, bar_ends)):
                seg_len = seg_e - seg_s
                seg_b_idx.append(b)
                seg_bar_idx.append(bar_i)
                if seg_len > 0:
                    all_segs.append(token_embs[b, seg_s:seg_e])  # [seg_len, D]
                    seg_lengths.append(seg_len)
                else:
                    # Empty segment → use a dummy zero tensor of length 1
                    # Final hidden state of zeros is what we want (zero bar_summary).
                    all_segs.append(token_embs.new_zeros(1, token_embs.shape[-1]))
                    seg_lengths.append(0)  # mark as empty

        N_segs = len(all_segs)

        # Run note_gru on non-empty segments via PackedSequence
        nonempty_mask = [l > 0 for l in seg_lengths]
        nonempty_indices = [i for i, m in enumerate(nonempty_mask) if m]
        nonempty_segs = [all_segs[i] for i in nonempty_indices]
        nonempty_lens = [seg_lengths[i] for i in nonempty_indices]

        # bar_summaries[i] = [D_bar] final hidden state for segment i
        bar_summaries = token_embs.new_zeros(N_segs, self.d_bar)

        if nonempty_segs:
            # Sort by length descending (required by pack_padded_sequence)
            sorted_order = sorted(range(len(nonempty_lens)), key=lambda x: -nonempty_lens[x])
            sorted_segs_tensors = [nonempty_segs[i] for i in sorted_order]
            sorted_lens = [nonempty_lens[i] for i in sorted_order]

            # Use pad_sequence (no loop scatter → no CopySlices)
            from torch.nn.utils.rnn import pad_sequence as _pad_seq
            padded = _pad_seq(sorted_segs_tensors, batch_first=True)  # [N_ne, max_len, D]

            # PackedSequence + note_gru
            packed = pack_padded_sequence(
                padded,
                lengths=torch.tensor(sorted_lens, dtype=torch.long),
                batch_first=True,
                enforce_sorted=True,
            )
            _, h_n = self.note_gru(packed)  # h_n: [1, N_ne, D_bar]
            h_n = h_n.squeeze(0)             # [N_ne, D_bar]

            # Unsort
            N_ne = len(sorted_order)
            unsort_order = [0] * N_ne
            for sorted_i, orig_i in enumerate(sorted_order):
                unsort_order[orig_i] = sorted_i
            h_n_unsorted = h_n[torch.tensor(unsort_order, device=device)]  # [N_ne, D_bar]

            # Write back to bar_summaries (scalar indexing: no CopySlices)
            for ne_i, seg_i in enumerate(nonempty_indices):
                bar_summaries[seg_i] = h_n_unsorted[ne_i]

        # Apply input dropout
        bar_summaries = self.input_dropout(bar_summaries)  # [N_segs, D_bar]

        # --- Step 2: bar_gru sequentially per batch (but each batch is one GRU call) ---
        # Rearrange bar_summaries into per-batch tensors
        # seg index mapping: for batch b, bar i → segment index
        seg_global_idx: List[List[int]] = [[] for _ in range(B)]
        seg_ptr = 0
        for i in range(N_segs):
            b = seg_b_idx[i]
            seg_global_idx[b].append(seg_ptr)
            seg_ptr += 1

        # Build padded bar_summary tensor: [B, max_N_bars, D_bar]
        max_n_bars = max(per_batch_n_bars) if per_batch_n_bars else 0
        bar_sum_padded = token_embs.new_zeros(B, max_n_bars, self.d_bar)
        bar_n_bars_tensor = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            N_b = per_batch_n_bars[b]
            bar_n_bars_tensor[b] = N_b
            if N_b == 0:
                continue
            for bar_i, seg_i in enumerate(seg_global_idx[b]):
                bar_sum_padded[b, bar_i] = bar_summaries[seg_i]

        # Run bar_gru: pack across batches for efficiency
        if max_n_bars > 0:
            # Sort by n_bars descending (required by pack_padded_sequence)
            sorted_b_order = torch.argsort(bar_n_bars_tensor, descending=True)
            sorted_lens_b = bar_n_bars_tensor[sorted_b_order].tolist()
            bar_sum_sorted = bar_sum_padded[sorted_b_order]  # [B, max_N_bars, D_bar]

            # Remove batches with 0 bars (they appear last after sorting)
            n_nonzero = sum(1 for l in sorted_lens_b if l > 0)
            if n_nonzero > 0:
                bar_sum_nz = bar_sum_sorted[:n_nonzero]
                lens_nz = sorted_lens_b[:n_nonzero]
                orig_b_of_sorted = sorted_b_order[:n_nonzero]  # original b indices [n_nonzero]

                packed_bar = pack_padded_sequence(
                    bar_sum_nz,
                    lengths=torch.tensor(lens_nz, dtype=torch.long),
                    batch_first=True,
                    enforce_sorted=True,
                )
                bar_out_packed, h_bar_n = self.bar_gru(packed_bar)
                bar_out_padded, _ = pad_packed_sequence(bar_out_packed, batch_first=True)
                # bar_out_padded: [n_nonzero, max_bars_in_nz, D_bar]
                # h_bar_n:        [1, n_nonzero, D_bar]
                # bar_out_padded[i] corresponds to batch orig_b_of_sorted[i]

                for i in range(n_nonzero):
                    b = int(orig_b_of_sorted[i].item())
                    N_b = per_batch_n_bars[b]
                    h_bar_per_bar = bar_out_padded[i, :N_b]   # [N_b, D_bar]
                    h_bar_final[0, b] = h_bar_n[0, i]

                    # Write h_bar to h_bar_scatter at each bar token position
                    bar_pos_list = per_batch_bar_pos_lists[b]
                    for bar_i, bar_pos in enumerate(bar_pos_list):
                        h_bar_scatter[b, bar_pos] = h_bar_per_bar[bar_i]

        # --- Step 3: Attention — compute com_t for all bars in parallel ---
        # Gather queries from h_bar_scatter at bar positions
        all_bar_pos = bar_mask.nonzero(as_tuple=False)  # [N_bars_total, 2]
        if all_bar_pos.shape[0] > 0:
            b_idx_all = all_bar_pos[:, 0]
            s_idx_all = all_bar_pos[:, 1]
            queries = h_bar_scatter[b_idx_all, s_idx_all]  # [N_total, D_bar]

            com_t_vals = self._compute_com_t_batched(
                queries, K, b_idx_all, T_oct, device, dtype,
            )  # [N_total, 1]

            com_t_all[b_idx_all, s_idx_all] = com_t_vals

    def _forward_slow(
        self,
        token_embs: torch.Tensor,
        pred_embs: torch.Tensor,
        K: torch.Tensor,           # [B, T_oct, D_bar]
        bar_mask: torch.Tensor,    # [B, S] bool
        tf_ratio: float,
        B: int, S: int, T_oct: int,
        device: torch.device, dtype: torch.dtype,
        com_t_all: torch.Tensor,
        h_bar_final: torch.Tensor,
        h_bar_scatter: torch.Tensor,
    ):
        """Slow path: per-batch per-bar loop for scheduled sampling (tf_ratio < 1.0)."""
        for b in range(B):
            batch_bar_positions = bar_mask[b].nonzero(as_tuple=False).squeeze(1)  # [N_bars_b]
            if batch_bar_positions.shape[0] == 0:
                continue

            h_b = torch.zeros(1, 1, self.d_bar, device=device, dtype=dtype)

            bar_starts = torch.cat([
                bar_mask[b].new_zeros(1),
                batch_bar_positions[:-1] + 1,
            ])
            bar_ends = batch_bar_positions

            for bar_i, (seg_start, seg_end) in enumerate(zip(bar_starts.tolist(), bar_ends.tolist())):
                seg_start = int(seg_start)
                seg_end = int(seg_end)

                seg_len = seg_end - seg_start
                if seg_len > 0:
                    if torch.rand(1).item() < tf_ratio:
                        seg_emb = token_embs[b, seg_start:seg_end]
                    else:
                        note_tf_mask = torch.rand(seg_len, device=device) < tf_ratio
                        gt_seg = token_embs[b, seg_start:seg_end]
                        pred_seg = pred_embs[b, seg_start:seg_end]
                        seg_emb = torch.where(note_tf_mask.unsqueeze(1), gt_seg, pred_seg)

                    seg_emb = seg_emb.unsqueeze(0)  # [1, seg_len, D]
                    _, h_note = self.note_gru(seg_emb)
                    bar_summary = h_note.squeeze(0)  # [1, D_bar]
                else:
                    bar_summary = torch.zeros(1, self.d_bar, device=device, dtype=dtype)

                bar_summary_dropped = self.input_dropout(bar_summary)
                _, h_b = self.bar_gru(bar_summary_dropped.unsqueeze(0), h_b)

                # Attention
                query = self.query_proj(h_b.squeeze(0))  # [1, D_bar]
                Q = query.view(1, self.n_heads, 1, self.head_dim)
                K_b = K[b].view(T_oct, self.n_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)
                scores = torch.einsum('nhqd,nhtd->nhqt', Q, K_b) / math.sqrt(self.head_dim)
                time_pos = torch.linspace(0, 1, T_oct, device=device, dtype=dtype)
                pos_bias = -self.pos_bias_scale * time_pos
                scores = scores + pos_bias.view(1, 1, -1)
                attn_w = F.softmax(scores.squeeze(2), dim=-1)
                com_t_b = (attn_w.mean(1) * time_pos).sum(-1, keepdim=True)

                bar_pos = int(batch_bar_positions[bar_i].item())
                com_t_all[b, bar_pos] = com_t_b.to(dtype)
                h_bar_scatter[b, bar_pos] = h_b[0, 0]

            h_bar_final[0, b] = h_b[0, 0]


class NoteGRU(nn.Module):
    """Note-level GRU for shared time_center (read-before / write-after).

    - Read: time_center = sigmoid(time_head(h_note)) before token processing.
    - Write: h_note = GRU(token_emb, h_note) after token processing.
    - Reset: if previous token is <bar>, h_note = init_proj(h_bar).

    scan_sequence uses nn.GRU (not GRUCell) for GPU-efficient sequence processing.
    The sequence is split at bar boundaries; each segment is run as a single
    nn.GRU call. This reduces the Python loop from S iterations to N_bars+1
    iterations (~20-50x fewer kernel launches for typical piano pieces).
    """

    def __init__(
        self,
        d_model: int = 512,
        d_bar: int = 256,
        d_note: int = 256,
        input_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_note = d_note
        self.d_bar = d_bar

        self.input_proj = nn.Linear(d_model, d_note)
        self.init_proj = nn.Linear(d_bar, d_note)
        # nn.GRU instead of GRUCell: allows full-segment forward in one CUDA kernel.
        # scan_sequence uses this for training; step() uses GRUCell semantics via
        # calling gru with seq_len=1.
        self.note_gru = nn.GRU(d_note, d_note, batch_first=True)
        self.time_head = nn.Linear(d_note, 1)
        # Initialize time_head bias to start from ~0 (not 0.5)
        # sigmoid(-6) ≈ 0.0025, so time_center starts near 0 and increases with GRU updates
        nn.init.constant_(self.time_head.bias, -6.0)

        self.input_dropout = nn.Dropout(input_dropout)

    def scan_sequence(
        self,
        inputs: torch.Tensor,         # [B, S, D_model] token embeddings
        bar_mask: torch.Tensor,        # [B, S] bool
        h_bar_carried: torch.Tensor,   # [B, S, D_bar]
        com_t_all: torch.Tensor,       # [B, S, 1] com_t at <bar> positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scan over a full sequence, returning time_center and h_notes_all.

        time_center at position s uses h_note before update, with bar override.

        Vectorized strategy: split the sequence at bar boundaries and run ALL
        segments in a single PackedSequence GRU call.

        Key insight: h0 for segment i (i > 0) is init_proj(h_bar_carried[b, bar_pos_{i-1}]),
        which depends only on h_bar_carried (an input), NOT on GRU output.
        Therefore all segment h0s can be precomputed, and every segment is independent
        given its h0.  We collect all (b, seg_i) non-empty segments, sort by length,
        pad into [N_segs_total, max_len, D_note], set h_0 per segment, and run one
        nn.GRU call.  This reduces GRU kernel launches from ~B*N_bars to 1.

        time_center[b_k] is overridden with com_t_all[:, b_k] (bar-token override).
        time_center for all other positions = sigmoid(time_head(h_before_update)).
        """
        B, S, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        # Project all inputs at once: one big matmul instead of S small ones
        inputs_proj = self.input_dropout(self.input_proj(inputs))  # [B, S, D_note]

        h_before = torch.zeros(B, S, self.d_note, device=device, dtype=dtype)
        h_after  = torch.zeros(B, S, self.d_note, device=device, dtype=dtype)

        # Collect ALL non-empty segments across all batch items.
        # For each segment: (b, seg_s, seg_len, h0_vec [D_note])
        # h0 is precomputed: zeros for first segment, init_proj(h_bar_carried) for others.
        all_segs: List[Tuple[int, int, int]] = []  # (b, seg_s, seg_len)
        all_h0s: List[torch.Tensor] = []           # each: [D_note]

        for b in range(B):
            bar_pos_b = bar_mask[b].nonzero(as_tuple=False).squeeze(1)  # [N_bars]
            seg_starts_t = torch.cat([bar_pos_b.new_zeros(1), bar_pos_b + 1])
            seg_ends_t   = torch.cat([bar_pos_b + 1, bar_pos_b.new_full((1,), S)])
            N_segs_b = len(seg_starts_t)

            for i in range(N_segs_b):
                seg_s = int(seg_starts_t[i].item())
                seg_e = int(seg_ends_t[i].item())
                seg_len = seg_e - seg_s
                if seg_len == 0:
                    continue

                if i == 0:
                    h0_vec = inputs_proj.new_zeros(self.d_note)
                else:
                    bar_reset_pos = int(seg_ends_t[i - 1].item()) - 1
                    h0_vec = self.init_proj(h_bar_carried[b, bar_reset_pos])  # [D_note]

                all_segs.append((b, seg_s, seg_len))
                all_h0s.append(h0_vec)

        if all_segs:
            # Sort by seg_len descending (required by pack_padded_sequence)
            order = sorted(range(len(all_segs)), key=lambda i: -all_segs[i][2])
            sorted_segs = [all_segs[i] for i in order]
            sorted_lens = [all_segs[i][2] for i in order]
            sorted_h0s  = [all_h0s[i] for i in order]

            N_total  = len(sorted_segs)
            max_len  = sorted_lens[0]

            # Gather h0s: [1, N_total, D_note]
            h0_all = torch.stack(sorted_h0s, dim=0).unsqueeze(0)  # [1, N_total, D_note]

            # Pad inputs via pad_sequence (no loop slice → no CopySlices in backward)
            from torch.nn.utils.rnn import pad_sequence as _pad_seq
            segs_list = [inputs_proj[b, seg_s:seg_s + seg_len] for b, seg_s, seg_len in sorted_segs]
            padded = _pad_seq(segs_list, batch_first=True)  # [N_total, max_len, D_note]

            # Single PackedSequence GRU call over all segments
            packed = pack_padded_sequence(
                padded,
                lengths=torch.tensor(sorted_lens, dtype=torch.long),
                batch_first=True,
                enforce_sorted=True,
            )
            seg_out_packed, _ = self.note_gru(packed, h0_all)
            seg_out_padded, _ = pad_packed_sequence(seg_out_packed, batch_first=True)
            # seg_out_padded: [N_total, max_len, D_note]

            # Build flat indices for h_before and h_after to avoid in-place slice
            # assignments (which produce CopySlices nodes in backward).
            # Strategy: collect all (flat_position, source_row, source_col) tuples,
            # then do a single index_put_ on the flat [B*S, D_note] views.
            #
            # h_before at position (b, seg_s + k) = seg_out_padded[idx, k-1] for k >= 1
            #                                       = h0_b              for k == 0
            # h_after  at position (b, seg_s + k) = seg_out_padded[idx, k]

            # Collect flat indices and the corresponding values from seg_out_padded.
            # We build two lists of flat_idx (int) and two lists of (seg_idx, time_offset)
            # pointing into seg_out_padded so we can gather them in one tensor op.

            # before_flat_idx[j]: flat index b*S+s for j-th h_before position
            # before_src_idx[j]:  row in seg_out_padded (seg index)
            # before_src_t[j]:    time offset in seg_out_padded (-1 means use h0)
            before_flat_idx: List[int] = []
            before_src_seg:  List[int] = []
            before_src_t:    List[int] = []   # -1 → use h0_all[0, seg, :]

            after_flat_idx: List[int] = []
            after_src_seg:  List[int] = []
            after_src_t:    List[int] = []

            for idx, (b, seg_s, seg_len) in enumerate(sorted_segs):
                for k in range(seg_len):
                    flat = b * S + seg_s + k
                    # h_before
                    before_flat_idx.append(flat)
                    before_src_seg.append(idx)
                    before_src_t.append(k - 1)  # -1 means h0
                    # h_after
                    after_flat_idx.append(flat)
                    after_src_seg.append(idx)
                    after_src_t.append(k)

            # Build index tensors
            bf_idx = torch.tensor(before_flat_idx, dtype=torch.long, device=device)
            af_idx = torch.tensor(after_flat_idx,  dtype=torch.long, device=device)
            seg_t_before = torch.tensor(before_src_t, dtype=torch.long, device=device)  # may contain -1
            seg_t_after  = torch.tensor(after_src_t,  dtype=torch.long, device=device)
            seg_i_before = torch.tensor(before_src_seg, dtype=torch.long, device=device)
            seg_i_after  = torch.tensor(after_src_seg,  dtype=torch.long, device=device)

            # Gather h_before values:
            #   For k>0: seg_out_padded[seg_i, k-1, :]
            #   For k=0: h0_all[0, seg_i, :]   (h0_all shape: [1, N_total, D_note])
            is_h0_mask = (seg_t_before < 0)  # [M_total] — positions that use h0

            # Clamp to 0 so we can gather without OOB (h0 positions will be overwritten)
            seg_t_before_clamped = seg_t_before.clamp(min=0)
            # seg_out_padded: [N_total, max_len, D_note]
            # Gather: [M_total, D_note]
            h_before_vals = seg_out_padded[seg_i_before, seg_t_before_clamped]
            # Override h0 positions with the actual h0 vectors
            if is_h0_mask.any():
                h0_flat = h0_all[0]  # [N_total, D_note]
                h_before_vals = h_before_vals.clone()
                h_before_vals[is_h0_mask] = h0_flat[seg_i_before[is_h0_mask]]

            # Gather h_after values:
            h_after_vals = seg_out_padded[seg_i_after, seg_t_after]  # [M_total, D_note]

            # Write back via index_put_ on flat view (single CopySlices node for entire scatter)
            h_before_flat = h_before.view(B * S, self.d_note)
            h_after_flat  = h_after.view(B * S, self.d_note)
            h_before_flat = h_before_flat.index_put((bf_idx,), h_before_vals)
            h_after_flat  = h_after_flat.index_put((af_idx,), h_after_vals)
            h_before = h_before_flat.view(B, S, self.d_note)
            h_after  = h_after_flat.view(B, S, self.d_note)

        # time_center: read-before-write = sigmoid(time_head(h_before))
        time_center = torch.sigmoid(self.time_head(h_before))  # [B, S, 1]

        # Bar token override (shifted): the token AFTER a <bar> gets time_center = com_t
        # of that bar (audio-grounded center from BarGRU).
        # This matches the old GRUCell logic where position s checks bar_mask[s-1].
        if bar_mask.any():
            # prev_was_bar[b, s] = True iff bar_mask[b, s-1] is True
            prev_was_bar = torch.zeros_like(bar_mask)
            prev_was_bar[:, 1:] = bar_mask[:, :-1]  # shift right by 1

            # com_t to inject: com_t_all[b, s-1] at position s → shift right
            com_t_shifted = torch.zeros_like(com_t_all)
            com_t_shifted[:, 1:] = com_t_all[:, :-1]

            prev_was_bar_3d = prev_was_bar.unsqueeze(-1)  # [B, S, 1]
            time_center = torch.where(prev_was_bar_3d, com_t_shifted, time_center)

        return time_center, h_after

    def step(
        self,
        input_t: torch.Tensor,  # [B, D_model]
        h: torch.Tensor,        # [B, D_note]
    ) -> torch.Tensor:
        """Single-step update for inference."""
        x = self.input_dropout(self.input_proj(input_t))
        # nn.GRU with seq_len=1: equivalent to GRUCell
        h_out, _ = self.note_gru(x.unsqueeze(1), h.unsqueeze(0))
        return h_out.squeeze(1)


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
    """Perceiver IO beat tracker: builds audio_latent from S0+S1.

    Role: Encode-only — does NOT modify tgt.
    - Compresses active encoder levels (S0+S1) into M latent slots
    - Passes tgt through unchanged
    - audio_latent [B, M, D] cached in _cached_audio_latent for ClefDecoder
      and consumed by the downstream use_audio_ca SAWindowCALayer (L1)

    S0 (0.16s): octave/fifth relationships, tonality — best bar discriminator (probe)
    S1 (0.32s): beat perception, pitch register

    Architecture: 2 rounds of cross-attn (one per active level), shared weights.
    Perceiver IO (Jaegle et al., 2021).
    """

    def __init__(self, d_model=512, n_heads=8, ff_dim=2048, dropout=0.1,
                 n_latents=512, active_ca_levels=None, n_levels=6,
                 **kwargs):  # absorbs unused base_kwargs
        super().__init__()
        self.active_ca_levels = active_ca_levels  # e.g. [2, 3] for S0+S1
        self.n_levels = n_levels
        self.d_model = d_model

        # Learned latent queries [M, D] — content specialization per slot
        self.latent_queries = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        # Fixed sinusoidal temporal PE for latent slots [M, D].
        # Slot i has a prior "I represent time i/M of the audio."
        # Normalized to [0,1] → consistent with encoder token time_norm (col/W).
        # Fixed (not learned) so temporal ordering is always well-defined.
        self.register_buffer('latent_pe', self._make_sinusoidal_pe(
            torch.linspace(0, 1, n_latents), d_model
        ))

        # Shared cross-attention: latents attend to each active encoder level
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # FFN applied between rounds
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Output cache — read by ClefDecoder after forward()
        self._cached_audio_latent: Optional[torch.Tensor] = None

    @staticmethod
    def _make_sinusoidal_pe(t_norm: torch.Tensor, d_model: int) -> torch.Tensor:
        """Sinusoidal PE for normalized time positions [0, 1].

        Same formula as SAFullCALayer._sinusoidal_time_pe — ensures latent slot i
        and encoder token at col/W produce the same PE when t_norm matches.
        Scale by N so adjacent slots have distinct encodings.
        """
        N = t_norm.shape[0]
        half_d = d_model // 2
        position = t_norm.float() * N                              # [N]
        dim = torch.arange(half_d, device=t_norm.device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (dim / half_d))              # [D/2]
        angles = position.unsqueeze(1) * inv_freq.unsqueeze(0)    # [N, D/2]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)    # [N, D]

    def forward(
        self,
        tgt: torch.Tensor,              # [B, S, D] — passed through unchanged
        memory: torch.Tensor,           # [B, N_total, D]
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        tgt_pos=None,
        past_state=None,
        use_cache: bool = False,
        value_cache=None,
        audio_latent=None,              # unused (we produce it)
        window_center=None,             # unused
        guidance_bounds=None,           # unused
        onset_1d=None,                  # unused (no bar attention in Perceiver)
        input_ids=None,                 # unused
    ):
        """Build audio_latent by attending to active encoder levels, pass tgt through.

        Both latents and encoder KV tokens receive time PE so cross-attn can
        align by time position: latent i (time i/M) attends to encoder tokens
        at col/W ≈ i/M regardless of encoder level resolution.
        """
        B = tgt.shape[0]
        n_enc_levels = level_start_index.shape[0]

        levels = self.active_ca_levels if self.active_ca_levels else list(range(n_enc_levels))

        # Latent queries + fixed temporal PE: slot i biased toward time i/M
        latents = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]
        latents = latents + self.latent_pe.unsqueeze(0)               # [B, M, D]

        # One cross-attn round per active level (shared weights = Perceiver IO style).
        # Time PE added to K only (not V): Q (latent) finds correct time via PE,
        # V stays as pure audio content — same principle as SAFullCALayer.
        for lvl in levels:
            start = level_start_index[lvl].item()
            end = (level_start_index[lvl + 1].item()
                   if lvl + 1 < n_enc_levels else memory.shape[1])
            kv = memory[:, start:end, :]   # [B, N_lvl, D]

            # Build time_norm for this level: col/W ∈ [0, 1] per encoder token
            H_l = int(spatial_shapes[lvl, 0].item())
            W_l = int(spatial_shapes[lvl, 1].item())
            w_norm = (torch.arange(W_l, device=tgt.device, dtype=torch.float32)
                      / max(W_l - 1, 1))
            # Each row (freq bin) gets the same time_norm as its column
            time_norm = w_norm.unsqueeze(0).expand(H_l, -1).reshape(-1)  # [H_l*W_l]

            # Time PE on K only — latent slot i and encoder token at col/W≈i/M
            # share the same PE value, enabling time-aligned cross-attn
            time_pe = self._make_sinusoidal_pe(time_norm, self.d_model)  # [N_lvl, D]
            k = kv + time_pe.to(kv.dtype)

            latents2, _ = self.cross_attn(latents, k, kv)
            latents = self.norm1(latents + latents2)
            latents = self.norm2(latents + self.ffn(latents))

        self._cached_audio_latent = latents  # [B, M, D]

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
        freq_center_init = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)
        self._cached_window_center = (time_pred, freq_center_init)

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
        bar_gru_hidden_size: int = 256,       # d_bar for BarGRU
        bar_gru_input_dropout: float = 0.1,   # dropout on note_gru → bar_gru input
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
        self.bar_gru_hidden_size = bar_gru_hidden_size

        # Causal Self-Attention with RoPE
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.rope = RotaryPositionalEmbedding(self.sa_head_dim)

        # BarGRU provides cross-bar time tracking via note_gru + bar_gru + FullAttn.
        if bar_token_id is not None:
            # L1: build BarGRU (note_gru + bar_gru + audio-grounded FullAttn)
            self.bar_gru_module = BarGRU(
                d_model=d_model,
                d_bar=bar_gru_hidden_size,
                onset_1d_channels=onset_1d_channels,
                input_dropout=bar_gru_input_dropout,
                n_heads=n_heads,
            )
        else:
            self.bar_gru_module = None

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

        # Caches read by ClefDecoder after forward.
        # _cached_window_center: (time_center, freq_center) input to WindowCA.
        # _cached_com_f: [B, S, 1] com_f from this layer's WindowCA (passed to next layer as freq_center_in).
        # _cached_com_t_wca: [B, S, 1] com_t from this layer's WindowCA (time cascade to next layer).
        # _cached_bar_gru_h: [1, B, D_bar] last bar_gru hidden state (L1 only).
        # _cached_h_bar_carried: [B, S, D_bar] per-position h_bar (carry-forwarded, L1 only).
        # _cached_guidance_loss: optional hinge loss on bar_center at <bar> positions.
        self._cached_window_center: Optional[Tuple] = None
        self._cached_com_f: Optional[torch.Tensor] = None
        self._cached_com_t_wca: Optional[torch.Tensor] = None
        self._cached_bar_gru_h: Optional[torch.Tensor] = None
        self._cached_h_bar_carried: Optional[torch.Tensor] = None
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
        onset_1d=None,          # [B, T_octopus, C] — required for L1 (BarGRU)
        input_ids=None,         # [B, S] int — required for L1 to detect <bar> tokens
        time_center_in=None,    # [B, S, 1] from ClefDecoder (NoteGRU read + cascade)
        freq_center_in=None,    # [B, S, 1] com_f from previous layer's WindowCA (L3/L5 only)
        guidance_bounds=None,   # [B, S, 2] optional hinge supervision at <bar> positions
        tf_ratio: float = 1.0,  # teacher-forcing ratio for note_gru scheduled sampling
        pred_embs: Optional[torch.Tensor] = None,  # [B, S, D] for TF (optional)
        bar_mask: Optional[torch.Tensor] = None,   # [B, S] bool, precomputed in ClefDecoder
        com_t_all: Optional[torch.Tensor] = None,  # [B, S, 1] com_t at <bar> positions
        h_bar_final: Optional[torch.Tensor] = None,  # [1, B, D_bar]
        h_bar_carried: Optional[torch.Tensor] = None,  # [B, S, D_bar]
        # Legacy kwargs passed by ClefDecoder (ignored here)
        audio_latent=None,
    ):
        """SA + BarGRU (L1 only) + Window CA + FFN.

        BarGRU provides audio-grounded com_t at <bar> positions. Time centers
        are computed in ClefDecoder (NoteGRU read-before), then passed in.
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

        # 2. Determine bar info (com_t_all / h_bar) for guidance and caching.
        self._cached_guidance_loss = None
        self._cached_bar_gru_h = None
        self._cached_h_bar_carried = None

        if (com_t_all is None or h_bar_carried is None or h_bar_final is None) and self.bar_gru_module is not None:
            if onset_1d is not None and input_ids is not None:
                # L1 fallback: run BarGRU if precomputed values are not provided.
                bar_mask = (input_ids == self.bar_token_id)  # [B, S]
                com_t_all, h_bar_final, h_bar_carried = self.bar_gru_module(
                    token_embs=tgt,
                    onset_1d=onset_1d,
                    bar_mask=bar_mask,
                    tf_ratio=tf_ratio,
                    pred_embs=pred_embs,
                )

        if bar_mask is None:
            bar_mask = torch.zeros(B, S, dtype=torch.bool, device=tgt.device)
        if com_t_all is None:
            com_t_all = torch.zeros(B, S, 1, device=tgt.device, dtype=tgt.dtype)
        if h_bar_final is None:
            h_bar_final = torch.zeros(1, B, self.bar_gru_hidden_size,
                                      device=tgt.device, dtype=tgt.dtype)
        if h_bar_carried is None:
            h_bar_carried = torch.zeros(B, S, h_bar_final.shape[-1], device=tgt.device, dtype=tgt.dtype)

        self._cached_bar_gru_h = h_bar_final       # [1, B, D_bar]
        self._cached_h_bar_carried = h_bar_carried  # [B, S, D_bar]

        # Freq cascade: L1 starts at 0.5 (no prior); L3 uses L1's com_f; L5 uses L3's com_f.
        # com_f is the frequency center-of-mass from the previous layer's WindowCA attention
        # weights — audio-grounded, meaningful only when time is already aligned (NoteGRU handles that).
        if freq_center_in is not None:
            freq_center = freq_center_in  # [B, S, 1] from previous layer
        else:
            freq_center = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)

        # Store window center as cached output (used by ClefDecoder + sanity check)
        if time_center_in is None:
            time_center = torch.full((B, S, 1), 0.5, device=tgt.device, dtype=tgt.dtype)
        else:
            time_center = time_center_in
        self._cached_window_center = (time_center, freq_center)

        # Time center guidance loss: hinge loss on note tokens AND <bar> tokens
        # Supervises time_center to fall within correct measure range
        # <bar> tokens now receive guidance bounds → supervises BarGRU com_t (after override)
        # MUST come AFTER time_center is defined above
        if self.training and guidance_bounds is not None:
            lo = guidance_bounds[:, :, 0]  # [B, S] measure start
            hi = guidance_bounds[:, :, 1]  # [B, S] measure end
            valid = (lo >= 0)  # Supervise note tokens AND <bar> tokens (structural tokens get -1)
            if valid.any():
                tc = time_center.squeeze(-1)  # [B, S] time_center (NoteGRU base + BarGRU override at bars)
                loss_hinge = F.relu(lo - tc) + F.relu(tc - hi)
                self._cached_guidance_loss = loss_hinge[valid].mean()
        else:
            self._cached_guidance_loss = None
        self._cached_com_f = None
        self._cached_com_t_wca = None

        # 4. Window Cross-Attention
        tgt2, _com_t_wca, com_f_wca = self.window_ca(
            query=tgt,
            time_center=time_center,
            freq_center=freq_center,
            value=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            active_levels=self.active_ca_levels,
            kv_cache=value_cache,
        )
        # Cache com_f for ClefDecoder to pass to the next window_ca layer
        self._cached_com_f = com_f_wca  # [B, S, 1]
        self._cached_com_t_wca = _com_t_wca  # [B, S, 1]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 5. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        if use_cache:
            return tgt, (new_kv, None)
        return tgt


class ClefDecoder(nn.Module):
    """Hybrid decoder: Mamba2 + Self-Attention + Cross-Attention.

    Layer pattern (clef_piano_base, 6 layers):
      L1: window_ca  — SA + BarGRU (onset_1d) + Window CA on S2+S3
      L2: mamba_only — sequential writing
      L3: window_ca  — SA + Window CA on S0+S1
      L4: mamba_only — sequential writing
      L5: window_ca  — SA + Window CA on L0+L1
      L6: mamba_only — sequential writing

    Temporal responsibility (Global NoteGRU):
      L1 BarGRU computes h_bar + com_t (audio-grounded).
      NoteGRU provides shared time_center (read-before / write-after).
      Time cascade: L1 com_t_wca → L3, L3 com_t_wca → L5 (blend with base time_center).
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
        bar_token_id: Optional[int] = None,   # set for first window_ca (L1 BarGRU)
        onset_1d_channels: int = 32,          # C from Octopus F-pooled for bar attention
        decoder_layer_full_freq: Optional[List] = None,  # per-layer: True/False or List[int] of full-freq level IDs
        decoder_layer_cascade_com: Optional[List[bool]] = None,  # per-layer cascade CoM flag
        window_ca_use_checkpoint: bool = True,  # passed to all SAWindowCALayer instances
        # Bar-GRU redesign params (forwarded to SAWindowCALayer)
        bar_gru_hidden_size: int = 256,
        bar_gru_input_dropout: float = 0.1,
        note_gru_hidden_size: int = 256,
        note_gru_input_dropout: float = 0.1,
        **kwargs,              # absorb deprecated FluxAttention params (n_points_*, *_scale, etc.)
    ):
        super().__init__()

        if decoder_layer_types is None:
            decoder_layer_types = ['window_ca', 'mamba_only',
                                   'window_ca', 'mamba_only', 'window_ca', 'mamba_only']

        self.layer_types = decoder_layer_types
        self.bar_token_id = bar_token_id
        self.use_rope = use_rope
        self.bar_gru_hidden_size = bar_gru_hidden_size
        self.note_gru_hidden_size = note_gru_hidden_size
        self.note_gru_input_dropout = note_gru_input_dropout

        # Shared kwargs passed to all layer constructors (unused keys absorbed by **kwargs)
        base_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            ff_dim=ff_dim,
            dropout=dropout,
            rope_base=rope_base,
        )

        self.note_gru = NoteGRU(
            d_model=d_model,
            d_bar=bar_gru_hidden_size,
            d_note=note_gru_hidden_size,
            input_dropout=note_gru_input_dropout,
        )

        # The first window_ca layer gets bar_token_id for BarGRU.
        bar_ca_assigned = False

        self.layers = nn.ModuleList()
        for i, lt in enumerate(decoder_layer_types):
            ca_levels = decoder_layer_ca_levels[i] if decoder_layer_ca_levels else None
            if lt == 'perceiver':
                # Legacy: keep PerceiverLayer available if config still uses it
                self.layers.append(PerceiverLayer(
                    active_ca_levels=ca_levels,
                    **base_kwargs,
                ))
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
                # First window_ca: assign bar_token_id for bar full-attention
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
        self.gradient_checkpointing = False  # enabled via enable_gradient_checkpointing()

        # NoteGRU inference state.
        self._note_gru_h: Optional[torch.Tensor] = None  # [B, D_note]
        self._note_gru_prev_bar: Optional[torch.Tensor] = None  # [B] bool
        self._note_gru_prev_bar_com: Optional[torch.Tensor] = None  # [B, 1]
        self._note_gru_prev_bar_h: Optional[torch.Tensor] = None  # [B, D_bar]

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

        # Precompute BarGRU outputs (for NoteGRU read + bar reset)
        bar_mask = None
        com_t_all = None
        h_bar_final = None
        h_bar_carried = None
        if input_ids is not None and self.bar_token_id is not None:
            bar_mask = (input_ids == self.bar_token_id)

        if onset_1d is not None and input_ids is not None:
            # Find first SAWindowCALayer with bar_token_id to run BarGRU once
            for layer in self.layers:
                if isinstance(layer, SAWindowCALayer) and layer.bar_token_id is not None:
                    bar_mask = (input_ids == layer.bar_token_id)
                    com_t_all, h_bar_final, h_bar_carried = layer.bar_gru_module(
                        token_embs=tgt,
                        onset_1d=onset_1d,
                        bar_mask=bar_mask,
                        tf_ratio=tf_ratio,
                        pred_embs=pred_embs,
                    )
                    break

        if bar_mask is None:
            bar_mask = torch.zeros(B, S, dtype=torch.bool, device=output.device)
        if com_t_all is None:
            com_t_all = torch.zeros(B, S, 1, device=output.device, dtype=output.dtype)
        if h_bar_carried is None:
            h_bar_carried = torch.zeros(B, S, self.bar_gru_hidden_size,
                                        device=output.device, dtype=output.dtype)

        # NoteGRU: read time_center before decoding, write after decoding
        time_center_base = None
        if not use_cache:
            time_center_base, _ = self.note_gru.scan_sequence(
                inputs=tgt,
                bar_mask=bar_mask,
                h_bar_carried=h_bar_carried,
                com_t_all=com_t_all,
            )
        else:
            # Inference: compute time_center from cached NoteGRU state
            if self._note_gru_h is None:
                self._note_gru_h = torch.zeros(B, self.note_gru_hidden_size,
                                               device=output.device, dtype=output.dtype)
            if self._note_gru_prev_bar is None:
                self._note_gru_prev_bar = torch.zeros(B, dtype=torch.bool, device=output.device)
            if self._note_gru_prev_bar_h is None:
                self._note_gru_prev_bar_h = torch.zeros(B, self.bar_gru_hidden_size,
                                                        device=output.device, dtype=output.dtype)
            if self._note_gru_prev_bar_com is None:
                self._note_gru_prev_bar_com = torch.full((B, 1), 0.5,
                                                         device=output.device, dtype=output.dtype)

            if self._note_gru_prev_bar.any():
                h_reset = self.note_gru.init_proj(self._note_gru_prev_bar_h)
                self._note_gru_h = torch.where(self._note_gru_prev_bar.unsqueeze(1), h_reset, self._note_gru_h)

            time_center_base = torch.sigmoid(self.note_gru.time_head(self._note_gru_h)).unsqueeze(1)
            time_center_base = torch.where(
                self._note_gru_prev_bar.unsqueeze(1).unsqueeze(-1),
                self._note_gru_prev_bar_com.unsqueeze(1),
                time_center_base,
            )


        # com_t cascade: L1's WindowCA com_t → L3 time_center; L3's → L5 time_center.
        last_com_t: Optional[torch.Tensor] = None  # [B, S, 1]
        # com_f cascade: L1's WindowCA com_f → L3 freq_center; L3's → L5 freq_center.
        last_com_f: Optional[torch.Tensor] = None  # [B, S, 1]

        for i, layer in enumerate(self.layers):
            layer_state = past_states[i] if past_states else None
            layer_value_cache = value_cache_list[i] if value_cache_list else None

            # Determine what to pass to this layer.
            layer_onset_1d = None
            layer_input_ids = None
            layer_window_center = None
            layer_freq_center_in = None
            layer_guidance_bounds = None

            if isinstance(layer, PerceiverLayer):
                pass  # Legacy Perceiver: no bar_center / onset_1d input

            elif isinstance(layer, SAWindowCALayer):
                if layer.bar_token_id is not None:
                    # L1: receives onset_1d and input_ids, runs BarGRU internally if needed.
                    layer_onset_1d = onset_1d
                    layer_input_ids = input_ids
                    layer_guidance_bounds = guidance_bounds
                    # L1 freq_center starts at 0.5 (no prior)
                else:
                    # L3, L5: use cascaded time/freq centers from previous window_ca layers.
                    layer_freq_center_in = last_com_f

                # Time cascade: base NoteGRU time_center, then blend with com_t_wca from prior layers
                if time_center_base is not None:
                    if last_com_t is not None:
                        layer_window_center = 0.5 * last_com_t + 0.5 * time_center_base
                    else:
                        layer_window_center = time_center_base

            if use_cache:
                _extra = {}
                if isinstance(layer, SAWindowCALayer):
                    _extra = dict(tf_ratio=tf_ratio, pred_embs=pred_embs)
                output, new_state = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    tgt_pos=tgt_pos,
                    past_state=layer_state,
                    use_cache=True,
                    value_cache=layer_value_cache,
                    onset_1d=layer_onset_1d,
                    input_ids=layer_input_ids,
                    time_center_in=layer_window_center,
                    freq_center_in=layer_freq_center_in,
                    guidance_bounds=layer_guidance_bounds,
                    bar_mask=bar_mask,
                    com_t_all=com_t_all,
                    h_bar_final=h_bar_final,
                    h_bar_carried=h_bar_carried,
                    **_extra,
                )
                new_states.append(new_state)
            elif self.training and self.gradient_checkpointing:
                # Layer-level gradient checkpoint.
                # Use default-argument capture to avoid Python closure bug.
                _onset = layer_onset_1d
                _guidance = layer_guidance_bounds
                _freq_in = layer_freq_center_in
                _is_window = isinstance(layer, SAWindowCALayer)
                def _layer_fn(out, mem, onset, gb, time_in, freq_in,
                              _l=layer, _sp=spatial_shapes, _lsi=level_start_index,
                              _vr=valid_ratios, _tp=tgt_pos, _ids=layer_input_ids,
                              _tf=tf_ratio, _pe=pred_embs, _iw=_is_window,
                              _bm=bar_mask, _ct=com_t_all, _hbf=h_bar_final, _hbc=h_bar_carried):
                    _extra = dict(tf_ratio=_tf, pred_embs=_pe) if _iw else {}
                    return _l(out, mem, _sp, _lsi, _vr, tgt_pos=_tp,
                              onset_1d=onset, input_ids=_ids,
                              time_center_in=time_in, freq_center_in=freq_in, guidance_bounds=gb,
                              bar_mask=_bm, com_t_all=_ct, h_bar_final=_hbf, h_bar_carried=_hbc,
                              **_extra)
                output = grad_checkpoint(_layer_fn, output, memory,
                                        _onset, _guidance, layer_window_center, _freq_in,
                                        use_reentrant=False)
            else:
                _extra = dict(tf_ratio=tf_ratio, pred_embs=pred_embs) if isinstance(layer, SAWindowCALayer) else {}
                output = layer(
                    output, memory,
                    spatial_shapes, level_start_index, valid_ratios,
                    tgt_pos=tgt_pos,
                    onset_1d=layer_onset_1d,
                    input_ids=layer_input_ids,
                    time_center_in=layer_window_center,
                    freq_center_in=layer_freq_center_in,
                    guidance_bounds=layer_guidance_bounds,
                    bar_mask=bar_mask,
                    com_t_all=com_t_all,
                    h_bar_final=h_bar_final,
                    h_bar_carried=h_bar_carried,
                    **_extra,
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

            elif isinstance(layer, SAFullCALayer):
                # Legacy support if full_ca layers still exist in config
                self._cached_guidance_loss = layer._cached_guidance_loss

        output = self.norm(output)

        # NoteGRU write-after: update hidden with token embeddings
        if use_cache:
            if self._note_gru_h is None:
                self._note_gru_h = torch.zeros(B, self.note_gru_hidden_size,
                                               device=output.device, dtype=output.dtype)
            self._note_gru_h = self.note_gru.step(tgt[:, -1], self._note_gru_h)
            prev_was_bar = bar_mask[:, -1]
            self._note_gru_prev_bar = prev_was_bar
            if prev_was_bar.any():
                self._note_gru_prev_bar_h = h_bar_carried[:, -1, :]
                self._note_gru_prev_bar_com = com_t_all[:, -1, :]

        if use_cache:
            return output, new_states
        return output
