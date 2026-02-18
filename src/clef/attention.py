"""
Clef Attention
==============

WindowCrossAttention: dense window cross-attention for audio-to-score decoding.

For each decoder query, attends to all encoder tokens within a local
time × freq window centered at (time_center, freq_center) — a free
byproduct of SAFullCALayer's attention weights, requiring no extra parameters.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class WindowCrossAttention(nn.Module):
    """Dense window cross-attention as replacement for FluxAttention.

    For each decoder query, attends to all encoder tokens within a local
    time × freq window centered at (time_center, freq_center).

    time_center and freq_center are the center-of-mass of SAFullCALayer's
    attention weights over S2+S3 — a free byproduct that directly encodes
    where in the audio each decoder token aligned.

    Compared to deformable FluxAttention:
    - Dense within window: all window tokens contribute gradient
    - Robust: center error < window_half still succeeds
    - No learned reference points, no time_prior/freq_prior needed
    - Standard QK attention weights (not factored level/point weights)

    Memory design: K/V are never pre-projected for the full level.
    Instead, for each seq_chunk, window token indices are computed from
    (time_center, freq_center), raw encoder tokens are gathered, then
    key_proj/value_proj are applied only on those K_l tokens.

    Peak memory per level per chunk: O(seq_chunk_size × K_l × D)
    e.g. chunk=128, K_l=128, D=512 → 32 MiB  vs  750 MiB for Octopus full level.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_levels: int = 6,
        n_heads: int = 8,
        window_time: Union[int, List[int]] = 16,   # int (all levels) or List[int] (per level)
        window_freq: Union[int, List[int]] = 8,    # int (all levels) or List[int] (per level)
        seq_chunk_size: int = 512,  # process N_q in chunks to bound peak memory
    ):
        super().__init__()

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.seq_chunk_size = seq_chunk_size

        # Normalize to per-level lists
        if isinstance(window_time, int):
            self.window_time_per_level = [window_time] * n_levels
        else:
            self.window_time_per_level = list(window_time)
        if isinstance(window_freq, int):
            self.window_freq_per_level = [window_freq] * n_levels
        else:
            self.window_freq_per_level = list(window_freq)

        # Legacy scalar aliases (for any code that reads .window_time / .window_freq)
        self.window_time = self.window_time_per_level[0]
        self.window_freq = self.window_freq_per_level[0]

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0.)

    def _gather_window_tokens(
        self,
        value: torch.Tensor,         # [B, N_v, D]
        level_start_index: torch.Tensor,  # [L]
        tc_chunk: torch.Tensor,      # [B, C]  in [0, 1]
        fc_chunk: torch.Tensor,      # [B, C]  in [0, 1]
        lid: int,
        H_l: int,
        W_l: int,
    ) -> Tuple[torch.Tensor, int]:
        """Gather window tokens for one level using nearest-neighbour integer indexing.

        Single gather call (vs. 4 bilinear-corner calls). For a dense window
        (wt × wf tokens), rounding the centre to the nearest pixel shifts the
        window by ≤ 0.5 frames — negligible compared to window size.
        OOB positions (outside the feature map) are zeroed out.

        Returns:
            tokens:  [B, C, K_l, D]
            K_l:     int
        """
        B, C = tc_chunk.shape
        device = value.device

        wt = min(self.window_time_per_level[lid], W_l)
        wf = min(self.window_freq_per_level[lid], H_l) if H_l > 1 else 1
        K_l = wt * wf

        # Nearest-pixel centre (align_corners=False: px = x*W - 0.5, then round)
        tc_px = (tc_chunk * W_l - 0.5).round().long()   # [B, C]
        fc_px = (fc_chunk * H_l - 0.5).round().long()   # [B, C]

        # Window offsets centred around 0 (integer)
        t_offsets = torch.arange(wt, device=device) - wt // 2   # [wt]
        f_offsets = torch.arange(wf, device=device) - wf // 2   # [wf]

        # Grid positions: [B, C, wt] and [B, C, wf]
        tc_grid = tc_px.unsqueeze(-1) + t_offsets   # [B, C, wt]
        fc_grid = fc_px.unsqueeze(-1) + f_offsets   # [B, C, wf]

        # Outer product → [B, C, wf, wt] → flatten to [B, C, K_l]
        tc_flat = tc_grid[:, :, None, :].expand(B, C, wf, wt).reshape(B, C, K_l)
        fc_flat = fc_grid[:, :, :, None].expand(B, C, wf, wt).reshape(B, C, K_l)

        # OOB detection, then clamp for safe indexing
        oob = (tc_flat < 0) | (tc_flat >= W_l) | (fc_flat < 0) | (fc_flat >= H_l)
        tc_c = tc_flat.clamp(0, W_l - 1)
        fc_c = fc_flat.clamp(0, H_l - 1)

        level_offset = level_start_index[lid].item()
        flat_idx = level_offset + fc_c * W_l + tc_c           # [B, C, K_l]

        # Single gather: [B, C*K_l, D] → reshape [B, C, K_l, D]
        idx_exp = flat_idx.reshape(B, C * K_l).unsqueeze(-1).expand(B, C * K_l, self.d_model)
        tokens = value.gather(1, idx_exp).reshape(B, C, K_l, self.d_model)

        # Zero out OOB positions so they don't contribute to attention
        if oob.any():
            tokens = tokens.masked_fill(oob.unsqueeze(-1), 0.0)

        return tokens, K_l

    def _forward_chunk(
        self,
        q_chunk: torch.Tensor,           # [B, C, H, D_head]
        tc_chunk: torch.Tensor,          # [B, C]  in [0, 1]
        fc_chunk: torch.Tensor,          # [B, C]  in [0, 1]
        value: torch.Tensor,             # [B, N_v, D]
        levels: list,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        B: int,
        C: int,
    ) -> torch.Tensor:
        """Process one chunk of C decoder queries. Returns [B, C, D].

        For each level, gathers only the window tokens from the raw encoder
        sequence, projects them to K/V, then accumulates with online softmax.

        Peak memory per level: O(C × K_l × D) for the gathered tokens +
        O(C × K_l × H × D_head) for K/V after projection.
        """
        scale = math.sqrt(self.head_dim)
        q_exp = q_chunk.unsqueeze(-2)   # [B, C, H, 1, D_head]

        # Online softmax state
        out_acc = None
        lse_acc = None

        for lid in levels:
            H_l = int(spatial_shapes[lid][0])
            W_l = int(spatial_shapes[lid][1])

            # Gather bilinear-interpolated window tokens: [B, C, K_l, D]
            tokens, K_l = self._gather_window_tokens(
                value, level_start_index, tc_chunk, fc_chunk, lid, H_l, W_l
            )

            # Project gathered tokens to K/V: [B, C, K_l, D]
            k_gathered = self.key_proj(tokens)
            v_gathered = self.value_proj(tokens)

            # [B, C, K_l, H, D_head] → [B, C, H, K_l, D_head]
            k_s = k_gathered.view(B, C, K_l, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            v_s = v_gathered.view(B, C, K_l, self.n_heads, self.head_dim).permute(0, 1, 3, 2, 4)

            # scores: [B, C, H, 1, K_l]
            scores_l = torch.matmul(q_exp, k_s.transpose(-1, -2)) / scale

            # Online softmax accumulation
            lse_l = torch.logsumexp(scores_l, dim=-1, keepdim=True)  # [B, C, H, 1, 1]
            w_l = torch.exp(scores_l - lse_l)                        # [B, C, H, 1, K_l]
            out_l = torch.matmul(w_l, v_s).squeeze(-2)               # [B, C, H, D_head]

            if lse_acc is None:
                out_acc = out_l
                lse_acc = lse_l.squeeze(-1)                           # [B, C, H, 1]
            else:
                lse_l_sq = lse_l.squeeze(-1)
                lse_new = torch.logaddexp(lse_acc, lse_l_sq)
                alpha_prev = torch.exp(lse_acc - lse_new)
                alpha_curr = torch.exp(lse_l_sq - lse_new)
                out_acc = alpha_prev * out_acc + alpha_curr * out_l
                lse_acc = lse_new

        return out_acc.reshape(B, C, -1)  # [B, C, D]

    def forward(
        self,
        query: torch.Tensor,               # [B, N_q, D]
        time_center: torch.Tensor,         # [B, N_q, 1] in [0, 1]
        freq_center: torch.Tensor,         # [B, N_q, 1] in [0, 1]
        value: torch.Tensor,               # [B, N_v, D]
        spatial_shapes: torch.Tensor,      # [L, 2]
        level_start_index: torch.Tensor,   # [L]
        active_levels: Optional[List[int]] = None,
        kv_cache: Optional[list] = None,
    ) -> torch.Tensor:
        """Dense window cross-attention with chunked sequence processing.

        kv_cache: if provided (inference), list of (k_seq, v_seq, H_l, W_l)
        per level (None for inactive).  During training, kv_cache=None and
        K/V are computed on-the-fly per seq_chunk from the raw value sequence.
        """
        B, N_q, _ = query.shape

        q = self.query_proj(query)  # [B, N_q, D]
        q = q.view(B, N_q, self.n_heads, self.head_dim)

        levels = active_levels if active_levels is not None else list(range(self.n_levels))

        tc = time_center.squeeze(-1)   # [B, N_q]  in [0, 1]
        fc = freq_center.squeeze(-1)   # [B, N_q]  in [0, 1]

        if kv_cache is not None:
            # Inference path: use pre-projected K/V sequences (no checkpointing needed)
            out = self._forward_with_cache(q, tc, fc, kv_cache, levels, spatial_shapes, B, N_q)
        elif N_q <= self.seq_chunk_size:
            out = self._run_chunk(q, tc, fc, value, levels, spatial_shapes, level_start_index, B, N_q)
        else:
            chunks = []
            for start in range(0, N_q, self.seq_chunk_size):
                end = min(start + self.seq_chunk_size, N_q)
                C = end - start
                out_c = self._run_chunk(
                    q[:, start:end],
                    tc[:, start:end],
                    fc[:, start:end],
                    value, levels, spatial_shapes, level_start_index, B, C,
                )
                chunks.append(out_c)
            out = torch.cat(chunks, dim=1)

        return self.output_proj(out)

    def _run_chunk(
        self,
        q_chunk: torch.Tensor,
        tc_chunk: torch.Tensor,
        fc_chunk: torch.Tensor,
        value: torch.Tensor,
        levels: list,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        B: int,
        C: int,
    ) -> torch.Tensor:
        if not self.training:
            return self._forward_chunk(
                q_chunk, tc_chunk, fc_chunk,
                value, levels, spatial_shapes, level_start_index, B, C,
            )
        # Gradient checkpointing: recompute _forward_chunk on backward instead of
        # storing all intermediate activations (K/V projections, gathered tokens).
        # Non-tensor args (levels, B, C) captured by closure.
        def fn(q_chunk, tc_chunk, fc_chunk, value, spatial_shapes, level_start_index):
            return self._forward_chunk(
                q_chunk, tc_chunk, fc_chunk,
                value, levels, spatial_shapes, level_start_index, B, C,
            )
        return grad_checkpoint(
            fn,
            q_chunk, tc_chunk, fc_chunk, value, spatial_shapes, level_start_index,
            use_reentrant=False,
        )

    def _forward_with_cache(
        self,
        q: torch.Tensor,              # [B, N_q, H, D_head]
        tc: torch.Tensor,             # [B, N_q]  in [0, 1]
        fc: torch.Tensor,             # [B, N_q]  in [0, 1]
        kv_cache: list,
        levels: list,
        spatial_shapes: torch.Tensor,
        B: int,
        N_q: int,
    ) -> torch.Tensor:
        """Inference path: K/V already projected, use bilinear gather from cache seqs."""
        scale = math.sqrt(self.head_dim)

        if N_q <= self.seq_chunk_size:
            return self._cache_chunk(q, tc, fc, kv_cache, levels, spatial_shapes, B, N_q)

        chunks = []
        for start in range(0, N_q, self.seq_chunk_size):
            end = min(start + self.seq_chunk_size, N_q)
            C = end - start
            chunks.append(self._cache_chunk(
                q[:, start:end], tc[:, start:end], fc[:, start:end],
                kv_cache, levels, spatial_shapes, B, C,
            ))
        return torch.cat(chunks, dim=1)

    def _cache_chunk(
        self,
        q_chunk: torch.Tensor,    # [B, C, H, D_head]
        tc_chunk: torch.Tensor,   # [B, C]
        fc_chunk: torch.Tensor,   # [B, C]
        kv_cache: list,
        levels: list,
        spatial_shapes: torch.Tensor,
        B: int,
        C: int,
    ) -> torch.Tensor:
        """One chunk of inference forward using cached K/V sequences."""
        scale = math.sqrt(self.head_dim)
        q_exp = q_chunk.unsqueeze(-2)

        out_acc = None
        lse_acc = None

        for lid in levels:
            k_seq, v_seq, H_l, W_l = kv_cache[lid]  # [B, N_l, D] each

            wt = min(self.window_time_per_level[lid], W_l)
            wf = min(self.window_freq_per_level[lid], H_l) if H_l > 1 else 1
            K_l = wt * wf

            # Gather bilinear-interpolated K/V from projected sequences
            # Reuse _gather_window_tokens logic but on projected seqs

            t_offsets = torch.arange(wt, device=q_chunk.device, dtype=q_chunk.dtype) - wt / 2.0 + 0.5
            f_offsets = torch.arange(wf, device=q_chunk.device, dtype=q_chunk.dtype) - wf / 2.0 + 0.5

            tc_px = tc_chunk * W_l - 0.5
            fc_px = fc_chunk * H_l - 0.5

            tc_grid = tc_px.unsqueeze(-1) + t_offsets
            fc_grid = fc_px.unsqueeze(-1) + f_offsets

            tc_flat = tc_grid[:, :, None, :].expand(B, C, wf, wt).reshape(B, C, K_l)
            fc_flat = fc_grid[:, :, :, None].expand(B, C, wf, wt).reshape(B, C, K_l)

            x0 = tc_flat.floor().long()
            y0 = fc_flat.floor().long()
            x1 = x0 + 1
            y1 = y0 + 1
            wa = tc_flat - x0.float()
            wb = fc_flat - y0.float()

            w00 = (1 - wa) * (1 - wb)
            w10 = wa       * (1 - wb)
            w01 = (1 - wa) * wb
            w11 = wa       * wb

            def gather_seq(seq, xi, yi, weight):
                # seq: [B, N_l, D], N_l = H_l * W_l
                valid = ((xi >= 0) & (xi < W_l) & (yi >= 0) & (yi < H_l)).float()
                xi_c = xi.clamp(0, W_l - 1)
                yi_c = yi.clamp(0, H_l - 1)
                flat_idx = yi_c * W_l + xi_c   # [B, C, K_l]
                tok = seq.gather(
                    1,
                    flat_idx.reshape(B, C * K_l).unsqueeze(-1).expand(B, C * K_l, seq.shape[-1])
                ).reshape(B, C, K_l, seq.shape[-1])
                w = (weight * valid).unsqueeze(-1)
                return tok * w

            def gather_kv(seq, xi, yi, weight):
                # seq: [B, N_l, D]
                D = seq.shape[-1]
                tok = gather_seq(seq, xi, yi, weight)   # [B, C, K_l, D]
                return tok.view(B, C, K_l, self.n_heads, self.head_dim)

            k_raw = (gather_kv(k_seq, x0, y0, w00)
                   + gather_kv(k_seq, x1, y0, w10)
                   + gather_kv(k_seq, x0, y1, w01)
                   + gather_kv(k_seq, x1, y1, w11))   # [B, C, K_l, H, D_head]
            v_raw = (gather_kv(v_seq, x0, y0, w00)
                   + gather_kv(v_seq, x1, y0, w10)
                   + gather_kv(v_seq, x0, y1, w01)
                   + gather_kv(v_seq, x1, y1, w11))

            k_s = k_raw.permute(0, 1, 3, 2, 4)   # [B, C, H, K_l, D_head]
            v_s = v_raw.permute(0, 1, 3, 2, 4)

            scores_l = torch.matmul(q_exp, k_s.transpose(-1, -2)) / scale
            lse_l = torch.logsumexp(scores_l, dim=-1, keepdim=True)
            w_l = torch.exp(scores_l - lse_l)
            out_l = torch.matmul(w_l, v_s).squeeze(-2)

            if lse_acc is None:
                out_acc = out_l
                lse_acc = lse_l.squeeze(-1)
            else:
                lse_l_sq = lse_l.squeeze(-1)
                lse_new = torch.logaddexp(lse_acc, lse_l_sq)
                alpha_prev = torch.exp(lse_acc - lse_new)
                alpha_curr = torch.exp(lse_l_sq - lse_new)
                out_acc = alpha_prev * out_acc + alpha_curr * out_l
                lse_acc = lse_new

        return out_acc.reshape(B, C, -1)

    def compute_kv_cache(
        self,
        value: torch.Tensor,              # [B, N_v, D]
        spatial_shapes: torch.Tensor,     # [L, 2]
        level_start_index: torch.Tensor,  # [L]
        active_levels: Optional[List[int]] = None,
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]]:
        """Pre-compute projected K/V sequences for inference.

        Returns list of length n_levels: inactive entries are None,
        active entries are (k_seq, v_seq, H_l, W_l) with k_seq/v_seq [B, N_l, D].

        Only called during inference (prepare_value_cache). During training,
        K/V are computed on-the-fly in _forward_chunk to avoid storing full-level
        projections (e.g. Octopus L0 at 384k tokens = 750 MiB K+V for 4-min chunks).
        """
        B = value.shape[0]
        levels_to_compute = active_levels if active_levels is not None else list(range(self.n_levels))

        active_starts = [level_start_index[lid].item() for lid in levels_to_compute]
        active_ends = [
            active_starts[i] + int(spatial_shapes[lid][0]) * int(spatial_shapes[lid][1])
            for i, lid in enumerate(levels_to_compute)
        ]

        index_tensors = [torch.arange(s, e, device=value.device) for s, e in zip(active_starts, active_ends)]
        active_indices = torch.cat(index_tensors)
        value_active = value[:, active_indices, :]

        k_active = self.key_proj(value_active)
        v_active = self.value_proj(value_active)

        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]] = [None] * self.n_levels
        offset = 0
        for lid in levels_to_compute:
            H_l = int(spatial_shapes[lid][0])
            W_l = int(spatial_shapes[lid][1])
            n_tokens = H_l * W_l
            kv_cache[lid] = (
                k_active[:, offset:offset + n_tokens, :],
                v_active[:, offset:offset + n_tokens, :],
                H_l, W_l,
            )
            offset += n_tokens

        return kv_cache
