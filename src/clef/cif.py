"""
CIF Module — Continuous Integrate-and-Fire
==========================================

Maps encoder frames [B, T, D] → per-token acoustic embeddings [B, N, D].

References:
    Dong & Xu 2020, "CIF: Continuous Integrate-and-Fire for End-to-End
    Speech Recognition", ICASSP 2020.

Design:
    Weight predictor:
        DepthwiseConv1d(kernel=3, groups=d_model) → LayerNorm → Linear(d_model, 1) → Sigmoid
        Each frame outputs α ∈ (0, 1) = "how much of a token's info does this frame carry"

    CIF integration:
        Accumulate α * h_t. When accumulator ≥ threshold=1.0, fire and emit
        one acoustic embedding. Remainder carries over to the next integration.

    Training: batch mode (all frames available, teacher-forcing)
    Inference: run once on full encoder output → cache acoustic_embs → AR decodes sequentially

Sidechain compressor semantics:
    fused = ducked_tgt + acoustic_emb
    Audio (acoustic_emb) is the immutable base; tgt is ducked when they agree.
    Tiny model: acoustic_emb replaces full CA entirely (no WindowCrossAttention needed).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Structural token IDs excluded when computing acoustic quantity targets.
# These tokens are pure syntax/structure with no direct acoustic correspondence.
# pad=0, sos=1, eos=2, coc=3, continue=5, split=7, merge=8, wildcard=9
# <bar>=4 and nl=6 are INCLUDED (they mark musically meaningful boundaries).
# Schema tokens 10-35 are INCLUDED (key/time signature have acoustic correlates).
STRUCTURAL_IDS = frozenset({0, 1, 2, 3, 5, 7, 8, 9})


# Token IDs that trigger a CIF pointer advance (time step boundaries).
# <bar>=4 triggers advance on every occurrence EXCEPT the first one per sequence.
# <nl>=6 always triggers advance.
NL_TOKEN_ID  = 6
BAR_TOKEN_ID = 4


def compute_acoustic_target_lengths(
    input_ids: torch.Tensor,  # [B, S]
) -> torch.Tensor:
    """Compute per-batch CIF fire count target.

    Target = count(<bar>) + count(<nl>).
    - <bar>: fires at each barline (measure boundary).
    - <nl>: fires once per time step, advancing to the next emb.

    Returns: [B] float tensor of target fire counts.
    """
    bar_counts = (input_ids == BAR_TOKEN_ID).sum(dim=1).float()
    nl_counts  = (input_ids == NL_TOKEN_ID).sum(dim=1).float()
    return bar_counts + nl_counts


def build_cif_alignment(
    input_ids: torch.Tensor,  # [B, S]
) -> torch.Tensor:
    """Build per-token CIF embedding index (scatter alignment).

    Rules:
      - ptr starts at 0.
      - <nl>: assigned emb[ptr] THEN ptr advances.
      - <bar>: assigned emb[ptr] THEN ptr advances.
      - All other tokens (<sos>, schema, notes, <coc>...): no advance.

    emb[0..N] = musical time steps (<nl> and <bar> positions).

    Returns: alignment [B, S] int64 — alignment[b, i] = emb index for token i.
    """
    B, S = input_ids.shape
    alignment = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)
    for b in range(B):
        ptr = 0
        for i in range(S):
            alignment[b, i] = ptr
            tid = input_ids[b, i].item()
            if tid == NL_TOKEN_ID or tid == BAR_TOKEN_ID:
                ptr += 1
    return alignment


class CIFModule(nn.Module):
    """Continuous Integrate-and-Fire: encoder frames → per-token acoustic embeddings.

    Decoupled dual-channel design (no FiLM, direct concatenation):
      - fire_signal (BiMamba @ 100fps): predicts α timing + provides temporal/duration features
      - acoustic_src (Swin S0 @ 12.5fps): provides pitch/harmonic features (upsampled 8x)
      - Both concatenated and projected to d_model

    Architecture:
        1. Weight predictor (from fire_signal):
               α = sigmoid(Linear(LayerNorm(DepthwiseConv1d(fire_signal))))  # α ∈ (0, 1)
        2. Dual-channel CIF integration:
               acoustic_temporal = CIF(fire_signal, α)  # [B, N, 128] duration/timing
               acoustic_pitch = CIF(acoustic_src↑8x, α)  # [B, N, 192] pitch/harmonic
        3. Concatenation:
               acoustic_cat = [acoustic_temporal, acoustic_pitch]  # [B, N, 320]
               (if S1 exists: [temporal, pitch, beat] → [B, N, 512])
        4. Project to d_model:
               out = Linear(concat_dim → d_model)(acoustic_cat)  # [B, N, d_model]
        5. Quantity loss: |Σα - N_structural| where N_structural = count(<bar>) + count(<nl>).

    forward() returns (acoustic_embs, alpha, quantity_loss).
    align_to_seq_len() pads/truncates acoustic_embs from [B, N, D] to [B, S, D]
    for teacher-forcing.
    """

    def __init__(
        self,
        fire_signal_dim: int = 128,     # BiMamba output dim (temporal timing)
        acoustic_dim: int = 192,         # Swin S0 output dim (pitch content, after downsample)
        beat_dim: int = 192,             # Swin S1 output dim (beat-aware, pre-downsample)
        d_model: int = 512,              # Final acoustic embedding dim (decoder expects)
        threshold: float = 1.0,          # Base threshold (overridden by dynamic if enabled)
        conv_kernel: int = 1,
        target_fires: int = 128,         # Target fire count (avg structural tokens per chunk)
        encoder_len: int = 3000,
        use_dynamic_threshold: bool = True,  # Paraformer dynamic threshold: β = Σα / ⌈Σα⌉
    ):
        super().__init__()
        self.fire_signal_dim = fire_signal_dim
        self.acoustic_dim = acoustic_dim
        self.beat_dim = beat_dim
        self.d_model = d_model
        self.threshold = threshold
        self.use_dynamic_threshold = use_dynamic_threshold

        padding = conv_kernel // 2
        # Depthwise conv on fire_signal: captures local temporal context for onset detection
        self.conv = nn.Conv1d(
            fire_signal_dim, fire_signal_dim,
            kernel_size=conv_kernel,
            padding=padding,
            groups=fire_signal_dim,
            bias=False,
        )
        self.weight_norm = nn.LayerNorm(fire_signal_dim)
        self.weight_proj = nn.Linear(fire_signal_dim, 1, bias=True)

        # Use PyTorch defaults: Kaiming uniform for weight, zeros for bias.
        # Kaiming init allows weight_proj to immediately discriminate onset vs non-onset
        # frames from BiMamba features, rather than starting uniform (zeros would push
        # quantity loss gradient uniformly across all frames).

        # === Zeng-style concatenation (no FiLM) ===
        # Concat temporal + pitch (+ beat), then project (like Zeng line 397-400)
        # Without S1: 128 + 192 = 320 → d_model
        # With S1:    128 + 192 + 192 = 512 → d_model
        self.combined_proj = nn.Linear(fire_signal_dim + acoustic_dim, d_model)  # 320 → d_model
        self.combined_proj_with_beat = nn.Linear(
            fire_signal_dim + acoustic_dim + beat_dim, d_model
        )  # 512 → d_model

    def forward(
        self,
        fire_signal: torch.Tensor,                           # [B, T_bi, D_bi] BiMamba output (timing)
        acoustic_src: torch.Tensor,                          # [B, T_sw, D_sw] Swin S0 output (pitch)
        acoustic_src_s1: Optional[torch.Tensor] = None,     # [B, T_s1, D_s1] Swin S1 output (beat)
        input_lengths: Optional[torch.Tensor] = None,        # [B] valid frame counts
        target_lengths: Optional[torch.Tensor] = None,       # [B] acoustic token counts
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute CIF acoustic embeddings and quantity loss.

        Triple-channel design with Zeng-style concatenation (no FiLM):
          - fire_signal (BiMamba, 100fps): predicts α + provides temporal embeddings
          - acoustic_src (Swin S0, 12.5fps): pitch/harmonic features (integer lookup at fire positions)
          - acoustic_src_s1 (Swin S1, 6.25fps): beat/rhythm features (integer lookup at fire positions, optional)
          - Concat [temporal, pitch, beat] → project to d_model (like Zeng line 397-400)

        Gradient flow:
          - CE loss → combined_proj → acoustic_src/acoustic_src_s1 (encoder gets gradient) ✓
          - CE loss -X-> acoustic_temporal (detached) -X-> α / weight_proj (blocked)
          - CE loss -X-> fire_frames (integer indices block gradient to α)
          - Only qty_loss supervises α / weight_proj (decoupled training)

        Args:
            fire_signal: BiMamba output [B, T_bi, D_bi] for α prediction (100fps).
            acoustic_src: Swin S0 output [B, T_sw, D_sw] for pitch content (12.5fps).
            acoustic_src_s1: Swin S1 output [B, T_s1, D_s1] for beat content (6.25fps). Optional.
            input_lengths: Valid frame count per batch item (for padding mask).
            target_lengths: Target acoustic token count per item (for quantity loss).

        Returns:
            acoustic_embs: [B, N_max, d_model] — one embedding per CIF fire.
            alpha: [B, T_bi] — frame weight sequence (Σα ≈ N_acoustic).
            quantity_loss: scalar — mean |Σα - N_target| over batch.
        """
        B, T_bi, D_bi = fire_signal.shape
        B_sw, T_sw, D_sw = acoustic_src.shape
        device, dtype = fire_signal.device, fire_signal.dtype

        assert B == B_sw, f"Batch size mismatch: fire_signal={B}, acoustic_src={B_sw}"

        # ── Step 1: Predict α from fire_signal (BiMamba) ──────────────────────
        x = fire_signal.permute(0, 2, 1)   # [B, D_bi, T_bi]
        x = self.conv(x)                   # [B, D_bi, T_bi] depthwise local context
        x = x.permute(0, 2, 1)             # [B, T_bi, D_bi]
        x = self.weight_norm(x)
        # Simple sigmoid: alpha ∈ (0, 1). CIF fires when cumulative alpha reaches threshold=1.0.
        alpha = torch.sigmoid(self.weight_proj(x)).squeeze(-1)  # [B, T_bi]

        # Mask padding frames to zero weight
        if input_lengths is not None:
            pad_mask = torch.arange(T_bi, device=device).unsqueeze(0) >= input_lengths.unsqueeze(1)
            alpha = alpha.masked_fill(pad_mask, 0.0)

        # Quantity loss on raw α (before scaling) — supervises the raw prediction
        # so that Σα ≈ target at inference (where no scaling is applied).
        qty_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        if target_lengths is not None:
            qty_loss = self.compute_quantity_loss(alpha, target_lengths.to(dtype))

        # ── Step 2: Scaling strategy ────────────────────────────────────────────
        #   Training: scale α' = α * (target / Σα) so CIF fires exactly target times.
        #   Inference: use raw α; CIF fires round(Σα) times.
        if self.training and target_lengths is not None:
            sum_alpha = alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]
            alpha_cif = alpha * (target_lengths.unsqueeze(1).to(dtype) / sum_alpha)
        else:
            alpha_cif = alpha

        # ── Step 2.5: Dynamic threshold (Paraformer) ─────────────────────────────
        # β = Σα / ⌈Σα⌉ reduces training/inference mismatch without changing alpha.
        if self.use_dynamic_threshold:
            sum_alpha_cif = alpha_cif.sum(dim=1)  # [B]
            # ⌈Σα⌉ per batch sample (detach to avoid gradient through ceil)
            ceil_sum = torch.ceil(sum_alpha_cif).clamp(min=1.0).detach()
            # β = Σα / ⌈Σα⌉ (gradient flows through sum_alpha_cif, not ceil_sum)
            beta = (sum_alpha_cif / ceil_sum).unsqueeze(1)  # [B, 1]
            threshold_cif = beta  # dynamic threshold per sample
        else:
            threshold_cif = torch.full((B, 1), self.threshold, device=device, dtype=dtype)

        # ── Step 3: CIF integration on fire_signal only ─────────────────────────
        # Only fire_signal goes through CIF integration — it determines fire positions.
        # S0 and S1 use post-fire lookup (no upsample needed, much more memory-efficient).
        # Note: _cif_forward expects scalar threshold, so we'll modify it to accept [B, 1]
        acoustic_temporal = self._cif_forward(fire_signal, alpha_cif, threshold_cif)  # [B, N, D_bi]
        N = acoustic_temporal.shape[1]

        # Detach acoustic_temporal to block CE → alpha gradient path
        # CE can still train encoder (Swin/BiMamba) and combined_proj weights
        acoustic_temporal_for_proj = acoustic_temporal.detach()

        # ── Step 4: Post-fire lookup for S0 and S1 (differentiable bilinear) ──────
        # Compute continuous fire positions so gradients flow back to alpha.
        #
        # Fire n happens where cum crosses threshold*n.  With dynamic threshold β,
        # fires occur at β, 2β, 3β, ... instead of 1, 2, 3, ...
        # Between the frame just before the crossing (t_lo = fire_frame - 1) and
        # the crossing frame (t_hi = fire_frame), the cumsum is linear in alpha,
        # so the exact fractional crossing time is:
        #
        #   t_cont = t_lo + (threshold*n - cum[t_lo]) / alpha[t_hi]
        #
        # This is differentiable w.r.t. alpha through cum and alpha[t_hi],
        # allowing gradients from acoustic_pitch to reach weight_proj.
        cum = alpha_cif.cumsum(dim=1)                                    # [B, T_bi]
        # thresholds = [β, 2β, 3β, ...] or [1, 2, 3, ...] depending on dynamic flag
        thresholds = (torch.arange(1, N + 1, device=device, dtype=dtype)
                      .unsqueeze(0).expand(B, -1)                        # [B, N]
                      * threshold_cif)                                   # broadcast [B, 1]
        fire_frames = torch.searchsorted(
            cum.contiguous(), thresholds.contiguous()
        ).clamp(max=T_bi - 1)                                            # [B, N] int

        # Integer fire position (non-differentiable lookup to block CE gradient to alpha)
        # CE gradient stops at acoustic_src values, cannot flow back to fire timing.
        # Only qty_loss supervises alpha / weight_proj.

        # S0 integer lookup (downsampled from BiMamba 100fps to S0 12.5fps)
        T_sw = acoustic_src.shape[1]
        D_sw = acoustic_src.shape[2]
        fire_frames_s0 = (fire_frames.float() * T_sw / T_bi).long().clamp(0, T_sw - 1)  # [B, N] int
        acoustic_pitch = acoustic_src.gather(
            1, fire_frames_s0.unsqueeze(-1).expand(B, N, D_sw)
        )  # [B, N, D_sw] — gradient flows to acoustic_src, NOT to fire_frames_s0

        # ── Step 5: Zeng-style concatenation (line 397-400) ─────────────────────
        # Concat temporal (detached) + pitch (+ beat), then project
        # Gradient: CE → combined_proj weights ✓, CE -X-> alpha (blocked by detach)

        if acoustic_src_s1 is not None:
            # With S1: concat all three channels
            T_s1 = acoustic_src_s1.shape[1]
            D_s1 = acoustic_src_s1.shape[2]
            fire_frames_s1 = (fire_frames.float() * T_s1 / T_bi).long().clamp(0, T_s1 - 1)
            acoustic_beat = acoustic_src_s1.gather(
                1, fire_frames_s1.unsqueeze(-1).expand(B, N, D_s1)
            )  # [B, N, D_s1] — gradient blocked at indices

            acoustic_cat = torch.cat([
                acoustic_temporal_for_proj,  # [B, N, 128] detached
                acoustic_pitch,              # [B, N, 192]
                acoustic_beat,               # [B, N, 192]
            ], dim=-1)  # [B, N, 512]
            acoustic_embs = self.combined_proj_with_beat(acoustic_cat)  # [B, N, d_model]
        else:
            # Without S1: concat temporal + pitch only
            acoustic_cat = torch.cat([
                acoustic_temporal_for_proj,  # [B, N, 128] detached
                acoustic_pitch,              # [B, N, 192]
            ], dim=-1)  # [B, N, 320]
            acoustic_embs = self.combined_proj(acoustic_cat)  # [B, N, d_model]

        return acoustic_embs, alpha, qty_loss

    def _cif_forward(
        self,
        encoder_output: torch.Tensor,  # [B, T, D]
        alpha: torch.Tensor,           # [B, T]
        threshold: torch.Tensor,       # [B, 1] dynamic threshold per sample
    ) -> torch.Tensor:
        """Vectorized CIF via cumsum + scatter_add (no Python time loop).

        Key insight: the CIF integration window for token k spans frames where
        cumsum(α) crosses the k*threshold boundary. Two scatter_add ops handle
        all cases in parallel across both batch and time dimensions.

        With dynamic threshold β = Σα / ⌈Σα⌉, boundaries are at β, 2β, 3β, ...
        To reuse the floor-based logic, we normalize: cum_norm = cum / β,
        then boundaries are at 1, 2, 3, ... as before.

        For each frame t:
          cum_prev = Σ_{i<t} α_i,  cum = Σ_{i≤t} α_i
          cum_norm = cum / β,  cum_prev_norm = cum_prev / β
          k_lo = floor(cum_prev_norm)       — primary token index
          k_hi = floor(cum_norm)            — upper index (= k_lo + 1 if boundary crossed)
          w_lo = min(cum_norm, k_lo+1) - cum_prev_norm  — weight for k_lo
          w_hi = max(cum_norm - (k_lo+1), 0)            — weight for k_hi

        emb[k] = Σ_t w_t_k * h_t  (scatter_add over k_lo and k_hi)

        Assumption: α_t < threshold for all t (with dynamic threshold, β ≈ 1.0,
        so at most one boundary is crossed per frame).

        Complexity: O(T) parallel ops, O(1) Python overhead.
        Gradients flow through scatter_add into both alpha and encoder_output.

        Returns: [B, N_max, D] where N_max is max fire count across batch.
        """
        B, T, D = encoder_output.shape
        device, dtype = encoder_output.device, encoder_output.dtype

        # Cumulative sum of α: cum[b, t] = Σ_{i≤t} α[b,i]
        cum = alpha.cumsum(dim=1)                                   # [B, T]
        cum_prev = F.pad(cum[:, :-1], (1, 0), value=0.0)           # [B, T]  (shifted by 1)

        # Normalize by threshold: cum_norm = cum / β
        # threshold: [B, 1] → broadcast to [B, T]
        cum_norm = cum / threshold.clamp(min=1e-8)                  # [B, T]
        cum_prev_norm = cum_prev / threshold.clamp(min=1e-8)        # [B, T]

        # Token indices each frame contributes to (using normalized cumsum)
        k_lo = cum_prev_norm.floor().long()                         # [B, T]
        k_hi = cum_norm.floor().long()                              # [B, T]

        # Weights: w_lo + w_hi = α_t / β (normalized)
        # But we want unnormalized weights for actual integration, so scale back
        boundary_norm = (k_lo.float() + 1.0)                       # [B, T]
        w_lo_norm = (torch.min(cum_norm, boundary_norm) - cum_prev_norm).clamp(min=0)  # [B, T]
        w_hi_norm = (cum_norm - boundary_norm).clamp(min=0)                             # [B, T]

        # Scale back to unnormalized weights (multiply by β)
        w_lo = w_lo_norm * threshold                                # [B, T]
        w_hi = w_hi_norm * threshold                                # [B, T]

        # Maximum number of fires across batch (using normalized cumsum)
        n_max = int(cum_norm[:, -1].max().item()) + 1   # round up for safety
        n_max = max(n_max, 1)

        output = torch.zeros(B, n_max, D, device=device, dtype=dtype)

        # Primary scatter: w_lo * h_t → token k_lo
        k_lo_clamped = k_lo.clamp(0, n_max - 1)                    # [B, T]
        idx_lo = k_lo_clamped.unsqueeze(-1).expand(-1, -1, D)      # [B, T, D]
        output.scatter_add_(1, idx_lo, encoder_output * w_lo.unsqueeze(-1))

        # Overflow scatter: w_hi * h_t → token k_hi (only when k_hi > k_lo)
        k_hi_clamped = k_hi.clamp(0, n_max - 1)                    # [B, T]
        idx_hi = k_hi_clamped.unsqueeze(-1).expand(-1, -1, D)      # [B, T, D]
        output.scatter_add_(1, idx_hi, encoder_output * w_hi.unsqueeze(-1))

        # Trim trailing zeros: actual N per item = floor(cum_norm[:, -1])
        actual_n = int(cum_norm[:, -1].floor().max().item())
        actual_n = max(actual_n, 1)
        return output[:, :actual_n]  # [B, N, D]

    def compute_quantity_loss(
        self,
        alpha: torch.Tensor,          # [B, T]
        target_lengths: torch.Tensor, # [B] float
    ) -> torch.Tensor:
        """Quantity loss: mean |Σα_i - N_acoustic_i| over batch.

        Drives α to sum to the number of acoustic tokens, ensuring CIF
        fires approximately the right number of times per sequence.
        """
        sum_alpha = alpha.sum(dim=1)  # [B]
        return (sum_alpha - target_lengths).abs().mean()

    def align_to_seq_len(
        self,
        acoustic_embs: torch.Tensor,  # [B, N, D]
        seq_len: int,
        input_ids: Optional[torch.Tensor] = None,  # [B, S] for scatter alignment
    ) -> torch.Tensor:
        """Scatter acoustic_embs [B, N, D] → [B, seq_len, D] using CIF alignment.

        If input_ids is provided, uses build_cif_alignment() to assign each
        token position the correct emb index (Option B: fire-at-start-of-next-step).

        If N is too small (quantity loss not yet converged), clamps index to N-1
        so later time steps gracefully reuse the last fire rather than crashing.

        If input_ids is None, falls back to naive pad/truncate (backward compat).
        """
        B, N, D = acoustic_embs.shape

        if input_ids is None:
            # Fallback: naive pad/truncate (no structural awareness)
            if N == seq_len:
                return acoustic_embs
            if N < seq_len:
                pad = torch.zeros(B, seq_len - N, D,
                                  device=acoustic_embs.device,
                                  dtype=acoustic_embs.dtype)
                return torch.cat([acoustic_embs, pad], dim=1)
            return acoustic_embs[:, :seq_len]

        # Scatter: each position gets the emb at its CIF pointer index
        alignment = build_cif_alignment(input_ids)          # [B, S]
        alignment = alignment.clamp(max=N - 1)              # guard if N < target
        idx = alignment.unsqueeze(-1).expand(B, seq_len, D) # [B, S, D]
        return acoustic_embs.gather(1, idx)                 # [B, S, D]
