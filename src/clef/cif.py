"""
CIF Module — Continuous Integrate-and-Fire
==========================================

Maps encoder frames [B, T, D] → per-token acoustic embeddings [B, N, D].

References:
    Dong & Xu 2020, "CIF: Continuous Integrate-and-Fire for End-to-End
    Speech Recognition", ICASSP 2020.

Design (Cascaded Energy-Probability Architecture):
    Layer 1 — Energy Envelope (α):
        DepthwiseConv1d(kernel=3, groups=d_model) → LayerNorm → Linear(d_model, 1) → Softplus
        Softplus produces α ∈ (0, ∞): unbounded acoustic energy per frame.
        α > threshold fires the CIF; α < threshold stays silent.

    Layer 2 — Onset Probability (P) via Soft Schmitt Trigger:
        P = σ((α − threshold) / temperature)  — fixed, no trainable weights.
        Small temperature (0.1) forces P to strictly reflect whether α crosses threshold.
        Quantity loss supervises sum(P) ≈ N_target, driving α to form real peaks.

    CIF integration (Hard Reset, no rescaling):
        Accumulate α * h_t. When accumulator ≥ threshold=1.0, fire and reset to 0.
        No rescaling: α represents true acoustic energy; rescaling would corrupt peaks.

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
               α = softplus(Linear(ReLU(Linear(LayerNorm(DepthwiseConv1d(fire_signal))))))  # α ∈ (0, ∞)
        2. Onset probability via Soft Schmitt Trigger (no trainable weights):
               P = σ((α − threshold) / temperature)  # P ∈ (0, 1)
        3. Dual-channel CIF integration (raw α, peak-detection, no accumulation):
               acoustic_temporal = CIF(fire_signal, α)  # [B, N, D_bi] fires at α > threshold
               acoustic_pitch = lookup(acoustic_src, fire_frames)  # [B, N, 192] S0 at fire positions
        4. Concatenation:
               acoustic_cat = [acoustic_temporal, acoustic_pitch]  # [B, N, 320]
               (if S1 exists: [temporal, pitch, beat] → [B, N, 512])
        5. Project to d_model:
               out = Linear(concat_dim → d_model)(acoustic_cat)  # [B, N, d_model]
        6. Quantity loss: |sum(P) - N_structural| where N_structural = count(<bar>) + count(<nl>).

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
        threshold: float = 1.0,          # CIF fire threshold (fixed)
        conv_kernel: int = 3,
        cif_hidden_dim: int = 128,       # hidden dim for weight predictor Dense layer
        target_fires: int = 128,         # Target fire count (avg structural tokens per chunk)
        encoder_len: int = 3000,
        schmitt_temp: float = 0.1,       # Soft Schmitt Trigger temperature: smaller = sharper onset gate
    ):
        super().__init__()
        self.fire_signal_dim = fire_signal_dim
        self.acoustic_dim = acoustic_dim
        self.beat_dim = beat_dim
        self.d_model = d_model
        self.threshold = threshold
        self.cif_hidden_dim = cif_hidden_dim
        self.schmitt_temp = schmitt_temp

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
        # Dense layer for weight prediction (matches original CIF design: 128 → 128 → 1)
        self.weight_dense = nn.Linear(fire_signal_dim, cif_hidden_dim)
        self.weight_proj = nn.Linear(cif_hidden_dim, 1, bias=True)

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

        Cascaded Energy-Probability design (no rescaling):
          - α (Softplus): unbounded energy envelope; α > 1 fires the CIF
          - P (Soft Schmitt Trigger): P = σ((α − threshold) / 0.1); quantity loss on sum(P)
          - No rescaling of α: true energy decides fire timing; scaling would corrupt peaks

        Gradient flow (FunASR-style):
          - CE loss → combined_proj → acoustic_src/acoustic_src_s1 (encoder gets gradient) ✓
          - CE loss -X-> acoustic_temporal (detached) -X-> α / weight_proj (blocked)
          - CE loss -X-> fire_frames (integer indices block gradient to α)
          - Qty loss → onset_prob → α → weight_proj → fire_signal → BiMamba ✓
          - BiMamba receives gradients from both CE and quantity loss (co-optimization)

        Args:
            fire_signal: BiMamba output [B, T_bi, D_bi] for α prediction (100fps).
            acoustic_src: Swin S0 output [B, T_sw, D_sw] for pitch content (12.5fps).
            acoustic_src_s1: Swin S1 output [B, T_s1, D_s1] for beat content (6.25fps). Optional.
            input_lengths: Valid frame count per batch item (for padding mask).
            target_lengths: Target acoustic token count per item (for quantity loss).

        Returns:
            acoustic_embs: [B, N_max, d_model] — one embedding per CIF fire.
            alpha: [B, T_bi] — energy envelope (Softplus; α > 1 fires the CIF).
            quantity_loss: scalar — mean |sum(P_onset) - N_target| over batch.
        """
        B, T_bi, D_bi = fire_signal.shape
        B_sw, T_sw, D_sw = acoustic_src.shape
        device, dtype = fire_signal.device, fire_signal.dtype

        assert B == B_sw, f"Batch size mismatch: fire_signal={B}, acoustic_src={B_sw}"

        # ── Step 1: Predict α from fire_signal (BiMamba) ──────────────────────
        # Cascaded Energy-Probability: α is the raw acoustic energy envelope.
        # Softplus (instead of Sigmoid) keeps α unbounded — α > 1 fires the CIF.
        # Gradient from qty_loss flows: onset_prob → α → weight_proj → BiMamba.
        context = fire_signal.permute(0, 2, 1)      # [B, D_bi, T_bi]
        memory = self.conv(context)                 # [B, D_bi, T_bi] depthwise conv with temporal context
        x = memory + context                        # Residual connection (preserves onset features)
        x = x.permute(0, 2, 1)                      # [B, T_bi, D_bi]
        x = self.weight_norm(x)                     # LayerNorm for stability
        x = self.weight_dense(x)                    # Dense layer: D_bi → cif_hidden_dim
        x = F.relu(x)                               # Non-linearity after Dense
        alpha = F.softplus(self.weight_proj(x)).squeeze(-1)  # [B, T_bi] α ∈ (0, ∞)

        # ── Step 1.5: Onset Probability via Soft Schmitt Trigger ──────────────
        # Fixed (no trainable weights): P = σ((α − threshold) / temperature).
        # Small temperature forces P to strictly reflect whether α crosses threshold.
        # Quantity loss on sum(P) drives α to form real energy peaks > threshold.
        onset_prob = torch.sigmoid((alpha - self.threshold) / self.schmitt_temp)  # [B, T_bi]

        # Mask padding frames to zero weight and zero probability
        if input_lengths is not None:
            pad_mask = torch.arange(T_bi, device=device).unsqueeze(0) >= input_lengths.unsqueeze(1)
            alpha = alpha.masked_fill(pad_mask, 0.0)
            onset_prob = onset_prob.masked_fill(pad_mask, 0.0)

        # Quantity loss on onset_prob: supervises expected fire count (not α directly).
        # sum(P) = expected number of frames where α > threshold = expected fire count.
        qty_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        if target_lengths is not None:
            N_pred = onset_prob.sum(dim=1)  # [B]
            qty_loss = (N_pred - target_lengths.to(dtype)).abs().mean()

        # ── Step 2: CIF integration — raw α, no rescaling ──────────────────────
        # α represents true acoustic energy. Rescaling would corrupt energy peaks:
        # e.g. forcing sum(α)=129 might inflate background noise 0.1 → 1.0 (false fires)
        # or deflate real peaks. The Hard Reset mechanism handles count implicitly:
        # qty_loss trains α to have enough peaks > 1.0 to reach the target count.
        alpha_cif = alpha  # use raw energy directly

        # ── Step 3: Peak-detection CIF on fire_signal ───────────────────────────
        # Fire wherever α > threshold; each fired frame emits encoder_output[t] directly.
        # S0 and S1 use post-fire lookup (integer index, no upsample needed).
        acoustic_temporal = self._cif_forward(fire_signal, alpha_cif, self.threshold)  # [B, N, D_bi]
        N = acoustic_temporal.shape[1]

        # Detach acoustic_temporal to block CE → alpha gradient path
        # CE can still train encoder (Swin/BiMamba) and combined_proj weights
        acoustic_temporal_for_proj = acoustic_temporal

        # ── Step 4: Post-fire lookup for S0 and S1 (integer index) ──────
        # fire_frames[b, n] = frame index of the n-th fire in batch item b.
        # New mechanism: fires at frames where α > threshold (not cumsum crossing).
        # Use nonzero positions of fire_mask, padded to uniform shape [B, N].
        #
        # Integer index lookup blocks CE gradient to alpha (only qty_loss trains α).
        fire_mask = alpha_cif > self.threshold                                       # [B, T_bi] bool
        # cumsum gives each fired frame its 1-based slot; subtract 1 for 0-based.
        fire_cumsum = fire_mask.long().cumsum(dim=1)                                 # [B, T_bi]
        # For each slot n (0-based), find the first frame where cumsum == n+1.
        # searchsorted on fire_cumsum finds that frame efficiently.
        slot_targets = torch.arange(1, N + 1, device=device, dtype=torch.long
                       ).unsqueeze(0).expand(B, -1)                                  # [B, N]
        fire_frames = torch.searchsorted(
            fire_cumsum.contiguous(), slot_targets.contiguous()
        ).clamp(max=T_bi - 1)                                                        # [B, N] int

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
        alpha: torch.Tensor,           # [B, T]  Softplus energy, unbounded
        threshold: float,              # Fire threshold (1.0)
    ) -> torch.Tensor:
        """Peak-detection CIF: fire wherever α > threshold (no accumulation).

        New semantics (Cascaded Energy-Probability Architecture):
            Each frame fires independently if α[t] > threshold.
            The fired embedding is encoder_output[t] itself — no weighted integration.

        This replaces the old cumsum-based accumulation, which assumed α ∈ (0, 1)
        and required sum(α) fires total.  With Softplus α can exceed 1.0 on a single
        frame, so accumulation would fire multiple times per frame — wrong.

        Implementation:
            fire_mask[b, t] = (alpha[b, t] > threshold)          # bool [B, T]
            slot[b, t]      = cumsum(fire_mask, dim=1)[b, t] - 1  # 0-based index of this fire
            scatter_add encoder_output[b, t] → output[b, slot[b,t]]  only where fire_mask

        Complexity: O(T) parallel ops, O(1) Python overhead.
        Gradients: encoder_output gets CE gradient via scatter_add;
                   alpha gradient flows from qty_loss only (fire_mask is non-differentiable).

        Returns: [B, N_max, D] where N_max = max fires across batch.
        """
        B, T, D = encoder_output.shape
        device, dtype = encoder_output.device, encoder_output.dtype

        # Boolean fire mask: fire wherever energy exceeds threshold
        fire_mask = alpha > threshold                               # [B, T] bool

        # Cumulative fire count: slot index for each fired frame (0-based)
        fire_cumsum = fire_mask.long().cumsum(dim=1)               # [B, T] int
        slot = (fire_cumsum - 1).clamp(min=0)                      # [B, T] int, 0-based slot

        # Max fires across batch (determines output size)
        n_max = int(fire_cumsum[:, -1].max().item())
        n_max = max(n_max, 1)

        output = torch.zeros(B, n_max, D, device=device, dtype=dtype)

        # Scatter: fired frames write encoder_output[t] into the corresponding slot
        idx = slot.unsqueeze(-1).expand(-1, -1, D)                 # [B, T, D]
        fire_mask_expanded = fire_mask.unsqueeze(-1).expand(-1, -1, D)  # [B, T, D]
        output.scatter_add_(1, idx, encoder_output * fire_mask_expanded.to(dtype))

        # Trim to actual number of fires (= max slot used + 1)
        actual_n = int(fire_cumsum[:, -1].max().item())
        actual_n = max(actual_n, 1)
        return output[:, :actual_n]  # [B, N, D]

    def compute_quantity_loss(
        self,
        onset_prob: torch.Tensor,     # [B, T] onset probability per frame
        target_lengths: torch.Tensor  # [B]
    ) -> torch.Tensor:
        """Compute L1 loss between expected fire count and target count.

        Args:
            onset_prob: Probability of onset at each frame.
            target_lengths: Target number of fires.
        """
        N_pred = onset_prob.sum(dim=1)
        return (N_pred - target_lengths).abs().mean()

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
