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
    """Continuous Integrate-and-Fire: encoder frames → per-fire acoustic token sequences.

    Polyphonic design (v3):
      Each fire slot exposes a SEQUENCE of frequency tokens (not one vector), so the
      decoder can attend selectively to individual notes within a polyphonic onset.

      Sources:
        - fire_signal (Octopus onset_1d @ 100fps): predicts α timing ONLY
        - pitch_tokens (PitchSA @ fire positions): [B, N, P, d_pitch] pitch-bin tokens
        - swin_2d (SwinEncoder 2D @ 12.5fps): [B, H, W, D_sw] freq×time spatial grid

      At each fire slot n:
        - Gather Swin freq slice: swin_2d[:, :, w_n, :] → [B, H, D_sw]
        - Use PitchSA tokens:     pitch_tokens[:, n, :, :] → [B, P, d_pitch]
        - Project both to d_model, concat → [B, L_freq, d_model] where L_freq = P + H

      Returns (acoustic_token_seqs, alpha, qty_loss):
        acoustic_token_seqs: [B, N, L_freq, d_model]  — decoder CAs over L_freq per slot
        alpha:               [B, T_fire] energy envelope
        qty_loss:            scalar quantity loss

    forward() returns (acoustic_token_seqs, alpha, qty_loss).
    align_to_seq_len() maps [B, N, L, D] → [B, S, L, D] for teacher-forcing.
    """

    def __init__(
        self,
        fire_signal_dim: int = 128,       # BiMamba/Octopus output dim (for α prediction)
        pitch_token_dim: int = 32,        # PitchSA d_pitch (each pitch bin feature dim)
        n_pitch: int = 128,              # number of pitch bins (Flow output dim)
        swin_dim: int = 192,             # SwinEncoder S0 output dim per spatial position
        swin_s1_dim: int = 192,          # SwinEncoder S1 output dim (same as S0)
        d_model: int = 384,              # Decoder d_model (matches config.d_model)
        threshold: float = 0.5,          # CIF probability threshold
        conv_kernel: int = 3,
        cif_hidden_dim: int = 128,
        target_fires: int = 128,
        encoder_len: int = 3000,
        perceiver_m: int = 8,            # Number of Perceiver summary tokens per fire slot
    ):
        super().__init__()
        self.fire_signal_dim = fire_signal_dim
        self.n_pitch = n_pitch
        self.pitch_token_dim = pitch_token_dim
        self.swin_dim = swin_dim
        self.swin_s1_dim = swin_s1_dim
        self.d_model = d_model
        self.threshold = threshold
        self.cif_hidden_dim = cif_hidden_dim
        self.perceiver_m = perceiver_m

        # Prob predictor: depthwise conv + 2-layer MLP + Sigmoid
        padding = conv_kernel // 2
        self.conv = nn.Conv1d(
            fire_signal_dim, fire_signal_dim,
            kernel_size=conv_kernel,
            padding=padding,
            groups=fire_signal_dim,
            bias=False,
        )
        self.weight_norm = nn.LayerNorm(fire_signal_dim)
        self.weight_dense = nn.Linear(fire_signal_dim, cif_hidden_dim)
        self.weight_proj = nn.Linear(cif_hidden_dim, 1, bias=True)

        # Project pitch tokens (from PitchSA) → d_model
        self.pitch_proj = nn.Linear(pitch_token_dim, d_model)

        # Project Swin S0 freq-slice tokens → d_model
        self.swin_proj = nn.Linear(swin_dim, d_model)

        # Project Swin S1 freq-slice tokens → d_model
        self.swin_s1_proj = nn.Linear(swin_s1_dim, d_model)

        # Perceiver Compressor: [B, N, L, D] → [B, N, M, D]
        # Shared learned queries compress each fire slot's freq tokens into M summary tokens.
        # Used by MambaSALayer for windowed CA-based CIF pointer calibration.
        self.perceiver_queries = nn.Parameter(torch.randn(perceiver_m, d_model) * 0.02)
        self.perceiver_norm_q  = nn.LayerNorm(d_model)
        self.perceiver_norm_kv = nn.LayerNorm(d_model)
        self.perceiver_q_proj  = nn.Linear(d_model, d_model)
        self.perceiver_k_proj  = nn.Linear(d_model, d_model)
        self.perceiver_v_proj  = nn.Linear(d_model, d_model)
        self.perceiver_out     = nn.Linear(d_model, d_model)

    def compute_fires(
        self,
        fire_signal: torch.Tensor,                 # [B, T_fire, D_fire] Octopus onset_1d
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alpha prediction + peak-detection → fire_frames (no token projection).

        Separated from token assembly so PitchSA can run between compute_fires()
        and forward(), ensuring exactly ONE pass through the learnable layers
        (pitch_proj, swin_proj) — required for correct DDP gradient reduction.

        Returns:
          fire_frames: [B, N]      integer fire positions
          alpha:       [B, T_fire] Softplus energy
          qty_loss:    scalar
        """
        B, T_fire, _ = fire_signal.shape
        device, dtype = fire_signal.device, fire_signal.dtype

        # Onset Prob prediction
        context = fire_signal.permute(0, 2, 1)
        memory  = self.conv(context)
        x = (memory + context).permute(0, 2, 1)
        x = F.relu(self.weight_dense(self.weight_norm(x)))
        logits = self.weight_proj(x).squeeze(-1)  # [B, T_fire]
        
        # Pure Probability prediction
        onset_prob = torch.sigmoid(logits)

        if input_lengths is not None:
            pad_mask = torch.arange(T_fire, device=device).unsqueeze(0) >= input_lengths.unsqueeze(1)
            onset_prob = onset_prob.masked_fill(pad_mask, 0.0)

        qty_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        if target_lengths is not None:
            qty_loss = (onset_prob.sum(dim=1) - target_lengths.to(dtype)).abs().mean()

        # Peak-detection → fire_frames
        fire_mask   = onset_prob >= self.threshold
        fire_cumsum = fire_mask.long().cumsum(dim=1)
        N = max(int(fire_cumsum[:, -1].max().item()), 1)

        # Fallback: if no onsets detected, force one at frame 0 (start of audio)
        if N == 0:
            fire_frames = torch.zeros(B, 1, dtype=torch.long, device=device)
        else:
            slot_targets = torch.arange(1, N + 1, device=device).unsqueeze(0).expand(B, -1)
            fire_frames = torch.searchsorted(
                fire_cumsum.contiguous(), slot_targets.contiguous()
            ).clamp(max=T_fire - 1)  # [B, N] int

        # Return onset_prob instead of alpha descriptor
        return fire_frames, onset_prob, qty_loss

    def forward(
        self,
        fire_signal: torch.Tensor,                           # [B, T_fire, D_fire]
        swin_2d: torch.Tensor,                               # [B, H_freq, W_time, D_sw] S0
        fire_frames: torch.Tensor,                           # [B, N]  from compute_fires()
        pitch_tokens: Optional[torch.Tensor] = None,         # [B, N, P, d_pitch]
        swin_2d_s1: Optional[torch.Tensor] = None,          # [B, H_freq, W_time, D_sw] S1 (optional)
    ) -> torch.Tensor:
        """Assemble acoustic token sequences — single pass through learnable layers.

        Call compute_fires() first, then optionally PitchSA(flow_feat, fire_frames),
        then call forward() once.  Never call forward() twice in one step.

        Returns:
          acoustic_token_seqs: [B, N, L_freq, d_model]  (L_freq = P + H_s0 + H_s1)
        """
        B, N = fire_frames.shape
        T_fire = fire_signal.shape[1]
        H_freq, W_time, D_sw = swin_2d.shape[1], swin_2d.shape[2], swin_2d.shape[3]
        device, dtype = fire_frames.device, swin_2d.dtype

        # Swin S0 freq-slice tokens at fire positions
        fire_w = (fire_frames.float() * W_time / T_fire).long().clamp(0, W_time - 1)
        swin_t        = swin_2d.permute(0, 1, 3, 2)                           # [B, H, D, W]
        fw_idx        = fire_w.unsqueeze(1).unsqueeze(2).expand(B, H_freq, D_sw, N)
        swin_at_fires = swin_t.gather(3, fw_idx).permute(0, 3, 1, 2)         # [B, N, H, D_sw]
        swin_tokens   = self.swin_proj(swin_at_fires)                         # [B, N, H_freq, d_model]

        # Swin S1 freq-slice tokens at fire positions (same logic as S0)
        if swin_2d_s1 is not None:
            D_sw_s1 = swin_2d_s1.shape[3]
            swin_t_s1 = swin_2d_s1.permute(0, 1, 3, 2)                       # [B, H, D, W]
            fw_idx_s1 = fire_w.unsqueeze(1).unsqueeze(2).expand(B, H_freq, D_sw_s1, N)
            swin_at_fires_s1 = swin_t_s1.gather(3, fw_idx_s1).permute(0, 3, 1, 2)  # [B, N, H, D_sw_s1]
            swin_s1_tokens = self.swin_s1_proj(swin_at_fires_s1)              # [B, N, H_freq, d_model]
        else:
            swin_s1_tokens = torch.zeros(B, N, H_freq, self.d_model, device=device, dtype=dtype)

        # Pitch tokens — always pass through pitch_proj (DDP: params must participate every step)
        if pitch_tokens is not None:
            pitch_toks = self.pitch_proj(pitch_tokens)                         # [B, N, P, d_model]
        else:
            # Zero dummy so pitch_proj gets zero gradient — safe for DDP (grad = 0, not missing)
            dummy = torch.zeros(B, N, 1, self.pitch_token_dim, device=device, dtype=dtype)
            pitch_toks = self.pitch_proj(dummy)                                # [B, N, 1, d_model]

        full_tokens = torch.cat([pitch_toks, swin_tokens, swin_s1_tokens], dim=2)  # [B, N, L, d_model]

        # Perceiver Compressor: [B, N, L, D] → [B, N, M, D]
        # Shared queries attend over each fire slot's L freq tokens independently.
        BN = B * N
        L = full_tokens.shape[2]
        kv = full_tokens.reshape(BN, L, self.d_model)                          # [B*N, L, D]
        q  = self.perceiver_queries.unsqueeze(0).expand(BN, -1, -1)            # [B*N, M, D]

        q_n  = self.perceiver_q_proj(self.perceiver_norm_q(q))                 # [B*N, M, D]
        kv_n = self.perceiver_norm_kv(kv)
        k_n  = self.perceiver_k_proj(kv_n)                                     # [B*N, L, D]
        v_n  = self.perceiver_v_proj(kv_n)                                     # [B*N, L, D]

        attn = F.scaled_dot_product_attention(q_n, k_n, v_n)                   # [B*N, M, D]
        attn = self.perceiver_out(attn)                                        # [B*N, M, D]
        summary_tokens = (q + attn).reshape(B, N, self.perceiver_m, self.d_model)  # [B, N, M, D] w/ residual

        return full_tokens, summary_tokens  # [B, N, L, D], [B, N, M, D]




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
        acoustic_embs: torch.Tensor,  # [B, N, D] or [B, N, L, D]
        seq_len: int,
        input_ids: Optional[torch.Tensor] = None,  # [B, S]
    ) -> torch.Tensor:
        """Scatter acoustic_embs [B, N, ...] → [B, seq_len, ...] using CIF alignment.

        Supports both:
          - [B, N, D]    (legacy single-vector)
          - [B, N, L, D] (new polyphonic freq-token sequences)

        Returns tensor of same shape except dim 1 = seq_len.
        """
        shape = acoustic_embs.shape   # [B, N, ...extra]
        B, N = shape[0], shape[1]
        extra = shape[2:]             # () or (L, D) or (D,)

        if input_ids is None:
            # Fallback: naive pad/truncate
            if N == seq_len:
                return acoustic_embs
            if N < seq_len:
                pad = torch.zeros(B, seq_len - N, *extra,
                                  device=acoustic_embs.device,
                                  dtype=acoustic_embs.dtype)
                return torch.cat([acoustic_embs, pad], dim=1)
            return acoustic_embs[:, :seq_len]

        # Build per-token pointer into CIF fire slots
        alignment = build_cif_alignment(input_ids)          # [B, S]
        alignment = alignment.clamp(max=N - 1)              # guard if N < target

        if acoustic_embs.dim() == 3:
            # [B, N, D] → [B, S, D]
            D = extra[0]
            idx = alignment.unsqueeze(-1).expand(B, seq_len, D)
            return acoustic_embs.gather(1, idx)             # [B, S, D]
        elif acoustic_embs.dim() == 4:
            # [B, N, L, D] → [B, S, L, D]
            L, D = extra[0], extra[1]
            idx = alignment.unsqueeze(-1).unsqueeze(-1).expand(B, seq_len, L, D)
            return acoustic_embs.gather(1, idx)             # [B, S, L, D]
        else:
            raise ValueError(f"acoustic_embs must be 3D or 4D, got {acoustic_embs.dim()}D")

