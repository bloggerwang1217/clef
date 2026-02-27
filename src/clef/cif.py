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

import math
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


def compute_acoustic_target_lengths(
    input_ids: torch.Tensor,  # [B, S]
) -> torch.Tensor:
    """Compute per-batch acoustic token count (for CIF quantity loss).

    Counts non-structural, non-padding tokens in input_ids.
    STRUCTURAL_IDS are excluded; everything else counts as one acoustic target.

    Returns: [B] float tensor of target fire counts.
    """
    B, S = input_ids.shape
    device = input_ids.device
    # Build structural mask on device
    max_id = int(input_ids.max().item()) + 1
    is_structural = torch.zeros(max_id, dtype=torch.bool, device=device)
    for sid in STRUCTURAL_IDS:
        if sid < max_id:
            is_structural[sid] = True
    # Clamp for safety
    ids_clamped = input_ids.clamp(0, max_id - 1)
    acoustic_mask = ~is_structural[ids_clamped]  # [B, S]
    return acoustic_mask.sum(dim=1).float()  # [B]


class CIFModule(nn.Module):
    """Continuous Integrate-and-Fire: encoder frames → per-token acoustic embeddings.

    Replaces BarMamba in the tiny model decoder. Provides one acoustic embedding
    per output token via a soft-alignment learned from a scalar weight α_t per frame.

    Architecture:
        1. Weight predictor:
               x = encoder_1d.permute(0,2,1)         # [B, D, T]
               x = DepthwiseConv1d(x)               # local temporal context
               x = x.permute(0,2,1)                  # [B, T, D]
               x = LayerNorm(x)
               α  = sigmoid(Linear(x, 1)).squeeze()  # [B, T] ∈ (0, 1)
        2. CIF integration (batched time loop):
               Fire when cumulative α ≥ threshold=1.0.
               Each fire produces one acoustic embedding ≈ weighted sum of frames.
        3. Quantity loss: |Σα - N_acoustic| drives α to sum to the token count.

    forward() returns (acoustic_embs, alpha, quantity_loss).
    align_to_seq_len() pads/truncates acoustic_embs from [B, N, D] to [B, S, D]
    for teacher-forcing.
    """

    def __init__(
        self,
        d_model: int,
        threshold: float = 1.0,
        conv_kernel: int = 3,
        max_seq_len: int = 2048,
        encoder_len: int = 3000,
    ):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold

        padding = conv_kernel // 2
        # Depthwise conv: captures local temporal context, no cross-channel mixing
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=conv_kernel,
            padding=padding,
            groups=d_model,
            bias=False,
        )
        self.weight_norm = nn.LayerNorm(d_model)
        self.weight_proj = nn.Linear(d_model, 1, bias=True)
        # Initialize bias so Σα ≈ max_seq_len at the start (upper-bound estimate).
        # α_init = max_seq_len / encoder_len  → bias = logit(α_init)
        # Quantity loss then converges Σα to the actual token count within 1-2 epochs.
        alpha_init = min(max_seq_len / encoder_len, 0.99)
        bias_init = math.log(alpha_init / (1.0 - alpha_init))
        nn.init.constant_(self.weight_proj.bias, bias_init)

    def forward(
        self,
        encoder_1d: torch.Tensor,                       # [B, T, D]
        input_lengths: Optional[torch.Tensor] = None,   # [B] valid frame counts
        target_lengths: Optional[torch.Tensor] = None,  # [B] acoustic token counts
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute CIF acoustic embeddings and quantity loss.

        Args:
            encoder_1d: BiMamba output [B, T, D].
            input_lengths: Valid frame count per batch item (for padding mask).
                           If None, all frames are considered valid.
            target_lengths: Target acoustic token count per item (for quantity loss).
                            Typically computed via compute_acoustic_target_lengths().

        Returns:
            acoustic_embs: [B, N_max, D] — one embedding per CIF fire.
            alpha: [B, T] — frame weight sequence (Σα ≈ N_acoustic).
            quantity_loss: scalar — mean |Σα - N_target| over batch.
        """
        B, T, D = encoder_1d.shape
        device, dtype = encoder_1d.device, encoder_1d.dtype

        # Weight predictor
        x = encoder_1d.permute(0, 2, 1)   # [B, D, T]
        x = self.conv(x)                   # [B, D, T] depthwise local context
        x = x.permute(0, 2, 1)            # [B, T, D]
        x = self.weight_norm(x)
        alpha = torch.sigmoid(self.weight_proj(x)).squeeze(-1)  # [B, T]

        # Mask padding frames to zero weight
        if input_lengths is not None:
            pad_mask = torch.arange(T, device=device).unsqueeze(0) >= input_lengths.unsqueeze(1)
            alpha = alpha.masked_fill(pad_mask, 0.0)

        # Quantity loss on raw α (before scaling) — supervises the raw prediction
        # so that Σα ≈ target at inference (where no scaling is applied).
        qty_loss = torch.zeros(1, device=device, dtype=dtype).squeeze()
        if target_lengths is not None:
            qty_loss = self.compute_quantity_loss(alpha, target_lengths.to(dtype))

        # Scaling strategy (Dong & Xu 2020, Section 3.2):
        #   Training: scale α' = α * (target / Σα) so CIF fires exactly target times.
        #             Every token position gets a valid acoustic embedding (no zero pad).
        #   Inference: use raw α; CIF fires round(Σα) times.
        #             Quantity loss ensures Σα ≈ target after training converges,
        #             keeping the training/inference gap small.
        if self.training and target_lengths is not None:
            sum_alpha = alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]
            alpha_cif = alpha * (target_lengths.unsqueeze(1).to(dtype) / sum_alpha)
        else:
            alpha_cif = alpha

        # CIF integration
        acoustic_embs = self._cif_forward(encoder_1d, alpha_cif, self.threshold)
        # [B, N_max, D]  (N_max ≈ target in training, ≈ round(Σα) in inference)

        return acoustic_embs, alpha, qty_loss

    def _cif_forward(
        self,
        encoder_output: torch.Tensor,  # [B, T, D]
        alpha: torch.Tensor,           # [B, T]
        threshold: float,
    ) -> torch.Tensor:
        """Vectorized CIF via cumsum + scatter_add (no Python time loop).

        Key insight: the CIF integration window for token k spans frames where
        cumsum(α) crosses the k-th integer boundary. Two scatter_add ops handle
        all cases in parallel across both batch and time dimensions.

        For each frame t:
          cum_prev = Σ_{i<t} α_i,  cum = Σ_{i≤t} α_i
          k_lo = floor(cum_prev)           — primary token index
          k_hi = floor(cum)                — upper index (= k_lo + 1 if boundary crossed)
          w_lo = min(cum, k_lo+1) - cum_prev  — weight for k_lo
          w_hi = max(cum - (k_lo+1), 0)       — weight for k_hi (= α_t - w_lo)

        emb[k] = Σ_t w_t_k * h_t  (scatter_add over k_lo and k_hi)

        Assumption: α_t < threshold for all t (guaranteed since α = sigmoid(·) < 1
        and threshold = 1.0, so at most one boundary is crossed per frame).

        Complexity: O(T) parallel ops, O(1) Python overhead.
        Gradients flow through scatter_add into both alpha and encoder_output.

        Returns: [B, N_max, D] where N_max is max fire count across batch.
        """
        B, T, D = encoder_output.shape
        device, dtype = encoder_output.device, encoder_output.dtype

        # Cumulative sum of α: cum[b, t] = Σ_{i≤t} α[b,i]
        cum = alpha.cumsum(dim=1)                                   # [B, T]
        cum_prev = F.pad(cum[:, :-1], (1, 0), value=0.0)           # [B, T]  (shifted by 1)

        # Token indices each frame contributes to
        k_lo = cum_prev.floor().long()                              # [B, T]
        k_hi = cum.floor().long()                                   # [B, T]

        # Weights: w_lo + w_hi = α_t (always)
        boundary = (k_lo.float() + 1.0)                            # [B, T]
        w_lo = (torch.min(cum, boundary) - cum_prev).clamp(min=0)  # [B, T]
        w_hi = (cum - boundary).clamp(min=0)                       # [B, T]

        # Maximum number of fires across batch
        n_max = int(cum[:, -1].max().item()) + 1   # round up for safety
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

        # Trim trailing zeros: actual N per item = floor(cum[:, -1])
        actual_n = int(cum[:, -1].floor().max().item())
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
    ) -> torch.Tensor:
        """Pad or truncate acoustic_embs [B, N, D] → [B, seq_len, D].

        In early training, N ≠ seq_len until quantity loss converges.
          N < seq_len: pad with zeros (no acoustic signal for those positions)
          N > seq_len: truncate (discard extra firings)
        """
        B, N, D = acoustic_embs.shape
        if N == seq_len:
            return acoustic_embs
        if N < seq_len:
            pad = torch.zeros(
                B, seq_len - N, D,
                device=acoustic_embs.device,
                dtype=acoustic_embs.dtype,
            )
            return torch.cat([acoustic_embs, pad], dim=1)
        return acoustic_embs[:, :seq_len]
