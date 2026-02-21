"""
Bar Tracker
===========

Symmetric counterpart to HarmonizingFlow:
  Flow:       Octopus [B,C,F,T] → project along F → pitch features
  BarTracker: Octopus [B,C,F,T] → pool along F → Mamba → bar_phase [B,T,1]

The bar_phase output is a staircase signal: at audio frame t,
bar_phase[t] = normalized time of the current bar's start (∈ [0,1]).
This is a direct, learnable time_prior for the decoder.

Guidance toggle: pass bar_times GT to get a supervised MSE loss on bar_phase.
Without GT, the module still outputs bar_phase (useful as unsupervised prior).
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BarTracker(nn.Module):
    """Temporal Mamba that tracks bar phase from Octopus onset features.

    Analogous to Flow extracting frequency structure, BarTracker extracts
    temporal structure — specifically which bar we are currently in.

    The key inductive bias: Mamba processes onset_raw sequentially in time,
    building a state that tracks (tempo, phase). The bar_phase output is
    a monotone staircase: constant within a bar, jumps at bar boundaries.

    Args:
        in_channels: Octopus channel count (C in onset_raw)
        freq_bins: Octopus frequency bins (F in onset_raw, typically 128)
        d_model: internal hidden dimension
        n_layers: number of Mamba layers
        dropout: dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        freq_bins: int,
        d_model: int = 256,
        n_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        from mamba_ssm import Mamba2

        # Project [B, T, C*F_pool] → [B, T, d_model]
        # We mean-pool over F first (no learned weights — frequency is not the signal)
        # then project C channels to d_model
        self.freq_pool_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # Mamba layers: track (tempo, phase) state sequentially
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        # Bar phase head: outputs normalized bar-start time ∈ [0, 1]
        self.phase_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        onset_raw: torch.Tensor,
        bar_times: Optional[List[Optional[torch.Tensor]]] = None,
        audio_duration_frames: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Args:
            onset_raw: [B, C, F, T] — Octopus pre-pooling features
            bar_times: list of length B; each element is a 1D Tensor of bar onset
                       times in Octopus-resolution frames, or None if GT unavailable.
                       Pass None (the argument) to skip guidance entirely.
            audio_duration_frames: [B] total frame count per sample (for normalization).
                                   Required when bar_times is provided.

        Returns:
            bar_phase: [B, T, 1] — normalized bar-start time for each audio frame.
                       Staircase: constant within bar, jumps at bar boundaries.
            guidance_loss: scalar Tensor or None (None when bar_times is not provided).
        """
        B, C, F, T = onset_raw.shape

        # Pool over frequency axis: [B, C, F, T] → [B, C, T]
        x = onset_raw.mean(dim=2)   # [B, C, T]

        # [B, T, C]
        x = x.permute(0, 2, 1).contiguous()

        # Project to d_model
        x = self.freq_pool_proj(x)  # [B, T, d_model]

        # Mamba layers (sequential temporal processing)
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x) + residual

        # Predict bar phase: [B, T, 1] ∈ [0, 1]
        bar_phase = self.phase_head(x)

        # Guidance loss (only when GT bar_times are provided)
        guidance_loss = None
        if bar_times is not None and audio_duration_frames is not None:
            guidance_loss = self._compute_guidance_loss(
                bar_phase, bar_times, audio_duration_frames, T
            )

        return bar_phase, guidance_loss

    def _compute_guidance_loss(
        self,
        bar_phase: torch.Tensor,          # [B, T, 1]
        bar_times: List[Optional[torch.Tensor]],
        audio_duration_frames: torch.Tensor,  # [B]
        T: int,
    ) -> torch.Tensor:
        """MSE loss between predicted bar_phase and GT staircase.

        GT construction: for each audio frame t, find which bar it belongs to,
        then assign that bar's normalized start time as the label.

        bar_phase_gt[b, t] = bar_times[b][k] / audio_duration_frames[b]
        where k = argmax{i : bar_times[b][i] <= t}
        """
        device = bar_phase.device
        total_loss = bar_phase.new_zeros(1)
        valid_count = 0

        for b in range(bar_phase.shape[0]):
            if bar_times[b] is None or len(bar_times[b]) == 0:
                continue

            bt = bar_times[b].to(device).float()  # [N_bars]
            dur = audio_duration_frames[b].float()

            # Build GT staircase for this sample: [T]
            # frame_indices: [T] - each frame's index into [0, T)
            frame_idx = torch.arange(T, device=device, dtype=torch.float32)

            # For each frame, find the latest bar onset <= frame_idx
            # bt: [N_bars], frame_idx: [T]
            # diff[i, t] = frame_idx[t] - bt[i]; want max i where diff >= 0
            diff = frame_idx.unsqueeze(0) - bt.unsqueeze(1)  # [N_bars, T]
            diff = diff.masked_fill(diff < 0, -1e9)
            bar_idx = diff.argmax(dim=0)  # [T] — which bar each frame belongs to

            # Normalized bar-start time for each frame
            bar_phase_gt = bt[bar_idx] / dur.clamp(min=1.0)  # [T]

            # MSE loss on this sample
            pred = bar_phase[b, :, 0]  # [T]
            loss_b = F.mse_loss(pred, bar_phase_gt)
            total_loss = total_loss + loss_b
            valid_count += 1

        if valid_count == 0:
            return bar_phase.new_zeros(1).squeeze()

        return (total_loss / valid_count).squeeze()
