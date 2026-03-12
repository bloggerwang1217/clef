"""
Clef Piano Tiny Model — Isotonic Attention Architecture
========================================================

Encoder: Octopus2D → Flow → SwinEncoder → freq_conv → BiMamba
         + Onset detector head → onset_logits [B, T]

Decoder: MambaIsotonicAttnLayer (IOHMM formulation, Bengio & Frasconi 1995)
  Step 1: y = Mamba(tgt)                               [B, S, D]
  Step 2: p_{i,j} = σ(λ Q_i·K_j/√d + ℓ_j - μ)       (IOHMM + cardinality relaxation)
          α_{i,j} via Raffel recurrence  →  c [B, S, D]
  Step 3: W_fuse(cat([y_i, c_i])) → pred

Alignment: LP relaxation of cardinality-constrained IP (Σz=N, z∈{0,1}).
           Bisection finds μ per sample so Σ_j σ(ℓ_j - μ) = N.
           Gradient: CE → α → p_{i,j} → (ℓ_j - μ) → onset_logits → BiMamba.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ClefPianoConfig
from ..bimamba import BiMambaEncoder
from ..decoder import ClefDecoder
from ..flow import HarmonizingFlow, Octopus2D
from ..swin import SwinEncoder


class OnsetDetector(nn.Module):
    """Onset detector on flow_with_onset [B, T=3000, 128] @100fps.

    Architecture (Dong & Xu 2020 CIF weight predictor):
      DWConv1d + residual → LayerNorm → Linear(D→hidden) → ReLU → Linear(hidden→1) → logits

    Returns logits (before sigmoid) so callers can:
      - max_pool logits → onset_logit_bias [B, T'] for cardinality_relaxation + attention
    """

    def __init__(self, d_model: int, hidden_dim: int = 128, conv_kernel: int = 3):
        super().__init__()
        padding = conv_kernel // 2
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=conv_kernel,
            padding=padding,
            groups=d_model,
            bias=False,
        )
        self.norm   = nn.LayerNorm(d_model)
        self.dense  = nn.Linear(d_model, hidden_dim)
        self.proj   = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] (flow_with_onset @100fps, D=128)
        Returns:
            logits: [B, T]  (raw scores, before sigmoid)
        """
        h = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1) + x   # [B, T, D]  DWConv + residual
        h = F.relu(self.dense(self.norm(h)))                       # [B, T, hidden]
        return self.proj(h).squeeze(-1)                            # [B, T] logits


class ClefPianoTiny(nn.Module):
    """Clef Piano Tiny with Isotonic Attention decoder.

    Encoder:  Octopus → Flow → Swin → freq_conv → BiMamba → h [B, T=375, D=384]
              + OnsetDetector(h) → p_onset [B, T]
    Decoder:  MambaIsotonicAttnLayer (IOHMM, see decoder.py)
    """

    def __init__(self, config: ClefPianoConfig):
        super().__init__()
        self.config = config

        # === Octopus2D (onset detection) ===
        self.octopus = Octopus2D(
            freq_kernel=getattr(config, 'octopus_freq_kernel', 31),
            time_kernel=getattr(config, 'octopus_time_kernel', 3),
            channels=getattr(config, 'octopus_channels', 32),
            time_pool_stride=getattr(config, 'octopus_time_pool_stride', 2),
        )

        # Project onset from mel space to pitch space
        octopus_channels = getattr(config, 'octopus_channels', 32)
        self.onset_to_pitch = nn.Conv2d(octopus_channels, 1, kernel_size=1)
        nn.init.normal_(self.onset_to_pitch.weight, mean=0, std=0.01)
        nn.init.zeros_(self.onset_to_pitch.bias)

        # === HarmonizingFlow (pitch space transform) ===
        self.flow = HarmonizingFlow(
            n_mels=config.n_mels,
            n_harmonics=getattr(config, 'n_harmonics', 6),
            init=getattr(config, 'flow_init', 'log'),
        )

        # === SwinEncoder (S0 + S1) ===
        # flow_with_onset [B, T, 128] → Swin → [B, H, W, 192] @12.5fps
        self.swin = SwinEncoder(
            input_dim=config.n_mels,
            swin_model=getattr(config, 'swin_model', "microsoft/swinv2-tiny-patch4-window8-256"),
            use_gradient_checkpointing=getattr(config, 'swin_use_gradient_checkpointing', False),
        )

        # freq_conv: concat S0+S1 [B, H=16, W, 384] → [B, D, 1, W] → [B, W, D]
        bimamba_d_model = getattr(config, 'bimamba_d_model', config.d_model)
        swin_concat_dim = self.swin.output_dim * 2  # 192 * 2 = 384
        self.freq_conv = nn.Sequential(
            nn.Conv2d(swin_concat_dim, bimamba_d_model, kernel_size=(4, 1), stride=(4, 1)),
            nn.GELU(),
            nn.Conv2d(bimamba_d_model, bimamba_d_model, kernel_size=(4, 1), stride=(4, 1)),
        )

        # === BiMamba Encoder (decoder memory) ===
        # swin_feat [B, T/8, D] → BiMamba → h [B, T/8, D]  (decoder memory h_j)
        self.bimamba_encoder = BiMambaEncoder(
            d_model=bimamba_d_model,
            d_state=getattr(config, 'bimamba_d_state', 128),
            d_conv=getattr(config, 'bimamba_d_conv', 4),
            dropout=getattr(config, 'bimamba_dropout', 0.1),
        )

        # === Onset Detector head ===
        # Operates on flow_with_onset [B, T=3000, 128] @100fps directly.
        # Output p_onset_hires [B, 3000] is max-pooled 8x → [B, 375] for attention bias.
        self.onset_detector = OnsetDetector(
            d_model=config.n_mels,                                # 128
            hidden_dim=getattr(config, 'onset_hidden_dim', 128),
            conv_kernel=getattr(config, 'onset_conv_kernel', 3),
        )

        # === Decoder: MambaIsotonicAttnLayer ===
        self.decoder = ClefDecoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.decoder_layers,
            n_levels=config.n_levels,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            rope_base=config.rope_base,
            window_time_frames=config.window_time_frames,
            window_freq_bins=config.window_freq_bins,
            window_seq_chunk_size=getattr(config, 'window_seq_chunk_size', 10000),
            window_ca_use_checkpoint=getattr(config, 'window_ca_use_checkpoint', False),
            decoder_layer_types=config.decoder_layer_types,
            decoder_layer_ca_levels=config.decoder_layer_ca_levels,
            decoder_layer_full_freq=getattr(config, 'decoder_layer_full_freq', None),
            decoder_layer_cascade_com=getattr(config, 'decoder_layer_cascade_com', None),
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            use_rope=True,
            bar_token_id=None,  # MambaIsotonicAttnLayer handles alignment, no BarMamba needed
            onset_1d_channels=octopus_channels,
        )

        # === Token embedding + output projection ===
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        # === Dual variable for cardinality relaxation (Eq. dual_ascent_z) ===
        # μ ← μ + α(Σ p_s - N); updated each training step, not a learnable param.
        # Dual variable μ for cardinality LP relaxation.
        # Managed by a separate dual_optimizer in the training loop (not updated by main optimizer).
        self.dual_mu = nn.Parameter(torch.full((1,), 0.5))

        if getattr(config, 'gradient_checkpointing', False):
            self.decoder.gradient_checkpointing = True

    def encode(self, mel: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Encode mel spectrogram.

        Returns:
            memory:            [B, T, D]  BiMamba output (h_j for monotonic attention)
            spatial_shapes:    [1, 2]
            level_start_index: [1]
            valid_ratios:      [B, 1, 2]
            p_onset:           [B, T]     onset probabilities for attention prior
        """
        B, _, _, T = mel.shape

        # Step 1: Octopus2D
        onset_level, onset_raw, _ = self.octopus(mel)

        # Step 2: Flow + onset integration
        flow_feat = self.flow(mel)                                      # [B, T, 128]
        onset_signal = self.onset_to_pitch(onset_raw).squeeze(1)        # [B, 128, T]
        onset_signal = onset_signal.permute(0, 2, 1)                    # [B, T, 128]
        onset_in_pitch = onset_signal @ self.flow.transform.T           # [B, T, 128]
        flow_with_onset = flow_feat + onset_in_pitch                    # [B, T, 128]

        # Step 3: Swin + freq_conv
        swin_s0, swin_s1 = self.swin(flow_with_onset)                  # each [B, H, W, 192]
        T_swin = swin_s0.shape[2]
        swin_2d = torch.cat([swin_s0, swin_s1], dim=-1)                # [B, H, W, 384]
        swin_2d = swin_2d.permute(0, 3, 1, 2)                         # [B, 384, H, W]
        swin_2d = self.freq_conv(swin_2d)                               # [B, D, 1, W]
        swin_feat = swin_2d.squeeze(2).permute(0, 2, 1)                # [B, T_swin, D]

        # Step 4: Onset Detector @100fps
        onset_logits = self.onset_detector(flow_with_onset)             # [B, T=3000]
        p_onset_hires = torch.sigmoid(onset_logits)                     # [B, 3000]
        onset_logit_bias = F.max_pool1d(
            onset_logits.unsqueeze(1), kernel_size=8, stride=8
        ).squeeze(1)                                                    # [B, T_swin=375]

        # Step 5: BiMamba → h_j (decoder memory, bidirectional temporal context)
        h = self.bimamba_encoder(swin_feat)                             # [B, T_swin, D]

        memory = h
        spatial_shapes = torch.tensor([[1, T_swin]], dtype=torch.long, device=mel.device)
        level_start_index = torch.tensor([0], dtype=torch.long, device=mel.device)
        valid_ratios = torch.ones(B, 1, 2, dtype=torch.float32, device=mel.device)

        return memory, spatial_shapes, level_start_index, valid_ratios, onset_logit_bias, p_onset_hires

    def forward(
        self,
        mel: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mel_valid_ratios: Optional[torch.Tensor] = None,
        chunk_audio_measures: Optional[list] = None,
        chunk_start_frames: Optional[list] = None,
        chunk_end_frames: Optional[list] = None,
        guidance_loss_weight: Optional[float] = None,
        tf_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass.

        Returns:
            logits:     [B, S, vocab_size]
            loss:       CE loss
            total_loss: CE loss + qty_weight * |Σσ(ℓ_j) - N|
                        dual_mu (updated externally) enforces Σσ(ℓ-μ)=N in decoder
        """
        memory, spatial_shapes, level_start_index, valid_ratios, onset_logit_bias, p_onset_hires = self.encode(mel)

        # Cardinality relaxation: p_s = σ(ℓ_s - μ), where μ is updated by the
        # training loop via update_dual_mu() after each optimizer step.
        N_target = ((input_ids == 4).sum(dim=1) +
                    (input_ids == 6).sum(dim=1)).float()        # <bar>=4, <nl>=6
        onset_logit_bias = onset_logit_bias - self.dual_mu   # p_s = σ(ℓ_s - μ)

        tgt = self.token_embed(input_ids)   # [B, S, D]

        decoder_out = self.decoder(
            tgt,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            input_ids=input_ids,
            p_onset=onset_logit_bias,       # [B, T_swin] — shifted logits (ℓ - μ)
        )
        logits = self.output_projection(decoder_out)

        loss = None
        total_loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=0,
                label_smoothing=getattr(self.config, 'label_smoothing', 0.0),
            )
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            # Augmented Lagrangian: μ·gap + ρ·gap²
            # - Lagrangian term: ∂L/∂ℓ_j = μ·p(1-p), dual_mu detached so only dual_optimizer updates it
            # - Quadratic term: ∂L/∂ℓ_j = 2ρ·gap·p(1-p), always-on stabiliser (dampens dual oscillation)
            signed_gap = (torch.sigmoid(onset_logit_bias).sum(dim=1) - N_target).mean()
            lagrangian_loss = self.dual_mu.detach() * signed_gap
            quantity_loss = self.config.quantity_loss_weight * signed_gap ** 2
            total_loss = loss + lagrangian_loss + quantity_loss
            self._qty_loss = (lagrangian_loss + quantity_loss).detach().abs()
            self._dual_signed_gap = signed_gap.detach()   # for dual_optimizer step
        else:
            self._dual_signed_gap = None
            self._qty_loss = None

        return logits, loss, total_loss

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        max_len: int = 1024,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Greedy autoregressive generation.

        MambaIsotonicAttnLayer is used in full-sequence soft mode each step:
        the decoder re-runs over the growing generated sequence and reads the
        last token's logits.  N onsets are counted from the generated tokens.
        """
        B = mel.shape[0]
        device = mel.device

        memory, spatial_shapes, level_start_index, valid_ratios, onset_logit_bias, _ = self.encode(mel)

        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt = self.token_embed(generated)                         # [B, S, D]

            decoder_out = self.decoder(
                tgt, memory,
                spatial_shapes, level_start_index, valid_ratios,
                input_ids=generated,
                p_onset=onset_logit_bias,
            )                                                         # [B, S, D]

            logits      = self.output_projection(decoder_out[:, -1]) # [B, vocab]
            next_token  = logits.argmax(dim=-1, keepdim=True)        # [B, 1]
            generated   = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated
