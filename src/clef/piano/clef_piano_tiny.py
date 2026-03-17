"""
Clef Piano Tiny Model
=====================

Minimal PoC: Octopus + Flow + BiMamba + 2-layer decoder

Goal: Beat SOTA with minimal architecture following Zeng et al. 2024's design.
This is a clean rewrite, importing modules from the main codebase.

Architecture (Zeng-style):
- Encoder: Octopus2D → Flow → BiMamba (3 levels total)
  - Flow: 10ms resolution, clear piano roll (Top-1 = 44%)
  - BiMamba: Time-oriented bidirectional Mamba (like Zeng's Bi-GRU)
    - Linear projection for frequency integration (88 pitch → d_model)
    - Temporal modeling on time axis (onset, sustain, release)
- Decoder: SA+CA (to BiMamba) + Mamba → Output
"""

from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swinv2Model

from .config import ClefPianoConfig
from ..bimamba import BiMambaEncoder
from ..decoder import ClefDecoder
from ..flow import HarmonizingFlow, Octopus2D
from ..swin import SwinEncoder


class ClefPianoTiny(nn.Module):
    """Minimal Clef Piano model for 30-second audio clips (Zeng-style).

    Encoder: Octopus + Flow + BiMamba (3 levels, following Zeng et al. 2024)
    - Flow: 10ms resolution, clear piano roll representation
    - BiMamba: Time-oriented Bi-Mamba (like Zeng's Bi-GRU)
      - Linear projection: 88 pitch × 32 features → d_model (frequency integration)
      - Bi-Mamba: Temporal modeling on time axis (forward + backward)

    Decoder: 2 layers (SA+CA to BiMamba + Mamba)

    Target: ~1024 tokens max sequence length for 30s audio.
    """

    def __init__(self, config: ClefPianoConfig):
        super().__init__()
        self.config = config

        # === NO Swin in Zeng-style design ===
        # BiMamba directly processes Flow output

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

        # === SwinEncoder (S0 + S1, acoustic feature extraction) ===
        # Flow [B, T, 128] → SwinEncoder → [B, T/8, 384] @12.5fps
        # S0: pitch/harmony features, S1: beat-aware features
        self.swin = SwinEncoder(
            input_dim=config.n_mels,  # 128 (Flow output dimension)
            swin_model=getattr(config, 'swin_model', "microsoft/swinv2-tiny-patch4-window8-256"),
            use_gradient_checkpointing=getattr(config, 'swin_use_gradient_checkpointing', False),
        )

        # Conv along freq axis to aggregate harmonic structure.
        # Concat S0+S1: [B, H=16, W, 384] → permute → [B, 384, H=16, W]
        # Two strided Conv2d: H=16 → 4 → 1 (each layer has local receptive field of 4 bins)
        # Harmonic intervals are fixed in log-mel space, so local conv can learn them.
        bimamba_d_model = getattr(config, 'bimamba_d_model', config.d_model)
        swin_concat_dim = self.swin.output_dim * 2  # 192 * 2 = 384
        self.freq_conv = nn.Sequential(
            # [B, 384, H=16, W] → [B, bimamba_d_model, 4, W]
            nn.Conv2d(swin_concat_dim, bimamba_d_model, kernel_size=(4, 1), stride=(4, 1)),
            nn.GELU(),
            # [B, bimamba_d_model, 4, W] → [B, bimamba_d_model, 1, W]
            nn.Conv2d(bimamba_d_model, bimamba_d_model, kernel_size=(4, 1), stride=(4, 1)),
        )

        # === BiMamba Encoder (time-oriented, after Swin) ===
        # swin_proj output [B, T/8, bimamba_d_model] → BiMamba temporal modeling
        self.use_bimamba = getattr(config, 'use_bimamba_encoder', True)
        if self.use_bimamba:
            self.bimamba_encoder = BiMambaEncoder(
                input_dim=bimamba_d_model,  # swin_proj output dimension
                d_model=getattr(config, 'bimamba_d_model', config.d_model),
                d_state=getattr(config, 'bimamba_d_state', 128),
                d_conv=getattr(config, 'bimamba_d_conv', 4),
                num_layers=getattr(config, 'bimamba_num_layers', 2),
                dropout=getattr(config, 'bimamba_dropout', 0.1),
            )

        # === Decoder: 2 layers (SA+CA + Mamba) ===
        self.decoder = ClefDecoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.decoder_layers,  # 2 (legacy fallback, ignored)
            n_levels=config.n_levels,  # 3
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            rope_base=config.rope_base,
            # Window CA params
            window_time_frames=config.window_time_frames,
            window_freq_bins=config.window_freq_bins,
            window_seq_chunk_size=getattr(config, 'window_seq_chunk_size', 10000),
            window_ca_use_checkpoint=getattr(config, 'window_ca_use_checkpoint', True),
            # Layer config (CRITICAL: correct parameter names!)
            decoder_layer_types=config.decoder_layer_types,  # ✅ decoder_layer_types not layer_types
            decoder_layer_ca_levels=config.decoder_layer_ca_levels,  # ✅ decoder_layer_ca_levels
            decoder_layer_full_freq=getattr(config, 'decoder_layer_full_freq', None),
            decoder_layer_cascade_com=getattr(config, 'decoder_layer_cascade_com', None),
            # Mamba config
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            # Position encoding
            use_rope=True,
            # Bar/Note GRU config
            bar_token_id=getattr(config, 'bar_token_id', None),
            onset_1d_channels=getattr(config, 'octopus_channels', 32),
            bar_gru_hidden_size=getattr(config, 'bar_gru_hidden_size', 256),
            bar_gru_input_dropout=getattr(config, 'bar_gru_input_dropout', 0.1),
            # Curriculum Learning for mamba_full_ca path
            curriculum_warmup_steps=getattr(config, 'curriculum_warmup_steps', 0),
            tbptt_chunk_size=getattr(config, 'tbptt_chunk_size', 256),
        )


        # === Token embedding ===
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Output projection: decoder output is [B, S, D] (MambaFullCALayer.out_proj handles 2D→D)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        # Gradient checkpointing (enabled via config)
        if getattr(config, 'gradient_checkpointing', False):
            self.decoder.gradient_checkpointing = True

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad frequency and time to multiple of `multiple`."""
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        return x, (H, W)

    def _compute_spatial_shapes(self, H: int, W: int) -> list:
        """Compute Swin spatial shapes for S0 only."""
        # Swin V2 Tiny: patch=4, [2,2,6,2] blocks with downsample between stages
        # After embedding: H/4, W/4
        # After S0 (2 blocks + downsample): H/8, W/8
        return [(H // 8, W // 8)]

    def encode(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Encode mel spectrogram to multi-level features.

        Args:
            mel: [B, 1, 128, T] mel spectrogram

        Returns:
            memory: [B, N, D] concatenated encoder features
            spatial_shapes: [n_levels, 2] (H, W) for each level
            level_start_index: [n_levels] start index in memory for each level
            valid_ratios: [B, n_levels, 2] valid ratios for each level
        """
        B, _, _, T = mel.shape

        features = []
        spatial_shapes_list = []
        valid_ratios_list = []

        # === Step 1: Octopus2D (onset detection) ===
        onset_level, onset_raw = self.octopus(mel)
        # onset_level: [B, H*W, C] where H=32, W=T//2
        # onset_raw: [B, C, 128, T]

        freq_pool = getattr(self.config, 'octopus_freq_pool_stride', 4)
        time_pool = getattr(self.config, 'octopus_time_pool_stride', 2)
        H_onset = 128 // freq_pool  # 32
        W_onset = T // time_pool

        features.append(onset_level)
        spatial_shapes_list.append((H_onset, W_onset))
        valid_ratios_list.append([1.0, 1.0])

        # === Step 2: Flow (pitch space transform) ===
        flow_feat = self.flow(mel)  # [B, T, 128] @100fps
        T_flow = flow_feat.shape[1]

        # === Step 2.5: Integrate onset into pitch space ===
        onset_signal = self.onset_to_pitch(onset_raw).squeeze(1)  # [B, 128, T]
        onset_signal = onset_signal.permute(0, 2, 1)  # [B, T, 128]
        onset_in_pitch = onset_signal @ self.flow.transform.T  # [B, T, 128]
        flow_with_onset = flow_feat + onset_in_pitch

        features.append(flow_with_onset)
        spatial_shapes_list.append((1, T_flow))
        valid_ratios_list.append([1.0, 1.0])

        # === Step 3: SwinEncoder (S0 + S1) ===
        # flow_with_onset: [B, T, 128] @100fps
        # SwinEncoder: patch_embed (4x) + S0 downsample (2x) = 8x temporal reduction
        # Output: [B, H, W, 192] @12.5fps where W = T/8
        swin_s0, swin_s1 = self.swin(flow_with_onset)  # each [B, H, W, 192]
        B_swin, H_swin, W_swin, D_swin = swin_s0.shape
        T_swin = W_swin  # time dimension

        # Concat S0+S1 along channel dim, then conv along freq axis.
        # [B, H, W, 192] × 2 → [B, H, W, 384] → [B, 384, H, W] → freq_conv → [B, bimamba_d_model, 1, W]
        swin_2d = torch.cat([swin_s0, swin_s1], dim=-1)          # [B, H, W, 384]
        swin_2d = swin_2d.permute(0, 3, 1, 2)                    # [B, 384, H, W]
        swin_2d = self.freq_conv(swin_2d)                         # [B, bimamba_d_model, 1, W]
        swin_feat = swin_2d.squeeze(2).permute(0, 2, 1)          # [B, W, bimamba_d_model]

        # === Step 4: BiMamba Encoder (time-oriented) ===
        # swin_feat: [B, T_swin, 384] @12.5fps → BiMamba projects to d_model
        if self.use_bimamba:
            feat_bimamba = self.bimamba_encoder(swin_feat)
            # feat_bimamba: [B, T_swin, 384] @12.5fps
            memory = feat_bimamba
            spatial_shapes = torch.tensor([[1, T_swin]], dtype=torch.long, device=mel.device)
            level_start_index = torch.tensor([0], dtype=torch.long, device=mel.device)
            valid_ratios = torch.ones(B, 1, 2, dtype=torch.float32, device=mel.device)
            return memory, spatial_shapes, level_start_index, valid_ratios

        # Fallback (should not happen since use_bimamba=True by default)
        memory = swin_feat
        spatial_shapes = torch.tensor([[1, T_swin]], dtype=torch.long, device=mel.device)
        level_start_index = torch.tensor([0], dtype=torch.long, device=mel.device)
        valid_ratios = torch.ones(B, 1, 2, dtype=torch.float32, device=mel.device)
        return memory, spatial_shapes, level_start_index, valid_ratios

    def forward(
        self,
        mel: torch.Tensor,              # [B, 1, 128, T]
        input_ids: torch.Tensor,        # [B, S]
        labels: Optional[torch.Tensor] = None,  # [B, S]
        ss_epsilon: float = 0.0,                # Scheduled sampling mixing ratio
        **_,                                    # absorb base-model-only kwargs (mel_valid_ratios, guidance_loss_weight, etc.)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass (matches ClefPianoBase signature).

        Args:
            mel: [B, 1, 128, T] mel spectrogram
            input_ids: [B, S] input token IDs
            labels: [B, S] target labels (for loss computation)
            ss_epsilon: Two-pass soft embedding scheduled sampling ratio.
                0.0 = pure teacher forcing (no overhead).
                0.2 = 20% of decoder input positions replaced with soft predicted embedding.

        Returns:
            logits: [B, S, vocab_size]
            loss: CE loss (if labels provided)
            total_loss: Same as loss (for compatibility)
        """
        # Encode
        memory, spatial_shapes, level_start_index, valid_ratios = self.encode(mel)

        tgt = self.token_embed(input_ids)  # [B, S, D]

        # Two-pass soft embedding scheduled sampling.
        #
        # Pass 1 (no_grad): run decoder on gold embeddings to get soft predicted embeddings.
        # Pass 2 (with grad): run decoder on mixed input (gold + soft predictions).
        #
        # Mixing rule (Two-pass, Mihaylova & Martins 2019 adapted for Mamba):
        #   position i input = soft_emb[i-1]  with prob ss_epsilon
        #                    = gold_emb[i]     with prob 1 - ss_epsilon
        # Shift by 1 so that position i sees the model's prediction for position i-1,
        # matching the sequential error propagation that occurs during AR inference.
        # Position 0 (SOS) is never replaced to keep the start token anchored.
        if self.training and ss_epsilon > 0.0:
            with torch.no_grad():
                _decoder_out = self.decoder(
                    tgt, memory, spatial_shapes, level_start_index, valid_ratios,
                    input_ids=input_ids,
                )
                _logits = self.output_projection(_decoder_out)
                # Soft embedding: weighted sum over vocab (fully differentiable via Pass 2)
                _probs = torch.softmax(_logits.float(), dim=-1).to(tgt.dtype)  # [B, S, V]
                soft_embs = _probs @ self.token_embed.weight                   # [B, S, D]

            # Shift: position i gets soft_emb[i-1]
            B, S, D = tgt.shape
            soft_shifted = torch.cat([tgt[:, :1, :], soft_embs[:, :-1, :]], dim=1)  # [B, S, D]

            # Sample replacement mask; never replace position 0 (SOS)
            mask = torch.zeros(B, S, dtype=torch.bool, device=tgt.device)
            mask[:, 1:] = torch.rand(B, S - 1, device=tgt.device) < ss_epsilon
            tgt = torch.where(mask.unsqueeze(-1), soft_shifted, tgt)

        decoder_out = self.decoder(
            tgt,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            input_ids=input_ids,
        )
        logits = self.output_projection(decoder_out)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # ── Curriculum: mask labels for compressed bars ──────────────────
            # decoder.forward sets _curriculum_mask=[B,S] True for zeroed positions.
            # We set those label positions to 0 (ignore_index) so loss ignores them.
            curriculum_mask = getattr(self.decoder, '_curriculum_mask', None)
            if curriculum_mask is not None and curriculum_mask.any():
                labels = labels.clone()
                labels[curriculum_mask] = 0  # ignore_index in CrossEntropyLoss

            loss_fn = nn.CrossEntropyLoss(
                ignore_index=0,  # <pad> token
                label_smoothing=self.config.label_smoothing if hasattr(self.config, 'label_smoothing') else 0.0,
            )
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

        return logits, loss, loss

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
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
        """Greedy generation.

        Args:
            mel: [B, 1, 128, T]
            max_len: maximum sequence length
            bos_token_id: beginning of sequence token
            eos_token_id: end of sequence token

        Returns:
            generated_ids: [B, S]
        """
        B = mel.shape[0]
        device = mel.device

        # Encode once
        memory, spatial_shapes, level_start_index, valid_ratios = self.encode(mel)

        # Initialize with BOS
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # Embed tokens
            tgt = self.token_embed(generated).contiguous()

            # Mamba2 workaround: when S=1, zxbcdt slice has stride(1)=total_dim=1804
            # which is not a multiple of 8. PyTorch sees [1,1,D] as already-contiguous
            # (size-1 dims), so ensure_stride's .contiguous() is a no-op → kernel crash.
            # Fix: repeat last token to make S=2 so .contiguous() actually recomputes strides.
            if tgt.shape[1] == 1:
                tgt = tgt.repeat(1, 2, 1)  # [B, 2, D], now S=2 → strides fix correctly

            # Decode
            decoder_out = self.decoder(
                tgt,
                memory,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                input_ids=generated,
            )
            if isinstance(decoder_out, tuple):
                decoder_out = decoder_out[0]
            logits = self.output_projection(decoder_out[:, -1:, :])  # [B, 1, vocab]
            next_token = logits.argmax(dim=-1)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated
