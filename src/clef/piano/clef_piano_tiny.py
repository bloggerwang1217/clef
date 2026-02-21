"""
Clef Piano Tiny Model
=====================

Minimal PoC: Octopus + Flow + Swin S0 only + 2-layer decoder

Goal: Beat SOTA with minimal architecture on short clips (30s).
This is a clean rewrite, importing modules from the main codebase.

Architecture:
- Encoder: Octopus2D → Flow → Swin S0 only (3 levels total)
- Decoder: SA+Full CA (to S0) + Mamba → Output
"""

from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swinv2Model

from .config import ClefPianoConfig
from ..bridge import MultiScaleBridge
from ..decoder import ClefDecoder
from ..flow import HarmonizingFlow, Octopus2D


class ClefPianoTiny(nn.Module):
    """Minimal Clef Piano model for 30-second audio clips.

    Encoder: Octopus + Flow + Swin S0 only (3 levels)
    Decoder: 2 layers (SA+Full CA + Mamba)

    Target: ~1024 tokens max sequence length for 30s audio.
    """

    def __init__(self, config: ClefPianoConfig):
        super().__init__()
        self.config = config

        # === Encoder: Swin V2 (frozen) ===
        self.swin = Swinv2Model.from_pretrained(
            config.swin_model,
            output_hidden_states=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        if config.freeze_encoder:
            self.swin.eval()
            for p in self.swin.parameters():
                p.requires_grad = False
            # Selective unfreeze
            swin_unfreeze = getattr(config, 'swin_unfreeze', [])
            if swin_unfreeze:
                self._unfreeze_swin_components(swin_unfreeze)

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

        # === Bridge: minimal (1 layer) ===
        # Input dimensions for each level:
        # Level 0: Octopus (32 channels)
        # Level 1: Flow (128 dims)
        # Level 2: Swin S0 (192 dims - hidden_states[1] after first stage)
        octopus_channels = getattr(config, 'octopus_channels', 32)
        s0_dim = 192  # Swin V2 Tiny: embedding=96, S0=192, S1=384, S2=768, S3=768
        input_dims = [octopus_channels, config.n_mels, s0_dim]

        self.bridge = MultiScaleBridge(
            swin_dims=config.swin_dims,  # Keep original for compatibility
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_levels=config.n_levels,  # 3: Octopus + Flow + S0
            n_points_freq=config.n_points_freq,
            n_points_time=config.n_points_time,
            freq_offset_scale=config.freq_offset_scale,
            time_offset_scale=config.time_offset_scale,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            n_layers=getattr(config, 'bridge_layers', 1),
            input_dims=input_dims,  # Specify actual input dimensions
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
            use_rope=config.use_rope,
            # Bar/Note GRU config
            bar_token_id=getattr(config, 'bar_token_id', None),
            onset_1d_channels=getattr(config, 'octopus_channels', 32),
            bar_gru_hidden_size=getattr(config, 'bar_gru_hidden_size', 256),
            bar_gru_input_dropout=getattr(config, 'bar_gru_input_dropout', 0.1),
            note_gru_hidden_size=getattr(config, 'note_gru_hidden_size', 256),
            note_gru_input_dropout=getattr(config, 'note_gru_input_dropout', 0.1),
        )

        # === Token embedding ===
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)

        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        # Gradient checkpointing (enabled via config)
        if getattr(config, 'gradient_checkpointing', False):
            self.decoder.gradient_checkpointing = True
            if hasattr(self.swin, 'gradient_checkpointing_enable'):
                self.swin.gradient_checkpointing_enable()

    def _unfreeze_swin_components(self, components: list):
        """Selectively unfreeze Swin components."""
        for name, param in self.swin.named_parameters():
            for comp in components:
                if comp in name:
                    param.requires_grad = True
                    break

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
        flow_feat = self.flow(mel)  # [B, T, 128]

        # Temporal pooling
        pool_stride = getattr(self.config, 'flow_pool_stride', 4)
        if pool_stride > 1:
            flow_feat = flow_feat.permute(0, 2, 1)  # [B, 128, T]
            flow_feat = F.avg_pool1d(flow_feat, kernel_size=pool_stride, stride=pool_stride)
            flow_feat = flow_feat.permute(0, 2, 1)  # [B, T//4, 128]

        T_flow = flow_feat.shape[1]

        # === Step 2.5: Integrate onset into pitch space ===
        onset_signal = self.onset_to_pitch(onset_raw).squeeze(1)  # [B, 128, T]

        if pool_stride > 1:
            onset_signal = F.avg_pool1d(onset_signal, kernel_size=pool_stride, stride=pool_stride)

        onset_signal = onset_signal.permute(0, 2, 1)  # [B, T//4, 128]
        onset_in_pitch = onset_signal @ self.flow.transform.T  # [B, T//4, 128]

        flow_with_onset = flow_feat + onset_in_pitch

        features.append(flow_with_onset)
        spatial_shapes_list.append((1, T_flow))
        valid_ratios_list.append([1.0, 1.0])

        # === Step 3: Swin S0 only ===
        swin_input = flow_with_onset.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 128, T_flow]
        swin_input, (orig_H_s, orig_W_s) = self._pad_to_multiple(swin_input, 32)
        swin_input = swin_input.repeat(1, 3, 1, 1)  # [B, 3, 128, T_pad]
        _, _, H_pad_s, T_pad_s = swin_input.shape

        valid_ratio_h_s = orig_H_s / H_pad_s
        valid_ratio_w_s = orig_W_s / T_pad_s

        # Run Swin
        swin_fully_frozen = (self.config.freeze_encoder
                             and not getattr(self.config, 'swin_unfreeze', []))
        ctx = torch.no_grad() if swin_fully_frozen else nullcontext()
        with ctx:
            swin_out = self.swin(swin_input, output_hidden_states=True)

        # Extract S0 only (hidden_states[1] = S0 output)
        all_swin_shapes = self._compute_spatial_shapes(H_pad_s, T_pad_s)
        feat_s0 = swin_out.hidden_states[1]  # [B, N_patches, C]
        H_s0, W_s0 = all_swin_shapes[0]

        features.append(feat_s0)
        spatial_shapes_list.append((H_s0, W_s0))
        valid_ratios_list.append([valid_ratio_w_s, valid_ratio_h_s])

        # === Bridge fusion ===
        # Convert valid ratios to tensor
        level_valid_ratios = torch.tensor(
            valid_ratios_list,
            device=mel.device,
            dtype=torch.float32
        ).unsqueeze(0).expand(B, -1, -1)  # [B, 3, 2]

        # Bridge will handle projection and concatenation
        memory, spatial_shapes, level_start_index, valid_ratios = self.bridge(
            features, spatial_shapes_list, level_valid_ratios
        )

        return memory, spatial_shapes, level_start_index, valid_ratios

    def forward(
        self,
        mel: torch.Tensor,              # [B, 1, 128, T]
        input_ids: torch.Tensor,        # [B, S]
        labels: Optional[torch.Tensor] = None,  # [B, S]
        mel_valid_ratios: Optional[torch.Tensor] = None,  # Ignored for tiny
        chunk_audio_measures: Optional[list] = None,      # Ignored for tiny
        chunk_start_frames: Optional[list] = None,        # Ignored for tiny
        chunk_end_frames: Optional[list] = None,          # Ignored for tiny
        guidance_loss_weight: Optional[float] = None,     # Ignored for tiny
        tf_ratio: float = 1.0,                            # Ignored for tiny
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass (matches ClefPianoBase signature).

        Args:
            mel: [B, 1, 128, T] mel spectrogram
            input_ids: [B, S] input token IDs
            labels: [B, S] target labels (for loss computation)
            mel_valid_ratios: Ignored (for compatibility)
            chunk_audio_measures: Ignored (no guided attention)
            chunk_start_frames: Ignored (no guided attention)
            chunk_end_frames: Ignored (no guided attention)
            guidance_loss_weight: Ignored (no guided attention)
            tf_ratio: Ignored (no teacher forcing)

        Returns:
            logits: [B, S, vocab_size]
            loss: CE loss (if labels provided)
            total_loss: Same as loss (for compatibility)
        """
        # Encode
        memory, spatial_shapes, level_start_index, valid_ratios = self.encode(mel)

        # Embed tokens
        tgt = self.token_embed(input_ids)  # [B, S] → [B, S, D]

        # Decode
        decoder_out = self.decoder(
            tgt,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        )
        logits = self.output_projection(decoder_out)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=0,  # <pad> token
                label_smoothing=self.config.label_smoothing if hasattr(self.config, 'label_smoothing') else 0.0,
            )
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
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
            # Decode
            decoder_out = self.decoder(
                generated,
                memory,
                spatial_shapes,
                level_start_index,
                valid_ratios,
            )
            logits = self.output_projection(decoder_out[:, -1:, :])  # [B, 1, vocab]
            next_token = logits.argmax(dim=-1)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated
