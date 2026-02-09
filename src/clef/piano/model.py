"""
Clef Piano Base Model
=====================

ISMIR 2026 version: Swin V2 + FluxAttention

Architecture highlights:
- Swin V2 (frozen) extracts F1/F2/F3/F4 four-scale features
- Deformable Bridge: Sparse self-attention fuses multi-scale features
- FluxAttention Decoder: Content-aware Learned-prior Event Focusing
- Solves grace note problem: Can directly access F1 (10ms resolution, 100 fps)

FluxAttention features:
- freq_prior: Predict "high or low frequency" from content (stream tracking)
- time_prior: Predict "which time point" from position
- Square sampling 2x2: prior locates, offset refines locally
"""

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swinv2Model

from .config import ClefPianoConfig
from ..bridge import DeformableBridge
from ..decoder import ClefDecoder, SADecoderLayer, MambaDecoderLayer
from ..flow import HarmonizingFlow, Octopus2D


class ClefPianoBase(nn.Module):
    """Clef Piano Base: Swin V2 + FluxAttention for piano transcription.

    Model flow:
    1. Mel spectrogram -> Swin V2 (frozen) -> F1, F2, F3, F4
    2. DeformableBridge fuses multi-scale features
    3. ClefDecoder generates **kern tokens autoregressively

    Memory estimation (RTX 3090, 24GB):
    - 4 min audio -> ~255K encoder tokens
    - Batch=2 + grad_accum=4 -> ~20GB
    """

    def __init__(self, config: ClefPianoConfig):
        super().__init__()
        self.config = config

        # === Encoder: Swin V2 (frozen) ===
        # Force CPU loading to prevent GPU memory leak during DDP
        # The model will be moved to the correct GPU later by Trainer
        self.swin = Swinv2Model.from_pretrained(
            config.swin_model,
            output_hidden_states=True,
            torch_dtype=torch.float32,  # Explicit dtype, no auto device
            low_cpu_mem_usage=True,     # Prevent GPU probing during loading
        )
        if config.freeze_encoder:
            self.swin.eval()
            for p in self.swin.parameters():
                p.requires_grad = False

        # === Octopus2D (optional, cross-frequency onset detection) ===
        self.use_octopus = getattr(config, 'use_octopus', False)
        if self.use_octopus:
            self.octopus = Octopus2D(
                freq_kernel=getattr(config, 'octopus_freq_kernel', 31),
                time_kernel=getattr(config, 'octopus_time_kernel', 3),
                channels=getattr(config, 'octopus_channels', 32),
                time_pool_stride=getattr(config, 'octopus_time_pool_stride', 2),
            )

        # === HarmonizingFlow (optional, pitch space transform) ===
        self.use_flow = getattr(config, 'use_flow', False)
        self.use_temporal_cnn = getattr(config, 'use_temporal_cnn', False)
        self.swin_on_pitch_space = getattr(config, 'swin_on_pitch_space', False)
        if self.use_flow:
            self.flow = HarmonizingFlow(
                n_mels=config.n_mels,
                n_pitches=88,
                n_harmonics=getattr(config, 'n_harmonics', 6),
                f_min=config.f_min,
                f_max=config.f_max,
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                init=getattr(config, 'flow_init', 'harmonic'),
                use_temporal_cnn=self.use_temporal_cnn,
                temporal_pool_stride=getattr(config, 'temporal_pool_stride', 8),
            )

        # === Deformable Bridge ===
        # Build bridge input dims based on active levels
        self.swin_start_stage = getattr(config, 'swin_start_stage', 0)
        swin_dims_used = config.swin_dims[self.swin_start_stage:]

        bridge_input_dims = None
        # Serial encoder: Octopus + Flow + Swin stages
        if self.use_octopus and self.swin_on_pitch_space:
            octopus_dim = getattr(config, 'octopus_channels', 32)
            bridge_input_dims = [octopus_dim, config.n_mels] + swin_dims_used
        elif self.use_flow:
            flow_dims = [config.n_mels]
            if self.use_temporal_cnn:
                flow_dims = [config.n_mels, config.n_mels]
            bridge_input_dims = flow_dims + swin_dims_used
        elif self.swin_start_stage > 0:
            bridge_input_dims = swin_dims_used

        self.bridge = DeformableBridge(
            swin_dims=config.swin_dims,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_levels=config.n_levels,
            n_points_freq=config.n_points_freq,
            n_points_time=config.n_points_time,
            freq_offset_scale=config.freq_offset_scale,
            time_offset_scale=config.time_offset_scale,
            n_layers=config.bridge_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            mel_height=config.n_mels,
            input_dims=bridge_input_dims,
        )

        # === Decoder ===
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder = ClefDecoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.decoder_layers,
            n_levels=config.n_levels,
            n_points_freq=config.n_points_freq,
            n_points_time=config.n_points_time,
            freq_offset_scale=config.freq_offset_scale,
            time_offset_scale=config.time_offset_scale,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            use_time_prior=config.use_time_prior,
            use_freq_prior=config.use_freq_prior,
            n_freq_groups=config.n_freq_groups,
            refine_range=config.refine_range,
            decoder_layer_types=config.decoder_layer_types,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )

        # === Output ===
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

        # Causal mask cache
        self._causal_mask_cache: Dict[int, torch.Tensor] = {}

        # Standard weight initialization for stable training
        # Skip Swin (pretrained) and Flow (physics-initialized) —
        # only init bridge, decoder, embeddings, output_head, octopus
        self.bridge.apply(self._init_weights)
        self.decoder.apply(self._init_weights)
        self._init_weights(self.token_embed)
        self._init_weights(self.output_head)
        if self.use_octopus:
            self.octopus.apply(self._init_weights)
        # NOTE: Flow.transform is NOT re-initialized here — it uses physics init

        # Re-apply special initialization that _init_weights overwrites.
        # _init_weights sets all Linear biases to 0, which destroys:
        # - FluxAttention offset bias (±0.5 grid pattern)
        # - FluxAttention level_weights bias (+1.0 level specialization)
        # - DecoderLayer freq_prior/time_prior/reference_refine bias
        # - PC gain bias (softplus(0.7) ≈ 1.0, "start open")
        for layer in self.decoder.layers:
            layer._init_reference_predictors()
            layer.cross_attn._reset_parameters()
            if hasattr(layer, 'ca_gain'):
                nn.init.constant_(layer.ca_gain.bias, 0.7)

    @staticmethod
    def _init_weights(module: nn.Module):
        """Initialize weights following GPT-2 / BART convention.

        Skips Swin V2 (frozen pretrained) and LayerNorm (default init is fine).
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def encode(
        self,
        mel: torch.Tensor,  # [B, 1, 128, T]
        mel_valid_ratios: Optional[torch.Tensor] = None,  # [B] (ignored, we compute from padding)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode mel spectrogram to multi-scale features.

        Args:
            mel: Log-mel spectrogram [B, 1, 128, T]
            mel_valid_ratios: Ignored (we compute from padding instead)

        Returns:
            memory: Fused features [B, N_total, D]
            spatial_shapes: [L, 2]
            level_start_index: [L]
            valid_ratios: [B, L, 2]
        """
        B, C, H, T = mel.shape

        features = []
        spatial_shapes_list = []
        valid_ratios_list = []

        if self.use_octopus and self.swin_on_pitch_space:
            # === Serial encoder: mel → Octopus → Flow → Swin (pitch space) ===
            self._encode_serial(
                mel, B, T, features, spatial_shapes_list, valid_ratios_list
            )
        else:
            # === Legacy parallel encoder: Flow || Swin (both on mel) ===
            self._encode_parallel(
                mel, B, features, spatial_shapes_list, valid_ratios_list
            )

        # Compute per-level valid ratios [B, L, 2]
        level_valid_ratios = torch.tensor(
            valid_ratios_list,
            device=mel.device, dtype=mel.dtype
        ).unsqueeze(0).expand(B, -1, -1)  # [B, L, 2]

        # Run bridge with pre-computed spatial shapes and valid ratios
        memory, spatial_shapes, level_start_index, valid_ratios = self.bridge(
            features, spatial_shapes_list, level_valid_ratios
        )

        return memory, spatial_shapes, level_start_index, valid_ratios

    def _encode_serial(
        self,
        mel: torch.Tensor,
        B: int,
        T: int,
        features: list,
        spatial_shapes_list: list,
        valid_ratios_list: list,
    ) -> None:
        """Serial encoder: mel → Octopus2D → Flow → Swin on pitch space.

        Brain-inspired serial pipeline:
        - Octopus2D = CN octopus cells (cross-frequency onset detection)
        - Flow = IC harmonic integration (pitch identification)
        - Swin = A1 cortex (multi-scale 2D processing in pitch space)
        """
        # Step 1: Octopus2D — cross-frequency onset detection on raw mel
        enhanced_mel, onset_level = self.octopus(mel)
        # onset_level: [B, T//pool, C]
        T_onset = onset_level.shape[1]
        features.append(onset_level)
        spatial_shapes_list.append((1, T_onset))
        valid_ratios_list.append([1.0, 1.0])

        # Step 2: Flow — harmonic template matching on onset-enhanced mel
        # use_temporal_cnn=False in serial mode, so flow returns tensor directly
        flow_feat = self.flow(enhanced_mel)  # [B, T, 128]

        # Temporal pooling
        pool_stride = getattr(self.config, 'flow_pool_stride', 4)
        if pool_stride > 1:
            flow_feat = flow_feat.permute(0, 2, 1)  # [B, 128, T]
            flow_feat = F.avg_pool1d(
                flow_feat, kernel_size=pool_stride, stride=pool_stride
            )
            flow_feat = flow_feat.permute(0, 2, 1)  # [B, T//4, 128]

        T_flow = flow_feat.shape[1]
        features.append(flow_feat)
        spatial_shapes_list.append((1, T_flow))
        valid_ratios_list.append([1.0, 1.0])

        # Step 3: Reshape Flow output as pitch-space "image" for Swin
        # flow_feat: [B, T_flow, 128] → [B, 1, 128, T_flow]
        swin_input = flow_feat.detach().permute(0, 2, 1).unsqueeze(1)
        swin_input, (orig_H_s, orig_W_s) = self._pad_to_multiple(swin_input, 32)
        swin_input = swin_input.repeat(1, 3, 1, 1)  # [B, 3, 128, T_pad]
        _, _, H_pad_s, T_pad_s = swin_input.shape

        valid_ratio_h_s = orig_H_s / H_pad_s  # 128/128 = 1.0
        valid_ratio_w_s = orig_W_s / T_pad_s

        # Step 4: Run frozen Swin on pitch-space image
        ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with ctx:
            swin_out = self.swin(swin_input, output_hidden_states=True)

        # Step 5: Collect Swin stages
        n_swin_stages = len(self.config.swin_dims)
        all_swin_shapes = self._compute_spatial_shapes(H_pad_s, T_pad_s)
        swin_pools = getattr(self.config, 'swin_pool_strides', [1] * n_swin_stages)

        for i in range(self.swin_start_stage, n_swin_stages):
            feat = swin_out.hidden_states[i]  # [B, N_patches, C]
            H_s, W_s = all_swin_shapes[i]
            pool_s = swin_pools[i] if i < len(swin_pools) else 1

            if pool_s > 1:
                feat = feat.view(B, H_s, W_s, -1).permute(0, 3, 1, 2)
                feat = F.avg_pool2d(feat, kernel_size=(1, pool_s),
                                    stride=(1, pool_s))
                feat = feat.permute(0, 2, 3, 1)
                W_s = feat.shape[2]
                feat = feat.reshape(B, H_s * W_s, -1)

            features.append(feat)
            spatial_shapes_list.append((H_s, W_s))
            valid_ratios_list.append([valid_ratio_w_s, valid_ratio_h_s])

    def _encode_parallel(
        self,
        mel: torch.Tensor,
        B: int,
        features: list,
        spatial_shapes_list: list,
        valid_ratios_list: list,
    ) -> None:
        """Legacy parallel encoder: Flow and Swin both read raw mel independently."""
        # Pad to multiple of 32 (Swin V2 window size constraint)
        padded_mel, (orig_H, orig_W) = self._pad_to_multiple(mel, multiple=32)
        _, _, H_pad, T_pad = padded_mel.shape

        valid_ratio_h = orig_H / H_pad
        valid_ratio_w = orig_W / T_pad

        # Expand to 3 channels for Swin
        x = padded_mel.repeat(1, 3, 1, 1)  # [B, 3, H_pad, T_pad]

        # Run Swin encoder
        ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with ctx:
            swin_out = self.swin(x, output_hidden_states=True)

        # Level 0 (+Level 1 if TemporalCNN): HarmonizingFlow
        if self.use_flow:
            flow_out = self.flow(mel)  # tuple or tensor

            if self.use_temporal_cnn:
                flow_feat, temporal_feat = flow_out
            else:
                flow_feat = flow_out

            pool_stride = getattr(self.config, 'flow_pool_stride', 4)
            if pool_stride > 1:
                flow_feat = flow_feat.permute(0, 2, 1)
                flow_feat = F.avg_pool1d(flow_feat, kernel_size=pool_stride,
                                         stride=pool_stride)
                flow_feat = flow_feat.permute(0, 2, 1)

            T_flow = flow_feat.shape[1]
            features.append(flow_feat)
            spatial_shapes_list.append((1, T_flow))
            valid_ratios_list.append([1.0, 1.0])

            if self.use_temporal_cnn:
                T_temporal = temporal_feat.shape[1]
                features.append(temporal_feat)
                spatial_shapes_list.append((1, T_temporal))
                valid_ratios_list.append([1.0, 1.0])

        # Swin stages
        n_swin_stages = len(self.config.swin_dims)
        all_swin_shapes = self._compute_spatial_shapes(H_pad, T_pad)
        swin_pools = getattr(self.config, 'swin_pool_strides', [1] * n_swin_stages)

        for i in range(self.swin_start_stage, n_swin_stages):
            feat = swin_out.hidden_states[i]
            H_s, W_s = all_swin_shapes[i]
            pool_s = swin_pools[i] if i < len(swin_pools) else 1

            if pool_s > 1:
                feat = feat.view(B, H_s, W_s, -1).permute(0, 3, 1, 2)
                feat = F.avg_pool2d(feat, kernel_size=(1, pool_s),
                                    stride=(1, pool_s))
                feat = feat.permute(0, 2, 3, 1)
                W_s = feat.shape[2]
                feat = feat.reshape(B, H_s * W_s, -1)

            features.append(feat)
            spatial_shapes_list.append((H_s, W_s))
            valid_ratios_list.append([valid_ratio_w, valid_ratio_h])

    def _pad_to_multiple(
        self, mel: torch.Tensor, multiple: int = 32
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad mel spectrogram to be divisible by Swin window size.

        Args:
            mel: [B, 1, H, W]
            multiple: Must be divisible by this (Swin window_size * patch_size)

        Returns:
            padded_mel: [B, 1, H_pad, W_pad]
            original_size: (H, W) for computing valid_ratios
        """
        B, C, H, W = mel.shape

        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            mel = F.pad(mel, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return mel, (H, W)

    def _compute_spatial_shapes(
        self, mel_height: int, mel_width: int
    ) -> List[Tuple[int, int]]:
        """Compute spatial shapes for each Swin stage.

        Swin V2 downsampling pattern:
        - Stage 1: H/4, W/4
        - Stage 2: H/8, W/8
        - Stage 3: H/16, W/16
        - Stage 4: H/32, W/32

        For mel_height=128:
        - Stage 1: 32, W/4
        - Stage 2: 16, W/8
        - Stage 3: 8, W/16
        - Stage 4: 4, W/32
        """
        shapes = []
        for stage in range(4):
            divisor = 4 * (2 ** stage)  # 4, 8, 16, 32
            h = mel_height // divisor
            w = mel_width // divisor
            shapes.append((h, w))
        return shapes

    def prepare_value_cache(
        self,
        memory: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> list:
        """Pre-compute cross-attention value projections for all decoder layers.

        Call once per chunk, pass result to decode() for all steps.
        Avoids recomputing value_proj(memory) at every decoding step.
        """
        value_cache_list = []
        for layer in self.decoder.layers:
            cache = layer.cross_attn.compute_value_cache(
                memory, spatial_shapes, level_start_index,
            )
            value_cache_list.append(cache)
        return value_cache_list

    def _get_cached_len(self, past_states: list) -> int:
        """Get the number of previously cached tokens from past_states.

        Checks SA layers for KV-cache shape, or Mamba InferenceParams for seqlen_offset.
        """
        for i, layer in enumerate(self.decoder.layers):
            state = past_states[i]
            if state is None:
                continue
            if isinstance(layer, SADecoderLayer) and isinstance(state, tuple):
                return state[0].shape[2]  # K cache: [B, H, S_cached, D_head]
            if hasattr(state, 'seqlen_offset'):
                return state.seqlen_offset  # Mamba InferenceParams
        return 0

    def decode(
        self,
        input_ids: torch.Tensor,  # [B, S]
        memory: torch.Tensor,     # [B, N_total, D]
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        past_states: Optional[list] = None,
        use_cache: bool = False,
        value_cache: Optional[list] = None,
    ):
        """Decode with Jamba hybrid decoder.

        Args:
            input_ids: Input token IDs [B, S]
            memory: Encoder features [B, N_total, D]
            spatial_shapes: [L, 2]
            level_start_index: [L]
            valid_ratios: [B, L, 2]
            past_states: List of per-layer state (KV tuple for SA, InferenceParams for Mamba)
            use_cache: Whether to return updated states
            value_cache: Pre-computed cross-attn value maps (from prepare_value_cache)

        Returns:
            logits: Output logits [B, S, vocab_size]
            new_states: (only if use_cache=True) list of states per layer
        """
        B, S = input_ids.shape

        # Token embedding
        tgt = self.token_embed(input_ids)  # [B, S, D]

        # Position embedding: use correct position offset when using cache
        if past_states is not None:
            cached_len = self._get_cached_len(past_states)
            tgt_pos = self.decoder_pos_embed[:, cached_len:cached_len + S, :]
        else:
            tgt_pos = self.decoder_pos_embed[:, :S, :]

        # Run decoder
        if use_cache:
            tgt, new_states = self.decoder(
                tgt, memory,
                spatial_shapes, level_start_index, valid_ratios,
                tgt_pos=tgt_pos,
                past_states=past_states,
                use_cache=True,
                value_cache_list=value_cache,
            )
            # Increment Mamba InferenceParams.seqlen_offset after each step.
            # mamba-ssm does NOT auto-increment; the caller must do it.
            for state in new_states:
                if hasattr(state, 'seqlen_offset'):
                    state.seqlen_offset += S
                    break  # shared object, only increment once
            logits = self.output_head(tgt)
            return logits, new_states
        else:
            tgt = self.decoder(
                tgt, memory,
                spatial_shapes, level_start_index, valid_ratios,
                tgt_pos=tgt_pos,
            )
            logits = self.output_head(tgt)
            return logits

    def forward(
        self,
        mel: torch.Tensor,              # [B, 1, 128, T]
        input_ids: torch.Tensor,        # [B, S]
        labels: Optional[torch.Tensor] = None,  # [B, S]
        mel_valid_ratios: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Full forward pass.

        Args:
            mel: Log-mel spectrogram [B, 1, 128, T]
            input_ids: Input token IDs (shifted right) [B, S]
            labels: Target labels (original sequence) [B, S]
            mel_valid_ratios: Time-axis valid ratio [B]

        Returns:
            logits: Output logits [B, S, vocab_size]
            loss: Cross-entropy loss (if labels provided)
        """
        # Encode
        memory, spatial_shapes, level_start_index, valid_ratios = self.encode(
            mel, mel_valid_ratios
        )

        # Decode
        logits = self.decode(
            input_ids, memory, spatial_shapes, level_start_index, valid_ratios
        )

        # Compute loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=0,  # <pad> token
                label_smoothing=self.config.label_smoothing,
            )
            ce_loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            loss = ce_loss

            # Predictive coding auxiliary loss (predictor MSE)
            pred_loss = self.decoder.collect_pred_loss()
            if pred_loss is not None:
                loss = loss + self.config.pred_loss_weight * pred_loss
                self._last_pred_loss = pred_loss.detach()
            else:
                self._last_pred_loss = None

        return logits, loss

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal attention mask.

        Returns:
            mask: [S, S] with -inf for positions that should not attend
        """
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
            self._causal_mask_cache[seq_len] = mask

        return self._causal_mask_cache[seq_len].to(device)

    def _init_inference_states(self, batch_size: int, max_length: int, device: torch.device) -> list:
        """Initialize per-layer inference states.

        SA layers: None (KV-cache built incrementally)
        Mamba layers: shared InferenceParams object (mamba-ssm manages per-layer state)
        """
        from mamba_ssm.utils.generation import InferenceParams
        mamba_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

        past_states = []
        for layer in self.decoder.layers:
            if isinstance(layer, MambaDecoderLayer):
                past_states.append(mamba_params)  # shared object
            else:
                past_states.append(None)  # SA, no cache yet
        return past_states

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        max_length: int = 4096,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Generate token sequence autoregressively.

        Args:
            mel: Log-mel spectrogram [B, 1, 128, T]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)
            top_p: Top-p (nucleus) sampling
            sos_token_id: Start token ID
            eos_token_id: End token ID

        Returns:
            generated: Generated token IDs [B, L]
        """
        B = mel.shape[0]
        device = mel.device

        # Encode once
        memory, spatial_shapes, level_start_index, valid_ratios = self.encode(mel)

        # Initialize inference states (Mamba InferenceParams + SA KV-cache)
        past_states = self._init_inference_states(B, max_length, device)

        # Start with <sos>
        generated = torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)
        input_ids = generated

        for _ in range(max_length - 1):
            # Decode with cache
            logits, past_states = self.decode(
                input_ids, memory, spatial_shapes, level_start_index, valid_ratios,
                past_states=past_states, use_cache=True,
            )

            # Get last token logits
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Apply top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for b in range(B):
                    next_logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float('-inf')

            # Sample or greedy
            if temperature > 0 and (top_k is not None or top_p is not None):
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Append
            generated = torch.cat([generated, next_token], dim=1)
            input_ids = next_token  # next step: single token

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
