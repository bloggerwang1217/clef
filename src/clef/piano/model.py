"""
Clef Piano Base Model
=====================

ISMIR 2026 version: Swin V2 + ClefAttention

Architecture highlights:
- Swin V2 (frozen) extracts F1/F2/F3/F4 four-scale features
- Deformable Bridge: Sparse self-attention fuses multi-scale features
- ClefAttention Decoder: Content-aware Learned-prior Event Focusing
- Solves grace note problem: Can directly access F1 (10ms resolution, 100 fps)

ClefAttention features:
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
from ..decoder import ClefDecoder


class ClefPianoBase(nn.Module):
    """Clef Piano Base: Swin V2 + ClefAttention for piano transcription.

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

        # === Deformable Bridge ===
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
            refine_range=config.refine_range,
        )

        # === Output ===
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

        # Causal mask cache
        self._causal_mask_cache: Dict[int, torch.Tensor] = {}

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

        # Pad to multiple of 32 (Swin V2 window size constraint)
        padded_mel, (orig_H, orig_W) = self._pad_to_multiple(mel, multiple=32)
        _, _, H_pad, T_pad = padded_mel.shape

        # Compute valid ratios from original vs padded size
        # This tells attention "which part is real vs padding"
        valid_ratio_h = orig_H / H_pad  # freq axis (usually 1.0 since 128 % 32 == 0)
        valid_ratio_w = orig_W / T_pad  # time axis

        # Expand to 3 channels for Swin
        x = padded_mel.repeat(1, 3, 1, 1)  # [B, 3, H_pad, T_pad]

        # Run Swin encoder
        ctx = torch.no_grad() if self.config.freeze_encoder else nullcontext()
        with ctx:
            swin_out = self.swin(x, output_hidden_states=True)

        # Extract features from all 4 stages
        # hidden_states: [stage0(96), stage1(192), stage2(384), stage3(768), stage4(768)]
        # We use stages 0-3 (indices 0-3) to match swin_dims=[96, 192, 384, 768]
        features = []
        for i in range(4):
            feat = swin_out.hidden_states[i]  # [B, N_patches, C]
            features.append(feat)

        # Compute spatial shapes from PADDED dimensions (guaranteed divisible by 32)
        spatial_shapes_list = self._compute_spatial_shapes(H_pad, T_pad)

        # Compute per-level valid ratios [B, L, 2]
        # Each level has same valid ratio (time, freq)
        level_valid_ratios = torch.tensor(
            [[valid_ratio_w, valid_ratio_h] for _ in range(self.config.n_levels)],
            device=mel.device, dtype=mel.dtype
        ).unsqueeze(0).expand(B, -1, -1)  # [B, L, 2]

        # Run bridge with pre-computed spatial shapes and valid ratios
        memory, spatial_shapes, level_start_index, valid_ratios = self.bridge(
            features, spatial_shapes_list, level_valid_ratios
        )

        return memory, spatial_shapes, level_start_index, valid_ratios

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

    def decode(
        self,
        input_ids: torch.Tensor,  # [B, S]
        memory: torch.Tensor,     # [B, N_total, D]
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> torch.Tensor:
        """Decode with ClefAttention.

        Args:
            input_ids: Input token IDs [B, S]
            memory: Encoder features [B, N_total, D]
            spatial_shapes: [L, 2]
            level_start_index: [L]
            valid_ratios: [B, L, 2]

        Returns:
            logits: Output logits [B, S, vocab_size]
        """
        B, S = input_ids.shape

        # Token embedding
        tgt = self.token_embed(input_ids)  # [B, S, D]

        # Position embedding
        tgt_pos = self.decoder_pos_embed[:, :S, :]

        # Run decoder (uses is_causal=True for Flash Attention, no explicit mask needed)
        tgt = self.decoder(
            tgt, memory,
            spatial_shapes, level_start_index, valid_ratios,
            tgt_pos=tgt_pos,
        )

        # Output projection
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
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

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

        # Start with <sos>
        generated = torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Decode current sequence
            logits = self.decode(
                generated, memory, spatial_shapes, level_start_index, valid_ratios
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

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
