"""
SwinEncoder (S0 + S1)
=====================

Combined Swin S0+S1 encoder for acoustic feature extraction.

Design:
    Flow [B, T, 128] → SwinEncoder → feat_s0 [B, T/8, 192]  S0 post-downsample (pitch/harmony @ 12.5fps)
                                   → feat_s1 [B, T/8, 192]  S1 pre-downsample  (beat-aware    @ 12.5fps)

S0: patch_embed → S0 blocks → S0 downsample  → 192-dim, 8x temporal reduction
S1: S0 output  → S1 blocks  (no downsample)  → 192-dim, same temporal resolution, beat-level context added

Weights are extracted directly from a pretrained Swinv2Model.
Gradient checkpointing via torch.utils.checkpoint (per-block).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class SwinEncoder(nn.Module):
    """Combined Swin S0+S1 encoder.

    Args:
        input_dim: Input feature dimension (128 for Flow output).
        swin_model: HuggingFace model name to load pretrained weights from.
        use_gradient_checkpointing: Wrap each Swin block with grad checkpoint.
    """

    def __init__(
        self,
        input_dim: int = 128,
        swin_model: str = "microsoft/swinv2-tiny-patch4-window8-256",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        from transformers import Swinv2Model

        print(f"Loading SwinEncoder (S0+S1 pretrained weights): {swin_model}")
        full_swin = Swinv2Model.from_pretrained(swin_model)
        embed_dim = full_swin.config.embed_dim  # 96 for swinv2-tiny

        # --- patch_embed (adapt 3-channel → 1-channel) ---
        self.patch_embed = copy.deepcopy(full_swin.embeddings)
        old_proj = self.patch_embed.patch_embeddings.projection
        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.patch_embed.patch_embeddings.projection = new_proj

        # --- S0: blocks + downsample ---
        self.s0_blocks = nn.ModuleList(
            copy.deepcopy(b) for b in full_swin.encoder.layers[0].blocks
        )
        self.s0_downsample = copy.deepcopy(full_swin.encoder.layers[0].downsample)

        # --- S1: blocks only (no downsample — keep T/8 resolution) ---
        self.s1_blocks = nn.ModuleList(
            copy.deepcopy(b) for b in full_swin.encoder.layers[1].blocks
        )

        del full_swin

        # Both outputs are 192-dim (embed_dim * 2) at T/8 temporal resolution
        self.output_dim = embed_dim * 2  # 192

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ✓ SwinEncoder: {n_params/1e6:.2f}M params, "
              f"S0/S1 → [B, T/8, {self.output_dim}], "
              f"grad_ckpt={use_gradient_checkpointing}, compiled=False")

    def _run_blocks(self, x: torch.Tensor, blocks: nn.ModuleList,
                    H: int, W: int) -> torch.Tensor:
        """Run Swin blocks, optionally with per-block gradient checkpointing."""
        for block in blocks:
            if self.use_gradient_checkpointing and self.training:
                def _fn(x, b=block):
                    return b(x, input_dimensions=(H, W))[0]
                x = grad_checkpoint(_fn, x, use_reentrant=False)
            else:
                x = block(x, input_dimensions=(H, W))[0]
        return x

    def _to_1d(self, x: torch.Tensor, H: int, W: int, dim: int) -> torch.Tensor:
        """Reshape [B, H*W, dim] → mean-pool H → [B, W, dim]."""
        B = x.shape[0]
        return (x
                .view(B, H, W, dim)   # [B, H, W, dim]
                .permute(0, 2, 1, 3)  # [B, W, H, dim]
                .mean(dim=2))         # [B, W, dim]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through Swin S0+S1.

        Args:
            x: [B, T, C=128] time-major Flow output

        Returns:
            feat_s0: [B, T/8, 192] S0 post-downsample (pitch/harmony)
            feat_s1: [B, T/8, 192] S1 pre-downsample  (beat-aware, same resolution)
        """
        B, T, C = x.shape

        # Reshape to 2D image: [B, 1, C, T]
        x = x.permute(0, 2, 1).unsqueeze(1)

        # Pad to multiple of 8 (patch_embed 4x + S0 downsample 2x)
        pad_h = (8 - C % 8) % 8
        pad_w = (8 - T % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # patch_embed: [B, 1, C, T] → [B, N_pe, 96]  N_pe = (C/4)*(T/4)
        x, (H_pe, W_pe) = self.patch_embed(x)

        # S0 blocks + downsample → [B, N_s0, 192]  H_s0=C/8, W_s0=T/8
        x = self._run_blocks(x, self.s0_blocks, H_pe, W_pe)
        x = self.s0_downsample(x, (H_pe, W_pe))
        H_s0, W_s0 = (H_pe + 1) // 2, (W_pe + 1) // 2

        feat_s0 = self._to_1d(x, H_s0, W_s0, self.output_dim)  # [B, T/8, 192]

        # S1 blocks (no downsample) → beat-aware features at same resolution
        x = self._run_blocks(x, self.s1_blocks, H_s0, W_s0)

        feat_s1 = self._to_1d(x, H_s0, W_s0, self.output_dim)  # [B, T/8, 192]

        return feat_s0, feat_s1
