"""
BiMamba Encoder
===============

Time-oriented bidirectional Mamba encoder following Zeng et al. 2024's Bi-GRU design.

Design philosophy (following Zeng et al. 2024):
    1. Linear projection for frequency integration (all frequencies → compact features)
    2. Bi-Mamba for temporal modeling (onset, sustain, release dynamics)
    3. Output suitable for cross-attention lookups

Architecture:
    Input (B, W, C) — W: time steps, C: frequency features (like Zeng's freq_bins*40)
        → Linear projection (frequency integration)
        → Bi-Mamba (temporal modeling)
        → Output (B, W, d_model)

This matches Zeng's Bi-GRU exactly:
    - Linear layer integrates all frequencies at each time step
    - Bi-Mamba models temporal evolution (like Bi-GRU)
    - Preserves harmonic information (all frequencies visible at each time step)
"""

from typing import Optional

import torch
import torch.nn as nn


class BiMambaEncoder(nn.Module):
    """Bidirectional Mamba encoder with time-oriented processing (Zeng-style).

    Following Zeng et al. 2024's design philosophy:
        - Linear layer integrates all frequencies at each time step
        - Bi-Mamba models temporal evolution (like Bi-GRU)
        - Preserves harmonic information (all frequencies visible at each time step)

    Args:
        input_dim: Input feature dimension (C), e.g., 128 for mel/pitch features
        d_model: Hidden dimension after frequency integration
        d_state: Mamba state dimension (default: 128)
        d_conv: Mamba conv kernel size (default: 4)
        num_layers: Number of Bi-Mamba layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers

        # Frequency integration (like Zeng's Linear layer)
        # Input: C (all frequency bins concatenated)
        # Output: d_model (compact temporal features)
        self.freq_proj = nn.Linear(input_dim, d_model)

        # Bi-Mamba layers (temporal modeling)
        self.layers = nn.ModuleList([
            BiMambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with time-oriented Bi-Mamba.

        Args:
            x: [B, W, C] time-major input (like Zeng's Bi-GRU input)

        Returns:
            [B, W, d_model] temporally encoded features
        """
        # Frequency integration (like Zeng's Linear projection)
        x = self.freq_proj(x)
        # [B, W, d_model] — frequencies integrated into compact features

        # Apply Bi-Mamba layers (temporal modeling)
        for layer in self.layers:
            x = layer(x)
        # [B, W, d_model]

        return x


class BiMambaLayer(nn.Module):
    """Single Bi-Mamba layer with forward + backward processing.

    Architecture (Caduceus / RawBMamba style):
        x_forward  = Mamba(x)               # [B, W, D]
        x_backward = Mamba(reverse(x))      # [B, W, D], then re-reversed
        output = LayerNorm(out_proj(concat([x_fwd, x_bwd])) + residual)

    Why concat instead of add:
        - Matches Caduceus (arXiv:2403.03234) "BiMamba" design
        - RawBMamba (Interspeech 2024) shows concat beats add on audio tasks
        - Mamba SSM kernels are directional (Δ/A/B/C are causal); adding
          forward and backward in the same dimension causes gradient
          interference. Concat keeps them in separate channels.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Import Mamba2 here to avoid dependency issues
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                "mamba_ssm is required for BiMambaEncoder. "
                "Install with: pip install mamba-ssm"
            )

        # Forward and backward Mamba blocks (independent weights)
        self.mamba_forward = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
        )
        self.mamba_backward = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
        )

        # Caduceus-style fusion: concat [fwd, bwd] → project back to d_model
        self.out_proj = nn.Linear(d_model * 2, d_model)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional Mamba on time dimension.

        Args:
            x: [B, W, D] time-major sequences

        Returns:
            [B, W, D] bidirectionally encoded
        """
        residual = x

        # Forward pass
        x_forward = self.mamba_forward(x)                              # [B, W, D]

        # Backward pass (flip → Mamba → flip back)
        x_backward = self.mamba_backward(x.flip(dims=[1]))             # [B, W, D]
        x_backward = x_backward.flip(dims=[1])

        # Caduceus: concat and project (keeps forward/backward channels separate)
        x = self.out_proj(torch.cat([x_forward, x_backward], dim=-1)) # [B, W, D]

        # Residual + norm
        x = self.norm(x + residual)
        x = self.dropout(x)

        return x


def test_bimamba_encoder():
    """Test BiMambaEncoder with dummy data."""
    print("Testing BiMambaEncoder (Zeng-style)...")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping Mamba2 test (requires GPU)")
        print("BiMambaEncoder will work fine in actual training with CUDA.")
        return

    device = torch.device("cuda")

    # Config
    B, W, C = 2, 375, 128  # Flow output: time steps, 128 features (mel or pitch)
    d_model = 512

    # Create encoder
    encoder = BiMambaEncoder(
        input_dim=C,
        d_model=d_model,
        d_state=128,
        d_conv=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    # Test input (Flow output)
    x = torch.randn(B, W, C, device=device)
    print(f"Input (Flow): {x.shape} — (B, W, C)")

    # Forward
    output = encoder(x)
    print(f"Output: {output.shape} — (B, W, d_model)")

    assert output.shape == (B, W, d_model), f"Expected (B={B}, W={W}, d_model={d_model}), got {output.shape}"

    print("✓ BiMambaEncoder test passed!")


if __name__ == "__main__":
    test_bimamba_encoder()
