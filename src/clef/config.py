"""
Clef Model Configuration
========================

Base configuration dataclass for Clef models.
Inherited by piano/solo/tutti variants.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClefConfig:
    """Base configuration for Clef models.

    Contains all architectural hyperparameters.
    Subclasses can extend with domain-specific settings.
    """

    # === Swin V2 Encoder ===
    swin_model: str = "microsoft/swinv2-tiny-patch4-window8-256"
    swin_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    freeze_encoder: bool = True

    # === Deformable Attention (CLEF) ===
    d_model: int = 512
    n_heads: int = 8
    n_levels: int = 4
    ff_dim: int = 2048
    dropout: float = 0.1

    # Square sampling + Content-Dependent Prior
    n_points_freq: int = 2         # Frequency direction: local detail
    n_points_time: int = 2         # Time direction: local detail
    freq_offset_scale: float = 0.15   # +/-15% (same as time for square)
    time_offset_scale: float = 0.15   # +/-15% (same as freq for square)

    # Content-Dependent Reference Points
    use_time_prior: bool = True    # time_prior(tgt_pos) -> time location
    use_freq_prior: bool = True    # freq_prior(tgt) -> freq region
    refine_range: float = 0.1      # +/-10% refinement

    # === Bridge ===
    bridge_layers: int = 2

    # === Decoder ===
    decoder_layers: int = 6
    max_seq_len: int = 4096
    vocab_size: int = 512   # Will be set from tokenizer

    # === Audio Transform ===
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 160   # 100 fps (matches Zeng for fair comparison)
    f_min: float = 27.5     # A0 (matches Zeng)
    f_max: float = 7040.0

    # === Training ===
    label_smoothing: float = 0.0

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_levels == len(self.swin_dims), "n_levels must match swin_dims"
        assert self.n_points_freq > 0, "n_points_freq must be positive"
        assert self.n_points_time > 0, "n_points_time must be positive"
