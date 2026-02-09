"""
Clef Model Configuration
========================

Base configuration dataclass for Clef models.
Inherited by piano/solo/tutti variants.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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

    # === HarmonizingFlow (pitch space transform) ===
    use_flow: bool = False
    n_harmonics: int = 6
    flow_init: str = 'harmonic'  # 'harmonic' (physics) or 'orthogonal' (random)
    flow_pool_stride: int = 4    # Temporal pooling to match Swin S0 resolution
    use_temporal_cnn: bool = False  # Causal 1D CNN on Flow output for onset/duration
    temporal_pool_stride: int = 4   # Pool stride for temporal CNN output (T -> T/4)

    # === Octopus2D (cross-frequency onset detection) ===
    use_octopus: bool = False          # Enable Octopus2D onset detector
    octopus_freq_kernel: int = 31      # Freq span (~4 harmonics of mid-register note)
    octopus_time_kernel: int = 3       # Time span (onset transient: 10-30ms)
    octopus_channels: int = 32         # Number of onset pattern detectors
    octopus_time_pool_stride: int = 2  # Temporal pool for Level 0 (T -> T/2, 20ms)

    # === Swin Stage Selection ===
    swin_start_stage: int = 0  # Skip Swin stages before this index (0=use all, 1=skip S0)
    swin_on_pitch_space: bool = False  # True = serial (Swin eats Flow output, not raw mel)
    # Time-axis pool per Swin stage (remove overlapping-window redundancy)
    # Swin0: RF~320ms, grid 40ms → pool 8x (Nyquist, chord is discrete)
    # Swin1: RF~640ms, grid 80ms → pool 4x (50% overlap, melody needs continuity)
    # Swin2/3: keep as-is (already compact)
    swin_pool_strides: List[int] = field(default_factory=lambda: [1, 1, 1, 1])

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
    n_freq_groups: int = 1         # Per-head freq_prior groups (1=shared, n_heads=fully independent)
    refine_range: float = 0.1      # +/-10% refinement

    # === Bridge ===
    bridge_layers: int = 2

    # === Decoder ===
    ca_gate_type: str = 'predictive_coding'
    pred_loss_weight: float = 0.1     # Predictor MSE loss weight (predictive_coding mode only)
    decoder_layers: int = 6
    decoder_layer_types: List[str] = field(
        default_factory=lambda: ['mamba', 'mamba', 'sa', 'mamba', 'mamba', 'sa']
    )
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
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
        # Derive decoder_layers from decoder_layer_types for backward compat
        self.decoder_layers = len(self.decoder_layer_types)
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        n_swin_used = len(self.swin_dims) - self.swin_start_stage
        expected_levels = (n_swin_used
                          + (1 if self.use_flow else 0)
                          + (1 if self.use_octopus else 0)
                          + (1 if self.use_temporal_cnn else 0))
        assert self.n_levels == expected_levels, (
            f"n_levels({self.n_levels}) must be "
            f"swin({n_swin_used}) + flow({1 if self.use_flow else 0}) "
            f"+ octopus({1 if self.use_octopus else 0}) "
            f"+ cnn({1 if self.use_temporal_cnn else 0}) = {expected_levels}"
        )
        assert self.n_points_freq > 0, "n_points_freq must be positive"
        assert self.n_points_time > 0, "n_points_time must be positive"
