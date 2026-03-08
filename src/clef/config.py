"""
Clef Model Configuration
========================

Base configuration dataclass for Clef models.
Inherited by piano/solo/tutti variants.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


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
    swin_use_gradient_checkpointing: bool = False
    # Selective unfreeze: fine-tune specific Swin components while keeping attention frozen
    # Valid components: "patch_embed", "position_bias", "downsample"
    swin_unfreeze: List[str] = field(default_factory=list)
    swin_lr_scale: float = 0.1  # LR multiplier for unfrozen Swin params (vs base_lr)

    # === HarmonizingFlow (pitch space transform) ===
    use_flow: bool = False
    n_harmonics: int = 6
    flow_init: str = 'harmonic'  # 'harmonic' (physics) or 'orthogonal' (random)
    flow_pool_stride: int = 4    # Temporal pooling to match Swin S0 resolution

    # === Octopus2D (cross-frequency onset detection) ===
    use_octopus: bool = False          # Enable Octopus2D onset detector
    octopus_freq_kernel: int = 31      # Freq span (~4 harmonics of mid-register note)
    octopus_time_kernel: int = 3       # Time span (onset transient: 10-30ms)
    octopus_channels: int = 32         # Number of onset pattern detectors
    octopus_time_pool_stride: int = 2  # Temporal pool for Level 0 (T -> T/2, 20ms)
    octopus_freq_pool_stride: int = 4  # Frequency pool for Level 0 (128 -> 32)

    # === Swin Stage Selection ===
    swin_start_stage: int = 0  # Skip Swin stages before this index (0=use all, 1=skip S0)
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

    # Time prior: Two-stage Mamba (context + time)
    rope_base: float = 10000.0     # RoPE frequency base (same as SA layers)
    time_prior_d_state: int = 128  # Time Mamba state dimension (multi-scale timing)
    time_prior_d_state_context: int = 32  # Context Mamba state dimension (temporal structure)

    # === Bridge ===
    bridge_layers: int = 2

    # === Decoder ===
    decoder_layers: int = 6
    decoder_layer_types: List[str] = field(
        default_factory=lambda: ['sa_window_ca', 'mamba_only', 'sa_window_ca', 'mamba_only', 'sa_window_ca', 'mamba_only']
    )
    decoder_layer_ca_levels: Optional[List] = None  # per-layer active CA levels, None=all
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    max_seq_len: int = 4096
    vocab_size: int = 512   # Will be set from tokenizer
    # Window cross-attention (for 'window_ca' layer type)
    window_time_frames: Union[int, List[int]] = 16   # int (all levels) or List[int] (per level)
    window_freq_bins: Union[int, List[int]] = 8      # int (all levels) or List[int] (per level)
    window_seq_chunk_size: int = 10000               # process N_q in chunks; large = no chunking

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
        
        # For tiny model: SwinEncoder is internal, not a bridge level
        # n_levels should match: flow + octopus (if used) or just bimamba
        # Skip this validation for tiny model (swin_dims=[] or special config)
        if len(self.swin_dims) == 0:
            # Tiny model: n_levels should be octopus(1) + flow(1) or just bimamba(1)
            pass  # Skip validation for tiny model
        else:
            n_swin_used = len(self.swin_dims) - self.swin_start_stage
            use_bimamba = getattr(self, 'use_bimamba_encoder', False)
            expected_levels = (n_swin_used
                              + (1 if self.use_flow else 0)
                              + (1 if self.use_octopus else 0)
                              + (1 if use_bimamba else 0))
            assert self.n_levels == expected_levels, (
                f"n_levels({self.n_levels}) must be "
                f"swin({n_swin_used}) + flow({1 if self.use_flow else 0}) "
                f"+ octopus({1 if self.use_octopus else 0}) "
                f"+ bimamba({1 if use_bimamba else 0}) = {expected_levels}"
            )
