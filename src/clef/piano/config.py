"""
Clef Piano Configuration
========================

Configuration for clef-piano-base (ISMIR 2026) model.
Extends base ClefConfig with piano-specific settings.
"""

from dataclasses import dataclass
from typing import Optional

from ..config import ClefConfig


@dataclass
class ClefPianoConfig(ClefConfig):
    """Configuration for clef-piano-base model.

    ISMIR 2026 target: Single-instrument (piano) transcription
    using Swin V2 encoder + WindowCrossAttention decoder.
    """

    # Piano-specific defaults
    vocab_size: int = 512   # ~220 factorized tokens + padding

    # Chunking for long pieces
    chunk_frames: int = 24000           # 4 min @ 100 fps (primary)
    overlap_frames: int = 12000         # 2 min overlap (primary)
    min_chunk_ratio: float = 0.5        # Last chunk at least 2 min
    # Fallback chunking: used when full-piece token count > max_seq_len
    fallback_chunk_frames: int = 12000  # 2 min fallback chunk
    fallback_overlap_frames: int = 6000 # 1 min fallback overlap

    # Piano-specific: often right-hand (high freq) + left-hand (low freq)
    # freq_prior learns to focus on relevant register
    use_freq_prior: bool = True

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    training_seed: int = 1234

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ClefPianoConfig":
        """Load config from YAML file."""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Extract model config
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        training_cfg = cfg.get("training", {})
        seed_cfg = cfg.get("seed", {})
        audio_cfg = data_cfg.get("audio", {})
        chunk_cfg = data_cfg.get("chunking", {})

        # Get defaults from a temporary instance
        defaults = cls()

        return cls(
            # Swin
            swin_model=model_cfg.get("swin_model", defaults.swin_model),
            swin_dims=model_cfg.get("swin_dims", defaults.swin_dims),
            freeze_encoder=model_cfg.get("freeze_encoder", defaults.freeze_encoder),
            swin_unfreeze=model_cfg.get("swin_unfreeze", defaults.swin_unfreeze),
            swin_lr_scale=model_cfg.get("swin_lr_scale", defaults.swin_lr_scale),

            # HarmonizingFlow
            use_flow=model_cfg.get("use_flow", defaults.use_flow),
            n_harmonics=model_cfg.get("n_harmonics", defaults.n_harmonics),
            flow_init=model_cfg.get("flow_init", defaults.flow_init),
            flow_pool_stride=model_cfg.get("flow_pool_stride", defaults.flow_pool_stride),
            use_temporal_cnn=model_cfg.get("use_temporal_cnn", defaults.use_temporal_cnn),
            temporal_pool_stride=model_cfg.get("temporal_pool_stride", defaults.temporal_pool_stride),

            # Octopus2D
            use_octopus=model_cfg.get("use_octopus", defaults.use_octopus),
            octopus_freq_kernel=model_cfg.get("octopus_freq_kernel", defaults.octopus_freq_kernel),
            octopus_time_kernel=model_cfg.get("octopus_time_kernel", defaults.octopus_time_kernel),
            octopus_channels=model_cfg.get("octopus_channels", defaults.octopus_channels),
            octopus_time_pool_stride=model_cfg.get("octopus_time_pool_stride", defaults.octopus_time_pool_stride),

            # Swin input mode
            swin_start_stage=model_cfg.get("swin_start_stage", defaults.swin_start_stage),
            swin_pool_strides=model_cfg.get("swin_pool_strides", defaults.swin_pool_strides),
            swin_on_pitch_space=model_cfg.get("swin_on_pitch_space", defaults.swin_on_pitch_space),

            # Attention
            d_model=model_cfg.get("d_model", defaults.d_model),
            n_heads=model_cfg.get("n_heads", defaults.n_heads),
            n_levels=model_cfg.get("n_levels", defaults.n_levels),
            ff_dim=model_cfg.get("ff_dim", defaults.ff_dim),
            dropout=model_cfg.get("dropout", defaults.dropout),

            # Sampling (square: same scale for both dimensions)
            n_points_freq=model_cfg.get("n_points_freq", defaults.n_points_freq),
            n_points_time=model_cfg.get("n_points_time", defaults.n_points_time),
            freq_offset_scale=model_cfg.get("freq_offset_scale", defaults.freq_offset_scale),
            time_offset_scale=model_cfg.get("time_offset_scale", defaults.time_offset_scale),

            # Priors
            use_time_prior=model_cfg.get("use_time_prior", defaults.use_time_prior),
            use_freq_prior=model_cfg.get("use_freq_prior", defaults.use_freq_prior),
            n_freq_groups=model_cfg.get("n_freq_groups", defaults.n_freq_groups),
            refine_range=model_cfg.get("refine_range", defaults.refine_range),
            rope_base=model_cfg.get("rope_base", defaults.rope_base),

            # Architecture
            bridge_layers=model_cfg.get("bridge_layers", defaults.bridge_layers),
            decoder_layers=model_cfg.get("decoder_layers", defaults.decoder_layers),
            decoder_layer_types=model_cfg.get("decoder_layer_types", defaults.decoder_layer_types),
            decoder_layer_ca_levels=model_cfg.get("decoder_layer_ca_levels", defaults.decoder_layer_ca_levels),
            mamba_d_state=model_cfg.get("mamba_d_state", defaults.mamba_d_state),
            mamba_d_conv=model_cfg.get("mamba_d_conv", defaults.mamba_d_conv),
            mamba_expand=model_cfg.get("mamba_expand", defaults.mamba_expand),
            max_seq_len=model_cfg.get("max_seq_len", defaults.max_seq_len),
            vocab_size=model_cfg.get("vocab_size", defaults.vocab_size),
            use_rope=model_cfg.get("use_rope", defaults.use_rope),
            window_time_frames=model_cfg.get("window_time_frames", defaults.window_time_frames),
            window_freq_bins=model_cfg.get("window_freq_bins", defaults.window_freq_bins),

            # Audio
            sample_rate=audio_cfg.get("sample_rate", defaults.sample_rate),
            n_mels=audio_cfg.get("n_mels", defaults.n_mels),
            n_fft=audio_cfg.get("n_fft", defaults.n_fft),
            hop_length=audio_cfg.get("hop_length", defaults.hop_length),
            f_min=audio_cfg.get("f_min", defaults.f_min),
            f_max=audio_cfg.get("f_max", defaults.f_max),

            # Chunking
            chunk_frames=chunk_cfg.get("chunk_frames", defaults.chunk_frames),
            overlap_frames=chunk_cfg.get("overlap_frames", defaults.overlap_frames),
            min_chunk_ratio=chunk_cfg.get("min_chunk_ratio", defaults.min_chunk_ratio),
            fallback_chunk_frames=chunk_cfg.get("fallback_chunk_frames", defaults.fallback_chunk_frames),
            fallback_overlap_frames=chunk_cfg.get("fallback_overlap_frames", defaults.fallback_overlap_frames),

            # Training
            learning_rate=training_cfg.get("learning_rate", 1e-4),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            warmup_steps=training_cfg.get("warmup_steps", 1000),
            max_epochs=training_cfg.get("max_epochs", 100),
            training_seed=seed_cfg.get("training", 1234),
        )
