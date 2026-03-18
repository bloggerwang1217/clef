"""
Clef Piano Configuration
========================

Configuration for clef-piano-base (ISMIR 2026) model.
Extends base ClefConfig with piano-specific settings.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..config import ClefConfig


@dataclass
class ClefPianoConfig(ClefConfig):
    """Configuration for clef-piano-base model.

    ISMIR 2026 target: Single-instrument (piano) transcription
    using Swin V2 encoder + WindowCrossAttention decoder.
    """

    # Model name (for model selection in train.py)
    name: str = "clef-piano-base"

    # Piano-specific defaults
    vocab_size: int = 5589  # compound note tokens (N_DUR×N_PITCH + struct/schema/special)

    # Guided attention loss weight schedule.
    # Weight held constant at guidance_loss_weight during warmup, then cosine-decays
    # to guidance_loss_weight_end over guidance_decay_steps (relative to warmup end).
    # Example: guidance_decay_steps=2500 means decay from warmup_steps to warmup_steps+2500.
    # Supervises L1 Full CA to attend to the correct audio measure.
    guidance_loss_weight: float = 0.0        # starting weight (high → forces CA alignment)
    guidance_loss_weight_end: float = 0.0    # final weight after decay (lower → CE dominates)
    guidance_decay_steps: int = 0            # steps AFTER warmup for decay (0 = use total_steps - warmup_steps)

    # Bar token ID for L1 bar full-attention (Zeng extended vocabulary).
    # <bar> tokens attend to onset_1d to compute bar_center (temporal authority).
    bar_token_id: int = 4
    curriculum_warmup_steps: int = 0

    # Global NoteGRU redesign (global-notegru-plan.md)
    # bar_gru: cross-bar time tracking; note_gru: token-level shared time prior
    bar_gru_hidden_size: int = 256
    bar_gru_input_dropout: float = 0.1   # dropout on note_gru output before bar_gru input
    note_gru_hidden_size: int = 256
    note_gru_input_dropout: float = 0.1  # dropout on note_gru input projection
    tf_anneal_steps: int = 5000          # step at which tf_ratio reaches 0.0 (linear from warmup_steps)

    # Per-layer full_freq config for window_ca layers.
    # Each entry: True (all levels), False (none), or List[int] (specific level IDs).
    decoder_layer_full_freq: Optional[List] = None
    # Per-layer cascade_com flag: after each level, use its CoM as next level's window center.
    decoder_layer_cascade_com: Optional[List] = None


    # Exponential decay soft mask for window CA (sa_window_ca and mamba_window_ca only).
    # score_j += -lambda * |t_j - com_t|  before softmax → L1-style gradient pull on com_t.
    # 0.0 = disabled; recommended range: 10.0 (soft) to 50.0 (strong).
    window_exp_decay_lambda: float = 0.0

    tbptt_chunk_size: int = 256  # detach Mamba state every N steps for TBPTT (0 = full BPTT)
    bucket_boundaries: Optional[List[int]] = None  # None = use BucketSampler default

    # Gradient checkpointing (trades compute for memory)
    gradient_checkpointing: bool = False
    window_ca_use_checkpoint: bool = True  # checkpoint per seq-chunk inside WindowCrossAttention

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

    # BiMamba Encoder (per-pitch-track temporal modeling, Zeng 2024 inspired)
    use_bimamba_encoder: bool = False
    bimamba_input_dim: int = 192        # S0 output channels (Swin V2 Tiny)
    bimamba_d_model: int = 512          # Match decoder d_model
    bimamba_d_state: int = 128
    bimamba_d_conv: int = 4
    bimamba_num_layers: int = 2
    bimamba_dropout: float = 0.1

    # Training hyperparameters (read from YAML; single source of truth)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_min_ratio: float = 0.1   # min_lr = learning_rate * lr_min_ratio (cosine decay endpoint)
    lr_flat_ratio: float = 0.0  # WSD: fraction of post-warmup steps held at peak LR (0.0 = original cosine)
    max_epochs: int = 100
    training_seed: int = 1234

    # Runtime / infrastructure (read from YAML; CLI args are only --config/--resume/--wandb/--debug/--sanity-check)
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_clip: float = 1.0
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 0   # 0 = disabled
    validate_every_n_steps: int = 500  # 0 = epoch-only validation

    # Paths (defaults match legacy CLI defaults; override in YAML)
    manifest_dir: str = "data/experiments/clef_piano_base"
    checkpoint_dir: str = "checkpoints/clef_piano_base"

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
            # Model name
            name=model_cfg.get("name", defaults.name),

            # Swin
            swin_model=model_cfg.get("swin_model", defaults.swin_model),
            swin_dims=model_cfg.get("swin_dims", defaults.swin_dims),
            freeze_encoder=model_cfg.get("freeze_encoder", defaults.freeze_encoder),
            swin_use_gradient_checkpointing=model_cfg.get("swin_use_gradient_checkpointing", defaults.swin_use_gradient_checkpointing),
            swin_unfreeze=model_cfg.get("swin_unfreeze", defaults.swin_unfreeze),
            swin_lr_scale=model_cfg.get("swin_lr_scale", defaults.swin_lr_scale),

            # HarmonizingFlow
            use_flow=model_cfg.get("use_flow", defaults.use_flow),
            n_harmonics=model_cfg.get("n_harmonics", defaults.n_harmonics),
            flow_init=model_cfg.get("flow_init", defaults.flow_init),
            flow_pool_stride=model_cfg.get("flow_pool_stride", defaults.flow_pool_stride),

            # Octopus2D
            use_octopus=model_cfg.get("use_octopus", defaults.use_octopus),
            octopus_freq_kernel=model_cfg.get("octopus_freq_kernel", defaults.octopus_freq_kernel),
            octopus_time_kernel=model_cfg.get("octopus_time_kernel", defaults.octopus_time_kernel),
            octopus_channels=model_cfg.get("octopus_channels", defaults.octopus_channels),
            octopus_time_pool_stride=model_cfg.get("octopus_time_pool_stride", defaults.octopus_time_pool_stride),
            octopus_freq_pool_stride=model_cfg.get("octopus_freq_pool_stride", 4),

            # Swin input mode
            swin_start_stage=model_cfg.get("swin_start_stage", defaults.swin_start_stage),
            swin_pool_strides=model_cfg.get("swin_pool_strides", defaults.swin_pool_strides),

            # Attention
            d_model=model_cfg.get("d_model", defaults.d_model),
            n_heads=model_cfg.get("n_heads", defaults.n_heads),
            n_levels=model_cfg.get("n_levels", defaults.n_levels),
            ff_dim=model_cfg.get("ff_dim", defaults.ff_dim),
            dropout=model_cfg.get("dropout", defaults.dropout),

            # Priors
            use_freq_prior=model_cfg.get("use_freq_prior", defaults.use_freq_prior),
            rope_base=model_cfg.get("rope_base", defaults.rope_base),

            # BiMamba Encoder
            use_bimamba_encoder=model_cfg.get("use_bimamba_encoder", defaults.use_bimamba_encoder),
            bimamba_input_dim=model_cfg.get("bimamba_input_dim", defaults.bimamba_input_dim),
            bimamba_d_model=model_cfg.get("bimamba_d_model", defaults.bimamba_d_model),
            bimamba_d_state=model_cfg.get("bimamba_d_state", defaults.bimamba_d_state),
            bimamba_d_conv=model_cfg.get("bimamba_d_conv", defaults.bimamba_d_conv),
            bimamba_num_layers=model_cfg.get("bimamba_num_layers", defaults.bimamba_num_layers),
            bimamba_dropout=model_cfg.get("bimamba_dropout", defaults.bimamba_dropout),

            # Bar attention
            bar_token_id=model_cfg.get("bar_token_id", defaults.bar_token_id),
            curriculum_warmup_steps=model_cfg.get("curriculum_warmup_steps", defaults.curriculum_warmup_steps),

            # Global NoteGRU redesign
            bar_gru_hidden_size=model_cfg.get("bar_gru_hidden_size", defaults.bar_gru_hidden_size),
            bar_gru_input_dropout=model_cfg.get("bar_gru_input_dropout", defaults.bar_gru_input_dropout),
            note_gru_hidden_size=model_cfg.get("note_gru_hidden_size", defaults.note_gru_hidden_size),
            note_gru_input_dropout=model_cfg.get("note_gru_input_dropout", defaults.note_gru_input_dropout),
            tf_anneal_steps=model_cfg.get("tf_anneal_steps", training_cfg.get("tf_anneal_steps", defaults.tf_anneal_steps)),


            # Guided attention loss
            guidance_loss_weight=model_cfg.get("guidance_loss_weight", defaults.guidance_loss_weight),
            guidance_loss_weight_end=model_cfg.get("guidance_loss_weight_end",
                                                   model_cfg.get("guidance_loss_weight",  # fallback = same as start
                                                                  defaults.guidance_loss_weight_end)),
            guidance_decay_steps=model_cfg.get("guidance_decay_steps", defaults.guidance_decay_steps),

            # Architecture
            bridge_layers=model_cfg.get("bridge_layers", defaults.bridge_layers),
            decoder_layers=model_cfg.get("decoder_layers", defaults.decoder_layers),
            decoder_layer_types=model_cfg.get("decoder_layer_types", defaults.decoder_layer_types),
            decoder_layer_ca_levels=model_cfg.get("decoder_layer_ca_levels", defaults.decoder_layer_ca_levels),
            decoder_layer_full_freq=model_cfg.get("decoder_layer_full_freq", defaults.decoder_layer_full_freq),
            decoder_layer_cascade_com=model_cfg.get("decoder_layer_cascade_com", defaults.decoder_layer_cascade_com),
            gradient_checkpointing=model_cfg.get("gradient_checkpointing", defaults.gradient_checkpointing),
            window_ca_use_checkpoint=model_cfg.get("window_ca_use_checkpoint", defaults.window_ca_use_checkpoint),
            mamba_d_state=model_cfg.get("mamba_d_state", defaults.mamba_d_state),
            mamba_d_conv=model_cfg.get("mamba_d_conv", defaults.mamba_d_conv),
            mamba_expand=model_cfg.get("mamba_expand", defaults.mamba_expand),
            max_seq_len=model_cfg.get("max_seq_len", defaults.max_seq_len),
            vocab_size=model_cfg.get("vocab_size", defaults.vocab_size),
            window_time_frames=model_cfg.get("window_time_frames", defaults.window_time_frames),
            window_freq_bins=model_cfg.get("window_freq_bins", defaults.window_freq_bins),
            window_exp_decay_lambda=model_cfg.get("window_exp_decay_lambda", defaults.window_exp_decay_lambda),
            tbptt_chunk_size=model_cfg.get("tbptt_chunk_size", defaults.tbptt_chunk_size),
            bucket_boundaries=training_cfg.get("bucket_boundaries", defaults.bucket_boundaries),

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
            lr_min_ratio=training_cfg.get("lr_min_ratio", 0.1),
            lr_flat_ratio=training_cfg.get("lr_flat_ratio", 0.0),
            max_epochs=training_cfg.get("max_epochs", 100),
            training_seed=seed_cfg.get("training", 1234),

            # Runtime / infrastructure
            batch_size=training_cfg.get("batch_size", defaults.batch_size),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", defaults.gradient_accumulation_steps),
            gradient_clip=training_cfg.get("gradient_clip", defaults.gradient_clip),
            save_every_n_epochs=training_cfg.get("save_every_n_epochs", defaults.save_every_n_epochs),
            early_stopping_patience=training_cfg.get("early_stopping_patience", defaults.early_stopping_patience),
            validate_every_n_steps=training_cfg.get("validate_every_n_steps", defaults.validate_every_n_steps),

            # Paths
            manifest_dir=cfg.get("paths", {}).get("manifest_dir", defaults.manifest_dir),
            checkpoint_dir=cfg.get("paths", {}).get("checkpoint_dir", defaults.checkpoint_dir),
        )
