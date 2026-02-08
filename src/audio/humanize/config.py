"""
Configuration system for KTH-style humanization rules.

Implements the k-value system where each rule has a strength parameter
that can be randomized for data augmentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class RuleConfig:
    """Configuration for a single KTH-style rule."""

    k: float = 1.0  # Current k value (strength)
    k_range: Tuple[float, float] = (0.5, 1.5)  # Randomization range
    enabled: bool = True  # Toggle rule on/off

    def randomize(self, rng: np.random.Generator) -> 'RuleConfig':
        """Return new config with randomized k within range."""
        new_k = rng.uniform(self.k_range[0], self.k_range[1])
        return RuleConfig(k=new_k, k_range=self.k_range, enabled=self.enabled)


@dataclass
class HumanizationConfig:
    """
    KTH-style humanization configuration with k-value system.

    All effects are ADDITIVE (疊加), referenced to:
    - 0 dB = MIDI velocity 64 (KTH standard)
    - BasisMixer default velocity = 55
    """

    # === Global Settings ===
    reference_velocity: int = 64  # 0 dB reference point
    default_velocity: int = 90  # Default baseline (f), same as clef-piano-base
    default_bpm: float = 108.0  # BasisMixer default

    # === Dynamics Rules ===
    # Dynamic marking → velocity mapping (not affected by k)
    dynamics_map: Dict[str, int] = field(default_factory=lambda: {
        'ppp': 20, 'pp': 35, 'p': 50,
        'mp': 60, 'mf': 70,
        'f': 85, 'ff': 100, 'fff': 115,
        'sf': 95, 'sfz': 100, 'fp': 85,
    })

    # === Velocity Rules ===

    # High-loud: higher pitch → louder
    # Effect: +k × 0.5 dB per semitone above middle C
    high_loud: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))

    # Phrase-arch: louder in phrase middle, softer at boundaries
    # Effect: ±k × 6 dB at phrase peak/boundaries
    phrase_arch: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))
    phrase_peak_position: float = 0.6  # 0-1, where peak occurs

    # Duration-contrast: longer notes are louder and more extended
    # Effect: ±k × 3 dB based on relative duration
    duration_contrast: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))

    # Melodic-charge: emphasize melodic tension (non-chord tones)
    # Effect: +k × 2 dB for non-chord tones
    melodic_charge: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=0.8, k_range=(0.0, 2.0)
    ))
    nct_boost_dB: float = 2.0

    # === Timing Rules ===

    # Phrase rubato: tempo variation within phrases
    phrase_rubato: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))

    # Beat jitter: micro timing fluctuations (motor noise)
    beat_jitter: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))

    # Final-ritard: gradual slowdown at the end
    # Effect: tempo × (1 - k × sqrt(position)) in final section
    final_ritard: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))
    final_ritard_measures: float = 2.0  # Duration of ritardando in measures (bars)

    # Fermata handling
    fermata: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5)
    ))
    fermata_duration_multiplier: float = 2.0
    fermata_pause_beats: float = 0.5

    # Dynamics-tempo coupling
    crescendo_tempo: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))
    crescendo_tempo_max_change: float = 0.015  # ±1.5% tempo (subtle coupling)
    crescendo_velocity_max_change_dB: float = 3.0  # Max velocity change (subtle crescendo)



    # Punctuation (breathing between phrases)
    punctuation: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5)
    ))
    micropause_ms: float = 20.0
    phrase_end_shorten_ratio: float = 0.15

    # Repetition handling
    repetition: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))
    repetition_micropause_ms: float = 20.0

    # === Articulation Rules ===

    # Staccato shortening
    # Effect: duration × (1 - k × 0.5) for staccato notes
    staccato: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))

    # Legato overlap
    # Effect: +k × 30ms overlap for legato phrases
    legato: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))
    legato_overlap_base_ms: float = 30.0

    # Tenuto: hold full value, slightly lengthen
    # Effect: duration × (1 + k × 0.05), velocity +k × 1dB
    tenuto: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))
    tenuto_extension_ratio: float = 0.05  # 5% extension

    # Accent (>): emphasis through velocity and optional timing
    # Effect: velocity +k × 3dB
    accent: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 1.5)
    ))
    accent_velocity_boost_dB: float = 1.5
    accent_delay_ms: float = 0.0  # Optional agogic delay (0 = disabled)

    # Marcato (^): strong accent
    # Effect: velocity +k × 5dB, duration × (1 - k × 0.05)
    marcato: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)
    ))
    marcato_velocity_boost_dB: float = 5.0
    marcato_shortening_ratio: float = 0.05  # 5% shortening

    # === Ornament Rules ===

    grace_note: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5)
    ))
    acciaccatura_ms: float = 50.0
    appoggiatura_ratio: float = 0.25

    trill: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(1.0, 2.0)
    ))
    trill_speed: float = 8.0  # notes per second
    trill_start_on_upper: bool = False

    mordent: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 2.0)
    ))

    tremolo: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 2.0)  # k=0: no variation, k=2: strong variation (±1.0dB)
    ))
    tremolo_velocity_variation: float = 0.5  # dB variation between tremolo notes (k=1.0 → ±0.5dB, k=2.0 → ±1.0dB)
    tremolo_timing_jitter_ms: float = 2.0    # Timing jitter per tremolo note (subtle, not hand-shaking)

    # === Safety Rules ===

    social_duration_care: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.8, 1.2)
    ))
    min_audible_duration_ms: float = 50.0

    # Global normalizer settings
    normalize_velocity: bool = True
    target_rms_velocity: int = 70
    max_velocity: int = 115
    soft_clip_threshold: int = 100

    # === Pedal Settings ===

    pedal: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5), enabled=True
    ))
    pedal_lift_before_ms: float = 30.0
    pedal_press_after_ms: float = 20.0

    # === Tempo Settings ===

    randomize_tempo: bool = True  # Sample within marking range
    tempo_variation_range: Tuple[float, float] = (0.9, 1.1)  # ±10% variation

    # === Methods ===

    def randomize(self, seed: Optional[int] = None) -> 'HumanizationConfig':
        """
        Create a new config with all k values randomized within their ranges.
        This is the core of data augmentation diversity.
        """
        rng = np.random.default_rng(seed)

        return HumanizationConfig(
            # Keep global settings
            reference_velocity=self.reference_velocity,
            default_velocity=self.default_velocity,
            default_bpm=self.default_bpm,
            dynamics_map=self.dynamics_map.copy(),

            # Randomize all rule k values
            high_loud=self.high_loud.randomize(rng),
            phrase_arch=self.phrase_arch.randomize(rng),
            phrase_peak_position=rng.uniform(0.5, 0.7),
            duration_contrast=self.duration_contrast.randomize(rng),
            melodic_charge=self.melodic_charge.randomize(rng),
            nct_boost_dB=self.nct_boost_dB,

            # Timing rules
            phrase_rubato=self.phrase_rubato.randomize(rng),
            beat_jitter=self.beat_jitter.randomize(rng),
            final_ritard=self.final_ritard.randomize(rng),
            final_ritard_measures=self.final_ritard_measures,
            fermata=self.fermata.randomize(rng),
            fermata_duration_multiplier=self.fermata_duration_multiplier,
            fermata_pause_beats=self.fermata_pause_beats,
            crescendo_tempo=self.crescendo_tempo.randomize(rng),
            crescendo_tempo_max_change=self.crescendo_tempo_max_change,
            crescendo_velocity_max_change_dB=self.crescendo_velocity_max_change_dB,
            punctuation=self.punctuation.randomize(rng),
            micropause_ms=self.micropause_ms,
            phrase_end_shorten_ratio=self.phrase_end_shorten_ratio,
            repetition=self.repetition.randomize(rng),
            repetition_micropause_ms=self.repetition_micropause_ms,

            # Articulation
            staccato=self.staccato.randomize(rng),
            legato=self.legato.randomize(rng),
            legato_overlap_base_ms=self.legato_overlap_base_ms,
            tenuto=self.tenuto.randomize(rng),
            tenuto_extension_ratio=self.tenuto_extension_ratio,
            accent=self.accent.randomize(rng),
            accent_velocity_boost_dB=self.accent_velocity_boost_dB,
            accent_delay_ms=self.accent_delay_ms,
            marcato=self.marcato.randomize(rng),
            marcato_velocity_boost_dB=self.marcato_velocity_boost_dB,
            marcato_shortening_ratio=self.marcato_shortening_ratio,

            # Ornaments
            grace_note=self.grace_note.randomize(rng),
            acciaccatura_ms=self.acciaccatura_ms,
            appoggiatura_ratio=self.appoggiatura_ratio,
            trill=self.trill.randomize(rng),
            trill_speed=self.trill_speed,
            trill_start_on_upper=self.trill_start_on_upper,
            mordent=self.mordent.randomize(rng),
            tremolo=self.tremolo.randomize(rng),
            tremolo_velocity_variation=self.tremolo_velocity_variation,
            tremolo_timing_jitter_ms=self.tremolo_timing_jitter_ms,

            # Safety
            social_duration_care=self.social_duration_care.randomize(rng),
            min_audible_duration_ms=self.min_audible_duration_ms,
            normalize_velocity=self.normalize_velocity,
            target_rms_velocity=self.target_rms_velocity,
            max_velocity=self.max_velocity,
            soft_clip_threshold=self.soft_clip_threshold,

            # Pedal
            pedal=self.pedal.randomize(rng),
            pedal_lift_before_ms=self.pedal_lift_before_ms,
            pedal_press_after_ms=self.pedal_press_after_ms,

            # Tempo
            randomize_tempo=self.randomize_tempo,
            tempo_variation_range=self.tempo_variation_range,
        )

    def to_dict(self) -> Dict:
        """Export config for logging/reproducibility."""
        return {
            'high_loud_k': self.high_loud.k,
            'phrase_arch_k': self.phrase_arch.k,
            'phrase_peak_position': self.phrase_peak_position,
            'duration_contrast_k': self.duration_contrast.k,
            'melodic_charge_k': self.melodic_charge.k,
            'phrase_rubato_k': self.phrase_rubato.k,
            'beat_jitter_k': self.beat_jitter.k,
            'final_ritard_k': self.final_ritard.k,
            'final_ritard_measures': self.final_ritard_measures,
            'fermata_k': self.fermata.k,
            'crescendo_tempo_k': self.crescendo_tempo.k,
            'punctuation_k': self.punctuation.k,
            'repetition_k': self.repetition.k,
            'staccato_k': self.staccato.k,
            'legato_k': self.legato.k,
            'tenuto_k': self.tenuto.k,
            'accent_k': self.accent.k,
            'marcato_k': self.marcato.k,
            'grace_note_k': self.grace_note.k,
            'trill_k': self.trill.k,
            'mordent_k': self.mordent.k,
            'tremolo_k': self.tremolo.k,
            'social_duration_care_k': self.social_duration_care.k,
            'pedal_k': self.pedal.k,
        }
