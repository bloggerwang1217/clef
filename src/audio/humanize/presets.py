"""
Performance style presets.

Provides pre-configured HumanizationConfig for different musical styles:
- ROMANTIC: More expressive, larger dynamic and timing variations
- CLASSICAL: More restrained, precise
- BALANCED: Default, middle ground
"""

from .config import HumanizationConfig, RuleConfig


# Romantic style: more expressive
ROMANTIC = HumanizationConfig(
    # Velocity rules - more extreme
    high_loud=RuleConfig(k=1.2, k_range=(0.8, 1.5)),
    phrase_arch=RuleConfig(k=1.3, k_range=(0.8, 1.8)),
    duration_contrast=RuleConfig(k=1.2, k_range=(0.8, 1.6)),
    melodic_charge=RuleConfig(k=1.0, k_range=(0.5, 1.5)),

    # Timing rules - more rubato
    phrase_rubato=RuleConfig(k=1.3, k_range=(0.8, 1.8)),
    final_ritard=RuleConfig(k=1.5, k_range=(1.0, 2.0)),

    # Motor noise
    beat_jitter=RuleConfig(k=1.2, k_range=(0.8, 1.5)),

    # More pronounced effects
    crescendo_tempo=RuleConfig(k=1.2, k_range=(0.5, 1.8)),
    punctuation=RuleConfig(k=1.2, k_range=(0.8, 1.6)),

    # Articulation
    staccato=RuleConfig(k=1.0, k_range=(0.7, 1.3)),
    legato=RuleConfig(k=1.2, k_range=(0.8, 1.6)),
)


# Classical style: more restrained
CLASSICAL = HumanizationConfig(
    # Velocity rules - subtle
    high_loud=RuleConfig(k=0.8, k_range=(0.5, 1.0)),
    phrase_arch=RuleConfig(k=0.8, k_range=(0.5, 1.2)),
    duration_contrast=RuleConfig(k=0.7, k_range=(0.4, 1.0)),
    melodic_charge=RuleConfig(k=0.6, k_range=(0.3, 1.0)),

    # Timing rules - precise
    phrase_rubato=RuleConfig(k=0.7, k_range=(0.4, 1.0)),
    final_ritard=RuleConfig(k=0.8, k_range=(0.5, 1.2)),

    # Motor noise - tighter
    beat_jitter=RuleConfig(k=0.7, k_range=(0.5, 1.0)),

    # Subtle effects
    crescendo_tempo=RuleConfig(k=0.7, k_range=(0.3, 1.2)),
    punctuation=RuleConfig(k=0.8, k_range=(0.5, 1.2)),

    # Articulation
    staccato=RuleConfig(k=1.0, k_range=(0.8, 1.2)),
    legato=RuleConfig(k=0.8, k_range=(0.5, 1.2)),
)


# Balanced (default): middle ground
BALANCED = HumanizationConfig()  # All k=1.0 with default ranges


# Map from style name to config
PRESETS = {
    'romantic': ROMANTIC,
    'classical': CLASSICAL,
    'balanced': BALANCED,
}


def get_preset(name: str) -> HumanizationConfig:
    """
    Get a preset by name.

    Args:
        name: Preset name ('romantic', 'classical', 'balanced')

    Returns:
        HumanizationConfig

    Raises:
        ValueError: If preset name not found
    """
    name_lower = name.lower()

    if name_lower not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {', '.join(PRESETS.keys())}"
        )

    return PRESETS[name_lower]
