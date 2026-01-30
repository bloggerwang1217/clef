"""
Dynamics-tempo coupling rules.

Based on KTH research: dynamics and tempo are highly correlated in performance.
- Crescendo → accelerando
- Diminuendo → ritardando
- Accent → agogic accent (slight delay)
"""

from typing import Dict, Any
from .base import Rule


class CrescendoTempoRule(Rule):
    """
    Crescendo/diminuendo affects both velocity and tempo.

    Based on KTH research: dynamics and tempo are coupled.
    - Crescendo → louder + slight accelerando
    - Diminuendo → softer + slight ritardando
    """

    def __init__(self, config, max_tempo_change: float = 0.1, max_velocity_change_dB: float = 3.0):
        """
        Initialize crescendo-tempo rule.

        Args:
            config: RuleConfig with k value
            max_tempo_change: Maximum tempo change ratio (default 0.1 = ±10%)
            max_velocity_change_dB: Maximum velocity change in dB (default 3.0 = subtle crescendo)
        """
        super().__init__(config)
        self.max_tempo_change = max_tempo_change
        self.max_velocity_change_dB = max_velocity_change_dB

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply velocity change based on crescendo/diminuendo from Partitura features."""
        if not self.enabled:
            return 0.0

        # Get loudness direction from Partitura features (0-1 values)
        # Note: In Partitura 1.7.0, descriptor names use _feature prefix
        loudness_incr = features.get('loudness_direction_feature.loudness_incr', 0.0)
        loudness_decr = features.get('loudness_direction_feature.loudness_decr', 0.0)

        # Crescendo: +max_velocity_change_dB
        # Diminuendo: -max_velocity_change_dB
        velocity_change = (loudness_incr - loudness_decr) * self.k * self.max_velocity_change_dB

        return velocity_change

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply tempo change based on crescendo/diminuendo from Partitura features."""
        if not self.enabled:
            return 0.0

        # Get loudness direction from Partitura features (0-1 values)
        loudness_incr = features.get('loudness_direction_feature.loudness_incr', 0.0)
        loudness_decr = features.get('loudness_direction_feature.loudness_decr', 0.0)
        loudness_change = loudness_incr - loudness_decr

        # Positive = crescendo = faster = negative timing offset
        tempo_ratio = self.k * self.max_tempo_change * loudness_change
        beat_duration = features.get('beat_duration', 0.5)

        # Negative = earlier (faster)
        return -tempo_ratio * beat_duration

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0


class AgogicAccentRule(Rule):
    """
    Agogic accent: accented notes are slightly delayed.

    Creates emphasis through timing, not just velocity.
    """

    def __init__(self, config, delay_ms: float = 20.0):
        """
        Initialize agogic accent rule.

        Args:
            config: RuleConfig with k value
            delay_ms: Delay for accented notes (milliseconds), default 20
        """
        super().__init__(config)
        self.delay_ms = delay_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect (handled by accent marking)."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply agogic delay to accented notes."""
        if not self.enabled:
            return 0.0

        # Check for accent marking from articulation_basis
        has_accent = features.get('accent', 0.0) > 0.5
        has_sf = features.get('sf', 0.0) > 0.5
        has_sfz = features.get('sfz', 0.0) > 0.5

        if has_accent or has_sf or has_sfz:
            return self.k * self.delay_ms / 1000

        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0
