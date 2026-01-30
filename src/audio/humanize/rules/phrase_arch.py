"""
Phrase-arch rule: louder and faster in middle of phrase, softer at boundaries.

KTH principle: 樂句呈現弧形 - 開始弱/慢 → 中間強/快 → 結尾弱/慢
Effect (velocity): ±k × 6 dB at phrase peak/boundaries
Effect (tempo): coupled with velocity change
"""

from typing import Dict, Any
import numpy as np
from .base import Rule


class PhraseArchRule(Rule):
    """Phrase-arch: louder in middle of phrase, softer at boundaries."""

    MAX_EFFECT_DB = 1.0  # Maximum dB change at peak (Subtle adjustment for high base velocity)

    def __init__(self, config, peak_position: float = 0.6):
        """
        Initialize phrase-arch rule.

        Args:
            config: RuleConfig with k value
            peak_position: Where phrase peak occurs (0-1), default 0.6
                          Asymmetric - peak slightly past middle
        """
        super().__init__(config)
        self.peak_position = peak_position

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply phrase-arch effect to velocity."""
        if not self.enabled:
            return 0.0

        phrase_pos = features.get('phrase_position')
        if phrase_pos is None:
            return 0.0

        # Parabolic arch centered at peak_position
        # Shape: Convex (Dome-like), louder in middle
        # Formula: 1 - x^2 (where x is normalized distance from peak)
        
        if phrase_pos <= self.peak_position:
            # Rising phase
            # dist goes from 1.0 (start) to 0.0 (peak)
            dist = (self.peak_position - phrase_pos) / self.peak_position
            arch = 1.0 - dist ** 2
        else:
            # Falling phase
            # dist goes from 0.0 (peak) to 1.0 (end)
            dist = (phrase_pos - self.peak_position) / (1.0 - self.peak_position)
            arch = 1.0 - dist ** 2

        # Convert arch [0-1] to dB delta
        return self.k * self.MAX_EFFECT_DB * arch

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """
        Apply phrase-arch tempo effect.

        Phrase timing: slower at start/end, faster in middle.
        Coupled with velocity for natural musicality.
        """
        if not self.enabled:
            return 0.0

        phrase_pos = features.get('phrase_position')
        if phrase_pos is None:
            return 0.0

        beat_duration = features.get('beat_duration', 0.5)

        # Tempo curve: start slow, peak fast, end slower
        # Range: -0.1 (slower) to +0.1 (faster)
        if phrase_pos < self.peak_position:
            # Accelerating: -0.1 → +0.1
            tempo_ratio = -0.1 + 0.2 * (phrase_pos / self.peak_position)
        else:
            # Decelerating: +0.1 → -0.15
            decel_pos = (phrase_pos - self.peak_position) / (1 - self.peak_position)
            tempo_ratio = 0.1 - 0.25 * decel_pos

        # Negative tempo_ratio (slower) = positive timing offset (later)
        return -self.k * tempo_ratio * beat_duration * 0.5

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect (handled by timing)."""
        return 1.0
