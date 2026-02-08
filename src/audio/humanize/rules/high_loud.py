"""
High-loud rule: higher pitches are played louder.

KTH principle: pitch 越高 → 越大聲
Effect: +k × 0.5 dB per semitone above middle C (MIDI 60)
"""

from typing import Dict, Any
from .base import Rule


class HighLoudRule(Rule):
    """High-loud: higher pitches are played louder."""

    SEMITONE_COEFFICIENT = 0.03  # dB per semitone
    MAX_EFFECT_DB = 1.0  # Cap to prevent extreme boost at very high/low pitches

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply high-loud effect to velocity."""
        if not self.enabled:
            return 0.0

        pitch = getattr(note, 'midi_pitch', None) or features.get('pitch', 60)
        semitones_above_c4 = pitch - 60
        raw = self.k * self.SEMITONE_COEFFICIENT * semitones_above_c4
        return max(-self.MAX_EFFECT_DB, min(self.MAX_EFFECT_DB, raw))

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0
