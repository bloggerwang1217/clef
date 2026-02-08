"""
Duration-contrast rule: longer notes are played louder and stretched.

KTH principle: 長音更長更大聲，短音更短更小聲
Effect (velocity): ±k × 3 dB based on relative duration
Effect (duration): longer notes stretched further
"""

from typing import Dict, Any
import numpy as np
from .base import Rule


class DurationContrastRule(Rule):
    """Duration-contrast: longer notes louder and stretched."""

    MAX_VELOCITY_EFFECT_DB = 0.5  # Maximum dB boost for long notes (Lowered from 3.0)
    DURATION_STRETCH_FACTOR = 0.05  # 5% stretch for 2x duration (Lowered from 0.1)

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply duration-contrast effect to velocity."""
        if not self.enabled:
            return 0.0

        # Get relative duration (vs local average)
        rel_dur = features.get('relative_duration', 1.0)

        # Log relationship: 2x duration = +3dB, 0.5x duration = -3dB
        dB_delta = self.k * self.MAX_VELOCITY_EFFECT_DB * np.log2(rel_dur)

        return dB_delta

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply duration-contrast effect to duration."""
        if not self.enabled:
            return 1.0

        # Get relative duration
        rel_dur = features.get('relative_duration', 1.0)

        # Stretch longer notes, compress shorter notes
        # Example: 2x duration → 1.1x stretch, 0.5x duration → 0.95x compress
        duration_multiplier = 1.0 + self.k * self.DURATION_STRETCH_FACTOR * (rel_dur - 1.0)

        return duration_multiplier
