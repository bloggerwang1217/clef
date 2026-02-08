"""
Articulation-tempo coupling rule.

Articulation affects local tempo feel:
- Legato passages: slightly slower, more connected
- Tenuto: slight lengthening and delay of next note
"""

from typing import Dict, Any
from .base import Rule


class ArticulationTempoRule(Rule):
    """
    Articulation affects local tempo feel.

    - Legato passages: slightly slower, more connected
    - Staccato passages: can feel slightly faster
    - Tenuto: slight lengthening and delay of next note
    """

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply timing adjustment based on articulation."""
        if not self.enabled:
            return 0.0

        timing_offset = 0.0

        # Tenuto: hold slightly longer, delay next
        if features.get('tenuto', False):
            timing_offset += self.k * 0.015  # +15ms

        # Legato context: slightly broader timing
        if features.get('in_slur', False):
            timing_offset += self.k * 0.005  # +5ms tendency

        return timing_offset

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect (handled by articulation rules)."""
        return 1.0
