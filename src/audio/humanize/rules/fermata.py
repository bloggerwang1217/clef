"""
Fermata rule: extend note duration and add pause after.

Typical fermata effect:
- Note duration × 1.5-2.5 (depends on context)
- Small pause (breath) after fermata note
- Often accompanied by ritardando leading into fermata
"""

from typing import Dict, Any
from .base import Rule


class FermataRule(Rule):
    """Fermata: extend note duration and add pause after."""

    def __init__(
        self,
        config,
        duration_multiplier: float = 2.0,
        pause_beats: float = 0.5
    ):
        """
        Initialize fermata rule.

        Args:
            config: RuleConfig with k value
            duration_multiplier: How much to extend fermata note (default 2.0)
            pause_beats: Pause duration after fermata in beats (default 0.5)
        """
        super().__init__(config)
        self.duration_multiplier = duration_multiplier
        self.pause_beats = pause_beats

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """
        Fermata timing is handled by engine post-processing (onset propagation),
        not by per-note timing offset, because the shift must apply to ALL
        subsequent notes, not just the one immediately after.
        """
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Extend fermata note duration."""
        if not self.enabled:
            return 1.0

        has_fermata = features.get('has_fermata', False)

        if not has_fermata:
            return 1.0

        # Extend duration
        return 1.0 + self.k * (self.duration_multiplier - 1.0)
