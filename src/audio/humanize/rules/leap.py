"""
Leap handling rule: adjust timing/duration around large intervals.

KTH rules:
- Leap-tone-duration: upward leap shortens first note, downward leap lengthens
- Leap-articulation-dro: micropause after large leap
"""

from typing import Dict, Any
from .base import Rule


class LeapRule(Rule):
    """
    Leap handling: adjust timing/duration around large intervals.

    - Upward leap: shorten first note (lighter)
    - Downward leap: lengthen first note (weightier)
    - After large leap: small micropause
    """

    def __init__(
        self,
        config,
        leap_threshold: int = 7,  # semitones (perfect 5th)
        duration_effect: float = 0.1,
        micropause_ms: float = 15.0
    ):
        """
        Initialize leap rule.

        Args:
            config: RuleConfig with k value
            leap_threshold: Minimum interval for "leap" (semitones), default 7
            duration_effect: Duration change ratio (0-1), default 0.1
            micropause_ms: Micropause after leap (milliseconds), default 15
        """
        super().__init__(config)
        self.leap_threshold = leap_threshold
        self.duration_effect = duration_effect
        self.micropause_ms = micropause_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Add micropause after landing from a large leap."""
        if not self.enabled:
            return 0.0

        interval_from_prev = features.get('interval_from_prev', 0)

        if abs(interval_from_prev) >= self.leap_threshold:
            return self.k * self.micropause_ms / 1000

        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Adjust duration before leap."""
        if not self.enabled:
            return 1.0

        interval_to_next = features.get('interval_to_next', 0)

        if abs(interval_to_next) >= self.leap_threshold:
            if interval_to_next > 0:  # Upward leap
                return 1.0 - self.k * self.duration_effect  # Shorten
            else:  # Downward leap
                return 1.0 + self.k * self.duration_effect  # Lengthen

        return 1.0
