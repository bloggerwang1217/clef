"""
Repetition handling rule: add micropause between repeated notes.

KTH principle: Repetition-articulation-dro - micropause for repeated notes
Prevents "machine gun" effect on repeated notes.
"""

from typing import Dict, Any
from .base import Rule


class RepetitionRule(Rule):
    """
    Repetition handling: add micropause between repeated notes.

    Prevents "machine gun" effect on repeated notes.
    """

    def __init__(self, config, micropause_ms: float = 20.0):
        """
        Initialize repetition rule.

        Args:
            config: RuleConfig with k value
            micropause_ms: Micropause duration (milliseconds), default 20
        """
        super().__init__(config)
        self.micropause_ms = micropause_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect - gap is created by shortening duration only."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Shorten repeated notes slightly."""
        if not self.enabled:
            return 1.0

        is_repeated = features.get('is_repeated_note', False)

        if is_repeated:
            return 1.0 - self.k * 0.1  # -10% duration

        return 1.0
