"""
Punctuation rule: create silence (breathing) between phrases.

KTH principle: 自動標記樂句，最後音符延長 + micropause
Creates actual gaps, not just tempo changes.
Critical for A2S models to learn phrase boundaries.
"""

from typing import Dict, Any
from .base import Rule


class PunctuationRule(Rule):
    """
    Punctuation: create silence (breathing) between phrases.

    Creates actual gaps, not just tempo changes.
    Critical for A2S models to learn phrase boundaries.
    """

    def __init__(
        self,
        config,
        micropause_ms: float = 30.0,
        last_note_shorten_ratio: float = 0.15
    ):
        """
        Initialize punctuation rule.

        Args:
            config: RuleConfig with k value
            micropause_ms: Micropause duration (milliseconds), default 30
            last_note_shorten_ratio: How much to shorten phrase-end note (0-1)
        """
        super().__init__(config)
        self.micropause_ms = micropause_ms
        self.last_note_shorten_ratio = last_note_shorten_ratio

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Delay the first note of a new phrase."""
        if not self.enabled:
            return 0.0

        is_phrase_start = features.get('is_phrase_start', False)
        phrase_number = features.get('phrase_number', 0)

        # Add micropause before new phrase (but not the first phrase)
        if is_phrase_start and phrase_number > 0:
            return self.k * self.micropause_ms / 1000

        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Shorten the last note of a phrase to create gap."""
        if not self.enabled:
            return 1.0

        is_phrase_end = features.get('is_phrase_end', False)

        if is_phrase_end:
            # Shorten note to create micropause
            return 1.0 - self.k * self.last_note_shorten_ratio

        return 1.0
