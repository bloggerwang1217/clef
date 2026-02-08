"""
Base class for KTH-style humanization rules.

All rules inherit from this abstract class and implement
velocity, timing, and duration effects.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ..config import RuleConfig


class Rule(ABC):
    """Base class for KTH-style humanization rules."""

    def __init__(self, config: RuleConfig):
        """
        Initialize rule with configuration.

        Args:
            config: RuleConfig with k value and settings
        """
        self.config = config
        # Whether this rule affects tempo (requires cumulative timing)
        # Tempo-affecting rules: PhraseRubato, BeatRubato, FinalRitard
        # Position-only rules: MicroTiming, ChordAsync, etc.
        self.is_tempo_affecting = False

    @property
    def k(self) -> float:
        """Rule strength parameter (k value)."""
        return self.config.k

    @property
    def enabled(self) -> bool:
        """Whether this rule is enabled."""
        return self.config.enabled

    @abstractmethod
    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """
        Return velocity delta in dB.

        Args:
            note: Note object from partitura
            features: Per-note features dict

        Returns:
            Velocity change in dB (additive)
        """
        pass

    @abstractmethod
    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """
        Return timing delta in seconds.

        Args:
            note: Note object from partitura
            features: Per-note features dict

        Returns:
            Timing offset in seconds (can be negative for earlier)
        """
        pass

    @abstractmethod
    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """
        Return duration multiplier.

        Args:
            note: Note object from partitura
            features: Per-note features dict

        Returns:
            Duration multiplier (1.0 = no change, <1.0 = shorter, >1.0 = longer)
        """
        pass
