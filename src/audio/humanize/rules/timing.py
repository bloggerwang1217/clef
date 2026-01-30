"""
Micro-timing rules: small timing variations and chord asynchrony.

Micro-timing: small random timing jitter (±10-15ms)
Chord asynchrony: melody leads bass (20-50ms)
"""

from typing import Dict, Any, Optional
import numpy as np
from .base import Rule


class MicroTimingRule(Rule):
    """Micro-timing: small random timing variations."""

    def __init__(self, config, std_ms: float = 15.0):
        """
        Initialize micro-timing rule.

        Args:
            config: RuleConfig with k value
            std_ms: Standard deviation of jitter in milliseconds (default 15)
        """
        super().__init__(config)
        self.std_ms = std_ms
        self.rng: Optional[np.random.Generator] = None  # Set by engine

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self.rng = rng

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply micro-timing jitter."""
        if not self.enabled or self.rng is None:
            return 0.0

        # Gaussian jitter
        jitter = self.rng.normal(0, self.std_ms / 1000)

        return self.k * jitter

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0


class ChordAsyncRule(Rule):
    """Chord asynchrony: melody leads bass."""

    def __init__(self, config, lead_ms: float = 25.0):
        """
        Initialize chord async rule.

        Args:
            config: RuleConfig with k value
            lead_ms: How much melody leads (milliseconds), default 25
        """
        super().__init__(config)
        self.lead_ms = lead_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply chord asynchrony - melody leads."""
        if not self.enabled:
            return 0.0

        is_melody = features.get('is_melody', False)

        if is_melody:
            # Negative = earlier
            return -self.k * self.lead_ms / 1000

        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0
