"""
Jitter rule: micro timing fluctuations (motor noise).

Represents human error or motor noise, distinct from expressive rubato.
"""

from typing import Dict, Any, Optional
import numpy as np
from .base import Rule


class BeatJitterRule(Rule):
    """
    Beat-level jitter: micro timing fluctuations on beats.

    Represents motor noise / human error, not expressive rubato.
    Independent of tempo (motor noise is physical).
    """

    def __init__(self, config):
        """Initialize beat jitter rule."""
        super().__init__(config)
        self.rng: Optional[np.random.Generator] = None  # Set by engine
        self.is_tempo_affecting = False  # Micro fluctuations

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self.rng = rng

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply micro jitter timing variation."""
        if not self.enabled:
            return 0.0

        if self.rng is None:
            return 0.0

        # Random but correlated across nearby notes
        # Base jitter: 5ms std (Very tight professional level)
        # k=1.0 -> 5ms, k=2.0 -> 10ms (still tight)
        base_jitter = self.rng.normal(0, 0.005)

        # Downbeats tend to be slightly late (Weight/Stability)
        if features.get('is_downbeat', False):
            base_jitter += 0.002  # +2ms tendency

        return self.k * base_jitter

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0
