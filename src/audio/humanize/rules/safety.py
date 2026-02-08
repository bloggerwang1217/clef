"""
Safety rules and global normalization.

Social-duration-care: auto-lengthen very short notes
GlobalNormalizer: prevent velocity clipping and "smashing piano"
"""

from typing import Dict, Any
import numpy as np
from .base import Rule


class SocialDurationCareRule(Rule):
    """
    Social-duration-care: auto-lengthen very short notes.

    Prevents notes from being too short to hear.
    KTH principle: "care for the listener"
    """

    def __init__(self, config, min_duration_ms: float = 50.0):
        """
        Initialize social-duration-care rule.

        Args:
            config: RuleConfig with k value
            min_duration_ms: Minimum audible duration (milliseconds), default 50
        """
        super().__init__(config)
        self.min_duration_ms = min_duration_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Extend very short notes to minimum audible duration."""
        if not self.enabled:
            return 1.0

        duration = features.get('duration', 0.5)
        note_duration_ms = duration * 1000

        if note_duration_ms < self.min_duration_ms:
            # Extend to minimum audible duration
            target_ratio = self.min_duration_ms / note_duration_ms
            return 1.0 + self.k * (target_ratio - 1.0)

        return 1.0


class GlobalNormalizer:
    """
    Global velocity normalization / soft limiting.

    Prevents "smashing piano" when multiple rules stack up.
    Applied as post-processing after all rules.
    """

    def __init__(
        self,
        target_rms_velocity: int = 70,
        max_velocity: int = 115,
        soft_clip_threshold: int = 100
    ):
        """
        Initialize global normalizer.

        Args:
            target_rms_velocity: Target RMS velocity level, default 70
            max_velocity: Maximum allowed velocity, default 115
            soft_clip_threshold: Threshold for soft clipping, default 100
        """
        self.target_rms_velocity = target_rms_velocity
        self.max_velocity = max_velocity
        self.soft_clip_threshold = soft_clip_threshold

    def normalize(self, velocities: np.ndarray) -> np.ndarray:
        """
        Apply global normalization and soft clipping.

        Args:
            velocities: Array of MIDI velocities

        Returns:
            Normalized velocities (1-127)
        """
        velocities = velocities.astype(float)

        # 1. RMS normalization (optional - can be disabled)
        # Uncomment if needed:
        # current_rms = np.sqrt(np.mean(velocities ** 2))
        # if current_rms > 0:
        #     scale = self.target_rms_velocity / current_rms
        #     velocities = velocities * scale

        # 2. Soft clipping for peaks
        # Use tanh-style soft clip above threshold
        above_threshold = velocities > self.soft_clip_threshold

        if np.any(above_threshold):
            excess = velocities[above_threshold] - self.soft_clip_threshold
            max_excess = self.max_velocity - self.soft_clip_threshold

            # Soft clip: compress excess into remaining headroom
            compressed = max_excess * np.tanh(excess / max_excess)
            velocities[above_threshold] = self.soft_clip_threshold + compressed

        # 3. Hard clip as safety
        velocities = np.clip(velocities, 1, 127)

        return velocities.astype(int)
