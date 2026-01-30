"""
Rubato rules: phrase-level tempo variations.

Phrase rubato: slower at phrase boundaries, faster in middle
"""

from typing import Dict, Any
from .base import Rule


class PhraseRubatoRule(Rule):
    """
    Phrase-level rubato: tempo varies within phrases.

    Based on KTH Phrase-arch rule (tempo component):
    - Slower at phrase start (settling in)
    - Faster in middle (forward momentum)
    - Slower at phrase end (breathing, punctuation)
    """

    def __init__(self, config, peak_position: float = 0.6):
        """
        Initialize phrase rubato rule.

        Args:
            config: RuleConfig with k value
            peak_position: Where tempo peak occurs (0-1), default 0.6
        """
        super().__init__(config)
        self.peak_position = peak_position
        self.is_tempo_affecting = False  # Agogic rubato: time is "stolen" then returned

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect (handled by PhraseArchRule)."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Return cumulative timing offset based on phrase position."""
        if not self.enabled:
            return 0.0

        phrase_pos = features.get('phrase_position')
        if phrase_pos is None:
            return 0.0

        # 1. Get current BPM for tempo-sensitive scaling
        current_bpm = features.get('bpm', 120.0)

        # 2. BPM Scaling Factor
        # Rubato strength should be inversely proportional to tempo
        # Reference: Repp (1995) "Quantitative Effects of Global Tempo",
        #            Honing (2006) "The Scaling of Expressive Timing"
        # Fast pieces (Presto) need less rubato to maintain rhythmic stability
        REFERENCE_BPM = 100.0  # Andante/Moderato baseline
        bpm_scale = (REFERENCE_BPM / current_bpm) ** 2  # Quadratic scaling

        # Limit scaling to avoid excessive rubato in very slow pieces
        bpm_scale = min(bpm_scale, 1.5)

        beat_duration = features.get('beat_duration', 0.5)

        # 3. Compute local tempo ratio
        # Asymmetric: slower start, faster middle, slowest end
        if phrase_pos < self.peak_position:
            # Accelerating phase: -0.1 → +0.1
            ratio = -0.1 + 0.2 * (phrase_pos / self.peak_position)
        else:
            # Decelerating phase: +0.1 → -0.15
            decel_pos = (phrase_pos - self.peak_position) / (1 - self.peak_position)
            ratio = 0.1 - 0.25 * decel_pos

        # 4. Convert tempo ratio to timing offset with BPM scaling
        # Negative ratio (slower) = positive timing offset (later)
        # For Hornet (145 BPM): bpm_scale = (100/145)^2 ≈ 0.476
        # Effective tempo variation: ±10% → ±4.76% (more stable)
        return -self.k * bpm_scale * ratio * beat_duration * 0.5

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0
