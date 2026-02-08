"""
Melodic-charge rule: emphasis on non-chord tones and notes with melodic tension.

KTH principle: 強調遠離和弦根音的音符 (不和諧或具導向性)
Effect (velocity): +k × 2 dB for high-confidence NCTs
Effect (timing): slight delay for appoggiaturas (agogic accent)
"""

from typing import Dict, Any
from .base import Rule
from ..non_chord_tone import NonChordToneDetector, NCTType


class MelodicChargeRule(Rule):
    """
    Melodic-charge: emphasis on non-chord tones.

    Uses heuristic NCT detection based on:
    1. Metric position (weak beat = likely NCT)
    2. Duration (short = likely NCT)
    3. Melodic motion (step-step patterns)
    4. Dissonance with concurrent notes
    """

    def __init__(self, config, nct_boost_dB: float = 2.0):
        """
        Initialize melodic-charge rule.

        Args:
            config: RuleConfig with k value
            nct_boost_dB: Maximum dB boost for NCTs (default 2.0)
        """
        super().__init__(config)
        self.nct_boost_dB = nct_boost_dB
        self.detector = NonChordToneDetector()

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply melodic-charge effect to velocity."""
        if not self.enabled:
            return 0.0

        # Get NCT analysis from features (computed by engine)
        # or compute on demand
        analysis = features.get('nct_analysis')

        if analysis is None:
            # Compute if not cached
            note_idx = features.get('note_idx', 0)
            note_array = features.get('note_array', None)

            if note_array is None:
                return 0.0

            analysis = self.detector.analyze_note(note_idx, note_array, features)

        # NCT confidence (0-1) → dB boost (0-2 dB)
        return self.k * analysis.melodic_charge

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """
        Appoggiaturas get agogic accent (slight delay).

        Appoggiatura is a strong-beat dissonance that resolves by step,
        creating melodic tension. Slight delay emphasizes this.
        """
        if not self.enabled:
            return 0.0

        analysis = features.get('nct_analysis')

        if analysis is None:
            note_idx = features.get('note_idx', 0)
            note_array = features.get('note_array', None)

            if note_array is None:
                return 0.0

            analysis = self.detector.analyze_note(note_idx, note_array, features)

        # Appoggiatura: slight delay (agogic accent)
        if analysis.nct_type == NCTType.APPOGGIATURA and analysis.confidence > 0.5:
            return self.k * 0.02  # 20ms delay

        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0
