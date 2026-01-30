"""
Tremolo rule - rapid alternation between two notes or repetition of single note.

Based on MusicXML tremolo ornaments (beamed tremolo, not bowed tremolo).
"""

from typing import Dict, Any, List
import numpy as np
from .base import Rule


class TremoloRule(Rule):
    """
    Tremolo: expand tremolo notation into rapid note repetitions.

    Two types:
    - Single-note tremolo: rapid repetition of same note (C-C-C-C...)
    - Two-note tremolo: rapid alternation between two notes (C4-C5-C4-C5...)

    Tremolo speed (from slash count):
    - 1 slash = 8th notes (1/8 beat)
    - 2 slashes = 16th notes (1/16 beat)
    - 3 slashes = 32nd notes (1/32 beat)
    """

    def __init__(self, config, velocity_variation: float = 3.0, timing_jitter_ms: float = 5.0):
        """
        Initialize tremolo rule.

        Args:
            config: RuleConfig with k value
            velocity_variation: Velocity variation between tremolo notes (dB)
            timing_jitter_ms: Timing jitter per tremolo note (milliseconds)
        """
        super().__init__(config)
        self.velocity_variation = velocity_variation
        self.timing_jitter_ms = timing_jitter_ms
        self.rng = None  # Set by engine

    def set_rng(self, rng: np.random.Generator):
        """Set random number generator."""
        self.rng = rng

    def expand_tremolo(
        self,
        note1: Dict[str, Any],
        note2: Dict[str, Any],
        tremolo_speed: int,
        features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Expand a tremolo pair into alternating notes.

        Args:
            note1: First note dict {pitch, onset, duration, velocity}
            note2: Second note dict (can be same as note1 for single-note tremolo)
            tremolo_speed: Number of slashes (1=8th, 2=16th, 3=32nd)
            features: Additional features (beat_duration, etc.)

        Returns:
            List of expanded note dicts
        """
        if not self.enabled:
            # Return original notes unchanged
            return [note1, note2] if note1 != note2 else [note1]

        # Calculate tremolo note duration
        beat_duration = features.get('beat_duration', 0.5)
        slashes_to_subdivision = {1: 8, 2: 16, 3: 32}
        subdivision = slashes_to_subdivision.get(tremolo_speed, 16)
        tremolo_note_duration = beat_duration / (subdivision / 4)  # Convert to beat fraction

        # Total duration covered by tremolo
        # For two-note tremolo (beamed tremolo), the two notes occur sequentially
        # in the score notation, but are replaced by rapid alternation between them.
        # Total duration = span from first note onset to end of second note
        onset_start = note1['onset']
        onset_end = note2['onset'] + note2['duration']
        total_duration = onset_end - onset_start

        # Calculate number of notes
        num_notes = int(total_duration / tremolo_note_duration)
        num_notes = max(2, num_notes)  # At least 2 notes

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Tremolo calc: duration={total_duration:.3f}s, tremolo_note_dur={tremolo_note_duration:.3f}s, "
                    f"num_notes={num_notes}, slashes={tremolo_speed}, pitches={note1['pitch']}-{note2['pitch']}")

        # Generate alternating notes
        notes = []
        current_time = onset_start
        is_first = True

        for i in range(num_notes):
            # Alternate between note1 and note2
            source_note = note1 if is_first else note2

            # Apply velocity variation (random per note)
            if self.rng is not None:
                velocity_delta = self.rng.normal(0, self.k * self.velocity_variation)
            else:
                velocity_delta = 0

            # Apply timing jitter
            if self.rng is not None:
                timing_delta = self.rng.normal(0, self.k * self.timing_jitter_ms / 1000)
            else:
                timing_delta = 0

            # Ensure we don't exceed the total duration
            remaining = onset_end - current_time
            actual_duration = min(tremolo_note_duration, remaining)

            if actual_duration <= 0:
                break

            notes.append({
                'pitch': source_note['pitch'],
                'onset': current_time + timing_delta,
                'duration': actual_duration * 0.95,  # Slight gap between notes
                'velocity': source_note['velocity'],
                'velocity_delta_dB': velocity_delta,
                'is_tremolo': True
            })

            current_time += tremolo_note_duration
            is_first = not is_first

        return notes

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No global velocity effect (handled in expand_tremolo)."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No global timing effect (handled in expand_tremolo)."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No global duration effect (handled in expand_tremolo)."""
        return 1.0
