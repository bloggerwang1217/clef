"""
Tremolo rule - rapid alternation between two notes or repetition of single note.

Based on MusicXML tremolo ornaments (beamed tremolo, not bowed tremolo).
"""

from typing import Dict, Any, List
import numpy as np
from .base import Rule

import logging
logger = logging.getLogger(__name__)


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

    Humanization:
    - Ramp-up: first few notes slightly slower (hand settling in)
    - Velocity contour: gentle arch (louder in middle, softer at edges)
    - Balanced alternation: both pitches at similar velocity
    - Micro-timing jitter: small random variation per note
    """

    def __init__(self, config, velocity_variation: float = 0.5, timing_jitter_ms: float = 5.0):
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
        Expand a tremolo pair into alternating notes with human-like feel.

        Args:
            note1: First note dict {pitch, onset, duration, velocity}
            note2: Second note dict (same as note1 for single-note tremolo)
            tremolo_speed: Number of slashes (1=8th, 2=16th, 3=32nd)
            features: Additional features (beat_duration, etc.)

        Returns:
            List of expanded note dicts
        """
        if not self.enabled:
            return [note1, note2] if note1['pitch'] != note2['pitch'] else [note1]

        # Tremolo speed as hits-per-second (human motor frequency).
        # Slash count is a hint, not a strict subdivision.
        #
        # Two-note alternation (different keys): fingers alternate freely,
        # limited by hand speed. Range ~6-14 Hz.
        #
        # Single-note repetition (same key): key must fully reset (~40-50ms)
        # before re-strike, even with finger alternation (2-1-2-1).
        # Practical limit ~8-10 Hz.
        is_single = (note1['pitch'] == note2['pitch'])

        if is_single:
            SLASH_TO_HZ = {1: 6.0, 2: 8.0, 3: 10.0}
        else:
            SLASH_TO_HZ = {1: 8.0, 2: 11.0, 3: 14.0}

        base_hz = SLASH_TO_HZ.get(tremolo_speed, 8.0 if is_single else 11.0)
        base_note_duration = 1.0 / base_hz

        onset_start = note1['onset']
        onset_end = note2['onset'] + note2['duration']
        total_duration = onset_end - onset_start

        num_notes = int(total_duration / base_note_duration)
        num_notes = max(4, num_notes)  # At least 4 hits for any tremolo

        # Average velocity for balanced alternation
        avg_velocity = (note1['velocity'] + note2['velocity']) / 2

        logger.debug(f"Tremolo: dur={total_duration:.3f}s, note_dur={base_note_duration:.3f}s, "
                    f"n={num_notes}, {'single' if is_single else 'two-note'} "
                    f"({note1['pitch']}-{note2['pitch']})")

        notes = []
        current_time = onset_start
        is_first = True

        for i in range(num_notes):
            # Normalized position 0-1 through the tremolo
            pos = i / max(1, num_notes - 1)

            # 1. Ramp-up: first 2 notes are very slightly slower (hand settling in)
            if i < 2:
                ramp = 1.0 + (2 - i) * 0.04 * self.k  # +8%, +4%
            else:
                ramp = 1.0
            actual_note_dur = base_note_duration * ramp

            # 2. Velocity contour: gentle arch shape
            # Peak at ~60% through, softer at start and end
            arch = 1.0 - 0.6 * (2 * pos - 1.2) ** 2  # peaks at pos=0.6
            arch = max(0.0, min(1.0, arch))
            # Scale: ±1dB contour at k=1
            contour_dB = (arch - 0.5) * 2.0 * self.k

            # 3. Per-note random jitter (small)
            if self.rng is not None:
                vel_noise = self.rng.normal(0, self.k * self.velocity_variation)
                time_noise = self.rng.normal(0, self.k * self.timing_jitter_ms / 1000)
            else:
                vel_noise = 0
                time_noise = 0

            # 4. Pitch selection
            if is_single:
                pitch = note1['pitch']
            else:
                pitch = note1['pitch'] if is_first else note2['pitch']

            # Ensure we don't exceed total duration
            remaining = onset_end - current_time
            final_dur = min(actual_note_dur, remaining)
            if final_dur <= 0:
                break

            # For single-note tremolo, use wider gap to avoid same-pitch overlap
            gap_ratio = 0.85 if is_single else 0.95

            notes.append({
                'pitch': pitch,
                'onset': current_time + time_noise,
                'duration': final_dur * gap_ratio,
                'velocity': int(np.clip(avg_velocity, 1, 127)),
                'velocity_delta_dB': contour_dB + vel_noise,
                'is_tremolo': True,
                'original_note': note1.get('original_note'),
            })

            current_time += final_dur
            is_first = not is_first

        return notes

    def expand_single_tremolo(
        self,
        note: Dict[str, Any],
        tremolo_speed: int,
        features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Expand a single-note tremolo into repeated notes.

        Delegates to expand_tremolo with note2 = note1.
        """
        return self.expand_tremolo(note, note, tremolo_speed, features)

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No global velocity effect (handled in expand_tremolo)."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No global timing effect (handled in expand_tremolo)."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No global duration effect (handled in expand_tremolo)."""
        return 1.0
