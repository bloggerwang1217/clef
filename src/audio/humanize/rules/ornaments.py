"""
Ornament handling rules: grace notes, trills, mordents.

These rules handle expanding ornament notation into actual notes.
Note: Full implementation requires integration with score parsing.
"""

from typing import Dict, Any, List
from .base import Rule


class GraceNoteRule(Rule):
    """
    Grace notes: play before the beat, steal time from previous note.

    Two styles:
    - Acciaccatura (斜線): very short, "crushed" into main note
    - Appoggiatura (無斜線): longer, more expressive
    """

    def __init__(
        self,
        config,
        acciaccatura_ms: float = 50.0,
        appoggiatura_ratio: float = 0.25
    ):
        """
        Initialize grace note rule.

        Args:
            config: RuleConfig with k value
            acciaccatura_ms: Duration of acciaccatura (milliseconds), default 50
            appoggiatura_ratio: Duration ratio of appoggiatura (0-1), default 0.25
        """
        super().__init__(config)
        self.acciaccatura_ms = acciaccatura_ms
        self.appoggiatura_ratio = appoggiatura_ratio

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect (handled during expansion)."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect (handled during expansion)."""
        return 1.0

    def compute_grace_timing(
        self,
        grace_note: Any,
        main_note: Any,
        features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute timing for grace note.

        Returns dict with:
        - grace_onset: when grace note starts
        - grace_duration: how long grace note lasts
        - main_onset_shift: how much main note is delayed (usually 0)
        """
        is_acciaccatura = features.get('is_acciaccatura', True)

        if is_acciaccatura:
            # Short, before the beat
            grace_duration = self.k * self.acciaccatura_ms / 1000
            main_onset = getattr(main_note, 'onset', 0.0)
            grace_onset = main_onset - grace_duration

            return {
                'grace_onset': grace_onset,
                'grace_duration': grace_duration,
                'main_onset_shift': 0.0
            }
        else:
            # Appoggiatura: takes time from main note
            beat_duration = features.get('beat_duration', 0.5)
            grace_duration = self.k * self.appoggiatura_ratio * beat_duration
            main_onset = getattr(main_note, 'onset', 0.0)

            return {
                'grace_onset': main_onset,
                'grace_duration': grace_duration,
                'main_onset_shift': grace_duration
            }


class TrillRule(Rule):
    """
    Trills: rapid alternation between main note and upper neighbor.

    Parameters:
    - trill_speed: notes per second (typically 6-12)
    - start_on_upper: whether to start on upper note (Baroque) or main (Romantic)
    """

    def __init__(
        self,
        config,
        trill_speed: float = 8.0,
        start_on_upper: bool = False
    ):
        """
        Initialize trill rule.

        Args:
            config: RuleConfig with k value
            trill_speed: Speed in notes per second, default 8.0
            start_on_upper: Start on upper note (Baroque style), default False
        """
        super().__init__(config)
        self.trill_speed = trill_speed
        self.start_on_upper = start_on_upper

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect (handled during expansion)."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect (handled during expansion)."""
        return 1.0

    def expand_trill(self, note: Any, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand a trilled note into alternating pitches.

        Returns list of {pitch, onset, duration, velocity} dicts.
        """
        has_trill = features.get('has_trill', False)

        if not has_trill:
            # No trill, return original note
            pitch = getattr(note, 'midi_pitch', 60)
            onset = getattr(note, 'onset', 0.0)
            duration = getattr(note, 'duration', 0.5)
            velocity = getattr(note, 'velocity', 64)

            return [{
                'pitch': pitch,
                'onset': onset,
                'duration': duration,
                'velocity': velocity
            }]

        # Expand trill
        notes = []
        pitch = getattr(note, 'midi_pitch', 60)
        onset = getattr(note, 'onset', 0.0)
        duration = getattr(note, 'duration', 0.5)
        velocity = getattr(note, 'velocity', 64)

        current_time = onset
        end_time = onset + duration

        # Base trill speed: 16th note (tempo-dependent)
        # k controls speed: k=1 means 16th notes, k=2 means 32nd notes (twice as fast)
        beat_duration = features.get('beat_duration', 0.5)
        sixteenth_duration = beat_duration / 4  # 16th note = 1/4 beat
        note_duration = sixteenth_duration / self.k

        trill_interval = features.get('trill_interval', 2)  # Usually whole/half step
        upper_pitch = pitch + trill_interval
        is_upper = self.start_on_upper

        while current_time < end_time - note_duration * 0.5:
            current_pitch = upper_pitch if is_upper else pitch
            dur = min(note_duration, end_time - current_time)

            notes.append({
                'pitch': current_pitch,
                'onset': current_time,
                'duration': dur,
                'velocity': velocity - (5 if is_upper else 0)  # Upper slightly softer
            })

            current_time += note_duration
            is_upper = not is_upper

        return notes


class MordentRule(Rule):
    """Mordent: quick alternation (main-upper-main or main-lower-main)."""

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect (handled during expansion)."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect (handled during expansion)."""
        return 1.0

    def expand_mordent(self, note: Any, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand mordent into 3 notes."""
        has_mordent = features.get('has_mordent', False)

        if not has_mordent:
            # No mordent, return original note
            pitch = getattr(note, 'midi_pitch', 60)
            onset = getattr(note, 'onset', 0.0)
            duration = getattr(note, 'duration', 0.5)
            velocity = getattr(note, 'velocity', 64)

            return [{
                'pitch': pitch,
                'onset': onset,
                'duration': duration,
                'velocity': velocity
            }]

        # Expand mordent
        pitch = getattr(note, 'midi_pitch', 60)
        onset = getattr(note, 'onset', 0.0)
        duration = getattr(note, 'duration', 0.5)
        velocity = getattr(note, 'velocity', 64)

        # Base mordent duration: ~80ms total
        # k controls speed: higher k = faster (shorter duration)
        # This matches trill behavior where k scales speed
        base_duration = 0.08  # 80ms base
        mordent_duration = base_duration / self.k  # k=1: 80ms, k=2: 40ms (faster)
        single_note_dur = mordent_duration / 3

        is_upper = features.get('mordent_type', 'upper') == 'upper'
        aux_pitch = pitch + (2 if is_upper else -2)

        return [
            {
                'pitch': pitch,
                'onset': onset,
                'duration': single_note_dur,
                'velocity': velocity
            },
            {
                'pitch': aux_pitch,
                'onset': onset + single_note_dur,
                'duration': single_note_dur,
                'velocity': velocity - 5
            },
            {
                'pitch': pitch,
                'onset': onset + 2 * single_note_dur,
                'duration': duration - 2 * single_note_dur,
                'velocity': velocity
            },
        ]
