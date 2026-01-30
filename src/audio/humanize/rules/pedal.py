"""
Auto pedal rule: syncopated pedaling based on harmony changes.

Generates sustain pedal (CC64) events:
- Lift pedal before chord change (avoid muddiness)
- Press pedal after new chord (syncopated pedaling)
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from .base import Rule


@dataclass
class PedalEvent:
    """Sustain pedal event (CC64)."""
    time: float  # Seconds
    value: int  # 0 = off, 127 = on


class AutoPedalRule(Rule):
    """Auto pedal: syncopated pedaling based on harmony changes."""

    def __init__(
        self,
        config,
        lift_before_ms: float = 30.0,
        press_after_ms: float = 20.0
    ):
        """
        Initialize auto pedal rule.

        Args:
            config: RuleConfig with k value
            lift_before_ms: Lift pedal before chord change (milliseconds)
            press_after_ms: Press pedal after chord change (milliseconds)
        """
        super().__init__(config)
        self.lift_before_ms = lift_before_ms
        self.press_after_ms = press_after_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0

    def generate_pedal_events(
        self,
        notes: List[Any],
        features_list: List[Dict[str, Any]]
    ) -> List[PedalEvent]:
        """
        Generate CC64 events for sustain pedal.

        Args:
            notes: List of note objects
            features_list: List of per-note features

        Returns:
            List of PedalEvent objects
        """
        if not self.enabled:
            return []

        events = []

        # Simple heuristic: detect chord changes based on onset clusters
        # This is a simplified implementation
        # Full implementation would use harmony analysis

        last_chord_time = None
        chord_threshold = 0.1  # Notes within 100ms are part of same chord

        for i, (note, features) in enumerate(zip(notes, features_list)):
            onset = getattr(note, 'onset', 0.0)

            # Check if this is a new chord (onset significantly different from last)
            is_new_chord = False

            if last_chord_time is None:
                is_new_chord = True
            elif abs(onset - last_chord_time) > chord_threshold:
                is_new_chord = True

            if is_new_chord:
                if last_chord_time is not None:
                    # Lift pedal before new chord
                    lift_time = onset - self.k * self.lift_before_ms / 1000
                    events.append(PedalEvent(time=lift_time, value=0))

                # Press pedal after new chord starts
                press_time = onset + self.k * self.press_after_ms / 1000
                events.append(PedalEvent(time=press_time, value=127))

                last_chord_time = onset

        # Lift pedal at end
        if notes:
            last_note = notes[-1]
            last_onset = getattr(last_note, 'onset', 0.0)
            last_duration = getattr(last_note, 'duration', 0.5)
            end_time = last_onset + last_duration

            events.append(PedalEvent(time=end_time, value=0))

        return sorted(events, key=lambda e: e.time)
