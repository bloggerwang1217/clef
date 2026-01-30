"""
Final-ritard rule: gradual slowdown at the end (runner stopping model).

KTH principle: 結尾漸慢，基於「跑步者停止」的物理模型
Reference: Friberg et al. (2006), Director Musices

Formula:
  v(x) = √(1 - k·x)                    // velocity (tempo) at position x
  Duration(x) = NominalDuration / v(x) // duration multiplier

Where x ∈ [0, 1] is position within ritard section.
"""

from typing import Dict, Any
import numpy as np
from .base import Rule


class FinalRitardRule(Rule):
    """Final-ritard: gradual slowdown at the end."""

    def __init__(self, config, ritard_measures: float = 2.0, v_min: float = 0.8):
        """
        Initialize final-ritard rule.

        Args:
            config: RuleConfig with k value
            ritard_measures: Duration of the ritardando in measures (default 2.0)
            v_min: Minimum velocity (tempo) for k=1, scales linearly with k (default 0.8 = 80% of original)
                   k=1 → v_min=0.8 (1.25x slower)
                   k=2 → v_min=0.6 (1.67x slower)
        """
        super().__init__(config)
        self.ritard_measures = ritard_measures
        self.v_min = v_min
        self.is_tempo_affecting = True

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply final ritardando timing delay."""
        if not self.enabled:
            return 0.0

        note_array = features.get('note_array')
        if note_array is None:
            return 0.0

        # Calculate end time of the piece
        last_note = note_array[-1]
        end_time = last_note['onset'] + last_note['duration']
        
        current_onset = features.get('onset', 0.0)
        time_remaining = end_time - current_onset

        beat_duration = features.get('beat_duration', 0.5)

        # Calculate measure duration from time signature
        # beats_per_measure is in quarter note units (e.g., 6/8 → 3, 4/4 → 4)
        beats_per_measure = features.get('beats_per_measure', 4.0)
        measure_duration = beats_per_measure * beat_duration

        # Scale ritardando by k: k=1 → 2 measures, k=2 → 3 measures
        actual_measures = self.ritard_measures + (self.k - 1)
        effective_duration = actual_measures * measure_duration

        # Only apply if within the last few seconds
        if time_remaining > effective_duration:
            return 0.0

        # Normalize position within ritard section (0=start, 1=end)
        # Avoid division by zero
        if effective_duration <= 0:
            return 0.0
            
        ritard_pos = 1.0 - (time_remaining / effective_duration)
        ritard_pos = max(0.0, min(1.0, ritard_pos))

        # KTH runner-stopping model: v(x) = √(1 - k·x)
        # Where v is relative velocity (tempo), x is position
        # Ensure non-negative argument to avoid NaN
        sqrt_arg = max(0.0, 1.0 - self.k * ritard_pos)
        v = np.sqrt(sqrt_arg)

        # Clamp to minimum velocity with linear scaling
        # k=1 → v_min=0.8 (1.25x slower)
        # k=2 → v_min=0.6 (1.67x slower)
        # k=3 → v_min=0.4 (2.5x slower)
        effective_v_min = max(self.v_min - 0.2 * (self.k - 1), 0.2)
        v = max(v, effective_v_min)

        # Duration multiplier: how much longer each beat takes
        duration_factor = 1.0 / v

        # For cumulative timing: return incremental delay per beat
        # delay = (duration_factor - 1) × beat_duration
        # This represents how much extra time this beat needs
        incremental_delay = (duration_factor - 1.0) * beat_duration

        return incremental_delay

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Apply duration stretching in ritard section."""
        if not self.enabled:
            return 1.0

        note_array = features.get('note_array')
        if note_array is None:
            return 1.0

        # Calculate ritard position (same logic as apply_timing)
        last_note = note_array[-1]
        end_time = last_note['onset'] + last_note['duration']
        current_onset = features.get('onset', 0.0)
        time_remaining = end_time - current_onset

        beat_duration = features.get('beat_duration', 0.5)

        # Calculate measure duration from time signature
        beats_per_measure = features.get('beats_per_measure', 4.0)
        measure_duration = beats_per_measure * beat_duration

        # Scale ritardando by k: k=1 → 2 measures, k=2 → 3 measures
        actual_measures = self.ritard_measures + (self.k - 1)
        effective_duration = actual_measures * measure_duration

        if time_remaining > effective_duration or effective_duration <= 0:
            return 1.0

        ritard_pos = 1.0 - (time_remaining / effective_duration)
        ritard_pos = max(0.0, min(1.0, ritard_pos))

        # KTH model: v(x) = √(1 - k·x)
        # Ensure non-negative argument to avoid NaN
        sqrt_arg = max(0.0, 1.0 - self.k * ritard_pos)
        v = np.sqrt(sqrt_arg)

        # Clamp to minimum velocity with linear scaling
        effective_v_min = max(self.v_min - 0.2 * (self.k - 1), 0.2)
        v = max(v, effective_v_min)

        # Duration multiplier: notes get longer as tempo slows
        # At ritard_pos=0: factor = 1.0 (normal)
        # k=1: factor = 1/0.8 = 1.25 (1.25x slower)
        # k=2: factor = 1/0.6 = 1.67 (1.67x slower)
        duration_factor = 1.0 / v

        return duration_factor
