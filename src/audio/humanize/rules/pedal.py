"""
Auto pedal rule: syncopated pedaling.

Priority:
1. Use score pedal markings if available (from MusicXML)
2. Fallback: pitch class set change detection per beat

Generates sustain pedal (CC64) events with syncopated timing.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict
from .base import Rule

import logging
logger = logging.getLogger(__name__)


@dataclass
class PedalEvent:
    """Sustain pedal event (CC64)."""
    time: float  # Seconds
    value: int  # 0 = off, 127 = on


class AutoPedalRule(Rule):
    """Auto pedal: syncopated pedaling from score markings or harmony detection."""

    def __init__(
        self,
        config,
        lift_before_ms: float = 30.0,
        press_after_ms: float = 20.0
    ):
        super().__init__(config)
        self.lift_before_ms = lift_before_ms
        self.press_after_ms = press_after_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        return 1.0

    # Minimum gap (ms) for a clean pedal change cycle:
    # lift (~30ms) + string damping (~70ms) + press (~30ms) = ~130ms
    # Use 150ms as safe minimum with margin
    MIN_PEDAL_CYCLE_MS = 150.0

    # Bass register threshold: measures where the lowest note is above this
    # MIDI pitch have no bass voice and should not use sustain pedal.
    # C3 (MIDI 48) is the standard bass/tenor boundary in piano music.
    NO_BASS_THRESHOLD = 48

    def generate_pedal_events(
        self,
        notes: List[Any],
        features_list: List[Dict[str, Any]],
        score_pedals: Optional[List[Dict]] = None
    ) -> List[PedalEvent]:
        """
        Generate CC64 events for sustain pedal.

        Uses score markings where available, fills gaps with heuristic.
        Skips pedal in measures with no bass voice (fast treble-only passages).
        """
        if not self.enabled or not features_list:
            return []

        beat_duration = features_list[0].get('beat_duration', 0.5)
        beats_per_measure = features_list[0].get('beats_per_measure', 4.0)
        measure_duration = beats_per_measure * beat_duration

        # Detect harmony change points using pitch class sets
        harmony_changes = self._detect_harmony_changes(features_list, beat_duration, beats_per_measure)

        # If score has pedal markings, merge: use score where available, heuristic elsewhere
        if score_pedals:
            harmony_changes = self._merge_with_score(harmony_changes, score_pedals)

        if not harmony_changes:
            return []

        # Filter out changes that are physically too close for a pedal cycle
        min_gap_sec = self.MIN_PEDAL_CYCLE_MS / 1000
        filtered = [harmony_changes[0]]
        skipped = 0
        for i in range(1, len(harmony_changes)):
            gap = harmony_changes[i] - filtered[-1]
            if gap >= min_gap_sec:
                filtered.append(harmony_changes[i])
            else:
                skipped += 1
                logger.debug(f"Skipping pedal change at {harmony_changes[i]:.3f}s "
                            f"(gap={gap*1000:.0f}ms < {self.MIN_PEDAL_CYCLE_MS:.0f}ms)")

        if skipped > 0:
            logger.info(f"Skipped {skipped} harmony changes (gap < {self.MIN_PEDAL_CYCLE_MS:.0f}ms)")
        harmony_changes = filtered

        # Detect no-bass measures: measures where the lowest note is above
        # the bass register threshold. These are treble-only passages (e.g.
        # fast scale runs) where sustain pedal would muddy the sound.
        measure_lowest = {}
        for features in features_list:
            onset = features.get('onset', 0.0)
            pitch = features.get('pitch', 60)
            m = int(onset / measure_duration)
            if m not in measure_lowest or pitch < measure_lowest[m]:
                measure_lowest[m] = pitch

        no_bass_measures = {m for m, lo in measure_lowest.items()
                           if lo >= self.NO_BASS_THRESHOLD}
        if no_bass_measures:
            no_bass_count = len(no_bass_measures)
            logger.info(f"Found {no_bass_count} no-bass measures (lowest >= MIDI {self.NO_BASS_THRESHOLD})")

        # Inject change points at the start of every no-bass measure.
        # Without this, pedal from a preceding measure with bass would
        # bleed into the no-bass measure (the lift logic only fires
        # on existing change points).
        change_set = set(harmony_changes)
        for m in no_bass_measures:
            m_onset = m * measure_duration
            if m_onset not in change_set:
                harmony_changes.append(m_onset)
                change_set.add(m_onset)
        harmony_changes = sorted(harmony_changes)

        # Build velocity map per pedal region for dynamic depth
        region_velocities = self._compute_region_velocities(
            harmony_changes, features_list
        )
        self._beat_duration = beat_duration  # Store for depth calculation

        # Generate syncopated pedal events
        lift_offset = self.k * self.lift_before_ms / 1000
        press_offset = self.k * self.press_after_ms / 1000

        events = []
        pedal_is_on = False

        for i, change_time in enumerate(harmony_changes):
            # Check if this change falls in a no-bass measure
            m = int(change_time / measure_duration)
            in_no_bass = m in no_bass_measures

            if in_no_bass:
                # No bass voice: lift pedal if on, don't press
                if pedal_is_on:
                    lift_time = max(0.0, change_time - lift_offset)
                    events.append(PedalEvent(time=lift_time, value=0))
                    pedal_is_on = False
                continue

            # Dynamic pedal depth based on local velocity
            pedal_value = self._velocity_to_depth(
                region_velocities.get(i, 80)
            )

            # Normal: lift before change, press after
            if pedal_is_on:
                lift_time = max(0.0, change_time - lift_offset)
                events.append(PedalEvent(time=lift_time, value=0))
            press_time = change_time + press_offset
            events.append(PedalEvent(time=press_time, value=pedal_value))
            pedal_is_on = True

        # Lift at end
        last_onset = features_list[-1].get('onset', 0.0)
        last_dur = features_list[-1].get('duration', 0.5)
        events.append(PedalEvent(time=last_onset + last_dur, value=0))

        logger.info(f"Generated {len(events)} pedal events "
                     f"({len(harmony_changes)} harmony changes)")

        return sorted(events, key=lambda e: e.time)

    def _compute_region_velocities(
        self,
        change_points: List[float],
        features_list: List[Dict]
    ) -> Dict[int, float]:
        """
        Compute average humanized velocity for each pedal region.

        Each region spans from change_points[i] to change_points[i+1].
        Returns dict mapping region index -> average velocity.
        """
        if not change_points or not features_list:
            return {}

        # Sort notes by onset for efficient region assignment
        note_data = []
        for f in features_list:
            onset = f.get('onset', 0.0)
            vel = f.get('humanized_velocity', f.get('velocity', 64))
            note_data.append((onset, vel))

        region_vels: Dict[int, list] = defaultdict(list)
        for onset, vel in note_data:
            # Binary search for region (which change_point interval)
            region_idx = 0
            for j, cp in enumerate(change_points):
                if onset >= cp:
                    region_idx = j
                else:
                    break
            region_vels[region_idx].append(vel)

        # Average velocity per region
        result = {}
        for idx, vels in region_vels.items():
            result[idx] = sum(vels) / len(vels)

        return result

    def _velocity_to_depth(self, avg_velocity: float) -> int:
        """
        Map average velocity to pedal depth (CC64 value).

        Two factors:
        1. Velocity: louder passages -> deeper pedal (more resonance)
        2. Tempo: faster tempo -> shallower pedal (avoid muddiness)

        At 80 BPM (beat=0.75s): full depth available
        At 145 BPM (beat=0.41s): depth scaled to ~55%
        """
        # Normalize velocity to 0-1 range (40-110 practical range)
        vel_norm = max(0.0, min(1.0, (avg_velocity - 40) / 70))

        # Tempo scaling: slower = deeper, faster = shallower
        # Reference: 0.75s beat (80 BPM) = scale 1.0
        beat_dur = getattr(self, '_beat_duration', 0.5)
        tempo_scale = min(1.0, beat_dur / 0.75)

        # Base depth + velocity contribution scaled by tempo and k
        depth = 50 + self.k * vel_norm * tempo_scale * 60
        return min(127, max(40, int(depth)))

    def _detect_harmony_changes(
        self,
        features_list: List[Dict],
        beat_duration: float,
        beats_per_measure: float
    ) -> List[float]:
        """
        Detect harmony change points using pitch class set comparison.

        For each beat, collect all sounding pitch classes.
        When the set changes significantly -> harmony change.
        Quantize to compound beats (e.g., in 6/8, compare every dotted quarter).
        """
        # Time-based analysis window, independent of meter.
        # Human ear needs ~0.6s to establish harmonic context.
        # At slow tempos (<=80 BPM), one beat is already sufficient.
        # At fast tempos (>=140 BPM), one beat is too short for arpeggiated chords.
        HARMONIC_CONTEXT_SEC = 0.6
        analysis_window = max(beat_duration, HARMONIC_CONTEXT_SEC)

        # 1. Collect pitch classes per analysis window
        window_pcs = defaultdict(set)  # window_idx -> set of pitch classes
        window_onsets = {}  # window_idx -> earliest onset

        for features in features_list:
            onset = features.get('onset', 0.0)
            pitch = features.get('pitch', 60)
            window_idx = int(onset / analysis_window)
            window_pcs[window_idx].add(pitch % 12)
            if window_idx not in window_onsets or onset < window_onsets[window_idx]:
                window_onsets[window_idx] = onset

        if not window_pcs:
            return []

        sorted_windows = sorted(window_pcs.keys())

        # 2. Track bass note changes at half-measure granularity.
        # Different bass notes create conflicting low-frequency resonance,
        # so pedal MUST change whenever the bass root changes.
        # Use half-measure windows (2 beats) to smooth over arpeggiation:
        # e.g., D2+A2 → A2 → D2+A2 is one chord, not three bass changes.
        # The lowest note in a half-measure represents the true bass root.
        half_measure = beat_duration * 2
        hm_bass = {}    # half_measure_idx -> lowest MIDI pitch below threshold
        hm_onsets = {}  # half_measure_idx -> earliest onset in that half-measure

        for features in features_list:
            onset = features.get('onset', 0.0)
            pitch = features.get('pitch', 60)
            if pitch < self.NO_BASS_THRESHOLD:
                hm_idx = int(onset / half_measure)
                if hm_idx not in hm_bass or pitch < hm_bass[hm_idx]:
                    hm_bass[hm_idx] = pitch
                if hm_idx not in hm_onsets or onset < hm_onsets[hm_idx]:
                    hm_onsets[hm_idx] = onset

        bass_change_times = set()
        last_bass_pitch = None
        for hm_idx in sorted(hm_bass.keys()):
            current_bass = hm_bass[hm_idx]
            if last_bass_pitch is not None and current_bass != last_bass_pitch:
                bass_change_times.add(hm_onsets[hm_idx])
            last_bass_pitch = current_bass

        if bass_change_times:
            logger.info(f"Detected {len(bass_change_times)} bass root changes (half-measure)")

        # 3. Compare adjacent windows: Jaccard distance
        change_points = [window_onsets[sorted_windows[0]]]  # Always start with first window
        last_pcs = window_pcs[sorted_windows[0]]

        for window_idx in sorted_windows[1:]:
            current_pcs = window_pcs[window_idx]

            # Jaccard similarity: |intersection| / |union|
            intersection = last_pcs & current_pcs
            union = last_pcs | current_pcs

            if union:
                similarity = len(intersection) / len(union)
            else:
                similarity = 1.0

            # Harmony change if Jaccard similarity < 0.5 (general harmonic shift)
            if similarity < 0.5:
                change_points.append(window_onsets[window_idx])

            last_pcs = current_pcs

        # Add all bass note change points directly (onset-level, not window-limited).
        # This ensures every bass pitch change gets its own pedal cycle.
        change_points_set = set(change_points)
        for bass_time in bass_change_times:
            if bass_time not in change_points_set:
                change_points.append(bass_time)

        return sorted(change_points)

    def _merge_with_score(
        self,
        heuristic_changes: List[float],
        score_pedals: List[Dict]
    ) -> List[float]:
        """
        Merge score pedal markings with heuristic harmony changes.

        Score pedal markings define WHERE pedal is used (on/off regions).
        Within pedal regions, harmony changes still trigger lift+re-press
        for clean pedaling. Heuristic fills gaps with no score markings.
        """
        if not score_pedals:
            return heuristic_changes

        # Score pedal regions: add their start times as change points
        # (pedal engages at each score marking start)
        covered = []
        score_starts = set()
        for pedal in sorted(score_pedals, key=lambda p: p['start']):
            covered.append((pedal['start'], pedal['end']))
            score_starts.add(pedal['start'])

        # Keep ALL heuristic changes that fall within score pedal regions
        # (harmony changes within a long pedal marking still need lift+press).
        # Also keep heuristic changes outside score regions (gap-filling).
        merged = set(score_starts)
        for change_time in heuristic_changes:
            merged.add(change_time)

        return sorted(merged)
