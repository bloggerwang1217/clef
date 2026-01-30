"""
HumanizationEngine: main engine that applies all KTH rules.

Integrates all rules and generates humanized MIDI from scores.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import logging

from .config import HumanizationConfig
from .convert import velocity_to_dB, dB_to_velocity
from .metadata import HumanizationMetadata, create_metadata

# Import all rules
from .rules.tempo import TempoInterpreter
from .rules.high_loud import HighLoudRule
from .rules.phrase_arch import PhraseArchRule
from .rules.duration_contrast import DurationContrastRule
from .rules.melodic_charge import MelodicChargeRule
from .rules.rubato import PhraseRubatoRule
from .rules.jitter import BeatJitterRule
from .rules.final_ritard import FinalRitardRule
from .rules.fermata import FermataRule
from .rules.dynamics_tempo import CrescendoTempoRule
from .rules.articulation_tempo import ArticulationTempoRule
from .rules.punctuation import PunctuationRule
from .rules.leap import LeapRule
from .rules.repetition import RepetitionRule
from .rules.articulation import StaccatoRule, LegatoRule, TenutoRule, AccentRule, MarcatoRule
from .rules.ornaments import GraceNoteRule, TrillRule, MordentRule
from .rules.tremolo import TremoloRule
from .rules.safety import SocialDurationCareRule, GlobalNormalizer
from .rules.pedal import AutoPedalRule

logger = logging.getLogger(__name__)


class HumanizationEngine:
    """Main engine that applies all KTH rules to generate humanized MIDI."""

    def __init__(self, config: HumanizationConfig, seed: Optional[int] = None):
        """
        Initialize humanization engine.

        Args:
            config: HumanizationConfig with all rule settings
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._init_rules()

    def _init_rules(self):
        """Initialize all rule instances from config."""
        cfg = self.config

        # Tempo interpreter (not a rule - sets base tempo)
        self.tempo_interpreter = TempoInterpreter(cfg.default_bpm)

        # Velocity rules
        crescendo_tempo_rule = CrescendoTempoRule(cfg.crescendo_tempo, cfg.crescendo_tempo_max_change, cfg.crescendo_velocity_max_change_dB)

        # Articulation with velocity effects
        tenuto_rule = TenutoRule(cfg.tenuto, cfg.tenuto_extension_ratio)
        accent_rule = AccentRule(cfg.accent, cfg.accent_velocity_boost_dB, cfg.accent_delay_ms)
        marcato_rule = MarcatoRule(cfg.marcato, cfg.marcato_velocity_boost_dB, cfg.marcato_shortening_ratio)

        self.velocity_rules = [
            HighLoudRule(cfg.high_loud),
            PhraseArchRule(cfg.phrase_arch, cfg.phrase_peak_position),
            DurationContrastRule(cfg.duration_contrast),
            MelodicChargeRule(cfg.melodic_charge, cfg.nct_boost_dB),
            crescendo_tempo_rule,  # Add for velocity effect
            tenuto_rule,           # Tenuto: +1dB
            accent_rule,           # Accent: +3dB
            marcato_rule,          # Marcato: +5dB
        ]

        # Timing rules
        beat_jitter = BeatJitterRule(cfg.beat_jitter)
        beat_jitter.set_rng(self.rng)

        self.timing_rules = [
            PhraseRubatoRule(cfg.phrase_rubato, cfg.phrase_peak_position),
            beat_jitter,
            FinalRitardRule(cfg.final_ritard, cfg.final_ritard_measures),
            FermataRule(cfg.fermata, cfg.fermata_duration_multiplier, cfg.fermata_pause_beats),
            crescendo_tempo_rule,  # Reuse same instance for timing effect
            ArticulationTempoRule(cfg.articulation_tempo),
            PunctuationRule(cfg.punctuation, cfg.micropause_ms, cfg.phrase_end_shorten_ratio),
            LeapRule(cfg.leap, cfg.leap_threshold_semitones, cfg.leap_duration_effect, cfg.leap_micropause_ms),
            RepetitionRule(cfg.repetition, cfg.repetition_micropause_ms),
        ]

        # Articulation rules (reuse instances from velocity_rules for consistency)
        self.articulation_rules = [
            StaccatoRule(cfg.staccato),
            LegatoRule(cfg.legato, cfg.legato_overlap_base_ms),
            tenuto_rule,   # Reuse: slight duration extension
            accent_rule,   # Reuse: no duration effect, just for consistency
            marcato_rule,  # Reuse: slight duration shortening
        ]

        # Safety
        self.social_care = SocialDurationCareRule(cfg.social_duration_care, cfg.min_audible_duration_ms)
        self.normalizer = GlobalNormalizer(
            cfg.target_rms_velocity,
            cfg.max_velocity,
            cfg.soft_clip_threshold
        ) if cfg.normalize_velocity else None

        # Special
        self.pedal_rule = AutoPedalRule(cfg.pedal, cfg.pedal_lift_before_ms, cfg.pedal_press_after_ms)

        # Ornaments (handle separately - require expansion)
        self.grace_note_rule = GraceNoteRule(cfg.grace_note, cfg.acciaccatura_ms, cfg.appoggiatura_ratio)
        self.trill_rule = TrillRule(cfg.trill, cfg.trill_speed, cfg.trill_start_on_upper)
        self.mordent_rule = MordentRule(cfg.mordent)
        self.tremolo_rule = TremoloRule(cfg.tremolo, cfg.tremolo_velocity_variation, cfg.tremolo_timing_jitter_ms)
        self.tremolo_rule.set_rng(self.rng)

    def humanize_from_score(
        self,
        score_path: Path,
        output_midi_path: Path,
        format: str = 'kern',
        version: int = 0
    ) -> Tuple[Any, HumanizationMetadata]:
        """
        Main entry point: Score → Humanized MIDI.

        Args:
            score_path: Path to score file (kern, musicxml, etc.)
            output_midi_path: Path to output MIDI file
            format: Score format ('kern', 'musicxml', 'midi')
            version: Version number for metadata

        Returns:
            Tuple of (MidiFile, HumanizationMetadata)
        """
        try:
            import partitura as pt
            import mido
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            raise

        # 1. Load score with partitura
        logger.info(f"Loading score from {score_path}")
        if format == 'kern':
            score = pt.load_kern(str(score_path))
        elif format == 'musicxml':
            score = pt.load_musicxml(str(score_path))
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Get first part (piano solo)
        # Handle different return types from partitura
        if isinstance(score, list):
            part = score[0]
        elif hasattr(score, 'parts') and len(score.parts) > 0:
            # Score object with parts attribute
            part = score.parts[0]
        elif hasattr(score, 'notes_tied'):
            # Already a Part object
            part = score
        else:
            raise ValueError(f"Unsupported score type: {type(score)}")

        # 2. Get base tempo
        base_bpm = self.tempo_interpreter.get_tempo_from_score(part)

        # Fallback: if still at default, parse XML directly (much faster than music21)
        if base_bpm == self.config.default_bpm and format == 'musicxml':
            logger.info("Trying XML fallback for tempo...")
            base_bpm = self.tempo_interpreter.get_tempo_from_musicxml(str(score_path))
            if base_bpm != self.config.default_bpm:
                logger.info(f"Found tempo via XML: {base_bpm} BPM")

        logger.info(f"Base tempo: {base_bpm} BPM")

        # 2.5. Get notes list
        notes = part.notes_tied

        # 2.6. Detect tremolo notes from Partitura ornaments
        tremolo_notes = set()
        for i, note in enumerate(notes):
            if hasattr(note, 'ornaments') and note.ornaments and 'tremolo' in note.ornaments:
                tremolo_notes.add(i)
        logger.info(f"Found {len(tremolo_notes)} tremolo notes")

        # 2.7. Pair tremolo notes and get slash counts
        tremolo_pairs = {}
        if tremolo_notes and format == 'musicxml':
            tremolo_pairs = self._detect_tremolo_pairs(notes, tremolo_notes, str(score_path))
            logger.info(f"Found {len(tremolo_pairs)//2} tremolo pairs")

        # 2.8. Detect articulations from Partitura note attributes
        articulation_map = {}
        for i, note in enumerate(notes):
            if hasattr(note, 'articulations') and note.articulations:
                articulation_map[i] = note.articulations

        # Count articulation types (including detached-legato workaround)
        detached_legato_count = sum(1 for arts in articulation_map.values() if 'detached-legato' in arts)
        accent_count = sum(1 for arts in articulation_map.values() if 'accent' in arts or 'detached-legato' in arts)
        staccato_count = sum(1 for arts in articulation_map.values() if 'staccato' in arts or 'detached-legato' in arts)
        tenuto_count = sum(1 for arts in articulation_map.values() if 'tenuto' in arts)
        marcato_count = sum(1 for arts in articulation_map.values() if 'strong-accent' in arts)

        if detached_legato_count > 0:
            logger.info(f"Found articulations: {accent_count} accents ({detached_legato_count} from detached-legato), "
                       f"{staccato_count} staccatos ({detached_legato_count} from detached-legato), "
                       f"{tenuto_count} tenutos, {marcato_count} marcatos")
        else:
            logger.info(f"Found articulations: {accent_count} accents, {staccato_count} staccatos, "
                       f"{tenuto_count} tenutos, {marcato_count} marcatos")

        # 3. Extract features
        logger.info("Extracting features from score")
        features_list = self._extract_features(part, base_bpm, tremolo_pairs, articulation_map)

        # 4. Apply all rules
        logger.info("Applying humanization rules")
        humanized_notes = self._apply_rules(part.notes_tied, features_list)

        # 5. Generate pedal events
        pedal_events = self.pedal_rule.generate_pedal_events(part.notes_tied, features_list)

        # 6. Convert to MIDI
        logger.info(f"Writing MIDI to {output_midi_path}")
        midi_file = self._notes_to_midi(humanized_notes, pedal_events, base_bpm)
        midi_file.save(str(output_midi_path))

        # 7. Create metadata
        metadata = create_metadata(
            source_file=str(score_path),
            version=version,
            config=self.config,
            seed=self.rng.bit_generator.state.get('state', {}).get('state', None)
        )
        metadata.save(output_midi_path)

        return midi_file, metadata

    def _extract_features(self, part, base_bpm: float, tremolo_pairs: Dict[int, Dict] = None,
                         articulation_map: Dict[int, List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract per-note features using partitura.

        Args:
            part: partitura Part object
            base_bpm: Base tempo in BPM
            tremolo_pairs: Dict from _parse_tremolo_from_musicxml (optional)
            articulation_map: Dict mapping note index to list of articulation strings (optional)

        Returns:
            List of feature dicts (one per note)
        """
        if tremolo_pairs is None:
            tremolo_pairs = {}
        if articulation_map is None:
            articulation_map = {}
        # 0. Calculate seconds_per_unit for time conversion
        beat_duration = 60.0 / base_bpm
        try:
            # Get divisions per quarter note from Partitura
            qd_list = list(part.quarter_durations())
            quarter = qd_list[0][1] if qd_list else 1
            seconds_per_unit = beat_duration / quarter
            logger.info(f"Time conversion: {quarter} divisions per quarter note, {seconds_per_unit:.6f} seconds per division")
        except Exception as e:
            logger.warning(f"Failed to get quarter duration: {e}, using fallback")
            seconds_per_unit = beat_duration

        # 1. Collect Direction objects (dynamics, tempo changes, pedal)
        logger.info("Collecting Direction objects from score")
        directions = self._collect_directions(part, seconds_per_unit)

        # 2. Try to extract basis functions from Partitura
        import partitura.musicanalysis as ma

        # Complete feature functions for music expression analysis
        # Note: function names end with _feature (not _basis)
        # The returned descriptors are named as <function_name>.<descriptor>
        # e.g., polynomial_pitch_feature → polynomial_pitch_feature.pitch
        #
        # See docs/partitura-feature-functions.md for complete reference
        feature_functions = [
            # === Core features ===
            'polynomial_pitch_feature',       # Pitch analysis
            'loudness_direction_feature',     # Dynamics (crescendo/diminuendo)
            'tempo_direction_feature',        # Tempo markings
            'articulation_direction_feature', # Articulation (staccato, tenuto, accent, marcato)
                                              # Note: articulation_feature has bugs, use this instead
            'duration_feature',               # Note duration
            'slur_feature',                   # Phrase boundaries (slur_incr/slur_decr)
            'fermata_feature',                # Fermata marks
            'grace_feature',                  # Grace notes

            # === Enhanced features ===
            'metrical_strength_feature',      # Metrical strength (better than metrical_feature)
            'vertical_neighbor_feature',      # Harmonic context (for NCT detection)
            'onset_feature',                  # Score position (for ritardando)
        ]

        # Try to get feature functions
        try:
            logger.info(f"Extracting Partitura features: {feature_functions}")
            basis_matrix, basis_names = ma.make_note_feats(part, feature_functions)
            logger.info(f"Successfully extracted {len(basis_names)} features")
        except Exception as e:
            logger.warning(f"Failed to extract basis functions: {e}")
            logger.info("Falling back to minimal features")
            # Fallback: create minimal features
            features_list = self._create_minimal_features(part, base_bpm)
            self._add_direction_features(features_list, directions)
            return features_list

        # Convert to per-note dicts
        features_list = []
        notes = part.notes_tied
        beat_duration = 60.0 / base_bpm

        for i, note in enumerate(notes):
            # Extract basis features
            features = {
                basis_names[j]: basis_matrix[i, j]
                for j in range(len(basis_names))
            }

            # Add computed features
            # Convert Partitura times (divisions) to seconds
            onset_sec = note.start.t * seconds_per_unit
            duration_sec = note.duration_tied * seconds_per_unit

            features.update({
                'note_idx': i,
                'pitch': note.midi_pitch,
                'onset': onset_sec,  # In seconds
                'duration': duration_sec,  # In seconds
                'beat_duration': beat_duration,
                'bpm': base_bpm,
                'piece_position': i / len(notes),
                'note_array': np.array([(n.midi_pitch, n.start.t * seconds_per_unit, n.duration_tied * seconds_per_unit) for n in notes],
                                      dtype=[('pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')]),
            })

            # Compute phrase position from slur basis
            features['phrase_position'] = self._compute_phrase_position(i, features, notes)

            # Compute intervals
            if i > 0:
                features['interval_from_prev'] = note.midi_pitch - notes[i-1].midi_pitch
            else:
                features['interval_from_prev'] = 0

            if i < len(notes) - 1:
                features['interval_to_next'] = notes[i+1].midi_pitch - note.midi_pitch
            else:
                features['interval_to_next'] = 0

            # Check for repeated note
            features['is_repeated_note'] = (
                i > 0 and note.midi_pitch == notes[i-1].midi_pitch
            )

            # Relative duration (vs local average)
            local_window = 5
            start = max(0, i - local_window)
            end = min(len(notes), i + local_window + 1)
            local_durations = [n.duration_tied * seconds_per_unit for n in notes[start:end]]
            avg_duration = np.mean(local_durations) if local_durations else duration_sec
            features['relative_duration'] = duration_sec / avg_duration if avg_duration > 0 else 1.0

            # Add tremolo features
            if i in tremolo_pairs:
                features['has_tremolo'] = True
                features['tremolo_partner_idx'] = tremolo_pairs[i]['partner_idx']
                features['tremolo_slashes'] = tremolo_pairs[i]['slashes']
                features['tremolo_is_start'] = tremolo_pairs[i]['is_start']
            else:
                features['has_tremolo'] = False
                features['tremolo_partner_idx'] = None
                features['tremolo_slashes'] = 0
                features['tremolo_is_start'] = False

            # Add articulation features from note.articulations attribute
            # This is more reliable than articulation_direction_feature which may return zeros
            articulations = articulation_map.get(i, [])

            # Workaround: Some MusicXML exporters (e.g., MuseScore) incorrectly export
            # "staccato + accent" as "detached-legato". Treat it as both.
            has_detached_legato = 'detached-legato' in articulations

            features['accent'] = 'accent' in articulations or has_detached_legato
            features['staccato'] = 'staccato' in articulations or has_detached_legato
            features['tenuto'] = 'tenuto' in articulations
            features['marcato'] = 'strong-accent' in articulations  # MusicXML uses 'strong-accent' for marcato

            features_list.append(features)

        return features_list

    def _create_minimal_features(self, part, base_bpm: float) -> List[Dict[str, Any]]:
        """Create minimal features when basis functions fail."""
        import partitura as pt

        notes = part.notes_tied
        beat_duration = 60.0 / base_bpm  # Seconds per quarter note

        # Get divisions per quarter note from first measure
        # In MusicXML, note.start.t and note.duration are in divisions
        # We need to find how many divisions = 1 quarter note
        try:
            # Get time signature
            ts = list(part.iter_all(pt.score.TimeSignature))[0]
            ts_beats = ts.beats
            ts_beat_type = ts.beat_type
            logger.info(f"Time signature: {ts_beats}/{ts_beat_type}")

            # Calculate beats per measure in quarter note units
            # For x/4 time: beats_per_measure = ts_beats (e.g., 4/4 → 4 quarters)
            # For x/8 time: beats_per_measure = ts_beats / 2 (e.g., 6/8 → 3 quarters)
            if ts_beat_type == 8:
                beats_per_measure = ts_beats / 2.0
            else:  # Assume /4 or /2
                beats_per_measure = ts_beats * (4.0 / ts_beat_type)

            logger.info(f"Beats per measure (in quarter notes): {beats_per_measure}")

            # In MusicXML, divisions are per quarter note
            # The relationship between note.duration and actual duration depends on note type
            # For a quarter note, note.duration should equal divisions
            # For a half note, note.duration should equal 2 * divisions

            # Find a note with known symbolic_duration to calibrate
            divisions_per_quarter = None
            for note in notes[:20]:  # Check first 20 notes
                if hasattr(note, 'symbolic_duration') and note.symbolic_duration:
                    sym_dur = note.symbolic_duration
                    note_type = sym_dur.get('type')
                    dots = sym_dur.get('dots', 0)

                    # Calculate quarter note equivalent for this note type
                    # Base durations in quarter notes
                    type_to_quarters = {
                        'whole': 4.0,
                        'half': 2.0,
                        'quarter': 1.0,
                        'eighth': 0.5,
                        '16th': 0.25,
                        '32nd': 0.125,
                    }

                    if note_type in type_to_quarters:
                        base_quarters = type_to_quarters[note_type]

                        # Apply dots: each dot adds half of the previous duration
                        # 1 dot: × 1.5, 2 dots: × 1.75, etc.
                        dotted_multiplier = 1.0
                        for _ in range(dots):
                            dotted_multiplier += 0.5 ** (_ + 1)

                        quarters = base_quarters * dotted_multiplier

                        # divisions_per_quarter = note.duration / quarters
                        divisions_per_quarter = note.duration / quarters
                        logger.info(f"Calibration note: {note_type} (dots={dots}) → {quarters} quarters, duration={note.duration} → divisions/quarter={divisions_per_quarter}")
                        break

            if divisions_per_quarter is None:
                # Fallback: assume divisions = 4 (common in MusicXML)
                logger.warning("Could not determine divisions per quarter, assuming 4")
                divisions_per_quarter = 4

            seconds_per_division = beat_duration / divisions_per_quarter
            logger.info(f"Divisions per quarter note: {divisions_per_quarter}")

        except Exception as e:
            logger.warning(f"Failed to determine divisions: {e}")
            # Fallback: assume 4 divisions per quarter (common default)
            divisions_per_quarter = 4
            seconds_per_division = beat_duration / 4
            # Fallback time signature (4/4)
            ts_beats = 4
            ts_beat_type = 4
            beats_per_measure = 4.0

        # Pre-compute note array for FinalRitard and other rules
        all_durations = [n.duration * seconds_per_division for n in notes]
        note_array = np.array(
            [(n.midi_pitch, n.start.t * seconds_per_division, n.duration * seconds_per_division) for n in notes],
            dtype=[('pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')]
        )

        features_list = []
        for i, note in enumerate(notes):
            # Use start.t and duration (in divisions)
            onset_sec = note.start.t * seconds_per_division
            duration_sec = note.duration * seconds_per_division

            # Calculate intervals
            if i > 0:
                interval_from_prev = note.midi_pitch - notes[i-1].midi_pitch
            else:
                interval_from_prev = 0

            if i < len(notes) - 1:
                interval_to_next = notes[i+1].midi_pitch - note.midi_pitch
            else:
                interval_to_next = 0

            # Check for repeated note
            is_repeated = i > 0 and note.midi_pitch == notes[i-1].midi_pitch

            # Calculate relative duration (vs local window of 5 notes)
            window_size = 5
            start_idx = max(0, i - window_size)
            end_idx = min(len(notes), i + window_size + 1)
            local_durations = all_durations[start_idx:end_idx]
            avg_duration = sum(local_durations) / len(local_durations) if local_durations else duration_sec
            relative_duration = duration_sec / avg_duration if avg_duration > 0 else 1.0

            features = {
                'note_idx': i,
                'pitch': note.midi_pitch,
                'onset': onset_sec,
                'duration': duration_sec,
                'beat_duration': beat_duration,
                'bpm': base_bpm,
                'piece_position': i / len(notes),
                'note_array': note_array,  # For analysis context
                'ts_beats': ts_beats,  # Time signature numerator
                'ts_beat_type': ts_beat_type,  # Time signature denominator
                'beats_per_measure': beats_per_measure,  # Beats per measure in quarter note units
                'phrase_position': 0.5,  # Default to middle
                'interval_from_prev': interval_from_prev,
                'interval_to_next': interval_to_next,
                'is_repeated_note': is_repeated,
                'relative_duration': relative_duration,
                'beat_strength': 0.5,
                'is_downbeat': False,
            }
            features_list.append(features)

        return features_list

    def _compute_phrase_position(self, note_idx: int, features: Dict, notes: List) -> float:
        """Compute position within current phrase (0-1)."""
        # Use slur_basis if available
        slur_incr = features.get('slur_basis.slur_incr', 0)
        slur_decr = features.get('slur_basis.slur_decr', 0)

        # Simple heuristic: assume phrases every 8 notes if no slur info
        if slur_incr == 0 and slur_decr == 0:
            phrase_len = 8
            return (note_idx % phrase_len) / phrase_len

        # TODO: More sophisticated phrase detection from slur basis
        return 0.5  # Default to middle

    def _collect_directions(self, part, seconds_per_unit: float) -> Dict[str, List]:
        """
        Collect all Direction objects from score.

        Args:
            part: partitura Part object
            seconds_per_unit: Conversion factor from symbolic time to seconds

        Returns:
            Dict with keys: 'dynamics', 'crescendos', 'tempo_changes', 'pedals'
        """
        import partitura as pt

        directions = {
            'dynamics': [],       # ConstantLoudnessDirection
            'crescendos': [],     # IncreasingLoudnessDirection
            'diminuendos': [],    # DecreasingLoudnessDirection (if exists)
            'tempo_changes': [],  # DecreasingTempoDirection (rit., accel.)
            'pedals': [],         # SustainPedalDirection
        }

        # Collect ConstantLoudnessDirection (f, mf, p, etc.)
        for obj in part.iter_all(pt.score.ConstantLoudnessDirection):
            directions['dynamics'].append({
                'start': obj.start.t * seconds_per_unit,  # Convert to seconds
                'end': obj.end.t * seconds_per_unit,
                'text': obj.text,
                'staff': obj.staff,
            })

        # Collect IncreasingLoudnessDirection (crescendo)
        for obj in part.iter_all(pt.score.IncreasingLoudnessDirection):
            directions['crescendos'].append({
                'start': obj.start.t * seconds_per_unit,
                'end': obj.end.t * seconds_per_unit,
                'text': obj.text,
                'wedge': getattr(obj, 'wedge', False),
            })

        # Collect DecreasingTempoDirection (rit., rall., accel.)
        for obj in part.iter_all(pt.score.DecreasingTempoDirection):
            directions['tempo_changes'].append({
                'start': obj.start.t * seconds_per_unit,
                'end': obj.end.t * seconds_per_unit,
                'text': obj.text,
                'raw_text': getattr(obj, 'raw_text', None),
            })

        # Collect SustainPedalDirection
        for obj in part.iter_all(pt.score.SustainPedalDirection):
            directions['pedals'].append({
                'start': obj.start.t * seconds_per_unit,
                'end': obj.end.t * seconds_per_unit,
                'staff': obj.staff,
            })

        logger.info(f"Collected directions: {len(directions['dynamics'])} dynamics, "
                   f"{len(directions['crescendos'])} crescendos, "
                   f"{len(directions['tempo_changes'])} tempo changes, "
                   f"{len(directions['pedals'])} pedals")

        return directions

    def _add_direction_features(self, features_list: List[Dict], directions: Dict):
        """
        Add Direction info to each note's features.

        Args:
            features_list: List of feature dicts (will be modified in-place)
            directions: Dict from _collect_directions
        """
        for features in features_list:
            note_time = features['onset']  # in seconds (converted from symbolic time)

            # Find current dynamic marking
            features['current_dynamic'] = self._find_active_direction(
                note_time, directions['dynamics']
            )

            # Check if in crescendo and calculate position (0-1)
            crescendo = self._find_active_direction(note_time, directions['crescendos'])
            if crescendo:
                features['in_crescendo'] = True
                # Calculate position within crescendo (0=start, 1=end)
                duration = crescendo['end'] - crescendo['start']
                if duration > 0:
                    features['crescendo_position'] = (note_time - crescendo['start']) / duration
                else:
                    features['crescendo_position'] = 0.0
            else:
                features['in_crescendo'] = False
                features['crescendo_position'] = 0.0

            # Check if in diminuendo and calculate position (0-1)
            diminuendo = self._find_active_direction(note_time, directions.get('diminuendos', []))
            if diminuendo:
                features['in_diminuendo'] = True
                duration = diminuendo['end'] - diminuendo['start']
                if duration > 0:
                    features['diminuendo_position'] = (note_time - diminuendo['start']) / duration
                else:
                    features['diminuendo_position'] = 0.0
            else:
                features['in_diminuendo'] = False
                features['diminuendo_position'] = 0.0

            # Check if in tempo change (rit./accel.)
            features['tempo_direction'] = self._find_active_direction(
                note_time, directions['tempo_changes']
            )

            # Check if pedal is down
            features['pedal_down'] = self._is_in_direction(
                note_time, directions['pedals']
            )

    def _find_active_direction(self, note_time: float, direction_list: List[Dict]) -> Optional[Dict]:
        """Find the direction that is active at note_time."""
        for direction in direction_list:
            if direction['start'] <= note_time < direction['end']:
                return direction
        return None

    def _is_in_direction(self, note_time: float, direction_list: List[Dict]) -> bool:
        """Check if note_time is within any direction's range."""
        return self._find_active_direction(note_time, direction_list) is not None

    def _apply_rules(
        self,
        notes: List,
        features_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply all rules to notes.

        Args:
            notes: List of Note objects from partitura
            features_list: List of feature dicts

        Returns:
            List of humanized note dicts with velocity, onset, duration
        """
        humanized = []

        # Cumulative drift for tempo-affecting rules (e.g., FinalRitard)
        # Position-only rules (e.g., PhraseRubato, MicroTiming) don't accumulate
        cumulative_drift = 0.0
        previous_onset = -1.0  # Track onset to handle chords correctly

        for note, features in zip(notes, features_list):
            # Apply velocity rules (additive in dB)
            base_velocity = self._get_base_velocity(note, features)
            base_dB = velocity_to_dB(base_velocity, self.config.reference_velocity)

            for rule in self.velocity_rules:
                base_dB += rule.apply_velocity(note, features)

            final_velocity = dB_to_velocity(base_dB, self.config.reference_velocity)

            # Apply timing rules
            # Separate tempo-affecting (cumulative) and position-only (local) rules
            base_onset = features.get('onset', note.start.t)
            tempo_offset = 0.0      # Cumulative (e.g., FinalRitard)
            position_offset = 0.0   # Local only (e.g., PhraseRubato, MicroTiming)

            for rule in self.timing_rules:
                offset = rule.apply_timing(note, features)
                if rule.is_tempo_affecting:
                    tempo_offset += offset
                else:
                    position_offset += offset

            # Accumulate tempo-affecting offsets ONLY when time moves forward
            # This ensures chord notes (same onset) get the same drift
            if base_onset > previous_onset:
                cumulative_drift += tempo_offset
                previous_onset = base_onset

            # Apply both cumulative drift and local position offset
            final_onset = base_onset + cumulative_drift + position_offset

            # Apply duration rules (multiplicative)
            # Use duration from features (already in seconds)
            base_duration = features.get('duration', 0.5)
            duration_multiplier = 1.0

            # Articulation rules
            for rule in self.articulation_rules:
                duration_multiplier *= rule.apply_duration(note, features)

            # Safety rule
            duration_multiplier *= self.social_care.apply_duration(note, features)

            # Timing rules can also affect duration
            for rule in self.timing_rules:
                duration_multiplier *= rule.apply_duration(note, features)

            final_duration = base_duration * duration_multiplier

            note_dict = {
                'pitch': note.midi_pitch,
                'onset': max(0, final_onset),  # Ensure non-negative
                'duration': max(0.01, final_duration),  # Ensure positive
                'velocity': int(np.clip(final_velocity, 1, 127)),
                'original_note': note
            }

            # Check for tremolo
            if features.get('has_tremolo', False):
                if features.get('tremolo_is_start', False):
                    # This is the first note of a tremolo pair
                    # Get partner note
                    partner_idx = features['tremolo_partner_idx']
                    logger.debug(f"Processing tremolo start at note {features['note_idx']}, partner {partner_idx}")
                    if partner_idx < len(notes):
                        partner_note = notes[partner_idx]
                        partner_features = features_list[partner_idx]

                        # Get partner's humanized parameters
                        partner_velocity = self._get_base_velocity(partner_note, partner_features)
                        partner_dB = velocity_to_dB(partner_velocity, self.config.reference_velocity)
                        for rule in self.velocity_rules:
                            partner_dB += rule.apply_velocity(partner_note, partner_features)
                        partner_velocity_final = dB_to_velocity(partner_dB, self.config.reference_velocity)

                        partner_onset = partner_features.get('onset', partner_note.start.t)
                        partner_duration = partner_features.get('duration', 0.5)

                        # Expand tremolo
                        # IMPORTANT: Use original durations, not the modified ones!
                        tremolo_notes = self.tremolo_rule.expand_tremolo(
                            note1={
                                'pitch': note.midi_pitch,
                                'onset': note_dict['onset'],
                                'duration': base_duration,  # Original duration before multiplier!
                                'velocity': note_dict['velocity'],
                            },
                            note2={
                                'pitch': partner_note.midi_pitch,
                                'onset': partner_onset + cumulative_drift,
                                'duration': partner_features.get('duration', 0.5),  # Original duration!
                                'velocity': int(np.clip(partner_velocity_final, 1, 127)),
                            },
                            tremolo_speed=features['tremolo_slashes'],
                            features=features
                        )

                        logger.debug(f"Expanded tremolo: {len(tremolo_notes)} notes from {note.midi_pitch}-{partner_note.midi_pitch}")

                        # Add all expanded notes
                        for tn in tremolo_notes:
                            # Apply velocity delta from tremolo rule
                            velocity_with_variation = dB_to_velocity(
                                velocity_to_dB(tn['velocity'], self.config.reference_velocity) + tn.get('velocity_delta_dB', 0),
                                self.config.reference_velocity
                            )
                            humanized.append({
                                'pitch': tn['pitch'],
                                'onset': tn['onset'],
                                'duration': tn['duration'],
                                'velocity': int(np.clip(velocity_with_variation, 1, 127)),
                                'original_note': note
                            })
                    else:
                        # Partner not found, add original note
                        humanized.append(note_dict)
                else:
                    # This is the second note of a tremolo pair, skip it
                    # (already handled when processing the first note)
                    pass
            else:
                # Not a tremolo note, add normally
                humanized.append(note_dict)

        # Apply global normalization to velocities
        if self.normalizer is not None:
            velocities = np.array([n['velocity'] for n in humanized])
            normalized = self.normalizer.normalize(velocities)
            for i, vel in enumerate(normalized):
                humanized[i]['velocity'] = vel

        return humanized

    def _get_base_velocity(self, note: Any, features: Dict[str, Any]) -> int:
        """Get base velocity from dynamics marking or default."""
        # Check for dynamics from loudness_direction_basis
        for marking, velocity in self.config.dynamics_map.items():
            key = f'loudness_direction_basis.{marking}'
            if features.get(key, 0) > 0.5:
                return velocity

        return self.config.default_velocity

    def _notes_to_midi(
        self,
        notes: List[Dict[str, Any]],
        pedal_events: List,
        bpm: float
    ) -> Any:
        """
        Convert humanized notes to MIDI file.

        Args:
            notes: List of humanized note dicts
            pedal_events: List of PedalEvent objects
            bpm: Tempo in BPM

        Returns:
            mido.MidiFile object
        """
        import mido

        # Create MIDI file
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))

        # Convert notes to MIDI events
        events = []

        for note in notes:
            # Note on
            events.append({
                'time': note['onset'],
                'type': 'note_on',
                'note': note['pitch'],
                'velocity': note['velocity'],
            })

            # Note off
            events.append({
                'time': note['onset'] + note['duration'],
                'type': 'note_off',
                'note': note['pitch'],
                'velocity': 0,
            })

        # Add pedal events
        for pedal in pedal_events:
            events.append({
                'time': pedal.time,
                'type': 'control_change',
                'control': 64,  # Sustain pedal
                'value': pedal.value,
            })

        # Sort events by time
        events.sort(key=lambda e: e['time'])

        # Convert to MIDI messages with delta times
        ticks_per_second = mid.ticks_per_beat * (bpm / 60.0)
        last_time = 0

        for event in events:
            delta_time = event['time'] - last_time
            delta_ticks = int(delta_time * ticks_per_second)

            if event['type'] == 'note_on':
                track.append(mido.Message(
                    'note_on',
                    note=event['note'],
                    velocity=event['velocity'],
                    time=delta_ticks
                ))
            elif event['type'] == 'note_off':
                track.append(mido.Message(
                    'note_off',
                    note=event['note'],
                    velocity=0,
                    time=delta_ticks
                ))
            elif event['type'] == 'control_change':
                track.append(mido.Message(
                    'control_change',
                    control=event['control'],
                    value=event['value'],
                    time=delta_ticks
                ))

            last_time = event['time']

        # Add end of track
        track.append(mido.MetaMessage('end_of_track', time=0))

        return mid

    def _detect_tremolo_pairs(self, notes: List, tremolo_notes: set, xml_path: str) -> Dict[int, Dict]:
        """
        Detect tremolo pairs from Partitura notes with ornaments.

        Strategy:
        1. Find consecutive tremolo notes (they form pairs)
        2. Parse slash count from MusicXML for each pair

        Args:
            notes: List of partitura Note objects
            tremolo_notes: Set of note indices with tremolo ornament
            xml_path: Path to MusicXML (for slash count)

        Returns:
            Dict mapping note index to tremolo info
        """
        import xml.etree.ElementTree as ET
        import zipfile

        tremolo_pairs = {}

        # Build list of tremolo indices
        tremolo_indices = sorted(list(tremolo_notes))

        # Parse slash counts from XML
        slash_counts = {}  # onset -> slashes
        try:
            if xml_path.endswith('.mxl'):
                with zipfile.ZipFile(xml_path) as z:
                    for name in z.namelist():
                        if name.endswith('.xml') and not name.startswith('META-INF'):
                            with z.open(name) as f:
                                tree = ET.parse(f)
                                root = tree.getroot()
                            break
            else:
                tree = ET.parse(xml_path)
                root = tree.getroot()

            # Extract slash counts by onset time
            for note_elem in root.findall('.//note'):
                tremolo = note_elem.find('.//ornaments/tremolo')
                if tremolo is not None and tremolo.get('type') in ['start', 'stop']:
                    slashes = int(tremolo.text) if tremolo.text else 2
                    # Get pitch and duration to match with Partitura note
                    pitch_elem = note_elem.find('.//pitch')
                    if pitch_elem is not None:
                        step = pitch_elem.find('step').text
                        octave = int(pitch_elem.find('octave').text)
                        alter = pitch_elem.find('alter')
                        alter_val = int(alter.text) if alter is not None else 0

                        # Convert to MIDI pitch
                        pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                        midi_pitch = pitch_map[step] + alter_val + (octave + 1) * 12

                        # Store by pitch (we'll match later)
                        slash_counts[midi_pitch] = slashes

        except Exception as e:
            logger.warning(f"Failed to parse tremolo slashes from XML: {e}")
            # Default to 2 slashes
            for idx in tremolo_indices:
                slash_counts[notes[idx].midi_pitch] = 2

        # Pair consecutive tremolo notes
        i = 0
        while i < len(tremolo_indices) - 1:
            idx1 = tremolo_indices[i]
            idx2 = tremolo_indices[i + 1]

            # Check if they're close enough to be a pair (within 5 notes)
            if idx2 - idx1 <= 5:
                note1 = notes[idx1]
                note2 = notes[idx2]

                # Get slash count (default 2)
                slashes = slash_counts.get(note1.midi_pitch, 2)

                # Create pair
                tremolo_pairs[idx1] = {
                    'partner_idx': idx2,
                    'slashes': slashes,
                    'is_start': True
                }
                tremolo_pairs[idx2] = {
                    'partner_idx': idx1,
                    'slashes': slashes,
                    'is_start': False
                }

                logger.debug(f"Tremolo pair: {idx1} ({note1.midi_pitch}) - {idx2} ({note2.midi_pitch}), slashes={slashes}")

                # Skip both notes
                i += 2
            else:
                # Isolated tremolo note, skip it
                logger.warning(f"Isolated tremolo note at index {idx1}, skipping")
                i += 1

        return tremolo_pairs
