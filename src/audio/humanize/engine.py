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
from .rules.punctuation import PunctuationRule
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

        # Articulation with velocity effects
        tenuto_rule = TenutoRule(cfg.tenuto, cfg.tenuto_extension_ratio)
        accent_rule = AccentRule(cfg.accent, cfg.accent_velocity_boost_dB, cfg.accent_delay_ms)
        marcato_rule = MarcatoRule(cfg.marcato, cfg.marcato_velocity_boost_dB, cfg.marcato_shortening_ratio)

        crescendo_tempo_rule = CrescendoTempoRule(
            cfg.crescendo_tempo,
            cfg.crescendo_tempo_max_change,
            cfg.crescendo_velocity_max_change_dB
        )

        self.velocity_rules = [
            HighLoudRule(cfg.high_loud),
            PhraseArchRule(cfg.phrase_arch, cfg.phrase_peak_position),
            DurationContrastRule(cfg.duration_contrast),
            MelodicChargeRule(cfg.melodic_charge, cfg.nct_boost_dB),
            crescendo_tempo_rule,  # Crescendo: velocity coupling
            tenuto_rule,           # Tenuto: +1dB
            accent_rule,           # Accent: +1.5dB
            marcato_rule,          # Marcato: +5dB
        ]

        # Timing rules
        beat_jitter = BeatJitterRule(cfg.beat_jitter)
        beat_jitter.set_rng(self.rng)

        self.fermata_rule = FermataRule(cfg.fermata, cfg.fermata_duration_multiplier, cfg.fermata_pause_beats)
        self.timing_rules = [
            PhraseRubatoRule(cfg.phrase_rubato, cfg.phrase_peak_position),
            beat_jitter,
            FinalRitardRule(cfg.final_ritard, cfg.final_ritard_measures),
            self.fermata_rule,
            crescendo_tempo_rule,  # Crescendo: timing coupling
            PunctuationRule(cfg.punctuation, cfg.micropause_ms, cfg.phrase_end_shorten_ratio),
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

        # 2.5. Get notes list (filter out grace notes - they will be handled separately)
        import partitura as pt
        all_notes = part.notes_tied
        # Filter out GraceNote objects (they have duration=0 and should not go through normal pipeline)
        notes = [n for n in all_notes if not isinstance(n, pt.score.GraceNote)]
        logger.info(f"Filtered {len(all_notes)} notes -> {len(notes)} main notes ({len(all_notes) - len(notes)} grace notes excluded)")

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

        # 2.9. Detect slurs (legato) from Partitura note attributes
        slur_map = {}
        active_slurs = set()
        for i, note in enumerate(notes):
            # Add new slurs starting at this note
            if hasattr(note, 'slur_starts') and note.slur_starts:
                for slur in note.slur_starts:
                    active_slurs.add(id(slur))

            # Mark this note as in_slur if any slur is active
            if active_slurs:
                slur_map[i] = True

            # Remove slurs ending at this note
            if hasattr(note, 'slur_stops') and note.slur_stops:
                for slur in note.slur_stops:
                    active_slurs.discard(id(slur))

        slur_count = sum(1 for in_slur in slur_map.values() if in_slur)
        logger.info(f"Found {slur_count} notes in slurs")

        # 2.9.5. Detect phrase boundaries from slurs and gaps
        # Collect boundary ONSET TIMES first, then mark ALL notes at those times.
        # This ensures both hands breathe together (not just the slur voice).
        phrase_start_onsets = set()
        phrase_end_onsets = set()

        # From slurs
        for i, note in enumerate(notes):
            if hasattr(note, 'slur_starts') and note.slur_starts:
                phrase_start_onsets.add(note.start.t)
            if hasattr(note, 'slur_stops') and note.slur_stops:
                phrase_end_onsets.add(note.start.t)

        # From gaps
        for i in range(len(notes) - 1):
            this_end = notes[i].start.t + notes[i].duration_tied
            next_onset = notes[i + 1].start.t
            if next_onset > this_end:
                phrase_end_onsets.add(notes[i].start.t)
                phrase_start_onsets.add(next_onset)

        # Mark ALL notes at boundary onsets (both hands)
        phrase_start_set = set()
        phrase_end_set = set()
        for i, note in enumerate(notes):
            if note.start.t in phrase_start_onsets:
                phrase_start_set.add(i)
            if note.start.t in phrase_end_onsets:
                phrase_end_set.add(i)

        # Assign phrase numbers
        phrase_number_map = {}
        current_phrase = 0
        for i in range(len(notes)):
            if i in phrase_start_set and i > 0:
                current_phrase += 1
            phrase_number_map[i] = current_phrase

        logger.info(f"Found {len(phrase_start_set)} phrase starts, {len(phrase_end_set)} phrase ends "
                     f"(from {len(phrase_start_onsets)} boundary onsets)")

        # 2.10. Detect trills and mordents from Partitura ornaments
        trill_notes = set()
        mordent_notes = {}  # note_index -> mordent_type ('upper' or 'lower')
        for i, note in enumerate(notes):
            if hasattr(note, 'ornaments') and note.ornaments:
                if 'trill-mark' in note.ornaments:
                    trill_notes.add(i)
                if 'mordent' in note.ornaments:
                    mordent_notes[i] = 'upper'  # Default to upper mordent
                if 'inverted-mordent' in note.ornaments:
                    mordent_notes[i] = 'lower'

        logger.info(f"Found {len(trill_notes)} trills, {len(mordent_notes)} mordents")

        # 2.10.5. Detect fermatas from MusicXML
        fermata_indices = set()
        if format == 'musicxml':
            fermata_indices = self._parse_fermatas_from_xml(Path(score_path))

        # 2.11. Detect grace notes and pair with main notes
        grace_note_pairs = {}  # main_note_index -> list of grace notes
        if format == 'musicxml':
            import partitura as pt
            grace_notes = list(part.iter_all(pt.score.GraceNote))

            if grace_notes:
                # Build onset -> note_index mapping for main notes
                onset_to_index = {}
                for i, note in enumerate(notes):
                    onset = note.start.t
                    if onset not in onset_to_index:
                        onset_to_index[onset] = []
                    onset_to_index[onset].append(i)

                # Pair grace notes with main notes (same onset)
                for grace_note in grace_notes:
                    grace_onset = grace_note.start.t
                    if grace_onset in onset_to_index:
                        # Find the first main note at this onset
                        main_indices = onset_to_index[grace_onset]
                        for main_idx in main_indices:
                            if main_idx not in grace_note_pairs:
                                grace_note_pairs[main_idx] = []
                            grace_note_pairs[main_idx].append(grace_note)
                            logger.debug(f"Paired grace note {grace_note.midi_pitch} with main note idx={main_idx} (pitch {notes[main_idx].midi_pitch})")

                logger.info(f"Found {len(grace_notes)} grace notes paired with {len(grace_note_pairs)} main notes")

        # 3. Extract features
        logger.info("Extracting features from score")
        phrase_info = {
            'starts': phrase_start_set,
            'ends': phrase_end_set,
            'numbers': phrase_number_map,
        }
        features_list = self._extract_features(part, base_bpm, tremolo_pairs, articulation_map, slur_map,
                                               trill_notes, mordent_notes, grace_note_pairs, fermata_indices,
                                               phrase_info)

        # 4. Apply all rules
        logger.info("Applying humanization rules")
        self._apply_rules_notes = notes  # Store for _build_pedal_features
        humanized_notes = self._apply_rules(notes, features_list)

        # 5. Generate pedal events using HUMANIZED onsets
        # Timing rules (rubato, ritard, fermata, etc.) shift note onsets.
        # Pedal must use the same shifted onsets so lift/press aligns with
        # the actual sounding notes, not the original score positions.
        score_pedals = getattr(self, '_directions', {}).get('pedals', [])
        pedal_features = self._build_pedal_features(humanized_notes, features_list)
        # Score pedal markings are in original time; shift to humanized time
        shifted_pedals = self._shift_score_pedals(score_pedals, features_list, pedal_features)
        pedal_events = self.pedal_rule.generate_pedal_events(notes, pedal_features, shifted_pedals)

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
                         articulation_map: Dict[int, List[str]] = None,
                         slur_map: Dict[int, bool] = None,
                         trill_notes: set = None,
                         mordent_notes: Dict[int, str] = None,
                         grace_note_pairs: Dict[int, List] = None,
                         fermata_indices: set = None,
                         phrase_info: Dict = None) -> List[Dict[str, Any]]:
        """
        Extract per-note features using partitura.

        Returns:
            List of feature dicts (one per note)
        """
        if tremolo_pairs is None:
            tremolo_pairs = {}
        if articulation_map is None:
            articulation_map = {}
        if slur_map is None:
            slur_map = {}
        if trill_notes is None:
            trill_notes = set()
        if mordent_notes is None:
            mordent_notes = {}
        if grace_note_pairs is None:
            grace_note_pairs = {}
        if fermata_indices is None:
            fermata_indices = set()
        if phrase_info is None:
            phrase_info = {'starts': set(), 'ends': set(), 'numbers': {}}
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
        self._directions = directions  # Store for pedal rule access

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

        # Extract basis features from Partitura (pitch, loudness, tempo, etc.)
        # These provide dynamics context for _get_base_velocity().
        # If extraction fails, we proceed without them (all basis values = 0).
        try:
            logger.info(f"Extracting Partitura features: {feature_functions}")
            basis_matrix, basis_names = ma.make_note_feats(part, feature_functions)
            logger.info(f"Successfully extracted {len(basis_names)} features")
        except Exception as e:
            logger.warning(f"Failed to extract basis functions: {e}, proceeding without them")
            basis_matrix = np.zeros((len(notes), 0))
            basis_names = []

        # Convert to per-note dicts
        features_list = []
        notes = part.notes_tied
        beat_duration = 60.0 / base_bpm

        # Get beats_per_measure from time signature
        import partitura as pt
        try:
            ts = list(part.iter_all(pt.score.TimeSignature))[0]
            if ts.beat_type == 8:
                beats_per_measure = ts.beats / 2.0  # 6/8->3, 12/8->6, 9/8->4.5
            else:
                beats_per_measure = ts.beats * (4.0 / ts.beat_type)
        except (IndexError, AttributeError):
            beats_per_measure = 4.0

        # Pre-compute phrase spans (phrase_number -> (start_onset_sec, end_onset_sec))
        # so _compute_phrase_position can look up the time range for each phrase
        self._phrase_spans = {}
        for i, note in enumerate(notes):
            ph_num = phrase_info['numbers'].get(i, 0)
            onset_sec = note.start.t * seconds_per_unit
            end_sec = onset_sec + note.duration_tied * seconds_per_unit
            if ph_num not in self._phrase_spans:
                self._phrase_spans[ph_num] = [onset_sec, end_sec]
            else:
                self._phrase_spans[ph_num][0] = min(self._phrase_spans[ph_num][0], onset_sec)
                self._phrase_spans[ph_num][1] = max(self._phrase_spans[ph_num][1], end_sec)
        # Convert to tuples
        self._phrase_spans = {k: tuple(v) for k, v in self._phrase_spans.items()}
        logger.info(f"Pre-computed {len(self._phrase_spans)} phrase spans for rubato")

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
                'beats_per_measure': beats_per_measure,
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
                features['tremolo_chord_pitches'] = tremolo_pairs[i].get('chord_pitches', None)
                features['tremolo_is_chord_member'] = tremolo_pairs[i].get('is_chord_member', False)
            else:
                features['has_tremolo'] = False
                features['tremolo_partner_idx'] = None
                features['tremolo_slashes'] = 0
                features['tremolo_is_start'] = False
                features['tremolo_chord_pitches'] = None
                features['tremolo_is_chord_member'] = False

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

            # Add slur feature (for legato)
            features['in_slur'] = slur_map.get(i, False)

            # Add ornament features (trill, mordent, grace notes)
            features['has_trill'] = i in trill_notes
            features['has_mordent'] = i in mordent_notes
            features['mordent_type'] = mordent_notes.get(i, 'upper')  # 'upper' or 'lower'

            # Grace notes: check if this note has associated grace notes
            if i in grace_note_pairs:
                grace_list = grace_note_pairs[i]
                features['has_grace_notes'] = True
                features['grace_notes'] = grace_list  # List of GraceNote objects
                # Determine grace type from first grace note
                if grace_list and hasattr(grace_list[0], 'grace_type'):
                    features['is_acciaccatura'] = grace_list[0].grace_type == 'acciaccatura'
                else:
                    features['is_acciaccatura'] = True  # Default to acciaccatura
            else:
                features['has_grace_notes'] = False
                features['grace_notes'] = []
                features['is_acciaccatura'] = True

            # Phrase boundaries (for PunctuationRule)
            features['is_phrase_start'] = i in phrase_info['starts']
            features['is_phrase_end'] = i in phrase_info['ends']
            features['phrase_number'] = phrase_info['numbers'].get(i, 0)

            # Fermata: check if this note index has fermata
            # Use the fermata_indices set parsed from MusicXML
            features['has_fermata'] = i in fermata_indices

            features_list.append(features)

        # Second pass: mark notes after fermata
        for i in range(len(features_list)):
            if i > 0 and features_list[i-1]['has_fermata']:
                features_list[i]['after_fermata'] = True
            else:
                features_list[i]['after_fermata'] = False

        return features_list

    def _compute_phrase_position(self, note_idx: int, features: Dict, notes: List) -> float:
        """Compute position within current phrase (0-1) using pre-computed phrase spans."""
        if not hasattr(self, '_phrase_spans') or not self._phrase_spans:
            return 0.5

        phrase_number = features.get('phrase_number', 0)
        if phrase_number in self._phrase_spans:
            start_onset, end_onset = self._phrase_spans[phrase_number]
            phrase_duration = end_onset - start_onset

            # Skip rubato for very short phrases (< 2 seconds)
            # Short fragments don't have meaningful phrase shape
            if phrase_duration < 2.0:
                return 0.5

            current_onset = features.get('onset', 0)
            pos = (current_onset - start_onset) / phrase_duration
            return max(0.0, min(1.0, pos))

        return 0.5

    def _parse_fermatas_from_xml(self, xml_path: Path) -> set:
        """
        Parse fermata markings directly from MusicXML file.

        Returns set of note indices (0-indexed) that have fermatas.
        Supports both .musicxml (plain XML) and .mxl (compressed) formats.
        """
        try:
            import xml.etree.ElementTree as ET
            import zipfile

            # Handle .mxl (compressed MusicXML)
            if str(xml_path).endswith('.mxl'):
                with zipfile.ZipFile(xml_path, 'r') as z:
                    # Find the main XML file inside the archive
                    xml_files = [f for f in z.namelist()
                                 if f.endswith('.xml') and not f.startswith('META-INF')]
                    if not xml_files:
                        logger.warning("No XML file found inside .mxl archive")
                        return set()
                    with z.open(xml_files[0]) as f:
                        tree = ET.parse(f)
            else:
                tree = ET.parse(xml_path)
            root = tree.getroot()

            logger.info(f"Parsing fermatas from {xml_path}")

            # Find all note elements
            fermata_indices = set()
            note_idx = 0
            total_notes = 0

            for note in root.findall('.//note'):
                total_notes += 1
                # Skip grace notes and rests
                if note.find('grace') is not None or note.find('rest') is not None:
                    logger.debug(f"Skipping grace/rest note")
                    continue

                # Check for fermata in notations
                notations = note.find('notations')
                if notations is not None:
                    fermata = notations.find('fermata')
                    if fermata is not None:
                        fermata_indices.add(note_idx)
                        logger.info(f"Found fermata on note {note_idx}")
                    else:
                        logger.debug(f"Note {note_idx} has notations but no fermata")
                else:
                    logger.debug(f"Note {note_idx} has no notations")

                note_idx += 1

            logger.info(f"Parsed {total_notes} total notes, {note_idx} main notes, found {len(fermata_indices)} fermatas")
            return fermata_indices
        except Exception as e:
            logger.warning(f"Failed to parse fermatas from XML: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return set()

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

            # Timing rules can also affect duration
            for rule in self.timing_rules:
                duration_multiplier *= rule.apply_duration(note, features)

            final_duration = base_duration * duration_multiplier

            # Safety: ensure minimum audible duration (post-processing)
            # Must run AFTER all other duration rules so it catches notes
            # that became too short due to staccato or other articulations
            if self.social_care.enabled and self.social_care.k > 0:
                min_dur_sec = self.config.min_audible_duration_ms / 1000
                if final_duration < min_dur_sec:
                    correction = min_dur_sec - final_duration
                    final_duration += self.social_care.k * correction

            note_dict = {
                'pitch': note.midi_pitch,
                'onset': max(0, final_onset),  # Ensure non-negative
                'duration': max(0.01, final_duration),  # Ensure positive
                'velocity': int(np.clip(final_velocity, 1, 127)),
                'original_note': note,
                '_has_fermata': features.get('has_fermata', False),
                '_fermata_extra_duration': max(0, final_duration - base_duration) if features.get('has_fermata', False) else 0,
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
                        chord_pitches = features.get('tremolo_chord_pitches', None)
                        for tn in tremolo_notes:
                            # Apply velocity delta from tremolo rule
                            velocity_with_variation = dB_to_velocity(
                                velocity_to_dB(tn['velocity'], self.config.reference_velocity) + tn.get('velocity_delta_dB', 0),
                                self.config.reference_velocity
                            )
                            vel_final = int(np.clip(velocity_with_variation, 1, 127))
                            if chord_pitches:
                                # Chord tremolo: each hit plays all chord pitches
                                for cp in chord_pitches:
                                    humanized.append({
                                        'pitch': cp,
                                        'onset': tn['onset'],
                                        'duration': tn['duration'],
                                        'velocity': vel_final,
                                        'original_note': note
                                    })
                            else:
                                humanized.append({
                                    'pitch': tn['pitch'],
                                    'onset': tn['onset'],
                                    'duration': tn['duration'],
                                    'velocity': vel_final,
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
                # Not a tremolo note, check for other ornaments

                # Check for trill
                if features.get('has_trill', False) and self.trill_rule.enabled and self.trill_rule.k > 0:
                    # Create a note dict for expansion
                    note_for_expansion = {
                        'midi_pitch': note.midi_pitch,
                        'onset': note_dict['onset'],
                        'duration': note_dict['duration'],
                        'velocity': note_dict['velocity'],
                    }
                    # Temporarily store these in a fake object
                    class FakeNote:
                        def __init__(self, d):
                            self.midi_pitch = d['midi_pitch']
                            self.onset = d['onset']
                            self.duration = d['duration']
                            self.velocity = d['velocity']

                    fake_note = FakeNote(note_for_expansion)
                    trill_notes = self.trill_rule.expand_trill(fake_note, features)
                    for tn in trill_notes:
                        humanized.append({
                            'pitch': tn['pitch'],
                            'onset': tn['onset'],  # Already includes cumulative_drift from note_dict
                            'duration': tn['duration'],
                            'velocity': tn['velocity'],
                        })
                    logger.debug(f"Expanded trill: {len(trill_notes)} notes from pitch {note.midi_pitch}")

                # Check for mordent
                elif features.get('has_mordent', False) and self.mordent_rule.enabled and self.mordent_rule.k > 0:
                    # Create a note dict for expansion
                    class FakeNote:
                        def __init__(self, d):
                            self.midi_pitch = d['midi_pitch']
                            self.onset = d['onset']
                            self.duration = d['duration']
                            self.velocity = d['velocity']

                    note_for_expansion = {
                        'midi_pitch': note.midi_pitch,
                        'onset': note_dict['onset'],
                        'duration': note_dict['duration'],
                        'velocity': note_dict['velocity'],
                    }
                    fake_note = FakeNote(note_for_expansion)
                    mordent_notes = self.mordent_rule.expand_mordent(fake_note, features)
                    for mn in mordent_notes:
                        humanized.append({
                            'pitch': mn['pitch'],
                            'onset': mn['onset'],  # Already includes cumulative_drift from note_dict
                            'duration': mn['duration'],
                            'velocity': mn['velocity'],
                        })
                    logger.debug(f"Expanded mordent: {len(mordent_notes)} notes from pitch {note.midi_pitch}")

                # Check for grace notes
                elif features.get('has_grace_notes', False) and self.grace_note_rule.enabled and self.grace_note_rule.k > 0:
                    grace_list = features.get('grace_notes', [])
                    is_acciaccatura = features.get('is_acciaccatura', True)

                    # Calculate grace note duration from symbolic_duration
                    # Grace notes should be VERY SHORT (30-80ms), but relative lengths matter
                    # e.g., eighth grace note should be slightly longer than 16th

                    # Base duration for 16th grace note (in seconds)
                    base_grace_ms = 50.0  # 50ms for a typical 16th grace note

                    # Relative multipliers based on symbolic duration
                    # These maintain relative differences while keeping all grace notes short
                    type_to_multiplier = {
                        'whole': 2.0,      # 100ms (rare, but very deliberate)
                        'half': 1.8,       # 90ms
                        'quarter': 1.5,    # 75ms
                        'eighth': 1.2,     # 60ms (most common for acciaccatura)
                        '16th': 1.0,       # 50ms (baseline)
                        '32nd': 0.7,       # 35ms (very quick)
                        '64th': 0.5,       # 25ms (extremely quick)
                    }

                    # Get grace note duration based on symbolic_duration
                    grace_duration_ms = base_grace_ms
                    if grace_list and hasattr(grace_list[0], 'symbolic_duration') and grace_list[0].symbolic_duration:
                        sym_type = grace_list[0].symbolic_duration.get('type', '16th')
                        multiplier = type_to_multiplier.get(sym_type, 1.0)
                        grace_duration_ms = base_grace_ms * multiplier
                        logger.debug(f"Grace note symbolic_duration={sym_type}, multiplier={multiplier:.1f}x, duration={grace_duration_ms:.1f}ms")
                    else:
                        logger.debug(f"Grace note using default duration={grace_duration_ms:.1f}ms")

                    # Convert to seconds and apply k scaling
                    grace_duration = (grace_duration_ms / 1000.0) * self.grace_note_rule.k

                    if is_acciaccatura:
                        # Acciaccatura: grace note before the beat
                        # The pianist cannot play two notes with one finger,
                        # so main note MUST start after grace note finishes

                        # Random gap before grace note starts
                        # Maximum = 16th note duration (tempo-dependent)
                        beat_duration = features.get('beat_duration', 0.5)
                        sixteenth_duration = beat_duration / 4  # 16th note = 1/4 beat
                        max_gap = sixteenth_duration * self.grace_note_rule.k
                        random_gap = self.rng.uniform(0.0, max_gap)

                        # Calculate grace note onset: before the main note's beat
                        # grace_onset = main_onset - grace_duration - random_gap
                        ideal_grace_onset = note_dict['onset'] - grace_duration - random_gap

                        # Handle negative onset (grace note at beginning of piece)
                        if ideal_grace_onset < 0:
                            # Grace note starts at 0, main note shifts forward
                            grace_onset = 0.0
                            # Main note starts after grace note finishes
                            note_dict['onset'] = grace_duration
                            logger.debug(f"Acciaccatura at beginning: grace at 0, main delayed to {note_dict['onset']:.4f}s")
                        else:
                            # Normal case: grace note before the beat
                            grace_onset = ideal_grace_onset
                            # Main note starts right after grace note (eating into the beat)
                            original_onset = note_dict['onset']
                            note_dict['onset'] = grace_onset + grace_duration
                            logger.debug(f"Acciaccatura: grace at {grace_onset:.4f}s, main delayed from {original_onset:.4f}s to {note_dict['onset']:.4f}s")

                        # Add grace note(s)
                        for gn in grace_list:
                            humanized.append({
                                'pitch': gn.midi_pitch,
                                'onset': grace_onset,
                                'duration': grace_duration,
                                'velocity': max(30, note_dict['velocity'] - 15),  # Slightly softer
                            })

                        # Add main note (always delayed to avoid overlap)
                        humanized.append(note_dict)
                        logger.debug(f"Added acciaccatura: grace_dur={grace_duration*1000:.1f}ms, random_gap={random_gap*1000:.1f}ms")

                    else:
                        # Appoggiatura: grace note can start on or before the beat
                        # Like acciaccatura, but typically slightly longer
                        # Can also have randomness in timing (how close to the beat)

                        appoggiatura_duration = grace_duration  # Already scaled by k

                        # Random gap: appoggiatura can also start before the beat
                        # "grace note 可以接受在正拍前出現或是正拍上"
                        # Maximum = 16th note duration (tempo-dependent)
                        beat_duration = features.get('beat_duration', 0.5)
                        sixteenth_duration = beat_duration / 4
                        max_early = sixteenth_duration * self.grace_note_rule.k
                        early_offset = self.rng.uniform(0.0, max_early)

                        # Grace note starts before or on the beat
                        grace_onset = note_dict['onset'] - early_offset

                        # Handle negative onset
                        if grace_onset < 0:
                            grace_onset = 0.0
                            # Main note starts after grace note
                            note_dict['onset'] = appoggiatura_duration
                        else:
                            # Main note starts after grace note finishes
                            # This "eats into" the main note's time
                            note_dict['onset'] = grace_onset + appoggiatura_duration

                        # Add grace note(s)
                        for gn in grace_list:
                            humanized.append({
                                'pitch': gn.midi_pitch,
                                'onset': grace_onset,
                                'duration': appoggiatura_duration,
                                'velocity': note_dict['velocity'],
                            })

                        # Reduce main note duration (since it starts later)
                        original_end = note_dict['onset'] + note_dict['duration']
                        note_dict['duration'] = max(0.01, original_end - note_dict['onset'])

                        # Add main note (shifted)
                        humanized.append(note_dict)
                        logger.debug(f"Added appoggiatura: grace_dur={appoggiatura_duration*1000:.1f}ms, early={early_offset*1000:.1f}ms, main_onset={note_dict['onset']:.4f}s")

                    logger.debug(f"Added {len(grace_list)} grace notes before pitch {note.midi_pitch}")

                else:
                    # No ornaments, add normally
                    humanized.append(note_dict)

        # Propagate fermata time shifts: when a fermata extends a note's duration,
        # all subsequent notes must be pushed later in time
        cumulative_fermata_shift = 0.0
        beat_duration = features_list[0].get('beat_duration', 0.5) if features_list else 0.5
        for note_dict in humanized:
            # Push this note by accumulated fermata shift
            note_dict['onset'] += cumulative_fermata_shift

            # If this note has fermata, accumulate shift for subsequent notes
            if note_dict.get('_has_fermata', False):
                extra_duration = note_dict.get('_fermata_extra_duration', 0)
                pause = self.fermata_rule.k * self.fermata_rule.pause_beats * beat_duration
                cumulative_fermata_shift += extra_duration + pause
                logger.debug(f"Fermata shift: +{extra_duration:.3f}s duration + {pause:.3f}s pause = {cumulative_fermata_shift:.3f}s total")

        # Apply global normalization to velocities
        if self.normalizer is not None:
            velocities = np.array([n['velocity'] for n in humanized])
            normalized = self.normalizer.normalize(velocities)
            for i, vel in enumerate(normalized):
                humanized[i]['velocity'] = vel

        return humanized

    def _shift_score_pedals(
        self,
        score_pedals: List[Dict],
        original_features: List[Dict[str, Any]],
        pedal_features: List[Dict[str, Any]]
    ) -> List[Dict]:
        """
        Shift score pedal markings from original time to humanized time.

        Builds a time mapping from original -> humanized onsets,
        then interpolates pedal start/end times.
        """
        if not score_pedals or not original_features:
            return score_pedals

        # Build sorted time mapping: (original_onset, humanized_onset)
        time_pairs = []
        for orig_f, ped_f in zip(original_features, pedal_features):
            orig_t = orig_f.get('onset', 0.0)
            hum_t = ped_f.get('onset', orig_t)
            time_pairs.append((orig_t, hum_t))

        # Deduplicate and sort by original time
        seen = set()
        unique_pairs = []
        for orig_t, hum_t in sorted(time_pairs):
            # Use first occurrence for each original onset
            key = round(orig_t, 6)
            if key not in seen:
                seen.add(key)
                unique_pairs.append((orig_t, hum_t))

        if not unique_pairs:
            return score_pedals

        def interpolate_time(orig_time: float) -> float:
            """Map original time to humanized time via linear interpolation."""
            if orig_time <= unique_pairs[0][0]:
                # Before first note: apply same shift as first note
                shift = unique_pairs[0][1] - unique_pairs[0][0]
                return orig_time + shift
            if orig_time >= unique_pairs[-1][0]:
                # After last note: apply same shift as last note
                shift = unique_pairs[-1][1] - unique_pairs[-1][0]
                return orig_time + shift

            # Binary search for surrounding points
            import bisect
            idx = bisect.bisect_right([p[0] for p in unique_pairs], orig_time)
            if idx == 0:
                idx = 1
            p0 = unique_pairs[idx - 1]
            p1 = unique_pairs[idx] if idx < len(unique_pairs) else unique_pairs[-1]

            if abs(p1[0] - p0[0]) < 1e-9:
                return p0[1]

            # Linear interpolation
            t = (orig_time - p0[0]) / (p1[0] - p0[0])
            return p0[1] + t * (p1[1] - p0[1])

        shifted = []
        for pedal in score_pedals:
            new_start = interpolate_time(pedal['start'])
            new_end = interpolate_time(pedal['end'])
            # Safety: ensure end > start (extrapolation can cause inversion)
            if new_end <= new_start:
                original_duration = pedal['end'] - pedal['start']
                new_end = new_start + max(original_duration, 0.05)
            shifted.append({
                **pedal,
                'start': new_start,
                'end': new_end,
            })

        return shifted

    def _build_pedal_features(
        self,
        humanized_notes: List[Dict[str, Any]],
        original_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build features list with humanized onsets for pedal generation.

        Timing rules shift note onsets (rubato, ritard, fermata, etc.).
        Pedal events must use the shifted onsets so that lift/press aligns
        with actual sounding notes. This method copies each feature dict
        and replaces its onset with the corresponding humanized onset.

        humanized_notes may contain extra entries (grace notes, tremolo
        expansion), so we map back via the 'original_note' reference.
        """
        # Build onset+velocity lookup: original note id -> (onset, velocity)
        note_onset_map = {}
        note_velocity_map = {}
        for h_note in humanized_notes:
            orig = h_note.get('original_note')
            if orig is not None:
                key = id(orig)
                # Keep earliest onset (grace notes come before main note)
                if key not in note_onset_map or h_note['onset'] < note_onset_map[key]:
                    note_onset_map[key] = h_note['onset']
                # Keep main note velocity (highest velocity for this original)
                vel = h_note.get('velocity', 64)
                if key not in note_velocity_map or vel > note_velocity_map[key]:
                    note_velocity_map[key] = vel

        # Copy features with updated onsets and humanized velocities
        pedal_features = []
        for i, features in enumerate(original_features):
            f_copy = dict(features)
            # Match via note_idx -> notes list -> original_note id
            note_idx = features.get('note_idx', i)
            if hasattr(self, '_apply_rules_notes') and note_idx < len(self._apply_rules_notes):
                orig_note = self._apply_rules_notes[note_idx]
                key = id(orig_note)
                if key in note_onset_map:
                    f_copy['onset'] = note_onset_map[key]
                if key in note_velocity_map:
                    f_copy['humanized_velocity'] = note_velocity_map[key]
            pedal_features.append(f_copy)

        return pedal_features

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
        single_tremolo_pitches = set()
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

            # Extract slash counts and single-note tremolo by onset time
            single_tremolo_pitches = set()  # MIDI pitches with single-note tremolo
            for note_elem in root.findall('.//note'):
                tremolo = note_elem.find('.//ornaments/tremolo')
                if tremolo is not None:
                    trem_type = tremolo.get('type', '')
                    slashes = int(tremolo.text) if tremolo.text else 2
                    pitch_elem = note_elem.find('.//pitch')
                    if pitch_elem is not None:
                        step = pitch_elem.find('step').text
                        octave = int(pitch_elem.find('octave').text)
                        alter = pitch_elem.find('alter')
                        alter_val = int(alter.text) if alter is not None else 0

                        pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                        midi_pitch = pitch_map[step] + alter_val + (octave + 1) * 12

                        if trem_type in ['start', 'stop']:
                            slash_counts[midi_pitch] = slashes
                        elif trem_type == 'single':
                            slash_counts[midi_pitch] = slashes
                            single_tremolo_pitches.add(midi_pitch)

        except Exception as e:
            logger.warning(f"Failed to parse tremolo slashes from XML: {e}")
            # Default to 2 slashes
            for idx in tremolo_indices:
                slash_counts[notes[idx].midi_pitch] = 2

        # 1. Handle single-note tremolo first (remove from pairing candidates)
        pair_candidates = []
        for idx in tremolo_indices:
            note = notes[idx]
            if note.midi_pitch in single_tremolo_pitches:
                slashes = slash_counts.get(note.midi_pitch, 2)
                tremolo_pairs[idx] = {
                    'partner_idx': idx,  # Points to itself
                    'slashes': slashes,
                    'is_start': True,
                    'is_single': True
                }
                logger.debug(f"Single-note tremolo at {idx} ({note.midi_pitch}), slashes={slashes}")
            else:
                pair_candidates.append(idx)

        # 1.5. Find chord members for single-note tremolos
        # When a chord note has tremolo, all notes at the same onset
        # (chord members) should participate in the tremolo.
        onset_to_indices = {}
        for idx_n, n in enumerate(notes):
            onset = n.start.t
            if onset not in onset_to_indices:
                onset_to_indices[onset] = []
            onset_to_indices[onset].append(idx_n)

        for idx in list(tremolo_pairs.keys()):
            info = tremolo_pairs[idx]
            if not info.get('is_single', False) or not info.get('is_start', False):
                continue
            note = notes[idx]
            chord_members = [ci for ci in onset_to_indices.get(note.start.t, []) if ci != idx]
            if chord_members:
                chord_pitches = [note.midi_pitch] + [notes[ci].midi_pitch for ci in chord_members]
                info['chord_pitches'] = chord_pitches
                # Mark chord members as handled (skip in normal processing)
                for ci in chord_members:
                    if ci not in tremolo_pairs:
                        tremolo_pairs[ci] = {
                            'partner_idx': idx,
                            'slashes': info['slashes'],
                            'is_start': False,
                            'is_chord_member': True,
                            'is_single': True,
                        }
                logger.debug(f"Chord tremolo at {idx}: pitches={chord_pitches}")

        # 2. Pair remaining consecutive tremolo notes (two-note tremolo)
        i = 0
        while i < len(pair_candidates) - 1:
            idx1 = pair_candidates[i]
            idx2 = pair_candidates[i + 1]

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
                # Isolated two-note tremolo note (no partner found)
                logger.warning(f"Isolated tremolo note at index {idx1}, skipping")
                i += 1

        return tremolo_pairs
