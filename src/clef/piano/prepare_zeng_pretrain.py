"""
clef-piano-base Preprocessing Pipeline
=======================================

Prepares training data for clef-piano-base model:
- Phase 1: Score → Kern (MuseSyn XML + HumSyn kern)
- Phase 2: Kern → MIDI → Audio (with data augmentation)
- Phase 2.5: Audio → Mel Spectrogram
- Phase 3: Create training manifests (train/valid/test)

Key differences from Zeng et al.:
- converter21 instead of verovio (higher success rate)
- Full-song instead of 5-bar chunks
- Preserves grace notes
- Per-song augmentation instead of per-chunk

Data augmentation matches Zeng et al.:
- Key-aware transpose (feasible_transposes)
- Tempo scaling (0.85~1.15x via MIDIProcess)
- Multi-soundfont synthesis
- Loudness normalization (-15 LUFS)

Usage:
    python -m src.clef.piano.prepare_zeng_pretrain --phase 1      # Score → Kern
    python -m src.clef.piano.prepare_zeng_pretrain --phase 2      # Kern → Audio
    python -m src.clef.piano.prepare_zeng_pretrain --phase 2.5    # Audio → Mel
    python -m src.clef.piano.prepare_zeng_pretrain --phase 3      # Create manifests
    python -m src.clef.piano.prepare_zeng_pretrain --phase 2 --workers 8  # Parallel
    python -m src.clef.piano.prepare_zeng_pretrain                # Full pipeline (1,2)
    python -m src.clef.piano.prepare_zeng_pretrain --phase all    # All phases (1,2,2.5,3)
"""

import hashlib
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import converter21
import music21 as m21
import pandas as pd
from tqdm import tqdm

from src.audio.mel import load_mel_config, process_audio_file
from src.clef.piano.generate_zeng_augmentation_metadata import generate_metadata
from src.datasets.syn.syn_manifest import create_manifest
from src.preprocessing.humsyn_processor import HumSynProcessor
from src.preprocessing.musesyn_processor import MuseSynProcessor
from src.score.clean_kern import clean_kern_sequence
from src.score.kern_zeng_compat import (
    expand_tuplets_to_zeng_vocab,
    resolve_rscale_regions,
    quantize_oov_tuplets,
    quantize_tuplet_ratios,
    apply_zeng_pitch_compat,
)
from src.score.sanitize_kern import fix_kern_spine_timing
from src.utils import set_seed, SEED_DATA_AUGMENTATION

# Register converter21 for robust humdrum parsing
converter21.register()

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HUMSYN_DIR = Path("data/datasets/HumSyn")
DEFAULT_MUSESYN_DIR = Path("data/datasets/MuseSyn")
DEFAULT_OUTPUT_DIR = Path("data/experiments/clef_piano_base")
DEFAULT_METADATA_DIR = Path("src/datasets/syn")
DEFAULT_SOUNDFONT_DIR = Path("data/soundfonts/piano")
DEFAULT_AUG_CONFIG = Path("configs/zeng_augmentation.json")
DEFAULT_MODEL_CONFIG = Path("configs/clef_piano_base.yaml")


def load_augmentation_config(config_path: Path = DEFAULT_AUG_CONFIG) -> Dict[str, Any]:
    """Load augmentation settings from JSON config.

    Returns:
        Dictionary with keys:
        - transpose_enabled: bool
        - feasible_transposes: dict (only used if transpose_enabled)
        - tempo_enabled: bool
        - tempo_range: tuple (min_scale, max_scale)
        - train_soundfonts: list
        - valid_soundfonts: list
        - test_soundfonts: list
        - num_versions: dict
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Check if transpose is enabled
    # NOTE: Transpose is disabled for clef-piano-base because:
    # 1. Piano voicing is key-specific - transposing disrupts idiomatic patterns
    # 2. Zeng's original implementation had a multiprocessing bug (fork inherits
    #    random state) causing only ~2 unique transposes across 10 versions
    # 3. Real recordings (ASAP) are not transposed, so this improves evaluation fairness
    transpose_enabled = config['transpose'].get('enabled', False)

    # Convert string keys to int for feasible_transposes (kept for reference)
    feasible_transposes = {}
    if transpose_enabled:
        for key, value in config['transpose']['feasible_transposes'].items():
            if not key.startswith('_'):  # Skip comment keys
                feasible_transposes[int(key)] = value

    # Tempo augmentation settings
    tempo_config = config.get('tempo', {})
    tempo_enabled = tempo_config.get('enabled', True)
    tempo_range = (
        tempo_config.get('min_scale', 0.85),
        tempo_config.get('max_scale', 1.15),
    )

    return {
        'transpose_enabled': transpose_enabled,
        'feasible_transposes': feasible_transposes,
        'tempo_enabled': tempo_enabled,
        'tempo_range': tempo_range,
        'train_soundfonts': config['soundfonts']['train'],
        'valid_soundfonts': config['soundfonts']['valid'],
        'test_soundfonts': config['soundfonts']['test'],
        'num_versions': config['num_versions'],
    }


def get_key_signature(score: m21.stream.Score) -> int:
    """Extract key signature (number of sharps, negative for flats)."""
    try:
        for part in score.parts:
            for measure in part.getElementsByClass(m21.stream.Measure):
                ks = measure.keySignature
                if ks is not None:
                    return ks.sharps
        # Fallback: search recursively
        key_sigs = score.flatten().getElementsByClass(m21.key.KeySignature)
        if key_sigs:
            return key_sigs[0].sharps
    except Exception as e:
        logger.debug(f"Could not extract key signature: {e}")
    return 0  # Default to C major


def extract_measure_times(score: m21.stream.Score) -> List[Dict[str, Any]]:
    """Extract measure timing from a music21 Score (single source of truth).

    Uses the Score's internal MetronomeMark objects and measure offsets.
    This matches exactly what score.write('midi') produces, so timing is
    guaranteed to align with the rendered audio.

    Args:
        score: music21 Score object (already parsed via converter21)

    Returns:
        List of measure info (in sequential order, may have duplicate numbers
        from repeat expansion):
        [
            {"measure": 1, "start_sec": 0.0, "end_sec": 1.25},
            {"measure": 2, "start_sec": 1.25, "end_sec": 2.50},
            ...
        ]
    """
    if not score.parts:
        return []

    part = score.parts[0]

    # Build tempo map from MetronomeMark objects: [(offset_ql, qpm)]
    flat_score = score.flatten()
    tempo_marks = list(flat_score.getElementsByClass(m21.tempo.MetronomeMark))
    tempo_marks.sort(key=lambda t: t.offset)

    tempo_map = []
    for tm in tempo_marks:
        # numberSounding gives quarter-note BPM regardless of beat unit.
        # However, converter21 often leaves numberSounding=None even when
        # referent != quarter (e.g. !!!OMD "[half-dot] = 60" → number=60,
        # referent=dotted-half, numberSounding=None).  In that case we must
        # convert manually: qpm = number * referent.quarterLength.
        if tm.numberSounding:
            qpm = tm.numberSounding
        elif tm.number is not None and tm.referent is not None:
            qpm = tm.number * tm.referent.quarterLength
        else:
            qpm = tm.number
        if qpm is None:
            # converter21 creates MetronomeMark from !!!OMD records (e.g. "TRIO")
            # that are section labels, not tempo changes. Skip them.
            continue
        tempo_map.append((float(tm.offset), float(qpm)))

    if not tempo_map:
        tempo_map = [(0.0, 120.0)]

    def offset_to_seconds(offset: float) -> float:
        """Convert quarter-note offset to seconds using tempo map."""
        seconds = 0.0
        prev_offset = 0.0
        prev_qpm = tempo_map[0][1]

        for t_offset, t_qpm in tempo_map:
            if t_offset >= offset:
                break
            seconds += (t_offset - prev_offset) * (60.0 / prev_qpm)
            prev_offset = t_offset
            prev_qpm = t_qpm

        seconds += (offset - prev_offset) * (60.0 / prev_qpm)
        return seconds

    # Extract ALL measures in sequential order (no dedup by number,
    # since repeat expansion can produce duplicate measure numbers)
    measures_info = []
    for measure in part.getElementsByClass(m21.stream.Measure):
        try:
            offset_start = float(measure.offset)
            offset_end = offset_start + float(measure.duration.quarterLength)

            measures_info.append({
                "measure": measure.number,
                "start_sec": round(offset_to_seconds(offset_start), 4),
                "end_sec": round(offset_to_seconds(offset_end), 4),
            })
        except Exception:
            continue

    return measures_info


def extract_measure_offsets(score: m21.stream.Score) -> List[Dict[str, Any]]:
    """Extract measure boundaries as quarter-note offsets from a music21 Score.

    These offsets are used to inject MIDI marker events at exact tick
    positions, avoiding the tempo-map mismatch between music21's internal
    offset_to_seconds and its MIDI writer.

    Args:
        score: music21 Score object

    Returns:
        List of {"measure": int, "start_qn": float, "end_qn": float}
    """
    if not score.parts:
        return []

    part = score.parts[0]
    offsets = []
    for measure in part.getElementsByClass(m21.stream.Measure):
        try:
            start = float(measure.offset)
            end = start + float(measure.duration.quarterLength)
            offsets.append({
                "measure": measure.number,
                "start_qn": start,
                "end_qn": end,
            })
        except Exception:
            continue
    return offsets


def inject_measure_markers(midi_path: str, measure_offsets: List[Dict[str, Any]]) -> None:
    """Inject marker meta-events at measure boundaries into a MIDI file.

    Markers are placed on a dedicated track so they don't interfere with
    note data.  FluidSynth ignores all meta-events, so the audio output
    is unaffected.  After tempo scaling (which also scales marker delta
    times), the markers can be read back with ``read_measure_times_from_midi``
    to obtain precise measure boundary times in seconds.

    Args:
        midi_path: Path to the MIDI file (modified in place).
        measure_offsets: Output of ``extract_measure_offsets``.
    """
    from mido import MidiFile, MidiTrack, MetaMessage

    mid = MidiFile(midi_path)
    tpb = mid.ticks_per_beat

    marker_track = MidiTrack()
    prev_tick = 0
    for m in measure_offsets:
        tick = int(round(m["start_qn"] * tpb))
        delta = max(0, tick - prev_tick)
        marker_track.append(
            MetaMessage("marker", text=f"bar_{m['measure']}", time=delta)
        )
        prev_tick = tick

    # End-of-piece marker at the last measure's end
    if measure_offsets:
        end_tick = int(round(measure_offsets[-1]["end_qn"] * tpb))
        delta = max(0, end_tick - prev_tick)
        marker_track.append(MetaMessage("marker", text="end", time=delta))

    mid.tracks.append(marker_track)
    mid.save(midi_path)


def read_measure_times_from_midi(midi_path: str) -> List[Dict[str, Any]]:
    """Read measure boundary times (in seconds) from MIDI marker events.

    mido's file iterator automatically applies the MIDI tempo map when
    converting tick deltas to seconds, so the returned times are guaranteed
    to match what FluidSynth renders.

    Args:
        midi_path: Path to a MIDI file containing ``bar_*`` / ``end`` markers.

    Returns:
        List of {"measure": int, "start_sec": float, "end_sec": float}.
        Returns [] if no markers are found.
    """
    from mido import MidiFile

    mid = MidiFile(midi_path)

    # Collect absolute times for each marker
    markers: List[tuple] = []  # (abs_sec, text)
    abs_time = 0.0
    for msg in mid:
        abs_time += msg.time
        if msg.type == "marker":
            markers.append((abs_time, msg.text))

    if not markers:
        return []

    # Build measure list from consecutive bar_* markers
    measures = []
    for i, (t, text) in enumerate(markers):
        if not text.startswith("bar_"):
            continue
        measure_num = int(text[4:])
        # end_sec is the next marker's time (bar_* or "end")
        if i + 1 < len(markers):
            end_sec = markers[i + 1][0]
        else:
            end_sec = t  # fallback: zero-length last measure
        measures.append({
            "measure": measure_num,
            "start_sec": round(t, 4),
            "end_sec": round(end_sec, 4),
        })

    return measures


def extract_measure_times_from_kern(kern_content: str) -> List[Dict[str, Any]]:
    """Fallback: Extract measure times directly from kern content.

    Used when converter21 fails to create a valid music21 score.
    Parses barlines and durations directly from the kern text.

    Args:
        kern_content: Raw kern file content (should be already expanded)

    Returns:
        List of measure info like extract_measure_times()
    """
    import re

    # Duration mapping: kern duration -> quarter note length
    DURATION_MAP = {
        '00': 8.0, '0': 4.0, '1': 4.0, '2': 2.0, '4': 1.0,
        '8': 0.5, '16': 0.25, '32': 0.125, '64': 0.0625,
        '3': 4.0/3, '6': 2.0/3, '12': 1.0/3, '24': 0.5/3, '48': 0.25/3,
    }

    lines = kern_content.split('\n')
    measures_info = []

    # State
    current_offset = 0.0  # in quarter notes
    current_tempo = 120.0  # BPM
    current_measure_start = 0.0
    current_measure_num = 0
    time_sig_quarters = 4.0  # quarters per measure (default 4/4)

    # Tempo map for offset -> seconds conversion
    tempo_changes = [(0.0, 120.0)]  # (offset, bpm)

    def offset_to_seconds(offset: float) -> float:
        """Convert quarter note offset to seconds."""
        seconds = 0.0
        prev_offset = 0.0
        prev_bpm = tempo_changes[0][1]
        for t_offset, t_bpm in tempo_changes:
            if t_offset >= offset:
                break
            seconds += (t_offset - prev_offset) * (60.0 / prev_bpm)
            prev_offset = t_offset
            prev_bpm = t_bpm
        seconds += (offset - prev_offset) * (60.0 / prev_bpm)
        return seconds

    def parse_duration(token: str) -> float:
        """Parse kern token duration."""
        match = re.match(r'^(\d+)', token)
        if not match:
            return 0.0
        dur_str = match.group(1)
        base_dur = DURATION_MAP.get(dur_str, 1.0)
        # Apply dots
        dots = token.count('.')
        if dots > 0:
            multiplier = sum(0.5 ** i for i in range(dots + 1))
            base_dur *= multiplier
        return base_dur

    for line in lines:
        line = line.strip()
        if not line or line.startswith('!'):
            continue

        # Parse tempo
        if '*MM' in line:
            match = re.search(r'\*MM=?(\d+)', line)
            if match:
                current_tempo = float(match.group(1))
                tempo_changes.append((current_offset, current_tempo))

        # Parse time signature
        if '*M' in line and '/' in line:
            match = re.search(r'\*M(\d+)/(\d+)', line)
            if match:
                num, denom = int(match.group(1)), int(match.group(2))
                time_sig_quarters = num * (4.0 / denom)

        # Barline - record measure
        if line.startswith('='):
            parts = line.split('\t')
            bar_part = parts[0]

            # Extract measure number
            match = re.match(r'=+(\d+)?', bar_part)
            if match:
                measure_end = current_offset
                if current_measure_num > 0 or '-' in bar_part:
                    # Record previous measure
                    measures_info.append({
                        "measure": current_measure_num,
                        "start_sec": round(offset_to_seconds(current_measure_start), 4),
                        "end_sec": round(offset_to_seconds(measure_end), 4),
                    })
                # Start new measure
                current_measure_start = current_offset
                if match.group(1):
                    current_measure_num = int(match.group(1))
                else:
                    current_measure_num += 1
            continue

        # Skip interpretation lines
        if line.startswith('*'):
            continue

        # Data line - find max duration
        parts = line.split('\t')
        max_dur = 0.0
        for part in parts:
            if part == '.' or not part:
                continue
            # Handle chords (space-separated)
            for subtoken in part.split():
                if 'r' in subtoken or re.search(r'[a-gA-G]', subtoken):
                    dur = parse_duration(subtoken)
                    max_dur = max(max_dur, dur)

        if max_dur > 0:
            current_offset += max_dur

    # Record last measure
    if current_measure_num > 0:
        measures_info.append({
            "measure": current_measure_num,
            "start_sec": round(offset_to_seconds(current_measure_start), 4),
            "end_sec": round(offset_to_seconds(current_offset), 4),
        })

    return measures_info


def phase1_score_to_kern(
    humsyn_dir: Path = DEFAULT_HUMSYN_DIR,
    musesyn_dir: Path = DEFAULT_MUSESYN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    preset: Optional[str] = None,
    save_visual_info: bool = True,
) -> Dict[str, str]:
    """Phase 1: Convert scores to kern format.

    Args:
        humsyn_dir: Path to HumSyn directory
        musesyn_dir: Path to MuseSyn directory
        output_dir: Path to output directory
        metadata_dir: Path to metadata directory
        preset: Processing preset:
            - "clef-piano-base": Chopin filter + Joplin strip
            - "clef-piano-full": Chopin filter only
            - None: No filtering (clef-tutti)
        save_visual_info: Whether to save visual info JSON files for Visual Aux Head

    Returns:
        Dictionary of {filename: status} for all processed files
    """
    kern_output_dir = output_dir / "kern"
    kern_output_dir.mkdir(parents=True, exist_ok=True)

    # Visual info output directory (for Visual Auxiliary Head ground truth)
    visual_output_dir = output_dir / "visual" if save_visual_info else None
    if visual_output_dir:
        visual_output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Process HumSyn
    logger.info("Processing HumSyn...")
    selected_chopin_path = metadata_dir / "selected_chopin.txt"
    repeat_map_dir = output_dir / "repeat_map"
    repeat_map_dir.mkdir(parents=True, exist_ok=True)

    humsyn_processor = HumSynProcessor(
        input_dir=humsyn_dir,
        output_dir=kern_output_dir,
        visual_dir=visual_output_dir,
        repeat_map_dir=repeat_map_dir,
        selected_chopin_path=selected_chopin_path,
        preset=preset,
    )
    humsyn_results = humsyn_processor.process_all()
    all_results.update(humsyn_results)

    # Process MuseSyn
    logger.info("Processing MuseSyn...")
    musesyn_processor = MuseSynProcessor(
        input_dir=musesyn_dir,
        output_dir=kern_output_dir,
        visual_dir=visual_output_dir,
        repeat_map_dir=repeat_map_dir,
        preset=preset,
    )
    musesyn_results = musesyn_processor.process_all()
    all_results.update(musesyn_results)

    return all_results


def create_ground_truth_kern(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
) -> Dict[str, str]:
    """Create Zeng-vocab compatible ground truth kern files.

    Applies full transformation pipeline:
    1. clean_kern_sequence() - standardize tokens, convert dotted triplets, remove beams
    2. apply_zeng_pitch_compat() - Zeng pitch/accidental compatibility:
       - Split breves (0, 00) into tied whole notes
       - Strip natural accidentals (n)
       - Convert double accidentals (## → enharmonic, -- → enharmonic)
       - Remove slur/phrase markers ((), {})
       - Clamp extreme pitches to piano range
    3. expand_tuplets_to_zeng_vocab() - expand 5/7/11-tuplets to standard durations
    4. quantize_oov_tuplets() - IP-based quantization for remaining OOV durations
    5. X%Y → quantize to nearest standard duration

    Args:
        output_dir: Base output directory (reads from kern/, writes to kern_gt/)
        seed: Random seed for reproducible quantization

    Returns:
        Dictionary of {filename: status}
    """
    import re

    kern_dir = output_dir / "kern"
    kern_gt_dir = output_dir / "kern_gt"
    kern_gt_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    kern_files = sorted(kern_dir.glob("*.krn"))

    logger.info(f"Creating ground truth kern files: {len(kern_files)} files")

    for kern_path in tqdm(kern_files, desc="Creating kern_gt"):
        stem = kern_path.stem

        try:
            with open(kern_path, "r", encoding="utf-8") as f:
                raw_kern = f.read()

            # kern/ files are already repeat-expanded (Phase 1).
            # Repeat maps are saved in Phase 1 alongside kern/.

            # Step 0: Fix spine timing (insert rests to sync spines)
            # This ensures kern_gt timing matches audio timing
            spine_fixed = fix_kern_spine_timing(raw_kern)

            # Step 1: Clean kern sequence (standardize tokens, dotted triplets, remove beams)
            # strip_cue=True: kern/ retains cue notes (needed for Phase 2 MIDI generation);
            # kern_gt/ must strip them so the model doesn't predict orchestral cue notes.
            cleaned = clean_kern_sequence(spine_fixed, warn_tuplet_ratio=False, strip_cue=True)

            # Step 2: Apply Zeng pitch/accidental compatibility
            # (split breves, strip n, convert ##/--, remove slur/phrase, clamp extreme)
            pitch_compat = apply_zeng_pitch_compat(cleaned)

            # Step 3: Resolve *rscale regions first (divide durations by factor,
            # strip markers) so expand_tuplets sees actual recip values.
            # e.g., *rscale:2 + dur 36 → dur 18, which NINE_TUPLET_MAP expands
            # to uniform 20s. Must run BEFORE expand_tuplets to avoid IP solver
            # producing mixed 16/20 or 32/40 that breaks converter21 tuplet inference.
            rscale_resolved = resolve_rscale_regions(pitch_compat)

            # Step 3.5: Expand standard tuplets to Zeng vocab
            tuplet_expanded = expand_tuplets_to_zeng_vocab(rscale_resolved)

            # Step 4: Quantize remaining OOV tuplets with IP solver
            quantized = quantize_oov_tuplets(tuplet_expanded, seed=seed)

            # Step 5: Quantize X%Y tuplet ratios + strip *tuplet markers
            quantized_final = quantize_tuplet_ratios(quantized)

            # Step 6: Final pass for double accidentals
            ground_truth = apply_zeng_pitch_compat(quantized_final)

            # Write ground truth file
            gt_path = kern_gt_dir / f"{stem}.krn"
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write(ground_truth)

            results[stem] = "success"

        except Exception as e:
            logger.error(f"Error processing {stem}: {e}")
            results[stem] = f"error: {e}"

    # Summary
    success = sum(1 for v in results.values() if v == "success")
    errors = sum(1 for v in results.values() if v.startswith("error"))
    logger.info(f"Ground truth creation complete: {success} success, {errors} errors")

    return results


def _process_single_kern(args: Tuple) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """Worker function for parallel processing a single kern file.

    Args:
        args: Tuple of (kern_path, output_dir, soundfont_dir, split, num_versions,
                        available_soundfonts, transpose_enabled, feasible_transposes,
                        tempo_enabled, tempo_range, musesyn_dir)

    Returns:
        Tuple of:
        - results: Dictionary of {audio_name: status}
        - alignment: Dictionary of {audio_key: alignment_info}
          alignment_info contains: tempo_scaling, duration_sec, audio_measures
    """
    (kern_path, output_dir, soundfont_dir, split, num_versions, available_soundfonts,
     transpose_enabled, feasible_transposes, tempo_enabled, tempo_range, musesyn_dir) = args

    # Import inside worker to avoid pickling issues
    import converter21
    import music21 as m21
    import tempfile
    converter21.register()

    try:
        from midi2audio import FluidSynth
        from src.audio.zeng_synthesis import MIDIProcess, render_one_midi, create_default_compressor
        has_zeng = True
    except ImportError:
        from midi2audio import FluidSynth
        has_zeng = False

    from src.score.sanitize_kern import sanitize_kern_for_audio
    from src.score.sanitize_piano_score import sanitize_score
    from src.score.expand_repeat import expand_musesyn_score

    midi_dir = output_dir / "midi"
    audio_dir = output_dir / "audio"
    stem = kern_path.stem
    results = {}
    alignment_info = {}  # {audio_key: {tempo_scaling, duration_sec, audio_measures}}

    logger.debug(f"Processing: {stem}")

    # Set reproducible seed based on filename
    # NOTE: Use hashlib.md5 instead of hash() because Python's hash() is
    # randomized across interpreter sessions (PYTHONHASHSEED). This ensures
    # reproducible augmentation choices across multiple runs.
    file_seed = int(hashlib.md5(stem.encode()).hexdigest(), 16) % (2**32)
    set_seed(SEED_DATA_AUGMENTATION + file_seed)

    # Check if this is a MuseSyn file
    is_musesyn = stem.startswith("musesyn_")

    expanded_kern = None

    try:
        if is_musesyn:
            # For MuseSyn: use original XML + sanitize_score
            # This avoids rhythm inconsistency issues in kern parsing
            original_name = stem[8:]  # Remove "musesyn_" prefix
            xml_path = musesyn_dir / "xml" / f"{original_name}.xml"

            if not xml_path.exists():
                return {f"{stem}_v0": f"error: XML not found - {xml_path}"}, {}

            score = m21.converter.parse(str(xml_path))
            sanitize_score(score)

            # Expand repeats at music21 level for accurate measure timing.
            # score.write('midi') auto-expands repeats, so measure offsets
            # must also come from the expanded score to match MIDI duration.
            score, _ = expand_musesyn_score(score)
        else:
            # For HumSyn: use kern file with repeat expansion and sanitization
            # Original kern with repeat markers is preserved as ground truth
            with open(kern_path, "r", encoding="utf-8") as f:
                kern_content = f.read()

            # Sanitize kern for audio generation:
            # 1. Expand repeats
            # 2. Fix spine timing inconsistencies
            expanded_kern = sanitize_kern_for_audio(kern_content)

            # Write sanitized kern to temp file for music21 parsing
            with tempfile.NamedTemporaryFile(mode="w", suffix=".krn", delete=False) as tmp:
                tmp.write(expanded_kern)
                expanded_kern_path = tmp.name

            score = m21.converter.parse(expanded_kern_path, format="humdrum")
            sanitize_score(score)

            # Clean up temp file
            Path(expanded_kern_path).unlink(missing_ok=True)

    except Exception as e:
        return {f"{stem}_v0": f"error: parse failed - {e}"}, {}

    # Get key signature for transpose selection
    original_key = get_key_signature(score)
    original_key = max(-6, min(7, original_key))  # Clamp to valid range

    # Ensure instruments are set
    for part in score.parts:
        if not part.getElementsByClass(m21.instrument.Instrument):
            part.insert(0, m21.instrument.Piano())

    # Extract measure boundary offsets (in quarter-notes) for MIDI marker
    # injection.  These are converted to ticks and embedded into the MIDI
    # file so that after tempo scaling we can read back precise measure
    # times in seconds directly from the MIDI's own tempo map.
    measure_offsets = extract_measure_offsets(score)

    # Fallback: if Score-based extraction fails, try kern-based extraction
    # (returns seconds, not quarter-note offsets — used only for the legacy
    # score_measures path when marker injection is not possible).
    score_measures_fallback: Optional[List[Dict[str, Any]]] = None
    if not measure_offsets and expanded_kern:
        score_measures_fallback = extract_measure_times_from_kern(expanded_kern)

    for version in range(num_versions):
        # Augmentation parameters
        # NOTE: Transpose is disabled by default. See load_augmentation_config() for rationale.
        if split == "train":
            if transpose_enabled and feasible_transposes:
                transpose = random.choice(feasible_transposes[original_key])
            else:
                transpose = 0  # No transpose - preserves piano voicing patterns
            # Soundfont selection: deterministic mapping (version -> soundfont)
            soundfont = available_soundfonts[version % len(available_soundfonts)]
        elif split == "valid":
            transpose = 0
            soundfont = available_soundfonts[0]
        else:  # test: one version per soundfont, deterministic
            transpose = 0
            soundfont = available_soundfonts[version % len(available_soundfonts)]

        # Build output names
        # Train uses _v{N} suffix; valid/test use soundfont name as differentiator
        if split == "train":
            version_suffix = f"_v{version}"
            midi_name = f"{stem}{version_suffix}.mid"
            audio_name = f"{stem}{version_suffix}~{soundfont[:-4]}.wav"
            audio_key = f"{stem}{version_suffix}~{soundfont[:-4]}"
        else:
            midi_name = f"{stem}~{soundfont[:-4]}.mid"
            audio_name = f"{stem}~{soundfont[:-4]}.wav"
            audio_key = f"{stem}~{soundfont[:-4]}"

        midi_path = midi_dir / midi_name
        audio_path = audio_dir / audio_name

        # Check if audio already exists
        audio_exists = audio_path.exists()

        try:
            # Apply transpose if needed
            if transpose != 0:
                transposed_score = score.transpose(transpose)
            else:
                transposed_score = score

            # Draw tempo scaling once from RNG (1 call per version, deterministic).
            # This value is used both for metadata and passed to MIDIProcess.process()
            # so that MIDIProcess does NOT draw its own RNG call.
            import numpy as np
            tempo_scaling = 1.0  # Default: no scaling
            if tempo_enabled and split == "train":
                tempo_scaling = np.random.uniform(tempo_range[0], tempo_range[1])

            # Skip audio generation if file already exists.
            # Read audio_measures from the MIDI's embedded markers (placed
            # during a previous run).  This is the single source of truth.
            if audio_exists:
                audio_measures = read_measure_times_from_midi(str(midi_path))
                duration_sec = audio_measures[-1]["end_sec"] if audio_measures else 0.0
                results[audio_name] = "skipped (exists)"
                alignment_info[audio_key] = {
                    "tempo_scaling": round(tempo_scaling, 6),
                    "duration_sec": round(duration_sec, 4),
                    "audio_measures": audio_measures,
                }
                continue

            # Write MIDI
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            midi_written = False
            try:
                transposed_score.write("midi", fp=str(midi_path))
                if midi_path.exists():
                    print(f"[MIDI OK] {midi_path.name}", flush=True)
                    midi_written = True
                else:
                    print(f"[MIDI FAIL] {midi_path.name} - file not created", flush=True)
            except Exception as e:
                print(f"[MIDI FAIL] {midi_path.name} - {e}", flush=True)

            if not midi_written:
                results[audio_name] = "error: midi write failed"
                continue

            # Inject measure-boundary markers into the MIDI file.
            # These survive tempo scaling (apply_scaling also scales marker
            # delta times) and can be read back afterwards for precise timing.
            if measure_offsets:
                inject_measure_markers(str(midi_path), measure_offsets)

            # Apply tempo scaling via MIDIProcess (Zeng-style)
            # Pass pre-drawn tempo_scaling to avoid RNG double-consumption.
            tempo_scaling_success = True
            if has_zeng and tempo_enabled:
                temp_midi_path = midi_dir / f"temp_{stem}_{version}.mid"
                midi_proc = MIDIProcess(str(midi_path), split)
                result = midi_proc.process(
                    str(midi_path),
                    str(temp_midi_path),
                    tempo_range=tempo_range,
                    scaling=tempo_scaling,
                )
                scaling, original_length, tempo_scaling_success = result
                # Clean up temp file
                if temp_midi_path.exists():
                    temp_midi_path.unlink()

                # If tempo scaling failed (negative delta time), fall back to tempo=1.0
                if not tempo_scaling_success:
                    tempo_scaling = 1.0

            # Read audio_measures from MIDI markers (single source of truth).
            # The markers were injected above and scaled by MIDIProcess.
            # mido's iterator applies the MIDI tempo map, so the times are
            # guaranteed to match what FluidSynth renders.
            audio_measures = read_measure_times_from_midi(str(midi_path))
            if not audio_measures and score_measures_fallback:
                # Fallback: kern-based timing (no markers available)
                audio_measures = [
                    {
                        "measure": m["measure"],
                        "start_sec": round(m["start_sec"] * tempo_scaling, 4),
                        "end_sec": round(m["end_sec"] * tempo_scaling, 4),
                    }
                    for m in score_measures_fallback
                ]
            duration_sec = audio_measures[-1]["end_sec"] if audio_measures else 0.0

            # Render MIDI -> Audio with loudness normalization
            sf_path = soundfont_dir / soundfont
            fs = FluidSynth(str(sf_path), sample_rate=44100)

            try:
                if has_zeng:
                    compressor = create_default_compressor()
                    render_one_midi(fs, compressor, str(midi_path), str(audio_path))
                else:
                    fs.midi_to_audio(str(midi_path), str(audio_path))

                if audio_path.exists():
                    print(f"[AUDIO OK] {audio_path.name}", flush=True)
                    results[audio_name] = "success"
                    # Store alignment info for newly generated files
                    alignment_info[audio_key] = {
                        "tempo_scaling": round(tempo_scaling, 6),
                        "duration_sec": round(duration_sec, 4),
                        "audio_measures": audio_measures,
                    }
                else:
                    print(f"[AUDIO FAIL] {audio_path.name} - file not created", flush=True)
                    results[audio_name] = "error: audio not created"
            except Exception as e:
                print(f"[AUDIO FAIL] {audio_path.name} - {e}", flush=True)
                results[audio_name] = f"error: audio render failed - {e}"

        except Exception as e:
            results[audio_name] = f"error: {e}"

    return results, alignment_info


def phase2_kern_to_audio(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    soundfont_dir: Path = DEFAULT_SOUNDFONT_DIR,
    musesyn_dir: Path = DEFAULT_MUSESYN_DIR,
    aug_config_path: Path = DEFAULT_AUG_CONFIG,
    workers: int = 1,
) -> Dict[str, str]:
    """Phase 2: Convert kern files to audio with augmentation.

    Args:
        output_dir: Path to output directory (contains kern/)
        metadata_dir: Path to metadata directory
        soundfont_dir: Path to soundfont directory
        musesyn_dir: Path to MuseSyn directory (for XML source files)
        aug_config_path: Path to augmentation config JSON
        workers: Number of parallel workers (1 = sequential)

    Returns:
        Dictionary of {filename: status}
    """
    # Load augmentation config
    aug_config = load_augmentation_config(aug_config_path)
    transpose_enabled = aug_config['transpose_enabled']
    feasible_transposes = aug_config['feasible_transposes']
    tempo_enabled = aug_config['tempo_enabled']
    tempo_range = aug_config['tempo_range']
    train_soundfonts = aug_config['train_soundfonts']
    valid_soundfonts = aug_config['valid_soundfonts']
    test_soundfonts = aug_config['test_soundfonts']
    num_versions_config = aug_config['num_versions']

    if not transpose_enabled:
        logger.info("Transpose augmentation DISABLED (preserves piano voicing patterns)")
    if tempo_enabled:
        logger.info(f"Tempo augmentation ENABLED (range: {tempo_range[0]:.2f}-{tempo_range[1]:.2f})")
    else:
        logger.info("Tempo augmentation DISABLED")

    kern_dir = output_dir / "kern"
    midi_dir = output_dir / "midi"
    audio_dir = output_dir / "audio"

    midi_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load split files
    test_split = set()
    valid_split = set()

    test_split_path = metadata_dir / "test_split.txt"
    valid_split_path = metadata_dir / "valid_split.txt"

    if test_split_path.exists():
        df = pd.read_csv(test_split_path)
        test_split = set(df["name"].tolist())
    if valid_split_path.exists():
        df = pd.read_csv(valid_split_path)
        valid_split = set(df["name"].tolist())

    # Load skip list (files with quality issues)
    skip_files = set()
    skip_list_path = metadata_dir / "skip_files.txt"
    if skip_list_path.exists():
        with open(skip_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    skip_files.add(line)
        if skip_files:
            logger.info(f"Skipping {len(skip_files)} files from skip_files.txt")

    # Check available soundfonts (train, valid, test separately)
    available_train_soundfonts = [sf for sf in train_soundfonts if (soundfont_dir / sf).exists()]
    available_valid_soundfonts = [sf for sf in valid_soundfonts if (soundfont_dir / sf).exists()]
    available_test_soundfonts = [sf for sf in test_soundfonts if (soundfont_dir / sf).exists()]

    if not available_train_soundfonts:
        logger.error(f"No train soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {train_soundfonts}")
        return {}

    if not available_valid_soundfonts:
        logger.error(f"No valid soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {valid_soundfonts}")
        return {}

    if not available_test_soundfonts:
        logger.error(f"No test soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {test_soundfonts}")
        return {}

    logger.info(f"Available train soundfonts: {available_train_soundfonts}")
    logger.info(f"Available valid soundfonts: {available_valid_soundfonts}")
    logger.info(f"Available test soundfonts: {available_test_soundfonts}")

    # Prepare task list
    kern_files = sorted(kern_dir.glob("*.krn"))
    tasks = []

    for kern_path in kern_files:
        stem = kern_path.stem

        # Skip files with quality issues
        if stem in skip_files:
            logger.debug(f"Skipping {stem} (in skip_files.txt)")
            continue

        # Determine split
        split = "train"
        for test_name in test_split:
            if _match_split_name(stem, test_name):
                split = "test"
                break
        if split == "train":
            for valid_name in valid_split:
                if _match_split_name(stem, valid_name):
                    split = "valid"
                    break

        n_versions = num_versions_config[split]
        # Each split uses its own soundfont list
        if split == "train":
            soundfonts_for_split = available_train_soundfonts
        elif split == "valid":
            soundfonts_for_split = available_valid_soundfonts
        else:  # test
            soundfonts_for_split = available_test_soundfonts

        tasks.append((kern_path, output_dir, soundfont_dir, split, n_versions, soundfonts_for_split,
                      transpose_enabled, feasible_transposes, tempo_enabled, tempo_range, musesyn_dir))

    logger.info(f"Processing {len(tasks)} kern files with {workers} workers...")

    all_results = {}

    # Generate augmentation metadata first (creates fresh metadata with kern_measures)
    # This replaces the old incremental load, ensuring metadata is always in sync with kern files
    logger.info("Generating augmentation metadata...")
    metadata = generate_metadata(
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        aug_config_path=aug_config_path,
    )
    metadata_path = output_dir / "augmentation_metadata.json"

    # Save initial metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Generated metadata for {len(metadata)} audio entries")

    updated_count = 0
    save_interval = 10  # Save metadata every N kern files

    alignment_mismatches = []

    def update_metadata_incremental(alignment: dict) -> int:
        """Update metadata in memory with alignment info. Returns count of updates."""
        count = 0
        for audio_key, align_info in alignment.items():
            if audio_key in metadata:
                # Validate: Score measure count must match kern_gt measure count.
                # Score is the single source of truth for measure numbering/timing.
                kern_measures = metadata[audio_key].get("kern_measures", [])
                audio_measures = align_info["audio_measures"]
                if kern_measures and audio_measures and len(kern_measures) != len(audio_measures):
                    alignment_mismatches.append(
                        f"{audio_key}: kern_measures={len(kern_measures)}, "
                        f"audio_measures(Score)={len(audio_measures)}"
                    )

                metadata[audio_key]["tempo_scaling"] = align_info["tempo_scaling"]
                metadata[audio_key]["duration_sec"] = align_info["duration_sec"]
                metadata[audio_key]["audio_measures"] = align_info["audio_measures"]
                count += 1
        return count

    def save_metadata():
        """Save metadata to disk."""
        if metadata:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    if workers == 1:
        # Sequential processing with incremental metadata updates
        for i, task in enumerate(tqdm(tasks, desc="Processing kern files")):
            results, alignment = _process_single_kern(task)
            all_results.update(results)
            updated_count += update_metadata_incremental(alignment)

            # Save periodically
            if (i + 1) % save_interval == 0:
                save_metadata()
    else:
        # Parallel processing with incremental metadata updates (in main thread)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_single_kern, task): task[0] for task in tasks}

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing kern files")):
                kern_path = futures[future]
                try:
                    results, alignment = future.result()
                    all_results.update(results)
                    updated_count += update_metadata_incremental(alignment)

                    # Save periodically
                    if (i + 1) % save_interval == 0:
                        save_metadata()
                except Exception as e:
                    import traceback
                    logger.error(f"Error processing {kern_path}: {e}\n{traceback.format_exc()}")
                    all_results[kern_path.stem] = f"error: {e}"

    # Final save
    save_metadata()
    logger.info(f"Updated {updated_count} entries in {metadata_path}")

    # Report alignment validation
    if alignment_mismatches:
        logger.warning(
            f"ALIGNMENT MISMATCH: {len(alignment_mismatches)} entries have "
            f"kern_measures vs audio_measures(Score) count disagreement:"
        )
        for msg in alignment_mismatches[:20]:
            logger.warning(f"  {msg}")
        if len(alignment_mismatches) > 20:
            logger.warning(f"  ... and {len(alignment_mismatches) - 20} more")
    else:
        logger.info("Alignment validation PASSED: all kern_measures/audio_measures counts match")

    # Summary
    success = sum(1 for v in all_results.values() if v == "success")
    skipped = sum(1 for v in all_results.values() if v.startswith("skipped"))
    errors = sum(1 for v in all_results.values() if v.startswith("error"))
    logger.info(f"Phase 2 complete: {success} success, {skipped} skipped, {errors} errors")

    return all_results


# ============================================================
# Phase 2.5: Audio → Mel Spectrogram
# ============================================================

def _process_one_audio_wrapper(args: Tuple) -> str:
    """Worker function wrapper for multiprocessing mel generation.

    Args:
        args: Tuple of (audio_path, mel_path, sample_rate, n_mels,
                        n_fft, hop_length, f_min, f_max)

    Returns:
        Status string: "generated", "skipped", "missing", or "failed"
    """
    (audio_path, mel_path, sample_rate, n_mels,
     n_fft, hop_length, f_min, f_max) = args

    status, _ = process_audio_file(
        audio_path=audio_path,
        mel_path=mel_path,
        target_sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        normalize=True,
        skip_existing=True,
    )
    return status


def phase2_5_audio_to_mel(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_config: Path = DEFAULT_MODEL_CONFIG,
    workers: int = 4,
) -> Dict[str, int]:
    """Phase 2.5: Convert audio files to mel spectrograms.

    Processes all audio files listed in augmentation_metadata.json and
    saves mel spectrograms as .pt files.

    Mel parameters are loaded from the model config file (data.audio section).

    Args:
        output_dir: Base output directory (contains audio/, writes to mel/)
        model_config: Path to model config YAML (for mel parameters)
        workers: Number of parallel workers

    Returns:
        Statistics dictionary with counts of generated, skipped, missing, failed
    """
    # Load mel parameters from config
    mel_config = load_mel_config(model_config)
    sample_rate = mel_config["sample_rate"]
    n_mels = mel_config["n_mels"]
    n_fft = mel_config["n_fft"]
    hop_length = mel_config["hop_length"]
    f_min = mel_config["f_min"]
    f_max = mel_config["f_max"]

    # Load metadata
    metadata_path = output_dir / "augmentation_metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        logger.info("Run Phase 2 first to generate augmentation_metadata.json")
        return {"generated": 0, "skipped": 0, "missing": 0, "failed": 0}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    mel_dir = output_dir / "mel"
    mel_dir.mkdir(exist_ok=True)
    audio_dir = output_dir / "audio"

    logger.info(f"Phase 2.5: Converting {len(metadata)} audio files to mel spectrograms")
    logger.info(f"Config: {model_config}")
    logger.info(f"Parameters: sr={sample_rate}, n_mels={n_mels}, n_fft={n_fft}, "
                f"hop={hop_length}, f_min={f_min}, f_max={f_max}")

    # Prepare tasks (without key, just paths and params)
    tasks = [
        (
            audio_dir / f"{key}.wav",
            mel_dir / f"{key}.pt",
            sample_rate, n_mels, n_fft, hop_length, f_min, f_max,
        )
        for key in metadata.keys()
    ]

    # Process with multiprocessing
    stats = {"generated": 0, "skipped": 0, "missing": 0, "failed": 0}

    if workers == 1:
        # Sequential processing
        for task in tqdm(tasks, desc="Generating mel"):
            result = _process_one_audio_wrapper(task)
            stats[result] += 1
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(
                executor.map(_process_one_audio_wrapper, tasks),
                total=len(tasks),
                desc="Generating mel",
            ))

        for result in results:
            stats[result] += 1

    # Summary
    logger.info(f"\nPhase 2.5 complete:")
    logger.info(f"  Generated: {stats['generated']}")
    logger.info(f"  Skipped:   {stats['skipped']}")
    logger.info(f"  Missing:   {stats['missing']}")
    logger.info(f"  Failed:    {stats['failed']}")
    logger.info(f"  Total mel: {len(list(mel_dir.glob('*.pt')))}")

    return stats


def _match_split_name(processed_name: str, split_name: str) -> bool:
    """Check if a processed filename matches a split file entry.

    Handles mappings like:
    - "beethoven_piano_sonatas_sonata01-1" ↔ "beethoven#sonata01-1"
    - "humdrum_chopin_first_editions_001-1a-HO" ↔ "chopin#001-1a-HO"
    - "joplin_entertainer" ↔ "joplin#entertainer"
    - "musesyn_SomeSong" ↔ "SomeSong"
    """
    # Direct match
    if processed_name == split_name:
        return True

    # Handle "repo#piece" format
    if "#" in split_name:
        prefix, piece = split_name.split("#", 1)

        # Map prefix to our naming convention
        prefix_map = {
            "beethoven": "beethoven_piano_sonatas",
            "haydn": "haydn_piano_sonatas",
            "mozart": "mozart_piano_sonatas",
            "chopin": "humdrum_chopin_first_editions",
            "joplin": "joplin",
            "scarlatti": "scarlatti_keyboard_sonatas",
        }

        if prefix in prefix_map:
            expected_prefix = prefix_map[prefix]
            expected_name = f"{expected_prefix}_{piece}"
            if processed_name == expected_name:
                return True

    # Handle MuseSyn (no prefix in split file)
    if processed_name.startswith("musesyn_"):
        musesyn_name = processed_name[8:]  # Remove "musesyn_" prefix
        if musesyn_name == split_name:
            return True

    return False


def main():
    """Main entry point for preprocessing pipeline."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="clef-piano-base preprocessing pipeline")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "1.5", "2", "2.5", "3", "all"],
        help="Phase to run: 1 (Score→Kern), 2 (Kern→Audio), 2.5 (Audio→Mel), "
             "3 (Create manifests), all (all phases). If not specified, runs 1 and 2.",
    )
    parser.add_argument(
        "--humsyn-dir",
        type=Path,
        default=DEFAULT_HUMSYN_DIR,
        help="HumSyn input directory",
    )
    parser.add_argument(
        "--musesyn-dir",
        type=Path,
        default=DEFAULT_MUSESYN_DIR,
        help="MuseSyn input directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Metadata directory",
    )
    parser.add_argument(
        "--soundfont-dir",
        type=Path,
        default=DEFAULT_SOUNDFONT_DIR,
        help="Soundfont directory",
    )
    parser.add_argument(
        "--aug-config",
        type=Path,
        default=DEFAULT_AUG_CONFIG,
        help="Augmentation config JSON file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for Phase 2 (default: 1 = sequential)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["clef-piano-base", "clef-piano-full"],
        default=None,
        help="Processing preset: clef-piano-base (Chopin filter + Joplin strip), "
             "clef-piano-full (Chopin filter only), or none (default, no filtering)",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Model config YAML file (for mel parameters in Phase 2.5 and 3)",
    )

    args = parser.parse_args()

    # Determine which phases to run
    phase = args.phase
    run_phase1 = phase is None or phase == "1" or phase == "all"
    run_phase1b = phase == "1.5" or phase == "all"
    run_phase2 = phase is None or phase == "2" or phase == "all"
    run_phase2_5 = phase == "2.5" or phase == "all"
    run_phase3 = phase == "3" or phase == "all"

    # Run requested phases
    if run_phase1:
        logger.info("=== Phase 1: Score → Kern ===")
        if args.preset:
            logger.info(f"Using preset: {args.preset}")
        else:
            logger.info("No preset (full data, no filtering)")
        phase1_results = phase1_score_to_kern(
            humsyn_dir=args.humsyn_dir,
            musesyn_dir=args.musesyn_dir,
            output_dir=args.output_dir,
            metadata_dir=args.metadata_dir,
            preset=args.preset,
        )
        success = sum(1 for v in phase1_results.values() if v == "success")
        logger.info(f"Phase 1 complete: {success}/{len(phase1_results)} successful")

    if run_phase1 or run_phase1b:
        # Create ground truth kern files (Zeng vocab compatible)
        logger.info("=== Phase 1b: Creating Ground Truth Kern (Zeng Vocab) ===")
        # Read seed from model config
        import yaml
        with open(args.model_config) as _f:
            _cfg = yaml.safe_load(_f)
        data_aug_seed = _cfg.get('seed', {}).get('data_augmentation', 0)
        gt_results = create_ground_truth_kern(
            output_dir=args.output_dir,
            seed=data_aug_seed,
        )
        gt_success = sum(1 for v in gt_results.values() if v == "success")
        logger.info(f"Phase 1b complete: {gt_success}/{len(gt_results)} successful")

    if run_phase2:
        logger.info("=== Phase 2: Kern → Audio ===")
        phase2_results = phase2_kern_to_audio(
            output_dir=args.output_dir,
            metadata_dir=args.metadata_dir,
            soundfont_dir=args.soundfont_dir,
            musesyn_dir=args.musesyn_dir,
            aug_config_path=args.aug_config,
            workers=args.workers,
        )
        success = sum(1 for v in phase2_results.values() if v == "success")
        logger.info(f"Phase 2 complete: {success}/{len(phase2_results)} successful")

    if run_phase2_5:
        logger.info("=== Phase 2.5: Audio → Mel ===")
        phase2_5_stats = phase2_5_audio_to_mel(
            output_dir=args.output_dir,
            model_config=args.model_config,
            workers=args.workers,
        )
        logger.info(f"Phase 2.5 complete: {phase2_5_stats['generated']} generated, "
                    f"{phase2_5_stats['skipped']} skipped")

    if run_phase3:
        logger.info("=== Phase 3: Create Manifests ===")
        phase3_counts = create_manifest(
            data_dir=args.output_dir,
            config_path=args.model_config,
        )
        total = sum(phase3_counts.values())
        logger.info(f"Phase 3 complete: {total} total samples in manifests")


if __name__ == "__main__":
    main()
