"""
clef-piano-base Preprocessing Pipeline
=======================================

Prepares training data for clef-piano-base model:
- Phase 1: Score → Kern (MuseSyn XML + HumSyn kern)
- Phase 2: Kern → MIDI → Audio (with data augmentation)

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
    python -m src.clef.piano.prepare_zeng_pretrain --phase 1  # Score → Kern
    python -m src.clef.piano.prepare_zeng_pretrain --phase 2  # Kern → Audio
    python -m src.clef.piano.prepare_zeng_pretrain --phase 2 --workers 8  # Parallel
    python -m src.clef.piano.prepare_zeng_pretrain            # Full pipeline
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

from src.preprocessing.humsyn_processor import HumSynProcessor
from src.preprocessing.musesyn_processor import MuseSynProcessor
from src.score.clean_kern import clean_kern_sequence
from src.score.kern_zeng_compat import (
    expand_tuplets_to_zeng_vocab,
    quantize_oov_tuplets,
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


def load_augmentation_config(config_path: Path = DEFAULT_AUG_CONFIG) -> Dict[str, Any]:
    """Load augmentation settings from JSON config.

    Returns:
        Dictionary with keys:
        - transpose_enabled: bool
        - feasible_transposes: dict (only used if transpose_enabled)
        - tempo_enabled: bool
        - tempo_range: tuple (min_scale, max_scale)
        - train_soundfonts: list
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
    """Extract measure timing information from a music21 score.

    Uses the score's tempo markings to convert offsets to seconds.
    Handles tempo changes within the piece.

    Args:
        score: music21 Score object

    Returns:
        List of measure info:
        [
            {"measure": 1, "start_sec": 0.0, "end_sec": 2.5},
            {"measure": 2, "start_sec": 2.5, "end_sec": 5.1},
            ...
        ]
    """
    measures_info = []
    seen_measures = set()

    # Get the first part (assume all parts have same measure structure)
    if not score.parts:
        return measures_info

    part = score.parts[0]

    # Get all tempo marks from the flattened score, sorted by offset
    flat_score = score.flatten()
    tempo_marks = list(flat_score.getElementsByClass(m21.tempo.MetronomeMark))
    tempo_marks.sort(key=lambda t: t.offset)

    # Build tempo map: list of (offset, qpm) tuples
    tempo_map = []
    for tm in tempo_marks:
        tempo_map.append((tm.offset, tm.number))

    # Default tempo if none specified
    if not tempo_map:
        tempo_map = [(0.0, 120.0)]

    def get_tempo_at_offset(offset: float) -> float:
        """Get tempo (qpm) at a given offset."""
        qpm = tempo_map[0][1]  # Default to first tempo
        for t_offset, t_qpm in tempo_map:
            if t_offset <= offset:
                qpm = t_qpm
            else:
                break
        return qpm

    def offset_to_seconds(offset: float) -> float:
        """Convert offset in quarter notes to seconds, handling tempo changes."""
        seconds = 0.0
        prev_offset = 0.0
        prev_qpm = tempo_map[0][1]

        for t_offset, t_qpm in tempo_map:
            if t_offset >= offset:
                break
            # Add time from prev_offset to t_offset at prev_qpm
            seconds += (t_offset - prev_offset) * (60.0 / prev_qpm)
            prev_offset = t_offset
            prev_qpm = t_qpm

        # Add remaining time from prev_offset to target offset
        seconds += (offset - prev_offset) * (60.0 / prev_qpm)
        return seconds

    for measure in part.getElementsByClass(m21.stream.Measure):
        measure_num = measure.number
        if measure_num in seen_measures:
            continue
        seen_measures.add(measure_num)

        try:
            # Get offset in quarter notes
            offset_start = measure.offset
            offset_end = offset_start + measure.duration.quarterLength

            # Convert to seconds
            start_sec = offset_to_seconds(offset_start)
            end_sec = offset_to_seconds(offset_end)

            measures_info.append({
                "measure": measure_num,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
            })
        except Exception:
            # Fallback: skip this measure
            continue

    # Sort by measure number
    measures_info.sort(key=lambda x: x["measure"])

    return measures_info


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
    humsyn_processor = HumSynProcessor(
        input_dir=humsyn_dir,
        output_dir=kern_output_dir,
        visual_dir=visual_output_dir,
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
       - Mark extreme pitches as <unk>
    3. expand_tuplets_to_zeng_vocab() - expand 5/7/11-tuplets to standard durations
    4. quantize_oov_tuplets() - IP-based quantization for remaining OOV durations
    5. X%Y → <unk> - mark complex tuplet ratios as unknown

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

            # Step 0: Fix spine timing (insert rests to sync spines)
            # This ensures kern_gt timing matches audio timing
            spine_fixed = fix_kern_spine_timing(raw_kern)

            # Step 1: Clean kern sequence (standardize tokens, dotted triplets, remove beams)
            cleaned = clean_kern_sequence(spine_fixed, warn_tuplet_ratio=False)

            # Step 2: Apply Zeng pitch/accidental compatibility
            # (split breves, strip n, convert ##/--, remove slur/phrase, mark extreme)
            pitch_compat = apply_zeng_pitch_compat(cleaned)

            # Step 3: Expand standard tuplets to Zeng vocab
            expanded = expand_tuplets_to_zeng_vocab(pitch_compat)

            # Step 4: Quantize remaining OOV tuplets with IP solver
            quantized = quantize_oov_tuplets(expanded, seed=seed)

            # Step 5: Mark X%Y tuplet ratios as <unk>
            # Pattern: digits followed by % followed by digits (e.g., 56%3, 8%9)
            def replace_tuplet_ratio(match):
                # Keep pitch and other markers, replace duration with <unk>
                token = match.group(0)
                # Extract pitch part (letters after the duration)
                pitch_match = re.search(r'[a-gA-Gr]', token)
                if pitch_match:
                    pitch_start = pitch_match.start()
                    return '<unk>' + token[pitch_start:]
                return '<unk>'

            # Replace tokens with X%Y duration pattern
            # Note: \.? handles dotted durations like 20%3.a
            with_unk = re.sub(
                r'\b\d+%\d+\.?[a-gA-Gr][^\s\t]*',
                replace_tuplet_ratio,
                quantized
            )

            # Step 6: Final pass for double accidentals
            # Step 5 may create tokens like <unk>FF## that need conversion
            ground_truth = apply_zeng_pitch_compat(with_unk)

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

    expanded_kern = None  # For fallback measure extraction

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
        else:
            # For HumSyn: use kern file with repeat expansion and sanitization
            # Original kern with repeat markers is preserved as ground truth
            with open(kern_path, "r", encoding="utf-8") as f:
                kern_content = f.read()

            # Sanitize kern for audio generation:
            # 1. Expand repeats
            # 2. Fix tuplet durations
            # 3. Fix spine timing inconsistencies
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

    for version in range(num_versions):
        # Augmentation parameters
        # NOTE: Transpose is disabled by default. See load_augmentation_config() for rationale.
        if split == "train":
            if transpose_enabled and feasible_transposes:
                transpose = random.choice(feasible_transposes[original_key])
            else:
                transpose = 0  # No transpose - preserves piano voicing patterns
            # Soundfont selection: deterministic mapping (version -> soundfont)
            # Zeng original: random.choice() which causes duplicate soundfonts across versions
            # Our choice: version % len(soundfonts) ensures each soundfont appears exactly once
            # Tempo is still random (via MIDIProcess.random_scaling), maintaining Zeng's spirit
            soundfont = available_soundfonts[version % len(available_soundfonts)]
        else:
            transpose = 0
            soundfont = available_soundfonts[0]

        # Build output names
        version_suffix = f"_v{version}" if num_versions > 1 else ""
        midi_name = f"{stem}{version_suffix}.mid"
        audio_name = f"{stem}{version_suffix}~{soundfont[:-4]}.wav"

        midi_path = midi_dir / midi_name
        audio_path = audio_dir / audio_name

        # Audio key for metadata (without .wav extension)
        audio_key = f"{stem}{version_suffix}~{soundfont[:-4]}"

        # Check if audio already exists
        audio_exists = audio_path.exists()

        try:
            # Apply transpose if needed
            if transpose != 0:
                transposed_score = score.transpose(transpose)
            else:
                transposed_score = score

            # Extract measure times (for alignment - needed even for skipped files)
            measure_times = extract_measure_times(transposed_score)

            # Fallback: if score parsing failed, extract from kern directly
            if not measure_times and expanded_kern:
                measure_times = extract_measure_times_from_kern(expanded_kern)

            # Calculate tempo scaling (deterministic based on seed)
            # For existing files, we simulate what MIDIProcess would do
            tempo_scaling = 1.0  # Default: no scaling
            if tempo_enabled and split == "train":
                # Simulate MIDIProcess.random_scaling logic
                # The score length approximates MIDI length
                try:
                    score_length = transposed_score.duration.quarterLength
                    # Estimate original_length in seconds using first tempo
                    flat_score = transposed_score.flatten()
                    tempo_marks = list(flat_score.getElementsByClass(m21.tempo.MetronomeMark))
                    if tempo_marks:
                        qpm = tempo_marks[0].number
                    else:
                        qpm = 120.0
                    original_length = score_length * (60.0 / qpm)

                    # Same logic as MIDIProcess.random_scaling
                    import numpy as np
                    lower_bound = max(tempo_range[0], 4.0 / original_length)
                    upper_bound = min(tempo_range[1], 12.0 / original_length)
                    if lower_bound <= upper_bound:
                        tempo_scaling = np.random.uniform(lower_bound, upper_bound)
                except Exception:
                    tempo_scaling = 1.0

            # Calculate audio_measures (apply tempo scaling to measure times)
            audio_measures = []
            for m in measure_times:
                audio_measures.append({
                    "measure": m["measure"],
                    "start_sec": round(m["start_sec"] * tempo_scaling, 4),
                    "end_sec": round(m["end_sec"] * tempo_scaling, 4),
                })

            # Calculate duration from last measure end time
            duration_sec = audio_measures[-1]["end_sec"] if audio_measures else 0.0

            # Skip audio generation if file already exists
            # But still collect alignment info for metadata update
            if audio_exists:
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

            # Apply tempo scaling via MIDIProcess (Zeng-style)
            tempo_scaling_success = True
            if has_zeng and tempo_enabled:
                temp_midi_path = midi_dir / f"temp_{stem}_{version}.mid"
                midi_proc = MIDIProcess(str(midi_path), split)
                result = midi_proc.process(
                    str(midi_path),
                    str(temp_midi_path),
                    tempo_range=tempo_range,
                )
                # Handle both old (2-tuple) and new (3-tuple) return format
                if len(result) == 3:
                    scaling, original_length, tempo_scaling_success = result
                else:
                    scaling, original_length = result
                    tempo_scaling_success = True
                # Clean up temp file
                if temp_midi_path.exists():
                    temp_midi_path.unlink()

                # If tempo scaling failed, use original MIDI with tempo=1.0
                # Still generate all 4 soundfont versions (v0-v3)
                if not tempo_scaling_success:
                    tempo_scaling = 1.0
                    # Recalculate audio_measures with tempo=1.0
                    audio_measures = []
                    for m in measure_times:
                        audio_measures.append({
                            "measure": m["measure"],
                            "start_sec": round(m["start_sec"], 4),
                            "end_sec": round(m["end_sec"], 4),
                        })
                    duration_sec = audio_measures[-1]["end_sec"] if audio_measures else 0.0
                elif scaling is not None:
                    # Update with actual scaling (should match our simulation)
                    tempo_scaling = scaling
                    # Recalculate audio_measures with actual scaling
                    audio_measures = []
                    for m in measure_times:
                        audio_measures.append({
                            "measure": m["measure"],
                            "start_sec": round(m["start_sec"] * tempo_scaling, 4),
                            "end_sec": round(m["end_sec"] * tempo_scaling, 4),
                        })
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

    # Check available soundfonts (train and test separately)
    available_train_soundfonts = [sf for sf in train_soundfonts if (soundfont_dir / sf).exists()]
    available_test_soundfonts = [sf for sf in test_soundfonts if (soundfont_dir / sf).exists()]

    if not available_train_soundfonts:
        logger.error(f"No train soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {train_soundfonts}")
        return {}

    if not available_test_soundfonts:
        logger.error(f"No test soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {test_soundfonts}")
        return {}

    logger.info(f"Available train soundfonts: {available_train_soundfonts}")
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
        # Use train soundfonts for train, test soundfonts for valid/test
        soundfonts_for_split = available_train_soundfonts if split == "train" else available_test_soundfonts
        tasks.append((kern_path, output_dir, soundfont_dir, split, n_versions, soundfonts_for_split,
                      transpose_enabled, feasible_transposes, tempo_enabled, tempo_range, musesyn_dir))

    logger.info(f"Processing {len(tasks)} kern files with {workers} workers...")

    all_results = {}

    # Load metadata once at the beginning for incremental updates
    metadata_path = output_dir / "augmentation_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata, starting fresh: {e}")

    updated_count = 0
    save_interval = 10  # Save metadata every N kern files

    def update_metadata_incremental(alignment: dict) -> int:
        """Update metadata in memory with alignment info. Returns count of updates."""
        count = 0
        for audio_key, align_info in alignment.items():
            if audio_key in metadata:
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
                    logger.error(f"Error processing {kern_path}: {e}")
                    all_results[kern_path.stem] = f"error: {e}"

    # Final save
    save_metadata()
    logger.info(f"Updated {updated_count} entries in {metadata_path}")

    # Summary
    success = sum(1 for v in all_results.values() if v == "success")
    skipped = sum(1 for v in all_results.values() if v.startswith("skipped"))
    errors = sum(1 for v in all_results.values() if v.startswith("error"))
    logger.info(f"Phase 2 complete: {success} success, {skipped} skipped, {errors} errors")

    return all_results


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
        type=int,
        choices=[1, 2],
        help="Phase to run (1: Score→Kern, 2: Kern→Audio). If not specified, runs both.",
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

    args = parser.parse_args()

    # Run requested phases
    if args.phase is None or args.phase == 1:
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

        # Create ground truth kern files (Zeng vocab compatible)
        logger.info("=== Phase 1b: Creating Ground Truth Kern (Zeng Vocab) ===")
        gt_results = create_ground_truth_kern(
            output_dir=args.output_dir,
            seed=42,
        )
        gt_success = sum(1 for v in gt_results.values() if v == "success")
        logger.info(f"Phase 1b complete: {gt_success}/{len(gt_results)} successful")

    if args.phase is None or args.phase == 2:
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


if __name__ == "__main__":
    main()
