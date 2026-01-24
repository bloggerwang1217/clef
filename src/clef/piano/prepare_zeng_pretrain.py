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
from src.utils import set_seed, SEED_DATA_AUGMENTATION

# Register converter21 for robust humdrum parsing
converter21.register()

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HUMSYN_DIR = Path("data/datasets/HumSyn")
DEFAULT_MUSESYN_DIR = Path("data/datasets/MuseSyn")
DEFAULT_OUTPUT_DIR = Path("data/experiments/clef_piano_base")
DEFAULT_METADATA_DIR = Path("data/metadata")
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


def _process_single_kern(args: Tuple) -> Dict[str, str]:
    """Worker function for parallel processing a single kern file.

    Args:
        args: Tuple of (kern_path, output_dir, soundfont_dir, split, num_versions,
                        available_soundfonts, transpose_enabled, feasible_transposes,
                        tempo_enabled, tempo_range, musesyn_dir)

    Returns:
        Dictionary of {audio_name: status}
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

    from src.score.kern_repeat import expand_kern_repeats
    from src.score.sanitize_piano_score import sanitize_score

    midi_dir = output_dir / "midi"
    audio_dir = output_dir / "audio"
    stem = kern_path.stem
    results = {}

    # Set reproducible seed based on filename
    # NOTE: Use hashlib.md5 instead of hash() because Python's hash() is
    # randomized across interpreter sessions (PYTHONHASHSEED). This ensures
    # reproducible augmentation choices across multiple runs.
    file_seed = int(hashlib.md5(stem.encode()).hexdigest(), 16) % (2**32)
    set_seed(SEED_DATA_AUGMENTATION + file_seed)

    # Check if this is a MuseSyn file
    is_musesyn = stem.startswith("musesyn_")

    try:
        if is_musesyn:
            # For MuseSyn: use original XML + sanitize_score
            # This avoids rhythm inconsistency issues in kern parsing
            original_name = stem[8:]  # Remove "musesyn_" prefix
            xml_path = musesyn_dir / "xml" / f"{original_name}.xml"

            if not xml_path.exists():
                return {f"{stem}_v0": f"error: XML not found - {xml_path}"}

            score = m21.converter.parse(str(xml_path))
            sanitize_score(score)
        else:
            # For HumSyn: use kern file with repeat expansion
            # Original kern with repeat markers is preserved as ground truth
            with open(kern_path, "r", encoding="utf-8") as f:
                kern_content = f.read()

            # Expand repeats for audio generation
            expanded_kern = expand_kern_repeats(kern_content)

            # Write expanded kern to temp file for music21 parsing
            with tempfile.NamedTemporaryFile(mode="w", suffix=".krn", delete=False) as tmp:
                tmp.write(expanded_kern)
                expanded_kern_path = tmp.name

            score = m21.converter.parse(expanded_kern_path, format="humdrum")

            # Clean up temp file
            Path(expanded_kern_path).unlink(missing_ok=True)

    except Exception as e:
        return {f"{stem}_v0": f"error: parse failed - {e}"}

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

        if audio_path.exists():
            results[audio_name] = "skipped (exists)"
            continue

        try:
            # Apply transpose if needed
            if transpose != 0:
                transposed_score = score.transpose(transpose)
            else:
                transposed_score = score

            # Write MIDI
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            transposed_score.write("midi", fp=str(midi_path))

            if not midi_path.exists():
                results[audio_name] = "error: midi write failed"
                continue

            # Apply tempo scaling via MIDIProcess (Zeng-style)
            if has_zeng and tempo_enabled:
                midi_proc = MIDIProcess(str(midi_path), split)
                scaling, original_length = midi_proc.process(
                    str(midi_path),
                    str(midi_dir / f"temp_{stem}_{version}.mid"),
                    tempo_range=tempo_range,
                )

            # Render MIDI -> Audio with loudness normalization
            sf_path = soundfont_dir / soundfont
            fs = FluidSynth(str(sf_path), sample_rate=44100)

            if has_zeng:
                compressor = create_default_compressor()
                render_one_midi(fs, compressor, str(midi_path), str(audio_path))
            else:
                fs.midi_to_audio(str(midi_path), str(audio_path))

            if audio_path.exists():
                results[audio_name] = "success"
            else:
                results[audio_name] = "error: audio not created"

        except Exception as e:
            results[audio_name] = f"error: {e}"

    return results


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

    # Check available soundfonts
    available_soundfonts = [sf for sf in train_soundfonts if (soundfont_dir / sf).exists()]

    if not available_soundfonts:
        logger.error(f"No soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {train_soundfonts}")
        return {}

    logger.info(f"Available soundfonts: {available_soundfonts}")

    # Prepare task list
    kern_files = sorted(kern_dir.glob("*.krn"))
    tasks = []

    for kern_path in kern_files:
        stem = kern_path.stem

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
        tasks.append((kern_path, output_dir, soundfont_dir, split, n_versions, available_soundfonts,
                      transpose_enabled, feasible_transposes, tempo_enabled, tempo_range, musesyn_dir))

    logger.info(f"Processing {len(tasks)} kern files with {workers} workers...")

    all_results = {}

    if workers == 1:
        # Sequential processing
        for task in tqdm(tasks, desc="Processing kern files"):
            results = _process_single_kern(task)
            all_results.update(results)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_single_kern, task): task[0] for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing kern files"):
                kern_path = futures[future]
                try:
                    results = future.result()
                    all_results.update(results)
                except Exception as e:
                    logger.error(f"Error processing {kern_path}: {e}")
                    all_results[kern_path.stem] = f"error: {e}"

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
