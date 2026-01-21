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

Usage:
    python -m src.preprocessing.prepare_zeng_pretrain --phase 1  # Score → Kern
    python -m src.preprocessing.prepare_zeng_pretrain --phase 2  # Kern → Audio
    python -m src.preprocessing.prepare_zeng_pretrain            # Full pipeline
"""

import logging
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.humsyn_processor import HumSynProcessor
from src.preprocessing.musesyn_processor import MuseSynProcessor

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HUMSYN_DIR = Path("data/datasets/HumSyn")
DEFAULT_MUSESYN_DIR = Path("data/datasets/MuseSyn")
DEFAULT_OUTPUT_DIR = Path("data/experiments/clef_piano_base")
DEFAULT_METADATA_DIR = Path("data/metadata")
DEFAULT_SOUNDFONT_DIR = Path("data/soundfonts/piano")

# Soundfont configurations
TRAIN_SOUNDFONTS = [
    "TimGM6mb.sf2",
    "FluidR3_GM.sf2",
    "UprightPianoKW-20220221.sf2",
    "SalamanderGrandPiano-V3+20200602.sf2",
]
TEST_SOUNDFONTS = ["SalamanderGrandPiano-V3+20200602.sf2"]

# Transposition table (key signature → valid transpositions)
# From Zeng et al.: avoid going beyond ±6 semitones from original key
FEASIBLE_TRANSPOSES = {
    -6: [0, -1, -2, -3, 2, 3],       # avoid going more negative
    -5: [0, -1, -2, -3, 2, 3],
    -4: [0, -1, -2, -3, 2, 3, 4],
    -3: [0, -1, -2, -3, 2, 3, 4],
    -2: [0, -1, -2, -3, -4, 2, 3, 4],
    -1: [0, -1, -2, -3, -4, 1, 2, 3, 4],
    0:  [0, -1, -2, -3, -4, 1, 2, 3, 4],
    1:  [0, -1, -2, -3, -4, 1, 2, 3, 4],
    2:  [0, -1, -2, -3, -4, 1, 2, 3, 4],
    3:  [0, -2, -3, -4, 1, 2, 3, 4],
    4:  [0, -2, -3, -4, 1, 2, 3],
    5:  [0, -2, -4, 1, 2, 3],
    6:  [0, -2, -4, 1, 3],
    7:  [0, -2, -4, 1, 3],
}


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


def kern_to_midi(kern_path: Path, midi_path: Path, use_epr: bool = False) -> bool:
    """Convert kern file to MIDI.

    Args:
        kern_path: Path to input kern file
        midi_path: Path to output MIDI file
        use_epr: Whether to use VirtuosoNet for expressive rendering

    Returns:
        True if successful, False otherwise
    """
    try:
        if use_epr:
            # TODO: Implement VirtuosoNet EPR
            # For now, fall back to score-based rendering
            logger.warning("VirtuosoNet EPR not implemented, using score rendering")

        # Use hum2mid for kern → MIDI conversion
        result = subprocess.run(
            ["hum2mid", str(kern_path), "-o", str(midi_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"hum2mid failed for {kern_path}: {result.stderr}")
            return False

        return True

    except FileNotFoundError:
        logger.error("hum2mid not found. Install Humdrum Extras.")
        return False
    except Exception as e:
        logger.error(f"Error converting {kern_path} to MIDI: {e}")
        return False


def phase2_kern_to_audio(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    soundfont_dir: Path = DEFAULT_SOUNDFONT_DIR,
    num_augmentations: int = 10,
    use_epr: bool = False,
) -> Dict[str, str]:
    """Phase 2: Convert kern files to audio with augmentation.

    Args:
        output_dir: Path to output directory (contains kern/)
        metadata_dir: Path to metadata directory
        soundfont_dir: Path to soundfont directory
        num_augmentations: Number of augmentation versions for training data
        use_epr: Whether to use VirtuosoNet for expressive rendering

    Returns:
        Dictionary of {filename: status}
    """
    # Import audio synthesis utilities
    try:
        from midi2audio import FluidSynth
        from src.audio.zeng_synthesis import MIDIProcess, render_one_midi, create_default_compressor
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        return {}

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
    available_soundfonts = []
    for sf_name in TRAIN_SOUNDFONTS:
        sf_path = soundfont_dir / sf_name
        if sf_path.exists():
            available_soundfonts.append(sf_name)

    if not available_soundfonts:
        logger.error(f"No soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {TRAIN_SOUNDFONTS}")
        return {}

    logger.info(f"Available soundfonts: {available_soundfonts}")

    results = {}
    kern_files = sorted(kern_dir.glob("*.krn"))

    for kern_path in tqdm(kern_files, desc="Processing kern files"):
        # Determine split based on filename
        # Extract base name without prefix (e.g., "beethoven#sonata01-1" from "beethoven_piano_sonatas_sonata01-1.krn")
        stem = kern_path.stem

        # Check against split files
        # The split files use format like "beethoven#sonata01-1"
        # Our files use format like "beethoven_piano_sonatas_sonata01-1"
        # Need to map between them

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

        # Number of versions to generate
        n_versions = num_augmentations if split == "train" else 1

        for version in range(n_versions):
            # Generate augmentation parameters for training
            if split == "train":
                transpose = random.choice(list(range(-4, 5)))  # ±4 semitones
                tempo_scale = random.uniform(0.85, 1.15)
                soundfont = random.choice(available_soundfonts)
            else:
                transpose = 0
                tempo_scale = 1.0
                soundfont = available_soundfonts[0]  # Use first available

            # Output paths
            version_suffix = f"_v{version}" if n_versions > 1 else ""
            midi_name = f"{stem}{version_suffix}.mid"
            audio_name = f"{stem}{version_suffix}~{soundfont[:-4]}.wav"

            midi_path = midi_dir / midi_name
            audio_path = audio_dir / audio_name

            try:
                # Convert kern → MIDI
                if not kern_to_midi(kern_path, midi_path, use_epr=use_epr):
                    results[audio_name] = "error: kern to midi failed"
                    continue

                # Apply MIDI processing (tempo scaling, cut blank/pedal)
                midi_proc = MIDIProcess(str(midi_path), split)
                scaling, original_length = midi_proc.process(
                    str(midi_path),
                    str(midi_dir / "temp.mid"),
                )

                if scaling is None:
                    results[audio_name] = "error: midi processing failed"
                    continue

                # Render MIDI → Audio
                sf_path = soundfont_dir / soundfont
                fs = FluidSynth(str(sf_path), sample_rate=44100)
                compressor = create_default_compressor()

                render_one_midi(fs, compressor, str(midi_path), str(audio_path))
                results[audio_name] = "success"

            except Exception as e:
                logger.error(f"Error processing {kern_path}: {e}")
                results[audio_name] = f"error: {e}"

    # Summary
    success = sum(1 for v in results.values() if v == "success")
    errors = sum(1 for v in results.values() if v.startswith("error"))
    logger.info(f"Phase 2 complete: {success} success, {errors} errors")

    return results


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
        "--num-augmentations",
        type=int,
        default=10,
        help="Number of augmentation versions for training",
    )
    parser.add_argument(
        "--use-epr",
        action="store_true",
        help="Use VirtuosoNet for expressive performance rendering",
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
            num_augmentations=args.num_augmentations,
            use_epr=args.use_epr,
        )
        success = sum(1 for v in phase2_results.values() if v == "success")
        logger.info(f"Phase 2 complete: {success}/{len(phase2_results)} successful")


if __name__ == "__main__":
    main()
