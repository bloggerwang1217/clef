"""
Generate Augmentation Metadata
==============================

Generates metadata for Phase 2 audio synthesis, including:
1. Augmentation settings (transpose, soundfont, tempo_range)
2. Kern measure boundaries (for chunking during training)

The alignment timing (audio_measures, tempo_scaling, duration_sec) will be
added by prepare_zeng_pretrain.py during Phase 2 audio synthesis.

Output: data/experiments/clef_piano_base/augmentation_metadata.json
{
    "beethoven_sonatas_01-1_v0~FluidR3_GM": {
        "kern_file": "beethoven_sonatas_01-1.krn",
        "soundfont": "FluidR3_GM.sf2",
        "split": "train",
        "transpose": 0,
        "tempo_range": [0.85, 1.15],
        "kern_measures": [
            {"measure": 1, "line_start": 25, "line_end": 29},
            {"measure": 2, "line_start": 30, "line_end": 38},
            ...
        ],
        # === Added by Phase 2 ===
        "tempo_scaling": 0.923,
        "duration_sec": 195.3,
        "audio_measures": [
            {"measure": 1, "start_sec": 0.0, "end_sec": 2.71},
            ...
        ]
    },
    ...
}

Usage:
    python -m src.clef.piano.generate_zeng_augmentation_metadata
"""

import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from src.utils import set_seed, SEED_DATA_AUGMENTATION

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/experiments/clef_piano_base")
DEFAULT_METADATA_DIR = Path("src/datasets/syn")
DEFAULT_AUG_CONFIG = Path("configs/zeng_augmentation.json")


def load_augmentation_config(config_path: Path = DEFAULT_AUG_CONFIG) -> Dict[str, Any]:
    """Load augmentation settings from JSON config.

    NOTE: Transpose is disabled for clef-piano-base because:
    1. Piano voicing is key-specific - transposing disrupts idiomatic patterns
    2. Zeng's original implementation had a multiprocessing bug causing only ~2 unique transposes
    3. Real recordings (ASAP) are not transposed, improving evaluation fairness
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Check if transpose is enabled
    transpose_enabled = config['transpose'].get('enabled', False)

    # Convert string keys to int for feasible_transposes
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


def _match_split_name(processed_name: str, split_name: str) -> bool:
    """Check if a processed filename matches a split file entry."""
    if processed_name == split_name:
        return True

    if "#" in split_name:
        prefix, piece = split_name.split("#", 1)
        prefix_map = {
            "beethoven": "beethoven_piano_sonatas",
            "haydn": "haydn_piano_sonatas",
            "mozart": "mozart_piano_sonatas",
            "chopin": "humdrum_chopin_first_editions",
            "joplin": "joplin",
            "scarlatti": "scarlatti_keyboard_sonatas",
        }
        if prefix in prefix_map:
            expected_name = f"{prefix_map[prefix]}_{piece}"
            if processed_name == expected_name:
                return True

    if processed_name.startswith("musesyn_"):
        musesyn_name = processed_name[8:]
        if musesyn_name == split_name:
            return True

    return False


def get_key_signature_from_kern(kern_path: Path) -> int:
    """Extract key signature from kern file without full parsing.

    Looks for *k[...] interpretation token.
    Returns number of sharps (negative for flats).
    """
    with open(kern_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('*k['):
                # Parse key signature: *k[f#c#] = 2 sharps, *k[b-e-] = 2 flats
                key_str = line.split('\t')[0]  # Take first spine
                if '*k[' in key_str:
                    inner = key_str[key_str.index('*k[')+3:key_str.index(']')]
                    if not inner:
                        return 0  # C major
                    if '-' in inner:
                        # Flats: count number of note names
                        return -len(inner.replace('-', '')) // 1
                    elif '#' in inner:
                        # Sharps
                        return len(inner.replace('#', ''))
                    else:
                        return 0
    return 0  # Default to C major


def extract_kern_measures(kern_path: Path) -> List[Dict[str, Any]]:
    """Extract measure boundaries from kern file.

    Parses barlines (=N, =N-, =N:|!, etc.) to find measure numbers and line ranges.
    Used for chunking during training - we cut at measure boundaries.

    Args:
        kern_path: Path to kern file

    Returns:
        List of measure info dicts:
        [
            {"measure": 1, "line_start": 25, "line_end": 29},
            {"measure": 2, "line_start": 30, "line_end": 38},
            ...
        ]
        line_start is the first data line of the measure (after the barline)
        line_end is the last data line before the next barline (or end of file)
    """
    measures = []
    current_measure = None
    measure_start_line = None

    # Pattern to match barlines: =N, =N-, =N:|!, ==, etc.
    # Captures the measure number if present
    barline_pattern = re.compile(r'^=(\d+)?')

    with open(kern_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('!'):
            continue

        # Check for barline
        first_token = line.split('\t')[0]
        match = barline_pattern.match(first_token)

        if match:
            # Found a barline
            measure_num_str = match.group(1)

            # Close previous measure
            if current_measure is not None and measure_start_line is not None:
                measures.append({
                    "measure": current_measure,
                    "line_start": measure_start_line,
                    "line_end": line_num - 1,  # End before this barline
                })

            # Start new measure
            if measure_num_str:
                current_measure = int(measure_num_str)
            elif current_measure is not None:
                # Barline without number (e.g., == for final barline)
                current_measure += 1
            else:
                current_measure = 1

            measure_start_line = line_num + 1  # Start after barline

        elif first_token.startswith('*'):
            # Interpretation line (metadata) - not part of measure content
            # But update measure_start_line if we haven't started the measure yet
            if measure_start_line == line_num:
                measure_start_line = line_num + 1
            continue

    # Close the last measure
    if current_measure is not None and measure_start_line is not None:
        # Find the last non-empty, non-comment, non-spine-terminator line
        last_data_line = len(lines)
        for i in range(len(lines) - 1, measure_start_line - 2, -1):
            line = lines[i].strip()
            if line and not line.startswith('!') and not line.startswith('*-'):
                last_data_line = i + 1
                break

        if last_data_line >= measure_start_line:
            measures.append({
                "measure": current_measure,
                "line_start": measure_start_line,
                "line_end": last_data_line,
            })

    return measures


def generate_metadata(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    aug_config_path: Path = DEFAULT_AUG_CONFIG,
) -> Dict[str, Any]:
    """Generate augmentation metadata for all kern files.

    Reproduces the exact random choices made during Phase 2.
    """
    # Load augmentation config from JSON
    aug_config = load_augmentation_config(aug_config_path)
    transpose_enabled = aug_config['transpose_enabled']
    feasible_transposes = aug_config['feasible_transposes']
    tempo_enabled = aug_config['tempo_enabled']
    tempo_range = aug_config['tempo_range']
    train_soundfonts = aug_config['train_soundfonts']
    test_soundfonts = aug_config['test_soundfonts']
    num_versions = aug_config['num_versions']

    if not transpose_enabled:
        logger.info("Transpose augmentation DISABLED (preserves piano voicing patterns)")
    if tempo_enabled:
        logger.info(f"Tempo augmentation ENABLED (range: {tempo_range[0]:.2f}-{tempo_range[1]:.2f})")
    else:
        logger.info("Tempo augmentation DISABLED")

    kern_dir = output_dir / "kern"

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

    kern_files = sorted(kern_dir.glob("*.krn"))
    metadata = {}

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

    # Cache for kern_measures (shared across versions of same kern file)
    kern_measures_cache = {}

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

        # Get key signature
        original_key = get_key_signature_from_kern(kern_path)
        original_key = max(-6, min(7, original_key))  # Clamp to valid range

        # Extract kern measures (cached - same for all versions)
        if stem not in kern_measures_cache:
            kern_measures_cache[stem] = extract_kern_measures(kern_path)
        kern_measures = kern_measures_cache[stem]

        # Reproduce random seed (same as _process_single_kern)
        # Use hashlib.md5 for deterministic hashing across Python sessions
        file_seed = int(hashlib.md5(stem.encode()).hexdigest(), 16) % (2**32)
        set_seed(SEED_DATA_AUGMENTATION + file_seed)

        n_versions = num_versions[split]

        for version in range(n_versions):
            # Reproduce random choices (same order as _process_single_kern)
            if split == "train":
                # Transpose is disabled - preserves piano voicing patterns
                if transpose_enabled and feasible_transposes:
                    transpose = random.choice(feasible_transposes[original_key])
                else:
                    transpose = 0
                # Soundfont selection: deterministic mapping (version -> soundfont)
                # Zeng original: random.choice() which causes duplicate soundfonts
                # Our choice: version % len(soundfonts) ensures each soundfont appears once
                soundfont = train_soundfonts[version % len(train_soundfonts)]
            else:
                transpose = 0
                soundfont = test_soundfonts[0]

            # Build audio name (same as _process_single_kern)
            # Note: key is without .wav extension for easier lookup
            version_suffix = f"_v{version}" if n_versions > 1 else ""
            audio_key = f"{stem}{version_suffix}~{soundfont[:-4]}"

            metadata[audio_key] = {
                "kern_file": f"{stem}.krn",
                "transpose": transpose,
                "soundfont": soundfont,
                "split": split,
                "version": version,
                "original_key": original_key,
                "tempo_augmented": tempo_enabled and split == "train",
                "tempo_range": list(tempo_range) if (tempo_enabled and split == "train") else None,
                "kern_measures": kern_measures,
                # === To be filled by Phase 2 ===
                "tempo_scaling": None,
                "duration_sec": None,
                "audio_measures": None,
            }

    return metadata


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate augmentation metadata")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (contains kern/)",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Metadata directory (contains split files)",
    )
    parser.add_argument(
        "--aug-config",
        type=Path,
        default=DEFAULT_AUG_CONFIG,
        help="Augmentation config JSON file",
    )

    args = parser.parse_args()

    logger.info("Generating augmentation metadata...")

    metadata = generate_metadata(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        aug_config_path=args.aug_config,
    )

    # Save metadata
    output_path = args.output_dir / "augmentation_metadata.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    splits = {}
    transpose_counts = {"0": 0, "transposed": 0}
    total_measures = 0
    unique_kern_files = set()

    for info in metadata.values():
        split = info["split"]
        splits[split] = splits.get(split, 0) + 1

        if info["transpose"] == 0:
            transpose_counts["0"] += 1
        else:
            transpose_counts["transposed"] += 1

        if info["kern_file"] not in unique_kern_files:
            unique_kern_files.add(info["kern_file"])
            total_measures += len(info["kern_measures"])

    logger.info(f"Generated metadata for {len(metadata)} audio entries")
    logger.info(f"  Unique kern files: {len(unique_kern_files)}")
    logger.info(f"  Total measures: {total_measures}")
    logger.info(f"  Splits: {splits}")
    logger.info(f"  Transpose: {transpose_counts}")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
