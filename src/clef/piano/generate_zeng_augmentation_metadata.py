"""
Generate Augmentation Metadata
==============================

Reproduces the random augmentation choices made during Phase 2 audio generation.
Since seeds are deterministic, we can recreate the exact transpose intervals
without re-parsing any files.

Output: data/experiments/clef_piano_base/augmentation_metadata.json
{
    "musesyn_SongName_v0~SoundFont.wav": {
        "transpose": "M2",      # or 0 for no transpose
        "soundfont": "FluidR3_GM.sf2",
        "split": "train"
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
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.utils import set_seed, SEED_DATA_AUGMENTATION

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/experiments/clef_piano_base")
DEFAULT_METADATA_DIR = Path("data/metadata")
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

        # Get key signature
        original_key = get_key_signature_from_kern(kern_path)
        original_key = max(-6, min(7, original_key))  # Clamp to valid range

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
            version_suffix = f"_v{version}" if n_versions > 1 else ""
            audio_name = f"{stem}{version_suffix}~{soundfont[:-4]}.wav"

            metadata[audio_name] = {
                "kern_file": f"{stem}.krn",
                "transpose": transpose,
                "soundfont": soundfont,
                "split": split,
                "version": version,
                "original_key": original_key,
                "tempo_augmented": tempo_enabled and split == "train",
                "tempo_range": list(tempo_range) if (tempo_enabled and split == "train") else None,
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

    for info in metadata.values():
        split = info["split"]
        splits[split] = splits.get(split, 0) + 1

        if info["transpose"] == 0:
            transpose_counts["0"] += 1
        else:
            transpose_counts["transposed"] += 1

    logger.info(f"Generated metadata for {len(metadata)} audio files")
    logger.info(f"Splits: {splits}")
    logger.info(f"Transpose: {transpose_counts}")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
