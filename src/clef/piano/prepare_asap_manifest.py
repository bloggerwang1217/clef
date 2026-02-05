"""
ASAP Test Set Manifest Builder
==============================

Builds a manifest JSON for ASAP test set audio files, compatible with
clef_piano_inference.py. Also converts audio files to log-mel spectrograms.

Pipeline:
1. Scan ASAP test set directory for performance .wav files
2. Convert each .wav to log-mel spectrogram (.pt)
3. Build manifest JSON with same format as syn test_manifest.json

ASAP test set structure:
    data/datasets/asap_test_set/
    ├── test_asap.txt              <- piece list (Zeng-style piece_id)
    ├── Composer/
    │   ├── Work/
    │   │   ├── Piece/
    │   │   │   ├── midi_score.mid     <- ground truth
    │   │   │   ├── xml_score.musicxml
    │   │   │   ├── Performance1.wav   <- performance audio
    │   │   │   ├── Performance1.mid   <- performance MIDI
    │   │   │   └── ...

Output manifest entry format:
    {
        "id": "Bach__Prelude__bwv_875__Ahfat01M",
        "mel_path": "mel/Bach__Prelude__bwv_875__Ahfat01M.pt",
        "audio_path": "Bach/Prelude/bwv_875/Ahfat01M.wav",
        "gt_midi_path": "Bach/Prelude/bwv_875/midi_score.mid",
        "piece_id": "Bach#Prelude#bwv_875",
        "performance_id": "Ahfat01M",
        "duration_sec": 123.4,
        "n_frames": 12340,
        "split": "test"
    }

ID convention uses double underscore (__) as hierarchy separator to avoid
ambiguity with underscores in piece names (e.g., "bwv_875", "op.90_D.899").

Usage:
    # Scan + convert mel + build manifest
    python -m src.clef.piano.prepare_asap_manifest

    # Custom paths
    python -m src.clef.piano.prepare_asap_manifest \
        --asap-dir data/datasets/asap_test_set \
        --output-dir data/experiments/clef_asap_test \
        --config configs/clef_piano_base.yaml

    # Skip mel conversion (manifest only)
    python -m src.clef.piano.prepare_asap_manifest --skip-mel

    # Parallel mel conversion
    python -m src.clef.piano.prepare_asap_manifest --workers 8
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from tqdm import tqdm

from src.audio.mel import load_mel_config, process_audio_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Files to skip when scanning for performance audio
SKIP_FILES = {"midi_score", "xml_score", "midi_score_annotations"}


def scan_asap_test_set(asap_dir: Path) -> List[Dict[str, Any]]:
    """Scan ASAP test set directory for performance audio files.

    Reads test_asap.txt for the piece list, then finds all .wav files
    under each piece directory.

    Args:
        asap_dir: Path to ASAP test set directory

    Returns:
        List of dicts with keys: piece_id, performance_id, audio_rel_path,
        gt_midi_rel_path, composer, work, piece
    """
    test_list_path = asap_dir / "test_asap.txt"
    if not test_list_path.exists():
        raise FileNotFoundError(f"test_asap.txt not found in {asap_dir}")

    # Read piece list (Zeng-style: Bach#Prelude#bwv_875)
    with open(test_list_path) as f:
        lines = f.read().strip().split("\n")

    # Skip header if present
    piece_ids = []
    for line in lines:
        line = line.strip()
        if not line or line == "name":
            continue
        piece_ids.append(line)

    logger.info(f"Found {len(piece_ids)} pieces in test_asap.txt")

    entries = []

    for piece_id in piece_ids:
        # Convert piece_id to path: Bach#Prelude#bwv_875 -> Bach/Prelude/bwv_875
        path_parts = piece_id.split("#")
        piece_dir = asap_dir / "/".join(path_parts)

        if not piece_dir.exists():
            logger.warning(f"Piece directory not found: {piece_dir}")
            continue

        # Find ground truth MIDI
        gt_midi_rel = None
        for gt_name in ["midi_score.mid", "midi_score.midi"]:
            if (piece_dir / gt_name).exists():
                gt_midi_rel = str(Path("/".join(path_parts)) / gt_name)
                break

        if gt_midi_rel is None:
            logger.warning(f"No ground truth MIDI found for {piece_id}")
            continue

        # Find all performance .wav files
        wav_files = sorted(piece_dir.glob("*.wav"))
        if not wav_files:
            logger.warning(f"No .wav files found for {piece_id}")
            continue

        for wav_path in wav_files:
            perf_id = wav_path.stem

            # Skip non-performance files
            if perf_id in SKIP_FILES:
                continue

            audio_rel = str(Path("/".join(path_parts)) / wav_path.name)

            entries.append({
                "piece_id": piece_id,
                "performance_id": perf_id,
                "audio_rel_path": audio_rel,
                "gt_midi_rel_path": gt_midi_rel,
                "composer": path_parts[0],
                "work": path_parts[1] if len(path_parts) > 1 else "",
                "piece": path_parts[2] if len(path_parts) > 2 else "",
            })

    logger.info(f"Found {len(entries)} performance audio files across {len(piece_ids)} pieces")
    return entries


def _convert_one_mel(
    audio_path: str,
    mel_path: str,
    mel_config: Dict[str, Any],
) -> Tuple[str, str, Optional[Tuple[int, ...]]]:
    """Worker function for parallel mel conversion.

    Returns:
        (audio_path, status, mel_shape)
    """
    status, shape = process_audio_file(
        audio_path=audio_path,
        mel_path=mel_path,
        target_sample_rate=mel_config["sample_rate"],
        n_mels=mel_config["n_mels"],
        n_fft=mel_config["n_fft"],
        hop_length=mel_config["hop_length"],
        f_min=mel_config["f_min"],
        f_max=mel_config["f_max"],
        normalize=True,
        skip_existing=True,
    )
    return audio_path, status, shape


def convert_mels(
    entries: List[Dict[str, Any]],
    asap_dir: Path,
    output_dir: Path,
    mel_config: Dict[str, Any],
    workers: int = 1,
) -> Dict[str, Tuple[float, int]]:
    """Convert all audio files to mel spectrograms.

    Args:
        entries: List of scan entries
        asap_dir: ASAP test set directory
        output_dir: Output directory (mel files go to output_dir/mel/)
        mel_config: Mel spectrogram parameters
        workers: Number of parallel workers

    Returns:
        Dict mapping entry ID to (duration_sec, n_frames)
    """
    mel_dir = output_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for entry in entries:
        entry_id = make_entry_id(entry)
        audio_path = str(asap_dir / entry["audio_rel_path"])
        mel_path = str(mel_dir / f"{entry_id}.pt")
        tasks.append((audio_path, mel_path, entry_id))

    logger.info(f"Converting {len(tasks)} audio files to mel spectrograms")

    results = {}
    stats = {"generated": 0, "skipped": 0, "missing": 0, "failed": 0}

    if workers <= 1:
        # Sequential
        for audio_path, mel_path, entry_id in tqdm(tasks, desc="Converting mel"):
            _, status, shape = _convert_one_mel(audio_path, mel_path, mel_config)
            stats[status] += 1
            if status in ("generated", "skipped"):
                dur, frames = _get_mel_info(mel_path, mel_config)
                results[entry_id] = (dur, frames)
    else:
        # Parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for audio_path, mel_path, entry_id in tasks:
                future = executor.submit(_convert_one_mel, audio_path, mel_path, mel_config)
                futures[future] = (mel_path, entry_id)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting mel"):
                mel_path, entry_id = futures[future]
                _, status, shape = future.result()
                stats[status] += 1
                if status in ("generated", "skipped"):
                    dur, frames = _get_mel_info(mel_path, mel_config)
                    results[entry_id] = (dur, frames)

    logger.info(
        f"Mel conversion: {stats['generated']} generated, {stats['skipped']} skipped, "
        f"{stats['missing']} missing, {stats['failed']} failed"
    )
    return results


def _get_mel_info(mel_path: str, mel_config: Dict[str, Any]) -> Tuple[float, int]:
    """Get duration and frame count from a mel .pt file."""
    mel = torch.load(mel_path, map_location="cpu", weights_only=True)
    n_frames = mel.shape[-1]
    duration_sec = n_frames * mel_config["hop_length"] / mel_config["sample_rate"]
    return round(duration_sec, 4), n_frames


def make_entry_id(entry: Dict[str, Any]) -> str:
    """Build manifest entry ID from ASAP path components.

    Uses double underscore (__) as hierarchy separator to avoid
    ambiguity with underscores in ASAP names (e.g., bwv_875).

    Example:
        Bach#Prelude#bwv_875 + Ahfat01M -> Bach__Prelude__bwv_875__Ahfat01M
    """
    parts = entry["piece_id"].split("#")
    parts.append(entry["performance_id"])
    return "__".join(parts)


def build_manifest(
    entries: List[Dict[str, Any]],
    mel_info: Dict[str, Tuple[float, int]],
) -> List[Dict[str, Any]]:
    """Build manifest JSON from scan entries and mel info.

    Args:
        entries: List of scan entries from scan_asap_test_set()
        mel_info: Dict mapping entry_id to (duration_sec, n_frames)

    Returns:
        Manifest as list of dicts
    """
    manifest = []

    for entry in entries:
        entry_id = make_entry_id(entry)

        if entry_id not in mel_info:
            logger.warning(f"Skipping {entry_id}: mel conversion failed")
            continue

        duration_sec, n_frames = mel_info[entry_id]

        manifest.append({
            "id": entry_id,
            "mel_path": f"mel/{entry_id}.pt",
            "audio_path": entry["audio_rel_path"],
            "gt_midi_path": entry["gt_midi_rel_path"],
            "piece_id": entry["piece_id"],
            "performance_id": entry["performance_id"],
            "duration_sec": duration_sec,
            "n_frames": n_frames,
            "split": "test",
        })

    # Sort by ID for reproducibility
    manifest.sort(key=lambda x: x["id"])
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build ASAP test set manifest for clef-piano inference"
    )
    parser.add_argument(
        "--asap-dir",
        type=str,
        default="data/datasets/asap_test_set",
        help="Path to ASAP test set directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments/clef_asap_test",
        help="Output directory for mel files and manifest",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/clef_piano_base.yaml",
        help="Config file for mel parameters",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers for mel conversion",
    )
    parser.add_argument(
        "--skip-mel",
        action="store_true",
        help="Skip mel conversion (assume mel files already exist)",
    )
    args = parser.parse_args()

    asap_dir = Path(args.asap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mel config
    mel_config = load_mel_config(args.config)
    logger.info(f"Mel config: {mel_config}")

    # Step 1: Scan ASAP test set
    entries = scan_asap_test_set(asap_dir)

    if not entries:
        logger.error("No audio files found. Check --asap-dir path.")
        return

    # Step 2: Convert audio to mel spectrograms
    if args.skip_mel:
        logger.info("Skipping mel conversion (--skip-mel)")
        # Load mel info from existing files
        mel_info = {}
        mel_dir = output_dir / "mel"
        for entry in entries:
            entry_id = make_entry_id(entry)
            mel_path = mel_dir / f"{entry_id}.pt"
            if mel_path.exists():
                dur, frames = _get_mel_info(str(mel_path), mel_config)
                mel_info[entry_id] = (dur, frames)
            else:
                logger.warning(f"Mel file not found: {mel_path}")
    else:
        mel_info = convert_mels(entries, asap_dir, output_dir, mel_config, args.workers)

    # Step 3: Build manifest
    manifest = build_manifest(entries, mel_info)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest saved: {manifest_path} ({len(manifest)} entries)")

    # Summary
    pieces = set(e["piece_id"] for e in manifest)
    logger.info(f"Summary: {len(manifest)} performances across {len(pieces)} pieces")

    # Per-piece counts
    from collections import Counter
    piece_counts = Counter(e["piece_id"] for e in manifest)
    for piece_id, count in sorted(piece_counts.items()):
        logger.info(f"  {piece_id}: {count} performances")


if __name__ == "__main__":
    main()
