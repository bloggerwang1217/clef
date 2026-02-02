"""
Synthetic Dataset Manifest Builder
==================================

Creates train/valid/test manifest JSON files for synthetic datasets.
Manifests contain paths to mel spectrograms and ground truth kern files.

Usage:
    from src.datasets.syn.syn_manifest import create_manifest
    counts = create_manifest(data_dir, config_path)
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm

from src.audio.mel import load_mel_config

logger = logging.getLogger(__name__)


def create_manifest(
    data_dir: Union[str, Path],
    config_path: Union[str, Path] = "configs/clef_piano_base.yaml",
    metadata_filename: str = "augmentation_metadata.json",
    validate_files: bool = True,
    sample_validation_count: int = 5,
) -> Dict[str, int]:
    """Create train/valid/test manifest files from augmentation metadata.

    Reads augmentation_metadata.json and creates manifest JSON files
    for each split, containing paths to mel spectrograms and ground truth kern.

    Args:
        data_dir: Base data directory containing audio/, mel/, kern_gt/
        config_path: Path to model config YAML (for mel parameters)
        metadata_filename: Name of the metadata JSON file
        validate_files: Whether to validate existence of mel and kern_gt files
        sample_validation_count: Number of mel files to sample for n_frames validation

    Returns:
        Dictionary with counts per split: {"train": N, "valid": N, "test": N}

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    data_dir = Path(data_dir)
    config_path = Path(config_path)

    # Load mel parameters from config (for n_frames calculation)
    mel_config = load_mel_config(config_path)
    sample_rate = mel_config["sample_rate"]
    hop_length = mel_config["hop_length"]

    # Load metadata
    metadata_path = data_dir / metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    mel_dir = data_dir / "mel"
    kern_gt_dir = data_dir / "kern_gt"

    logger.info(f"Creating manifests from {len(metadata)} entries")
    logger.info(f"Config: {config_path}")

    manifests: Dict[str, List[Dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    missing_mel: List[str] = []
    missing_kern: set = set()
    metadata_updated = 0

    for key, entry in tqdm(metadata.items(), desc="Building manifest"):
        mel_path = mel_dir / f"{key}.pt"
        kern_gt_path = kern_gt_dir / entry["kern_file"]

        # Validate existence
        if validate_files:
            if not mel_path.exists():
                missing_mel.append(key)
                continue
            if not kern_gt_path.exists():
                missing_kern.add(entry["kern_file"])
                continue

        # Read n_frames from mel file (needed for tensor loading)
        mel = torch.load(mel_path, weights_only=True)
        n_frames = mel.shape[-1]

        # duration_sec comes from Phase 2 (Score-based timing x tempo_scaling).
        # Do NOT recalculate from mel frames — Score is the single source of truth
        # for timing alignment with audio_measures/kern_measures.
        duration_sec = entry.get("duration_sec")
        if duration_sec is None:
            # Fallback: Phase 2 didn't set it (shouldn't happen in normal flow)
            duration_sec = n_frames / (sample_rate / hop_length)
            logger.warning(f"{key}: duration_sec missing from Phase 2, using mel-based fallback")

        # Update n_frames in metadata (lightweight, no timing overwrite)
        if entry.get("n_frames") != n_frames:
            metadata[key]["n_frames"] = n_frames
            metadata_updated += 1

        manifests[entry["split"]].append({
            "id": key,
            "mel_path": f"mel/{key}.pt",
            "kern_gt_path": f"kern_gt/{entry['kern_file']}",
            "duration_sec": round(duration_sec, 4),
            "n_frames": n_frames,
            "split": entry["split"],
        })

    # Save updated metadata back to file
    if metadata_updated > 0:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated {metadata_updated} entries in {metadata_path}")

    # Report missing files (log all for debugging)
    if missing_mel:
        logger.error(f"Missing mel files: {len(missing_mel)}")
        for key in missing_mel:
            logger.error(f"  [MISSING_MEL] {key}")

    if missing_kern:
        logger.error(f"Missing kern_gt files: {len(missing_kern)}")
        for kern in sorted(missing_kern):
            logger.error(f"  [MISSING_KERN] {kern}")

    # Save manifests
    counts = {}
    for split, samples in manifests.items():
        manifest_path = data_dir / f"{split}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        counts[split] = len(samples)
        logger.info(f"  {split}: {len(samples)} samples -> {manifest_path}")

    # Validate n_frames by sampling mel files
    if sample_validation_count > 0 and mel_dir.exists():
        _validate_n_frames(manifests, mel_dir, sample_validation_count)

    # Final summary
    total = sum(counts.values())
    logger.info(f"Manifest creation complete: {total} total samples")
    logger.info(f"  Train: {counts.get('train', 0)}")
    logger.info(f"  Valid: {counts.get('valid', 0)}")
    logger.info(f"  Test:  {counts.get('test', 0)}")

    return counts


def _validate_n_frames(
    manifests: Dict[str, List[Dict]],
    mel_dir: Path,
    sample_count: int = 5,
) -> None:
    """Validate n_frames calculation by sampling mel files.

    Args:
        manifests: Dictionary of split -> list of manifest entries
        mel_dir: Path to mel directory
        sample_count: Number of samples to validate
    """
    logger.info(f"\n[Validation] Sampling {sample_count} mel files to verify n_frames...")

    all_entries = [e for entries in manifests.values() for e in entries]
    if not all_entries:
        logger.warning("No entries to validate")
        return

    sample_size = min(sample_count, len(all_entries))
    samples = random.sample(all_entries, sample_size)

    mismatches = 0
    for entry in samples:
        mel_path = mel_dir / f"{entry['id']}.pt"
        if not mel_path.exists():
            continue

        mel = torch.load(mel_path, weights_only=True)
        actual = mel.shape[-1]
        expected = entry["n_frames"]
        diff = abs(actual - expected)

        if diff > 1:
            status = "MISMATCH"
            mismatches += 1
        else:
            status = "OK"

        # Truncate ID for display
        display_id = entry["id"][:50] + "..." if len(entry["id"]) > 50 else entry["id"]
        logger.info(f"  {display_id}: expected={expected}, actual={actual} [{status}]")

    if mismatches > 0:
        logger.warning(f"Found {mismatches} n_frames mismatches!")


def load_manifest(
    manifest_path: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load a manifest JSON file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        List of manifest entries
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_manifest_stats(manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics from a manifest.

    Args:
        manifest: List of manifest entries

    Returns:
        Dictionary with statistics:
        - count: number of entries
        - total_duration_sec: total duration in seconds
        - total_frames: total number of mel frames
        - avg_duration_sec: average duration per sample
    """
    if not manifest:
        return {
            "count": 0,
            "total_duration_sec": 0.0,
            "total_frames": 0,
            "avg_duration_sec": 0.0,
        }

    total_duration = sum(e["duration_sec"] for e in manifest)
    total_frames = sum(e["n_frames"] for e in manifest)

    return {
        "count": len(manifest),
        "total_duration_sec": total_duration,
        "total_frames": total_frames,
        "avg_duration_sec": total_duration / len(manifest),
    }
