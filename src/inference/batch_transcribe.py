"""
MT3 Batch Transcription Client

Sends audio files to MT3 Docker service for transcription.
Paths are container paths (not host paths) - no local file checks.

Usage:
    # Single file
    python -m src.inference.batch_transcribe \
        --mode single_file \
        --input-file /data/test/test.mp3 \
        --output-dir /data/test_output

    # ASAP batch
    python -m src.inference.batch_transcribe \
        --mode asap_batch \
        --input-dir /data/asap_test_set \
        --metadata-csv /data/asap_test_set/metadata.csv \
        --output-dir /data/asap_midi_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


# =============================================================================
# Constants
# =============================================================================

ASAP_COMPOSERS = [
    "Bach", "Beethoven", "Chopin", "Haydn",
    "Liszt", "Mozart", "Schubert", "Schumann"
]

DEFAULT_API_URL = "http://localhost:5000/batch-transcribe"
DEFAULT_TIMEOUT = 600  # 10 minutes


# =============================================================================
# File Collection
# =============================================================================

def collect_asap_files(input_dir: Path, metadata_csv: Path) -> list[Path]:
    """Read ASAP metadata and return list of audio paths."""
    df = pd.read_csv(metadata_csv)
    files = []
    for _, row in df.iterrows():
        rel = row.get("audio_performance")
        if pd.notna(rel) and str(rel).endswith(".wav"):
            files.append(input_dir / rel)
    return files


def collect_single_file(input_file: Path) -> list[Path]:
    """Return single file as list."""
    return [input_file]


# =============================================================================
# Job Building
# =============================================================================

def get_output_path(audio_path: Path, output_dir: Path, mode: str) -> Path:
    """Determine MIDI output path for an audio file."""
    if mode == "asap_batch":
        # Preserve ASAP structure: Composer/Type/Movement/file.mid
        rel = extract_asap_relative(audio_path)
    else:
        # Just use filename
        rel = Path(audio_path.name)
    return output_dir / rel.with_suffix(".mid")


def extract_asap_relative(audio_path: Path) -> Path:
    """Extract relative path from composer onwards (e.g., Bach/Prelude/bwv_875/file.wav)."""
    parts = audio_path.parts
    for idx, part in enumerate(parts):
        if part in ASAP_COMPOSERS:
            return Path(*parts[idx:])
    return Path(audio_path.name)


def build_jobs(audio_files: list[Path], output_dir: Path, mode: str) -> list[dict]:
    """Build job list for API request."""
    return [
        {
            "audio_path": str(audio),
            "midi_path": str(get_output_path(audio, output_dir, mode)),
        }
        for audio in audio_files
    ]


# =============================================================================
# API Communication
# =============================================================================

def send_batch_request(api_url: str, model: str, jobs: list[dict], timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Send batch transcription request to MT3 Docker API."""
    payload = {"model": model, "jobs": jobs}
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def print_results(results: list[dict]) -> None:
    """Print transcription results summary."""
    successes = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") != "success"]

    print(f"[done] success={len(successes)} / total={len(results)}")

    if errors:
        print("[errors]")
        for err in errors:
            print(f"  {err.get('audio_path')}: {err.get('status')}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MT3 batch transcription client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    p.add_argument(
        "--mode",
        choices=["asap_batch", "single_file"],
        required=True,
        help="asap_batch: process ASAP test set; single_file: process one file",
    )

    # Input paths (container paths)
    p.add_argument("--input-dir", type=Path, help="Input directory (for asap_batch)")
    p.add_argument("--input-file", type=Path, help="Input file path (for single_file)")
    p.add_argument("--metadata-csv", type=Path, help="ASAP metadata.csv path")

    # Output
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    # API settings
    p.add_argument("--api-url", default=DEFAULT_API_URL, help="MT3 Docker API URL")
    p.add_argument(
        "--model",
        choices=["piano", "mt3"],
        default="piano",
        help="piano=ismir2021 (with velocity), mt3=multitrack (no velocity)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Collect files based on mode
    if args.mode == "asap_batch":
        if not args.input_dir or not args.metadata_csv:
            raise ValueError("asap_batch requires --input-dir and --metadata-csv")
        audio_files = collect_asap_files(args.input_dir, args.metadata_csv)
    else:
        if not args.input_file:
            raise ValueError("single_file requires --input-file")
        audio_files = collect_single_file(args.input_file)

    # Build and send jobs
    jobs = build_jobs(audio_files, args.output_dir, args.mode)
    print(f"[info] sending {len(jobs)} jobs to {args.api_url} (model={args.model})")

    result = send_batch_request(args.api_url, args.model, jobs)
    print_results(result.get("results", []))


if __name__ == "__main__":
    main()
