"""
Batch transcription client for MT3 Docker service.

Usage example:
    python -m inference.batch_transcribe \
        --mode asap_batch \
        --input-dir /path/to/data/asap_test_set \
        --metadata-csv /path/to/data/asap_test_set/metadata.csv \
        --output-dir /path/to/output/asap_midi_output \
        --api-url http://localhost:5000/batch-transcribe \
        --model piano

Assumes Docker container is running with volumes mounted so the paths above
are visible inside the container (e.g. -v /path/to/data:/data/input,
and you pass /data/input/... paths to this script).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests


ASAP_COMPOSERS = ["Bach", "Beethoven", "Chopin", "Haydn", "Liszt", "Mozart", "Schubert", "Schumann"]


def _find_asap_relative(audio_path: Path) -> Path:
    """Return path from composer onward; fall back to filename."""
    parts = audio_path.parts
    for idx, part in enumerate(parts):
        if part in ASAP_COMPOSERS:
            return Path(*parts[idx:])
    return audio_path.name


def collect_audio_files(mode: str, input_dir: Path, metadata_csv: Path, single_file: Path | None) -> List[Path]:
    if mode == "asap_batch":
        df = pd.read_csv(metadata_csv)
        files: List[Path] = []
        for _, row in df.iterrows():
            rel = row.get("audio_performance")
            if pd.notna(rel) and str(rel).endswith(".wav"):
                candidate = input_dir / rel
                if candidate.exists():
                    files.append(candidate)
                else:
                    print(f"[warn] missing file: {rel}")
        return files
    if mode == "single_file":
        if not single_file:
            raise ValueError("single_file mode requires --input-file")
        if not single_file.exists():
            raise FileNotFoundError(single_file)
        return [single_file]
    raise ValueError(f"unknown mode: {mode}")


def build_jobs(audio_files: List[Path], output_dir: Path, mode: str) -> List[Dict[str, str]]:
    jobs = []
    for audio_path in audio_files:
        if mode == "asap_batch":
            rel = _find_asap_relative(audio_path)
            midi_rel = rel.with_suffix(".mid")
        else:
            midi_rel = Path(audio_path.name).with_suffix(".mid")
        midi_path = output_dir / midi_rel
        jobs.append(
            {
                "audio_path": str(audio_path),
                "midi_path": str(midi_path),
            }
        )
    return jobs


def post_batch(api_url: str, model: str, jobs: List[Dict[str, str]], timeout: int = 600) -> Dict:
    payload = {"model": model, "jobs": jobs}
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MT3 batch transcription via Docker API.")
    p.add_argument("--mode", choices=["asap_batch", "single_file"], required=True)
    p.add_argument("--input-dir", type=Path, required=True, help="Base input directory (mount path inside container).")
    p.add_argument("--metadata-csv", type=Path, help="Path to metadata.csv for ASAP.")
    p.add_argument("--input-file", type=Path, help="Single file path when mode=single_file.")
    p.add_argument("--output-dir", type=Path, required=True, help="Base output directory (mount path inside container).")
    p.add_argument("--api-url", default="http://localhost:5000/batch-transcribe")
    p.add_argument("--model", choices=["piano", "mt3"], default="piano", help="piano=ismir2021, mt3=multitrack")
    return p.parse_args()


def main():
    args = parse_args()
    audio_files = collect_audio_files(args.mode, args.input_dir, args.metadata_csv or Path(""), args.input_file)
    jobs = build_jobs(audio_files, args.output_dir, args.mode)
    print(f"[info] sending {len(jobs)} jobs to {args.api_url} using model={args.model}")
    result = post_batch(args.api_url, args.model, jobs)
    results = result.get("results", [])
    successes = sum(1 for r in results if r.get("status") == "success")
    errors = [r for r in results if r.get("status") != "success"]
    print(f"[done] success={successes} / total={len(results)}")
    if errors:
        print("[errors]")
        for err in errors:
            print(f"  {err.get('audio_path')}: {err.get('status')}")


if __name__ == "__main__":
    main()
