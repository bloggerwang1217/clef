"""
MV2H evaluation on the synthetic piano test set (HumSyn / MuseSyn).

Reads pre-generated MIDI predictions produced by clef_piano_tiny_inference.py
(bar5 mode), compares them against GT kern slices, and writes a CSV with
per-chunk MV2H scores.

Expected prediction MIDI naming: {pred_midi_dir}/{perf_id}.{chunk_index}.mid

Usage:
    poetry run python src/evaluation/syn/eval.py \\
        --pred-midi-dir data/experiments/clef_piano_base/test_kern_pred_5_bar_midi \\
        --output        results/clef_tiny_syn_test.csv

    # Quick smoke-test (first 10 chunks only):
    poetry run python src/evaluation/syn/eval.py \\
        --pred-midi-dir data/experiments/clef_piano_base/test_kern_pred_5_bar_midi \\
        --output        results/clef_tiny_syn_test.csv \\
        --max-chunks 10
"""

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.mv2h import MV2HEvaluator
from src.evaluation.syn import SynDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger(__name__)

FIELDNAMES = [
    "chunk_id",
    "perf_id",
    "start_measure",
    "end_measure",
    "status",
    "Multi-pitch",
    "Voice",
    "Meter",
    "Value",
    "Harmony",
    "MV2H",
    "MV2H_custom",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-midi-dir",
        default="data/experiments/clef_piano_base/test_kern_pred_5_bar_midi",
        help="Directory with pre-generated pred MIDI files ({perf_id}.{chunk_idx}.mid)",
    )
    parser.add_argument(
        "--manifest",
        default="data/experiments/clef_piano_base/test_manifest.json",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/experiments/clef_piano_base",
    )
    parser.add_argument(
        "--metadata",
        default="data/experiments/clef_piano_base/augmentation_metadata.json",
    )
    parser.add_argument(
        "--kern-gt-dir",
        default="data/experiments/clef_piano_base/kern_gt",
    )
    parser.add_argument(
        "--output",
        default="results/clef_tiny_syn_test.csv",
    )
    parser.add_argument("--mv2h-bin", default="MV2H/bin")
    parser.add_argument("--n-bars", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit for quick test")
    args = parser.parse_args()

    pred_midi_dir = Path(args.pred_midi_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    syn = SynDataset(
        manifest_path=args.manifest,
        metadata_path=args.metadata,
        kern_gt_dir=args.kern_gt_dir,
        manifest_dir=args.manifest_dir,
    )
    evaluator = MV2HEvaluator(args.mv2h_bin, timeout=10)

    total = success = 0

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for chunk in syn.iter_5bar_windows(n_bars=args.n_bars):
            if args.max_chunks is not None and total >= args.max_chunks:
                break

            row = {
                "chunk_id": chunk.chunk_id,
                "perf_id": chunk.perf_id,
                "start_measure": chunk.start_measure,
                "end_measure": chunk.end_measure,
                "status": "missing_pred",
            }

            # Locate pre-generated pred MIDI
            pred_midi = pred_midi_dir / f"{chunk.perf_id}.{chunk.chunk_index}.mid"
            if not pred_midi.exists():
                logger.debug(f"No pred MIDI for {chunk.chunk_id}")
                writer.writerow(row)
                total += 1
                continue

            # GT kern slice → temp MIDI
            gt_midi = syn.get_gt_kern_slice_midi(
                chunk.kern_gt_path, chunk.kern_line_start, chunk.kern_line_end
            )
            if gt_midi is None:
                row["status"] = "gt_midi_failed"
                writer.writerow(row)
                total += 1
                continue

            # MV2H evaluation
            result = evaluator.evaluate(gt_midi, str(pred_midi))
            if result is not None:
                row.update(
                    {
                        "status": "success",
                        "Multi-pitch": result.multi_pitch,
                        "Voice": result.voice,
                        "Meter": result.meter,
                        "Value": result.value,
                        "Harmony": result.harmony,
                        "MV2H": result.mv2h,
                        "MV2H_custom": result.mv2h_custom,
                    }
                )
                success += 1
            else:
                row["status"] = "mv2h_failed"

            writer.writerow(row)
            total += 1

            if total % 100 == 0:
                csvfile.flush()
                logger.info(f"Progress: {success}/{total} succeeded")

    logger.info(f"Done: {success}/{total} chunks evaluated → {args.output}")

    # Summary
    df = pd.read_csv(args.output)
    suc = df[df["status"] == "success"]
    if len(suc):
        print(f"\n=== Summary (n={len(suc)}/{len(df)}) ===")
        for col in ["Multi-pitch", "Voice", "Value", "Harmony", "MV2H", "MV2H_custom"]:
            print(f"  {col}: {suc[col].mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
