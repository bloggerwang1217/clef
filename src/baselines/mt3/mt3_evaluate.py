#!/usr/bin/env python3
"""
MT3 Baseline MV2H Evaluation Pipeline (MuseScore Studio 4.6.5 Edition)

This script evaluates MT3 baseline using MuseScore 4.6.5 for MIDI→MusicXML
conversion and MV2H for scoring. Supports parallel processing.

For other models (Clef, Transkun+Beyer), use the shared modules directly:
    from evaluation.mv2h import MV2HEvaluator, MV2HResult
    from evaluation.asap import ASAPDataset

Evaluation Modes:
    full   - Evaluate entire pieces
    chunks - 5-bar chunk evaluation (Zeng et al.)

Usage:
    # Full song evaluation
    python mt3_evaluate.py --mode full \
        --pred_dir data/mt3_midi \
        --gt_dir /path/to/asap \
        --mv2h_bin MV2H/bin \
        --output results.csv \
        --workers 8

    # Retry failed tasks
    python mt3_evaluate.py --retry_file failed.txt \
        --mv2h_bin MV2H/bin \
        --output results_retry.csv \
        --timeout 300
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Import shared modules
from evaluation.mv2h import MV2HEvaluator, MV2HResult, aggregate_mv2h_results, print_mv2h_summary
from evaluation.asap import ASAPDataset, ChunkInfo

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class MuseScoreConfig:
    """MuseScore 4.6.5 configuration."""

    binary_path: str = "tools/mscore"
    timeout: int = 60
    force_overwrite: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    mv2h_bin: str = "MV2H/bin"
    mv2h_timeout: int = 120
    mscore_config: MuseScoreConfig = None
    workers: int = os.cpu_count() or 4

    def __post_init__(self):
        if self.mscore_config is None:
            self.mscore_config = MuseScoreConfig()


# =============================================================================
# MUSESCORE CONVERSION
# =============================================================================


def convert_midi_to_musicxml(
    midi_path: str,
    output_path: str,
    config: MuseScoreConfig,
) -> bool:
    """Convert MIDI to MusicXML using MuseScore 4.6.5."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [config.binary_path, midi_path, "-o", output_path]
    if config.force_overwrite:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=config.timeout
        )
        return result.returncode == 0 and Path(output_path).exists()

    except subprocess.TimeoutExpired:
        logger.debug(f"MuseScore timeout: {midi_path}")
        return False
    except FileNotFoundError:
        logger.error(f"MuseScore not found: {config.binary_path}")
        return False
    except Exception as e:
        logger.debug(f"MuseScore error: {e}")
        return False


def convert_musicxml_to_midi(
    musicxml_path: str,
    midi_path: str,
    config: MuseScoreConfig,
) -> bool:
    """Convert MusicXML to MIDI using MuseScore."""
    Path(midi_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [config.binary_path, musicxml_path, "-o", midi_path]
    if config.force_overwrite:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=config.timeout
        )
        return result.returncode == 0 and Path(midi_path).exists()
    except Exception as e:
        logger.debug(f"Conversion error: {e}")
        return False


# =============================================================================
# EVALUATION TASK
# =============================================================================


@dataclass
class EvalTask:
    """Single evaluation task."""

    task_id: str
    pred_midi_path: str
    gt_midi_path: str
    output_dir: str
    mscore_config: MuseScoreConfig
    mv2h_bin: str
    mv2h_timeout: int


@dataclass
class EvalResult:
    """Evaluation result."""

    task_id: str
    pred_path: str
    gt_path: str
    status: str
    metrics: Optional[MV2HResult] = None
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "pred_path": self.pred_path,
            "gt_path": self.gt_path,
            "status": self.status,
            "error_message": self.error_message,
        }
        if self.metrics:
            result.update(self.metrics.to_dict())
        return result


def evaluate_single_task(task: EvalTask) -> EvalResult:
    """
    Evaluate single prediction.

    Pipeline:
    1. MIDI → MusicXML (MuseScore)
    2. MusicXML → MIDI (consistent format)
    3. MV2H evaluation
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Convert pred MIDI → MusicXML
            pred_xml = os.path.join(temp_dir, f"{task.task_id}.musicxml")
            if not convert_midi_to_musicxml(task.pred_midi_path, pred_xml, task.mscore_config):
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="musescore_failed",
                )

            # Step 2: MusicXML → MIDI (for MV2H)
            pred_midi = os.path.join(temp_dir, f"{task.task_id}_converted.mid")
            if not convert_musicxml_to_midi(pred_xml, pred_midi, task.mscore_config):
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="midi_conversion_failed",
                )

            # Save MusicXML if output_dir specified
            if task.output_dir:
                out_xml = os.path.join(task.output_dir, "musicxml", f"{task.task_id}.musicxml")
                Path(out_xml).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pred_xml, out_xml)

            # Step 3: MV2H evaluation
            evaluator = MV2HEvaluator(task.mv2h_bin, timeout=task.mv2h_timeout)
            metrics = evaluator.evaluate(task.gt_midi_path, pred_midi)

            if metrics is None:
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="mv2h_failed",
                )

            return EvalResult(
                task_id=task.task_id,
                pred_path=task.pred_midi_path,
                gt_path=task.gt_midi_path,
                status="success",
                metrics=metrics,
            )

    except Exception as e:
        return EvalResult(
            task_id=task.task_id,
            pred_path=task.pred_midi_path,
            gt_path=task.gt_midi_path,
            status="error",
            error_message=str(e),
        )


# =============================================================================
# PARALLEL EVALUATION
# =============================================================================


def run_parallel_evaluation(
    tasks: List[EvalTask],
    workers: int,
) -> List[EvalResult]:
    """Run tasks in parallel."""
    results = []
    n_workers = min(workers, len(tasks))

    logger.info(f"Running {len(tasks)} tasks with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_single_task, t): t for t in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating"):
            try:
                results.append(future.result())
            except Exception as e:
                task = futures[future]
                results.append(EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="executor_error",
                    error_message=str(e),
                ))

    return results


# =============================================================================
# FULL SONG EVALUATION
# =============================================================================


def run_full_song_evaluation(
    pred_dir: str,
    gt_dir: str,
    output_dir: str,
    config: EvalConfig,
) -> List[EvalResult]:
    """Run full song evaluation."""
    asap = ASAPDataset(gt_dir)
    pred_path = Path(pred_dir)

    # Find MIDI files
    midi_files = list(pred_path.rglob("*.mid")) + list(pred_path.rglob("*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files")

    # Build tasks
    tasks = []
    skipped = []

    for midi_file in midi_files:
        file_id = midi_file.stem
        gt_path = asap.find_ground_truth_midi(str(midi_file))

        if gt_path is None:
            skipped.append(file_id)
            continue

        tasks.append(EvalTask(
            task_id=file_id,
            pred_midi_path=str(midi_file),
            gt_midi_path=gt_path,
            output_dir=output_dir,
            mscore_config=config.mscore_config,
            mv2h_bin=config.mv2h_bin,
            mv2h_timeout=config.mv2h_timeout,
        ))

    if skipped:
        logger.warning(f"Skipped {len(skipped)} files without ground truth")

    # Run evaluation
    results = run_parallel_evaluation(tasks, config.workers)

    # Add skipped
    for file_id in skipped:
        results.append(EvalResult(
            task_id=file_id, pred_path="", gt_path="", status="no_ground_truth"
        ))

    return results


# =============================================================================
# CHUNK EVALUATION
# =============================================================================


def run_chunk_evaluation(
    pred_dir: str,
    gt_dir: str,
    chunk_csv: str,
    output_dir: str,
    config: EvalConfig,
) -> List[EvalResult]:
    """Run 5-bar chunk evaluation."""
    asap = ASAPDataset(gt_dir)
    chunks = asap.load_chunks(chunk_csv)
    grouped = asap.group_chunks_by_piece(chunks)

    logger.info(f"Processing {len(grouped)} pieces, {len(chunks)} total chunks")

    # For chunk evaluation, we evaluate full files but track chunk info
    # (Actual chunk extraction requires additional implementation)
    tasks = []
    pred_path = Path(pred_dir)

    for piece_id, piece_chunks in grouped.items():
        pred_files = list(pred_path.rglob(f"*{piece_id}*.mid"))
        if not pred_files:
            logger.warning(f"No prediction for: {piece_id}")
            continue

        pred_file = str(pred_files[0])
        gt_path = asap.find_ground_truth_midi(pred_file)

        if gt_path is None:
            continue

        # Create task for each chunk (using full file for now)
        for chunk in piece_chunks:
            tasks.append(EvalTask(
                task_id=chunk.chunk_id,
                pred_midi_path=pred_file,
                gt_midi_path=gt_path,
                output_dir=output_dir,
                mscore_config=config.mscore_config,
                mv2h_bin=config.mv2h_bin,
                mv2h_timeout=config.mv2h_timeout,
            ))

    return run_parallel_evaluation(tasks, config.workers)


# =============================================================================
# RETRY FUNCTIONALITY
# =============================================================================


def save_failed_tasks(results: List[EvalResult], output_path: str) -> int:
    """Save failed tasks for retry."""
    failed_statuses = ["mv2h_failed", "musescore_failed", "midi_conversion_failed",
                       "error", "executor_error", "zero_score"]
    failed = [r for r in results if r.status in failed_statuses]

    if not failed:
        return 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Failed tasks for retry\n")
        f.write("# task_id\\tpred_path\\tgt_path\\tstatus\n")
        for r in failed:
            f.write(f"{r.task_id}\t{r.pred_path}\t{r.gt_path}\t{r.status}\n")

    logger.info(f"Saved {len(failed)} failed tasks to: {output_path}")
    return len(failed)


def load_retry_tasks(
    retry_path: str,
    output_dir: str,
    config: EvalConfig,
) -> List[EvalTask]:
    """Load tasks from retry file."""
    tasks = []

    with open(retry_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            task_id, pred_path, gt_path = parts[0], parts[1], parts[2]

            if not Path(pred_path).exists() or not Path(gt_path).exists():
                continue

            tasks.append(EvalTask(
                task_id=task_id,
                pred_midi_path=pred_path,
                gt_midi_path=gt_path,
                output_dir=output_dir,
                mscore_config=config.mscore_config,
                mv2h_bin=config.mv2h_bin,
                mv2h_timeout=config.mv2h_timeout,
            ))

    logger.info(f"Loaded {len(tasks)} retry tasks")
    return tasks


# =============================================================================
# RESULTS OUTPUT
# =============================================================================


def save_results_csv(results: List[EvalResult], output_path: str) -> None:
    """Save results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["task_id", "pred_path", "gt_path", "status", "error_message",
                  "Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    logger.info(f"Results saved to: {output_path}")


def compute_summary(results: List[EvalResult]) -> Dict[str, Any]:
    """Compute summary statistics."""
    successful = [r.metrics for r in results if r.status == "success" and r.metrics]

    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    if not successful:
        return {"n_total": len(results), "n_success": 0, "status_breakdown": status_counts}

    agg = aggregate_mv2h_results(successful)
    agg["n_total"] = len(results)
    agg["n_success"] = len(successful)
    agg["n_failed"] = len(results) - len(successful)
    agg["status_breakdown"] = status_counts
    return agg


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MT3 + MuseScore 4.6.5 Baseline MV2H Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", choices=["full", "chunks"], default="full")
    parser.add_argument("--pred_dir", help="Prediction MIDI directory")
    parser.add_argument("--gt_dir", help="ASAP dataset directory")
    parser.add_argument("--chunk_csv", help="Chunk CSV (for chunks mode)")

    # Paths
    parser.add_argument("--mv2h_bin", required=True, help="MV2H bin directory")
    parser.add_argument("--mscore_bin", default="tools/mscore", help="MuseScore binary")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--output_dir", help="Output directory for MusicXML")

    # Processing
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--timeout", type=int, default=120, help="MV2H timeout (sec)")
    parser.add_argument("--mscore_timeout", type=int, default=60)

    # Retry
    parser.add_argument("--retry_file", help="Retry file path")
    parser.add_argument("--save_failed", help="Save failed tasks path")

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine mode
    retry_mode = args.retry_file is not None

    # Validate
    if not retry_mode:
        if not args.pred_dir:
            parser.error("--pred_dir required (or use --retry_file)")
        if not args.gt_dir:
            parser.error("--gt_dir required (or use --retry_file)")
        if args.mode == "chunks" and not args.chunk_csv:
            parser.error("--chunk_csv required for chunks mode")
    else:
        if not Path(args.retry_file).exists():
            print(f"Retry file not found: {args.retry_file}")
            sys.exit(1)

    if not Path(args.mv2h_bin).exists():
        print(f"MV2H bin not found: {args.mv2h_bin}")
        sys.exit(1)

    if not Path(args.mscore_bin).exists():
        print(f"MuseScore not found: {args.mscore_bin}")
        sys.exit(1)

    # Config
    output_dir = args.output_dir or str(Path(args.output).parent)
    config = EvalConfig(
        mv2h_bin=args.mv2h_bin,
        mv2h_timeout=args.timeout,
        mscore_config=MuseScoreConfig(binary_path=args.mscore_bin, timeout=args.mscore_timeout),
        workers=args.workers,
    )

    # Print config
    print("=" * 60)
    print("MT3 + MuseScore 4.6.5 | MV2H Evaluation")
    print("=" * 60)
    print(f"Mode:      {'RETRY' if retry_mode else args.mode}")
    print(f"Workers:   {args.workers}")
    print(f"Timeout:   MV2H={args.timeout}s, MuseScore={args.mscore_timeout}s")
    print("=" * 60)

    # Run
    if retry_mode:
        tasks = load_retry_tasks(args.retry_file, output_dir, config)
        results = run_parallel_evaluation(tasks, config.workers)
    elif args.mode == "full":
        results = run_full_song_evaluation(args.pred_dir, args.gt_dir, output_dir, config)
    else:
        results = run_chunk_evaluation(args.pred_dir, args.gt_dir, args.chunk_csv, output_dir, config)

    # Save results
    save_results_csv(results, args.output)

    # Save failed
    failed_path = args.save_failed or os.path.join(output_dir, "failed.txt")
    n_failed = save_failed_tasks(results, failed_path)
    if n_failed > 0:
        print(f"\nFailed tasks: {failed_path}")
        print(f"Retry: python mt3_evaluate.py --retry_file {failed_path} --timeout 300 ...")

    # Summary
    summary = compute_summary(results)
    print_mv2h_summary(summary, f"MT3 Baseline ({args.mode})")

    # Save summary
    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
