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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# Import shared modules
from src.evaluation.mv2h import MV2HEvaluator, MV2HResult, aggregate_mv2h_results, print_mv2h_summary
from src.evaluation.asap import ASAPDataset, ChunkInfo, extract_measures_to_midi, extract_chunks_batch

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
    mv2h_chunk_timeout: int = 10  # Shorter timeout for chunk evaluation (Zeng's setting)
    mscore_config: MuseScoreConfig = None
    workers: int = os.cpu_count() or 4

    def __post_init__(self):
        if self.mscore_config is None:
            self.mscore_config = MuseScoreConfig()


# =============================================================================
# YAML CONFIG LOADING
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load evaluation configuration from YAML file.

    The YAML config allows centralizing all paths and parameters for easier
    management and reproducibility of experiments.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration values
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def group_chunks_by_piece_performance(
    chunks: List[ChunkInfo],
) -> Dict[Tuple[str, str], List[ChunkInfo]]:
    """
    Group chunks by (piece_id, performance) for efficient batch processing.

    The chunk_id in Zeng format encodes both piece and performance:
        chunk_id = 'Bach#Prelude#bwv_875#Ahfat01M.10'
        -> piece_id = 'Bach#Prelude#bwv_875'
        -> performance = 'Ahfat01M'
        -> chunk_index = 10

    By grouping chunks, we can convert each prediction MIDI to MusicXML
    only once per (piece, performance) pair, then extract all chunks from it.

    Args:
        chunks: List of ChunkInfo objects from Zeng CSV

    Returns:
        Dictionary mapping (piece_id, performance) tuple to list of chunks
    """
    grouped: Dict[Tuple[str, str], List[ChunkInfo]] = {}

    for chunk in chunks:
        # Parse chunk_id: 'Bach#Prelude#bwv_875#Ahfat01M.10'
        # Split from right to separate piece_id from performance.chunk_index
        parts = chunk.chunk_id.rsplit("#", 1)

        if len(parts) != 2:
            logger.warning(f"Invalid chunk_id format: {chunk.chunk_id}")
            continue

        piece_id = parts[0]  # 'Bach#Prelude#bwv_875'
        perf_chunk = parts[1]  # 'Ahfat01M.10'

        # Extract performance name (remove chunk index)
        performance = perf_chunk.rsplit(".", 1)[0]  # 'Ahfat01M'

        key = (piece_id, performance)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(chunk)

    return grouped


def find_pred_midi(
    pred_dir: str,
    piece_id: str,
    performance: str,
) -> Optional[str]:
    """
    Find prediction MIDI file for given piece and performance.

    Supports two directory structures:
    1. Flat: pred_dir/performance.mid
    2. ASAP-style: pred_dir/Composer/Work/Piece/performance.mid

    The ASAP-style structure matches the output of MT3 in 'asap_batch' mode.

    Args:
        pred_dir: Root directory containing prediction MIDI files
        piece_id: Zeng-style piece identifier (e.g., 'Bach#Prelude#bwv_875')
        performance: Performance ID (e.g., 'Ahfat01M')

    Returns:
        Path to prediction MIDI file or None if not found
    """
    pred_path = Path(pred_dir)

    # Convert piece_id from '#' separator to path components
    # 'Bach#Prelude#bwv_875' -> ['Bach', 'Prelude', 'bwv_875']
    path_parts = piece_id.split("#")

    # Try ASAP-style structure first: pred_dir/Bach/Prelude/bwv_875/Ahfat01M.mid
    search_dir = pred_path / "/".join(path_parts)
    if search_dir.exists():
        # Look for files starting with performance ID
        for ext in [".mid", ".midi"]:
            midi_file = search_dir / f"{performance}{ext}"
            if midi_file.exists():
                return str(midi_file)

        # Also try glob pattern for variations (e.g., performance_001.mid)
        for midi_file in search_dir.glob(f"{performance}*"):
            if midi_file.suffix.lower() in [".mid", ".midi"]:
                return str(midi_file)

    # Try flat structure: pred_dir/Bach_Prelude_bwv_875_Ahfat01M.mid
    flat_name = "_".join(path_parts) + f"_{performance}"
    for ext in [".mid", ".midi"]:
        flat_path = pred_path / f"{flat_name}{ext}"
        if flat_path.exists():
            return str(flat_path)

    # Try recursive search as fallback
    for midi_file in pred_path.rglob(f"*{performance}*.mid"):
        return str(midi_file)

    logger.debug(f"No prediction found for {piece_id}#{performance}")
    return None


def convert_to_musicxml_cached(
    midi_path: str,
    cache_dir: str,
    config: MuseScoreConfig,
) -> Optional[str]:
    """
    Convert MIDI to MusicXML using MuseScore, with caching.

    Caching ensures each MIDI file is only converted once, even when
    extracting multiple chunks from the same prediction.

    MusicXML files are stored in data/experiments/mt3/full_musicxml/
    to allow reuse across evaluation runs.

    Args:
        midi_path: Path to input MIDI file
        cache_dir: Directory to store converted MusicXML files (unused, kept for API compatibility)
        config: MuseScore configuration

    Returns:
        Path to MusicXML file or None if conversion failed
    """
    # Fixed MusicXML output directory for reuse across runs
    cache_subdir = Path("data/experiments/mt3/full_musicxml")
    cache_subdir.mkdir(parents=True, exist_ok=True)

    # Create cache path based on MIDI filename
    midi_stem = Path(midi_path).stem
    musicxml_path = cache_subdir / f"{midi_stem}.musicxml"

    # Return cached version if exists
    if musicxml_path.exists():
        logger.debug(f"Using cached MusicXML: {musicxml_path}")
        return str(musicxml_path)

    # Convert MIDI to MusicXML
    if convert_midi_to_musicxml(midi_path, str(musicxml_path), config):
        return str(musicxml_path)

    return None


@dataclass
class ChunkEvalTask:
    """
    Evaluation task for a single 5-bar chunk.

    Unlike full song evaluation, chunk evaluation uses pre-extracted
    MIDI files from MusicXML (which preserves measure boundaries).
    """

    task_id: str  # chunk_id from CSV
    pred_chunk_midi: str  # Extracted pred chunk MIDI
    gt_chunk_midi: str  # Extracted GT chunk MIDI
    mv2h_bin: str
    mv2h_timeout: int


def evaluate_chunk_task(task: ChunkEvalTask) -> EvalResult:
    """
    Evaluate a single 5-bar chunk with MV2H.

    The chunk MIDI files should already be extracted from MusicXML
    with proper measure boundaries. This function directly runs MV2H
    without additional MuseScore conversion.

    Args:
        task: ChunkEvalTask containing paths and configuration

    Returns:
        EvalResult with status and metrics (if successful)

    Error statuses:
        - timeout: MV2H exceeded time limit (likely complex alignment)
        - zero_score: MV2H returned 0 (usually MIDI parsing error)
        - mv2h_failed: MV2H process failed unexpectedly
        - error: Unexpected exception during evaluation
    """
    try:
        evaluator = MV2HEvaluator(task.mv2h_bin, timeout=task.mv2h_timeout)
        metrics = evaluator.evaluate(task.gt_chunk_midi, task.pred_chunk_midi)

        if metrics is None:
            # MV2H process failed - could be timeout or internal error
            return EvalResult(
                task_id=task.task_id,
                pred_path=task.pred_chunk_midi,
                gt_path=task.gt_chunk_midi,
                status="mv2h_failed",
                error_message="MV2H returned None (check Java process logs)",
            )

        if metrics.mv2h == 0 and metrics.multi_pitch == 0:
            # Zero score typically indicates MIDI parsing failure in MV2H
            # This happens when the MIDI file is empty or malformed
            return EvalResult(
                task_id=task.task_id,
                pred_path=task.pred_chunk_midi,
                gt_path=task.gt_chunk_midi,
                status="zero_score",
                error_message="MV2H returned 0 (likely MIDI parsing error)",
                metrics=metrics,  # Still save the metrics for analysis
            )

        return EvalResult(
            task_id=task.task_id,
            pred_path=task.pred_chunk_midi,
            gt_path=task.gt_chunk_midi,
            status="success",
            metrics=metrics,
        )

    except subprocess.TimeoutExpired:
        # Timeout usually means DTW alignment is taking too long
        # This often occurs when prediction differs significantly from GT
        return EvalResult(
            task_id=task.task_id,
            pred_path=task.pred_chunk_midi,
            gt_path=task.gt_chunk_midi,
            status="timeout",
            error_message=f"MV2H exceeded {task.mv2h_timeout}s timeout",
        )

    except Exception as e:
        return EvalResult(
            task_id=task.task_id,
            pred_path=task.pred_chunk_midi,
            gt_path=task.gt_chunk_midi,
            status="error",
            error_message=str(e),
        )


def save_failed_chunks(
    results: List[EvalResult],
    output_dir: str,
) -> Tuple[int, int]:
    """
    Save failed chunks for analysis and potential retry.

    Separates failures into categories matching Zeng's evaluation:
    - timeouts.txt: Chunks that exceeded MV2H time limit
    - errors.txt: All other failures with details

    Args:
        results: List of EvalResult objects
        output_dir: Directory to save failure lists

    Returns:
        Tuple of (timeout_count, error_count)
    """
    timeouts = [r for r in results if r.status == "timeout"]
    errors = [r for r in results if r.status not in ["success", "timeout"]]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if timeouts:
        timeout_path = os.path.join(output_dir, "timeouts.txt")
        with open(timeout_path, "w") as f:
            f.write(f"# Total: {len(timeouts)} chunks timeout\n")
            f.write("# Format: chunk_id\n")
            for r in timeouts:
                f.write(f"{r.task_id}\n")
        logger.info(f"Timeout chunks saved to: {timeout_path}")

    if errors:
        error_path = os.path.join(output_dir, "errors.txt")
        with open(error_path, "w") as f:
            f.write(f"# Total: {len(errors)} chunks with errors\n")
            f.write("# Format: chunk_id\\tstatus\\terror_message\n")
            for r in errors:
                f.write(f"{r.task_id}\t{r.status}\t{r.error_message}\n")
        logger.info(f"Error chunks saved to: {error_path}")

    return len(timeouts), len(errors)


def compute_chunk_summary(
    results: List[EvalResult],
    total_chunks: int,
) -> Dict[str, Any]:
    """
    Compute MV2H summary with BOTH reporting methods for transparency.

    Method 1 (Zeng's, less rigorous): Only average successful chunks
      - MV2H = sum(successful) / n_successful
      - Used by Zeng's original evaluation, ignores failures

    Method 2 (include failures): Treat failures as 0
      - MV2H = sum(successful) / n_total
      - More rigorous, penalizes failures

    Both methods are reported to allow fair comparison while maintaining
    scientific rigor.

    Args:
        results: List of EvalResult objects
        total_chunks: Total number of chunks (including those not evaluated)

    Returns:
        Dictionary with both summary methods and status breakdown
    """
    successful = [r.metrics for r in results if r.status == "success" and r.metrics]
    n_successful = len(successful)
    n_evaluated = len(results)
    n_failed = n_evaluated - n_successful

    # Status breakdown for debugging
    status_counts: Dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    # Method 1: Zeng's method (exclude failures)
    if successful:
        zeng_metrics = aggregate_mv2h_results(successful)
    else:
        zeng_metrics = {
            "Multi-pitch": 0.0,
            "Voice": 0.0,
            "Meter": 0.0,
            "Value": 0.0,
            "Harmony": 0.0,
            "MV2H": 0.0,
            "MV2H_custom": 0.0,
        }

    # Method 2: Include failures as 0
    # Weighted: (zeng_avg * n_successful) / n_total
    include_failures_metrics = {}
    if total_chunks > 0 and successful:
        for key in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom"]:
            include_failures_metrics[key] = zeng_metrics.get(key, 0.0) * n_successful / total_chunks
    else:
        include_failures_metrics = zeng_metrics.copy()

    return {
        "n_total": total_chunks,
        "n_evaluated": n_evaluated,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "success_rate": n_successful / n_evaluated if n_evaluated > 0 else 0.0,
        "status_breakdown": status_counts,
        "zeng_method": zeng_metrics,
        "include_failures": include_failures_metrics,
    }


def print_chunk_summary(summary: Dict[str, Any]) -> None:
    """
    Print formatted chunk evaluation summary.

    Displays both Zeng's method (exclude failures) and the more rigorous
    method (include failures as 0) for transparency.
    """
    print("\n" + "=" * 70)
    print("MV2H Chunk Evaluation Summary")
    print("=" * 70)

    print(f"\nTotal chunks in CSV:  {summary['n_total']}")
    print(f"Evaluated:            {summary['n_evaluated']}")
    print(f"Successful:           {summary['n_successful']} ({summary['success_rate']*100:.1f}%)")
    print(f"Failed:               {summary['n_failed']}")

    print("\nStatus breakdown:")
    for status, count in sorted(summary["status_breakdown"].items()):
        print(f"  {status:20s}: {count}")

    # Zeng's method
    print("\n" + "-" * 70)
    print("Method 1: Zeng's method (exclude failures)")
    print("-" * 70)
    zeng = summary["zeng_method"]
    print(f"Samples: {summary['n_successful']}")
    for metric in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony"]:
        print(f"  {metric:15s}: {zeng.get(metric, 0.0) * 100:6.2f}%")
    print(f"  {'MV2H (official)':15s}: {zeng.get('MV2H', 0.0) * 100:6.2f}%")
    print(f"  {'MV2H (custom)':15s}: {zeng.get('MV2H_custom', 0.0) * 100:6.2f}%  <- (MP+V+Va+H)/4")

    # Include failures method
    print("\n" + "-" * 70)
    print("Method 2: Include failures as 0")
    print("-" * 70)
    incl = summary["include_failures"]
    print(f"Samples: {summary['n_total']} (failures counted as 0)")
    for metric in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony"]:
        print(f"  {metric:15s}: {incl.get(metric, 0.0) * 100:6.2f}%")
    print(f"  {'MV2H (official)':15s}: {incl.get('MV2H', 0.0) * 100:6.2f}%")
    print(f"  {'MV2H (custom)':15s}: {incl.get('MV2H_custom', 0.0) * 100:6.2f}%  <- (MP+V+Va+H)/4")

    print("=" * 70)


def run_chunk_evaluation(
    pred_dir: str,
    gt_dir: str,
    chunk_csv: str,
    output_dir: str,
    config: EvalConfig,
    output_csv: Optional[str] = None,
    musicxml_dir: Optional[str] = None,
) -> Tuple[List[EvalResult], Dict[str, Any]]:
    """
    Run 5-bar chunk evaluation with actual measure extraction.

    This function implements apple-to-apple comparison with Zeng's evaluation:
    1. Load chunk definitions from Zeng CSV (piece#performance.chunk_index format)
    2. Group chunks by (piece, performance) for efficient processing
    3. Convert each prediction MIDI to MusicXML (via MuseScore, once per piece)
    4. Extract 5-bar chunks from both pred MusicXML and GT MusicXML
    5. Run MV2H on extracted chunk MIDI files
    6. Report results using both Zeng's method and include-failures method

    Incremental saving:
    - Results are saved to CSV immediately after each chunk completes
    - Already completed chunks (status='success' in CSV) are skipped
    - This enables resuming evaluation after interruption

    Args:
        pred_dir: Directory containing MT3 prediction MIDI files
        gt_dir: ASAP dataset directory with ground truth
        chunk_csv: Path to Zeng chunk CSV file
        output_dir: Output directory for results and intermediate files
        config: Evaluation configuration
        output_csv: Path to results CSV (enables incremental saving and resume)

    Returns:
        Tuple of (results list, summary dict)
    """
    asap = ASAPDataset(gt_dir)
    chunks = asap.load_chunks(chunk_csv)
    grouped = group_chunks_by_piece_performance(chunks)

    total_chunks = len(chunks)
    logger.info(f"Processing {len(grouped)} (piece, performance) pairs, {total_chunks} total chunks")

    # Load already completed chunks if output_csv exists
    completed_chunks = set()
    if output_csv:
        completed_chunks = load_completed_chunks(output_csv)
        if completed_chunks:
            logger.info(f"Found {len(completed_chunks)} already completed chunks, will skip")
        # Initialize CSV with headers if needed
        init_csv_file(output_csv, CHUNK_CSV_FIELDS)

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chunk_midi_dir = os.path.join(output_dir, "chunk_midi")
    Path(chunk_midi_dir).mkdir(parents=True, exist_ok=True)

    # Prepare all chunk evaluation tasks
    tasks: List[ChunkEvalTask] = []
    skipped_pieces: List[str] = []
    conversion_errors: List[str] = []
    skipped_completed: int = 0

    # Process each (piece, performance) group
    for (piece_id, performance), piece_chunks in tqdm(
        grouped.items(),
        desc="Extracting chunks",
        unit="piece",
    ):
        # 1. Find prediction MIDI
        pred_file = find_pred_midi(pred_dir, piece_id, performance)
        if pred_file is None:
            skipped_pieces.append(f"{piece_id}#{performance}")
            continue

        # 2. Find ground truth MusicXML
        gt_xml = asap.find_ground_truth_xml_by_piece_id(piece_id)
        if gt_xml is None:
            logger.debug(f"No GT MusicXML for: {piece_id}")
            skipped_pieces.append(f"{piece_id}#{performance}")
            continue

        # 3. Convert prediction MIDI to MusicXML (cached)
        pred_xml = convert_to_musicxml_cached(pred_file, output_dir, config.mscore_config)
        if pred_xml is None:
            conversion_errors.append(f"{piece_id}#{performance}")
            continue

        # 4. Filter out already completed chunks
        chunks_to_extract = []
        for chunk in piece_chunks:
            if chunk.chunk_id in completed_chunks:
                skipped_completed += 1
            else:
                chunks_to_extract.append(chunk)

        if not chunks_to_extract:
            continue

        # 5. Prepare batch extraction lists (parse each XML only once!)
        pred_chunks_list = [
            (
                chunk.start_measure,
                chunk.end_measure,
                os.path.join(chunk_midi_dir, f"{chunk.chunk_id.replace('#', '_')}_pred.mid"),
            )
            for chunk in chunks_to_extract
        ]
        gt_chunks_list = [
            (
                chunk.start_measure,
                chunk.end_measure,
                os.path.join(chunk_midi_dir, f"{chunk.chunk_id.replace('#', '_')}_gt.mid"),
            )
            for chunk in chunks_to_extract
        ]

        # 6. Batch extract (parse XML once, extract all chunks)
        pred_results = extract_chunks_batch(pred_xml, pred_chunks_list)
        gt_results = extract_chunks_batch(gt_xml, gt_chunks_list)

        # 7. Create evaluation tasks
        for chunk in chunks_to_extract:
            pred_chunk_path = os.path.join(
                chunk_midi_dir, f"{chunk.chunk_id.replace('#', '_')}_pred.mid"
            )
            gt_chunk_path = os.path.join(
                chunk_midi_dir, f"{chunk.chunk_id.replace('#', '_')}_gt.mid"
            )

            pred_chunk = pred_results.get(pred_chunk_path)
            gt_chunk = gt_results.get(gt_chunk_path)

            if pred_chunk and gt_chunk:
                tasks.append(ChunkEvalTask(
                    task_id=chunk.chunk_id,
                    pred_chunk_midi=pred_chunk,
                    gt_chunk_midi=gt_chunk,
                    mv2h_bin=config.mv2h_bin,
                    mv2h_timeout=config.mv2h_chunk_timeout,  # Use chunk timeout (10s)
                ))
            else:
                logger.debug(f"Failed to extract chunk: {chunk.chunk_id}")

    if skipped_completed:
        logger.info(f"Skipped {skipped_completed} already completed chunks")
    if skipped_pieces:
        logger.warning(f"Skipped {len(skipped_pieces)} pieces (no prediction or GT)")
    if conversion_errors:
        logger.warning(f"Conversion failed for {len(conversion_errors)} pieces")

    logger.info(f"Prepared {len(tasks)} new chunk evaluation tasks")

    # Run parallel evaluation with incremental saving
    new_results: List[EvalResult] = []
    n_workers = min(config.workers, len(tasks)) if tasks else 1

    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(evaluate_chunk_task, t): t for t in tasks}

            for future in tqdm(
                as_completed(futures),
                total=len(tasks),
                desc="Evaluating chunks",
            ):
                try:
                    result = future.result()
                except Exception as e:
                    task = futures[future]
                    result = EvalResult(
                        task_id=task.task_id,
                        pred_path=task.pred_chunk_midi,
                        gt_path=task.gt_chunk_midi,
                        status="executor_error",
                        error_message=str(e),
                    )

                new_results.append(result)

                # Incremental save to CSV
                if output_csv:
                    append_result_to_csv(result, output_csv, CHUNK_CSV_FIELDS)

    # Save failed chunks to separate file for easy retry
    save_failed_chunks(new_results, output_dir)

    # Load all results (including previously completed) for summary
    all_results = new_results
    if output_csv and completed_chunks:
        # Reload all results from CSV for accurate summary
        all_results = load_results_from_csv(output_csv)
        logger.info(f"Loaded {len(all_results)} total results for summary")

    # Compute summary
    summary = compute_chunk_summary(all_results, total_chunks)

    return all_results, summary


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


# CSV field names for chunk evaluation results
# Includes chunk_index for position-based analysis (e.g., success rate by chunk position)
CHUNK_CSV_FIELDS = [
    "task_id", "chunk_index", "piece_id", "performance",
    "status", "error_message",
    "Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom",
    "pred_path", "gt_path",
]

# CSV field names for full song evaluation
FULL_CSV_FIELDS = [
    "task_id", "status", "error_message",
    "Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom",
    "pred_path", "gt_path",
]


def parse_chunk_id(chunk_id: str) -> Dict[str, Any]:
    """
    Parse Zeng-style chunk_id into components for analysis.

    Args:
        chunk_id: Zeng chunk identifier (e.g., 'Bach#Prelude#bwv_875#Ahfat01M.10')

    Returns:
        Dictionary with piece_id, performance, chunk_index

    Example:
        parse_chunk_id('Bach#Prelude#bwv_875#Ahfat01M.10')
        -> {'piece_id': 'Bach#Prelude#bwv_875', 'performance': 'Ahfat01M', 'chunk_index': 10}
    """
    # chunk_id = "Bach#Prelude#bwv_875#Ahfat01M.10"
    parts = chunk_id.rsplit("#", 1)
    if len(parts) != 2:
        return {"piece_id": chunk_id, "performance": "", "chunk_index": 0}

    piece_id = parts[0]  # Bach#Prelude#bwv_875
    perf_chunk = parts[1]  # Ahfat01M.10

    perf_parts = perf_chunk.rsplit(".", 1)
    if len(perf_parts) == 2:
        performance = perf_parts[0]  # Ahfat01M
        try:
            chunk_index = int(perf_parts[1])  # 10
        except ValueError:
            chunk_index = 0
    else:
        performance = perf_chunk
        chunk_index = 0

    return {"piece_id": piece_id, "performance": performance, "chunk_index": chunk_index}


def load_completed_chunks(csv_path: str) -> set:
    """
    Load task_ids of already completed chunks from existing CSV.

    This enables resuming evaluation from where it left off.
    Only chunks with status='success' are considered completed.

    Args:
        csv_path: Path to existing results CSV

    Returns:
        Set of completed task_ids
    """
    completed = set()
    if not Path(csv_path).exists():
        return completed

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only skip if successfully evaluated (not failed)
                if row.get("status") == "success":
                    completed.add(row.get("task_id", ""))
    except Exception as e:
        logger.warning(f"Failed to read existing CSV: {e}")

    return completed


def init_csv_file(csv_path: str, fieldnames: List[str]) -> None:
    """
    Initialize CSV file with headers if it doesn't exist.

    Args:
        csv_path: Path to CSV file
        fieldnames: CSV column names
    """
    if Path(csv_path).exists():
        return

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()


def append_result_to_csv(result: EvalResult, csv_path: str, fieldnames: List[str]) -> None:
    """
    Append a single evaluation result to CSV file.

    This enables incremental saving so results are preserved even if
    the process is interrupted.

    Args:
        result: Evaluation result to append
        csv_path: Path to CSV file
        fieldnames: CSV column names
    """
    row = result.to_dict()

    # Add parsed chunk info for chunk evaluation
    if "chunk_index" in fieldnames:
        parsed = parse_chunk_id(result.task_id)
        row.update(parsed)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def load_results_from_csv(csv_path: str) -> List[EvalResult]:
    """
    Load all evaluation results from CSV file.

    This is used to reload results after resuming for accurate summary computation.

    Args:
        csv_path: Path to results CSV file

    Returns:
        List of EvalResult objects
    """
    results = []
    if not Path(csv_path).exists():
        return results

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Reconstruct MV2HResult if metrics are present
                metrics = None
                if row.get("Multi-pitch") and row.get("status") == "success":
                    try:
                        metrics = MV2HResult(
                            multi_pitch=float(row.get("Multi-pitch", 0)),
                            voice=float(row.get("Voice", 0)),
                            meter=float(row.get("Meter", 0)),
                            value=float(row.get("Value", 0)),
                            harmony=float(row.get("Harmony", 0)),
                            mv2h=float(row.get("MV2H", 0)),
                        )
                    except (ValueError, TypeError):
                        pass

                results.append(EvalResult(
                    task_id=row.get("task_id", ""),
                    pred_path=row.get("pred_path", ""),
                    gt_path=row.get("gt_path", ""),
                    status=row.get("status", "unknown"),
                    metrics=metrics,
                    error_message=row.get("error_message", ""),
                ))
    except Exception as e:
        logger.error(f"Failed to load results from CSV: {e}")

    return results


def save_results_csv(results: List[EvalResult], output_path: str, is_chunk_mode: bool = False) -> None:
    """
    Save all results to CSV (used for final output or non-incremental mode).

    Args:
        results: List of evaluation results
        output_path: Path to output CSV file
        is_chunk_mode: If True, use chunk CSV format with chunk_index
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = CHUNK_CSV_FIELDS if is_chunk_mode else FULL_CSV_FIELDS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            if is_chunk_mode:
                parsed = parse_chunk_id(r.task_id)
                row.update(parsed)
            writer.writerow(row)

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
        epilog="""
Examples:
  # Full song evaluation
  python -m src.baselines.mt3.mt3_evaluate --mode full \\
      --pred_dir data/mt3_midi --gt_dir /path/to/asap \\
      --mv2h_bin MV2H/bin --output results/full.csv

  # 5-bar chunk evaluation (matching Zeng)
  python -m src.baselines.mt3.mt3_evaluate --mode chunks \\
      --pred_dir data/mt3_midi --gt_dir /path/to/asap \\
      --chunk_csv /path/to/zeng_test_chunk_set.csv \\
      --mv2h_bin MV2H/bin --output results/chunks.csv

  # Using YAML config
  python -m src.baselines.mt3.mt3_evaluate --config configs/mt3_evaluate.yaml
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        help="YAML config file (CLI args override config values)",
    )

    # Mode
    parser.add_argument("--mode", choices=["full", "chunks"], default=None)
    parser.add_argument("--pred_dir", help="Prediction MIDI directory")
    parser.add_argument("--gt_dir", help="ASAP dataset directory")
    parser.add_argument("--chunk_csv", help="Chunk CSV (for chunks mode)")

    # Paths
    parser.add_argument("--mv2h_bin", help="MV2H bin directory")
    parser.add_argument("--mscore_bin", help="MuseScore binary")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--output_dir", help="Output directory for MusicXML and chunks")

    # Processing
    parser.add_argument("-j", "--workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="MV2H timeout (sec)")
    parser.add_argument("--chunk_timeout", type=int, default=None, help="Chunk MV2H timeout (sec)")
    parser.add_argument("--mscore_timeout", type=int, default=None)

    # Retry
    parser.add_argument("--retry_file", help="Retry file path")
    parser.add_argument("--save_failed", help="Save failed tasks path")

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load YAML config if provided
    file_config: Dict[str, Any] = {}
    if args.config:
        if not Path(args.config).exists():
            print(f"Config file not found: {args.config}")
            sys.exit(1)
        file_config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")

    # Merge config: CLI args override file config
    def get_value(cli_val, config_key, default=None):
        if cli_val is not None:
            return cli_val
        return file_config.get(config_key, default)

    mode = get_value(args.mode, "mode", "full")
    pred_dir = get_value(args.pred_dir, "pred_dir")
    gt_dir = get_value(args.gt_dir, "gt_dir")
    chunk_csv = get_value(args.chunk_csv, "chunk_csv")
    mv2h_bin = get_value(args.mv2h_bin, "mv2h_bin", "MV2H/bin")
    mscore_bin = get_value(args.mscore_bin, "mscore_bin", "tools/mscore")
    output_path = get_value(args.output, "output_csv", "results/mt3_eval.csv")
    output_dir = get_value(args.output_dir, "output_dir")
    workers = get_value(args.workers, "workers", os.cpu_count() or 4)
    timeout = get_value(args.timeout, "timeout", 120)
    chunk_timeout = get_value(args.chunk_timeout, "chunk_timeout", 10)
    mscore_timeout = get_value(args.mscore_timeout, "mscore_timeout", 60)

    # Determine mode
    retry_mode = args.retry_file is not None

    # Validate
    if not retry_mode:
        if not pred_dir:
            parser.error("--pred_dir required (or use --retry_file or --config)")
        if not gt_dir:
            parser.error("--gt_dir required (or use --retry_file or --config)")
        if mode == "chunks" and not chunk_csv:
            parser.error("--chunk_csv required for chunks mode")

        if not Path(pred_dir).exists():
            print(f"Prediction directory not found: {pred_dir}")
            sys.exit(1)
        if not Path(gt_dir).exists():
            print(f"Ground truth directory not found: {gt_dir}")
            sys.exit(1)
    else:
        if not Path(args.retry_file).exists():
            print(f"Retry file not found: {args.retry_file}")
            sys.exit(1)

    if not Path(mv2h_bin).exists():
        print(f"MV2H bin not found: {mv2h_bin}")
        sys.exit(1)

    if not Path(mscore_bin).exists():
        print(f"MuseScore not found: {mscore_bin}")
        sys.exit(1)

    # Set default output_dir from output path
    if output_dir is None:
        output_dir = str(Path(output_path).parent)

    # Config
    config = EvalConfig(
        mv2h_bin=mv2h_bin,
        mv2h_timeout=timeout,
        mv2h_chunk_timeout=chunk_timeout,
        mscore_config=MuseScoreConfig(binary_path=mscore_bin, timeout=mscore_timeout),
        workers=workers,
    )

    # Print config
    print("=" * 70)
    print("MT3 + MuseScore 4.6.5 | MV2H Evaluation")
    print("=" * 70)
    print(f"Mode:           {'RETRY' if retry_mode else mode}")
    print(f"Workers:        {workers}")
    if mode == "chunks":
        print(f"Chunk timeout:  {chunk_timeout}s (MV2H), {mscore_timeout}s (MuseScore)")
    else:
        print(f"Timeout:        {timeout}s (MV2H), {mscore_timeout}s (MuseScore)")
    print(f"Prediction dir: {pred_dir}")
    print(f"Ground truth:   {gt_dir}")
    print(f"Output:         {output_path}")
    if mode == "chunks":
        print(f"Chunk CSV:      {chunk_csv}")
    print("=" * 70)

    # Run evaluation
    if retry_mode:
        tasks = load_retry_tasks(args.retry_file, output_dir, config)
        results = run_parallel_evaluation(tasks, config.workers)
        summary = compute_summary(results)
        # Print standard summary for retry mode
        print_mv2h_summary(summary, "MT3 Baseline (RETRY)")

    elif mode == "full":
        results = run_full_song_evaluation(pred_dir, gt_dir, output_dir, config)
        summary = compute_summary(results)
        # Save failed for full mode
        failed_path = args.save_failed or os.path.join(output_dir, "failed.txt")
        n_failed = save_failed_tasks(results, failed_path)
        if n_failed > 0:
            print(f"\nFailed tasks saved to: {failed_path}")
            print(f"Retry: python -m src.baselines.mt3.mt3_evaluate --retry_file {failed_path} --timeout 300 ...")
        # Print standard summary
        print_mv2h_summary(summary, "MT3 Baseline (full)")

    else:  # chunks mode
        results, summary = run_chunk_evaluation(
            pred_dir, gt_dir, chunk_csv, output_dir, config,
            output_csv=output_path,  # Enable incremental saving
        )
        # Print chunk-specific summary (both methods)
        print_chunk_summary(summary)
        # Note: Results are saved incrementally during evaluation

    # For full mode, save results (chunk mode saves incrementally)
    if mode == "full":
        save_results_csv(results, output_path, is_chunk_mode=False)

    # Save summary JSON
    summary_path = Path(output_path).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults CSV:  {output_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
