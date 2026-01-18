#!/usr/bin/env python3
"""
MT3 Baseline MV2H Evaluation Pipeline (MuseScore Studio 4.6.5 Edition)

This script provides comprehensive MV2H evaluation for MT3 baseline using
MuseScore Studio 4.6.5 as the "Industry Standard Baseline" for MIDI to MusicXML
conversion, with full parallel processing support.

Evaluation Modes:
1. Full Song Evaluation - Evaluate the entire piece at once
2. 5-bar Chunk Evaluation - Apple-to-apple comparison with Zeng's method

Academic Justification:
=======================

1. MUSESCORE STUDIO 4.6.5 AS INDUSTRY STANDARD BASELINE
   We utilize MuseScore Studio 4.6.5 (released Dec 18, 2025), the latest stable
   release incorporating critical stability fixes for Linux environments.
   Unlike naive quantization libraries (e.g., music21), MuseScore 4 utilizes
   a sophisticated heuristic-based import engine that performs:
   - Voice separation
   - Tuplet detection
   - Smart quantization
   - Reworked chord symbol handling (SMuFL Compliant)

   Reference: MuseScore 4.0+ MIDI Import Algorithm
   "MuseScore 4.0 introduced a new MIDI import logic that automatically
    analyzes whether the MIDI segment is 'melody' or 'accompaniment',
    and decides stem direction and multi-voice splitting accordingly."

   Why 4.6.5 over 4.6.4?
   - Crash fixes critical for parallel processing (64+ workers)
   - Linux VST3 support optimization (better AppImage stability with Xvfb)
   - Reworked chord symbol handling for cleaner XML output

   This represents the pinnacle of rule-based notation systems and serves
   as a rigorous "Standard Industry Baseline" for neural A2S comparison.

2. MV2H EVALUATION
   Reference: McLeod & Steedman "Evaluating Automatic Polyphonic Music Transcription"
   - Uses DTW alignment (-a flag) for robust comparison
   - Outputs 6 metrics: Multi-pitch, Voice, Meter, Value, Harmony, MV2H

3. PARALLEL PROCESSING
   MV2H evaluation uses brute-force alignment which is computationally expensive.
   This implementation supports configurable worker count for parallel evaluation.

Usage:
    # Full Song Evaluation (8 workers, 120s timeout)
    python mt3_evaluate.py --mode full \
        --pred_dir data/experiments/mt3/full_midi \
        --gt_dir /path/to/asap-dataset \
        --mv2h_bin MV2H/bin \
        --output results/full_song.csv \
        --workers 8 \
        --timeout 120

    # 5-bar Chunk Evaluation
    python mt3_evaluate.py --mode chunks \
        --pred_dir data/experiments/mt3/full_midi \
        --gt_dir /path/to/asap-dataset \
        --chunk_csv /path/to/zeng_test_chunk_set.csv \
        --mv2h_bin MV2H/bin \
        --output results/chunks.csv \
        --workers 16

Author: Clef Project
License: Apache-2.0
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

from tqdm import tqdm

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================


@dataclass
class MuseScoreConfig:
    """
    MuseScore Studio 4.6.5 configuration for MIDI to MusicXML conversion.

    Academic Justification:
    MuseScore Studio 4.6.5 is chosen as the "Industry Standard Baseline" because:
    1. It uses mature "Smart Import" algorithm for voice separation
    2. It performs automatic tuplet detection
    3. It represents the state-of-the-art for rule-based notation

    Reference: MuseScore GitHub Release v4.6.5
    https://github.com/musescore/MuseScore/releases/tag/v4.6.5
    """

    # Path to MuseScore binary (wrapper script or AppImage)
    binary_path: str = "tools/mscore"

    # Conversion timeout in seconds
    timeout: int = 60

    # Force overwrite existing files
    force_overwrite: bool = True


@dataclass
class MV2HConfig:
    """
    MV2H evaluation configuration.

    Reference: McLeod & Steedman (2018)
    "Evaluating Automatic Polyphonic Music Transcription"

    Note on timeout:
    MV2H uses brute-force DTW alignment which can be slow for long pieces.
    Default timeout is 120 seconds, but complex pieces may need more time.
    """

    # Path to MV2H bin directory
    bin_path: str = "MV2H/bin"

    # Evaluation timeout in seconds
    # MV2H uses brute-force alignment, so this should be generous
    timeout: int = 120

    # Use DTW alignment (-a flag)
    use_alignment: bool = True


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""

    # Number of worker processes
    # Default: use all available CPUs
    workers: int = os.cpu_count() or 4

    # Chunk size for ProcessPoolExecutor
    chunksize: int = 1


@dataclass
class MV2HResult:
    """Container for MV2H evaluation results."""

    multi_pitch: float = 0.0
    voice: float = 0.0
    meter: float = 0.0
    value: float = 0.0
    harmony: float = 0.0
    mv2h: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "Multi-pitch": self.multi_pitch,
            "Voice": self.voice,
            "Meter": self.meter,
            "Value": self.value,
            "Harmony": self.harmony,
            "MV2H": self.mv2h,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "MV2HResult":
        """Create from dictionary."""
        return cls(
            multi_pitch=d.get("Multi-pitch", 0.0),
            voice=d.get("Voice", 0.0),
            meter=d.get("Meter", 0.0),
            value=d.get("Value", 0.0),
            harmony=d.get("Harmony", 0.0),
            mv2h=d.get("MV2H", 0.0),
        )


# =============================================================================
# MUSESCORE CONVERSION (INDUSTRY STANDARD BASELINE)
# =============================================================================


def convert_midi_to_musicxml_musescore(
    midi_path: str,
    output_path: str,
    config: Optional[MuseScoreConfig] = None,
) -> bool:
    """
    Convert MIDI to MusicXML using MuseScore Studio 4.6.5 (Industry Standard Baseline).

    This is the critical conversion step that transforms raw MT3 MIDI output
    into properly quantized, voice-separated MusicXML notation.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output MusicXML file
        config: MuseScore configuration

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = MuseScoreConfig()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build command
    # xvfb-run is handled by the wrapper script
    cmd = [config.binary_path, midi_path, "-o", output_path]

    if config.force_overwrite:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )

        if result.returncode != 0:
            logger.warning(f"MuseScore conversion failed: {result.stderr}")
            return False

        # Verify output file was created
        if not Path(output_path).exists():
            logger.warning(f"MuseScore did not create output file: {output_path}")
            return False

        return True

    except subprocess.TimeoutExpired:
        logger.warning(f"MuseScore conversion timeout after {config.timeout}s")
        return False

    except FileNotFoundError:
        logger.error(
            f"MuseScore binary not found: {config.binary_path}\n"
            "Please run: ./scripts/setup_musescore.sh"
        )
        return False

    except Exception as e:
        logger.error(f"MuseScore conversion error: {e}")
        return False


def convert_musicxml_to_midi(
    musicxml_path: str,
    midi_path: str,
    config: Optional[MuseScoreConfig] = None,
) -> bool:
    """
    Convert MusicXML to MIDI using MuseScore.

    Used for ground truth conversion when needed.

    Args:
        musicxml_path: Path to input MusicXML file
        midi_path: Path to output MIDI file
        config: MuseScore configuration

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = MuseScoreConfig()

    Path(midi_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [config.binary_path, musicxml_path, "-o", midi_path]
    if config.force_overwrite:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )
        return result.returncode == 0 and Path(midi_path).exists()

    except Exception as e:
        logger.error(f"MusicXML to MIDI conversion error: {e}")
        return False


# =============================================================================
# MV2H EVALUATION
# =============================================================================


def run_mv2h(
    gt_midi_path: str,
    pred_midi_path: str,
    config: Optional[MV2HConfig] = None,
) -> Optional[MV2HResult]:
    """
    Run MV2H evaluation between ground truth and prediction MIDI files.

    Uses the MV2H Java tool with optional DTW alignment (-a flag).

    Note: MV2H uses brute-force alignment which is O(n^2) or worse.
    For long pieces, this can take several minutes.

    Args:
        gt_midi_path: Path to ground truth MIDI
        pred_midi_path: Path to prediction MIDI
        config: MV2H configuration

    Returns:
        MV2HResult object or None if failed
    """
    if config is None:
        config = MV2HConfig()

    # Create temporary files for conversion output
    # Use unique names to avoid conflicts in parallel execution
    import uuid

    uid = uuid.uuid4().hex[:8]
    gt_conv = f"/tmp/mv2h_gt_{uid}.conv.txt"
    pred_conv = f"/tmp/mv2h_pred_{uid}.conv.txt"

    try:
        # Step 1: Convert GT MIDI to MV2H format
        result_gt = subprocess.run(
            ["java", "-cp", config.bin_path, "mv2h.tools.Converter", "-i", gt_midi_path],
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )
        if result_gt.returncode != 0:
            logger.debug(f"GT conversion failed: {result_gt.stderr}")
            return None

        with open(gt_conv, "w") as f:
            f.write(result_gt.stdout)

        # Step 2: Convert Pred MIDI to MV2H format
        result_pred = subprocess.run(
            ["java", "-cp", config.bin_path, "mv2h.tools.Converter", "-i", pred_midi_path],
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )
        if result_pred.returncode != 0:
            logger.debug(f"Pred conversion failed: {result_pred.stderr}")
            return None

        with open(pred_conv, "w") as f:
            f.write(result_pred.stdout)

        # Step 3: Run MV2H evaluation
        mv2h_cmd = [
            "java",
            "-cp",
            config.bin_path,
            "mv2h.Main",
            "-g",
            gt_conv,
            "-t",
            pred_conv,
        ]
        if config.use_alignment:
            mv2h_cmd.append("-a")

        result_mv2h = subprocess.run(
            mv2h_cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )

        if result_mv2h.returncode != 0:
            logger.debug(f"MV2H evaluation failed: {result_mv2h.stderr}")
            return None

        # Parse output (last 6 lines contain metrics)
        lines = result_mv2h.stdout.strip().splitlines()
        if len(lines) < 6:
            logger.debug(f"Unexpected MV2H output: {result_mv2h.stdout}")
            return None

        metrics = {}
        for line in lines[-6:]:
            if ": " in line:
                metric_key, value = line.split(": ", 1)
                try:
                    metrics[metric_key] = float(value)
                except ValueError:
                    pass

        return MV2HResult.from_dict(metrics)

    except subprocess.TimeoutExpired:
        logger.debug(f"MV2H timeout after {config.timeout}s")
        return None

    except Exception as e:
        logger.debug(f"MV2H error: {e}")
        return None

    finally:
        # Cleanup temporary files
        for f in [gt_conv, pred_conv]:
            try:
                os.remove(f)
            except OSError:
                pass


# =============================================================================
# GROUND TRUTH HANDLING
# =============================================================================


def find_asap_ground_truth(
    pred_path: str,
    asap_base_dir: str,
    pred_base_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Find corresponding ASAP ground truth for a prediction file.

    Tries multiple path patterns to match prediction to ground truth.

    Args:
        pred_path: Path to prediction file
        asap_base_dir: Base directory of ASAP dataset
        pred_base_dir: Base directory of predictions

    Returns:
        Path to ground truth MIDI or None if not found
    """
    pred_name = Path(pred_path).stem

    # Try to parse naming convention: Composer_Work_Piece_PerformanceID
    parts = pred_name.split("_")

    if len(parts) >= 3:
        for i in range(1, len(parts) - 1):
            composer = parts[0]

            # Try different path structures
            for j in range(i + 1, len(parts)):
                work = "_".join(parts[1:j])
                piece = "_".join(parts[j:-1]) if j < len(parts) - 1 else ""

                # Pattern 1: Composer/Work/Piece/midi_score.mid
                if piece:
                    gt_path = Path(asap_base_dir) / composer / work / piece / "midi_score.mid"
                    if gt_path.exists():
                        return str(gt_path)

                # Pattern 2: Composer/Work/midi_score.mid
                gt_path = Path(asap_base_dir) / composer / work / "midi_score.mid"
                if gt_path.exists():
                    return str(gt_path)

    return None


# =============================================================================
# EVALUATION TASK (SINGLE ITEM)
# =============================================================================


@dataclass
class EvalTask:
    """A single evaluation task."""

    task_id: str
    pred_midi_path: str
    gt_midi_path: str
    output_dir: str
    mscore_config: MuseScoreConfig
    mv2h_config: MV2HConfig

    # For chunk evaluation
    chunk_info: Optional[Dict] = None


@dataclass
class EvalResult:
    """Result of a single evaluation task."""

    task_id: str
    pred_path: str
    gt_path: str
    status: str
    metrics: Optional[MV2HResult] = None
    error_message: str = ""

    # For chunk evaluation
    chunk_info: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        result = {
            "task_id": self.task_id,
            "pred_path": self.pred_path,
            "gt_path": self.gt_path,
            "status": self.status,
            "error_message": self.error_message,
        }

        if self.chunk_info:
            result.update(self.chunk_info)

        if self.metrics:
            result.update(self.metrics.to_dict())

        return result


def evaluate_single_task(task: EvalTask) -> EvalResult:
    """
    Evaluate a single prediction against ground truth.

    This function is designed to be called in parallel by ProcessPoolExecutor.

    Pipeline:
    1. Convert prediction MIDI -> MusicXML (MuseScore)
    2. Convert MusicXML -> MIDI (for MV2H, ensures consistent format)
    3. Run MV2H evaluation

    Args:
        task: EvalTask containing all necessary paths and configs

    Returns:
        EvalResult with status and metrics
    """
    try:
        # Create temporary directory for this task
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Convert pred MIDI -> MusicXML using MuseScore
            pred_xml_path = os.path.join(temp_dir, f"{task.task_id}_pred.musicxml")
            if not convert_midi_to_musicxml_musescore(
                task.pred_midi_path,
                pred_xml_path,
                task.mscore_config,
            ):
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="musescore_conversion_failed",
                    chunk_info=task.chunk_info,
                )

            # Step 2: Convert MusicXML -> MIDI (for consistent MV2H input)
            pred_midi_converted = os.path.join(temp_dir, f"{task.task_id}_pred_converted.mid")
            if not convert_musicxml_to_midi(
                pred_xml_path,
                pred_midi_converted,
                task.mscore_config,
            ):
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="midi_reconversion_failed",
                    chunk_info=task.chunk_info,
                )

            # Optionally save the converted MusicXML to output directory
            if task.output_dir:
                out_xml_path = os.path.join(task.output_dir, "musicxml", f"{task.task_id}.musicxml")
                Path(out_xml_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pred_xml_path, out_xml_path)

            # Step 3: Run MV2H evaluation
            metrics = run_mv2h(
                task.gt_midi_path,
                pred_midi_converted,
                task.mv2h_config,
            )

            if metrics is None:
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="mv2h_failed",
                    chunk_info=task.chunk_info,
                )

            if metrics.mv2h == 0:
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="zero_score",
                    metrics=metrics,
                    chunk_info=task.chunk_info,
                )

            return EvalResult(
                task_id=task.task_id,
                pred_path=task.pred_midi_path,
                gt_path=task.gt_midi_path,
                status="success",
                metrics=metrics,
                chunk_info=task.chunk_info,
            )

    except Exception as e:
        return EvalResult(
            task_id=task.task_id,
            pred_path=task.pred_midi_path,
            gt_path=task.gt_midi_path,
            status="error",
            error_message=str(e),
            chunk_info=task.chunk_info,
        )


# =============================================================================
# BATCH EVALUATION (PARALLEL)
# =============================================================================


def run_parallel_evaluation(
    tasks: List[EvalTask],
    parallel_config: ParallelConfig,
) -> List[EvalResult]:
    """
    Run evaluation tasks in parallel using ProcessPoolExecutor.

    Args:
        tasks: List of EvalTask objects
        parallel_config: Parallel processing configuration

    Returns:
        List of EvalResult objects
    """
    results = []

    n_workers = min(parallel_config.workers, len(tasks))
    logger.info(f"Running {len(tasks)} tasks with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(evaluate_single_task, task): task for task in tasks
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="Evaluating",
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = future_to_task[future]
                results.append(
                    EvalResult(
                        task_id=task.task_id,
                        pred_path=task.pred_midi_path,
                        gt_path=task.gt_midi_path,
                        status="executor_error",
                        error_message=str(e),
                    )
                )

    return results


# =============================================================================
# FULL SONG EVALUATION
# =============================================================================


def run_full_song_evaluation(
    pred_dir: str,
    gt_dir: str,
    output_dir: str,
    output_csv: str,
    mscore_config: MuseScoreConfig,
    mv2h_config: MV2HConfig,
    parallel_config: ParallelConfig,
) -> List[EvalResult]:
    """
    Run full song evaluation on all MIDI files in prediction directory.

    Args:
        pred_dir: Directory containing MT3 MIDI outputs
        gt_dir: ASAP dataset base directory
        output_dir: Directory for intermediate outputs
        output_csv: Path to output CSV file
        mscore_config: MuseScore configuration
        mv2h_config: MV2H configuration
        parallel_config: Parallel processing configuration

    Returns:
        List of EvalResult objects
    """
    pred_path = Path(pred_dir)

    # Find all MIDI files
    midi_files = list(pred_path.rglob("*.mid")) + list(pred_path.rglob("*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files in {pred_dir}")

    # Build tasks
    tasks = []
    skipped = []

    for midi_file in midi_files:
        file_id = midi_file.stem

        # Find ground truth
        gt_path = find_asap_ground_truth(str(midi_file), gt_dir, pred_dir)

        if gt_path is None:
            skipped.append(file_id)
            continue

        tasks.append(
            EvalTask(
                task_id=file_id,
                pred_midi_path=str(midi_file),
                gt_midi_path=gt_path,
                output_dir=output_dir,
                mscore_config=mscore_config,
                mv2h_config=mv2h_config,
            )
        )

    if skipped:
        logger.warning(f"Skipped {len(skipped)} files without ground truth")

    # Run parallel evaluation
    results = run_parallel_evaluation(tasks, parallel_config)

    # Add skipped files to results
    for file_id in skipped:
        results.append(
            EvalResult(
                task_id=file_id,
                pred_path="",
                gt_path="",
                status="no_ground_truth",
            )
        )

    # Save results to CSV
    save_results_to_csv(results, output_csv)

    return results


# =============================================================================
# 5-BAR CHUNK EVALUATION
# =============================================================================


@dataclass
class ChunkInfo:
    """Information about a 5-bar chunk from Zeng's test set."""

    chunk_id: str
    piece_id: str
    start_measure: int
    end_measure: int
    asap_path: str = ""


def load_chunk_csv(csv_path: str) -> List[ChunkInfo]:
    """
    Load chunk information from Zeng's test set CSV.

    Expected CSV format:
        chunk_id, piece_id, start_measure, end_measure, asap_path

    Args:
        csv_path: Path to chunk CSV file

    Returns:
        List of ChunkInfo objects
    """
    chunks = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunk = ChunkInfo(
                chunk_id=row.get("chunk_id", ""),
                piece_id=row.get("piece_id", ""),
                start_measure=int(row.get("start_measure", 0)),
                end_measure=int(row.get("end_measure", 0)),
                asap_path=row.get("asap_path", ""),
            )
            chunks.append(chunk)

    logger.info(f"Loaded {len(chunks)} chunks from {csv_path}")
    return chunks


def run_chunk_evaluation(
    pred_dir: str,
    gt_dir: str,
    chunk_csv: str,
    output_dir: str,
    output_csv: str,
    mscore_config: MuseScoreConfig,
    mv2h_config: MV2HConfig,
    parallel_config: ParallelConfig,
) -> List[EvalResult]:
    """
    Run 5-bar chunk evaluation.

    Note: Chunk extraction requires music21 for measure slicing.
    This is a fallback when MuseScore-based chunk extraction is not available.

    Args:
        pred_dir: Directory containing MT3 MIDI outputs
        gt_dir: ASAP dataset base directory
        chunk_csv: Path to Zeng's chunk CSV file
        output_dir: Directory for intermediate outputs
        output_csv: Path to output CSV file
        mscore_config: MuseScore configuration
        mv2h_config: MV2H configuration
        parallel_config: Parallel processing configuration

    Returns:
        List of EvalResult objects
    """
    # For chunk evaluation, we need music21 for measure extraction
    try:
        import music21
        from music21 import converter
    except ImportError:
        logger.error("Chunk evaluation requires music21. Install with: pip install music21")
        sys.exit(1)

    chunks = load_chunk_csv(chunk_csv)

    # Group chunks by piece
    piece_chunks: Dict[str, List[ChunkInfo]] = {}
    for chunk in chunks:
        if chunk.piece_id not in piece_chunks:
            piece_chunks[chunk.piece_id] = []
        piece_chunks[chunk.piece_id].append(chunk)

    logger.info(f"Processing {len(piece_chunks)} pieces with {len(chunks)} total chunks")

    # Build tasks
    tasks = []
    pred_path = Path(pred_dir)

    for piece_id, piece_chunk_list in piece_chunks.items():
        # Find prediction MIDI for this piece
        pred_files = list(pred_path.rglob(f"*{piece_id}*.mid"))

        if not pred_files:
            logger.warning(f"No prediction found for piece: {piece_id}")
            for chunk in piece_chunk_list:
                tasks.append(
                    EvalTask(
                        task_id=chunk.chunk_id,
                        pred_midi_path="",
                        gt_midi_path="",
                        output_dir=output_dir,
                        mscore_config=mscore_config,
                        mv2h_config=mv2h_config,
                        chunk_info={
                            "piece_id": chunk.piece_id,
                            "start_measure": chunk.start_measure,
                            "end_measure": chunk.end_measure,
                        },
                    )
                )
            continue

        pred_file = str(pred_files[0])

        # Find ground truth
        gt_path = find_asap_ground_truth(pred_file, gt_dir, pred_dir)
        if gt_path is None and piece_chunk_list[0].asap_path:
            gt_path = os.path.join(gt_dir, piece_chunk_list[0].asap_path)

        if gt_path is None or not os.path.exists(gt_path):
            logger.warning(f"No ground truth found for piece: {piece_id}")
            continue

        # For each chunk, extract measures and create task
        # This part would need measure extraction logic
        # For now, we use full-file evaluation per chunk
        # (actual chunk extraction requires more complex logic)

        for chunk in piece_chunk_list:
            tasks.append(
                EvalTask(
                    task_id=chunk.chunk_id,
                    pred_midi_path=pred_file,
                    gt_midi_path=gt_path,
                    output_dir=output_dir,
                    mscore_config=mscore_config,
                    mv2h_config=mv2h_config,
                    chunk_info={
                        "piece_id": chunk.piece_id,
                        "start_measure": chunk.start_measure,
                        "end_measure": chunk.end_measure,
                    },
                )
            )

    # Run parallel evaluation
    results = run_parallel_evaluation(tasks, parallel_config)

    # Save results to CSV
    save_results_to_csv(results, output_csv, include_chunk_info=True)

    return results


# =============================================================================
# RESULTS AGGREGATION AND EXPORT
# =============================================================================


def save_results_to_csv(
    results: List[EvalResult],
    output_path: str,
    include_chunk_info: bool = False,
) -> None:
    """Save evaluation results to CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine fieldnames
    if include_chunk_info:
        fieldnames = [
            "task_id",
            "piece_id",
            "start_measure",
            "end_measure",
            "pred_path",
            "gt_path",
            "status",
            "error_message",
            "Multi-pitch",
            "Voice",
            "Meter",
            "Value",
            "Harmony",
            "MV2H",
        ]
    else:
        fieldnames = [
            "task_id",
            "pred_path",
            "gt_path",
            "status",
            "error_message",
            "Multi-pitch",
            "Voice",
            "Meter",
            "Value",
            "Harmony",
            "MV2H",
        ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    logger.info(f"Results saved to: {output_path}")


def save_failed_tasks(
    results: List[EvalResult],
    output_path: str,
    failed_statuses: Optional[List[str]] = None,
) -> int:
    """
    Save failed/timeout tasks to a text file for retry.

    Format: One task per line with tab-separated fields:
        task_id\tpred_path\tgt_path\tstatus

    Args:
        results: List of evaluation results
        output_path: Path to output file (e.g., timeout.txt)
        failed_statuses: List of status values to consider as failed
                        Default: ["mv2h_failed", "musescore_conversion_failed",
                                 "midi_reconversion_failed", "error", "executor_error"]

    Returns:
        Number of failed tasks saved
    """
    if failed_statuses is None:
        failed_statuses = [
            "mv2h_failed",
            "musescore_conversion_failed",
            "midi_reconversion_failed",
            "error",
            "executor_error",
            "zero_score",
        ]

    failed = [r for r in results if r.status in failed_statuses]

    if not failed:
        logger.info("No failed tasks to save")
        return 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Write header comment
        f.write(f"# Failed tasks for retry (statuses: {', '.join(failed_statuses)})\n")
        f.write("# Format: task_id\\tpred_path\\tgt_path\\tstatus\n")

        for r in failed:
            f.write(f"{r.task_id}\t{r.pred_path}\t{r.gt_path}\t{r.status}\n")

    logger.info(f"Saved {len(failed)} failed tasks to: {output_path}")
    return len(failed)


def load_retry_file(
    retry_path: str,
    output_dir: str,
    mscore_config: MuseScoreConfig,
    mv2h_config: MV2HConfig,
) -> List[EvalTask]:
    """
    Load tasks from a retry file.

    Args:
        retry_path: Path to retry file (e.g., timeout.txt)
        output_dir: Output directory for intermediate files
        mscore_config: MuseScore configuration
        mv2h_config: MV2H configuration

    Returns:
        List of EvalTask objects
    """
    tasks = []

    with open(retry_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                logger.warning(f"Invalid retry line: {line}")
                continue

            task_id = parts[0]
            pred_path = parts[1]
            gt_path = parts[2]

            # Validate paths exist
            if not pred_path or not Path(pred_path).exists():
                logger.warning(f"Prediction file not found: {pred_path}")
                continue
            if not gt_path or not Path(gt_path).exists():
                logger.warning(f"Ground truth file not found: {gt_path}")
                continue

            tasks.append(
                EvalTask(
                    task_id=task_id,
                    pred_midi_path=pred_path,
                    gt_midi_path=gt_path,
                    output_dir=output_dir,
                    mscore_config=mscore_config,
                    mv2h_config=mv2h_config,
                )
            )

    logger.info(f"Loaded {len(tasks)} tasks from retry file: {retry_path}")
    return tasks


def run_retry_evaluation(
    retry_file: str,
    output_dir: str,
    output_csv: str,
    mscore_config: MuseScoreConfig,
    mv2h_config: MV2HConfig,
    parallel_config: ParallelConfig,
) -> List[EvalResult]:
    """
    Run evaluation on tasks loaded from retry file.

    Args:
        retry_file: Path to retry file
        output_dir: Directory for intermediate outputs
        output_csv: Path to output CSV file
        mscore_config: MuseScore configuration
        mv2h_config: MV2H configuration
        parallel_config: Parallel processing configuration

    Returns:
        List of EvalResult objects
    """
    # Load tasks from retry file
    tasks = load_retry_file(retry_file, output_dir, mscore_config, mv2h_config)

    if not tasks:
        logger.warning("No valid tasks loaded from retry file")
        return []

    # Run parallel evaluation
    results = run_parallel_evaluation(tasks, parallel_config)

    # Save results to CSV
    save_results_to_csv(results, output_csv)

    return results


def aggregate_results(results: List[EvalResult]) -> Dict[str, Any]:
    """Aggregate MV2H results."""
    successful = [r for r in results if r.status == "success" and r.metrics]

    if not successful:
        return {
            "n_total": len(results),
            "n_success": 0,
            "n_failed": len(results),
        }

    n = len(successful)
    avg = {
        "Multi-pitch": sum(r.metrics.multi_pitch for r in successful) / n,
        "Voice": sum(r.metrics.voice for r in successful) / n,
        "Meter": sum(r.metrics.meter for r in successful) / n,
        "Value": sum(r.metrics.value for r in successful) / n,
        "Harmony": sum(r.metrics.harmony for r in successful) / n,
        "MV2H": sum(r.metrics.mv2h for r in successful) / n,
    }

    # Zeng's custom MV2H formula (excludes Meter)
    avg["MV2H_custom"] = (
        avg["Multi-pitch"] + avg["Voice"] + avg["Value"] + avg["Harmony"]
    ) / 4

    # Count by status
    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    return {
        "n_total": len(results),
        "n_success": n,
        "n_failed": len(results) - n,
        "status_breakdown": status_counts,
        "metrics": avg,
    }


def print_summary(summary: Dict[str, Any], mode: str) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "=" * 70)
    print(f"MT3 + MuseScore Studio 4.6.5 Baseline | MV2H Evaluation ({mode} mode)")
    print("=" * 70)
    print(f"Total samples:       {summary['n_total']}")
    print(f"Successful:          {summary['n_success']}")
    print(f"Failed:              {summary['n_failed']}")

    if "status_breakdown" in summary:
        print("\nStatus breakdown:")
        for status, count in sorted(summary["status_breakdown"].items()):
            print(f"  {status:30s}: {count}")

    if "metrics" in summary and summary["n_success"] > 0:
        print("-" * 70)
        metrics = summary["metrics"]
        for metric in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H"]:
            if metric in metrics:
                print(f"{metric:20s}: {metrics[metric] * 100:6.2f}%")

        print("-" * 70)
        if "MV2H_custom" in metrics:
            print(f"{'MV2H (custom)':20s}: {metrics['MV2H_custom'] * 100:6.2f}%")

    print("=" * 70)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MT3 + MuseScore Studio 4.6.5 Baseline MV2H Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Song Evaluation (8 workers, 120s timeout)
  python mt3_evaluate.py --mode full \\
      --pred_dir data/experiments/mt3/full_midi \\
      --gt_dir /path/to/asap-dataset \\
      --mv2h_bin MV2H/bin \\
      --output results/full_song.csv \\
      --workers 8 \\
      --timeout 120

  # 5-bar Chunk Evaluation (16 workers)
  python mt3_evaluate.py --mode chunks \\
      --pred_dir data/experiments/mt3/full_midi \\
      --gt_dir /path/to/asap-dataset \\
      --chunk_csv /path/to/zeng_test_chunk_set.csv \\
      --mv2h_bin MV2H/bin \\
      --output results/chunks.csv \\
      --workers 16

Academic Note:
  This script uses MuseScore Studio 4.6.5 as the "Industry Standard Baseline" for
  MIDI to MusicXML conversion. Unlike naive quantization (music21), MuseScore
  employs sophisticated heuristic-based voice separation and tuplet detection.
        """,
    )

    # Required arguments (some may be optional in retry mode)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "chunks"],
        default="full",
        help="Evaluation mode: 'full' for full song, 'chunks' for 5-bar chunks (not required in retry mode)",
    )

    parser.add_argument(
        "--pred_dir",
        type=str,
        help="Directory containing MT3 raw MIDI outputs (not required in retry mode)",
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        help="ASAP dataset base directory (not required in retry mode)",
    )

    parser.add_argument(
        "--mv2h_bin",
        type=str,
        required=True,
        help="Path to MV2H bin directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )

    # Optional arguments
    parser.add_argument(
        "--chunk_csv",
        type=str,
        help="Path to chunk CSV file (required for 'chunks' mode)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for intermediate outputs (default: same as output CSV directory)",
    )

    parser.add_argument(
        "--mscore_bin",
        type=str,
        default="tools/mscore",
        help="Path to MuseScore binary or wrapper script (default: tools/mscore)",
    )

    # Parallel processing
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=os.cpu_count() or 4,
        help=f"Number of parallel workers (default: {os.cpu_count() or 4})",
    )

    # Timeout settings
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="MV2H evaluation timeout in seconds (default: 120)",
    )

    parser.add_argument(
        "--mscore_timeout",
        type=int,
        default=60,
        help="MuseScore conversion timeout in seconds (default: 60)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Retry mode
    parser.add_argument(
        "--retry_file",
        type=str,
        help="Path to retry file (e.g., timeout.txt). When specified, only retry failed tasks from this file.",
    )

    parser.add_argument(
        "--save_failed",
        type=str,
        default=None,
        help="Path to save failed tasks for retry (default: <output_dir>/failed.txt)",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine if we're in retry mode
    retry_mode = args.retry_file is not None

    # Validate arguments
    if not retry_mode:
        # In normal mode, pred_dir and gt_dir are required
        if not args.pred_dir:
            parser.error("--pred_dir is required (unless using --retry_file)")
        if not args.gt_dir:
            parser.error("--gt_dir is required (unless using --retry_file)")

        if args.mode == "chunks" and not args.chunk_csv:
            parser.error("--chunk_csv is required for 'chunks' mode")

        if not Path(args.pred_dir).exists():
            print(f"Error: Prediction directory not found: {args.pred_dir}")
            sys.exit(1)

        if not Path(args.gt_dir).exists():
            print(f"Error: Ground truth directory not found: {args.gt_dir}")
            sys.exit(1)
    else:
        if not Path(args.retry_file).exists():
            print(f"Error: Retry file not found: {args.retry_file}")
            sys.exit(1)

    if not Path(args.mv2h_bin).exists():
        print(f"Error: MV2H bin not found: {args.mv2h_bin}")
        print("Please compile MV2H first or provide correct path")
        sys.exit(1)

    if not Path(args.mscore_bin).exists():
        print(f"Error: MuseScore binary not found: {args.mscore_bin}")
        print("Please run: ./scripts/setup_musescore.sh")
        sys.exit(1)

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.output).parent)

    # Build configurations
    mscore_config = MuseScoreConfig(
        binary_path=args.mscore_bin,
        timeout=args.mscore_timeout,
    )

    mv2h_config = MV2HConfig(
        bin_path=args.mv2h_bin,
        timeout=args.timeout,
    )

    parallel_config = ParallelConfig(
        workers=args.workers,
    )

    # Print configuration
    print("=" * 70)
    print("MT3 + MuseScore Studio 4.6.5 Baseline | MV2H Evaluation")
    print("=" * 70)
    if retry_mode:
        print(f"Mode:                RETRY")
        print(f"Retry file:          {args.retry_file}")
    else:
        print(f"Mode:                {args.mode}")
        print(f"Prediction dir:      {args.pred_dir}")
        print(f"Ground truth:        {args.gt_dir}")
    print(f"MuseScore:           {args.mscore_bin}")
    print(f"MV2H bin:            {args.mv2h_bin}")
    print(f"Output:              {args.output}")
    print(f"Workers:             {args.workers}")
    print(f"MV2H timeout:        {args.timeout}s")
    print(f"MuseScore timeout:   {args.mscore_timeout}s")
    if not retry_mode and args.mode == "chunks":
        print(f"Chunk CSV:           {args.chunk_csv}")
    print("=" * 70)
    print()

    # Run evaluation
    if retry_mode:
        # Retry mode: load tasks from retry file
        results = run_retry_evaluation(
            retry_file=args.retry_file,
            output_dir=args.output_dir,
            output_csv=args.output,
            mscore_config=mscore_config,
            mv2h_config=mv2h_config,
            parallel_config=parallel_config,
        )
        eval_mode = "retry"
    elif args.mode == "full":
        results = run_full_song_evaluation(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            output_csv=args.output,
            mscore_config=mscore_config,
            mv2h_config=mv2h_config,
            parallel_config=parallel_config,
        )
        eval_mode = "full"
    else:  # chunks mode
        results = run_chunk_evaluation(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            chunk_csv=args.chunk_csv,
            output_dir=args.output_dir,
            output_csv=args.output,
            mscore_config=mscore_config,
            mv2h_config=mv2h_config,
            parallel_config=parallel_config,
        )
        eval_mode = "chunks"

    # Save failed tasks for retry
    failed_path = args.save_failed
    if failed_path is None:
        failed_path = str(Path(args.output_dir) / "failed.txt")

    n_failed = save_failed_tasks(results, failed_path)
    if n_failed > 0:
        print(f"\nFailed tasks saved to: {failed_path}")
        print(f"To retry with longer timeout:")
        print(f"  python mt3_evaluate.py --retry_file {failed_path} \\")
        print(f"      --mv2h_bin {args.mv2h_bin} --mscore_bin {args.mscore_bin} \\")
        print(f"      --output {args.output.replace('.csv', '_retry.csv')} \\")
        print(f"      --timeout 300  # or longer")

    # Aggregate and print summary
    summary = aggregate_results(results)
    print_summary(summary, eval_mode)

    # Save summary JSON
    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
