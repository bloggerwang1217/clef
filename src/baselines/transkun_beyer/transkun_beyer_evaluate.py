#!/usr/bin/env python3
"""
Transkun + Beyer SOTA Baseline MV2H Evaluation Pipeline

This script evaluates the Transkun + Beyer SOTA pipeline:
    Audio → Transkun (SOTA AMT) → MIDI → Beyer (ISMIR'24) → MusicXML → MV2H

This is the learned pipeline, in contrast to the industrial pipeline:
    Audio → MT3 → MIDI → MuseScore (rule-based) → MusicXML → MV2H

Evaluation Modes:
    full   - Evaluate entire pieces
    chunks - 5-bar chunk evaluation (Zeng et al.)

Usage:
    # Full song evaluation
    python -m src.baselines.transkun_beyer.transkun_beyer_evaluate --mode full \\
        --audio_dir data/asap_audio \\
        --gt_dir /path/to/asap \\
        --output results/transkun_beyer.csv

    # Chunk evaluation
    python -m src.baselines.transkun_beyer.transkun_beyer_evaluate --mode chunks \\
        --audio_dir data/asap_audio \\
        --gt_dir /path/to/asap \\
        --chunk_csv src/evaluation/asap/test_chunk_set.csv \\
        --output results/transkun_beyer_chunks.csv

    # Using YAML config
    python -m src.baselines.transkun_beyer.transkun_beyer_evaluate \\
        --config configs/transkun_beyer_evaluate.yaml
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

# Import inference modules
from src.baselines.transkun_beyer.transkun_inference import transcribe_audio
from src.baselines.transkun_beyer.beyer_inference import convert_midi_to_musicxml

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
class TranskunConfig:
    """Transkun configuration."""
    device: str = "cuda"
    model_version: str = "v2"
    timeout: int = 600


@dataclass
class BeyerConfig:
    """Beyer configuration."""
    checkpoint_path: str = "/home/bloggerwang/MIDI2ScoreTransformer/MIDI2ScoreTF.ckpt"
    conda_env: str = "beyer"
    beyer_repo: str = "/home/bloggerwang/MIDI2ScoreTransformer"
    musescore_path: str = "tools/mscore"
    device: str = "cuda"
    timeout: int = 300


@dataclass
class MuseScoreConfig:
    """MuseScore configuration for MusicXML → MIDI conversion."""
    binary_path: str = "tools/mscore"
    timeout: int = 60
    force_overwrite: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    mv2h_bin: str = "MV2H/bin"
    mv2h_timeout: int = 120
    mv2h_chunk_timeout: int = 10
    transkun_config: TranskunConfig = None
    beyer_config: BeyerConfig = None
    mscore_config: MuseScoreConfig = None
    workers: int = 1  # Sequential by default due to GPU usage

    def __post_init__(self):
        if self.transkun_config is None:
            self.transkun_config = TranskunConfig()
        if self.beyer_config is None:
            self.beyer_config = BeyerConfig()
        if self.mscore_config is None:
            self.mscore_config = MuseScoreConfig()


# =============================================================================
# YAML CONFIG LOADING
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# MUSESCORE CONVERSION (for MV2H input)
# =============================================================================


def convert_musicxml_to_midi(
    musicxml_path: str,
    midi_path: str,
    config: MuseScoreConfig,
) -> bool:
    """Convert MusicXML to MIDI using MuseScore (for MV2H evaluation)."""
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
# PIPELINE STAGES
# =============================================================================


def run_transkun(
    audio_path: str,
    output_midi_path: str,
    config: TranskunConfig,
) -> bool:
    """
    Stage 1: Transcribe audio to MIDI using Transkun.

    Returns:
        True if transcription succeeded
    """
    return transcribe_audio(
        audio_path,
        output_midi_path,
        device=config.device,
        model_version=config.model_version,
    )


def run_beyer(
    midi_path: str,
    output_xml_path: str,
    config: BeyerConfig,
) -> bool:
    """
    Stage 2: Convert MIDI to MusicXML using Beyer Transformer.

    Returns:
        True if conversion succeeded
    """
    return convert_midi_to_musicxml(
        midi_path,
        output_xml_path,
        checkpoint_path=config.checkpoint_path,
        conda_env=config.conda_env,
        beyer_repo=config.beyer_repo,
        musescore_path=config.musescore_path,
        device=config.device,
        timeout=config.timeout,
    )


# =============================================================================
# EVALUATION TASK
# =============================================================================


@dataclass
class EvalTask:
    """Single evaluation task."""
    task_id: str
    audio_path: str  # Input audio
    gt_midi_path: str  # Ground truth MIDI
    output_dir: str  # For intermediate files
    transkun_config: TranskunConfig
    beyer_config: BeyerConfig
    mscore_config: MuseScoreConfig
    mv2h_bin: str
    mv2h_timeout: int


@dataclass
class EvalResult:
    """Evaluation result."""
    task_id: str
    audio_path: str
    gt_path: str
    status: str
    metrics: Optional[MV2HResult] = None
    error_message: str = ""
    pred_midi_path: str = ""
    pred_xml_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "audio_path": self.audio_path,
            "gt_path": self.gt_path,
            "status": self.status,
            "error_message": self.error_message,
            "pred_midi_path": self.pred_midi_path,
            "pred_xml_path": self.pred_xml_path,
        }
        if self.metrics:
            result.update(self.metrics.to_dict())
        return result


def evaluate_single_task(task: EvalTask) -> EvalResult:
    """
    Evaluate single audio file through full pipeline.

    Pipeline:
    1. Audio → Transkun → MIDI
    2. MIDI → Beyer → MusicXML
    3. MusicXML → MuseScore → MIDI (for MV2H)
    4. MV2H evaluation
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Transkun (Audio → MIDI)
            pred_midi = os.path.join(temp_dir, f"{task.task_id}_transkun.mid")
            if not run_transkun(task.audio_path, pred_midi, task.transkun_config):
                return EvalResult(
                    task_id=task.task_id,
                    audio_path=task.audio_path,
                    gt_path=task.gt_midi_path,
                    status="transkun_failed",
                )

            # Step 2: Beyer (MIDI → MusicXML)
            pred_xml = os.path.join(temp_dir, f"{task.task_id}_beyer.musicxml")
            if not run_beyer(pred_midi, pred_xml, task.beyer_config):
                return EvalResult(
                    task_id=task.task_id,
                    audio_path=task.audio_path,
                    gt_path=task.gt_midi_path,
                    status="beyer_failed",
                    pred_midi_path=pred_midi,
                )

            # Step 3: MusicXML → MIDI (for MV2H)
            pred_midi_converted = os.path.join(temp_dir, f"{task.task_id}_converted.mid")
            if not convert_musicxml_to_midi(pred_xml, pred_midi_converted, task.mscore_config):
                return EvalResult(
                    task_id=task.task_id,
                    audio_path=task.audio_path,
                    gt_path=task.gt_midi_path,
                    status="midi_conversion_failed",
                    pred_midi_path=pred_midi,
                    pred_xml_path=pred_xml,
                )

            # Save outputs if output_dir specified
            if task.output_dir:
                saved_midi = os.path.join(task.output_dir, "midi_from_transkun", f"{task.task_id}.mid")
                saved_xml = os.path.join(task.output_dir, "musicxml_from_beyer", f"{task.task_id}.musicxml")
                Path(saved_midi).parent.mkdir(parents=True, exist_ok=True)
                Path(saved_xml).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pred_midi, saved_midi)
                shutil.copy2(pred_xml, saved_xml)

            # Step 4: MV2H evaluation
            evaluator = MV2HEvaluator(task.mv2h_bin, timeout=task.mv2h_timeout)
            metrics = evaluator.evaluate(task.gt_midi_path, pred_midi_converted)

            if metrics is None:
                return EvalResult(
                    task_id=task.task_id,
                    audio_path=task.audio_path,
                    gt_path=task.gt_midi_path,
                    status="mv2h_failed",
                    pred_midi_path=pred_midi,
                    pred_xml_path=pred_xml,
                )

            return EvalResult(
                task_id=task.task_id,
                audio_path=task.audio_path,
                gt_path=task.gt_midi_path,
                status="success",
                metrics=metrics,
                pred_midi_path=pred_midi,
                pred_xml_path=pred_xml,
            )

    except Exception as e:
        return EvalResult(
            task_id=task.task_id,
            audio_path=task.audio_path,
            gt_path=task.gt_midi_path,
            status="error",
            error_message=str(e),
        )


# =============================================================================
# SEQUENTIAL EVALUATION (GPU-bound)
# =============================================================================


def run_sequential_evaluation(
    tasks: List[EvalTask],
) -> List[EvalResult]:
    """Run tasks sequentially (for GPU-bound inference)."""
    results = []
    logger.info(f"Running {len(tasks)} tasks sequentially...")

    for task in tqdm(tasks, desc="Evaluating"):
        results.append(evaluate_single_task(task))

    return results


# =============================================================================
# FULL SONG EVALUATION
# =============================================================================


def run_full_song_evaluation(
    audio_dir: str,
    gt_dir: str,
    output_dir: str,
    config: EvalConfig,
) -> List[EvalResult]:
    """
    Run full song evaluation.

    This mode:
    1. Finds all audio files in audio_dir
    2. Matches them with ground truth MIDI from ASAP
    3. Runs the full Transkun + Beyer pipeline
    4. Evaluates with MV2H
    """
    asap = ASAPDataset(gt_dir)
    audio_path = Path(audio_dir)

    # Find audio files
    audio_files = list(audio_path.rglob("*.wav")) + list(audio_path.rglob("*.mp3"))
    logger.info(f"Found {len(audio_files)} audio files")

    # Build tasks
    tasks = []
    skipped = []

    for audio_file in audio_files:
        file_id = audio_file.stem
        gt_path = asap.find_ground_truth_midi(str(audio_file))

        if gt_path is None:
            skipped.append(file_id)
            continue

        tasks.append(EvalTask(
            task_id=file_id,
            audio_path=str(audio_file),
            gt_midi_path=gt_path,
            output_dir=output_dir,
            transkun_config=config.transkun_config,
            beyer_config=config.beyer_config,
            mscore_config=config.mscore_config,
            mv2h_bin=config.mv2h_bin,
            mv2h_timeout=config.mv2h_timeout,
        ))

    if skipped:
        logger.warning(f"Skipped {len(skipped)} files without ground truth")

    # Run evaluation (sequential due to GPU)
    results = run_sequential_evaluation(tasks)

    # Add skipped
    for file_id in skipped:
        results.append(EvalResult(
            task_id=file_id, audio_path="", gt_path="", status="no_ground_truth"
        ))

    return results


# =============================================================================
# CHUNK EVALUATION
# =============================================================================


def find_audio_file(
    audio_dir: str,
    piece_id: str,
    performance: str,
) -> Optional[str]:
    """
    Find audio file for given piece and performance.

    Supports two directory structures:
    1. Flat: audio_dir/performance.wav
    2. ASAP-style: audio_dir/Composer/Work/Piece/performance.wav
    """
    audio_path = Path(audio_dir)

    # Convert piece_id from '#' separator to path components
    path_parts = piece_id.split("#")

    # Try ASAP-style structure first
    search_dir = audio_path / "/".join(path_parts)
    if search_dir.exists():
        for ext in [".wav", ".mp3", ".flac"]:
            audio_file = search_dir / f"{performance}{ext}"
            if audio_file.exists():
                return str(audio_file)

        # Glob pattern
        for audio_file in search_dir.glob(f"{performance}*"):
            if audio_file.suffix.lower() in [".wav", ".mp3", ".flac"]:
                return str(audio_file)

    # Try flat structure
    flat_name = "_".join(path_parts) + f"_{performance}"
    for ext in [".wav", ".mp3", ".flac"]:
        flat_path = audio_path / f"{flat_name}{ext}"
        if flat_path.exists():
            return str(flat_path)

    # Recursive search as fallback
    for audio_file in audio_path.rglob(f"*{performance}*.wav"):
        return str(audio_file)

    return None


def group_chunks_by_piece_performance(
    chunks: List[ChunkInfo],
) -> Dict[Tuple[str, str], List[ChunkInfo]]:
    """Group chunks by (piece_id, performance)."""
    grouped: Dict[Tuple[str, str], List[ChunkInfo]] = {}

    for chunk in chunks:
        parts = chunk.chunk_id.rsplit("#", 1)
        if len(parts) != 2:
            continue

        piece_id = parts[0]
        perf_chunk = parts[1]
        performance = perf_chunk.rsplit(".", 1)[0]

        key = (piece_id, performance)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(chunk)

    return grouped


@dataclass
class ChunkEvalTask:
    """Evaluation task for a single 5-bar chunk."""
    task_id: str
    pred_chunk_midi: str
    gt_chunk_midi: str
    mv2h_bin: str
    mv2h_timeout: int


def evaluate_chunk_task(task: ChunkEvalTask) -> EvalResult:
    """Evaluate a single 5-bar chunk with MV2H."""
    try:
        evaluator = MV2HEvaluator(task.mv2h_bin, timeout=task.mv2h_timeout)
        metrics = evaluator.evaluate(task.gt_chunk_midi, task.pred_chunk_midi)

        if metrics is None:
            return EvalResult(
                task_id=task.task_id,
                audio_path="",
                gt_path=task.gt_chunk_midi,
                status="mv2h_failed",
            )

        if metrics.mv2h == 0 and metrics.multi_pitch == 0:
            return EvalResult(
                task_id=task.task_id,
                audio_path="",
                gt_path=task.gt_chunk_midi,
                status="zero_score",
                metrics=metrics,
            )

        return EvalResult(
            task_id=task.task_id,
            audio_path="",
            gt_path=task.gt_chunk_midi,
            status="success",
            metrics=metrics,
        )

    except subprocess.TimeoutExpired:
        return EvalResult(
            task_id=task.task_id,
            audio_path="",
            gt_path=task.gt_chunk_midi,
            status="timeout",
        )
    except Exception as e:
        return EvalResult(
            task_id=task.task_id,
            audio_path="",
            gt_path=task.gt_chunk_midi,
            status="error",
            error_message=str(e),
        )


def run_chunk_evaluation(
    audio_dir: str,
    gt_dir: str,
    chunk_csv: str,
    output_dir: str,
    config: EvalConfig,
    output_csv: Optional[str] = None,
) -> Tuple[List[EvalResult], Dict[str, Any]]:
    """
    Run 5-bar chunk evaluation.

    Strategy:
    1. For each (piece, performance), transcribe full audio with Transkun
    2. Convert full MIDI to MusicXML with Beyer
    3. Extract 5-bar chunks from MusicXML using music21
    4. Run MV2H on each chunk
    """
    asap = ASAPDataset(gt_dir)
    chunks = asap.load_chunks(chunk_csv)
    grouped = group_chunks_by_piece_performance(chunks)

    total_chunks = len(chunks)
    logger.info(f"Processing {len(grouped)} (piece, performance) pairs, {total_chunks} total chunks")

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    midi_dir = os.path.join(output_dir, "midi_from_transkun")
    xml_dir = os.path.join(output_dir, "musicxml_from_beyer")
    chunk_midi_dir = os.path.join(output_dir, "chunk_midi")
    Path(midi_dir).mkdir(parents=True, exist_ok=True)
    Path(xml_dir).mkdir(parents=True, exist_ok=True)
    Path(chunk_midi_dir).mkdir(parents=True, exist_ok=True)

    # Load completed chunks if resuming
    completed_chunks = set()
    if output_csv and Path(output_csv).exists():
        completed_chunks = load_completed_chunks(output_csv)
        if completed_chunks:
            logger.info(f"Found {len(completed_chunks)} already completed chunks")
    if output_csv:
        init_csv_file(output_csv, CHUNK_CSV_FIELDS)

    # Process each (piece, performance) pair
    chunk_tasks: List[ChunkEvalTask] = []
    skipped_pieces = []
    pipeline_errors = []
    skipped_completed = 0

    for (piece_id, performance), piece_chunks in tqdm(
        grouped.items(),
        desc="Running pipeline",
        unit="piece",
    ):
        # 1. Find audio file
        audio_file = find_audio_file(audio_dir, piece_id, performance)
        if audio_file is None:
            skipped_pieces.append(f"{piece_id}#{performance}")
            continue

        # 2. Find ground truth MusicXML
        gt_xml = asap.find_ground_truth_xml_by_piece_id(piece_id)
        if gt_xml is None:
            skipped_pieces.append(f"{piece_id}#{performance}")
            continue

        # 3. Check if already processed
        pred_midi = os.path.join(midi_dir, f"{piece_id.replace('#', '_')}_{performance}.mid")
        pred_xml = os.path.join(xml_dir, f"{piece_id.replace('#', '_')}_{performance}.musicxml")

        # Run Transkun if MIDI doesn't exist
        if not Path(pred_midi).exists():
            logger.info(f"Transcribing: {piece_id}#{performance}")
            if not run_transkun(audio_file, pred_midi, config.transkun_config):
                pipeline_errors.append(f"{piece_id}#{performance} (transkun)")
                continue

        # Run Beyer if MusicXML doesn't exist
        if not Path(pred_xml).exists():
            logger.info(f"Converting: {piece_id}#{performance}")
            if not run_beyer(pred_midi, pred_xml, config.beyer_config):
                pipeline_errors.append(f"{piece_id}#{performance} (beyer)")
                continue

        # 4. Filter completed chunks
        chunks_to_extract = []
        for chunk in piece_chunks:
            if chunk.chunk_id in completed_chunks:
                skipped_completed += 1
            else:
                chunks_to_extract.append(chunk)

        if not chunks_to_extract:
            continue

        # 5. Extract chunks from both pred and GT
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

        # Batch extract
        pred_results = extract_chunks_batch(pred_xml, pred_chunks_list)
        gt_results = extract_chunks_batch(gt_xml, gt_chunks_list)

        # Create chunk evaluation tasks
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
                chunk_tasks.append(ChunkEvalTask(
                    task_id=chunk.chunk_id,
                    pred_chunk_midi=pred_chunk,
                    gt_chunk_midi=gt_chunk,
                    mv2h_bin=config.mv2h_bin,
                    mv2h_timeout=config.mv2h_chunk_timeout,
                ))

    if skipped_completed:
        logger.info(f"Skipped {skipped_completed} already completed chunks")
    if skipped_pieces:
        logger.warning(f"Skipped {len(skipped_pieces)} pieces (no audio/GT)")
    if pipeline_errors:
        logger.warning(f"Pipeline failed for {len(pipeline_errors)} pieces")

    logger.info(f"Prepared {len(chunk_tasks)} chunk evaluation tasks")

    # Run MV2H evaluation in parallel (CPU-bound)
    new_results: List[EvalResult] = []
    n_workers = min(config.workers, len(chunk_tasks)) if chunk_tasks else 1

    if chunk_tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(evaluate_chunk_task, t): t for t in chunk_tasks}

            for future in tqdm(
                as_completed(futures),
                total=len(chunk_tasks),
                desc="MV2H evaluation",
            ):
                try:
                    result = future.result()
                except Exception as e:
                    task = futures[future]
                    result = EvalResult(
                        task_id=task.task_id,
                        audio_path="",
                        gt_path=task.gt_chunk_midi,
                        status="executor_error",
                        error_message=str(e),
                    )

                new_results.append(result)

                # Incremental save
                if output_csv:
                    append_result_to_csv(result, output_csv, CHUNK_CSV_FIELDS)

    # Load all results for summary
    all_results = new_results
    if output_csv and completed_chunks:
        all_results = load_results_from_csv(output_csv)

    # Compute summary
    summary = compute_chunk_summary(all_results, total_chunks)

    return all_results, summary


# =============================================================================
# RESULTS OUTPUT
# =============================================================================


CHUNK_CSV_FIELDS = [
    "task_id", "chunk_index", "piece_id", "performance",
    "status", "error_message",
    "Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom",
    "audio_path", "gt_path",
]

FULL_CSV_FIELDS = [
    "task_id", "status", "error_message",
    "Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom",
    "audio_path", "gt_path", "pred_midi_path", "pred_xml_path",
]


def parse_chunk_id(chunk_id: str) -> Dict[str, Any]:
    """Parse Zeng-style chunk_id into components."""
    parts = chunk_id.rsplit("#", 1)
    if len(parts) != 2:
        return {"piece_id": chunk_id, "performance": "", "chunk_index": 0}

    piece_id = parts[0]
    perf_chunk = parts[1]

    perf_parts = perf_chunk.rsplit(".", 1)
    if len(perf_parts) == 2:
        performance = perf_parts[0]
        try:
            chunk_index = int(perf_parts[1])
        except ValueError:
            chunk_index = 0
    else:
        performance = perf_chunk
        chunk_index = 0

    return {"piece_id": piece_id, "performance": performance, "chunk_index": chunk_index}


def load_completed_chunks(csv_path: str) -> set:
    """Load task_ids of completed chunks from CSV."""
    completed = set()
    if not Path(csv_path).exists():
        return completed

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "success":
                    completed.add(row.get("task_id", ""))
    except Exception as e:
        logger.warning(f"Failed to read CSV: {e}")

    return completed


def init_csv_file(csv_path: str, fieldnames: List[str]) -> None:
    """Initialize CSV with headers if doesn't exist."""
    if Path(csv_path).exists():
        return

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()


def append_result_to_csv(result: EvalResult, csv_path: str, fieldnames: List[str]) -> None:
    """Append result to CSV."""
    row = result.to_dict()

    if "chunk_index" in fieldnames:
        parsed = parse_chunk_id(result.task_id)
        row.update(parsed)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def load_results_from_csv(csv_path: str) -> List[EvalResult]:
    """Load all results from CSV."""
    results = []
    if not Path(csv_path).exists():
        return results

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                    audio_path=row.get("audio_path", ""),
                    gt_path=row.get("gt_path", ""),
                    status=row.get("status", "unknown"),
                    metrics=metrics,
                    error_message=row.get("error_message", ""),
                ))
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")

    return results


def save_results_csv(results: List[EvalResult], output_path: str, is_chunk_mode: bool = False) -> None:
    """Save all results to CSV."""
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


def compute_chunk_summary(
    results: List[EvalResult],
    total_chunks: int,
) -> Dict[str, Any]:
    """Compute chunk evaluation summary."""
    successful = [r.metrics for r in results if r.status == "success" and r.metrics]
    n_successful = len(successful)
    n_evaluated = len(results)
    n_failed = n_evaluated - n_successful

    status_counts: Dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    # Zeng's method (exclude failures)
    if successful:
        zeng_metrics = aggregate_mv2h_results(successful)
    else:
        zeng_metrics = {
            "Multi-pitch": 0.0, "Voice": 0.0, "Meter": 0.0,
            "Value": 0.0, "Harmony": 0.0, "MV2H": 0.0, "MV2H_custom": 0.0,
        }

    # Include failures as 0
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
    """Print chunk evaluation summary."""
    print("\n" + "=" * 70)
    print("Transkun + Beyer | MV2H Chunk Evaluation Summary")
    print("=" * 70)

    print(f"\nTotal chunks in CSV:  {summary['n_total']}")
    print(f"Evaluated:            {summary['n_evaluated']}")
    print(f"Successful:           {summary['n_successful']} ({summary['success_rate']*100:.1f}%)")
    print(f"Failed:               {summary['n_failed']}")

    print("\nStatus breakdown:")
    for status, count in sorted(summary["status_breakdown"].items()):
        print(f"  {status:20s}: {count}")

    print("\n" + "-" * 70)
    print("Method 1: Zeng's method (exclude failures)")
    print("-" * 70)
    zeng = summary["zeng_method"]
    print(f"Samples: {summary['n_successful']}")
    for metric in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony"]:
        print(f"  {metric:15s}: {zeng.get(metric, 0.0) * 100:6.2f}%")
    print(f"  {'MV2H (official)':15s}: {zeng.get('MV2H', 0.0) * 100:6.2f}%")
    print(f"  {'MV2H (custom)':15s}: {zeng.get('MV2H_custom', 0.0) * 100:6.2f}%")

    print("\n" + "-" * 70)
    print("Method 2: Include failures as 0")
    print("-" * 70)
    incl = summary["include_failures"]
    print(f"Samples: {summary['n_total']} (failures counted as 0)")
    for metric in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony"]:
        print(f"  {metric:15s}: {incl.get(metric, 0.0) * 100:6.2f}%")
    print(f"  {'MV2H (official)':15s}: {incl.get('MV2H', 0.0) * 100:6.2f}%")
    print(f"  {'MV2H (custom)':15s}: {incl.get('MV2H_custom', 0.0) * 100:6.2f}%")

    print("=" * 70)


def compute_summary(results: List[EvalResult]) -> Dict[str, Any]:
    """Compute summary for full song evaluation."""
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
        description="Transkun + Beyer SOTA Baseline MV2H Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file
    parser.add_argument("--config", help="YAML config file")

    # Mode
    parser.add_argument("--mode", choices=["full", "chunks"], default=None)
    parser.add_argument("--audio_dir", help="Audio directory")
    parser.add_argument("--gt_dir", help="ASAP dataset directory")
    parser.add_argument("--chunk_csv", help="Chunk CSV (for chunks mode)")

    # Paths
    parser.add_argument("--mv2h_bin", help="MV2H bin directory")
    parser.add_argument("--mscore_bin", help="MuseScore binary")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--output_dir", help="Output directory")

    # Transkun settings
    parser.add_argument("--transkun_device", default=None)
    parser.add_argument("--transkun_model", default=None)

    # Beyer settings
    parser.add_argument("--beyer_checkpoint", default=None)
    parser.add_argument("--beyer_conda_env", default=None)
    parser.add_argument("--beyer_repo", default=None)
    parser.add_argument("--beyer_device", default=None)

    # Processing
    parser.add_argument("-j", "--workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--chunk_timeout", type=int, default=None)

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load YAML config
    file_config: Dict[str, Any] = {}
    if args.config:
        if not Path(args.config).exists():
            print(f"Config file not found: {args.config}")
            sys.exit(1)
        file_config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")

    # Merge config
    def get_value(cli_val, config_key, default=None):
        if cli_val is not None:
            return cli_val
        return file_config.get(config_key, default)

    mode = get_value(args.mode, "mode", "full")
    audio_dir = get_value(args.audio_dir, "audio_dir")
    gt_dir = get_value(args.gt_dir, "gt_dir")
    chunk_csv = get_value(args.chunk_csv, "chunk_csv")
    mv2h_bin = get_value(args.mv2h_bin, "mv2h_bin", "MV2H/bin")
    mscore_bin = get_value(args.mscore_bin, "mscore_bin", "tools/mscore")
    output_path = get_value(args.output, "output_csv", "results/transkun_beyer.csv")
    output_dir = get_value(args.output_dir, "output_dir", "data/experiments/transkun_beyer")
    workers = get_value(args.workers, "workers", 8)
    timeout = get_value(args.timeout, "timeout", 120)
    chunk_timeout = get_value(args.chunk_timeout, "chunk_timeout", 10)

    # Transkun config
    transkun_device = get_value(args.transkun_device, "transkun_device", "cuda")
    transkun_model = get_value(args.transkun_model, "transkun_model", "v2")

    # Beyer config
    beyer_checkpoint = get_value(
        args.beyer_checkpoint, "beyer_checkpoint",
        "/home/bloggerwang/MIDI2ScoreTransformer/MIDI2ScoreTF.ckpt"
    )
    beyer_conda_env = get_value(args.beyer_conda_env, "beyer_conda_env", "beyer")
    beyer_repo = get_value(
        args.beyer_repo, "beyer_repo",
        "/home/bloggerwang/MIDI2ScoreTransformer"
    )
    beyer_device = get_value(args.beyer_device, "beyer_device", "cuda")

    # Validate
    if not audio_dir:
        parser.error("--audio_dir required (or use --config)")
    if not gt_dir:
        parser.error("--gt_dir required (or use --config)")
    if mode == "chunks" and not chunk_csv:
        parser.error("--chunk_csv required for chunks mode")

    if not Path(audio_dir).exists():
        print(f"Audio directory not found: {audio_dir}")
        sys.exit(1)
    if not Path(gt_dir).exists():
        print(f"Ground truth directory not found: {gt_dir}")
        sys.exit(1)
    if not Path(mv2h_bin).exists():
        print(f"MV2H bin not found: {mv2h_bin}")
        sys.exit(1)

    # Build config
    config = EvalConfig(
        mv2h_bin=mv2h_bin,
        mv2h_timeout=timeout,
        mv2h_chunk_timeout=chunk_timeout,
        transkun_config=TranskunConfig(
            device=transkun_device,
            model_version=transkun_model,
        ),
        beyer_config=BeyerConfig(
            checkpoint_path=beyer_checkpoint,
            conda_env=beyer_conda_env,
            beyer_repo=beyer_repo,
            musescore_path=mscore_bin,
            device=beyer_device,
        ),
        mscore_config=MuseScoreConfig(binary_path=mscore_bin),
        workers=workers,
    )

    # Print config
    print("=" * 70)
    print("Transkun + Beyer SOTA Baseline | MV2H Evaluation")
    print("=" * 70)
    print(f"Mode:            {mode}")
    print(f"Workers:         {workers}")
    print(f"Audio dir:       {audio_dir}")
    print(f"Ground truth:    {gt_dir}")
    print(f"Output:          {output_path}")
    print(f"Transkun:        device={transkun_device}, model={transkun_model}")
    print(f"Beyer:           env={beyer_conda_env}, device={beyer_device}")
    if mode == "chunks":
        print(f"Chunk CSV:       {chunk_csv}")
    print("=" * 70)

    # Run evaluation
    if mode == "full":
        results = run_full_song_evaluation(audio_dir, gt_dir, output_dir, config)
        summary = compute_summary(results)
        print_mv2h_summary(summary, "Transkun + Beyer (full)")
        save_results_csv(results, output_path, is_chunk_mode=False)
    else:
        results, summary = run_chunk_evaluation(
            audio_dir, gt_dir, chunk_csv, output_dir, config,
            output_csv=output_path,
        )
        print_chunk_summary(summary)

    # Save summary JSON
    summary_path = Path(output_path).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults CSV:  {output_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
