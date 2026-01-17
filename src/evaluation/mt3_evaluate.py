#!/usr/bin/env python3
"""
MT3 Baseline MV2H Evaluation Pipeline

This script provides comprehensive MV2H evaluation for MT3 baseline, supporting:
1. Full Song Evaluation - Evaluate the entire piece at once
2. 5-bar Chunk Evaluation - Apple-to-apple comparison with Zeng's method

The pipeline converts MT3's raw MIDI output through quantization and hand separation,
then evaluates against ASAP ground truth using the MV2H metric.

Design Rationale (Academic Justification):
==========================================

1. QUANTIZATION SETTINGS
   - Default: quarterLengthDivisors=(4, 3)
   - 4 = sixteenth notes (1/16)
   - 3 = eighth-note triplets (1/8T)
   - Reference: music21 default settings
     https://www.music21.org/music21docs/moduleReference/moduleMidiTranslate.html

2. HAND SEPARATION ALGORITHM
   - Method: Pitch-based heuristic (split at Middle C, MIDI pitch 60)
   - Reference: Hadjakos et al. "Detecting Hands from Piano MIDI Data" (2019)
   - This is the "industry-default" naive baseline used by MuseScore, Finale, etc.

3. MV2H EVALUATION
   - Reference: McLeod & Steedman "Evaluating Automatic Polyphonic Music Transcription"
   - Uses DTW alignment (-a flag) for robust comparison
   - Outputs 6 metrics: Multi-pitch, Voice, Meter, Value, Harmony, MV2H

Usage:
    # Full Song Evaluation
    python mt3_evaluate.py --mode full \
        --pred_dir data/experiments/mt3/full_midi \
        --gt_dir /path/to/asap-dataset \
        --mv2h_bin MV2H/bin \
        --output data/experiments/mt3/results/full_song.csv

    # 5-bar Chunk Evaluation
    python mt3_evaluate.py --mode chunks \
        --pred_dir data/experiments/mt3/full_midi \
        --gt_dir /path/to/asap-dataset \
        --chunk_csv /path/to/zeng_test_chunk_set.csv \
        --mv2h_bin MV2H/bin \
        --output data/experiments/mt3/results/chunks.csv

Author: Clef Project
License: Apache-2.0
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import music21
from music21 import chord, clef, converter, instrument, key, meter, note, stream
from tqdm import tqdm

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress music21 warnings for cleaner output
if hasattr(music21, "Music21DeprecationWarning"):
    warnings.filterwarnings("ignore", category=music21.Music21DeprecationWarning)
else:
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*music21.*deprecated.*"
    )


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================


@dataclass
class QuantizationConfig:
    """
    Quantization settings with academic justification.

    Reference: music21.midi.translate module documentation
    https://www.music21.org/music21docs/moduleReference/moduleMidiTranslate.html
    """

    # Default: sixteenth notes + eighth-note triplets
    # This is music21's default and covers most classical piano repertoire
    quarter_length_divisors: Tuple[int, ...] = (4, 3)

    # Process both note onsets and durations
    process_offsets: bool = True
    process_durations: bool = True


@dataclass
class HandSeparationConfig:
    """
    Hand separation settings with academic justification.

    Reference: Hadjakos et al. "Detecting Hands from Piano MIDI Data" (2019)
    - Naive pitch-based split is used as baseline
    - Split point at Middle C (MIDI 60) is industry standard
    """

    # Split point: Middle C (C4) = MIDI pitch 60
    # Notes >= 60 go to right hand (treble clef)
    # Notes < 60 go to left hand (bass clef)
    split_point: int = 60

    # Minimum notes required to create a hand part
    min_notes_per_hand: int = 1


@dataclass
class EvaluationConfig:
    """Configuration for MV2H evaluation."""

    # MV2H binary path
    mv2h_bin: str = "MV2H/bin"

    # Evaluation script path
    eval_script: str = "evaluate_midi_mv2h.sh"

    # Timeout for MV2H evaluation (seconds)
    timeout: int = 60


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
# MIDI CONVERSION FUNCTIONS
# =============================================================================


def load_midi(
    midi_path: str,
    quantize: bool = True,
    config: Optional[QuantizationConfig] = None,
) -> music21.stream.Score:
    """
    Load MIDI file and optionally quantize.

    Args:
        midi_path: Path to MIDI file
        quantize: Whether to apply quantization during parsing
        config: Quantization configuration

    Returns:
        music21 Score object
    """
    if config is None:
        config = QuantizationConfig()

    logger.debug(f"Loading MIDI: {midi_path}")

    if quantize:
        score = converter.parse(
            midi_path,
            format="midi",
            forceSource=True,
            quantizePost=True,
            quarterLengthDivisors=config.quarter_length_divisors,
        )
    else:
        score = converter.parse(
            midi_path,
            format="midi",
            forceSource=True,
            quantizePost=False,
        )

    return score


def quantize_score(
    score: music21.stream.Score,
    config: Optional[QuantizationConfig] = None,
) -> music21.stream.Score:
    """
    Apply quantization to a score.

    This is the critical step that converts continuous MIDI timing
    to discrete musical notation values.

    Args:
        score: music21 Score object
        config: Quantization configuration

    Returns:
        Quantized score
    """
    if config is None:
        config = QuantizationConfig()

    logger.debug(f"Quantizing with divisors: {config.quarter_length_divisors}")

    score.quantize(
        quarterLengthDivisors=list(config.quarter_length_divisors),
        processOffsets=config.process_offsets,
        processDurations=config.process_durations,
        inPlace=True,
    )

    return score


def separate_hands(
    score: music21.stream.Score,
    config: Optional[HandSeparationConfig] = None,
) -> music21.stream.Score:
    """
    Separate notes into right hand (treble) and left hand (bass) parts.

    Algorithm: Pitch-based heuristic split at Middle C (MIDI 60)

    Args:
        score: music21 Score object (typically single-track from MT3)
        config: Hand separation configuration

    Returns:
        New score with two parts (right hand, left hand)
    """
    if config is None:
        config = HandSeparationConfig()

    logger.debug(f"Separating hands at MIDI pitch {config.split_point}")

    # Create new score with piano instrument
    new_score = stream.Score()

    # Create right hand (treble) and left hand (bass) parts
    right_hand = stream.Part()
    right_hand.id = "Right Hand"
    right_hand.insert(0, instrument.Piano())
    right_hand.insert(0, clef.TrebleClef())

    left_hand = stream.Part()
    left_hand.id = "Left Hand"
    left_hand.insert(0, instrument.Piano())
    left_hand.insert(0, clef.BassClef())

    # Collect all notes from the original score
    all_notes = list(score.flatten().notesAndRests)

    # Track statistics
    rh_count = 0
    lh_count = 0

    for element in all_notes:
        if isinstance(element, note.Note):
            # Single note: assign based on pitch
            if element.pitch.midi >= config.split_point:
                right_hand.insert(element.offset, element)
                rh_count += 1
            else:
                left_hand.insert(element.offset, element)
                lh_count += 1

        elif isinstance(element, chord.Chord):
            # Chord: split by pitch, may result in separate chords per hand
            rh_pitches = []
            lh_pitches = []

            for pitch in element.pitches:
                if pitch.midi >= config.split_point:
                    rh_pitches.append(pitch)
                else:
                    lh_pitches.append(pitch)

            # Create separate chords for each hand
            if rh_pitches:
                rh_chord = chord.Chord(rh_pitches)
                rh_chord.duration = element.duration
                right_hand.insert(element.offset, rh_chord)
                rh_count += len(rh_pitches)

            if lh_pitches:
                lh_chord = chord.Chord(lh_pitches)
                lh_chord.duration = element.duration
                left_hand.insert(element.offset, lh_chord)
                lh_count += len(lh_pitches)

        elif isinstance(element, note.Rest):
            # Rests: add aligned rests to both hands
            ql = element.duration.quarterLength
            right_hand.insert(element.offset, note.Rest(quarterLength=ql))
            left_hand.insert(element.offset, note.Rest(quarterLength=ql))

    logger.debug(f"Hand separation complete: RH={rh_count} notes, LH={lh_count} notes")

    # Only add parts that have notes
    if rh_count >= config.min_notes_per_hand:
        new_score.insert(0, right_hand)
    if lh_count >= config.min_notes_per_hand:
        new_score.insert(0, left_hand)

    return new_score


def add_notation_elements(
    score: music21.stream.Score,
    time_signature: str = "4/4",
    key_signature: str = "C",
) -> music21.stream.Score:
    """
    Add essential notation elements for valid MusicXML output.

    Args:
        score: music21 Score object
        time_signature: Time signature string (e.g., '4/4', '3/4')
        key_signature: Key signature string (e.g., 'C', 'G', 'F#')

    Returns:
        Score with notation elements added
    """
    logger.debug(f"Adding notation: time={time_signature}, key={key_signature}")

    # Add time signature and key signature to each part
    for part in score.parts:
        if not part.flatten().getElementsByClass(meter.TimeSignature):
            part.insert(0, meter.TimeSignature(time_signature))

        if not part.flatten().getElementsByClass(key.KeySignature):
            part.insert(0, key.Key(key_signature))

    # Make notation (add measures, beams, etc.)
    score.makeNotation(inPlace=True)

    return score


def convert_mt3_midi(
    input_midi_path: str,
    output_midi_path: Optional[str] = None,
    output_musicxml_path: Optional[str] = None,
    quant_config: Optional[QuantizationConfig] = None,
    hand_config: Optional[HandSeparationConfig] = None,
) -> music21.stream.Score:
    """
    Full conversion pipeline: MT3 raw MIDI -> Quantized + Hand-separated Score

    Pipeline Steps:
    1. Load MIDI with quantization
    2. Separate hands using pitch-based heuristic
    3. Add notation elements
    4. Export to MIDI and/or MusicXML (optional)

    Args:
        input_midi_path: Input MT3 MIDI file path
        output_midi_path: Output quantized MIDI file path (optional)
        output_musicxml_path: Output MusicXML file path (optional)
        quant_config: Quantization configuration
        hand_config: Hand separation configuration

    Returns:
        Processed music21 Score object
    """
    # Step 1: Load and quantize
    score = load_midi(input_midi_path, quantize=True, config=quant_config)

    # Step 2: Separate hands
    score = separate_hands(score, config=hand_config)

    # Step 3: Add notation elements
    score = add_notation_elements(score)

    # Step 4: Export if paths provided
    if output_midi_path:
        Path(output_midi_path).parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=output_midi_path)
        logger.debug(f"Written quantized MIDI: {output_midi_path}")

    if output_musicxml_path:
        Path(output_musicxml_path).parent.mkdir(parents=True, exist_ok=True)
        score.write("musicxml", fp=output_musicxml_path)
        logger.debug(f"Written MusicXML: {output_musicxml_path}")

    return score


# =============================================================================
# GROUND TRUTH HANDLING
# =============================================================================


def find_asap_ground_truth(
    pred_path: str,
    asap_base_dir: str,
    pred_base_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Find corresponding ASAP ground truth MIDI for a prediction file.

    ASAP Structure:
        asap-dataset/Composer/Work/Piece/midi_score.mid

    Prediction file naming convention (MT3 output):
        <Composer>_<Work>_<Piece>_<PerformanceID>.mid
        Example: Bach_Prelude_bwv_875_ASAP_acpiano_16.mid

    Args:
        pred_path: Path to prediction MIDI file
        asap_base_dir: Base directory of ASAP dataset
        pred_base_dir: Base directory of predictions (for relative path extraction)

    Returns:
        Path to ground truth MIDI or None if not found
    """
    pred_name = Path(pred_path).stem

    # Try to parse MT3 output naming convention
    # Format: Composer_Work_Piece_PerformanceID
    parts = pred_name.split("_")

    if len(parts) >= 3:
        # Try various path combinations
        for i in range(1, len(parts) - 1):
            composer = parts[0]
            work = "_".join(parts[1:i + 1])
            piece = "_".join(parts[i + 1:-1]) if i + 1 < len(parts) - 1 else parts[i + 1]

            # Try: Composer/Work/Piece/midi_score.mid
            gt_path = Path(asap_base_dir) / composer / work / piece / "midi_score.mid"
            if gt_path.exists():
                return str(gt_path)

            # Try: Composer/Work/midi_score.mid
            gt_path = Path(asap_base_dir) / composer / work / "midi_score.mid"
            if gt_path.exists():
                return str(gt_path)

    # Fallback: Try to match by directory structure if pred_base_dir provided
    if pred_base_dir:
        try:
            rel_path = Path(pred_path).relative_to(pred_base_dir)
            # Use directory structure
            gt_path = Path(asap_base_dir) / rel_path.parent / "midi_score.mid"
            if gt_path.exists():
                return str(gt_path)
        except ValueError:
            pass

    return None


def convert_ground_truth_to_midi(
    gt_path: str,
    output_midi_path: str,
) -> bool:
    """
    Convert ground truth (MusicXML or MIDI) to MIDI format.

    Args:
        gt_path: Path to ground truth file (MusicXML or MIDI)
        output_midi_path: Output MIDI path

    Returns:
        True if successful, False otherwise
    """
    try:
        # If already MIDI, just copy or return the path
        if gt_path.lower().endswith((".mid", ".midi")):
            # Load and re-export to ensure consistent format
            score = converter.parse(gt_path, format="midi")
        else:
            # Assume MusicXML
            score = converter.parse(gt_path)

        Path(output_midi_path).parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=output_midi_path)
        return True

    except Exception as e:
        logger.error(f"Failed to convert GT: {gt_path} - {e}")
        return False


# =============================================================================
# MV2H EVALUATION
# =============================================================================


def run_mv2h(
    gt_midi_path: str,
    pred_midi_path: str,
    mv2h_bin: str,
    timeout: int = 60,
) -> Optional[MV2HResult]:
    """
    Run MV2H evaluation between ground truth and prediction MIDI files.

    Uses the MV2H Java tool with DTW alignment (-a flag).

    Args:
        gt_midi_path: Path to ground truth MIDI
        pred_midi_path: Path to prediction MIDI
        mv2h_bin: Path to MV2H bin directory
        timeout: Timeout in seconds

    Returns:
        MV2HResult object or None if failed
    """
    # Create temporary files for conversion output
    gt_conv = gt_midi_path + ".conv.txt"
    pred_conv = pred_midi_path + ".conv.txt"

    try:
        # Step 1: Convert GT MIDI to MV2H format
        result_gt = subprocess.run(
            ["java", "-cp", mv2h_bin, "mv2h.tools.Converter", "-i", gt_midi_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result_gt.returncode != 0:
            logger.warning(f"GT conversion failed: {result_gt.stderr}")
            return None

        with open(gt_conv, "w") as f:
            f.write(result_gt.stdout)

        # Step 2: Convert Pred MIDI to MV2H format
        result_pred = subprocess.run(
            ["java", "-cp", mv2h_bin, "mv2h.tools.Converter", "-i", pred_midi_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result_pred.returncode != 0:
            logger.warning(f"Pred conversion failed: {result_pred.stderr}")
            return None

        with open(pred_conv, "w") as f:
            f.write(result_pred.stdout)

        # Step 3: Run MV2H evaluation with DTW alignment
        result_mv2h = subprocess.run(
            [
                "java",
                "-cp",
                mv2h_bin,
                "mv2h.Main",
                "-g",
                gt_conv,
                "-t",
                pred_conv,
                "-a",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result_mv2h.returncode != 0:
            logger.warning(f"MV2H evaluation failed: {result_mv2h.stderr}")
            return None

        # Parse output (last 6 lines contain metrics)
        lines = result_mv2h.stdout.strip().splitlines()
        if len(lines) < 6:
            logger.warning(f"Unexpected MV2H output format: {result_mv2h.stdout}")
            return None

        metrics = {}
        for line in lines[-6:]:
            if ": " in line:
                metric_key, value = line.split(": ", 1)
                metrics[metric_key] = float(value)

        return MV2HResult.from_dict(metrics)

    except subprocess.TimeoutExpired:
        logger.warning(f"MV2H timeout after {timeout}s")
        return None

    except Exception as e:
        logger.error(f"MV2H execution error: {e}")
        return None

    finally:
        # Cleanup temporary files
        for f in [gt_conv, pred_conv]:
            if os.path.exists(f):
                os.remove(f)


# =============================================================================
# CHUNK EXTRACTION (5-BAR EVALUATION)
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


def extract_measures(
    score: music21.stream.Score,
    start_measure: int,
    end_measure: int,
) -> music21.stream.Score:
    """
    Extract a range of measures from a score.

    Args:
        score: music21 Score object
        start_measure: Starting measure number (1-indexed)
        end_measure: Ending measure number (inclusive)

    Returns:
        New score containing only the specified measures
    """
    new_score = stream.Score()

    for part in score.parts:
        new_part = stream.Part()
        new_part.id = part.id

        # Copy instrument and clef
        for elem in part.flatten().getElementsByClass([instrument.Instrument, clef.Clef]):
            new_part.insert(0, elem)

        # Extract measures
        measures = part.measures(start_measure, end_measure)
        if measures:
            for m in measures:
                new_part.append(m)

        if len(new_part.flatten().notes) > 0:
            new_score.insert(0, new_part)

    return new_score


# =============================================================================
# FULL SONG EVALUATION
# =============================================================================


@dataclass
class FullSongResult:
    """Result for a single full song evaluation."""

    file_id: str
    pred_path: str
    gt_path: str
    status: str
    metrics: Optional[MV2HResult] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/JSON export."""
        result = {
            "file_id": self.file_id,
            "pred_path": self.pred_path,
            "gt_path": self.gt_path,
            "status": self.status,
        }
        if self.metrics:
            result.update(self.metrics.to_dict())
        return result


def evaluate_full_song(
    pred_midi_path: str,
    gt_midi_path: str,
    output_dir: str,
    mv2h_bin: str,
    file_id: str,
    quant_config: Optional[QuantizationConfig] = None,
    hand_config: Optional[HandSeparationConfig] = None,
) -> FullSongResult:
    """
    Evaluate a single full song.

    Pipeline:
    1. Convert MT3 MIDI -> Quantized MIDI (with hand separation)
    2. Run MV2H evaluation against ground truth

    Args:
        pred_midi_path: Path to MT3 raw MIDI output
        gt_midi_path: Path to ground truth MIDI
        output_dir: Directory for intermediate outputs
        mv2h_bin: Path to MV2H bin
        file_id: Unique identifier for this file
        quant_config: Quantization configuration
        hand_config: Hand separation configuration

    Returns:
        FullSongResult object
    """
    # Output paths
    output_base = Path(output_dir)
    quantized_midi_path = str(output_base / "quantized_midi" / f"{file_id}.mid")
    quantized_xml_path = str(output_base / "quantized_musicxml" / f"{file_id}.musicxml")

    try:
        # Step 1: Convert MT3 MIDI
        convert_mt3_midi(
            pred_midi_path,
            output_midi_path=quantized_midi_path,
            output_musicxml_path=quantized_xml_path,
            quant_config=quant_config,
            hand_config=hand_config,
        )

        # Step 2: Run MV2H evaluation
        metrics = run_mv2h(gt_midi_path, quantized_midi_path, mv2h_bin)

        if metrics is None:
            return FullSongResult(
                file_id=file_id,
                pred_path=pred_midi_path,
                gt_path=gt_midi_path,
                status="mv2h_failed",
            )

        if metrics.mv2h == 0:
            return FullSongResult(
                file_id=file_id,
                pred_path=pred_midi_path,
                gt_path=gt_midi_path,
                status="zero_score",
            )

        return FullSongResult(
            file_id=file_id,
            pred_path=pred_midi_path,
            gt_path=gt_midi_path,
            status="success",
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Full song evaluation failed for {file_id}: {e}")
        return FullSongResult(
            file_id=file_id,
            pred_path=pred_midi_path,
            gt_path=gt_midi_path,
            status=f"error: {str(e)}",
        )


def run_full_song_evaluation(
    pred_dir: str,
    gt_dir: str,
    output_dir: str,
    mv2h_bin: str,
    output_csv: str,
) -> List[FullSongResult]:
    """
    Run full song evaluation on all MIDI files in prediction directory.

    Args:
        pred_dir: Directory containing MT3 MIDI outputs
        gt_dir: ASAP dataset base directory
        output_dir: Directory for intermediate outputs
        mv2h_bin: Path to MV2H bin
        output_csv: Path to output CSV file

    Returns:
        List of FullSongResult objects
    """
    pred_path = Path(pred_dir)
    output_path = Path(output_dir)

    # Create output directories
    (output_path / "quantized_midi").mkdir(parents=True, exist_ok=True)
    (output_path / "quantized_musicxml").mkdir(parents=True, exist_ok=True)

    # Find all MIDI files
    midi_files = list(pred_path.rglob("*.mid")) + list(pred_path.rglob("*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files in {pred_dir}")

    results = []

    for midi_file in tqdm(midi_files, desc="Full Song Evaluation"):
        file_id = midi_file.stem

        # Find ground truth
        gt_path = find_asap_ground_truth(str(midi_file), gt_dir, pred_dir)

        if gt_path is None:
            logger.warning(f"No ground truth found for: {file_id}")
            results.append(
                FullSongResult(
                    file_id=file_id,
                    pred_path=str(midi_file),
                    gt_path="",
                    status="no_ground_truth",
                )
            )
            continue

        # Evaluate
        result = evaluate_full_song(
            pred_midi_path=str(midi_file),
            gt_midi_path=gt_path,
            output_dir=output_dir,
            mv2h_bin=mv2h_bin,
            file_id=file_id,
        )
        results.append(result)

    # Save results to CSV
    save_results_to_csv(results, output_csv)

    return results


# =============================================================================
# 5-BAR CHUNK EVALUATION
# =============================================================================


@dataclass
class ChunkResult:
    """Result for a single chunk evaluation."""

    chunk_id: str
    piece_id: str
    start_measure: int
    end_measure: int
    pred_path: str
    gt_path: str
    status: str
    metrics: Optional[MV2HResult] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/JSON export."""
        result = {
            "chunk_id": self.chunk_id,
            "piece_id": self.piece_id,
            "start_measure": self.start_measure,
            "end_measure": self.end_measure,
            "pred_path": self.pred_path,
            "gt_path": self.gt_path,
            "status": self.status,
        }
        if self.metrics:
            result.update(self.metrics.to_dict())
        return result


def evaluate_chunk(
    pred_score: music21.stream.Score,
    gt_score: music21.stream.Score,
    chunk_info: ChunkInfo,
    mv2h_bin: str,
    temp_dir: str,
) -> ChunkResult:
    """
    Evaluate a single 5-bar chunk.

    Args:
        pred_score: Quantized prediction score
        gt_score: Ground truth score
        chunk_info: Chunk information
        mv2h_bin: Path to MV2H bin
        temp_dir: Temporary directory for intermediate files

    Returns:
        ChunkResult object
    """
    try:
        # Extract measures from both scores
        pred_chunk = extract_measures(
            pred_score,
            chunk_info.start_measure,
            chunk_info.end_measure,
        )
        gt_chunk = extract_measures(
            gt_score,
            chunk_info.start_measure,
            chunk_info.end_measure,
        )

        # Check if chunks have notes
        if len(pred_chunk.flatten().notes) == 0:
            return ChunkResult(
                chunk_id=chunk_info.chunk_id,
                piece_id=chunk_info.piece_id,
                start_measure=chunk_info.start_measure,
                end_measure=chunk_info.end_measure,
                pred_path="",
                gt_path=chunk_info.asap_path,
                status="empty_prediction",
            )

        if len(gt_chunk.flatten().notes) == 0:
            return ChunkResult(
                chunk_id=chunk_info.chunk_id,
                piece_id=chunk_info.piece_id,
                start_measure=chunk_info.start_measure,
                end_measure=chunk_info.end_measure,
                pred_path="",
                gt_path=chunk_info.asap_path,
                status="empty_ground_truth",
            )

        # Write temporary MIDI files
        pred_midi = os.path.join(temp_dir, f"{chunk_info.chunk_id}_pred.mid")
        gt_midi = os.path.join(temp_dir, f"{chunk_info.chunk_id}_gt.mid")

        pred_chunk.write("midi", fp=pred_midi)
        gt_chunk.write("midi", fp=gt_midi)

        # Run MV2H
        metrics = run_mv2h(gt_midi, pred_midi, mv2h_bin)

        if metrics is None:
            return ChunkResult(
                chunk_id=chunk_info.chunk_id,
                piece_id=chunk_info.piece_id,
                start_measure=chunk_info.start_measure,
                end_measure=chunk_info.end_measure,
                pred_path=pred_midi,
                gt_path=gt_midi,
                status="mv2h_failed",
            )

        if metrics.mv2h == 0:
            return ChunkResult(
                chunk_id=chunk_info.chunk_id,
                piece_id=chunk_info.piece_id,
                start_measure=chunk_info.start_measure,
                end_measure=chunk_info.end_measure,
                pred_path=pred_midi,
                gt_path=gt_midi,
                status="zero_score",
            )

        return ChunkResult(
            chunk_id=chunk_info.chunk_id,
            piece_id=chunk_info.piece_id,
            start_measure=chunk_info.start_measure,
            end_measure=chunk_info.end_measure,
            pred_path=pred_midi,
            gt_path=gt_midi,
            status="success",
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Chunk evaluation failed for {chunk_info.chunk_id}: {e}")
        return ChunkResult(
            chunk_id=chunk_info.chunk_id,
            piece_id=chunk_info.piece_id,
            start_measure=chunk_info.start_measure,
            end_measure=chunk_info.end_measure,
            pred_path="",
            gt_path=chunk_info.asap_path,
            status=f"error: {str(e)}",
        )


def run_chunk_evaluation(
    pred_dir: str,
    gt_dir: str,
    chunk_csv: str,
    output_dir: str,
    mv2h_bin: str,
    output_csv: str,
) -> List[ChunkResult]:
    """
    Run 5-bar chunk evaluation.

    Args:
        pred_dir: Directory containing MT3 MIDI outputs
        gt_dir: ASAP dataset base directory
        chunk_csv: Path to Zeng's chunk CSV file
        output_dir: Directory for intermediate outputs
        mv2h_bin: Path to MV2H bin
        output_csv: Path to output CSV file

    Returns:
        List of ChunkResult objects
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load chunk information
    chunks = load_chunk_csv(chunk_csv)

    # Group chunks by piece
    piece_chunks: Dict[str, List[ChunkInfo]] = {}
    for chunk in chunks:
        if chunk.piece_id not in piece_chunks:
            piece_chunks[chunk.piece_id] = []
        piece_chunks[chunk.piece_id].append(chunk)

    logger.info(f"Processing {len(piece_chunks)} pieces with {len(chunks)} total chunks")

    results = []

    # Process each piece
    with tempfile.TemporaryDirectory() as temp_dir:
        for piece_id, piece_chunk_list in tqdm(piece_chunks.items(), desc="Chunk Evaluation"):
            # Find prediction MIDI for this piece
            pred_path = Path(pred_dir)
            pred_files = list(pred_path.rglob(f"*{piece_id}*.mid"))

            if not pred_files:
                logger.warning(f"No prediction found for piece: {piece_id}")
                for chunk in piece_chunk_list:
                    results.append(
                        ChunkResult(
                            chunk_id=chunk.chunk_id,
                            piece_id=chunk.piece_id,
                            start_measure=chunk.start_measure,
                            end_measure=chunk.end_measure,
                            pred_path="",
                            gt_path=chunk.asap_path,
                            status="no_prediction",
                        )
                    )
                continue

            pred_file = pred_files[0]

            # Convert MT3 MIDI to quantized score
            try:
                pred_score = convert_mt3_midi(str(pred_file))
            except Exception as e:
                logger.error(f"Failed to convert prediction: {pred_file} - {e}")
                for chunk in piece_chunk_list:
                    results.append(
                        ChunkResult(
                            chunk_id=chunk.chunk_id,
                            piece_id=chunk.piece_id,
                            start_measure=chunk.start_measure,
                            end_measure=chunk.end_measure,
                            pred_path=str(pred_file),
                            gt_path=chunk.asap_path,
                            status="conversion_failed",
                        )
                    )
                continue

            # Find ground truth
            gt_path = find_asap_ground_truth(str(pred_file), gt_dir, pred_dir)
            if gt_path is None and piece_chunk_list[0].asap_path:
                gt_path = os.path.join(gt_dir, piece_chunk_list[0].asap_path)

            if gt_path is None or not os.path.exists(gt_path):
                logger.warning(f"No ground truth found for piece: {piece_id}")
                for chunk in piece_chunk_list:
                    results.append(
                        ChunkResult(
                            chunk_id=chunk.chunk_id,
                            piece_id=chunk.piece_id,
                            start_measure=chunk.start_measure,
                            end_measure=chunk.end_measure,
                            pred_path=str(pred_file),
                            gt_path="",
                            status="no_ground_truth",
                        )
                    )
                continue

            # Load ground truth score
            try:
                gt_score = converter.parse(gt_path)
            except Exception as e:
                logger.error(f"Failed to load ground truth: {gt_path} - {e}")
                for chunk in piece_chunk_list:
                    results.append(
                        ChunkResult(
                            chunk_id=chunk.chunk_id,
                            piece_id=chunk.piece_id,
                            start_measure=chunk.start_measure,
                            end_measure=chunk.end_measure,
                            pred_path=str(pred_file),
                            gt_path=gt_path,
                            status="gt_load_failed",
                        )
                    )
                continue

            # Evaluate each chunk
            for chunk in piece_chunk_list:
                result = evaluate_chunk(
                    pred_score=pred_score,
                    gt_score=gt_score,
                    chunk_info=chunk,
                    mv2h_bin=mv2h_bin,
                    temp_dir=temp_dir,
                )
                results.append(result)

    # Save results to CSV
    save_chunk_results_to_csv(results, output_csv)

    return results


# =============================================================================
# RESULTS AGGREGATION AND EXPORT
# =============================================================================


def save_results_to_csv(
    results: List[FullSongResult],
    output_path: str,
) -> None:
    """Save full song results to CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_id",
        "pred_path",
        "gt_path",
        "status",
        "Multi-pitch",
        "Voice",
        "Meter",
        "Value",
        "Harmony",
        "MV2H",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    logger.info(f"Results saved to: {output_path}")


def save_chunk_results_to_csv(
    results: List[ChunkResult],
    output_path: str,
) -> None:
    """Save chunk results to CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "chunk_id",
        "piece_id",
        "start_measure",
        "end_measure",
        "pred_path",
        "gt_path",
        "status",
        "Multi-pitch",
        "Voice",
        "Meter",
        "Value",
        "Harmony",
        "MV2H",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    logger.info(f"Results saved to: {output_path}")


def aggregate_results(
    results: List[FullSongResult],
) -> Dict[str, float]:
    """Aggregate MV2H results from full song evaluation."""
    successful = [r for r in results if r.status == "success" and r.metrics]

    if not successful:
        return {"n_samples": 0}

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

    avg["n_samples"] = n
    avg["n_failed"] = len(results) - n

    return avg


def aggregate_chunk_results(
    results: List[ChunkResult],
) -> Dict[str, float]:
    """Aggregate MV2H results from chunk evaluation."""
    successful = [r for r in results if r.status == "success" and r.metrics]

    if not successful:
        return {"n_samples": 0}

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

    avg["n_samples"] = n
    avg["n_failed"] = len(results) - n

    return avg


def print_summary(summary: Dict[str, float], mode: str) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "=" * 70)
    print(f"MT3 Baseline MV2H Evaluation Results ({mode} mode)")
    print("=" * 70)
    print(f"Successful evaluations: {summary.get('n_samples', 0)}")
    print(f"Failed evaluations:     {summary.get('n_failed', 0)}")
    print("-" * 70)

    for metric in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H"]:
        if metric in summary:
            print(f"{metric:20s}: {summary[metric] * 100:6.2f}%")

    print("-" * 70)
    if "MV2H_custom" in summary:
        print(f"{'MV2H (custom)':20s}: {summary['MV2H_custom'] * 100:6.2f}%")
    print("=" * 70)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MT3 Baseline MV2H Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Song Evaluation
  python mt3_evaluate.py --mode full \\
      --pred_dir data/experiments/mt3/full_midi \\
      --gt_dir /path/to/asap-dataset \\
      --mv2h_bin MV2H/bin \\
      --output data/experiments/mt3/results/full_song.csv

  # 5-bar Chunk Evaluation
  python mt3_evaluate.py --mode chunks \\
      --pred_dir data/experiments/mt3/full_midi \\
      --gt_dir /path/to/asap-dataset \\
      --chunk_csv /path/to/zeng_test_chunk_set.csv \\
      --mv2h_bin MV2H/bin \\
      --output data/experiments/mt3/results/chunks.csv
        """,
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full", "chunks"],
        help="Evaluation mode: 'full' for full song, 'chunks' for 5-bar chunks",
    )

    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory containing MT3 raw MIDI outputs",
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="ASAP dataset base directory",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if args.mode == "chunks" and not args.chunk_csv:
        parser.error("--chunk_csv is required for 'chunks' mode")

    if not Path(args.pred_dir).exists():
        print(f"Error: Prediction directory not found: {args.pred_dir}")
        sys.exit(1)

    if not Path(args.gt_dir).exists():
        print(f"Error: Ground truth directory not found: {args.gt_dir}")
        sys.exit(1)

    if not Path(args.mv2h_bin).exists():
        print(f"Error: MV2H bin not found: {args.mv2h_bin}")
        print("Please compile MV2H first or provide correct path")
        sys.exit(1)

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.output).parent)

    # Print configuration
    print(f"Mode:            {args.mode}")
    print(f"Prediction dir:  {args.pred_dir}")
    print(f"Ground truth:    {args.gt_dir}")
    print(f"MV2H bin:        {args.mv2h_bin}")
    print(f"Output:          {args.output}")
    if args.mode == "chunks":
        print(f"Chunk CSV:       {args.chunk_csv}")
    print()

    # Run evaluation
    if args.mode == "full":
        results = run_full_song_evaluation(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            mv2h_bin=args.mv2h_bin,
            output_csv=args.output,
        )
        summary = aggregate_results(results)

    else:  # chunks mode
        results = run_chunk_evaluation(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            chunk_csv=args.chunk_csv,
            output_dir=args.output_dir,
            mv2h_bin=args.mv2h_bin,
            output_csv=args.output,
        )
        summary = aggregate_chunk_results(results)

    # Print summary
    print_summary(summary, args.mode)

    # Save summary JSON
    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
