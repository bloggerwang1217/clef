#!/usr/bin/env python3
"""
ASAP Dataset Handling Module

Standalone module for ASAP (Aligned Scores and Performances) dataset operations.
Can be used by any model: MT3 baseline, Clef, Transkun+Beyer, etc.

ASAP Dataset Structure:
    asap-dataset/
    ├── Bach/
    │   ├── Prelude/
    │   │   ├── bwv_875/
    │   │   │   ├── midi_score.mid      <- Ground truth MIDI
    │   │   │   ├── xml_score.musicxml  <- Ground truth MusicXML
    │   │   │   └── <performance_id>/   <- Performance recordings
    │   │   └── ...
    │   └── ...
    └── ...

Chunk Evaluation:
    For 5-bar chunk evaluation (Zeng et al.), this module supports:
    1. Loading chunk definitions from CSV
    2. Extracting measure ranges from MIDI/MusicXML
    3. Matching predictions to ground truth

Usage:
    from evaluation.asap import ASAPDataset, ChunkInfo

    asap = ASAPDataset("/path/to/asap-dataset")
    gt_path = asap.find_ground_truth("Bach_Prelude_bwv_875_performance1")

    # For chunk evaluation
    chunks = asap.load_chunks("/path/to/zeng_test_chunk_set.csv")
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ChunkInfo:
    """
    Information about a 5-bar chunk for evaluation.

    Used for Zeng-style chunk evaluation where full songs are
    divided into 5-bar segments for fine-grained assessment.

    Attributes:
        chunk_id: Unique identifier for the chunk
        piece_id: Identifier of the parent piece
        start_measure: Starting measure number (1-indexed)
        end_measure: Ending measure number (inclusive)
        asap_path: Relative path within ASAP dataset
    """

    chunk_id: str
    piece_id: str
    start_measure: int
    end_measure: int
    asap_path: str = ""

    def __repr__(self) -> str:
        return f"ChunkInfo({self.chunk_id}, m{self.start_measure}-{self.end_measure})"


@dataclass
class PieceInfo:
    """
    Information about a piece in the ASAP dataset.

    Attributes:
        piece_id: Unique identifier (e.g., "Bach_Prelude_bwv_875")
        composer: Composer name
        work: Work name
        piece: Piece name (may be empty)
        midi_score_path: Path to ground truth MIDI
        xml_score_path: Path to ground truth MusicXML
        performances: List of performance IDs
    """

    piece_id: str
    composer: str
    work: str
    piece: str = ""
    midi_score_path: str = ""
    xml_score_path: str = ""
    performances: List[str] = field(default_factory=list)


# =============================================================================
# ASAP DATASET CLASS
# =============================================================================


class ASAPDataset:
    """
    ASAP dataset handler for ground truth operations.

    Provides utilities for:
    - Finding ground truth files for predictions
    - Loading chunk definitions for 5-bar evaluation
    - Iterating over pieces and performances

    Args:
        base_dir: Path to ASAP dataset root directory

    Example:
        asap = ASAPDataset("/data/asap-dataset")

        # Find ground truth for a prediction
        gt_path = asap.find_ground_truth_midi("Bach_Prelude_bwv_875_perf1")

        # Load chunks for evaluation
        chunks = asap.load_chunks("zeng_test_chunks.csv")
    """

    # Common ground truth filenames in ASAP
    GT_MIDI_NAMES = ["midi_score.mid", "midi_score.midi"]
    GT_XML_NAMES = ["xml_score.musicxml", "xml_score.xml"]

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

        if not self.base_dir.exists():
            raise FileNotFoundError(f"ASAP dataset not found: {base_dir}")

        logger.info(f"ASAP dataset initialized: {base_dir}")

    def find_ground_truth_midi(
        self,
        pred_identifier: str,
        pred_base_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find ground truth MIDI for a prediction file.

        Tries multiple naming conventions and path patterns to match.

        Args:
            pred_identifier: Prediction file stem or path
                Examples:
                - "Bach_Prelude_bwv_875_performance1"
                - "/path/to/pred/Bach_Prelude_bwv_875_performance1.mid"
            pred_base_dir: Base directory of predictions (for relative path extraction)

        Returns:
            Path to ground truth MIDI or None if not found
        """
        # Extract filename stem if full path provided
        if os.path.sep in pred_identifier or pred_identifier.endswith((".mid", ".midi")):
            pred_name = Path(pred_identifier).stem
        else:
            pred_name = pred_identifier

        # Try to parse naming convention: Composer_Work_Piece_PerformanceID
        # or: Composer_Work_PerformanceID
        parts = pred_name.split("_")

        if len(parts) >= 3:
            composer = parts[0]

            # Try different path structures
            for i in range(1, len(parts)):
                for j in range(i + 1, len(parts) + 1):
                    work = "_".join(parts[1:j])
                    piece = "_".join(parts[j:-1]) if j < len(parts) - 1 else ""

                    # Try: Composer/Work/Piece/midi_score.mid
                    if piece:
                        for gt_name in self.GT_MIDI_NAMES:
                            gt_path = self.base_dir / composer / work / piece / gt_name
                            if gt_path.exists():
                                return str(gt_path)

                    # Try: Composer/Work/midi_score.mid
                    for gt_name in self.GT_MIDI_NAMES:
                        gt_path = self.base_dir / composer / work / gt_name
                        if gt_path.exists():
                            return str(gt_path)

        logger.debug(f"No ground truth found for: {pred_identifier}")
        return None

    def find_ground_truth_xml(
        self,
        pred_identifier: str,
    ) -> Optional[str]:
        """
        Find ground truth MusicXML for a prediction file.

        Args:
            pred_identifier: Prediction file stem or path

        Returns:
            Path to ground truth MusicXML or None if not found
        """
        # First find the MIDI ground truth
        midi_path = self.find_ground_truth_midi(pred_identifier)
        if midi_path is None:
            return None

        # Look for XML in same directory
        midi_dir = Path(midi_path).parent
        for xml_name in self.GT_XML_NAMES:
            xml_path = midi_dir / xml_name
            if xml_path.exists():
                return str(xml_path)

        return None

    def load_chunks(self, csv_path: str) -> List[ChunkInfo]:
        """
        Load chunk definitions from CSV file.

        Expected CSV format:
            chunk_id,piece_id,start_measure,end_measure,asap_path

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

    def group_chunks_by_piece(
        self,
        chunks: List[ChunkInfo],
    ) -> Dict[str, List[ChunkInfo]]:
        """
        Group chunks by piece ID.

        Args:
            chunks: List of ChunkInfo objects

        Returns:
            Dictionary mapping piece_id to list of chunks
        """
        grouped: Dict[str, List[ChunkInfo]] = {}
        for chunk in chunks:
            if chunk.piece_id not in grouped:
                grouped[chunk.piece_id] = []
            grouped[chunk.piece_id].append(chunk)
        return grouped

    def iter_pieces(self) -> List[PieceInfo]:
        """
        Iterate over all pieces in the dataset.

        Yields:
            PieceInfo objects for each piece found
        """
        pieces = []

        for composer_dir in self.base_dir.iterdir():
            if not composer_dir.is_dir():
                continue
            composer = composer_dir.name

            for work_dir in composer_dir.iterdir():
                if not work_dir.is_dir():
                    continue
                work = work_dir.name

                # Check if this directory has ground truth (is a piece)
                has_gt = any(
                    (work_dir / name).exists()
                    for name in self.GT_MIDI_NAMES + self.GT_XML_NAMES
                )

                if has_gt:
                    # This is a piece directory
                    piece_id = f"{composer}_{work}"
                    midi_path = ""
                    xml_path = ""

                    for name in self.GT_MIDI_NAMES:
                        if (work_dir / name).exists():
                            midi_path = str(work_dir / name)
                            break

                    for name in self.GT_XML_NAMES:
                        if (work_dir / name).exists():
                            xml_path = str(work_dir / name)
                            break

                    pieces.append(
                        PieceInfo(
                            piece_id=piece_id,
                            composer=composer,
                            work=work,
                            midi_score_path=midi_path,
                            xml_score_path=xml_path,
                        )
                    )
                else:
                    # Check subdirectories for pieces
                    for piece_dir in work_dir.iterdir():
                        if not piece_dir.is_dir():
                            continue

                        has_piece_gt = any(
                            (piece_dir / name).exists()
                            for name in self.GT_MIDI_NAMES + self.GT_XML_NAMES
                        )

                        if has_piece_gt:
                            piece = piece_dir.name
                            piece_id = f"{composer}_{work}_{piece}"
                            midi_path = ""
                            xml_path = ""

                            for name in self.GT_MIDI_NAMES:
                                if (piece_dir / name).exists():
                                    midi_path = str(piece_dir / name)
                                    break

                            for name in self.GT_XML_NAMES:
                                if (piece_dir / name).exists():
                                    xml_path = str(piece_dir / name)
                                    break

                            pieces.append(
                                PieceInfo(
                                    piece_id=piece_id,
                                    composer=composer,
                                    work=work,
                                    piece=piece,
                                    midi_score_path=midi_path,
                                    xml_score_path=xml_path,
                                )
                            )

        logger.info(f"Found {len(pieces)} pieces in ASAP dataset")
        return pieces


# =============================================================================
# CHUNK EXTRACTION UTILITIES
# =============================================================================


def extract_measures_midi(
    midi_path: str,
    start_measure: int,
    end_measure: int,
    output_path: str,
) -> bool:
    """
    Extract a range of measures from a MIDI file.

    Note: This requires music21 for measure-based extraction.

    Args:
        midi_path: Path to input MIDI file
        start_measure: Starting measure (1-indexed)
        end_measure: Ending measure (inclusive)
        output_path: Path to output MIDI file

    Returns:
        True if successful, False otherwise
    """
    try:
        from music21 import converter

        score = converter.parse(midi_path)
        extracted = score.measures(start_measure, end_measure)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        extracted.write("midi", fp=output_path)
        return True

    except ImportError:
        logger.error("music21 required for measure extraction: pip install music21")
        return False

    except Exception as e:
        logger.error(f"Measure extraction failed: {e}")
        return False


def extract_measures_musicxml(
    musicxml_path: str,
    start_measure: int,
    end_measure: int,
    output_path: str,
) -> bool:
    """
    Extract a range of measures from a MusicXML file.

    Args:
        musicxml_path: Path to input MusicXML file
        start_measure: Starting measure (1-indexed)
        end_measure: Ending measure (inclusive)
        output_path: Path to output MusicXML file

    Returns:
        True if successful, False otherwise
    """
    try:
        from music21 import converter

        score = converter.parse(musicxml_path)
        extracted = score.measures(start_measure, end_measure)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        extracted.write("musicxml", fp=output_path)
        return True

    except ImportError:
        logger.error("music21 required for measure extraction: pip install music21")
        return False

    except Exception as e:
        logger.error(f"Measure extraction failed: {e}")
        return False
