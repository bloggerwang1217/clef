"""
MusicXML parsing utilities.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import music21

logger = logging.getLogger(__name__)


@dataclass
class NoteData:
    """Represents a single note with its properties."""

    pitch: int  # MIDI note number (0-127)
    onset: float  # Start time in quarter notes
    duration: float  # Duration in quarter notes
    part: str = ""  # Part name
    voice: int = 1  # Voice number within part

    @property
    def offset(self) -> float:
        """Alias for onset (music21 terminology)."""
        return self.onset

    @property
    def end(self) -> float:
        """End time in quarter notes."""
        return self.onset + self.duration


def parse_musicxml(path: Union[str, Path]) -> music21.stream.Score:
    """
    Parse a MusicXML file and return a music21 Score object.

    Args:
        path: Path to MusicXML file

    Returns:
        music21 Score object

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MusicXML file not found: {path}")

    logger.info(f"Parsing MusicXML: {path.name}")

    try:
        score = music21.converter.parse(str(path))
        logger.info(f"Successfully parsed score with {len(score.parts)} parts")
        return score
    except Exception as e:
        logger.error(f"Failed to parse MusicXML: {e}")
        raise


def extract_notes_from_score(
    score: music21.stream.Score,
    parts: Optional[List[str]] = None,
) -> List[NoteData]:
    """
    Extract all notes from a score as NoteData objects.

    Args:
        score: music21 Score object
        parts: List of part names to include (None = all parts)

    Returns:
        List of NoteData objects
    """
    notes = []

    for part in score.parts:
        part_name = part.partName or "Unknown"

        # Filter by part name if specified
        if parts is not None and part_name not in parts:
            continue

        logger.debug(f"Extracting notes from part: {part_name}")

        for element in part.flatten().notesAndRests:
            if isinstance(element, music21.note.Note):
                note_data = NoteData(
                    pitch=element.pitch.midi,
                    onset=float(element.offset),
                    duration=float(element.quarterLength),
                    part=part_name,
                )
                notes.append(note_data)

            elif isinstance(element, music21.chord.Chord):
                # Expand chords into individual notes
                for pitch in element.pitches:
                    note_data = NoteData(
                        pitch=pitch.midi,
                        onset=float(element.offset),
                        duration=float(element.quarterLength),
                        part=part_name,
                    )
                    notes.append(note_data)

    logger.info(f"Extracted {len(notes)} notes from score")
    return notes


def extract_notes_from_file(
    path: Union[str, Path],
    parts: Optional[List[str]] = None,
) -> List[NoteData]:
    """
    Convenience function to parse file and extract notes in one step.

    Args:
        path: Path to MusicXML file
        parts: List of part names to include (None = all parts)

    Returns:
        List of NoteData objects
    """
    score = parse_musicxml(path)
    return extract_notes_from_score(score, parts)


def get_part_names(score: music21.stream.Score) -> List[str]:
    """
    Get list of part names in a score.

    Args:
        score: music21 Score object

    Returns:
        List of part names
    """
    return [part.partName or f"Part {i+1}" for i, part in enumerate(score.parts)]


def get_score_info(score: music21.stream.Score) -> dict:
    """
    Get basic information about a score.

    Args:
        score: music21 Score object

    Returns:
        Dictionary with score metadata
    """
    info = {
        "num_parts": len(score.parts),
        "part_names": get_part_names(score),
        "total_duration": float(score.duration.quarterLength),
    }

    # Try to get time signature
    time_sigs = score.flatten().getElementsByClass(music21.meter.TimeSignature)
    if time_sigs:
        info["time_signature"] = str(time_sigs[0])

    # Try to get key signature
    key_sigs = score.flatten().getElementsByClass(music21.key.KeySignature)
    if key_sigs:
        info["key_signature"] = str(key_sigs[0])

    # Try to get tempo
    tempos = score.flatten().getElementsByClass(music21.tempo.MetronomeMark)
    if tempos:
        info["tempo"] = tempos[0].number

    return info
