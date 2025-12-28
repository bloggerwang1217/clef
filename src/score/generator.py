"""
MusicXML generation utilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import music21

from .parser import NoteData

logger = logging.getLogger(__name__)


def create_part_from_notes(
    notes: List[NoteData],
    part_name: str = "Part",
    part_id: Optional[str] = None,
) -> music21.stream.Part:
    """
    Create a music21 Part from a list of NoteData objects.

    Args:
        notes: List of NoteData objects
        part_name: Name for the part
        part_id: ID for the part (defaults to lowercase part_name)

    Returns:
        music21 Part object
    """
    part = music21.stream.Part()
    part.partName = part_name
    part.id = part_id or part_name.lower().replace(" ", "_")

    # Sort notes by onset time
    sorted_notes = sorted(notes, key=lambda n: n.onset)

    for note_data in sorted_notes:
        note = music21.note.Note()
        note.pitch.midi = note_data.pitch
        note.quarterLength = note_data.duration
        part.insert(note_data.onset, note)

    return part


def create_musicxml_score(
    parts_data: Dict[str, List[NoteData]],
    title: Optional[str] = None,
    time_signature: str = "4/4",
    add_default_time_sig: bool = True,
) -> music21.stream.Score:
    """
    Create a MusicXML score from multiple parts.

    Args:
        parts_data: Dictionary mapping part names to lists of NoteData
        title: Optional title for the score
        time_signature: Time signature string (e.g., "4/4", "3/4")
        add_default_time_sig: Add time signature to parts if not present

    Returns:
        music21 Score object
    """
    score = music21.stream.Score()

    if title:
        score.insert(0, music21.metadata.Metadata())
        score.metadata.title = title

    for part_name, notes in parts_data.items():
        part = create_part_from_notes(notes, part_name)

        # Add time signature if requested
        if add_default_time_sig:
            existing_ts = part.getElementsByClass(music21.meter.TimeSignature)
            if not existing_ts:
                ts = music21.meter.TimeSignature(time_signature)
                part.insert(0, ts)

        score.insert(0, part)

    return score


def save_musicxml(
    score: music21.stream.Score,
    path: Union[str, Path],
) -> Path:
    """
    Save a music21 Score to MusicXML file.

    Args:
        score: music21 Score object
        path: Output file path

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing MusicXML to: {path}")
    score.write("musicxml", fp=str(path))

    return path


def notes_to_musicxml(
    parts_data: Dict[str, List[NoteData]],
    output_path: Union[str, Path],
    title: Optional[str] = None,
    time_signature: str = "4/4",
) -> Path:
    """
    Convenience function to create and save MusicXML in one step.

    Args:
        parts_data: Dictionary mapping part names to lists of NoteData
        output_path: Output file path
        title: Optional title for the score
        time_signature: Time signature string

    Returns:
        Path to saved file
    """
    score = create_musicxml_score(
        parts_data, title=title, time_signature=time_signature
    )
    return save_musicxml(score, output_path)


class ScoreBuilder:
    """
    Builder class for constructing scores incrementally.

    Example:
        builder = ScoreBuilder(title="My Score")
        builder.add_note("melody", pitch=60, onset=0, duration=1)
        builder.add_note("melody", pitch=62, onset=1, duration=1)
        builder.add_note("harmony", pitch=48, onset=0, duration=2)
        score = builder.build()
        builder.save("output.musicxml")
    """

    def __init__(
        self,
        title: Optional[str] = None,
        time_signature: str = "4/4",
    ):
        """
        Initialize the score builder.

        Args:
            title: Optional title for the score
            time_signature: Time signature string
        """
        self.title = title
        self.time_signature = time_signature
        self._parts: Dict[str, List[NoteData]] = {}

    def add_note(
        self,
        part_name: str,
        pitch: int,
        onset: float,
        duration: float,
    ) -> "ScoreBuilder":
        """
        Add a note to a part.

        Args:
            part_name: Name of the part
            pitch: MIDI note number
            onset: Start time in quarter notes
            duration: Duration in quarter notes

        Returns:
            Self for chaining
        """
        if part_name not in self._parts:
            self._parts[part_name] = []

        self._parts[part_name].append(
            NoteData(pitch=pitch, onset=onset, duration=duration, part=part_name)
        )
        return self

    def add_notes(
        self,
        part_name: str,
        notes: List[NoteData],
    ) -> "ScoreBuilder":
        """
        Add multiple notes to a part.

        Args:
            part_name: Name of the part
            notes: List of NoteData objects

        Returns:
            Self for chaining
        """
        if part_name not in self._parts:
            self._parts[part_name] = []

        self._parts[part_name].extend(notes)
        return self

    def build(self) -> music21.stream.Score:
        """
        Build and return the score.

        Returns:
            music21 Score object
        """
        return create_musicxml_score(
            self._parts,
            title=self.title,
            time_signature=self.time_signature,
        )

    def save(self, path: Union[str, Path]) -> Path:
        """
        Build and save the score.

        Args:
            path: Output file path

        Returns:
            Path to saved file
        """
        score = self.build()
        return save_musicxml(score, path)
