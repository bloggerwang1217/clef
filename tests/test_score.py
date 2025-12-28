"""Tests for score processing utilities."""

import pytest

from src.score.parser import NoteData


class TestNoteData:
    """Test NoteData dataclass."""

    def test_note_data_creation(self):
        """NoteData should store pitch, onset, duration correctly."""
        note = NoteData(pitch=60, onset=0.0, duration=1.0, part="Piano")
        assert note.pitch == 60
        assert note.onset == 0.0
        assert note.duration == 1.0
        assert note.part == "Piano"

    def test_note_data_end_time(self):
        """End time should be onset + duration."""
        note = NoteData(pitch=60, onset=2.0, duration=0.5)
        assert note.end == 2.5

    def test_note_data_offset_alias(self):
        """Offset should be an alias for onset."""
        note = NoteData(pitch=60, onset=1.5, duration=1.0)
        assert note.offset == note.onset


class TestScoreBuilder:
    """Test ScoreBuilder class."""

    def test_builder_chaining(self):
        """Builder methods should return self for chaining."""
        from src.score.generator import ScoreBuilder

        builder = ScoreBuilder(title="Test")
        result = builder.add_note("melody", pitch=60, onset=0, duration=1)
        assert result is builder
