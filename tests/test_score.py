"""Tests for score processing utilities."""

import pytest
import tempfile
from pathlib import Path


class TestKernToMusicxml:
    """Test kern_to_musicxml function."""

    def test_kern_to_musicxml_import(self):
        """kern_to_musicxml should be importable from src.score."""
        from src.score import kern_to_musicxml
        assert callable(kern_to_musicxml)
