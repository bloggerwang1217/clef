"""Score processing utilities (MusicXML, Kern)."""

from .parser import parse_musicxml, extract_notes_from_score
from .generator import create_musicxml_score

__all__ = ["parse_musicxml", "extract_notes_from_score", "create_musicxml_score"]
