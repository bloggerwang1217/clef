"""Score processing utilities (MusicXML, Kern)."""

from .generate_score import kern_to_musicxml
from .clean_kern import (
    extract_visual_info,
    extract_visual_from_sequence,
    strip_cue_passages,
    strip_non_kern_spines,
    clean_kern_token,
    clean_kern_sequence,
)
from .sanitize_piano_score import heal_cross_staff

__all__ = [
    "kern_to_musicxml",
    # Visual information extraction (for Visual Auxiliary Head)
    "extract_visual_info",
    "extract_visual_from_sequence",
    # Cue passage removal
    "strip_cue_passages",
    # Spine filtering
    "strip_non_kern_spines",
    # Kern token cleaning
    "clean_kern_token",
    "clean_kern_sequence",
    # Score sanitization
    "heal_cross_staff",
]
