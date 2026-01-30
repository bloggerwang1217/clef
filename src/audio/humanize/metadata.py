"""
Metadata tracking for reproducibility.

Every humanized MIDI should have metadata recording all settings
to ensure experiments can be reproduced.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional
import json
from pathlib import Path


@dataclass
class HumanizationMetadata:
    """Metadata for reproducibility."""

    source_file: str  # Original kern/midi path
    version: int  # Augmentation version (0-3)
    seed: Optional[int]  # Random seed
    timestamp: str  # ISO format
    k_values: Dict[str, float]  # All k values
    base_config: Dict  # HumanizationConfig settings
    soundfont: Optional[str] = None  # If rendered
    partitura_version: Optional[str] = None
    humanize_version: str = "0.1.0"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, output_path: Path):
        """
        Save metadata as JSON sidecar file.

        Args:
            output_path: Path to MIDI file (metadata saved as .json)
        """
        from pathlib import Path
        output_path = Path(output_path)
        json_path = output_path.with_suffix('.json')
        json_path.write_text(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> 'HumanizationMetadata':
        """Load from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def load(cls, midi_path: Path) -> Optional['HumanizationMetadata']:
        """
        Load metadata from JSON sidecar file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            HumanizationMetadata or None if not found
        """
        json_path = midi_path.with_suffix('.json')

        if not json_path.exists():
            return None

        return cls.from_json(json_path.read_text())


def create_metadata(
    source_file: str,
    version: int,
    config,
    seed: Optional[int] = None,
    soundfont: Optional[str] = None
) -> HumanizationMetadata:
    """
    Create metadata for a humanization run.

    Args:
        source_file: Source file path
        version: Version number (0-3)
        config: HumanizationConfig object
        seed: Random seed
        soundfont: Soundfont used for rendering

    Returns:
        HumanizationMetadata object
    """
    from . import __version__

    # Try to get partitura version
    partitura_version = None
    try:
        import partitura
        partitura_version = partitura.__version__
    except Exception:
        pass

    return HumanizationMetadata(
        source_file=source_file,
        version=version,
        seed=seed,
        timestamp=datetime.now().isoformat(),
        k_values=config.to_dict(),
        base_config={
            'reference_velocity': config.reference_velocity,
            'default_velocity': config.default_velocity,
            'default_bpm': config.default_bpm,
        },
        soundfont=soundfont,
        partitura_version=partitura_version,
        humanize_version=__version__,
    )
