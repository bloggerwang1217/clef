"""
dB ↔ MIDI velocity conversion utilities.

Based on KTH Director Musices standard:
- 0 dB = MIDI velocity 64
- Conversion uses logarithmic relationship
"""

import numpy as np
from typing import Dict


def velocity_to_dB(velocity: int, reference: int = 64) -> float:
    """
    Convert MIDI velocity to dB (0 dB = reference velocity).

    Args:
        velocity: MIDI velocity (1-127)
        reference: Reference velocity for 0 dB (default 64)

    Returns:
        dB value relative to reference
    """
    if velocity <= 0:
        return -np.inf
    return 20 * np.log10(velocity / reference)


def dB_to_velocity(dB: float, reference: int = 64) -> int:
    """
    Convert dB back to MIDI velocity, clamped to 1-127.

    Args:
        dB: dB value relative to reference
        reference: Reference velocity for 0 dB (default 64)

    Returns:
        MIDI velocity (1-127)
    """
    velocity = reference * (10 ** (dB / 20))
    return int(np.clip(velocity, 1, 127))


def dynamics_to_velocity(marking: str, dynamics_map: Dict[str, int]) -> int:
    """
    Convert dynamic marking (p, f, etc.) to velocity.

    Args:
        marking: Dynamic marking string (e.g., "pp", "f", "sfz")
        dynamics_map: Mapping from marking to velocity

    Returns:
        MIDI velocity
    """
    marking_lower = marking.lower().strip()
    return dynamics_map.get(marking_lower, 64)  # Default to mf
