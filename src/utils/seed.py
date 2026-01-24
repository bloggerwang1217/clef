"""Seed utilities for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    This matches Zeng et al.'s seed setting for fair comparison.

    Args:
        seed: Random seed value

    Usage:
        # Data augmentation (use seed=0 to match Zeng)
        set_seed(0)

        # Training (use seed=1234 to match Zeng)
        set_seed(1234)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Default seeds (matching Zeng et al.)
SEED_DATA_AUGMENTATION = 0
SEED_TRAINING = 1234
