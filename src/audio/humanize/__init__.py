"""
KTH-style MIDI humanization with k-value system.

Usage:
    from src.audio.humanize import HumanizationEngine, HumanizationConfig

    config = HumanizationConfig().randomize(seed=42)
    engine = HumanizationEngine(config)
    engine.humanize('score.krn', 'output.mid')
"""

__version__ = "0.1.0"

from .config import RuleConfig, HumanizationConfig
from .engine import HumanizationEngine
from .presets import ROMANTIC, CLASSICAL, BALANCED

__all__ = [
    'RuleConfig',
    'HumanizationConfig',
    'HumanizationEngine',
    'ROMANTIC',
    'CLASSICAL',
    'BALANCED',
]
