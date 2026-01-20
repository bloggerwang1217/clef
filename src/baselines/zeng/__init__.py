# Copyright 2024 Wei Zeng (National University of Singapore)
# Licensed under the Apache License, Version 2.0
# Original source: https://github.com/wei-zeng98/piano-a2s
#
# This module contains code adapted from the piano-a2s repository for
# apple-to-apple comparison with Zeng et al.'s baseline.

from .synthesis import MIDIProcess, render_one_midi, create_default_compressor

__all__ = [
    "MIDIProcess",
    "render_one_midi",
    "create_default_compressor",
]
