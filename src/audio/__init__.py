"""Audio processing utilities."""

from .converter import convert_to_wav, load_audio
from .mel import (
    audio_to_mel,
    load_audio_to_mel,
    process_audio_file,
    load_mel_config,
    duration_to_frames,
    frames_to_duration,
)
from .separator import VocalSeparator
from .zeng_synthesis import MIDIProcess, render_one_midi, create_default_compressor

__all__ = [
    # converter
    "convert_to_wav",
    "load_audio",
    # mel
    "audio_to_mel",
    "load_audio_to_mel",
    "process_audio_file",
    "load_mel_config",
    "duration_to_frames",
    "frames_to_duration",
    # separator
    "VocalSeparator",
    # zeng_synthesis
    "MIDIProcess",
    "render_one_midi",
    "create_default_compressor",
]
