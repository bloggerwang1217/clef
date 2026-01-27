"""
Mel Spectrogram Utilities
=========================

Provides functions for converting audio files to log-mel spectrograms.
Used in training data preparation (Phase 2.5) and inference.

Default parameters match clef-piano-base configuration:
- sample_rate: 16000 Hz
- n_mels: 128
- n_fft: 2048
- hop_length: 256 (16ms per frame)
- f_min: 20 Hz
- f_max: 8000 Hz

These defaults can be overridden by loading from a config file.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchaudio


# Default mel spectrogram parameters (fallback values)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 160
DEFAULT_F_MIN = 27.5
DEFAULT_F_MAX = 7040.0


def load_mel_config(config_path: Union[str, Path] = "configs/clef_piano_base.yaml") -> Dict[str, Any]:
    """Load mel spectrogram parameters from config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with mel parameters:
        - sample_rate, n_mels, n_fft, hop_length, f_min, f_max
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        # Return defaults if config not found
        return {
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "n_mels": DEFAULT_N_MELS,
            "n_fft": DEFAULT_N_FFT,
            "hop_length": DEFAULT_HOP_LENGTH,
            "f_min": DEFAULT_F_MIN,
            "f_max": DEFAULT_F_MAX,
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    audio_config = config.get("data", {}).get("audio", {})

    return {
        "sample_rate": audio_config.get("sample_rate", DEFAULT_SAMPLE_RATE),
        "n_mels": audio_config.get("n_mels", DEFAULT_N_MELS),
        "n_fft": audio_config.get("n_fft", DEFAULT_N_FFT),
        "hop_length": audio_config.get("hop_length", DEFAULT_HOP_LENGTH),
        "f_min": audio_config.get("f_min", DEFAULT_F_MIN),
        "f_max": audio_config.get("f_max", DEFAULT_F_MAX),
    }


def audio_to_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    f_min: float = DEFAULT_F_MIN,
    f_max: float = DEFAULT_F_MAX,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert waveform to log-mel spectrogram.

    Args:
        waveform: Audio tensor of shape [channels, samples]
        sample_rate: Input sample rate
        target_sample_rate: Target sample rate for resampling
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length between frames
        f_min: Minimum frequency for mel filterbank
        f_max: Maximum frequency for mel filterbank
        normalize: If True, apply per-sample normalization (mean=0, std=1)

    Returns:
        Log-mel spectrogram tensor of shape [1, n_mels, T]
        where T = num_samples / hop_length
    """
    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Ensure shape is [1, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )

    # Generate mel spectrogram: [1, n_mels, T]
    mel = mel_transform(waveform)

    # Log mel (with small epsilon for numerical stability)
    mel = torch.log(mel + 1e-9)

    # Per-sample normalization
    if normalize:
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    return mel


def load_audio_to_mel(
    audio_path: Union[str, Path],
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    f_min: float = DEFAULT_F_MIN,
    f_max: float = DEFAULT_F_MAX,
    normalize: bool = True,
) -> torch.Tensor:
    """Load audio file and convert to log-mel spectrogram.

    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
        target_sample_rate: Target sample rate
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length between frames
        f_min: Minimum frequency
        f_max: Maximum frequency
        normalize: If True, apply per-sample normalization

    Returns:
        Log-mel spectrogram tensor of shape [1, n_mels, T]
    """
    waveform, sample_rate = torchaudio.load(str(audio_path))

    return audio_to_mel(
        waveform=waveform,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        normalize=normalize,
    )


def process_audio_file(
    audio_path: Union[str, Path],
    mel_path: Union[str, Path],
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    f_min: float = DEFAULT_F_MIN,
    f_max: float = DEFAULT_F_MAX,
    normalize: bool = True,
    skip_existing: bool = True,
) -> Tuple[str, Optional[Tuple[int, ...]]]:
    """Process a single audio file and save mel spectrogram.

    Args:
        audio_path: Path to input audio file
        mel_path: Path to output mel file (.pt)
        target_sample_rate: Target sample rate
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length between frames
        f_min: Minimum frequency
        f_max: Maximum frequency
        normalize: If True, apply per-sample normalization
        skip_existing: If True, skip if mel_path already exists

    Returns:
        Tuple of (status, shape) where status is one of:
        - "generated": Successfully generated mel
        - "skipped": File already exists (if skip_existing=True)
        - "missing": Audio file not found
        - "failed": Processing error
        And shape is the mel tensor shape (or None on error)
    """
    audio_path = Path(audio_path)
    mel_path = Path(mel_path)

    if skip_existing and mel_path.exists():
        return "skipped", None

    if not audio_path.exists():
        return "missing", None

    try:
        mel = load_audio_to_mel(
            audio_path=audio_path,
            target_sample_rate=target_sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            normalize=normalize,
        )

        # Ensure parent directory exists
        mel_path.parent.mkdir(parents=True, exist_ok=True)

        # Save mel spectrogram
        torch.save(mel, str(mel_path))

        return "generated", tuple(mel.shape)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to process {audio_path}: {e}")
        return "failed", None


def duration_to_frames(
    duration_sec: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> int:
    """Calculate number of mel frames from audio duration.

    Args:
        duration_sec: Audio duration in seconds
        sample_rate: Sample rate
        hop_length: Hop length

    Returns:
        Number of mel frames
    """
    return int(duration_sec * sample_rate / hop_length)


def frames_to_duration(
    n_frames: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> float:
    """Calculate audio duration from number of mel frames.

    Args:
        n_frames: Number of mel frames
        sample_rate: Sample rate
        hop_length: Hop length

    Returns:
        Duration in seconds
    """
    return n_frames * hop_length / sample_rate
