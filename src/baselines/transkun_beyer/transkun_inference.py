#!/usr/bin/env python3
"""
Transkun Inference Module

Provides functions to transcribe audio files to MIDI using Transkun.
Transkun is a SOTA automatic music transcription model.

Installation:
    pip install transkun

Usage:
    python -m src.baselines.transkun_beyer.transkun_inference \
        --input audio.mp3 --output output.mid --device cuda

References:
    - https://github.com/transkun/transkun
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def check_transkun_installed() -> bool:
    """Check if transkun is installed."""
    try:
        import transkun
        return True
    except ImportError:
        return False


def transcribe_audio(
    audio_path: str,
    output_midi_path: str,
    device: str = "cuda",
    model_version: str = "v2",
) -> bool:
    """
    Transcribe audio file to MIDI using Transkun.

    Args:
        audio_path: Path to input audio file (mp3, wav, etc.)
        output_midi_path: Path to output MIDI file
        device: Device to use ('cuda' or 'cpu')
        model_version: Transkun model version ('v2', 'v2_aug', 'v2_no_ext')

    Returns:
        True if transcription succeeded, False otherwise
    """
    # Create output directory if needed
    Path(output_midi_path).parent.mkdir(parents=True, exist_ok=True)

    # Use transkun CLI (installed as entry point)
    cmd = [
        "transkun",
        audio_path,
        output_midi_path,
        "--device", device,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for long audio
        )

        if result.returncode == 0 and Path(output_midi_path).exists():
            return True

        logger.debug(f"Transkun stderr: {result.stderr}")
        return False

    except subprocess.TimeoutExpired:
        logger.error(f"Transkun timeout for: {audio_path}")
        return False
    except Exception as e:
        logger.error(f"Transkun error: {e}")
        return False


def transcribe_audio_python(
    audio_path: str,
    output_midi_path: str,
    device: str = "cuda",
) -> bool:
    """
    Transcribe audio using Python API (alternative to CLI).

    This method loads the model once and can be reused for batch processing.

    Args:
        audio_path: Path to input audio file
        output_midi_path: Path to output MIDI file
        device: Device to use

    Returns:
        True if successful
    """
    try:
        import transkun
        from transkun import transcribe

        # Transcribe
        midi_data = transcribe(audio_path, device=device)

        # Save MIDI
        Path(output_midi_path).parent.mkdir(parents=True, exist_ok=True)
        midi_data.write(output_midi_path)
        return True

    except ImportError:
        logger.error("transkun not installed. Run: pip install transkun")
        return False
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return False


def batch_transcribe(
    audio_files: list,
    output_dir: str,
    device: str = "cuda",
    skip_existing: bool = True,
) -> dict:
    """
    Batch transcribe multiple audio files.

    Args:
        audio_files: List of audio file paths
        output_dir: Output directory for MIDI files
        device: Device to use
        skip_existing: Skip files that already have output

    Returns:
        Dictionary with 'success' and 'failed' lists
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {"success": [], "failed": []}

    for audio_file in audio_files:
        audio_path = Path(audio_file)
        midi_path = output_path / f"{audio_path.stem}.mid"

        if skip_existing and midi_path.exists():
            logger.info(f"Skipping (exists): {audio_path.name}")
            results["success"].append(str(audio_file))
            continue

        logger.info(f"Transcribing: {audio_path.name}")

        if transcribe_audio(str(audio_file), str(midi_path), device):
            results["success"].append(str(audio_file))
        else:
            results["failed"].append(str(audio_file))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to MIDI using Transkun"
    )
    parser.add_argument("--input", "-i", required=True, help="Input audio file")
    parser.add_argument("--output", "-o", required=True, help="Output MIDI file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", default="v2", help="Model version")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not check_transkun_installed():
        print("ERROR: transkun not installed. Run: pip install transkun")
        sys.exit(1)

    success = transcribe_audio(args.input, args.output, args.device, args.model)

    if success:
        print(f"Transcription complete: {args.output}")
    else:
        print("Transcription failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
