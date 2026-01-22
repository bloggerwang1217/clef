#!/usr/bin/env python3
"""
Beyer MIDI-to-Score Inference Module

Provides functions to convert MIDI files to MusicXML using the Beyer Transformer
(ISMIR 2024: "End-to-end Piano Performance-MIDI to Score Conversion with Transformers").

Since Beyer requires Python 3.11 and custom dependencies, this module wraps calls
to a conda environment where Beyer is installed.

Setup:
    1. Create conda environment: conda create -n beyer python=3.11
    2. Activate: conda activate beyer
    3. Install dependencies in MIDI2ScoreTransformer repo
    4. Download checkpoint from GitHub Releases

Usage:
    python -m src.baselines.transkun_beyer.beyer_inference \
        --input input.mid --output output.musicxml

References:
    - Paper: https://arxiv.org/abs/2410.00210
    - Code: https://github.com/TimFelixBeyer/MIDI2ScoreTransformer
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default paths (can be overridden via config)
DEFAULT_BEYER_REPO = "/home/bloggerwang/MIDI2ScoreTransformer"
DEFAULT_CONDA_ENV = "beyer"
DEFAULT_CHECKPOINT = "/home/bloggerwang/MIDI2ScoreTransformer/MIDI2ScoreTF.ckpt"
DEFAULT_MUSESCORE = "/home/bloggerwang/clef/tools/mscore"


def get_runner_script() -> str:
    """
    Return the Python code for the Beyer runner script.

    This script runs inside the Beyer conda environment and performs the
    actual MIDI-to-MusicXML conversion.
    """
    return '''#!/usr/bin/env python3
"""
Beyer Runner Script (runs in beyer conda environment)

This script is called by beyer_inference.py from the clef repo.
It loads the Beyer model and converts MIDI to MusicXML.

Supports:
  - Single file mode: --input file.mid --output file.musicxml
  - Batch mode: --input-dir midi/ --output-dir xml/
  - Watch mode: --input-dir midi/ --output-dir xml/ --watch
"""
import argparse
import json
import sys
import os
import time
from pathlib import Path

# Add MIDI2ScoreTransformer to path
BEYER_REPO = os.environ.get("BEYER_REPO", "/home/bloggerwang/MIDI2ScoreTransformer")
sys.path.insert(0, os.path.join(BEYER_REPO, "midi2scoretransformer"))

# Set MuseScore path before importing
MUSESCORE_PATH = os.environ.get("MUSESCORE_PATH", "/home/bloggerwang/clef/tools/mscore")
import constants
constants.MUSESCORE_PATH = MUSESCORE_PATH


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load the Beyer model from checkpoint."""
    import torch
    from models.roformer import Roformer

    print(f"Loading Beyer model from {checkpoint_path}...", flush=True)

    # PyTorch 2.6+ defaults to weights_only=True, but Beyer checkpoint
    # contains custom config classes. Force weights_only=False.
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    try:
        model = Roformer.load_from_checkpoint(checkpoint_path, map_location=device)
    finally:
        torch.load = original_load
    model.eval()
    model.to(device)
    print("Model loaded.", flush=True)
    return model


def convert_midi_to_musicxml(midi_path: str, output_path: str, model, verbose: bool = False) -> dict:
    """Convert MIDI to MusicXML using Beyer model."""
    from utils import quantize_path

    try:
        # Run inference
        mxl = quantize_path(midi_path, model, verbose=verbose)

        # Save as MusicXML
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        mxl.write("musicxml", output_path)

        return {"status": "success", "output": output_path}

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


def get_pending_files(input_dir: Path, output_dir: Path, skip_existing: bool = True) -> list:
    """Get list of MIDI files that need processing."""
    midi_files = sorted(input_dir.glob("*.mid"))
    if not skip_existing:
        return midi_files

    pending = []
    for midi_file in midi_files:
        xml_file = output_dir / f"{midi_file.stem}.musicxml"
        if not xml_file.exists():
            pending.append(midi_file)
    return pending


def process_batch(midi_files: list, output_dir: Path, model, verbose: bool = False) -> dict:
    """Process a batch of MIDI files."""
    results = {"success": [], "failed": []}

    for i, midi_file in enumerate(midi_files, 1):
        xml_file = output_dir / f"{midi_file.stem}.musicxml"
        print(f"[{i}/{len(midi_files)}] {midi_file.name}", flush=True)

        result = convert_midi_to_musicxml(str(midi_file), str(xml_file), model, verbose)

        if result["status"] == "success":
            size = xml_file.stat().st_size / 1024
            print(f"  -> OK ({size:.0f}K)", flush=True)
            results["success"].append(str(midi_file))
        else:
            print(f"  -> FAILED: {result.get('error', 'unknown')}", flush=True)
            results["failed"].append(str(midi_file))

    return results


def main():
    parser = argparse.ArgumentParser(description="Beyer MIDI to MusicXML converter")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    # Single file mode
    parser.add_argument("--input", help="Input MIDI file (single file mode)")
    parser.add_argument("--output", help="Output MusicXML file (single file mode)")

    # Batch mode
    parser.add_argument("--input-dir", help="Input directory with MIDI files (batch mode)")
    parser.add_argument("--output-dir", help="Output directory for MusicXML files (batch mode)")
    parser.add_argument("--watch", action="store_true", help="Watch mode: keep running and check for new files")
    parser.add_argument("--watch-interval", type=int, default=30, help="Watch interval in seconds")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip existing output files")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Load model once
    model = load_model(args.checkpoint, args.device)

    # Single file mode
    if args.input and args.output:
        result = convert_midi_to_musicxml(args.input, args.output, model, args.verbose)
        print(json.dumps(result))
        return

    # Batch mode
    if args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.watch:
            print(f"Watch mode: checking {input_dir} every {args.watch_interval}s", flush=True)
            print("Press Ctrl+C to stop.", flush=True)

            total_processed = 0
            total_failed = 0

            try:
                while True:
                    pending = get_pending_files(input_dir, output_dir, args.skip_existing)

                    if pending:
                        print(f"\\nFound {len(pending)} new file(s) to process...", flush=True)
                        results = process_batch(pending, output_dir, model, args.verbose)
                        total_processed += len(results["success"])
                        total_failed += len(results["failed"])
                        print(f"Batch done. Total: {total_processed} success, {total_failed} failed", flush=True)

                    time.sleep(args.watch_interval)

            except KeyboardInterrupt:
                print(f"\\nStopped. Total: {total_processed} success, {total_failed} failed", flush=True)

        else:
            # One-shot batch
            pending = get_pending_files(input_dir, output_dir, args.skip_existing)
            if not pending:
                print("No files to process.")
                return

            print(f"Processing {len(pending)} file(s)...", flush=True)
            results = process_batch(pending, output_dir, model, args.verbose)
            print(f"\\nDone: {len(results['success'])} success, {len(results['failed'])} failed", flush=True)

        return

    parser.error("Either --input/--output or --input-dir/--output-dir required")


if __name__ == "__main__":
    main()
'''


def ensure_runner_script(beyer_repo: str) -> str:
    """
    Ensure the runner script exists in the Beyer repo.

    Returns:
        Path to the runner script
    """
    runner_path = os.path.join(beyer_repo, "beyer_runner.py")

    # Always write to ensure it's up to date
    with open(runner_path, "w") as f:
        f.write(get_runner_script())

    return runner_path


def convert_midi_to_musicxml(
    midi_path: str,
    output_path: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    conda_env: str = DEFAULT_CONDA_ENV,
    beyer_repo: str = DEFAULT_BEYER_REPO,
    musescore_path: str = DEFAULT_MUSESCORE,
    device: str = "cuda",
    timeout: int = 300,
    verbose: bool = False,
) -> bool:
    """
    Convert MIDI to MusicXML using Beyer Transformer.

    This function calls the Beyer model via a subprocess in the conda environment.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output MusicXML file
        checkpoint_path: Path to Beyer checkpoint (.ckpt)
        conda_env: Name of conda environment with Beyer dependencies
        beyer_repo: Path to MIDI2ScoreTransformer repository
        musescore_path: Path to MuseScore binary
        device: Device to use ('cuda' or 'cpu')
        timeout: Timeout in seconds
        verbose: Print verbose output

    Returns:
        True if conversion succeeded, False otherwise
    """
    # Validate inputs
    if not Path(midi_path).exists():
        logger.error(f"MIDI file not found: {midi_path}")
        return False

    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Ensure runner script exists
    runner_script = ensure_runner_script(beyer_repo)

    # Build command to run in conda environment
    # Using conda run to execute in the specified environment
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", runner_script,
        "--checkpoint", checkpoint_path,
        "--device", device,
        "--input", midi_path,
        "--output", output_path,
    ]

    if verbose:
        cmd.append("--verbose")

    # Set environment variables
    env = os.environ.copy()
    env["BEYER_REPO"] = beyer_repo
    env["MUSESCORE_PATH"] = musescore_path

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=beyer_repo,
        )

        # Parse output
        if result.returncode == 0:
            # Try to parse JSON result from stdout
            try:
                # Find JSON in output (may have other prints)
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('{'):
                        output = json.loads(line)
                        if output.get("status") == "success":
                            return Path(output_path).exists()
                        else:
                            logger.error(f"Beyer error: {output.get('error')}")
                            return False

                # No JSON found but file exists
                return Path(output_path).exists()

            except json.JSONDecodeError:
                # Check if output file exists anyway
                return Path(output_path).exists()

        logger.debug(f"Beyer stdout: {result.stdout}")
        logger.debug(f"Beyer stderr: {result.stderr}")
        return False

    except subprocess.TimeoutExpired:
        logger.error(f"Beyer timeout for: {midi_path}")
        return False
    except FileNotFoundError:
        logger.error("conda not found. Make sure conda is in PATH.")
        return False
    except Exception as e:
        logger.error(f"Beyer error: {e}")
        return False


def convert_midi_to_musicxml_direct(
    midi_path: str,
    output_path: str,
    model,
    verbose: bool = False,
) -> bool:
    """
    Convert MIDI to MusicXML directly (when running in Beyer environment).

    This is for use when the script is run directly in the conda environment,
    allowing model reuse for batch processing.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path to output MusicXML file
        model: Loaded Beyer model
        verbose: Print verbose output

    Returns:
        True if conversion succeeded
    """
    try:
        # Import Beyer utilities (only works in Beyer environment)
        import sys
        sys.path.insert(0, os.path.join(DEFAULT_BEYER_REPO, "midi2scoretransformer"))
        from utils import quantize_path

        # Run inference
        mxl = quantize_path(midi_path, model, verbose=verbose)

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mxl.write("musicxml", output_path)
        return True

    except Exception as e:
        logger.error(f"Direct conversion failed: {e}")
        return False


def batch_convert(
    midi_files: list,
    output_dir: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    conda_env: str = DEFAULT_CONDA_ENV,
    beyer_repo: str = DEFAULT_BEYER_REPO,
    musescore_path: str = DEFAULT_MUSESCORE,
    device: str = "cuda",
    skip_existing: bool = True,
) -> dict:
    """
    Batch convert multiple MIDI files to MusicXML.

    Args:
        midi_files: List of MIDI file paths
        output_dir: Output directory for MusicXML files
        checkpoint_path: Path to Beyer checkpoint
        conda_env: Conda environment name
        beyer_repo: Path to Beyer repo
        musescore_path: Path to MuseScore
        device: Device to use
        skip_existing: Skip files with existing output

    Returns:
        Dictionary with 'success' and 'failed' lists
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {"success": [], "failed": []}

    for midi_file in midi_files:
        midi_path = Path(midi_file)
        xml_path = output_path / f"{midi_path.stem}.musicxml"

        if skip_existing and xml_path.exists():
            logger.info(f"Skipping (exists): {midi_path.name}")
            results["success"].append(str(midi_file))
            continue

        logger.info(f"Converting: {midi_path.name}")

        if convert_midi_to_musicxml(
            str(midi_file),
            str(xml_path),
            checkpoint_path=checkpoint_path,
            conda_env=conda_env,
            beyer_repo=beyer_repo,
            musescore_path=musescore_path,
            device=device,
        ):
            results["success"].append(str(midi_file))
        else:
            results["failed"].append(str(midi_file))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert MIDI to MusicXML using Beyer Transformer"
    )
    parser.add_argument("--input", "-i", required=True, help="Input MIDI file")
    parser.add_argument("--output", "-o", required=True, help="Output MusicXML file")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--beyer-repo", default=DEFAULT_BEYER_REPO)
    parser.add_argument("--musescore", default=DEFAULT_MUSESCORE)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    success = convert_midi_to_musicxml(
        args.input,
        args.output,
        checkpoint_path=args.checkpoint,
        conda_env=args.conda_env,
        beyer_repo=args.beyer_repo,
        musescore_path=args.musescore,
        device=args.device,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    if success:
        print(f"Conversion complete: {args.output}")
    else:
        print("Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
