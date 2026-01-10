#!/usr/bin/env python3
"""
Batch MV2H Evaluation using Zeng's evaluate_midi_mv2h.sh

This script:
1. Finds all MusicXML predictions
2. Converts them to MIDI using music21
3. Matches with ground truth MIDI from ASAP dataset
4. Calls Zeng's evaluate_midi_mv2h.sh for each pair
5. Aggregates results

Usage:
    python batch_evaluate_mv2h.py \
        --pred_dir data/asap_musicxml_output \
        --gt_dir data/asap_test_set \
        --output_dir results/mt3_mv2h \
        --mv2h_bin src/evaluation/MV2H/bin/mv2h \
        --eval_script src/evaluation/evaluate_midi_mv2h.sh
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

try:
    import music21
except ImportError:
    print("Error: music21 not installed. Run: pip install music21")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration with sensible defaults."""
    PRED_DIR = 'data/asap_musicxml_output'
    GT_DIR = 'data/asap_test_set'
    OUTPUT_DIR = 'results/mt3_mv2h'
    MV2H_BIN = 'src/evaluation/MV2H/bin/mv2h'
    EVAL_SCRIPT = 'src/evaluation/evaluate_midi_mv2h.sh'
    TIMEOUT = 10  # seconds


# =============================================================================
# Core Functions
# =============================================================================

def convert_xml_to_midi(xml_path: str, midi_path: str) -> bool:
    """
    Convert MusicXML to MIDI using music21.

    Args:
        xml_path: Path to input MusicXML file
        midi_path: Path to output MIDI file

    Returns:
        True if successful, False otherwise
    """
    try:
        score = music21.converter.parse(xml_path)
        score.write('midi', fp=midi_path)
        return True
    except Exception as e:
        print(f"  ✗ XML→MIDI failed: {e}")
        return False


def find_ground_truth_midi(pred_xml_path: str, gt_base_dir: str) -> Optional[str]:
    """
    Find corresponding ground truth MIDI for a prediction file.

    ASAP Structure:
        Prediction: asap_musicxml_output/Composer/Work/Piece/PerformanceID.musicxml
        Ground Truth: asap_test_set/Composer/Work/Piece/midi_score.mid

    Args:
        pred_xml_path: Path to prediction MusicXML
        gt_base_dir: Base directory of ASAP ground truth

    Returns:
        Path to ground truth MIDI or None if not found
    """
    parts = Path(pred_xml_path).parts

    # Extract path components after 'asap_musicxml_output'
    try:
        # Find the index where GT structure starts (Composer/Work/Piece)
        start_idx = None
        for i, part in enumerate(parts):
            if 'asap_musicxml_output' in part or 'musicxml_output' in part:
                start_idx = i + 1
                break

        if start_idx is None:
            # Fallback: assume last 4 parts are Composer/Work/Piece/Performance
            start_idx = len(parts) - 4

        # Extract Composer, Work, Piece (skip Performance filename)
        path_components = parts[start_idx:-1]

        if len(path_components) < 2:
            return None

        # Try full path: Composer/Work/Piece/midi_score.mid
        if len(path_components) >= 3:
            gt_path = Path(gt_base_dir) / path_components[0] / path_components[1] / path_components[2] / 'midi_score.mid'
            if gt_path.exists():
                return str(gt_path)

        # Try without Piece level: Composer/Work/midi_score.mid
        gt_path = Path(gt_base_dir) / path_components[0] / path_components[1] / 'midi_score.mid'
        if gt_path.exists():
            return str(gt_path)

    except Exception as e:
        print(f"  ✗ Path parsing error: {e}")

    return None


def run_mv2h_evaluation(
    ref_midi: str,
    pred_midi: str,
    mv2h_bin: str,
    eval_script: str,
    timeout: int = Config.TIMEOUT
) -> Optional[Dict[str, float]]:
    """
    Run Zeng's evaluate_midi_mv2h.sh script.

    Args:
        ref_midi: Path to reference (ground truth) MIDI
        pred_midi: Path to prediction MIDI
        mv2h_bin: Path to MV2H bin directory
        eval_script: Path to evaluate_midi_mv2h.sh
        timeout: Timeout in seconds

    Returns:
        Dictionary of MV2H metrics or None if failed
    """
    try:
        result = subprocess.run(
            ['sh', eval_script, ref_midi, pred_midi, mv2h_bin],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            print(f"  ✗ Script failed: {result.stderr}")
            return None

        # Parse output (last 6 lines contain metrics)
        lines = result.stdout.strip().splitlines()
        if len(lines) < 6:
            print(f"  ✗ Unexpected output format")
            return None

        metrics = {}
        for line in lines[-6:]:
            if ': ' in line:
                key, value = line.split(': ', 1)
                metrics[key] = float(value)

        return metrics

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"  ✗ Execution error: {e}")
        return None


# =============================================================================
# Batch Processing
# =============================================================================

def process_single_file(
    xml_file: Path,
    pred_base_dir: Path,
    gt_base_dir: str,
    output_dir: Path,
    mv2h_bin: str,
    eval_script: str
) -> Optional[Dict]:
    """
    Process a single prediction file.

    Returns:
        Result dictionary or None if failed
    """
    # Generate unique ID from relative path
    rel_path = xml_file.relative_to(pred_base_dir)
    file_id = str(rel_path).replace('/', '_').replace('.musicxml', '').replace('.xml', '')

    # Convert XML to MIDI
    pred_midi_path = output_dir / 'midi_pred' / f'{file_id}.mid'
    if not convert_xml_to_midi(str(xml_file), str(pred_midi_path)):
        return {'id': file_id, 'status': 'xml_conversion_failed'}

    # Find ground truth
    gt_midi_path = find_ground_truth_midi(str(xml_file), gt_base_dir)
    if gt_midi_path is None:
        return {'id': file_id, 'status': 'no_ground_truth'}

    # Run MV2H evaluation
    metrics = run_mv2h_evaluation(gt_midi_path, str(pred_midi_path), mv2h_bin, eval_script)

    if metrics is None:
        return {'id': file_id, 'status': 'mv2h_failed'}

    if metrics.get('MV2H', 0) == 0:
        return {'id': file_id, 'status': 'zero_score'}

    # Success
    return {
        'id': file_id,
        'status': 'success',
        'pred_xml': str(xml_file),
        'pred_midi': str(pred_midi_path),
        'gt_midi': gt_midi_path,
        'metrics': metrics
    }


def batch_evaluate(
    pred_dir: str,
    gt_dir: str,
    output_dir: str,
    mv2h_bin: str,
    eval_script: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Batch evaluate all predictions.

    Returns:
        (successes, errors)
    """
    pred_base = Path(pred_dir)
    output_base = Path(output_dir)

    # Create output directories
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / 'midi_pred').mkdir(exist_ok=True)
    (output_base / 'json').mkdir(exist_ok=True)

    # Find all MusicXML files
    xml_files = list(pred_base.rglob('*.musicxml')) + list(pred_base.rglob('*.xml'))
    print(f"Found {len(xml_files)} MusicXML files\n")

    successes = []
    errors = []

    for xml_file in tqdm(xml_files, desc="Evaluating"):
        result = process_single_file(
            xml_file, pred_base, gt_dir, output_base, mv2h_bin, eval_script
        )

        if result['status'] == 'success':
            successes.append(result)
            # Save individual result
            json_path = output_base / 'json' / f"{result['id']}_mv2h.json"
            with open(json_path, 'w') as f:
                json.dump(result['metrics'], f, indent=2)
        else:
            errors.append(result)

    return successes, errors


# =============================================================================
# Results Aggregation
# =============================================================================

def aggregate_results(successes: List[Dict], output_dir: Path) -> Dict:
    """
    Aggregate MV2H results.

    Returns:
        Summary dictionary
    """
    metric_keys = ['Multi-pitch', 'Voice', 'Meter', 'Value', 'Harmony', 'MV2H']

    avg = {k: 0.0 for k in metric_keys}
    n = len(successes)

    for result in successes:
        for key in metric_keys:
            avg[key] += result['metrics'][key]

    for key in metric_keys:
        avg[key] /= n

    # Compute custom MV2H (Zeng's formula)
    avg['MV2H_custom'] = (
        avg['Multi-pitch'] + avg['Voice'] + avg['Value'] + avg['Harmony']
    ) / 4

    summary = {
        'n_samples': n,
        'metrics': avg,
        'all_results': successes
    }

    # Save summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def print_summary(summary: Dict, n_errors: int):
    """Print formatted summary."""
    print("\n" + "="*70)
    print("MV2H Evaluation Results (using Zeng's evaluate_midi_mv2h.sh)")
    print("="*70)
    print(f"Successful evaluations: {summary['n_samples']}")
    print(f"Failed evaluations:     {n_errors}")
    print("-"*70)

    metrics = summary['metrics']
    for key in ['Multi-pitch', 'Voice', 'Meter', 'Value', 'Harmony', 'MV2H']:
        print(f"{key:20s}: {metrics[key]*100:6.2f}%")

    print("-"*70)
    print(f"{'MV2H (custom)':20s}: {metrics['MV2H_custom']*100:6.2f}%")
    print("="*70)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch MV2H evaluation using Zeng\'s script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python batch_evaluate_mv2h.py \\
        --pred_dir data/asap_musicxml_output \\
        --gt_dir data/asap_test_set \\
        --output_dir results/mt3_mv2h
        """
    )

    parser.add_argument(
        '--pred_dir',
        type=str,
        default=Config.PRED_DIR,
        help=f'Directory containing predicted MusicXML files (default: {Config.PRED_DIR})'
    )
    parser.add_argument(
        '--gt_dir',
        type=str,
        default=Config.GT_DIR,
        help=f'ASAP ground truth directory (default: {Config.GT_DIR})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=Config.OUTPUT_DIR,
        help=f'Output directory for results (default: {Config.OUTPUT_DIR})'
    )
    parser.add_argument(
        '--mv2h_bin',
        type=str,
        default=Config.MV2H_BIN,
        help=f'Path to MV2H bin directory (default: {Config.MV2H_BIN})'
    )
    parser.add_argument(
        '--eval_script',
        type=str,
        default=Config.EVAL_SCRIPT,
        help=f'Path to evaluate_midi_mv2h.sh (default: {Config.EVAL_SCRIPT})'
    )

    args = parser.parse_args()

    # Verify paths exist
    if not Path(args.pred_dir).exists():
        print(f"Error: Prediction directory not found: {args.pred_dir}")
        sys.exit(1)

    if not Path(args.gt_dir).exists():
        print(f"Error: Ground truth directory not found: {args.gt_dir}")
        sys.exit(1)

    if not Path(args.mv2h_bin).exists():
        print(f"Error: MV2H bin not found: {args.mv2h_bin}")
        print("Please compile MV2H first:")
        print("  cd src/evaluation/MV2H && make")
        sys.exit(1)

    if not Path(args.eval_script).exists():
        print(f"Error: Evaluation script not found: {args.eval_script}")
        sys.exit(1)

    # Run evaluation
    print(f"Prediction dir:  {args.pred_dir}")
    print(f"Ground truth:    {args.gt_dir}")
    print(f"Output dir:      {args.output_dir}")
    print(f"MV2H bin:        {args.mv2h_bin}")
    print(f"Eval script:     {args.eval_script}\n")

    successes, errors = batch_evaluate(
        args.pred_dir,
        args.gt_dir,
        args.output_dir,
        args.mv2h_bin,
        args.eval_script
    )

    # Save errors
    if errors:
        output_path = Path(args.output_dir)
        error_path = output_path / 'errors.json'
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\nErrors saved to: {error_path}")

    # Aggregate and print results
    if successes:
        summary = aggregate_results(successes, Path(args.output_dir))
        print_summary(summary, len(errors))
        print(f"\nResults saved to: {args.output_dir}/summary.json")
    else:
        print("\nNo successful evaluations!")
        sys.exit(1)


if __name__ == '__main__':
    main()
