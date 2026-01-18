#!/bin/bash
# =============================================================================
# MT3 + MuseScore Studio 4.6.5 Baseline Pipeline
# =============================================================================
#
# This script runs the complete MT3 baseline evaluation pipeline:
# 1. Setup: Install MuseScore Studio 4.6.5 and MV2H (if needed)
# 2. Inference: Run MT3 audio-to-MIDI transcription
# 3. Conversion: Convert MIDI to MusicXML using MuseScore
# 4. Evaluation: Run MV2H evaluation against ASAP ground truth
#
# Academic Justification:
# This pipeline uses MuseScore Studio 4.6.5 as the "Industry Standard Baseline" for
# MIDI to MusicXML conversion. Unlike naive quantization libraries (music21),
# MuseScore 4 employs sophisticated heuristic-based voice separation and
# tuplet detection, representing the state-of-the-art for rule-based notation.
#
# Usage:
#   ./run_mt3_evaluate_pipeline.sh [options]
#
# Examples:
#   # Full pipeline with 8 workers
#   ./run_mt3_evaluate_pipeline.sh --audio_dir data/audio --gt_dir data/asap -j 8
#
#   # Evaluation only (skip inference)
#   ./run_mt3_evaluate_pipeline.sh --skip_inference --pred_dir data/mt3_midi --gt_dir data/asap
#
# =============================================================================

set -e

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is now at src/baselines/mt3/, go up 3 levels to project root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Config file
CONFIG_FILE="${PROJECT_ROOT}/configs/mt3_evaluate.yaml"

# Load defaults from config file if it exists
if [[ -f "$CONFIG_FILE" ]]; then
    # Parse YAML using grep/sed (simple key: value format)
    _get_yaml_value() {
        grep "^$1:" "$CONFIG_FILE" 2>/dev/null | sed "s/^$1:[[:space:]]*//" | tr -d '"' | tr -d "'"
    }
    PRED_DIR_DEFAULT=$(_get_yaml_value "pred_dir")
    GT_DIR_DEFAULT=$(_get_yaml_value "gt_dir")
    CHUNK_CSV_DEFAULT=$(_get_yaml_value "chunk_csv")
    MV2H_BIN_DEFAULT=$(_get_yaml_value "mv2h_bin")
    MSCORE_BIN_DEFAULT=$(_get_yaml_value "mscore_bin")
    OUTPUT_DIR_DEFAULT=$(_get_yaml_value "output_dir")
fi

# Paths (use config defaults if available, otherwise use hardcoded defaults)
PRED_DIR="${PRED_DIR_DEFAULT:-${PROJECT_ROOT}/data/experiments/mt3/full_midi}"
GT_DIR="${GT_DIR_DEFAULT:-}"
OUTPUT_DIR="${OUTPUT_DIR_DEFAULT:-${PROJECT_ROOT}/results/mt3_baseline}"
MSCORE_BIN="${MSCORE_BIN_DEFAULT:-${PROJECT_ROOT}/tools/mscore}"
MV2H_BIN="${MV2H_BIN_DEFAULT:-${PROJECT_ROOT}/MV2H/bin}"
CHUNK_CSV="${CHUNK_CSV_DEFAULT:-}"

# Processing
WORKERS=$(nproc 2>/dev/null || echo 4)
MV2H_TIMEOUT=120
MV2H_CHUNK_TIMEOUT=10  # Shorter timeout for 5-bar chunk evaluation (Zeng's setting)
MSCORE_TIMEOUT=60
MODE="full"

# Flags
SKIP_SETUP=false
SKIP_VALIDATION=false
SKIP_EVALUATION=false
VERBOSE=false

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

usage() {
    cat << EOF
Usage: $(basename "$0") [options]

MT3 + MuseScore Studio 4.6.5 Baseline Evaluation Pipeline

Required:
  --gt_dir DIR          ASAP dataset base directory

Options:
  --pred_dir DIR        MT3 MIDI outputs directory (default: data/experiments/mt3/full_midi)
  --output_dir DIR      Output directory (default: results/mt3_baseline)
  --mode MODE           Evaluation mode: 'full' or 'chunks' (default: full)
  --chunk_csv FILE      Chunk CSV file (default for chunks: src/evaluation/test_chunk_set.csv)

Processing:
  -j, --workers N       Number of parallel workers (default: $WORKERS)
  --mv2h_timeout N      MV2H full song timeout in seconds (default: $MV2H_TIMEOUT)
  --chunk_timeout N     MV2H chunk timeout in seconds (default: $MV2H_CHUNK_TIMEOUT)
  --mscore_timeout N    MuseScore conversion timeout in seconds (default: $MSCORE_TIMEOUT)

Paths:
  --mscore_bin PATH     Path to MuseScore binary (default: tools/mscore)
  --mv2h_bin PATH       Path to MV2H bin directory (default: MV2H/bin)

Flags:
  --skip_setup          Skip MuseScore/MV2H setup
  --skip_validation     Skip MT3 MIDI file validation
  --skip_evaluation     Skip MV2H evaluation (only run setup and validation)
  -v, --verbose         Enable verbose output
  -h, --help            Show this help message

Examples:
  # Full song evaluation
  $(basename "$0") --gt_dir /path/to/asap-dataset --mode full

  # 5-bar chunk evaluation (compare with Zeng)
  $(basename "$0") --gt_dir /path/to/asap-dataset --mode chunks

  # Custom paths
  $(basename "$0") --gt_dir /path/to/asap-dataset \\
      --pred_dir /path/to/mt3_midi --mode chunks -j 16
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --pred_dir)
            PRED_DIR="$2"
            shift 2
            ;;
        --gt_dir)
            GT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --chunk_csv)
            CHUNK_CSV="$2"
            shift 2
            ;;
        -j|--workers)
            WORKERS="$2"
            shift 2
            ;;
        --mv2h_timeout)
            MV2H_TIMEOUT="$2"
            shift 2
            ;;
        --chunk_timeout)
            MV2H_CHUNK_TIMEOUT="$2"
            shift 2
            ;;
        --mscore_timeout)
            MSCORE_TIMEOUT="$2"
            shift 2
            ;;
        --mscore_bin)
            MSCORE_BIN="$2"
            shift 2
            ;;
        --mv2h_bin)
            MV2H_BIN="$2"
            shift 2
            ;;
        --skip_setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip_validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --skip_evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# VALIDATION
# =============================================================================

echo "=============================================="
echo "MT3 + MuseScore Studio 4.6.5 Baseline Pipeline"
echo "=============================================="
echo ""

# Show config file status
if [[ -f "$CONFIG_FILE" ]]; then
    echo "Config file: $CONFIG_FILE"
else
    echo "Config file: (not found, using defaults)"
fi
echo ""

# Check required arguments
if [[ -z "$GT_DIR" ]]; then
    echo "Error: --gt_dir is required (or set gt_dir in $CONFIG_FILE)"
    usage
    exit 1
fi

if [[ ! -d "$GT_DIR" ]]; then
    echo "Error: Ground truth directory not found: $GT_DIR"
    exit 1
fi

# Check prediction directory
if [[ ! -d "$PRED_DIR" ]]; then
    echo "Error: Prediction directory not found: $PRED_DIR"
    exit 1
fi

# Set default chunk CSV for chunks mode
if [[ "$MODE" == "chunks" && -z "$CHUNK_CSV" ]]; then
    CHUNK_CSV="${PROJECT_ROOT}/src/evaluation/test_chunk_set.csv"
fi

# Check chunk mode requirements
if [[ "$MODE" == "chunks" && ! -f "$CHUNK_CSV" ]]; then
    echo "Error: Chunk CSV not found: $CHUNK_CSV"
    exit 1
fi

# Print configuration
echo "Configuration:"
echo "  Mode:              $MODE"
echo "  Ground truth:      $GT_DIR"
echo "  Predictions:       $PRED_DIR"
echo "  Output:            $OUTPUT_DIR"
echo "  Workers:           $WORKERS"
if [[ "$MODE" == "chunks" ]]; then
    echo "  Chunk timeout:     ${MV2H_CHUNK_TIMEOUT}s"
    echo "  Chunk CSV:         $CHUNK_CSV"
else
    echo "  MV2H timeout:      ${MV2H_TIMEOUT}s"
fi
echo "  MuseScore timeout: ${MSCORE_TIMEOUT}s"
echo ""

# =============================================================================
# STEP 1: SETUP
# =============================================================================

if [[ "$SKIP_SETUP" == false ]]; then
    echo "=============================================="
    echo "Step 1: Setup"
    echo "=============================================="

    # Setup MuseScore
    if [[ ! -f "$MSCORE_BIN" ]]; then
        echo "Setting up MuseScore Studio 4.6.5..."
        bash "${SCRIPT_DIR}/setup_musescore.sh" --install-dir "${PROJECT_ROOT}/tools"
    else
        echo "MuseScore already installed: $MSCORE_BIN"
    fi

    # Setup MV2H
    if [[ ! -d "$MV2H_BIN" ]]; then
        echo ""
        echo "Setting up MV2H..."
        cd "$PROJECT_ROOT"
        if [[ ! -d "MV2H" ]]; then
            git clone https://github.com/apmcleod/MV2H.git
        fi
        cd MV2H
        make
        cd "$PROJECT_ROOT"
    else
        echo "MV2H already installed: $MV2H_BIN"
    fi

    echo ""
fi

# =============================================================================
# STEP 2: VALIDATE MT3 MIDI FILES
# =============================================================================

if [[ "$SKIP_VALIDATION" == false ]]; then
    echo "=============================================="
    echo "Step 2: Validate MT3 MIDI Files"
    echo "=============================================="

echo "Checking MT3 MIDI files against test set..."
echo "  MIDI dir:  $PRED_DIR"
if [[ "$MODE" == "chunks" ]]; then
    echo "  Chunk CSV: $CHUNK_CSV"
fi
echo ""

# Validate that MIDI files exist and align with test set
VALIDATION_RESULT=$(python3 << PYEOF
import sys
import os
from pathlib import Path

pred_dir = "$PRED_DIR"
mode = "$MODE"
chunk_csv = "$CHUNK_CSV"

if not os.path.isdir(pred_dir):
    print(f"ERROR: Prediction directory not found: {pred_dir}")
    sys.exit(1)

# Find all MIDI files
midi_files = list(Path(pred_dir).rglob("*.mid")) + list(Path(pred_dir).rglob("*.midi"))
print(f"Found {len(midi_files)} MIDI files in {pred_dir}")

if len(midi_files) == 0:
    print("ERROR: No MIDI files found!")
    sys.exit(1)

# For chunks mode, validate against chunk CSV
if mode == "chunks" and chunk_csv:
    import csv

    # Extract unique (piece, performance) pairs from chunk CSV
    expected_pairs = set()
    with open(chunk_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunk_id = row.get("chunk_id", "")
            # chunk_id format: Bach#Prelude#bwv_875#Ahfat01M.10
            parts = chunk_id.rsplit("#", 1)
            if len(parts) == 2:
                piece_id = parts[0]  # Bach#Prelude#bwv_875
                perf_chunk = parts[1]  # Ahfat01M.10
                performance = perf_chunk.rsplit(".", 1)[0]  # Ahfat01M
                expected_pairs.add((piece_id, performance))

    print(f"Test set requires {len(expected_pairs)} unique (piece, performance) pairs")

    # Check which pairs have MIDI files
    found_pairs = set()
    missing_pairs = []

    for piece_id, performance in expected_pairs:
        # Convert piece_id to path: Bach#Prelude#bwv_875 -> Bach/Prelude/bwv_875
        path_parts = piece_id.split("#")
        search_dir = Path(pred_dir) / "/".join(path_parts)

        found = False
        if search_dir.exists():
            for f in search_dir.iterdir():
                if f.suffix.lower() in [".mid", ".midi"] and performance in f.stem:
                    found = True
                    found_pairs.add((piece_id, performance))
                    break

        if not found:
            missing_pairs.append(f"{piece_id}#{performance}")

    print(f"Found MIDI files for {len(found_pairs)}/{len(expected_pairs)} pairs")

    if missing_pairs:
        print(f"\nERROR: Missing {len(missing_pairs)} MIDI files:")
        for p in missing_pairs[:10]:  # Show first 10
            print(f"  - {p}")
        if len(missing_pairs) > 10:
            print(f"  ... and {len(missing_pairs) - 10} more")
        sys.exit(1)

    print("All required MIDI files found!")

else:
    # For full mode, just report what we found
    print("Full mode: Will evaluate all MIDI files found")

sys.exit(0)
PYEOF
)

VALIDATION_EXIT_CODE=$?

if [[ $VALIDATION_EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "Validation failed! Please ensure all required MT3 MIDI files are present."
    echo "Expected location: $PRED_DIR/{Composer}/{Work}/{Piece}/{performance}.mid"
    exit 1
fi

echo ""
else
    echo "=============================================="
    echo "Step 2: Validate MT3 MIDI Files (skipped)"
    echo "=============================================="
    echo ""
fi

# =============================================================================
# STEP 3: MUSESCORE CONVERSION (MIDI -> MusicXML)
# =============================================================================

echo "=============================================="
echo "Step 3: MuseScore Conversion (MIDI -> MusicXML)"
echo "=============================================="

MUSICXML_DIR="${OUTPUT_DIR}/musicxml"
mkdir -p "$MUSICXML_DIR"

echo "Converting MIDI files to MusicXML using MuseScore Studio 4.6.5..."
echo "  Input:  $PRED_DIR"
echo "  Output: $MUSICXML_DIR"
echo ""

# Count MIDI files
MIDI_COUNT=$(find "$PRED_DIR" -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l)
echo "Found $MIDI_COUNT MIDI files to convert."
echo ""

# Conversion is handled by mt3_evaluate.py during evaluation
# This step is mainly for visibility
echo "Note: Conversion will be performed during evaluation step."
echo ""

# =============================================================================
# STEP 4: MV2H EVALUATION
# =============================================================================

if [[ "$SKIP_EVALUATION" == false ]]; then
    echo "=============================================="
    echo "Step 4: MV2H Evaluation"
    echo "=============================================="

    RESULTS_DIR="${OUTPUT_DIR}/results"
    mkdir -p "$RESULTS_DIR"

    OUTPUT_CSV="${RESULTS_DIR}/${MODE}_song.csv"

    echo "Running MV2H evaluation with $WORKERS workers..."
    echo "  Mode:    $MODE"
    echo "  Output:  $OUTPUT_CSV"
    echo ""

    # Build evaluation command
    EVAL_CMD=(
        python -m src.baselines.mt3.mt3_evaluate
        --mode "$MODE"
        --pred_dir "$PRED_DIR"
        --gt_dir "$GT_DIR"
        --mv2h_bin "$MV2H_BIN"
        --mscore_bin "$MSCORE_BIN"
        --output "$OUTPUT_CSV"
        --output_dir "$MUSICXML_DIR"
        --workers "$WORKERS"
        --timeout "$MV2H_TIMEOUT"
        --chunk_timeout "$MV2H_CHUNK_TIMEOUT"
        --mscore_timeout "$MSCORE_TIMEOUT"
    )

    if [[ "$MODE" == "chunks" ]]; then
        EVAL_CMD+=(--chunk_csv "$CHUNK_CSV")
    fi

    if [[ "$VERBOSE" == true ]]; then
        EVAL_CMD+=(-v)
    fi

    # Run evaluation
    "${EVAL_CMD[@]}"

    echo ""
    echo "=============================================="
    echo "Pipeline Complete!"
    echo "=============================================="
    echo ""
    echo "Results saved to: $OUTPUT_CSV"
    echo "Summary saved to: ${OUTPUT_CSV%.csv}.summary.json"
    echo "MusicXML files saved to: $MUSICXML_DIR"
    echo ""
else
    echo "Skipping evaluation (--skip_evaluation)"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "Output directory structure:"
echo "  ${OUTPUT_DIR}/"
echo "  ├── musicxml/          # MuseScore converted MusicXML"
if [[ "$MODE" == "chunks" ]]; then
    echo "  ├── musicxml_cache/    # Cached MusicXML conversions"
    echo "  ├── chunk_midi/        # Extracted 5-bar chunk MIDI files"
    echo "  ├── timeouts.txt       # Chunks that exceeded timeout"
    echo "  ├── errors.txt         # Chunks with errors"
fi
echo "  └── results/"
echo "      ├── ${MODE}_song.csv"
echo "      └── ${MODE}_song.summary.json"
echo ""
