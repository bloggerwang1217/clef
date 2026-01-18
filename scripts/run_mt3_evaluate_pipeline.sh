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
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths
AUDIO_DIR=""
PRED_DIR=""
GT_DIR=""
OUTPUT_DIR="${PROJECT_ROOT}/results/mt3_baseline"
MSCORE_BIN="${PROJECT_ROOT}/tools/mscore"
MV2H_BIN="${PROJECT_ROOT}/MV2H/bin"
CHUNK_CSV=""

# Processing
WORKERS=$(nproc 2>/dev/null || echo 4)
MV2H_TIMEOUT=120
MV2H_CHUNK_TIMEOUT=10  # Shorter timeout for 5-bar chunk evaluation (Zeng's setting)
MSCORE_TIMEOUT=60
MODE="full"

# Flags
SKIP_SETUP=false
SKIP_INFERENCE=false
SKIP_EVALUATION=false
VERBOSE=false

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

usage() {
    cat << EOF
Usage: $(basename "$0") [options]

MT3 + MuseScore Studio 4.6.5 Baseline Evaluation Pipeline

Required (one of):
  --audio_dir DIR       Directory containing audio files for MT3 inference
  --pred_dir DIR        Directory containing MT3 MIDI outputs (skip inference)

Required:
  --gt_dir DIR          ASAP dataset base directory

Options:
  --output_dir DIR      Output directory (default: results/mt3_baseline)
  --mode MODE           Evaluation mode: 'full' or 'chunks' (default: full)
  --chunk_csv FILE      Chunk CSV file (required for 'chunks' mode)

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
  --skip_inference      Skip MT3 inference (use existing MIDI files)
  --skip_evaluation     Skip MV2H evaluation (only run conversion)
  -v, --verbose         Enable verbose output
  -h, --help            Show this help message

Examples:
  # Full pipeline
  $(basename "$0") --audio_dir data/audio --gt_dir data/asap -j 8

  # Evaluation only
  $(basename "$0") --skip_inference --pred_dir data/mt3_midi --gt_dir data/asap

  # Chunk evaluation
  $(basename "$0") --pred_dir data/mt3_midi --gt_dir data/asap \\
      --mode chunks --chunk_csv data/zeng_test_chunks.csv
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio_dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
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
        --skip_inference)
            SKIP_INFERENCE=true
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

# Check required arguments
if [[ -z "$GT_DIR" ]]; then
    echo "Error: --gt_dir is required"
    usage
    exit 1
fi

if [[ ! -d "$GT_DIR" ]]; then
    echo "Error: Ground truth directory not found: $GT_DIR"
    exit 1
fi

# Determine prediction directory
if [[ "$SKIP_INFERENCE" == true ]]; then
    if [[ -z "$PRED_DIR" ]]; then
        echo "Error: --pred_dir is required when --skip_inference is set"
        exit 1
    fi
    if [[ ! -d "$PRED_DIR" ]]; then
        echo "Error: Prediction directory not found: $PRED_DIR"
        exit 1
    fi
else
    if [[ -z "$AUDIO_DIR" ]]; then
        echo "Error: --audio_dir is required (or use --skip_inference with --pred_dir)"
        exit 1
    fi
    if [[ ! -d "$AUDIO_DIR" ]]; then
        echo "Error: Audio directory not found: $AUDIO_DIR"
        exit 1
    fi
    PRED_DIR="${OUTPUT_DIR}/midi"
fi

# Check chunk mode requirements
if [[ "$MODE" == "chunks" && -z "$CHUNK_CSV" ]]; then
    echo "Error: --chunk_csv is required for 'chunks' mode"
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
# STEP 2: MT3 INFERENCE
# =============================================================================

if [[ "$SKIP_INFERENCE" == false ]]; then
    echo "=============================================="
    echo "Step 2: MT3 Inference (Audio -> MIDI)"
    echo "=============================================="

    mkdir -p "$PRED_DIR"

    # Check if MT3 is available
    if ! python -c "import t5x" 2>/dev/null; then
        echo "Warning: t5x not found. MT3 inference requires Google's T5X library."
        echo "Please install MT3 dependencies or use --skip_inference with existing MIDI files."
        echo ""
        echo "For MT3 setup, see: https://github.com/magenta/mt3"
        exit 1
    fi

    # Run MT3 inference
    # Note: This is a placeholder. Actual MT3 inference command depends on your setup.
    echo "Running MT3 inference..."
    echo "  Input:  $AUDIO_DIR"
    echo "  Output: $PRED_DIR"
    echo ""

    # Example MT3 inference command (adjust based on your setup):
    # python -m mt3.infer \
    #     --model_path=gs://mt3/checkpoints/mt3 \
    #     --audio_dir="$AUDIO_DIR" \
    #     --output_dir="$PRED_DIR"

    echo "Note: MT3 inference command should be customized for your setup."
    echo "Skipping actual inference - assuming MIDI files exist in $PRED_DIR"
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
if [[ "$SKIP_INFERENCE" == false ]]; then
    echo "  ├── midi/              # MT3 MIDI outputs"
fi
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
