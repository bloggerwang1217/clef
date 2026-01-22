#!/bin/bash
# =============================================================================
# Stage 2: Beyer - MIDI to MusicXML (Batch Mode)
# Usage: bash 02_run_beyer.sh [GPU_ID] [--watch]
#
# Model is loaded ONCE and processes all files, avoiding repeated loading.
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEF_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Parse arguments
GPU_ID="0"
WATCH_MODE=""
for arg in "$@"; do
    if [[ "$arg" == "--watch" ]]; then
        WATCH_MODE="--watch"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        GPU_ID="$arg"
    fi
done

# Configuration
INPUT_DIR="${INPUT_DIR:-$CLEF_ROOT/data/experiments/transkun_beyer/midi_from_transkun}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLEF_ROOT/data/experiments/transkun_beyer}"
XML_DIR="$OUTPUT_DIR/musicxml_from_beyer"

# Beyer configuration
BEYER_CHECKPOINT="${BEYER_CHECKPOINT:-/home/bloggerwang/MIDI2ScoreTransformer/MIDI2ScoreTF.ckpt}"
BEYER_CONDA_ENV="${BEYER_CONDA_ENV:-beyer}"
BEYER_REPO="${BEYER_REPO:-/home/bloggerwang/MIDI2ScoreTransformer}"
MUSESCORE_PATH="${MUSESCORE_PATH:-$CLEF_ROOT/tools/mscore}"

# Create output directory
mkdir -p "$XML_DIR"

echo "=============================================="
echo "Beyer: MIDI -> MusicXML (Batch Mode)"
echo "=============================================="
echo "Input dir:   $INPUT_DIR"
echo "Output dir:  $XML_DIR"
echo "GPU:         $GPU_ID"
echo "Watch mode:  ${WATCH_MODE:-disabled}"
echo "=============================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Activate conda environment and ensure runner script exists
source ~/miniconda3/etc/profile.d/conda.sh

# First, update the runner script (from base env)
conda activate base
python -c "
import sys
sys.path.insert(0, '$CLEF_ROOT/src/baselines/transkun_beyer')
from beyer_inference import ensure_runner_script
ensure_runner_script('$BEYER_REPO')
print('Runner script updated.')
"

# Set environment variables for the runner
export BEYER_REPO="$BEYER_REPO"
export MUSESCORE_PATH="$MUSESCORE_PATH"

# Run Beyer in batch mode (model loaded once)
conda run -n "$BEYER_CONDA_ENV" --no-capture-output \
    python "$BEYER_REPO/beyer_runner.py" \
    --checkpoint "$BEYER_CHECKPOINT" \
    --device cuda \
    --input-dir "$INPUT_DIR" \
    --output-dir "$XML_DIR" \
    --skip-existing \
    $WATCH_MODE

echo "=============================================="
echo "Beyer Complete"
echo "=============================================="
echo "Output:  $XML_DIR"
echo "=============================================="
