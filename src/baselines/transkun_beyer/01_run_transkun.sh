#!/bin/bash
# =============================================================================
# Stage 1: Transkun - Audio to MIDI
# Usage: bash 01_run_transkun.sh [GPU_ID]
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEF_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Configuration
ASAP_DIR="${ASAP_DIR:-$CLEF_ROOT/data/datasets/asap_test_set}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLEF_ROOT/data/experiments/transkun_beyer}"
GPU_ID="${1:-0}"  # First argument or default to 0
SKIP_EXISTING="${SKIP_EXISTING:-true}"

# Create output directory
MIDI_DIR="$OUTPUT_DIR/midi_from_transkun"
mkdir -p "$MIDI_DIR"

echo "=============================================="
echo "Transkun: Audio -> MIDI"
echo "=============================================="
echo "ASAP dir:    $ASAP_DIR"
echo "Output dir:  $MIDI_DIR"
echo "GPU:         $GPU_ID"
echo "Skip exist:  $SKIP_EXISTING"
echo "=============================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Find all audio files
AUDIO_FILES=$(find "$ASAP_DIR" -name "*.wav" -type f | sort)
TOTAL=$(echo "$AUDIO_FILES" | wc -l)
COUNT=0
SKIPPED=0
FAILED=0

LOG_FILE="$OUTPUT_DIR/transkun.log"
echo "Started at $(date)" > "$LOG_FILE"

for AUDIO in $AUDIO_FILES; do
    COUNT=$((COUNT + 1))

    # Extract relative path for output naming
    REL_PATH="${AUDIO#$ASAP_DIR/}"
    # Convert path to flat filename: Glinka/The_Lark/Denisova10M.wav -> Glinka_The_Lark_Denisova10M
    FLAT_NAME=$(echo "$REL_PATH" | sed 's|/|_|g' | sed 's|\.wav$||')
    OUTPUT_MIDI="$MIDI_DIR/${FLAT_NAME}.mid"

    # Skip if exists
    if [ "$SKIP_EXISTING" = "true" ] && [ -f "$OUTPUT_MIDI" ]; then
        echo "[$COUNT/$TOTAL] SKIP: $FLAT_NAME"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$COUNT/$TOTAL] Processing: $FLAT_NAME"

    # Run transkun
    if transkun "$AUDIO" "$OUTPUT_MIDI" --device cuda 2>> "$LOG_FILE"; then
        echo "  -> OK: $(ls -lh "$OUTPUT_MIDI" | awk '{print $5}')"
    else
        echo "  -> FAILED"
        FAILED=$((FAILED + 1))
        echo "FAILED: $AUDIO" >> "$LOG_FILE"
    fi
done

echo "=============================================="
echo "Transkun Complete"
echo "=============================================="
echo "Total:   $TOTAL"
echo "Skipped: $SKIPPED"
echo "Failed:  $FAILED"
echo "Success: $((TOTAL - SKIPPED - FAILED))"
echo "Output:  $MIDI_DIR"
echo "Log:     $LOG_FILE"
echo "=============================================="
