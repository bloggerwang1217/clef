#!/bin/bash
# =============================================================================
# CLEF Model Inference Script
# =============================================================================
#
# Run inference on test set using trained CLEF model.
#
# Usage:
#   # Default: Use GPU 0, infer on syn test set
#   ./src/evaluation/run_clef_inference.sh
#
#   # Custom GPU
#   GPU=2 ./src/evaluation/run_clef_inference.sh
#
#   # Custom checkpoint and output
#   CHECKPOINT=path/to/model.pt OUTPUT_DIR=path/to/output ./src/evaluation/run_clef_inference.sh
#

set -e

# =============================================================================
# Configuration
# =============================================================================

# GPU to use (defaults to 0)
GPU="${GPU:-0}"

# Checkpoint path
CHECKPOINT="${CHECKPOINT:-checkpoints/clef_piano_base_best/best.pt}"

# Config (optional, will use checkpoint config if available)
CONFIG="${CONFIG:-configs/clef_piano_base.yaml}"

# Test manifest
MANIFEST="${MANIFEST:-data/experiments/clef_piano_base/test_manifest.json}"
MANIFEST_DIR="${MANIFEST_DIR:-data/experiments/clef_piano_base}"

# Output directories
OUTPUT_DIR="${OUTPUT_DIR:-data/experiments/clef_piano_base/test_kern_pred}"
OUTPUT_MIDI_DIR="${OUTPUT_MIDI_DIR:-data/experiments/clef_piano_base/test_midi_pred}"
LOG_DIR="${LOG_DIR:-logs}"

# Max generation length (per chunk)
MAX_LENGTH="${MAX_LENGTH:-16384}"

# Chunking parameters
CHUNK_FRAMES="${CHUNK_FRAMES:-24000}"    # 4 min @ 100fps
OVERLAP_FRAMES="${OVERLAP_FRAMES:-6000}" # 1 min overlap, 3 min stride

# Create log directory and generate log filename
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/inference_${TIMESTAMP}.log"

# =============================================================================
# Print configuration
# =============================================================================

echo "=============================================="
echo "CLEF Inference"
echo "=============================================="
echo "GPU: $GPU"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Manifest: $MANIFEST"
echo "Kern output: $OUTPUT_DIR"
echo "MIDI output: $OUTPUT_MIDI_DIR"
echo "Max length per chunk: $MAX_LENGTH"
echo "Chunk frames: $CHUNK_FRAMES ($(echo "scale=1; $CHUNK_FRAMES/100/60" | bc) min)"
echo "Overlap frames: $OVERLAP_FRAMES ($(echo "scale=1; $OVERLAP_FRAMES/100/60" | bc) min)"
echo "Log file: $LOG_FILE"
echo "=============================================="
echo ""

# =============================================================================
# Run inference (with logging)
# =============================================================================

{
  CUDA_VISIBLE_DEVICES=$GPU poetry run python -m src.clef.piano.clef_piano_inference \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --output-midi-dir "$OUTPUT_MIDI_DIR" \
    --max-length "$MAX_LENGTH" \
    --chunk-frames "$CHUNK_FRAMES" \
    --overlap-frames "$OVERLAP_FRAMES" \
    --device cuda:0

  echo ""
  echo "=============================================="
  echo "Inference complete!"
  echo "Kern predictions: $OUTPUT_DIR"
  echo "MIDI files: $OUTPUT_MIDI_DIR"
  echo "Log saved to: $LOG_FILE"
  echo "=============================================="
} 2>&1 | tee "$LOG_FILE"
