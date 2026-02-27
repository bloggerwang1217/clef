#!/bin/bash
# =============================================================================
# clef-piano-tiny Single-GPU Training Script
# =============================================================================
#
# All hyperparameters come from the config YAML — this script only handles
# GPU selection and optional resume/wandb overrides.
#
# Usage (from project root):
#   # Default: GPU 0
#   bash src/clef/piano/train_tiny_single_gpu.sh
#
#   # Custom GPU
#   GPU=1 bash src/clef/piano/train_tiny_single_gpu.sh
#
#   # Resume from checkpoint
#   RESUME=checkpoints/clef_piano_tiny/epoch_010.pt bash src/clef/piano/train_tiny_single_gpu.sh
#
#   # Enable wandb with custom project
#   WANDB=true WANDB_PROJECT=my-project bash src/clef/piano/train_tiny_single_gpu.sh

set -euo pipefail

# =============================================================================
# Environment / GPU selection (the only things that belong in shell)
# =============================================================================

CONFIG="${CONFIG:-configs/clef_piano_tiny.yaml}"
GPU="${GPU:-0}"
RESUME="${RESUME:-}"
WANDB="${WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-clef-piano-base}"

# =============================================================================
# Print summary
# =============================================================================

echo "============================================="
echo "clef-piano-tiny Single-GPU Training"
echo "============================================="
echo "Config:        $CONFIG  (single source of truth)"
echo "GPU:           $GPU"
if [ -n "$RESUME" ]; then
    echo "Resume:        $RESUME"
fi
echo "Wandb:         $WANDB"
echo "Wandb project: $WANDB_PROJECT"
echo "============================================="
echo ""

# =============================================================================
# Build and run command
# =============================================================================

CMD="poetry run python -m src.clef.piano.train"
CMD="$CMD --config $CONFIG"
CMD="$CMD --num-workers 0"

if [ "$WANDB" = "true" ]; then
    CMD="$CMD --wandb"
    CMD="$CMD --wandb-project $WANDB_PROJECT"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

echo "Running:"
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU $CMD"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================="
echo ""

cd "$(dirname "$0")/../../.."
PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU $CMD
