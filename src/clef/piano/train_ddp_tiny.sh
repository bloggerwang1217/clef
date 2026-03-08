#!/bin/bash
# =============================================================================
# clef-piano-tiny DDP Training Script
# =============================================================================
#
# All hyperparameters (batch size, epochs, LR, paths, etc.) come from the
# config YAML — this script only handles GPU selection and optional overrides.
#
# Usage (from project root):
#   # Default: GPU 1 and 2, config from configs/clef_piano_tiny.yaml
#   bash src/clef/piano/train_ddp_tiny.sh
#
#   # Custom GPUs
#   GPUS=0,1 bash src/clef/piano/train_ddp_tiny.sh
#
#   # Resume from checkpoint
#   RESUME=checkpoints/clef_piano_tiny/epoch_010.pt bash src/clef/piano/train_ddp_tiny.sh
#
#   # Different config
#   CONFIG=configs/clef_piano_base.yaml bash src/clef/piano/train_ddp_tiny.sh

set -e

# =============================================================================
# Environment / GPU selection (the only things that belong in shell)
# =============================================================================

CONFIG="${1:-${CONFIG:-configs/clef_piano_tiny.yaml}}"
GPUS="${GPUS:-1,2}"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-29501}"
RESUME="${RESUME:-}"
WANDB="${WANDB:-true}"
RUN_NAME="${RUN_NAME:-}"
WANDB_PROJECT="${WANDB_PROJECT:-clef-piano-tiny}"

# =============================================================================
# Print summary
# =============================================================================

echo "=============================================="
echo "clef-piano DDP Training"
echo "=============================================="
echo "Config:      $CONFIG  (single source of truth)"
echo "GPUs:        $GPUS  ($NUM_GPUS GPUs)"
echo "Master port: $MASTER_PORT"
if [ -n "$RESUME" ]; then
    echo "Resume:      $RESUME"
fi
echo "Wandb:       $WANDB"
echo "=============================================="
echo ""

# =============================================================================
# Check GPU memory
# =============================================================================

echo "Checking GPU memory..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv --id=$GPUS
echo ""

# =============================================================================
# Build and run command
# =============================================================================

CMD="poetry run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m src.clef.piano.train"
CMD="$CMD --config $CONFIG"

if [ "$WANDB" = "true" ]; then
    CMD="$CMD --wandb --wandb-project $WANDB_PROJECT"
fi

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --wandb-run-name $RUN_NAME"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

echo "Running:"
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS $CMD"
echo ""
echo "Press Ctrl+C to stop"
echo "=============================================="
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS $CMD
