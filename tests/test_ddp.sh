#!/bin/bash
# =============================================================================
# Quick DDP Test - Verify multi-GPU + bucket sampler works
# =============================================================================
#
# Usage:
#   ./tests/test_ddp.sh
#

set -e

GPUS="${GPUS:-1,4}"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "=============================================="
echo "DDP Test: GPU $GPUS"
echo "=============================================="

cd "$(dirname "$0")/.."

# Quick test: 2 epochs, small batch, no wandb
# CUDA_DEVICE_ORDER ensures consistent GPU numbering by PCI bus ID
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS poetry run torchrun --nproc_per_node=$NUM_GPUS -m src.clef.piano.train \
    --config configs/clef_piano_base.yaml \
    --manifest-dir data/experiments/clef_piano_base \
    --checkpoint-dir /tmp/clef_test_ddp \
    --batch-size 2 \
    --gradient-accumulation-steps 1 \
    --max-epochs 1 \
    --validate-every-n-steps 10 \
    --num-workers 2 \
    --debug

echo ""
echo "=============================================="
echo "DDP Test PASSED!"
echo "=============================================="
