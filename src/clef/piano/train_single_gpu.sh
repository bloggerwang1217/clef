#!/bin/bash
# =============================================================================
# clef-piano-base Single-GPU Training Script
# =============================================================================
#
# Usage:
#   ./src/clef/piano/train_single_gpu.sh              # Default: GPU 0, B=2, accum=2
#   ./src/clef/piano/train_single_gpu.sh --gpu 1      # Use GPU 1
#   ./src/clef/piano/train_single_gpu.sh --bs 3       # Batch size 3
#   ./src/clef/piano/train_single_gpu.sh --resume checkpoints/clef_piano_base_mamba/epoch_010.pt
#
# GPU memory reference (bf16, seq=16384, mel=24000 frames):
#   A6000 (48GB):  B=3 -> 34.9 GB | B=2 -> 23.4 GB | B=1 -> 11.8 GB
#   RTX 3090 (24GB): B=2 -> 23.4 GB | B=1 -> 11.8 GB

set -euo pipefail

# =============================================================================
# Configuration (can be overridden via environment variables or flags)
# =============================================================================

CONFIG="${CONFIG:-configs/clef_piano_base.yaml}"

# Helper function to read config values
read_config() {
    poetry run python3 -c "
import yaml
with open('$CONFIG') as f:
    config = yaml.safe_load(f)
keys = '$1'.split('.')
val = config
for k in keys:
    val = val.get(k, None)
    if val is None:
        print('$2')
        exit(0)
if isinstance(val, bool):
    print(str(val).lower())
else:
    print(val)
"
}

# GPU
GPU="${GPU:-0}"

# Training parameters (read from config.yaml if not set)
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-2}"
GRADIENT_CLIP="${GRADIENT_CLIP:-$(read_config 'training.gradient_clip' '1.0')}"
MAX_EPOCHS="${MAX_EPOCHS:-$(read_config 'training.max_epochs' '50')}"
VALIDATE_EVERY="${VALIDATE_EVERY:-500}"
EARLY_STOPPING="${EARLY_STOPPING:-5}"
MANIFEST_DIR="${MANIFEST_DIR:-data/experiments/clef_piano_base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/clef_piano_base_mamba}"
RESUME="${RESUME:-}"

# Wandb (read from config if not set)
WANDB_ENABLED=$(read_config 'training.wandb.enabled' 'true')
WANDB="${WANDB:-${WANDB_ENABLED}}"
WANDB_PROJECT="${WANDB_PROJECT:-$(read_config 'training.wandb.project' 'clef-piano-base')}"
WANDB_ENTITY="${WANDB_ENTITY:-bloggerwang1217-national-taiwan-university}"

# Parse command-line flags (override above defaults)
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)       GPU="$2"; shift 2 ;;
        --bs)        BATCH_SIZE="$2"; shift 2 ;;
        --accum)     GRADIENT_ACCUM="$2"; shift 2 ;;
        --config)    CONFIG="$2"; shift 2 ;;
        --ckpt-dir)  CHECKPOINT_DIR="$2"; shift 2 ;;
        --epochs)    MAX_EPOCHS="$2"; shift 2 ;;
        --resume)    RESUME="$2"; shift 2 ;;
        --no-wandb)  WANDB="false"; shift ;;
        --validate-every) VALIDATE_EVERY="$2"; shift 2 ;;
        *)           EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

EFFECTIVE_BS=$((BATCH_SIZE * GRADIENT_ACCUM))

# =============================================================================
# Print configuration
# =============================================================================

echo "============================================="
echo "clef-piano-base Single-GPU Training"
echo "============================================="
echo "GPU:              $GPU"
echo "Batch size:       $BATCH_SIZE"
echo "Gradient accum:   $GRADIENT_ACCUM"
echo "Effective BS:     $EFFECTIVE_BS"
echo "Gradient clip:    $GRADIENT_CLIP"
echo "Max epochs:       $MAX_EPOCHS"
echo "Validate every:   $VALIDATE_EVERY steps"
echo "Early stopping:   $EARLY_STOPPING epochs"
echo "Config:           $CONFIG"
echo "Manifest dir:     $MANIFEST_DIR"
echo "Checkpoint dir:   $CHECKPOINT_DIR"
if [ -n "$RESUME" ]; then
    echo "Resume from:      $RESUME"
fi
echo "Wandb:            $WANDB"
echo "============================================="
echo ""

# =============================================================================
# Build command
# =============================================================================

CMD="poetry run python -m src.clef.piano.train"
CMD="$CMD --config $CONFIG"
CMD="$CMD --manifest-dir $MANIFEST_DIR"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --gradient-accumulation-steps $GRADIENT_ACCUM"
CMD="$CMD --gradient-clip $GRADIENT_CLIP"
CMD="$CMD --max-epochs $MAX_EPOCHS"
CMD="$CMD --validate-every-n-steps $VALIDATE_EVERY"
CMD="$CMD --early-stopping-patience $EARLY_STOPPING"
CMD="$CMD --num-workers 4"

if [ "$WANDB" = "true" ]; then
    CMD="$CMD --wandb"
    CMD="$CMD --wandb-project $WANDB_PROJECT"
    CMD="$CMD --wandb-entity $WANDB_ENTITY"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# =============================================================================
# Run training
# =============================================================================

echo "Running command:"
echo "CUDA_VISIBLE_DEVICES=$GPU $CMD"
echo ""

cd "$(dirname "$0")/../../.."
CUDA_VISIBLE_DEVICES=$GPU $CMD
