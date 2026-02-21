#!/bin/bash
# =============================================================================
# clef-piano-tiny Single-GPU Training Script
# =============================================================================
#
# Usage:
#   ./src/clef/piano/train_tiny_single_gpu.sh              # Default: GPU 0, auto batch size
#   ./src/clef/piano/train_tiny_single_gpu.sh --gpu 1      # Use GPU 1
#   ./src/clef/piano/train_tiny_single_gpu.sh --bs 8       # Manual batch size
#   ./src/clef/piano/train_tiny_single_gpu.sh --resume checkpoints/clef_piano_tiny/epoch_010.pt
#
# GPU memory reference (bf16, seq=1024, mel=3000 frames):
#   A6000 (48GB):  TBD (will auto-detect optimal batch size)
#   RTX 3090 (24GB): TBD

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

CONFIG="${CONFIG:-configs/clef_piano_tiny.yaml}"

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
# Note: Tiny model uses smaller sequences, so we can use larger batch sizes
BATCH_SIZE="${BATCH_SIZE:-}"  # Empty = auto-detect
GRADIENT_ACCUM="${GRADIENT_ACCUM:-2}"
GRADIENT_CLIP="${GRADIENT_CLIP:-$(read_config 'training.gradient_clip' '1.0')}"
MAX_EPOCHS="${MAX_EPOCHS:-$(read_config 'training.max_epochs' '100')}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1000}"
EARLY_STOPPING="${EARLY_STOPPING:-10}"
MANIFEST_DIR="${MANIFEST_DIR:-data/experiments/clef_piano_base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/clef_piano_tiny}"
RESUME="${RESUME:-}"

# Wandb (always enabled for tiny, custom project)
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-clef-piano-tiny}"
WANDB_ENTITY="${WANDB_ENTITY:-bloggerwang1217-national-taiwan-university}"

# Parse command-line flags
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
        --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
        *)           EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Auto-detect batch size if not specified
if [ -z "$BATCH_SIZE" ]; then
    echo "Auto-detecting optimal batch size for GPU $GPU..."

    # Run batch size finder
    BATCH_SIZE=$(CUDA_VISIBLE_DEVICES=$GPU poetry run python -c "
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.clef.piano import ClefPianoConfig, ClefPianoTiny

config = ClefPianoConfig.from_yaml('$CONFIG')
model = ClefPianoTiny(config).cuda()

# Try batch sizes from large to small
for bs in [16, 12, 8, 6, 4, 2, 1]:
    try:
        torch.cuda.empty_cache()
        mel = torch.randn(bs, 1, 128, 3000, device='cuda')
        tgt = torch.randint(0, config.vocab_size, (bs, 512), device='cuda')

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(mel, tgt)
            loss = out['logits'].sum()
            loss.backward()

        torch.cuda.empty_cache()
        print(bs)
        break
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            continue
        else:
            raise
else:
    print(1)  # Fallback to batch size 1
" 2>/dev/null || echo "4")  # Fallback to 4 if script fails

    echo "Auto-detected batch size: $BATCH_SIZE"
fi

EFFECTIVE_BS=$((BATCH_SIZE * GRADIENT_ACCUM))

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# =============================================================================
# Print configuration
# =============================================================================

echo "============================================="
echo "clef-piano-tiny Single-GPU Training"
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
echo "Wandb project:    $WANDB_PROJECT"
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
CMD="$CMD --num-workers 0"

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
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU $CMD"
echo ""

cd "$(dirname "$0")/../../.."
# Disable Python bytecode cache to avoid stale .pyc issues
PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU $CMD
