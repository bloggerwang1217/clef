#!/bin/bash
# =============================================================================
# clef-piano-base DDP Training Script
# =============================================================================
#
# Usage:
#   # Default: GPU 1 and 4, batch size 6 per GPU
#   ./src/clef/piano/train_ddp.sh
#
#   # Custom GPUs
#   GPUS=0,1 ./src/clef/piano/train_ddp.sh
#
#   # Custom batch size
#   BATCH_SIZE=4 ./src/clef/piano/train_ddp.sh
#
#   # Resume from checkpoint
#   RESUME=checkpoints/clef_piano_base/epoch_010.pt ./src/clef/piano/train_ddp.sh
#

set -e

# =============================================================================
# Configuration (can be overridden via environment variables)
# =============================================================================

# Paths (first positional arg or env var)
CONFIG="${1:-${CONFIG:-configs/clef_piano_base.yaml}}"

# Helper function to read config values
read_config() {
    poetry run python3 -c "
import yaml
with open('$CONFIG') as f:
    config = yaml.safe_load(f)
# Navigate nested dict
keys = '$1'.split('.')
val = config
for k in keys:
    val = val.get(k, None)
    if val is None:
        print('$2')
        exit(0)
# Convert Python True/False to shell true/false
if isinstance(val, bool):
    print(str(val).lower())
else:
    print(val)
"
}

# GPU configuration
GPUS="${GPUS:-1,2}"                        # Default: GPU 1 and 2
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# Training parameters (read from config.yaml if not set via env vars)
BATCH_SIZE="${BATCH_SIZE:-$(read_config 'training.batch_size' '2')}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-$(read_config 'training.gradient_accumulation_steps' '4')}"
GRADIENT_CLIP="${GRADIENT_CLIP:-$(read_config 'training.gradient_clip' '1.0')}"
MAX_EPOCHS="${MAX_EPOCHS:-$(read_config 'training.max_epochs' '100')}"
VALIDATE_EVERY="${VALIDATE_EVERY:-500}"    # Validate every N steps
EARLY_STOPPING="${EARLY_STOPPING:-5}"      # Early stop after N epochs without improvement (0=disabled)
MASTER_PORT="${MASTER_PORT:-29500}"        # Change if running multiple DDP jobs
MANIFEST_DIR="${MANIFEST_DIR:-data/experiments/clef_piano_base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(read_config 'paths.checkpoint_dir' 'checkpoints/clef_piano_base')}"
RESUME="${RESUME:-}"

# Wandb (read from config if not set)
WANDB_ENABLED=$(read_config 'training.wandb.enabled' 'true')
WANDB="${WANDB:-${WANDB_ENABLED}}"
WANDB_PROJECT="${WANDB_PROJECT:-$(read_config 'training.wandb.project' 'clef-piano-base')}"
WANDB_ENTITY="${WANDB_ENTITY:-bloggerwang1217-national-taiwan-university}"

# =============================================================================
# Print configuration
# =============================================================================

echo "=============================================="
echo "clef-piano-base DDP Training"
echo "=============================================="
echo "GPUs: $GPUS ($NUM_GPUS GPUs)"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUM))"
echo "Max epochs: $MAX_EPOCHS"
echo "Validate every: $VALIDATE_EVERY steps"
echo "Early stopping patience: $EARLY_STOPPING epochs"
echo "Master port: $MASTER_PORT"
echo "Config: $CONFIG"
echo "Manifest dir: $MANIFEST_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
if [ -n "$RESUME" ]; then
    echo "Resume from: $RESUME"
fi
echo "Wandb: $WANDB"
echo "=============================================="
echo ""

# =============================================================================
# Build command
# =============================================================================

CMD="poetry run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m src.clef.piano.train"
CMD="$CMD --config $CONFIG"
CMD="$CMD --manifest-dir $MANIFEST_DIR"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --gradient-accumulation-steps $GRADIENT_ACCUM"
CMD="$CMD --gradient-clip $GRADIENT_CLIP"
CMD="$CMD --max-epochs $MAX_EPOCHS"
CMD="$CMD --validate-every-n-steps $VALIDATE_EVERY"
CMD="$CMD --early-stopping-patience $EARLY_STOPPING"

if [ "$WANDB" = "true" ]; then
    CMD="$CMD --wandb"
    CMD="$CMD --wandb-project $WANDB_PROJECT"
    CMD="$CMD --wandb-entity $WANDB_ENTITY"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# =============================================================================
# Run training
# =============================================================================

echo "Running command:"
echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS $CMD"
echo ""

cd "$(dirname "$0")/../../.."
# CUDA_DEVICE_ORDER ensures consistent GPU numbering by PCI bus ID
# PYTORCH_CUDA_ALLOC_CONF prevents memory fragmentation (5GB reserved but unallocated)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS $CMD
