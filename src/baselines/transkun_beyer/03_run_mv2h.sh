#!/bin/bash
# =============================================================================
# Stage 3: MV2H Evaluation
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEF_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Configuration
ASAP_DIR="${ASAP_DIR:-$CLEF_ROOT/data/datasets/asap_test_set}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLEF_ROOT/data/experiments/transkun_beyer}"
CHUNK_CSV="${CHUNK_CSV:-$CLEF_ROOT/src/evaluation/asap/test_chunk_set.csv}"
MODE="${MODE:-chunks}"  # 'full' or 'chunks'
WORKERS="${WORKERS:-8}"

echo "=============================================="
echo "MV2H Evaluation"
echo "=============================================="
echo "Mode:        $MODE"
echo "ASAP dir:    $ASAP_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Workers:     $WORKERS"
if [ "$MODE" = "chunks" ]; then
    echo "Chunk CSV:   $CHUNK_CSV"
fi
echo "=============================================="

cd "$CLEF_ROOT"

if [ "$MODE" = "full" ]; then
    poetry run python -m src.baselines.transkun_beyer.transkun_beyer_evaluate \
        --mode full \
        --audio_dir "$ASAP_DIR" \
        --gt_dir "$ASAP_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/results/full.csv" \
        --workers "$WORKERS"
else
    poetry run python -m src.baselines.transkun_beyer.transkun_beyer_evaluate \
        --mode chunks \
        --audio_dir "$ASAP_DIR" \
        --gt_dir "$ASAP_DIR" \
        --chunk_csv "$CHUNK_CSV" \
        --output_dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/results/chunks.csv" \
        --workers "$WORKERS"
fi

echo "=============================================="
echo "Evaluation Complete"
echo "=============================================="
echo "Results: $OUTPUT_DIR/results/"
echo "=============================================="
