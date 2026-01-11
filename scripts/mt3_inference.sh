#!/bin/bash
# MT3 Docker Inference Script
# Usage:
#   ./scripts/mt3_inference.sh start       # Start Docker container
#   ./scripts/mt3_inference.sh test        # Test with single file
#   ./scripts/mt3_inference.sh asap        # Run ASAP batch
#   ./scripts/mt3_inference.sh stop        # Stop container

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

# Docker settings
CONTAINER_NAME="mt3-inference"
GPU_DEVICE="5"  # Change this for different GPU
API_PORT="5000"
API_URL="http://localhost:$API_PORT/batch-transcribe"

# Mount mapping: $DATA_DIR on host -> /data on container
# So ./data/test/test.mp3 on host = /data/test/test.mp3 in container

case "$1" in
  start)
    echo "Starting MT3 Docker container..."
    mkdir -p "$DATA_DIR/test_output"
    mkdir -p "$DATA_DIR/asap_midi_output"
    mkdir -p "$DATA_DIR/zeng_5bar_midi"

    # Stop existing container if running
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    # Mount entire data dir to /data
    docker run -d --name $CONTAINER_NAME \
      -p $API_PORT:5000 \
      --gpus "device=$GPU_DEVICE" \
      --ipc=host \
      -v "$DATA_DIR:/data" \
      mt3

    echo "Container started. Waiting for API to be ready..."
    sleep 15
    echo "API ready at $API_URL"
    echo ""
    echo "Mount mapping:"
    echo "  Host:      $DATA_DIR"
    echo "  Container: /data"
    ;;

  test)
    echo "Running single file test..."
    cd "$PROJECT_ROOT"

    # Paths use /data prefix (container mount point)
    python3 -m src.inference.batch_transcribe \
      --mode single_file \
      --input-dir /data \
      --input-file "/data/test/test.mp3" \
      --output-dir /data/test_output \
      --api-url "$API_URL" \
      --model piano

    echo ""
    echo "Done! Output: $DATA_DIR/test_output/"
    ls -la "$DATA_DIR/test_output/" 2>/dev/null || echo "(no output yet)"
    ;;

  asap)
    echo "Running ASAP test set batch inference..."
    cd "$PROJECT_ROOT"

    python3 -m src.inference.batch_transcribe \
      --mode asap_batch \
      --input-dir /data/asap_test_set \
      --metadata-csv "$DATA_DIR/asap_test_set/metadata.csv" \
      --output-dir /data/asap_midi_output \
      --api-url "$API_URL" \
      --model piano

    echo ""
    echo "Done! Output: $DATA_DIR/asap_midi_output/"
    ;;

  zeng5bar)
    echo "=========================================="
    echo "MT3 Inference: Zeng 5-bar Test Chunks"
    echo "=========================================="
    echo "Input:  $DATA_DIR/zeng_5bar_test/ (13,335 files)"
    echo "Output: $DATA_DIR/zeng_5bar_midi/"
    echo "GPU:    $GPU_DEVICE"
    echo "Estimated time: 3-5 hours"
    echo ""
    cd "$PROJECT_ROOT"

    # Use host path for glob, container path for API
    python3 -m src.inference.batch_transcribe \
      --mode zeng_5bar \
      --input-dir "$DATA_DIR/zeng_5bar_test" \
      --container-path /data/zeng_5bar_test \
      --output-dir /data/zeng_5bar_midi \
      --api-url "$API_URL" \
      --model piano

    echo ""
    echo "=========================================="
    echo "Done! Output: $DATA_DIR/zeng_5bar_midi/"
    total_files=$(ls $DATA_DIR/zeng_5bar_midi/*.mid 2>/dev/null | wc -l)
    echo "Total MIDI files: $total_files / 13335"
    echo "=========================================="
    ;;

  stop)
    echo "Stopping MT3 Docker container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo "Container stopped."
    ;;

  logs)
    docker logs -f $CONTAINER_NAME
    ;;

  status)
    echo "Container status:"
    docker ps -a | grep $CONTAINER_NAME || echo "Container not found"
    echo ""
    echo "Testing API connection..."
    curl -s -o /dev/null -w "%{http_code}" "$API_URL" 2>/dev/null || echo "API not reachable"
    ;;

  *)
    echo "MT3 Docker Inference Script"
    echo ""
    echo "Usage: $0 {start|test|asap|zeng5bar|stop|logs|status}"
    echo ""
    echo "Commands:"
    echo "  start    - Start Docker container (GPU $GPU_DEVICE)"
    echo "  test     - Run single file test (data/test/test.mp3)"
    echo "  asap     - Run ASAP test set batch (80 files)"
    echo "  zeng5bar - Run Zeng 5-bar chunks (13,335 files, ~3-5 hours)"
    echo "  stop     - Stop Docker container"
    echo "  logs     - View container logs"
    echo "  status   - Check container and API status"
    echo ""
    echo "Paths:"
    echo "  Project: $PROJECT_ROOT"
    echo "  Data:    $DATA_DIR"
    echo ""
    echo "Data structure:"
    echo "  data/test/test.mp3          - Single file test"
    echo "  data/asap_test_set/         - ASAP dataset"
    echo "  data/zeng_5bar_test/        - Zeng 5-bar chunks (13,335 files, 8.5GB)"
    echo "  data/test_output/           - Single file output"
    echo "  data/asap_midi_output/      - ASAP batch output"
    echo "  data/zeng_5bar_midi/        - Zeng 5-bar MIDI output"
    ;;
esac
