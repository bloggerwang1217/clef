#!/bin/bash
# MT3 Worker on GPU 3 for Zeng 5-bar Test Chunks (split2)
# Usage:
#   ./scripts/mt3_workers_3only.sh start        # Start Worker on GPU 3 (port 5002)
#   ./scripts/mt3_workers_3only.sh infer        # Run inference for split2
#   ./scripts/mt3_workers_3only.sh stop         # Stop container
#   ./scripts/mt3_workers_3only.sh status       # Check container status

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

case "$1" in
  start)
    echo "=========================================="
    echo "Starting MT3 Worker on GPU 3 (split2)"
    echo "=========================================="

    mkdir -p "$DATA_DIR/zeng_5bar_midi"
    mkdir -p "$DATA_DIR/zeng_5bar_split2"

    # Stop existing container if running
    docker stop mt3-gpu3-w2 2>/dev/null || true
    docker rm mt3-gpu3-w2 2>/dev/null || true

    echo ""
    echo "Starting Worker on GPU 3 (port 5002)..."
    docker run -d --name mt3-gpu3-w2 \
      -p 5002:5000 \
      --gpus "device=3" \
      --ipc=host \
      -v "$DATA_DIR:/data" \
      mt3

    echo ""
    echo "Waiting for API to be ready..."
    sleep 20

    echo ""
    echo "=========================================="
    echo "✓ Container started successfully!"
    echo "=========================================="
    echo "Worker (GPU 3): http://localhost:5002/batch-transcribe"
    ;;

  infer)
    echo "=========================================="
    echo "Starting Inference Worker on GPU 3 (split2)"
    echo "=========================================="
    echo "Split 2 files: $(ls $DATA_DIR/zeng_5bar_split2/*.wav 2>/dev/null | wc -l)"
    echo ""
    cd "$PROJECT_ROOT"

    echo "Starting Worker (GPU 3, port 5002)..."
    python3 -m src.inference.batch_transcribe \
      --mode zeng_5bar \
      --input-dir "$DATA_DIR/zeng_5bar_split2" \
      --container-path /data/zeng_5bar_split2 \
      --output-dir /data/zeng_5bar_midi \
      --api-url "http://localhost:5002/batch-transcribe" \
      --model piano &
    WORKER_PID=$!

    echo ""
    echo "=========================================="
    echo "✓ Worker started (PID: $WORKER_PID)"
    echo "=========================================="
    echo ""
    echo "Monitor progress:"
    echo "  watch -n 60 'ls $DATA_DIR/zeng_5bar_midi/*.mid | wc -l'"
    echo ""
    echo "Waiting for worker to complete..."
    wait $WORKER_PID

    echo ""
    echo "=========================================="
    echo "✓ Worker completed!"
    echo "=========================================="
    total_files=$(ls $DATA_DIR/zeng_5bar_midi/*.mid 2>/dev/null | wc -l)
    echo "Total MIDI files: $total_files / 13335"
    ;;

  stop)
    echo "Stopping MT3 Docker container..."
    docker stop mt3-gpu3-w2 2>/dev/null || true
    docker rm mt3-gpu3-w2 2>/dev/null || true
    echo "Container stopped."
    ;;

  status)
    echo "Container status:"
    docker ps -a | grep mt3-gpu3-w2 || echo "No container found"
    echo ""
    echo "Testing API connection..."
    echo -n "Worker (GPU 3, port 5002): "
    curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:5002/batch-transcribe" 2>/dev/null || echo "not reachable"
    ;;

  *)
    echo "MT3 Worker Script (GPU 3 Only)"
    echo ""
    echo "Usage: $0 {start|infer|stop|status}"
    echo ""
    echo "Commands:"
    echo "  start   - Start Docker container on GPU 3 (port 5002)"
    echo "  infer   - Run inference worker for split2 (blocking)"
    echo "  stop    - Stop Docker container"
    echo "  status  - Check container and API status"
    echo ""
    echo "Example workflow:"
    echo "  ./scripts/mt3_workers_3only.sh start"
    echo "  ./scripts/mt3_workers_3only.sh infer"
    echo "  ./scripts/mt3_workers_3only.sh stop"
    ;;
esac
