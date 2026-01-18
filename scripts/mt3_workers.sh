#!/bin/bash
# MT3 Parallel Workers for Zeng 5-bar Test Chunks
# Usage:
#   ./scripts/mt3_workers.sh start        # Start two Docker containers (Both GPU 5)
#   ./scripts/mt3_workers.sh infer        # Run two inference workers in parallel
#   ./scripts/mt3_workers.sh stop         # Stop both containers
#   ./scripts/mt3_workers.sh status       # Check container status

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

case "$1" in
  start)
    echo "=========================================="
    echo "Starting Two MT3 Docker Containers (Both GPU 5)"
    echo "=========================================="

    mkdir -p "$DATA_DIR/zeng_5bar_midi"
    mkdir -p "$DATA_DIR/zeng_5bar_split1"
    mkdir -p "$DATA_DIR/zeng_5bar_split2"

    # Stop existing containers if running
    docker stop mt3-gpu5-w1 mt3-gpu5-w2 2>/dev/null || true
    docker rm mt3-gpu5-w1 mt3-gpu5-w2 2>/dev/null || true

    echo ""
    echo "[1/2] Starting Worker 1 (GPU 5, port 5000)..."
    docker run -d --name mt3-gpu5-w1 \
      -p 5000:5000 \
      --gpus "device=5" \
      --ipc=host \
      -v "$DATA_DIR:/data" \
      mt3

    echo "[2/2] Starting Worker 2 (GPU 5, port 5001)..."
    docker run -d --name mt3-gpu5-w2 \
      -p 5001:5000 \
      --gpus "device=5" \
      --ipc=host \
      -v "$DATA_DIR:/data" \
      mt3

    echo ""
    echo "Waiting for APIs to be ready..."
    sleep 20

    echo ""
    echo "=========================================="
    echo "✓ Containers started successfully!"
    echo "=========================================="
    echo "Worker 1: http://localhost:5000/batch-transcribe"
    echo "Worker 2: http://localhost:5001/batch-transcribe"
    echo "(Both on GPU 5)"
    ;;

  infer)
    echo "=========================================="
    echo "Starting Two Inference Workers (Both on GPU 5)"
    echo "=========================================="
    echo "Split 1 (GPU 5): $(ls $DATA_DIR/zeng_5bar_split1/*.wav 2>/dev/null | wc -l) files"
    echo "Split 2 (GPU 5): $(ls $DATA_DIR/zeng_5bar_split2/*.wav 2>/dev/null | wc -l) files"
    echo ""
    cd "$PROJECT_ROOT"

    # Worker 1 - GPU 5 (port 5000)
    echo "[1/2] Starting Worker 1 (GPU 5, port 5000)..."
    python3 -m src.inference.batch_transcribe \
      --mode zeng_5bar \
      --input-dir "$DATA_DIR/zeng_5bar_split1" \
      --container-path /data/zeng_5bar_split1 \
      --output-dir /data/zeng_5bar_midi \
      --api-url "http://localhost:5000/batch-transcribe" \
      --model piano &
    WORKER1_PID=$!

    # Worker 2 - GPU 5 (port 5001)
    echo "[2/2] Starting Worker 2 (GPU 5, port 5001)..."
    python3 -m src.inference.batch_transcribe \
      --mode zeng_5bar \
      --input-dir "$DATA_DIR/zeng_5bar_split2" \
      --container-path /data/zeng_5bar_split2 \
      --output-dir /data/zeng_5bar_midi \
      --api-url "http://localhost:5001/batch-transcribe" \
      --model piano &
    WORKER2_PID=$!

    echo ""
    echo "=========================================="
    echo "✓ Both workers started (PID: $WORKER1_PID, $WORKER2_PID)"
    echo "=========================================="
    echo ""
    echo "Monitor progress:"
    echo "  watch -n 60 'ls $DATA_DIR/zeng_5bar_midi/*.mid | wc -l'"
    echo ""
    echo "Waiting for both workers to complete..."
    wait $WORKER1_PID $WORKER2_PID

    echo ""
    echo "=========================================="
    echo "✓ All workers completed!"
    echo "=========================================="
    total_files=$(ls $DATA_DIR/zeng_5bar_midi/*.mid 2>/dev/null | wc -l)
    echo "Total MIDI files: $total_files / 13335"
    ;;

  stop)
    echo "Stopping MT3 Docker containers..."
    docker stop mt3-gpu5-w1 mt3-gpu5-w2 2>/dev/null || true
    docker rm mt3-gpu5-w1 mt3-gpu5-w2 2>/dev/null || true
    echo "Containers stopped."
    ;;

  status)
    echo "Container status:"
    docker ps -a | grep -E "mt3-gpu5-w1|mt3-gpu5-w2" || echo "No containers found"
    echo ""
    echo "Testing API connections..."
    echo -n "Worker 1 (5000): "
    curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:5000/batch-transcribe" 2>/dev/null || echo "not reachable"
    echo -n "Worker 2 (5001): "
    curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:5001/batch-transcribe" 2>/dev/null || echo "not reachable"
    ;;

  *)
    echo "MT3 Parallel Workers Script"
    echo ""
    echo "Usage: $0 {start|infer|stop|status}"
    echo ""
    echo "Commands:"
    echo "  start   - Start two Docker containers (Both GPU 5)"
    echo "  infer   - Run two inference workers in parallel (blocking)"
    echo "  stop    - Stop both Docker containers"
    echo "  status  - Check container and API status"
    echo ""
    echo "Example workflow:"
    echo "  ./scripts/mt3_workers.sh start"
    echo "  ./scripts/mt3_workers.sh infer"
    echo "  ./scripts/mt3_workers.sh stop"
    ;;
esac
