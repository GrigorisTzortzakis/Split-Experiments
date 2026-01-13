#!/usr/bin/env bash
# Launch N clients + split server + Fed server via Docker Compose.
# Usage:  ./run_docker.sh <num_clients> <num_rounds> [local_epochs]

set -euo pipefail

NUM_CLIENTS=${1:-2}
NUM_ROUNDS=${2:-5}
LOCAL_EPOCHS=${3:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  DC=(docker compose -f "$COMPOSE_FILE")
elif command -v docker-compose >/dev/null 2>&1; then
  DC=(docker-compose -f "$COMPOSE_FILE")
else
  echo "ERROR: Docker Compose not found. Install Docker Desktop and ensure 'docker compose' works." >&2
  exit 1
fi

echo "Starting SplitFed Docker stack: ${NUM_CLIENTS} clients, ${NUM_ROUNDS} rounds, local_epochs=${LOCAL_EPOCHS}"

# Clean up any previous containers
"${DC[@]}" down --remove-orphans 2>/dev/null || true

# Build images (picks up code/Dockerfile changes)
"${DC[@]}" build

# Start servers
echo "Starting Fed server + split server + MLflow UI..."
NUM_CLIENTS=$NUM_CLIENTS "${DC[@]}" up -d fed_server server mlflow_ui

# Wait for server to be ready
sleep 2

# Start clients
echo "Starting ${NUM_CLIENTS} clients..."
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  echo "  Starting client ${i}..."
  NUM_CLIENTS=$NUM_CLIENTS CLIENT_ID=$i NUM_ROUNDS=$NUM_ROUNDS LOCAL_EPOCHS=$LOCAL_EPOCHS \
    "${DC[@]}" run -d --name "splitfed_client_${i}" client
done

echo ""
echo "All containers started."
echo "View logs:"
echo "  Split:    docker logs -f splitfed_server"
echo "  Fed:      docker logs -f splitfed_fed_server"
echo "  MLflow:   docker logs -f splitfed_mlflow_ui"
echo "  Client N: docker logs -f splitfed_client_N"

echo "MLflow UI (browser): http://127.0.0.1:5001"
echo ""
echo "Artifacts live in Docker named volumes (no host folders created)."
echo "To copy results to the current directory (files only):"
echo "  docker cp splitfed_server:/outputs/metrics.json ./metrics.json"
echo "  docker cp splitfed_server:/outputs/confusion_matrix.pt ./confusion_matrix.pt"
echo "  docker cp splitfed_server:/outputs/confusion_matrix_counts.png ./confusion_matrix_counts.png"
echo ""
echo "To stop everything:"
echo "  ${DC[*]} down"
