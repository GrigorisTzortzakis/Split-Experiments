#!/usr/bin/env bash
# Launch N clients + 1 server via Docker Compose.
# Usage:  ./run_docker.sh <num_clients> <num_epochs>

set -euo pipefail

NUM_CLIENTS=${1:-2}
NUM_EPOCHS=${2:-5}

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

echo "Starting split-learning Docker stack: ${NUM_CLIENTS} clients, ${NUM_EPOCHS} epochs"

# Clean up any previous containers
"${DC[@]}" down --remove-orphans 2>/dev/null || true

# Build images (picks up code/Dockerfile changes)
"${DC[@]}" build

# Start server
echo "Starting server..."
NUM_CLIENTS=$NUM_CLIENTS "${DC[@]}" up -d server

# Wait for server to be ready
sleep 2

# Start clients
echo "Starting ${NUM_CLIENTS} clients..."
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  echo "  Starting client ${i}..."
  NUM_CLIENTS=$NUM_CLIENTS CLIENT_ID=$i NUM_EPOCHS=$NUM_EPOCHS \
    "${DC[@]}" run -d --name "split_client_${i}" client
done

echo ""
echo "All containers started."
echo "View logs:"
echo "  Server:   docker logs -f split_server"
echo "  Client N: docker logs -f split_client_N"
echo ""
echo "Artifacts live in Docker named volumes (no host folders created)."
echo "To copy results to the current directory (files only):"
echo "  docker cp split_server:/outputs/metrics.json ./metrics.json"
echo "  docker cp split_server:/outputs/confusion_matrix.pt ./confusion_matrix.pt"
echo "  docker cp split_server:/outputs/confusion_matrix_counts.png ./confusion_matrix_counts.png"
echo ""
echo "To stop everything:"
echo "  ${DC[*]} down"
