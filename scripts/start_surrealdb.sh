#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/services/rag-service/data/surrealdb"

echo "🗄️  Starting SurrealDB..."
mkdir -p "$DATA_DIR"

surreal start \
  --log trace \
  --user root \
  --pass root \
  --bind 127.0.0.1:8000 \
  "surrealkv://$DATA_DIR/rag.db"
