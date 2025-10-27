#!/bin/bash
# Sync environment files across the project
# This ensures all services use the same unified .env configuration

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
WEB_API_ENV="$PROJECT_ROOT/services/web-api/.env"
WEB_UI_ENV="$PROJECT_ROOT/services/web-ui/.env"
RAG_API_ENV="$PROJECT_ROOT/services/rag-api/.env"

echo "🔄 Syncing environment files..."

# Check if root .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Root .env file not found at $ENV_FILE"
    exit 1
fi

# Copy to services
echo "📋 Copying $ENV_FILE to $WEB_API_ENV"
cp "$ENV_FILE" "$WEB_API_ENV"

echo "📋 Copying $ENV_FILE to $WEB_UI_ENV"
cp "$ENV_FILE" "$WEB_UI_ENV"

echo "📋 Copying $ENV_FILE to $RAG_API_ENV"
cp "$ENV_FILE" "$RAG_API_ENV"

echo "✅ Environment files synchronized successfully!"