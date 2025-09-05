# Multilingual RAG Platform - Monorepo Makefile

.PHONY: help setup dev test build clean deploy

# Default target
help:
	@echo "Multilingual RAG Platform - Available Commands:"
	@echo ""
	@echo "  setup     - Setup all services (install dependencies)"
	@echo "  dev       - Start development environment"
	@echo "  test      - Run all tests"
	@echo "  build     - Build all services"
	@echo "  clean     - Clean all build artifacts"
	@echo "  deploy    - Deploy to production"
	@echo ""
	@echo "Service-specific commands:"
	@echo "  rag-dev   - Start RAG service development"
	@echo "  api-dev   - Start Platform API development"
	@echo "  ui-dev    - Start User Frontend development"

# Setup all services
setup:
	@echo "🔧 Setting up all services..."
	@echo "📦 Setting up RAG service..."
	cd services/rag-service && python -m pip install -r requirements.txt
	@echo "⚗️ Setting up Platform API..."
	cd services/platform-api && mix deps.get
	@echo "⚛️ Setting up User Frontend..."
	cd services/user-frontend && npm install
	@echo "✅ All services setup complete!"

# Start development environment
dev:
	@echo "🚀 Starting development environment..."
	docker-compose -f shared/config/docker-compose.dev.yml up -d
	@echo "✅ Development environment started!"

# Run all tests
test:
	@echo "🧪 Running all tests..."
	@echo "🐍 Testing RAG service..."
	cd services/rag-service && python -m pytest tests/ -v
	@echo "⚗️ Testing Platform API..."
	cd services/platform-api && mix test
	@echo "⚛️ Testing User Frontend..."
	cd services/user-frontend && npm test
	@echo "✅ All tests completed!"

# Build all services
build:
	@echo "🏗️ Building all services..."
	@echo "🐍 Building RAG service..."
	cd services/rag-service && docker build -t rag-service .
	@echo "⚗️ Building Platform API..."
	cd services/platform-api && docker build -t platform-api .
	@echo "⚛️ Building User Frontend..."
	cd services/user-frontend && npm run build && docker build -t user-frontend .
	@echo "✅ All services built!"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning all services..."
	cd services/rag-service && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	cd services/platform-api && mix clean
	cd services/user-frontend && rm -rf dist/ node_modules/.cache/
	@echo "✅ Cleanup complete!"

# Deploy to production
deploy:
	@echo "🚀 Deploying to production..."
	docker-compose -f shared/config/docker-compose.yml up -d
	@echo "✅ Deployment complete!"

# Service-specific development commands
rag-dev:
	@echo "🐍 Starting RAG service development..."
	cd services/rag-service && python -c "import asyncio; from src.pipeline.rag_system import RAGSystem; asyncio.run(RAGSystem(language='hr').initialize())"

api-dev:
	@echo "⚗️ Starting Platform API development..."
	cd services/platform-api && mix phx.server

ui-dev:
	@echo "⚛️ Starting User Frontend development..."
	cd services/user-frontend && npm run dev
