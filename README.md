# Multilingual RAG Platform

## Overview

This is a monorepo containing a complete multilingual RAG (Retrieval-Augmented Generation) platform optimized for Croatian and English languages, featuring a production-ready Python RAG service, Elixir/Phoenix API orchestration, and modern React frontend.

## Services

### 🐍 RAG Service (Python)
Production-ready multilingual RAG system with BGE-M3 embeddings and qwen2.5:7b-instruct generation.
- **Location**: `services/rag-service/`
- **Documentation**: [RAG Service README](services/rag-service/README.md)

### ⚗️ Platform API (Elixir/Phoenix)
Job orchestration, rate limiting, and embedded LiveView admin interface.
- **Location**: `services/platform-api/`
- **Features**: Oban jobs, multi-layer rate limiting, feature flags

### ⚛️ User Frontend (React)
Modern search interface with real-time job progress tracking.
- **Location**: `services/user-frontend/`
- **Technology**: React + TypeScript + Vite

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+
- PostgreSQL 16
- Redis 7
- Weaviate 1.33.0
- Ollama with qwen2.5:7b-instruct
- (Coming soon) Elixir 1.15+ with Phoenix

### Installation

For local deployment (no Docker):
```bash
# Full automated installation
make install

# Or manual setup
make install-deps    # Install dependencies
make setup          # Setup environment + Prisma
make build          # Build all services
```

### Quick Start
```bash
# Start all services
make start

# Check status
make status

# View logs
make logs SERVICE=rag-api

# Stop all services
make stop
```

## Makefile Commands

### Service Management
```bash
make start              # Start all services
make stop               # Stop all services
make restart            # Restart all services
make restart-one SERVICE=web-api  # Restart specific service
make status             # Check service status
make logs SERVICE=rag-api         # View service logs
make health             # Run health checks
```

### Database (Prisma)
```bash
make prisma-generate    # Generate Prisma client
make prisma-migrate     # Apply migrations (production)
make prisma-push        # Push schema changes (development)
make prisma-studio      # Open database GUI
make prisma-validate    # Validate schema
```

### Build & Development
```bash
make build              # Build all services
make build-web-api      # Build web-api only
make build-web-ui       # Build web-ui only
make dev-web-api        # Run web-api in dev mode (hot reload)
make dev-web-ui         # Run web-ui in dev mode (hot reload)
```

### Testing
```bash
make test               # Run all tests
make test-coverage      # Run tests with coverage
```

### Utilities
```bash
make clean              # Clean build artifacts
make db-shell           # Connect to PostgreSQL
```

### External Access (ngrok)
```bash
# Manage ngrok tunnel for external access
./ngrok-tunnel.sh start     # Start tunnel
./ngrok-tunnel.sh status    # Get public URL
./ngrok-tunnel.sh stop      # Stop tunnel
./ngrok-tunnel.sh logs      # View logs
```

For complete list of commands:
```bash
make help
```

## Architecture

The platform uses a service-oriented monorepo architecture with:
- **RAG Service**: Core multilingual processing and generation
- **Platform API**: Job orchestration and admin interface
- **User Frontend**: React-based user interface
- **Shared Resources**: Common configuration and tooling

## Documentation

- 📚 [Platform Architecture Roadmap](docs/PLATFORM_ARCHITECTURE_ROADMAP.md)
- 🌐 [Intelligent Web Platform](docs/INTELLIGENT_WEB_PLATFORM.md)
- 🔌 [API Documentation](docs/api/)
- 🚀 [Deployment Guide](docs/deployment/)
- 🇭🇷 [Croatian Language Features](docs/croatian-language/)
- 👥 [User Guides](docs/user-guides/)
- ⚡ [Performance Optimization](docs/PERFORMANCE_OPTIMIZATION.md)
- 📱 [Mobile App Plan](docs/MOBILE_APP_PLAN.md)

## Development

See individual service READMEs for service-specific development instructions:
- [RAG Service Development](services/rag-service/README.md)
- [Platform API Development](services/platform-api/README.md)
- [User Frontend Development](services/user-frontend/README.md)
