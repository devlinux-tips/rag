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
- Elixir 1.15+ with Phoenix
- Node.js 18+
- PostgreSQL
- Ollama with qwen2.5:7b-instruct

### Quick Start
```bash
# Setup all services
make setup

# Start development environment
make dev

# Run tests
make test
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
