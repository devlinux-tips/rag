# RAG Platform - Docker Setup

Complete containerized deployment of the multilingual RAG platform with full pipeline.

## Architecture

The Docker setup provides a complete RAG platform with the following services:

- **Weaviate** (port 8080) - Vector database for document embeddings
- **PostgreSQL** (port 5434) - Relational database for web API
- **Redis** (port 6379) - Real-time features and caching
- **RAG API** (port 8082) - Python FastAPI service for RAG operations
- **Web API** (port 3000) - Node.js/TypeScript service with tRPC, auth, and multi-tenancy

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 8GB RAM (for ML models)
- 10GB free disk space

### 1. Start the Platform

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Wait for Services to Initialize

The services will start in dependency order:
1. Infrastructure (Weaviate, PostgreSQL, Redis)
2. RAG API (Python service - downloads ML models on first run)
3. Web API (Node.js service)

Initial startup may take 5-10 minutes while ML models are downloaded.

### 3. Test the Platform

#### Test RAG API directly:
```bash
curl -X POST "http://localhost:8082/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Što govore propisi o zaštiti okoliša i gorivima?",
    "tenant": "development",
    "user": "dev_user",
    "language": "hr",
    "scope": "feature",
    "feature": "narodne_novine"
  }'
```

#### Test Web API health:
```bash
curl http://localhost:3000/api/v1/health
```

## Configuration

### Environment Variables

Key environment variables are configured in `docker-compose.yml`:

- **Authentication**: `AUTH_MODE=mock` (mock authentication for testing)
- **OpenRouter**: Uses API key from config for LLM queries
- **Multi-tenancy**: Mock tenant `development` with user `dev_user`
- **Languages**: Croatian (`hr`) as default, English (`en`) supported

### Data Persistence

The following data is persisted in Docker volumes:
- `weaviate_data` - Vector embeddings and collections
- `postgres_data` - User sessions, chat history
- `redis_data` - Real-time session data

### File Mounts

Host directories mounted into containers:
- `./services/rag-service/data` - RAG documents and processing data
- `./services/rag-service/models` - ML models cache
- `./services/rag-service/config` - Configuration files

## Service Details

### RAG API (Python)
- **Port**: 8082
- **Health**: `http://localhost:8082/health`
- **Features**: Document processing, vector search, LLM generation
- **Models**: BAAI/bge-m3 (multilingual embeddings), Croatian ELECTRA

### Web API (Node.js)
- **Port**: 3000
- **Health**: `http://localhost:3000/api/v1/health`
- **Features**: tRPC API, authentication, WebSocket support, rate limiting
- **Frontend CORS**: Configured for ports 3001, 5173

### Infrastructure
- **Weaviate**: Vector database with HNSW indexing, compression enabled
- **PostgreSQL**: Relational database for structured data
- **Redis**: Session storage and real-time features

## Development

### Building Services

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build rag-api
docker-compose build web-api
```

### Debugging

```bash
# View service logs
docker-compose logs rag-api
docker-compose logs web-api

# Access service shell
docker-compose exec rag-api bash
docker-compose exec web-api sh

# Monitor resource usage
docker stats
```

### Updating Code

```bash
# Rebuild and restart after code changes
docker-compose down
docker-compose build
docker-compose up -d
```

## Production Considerations

### Security
- Change `JWT_SECRET` for production
- Use proper authentication (not mock mode)
- Configure firewall rules
- Enable HTTPS/TLS termination

### Scaling
- Use external PostgreSQL and Redis clusters
- Scale RAG API and Web API horizontally
- Configure load balancer
- Optimize Weaviate for production workloads

### Monitoring
- Add Prometheus metrics
- Configure health check endpoints
- Set up log aggregation
- Monitor resource usage

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase Docker memory limit to 8GB+
2. **Slow Startup**: ML model downloads take time on first run
3. **Port Conflicts**: Ensure ports 3000, 8080, 8082, 5434, 6379 are available
4. **Volume Permissions**: Check Docker volume mount permissions

### Useful Commands

```bash
# Reset all data
docker-compose down -v
docker-compose up -d

# Check service health
docker-compose exec rag-api curl http://localhost:8082/health
docker-compose exec web-api curl http://localhost:3000/api/v1/health

# View resource usage
docker system df
docker volume ls
```

## API Documentation

### Web API Endpoints
- `GET /api/v1/health` - Service health status
- `GET /api/v1/info` - Platform information
- `POST /trpc/*` - tRPC endpoints for chat, documents, etc.

### RAG API Endpoints
- `GET /health` - Service health
- `POST /api/v1/query` - RAG query processing
- `GET /api/v1/status` - System status

For detailed API documentation, see the OpenAPI specs in each service directory.