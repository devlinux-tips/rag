# RAG Service Docker Setup Guide

## Quick Start

```bash
# 1. Clone and navigate to the RAG service directory
cd services/rag-service

# 2. Copy environment file and customize if needed
cp .env.example .env

# 3. Create required directories
mkdir -p system/logs system/temp system/backups

# 4. Build and start all services
docker-compose up --build -d

# 5. Wait for Ollama model download (first time only)
docker-compose logs -f ollama-puller

# 6. Access the chat interface
open http://localhost:8080
```

## Architecture Overview

The Docker setup includes these services:

### Core Services
- **rag-app**: Main RAG application (FastAPI web server on port 8080)
- **rag-cli**: CLI interface for document processing and queries
- **postgres**: PostgreSQL database for chat persistence
- **ollama**: Local LLM server with qwen2.5:7b-instruct model

### Data Persistence
- **postgres_data**: PostgreSQL database files
- **rag_data**: Application data (documents, processed files)
- **rag_models**: ML model storage
- **ollama_data**: Ollama models and cache
- **chromadb_data**: Vector database storage

### Network
- **rag-network**: Bridge network (172.20.0.0/16)
- **Service IPs**:
  - postgres: 172.20.0.10
  - ollama: 172.20.0.20
  - rag-app: 172.20.0.30

## Configuration

### Environment Variables (.env)
```bash
# Database password
POSTGRES_PASSWORD=your_secure_password

# Optional: OpenRouter API for cloud LLM fallback
OPENROUTER_API_KEY=your_openrouter_key

# Application settings
RAG_ENV=docker
RAG_LOG_LEVEL=INFO
RAG_DEFAULT_TENANT=development
RAG_DEFAULT_USER=docker_user
RAG_DEFAULT_LANGUAGE=hr
```

### Docker Configuration Override
The system uses `config/docker.toml` which overrides the default `config.toml` for Docker-specific settings:

- Database host points to `postgres` service
- Ollama URL points to `ollama:11434`
- Paths configured for Docker volumes
- Reduced resource limits for container deployment

## Service Management

### Start All Services
```bash
docker-compose up -d
```

### Start Individual Services
```bash
# Database only
docker-compose up -d postgres

# LLM server only
docker-compose up -d ollama

# RAG application only
docker-compose up -d rag-app
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-app
docker-compose logs -f postgres
docker-compose logs -f ollama
```

### Health Checks
```bash
# Check all service status
docker-compose ps

# Test individual services
curl http://localhost:8080/health          # RAG app health
curl http://localhost:11434/api/version    # Ollama version
docker-compose exec postgres pg_isready -U raguser -d ragdb  # PostgreSQL
```

## Using the RAG System

### Web Interface
1. Open http://localhost:8080 in your browser
2. Click "New Conversation" to start chatting
3. Ask questions in Croatian or English
4. The system will automatically search documents and provide RAG-enhanced responses

### CLI Interface
```bash
# Access the CLI container
docker-compose exec rag-cli bash

# Inside the container, use the RAG CLI:
python rag.py --tenant development --user docker_user --language hr query "Što je RAG sustav?"
python rag.py --tenant development --user docker_user --language en query "What is RAG?"

# Process documents
python rag.py --tenant development --user docker_user --language hr process-docs /app/data/development/users/docker_user/documents/hr/

# Check system status
python rag.py --language hr status
python rag.py --language hr list-collections
```

### Document Processing
1. Copy documents to the data volume:
```bash
# Copy files to the container
docker-compose exec rag-app mkdir -p /app/data/development/users/docker_user/documents/hr
docker cp your_document.pdf $(docker-compose ps -q rag-app):/app/data/development/users/docker_user/documents/hr/
```

2. Process documents via CLI:
```bash
docker-compose exec rag-cli python rag.py --tenant development --user docker_user --language hr process-docs /app/data/development/users/docker_user/documents/hr/
```

## Troubleshooting

### Common Issues

#### Build Timeouts
The Docker build downloads large ML libraries (PyTorch ~900MB). If builds timeout:
```bash
# Increase Docker build timeout
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain
docker-compose build --no-cache rag-app
```

#### Ollama Model Download
First startup takes time to download the LLM model:
```bash
# Monitor model download progress
docker-compose logs -f ollama-puller

# Manually pull model if needed
docker-compose exec ollama ollama pull qwen2.5:7b-instruct
```

#### Database Connection Issues
```bash
# Check PostgreSQL is ready
docker-compose exec postgres pg_isready -U raguser -d ragdb

# Reset database if needed
docker-compose down -v
docker-compose up -d postgres
```

#### Memory Issues
If containers crash due to memory:
```bash
# Check Docker memory allocation
docker stats

# Reduce model workers in config/docker.toml:
# max_workers = 1
# batch_processing_size = 2
```

### Port Conflicts
If ports are already in use:
```bash
# Check what's using the ports
sudo netstat -tulpn | grep :8080
sudo netstat -tulpn | grep :5432
sudo netstat -tulpn | grep :11434

# Modify docker-compose.yml ports section:
# ports:
#   - "8081:8080"  # Change external port
```

### Volume Permissions
If you get permission errors:
```bash
# Fix volume permissions
sudo chown -R 1000:1000 data/ models/ system/

# Or recreate with proper permissions
docker-compose down -v
sudo rm -rf data/ models/ system/
mkdir -p system/logs system/temp system/backups
docker-compose up -d
```

## Development Tips

### Hot Reload
For development with hot reload:
```bash
# Start with volume mounting source code
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Debugging
```bash
# Access running container for debugging
docker-compose exec rag-app bash

# Run Python debugger
docker-compose exec rag-app python -c "
import sys; sys.path.insert(0, '/app/src')
from utils.config_loader import get_unified_config
print(get_unified_config())
"
```

### Performance Monitoring
```bash
# Monitor resource usage
docker stats

# Check container health
docker-compose exec rag-app curl -f http://localhost:8080/health

# Database performance
docker-compose exec postgres psql -U raguser -d ragdb -c "
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats WHERE tablename LIKE '%chat%';"
```

## Data Backup

### Backup All Data
```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d_%H%M%S)

# Backup PostgreSQL
docker-compose exec postgres pg_dump -U raguser ragdb > backups/$(date +%Y%m%d_%H%M%S)/postgres_backup.sql

# Backup volumes
docker run --rm -v rag_data:/data -v $(pwd)/backups/$(date +%Y%m%d_%H%M%S):/backup alpine tar czf /backup/rag_data.tar.gz -C /data .
docker run --rm -v ollama_data:/data -v $(pwd)/backups/$(date +%Y%m%d_%H%M%S):/backup alpine tar czf /backup/ollama_data.tar.gz -C /data .
docker run --rm -v chromadb_data:/data -v $(pwd)/backups/$(date +%Y%m%d_%H%M%S):/backup alpine tar czf /backup/chromadb_data.tar.gz -C /data .
```

### Restore Data
```bash
# Restore PostgreSQL
docker-compose exec -T postgres psql -U raguser ragdb < backups/20241220_120000/postgres_backup.sql

# Restore volumes
docker run --rm -v rag_data:/data -v $(pwd)/backups/20241220_120000:/backup alpine tar xzf /backup/rag_data.tar.gz -C /data
```

## Security Considerations

1. **Change default passwords** in `.env` file
2. **Use secrets** for production deployments
3. **Enable PostgreSQL SSL** for production
4. **Restrict network access** using Docker networks
5. **Regular security updates** for base images

## Production Deployment

For production deployment:

1. Use Docker Swarm or Kubernetes
2. Configure external PostgreSQL cluster
3. Use external load balancer
4. Enable SSL/TLS termination
5. Configure log aggregation
6. Set up monitoring and alerting
7. Use secrets management
8. Configure backup automation

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify health: `curl http://localhost:8080/health`
3. Test CLI: `docker-compose exec rag-cli python rag.py --help`
4. Review configuration: `config/docker.toml`