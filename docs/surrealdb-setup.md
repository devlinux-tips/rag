# SurrealDB Multi-tenant RAG System Setup Guide

Complete guide for installing, configuring, and using SurrealDB with the Multi-tenant RAG system.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Database Setup](#database-setup)
5. [Connection Methods](#connection-methods)
6. [Schema Overview](#schema-overview)
7. [Common Queries](#common-queries)
8. [Development Workflow](#development-workflow)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting](#troubleshooting)

## Overview

SurrealDB serves as the metadata and analytics database for the Multi-tenant RAG system, storing:

- **Tenant Management**: Multi-tenant configurations and settings
- **User Management**: User accounts, roles, and preferences
- **Document Metadata**: Document information, processing status, categories
- **Search Analytics**: Query tracking, performance metrics, user satisfaction
- **Categorization Templates**: Language-specific prompt templates
- **System Configuration**: Runtime settings and feature flags

The vector embeddings are stored separately in ChromaDB, while SurrealDB handles all relational metadata and analytics.

## Prerequisites

- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 512MB, recommended 2GB+
- **Disk Space**: 100MB for SurrealDB + space for your data
- **Network**: Port 8000 (default) available
- **Optional**: Docker and Docker Compose

## Installation

### Quick Installation

Use the automated installation script:

```bash
# Clone the repository (if not done already)
git clone <repository-url>
cd learn-rag

# Run installation script
./scripts/install_surrealdb.sh
```

The script offers multiple installation methods:
1. **Official installer** (recommended)
2. **Package manager** (Homebrew on macOS)
3. **Manual binary download**
4. **Docker setup**

### Manual Installation Options

#### Option 1: Official Installer (Recommended)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://install.surrealdb.com | sh
```

#### Option 2: Homebrew (macOS)

```bash
brew install surrealdb/tap/surreal
```

#### Option 3: Docker

```bash
# Using the provided docker-compose file
./scripts/install_surrealdb.sh --docker

# Or manually
docker run -p 8000:8000 surrealdb/surrealdb:latest start \
  --user root --pass root file:///data/database.db
```

#### Option 4: Manual Binary Download

Download from [GitHub releases](https://github.com/surrealdb/surrealdb/releases):

```bash
# Linux AMD64
wget https://github.com/surrealdb/surrealdb/releases/download/v2.1.0/surreal-v2.1.0.linux-amd64.tar.gz
tar -xzf surreal-v2.1.0.linux-amd64.tar.gz
sudo install surreal /usr/local/bin/
```

### Verify Installation

```bash
surreal version
# Expected output: surreal 2.1.0 for linux on x86_64
```

## Database Setup

### Automated Setup

Use the setup script for complete database initialization:

```bash
# Default setup (file-based database)
./scripts/setup_surrealdb.sh

# Docker setup
./scripts/setup_surrealdb.sh --docker

# Server setup with custom settings
./scripts/setup_surrealdb.sh --host localhost --port 8000 --user admin --pass secret
```

### Manual Setup

#### 1. Start SurrealDB Server

**File-based (Development):**
```bash
surreal start --user root --pass root --bind 0.0.0.0:8000 file://./data/surrealdb/rag.db
```

**In-memory (Testing):**
```bash
surreal start --user root --pass root --bind 0.0.0.0:8000 memory
```

**RocksDB (Production):**
```bash
surreal start --user root --pass root --bind 0.0.0.0:8000 rocksdb://./data/surrealdb/
```

#### 2. Load Schema

```bash
surreal import --conn ws://127.0.0.1:8000 \
  --user root --pass root \
  --ns rag --db multitenant \
  ./services/rag-service/schema/multitenant_schema.surql
```

#### 3. Verify Setup

```bash
surreal sql --conn ws://127.0.0.1:8000 \
  --user root --pass root \
  --ns rag --db multitenant \
  "SELECT * FROM tenant;"
```

## Connection Methods

### CLI Connection

```bash
# Standard connection
surreal sql --conn ws://127.0.0.1:8000 \
  --user root --pass root \
  --ns rag --db multitenant

# Pretty formatted output
surreal sql --conn ws://127.0.0.1:8000 \
  --user root --pass root \
  --ns rag --db multitenant \
  --pretty \
  "SELECT * FROM tenant;"
```

### Docker Connection

```bash
# Connect to Docker container
docker exec -it rag-surrealdb surreal sql \
  --conn http://localhost:8000 \
  --user root --pass root \
  --ns rag --db multitenant
```

### Programmatic Connection (Python)

```python
import asyncio
from surrealdb import Surreal

async def main():
    async with Surreal("ws://localhost:8000") as db:
        await db.signin({"user": "root", "pass": "root"})
        await db.use("rag", "multitenant")

        # Query tenants
        result = await db.select("tenant")
        print(result)

asyncio.run(main())
```

### Environment Variables

Set these for easier connections:

```bash
export SURREAL_HOST="127.0.0.1"
export SURREAL_PORT="8000"
export SURREAL_USER="root"
export SURREAL_PASS="root"
export SURREAL_NS="rag"
export SURREAL_DB="multitenant"
```

## Schema Overview

### Core Tables

#### Tenant Management
```sql
-- Tenant configuration and settings
SELECT * FROM tenant;
```

#### User Management
```sql
-- Users within tenants with roles and preferences
SELECT * FROM user;
```

#### Document Metadata
```sql
-- Document metadata, processing status, categories
SELECT * FROM document;
```

#### Vector Chunk Tracking
```sql
-- Metadata for ChromaDB vector chunks
SELECT * FROM chunk;
```

#### Search Analytics
```sql
-- Query tracking and performance metrics
SELECT * FROM search_query;
```

#### Categorization Templates
```sql
-- Language-specific prompt templates
SELECT * FROM categorization_template;
```

#### System Configuration
```sql
-- Runtime settings and feature flags
SELECT * FROM system_config;
```

### Key Relationships

```
tenant (1) ←→ (N) user
tenant (1) ←→ (N) document
user (1) ←→ (N) document
document (1) ←→ (N) chunk
user (1) ←→ (N) search_query
```

### Indexes

The schema includes optimized indexes for:
- Tenant-scoped queries
- User access patterns
- Document categorization
- Language-based filtering
- Time-based analytics

## Common Queries

### Basic Information

```sql
-- Get database info
INFO FOR DATABASE;

-- Count all records
SELECT
  'tenant' AS table, count() FROM tenant
UNION
  'user' AS table, count() FROM user
UNION
  'document' AS table, count() FROM document;
```

### Tenant Queries

```sql
-- List all tenants with user counts
SELECT
  tenant.*,
  count(<-user.tenant_id) AS user_count
FROM tenant;

-- Get tenant by slug
SELECT * FROM tenant WHERE slug = 'development';

-- Active tenants only
SELECT * FROM tenant WHERE status = 'active';
```

### User Queries

```sql
-- Users with tenant information
SELECT
  user.*,
  tenant_id.name AS tenant_name,
  tenant_id.slug AS tenant_slug
FROM user;

-- Admin users across all tenants
SELECT
  username,
  full_name,
  tenant_id.name AS tenant_name
FROM user
WHERE role = 'admin';

-- Users by language preference
SELECT language_preference, count()
FROM user
GROUP BY language_preference;
```

### Document Queries

```sql
-- Documents with processing status
SELECT
  title,
  filename,
  status,
  language,
  chunk_count,
  user_id.username AS owner,
  tenant_id.name AS tenant_name
FROM document;

-- Documents by category
SELECT
  categories,
  count() AS doc_count
FROM document
WHERE categories != []
GROUP BY categories;

-- Recent documents
SELECT title, created_at
FROM document
ORDER BY created_at DESC
LIMIT 10;
```

### Analytics Queries

```sql
-- Search query statistics
SELECT
  query_language,
  count() AS total_queries,
  math::round(math::mean(response_time_ms)) AS avg_response_ms,
  math::round(math::mean(satisfaction_rating)) AS avg_rating
FROM search_query
GROUP BY query_language;

-- Popular categories
SELECT
  primary_category,
  count() AS query_count
FROM search_query
WHERE primary_category IS NOT NONE
GROUP BY primary_category
ORDER BY query_count DESC;

-- Daily query trends
SELECT
  time::format(created_at, '%Y-%m-%d') AS date,
  count() AS query_count
FROM search_query
GROUP BY date
ORDER BY date DESC;
```

### Categorization Queries

```sql
-- Available templates by language
SELECT
  language,
  category,
  count() AS template_count
FROM categorization_template
GROUP BY language, category;

-- System default templates
SELECT name, category, language
FROM categorization_template
WHERE is_system_default = true
ORDER BY priority DESC;
```

### System Configuration

```sql
-- All system settings
SELECT * FROM system_config
WHERE is_system_config = true;

-- Tenant-specific settings
SELECT * FROM system_config
WHERE tenant_id = tenant:development;
```

## Development Workflow

### 1. Start Development Environment

```bash
# Start file-based database
./scripts/setup_surrealdb.sh --file-mode

# Or start Docker environment
./scripts/setup_surrealdb.sh --docker
```

### 2. Test Queries

```bash
# Run all query tests
./scripts/test_surrealdb_queries.sh

# Test specific areas
./scripts/test_surrealdb_queries.sh tenants
./scripts/test_surrealdb_queries.sh users
./scripts/test_surrealdb_queries.sh documents

# Interactive mode
./scripts/test_surrealdb_queries.sh --interactive
```

### 3. Schema Updates

```sql
-- Add new field
ALTER TABLE document ADD COLUMN ai_summary string;

-- Create new index
DEFINE INDEX document_summary_idx ON TABLE document COLUMNS ai_summary;

-- Update data
UPDATE document SET ai_summary = 'Generated summary...' WHERE id = document:sample;
```

### 4. Backup and Restore

```bash
# Export database
surreal export --conn ws://127.0.0.1:8000 \
  --user root --pass root \
  --ns rag --db multitenant \
  backup.sql

# Import database
surreal import --conn ws://127.0.0.1:8000 \
  --user root --pass root \
  --ns rag --db multitenant \
  backup.sql
```

## Production Deployment

### Recommended Configuration

```bash
# Production server with authentication
surreal start \
  --bind 0.0.0.0:8000 \
  --user admin \
  --pass $SECURE_PASSWORD \
  --log info \
  rocksdb:///var/lib/surrealdb/rag
```

### Security Considerations

1. **Change Default Credentials**
   ```bash
   # Use strong passwords
   surreal start --user admin --pass $(openssl rand -base64 32)
   ```

2. **Network Security**
   ```bash
   # Bind to localhost only for local access
   surreal start --bind 127.0.0.1:8000

   # Use TLS in production
   surreal start --web-crt cert.pem --web-key key.pem
   ```

3. **Firewall Configuration**
   ```bash
   # Allow only specific IPs
   ufw allow from 10.0.0.0/8 to any port 8000
   ```

### Docker Production

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  surrealdb:
    image: surrealdb/surrealdb:v2.1.0
    restart: always
    environment:
      - SURREAL_USER=admin
      - SURREAL_PASS_FILE=/run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - surrealdb_data:/data
      - ./certs:/certs:ro
    command: >
      start
      --bind 0.0.0.0:8000
      --web-crt /certs/server.crt
      --web-key /certs/server.key
      rocksdb:///data
    ports:
      - "8000:8000"

volumes:
  surrealdb_data:

secrets:
  db_password:
    external: true
```

### Monitoring

```sql
-- Monitor query performance
SELECT
  table_name,
  avg_response_time,
  total_queries
FROM (
  SELECT
    'search_query' AS table_name,
    math::mean(response_time_ms) AS avg_response_time,
    count() AS total_queries
  FROM search_query
  WHERE created_at > time::now() - 1h
);

-- Monitor storage usage
INFO FOR DATABASE;
```

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to SurrealDB
```bash
# Check if service is running
ps aux | grep surreal

# Check port availability
netstat -tlnp | grep 8000

# Test connection
curl http://127.0.0.1:8000/health
```

**Solution**: Ensure SurrealDB is running and port is available

### Schema Issues

**Problem**: Schema not loaded properly
```sql
-- Check if tables exist
INFO FOR DATABASE;

-- Reload schema
-- (Exit and re-import schema file)
```

**Solution**: Re-run setup script with `--reset` flag

### Performance Issues

**Problem**: Slow queries
```sql
-- Check query execution plan
EXPLAIN SELECT * FROM document WHERE tenant_id = tenant:development;

-- Add missing indexes
DEFINE INDEX document_tenant_status_idx ON TABLE document COLUMNS [tenant_id, status];
```

### Docker Issues

**Problem**: Container not starting
```bash
# Check container logs
docker logs rag-surrealdb

# Check container status
docker ps -a

# Restart container
docker-compose -f docker-compose.surrealdb.yml restart
```

### Data Corruption

**Problem**: Database corruption
```bash
# Backup current state
surreal export --conn ws://127.0.0.1:8000 \
  --user root --pass root \
  --ns rag --db multitenant \
  corrupted_backup.sql

# Reset and restore
./scripts/setup_surrealdb.sh --reset
surreal import --conn ws://127.0.0.1:8000 \
  --user root --pass root \
  --ns rag --db multitenant \
  clean_backup.sql
```

### Log Analysis

```bash
# Enable debug logging
surreal start --log debug --bind 0.0.0.0:8000 file://./debug.db

# Monitor logs
tail -f surrealdb.log | grep ERROR
```

## Additional Resources

- [SurrealDB Documentation](https://surrealdb.com/docs)
- [SurrealQL Query Language](https://surrealdb.com/docs/surrealql)
- [SurrealDB Python SDK](https://surrealdb.com/docs/integration/libraries/python)
- [Multi-tenant RAG Architecture](./architecture.md)

## Support

For issues specific to this setup:
1. Check the troubleshooting section above
2. Run the diagnostic script: `./scripts/test_surrealdb_queries.sh`
3. Check logs and connection settings
4. Review the schema file for any customizations needed

For general SurrealDB issues:
- [SurrealDB GitHub Issues](https://github.com/surrealdb/surrealdb/issues)
- [SurrealDB Discord Community](https://discord.gg/surrealdb)