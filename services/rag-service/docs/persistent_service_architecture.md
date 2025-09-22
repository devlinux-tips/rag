# Persistent RAG Service Architecture

**Date**: 2025-09-19
**Status**: Design Phase
**Authors**: Architecture Investigation

## Overview

This document outlines the architecture for transforming the RAG system from a monolithic code structure into a persistent, service-oriented architecture that can handle HTTP API integration and scale to 50+ commands.

## Current Problems

- `rag_system.py` is over 1200 lines
- `query()` method alone is 200+ lines
- Mixed concerns (validation, retrieval, generation, parsing)
- No persistent service capability
- No web API integration
- Difficult to add new functionality
- No database provider integration

## Proposed Solution: Event-Driven Microservice Architecture

### Core Architecture Pattern

**Event-Driven Microservice + Command Bus + Database Layer Integration**

```
External Web API ──HTTP──> RAG Service Daemon (Port 8080)
                              │
                      ┌───────┴───────┐
                      │ FastAPI Server │
                      │ - HTTP endpoints│
                      │ - Request validation│
                      │ - Response formatting│
                      └───────┬───────┘
                              │
                      ┌───────┴───────┐
                      │ Command Router │
                      │ - Parse commands│
                      │ - Route to services│
                      │ - Handle sync/async│
                      └───────┬───────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│QueryService │    │TenantService│    │DocumentSvc  │
│- Real-time  │    │- CRUD ops   │    │- Background │
│- Sync ops   │    │- Config mgmt│    │- Async proc │
└─────────────┘    └─────────────┘    └─────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                ┌─────────────────────────┐
                │  Database Provider      │
                │  - Our Protocol!        │
                │  - SurrealDB/Supabase   │
                │  - ChromaDB (vectors)   │
                └─────────────────────────┘
```

## Service Lifecycle Management

### Commands

```bash
# Service management
python rag-service.py start    # Start daemon
python rag-service.py stop     # Graceful shutdown
python rag-service.py restart  # Stop + Start
python rag-service.py reload   # Hot reload config
python rag-service.py status   # Check if running
```

### Implementation

```python
# rag-service.py - Main entry point
class RAGServiceDaemon:
    def __init__(self):
        self.database_provider = None
        self.http_server = None
        self.command_router = None
        self.services = {}

    async def start(self):
        """Start persistent service"""
        # 1. Initialize database connections
        self.database_provider = create_database_provider()
        await self.database_provider.initialize()

        # 2. Initialize services with database integration
        self.services = {
            'query': QueryService(self.database_provider),
            'tenant': TenantService(self.database_provider),
            'document': DocumentService(self.database_provider),
            'health': HealthService(self.database_provider)
        }

        # 3. Create command router
        self.command_router = CommandRouter(self.services)

        # 4. Start HTTP server
        app = create_fastapi_app(self.command_router)
        self.http_server = uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=8080)
        )
        await self.http_server.serve()

    async def stop(self):
        """Graceful shutdown"""
        # 1. Stop accepting new requests
        # 2. Finish processing current requests
        # 3. Close database connections
        # 4. Save state if needed
        pass

    async def reload_config(self):
        """Hot reload configuration"""
        # 1. Load new configuration
        # 2. Update services with new config
        # 3. Restart components that need it
        pass
```

## HTTP API Integration Patterns

### Synchronous Operations
Immediate response for real-time operations like queries:

```python
# Web API → RAG Service
POST /api/v1/query
{
  "text": "What is RAG?",
  "language": "hr",
  "tenant_id": "acme",
  "user_id": "john"
}

# Immediate response
{
  "answer": "RAG je...",
  "confidence": 0.85,
  "sources": ["doc1.pdf", "doc2.txt"],
  "response_time": 1.2,
  "query_id": "query_123"
}
```

### Asynchronous Operations
Background processing for long-running operations:

```python
# Start background task
POST /api/v1/documents/process
{
  "documents": ["file1.pdf", "file2.docx"],
  "tenant_id": "acme"
}
→ Response: {"task_id": "task_123", "status": "started"}

# Check status
GET /api/v1/tasks/task_123/status
→ Response: {
    "status": "processing",
    "progress": "60%",
    "processed": 2,
    "total": 5,
    "eta_seconds": 45
}

# Get results
GET /api/v1/tasks/task_123/results
→ Response: {
    "status": "completed",
    "documents": [...],
    "chunks": 150,
    "processing_time": 23.4
}
```

### Management Operations
Admin operations for system management:

```python
# Create tenant
POST /api/v1/tenants
{
  "name": "Acme Corp",
  "slug": "acme",
  "config": {"language": "hr"}
}
→ Response: {
    "id": "tenant_456",
    "slug": "acme",
    "collections_created": 3,
    "status": "active"
}

# Update configuration
PUT /api/v1/config/reload
{
  "component": "embeddings",
  "restart_required": false
}
→ Response: {
    "status": "reloaded",
    "components_updated": ["embeddings"],
    "restart_required": false
}
```

## Command Router Architecture

```python
class CommandRouter:
    def __init__(self, services: dict):
        self.services = services
        self.commands = {
            # Query operations
            "query": self.services['query'].execute_query,
            "search": self.services['query'].execute_search,
            "suggest": self.services['query'].get_suggestions,
            "query_history": self.services['query'].get_user_history,

            # Tenant operations
            "create_tenant": self.services['tenant'].create_tenant,
            "update_tenant": self.services['tenant'].update_tenant,
            "delete_tenant": self.services['tenant'].delete_tenant,
            "list_tenants": self.services['tenant'].list_tenants,

            # Document operations
            "process_documents": self.services['document'].process_documents,
            "delete_documents": self.services['document'].delete_documents,
            "reindex_documents": self.services['document'].reindex_documents,
            "get_document_status": self.services['document'].get_status,

            # System operations
            "health_check": self.services['health'].check_health,
            "get_metrics": self.services['health'].get_metrics,
            "get_performance": self.services['health'].get_performance_stats,
        }

    async def route_command(self, command_name: str, request: dict):
        """Route command to appropriate service"""
        if command_name not in self.commands:
            raise CommandNotFoundError(f"Command '{command_name}' not supported")

        handler = self.commands[command_name]
        return await handler(request)

    def register_command(self, name: str, handler):
        """Dynamically register new commands"""
        self.commands[name] = handler

    def list_commands(self) -> list[str]:
        """List all available commands"""
        return list(self.commands.keys())
```

## Database Provider Integration

### Natural Integration Points

```python
class QueryService:
    def __init__(self, database_provider: DatabaseProvider):
        self.db = database_provider
        self.rag_system = None  # Injected separately

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        # 1. Store query for analytics
        query_record = await self.db.create_query(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            query_text=request.text,
            language=request.language,
            timestamp=datetime.now()
        )

        # 2. Execute RAG pipeline (existing logic)
        response = await self._process_rag_pipeline(request)

        # 3. Update query with results
        await self.db.update_query(
            query_record.id,
            response_text=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            response_time=response.total_time
        )

        return response

    async def get_user_history(self, user_id: str, limit: int = 10) -> list[QueryRecord]:
        """Get user's query history"""
        return await self.db.get_user_queries(user_id, limit)

class TenantService:
    def __init__(self, database_provider: DatabaseProvider):
        self.db = database_provider

    async def create_tenant(self, request: CreateTenantRequest) -> TenantResponse:
        # 1. Create tenant in database
        tenant = await self.db.create_tenant(
            name=request.name,
            slug=request.slug,
            config=request.config,
            status="active"
        )

        # 2. Initialize tenant-specific resources
        await self._setup_tenant_collections(tenant)
        await self._create_tenant_folders(tenant)
        await self._initialize_tenant_config(tenant)

        return TenantResponse(tenant=tenant)

    async def list_tenants(self, filters: dict = None) -> list[Tenant]:
        """List all tenants with optional filtering"""
        return await self.db.list_tenants(filters)

class DocumentService:
    def __init__(self, database_provider: DatabaseProvider):
        self.db = database_provider

    async def process_documents(self, request: ProcessDocumentsRequest) -> TaskResponse:
        # 1. Create processing task record
        task = await self.db.create_task(
            type="document_processing",
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            status="started",
            metadata={"document_count": len(request.documents)}
        )

        # 2. Start background processing
        asyncio.create_task(self._process_documents_background(task.id, request))

        return TaskResponse(task_id=task.id, status="started")

    async def get_document_metadata(self, document_id: str) -> DocumentMetadata:
        """Get rich document metadata from database"""
        return await self.db.get_document(document_id)
```

### Database Schema Extensions

The database provider will handle these new entities:

```sql
-- Query analytics
CREATE TABLE queries (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    query_text TEXT NOT NULL,
    language VARCHAR(10) NOT NULL,
    response_text TEXT,
    confidence FLOAT,
    sources JSONB,
    response_time FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Document metadata (beyond vectors)
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    filename VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    author VARCHAR(255),
    language VARCHAR(10),
    content_type VARCHAR(100),
    file_size BIGINT,
    processing_status VARCHAR(50),
    chunk_count INTEGER,
    upload_date TIMESTAMP DEFAULT NOW(),
    last_processed TIMESTAMP
);

-- Background tasks
CREATE TABLE tasks (
    id UUID PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    status VARCHAR(50) NOT NULL,
    progress FLOAT DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- System metrics
CREATE TABLE metrics (
    id UUID PRIMARY KEY,
    component VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

## Service Implementation Details

### QueryService (~200 lines)

```python
class QueryService:
    """Handles all query-related operations"""

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Main query execution with database integration"""
        pass

    async def execute_search(self, request: SearchRequest) -> SearchResponse:
        """Document search without generation"""
        pass

    async def get_suggestions(self, request: SuggestionRequest) -> SuggestionResponse:
        """Query suggestions based on history"""
        pass

    async def get_user_history(self, user_id: str, limit: int = 10) -> list[QueryRecord]:
        """User query history from database"""
        pass
```

### TenantService (~150 lines)

```python
class TenantService:
    """Handles tenant management operations"""

    async def create_tenant(self, request: CreateTenantRequest) -> TenantResponse:
        """Create new tenant with full setup"""
        pass

    async def update_tenant(self, tenant_id: str, updates: dict) -> TenantResponse:
        """Update tenant configuration"""
        pass

    async def delete_tenant(self, tenant_id: str) -> dict:
        """Soft delete tenant and cleanup resources"""
        pass

    async def list_tenants(self, filters: dict = None) -> list[Tenant]:
        """List tenants with filtering"""
        pass
```

### DocumentService (~200 lines)

```python
class DocumentService:
    """Handles document processing and management"""

    async def process_documents(self, request: ProcessDocumentsRequest) -> TaskResponse:
        """Start background document processing"""
        pass

    async def delete_documents(self, request: DeleteDocumentsRequest) -> TaskResponse:
        """Delete documents and their vectors"""
        pass

    async def reindex_documents(self, request: ReindexRequest) -> TaskResponse:
        """Reprocess documents with updated settings"""
        pass

    async def get_document_status(self, document_id: str) -> DocumentStatus:
        """Get processing status and metadata"""
        pass

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get background task status"""
        pass
```

### HealthService (~100 lines)

```python
class HealthService:
    """Handles monitoring and health checks"""

    async def check_health(self) -> HealthResponse:
        """Comprehensive system health check"""
        pass

    async def get_metrics(self, timeframe: str = "1h") -> MetricsResponse:
        """System performance metrics"""
        pass

    async def get_performance_stats(self) -> PerformanceResponse:
        """Detailed performance statistics"""
        pass
```

## Proposed File Structure

```
src/
├── service/
│   ├── rag_service_daemon.py      # Main service entry point
│   ├── http_server.py             # FastAPI application
│   ├── command_router.py          # Command routing logic
│   ├── lifecycle_manager.py       # Start/stop/reload logic
│   └── task_manager.py            # Background task handling
├── services/
│   ├── query_service.py           # Query processing (~200 lines)
│   ├── tenant_service.py          # Tenant management (~150 lines)
│   ├── document_service.py        # Document operations (~200 lines)
│   └── health_service.py          # Monitoring (~100 lines)
├── api/
│   ├── endpoints/
│   │   ├── query_endpoints.py     # /api/v1/query routes
│   │   ├── tenant_endpoints.py    # /api/v1/tenants routes
│   │   ├── document_endpoints.py  # /api/v1/documents routes
│   │   └── health_endpoints.py    # /api/v1/health routes
│   ├── models/
│   │   ├── request_models.py      # Pydantic request models
│   │   ├── response_models.py     # Pydantic response models
│   │   └── task_models.py         # Task and status models
│   └── middleware/
│       ├── auth_middleware.py     # Authentication
│       ├── logging_middleware.py  # Request logging
│       └── error_middleware.py    # Error handling
├── commands/
│   ├── base_command.py            # Base command interface
│   ├── command_registry.py        # Dynamic command registration
│   ├── query/
│   │   ├── simple_query_command.py
│   │   ├── complex_query_command.py
│   │   └── search_command.py
│   ├── tenant/
│   │   ├── create_tenant_command.py
│   │   └── manage_tenant_command.py
│   └── document/
│       ├── process_documents_command.py
│       └── manage_documents_command.py
└── pipeline/
    └── rag_system.py              # Thin orchestrator (~100 lines)
```

## Migration Strategy

### Phase 1: Extract Services (Week 1)
- Create service layer classes
- Move business logic from rag_system.py to services
- Maintain existing public API for backward compatibility
- Add basic database integration points

### Phase 2: HTTP Server (Week 2)
- Create FastAPI server with endpoint routing
- Implement command router pattern
- Add request/response models
- Test HTTP integration

### Phase 3: Service Lifecycle (Week 3)
- Implement start/stop/restart functionality
- Add configuration hot reload
- Create background task management
- Add health monitoring

### Phase 4: Full Integration (Week 4)
- Complete database provider integration
- Add comprehensive error handling
- Performance optimization
- Documentation and testing

### Phase 5: Command Expansion (Ongoing)
- Add new commands as needed
- Expand database schema
- Add analytics and reporting
- Scale to 50+ commands

## Benefits of This Architecture

### Code Organization
- **Small Files**: Each service ~100-200 lines vs 1200-line monolith
- **Single Responsibility**: Each service owns one domain
- **Easy Testing**: Mock individual services and database provider
- **Maintainable**: Clear separation of concerns

### Service Capabilities
- **Persistent**: Runs continuously, maintains connections
- **HTTP Native**: Direct web API integration
- **Async Support**: Background processing for long operations
- **Database Integrated**: Natural persistence layer
- **Configurable**: Hot reload without restart

### Scalability
- **Command Pattern**: Easy to add 50+ commands
- **Service Registration**: Dynamic command discovery
- **Database Abstraction**: Swap SurrealDB ↔ Supabase easily
- **Microservice Ready**: Can split into multiple services later

### Developer Experience
- **Clear APIs**: Well-defined request/response models
- **Professional Tooling**: start/stop/restart/reload commands
- **Comprehensive Monitoring**: Health checks and metrics
- **Database Integration**: Rich metadata and analytics

## Implementation Priority

1. **Critical**: Service lifecycle and HTTP server
2. **High**: QueryService with database integration
3. **High**: TenantService for multi-tenancy
4. **Medium**: DocumentService for background processing
5. **Medium**: HealthService for monitoring
6. **Low**: Command expansion and optimization

This architecture provides both the **code organization** needed to manage complexity and the **persistent service capabilities** required for web API integration, while naturally incorporating our database provider protocol.