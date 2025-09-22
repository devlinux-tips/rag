# Hybrid Elixir/Python Architecture for Intelligent RAG Platform

## Executive Summary

This document presents a comprehensive hybrid architecture that combines Elixir/OTP's distributed systems strengths with Python's ML/AI capabilities to create an intelligent, collaborative LLM chat platform over RAG. The design leverages the best of both ecosystems while maintaining clear separation of concerns and enabling phased implementation.

## Architecture Overview

### Core Philosophy
- **Elixir/OTP**: Orchestration, real-time collaboration, fault tolerance, multi-tenancy
- **Python**: Specialized ML/AI operations, RAG pipeline, embeddings, LLM inference
- **Hybrid Communication**: GenServer-based Python process supervision with JSON-RPC protocol

### High-Level System Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                    ELIXIR/OTP PLATFORM                        │
├─────────────────────────────────────────────────────────────────┤
│  Phoenix LiveView UI │  GenServer Supervisors │  Oban Jobs     │
│  Real-time Chat      │  Python Process Mgmt   │  Async Tasks   │
│  Multi-tenant Auth   │  Fault Recovery         │  Scheduling    │
└─────────────────────┬───────────────────────────────────────────┘
                      │ JSON-RPC over Unix Sockets/HTTP
┌─────────────────────┴───────────────────────────────────────────┐
│                    PYTHON RAG SERVICE                         │
├─────────────────────────────────────────────────────────────────┤
│  Command Bus        │  Service Layer          │  ML Pipeline   │
│  Query Processing   │  Document Management    │  Embeddings    │
│  Response Streaming │  Vector Operations      │  Generation    │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Component Architecture

### 1. Elixir/OTP Platform Layer

#### 1.1 Phoenix LiveView Frontend
```elixir
defmodule RagPlatform.ChatLive do
  use Phoenix.LiveView

  # Real-time collaborative chat interface
  def handle_event("send_message", %{"message" => text}, socket) do
    tenant_id = socket.assigns.tenant_id
    user_id = socket.assigns.user_id

    # Async RAG query via GenServer
    RagPlatform.RagSupervisor.query_async(tenant_id, user_id, text)

    {:noreply, assign(socket, :loading, true)}
  end

  # Real-time response streaming
  def handle_info({:rag_response_chunk, chunk}, socket) do
    {:noreply, push_event(socket, "append_response", %{chunk: chunk})}
  end
end
```

#### 1.2 RAG Process Supervision
```elixir
defmodule RagPlatform.RagSupervisor do
  use GenServer

  # Manages Python RAG service lifecycle
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    # Start Python RAG service as supervised port
    port = Port.open({:spawn, "python rag_service.py"},
                     [:binary, packet: 4, args: ["--mode", "service"]])

    {:ok, %{port: port, pending_queries: %{}}}
  end

  def query_async(tenant_id, user_id, query) do
    GenServer.cast(__MODULE__, {:query, tenant_id, user_id, query})
  end

  def handle_cast({:query, tenant_id, user_id, query}, state) do
    query_id = generate_query_id()

    command = %{
      id: query_id,
      command: "query",
      params: %{
        tenant: tenant_id,
        user: user_id,
        query: query,
        stream: true
      }
    }

    Port.command(state.port, Jason.encode!(command))

    pending = Map.put(state.pending_queries, query_id, {tenant_id, user_id})
    {:noreply, %{state | pending_queries: pending}}
  end

  # Handle streaming responses from Python
  def handle_info({port, {:data, data}}, %{port: port} = state) do
    response = Jason.decode!(data)

    case response do
      %{"type" => "chunk", "query_id" => query_id, "content" => chunk} ->
        broadcast_chunk(query_id, chunk, state)

      %{"type" => "complete", "query_id" => query_id} ->
        cleanup_query(query_id, state)

      %{"type" => "error", "query_id" => query_id, "error" => error} ->
        broadcast_error(query_id, error, state)
    end

    {:noreply, state}
  end
end
```

#### 1.3 Multi-Tenant Job Orchestration
```elixir
defmodule RagPlatform.Jobs.DocumentProcessor do
  use Oban.Worker, queue: :document_processing

  def perform(%Oban.Job{args: %{"tenant_id" => tenant_id, "document_path" => path}}) do
    # Queue Python document processing job
    command = %{
      command: "process_documents",
      params: %{
        tenant: tenant_id,
        documents: [path],
        async: true
      }
    }

    RagPlatform.RagSupervisor.execute_command(command)
    :ok
  end
end
```

### 2. Python RAG Service Layer

#### 2.1 Enhanced Service Architecture
```python
# src/services/rag_service.py
class RAGService:
    """Main RAG service with command bus pattern."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.command_bus = CommandBus()
        self.event_bus = EventBus()
        self._register_commands()

    def _register_commands(self):
        """Register all available commands."""
        self.command_bus.register("query", QueryService(self.config))
        self.command_bus.register("process_documents", DocumentService(self.config))
        self.command_bus.register("create_tenant", TenantService(self.config))
        self.command_bus.register("manage_collections", CollectionService(self.config))

    async def execute_command(self, command: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Execute command and yield streaming responses."""
        command_name = command["command"]
        params = command["params"]
        query_id = command.get("id")

        try:
            service = self.command_bus.get_handler(command_name)

            if params.get("stream", False):
                async for chunk in service.execute_streaming(params):
                    yield {
                        "type": "chunk",
                        "query_id": query_id,
                        "content": chunk
                    }

                yield {
                    "type": "complete",
                    "query_id": query_id
                }
            else:
                result = await service.execute(params)
                yield {
                    "type": "result",
                    "query_id": query_id,
                    "data": result
                }

        except Exception as e:
            yield {
                "type": "error",
                "query_id": query_id,
                "error": str(e)
            }
```

#### 2.2 Streaming Query Service
```python
# src/services/query_service.py
class QueryService:
    """Handles RAG query processing with streaming responses."""

    def __init__(self, config: Dict[str, Any]):
        self.retrieval_service = RetrievalService(config)
        self.generation_service = GenerationService(config)

    async def execute_streaming(self, params: Dict[str, Any]) -> AsyncIterator[str]:
        """Execute query with streaming response generation."""
        tenant = params["tenant"]
        user = params["user"]
        query = params["query"]
        language = params.get("language", "hr")

        # Retrieve relevant documents
        documents = await self.retrieval_service.retrieve(
            tenant=tenant,
            user=user,
            query=query,
            language=language
        )

        # Stream LLM response
        async for chunk in self.generation_service.generate_streaming(
            query=query,
            context=documents,
            language=language
        ):
            yield chunk
```

#### 2.3 Service Communication Protocol
```python
# src/services/service_daemon.py
class RAGServiceDaemon:
    """Service daemon for Elixir/Python communication."""

    def __init__(self):
        self.rag_service = None

    async def start_service_mode(self):
        """Start service in daemon mode for Elixir communication."""
        config = get_unified_config()
        self.rag_service = RAGService(config)

        # Listen on stdin for commands from Elixir port
        async for line in sys.stdin:
            try:
                command = json.loads(line.strip())

                async for response in self.rag_service.execute_command(command):
                    # Send response back to Elixir
                    print(json.dumps(response), flush=True)

            except Exception as e:
                error_response = {
                    "type": "error",
                    "error": str(e)
                }
                print(json.dumps(error_response), flush=True)
```

### 3. Phased Implementation Strategy

#### Phase 1: Foundation (Weeks 1-4)
**Python Side:**
- Refactor `rag_system.py` into service layer architecture
- Implement command bus pattern with basic commands (query, process_documents)
- Create service daemon with JSON-RPC communication
- Maintain CLI compatibility for testing

**Elixir Side:**
- Basic Phoenix application with multi-tenant authentication
- Simple GenServer for Python process supervision
- Basic LiveView chat interface
- Database schema for tenants, users, conversations

#### Phase 2: Real-time Integration (Weeks 5-8)
**Enhanced Features:**
- Streaming response integration between Elixir and Python
- Real-time collaborative chat with LiveView
- Document upload and processing workflows
- Basic error handling and process recovery

#### Phase 3: Production Readiness (Weeks 9-12)
**Advanced Capabilities:**
- Comprehensive Oban job system for async operations
- Advanced fault tolerance with OTP supervision trees
- Performance monitoring and metrics collection
- Production deployment with Docker orchestration

#### Phase 4: Platform Features (Weeks 13-16)
**Platform Completion:**
- Advanced multi-tenancy with resource isolation
- Plugin system for custom RAG behaviors
- Advanced analytics and conversation history
- API for third-party integrations

### 4. Technical Benefits of Hybrid Approach

#### 4.1 Elixir/OTP Strengths
- **Real-time**: Native WebSocket support through Phoenix Channels
- **Fault Tolerance**: OTP supervision trees automatically restart failed Python processes
- **Concurrency**: Handle thousands of concurrent chat sessions efficiently
- **Distributed**: Natural clustering and load balancing capabilities
- **Job Orchestration**: Oban provides robust background job processing

#### 4.2 Python ML/AI Strengths
- **Ecosystem**: Rich ML libraries (transformers, sentence-transformers, etc.)
- **Performance**: Optimized numerical computing with NumPy/PyTorch
- **Flexibility**: Easy integration with various LLM providers and embedding models
- **Existing Codebase**: Leverage all existing RAG pipeline work

#### 4.3 Hybrid Communication Benefits
- **Process Isolation**: Python crashes don't affect Elixir platform
- **Language Optimization**: Each component uses its optimal language
- **Scalability**: Python processes can be distributed across multiple nodes
- **Development Velocity**: Teams can work independently on each layer

### 5. Deployment Architecture

#### 5.1 Development Environment
```yaml
# docker-compose.yml
version: '3.8'
services:
  elixir_platform:
    build: ./elixir_platform
    ports:
      - "4000:4000"
    environment:
      - RAG_SERVICE_PATH=/app/python_service
    volumes:
      - ./python_service:/app/python_service

  python_rag:
    build: ./services/rag-service
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    command: python rag_service.py --mode service

  surrealdb:
    image: surrealdb/surrealdb:latest
    ports:
      - "8000:8000"
    command: start --log trace --user root --pass root memory
```

#### 5.2 Production Deployment
- **Elixir Releases**: Create OTP releases for zero-downtime deployments
- **Python Services**: Containerized RAG services with health checks
- **Load Balancing**: HAProxy/nginx for HTTP traffic, native Elixir clustering for WebSockets
- **Database**: Managed Supabase for metadata, self-hosted vector databases for embeddings

### 6. Migration Path from Current State

#### 6.1 Immediate Steps
1. **Refactor Python**: Extract services from `rag_system.py` using existing database protocols
2. **Create Elixir App**: Basic Phoenix application with tenant/user models
3. **Communication Layer**: JSON-RPC over stdin/stdout for initial integration
4. **Test Integration**: Verify command execution and response streaming

#### 6.2 Gradual Enhancement
1. **Replace CLI**: Gradually move commands from CLI to Elixir frontend
2. **Add Real-time**: Implement LiveView chat interface with streaming
3. **Background Jobs**: Move document processing to Oban background jobs
4. **Advanced Features**: Add collaboration, history, analytics

### 7. Risk Mitigation

#### 7.1 Technical Risks
- **Inter-process Communication**: Start with simple JSON-RPC, evolve to more sophisticated protocols if needed
- **Process Management**: Use OTP supervision trees to handle Python process failures gracefully
- **Data Consistency**: Maintain database transactions across language boundaries

#### 7.2 Development Risks
- **Team Coordination**: Clear API contracts between Elixir and Python teams
- **Testing Strategy**: Integration tests to verify cross-language communication
- **Performance**: Monitor and optimize inter-process communication overhead

## Conclusion

This hybrid architecture leverages the unique strengths of both Elixir/OTP and Python to create a powerful, scalable, and maintainable intelligent RAG platform. The phased approach allows for incremental development while building toward a sophisticated real-time collaborative system that can handle enterprise-scale multi-tenant deployments.

The combination of Elixir's fault-tolerant, real-time capabilities with Python's AI/ML ecosystem creates a foundation for building next-generation intelligent applications that can scale to millions of users while maintaining low latency and high availability.