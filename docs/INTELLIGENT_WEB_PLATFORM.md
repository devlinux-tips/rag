# Intelligent Web Data & Knowledge Platform

**Project Overview**: A distributed, multi-tenant platform combining web crawling, job orchestration, rate limiting, feature flags, and AI-powered RAG (Retrieval-Augmented Generation) for multilingual document intelligence.

**Created**: 2025-09-03 16:32:18 UTC
**Author**: xajler
**Stack**: Elixir/Phoenix + Python RAG + PostgreSQL + Vector DB

## Vision

Transform raw web data into intelligent, searchable knowledge through:
- **Respectful web crawling** at scale with politeness controls
- **Multi-tenant job orchestration** with workflow-driven processing
- **Distributed rate limiting** and feature flags for safe operations
- **Multilingual RAG vectorization** for semantic search and Q&A

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                Phoenix Web Interface + RAG Query UI         │
│ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌─────────────────┐ │
│ │ Crawler  │ │Workflow  │ │   RAG     │ │    Tenant       │ │
│ │ Control  │ │ Monitor  │ │  Search   │ │    Admin        │ │
│ └──────────┘ └──────────┘ └───────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (Phoenix)                      │
│  /crawl  /workflows  /search  /vectorize  /limits  /flags   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│            Elixir Control & Orchestration Plane             │
│ ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
│ │   Feature   │ │     Rate     │ │      Scheduler           ││
│ │    Flags    │ │   Limiter    │ │      Leader              ││
│ │             │ │              │ │                          ││
│ └─────────────┘ └──────────────┘ └──────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Work Execution Layer                         │
│ ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
│ │   Crawler   │ │   Workflow   │ │     RAG Integration      ││
│ │   Workers   │ │   Workers    │ │     Workers              ││
│ │             │ │              │ │                          ││
│ └─────────────┘ └──────────────┘ └──────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                         │                     │
           ┌─────────────────────────┐        │
           │                         │        │
┌─────────────────────────────────────────────────────────────┐
│                 Python RAG Service                          │
│ ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
│ │  Document   │ │  Embedding   │ │    Vector Store          ││
│ │ Processing  │ │   Models     │ │   (ChromaDB/Qdrant)      ││
│ │             │ │              │ │                          ││
│ └─────────────┘ └──────────────┘ └──────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Storage Layer                                  │
│ Jobs │ Crawl Data │ Workflows │ Limits │ Flags │ Embeddings │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Multi-tenant Scheduler + Queue
**Role**: Central orchestration engine for all time-based and background work

**Key Features**:
- Per-tenant fairness and isolation using token buckets
- Leader election via Postgres advisory locks
- Oban for durable job persistence with retries
- Quantum for cron-like scheduling
- Dead letter queue handling

**Implementation**:
```elixir
defmodule MyApp.TenantLimiter do
  use GenServer

  def try_acquire(tenant_id) do
    # Token bucket per tenant
    :ets.update_counter(@table, tenant_id, {:in_flight, 1})
    |> case do
      new when new <= get_max(tenant_id) -> :ok
      _ -> release(tenant_id); :busy
    end
  end
end
```

### 2. Web Crawler with Politeness
**Role**: Respectful, scalable web data collection

**Key Features**:
- robots.txt compliance and per-host rate limiting
- Content deduplication using ETS + Bloom filters
- GenStage/Broadway for backpressure management
- Frontier queues for fair host scheduling

**Implementation**:
```elixir
defmodule Crawler.PolitenessManager do
  # Per-host rate limiting
  def can_fetch?(host) do
    case :ets.lookup(@rate_table, host) do
      [{^host, last_fetch}] ->
        Time.diff(Time.utc_now(), last_fetch) >= get_delay(host)
      [] -> true
    end
  end
end
```

### 3. Distributed Rate Limiter + Feature Flags
**Role**: Control plane for safe operations and gradual rollouts

**Key Features**:
- Token bucket rate limiting with tenant isolation
- libcluster for distributed coordination
- Feature flags with cache + PubSub invalidation
- Circuit breakers for external API protection

**Implementation**:
```elixir
defmodule RateLimiter.TokenBucket do
  def allow?(key, tenant_id) do
    bucket = get_or_create_bucket(key, tenant_id)
    case bucket.tokens > 0 do
      true ->
        update_bucket(bucket, -1)
        :allow
      false -> :deny
    end
  end
end
```

### 4. Distributed Job Orchestration
**Role**: Multi-step workflow engine with fault tolerance

**Key Features**:
- gen_statem for workflow state machines
- Durable state persistence in Postgres
- Compensation workflows (sagas) for rollback
- Workflow versioning and deterministic execution

**Implementation**:
```elixir
defmodule Workflow.DocumentProcessor do
  use :gen_statem

  def callback_mode(), do: :state_functions

  def crawling({:call, from}, :next, data) do
    case crawl_documents(data.urls) do
      {:ok, documents} ->
        {:next_state, :extracting, %{data | documents: documents},
         [{:reply, from, :ok}]}
      {:error, reason} ->
        {:next_state, :failed, %{data | error: reason}}
    end
  end
end
```

### 5. Python RAG Service
**Role**: AI-powered knowledge extraction and semantic search

**Key Features**:
- Multilingual document vectorization
- ChromaDB/Qdrant for vector storage
- FastAPI for HTTP integration with Elixir
- Language detection and content chunking

**Implementation**:
```python
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@app.post("/vectorize")
async def vectorize_documents(request: VectorizeRequest):
    # Detect language
    language = detect_language(request.text)

    # Chunk documents
    chunks = chunk_text(request.text, language)

    # Generate embeddings
    embeddings = model.encode(chunks)

    # Store in vector DB
    await store_embeddings(embeddings, chunks)

    return {"status": "completed", "chunks": len(chunks)}
```

## Integration Patterns

### Elixir ↔ Python Communication

**Option 1: HTTP API (Recommended)**
```elixir
defmodule RagIntegration.VectorizeWorker do
  use Oban.Worker

  def perform(%{args: %{"documents" => docs, "language" => lang, "tenant_id" => tenant}}) do
    case RateLimiter.acquire("rag_api", tenant) do
      :ok ->
        HTTPoison.post("http://rag-service:8000/vectorize",
          Jason.encode!(%{documents: docs, language: lang}))
      :rate_limited ->
        {:snooze, 30_000}
    end
  end
end
```

**Option 2: Message Queue**
```elixir
# Publish to RabbitMQ/Redis
Broadway.Message.new(%{
  documents: extracted_docs,
  language: detected_lang,
  tenant_id: tenant_id,
  callback_url: "http://elixir-app/vectorization/complete"
})
```

## Data Model

```sql
-- Tenants and limits
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    plan VARCHAR(50),
    max_concurrency INTEGER DEFAULT 10,
    daily_quota INTEGER DEFAULT 10000,
    status VARCHAR(20) DEFAULT 'active'
);

-- Crawl schedules
CREATE TABLE schedules (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    cron_expression VARCHAR(100),
    job_kind VARCHAR(100),
    args JSONB,
    enabled BOOLEAN DEFAULT true
);

-- Documents and content
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    source_url TEXT,
    content TEXT,
    language VARCHAR(10),
    content_hash VARCHAR(64),
    created_at TIMESTAMP,
    vectorized_at TIMESTAMP
);

-- Vector embeddings
CREATE TABLE embeddings (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    model_name VARCHAR(100),
    embedding VECTOR(1536),
    chunk_index INTEGER,
    chunk_text TEXT
);

-- RAG processing jobs
CREATE TABLE rag_jobs (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    document_ids UUID[],
    language VARCHAR(10),
    model_name VARCHAR(100),
    status VARCHAR(20), -- pending, processing, completed, failed
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

## Workflow Examples

### Multilingual Document Processing Pipeline
```elixir
defmodule Workflows.MultilingualDocumentPipeline do
  use RagPlatform.Workflow

  workflow "multilingual_doc_processing" do
    step :crawl_documents, timeout: 300_000
    step :extract_text_content
    step :detect_language
    step :translate_if_needed, if: &needs_translation?/1
    step :chunk_documents
    step :vectorize_chunks, retries: 3
    step :store_embeddings
    step :update_search_index
    step :notify_completion

    compensation :cleanup_failed_vectorization
  end

  def vectorize_chunks(%{chunks: chunks, language: lang, tenant_id: tenant}) do
    case RateLimiter.acquire("vectorization", tenant) do
      :ok ->
        PythonRag.vectorize(chunks, lang)
      :denied ->
        {:error, :rate_limited}
    end
  end
end
```

## Real-world Use Cases

### 1. E-commerce Price Intelligence
- **Crawler**: Respectfully fetch competitor product pages
- **Scheduler**: "Crawl competitor sites every 2 hours"
- **Orchestrator**: "crawl → extract prices → detect changes → notify"
- **RAG**: "What are pricing trends for product X across competitors?"

### 2. Legal Document Intelligence
- **Crawler**: Collect legal documents from courts, government sites
- **Orchestrator**: "crawl → extract PDF text → detect language → vectorize"
- **RAG**: Search across documents in multiple languages, answer legal questions
- **Rate limiter**: Respect court website limits, manage OpenAI API quotas

### 3. Technical Documentation Knowledge Base
- **Crawler**: Monitor GitHub repos, documentation sites, Stack Overflow
- **Orchestrator**: "crawl → extract code + docs → multilingual vectorization"
- **RAG**: Answer technical questions across languages, find similar issues
- **Feature flags**: Test new code extraction algorithms

## Project Structure

```
intelligent_web_platform/
├── apps/
│   ├── web_api/             # Phoenix API + LiveView UI
│   │   ├── lib/
│   │   │   ├── web_api_web/
│   │   │   │   ├── live/    # LiveView components
│   │   │   │   └── controllers/
│   │   │   └── web_api/
│   │   └── priv/
│   ├── scheduler/           # Multi-tenant scheduler + queue
│   │   ├── lib/
│   │   │   ├── oban_workers/
│   │   │   ├── quantum_schedulers/
│   │   │   └── tenant_limiter.ex
│   │   └── test/
│   ├── crawler/             # Web crawler with politeness
│   │   ├── lib/
│   │   │   ├── crawler/
│   │   │   │   ├── politeness.ex
│   │   │   │   ├── frontier.ex
│   │   │   │   └── workers/
│   │   │   └── broadway_pipelines/
│   │   └── test/
│   ├── rate_limiter/        # Distributed rate limiting
│   │   ├── lib/
│   │   │   ├── token_bucket.ex
│   │   │   ├── feature_flags.ex
│   │   │   └── cluster_sync.ex
│   │   └── test/
│   ├── orchestrator/        # Workflow engine
│   │   ├── lib/
│   │   │   ├── workflows/
│   │   │   ├── state_machines/
│   │   │   └── compensation.ex
│   │   └── test/
│   ├── rag_integration/     # Elixir ↔ Python RAG bridge
│   │   ├── lib/
│   │   │   ├── http_client.ex
│   │   │   ├── message_queue.ex
│   │   │   └── workers/
│   │   └── test/
│   └── shared/              # Common schemas, telemetry
│       ├── lib/
│       │   ├── schemas/
│       │   ├── telemetry/
│       │   └── utils/
│       └── test/
├── python_rag/              # Python RAG service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app
│   │   ├── models/
│   │   │   ├── multilingual.py
│   │   │   └── document.py
│   │   ├── vectorstore/
│   │   │   ├── chroma.py
│   │   │   └── qdrant.py
│   │   ├── routes/
│   │   │   ├── vectorize.py
│   │   │   └── search.py
│   │   └── utils/
│   │       ├── language.py
│   │       └── chunking.py
│   ├── Dockerfile
│   └── requirements.txt
├── config/
│   ├── config.exs
│   ├── dev.exs
│   ├── prod.exs
│   └── test.exs
├── docker-compose.yml       # Full stack development
├── k8s/                     # Kubernetes manifests
│   ├── elixir-app.yaml
│   ├── python-rag.yaml
│   ├── postgres.yaml
│   └── vector-db.yaml
├── mix.exs
└── README.md
```

## Implementation Roadmap

### Phase 1: Foundation + RAG MVP (6-8 weeks)
- [ ] Multi-tenant scheduler with Oban + Quantum
- [ ] Basic web crawler with robots.txt compliance
- [ ] Simple rate limiter + feature flags
- [ ] Python RAG service with basic vectorization
- [ ] HTTP integration between Elixir and Python
- [ ] Phoenix LiveView admin interface

### Phase 2: Intelligent Workflows (6-8 weeks)
- [ ] Workflow orchestration with gen_statem
- [ ] Multilingual document processing pipelines
- [ ] Advanced rate limiting for ML APIs
- [ ] Search interface in Phoenix LiveView
- [ ] Per-tenant quotas and billing metrics

### Phase 3: Production RAG (6-8 weeks)
- [ ] Distributed deployment with libcluster
- [ ] Vector database clustering/sharding
- [ ] Advanced chunking and embedding strategies
- [ ] Real-time RAG query performance optimization
- [ ] Comprehensive observability and alerting

### Phase 4: Advanced Intelligence (6-8 weeks)
- [ ] Knowledge graph integration
- [ ] Cross-language semantic search
- [ ] Automated content classification
- [ ] Advanced analytics dashboard
- [ ] ML model A/B testing framework

## Development Setup

### Prerequisites
- Elixir 1.15+ with OTP 26+
- PostgreSQL 15+ with pgvector extension
- Python 3.11+ with FastAPI
- Docker and Docker Compose
- Redis (optional for message queuing)

### Quick Start
```bash
# Clone and setup Elixir app
git clone <repo>
cd intelligent_web_platform
mix deps.get
mix ecto.setup

# Setup Python RAG service
cd python_rag
pip install -r requirements.txt
python -m app.main

# Start full stack with Docker
docker-compose up -d

# Run the platform
mix phx.server
```

### Environment Variables
```bash
# Elixir
DATABASE_URL=postgres://user:pass@localhost/platform_dev
SECRET_KEY_BASE=<generated_secret>
PHX_HOST=localhost
PORT=4000

# Python RAG
OPENAI_API_KEY=<your_openai_key>
VECTOR_DB_URL=http://localhost:8000
MODEL_CACHE_DIR=/app/models

# Optional
REDIS_URL=redis://localhost:6379
SENTRY_DSN=<your_sentry_dsn>
```

## Monitoring and Observability

### Key Metrics
- **Queue health**: depth, processing rate, retry rate per tenant
- **Crawler metrics**: pages/sec, robots.txt violations, error rates
- **Rate limiter**: requests/sec, denials, token bucket state
- **RAG performance**: vectorization latency, search accuracy, model costs
- **Workflow success**: completion rate, compensation triggers, step latency

### Telemetry Integration
```elixir
# Example telemetry events
:telemetry.execute([:crawler, :page, :fetched], %{duration: duration},
  %{tenant_id: tenant, url: url, status: status})

:telemetry.execute([:rag, :vectorization, :completed], %{chunks: count},
  %{tenant_id: tenant, language: lang, model: model})

:telemetry.execute([:rate_limiter, :request], %{tokens: remaining},
  %{tenant_id: tenant, key: key, result: :allow})
```

## Security Considerations

### Multi-tenancy Isolation
- Row-level security in Postgres
- Tenant-scoped API keys and authentication
- Resource quotas and rate limiting per tenant
- Audit logging for all tenant actions

### External API Safety
- Circuit breakers for OpenAI/external APIs
- Request signing and verification
- Input validation and sanitization
- Secrets management with encrypted storage

### Crawler Ethics
- Strict robots.txt compliance
- Configurable politeness delays
- User-agent identification
- Respect for copyright and terms of service

## Cost Management

### Resource Optimization
- Intelligent caching of embeddings and search results
- Model selection based on tenant plan (local vs cloud)
- Batch processing for cost-effective API usage
- Auto-scaling based on queue depth and load

### Billing Integration
- Track API costs per tenant and operation
- Usage-based pricing with quota enforcement
- Cost allocation across different model tiers
- Detailed usage reporting and analytics

## Minimum Requirements (Development/Small Scale)

**CPU:** 4-6 cores (8+ threads)
- Local LLM inference is CPU-intensive
- Multiple services (Elixir, Python/FastAPI, Chroma/Weaviate, PostgreSQL, Redis)

**RAM:** 16-24GB
- Local LLM: 4-8GB (depending on model size - 7B models ~4GB, 13B+ models 8GB+)
- Chroma/Weaviate: 2-4GB for vector storage and indexing
- PostgreSQL: 2-4GB
- Elixir/OTP: 1-2GB
- Python/FastAPI: 1-2GB
- Redis: 512MB-1GB
- OS overhead: 2-4GB

**Storage:** 100GB+ SSD
- Vector embeddings can be storage-heavy
- Local LLM models: 4-20GB depending on model
- Database storage growth
- Docker images and logs

## Production-Ready Requirements

**CPU:** 8-12 cores
**RAM:** 32-64GB
**Storage:** 500GB+ NVMe SSD

## Hosting Options by Budget:

**Local Development:**
- High-spec workstation or gaming PC
- Mac Studio/MacBook Pro M2+ (excellent for LLM inference)

**VPS Options:**
- **Hetzner dedicated servers:** €39-89/month for decent specs
- **DigitalOcean:** $80-160/month for CPU-optimized droplets
- **AWS/GCP:** $100-300/month (more expensive but better managed services)
- **OVH/Scaleway:** Often cheaper in EU

**Hybrid Approach:**
- Host lightweight services (Elixir, FastAPI, PostgreSQL, Redis) on cheaper VPS ($20-40/month)
- Keep local LLM and vector DB on powerful local machine
- Connect via secure tunnel (Tailscale, WireGuard)

The local LLM is your biggest resource constraint. Consider starting with smaller models (7B parameters) and scaling up based on performance needs. Would you be open to using cloud LLM APIs initially to reduce hardware requirements?

---

**Next Steps**: Choose your starting phase and begin with the multi-tenant scheduler foundation, then incrementally add crawler, RAG integration, and advanced workflow capabilities.

**Contact**: xajler
**Last Updated**: 2025-09-03 16:32:18 UTC
