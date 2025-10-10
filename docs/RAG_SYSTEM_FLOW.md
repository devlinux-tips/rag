# RAG System Architecture Flow

## Quick Overview - Complete Stack

```
👤 User Browser
    ↓
🔀 Nginx (Port 80) - Reverse Proxy
    ↓
🖥️ Web UI (Port 5173) - React Frontend
    ↓
🌐 Web API (Port 3000) - Express/Node.js
    ├─→ ⚡ Redis (Port 6379) - Cache
    ├─→ 💾 PostgreSQL (Port 5434) - Database
    └─→ ⚡ RAG API (Port 8082) - FastAPI
         └─→ 🧠 RAG Service - Core Engine
              ├─→ 🗄️ Weaviate (Port 8080/50051) - Vectors
              └─→ 🤖 OpenRouter - LLM (Qwen 3)
```

**7 SystemD Services** | **606K+ Documents** | **3-6s Query Time** | **Native Ubuntu 24.04**

### Deployment Model

**Current Setup**: Native SystemD services on Ubuntu 24.04 LTS
- No Docker containers - all services run directly on the host
- Managed via `manage-services.sh` and systemd
- Optimized for powerful servers (243GB RAM, 144 CPU cores)
- Installation via `setup-local-server.sh`
- Service control via `Makefile` commands

**Alternative**: Docker Compose (see docker-compose.yml)

---

## Complete End-to-End Flow Diagram

```mermaid
graph TB
    %% User Layer
    User[👤 User/Client Browser]

    %% Reverse Proxy
    Nginx[🔀 Nginx Reverse Proxy<br/>Port: 80<br/>Load balancer & routing]

    %% UI Layer
    UI[🖥️ Web UI<br/>React/Vite<br/>Port: 5173<br/>services/web-ui/]

    %% Web API Layer
    WebAPI[🌐 Web API<br/>Node.js/Express<br/>Port: 3000<br/>services/web-api/<br/>Auth, Sessions, API Gateway]

    %% RAG Service API Layer
    RAGAPI[⚡ RAG Service API<br/>Python/FastAPI<br/>Port: 8082<br/>services/rag-api/<br/>RAG query processing]

    %% RAG System Core
    RAGService[🧠 RAG Service<br/>Python Core<br/>services/rag-service/<br/>CLI & Future Service]

    %% Cache Layer
    Redis[(⚡ Redis<br/>Port: 6379<br/>Real-time cache<br/>Sessions, Rate limiting)]

    %% Query Processing
    QueryProcessor[📝 Query Processor<br/>- Language detection<br/>- Query expansion<br/>- Keyword extraction]

    %% Embedding Generation
    EmbedGen[🔢 Embedding Generator<br/>- Croatian: classla/bcms-bertic 768d<br/>- English: BAAI/bge-large-en-v1.5 1024d<br/>- Fallback: BAAI/bge-m3 1024d]

    %% Vector Database
    Weaviate[(🗄️ Weaviate Vector DB<br/>Port: 8080/50051<br/>606K+ documents)]

    %% Retrieval Components
    VectorSearch[🔍 Vector Search<br/>Cosine Similarity<br/>HNSW Index]

    HybridRetriever[🔄 Hybrid Retriever<br/>- Dense retrieval<br/>- Keyword matching<br/>- Score fusion]

    Reranker[📊 Reranker<br/>- Cross-encoder scoring<br/>- Relevance optimization<br/>- Top-K selection]

    %% Context Preparation
    ContextBuilder[📋 Context Builder<br/>- Chunk aggregation<br/>- Metadata enrichment<br/>- Token counting]

    %% LLM Generation
    PromptBuilder[📄 Prompt Builder<br/>- Template selection<br/>- System prompts<br/>- Context injection]

    LLM[🤖 LLM Service<br/>OpenRouter Qwen 3<br/>qwen3-30b model<br/>Ollama local option]

    ResponseParser[✅ Response Parser<br/>- Answer extraction<br/>- Source attribution<br/>- Confidence scoring]

    %% Token Tracking
    TokenTracker[📈 Token Tracker<br/>- Input tokens<br/>- Output tokens<br/>- Total usage<br/>Stored in DB]

    %% Database
    PostgreSQL[(💾 PostgreSQL<br/>- Conversations<br/>- Messages<br/>- Token usage<br/>- User sessions)]

    %% Document Ingestion Flow (Side flow)
    DocUpload[📤 Document Upload]
    Extractor[📄 Document Extractor<br/>- PDF: PyPDF2<br/>- HTML: BeautifulSoup<br/>- DOCX: python-docx<br/>- TXT: direct]

    Chunker[✂️ Chunker<br/>- Size: 512 chars<br/>- Overlap: 50 chars<br/>- Semantic boundaries]

    EmbedDocs[🔢 Embed Chunks]

    %% User Flow
    User -->|HTTP/HTTPS| Nginx
    Nginx -->|Route /| UI
    Nginx -->|Route /api| WebAPI
    UI -->|API Calls| WebAPI
    WebAPI <-->|Session/Cache| Redis
    WebAPI -->|RAG Query| RAGAPI
    RAGAPI -->|Process| RAGService

    %% RAG Query Flow
    RAGService -->|1. Process Query| QueryProcessor
    QueryProcessor -->|2. Generate Embedding| EmbedGen
    EmbedGen -->|3. Search Vectors| VectorSearch
    VectorSearch -->|Query| Weaviate
    Weaviate -->|Top 20 Results| VectorSearch
    VectorSearch -->|4. Hybrid Retrieval| HybridRetriever
    HybridRetriever -->|5. Rerank| Reranker
    Reranker -->|Top 5 Chunks| ContextBuilder
    ContextBuilder -->|6. Build Context| PromptBuilder
    PromptBuilder -->|7. Generate| LLM
    LLM -->|LLM Response| ResponseParser
    ResponseParser -->|8. Parse & Track| TokenTracker
    TokenTracker -->|Store metrics| PostgreSQL
    ResponseParser -->|Final Answer| RAGService
    RAGService -->|Response| RAGAPI
    RAGAPI -->|JSON Response| WebAPI
    WebAPI -->|Display| UI
    UI -->|Render| Nginx
    Nginx -->|HTTP Response| User

    %% Document Ingestion Flow
    User -->|Upload Docs| DocUpload
    DocUpload --> Extractor
    Extractor -->|Text extraction| Chunker
    Chunker -->|Chunks| EmbedDocs
    EmbedDocs -->|Generate embeddings| EmbedGen
    EmbedGen -->|Store vectors| Weaviate

    %% Styling - Dark text for better contrast
    classDef userLayer fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
    classDef proxyLayer fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000
    classDef uiLayer fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    classDef apiLayer fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000
    classDef ragLayer fill:#e8eaf6,stroke:#283593,stroke-width:2px,color:#000
    classDef dbLayer fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef cacheLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef llmLayer fill:#ffe0b2,stroke:#bf360c,stroke-width:2px,color:#000
    classDef processingLayer fill:#e0f2f1,stroke:#004d40,stroke-width:2px,color:#000

    class User userLayer
    class Nginx proxyLayer
    class UI uiLayer
    class WebAPI,RAGAPI apiLayer
    class RAGService,QueryProcessor,ContextBuilder,PromptBuilder,ResponseParser,TokenTracker ragLayer
    class Weaviate,PostgreSQL dbLayer
    class Redis cacheLayer
    class LLM,EmbedGen llmLayer
    class VectorSearch,HybridRetriever,Reranker,Extractor,Chunker,EmbedDocs processingLayer
```

## Complete Services Overview

### Infrastructure Services

#### 1. Nginx Reverse Proxy (Port 80)
- **Technology**: Nginx Alpine
- **Purpose**: Load balancer, reverse proxy, SSL termination
- **Routes**:
  - `/` → Web UI (Port 5173)
  - `/api/*` → Web API (Port 3000)
- **Features**: Health checks, request routing, static file serving

#### 2. Redis Cache (Port 6379)
- **Technology**: Redis 7 Alpine
- **Purpose**: Session storage, rate limiting, real-time cache
- **Features**:
  - Session persistence
  - API rate limiting
  - Real-time data caching
  - Pub/Sub for real-time features

### Application Services

#### 3. Web UI (Port 5173)
- **Technology**: React + Vite
- **Location**: `services/web-ui/`
- **Purpose**: User-facing frontend
- **Features**:
  - Chat interface
  - Document management
  - User authentication UI
  - Real-time query responses

#### 4. Web API (Port 3000)
- **Technology**: Node.js + Express + TypeScript
- **Location**: `services/web-api/`
- **Purpose**: API Gateway & Authentication
- **Features**:
  - JWT authentication
  - User session management
  - Request routing to RAG API
  - Rate limiting
  - CORS handling
- **Endpoints**:
  - `POST /api/v1/auth/login` - User login
  - `POST /api/v1/auth/register` - User registration
  - `POST /api/v1/chat/query` - Send query (proxies to RAG API)
  - `GET /api/v1/conversations` - List conversations
  - `GET /api/v1/health` - Health check

#### 5. RAG Service API (Port 8082)
- **Technology**: Python + FastAPI
- **Location**: `services/rag-api/`
- **Purpose**: RAG query processing API
- **Features**:
  - Query processing endpoint
  - Document ingestion
  - Vector search orchestration
  - Token tracking
- **Endpoints**:
  - `POST /query` - Process RAG queries
  - `POST /documents` - Upload documents
  - `GET /collections` - List vector collections
  - `GET /health` - Health check

#### 6. RAG Service (CLI/Future Service)
- **Technology**: Python 3.12
- **Location**: `services/rag-service/`
- **Current**: CLI tool (`rag.py`)
- **Future**: Background service
- **Purpose**: Core RAG processing engine
- **Features**:
  - Document processing pipeline
  - Embedding generation
  - Vector database operations
  - Query processing
  - LLM integration
- **CLI Commands**:
  ```bash
  python rag.py --language hr query "Što je RAG?"
  python rag.py --language hr process-docs data/features/narodne_novine/
  python rag.py --language hr status
  python rag.py --language hr list-collections
  ```

### Data Services

#### 7. PostgreSQL Database (Port 5434 → 5432)
- **Technology**: PostgreSQL 16 Alpine
- **Purpose**: Application data storage
- **Schema**:
  - `users` - User accounts
  - `tenants` - Multi-tenant data
  - `conversations` - Chat sessions
  - `messages` - Individual messages with token counts
  - `documents` - Document metadata
- **Connection**: `postgresql://raguser:***@postgres:5432/ragdb`

#### 8. Weaviate Vector Database (Port 8080, 50051)
- **Technology**: Weaviate 1.27.0
- **Purpose**: Vector storage & similarity search
- **Collections**:
  - User scope: `{tenant}_{user}_{lang}_documents`
  - Feature scope: `Features_{feature_name}_{lang}`
  - Example: `Features_narodne_novine_hr` (606,380 chunks)
- **Configuration**:
  - HNSW index (ef_construction=200, ef=100)
  - Scalar quantization compression
  - 200GB RAM allocation
  - 140 CPU cores
- **Features**:
  - Cosine similarity search
  - gRPC for high performance
  - Automatic schema management
  - Multi-tenant collections

### External Services

#### 9. OpenRouter LLM API
- **Technology**: OpenRouter API
- **Purpose**: Primary LLM for text generation
- **Model**: `qwen/qwen3-30b-a3b-instruct-2507`
- **API**: https://openrouter.ai/api/v1
- **Features**: Multilingual, instruction-following

#### 10. Ollama (Optional Local LLM)
- **Technology**: Ollama
- **Purpose**: Local LLM alternative
- **Model**: `qwen2.5:7b-instruct`
- **Port**: 11434
- **Status**: Available but not default

## Detailed Component Breakdown

### 4. RAG Core Processing Pipeline

#### A. Query Processing
```
Query → QueryProcessor
├── Language detection
├── Query expansion (synonyms)
├── Keyword extraction
└── Query optimization
```

#### B. Embedding Generation
```
Text → EmbedGenerator
├── Croatian: classla/bcms-bertic (768-dim ELECTRA)
├── English: BAAI/bge-large-en-v1.5 (1024-dim BGE)
└── Fallback: BAAI/bge-m3 (1024-dim multilingual)
```

#### C. Vector Search
```
Query Embedding → Weaviate
├── Collection selection (tenant_user_lang)
├── HNSW index search
├── Cosine similarity
└── Top-K candidates (default: 20)
```

#### D. Retrieval & Reranking
```
Initial Results → HybridRetriever → Reranker
├── Dense vector search (0.7 weight)
├── Keyword BM25 search (0.3 weight)
├── Score fusion
├── Cross-encoder reranking
└── Top-K final results (default: 5)
```

#### E. Context Preparation
```
Retrieved Chunks → ContextBuilder
├── Aggregate chunk content
├── Add metadata (source, chunk_id)
├── Count tokens
└── Truncate if needed (max: 2500 tokens)
```

#### F. LLM Generation
```
Context + Query → PromptBuilder → OpenRouter/Ollama
├── Load system prompt template
├── Inject context chunks
├── Add user query
├── Send to OpenRouter (qwen3-30b) or Ollama (qwen2.5:7b local)
└── Stream or batch response
```

#### G. Response Processing
```
LLM Output → ResponseParser
├── Extract answer
├── Parse source citations
├── Calculate confidence
├── Track token usage (input/output)
└── Store metrics in PostgreSQL
```

### 5. Vector Database (Weaviate)
- **Storage**: Cloud or local instance
- **Port**: 8080 (HTTP), 50051 (gRPC)
- **Collections**:
  - User scope: `{tenant}_{user}_{language}_documents`
  - Feature scope: `Features_{feature_name}_{language}`
- **Index**: HNSW (Hierarchical Navigable Small World)
- **Compression**: Product Quantization (PQ)
- **Sample**: 606,380 chunks in `Features_narodne_novine_hr`

### 6. LLM Service (OpenRouter / Ollama)
- **Primary**: OpenRouter API
  - **Model**: qwen3-30b-a3b-instruct-2507 (Qwen 3)
  - **API**: https://openrouter.ai/api/v1
- **Local**: Ollama (optional/fallback)
  - **Model**: qwen2.5:7b-instruct
  - **Port**: 11434
- **Features**:
  - Multilingual (Croatian + English)
  - Instruction-following
  - Context-aware generation
  - Streaming support (configurable)

### 7. Metadata Database (PostgreSQL)
- **Schema**:
  - `conversations` - Chat sessions
  - `messages` - Individual messages
  - `token_usage` - Token tracking per message
  - `users` - User accounts
  - `tenants` - Multi-tenant data

## Document Ingestion Flow

```mermaid
graph LR
    A[📄 Document Upload] -->|PDF/HTML/DOCX/TXT| B[📄 Extractor]
    B -->|Raw Text| C[✂️ Chunker]
    C -->|512-char chunks<br/>50-char overlap| D[🔢 Embed Generator]
    D -->|1024-dim vectors| E[(🗄️ Weaviate)]

    E -->|Store| F[Collection:<br/>tenant_user_lang_documents]

    style A fill:#e3f2fd,color:#000
    style B fill:#fff3e0,color:#000
    style C fill:#f3e5f5,color:#000
    style D fill:#e8f5e9,color:#000
    style E fill:#fce4ec,color:#000
    style F fill:#f1f8e9,color:#000
```

### Ingestion Steps:
1. **Upload**: User uploads document via API
2. **Extract**: Content extraction based on file type
3. **Chunk**: Split into overlapping chunks (512 chars, 50 overlap)
4. **Embed**: Generate vectors using language-specific model
5. **Store**: Save to Weaviate with metadata
6. **Index**: HNSW index automatically updated

## Configuration-Driven Routing

```mermaid
graph TB
    Query[User Query] --> LangDetect{Language?}

    LangDetect -->|hr| HrConfig[Croatian Config<br/>config/hr.toml]
    LangDetect -->|en| EnConfig[English Config<br/>config/en.toml]

    HrConfig --> HrModel[classla/bcms-bertic<br/>768-dim]
    EnConfig --> EnModel[BAAI/bge-large-en-v1.5<br/>1024-dim]

    HrModel --> HrCollection[(Collection:<br/>*_hr)]
    EnModel --> EnCollection[(Collection:<br/>*_en)]

    HrCollection --> Search[Vector Search]
    EnCollection --> Search

    Search --> Results[Retrieved Results]
```

## Multi-Tenant Data Isolation

```
Tenant: development
├── User: dev_user
│   ├── Language: hr
│   │   ├── Collection: development_dev_user_hr_documents
│   │   └── Data: data/development/users/dev_user/hr/
│   └── Language: en
│       ├── Collection: development_dev_user_en_documents
│       └── Data: data/development/users/dev_user/en/
│
└── Shared (tenant-level)
    ├── Language: hr
    │   └── Collection: development_shared_hr_documents
    └── Language: en
        └── Collection: development_shared_en_documents

Feature Scope (outside tenant):
└── narodne-novine
    └── Language: hr
        ├── Collection: Features_narodne_novine_hr
        └── Data: data/features/narodne_novine/documents/hr/
```

## Performance Characteristics

### Query Latency Breakdown (Typical):
- Query processing: ~50ms
- Embedding generation: ~200ms (CPU) / ~50ms (GPU)
- Vector search: ~100ms (606K vectors)
- Reranking: ~150ms (top 20 → top 5)
- LLM generation: ~2-5s (depends on output length)
- **Total**: ~3-6 seconds end-to-end

### Scaling Factors:
- Collection size: 606,380 documents (Narodne Novine)
- Vector dimension: 1024 (BGE-M3)
- Index type: HNSW (ef=128)
- Batch processing: 32 docs/batch
- Concurrent users: Handled by FastAPI async

## API Request/Response Examples

### Query Request
```json
POST /api/query
{
  "query": "Kolika je najviša cijena goriva?",
  "tenant": "development",
  "user": "dev_user",
  "language": "hr",
  "scope": "feature",
  "feature_name": "narodne-novine",
  "top_k": 5
}
```

### Query Response
```json
{
  "answer": "Prema podacima iz Narodnih novina...",
  "sources": [
    {
      "chunk_id": "2019_07_72_1535_0002",
      "source": "NN 72/2019",
      "relevance_score": 0.87,
      "content": "..."
    }
  ],
  "token_usage": {
    "input_tokens": 1247,
    "output_tokens": 156,
    "total_tokens": 1403
  },
  "processing_time_ms": 3421,
  "language": "hr"
}
```

## Service Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  USER LAYER                                                 │
│  👤 Browser/Client                                          │
└─────────────────────────────────────────────────────────────┘
                           ↓ HTTP/HTTPS
┌─────────────────────────────────────────────────────────────┐
│  PROXY LAYER                                                │
│  🔀 Nginx (Port 80) - Load Balancer & Reverse Proxy        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                         │
│  🖥️ Web UI (React, Port 5173)                               │
└─────────────────────────────────────────────────────────────┘
                           ↓ REST API
┌─────────────────────────────────────────────────────────────┐
│  API GATEWAY LAYER                                          │
│  🌐 Web API (Express, Port 3000)                            │
│     - Authentication (JWT)                                  │
│     - Session Management                                    │
│     - Request Routing                                       │
│     - Rate Limiting                                         │
└─────────────────────────────────────────────────────────────┘
         ↓ RAG Queries              ↕️ Cache
┌──────────────────────────┐   ┌─────────────────────┐
│  RAG PROCESSING LAYER    │   │  CACHE LAYER        │
│  ⚡ RAG API (FastAPI,    │   │  ⚡ Redis (6379)    │
│     Port 8082)           │   │    - Sessions       │
│  🧠 RAG Service (Python) │   │    - Rate Limits    │
│     - Core Logic         │   │    - Real-time Data │
│     - Embeddings         │   └─────────────────────┘
│     - Vector Search      │
│     - LLM Integration    │
└──────────────────────────┘
         ↓ Store/Retrieve
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                 │
│  💾 PostgreSQL (5434)      🗄️ Weaviate (8080/50051)         │
│     - Users, Messages      - Vector storage                 │
│     - Conversations        - 606K+ documents                │
│     - Token Tracking       - HNSW search index              │
└─────────────────────────────────────────────────────────────┘
                           ↓ Generate
┌─────────────────────────────────────────────────────────────┐
│  EXTERNAL SERVICES                                          │
│  🤖 OpenRouter API (Qwen 3) - Primary LLM                   │
│  🤖 Ollama (11434, Qwen 2.5) - Optional Local LLM           │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack Summary

| Service | Technology | Port | Deployment | Purpose |
|---------|-----------|------|------------|---------|
| **Infrastructure** |
| Nginx | Nginx | 80 | SystemD | Reverse proxy, load balancer |
| Redis | Redis 7 | 6379 | SystemD | Cache, sessions, rate limiting |
| **Application** |
| Web UI | React + Vite | 5173 | SystemD | Frontend interface |
| Web API | Node.js + Express | 3000 | SystemD | API Gateway, auth |
| RAG API | Python + FastAPI | 8082 | SystemD | RAG query processing |
| RAG Service | Python 3.12 | CLI | Native | Core RAG engine |
| **Data** |
| PostgreSQL | PostgreSQL 16 | 5434 | SystemD | Relational data |
| Weaviate | Weaviate 1.33 | 8080/50051 | SystemD | Vector database |
| **AI/ML** |
| OpenRouter | External API | HTTPS | Cloud | LLM (Qwen 3) |
| Ollama | Local service | 11434 | Optional | Local LLM (optional) |
| Embeddings | HuggingFace | - | Python | Vector generation |
| **Management** |
| SystemD | Service manager | - | Native | Service orchestration |
| Makefile | Build automation | - | Native | Task management |
| TOML | Config files | - | Native | Configuration |
| JSON Logger | Custom | - | Python | Structured logging |

## Key Files & Locations

### Application Files
```
/home/rag/src/rag/
├── Makefile                                   # Service management commands
├── manage-services.sh                         # Service control script
├── setup-local-server.sh                      # Installation script
├── .env.local                                 # Environment configuration
├── services/
│   ├── rag-api/              # FastAPI service (Port 8082)
│   │   └── main.py
│   ├── web-api/              # Express service (Port 3000)
│   │   ├── src/
│   │   └── prisma/           # Database schema
│   ├── web-ui/               # React frontend (Port 5173)
│   │   └── src/
│   └── rag-service/          # Core RAG logic
│       ├── src/
│       │   ├── pipeline/rag_system.py       # Main RAG orchestrator
│       │   ├── vectordb/
│       │   │   ├── embeddings.py            # Embedding generation
│       │   │   ├── search.py                # Vector search
│       │   │   └── storage.py               # Weaviate interface
│       │   ├── retrieval/
│       │   │   ├── hybrid_retriever.py      # Hybrid search
│       │   │   └── reranker.py              # Cross-encoder reranking
│       │   ├── generation/
│       │   │   ├── ollama_client.py         # Ollama integration
│       │   │   └── prompt_templates.py      # Prompt engineering
│       │   └── preprocessing/
│       │       ├── extractors.py            # Document extraction
│       │       └── chunkers.py              # Text chunking
│       └── config/
│           ├── config.toml                   # Shared config
│           ├── hr.toml                       # Croatian config
│           └── en.toml                       # English config
├── data/
│   ├── features/narodne_novine/              # Feature data
│   └── {tenant}/users/{user}/{lang}/         # User data
├── weaviate_data/                            # Weaviate vector storage
├── logs/                                     # Application logs
└── rag.py                                     # CLI entry point
```

### SystemD Service Files
```
/etc/systemd/system/
├── weaviate.service          # Weaviate vector database
├── rag-api.service           # RAG FastAPI service
├── web-api.service           # Web Express API
└── web-ui.service            # React frontend

# System services (pre-installed)
├── postgresql.service        # PostgreSQL 16
├── redis-server.service      # Redis 7
└── nginx.service             # Nginx reverse proxy
```

### Data Directories
```
/opt/weaviate/                # Weaviate binary
/home/rag/src/rag/weaviate_data/  # Weaviate data
/var/lib/postgresql/16/main/  # PostgreSQL data
/var/lib/redis/               # Redis data
```

## Service Deployment Options

### Option 1: Local SystemD Services (Current Setup)

**Native Ubuntu deployment** without Docker:

```bash
# SystemD services (managed via manage-services.sh)
● postgresql          # PostgreSQL 16 (Port 5434)
● redis-server        # Redis 7 (Port 6379)
● weaviate           # Weaviate 1.33.0 (Port 8080/50051)
● rag-api            # FastAPI RAG API (Port 8082)
● web-api            # Express Backend (Port 3000)
● web-ui             # React Frontend (Port 5173)
● nginx              # Reverse Proxy (Port 80)

# Service control via Makefile
make start           # Start all services
make stop            # Stop all services
make restart         # Restart all services
make status          # Check service status
make logs SERVICE=rag-api  # View logs
```

**Advantages:**
- Native performance (no container overhead)
- Direct systemd integration
- Simpler debugging
- Better resource utilization on powerful servers
- Optimized for 243GB RAM / 144 CPU cores

**Service locations:**
- Weaviate binary: `/opt/weaviate/`
- Weaviate data: `/home/rag/src/rag/weaviate_data/`
- PostgreSQL data: `/var/lib/postgresql/16/main/`
- Application code: `/home/rag/src/rag/`
- SystemD configs: `/etc/systemd/system/`

### Option 2: Docker Compose Stack

```yaml
# Complete service orchestration (docker-compose.yml)
services:
  nginx:           # Reverse Proxy (Port 80)
  web-ui:          # React Frontend (Port 5173)
  web-api:         # Express Backend (Port 3000)
  rag-api:         # FastAPI RAG API (Port 8082)
  rag-service:     # Python RAG Core (CLI mode)
  postgres:        # PostgreSQL DB (Port 5434)
  redis:           # Redis Cache (Port 6379)
  weaviate:        # Vector DB (Port 8080/50051)

# Network: rag_network (bridge)
# Volumes: postgres_data, redis_data, weaviate_data
```

**Advantages:**
- Portable across environments
- Isolated dependencies
- Easy scaling and orchestration
- Simplified deployment

### Service Dependencies

```mermaid
graph LR
    A[nginx] --> B[web-ui]
    A --> C[web-api]
    C --> D[rag-api]
    D --> E[rag-service]
    C --> F[redis]
    C --> G[postgres]
    E --> G
    E --> H[weaviate]
    D --> H

    style A fill:#ffebee,color:#000
    style B fill:#f3e5f5,color:#000
    style C fill:#fff9c4,color:#000
    style D fill:#fff9c4,color:#000
    style E fill:#e8eaf6,color:#000
    style F fill:#fff3e0,color:#000
    style G fill:#e8f5e9,color:#000
    style H fill:#e8f5e9,color:#000
```

## Deployment Commands

### Local SystemD Services (Current)

```bash
# Installation (first time)
sudo ./setup-local-server.sh

# Start all services
make start
# or: sudo ./manage-services.sh start

# Stop all services
make stop
# or: sudo ./manage-services.sh stop

# Restart all services
make restart
# or: sudo ./manage-services.sh restart

# Check service status
make status
# or: sudo ./manage-services.sh status

# View logs for specific service
make logs SERVICE=rag-api
# or: sudo ./manage-services.sh logs rag-api

# Follow all logs in real-time
sudo ./manage-services.sh logs-all

# Health checks
make health
# or: sudo ./manage-services.sh health

# Restart specific service
make restart-one SERVICE=weaviate
# or: sudo systemctl restart weaviate

# Enable auto-start on boot
sudo ./manage-services.sh enable

# Execute RAG CLI commands
cd /home/rag/src/rag
source venv/bin/activate
python rag.py --language hr query "Što je RAG?"
python rag.py --language hr process-docs data/features/narodne_novine/
```

### Docker Compose (Alternative)

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f rag-api

# Execute RAG CLI commands
docker-compose exec rag-service python rag.py --language hr query "Što je RAG?"

# Stop services
docker-compose down
```

## Health Check Endpoints

| Service | Endpoint | Expected Response |
|---------|----------|-------------------|
| Nginx | http://localhost/health | 200 OK |
| Web UI | http://localhost:5173 | React App |
| Web API | http://localhost:3000/api/v1/health | 200 OK |
| RAG API | http://localhost:8082/health | 200 OK |
| Weaviate | http://localhost:8080/v1/meta | Weaviate metadata |
| PostgreSQL | `psql -h localhost -p 5434 -U raguser -d ragdb` | Connected |
| Redis | `redis-cli -p 6379 ping` | PONG |

## Performance Metrics

### Resource Allocation (from docker-compose.yml)
- **Weaviate**: 200GB RAM, 140 CPU cores
- **PostgreSQL**: Default limits
- **Redis**: Default limits
- **Services**: Default limits

### Expected Query Latency
- **Web API → RAG API**: ~50ms
- **RAG Query Processing**: 3-6 seconds
  - Embedding: ~200ms (CPU)
  - Vector Search: ~100ms
  - Reranking: ~150ms
  - LLM Generation: 2-5s
  - Token Tracking: ~50ms

---

**Generated**: 2025-10-10
**RAG System Version**: 1.0.0
**Deployment**: Native SystemD services on Ubuntu 24.04 LTS
**Services**: 7 SystemD services (9 including external LLMs)
**Hardware**: 243GB RAM, 144 CPU cores
**Document**: Complete architecture flow diagram with local and Docker deployment options
