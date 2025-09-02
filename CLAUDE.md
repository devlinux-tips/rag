# Multilingual RAG Learning Project

## What this project does

This is a production-ready Retrieval-Augmented Generation (RAG) system for multilingual documents using state-of-the-art models and local LLM integration (Ollama). The system currently focuses on Croatian language support but is designed for multilingual expansion.

**Key capabilities:**
- Process Croatian documents (PDF, DOCX, TXT) with proper encoding and format handling
- Create semantic embeddings using **BGE-M3** (BAAI/bge-m3) - state-of-the-art multilingual model
- Store vectors in persistent ChromaDB with optimized collection management
- **Hybrid retrieval** combining dense and sparse search with BGE-reranker-v2-m3
- Generate contextual answers via Ollama with **Croatian OpenEU-LLM** (jobautomation/openeurollm-croatian:latest)
- **Multi-device support**: Auto-detection for CUDA (NVIDIA), MPS (Apple Silicon M1/M2/M3/M4), and CPU
- Handle Croatian language-specific challenges (diacritics, morphology, cultural context)

## Project Architecture & Design Principles

### **Core Design Principles Applied**

#### **1. DRY (Don't Repeat Yourself)**
- **Unified Configuration**: Single `config/config.toml` replacing 7 separate TOML files
- **Centralized Error Handling**: `handle_config_error()` pattern across all modules
- **Reusable Components**: Modular design with clear separation of concerns
- **Shared Utilities**: Croatian language utilities, device detection, config loading

#### **2. Clean Code Architecture**
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Explicit Dependencies**: Clear dependency injection and configuration management
- **Single Responsibility**: Each module has one clear purpose
- **Immutable Configuration**: TOML-based configuration with validation

#### **3. Modern Python Practices**
- **Async/Await**: Proper async implementation for I/O operations
- **Dataclasses**: Structured data models for configuration and results
- **Context Managers**: Resource management for database connections
- **Exception Handling**: Graceful degradation with meaningful error messages

#### **4. Production-Ready Standards**
- **Persistent Storage**: ChromaDB collections that survive restarts
- **Device Flexibility**: Automatic hardware detection and optimization
- **Security**: PyTorch 2.8.0+cu128 resolving security vulnerabilities
- **Performance**: BGE-M3 + BGE-reranker-v2-m3 for optimal Croatian language performance

### **Updated System Architecture**

```
learn-rag/
├── config/
│   ├── config.toml            # 🆕 UNIFIED configuration (replaces 7 files)
│   └── croatian.toml          # Croatian-specific settings
├── src/
│   ├── preprocessing/          # Document processing pipeline
│   │   ├── extractors.py      # PDF/DOCX/TXT extraction with proper encoding
│   │   ├── cleaners.py        # Croatian text normalization
│   │   └── chunkers.py        # Intelligent chunking strategies
│   ├── vectordb/              # Vector database operations
│   │   ├── embeddings.py      # 🆕 BGE-M3 embedding management
│   │   ├── storage.py         # 🆕 Persistent ChromaDB with collection management
│   │   └── search.py          # Optimized similarity search
│   ├── retrieval/             # 🆕 Advanced retrieval system
│   │   ├── query_processor.py # Croatian query preprocessing
│   │   ├── retriever.py       # Main retrieval logic
│   │   ├── hybrid_retriever.py # Dense + sparse hybrid search
│   │   ├── reranker.py        # 🆕 BGE-reranker-v2-m3 integration
│   │   └── ranker.py          # Result ranking & filtering
│   ├── generation/            # Local LLM integration
│   │   ├── ollama_client.py   # 🆕 Enhanced Ollama client
│   │   ├── prompt_templates.py # Croatian-optimized prompts
│   │   └── response_parser.py # Response processing
│   ├── pipeline/              # 🆕 Production RAG orchestration
│   │   ├── rag_system.py      # RAGSystem - main multilingual interface
│   │   └── config.py          # Configuration management
│   └── utils/
│       ├── croatian_utils.py  # Croatian language utilities
│       ├── config_loader.py   # 🆕 Centralized config loading
│       └── error_handler.py   # 🆕 DRY error handling patterns
├── data/
│   ├── raw/                   # Original Croatian documents
│   ├── vectordb/             # 🆕 Persistent vector storage
│   └── test/                  # Test documents and queries
├── models/
│   └── embeddings/            # Cached embedding models
└── notebooks/                 # Development and learning notebooks
```

### **🆕 Key Architectural Improvements**

#### **1. Unified Configuration System**
- **Before**: 7 separate TOML files with duplication
- **After**: Single `config/config.toml` + `config/croatian.toml` with all settings
- **Benefits**: No configuration drift, easier maintenance, single source of truth
- **⚠️ Critical**: Croatian prompt templates must be in `[prompts]` section, NOT under `[prompts.keywords]`

#### **2. Configuration Architecture Lessons Learned**
- **TOML Structure**: Subsections like `[prompts.keywords]` create new scope - templates after it become inaccessible
- **Template Organization**: All prompt templates must be at `[prompts]` root level before any subsections
- **Config Loading**: Croatian-specific components should use `get_croatian_settings()`, not general config
- **Error Handling**: Use consistent `handle_config_error()` pattern with proper parameters

#### **2. Enhanced Model Stack**
- **Embeddings**: BAAI/bge-m3 (1024-dim, multilingual, Croatian-optimized)
- **Reranking**: BAAI/bge-reranker-v2-m3 (Croatian multilingual support)
- **Generation**: jobautomation/openeurollm-croatian:latest (Croatian-specific LLM)
- **Security**: PyTorch 2.8.0+cu128 (resolved security vulnerabilities)

#### **3. Multi-Device Support**
- **Auto-Detection**: Automatic CUDA/MPS/CPU detection
- **Priority**: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU
- **Graceful Degradation**: Continues on CPU if GPU unavailable
- **Apple Silicon**: Full M1/M2/M3/M4 Pro support

#### **4. Production Storage**
- **Persistent Collections**: Data survives system restarts
- **Collection Management**: Proper naming ("croatian_documents")
- **Metadata Tracking**: Document counts, distance metrics
- **Storage Optimization**: Efficient chunk storage and retrieval

## Working on this project

### **Production Development Approach**
This project has evolved from learning-focused to **production-ready** implementation. Components are now integrated into a cohesive system with modern development practices.

**Current Implementation Status:**
1. ✅ **Document Processing** - Complete with proper encoding and format handling
2. ✅ **Vector Database** - BGE-M3 embeddings with persistent ChromaDB storage
3. ✅ **Advanced Retrieval** - Hybrid search with BGE-reranker-v2-m3
4. ✅ **Local LLM Integration** - Croatian OpenEU-LLM via Ollama
5. ✅ **Production Pipeline** - RAGSystem with full orchestration (designed for multilingual expansion)

### **Key Implementation Achievements**

#### **Configuration Management**
- **Unified TOML Configuration**: Single source of truth in `config/config.toml`
- **Environment Flexibility**: Auto-device detection (CUDA/MPS/CPU)
- **Error Resilience**: Graceful fallbacks with meaningful error messages
- **Croatian Localization**: Separate `croatian.toml` for language-specific settings

#### **Modern Architecture Patterns**
- **DRY Principle**: Eliminated code duplication across 7 config files
- **Clean Interfaces**: Clear separation between data, business logic, and presentation
- **Dependency Injection**: Configurable components with explicit dependencies
- **Async Programming**: Non-blocking I/O for database and API operations

#### **Croatian Language Optimization**
- **State-of-the-Art Models**: BGE-M3 + BGE-reranker-v2-m3 for Croatian
- **Cultural Context**: Croatian-specific prompt engineering and stop words
- **Encoding Robustness**: Proper UTF-8 handling for diacritics (Č, Ć, Š, Ž, Đ)
- **Morphological Awareness**: Croatian language utilities for text processing

#### **Production-Ready Features**
- **Data Persistence**: ChromaDB collections survive system restarts
- **Hardware Optimization**: Multi-device support with automatic selection
- **Security Compliance**: PyTorch 2.8.0+cu128 vulnerability fixes
- **Performance Monitoring**: Comprehensive logging and error tracking

### **Development Workflow**

#### **⚠️ Critical Development Patterns**

**Configuration Management:**
```python
# ✅ Correct: Use Croatian settings for Croatian components
from src.utils.config_loader import get_croatian_settings
config = get_croatian_settings()["prompts"]

# ❌ Wrong: Using general config for Croatian-specific templates
from src.utils.config_loader import get_generation_config
config = get_generation_config()["prompts"]  # Won't have Croatian templates
```

**Data Structure Access:**
```python
# ✅ Correct: RetrievalResult contains 'documents' (List[Dict])
retrieval_result = await retriever.retrieve(query)
chunks = [doc["content"] for doc in retrieval_result.documents]

# ❌ Wrong: Old pattern that was refactored
chunks = [doc.content for doc in retrieval_result.results]  # AttributeError
```

**Error Handling Pattern:**
```python
# ✅ Correct: Proper handle_config_error usage
config = handle_config_error(
    operation=lambda: get_croatian_settings(),
    fallback_value={"prompts": {}},
    config_file="config/croatian.toml",
    section="[prompts]"
)

# ❌ Wrong: Invalid parameter
config = handle_config_error(
    operation=lambda: get_croatian_settings(),
    fallback_value={"prompts": {}},
    operation_name="loading config"  # Parameter doesn't exist
)
```

#### **Primary Interface: RAGSystem**
```python
from src.pipeline.rag_system import RAGSystem, RAGQuery

# Initialize production-ready RAG system (currently Croatian-optimized)
rag = RAGSystem()
await rag.initialize()

# Process documents (persistent storage)
await rag.process_documents("data/raw")

# ✅ Correct: Query with RAGQuery object
query = RAGQuery(text="Što je RAG sustav?")
results = await rag.query(query)

# ❌ Wrong: Passing string directly (causes AttributeError)
results = await rag.query("Što je RAG sustav?")
```

#### **Configuration-Driven Development**
- **Single Source**: All settings in `config/config.toml`
- **Environment-Aware**: Automatic device and model selection
- **Override Support**: Runtime parameter overrides for testing
- **Validation**: Built-in configuration validation with meaningful errors

### **Technical Stack (Production)**
- **Python 3.12+** with modern async/await patterns
- **Vector DB**: ChromaDB with persistent collections
- **Embeddings**: **BAAI/bge-m3** (state-of-the-art multilingual, 1024-dim)
- **Reranking**: **BAAI/bge-reranker-v2-m3** (Croatian multilingual support)
- **LLM**: **jobautomation/openeurollm-croatian:latest** via Ollama
- **Security**: PyTorch 2.8.0+cu128 (CUDA 12.8 support)
- **Hardware**: CUDA (NVIDIA) + MPS (Apple Silicon M1/M2/M3/M4) + CPU fallback

### Claude Model Selection Strategy

**Default: Claude Sonnet 4**
- Document processing and chunking
- Basic embedding and retrieval implementation
- Standard prompt engineering
- Code debugging and refactoring
- Configuration and setup tasks

**Switch to Claude Opus 4.1 when:**
- **Complex Architecture Decisions**: Designing multi-component integrations
- **Advanced Retrieval Logic**: Implementing hybrid search or re-ranking algorithms
- **Croatian Language Challenges**: Complex morphological analysis or cultural context
- **Performance Optimization**: Algorithm optimization requiring deep reasoning
- **Error Analysis**: Debugging complex interaction issues between components
- **Research Questions**: Understanding theoretical RAG concepts deeply

**Auto-Switching Prompts:**
Add to your requests: "Suggest if this requires Opus-level reasoning" and Claude will recommend switching for complex tasks.

**Usage Pattern:**
Use `/model` command in Claude Code when Claude suggests switching, or when you encounter tasks requiring deep analytical thinking rather than speed.

### Technical stack
- **Python 3.9+** with sentence-transformers, chromadb, requests
- **Vector DB**: ChromaDB (free, local storage)
- **Embeddings**: BAAI/bge-m3 (BGE-M3: state-of-the-art multilingual embeddings with excellent Croatian support)
- **LLM**: Ollama with qwen2.5:7b-instruct (free, local, efficient on most hardware)
- **Documents**: PDF, DOCX, TXT support with Croatian encoding

### **Development Commands (Updated)**
```bash
# Environment Management
source venv/bin/activate          # Activate environment
deactivate                        # Deactivate environment

# Production RAG System
python -c "
import asyncio
from src.pipeline.rag_system import RAGSystem

async def main():
    rag = RAGSystem()
    await rag.initialize()
    # System ready for production use
asyncio.run(main())
"

# Configuration Testing
python -c "from src.utils.config_loader import get_unified_config; print(get_unified_config())"

# Component Testing
pytest tests/ -v                  # Run all tests
python -m pytest tests/test_rag_system.py -v     # Test main RAG system
python -m pytest tests/test_embeddings.py -v     # Test BGE-M3 embeddings
python -m pytest tests/test_storage.py -v        # Test persistent storage

# Development Tools
jupyter notebook notebooks/       # Explore development notebooks
black src/ --line-length 88      # Format code (Black)
ruff check src/                   # Fast linting (Ruff)
mypy src/                         # Type checking
```

### **Code Quality Standards**

#### **Type Safety & Modern Python**
```python
# ✅ Proper type hints
async def process_query(query: str, top_k: int = 5) -> List[RetrievalResult]:
    ...

# ✅ Dataclasses for structured data
@dataclass
class EmbeddingConfig:
    model_name: str
    device: str
    normalize_embeddings: bool

# ✅ Async/await for I/O operations
async def initialize_database() -> None:
    ...
```

#### **Configuration Patterns**
```python
# ✅ DRY error handling
config = handle_config_error(
    operation=lambda: get_reranking_config(),
    fallback_value=DEFAULT_CONFIG,
    config_file="config/config.toml",
    section="[reranking]"
)

# ✅ Centralized configuration loading
from src.utils.config_loader import get_unified_config
config = get_unified_config()
```

#### **Clean Architecture Principles**
- **Single Responsibility**: Each class has one clear purpose
- **Dependency Inversion**: Depend on abstractions, not concretions
- **Interface Segregation**: Small, focused interfaces
- **Open/Closed**: Open for extension, closed for modification

### **🔧 Common Refactoring Issues & Solutions**

#### **Issue 1: TOML Configuration Structure**
**Problem**: Templates placed after `[prompts.keywords]` become part of keywords subsection
```toml
# ❌ Wrong: Templates after subsection are inaccessible
[prompts]
base_setting = "value"

[prompts.keywords]
cultural = ["hrvatski", "kultura"]

# These templates become part of [prompts.keywords] scope!
cultural_context_system = "Ti si stručnjak..."  # INVISIBLE!

# ✅ Correct: Templates before any subsections
[prompts]
base_setting = "value"
cultural_context_system = "Ti si stručnjak..."  # ACCESSIBLE!

[prompts.keywords]
cultural = ["hrvatski", "kultura"]
```

#### **Issue 2: Refactoring Inconsistencies**
**Problem**: Attribute name changes not propagated throughout codebase
```python
# Old pattern (before refactoring)
retrieval_result.results[0].content

# New pattern (after refactoring)
retrieval_result.documents[0]["content"]
```

#### **Issue 3: Mixed Configuration Sources**
**Problem**: Components using wrong config sources after refactoring
```python
# ✅ Croatian components should use Croatian config
croatian_config = get_croatian_settings()
templates = croatian_config["prompts"]

# ❌ Not general generation config
general_config = get_generation_config()  # Missing Croatian templates
```

### Croatian-specific considerations
- Handle Croatian morphology and inflection in text processing
- Preserve diacritics in document chunking and retrieval
- Account for Croatian cultural context in prompt engineering
- Test with Croatian Wikipedia articles or news documents

## What to focus on

### **Production System Capabilities**
- **✅ Complete RAG Pipeline**: End-to-end document processing to answer generation
- **✅ Croatian Language Mastery**: State-of-the-art multilingual models optimized for Croatian
- **✅ Production Architecture**: DRY principles, unified configuration, clean code patterns
- **✅ Multi-Device Support**: Automatic CUDA/MPS/CPU detection and optimization
- **✅ Data Persistence**: ChromaDB collections that survive system restarts

### **Current System Performance**
- **Document Processing**: Handles PDF, DOCX, TXT with proper Croatian encoding
- **Vector Storage**: 275+ document chunks stored persistently in ChromaDB
- **Semantic Search**: BGE-M3 embeddings with 1024-dimensional vectors
- **Intelligent Reranking**: BGE-reranker-v2-m3 for Croatian context understanding
- **Answer Generation**: Croatian OpenEU-LLM via Ollama for cultural context

### **Architecture Achievements**

#### **Configuration Evolution**
- **Before**: 7 separate TOML files with configuration drift
- **After**: Single unified `config/config.toml` with validation
- **Impact**: Eliminated duplication, easier maintenance, single source of truth

#### **Model Optimization**
- **Embeddings**: Upgraded to BGE-M3 (best-in-class multilingual performance)
- **Reranking**: Implemented BGE-reranker-v2-m3 (Croatian language support)
- **Generation**: Croatian-specific OpenEU-LLM (cultural context awareness)
- **Security**: PyTorch 2.8.0+cu128 (resolved security vulnerabilities)

#### **Development Quality**
- **DRY Implementation**: Eliminated code duplication across modules
- **Type Safety**: Comprehensive type hints throughout codebase
- **Error Handling**: Graceful degradation with meaningful error messages
- **Testing Coverage**: Unit and integration tests for all components

### **Future Development Areas**

#### **Performance Optimization**
- **Batch Processing**: Optimize embedding generation for large document sets
- **Caching Strategies**: Implement intelligent query result caching
- **Index Optimization**: Fine-tune ChromaDB performance settings
- **Memory Management**: Optimize model loading for resource-constrained environments

#### **Croatian Language Enhancement**
- **Morphological Analysis**: Advanced Croatian language understanding
- **Cultural Context**: Enhanced prompt engineering for Croatian cultural nuances
- **Regional Variants**: Support for Croatian regional language variations
- **Historical Documents**: Handling of older Croatian text styles

#### **Production Features**
- **API Interface**: REST API for system integration
- **Monitoring**: Performance metrics and system health monitoring
- **Scalability**: Multi-user and concurrent query support
- **Security**: Authentication and access control mechanisms

### **System Validation**
✅ **Persistent Storage**: 275 chunks stored in "croatian_documents" collection
✅ **Multi-Device**: Tested on CUDA, MPS (Apple Silicon), and CPU
✅ **Croatian Processing**: Proper handling of diacritics and morphology
✅ **Configuration**: Unified TOML configuration working across all modules
✅ **Reranking**: BGE-reranker-v2-m3 providing intelligent result ordering
✅ **Error Handling**: Graceful degradation when components unavailable

**Ready for production use** with RAGSystem as the primary interface (currently optimized for Croatian with multilingual expansion planned).
