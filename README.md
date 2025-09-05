# Multilingual RAG Learning Project

A hands-on learning project to build a complete Retrieval-Augmented Generation (RAG) system for multilingual documents LLM processing. This system handles multiple languages with specialized support for Croatian, English, and extensible architecture for additional languages including German, French, Spanish, and more.

## 🌍 Multilingual RAG System - Learn Croatian & English Document Processing

A comprehensive Retrieval-Augmented Generation (RAG) system designed for multilingual document processing, with specialized support for Croatian and English languages. This project demonstrates advanced document understanding, cross-language search capabilities, and intelligent content generation.

## 🎯 Key Features

- **🇭🇷 Croatian Language Support**: Advanced morphological processing, diacritics handling, cultural context awareness
- **🇬🇧 English Document Processing**: Business documents, financial reports, technical specifications
- **🌐 Cross-Language Search**: Query in one language, find relevant content in any language
- **🚀 Local LLM Integration**: Powered by Qwen2.5 via Ollama for privacy-first generation
- **📊 Intelligent Chunking**: Context-aware text segmentation preserving document structure
- **🔍 Hybrid Retrieval**: Combines semantic similarity with keyword matching and reranking

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd learn-rag

# Install dependencies
pip install -r requirements.txt

# Setup Ollama LLM
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b-instruct

# Test the system
python simple_test.py
```

## �️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTILINGUAL RAG ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📄 Documents                🔍 Processing                      │
│  ┌─────────────┐            ┌──────────────┐                   │
│  │ Croatian 🇭🇷 │ ──────────→ │ BGE-M3       │                   │
│  │ - PDF/DOCX  │            │ Embeddings   │                   │
│  │ - Diacritics│            │ (Multilingual│                   │
│  │ - Morphology│            │  Sentence-T) │                   │
│  └─────────────┘            └──────────────┘                   │
│                                     │                          │
│  ┌─────────────┐                   ▼                          │
│  │ English 🇬🇧  │            ┌──────────────┐                   │
│  │ - Business  │ ──────────→ │ ChromaDB     │                   │
│  │ - Financial │            │ Vector Store │                   │
│  │ - Technical │            │ (Language-   │                   │
│  └─────────────┘            │  Specific)   │                   │
│                              └──────────────┘                   │
│                                     │                          │
│                              ┌──────▼──────┐                   │
│  💬 User Query               │ Retrieval   │                   │
│  ┌─────────────┐            │ & Ranking   │                   │
│  │"Koliki je   │ ──────────→ │ System      │                   │
│  │ ukupni      │            └─────────────┘                   │
│  │ iznos?"     │                    │                          │
│  └─────────────┘                   ▼                          │
│                              ┌──────────────┐                   │
│                              │ Qwen2.5      │                   │
│                              │ LLM          │                   │
│                              │ (Ollama)     │                   │
│                              └──────────────┘                   │
│                                     │                          │
│                                    ▼                          │
│                              📋 Generated Response             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
learn-rag/
├── config/                  # Configuration Files
│   ├── croatian.toml       # Croatian language settings
│   ├── english.toml        # English language settings
│   └── api_config.yaml     # API and model configurations
├── data/                    # Document Storage
│   ├── raw/                # Source documents
│   │   ├── hr/            # Croatian documents
│   │   └── en/            # English documents
│   ├── chromadb/          # Vector database storage
│   └── test/              # Sample test documents
├── src/                     # Core System
│   ├── preprocessing/      # Document processing
│   │   ├── extractors.py  # Text extraction (PDF, DOCX)
│   │   ├── cleaners.py    # Language-aware cleaning
│   │   └── chunkers.py    # Intelligent text segmentation
│   ├── vectordb/          # Vector Operations
│   │   ├── embeddings.py  # BGE-M3 multilingual embeddings
│   │   ├── storage.py     # ChromaDB management
│   │   └── search.py      # Similarity search
│   ├── retrieval/         # Search & Ranking
│   │   ├── retriever.py   # Document retrieval
│   │   ├── reranker.py    # Result reranking
│   │   └── query_processor.py # Query understanding
│   ├── generation/        # LLM Integration
│   │   ├── ollama_client.py   # Qwen2.5 interface
│   │   ├── prompt_templates.py # Language-specific prompts
│   │   └── response_parser.py  # Output processing
│   └── pipeline/          # System Orchestration
│       ├── rag_system.py  # Main RAG pipeline
│       └── config.py      # Configuration management
├── notebooks/               # Learning Materials
│   ├── 01_document_processing_learning.ipynb
│   ├── 02_vector_database_learning.ipynb
│   ├── 03_retrieval_system_learning.ipynb
│   ├── 04_generation_system_learning.ipynb
│   └── 05_complete_pipeline_learning.ipynb
└── tests/                    # Comprehensive Testing
    ├── test_*.py            # Unit tests for each component
    └── integration/         # Multilingual end-to-end tests
```

## 🚀 Quick Start

```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install language models
python -m spacy download hr_core_news_sm  # Croatian
python -m spacy download en_core_web_sm   # English

# Install Ollama for local LLM
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b-instruct  # Multilingual model

# Verify setup
python test_setup.py

# Run multilingual pipeline
python -m src.pipeline.rag_system --lang hr  # Croatian
python -m src.pipeline.rag_system --lang en  # English
```

## 🏗️ Architecture

Our multilingual RAG system follows a 5-step pipeline designed for cross-language processing:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │   User Query    │    │   Generated     │
│(PDF/DOCX/TXT)   │    │ (Any Language)  │    │   Answer        │
│ hr/ en/ de/     │    │                 │    │ (User Language) │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▲
┌─────────────────┐    ┌─────────────────┐    ┌─────────┴───────┐
│ 1. PREPROCESSING│    │ 3. RETRIEVAL    │    │ 4. GENERATION   │
│                 │    │                 │    │                 │
│ • Text Extract  │    │ • Query Process │    │ • Context Build │
│ • Multilingual  │    │ • Cross-Lang    │    │ • Qwen2.5 LLM   │
│   Cleaning      │    │   Search        │    │ • Language-     │
│ • Smart Chunk   │    │ • BGE-M3        │    │   Specific      │
│ • Lang Detect   │    │   Embeddings    │    │   Response      │
└─────────┬───────┘    └─────────┬───────┘    └─────────────────┘
          │                      ▲                      ▲
          ▼                      │                      │
┌─────────────────┐              │              ┌───────┴───────┐
│ 2. VECTOR DB    │──────────────┴──────────────│ 5. PIPELINE   │
│                 │                             │               │
│ • ChromaDB      │                             │ • Language    │
│ • Multilingual  │                             │   Detection   │
│   Collection    │                             │ • Smart Route │
│ • BGE-M3        │                             │ • Analytics   │
└─────────────────┘                             └───────────────┘
```
│                 │                             │               │
│ • Multilingual  │                             │ • Orchestrate │
│   Embeddings    │                             │ • Error Handle│
│ • ChromaDB      │                             │ • Optimize    │
│   Storage       │                             │               │
│ • Similarity    │                             │               │
│   Search        │                             │               │
└─────────────────┘                             └───────────────┘
```

**Data Flow:**
1. **Documents** → Extracted & chunked with Croatian awareness
2. **Chunks** → Embedded using multilingual models → Stored in ChromaDB
3. **Query** → Processed for Croatian morphology → Semantic search → Ranked results
4. **Context + Query** → Croatian-optimized prompts → Ollama LLM → Generated answer
5. **Pipeline** → Orchestrates all components with error handling

## ✨ Key Features

### Multilingual Language Support
- **🇭🇷 Croatian**: Diacritic preservation (Č, Ć, Š, Ž, Đ), morphological analysis, cultural context
- **🇬🇧 English**: Comprehensive processing with business and technical document support
- **🌐 Cross-Language**: Unified search across multiple languages with BGE-M3 embeddings
- **📁 Auto-Organization**: Language detection and automatic document folder organization
- **🔄 Translation Framework**: Ready for query/response translation capabilities

### Intelligent Retrieval
- **Multi-Strategy Search**: Semantic + keyword + metadata filtering across languages
- **Adaptive Retrieval**: Adjusts strategy based on query type and language
- **7-Signal Ranking**: Semantic similarity, keyword match, recency, authority, and more
- **Smart Query Processing**: Language-aware query classification and expansion
- **Cross-Language Search**: Find relevant content regardless of document language

### Local-First Architecture
- **No External APIs**: Everything runs locally using Ollama
- **Privacy Focused**: Documents never leave your machine
- **Cost Efficient**: No per-query API costs
- **Offline Capable**: Works without internet connection

## 📁 Project Structure

```
src/
├── preprocessing/          # Step 1: Document Processing
│   ├── extractors.py      # PDF/DOCX/TXT extraction
│   ├── cleaners.py        # Multilingual text cleaning
│   └── chunkers.py        # Language-aware document chunking
├── vectordb/              # Step 2: Vector Database
│   ├── embeddings.py      # BGE-M3 multilingual embeddings
│   ├── storage.py         # ChromaDB operations
│   └── search.py          # Cross-language similarity search
├── retrieval/             # Step 3: Intelligent Retrieval
│   ├── query_processor.py # Multilingual query understanding
│   ├── retriever.py       # Language-aware retrieval
│   └── ranker.py          # Multi-signal ranking
├── generation/            # Step 4: Local LLM Generation
│   ├── ollama_client.py   # Ollama integration
│   ├── prompt_templates.py# Language-specific prompts
│   └── response_parser.py # Multilingual response analysis
└── pipeline/              # Step 5: Complete Integration
    ├── rag_system.py      # Multilingual RAG pipeline
    └── config.py          # Language configuration management

data/                      # Language-Organized Data
├── raw/
│   ├── hr/               # Croatian documents
│   ├── en/               # English documents
│   └── multilingual/     # Mixed-language documents
├── processed/            # Language-specific processing cache
└── chromadb/            # Unified multilingual vector storage

scripts/                  # Batch Processing Tools
├── batch_process.py     # Language-aware document processing
└── analytics.py         # Multilingual usage analytics

notebooks/                # Learning Materials
├── 00_system_overview_and_architecture.ipynb
├── 01_document_processing_learning.ipynb
├── 02_vector_database_learning.ipynb
└── 03_retrieval_system_learning.ipynb

tests/                    # Comprehensive Testing
├── test_*.py            # Unit tests for each component
└── integration/         # Multilingual end-to-end tests
```

## 🚀 Example Usage

```bash
# Clear existing data (optional)
python clear_data.py all --dry-run    # Preview what will be cleared
python clear_data.py hr               # Clear only Croatian data
python clear_data.py en               # Clear only English data

# Add documents to the system
python -c "
from src.pipeline.rag_system import MultilingualRAGSystem
rag = MultilingualRAGSystem('config/croatian.toml')
rag.add_document('data/raw/hr/document.pdf')
"

# Query in Croatian
python -c "
from src.pipeline.rag_system import MultilingualRAGSystem
rag = MultilingualRAGSystem('config/croatian.toml')
result = rag.query('Koliki je ukupni iznos u EUR-ima?')
print(result['answer'])
"

# Query in English
python -c "
from src.pipeline.rag_system import MultilingualRAGSystem
rag = MultilingualRAGSystem('config/english.toml')
result = rag.query('What is the total amount in EUR?')
print(result['answer'])
"

# Quick testing
python simple_test.py                 # Test both languages
python test_complete_system.py        # Full system test
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific components
pytest tests/test_preprocessing.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_generation.py -v

# Test with actual documents
python test_specific_query.py
python interactive_test.py
```

## 🔧 Configuration

### Language-Specific Settings

The system uses TOML configuration files for each language:

**Croatian (`config/croatian.toml`)**:
```toml
[language]
code = "hr"
name = "Croatian"
language_uppercase_chars = ["Č", "Ć", "Š", "Ž", "Đ"]

[chunking]
strategy = "semantic_with_morphology"
chunk_size = 500
overlap = 50

[embeddings]
model = "BAAI/bge-m3"
cache_dir = "./models/embeddings/hr"
```

**English (`config/english.toml`)**:
```toml
[language]
code = "en"
name = "English"
language_uppercase_chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

[chunking]
strategy = "semantic"
chunk_size = 500
overlap = 50

[embeddings]
model = "BAAI/bge-m3"
cache_dir = "./models/embeddings/en"
```

### Key Features per Language

**Croatian Language Processing**:
- **Morphology**: Handles complex word inflections (Zagreb → Zagreba, Zagrebu, etc.)
- **Diacritics**: Preserves and processes Č, Ć, Š, Ž, Đ correctly
- **Cultural Context**: Recognizes "biser Jadrana" = Dubrovnik, historical references
- **Query Types**: Classifies Croatian questions (Koji? Kako? Zašto? etc.)

**English Language Processing**:
- **Business Documents**: Handles financial reports, legal documents, technical specs
- **Semantic Understanding**: Captures meaning beyond keyword matching
- **Query Processing**: Advanced question classification and expansion

**Cross-Language Capabilities**:
- **Unified Search**: Find relevant content regardless of document/query language
- **Language Detection**: Automatic language identification for documents and queries
- **Separate Collections**: Croatian documents stored in `croatian_documents`, English in `english_documents`

## 📚 Learning Path

This project is designed for hands-on multilingual RAG learning through Jupyter notebooks:

1. **[System Overview](notebooks/00_system_overview_and_architecture.ipynb)** - Architecture and component interaction
2. **[Document Processing](notebooks/01_document_processing_learning.ipynb)** - Text extraction and language-aware chunking
3. **[Vector Database](notebooks/02_vector_database_learning.ipynb)** - Cross-language embeddings and similarity search
4. **[Retrieval System](notebooks/03_retrieval_system_learning.ipynb)** - Intelligent multilingual document retrieval
5. **[Generation System](notebooks/04_generation_system_learning.ipynb)** - LLM integration with language-specific processing
6. **[Complete Pipeline](notebooks/05_complete_pipeline_learning.ipynb)** - Full system orchestration and testing

Each notebook includes comprehensive explanations, code examples, and language-specific implementation details.

## 🛠️ System Requirements

- **Python**: 3.8+
- **RAM**: 8GB+ recommended (for BGE-M3 embeddings)
- **Storage**: 5GB+ for models and data
- **Ollama**: For local LLM inference
- **Operating System**: Linux, macOS, Windows

## 📚 Dependencies

Key libraries and their purposes:

```
chromadb>=0.4.0           # Vector database
sentence-transformers     # BGE-M3 multilingual embeddings
ollama                   # Local LLM client
PyPDF2                   # PDF document processing
python-docx              # DOCX document processing
toml                     # Configuration file parsing
pytest                  # Testing framework
```

## 🔍 Troubleshooting

**Common Issues**:

1. **Ollama not responding**: Ensure Ollama service is running (`ollama serve`)
2. **Memory issues**: Reduce batch size in embeddings configuration
3. **Croatian text garbled**: Check file encoding (should be UTF-8)
4. **Empty search results**: Verify documents are properly indexed with `rag.get_stats()`

**Debug Tools**:
```bash
python debug_ollama.py           # Test LLM connection
python debug_rag_context.py      # Check retrieval system
python test_setup.py             # Verify system components
```

## 📄 License

MIT License - Feel free to use for learning and development.

---

**Built with ❤️ for multilingual AI learning**

## 📚 Documentation

For comprehensive documentation, visit the **[docs/ folder](docs/README.md)**:

- **[Multilingual Architecture](docs/MULTILINGUAL_ARCHITECTURE.md)** - Language-based folder structure and advanced features
- **[Performance Optimization](docs/PERFORMANCE_OPTIMIZATION.md)** - System tuning and optimization
- **[Web Interface Plan](docs/WEB_INTERFACE_PLAN.md)** - User interface development roadmap
- **[Device Setup Guides](docs/)** - Hardware configuration and troubleshooting

## 📚 Learning Path

This project is designed for hands-on multilingual RAG learning:

1. **Document Processing** - Handle multilingual text extraction and language-aware chunking
2. **Vector Database** - Implement cross-language embeddings and similarity search
3. **Retrieval System** - Build intelligent multilingual document retrieval
4. **Generation System** - Integrate local LLM with language-specific processing
5. **Complete Pipeline** - Orchestrate all components with language detection and routing

Each step includes comprehensive notebooks explaining the concepts and language-specific implementation details.

## 🔧 Configuration

Key settings in `config/`:

- **Language Support**: Croatian (`hr/`), English (`en/`), extensible for additional languages
- **Embedding Models**: BGE-M3 multilingual sentence-transformers
- **Language Processing**: Morphology patterns, stop words, cultural context per language
- **Retrieval Strategy**: Cross-language search weights, ranking signals, filtering
- **Generation**: Qwen2.5 model selection, language-specific prompt templates, response parsing

## � Multilingual Language Challenges Solved

### Croatian (🇭🇷)
- **Morphology**: Handles complex word inflections (Zagreb → Zagreba, Zagrebu, etc.)
- **Diacritics**: Preserves and processes Č, Ć, Š, Ž, Đ correctly
- **Cultural Context**: Recognizes "biser Jadrana" = Dubrovnik, historical references
- **Query Types**: Classifies Croatian questions (Koji? Kako? Zašto? etc.)

### English (🇬🇧)
- **Business Documents**: Handles financial reports, legal documents, technical specs
- **Semantic Understanding**: Captures meaning beyond keyword matching
- **Query Processing**: Advanced question classification and expansion

### Cross-Language (🌐)
- **Unified Search**: Find relevant content regardless of document/query language
- **Language Detection**: Automatic language identification for documents and queries
- **Translation Ready**: Framework for query translation and response localization

## 📄 License

MIT License - Feel free to use for learning and development.
