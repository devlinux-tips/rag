# Multilingual RAG Learning Project

A hands-on learning project to build a complete Retrieval-Augmented Generation (RAG) system for multilingual documents LLM processing. This system handles multiple languages with specialized support for Croatian, English, and extensible architecture for additional languages including German, French, Spanish, and more.

## 🌍 Multilingual Capabilities

- **🇭🇷 Croatian**: Full morphology support, diacritics handling, cultural context
- **🇬🇧 English**: Comprehensive processing with cross-language search
- **🌐 Cross-Language**: Unified search across multiple languages
- **📁 Language Organization**: Automatic document organization by language
- **🔄 Translation Ready**: Framework for query/response translation
- **📊 Analytics**: Language-specific usage patterns and performance metrics

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
# Process documents by language
python scripts/batch_process.py --language hr --input data/raw/hr/  # Croatian
python scripts/batch_process.py --language en --input data/raw/en/  # English

# Query in different languages
python -m src.pipeline.rag_system --query "Koliki je ukupni iznos u EUR-ima?" --lang hr
python -m src.pipeline.rag_system --query "What is the total amount in EUR?" --lang en

# Cross-language search (Croatian query, search all languages)
python -m src.pipeline.rag_system --query "Koliki je ukupni iznos?" --lang hr --search-all

# Auto-detect language and route appropriately
python -m src.pipeline.rag_system --query "Total investment value?" --auto-detect
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test specific components
pytest tests/test_preprocessing.py -v
pytest tests/test_retrieval.py -v

# Run with Croatian test documents
python -m pytest tests/ -k "croatian" -v
```

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
