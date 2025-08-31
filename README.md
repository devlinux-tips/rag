# Croatian RAG Learning Project

A hands-on learning project to build a complete Retrieval-Augmented Generation (RAG) system for Croatian documents using local LLM processing. This system handles Croatian language-specific challenges including diacritics, morphology, and cultural context.

## 🚀 Quick Start

```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Croatian language model
python -m spacy download hr_core_news_sm

# Install Ollama for local LLM
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull jobautomation/openeurollm-croatian:latest

# Verify setup
python test_setup.py

# Run example pipeline
python -m src.pipeline.rag_system
```

## 🏗️ Architecture

Our Croatian RAG system follows a 5-step pipeline designed for Croatian language processing:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │   User Query    │    │   Generated     │
│ (PDF/DOCX/TXT)  │    │  (Croatian)     │    │   Answer        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▲
┌─────────────────┐    ┌─────────────────┐    ┌─────────┴───────┐
│ 1. PREPROCESSING│    │ 3. RETRIEVAL    │    │ 4. GENERATION   │
│                 │    │                 │    │                 │
│ • Text Extract  │    │ • Query Process │    │ • Context Build │
│ • Croatian      │    │ • Semantic      │    │ • Ollama LLM    │
│   Cleaning      │    │   Search        │    │ • Croatian      │
│ • Smart Chunk   │    │ • Multi-Signal  │    │   Response      │
│                 │    │   Ranking       │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────────────┘
          │                      ▲                      ▲
          ▼                      │                      │
┌─────────────────┐              │              ┌───────┴───────┐
│ 2. VECTOR DB    │──────────────┴──────────────│ 5. PIPELINE   │
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

### Croatian Language Support
- **Diacritic Preservation**: Maintains Č, Ć, Š, Ž, Đ throughout pipeline
- **Morphological Analysis**: Handles Croatian word inflections and cases
- **Cultural Context**: Recognizes Croatian places, history, and cultural references
- **Stop Words**: Croatian-specific stop word filtering
- **Encoding**: Robust UTF-8 handling for Croatian text

### Intelligent Retrieval
- **Multi-Strategy Search**: Semantic + keyword + metadata filtering
- **Adaptive Retrieval**: Adjusts strategy based on query type (factual, explanatory, etc.)
- **7-Signal Ranking**: Semantic similarity, keyword match, recency, authority, and more
- **Query Understanding**: Classifies Croatian queries and expands with synonyms

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
│   ├── cleaners.py        # Croatian text cleaning
│   └── chunkers.py        # Smart document chunking
├── vectordb/              # Step 2: Vector Database
│   ├── embeddings.py      # Multilingual embeddings
│   ├── storage.py         # ChromaDB operations
│   └── search.py          # Similarity search
├── retrieval/             # Step 3: Intelligent Retrieval
│   ├── query_processor.py # Croatian query understanding
│   ├── retriever.py       # Retrieval orchestration
│   └── ranker.py          # Multi-signal ranking
├── generation/            # Step 4: Local LLM Generation
│   ├── ollama_client.py   # Ollama integration
│   ├── prompt_templates.py# Croatian-optimized prompts
│   └── response_parser.py # Response analysis
└── pipeline/              # Step 5: Complete Integration
    ├── rag_system.py      # Main RAG pipeline
    └── config.py          # Configuration management

notebooks/                 # Learning Materials
├── 00_system_overview_and_architecture.ipynb
├── 01_document_processing_learning.ipynb
├── 02_vector_database_learning.ipynb
└── 03_retrieval_system_learning.ipynb

tests/                     # Comprehensive Testing
├── test_*.py             # Unit tests for each component
└── integration/          # End-to-end pipeline tests
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

## 📚 Learning Path

This project is designed for hands-on RAG learning:

1. **Document Processing** - Handle Croatian text extraction and chunking
2. **Vector Database** - Implement embeddings and similarity search
3. **Retrieval System** - Build intelligent document retrieval
4. **Generation System** - Integrate local LLM processing
5. **Complete Pipeline** - Orchestrate all components

Each step includes comprehensive notebooks explaining the concepts and Croatian-specific implementation details.

## 🔧 Configuration

Key settings in `config/`:

- **Embedding Models**: Multilingual sentence-transformers
- **Croatian Processing**: Morphology patterns, stop words, cultural context
- **Retrieval Strategy**: Search weights, ranking signals, filtering
- **Generation**: Ollama model selection, prompt templates, response parsing

## 🇭🇷 Croatian Language Challenges Solved

- **Morphology**: Handles complex word inflections (Zagreb → Zagreba, Zagrebu, etc.)
- **Diacritics**: Preserves and processes Č, Ć, Š, Ž, Đ correctly
- **Cultural Context**: Recognizes "biser Jadrana" = Dubrovnik, historical references
- **Query Types**: Classifies Croatian questions (Koji? Kako? Zašto? etc.)
- **Semantic Understanding**: Captures meaning beyond keyword matching

## 📄 License

MIT License - Feel free to use for learning and development.
