# Croatian RAG Learning Project

## What this project does

This is a hands-on learning project to build a complete Retrieval-Augmented Generation (RAG) system for Croatian documents using Claude Sonnet API with batch processing. The system will process Croatian text documents, create vector embeddings for semantic search, and generate answers using Claude Sonnet with retrieved context.

**Key capabilities being built:**
- Process Croatian documents (PDF, DOCX, TXT) with proper encoding
- Create semantic embeddings using free multilingual models
- Store vectors in local ChromaDB for fast similarity search
- Retrieve relevant document chunks for user queries
- Generate contextual answers via Claude Sonnet batch API
- Handle Croatian language-specific challenges (diacritics, morphology)

## Project structure

```
local-rag-croatian/
├── src/
│   ├── preprocessing/          # Document processing pipeline
│   │   ├── extractors.py      # PDF/DOCX/TXT text extraction
│   │   ├── cleaners.py        # Croatian text cleaning & normalization
│   │   └── chunkers.py        # Document chunking strategies
│   ├── vectordb/              # Vector database operations
│   │   ├── embeddings.py      # Embedding model management
│   │   ├── storage.py         # ChromaDB operations
│   │   └── search.py          # Similarity search implementation
│   ├── retrieval/             # Intelligent document retrieval
│   │   ├── query_processor.py # Croatian query preprocessing
│   │   ├── retriever.py       # Main retrieval logic
│   │   └── ranker.py          # Result ranking & filtering
│   ├── claude_api/            # Claude API integration
│   │   ├── batch_client.py    # Batch processing client
│   │   ├── prompt_templates.py # RAG prompt engineering
│   │   └── response_parser.py # Response processing
│   ├── pipeline/              # Complete RAG orchestration
│   │   ├── rag_system.py      # Main RAG pipeline
│   │   └── config.py          # Configuration management
│   └── utils/
│       ├── croatian_utils.py  # Croatian language utilities
│       └── evaluation.py     # Quality metrics
├── data/
│   ├── raw/                   # Original Croatian documents
│   ├── processed/             # Cleaned and chunked documents
│   └── test/                  # Test documents and queries
├── config/                    # YAML configuration files
├── notebooks/                 # Development notebooks
├── tests/                     # Unit and integration tests
└── requirements.txt           # Python dependencies
```

## Working on this project

### Development approach
This is a **learning-focused project** where each component should be built step-by-step to understand RAG fundamentals. Implement components in this order:

1. **Document Processing** (`src/preprocessing/`) - Handle Croatian text extraction and chunking
2. **Vector Database** (`src/vectordb/`) - Implement embeddings and similarity search
3. **Retrieval System** (`src/retrieval/`) - Build intelligent document retrieval
4. **Claude Integration** (`src/claude_api/`) - Implement batch API processing
5. **Complete Pipeline** (`src/pipeline/`) - Orchestrate all components

### Key implementation priorities
- **Croatian Language Support**: Proper UTF-8 encoding, diacritics handling (Č, Ć, Š, Ž, Đ)
- **Free/Local Focus**: Use ChromaDB and sentence-transformers (no paid services except Claude API)
- **Batch Processing**: Implement efficient Claude Sonnet batch API usage for cost optimization
- **Learning Documentation**: Explain each component's purpose in the RAG pipeline
- **Component Testing**: Test each piece independently with Croatian text before integration

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
- **Python 3.9+** with sentence-transformers, chromadb, anthropic
- **Vector DB**: ChromaDB (free, local storage)
- **Embeddings**: Multilingual sentence-transformers models
- **API**: Claude Sonnet with batch processing
- **Documents**: PDF, DOCX, TXT support with Croatian encoding

### Development Commands
```bash
# Virtual environment
source venv/bin/activate          # Activate environment
deactivate                        # Deactivate environment

# Testing and quality
python test_setup.py              # Verify setup
pytest tests/                     # Run unit tests
python -m pytest -v               # Verbose testing
black src/                        # Format code
flake8 src/                       # Lint code

# Development
jupyter notebook                  # Start Jupyter for exploration
python src/pipeline/rag_system.py # Run main pipeline
```

### Code Style
- Use type hints for all function parameters and returns: `def process_text(text: str) -> List[str]:`
- Import order: standard library, third-party, local modules with blank lines between
- Use descriptive variable names: `croatian_chunks` not `chunks`, `embedding_model` not `model`
- Docstrings for all classes and public methods using Google style
- Use pathlib.Path for file operations, not os.path
- Exception handling: specific exceptions, not bare `except:`
- Use dataclasses or Pydantic models for configuration objects
- Async/await for API calls: `async def process_batch():`

### Croatian-specific considerations
- Handle Croatian morphology and inflection in text processing
- Preserve diacritics in document chunking and retrieval
- Account for Croatian cultural context in prompt engineering
- Test with Croatian Wikipedia articles or news documents

## What to focus on

### Learning objectives
- **Understand RAG Architecture**: Learn how retrieval and generation work together
- **Master Each Component**: Build preprocessing, embeddings, retrieval, and generation from scratch
- **Croatian NLP Challenges**: Handle non-English text processing complexities  
- **API Efficiency**: Implement batch processing for cost-effective Claude usage
- **System Integration**: Connect components into a production-ready pipeline

### Success criteria
- Process Croatian documents without encoding issues
- Retrieve semantically relevant chunks for Croatian queries
- Generate coherent Croatian answers using retrieved context
- Demonstrate understanding of each RAG pipeline component
- Handle batch processing efficiently with proper error handling

### Areas needing attention
- **Text Encoding**: Ensure proper UTF-8 handling throughout pipeline
- **Chunking Strategy**: Balance context preservation with retrieval precision
- **Embedding Quality**: Evaluate multilingual model performance on Croatian
- **Prompt Engineering**: Optimize Croatian language generation prompts
- **Error Handling**: Graceful failure handling across all components

Start with document processing to establish a solid foundation, then build each component incrementally while testing with Croatian documents at every step.