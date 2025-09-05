# RAG Service

Python-based multilingual RAG system with BGE-M3 embeddings and qwen2.5:7b-instruct generation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the RAG system
python rag.py

# Run tests
python -m pytest tests/ -v
```

## Configuration

Configuration is handled through TOML files in the `config/` directory:
- `config.toml` - Main unified configuration
- `croatian.toml` - Croatian language settings
- `english.toml` - English language settings

## Documentation

For complete documentation, architecture details, and usage examples, see the main project documentation in the repository root:

- [Platform Documentation](../../CLAUDE.md)
- [Architecture](../../docs/architecture/)
- [API Documentation](../../docs/api/)
- [Croatian Language Features](../../docs/croatian-language/)

## Development

This service is part of the larger multilingual RAG platform. For platform-wide development instructions, see the root README.md and CLAUDE.md files.
