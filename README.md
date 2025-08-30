# Croatian RAG Learning Project

A hands-on learning project to build a complete Retrieval-Augmented Generation (RAG) system for Croatian documents using Claude Sonnet API with batch processing.

## Quick Start

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Configure environment:**
   ```bash
   cp .env.template .env
   # Edit .env with your Claude API key
   ```

3. **Test installation:**
   ```bash
   python -c "import chromadb, sentence_transformers, anthropic; print('✅ All dependencies installed')"
   ```

## Project Structure

See `CLAUDE.md` for detailed project information and development approach.

## Next Steps

1. Add your Claude API key to `.env`
2. Place Croatian documents in `data/raw/`
3. Start implementing components following the learning path in `CLAUDE.md`
