# Croatian RAG Setup - macOS M4 Pro

## Quick Setup

```bash
# 1. Run the setup script
./rag_macos_m4_setup.sh

# 2. Activate uv environment  
uv shell

# 3. Test the setup
uv run python test_setup.py
```

## M4 Pro Optimizations

### 🚀 **Apple Silicon GPU (MPS)**
- Automatic GPU acceleration for embeddings
- 5-10x faster than CPU for neural operations
- No manual configuration needed

### ⚡ **uv Package Management**
- 10-100x faster than pip
- Parallel dependency resolution
- Built-in virtual environment management

### 🧠 **Ollama with ARM64**
- Native Apple Silicon LLM inference
- Optimized memory usage
- Croatian language support via qwen2.5:7b-instruct

## Development Workflow

### Environment Management
```bash
# Activate environment
uv shell

# Add new package
uv add sentence-transformers

# Add dev dependency
uv add --dev pytest

# Update all packages
uv sync --upgrade

# Show environment info
uv info
```

### Running the RAG System
```bash
# Process Croatian documents
uv run python -m src.cli.rag_cli --language hr process-docs data/hr/

# Query in Croatian
uv run python -m src.cli.rag_cli --language hr query "Što je RAG sustav?"

# Query in English
uv run python -m src.cli.rag_cli --language en query "What is a RAG system?"

# Get system status
uv run python -m src.cli.rag_cli status
```

### Code Quality (M4 Pro Optimized)
```bash
# Format code (preserves AI-friendly patterns)
uv run python format_code.py

# Type checking (excludes CLI)
uv run mypy src/

# Linting with Ruff
uv run ruff check src/

# Format with Ruff
uv run ruff format src/
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Parallel testing (uses M4 Pro cores)
uv run pytest -n auto
```

## Hardware-Specific Tips

### 🎯 **M4 Pro Performance**
- **Memory**: Unified memory architecture - no GPU memory limits
- **Cores**: Utilizes all P+E cores automatically
- **Storage**: SSD optimized for large model caching
- **Neural Engine**: Used by some ML frameworks automatically

### 📊 **Monitoring Performance**
```bash
# Check MPS availability
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Monitor system resources
activity monitor # Or use htop: brew install htop

# Check Ollama model usage
ollama ps
```

### 🛠 **Configuration Files**

| File | Purpose | M4 Pro Specific |
|------|---------|-----------------|
| `pyproject.toml` | Python project config | Python 3.13, ARM64 optimizations |
| `config/config.toml` | RAG system config | MPS device selection |
| `requirements.txt` | Dependencies | Apple Silicon wheels |
| `.pre-commit-config.yaml` | Code quality | Ruff v0.13.0+ |

### 🐛 **Troubleshooting**

**MPS Issues:**
```bash
# Check MPS support
python -c "import torch; print(torch.backends.mps.is_built())"

# Force CPU if MPS has issues
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**uv Issues:**
```bash
# Reset uv cache
uv cache clean

# Reinstall project
rm -rf .venv uv.lock
uv sync
```

**Ollama Issues:**
```bash
# Restart Ollama
brew services restart ollama

# Check model status
ollama list
```

## Performance Benchmarks

### Expected M4 Pro Performance:
- **Document Processing**: ~50 docs/minute
- **Embedding Generation**: ~2000 chunks/minute (MPS)
- **Search**: <50ms for 1000+ documents
- **LLM Generation**: 15-30 tokens/second (qwen2.5:7b)

### Optimization Checklist:
- ✅ MPS enabled for embeddings
- ✅ Ollama using ARM64 models
- ✅ uv for fast dependency management
- ✅ Homebrew ARM64 packages
- ✅ Python 3.13 for latest performance improvements

---

## Next Steps

1. **Run the setup**: `./rag_macos_m4_setup.sh`
2. **Test Croatian docs**: Add some HR documents to `data/hr/`
3. **Benchmark**: Compare performance with your specific workload
4. **Optimize**: Adjust batch sizes based on available memory (16GB/32GB)

For production deployment, consider Docker with Apple Silicon base images.