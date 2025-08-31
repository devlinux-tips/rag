#!/bin/bash

# Croatian RAG Learning Project Setup Script
# Ubuntu 24.04 with 64GB RAM, i7-11800H, NVIDIA T1200
# Optimized for learning RAG fundamentals with Croatian documents

set -e  # Exit on any error

echo "🚀 Setting up Croatian RAG Learning Project on Ubuntu 24.04..."
echo "System specs: 64GB RAM, i7-11800H, NVIDIA T1200"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run as root. Run as regular user."
    exit 1
fi

# Create project directory
PROJECT_NAME="local-rag-croatian"
print_status "Creating project directory: $PROJECT_NAME"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies for Croatian RAG
print_status "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    gfortran \
    sqlite3 \
    libsqlite3-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-hrv \
    libreoffice

print_success "System dependencies installed"

# Create Python virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and setuptools
print_status "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for embeddings - sufficient for learning)
print_status "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Croatian RAG dependencies
print_status "Installing Croatian RAG dependencies..."

# Core RAG components
pip install \
    chromadb \
    sentence-transformers \
    anthropic \
    numpy \
    pandas

# Document processing
pip install \
    pypdf2 \
    python-docx \
    beautifulsoup4 \
    pdfplumber \
    python-magic

# Croatian NLP support
pip install \
    spacy \
    nltk

# Configuration and utilities
pip install \
    python-dotenv \
    pyyaml \
    pydantic \
    click \
    tqdm

# Development and testing
pip install \
    jupyter \
    pytest \
    black \
    flake8

# Optional: Install spaCy Croatian model
print_status "Installing spaCy Croatian language model..."
python -m spacy download hr_core_news_sm

# Create project structure
print_status "Creating project directory structure..."
mkdir -p src/{preprocessing,vectordb,retrieval,claude_api,pipeline,utils}
mkdir -p data/{raw,processed,test}
mkdir -p config
mkdir -p notebooks
mkdir -p tests
mkdir -p docs

# Create requirements.txt
print_status "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core RAG components
chromadb==0.4.24
sentence-transformers==2.2.2
anthropic==0.7.8
numpy==1.24.3
pandas==2.0.3

# PyTorch (CPU version for embeddings)
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2

# Document processing
pypdf2==3.0.1
python-docx==0.8.11
beautifulsoup4==4.12.2
pdfplumber==0.9.0
python-magic==0.4.27

# Croatian NLP
spacy==3.6.1
nltk==3.8.1

# Configuration and utilities
python-dotenv==1.0.0
pyyaml==6.0.1
pydantic==2.4.2
click==8.1.7
tqdm==4.66.1

# Development and testing
jupyter==1.0.0
pytest==7.4.2
black==23.9.1
flake8==6.1.0
EOF

# Create .env template
print_status "Creating .env template..."
cat > .env.template << 'EOF'
# Claude API Configuration
ANTHROPIC_API_KEY=your_claude_api_key_here

# Vector Database Configuration
CHROMA_DB_PATH=./data/vectordb
CHROMA_COLLECTION_NAME=croatian_documents

# Embedding Model Configuration
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_BATCH_SIZE=32

# Document Processing Configuration
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50
SUPPORTED_FORMATS=pdf,docx,txt

# API Configuration
CLAUDE_MODEL=claude-3-sonnet-20240229
BATCH_SIZE=10
MAX_RETRIES=3

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/rag_system.log
EOF

# Create basic configuration files
print_status "Creating configuration files..."
cat > config/model_config.yaml << 'EOF'
embedding:
  model_name: "paraphrase-multilingual-MiniLM-L12-v2"
  device: "cpu"
  normalize_embeddings: true
  batch_size: 32

croatian_processing:
  preserve_diacritics: true
  normalize_spaces: true
  handle_encoding: "utf-8"
  sentence_splitter: "spacy"
EOF

cat > config/chunking_config.yaml << 'EOF'
chunking:
  strategies:
    semantic:
      max_chunk_size: 512
      overlap: 50
      respect_sentences: true
    fixed:
      max_chunk_size: 1024
      overlap: 100
    paragraph:
      preserve_structure: true
      min_chunk_size: 100

croatian_specific:
  sentence_endings: [".", "!", "?", "…"]
  preserve_quotes: true
  handle_dialogue: true
EOF

cat > config/api_config.yaml << 'EOF'
claude:
  model: "claude-3-sonnet-20240229"
  max_tokens: 1500
  temperature: 0.1
  batch_processing:
    enabled: true
    batch_size: 10
    max_concurrent: 5

prompts:
  system_prompt: |
    You are an expert assistant helping with questions about Croatian documents.
    Use the provided context to answer questions accurately and in Croatian when appropriate.
    If the context doesn't contain relevant information, say so clearly.

  rag_template: |
    Context: {context}

    Question: {question}

    Please provide a clear and accurate answer based on the context above.
EOF

# Create initial Python files with basic structure
print_status "Creating initial Python module structure..."

# Main init files
touch src/__init__.py
touch src/preprocessing/__init__.py
touch src/vectordb/__init__.py
touch src/retrieval/__init__.py
touch src/claude_api/__init__.py
touch src/pipeline/__init__.py
touch src/utils/__init__.py

# Create basic Croatian utilities
cat > src/utils/croatian_utils.py << 'EOF'
"""
Croatian language utilities for text processing.
"""
import re
from typing import List


class CroatianTextProcessor:
    """Utilities for processing Croatian text."""

    CROATIAN_CHARS = "ČčĆćŠšŽžĐđ"

    def __init__(self):
        self.diacritic_map = {
            'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'd',
            'Č': 'C', 'Ć': 'C', 'Š': 'S', 'Ž': 'Z', 'Đ': 'D'
        }

    def normalize_text(self, text: str) -> str:
        """Normalize Croatian text while preserving diacritics."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def has_croatian_chars(self, text: str) -> bool:
        """Check if text contains Croatian-specific characters."""
        return any(char in text for char in self.CROATIAN_CHARS)

    def remove_diacritics(self, text: str) -> str:
        """Remove Croatian diacritics (use sparingly for search)."""
        for croatian, latin in self.diacritic_map.items():
            text = text.replace(croatian, latin)
        return text
EOF

# Create basic configuration loader
cat > src/pipeline/config.py << 'EOF'
"""
Configuration management for Croatian RAG system.
"""
import os
import yaml
from pathlib import Path
from pydantic import BaseSettings
from typing import Dict, Any


class RAGConfig(BaseSettings):
    """Main configuration for RAG system."""

    # API Configuration
    anthropic_api_key: str = ""
    claude_model: str = "claude-3-sonnet-20240229"

    # Database Configuration
    chroma_db_path: str = "./data/vectordb"
    chroma_collection_name: str = "croatian_documents"

    # Embedding Configuration
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Processing Configuration
    max_chunk_size: int = 512
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}
EOF

# Create a basic test document
print_status "Creating test Croatian document..."
mkdir -p data/test
cat > data/test/sample_croatian.txt << 'EOF'
Zagreb je glavni grad Republike Hrvatske i najveći grad u zemlji.
Nalazi se na slivetišću rijeke Save i Medvednice na nadmorskoj visini od približno 122 metra.

Zagreb ima bogatu povijesnu baštinu koja seže u rimsko doba.
Današnji Zagreb nastao je 1850. godine spaјanjem gradova Gradeca i Kaptola.

Grad je poznat po svojoj arhitekturi, muzejima i kulturnim ustanovama.
Ban Jelačić trg je središnji trg grada i jedno od najpoznatijih mjesta u Zagrebu.

Sveučilište u Zagrebu osnovano je 1669. godine i jedno je od najstarijih sveučilišta u Europi.
EOF

# Create logs directory
mkdir -p logs

# Create basic README
cat > README.md << 'EOF'
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
EOF

# Create a simple test script
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify Croatian RAG setup is working correctly.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import chromadb
        import sentence_transformers
        import anthropic
        import spacy
        print("✅ Core packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_croatian_model():
    """Test Croatian spaCy model."""
    try:
        import spacy
        nlp = spacy.load("hr_core_news_sm")
        doc = nlp("Zagreb je glavni grad Hrvatske.")
        print(f"✅ Croatian spaCy model loaded: {len(doc)} tokens")
        return True
    except Exception as e:
        print(f"❌ Croatian model error: {e}")
        return False

def test_embeddings():
    """Test embedding model loading."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Test with Croatian text
        text = "Ovo je test rečenica na hrvatskom jeziku."
        embedding = model.encode(text)
        print(f"✅ Embedding model working: {len(embedding)} dimensions")
        return True
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return False

def test_file_structure():
    """Test that project structure is created correctly."""
    required_dirs = [
        "src/preprocessing", "src/vectordb", "src/retrieval",
        "src/claude_api", "src/pipeline", "data/raw", "data/processed"
    ]

    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)

    if missing:
        print(f"❌ Missing directories: {missing}")
        return False
    else:
        print("✅ Project structure created correctly")
        return True

if __name__ == "__main__":
    print("🧪 Testing Croatian RAG setup...\n")

    tests = [
        test_file_structure,
        test_imports,
        test_croatian_model,
        test_embeddings,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 Setup complete! Ready to start building your Croatian RAG system.")
        print("\nNext steps:")
        print("1. Add your Claude API key to .env file")
        print("2. Place Croatian documents in data/raw/")
        print("3. Follow the learning path in CLAUDE.md")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
EOF

# Make scripts executable
chmod +x test_setup.py

print_success "Croatian RAG project setup completed!"
print_status "Running setup verification..."
python test_setup.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy your Claude API key to .env: cp .env.template .env && nano .env"
echo "2. Add Croatian documents to data/raw/"
echo "3. Follow the learning progression in CLAUDE.md"
echo ""
echo "To activate environment: source venv/bin/activate"
echo "To test setup: python test_setup.py"
