#!/bin/bash

# Croatian RAG Learning Project Setup Script
# macOS (Apple Silicon M4 Pro) with Python 3.13 + uv + Homebrew
# Optimized for Apple Silicon performance with MPS acceleration

set -e  # Exit on any error

echo "🚀 Setting up Croatian RAG Learning Project on macOS (Apple Silicon M4 Pro)..."
echo "Using: Python 3.13 + uv + Homebrew + MPS acceleration"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_m4() {
    echo -e "${PURPLE}[M4 PRO]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run as root. Run as regular user."
    exit 1
fi

# Verify we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    print_error "This script is designed for Apple Silicon Macs (M1/M2/M3/M4)"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_error "Homebrew is required but not installed."
    echo "Install with: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_error "uv is required but not found in PATH."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check Python 3.13
if ! uv python list | grep -q "3.13"; then
    print_warning "Python 3.13 not found with uv. Installing..."
    uv python install 3.13
fi

print_success "✅ Prerequisites verified: Homebrew + uv + Python 3.13"

# Install system dependencies optimized for M4 Pro
print_status "Installing macOS system dependencies with Homebrew..."
print_m4 "Optimizing for Apple Silicon M4 Pro performance..."

# Core build tools and libraries (Apple Silicon optimized)
brew install \
    cmake \
    pkg-config \
    openblas \
    lapack \
    hdf5 \
    sqlite \
    poppler \
    tesseract \
    tesseract-lang

# Document processing tools
print_status "Installing document processing tools..."
brew install \
    pandoc \
    imagemagick \
    ghostscript

# Optional: LibreOffice for advanced document processing
print_status "Installing LibreOffice for document conversion..."
brew install --cask libreoffice

print_success "✅ System dependencies installed"

# Create uv project and virtual environment
print_status "Setting up uv project with Python 3.13..."
if [ ! -f "pyproject.toml" ]; then
    print_error "No pyproject.toml found. Please run from rag-service directory."
    exit 1
fi

print_success "✅ Found pyproject.toml with modern Python project configuration"

# Install dependencies with uv (much faster than pip)
print_status "Installing Python dependencies with uv..."
print_m4 "Using uv's parallel installation for M4 Pro speed..."

# Install PyTorch with MPS support (Apple Silicon GPU acceleration)
print_status "Installing PyTorch with MPS (Apple Silicon GPU) support..."
uv add torch torchvision torchaudio

# Install the requirements
if [ -f "requirements.txt" ]; then
    print_status "Installing from requirements.txt..."
    uv sync
else
    print_warning "No requirements.txt found, installing core dependencies..."
    
    # Core RAG dependencies
    uv add \
        sentence-transformers \
        chromadb \
        requests \
        numpy \
        pandas \
        python-docx \
        PyPDF2 \
        toml \
        pydantic \
        fastapi \
        uvicorn \
        pytest \
        black \
        ruff \
        mypy
fi

# Install Ollama for local LLM (optimized for Apple Silicon)
print_status "Installing Ollama for local LLM..."
if ! command -v ollama &> /dev/null; then
    print_m4 "Installing Ollama with Apple Silicon optimization..."
    brew install ollama
    
    # Start Ollama service
    print_status "Starting Ollama service..."
    brew services start ollama
    
    # Wait for service to start
    sleep 3
    
    # Pull Croatian-optimized model
    print_status "Pulling qwen2.5:7b-instruct model (Croatian support)..."
    ollama pull qwen2.5:7b-instruct
else
    print_success "✅ Ollama already installed"
fi

# Verify MPS (Apple Silicon GPU) availability
print_status "Verifying Apple Silicon MPS support..."
uv run python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('🚀 Apple Silicon GPU acceleration ready!')
else:
    print('⚠️  MPS not available, will use CPU')
"

# Make scripts executable
if [ -f "test_setup.py" ]; then
    chmod +x test_setup.py
fi

print_success "✅ Croatian RAG project setup completed!"

# Run setup verification if available
if [ -f "test_setup.py" ]; then
    print_status "Running setup verification..."
    uv run python test_setup.py
fi

echo ""
print_success "🎉 macOS M4 Pro setup complete!"
echo ""
print_m4 "Apple Silicon optimizations enabled:"
echo "  • MPS GPU acceleration for embeddings"
echo "  • Ollama with ARM64 optimization"
echo "  • uv for lightning-fast package management"
echo "  • Homebrew ARM64 optimized dependencies"
echo ""
echo "📋 Next steps:"
echo "  • Activate environment: uv shell"
echo "  • Test setup: uv run python test_setup.py"
echo "  • Run CLI: uv run python -m src.cli.rag_cli --help"
echo ""
echo "🔧 Development commands:"
echo "  • Add packages: uv add <package>"
echo "  • Run scripts: uv run <script>"
echo "  • Format code: uv run python format_code.py"
echo "  • Type check: uv run mypy src/"
echo ""
print_m4 "M4 Pro Performance Tips:"
echo "  • Use MPS device for embeddings (automatic)"
echo "  • Ollama uses ARM64 optimized models"
echo "  • uv manages dependencies 10-100x faster than pip"
echo "  • All Homebrew packages are Apple Silicon native"