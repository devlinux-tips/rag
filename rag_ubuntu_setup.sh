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

# Install PyTorch (CUDA version for embeddings)
print_status "Installing PyTorch (CUDA version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Make scripts executable
chmod +x test_setup.py

print_success "Croatian RAG project setup completed!"
print_status "Running setup verification..."
python test_setup.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo ""
echo "To activate environment: source venv/bin/activate"
echo "To test setup: python test_setup.py"
