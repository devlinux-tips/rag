# PyTorch Device Setup Guide for Croatian RAG System

## Overview

This comprehensive guide covers PyTorch setup with CPU, CUDA (NVIDIA), and Apple Silicon (MPS) support for the Croatian RAG system with BGE-M3 embedding model.

## 🎯 Quick Start

Your Croatian RAG system supports automatic device detection and can run on:
- **CPU**: Works everywhere (current working setup)
- **CUDA**: NVIDIA GPUs (Windows/Linux)
- **MPS**: Apple Silicon (M1/M2/M3/M4 Pro/Max/Ultra)

## 🖥️ Current System Status

### Ubuntu 24.04 (Current Setup) ✅
- **OS**: Ubuntu 24.04
- **GPU**: NVIDIA T1200 Laptop GPU (4GB VRAM)
- **CUDA**: 12.2 (Driver: 535.247.01)
- **PyTorch**: 2.8.0+cu128 ✅
- **Status**: BGE-M3 working on CPU, CUDA ready after restart

### MacBook Pro M4 Pro (Target Setup) 🍎
- **Chip**: Apple M4 Pro (expected 14-core CPU, 20-core GPU)
- **Memory**: Unified memory architecture (16GB-48GB)
- **Acceleration**: Metal Performance Shaders (MPS)
- **Expected Performance**: 2-5x faster than CPU, similar to mid-range NVIDIA

## 🚀 Installation Guide

### Option 1: Universal PyTorch (Recommended for M4 Pro)
```bash
# Works on all platforms - detects MPS/CUDA automatically
pip install torch torchvision torchaudio

# For Croatian RAG system
cd /path/to/croatian-rag
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Option 2: CUDA-Specific (Ubuntu/NVIDIA)
```bash
# For CUDA 12.1/12.2 compatibility
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 3: CPU-Only (Any platform)
```bash
# Fallback for any system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🍎 Apple Silicon M4 Pro Setup

### Prerequisites
```bash
# Ensure you have Python 3.8+ and pip
python3 --version
pip3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### Installation for M4 Pro
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install Croatian RAG dependencies
pip install sentence-transformers>=2.2.0
pip install safetensors>=0.4.0
pip install chromadb
pip install transformers
```

### Test M4 Pro Setup
```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.mm(x, y)
    print(f'✅ MPS test successful: {z.shape}')
else:
    print('❌ MPS not available, using CPU')
"
```

## ⚙️ Configuration

### Automatic Device Detection (Recommended)
```toml
# config/config.toml
[embeddings]
model_name = "BAAI/bge-m3"
device = "auto"  # Detects: MPS > CUDA > CPU
batch_size = 32  # Adjusted automatically per device
max_seq_length = 512
use_safetensors = true
```

### Device-Specific Configurations

#### M4 Pro Optimized
```toml
[embeddings]
device = "mps"  # Force Apple Silicon GPU
batch_size = 64  # M4 Pro can handle large batches
max_seq_length = 512
torch_dtype = "float16"  # Memory optimization
```

#### NVIDIA GPU Optimized
```toml
[embeddings]
device = "cuda"
batch_size = 64  # Adjust based on VRAM
max_seq_length = 512
torch_dtype = "auto"
```

#### CPU Fallback
```toml
[embeddings]
device = "cpu"
batch_size = 16  # Smaller batches for CPU
max_seq_length = 512
torch_dtype = "float32"
```

## 📊 Performance Expectations

### M4 Pro Performance Estimates
| Task | CPU (M4 Pro) | MPS (M4 Pro) | Speedup |
|------|-------------|-------------|---------|
| **Model Loading** | 8-12s | 3-5s | 2-3x faster |
| **Text Encoding** | 60-100 texts/s | 150-300 texts/s | 2.5-5x faster |
| **Batch Processing** | 16 texts/batch | 64-128 texts/batch | 4-8x throughput |
| **Memory Usage** | System RAM | Unified Memory | More efficient |

### Current Ubuntu Performance
| Device | Performance | Memory | Notes |
|--------|------------|---------|-------|
| **CPU** | 31 texts/s | 2.3GB RAM | ✅ Working |
| **CUDA** | 100-200 texts/s | 2.3GB VRAM | After restart |

## 🔧 Device Detection & Switching

### Automatic Detection Logic
```python
def _get_device(self) -> str:
    """Device priority: MPS > CUDA > CPU"""
    if self.config.device == "auto":
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon (M1/M2/M3/M4)
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPUs
        else:
            return "cpu"  # Universal fallback
    return self.config.device
```

### Runtime Device Switching
```python
from src.vectordb.embeddings import CroatianEmbeddingModel

model = CroatianEmbeddingModel()
model.load_model()

# Check current device
print(f"Current device: {model.device}")

# Switch to different devices
model.switch_device("mps")   # Apple Silicon
model.switch_device("cuda")  # NVIDIA GPU
model.switch_device("cpu")   # CPU fallback
```

### Device Information
```python
# Get comprehensive device info
device_info = model.get_device_info()
print(f"MPS available: {device_info['mps_available']}")
print(f"CUDA available: {device_info['cuda_available']}")
print(f"Current device: {device_info['current_device']}")
```

## 🧪 Testing Your Setup

### Quick Test Script
```bash
# Test on any platform
cd /path/to/croatian-rag
source venv/bin/activate

python -c "
from src.vectordb.embeddings import CroatianEmbeddingModel
model = CroatianEmbeddingModel()
model.load_model()

# Test Croatian text
test_texts = [
    'Ovo je test tekst na hrvatskom jeziku.',
    'This is a test text in English.',
    'Hola, esto es una prueba en español.'
]

embeddings = model.encode_text(test_texts)
print(f'✅ Success! Device: {model.device}')
print(f'Embeddings shape: {embeddings.shape}')
print(f'Model info: {model.get_model_info()}')
"
```

### Comprehensive Device Testing
```bash
python test_devices.py
```

This will show:
- Available devices (CPU/MPS/CUDA)
- Performance benchmarks
- Memory usage
- Device switching capabilities

## 🍎 M4 Pro Specific Optimizations

### Memory Management
```python
# M4 Pro unified memory optimization
import torch

if torch.backends.mps.is_available():
    # Enable memory-efficient attention
    torch.backends.mps.empty_cache()  # Clear MPS cache

    # Use memory-mapped files for large models
    model_kwargs = {
        "torch_dtype": torch.float16,  # Half precision
        "device_map": "mps",
        "low_cpu_mem_usage": True
    }
```

### Batch Size Optimization
```python
# Adaptive batch sizing for M4 Pro
def get_optimal_batch_size(device, available_memory_gb):
    if device == "mps":
        # M4 Pro can handle larger batches
        if available_memory_gb >= 32:
            return 128  # 32GB+ unified memory
        elif available_memory_gb >= 16:
            return 64   # 16GB unified memory
        else:
            return 32   # 8GB unified memory
    elif device == "cuda":
        return min(64, available_memory_gb * 8)  # ~8 texts per GB VRAM
    else:
        return 16  # CPU conservative
```

### M4 Pro Performance Tuning
```toml
# Optimized config for M4 Pro
[embeddings]
device = "auto"
batch_size = 64
max_seq_length = 512
torch_dtype = "float16"  # Memory efficient
normalize_embeddings = true

[system]
# M4 Pro specific optimizations
use_metal_performance_shaders = true
memory_fraction = 0.8  # Use 80% of unified memory
enable_graph_optimization = true
```

## 🔧 Troubleshooting

### MPS Issues (Apple Silicon)
```bash
# Common MPS problems and solutions

# 1. MPS not available
python -c "import torch; print(torch.backends.mps.is_available())"
# Solution: Update to macOS 12.3+ and PyTorch 1.12+

# 2. Memory errors
# Solution: Reduce batch size or use float16
device = "mps"
batch_size = 32  # Reduce if memory errors

# 3. Model loading issues
# Solution: Clear MPS cache
torch.backends.mps.empty_cache()
```

### CUDA Issues (NVIDIA)
```bash
# CUDA initialization problems

# 1. Restart terminal (most common fix)
# Close and reopen terminal, then:
python -c "import torch; print(torch.cuda.is_available())"

# 2. Environment variables
export CUDA_VISIBLE_DEVICES=0
unset CUDA_VISIBLE_DEVICES  # Reset if needed

# 3. Driver compatibility
nvidia-smi  # Check driver version
```

### General Issues
```bash
# PyTorch version conflicts
pip install torch==2.8.0 torchvision torchaudio

# Model loading security errors
pip install safetensors>=0.4.0

# Memory issues
# Reduce batch_size in config.toml
```

## 📱 Platform Comparison

### M4 Pro vs Current Ubuntu Setup

| Aspect | Ubuntu + T1200 | MacBook Pro M4 Pro |
|--------|----------------|-------------------|
| **Setup Complexity** | Medium (CUDA drivers) | Easy (native support) |
| **Performance** | Good (CUDA) | Excellent (MPS) |
| **Memory** | 4GB VRAM separate | 16-48GB unified |
| **Power Efficiency** | Low | Very High |
| **Portability** | Desktop/laptop | Highly portable |
| **Development** | Good | Excellent |

### Recommended Configurations

#### For M4 Pro Development
```toml
[embeddings]
device = "auto"  # Will use MPS
batch_size = 64
torch_dtype = "float16"
```

#### For Ubuntu Production
```toml
[embeddings]
device = "auto"  # Will use CUDA after restart
batch_size = 32
torch_dtype = "auto"
```

## 🎯 Migration to M4 Pro

### Step 1: Clone Repository
```bash
git clone <your-repo>
cd croatian-rag
```

### Step 2: Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Test Setup
```bash
python test_devices.py
```

### Step 4: Optimize Configuration
```bash
# Edit config/config.toml for M4 Pro
device = "auto"  # Will detect MPS
batch_size = 64  # M4 Pro can handle larger batches
```

## 🏆 Expected M4 Pro Benefits

1. **Better Performance**: 2-5x faster than CPU, competitive with mid-range NVIDIA
2. **Unified Memory**: No separate VRAM limitations
3. **Power Efficiency**: Much lower power consumption
4. **Native Support**: No driver installation needed
5. **Development Experience**: Excellent for AI/ML development

## 📋 Final Recommendations

### For Current Ubuntu System
- ✅ Keep using CPU (works great)
- 🚀 Restart terminal to enable CUDA for 3-7x speedup
- ⚙️ Use `device = "auto"` for automatic detection

### For M4 Pro Migration
- 🍎 Use universal PyTorch installation
- ⚙️ Configure `device = "auto"` (will use MPS)
- 📊 Expect 2-5x performance improvement over CPU
- 🔧 Set `batch_size = 64` and `torch_dtype = "float16"`

Your Croatian RAG system is designed to work optimally on both platforms with automatic device detection! 🎉
