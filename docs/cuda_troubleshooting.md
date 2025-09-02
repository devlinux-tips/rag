# CUDA Troubleshooting Guide for Croatian RAG System

## Issue: CUDA Initialization Error

**Error Message:**
```
CUDA initialization: CUDA unknown error - this may be due to an incorrectly
set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after
program start. Setting the available devices to be zero.
```

## Quick Solutions

### Solution 1: Restart Terminal (Recommended)
```bash
# Close current terminal and open a new one, then:
cd /home/x/src/rag/learn-rag
source venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Solution 2: Clear CUDA Environment
```bash
# In current terminal:
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
cd /home/x/src/rag/learn-rag
source venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Solution 3: Force CUDA Reinitialization
```bash
cd /home/x/src/rag/learn-rag
source venv/bin/activate
python -c "
import os
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Solution 4: Verify CUDA/Driver Compatibility
```bash
# Check CUDA version compatibility
nvidia-smi  # Shows CUDA 12.2
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"  # Should show 12.1

# CUDA 12.1 (PyTorch) is compatible with CUDA 12.2 (Driver)
```

## Alternative: CPU-Only Configuration

If CUDA continues to have issues, the system works excellently with CPU:

### Update Configuration for CPU-Only
```toml
# config/config.toml
[embeddings]
model_name = "BAAI/bge-m3"
device = "cpu"  # Force CPU mode
batch_size = 16  # Smaller batch for CPU
max_seq_length = 512
```

### Performance Expectations (CPU vs CUDA)
- **CPU Performance**: ~28 texts/second (as demonstrated)
- **Expected CUDA Performance**: ~100-200 texts/second (3-7x speedup)
- **Memory Usage**: BGE-M3 uses ~2.3GB (CPU RAM or GPU VRAM)

## System Information
- **GPU**: NVIDIA T1200 Laptop GPU (4GB VRAM)
- **Driver**: 535.247.01
- **CUDA**: 12.2
- **PyTorch**: 2.5.1+cu121

## Recommended Next Steps

### Option A: Use CPU (Immediate Solution)
```bash
# Works now, good performance for development
cd /home/x/src/rag/learn-rag
source venv/bin/activate
python test_devices.py  # Will show CPU performance
```

### Option B: Fix CUDA (Better Performance)
```bash
# 1. Restart terminal completely
# 2. Run CUDA test again
cd /home/x/src/rag/learn-rag
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Option C: Hybrid Approach
Keep device setting as "auto" - the system will:
1. Try CUDA first (if available)
2. Fall back to CPU automatically
3. Log which device is being used

## Configuration Examples

### Automatic Device Detection (Recommended)
```toml
[embeddings]
device = "auto"  # Tries CUDA, falls back to CPU
batch_size = 32  # Will be adjusted based on device
```

### Force CPU (Always Works)
```toml
[embeddings]
device = "cpu"
batch_size = 16  # CPU-optimized batch size
```

### Force CUDA (If Fixed)
```toml
[embeddings]
device = "cuda"
batch_size = 64  # GPU can handle larger batches
```

## Testing Script

After any fixes, test with:
```bash
cd /home/x/src/rag/learn-rag
source venv/bin/activate
python test_devices.py
```

This will show:
- Available devices
- Performance benchmarks
- Memory usage
- Device switching capabilities

## Current Status: CPU Working Perfectly ✅

Your Croatian RAG system is fully functional with CPU processing:
- BGE-M3 model loads successfully
- ~28 texts/second encoding speed
- Automatic device detection working
- All embedding functionality operational

CUDA is a performance enhancement, not a requirement for functionality.
