# PyTorch CUDA Setup Guide for Croatian RAG System

## Overview

This guide explains how to set up PyTorch with both CPU and CUDA support for the Croatian RAG system, allowing you to switch between devices based on availability and requirements.

## Current System Information

- **OS**: Ubuntu 24.04
- **NVIDIA Driver**: 535.247.01
- **CUDA Version**: 12.2
- **Current PyTorch**: 2.8.0+cpu (CPU-only)

## CUDA Compatibility

Your system supports CUDA 12.2, which is compatible with PyTorch CUDA builds.

## Installation Options

### Option 1: Install CUDA-Enabled PyTorch (Recommended)

Replace the current CPU-only PyTorch with CUDA-enabled version:

```bash
# Activate virtual environment
source venv/bin/activate

# Uninstall current CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA-enabled PyTorch (for CUDA 12.1/12.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: Universal PyTorch Installation

Install PyTorch that automatically detects CUDA availability:

```bash
# Activate virtual environment
source venv/bin/activate

# Uninstall current version
pip uninstall torch torchvision torchaudio

# Install universal PyTorch
pip install torch torchvision torchaudio

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Device Configuration in Croatian RAG System

### Automatic Device Detection (Current Setup)

The embedding system automatically detects the best available device:

```python
def _get_device(self) -> str:
    """Determine the best available device."""
    if self.config.device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    return self.config.device
```

### Configuration Options

In `config/config.toml`, you can set:

```toml
[embeddings]
model_name = "BAAI/bge-m3"
device = "auto"  # Options: "auto", "cpu", "cuda", "cuda:0", "cuda:1", etc.
```

### Device Priority Order

1. **"auto"** (Recommended):
   - CUDA (if available) → MPS (Apple Silicon) → CPU
2. **"cuda"**: Force CUDA (will fail if not available)
3. **"cuda:0"**: Specific CUDA device
4. **"cpu"**: Force CPU (always works)

## Testing Device Setup

### Test Script

Create and run this test to verify your setup:

```bash
cd /home/x/src/rag/learn-rag
source venv/bin/activate

python -c "
import torch
from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig

print('=== Device Capabilities ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  {i}: {torch.cuda.get_device_name(i)}')

print()
print('=== Testing BGE-M3 Model ===')

# Test with different device settings
configs_to_test = [
    ('auto', 'auto'),
    ('cpu', 'cpu'),
]

if torch.cuda.is_available():
    configs_to_test.append(('cuda', 'cuda'))

for device_name, device_setting in configs_to_test:
    try:
        print(f'Testing device: {device_name}')

        # Create config with specific device
        config = EmbeddingConfig.from_config()
        config.device = device_setting

        model = CroatianEmbeddingModel(config)
        print(f'  Resolved device: {model.device}')

        # Load model and test encoding
        model.load_model()
        embeddings = model.encode_text(['Test text'])
        print(f'  ✓ Success - Embedding shape: {embeddings.shape}')

    except Exception as e:
        print(f'  ✗ Failed: {e}')

    print()
"
```

## Performance Considerations

### CUDA vs CPU Performance

| Aspect | CUDA | CPU |
|--------|------|-----|
| **Speed** | 5-20x faster | Baseline |
| **Memory** | GPU VRAM required | System RAM |
| **Batch Size** | Can handle larger batches | Limited by system RAM |
| **Model Loading** | Faster for large models | Slower initial loading |
| **Power Usage** | Higher GPU power | Lower power consumption |

### Memory Requirements

- **BGE-M3 Model**: ~2.3GB VRAM/RAM
- **Recommended**: 8GB+ VRAM for comfortable operation
- **Your GPU**: Check available memory with `nvidia-smi`

## Switching Between Devices

### Method 1: Configuration File

Edit `config/config.toml`:

```toml
[embeddings]
device = "cuda"  # or "cpu", "auto"
```

### Method 2: Environment Variable

```bash
export RAG_DEVICE="cuda"  # or "cpu"
```

Then modify the config loader to check environment variables.

### Method 3: Runtime Override

```python
from src.vectordb.embeddings import EmbeddingConfig, CroatianEmbeddingModel

# Override device at runtime
config = EmbeddingConfig.from_config()
config.device = "cuda"  # or "cpu"

model = CroatianEmbeddingModel(config)
```

## Troubleshooting

### CUDA Installation Issues

1. **CUDA not detected after installation**:
   ```bash
   # Check NVIDIA driver
   nvidia-smi

   # Reinstall PyTorch with explicit CUDA version
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Out of memory errors**:
   ```bash
   # Reduce batch size in config
   [embeddings]
   batch_size = 8  # Reduce from 32
   ```

3. **Model loading fails**:
   ```bash
   # Clear cache and retry
   rm -rf ./models/embeddings/*
   rm -rf ~/.cache/huggingface/
   ```

### Performance Optimization

1. **Mixed Precision** (CUDA only):
   ```toml
   [embeddings]
   torch_dtype = "float16"  # Reduces memory usage
   ```

2. **Batch Size Tuning**:
   - CPU: 8-16
   - CUDA (8GB VRAM): 32-64
   - CUDA (16GB+ VRAM): 64-128

## Recommended Setup for Your System

Given your hardware (CUDA 12.2), I recommend:

1. **Install CUDA PyTorch**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Use automatic device detection**:
   ```toml
   [embeddings]
   device = "auto"
   batch_size = 32
   ```

3. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

This setup will:
- Use CUDA when available for maximum performance
- Automatically fall back to CPU if CUDA has issues
- Provide optimal batch sizes for your hardware

## Next Steps

1. Choose your installation method (Option 1 recommended)
2. Install CUDA-enabled PyTorch
3. Test with the provided script
4. Adjust batch sizes based on your GPU memory
5. Monitor performance with `nvidia-smi`

Your Croatian RAG system will then automatically leverage GPU acceleration when available while maintaining CPU compatibility as a fallback.
