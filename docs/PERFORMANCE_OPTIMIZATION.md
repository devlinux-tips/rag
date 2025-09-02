# Croatian RAG System - LLM Performance Optimization Guide

## 🚀 Current Performance Analysis

### 📊 Current Metrics (from last test)
- **Generation Time**: 123.16s (extremely slow)
- **Retrieval Time**: 0.13s (excellent)
- **Current Model**: `jobautomation/openeurollm-croatian:latest` (8.1 GB)
- **Available Models**: qwen2.5:7b-instruct (4.7 GB), qwen2.5:32b (19 GB), llama3.1:8b (4.9 GB)

### 🔍 Root Cause Analysis
The 123s generation time indicates **severe performance bottlenecks**. This is unusually slow even for CPU-only inference.

## 🎯 Immediate Optimizations (Quick Wins)

### 1. **Switch to Faster Model** ⚡
**Problem**: `jobautomation/openeurollm-croatian:latest` (8.1 GB) is significantly larger than alternatives
**Solution**: Use smaller, faster models with Croatian capability

```toml
# config/config.toml - IMMEDIATE CHANGE
[ollama]
model = "qwen2.5:7b-instruct"  # 4.7 GB instead of 8.1 GB
# model = "llama3.1:8b"        # Alternative: 4.9 GB
```

**Expected Impact**: 3-5x speed improvement (40-30s generation time)

### 2. **Optimize Generation Parameters** 🔧
**Current config analysis**:
```toml
max_tokens = 2000      # Too high - Croatian responses rarely need this
top_k = 64            # Too high - increases computation
num_predict = -1      # Unlimited - should be capped
```

**Optimized config**:
```toml
[ollama]
model = "qwen2.5:7b-instruct"
temperature = 0.3          # Lower = faster, more focused
max_tokens = 800           # Sufficient for most Croatian answers
top_p = 0.85              # Slightly lower for speed
top_k = 40                # Reduce from 64
timeout = 60.0            # Reduce from 120s
num_predict = 800         # Cap token generation
repeat_penalty = 1.05     # Lower for speed
```

**Expected Impact**: 2-3x speed improvement

### 3. **Enable Aggressive Streaming** 📡
**Current**: Streaming enabled but may not be optimized
**Optimization**:
```toml
[ollama]
stream = true
keep_alive = "1m"         # Reduce from "5m"
```

**Expected Impact**: Better perceived performance, faster response start

### 4. **Reduce Context Window** 📝
**Current**: Potentially sending too much context
**Check current context size**:
```python
# In your last query, context was 4346 characters
# This is quite large and slows generation
```

**Optimization**: Limit context in retrieval config:
```toml
[retrieval]
default_k = 3             # Reduce from 5
max_context_length = 2000 # Add explicit limit
```

## 🖥️ Hardware-Specific Optimizations

### Option A: **CPU Optimization** (Current Setup)
```bash
# 1. Ensure Ollama uses all CPU cores
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1

# 2. Start Ollama with optimizations
OLLAMA_HOST=0.0.0.0:11434 \
OLLAMA_ORIGINS=* \
OLLAMA_NUM_PARALLEL=4 \
ollama serve
```

### Option B: **GPU Acceleration** (Your Desktop)
```bash
# 1. Check GPU memory
nvidia-smi

# 2. Restart Ollama with GPU support
ollama serve

# 3. Verify GPU usage during generation
watch -n 1 nvidia-smi
```

**Expected GPU speedup**: 5-10x faster (12-6s generation time)

## 🔧 Advanced Optimizations

### 1. **Model Quantization** 📉
**Problem**: Current model may not be quantized
**Solution**: Use quantized versions

```bash
# Check if model supports quantization
ollama show jobautomation/openeurollm-croatian:latest

# Try quantized Croatian model (if available)
ollama pull jobautomation/openeurollm-croatian:q4_k_m
ollama pull jobautomation/openeurollm-croatian:q8_0
```

### 2. **Concurrent Processing** ⚡
**Current**: Single-threaded generation
**Optimization**: Enable parallel processing

```toml
[system]
max_concurrent_requests = 3    # Instead of 5
request_timeout = 30.0         # Reduce from 120s
```

### 3. **Response Caching** 💾
**Implement simple caching for repeated queries**:

```python
# Add to RAG system
import hashlib
from typing import Dict

class ResponseCache:
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size

    def get_cache_key(self, query: str, context: List[str]) -> str:
        content = f"{query}|{'|'.join(context[:3])}"  # First 3 chunks
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, context: List[str]) -> Optional[str]:
        key = self.get_cache_key(query, context)
        return self.cache.get(key)

    def set(self, query: str, context: List[str], response: str):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        key = self.get_cache_key(query, context)
        self.cache[key] = response
```

### 4. **Prompt Optimization** 📝
**Problem**: Croatian cultural context adds overhead
**Solution**: Streamline prompts for speed

```python
# config/croatian.toml - Optimized prompts
[prompts]
# Shorter, more direct prompts
system_prompt = "Odgovori kratko i precizno na hrvatskom jeziku koristeći dani kontekst."
user_prompt = "Pitanje: {query}\n\nKontekst: {context}\n\nOdgovor:"

# Disable heavy cultural context for speed
[generation]
include_cultural_context = false  # Disable for speed
prefer_formal_style = false       # Disable for speed
preserve_diacritics = true        # Keep this - minimal overhead
```

## 🚀 Implementation Priority

### **IMMEDIATE (< 5 minutes)** 🔥
1. **Change model** to `qwen2.5:7b-instruct`
2. **Reduce max_tokens** to 800
3. **Lower temperature** to 0.3
4. **Reduce top_k** to 40

### **SHORT TERM (< 30 minutes)** ⚡
1. **Optimize Ollama startup** with CPU/GPU flags
2. **Reduce retrieval context** (k=3, max_context=2000)
3. **Disable cultural context** for speed testing
4. **Implement basic caching**

### **MEDIUM TERM (< 2 hours)** 🔧
1. **Test quantized models**
2. **Implement response caching**
3. **Optimize prompts** for conciseness
4. **Add concurrent processing**

## 📊 Expected Performance Improvements

### **Conservative Estimates**:
| Optimization | Current Time | Target Time | Speedup |
|-------------|-------------|-------------|---------|
| **Model Change** | 123s | 40s | 3x |
| **+ Parameter Tuning** | 40s | 20s | 2x |
| **+ Context Reduction** | 20s | 15s | 1.3x |
| **+ Prompt Optimization** | 15s | 10s | 1.5x |
| **+ GPU (if available)** | 10s | 2-5s | 3-5x |

### **Aggressive Targets**:
- **CPU-only**: 8-12 seconds (10x improvement)
- **With GPU**: 2-4 seconds (30x improvement)

## 🔥 Quick Implementation Script

Create this optimization script:

```bash
#!/bin/bash
# quick_optimize.sh

echo "🚀 Optimizing Croatian RAG Performance..."

# 1. Backup current config
cp config/config.toml config/config.toml.backup

# 2. Apply optimizations
sed -i 's/model = "jobautomation\/openeurollm-croatian:latest"/model = "qwen2.5:7b-instruct"/' config/config.toml
sed -i 's/max_tokens = 2000/max_tokens = 800/' config/config.toml
sed -i 's/temperature = 0.7/temperature = 0.3/' config/config.toml
sed -i 's/top_k = 64/top_k = 40/' config/config.toml
sed -i 's/timeout = 120.0/timeout = 60.0/' config/config.toml

# 3. Add num_predict limit
echo 'num_predict = 800' >> config/config.toml

# 4. Test performance
echo "🧪 Testing optimized performance..."
python -m src.pipeline.rag_system --query "Kratko objašnjenje o dokumentu?"

echo "✅ Optimization complete!"
echo "📊 Check generation time in output above"
```

## 🎯 Monitoring Performance

### **Add Performance Tracking**:
```python
# In rag_system.py, add detailed timing
import time

class PerformanceTracker:
    def __init__(self):
        self.timings = {}

    def start_timer(self, operation: str):
        self.timings[operation] = time.time()

    def end_timer(self, operation: str) -> float:
        start_time = self.timings.get(operation, time.time())
        duration = time.time() - start_time
        print(f"⏱️  {operation}: {duration:.2f}s")
        return duration
```

### **Key Metrics to Track**:
- **Model Loading Time**: Should be < 3s
- **Prompt Processing**: Should be < 0.1s
- **Token Generation Rate**: Target > 20 tokens/second
- **Memory Usage**: Monitor for leaks

## 🔮 Advanced Strategies (Future)

### **1. Model Compilation** (vLLM/TensorRT)
```bash
# For production deployment
pip install vllm
# Use vLLM for 3-5x additional speedup
```

### **2. Speculative Decoding**
```bash
# Use smaller model to guide larger model
ollama pull qwen2.5:1.5b  # Draft model
# Configure speculative decoding in Ollama
```

### **3. Batch Processing**
```python
# Process multiple queries simultaneously
async def batch_generate(queries: List[str]) -> List[str]:
    tasks = [ollama_client.generate_text_async(q) for q in queries]
    return await asyncio.gather(*tasks)
```

## 💡 Conclusion

The current 123s generation time is **not normal** and indicates configuration issues rather than hardware limitations. With the optimizations above, you should achieve:

- **Immediate improvement**: 40-50s (model + config changes)
- **With tuning**: 10-15s (acceptable for production)
- **With GPU**: 2-5s (excellent performance)

**Recommended immediate action**: Change model to `qwen2.5:7b-instruct` and reduce `max_tokens` to 800. This alone should provide dramatic improvement.

The Croatian language quality will remain excellent with these optimizations while achieving much better performance! 🚀
