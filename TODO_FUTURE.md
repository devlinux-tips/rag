## 📋 TODO: Performance & Feature Enhancements

### **🚀 High Priority Performance Optimizations**

#### **Response Caching System**
- **Priority**: High 🔥
- **Status**: Not implemented
- **Implementation Strategy**:
  ```python
  # Implement Redis/Memory-based caching for repeated queries
  class ResponseCache:
      def __init__(self, max_size: int = 100, ttl: int = 3600):
          self.cache: Dict[str, CachedResponse] = {}
          self.max_size = max_size
          self.ttl = ttl  # Time to live in seconds

      def get_cache_key(self, query: str, context_hash: str) -> str:
          # Create unique cache key from query + context fingerprint

      def get_cached_response(self, query: str, context: List[str]) -> Optional[RAGResponse]:
          # Return cached response if available and not expired

      def cache_response(self, query: str, context: List[str], response: RAGResponse):
          # Store response with timestamp and context hash
  ```
- **Expected Benefit**: 95%+ speedup for repeated queries
- **Use Cases**: FAQ queries, document summaries, common Croatian phrases
- **Implementation Time**: 2-4 hours

#### **Parallel Processing for Multiple Queries**
- **Priority**: High 🔥
- **Status**: Not implemented
- **Implementation Strategy**:
  ```python
  # Implement async batch processing with concurrent limits
  class BatchQueryProcessor:
      def __init__(self, max_concurrent: int = 3):
          self.semaphore = asyncio.Semaphore(max_concurrent)

      async def process_batch(self, queries: List[RAGQuery]) -> List[RAGResponse]:
          # Process multiple queries concurrently with rate limiting
          tasks = [self._process_single_query(q) for q in queries]
          return await asyncio.gather(*tasks, return_exceptions=True)

      async def _process_single_query(self, query: RAGQuery) -> RAGResponse:
          async with self.semaphore:
              # Single query processing with resource management
  ```
- **Expected Benefit**: 2-3x throughput for multiple queries
- **Use Cases**: Batch document analysis, multi-user scenarios
- **Implementation Time**: 3-5 hours


## **🎯 Platform Development Roadmap**

### **Phase 1C: Simple Multi-User (1 week)**
**Goal**: Multiple users can use system independently
- **Minimal user system**: Basic login, document ownership
- **PostgreSQL**: Essential user/document tables only
- **Result**: Multi-user Croatian RAG system

### **Phase 2: Enhancement Layer (2-4 weeks)**
**Goal**: Production-scale capabilities (parallel development)
- **Elixir job orchestration**: Background processing, rate limiting, config validator schema checker, Embedding model dimension mismatch validator on startup and chuncks/vector data recreation
- **Real-time updates**: WebSocket job progress
- **Result**: Production-ready platform

### **Phase 3: Advanced Features (when needed)**
**Goal**: Scale and sophistication as user base grows
- **Multi-tenancy**: When multiple organizations need isolation
- **Advanced analytics**: When usage patterns justify complexity
- **Mobile apps**: When mobile access becomes priority
- **Enterprise features**: When enterprise customers require them


### **🔧 System Architecture Improvements**

#### **Production Monitoring**
- **Priority**: Medium ⚡
- **Features**:
  - Performance metrics dashboard
  - Query analytics and patterns
  - System health monitoring
  - Error tracking and alerting

#### **Configuration Management Enhancement**
- **Priority**: Low 🔧
- **Strategy**: Implement runtime configuration updates without restart
- **Benefit**: Better development and production flexibility

#### **Testing & CI/CD**
- **Priority**: Medium ⚡
- **Strategy**: Comprehensive test suite for Croatian language processing
- **Coverage**: Unit tests, integration tests, performance benchmarks

#### **Documentation & Examples**
- **Priority**: Low 🔧
- **Strategy**: Enhanced documentation with Croatian use cases
- **Content**: API docs, deployment guides, Croatian language examples

### **📊 Performance Benchmarks & Targets**

#### **Current Performance (Post-Optimization)**
- **Generation Time**: 83.5s (CPU, qwen2.5:7b-instruct)
- **Retrieval Time**: 0.12s (excellent)
- **Croatian Quality**: ✅ Excellent
- **Memory Usage**: ~3-4GB (GPU has 13GB available)

#### **Target Performance Goals**
- **With Caching**: < 1s for repeated queries (95% of FAQ use cases)
- **With GPU**: 8-15s generation time (5-10x improvement)
- **With Batch Processing**: 2-3x concurrent query throughput
- **With Quantization**: 40-60s generation time (additional 30-50% improvement)


--- NO IDEA ;)

### **🔧 Common Refactoring Issues & Solutions**

#### **Issue 1: TOML Configuration Structure**
**Problem**: Templates placed after `[prompts.keywords]` become part of keywords subsection
```toml
# ❌ Wrong: Templates after subsection are inaccessible
[prompts]
base_setting = "value"

[prompts.keywords]
cultural = ["hrvatski", "kultura"]

# These templates become part of [prompts.keywords] scope!
cultural_context_system = "Ti si stručnjak..."  # INVISIBLE!

# ✅ Correct: Templates before any subsections
[prompts]
base_setting = "value"
cultural_context_system = "Ti si stručnjak..."  # ACCESSIBLE!

[prompts.keywords]
cultural = ["hrvatski", "kultura"]
```

#### **Issue 2: Refactoring Inconsistencies**
**Problem**: Attribute name changes not propagated throughout codebase
```python
# Old pattern (before refactoring)
retrieval_result.results[0].content

# New pattern (after refactoring)
retrieval_result.documents[0]["content"]
```

#### **Issue 3: Mixed Configuration Sources**
**Problem**: Components using wrong config sources after refactoring
```python
# ✅ Language-specific components should use language-specific config
language_config = get_language_settings(language="hr")  # or "en", "multilingual"
templates = language_config["prompts"]

# ❌ Not general generation config
general_config = get_generation_config()  # Missing language-specific templates
```
