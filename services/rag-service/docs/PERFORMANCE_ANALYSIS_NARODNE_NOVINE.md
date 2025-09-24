# Narodne Novine Processing Performance Analysis

## Executive Summary

Performance analysis of processing 2,576 Croatian Narodne Novine HTML documents through the RAG system pipeline, identifying critical bottlenecks and optimization opportunities for large-scale document processing.

## Test Dataset

**Source:** Croatian Official Gazette (Narodne Novine) 2024
- **Document Count:** 2,576 HTML files
- **Document Structure:** Nested directories by issue number
- **Average File Size:** ~120KB per HTML document
- **Content Type:** Official legal documents in Croatian
- **Location:** `data/features/documents/narodne_novine/hr/2024/`

## Processing Commands

### Initial Command (Failed)
```bash
time python rag.py --tenant development --user dev_user --language hr process-docs data/features/documents/narodne_novine/hr/
```
**Result:** All 2,576 documents failed due to missing HTML format support
**Duration:** 13.6 seconds
**Issue:** `Unsupported format: .html`

### Fixed Command (Performance Test)
```bash
time python rag.py --tenant development --user dev_user --language hr process-docs data/features/documents/narodne_novine/hr/
```
**Result:** Processing started successfully but terminated due to performance
**Duration:** ~8 hours (terminated manually)
**Status:** Performance bottlenecks identified

## Technical Implementation

### HTML Format Support Added
**File:** `src/preprocessing/extractors.py`
**Changes:**
1. Added BeautifulSoup import
2. Created `_extract_html()` method
3. Updated format routing logic
4. Supports both `.html` and `.md` formats

```python
def _extract_html(self, file_path: Path) -> ExtractionResult:
    """Extract text from HTML using BeautifulSoup with encoding fallback."""
    html_binary = self._file_system.open_binary(file_path)
    html_content, encoding_used = extract_text_with_encoding_fallback(html_binary, self._config.text_encodings)

    # Parse HTML and extract text content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract text and clean up
    text = soup.get_text()
    processed_text = post_process_extracted_text(text)

    return ExtractionResult(
        text=processed_text,
        character_count=len(processed_text),
        extraction_method="HTML",
        encoding_used=encoding_used,
    )
```

### Configuration Updates
**File:** `config/config.toml`
**Existing Support:** HTML already listed in `supported_formats = [".pdf", ".docx", ".txt", ".html", ".md"]`

## Performance Results

### Successful Single Document Test
- **File:** `2024_01_10_188.html`
- **Characters Extracted:** 25,692
- **Chunks Created:** 48 meaningful chunks
- **Processing Time:** ~3 seconds total
- **Embedding Model:** BAAI/bge-m3 (1024-dim)

### Full Dataset Processing (Terminated)
- **Runtime:** ~8 hours before termination
- **Extraction Operations:** 4,472 completions logged
- **Documents Processed:** Partial (estimated ~300-500 documents)
- **Processing Rate:** ~1-2 documents per minute
- **Log Size:** 7.3MB processing log

## Identified Bottlenecks

### 1. Sequential Processing
**Issue:** Documents processed one at a time
**Impact:** No parallelization of I/O operations
**Evidence:** Single-threaded execution pattern in logs

### 2. Individual Embedding Operations
**Issue:** Each text chunk generates embeddings separately
**Impact:** Model inference called thousands of times
**Evidence:** `embedding_generator.generate_embeddings: STARTED` for every chunk

### 3. Database Transaction Overhead
**Issue:** Individual ChromaDB insertions per document/chunk
**Impact:** Network/disk I/O for each operation
**Evidence:** `chroma_collection.add: STARTED | doc_count=1` repeated

### 4. Model Loading Overhead
**Issue:** Potential model reinitialization
**Impact:** GPU/CPU memory management overhead
**Evidence:** Multiple model loading events in logs

### 5. Chunking Strategy Performance
**Issue:** Sliding window chunking with overlap processing
**Impact:** Additional text processing overhead
**Evidence:** 48 chunks per document × 2,576 documents = ~123,648 chunks

## Performance Metrics

### Current Performance
- **Documents per Hour:** ~60-120 (estimated)
- **Total Estimated Time:** 20-40 hours for full dataset
- **Memory Usage:** Stable (no memory leaks observed)
- **CPU Utilization:** Single-core usage pattern

### Target Performance Goals
- **Documents per Hour:** 1,000+ (17x improvement needed)
- **Total Processing Time:** <3 hours for full dataset
- **Parallel Processing:** Multi-core utilization
- **Batch Operations:** Reduced database overhead

## Optimization Recommendations

### 1. Batch Processing Implementation
**Priority:** High
**Impact:** 5-10x performance improvement
**Implementation:**
- Batch document extraction (10-50 documents)
- Batch embedding generation (100-500 chunks)
- Batch database operations (bulk inserts)

### 2. Parallel Processing
**Priority:** High
**Impact:** 4-8x performance improvement (CPU cores)
**Implementation:**
- Multi-process document extraction
- Parallel chunking operations
- Concurrent embedding generation

### 3. Database Optimization
**Priority:** Medium
**Impact:** 2-3x performance improvement
**Implementation:**
- ChromaDB bulk insert operations
- Connection pooling
- Transaction batching

### 4. Memory Optimization
**Priority:** Medium
**Impact:** Sustained performance, reduced memory usage
**Implementation:**
- Streaming document processing
- Chunk-based memory management
- Garbage collection optimization

### 5. Caching Strategy
**Priority:** Low
**Impact:** Reprocessing speed improvement
**Implementation:**
- Document hash-based caching
- Embedding result caching
- Processed chunk storage

## Recommended Architecture Changes

### Current Architecture
```
Document → Extract → Clean → Chunk → Embed → Store (Sequential)
```

### Optimized Architecture
```
Documents → [Batch Extract] → [Batch Clean] → [Batch Chunk] → [Batch Embed] → [Bulk Store]
     ↓
[Parallel Processing Pool]
     ↓
[Async Database Operations]
```

## Next Steps

### Immediate Actions
1. **Implement batch embedding generation**
2. **Add parallel document processing**
3. **Optimize ChromaDB operations**

### Medium-term Improvements
1. **Profile memory usage patterns**
2. **Implement processing progress tracking**
3. **Add resumable processing capability**

### Long-term Enhancements
1. **Distributed processing support**
2. **GPU acceleration for embeddings**
3. **Smart chunking strategies**

## Configuration Recommendations

### Processing Configuration
```toml
[processing]
batch_size = 50              # Documents per batch
parallel_workers = 8         # CPU cores - 1
embedding_batch_size = 100   # Chunks per embedding batch
chunk_batch_size = 500      # Database insert batch size
```

### Memory Management
```toml
[performance]
max_memory_mb = 8192        # 8GB memory limit
streaming_mode = true       # Process without full loading
garbage_collect_interval = 100  # Documents between GC
```

## Conclusion

The HTML extraction functionality has been successfully implemented and is working correctly. However, the current sequential processing architecture is not suitable for large document collections like the 2,576-file Narodne Novine dataset.

**Key Findings:**
- HTML processing works correctly (25,692 chars → 48 chunks)
- Performance bottleneck is in the processing pipeline, not HTML extraction
- Estimated 20-40 hours for full processing without optimization
- Batch processing and parallelization could achieve 17x performance improvement

**Recommendation:** Implement batch processing and parallel operations before attempting to process the full Narodne Novine dataset in production.

---

**Generated:** 2025-09-24T06:46:00Z
**Dataset:** Croatian Narodne Novine 2024 (2,576 HTML documents)
**Processing Status:** Performance optimization required