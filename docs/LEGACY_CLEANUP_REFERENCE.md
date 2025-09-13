# RAG Service - Detailed Dependency Reference
## Complete File-by-File Dependency List (No Legacy)

*Generated for legacy cleanup - ordered from zero dependencies to high dependencies*

---

## 🟢 **ZERO DEPENDENCIES (45 files)**
*Foundation modules - safe to modify, no internal dependencies*

### Core System Files
- **`__init__.py`** → *no dependencies*
- **`cli/__init__.py`** → *no dependencies*
- **`generation/__init__.py`** → *no dependencies*
- **`pipeline/__init__.py`** → *no dependencies*
- **`preprocessing/__init__.py`** → *no dependencies*
- **`retrieval/__init__.py`** → *no dependencies*
- **`utils/__init__.py`** → *no dependencies*
- **`vectordb/__init__.py`** → *no dependencies*

### Generation Module Files
- **`generation/http_clients.py`** → *no dependencies*
- **`generation/language_providers.py`** → *no dependencies* [PROVIDER]
- **`generation/ollama_client.py`** → *no dependencies*
- **`generation/prompt_templates.py`** → *no dependencies*
- **`generation/response_parser.py`** → *no dependencies*

### Pipeline Module Files
- **`pipeline/config.py`** → *no dependencies*
- **`pipeline/rag_system.py`** → *no dependencies* (1005 lines - LARGE)

### Preprocessing Module Files
- **`preprocessing/chunkers.py`** → *no dependencies*
- **`preprocessing/cleaners.py`** → *no dependencies*
- **`preprocessing/cleaners_providers.py`** → *no dependencies* [PROVIDER]
- **`preprocessing/extractors.py`** → *no dependencies*
- **`preprocessing/extractors_providers.py`** → *no dependencies* [PROVIDER]
- **`preprocessing/extractors_working_backup.py`** → *no dependencies*

### Retrieval Module Files
- **`retrieval/categorization.py`** → *no dependencies* ⭐ **HIGH IMPACT** (2 dependents)
- **`retrieval/categorization_providers.py`** → *no dependencies* [PROVIDER]
- **`retrieval/hierarchical_retriever.py`** → *no dependencies* (908 lines - LARGE)
- **`retrieval/hierarchical_retriever_providers.py`** → *no dependencies* [PROVIDER]
- **`retrieval/hybrid_retriever.py`** → *no dependencies*
- **`retrieval/query_processor.py`** → *no dependencies*
- **`retrieval/ranker.py`** → *no dependencies* (891 lines - LARGE)
- **`retrieval/ranker_providers.py`** → *no dependencies* [PROVIDER]
- **`retrieval/reranker.py`** → *no dependencies*
- **`retrieval/retriever.py`** → *no dependencies*

### Utils Module Files
- **`utils/config_loader.py`** → *no dependencies* ⭐ **HIGH IMPACT** (2 dependents)
- **`utils/config_protocol.py`** → *no dependencies*
- **`utils/error_handler.py`** → *no dependencies*
- **`utils/folder_manager.py`** → *no dependencies* (774 lines - LARGE)
- **`utils/folder_manager_providers.py`** → *no dependencies* [PROVIDER]
- **`utils/language_manager.py`** → *no dependencies*
- **`utils/language_manager_providers.py`** → *no dependencies* [PROVIDER]

### Vectordb Module Files
- **`vectordb/embedding_devices.py`** → *no dependencies*
- **`vectordb/embedding_loaders.py`** → *no dependencies*
- **`vectordb/embeddings.py`** → *no dependencies*
- **`vectordb/search.py`** → *no dependencies*
- **`vectordb/search_providers.py`** → *no dependencies* [PROVIDER]
- **`vectordb/storage.py`** → *no dependencies*
- **`vectordb/storage_factories.py`** → *no dependencies*

---

## 🟡 **SINGLE DEPENDENCY (2 files)**
*Integration modules - check these after modifying their dependencies*

- **`cli/rag_cli.py`** → depends on: [`pipeline/rag_system.py`]
  - 852 lines | CLI interface
  - ⚠️ Check this after modifying `pipeline/rag_system.py`

- **`generation/enhanced_prompt_templates.py`** → depends on: [`retrieval/categorization.py`]
  - 681 lines | Template system
  - ⚠️ Check this after modifying `retrieval/categorization.py`

---

## 🟠 **TWO DEPENDENCIES (2 files)**
*Higher complexity - check these after modifying their dependencies*

- **`models/multitenant_models.py`** → depends on: [`retrieval/categorization.py`, `utils/config_loader.py`]
  - 425 lines | Data models
  - ⚠️ Check this after modifying either dependency

- **`generation/enhanced_prompt_templates_providers.py`** → depends on: [`generation/enhanced_prompt_templates.py`, `retrieval/categorization.py`]
  - 471 lines | DI Provider
  - ⚠️ Check this after modifying either dependency

---

## 🎯 **LEGACY CLEANUP STRATEGY**

### Phase 1: Clean Foundation (Zero Dependencies)
1. **Start with zero-dependency files** - safest to clean first
2. **Focus on high-impact files**: `categorization.py`, `config_loader.py`
3. **Remove legacy imports** from these 45 files
4. **Test each category** after cleanup

### Phase 2: Integration Layer (1-2 Dependencies)
1. **Clean single-dependency files** after their dependencies are clean
2. **Update two-dependency files** last
3. **Run full test suite** after each file

### Phase 3: Verification
1. **Search entire codebase** for any remaining `_legacy` imports
2. **Remove actual legacy files** after import cleanup
3. **Update documentation** and dependency graphs

---

## 🔍 **SEARCH COMMANDS FOR LEGACY CLEANUP**

```bash
# Find all legacy imports in each file
grep -r "_legacy" src/ --include="*.py"

# Find imports by specific legacy module
grep -r "from.*_legacy" src/ --include="*.py"
grep -r "import.*_legacy" src/ --include="*.py"

# Check specific files for legacy dependencies
grep "_legacy" src/utils/config_loader.py
grep "_legacy" src/retrieval/categorization.py

# Verify cleanup completion
find src/ -name "*_legacy.py" | wc -l  # Should be 0 after cleanup
```

---

*This reference shows the exact order to clean legacy dependencies: start with zero-dependency files, then work your way up to higher dependency files.*
