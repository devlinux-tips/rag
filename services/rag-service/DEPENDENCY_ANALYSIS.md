# RAG Service Dependency Analysis Report
==================================================

## Summary Statistics
- **Total Modules**: 50
- **Legacy Modules**: 0
- **Provider Modules**: 11
- **Dependency Levels**: 5

## Module Categories
- **Cli**: 2 modules
- **Generation**: 8 modules
- **Models**: 1 modules
- **Pipeline**: 3 modules
- **Preprocessing**: 7 modules
- **Retrieval**: 12 modules
- **Root**: 1 modules
- **Utils**: 8 modules
- **Vectordb**: 8 modules

## Dependency Levels (Low to High Dependencies)

### Level 0 - 17 modules
*Modules with minimal external dependencies*

- **src.generation.__init__**
  - Category: Generation
  - Size: 0 lines

- **src.generation.http_clients**
  - Category: Generation
  - Size: 260 lines

- **src.generation.prompt_templates**
  - Category: Generation
  - Size: 652 lines

- **src.generation.response_parser**
  - Category: Generation
  - Size: 698 lines

- **src.pipeline.__init__**
  - Category: Pipeline
  - Size: 0 lines

- **src.pipeline.rag_system**
  - Category: Pipeline
  - Size: 1005 lines

- **src.retrieval.__init__**
  - Category: Retrieval
  - Size: 0 lines

- **src.retrieval.hybrid_retriever**
  - Category: Retrieval
  - Size: 760 lines

- **src.retrieval.reranker**
  - Category: Retrieval
  - Size: 700 lines

- **src.retrieval.retriever**
  - Category: Retrieval
  - Size: 705 lines

- **src.__init__**
  - Category: Root
  - Size: 0 lines

- **src.utils.__init__**
  - Category: Utils
  - Size: 0 lines

- **src.utils.config_loader**
  - Category: Utils
  - Size: 537 lines

- **src.utils.config_protocol**
  - Category: Utils
  - Size: 135 lines

- **src.utils.error_handler**
  - Category: Utils
  - Size: 83 lines

- **src.vectordb.__init__**
  - Category: Vectordb
  - Size: 0 lines

- **src.vectordb.search**
  - Category: Vectordb
  - Size: 739 lines


### Level 1 - 7 modules
*Modules depending on Level 0 and below*

- **src.cli.rag_cli**
  - Category: Cli
  - Size: 852 lines
  - Dependencies: src.pipeline.rag_system

- **src.pipeline.config**
  - Category: Pipeline
  - Size: 463 lines
  - Dependencies: src.utils.config_loader, src.utils.config_protocol

- **src.preprocessing.cleaners_providers** [PROVIDER]
  - Category: Preprocessing
  - Size: 446 lines
  - Dependencies: src.utils.config_loader

- **src.preprocessing.extractors_providers** [PROVIDER]
  - Category: Preprocessing
  - Size: 226 lines
  - Dependencies: src.utils.config_loader

- **src.preprocessing.extractors_working_backup**
  - Category: Preprocessing
  - Size: 133 lines
  - Dependencies: src.utils.config_loader

- **src.retrieval.query_processor**
  - Category: Retrieval
  - Size: 710 lines
  - Dependencies: src.utils.config_protocol

- **src.vectordb.search_providers** [PROVIDER]
  - Category: Vectordb
  - Size: 491 lines
  - Dependencies: src.utils.config_loader, src.vectordb.search


### Level 2 - 4 modules
*Modules depending on Level 1 and below*

- **src.cli.__init__**
  - Category: Cli
  - Size: 7 lines
  - Dependencies: src.cli.rag_cli

- **src.preprocessing.cleaners**
  - Category: Preprocessing
  - Size: 694 lines
  - Dependencies: src.preprocessing.cleaners_providers

- **src.preprocessing.extractors**
  - Category: Preprocessing
  - Size: 409 lines
  - Dependencies: src.preprocessing.extractors_providers

- **src.retrieval.query_processor_providers** [PROVIDER]
  - Category: Retrieval
  - Size: 411 lines
  - Dependencies: src.retrieval.query_processor, src.utils.config_loader


### Level 3 - 1 modules
*Modules depending on Level 2 and below*

- **src.preprocessing.chunkers**
  - Category: Preprocessing
  - Size: 691 lines
  - Dependencies: src.preprocessing.cleaners, src.utils.config_protocol


### Level 4 - 1 modules
*Modules depending on Level 3 and below*

- **src.preprocessing.__init__**
  - Category: Preprocessing
  - Size: 25 lines
  - Dependencies: src.preprocessing.chunkers, src.preprocessing.cleaners, src.preprocessing.extractors


## High-Level Architecture Overview

### Most Dependent Modules (Top 10)
1. **src.retrieval.hierarchical_retriever_providers** - 5 dependencies
2. **src.retrieval.ranker_providers** - 4 dependencies
3. **src.generation.enhanced_prompt_templates_providers** - 3 dependencies
4. **src.generation.ollama_client** - 3 dependencies
5. **src.preprocessing.__init__** - 3 dependencies
6. **src.retrieval.hierarchical_retriever** - 3 dependencies
7. **src.utils.language_manager_providers** - 2 dependencies
8. **src.utils.folder_manager** - 2 dependencies
9. **src.utils.folder_manager_providers** - 2 dependencies
10. **src.generation.enhanced_prompt_templates** - 2 dependencies

### Most Depended Upon Modules (Top 10)
1. **src.utils.config_loader** - 12 dependents
2. **src.retrieval.categorization** - 6 dependents
3. **src.utils.config_protocol** - 4 dependents
4. **src.preprocessing.cleaners** - 4 dependents
5. **src.retrieval.query_processor** - 2 dependents
6. **src.retrieval.categorization_providers** - 2 dependents
7. **src.vectordb.embeddings** - 2 dependents
8. **src.utils.language_manager_providers** - 1 dependents
9. **src.utils.language_manager** - 1 dependents
10. **src.utils.error_handler** - 1 dependents

## Detailed Dependency Listing
*Ordered from no dependencies to high dependencies*

### 📋 Complete Module Dependency Reference
*Format: `filename.py` → depends on: [`dependency1.py`, `dependency2.py`]*

#### 🟢 Zero Dependencies (Foundation Modules)

- **`__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Root

- **`generation/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Generation

- **`generation/http_clients.py`** → depends on: *none*
  - Size: 260 lines | Category: Generation
  - External imports: stdlib: asyncio, httpx, json, requests, typing | 3rd-party: .ollama_client_v2

- **`generation/prompt_templates.py`** → depends on: *none*
  - Size: 652 lines | Category: Generation
  - External imports: stdlib: abc, dataclasses, logging, typing

- **`generation/response_parser.py`** → depends on: *none*
  - Size: 698 lines | Category: Generation
  - External imports: stdlib: dataclasses, logging, re, typing, unicodedata

- **`pipeline/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Pipeline

- **`pipeline/rag_system.py`** → depends on: *none*
  - Size: 1005 lines | Category: Pipeline
  - External imports: stdlib: asyncio, dataclasses, hashlib, json, pathlib, time, typing

- **`retrieval/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Retrieval

- **`retrieval/hybrid_retriever.py`** → depends on: *none*
  - Size: 760 lines | Category: Retrieval
  - External imports: stdlib: dataclasses, logging, numpy, re, typing

- **`retrieval/reranker.py`** → depends on: *none*
  - Size: 700 lines | Category: Retrieval
  - External imports: stdlib: dataclasses, logging, numpy, random, typing

- **`retrieval/retriever.py`** → depends on: *none*
  - Size: 705 lines | Category: Retrieval
  - External imports: stdlib: asyncio, dataclasses, enum, logging, numpy, time, typing

- **`utils/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Utils

- **`utils/config_loader.py`** → depends on: *none*
  - Size: 537 lines | Category: Utils
  - External imports: stdlib: logging, pathlib, tomli, tomllib, typing

- **`utils/config_protocol.py`** → depends on: *none*
  - Size: 135 lines | Category: Utils
  - External imports: stdlib: typing

- **`utils/error_handler.py`** → depends on: *none*
  - Size: 83 lines | Category: Utils
  - External imports: stdlib: logging, typing

- **`vectordb/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Vectordb

- **`vectordb/search.py`** → depends on: *none*
  - Size: 739 lines | Category: Vectordb
  - External imports: stdlib: asyncio, dataclasses, enum, logging, numpy, time, typing

#### 🟡 Single Dependency (Integration Modules)

- **`cli/rag_cli.py`** → depends on: [`pipeline/rag_system.py`]
  - Size: 852 lines | Category: Cli
  - External imports: stdlib: argparse, asyncio, dataclasses, logging, pathlib, sys, time, typing

- **`preprocessing/cleaners_providers.py`** → depends on: [`utils/config_loader.py`]
  - Size: 446 lines | Category: Preprocessing [PROVIDER]
  - External imports: stdlib: locale, logging, os, typing | 3rd-party: ..utils.config_loader

- **`preprocessing/extractors_providers.py`** → depends on: [`utils/config_loader.py`]
  - Size: 226 lines | Category: Preprocessing [PROVIDER]
  - External imports: stdlib: logging, pathlib, typing | 3rd-party: ..utils.config_loader

- **`preprocessing/extractors_working_backup.py`** → depends on: [`utils/config_loader.py`]
  - Size: 133 lines | Category: Preprocessing
  - External imports: stdlib: docx, logging, pathlib, pypdf, typing | 3rd-party: ..utils.config_loader

- **`retrieval/query_processor.py`** → depends on: [`utils/config_protocol.py`]
  - Size: 710 lines | Category: Retrieval
  - External imports: stdlib: dataclasses, enum, logging, re, typing | 3rd-party: ..utils.config_protocol

- **`cli/__init__.py`** → depends on: [`cli/rag_cli.py`]
  - Size: 7 lines | Category: Cli
  - External imports: 3rd-party: .rag_cli

- **`preprocessing/cleaners.py`** → depends on: [`preprocessing/cleaners_providers.py`]
  - Size: 694 lines | Category: Preprocessing
  - External imports: stdlib: dataclasses, locale, logging, os, pathlib, re, typing, unicodedata | 3rd-party: .cleaners_providers

- **`preprocessing/extractors.py`** → depends on: [`preprocessing/extractors_providers.py`]
  - Size: 409 lines | Category: Preprocessing
  - External imports: stdlib: dataclasses, docx, io, logging, pathlib, pypdf, typing | 3rd-party: .extractors_providers

#### 🟠 Two Dependencies

- **`pipeline/config.py`** → depends on: [`utils/config_loader.py`, `utils/config_protocol.py`]
  - Size: 463 lines | Category: Pipeline
  - External imports: stdlib: dataclasses, os, pathlib, pydantic, pydantic_settings, typing, yaml | 3rd-party: ..utils.config_loader, ..utils.config_protocol

- **`vectordb/search_providers.py`** → depends on: [`utils/config_loader.py`, `vectordb/search.py`]
  - Size: 491 lines | Category: Vectordb [PROVIDER]
  - External imports: stdlib: asyncio, logging, numpy, sentence_transformers, typing | 3rd-party: ..utils.config_loader, .search

- **`retrieval/query_processor_providers.py`** → depends on: [`retrieval/query_processor.py`, `utils/config_loader.py`]
  - Size: 411 lines | Category: Retrieval [PROVIDER]
  - External imports: stdlib: logging, typing | 3rd-party: ..utils.config_loader, .query_processor

- **`preprocessing/chunkers.py`** → depends on: [`preprocessing/cleaners.py`, `utils/config_protocol.py`]
  - Size: 691 lines | Category: Preprocessing
  - External imports: stdlib: dataclasses, enum, logging, pathlib, re, typing | 3rd-party: ..utils.config_protocol, .cleaners

#### 🔴 3 Dependencies (High Complexity)

- **`preprocessing/__init__.py`** → depends on: [`preprocessing/chunkers.py`, `preprocessing/cleaners.py`, `preprocessing/extractors.py`]
  - Size: 25 lines | Category: Preprocessing
  - External imports: 3rd-party: .chunkers, .cleaners, .extractors

---

### 📊 Quick Reference
- **Total files analyzed**: 30
- **Zero dependencies**: 17 files
- **Single dependency**: 8 files
- **Multiple dependencies**: 5 files

## Recommendations

### 🎯 Refactoring Priorities
1. **Legacy Modules**: Consider migrating or removing legacy modules
2. **High Dependency Modules**: Review modules with many dependencies for simplification
3. **Core Dependencies**: Ensure stability of highly depended-upon modules

### 📐 Architecture Insights
- **Level 0 modules** are foundational and should be most stable
- **Provider modules** implement dependency injection patterns
- **Legacy modules** indicate areas needing modernization
