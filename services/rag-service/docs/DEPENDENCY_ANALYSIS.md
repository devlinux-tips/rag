# RAG Service Dependency Analysis Report
==================================================

## Summary Statistics
- **Total Modules**: 52
- **Legacy Modules**: 0
- **Provider Modules**: 11
- **Dependency Levels**: 5

## Module Categories
- **Cli**: 2 modules
- **Generation**: 8 modules
- **Models**: 1 modules
- **Pipeline**: 3 modules
- **Preprocessing**: 6 modules
- **Retrieval**: 12 modules
- **Root**: 1 modules
- **Utils**: 11 modules
- **Vectordb**: 8 modules

## Dependency Levels (Low to High Dependencies)

### Level 0 - 15 modules
*Modules with minimal external dependencies*

- **src.generation.__init__**
  - Category: Generation
  - Size: 0 lines

- **src.generation.prompt_templates**
  - Category: Generation
  - Size: 607 lines

- **src.generation.response_parser**
  - Category: Generation
  - Size: 665 lines

- **src.pipeline.__init__**
  - Category: Pipeline
  - Size: 0 lines

- **src.retrieval.__init__**
  - Category: Retrieval
  - Size: 0 lines

- **src.retrieval.retriever**
  - Category: Retrieval
  - Size: 703 lines

- **src.__init__**
  - Category: Root
  - Size: 0 lines

- **src.utils.__init__**
  - Category: Utils
  - Size: 0 lines

- **src.utils.config_loader**
  - Category: Utils
  - Size: 577 lines

- **src.utils.config_protocol**
  - Category: Utils
  - Size: 136 lines

- **src.utils.config_validator**
  - Category: Utils
  - Size: 711 lines

- **src.utils.error_handler**
  - Category: Utils
  - Size: 20 lines

- **src.utils.ocr_correction**
  - Category: Utils
  - Size: 179 lines

- **src.vectordb.__init__**
  - Category: Vectordb
  - Size: 0 lines

- **src.vectordb.search**
  - Category: Vectordb
  - Size: 716 lines


### Level 1 - 5 modules
*Modules depending on Level 0 and below*

- **src.generation.language_providers** [PROVIDER]
  - Category: Generation
  - Size: 116 lines
  - Dependencies: src.utils.config_loader

- **src.preprocessing.cleaners_providers** [PROVIDER]
  - Category: Preprocessing
  - Size: 419 lines
  - Dependencies: src.utils.config_loader

- **src.preprocessing.extractors_providers** [PROVIDER]
  - Category: Preprocessing
  - Size: 227 lines
  - Dependencies: src.utils.config_loader

- **src.utils.config_models**
  - Category: Utils
  - Size: 594 lines
  - Dependencies: src.utils.config_protocol

- **src.vectordb.search_providers** [PROVIDER]
  - Category: Vectordb
  - Size: 450 lines
  - Dependencies: src.utils.config_loader, src.vectordb.search


### Level 2 - 7 modules
*Modules depending on Level 1 and below*

- **src.pipeline.config**
  - Category: Pipeline
  - Size: 138 lines
  - Dependencies: src.utils.config_loader, src.utils.config_models, src.utils.config_protocol

- **src.pipeline.rag_system**
  - Category: Pipeline
  - Size: 848 lines
  - Dependencies: src.utils.config_loader, src.utils.config_models, src.utils.config_validator

- **src.preprocessing.cleaners**
  - Category: Preprocessing
  - Size: 722 lines
  - Dependencies: src.preprocessing.cleaners_providers

- **src.preprocessing.extractors**
  - Category: Preprocessing
  - Size: 378 lines
  - Dependencies: src.preprocessing.extractors_providers

- **src.retrieval.hybrid_retriever**
  - Category: Retrieval
  - Size: 797 lines
  - Dependencies: src.utils.config_models

- **src.retrieval.query_processor**
  - Category: Retrieval
  - Size: 625 lines
  - Dependencies: src.utils.config_models, src.utils.config_protocol

- **src.retrieval.reranker**
  - Category: Retrieval
  - Size: 699 lines
  - Dependencies: src.utils.config_models


### Level 3 - 2 modules
*Modules depending on Level 2 and below*

- **src.preprocessing.chunkers**
  - Category: Preprocessing
  - Size: 666 lines
  - Dependencies: src.preprocessing.cleaners, src.utils.config_protocol

- **src.retrieval.query_processor_providers** [PROVIDER]
  - Category: Retrieval
  - Size: 331 lines
  - Dependencies: src.retrieval.query_processor, src.utils.config_protocol


### Level 4 - 1 modules
*Modules depending on Level 3 and below*

- **src.preprocessing.__init__**
  - Category: Preprocessing
  - Size: 24 lines
  - Dependencies: src.preprocessing.chunkers, src.preprocessing.cleaners, src.preprocessing.extractors


## High-Level Architecture Overview

### Most Dependent Modules (Top 10)
1. **src.retrieval.hierarchical_retriever_providers** - 6 dependencies
2. **src.cli.rag_cli** - 4 dependencies
3. **src.generation.ollama_client** - 4 dependencies
4. **src.retrieval.ranker** - 4 dependencies
5. **src.retrieval.ranker_providers** - 4 dependencies
6. **src.utils.language_manager_providers** - 3 dependencies
7. **src.generation.enhanced_prompt_templates_providers** - 3 dependencies
8. **src.preprocessing.__init__** - 3 dependencies
9. **src.pipeline.config** - 3 dependencies
10. **src.pipeline.rag_system** - 3 dependencies

### Most Depended Upon Modules (Top 10)
1. **src.utils.config_loader** - 14 dependents
2. **src.utils.config_models** - 7 dependents
3. **src.utils.config_protocol** - 7 dependents
4. **src.retrieval.categorization** - 6 dependents
5. **src.preprocessing.cleaners** - 4 dependents
6. **src.utils.config_validator** - 2 dependents
7. **src.generation.ollama_client** - 2 dependents
8. **src.preprocessing.extractors** - 2 dependents
9. **src.retrieval.query_processor** - 2 dependents
10. **src.retrieval.categorization_providers** - 2 dependents

## Detailed Dependency Listing
*Ordered from no dependencies to high dependencies*

### 📋 Complete Module Dependency Reference
*Format: `filename.py` → depends on: [`dependency1.py`, `dependency2.py`]*

#### 🟢 Zero Dependencies (Foundation Modules)

- **`__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Root

- **`generation/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Generation

- **`generation/prompt_templates.py`** → depends on: *none*
  - Size: 607 lines | Category: Generation
  - External imports: stdlib: dataclasses, logging, typing

- **`generation/response_parser.py`** → depends on: *none*
  - Size: 665 lines | Category: Generation
  - External imports: stdlib: dataclasses, logging, re, typing, unicodedata

- **`pipeline/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Pipeline

- **`retrieval/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Retrieval

- **`retrieval/retriever.py`** → depends on: *none*
  - Size: 703 lines | Category: Retrieval
  - External imports: stdlib: asyncio, dataclasses, enum, logging, time, typing

- **`utils/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Utils

- **`utils/config_loader.py`** → depends on: *none*
  - Size: 577 lines | Category: Utils
  - External imports: stdlib: logging, pathlib, tomllib, typing

- **`utils/config_protocol.py`** → depends on: *none*
  - Size: 136 lines | Category: Utils
  - External imports: stdlib: typing

- **`utils/config_validator.py`** → depends on: *none*
  - Size: 711 lines | Category: Utils
  - External imports: stdlib: dataclasses, logging, typing

- **`utils/error_handler.py`** → depends on: *none*
  - Size: 20 lines | Category: Utils
  - External imports: stdlib: logging

- **`utils/ocr_correction.py`** → depends on: *none*
  - Size: 179 lines | Category: Utils
  - External imports: stdlib: re, typing

- **`vectordb/__init__.py`** → depends on: *none*
  - Size: 0 lines | Category: Vectordb

- **`vectordb/search.py`** → depends on: *none*
  - Size: 716 lines | Category: Vectordb
  - External imports: stdlib: asyncio, dataclasses, enum, logging, numpy, time, typing

#### 🟡 Single Dependency (Integration Modules)

- **`generation/language_providers.py`** → depends on: [`utils/config_loader.py`]
  - Size: 116 lines | Category: Generation [PROVIDER]
  - External imports: stdlib: typing | 3rd-party: ..utils.config_loader

- **`preprocessing/cleaners_providers.py`** → depends on: [`utils/config_loader.py`]
  - Size: 419 lines | Category: Preprocessing [PROVIDER]
  - External imports: stdlib: locale, logging, os, typing | 3rd-party: ..utils.config_loader

- **`preprocessing/extractors_providers.py`** → depends on: [`utils/config_loader.py`]
  - Size: 227 lines | Category: Preprocessing [PROVIDER]
  - External imports: stdlib: logging, pathlib, typing | 3rd-party: ..utils.config_loader

- **`utils/config_models.py`** → depends on: [`utils/config_protocol.py`]
  - Size: 594 lines | Category: Utils
  - External imports: stdlib: dataclasses, enum, logging, typing | 3rd-party: ..utils.config_protocol

- **`preprocessing/cleaners.py`** → depends on: [`preprocessing/cleaners_providers.py`]
  - Size: 722 lines | Category: Preprocessing
  - External imports: stdlib: dataclasses, locale, re, typing, unicodedata | 3rd-party: .cleaners_providers

- **`preprocessing/extractors.py`** → depends on: [`preprocessing/extractors_providers.py`]
  - Size: 378 lines | Category: Preprocessing
  - External imports: stdlib: dataclasses, docx, io, pathlib, pypdf, typing | 3rd-party: .extractors_providers

- **`retrieval/hybrid_retriever.py`** → depends on: [`utils/config_models.py`]
  - Size: 797 lines | Category: Retrieval
  - External imports: stdlib: dataclasses, logging, numpy, re, typing | 3rd-party: ..utils.config_models

- **`retrieval/reranker.py`** → depends on: [`utils/config_models.py`]
  - Size: 699 lines | Category: Retrieval
  - External imports: stdlib: dataclasses, logging, numpy, random, typing | 3rd-party: ..utils.config_models

#### 🟠 Two Dependencies

- **`vectordb/search_providers.py`** → depends on: [`utils/config_loader.py`, `vectordb/search.py`]
  - Size: 450 lines | Category: Vectordb [PROVIDER]
  - External imports: stdlib: asyncio, logging, numpy, sentence_transformers, typing | 3rd-party: ..utils.config_loader, .search

- **`retrieval/query_processor.py`** → depends on: [`utils/config_models.py`, `utils/config_protocol.py`]
  - Size: 625 lines | Category: Retrieval
  - External imports: stdlib: dataclasses, enum, logging, re, typing | 3rd-party: ..utils.config_models, ..utils.config_protocol

- **`preprocessing/chunkers.py`** → depends on: [`preprocessing/cleaners.py`, `utils/config_protocol.py`]
  - Size: 666 lines | Category: Preprocessing
  - External imports: stdlib: dataclasses, enum, logging, pathlib, re, typing | 3rd-party: ..utils.config_protocol, .cleaners

- **`retrieval/query_processor_providers.py`** → depends on: [`retrieval/query_processor.py`, `utils/config_protocol.py`]
  - Size: 331 lines | Category: Retrieval [PROVIDER]
  - External imports: stdlib: typing | 3rd-party: ..utils.config_protocol, .query_processor

#### 🔴 3 Dependencies (High Complexity)

- **`pipeline/config.py`** → depends on: [`utils/config_loader.py`, `utils/config_models.py`, `utils/config_protocol.py`]
  - Size: 138 lines | Category: Pipeline
  - External imports: stdlib: pathlib, pydantic_settings, typing, yaml | 3rd-party: ..utils.config_loader, ..utils.config_models, ..utils.config_protocol

- **`pipeline/rag_system.py`** → depends on: [`utils/config_loader.py`, `utils/config_models.py`, `utils/config_validator.py`]
  - Size: 848 lines | Category: Pipeline
  - External imports: stdlib: dataclasses, logging, pathlib, time, typing | 3rd-party: ..utils.config_loader, ..utils.config_models, ..utils.config_validator

- **`preprocessing/__init__.py`** → depends on: [`preprocessing/chunkers.py`, `preprocessing/cleaners.py`, `preprocessing/extractors.py`]
  - Size: 24 lines | Category: Preprocessing
  - External imports: 3rd-party: .chunkers, .cleaners, .extractors

---

### 📊 Quick Reference
- **Total files analyzed**: 30
- **Zero dependencies**: 15 files
- **Single dependency**: 8 files
- **Multiple dependencies**: 7 files

## Recommendations

### 🎯 Refactoring Priorities
1. **Legacy Modules**: Consider migrating or removing legacy modules
2. **High Dependency Modules**: Review modules with many dependencies for simplification
3. **Core Dependencies**: Ensure stability of highly depended-upon modules

### 📐 Architecture Insights
- **Level 0 modules** are foundational and should be most stable
- **Provider modules** implement dependency injection patterns
- **Legacy modules** indicate areas needing modernization
