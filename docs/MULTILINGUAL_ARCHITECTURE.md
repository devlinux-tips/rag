# Multilingual RAG System Architecture

## 📁 Folder Structure Strategy

### Current Implementation (September 2025)

```
data/
├── raw/                          # Source documents organized by language
│   ├── hr/                      # Croatian documents
│   │   ├── 110 - 11.docx       # Legal/financial docs
│   │   ├── 110 - 8.docx
│   │   ├── NN - 2025 - 115 - 1666.pdf
│   │   ├── NN - 2025 - 116 - 1671.pdf
│   │   └── NN - 2025 - 116 - 1683.pdf
│   ├── en/                      # English documents
│   │   └── (future English docs)
│   └── multilingual/            # Mixed-language documents
├── test/                        # Test data by language
│   ├── hr/
│   │   └── sample_croatian.txt
│   ├── en/
│   │   └── sample_english.txt
│   └── multilingual/            # Cross-language test scenarios
├── processed/                   # Processed data cache (language-specific)
│   ├── hr/                     # Croatian processed data
│   ├── en/                     # English processed data
│   └── shared/                 # Cross-language resources
├── chromadb/                   # Vector database storage
│   └── multilingual_documents/ # Unified collection for all languages
└── vectordb/                   # Alternative vector storage
```

## 🎯 Benefits of Language-Based Organization

### 1. **Scalability**
- Easy to add new languages (de/, fr/, es/, etc.)
- Clear separation of language-specific content
- Independent processing pipelines per language

### 2. **Processing Efficiency**
- Batch process documents by language
- Language-specific optimization settings
- Parallel processing capabilities

### 3. **Maintenance & Operations**
- Easy to identify and update language-specific content
- Clear audit trail for document sources
- Language-specific backup and recovery

### 4. **Development & Testing**
- Isolated test environments per language
- Language-specific unit tests
- Cross-language integration testing

## 🔧 Implementation Considerations

### Document Processing Pipeline
```python
# Language-aware document processing
def process_documents_by_language(language_code: str):
    source_dir = f"data/raw/{language_code}/"
    processed_dir = f"data/processed/{language_code}/"

    # Process with language-specific settings
    extractor = DocumentExtractor(language=language_code)
    cleaner = MultilingualTextCleaner(language=language_code)
    chunker = DocumentChunker(language=language_code)
```

### Vector Storage Strategy

#### Option A: Unified Collection (Current)
- Single "multilingual_documents" collection
- Language metadata in each chunk
- Cross-language semantic search possible

#### Option B: Language-Specific Collections
- Separate collections per language
- Language-specific optimization
- Independent scaling per language

```python
# Collection naming strategy
collection_names = {
    'hr': 'croatian_documents',
    'en': 'english_documents',
    'multilingual': 'multilingual_documents'
}
```

## 📊 Recommended Practices

### 1. **Document Ingestion**
```bash
# Language-specific ingestion
python -m src.pipeline.rag_system --add-docs data/raw/hr/ --lang hr
python -m src.pipeline.rag_system --add-docs data/raw/en/ --lang en
```

### 2. **Batch Processing**
```bash
# Process all Croatian documents
python scripts/batch_process.py --language hr --input data/raw/hr/

# Process all English documents
python scripts/batch_process.py --language en --input data/raw/en/
```

### 3. **Quality Assurance**
```bash
# Language-specific testing
python -m pytest tests/test_language_hr.py
python -m pytest tests/test_language_en.py

# Cross-language testing
python -m pytest tests/test_multilingual.py
```

## 🚀 Future Expansions

### Additional Languages
```
data/raw/
├── de/          # German
├── fr/          # French
├── es/          # Spanish
├── it/          # Italian
└── pt/          # Portuguese
```

### Domain-Specific Organization
```
data/raw/hr/
├── legal/       # Legal documents
├── financial/   # Financial reports
├── technical/   # Technical documentation
└── general/     # General content
```

### Advanced Features (Detailed Implementation)

#### 🌐 1. Cross-Language Document Linking
**Purpose:** Connect related documents across languages (same content, different languages)

```python
# Enhanced metadata for cross-language linking
class CrossLanguageLinker:
    def __init__(self):
        self.similarity_threshold = 0.85

    def link_documents(self, doc_hr: str, doc_en: str) -> dict:
        """Link Croatian and English versions of same document"""
        return {
            "link_id": "legal_2025_115",
            "documents": {
                "hr": "data/raw/hr/NN-2025-115-1666.pdf",
                "en": "data/raw/en/NN-2025-115-1666_EN.pdf"
            },
            "link_confidence": 0.95,
            "content_type": "legal_financial"
        }
```

**Implementation Steps:**
1. **Document fingerprinting** - Extract key identifiers (dates, amounts, names)
2. **Cross-language similarity matching** - Use BGE-M3 embeddings to find similar content
3. **Metadata enhancement** - Add link references to vector storage

#### 🔄 2. Translation Capabilities
**Purpose:** Provide on-demand translation for queries and responses

```python
# Translation service integration
class TranslationService:
    def __init__(self):
        self.local_translator = "Helsinki-NLP/opus-mt"  # Offline option
        self.api_translator = "google_translate"        # Online option

    async def translate_query(self, query: str, source_lang: str, target_lang: str):
        """Translate user query to enable cross-language search"""
        if source_lang == "hr" and target_lang == "en":
            # "Koliki je ukupni iznos?" → "What is the total amount?"
            return await self.translate(query, "hr-en")
```

**Use Cases:**
- **Query Translation:** User asks in Croatian, searches English documents
- **Response Translation:** System finds English content, responds in Croatian
- **Document Preview:** Show snippet translations in query results

#### 📊 3. Language-Specific Analytics
**Purpose:** Provide insights into multilingual content usage and performance

```python
# Analytics dashboard for multilingual RAG
class MultilingualAnalytics:
    def track_query_patterns(self) -> dict:
        """Analyze which languages are most queried"""
        return {
            "query_distribution": {
                "hr": 0.65,  # 65% Croatian queries
                "en": 0.30,  # 30% English queries
                "cross_lang": 0.05  # 5% cross-language queries
            },
            "popular_topics_by_lang": {
                "hr": ["pravni", "financijski", "nekretnine"],
                "en": ["legal", "financial", "real_estate"]
            }
        }
```

**Dashboard Features:**
- **Query Language Distribution** - Which languages users prefer
- **Content Gap Analysis** - Missing translations or language coverage
- **Performance by Language** - Response quality metrics per language
- **Cross-Language Usage** - How often users search across languages

#### 🤖 4. Automated Language Detection for Ingestion
**Purpose:** Automatically organize uploaded documents by detected language

```python
# Smart document organization
class LanguageDetector:
    def process_upload(self, file_path: str) -> str:
        """Auto-detect language and move to appropriate folder"""
        content = self.extract_text(file_path)
        detected_lang = self.detect_language(content)

        if detected_lang == "hr":
            target_dir = "data/raw/hr/"
        elif detected_lang == "en":
            target_dir = "data/raw/en/"
        else:
            target_dir = "data/raw/multilingual/"

        return self.move_file(file_path, target_dir)
```

#### 🎯 5. Smart Query Enhancement
**Purpose:** Improve search results using language-aware techniques

```python
# Enhanced query processing
class SmartQueryProcessor:
    def enhance_query(self, query: str, language: str) -> dict:
        """Enhance query with language-specific improvements"""
        enhancements = {
            "original": query,
            "expanded": self.add_synonyms(query, language),
            "translated": self.get_translation_variants(query),
            "search_strategy": self.determine_strategy(query, language)
        }
        return enhancements
```

### 🚀 Implementation Priority & Timeline

**Phase 1 (Week 1-2): Foundation**
- ✅ Language-based folder structure (DONE)
- 🔄 Enhanced analytics tracking
- 📊 Basic language detection

**Phase 2 (Week 3-4): Smart Features**
- 🔗 Cross-language document linking
- 🔍 Smart query enhancement
- 📈 Analytics dashboard

**Phase 3 (Week 5-6): Advanced Capabilities**
- 🌐 Translation service integration
- 🤖 Automated language detection
- 🎯 Cross-language search optimization

### 💡 Quick Wins Available Now
1. **Add language detection to batch processor**
2. **Create analytics tracking in query pipeline**
3. **Enhance document metadata with language confidence scores**

## 🔧 Configuration Updates Needed

### 1. Update document paths in configs
### 2. Add language-specific processing settings
### 3. Configure collection naming strategy
### 4. Update CLI tools for language folders

## 📈 Performance Benefits

1. **Faster Processing**: Language-specific optimizations
2. **Better Organization**: Clear content management
3. **Scalable Architecture**: Easy to add new languages
4. **Improved Testing**: Language-specific test suites
5. **Operational Excellence**: Clear deployment and maintenance procedures

---

*This architecture supports the transition from a Croatian-specific system to a fully multilingual RAG platform while maintaining backward compatibility and operational efficiency.*
