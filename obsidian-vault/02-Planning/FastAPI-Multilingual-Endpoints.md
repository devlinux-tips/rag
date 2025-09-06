# FastAPI Multilingual Endpoints Specification

**Created**: 2025-09-06
**Phase**: Phase 1A
**Priority**: High
**Component**: FastAPI Backend Wrapper

## 🎯 Overview

Comprehensive specification for FastAPI endpoints that wrap the existing Python RAG system with full multilingual support, language parameter handling, and seamless integration with the React multilingual frontend.

## 🌐 API Architecture

### **Language Parameter Strategy**

#### **Input Language Parameter** (`input_lang`)
Controls which language collection to search and how to process queries
- **Values**: `hr` (Croatian) | `en` (English) | `multilingual` (all languages)
- **Default**: `multilingual`
- **Usage**: Query parameter, request body field, or header

#### **Interface Language Parameter** (`ui_lang`)
Controls response language for messages, errors, and metadata
- **Values**: `hr` (Croatian) | `en` (English)
- **Default**: `en`
- **Usage**: `Accept-Language` header or query parameter

#### **Content Language Response** (`Content-Language`)
Indicates the primary language of the response content
- **Auto-detected**: Based on search results and input language
- **Header**: `Content-Language: hr` or `Content-Language: en`

## 📡 API Endpoints Specification

### **1. Health Check Endpoint**

```http
GET /health
```

**Response** (multilingual):
```json
{
  "status": "healthy",
  "message": {
    "en": "RAG system is operational",
    "hr": "RAG sustav je operativan"
  },
  "components": {
    "rag_system": "healthy",
    "embeddings": "healthy",
    "vector_db": "healthy",
    "llm": "healthy"
  },
  "languages_supported": ["hr", "en", "multilingual"],
  "timestamp": "2025-09-06T10:30:00Z"
}
```

### **2. Language Detection Endpoint**

```http
POST /detect-language
Content-Type: application/json

{
  "text": "Što je to sustav za dohvaćanje povećan generiranjem?"
}
```

**Response**:
```json
{
  "detected_language": "hr",
  "confidence": 0.95,
  "supported_languages": {
    "hr": 0.95,
    "en": 0.05,
    "multilingual": 0.90
  },
  "recommendation": {
    "input_lang": "hr",
    "message": {
      "en": "Croatian detected with high confidence",
      "hr": "Hrvatski jezik detektiran s visokom sigurnošću"
    }
  }
}
```

### **3. Document Upload Endpoint**

```http
POST /documents/upload?input_lang=hr&ui_lang=hr
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx]
language: hr (optional, overrides input_lang for these docs)
auto_detect: true (optional)
```

**Response**:
```json
{
  "upload_id": "upload_123456",
  "status": "processing",
  "message": {
    "en": "Documents uploaded successfully, processing started",
    "hr": "Dokumenti uspješno učitani, obrada pokrenuta"
  },
  "files": [
    {
      "filename": "document1.pdf",
      "size": 1024000,
      "detected_language": "hr",
      "confidence": 0.92,
      "status": "processing"
    },
    {
      "filename": "document2.docx",
      "size": 512000,
      "detected_language": "multilingual",
      "confidence": 0.78,
      "status": "processing"
    }
  ],
  "processing_time_estimate": {
    "en": "Approximately 2-3 minutes",
    "hr": "Približno 2-3 minute"
  }
}
```

### **4. Upload Status Endpoint**

```http
GET /documents/upload/{upload_id}/status?ui_lang=hr
```

**Response**:
```json
{
  "upload_id": "upload_123456",
  "status": "completed",
  "progress": {
    "total_files": 2,
    "processed_files": 2,
    "failed_files": 0,
    "percentage": 100
  },
  "message": {
    "en": "All documents processed successfully",
    "hr": "Svi dokumenti uspješno obrađeni"
  },
  "results": {
    "total_chunks": 45,
    "embeddings_generated": 45,
    "storage_language": "hr"
  }
}
```

### **5. Search Endpoint**

```http
POST /search?input_lang=hr&ui_lang=hr
Content-Type: application/json

{
  "query": "Što je RAG sustav?",
  "top_k": 5,
  "filters": {
    "document_type": ["pdf", "docx"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2025-09-06"
    }
  }
}
```

**Response**:
```json
{
  "query": "Što je RAG sustav?",
  "input_language": "hr",
  "ui_language": "hr",
  "results": [
    {
      "id": "doc_chunk_001",
      "content": "RAG (Retrieval-Augmented Generation) sustav je...",
      "source": {
        "filename": "rag_introduction.pdf",
        "page": 1,
        "language": "hr"
      },
      "metadata": {
        "relevance_score": 0.92,
        "language_confidence": 0.95,
        "chunk_language": "hr"
      }
    },
    {
      "id": "doc_chunk_045",
      "content": "Retrieval-Augmented Generation systems combine...",
      "source": {
        "filename": "multilingual_docs.pdf",
        "page": 3,
        "language": "multilingual"
      },
      "metadata": {
        "relevance_score": 0.87,
        "language_confidence": 0.82,
        "chunk_language": "en"
      }
    }
  ],
  "summary": {
    "total_results": 2,
    "languages_found": ["hr", "en"],
    "processing_time_ms": 120,
    "message": {
      "en": "Found results in Croatian and English",
      "hr": "Pronađeni rezultati na hrvatskom i engleskom"
    }
  }
}
```

### **6. Query with Generation Endpoint**

```http
POST /query?input_lang=hr&ui_lang=hr
Content-Type: application/json

{
  "query": "Objasni kako funkcionira RAG sustav",
  "generate_answer": true,
  "retrieval_options": {
    "top_k": 5,
    "use_reranking": true
  },
  "generation_options": {
    "max_length": 500,
    "temperature": 0.7
  }
}
```

**Response**:
```json
{
  "query": "Objasni kako funkcionira RAG sustav",
  "input_language": "hr",
  "ui_language": "hr",
  "answer": {
    "text": "RAG (Retrieval-Augmented Generation) sustav kombinira pretraživanje dokumenata s generiranjem odgovora...",
    "language": "hr",
    "confidence": 0.89
  },
  "sources": [
    {
      "id": "doc_chunk_001",
      "content": "RAG sustav je napredna tehnologija...",
      "source": "rag_tutorial.pdf",
      "relevance_score": 0.94,
      "language": "hr"
    }
  ],
  "metadata": {
    "retrieval_time_ms": 85,
    "generation_time_ms": 2340,
    "total_time_ms": 2425,
    "model_used": "qwen2.5:7b-instruct",
    "sources_used": 3
  }
}
```

### **7. Available Languages Endpoint**

```http
GET /languages?ui_lang=hr
```

**Response**:
```json
{
  "supported_languages": {
    "input_languages": [
      {
        "code": "hr",
        "name": {
          "en": "Croatian",
          "hr": "Hrvatski"
        },
        "document_count": 156,
        "chunk_count": 2340
      },
      {
        "code": "en",
        "name": {
          "en": "English",
          "hr": "Engleski"
        },
        "document_count": 89,
        "chunk_count": 1567
      },
      {
        "code": "multilingual",
        "name": {
          "en": "Multilingual",
          "hr": "Višejezično"
        },
        "document_count": 45,
        "chunk_count": 678
      }
    ],
    "interface_languages": [
      {
        "code": "hr",
        "name": {
          "en": "Croatian",
          "hr": "Hrvatski"
        },
        "completeness": 100
      },
      {
        "code": "en",
        "name": {
          "en": "English",
          "hr": "Engleski"
        },
        "completeness": 100
      }
    ]
  },
  "default_language": "multilingual",
  "message": {
    "en": "Language information retrieved successfully",
    "hr": "Informacije o jezicima uspješno dohvaćene"
  }
}
```

## 🔧 Implementation Details

### **Language Detection Integration**

```python
from src.pipeline.rag_system import RAGSystem
from langdetect import detect, detect_langs
import logging

class LanguageDetector:
    def __init__(self):
        self.supported_languages = ['hr', 'en']

    def detect_language(self, text: str) -> dict:
        try:
            # Primary detection
            detected = detect(text)
            confidence_scores = detect_langs(text)

            # Map to supported languages
            if detected in self.supported_languages:
                return {
                    "detected_language": detected,
                    "confidence": max([lang.prob for lang in confidence_scores if lang.lang == detected]),
                    "recommendation": detected
                }
            else:
                return {
                    "detected_language": "multilingual",
                    "confidence": 0.5,
                    "recommendation": "multilingual"
                }
        except:
            return {
                "detected_language": "multilingual",
                "confidence": 0.0,
                "recommendation": "multilingual"
            }
```

### **FastAPI Application Structure**

```python
from fastapi import FastAPI, HTTPException, Header, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import asyncio
from src.pipeline.rag_system import RAGSystem, RAGQuery

app = FastAPI(title="Multilingual RAG API", version="1.0.0")

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instances
rag_systems = {}

@app.on_event("startup")
async def startup_event():
    """Initialize RAG systems for all supported languages"""
    for lang in ['hr', 'en']:
        rag_systems[lang] = RAGSystem(language=lang)
        await rag_systems[lang].initialize()

    # Multilingual system
    rag_systems['multilingual'] = RAGSystem(language='multilingual')
    await rag_systems['multilingual'].initialize()

@app.post("/search")
async def search_documents(
    request: SearchRequest,
    input_lang: str = Query('multilingual', enum=['hr', 'en', 'multilingual']),
    ui_lang: str = Query('en', enum=['hr', 'en']),
    accept_language: Optional[str] = Header(None)
):
    """Search documents with language-aware processing"""

    # Language header override
    if accept_language:
        ui_lang = parse_accept_language(accept_language)

    # Get appropriate RAG system
    rag = rag_systems.get(input_lang, rag_systems['multilingual'])

    # Create language-aware query
    query = RAGQuery(text=request.query)

    # Execute search
    start_time = time.time()
    results = await rag.query(query)
    processing_time = int((time.time() - start_time) * 1000)

    # Format multilingual response
    response = format_search_response(
        results=results,
        input_lang=input_lang,
        ui_lang=ui_lang,
        processing_time=processing_time
    )

    return response
```

### **Error Handling with Localization**

```python
class MultilingualException(HTTPException):
    def __init__(
        self,
        status_code: int,
        message_en: str,
        message_hr: str,
        ui_lang: str = 'en'
    ):
        detail = message_hr if ui_lang == 'hr' else message_en
        super().__init__(status_code=status_code, detail=detail)

# Usage
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    ui_lang = request.query_params.get('ui_lang', 'en')
    raise MultilingualException(
        status_code=400,
        message_en=f"Invalid input: {str(exc)}",
        message_hr=f"Neispravan unos: {str(exc)}",
        ui_lang=ui_lang
    )
```

## 🚀 Implementation Priority

### **Phase 1: Core Endpoints (Week 1)**
1. ✅ Health check with multilingual support
2. ✅ Language detection endpoint
3. ✅ Basic search endpoint with language parameters
4. ✅ Error handling with localized messages

### **Phase 2: Document Management (Week 1.5)**
1. 🔲 Document upload with language specification
2. 🔲 Upload status tracking
3. 🔲 File validation and language detection
4. 🔲 Progress updates for document processing

### **Phase 3: Advanced Features (Week 2)**
1. 🔲 Query with generation endpoint
2. 🔲 Available languages metadata endpoint
3. 🔲 Advanced filtering and search options
4. 🔲 Batch operations and optimization

## ✅ Success Metrics

- [ ] All endpoints respond with appropriate `Content-Language` headers
- [ ] Language detection accuracy >90% for Croatian and English
- [ ] Error messages properly localized based on `ui_lang` parameter
- [ ] Search results correctly filtered by `input_lang` parameter
- [ ] Document upload supports language specification and auto-detection
- [ ] API response times <200ms for search, <2s for uploads
- [ ] CORS properly configured for React frontend integration
- [ ] Full OpenAPI documentation with multilingual examples
