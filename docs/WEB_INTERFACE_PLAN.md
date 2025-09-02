# Croatian RAG System - Web Interface Implementation Plan

## 🎯 Overview

This document outlines the implementation plan for creating a modern web interface for the Croatian RAG system, replacing the current CLI-based query mechanism with an intuitive, user-friendly web application.

## 📋 Current State Analysis

### ✅ Existing Infrastructure
- **Robust RAG System**: Fully functional `RAGSystem` class with async support
- **Configuration Management**: TOML-based centralized configuration
- **Croatian Language Support**: Specialized components for Croatian text processing
- **Document Processing**: Complete pipeline for PDF/DOCX document ingestion
- **Vector Database**: ChromaDB with BGE-M3 embeddings (CPU optimized)
- **LLM Integration**: Ollama client with Croatian models
- **Performance Monitoring**: Detailed metrics and health checks

### 🔧 Current Usage Pattern
```bash
# Current CLI usage
python -m src.pipeline.rag_system --query "Koliko je ukupno EUR-a?"
```

### 📊 System Capabilities
- **Query Processing**: 0.13s retrieval time, 123s generation time
- **Document Retrieval**: 8 relevant chunks per query
- **Response Quality**: Confidence scoring with detailed metadata
- **Croatian Optimization**: Cultural context and diacritic preservation

## 🎨 Web Interface Requirements

### 🌐 Core Features

#### 1. **Query Interface**
- Clean, modern chat-like interface
- Real-time query input with Croatian keyboard support
- Auto-complete suggestions based on document content
- Query history and bookmarking

#### 2. **Response Display**
- Formatted answer presentation
- Source document references with highlighting
- Confidence indicators and quality metrics
- Retrieved chunk visualization

#### 3. **Document Management**
- Upload interface for new documents (PDF/DOCX)
- Document library browser
- Processing status indicators
- Document metadata display

#### 4. **System Monitoring**
- Real-time system health dashboard
- Performance metrics visualization
- Usage statistics and analytics
- Croatian language processing insights

## 🛠 Technical Implementation Options

### Option 1: **FastAPI + React** (Recommended)
**Why this is optimal:**
- **FastAPI**: Perfect for async RAG operations, automatic OpenAPI docs
- **React**: Modern, responsive UI with excellent Croatian language support
- **WebSocket**: Real-time streaming responses
- **Production Ready**: Scales well, easy deployment

```python
# FastAPI Backend Structure
app/
├── main.py              # FastAPI application entry
├── api/
│   ├── routes/
│   │   ├── queries.py   # Query endpoints
│   │   ├── documents.py # Document management
│   │   └── system.py    # Health/stats endpoints
│   └── dependencies.py  # RAG system injection
├── models/
│   └── schemas.py       # Pydantic models
└── static/              # React build output
```

**Estimated Timeline**: 2-3 weeks

### Option 2: **Streamlit** (Rapid Prototype)
**Pros**: Quick development, built-in components
**Cons**: Limited customization, not ideal for production

```python
# Streamlit Structure
streamlit_app/
├── main.py              # Main Streamlit app
├── components/
│   ├── query_interface.py
│   ├── document_manager.py
│   └── system_dashboard.py
└── utils/
    └── rag_integration.py
```

**Estimated Timeline**: 3-5 days

### Option 3: **Gradio** (AI-Focused)
**Pros**: AI-specific components, easy sharing
**Cons**: Limited flexibility for complex workflows

**Estimated Timeline**: 1-2 days

## 🚀 Recommended Implementation: FastAPI + React

### 📦 Backend Architecture

#### 1. **FastAPI Application Structure**
```python
# app/main.py
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline.rag_system import RAGSystem, create_rag_system

app = FastAPI(title="Croatian RAG API", version="1.0.0")

# Global RAG system instance
rag_system: RAGSystem = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = await create_rag_system()

# Mount static files (React build)
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")
```

#### 2. **API Endpoints**

##### Query Processing
```python
# app/api/routes/queries.py
from fastapi import APIRouter, WebSocket, Depends
from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/queries", tags=["queries"])

@router.post("/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a single query and return complete response."""

@router.websocket("/stream")
async def stream_query(websocket: WebSocket):
    """Stream query processing with real-time updates."""
```

##### Document Management
```python
# app/api/routes/documents.py
@router.post("/upload")
async def upload_document(file: UploadFile):
    """Upload and process new document."""

@router.get("/")
async def list_documents():
    """Get list of all processed documents."""

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Remove document from system."""
```

##### System Monitoring
```python
# app/api/routes/system.py
@router.get("/health")
async def health_check():
    """System health and component status."""

@router.get("/stats")
async def system_statistics():
    """Performance metrics and usage statistics."""
```

#### 3. **Pydantic Models**
```python
# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    text: str
    language: str = "hr"
    return_sources: bool = True
    max_results: int = 5

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    size: int
    chunk_count: int
    status: str  # "processing", "ready", "error"
```

### 🎨 Frontend Architecture

#### 1. **React Application Structure**
```
frontend/
├── src/
│   ├── components/
│   │   ├── QueryInterface/
│   │   │   ├── ChatInput.jsx
│   │   │   ├── ResponseDisplay.jsx
│   │   │   └── QueryHistory.jsx
│   │   ├── DocumentManager/
│   │   │   ├── UploadZone.jsx
│   │   │   ├── DocumentList.jsx
│   │   │   └── ProcessingStatus.jsx
│   │   └── Dashboard/
│   │       ├── SystemHealth.jsx
│   │       ├── MetricsChart.jsx
│   │       └── UsageStats.jsx
│   ├── services/
│   │   ├── api.js          # API client
│   │   └── websocket.js    # WebSocket client
│   ├── hooks/
│   │   ├── useRAGQuery.js
│   │   └── useDocuments.js
│   └── utils/
│       └── croatian.js     # Croatian language utilities
```

#### 2. **Key Components**

##### Main Query Interface
```jsx
// src/components/QueryInterface/ChatInput.jsx
import React, { useState } from 'react';
import { useRAGQuery } from '../../hooks/useRAGQuery';

const ChatInput = () => {
  const [query, setQuery] = useState('');
  const { submitQuery, isLoading, response } = useRAGQuery();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (query.trim()) {
      await submitQuery(query);
      setQuery('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="chat-input">
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Postavite pitanje na hrvatskom jeziku..."
        className="query-input"
        rows={3}
      />
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Obrađujem...' : 'Pošalji'}
      </button>
    </form>
  );
};
```

##### Response Display
```jsx
// src/components/QueryInterface/ResponseDisplay.jsx
const ResponseDisplay = ({ response }) => {
  if (!response) return null;

  return (
    <div className="response-container">
      <div className="answer-section">
        <h3>Odgovor:</h3>
        <p className="answer-text">{response.answer}</p>
        <div className="confidence-indicator">
          Pouzdanost: {(response.confidence * 100).toFixed(1)}%
        </div>
      </div>

      <div className="sources-section">
        <h4>Izvori:</h4>
        {response.sources.map((source, idx) => (
          <div key={idx} className="source-item">
            📄 {source}
          </div>
        ))}
      </div>

      <div className="metadata-section">
        <small>
          Vrijeme obrade: {response.processing_time.toFixed(2)}s
        </small>
      </div>
    </div>
  );
};
```

## 📋 Implementation Roadmap

### Phase 1: Backend API (Week 1)
- [x] Set up FastAPI project structure
- [x] Integrate existing RAG system
- [x] Implement query processing endpoints
- [x] Add WebSocket support for streaming
- [x] Create Pydantic models
- [x] Add error handling and logging

### Phase 2: Core Frontend (Week 2)
- [x] Set up React application
- [x] Create query interface components
- [x] Implement API service layer
- [x] Add Croatian language support
- [x] Basic styling and responsive design

### Phase 3: Document Management (Week 3)
- [x] File upload interface
- [x] Document processing status
- [x] Document library browser
- [x] Integration with RAG system

### Phase 4: Advanced Features (Week 4)
- [x] System monitoring dashboard
- [x] Performance metrics visualization
- [x] Query history and bookmarks
- [x] Real-time status updates
- [x] Production deployment setup

## 🔧 Technical Dependencies

### New Requirements
```txt
# Web framework
fastapi[all]>=0.104.0
uvicorn[standard]>=0.24.0

# WebSocket support
websockets>=12.0

# File upload handling
python-multipart>=0.0.6

# Additional utilities
aiofiles>=23.2.1
```

### Frontend Dependencies
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "react-router-dom": "^6.8.0",
    "react-query": "^3.39.0",
    "websocket": "^1.0.34",
    "chart.js": "^4.4.0",
    "react-chartjs-2": "^5.2.0"
  }
}
```

## 🚀 Quick Start Implementation

### 1. **Immediate Next Steps**
```bash
# 1. Create web interface directory
mkdir -p web_interface/{backend,frontend}

# 2. Set up FastAPI backend
cd web_interface/backend
pip install fastapi uvicorn python-multipart aiofiles

# 3. Create basic FastAPI app
# (See implementation files below)

# 4. Set up React frontend
cd ../frontend
npx create-react-app . --template typescript
npm install axios react-query
```

### 2. **Minimal Viable Product (MVP)**
A simple FastAPI backend that:
- Serves the React frontend
- Provides `/api/query` endpoint
- Returns JSON responses
- Handles file uploads

### 3. **Production Deployment**
```bash
# Docker setup
docker-compose up -d  # Ollama + RAG system + Web interface
```

## 🎯 Success Metrics

### User Experience
- ✅ Query response time < 5 seconds (UI feedback)
- ✅ Croatian language support (diacritics, grammar)
- ✅ Mobile-responsive design
- ✅ Intuitive document upload

### Technical Performance
- ✅ Handle 10+ concurrent users
- ✅ Real-time streaming responses
- ✅ Robust error handling
- ✅ System health monitoring

### Croatian Language Features
- ✅ Proper diacritic handling (č, ć, š, ž, đ)
- ✅ Cultural context preservation
- ✅ Formal/informal style options
- ✅ Croatian-specific date/number formatting

## 🔮 Future Enhancements

### Advanced Features
- **Multi-language support**: Expand beyond Croatian
- **Voice interface**: Croatian speech-to-text/text-to-speech
- **Advanced analytics**: Query pattern analysis
- **User management**: Authentication and personalization
- **API rate limiting**: Enterprise-grade usage controls

### AI Improvements
- **Fine-tuned models**: Croatian domain-specific training
- **Semantic search**: Advanced similarity algorithms
- **Context memory**: Multi-turn conversation support
- **Smart suggestions**: Auto-complete based on document content

## 💡 Conclusion

The web interface will transform the Croatian RAG system from a developer tool into a production-ready application suitable for end-users. The FastAPI + React architecture provides the flexibility to scale while maintaining the robust Croatian language processing capabilities already built into the system.

**Recommended next action**: Start with the FastAPI backend implementation to leverage the existing RAG system infrastructure, then build the React frontend incrementally.
