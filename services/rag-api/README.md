# RAG API Service

FastAPI bridge between TypeScript Web API and Python RAG system.

## Purpose

This service provides a minimal HTTP API layer over the Python RAG system, allowing the TypeScript Web API to execute RAG queries without directly invoking Python scripts.

## Endpoints

### `POST /api/v1/query`

Execute a RAG query with scope-aware collection routing.

**Request:**
```json
{
  "query": "What is RAG?",
  "tenant": "development",
  "user": "dev_user",
  "language": "hr",
  "feature": "user",  // Scope: "user", "tenant", or feature name
  "max_documents": 5,
  "min_confidence": 0.7,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "RAG (Retrieval-Augmented Generation) is...",
  "sources": [...],
  "documentsRetrieved": 5,
  "documentsUsed": 3,
  "confidence": 0.89,
  "searchTimeMs": 234,
  "responseTimeMs": 1000,
  "model": "qwen2.5:7b-instruct",
  "tokensUsed": {
    "input": 150,
    "output": 300,
    "total": 450
  }
}
```

## Collection Routing

The service automatically routes to the correct collection based on the feature/scope:

- **User scope** (default): `{tenant}_{user}_{language}_documents`
- **Tenant scope**: `{tenant}_shared_{language}_documents`
- **Feature scope**: `Features_{feature_name}_{language}` (global datasets)

Global features include:
- narodne-novine
- financial-reports
- legal-docs
- medical-records

## Running

```bash
# From services/rag-api directory
pip install -r requirements.txt
python main.py
```

The server will start on http://localhost:8080