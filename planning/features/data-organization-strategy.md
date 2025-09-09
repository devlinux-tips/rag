---
type: "feature_specification"
phase: "1"
milestone: "rag_system_evolution"
status: "planning"
priority: "critical"
last_updated: "2025-09-07"
dependencies: ["surrealdb_installation"]
---

# Data Organization Strategy for RAG System Evolution

## 🎯 Objective

Transform current file-based RAG system data organization into modern, scalable structure using SurrealDB for metadata/users and ChromaDB for vectors.

## 📊 Current State Analysis

### **Existing Data Structure**
```
services/rag-service/data/
├── raw/                    # Original documents
│   ├── hr/                # Croatian documents
│   ├── en/                # English documents
│   └── multilingual/      # Mixed language documents
├── vectordb/              # ChromaDB storage (persistent)
│   ├── croatian_documents.db
│   ├── english_documents.db
│   └── multilingual_documents.db
└── test/                  # Test documents and queries
```

### **Current ChromaDB Collections**
- `croatian_documents` - 275+ Croatian document chunks
- `english_documents` - English document chunks
- `multilingual_documents` - Mixed language content

## 🏗️ Proposed New Architecture

### **Hybrid Database Strategy**
```
Data Storage Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    RAG System Data                         │
├─────────────────────────┬───────────────────────────────────┤
│      SurrealDB          │           ChromaDB                │
│   (Metadata & Users)    │      (Vectors & Search)           │
├─────────────────────────┼───────────────────────────────────┤
│ • Users & Auth          │ • Document embeddings             │
│ • Document Metadata     │ • Semantic search                 │
│ • Prompt Templates      │ • Vector similarity               │
│ • Categories & Tags     │ • Language-specific collections   │
│ • Job Tracking          │ • Proven Croatian performance     │
│ • User Preferences      │                                   │
└─────────────────────────┴───────────────────────────────────┘
```

## 🗄️ SurrealDB Schema Design

### **Namespace Strategy**
```sql
-- Production namespace for main platform
USE NS rag_platform DB production;

-- Development namespace for testing
USE NS rag_platform DB development;

-- Future: Per-tenant namespaces for enterprise
USE NS tenant_acme DB rag_platform;
```

### **Core Tables Structure**

#### **1. Users Table**
```sql
-- User management and authentication
CREATE users SET {
    email: "user@example.com",
    name: "User Name",
    password_hash: "bcrypt_hash",
    language_preference: "hr", -- hr, en, multilingual
    created_at: time::now(),
    updated_at: time::now(),
    active: true,
    role: "user" -- user, admin, tenant_admin
};
```

#### **2. Documents Table**
```sql
-- Document metadata and organization
CREATE documents SET {
    filename: "croatian_tourism.pdf",
    original_path: "data/raw/hr/croatian_tourism.pdf",
    language: "hr", -- hr, en, multilingual, auto_detect
    category: "tourism", -- cultural, tourism, technical, legal, faq, general
    subcategory: "accommodation", -- more specific classification
    user_id: "users:user123", -- document owner
    tenant_id: "default", -- future multi-tenancy
    processing_status: "completed", -- pending, processing, completed, failed
    chunk_count: 25,
    file_size: 1024576,
    file_type: "application/pdf",
    uploaded_at: time::now(),
    processed_at: time::now(),
    tags: ["croatia", "hotels", "travel"],
    metadata: {
        author: "Croatian Tourism Board",
        title: "Accommodation Guide",
        summary: "Comprehensive guide to Croatian accommodations"
    }
};
```

#### **3. Document Chunks Table**
```sql
-- Individual text chunks with metadata
CREATE document_chunks SET {
    document_id: "documents:doc123",
    chunk_index: 0,
    content: "Zagreb is the capital and largest city of Croatia...",
    language: "hr",
    category: "tourism",
    chunk_type: "paragraph", -- paragraph, header, list_item, table
    word_count: 150,
    character_count: 850,
    chroma_collection: "croatian_documents",
    chroma_id: "chunk_abc123", -- Reference to ChromaDB
    created_at: time::now()
};
```

#### **4. Prompt Templates Table**
```sql
-- User-customizable prompt templates
CREATE prompt_templates SET {
    name: "cultural_context",
    category: "cultural",
    language: "hr",
    user_id: "users:user123", -- null for system templates
    tenant_id: "default",
    system_prompt: "Ti si stručnjak za hrvatsku kulturu...",
    user_template: "Pitanje: {query}\\n\\nOdgovor:",
    context_template: "Kontekst:\\n{context}\\n\\n",
    is_system_default: false,
    is_active: true,
    usage_count: 150,
    created_at: time::now(),
    updated_at: time::now(),
    parent_template: "prompt_templates:system_cultural" -- inheritance
};
```

#### **5. User Queries Table**
```sql
-- Query history and analytics
CREATE user_queries SET {
    user_id: "users:user123",
    query_text: "Koje su najljepše destinacije u Hrvatskoj?",
    language: "hr",
    detected_category: "tourism",
    selected_category: "tourism", -- user override
    prompt_template: "prompt_templates:cultural_tourism",
    response_time_ms: 850,
    satisfaction_rating: 5, -- 1-5 scale
    documents_retrieved: 5,
    created_at: time::now()
};
```

#### **6. User Sessions Table**
```sql
-- User session management
CREATE user_sessions SET {
    user_id: "users:user123",
    session_token: "jwt_token_hash",
    ip_address: "192.168.1.100",
    user_agent: "Mozilla/5.0...",
    created_at: time::now(),
    expires_at: (time::now() + 7d),
    active: true
};
```

## 🔄 Data Migration Strategy

### **Phase 1: Current System + SurrealDB**
1. **Keep ChromaDB unchanged** - preserve existing vector data
2. **Add SurrealDB metadata** - create tables for documents, users, templates
3. **Build mapping layer** - connect SurrealDB metadata to ChromaDB vectors
4. **Gradual enhancement** - add features without breaking current system

### **Document Processing Pipeline**
```python
async def process_document(file_path: str, user_id: str, category: str = None):
    # 1. Create document record in SurrealDB
    doc_metadata = {
        "filename": Path(file_path).name,
        "original_path": file_path,
        "user_id": user_id,
        "category": category or "auto_detect",
        "processing_status": "processing"
    }
    doc_id = await surrealdb.create("documents", doc_metadata)

    # 2. Extract and process chunks (existing logic)
    chunks = await extract_and_process_document(file_path)

    # 3. Store chunks in ChromaDB (existing logic)
    chroma_ids = await store_chunks_in_chroma(chunks, language)

    # 4. Store chunk metadata in SurrealDB
    for i, (chunk, chroma_id) in enumerate(zip(chunks, chroma_ids)):
        await surrealdb.create("document_chunks", {
            "document_id": doc_id,
            "chunk_index": i,
            "content": chunk.content,
            "chroma_collection": f"{language}_documents",
            "chroma_id": chroma_id
        })

    # 5. Update document status
    await surrealdb.update(doc_id, {"processing_status": "completed"})
```

## 🔍 Query Enhancement Strategy

### **Hierarchical Query Routing**
```python
class EnhancedQueryProcessor:
    async def process_query(self, query: str, user_id: str):
        # 1. Get user preferences
        user = await surrealdb.select(f"users:{user_id}")

        # 2. Classify query category
        category = await classify_query_category(query, user.language_preference)

        # 3. Get appropriate prompt template
        template = await get_user_prompt_template(user_id, category)

        # 4. Enhanced retrieval with metadata filtering
        results = await retrieve_with_metadata_filter(
            query=query,
            language=user.language_preference,
            category=category,
            user_id=user_id
        )

        # 5. Generate response with custom template
        response = await generate_with_template(query, results, template)

        # 6. Log query for analytics
        await surrealdb.create("user_queries", {
            "user_id": user_id,
            "query_text": query,
            "detected_category": category,
            "response_time_ms": response.time_ms
        })

        return response
```

### **Multi-Language Query Support**
```python
async def multilingual_search(query: str, languages: list = ["hr", "en"]):
    """Search across multiple language collections simultaneously"""
    results = []

    for lang in languages:
        # Get language-specific results
        lang_results = await search_chroma_collection(
            query=query,
            collection=f"{lang}_documents",
            language=lang
        )

        # Enhance with SurrealDB metadata
        enhanced_results = await enhance_with_metadata(lang_results, lang)
        results.extend(enhanced_results)

    # Merge and rank results across languages
    return await merge_multilingual_results(results)
```

## 📈 Scalability Considerations

### **Performance Optimization**
- **SurrealDB Indexing**: Create indexes on frequently queried fields
- **ChromaDB Collections**: Keep language-specific collections for optimal search
- **Caching Layer**: Redis/memory cache for frequent user queries
- **Batch Operations**: Process multiple documents simultaneously

### **Multi-tenancy Preparation**
```sql
-- Future tenant isolation with namespaces
USE NS tenant_acme DB rag_platform;

-- Tenant-specific data with same schema
CREATE documents SET {
    filename: "acme_internal_doc.pdf",
    tenant_id: "acme",
    category: "internal"
    -- ... rest same as above
};
```

## ✅ Implementation Checkpoints

### **Week 1: SurrealDB Integration**
- [ ] Create SurrealDB schema and tables
- [ ] Implement document metadata storage
- [ ] Build SurrealDB ↔ ChromaDB mapping layer
- [ ] Test basic CRUD operations

### **Week 2: Enhanced Query Processing**
- [ ] Implement query classification system
- [ ] Add prompt template management
- [ ] Build hierarchical query routing
- [ ] Test multilingual query processing

### **Week 3: User Management**
- [ ] Implement user authentication
- [ ] Add user preferences and customization
- [ ] Build user session management
- [ ] Test user-specific document access

## 🎯 Success Criteria

### **Data Organization**
- [ ] All document metadata stored in SurrealDB with proper relationships
- [ ] ChromaDB collections maintained with existing performance
- [ ] User documents properly isolated and organized by category
- [ ] Prompt templates customizable per user with inheritance

### **Query Enhancement**
- [ ] Smart query routing based on content category
- [ ] Multilingual search working equally well for Croatian and English
- [ ] User preferences affect query processing and results
- [ ] Response quality maintained or improved from current system

### **Technical Foundation**
- [ ] Clean separation between metadata (SurrealDB) and vectors (ChromaDB)
- [ ] Scalable schema ready for multi-tenancy
- [ ] Performance acceptable for single-user and small team usage
- [ ] Migration path clear for moving to full multi-tenant system

This data organization strategy provides a solid foundation for Phase 1 while preparing for advanced features in future phases.
