# RAG Architecture Research: Multi-Tenant Search & Data Organization

**Research Date**: September 6, 2025
**Research Scope**: Multi-dimensional analysis of current RAG system architecture focusing on data organization, search patterns, and scalability for multi-tenant scenarios.

## Executive Summary

This document presents comprehensive research findings on evolving the current multilingual RAG system from language-based organization to support user/tenant-specific data organization while maintaining search categorization capabilities. The research recommends implementing a **Hierarchical Router Pattern with Progressive Enhancement** as the optimal architectural approach.

## Current System Analysis

### ✅ Existing Architecture Strengths

**Hybrid Search Implementation:**
- Dense embeddings (BGE-M3) with 0.7 weight + Sparse (BM25) with 0.3 weight
- `HybridRetriever` class with multilingual BM25 supporting Croatian diacritics
- Language-specific stop words and morphological analysis
- RetrievalStrategy enum: SIMPLE, ADAPTIVE, MULTI_PASS, HYBRID

**Data Organization:**
```
Current Structure:
/data/raw/
├── hr/           → croatian_documents collection (ChromaDB)
├── en/           → english_documents collection
└── multilingual/ → multilingual_documents collection

RAGSystem(language="hr")  # Language-specific initialization required
```

**Production-Ready Architecture:**
- Complete pipeline: preprocessing → embeddings → storage → retrieval → generation
- Language-agnostic components configured via TOML
- Multi-device support (CUDA/MPS/CPU auto-detection)
- Async/await patterns with proper error handling
- DRY principles implementation

### 📊 System Performance Metrics

**Current Performance:**
- Generation Time: 83.5s (CPU-based qwen2.5:7b-instruct)
- Retrieval Time: 0.12s (excellent performance)
- Croatian Quality: ✅ Excellent with cultural context
- Memory Usage: ~3-4GB (13GB GPU available for optimization)
- Data Storage: 275+ document chunks stored persistently

## Multi-Tenant Data Organization Challenges

### Current vs Future Requirements

**Current:** Language-only organization
- `hr/` → `croatian_documents`
- `en/` → `english_documents`
- `multilingual/` → `multilingual_documents`

**Future Requirements:**
- User-specific data within languages: `user1/hr/`, `user1/en/`, `user2/hr/`
- Tenant-specific data: `tenant1/user1/hr/`, `tenant1/user2/hr/`
- Cross-language search within user/tenant boundaries
- Different search behaviors per user/tenant

### Architectural Decision Matrix

| Pattern | Pros | Cons | Complexity |
|---------|------|------|------------|
| **Extended Current** | Minimal disruption, fast implementation | Collection explosion, limited categorization | Low |
| **Hierarchical Router** ⭐ | Flexible, scalable, natural multi-tenancy | More complex, requires query classification | Medium |
| **Metadata-Heavy Single** | Simple collection management | Security concerns, performance at scale | Low |
| **Federated Search** | Complete isolation, different models per system | Resource intensive, complex orchestration | High |

## Recommended Architecture: Hierarchical Router Pattern

### 🏗️ Architectural Overview

```
QueryRouter (Level 1)
├── Query Classification & Intent Analysis
├── User Preference Detection
└── Route to Specialized Retrievers

Specialized Retrievers (Level 2)
├── LanguageRouter (hr/en/multilingual)
├── CategoryRouter (technical/legal/faq/general)
├── ScopeRouter (personal/team/organization)
└── TenantRouter (user_id/tenant_id filtering)

Search Execution (Level 3)
├── Collections: {tenant}_{category}_{language}
├── Example: "acme_technical_hr", "personal_faq_en"
└── Hybrid search within each specialized collection
```

### 🎯 Core Design Principles

**Progressive Enhancement** - evolve the current system through three phases while maintaining backward compatibility:

1. **Phase 1: Metadata Extension** - Multi-tenant support with minimal disruption
2. **Phase 2: Smart Query Routing** - Intelligent query classification and routing
3. **Phase 3: Full Hierarchical System** - Complete specialized retriever architecture

### Collection Naming Strategy

**Hierarchical Convention:** `{tenant}_{category}_{language}`

**Examples:**
- `acme_technical_hr` (ACME Corp Croatian technical docs)
- `acme_general_en` (ACME Corp English general docs)
- `personal_faq_hr` (Personal Croatian FAQ)
- `shared_legal_multilingual` (Shared legal documents)

## Alternative Search Patterns Research

### Hierarchical Retrieval (Router Pattern)

**Architecture Benefits:**
- **Level 1 Router/Orchestrator**: Query classification, language detection, domain identification
- **Level 2 Specialized Retrievers**: Language-specific, domain-specific, user/tenant-specific, content-type-specific
- **Level 3 Search Execution**: Each specialized retriever has optimized collections, embeddings, and search parameters

**Key Advantages:**
- Better relevance through specialization
- Easier to optimize each retriever independently
- Natural isolation for multi-tenancy
- Different search policies per category

### Search Categorization Strategies

**Generic Search (Current System):**
- Search across all available documents
- Language-aware but content-agnostic
- Good for broad exploratory queries

**Specific/Categorized Search Options:**
1. **Content Category**: Technical, Legal, FAQ, Policy
2. **Source-Type**: Internal docs, external knowledge, user uploads
3. **Temporal**: Recent docs, historical docs, specific time ranges
4. **Authority**: Official docs, draft docs, community contributions
5. **Scope**: Personal, team, organization-wide

**User Choice Mechanisms:**
- **UI Filters**: `[ ] Technical Docs [ ] FAQ [ ] Policies`
- **Query Prefixes**: `"tech:", "faq:", "legal:"`
- **Advanced Search**: Dropdowns and faceted search
- **Smart Routing**: Based on query analysis
- **Personalized Defaults**: Based on user behavior

## Implementation Roadmap

### Phase 1: Metadata Extension (2-3 weeks)
**Goal**: Multi-tenant support with minimal disruption

**Changes Required:**
- Extend metadata schema: `user_id`, `tenant_id`, `category`, `scope`
- Add filtering to existing `IntelligentRetriever`
- Collection naming: `{language}_{tenant}_documents`

**Code Example:**
```python
# Extend existing RAGSystem initialization
rag = RAGSystem(language="hr", tenant_id="acme", user_id="user123")

# Add metadata filtering to queries
query = RAGQuery(
    text="Kako instalirati sustav?",
    context_filters={"tenant_id": "acme", "category": "technical"}
)
```

### Phase 2: Smart Query Routing (4-5 weeks)
**Goal**: Intelligent query classification and routing

**New Components:**
- `QueryClassifier`: Determines query intent, category, scope
- `QueryRouter`: Routes queries to appropriate retrievers
- Category-specific prompt templates

**Architecture:**
```python
class QueryRouter:
    def route_query(self, query: str, user_prefs: dict) -> RetrievalStrategy:
        # Classify query type (technical/faq/legal/general)
        # Determine optimal search scope (personal/team/organization)
        # Route to specialized retriever or fallback to generic
```

### Phase 3: Full Hierarchical System (6-8 weeks)
**Goal**: Complete specialized retriever architecture

**New Features:**
- Separate collections per category: `acme_technical_hr`, `acme_faq_en`
- Category-specific embedding optimizations
- Federated search across multiple collections
- Advanced result fusion and ranking

## Performance & Scalability Considerations

### Optimization Opportunities

**Response Caching System:**
- **Expected Benefit**: 95%+ speedup for repeated queries
- **Implementation**: Redis/Memory-based caching with TTL
- **Use Cases**: FAQ queries, document summaries, common Croatian phrases

**Parallel Processing:**
- **Expected Benefit**: 2-3x throughput for multiple queries
- **Implementation**: Async batch processing with concurrent limits
- **Use Cases**: Batch document analysis, multi-user scenarios

**GPU Acceleration:**
- **Expected Benefit**: 5-10x generation speedup (83s → 8-15s)
- **Current Status**: 13GB GPU available, currently using 3-4GB
- **Strategy**: Investigate Ollama GPU utilization optimization

### Scalability Metrics

**Target Performance Goals:**
- **Collections**: Support 100+ tenant/category combinations
- **Query Latency**: <2s for multi-collection searches
- **Concurrent Users**: 50+ simultaneous queries
- **Caching**: <1s for repeated queries (95% of FAQ use cases)
- **Storage**: Efficient sharding across collections

## Migration Strategy

### Backward Compatibility

**Preserve Existing Functionality:**
- Current `RAGSystem(language="hr")` continues to work unchanged
- Gradual migration of documents to new collection structure
- Fallback to current system if new features fail
- No disruption to existing production deployments

**Data Migration Process:**
```python
# Phase 1: Add metadata to existing documents
for document in existing_documents:
    document.metadata.update({
        "tenant_id": "default",
        "category": "general",
        "scope": "organization"
    })

# Phase 2: Create category-specific collections
# Phase 3: Migrate documents to specialized collections
```

## Risk Mitigation

### Critical Success Factors

**Must-Have Features:**
1. **Fallback Mechanism**: Always maintain "search everything" option
2. **Query Classification Accuracy**: >85% correct routing
3. **Performance**: No degradation from current 0.12s retrieval time
4. **Croatian Language**: Preserve existing Croatian optimization

**Risk Management:**
- **Gradual Rollout**: Phase-by-phase implementation with validation gates
- **A/B Testing**: Compare new vs current system performance
- **Error Handling**: Graceful degradation to simple search
- **Monitoring**: Track query routing accuracy and user satisfaction
- **Rollback Capability**: Maintain ability to revert changes

## Technical Implementation Details

### Enhanced Retrieval Architecture

**Modified IntelligentRetriever:**
```python
class EnhancedIntelligentRetriever(IntelligentRetriever):
    def __init__(self, tenant_id: str = None, user_id: str = None):
        self.tenant_id = tenant_id
        self.user_id = user_id
        super().__init__(...)

    def retrieve(self, query: str, category_hint: str = None) -> RetrievalResult:
        # Priority routing: category_hint → keyword_detection → default
        # Apply tenant/user filtering automatically
```

**Collection Management:**
```python
class MultiTenantCollectionManager:
    def get_collection_name(self, tenant_id: str, category: str, language: str) -> str:
        return f"{tenant_id}_{category}_{language}"

    def create_tenant_collections(self, tenant_id: str, categories: List[str]):
        # Automatically create all necessary collections for new tenant
```

### Database Schema Extensions

**Metadata Enhancement:**
```python
@dataclass
class EnhancedChunkMetadata:
    # Existing fields
    source: str
    chunk_index: int
    language: str

    # New multi-tenant fields
    tenant_id: str
    user_id: Optional[str]
    category: str  # technical, legal, hr, faq, general
    scope: str     # personal, team, organization, public
    access_level: int  # 0=public, 1=organization, 2=team, 3=personal
    tags: List[str]
    created_by: str
    updated_at: datetime
```

## Future Research Areas

### Advanced Features for Consideration

**Federated Learning:**
- Cross-tenant knowledge sharing (with privacy preservation)
- Collaborative filtering for improved search relevance
- Tenant-specific model fine-tuning

**Advanced Analytics:**
- Search pattern analysis per tenant/user
- Query success rate optimization
- Automated category detection improvement

**Integration Opportunities:**
- External knowledge bases (Wikipedia, industry databases)
- Real-time data sources (news feeds, market data)
- Collaborative editing and knowledge management

## Conclusion

The **Hierarchical Router Pattern with Progressive Enhancement** provides the optimal evolutionary path for the multilingual RAG system. This approach:

- **Preserves Investment**: Leverages existing hybrid search and multilingual capabilities
- **Enables Growth**: Supports unlimited tenants, users, and categories
- **Maintains Performance**: No degradation of current excellent retrieval performance
- **Provides Flexibility**: Multiple search modes for different user types
- **Ensures Scalability**: Architecture designed for enterprise-scale deployment

The phased implementation approach minimizes risk while delivering immediate value in Phase 1, with sophisticated capabilities emerging in Phases 2 and 3. This strategy positions the system for long-term success in multi-tenant, categorized search scenarios while maintaining the excellent Croatian language support that distinguishes the current implementation.
