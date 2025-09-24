# Multi-Source Query Routing Enhancement for RAG System

**Date**: September 23, 2025
**Scope**: Enhanced query routing for tenant-wide, user-specific, and feature-based data sources
**Status**: Architecture Design & Implementation Plan

## Executive Summary

This document outlines a comprehensive enhancement to the current RAG system's query architecture to support robust multi-source data querying. The enhancement introduces intelligent query routing that can handle tenant-wide queries, feature-specific data sources (Croatian Narodne Novine), and adaptive data source selection based on query intent.

## Current System Analysis

### Architecture Overview
The current RAG system operates with a single-scope query pattern:

```python
# Current Collection Naming Pattern
collection_name = f"{tenant}_{user}_{language}_documents"

# Current Query Flow
CLI → MultiTenantRAGCLI.execute_query_command() → RAGSystem.query() → HierarchicalRetriever.retrieve() → single collection
```

### Current Limitations
1. **Single Collection Scope**: Each query targets exactly one collection
2. **No Tenant-Wide Queries**: Users cannot query across tenant data
3. **No Feature Sources**: No support for special data sources like Narodne Novine
4. **Static Data Scoping**: Collection selection is deterministic based on context only

### Current Strengths
1. **Multi-language Support**: Well-designed language isolation
2. **Hierarchical Retrieval**: Intelligent categorization and strategy selection
3. **Multi-tenant Infrastructure**: Solid tenant/user separation
4. **Dependency Injection**: Clean, testable architecture

## Research Findings: Modern RAG Routing Strategies

### 1. Dynamic Query Routing (2024-2025 Standards)
- **LLM-Based Classification**: Use language models to analyze query intent
- **Multi-Agent Architecture**: Specialized agents for different data sources
- **Adaptive Retrieval**: Adjust strategy based on query complexity
- **Self-Querying**: Break complex questions into focused searches

### 2. Multi-Tenant Best Practices
- **Namespace-Based Isolation**: Per-tenant collections/namespaces
- **Pool vs Silo Patterns**: Shared vs separate data sources
- **Topic-Based Routing**: Route based on content domains
- **Performance Optimization**: Query only relevant collections

### 3. Production Patterns
- **Hybrid Search**: Combine dense embeddings with sparse retrieval
- **Result Aggregation**: Merge and rank multi-source results
- **Security**: Fine-grained access control with JWT
- **Caching**: Query result caching for performance

## Proposed Enhanced Architecture

### 1. Query Intent Classification

```python
from typing import Literal, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class QueryScope(Enum):
    """Available query scopes for data source selection."""
    USER_ONLY = "user_only"           # Current behavior: {tenant}_{user}_{lang}_documents
    TENANT_WIDE = "tenant_wide"       # Query across all tenant users
    FEATURE_SPECIFIC = "feature"      # Feature data sources (Narodne Novine)
    HYBRID = "hybrid"                 # Multiple sources combined

class QueryIntent(Enum):
    """Query intent categories for routing decisions."""
    PERSONAL_DOCUMENTS = "personal"    # User's own documents
    COMPANY_KNOWLEDGE = "company"      # Tenant-wide company information
    LEGAL_OFFICIAL = "legal"          # Narodne Novine, legal documents
    TECHNICAL_REFERENCE = "technical" # Technical documentation
    GENERAL_INQUIRY = "general"       # General questions, multiple sources

@dataclass
class QueryClassification:
    """Result of query intent analysis."""
    primary_intent: QueryIntent
    suggested_scope: QueryScope
    confidence: float
    reasoning: str
    data_sources: List[str]
    search_strategy: str
```

### 2. Intelligent Query Router

```python
class IntelligentQueryRouter:
    """Routes queries to appropriate data sources based on intent analysis."""

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        collection_manager: CollectionManager,
        feature_registry: FeatureRegistry,
        llm_client: GenerationClientProtocol,
    ):
        self.intent_classifier = intent_classifier
        self.collection_manager = collection_manager
        self.feature_registry = feature_registry
        self.llm_client = llm_client

    async def classify_and_route(
        self,
        query: str,
        context: TenantContext,
        language: str,
        user_preferences: Dict[str, Any] = None
    ) -> QueryRoutingPlan:
        """Analyze query and create routing plan."""

        # Step 1: Classify query intent using LLM
        classification = await self.intent_classifier.classify_query(
            query=query,
            language=language,
            context=context
        )

        # Step 2: Determine available data sources
        available_sources = await self.collection_manager.get_available_collections(
            tenant=context.tenant_slug,
            user=context.user_username,
            language=language
        )

        # Step 3: Check feature availability (e.g., Narodne Novine for Croatian)
        feature_sources = await self.feature_registry.get_available_features(
            language=language,
            tenant=context.tenant_slug,
            user_preferences=user_preferences
        )

        # Step 4: Create routing plan
        routing_plan = self._create_routing_plan(
            classification=classification,
            available_sources=available_sources,
            feature_sources=feature_sources,
            context=context,
            language=language
        )

        return routing_plan

    def _create_routing_plan(
        self,
        classification: QueryClassification,
        available_sources: List[CollectionInfo],
        feature_sources: List[FeatureSource],
        context: TenantContext,
        language: str
    ) -> QueryRoutingPlan:
        """Create execution plan for multi-source query."""

        collections_to_query = []
        search_strategies = {}

        # Route based on intent
        if classification.primary_intent == QueryIntent.PERSONAL_DOCUMENTS:
            # Current behavior: user-specific collection only
            user_collection = f"{context.tenant_slug}_{context.user_username}_{language}_documents"
            collections_to_query.append(user_collection)
            search_strategies[user_collection] = "semantic_focused"

        elif classification.primary_intent == QueryIntent.COMPANY_KNOWLEDGE:
            # Tenant-wide: query all users in tenant
            tenant_collections = [
                col.name for col in available_sources
                if col.name.startswith(f"{context.tenant_slug}_") and col.name.endswith(f"_{language}_documents")
            ]
            collections_to_query.extend(tenant_collections)
            for col in tenant_collections:
                search_strategies[col] = "hybrid"

        elif classification.primary_intent == QueryIntent.LEGAL_OFFICIAL:
            # Feature-specific: Narodne Novine for Croatian
            if language == "hr":
                narodne_novine = f"feature_narodne_novine_{language}_documents"
                collections_to_query.append(narodne_novine)
                search_strategies[narodne_novine] = "technical_precise"

            # Also include user documents for context
            user_collection = f"{context.tenant_slug}_{context.user_username}_{language}_documents"
            collections_to_query.append(user_collection)
            search_strategies[user_collection] = "semantic_focused"

        elif classification.primary_intent == QueryIntent.GENERAL_INQUIRY:
            # Hybrid: multiple sources with different weights
            if classification.confidence < 0.7:
                # Low confidence: query multiple sources
                user_collection = f"{context.tenant_slug}_{context.user_username}_{language}_documents"
                tenant_shared = f"{context.tenant_slug}_shared_{language}_documents"

                collections_to_query.extend([user_collection, tenant_shared])
                search_strategies[user_collection] = "semantic_focused"
                search_strategies[tenant_shared] = "hybrid"

                # Include features if enabled
                for feature in feature_sources:
                    if feature.enabled:
                        collections_to_query.append(feature.collection_name)
                        search_strategies[feature.collection_name] = feature.default_strategy

        return QueryRoutingPlan(
            collections=collections_to_query,
            strategies=search_strategies,
            aggregation_method="weighted_fusion",
            classification=classification
        )
```

### 3. Multi-Source Retrieval Engine

```python
class MultiSourceRetriever:
    """Retrieves and aggregates results from multiple data sources."""

    def __init__(
        self,
        hierarchical_retriever: HierarchicalRetriever,
        result_aggregator: ResultAggregator,
        performance_tracker: PerformanceTracker,
    ):
        self.hierarchical_retriever = hierarchical_retriever
        self.result_aggregator = result_aggregator
        self.performance_tracker = performance_tracker

    async def retrieve_multi_source(
        self,
        query: str,
        routing_plan: QueryRoutingPlan,
        max_results_per_source: int = 10,
        final_result_limit: int = 5
    ) -> MultiSourceRetrievalResult:
        """Execute retrieval across multiple sources and aggregate results."""

        source_results = {}
        retrieval_metrics = {}

        # Execute retrieval for each collection in parallel
        retrieval_tasks = []

        for collection_name in routing_plan.collections:
            strategy = routing_plan.strategies.get(collection_name, "default")

            task = self._retrieve_from_collection(
                query=query,
                collection_name=collection_name,
                strategy=strategy,
                max_results=max_results_per_source
            )
            retrieval_tasks.append((collection_name, task))

        # Execute all retrievals in parallel
        for collection_name, task in retrieval_tasks:
            try:
                start_time = time.time()
                result = await task
                retrieval_time = time.time() - start_time

                source_results[collection_name] = result
                retrieval_metrics[collection_name] = {
                    "retrieval_time": retrieval_time,
                    "results_count": len(result.documents),
                    "strategy_used": result.strategy_used
                }

            except Exception as e:
                # Log error but continue with other sources
                logger.error(f"Retrieval failed for {collection_name}: {e}")
                source_results[collection_name] = None

        # Aggregate results from all sources
        aggregated_result = await self.result_aggregator.aggregate_results(
            source_results=source_results,
            query=query,
            routing_plan=routing_plan,
            final_limit=final_result_limit
        )

        return MultiSourceRetrievalResult(
            aggregated_result=aggregated_result,
            source_results=source_results,
            routing_plan=routing_plan,
            metrics=retrieval_metrics
        )

    async def _retrieve_from_collection(
        self,
        query: str,
        collection_name: str,
        strategy: str,
        max_results: int
    ) -> HierarchicalRetrievalResult:
        """Retrieve from a single collection with specified strategy."""

        # Switch to specific collection
        original_collection = self.hierarchical_retriever._search_engine.current_collection
        await self.hierarchical_retriever._search_engine.switch_collection(collection_name)

        try:
            # Execute retrieval with strategy override
            context = {"strategy_override": strategy}
            result = await self.hierarchical_retriever.retrieve(
                query=query,
                max_results=max_results,
                context=context
            )
            return result

        finally:
            # Restore original collection
            await self.hierarchical_retriever._search_engine.switch_collection(original_collection)
```

### 4. Result Aggregation and Ranking

```python
class ResultAggregator:
    """Aggregates and ranks results from multiple data sources."""

    def __init__(self, reranker: RankerProtocol = None):
        self.reranker = reranker

    async def aggregate_results(
        self,
        source_results: Dict[str, HierarchicalRetrievalResult],
        query: str,
        routing_plan: QueryRoutingPlan,
        final_limit: int = 5
    ) -> AggregatedRetrievalResult:
        """Aggregate and rank results from multiple sources."""

        all_results = []
        source_weights = self._calculate_source_weights(routing_plan)

        # Collect all results with source information
        for collection_name, result in source_results.items():
            if result is None:
                continue

            source_weight = source_weights.get(collection_name, 1.0)
            source_type = self._determine_source_type(collection_name)

            for doc in result.documents:
                enhanced_doc = {
                    **doc,
                    "source_collection": collection_name,
                    "source_type": source_type,
                    "source_weight": source_weight,
                    "original_score": doc.get("final_score", doc.get("similarity_score", 0.0)),
                    "weighted_score": doc.get("final_score", 0.0) * source_weight
                }
                all_results.append(enhanced_doc)

        # Sort by weighted score
        all_results.sort(key=lambda x: x["weighted_score"], reverse=True)

        # Apply cross-source reranking if available
        if self.reranker and len(all_results) > final_limit:
            all_results = await self.reranker.rerank(
                query=query,
                documents=all_results[:final_limit * 2],  # Rerank top candidates
                category="multi_source"
            )

        # Limit final results
        final_results = all_results[:final_limit]

        # Calculate diversity metrics
        diversity_score = self._calculate_diversity_score(final_results)

        return AggregatedRetrievalResult(
            documents=final_results,
            total_sources_queried=len([r for r in source_results.values() if r is not None]),
            source_distribution=self._calculate_source_distribution(final_results),
            diversity_score=diversity_score,
            aggregation_metadata={
                "source_weights": source_weights,
                "routing_plan": routing_plan,
                "reranked": self.reranker is not None
            }
        )

    def _calculate_source_weights(self, routing_plan: QueryRoutingPlan) -> Dict[str, float]:
        """Calculate relevance weights for different data sources."""
        weights = {}
        intent = routing_plan.classification.primary_intent

        for collection in routing_plan.collections:
            if "feature_narodne_novine" in collection:
                # Legal/official sources get high weight for legal queries
                weights[collection] = 1.2 if intent == QueryIntent.LEGAL_OFFICIAL else 0.8
            elif "_shared_" in collection:
                # Tenant shared documents
                weights[collection] = 1.1 if intent == QueryIntent.COMPANY_KNOWLEDGE else 0.9
            else:
                # User-specific documents (baseline)
                weights[collection] = 1.0

        return weights

    def _determine_source_type(self, collection_name: str) -> str:
        """Determine the type of data source."""
        if "feature_narodne_novine" in collection_name:
            return "legal_official"
        elif "_shared_" in collection_name:
            return "tenant_shared"
        else:
            return "user_personal"

    def _calculate_diversity_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate diversity of result sources."""
        if not results:
            return 0.0

        source_types = set(doc["source_type"] for doc in results)
        return len(source_types) / 3.0  # Normalize by max possible types
```

## Narodne Novine Integration Strategy

### 1. Feature-Specific Collection Structure

```python
class NarodneNovineFeature:
    """Croatian Official Gazette (Narodne Novine) integration."""

    COLLECTION_NAME = "feature_narodne_novine_hr_documents"
    FEATURE_ID = "narodne_novine"

    def __init__(
        self,
        vector_storage: VectorStorageProtocol,
        document_processor: DocumentProcessorProtocol,
        config: NarodneNovineConfig
    ):
        self.vector_storage = vector_storage
        self.document_processor = document_processor
        self.config = config

    async def initialize_collection(self) -> None:
        """Initialize Narodne Novine collection with special metadata."""
        await self.vector_storage.initialize(
            collection_name=self.COLLECTION_NAME,
            reset_if_exists=False
        )

    async def ingest_official_documents(self, documents: List[NarodneNovineDocument]) -> None:
        """Ingest official documents with enhanced metadata."""

        processed_documents = []

        for doc in documents:
            # Enhanced metadata for legal documents
            enhanced_metadata = {
                "document_type": "official_gazette",
                "publication_number": doc.broj_nn,
                "publication_year": doc.godina,
                "publication_date": doc.datum_objave.isoformat(),
                "legal_category": doc.kategorija,
                "ministry": doc.ministarstvo,
                "legal_status": doc.status,  # aktivan, povučen, izmijenjen
                "authority_level": doc.razina_vlasti,  # državna, regionalna, lokalna
                "language": "hr",
                "content_type": "legal_text",
                "source": "narodne_novine_official",
                "searchable_keywords": doc.kljucne_rijeci,
                "legal_references": doc.pravni_odnosi,
                "effective_date": doc.datum_stupanja_na_snagu.isoformat() if doc.datum_stupanja_na_snagu else None
            }

            # Process document with legal-specific chunking
            chunks = await self.document_processor.process_legal_document(
                content=doc.sadrzaj,
                metadata=enhanced_metadata,
                chunking_strategy="legal_paragraph_aware"
            )

            processed_documents.extend(chunks)

        # Store with special embedding strategy for legal content
        await self.vector_storage.store_documents(
            documents=[chunk.content for chunk in processed_documents],
            embeddings=[chunk.embedding for chunk in processed_documents],
            metadata_list=[chunk.metadata for chunk in processed_documents]
        )

    def should_include_in_query(self, query: str, classification: QueryClassification) -> bool:
        """Determine if Narodne Novine should be included in query."""

        # Legal keywords in Croatian
        legal_keywords = [
            "zakon", "pravilnik", "uredba", "odluka", "rješenje",
            "propis", "ministarstvo", "vlada", "sabor", "parlament",
            "službeni glasnik", "narodne novine", "službeno glasilo",
            "pravni", "pravno", "pravna", "zakonski", "zakonska",
            "regulativa", "norma", "normativ"
        ]

        query_lower = query.lower()

        # Check for legal keywords
        has_legal_keywords = any(keyword in query_lower for keyword in legal_keywords)

        # Check classification confidence
        is_legal_intent = classification.primary_intent == QueryIntent.LEGAL_OFFICIAL

        # Include if high confidence legal intent OR legal keywords present
        return is_legal_intent or (has_legal_keywords and classification.confidence > 0.6)
```

### 2. Enhanced Query Processing for Legal Content

```python
class LegalQueryProcessor:
    """Specialized query processing for legal/official content."""

    def process_legal_query(self, query: str, language: str) -> ProcessedLegalQuery:
        """Process query with legal domain awareness."""

        # Extract legal entities (laws, regulations, dates, authorities)
        legal_entities = self._extract_legal_entities(query, language)

        # Expand query with legal synonyms
        expanded_query = self._expand_legal_terminology(query, language)

        # Identify temporal constraints (years, date ranges)
        temporal_filters = self._extract_temporal_constraints(query)

        # Identify authority filters (ministry, government level)
        authority_filters = self._extract_authority_filters(query, language)

        return ProcessedLegalQuery(
            original_query=query,
            expanded_query=expanded_query,
            legal_entities=legal_entities,
            temporal_filters=temporal_filters,
            authority_filters=authority_filters,
            search_strategy="legal_precise"
        )

    def _extract_legal_entities(self, query: str, language: str) -> List[LegalEntity]:
        """Extract legal entities like law names, regulation numbers."""
        entities = []

        if language == "hr":
            # Croatian legal entity patterns
            import re

            # Find law references (e.g., "Zakon o radu", "ZOR")
            law_pattern = r"[Zz]akon(?:\s+o\s+[\w\s]+)|[A-Z]{2,5}(?=\s|$)"
            laws = re.findall(law_pattern, query)

            # Find NN numbers (e.g., "NN 85/09", "Narodne novine 123/2023")
            nn_pattern = r"(?:NN|Narodne novine)\s*(\d+/\d+)"
            nn_refs = re.findall(nn_pattern, query, re.IGNORECASE)

            # Find year references
            year_pattern = r"\b(19\d{2}|20\d{2})\b"
            years = re.findall(year_pattern, query)

            for law in laws:
                entities.append(LegalEntity(type="law", value=law.strip()))

            for nn_ref in nn_refs:
                entities.append(LegalEntity(type="nn_reference", value=nn_ref))

            for year in years:
                entities.append(LegalEntity(type="year", value=year))

        return entities
```

## Enhanced RAGSystem Integration

### 1. Modified RAGSystem with Multi-Source Support

```python
class EnhancedRAGSystem(RAGSystem):
    """Enhanced RAG system with multi-source query capabilities."""

    def __init__(
        self,
        language: str,
        tenant_context: TenantContext,
        # Original dependencies
        **original_dependencies,
        # New multi-source dependencies
        query_router: IntelligentQueryRouter,
        multi_source_retriever: MultiSourceRetriever,
        feature_registry: FeatureRegistry,
    ):
        # Initialize parent
        super().__init__(language=language, **original_dependencies)

        # Multi-source components
        self.tenant_context = tenant_context
        self.query_router = query_router
        self.multi_source_retriever = multi_source_retriever
        self.feature_registry = feature_registry

    async def query(
        self,
        query: RAGQuery,
        return_sources: bool = True,
        return_debug_info: bool = False,
        allow_multi_source: bool = True,
        user_preferences: Dict[str, Any] = None
    ) -> EnhancedRAGResponse:
        """Enhanced query with multi-source support."""

        if not allow_multi_source:
            # Fallback to original single-source behavior
            original_response = await super().query(query, return_sources, return_debug_info)
            return EnhancedRAGResponse.from_original(original_response)

        # Step 1: Classify query and create routing plan
        routing_plan = await self.query_router.classify_and_route(
            query=query.text,
            context=self.tenant_context,
            language=self.language,
            user_preferences=user_preferences
        )

        # Step 2: Execute multi-source retrieval
        multi_source_result = await self.multi_source_retriever.retrieve_multi_source(
            query=query.text,
            routing_plan=routing_plan,
            max_results_per_source=query.max_results or 10,
            final_result_limit=5
        )

        # Step 3: Generate response using aggregated context
        aggregated_chunks = [
            doc["content"] for doc in multi_source_result.aggregated_result.documents
        ]

        # Build enhanced context with source attribution
        context_with_sources = self._build_attributed_context(
            multi_source_result.aggregated_result.documents
        )

        # Use original generation pipeline with enhanced context
        generation_response = await self._generate_response_with_attribution(
            query=query,
            context_chunks=aggregated_chunks,
            attributed_context=context_with_sources,
            routing_plan=routing_plan
        )

        return EnhancedRAGResponse(
            answer=generation_response.answer,
            query=query.text,
            retrieved_chunks=multi_source_result.aggregated_result.documents,
            sources=self._extract_attributed_sources(multi_source_result),
            routing_plan=routing_plan,
            source_distribution=multi_source_result.aggregated_result.source_distribution,
            diversity_score=multi_source_result.aggregated_result.diversity_score,
            classification=routing_plan.classification,
            total_sources_queried=multi_source_result.aggregated_result.total_sources_queried,
            # Original fields
            confidence=generation_response.confidence,
            generation_time=generation_response.generation_time,
            retrieval_time=sum(m["retrieval_time"] for m in multi_source_result.metrics.values()),
            total_time=generation_response.total_time,
            metadata=generation_response.metadata
        )

    def _build_attributed_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context with source attribution for better generation."""

        attributed_sections = []

        for doc in documents:
            source_type = doc["source_type"]
            source_label = self._get_source_label(source_type, doc["source_collection"])

            section = f"[Izvor: {source_label}]\n{doc['content']}\n"
            attributed_sections.append(section)

        return "\n---\n".join(attributed_sections)

    def _get_source_label(self, source_type: str, collection_name: str) -> str:
        """Get human-readable source label."""
        labels = {
            "user_personal": "Osobni dokumenti",
            "tenant_shared": "Dijeljeni dokumenti tvrtke",
            "legal_official": "Narodne novine (službeno)"
        }
        return labels.get(source_type, "Nepoznat izvor")
```

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-2)
1. **Intent Classification System**
   - Implement `QueryClassification` and `IntelligentQueryRouter`
   - Create LLM-based intent classifier
   - Add configuration for query routing rules

2. **Collection Management Enhancement**
   - Extend collection naming patterns
   - Implement `CollectionManager` for multi-collection queries
   - Add tenant-wide collection discovery

### Phase 2: Multi-Source Retrieval (Weeks 3-4)
1. **Multi-Source Retriever**
   - Implement parallel collection querying
   - Create result aggregation and ranking system
   - Add cross-source reranking capabilities

2. **Enhanced Query Processing**
   - Modify `RAGSystem` to support multi-source queries
   - Implement backward compatibility mode
   - Add performance monitoring for multi-source queries

### Phase 3: Narodne Novine Integration (Weeks 5-6)
1. **Feature Framework**
   - Implement `FeatureRegistry` for feature-specific data sources
   - Create `NarodneNovineFeature` with specialized processing
   - Add legal document ingestion pipeline

2. **Legal Query Processing**
   - Implement Croatian legal entity extraction
   - Add legal terminology expansion
   - Create specialized retrieval strategies for legal content

### Phase 4: Integration & Testing (Weeks 7-8)
1. **CLI Integration**
   - Modify `MultiTenantRAGCLI` to support multi-source queries
   - Add user preference configuration
   - Implement feature toggle system

2. **Performance Optimization**
   - Add query result caching
   - Optimize parallel retrieval performance
   - Implement query complexity detection

## Configuration Examples

### 1. Multi-Source Query Configuration

```toml
# config/query_routing.toml
[query_routing]
enabled = true
default_max_sources = 3
classification_confidence_threshold = 0.6
enable_cross_source_reranking = true

[query_routing.intent_classification]
model = "qwen2.5:7b-instruct"
temperature = 0.1
max_tokens = 200

[query_routing.source_weights]
user_personal = 1.0
tenant_shared = 0.9
legal_official = 1.2

[query_routing.strategies]
personal_documents = "semantic_focused"
company_knowledge = "hybrid"
legal_official = "technical_precise"
general_inquiry = "adaptive"
```

### 2. Narodne Novine Feature Configuration

```toml
# config/features/narodne_novine.toml
[narodne_novine]
enabled = true
collection_name = "feature_narodne_novine_hr_documents"
supported_languages = ["hr"]
auto_include_threshold = 0.7

[narodne_novine.ingestion]
chunking_strategy = "legal_paragraph_aware"
max_chunk_size = 1000
overlap_size = 100
preserve_legal_structure = true

[narodne_novine.search]
default_strategy = "technical_precise"
boost_recent_documents = true
authority_level_weights = { "državna" = 1.0, "regionalna" = 0.8, "lokalna" = 0.6 }

[narodne_novine.legal_processing]
extract_entities = true
expand_terminology = true
temporal_filtering = true
authority_filtering = true
```

## Migration Strategy

### 1. Backward Compatibility
- All existing queries continue to work unchanged
- Multi-source mode is opt-in via parameter: `allow_multi_source=True`
- Current collection naming remains valid
- No breaking changes to existing APIs

### 2. Gradual Rollout
1. **Week 1**: Deploy intent classification (read-only mode)
2. **Week 2**: Enable multi-source for specific tenants (beta)
3. **Week 3**: Enable Narodne Novine for Croatian language
4. **Week 4**: Full rollout with feature flags

### 3. Feature Flags
```python
# Feature flag configuration
FEATURES = {
    "multi_source_queries": {
        "enabled": True,
        "rollout_percentage": 100,
        "tenant_whitelist": ["development", "beta-tenant"]
    },
    "narodne_novine": {
        "enabled": True,
        "language_restriction": ["hr"],
        "user_opt_in_required": True
    }
}
```

## Performance Considerations

### 1. Query Complexity Detection
- **Simple queries**: Single source (current behavior)
- **Medium queries**: 2-3 sources with parallel retrieval
- **Complex queries**: Full multi-source with aggregation

### 2. Caching Strategy
- Cache query classifications for similar queries
- Cache collection metadata for faster routing
- Cache feature availability by tenant/user

### 3. Resource Management
- Limit concurrent source queries per user
- Implement query timeout and fallback strategies
- Monitor and alert on performance degradation

## Expected Benefits

### 1. Enhanced Query Capabilities
- **Tenant-wide knowledge access**: Users can query company-wide information
- **Feature-specific expertise**: Legal/official document queries via Narodne Novine
- **Intelligent routing**: Automatic selection of relevant data sources
- **Better result diversity**: Results from multiple relevant sources

### 2. Croatian Legal Domain Support
- **Official document access**: Direct queries to Narodne Novine content
- **Legal entity recognition**: Automatic extraction of laws, regulations, references
- **Temporal and authority filtering**: Find documents by date, ministry, government level
- **Legal terminology expansion**: Better matching of legal concepts

### 3. Performance and User Experience
- **Parallel retrieval**: Multiple sources queried simultaneously
- **Adaptive complexity**: Simple queries remain fast, complex queries get more sources
- **Source transparency**: Users see which sources contributed to answers
- **Relevance optimization**: Source-specific ranking and weighting

## Next Steps

1. **Review and Approve Architecture**: Validate approach with team
2. **Create Implementation Plan**: Detail technical tasks and dependencies
3. **Set Up Development Environment**: Configure test collections and data
4. **Begin Phase 1 Implementation**: Start with intent classification system
5. **Prepare Test Data**: Create sample multi-source scenarios for validation

This enhancement maintains the current system's strengths while adding powerful multi-source capabilities that will significantly improve the RAG system's usefulness for both individual users and tenant-wide knowledge access, with special support for Croatian legal and official content through Narodne Novine integration.