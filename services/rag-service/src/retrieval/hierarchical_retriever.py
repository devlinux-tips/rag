"""
Pure function hierarchical retrieval system with dependency injection.
100% testable architecture with no side effects and deterministic output.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import (Any, Dict, List, Optional, Protocol, Tuple,
                    runtime_checkable)

from .categorization import CategoryMatch, DocumentCategory, RetrievalStrategy


class RetrievalStrategyType(Enum):
    """Enumeration of retrieval strategy types."""

    SEMANTIC_FOCUSED = "semantic_focused"
    KEYWORD_HYBRID = "keyword_hybrid"
    TECHNICAL_PRECISE = "technical_precise"
    TEMPORAL_AWARE = "temporal_aware"
    FAQ_OPTIMIZED = "faq_optimized"
    COMPARATIVE_STRUCTURED = "comparative_structured"
    DEFAULT = "default"


@dataclass
class ProcessedQuery:
    """Processed query information."""

    original: str
    processed: str
    query_type: str
    keywords: List[str]
    expanded_terms: List[str]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Individual search result."""

    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    final_score: float
    boosts: Dict[str, float]


@dataclass
class HierarchicalRetrievalResult:
    """Result of hierarchical retrieval with routing information."""

    documents: List[Dict[str, Any]]
    category: str
    strategy_used: str
    retrieval_time: float
    total_results: int
    confidence: float
    routing_metadata: Dict[str, Any]


@dataclass
class RetrievalConfig:
    """Configuration for hierarchical retrieval."""

    default_max_results: int
    similarity_thresholds: Dict[str, float]
    boost_weights: Dict[str, float]
    strategy_mappings: Dict[str, str]
    performance_tracking: bool


@runtime_checkable
class QueryProcessor(Protocol):
    """Protocol for query processing operations."""

    def process_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ProcessedQuery:
        """Process query into structured format."""
        ...


@runtime_checkable
class Categorizer(Protocol):
    """Protocol for query categorization operations."""

    def categorize_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CategoryMatch:
        """Categorize query and determine retrieval strategy."""
        ...


@runtime_checkable
class SearchEngine(Protocol):
    """Protocol for search engine operations."""

    async def search(
        self, query_text: str, k: int = 5, similarity_threshold: float = 0.3
    ) -> List[SearchResult]:
        """Execute semantic search."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for document reranking operations."""

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance."""
        ...


@runtime_checkable
class LoggerProvider(Protocol):
    """Protocol for logging operations."""

    def info(self, message: str) -> None:
        """Log info message."""
        ...

    def debug(self, message: str) -> None:
        """Log debug message."""
        ...

    def error(self, message: str) -> None:
        """Log error message."""
        ...


# ================================
# PURE BUSINESS LOGIC FUNCTIONS
# ================================


def calculate_keyword_boost(
    content: str, keywords: List[str], boost_weight: float = 0.2
) -> float:
    """
    Calculate keyword match boost for content.
    Pure function with no side effects.

    Args:
        content: Document content to analyze
        keywords: Keywords to match against
        boost_weight: Weight for keyword matching

    Returns:
        Boost score based on keyword matches
    """
    if not keywords or not content:
        return 0.0

    content_lower = content.lower()
    keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
    return (keyword_matches / len(keywords)) * boost_weight


def calculate_technical_boost(content: str, boost_weight: float = 0.1) -> float:
    """
    Calculate technical content boost.
    Pure function with no side effects.

    Args:
        content: Document content to analyze
        boost_weight: Weight for each technical indicator

    Returns:
        Technical content boost score
    """
    technical_indicators = [
        "api",
        "kod",
        "programiranje",
        "algoritam",
        "software",
        "sistem",
        "tehnologija",
        "development",
        "programming",
        "technical",
        "implementation",
    ]

    content_lower = content.lower()
    technical_score = sum(
        boost_weight for indicator in technical_indicators if indicator in content_lower
    )

    return min(technical_score, boost_weight * 5)  # Cap the boost


def calculate_temporal_boost(
    content: str,
    metadata: Dict[str, Any],
    current_year: int = 2024,
    boost_weight: float = 0.15,
) -> float:
    """
    Calculate temporal relevance boost.
    Pure function with no side effects.

    Args:
        content: Document content to analyze
        metadata: Document metadata
        current_year: Current year for recency calculation
        boost_weight: Weight for temporal indicators

    Returns:
        Temporal relevance boost score
    """
    temporal_terms = [
        "danas",
        "jučer",
        "nedavno",
        "trenutno",
        "novo",
        "aktualno",
        "today",
        "yesterday",
        "recent",
        "current",
        "new",
        "latest",
    ]

    content_lower = content.lower()
    temporal_score = (
        sum(boost_weight for term in temporal_terms if term in content_lower) * 0.5
    )  # Limit temporal boost

    # Recent content boost based on metadata
    if "year" in metadata:
        try:
            doc_year = int(metadata["year"])
            if doc_year >= current_year - 1:  # Recent content
                temporal_score += 0.2
            elif doc_year >= current_year - 3:  # Moderately recent
                temporal_score += 0.1
        except (ValueError, TypeError):
            pass

    return temporal_score


def calculate_faq_boost(content: str, boost_weight: float = 0.1) -> float:
    """
    Calculate FAQ content boost.
    Pure function with no side effects.

    Args:
        content: Document content to analyze
        boost_weight: Weight for FAQ indicators

    Returns:
        FAQ content boost score
    """
    faq_indicators = [
        "pitanje",
        "odgovor",
        "često",
        "faq",
        "q:",
        "a:",
        "question",
        "answer",
        "frequently",
        "kako",
        "what",
        "why",
        "zašto",
        "što",
    ]

    content_lower = content.lower()
    faq_score = (
        sum(boost_weight for indicator in faq_indicators if indicator in content_lower)
        * 0.3
    )  # Moderate FAQ boost

    # Question-answer structure boost
    if any(
        pattern in content_lower for pattern in ["q:", "a:", "pitanje:", "odgovor:"]
    ):
        faq_score += 0.2

    # Short, concise answer boost (FAQ answers are usually brief)
    if 50 <= len(content) <= 300:
        faq_score += 0.1

    return faq_score


def calculate_comparative_boost(content: str, boost_weight: float = 0.1) -> float:
    """
    Calculate comparative content boost.
    Pure function with no side effects.

    Args:
        content: Document content to analyze
        boost_weight: Weight for comparative indicators

    Returns:
        Comparative content boost score
    """
    comparative_terms = [
        "usporedi",
        "razlika",
        "različit",
        "slično",
        "protiv",
        "vs",
        "versus",
        "compare",
        "difference",
        "different",
        "similar",
        "against",
        "better",
        "bolje",
    ]

    content_lower = content.lower()
    comparative_score = (
        sum(boost_weight for term in comparative_terms if term in content_lower) * 0.4
    )

    # Structure indicators boost (tables, lists, comparisons)
    structure_indicators = ["|", "vs", "•", "-", "1.", "2.", "prvo", "drugo"]
    structure_score = sum(
        0.05 for indicator in structure_indicators if indicator in content
    )

    return comparative_score + structure_score


def calculate_exact_match_boost(
    content: str, query_words: List[str], boost_weight: float = 0.2
) -> float:
    """
    Calculate exact term matching boost.
    Pure function with no side effects.

    Args:
        content: Document content to analyze
        query_words: Original query words for matching
        boost_weight: Weight for exact matches

    Returns:
        Exact match boost score
    """
    if not query_words:
        return 0.0

    content_words = set(content.lower().split())
    query_words_set = set(word.lower() for word in query_words)
    exact_matches = len(query_words_set.intersection(content_words))

    return (exact_matches / len(query_words_set)) * boost_weight


def apply_strategy_specific_processing(
    results: List[SearchResult],
    strategy: RetrievalStrategyType,
    processed_query: ProcessedQuery,
    config: RetrievalConfig,
) -> List[SearchResult]:
    """
    Apply strategy-specific processing to search results.
    Pure function with no side effects.

    Args:
        results: List of search results to process
        strategy: Retrieval strategy to apply
        processed_query: Processed query information
        config: Retrieval configuration

    Returns:
        Processed search results with strategy-specific boosts
    """
    if not results:
        return results

    processed_results = []

    for result in results:
        boosts = {}
        total_boost = 0.0

        # Apply strategy-specific boosts
        if strategy == RetrievalStrategyType.SEMANTIC_FOCUSED:
            # Pure semantic score, no additional boosts
            pass

        elif strategy == RetrievalStrategyType.KEYWORD_HYBRID:
            keyword_boost = calculate_keyword_boost(
                result.content,
                processed_query.keywords,
                config.boost_weights.get("keyword", 0.2),
            )
            boosts["keyword"] = keyword_boost
            total_boost += keyword_boost

        elif strategy == RetrievalStrategyType.TECHNICAL_PRECISE:
            technical_boost = calculate_technical_boost(
                result.content, config.boost_weights.get("technical", 0.1)
            )
            exact_boost = calculate_exact_match_boost(
                result.content,
                processed_query.original.split(),
                config.boost_weights.get("exact_match", 0.2),
            )
            boosts["technical"] = technical_boost
            boosts["exact_match"] = exact_boost
            total_boost += technical_boost + exact_boost

        elif strategy == RetrievalStrategyType.TEMPORAL_AWARE:
            temporal_boost = calculate_temporal_boost(
                result.content,
                result.metadata,
                boost_weight=config.boost_weights.get("temporal", 0.15),
            )
            boosts["temporal"] = temporal_boost
            total_boost += temporal_boost

        elif strategy == RetrievalStrategyType.FAQ_OPTIMIZED:
            faq_boost = calculate_faq_boost(
                result.content, config.boost_weights.get("faq", 0.1)
            )
            boosts["faq"] = faq_boost
            total_boost += faq_boost

        elif strategy == RetrievalStrategyType.COMPARATIVE_STRUCTURED:
            comparative_boost = calculate_comparative_boost(
                result.content, config.boost_weights.get("comparative", 0.1)
            )
            boosts["comparative"] = comparative_boost
            total_boost += comparative_boost

        # Create processed result
        final_score = min(1.0, result.similarity_score + total_boost)

        processed_result = SearchResult(
            content=result.content,
            metadata=result.metadata,
            similarity_score=result.similarity_score,
            final_score=final_score,
            boosts=boosts,
        )

        processed_results.append(processed_result)

    # Sort by final score
    processed_results.sort(key=lambda x: x.final_score, reverse=True)

    return processed_results


def filter_results_by_threshold(
    results: List[SearchResult], similarity_threshold: float
) -> List[SearchResult]:
    """
    Filter results by similarity threshold.
    Pure function with no side effects.

    Args:
        results: Search results to filter
        similarity_threshold: Minimum similarity score required

    Returns:
        Filtered search results
    """
    return [
        result for result in results if result.similarity_score >= similarity_threshold
    ]


def calculate_overall_confidence(
    category_confidence: float,
    top_results: List[SearchResult],
    weights: Tuple[float, float] = (0.6, 0.4),
) -> float:
    """
    Calculate overall retrieval confidence.
    Pure function with no side effects.

    Args:
        category_confidence: Confidence from categorization
        top_results: Top search results for scoring
        weights: Weights for (category_confidence, results_confidence)

    Returns:
        Overall confidence score
    """
    category_weight, results_weight = weights

    if not top_results:
        return category_confidence * category_weight

    # Average of top 3 results
    top_3 = top_results[:3]
    results_confidence = sum(result.final_score for result in top_3) / len(top_3)

    return category_confidence * category_weight + results_confidence * results_weight


def create_routing_metadata(
    processed_query: ProcessedQuery,
    categorization: CategoryMatch,
    strategy_used: RetrievalStrategyType,
    results: List[SearchResult],
    retrieval_time: float,
    reranking_applied: bool,
) -> Dict[str, Any]:
    """
    Create routing metadata for result tracking.
    Pure function with no side effects.

    Args:
        processed_query: Processed query information
        categorization: Query categorization result
        strategy_used: Strategy that was used
        results: Final search results
        retrieval_time: Time taken for retrieval
        reranking_applied: Whether reranking was applied

    Returns:
        Routing metadata dictionary
    """
    return {
        "query_processing": {
            "original": processed_query.original,
            "processed": processed_query.processed,
            "query_type": processed_query.query_type,
            "keywords": processed_query.keywords,
            "expanded_terms": processed_query.expanded_terms,
        },
        "categorization": {
            "primary": categorization.category.value,
            "confidence": categorization.confidence,
            "matched_patterns": categorization.matched_patterns,
            "cultural_indicators": categorization.cultural_indicators,
            "complexity": categorization.complexity.value,
        },
        "strategy": {
            "selected": strategy_used.value,
            "retrieval_strategy": categorization.retrieval_strategy,
        },
        "performance": {
            "retrieval_time": retrieval_time,
            "results_count": len(results),
            "reranking_applied": reranking_applied,
        },
    }


# ================================
# DEPENDENCY INJECTION ORCHESTRATION
# ================================


class HierarchicalRetriever:
    """Hierarchical retriever with dependency injection for 100% testability."""

    def __init__(
        self,
        query_processor: QueryProcessor,
        categorizer: Categorizer,
        search_engine: SearchEngine,
        config: RetrievalConfig,
        reranker: Optional[Reranker] = None,
        logger_provider: Optional[LoggerProvider] = None,
    ):
        """Initialize hierarchical retriever with injected dependencies."""
        self._query_processor = query_processor
        self._categorizer = categorizer
        self._search_engine = search_engine
        self._reranker = reranker
        self._logger = logger_provider
        self._config = config

        # Performance tracking (optional)
        self._retrieval_count = 0
        self._strategy_stats = {} if config.performance_tracking else None

    async def retrieve(
        self,
        query: str,
        max_results: int = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> HierarchicalRetrievalResult:
        """
        Execute hierarchical retrieval with intelligent routing.

        Args:
            query: Query string to process
            max_results: Maximum results to return (uses config default if None)
            context: Optional context for query processing

        Returns:
            HierarchicalRetrievalResult with routing metadata
        """
        start_time = time.time()
        max_results = max_results or self._config.default_max_results
        context = context or {}

        self._log_info(f"🎯 Hierarchical retrieval for: {query[:50]}...")

        try:
            # Step 1: Process and categorize query
            processed_query = self._query_processor.process_query(query, context)
            categorization = self._categorizer.categorize_query(query, context)

            self._log_info(
                f"📂 Category: {categorization.category.value} "
                f"(confidence: {categorization.confidence:.3f})"
            )

            # Step 2: Map strategy and get threshold
            strategy_type = self._map_retrieval_strategy(
                categorization.retrieval_strategy
            )
            similarity_threshold = self._config.similarity_thresholds.get(
                strategy_type.value, 0.3
            )

            # Step 3: Execute search with expanded results for processing
            expanded_results = max_results * 2
            raw_results = await self._search_engine.search(
                query_text=processed_query.processed,
                k=expanded_results,
                similarity_threshold=similarity_threshold,
            )

            # Step 4: Apply strategy-specific processing
            processed_results = apply_strategy_specific_processing(
                raw_results, strategy_type, processed_query, self._config
            )

            # Step 5: Filter and limit results
            filtered_results = filter_results_by_threshold(
                processed_results, similarity_threshold
            )[:max_results]

            # Step 6: Convert to dict format for compatibility
            result_dicts = [
                {
                    "content": result.content,
                    "metadata": {
                        **result.metadata,
                        "detected_category": categorization.category.value,
                    },
                    "similarity_score": result.similarity_score,
                    "final_score": result.final_score,
                    **result.boosts,  # Add boost scores
                }
                for result in filtered_results
            ]

            # Step 7: Apply reranking if available
            if self._reranker and len(result_dicts) > 1:
                self._log_debug("🔄 Applying reranking...")
                result_dicts = await self._reranker.rerank(
                    query=query,
                    documents=result_dicts,
                    category=categorization.category.value,
                )

            # Step 8: Calculate metrics and metadata
            retrieval_time = time.time() - start_time
            overall_confidence = calculate_overall_confidence(
                categorization.confidence, filtered_results
            )
            routing_metadata = create_routing_metadata(
                processed_query,
                categorization,
                strategy_type,
                filtered_results,
                retrieval_time,
                self._reranker is not None,
            )

            # Update statistics
            if self._strategy_stats is not None:
                self._update_strategy_stats(strategy_type, retrieval_time)

            result = HierarchicalRetrievalResult(
                documents=result_dicts,
                category=categorization.category.value,
                strategy_used=strategy_type.value,
                retrieval_time=retrieval_time,
                total_results=len(result_dicts),
                confidence=overall_confidence,
                routing_metadata=routing_metadata,
            )

            self._log_info(
                f"✅ Hierarchical retrieval complete: {len(result_dicts)} results "
                f"in {retrieval_time:.3f}s using {strategy_type.value}"
            )

            return result

        except Exception as e:
            self._log_error(f"❌ Hierarchical retrieval failed: {e}")

            # Return minimal fallback result
            return HierarchicalRetrievalResult(
                documents=[],
                category="general",
                strategy_used="default",
                retrieval_time=time.time() - start_time,
                total_results=0,
                confidence=0.0,
                routing_metadata={"error": str(e), "fallback_used": True},
            )

    def _map_retrieval_strategy(self, strategy_name: str) -> RetrievalStrategyType:
        """Map strategy name to RetrievalStrategyType."""
        strategy_mapping = {
            "hybrid": RetrievalStrategyType.KEYWORD_HYBRID,
            "dense": RetrievalStrategyType.SEMANTIC_FOCUSED,
            "cultural_context": RetrievalStrategyType.SEMANTIC_FOCUSED,
            "cultural_aware": RetrievalStrategyType.SEMANTIC_FOCUSED,
            "sparse": RetrievalStrategyType.TECHNICAL_PRECISE,
            "hierarchical": RetrievalStrategyType.COMPARATIVE_STRUCTURED,
            "default": RetrievalStrategyType.DEFAULT,
            "semantic_focused": RetrievalStrategyType.SEMANTIC_FOCUSED,
            "technical_precise": RetrievalStrategyType.TECHNICAL_PRECISE,
            "temporal_aware": RetrievalStrategyType.TEMPORAL_AWARE,
            "faq_optimized": RetrievalStrategyType.FAQ_OPTIMIZED,
            "comparative_structured": RetrievalStrategyType.COMPARATIVE_STRUCTURED,
        }

        return strategy_mapping.get(strategy_name, RetrievalStrategyType.DEFAULT)

    def _update_strategy_stats(
        self, strategy: RetrievalStrategyType, retrieval_time: float
    ) -> None:
        """Update performance statistics for strategies."""
        if not self._strategy_stats:
            return

        strategy_name = strategy.value

        if strategy_name not in self._strategy_stats:
            self._strategy_stats[strategy_name] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
            }

        stats = self._strategy_stats[strategy_name]
        stats["count"] += 1
        stats["total_time"] += retrieval_time
        stats["avg_time"] = stats["total_time"] / stats["count"]

        self._retrieval_count += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_retrievals": self._retrieval_count,
            "strategy_stats": self._strategy_stats.copy()
            if self._strategy_stats
            else {},
            "reranking_enabled": self._reranker is not None,
            "performance_tracking": self._config.performance_tracking,
        }

    def _log_info(self, message: str) -> None:
        """Log info message if logger available."""
        if self._logger:
            self._logger.info(message)

    def _log_debug(self, message: str) -> None:
        """Log debug message if logger available."""
        if self._logger:
            self._logger.debug(message)

    def _log_error(self, message: str) -> None:
        """Log error message if logger available."""
        if self._logger:
            self._logger.error(message)


# ================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# ================================


def create_hierarchical_retriever(
    search_engine=None,
    language: str = "hr",
    enable_reranking: bool = True,
    query_processor: QueryProcessor = None,
    categorizer: Categorizer = None,
    config: Optional[RetrievalConfig] = None,
    reranker: Optional[Reranker] = None,
    logger_provider: Optional[LoggerProvider] = None,
) -> HierarchicalRetriever:
    """
    Convenience function for creating hierarchical retriever.
    Supports both legacy interface and new dependency injection interface.

    Args:
        search_engine: Search engine component (legacy or new interface)
        language: Language for components (legacy interface)
        enable_reranking: Enable reranking (legacy interface)
        query_processor: Query processing component (new interface)
        categorizer: Query categorization component (new interface)
        config: Retrieval configuration (uses defaults if None)
        reranker: Optional reranker component (new interface)
        logger_provider: Optional logger provider (new interface)

    Returns:
        Configured HierarchicalRetriever instance
    """
    # New interface - all components provided via DI
    if query_processor and categorizer and search_engine:
        if config is None:
            config = RetrievalConfig(
                default_max_results=5,
                similarity_thresholds={
                    "semantic_focused": 0.5,
                    "keyword_hybrid": 0.4,
                    "technical_precise": 0.6,
                    "temporal_aware": 0.4,
                    "faq_optimized": 0.3,
                    "comparative_structured": 0.4,
                    "default": 0.3,
                },
                boost_weights={
                    "keyword": 0.2,
                    "technical": 0.1,
                    "exact_match": 0.2,
                    "temporal": 0.15,
                    "faq": 0.1,
                    "comparative": 0.1,
                },
                strategy_mappings={},
                performance_tracking=True,
            )

        return HierarchicalRetriever(
            query_processor=query_processor,
            categorizer=categorizer,
            search_engine=search_engine,
            config=config,
            reranker=reranker,
            logger_provider=logger_provider,
        )

    # Legacy interface - create components using providers
    elif search_engine:
        from .hierarchical_retriever_providers import create_production_setup

        components = create_production_setup(
            search_engine=search_engine,
            language=language,
            reranker=None,  # Legacy interface doesn't pass reranker directly
            enable_performance_tracking=True,
        )

        (
            query_processor,
            categorizer,
            search_engine_adapter,
            _,
            logger,
            config,
        ) = components

        # Create reranker if requested (legacy style)
        reranker_component = None
        if enable_reranking:
            try:
                from .reranker import DocumentReranker

                reranker_instance = DocumentReranker(language=language)
                from .hierarchical_retriever_providers import \
                    ProductionRerankerAdapter

                reranker_component = ProductionRerankerAdapter(
                    reranker_instance, language
                )
            except ImportError:
                pass  # Reranker not available

        return HierarchicalRetriever(
            query_processor=query_processor,
            categorizer=categorizer,
            search_engine=search_engine_adapter,
            config=config,
            reranker=reranker_component,
            logger_provider=logger,
        )

    else:
        raise ValueError(
            "Must provide either (query_processor, categorizer, search_engine) for new interface "
            "or (search_engine) for legacy interface"
        )


# ================================
# BACKWARD COMPATIBILITY ALIASES
# ================================


# Legacy class that matches the old HierarchicalRetriever interface
class LegacyHierarchicalRetriever:
    """
    Legacy hierarchical retriever for backward compatibility.
    Maps old interface to new dependency injection interface.
    """

    def __init__(
        self,
        search_engine,
        language: str = "hr",
        enable_reranking: bool = True,
    ):
        """Initialize with legacy interface pattern."""
        self.search_engine = search_engine
        self.language = language
        self.enable_reranking = enable_reranking

        # Create using legacy factory function
        self._retriever = create_hierarchical_retriever(
            search_engine=search_engine,
            language=language,
            enable_reranking=enable_reranking,
        )

    async def retrieve(
        self,
        query: str,
        max_results: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> HierarchicalRetrievalResult:
        """Legacy retrieve method interface."""
        return await self._retriever.retrieve(query, max_results, context)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Legacy performance stats method."""
        return self._retriever.get_performance_stats()

    async def health_check(self) -> Dict[str, Any]:
        """Legacy health check method."""
        stats = self._retriever.get_performance_stats()

        return {
            "status": "healthy",
            "components": {
                "search_engine": "healthy" if self.search_engine else "missing",
                "query_processor": "healthy",
                "categorizer": "healthy",
                "reranker": "healthy" if self.enable_reranking else "disabled",
            },
            "statistics": stats,
            "language": self.language,
        }


# Make the legacy class available as the original name
HierarchicalRetrieverLegacy = LegacyHierarchicalRetriever
