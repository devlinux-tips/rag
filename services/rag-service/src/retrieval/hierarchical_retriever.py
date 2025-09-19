"""
Pure function hierarchical retrieval system with dependency injection.
Clean architecture with no side effects and deterministic output.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_data_transformation,
    log_decision_point,
    log_performance_metric,
)
from .categorization import CategoryMatch


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
    keywords: list[str]
    expanded_terms: list[str]
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    """Individual search result."""

    content: str
    metadata: dict[str, Any]
    similarity_score: float
    final_score: float
    boosts: dict[str, float]


@dataclass
class HierarchicalRetrievalResult:
    """Result of hierarchical retrieval with routing information."""

    documents: list[dict[str, Any]]
    category: str
    strategy_used: str
    retrieval_time: float
    total_results: int
    confidence: float
    routing_metadata: dict[str, Any]


@dataclass
class RetrievalConfig:
    """Configuration for hierarchical retrieval."""

    default_max_results: int
    similarity_thresholds: dict[str, float]
    boost_weights: dict[str, float]
    strategy_mappings: dict[str, str]
    performance_tracking: bool


@runtime_checkable
class QueryProcessor(Protocol):
    """Protocol for query processing operations."""

    def process_query(self, query: str, context: dict[str, Any] | None = None) -> ProcessedQuery:
        """Process query into structured format."""
        ...


@runtime_checkable
class Categorizer(Protocol):
    """Protocol for query categorization operations."""

    def categorize_query(self, query: str, context: dict[str, Any] | None = None) -> CategoryMatch:
        """Categorize query and determine retrieval strategy."""
        ...


@runtime_checkable
class SearchEngine(Protocol):
    """Protocol for search engine operations."""

    async def search(self, query_text: str, k: int = 5, similarity_threshold: float = 0.3) -> list[SearchResult]:
        """Execute semantic search."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for document reranking operations."""

    async def rerank(
        self, query: str, documents: list[dict[str, Any]], category: str | None = None
    ) -> list[dict[str, Any]]:
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


def calculate_keyword_boost(content: str, keywords: list[str], boost_weight: float = 0.2) -> float:
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
    technical_score = sum(boost_weight for indicator in technical_indicators if indicator in content_lower)

    return min(technical_score, boost_weight * 5)  # Cap the boost


def calculate_temporal_boost(
    content: str, metadata: dict[str, Any], current_year: int = 2024, boost_weight: float = 0.15
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
    temporal_score = sum(boost_weight for term in temporal_terms if term in content_lower) * 0.5  # Limit temporal boost

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
        sum(boost_weight for indicator in faq_indicators if indicator in content_lower) * 0.3
    )  # Moderate FAQ boost

    # Question-answer structure boost
    if any(pattern in content_lower for pattern in ["q:", "a:", "pitanje:", "odgovor:"]):
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
    comparative_score = sum(boost_weight for term in comparative_terms if term in content_lower) * 0.4

    # Structure indicators boost (tables, lists, comparisons)
    structure_indicators = ["|", "vs", "•", "-", "1.", "2.", "prvo", "drugo"]
    structure_score = sum(0.05 for indicator in structure_indicators if indicator in content)

    return comparative_score + structure_score


def calculate_exact_match_boost(content: str, query_words: list[str], boost_weight: float = 0.2) -> float:
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
    results: list[SearchResult],
    strategy: RetrievalStrategyType,
    processed_query: ProcessedQuery,
    config: RetrievalConfig,
) -> list[SearchResult]:
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
            if "keyword" not in config.boost_weights:
                raise ValueError("Missing 'keyword' weight in boost configuration")
            keyword_boost = calculate_keyword_boost(
                result.content, processed_query.keywords, config.boost_weights["keyword"]
            )
            boosts["keyword"] = keyword_boost
            total_boost += keyword_boost

        elif strategy == RetrievalStrategyType.TECHNICAL_PRECISE:
            if "technical" not in config.boost_weights:
                raise ValueError("Missing 'technical' weight in boost configuration")
            if "exact_match" not in config.boost_weights:
                raise ValueError("Missing 'exact_match' weight in boost configuration")
            technical_boost = calculate_technical_boost(result.content, config.boost_weights["technical"])
            exact_boost = calculate_exact_match_boost(
                result.content, processed_query.original.split(), config.boost_weights["exact_match"]
            )
            boosts["technical"] = technical_boost
            boosts["exact_match"] = exact_boost
            total_boost += technical_boost + exact_boost

        elif strategy == RetrievalStrategyType.TEMPORAL_AWARE:
            if "temporal" not in config.boost_weights:
                raise ValueError("Missing 'temporal' weight in boost configuration")
            temporal_boost = calculate_temporal_boost(
                result.content, result.metadata, boost_weight=config.boost_weights["temporal"]
            )
            boosts["temporal"] = temporal_boost
            total_boost += temporal_boost

        elif strategy == RetrievalStrategyType.FAQ_OPTIMIZED:
            if "faq" not in config.boost_weights:
                raise ValueError("Missing 'faq' weight in boost configuration")
            faq_boost = calculate_faq_boost(result.content, config.boost_weights["faq"])
            boosts["faq"] = faq_boost
            total_boost += faq_boost

        elif strategy == RetrievalStrategyType.COMPARATIVE_STRUCTURED:
            if "comparative" not in config.boost_weights:
                raise ValueError("Missing 'comparative' weight in boost configuration")
            comparative_boost = calculate_comparative_boost(result.content, config.boost_weights["comparative"])
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


def filter_results_by_threshold(results: list[SearchResult], similarity_threshold: float) -> list[SearchResult]:
    """
    Filter results by similarity threshold.
    Pure function with no side effects.

    Args:
        results: Search results to filter
        similarity_threshold: Minimum similarity score required

    Returns:
        Filtered search results
    """
    return [result for result in results if result.similarity_score >= similarity_threshold]


def calculate_overall_confidence(
    category_confidence: float, top_results: list[SearchResult], weights: tuple[float, float] = (0.6, 0.4)
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
    results: list[SearchResult],
    retrieval_time: float,
    reranking_applied: bool,
) -> dict[str, Any]:
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
        "strategy": {"selected": strategy_used.value, "retrieval_strategy": categorization.retrieval_strategy},
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
    """Hierarchical retriever with dependency injection for testability."""

    def __init__(
        self,
        query_processor: QueryProcessor,
        categorizer: Categorizer,
        search_engine: SearchEngine,
        config: RetrievalConfig,
        reranker: Reranker | None = None,
        logger_provider: LoggerProvider | None = None,
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
        self._strategy_stats: dict[str, Any] | None = {} if config.performance_tracking else None

    async def retrieve(
        self, query: str, max_results: int | None = None, context: dict[str, Any] | None = None
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

        # Add structured logging for AI debugging
        logger = get_system_logger()
        log_component_start(
            "hierarchical_retriever",
            "retrieve",
            query_length=len(query),
            max_results=max_results,
            context_keys=list(context.keys()) if context else [],
            retrieval_count=self._retrieval_count + 1,
        )

        # Increment retrieval counter for performance tracking
        self._retrieval_count += 1

        self._log_info(f"🎯 Hierarchical retrieval for: {query[:50]}...")
        logger.debug("hierarchical_retriever", "retrieve", f"Processing query: '{query[:100]}...'")

        # Step 1: Process and categorize query
        logger.debug("hierarchical_retriever", "retrieve", "Step 1: Query processing and categorization")
        processed_query = self._query_processor.process_query(query, context)
        categorization = self._categorizer.categorize_query(query, context)

        log_data_transformation(
            "hierarchical_retriever",
            "query_processing",
            f"query[{len(query)}chars]",
            f"processed[{len(processed_query.processed)}chars]",
            original_query=query[:50],
            processed_query=processed_query.processed[:50],
        )

        log_decision_point(
            "hierarchical_retriever",
            "categorization",
            f"category={categorization.category.value}",
            f"confidence={categorization.confidence:.3f}",
            strategy=categorization.retrieval_strategy,
        )

        self._log_info(f"📂 Category: {categorization.category.value} (confidence: {categorization.confidence:.3f})")

        # Step 2: Map strategy and get threshold
        strategy_type = self._map_retrieval_strategy(categorization.retrieval_strategy)
        if strategy_type.value not in self._config.similarity_thresholds:
            raise ValueError(f"Missing similarity threshold for strategy '{strategy_type.value}'")
        similarity_threshold = self._config.similarity_thresholds[strategy_type.value]

        # Step 3: Execute search with expanded results for processing
        expanded_results = max_results * 2
        raw_results = await self._search_engine.search(
            query_text=processed_query.processed, k=expanded_results, similarity_threshold=similarity_threshold
        )

        # Step 4: Apply strategy-specific processing
        processed_results = apply_strategy_specific_processing(
            raw_results, strategy_type, processed_query, self._config
        )

        # Step 5: Filter and limit results
        filtered_results = filter_results_by_threshold(processed_results, similarity_threshold)[:max_results]

        # Step 6: Convert to dict format for compatibility
        self._log_debug(f"🔧 Converting {len(filtered_results)} results to dict format...")
        result_dicts = []
        for i, result in enumerate(filtered_results):
            self._log_debug(f"🔧 Result {i}: boosts type={type(result.boosts)}, value={result.boosts}")
            try:
                result_dict = {
                    "content": result.content,
                    "metadata": {**result.metadata, "detected_category": categorization.category.value},
                    "similarity_score": result.similarity_score,
                    "final_score": result.final_score,
                    **result.boosts,  # Add boost scores
                }
                result_dicts.append(result_dict)
            except Exception as e:
                self._log_error(f"🚨 Error unpacking result {i} boosts: {e}")
                self._log_error(f"🚨 Result boosts: type={type(result.boosts)}, value={result.boosts}")
                raise

        # Step 7: Apply reranking if available
        if self._reranker and len(result_dicts) > 1:
            self._log_debug("🔄 Applying reranking...")
            result_dicts = await self._reranker.rerank(
                query=query, documents=result_dicts, category=categorization.category.value
            )

        # Step 8: Calculate metrics and metadata
        retrieval_time = time.time() - start_time
        overall_confidence = calculate_overall_confidence(categorization.confidence, filtered_results)
        routing_metadata = create_routing_metadata(
            processed_query, categorization, strategy_type, filtered_results, retrieval_time, self._reranker is not None
        )

        # Update statistics
        if self._strategy_stats is not None:
            self._update_strategy_stats(strategy_type, retrieval_time)

        final_result = HierarchicalRetrievalResult(
            documents=result_dicts,
            category=categorization.category.value,
            strategy_used=strategy_type.value,
            retrieval_time=retrieval_time,
            total_results=len(result_dicts),
            confidence=overall_confidence,
            routing_metadata=routing_metadata,
        )

        # Log comprehensive performance metrics for AI debugging
        log_performance_metric("hierarchical_retriever", "retrieve", "retrieval_time_ms", retrieval_time * 1000)
        log_performance_metric("hierarchical_retriever", "retrieve", "results_count", len(result_dicts))
        log_performance_metric("hierarchical_retriever", "retrieve", "overall_confidence", overall_confidence)
        log_performance_metric("hierarchical_retriever", "retrieve", "similarity_threshold", similarity_threshold)

        log_data_transformation(
            "hierarchical_retriever",
            "result_filtering",
            f"raw_results[{len(raw_results)}]",
            f"final_results[{len(result_dicts)}]",
            strategy=strategy_type.value,
            reranked=self._reranker is not None,
        )

        self._log_info(
            f"✅ Hierarchical retrieval complete: {len(result_dicts)} results "
            f"in {retrieval_time:.3f}s using {strategy_type.value}"
        )

        logger.info(
            "hierarchical_retriever",
            "retrieve",
            f"Retrieval complete: {len(result_dicts)} results, confidence={overall_confidence:.3f}, strategy={strategy_type.value}",
        )

        log_component_end(
            "hierarchical_retriever",
            "retrieve",
            f"Retrieved {len(result_dicts)} results in {retrieval_time:.3f}s",
            results_count=len(result_dicts),
            retrieval_time=retrieval_time,
            strategy=strategy_type.value,
            confidence=overall_confidence,
        )

        return final_result

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

        if strategy_name not in strategy_mapping:
            raise ValueError(f"Unknown retrieval strategy: '{strategy_name}'")
        return strategy_mapping[strategy_name]

    def _update_strategy_stats(self, strategy: RetrievalStrategyType, retrieval_time: float) -> None:
        """Update performance statistics for strategies."""
        if self._strategy_stats is None:
            return

        strategy_name = strategy.value

        if strategy_name not in self._strategy_stats:
            self._strategy_stats[strategy_name] = {"count": 0, "total_time": 0.0, "avg_time": 0.0}

        stats = self._strategy_stats[strategy_name]
        stats["count"] += 1
        stats["total_time"] += retrieval_time
        stats["avg_time"] = stats["total_time"] / stats["count"]

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_retrievals": self._retrieval_count,
            "strategy_stats": (self._strategy_stats.copy() if self._strategy_stats else {}),
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
