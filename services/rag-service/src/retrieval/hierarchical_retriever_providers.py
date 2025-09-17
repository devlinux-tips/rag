"""
Provider implementations for hierarchical retriever dependency injection.
Production and mock providers for testable hierarchical retrieval system.
"""

from typing import Any

from .categorization import CategoryMatch, CategoryType, QueryComplexity
from .hierarchical_retriever import ProcessedQuery, RetrievalConfig, SearchResult


class MockQueryProcessor:
    """Mock query processor for testing."""

    def __init__(self, mock_responses: dict[str, ProcessedQuery] | None | None = None):
        """Initialize with optional mock responses."""
        self.mock_responses = mock_responses or {}
        self.call_history: list[dict[str, Any]] = []

    def set_mock_response(self, query: str, response: ProcessedQuery) -> None:
        """Set mock response for specific query."""
        self.mock_responses[query] = response

    def process_query(self, query: str, context: dict[str, Any] | None = None) -> ProcessedQuery:
        """Mock query processing."""
        self.call_history.append({"query": query, "context": context})

        if query in self.mock_responses:
            return self.mock_responses[query]

        # Default mock response
        return ProcessedQuery(
            original=query,
            processed=query.lower(),
            query_type="general",
            keywords=query.split(),
            expanded_terms=[f"expanded_{word}" for word in query.split()[:3]],
            metadata={"mock": True},
        )


class MockCategorizer:
    """Mock categorizer for testing."""

    def __init__(self, mock_responses: dict[str, CategoryMatch] | None = None):
        """Initialize with optional mock responses."""
        self.mock_responses = mock_responses or {}
        self.call_history: list[dict[str, Any]] = []

    def set_mock_response(self, query: str, response: CategoryMatch) -> None:
        """Set mock response for specific query."""
        self.mock_responses[query] = response

    def categorize_query(self, query: str, context: dict[str, Any] | None = None) -> CategoryMatch:
        """Mock query categorization."""
        self.call_history.append({"query": query, "context": context})

        if query in self.mock_responses:
            return self.mock_responses[query]

        # Default mock response
        if "api" in query.lower() or "kod" in query.lower():
            category = CategoryType.TECHNICAL
            strategy = "dense"
        elif "kultura" in query.lower() or "culture" in query.lower():
            category = CategoryType.CULTURAL
            strategy = "cultural_context"
        else:
            category = CategoryType.GENERAL
            strategy = "hybrid"

        return CategoryMatch(
            category=category,
            confidence=0.8,
            matched_patterns=[],
            cultural_indicators=[],
            complexity=QueryComplexity.MODERATE,
            retrieval_strategy=strategy,
        )


class MockSearchEngine:
    """Mock search engine for testing."""

    def __init__(self, mock_results: list[SearchResult] | None = None):
        """Initialize with optional mock results."""
        self.mock_results = mock_results or []
        self.call_history: list[dict[str, Any]] = []
        self.delay_seconds = 0.0  # Simulate search delay

    def set_mock_results(self, results: list[SearchResult]) -> None:
        """Set mock search results."""
        self.mock_results = results

    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for performance testing."""
        self.delay_seconds = seconds

    async def search(self, query_text: str, k: int = 5, similarity_threshold: float = 0.3) -> list[SearchResult]:
        """Mock search operation."""
        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        self.call_history.append({"query_text": query_text, "k": k, "similarity_threshold": similarity_threshold})

        # Filter mock results by threshold and limit
        filtered_results = [result for result in self.mock_results if result.similarity_score >= similarity_threshold]

        return filtered_results[:k]

    def create_mock_results(self, count: int = 5, base_similarity: float = 0.8) -> None:
        """Create mock search results for testing."""
        results = []
        for i in range(count):
            similarity = max(0.1, base_similarity - (i * 0.1))
            results.append(
                SearchResult(
                    content=f"Mock document {i + 1} content with relevant information",
                    metadata={"source": f"doc_{i + 1}", "mock": True},
                    similarity_score=similarity,
                    final_score=similarity,
                    boosts={},
                )
            )

        self.mock_results = results


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self, rerank_enabled: bool = True):
        """Initialize mock reranker."""
        self.rerank_enabled = rerank_enabled
        self.call_history: list[dict[str, Any]] = []
        self.delay_seconds = 0.0

    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for performance testing."""
        self.delay_seconds = seconds

    async def rerank(
        self, query: str, documents: list[dict[str, Any]], category: str | None = None
    ) -> list[dict[str, Any]]:
        """Mock reranking operation."""
        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        self.call_history.append({"query": query, "document_count": len(documents), "category": category})

        if not self.rerank_enabled:
            return documents

        # Simple mock reranking - reverse order to simulate change
        reranked = documents.copy()
        reranked.reverse()

        # Update final scores to show reranking effect
        for i, doc in enumerate(reranked):
            doc["final_score"] = max(0.1, doc.get("final_score", 0.5) + (0.1 * (len(reranked) - i)))
            doc["reranked"] = True

        return reranked


class MockLoggerProvider:
    """Mock logger provider that captures messages for testing."""

    def __init__(self):
        """Initialize message capture."""
        self.messages: dict[str, list[str]] = {"info": [], "debug": [], "error": []}

    def info(self, message: str) -> None:
        """Capture info message."""
        self.messages["info"].append(message)

    def debug(self, message: str) -> None:
        """Capture debug message."""
        self.messages["debug"].append(message)

    def error(self, message: str) -> None:
        """Capture error message."""
        self.messages["error"].append(message)

    def clear_messages(self) -> None:
        """Clear all captured messages."""
        for level in self.messages:
            self.messages[level].clear()

    def get_messages(self, level: str | None = None) -> dict[str, list] | list:
        """Get captured messages by level or all messages."""
        if level:
            return self.messages.get(level, [])
        return self.messages


class QueryProcessor:
    """Query processor wrapper."""

    def __init__(self, language: str = "hr"):
        """Initialize with query processor."""
        # Import at runtime to avoid circular dependencies
        from .query_processor import MultilingualQueryProcessor, create_query_processor

        try:
            # Try to use the new factory function if available
            # Import config loader to get proper main config and language config
            from ..utils.config_loader import load_config
            from ..utils.config_protocol import get_config_provider

            main_config = load_config("config")
            config_provider = get_config_provider()

            # Create the processor with proper config provider for filter configuration
            self._processor = create_query_processor(
                main_config=main_config, language=language, config_provider=config_provider
            )
        except (ImportError, TypeError, Exception):
            # Fallback to direct instantiation - may need config
            try:
                from .query_processor_providers import create_default_config

                config = create_default_config(language=language)

                # Try to get filter config manually for fallback
                filter_config = {}
                try:
                    from ..utils.config_loader import load_config

                    language_config = load_config(language)
                    if "topic_filters" in language_config:
                        filter_config = language_config["topic_filters"]
                    elif "query_filters" in language_config and "filters" in language_config["query_filters"]:
                        filter_config = language_config["query_filters"]["filters"]
                except Exception:
                    # If we can't load filter config, create a minimal one
                    filter_config = {"topic_filters": {}}

                self._processor = MultilingualQueryProcessor(config=config, filter_config=filter_config)
            except (ImportError, TypeError, Exception):
                # Final fallback to basic processor with minimal config
                from ..utils.config_models import QueryProcessingConfig

                minimal_config = QueryProcessingConfig(
                    language=language,
                    expand_synonyms=False,
                    normalize_case=True,
                    remove_stopwords=False,
                    min_query_length=1,
                    max_query_length=1000,
                    max_expanded_terms=5,
                    enable_morphological_analysis=False,
                    use_query_classification=False,
                    enable_spell_check=False,
                )
                # Create minimal filter config to avoid the missing topic_patterns error
                minimal_filter_config = {"topic_filters": {}}
                self._processor = MultilingualQueryProcessor(config=minimal_config, filter_config=minimal_filter_config)

    def process_query(self, query: str, context: dict[str, Any] | None = None) -> ProcessedQuery:
        """Process query using production processor."""
        # Handle case where processor couldn't be initialized
        if self._processor is None:
            raise Exception("Query processor not available")

        result = self._processor.process_query(query, context or {})

        # Convert to our ProcessedQuery format
        return ProcessedQuery(
            original=query,
            processed=getattr(result, "processed", query),
            query_type=getattr(result, "query_type", "general"),
            keywords=getattr(result, "keywords", query.split()),
            expanded_terms=getattr(result, "expanded_terms", []),
            metadata=getattr(result, "metadata", {}),
        )


class Categorizer:
    """Categorizer wrapper."""

    def __init__(self, language: str = "hr"):
        """Initialize with categorizer."""
        # Import at runtime to avoid circular dependencies
        from .categorization import QueryCategorizer
        from .categorization_providers import create_config_provider

        config_provider = create_config_provider()
        self._categorizer = QueryCategorizer(language, config_provider)

    def categorize_query(self, query: str, context: dict[str, Any] | None = None) -> CategoryMatch:
        """Categorize query using categorizer."""
        return self._categorizer.categorize_query(query)


class SearchEngineAdapter:
    """Adapter for search engine."""

    def __init__(self, search_engine):
        """Initialize with search engine."""
        self._search_engine = search_engine

    async def search(self, query_text: str, k: int = 5, similarity_threshold: float = 0.3) -> list[SearchResult]:
        """Adapt search engine to our interface."""
        # Use search_by_text method from ChromaDBSearchProvider
        raw_results = await self._search_engine.search_by_text(
            query_text=query_text, top_k=k, filters=None, include_metadata=True
        )

        # Convert ChromaDB results format to list of results
        results = []
        if raw_results and "documents" in raw_results and raw_results["documents"]:
            documents = raw_results["documents"][0] if raw_results["documents"] else []
            metadatas = raw_results.get("metadatas", [[]])[0] if raw_results.get("metadatas") else []
            distances = raw_results.get("distances", [[]])[0] if raw_results.get("distances") else []

            for i, doc in enumerate(documents):
                # Convert distance to similarity score (assuming cosine distance)
                distance = distances[i] if i < len(distances) else 1.0
                similarity = max(0.0, 1.0 - distance)  # Convert distance to similarity

                # Skip results below threshold
                if similarity < similarity_threshold:
                    continue

                metadata = metadatas[i] if i < len(metadatas) else {}

                results.append({"content": doc, "metadata": metadata, "similarity_score": similarity})

        # Convert to our SearchResult format
        adapted_results = []
        for result in results:
            # Results are already in dict format from our conversion above
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            similarity = result.get("similarity_score", 0.5)

            adapted_results.append(
                SearchResult(
                    content=content, metadata=metadata, similarity_score=similarity, final_score=similarity, boosts={}
                )
            )

        return adapted_results


class RerankerAdapter:
    """Adapter for reranker."""

    def __init__(self, reranker, language: str = "hr"):
        """Initialize with reranker."""
        self._reranker = reranker
        self.language = language

    async def rerank(
        self, query: str, documents: list[dict[str, Any]], category: str | None = None
    ) -> list[dict[str, Any]]:
        """Adapt reranker to our interface."""
        return await self._reranker.rerank(query=query, documents=documents, category=category)


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_mock_setup(
    query_responses: dict[str, ProcessedQuery] | None = None,
    category_responses: dict[str, CategoryMatch] | None = None,
    search_results: list[SearchResult] | None = None,
    enable_reranking: bool = True,
    search_delay: float = 0.0,
) -> tuple:
    """
    Create complete mock setup for testing.

    Args:
        query_responses: Mock query processor responses
        category_responses: Mock categorizer responses
        search_results: Mock search results
        enable_reranking: Whether to enable mock reranking
        search_delay: Artificial delay for search operations

    Returns:
        Tuple of (query_processor, categorizer, search_engine, reranker, logger, config)
    """
    # Create mock components
    query_processor = MockQueryProcessor(query_responses)
    categorizer = MockCategorizer(category_responses)
    search_engine = MockSearchEngine(search_results)
    reranker = MockReranker(enable_reranking)
    logger = MockLoggerProvider()

    # Set delays if specified
    search_engine.set_delay(search_delay)
    reranker.set_delay(search_delay)

    # Create default search results if none provided
    if search_results is None:
        search_engine.create_mock_results(count=5)

    # Create test configuration
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

    return query_processor, categorizer, search_engine, reranker, logger, config


def create_hierarchical_retriever(
    search_engine, language: str = "hr", reranker=None, enable_performance_tracking: bool = True
):
    """
    Create hierarchical retriever with real components.

    Args:
        search_engine: Search engine instance
        language: Language for processing components
        reranker: Optional reranker
        enable_performance_tracking: Whether to track performance

    Returns:
        HierarchicalRetriever instance
    """
    # Import the actual HierarchicalRetriever class
    from .hierarchical_retriever import HierarchicalRetriever

    # Create components
    query_processor = QueryProcessor(language)
    categorizer = Categorizer(language)
    search_engine_adapter = SearchEngineAdapter(search_engine)
    reranker_adapter = RerankerAdapter(reranker, language) if reranker else None

    # Use Python's standard logger
    import logging

    class StandardLoggerProvider:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def info(self, message: str) -> None:
            self.logger.info(message)

        def debug(self, message: str) -> None:
            self.logger.debug(message)

        def error(self, message: str) -> None:
            self.logger.error(message)

    logger = StandardLoggerProvider()

    # Create production configuration
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
        performance_tracking=enable_performance_tracking,
    )

    # Create and return the actual HierarchicalRetriever instance
    return HierarchicalRetriever(
        query_processor=query_processor,
        categorizer=categorizer,
        search_engine=search_engine_adapter,
        config=config,
        reranker=reranker_adapter,
        logger_provider=logger,
    )


def create_test_config(max_results: int = 5, performance_tracking: bool = True) -> RetrievalConfig:
    """Create test configuration with customizable parameters."""
    return RetrievalConfig(
        default_max_results=max_results,
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
        performance_tracking=performance_tracking,
    )
