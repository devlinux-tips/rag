"""
Provider implementations for hierarchical retriever dependency injection.
Production and mock providers for 100% testable hierarchical retrieval system.
"""

import time
from typing import Any, Dict, List, Optional

from .categorization import CategoryMatch, CategoryType, QueryComplexity
from .hierarchical_retriever import (Categorizer, LoggerProvider,
                                     ProcessedQuery, QueryProcessor, Reranker,
                                     RetrievalConfig, SearchEngine,
                                     SearchResult)


class MockQueryProcessor:
    """Mock query processor for testing."""

    def __init__(self, mock_responses: Dict[str, ProcessedQuery] = None):
        """Initialize with optional mock responses."""
        self.mock_responses = mock_responses or {}
        self.call_history = []

    def set_mock_response(self, query: str, response: ProcessedQuery) -> None:
        """Set mock response for specific query."""
        self.mock_responses[query] = response

    def process_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ProcessedQuery:
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

    def __init__(self, mock_responses: Dict[str, CategoryMatch] = None):
        """Initialize with optional mock responses."""
        self.mock_responses = mock_responses or {}
        self.call_history = []

    def set_mock_response(self, query: str, response: CategoryMatch) -> None:
        """Set mock response for specific query."""
        self.mock_responses[query] = response

    def categorize_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CategoryMatch:
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

    def __init__(self, mock_results: List[SearchResult] = None):
        """Initialize with optional mock results."""
        self.mock_results = mock_results or []
        self.call_history = []
        self.delay_seconds = 0.0  # Simulate search delay

    def set_mock_results(self, results: List[SearchResult]) -> None:
        """Set mock search results."""
        self.mock_results = results

    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for performance testing."""
        self.delay_seconds = seconds

    async def search(
        self, query_text: str, k: int = 5, similarity_threshold: float = 0.3
    ) -> List[SearchResult]:
        """Mock search operation."""
        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        self.call_history.append(
            {
                "query_text": query_text,
                "k": k,
                "similarity_threshold": similarity_threshold,
            }
        )

        # Filter mock results by threshold and limit
        filtered_results = [
            result
            for result in self.mock_results
            if result.similarity_score >= similarity_threshold
        ]

        return filtered_results[:k]

    def create_mock_results(self, count: int = 5, base_similarity: float = 0.8) -> None:
        """Create mock search results for testing."""
        results = []
        for i in range(count):
            similarity = max(0.1, base_similarity - (i * 0.1))
            results.append(
                SearchResult(
                    content=f"Mock document {i+1} content with relevant information",
                    metadata={"source": f"doc_{i+1}", "mock": True},
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
        self.call_history = []
        self.delay_seconds = 0.0

    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for performance testing."""
        self.delay_seconds = seconds

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Mock reranking operation."""
        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        self.call_history.append(
            {"query": query, "document_count": len(documents), "category": category}
        )

        if not self.rerank_enabled:
            return documents

        # Simple mock reranking - reverse order to simulate change
        reranked = documents.copy()
        reranked.reverse()

        # Update final scores to show reranking effect
        for i, doc in enumerate(reranked):
            doc["final_score"] = max(
                0.1, doc.get("final_score", 0.5) + (0.1 * (len(reranked) - i))
            )
            doc["reranked"] = True

        return reranked


class MockLoggerProvider:
    """Mock logger provider that captures messages for testing."""

    def __init__(self):
        """Initialize message capture."""
        self.messages = {"info": [], "debug": [], "error": []}

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

    def get_messages(self, level: str = None) -> Dict[str, list] | list:
        """Get captured messages by level or all messages."""
        if level:
            return self.messages.get(level, [])
        return self.messages


class ProductionQueryProcessor:
    """Production query processor wrapper."""

    def __init__(self, language: str = "hr"):
        """Initialize with production query processor."""
        # Import at runtime to avoid circular dependencies
        from .query_processor import (MultilingualQueryProcessor,
                                      create_query_processor)

        try:
            # Try to use the new factory function if available
            self._processor = create_query_processor(language=language)
        except (ImportError, TypeError):
            # Fallback to direct instantiation - may need config
            try:
                from .query_processor import QueryProcessingConfig
                from .query_processor_providers import create_default_config

                config = create_default_config(language=language)
                self._processor = MultilingualQueryProcessor(config=config)
            except (ImportError, TypeError):
                # Final fallback to basic processor
                self._processor = None

    def process_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ProcessedQuery:
        """Process query using production processor."""
        try:
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
        except Exception as e:
            # Fallback to basic processing
            return ProcessedQuery(
                original=query,
                processed=query.lower(),
                query_type="general",
                keywords=query.split(),
                expanded_terms=[],
                metadata={"error": str(e)},
            )


class ProductionCategorizer:
    """Production categorizer wrapper."""

    def __init__(self, language: str = "hr"):
        """Initialize with production categorizer."""
        # Import at runtime to avoid circular dependencies
        from .categorization import QueryCategorizer
        from .categorization_providers import create_config_provider

        config_provider = create_config_provider()
        self._categorizer = QueryCategorizer(language, config_provider)

    def categorize_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CategoryMatch:
        """Categorize query using production categorizer."""
        return self._categorizer.categorize_query(query)


class ProductionSearchEngineAdapter:
    """Adapter for production search engine."""

    def __init__(self, search_engine):
        """Initialize with production search engine."""
        self._search_engine = search_engine

    async def search(
        self, query_text: str, k: int = 5, similarity_threshold: float = 0.3
    ) -> List[SearchResult]:
        """Adapt production search engine to our interface."""
        try:
            # Assuming production search engine has similar interface
            results = await self._search_engine.search(
                query_text=query_text, k=k, similarity_threshold=similarity_threshold
            )

            # Convert to our SearchResult format
            adapted_results = []
            for result in results:
                # Handle different result formats
                if hasattr(result, "content"):
                    content = result.content
                    metadata = getattr(result, "metadata", {})
                    similarity = getattr(result, "similarity_score", 0.5)
                elif isinstance(result, dict):
                    content = result.get("content", "")
                    metadata = result.get("metadata", {})
                    similarity = result.get("similarity_score", 0.5)
                else:
                    content = str(result)
                    metadata = {}
                    similarity = 0.5

                adapted_results.append(
                    SearchResult(
                        content=content,
                        metadata=metadata,
                        similarity_score=similarity,
                        final_score=similarity,
                        boosts={},
                    )
                )

            return adapted_results

        except Exception as e:
            # Return empty results on error
            return []


class ProductionRerankerAdapter:
    """Adapter for production reranker."""

    def __init__(self, reranker, language: str = "hr"):
        """Initialize with production reranker."""
        self._reranker = reranker
        self.language = language

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Adapt production reranker to our interface."""
        try:
            return await self._reranker.rerank(
                query=query, documents=documents, category=category
            )
        except Exception as e:
            # Return original documents on error
            return documents


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_mock_setup(
    query_responses: Dict[str, ProcessedQuery] = None,
    category_responses: Dict[str, CategoryMatch] = None,
    search_results: List[SearchResult] = None,
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


def create_production_setup(
    search_engine,
    language: str = "hr",
    reranker=None,
    enable_performance_tracking: bool = True,
) -> tuple:
    """
    Create production setup with real components.

    Args:
        search_engine: Production search engine instance
        language: Language for processing components
        reranker: Optional production reranker
        enable_performance_tracking: Whether to track performance

    Returns:
        Tuple of (query_processor, categorizer, search_engine, reranker, logger, config)
    """
    # Create production components
    query_processor = ProductionQueryProcessor(language)
    categorizer = ProductionCategorizer(language)
    search_engine_adapter = ProductionSearchEngineAdapter(search_engine)
    reranker_adapter = (
        ProductionRerankerAdapter(reranker, language) if reranker else None
    )

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

    return (
        query_processor,
        categorizer,
        search_engine_adapter,
        reranker_adapter,
        logger,
        config,
    )


def create_test_config(
    max_results: int = 5, performance_tracking: bool = True
) -> RetrievalConfig:
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
