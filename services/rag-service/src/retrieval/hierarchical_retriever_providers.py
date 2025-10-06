"""
Provider implementations for hierarchical retriever dependency injection.
Production and mock providers for testable hierarchical retrieval system.
"""

from typing import Any

from ..utils.config_loader import get_config_section
from ..utils.logging_factory import get_system_logger
from .categorization import CategoryMatch
from .hierarchical_retriever import ProcessedQuery, SearchResult


def get_similarity_thresholds() -> dict[str, float]:
    """Get similarity thresholds from configuration."""
    try:
        retrieval_config = get_config_section("config", "retrieval")
        if "similarity_thresholds" not in retrieval_config:
            raise KeyError("Missing 'similarity_thresholds' section in retrieval configuration")
        return retrieval_config["similarity_thresholds"]
    except Exception as e:
        from ..utils.logging_factory import get_system_logger

        logger = get_system_logger()
        logger.error(
            "config_error", "get_similarity_thresholds", f"Failed to load similarity thresholds from config: {str(e)}"
        )
        raise RuntimeError(f"Configuration error: Cannot load similarity thresholds - {str(e)}") from e


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
                minimal_filter_config: dict[str, Any] = {"topic_filters": {}}
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

    def categorize_query(self, query: str, scope_context: dict[str, Any] | None = None) -> CategoryMatch:
        """Categorize query using categorizer."""
        return self._categorizer.categorize_query(query, scope_context)


class SearchEngineAdapter:
    """Adapter for search engine."""

    def __init__(self, search_engine):
        """Initialize with search engine."""
        self._search_engine = search_engine

    async def search(self, query_text: str, k: int = 5, similarity_threshold: float = 0.3) -> list[SearchResult]:
        """Adapt search engine to our interface."""
        # AI DEBUGGING: Comprehensive trace logging for SearchEngineAdapter

        logger = get_system_logger()

        logger.trace(
            "search_engine_adapter",
            "search_start",
            f"ADAPTER_SEARCH | query_preview={query_text[:50]} | k={k} | "
            f"similarity_threshold={similarity_threshold} | "
            f"engine_type={type(self._search_engine).__name__}",
        )

        # Handle both ChromaDB and Weaviate interfaces - use class name for definitive identification
        if type(self._search_engine).__name__ == "WeaviateSearchProvider":
            # Weaviate provider - use search_by_text method
            logger.trace(
                "search_engine_adapter",
                "weaviate_path",
                f"WEAVIATE_CALL | using_search_by_text=true | passed_threshold={similarity_threshold}",
            )
            raw_results = await self._search_engine.search_by_text(
                query_text=query_text, top_k=k, filters=None, include_metadata=True
            )
        elif hasattr(self._search_engine, "search_by_text"):
            # ChromaDB interface
            logger.trace("search_engine_adapter", "chromadb_path", "CHROMADB_CALL | using_search_by_text=true")
            raw_results = await self._search_engine.search_by_text(
                query_text=query_text, top_k=k, filters=None, include_metadata=True
            )
        else:
            raise ValueError(
                f"Search engine {type(self._search_engine)} doesn't support search_by_text or search methods"
            )

        logger.trace(
            "search_engine_adapter",
            "raw_results_received",
            f"RAW_RESULTS | count={len(raw_results) if isinstance(raw_results, list) else 'dict_format'} | "
            f"type={type(raw_results).__name__}",
        )

        # Convert results from both ChromaDB and Weaviate formats
        results = []

        if type(self._search_engine).__name__ != "WeaviateSearchProvider":
            # ChromaDB format: {"documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
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
        else:
            # Weaviate format: list of dicts with content, metadata, and similarity_score
            logger.trace(
                "search_engine_adapter",
                "weaviate_processing",
                f"WEAVIATE_PROCESSING | raw_results_count={len(raw_results) if raw_results else 0}",
            )

            if raw_results:
                for i, result in enumerate(raw_results):
                    if isinstance(result, dict):
                        similarity = result.get("similarity_score", 0.5)
                        passes_filter = similarity >= similarity_threshold
                        filter_status = "PASS" if passes_filter else "FAIL"

                        logger.trace(
                            "search_engine_adapter",
                            "weaviate_filtering",
                            f"WEAVIATE_RESULT_{i:02d} | similarity={similarity:.6f} | "
                            f"threshold={similarity_threshold} | passes={passes_filter} | "
                            f"status={filter_status} | content_preview={str(result.get('content', ''))[:50]}",
                        )

                        if passes_filter:
                            results.append(result)
                    else:
                        # Fallback for unexpected result format
                        logger.warning(
                            "search_engine_adapter",
                            "unexpected_format",
                            f"UNEXPECTED_FORMAT | result_type={type(result).__name__} | fallback_similarity=0.5",
                        )
                        results.append({"content": str(result), "metadata": {}, "similarity_score": 0.5})

            logger.trace(
                "search_engine_adapter",
                "weaviate_filtering_summary",
                f"WEAVIATE_SUMMARY | input_count={len(raw_results) if raw_results else 0} | "
                f"passed_filter={len(results)} | threshold={similarity_threshold}",
            )

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


def create_hierarchical_retriever(search_engine, reranker=None, language: str = "hr"):
    """
    Create hierarchical retriever with all dependencies.

    Args:
        search_engine: Search engine instance (ChromaDB or Weaviate)
        reranker: Optional reranker instance
        language: Language code for processing

    Returns:
        HierarchicalRetriever instance with configured adapters
    """
    from .hierarchical_retriever import HierarchicalRetriever, RetrievalConfig

    # Create adapter instances
    query_processor = QueryProcessor(language=language)
    categorizer = Categorizer(language=language)
    search_adapter = SearchEngineAdapter(search_engine)
    reranker_adapter = RerankerAdapter(reranker, language=language) if reranker else None

    # Create default config
    config = RetrievalConfig(
        default_max_results=10,
        similarity_thresholds=get_similarity_thresholds(),
        boost_weights={
            "keyword": 0.2,
            "technical": 0.1,
            "exact_match": 0.2,
            "temporal": 0.15,
            "faq": 0.1,
            "comparative": 0.1,
        },
        strategy_mappings={},
        performance_tracking=False,
    )

    # Create and return retriever
    return HierarchicalRetriever(
        query_processor=query_processor,
        categorizer=categorizer,
        search_engine=search_adapter,
        config=config,
        reranker=reranker_adapter,
    )
