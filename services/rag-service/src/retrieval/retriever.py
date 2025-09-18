"""
Document retrieval system for multilingual RAG applications.
Clean architecture with pure functions and dependency injection.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_decision_point,
    log_error_context,
    log_performance_metric,
)


# Pure Data Structures
class RetrievalStrategy(Enum):
    """Available retrieval strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    MULTI_PASS = "multi_pass"


class QueryType(Enum):
    """Detected query types for adaptive retrieval."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"


@dataclass
class RetrievalQuery:
    """Structured retrieval query with processing metadata."""

    original_text: str
    processed_text: str
    query_type: QueryType
    language: str
    keywords: list[str]
    filters: dict[str, Any] | None = None
    max_results: int = 10
    similarity_threshold: float = 0.1
    strategy_override: RetrievalStrategy | None = None

    def __post_init__(self):
        """Validate query parameters."""
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")


@dataclass
class RetrievedDocument:
    """Individual retrieved document with relevance metadata."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any]
    retrieval_method: str
    rank: int | None = None
    query_match_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "retrieval_method": self.retrieval_method,
            "rank": self.rank,
            "query_match_info": self.query_match_info,
        }


@dataclass
class RetrievalResult:
    """Complete retrieval result with timing and strategy metadata."""

    query: str
    documents: list[RetrievedDocument]
    total_found: int
    retrieval_time: float
    strategy_used: RetrievalStrategy
    query_type: QueryType
    language: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Add ranking to documents if not present."""
        for i, doc in enumerate(self.documents):
            if doc.rank is None:
                doc.rank = i + 1


# Protocol-based Dependencies (testable interfaces)
class QueryProcessor(Protocol):
    """Protocol for query processing."""

    async def process_query(self, query: str, language: str) -> RetrievalQuery:
        """Process raw query into structured format."""
        ...


class SearchEngine(Protocol):
    """Protocol for search operations."""

    async def search(
        self,
        query: str,
        top_k: int,
        method: str = "semantic",
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Execute search and return results."""
        ...


class ResultRanker(Protocol):
    """Protocol for result ranking and reordering."""

    def rank_results(self, query: RetrievalQuery, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Rank documents by relevance."""
        ...


class RetrievalConfig(Protocol):
    """Protocol for retrieval configuration."""

    def get_strategy_config(self, strategy: RetrievalStrategy) -> dict[str, Any]:
        """Get configuration for specific strategy."""
        ...

    def get_adaptive_config(self, query_type: QueryType) -> dict[str, Any]:
        """Get adaptive configuration for query type."""
        ...


# Pure Functions (testable utilities)
def select_retrieval_strategy(
    query: RetrievalQuery, default_strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC
) -> RetrievalStrategy:
    """
    Select optimal retrieval strategy based on query characteristics.

    Args:
        query: Processed query with metadata
        default_strategy: Fallback strategy

    Returns:
        Selected retrieval strategy
    """
    logger = get_system_logger()
    logger.debug(
        "retrieval_strategy",
        "select_retrieval_strategy",
        f"Selecting strategy for query type: {query.query_type.value}",
        metadata={
            "query_type": query.query_type.value,
            "keywords_count": len(query.keywords),
            "default": default_strategy.value,
        },
    )

    # Use explicit override if provided
    if query.strategy_override:
        log_decision_point(
            "retrieval_strategy",
            "select_retrieval_strategy",
            "explicit override provided",
            f"using {query.strategy_override.value}",
        )
        return query.strategy_override

    # Strategy selection based on query type
    strategy_map = {
        QueryType.FACTUAL: RetrievalStrategy.KEYWORD,
        QueryType.CONCEPTUAL: RetrievalStrategy.SEMANTIC,
        QueryType.PROCEDURAL: RetrievalStrategy.HYBRID,
        QueryType.COMPARATIVE: RetrievalStrategy.HYBRID,
        QueryType.EXPLORATORY: RetrievalStrategy.SEMANTIC,
    }

    if query.query_type not in strategy_map:
        log_decision_point(
            "retrieval_strategy",
            "select_retrieval_strategy",
            f"unknown query type {query.query_type.value}",
            f"using default {default_strategy.value}",
        )
        selected = default_strategy
    else:
        mapped_strategy = strategy_map[query.query_type]
        log_decision_point(
            "retrieval_strategy",
            "select_retrieval_strategy",
            f"query type {query.query_type.value}",
            f"mapped to {mapped_strategy.value}",
        )
        selected = mapped_strategy

    # Adjust based on query characteristics
    if len(query.keywords) >= 3 and selected == RetrievalStrategy.SEMANTIC:
        log_decision_point(
            "retrieval_strategy",
            "select_retrieval_strategy",
            f"many keywords ({len(query.keywords)} >= 3)",
            "switching semantic to hybrid",
        )
        selected = RetrievalStrategy.HYBRID

    logger.debug("retrieval_strategy", "select_retrieval_strategy", f"Final strategy selected: {selected.value}")
    return selected


def calculate_adaptive_top_k(query: RetrievalQuery, base_top_k: int = 10) -> int:
    """
    Calculate adaptive top_k based on query characteristics.

    Args:
        query: Processed query
        base_top_k: Base number of results

    Returns:
        Adjusted top_k value
    """
    logger = get_system_logger()
    logger.debug(
        "retrieval_adapter",
        "calculate_adaptive_top_k",
        f"Calculating adaptive top_k for {query.query_type.value}",
        metadata={"base_top_k": base_top_k, "keywords_count": len(query.keywords)},
    )

    # Base adjustment factors
    multipliers = {
        QueryType.FACTUAL: 0.8,  # Precise queries need fewer results
        QueryType.CONCEPTUAL: 1.2,  # Broad queries benefit from more results
        QueryType.PROCEDURAL: 1.0,  # Standard amount
        QueryType.COMPARATIVE: 1.3,  # Need examples for comparison
        QueryType.EXPLORATORY: 1.5,  # Exploration needs variety
    }

    if query.query_type not in multipliers:
        multiplier = 1.0
        logger.trace(
            "retrieval_adapter",
            "calculate_adaptive_top_k",
            f"Unknown query type {query.query_type.value}, using default multiplier 1.0",
        )
    else:
        multiplier = multipliers[query.query_type]
        logger.trace(
            "retrieval_adapter",
            "calculate_adaptive_top_k",
            f"Query type {query.query_type.value} multiplier: {multiplier}",
        )

    # Adjust for query complexity
    if len(query.keywords) > 5:
        multiplier *= 1.2
        logger.debug(
            "retrieval_adapter",
            "calculate_adaptive_top_k",
            f"Complex query ({len(query.keywords)} keywords), increasing multiplier by 1.2x",
        )
    elif len(query.keywords) <= 2:
        multiplier *= 0.9
        logger.debug(
            "retrieval_adapter",
            "calculate_adaptive_top_k",
            f"Simple query ({len(query.keywords)} keywords), decreasing multiplier by 0.9x",
        )

    # Apply bounds
    adjusted = int(base_top_k * multiplier)
    final_top_k = max(5, min(50, adjusted))

    log_decision_point(
        "retrieval_adapter",
        "calculate_adaptive_top_k",
        f"multiplier {multiplier} applied to base {base_top_k}",
        f"final top_k {final_top_k} (clamped to 5-50 range)",
    )
    return final_top_k


def merge_retrieval_results(
    semantic_docs: list[RetrievedDocument],
    keyword_docs: list[RetrievedDocument],
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[RetrievedDocument]:
    """
    Merge results from different retrieval methods.

    Args:
        semantic_docs: Results from semantic search
        keyword_docs: Results from keyword search
        semantic_weight: Weight for semantic scores
        keyword_weight: Weight for keyword scores

    Returns:
        Merged and deduplicated results
    """
    if not semantic_docs and not keyword_docs:
        return []

    if not semantic_docs:
        return keyword_docs

    if not keyword_docs:
        return semantic_docs

    # Normalize weights
    total_weight = semantic_weight + keyword_weight
    if total_weight == 0:
        return semantic_docs  # Fallback

    norm_semantic = semantic_weight / total_weight
    norm_keyword = keyword_weight / total_weight

    # Index documents by ID for merging
    merged = {}

    # Add semantic results
    for doc in semantic_docs:
        merged[doc.id] = RetrievedDocument(
            id=doc.id,
            content=doc.content,
            score=doc.score * norm_semantic,
            metadata=doc.metadata,
            retrieval_method="hybrid",
            query_match_info={"semantic_score": doc.score, "keyword_score": 0.0},
        )

    # Add/merge keyword results
    for doc in keyword_docs:
        if doc.id in merged:
            # Combine scores for documents found by both methods
            existing = merged[doc.id]
            existing.score += doc.score * norm_keyword
            if existing.query_match_info is not None:
                existing.query_match_info["keyword_score"] = doc.score
        else:
            # New document from keyword search only
            merged[doc.id] = RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=doc.score * norm_keyword,
                metadata=doc.metadata,
                retrieval_method="hybrid",
                query_match_info={"semantic_score": 0.0, "keyword_score": doc.score},
            )

    # Convert to list and sort by combined score
    result = list(merged.values())
    result.sort(key=lambda x: x.score, reverse=True)

    return result


def filter_results_by_threshold(documents: list[RetrievedDocument], threshold: float) -> list[RetrievedDocument]:
    """
    Filter results by similarity threshold.

    Args:
        documents: Retrieved documents
        threshold: Minimum score threshold

    Returns:
        Filtered documents
    """
    return [doc for doc in documents if doc.score >= threshold]


def limit_results(documents: list[RetrievedDocument], max_results: int) -> list[RetrievedDocument]:
    """
    Limit number of results.

    Args:
        documents: Retrieved documents
        max_results: Maximum number to return

    Returns:
        Limited documents
    """
    return documents[:max_results] if max_results > 0 else documents


def add_query_match_analysis(query: RetrievalQuery, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
    """
    Add query match analysis to document metadata.

    Args:
        query: Original retrieval query
        documents: Retrieved documents

    Returns:
        Documents with enhanced match information
    """
    query_terms = set(query.keywords)

    for doc in documents:
        if not doc.query_match_info:
            doc.query_match_info = {}

        # Analyze keyword matches
        doc_words = set(doc.content.lower().split())
        keyword_matches = query_terms.intersection(doc_words)

        doc.query_match_info.update(
            {
                "keyword_matches": list(keyword_matches),
                "keyword_match_ratio": (len(keyword_matches) / len(query_terms) if query_terms else 0),
                "content_length": len(doc.content),
                "query_type": query.query_type.value,
                "language": query.language,
            }
        )

    return documents


def calculate_diversity_score(documents: list[RetrievedDocument]) -> float:
    """
    Calculate diversity score for result set.

    Args:
        documents: Retrieved documents

    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    if len(documents) <= 1:
        return 1.0

    # Simple diversity based on content similarity
    # In production, this could use actual embedding similarity
    diversity_factors = []

    for i, doc1 in enumerate(documents):
        for doc2 in documents[i + 1 :]:
            # Simple content-based diversity (word overlap)
            words1 = set(doc1.content.lower().split())
            words2 = set(doc2.content.lower().split())

            if words1 and words2:
                overlap = len(words1.intersection(words2))
                total = len(words1.union(words2))
                similarity = overlap / total if total > 0 else 0
                diversity_factors.append(1 - similarity)

    return sum(diversity_factors) / len(diversity_factors) if diversity_factors else 0.5


# Main Retrieval Engine Class
class DocumentRetriever:
    """
    Document retrieval engine using dependency injection for testability.
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        search_engine: SearchEngine,
        result_ranker: ResultRanker,
        config: RetrievalConfig,
    ):
        """
        Initialize retriever with injected dependencies.

        Args:
            query_processor: Query processing service
            search_engine: Document search service
            result_ranker: Result ranking service
            config: Configuration service
        """
        logger = get_system_logger()
        log_component_start("document_retriever", "init")

        self.query_processor = query_processor
        self.search_engine = search_engine
        self.result_ranker = result_ranker
        self.config = config

        logger.debug("document_retriever", "init", "DocumentRetriever initialized with dependency injection")
        log_component_end("document_retriever", "init", "DocumentRetriever ready for retrieval operations")

    async def retrieve_documents(
        self,
        query: str,
        language: str = "en",
        max_results: int | None = None,
        strategy: RetrievalStrategy | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for query.

        Args:
            query: Natural language query
            language: Query language
            max_results: Maximum results to return
            strategy: Retrieval strategy override
            filters: Optional metadata filters

        Returns:
            Retrieval result with documents and metadata

        Raises:
            ValueError: For invalid parameters
        """
        logger = get_system_logger()
        log_component_start(
            "document_retriever",
            "retrieve_documents",
            query_length=len(query),
            language=language,
            max_results=max_results,
            strategy_override=strategy.value if strategy else None,
        )

        if not query or not query.strip():
            logger.error("document_retriever", "retrieve_documents", "Empty query provided")
            raise ValueError("Query cannot be empty")

        start_time = time.time()

        try:
            # Process query
            logger.debug("document_retriever", "retrieve_documents", "Processing query")
            processed_query = await self.query_processor.process_query(query, language)

            # Apply overrides
            if max_results is not None:
                logger.trace(
                    "document_retriever", "retrieve_documents", f"Applying max_results override: {max_results}"
                )
                processed_query.max_results = max_results
            if strategy is not None:
                logger.trace(
                    "document_retriever", "retrieve_documents", f"Applying strategy override: {strategy.value}"
                )
                processed_query.strategy_override = strategy
            if filters is not None:
                logger.trace("document_retriever", "retrieve_documents", f"Applying filters: {filters}")
                processed_query.filters = filters

            # Select retrieval strategy
            logger.debug("document_retriever", "retrieve_documents", "Selecting retrieval strategy")
            selected_strategy = select_retrieval_strategy(processed_query)

            # Execute retrieval based on strategy
            logger.info(
                "document_retriever", "retrieve_documents", f"Executing {selected_strategy.value} retrieval strategy"
            )
            documents = await self._execute_retrieval_strategy(processed_query, selected_strategy)

            # Apply post-processing
            logger.debug(
                "document_retriever", "retrieve_documents", f"Post-processing {len(documents)} retrieved documents"
            )
            documents = self._post_process_results(processed_query, documents)

            retrieval_time = time.time() - start_time

            diversity_score = calculate_diversity_score(documents)
            log_performance_metric("document_retriever", "retrieve_documents", "retrieval_time", retrieval_time)
            log_performance_metric("document_retriever", "retrieve_documents", "documents_retrieved", len(documents))
            log_performance_metric("document_retriever", "retrieve_documents", "diversity_score", diversity_score)

            result = RetrievalResult(
                query=query,
                documents=documents,
                total_found=len(documents),
                retrieval_time=retrieval_time,
                strategy_used=selected_strategy,
                query_type=processed_query.query_type,
                language=processed_query.language,
                metadata={
                    "processed_query": processed_query.processed_text,
                    "keywords": processed_query.keywords,
                    "filters": processed_query.filters,
                    "diversity_score": diversity_score,
                },
            )

            log_component_end(
                "document_retriever",
                "retrieve_documents",
                f"Retrieved {len(documents)} documents in {retrieval_time:.3f}s using {selected_strategy.value}",
                duration=retrieval_time,
            )
            return result

        except Exception as e:
            log_error_context(
                "document_retriever",
                "retrieve_documents",
                e,
                {
                    "query": query,
                    "language": language,
                    "max_results": max_results,
                    "strategy_override": strategy.value if strategy else None,
                },
            )
            raise

    async def _execute_retrieval_strategy(
        self, query: RetrievalQuery, strategy: RetrievalStrategy
    ) -> list[RetrievedDocument]:
        """Execute retrieval using specified strategy."""
        if strategy == RetrievalStrategy.SEMANTIC:
            return await self._semantic_retrieval(query)
        elif strategy == RetrievalStrategy.KEYWORD:
            return await self._keyword_retrieval(query)
        elif strategy == RetrievalStrategy.HYBRID:
            return await self._hybrid_retrieval(query)
        elif strategy == RetrievalStrategy.ADAPTIVE:
            return await self._adaptive_retrieval(query)
        elif strategy == RetrievalStrategy.MULTI_PASS:
            return await self._multi_pass_retrieval(query)
        else:
            self.logger.warning(f"Unknown strategy {strategy}, falling back to semantic")
            return await self._semantic_retrieval(query)

    async def _semantic_retrieval(self, query: RetrievalQuery) -> list[RetrievedDocument]:
        """Execute semantic similarity retrieval."""
        top_k = calculate_adaptive_top_k(query, query.max_results)

        results = await self.search_engine.search(
            query=query.processed_text,
            top_k=top_k,
            method="semantic",
            filters=query.filters,
            similarity_threshold=query.similarity_threshold,
        )

        documents = []
        for result in results:
            # Validate required fields - fail fast if missing
            if "id" not in result:
                raise ValueError("Search result missing required 'id' field")
            if "content" not in result:
                raise ValueError("Search result missing required 'content' field")
            if "score" not in result:
                raise ValueError("Search result missing required 'score' field")
            if "metadata" not in result:
                raise ValueError("Search result missing required 'metadata' field")

            documents.append(
                RetrievedDocument(
                    id=result["id"],
                    content=result["content"],
                    score=result["score"],
                    metadata=result["metadata"],
                    retrieval_method="semantic",
                )
            )

        return documents

    async def _keyword_retrieval(self, query: RetrievalQuery) -> list[RetrievedDocument]:
        """Execute keyword-based retrieval."""
        top_k = calculate_adaptive_top_k(query, query.max_results)

        results = await self.search_engine.search(
            query=query.processed_text,
            top_k=top_k,
            method="keyword",
            filters=query.filters,
            similarity_threshold=query.similarity_threshold,
        )

        documents = []
        for result in results:
            # Validate required fields - fail fast if missing
            if "id" not in result:
                raise ValueError("Keyword search result missing required 'id' field")
            if "content" not in result:
                raise ValueError("Keyword search result missing required 'content' field")
            if "score" not in result:
                raise ValueError("Keyword search result missing required 'score' field")
            if "metadata" not in result:
                raise ValueError("Keyword search result missing required 'metadata' field")

            documents.append(
                RetrievedDocument(
                    id=result["id"],
                    content=result["content"],
                    score=result["score"],
                    metadata=result["metadata"],
                    retrieval_method="keyword",
                )
            )

        return documents

    async def _hybrid_retrieval(self, query: RetrievalQuery) -> list[RetrievedDocument]:
        """Execute hybrid retrieval combining semantic and keyword approaches."""
        calculate_adaptive_top_k(query, query.max_results)

        # Execute both approaches concurrently
        semantic_task = self._semantic_retrieval(query)
        keyword_task = self._keyword_retrieval(query)

        semantic_docs, keyword_docs = await asyncio.gather(semantic_task, keyword_task, return_exceptions=True)

        # Handle exceptions
        if isinstance(semantic_docs, Exception):
            self.logger.warning(f"Semantic retrieval failed: {semantic_docs}")
            semantic_docs = []

        if isinstance(keyword_docs, Exception):
            self.logger.warning(f"Keyword retrieval failed: {keyword_docs}")
            keyword_docs = []

        # Get weights from config - validate required keys
        strategy_config = self.config.get_strategy_config(RetrievalStrategy.HYBRID)
        if "semantic_weight" not in strategy_config:
            raise ValueError("Missing 'semantic_weight' in hybrid strategy configuration")
        if "keyword_weight" not in strategy_config:
            raise ValueError("Missing 'keyword_weight' in hybrid strategy configuration")
        semantic_weight = strategy_config["semantic_weight"]
        keyword_weight = strategy_config["keyword_weight"]

        # Merge results (at this point, both are guaranteed to be lists after exception handling)
        assert isinstance(semantic_docs, list), "semantic_docs should be list after exception handling"
        assert isinstance(keyword_docs, list), "keyword_docs should be list after exception handling"
        merged_docs = merge_retrieval_results(semantic_docs, keyword_docs, semantic_weight, keyword_weight)

        return merged_docs

    async def _adaptive_retrieval(self, query: RetrievalQuery) -> list[RetrievedDocument]:
        """Execute adaptive retrieval based on query characteristics."""
        # Get adaptive configuration
        self.config.get_adaptive_config(query.query_type)

        # Select sub-strategy based on query type
        if query.query_type in [QueryType.FACTUAL]:
            return await self._keyword_retrieval(query)
        elif query.query_type in [QueryType.CONCEPTUAL, QueryType.EXPLORATORY]:
            return await self._semantic_retrieval(query)
        else:
            return await self._hybrid_retrieval(query)

    async def _multi_pass_retrieval(self, query: RetrievalQuery) -> list[RetrievedDocument]:
        """Execute multi-pass retrieval with progressive refinement."""
        # First pass: broad semantic search
        broad_query = RetrievalQuery(
            original_text=query.original_text,
            processed_text=query.processed_text,
            query_type=query.query_type,
            language=query.language,
            keywords=query.keywords,
            filters=query.filters,
            max_results=query.max_results * 2,  # Get more for refinement
            similarity_threshold=0.0,  # Lower threshold for first pass
        )

        first_pass = await self._semantic_retrieval(broad_query)

        # If we have enough high-quality results, return them
        high_quality = [doc for doc in first_pass if doc.score >= query.similarity_threshold]
        if len(high_quality) >= query.max_results:
            return high_quality[: query.max_results]

        # Second pass: keyword search for additional results
        second_pass = await self._keyword_retrieval(query)

        # Merge and deduplicate
        all_docs = merge_retrieval_results(first_pass, second_pass)

        # Filter and limit
        filtered = filter_results_by_threshold(all_docs, query.similarity_threshold)
        return limit_results(filtered, query.max_results)

    def _post_process_results(
        self, query: RetrievalQuery, documents: list[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """Apply post-processing to retrieval results."""
        if not documents:
            return documents

        # Add query match analysis
        documents = add_query_match_analysis(query, documents)

        # Apply result ranking
        documents = self.result_ranker.rank_results(query, documents)

        # Apply final filtering
        documents = filter_results_by_threshold(documents, query.similarity_threshold)
        documents = limit_results(documents, query.max_results)

        return documents


# Factory Functions
def create_retrieval_query(
    original_text: str, processed_text: str, query_type: QueryType, language: str, keywords: list[str], **kwargs
) -> RetrievalQuery:
    """Factory function to create RetrievalQuery with validation."""
    return RetrievalQuery(
        original_text=original_text,
        processed_text=processed_text,
        query_type=query_type,
        language=language,
        keywords=keywords,
        **kwargs,
    )


def create_document_retriever(
    query_processor: QueryProcessor, search_engine: SearchEngine, result_ranker: ResultRanker, config: RetrievalConfig
) -> DocumentRetriever:
    """Factory function to create DocumentRetriever."""
    return DocumentRetriever(
        query_processor=query_processor, search_engine=search_engine, result_ranker=result_ranker, config=config
    )
