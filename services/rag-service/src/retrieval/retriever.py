"""
Main retrieval logic for multilingual RAG system.
Orchestrates query processing, search, and result ranking.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..utils.config_loader import get_retrieval_config
from ..utils.error_handler import handle_config_error
from ..vectordb.embeddings import MultilingualEmbeddingModel
from ..vectordb.search import SearchMethod, SearchResponse, SemanticSearchEngine
from ..vectordb.storage import ChromaDBStorage
from .query_processor import MultilingualQueryProcessor, ProcessedQuery, QueryType


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""

    SIMPLE = "simple"  # Basic semantic search
    ADAPTIVE = "adaptive"  # Adapt based on query type
    MULTI_PASS = "multi_pass"  # Multiple search passes
    HYBRID = "hybrid"  # Combine multiple approaches


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval."""

    strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    max_results: int = 10
    min_similarity: float = 0.1
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    adaptive_top_k: bool = True
    fallback_enabled: bool = True
    timeout_seconds: int = 30

    @classmethod
    def from_config(cls) -> "RetrievalConfig":
        """Load configuration from TOML files."""

        def load_config():
            config = get_retrieval_config()["retrieval"]

            # Convert string to enum
            strategy_str = config["default_strategy"]
            strategy = RetrievalStrategy.HYBRID  # default
            for strategy_enum in RetrievalStrategy:
                if strategy_enum.value == strategy_str:
                    strategy = strategy_enum
                    break

            return cls(
                strategy=strategy,
                max_results=config["top_k"],
                min_similarity=config["similarity_threshold"],
                fallback_enabled=config["fallback_enabled"],
                timeout_seconds=config.get("max_retries", 30),  # Reuse as timeout
            )

        return handle_config_error(
            operation=load_config,
            fallback_value=cls(),  # Default constructor
            config_file="config/config.toml",
            section="[retrieval]",
        )


@dataclass
class RetrievalResult:
    """Result of document retrieval operation."""

    query: str
    processed_query: ProcessedQuery
    documents: List[Dict[str, Any]]
    strategy_used: RetrievalStrategy
    search_time: float
    total_time: float
    result_count: int
    confidence: float
    metadata: Dict[str, Any]


class IntelligentRetriever:
    """Intelligent document retriever for multilingual RAG system."""

    def __init__(
        self,
        query_processor: MultilingualQueryProcessor,
        search_engine: SemanticSearchEngine,
        config: RetrievalConfig = None,
    ):
        """
        Initialize intelligent retriever.

        Args:
            query_processor: Multilingual query processor
            search_engine: Semantic search engine
            config: Retrieval configuration
        """
        self.query_processor = query_processor
        self.search_engine = search_engine
        self.config = config or RetrievalConfig()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self._query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_time": 0.0,
            "strategy_usage": {strategy: 0 for strategy in RetrievalStrategy},
        }

    def retrieve(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for multilingual query.

        Args:
            query: User query string
            context: Optional context information
            strategy: Optional override for retrieval strategy

        Returns:
            RetrievalResult with documents and metadata
        """
        start_time = time.time()
        strategy = strategy or self.config.strategy
        context = context or {}

        try:
            self.logger.info(f"Retrieving documents for query: '{query[:50]}...'")

            # Step 1: Process query
            processed_query = self.query_processor.process_query(query, context)

            if processed_query.confidence < 0.1:
                return self._create_low_confidence_result(query, processed_query, start_time)

            # Step 2: Choose retrieval strategy
            if strategy == RetrievalStrategy.ADAPTIVE:
                strategy = self._choose_adaptive_strategy(processed_query)

            # Step 3: Execute retrieval
            search_start = time.time()
            search_response = self._execute_search(processed_query, strategy, context)
            search_time = time.time() - search_start

            # Step 4: Post-process results
            documents = self._post_process_results(search_response, processed_query)

            # Step 5: Calculate overall confidence
            confidence = self._calculate_retrieval_confidence(
                processed_query, search_response, documents
            )

            total_time = time.time() - start_time

            # Update statistics
            self._update_stats(strategy, total_time, len(documents) > 0)

            # Create result
            result = RetrievalResult(
                query=query,
                processed_query=processed_query,
                documents=documents,
                strategy_used=strategy,
                search_time=search_time,
                total_time=total_time,
                result_count=len(documents),
                confidence=confidence,
                metadata={
                    "search_method": (search_response.method.value if search_response else None),
                    "original_results": (search_response.total_results if search_response else 0),
                    "filtered_results": len(documents),
                    "query_type": processed_query.query_type.value,
                    "processing_confidence": processed_query.confidence,
                    "context_used": bool(context),
                },
            )

            self.logger.info(
                f"Retrieved {len(documents)} documents in {total_time:.3f}s "
                f"using {strategy.value} strategy"
            )

            return result

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return self._create_error_result(query, str(e), start_time)

    def _choose_adaptive_strategy(self, processed_query: ProcessedQuery) -> RetrievalStrategy:
        """
        Choose best retrieval strategy based on query characteristics.

        Args:
            processed_query: Processed query information

        Returns:
            Best RetrievalStrategy for this query
        """
        query_type = processed_query.query_type
        keyword_count = len(processed_query.keywords)
        confidence = processed_query.confidence

        # Low confidence queries need more comprehensive search
        if confidence < 0.4:
            return RetrievalStrategy.MULTI_PASS

        # Factual queries with specific keywords work well with simple search
        if query_type == QueryType.FACTUAL and keyword_count >= 2:
            return RetrievalStrategy.SIMPLE

        # Complex queries benefit from hybrid approach
        if query_type in [QueryType.EXPLANATORY, QueryType.COMPARISON]:
            return RetrievalStrategy.HYBRID

        # Summarization queries need multi-pass to get diverse content
        if query_type == QueryType.SUMMARIZATION:
            return RetrievalStrategy.MULTI_PASS

        # Default to simple for general queries
        return RetrievalStrategy.SIMPLE

    def _execute_search(
        self,
        processed_query: ProcessedQuery,
        strategy: RetrievalStrategy,
        context: Dict[str, Any],
    ) -> Optional[SearchResponse]:
        """
        Execute search based on strategy.

        Args:
            processed_query: Processed query
            strategy: Retrieval strategy to use
            context: Additional context

        Returns:
            Search response or None if failed
        """
        try:
            if strategy == RetrievalStrategy.SIMPLE:
                return self._simple_search(processed_query, context)

            elif strategy == RetrievalStrategy.HYBRID:
                return self._hybrid_search(processed_query, context)

            elif strategy == RetrievalStrategy.MULTI_PASS:
                return self._multi_pass_search(processed_query, context)

            else:  # Fallback to simple
                return self._simple_search(processed_query, context)

        except Exception as e:
            self.logger.error(f"Search execution failed: {e}")

            # Try fallback if enabled
            if self.config.fallback_enabled and strategy != RetrievalStrategy.SIMPLE:
                self.logger.info("Attempting fallback to simple search")
                return self._simple_search(processed_query, context)

            return None

    def _simple_search(
        self, processed_query: ProcessedQuery, context: Dict[str, Any]
    ) -> SearchResponse:
        """
        Execute simple semantic search.

        Args:
            processed_query: Processed query
            context: Search context

        Returns:
            Search response
        """
        # Determine top_k based on query type
        if self.config.adaptive_top_k:
            if processed_query.query_type == QueryType.SUMMARIZATION:
                top_k = min(15, self.config.max_results + 5)  # More results for summaries
            elif processed_query.query_type == QueryType.FACTUAL:
                top_k = min(8, self.config.max_results)  # Fewer for specific facts
            else:
                top_k = self.config.max_results
        else:
            top_k = self.config.max_results

        return self.search_engine.search(
            query=processed_query.processed,
            top_k=top_k,
            filters=processed_query.filters,
            method=SearchMethod.SEMANTIC,
        )

    def _hybrid_search(
        self, processed_query: ProcessedQuery, context: Dict[str, Any]
    ) -> SearchResponse:
        """
        Execute hybrid semantic + keyword search.

        Args:
            processed_query: Processed query
            context: Search context

        Returns:
            Combined search response
        """
        return self.search_engine.search(
            query=processed_query.processed,
            top_k=self.config.max_results,
            filters=processed_query.filters,
            method=SearchMethod.HYBRID,
        )

    def _multi_pass_search(
        self, processed_query: ProcessedQuery, context: Dict[str, Any]
    ) -> SearchResponse:
        """
        Execute multiple search passes with different approaches.

        Args:
            processed_query: Processed query
            context: Search context

        Returns:
            Combined search response
        """
        all_results = []
        seen_ids = set()

        # Pass 1: Semantic search with original query
        try:
            semantic_response = self.search_engine.search(
                query=processed_query.processed,
                top_k=max(5, self.config.max_results // 2),
                filters=processed_query.filters,
                method=SearchMethod.SEMANTIC,
            )

            for result in semantic_response.results:
                if result.id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.id)

        except Exception as e:
            self.logger.warning(f"Semantic search pass failed: {e}")

        # Pass 2: Keyword search if we have good keywords
        if processed_query.keywords:
            try:
                keyword_query = " ".join(processed_query.keywords[:3])  # Top 3 keywords
                keyword_response = self.search_engine.search(
                    query=keyword_query,
                    top_k=max(3, self.config.max_results // 3),
                    filters=processed_query.filters,
                    method=SearchMethod.KEYWORD,
                )

                for result in keyword_response.results:
                    if result.id not in seen_ids:
                        all_results.append(result)
                        seen_ids.add(result.id)

            except Exception as e:
                self.logger.warning(f"Keyword search pass failed: {e}")

        # Pass 3: Expanded terms search if available
        if processed_query.expanded_terms:
            try:
                expanded_query = " ".join(processed_query.expanded_terms[:3])
                expanded_response = self.search_engine.search(
                    query=expanded_query,
                    top_k=max(2, self.config.max_results // 5),
                    filters=processed_query.filters,
                    method=SearchMethod.SEMANTIC,
                )

                for result in expanded_response.results:
                    if result.id not in seen_ids:
                        all_results.append(result)
                        seen_ids.add(result.id)

            except Exception as e:
                self.logger.warning(f"Expanded terms search pass failed: {e}")

        # Create combined response
        from ..vectordb.search import SearchMethod, SearchResponse

        # Sort by relevance score
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Limit to max results
        final_results = all_results[: self.config.max_results]

        # Update ranks
        for i, result in enumerate(final_results):
            result.rank = i + 1

        return SearchResponse(
            results=final_results,
            query=processed_query.processed,
            method=SearchMethod.HYBRID,  # Multi-pass is a type of hybrid
            total_time=0.0,  # Will be set by caller
            total_results=len(final_results),
            metadata={"multi_pass": True, "passes": 3},
        )

    def _post_process_results(
        self, search_response: Optional[SearchResponse], processed_query: ProcessedQuery
    ) -> List[Dict[str, Any]]:
        """
        Post-process search results for final output.

        Args:
            search_response: Search response to process
            processed_query: Original processed query

        Returns:
            List of processed document dictionaries
        """
        if not search_response or not search_response.results:
            return []

        documents = []

        for result in search_response.results:
            # Filter by minimum similarity threshold
            if result.score < self.config.min_similarity:
                continue

            # Create document dictionary
            document = {
                "id": result.id,
                "content": result.content,
                "metadata": result.metadata,
                "relevance_score": result.score,
                "rank": result.rank,
                "retrieval_metadata": {
                    "query_type": processed_query.query_type.value,
                    "matching_keywords": self._find_matching_keywords(
                        result.content, processed_query.keywords
                    ),
                    "content_length": len(result.content),
                    "has_title": bool(result.metadata.get("title")),
                    "source": result.metadata.get("source", "unknown"),
                },
            }

            documents.append(document)

        return documents

    def _find_matching_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """
        Find which keywords appear in the content.

        Args:
            content: Document content
            keywords: List of keywords to check

        Returns:
            List of matching keywords
        """
        content_lower = content.lower()
        matching = []

        for keyword in keywords:
            if keyword.lower() in content_lower:
                matching.append(keyword)

        return matching

    def _calculate_retrieval_confidence(
        self,
        processed_query: ProcessedQuery,
        search_response: Optional[SearchResponse],
        documents: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate confidence in retrieval results.

        Args:
            processed_query: Processed query
            search_response: Search response
            documents: Final documents

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not documents:
            return 0.0

        confidence = processed_query.confidence * 0.4  # Base from query processing

        # Boost for good result count
        if 3 <= len(documents) <= 8:
            confidence += 0.2
        elif 1 <= len(documents) <= 2:
            confidence += 0.1

        # Boost for high relevance scores
        if documents:
            avg_score = sum(doc["relevance_score"] for doc in documents) / len(documents)
            if avg_score > 0.7:
                confidence += 0.2
            elif avg_score > 0.5:
                confidence += 0.1

        # Boost for keyword matches
        total_matches = sum(
            len(doc["retrieval_metadata"]["matching_keywords"]) for doc in documents
        )
        if total_matches > len(processed_query.keywords):
            confidence += 0.1

        # Boost for diverse sources
        sources = set(doc["retrieval_metadata"]["source"] for doc in documents)
        if len(sources) > 2:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _create_low_confidence_result(
        self, query: str, processed_query: ProcessedQuery, start_time: float
    ) -> RetrievalResult:
        """Create result for low confidence query."""
        return RetrievalResult(
            query=query,
            processed_query=processed_query,
            documents=[],
            strategy_used=RetrievalStrategy.SIMPLE,
            search_time=0.0,
            total_time=time.time() - start_time,
            result_count=0,
            confidence=processed_query.confidence,
            metadata={
                "error": "Low query processing confidence",
                "suggestions": self.query_processor.suggest_query_improvements(processed_query),
            },
        )

    def _create_error_result(self, query: str, error: str, start_time: float) -> RetrievalResult:
        """Create result for retrieval error."""
        return RetrievalResult(
            query=query,
            processed_query=ProcessedQuery(
                original=query,
                processed=query,
                query_type=QueryType.GENERAL,
                keywords=[],
                expanded_terms=[],
                filters={},
                confidence=0.0,
                metadata={"error": error},
            ),
            documents=[],
            strategy_used=RetrievalStrategy.SIMPLE,
            search_time=0.0,
            total_time=time.time() - start_time,
            result_count=0,
            confidence=0.0,
            metadata={"error": error},
        )

    def _update_stats(self, strategy: RetrievalStrategy, time_taken: float, success: bool):
        """Update performance statistics."""
        self._query_stats["total_queries"] += 1
        self._query_stats["strategy_usage"][strategy] += 1

        if success:
            self._query_stats["successful_queries"] += 1

        # Update average time (moving average)
        total = self._query_stats["total_queries"]
        current_avg = self._query_stats["average_time"]
        self._query_stats["average_time"] = (current_avg * (total - 1) + time_taken) / total

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get retrieval performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        total = self._query_stats["total_queries"]
        if total == 0:
            return {"no_queries": True}

        return {
            "total_queries": total,
            "success_rate": self._query_stats["successful_queries"] / total,
            "average_time": self._query_stats["average_time"],
            "strategy_usage": {
                strategy.value: count / total
                for strategy, count in self._query_stats["strategy_usage"].items()
            },
        }

    def suggest_retrieval_improvements(self, result: RetrievalResult) -> List[str]:
        """
        Suggest improvements for better retrieval results.

        Args:
            result: Retrieval result to analyze

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Low result count
        if result.result_count == 0:
            suggestions.append("Probajte proširiti upit ili koristiti sinonime")
            suggestions.append("Provjerite pravopis ključnih riječi")
        elif result.result_count < 3:
            suggestions.append("Upit je možda previše specifičan - probajte generalniji pristup")

        # Low confidence
        if result.confidence < 0.5:
            suggestions.extend(
                self.query_processor.suggest_query_improvements(result.processed_query)
            )

        # Strategy-specific suggestions
        if result.strategy_used == RetrievalStrategy.SIMPLE:
            if result.processed_query.query_type == QueryType.COMPARISON:
                suggestions.append("Za usporedbe koristite hibridnu strategiju pretrage")

        # Long retrieval time
        if result.total_time > 2.0:
            suggestions.append("Razmotriti skraćivanje upita za brže rezultate")

        return suggestions


def create_intelligent_retriever(
    embedding_model: MultilingualEmbeddingModel,
    storage: ChromaDBStorage,
    strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE,
    language: str = "hr",
) -> IntelligentRetriever:
    """
    Factory function to create intelligent retriever.

    Args:
        embedding_model: Multilingual embedding model
        storage: ChromaDB storage
        strategy: Default retrieval strategy
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Configured IntelligentRetriever
    """
    from ..vectordb.search import create_search_engine
    from .query_processor import create_query_processor

    # Create components
    query_processor = create_query_processor(language=language)
    search_engine = create_search_engine(embedding_model, storage)
