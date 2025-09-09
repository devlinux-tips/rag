"""
Hierarchical Retriever with Smart Routing and Category-Specific Strategies.

This module implements intelligent query routing based on content categorization,
with specialized retrieval strategies for different types of content and queries.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..vectordb.search import SearchResult, SemanticSearchEngine
from .categorization import (CategorizationResult, DocumentCategory,
                             EnhancedQueryCategorizer, RetrievalStrategy)
from .hybrid_retriever import HybridRetriever
from .query_processor import MultilingualQueryProcessor, ProcessedQuery
from .reranker import DocumentReranker


@dataclass
class HierarchicalRetrievalResult:
    """Result of hierarchical retrieval with routing information."""

    documents: List[Dict[str, Any]]
    category: DocumentCategory
    strategy_used: RetrievalStrategy
    retrieval_time: float
    total_results: int
    confidence: float
    routing_metadata: Dict[str, Any]


class HierarchicalRetriever:
    """Intelligent retriever with category-based routing and specialized strategies."""

    def __init__(
        self,
        search_engine: SemanticSearchEngine,
        language: str = "hr",
        enable_reranking: bool = True,
    ):
        """Initialize hierarchical retriever."""
        self.search_engine = search_engine
        self.language = language
        self.enable_reranking = enable_reranking
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.query_processor = MultilingualQueryProcessor(language=language)
        self.categorizer = EnhancedQueryCategorizer(language=language)

        if enable_reranking:
            self.reranker = DocumentReranker(language=language)
        else:
            self.reranker = None

        # Initialize strategy handlers
        self._initialize_strategy_handlers()

        # Performance tracking
        self._retrieval_count = 0
        self._strategy_stats = {}

    def _initialize_strategy_handlers(self) -> None:
        """Initialize strategy-specific retrieval handlers."""
        self.strategy_handlers = {
            RetrievalStrategy.SEMANTIC_FOCUSED: self._semantic_focused_retrieval,
            RetrievalStrategy.KEYWORD_HYBRID: self._keyword_hybrid_retrieval,
            # Note: Cultural context handled by semantic search naturally
            RetrievalStrategy.TECHNICAL_PRECISE: self._technical_precise_retrieval,
            RetrievalStrategy.TEMPORAL_AWARE: self._temporal_aware_retrieval,
            RetrievalStrategy.FAQ_OPTIMIZED: self._faq_optimized_retrieval,
            RetrievalStrategy.COMPARATIVE_STRUCTURED: self._comparative_structured_retrieval,
            RetrievalStrategy.DEFAULT: self._default_retrieval,
        }

    async def retrieve(
        self,
        query: str,
        max_results: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> HierarchicalRetrievalResult:
        """Execute hierarchical retrieval with intelligent routing."""
        start_time = time.time()
        context = context or {}

        self.logger.info(f"🎯 Hierarchical retrieval for: {query[:50]}...")

        try:
            # Step 1: Process and categorize query
            processed_query = self.query_processor.process_query(query, context)
            categorization = self.categorizer.categorize_query(query, context)

            self.logger.info(
                f"📂 Category: {categorization.primary_category.value} "
                f"(confidence: {categorization.confidence:.3f})"
            )

            # Step 2: Select and execute retrieval strategy
            strategy_handler = self.strategy_handlers.get(
                categorization.suggested_strategy,
                self.strategy_handlers[RetrievalStrategy.DEFAULT],
            )

            # Get category configuration
            category_config = self.categorizer.get_category_config(
                categorization.primary_category
            )
            effective_max_results = (
                category_config.max_results if category_config else max_results
            )

            # Execute strategy-specific retrieval
            retrieval_results = await strategy_handler(
                processed_query=processed_query,
                categorization=categorization,
                max_results=effective_max_results,
                context=context,
            )

            # Step 3: Apply category-specific post-processing
            enhanced_results = await self._apply_category_post_processing(
                results=retrieval_results,
                categorization=categorization,
                processed_query=processed_query,
            )

            # Step 4: Rerank if enabled
            if self.reranker and len(enhanced_results) > 1:
                reranked_results = await self.reranker.rerank(
                    query=query,
                    documents=enhanced_results,
                    category=categorization.primary_category.value,
                )
                enhanced_results = reranked_results

            # Step 5: Build routing metadata
            retrieval_time = time.time() - start_time

            routing_metadata = {
                "query_processing": {
                    "original": processed_query.original,
                    "processed": processed_query.processed,
                    "query_type": processed_query.query_type.value,
                    "keywords": processed_query.keywords,
                    "expanded_terms": processed_query.expanded_terms,
                },
                "categorization": {
                    "primary": categorization.primary_category.value,
                    "secondary": [
                        cat.value for cat in categorization.secondary_categories
                    ],
                    "confidence": categorization.confidence,
                    "detection_signals": categorization.signals,
                },
                "strategy": {
                    "selected": categorization.suggested_strategy.value,
                    "config_used": category_config.name
                    if category_config
                    else "default",
                    "similarity_threshold": category_config.similarity_threshold
                    if category_config
                    else 0.3,
                },
                "performance": {
                    "retrieval_time": retrieval_time,
                    "results_count": len(enhanced_results),
                    "reranking_applied": self.reranker is not None,
                },
            }

            # Update statistics
            self._update_strategy_stats(
                categorization.suggested_strategy, retrieval_time
            )

            # Calculate overall confidence
            overall_confidence = (
                categorization.confidence * 0.6
                + (  # Category confidence
                    sum(doc["final_score"] for doc in enhanced_results[:3])
                    / min(3, len(enhanced_results))
                )
                * 0.4  # Top results confidence
                if enhanced_results
                else categorization.confidence * 0.6
            )

            result = HierarchicalRetrievalResult(
                documents=enhanced_results,
                category=categorization.primary_category,
                strategy_used=categorization.suggested_strategy,
                retrieval_time=retrieval_time,
                total_results=len(enhanced_results),
                confidence=overall_confidence,
                routing_metadata=routing_metadata,
            )

            self.logger.info(
                f"✅ Hierarchical retrieval complete: {len(enhanced_results)} results "
                f"in {retrieval_time:.3f}s using {categorization.suggested_strategy.value}"
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ Hierarchical retrieval failed: {e}")

            # Fallback to simple retrieval
            fallback_results = await self._default_retrieval(
                processed_query=self.query_processor.process_query(query),
                categorization=None,
                max_results=max_results,
                context=context,
            )

            return HierarchicalRetrievalResult(
                documents=fallback_results,
                category=DocumentCategory.GENERAL,
                strategy_used=RetrievalStrategy.DEFAULT,
                retrieval_time=time.time() - start_time,
                total_results=len(fallback_results),
                confidence=0.3,  # Low confidence for fallback
                routing_metadata={"error": str(e), "fallback_used": True},
            )

    async def _semantic_focused_retrieval(
        self,
        processed_query: ProcessedQuery,
        categorization: Optional[CategorizationResult],
        max_results: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Semantic-focused retrieval strategy."""
        self.logger.debug("🔍 Using semantic-focused retrieval")

        # Use processed query for better semantic matching
        search_query = processed_query.processed

        # Higher similarity threshold for precision
        similarity_threshold = 0.5
        if categorization and categorization.filters.get("similarity_threshold"):
            similarity_threshold = categorization.filters["similarity_threshold"]

        # Expand search slightly for better recall
        expanded_results = max_results * 2

        results = await self.search_engine.search(
            query_text=search_query,
            k=expanded_results,
            similarity_threshold=similarity_threshold,
        )

        # Convert SearchResult to dict format and filter by threshold
        formatted_results = []
        for result in results[:max_results]:
            if result.similarity_score >= similarity_threshold:
                formatted_results.append(
                    {
                        "content": result.content,
                        "metadata": result.metadata,
                        "similarity_score": result.similarity_score,
                        "final_score": result.similarity_score,  # Pure semantic score
                    }
                )

        return formatted_results

    async def _keyword_hybrid_retrieval(
        self,
        processed_query: ProcessedQuery,
        categorization: Optional[CategorizationResult],
        max_results: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Hybrid semantic + keyword retrieval strategy."""
        self.logger.debug("🔍 Using keyword-hybrid retrieval")

        # Use both processed query and expanded terms
        primary_query = processed_query.processed
        expanded_query = " ".join(processed_query.expanded_terms[:5])  # Limit expansion

        # Combine queries
        combined_query = f"{primary_query} {expanded_query}".strip()

        # Adjust similarity threshold for hybrid approach
        similarity_threshold = 0.4
        if categorization and categorization.filters.get("similarity_threshold"):
            similarity_threshold = categorization.filters["similarity_threshold"] * 0.8

        results = await self.search_engine.search(
            query_text=combined_query,
            k=max_results * 2,
            similarity_threshold=similarity_threshold,
        )

        # Boost results that contain original keywords
        formatted_results = []
        keywords = set(processed_query.keywords)

        for result in results[:max_results]:
            content_lower = result.content.lower()

            # Calculate keyword match bonus
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            keyword_bonus = (keyword_matches / len(keywords)) * 0.2 if keywords else 0

            final_score = result.similarity_score + keyword_bonus

            formatted_results.append(
                {
                    "content": result.content,
                    "metadata": result.metadata,
                    "similarity_score": result.similarity_score,
                    "keyword_bonus": keyword_bonus,
                    "final_score": min(1.0, final_score),
                }
            )

        # Sort by final score
        formatted_results.sort(key=lambda x: x["final_score"], reverse=True)

        return formatted_results

    # Note: Cultural context removed - multilingual embeddings naturally capture cultural nuances

    async def _technical_precise_retrieval(
        self,
        processed_query: ProcessedQuery,
        categorization: Optional[CategorizationResult],
        max_results: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Precise retrieval for technical content."""
        self.logger.debug("🔍 Using technical-precise retrieval")

        # Use original query to preserve technical terms
        search_query = processed_query.original

        # Higher similarity threshold for precision
        similarity_threshold = 0.6

        results = await self.search_engine.search(
            query_text=search_query,
            k=max_results * 2,
            similarity_threshold=similarity_threshold,
        )

        # Boost technical content
        formatted_results = []
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

        for result in results:
            content_lower = result.content.lower()

            # Technical content boost
            technical_score = sum(
                0.1 for indicator in technical_indicators if indicator in content_lower
            )

            # Exact term matching boost
            query_words = set(processed_query.original.lower().split())
            content_words = set(content_lower.split())
            exact_matches = len(query_words.intersection(content_words))
            exact_bonus = (exact_matches / len(query_words)) * 0.2 if query_words else 0

            final_score = result.similarity_score + technical_score + exact_bonus

            formatted_results.append(
                {
                    "content": result.content,
                    "metadata": result.metadata,
                    "similarity_score": result.similarity_score,
                    "technical_boost": technical_score,
                    "exact_match_bonus": exact_bonus,
                    "final_score": min(1.0, final_score),
                }
            )

        # Sort by final score
        formatted_results.sort(key=lambda x: x["final_score"], reverse=True)

        return formatted_results[:max_results]

    async def _temporal_aware_retrieval(
        self,
        processed_query: ProcessedQuery,
        categorization: Optional[CategorizationResult],
        max_results: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Temporal-aware retrieval for time-sensitive content."""
        self.logger.debug("🔍 Using temporal-aware retrieval")

        search_query = processed_query.processed

        results = await self.search_engine.search(
            query_text=search_query,
            k=max_results * 2,
            similarity_threshold=0.4,
        )

        # Boost recent content and time-sensitive terms
        formatted_results = []
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

        current_year = 2024  # Could be made configurable

        for result in results:
            content_lower = result.content.lower()
            metadata = result.metadata or {}

            # Temporal relevance boost
            temporal_score = (
                sum(0.15 for term in temporal_terms if term in content_lower) * 0.5
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

            final_score = result.similarity_score + temporal_score

            formatted_results.append(
                {
                    "content": result.content,
                    "metadata": metadata,
                    "similarity_score": result.similarity_score,
                    "temporal_boost": temporal_score,
                    "final_score": min(1.0, final_score),
                }
            )

        # Sort by final score (recent content prioritized)
        formatted_results.sort(key=lambda x: x["final_score"], reverse=True)

        return formatted_results[:max_results]

    async def _faq_optimized_retrieval(
        self,
        processed_query: ProcessedQuery,
        categorization: Optional[CategorizationResult],
        max_results: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FAQ-optimized retrieval for question-answer pairs."""
        self.logger.debug("🔍 Using FAQ-optimized retrieval")

        # Lower similarity threshold for FAQ matching
        similarity_threshold = 0.3

        results = await self.search_engine.search(
            query_text=processed_query.processed,
            k=max_results * 3,
            similarity_threshold=similarity_threshold,
        )

        # Boost FAQ-style content
        formatted_results = []
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

        for result in results:
            content_lower = result.content.lower()

            # FAQ content boost
            faq_score = (
                sum(0.1 for indicator in faq_indicators if indicator in content_lower)
                * 0.3
            )  # Moderate FAQ boost

            # Question-answer structure boost
            if any(
                pattern in content_lower
                for pattern in ["q:", "a:", "pitanje:", "odgovor:"]
            ):
                faq_score += 0.2

            # Short, concise answer boost (FAQ answers are usually brief)
            if 50 <= len(result.content) <= 300:
                faq_score += 0.1

            final_score = result.similarity_score + faq_score

            formatted_results.append(
                {
                    "content": result.content,
                    "metadata": result.metadata,
                    "similarity_score": result.similarity_score,
                    "faq_boost": faq_score,
                    "final_score": min(1.0, final_score),
                }
            )

        # Sort by final score
        formatted_results.sort(key=lambda x: x["final_score"], reverse=True)

        return formatted_results[:max_results]

    async def _comparative_structured_retrieval(
        self,
        processed_query: ProcessedQuery,
        categorization: Optional[CategorizationResult],
        max_results: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Structured retrieval for comparative content."""
        self.logger.debug("🔍 Using comparative-structured retrieval")

        search_query = processed_query.processed

        results = await self.search_engine.search(
            query_text=search_query,
            k=max_results * 2,
            similarity_threshold=0.4,
        )

        # Boost comparative content
        formatted_results = []
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

        for result in results:
            content_lower = result.content.lower()

            # Comparative content boost
            comparative_score = (
                sum(0.1 for term in comparative_terms if term in content_lower) * 0.4
            )

            # Structure indicators boost (tables, lists, comparisons)
            structure_indicators = ["|", "vs", "•", "-", "1.", "2.", "prvo", "drugo"]
            structure_score = sum(
                0.05
                for indicator in structure_indicators
                if indicator in result.content
            )

            final_score = result.similarity_score + comparative_score + structure_score

            formatted_results.append(
                {
                    "content": result.content,
                    "metadata": result.metadata,
                    "similarity_score": result.similarity_score,
                    "comparative_boost": comparative_score,
                    "structure_boost": structure_score,
                    "final_score": min(1.0, final_score),
                }
            )

        # Sort by final score
        formatted_results.sort(key=lambda x: x["final_score"], reverse=True)

        return formatted_results[:max_results]

    async def _default_retrieval(
        self,
        processed_query: ProcessedQuery,
        categorization: Optional[CategorizationResult],
        max_results: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Default retrieval strategy."""
        self.logger.debug("🔍 Using default retrieval")

        search_query = processed_query.processed

        results = await self.search_engine.search(
            query_text=search_query,
            k=max_results,
            similarity_threshold=0.3,
        )

        # Simple conversion to dict format
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "content": result.content,
                    "metadata": result.metadata,
                    "similarity_score": result.similarity_score,
                    "final_score": result.similarity_score,
                }
            )

        return formatted_results

    async def _apply_category_post_processing(
        self,
        results: List[Dict[str, Any]],
        categorization: CategorizationResult,
        processed_query: ProcessedQuery,
    ) -> List[Dict[str, Any]]:
        """Apply category-specific post-processing to results."""
        if not results:
            return results

        # Apply category-specific filtering
        category_config = self.categorizer.get_category_config(
            categorization.primary_category
        )
        if not category_config:
            return results

        # Filter by similarity threshold
        filtered_results = [
            result
            for result in results
            if result["similarity_score"] >= category_config.similarity_threshold
        ]

        # Apply penalty keywords if configured
        if category_config.penalty_keywords:
            penalty_filtered = []
            for result in filtered_results:
                content_lower = result["content"].lower()
                penalty_score = sum(
                    0.1
                    for keyword in category_config.penalty_keywords
                    if keyword in content_lower
                )

                if penalty_score < 0.3:  # Threshold for penalty rejection
                    final_score = result["final_score"] - penalty_score
                    result["final_score"] = max(0.0, final_score)
                    result["penalty_applied"] = penalty_score
                    penalty_filtered.append(result)

            filtered_results = penalty_filtered

        # Add category metadata to results
        for result in filtered_results:
            result.setdefault("metadata", {})[
                "detected_category"
            ] = categorization.primary_category.value
            result.setdefault("metadata", {})[
                "retrieval_strategy"
            ] = categorization.suggested_strategy.value

        # Sort by final score and limit to category max_results
        filtered_results.sort(key=lambda x: x["final_score"], reverse=True)

        return filtered_results[: category_config.max_results]

    def _update_strategy_stats(
        self, strategy: RetrievalStrategy, retrieval_time: float
    ) -> None:
        """Update performance statistics for strategies."""
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
            "strategy_stats": self._strategy_stats.copy(),
            "language": self.language,
            "reranking_enabled": self.reranker is not None,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on hierarchical retriever."""
        health = {
            "status": "healthy",
            "components": {
                "search_engine": "healthy" if self.search_engine else "missing",
                "query_processor": "healthy" if self.query_processor else "missing",
                "categorizer": "healthy" if self.categorizer else "missing",
                "reranker": "healthy" if self.reranker else "disabled",
            },
            "statistics": self.get_performance_stats(),
            "language": self.language,
        }

        # Check if any components are missing
        if any(status == "missing" for status in health["components"].values()):
            health["status"] = "degraded"

        return health


def create_hierarchical_retriever(
    search_engine: SemanticSearchEngine,
    language: str = "hr",
    enable_reranking: bool = True,
) -> HierarchicalRetriever:
    """Factory function to create hierarchical retriever."""
    return HierarchicalRetriever(
        search_engine=search_engine,
        language=language,
        enable_reranking=enable_reranking,
    )
