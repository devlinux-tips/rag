"""
Similarity search implementation for multilingual RAG system.
Clean slate architecture with pure functions and dependency injection.
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# Pure Data Structures
@dataclass
class SearchQuery:
    """Search query with parameters."""

    text: str
    top_k: int = 5
    method: str = "semantic"
    filters: dict[str, Any] | None = None
    similarity_threshold: float = 0.0
    max_context_length: int = 2000
    rerank: bool = True

    def __post_init__(self):
        """Validate query parameters."""
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")


@dataclass
class SearchResult:
    """Individual search result."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any]
    rank: int | None = None
    method_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SearchResponse:
    """Complete search response with timing and metadata."""

    query: str
    results: list[SearchResult]
    total_results: int
    search_time: float
    method_used: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Add ranking to results if not present."""
        for i, result in enumerate(self.results):
            if result.rank is None:
                result.rank = i + 1
                result.method_used = self.method_used


class SearchMethod(Enum):
    """Available search methods."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# Protocol-based Dependencies (testable interfaces)
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    async def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        ...


class VectorSearchProvider(Protocol):
    """Protocol for vector database search operations."""

    async def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Search by embedding vector."""
        ...

    async def search_by_text(
        self, query_text: str, top_k: int, filters: dict[str, Any] | None = None, include_metadata: bool = True
    ) -> dict[str, Any]:
        """Search by text (if supported by provider)."""
        ...

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get document by ID."""
        ...


class ConfigProvider(Protocol):
    """Protocol for search configuration."""

    def get_search_config(self) -> dict[str, Any]:
        """Get search configuration."""
        ...

    def get_scoring_weights(self) -> dict[str, float]:
        """Get scoring weights for hybrid search."""
        ...


# Pure Functions (testable utilities)
def validate_search_query(query: SearchQuery) -> list[str]:
    """
    Validate search query parameters.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not query.text or not query.text.strip():
        errors.append("Query text cannot be empty")

    if query.top_k <= 0:
        errors.append("top_k must be positive")

    if query.top_k > 100:
        errors.append("top_k cannot exceed 100")

    if query.similarity_threshold < 0 or query.similarity_threshold > 1:
        errors.append("similarity_threshold must be between 0 and 1")

    if query.max_context_length <= 0:
        errors.append("max_context_length must be positive")

    valid_methods = {method.value for method in SearchMethod}
    if query.method not in valid_methods:
        errors.append(f"method must be one of: {', '.join(valid_methods)}")

    return errors


def parse_vector_search_results(raw_results: dict[str, Any], method_used: str = "semantic") -> list[SearchResult]:
    """
    Parse raw vector database results into SearchResult objects.

    Args:
        raw_results: Raw results from vector database
        method_used: Search method identifier

    Returns:
        List of parsed search results
    """
    results: list[SearchResult] = []

    if not raw_results or not raw_results.get("ids"):
        return results

    # Handle ChromaDB-style nested lists
    ids = raw_results["ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]

    # Validate ChromaDB response format - fail fast if unexpected structure
    if "documents" not in raw_results:
        raise ValueError("ChromaDB response missing 'documents' field")
    if "metadatas" not in raw_results:
        raise ValueError("ChromaDB response missing 'metadatas' field")
    if "distances" not in raw_results:
        raise ValueError("ChromaDB response missing 'distances' field")

    documents = raw_results["documents"]
    if isinstance(documents, list) and documents and isinstance(documents[0], list):
        documents = documents[0]

    metadatas = raw_results["metadatas"]
    if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]

    distances = raw_results["distances"]
    if isinstance(distances, list) and distances and isinstance(distances[0], list):
        distances = distances[0]

    # Create SearchResult objects
    for i, doc_id in enumerate(ids):
        content = documents[i] if i < len(documents) else ""
        metadata = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else 0.0

        # Convert distance to similarity score
        score = distance_to_similarity(distance)

        result = SearchResult(id=str(doc_id), content=content, score=score, metadata=metadata, method_used=method_used)
        results.append(result)

    return results


def distance_to_similarity(distance: float) -> float:
    """
    Convert distance metric to similarity score.

    Args:
        distance: Distance value (0 = identical, higher = less similar)

    Returns:
        Similarity score (0-1, higher = more similar)
    """
    # For cosine distance: similarity = 1 - distance
    # Clamp to valid range
    similarity = max(0.0, min(1.0, 1.0 - distance))
    return similarity


def calculate_keyword_score(query_terms: list[str], document_text: str) -> float:
    """
    Calculate keyword-based relevance score.

    Args:
        query_terms: List of query terms (lowercased)
        document_text: Document text (lowercased)

    Returns:
        Keyword relevance score (0-1)
    """
    if not query_terms or not document_text:
        return 0.0

    doc_words = document_text.split()
    if not doc_words:
        return 0.0

    # Count term matches
    matches = 0
    for term in query_terms:
        if term in document_text:
            matches += 1

    # Basic TF-like score
    score = matches / len(query_terms)

    # Boost for exact phrase matches
    query_phrase = " ".join(query_terms)
    if query_phrase in document_text:
        score *= 1.5

    return min(1.0, score)


def combine_scores(
    semantic_score: float, keyword_score: float, semantic_weight: float = 0.7, keyword_weight: float = 0.3
) -> float:
    """
    Combine semantic and keyword scores for hybrid search.

    Args:
        semantic_score: Semantic similarity score (0-1)
        keyword_score: Keyword relevance score (0-1)
        semantic_weight: Weight for semantic score
        keyword_weight: Weight for keyword score

    Returns:
        Combined score
    """
    if semantic_weight + keyword_weight == 0:
        return 0.0

    # Normalize weights
    total_weight = semantic_weight + keyword_weight
    norm_semantic = semantic_weight / total_weight
    norm_keyword = keyword_weight / total_weight

    combined = (semantic_score * norm_semantic) + (keyword_score * norm_keyword)
    return min(1.0, max(0.0, combined))


def rerank_results_by_relevance(
    query_text: str, results: list[SearchResult], boost_factors: dict[str, float] | None = None
) -> list[SearchResult]:
    """
    Rerank search results based on additional relevance factors.

    Args:
        query_text: Original search query
        results: Search results to rerank
        boost_factors: Optional scoring boost factors

    Returns:
        Reranked results (sorted by score descending)
    """
    if not results:
        return results

    # Default boost factors
    if boost_factors is None:
        boost_factors = {
            "term_overlap": 0.2,
            "length_optimal": 1.0,
            "length_short": 0.8,
            "length_long": 0.9,
            "title_boost": 1.1,
        }

    query_terms = set(query_text.lower().split())

    # Apply relevance boosts
    for result in results:
        content_terms = set(result.content.lower().split())

        # Term overlap boost
        if query_terms:
            term_overlap = len(query_terms.intersection(content_terms)) / len(query_terms)
            if "term_overlap" not in boost_factors:
                raise ValueError("Missing 'term_overlap' in boost_factors configuration")
            overlap_boost = 1 + (term_overlap * boost_factors["term_overlap"])
        else:
            overlap_boost = 1.0

        # Content length scoring
        content_length = len(result.content)
        if content_length < 100:
            if "length_short" not in boost_factors:
                raise ValueError("Missing 'length_short' in boost_factors configuration")
            length_boost = boost_factors["length_short"]
        elif content_length > 1000:
            if "length_long" not in boost_factors:
                raise ValueError("Missing 'length_long' in boost_factors configuration")
            length_boost = boost_factors["length_long"]
        else:
            if "length_optimal" not in boost_factors:
                raise ValueError("Missing 'length_optimal' in boost_factors configuration")
            length_boost = boost_factors["length_optimal"]

        # Title/metadata boost
        if "title_boost" not in boost_factors:
            raise ValueError("Missing 'title_boost' in boost_factors configuration")
        title_boost = (
            boost_factors["title_boost"]
            if result.metadata.get("title")  # Keep .get() - metadata from external sources
            else 1.0
        )

        # Apply all boosts
        result.score = result.score * overlap_boost * length_boost * title_boost

    # Sort by updated scores
    results.sort(key=lambda x: x.score, reverse=True)
    return results


def filter_results_by_threshold(results: list[SearchResult], threshold: float) -> list[SearchResult]:
    """
    Filter search results by similarity threshold.

    Args:
        results: Search results to filter
        threshold: Minimum similarity threshold (0-1)

    Returns:
        Filtered results above threshold
    """
    return [result for result in results if result.score >= threshold]


def limit_results(results: list[SearchResult], max_results: int) -> list[SearchResult]:
    """
    Limit number of search results.

    Args:
        results: Search results
        max_results: Maximum number of results to return

    Returns:
        Limited results
    """
    return results[:max_results] if max_results > 0 else results


def extract_context_from_results(results: list[SearchResult], max_context_length: int, separator: str = "\n\n") -> str:
    """
    Extract context text from search results for RAG generation.

    Args:
        results: Search results
        max_context_length: Maximum total context length
        separator: Separator between result contents

    Returns:
        Combined context text
    """
    if not results:
        return ""

    context_parts: list[str] = []
    current_length = 0

    for result in results:
        content = result.content.strip()
        if not content:
            continue

        # Check if adding this content would exceed limit
        additional_length = len(content) + len(separator)
        if current_length + additional_length > max_context_length:
            # If we haven't added any content yet, add truncated version
            if not context_parts:
                remaining = max_context_length - len(separator) - 3  # Space for "..."
                if remaining > 0:
                    context_parts.append(content[:remaining] + "...")
            break

        context_parts.append(content)
        current_length += additional_length

    return separator.join(context_parts)


# Main Search Engine Class
class SemanticSearchEngine:
    """
    Semantic search engine using dependency injection for testability.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        search_provider: VectorSearchProvider,
        config_provider: ConfigProvider,
    ):
        """
        Initialize search engine with injected dependencies.

        Args:
            embedding_provider: Text embedding service
            search_provider: Vector search service
            config_provider: Configuration service
        """
        self.embedding_provider = embedding_provider
        self.search_provider = search_provider
        self.config_provider = config_provider
        self.logger = logging.getLogger(__name__)

    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Execute search query using specified method.

        Args:
            query: Search query with parameters

        Returns:
            Search response with results and metadata

        Raises:
            ValueError: For invalid query parameters
        """
        start_time = time.time()

        # Validate query
        validation_errors = validate_search_query(query)
        if validation_errors:
            raise ValueError(f"Invalid query: {'; '.join(validation_errors)}")

        try:
            # Execute search based on method
            if query.method == SearchMethod.SEMANTIC.value:
                results = await self._semantic_search(query)
            elif query.method == SearchMethod.KEYWORD.value:
                results = await self._keyword_search(query)
            elif query.method == SearchMethod.HYBRID.value:
                results = await self._hybrid_search(query)
            else:
                raise ValueError(f"Unknown search method: {query.method}")

            # Apply post-processing
            if query.rerank and len(results) > 1:
                results = rerank_results_by_relevance(query.text, results)

            # Filter by threshold
            if query.similarity_threshold > 0:
                results = filter_results_by_threshold(results, query.similarity_threshold)

            # Limit results
            results = limit_results(results, query.top_k)

            search_time = time.time() - start_time

            return SearchResponse(
                query=query.text,
                results=results,
                total_results=len(results),
                search_time=search_time,
                method_used=query.method,
                metadata={
                    "filters": query.filters,
                    "similarity_threshold": query.similarity_threshold,
                    "reranked": query.rerank,
                },
            )

        except Exception as e:
            self.logger.error(f"Search failed for query '{query.text}': {e}")
            raise

    async def _semantic_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute semantic similarity search."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_provider.encode_text(query.text)

            # Search vector database
            raw_results = await self.search_provider.search_by_embedding(
                query_embedding=query_embedding,
                top_k=query.top_k * 2,  # Get extra for filtering
                filters=query.filters,
                include_metadata=True,
            )

            # Parse results
            results = parse_vector_search_results(raw_results, "semantic")
            return results

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            raise

    async def _keyword_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute keyword-based search."""
        try:
            # Use text search if provider supports it, otherwise fall back to semantic
            if hasattr(self.search_provider, "search_by_text"):
                raw_results = await self.search_provider.search_by_text(
                    query_text=query.text, top_k=query.top_k * 2, filters=query.filters, include_metadata=True
                )
                results = parse_vector_search_results(raw_results, "keyword")
            else:
                # Fallback: use semantic search and score by keywords
                results = await self._semantic_search(query)
                query_terms = query.text.lower().split()

                # Re-score using keyword matching
                for result in results:
                    keyword_score = calculate_keyword_score(query_terms, result.content.lower())
                    result.score = keyword_score
                    result.method_used = "keyword"

            # Sort by keyword score
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            raise

    async def _hybrid_search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute hybrid search combining semantic and keyword methods."""
        try:
            # Get weights from config - validate required keys
            weights = self.config_provider.get_scoring_weights()
            if "semantic" not in weights:
                raise ValueError("Missing 'semantic' weight in scoring weights configuration")
            if "keyword" not in weights:
                raise ValueError("Missing 'keyword' weight in scoring weights configuration")
            semantic_weight = weights["semantic"]
            keyword_weight = weights["keyword"]

            # Execute both searches concurrently
            semantic_query = SearchQuery(
                text=query.text,
                top_k=query.top_k,
                method="semantic",
                filters=query.filters,
                similarity_threshold=0.0,  # Apply threshold later
                rerank=False,  # Rerank at the end
            )

            keyword_query = SearchQuery(
                text=query.text,
                top_k=query.top_k,
                method="keyword",
                filters=query.filters,
                similarity_threshold=0.0,
                rerank=False,
            )

            # Run searches concurrently
            semantic_results, keyword_results = await asyncio.gather(
                self._semantic_search(semantic_query), self._keyword_search(keyword_query), return_exceptions=True
            )

            # Handle exceptions
            if isinstance(semantic_results, Exception):
                self.logger.warning(f"Semantic search failed in hybrid mode: {semantic_results}")
                semantic_results = []

            if isinstance(keyword_results, Exception):
                self.logger.warning(f"Keyword search failed in hybrid mode: {keyword_results}")
                keyword_results = []

            # Combine results with weighted scoring
            combined_results = {}

            # At this point, both are guaranteed to be lists after exception handling
            assert isinstance(semantic_results, list), "semantic_results should be list after exception handling"
            assert isinstance(keyword_results, list), "keyword_results should be list after exception handling"

            # Add semantic results
            for result in semantic_results:
                combined_results[result.id] = result
                result.score = combine_scores(result.score, 0.0, semantic_weight, 0.0)
                result.method_used = "hybrid"

            # Add/combine keyword results
            query.text.lower().split()
            for result in keyword_results:
                if result.id in combined_results:
                    # Combine scores for documents found by both methods
                    semantic_score = combined_results[result.id].score / semantic_weight
                    keyword_score = result.score
                    combined_score = combine_scores(semantic_score, keyword_score, semantic_weight, keyword_weight)
                    combined_results[result.id].score = combined_score
                else:
                    # New document from keyword search
                    result.score = combine_scores(0.0, result.score, 0.0, keyword_weight)
                    result.method_used = "hybrid"
                    combined_results[result.id] = result

            # Convert to list and sort
            hybrid_results = list(combined_results.values())
            hybrid_results.sort(key=lambda x: x.score, reverse=True)

            return hybrid_results

        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            raise

    async def find_similar_documents(
        self, document_id: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> SearchResponse:
        """
        Find documents similar to a given document.

        Args:
            document_id: ID of reference document
            top_k: Number of similar documents to find
            filters: Optional metadata filters

        Returns:
            Search response with similar documents
        """
        try:
            # Get reference document
            doc_data = await self.search_provider.get_document(document_id)
            if not doc_data or not doc_data.get("content"):
                raise ValueError(f"Document {document_id} not found")

            # Use document content as query
            reference_content = doc_data["content"]

            # Create search query
            similarity_query = SearchQuery(
                text=reference_content,
                top_k=top_k + 1,  # Get extra to exclude self
                method="semantic",
                filters=filters,
            )

            # Execute search
            response = await self.search(similarity_query)

            # Filter out the reference document itself
            similar_results = [result for result in response.results if result.id != document_id][:top_k]

            return SearchResponse(
                query=f"Similar to document {document_id}",
                results=similar_results,
                total_results=len(similar_results),
                search_time=response.search_time,
                method_used="semantic_similarity",
                metadata={"reference_document_id": document_id},
            )

        except Exception as e:
            self.logger.error(f"Similar document search failed: {e}")
            raise


# Factory Functions
def create_search_query(text: str, top_k: int = 5, method: str = "semantic", **kwargs) -> SearchQuery:
    """Factory function to create SearchQuery with validation."""
    return SearchQuery(text=text, top_k=top_k, method=method, **kwargs)


def create_search_engine(
    embedding_provider: EmbeddingProvider, search_provider: VectorSearchProvider, config_provider: ConfigProvider
) -> SemanticSearchEngine:
    """Factory function to create SemanticSearchEngine."""
    return SemanticSearchEngine(
        embedding_provider=embedding_provider, search_provider=search_provider, config_provider=config_provider
    )
