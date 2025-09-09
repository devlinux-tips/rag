"""
Similarity search implementation for multilingual RAG system.
Handles semantic search, ranking, and retrieval optimization.
"""

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..utils.config_protocol import ConfigProvider

import numpy as np

from ..utils.config_loader import (get_language_specific_config,
                                   get_search_config, get_shared_config)
from ..utils.error_handler import create_config_loader
from .embeddings import MultilingualEmbeddingModel
from .storage import ChromaDBStorage

logger = logging.getLogger(__name__)

# Create specialized config loaders
load_vectordb_config = create_config_loader("config/config.toml", __name__)


class SearchMethod(Enum):
    """Available search methods."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class SearchConfig:
    """Configuration for search operations."""

    method: SearchMethod
    top_k: int
    similarity_threshold: float
    max_context_length: int
    rerank: bool
    include_metadata: bool
    include_distances: bool

    @classmethod
    def from_config(
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "SearchConfig":
        """Load configuration from dictionary or config provider."""
        if config_dict:
            search_config = config_dict.get("search", config_dict)
            shared_config = config_dict.get("shared", {})
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get configs through provider
            full_config = config_provider.load_config("config")
            search_config = full_config["search"]
            shared_config = full_config["shared"]

        # Convert string to enum
        method = SearchMethod(search_config["default_method"])

        return cls(
            method=method,
            top_k=search_config["top_k"],
            similarity_threshold=search_config.get(
                "similarity_threshold", shared_config["similarity_threshold"]
            ),
            max_context_length=search_config["max_context_length"],
            rerank=search_config["rerank"],
            include_metadata=search_config["include_metadata"],
            include_distances=search_config["include_distances"],
        )


@dataclass
class SearchResult:
    """Individual search result."""

    content: str
    score: float
    metadata: Dict[str, Any]
    id: str
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SearchResponse:
    """Complete search response."""

    results: List[SearchResult]
    query: str
    method: SearchMethod
    total_time: float
    total_results: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticSearchEngine:
    """Semantic search engine for multilingual documents."""

    def __init__(
        self,
        embedding_model: MultilingualEmbeddingModel,
        storage: ChromaDBStorage,
        config: SearchConfig = None,
    ):
        """
        Initialize search engine.

        Args:
            embedding_model: Embedding model for query encoding
            storage: ChromaDB storage client
            config: Search configuration
        """
        self.embedding_model = embedding_model
        self.storage = storage
        self.config = config or SearchConfig.from_config()
        self.logger = logging.getLogger(__name__)

        # Ensure collection exists
        if not self.storage.collection:
            self.storage.create_collection()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        method: Optional[SearchMethod] = None,
    ) -> SearchResponse:
        """
        Search for relevant documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters
            method: Search method to use

        Returns:
            SearchResponse with results
        """
        start_time = time.time()

        top_k = top_k or self.config.top_k
        method = method or self.config.method

        try:
            if method == SearchMethod.SEMANTIC:
                results = self._semantic_search(query, top_k, filters)
            elif method == SearchMethod.KEYWORD:
                results = self._keyword_search(query, top_k, filters)
            elif method == SearchMethod.HYBRID:
                results = self._hybrid_search(query, top_k, filters)
            else:
                raise ValueError(f"Unknown search method: {method}")

            # Apply post-processing
            if self.config.rerank and len(results) > 1:
                results = self._rerank_results(query, results)

            # Filter by similarity threshold
            results = self._filter_by_threshold(results)

            # Limit results
            results = results[:top_k]

            # Add ranking
            for i, result in enumerate(results):
                result.rank = i + 1

            total_time = time.time() - start_time

            return SearchResponse(
                results=results,
                query=query,
                method=method,
                total_time=total_time,
                total_results=len(results),
                metadata={
                    "filters": filters,
                    "similarity_threshold": self.config.similarity_threshold,
                    "reranked": self.config.rerank,
                },
            )

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    def _semantic_search(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        Perform semantic similarity search.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of search results
        """
        # Encode query
        query_embedding = self.embedding_model.encode_text(query)

        # Ensure embedding is 1D for ChromaDB (some models return 2D arrays)
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]  # Take first (and only) embedding

        # Search in ChromaDB
        chroma_results = self.storage.query_similar(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2,  # Get extra results for filtering/reranking
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        results = []

        if chroma_results and chroma_results.get("ids"):
            ids = chroma_results["ids"][0]
            documents = chroma_results["documents"][0]
            metadatas = chroma_results["metadatas"][0]
            distances = chroma_results["distances"][0]

            for i, doc_id in enumerate(ids):
                # Convert distance to similarity score (ChromaDB uses distance)
                distance = distances[i] if i < len(distances) else 1.0
                score = self._distance_to_similarity(distance)

                result = SearchResult(
                    content=documents[i] if i < len(documents) else "",
                    score=score,
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    id=doc_id,
                )
                results.append(result)

        return results

    def _keyword_search(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        Perform keyword-based search using ChromaDB's where_document filter.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of search results
        """
        # Simple keyword matching using ChromaDB's document filter
        query_terms = query.lower().split()

        # Create document filter for keyword matching
        document_filter = {"$contains": query_terms[0]} if query_terms else None

        try:
            chroma_results = self.storage.query_similar(
                query_texts=[query],  # Still use semantic for ranking
                n_results=top_k * 2,
                where=filters,
                where_document=document_filter,
                include=["documents", "metadatas", "distances"],
            )

            # Convert and score based on keyword matches
            results = []

            if chroma_results and chroma_results.get("ids"):
                ids = chroma_results["ids"][0]
                documents = chroma_results["documents"][0]
                metadatas = chroma_results["metadatas"][0]

                for i, doc_id in enumerate(ids):
                    # Calculate keyword-based score
                    doc_text = documents[i].lower() if i < len(documents) else ""
                    score = self._calculate_keyword_score(query_terms, doc_text)

                    result = SearchResult(
                        content=documents[i] if i < len(documents) else "",
                        score=score,
                        metadata=metadatas[i] if i < len(metadatas) else {},
                        id=doc_id,
                    )
                    results.append(result)

            # Sort by keyword score
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        except Exception as e:
            self.logger.warning(f"Keyword search failed, falling back to semantic: {e}")
            return self._semantic_search(query, top_k, filters)

    def _hybrid_search(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword methods.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of search results
        """
        # Get results from both methods
        semantic_results = self._semantic_search(query, top_k, filters)
        keyword_results = self._keyword_search(query, top_k, filters)

        # Combine and deduplicate results
        combined_results = {}

        # Get weights from config
        search_config = get_search_config()
        weights = {
            "semantic": search_config["weights"]["semantic_weight"],
            "keyword": search_config["weights"]["keyword_weight"],
        }

        semantic_weight = weights["semantic"]
        keyword_weight = weights["keyword"]

        # Add semantic results with weight
        for result in semantic_results:
            combined_results[result.id] = result
            result.score *= semantic_weight

        # Add keyword results with weight
        for result in keyword_results:
            if result.id in combined_results:
                # Combine scores for documents found by both methods
                combined_results[result.id].score += result.score * keyword_weight
            else:
                result.score *= keyword_weight
                combined_results[result.id] = result

        # Convert back to list and sort
        hybrid_results = list(combined_results.values())
        hybrid_results.sort(key=lambda x: x.score, reverse=True)

        return hybrid_results

    def _rerank_results(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank results based on additional criteria.

        Args:
            query: Original query
            results: Initial search results

        Returns:
            Reranked results
        """
        query_terms = set(query.lower().split())

        for result in results:
            # Additional scoring factors
            content_terms = set(result.content.lower().split())

            # Boost score based on:
            # 1. Term overlap
            term_overlap = len(query_terms.intersection(content_terms)) / len(
                query_terms
            )

            # 2. Content length (prefer neither too short nor too long)
            content_length = len(result.content)
            try:
                scoring_config = get_search_config()["scoring"]
                length_score_short = scoring_config["length_score_short"]
                length_score_medium = scoring_config["length_score_medium"]
                length_score_optimal = scoring_config["length_score_optimal"]
                metadata_title_boost = scoring_config["metadata_title_boost"]
            except Exception:
                length_score_short = 0.8
                length_score_medium = 0.9
                length_score_optimal = 1.0
                metadata_title_boost = 1.1

            length_score = length_score_optimal
            if content_length < 100:
                length_score = length_score_short
            elif content_length > 1000:
                length_score = length_score_medium

            # 3. Metadata quality (if title exists, boost score)
            metadata_boost = (
                metadata_title_boost if result.metadata.get("title") else 1.0
            )

            # Apply boosts
            result.score = (
                result.score * (1 + term_overlap * 0.2) * length_score * metadata_boost
            )

        # Re-sort by updated scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _filter_by_threshold(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Filter results by similarity threshold.

        Args:
            results: Search results to filter

        Returns:
            Filtered results
        """
        return [
            result
            for result in results
            if result.score >= self.config.similarity_threshold
        ]

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert ChromaDB distance to similarity score.

        Args:
            distance: Distance value from ChromaDB

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # For cosine distance: similarity = 1 - distance
        # Clamp to [0, 1] range
        similarity = max(0.0, min(1.0, 1.0 - distance))
        return similarity

    def _calculate_keyword_score(self, query_terms: List[str], doc_text: str) -> float:
        """
        Calculate keyword-based relevance score.

        Args:
            query_terms: List of query terms
            doc_text: Document text (lowercased)

        Returns:
            Keyword relevance score
        """
        if not query_terms or not doc_text:
            return 0.0

        doc_terms = doc_text.split()
        doc_term_count = len(doc_terms)

        if doc_term_count == 0:
            return 0.0

        # Count term matches
        matches = sum(1 for term in query_terms if term in doc_text)

        # Calculate TF-like score
        score = matches / len(query_terms)

        # Boost for exact phrase matches
        query_phrase = " ".join(query_terms)
        if query_phrase in doc_text:
            try:
                scoring_config = get_search_config()["scoring"]
                phrase_match_boost = scoring_config["phrase_match_boost"]
            except Exception:
                phrase_match_boost = 1.5
            score *= phrase_match_boost

        return min(1.0, score)

    def get_similar_documents(
        self, document_id: str, top_k: int = 5
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document.

        Args:
            document_id: ID of the reference document
            top_k: Number of similar documents to return

        Returns:
            List of similar documents
        """
        try:
            # Get the reference document
            doc_data = self.storage.get_documents(ids=[document_id])

            if not doc_data or not doc_data.get("documents"):
                raise ValueError(f"Document {document_id} not found")

            reference_doc = doc_data["documents"][0]

            # Use the document content as query
            return self.search(reference_doc, top_k=top_k + 1).results[
                1:
            ]  # Exclude self

        except Exception as e:
            self.logger.error(f"Failed to find similar documents: {e}")
            return []


class SearchResultFormatter:
    """Utility class for formatting search results."""

    @staticmethod
    def format_for_display(response: SearchResponse) -> str:
        """
        Format search response for console display.

        Args:
            response: Search response to format

        Returns:
            Formatted string representation
        """
        lines = []
        lines.append(f"Search Results for: '{response.query}'")
        lines.append(
            f"Method: {response.method.value}, Time: {response.total_time:.3f}s"
        )
        lines.append(f"Found {response.total_results} results")
        lines.append("-" * 50)

        for result in response.results:
            lines.append(f"#{result.rank} (Score: {result.score:.3f})")

            # Show title if available
            if result.metadata.get("title"):
                lines.append(f"Title: {result.metadata['title']}")

            # Show content preview
            content_preview = (
                result.content[:200] + "..."
                if len(result.content) > 200
                else result.content
            )
            lines.append(f"Content: {content_preview}")

            # Show source if available
            if result.metadata.get("source"):
                lines.append(f"Source: {result.metadata['source']}")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def extract_context_chunks(
        response: SearchResponse, max_length: int = None
    ) -> List[str]:
        """
        Extract context chunks for RAG generation.

        Args:
            response: Search response
            max_length: Maximum total length of context (defaults from config)

        Returns:
            List of context chunks
        """
        # Use config default for max_length if not provided
        if max_length is None:
            try:
                search_config = get_search_config()
                max_length = search_config["max_context_length"]
            except Exception:
                max_length = 2000

        chunks = []
        total_length = 0

        for result in response.results:
            chunk_length = len(result.content)

            if total_length + chunk_length > max_length:
                if not chunks:  # Include at least one chunk
                    remaining = max_length - 50  # Leave space for truncation marker
                    truncated = result.content[:remaining] + "..."
                    chunks.append(truncated)
                break

            chunks.append(result.content)
            total_length += chunk_length

        return chunks


def create_search_engine(
    embedding_model: MultilingualEmbeddingModel,
    storage: ChromaDBStorage,
    method: SearchMethod = None,
    top_k: int = None,
) -> SemanticSearchEngine:
    """
    Factory function to create search engine.

    Args:
        embedding_model: Embedding model instance
        storage: Storage client instance
        method: Default search method (defaults from config)
        top_k: Default number of results (defaults from config)

    Returns:
        Configured SemanticSearchEngine
    """
    # Use config defaults if not provided
    if method is None or top_k is None:
        try:
            search_config = get_search_config()
            if method is None:
                method_str = search_config["default_method"]
                method = SearchMethod.SEMANTIC  # default
                for search_method in SearchMethod:
                    if search_method.value == method_str:
                        method = search_method
                        break
            if top_k is None:
                top_k = search_config["top_k"]
        except Exception:
            method = method or SearchMethod.SEMANTIC
            top_k = top_k or 5

    config = SearchConfig(
        method=method,
        top_k=top_k,
        similarity_threshold=0.0,
        max_context_length=2000,
        rerank=True,
        include_metadata=True,
        include_distances=True,
    )
    return SemanticSearchEngine(embedding_model, storage, config)
