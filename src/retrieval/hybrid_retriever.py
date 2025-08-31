"""
Hybrid retrieval combining dense (embeddings) + sparse (BM25) for Croatian text
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""

    content: str
    score: float
    dense_score: float
    bm25_score: float
    metadata: Dict[str, Any]
    chunk_id: str = ""


class CroatianBM25:
    """BM25 with Croatian text preprocessing."""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with Croatian preprocessing.

        Args:
            documents: List of document texts
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling length normalization
        """
        self.documents = documents
        self.k1 = k1
        self.b = b

        # Preprocess documents for BM25
        self.processed_docs = [self._preprocess_croatian(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.processed_docs, k1=k1, b=b)

    def _preprocess_croatian(self, text: str) -> List[str]:
        """
        Preprocess Croatian text for BM25.

        Args:
            text: Input text

        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation but keep Croatian characters
        text = re.sub(r"[^\w\sčćžšđ]", " ", text)

        # Split into tokens
        tokens = text.split()

        # Remove very short tokens and numbers
        tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]

        return tokens

    def get_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for query."""
        processed_query = self._preprocess_croatian(query)
        return self.bm25.get_scores(processed_query)

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[int, float]]:
        """Get top N document indices and scores."""
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:n]
        return [(idx, scores[idx]) for idx in top_indices]


class HybridRetriever:
    """Hybrid retrieval combining dense embeddings and BM25."""

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_weight: Weight for dense (embedding) scores
            sparse_weight: Weight for sparse (BM25) scores
            bm25_k1: BM25 parameter k1
            bm25_b: BM25 parameter b
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self.documents = []
        self.metadatas = []
        self.bm25 = None
        self.is_indexed = False

    def index_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Index documents for hybrid search.

        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
        """
        self.documents = documents
        self.metadatas = metadatas

        # Initialize BM25
        self.bm25 = CroatianBM25(documents, k1=self.bm25_k1, b=self.bm25_b)
        self.is_indexed = True

    def search(
        self, query: str, dense_results: List[Dict[str, Any]], n_results: int = 5
    ) -> List[HybridResult]:
        """
        Perform hybrid search combining dense and sparse results.

        Args:
            query: Search query
            dense_results: Results from dense (embedding) search
            n_results: Number of results to return

        Returns:
            List of HybridResult objects
        """
        if not self.is_indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")

        # Get BM25 scores for all documents
        bm25_scores = self.bm25.get_scores(query)

        # Normalize BM25 scores to [0, 1]
        if len(bm25_scores) > 0:
            max_bm25 = np.max(bm25_scores)
            if max_bm25 > 0:
                bm25_scores = bm25_scores / max_bm25

        # Create mapping from document content to BM25 score
        content_to_bm25 = {}
        for i, (doc, score) in enumerate(zip(self.documents, bm25_scores)):
            content_to_bm25[doc] = score

        # Process dense results and combine with BM25
        hybrid_results = []

        for dense_result in dense_results:
            # Extract data from dense result
            if isinstance(dense_result, dict):
                content = dense_result.get("content", "")
                dense_score = 1.0 - dense_result.get(
                    "distance", 0.0
                )  # Convert distance to similarity
                metadata = dense_result.get("metadata", {})
            else:
                # Handle list format from ChromaDB
                content = str(dense_result)
                dense_score = 0.5  # Default score
                metadata = {}

            # Normalize dense score to [0, 1]
            dense_score = max(0.0, min(1.0, dense_score))

            # Get BM25 score for this document
            bm25_score = content_to_bm25.get(content, 0.0)

            # Calculate hybrid score
            hybrid_score = self.dense_weight * dense_score + self.sparse_weight * bm25_score

            hybrid_results.append(
                HybridResult(
                    content=content,
                    score=hybrid_score,
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    metadata=metadata,
                    chunk_id=metadata.get("chunk_id", f"chunk_{len(hybrid_results)}"),
                )
            )

        # Sort by hybrid score and return top N
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:n_results]

    def explain_scores(self, results: List[HybridResult]) -> str:
        """Generate explanation of scoring for debugging."""
        explanation = "🔍 Hybrid Retrieval Scores:\n"
        explanation += (
            f"Dense weight: {self.dense_weight:.1f}, BM25 weight: {self.sparse_weight:.1f}\n\n"
        )

        for i, result in enumerate(results, 1):
            explanation += f"{i}. Score: {result.score:.3f}\n"
            explanation += f"   Dense: {result.dense_score:.3f}, BM25: {result.bm25_score:.3f}\n"
            explanation += f"   Content: {result.content[:100]}...\n\n"

        return explanation
