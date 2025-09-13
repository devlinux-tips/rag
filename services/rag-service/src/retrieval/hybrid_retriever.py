"""
Hybrid retrieval system combining dense and sparse search methods.
Clean architecture with dependency injection and modular components.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from ..utils.config_models import HybridRetrievalConfig

logger = logging.getLogger(__name__)


# ===== PURE FUNCTIONS =====


def normalize_text_for_bm25(
    text: str, stop_words: set, min_token_length: int = 3
) -> list[str]:
    """
    Preprocess text for BM25 scoring.
    Pure function - no side effects, deterministic output.

    Args:
        text: Input text to process
        stop_words: Set of stop words to remove
        min_token_length: Minimum token length to keep

    Returns:
        List of processed tokens

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be string, got {type(text)}")

    if not isinstance(stop_words, set):
        raise ValueError(f"Stop words must be set, got {type(stop_words)}")

    if not isinstance(min_token_length, int) or min_token_length < 1:
        raise ValueError(
            f"Min token length must be positive integer, got {min_token_length}"
        )

    # Convert to lowercase
    processed_text = text.lower()

    # Remove punctuation, keep alphanumeric and Croatian diacritics
    processed_text = re.sub(r"[^\w\sčćšđžČĆŠĐŽ]", " ", processed_text)

    # Split into tokens
    tokens = processed_text.split()

    # Filter tokens
    filtered_tokens = [
        token
        for token in tokens
        if len(token) >= min_token_length
        and not token.isdigit()
        and token not in stop_words
    ]

    return filtered_tokens


def calculate_bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
    avgdl: float = 0.0,
    corpus_size: int = 1,
) -> float:
    """
    Calculate BM25 score for a document against query.
    Pure function - no side effects, deterministic output.

    Args:
        query_tokens: Tokenized query
        doc_tokens: Tokenized document
        k1: BM25 parameter for term frequency saturation
        b: BM25 parameter for length normalization
        avgdl: Average document length in corpus
        corpus_size: Size of corpus for IDF calculation

    Returns:
        BM25 score for document

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(query_tokens, list):
        raise ValueError(f"Query tokens must be list, got {type(query_tokens)}")

    if not isinstance(doc_tokens, list):
        raise ValueError(f"Doc tokens must be list, got {type(doc_tokens)}")

    if not isinstance(k1, (int, float)) or k1 < 0:
        raise ValueError(f"k1 must be non-negative number, got {k1}")

    if not isinstance(b, (int, float)) or not (0 <= b <= 1):
        raise ValueError(f"b must be between 0 and 1, got {b}")

    if len(query_tokens) == 0:
        return 0.0

    if len(doc_tokens) == 0:
        return 0.0

    doc_len = len(doc_tokens)
    if avgdl <= 0:
        avgdl = doc_len

    # Count term frequencies in document
    doc_term_freq = {}
    for token in doc_tokens:
        if token not in doc_term_freq:
            doc_term_freq[token] = 0
        doc_term_freq[token] += 1

    score = 0.0

    for query_token in query_tokens:
        if query_token in doc_term_freq:
            tf = doc_term_freq[query_token]

            # Simple IDF approximation (can be enhanced with corpus statistics)
            idf = np.log((corpus_size + 1) / (1 + 1))  # Simplified IDF

            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))

            score += idf * (numerator / denominator)

    return max(0.0, score)


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize scores to [0, 1] range.
    Pure function - no side effects, deterministic output.

    Args:
        scores: List of scores to normalize

    Returns:
        List of normalized scores

    Raises:
        ValueError: If scores is invalid
    """
    if not isinstance(scores, list):
        raise ValueError(f"Scores must be list, got {type(scores)}")

    if not all(isinstance(s, (int, float)) for s in scores):
        raise ValueError("All scores must be numbers")

    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # All scores are equal
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def combine_hybrid_scores(
    dense_scores: list[float],
    sparse_scores: list[float],
    dense_weight: float,
    sparse_weight: float,
) -> list[float]:
    """
    Combine dense and sparse scores with weights.
    Pure function - no side effects, deterministic output.

    Args:
        dense_scores: Dense similarity scores
        sparse_scores: Sparse BM25 scores
        dense_weight: Weight for dense scores
        sparse_weight: Weight for sparse scores

    Returns:
        List of combined hybrid scores

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(dense_scores, list):
        raise ValueError(f"Dense scores must be list, got {type(dense_scores)}")

    if not isinstance(sparse_scores, list):
        raise ValueError(f"Sparse scores must be list, got {type(sparse_scores)}")

    if len(dense_scores) != len(sparse_scores):
        raise ValueError(
            f"Score lists must have same length: {len(dense_scores)} vs {len(sparse_scores)}"
        )

    if not isinstance(dense_weight, (int, float)) or dense_weight < 0:
        raise ValueError(
            f"Dense weight must be non-negative number, got {dense_weight}"
        )

    if not isinstance(sparse_weight, (int, float)) or sparse_weight < 0:
        raise ValueError(
            f"Sparse weight must be non-negative number, got {sparse_weight}"
        )

    if dense_weight + sparse_weight == 0:
        raise ValueError("At least one weight must be positive")

    # Normalize weights
    total_weight = dense_weight + sparse_weight
    normalized_dense_weight = dense_weight / total_weight
    normalized_sparse_weight = sparse_weight / total_weight

    hybrid_scores = [
        normalized_dense_weight * dense + normalized_sparse_weight * sparse
        for dense, sparse in zip(dense_scores, sparse_scores)
    ]

    return hybrid_scores


def rank_results_by_score(
    results: list[dict[str, Any]], score_key: str = "score", descending: bool = True
) -> list[dict[str, Any]]:
    """
    Rank results by score.
    Pure function - no side effects, deterministic output.

    Args:
        results: List of result dictionaries
        score_key: Key name for score in dictionaries
        descending: Whether to sort in descending order

    Returns:
        Sorted list of results

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(results, list):
        raise ValueError(f"Results must be list, got {type(results)}")

    if not isinstance(score_key, str):
        raise ValueError(f"Score key must be string, got {type(score_key)}")

    # Validate that all results have the score key
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"Result at index {i} must be dict, got {type(result)}")
        if score_key not in result:
            raise ValueError(f"Result at index {i} missing score key '{score_key}'")
        if not isinstance(result[score_key], (int, float)):
            raise ValueError(
                f"Score at index {i} must be number, got {type(result[score_key])}"
            )

    # Sort results
    return sorted(results, key=lambda x: x[score_key], reverse=descending)


def calculate_corpus_statistics(documents: list[list[str]]) -> dict[str, float]:
    """
    Calculate corpus statistics for BM25.
    Pure function - no side effects, deterministic output.

    Args:
        documents: List of tokenized documents

    Returns:
        Dictionary with corpus statistics

    Raises:
        ValueError: If documents is invalid
    """
    if not isinstance(documents, list):
        raise ValueError(f"Documents must be list, got {type(documents)}")

    if not all(isinstance(doc, list) for doc in documents):
        raise ValueError("All documents must be lists of tokens")

    if not documents:
        return {"avgdl": 0.0, "total_docs": 0, "total_tokens": 0}

    total_tokens = sum(len(doc) for doc in documents)
    total_docs = len(documents)
    avgdl = total_tokens / total_docs if total_docs > 0 else 0.0

    return {"avgdl": avgdl, "total_docs": total_docs, "total_tokens": total_tokens}


# ===== DATA STRUCTURES =====


# Note: HybridRetrievalConfig is now imported from config_models.py
# Keep local HybridConfig for internal processing configuration
@dataclass
class HybridConfig:
    """Internal configuration for hybrid retrieval processing."""

    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    min_token_length: int = 3
    normalize_scores: bool = True
    min_score_threshold: float = 0.0

    @classmethod
    def from_validated_config(
        cls, hybrid_config: HybridRetrievalConfig
    ) -> "HybridConfig":
        """Create HybridConfig from validated HybridRetrievalConfig."""
        return cls(
            dense_weight=hybrid_config.dense_weight,
            sparse_weight=hybrid_config.sparse_weight,
            bm25_k1=hybrid_config.bm25_k1,
            bm25_b=hybrid_config.bm25_b,
            # Set defaults for fields not in HybridRetrievalConfig
            min_token_length=3,
            normalize_scores=True,
            min_score_threshold=0.0,
        )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.dense_weight, (int, float)) or self.dense_weight < 0:
            raise ValueError("Dense weight must be non-negative number")

        if not isinstance(self.sparse_weight, (int, float)) or self.sparse_weight < 0:
            raise ValueError("Sparse weight must be non-negative number")

        if self.dense_weight + self.sparse_weight == 0:
            raise ValueError("At least one weight must be positive")

        if not isinstance(self.bm25_k1, (int, float)) or self.bm25_k1 < 0:
            raise ValueError("BM25 k1 must be non-negative number")

        if not isinstance(self.bm25_b, (int, float)) or not (0 <= self.bm25_b <= 1):
            raise ValueError("BM25 b must be between 0 and 1")

        if not isinstance(self.min_token_length, int) or self.min_token_length < 1:
            raise ValueError("Min token length must be positive integer")

        if not isinstance(self.normalize_scores, bool):
            raise ValueError("Normalize scores must be boolean")

        if not isinstance(self.min_score_threshold, (int, float)):
            raise ValueError("Min score threshold must be number")


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""

    content: str
    score: float
    dense_score: float
    sparse_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result after initialization."""
        if not isinstance(self.content, str):
            raise ValueError("Content must be string")

        if not isinstance(self.score, (int, float)):
            raise ValueError("Score must be number")

        if not isinstance(self.dense_score, (int, float)):
            raise ValueError("Dense score must be number")

        if not isinstance(self.sparse_score, (int, float)):
            raise ValueError("Sparse score must be number")

        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be dict")


@dataclass
class DenseResult:
    """Dense search result structure."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ===== PROTOCOLS =====


class StopWordsProvider(Protocol):
    """Protocol for stop words providers."""

    def get_stop_words(self, language: str) -> set:
        """Get stop words for language."""
        ...


class CorpusIndexer(Protocol):
    """Protocol for corpus indexing."""

    def index_documents(self, documents: list[str]) -> None:
        """Index documents for BM25 scoring."""
        ...

    def get_scores(self, query: str) -> list[float]:
        """Get BM25 scores for query against indexed documents."""
        ...


# ===== CORE CLASSES =====


class BM25Scorer:
    """BM25 scorer with multilingual preprocessing."""

    def __init__(
        self,
        stop_words_provider: StopWordsProvider,
        language: str,
        config: HybridConfig,
    ):
        """
        Initialize BM25 scorer with dependency injection.

        Args:
            stop_words_provider: Provider for stop words
            language: Language code
            config: Hybrid configuration
        """
        self.stop_words_provider = stop_words_provider
        self.language = language
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Get stop words (fail-fast approach)
        self.stop_words = stop_words_provider.get_stop_words(language)
        self.logger.debug(f"Loaded {len(self.stop_words)} stop words for {language}")

        # Initialize corpus data
        self.documents = []
        self.tokenized_docs = []
        self.corpus_stats = {}
        self.is_indexed = False

    def index_documents(self, documents: list[str]) -> None:
        """
        Index documents for BM25 scoring.

        Args:
            documents: List of document strings

        Raises:
            ValueError: If documents is invalid
        """
        if not isinstance(documents, list):
            raise ValueError(f"Documents must be list, got {type(documents)}")

        if not all(isinstance(doc, str) for doc in documents):
            raise ValueError("All documents must be strings")

        try:
            self.documents = documents

            # Tokenize all documents
            self.tokenized_docs = [
                normalize_text_for_bm25(
                    doc, self.stop_words, self.config.min_token_length
                )
                for doc in documents
            ]

            # Calculate corpus statistics
            self.corpus_stats = calculate_corpus_statistics(self.tokenized_docs)

            self.is_indexed = True
            self.logger.debug(f"Indexed {len(documents)} documents for BM25")

        except Exception as e:
            self.logger.error(f"Failed to index documents: {e}")
            raise

    def get_scores(self, query: str) -> list[float]:
        """
        Get BM25 scores for query against indexed documents.

        Args:
            query: Search query string

        Returns:
            List of BM25 scores

        Raises:
            ValueError: If query is invalid or documents not indexed
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be string, got {type(query)}")

        if not self.is_indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")

        try:
            # Tokenize query
            query_tokens = normalize_text_for_bm25(
                query, self.stop_words, self.config.min_token_length
            )

            if not query_tokens:
                return [0.0] * len(self.documents)

            # Calculate BM25 scores for all documents
            scores = []
            for doc_tokens in self.tokenized_docs:
                score = calculate_bm25_score(
                    query_tokens=query_tokens,
                    doc_tokens=doc_tokens,
                    k1=self.config.bm25_k1,
                    b=self.config.bm25_b,
                    avgdl=self.corpus_stats["avgdl"],
                    corpus_size=self.corpus_stats["total_docs"],
                )
                scores.append(score)

            return scores

        except Exception as e:
            self.logger.error(f"Failed to calculate BM25 scores: {e}")
            raise


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search."""

    def __init__(
        self,
        stop_words_provider: StopWordsProvider,
        language: str,
        config: HybridConfig,
    ):
        """
        Initialize hybrid retriever with dependency injection.

        Args:
            stop_words_provider: Provider for stop words
            language: Language code
            config: Hybrid configuration
        """
        self.language = language
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize BM25 scorer
        self.bm25_scorer = BM25Scorer(stop_words_provider, language, config)

        # Initialize state
        self.documents = []
        self.is_ready = False

    def index_documents(self, documents: list[str]) -> None:
        """
        Index documents for hybrid retrieval.

        Args:
            documents: List of document strings
        """
        try:
            self.documents = documents
            self.bm25_scorer.index_documents(documents)
            self.is_ready = True

            self.logger.info(f"Indexed {len(documents)} documents for hybrid retrieval")

        except Exception as e:
            self.logger.error(f"Failed to index documents: {e}")
            raise

    def retrieve(
        self, query: str, dense_results: list[DenseResult], top_k: int = 10
    ) -> list[HybridResult]:
        """
        Perform hybrid retrieval combining dense and sparse results.

        Args:
            query: Search query
            dense_results: Results from dense search
            top_k: Number of results to return

        Returns:
            List of hybrid results

        Raises:
            ValueError: If inputs are invalid or retriever not ready
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be string, got {type(query)}")

        if not isinstance(dense_results, list):
            raise ValueError(f"Dense results must be list, got {type(dense_results)}")

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"Top k must be positive integer, got {top_k}")

        if not self.is_ready:
            raise ValueError("Retriever not ready. Call index_documents() first.")

        try:
            # Get BM25 scores for all documents
            bm25_scores = self.bm25_scorer.get_scores(query)

            # Normalize BM25 scores if requested
            if self.config.normalize_scores:
                bm25_scores = normalize_scores(bm25_scores)

            # Create content to BM25 score mapping
            content_to_bm25 = {
                doc: score for doc, score in zip(self.documents, bm25_scores)
            }

            # Process dense results and combine with BM25
            hybrid_results = []

            for dense_result in dense_results:
                content = dense_result.content
                dense_score = dense_result.score

                # Get BM25 score for this content
                if content not in content_to_bm25:
                    raise ValueError(
                        f"Content not indexed in BM25 scorer: {content[:100]}..."
                    )
                bm25_score = content_to_bm25[content]

                # Calculate hybrid score
                hybrid_score = combine_hybrid_scores(
                    dense_scores=[dense_score],
                    sparse_scores=[bm25_score],
                    dense_weight=self.config.dense_weight,
                    sparse_weight=self.config.sparse_weight,
                )[0]

                # Apply minimum score threshold
                if hybrid_score >= self.config.min_score_threshold:
                    hybrid_results.append(
                        HybridResult(
                            content=content,
                            score=hybrid_score,
                            dense_score=dense_score,
                            sparse_score=bm25_score,
                            metadata=dense_result.metadata,
                        )
                    )

            # Rank by hybrid score
            ranked_results = rank_results_by_score(
                [{"result": r, "score": r.score} for r in hybrid_results],
                score_key="score",
                descending=True,
            )

            # Extract results and apply top_k limit
            final_results = [item["result"] for item in ranked_results[:top_k]]

            self.logger.debug(f"Retrieved {len(final_results)} hybrid results")
            return final_results

        except Exception as e:
            self.logger.error(f"Failed to perform hybrid retrieval: {e}")
            raise

    def explain_scores(self, results: list[HybridResult]) -> str:
        """
        Generate explanation of hybrid scoring.

        Args:
            results: List of hybrid results

        Returns:
            Formatted explanation string
        """
        if not results:
            return "No results to explain."

        explanation = ["🔍 Hybrid Retrieval Score Explanation:"]
        explanation.append(f"Dense weight: {self.config.dense_weight:.2f}")
        explanation.append(f"Sparse weight: {self.config.sparse_weight:.2f}")
        explanation.append("")

        for i, result in enumerate(results, 1):
            explanation.append(f"{i}. Hybrid Score: {result.score:.4f}")
            explanation.append(f"   Dense: {result.dense_score:.4f}")
            explanation.append(f"   Sparse: {result.sparse_score:.4f}")

            # Show content preview
            content_preview = result.content[:80].replace("\n", " ")
            if len(result.content) > 80:
                content_preview += "..."
            explanation.append(f"   Content: {content_preview}")
            explanation.append("")

        return "\n".join(explanation)


# ===== FACTORY FUNCTIONS =====


def create_hybrid_retriever(
    stop_words_provider: StopWordsProvider,
    language: str = "hr",
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> HybridRetriever:
    """
    Factory function to create hybrid retriever.

    Args:
        stop_words_provider: Provider for stop words
        language: Language code
        dense_weight: Weight for dense scores
        sparse_weight: Weight for sparse scores
        bm25_k1: BM25 k1 parameter
        bm25_b: BM25 b parameter

    Returns:
        Configured HybridRetriever instance
    """
    config = HybridConfig(
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )

    return HybridRetriever(stop_words_provider, language, config)


def create_hybrid_retriever_from_config(
    main_config: dict[str, Any],
    stop_words_provider: StopWordsProvider,
    language: str = "hr",
) -> HybridRetriever:
    """
    Factory function to create hybrid retriever from validated configuration.

    Args:
        main_config: Validated main configuration dictionary
        stop_words_provider: Provider for stop words
        language: Language code

    Returns:
        Configured HybridRetriever instance
    """
    # Create validated config from main config
    hybrid_config = HybridRetrievalConfig.from_validated_config(main_config)

    # Convert to internal HybridConfig
    config = HybridConfig.from_validated_config(hybrid_config)

    return HybridRetriever(stop_words_provider, language, config)


def create_mock_stop_words_provider(
    croatian_stop_words: Optional[set] = None, english_stop_words: Optional[set] = None
) -> StopWordsProvider:
    """
    Factory function to create mock stop words provider.

    Args:
        croatian_stop_words: Custom Croatian stop words
        english_stop_words: Custom English stop words

    Returns:
        Mock StopWordsProvider
    """

    class MockStopWordsProvider:
        def __init__(self):
            self.default_croatian = croatian_stop_words or {
                "i",
                "je",
                "su",
                "da",
                "se",
                "na",
                "u",
                "za",
                "od",
                "do",
                "s",
                "sa",
                "kao",
                "ima",
                "biti",
                "bilo",
                "mogu",
                "možete",
                "mogu",
                "ili",
                "ako",
                "kada",
                "gdje",
            }
            self.default_english = english_stop_words or {
                "the",
                "is",
                "are",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "up",
                "about",
                "into",
                "through",
                "during",
            }

        def get_stop_words(self, language: str) -> set:
            if language == "hr":
                return self.default_croatian
            elif language == "en":
                return self.default_english
            else:
                return set()  # Unknown language

    return MockStopWordsProvider()
