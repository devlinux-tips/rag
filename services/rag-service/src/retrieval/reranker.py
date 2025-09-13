"""
Multilingual reranker system for document scoring and ranking.
Clean architecture with dependency injection and pure functions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from ..utils.config_models import ReRankingConfig

logger = logging.getLogger(__name__)


# ===== PURE FUNCTIONS =====


def calculate_rank_changes(
    original_ranks: list[int], new_ranks: list[int]
) -> list[int]:
    """
    Calculate rank change for each item.
    Pure function - no side effects, deterministic output.

    Args:
        original_ranks: Original ranking positions
        new_ranks: New ranking positions

    Returns:
        List of rank changes (positive = moved up, negative = moved down)

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(original_ranks, list):
        raise ValueError(f"Original ranks must be list, got {type(original_ranks)}")

    if not isinstance(new_ranks, list):
        raise ValueError(f"New ranks must be list, got {type(new_ranks)}")

    if len(original_ranks) != len(new_ranks):
        raise ValueError(
            f"Rank lists must have same length: {len(original_ranks)} vs {len(new_ranks)}"
        )

    if not all(isinstance(rank, int) for rank in original_ranks):
        raise ValueError("All original ranks must be integers")

    if not all(isinstance(rank, int) for rank in new_ranks):
        raise ValueError("All new ranks must be integers")

    return [orig - new for orig, new in zip(original_ranks, new_ranks, strict=False)]


def sort_by_scores(
    items: list[Any], scores: list[float], descending: bool = True
) -> tuple[list[Any], list[float], list[int]]:
    """
    Sort items by scores and return sorted items, scores, and original indices.
    Pure function - no side effects, deterministic output.

    Args:
        items: List of items to sort
        scores: List of scores for sorting
        descending: Whether to sort in descending order

    Returns:
        Tuple of (sorted_items, sorted_scores, original_indices)

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(items, list):
        raise ValueError(f"Items must be list, got {type(items)}")

    if not isinstance(scores, list):
        raise ValueError(f"Scores must be list, got {type(scores)}")

    if len(items) != len(scores):
        raise ValueError(
            f"Items and scores must have same length: {len(items)} vs {len(scores)}"
        )

    if not all(isinstance(score, (int, float)) for score in scores):
        raise ValueError("All scores must be numbers")

    # Create indexed pairs and sort
    indexed_pairs = list(enumerate(zip(items, scores, strict=False)))
    indexed_pairs.sort(key=lambda x: x[1][1], reverse=descending)

    # Extract sorted data
    sorted_items = [pair[1][0] for pair in indexed_pairs]
    sorted_scores = [pair[1][1] for pair in indexed_pairs]
    original_indices = [pair[0] for pair in indexed_pairs]

    return sorted_items, sorted_scores, original_indices


def normalize_scores_to_range(
    scores: list[float], min_val: float = 0.0, max_val: float = 1.0
) -> list[float]:
    """
    Normalize scores to specified range.
    Pure function - no side effects, deterministic output.

    Args:
        scores: List of scores to normalize
        min_val: Minimum value in output range
        max_val: Maximum value in output range

    Returns:
        List of normalized scores

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(scores, list):
        raise ValueError(f"Scores must be list, got {type(scores)}")

    if not all(isinstance(score, (int, float)) for score in scores):
        raise ValueError("All scores must be numbers")

    if not isinstance(min_val, (int, float)):
        raise ValueError(f"Min value must be number, got {type(min_val)}")

    if not isinstance(max_val, (int, float)):
        raise ValueError(f"Max value must be number, got {type(max_val)}")

    if min_val >= max_val:
        raise ValueError(
            f"Min value ({min_val}) must be less than max value ({max_val})"
        )

    if not scores:
        return []

    original_min = min(scores)
    original_max = max(scores)

    if original_min == original_max:
        # All scores are equal, return middle of range
        mid_val = (min_val + max_val) / 2
        return [mid_val] * len(scores)

    # Normalize to [0, 1] then scale to desired range
    normalized = [
        (score - original_min) / (original_max - original_min) for score in scores
    ]

    # Scale to desired range
    scaled = [min_val + norm * (max_val - min_val) for norm in normalized]

    return scaled


def calculate_reranking_metrics(
    original_ranks: list[int], new_ranks: list[int]
) -> dict[str, float]:
    """
    Calculate metrics for reranking quality.
    Pure function - no side effects, deterministic output.

    Args:
        original_ranks: Original ranking positions
        new_ranks: New ranking positions

    Returns:
        Dictionary with reranking metrics

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(original_ranks, list) or not isinstance(new_ranks, list):
        raise ValueError("Both rank lists must be lists")

    if len(original_ranks) != len(new_ranks):
        raise ValueError("Rank lists must have same length")

    if not all(isinstance(rank, int) for rank in original_ranks + new_ranks):
        raise ValueError("All ranks must be integers")

    n = len(original_ranks)

    if n == 0:
        return {
            "kendall_tau": 0.0,
            "spearman_rho": 0.0,
            "rank_correlation": 0.0,
            "items_moved": 0,
            "average_rank_change": 0.0,
        }

    # Calculate rank changes
    rank_changes = calculate_rank_changes(original_ranks, new_ranks)

    # Items moved (rank changed)
    items_moved = sum(1 for change in rank_changes if change != 0)

    # Average rank change
    avg_rank_change = sum(abs(change) for change in rank_changes) / n

    # Simple rank correlation (Spearman-like)
    if n > 1:
        # Convert ranks to relative positions for correlation
        orig_positions = [
            original_ranks.index(i) if i in original_ranks else 0 for i in range(n)
        ]
        new_positions = [new_ranks.index(i) if i in new_ranks else 0 for i in range(n)]

        # Simple correlation calculation
        mean_orig = sum(orig_positions) / n
        mean_new = sum(new_positions) / n

        numerator = sum(
            (o - mean_orig) * (n - mean_new)
            for o, n in zip(orig_positions, new_positions, strict=False)
        )

        orig_var = sum((o - mean_orig) ** 2 for o in orig_positions)
        new_var = sum((n - mean_new) ** 2 for n in new_positions)

        if orig_var > 0 and new_var > 0:
            correlation = numerator / (orig_var * new_var) ** 0.5
        else:
            correlation = 0.0
    else:
        correlation = 1.0

    return {
        "kendall_tau": correlation,  # Simplified
        "spearman_rho": correlation,
        "rank_correlation": correlation,
        "items_moved": items_moved,
        "average_rank_change": avg_rank_change,
    }


def create_query_document_pairs(
    query: str, documents: list[str]
) -> list[tuple[str, str]]:
    """
    Create query-document pairs for reranking.
    Pure function - no side effects, deterministic output.

    Args:
        query: Search query
        documents: List of documents

    Returns:
        List of (query, document) pairs

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be string, got {type(query)}")

    if not isinstance(documents, list):
        raise ValueError(f"Documents must be list, got {type(documents)}")

    if not all(isinstance(doc, str) for doc in documents):
        raise ValueError("All documents must be strings")

    return [(query, doc) for doc in documents]


# ===== DATA STRUCTURES =====


# Note: ReRankingConfig is now imported from config_models.py
# Keep local RerankerConfig for internal processing with additional fields
@dataclass
class RerankerConfig:
    """Internal configuration for reranker processing."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    max_length: int = 512
    batch_size: int = 4
    normalize_scores: bool = True
    score_threshold: float = 0.0

    @classmethod
    def from_validated_config(
        cls, reranking_config: ReRankingConfig
    ) -> "RerankerConfig":
        """Create RerankerConfig from validated ReRankingConfig."""
        return cls(
            model_name=reranking_config.model_name,
            max_length=reranking_config.max_length,
            batch_size=reranking_config.batch_size,
            normalize_scores=reranking_config.normalize,
            # Set defaults for fields not in ReRankingConfig
            device="cpu",  # Default device
            score_threshold=0.0,  # Default threshold
        )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.model_name, str):
            raise ValueError("Model name must be string")

        if not isinstance(self.device, str):
            raise ValueError("Device must be string")

        if not isinstance(self.max_length, int) or self.max_length < 1:
            raise ValueError("Max length must be positive integer")

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("Batch size must be positive integer")

        if not isinstance(self.normalize_scores, bool):
            raise ValueError("Normalize scores must be boolean")

        if not isinstance(self.score_threshold, (int, float)):
            raise ValueError("Score threshold must be number")


@dataclass
class RerankingResult:
    """Result from reranking operation."""

    content: str
    score: float
    original_rank: int
    new_rank: int
    rank_change: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and calculate derived fields."""
        if not isinstance(self.content, str):
            raise ValueError("Content must be string")

        if not isinstance(self.score, (int, float)):
            raise ValueError("Score must be number")

        if not isinstance(self.original_rank, int) or self.original_rank < 0:
            raise ValueError("Original rank must be non-negative integer")

        if not isinstance(self.new_rank, int) or self.new_rank < 0:
            raise ValueError("New rank must be non-negative integer")

        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be dict")

        # Calculate rank change if not provided
        if (
            self.rank_change == 0
            and hasattr(self, "original_rank")
            and hasattr(self, "new_rank")
        ):
            self.rank_change = self.original_rank - self.new_rank


@dataclass
class DocumentItem:
    """Document item for reranking."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    original_score: float = 0.0

    def __post_init__(self):
        """Validate document item."""
        if not isinstance(self.content, str):
            raise ValueError("Content must be string")

        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be dict")

        if not isinstance(self.original_score, (int, float)):
            raise ValueError("Original score must be number")


# ===== PROTOCOLS =====


class ModelLoader(Protocol):
    """Protocol for model loading."""

    def load_model(self, model_name: str, device: str) -> Any:
        """Load reranker model."""
        ...

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        ...


class ScoreCalculator(Protocol):
    """Protocol for score calculation."""

    def calculate_scores(
        self, query_document_pairs: list[tuple[str, str]], batch_size: int
    ) -> list[float]:
        """Calculate relevance scores for query-document pairs."""
        ...


# ===== CORE CLASSES =====


class MultilingualReranker:
    """Multilingual document reranker with dependency injection."""

    def __init__(
        self,
        model_loader: ModelLoader,
        score_calculator: ScoreCalculator,
        config: RerankerConfig,
    ):
        """
        Initialize reranker with dependency injection.

        Args:
            model_loader: Model loading interface
            score_calculator: Score calculation interface
            config: Reranker configuration
        """
        self.model_loader = model_loader
        self.score_calculator = score_calculator
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.is_ready = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the reranker."""
        try:
            self.model_loader.load_model(self.config.model_name, self.config.device)
            self.is_ready = self.model_loader.is_model_loaded()

            if self.is_ready:
                self.logger.info(f"Reranker initialized with {self.config.model_name}")
            else:
                self.logger.warning("Reranker model not loaded, using fallback scoring")

        except Exception as e:
            self.logger.error(f"Failed to initialize reranker: {e}")
            self.is_ready = False

    def rerank(
        self, query: str, documents: list[DocumentItem], top_k: int | None = None
    ) -> list[RerankingResult]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return

        Returns:
            List of reranking results

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be string, got {type(query)}")

        if not isinstance(documents, list):
            raise ValueError(f"Documents must be list, got {type(documents)}")

        if not all(isinstance(doc, DocumentItem) for doc in documents):
            raise ValueError("All documents must be DocumentItem instances")

        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("Top k must be positive integer or None")

        if not documents:
            return []

        try:
            # Create query-document pairs
            doc_contents = [doc.content for doc in documents]
            pairs = create_query_document_pairs(query, doc_contents)

            # Calculate relevance scores
            scores = self.score_calculator.calculate_scores(
                query_document_pairs=pairs, batch_size=self.config.batch_size
            )

            # Normalize scores if requested
            if self.config.normalize_scores:
                scores = normalize_scores_to_range(scores, 0.0, 1.0)

            # Sort by scores to get new ranking
            sorted_docs, sorted_scores, original_indices = sort_by_scores(
                items=documents, scores=scores, descending=True
            )

            # Create reranking results
            results = []
            for new_rank, (doc, score, orig_idx) in enumerate(
                zip(sorted_docs, sorted_scores, original_indices, strict=False)
            ):
                # Apply score threshold filter
                if score >= self.config.score_threshold:
                    result = RerankingResult(
                        content=doc.content,
                        score=score,
                        original_rank=orig_idx,
                        new_rank=new_rank,
                        rank_change=orig_idx - new_rank,
                        metadata=doc.metadata.copy(),
                    )
                    results.append(result)

            # Apply top_k limit
            if top_k is not None:
                results = results[:top_k]

            self.logger.debug(
                f"Reranked {len(documents)} documents, returning {len(results)} results"
            )
            return results

        except Exception as e:
            self.logger.error(f"Failed to rerank documents: {e}")
            raise

    def calculate_reranking_quality(
        self, results: list[RerankingResult]
    ) -> dict[str, float]:
        """
        Calculate quality metrics for reranking.

        Args:
            results: List of reranking results

        Returns:
            Dictionary with quality metrics
        """
        if not results:
            return {}

        try:
            original_ranks = [result.original_rank for result in results]
            new_ranks = [result.new_rank for result in results]

            metrics = calculate_reranking_metrics(original_ranks, new_ranks)

            # Add score-based metrics
            scores = [result.score for result in results]
            metrics.update(
                {
                    "mean_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "score_std": np.std(scores) if len(scores) > 1 else 0.0,
                }
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate reranking quality: {e}")
            return {}

    def explain_reranking(self, results: list[RerankingResult]) -> str:
        """
        Generate explanation of reranking results.

        Args:
            results: List of reranking results

        Returns:
            Formatted explanation string
        """
        if not results:
            return "No reranking results to explain."

        explanation = ["🎯 Reranking Explanation:"]
        explanation.append(f"Model: {self.config.model_name}")
        explanation.append(f"Device: {self.config.device}")
        explanation.append(f"Total documents reranked: {len(results)}")
        explanation.append("")

        for i, result in enumerate(results[:10]):  # Show top 10
            rank_change = result.rank_change

            if rank_change > 0:
                change_indicator = f"📈 +{rank_change}"
            elif rank_change < 0:
                change_indicator = f"📉 {rank_change}"
            else:
                change_indicator = "➡️ No change"

            explanation.append(f"{i+1}. Score: {result.score:.4f} {change_indicator}")
            explanation.append(f"    Original rank: {result.original_rank + 1}")

            # Content preview
            content_preview = result.content[:80].replace("\n", " ")
            if len(result.content) > 80:
                content_preview += "..."
            explanation.append(f"    Content: {content_preview}")
            explanation.append("")

        if len(results) > 10:
            explanation.append(f"... and {len(results) - 10} more results")

        return "\n".join(explanation)


# ===== FACTORY FUNCTIONS =====


def create_multilingual_reranker(
    model_loader: ModelLoader,
    score_calculator: ScoreCalculator,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cpu",
    batch_size: int = 4,
) -> MultilingualReranker:
    """
    Factory function to create multilingual reranker.

    Args:
        model_loader: Model loading interface
        score_calculator: Score calculation interface
        model_name: HuggingFace model name
        device: Device to run on
        batch_size: Batch size for processing

    Returns:
        Configured MultilingualReranker instance
    """
    config = RerankerConfig(model_name=model_name, device=device, batch_size=batch_size)

    return MultilingualReranker(model_loader, score_calculator, config)


def create_multilingual_reranker_from_config(
    main_config: dict[str, Any],
    model_loader: ModelLoader,
    score_calculator: ScoreCalculator,
) -> MultilingualReranker:
    """
    Factory function to create multilingual reranker from validated configuration.

    Args:
        main_config: Validated main configuration dictionary
        model_loader: Model loading interface
        score_calculator: Score calculation interface

    Returns:
        Configured MultilingualReranker instance
    """
    # Create validated config from main config
    reranking_config = ReRankingConfig.from_validated_config(main_config)

    # Convert to internal RerankerConfig
    config = RerankerConfig.from_validated_config(reranking_config)

    return MultilingualReranker(model_loader, score_calculator, config)


def create_mock_model_loader(
    should_load_successfully: bool = True, is_loaded: bool = True
) -> ModelLoader:
    """
    Factory function to create mock model loader.

    Args:
        should_load_successfully: Whether load_model should succeed
        is_loaded: Whether model should report as loaded

    Returns:
        Mock ModelLoader
    """

    class MockModelLoader:
        def __init__(self):
            self._is_loaded = False
            self._should_succeed = should_load_successfully
            self._target_loaded_state = is_loaded

        def load_model(self, model_name: str, device: str) -> Any:
            if not self._should_succeed:
                raise ValueError(f"Mock failure loading {model_name}")
            self._is_loaded = self._target_loaded_state
            return "mock_model"

        def is_model_loaded(self) -> bool:
            return self._is_loaded

    return MockModelLoader()


def create_mock_score_calculator(
    base_scores: list[float] | None = None, add_noise: bool = False
) -> ScoreCalculator:
    """
    Factory function to create mock score calculator.

    Args:
        base_scores: Base scores to return (generates if None)
        add_noise: Whether to add random noise to scores

    Returns:
        Mock ScoreCalculator
    """

    class MockScoreCalculator:
        def __init__(self):
            self.base_scores = base_scores
            self.add_noise = add_noise

        def calculate_scores(
            self, query_document_pairs: list[tuple[str, str]], batch_size: int
        ) -> list[float]:
            n_pairs = len(query_document_pairs)

            if self.base_scores:
                # Use provided scores, cycling if necessary
                scores = [
                    self.base_scores[i % len(self.base_scores)] for i in range(n_pairs)
                ]
            else:
                # Generate mock scores based on query-document similarity
                scores = []
                for query, doc in query_document_pairs:
                    # Simple mock scoring based on common words
                    query_words = set(query.lower().split())
                    doc_words = set(doc.lower().split())

                    if query_words and doc_words:
                        overlap = len(query_words & doc_words)
                        total_unique = len(query_words | doc_words)
                        score = overlap / total_unique if total_unique > 0 else 0.0
                    else:
                        score = 0.0

                    scores.append(score)

            # Add noise if requested
            if self.add_noise:
                import random

                scores = [
                    max(0.0, min(1.0, score + random.uniform(-0.1, 0.1)))
                    for score in scores
                ]

            return scores

    return MockScoreCalculator()
