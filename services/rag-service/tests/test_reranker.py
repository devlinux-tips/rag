"""
Comprehensive tests for multilingual reranker system.
Tests pure functions, data structures, and dependency injection patterns.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from dataclasses import dataclass
from typing import Any

from src.retrieval.reranker import (
    # Pure functions
    calculate_rank_changes,
    sort_by_scores,
    normalize_scores_to_range,
    calculate_reranking_metrics,
    create_query_document_pairs,

    # Data structures
    RerankerConfig,
    RerankingResult,
    DocumentItem,

    # Core classes
    MultilingualReranker,

    # Factory functions
    create_multilingual_reranker,
    create_multilingual_reranker_from_config,
    create_mock_model_loader,
    create_mock_score_calculator,
)
from src.utils.config_models import ReRankingConfig


# ===== PURE FUNCTION TESTS =====

class TestCalculateRankChanges:
    """Test calculate_rank_changes pure function."""

    def test_same_length_lists(self):
        """Test with equal length rank lists."""
        original = [0, 1, 2, 3]
        new = [2, 0, 3, 1]
        result = calculate_rank_changes(original, new)
        assert result == [-2, 1, -1, 2]

    def test_empty_lists(self):
        """Test with empty lists."""
        result = calculate_rank_changes([], [])
        assert result == []

    def test_single_item(self):
        """Test with single item."""
        result = calculate_rank_changes([0], [0])
        assert result == [0]

    def test_no_changes(self):
        """Test when ranks don't change."""
        original = [0, 1, 2]
        new = [0, 1, 2]
        result = calculate_rank_changes(original, new)
        assert result == [0, 0, 0]

    def test_invalid_input_types(self):
        """Test with invalid input types."""
        with pytest.raises(ValueError, match="Original ranks must be list"):
            calculate_rank_changes("not_list", [1, 2])

        with pytest.raises(ValueError, match="New ranks must be list"):
            calculate_rank_changes([1, 2], "not_list")

    def test_mismatched_lengths(self):
        """Test with different length lists."""
        with pytest.raises(ValueError, match="Rank lists must have same length"):
            calculate_rank_changes([1, 2], [1, 2, 3])

    def test_non_integer_ranks(self):
        """Test with non-integer ranks."""
        with pytest.raises(ValueError, match="All original ranks must be integers"):
            calculate_rank_changes([1.5, 2], [1, 2])

        with pytest.raises(ValueError, match="All new ranks must be integers"):
            calculate_rank_changes([1, 2], [1.5, 2])


class TestSortByScores:
    """Test sort_by_scores pure function."""

    def test_descending_sort(self):
        """Test descending sort (default)."""
        items = ["doc1", "doc2", "doc3"]
        scores = [0.3, 0.8, 0.5]

        sorted_items, sorted_scores, original_indices = sort_by_scores(items, scores)

        assert sorted_items == ["doc2", "doc3", "doc1"]
        assert sorted_scores == [0.8, 0.5, 0.3]
        assert original_indices == [1, 2, 0]

    def test_ascending_sort(self):
        """Test ascending sort."""
        items = ["doc1", "doc2", "doc3"]
        scores = [0.3, 0.8, 0.5]

        sorted_items, sorted_scores, original_indices = sort_by_scores(items, scores, descending=False)

        assert sorted_items == ["doc1", "doc3", "doc2"]
        assert sorted_scores == [0.3, 0.5, 0.8]
        assert original_indices == [0, 2, 1]

    def test_empty_lists(self):
        """Test with empty lists."""
        sorted_items, sorted_scores, original_indices = sort_by_scores([], [])
        assert sorted_items == []
        assert sorted_scores == []
        assert original_indices == []

    def test_single_item(self):
        """Test with single item."""
        sorted_items, sorted_scores, original_indices = sort_by_scores(["doc1"], [0.5])
        assert sorted_items == ["doc1"]
        assert sorted_scores == [0.5]
        assert original_indices == [0]

    def test_equal_scores(self):
        """Test with equal scores."""
        items = ["doc1", "doc2", "doc3"]
        scores = [0.5, 0.5, 0.5]

        sorted_items, sorted_scores, original_indices = sort_by_scores(items, scores)

        # Should maintain original order for equal scores
        assert len(sorted_items) == 3
        assert all(score == 0.5 for score in sorted_scores)
        assert set(original_indices) == {0, 1, 2}

    def test_invalid_input_types(self):
        """Test with invalid input types."""
        with pytest.raises(ValueError, match="Items must be list"):
            sort_by_scores("not_list", [1, 2])

        with pytest.raises(ValueError, match="Scores must be list"):
            sort_by_scores([1, 2], "not_list")

    def test_mismatched_lengths(self):
        """Test with mismatched lengths."""
        with pytest.raises(ValueError, match="Items and scores must have same length"):
            sort_by_scores(["a", "b"], [1, 2, 3])

    def test_non_numeric_scores(self):
        """Test with non-numeric scores."""
        with pytest.raises(ValueError, match="All scores must be numbers"):
            sort_by_scores(["a", "b"], [1, "not_number"])

    def test_mixed_numeric_types(self):
        """Test with mixed int/float scores."""
        items = ["doc1", "doc2", "doc3"]
        scores = [1, 2.5, 3]

        sorted_items, sorted_scores, original_indices = sort_by_scores(items, scores)

        assert sorted_items == ["doc3", "doc2", "doc1"]
        assert sorted_scores == [3, 2.5, 1]
        assert original_indices == [2, 1, 0]


class TestNormalizeScoresToRange:
    """Test normalize_scores_to_range pure function."""

    def test_normalize_to_zero_one(self):
        """Test normalization to [0, 1]."""
        scores = [1.0, 3.0, 5.0]
        result = normalize_scores_to_range(scores, 0.0, 1.0)
        assert result == [0.0, 0.5, 1.0]

    def test_normalize_to_custom_range(self):
        """Test normalization to custom range."""
        scores = [10, 20, 30]
        result = normalize_scores_to_range(scores, 2.0, 8.0)
        assert result == [2.0, 5.0, 8.0]

    def test_empty_list(self):
        """Test with empty list."""
        result = normalize_scores_to_range([], 0.0, 1.0)
        assert result == []

    def test_single_score(self):
        """Test with single score."""
        result = normalize_scores_to_range([5.0], 0.0, 1.0)
        assert result == [0.5]  # Middle of range

    def test_equal_scores(self):
        """Test with equal scores."""
        scores = [3.0, 3.0, 3.0]
        result = normalize_scores_to_range(scores, 0.0, 1.0)
        assert result == [0.5, 0.5, 0.5]  # All at middle

    def test_already_normalized(self):
        """Test with already normalized scores."""
        scores = [0.0, 0.5, 1.0]
        result = normalize_scores_to_range(scores, 0.0, 1.0)
        assert result == [0.0, 0.5, 1.0]

    def test_negative_scores(self):
        """Test with negative scores."""
        scores = [-2.0, 0.0, 2.0]
        result = normalize_scores_to_range(scores, 0.0, 1.0)
        assert result == [0.0, 0.5, 1.0]

    def test_invalid_input_types(self):
        """Test with invalid input types."""
        with pytest.raises(ValueError, match="Scores must be list"):
            normalize_scores_to_range("not_list", 0.0, 1.0)

        with pytest.raises(ValueError, match="Min value must be number"):
            normalize_scores_to_range([1, 2], "not_number", 1.0)

        with pytest.raises(ValueError, match="Max value must be number"):
            normalize_scores_to_range([1, 2], 0.0, "not_number")

    def test_invalid_range(self):
        """Test with invalid range."""
        with pytest.raises(ValueError, match="Min value .* must be less than max value"):
            normalize_scores_to_range([1, 2], 1.0, 0.0)

        with pytest.raises(ValueError, match="Min value .* must be less than max value"):
            normalize_scores_to_range([1, 2], 1.0, 1.0)

    def test_non_numeric_scores(self):
        """Test with non-numeric scores."""
        with pytest.raises(ValueError, match="All scores must be numbers"):
            normalize_scores_to_range([1, "not_number"], 0.0, 1.0)


class TestCalculateRerankingMetrics:
    """Test calculate_reranking_metrics pure function."""

    def test_no_changes(self):
        """Test when ranks don't change."""
        original = [0, 1, 2, 3]
        new = [0, 1, 2, 3]

        metrics = calculate_reranking_metrics(original, new)

        assert metrics["items_moved"] == 0
        assert metrics["average_rank_change"] == 0.0
        assert "rank_correlation" in metrics
        assert "kendall_tau" in metrics
        assert "spearman_rho" in metrics

    def test_complete_reversal(self):
        """Test complete rank reversal."""
        original = [0, 1, 2, 3]
        new = [3, 2, 1, 0]

        metrics = calculate_reranking_metrics(original, new)

        assert metrics["items_moved"] == 4
        assert metrics["average_rank_change"] == 2.0  # (3+1+1+3)/4 = 2.0

    def test_partial_changes(self):
        """Test partial rank changes."""
        original = [0, 1, 2, 3]
        new = [1, 0, 2, 3]

        metrics = calculate_reranking_metrics(original, new)

        assert metrics["items_moved"] == 2
        assert metrics["average_rank_change"] == 0.5

    def test_empty_lists(self):
        """Test with empty lists."""
        metrics = calculate_reranking_metrics([], [])

        assert metrics["items_moved"] == 0
        assert metrics["average_rank_change"] == 0.0
        assert metrics["kendall_tau"] == 0.0
        assert metrics["spearman_rho"] == 0.0
        assert metrics["rank_correlation"] == 0.0

    def test_single_item(self):
        """Test with single item."""
        metrics = calculate_reranking_metrics([0], [0])

        assert metrics["items_moved"] == 0
        assert metrics["average_rank_change"] == 0.0
        assert metrics["rank_correlation"] == 1.0

    def test_invalid_input_types(self):
        """Test with invalid input types."""
        with pytest.raises(ValueError, match="Both rank lists must be lists"):
            calculate_reranking_metrics("not_list", [1, 2])

        with pytest.raises(ValueError, match="Both rank lists must be lists"):
            calculate_reranking_metrics([1, 2], "not_list")

    def test_mismatched_lengths(self):
        """Test with mismatched lengths."""
        with pytest.raises(ValueError, match="Rank lists must have same length"):
            calculate_reranking_metrics([1, 2], [1, 2, 3])

    def test_non_integer_ranks(self):
        """Test with non-integer ranks."""
        with pytest.raises(ValueError, match="All ranks must be integers"):
            calculate_reranking_metrics([1.5, 2], [1, 2])


class TestCreateQueryDocumentPairs:
    """Test create_query_document_pairs pure function."""

    def test_basic_pairing(self):
        """Test basic query-document pairing."""
        query = "search query"
        documents = ["doc1", "doc2", "doc3"]

        pairs = create_query_document_pairs(query, documents)

        expected = [("search query", "doc1"), ("search query", "doc2"), ("search query", "doc3")]
        assert pairs == expected

    def test_empty_documents(self):
        """Test with empty document list."""
        pairs = create_query_document_pairs("query", [])
        assert pairs == []

    def test_single_document(self):
        """Test with single document."""
        pairs = create_query_document_pairs("query", ["doc1"])
        assert pairs == [("query", "doc1")]

    def test_invalid_query_type(self):
        """Test with invalid query type."""
        with pytest.raises(ValueError, match="Query must be string"):
            create_query_document_pairs(123, ["doc1"])

    def test_invalid_documents_type(self):
        """Test with invalid documents type."""
        with pytest.raises(ValueError, match="Documents must be list"):
            create_query_document_pairs("query", "not_list")

    def test_invalid_document_types(self):
        """Test with non-string documents."""
        with pytest.raises(ValueError, match="All documents must be strings"):
            create_query_document_pairs("query", ["doc1", 123, "doc3"])


# ===== DATA STRUCTURE TESTS =====

class TestRerankerConfig:
    """Test RerankerConfig data class."""

    def test_default_initialization(self):
        """Test default configuration."""
        config = RerankerConfig()

        assert config.model_name == "BAAI/bge-reranker-v2-m3"
        assert config.device == "cpu"
        assert config.max_length == 512
        assert config.batch_size == 4
        assert config.normalize_scores is True
        assert config.score_threshold == 0.0

    def test_custom_initialization(self):
        """Test custom configuration."""
        config = RerankerConfig(
            model_name="custom-model",
            device="cuda",
            max_length=256,
            batch_size=8,
            normalize_scores=False,
            score_threshold=0.5
        )

        assert config.model_name == "custom-model"
        assert config.device == "cuda"
        assert config.max_length == 256
        assert config.batch_size == 8
        assert config.normalize_scores is False
        assert config.score_threshold == 0.5

    def test_from_validated_config(self):
        """Test creation from ReRankingConfig."""
        main_config = {
            "reranking": {
                "enabled": True,
                "model_name": "test-model",
                "max_length": 256,
                "batch_size": 8,
                "top_k": 10,
                "use_fp16": False,
                "normalize": True
            }
        }

        reranking_config = ReRankingConfig.from_validated_config(main_config)
        config = RerankerConfig.from_validated_config(reranking_config)

        assert config.model_name == "test-model"
        assert config.max_length == 256
        assert config.batch_size == 8
        assert config.normalize_scores is True
        assert config.device == "cpu"  # Default
        assert config.score_threshold == 0.0  # Default

    def test_validation_errors(self):
        """Test validation errors."""
        with pytest.raises(ValueError, match="Model name must be string"):
            RerankerConfig(model_name=123)

        with pytest.raises(ValueError, match="Device must be string"):
            RerankerConfig(device=123)

        with pytest.raises(ValueError, match="Max length must be positive integer"):
            RerankerConfig(max_length=0)

        with pytest.raises(ValueError, match="Max length must be positive integer"):
            RerankerConfig(max_length="not_int")

        with pytest.raises(ValueError, match="Batch size must be positive integer"):
            RerankerConfig(batch_size=0)

        with pytest.raises(ValueError, match="Normalize scores must be boolean"):
            RerankerConfig(normalize_scores="not_bool")

        with pytest.raises(ValueError, match="Score threshold must be number"):
            RerankerConfig(score_threshold="not_number")


class TestRerankingResult:
    """Test RerankingResult data class."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        result = RerankingResult(
            content="document content",
            score=0.8,
            original_rank=2,
            new_rank=0
        )

        assert result.content == "document content"
        assert result.score == 0.8
        assert result.original_rank == 2
        assert result.new_rank == 0
        assert result.rank_change == 2  # Calculated automatically
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test with metadata."""
        metadata = {"source": "test", "category": "doc"}
        result = RerankingResult(
            content="content",
            score=0.5,
            original_rank=1,
            new_rank=1,
            metadata=metadata
        )

        assert result.metadata == metadata
        assert result.rank_change == 0

    def test_explicit_rank_change(self):
        """Test with explicit rank change."""
        result = RerankingResult(
            content="content",
            score=0.5,
            original_rank=3,
            new_rank=1,
            rank_change=5  # Explicit value should be kept
        )

        assert result.rank_change == 5  # Should NOT be recalculated

    def test_validation_errors(self):
        """Test validation errors."""
        with pytest.raises(ValueError, match="Content must be string"):
            RerankingResult(content=123, score=0.5, original_rank=0, new_rank=0)

        with pytest.raises(ValueError, match="Score must be number"):
            RerankingResult(content="content", score="not_number", original_rank=0, new_rank=0)

        with pytest.raises(ValueError, match="Original rank must be non-negative integer"):
            RerankingResult(content="content", score=0.5, original_rank=-1, new_rank=0)

        with pytest.raises(ValueError, match="Original rank must be non-negative integer"):
            RerankingResult(content="content", score=0.5, original_rank="not_int", new_rank=0)

        with pytest.raises(ValueError, match="New rank must be non-negative integer"):
            RerankingResult(content="content", score=0.5, original_rank=0, new_rank=-1)

        with pytest.raises(ValueError, match="Metadata must be dict"):
            RerankingResult(content="content", score=0.5, original_rank=0, new_rank=0, metadata="not_dict")


class TestDocumentItem:
    """Test DocumentItem data class."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        doc = DocumentItem(content="document content")

        assert doc.content == "document content"
        assert doc.metadata == {}
        assert doc.original_score == 0.0

    def test_with_metadata_and_score(self):
        """Test with metadata and score."""
        metadata = {"source": "test"}
        doc = DocumentItem(
            content="content",
            metadata=metadata,
            original_score=0.7
        )

        assert doc.content == "content"
        assert doc.metadata == metadata
        assert doc.original_score == 0.7

    def test_validation_errors(self):
        """Test validation errors."""
        with pytest.raises(ValueError, match="Content must be string"):
            DocumentItem(content=123)

        with pytest.raises(ValueError, match="Metadata must be dict"):
            DocumentItem(content="content", metadata="not_dict")

        with pytest.raises(ValueError, match="Original score must be number"):
            DocumentItem(content="content", original_score="not_number")


# ===== MOCK FACTORY TESTS =====

class TestMockFactories:
    """Test mock factory functions."""

    def test_create_mock_model_loader_success(self):
        """Test successful mock model loader."""
        loader = create_mock_model_loader(should_load_successfully=True, is_loaded=True)

        model = loader.load_model("test-model", "cpu")
        assert model == "mock_model"
        assert loader.is_model_loaded() is True

    def test_create_mock_model_loader_failure(self):
        """Test failing mock model loader."""
        loader = create_mock_model_loader(should_load_successfully=False, is_loaded=False)

        with pytest.raises(ValueError, match="Mock failure loading test-model"):
            loader.load_model("test-model", "cpu")

        assert loader.is_model_loaded() is False

    def test_create_mock_model_loader_load_but_not_ready(self):
        """Test loader that loads but reports not ready."""
        loader = create_mock_model_loader(should_load_successfully=True, is_loaded=False)

        loader.load_model("test-model", "cpu")
        assert loader.is_model_loaded() is False

    def test_create_mock_score_calculator_with_base_scores(self):
        """Test mock score calculator with base scores."""
        base_scores = [0.8, 0.6, 0.4]
        calculator = create_mock_score_calculator(base_scores=base_scores)

        pairs = [("query", "doc1"), ("query", "doc2"), ("query", "doc3")]
        scores = calculator.calculate_scores(pairs, batch_size=2)

        assert scores == [0.8, 0.6, 0.4]

    def test_create_mock_score_calculator_cycling_scores(self):
        """Test score cycling for more pairs than base scores."""
        base_scores = [0.8, 0.6]
        calculator = create_mock_score_calculator(base_scores=base_scores)

        pairs = [("query", "doc1"), ("query", "doc2"), ("query", "doc3"), ("query", "doc4")]
        scores = calculator.calculate_scores(pairs, batch_size=2)

        assert scores == [0.8, 0.6, 0.8, 0.6]

    def test_create_mock_score_calculator_generated_scores(self):
        """Test generated scores based on word overlap."""
        calculator = create_mock_score_calculator()

        pairs = [
            ("python programming", "python code example"),
            ("java development", "javascript tutorial"),
            ("machine learning", "deep learning models")
        ]
        scores = calculator.calculate_scores(pairs, batch_size=2)

        assert len(scores) == 3
        assert all(0.0 <= score <= 1.0 for score in scores)
        # First pair should have highest score (python overlap)
        assert scores[0] > 0

    def test_create_mock_score_calculator_with_noise(self):
        """Test score calculator with noise."""
        base_scores = [0.5, 0.5, 0.5]
        calculator = create_mock_score_calculator(base_scores=base_scores, add_noise=True)

        pairs = [("query", "doc1"), ("query", "doc2"), ("query", "doc3")]
        scores = calculator.calculate_scores(pairs, batch_size=2)

        assert len(scores) == 3
        assert all(0.0 <= score <= 1.0 for score in scores)
        # With noise, scores should vary from base
        assert not all(score == 0.5 for score in scores)


# ===== CORE CLASS TESTS =====

class TestMultilingualReranker:
    """Test MultilingualReranker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RerankerConfig(batch_size=2)
        self.mock_loader = create_mock_model_loader()
        # Use non-normalized scores that will be normalized by the reranker
        self.mock_calculator = create_mock_score_calculator(base_scores=[9, 7, 5, 3])

    def test_initialization_success(self):
        """Test successful initialization."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        assert reranker.is_ready is True
        assert reranker.config == self.config

    def test_initialization_failure(self):
        """Test initialization failure."""
        failing_loader = create_mock_model_loader(should_load_successfully=False)

        reranker = MultilingualReranker(failing_loader, self.mock_calculator, self.config)

        assert reranker.is_ready is False

    def test_basic_reranking(self):
        """Test basic reranking functionality."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        documents = [
            DocumentItem("first document", metadata={"id": "1"}),
            DocumentItem("second document", metadata={"id": "2"}),
            DocumentItem("third document", metadata={"id": "3"}),
            DocumentItem("fourth document", metadata={"id": "4"})
        ]

        results = reranker.rerank("test query", documents)

        assert len(results) == 4
        # Should be sorted by score (descending) and normalized to [0,1]
        assert results[0].score == 1.0  # 9 -> 1.0 (max)
        assert results[1].score == 2/3  # 7 -> 0.667 (normalized)
        assert results[2].score == 1/3  # 5 -> 0.333 (normalized)
        assert results[3].score == 0.0  # 3 -> 0.0 (min)

        # Check rank changes
        assert results[0].original_rank == 0
        assert results[0].new_rank == 0
        assert results[0].rank_change == 0

    def test_reranking_with_top_k(self):
        """Test reranking with top_k limit."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        documents = [
            DocumentItem("first document"),
            DocumentItem("second document"),
            DocumentItem("third document"),
            DocumentItem("fourth document")
        ]

        results = reranker.rerank("test query", documents, top_k=2)

        assert len(results) == 2
        assert results[0].score == 1.0  # Normalized max
        assert results[1].score == 2/3  # Normalized second

    def test_reranking_with_score_threshold(self):
        """Test reranking with score threshold."""
        config = RerankerConfig(score_threshold=0.6)
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, config)

        documents = [
            DocumentItem("first document"),
            DocumentItem("second document"),
            DocumentItem("third document"),
            DocumentItem("fourth document")
        ]

        results = reranker.rerank("test query", documents)

        # Only scores >= 0.6 should be included (after normalization)
        # 9->1.0, 7->0.667, 5->0.333, 3->0.0
        assert len(results) == 2  # Only 1.0 and 0.667 are >= 0.6
        assert all(result.score >= 0.6 for result in results)

    def test_reranking_with_normalization(self):
        """Test reranking with score normalization."""
        config = RerankerConfig(normalize_scores=True)
        calculator = create_mock_score_calculator(base_scores=[10, 20, 30])
        reranker = MultilingualReranker(self.mock_loader, calculator, config)

        documents = [
            DocumentItem("first document"),
            DocumentItem("second document"),
            DocumentItem("third document")
        ]

        results = reranker.rerank("test query", documents)

        # Scores should be normalized to [0, 1]
        assert len(results) == 3
        assert results[0].score == 1.0  # Max normalized
        assert results[1].score == 0.5  # Middle
        assert results[2].score == 0.0  # Min normalized

    def test_empty_documents(self):
        """Test with empty document list."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        results = reranker.rerank("test query", [])

        assert results == []

    def test_invalid_inputs(self):
        """Test with invalid inputs."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        with pytest.raises(ValueError, match="Query must be string"):
            reranker.rerank(123, [])

        with pytest.raises(ValueError, match="Documents must be list"):
            reranker.rerank("query", "not_list")

        with pytest.raises(ValueError, match="All documents must be DocumentItem instances"):
            reranker.rerank("query", ["not_document_item"])

        with pytest.raises(ValueError, match="Top k must be positive integer or None"):
            reranker.rerank("query", [], top_k=0)

    def test_calculate_reranking_quality(self):
        """Test reranking quality calculation."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        results = [
            RerankingResult("doc1", 0.9, 2, 0, rank_change=2),
            RerankingResult("doc2", 0.7, 1, 1, rank_change=0),
            RerankingResult("doc3", 0.5, 0, 2, rank_change=-2)
        ]

        metrics = reranker.calculate_reranking_quality(results)

        assert "items_moved" in metrics
        assert "average_rank_change" in metrics
        assert "mean_score" in metrics
        assert "min_score" in metrics
        assert "max_score" in metrics
        assert "score_std" in metrics

        assert abs(metrics["mean_score"] - 0.7) < 0.001  # Handle floating point precision
        assert metrics["min_score"] == 0.5
        assert metrics["max_score"] == 0.9

    def test_calculate_reranking_quality_empty(self):
        """Test quality calculation with empty results."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        metrics = reranker.calculate_reranking_quality([])

        assert metrics == {}

    def test_explain_reranking(self):
        """Test reranking explanation."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        results = [
            RerankingResult("First document content", 0.9, 2, 0, rank_change=2),
            RerankingResult("Second document", 0.7, 1, 1, rank_change=0),
            RerankingResult("Third document", 0.5, 0, 2, rank_change=-2)
        ]

        explanation = reranker.explain_reranking(results)

        assert "🎯 Reranking Explanation:" in explanation
        assert "Model: BAAI/bge-reranker-v2-m3" in explanation
        assert "Device: cpu" in explanation
        assert "Total documents reranked: 3" in explanation
        assert "Score: 0.9000" in explanation
        assert "📈 +2" in explanation  # Rank improvement
        assert "➡️ No change" in explanation  # No rank change
        assert "📉 -2" in explanation  # Rank decline

    def test_explain_reranking_empty(self):
        """Test explanation with empty results."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        explanation = reranker.explain_reranking([])

        assert explanation == "No reranking results to explain."

    def test_explain_reranking_long_content(self):
        """Test explanation with long content."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        long_content = "This is a very long document content that should be truncated in the explanation because it exceeds the maximum length"
        results = [
            RerankingResult(long_content, 0.9, 0, 0, rank_change=0)
        ]

        explanation = reranker.explain_reranking(results)

        # Check for truncation marker - exact text may vary
        assert "..." in explanation
        assert "This is a very long document content" in explanation

    def test_reranking_preserves_metadata(self):
        """Test that reranking preserves document metadata."""
        reranker = MultilingualReranker(self.mock_loader, self.mock_calculator, self.config)

        documents = [
            DocumentItem("first", metadata={"source": "A", "category": "1"}),
            DocumentItem("second", metadata={"source": "B", "category": "2"})
        ]

        results = reranker.rerank("query", documents)

        assert len(results) == 2
        assert results[0].metadata == {"source": "A", "category": "1"}
        assert results[1].metadata == {"source": "B", "category": "2"}


# ===== FACTORY FUNCTION TESTS =====

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_multilingual_reranker(self):
        """Test basic factory function."""
        loader = create_mock_model_loader()
        calculator = create_mock_score_calculator()

        reranker = create_multilingual_reranker(
            model_loader=loader,
            score_calculator=calculator,
            model_name="custom-model",
            device="cuda",
            batch_size=8
        )

        assert isinstance(reranker, MultilingualReranker)
        assert reranker.config.model_name == "custom-model"
        assert reranker.config.device == "cuda"
        assert reranker.config.batch_size == 8

    def test_create_multilingual_reranker_from_config(self):
        """Test factory function from config."""
        main_config = {
            "reranking": {
                "enabled": True,
                "model_name": "config-model",
                "max_length": 256,
                "batch_size": 4,
                "top_k": 10,
                "use_fp16": False,
                "normalize": True
            }
        }

        loader = create_mock_model_loader()
        calculator = create_mock_score_calculator()

        reranker = create_multilingual_reranker_from_config(
            main_config=main_config,
            model_loader=loader,
            score_calculator=calculator
        )

        assert isinstance(reranker, MultilingualReranker)
        assert reranker.config.model_name == "config-model"
        assert reranker.config.max_length == 256
        assert reranker.config.batch_size == 4
        assert reranker.config.normalize_scores is True


# ===== INTEGRATION TESTS =====

class TestIntegration:
    """Integration tests for complete reranking workflow."""

    def test_end_to_end_reranking_workflow(self):
        """Test complete reranking workflow."""
        # Create realistic mock calculator
        calculator = create_mock_score_calculator()
        loader = create_mock_model_loader()
        config = RerankerConfig(normalize_scores=True, score_threshold=0.1)

        reranker = MultilingualReranker(loader, calculator, config)

        # Create documents with varying relevance
        documents = [
            DocumentItem("python programming tutorial", metadata={"type": "tutorial"}),
            DocumentItem("java development guide", metadata={"type": "guide"}),
            DocumentItem("python code examples", metadata={"type": "examples"}),
            DocumentItem("machine learning basics", metadata={"type": "basics"})
        ]

        # Query should match python documents better
        query = "python programming"
        results = reranker.rerank(query, documents, top_k=3)

        assert len(results) <= 3
        assert all(isinstance(result, RerankingResult) for result in results)
        assert all(result.score >= 0.1 for result in results)  # Above threshold

        # Calculate quality metrics
        quality = reranker.calculate_reranking_quality(results)
        assert isinstance(quality, dict)
        assert "mean_score" in quality

        # Generate explanation
        explanation = reranker.explain_reranking(results)
        assert isinstance(explanation, str)
        assert "🎯 Reranking Explanation:" in explanation

    def test_error_recovery_in_scoring(self):
        """Test error recovery during scoring."""
        # Create a calculator that raises an exception
        class FailingCalculator:
            def calculate_scores(self, query_document_pairs, batch_size):
                raise RuntimeError("Scoring failed")

        loader = create_mock_model_loader()
        calculator = FailingCalculator()
        config = RerankerConfig()

        reranker = MultilingualReranker(loader, calculator, config)
        documents = [DocumentItem("test document")]

        # Should re-raise the exception
        with pytest.raises(RuntimeError, match="Scoring failed"):
            reranker.rerank("test query", documents)

    def test_large_document_set_performance(self):
        """Test performance with larger document set."""
        calculator = create_mock_score_calculator()
        loader = create_mock_model_loader()
        config = RerankerConfig(batch_size=10)

        reranker = MultilingualReranker(loader, calculator, config)

        # Create 50 documents
        documents = [DocumentItem(f"Document {i} content") for i in range(50)]

        results = reranker.rerank("test query", documents, top_k=10)

        assert len(results) == 10
        assert all(isinstance(result, RerankingResult) for result in results)

        # Verify ranking is maintained
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)  # Descending order