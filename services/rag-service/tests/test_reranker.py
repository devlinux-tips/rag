"""
Comprehensive tests for reranker.py demonstrating 100% testability.
Tests pure functions, dependency injection, and integration scenarios.
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from src.retrieval.reranker import (  # Pure functions; Data structures; Core classes; Factory functions
    DocumentItem, MultilingualReranker, RerankerConfig, RerankingResult,
    calculate_rank_changes, calculate_reranking_metrics,
    create_mock_model_loader, create_mock_score_calculator,
    create_multilingual_reranker, create_query_document_pairs,
    normalize_scores_to_range, sort_by_scores)


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_calculate_rank_changes_basic(self):
        """Test basic rank change calculation."""
        original_ranks = [0, 1, 2, 3]
        new_ranks = [2, 0, 3, 1]

        changes = calculate_rank_changes(original_ranks, new_ranks)

        assert changes == [
            -2,
            1,
            -1,
            2,
        ]  # [moved down 2, moved up 1, moved down 1, moved up 2]

    def test_calculate_rank_changes_no_change(self):
        """Test rank changes when no reranking occurred."""
        original_ranks = [0, 1, 2, 3]
        new_ranks = [0, 1, 2, 3]

        changes = calculate_rank_changes(original_ranks, new_ranks)

        assert changes == [0, 0, 0, 0]

    def test_calculate_rank_changes_reverse_order(self):
        """Test rank changes with complete reversal."""
        original_ranks = [0, 1, 2]
        new_ranks = [2, 1, 0]

        changes = calculate_rank_changes(original_ranks, new_ranks)

        assert changes == [
            -2,
            0,
            2,
        ]  # First moved down 2, middle unchanged, last moved up 2

    def test_calculate_rank_changes_invalid_inputs(self):
        """Test rank change calculation with invalid inputs."""
        with pytest.raises(ValueError, match="Original ranks must be list"):
            calculate_rank_changes("not list", [1, 2, 3])

        with pytest.raises(ValueError, match="New ranks must be list"):
            calculate_rank_changes([1, 2, 3], "not list")

        with pytest.raises(ValueError, match="Rank lists must have same length"):
            calculate_rank_changes([1, 2, 3], [1, 2])

        with pytest.raises(ValueError, match="All original ranks must be integers"):
            calculate_rank_changes([1, 2.5, 3], [1, 2, 3])

        with pytest.raises(ValueError, match="All new ranks must be integers"):
            calculate_rank_changes([1, 2, 3], [1, "2", 3])

    def test_sort_by_scores_descending(self):
        """Test sorting items by scores in descending order."""
        items = ["doc_a", "doc_b", "doc_c", "doc_d"]
        scores = [0.3, 0.8, 0.1, 0.6]

        sorted_items, sorted_scores, orig_indices = sort_by_scores(
            items, scores, descending=True
        )

        assert sorted_items == ["doc_b", "doc_d", "doc_a", "doc_c"]
        assert sorted_scores == [0.8, 0.6, 0.3, 0.1]
        assert orig_indices == [1, 3, 0, 2]

    def test_sort_by_scores_ascending(self):
        """Test sorting items by scores in ascending order."""
        items = ["doc_a", "doc_b", "doc_c"]
        scores = [0.7, 0.2, 0.9]

        sorted_items, sorted_scores, orig_indices = sort_by_scores(
            items, scores, descending=False
        )

        assert sorted_items == ["doc_b", "doc_a", "doc_c"]
        assert sorted_scores == [0.2, 0.7, 0.9]
        assert orig_indices == [1, 0, 2]

    def test_sort_by_scores_equal_scores(self):
        """Test sorting with equal scores (should maintain relative order)."""
        items = ["doc_a", "doc_b", "doc_c"]
        scores = [0.5, 0.5, 0.5]

        sorted_items, sorted_scores, orig_indices = sort_by_scores(
            items, scores, descending=True
        )

        assert len(sorted_items) == 3
        assert all(score == 0.5 for score in sorted_scores)
        assert set(orig_indices) == {0, 1, 2}

    def test_sort_by_scores_empty_lists(self):
        """Test sorting empty lists."""
        sorted_items, sorted_scores, orig_indices = sort_by_scores(
            [], [], descending=True
        )

        assert sorted_items == []
        assert sorted_scores == []
        assert orig_indices == []

    def test_sort_by_scores_invalid_inputs(self):
        """Test sorting with invalid inputs."""
        with pytest.raises(ValueError, match="Items must be list"):
            sort_by_scores("not list", [0.5])

        with pytest.raises(ValueError, match="Scores must be list"):
            sort_by_scores(["item"], "not list")

        with pytest.raises(ValueError, match="Items and scores must have same length"):
            sort_by_scores(["item1", "item2"], [0.5])

        with pytest.raises(ValueError, match="All scores must be numbers"):
            sort_by_scores(["item"], ["not number"])

    def test_normalize_scores_to_range_default(self):
        """Test score normalization to default [0, 1] range."""
        scores = [10.0, 20.0, 5.0, 30.0]

        normalized = normalize_scores_to_range(scores)

        assert abs(normalized[0] - 0.2) < 0.01  # (10-5)/(30-5) = 0.2
        assert abs(normalized[1] - 0.6) < 0.01  # (20-5)/(30-5) = 0.6
        assert abs(normalized[2] - 0.0) < 0.01  # (5-5)/(30-5) = 0.0
        assert abs(normalized[3] - 1.0) < 0.01  # (30-5)/(30-5) = 1.0

    def test_normalize_scores_to_range_custom(self):
        """Test score normalization to custom range."""
        scores = [0.0, 0.5, 1.0]

        normalized = normalize_scores_to_range(scores, min_val=-1.0, max_val=1.0)

        assert abs(normalized[0] - (-1.0)) < 0.01
        assert abs(normalized[1] - 0.0) < 0.01
        assert abs(normalized[2] - 1.0) < 0.01

    def test_normalize_scores_to_range_equal_scores(self):
        """Test normalization with equal scores."""
        scores = [0.7, 0.7, 0.7]

        normalized = normalize_scores_to_range(scores, min_val=0.0, max_val=1.0)

        assert all(
            abs(score - 0.5) < 0.01 for score in normalized
        )  # Should be middle of range

    def test_normalize_scores_to_range_empty_list(self):
        """Test normalization with empty list."""
        normalized = normalize_scores_to_range([])
        assert normalized == []

    def test_normalize_scores_to_range_invalid_inputs(self):
        """Test score normalization with invalid inputs."""
        with pytest.raises(ValueError, match="Scores must be list"):
            normalize_scores_to_range("not list")

        with pytest.raises(ValueError, match="All scores must be numbers"):
            normalize_scores_to_range([1.0, "not number", 3.0])

        with pytest.raises(ValueError, match="Min value must be number"):
            normalize_scores_to_range([1.0], min_val="not number")

        with pytest.raises(ValueError, match="Max value must be number"):
            normalize_scores_to_range([1.0], max_val="not number")

        with pytest.raises(
            ValueError, match="Min value .* must be less than max value"
        ):
            normalize_scores_to_range([1.0], min_val=1.0, max_val=0.5)

    def test_calculate_reranking_metrics_basic(self):
        """Test basic reranking metrics calculation."""
        original_ranks = [0, 1, 2, 3]
        new_ranks = [1, 0, 3, 2]

        metrics = calculate_reranking_metrics(original_ranks, new_ranks)

        assert "items_moved" in metrics
        assert "average_rank_change" in metrics
        assert "rank_correlation" in metrics
        assert metrics["items_moved"] == 4  # All items moved
        assert metrics["average_rank_change"] == 1.0  # Average absolute change

    def test_calculate_reranking_metrics_no_change(self):
        """Test metrics when no reranking occurred."""
        original_ranks = [0, 1, 2]
        new_ranks = [0, 1, 2]

        metrics = calculate_reranking_metrics(original_ranks, new_ranks)

        assert metrics["items_moved"] == 0
        assert metrics["average_rank_change"] == 0.0
        assert metrics["rank_correlation"] == 1.0  # Perfect correlation

    def test_calculate_reranking_metrics_empty_lists(self):
        """Test metrics with empty lists."""
        metrics = calculate_reranking_metrics([], [])

        assert metrics["items_moved"] == 0
        assert metrics["average_rank_change"] == 0.0
        assert metrics["rank_correlation"] == 0.0

    def test_calculate_reranking_metrics_invalid_inputs(self):
        """Test metrics calculation with invalid inputs."""
        with pytest.raises(ValueError, match="Both rank lists must be lists"):
            calculate_reranking_metrics("not list", [1, 2])

        with pytest.raises(ValueError, match="Rank lists must have same length"):
            calculate_reranking_metrics([1, 2], [1])

        with pytest.raises(ValueError, match="All ranks must be integers"):
            calculate_reranking_metrics([1, 2.5], [1, 2])

    def test_create_query_document_pairs_basic(self):
        """Test creating query-document pairs."""
        query = "What is machine learning?"
        documents = [
            "ML is AI subset",
            "Deep learning uses neural networks",
            "Python is programming language",
        ]

        pairs = create_query_document_pairs(query, documents)

        assert len(pairs) == 3
        assert all(pair[0] == query for pair in pairs)
        assert pairs[0][1] == "ML is AI subset"
        assert pairs[1][1] == "Deep learning uses neural networks"
        assert pairs[2][1] == "Python is programming language"

    def test_create_query_document_pairs_empty_documents(self):
        """Test creating pairs with empty document list."""
        pairs = create_query_document_pairs("query", [])
        assert pairs == []

    def test_create_query_document_pairs_invalid_inputs(self):
        """Test pair creation with invalid inputs."""
        with pytest.raises(ValueError, match="Query must be string"):
            create_query_document_pairs(123, ["doc"])

        with pytest.raises(ValueError, match="Documents must be list"):
            create_query_document_pairs("query", "not list")

        with pytest.raises(ValueError, match="All documents must be strings"):
            create_query_document_pairs("query", ["valid", 123, "also valid"])


class TestDataStructures:
    """Test data structure classes."""

    def test_reranker_config_creation(self):
        """Test creating RerankerConfig."""
        config = RerankerConfig(
            model_name="custom/model",
            device="cuda",
            max_length=1024,
            batch_size=8,
            normalize_scores=False,
            score_threshold=0.5,
        )

        assert config.model_name == "custom/model"
        assert config.device == "cuda"
        assert config.max_length == 1024
        assert config.batch_size == 8
        assert config.normalize_scores is False
        assert config.score_threshold == 0.5

    def test_reranker_config_defaults(self):
        """Test RerankerConfig with default values."""
        config = RerankerConfig()

        assert config.model_name == "BAAI/bge-reranker-v2-m3"
        assert config.device == "cpu"
        assert config.max_length == 512
        assert config.batch_size == 4
        assert config.normalize_scores is True
        assert config.score_threshold == 0.0

    def test_reranker_config_validation(self):
        """Test RerankerConfig validation."""
        with pytest.raises(ValueError, match="Model name must be string"):
            RerankerConfig(model_name=123)

        with pytest.raises(ValueError, match="Device must be string"):
            RerankerConfig(device=123)

        with pytest.raises(ValueError, match="Max length must be positive integer"):
            RerankerConfig(max_length=0)

        with pytest.raises(ValueError, match="Batch size must be positive integer"):
            RerankerConfig(batch_size=-1)

        with pytest.raises(ValueError, match="Normalize scores must be boolean"):
            RerankerConfig(normalize_scores="true")

        with pytest.raises(ValueError, match="Score threshold must be number"):
            RerankerConfig(score_threshold="0.5")

    def test_reranking_result_creation(self):
        """Test creating RerankingResult."""
        result = RerankingResult(
            content="Test document content",
            score=0.85,
            original_rank=2,
            new_rank=0,
            rank_change=2,
            metadata={"source": "test.txt"},
        )

        assert result.content == "Test document content"
        assert result.score == 0.85
        assert result.original_rank == 2
        assert result.new_rank == 0
        assert result.rank_change == 2
        assert result.metadata == {"source": "test.txt"}

    def test_reranking_result_auto_rank_change(self):
        """Test automatic rank change calculation."""
        result = RerankingResult(
            content="Test content", score=0.9, original_rank=3, new_rank=1
        )

        assert result.rank_change == 2  # 3 - 1 = 2 (moved up)

    def test_reranking_result_defaults(self):
        """Test RerankingResult with default values."""
        result = RerankingResult(
            content="Test content", score=0.8, original_rank=1, new_rank=0
        )

        assert result.rank_change == 1
        assert result.metadata == {}

    def test_reranking_result_validation(self):
        """Test RerankingResult validation."""
        with pytest.raises(ValueError, match="Content must be string"):
            RerankingResult(content=123, score=0.8, original_rank=1, new_rank=0)

        with pytest.raises(ValueError, match="Score must be number"):
            RerankingResult(content="test", score="high", original_rank=1, new_rank=0)

        with pytest.raises(
            ValueError, match="Original rank must be non-negative integer"
        ):
            RerankingResult(content="test", score=0.8, original_rank=-1, new_rank=0)

        with pytest.raises(ValueError, match="New rank must be non-negative integer"):
            RerankingResult(content="test", score=0.8, original_rank=1, new_rank=-1)

        with pytest.raises(ValueError, match="Metadata must be dict"):
            RerankingResult(
                content="test", score=0.8, original_rank=1, new_rank=0, metadata="meta"
            )

    def test_document_item_creation(self):
        """Test creating DocumentItem."""
        doc = DocumentItem(
            content="Document content here",
            metadata={"type": "article", "length": 100},
            original_score=0.75,
        )

        assert doc.content == "Document content here"
        assert doc.metadata == {"type": "article", "length": 100}
        assert doc.original_score == 0.75

    def test_document_item_defaults(self):
        """Test DocumentItem with default values."""
        doc = DocumentItem(content="Test content")

        assert doc.content == "Test content"
        assert doc.metadata == {}
        assert doc.original_score == 0.0

    def test_document_item_validation(self):
        """Test DocumentItem validation."""
        with pytest.raises(ValueError, match="Content must be string"):
            DocumentItem(content=123)

        with pytest.raises(ValueError, match="Metadata must be dict"):
            DocumentItem(content="test", metadata="meta")

        with pytest.raises(ValueError, match="Original score must be number"):
            DocumentItem(content="test", original_score="high")


class TestMultilingualReranker:
    """Test MultilingualReranker with dependency injection."""

    @pytest.fixture
    def mock_model_loader(self):
        """Create mock model loader."""
        return create_mock_model_loader(should_load_successfully=True, is_loaded=True)

    @pytest.fixture
    def mock_score_calculator(self):
        """Create mock score calculator."""
        return create_mock_score_calculator()

    @pytest.fixture
    def reranker_config(self):
        """Create test reranker config."""
        return RerankerConfig(batch_size=2, normalize_scores=True)

    @pytest.fixture
    def reranker(self, mock_model_loader, mock_score_calculator, reranker_config):
        """Create multilingual reranker."""
        return MultilingualReranker(
            mock_model_loader, mock_score_calculator, reranker_config
        )

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            DocumentItem(
                "Prvi dokument o programiranju u Pythonu", {"source": "doc1.txt"}, 0.8
            ),
            DocumentItem(
                "Drugi dokument govori o umjetnoj inteligenciji",
                {"source": "doc2.txt"},
                0.7,
            ),
            DocumentItem(
                "Treći dokument sadrži informacije o bazama podataka",
                {"source": "doc3.txt"},
                0.6,
            ),
            DocumentItem(
                "Četvrti dokument opisuje web razvoj tehnologije",
                {"source": "doc4.txt"},
                0.5,
            ),
        ]

    def test_reranker_initialization_success(
        self, mock_model_loader, mock_score_calculator, reranker_config
    ):
        """Test successful reranker initialization."""
        reranker = MultilingualReranker(
            mock_model_loader, mock_score_calculator, reranker_config
        )

        assert reranker.model_loader == mock_model_loader
        assert reranker.score_calculator == mock_score_calculator
        assert reranker.config == reranker_config
        assert reranker.is_ready is True

    def test_reranker_initialization_failure(
        self, mock_score_calculator, reranker_config
    ):
        """Test reranker initialization with model loading failure."""
        failing_loader = create_mock_model_loader(should_load_successfully=False)

        reranker = MultilingualReranker(
            failing_loader, mock_score_calculator, reranker_config
        )

        assert reranker.is_ready is False

    def test_reranker_rerank_basic(self, reranker, sample_documents):
        """Test basic document reranking."""
        query = "programiranje Python"

        results = reranker.rerank(query, sample_documents, top_k=3)

        assert len(results) <= 3
        assert all(isinstance(result, RerankingResult) for result in results)

        # Results should be sorted by score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

        # Each result should have all required fields
        for result in results:
            assert isinstance(result.content, str)
            assert isinstance(result.score, float)
            assert isinstance(result.original_rank, int)
            assert isinstance(result.new_rank, int)
            assert isinstance(result.metadata, dict)

    def test_reranker_rerank_score_normalization(
        self, mock_model_loader, sample_documents
    ):
        """Test score normalization in reranking."""
        # Create calculator with known scores
        mock_calculator = create_mock_score_calculator(
            base_scores=[10.0, 5.0, 20.0, 1.0]
        )
        config = RerankerConfig(normalize_scores=True)

        reranker = MultilingualReranker(mock_model_loader, mock_calculator, config)

        results = reranker.rerank("test query", sample_documents)

        # Scores should be normalized to [0, 1]
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_reranker_rerank_no_normalization(
        self, mock_model_loader, sample_documents
    ):
        """Test reranking without score normalization."""
        mock_calculator = create_mock_score_calculator(base_scores=[2.0, 1.0, 3.0, 0.5])
        config = RerankerConfig(normalize_scores=False)

        reranker = MultilingualReranker(mock_model_loader, mock_calculator, config)

        results = reranker.rerank("test query", sample_documents)

        # Should maintain original score ranges
        assert any(result.score > 1.0 for result in results)

    def test_reranker_rerank_score_threshold(self, mock_model_loader, sample_documents):
        """Test score threshold filtering."""
        mock_calculator = create_mock_score_calculator(base_scores=[0.9, 0.3, 0.7, 0.1])
        config = RerankerConfig(score_threshold=0.5, normalize_scores=False)

        reranker = MultilingualReranker(mock_model_loader, mock_calculator, config)

        results = reranker.rerank("test query", sample_documents)

        # Only results with score >= 0.5 should be returned
        assert all(result.score >= 0.5 for result in results)
        assert len(results) == 2  # Only 2 documents meet threshold

    def test_reranker_rerank_empty_documents(self, reranker):
        """Test reranking with empty document list."""
        results = reranker.rerank("test query", [])
        assert results == []

    def test_reranker_rerank_top_k_limit(self, reranker, sample_documents):
        """Test top_k limiting."""
        results = reranker.rerank("test query", sample_documents, top_k=2)
        assert len(results) <= 2

    def test_reranker_rerank_invalid_inputs(self, reranker, sample_documents):
        """Test reranking with invalid inputs."""
        with pytest.raises(ValueError, match="Query must be string"):
            reranker.rerank(123, sample_documents)

        with pytest.raises(ValueError, match="Documents must be list"):
            reranker.rerank("query", "not list")

        with pytest.raises(
            ValueError, match="All documents must be DocumentItem instances"
        ):
            reranker.rerank("query", ["not document item"])

        with pytest.raises(ValueError, match="Top k must be positive integer"):
            reranker.rerank("query", sample_documents, top_k=0)

    def test_reranker_calculate_quality_metrics(self, reranker, sample_documents):
        """Test reranking quality metrics calculation."""
        results = reranker.rerank("test query", sample_documents)

        metrics = reranker.calculate_reranking_quality(results)

        assert isinstance(metrics, dict)
        assert "items_moved" in metrics
        assert "average_rank_change" in metrics
        assert "mean_score" in metrics
        assert "min_score" in metrics
        assert "max_score" in metrics
        assert "score_std" in metrics

        # Verify metric values are reasonable
        assert isinstance(metrics["items_moved"], int)
        assert isinstance(metrics["average_rank_change"], float)
        assert 0.0 <= metrics["mean_score"] <= 1.0

    def test_reranker_calculate_quality_empty_results(self, reranker):
        """Test quality metrics with empty results."""
        metrics = reranker.calculate_reranking_quality([])
        assert metrics == {}

    def test_reranker_explain_reranking(self, reranker, sample_documents):
        """Test reranking explanation generation."""
        results = reranker.rerank("programiranje", sample_documents[:3])

        explanation = reranker.explain_reranking(results)

        assert "Reranking Explanation" in explanation
        assert "Model:" in explanation
        assert "Device:" in explanation
        assert "Score:" in explanation
        assert "Original rank:" in explanation

        # Should contain rank change indicators
        assert any(indicator in explanation for indicator in ["📈", "📉", "➡️"])

    def test_reranker_explain_empty_results(self, reranker):
        """Test explanation with empty results."""
        explanation = reranker.explain_reranking([])
        assert "No reranking results to explain" in explanation


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_multilingual_reranker(self):
        """Test creating multilingual reranker."""
        model_loader = create_mock_model_loader()
        score_calculator = create_mock_score_calculator()

        reranker = create_multilingual_reranker(
            model_loader=model_loader,
            score_calculator=score_calculator,
            model_name="custom/model",
            device="cuda",
            batch_size=8,
        )

        assert isinstance(reranker, MultilingualReranker)
        assert reranker.config.model_name == "custom/model"
        assert reranker.config.device == "cuda"
        assert reranker.config.batch_size == 8

    def test_create_mock_model_loader_success(self):
        """Test creating successful mock model loader."""
        loader = create_mock_model_loader(should_load_successfully=True, is_loaded=True)

        model = loader.load_model("test/model", "cpu")
        assert model == "mock_model"
        assert loader.is_model_loaded() is True

    def test_create_mock_model_loader_failure(self):
        """Test creating failing mock model loader."""
        loader = create_mock_model_loader(should_load_successfully=False)

        with pytest.raises(ValueError, match="Mock failure loading"):
            loader.load_model("test/model", "cpu")

    def test_create_mock_model_loader_not_loaded(self):
        """Test creating mock model loader that reports not loaded."""
        loader = create_mock_model_loader(
            should_load_successfully=True, is_loaded=False
        )

        loader.load_model("test/model", "cpu")
        assert loader.is_model_loaded() is False

    def test_create_mock_score_calculator_default(self):
        """Test creating mock score calculator with default behavior."""
        calculator = create_mock_score_calculator()

        pairs = [("query", "document with query words"), ("query", "unrelated content")]
        scores = calculator.calculate_scores(pairs, batch_size=2)

        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        assert scores[0] > scores[1]  # First should score higher due to word overlap

    def test_create_mock_score_calculator_custom_scores(self):
        """Test creating mock score calculator with custom scores."""
        custom_scores = [0.9, 0.5, 0.7]
        calculator = create_mock_score_calculator(base_scores=custom_scores)

        pairs = [("q", "d1"), ("q", "d2"), ("q", "d3"), ("q", "d4"), ("q", "d5")]
        scores = calculator.calculate_scores(pairs, batch_size=3)

        assert len(scores) == 5
        assert scores[0] == 0.9
        assert scores[1] == 0.5
        assert scores[2] == 0.7
        assert scores[3] == 0.9  # Should cycle
        assert scores[4] == 0.5  # Should cycle

    def test_create_mock_score_calculator_with_noise(self):
        """Test creating mock score calculator with noise."""
        calculator = create_mock_score_calculator(base_scores=[0.5], add_noise=True)

        pairs = [("query", "document")] * 10
        scores = calculator.calculate_scores(pairs, batch_size=5)

        assert len(scores) == 10
        # Scores should vary due to noise but stay in valid range
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert len(set(scores)) > 1  # Should have variation due to noise


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.fixture
    def multilingual_reranker(self):
        """Create reranker for multilingual testing."""
        model_loader = create_mock_model_loader(
            should_load_successfully=True, is_loaded=True
        )
        score_calculator = create_mock_score_calculator()
        config = RerankerConfig(normalize_scores=True, batch_size=3)

        return MultilingualReranker(model_loader, score_calculator, config)

    def test_croatian_document_reranking(self, multilingual_reranker):
        """Test complete Croatian document reranking workflow."""
        # Croatian documents with varied relevance to query
        documents = [
            DocumentItem(
                "Python je programski jezik koji se često koristi za data science projekte",
                {"source": "python_tutorial.txt", "category": "programming"},
                0.7,
            ),
            DocumentItem(
                "Hrvatska je zemlja u jugoistočnoj Europi s bogatom kulturnom baštinom",
                {"source": "croatia_info.txt", "category": "geography"},
                0.8,
            ),
            DocumentItem(
                "Algoritmi strojnog učenja omogućavaju računalima da uče iz podataka",
                {"source": "ml_basics.txt", "category": "ai"},
                0.9,
            ),
            DocumentItem(
                "Django je web framework za Python koji omogućava brzu web aplikacija",
                {"source": "django_guide.txt", "category": "web"},
                0.6,
            ),
            DocumentItem(
                "Baze podataka su ključne za čuvanje i dohvaćanje informacija u aplikacijama",
                {"source": "database_intro.txt", "category": "data"},
                0.5,
            ),
        ]

        # Query about programming and Python
        query = "Python programiranje algoritmi"

        results = multilingual_reranker.rerank(query, documents, top_k=3)

        # Verify reranking results
        assert len(results) <= 3
        assert all(isinstance(result, RerankingResult) for result in results)

        # Results should be ordered by relevance score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

        # Check that relevant documents score higher
        result_contents = [result.content for result in results]
        python_docs = [
            content for content in result_contents if "python" in content.lower()
        ]
        assert len(python_docs) > 0  # Should include Python-related documents

        # Verify metadata preservation
        for result in results:
            assert "source" in result.metadata
            assert "category" in result.metadata

        # Test explanation generation
        explanation = multilingual_reranker.explain_reranking(results)
        assert "Python" in explanation or "programiranje" in explanation
        assert "📈" in explanation or "📉" in explanation or "➡️" in explanation

    def test_multilingual_mixed_content_reranking(self, multilingual_reranker):
        """Test reranking with mixed Croatian-English content."""
        mixed_documents = [
            DocumentItem(
                "Python programming je vrlo popularan za machine learning projekte",
                {"lang": "mixed"},
                0.8,
            ),
            DocumentItem(
                "Deep learning algoritmi koriste neural networks za pattern recognition",
                {"lang": "mixed"},
                0.7,
            ),
            DocumentItem(
                "Web development s React bibliotekom omogućava interactive aplikacije",
                {"lang": "mixed"},
                0.6,
            ),
            DocumentItem(
                "Database management sistemas kao PostgreSQL su enterprise-ready",
                {"lang": "mixed"},
                0.9,
            ),
            DocumentItem(
                "Cloud computing platforms poput AWS-a pružaju scalable infrastructure",
                {"lang": "mixed"},
                0.5,
            ),
        ]

        # Query mixing languages
        query = "machine learning development Python"

        results = multilingual_reranker.rerank(query, mixed_documents, top_k=4)

        assert len(results) <= 4

        # Should handle mixed language content gracefully
        for result in results:
            assert isinstance(result.content, str)
            assert len(result.content) > 0
            assert 0.0 <= result.score <= 1.0

    def test_large_document_corpus_reranking(self, multilingual_reranker):
        """Test reranking performance with larger document corpus."""
        # Generate larger set of documents
        base_topics = [
            "programiranje u Python jeziku",
            "umjetna inteligencija i strojno učenje",
            "web development s JavaScript frameworkovima",
            "baze podataka i SQL optimizacija",
            "cloud computing i DevOps praksa",
        ]

        # Create variations to simulate larger corpus
        documents = []
        for i in range(20):  # 100 documents total
            for j, topic in enumerate(base_topics):
                doc_id = i * len(base_topics) + j
                documents.append(
                    DocumentItem(
                        f"{topic} - dodatne informacije i detalji {i+1}",
                        {"doc_id": doc_id, "topic": j, "iteration": i},
                        0.5 + (doc_id % 10) * 0.05,  # Varied original scores
                    )
                )

        query = "Python programiranje strojno učenje"

        # Should handle large corpus efficiently
        results = multilingual_reranker.rerank(query, documents, top_k=10)

        assert len(results) <= 10
        assert all(isinstance(result, RerankingResult) for result in results)

        # Verify quality metrics calculation
        metrics = multilingual_reranker.calculate_reranking_quality(results)
        assert isinstance(metrics, dict)
        assert all(
            key in metrics
            for key in ["items_moved", "mean_score", "average_rank_change"]
        )

    def test_reranking_quality_assessment(self, multilingual_reranker):
        """Test comprehensive reranking quality assessment."""
        # Documents with clear relevance hierarchy for testing
        documents = [
            DocumentItem(
                "Completely irrelevant document about cooking",
                {"relevance": "none"},
                0.9,
            ),
            DocumentItem("Python je programski jezik", {"relevance": "high"}, 0.1),
            DocumentItem(
                "Programming languages are tools", {"relevance": "medium"}, 0.5
            ),
            DocumentItem("Machine learning koristi Python", {"relevance": "high"}, 0.2),
            DocumentItem("Random unrelated content here", {"relevance": "none"}, 0.8),
        ]

        query = "Python programming machine learning"

        # Rerank documents
        results = multilingual_reranker.rerank(query, documents)

        # Calculate quality metrics
        metrics = multilingual_reranker.calculate_reranking_quality(results)

        # Verify comprehensive metrics
        expected_metrics = [
            "kendall_tau",
            "spearman_rho",
            "rank_correlation",
            "items_moved",
            "average_rank_change",
            "mean_score",
            "min_score",
            "max_score",
            "score_std",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

        # Test explanation quality
        explanation = multilingual_reranker.explain_reranking(results)

        # Should contain key information
        assert "Model:" in explanation
        assert "Total documents reranked:" in explanation
        assert "Score:" in explanation

        # Should show rank changes
        rank_change_indicators = ["📈", "📉", "➡️"]
        assert any(indicator in explanation for indicator in rank_change_indicators)

    def test_edge_cases_and_error_handling(self, multilingual_reranker):
        """Test handling of various edge cases."""
        edge_case_documents = [
            DocumentItem("", {"empty": True}, 0.5),  # Empty content
            DocumentItem("a", {"very_short": True}, 0.7),  # Very short content
            DocumentItem("Word " * 1000, {"very_long": True}, 0.3),  # Very long content
            DocumentItem(
                "!@#$%^&*()", {"special_chars": True}, 0.8
            ),  # Special characters only
            DocumentItem("123456789", {"numbers_only": True}, 0.2),  # Numbers only
            DocumentItem(
                "Normal content with Croatian čćšđž characters", {"normal": True}, 0.6
            ),
        ]

        queries = [
            "",  # Empty query
            "a",  # Very short query
            "very long query with many words that could potentially cause issues with tokenization or processing",  # Very long query
            "!@#$%^&*()",  # Special characters query
            "123456789",  # Numbers query
            "normal query with čćšđž",  # Normal Croatian query
        ]

        # Test with various combinations
        for query in queries[:3]:  # Test subset to avoid too many combinations
            try:
                results = multilingual_reranker.rerank(
                    query, edge_case_documents[:4], top_k=3
                )

                # Should handle gracefully
                assert isinstance(results, list)
                assert all(isinstance(result, RerankingResult) for result in results)

                # All scores should be valid
                for result in results:
                    assert 0.0 <= result.score <= 1.0
                    assert isinstance(result.original_rank, int)
                    assert isinstance(result.new_rank, int)

            except Exception as e:
                pytest.fail(f"Edge case failed - query: '{query[:20]}...', error: {e}")

    def test_batch_processing_consistency(self, multilingual_reranker):
        """Test that batch processing produces consistent results."""
        documents = [
            DocumentItem(
                f"Document {i} about various topics and subjects", {"id": i}, 0.5
            )
            for i in range(15)  # More than typical batch size
        ]

        query = "topics subjects various"

        # Run reranking multiple times
        results1 = multilingual_reranker.rerank(query, documents, top_k=10)
        results2 = multilingual_reranker.rerank(query, documents, top_k=10)

        # Results should be consistent (assuming deterministic scoring)
        assert len(results1) == len(results2)

        for r1, r2 in zip(results1, results2):
            assert r1.content == r2.content
            assert r1.original_rank == r2.original_rank
            assert r1.new_rank == r2.new_rank
            # Scores might have slight variations due to mock randomness, but structure should be same


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
