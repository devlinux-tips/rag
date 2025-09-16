"""
Comprehensive tests for hybrid retrieval system.
Tests pure functions, BM25 scoring, hybrid combination, and dependency injection patterns.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from dataclasses import dataclass
from typing import Any

from src.retrieval.hybrid_retriever import (
    # Pure functions
    normalize_text_for_bm25,
    calculate_bm25_score,
    normalize_scores,
    combine_hybrid_scores,
    rank_results_by_score,
    calculate_corpus_statistics,

    # Data structures
    HybridConfig,
    HybridResult,
    DenseResult,

    # Core classes
    BM25Scorer,
    HybridRetriever,

    # Factory functions
    create_hybrid_retriever,
    create_hybrid_retriever_from_config,
    create_mock_stop_words_provider,
)
from src.utils.config_models import HybridRetrievalConfig


# ===== PURE FUNCTION TESTS =====

class TestNormalizeTextForBM25:
    """Test normalize_text_for_bm25 pure function."""

    def test_basic_normalization(self):
        """Test basic text normalization."""
        text = "Hello World! This is a test."
        stop_words = {"is", "a"}

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        assert "hello" in result
        assert "world" in result
        assert "this" in result
        assert "test" in result
        assert "is" not in result  # Stop word
        assert "a" not in result   # Stop word

    def test_croatian_text_normalization(self):
        """Test Croatian text with diacritics."""
        text = "Ovo je test sa hrvatskim znakovima: čćšđž"
        stop_words = {"je", "sa"}

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        assert "ovo" in result
        assert "test" in result
        assert "hrvatskim" in result
        assert "znakovima" in result
        assert "čćšđž" in result
        assert "je" not in result  # Stop word
        assert "sa" not in result  # Stop word

    def test_punctuation_removal(self):
        """Test punctuation and special character removal."""
        text = "Hello, world! This is... a test? With (parentheses) and [brackets]."
        stop_words = set()

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        expected = {"hello", "world", "this", "test", "with", "parentheses", "and", "brackets"}
        assert set(result) == expected

    def test_min_token_length_filter(self):
        """Test minimum token length filtering."""
        text = "a bb ccc dddd"
        stop_words = set()

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        assert "a" not in result    # Too short
        assert "bb" not in result   # Too short
        assert "ccc" in result      # Exactly 3 chars
        assert "dddd" in result     # Longer than 3

    def test_digit_removal(self):
        """Test digit-only token removal."""
        text = "word 123 another 456 final"
        stop_words = set()

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        assert "word" in result
        assert "another" in result
        assert "final" in result
        assert "123" not in result  # Digit-only
        assert "456" not in result  # Digit-only

    def test_empty_text(self):
        """Test empty text."""
        result = normalize_text_for_bm25("", set(), min_token_length=3)
        assert result == []

    def test_whitespace_only(self):
        """Test whitespace-only text."""
        result = normalize_text_for_bm25("   \n\t  ", set(), min_token_length=3)
        assert result == []

    def test_invalid_inputs(self):
        """Test invalid input types."""
        stop_words = {"test"}

        with pytest.raises(ValueError, match="Text must be string"):
            normalize_text_for_bm25(123, stop_words)

        with pytest.raises(ValueError, match="Stop words must be set"):
            normalize_text_for_bm25("text", ["not", "set"])

        with pytest.raises(ValueError, match="Min token length must be positive integer"):
            normalize_text_for_bm25("text", stop_words, min_token_length=0)

        with pytest.raises(ValueError, match="Min token length must be positive integer"):
            normalize_text_for_bm25("text", stop_words, min_token_length="not_int")


class TestCalculateBM25Score:
    """Test calculate_bm25_score pure function."""

    def test_basic_bm25_calculation(self):
        """Test basic BM25 score calculation."""
        query_tokens = ["python", "programming"]
        doc_tokens = ["python", "is", "great", "for", "programming", "python"]

        # Use corpus_size > 1 to get positive IDF
        score = calculate_bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.75, avgdl=5.0, corpus_size=10)

        assert score > 0  # Should have positive score due to matches
        assert isinstance(score, float)

    def test_no_query_matches(self):
        """Test when query tokens don't match document."""
        query_tokens = ["java", "development"]
        doc_tokens = ["python", "programming", "tutorial"]

        score = calculate_bm25_score(query_tokens, doc_tokens)

        assert score == 0.0

    def test_empty_query(self):
        """Test empty query."""
        score = calculate_bm25_score([], ["some", "document"], k1=1.5, b=0.75)
        assert score == 0.0

    def test_empty_document(self):
        """Test empty document."""
        score = calculate_bm25_score(["query"], [], k1=1.5, b=0.75)
        assert score == 0.0

    def test_both_empty(self):
        """Test both query and document empty."""
        score = calculate_bm25_score([], [], k1=1.5, b=0.75)
        assert score == 0.0

    def test_exact_match(self):
        """Test query that exactly matches document."""
        tokens = ["python", "programming"]

        score = calculate_bm25_score(tokens, tokens, k1=1.5, b=0.75, avgdl=2.0, corpus_size=5)

        assert score > 0

    def test_repeated_terms(self):
        """Test handling of repeated terms."""
        query_tokens = ["python", "python"]
        doc_tokens = ["python", "programming", "python", "tutorial"]

        score = calculate_bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.75, avgdl=4.0, corpus_size=3)

        assert score > 0

    def test_parameter_variations(self):
        """Test different parameter values."""
        query_tokens = ["test"]
        doc_tokens = ["test", "document", "test"]  # Repeated term and different length

        # Higher k1 should affect scoring with repeated terms
        score_k1_low = calculate_bm25_score(query_tokens, doc_tokens, k1=0.5, b=0.75, avgdl=2.0, corpus_size=5)
        score_k1_high = calculate_bm25_score(query_tokens, doc_tokens, k1=2.0, b=0.75, avgdl=2.0, corpus_size=5)

        assert score_k1_low != score_k1_high

        # Different b values should affect scoring with length normalization
        score_b_low = calculate_bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.1, avgdl=2.0, corpus_size=5)
        score_b_high = calculate_bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.9, avgdl=2.0, corpus_size=5)

        assert score_b_low != score_b_high

    def test_zero_avgdl_handling(self):
        """Test handling of zero or negative avgdl."""
        query_tokens = ["test"]
        doc_tokens = ["test"]

        # Should use document length when avgdl <= 0
        score = calculate_bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.75, avgdl=0.0, corpus_size=3)
        assert score > 0

    def test_invalid_inputs(self):
        """Test invalid input types and values."""
        query_tokens = ["test"]
        doc_tokens = ["test"]

        with pytest.raises(ValueError, match="Query tokens must be list"):
            calculate_bm25_score("not_list", doc_tokens)

        with pytest.raises(ValueError, match="Doc tokens must be list"):
            calculate_bm25_score(query_tokens, "not_list")

        with pytest.raises(ValueError, match="k1 must be non-negative number"):
            calculate_bm25_score(query_tokens, doc_tokens, k1=-1)

        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            calculate_bm25_score(query_tokens, doc_tokens, b=-0.1)

        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            calculate_bm25_score(query_tokens, doc_tokens, b=1.1)


class TestNormalizeScores:
    """Test normalize_scores pure function."""

    def test_basic_normalization(self):
        """Test basic score normalization."""
        scores = [1.0, 3.0, 5.0]
        result = normalize_scores(scores)

        assert result == [0.0, 0.5, 1.0]

    def test_single_score(self):
        """Test normalization with single score."""
        result = normalize_scores([5.0])
        assert result == [1.0]

    def test_equal_scores(self):
        """Test normalization when all scores are equal."""
        scores = [3.0, 3.0, 3.0]
        result = normalize_scores(scores)

        assert result == [1.0, 1.0, 1.0]

    def test_empty_list(self):
        """Test empty score list."""
        result = normalize_scores([])
        assert result == []

    def test_negative_scores(self):
        """Test normalization with negative scores."""
        scores = [-2.0, 0.0, 2.0]
        result = normalize_scores(scores)

        assert result == [0.0, 0.5, 1.0]

    def test_already_normalized(self):
        """Test scores already in [0,1] range."""
        scores = [0.0, 0.5, 1.0]
        result = normalize_scores(scores)

        assert result == [0.0, 0.5, 1.0]

    def test_invalid_inputs(self):
        """Test invalid input types."""
        with pytest.raises(ValueError, match="Scores must be list"):
            normalize_scores("not_list")

        with pytest.raises(ValueError, match="All scores must be numbers"):
            normalize_scores([1.0, "not_number", 3.0])


class TestCombineHybridScores:
    """Test combine_hybrid_scores pure function."""

    def test_basic_combination(self):
        """Test basic score combination."""
        dense_scores = [0.8, 0.6, 0.4]
        sparse_scores = [0.2, 0.4, 0.6]

        result = combine_hybrid_scores(dense_scores, sparse_scores, dense_weight=0.7, sparse_weight=0.3)

        assert len(result) == 3
        # First score: 0.7 * 0.8 + 0.3 * 0.2 = 0.56 + 0.06 = 0.62
        assert abs(result[0] - 0.62) < 0.001

    def test_equal_weights(self):
        """Test with equal weights."""
        dense_scores = [0.8, 0.6]
        sparse_scores = [0.2, 0.4]

        result = combine_hybrid_scores(dense_scores, sparse_scores, dense_weight=1.0, sparse_weight=1.0)

        # With equal weights (1.0, 1.0), each should be normalized to 0.5
        # First: 0.5 * 0.8 + 0.5 * 0.2 = 0.5
        assert abs(result[0] - 0.5) < 0.001
        # Second: 0.5 * 0.6 + 0.5 * 0.4 = 0.5
        assert abs(result[1] - 0.5) < 0.001

    def test_dense_only(self):
        """Test with sparse weight zero."""
        dense_scores = [0.8, 0.6]
        sparse_scores = [0.2, 0.4]

        result = combine_hybrid_scores(dense_scores, sparse_scores, dense_weight=1.0, sparse_weight=0.0)

        assert result == dense_scores

    def test_sparse_only(self):
        """Test with dense weight zero."""
        dense_scores = [0.8, 0.6]
        sparse_scores = [0.2, 0.4]

        result = combine_hybrid_scores(dense_scores, sparse_scores, dense_weight=0.0, sparse_weight=1.0)

        assert result == sparse_scores

    def test_empty_lists(self):
        """Test with empty score lists."""
        result = combine_hybrid_scores([], [], dense_weight=0.7, sparse_weight=0.3)
        assert result == []

    def test_invalid_inputs(self):
        """Test invalid input types and values."""
        dense_scores = [0.8]
        sparse_scores = [0.2]

        with pytest.raises(ValueError, match="Dense scores must be list"):
            combine_hybrid_scores("not_list", sparse_scores, 0.7, 0.3)

        with pytest.raises(ValueError, match="Sparse scores must be list"):
            combine_hybrid_scores(dense_scores, "not_list", 0.7, 0.3)

        with pytest.raises(ValueError, match="Score lists must have same length"):
            combine_hybrid_scores([0.8, 0.6], [0.2], 0.7, 0.3)

        with pytest.raises(ValueError, match="Dense weight must be non-negative number"):
            combine_hybrid_scores(dense_scores, sparse_scores, -1.0, 0.3)

        with pytest.raises(ValueError, match="Sparse weight must be non-negative number"):
            combine_hybrid_scores(dense_scores, sparse_scores, 0.7, -1.0)

        with pytest.raises(ValueError, match="At least one weight must be positive"):
            combine_hybrid_scores(dense_scores, sparse_scores, 0.0, 0.0)


class TestRankResultsByScore:
    """Test rank_results_by_score pure function."""

    def test_basic_ranking(self):
        """Test basic result ranking."""
        results = [
            {"content": "doc1", "score": 0.3},
            {"content": "doc2", "score": 0.8},
            {"content": "doc3", "score": 0.5}
        ]

        ranked = rank_results_by_score(results, score_key="score", descending=True)

        assert len(ranked) == 3
        assert ranked[0]["content"] == "doc2"  # Highest score
        assert ranked[1]["content"] == "doc3"  # Middle score
        assert ranked[2]["content"] == "doc1"  # Lowest score

    def test_ascending_ranking(self):
        """Test ascending ranking."""
        results = [
            {"content": "doc1", "score": 0.8},
            {"content": "doc2", "score": 0.3},
            {"content": "doc3", "score": 0.5}
        ]

        ranked = rank_results_by_score(results, score_key="score", descending=False)

        assert ranked[0]["content"] == "doc2"  # Lowest score
        assert ranked[1]["content"] == "doc3"  # Middle score
        assert ranked[2]["content"] == "doc1"  # Highest score

    def test_custom_score_key(self):
        """Test with custom score key."""
        results = [
            {"content": "doc1", "relevance": 0.3},
            {"content": "doc2", "relevance": 0.8}
        ]

        ranked = rank_results_by_score(results, score_key="relevance", descending=True)

        assert ranked[0]["content"] == "doc2"

    def test_equal_scores(self):
        """Test with equal scores."""
        results = [
            {"content": "doc1", "score": 0.5},
            {"content": "doc2", "score": 0.5},
            {"content": "doc3", "score": 0.5}
        ]

        ranked = rank_results_by_score(results, score_key="score", descending=True)

        assert len(ranked) == 3
        # Order should be stable for equal scores

    def test_empty_results(self):
        """Test with empty results."""
        ranked = rank_results_by_score([], score_key="score")
        assert ranked == []

    def test_invalid_inputs(self):
        """Test invalid input types and values."""
        valid_results = [{"content": "doc1", "score": 0.5}]

        with pytest.raises(ValueError, match="Results must be list"):
            rank_results_by_score("not_list", "score")

        with pytest.raises(ValueError, match="Score key must be string"):
            rank_results_by_score(valid_results, 123)

        with pytest.raises(ValueError, match="Result at index 0 must be dict"):
            rank_results_by_score(["not_dict"], "score")

        with pytest.raises(ValueError, match="Result at index 0 missing score key"):
            rank_results_by_score([{"content": "doc1"}], "score")

        with pytest.raises(ValueError, match="Score at index 0 must be number"):
            rank_results_by_score([{"score": "not_number"}], "score")


class TestCalculateCorpusStatistics:
    """Test calculate_corpus_statistics pure function."""

    def test_basic_statistics(self):
        """Test basic corpus statistics calculation."""
        documents = [
            ["word1", "word2", "word3"],
            ["word4", "word5"],
            ["word6", "word7", "word8", "word9"]
        ]

        stats = calculate_corpus_statistics(documents)

        assert stats["total_docs"] == 3
        assert stats["total_tokens"] == 9  # 3 + 2 + 4
        assert abs(stats["avgdl"] - 3.0) < 0.001  # 9/3 = 3.0

    def test_single_document(self):
        """Test with single document."""
        documents = [["word1", "word2"]]

        stats = calculate_corpus_statistics(documents)

        assert stats["total_docs"] == 1
        assert stats["total_tokens"] == 2
        assert stats["avgdl"] == 2.0

    def test_empty_documents(self):
        """Test with documents containing empty lists."""
        documents = [["word1"], [], ["word2", "word3"]]

        stats = calculate_corpus_statistics(documents)

        assert stats["total_docs"] == 3
        assert stats["total_tokens"] == 3
        assert stats["avgdl"] == 1.0

    def test_no_documents(self):
        """Test with no documents."""
        stats = calculate_corpus_statistics([])

        assert stats["total_docs"] == 0
        assert stats["total_tokens"] == 0
        assert stats["avgdl"] == 0.0

    def test_invalid_inputs(self):
        """Test invalid input types."""
        with pytest.raises(ValueError, match="Documents must be list"):
            calculate_corpus_statistics("not_list")

        with pytest.raises(ValueError, match="All documents must be lists of tokens"):
            calculate_corpus_statistics([["valid"], "not_list"])


# ===== DATA STRUCTURE TESTS =====

class TestHybridConfig:
    """Test HybridConfig data class."""

    def test_default_initialization(self):
        """Test default configuration."""
        config = HybridConfig()

        assert config.dense_weight == 0.7
        assert config.sparse_weight == 0.3
        assert config.bm25_k1 == 1.5
        assert config.bm25_b == 0.75
        assert config.min_token_length == 3
        assert config.normalize_scores is True
        assert config.min_score_threshold == 0.0

    def test_custom_initialization(self):
        """Test custom configuration."""
        config = HybridConfig(
            dense_weight=0.8,
            sparse_weight=0.2,
            bm25_k1=2.0,
            bm25_b=0.5,
            min_token_length=2,
            normalize_scores=False,
            min_score_threshold=0.1
        )

        assert config.dense_weight == 0.8
        assert config.sparse_weight == 0.2
        assert config.bm25_k1 == 2.0
        assert config.bm25_b == 0.5
        assert config.min_token_length == 2
        assert config.normalize_scores is False
        assert config.min_score_threshold == 0.1

    def test_from_validated_config(self):
        """Test creation from HybridRetrievalConfig."""
        main_config = {
            "hybrid_retrieval": {
                "dense_weight": 0.8,
                "sparse_weight": 0.2,
                "fusion_method": "linear",
                "bm25_k1": 2.0,
                "bm25_b": 0.5
            }
        }

        hybrid_config = HybridRetrievalConfig.from_validated_config(main_config)
        config = HybridConfig.from_validated_config(hybrid_config)

        assert config.dense_weight == 0.8
        assert config.sparse_weight == 0.2
        assert config.bm25_k1 == 2.0
        assert config.bm25_b == 0.5
        # Defaults for fields not in HybridRetrievalConfig
        assert config.min_token_length == 3
        assert config.normalize_scores is True
        assert config.min_score_threshold == 0.0

    def test_validation_errors(self):
        """Test validation errors."""
        with pytest.raises(ValueError, match="Dense weight must be non-negative number"):
            HybridConfig(dense_weight=-1.0)

        with pytest.raises(ValueError, match="Sparse weight must be non-negative number"):
            HybridConfig(sparse_weight=-1.0)

        with pytest.raises(ValueError, match="At least one weight must be positive"):
            HybridConfig(dense_weight=0.0, sparse_weight=0.0)

        with pytest.raises(ValueError, match="BM25 k1 must be non-negative number"):
            HybridConfig(bm25_k1=-1.0)

        with pytest.raises(ValueError, match="BM25 b must be between 0 and 1"):
            HybridConfig(bm25_b=-0.1)

        with pytest.raises(ValueError, match="BM25 b must be between 0 and 1"):
            HybridConfig(bm25_b=1.1)

        with pytest.raises(ValueError, match="Min token length must be positive integer"):
            HybridConfig(min_token_length=0)

        with pytest.raises(ValueError, match="Normalize scores must be boolean"):
            HybridConfig(normalize_scores="not_bool")

        with pytest.raises(ValueError, match="Min score threshold must be number"):
            HybridConfig(min_score_threshold="not_number")


class TestHybridResult:
    """Test HybridResult data class."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        result = HybridResult(
            content="test content",
            score=0.8,
            dense_score=0.7,
            sparse_score=0.2
        )

        assert result.content == "test content"
        assert result.score == 0.8
        assert result.dense_score == 0.7
        assert result.sparse_score == 0.2
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test with metadata."""
        metadata = {"source": "test", "category": "doc"}
        result = HybridResult(
            content="content",
            score=0.5,
            dense_score=0.4,
            sparse_score=0.6,
            metadata=metadata
        )

        assert result.metadata == metadata

    def test_validation_errors(self):
        """Test validation errors."""
        with pytest.raises(ValueError, match="Content must be string"):
            HybridResult(content=123, score=0.5, dense_score=0.4, sparse_score=0.6)

        with pytest.raises(ValueError, match="Score must be number"):
            HybridResult(content="test", score="not_number", dense_score=0.4, sparse_score=0.6)

        with pytest.raises(ValueError, match="Dense score must be number"):
            HybridResult(content="test", score=0.5, dense_score="not_number", sparse_score=0.6)

        with pytest.raises(ValueError, match="Sparse score must be number"):
            HybridResult(content="test", score=0.5, dense_score=0.4, sparse_score="not_number")

        with pytest.raises(ValueError, match="Metadata must be dict"):
            HybridResult(content="test", score=0.5, dense_score=0.4, sparse_score=0.6, metadata="not_dict")


class TestDenseResult:
    """Test DenseResult data class."""

    def test_initialization(self):
        """Test basic initialization."""
        result = DenseResult(content="test content", score=0.8)

        assert result.content == "test content"
        assert result.score == 0.8
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test with metadata."""
        metadata = {"source": "test"}
        result = DenseResult(content="content", score=0.5, metadata=metadata)

        assert result.metadata == metadata


# ===== MOCK FACTORY TESTS =====

class TestMockStopWordsProvider:
    """Test mock stop words provider factory."""

    def test_default_stop_words(self):
        """Test default stop words."""
        provider = create_mock_stop_words_provider()

        hr_words = provider.get_stop_words("hr")
        en_words = provider.get_stop_words("en")

        assert "je" in hr_words
        assert "se" in hr_words
        assert "the" in en_words
        assert "is" in en_words

    def test_custom_stop_words(self):
        """Test custom stop words."""
        custom_hr = {"custom", "croatian"}
        custom_en = {"custom", "english"}

        provider = create_mock_stop_words_provider(
            croatian_stop_words=custom_hr,
            english_stop_words=custom_en
        )

        assert provider.get_stop_words("hr") == custom_hr
        assert provider.get_stop_words("en") == custom_en

    def test_unknown_language(self):
        """Test unknown language."""
        provider = create_mock_stop_words_provider()

        words = provider.get_stop_words("unknown")
        assert words == set()


# ===== CORE CLASS TESTS =====

class TestBM25Scorer:
    """Test BM25Scorer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = HybridConfig()
        self.stop_words_provider = create_mock_stop_words_provider()
        self.scorer = BM25Scorer(self.stop_words_provider, "hr", self.config)

    def test_initialization(self):
        """Test successful initialization."""
        assert self.scorer.language == "hr"
        assert self.scorer.config == self.config
        assert len(self.scorer.stop_words) > 0
        assert self.scorer.is_indexed is False

    def test_index_documents(self):
        """Test document indexing."""
        documents = [
            "Python je programski jezik za učenje",
            "Java je također programski jezik",
            "JavaScript se koristi za web razvoj"
        ]

        self.scorer.index_documents(documents)

        assert self.scorer.is_indexed is True
        assert len(self.scorer.documents) == 3
        assert len(self.scorer.tokenized_docs) == 3
        assert "avgdl" in self.scorer.corpus_stats

    def test_get_scores_basic(self):
        """Test basic score calculation."""
        documents = [
            "Python programiranje je zabavno",
            "Java programiranje je korisno",
            "Web razvoj sa JavaScript"
        ]

        self.scorer.index_documents(documents)
        scores = self.scorer.get_scores("Python programiranje")

        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)
        assert scores[0] > 0  # First document should have highest score

    def test_get_scores_no_matches(self):
        """Test scores when query doesn't match."""
        documents = ["Python programming", "Java development"]

        self.scorer.index_documents(documents)
        scores = self.scorer.get_scores("completely different query")

        assert all(score == 0.0 for score in scores)

    def test_get_scores_empty_query(self):
        """Test scores with empty query."""
        documents = ["Python programming"]

        self.scorer.index_documents(documents)
        scores = self.scorer.get_scores("")

        assert scores == [0.0]

    def test_get_scores_without_indexing(self):
        """Test error when documents not indexed."""
        with pytest.raises(ValueError, match="Documents not indexed"):
            self.scorer.get_scores("test query")

    def test_invalid_documents(self):
        """Test invalid document types."""
        with pytest.raises(ValueError, match="Documents must be list"):
            self.scorer.index_documents("not_list")

        with pytest.raises(ValueError, match="All documents must be strings"):
            self.scorer.index_documents(["valid", 123, "strings"])

    def test_invalid_query(self):
        """Test invalid query type."""
        documents = ["Test document"]
        self.scorer.index_documents(documents)

        with pytest.raises(ValueError, match="Query must be string"):
            self.scorer.get_scores(123)


class TestHybridRetriever:
    """Test HybridRetriever class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = HybridConfig()
        self.stop_words_provider = create_mock_stop_words_provider()
        self.retriever = HybridRetriever(self.stop_words_provider, "hr", self.config)

        # Sample documents and dense results
        self.documents = [
            "Python je programski jezik",
            "Java programiranje za početnike",
            "Web development sa JavaScript"
        ]

        self.dense_results = [
            DenseResult("Python je programski jezik", 0.9, {"id": "1"}),
            DenseResult("Java programiranje za početnike", 0.7, {"id": "2"}),
            DenseResult("Web development sa JavaScript", 0.5, {"id": "3"})
        ]

    def test_initialization(self):
        """Test successful initialization."""
        assert self.retriever.language == "hr"
        assert self.retriever.config == self.config
        assert isinstance(self.retriever.bm25_scorer, BM25Scorer)
        assert self.retriever.is_ready is False

    def test_index_documents(self):
        """Test document indexing."""
        self.retriever.index_documents(self.documents)

        assert self.retriever.is_ready is True
        assert self.retriever.documents == self.documents
        assert self.retriever.bm25_scorer.is_indexed is True

    def test_basic_retrieval(self):
        """Test basic hybrid retrieval."""
        self.retriever.index_documents(self.documents)

        results = self.retriever.retrieve("Python programiranje", self.dense_results, top_k=2)

        assert len(results) <= 2
        assert all(isinstance(result, HybridResult) for result in results)
        assert all(hasattr(result, 'score') for result in results)
        assert all(hasattr(result, 'dense_score') for result in results)
        assert all(hasattr(result, 'sparse_score') for result in results)

    def test_retrieval_preserves_metadata(self):
        """Test that retrieval preserves metadata."""
        self.retriever.index_documents(self.documents)

        results = self.retriever.retrieve("Python", self.dense_results)

        for result in results:
            # Find corresponding dense result
            dense_result = next((dr for dr in self.dense_results if dr.content == result.content), None)
            if dense_result:
                assert result.metadata == dense_result.metadata

    def test_retrieval_ranking(self):
        """Test that results are properly ranked."""
        self.retriever.index_documents(self.documents)

        results = self.retriever.retrieve("Python", self.dense_results)

        # Results should be sorted by hybrid score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_retrieval_with_score_threshold(self):
        """Test retrieval with minimum score threshold."""
        config = HybridConfig(min_score_threshold=0.5)
        retriever = HybridRetriever(self.stop_words_provider, "hr", config)
        retriever.index_documents(self.documents)

        results = retriever.retrieve("Python", self.dense_results)

        # All results should meet threshold
        assert all(result.score >= 0.5 for result in results)

    def test_retrieval_without_indexing(self):
        """Test error when retriever not ready."""
        with pytest.raises(ValueError, match="Retriever not ready"):
            self.retriever.retrieve("test", self.dense_results)

    def test_content_not_indexed_error(self):
        """Test error when dense result content not in indexed documents."""
        self.retriever.index_documents(["Different document"])

        with pytest.raises(ValueError, match="Content not indexed"):
            self.retriever.retrieve("test", self.dense_results)

    def test_invalid_inputs(self):
        """Test invalid input types."""
        self.retriever.index_documents(self.documents)

        with pytest.raises(ValueError, match="Query must be string"):
            self.retriever.retrieve(123, self.dense_results)

        with pytest.raises(ValueError, match="Dense results must be list"):
            self.retriever.retrieve("query", "not_list")

        with pytest.raises(ValueError, match="Top k must be positive integer"):
            self.retriever.retrieve("query", self.dense_results, top_k=0)

    def test_explain_scores(self):
        """Test score explanation generation."""
        self.retriever.index_documents(self.documents)
        results = self.retriever.retrieve("Python", self.dense_results, top_k=2)

        explanation = self.retriever.explain_scores(results)

        assert "🔍 Hybrid Retrieval Score Explanation:" in explanation
        assert f"Dense weight: {self.config.dense_weight:.2f}" in explanation
        assert f"Sparse weight: {self.config.sparse_weight:.2f}" in explanation
        assert "Hybrid Score:" in explanation
        assert "Dense:" in explanation
        assert "Sparse:" in explanation

    def test_explain_scores_empty(self):
        """Test explanation with empty results."""
        explanation = self.retriever.explain_scores([])
        assert explanation == "No results to explain."


# ===== FACTORY FUNCTION TESTS =====

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_hybrid_retriever(self):
        """Test basic factory function."""
        stop_words_provider = create_mock_stop_words_provider()

        retriever = create_hybrid_retriever(
            stop_words_provider=stop_words_provider,
            language="en",
            dense_weight=0.8,
            sparse_weight=0.2,
            bm25_k1=2.0,
            bm25_b=0.5
        )

        assert isinstance(retriever, HybridRetriever)
        assert retriever.language == "en"
        assert retriever.config.dense_weight == 0.8
        assert retriever.config.sparse_weight == 0.2
        assert retriever.config.bm25_k1 == 2.0
        assert retriever.config.bm25_b == 0.5

    def test_create_hybrid_retriever_from_config(self):
        """Test factory function from config."""
        main_config = {
            "hybrid_retrieval": {
                "dense_weight": 0.6,
                "sparse_weight": 0.4,
                "fusion_method": "linear",
                "bm25_k1": 1.8,
                "bm25_b": 0.6
            }
        }

        stop_words_provider = create_mock_stop_words_provider()

        retriever = create_hybrid_retriever_from_config(
            main_config=main_config,
            stop_words_provider=stop_words_provider,
            language="hr"
        )

        assert isinstance(retriever, HybridRetriever)
        assert retriever.language == "hr"
        assert retriever.config.dense_weight == 0.6
        assert retriever.config.sparse_weight == 0.4
        assert retriever.config.bm25_k1 == 1.8
        assert retriever.config.bm25_b == 0.6


# ===== INTEGRATION TESTS =====

class TestIntegration:
    """Integration tests for complete hybrid retrieval workflow."""

    def test_end_to_end_hybrid_retrieval(self):
        """Test complete hybrid retrieval workflow."""
        # Setup
        stop_words_provider = create_mock_stop_words_provider()
        config = HybridConfig(dense_weight=0.6, sparse_weight=0.4)
        retriever = HybridRetriever(stop_words_provider, "hr", config)

        # Documents with varying relevance
        documents = [
            "Python programiranje za početnike",
            "Java development tutorial",
            "Python advanced concepts",
            "JavaScript web development",
            "Machine learning with Python"
        ]

        # Dense results (simulating semantic search results)
        dense_results = [
            DenseResult("Python programiranje za početnike", 0.9, {"source": "tutorial"}),
            DenseResult("Python advanced concepts", 0.8, {"source": "advanced"}),
            DenseResult("Machine learning with Python", 0.7, {"source": "ml"}),
            DenseResult("Java development tutorial", 0.5, {"source": "java"}),
            DenseResult("JavaScript web development", 0.3, {"source": "js"})
        ]

        # Index documents
        retriever.index_documents(documents)

        # Perform hybrid retrieval
        query = "Python programiranje"
        results = retriever.retrieve(query, dense_results, top_k=3)

        # Validate results
        assert len(results) <= 3
        assert all(isinstance(result, HybridResult) for result in results)

        # Results should be ranked by combined score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

        # Verify hybrid scores are combination of dense and sparse
        for result in results:
            assert 0 <= result.score <= 1
            assert isinstance(result.dense_score, float)
            assert isinstance(result.sparse_score, float)

        # Generate explanation
        explanation = retriever.explain_scores(results)
        assert isinstance(explanation, str)
        assert "Hybrid Retrieval Score Explanation" in explanation

    def test_multilingual_processing(self):
        """Test multilingual text processing."""
        # Croatian stop words provider
        stop_words_provider = create_mock_stop_words_provider()
        retriever = HybridRetriever(stop_words_provider, "hr", HybridConfig())

        # Croatian documents with diacritics
        documents = [
            "Programiranje u Python-u je vrlo zanimljivo",
            "Java jezik se široko koristi u industriji",
            "Web stranice mogu biti napravljene sa JavaScript-om"
        ]

        dense_results = [
            DenseResult(doc, 0.8 - i*0.1, {"id": str(i)})
            for i, doc in enumerate(documents)
        ]

        retriever.index_documents(documents)

        # Query with Croatian text
        results = retriever.retrieve("programiranje Python", dense_results)

        assert len(results) > 0
        # First result should be the Python document
        assert "Python" in results[0].content

    def test_score_normalization_effects(self):
        """Test effects of score normalization."""
        stop_words_provider = create_mock_stop_words_provider()

        # Test with normalization enabled
        config_with_norm = HybridConfig(normalize_scores=True)
        retriever_norm = HybridRetriever(stop_words_provider, "hr", config_with_norm)

        # Test without normalization
        config_no_norm = HybridConfig(normalize_scores=False)
        retriever_no_norm = HybridRetriever(stop_words_provider, "hr", config_no_norm)

        documents = ["Python programming", "Java development"]
        dense_results = [
            DenseResult("Python programming", 0.8),
            DenseResult("Java development", 0.6)
        ]

        retriever_norm.index_documents(documents)
        retriever_no_norm.index_documents(documents)

        results_norm = retriever_norm.retrieve("Python", dense_results)
        results_no_norm = retriever_no_norm.retrieve("Python", dense_results)

        # Both should return results, but scores may differ
        assert len(results_norm) > 0
        assert len(results_no_norm) > 0

    def test_performance_with_large_corpus(self):
        """Test performance with larger document corpus."""
        stop_words_provider = create_mock_stop_words_provider()
        retriever = HybridRetriever(stop_words_provider, "hr", HybridConfig())

        # Create larger corpus
        documents = [f"Document {i} with various content about programming" for i in range(100)]
        dense_results = [
            DenseResult(doc, 0.9 - (i % 10) * 0.1, {"id": str(i)})
            for i, doc in enumerate(documents)
        ]

        retriever.index_documents(documents)
        results = retriever.retrieve("programming", dense_results, top_k=10)

        assert len(results) == 10
        assert all(isinstance(result, HybridResult) for result in results)

    def test_edge_cases(self):
        """Test various edge cases."""
        stop_words_provider = create_mock_stop_words_provider()
        retriever = HybridRetriever(stop_words_provider, "hr", HybridConfig())

        # Edge case: Single document
        documents = ["Single test document"]
        dense_results = [DenseResult("Single test document", 0.8)]

        retriever.index_documents(documents)
        results = retriever.retrieve("test", dense_results)

        assert len(results) == 1

        # Edge case: Empty query (should handle gracefully)
        results = retriever.retrieve("", dense_results)
        assert len(results) >= 0  # Should not crash

    def test_error_recovery(self):
        """Test error recovery scenarios."""
        stop_words_provider = create_mock_stop_words_provider()
        retriever = HybridRetriever(stop_words_provider, "hr", HybridConfig())

        # Test with mismatched content
        documents = ["Document A", "Document B"]
        dense_results = [DenseResult("Document C", 0.8)]  # Not in indexed docs

        retriever.index_documents(documents)

        with pytest.raises(ValueError, match="Content not indexed"):
            retriever.retrieve("test", dense_results)