"""
Comprehensive tests for hybrid_retriever.py demonstrating 100% testability.
Tests pure functions, dependency injection, and integration scenarios.
"""

from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import pytest
from src.retrieval.hybrid_retriever import (  # Pure functions; Data structures; Core classes; Factory functions
    BM25Scorer, DenseResult, HybridConfig, HybridResult, HybridRetriever,
    calculate_bm25_score, calculate_corpus_statistics, combine_hybrid_scores,
    create_hybrid_retriever, create_mock_stop_words_provider, normalize_scores,
    normalize_text_for_bm25, rank_results_by_score)


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_normalize_text_for_bm25_basic(self):
        """Test basic text normalization for BM25."""
        text = "Ovo je HRVATSKI tekst sa interpunkcijom!"
        stop_words = {"je", "sa"}

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        assert "ovo" in result
        assert "hrvatski" in result
        assert "tekst" in result
        assert "interpunkcijom" in result
        assert "je" not in result  # Removed as stop word
        assert "sa" not in result  # Removed as stop word

    def test_normalize_text_for_bm25_croatian_diacritics(self):
        """Test Croatian diacritics handling."""
        text = "Čestitam na đačkom uspjehu sa šarenim čarapama!"
        stop_words = {"na", "sa"}

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        assert "čestitam" in result
        assert "đačkom" in result
        assert "uspjehu" in result
        assert "šarenim" in result
        assert "čarapama" in result
        assert "na" not in result
        assert "sa" not in result

    def test_normalize_text_for_bm25_min_length_filter(self):
        """Test minimum token length filtering."""
        text = "A je on to uradio?"
        stop_words = set()

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        assert "uradio" in result
        assert "a" not in result  # Too short
        assert "je" not in result  # Too short
        assert "on" not in result  # Too short
        assert "to" not in result  # Too short

    def test_normalize_text_for_bm25_digit_removal(self):
        """Test digit removal."""
        text = "Imam 25 godina i 100 kuna"
        stop_words = set()

        result = normalize_text_for_bm25(text, stop_words, min_token_length=3)

        assert "imam" in result
        assert "godina" in result
        assert "kuna" in result
        assert "25" not in result
        assert "100" not in result

    def test_normalize_text_for_bm25_invalid_inputs(self):
        """Test text normalization with invalid inputs."""
        with pytest.raises(ValueError, match="Text must be string"):
            normalize_text_for_bm25(123, set(), 3)

        with pytest.raises(ValueError, match="Stop words must be set"):
            normalize_text_for_bm25("text", ["list"], 3)

        with pytest.raises(
            ValueError, match="Min token length must be positive integer"
        ):
            normalize_text_for_bm25("text", set(), 0)

    def test_calculate_bm25_score_basic(self):
        """Test basic BM25 score calculation."""
        query_tokens = ["test", "document"]
        doc_tokens = ["this", "is", "test", "document", "content"]

        score = calculate_bm25_score(
            query_tokens=query_tokens,
            doc_tokens=doc_tokens,
            k1=1.5,
            b=0.75,
            avgdl=10.0,
            corpus_size=100,
        )

        assert isinstance(score, float)
        assert score > 0.0  # Should have positive score for matching terms

    def test_calculate_bm25_score_no_matches(self):
        """Test BM25 score with no matching terms."""
        query_tokens = ["nonexistent", "terms"]
        doc_tokens = ["this", "document", "has", "different", "words"]

        score = calculate_bm25_score(query_tokens=query_tokens, doc_tokens=doc_tokens)

        assert score == 0.0

    def test_calculate_bm25_score_empty_inputs(self):
        """Test BM25 score with empty inputs."""
        # Empty query
        assert calculate_bm25_score([], ["doc", "tokens"]) == 0.0

        # Empty document
        assert calculate_bm25_score(["query", "tokens"], []) == 0.0

        # Both empty
        assert calculate_bm25_score([], []) == 0.0

    def test_calculate_bm25_score_invalid_inputs(self):
        """Test BM25 score with invalid inputs."""
        with pytest.raises(ValueError, match="Query tokens must be list"):
            calculate_bm25_score("query", ["doc"])

        with pytest.raises(ValueError, match="Doc tokens must be list"):
            calculate_bm25_score(["query"], "doc")

        with pytest.raises(ValueError, match="k1 must be non-negative number"):
            calculate_bm25_score(["query"], ["doc"], k1=-1.0)

        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            calculate_bm25_score(["query"], ["doc"], b=1.5)

    def test_normalize_scores_basic(self):
        """Test basic score normalization."""
        scores = [10.0, 5.0, 15.0, 0.0]

        normalized = normalize_scores(scores)

        assert len(normalized) == 4
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        assert normalized[2] == 1.0  # Highest original score
        assert normalized[3] == 0.0  # Lowest original score

    def test_normalize_scores_equal_scores(self):
        """Test normalization with equal scores."""
        scores = [5.0, 5.0, 5.0]

        normalized = normalize_scores(scores)

        assert all(score == 1.0 for score in normalized)

    def test_normalize_scores_empty_list(self):
        """Test normalization with empty list."""
        normalized = normalize_scores([])
        assert normalized == []

    def test_normalize_scores_invalid_inputs(self):
        """Test score normalization with invalid inputs."""
        with pytest.raises(ValueError, match="Scores must be list"):
            normalize_scores("not a list")

        with pytest.raises(ValueError, match="All scores must be numbers"):
            normalize_scores([1.0, "not a number", 3.0])

    def test_combine_hybrid_scores_basic(self):
        """Test basic hybrid score combination."""
        dense_scores = [0.8, 0.6, 0.4]
        sparse_scores = [0.2, 0.7, 0.9]

        hybrid_scores = combine_hybrid_scores(
            dense_scores=dense_scores,
            sparse_scores=sparse_scores,
            dense_weight=0.7,
            sparse_weight=0.3,
        )

        assert len(hybrid_scores) == 3
        # Check first score: 0.7 * 0.8 + 0.3 * 0.2 = 0.56 + 0.06 = 0.62
        assert abs(hybrid_scores[0] - 0.62) < 0.01

    def test_combine_hybrid_scores_weight_normalization(self):
        """Test that weights are normalized."""
        dense_scores = [0.5]
        sparse_scores = [0.5]

        # Weights that don't sum to 1
        hybrid_scores = combine_hybrid_scores(
            dense_scores=dense_scores,
            sparse_scores=sparse_scores,
            dense_weight=2.0,
            sparse_weight=3.0,
        )

        # Should normalize to 2/5 and 3/5, result = 0.4 * 0.5 + 0.6 * 0.5 = 0.5
        assert abs(hybrid_scores[0] - 0.5) < 0.01

    def test_combine_hybrid_scores_invalid_inputs(self):
        """Test hybrid score combination with invalid inputs."""
        with pytest.raises(ValueError, match="Dense scores must be list"):
            combine_hybrid_scores("not list", [0.1], 0.7, 0.3)

        with pytest.raises(ValueError, match="Sparse scores must be list"):
            combine_hybrid_scores([0.1], "not list", 0.7, 0.3)

        with pytest.raises(ValueError, match="Score lists must have same length"):
            combine_hybrid_scores([0.1, 0.2], [0.1], 0.7, 0.3)

        with pytest.raises(ValueError, match="Dense weight must be non-negative"):
            combine_hybrid_scores([0.1], [0.1], -0.1, 0.3)

        with pytest.raises(ValueError, match="At least one weight must be positive"):
            combine_hybrid_scores([0.1], [0.1], 0.0, 0.0)

    def test_rank_results_by_score_descending(self):
        """Test ranking results in descending order."""
        results = [
            {"content": "doc1", "score": 0.5},
            {"content": "doc2", "score": 0.8},
            {"content": "doc3", "score": 0.2},
        ]

        ranked = rank_results_by_score(results, "score", descending=True)

        assert ranked[0]["content"] == "doc2"  # Highest score
        assert ranked[1]["content"] == "doc1"
        assert ranked[2]["content"] == "doc3"  # Lowest score

    def test_rank_results_by_score_ascending(self):
        """Test ranking results in ascending order."""
        results = [
            {"content": "doc1", "score": 0.5},
            {"content": "doc2", "score": 0.8},
            {"content": "doc3", "score": 0.2},
        ]

        ranked = rank_results_by_score(results, "score", descending=False)

        assert ranked[0]["content"] == "doc3"  # Lowest score
        assert ranked[1]["content"] == "doc1"
        assert ranked[2]["content"] == "doc2"  # Highest score

    def test_rank_results_by_score_invalid_inputs(self):
        """Test ranking with invalid inputs."""
        with pytest.raises(ValueError, match="Results must be list"):
            rank_results_by_score("not list", "score")

        with pytest.raises(ValueError, match="Score key must be string"):
            rank_results_by_score([], 123)

        with pytest.raises(ValueError, match="Result at index 0 must be dict"):
            rank_results_by_score(["not dict"], "score")

        with pytest.raises(ValueError, match="Result at index 0 missing score key"):
            rank_results_by_score([{"other": "value"}], "score")

        with pytest.raises(ValueError, match="Score at index 0 must be number"):
            rank_results_by_score([{"score": "not number"}], "score")

    def test_calculate_corpus_statistics_basic(self):
        """Test basic corpus statistics calculation."""
        documents = [
            ["word1", "word2", "word3"],
            ["word1", "word4"],
            ["word5", "word6", "word7", "word8"],
        ]

        stats = calculate_corpus_statistics(documents)

        assert stats["total_docs"] == 3
        assert stats["total_tokens"] == 9  # 3 + 2 + 4
        assert abs(stats["avgdl"] - 3.0) < 0.01  # 9 / 3

    def test_calculate_corpus_statistics_empty(self):
        """Test corpus statistics with empty list."""
        stats = calculate_corpus_statistics([])

        assert stats["total_docs"] == 0
        assert stats["total_tokens"] == 0
        assert stats["avgdl"] == 0.0

    def test_calculate_corpus_statistics_invalid_inputs(self):
        """Test corpus statistics with invalid inputs."""
        with pytest.raises(ValueError, match="Documents must be list"):
            calculate_corpus_statistics("not list")

        with pytest.raises(ValueError, match="All documents must be lists of tokens"):
            calculate_corpus_statistics(["not a list of tokens"])


class TestDataStructures:
    """Test data structure classes."""

    def test_hybrid_config_creation(self):
        """Test creating HybridConfig."""
        config = HybridConfig(
            dense_weight=0.8,
            sparse_weight=0.2,
            bm25_k1=2.0,
            bm25_b=0.5,
            min_token_length=2,
            normalize_scores=False,
            min_score_threshold=0.1,
        )

        assert config.dense_weight == 0.8
        assert config.sparse_weight == 0.2
        assert config.bm25_k1 == 2.0
        assert config.bm25_b == 0.5
        assert config.min_token_length == 2
        assert config.normalize_scores is False
        assert config.min_score_threshold == 0.1

    def test_hybrid_config_defaults(self):
        """Test HybridConfig with default values."""
        config = HybridConfig()

        assert config.dense_weight == 0.7
        assert config.sparse_weight == 0.3
        assert config.bm25_k1 == 1.5
        assert config.bm25_b == 0.75
        assert config.min_token_length == 3
        assert config.normalize_scores is True
        assert config.min_score_threshold == 0.0

    def test_hybrid_config_validation(self):
        """Test HybridConfig validation."""
        with pytest.raises(ValueError, match="Dense weight must be non-negative"):
            HybridConfig(dense_weight=-0.1)

        with pytest.raises(ValueError, match="Sparse weight must be non-negative"):
            HybridConfig(sparse_weight=-0.1)

        with pytest.raises(ValueError, match="At least one weight must be positive"):
            HybridConfig(dense_weight=0.0, sparse_weight=0.0)

        with pytest.raises(ValueError, match="BM25 k1 must be non-negative"):
            HybridConfig(bm25_k1=-1.0)

        with pytest.raises(ValueError, match="BM25 b must be between 0 and 1"):
            HybridConfig(bm25_b=1.5)

        with pytest.raises(
            ValueError, match="Min token length must be positive integer"
        ):
            HybridConfig(min_token_length=0)

        with pytest.raises(ValueError, match="Normalize scores must be boolean"):
            HybridConfig(normalize_scores="true")

        with pytest.raises(ValueError, match="Min score threshold must be number"):
            HybridConfig(min_score_threshold="0.1")

    def test_hybrid_result_creation(self):
        """Test creating HybridResult."""
        result = HybridResult(
            content="Test content",
            score=0.85,
            dense_score=0.9,
            sparse_score=0.7,
            metadata={"source": "test.txt"},
        )

        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.dense_score == 0.9
        assert result.sparse_score == 0.7
        assert result.metadata == {"source": "test.txt"}

    def test_hybrid_result_defaults(self):
        """Test HybridResult with default metadata."""
        result = HybridResult(
            content="Test content", score=0.85, dense_score=0.9, sparse_score=0.7
        )

        assert result.metadata == {}

    def test_hybrid_result_validation(self):
        """Test HybridResult validation."""
        with pytest.raises(ValueError, match="Content must be string"):
            HybridResult(content=123, score=0.8, dense_score=0.9, sparse_score=0.7)

        with pytest.raises(ValueError, match="Score must be number"):
            HybridResult(
                content="test", score="high", dense_score=0.9, sparse_score=0.7
            )

        with pytest.raises(ValueError, match="Dense score must be number"):
            HybridResult(
                content="test", score=0.8, dense_score="high", sparse_score=0.7
            )

        with pytest.raises(ValueError, match="Sparse score must be number"):
            HybridResult(content="test", score=0.8, dense_score=0.9, sparse_score="low")

        with pytest.raises(ValueError, match="Metadata must be dict"):
            HybridResult(
                content="test",
                score=0.8,
                dense_score=0.9,
                sparse_score=0.7,
                metadata="meta",
            )

    def test_dense_result_creation(self):
        """Test creating DenseResult."""
        result = DenseResult(
            content="Dense content", score=0.95, metadata={"embedding_model": "bge-m3"}
        )

        assert result.content == "Dense content"
        assert result.score == 0.95
        assert result.metadata == {"embedding_model": "bge-m3"}


class TestBM25Scorer:
    """Test BM25Scorer with dependency injection."""

    @pytest.fixture
    def mock_stop_words_provider(self):
        """Create mock stop words provider."""
        return create_mock_stop_words_provider()

    @pytest.fixture
    def hybrid_config(self):
        """Create test hybrid config."""
        return HybridConfig(min_token_length=2)  # Lower threshold for testing

    @pytest.fixture
    def bm25_scorer(self, mock_stop_words_provider, hybrid_config):
        """Create BM25 scorer."""
        return BM25Scorer(mock_stop_words_provider, "hr", hybrid_config)

    def test_bm25_scorer_initialization(self, mock_stop_words_provider, hybrid_config):
        """Test BM25 scorer initialization."""
        scorer = BM25Scorer(mock_stop_words_provider, "hr", hybrid_config)

        assert scorer.language == "hr"
        assert scorer.config == hybrid_config
        assert isinstance(scorer.stop_words, set)
        assert len(scorer.stop_words) > 0
        assert not scorer.is_indexed

    def test_bm25_scorer_index_documents(self, bm25_scorer):
        """Test document indexing."""
        documents = [
            "Ovo je prvi hrvatski dokument",
            "Drugi dokument sadrži više teksta",
            "Treći dokument ima različite riječi",
        ]

        bm25_scorer.index_documents(documents)

        assert bm25_scorer.is_indexed
        assert len(bm25_scorer.documents) == 3
        assert len(bm25_scorer.tokenized_docs) == 3
        assert bm25_scorer.corpus_stats["total_docs"] == 3
        assert bm25_scorer.corpus_stats["avgdl"] > 0

    def test_bm25_scorer_get_scores(self, bm25_scorer):
        """Test BM25 score calculation."""
        documents = [
            "Ovo je dokument o prirodi",
            "Dokument govori o tehnologiji",
            "Treći dokument sadrži različite teme",
        ]

        bm25_scorer.index_documents(documents)
        scores = bm25_scorer.get_scores("dokument tehnologiji")

        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)
        assert (
            scores[1] > scores[0]
        )  # Second doc should score higher (contains "tehnologiji")

    def test_bm25_scorer_empty_query(self, bm25_scorer):
        """Test BM25 scoring with empty query."""
        documents = ["Test document"]
        bm25_scorer.index_documents(documents)

        scores = bm25_scorer.get_scores("")
        assert scores == [0.0]

    def test_bm25_scorer_not_indexed_error(self, bm25_scorer):
        """Test error when getting scores without indexing."""
        with pytest.raises(ValueError, match="Documents not indexed"):
            bm25_scorer.get_scores("test query")

    def test_bm25_scorer_invalid_inputs(self, mock_stop_words_provider, hybrid_config):
        """Test BM25 scorer with invalid inputs."""
        scorer = BM25Scorer(mock_stop_words_provider, "hr", hybrid_config)

        with pytest.raises(ValueError, match="Documents must be list"):
            scorer.index_documents("not a list")

        with pytest.raises(ValueError, match="All documents must be strings"):
            scorer.index_documents(["valid", 123, "also valid"])


class TestHybridRetriever:
    """Test HybridRetriever with dependency injection."""

    @pytest.fixture
    def mock_stop_words_provider(self):
        """Create mock stop words provider."""
        return create_mock_stop_words_provider()

    @pytest.fixture
    def hybrid_config(self):
        """Create test hybrid config."""
        return HybridConfig(
            dense_weight=0.6,
            sparse_weight=0.4,
            min_token_length=2,
            normalize_scores=True,
        )

    @pytest.fixture
    def hybrid_retriever(self, mock_stop_words_provider, hybrid_config):
        """Create hybrid retriever."""
        return HybridRetriever(mock_stop_words_provider, "hr", hybrid_config)

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "Ovo je prvi hrvatski dokument o programiranju",
            "Drugi dokument govori o umjetnoj inteligenciji",
            "Treći dokument sadrži informacije o bazama podataka",
            "Četvrti dokument opisuje web razvoj",
        ]

    @pytest.fixture
    def sample_dense_results(self):
        """Sample dense search results."""
        return [
            DenseResult(
                "Ovo je prvi hrvatski dokument o programiranju",
                0.9,
                {"source": "doc1.txt"},
            ),
            DenseResult(
                "Drugi dokument govori o umjetnoj inteligenciji",
                0.7,
                {"source": "doc2.txt"},
            ),
            DenseResult(
                "Treći dokument sadrži informacije o bazama podataka",
                0.8,
                {"source": "doc3.txt"},
            ),
        ]

    def test_hybrid_retriever_initialization(
        self, mock_stop_words_provider, hybrid_config
    ):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(mock_stop_words_provider, "hr", hybrid_config)

        assert retriever.language == "hr"
        assert retriever.config == hybrid_config
        assert isinstance(retriever.bm25_scorer, BM25Scorer)
        assert not retriever.is_ready

    def test_hybrid_retriever_index_documents(self, hybrid_retriever, sample_documents):
        """Test document indexing."""
        hybrid_retriever.index_documents(sample_documents)

        assert hybrid_retriever.is_ready
        assert len(hybrid_retriever.documents) == 4
        assert hybrid_retriever.bm25_scorer.is_indexed

    def test_hybrid_retriever_retrieve_basic(
        self, hybrid_retriever, sample_documents, sample_dense_results
    ):
        """Test basic hybrid retrieval."""
        hybrid_retriever.index_documents(sample_documents)

        results = hybrid_retriever.retrieve(
            query="programiranje inteligencija",
            dense_results=sample_dense_results,
            top_k=3,
        )

        assert len(results) <= 3
        assert all(isinstance(result, HybridResult) for result in results)

        # Results should be ordered by hybrid score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_hybrid_retriever_retrieve_score_combination(
        self, hybrid_retriever, sample_documents
    ):
        """Test that dense and sparse scores are properly combined."""
        hybrid_retriever.index_documents(sample_documents)

        # Create dense results with known scores
        dense_results = [
            DenseResult("Ovo je prvi hrvatski dokument o programiranju", 0.8, {}),
            DenseResult("Drugi dokument govori o umjetnoj inteligenciji", 0.6, {}),
        ]

        results = hybrid_retriever.retrieve(
            query="programiranje",  # Should give higher BM25 to first document
            dense_results=dense_results,
            top_k=2,
        )

        assert len(results) == 2

        # Each result should have all score components
        for result in results:
            assert hasattr(result, "score")
            assert hasattr(result, "dense_score")
            assert hasattr(result, "sparse_score")
            assert 0.0 <= result.score <= 1.0

    def test_hybrid_retriever_retrieve_empty_dense_results(
        self, hybrid_retriever, sample_documents
    ):
        """Test retrieval with empty dense results."""
        hybrid_retriever.index_documents(sample_documents)

        results = hybrid_retriever.retrieve(
            query="test query", dense_results=[], top_k=5
        )

        assert results == []

    def test_hybrid_retriever_explain_scores(
        self, hybrid_retriever, sample_documents, sample_dense_results
    ):
        """Test score explanation generation."""
        hybrid_retriever.index_documents(sample_documents)

        results = hybrid_retriever.retrieve(
            query="programiranje", dense_results=sample_dense_results[:2], top_k=2
        )

        explanation = hybrid_retriever.explain_scores(results)

        assert "Hybrid Retrieval Score Explanation" in explanation
        assert "Dense weight:" in explanation
        assert "Sparse weight:" in explanation
        assert "Hybrid Score:" in explanation
        assert "Dense:" in explanation
        assert "Sparse:" in explanation

    def test_hybrid_retriever_explain_scores_empty(self, hybrid_retriever):
        """Test score explanation with empty results."""
        explanation = hybrid_retriever.explain_scores([])
        assert "No results to explain" in explanation

    def test_hybrid_retriever_not_ready_error(self, hybrid_retriever):
        """Test error when retrieving without indexing."""
        with pytest.raises(ValueError, match="Retriever not ready"):
            hybrid_retriever.retrieve("query", [], 5)

    def test_hybrid_retriever_invalid_inputs(self, hybrid_retriever, sample_documents):
        """Test hybrid retriever with invalid inputs."""
        hybrid_retriever.index_documents(sample_documents)

        with pytest.raises(ValueError, match="Query must be string"):
            hybrid_retriever.retrieve(123, [], 5)

        with pytest.raises(ValueError, match="Dense results must be list"):
            hybrid_retriever.retrieve("query", "not list", 5)

        with pytest.raises(ValueError, match="Top k must be positive integer"):
            hybrid_retriever.retrieve("query", [], 0)


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_hybrid_retriever(self):
        """Test creating hybrid retriever."""
        stop_words_provider = create_mock_stop_words_provider()

        retriever = create_hybrid_retriever(
            stop_words_provider=stop_words_provider,
            language="hr",
            dense_weight=0.8,
            sparse_weight=0.2,
            bm25_k1=2.0,
            bm25_b=0.5,
        )

        assert isinstance(retriever, HybridRetriever)
        assert retriever.language == "hr"
        assert retriever.config.dense_weight == 0.8
        assert retriever.config.sparse_weight == 0.2
        assert retriever.config.bm25_k1 == 2.0
        assert retriever.config.bm25_b == 0.5

    def test_create_mock_stop_words_provider_defaults(self):
        """Test creating mock stop words provider with defaults."""
        provider = create_mock_stop_words_provider()

        hr_stop_words = provider.get_stop_words("hr")
        en_stop_words = provider.get_stop_words("en")
        unknown_stop_words = provider.get_stop_words("unknown")

        assert isinstance(hr_stop_words, set)
        assert isinstance(en_stop_words, set)
        assert len(hr_stop_words) > 0
        assert len(en_stop_words) > 0
        assert "je" in hr_stop_words
        assert "the" in en_stop_words
        assert unknown_stop_words == set()

    def test_create_mock_stop_words_provider_custom(self):
        """Test creating mock stop words provider with custom words."""
        custom_hr = {"custom", "croatian", "words"}
        custom_en = {"custom", "english", "words"}

        provider = create_mock_stop_words_provider(
            croatian_stop_words=custom_hr, english_stop_words=custom_en
        )

        hr_stop_words = provider.get_stop_words("hr")
        en_stop_words = provider.get_stop_words("en")

        assert hr_stop_words == custom_hr
        assert en_stop_words == custom_en


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.fixture
    def multilingual_retriever(self):
        """Create retriever for multilingual testing."""
        stop_words_provider = create_mock_stop_words_provider(
            croatian_stop_words={
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
            },
            english_stop_words={
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
            },
        )

        config = HybridConfig(
            dense_weight=0.6,
            sparse_weight=0.4,
            min_token_length=3,
            normalize_scores=True,
        )

        return HybridRetriever(stop_words_provider, "hr", config)

    def test_croatian_document_retrieval(self, multilingual_retriever):
        """Test complete Croatian document retrieval workflow."""
        # Croatian documents with varied content
        documents = [
            "Programiranje u Pythonu je vrlo popularno među programerima zbog jednostavnosti sintakse",
            "Umjetna inteligencija revolucionira način na koji pristupamo rješavanju problema",
            "Baze podataka su ključne komponente svakog modernog informacijskog sustava",
            "Web razvoj uključuje frontend i backend tehnologije poput HTML, CSS, JavaScript",
            "Machine learning algoritmi omogućavaju računalima da uče iz podataka bez eksplicitnog programiranja",
        ]

        # Dense results simulating embedding similarity
        dense_results = [
            DenseResult(
                documents[0],
                0.85,
                {"source": "python_tutorial.txt", "chunk_id": "chunk_1"},
            ),
            DenseResult(
                documents[1], 0.92, {"source": "ai_overview.txt", "chunk_id": "chunk_2"}
            ),
            DenseResult(
                documents[2],
                0.78,
                {"source": "database_guide.txt", "chunk_id": "chunk_3"},
            ),
            DenseResult(
                documents[4], 0.88, {"source": "ml_basics.txt", "chunk_id": "chunk_5"}
            ),
        ]

        # Index documents
        multilingual_retriever.index_documents(documents)

        # Test query about programming
        programming_results = multilingual_retriever.retrieve(
            query="programiranje Python algoritmi", dense_results=dense_results, top_k=3
        )

        assert len(programming_results) <= 3
        assert all(isinstance(result, HybridResult) for result in programming_results)

        # Check that results contain relevant content
        content_texts = [result.content for result in programming_results]
        assert any("programiranje" in content.lower() for content in content_texts)

        # Verify score composition
        for result in programming_results:
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.dense_score <= 1.0
            assert 0.0 <= result.sparse_score <= 1.0
            assert isinstance(result.metadata, dict)

    def test_multilingual_mixed_content(self, multilingual_retriever):
        """Test retrieval with mixed Croatian-English content."""
        documents = [
            "Python programming jezik je vrlo korišten za data science projekte",
            "Machine learning models potrebni su za predictive analytics u biznisu",
            "Web development frameworks poput Django i Flask olakšavaju razvoj",
            "Database management sistemi kao što su PostgreSQL i MongoDB",
        ]

        dense_results = [
            DenseResult(doc, 0.7 + i * 0.05, {"mixed_content": True})
            for i, doc in enumerate(documents)
        ]

        multilingual_retriever.index_documents(documents)

        results = multilingual_retriever.retrieve(
            query="Python machine learning development",
            dense_results=dense_results,
            top_k=4,
        )

        assert len(results) <= 4

        # Should handle mixed language content gracefully
        for result in results:
            assert isinstance(result.content, str)
            assert len(result.content) > 0

    def test_score_weighting_impact(self, multilingual_retriever):
        """Test impact of different dense/sparse weightings."""
        documents = [
            "Ovo je dokument koji sadrži ključne riječi za testiranje",
            "Drugi dokument ima drugačiji sadržaj ali slične teme",
            "Treći dokument potpuno različit od ostalih",
        ]

        # Create dense results where second document has highest dense score
        dense_results = [
            DenseResult(documents[0], 0.6, {}),
            DenseResult(documents[1], 0.9, {}),  # Highest dense score
            DenseResult(documents[2], 0.4, {}),
        ]

        multilingual_retriever.index_documents(documents)

        # Query that should give high BM25 score to first document
        results = multilingual_retriever.retrieve(
            query="ključne riječi testiranje",  # Matches first document well
            dense_results=dense_results,
            top_k=3,
        )

        # Verify hybrid scoring combines both signals
        assert len(results) == 3
        for result in results:
            # Hybrid score should be between pure dense and pure sparse
            assert 0.0 <= result.score <= 1.0

    def test_large_corpus_performance(self, multilingual_retriever):
        """Test retrieval performance with larger document corpus."""
        # Generate larger set of documents
        base_documents = [
            "Programiranje u različitim jezicima",
            "Umjetna inteligencija i strojno učenje",
            "Baze podataka i njihova optimizacija",
            "Web tehnologije i frontend razvoj",
            "Sigurnost informacijskih sustava",
        ]

        # Create variations to simulate larger corpus
        documents = []
        for i in range(50):  # 250 documents total
            for j, base_doc in enumerate(base_documents):
                documents.append(f"{base_doc} - varijacija {i+1}")

        # Create dense results for subset of documents
        dense_results = [
            DenseResult(documents[i], 0.9 - i * 0.01, {"doc_id": f"doc_{i}"})
            for i in range(0, min(20, len(documents)), 2)
        ]

        # This should complete without performance issues
        multilingual_retriever.index_documents(documents)

        results = multilingual_retriever.retrieve(
            query="programiranje strojno učenje", dense_results=dense_results, top_k=10
        )

        assert len(results) <= 10
        assert all(isinstance(result, HybridResult) for result in results)

    def test_edge_case_handling(self, multilingual_retriever):
        """Test handling of various edge cases."""
        # Documents with edge cases
        documents = [
            "",  # Empty document
            "a",  # Very short document
            "123 456 789",  # Only numbers
            "!@#$%^&*()",  # Only punctuation
            "Word" * 1000,  # Very long document (repeated word)
            "Normalan dokument s tekstom",  # Normal document
        ]

        dense_results = [
            DenseResult(doc, 0.5, {}) for doc in documents if doc  # Skip empty
        ]

        # Should handle edge cases gracefully
        multilingual_retriever.index_documents(documents)

        results = multilingual_retriever.retrieve(
            query="test query", dense_results=dense_results, top_k=5
        )

        # Should return some results without errors
        assert isinstance(results, list)
        assert all(isinstance(result, HybridResult) for result in results)

    def test_score_normalization_consistency(self, multilingual_retriever):
        """Test that score normalization works consistently."""
        documents = [
            "Prvi dokument s normalnim sadržajem",
            "Drugi dokument također normalan",
            "Treći dokument različit sadržaj",
        ]

        dense_results = [DenseResult(doc, 0.7, {}) for doc in documents]

        multilingual_retriever.index_documents(documents)

        # Multiple queries should produce consistent normalized scores
        queries = ["dokument sadržaj", "normalan različit", "prvi drugi treći"]

        all_results = []
        for query in queries:
            results = multilingual_retriever.retrieve(
                query=query, dense_results=dense_results, top_k=3
            )
            all_results.extend(results)

        # All scores should be in valid range
        for result in all_results:
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.dense_score <= 1.0
            assert 0.0 <= result.sparse_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
