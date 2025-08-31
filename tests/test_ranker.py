"""
Unit tests for ranker module.
Tests result ranking and filtering algorithms for Croatian content.
"""

from unittest.mock import Mock

import pytest

from src.retrieval.query_processor import ProcessedQuery, QueryType
from src.retrieval.ranker import (
    CroatianResultRanker,
    RankedDocument,
    RankingConfig,
    RankingMethod,
    RankingSignal,
    create_result_ranker,
)


class TestRankingConfig:
    """Test ranking configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RankingConfig()

        assert config.method == RankingMethod.CROATIAN_ENHANCED
        assert config.enable_diversity is True
        assert config.diversity_threshold == 0.8
        assert config.boost_recent is False
        assert config.boost_authoritative is True
        assert config.content_length_factor is True
        assert config.keyword_density_factor is True
        assert config.croatian_specific_boost is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RankingConfig(
            method=RankingMethod.BM25,
            enable_diversity=False,
            diversity_threshold=0.9,
            boost_authoritative=False,
        )

        assert config.method == RankingMethod.BM25
        assert config.enable_diversity is False
        assert config.diversity_threshold == 0.9
        assert config.boost_authoritative is False


class TestRankingSignal:
    """Test ranking signal structure."""

    def test_ranking_signal_creation(self):
        """Test creating ranking signal."""
        signal = RankingSignal(name="test_signal", score=0.85, weight=0.3, metadata={"test": True})

        assert signal.name == "test_signal"
        assert signal.score == 0.85
        assert signal.weight == 0.3
        assert signal.metadata["test"] is True


class TestCroatianResultRanker:
    """Test Croatian result ranker functionality."""

    @pytest.fixture
    def ranker(self):
        """Create ranker for testing."""
        return CroatianResultRanker()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for ranking tests."""
        return [
            {
                "id": "doc1",
                "content": "Zagreb je glavni grad Hrvatske s oko 800.000 stanovnika.",
                "metadata": {
                    "source": "zagreb_wiki.txt",
                    "title": "Zagreb - glavni grad",
                    "language": "hr",
                    "content_type": "encyclopedia",
                },
                "relevance_score": 0.9,
            },
            {
                "id": "doc2",
                "content": "Split je drugi najveći grad u Hrvatskoj nakon Zagreba.",
                "metadata": {
                    "source": "cities.txt",
                    "title": "Hrvatští gradovi",
                    "language": "hr",
                },
                "relevance_score": 0.7,
            },
            {
                "id": "doc3",
                "content": "Dubrovnik biser Jadrana poznat po UNESCO baštini.",
                "metadata": {
                    "source": "tourism.txt",
                    "language": "hr",
                    "content_type": "blog",
                },
                "relevance_score": 0.8,
            },
            {
                "id": "doc4",
                "content": "Kratka bilješka o gradu.",
                "metadata": {"source": "notes.txt", "language": "hr"},
                "relevance_score": 0.4,
            },
        ]

    @pytest.fixture
    def sample_query(self):
        """Sample processed query."""
        return ProcessedQuery(
            original="Koji je glavni grad Hrvatske?",
            processed="koji je glavni grad hrvatske",
            query_type=QueryType.FACTUAL,
            keywords=["glavni", "grad", "hrvatske"],
            expanded_terms=["glavnog", "grada"],
            filters={"language": "hr"},
            confidence=0.8,
            metadata={},
        )

    def test_initialization(self):
        """Test ranker initialization."""
        ranker = CroatianResultRanker()

        assert ranker.config.method == RankingMethod.CROATIAN_ENHANCED
        assert len(ranker.croatian_importance_words) > 0
        assert "zagreb" in ranker.croatian_importance_words
        assert "positive" in ranker.quality_indicators
        assert "negative" in ranker.quality_indicators

    def test_empty_documents_ranking(self, ranker, sample_query):
        """Test ranking with empty document list."""
        result = ranker.rank_documents([], sample_query)

        assert result == []

    def test_basic_document_ranking(self, ranker, sample_documents, sample_query):
        """Test basic document ranking functionality."""
        ranked_docs = ranker.rank_documents(sample_documents, sample_query)

        # Should return same number of documents
        assert len(ranked_docs) == len(sample_documents)

        # All should be RankedDocument instances
        for doc in ranked_docs:
            assert isinstance(doc, RankedDocument)
            assert hasattr(doc, "final_score")
            assert hasattr(doc, "ranking_signals")
            assert hasattr(doc, "rank")

        # Should be sorted by final score (descending)
        scores = [doc.final_score for doc in ranked_docs]
        assert scores == sorted(scores, reverse=True)

        # Ranks should be sequential starting from 1
        ranks = [doc.rank for doc in ranked_docs]
        assert ranks == list(range(1, len(ranked_docs) + 1))


class TestRankingSignals:
    """Test individual ranking signals."""

    @pytest.fixture
    def ranker(self):
        return CroatianResultRanker()

    @pytest.fixture
    def factual_query(self):
        return ProcessedQuery(
            original="Koliko ima stanovnika Zagreb?",
            processed="koliko ima stanovnika zagreb",
            query_type=QueryType.FACTUAL,
            keywords=["koliko", "stanovnika", "zagreb"],
            expanded_terms=[],
            filters={"language": "hr"},
            confidence=0.8,
            metadata={},
        )

    def test_keyword_relevance_signal(self, ranker, factual_query):
        """Test keyword relevance scoring."""
        # Content with good keyword matches
        good_content = "Zagreb ima oko 800.000 stanovnika prema zadnjem popisu."
        signal = ranker._calculate_keyword_relevance(good_content, factual_query)

        assert signal.name == "keyword_relevance"
        assert signal.score > 0.0
        assert signal.weight == 0.25
        assert "matches" in signal.metadata
        assert "coverage" in signal.metadata

        # Should find keywords
        assert signal.metadata["matches"] > 0
        assert signal.metadata["coverage"] > 0

        # Content with no keyword matches
        poor_content = "Ovo je tekst koji ne sadrži relevantne termine."
        poor_signal = ranker._calculate_keyword_relevance(poor_content, factual_query)

        assert poor_signal.score == 0.0
        assert poor_signal.metadata["matches"] == 0

    def test_content_quality_signal(self, ranker):
        """Test content quality assessment."""
        # High quality content
        quality_content = """
        Zagreb je glavni grad Republike Hrvatske s detaljnim opisom.
        Grad ima bogatu povijest i predstavlja kulturno središte zemlje.
        Prema službenim podacima, Zagreb broji oko 800.000 stanovnika.
        """

        quality_signal = ranker._calculate_content_quality(
            quality_content, {"title": "Zagreb - detaljni vodič", "author": "Stručnjak"}
        )

        assert quality_signal.name == "content_quality"
        assert quality_signal.score > 0.5
        assert quality_signal.metadata["has_title"] is True

        # Poor quality content
        poor_content = "Zagreb možda ima puno ljudi."
        poor_signal = ranker._calculate_content_quality(poor_content, {})

        assert poor_signal.score < quality_signal.score
        assert poor_signal.metadata["has_title"] is False

    def test_croatian_relevance_signal(self, ranker, factual_query):
        """Test Croatian-specific relevance scoring."""
        # Content rich in Croatian characteristics
        croatian_content = """
        Zagreb je glavni grad Republike Hrvatske. Grad se nalazi na rijeci Savi
        i predstavlja kulturno i gospodarsko središte zemlje. Poznate su zagrebačke
        atrakcije kao što su Gornji grad, katedrala i Maksimir.
        """

        signal = ranker._calculate_croatian_relevance(croatian_content, factual_query)

        assert signal.name == "croatian_relevance"
        assert signal.score > 0.0
        assert signal.weight == 0.2

        # Should detect Croatian characteristics
        metadata = signal.metadata
        assert "diacritic_density" in metadata
        assert "importance_words" in metadata

        # Content without Croatian characteristics
        non_croatian = "This is English text without Croatian features."
        poor_signal = ranker._calculate_croatian_relevance(non_croatian, factual_query)

        assert poor_signal.score < signal.score

    def test_authority_score_signal(self, ranker):
        """Test document authority scoring."""
        # Authoritative source
        auth_metadata = {
            "source": "hr.wikipedia.org/zagreb",
            "content_type": "encyclopedia",
            "title": "Zagreb",
            "author": "Stručni tim",
            "date": "2023-01-01",
        }

        auth_signal = ranker._calculate_authority_score(auth_metadata)

        assert auth_signal.name == "authority_score"
        assert auth_signal.score > 0.5
        assert auth_signal.weight == 0.1

        # Non-authoritative source
        non_auth_metadata = {"source": "random-blog.com", "content_type": "social"}

        non_auth_signal = ranker._calculate_authority_score(non_auth_metadata)

        assert non_auth_signal.score < auth_signal.score

    def test_length_appropriateness_signal(self, ranker):
        """Test content length appropriateness for different query types."""
        # Test for factual query (prefers shorter content)
        factual_query = QueryType.FACTUAL

        # Optimal length for factual
        optimal_content = "Zagreb je glavni grad Hrvatske. Ima 800.000 stanovnika."
        signal = ranker._calculate_length_appropriateness(optimal_content, factual_query)

        assert signal.score >= 0.8

        # Too long for factual
        long_content = "Zagreb je glavni grad Hrvatske. " * 50  # Very long
        long_signal = ranker._calculate_length_appropriateness(long_content, factual_query)

        assert long_signal.score < signal.score

        # Test for explanatory query (prefers medium length)
        explanatory_query = QueryType.EXPLANATORY
        medium_content = "Zagreb je glavni grad Hrvatske. " * 10  # Medium length

        exp_signal = ranker._calculate_length_appropriateness(medium_content, explanatory_query)
        assert exp_signal.score >= 0.7

    def test_query_type_match_signal(self, ranker):
        """Test query type matching."""
        # Factual content for factual query
        factual_content = "Zagreb ima 800.000 stanovnika prema popisu iz 2021. godine."
        factual_signal = ranker._calculate_query_type_match(factual_content, QueryType.FACTUAL)

        assert factual_signal.score > 0.5

        # Explanatory content for explanatory query
        explanatory_content = """
        Zagreb je postao glavni grad zbog svoje strategijske pozicije.
        Razlog tome je što se nalazi na rijeci Savi i predstavlja
        prirodno središte regije.
        """

        exp_signal = ranker._calculate_query_type_match(explanatory_content, QueryType.EXPLANATORY)
        assert exp_signal.score > 0.5

        # Comparison content for comparison query
        comparison_content = """
        Za razliku od Splita, Zagreb je veći grad.
        S druge strane, Split ima bolju klimu.
        Usporedba ovih gradova pokazuje različite prednosti.
        """

        comp_signal = ranker._calculate_query_type_match(comparison_content, QueryType.COMPARISON)
        assert comp_signal.score > 0.5


class TestSignalCombination:
    """Test signal combination and final scoring."""

    @pytest.fixture
    def ranker(self):
        return CroatianResultRanker()

    def test_signal_combination(self, ranker):
        """Test combining multiple ranking signals."""
        signals = [
            RankingSignal("signal1", 0.8, 0.3),
            RankingSignal("signal2", 0.6, 0.4),
            RankingSignal("signal3", 0.9, 0.2),
            RankingSignal("signal4", 0.5, 0.1),
        ]

        final_score = ranker._combine_ranking_signals(signals)

        # Should be weighted average
        expected_score = (0.8 * 0.3 + 0.6 * 0.4 + 0.9 * 0.2 + 0.5 * 0.1) / (0.3 + 0.4 + 0.2 + 0.1)

        assert abs(final_score - expected_score) < 0.001
        assert 0.0 <= final_score <= 1.0

    def test_empty_signals_combination(self, ranker):
        """Test combining empty signal list."""
        final_score = ranker._combine_ranking_signals([])
        assert final_score == 0.0

    def test_zero_weight_signals(self, ranker):
        """Test handling signals with zero weights."""
        signals = [
            RankingSignal("signal1", 0.8, 0.0),  # Zero weight
            RankingSignal("signal2", 0.6, 0.0),  # Zero weight
        ]

        final_score = ranker._combine_ranking_signals(signals)
        assert final_score == 0.0


class TestDiversityFiltering:
    """Test diversity filtering functionality."""

    @pytest.fixture
    def ranker(self):
        config = RankingConfig(enable_diversity=True, diversity_threshold=0.7)
        return CroatianResultRanker(config)

    def test_diversity_filtering(self, ranker):
        """Test diversity filtering removes similar documents."""
        # Create similar documents
        similar_docs = [
            RankedDocument(
                id="doc1",
                content="Zagreb je glavni grad Hrvatske s 800.000 stanovnika.",
                metadata={},
                original_score=0.9,
                final_score=0.9,
                rank=0,
                ranking_signals=[],
                ranking_metadata={},
            ),
            RankedDocument(
                id="doc2",
                content="Zagreb glavni grad Hrvatske 800000 stanovnika grad.",  # Very similar
                metadata={},
                original_score=0.85,
                final_score=0.85,
                rank=0,
                ranking_signals=[],
                ranking_metadata={},
            ),
            RankedDocument(
                id="doc3",
                content="Dubrovnik je biser Jadrana na jugu Hrvatske.",  # Different
                metadata={},
                original_score=0.8,
                final_score=0.8,
                rank=0,
                ranking_signals=[],
                ranking_metadata={},
            ),
        ]

        diverse_docs = ranker._apply_diversity_filtering(similar_docs)

        # Should keep first (highest scoring) and different document
        assert len(diverse_docs) <= len(similar_docs)

        # Should keep the highest scoring document
        assert diverse_docs[0].id == "doc1"

        # Should prefer diverse content
        diverse_ids = [doc.id for doc in diverse_docs]
        assert "doc3" in diverse_ids  # Different content should be kept

    def test_diversity_with_few_documents(self, ranker):
        """Test diversity filtering with few documents."""
        # With only 2-3 documents, should keep all
        few_docs = [
            RankedDocument("doc1", "content1", {}, 0.9, 0.9, 0, [], {}),
            RankedDocument("doc2", "content2", {}, 0.8, 0.8, 0, [], {}),
        ]

        result = ranker._apply_diversity_filtering(few_docs)

        # Should keep all documents when few are provided
        assert len(result) == len(few_docs)


class TestRankingExplanation:
    """Test ranking explanation functionality."""

    @pytest.fixture
    def sample_ranked_doc(self):
        """Create sample ranked document for explanation testing."""
        signals = [
            RankingSignal("semantic_similarity", 0.9, 0.3, {"source": "vector_search"}),
            RankingSignal("keyword_relevance", 0.7, 0.25, {"matches": 3}),
            RankingSignal("croatian_relevance", 0.8, 0.2, {"diacritic_density": 0.05}),
        ]

        return RankedDocument(
            id="doc1",
            content="Zagreb je glavni grad Hrvatske.",
            metadata={"title": "Zagreb", "source": "wikipedia"},
            original_score=0.85,
            final_score=0.82,
            rank=1,
            ranking_signals=signals,
            ranking_metadata={"method": "croatian_enhanced"},
        )

    def test_ranking_explanation(self, sample_ranked_doc):
        """Test ranking explanation generation."""
        ranker = CroatianResultRanker()

        explanation = ranker.explain_ranking(sample_ranked_doc)

        assert isinstance(explanation, str)
        assert len(explanation) > 0

        # Should contain key information
        assert "Rank #1" in explanation
        assert "Score: 0.820" in explanation
        assert "doc1" in explanation
        assert "semantic_similarity" in explanation
        assert "keyword_relevance" in explanation
        assert "Original search score: 0.850" in explanation

        # Should show signal contributions
        lines = explanation.split("\n")
        signal_lines = [
            line for line in lines if "semantic_similarity" in line or "keyword_relevance" in line
        ]
        assert len(signal_lines) >= 2


class TestFallbackRanking:
    """Test fallback ranking for error scenarios."""

    def test_fallback_ranking_creation(self):
        """Test fallback ranking when main ranking fails."""
        ranker = CroatianResultRanker()

        original_docs = [
            {
                "id": "doc1",
                "content": "content1",
                "metadata": {},
                "relevance_score": 0.9,
            },
            {
                "id": "doc2",
                "content": "content2",
                "metadata": {},
                "relevance_score": 0.7,
            },
        ]

        fallback_docs = ranker._create_fallback_ranking(original_docs)

        # Should create valid ranked documents
        assert len(fallback_docs) == len(original_docs)

        for i, doc in enumerate(fallback_docs):
            assert isinstance(doc, RankedDocument)
            assert doc.rank == i + 1
            assert doc.final_score == original_docs[i]["relevance_score"]
            assert doc.ranking_metadata["fallback"] is True


class TestCroatianSpecificFeatures:
    """Test Croatian-specific ranking features."""

    @pytest.fixture
    def ranker(self):
        return CroatianResultRanker()

    def test_croatian_importance_words_recognition(self, ranker):
        """Test recognition of Croatian importance words."""
        # Content with Croatian importance words
        important_content = "Zagreb je glavni grad Hrvatske, poznati turistički centar."

        query = ProcessedQuery(
            original="hrvatski gradovi",
            processed="hrvatski gradovi",
            query_type=QueryType.GENERAL,
            keywords=["hrvatski", "gradovi"],
            expanded_terms=[],
            filters={"language": "hr"},
            confidence=0.8,
            metadata={},
        )

        signal = ranker._calculate_croatian_relevance(important_content, query)

        # Should boost score for importance words
        assert signal.score > 0.0
        assert signal.metadata["importance_words"] > 0

    def test_cultural_references_recognition(self, ranker):
        """Test recognition of Croatian cultural references."""
        cultural_content = "Dubrovnik je biser Jadrana, UNESCO svjetska baština."

        query = ProcessedQuery(
            original="dubrovnik",
            processed="dubrovnik",
            query_type=QueryType.GENERAL,
            keywords=["dubrovnik"],
            expanded_terms=[],
            filters={"language": "hr"},
            confidence=0.8,
            metadata={},
        )

        signal = ranker._calculate_croatian_relevance(cultural_content, query)

        # Should recognize cultural references
        assert signal.score > 0.0
        assert signal.metadata["cultural_references"] > 0

    def test_diacritic_density_calculation(self, ranker):
        """Test Croatian diacritic density calculation."""
        # Text with many Croatian diacritics
        diacritic_text = "Čakovec je grad u Međimurju, poznato po šumi i žitnim poljima."

        query = ProcessedQuery(
            original="test",
            processed="test",
            query_type=QueryType.GENERAL,
            keywords=["test"],
            expanded_terms=[],
            filters={},
            confidence=0.8,
            metadata={},
        )

        signal = ranker._calculate_croatian_relevance(diacritic_text, query)

        # Should calculate diacritic density
        assert signal.metadata["diacritic_density"] > 0
        assert signal.score > 0  # Should boost score for Croatian text


class TestFactoryFunction:
    """Test factory function."""

    def test_create_result_ranker(self):
        """Test result ranker factory function."""
        ranker = create_result_ranker()

        assert isinstance(ranker, CroatianResultRanker)
        assert ranker.config.method == RankingMethod.CROATIAN_ENHANCED
        assert ranker.config.enable_diversity is True

        # Test with custom parameters
        ranker_custom = create_result_ranker(method=RankingMethod.BM25, enable_diversity=False)

        assert ranker_custom.config.method == RankingMethod.BM25
        assert ranker_custom.config.enable_diversity is False


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_ranking_with_malformed_documents(self):
        """Test ranking with malformed document structures."""
        ranker = CroatianResultRanker()

        malformed_docs = [
            {"id": "doc1"},  # Missing content and metadata
            {"content": "test"},  # Missing id and metadata
            {},  # Completely empty
        ]

        query = ProcessedQuery(
            original="test",
            processed="test",
            query_type=QueryType.GENERAL,
            keywords=["test"],
            expanded_terms=[],
            filters={},
            confidence=0.8,
            metadata={},
        )

        # Should not crash, should return fallback ranking
        result = ranker.rank_documents(malformed_docs, query)

        assert isinstance(result, list)
        assert len(result) == len(malformed_docs)


if __name__ == "__main__":
    pytest.main([__file__])
