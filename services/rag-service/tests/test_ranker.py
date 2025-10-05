"""
Comprehensive tests for retrieval/ranker.py - Level 3 module
Tests all data classes, pure functions, and dependency injection patterns.
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Any

from src.retrieval.ranker import (
    # Data classes
    RankingSignal,
    RankedDocument,
    LanguageFeatures,
    ProcessedQuery,
    LanguageProvider,
    # Pure functions
    calculate_keyword_relevance_score,
    calculate_content_quality_score,
    calculate_language_relevance_score,
    calculate_authority_score,
    calculate_length_appropriateness_score,
    calculate_query_type_match_score,
    combine_ranking_signals,
    apply_diversity_filtering,
    create_ranking_explanation,
    # Main class
    DocumentRanker,
    # Factory functions
    create_document_ranker,
)
from tests.conftest import (
    create_mock_ranker,
)
from src.utils.config_models import RankingConfig, RankingMethod


# ===== DATA CLASS TESTS =====

class TestRankingSignal:
    """Test RankingSignal data class."""

    def test_ranking_signal_creation(self):
        """Test basic RankingSignal creation."""
        signal = RankingSignal(
            name="keyword_relevance",
            score=0.85,
            weight=0.3,
            metadata={"matches": 5}
        )

        assert signal.name == "keyword_relevance"
        assert signal.score == 0.85
        assert signal.weight == 0.3
        assert signal.metadata == {"matches": 5}

    def test_ranking_signal_without_metadata(self):
        """Test RankingSignal creation without metadata."""
        signal = RankingSignal(
            name="content_quality",
            score=0.75,
            weight=0.2
        )

        assert signal.name == "content_quality"
        assert signal.score == 0.75
        assert signal.weight == 0.2
        assert signal.metadata is None


class TestRankedDocument:
    """Test RankedDocument data class."""

    def test_ranked_document_creation(self):
        """Test basic RankedDocument creation."""
        signals = [
            RankingSignal("keyword", 0.8, 0.3),
            RankingSignal("quality", 0.7, 0.2)
        ]

        doc = RankedDocument(
            id="doc_123",
            content="Test content",
            metadata={"title": "Test Doc"},
            original_score=0.75,
            final_score=0.85,
            rank=1,
            ranking_signals=signals,
            ranking_metadata={"method": "hybrid"}
        )

        assert doc.id == "doc_123"
        assert doc.content == "Test content"
        assert doc.metadata == {"title": "Test Doc"}
        assert doc.original_score == 0.75
        assert doc.final_score == 0.85
        assert doc.rank == 1
        assert len(doc.ranking_signals) == 2
        assert doc.ranking_metadata == {"method": "hybrid"}


class TestLanguageFeatures:
    """Test LanguageFeatures data class."""

    def test_language_features_creation(self):
        """Test LanguageFeatures creation with all fields."""
        features = LanguageFeatures(
            importance_words={"važno", "ključno"},
            quality_indicators={
                "positive": ["dobro", "kvalitetno"],
                "negative": ["loše", "neispravno"]
            },
            cultural_patterns=["hrvatska", "zagreb"],
            grammar_patterns=["\\bje\\b", "\\bsu\\b"],
            type_weights={"academic": 1.2, "news": 0.8}
        )

        assert features.importance_words == {"važno", "ključno"}
        assert features.quality_indicators["positive"] == ["dobro", "kvalitetno"]
        assert features.quality_indicators["negative"] == ["loše", "neispravno"]
        assert features.cultural_patterns == ["hrvatska", "zagreb"]
        assert features.grammar_patterns == ["\\bje\\b", "\\bsu\\b"]
        assert features.type_weights == {"academic": 1.2, "news": 0.8}


class TestProcessedQuery:
    """Test ProcessedQuery data class."""

    def test_processed_query_creation(self):
        """Test ProcessedQuery creation."""
        query = ProcessedQuery(
            text="Što je RAG?",
            keywords=["RAG", "što", "je"],
            query_type="factual",
            language="hr",
            metadata={"complexity": "simple"}
        )

        assert query.text == "Što je RAG?"
        assert query.keywords == ["RAG", "što", "je"]
        assert query.query_type == "factual"
        assert query.language == "hr"
        assert query.metadata == {"complexity": "simple"}

    def test_processed_query_without_metadata(self):
        """Test ProcessedQuery creation without metadata."""
        query = ProcessedQuery(
            text="What is RAG?",
            keywords=["RAG", "what", "is"],
            query_type="factual",
            language="en"
        )

        assert query.text == "What is RAG?"
        assert query.keywords == ["RAG", "what", "is"]
        assert query.query_type == "factual"
        assert query.language == "en"
        assert query.metadata is None


# ===== PURE FUNCTION TESTS =====

class TestKeywordRelevanceScore:
    """Test calculate_keyword_relevance_score function."""

    def test_keyword_relevance_basic(self):
        """Test basic keyword relevance calculation."""
        content = "This is a test document about RAG and AI systems"
        keywords = ["test", "RAG", "AI"]

        score, metadata = calculate_keyword_relevance_score(content, keywords)

        assert score > 0
        assert metadata["matches"] == 3
        assert metadata["unique_keywords"] == 3
        assert metadata["coverage"] == 1.0

    def test_keyword_relevance_no_keywords(self):
        """Test keyword relevance with empty keywords."""
        content = "Test content"
        keywords = []

        score, metadata = calculate_keyword_relevance_score(content, keywords)

        assert score == 0.0
        assert metadata["matches"] == 0
        assert metadata["unique_keywords"] == 0
        assert metadata["coverage"] == 0.0

    def test_keyword_relevance_empty_content(self):
        """Test keyword relevance with empty content."""
        content = ""
        keywords = ["test"]

        score, metadata = calculate_keyword_relevance_score(content, keywords)

        assert score == 0.0
        assert metadata["matches"] == 0

    def test_keyword_relevance_case_insensitive(self):
        """Test keyword relevance is case insensitive."""
        content = "TEST Content with RAG"
        keywords = ["test", "rag"]

        score, metadata = calculate_keyword_relevance_score(content, keywords)

        assert metadata["matches"] == 2
        assert metadata["unique_keywords"] == 2

    def test_keyword_relevance_with_and_without_boost(self):
        """Test keyword relevance with and without unique coverage boost."""
        content = "This is a test document with some content"
        keywords = ["test", "missing", "another"]  # Mix of found and missing keywords

        score_with_boost, metadata_with_boost = calculate_keyword_relevance_score(content, keywords, boost_unique_coverage=True)
        score_without_boost, metadata_without_boost = calculate_keyword_relevance_score(content, keywords, boost_unique_coverage=False)

        # Both should have same TF score and coverage
        assert metadata_with_boost["coverage"] == metadata_without_boost["coverage"]
        assert metadata_with_boost["tf_score"] == metadata_without_boost["tf_score"]

        # With boost should be >= without boost due to coverage component
        assert score_with_boost >= score_without_boost


class TestContentQualityScore:
    """Test calculate_content_quality_score function."""

    def test_content_quality_basic(self):
        """Test basic content quality calculation."""
        content = "This is excellent content with good quality"
        quality_indicators = {
            "positive": ["excellent", "good"],
            "negative": ["bad", "poor"]
        }
        metadata = {"title": "Test Title"}

        score, result_metadata = calculate_content_quality_score(content, quality_indicators, metadata)

        assert score > 0.5  # Should be above base score
        assert result_metadata["positive_indicators"] == 2
        assert result_metadata["negative_indicators"] == 0
        assert result_metadata["has_title"] is True

    def test_content_quality_negative_indicators(self):
        """Test content quality with negative indicators."""
        content = "This is bad content with poor quality"
        quality_indicators = {
            "positive": ["excellent", "good"],
            "negative": ["bad", "poor"]
        }
        metadata = {}

        score, result_metadata = calculate_content_quality_score(content, quality_indicators, metadata)

        assert score < 0.5  # Should be below base score
        assert result_metadata["negative_indicators"] == 2
        assert result_metadata["has_title"] is False

    def test_content_quality_length_scoring(self):
        """Test content quality length-based scoring."""
        # Optimal length content
        optimal_content = "A" * 200
        quality_indicators = {"positive": [], "negative": []}
        metadata = {}

        optimal_score, _ = calculate_content_quality_score(optimal_content, quality_indicators, metadata)

        # Too short content
        short_content = "A" * 30
        short_score, _ = calculate_content_quality_score(short_content, quality_indicators, metadata)

        assert optimal_score > short_score

    def test_content_quality_structured_content(self):
        """Test content quality with structured content boost."""
        structured_content = "• First point\n• Second point\nTitle: Important section"
        quality_indicators = {"positive": [], "negative": []}
        metadata = {}

        score_with_boost, metadata_with_boost = calculate_content_quality_score(
            structured_content, quality_indicators, metadata, structured_content_boost=True
        )
        score_without_boost, _ = calculate_content_quality_score(
            structured_content, quality_indicators, metadata, structured_content_boost=False
        )

        assert score_with_boost > score_without_boost
        assert metadata_with_boost["has_structure"] is True


class TestLanguageRelevanceScore:
    """Test calculate_language_relevance_score function."""

    @patch('src.utils.config_loader.get_language_ranking_features')
    def test_language_relevance_success(self, mock_get_features):
        """Test successful language relevance calculation."""
        # Mock configuration
        mock_features = {
            "special_characters": {
                "enabled": True,
                "characters": ["č", "ć", "š", "ž", "đ"],
                "density_factor": 50,
                "max_score": 0.3
            },
            "importance_words": {
                "enabled": True,
                "words": ["je", "su", "i"],
                "word_boost": 0.1,
                "max_score": 0.2
            },
            "cultural_patterns": {
                "enabled": True,
                "patterns": ["hrvatska", "zagreb"],
                "pattern_boost": 0.1,
                "max_score": 0.15
            },
            "grammar_patterns": {
                "enabled": True,
                "patterns": ["\\bje\\b", "\\bsu\\b"],
                "density_factor": 10,
                "max_score": 0.1
            },
            "capitalization": {
                "enabled": True,
                "proper_nouns": ["Zagreb", "Croatia"],
                "capitalization_boost": 0.05,
                "max_score": 0.1
            },
            "vocabulary_patterns": {
                "enabled": True,
                "patterns": ["\\btako\\b", "\\bda\\b"],
                "pattern_boost": 0.1,
                "max_score": 0.15
            }
        }
        mock_get_features.return_value = mock_features

        content = "Ovo je tekst o Zagrebu i Hrvatskoj. Tako je da je dobro."
        language = "hr"
        language_features = LanguageFeatures(
            importance_words=set(),
            quality_indicators={},
            cultural_patterns=[],
            grammar_patterns=[],
            type_weights={}
        )

        score, metadata = calculate_language_relevance_score(content, language, language_features)

        assert score > 0
        assert metadata["config_driven"] is True
        assert metadata["language"] == "hr"
        assert "features_detected" in metadata

    @patch('src.utils.config_loader.get_language_ranking_features')
    def test_language_relevance_config_failure(self, mock_get_features):
        """Test language relevance with configuration failure."""
        mock_get_features.side_effect = Exception("Config not found")

        content = "Test content"
        language = "hr"
        language_features = LanguageFeatures(
            importance_words=set(),
            quality_indicators={},
            cultural_patterns=[],
            grammar_patterns=[],
            type_weights={}
        )

        score, metadata = calculate_language_relevance_score(content, language, language_features)

        assert score == 0.0
        assert metadata["config_driven"] is False
        assert metadata["fallback_reason"] == "Configuration system unavailable"


class TestAuthorityScore:
    """Test calculate_authority_score function."""

    def test_authority_score_basic(self):
        """Test basic authority score calculation."""
        metadata = {
            "source": "wikipedia.org",
            "content_type": "academic",
            "title": "Test Article",
            "author": "Expert Author"
        }
        type_weights = {"academic": 1.2, "news": 0.8}
        authoritative_sources = ["wikipedia", "gov.hr", "akademija"]

        score, result_metadata = calculate_authority_score(metadata, type_weights, authoritative_sources)

        assert score > 0.5
        assert result_metadata["source_authority"] > 0
        assert result_metadata["type_multiplier"] == 1.2

    def test_authority_score_educational_source(self):
        """Test authority score for educational sources."""
        metadata = {
            "source": "sveučilište.edu",
            "content_type": "educational"
        }
        type_weights = {}
        authoritative_sources = ["sveučilište", ".edu"]

        score, result_metadata = calculate_authority_score(metadata, type_weights, authoritative_sources)

        assert result_metadata["source_authority"] == 0.2

    def test_authority_score_metadata_completeness(self):
        """Test authority score based on metadata completeness."""
        complete_metadata = {
            "title": "Complete Article",
            "author": "Author Name",
            "date": "2024-01-01",
            "source": "reliable.com",
            "language": "hr"
        }
        incomplete_metadata = {"title": "Incomplete"}

        type_weights = {}
        authoritative_sources = []

        complete_score, complete_result = calculate_authority_score(complete_metadata, type_weights, authoritative_sources)
        incomplete_score, incomplete_result = calculate_authority_score(incomplete_metadata, type_weights, authoritative_sources)

        assert complete_result["metadata_completeness"] > incomplete_result["metadata_completeness"]
        assert complete_score > incomplete_score


class TestLengthAppropriatenessScore:
    """Test calculate_length_appropriateness_score function."""

    def test_length_appropriateness_optimal(self):
        """Test length appropriateness for optimal length."""
        content = "A" * 300  # Optimal length
        query_type = "factual"
        optimal_ranges = {"factual": (200, 400), "explanatory": (500, 1000)}

        score, metadata = calculate_length_appropriateness_score(content, query_type, optimal_ranges)

        assert score == 1.0
        assert metadata["length_category"] == "optimal"
        assert metadata["content_length"] == 300

    def test_length_appropriateness_too_short(self):
        """Test length appropriateness for too short content."""
        content = "A" * 50  # Too short
        query_type = "factual"
        optimal_ranges = {"factual": (200, 400)}

        score, metadata = calculate_length_appropriateness_score(content, query_type, optimal_ranges)

        assert score < 1.0
        assert metadata["length_category"] == "too_short"

    def test_length_appropriateness_too_long(self):
        """Test length appropriateness for too long content."""
        content = "A" * 800  # Too long
        query_type = "factual"
        optimal_ranges = {"factual": (200, 400)}

        score, metadata = calculate_length_appropriateness_score(content, query_type, optimal_ranges)

        assert score < 1.0
        assert metadata["length_category"] == "too_long"

    def test_length_appropriateness_unknown_query_type(self):
        """Test length appropriateness for unknown query type."""
        content = "A" * 300
        query_type = "unknown"
        optimal_ranges = {"factual": (200, 400)}

        score, metadata = calculate_length_appropriateness_score(content, query_type, optimal_ranges)

        assert metadata["optimal_range"] == "100-500 chars"  # Default range


class TestQueryTypeMatchScore:
    """Test calculate_query_type_match_score function."""

    def test_query_type_match_patterns(self):
        """Test query type match with pattern matching."""
        content = "What is the definition of RAG? It means retrieval augmented generation."
        query_type = "factual"
        type_patterns = {
            "factual": ["what", "definition", "means"],
            "comparison": ["versus", "compared", "difference"]
        }

        score, metadata = calculate_query_type_match_score(content, query_type, type_patterns)

        assert score > 0.5
        assert metadata["pattern_matches"] > 0
        assert metadata["query_type"] == "factual"

    def test_query_type_match_comparison_structure(self):
        """Test query type match for comparison with structural indicators."""
        content = "• Option A vs Option B\n• First choice compared to second"
        query_type = "comparison"
        type_patterns = {"comparison": ["versus", "compared"]}

        score, metadata = calculate_query_type_match_score(content, query_type, type_patterns)

        assert score > 0.5
        assert metadata["structural_indicators"] is True

    def test_query_type_match_explanatory_structure(self):
        """Test query type match for explanatory with structural indicators."""
        content = "1. First step in process\n2. Second step\nPrvo učitajte podatke, drugo analizirajte ih"
        query_type = "explanatory"
        type_patterns = {"explanatory": ["step", "process"]}

        score, metadata = calculate_query_type_match_score(content, query_type, type_patterns)

        assert score > 0.5
        assert metadata["structural_indicators"] is True


class TestCombineRankingSignals:
    """Test combine_ranking_signals function."""

    def test_combine_ranking_signals_basic(self):
        """Test basic ranking signals combination."""
        signals = [
            RankingSignal("signal1", 0.8, 0.3),
            RankingSignal("signal2", 0.6, 0.4),
            RankingSignal("signal3", 0.9, 0.3)
        ]

        final_score = combine_ranking_signals(signals)

        # Weighted average: (0.8*0.3 + 0.6*0.4 + 0.9*0.3) / (0.3+0.4+0.3) = 0.75
        expected = (0.8*0.3 + 0.6*0.4 + 0.9*0.3) / 1.0
        assert abs(final_score - expected) < 0.001

    def test_combine_ranking_signals_empty(self):
        """Test combining empty signals list."""
        signals = []
        final_score = combine_ranking_signals(signals)
        assert final_score == 0.0

    def test_combine_ranking_signals_zero_weights(self):
        """Test combining signals with zero weights."""
        signals = [
            RankingSignal("signal1", 0.8, 0.0),
            RankingSignal("signal2", 0.6, 0.0)
        ]

        final_score = combine_ranking_signals(signals)
        assert final_score == 0.0


class TestDiversityFiltering:
    """Test apply_diversity_filtering function."""

    def test_diversity_filtering_basic(self):
        """Test basic diversity filtering."""
        docs = [
            RankedDocument("1", "test content about python", {}, 0.9, 0.9, 1, [], {}),
            RankedDocument("2", "test content about python programming", {}, 0.8, 0.8, 2, [], {}),
            RankedDocument("3", "completely different javascript content", {}, 0.7, 0.7, 3, [], {})
        ]

        filtered = apply_diversity_filtering(docs, diversity_threshold=0.7, min_results=2)

        # Should keep first and third (different content)
        assert len(filtered) >= 2
        assert filtered[0].id == "1"
        assert any(doc.id == "3" for doc in filtered)

    def test_diversity_filtering_min_results(self):
        """Test diversity filtering respects minimum results."""
        docs = [
            RankedDocument("1", "same content", {}, 0.9, 0.9, 1, [], {}),
            RankedDocument("2", "same content", {}, 0.8, 0.8, 2, [], {})
        ]

        filtered = apply_diversity_filtering(docs, diversity_threshold=0.1, min_results=2)

        # Should keep both despite similarity due to min_results
        assert len(filtered) == 2

    def test_diversity_filtering_small_list(self):
        """Test diversity filtering with list smaller than min_results."""
        docs = [
            RankedDocument("1", "content", {}, 0.9, 0.9, 1, [], {})
        ]

        filtered = apply_diversity_filtering(docs, min_results=3)

        # Should return all documents unchanged
        assert len(filtered) == 1
        assert filtered[0].id == "1"


class TestCreateRankingExplanation:
    """Test create_ranking_explanation function."""

    def test_create_ranking_explanation(self):
        """Test creating ranking explanation."""
        signals = [
            RankingSignal("keyword_relevance", 0.8, 0.3, {"matches": 5}),
            RankingSignal("content_quality", 0.6, 0.2, {"quality": "good"})
        ]

        doc = RankedDocument(
            id="doc_123",
            content="Test content",
            metadata={"title": "Test"},
            original_score=0.75,
            final_score=0.85,
            rank=1,
            ranking_signals=signals,
            ranking_metadata={}
        )

        explanation = create_ranking_explanation(doc)

        assert "Rank #1" in explanation
        assert "Score: 0.850" in explanation
        assert "doc_123" in explanation
        assert "keyword_relevance" in explanation
        assert "content_quality" in explanation
        assert "Original search score: 0.750" in explanation


# ===== MAIN CLASS TESTS =====

class TestDocumentRanker:
    """Test DocumentRanker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config_provider = Mock()
        self.mock_language_provider = Mock()

        # Mock configuration
        self.mock_config = RankingConfig(
            method=RankingMethod.LANGUAGE_ENHANCED,
            enable_diversity=True,
            diversity_threshold=0.8,
            boost_recent=True,
            boost_authoritative=True,
            content_length_factor=True,
            keyword_density_factor=True,
            language_specific_boost=True
        )

        # Mock language features
        self.mock_language_features = LanguageFeatures(
            importance_words={"važno", "ključno"},
            quality_indicators={
                "positive": ["dobro", "kvalitetno"],
                "negative": ["loše", "neispravno"]
            },
            cultural_patterns=["hrvatska", "zagreb"],
            grammar_patterns=["\\bje\\b", "\\bsu\\b"],
            type_weights={"academic": 1.2, "news": 0.8}
        )

        # Configure mocks
        self.mock_config_provider.load_config.return_value = {
            "ranking": {
                "method": "language_enhanced",
                "enable_diversity": True,
                "diversity_threshold": 0.8,
                "boost_recent": True,
                "boost_authoritative": True,
                "content_length_factor": True,
                "keyword_density_factor": True,
                "language_specific_boost": True
            }
        }
        self.mock_language_provider.get_language_features.return_value = self.mock_language_features

    def test_document_ranker_initialization(self):
        """Test DocumentRanker initialization."""
        ranker = DocumentRanker(
            self.mock_config_provider,
            self.mock_language_provider,
            "hr"
        )

        assert ranker.language == "hr"
        assert ranker.config_provider == self.mock_config_provider
        assert ranker.language_provider == self.mock_language_provider
        assert ranker.config.method == RankingMethod.LANGUAGE_ENHANCED
        assert ranker.language_features == self.mock_language_features

    @patch('src.retrieval.ranker.calculate_language_relevance_score')
    def test_rank_documents_basic(self, mock_lang_score):
        """Test basic document ranking."""
        mock_lang_score.return_value = (0.7, {"test": "metadata"})

        ranker = DocumentRanker(
            self.mock_config_provider,
            self.mock_language_provider,
            "hr"
        )

        documents = [
            {
                "id": "doc1",
                "content": "Test content about RAG",
                "metadata": {"title": "RAG Guide"},
                "relevance_score": 0.8
            },
            {
                "id": "doc2",
                "content": "Different content about AI",
                "metadata": {"title": "AI Basics"},
                "relevance_score": 0.6
            }
        ]

        query = ProcessedQuery(
            text="What is RAG?",
            keywords=["RAG", "what", "is"],
            query_type="factual",
            language="hr"
        )

        ranked_docs = ranker.rank_documents(documents, query)

        assert len(ranked_docs) == 2
        assert all(isinstance(doc, RankedDocument) for doc in ranked_docs)
        assert ranked_docs[0].rank == 1
        assert ranked_docs[1].rank == 2
        assert ranked_docs[0].final_score >= ranked_docs[1].final_score

    def test_rank_documents_empty_list(self):
        """Test ranking empty document list."""
        ranker = DocumentRanker(
            self.mock_config_provider,
            self.mock_language_provider,
            "hr"
        )

        query = ProcessedQuery("test", ["test"], "factual", "hr")
        ranked_docs = ranker.rank_documents([], query)

        assert ranked_docs == []

    def test_rank_single_document(self):
        """Test ranking a single document."""
        ranker = DocumentRanker(
            self.mock_config_provider,
            self.mock_language_provider,
            "hr"
        )

        document = {
            "id": "doc1",
            "content": "Test content",
            "metadata": {},
            "relevance_score": 0.7
        }

        query = ProcessedQuery("test", ["test"], "factual", "hr")

        ranked_doc = ranker._rank_single_document(document, query, {})

        assert isinstance(ranked_doc, RankedDocument)
        assert ranked_doc.id == "doc1"
        assert ranked_doc.content == "Test content"
        assert ranked_doc.original_score == 0.7
        assert len(ranked_doc.ranking_signals) > 0

    def test_explain_ranking(self):
        """Test ranking explanation generation."""
        ranker = DocumentRanker(
            self.mock_config_provider,
            self.mock_language_provider,
            "hr"
        )

        signals = [RankingSignal("test", 0.8, 0.3)]
        doc = RankedDocument("1", "content", {}, 0.7, 0.8, 1, signals, {})

        explanation = ranker.explain_ranking(doc)

        assert isinstance(explanation, str)
        assert "Rank #1" in explanation
        assert "doc" in explanation.lower()


# ===== FACTORY FUNCTION TESTS =====

class TestFactoryFunctions:
    """Test factory functions."""

    @patch('src.retrieval.ranker_providers.create_config_provider')
    @patch('src.retrieval.ranker_providers.create_language_provider')
    def test_create_document_ranker_default_providers(self, mock_lang_provider, mock_config_provider):
        """Test creating document ranker with default providers."""
        mock_config = Mock()
        mock_language = Mock()
        mock_config_provider.return_value = mock_config
        mock_lang_provider.return_value = mock_language

        # Mock the config provider behavior
        mock_config.load_config.return_value = {
            "ranking": {
                "method": "language_enhanced",
                "enable_diversity": True,
                "diversity_threshold": 0.8,
                "boost_recent": True,
                "boost_authoritative": True,
                "content_length_factor": True,
                "keyword_density_factor": True,
                "language_specific_boost": True
            }
        }

        # Mock language features
        mock_language.get_language_features.return_value = LanguageFeatures(
            importance_words=set(),
            quality_indicators={},
            cultural_patterns=[],
            grammar_patterns=[],
            type_weights={}
        )

        ranker = create_document_ranker("hr")

        assert isinstance(ranker, DocumentRanker)
        assert ranker.language == "hr"
        mock_config_provider.assert_called_once()
        mock_lang_provider.assert_called_once()

    @patch('src.retrieval.ranker_providers.create_mock_config_provider')
    @patch('src.retrieval.ranker_providers.create_mock_language_provider')
    def test_create_mock_ranker(self, mock_lang_provider, mock_config_provider):
        """Test creating mock ranker."""
        mock_config = Mock()
        mock_language = Mock()
        mock_config_provider.return_value = mock_config
        mock_lang_provider.return_value = mock_language

        # Mock the config provider behavior
        mock_config.load_config.return_value = {
            "ranking": {
                "method": "language_enhanced",
                "enable_diversity": True,
                "diversity_threshold": 0.8,
                "boost_recent": True,
                "boost_authoritative": True,
                "content_length_factor": True,
                "keyword_density_factor": True,
                "language_specific_boost": True
            }
        }

        # Mock language features
        mock_language.get_language_features.return_value = LanguageFeatures(
            importance_words=set(),
            quality_indicators={},
            cultural_patterns=[],
            grammar_patterns=[],
            type_weights={}
        )

        custom_config = {"test": "value"}
        ranker = create_mock_ranker("en", custom_config)

        assert isinstance(ranker, DocumentRanker)
        assert ranker.language == "en"
        mock_config_provider.assert_called_once_with(custom_config)
        mock_lang_provider.assert_called_once()

    @patch('src.retrieval.ranker_providers.create_mock_config_provider')
    @patch('src.retrieval.ranker_providers.create_mock_language_provider')
    def test_create_mock_ranker_default_config(self, mock_lang_provider, mock_config_provider):
        """Test creating mock ranker with default config."""
        mock_config = Mock()
        mock_language = Mock()
        mock_config_provider.return_value = mock_config
        mock_lang_provider.return_value = mock_language

        # Mock the config provider behavior
        mock_config.load_config.return_value = {
            "ranking": {
                "method": "language_enhanced",
                "enable_diversity": True,
                "diversity_threshold": 0.8,
                "boost_recent": True,
                "boost_authoritative": True,
                "content_length_factor": True,
                "keyword_density_factor": True,
                "language_specific_boost": True
            }
        }

        # Mock language features
        mock_language.get_language_features.return_value = LanguageFeatures(
            importance_words=set(),
            quality_indicators={},
            cultural_patterns=[],
            grammar_patterns=[],
            type_weights={}
        )

        ranker = create_mock_ranker()

        assert isinstance(ranker, DocumentRanker)
        assert ranker.language == "hr"  # Default language
        mock_config_provider.assert_called_once_with({})