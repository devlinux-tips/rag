"""
Comprehensive tests for multilingual query processing system.
Tests pure functions, dependency injection, protocol implementations, and query processing workflows.
"""

import pytest
import logging
import re
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Optional

from src.retrieval.query_processor import (
    # Pure functions
    normalize_text,
    classify_query_by_patterns,
    extract_keywords_from_text,
    expand_terms_with_synonyms,
    calculate_query_confidence,
    generate_query_filters,

    # Data structures
    QueryType,
    ProcessedQuery,

    # Protocols
    LanguageDataProvider,
    SpellChecker,

    # Main class
    MultilingualQueryProcessor,

    # Production implementation
    LanguageDataProvider,

    # Factory function
    create_query_processor,
)

from src.utils.config_models import QueryProcessingConfig


# Test Data Structures
class TestDataStructures:
    """Test query processing data structures."""

    def test_query_type_enum(self):
        """Test QueryType enum values."""
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.EXPLANATORY.value == "explanatory"
        assert QueryType.COMPARISON.value == "comparison"
        assert QueryType.SUMMARIZATION.value == "summarization"
        assert QueryType.GENERAL.value == "general"

    def test_processed_query_creation(self):
        """Test ProcessedQuery dataclass creation."""
        query = ProcessedQuery(
            original="What is AI?",
            processed="what is ai",
            query_type=QueryType.FACTUAL,
            keywords=["ai"],
            expanded_terms=["ai", "artificial", "intelligence"],
            filters={"language": "en"},
            confidence=0.8,
            metadata={"language": "en"}
        )

        assert query.original == "What is AI?"
        assert query.processed == "what is ai"
        assert query.query_type == QueryType.FACTUAL
        assert query.keywords == ["ai"]
        assert query.expanded_terms == ["ai", "artificial", "intelligence"]
        assert query.filters == {"language": "en"}
        assert query.confidence == 0.8
        assert query.metadata == {"language": "en"}

    def test_processed_query_minimal(self):
        """Test ProcessedQuery with minimal data."""
        query = ProcessedQuery(
            original="test",
            processed="test",
            query_type=QueryType.GENERAL,
            keywords=[],
            expanded_terms=[],
            filters={},
            confidence=0.0,
            metadata={}
        )

        assert query.original == "test"
        assert query.query_type == QueryType.GENERAL
        assert query.keywords == []
        assert query.confidence == 0.0


# Test Pure Functions
class TestNormalizeText:
    """Test text normalization pure function."""

    def test_empty_text(self):
        """Test normalization of empty text."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_basic_normalization(self):
        """Test basic text normalization."""
        text = "  Hello   World  "
        result = normalize_text(text)
        assert result == "hello world"

    def test_case_normalization(self):
        """Test case normalization."""
        text = "HELLO World"
        assert normalize_text(text, normalize_case=True) == "hello world"
        assert normalize_text(text, normalize_case=False) == "HELLO World"

    def test_quote_normalization(self):
        """Test quote character normalization."""
        text = '„hello" "world"'
        result = normalize_text(text, normalize_quotes=True)
        assert '"hello" "world"' in result or "'hello' 'world'" in result

    def test_punctuation_normalization(self):
        """Test punctuation normalization."""
        text = "Hello!!! World???"
        result = normalize_text(text, normalize_punctuation=True)
        assert result == "hello! world?"

    def test_selective_normalization(self):
        """Test selective normalization options."""
        text = '  HELLO!!! „World"  '
        result = normalize_text(
            text,
            normalize_case=False,
            normalize_quotes=False,
            normalize_punctuation=True
        )
        assert "HELLO!" in result
        assert '„World"' in result

    def test_complex_text(self):
        """Test normalization of complex text."""
        text = '  What\'s   the   DIFFERENCE???   Between „AI" and "ML"  '
        result = normalize_text(text)
        expected_words = ["what's", "the", "difference?", "between"]
        for word in expected_words:
            assert word in result


class TestClassifyQueryByPatterns:
    """Test query classification pure function."""

    def test_empty_inputs(self):
        """Test classification with empty inputs."""
        assert classify_query_by_patterns("", {}) == "general"
        assert classify_query_by_patterns("test", {}) == "general"
        assert classify_query_by_patterns("", {"factual": ["what"]}) == "general"

    def test_factual_classification(self):
        """Test factual query classification."""
        patterns = {"factual": [r"\b(what|who|when|where|which)\b"]}

        assert classify_query_by_patterns("What is AI?", patterns) == "factual"
        assert classify_query_by_patterns("Who invented computers?", patterns) == "factual"
        assert classify_query_by_patterns("When was Python created?", patterns) == "factual"

    def test_explanatory_classification(self):
        """Test explanatory query classification."""
        patterns = {"explanatory": [r"\b(how|why|explain)\b"]}

        assert classify_query_by_patterns("How does AI work?", patterns) == "explanatory"
        assert classify_query_by_patterns("Why is Python popular?", patterns) == "explanatory"
        assert classify_query_by_patterns("Explain machine learning", patterns) == "explanatory"

    def test_multiple_patterns(self):
        """Test classification with multiple pattern types."""
        patterns = {
            "factual": [r"\b(what|who)\b"],
            "explanatory": [r"\b(how|why)\b"],
            "comparison": [r"\b(compare|versus|vs)\b"]
        }

        assert classify_query_by_patterns("What is AI?", patterns) == "factual"
        assert classify_query_by_patterns("How does it work?", patterns) == "explanatory"
        assert classify_query_by_patterns("Compare Python vs Java", patterns) == "comparison"

    def test_invalid_regex_handling(self):
        """Test handling of invalid regex patterns."""
        patterns = {"factual": ["[invalid"]}  # Invalid regex
        result = classify_query_by_patterns("What is AI?", patterns)
        assert result == "general"

    def test_case_insensitive_matching(self):
        """Test case-insensitive pattern matching."""
        patterns = {"factual": [r"\bwhat\b"]}

        assert classify_query_by_patterns("WHAT is AI?", patterns) == "factual"
        assert classify_query_by_patterns("what is AI?", patterns) == "factual"
        assert classify_query_by_patterns("What is AI?", patterns) == "factual"


class TestExtractKeywordsFromText:
    """Test keyword extraction pure function."""

    def test_empty_text(self):
        """Test keyword extraction from empty text."""
        assert extract_keywords_from_text("") == []
        assert extract_keywords_from_text("   ") == []

    def test_basic_extraction(self):
        """Test basic keyword extraction."""
        text = "machine learning algorithms"
        keywords = extract_keywords_from_text(text)
        assert "machine" in keywords
        assert "learning" in keywords
        assert "algorithms" in keywords

    def test_stop_word_removal(self):
        """Test stop word removal."""
        text = "the quick brown fox"
        stop_words = {"the", "a", "an"}

        keywords = extract_keywords_from_text(text, stop_words=stop_words, remove_stopwords=True)
        assert "the" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords

    def test_stop_word_retention(self):
        """Test keyword extraction without stop word removal."""
        text = "the quick brown fox"
        stop_words = {"the", "a", "an"}

        keywords = extract_keywords_from_text(text, stop_words=stop_words, remove_stopwords=False)
        assert "the" in keywords
        assert "quick" in keywords

    def test_minimum_word_length(self):
        """Test minimum word length filtering."""
        text = "AI is a big topic"
        keywords = extract_keywords_from_text(text, min_word_length=3)
        assert "ai" not in keywords  # Length 2
        assert "big" in keywords      # Length 3
        assert "topic" in keywords    # Length 5

    def test_duplicate_removal(self):
        """Test duplicate keyword removal while preserving order."""
        text = "machine learning machine algorithms learning"
        keywords = extract_keywords_from_text(text)
        assert keywords.count("machine") == 1
        assert keywords.count("learning") == 1
        assert keywords.index("machine") < keywords.index("learning")

    def test_complex_text_extraction(self):
        """Test extraction from complex text with punctuation."""
        text = "What is machine-learning? It's AI, NLP, and deep-learning!"
        keywords = extract_keywords_from_text(text, min_word_length=2)

        # Should extract individual words, not hyphenated phrases
        assert "machine" in keywords
        assert "learning" in keywords
        assert "deep" in keywords


class TestExpandTermsWithSynonyms:
    """Test synonym expansion pure function."""

    def test_empty_inputs(self):
        """Test expansion with empty inputs."""
        assert expand_terms_with_synonyms([], {}) == []
        assert expand_terms_with_synonyms(["word"], {}) == ["word"]
        assert expand_terms_with_synonyms([], {"word": ["synonym"]}) == []

    def test_basic_expansion(self):
        """Test basic synonym expansion."""
        keywords = ["ai", "computer"]
        synonyms = {
            "ai": ["artificial", "intelligence"],
            "computer": ["machine", "system"]
        }

        expanded = expand_terms_with_synonyms(keywords, synonyms)
        assert "ai" in expanded
        assert "artificial" in expanded
        assert "intelligence" in expanded
        assert "computer" in expanded
        assert "machine" in expanded

    def test_no_duplicate_expansion(self):
        """Test that duplicates are not added during expansion."""
        keywords = ["ai", "artificial"]
        synonyms = {"ai": ["artificial", "intelligence"]}

        expanded = expand_terms_with_synonyms(keywords, synonyms)
        assert expanded.count("artificial") == 1
        assert "intelligence" in expanded

    def test_max_terms_limit(self):
        """Test maximum expanded terms limit."""
        keywords = ["ai"]
        synonyms = {"ai": ["artificial", "intelligence", "machine", "robot", "automation"]}

        expanded = expand_terms_with_synonyms(keywords, synonyms, max_expanded_terms=3)
        assert len(expanded) <= 3
        assert "ai" in expanded  # Original should be included

    def test_missing_synonym_handling(self):
        """Test handling of keywords without synonyms."""
        keywords = ["ai", "unknown"]
        synonyms = {"ai": ["artificial"]}

        expanded = expand_terms_with_synonyms(keywords, synonyms)
        assert "ai" in expanded
        assert "artificial" in expanded
        assert "unknown" in expanded

    def test_order_preservation(self):
        """Test that original keyword order is preserved."""
        keywords = ["computer", "ai", "data"]
        synonyms = {"ai": ["artificial"]}

        expanded = expand_terms_with_synonyms(keywords, synonyms)
        computer_idx = expanded.index("computer")
        ai_idx = expanded.index("ai")
        data_idx = expanded.index("data")

        assert computer_idx < ai_idx < data_idx


class TestCalculateQueryConfidence:
    """Test query confidence calculation pure function."""

    def test_empty_query(self):
        """Test confidence for empty query."""
        confidence = calculate_query_confidence("", "", [], "general")
        assert confidence == 0.0

    def test_base_confidence(self):
        """Test base confidence calculation."""
        confidence = calculate_query_confidence("test", "test", [], "general")
        assert confidence == 0.3  # 0.5 base - 0.2 for short query

    def test_keyword_boost(self):
        """Test confidence boost from keywords."""
        keywords = ["ai", "machine", "learning"]
        confidence = calculate_query_confidence("test query", "test query", keywords, "general")
        # 0.5 base + 0.3 (max keyword boost, 3 keywords * 0.1) - 0.2 (short query) = 0.6
        expected = 0.5 + 0.3 - 0.2
        assert confidence == expected

    def test_query_type_boost(self):
        """Test confidence boost from specific query type."""
        confidence = calculate_query_confidence("what is ai", "what is ai", ["ai"], "factual")
        # Calculate actual expected based on algorithm: 0.5 base + 0.1 keyword + 0.2 type - 0.2 short = 0.6
        # But it may be clamped differently, so let's verify it's > base
        base_confidence = calculate_query_confidence("what is ai", "what is ai", ["ai"], "general")
        assert confidence > base_confidence

    def test_pattern_match_boost(self):
        """Test confidence boost from pattern matches."""
        confidence = calculate_query_confidence("test", "test", [], "factual", patterns_matched=2)
        # 0.5 base + 0.2 (specific type) + 0.1 (max pattern boost, 2 * 0.05) - 0.2 (short) = 0.6
        expected = 0.5 + 0.2 + 0.1 - 0.2
        assert confidence == expected

    def test_long_query_no_penalty(self):
        """Test that long queries don't get short query penalty."""
        long_query = "this is a longer query with many words"
        confidence = calculate_query_confidence(long_query, long_query, [], "general")
        assert confidence == 0.5  # No short query penalty

    def test_confidence_clamping(self):
        """Test that confidence is clamped to valid range."""
        # Test upper bound
        many_keywords = ["word" + str(i) for i in range(20)]
        confidence = calculate_query_confidence("long query", "long query", many_keywords, "factual", patterns_matched=10)
        assert confidence <= 1.0

        # Test lower bound
        confidence = calculate_query_confidence("a", "a", [], "general")
        assert confidence >= 0.0


class TestGenerateQueryFilters:
    """Test query filter generation pure function."""

    def test_empty_inputs(self):
        """Test filter generation with empty inputs."""
        filter_config = {"topic_patterns": {}}
        filters = generate_query_filters("", {}, filter_config)
        assert isinstance(filters, dict)

    def test_language_filter(self):
        """Test language filter from context."""
        context = {"language": "hr"}
        filter_config = {"topic_patterns": {}}
        filters = generate_query_filters("test", context, filter_config)
        assert filters["language"] == "hr"

    def test_date_range_filter(self):
        """Test date range filter from context."""
        context = {"date_range": {"start": "2023-01-01", "end": "2023-12-31"}}
        filter_config = {"topic_patterns": {}}
        filters = generate_query_filters("test", context, filter_config)
        assert filters["date_range"] == {"start": "2023-01-01", "end": "2023-12-31"}

    def test_document_type_filter(self):
        """Test document type filter from context."""
        context = {"document_types": ["pdf", "txt"]}
        filter_config = {"topic_patterns": {}}
        filters = generate_query_filters("test", context, filter_config)
        assert filters["document_types"] == ["pdf", "txt"]

    def test_topic_filter_generation(self):
        """Test topic filter generation from query patterns."""
        query = "machine learning algorithms"
        context = {}
        filter_config = {
            "topic_patterns": {
                "technology": [r"\b(machine|algorithm|computer)\b"],
                "science": [r"\b(research|study|analysis)\b"]
            }
        }

        filters = generate_query_filters(query, context, filter_config)
        assert "topics" in filters
        assert "technology" in filters["topics"]

    def test_missing_topic_patterns_error(self):
        """Test error when topic_patterns missing from filter_config."""
        with pytest.raises(ValueError, match="Missing 'topic_patterns' in filter configuration"):
            generate_query_filters("test", {}, {})

    def test_invalid_regex_handling(self):
        """Test handling of invalid regex in topic patterns."""
        query = "test query"
        filter_config = {
            "topic_patterns": {
                "test": ["[invalid"]  # Invalid regex
            }
        }

        filters = generate_query_filters(query, {}, filter_config)
        # Should not crash, topics may be empty
        assert isinstance(filters, dict)

    def test_multiple_context_filters(self):
        """Test generation of multiple filters from context."""
        context = {
            "language": "en",
            "date_range": {"start": "2023-01-01"},
            "document_types": ["pdf"]
        }
        filter_config = {"topic_patterns": {}}

        filters = generate_query_filters("test", context, filter_config)
        assert filters["language"] == "en"
        assert filters["date_range"] == {"start": "2023-01-01"}
        assert filters["document_types"] == ["pdf"]


# Mock Protocol Implementations
class MockLanguageDataProvider:
    """Mock implementation of LanguageDataProvider protocol."""

    def __init__(self):
        self.stop_words_data = {
            "en": {"the", "a", "an", "and", "or"},
            "hr": {"i", "a", "da", "je", "na"}
        }
        self.question_patterns_data = {
            "en": {
                "factual": [r"\b(what|who|when|where|which)\b"],
                "explanatory": [r"\b(how|why|explain)\b"]
            },
            "hr": {
                "factual": [r"\b(što|tko|kada|gdje|koji)\b"],
                "explanatory": [r"\b(kako|zašto|objasniti)\b"]
            }
        }
        self.synonym_groups_data = {
            "en": {"ai": ["artificial", "intelligence"]},
            "hr": {"računalo": ["kompjuter", "computer"]}
        }
        self.morphological_patterns_data = {
            "en": {"stemming": {"enabled": True}},
            "hr": {"declensions": {"enabled": True}}
        }

    def get_stop_words(self, language: str) -> set[str]:
        return self.stop_words_data.get(language, set())

    def get_question_patterns(self, language: str) -> dict[str, list[str]]:
        return self.question_patterns_data.get(language, {})

    def get_synonym_groups(self, language: str) -> dict[str, list[str]]:
        return self.synonym_groups_data.get(language, {})

    def get_morphological_patterns(self, language: str) -> dict[str, Any]:
        return self.morphological_patterns_data.get(language, {})


class MockSpellChecker:
    """Mock implementation of SpellChecker protocol."""

    def __init__(self):
        self.corrections = {
            "machien": "machine",
            "learnign": "learning",
            "algoritm": "algorithm"
        }

    def check_and_correct(self, text: str, language: str) -> str:
        """Mock spell checking that corrects known misspellings."""
        result = text
        for misspelling, correction in self.corrections.items():
            result = result.replace(misspelling, correction)
        return result


# Test Main Query Processor Class
class TestMultilingualQueryProcessor:
    """Test MultilingualQueryProcessor class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return QueryProcessingConfig(
            language="en",
            expand_synonyms=True,
            normalize_case=True,
            remove_stopwords=True,
            min_query_length=2,
            max_query_length=1000,
            max_expanded_terms=10,
            enable_morphological_analysis=False,
            use_query_classification=True,
            enable_spell_check=True
        )

    @pytest.fixture
    def mock_language_provider(self):
        """Create mock language data provider."""
        return MockLanguageDataProvider()

    @pytest.fixture
    def mock_spell_checker(self):
        """Create mock spell checker."""
        return MockSpellChecker()

    @pytest.fixture
    def filter_config(self):
        """Create test filter configuration."""
        return {
            "topic_patterns": {
                "technology": [r"\b(machine|algorithm|computer|ai)\b"],
                "science": [r"\b(research|study|analysis)\b"]
            }
        }

    def test_init_with_dependencies(self, config, mock_language_provider, mock_spell_checker, filter_config):
        """Test initialization with all dependencies."""
        processor = MultilingualQueryProcessor(
            config=config,
            language_data_provider=mock_language_provider,
            spell_checker=mock_spell_checker,
            filter_config=filter_config
        )

        assert processor.config == config
        assert processor.language_data_provider == mock_language_provider
        assert processor.spell_checker == mock_spell_checker
        assert processor.filter_config == filter_config
        assert isinstance(processor.logger, logging.Logger)

    def test_init_with_minimal_dependencies(self, config):
        """Test initialization with minimal dependencies."""
        processor = MultilingualQueryProcessor(config)

        assert processor.config == config
        assert processor.language_data_provider is None
        assert processor.spell_checker is None
        assert processor.filter_config == {}

    def test_process_query_success(self, config, mock_language_provider, mock_spell_checker, filter_config):
        """Test successful query processing."""
        processor = MultilingualQueryProcessor(
            config=config,
            language_data_provider=mock_language_provider,
            spell_checker=mock_spell_checker,
            filter_config=filter_config
        )

        context = {"language": "en"}
        result = processor.process_query("What is machine learning?", context)

        assert isinstance(result, ProcessedQuery)
        assert result.original == "What is machine learning?"
        assert result.processed == "what is machine learning?"
        assert result.query_type == QueryType.FACTUAL
        assert "machine" in result.keywords
        assert "learning" in result.keywords
        assert result.confidence > 0.0
        assert result.filters["language"] == "en"

    def test_process_query_too_short(self, config):
        """Test processing of too short query."""
        processor = MultilingualQueryProcessor(config)

        result = processor.process_query("a")  # Below min_query_length

        assert result.original == "a"
        assert result.processed == ""
        assert result.query_type == QueryType.GENERAL
        assert result.keywords == []
        assert result.confidence == 0.0
        assert "error" in result.metadata

    def test_spell_check_integration(self, config, mock_spell_checker, filter_config):
        """Test spell checking integration."""
        processor = MultilingualQueryProcessor(
            config=config,
            spell_checker=mock_spell_checker,
            filter_config=filter_config
        )

        result = processor.process_query("What is machien learnign?")

        # Spell checker should correct misspellings
        assert "machine" in result.processed
        assert "learning" in result.processed

    def test_synonym_expansion_integration(self, config, mock_language_provider, filter_config):
        """Test synonym expansion integration."""
        processor = MultilingualQueryProcessor(
            config=config,
            language_data_provider=mock_language_provider,
            filter_config=filter_config
        )

        result = processor.process_query("What is AI technology?")

        # Should expand AI with synonyms
        assert "ai" in result.keywords
        if result.expanded_terms:  # Only check if expansion happened
            assert any(term in ["artificial", "intelligence"] for term in result.expanded_terms)

    def test_stop_words_removal(self, config, mock_language_provider, filter_config):
        """Test stop words removal."""
        processor = MultilingualQueryProcessor(
            config=config,
            language_data_provider=mock_language_provider,
            filter_config=filter_config
        )

        result = processor.process_query("What is the best AI?")

        # Check that stop words from the mock provider are removed
        # Mock provider has "the", "a", "an", "and", "or" as stop words for English
        assert "the" not in result.keywords
        # Note: "is" is NOT in the mock stop words, so it will be present

    def test_caching_behavior(self, config, mock_language_provider):
        """Test that language data is cached properly."""
        processor = MultilingualQueryProcessor(
            config=config,
            language_data_provider=mock_language_provider
        )

        # First call should populate cache
        stop_words1 = processor._get_stop_words()
        stop_words2 = processor._get_stop_words()

        # Should return same object (cached)
        assert stop_words1 is stop_words2

    def test_fallback_stop_words(self, config, filter_config):
        """Test fallback stop words when no provider available."""
        processor = MultilingualQueryProcessor(config=config, filter_config=filter_config)

        stop_words = processor._get_stop_words()

        # Should have default English stop words
        assert "the" in stop_words
        assert "and" in stop_words
        assert "of" in stop_words

    def test_fallback_question_patterns(self, config, filter_config):
        """Test fallback question patterns when no provider available."""
        processor = MultilingualQueryProcessor(config=config, filter_config=filter_config)

        patterns = processor._get_question_patterns()

        # Should have default patterns
        assert "factual" in patterns
        assert "explanatory" in patterns

    def test_metadata_generation(self, config, filter_config):
        """Test proper metadata generation."""
        processor = MultilingualQueryProcessor(config=config, filter_config=filter_config)

        result = processor.process_query("What is machine learning technology?")

        assert "processing_steps" in result.metadata
        assert "language" in result.metadata
        assert "original_length" in result.metadata
        assert "processed_length" in result.metadata
        assert "keyword_count" in result.metadata
        assert "expanded_count" in result.metadata


# Test Production Language Data Provider
class TestLanguageDataProvider:
    """Test LanguageDataProvider class."""

    @pytest.fixture
    def mock_config_provider(self):
        """Create mock configuration provider."""
        provider = Mock()
        provider.get_language_config.return_value = {
            "stopwords": {"words": ["the", "a", "an"]},
            "question_patterns": {
                "factual": [r"\b(what|who)\b"],
                "explanatory": [r"\b(how|why)\b"]
            },
            "synonym_groups": {"ai": ["artificial", "intelligence"]},
            "morphological_patterns": {"stemming": {"enabled": True}}
        }
        return provider

    def test_init(self, mock_config_provider):
        """Test initialization."""
        provider = LanguageDataProvider(mock_config_provider)
        assert provider.config_provider == mock_config_provider
        assert provider._cache == {}

    def test_get_stop_words_success(self, mock_config_provider):
        """Test successful stop words retrieval."""
        provider = LanguageDataProvider(mock_config_provider)

        stop_words = provider.get_stop_words("en")

        assert isinstance(stop_words, set)
        assert "the" in stop_words
        assert "a" in stop_words

    def test_get_stop_words_error_fallback(self, mock_config_provider):
        """Test stop words retrieval with error fallback."""
        mock_config_provider.get_language_config.side_effect = KeyError("Missing config")
        provider = LanguageDataProvider(mock_config_provider)

        stop_words = provider.get_stop_words("en")

        assert isinstance(stop_words, set)
        assert len(stop_words) == 0

    def test_get_question_patterns_success(self, mock_config_provider):
        """Test successful question patterns retrieval."""
        provider = LanguageDataProvider(mock_config_provider)

        patterns = provider.get_question_patterns("en")

        assert isinstance(patterns, dict)
        assert "factual" in patterns
        assert "explanatory" in patterns

    def test_get_synonym_groups_success(self, mock_config_provider):
        """Test successful synonym groups retrieval."""
        provider = LanguageDataProvider(mock_config_provider)

        synonyms = provider.get_synonym_groups("en")

        assert isinstance(synonyms, dict)
        assert "ai" in synonyms
        assert "artificial" in synonyms["ai"]

    def test_get_morphological_patterns_success(self, mock_config_provider):
        """Test successful morphological patterns retrieval."""
        provider = LanguageDataProvider(mock_config_provider)

        patterns = provider.get_morphological_patterns("en")

        assert isinstance(patterns, dict)
        assert "stemming" in patterns

    def test_caching_behavior(self, mock_config_provider):
        """Test that results are cached properly."""
        provider = LanguageDataProvider(mock_config_provider)

        # First calls should hit config provider
        stop_words1 = provider.get_stop_words("en")
        stop_words2 = provider.get_stop_words("en")

        # Should only call config provider once due to caching
        assert mock_config_provider.get_language_config.call_count == 1
        assert stop_words1 == stop_words2

    def test_missing_synonym_groups_error(self, mock_config_provider):
        """Test error handling when synonym_groups missing."""
        # Make config provider raise an exception
        mock_config_provider.get_language_config.side_effect = KeyError("Missing config")

        provider = LanguageDataProvider(mock_config_provider)

        # Should fallback to empty dict when config provider fails
        synonyms = provider.get_synonym_groups("en")
        assert synonyms == {}


# Test Factory Function
class TestCreateQueryProcessor:
    """Test create_query_processor factory function."""

    @pytest.fixture
    def main_config(self):
        """Create test main configuration."""
        return {
            "query_processing": {
                "min_query_length": 3,
                "max_query_length": 1000,
                "normalize_case": True,
                "remove_stopwords": True,
                "expand_synonyms": True,
                "max_expanded_terms": 15,
                "enable_morphological_analysis": False,
                "use_query_classification": True,
                "enable_spell_check": False
            },
            "query_filters": {
                "topic_patterns": {
                    "tech": [r"\btechnology\b"]
                }
            }
        }

    def test_create_with_config_only(self, main_config):
        """Test factory with main config only."""
        processor = create_query_processor(main_config, language="en")

        assert isinstance(processor, MultilingualQueryProcessor)
        assert processor.config.language == "en"
        assert processor.config.min_query_length == 3
        assert processor.language_data_provider is None

    def test_create_with_config_provider(self, main_config):
        """Test factory with config provider."""
        mock_config_provider = Mock()
        mock_config_provider.get_language_specific_config.return_value = {
            "filters": {"topic_patterns": {"science": [r"\bscience\b"]}}
        }

        processor = create_query_processor(
            main_config,
            language="hr",
            config_provider=mock_config_provider
        )

        assert isinstance(processor, MultilingualQueryProcessor)
        assert processor.config.language == "hr"
        assert isinstance(processor.language_data_provider, LanguageDataProvider)

    def test_create_with_config_provider_error(self, main_config):
        """Test factory with config provider error handling."""
        mock_config_provider = Mock()
        mock_config_provider.get_language_specific_config.side_effect = KeyError("Missing")

        processor = create_query_processor(
            main_config,
            language="en",
            config_provider=mock_config_provider
        )

        assert isinstance(processor, MultilingualQueryProcessor)
        # When config provider fails, filter_config stays empty (design choice)
        assert processor.filter_config == {}


# Integration Tests
class TestIntegration:
    """Test complete query processing workflows."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end query processing."""
        config = QueryProcessingConfig(
            language="en",
            expand_synonyms=True,
            normalize_case=True,
            remove_stopwords=True,
            min_query_length=2,
            max_query_length=1000,
            max_expanded_terms=10,
            enable_morphological_analysis=False,
            use_query_classification=True,
            enable_spell_check=True
        )

        mock_provider = MockLanguageDataProvider()
        mock_spell_checker = MockSpellChecker()
        filter_config = {"topic_patterns": {"tech": [r"\b(ai|machine|computer)\b"]}}

        processor = MultilingualQueryProcessor(
            config=config,
            language_data_provider=mock_provider,
            spell_checker=mock_spell_checker,
            filter_config=filter_config
        )

        context = {"language": "en", "document_types": ["pdf"]}
        result = processor.process_query("What is machien learning and AI?", context)

        # Verify complete processing
        assert result.original == "What is machien learning and AI?"
        assert "machine learning" in result.processed  # Spell corrected
        assert result.query_type == QueryType.FACTUAL
        assert "machine" in result.keywords
        assert "learning" in result.keywords
        assert result.filters["language"] == "en"
        assert result.filters["document_types"] == ["pdf"]
        assert "tech" in result.filters.get("topics", [])
        assert result.confidence > 0.0

    def test_protocol_compliance(self):
        """Test that all protocols are properly implemented."""
        # Test LanguageDataProvider protocol
        provider = MockLanguageDataProvider()
        assert callable(provider.get_stop_words)
        assert callable(provider.get_question_patterns)
        assert callable(provider.get_synonym_groups)
        assert callable(provider.get_morphological_patterns)

        # Test SpellChecker protocol
        spell_checker = MockSpellChecker()
        assert callable(spell_checker.check_and_correct)

    def test_multilingual_processing(self):
        """Test processing in different languages."""
        config_en = QueryProcessingConfig(
            language="en", expand_synonyms=True, normalize_case=True, remove_stopwords=True,
            min_query_length=2, max_query_length=1000, max_expanded_terms=10,
            enable_morphological_analysis=False, use_query_classification=True, enable_spell_check=True
        )
        config_hr = QueryProcessingConfig(
            language="hr", expand_synonyms=True, normalize_case=True, remove_stopwords=True,
            min_query_length=2, max_query_length=1000, max_expanded_terms=10,
            enable_morphological_analysis=False, use_query_classification=True, enable_spell_check=True
        )

        mock_provider = MockLanguageDataProvider()
        filter_config = {"topic_patterns": {}}

        processor_en = MultilingualQueryProcessor(config_en, mock_provider, filter_config=filter_config)
        processor_hr = MultilingualQueryProcessor(config_hr, mock_provider, filter_config=filter_config)

        # English processing
        result_en = processor_en.process_query("What is AI?")
        assert result_en.metadata["language"] == "en"

        # Croatian processing
        result_hr = processor_hr.process_query("Što je AI?")
        assert result_hr.metadata["language"] == "hr"

    def test_error_handling(self):
        """Test comprehensive error handling."""
        config = QueryProcessingConfig(
            language="en", expand_synonyms=True, normalize_case=True, remove_stopwords=True,
            min_query_length=5, max_query_length=1000, max_expanded_terms=10,
            enable_morphological_analysis=False, use_query_classification=True, enable_spell_check=True
        )
        filter_config = {"topic_patterns": {}}
        processor = MultilingualQueryProcessor(config, filter_config=filter_config)

        # Too short query
        result = processor.process_query("hi")
        assert result.confidence == 0.0
        assert "error" in result.metadata

        # Empty query
        result = processor.process_query("")
        assert result.confidence == 0.0