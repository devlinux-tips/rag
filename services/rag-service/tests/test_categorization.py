"""
Tests for query categorization system.

Tests all data classes, enums, pure functions, and the QueryCategorizer class
with proper dependency injection patterns.
"""

import pytest
from unittest.mock import Mock

from src.retrieval.categorization import (
    # Enums
    CategoryType,
    QueryComplexity,

    # Data Classes
    CategoryMatch,
    CategorizationConfig,

    # Protocols
    ConfigProvider,
    LoggerProvider,

    # Pure Functions
    normalize_query_text,
    extract_cultural_indicators,
    calculate_query_complexity,
    match_category_patterns,
    determine_retrieval_strategy,
    categorize_query_pure,

    # Main Class
    QueryCategorizer,

    # Factory Functions
    create_query_categorizer,
)


class TestCategoryType:
    """Test CategoryType enum."""

    def test_category_type_values(self):
        """Test category type enum values."""
        assert CategoryType.GENERAL.value == "general"
        assert CategoryType.TECHNICAL.value == "technical"
        assert CategoryType.CULTURAL.value == "cultural"
        assert CategoryType.HISTORICAL.value == "historical"
        assert CategoryType.ACADEMIC.value == "academic"
        assert CategoryType.LEGAL.value == "legal"
        assert CategoryType.MEDICAL.value == "medical"
        assert CategoryType.BUSINESS.value == "business"
        assert CategoryType.TOURISM.value == "tourism"
        assert CategoryType.EDUCATION.value == "education"

    def test_category_type_enum_creation(self):
        """Test creating category type from string."""
        assert CategoryType("general") == CategoryType.GENERAL
        assert CategoryType("technical") == CategoryType.TECHNICAL

    def test_category_type_invalid_value(self):
        """Test creating category type with invalid value."""
        with pytest.raises(ValueError):
            CategoryType("invalid_category")


class TestQueryComplexity:
    """Test QueryComplexity enum."""

    def test_query_complexity_values(self):
        """Test query complexity enum values."""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MODERATE.value == "moderate"
        assert QueryComplexity.COMPLEX.value == "complex"
        assert QueryComplexity.ANALYTICAL.value == "analytical"

    def test_query_complexity_enum_creation(self):
        """Test creating complexity from string."""
        assert QueryComplexity("simple") == QueryComplexity.SIMPLE
        assert QueryComplexity("analytical") == QueryComplexity.ANALYTICAL


class TestCategoryMatch:
    """Test CategoryMatch data class."""

    def test_category_match_creation(self):
        """Test category match creation."""
        match = CategoryMatch(
            category=CategoryType.TECHNICAL,
            confidence=0.85,
            matched_patterns=["programming", "code"],
            cultural_indicators=["programming:python"],
            complexity=QueryComplexity.MODERATE,
            retrieval_strategy="technical_search"
        )

        assert match.category == CategoryType.TECHNICAL
        assert match.confidence == 0.85
        assert match.matched_patterns == ["programming", "code"]
        assert match.cultural_indicators == ["programming:python"]
        assert match.complexity == QueryComplexity.MODERATE
        assert match.retrieval_strategy == "technical_search"


class TestCategorizationConfig:
    """Test CategorizationConfig data class."""

    def test_categorization_config_creation(self):
        """Test categorization configuration creation."""
        categories = {"technical": {"priority": 1}}
        patterns = {"technical": ["programming", "code"]}
        cultural_keywords = {"programming": ["python", "java"]}
        complexity_thresholds = {"simple": 0.5, "moderate": 1.0, "complex": 2.0, "analytical": 3.0}
        retrieval_strategies = {"default": "basic_search"}

        config = CategorizationConfig(
            categories=categories,
            patterns=patterns,
            cultural_keywords=cultural_keywords,
            complexity_thresholds=complexity_thresholds,
            retrieval_strategies=retrieval_strategies
        )

        assert config.categories == categories
        assert config.patterns == patterns
        assert config.cultural_keywords == cultural_keywords
        assert config.complexity_thresholds == complexity_thresholds
        assert config.retrieval_strategies == retrieval_strategies


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_normalize_query_text_basic(self):
        """Test basic query text normalization."""
        result = normalize_query_text("  What is   Python programming?  ")

        assert result == "what is python programming"

    def test_normalize_query_text_empty(self):
        """Test normalizing empty query."""
        result = normalize_query_text("")
        assert result == ""

        result = normalize_query_text("   ")
        assert result == ""

    def test_normalize_query_text_punctuation(self):
        """Test removing punctuation while preserving diacritics."""
        result = normalize_query_text("Što je to RAG? Kako funkcioniše?")

        assert result == "što je to rag kako funkcioniše"

    def test_normalize_query_text_whitespace(self):
        """Test normalizing multiple whitespace."""
        result = normalize_query_text("query\twith\t\ttabs\nand\n\nnewlines")

        assert result == "query with tabs and newlines"

    def test_normalize_query_text_diacritics(self):
        """Test preserving Croatian diacritics."""
        result = normalize_query_text("Što je mašinsko učenje?")

        assert result == "što je mašinsko učenje"

    def test_extract_cultural_indicators_basic(self):
        """Test extracting cultural indicators from query."""
        cultural_keywords = {
            "programming": ["python", "java", "programming"],
            "croatian": ["zagreb", "croatia", "hrvatski"]
        }

        result = extract_cultural_indicators("I want to learn Python programming", cultural_keywords)

        assert "programming:python" in result
        assert "programming:programming" in result
        assert len(result) == 2

    def test_extract_cultural_indicators_case_insensitive(self):
        """Test case insensitive cultural indicator extraction."""
        cultural_keywords = {
            "languages": ["python", "java"]
        }

        result = extract_cultural_indicators("PYTHON is great", cultural_keywords)

        assert "languages:python" in result

    def test_extract_cultural_indicators_empty(self):
        """Test extracting indicators from empty query."""
        cultural_keywords = {"test": ["keyword"]}

        result = extract_cultural_indicators("", cultural_keywords)

        assert result == []

    def test_extract_cultural_indicators_no_matches(self):
        """Test extracting indicators with no matches."""
        cultural_keywords = {
            "programming": ["python", "java"]
        }

        result = extract_cultural_indicators("What is the weather today?", cultural_keywords)

        assert result == []

    def test_calculate_query_complexity_simple(self):
        """Test calculating simple query complexity."""
        thresholds = {"simple": 0.5, "moderate": 1.5, "complex": 2.5, "analytical": 3.5}

        result = calculate_query_complexity("What is RAG?", thresholds)

        assert result == QueryComplexity.SIMPLE

    def test_calculate_query_complexity_moderate(self):
        """Test calculating moderate query complexity."""
        thresholds = {"simple": 0.5, "moderate": 1.0, "complex": 2.5, "analytical": 3.5}

        result = calculate_query_complexity("How does machine learning work in practice?", thresholds)

        assert result == QueryComplexity.MODERATE

    def test_calculate_query_complexity_complex(self):
        """Test calculating complex query complexity."""
        thresholds = {"simple": 0.5, "moderate": 1.5, "complex": 2.0, "analytical": 4.0}

        result = calculate_query_complexity(
            "If I wanted to implement a retrieval augmented generation system, "
            "what would be the better approach compared to traditional search?",
            thresholds
        )

        assert result == QueryComplexity.COMPLEX

    def test_calculate_query_complexity_analytical(self):
        """Test calculating analytical query complexity."""
        thresholds = {"simple": 0.5, "moderate": 1.5, "complex": 2.5, "analytical": 3.0}

        result = calculate_query_complexity(
            "How would you compare the performance of different retrieval strategies "
            "when dealing with multilingual documents, assuming we have limited computational resources? "
            "What factors should we consider when choosing between semantic and keyword-based approaches?",
            thresholds
        )

        assert result == QueryComplexity.ANALYTICAL

    def test_calculate_query_complexity_empty(self):
        """Test calculating complexity for empty query."""
        thresholds = {"simple": 0.5, "moderate": 1.5, "complex": 2.5, "analytical": 3.5}

        result = calculate_query_complexity("", thresholds)

        assert result == QueryComplexity.SIMPLE

    def test_calculate_query_complexity_missing_thresholds(self):
        """Test complexity calculation with missing thresholds."""
        incomplete_thresholds = {"simple": 0.5, "moderate": 1.5}

        with pytest.raises(ValueError, match="Missing 'analytical' threshold"):
            calculate_query_complexity("test query", incomplete_thresholds)

        incomplete_thresholds = {"simple": 0.5, "moderate": 1.5, "analytical": 3.5}
        with pytest.raises(ValueError, match="Missing 'complex' threshold"):
            calculate_query_complexity("test query", incomplete_thresholds)

        incomplete_thresholds = {"simple": 0.5, "complex": 2.5, "analytical": 3.5}
        with pytest.raises(ValueError, match="Missing 'moderate' threshold"):
            calculate_query_complexity("test query", incomplete_thresholds)

    def test_match_category_patterns_basic(self):
        """Test basic category pattern matching."""
        patterns = {
            "technical": ["programming", "code", "software"],
            "general": ["what", "how", "explain"]
        }

        result = match_category_patterns("How to code in Python programming?", patterns)

        assert len(result) == 2
        # Should be sorted by confidence (highest first)
        category, confidence, matched = result[0]
        assert category == CategoryType.TECHNICAL
        assert confidence > 0
        assert "programming" in matched or "code" in matched

    def test_match_category_patterns_wildcard(self):
        """Test pattern matching with wildcards."""
        patterns = {
            "technical": ["python*", "*programming*", "machine*learning"]
        }

        result = match_category_patterns("python development and machine learning", patterns)

        assert len(result) == 1
        category, confidence, matched = result[0]
        assert category == CategoryType.TECHNICAL
        assert len(matched) >= 2  # Should match python* and machine*learning

    def test_match_category_patterns_no_matches(self):
        """Test pattern matching with no matches."""
        patterns = {
            "technical": ["programming", "code"],
            "medical": ["doctor", "medicine"]
        }

        result = match_category_patterns("What is the weather today?", patterns)

        assert result == []

    def test_match_category_patterns_invalid_category(self):
        """Test pattern matching with invalid category name."""
        patterns = {
            "invalid_category": ["test", "pattern"],
            "technical": ["programming", "code"]
        }

        result = match_category_patterns("programming test", patterns)

        # Should only return valid categories
        assert len(result) == 1
        assert result[0][0] == CategoryType.TECHNICAL

    def test_determine_retrieval_strategy_category_specific(self):
        """Test retrieval strategy with category-specific strategy."""
        strategy_config = {
            "technical": "advanced_search",  # Use the actual enum value
            "default": "basic_search"
        }

        result = determine_retrieval_strategy(
            CategoryType.TECHNICAL,
            QueryComplexity.SIMPLE,
            [],
            strategy_config
        )

        assert result == "advanced_search"

    def test_determine_retrieval_strategy_cultural_context(self):
        """Test retrieval strategy with cultural context."""
        strategy_config = {
            "cultural_context": "cultural_search",
            "default": "basic_search"
        }

        result = determine_retrieval_strategy(
            CategoryType.GENERAL,
            QueryComplexity.SIMPLE,
            ["cultural:indicator"],
            strategy_config
        )

        assert result == "cultural_search"

    def test_determine_retrieval_strategy_complexity_based(self):
        """Test retrieval strategy based on complexity."""
        strategy_config = {
            "complexity_analytical": "deep_search",  # Use complexity_ prefix
            "default": "basic_search"
        }

        result = determine_retrieval_strategy(
            CategoryType.GENERAL,
            QueryComplexity.ANALYTICAL,
            [],
            strategy_config
        )

        assert result == "deep_search"

    def test_determine_retrieval_strategy_default(self):
        """Test default retrieval strategy."""
        strategy_config = {
            "default": "basic_search"
        }

        result = determine_retrieval_strategy(
            CategoryType.GENERAL,
            QueryComplexity.SIMPLE,
            [],
            strategy_config
        )

        assert result == "basic_search"

    def test_determine_retrieval_strategy_missing_default(self):
        """Test retrieval strategy with missing default."""
        strategy_config = {
            "category_technical": "advanced_search"
        }

        with pytest.raises(ValueError, match="Missing 'default' strategy"):
            determine_retrieval_strategy(
                CategoryType.GENERAL,
                QueryComplexity.SIMPLE,
                [],
                strategy_config
            )

    def test_determine_retrieval_strategy_priority(self):
        """Test retrieval strategy priority order."""
        strategy_config = {
            "technical": "tech_search",  # Use actual enum value
            "cultural_context": "cultural_search",
            "complexity_complex": "complex_search",  # Use complexity_ prefix
            "default": "basic_search"
        }

        # Category-specific should have highest priority
        result = determine_retrieval_strategy(
            CategoryType.TECHNICAL,
            QueryComplexity.COMPLEX,
            ["cultural:indicator"],
            strategy_config
        )
        assert result == "tech_search"

        # Cultural should be second priority
        result = determine_retrieval_strategy(
            CategoryType.GENERAL,
            QueryComplexity.COMPLEX,
            ["cultural:indicator"],
            strategy_config
        )
        assert result == "cultural_search"

        # Complexity should be third priority
        result = determine_retrieval_strategy(
            CategoryType.GENERAL,
            QueryComplexity.COMPLEX,
            [],
            strategy_config
        )
        assert result == "complex_search"

    def test_categorize_query_pure_basic(self):
        """Test basic pure query categorization."""
        config = CategorizationConfig(
            categories={"technical": {"priority": 1}},
            patterns={"technical": ["programming", "code"]},
            cultural_keywords={"programming": ["python"]},
            complexity_thresholds={"simple": 0.5, "moderate": 1.5, "complex": 2.5, "analytical": 3.5},
            retrieval_strategies={"default": "basic_search", "technical": "tech_search"}  # Use actual enum value
        )

        result = categorize_query_pure("How to code in Python programming?", config)

        assert result.category == CategoryType.TECHNICAL
        assert result.confidence > 0
        assert result.retrieval_strategy == "tech_search"
        assert result.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
        assert "programming:python" in result.cultural_indicators

    def test_categorize_query_pure_empty_query(self):
        """Test pure categorization with empty query."""
        config = CategorizationConfig(
            categories={},
            patterns={},
            cultural_keywords={},
            complexity_thresholds={"simple": 0.5, "moderate": 1.5, "complex": 2.5, "analytical": 3.5},
            retrieval_strategies={"default": "basic_search"}
        )

        result = categorize_query_pure("", config)

        assert result.category == CategoryType.GENERAL
        assert result.confidence == 0.0
        assert result.matched_patterns == []
        assert result.cultural_indicators == []
        assert result.complexity == QueryComplexity.SIMPLE
        assert result.retrieval_strategy == "default"

    def test_categorize_query_pure_no_config(self):
        """Test pure categorization with no config."""
        result = categorize_query_pure("test query", None)

        assert result.category == CategoryType.GENERAL
        assert result.confidence == 0.0
        assert result.retrieval_strategy == "default"

    def test_categorize_query_pure_no_matches(self):
        """Test pure categorization with no pattern matches."""
        config = CategorizationConfig(
            categories={"technical": {"priority": 1}},
            patterns={"technical": ["programming", "code"]},
            cultural_keywords={},
            complexity_thresholds={"simple": 0.5, "moderate": 1.5, "complex": 2.5, "analytical": 3.5},
            retrieval_strategies={"default": "basic_search"}
        )

        result = categorize_query_pure("What is the weather today?", config)

        assert result.category == CategoryType.GENERAL
        assert result.confidence == 0.1  # Low confidence for general
        assert result.matched_patterns == []

    def test_categorize_query_pure_priority_selection(self):
        """Test category selection based on priority."""
        config = CategorizationConfig(
            categories={
                "technical": {"priority": 2},
                "general": {"priority": 1}
            },
            patterns={
                "technical": ["code"],
                "general": ["what", "how"]
            },
            cultural_keywords={},
            complexity_thresholds={"simple": 0.5, "moderate": 1.5, "complex": 2.5, "analytical": 3.5},
            retrieval_strategies={"default": "basic_search"}
        )

        result = categorize_query_pure("What is code?", config)

        # Should choose technical due to higher priority, even if general might have more matches
        assert result.category == CategoryType.TECHNICAL


class TestQueryCategorizer:
    """Test QueryCategorizer class."""

    def create_test_providers(self):
        """Create mock providers for testing."""
        config_provider = Mock(spec=ConfigProvider)
        logger_provider = Mock(spec=LoggerProvider)

        # Mock configuration
        config_data = {
            "categories": {
                "technical": {"priority": 1},
                "general": {"priority": 0}
            },
            "patterns": {
                "technical": ["programming", "code", "software"],
                "general": ["what", "how", "explain"]
            },
            "cultural_keywords": {
                "programming": ["python", "java", "javascript"],
                "languages": ["croatian", "english"]
            },
            "complexity_thresholds": {
                "simple": 0.5,
                "moderate": 1.5,
                "complex": 2.5,
                "analytical": 3.5
            },
            "retrieval_strategies": {
                "default": "basic_search",
                "category_technical": "technical_search",
                "complexity_analytical": "deep_search",
                "cultural_context": "cultural_search"
            }
        }

        config_provider.get_categorization_config.return_value = config_data

        return config_provider, logger_provider

    def test_query_categorizer_initialization(self):
        """Test query categorizer initialization."""
        config_provider, logger_provider = self.create_test_providers()

        categorizer = QueryCategorizer("en", config_provider, logger_provider)

        assert categorizer.language == "en"
        assert categorizer._config_provider == config_provider
        assert categorizer._logger == logger_provider
        config_provider.get_categorization_config.assert_called_once_with("en")

    def test_query_categorizer_without_logger(self):
        """Test categorizer initialization without logger."""
        config_provider, _ = self.create_test_providers()

        categorizer = QueryCategorizer("hr", config_provider, None)

        assert categorizer.language == "hr"
        assert categorizer._logger is None

    def test_query_categorizer_missing_config_categories(self):
        """Test categorizer with missing categories config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_categorization_config.return_value = {
            "patterns": {},
            "cultural_keywords": {},
            "complexity_thresholds": {},
            "retrieval_strategies": {}
        }

        with pytest.raises(ValueError, match="Missing 'categories'"):
            QueryCategorizer("en", config_provider)

    def test_query_categorizer_missing_config_patterns(self):
        """Test categorizer with missing patterns config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_categorization_config.return_value = {
            "categories": {},
            "cultural_keywords": {},
            "complexity_thresholds": {},
            "retrieval_strategies": {}
        }

        with pytest.raises(ValueError, match="Missing 'patterns'"):
            QueryCategorizer("en", config_provider)

    def test_query_categorizer_missing_config_cultural_keywords(self):
        """Test categorizer with missing cultural keywords config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_categorization_config.return_value = {
            "categories": {},
            "patterns": {},
            "complexity_thresholds": {},
            "retrieval_strategies": {}
        }

        with pytest.raises(ValueError, match="Missing 'cultural_keywords'"):
            QueryCategorizer("en", config_provider)

    def test_query_categorizer_missing_config_complexity_thresholds(self):
        """Test categorizer with missing complexity thresholds config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_categorization_config.return_value = {
            "categories": {},
            "patterns": {},
            "cultural_keywords": {},
            "retrieval_strategies": {}
        }

        with pytest.raises(ValueError, match="Missing 'complexity_thresholds'"):
            QueryCategorizer("en", config_provider)

    def test_query_categorizer_missing_config_retrieval_strategies(self):
        """Test categorizer with missing retrieval strategies config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_categorization_config.return_value = {
            "categories": {},
            "patterns": {},
            "cultural_keywords": {},
            "complexity_thresholds": {}
        }

        with pytest.raises(ValueError, match="Missing 'retrieval_strategies'"):
            QueryCategorizer("en", config_provider)

    def test_categorize_query_success(self):
        """Test successful query categorization."""
        config_provider, logger_provider = self.create_test_providers()
        categorizer = QueryCategorizer("en", config_provider, logger_provider)

        result = categorizer.categorize_query("How to code in Python programming?")

        assert isinstance(result, CategoryMatch)
        assert result.category == CategoryType.TECHNICAL
        assert result.confidence > 0
        logger_provider.debug.assert_called()
        logger_provider.info.assert_called()

    def test_categorize_query_general_fallback(self):
        """Test query categorization with general fallback."""
        config_provider, logger_provider = self.create_test_providers()
        categorizer = QueryCategorizer("en", config_provider, logger_provider)

        result = categorizer.categorize_query("What is the weather today?")

        assert isinstance(result, CategoryMatch)
        assert result.category == CategoryType.GENERAL

    def test_logging_methods(self):
        """Test logging methods."""
        config_provider, logger_provider = self.create_test_providers()
        categorizer = QueryCategorizer("en", config_provider, logger_provider)

        categorizer._log_info("Test info")
        categorizer._log_debug("Test debug")
        categorizer._log_warning("Test warning")

        logger_provider.info.assert_called()
        logger_provider.debug.assert_called()
        logger_provider.warning.assert_called_with("Test warning")

    def test_logging_methods_without_logger(self):
        """Test logging methods without logger provider."""
        config_provider, _ = self.create_test_providers()
        categorizer = QueryCategorizer("en", config_provider, None)

        # Should not raise exceptions
        categorizer._log_info("Test")
        categorizer._log_debug("Test")
        categorizer._log_warning("Test")


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_query_categorizer_with_provider(self):
        """Test creating categorizer with explicit provider."""
        config_provider = Mock(spec=ConfigProvider)

        # Mock configuration
        config_data = {
            "categories": {},
            "patterns": {},
            "cultural_keywords": {},
            "complexity_thresholds": {"simple": 0.5, "moderate": 1.5, "complex": 2.5, "analytical": 3.5},
            "retrieval_strategies": {"default": "basic_search"}
        }
        config_provider.get_categorization_config.return_value = config_data

        categorizer = create_query_categorizer("en", config_provider)

        assert isinstance(categorizer, QueryCategorizer)
        assert categorizer.language == "en"