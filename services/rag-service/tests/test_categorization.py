"""
Comprehensive test suite for categorization_v2.py demonstrating 100% testability.
Tests pure functions and dependency injection patterns.
"""

from typing import Any, Dict

import pytest
from src.retrieval.categorization import (CategorizationConfig, CategoryMatch,
                                          CategoryType, QueryCategorizer,
                                          QueryComplexity,
                                          calculate_query_complexity,
                                          categorize_query_pure,
                                          determine_retrieval_strategy,
                                          extract_cultural_indicators,
                                          match_category_patterns,
                                          normalize_query_text)
from src.retrieval.categorization_providers import (
    MockConfigProvider, NoOpLoggerProvider, TestLoggerProvider,
    create_complex_test_config, create_minimal_test_config,
    create_test_categorization_setup)


class TestPureFunctions:
    """Test pure business logic functions with no dependencies."""

    def test_normalize_query_text_basic(self):
        """Test basic query text normalization."""
        result = normalize_query_text("  What is Croatia?  ")
        assert result == "what is croatia"

    def test_normalize_query_text_croatian_diacritics(self):
        """Test normalization preserves Croatian diacritics."""
        result = normalize_query_text("Što je Hrvatska kultura?")
        assert result == "što je hrvatska kultura"
        assert "š" in result and "č" in result  # Diacritics preserved

    def test_normalize_query_text_punctuation_removal(self):
        """Test punctuation removal while preserving words."""
        result = normalize_query_text("What's the API documentation?!")
        assert result == "whats the api documentation"

    def test_normalize_query_text_empty_input(self):
        """Test normalization handles empty input gracefully."""
        assert normalize_query_text("") == ""
        assert normalize_query_text(None) == ""

    def test_extract_cultural_indicators_croatian(self):
        """Test extraction of Croatian cultural indicators."""
        cultural_keywords = {
            "croatian_culture": ["hrvatski", "hrvatska", "zagreb"],
            "croatian_history": ["domovinski rat", "jugoslavija"],
        }

        indicators = extract_cultural_indicators(
            "Što je hrvatski domovinski rat?", cultural_keywords
        )

        assert len(indicators) == 2
        assert "croatian_culture:hrvatski" in indicators
        assert "croatian_history:domovinski rat" in indicators

    def test_extract_cultural_indicators_no_matches(self):
        """Test extraction when no cultural indicators present."""
        cultural_keywords = {"croatian_culture": ["hrvatski", "hrvatska"]}

        indicators = extract_cultural_indicators(
            "What is API documentation?", cultural_keywords
        )

        assert indicators == []

    def test_calculate_query_complexity_simple(self):
        """Test simple query complexity calculation."""
        complexity = calculate_query_complexity(
            "What is API?",
            {"simple": 2.0, "moderate": 5.0, "complex": 8.0, "analytical": 12.0},
        )
        assert complexity == QueryComplexity.SIMPLE

    def test_calculate_query_complexity_analytical(self):
        """Test analytical query complexity calculation."""
        complex_query = (
            "How does the Croatian education system compare to other European "
            "systems in terms of cultural preservation and modern technological "
            "integration, and what are the implications for future development?"
        )

        complexity = calculate_query_complexity(
            complex_query,
            {"simple": 2.0, "moderate": 5.0, "complex": 8.0, "analytical": 12.0},
        )
        assert complexity in [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL]

    def test_calculate_query_complexity_empty_input(self):
        """Test complexity calculation handles empty input."""
        complexity = calculate_query_complexity("", {})
        assert complexity == QueryComplexity.SIMPLE

    def test_match_category_patterns_exact_match(self):
        """Test exact pattern matching for categories."""
        patterns = {
            "technical": ["API", "database", "programming"],
            "cultural": ["kultura", "tradicija"],
        }

        matches = match_category_patterns("What is API documentation?", patterns)

        assert len(matches) >= 1
        category, confidence, matched_patterns = matches[0]
        assert category == CategoryType.TECHNICAL
        assert confidence > 0
        assert "API" in matched_patterns

    def test_match_category_patterns_multiple_matches(self):
        """Test pattern matching with multiple category matches."""
        patterns = {
            "technical": ["programming", "software"],
            "general": ["what", "how"],
        }

        matches = match_category_patterns("What is programming?", patterns)

        # Should match both categories, sorted by confidence
        assert len(matches) >= 2
        categories = [match[0] for match in matches]
        assert CategoryType.TECHNICAL in categories
        assert CategoryType.GENERAL in categories

    def test_match_category_patterns_no_matches(self):
        """Test pattern matching when no patterns match."""
        patterns = {
            "technical": ["API", "database"],
            "cultural": ["kultura", "tradicija"],
        }

        matches = match_category_patterns("Random unrelated query", patterns)
        assert matches == []

    def test_determine_retrieval_strategy_category_specific(self):
        """Test retrieval strategy determination based on category."""
        strategy_config = {"category_technical": "dense_search", "default": "hybrid"}

        strategy = determine_retrieval_strategy(
            CategoryType.TECHNICAL, QueryComplexity.SIMPLE, [], strategy_config
        )

        assert strategy == "dense_search"

    def test_determine_retrieval_strategy_cultural_context(self):
        """Test retrieval strategy for cultural context."""
        strategy_config = {"cultural_context": "cultural_aware", "default": "hybrid"}

        strategy = determine_retrieval_strategy(
            CategoryType.GENERAL,
            QueryComplexity.SIMPLE,
            ["croatian_culture:hrvatska"],
            strategy_config,
        )

        assert strategy == "cultural_aware"

    def test_determine_retrieval_strategy_complexity_based(self):
        """Test retrieval strategy based on complexity."""
        strategy_config = {"complexity_analytical": "hierarchical", "default": "hybrid"}

        strategy = determine_retrieval_strategy(
            CategoryType.GENERAL, QueryComplexity.ANALYTICAL, [], strategy_config
        )

        assert strategy == "hierarchical"

    def test_determine_retrieval_strategy_default_fallback(self):
        """Test default strategy fallback."""
        strategy_config = {"default": "hybrid"}

        strategy = determine_retrieval_strategy(
            CategoryType.GENERAL, QueryComplexity.SIMPLE, [], strategy_config
        )

        assert strategy == "hybrid"

    def test_categorize_query_pure_complete_workflow(self):
        """Test complete query categorization workflow."""
        config = CategorizationConfig(
            categories={"technical": {"priority": 1}},
            patterns={"technical": ["API", "kod", "programming"]},
            cultural_keywords={"tech": ["software", "development"]},
            complexity_thresholds={
                "simple": 2.0,
                "moderate": 5.0,
                "complex": 8.0,
                "analytical": 12.0,
            },
            retrieval_strategies={"category_technical": "dense", "default": "hybrid"},
        )

        result = categorize_query_pure("What is API programming?", config)

        assert isinstance(result, CategoryMatch)
        assert result.category == CategoryType.TECHNICAL
        assert result.confidence > 0
        assert len(result.matched_patterns) > 0
        assert result.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
        assert result.retrieval_strategy == "dense"

    def test_categorize_query_pure_empty_input(self):
        """Test categorization handles empty input gracefully."""
        config = CategorizationConfig(
            categories={},
            patterns={},
            cultural_keywords={},
            complexity_thresholds={},
            retrieval_strategies={},
        )

        result = categorize_query_pure("", config)

        assert result.category == CategoryType.GENERAL
        assert result.confidence == 0.0
        assert result.matched_patterns == []
        assert result.complexity == QueryComplexity.SIMPLE
        assert result.retrieval_strategy == "default"

    def test_categorize_query_pure_none_config(self):
        """Test categorization handles None config gracefully."""
        result = categorize_query_pure("test query", None)

        assert result.category == CategoryType.GENERAL
        assert result.confidence == 0.0


class TestDependencyInjection:
    """Test dependency injection patterns and QueryCategorizer class."""

    def test_query_categorizer_initialization(self):
        """Test QueryCategorizer initialization with dependencies."""
        mock_config, test_logger = create_test_categorization_setup("hr")

        categorizer = QueryCategorizer("hr", mock_config, test_logger)

        assert categorizer.language == "hr"
        assert categorizer._config_provider == mock_config
        assert categorizer._logger == test_logger

    def test_query_categorizer_with_mock_config(self):
        """Test categorization using mock configuration."""
        mock_config = MockConfigProvider()
        mock_config.set_categorization_config("hr", create_minimal_test_config())
        test_logger = TestLoggerProvider()

        categorizer = QueryCategorizer("hr", mock_config, test_logger)
        result = categorizer.categorize_query("test API kod")

        assert isinstance(result, CategoryMatch)
        assert result.category == CategoryType.TECHNICAL
        assert len(test_logger.get_messages("debug")) > 0
        assert len(test_logger.get_messages("info")) > 0

    def test_query_categorizer_no_logger(self):
        """Test categorization without logger provider."""
        mock_config = MockConfigProvider()
        mock_config.set_categorization_config("hr", create_minimal_test_config())

        categorizer = QueryCategorizer("hr", mock_config)  # No logger
        result = categorizer.categorize_query("test query")

        # Should not raise exception without logger
        assert isinstance(result, CategoryMatch)

    def test_query_categorizer_error_handling(self):
        """Test categorization error handling with mock exceptions."""
        # Create config that will cause errors
        mock_config = MockConfigProvider()
        mock_config.set_categorization_config("hr", {})  # Empty config
        test_logger = TestLoggerProvider()

        categorizer = QueryCategorizer("hr", mock_config, test_logger)
        result = categorizer.categorize_query("test query")

        # Should return safe fallback
        assert result.category == CategoryType.GENERAL
        assert result.confidence == 0.0
        assert len(test_logger.get_messages("warning")) > 0


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_convenience_function_with_mock_provider(self):
        """Test convenience function with injected mock provider."""
        from src.retrieval.categorization import categorize_query

        mock_config = MockConfigProvider()
        mock_config.set_categorization_config("hr", create_minimal_test_config())

        result = categorize_query("test API", "hr", mock_config)

        assert isinstance(result, CategoryMatch)
        assert result.category == CategoryType.TECHNICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
