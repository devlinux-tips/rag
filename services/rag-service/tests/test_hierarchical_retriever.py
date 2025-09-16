"""
Comprehensive tests for hierarchical retrieval system.
Tests pure functions, strategy routing, boost calculations, and dependency injection patterns.
"""

import pytest
from unittest.mock import Mock, AsyncMock
import time
from typing import Any

from src.retrieval.hierarchical_retriever import (
    # Enums
    RetrievalStrategyType,

    # Data structures
    ProcessedQuery,
    SearchResult,
    HierarchicalRetrievalResult,
    RetrievalConfig,

    # Pure functions
    calculate_keyword_boost,
    calculate_technical_boost,
    calculate_temporal_boost,
    calculate_faq_boost,
    calculate_comparative_boost,
    calculate_exact_match_boost,
    apply_strategy_specific_processing,
    filter_results_by_threshold,
    calculate_overall_confidence,
    create_routing_metadata,

    # Core class
    HierarchicalRetriever,
)
from src.retrieval.categorization import CategoryMatch, CategoryType, QueryComplexity


# ===== PURE FUNCTION TESTS =====

class TestCalculateKeywordBoost:
    """Test calculate_keyword_boost pure function."""

    def test_basic_keyword_matching(self):
        """Test basic keyword matching."""
        content = "Python programming is great for data science"
        keywords = ["python", "programming"]

        boost = calculate_keyword_boost(content, keywords, boost_weight=0.2)

        # 2 out of 2 keywords found: (2/2) * 0.2 = 0.2
        assert boost == 0.2

    def test_partial_keyword_matching(self):
        """Test partial keyword matching."""
        content = "Python is a programming language"
        keywords = ["python", "java", "programming"]

        boost = calculate_keyword_boost(content, keywords, boost_weight=0.3)

        # 2 out of 3 keywords found: (2/3) * 0.3 = 0.2
        assert abs(boost - 0.2) < 0.001

    def test_case_insensitive_matching(self):
        """Test case insensitive keyword matching."""
        content = "PYTHON Programming"
        keywords = ["python", "Programming"]

        boost = calculate_keyword_boost(content, keywords, boost_weight=0.2)

        assert boost == 0.2

    def test_no_keyword_matches(self):
        """Test when no keywords match."""
        content = "Java development tutorial"
        keywords = ["python", "rust"]

        boost = calculate_keyword_boost(content, keywords)

        assert boost == 0.0

    def test_empty_inputs(self):
        """Test with empty inputs."""
        assert calculate_keyword_boost("", ["python"]) == 0.0
        assert calculate_keyword_boost("content", []) == 0.0
        assert calculate_keyword_boost("", []) == 0.0

    def test_custom_boost_weight(self):
        """Test with different boost weights."""
        content = "Python programming"
        keywords = ["python"]

        boost_low = calculate_keyword_boost(content, keywords, boost_weight=0.1)
        boost_high = calculate_keyword_boost(content, keywords, boost_weight=0.5)

        assert boost_low == 0.1
        assert boost_high == 0.5


class TestCalculateTechnicalBoost:
    """Test calculate_technical_boost pure function."""

    def test_technical_content_detection(self):
        """Test technical content detection."""
        content = "This API provides programming functionality for software development"

        boost = calculate_technical_boost(content, boost_weight=0.1)

        # Should find: api, programming, software, development = 4 indicators
        assert boost == 0.4

    def test_croatian_technical_terms(self):
        """Test Croatian technical terms."""
        content = "Ovo je kod za programiranje sistema sa algoritmom"

        boost = calculate_technical_boost(content, boost_weight=0.1)

        # Should find: kod, programiranje, sistem = 3 indicators (algoritmom doesn't match algoritam exactly)
        assert abs(boost - 0.3) < 0.001

    def test_mixed_language_technical(self):
        """Test mixed language technical content."""
        content = "Software development sa tehnologija implementation"

        boost = calculate_technical_boost(content, boost_weight=0.1)

        # Should find: software, development, tehnologija, implementation = 4 indicators
        assert boost == 0.4

    def test_boost_capping(self):
        """Test that boost is capped at maximum."""
        content = "api kod programiranje algoritam software sistem tehnologija development programming technical implementation"

        boost = calculate_technical_boost(content, boost_weight=0.1)

        # Should be capped at 0.1 * 5 = 0.5
        assert boost == 0.5

    def test_no_technical_content(self):
        """Test non-technical content."""
        content = "This is a simple story about everyday life"

        boost = calculate_technical_boost(content)

        assert boost == 0.0

    def test_case_insensitive_detection(self):
        """Test case insensitive technical term detection."""
        content = "API and SOFTWARE development"

        boost = calculate_technical_boost(content, boost_weight=0.1)

        assert abs(boost - 0.3) < 0.001  # api, software, development (floating point safe)


class TestCalculateTemporalBoost:
    """Test calculate_temporal_boost pure function."""

    def test_temporal_terms_detection(self):
        """Test temporal terms detection."""
        content = "This is a recent update with current information"
        metadata = {}

        boost = calculate_temporal_boost(content, metadata, boost_weight=0.15)

        # Should find: recent, current = 2 terms * 0.15 * 0.5 = 0.15
        assert boost == 0.15

    def test_croatian_temporal_terms(self):
        """Test Croatian temporal terms."""
        content = "Danas je ovo aktualno i novo"
        metadata = {}

        boost = calculate_temporal_boost(content, metadata, boost_weight=0.15)

        # Should find: danas, aktualno, novo = 3 terms * 0.15 * 0.5 = 0.225
        assert abs(boost - 0.225) < 0.001  # 3 terms * 0.15 * 0.5 multiplier = 0.225

    def test_recent_year_boost(self):
        """Test recent year metadata boost."""
        content = "Some content"
        metadata = {"year": "2024"}

        boost = calculate_temporal_boost(content, metadata, current_year=2024, boost_weight=0.15)

        # Recent year boost: 0.2
        assert boost == 0.2

    def test_moderately_recent_year(self):
        """Test moderately recent year boost."""
        content = "Some content"
        metadata = {"year": "2022"}

        boost = calculate_temporal_boost(content, metadata, current_year=2024, boost_weight=0.15)

        # Moderately recent boost: 0.1
        assert boost == 0.1

    def test_old_year_no_boost(self):
        """Test old year gets no boost."""
        content = "Some content"
        metadata = {"year": "2020"}

        boost = calculate_temporal_boost(content, metadata, current_year=2024, boost_weight=0.15)

        assert boost == 0.0

    def test_combined_temporal_and_metadata_boost(self):
        """Test combined temporal terms and metadata boost."""
        content = "Recent news update"
        metadata = {"year": "2024"}

        boost = calculate_temporal_boost(content, metadata, current_year=2024, boost_weight=0.15)

        # recent * 0.15 * 0.5 + year boost 0.2 = 0.075 + 0.2 = 0.275
        assert abs(boost - 0.35) < 0.001  # 2 terms * 0.15 * 0.5 + 0.2 (recent year) = 0.35

    def test_invalid_year_metadata(self):
        """Test invalid year metadata handling."""
        content = "Some content"
        metadata = {"year": "invalid"}

        boost = calculate_temporal_boost(content, metadata, boost_weight=0.15)

        assert boost == 0.0


class TestCalculateFaqBoost:
    """Test calculate_faq_boost pure function."""

    def test_faq_indicators_detection(self):
        """Test FAQ indicators detection."""
        content = "Pitanje: Što je Python? Odgovor: Python je programski jezik"

        boost = calculate_faq_boost(content, boost_weight=0.1)

        # FAQ indicators + structure boost + length boost
        # pitanje, odgovor, što: 3 * 0.1 * 0.3 = 0.09
        # Structure boost: pitanje:, odgovor: = 0.2
        # Length boost (50-300 chars): 0.1
        # Total: 0.09 + 0.2 + 0.1 = 0.39
        assert abs(boost - 0.39) < 0.01

    def test_english_faq_pattern(self):
        """Test English FAQ pattern."""
        content = "Q: What is programming? A: Programming is coding"

        boost = calculate_faq_boost(content, boost_weight=0.1)

        # question, what, programming: 3 * 0.1 * 0.3 = 0.09
        # Structure boost: Q:, A: = 0.2
        # Total: 0.09 + 0.2 = 0.29
        assert abs(boost - 0.29) < 0.01

    def test_faq_length_boost(self):
        """Test FAQ length boost for concise answers."""
        content = "Q: What is API? A: Application Programming Interface for software communication"
        # Length is ~80 chars, within 50-300 range

        boost = calculate_faq_boost(content, boost_weight=0.1)

        # Should include length boost of 0.1
        assert boost > 0.2  # Base boosts + length boost

    def test_too_short_content(self):
        """Test content too short for length boost."""
        content = "Q: What? A: Yes"  # ~15 chars

        boost = calculate_faq_boost(content, boost_weight=0.1)

        # Should not include length boost (too short)
        # q:, a:, what: 3 * 0.1 * 0.3 = 0.09
        # Structure: Q:, A: = 0.2
        # Total: 0.29
        assert abs(boost - 0.29) < 0.01

    def test_non_faq_content(self):
        """Test non-FAQ content."""
        content = "This is a regular document about technology"

        boost = calculate_faq_boost(content)

        assert boost == 0.0


class TestCalculateComparativeBoost:
    """Test calculate_comparative_boost pure function."""

    def test_comparative_terms_detection(self):
        """Test comparative terms detection."""
        content = "Compare Python vs Java - different approaches but similar results"

        boost = calculate_comparative_boost(content, boost_weight=0.1)

        # compare, vs, different, similar: 4 * 0.1 * 0.4 = 0.16
        # Structure indicators: vs, - = 2 * 0.05 = 0.1
        # Total: 0.16 + 0.1 = 0.26
        assert abs(boost - 0.26) < 0.01

    def test_croatian_comparative_terms(self):
        """Test Croatian comparative terms."""
        content = "Usporedi razliku između različitih pristupa - slično je bolje"

        boost = calculate_comparative_boost(content, boost_weight=0.1)

        # usporedi, različit, slično, bolje: 4 * 0.1 * 0.4 = 0.16
        # Structure indicators: - = 0.05
        # Total: 0.16 + 0.05 = 0.21
        assert abs(boost - 0.21) < 0.01

    def test_structure_indicators(self):
        """Test structure indicators boost."""
        content = "1. First option 2. Second option • Third point | Table data"

        boost = calculate_comparative_boost(content, boost_weight=0.1)

        # Structure indicators: |, •, 1., 2. = 4 * 0.05 = 0.2
        assert boost == 0.2

    def test_non_comparative_content(self):
        """Test non-comparative content."""
        content = "This is a simple explanation of programming concepts"

        boost = calculate_comparative_boost(content)

        assert boost == 0.0


class TestCalculateExactMatchBoost:
    """Test calculate_exact_match_boost pure function."""

    def test_exact_word_matching(self):
        """Test exact word matching."""
        content = "Python programming tutorial for beginners"
        query_words = ["Python", "programming"]

        boost = calculate_exact_match_boost(content, query_words, boost_weight=0.2)

        # 2 out of 2 words match: (2/2) * 0.2 = 0.2
        assert boost == 0.2

    def test_partial_word_matching(self):
        """Test partial word matching."""
        content = "Python programming tutorial"
        query_words = ["Python", "Java", "programming"]

        boost = calculate_exact_match_boost(content, query_words, boost_weight=0.3)

        # 2 out of 3 words match: (2/3) * 0.3 = 0.2
        assert abs(boost - 0.2) < 0.001

    def test_case_insensitive_matching(self):
        """Test case insensitive matching."""
        content = "PYTHON Programming"
        query_words = ["python", "Programming"]

        boost = calculate_exact_match_boost(content, query_words, boost_weight=0.2)

        assert boost == 0.2

    def test_no_matches(self):
        """Test when no words match."""
        content = "Java development guide"
        query_words = ["Python", "Rust"]

        boost = calculate_exact_match_boost(content, query_words)

        assert boost == 0.0

    def test_empty_query_words(self):
        """Test with empty query words."""
        content = "Some content"
        query_words = []

        boost = calculate_exact_match_boost(content, query_words)

        assert boost == 0.0


class TestApplyStrategySpecificProcessing:
    """Test apply_strategy_specific_processing pure function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.results = [
            SearchResult("Python programming tutorial", {"type": "tutorial"}, 0.8, 0.8, {}),
            SearchResult("Java development guide", {"type": "guide"}, 0.7, 0.7, {}),
            SearchResult("API documentation for developers", {"type": "docs"}, 0.6, 0.6, {})
        ]

        self.processed_query = ProcessedQuery(
            original="Python programming",
            processed="python programming tutorial",
            query_type="technical",
            keywords=["python", "programming"],
            expanded_terms=["coding", "development"],
            metadata={"language": "en"}
        )

        self.config = RetrievalConfig(
            default_max_results=10,
            similarity_thresholds={"semantic_focused": 0.3},
            boost_weights={
                "keyword": 0.2,
                "technical": 0.1,
                "exact_match": 0.15,
                "temporal": 0.15,
                "faq": 0.1,
                "comparative": 0.1
            },
            strategy_mappings={},
            performance_tracking=True
        )

    def test_semantic_focused_strategy(self):
        """Test semantic focused strategy (no boosts)."""
        processed = apply_strategy_specific_processing(
            self.results, RetrievalStrategyType.SEMANTIC_FOCUSED, self.processed_query, self.config
        )

        assert len(processed) == 3
        # Should maintain original similarity scores as final scores
        for original, processed_result in zip(self.results, processed):
            assert processed_result.final_score == original.similarity_score
            assert processed_result.boosts == {}

    def test_keyword_hybrid_strategy(self):
        """Test keyword hybrid strategy."""
        processed = apply_strategy_specific_processing(
            self.results, RetrievalStrategyType.KEYWORD_HYBRID, self.processed_query, self.config
        )

        assert len(processed) == 3
        # First result should have keyword boost
        assert "keyword" in processed[0].boosts
        assert processed[0].final_score > processed[0].similarity_score

    def test_technical_precise_strategy(self):
        """Test technical precise strategy."""
        processed = apply_strategy_specific_processing(
            self.results, RetrievalStrategyType.TECHNICAL_PRECISE, self.processed_query, self.config
        )

        assert len(processed) == 3
        # Results should have technical and exact match boosts
        for result in processed:
            assert "technical" in result.boosts
            assert "exact_match" in result.boosts

    def test_temporal_aware_strategy(self):
        """Test temporal aware strategy."""
        processed = apply_strategy_specific_processing(
            self.results, RetrievalStrategyType.TEMPORAL_AWARE, self.processed_query, self.config
        )

        assert len(processed) == 3
        # Results should have temporal boost
        for result in processed:
            assert "temporal" in result.boosts

    def test_faq_optimized_strategy(self):
        """Test FAQ optimized strategy."""
        processed = apply_strategy_specific_processing(
            self.results, RetrievalStrategyType.FAQ_OPTIMIZED, self.processed_query, self.config
        )

        assert len(processed) == 3
        # Results should have FAQ boost
        for result in processed:
            assert "faq" in result.boosts

    def test_comparative_structured_strategy(self):
        """Test comparative structured strategy."""
        processed = apply_strategy_specific_processing(
            self.results, RetrievalStrategyType.COMPARATIVE_STRUCTURED, self.processed_query, self.config
        )

        assert len(processed) == 3
        # Results should have comparative boost
        for result in processed:
            assert "comparative" in result.boosts

    def test_missing_boost_weight_error(self):
        """Test error when boost weight is missing."""
        incomplete_config = RetrievalConfig(
            default_max_results=10,
            similarity_thresholds={"keyword_hybrid": 0.3},
            boost_weights={},  # Missing keyword weight
            strategy_mappings={},
            performance_tracking=False
        )

        with pytest.raises(ValueError, match="Missing 'keyword' weight"):
            apply_strategy_specific_processing(
                self.results, RetrievalStrategyType.KEYWORD_HYBRID, self.processed_query, incomplete_config
            )

    def test_results_sorted_by_final_score(self):
        """Test that results are sorted by final score."""
        # Create results with different boost potential
        results = [
            SearchResult("Regular content", {}, 0.9, 0.9, {}),
            SearchResult("Python programming API technical", {}, 0.5, 0.5, {}),  # High boost potential
            SearchResult("Some other content", {}, 0.7, 0.7, {})
        ]

        processed = apply_strategy_specific_processing(
            results, RetrievalStrategyType.TECHNICAL_PRECISE, self.processed_query, self.config
        )

        # Results should be sorted by final_score (descending)
        for i in range(len(processed) - 1):
            assert processed[i].final_score >= processed[i + 1].final_score

    def test_final_score_capped_at_one(self):
        """Test that final score is capped at 1.0."""
        # Create result with high similarity that could exceed 1.0 with boosts
        results = [
            SearchResult("Python programming API technical kod programiranje", {}, 0.95, 0.95, {})
        ]

        processed = apply_strategy_specific_processing(
            results, RetrievalStrategyType.TECHNICAL_PRECISE, self.processed_query, self.config
        )

        assert processed[0].final_score <= 1.0

    def test_empty_results(self):
        """Test with empty results list."""
        processed = apply_strategy_specific_processing(
            [], RetrievalStrategyType.SEMANTIC_FOCUSED, self.processed_query, self.config
        )

        assert processed == []


class TestFilterResultsByThreshold:
    """Test filter_results_by_threshold pure function."""

    def test_basic_filtering(self):
        """Test basic threshold filtering."""
        results = [
            SearchResult("High score", {}, 0.8, 0.8, {}),
            SearchResult("Medium score", {}, 0.5, 0.5, {}),
            SearchResult("Low score", {}, 0.2, 0.2, {})
        ]

        filtered = filter_results_by_threshold(results, similarity_threshold=0.4)

        assert len(filtered) == 2
        assert filtered[0].similarity_score == 0.8
        assert filtered[1].similarity_score == 0.5

    def test_no_results_pass_threshold(self):
        """Test when no results pass threshold."""
        results = [
            SearchResult("Low score 1", {}, 0.2, 0.2, {}),
            SearchResult("Low score 2", {}, 0.1, 0.1, {})
        ]

        filtered = filter_results_by_threshold(results, similarity_threshold=0.5)

        assert filtered == []

    def test_all_results_pass_threshold(self):
        """Test when all results pass threshold."""
        results = [
            SearchResult("High score 1", {}, 0.8, 0.8, {}),
            SearchResult("High score 2", {}, 0.7, 0.7, {})
        ]

        filtered = filter_results_by_threshold(results, similarity_threshold=0.5)

        assert len(filtered) == 2

    def test_exact_threshold_match(self):
        """Test result exactly matching threshold."""
        results = [
            SearchResult("Exact match", {}, 0.5, 0.5, {}),
            SearchResult("Below threshold", {}, 0.4, 0.4, {})
        ]

        filtered = filter_results_by_threshold(results, similarity_threshold=0.5)

        assert len(filtered) == 1
        assert filtered[0].similarity_score == 0.5

    def test_empty_results(self):
        """Test with empty results."""
        filtered = filter_results_by_threshold([], similarity_threshold=0.5)

        assert filtered == []


class TestCalculateOverallConfidence:
    """Test calculate_overall_confidence pure function."""

    def test_basic_confidence_calculation(self):
        """Test basic confidence calculation."""
        category_confidence = 0.8
        results = [
            SearchResult("Result 1", {}, 0.9, 0.9, {}),
            SearchResult("Result 2", {}, 0.8, 0.8, {}),
            SearchResult("Result 3", {}, 0.7, 0.7, {})
        ]

        confidence = calculate_overall_confidence(category_confidence, results)

        # Default weights: (0.6, 0.4)
        # Category: 0.8 * 0.6 = 0.48
        # Results: (0.9 + 0.8 + 0.7) / 3 * 0.4 = 0.8 * 0.4 = 0.32
        # Total: 0.48 + 0.32 = 0.8
        expected = 0.8 * 0.6 + (0.9 + 0.8 + 0.7) / 3 * 0.4
        assert abs(confidence - expected) < 0.001

    def test_custom_weights(self):
        """Test with custom weights."""
        category_confidence = 0.9
        results = [SearchResult("Result", {}, 0.8, 0.8, {})]
        weights = (0.7, 0.3)

        confidence = calculate_overall_confidence(category_confidence, results, weights)

        expected = 0.9 * 0.7 + 0.8 * 0.3
        assert abs(confidence - expected) < 0.001

    def test_empty_results(self):
        """Test with empty results."""
        category_confidence = 0.8

        confidence = calculate_overall_confidence(category_confidence, [])

        # Should return category confidence weighted by category weight
        expected = 0.8 * 0.6  # Default category weight
        assert abs(confidence - expected) < 0.001

    def test_single_result(self):
        """Test with single result."""
        category_confidence = 0.7
        results = [SearchResult("Single result", {}, 0.9, 0.9, {})]

        confidence = calculate_overall_confidence(category_confidence, results)

        expected = 0.7 * 0.6 + 0.9 * 0.4
        assert abs(confidence - expected) < 0.001

    def test_more_than_three_results(self):
        """Test with more than 3 results (should use only top 3)."""
        category_confidence = 0.8
        results = [
            SearchResult("Result 1", {}, 0.9, 0.9, {}),
            SearchResult("Result 2", {}, 0.8, 0.8, {}),
            SearchResult("Result 3", {}, 0.7, 0.7, {}),
            SearchResult("Result 4", {}, 0.6, 0.6, {}),  # Should be ignored
            SearchResult("Result 5", {}, 0.5, 0.5, {})   # Should be ignored
        ]

        confidence = calculate_overall_confidence(category_confidence, results)

        # Should only use top 3 results
        expected = 0.8 * 0.6 + (0.9 + 0.8 + 0.7) / 3 * 0.4
        assert abs(confidence - expected) < 0.001


class TestCreateRoutingMetadata:
    """Test create_routing_metadata pure function."""

    def test_complete_routing_metadata(self):
        """Test complete routing metadata creation."""
        processed_query = ProcessedQuery(
            original="Python programming",
            processed="python programming tutorial",
            query_type="technical",
            keywords=["python", "programming"],
            expanded_terms=["coding", "development"],
            metadata={"language": "en"}
        )

        categorization = CategoryMatch(
            category=CategoryType.TECHNICAL,
            confidence=0.85,
            matched_patterns=["programming", "technical"],
            cultural_indicators=["english"],
            complexity=QueryComplexity.MODERATE,
            retrieval_strategy="technical_precise"
        )

        strategy_used = RetrievalStrategyType.TECHNICAL_PRECISE
        results = [SearchResult("Test result", {}, 0.8, 0.8, {})]
        retrieval_time = 0.123
        reranking_applied = True

        metadata = create_routing_metadata(
            processed_query, categorization, strategy_used, results, retrieval_time, reranking_applied
        )

        # Verify structure
        assert "query_processing" in metadata
        assert "categorization" in metadata
        assert "strategy" in metadata
        assert "performance" in metadata

        # Verify query processing data
        query_data = metadata["query_processing"]
        assert query_data["original"] == "Python programming"
        assert query_data["processed"] == "python programming tutorial"
        assert query_data["query_type"] == "technical"
        assert query_data["keywords"] == ["python", "programming"]
        assert query_data["expanded_terms"] == ["coding", "development"]

        # Verify categorization data
        cat_data = metadata["categorization"]
        assert cat_data["primary"] == "technical"
        assert cat_data["confidence"] == 0.85
        assert cat_data["matched_patterns"] == ["programming", "technical"]
        assert cat_data["cultural_indicators"] == ["english"]
        assert cat_data["complexity"] == "moderate"

        # Verify strategy data
        strategy_data = metadata["strategy"]
        assert strategy_data["selected"] == "technical_precise"
        assert strategy_data["retrieval_strategy"] == "technical_precise"

        # Verify performance data
        perf_data = metadata["performance"]
        assert perf_data["retrieval_time"] == 0.123
        assert perf_data["results_count"] == 1
        assert perf_data["reranking_applied"] is True


# ===== DATA STRUCTURE TESTS =====

class TestDataStructures:
    """Test data structure validation."""

    def test_processed_query_creation(self):
        """Test ProcessedQuery creation."""
        query = ProcessedQuery(
            original="test query",
            processed="processed query",
            query_type="simple",
            keywords=["test"],
            expanded_terms=["testing"],
            metadata={"lang": "en"}
        )

        assert query.original == "test query"
        assert query.processed == "processed query"
        assert query.query_type == "simple"
        assert query.keywords == ["test"]
        assert query.expanded_terms == ["testing"]
        assert query.metadata == {"lang": "en"}

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            content="test content",
            metadata={"type": "doc"},
            similarity_score=0.8,
            final_score=0.85,
            boosts={"keyword": 0.05}
        )

        assert result.content == "test content"
        assert result.metadata == {"type": "doc"}
        assert result.similarity_score == 0.8
        assert result.final_score == 0.85
        assert result.boosts == {"keyword": 0.05}

    def test_hierarchical_retrieval_result_creation(self):
        """Test HierarchicalRetrievalResult creation."""
        result = HierarchicalRetrievalResult(
            documents=[{"content": "doc1"}],
            category="technical",
            strategy_used="semantic_focused",
            retrieval_time=0.123,
            total_results=1,
            confidence=0.85,
            routing_metadata={"test": "data"}
        )

        assert result.documents == [{"content": "doc1"}]
        assert result.category == "technical"
        assert result.strategy_used == "semantic_focused"
        assert result.retrieval_time == 0.123
        assert result.total_results == 1
        assert result.confidence == 0.85
        assert result.routing_metadata == {"test": "data"}

    def test_retrieval_config_creation(self):
        """Test RetrievalConfig creation."""
        config = RetrievalConfig(
            default_max_results=10,
            similarity_thresholds={"default": 0.3},
            boost_weights={"keyword": 0.2},
            strategy_mappings={"test": "mapping"},
            performance_tracking=True
        )

        assert config.default_max_results == 10
        assert config.similarity_thresholds == {"default": 0.3}
        assert config.boost_weights == {"keyword": 0.2}
        assert config.strategy_mappings == {"test": "mapping"}
        assert config.performance_tracking is True


# ===== MOCK FACTORIES =====

def create_mock_query_processor():
    """Create mock query processor."""
    mock = Mock()
    mock.process_query.return_value = ProcessedQuery(
        original="test query",
        processed="processed test query",
        query_type="general",
        keywords=["test", "query"],
        expanded_terms=["testing", "questioning"],
        metadata={"lang": "en"}
    )
    return mock


def create_mock_categorizer():
    """Create mock categorizer."""
    mock = Mock()
    mock.categorize_query.return_value = CategoryMatch(
        category=CategoryType.GENERAL,
        confidence=0.8,
        matched_patterns=["general"],
        cultural_indicators=["english"],
        complexity=QueryComplexity.SIMPLE,
        retrieval_strategy="semantic_focused"
    )
    return mock


def create_mock_search_engine():
    """Create mock search engine."""
    mock = AsyncMock()
    mock.search.return_value = [
        SearchResult("Test document 1", {"id": "1"}, 0.9, 0.9, {}),
        SearchResult("Test document 2", {"id": "2"}, 0.8, 0.8, {}),
        SearchResult("Test document 3", {"id": "3"}, 0.7, 0.7, {})
    ]
    return mock


def create_mock_reranker():
    """Create mock reranker."""
    mock = AsyncMock()
    mock.rerank.return_value = [
        {"content": "Reranked doc 1", "metadata": {"id": "1"}, "score": 0.95},
        {"content": "Reranked doc 2", "metadata": {"id": "2"}, "score": 0.85}
    ]
    return mock


def create_mock_logger():
    """Create mock logger."""
    mock = Mock()
    mock.info = Mock()
    mock.debug = Mock()
    mock.error = Mock()
    return mock


def create_test_config():
    """Create test retrieval config."""
    return RetrievalConfig(
        default_max_results=5,
        similarity_thresholds={
            "semantic_focused": 0.3,
            "keyword_hybrid": 0.2,
            "technical_precise": 0.4,
            "temporal_aware": 0.25,
            "faq_optimized": 0.2,
            "comparative_structured": 0.3,
            "default": 0.3
        },
        boost_weights={
            "keyword": 0.2,
            "technical": 0.1,
            "exact_match": 0.15,
            "temporal": 0.15,
            "faq": 0.1,
            "comparative": 0.1
        },
        strategy_mappings={},
        performance_tracking=True
    )


# ===== CORE CLASS TESTS =====

class TestHierarchicalRetriever:
    """Test HierarchicalRetriever class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.query_processor = create_mock_query_processor()
        self.categorizer = create_mock_categorizer()
        self.search_engine = create_mock_search_engine()
        self.reranker = create_mock_reranker()
        self.logger = create_mock_logger()
        self.config = create_test_config()

        self.retriever = HierarchicalRetriever(
            query_processor=self.query_processor,
            categorizer=self.categorizer,
            search_engine=self.search_engine,
            config=self.config,
            reranker=self.reranker,
            logger_provider=self.logger
        )

    @pytest.mark.asyncio
    async def test_basic_retrieval(self):
        """Test basic retrieval workflow."""
        result = await self.retriever.retrieve("test query")

        assert isinstance(result, HierarchicalRetrievalResult)
        assert result.category == "general"
        assert result.strategy_used == "semantic_focused"
        assert len(result.documents) > 0
        assert result.total_results > 0
        assert 0 <= result.confidence <= 1
        assert result.retrieval_time > 0

        # Verify dependencies were called
        self.query_processor.process_query.assert_called_once()
        self.categorizer.categorize_query.assert_called_once()
        self.search_engine.search.assert_called_once()
        self.reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieval_without_reranker(self):
        """Test retrieval without reranker."""
        retriever = HierarchicalRetriever(
            query_processor=self.query_processor,
            categorizer=self.categorizer,
            search_engine=self.search_engine,
            config=self.config
        )

        result = await retriever.retrieve("test query")

        assert isinstance(result, HierarchicalRetrievalResult)
        # Reranker should not be called
        self.reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieval_with_custom_max_results(self):
        """Test retrieval with custom max results."""
        await self.retriever.retrieve("test query", max_results=3)

        # Search should be called with expanded results (3 * 2 = 6)
        self.search_engine.search.assert_called_once()
        call_args = self.search_engine.search.call_args
        assert call_args[1]["k"] == 6  # expanded_results

    @pytest.mark.asyncio
    async def test_retrieval_with_context(self):
        """Test retrieval with context."""
        context = {"user_id": "123", "session": "abc"}

        await self.retriever.retrieve("test query", context=context)

        # Context should be passed to query processor and categorizer
        self.query_processor.process_query.assert_called_with("test query", context)
        self.categorizer.categorize_query.assert_called_with("test query", context)

    @pytest.mark.asyncio
    async def test_strategy_mapping(self):
        """Test different strategy mappings."""
        # Test technical strategy
        self.categorizer.categorize_query.return_value = CategoryMatch(
            category=CategoryType.TECHNICAL,
            confidence=0.9,
            matched_patterns=["api", "programming"],
            cultural_indicators=["english"],
            complexity=QueryComplexity.MODERATE,
            retrieval_strategy="technical_precise"
        )

        result = await self.retriever.retrieve("API programming question")

        assert result.strategy_used == "technical_precise"

    @pytest.mark.asyncio
    async def test_unknown_strategy_error(self):
        """Test error handling for unknown strategy."""
        self.categorizer.categorize_query.return_value = CategoryMatch(
            category=CategoryType.TECHNICAL,
            confidence=0.9,
            matched_patterns=["test"],
            cultural_indicators=["english"],
            complexity=QueryComplexity.SIMPLE,
            retrieval_strategy="unknown_strategy"
        )

        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            await self.retriever.retrieve("test query")

    @pytest.mark.asyncio
    async def test_missing_similarity_threshold_error(self):
        """Test error when similarity threshold is missing."""
        # Create config without threshold for the strategy
        incomplete_config = RetrievalConfig(
            default_max_results=5,
            similarity_thresholds={},  # Missing threshold
            boost_weights={"keyword": 0.2},
            strategy_mappings={},
            performance_tracking=False
        )

        retriever = HierarchicalRetriever(
            query_processor=self.query_processor,
            categorizer=self.categorizer,
            search_engine=self.search_engine,
            config=incomplete_config
        )

        with pytest.raises(ValueError, match="Missing similarity threshold"):
            await retriever.retrieve("test query")

    @pytest.mark.asyncio
    async def test_performance_stats_tracking(self):
        """Test performance statistics tracking."""
        # Execute multiple retrievals
        await self.retriever.retrieve("query 1")
        await self.retriever.retrieve("query 2")

        stats = self.retriever.get_performance_stats()

        assert stats["total_retrievals"] == 2
        assert "strategy_stats" in stats
        assert stats["reranking_enabled"] is True
        assert stats["performance_tracking"] is True

        # Check strategy stats
        strategy_stats = stats["strategy_stats"]
        assert "semantic_focused" in strategy_stats
        assert strategy_stats["semantic_focused"]["count"] == 2
        assert "avg_time" in strategy_stats["semantic_focused"]

    @pytest.mark.asyncio
    async def test_performance_tracking_disabled(self):
        """Test when performance tracking is disabled."""
        config = create_test_config()
        config.performance_tracking = False

        retriever = HierarchicalRetriever(
            query_processor=self.query_processor,
            categorizer=self.categorizer,
            search_engine=self.search_engine,
            config=config
        )

        await retriever.retrieve("test query")

        stats = retriever.get_performance_stats()
        assert stats["strategy_stats"] == {}
        assert stats["performance_tracking"] is False

    @pytest.mark.asyncio
    async def test_logging_functionality(self):
        """Test logging functionality."""
        await self.retriever.retrieve("test query")

        # Verify logging calls were made
        self.logger.info.assert_called()
        self.logger.debug.assert_called()

        # Check specific log messages
        info_calls = [call[0][0] for call in self.logger.info.call_args_list]
        assert any("Hierarchical retrieval for:" in msg for msg in info_calls)
        assert any("Category:" in msg for msg in info_calls)
        assert any("Hierarchical retrieval complete:" in msg for msg in info_calls)

    @pytest.mark.asyncio
    async def test_retrieval_without_logger(self):
        """Test retrieval without logger (should not crash)."""
        retriever = HierarchicalRetriever(
            query_processor=self.query_processor,
            categorizer=self.categorizer,
            search_engine=self.search_engine,
            config=self.config
        )

        # Should complete without errors
        result = await retriever.retrieve("test query")
        assert isinstance(result, HierarchicalRetrievalResult)

    @pytest.mark.asyncio
    async def test_single_result_no_reranking(self):
        """Test that single result skips reranking."""
        # Configure search engine to return only one result
        self.search_engine.search.return_value = [
            SearchResult("Single document", {"id": "1"}, 0.9, 0.9, {})
        ]

        await self.retriever.retrieve("test query")

        # Reranker should not be called for single result
        self.reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_routing_metadata_structure(self):
        """Test routing metadata structure."""
        result = await self.retriever.retrieve("test query")

        metadata = result.routing_metadata
        assert "query_processing" in metadata
        assert "categorization" in metadata
        assert "strategy" in metadata
        assert "performance" in metadata

        # Verify nested structure
        assert "original" in metadata["query_processing"]
        assert "processed" in metadata["query_processing"]
        assert "primary" in metadata["categorization"]
        assert "confidence" in metadata["categorization"]
        assert "selected" in metadata["strategy"]
        assert "retrieval_time" in metadata["performance"]
        assert "results_count" in metadata["performance"]


# ===== INTEGRATION TESTS =====

class TestIntegration:
    """Integration tests for complete hierarchical retrieval workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_retrieval_workflow(self):
        """Test complete end-to-end retrieval workflow."""
        # Set up complete mock environment
        query_processor = create_mock_query_processor()
        categorizer = create_mock_categorizer()
        search_engine = create_mock_search_engine()
        reranker = create_mock_reranker()
        logger = create_mock_logger()
        config = create_test_config()

        # Configure specific responses for technical query
        query_processor.process_query.return_value = ProcessedQuery(
            original="Python API documentation",
            processed="python api documentation tutorial",
            query_type="technical",
            keywords=["python", "api", "documentation"],
            expanded_terms=["programming", "interface", "docs"],
            metadata={"language": "en", "domain": "technical"}
        )

        categorizer.categorize_query.return_value = CategoryMatch(
            category=CategoryType.TECHNICAL,
            confidence=0.92,
            matched_patterns=["api", "documentation", "python"],
            cultural_indicators=["english", "technical"],
            complexity=QueryComplexity.MODERATE,
            retrieval_strategy="technical_precise"
        )

        search_engine.search.return_value = [
            SearchResult("Python API reference guide", {"type": "api_docs", "year": "2024"}, 0.95, 0.95, {}),
            SearchResult("Programming tutorial for beginners", {"type": "tutorial"}, 0.85, 0.85, {}),
            SearchResult("Technical documentation standards", {"type": "docs"}, 0.75, 0.75, {})
        ]

        # Create retriever
        retriever = HierarchicalRetriever(
            query_processor=query_processor,
            categorizer=categorizer,
            search_engine=search_engine,
            config=config,
            reranker=reranker,
            logger_provider=logger
        )

        # Execute retrieval
        result = await retriever.retrieve("Python API documentation", max_results=3)

        # Verify result structure
        assert isinstance(result, HierarchicalRetrievalResult)
        assert result.category == "technical"
        assert result.strategy_used == "technical_precise"
        assert result.confidence > 0.8
        assert len(result.documents) > 0
        assert result.total_results > 0

        # Verify routing metadata
        metadata = result.routing_metadata
        assert metadata["query_processing"]["original"] == "Python API documentation"
        assert metadata["categorization"]["primary"] == "technical"
        assert metadata["categorization"]["confidence"] == 0.92
        assert metadata["strategy"]["selected"] == "technical_precise"
        assert metadata["performance"]["reranking_applied"] is True

        # Verify all components were called
        query_processor.process_query.assert_called_once()
        categorizer.categorize_query.assert_called_once()
        search_engine.search.assert_called_once()
        reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_strategy_workflows(self):
        """Test different strategy workflows."""
        base_retriever_components = {
            "query_processor": create_mock_query_processor(),
            "categorizer": create_mock_categorizer(),
            "search_engine": create_mock_search_engine(),
            "config": create_test_config()
        }

        strategies_to_test = [
            ("semantic_focused", RetrievalStrategyType.SEMANTIC_FOCUSED),
            ("technical_precise", RetrievalStrategyType.TECHNICAL_PRECISE),
            ("temporal_aware", RetrievalStrategyType.TEMPORAL_AWARE),
            ("faq_optimized", RetrievalStrategyType.FAQ_OPTIMIZED),
            ("comparative_structured", RetrievalStrategyType.COMPARATIVE_STRUCTURED)
        ]

        for strategy_name, strategy_type in strategies_to_test:
            # Configure categorizer for this strategy
            categorizer = create_mock_categorizer()
            categorizer.categorize_query.return_value = CategoryMatch(
                category=CategoryType.GENERAL,
                confidence=0.8,
                matched_patterns=["test"],
                cultural_indicators=["english"],
                complexity=QueryComplexity.SIMPLE,
                retrieval_strategy=strategy_name
            )

            # Update the categorizer in components
            components = base_retriever_components.copy()
            components["categorizer"] = categorizer

            retriever = HierarchicalRetriever(**components)

            result = await retriever.retrieve(f"Test query for {strategy_name}")

            assert result.strategy_used == strategy_type.value

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test performance with multiple concurrent requests."""
        # Set up retriever
        retriever = HierarchicalRetriever(
            query_processor=create_mock_query_processor(),
            categorizer=create_mock_categorizer(),
            search_engine=create_mock_search_engine(),
            config=create_test_config()
        )

        # Execute multiple retrievals
        queries = [f"Test query {i}" for i in range(5)]
        results = []

        for query in queries:
            result = await retriever.retrieve(query)
            results.append(result)

        # Verify all completed successfully
        assert len(results) == 5
        assert all(isinstance(r, HierarchicalRetrievalResult) for r in results)

        # Check performance stats
        stats = retriever.get_performance_stats()
        assert stats["total_retrievals"] == 5

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        # Test with failing search engine
        failing_search_engine = AsyncMock()
        failing_search_engine.search.side_effect = Exception("Search engine failure")

        retriever = HierarchicalRetriever(
            query_processor=create_mock_query_processor(),
            categorizer=create_mock_categorizer(),
            search_engine=failing_search_engine,
            config=create_test_config()
        )

        # Should propagate the exception
        with pytest.raises(Exception, match="Search engine failure"):
            await retriever.retrieve("test query")

    @pytest.mark.asyncio
    async def test_boost_calculation_integration(self):
        """Test boost calculation integration in real workflow."""
        # Create search results with boost potential
        search_engine = AsyncMock()
        search_engine.search.return_value = [
            SearchResult("Python programming API technical kod", {"year": "2024"}, 0.8, 0.8, {}),
            SearchResult("Regular content without special terms", {}, 0.9, 0.9, {}),
            SearchResult("FAQ: What is programming? Answer: Coding", {}, 0.7, 0.7, {})
        ]

        # Configure for technical strategy
        categorizer = create_mock_categorizer()
        categorizer.categorize_query.return_value = CategoryMatch(
            category=CategoryType.TECHNICAL,
            confidence=0.9,
            matched_patterns=["programming"],
            cultural_indicators=["english"],
            complexity=QueryComplexity.MODERATE,
            retrieval_strategy="technical_precise"
        )

        retriever = HierarchicalRetriever(
            query_processor=create_mock_query_processor(),
            categorizer=categorizer,
            search_engine=search_engine,
            config=create_test_config()
        )

        result = await retriever.retrieve("Python programming")

        # Verify boosts were applied
        documents = result.documents
        assert len(documents) > 0

        # First document should have received technical and exact match boosts
        # and should likely be ranked higher due to boosts despite lower base similarity
        boosted_doc = next((doc for doc in documents if "technical" in doc["content"]), None)
        assert boosted_doc is not None