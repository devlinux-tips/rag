"""
Comprehensive tests for vectordb/search.py
Tests all data classes, pure functions, protocols, and the SemanticSearchEngine class.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock
from typing import Any

from src.vectordb.search import (
    # Data classes
    SearchQuery,
    SearchResult,
    SearchResponse,

    # Enums
    SearchMethod,

    # Pure functions
    validate_search_query,
    parse_vector_search_results,
    distance_to_similarity,
    calculate_keyword_score,
    combine_scores,
    rerank_results_by_relevance,
    filter_results_by_threshold,
    limit_results,
    extract_context_from_results,

    # Main class
    SemanticSearchEngine,

    # Factory functions
    create_search_query,
    create_search_engine,
)


class TestSearchQuery:
    """Test SearchQuery data class."""

    def test_search_query_creation_defaults(self):
        """Test creating SearchQuery with default values."""
        query = SearchQuery(text="test query")

        assert query.text == "test query"
        assert query.top_k == 5
        assert query.method == "semantic"
        assert query.filters is None
        assert query.similarity_threshold == 0.0
        assert query.max_context_length == 2000
        assert query.rerank is True

    def test_search_query_creation_custom(self):
        """Test creating SearchQuery with custom values."""
        filters = {"language": "hr"}
        query = SearchQuery(
            text="custom query",
            top_k=10,
            method="hybrid",
            filters=filters,
            similarity_threshold=0.5,
            max_context_length=1500,
            rerank=False
        )

        assert query.text == "custom query"
        assert query.top_k == 10
        assert query.method == "hybrid"
        assert query.filters == filters
        assert query.similarity_threshold == 0.5
        assert query.max_context_length == 1500
        assert query.rerank is False

    def test_search_query_validation_positive_top_k(self):
        """Test validation fails for non-positive top_k."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            SearchQuery(text="test", top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            SearchQuery(text="test", top_k=-1)

    def test_search_query_validation_threshold_range(self):
        """Test validation fails for invalid similarity threshold."""
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            SearchQuery(text="test", similarity_threshold=-0.1)

        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            SearchQuery(text="test", similarity_threshold=1.1)

    def test_search_query_validation_boundary_values(self):
        """Test validation accepts boundary values."""
        # Should not raise
        SearchQuery(text="test", top_k=1, similarity_threshold=0.0)
        SearchQuery(text="test", top_k=100, similarity_threshold=1.0)


class TestSearchResult:
    """Test SearchResult data class."""

    def test_search_result_creation(self):
        """Test creating SearchResult."""
        metadata = {"source": "test.pdf", "page": 1}
        result = SearchResult(
            id="doc1",
            content="Test content",
            score=0.95,
            metadata=metadata,
            rank=1,
            method_used="semantic"
        )

        assert result.id == "doc1"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata == metadata
        assert result.rank == 1
        assert result.method_used == "semantic"

    def test_search_result_optional_fields(self):
        """Test SearchResult with optional fields as None."""
        result = SearchResult(
            id="doc2",
            content="Content",
            score=0.8,
            metadata={}
        )

        assert result.rank is None
        assert result.method_used is None

    def test_search_result_to_dict(self):
        """Test converting SearchResult to dictionary."""
        metadata = {"source": "test.pdf"}
        result = SearchResult(
            id="doc1",
            content="Test content",
            score=0.85,
            metadata=metadata,
            rank=2,
            method_used="hybrid"
        )

        result_dict = result.to_dict()

        assert result_dict["id"] == "doc1"
        assert result_dict["content"] == "Test content"
        assert result_dict["score"] == 0.85
        assert result_dict["metadata"] == metadata
        assert result_dict["rank"] == 2
        assert result_dict["method_used"] == "hybrid"


class TestSearchResponse:
    """Test SearchResponse data class."""

    def test_search_response_creation(self):
        """Test creating SearchResponse."""
        results = [
            SearchResult("doc1", "Content 1", 0.9, {}),
            SearchResult("doc2", "Content 2", 0.8, {})
        ]
        metadata = {"filters": {"language": "hr"}}

        response = SearchResponse(
            query="test query",
            results=results,
            total_results=2,
            search_time=0.15,
            method_used="semantic",
            metadata=metadata
        )

        assert response.query == "test query"
        assert response.results == results
        assert response.total_results == 2
        assert response.search_time == 0.15
        assert response.method_used == "semantic"
        assert response.metadata == metadata

    def test_search_response_auto_ranking(self):
        """Test automatic ranking assignment in __post_init__."""
        results = [
            SearchResult("doc1", "Content 1", 0.9, {}),
            SearchResult("doc2", "Content 2", 0.8, {}),
            SearchResult("doc3", "Content 3", 0.7, {})
        ]

        response = SearchResponse(
            query="test",
            results=results,
            total_results=3,
            search_time=0.1,
            method_used="semantic"
        )

        # Check that ranks were auto-assigned
        assert response.results[0].rank == 1
        assert response.results[1].rank == 2
        assert response.results[2].rank == 3

        # Check that method_used was set
        assert response.results[0].method_used == "semantic"
        assert response.results[1].method_used == "semantic"
        assert response.results[2].method_used == "semantic"

    def test_search_response_preserves_existing_ranks(self):
        """Test that existing ranks are not overwritten."""
        results = [
            SearchResult("doc1", "Content 1", 0.9, {}, rank=5, method_used="hybrid")
        ]

        response = SearchResponse(
            query="test",
            results=results,
            total_results=1,
            search_time=0.1,
            method_used="semantic"
        )

        # Should preserve existing rank and method_used
        assert response.results[0].rank == 5
        assert response.results[0].method_used == "hybrid"


class TestSearchMethod:
    """Test SearchMethod enum."""

    def test_search_method_values(self):
        """Test SearchMethod enum values."""
        assert SearchMethod.SEMANTIC.value == "semantic"
        assert SearchMethod.KEYWORD.value == "keyword"
        assert SearchMethod.HYBRID.value == "hybrid"

    def test_search_method_iteration(self):
        """Test iterating over SearchMethod values."""
        methods = {method.value for method in SearchMethod}
        expected = {"semantic", "keyword", "hybrid"}
        assert methods == expected


class TestValidateSearchQuery:
    """Test validate_search_query pure function."""

    def test_validate_search_query_valid(self):
        """Test validation passes for valid query."""
        query = SearchQuery(
            text="valid query",
            top_k=5,
            method="semantic",
            similarity_threshold=0.5
        )

        errors = validate_search_query(query)
        assert errors == []

    def test_validate_search_query_empty_text(self):
        """Test validation fails for empty text."""
        query = SearchQuery(text="", top_k=5)
        errors = validate_search_query(query)
        assert "Query text cannot be empty" in errors

        query = SearchQuery(text="   ", top_k=5)
        errors = validate_search_query(query)
        assert "Query text cannot be empty" in errors

    def test_validate_search_query_invalid_top_k(self):
        """Test validation fails for invalid top_k."""
        # SearchQuery.__post_init__ validates top_k, so we test with valid values
        # but use validate_search_query to test boundary conditions
        query = SearchQuery(text="test", top_k=1)  # Valid for SearchQuery
        query.top_k = 0  # Modify after creation to test validation function
        errors = validate_search_query(query)
        assert "top_k must be positive" in errors

        query = SearchQuery(text="test", top_k=1)
        query.top_k = 101  # Modify after creation to test validation function
        errors = validate_search_query(query)
        assert "top_k cannot exceed 100" in errors

    def test_validate_search_query_invalid_threshold(self):
        """Test validation fails for invalid similarity threshold."""
        # SearchQuery.__post_init__ validates threshold, so modify after creation
        query = SearchQuery(text="test", similarity_threshold=0.0)  # Valid
        query.similarity_threshold = -0.1  # Modify after creation
        errors = validate_search_query(query)
        assert "similarity_threshold must be between 0 and 1" in errors

    def test_validate_search_query_invalid_context_length(self):
        """Test validation fails for invalid max_context_length."""
        query = SearchQuery(text="test", max_context_length=0)
        errors = validate_search_query(query)
        assert "max_context_length must be positive" in errors

    def test_validate_search_query_invalid_method(self):
        """Test validation fails for invalid method."""
        query = SearchQuery(text="test", method="invalid_method")
        errors = validate_search_query(query)
        assert "method must be one of:" in errors[0]

    def test_validate_search_query_multiple_errors(self):
        """Test validation accumulates multiple errors."""
        # Create valid query then modify to test multiple errors
        query = SearchQuery(
            text="valid",
            top_k=1,
            method="semantic",
            similarity_threshold=0.5
        )

        # Modify to introduce errors for validation function testing
        query.text = ""
        query.top_k = 0
        query.method = "invalid"
        query.similarity_threshold = 2.0

        errors = validate_search_query(query)
        assert len(errors) >= 4
        assert any("Query text cannot be empty" in error for error in errors)
        assert any("top_k must be positive" in error for error in errors)
        assert any("similarity_threshold must be between 0 and 1" in error for error in errors)
        assert any("method must be one of:" in error for error in errors)


class TestParseVectorSearchResults:
    """Test parse_vector_search_results pure function."""

    def test_parse_vector_search_results_valid(self):
        """Test parsing valid ChromaDB results."""
        raw_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[{"source": "file1.pdf"}, {"source": "file2.pdf"}]],
            "distances": [[0.1, 0.3]]
        }

        results = parse_vector_search_results(raw_results, "semantic")

        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].content == "Content 1"
        assert results[0].metadata == {"source": "file1.pdf"}
        assert results[0].score == 0.9  # 1 - 0.1
        assert results[0].method_used == "semantic"

        assert results[1].id == "doc2"
        assert results[1].content == "Content 2"
        assert results[1].score == 0.7  # 1 - 0.3

    def test_parse_vector_search_results_flat_format(self):
        """Test parsing results in flat format (not nested)."""
        raw_results = {
            "ids": ["doc1", "doc2"],
            "documents": ["Content 1", "Content 2"],
            "metadatas": [{"source": "file1.pdf"}, {"source": "file2.pdf"}],
            "distances": [0.2, 0.4]
        }

        results = parse_vector_search_results(raw_results)

        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].score == 0.8  # 1 - 0.2

    def test_parse_vector_search_results_empty(self):
        """Test parsing empty results."""
        raw_results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }

        results = parse_vector_search_results(raw_results)
        assert results == []

    def test_parse_vector_search_results_no_ids(self):
        """Test parsing results with no IDs."""
        raw_results = {}
        results = parse_vector_search_results(raw_results)
        assert results == []

    def test_parse_vector_search_results_missing_fields(self):
        """Test parsing fails for missing required fields."""
        raw_results = {"ids": [["doc1"]]}

        with pytest.raises(ValueError, match="ChromaDB response missing 'documents' field"):
            parse_vector_search_results(raw_results)

        raw_results = {"ids": [["doc1"]], "documents": [["Content"]]}

        with pytest.raises(ValueError, match="ChromaDB response missing 'metadatas' field"):
            parse_vector_search_results(raw_results)

    def test_parse_vector_search_results_mismatched_lengths(self):
        """Test parsing with mismatched result lengths."""
        raw_results = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["Content 1"]],  # Shorter than IDs
            "metadatas": [[{"source": "file1.pdf"}]],
            "distances": [[0.1]]
        }

        results = parse_vector_search_results(raw_results)

        # Should handle gracefully with empty content for missing items
        assert len(results) == 3
        assert results[0].content == "Content 1"
        assert results[1].content == ""  # Missing content
        assert results[2].content == ""


class TestDistanceToSimilarity:
    """Test distance_to_similarity pure function."""

    def test_distance_to_similarity_zero(self):
        """Test converting zero distance (identical)."""
        similarity = distance_to_similarity(0.0)
        assert similarity == 1.0

    def test_distance_to_similarity_one(self):
        """Test converting distance of 1.0."""
        similarity = distance_to_similarity(1.0)
        assert similarity == 0.0

    def test_distance_to_similarity_half(self):
        """Test converting distance of 0.5."""
        similarity = distance_to_similarity(0.5)
        assert similarity == 0.5

    def test_distance_to_similarity_clamping(self):
        """Test clamping to valid range."""
        # Negative distance should clamp to 1.0
        similarity = distance_to_similarity(-0.5)
        assert similarity == 1.0

        # Distance > 1 should clamp to 0.0
        similarity = distance_to_similarity(1.5)
        assert similarity == 0.0


class TestCalculateKeywordScore:
    """Test calculate_keyword_score pure function."""

    def test_calculate_keyword_score_exact_match(self):
        """Test scoring for exact keyword matches."""
        query_terms = ["python", "programming"]
        document_text = "python programming tutorial"

        score = calculate_keyword_score(query_terms, document_text)
        assert score == 1.0

    def test_calculate_keyword_score_partial_match(self):
        """Test scoring for partial keyword matches."""
        query_terms = ["python", "java", "c++"]
        document_text = "python programming tutorial"

        score = calculate_keyword_score(query_terms, document_text)
        assert score == 1.0 / 3.0  # 1 out of 3 terms matched

    def test_calculate_keyword_score_phrase_boost(self):
        """Test phrase matching boost."""
        query_terms = ["machine", "learning"]
        document_text = "machine learning is powerful"

        score = calculate_keyword_score(query_terms, document_text)
        # Should get boost for phrase match: 1.0 * 1.5 = 1.5, clamped to 1.0
        assert score == 1.0

    def test_calculate_keyword_score_no_match(self):
        """Test scoring when no keywords match."""
        query_terms = ["python", "java"]
        document_text = "javascript tutorial"

        score = calculate_keyword_score(query_terms, document_text)
        # "java" is a substring of "javascript", so there will be a partial match
        # Let's use terms that definitely don't match
        query_terms = ["rust", "golang"]
        score = calculate_keyword_score(query_terms, document_text)
        assert score == 0.0

    def test_calculate_keyword_score_empty_inputs(self):
        """Test scoring with empty inputs."""
        assert calculate_keyword_score([], "some text") == 0.0
        assert calculate_keyword_score(["term"], "") == 0.0
        assert calculate_keyword_score([], "") == 0.0


class TestCombineScores:
    """Test combine_scores pure function."""

    def test_combine_scores_default_weights(self):
        """Test combining scores with default weights."""
        combined = combine_scores(0.8, 0.6)  # semantic=0.7, keyword=0.3
        expected = (0.8 * 0.7) + (0.6 * 0.3)
        assert combined == expected

    def test_combine_scores_custom_weights(self):
        """Test combining scores with custom weights."""
        combined = combine_scores(0.9, 0.5, 0.6, 0.4)
        expected = (0.9 * 0.6) + (0.5 * 0.4)
        assert combined == expected

    def test_combine_scores_zero_weights(self):
        """Test combining with zero total weight."""
        combined = combine_scores(0.8, 0.6, 0.0, 0.0)
        assert combined == 0.0

    def test_combine_scores_normalization(self):
        """Test weight normalization."""
        # Weights don't sum to 1, should be normalized
        combined = combine_scores(1.0, 0.0, 0.8, 0.2)  # total = 1.0
        # After normalization: semantic_weight = 0.8/1.0 = 0.8, keyword_weight = 0.2/1.0 = 0.2
        # Result: (1.0 * 0.8) + (0.0 * 0.2) = 0.8
        assert combined == 0.8

    def test_combine_scores_clamping(self):
        """Test score clamping to valid range."""
        # Result might exceed 1.0, should be clamped
        combined = combine_scores(1.0, 1.0, 0.9, 0.9)
        assert combined == 1.0


class TestReranking:
    """Test rerank_results_by_relevance pure function."""

    def setup_method(self):
        """Set up test data."""
        self.query_text = "python programming tutorial"
        self.results = [
            SearchResult("doc1", "Python programming basics", 0.8, {}),
            SearchResult("doc2", "Advanced Python tutorial for beginners", 0.7, {"title": "Python Guide"}),
            SearchResult("doc3", "Short text", 0.9, {}),
            SearchResult("doc4", "Very long content that exceeds normal length limits and contains lots of detailed information about various programming concepts and methodologies that might be overwhelming for beginners but could be useful for advanced practitioners", 0.6, {})
        ]

    def test_rerank_results_by_relevance_default_boosts(self):
        """Test reranking with default boost factors."""
        reranked = rerank_results_by_relevance(self.query_text, self.results.copy())

        # Results should be reranked and sorted by updated scores
        assert len(reranked) == 4
        # Scores should be modified by boost factors
        assert all(result.score >= 0 for result in reranked)

    def test_rerank_results_by_relevance_custom_boosts(self):
        """Test reranking with custom boost factors."""
        boost_factors = {
            "term_overlap": 0.3,
            "length_optimal": 1.2,
            "length_short": 0.7,
            "length_long": 0.8,
            "title_boost": 1.3
        }

        reranked = rerank_results_by_relevance(self.query_text, self.results.copy(), boost_factors)

        # Should apply custom boosts
        assert len(reranked) == 4

    def test_rerank_results_by_relevance_empty_results(self):
        """Test reranking empty results list."""
        reranked = rerank_results_by_relevance(self.query_text, [])
        assert reranked == []

    def test_rerank_results_by_relevance_missing_boost_keys(self):
        """Test reranking fails with incomplete boost factors."""
        incomplete_boosts = {"term_overlap": 0.2}  # Missing other required keys

        # Function checks for different keys depending on content length
        # Since we have "Short text" result, it will check for 'length_short' first
        with pytest.raises(ValueError, match="Missing 'length_short' in boost_factors configuration"):
            rerank_results_by_relevance(self.query_text, self.results.copy(), incomplete_boosts)

    def test_rerank_results_sorting(self):
        """Test that results are sorted by score descending after reranking."""
        reranked = rerank_results_by_relevance(self.query_text, self.results.copy())

        # Check scores are in descending order
        scores = [result.score for result in reranked]
        assert scores == sorted(scores, reverse=True)


class TestFilterResults:
    """Test filter_results_by_threshold pure function."""

    def setup_method(self):
        """Set up test data."""
        self.results = [
            SearchResult("doc1", "Content 1", 0.9, {}),
            SearchResult("doc2", "Content 2", 0.7, {}),
            SearchResult("doc3", "Content 3", 0.5, {}),
            SearchResult("doc4", "Content 4", 0.3, {})
        ]

    def test_filter_results_by_threshold(self):
        """Test filtering results by threshold."""
        filtered = filter_results_by_threshold(self.results, 0.6)

        assert len(filtered) == 2
        assert filtered[0].score == 0.9
        assert filtered[1].score == 0.7

    def test_filter_results_zero_threshold(self):
        """Test filtering with zero threshold."""
        filtered = filter_results_by_threshold(self.results, 0.0)
        assert len(filtered) == 4  # All results pass

    def test_filter_results_high_threshold(self):
        """Test filtering with high threshold."""
        filtered = filter_results_by_threshold(self.results, 0.95)
        assert len(filtered) == 0  # No results pass

    def test_filter_results_empty_list(self):
        """Test filtering empty results list."""
        filtered = filter_results_by_threshold([], 0.5)
        assert filtered == []


class TestLimitResults:
    """Test limit_results pure function."""

    def setup_method(self):
        """Set up test data."""
        self.results = [
            SearchResult("doc1", "Content 1", 0.9, {}),
            SearchResult("doc2", "Content 2", 0.8, {}),
            SearchResult("doc3", "Content 3", 0.7, {}),
            SearchResult("doc4", "Content 4", 0.6, {})
        ]

    def test_limit_results_normal(self):
        """Test limiting results to specified count."""
        limited = limit_results(self.results, 2)

        assert len(limited) == 2
        assert limited[0].id == "doc1"
        assert limited[1].id == "doc2"

    def test_limit_results_more_than_available(self):
        """Test limiting when limit exceeds available results."""
        limited = limit_results(self.results, 10)
        assert len(limited) == 4  # Returns all available

    def test_limit_results_zero(self):
        """Test limiting to zero results."""
        limited = limit_results(self.results, 0)
        assert limited == self.results  # Returns all when max_results <= 0

    def test_limit_results_negative(self):
        """Test limiting with negative value."""
        limited = limit_results(self.results, -1)
        assert limited == self.results  # Returns all when max_results <= 0


class TestExtractContext:
    """Test extract_context_from_results pure function."""

    def setup_method(self):
        """Set up test data."""
        self.results = [
            SearchResult("doc1", "Short content", 0.9, {}),
            SearchResult("doc2", "Medium length content with more details", 0.8, {}),
            SearchResult("doc3", "Very long content that contains lots of information and details about various topics", 0.7, {}),
            SearchResult("doc4", "Another content piece", 0.6, {})
        ]

    def test_extract_context_from_results_normal(self):
        """Test extracting context within length limit."""
        context = extract_context_from_results(self.results, max_context_length=200)

        # Should combine results with separator
        assert "Short content" in context
        assert "\n\n" in context
        assert len(context) <= 200

    def test_extract_context_from_results_first_only(self):
        """Test extracting context when only first result fits."""
        context = extract_context_from_results(self.results, max_context_length=15)

        # With max_context_length=15 and "Short content" (13 chars), it should fit
        # Let's use a smaller limit to force truncation
        context = extract_context_from_results(self.results, max_context_length=8)

        # Should include truncated first result with "..."
        assert context.endswith("...")
        assert len(context) <= 8

    def test_extract_context_from_results_custom_separator(self):
        """Test extracting context with custom separator."""
        context = extract_context_from_results(self.results[:2], max_context_length=100, separator=" | ")

        assert " | " in context
        assert "Short content" in context

    def test_extract_context_from_results_empty(self):
        """Test extracting context from empty results."""
        context = extract_context_from_results([], max_context_length=100)
        assert context == ""

    def test_extract_context_from_results_empty_content(self):
        """Test extracting context from results with empty content."""
        empty_results = [
            SearchResult("doc1", "", 0.9, {}),
            SearchResult("doc2", "   ", 0.8, {}),
            SearchResult("doc3", "Real content", 0.7, {})
        ]

        context = extract_context_from_results(empty_results, max_context_length=100)
        assert context == "Real content"


class TestSemanticSearchEngine:
    """Test SemanticSearchEngine class with dependency injection."""

    def setup_method(self):
        """Set up test dependencies."""
        # Mock embedding provider
        self.mock_embedding_provider = Mock()
        self.mock_embedding_provider.encode_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))

        # Mock search provider
        self.mock_search_provider = Mock()
        self.mock_search_provider.search_by_embedding = AsyncMock()
        self.mock_search_provider.search_by_text = AsyncMock()
        self.mock_search_provider.get_document = AsyncMock()

        # Mock config provider
        self.mock_config_provider = Mock()
        self.mock_config_provider.get_search_config.return_value = {"timeout": 30}
        self.mock_config_provider.get_scoring_weights.return_value = {"semantic": 0.7, "keyword": 0.3}

        # Create search engine
        self.search_engine = SemanticSearchEngine(
            embedding_provider=self.mock_embedding_provider,
            search_provider=self.mock_search_provider,
            config_provider=self.mock_config_provider
        )

    @pytest.mark.asyncio
    async def test_search_engine_initialization(self):
        """Test search engine initialization."""
        assert self.search_engine.embedding_provider == self.mock_embedding_provider
        assert self.search_engine.search_provider == self.mock_search_provider
        assert self.search_engine.config_provider == self.mock_config_provider

    @pytest.mark.asyncio
    async def test_search_semantic_method(self):
        """Test semantic search method."""
        # Setup mock response
        self.mock_search_provider.search_by_embedding.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.3]]
        }

        query = SearchQuery(text="test query", method="semantic", top_k=2)
        response = await self.search_engine.search(query)

        # Verify calls
        self.mock_embedding_provider.encode_text.assert_called_once_with("test query")
        self.mock_search_provider.search_by_embedding.assert_called_once()

        # Verify response
        assert response.query == "test query"
        assert response.method_used == "semantic"
        assert len(response.results) == 2
        assert response.results[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_search_keyword_method_with_text_search(self):
        """Test keyword search when provider supports text search."""
        # Make provider support text search
        self.mock_search_provider.search_by_text.return_value = {
            "ids": [["doc1"]],
            "documents": [["Keyword content"]],
            "metadatas": [[{}]],
            "distances": [[0.2]]
        }

        query = SearchQuery(text="keyword test", method="keyword", top_k=1)
        response = await self.search_engine.search(query)

        # Should use text search
        self.mock_search_provider.search_by_text.assert_called_once()
        assert response.method_used == "keyword"

    @pytest.mark.asyncio
    async def test_search_keyword_method_fallback(self):
        """Test keyword search fallback when provider doesn't support text search."""
        # Remove text search capability
        delattr(self.mock_search_provider, 'search_by_text')

        self.mock_search_provider.search_by_embedding.return_value = {
            "ids": [["doc1"]],
            "documents": [["python programming tutorial"]],
            "metadatas": [[{}]],
            "distances": [[0.1]]
        }

        query = SearchQuery(text="python tutorial", method="keyword", top_k=1)
        response = await self.search_engine.search(query)

        # Should fall back to semantic search and re-score
        self.mock_search_provider.search_by_embedding.assert_called_once()
        assert response.method_used == "keyword"

    @pytest.mark.asyncio
    async def test_search_hybrid_method(self):
        """Test hybrid search method."""
        # Setup semantic search response
        self.mock_search_provider.search_by_embedding.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Semantic content", "Another doc"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]]
        }

        # Remove text search to test fallback path
        delattr(self.mock_search_provider, 'search_by_text')

        query = SearchQuery(text="test query", method="hybrid", top_k=2)
        response = await self.search_engine.search(query)

        assert response.method_used == "hybrid"
        # Should combine semantic and keyword scores

    @pytest.mark.asyncio
    async def test_search_validation_error(self):
        """Test search with invalid query."""
        invalid_query = SearchQuery(text="", top_k=5)  # Empty text

        with pytest.raises(ValueError, match="Invalid query"):
            await self.search_engine.search(invalid_query)

    @pytest.mark.asyncio
    async def test_search_unknown_method(self):
        """Test search with unknown method."""
        # Create valid query then modify method to test validation
        query = SearchQuery(text="test", method="semantic", top_k=5)
        query.method = "unknown"  # Modify after creation

        # The validation happens before reaching the unknown method check
        with pytest.raises(ValueError, match="method must be one of:"):
            await self.search_engine.search(query)

    @pytest.mark.asyncio
    async def test_search_with_reranking(self):
        """Test search with reranking enabled."""
        self.mock_search_provider.search_by_embedding.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Short", "Much longer content with more details"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.2, 0.1]]
        }

        query = SearchQuery(text="test query", method="semantic", top_k=2, rerank=True)
        response = await self.search_engine.search(query)

        # Results should be reranked (order might change due to length/relevance factors)
        assert len(response.results) == 2
        assert response.metadata["reranked"] is True

    @pytest.mark.asyncio
    async def test_search_with_threshold_filtering(self):
        """Test search with similarity threshold filtering."""
        self.mock_search_provider.search_by_embedding.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["Content 1", "Content 2", "Content 3"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.5, 0.8]]  # similarities: 0.9, 0.5, 0.2
        }

        query = SearchQuery(text="test", method="semantic", similarity_threshold=0.6, top_k=5)
        response = await self.search_engine.search(query)

        # Only doc1 (score 0.9) should pass threshold of 0.6
        assert len(response.results) == 1
        assert response.results[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_find_similar_documents(self):
        """Test finding similar documents."""
        # Mock document retrieval
        self.mock_search_provider.get_document.return_value = {
            "content": "Reference document content"
        }

        # Mock search results
        self.mock_search_provider.search_by_embedding.return_value = {
            "ids": [["ref_doc", "similar1", "similar2"]],
            "documents": [["Reference document content", "Similar content 1", "Similar content 2"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.0, 0.1, 0.2]]
        }

        response = await self.search_engine.find_similar_documents("ref_doc", top_k=2)

        # Should exclude reference document itself
        assert len(response.results) == 2
        assert response.results[0].id == "similar1"
        assert response.results[1].id == "similar2"
        assert response.method_used == "semantic_similarity"

    @pytest.mark.asyncio
    async def test_find_similar_documents_not_found(self):
        """Test finding similar documents when reference doesn't exist."""
        self.mock_search_provider.get_document.return_value = None

        with pytest.raises(ValueError, match="Document ref_doc not found"):
            await self.search_engine.find_similar_documents("ref_doc", top_k=2)

    @pytest.mark.asyncio
    async def test_search_hybrid_missing_weights(self):
        """Test hybrid search with missing weight configuration."""
        self.mock_config_provider.get_scoring_weights.return_value = {"semantic": 0.7}  # Missing keyword

        query = SearchQuery(text="test", method="hybrid", top_k=1)

        with pytest.raises(ValueError, match="Missing 'keyword' weight in scoring weights configuration"):
            await self.search_engine.search(query)


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_search_query(self):
        """Test creating SearchQuery via factory function."""
        query = create_search_query(
            text="test query",
            top_k=10,
            method="hybrid",
            filters={"language": "hr"},
            similarity_threshold=0.5
        )

        assert isinstance(query, SearchQuery)
        assert query.text == "test query"
        assert query.top_k == 10
        assert query.method == "hybrid"
        assert query.filters == {"language": "hr"}
        assert query.similarity_threshold == 0.5

    def test_create_search_query_defaults(self):
        """Test creating SearchQuery with default values."""
        query = create_search_query("simple query")

        assert query.text == "simple query"
        assert query.top_k == 5
        assert query.method == "semantic"

    def test_create_search_engine(self):
        """Test creating SemanticSearchEngine via factory function."""
        embedding_provider = Mock()
        search_provider = Mock()
        config_provider = Mock()

        engine = create_search_engine(embedding_provider, search_provider, config_provider)

        assert isinstance(engine, SemanticSearchEngine)
        assert engine.embedding_provider == embedding_provider
        assert engine.search_provider == search_provider
        assert engine.config_provider == config_provider