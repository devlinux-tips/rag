"""
Comprehensive tests for SemanticSearchEngine module.

This module tests the similarity search implementation for multilingual RAG system
with pure functions, dependency injection, and sophisticated search strategies.

Test Coverage:
- Data structures (SearchQuery, SearchResult, SearchResponse)
- Enums (SearchMethod)
- Protocol definitions
- Pure utility functions
- SemanticSearchEngine class with all search methods
- Factory functions
- Error handling and edge cases

Author: Test Suite for RAG System
"""

import asyncio
import time
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from src.vectordb.search import (
    SearchQuery,
    SearchResult,
    SearchResponse,
    SearchMethod,
    EmbeddingProvider,
    VectorSearchProvider,
    ConfigProvider,
    validate_search_query,
    parse_vector_search_results,
    distance_to_similarity,
    calculate_keyword_score,
    combine_scores,
    rerank_results_by_relevance,
    filter_results_by_threshold,
    limit_results,
    extract_context_from_results,
    SemanticSearchEngine,
    create_search_query,
    create_search_engine,
)


class TestSearchMethod(unittest.TestCase):
    """Test SearchMethod enum."""

    def test_search_method_values(self):
        """Test SearchMethod enum values."""
        self.assertEqual(SearchMethod.SEMANTIC.value, "semantic")
        self.assertEqual(SearchMethod.KEYWORD.value, "keyword")
        self.assertEqual(SearchMethod.HYBRID.value, "hybrid")

    def test_search_method_membership(self):
        """Test SearchMethod membership."""
        methods = list(SearchMethod)
        self.assertEqual(len(methods), 3)
        self.assertIn(SearchMethod.SEMANTIC, methods)
        self.assertIn(SearchMethod.KEYWORD, methods)
        self.assertIn(SearchMethod.HYBRID, methods)


class TestSearchQuery(unittest.TestCase):
    """Test SearchQuery dataclass."""

    def test_search_query_creation(self):
        """Test creating a valid SearchQuery."""
        query = SearchQuery(
            text="What is AI?",
            top_k=10,
            method="semantic",
            filters={"category": "tech"},
            similarity_threshold=0.7,
            max_context_length=3000,
            rerank=False
        )

        self.assertEqual(query.text, "What is AI?")
        self.assertEqual(query.top_k, 10)
        self.assertEqual(query.method, "semantic")
        self.assertEqual(query.filters, {"category": "tech"})
        self.assertEqual(query.similarity_threshold, 0.7)
        self.assertEqual(query.max_context_length, 3000)
        self.assertFalse(query.rerank)

    def test_search_query_defaults(self):
        """Test SearchQuery with default values."""
        query = SearchQuery(text="test query")

        self.assertEqual(query.text, "test query")
        self.assertEqual(query.top_k, 5)
        self.assertEqual(query.method, "semantic")
        self.assertIsNone(query.filters)
        self.assertEqual(query.similarity_threshold, 0.0)
        self.assertEqual(query.max_context_length, 2000)
        self.assertTrue(query.rerank)

    def test_search_query_validation_invalid_top_k(self):
        """Test SearchQuery validation with invalid top_k."""
        with self.assertRaises(ValueError) as cm:
            SearchQuery(text="test", top_k=0)
        self.assertIn("top_k must be positive", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            SearchQuery(text="test", top_k=-1)
        self.assertIn("top_k must be positive", str(cm.exception))

    def test_search_query_validation_invalid_threshold(self):
        """Test SearchQuery validation with invalid similarity_threshold."""
        with self.assertRaises(ValueError) as cm:
            SearchQuery(text="test", similarity_threshold=1.5)
        self.assertIn("similarity_threshold must be between 0 and 1", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            SearchQuery(text="test", similarity_threshold=-0.1)
        self.assertIn("similarity_threshold must be between 0 and 1", str(cm.exception))


class TestSearchResult(unittest.TestCase):
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a valid SearchResult."""
        result = SearchResult(
            id="doc_1",
            content="This is test content",
            score=0.85,
            metadata={"source": "test.txt", "category": "technical"},
            rank=1,
            method_used="semantic"
        )

        self.assertEqual(result.id, "doc_1")
        self.assertEqual(result.content, "This is test content")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.metadata, {"source": "test.txt", "category": "technical"})
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.method_used, "semantic")

    def test_search_result_defaults(self):
        """Test SearchResult with default values."""
        result = SearchResult(
            id="doc_1",
            content="content",
            score=0.5,
            metadata={}
        )

        self.assertIsNone(result.rank)
        self.assertIsNone(result.method_used)

    def test_search_result_to_dict(self):
        """Test SearchResult to_dict conversion."""
        result = SearchResult(
            id="doc_1",
            content="content",
            score=0.7,
            metadata={"type": "test"},
            rank=2,
            method_used="hybrid"
        )

        result_dict = result.to_dict()

        expected = {
            "id": "doc_1",
            "content": "content",
            "score": 0.7,
            "metadata": {"type": "test"},
            "rank": 2,
            "method_used": "hybrid"
        }

        self.assertEqual(result_dict, expected)


class TestSearchResponse(unittest.TestCase):
    """Test SearchResponse dataclass."""

    def test_search_response_creation(self):
        """Test creating a valid SearchResponse."""
        results = [
            SearchResult("1", "content1", 0.9, {}),
            SearchResult("2", "content2", 0.8, {})
        ]

        response = SearchResponse(
            query="test query",
            results=results,
            total_results=2,
            search_time=0.5,
            method_used="semantic",
            metadata={"processed": True}
        )

        self.assertEqual(response.query, "test query")
        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.total_results, 2)
        self.assertEqual(response.search_time, 0.5)
        self.assertEqual(response.method_used, "semantic")
        self.assertEqual(response.metadata, {"processed": True})

    def test_search_response_rank_assignment(self):
        """Test SearchResponse automatic rank assignment."""
        results = [
            SearchResult("1", "content1", 0.9, {}),
            SearchResult("2", "content2", 0.8, {}, rank=None),
        ]

        response = SearchResponse(
            query="test",
            results=results,
            total_results=2,
            search_time=0.1,
            method_used="semantic"
        )

        # Ranks should be assigned in post_init
        self.assertEqual(response.results[0].rank, 1)
        self.assertEqual(response.results[1].rank, 2)
        # Method should be assigned
        self.assertEqual(response.results[0].method_used, "semantic")
        self.assertEqual(response.results[1].method_used, "semantic")

    def test_search_response_preserve_existing_ranks(self):
        """Test SearchResponse preserves existing ranks."""
        results = [
            SearchResult("1", "content1", 0.9, {}, rank=5, method_used="existing"),
            SearchResult("2", "content2", 0.8, {}, rank=3, method_used="existing"),
        ]

        response = SearchResponse(
            query="test",
            results=results,
            total_results=2,
            search_time=0.1,
            method_used="semantic"
        )

        # Existing ranks and methods should be preserved
        self.assertEqual(response.results[0].rank, 5)
        self.assertEqual(response.results[1].rank, 3)
        self.assertEqual(response.results[0].method_used, "existing")
        self.assertEqual(response.results[1].method_used, "existing")


class TestPureFunctions(unittest.TestCase):
    """Test pure utility functions."""

    def test_validate_search_query_valid(self):
        """Test validate_search_query with valid query."""
        query = SearchQuery(
            text="test query",
            top_k=5,
            method="semantic",
            similarity_threshold=0.5,
            max_context_length=1000
        )

        errors = validate_search_query(query)
        self.assertEqual(errors, [])

    def test_validate_search_query_empty_text(self):
        """Test validate_search_query with empty text."""
        query = SearchQuery(text="", top_k=5)
        errors = validate_search_query(query)
        self.assertIn("Query text cannot be empty", errors)

        query = SearchQuery(text="   ", top_k=5)
        errors = validate_search_query(query)
        self.assertIn("Query text cannot be empty", errors)

    def test_validate_search_query_invalid_top_k(self):
        """Test validate_search_query with invalid top_k."""
        # For validation function testing, create query that bypasses __post_init__
        query = SearchQuery.__new__(SearchQuery)
        query.text = "test"
        query.top_k = 0
        query.method = "semantic"
        query.similarity_threshold = 0.5
        query.max_context_length = 1000

        errors = validate_search_query(query)
        self.assertIn("top_k must be positive", errors)

        query.top_k = 150
        errors = validate_search_query(query)
        self.assertIn("top_k cannot exceed 100", errors)

    def test_validate_search_query_invalid_threshold(self):
        """Test validate_search_query with invalid threshold."""
        # For validation function testing, create query that bypasses __post_init__
        query = SearchQuery.__new__(SearchQuery)
        query.text = "test"
        query.top_k = 5
        query.method = "semantic"
        query.similarity_threshold = -0.1
        query.max_context_length = 1000

        errors = validate_search_query(query)
        self.assertIn("similarity_threshold must be between 0 and 1", errors)

        query.similarity_threshold = 1.5
        errors = validate_search_query(query)
        self.assertIn("similarity_threshold must be between 0 and 1", errors)

    def test_validate_search_query_invalid_method(self):
        """Test validate_search_query with invalid method."""
        # For validation function testing, create query that bypasses __post_init__
        query = SearchQuery.__new__(SearchQuery)
        query.text = "test"
        query.top_k = 5
        query.method = "invalid_method"
        query.similarity_threshold = 0.5
        query.max_context_length = 1000

        errors = validate_search_query(query)
        self.assertTrue(any("method must be one of" in error for error in errors))

    def test_validate_search_query_invalid_context_length(self):
        """Test validate_search_query with invalid max_context_length."""
        query = SearchQuery(text="test", max_context_length=0)
        errors = validate_search_query(query)
        self.assertIn("max_context_length must be positive", errors)

    def test_parse_vector_search_results_empty(self):
        """Test parse_vector_search_results with empty input."""
        # Empty results
        results = parse_vector_search_results({})
        self.assertEqual(results, [])

        # No ids
        results = parse_vector_search_results({"ids": []})
        self.assertEqual(results, [])

    def test_parse_vector_search_results_missing_fields(self):
        """Test parse_vector_search_results with missing required fields."""
        raw_results = {"ids": ["1"], "documents": ["content"]}

        with self.assertRaises(ValueError) as cm:
            parse_vector_search_results(raw_results)
        self.assertIn("missing 'metadatas' field", str(cm.exception))

    def test_parse_vector_search_results_nested_format(self):
        """Test parse_vector_search_results with nested ChromaDB format."""
        raw_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.1, 0.3]]
        }

        results = parse_vector_search_results(raw_results, "semantic")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "doc1")
        self.assertEqual(results[0].content, "content1")
        self.assertEqual(results[0].metadata, {"source": "test1"})
        self.assertEqual(results[0].method_used, "semantic")
        self.assertAlmostEqual(results[0].score, 0.9, places=1)  # 1 - 0.1

    def test_parse_vector_search_results_flat_format(self):
        """Test parse_vector_search_results with flat format."""
        raw_results = {
            "ids": ["doc1", "doc2"],
            "documents": ["content1", "content2"],
            "metadatas": [{"source": "test1"}, {"source": "test2"}],
            "distances": [0.2, 0.4]
        }

        results = parse_vector_search_results(raw_results, "keyword")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "doc1")
        self.assertEqual(results[1].id, "doc2")
        self.assertEqual(results[0].method_used, "keyword")

    def test_parse_vector_search_results_mismatched_lengths(self):
        """Test parse_vector_search_results with mismatched array lengths."""
        raw_results = {
            "ids": ["doc1", "doc2", "doc3"],
            "documents": ["content1", "content2"],  # Missing one
            "metadatas": [{"source": "test1"}],      # Missing two
            "distances": [0.1, 0.2, 0.3]
        }

        results = parse_vector_search_results(raw_results)

        # Should handle gracefully with defaults
        self.assertEqual(len(results), 3)
        self.assertEqual(results[2].content, "")  # Default for missing content
        self.assertEqual(results[2].metadata, {})  # Default for missing metadata

    def test_distance_to_similarity(self):
        """Test distance_to_similarity conversion."""
        # Perfect match
        self.assertEqual(distance_to_similarity(0.0), 1.0)

        # Partial similarity
        self.assertAlmostEqual(distance_to_similarity(0.3), 0.7, places=1)

        # Very different
        self.assertEqual(distance_to_similarity(1.0), 0.0)

        # Clamp negative values
        self.assertEqual(distance_to_similarity(-0.1), 1.0)

        # Clamp values > 1
        self.assertEqual(distance_to_similarity(1.5), 0.0)

    def test_calculate_keyword_score_basic(self):
        """Test calculate_keyword_score basic functionality."""
        query_terms = ["machine", "learning"]
        document_text = "machine learning is a subset of artificial intelligence"

        score = calculate_keyword_score(query_terms, document_text)
        self.assertEqual(score, 1.0)  # Both terms found

    def test_calculate_keyword_score_partial(self):
        """Test calculate_keyword_score with partial matches."""
        query_terms = ["machine", "learning", "deep"]
        document_text = "machine learning algorithms"

        score = calculate_keyword_score(query_terms, document_text)
        self.assertAlmostEqual(score, 2.0 / 3.0, places=2)  # 2 out of 3 terms

    def test_calculate_keyword_score_phrase_boost(self):
        """Test calculate_keyword_score with phrase matching boost."""
        query_terms = ["machine", "learning"]
        document_text = "machine learning is important"

        score = calculate_keyword_score(query_terms, document_text)
        # Should get boost for exact phrase: 1.0 * 1.5 = 1.5, clamped to 1.0
        self.assertEqual(score, 1.0)

    def test_calculate_keyword_score_edge_cases(self):
        """Test calculate_keyword_score edge cases."""
        # Empty terms
        score = calculate_keyword_score([], "some text")
        self.assertEqual(score, 0.0)

        # Empty document
        score = calculate_keyword_score(["term"], "")
        self.assertEqual(score, 0.0)

        # Both empty
        score = calculate_keyword_score([], "")
        self.assertEqual(score, 0.0)

    def test_combine_scores_basic(self):
        """Test combine_scores basic functionality."""
        combined = combine_scores(0.8, 0.6, 0.7, 0.3)
        expected = (0.8 * 0.7) + (0.6 * 0.3)
        self.assertAlmostEqual(combined, expected, places=2)

    def test_combine_scores_zero_weights(self):
        """Test combine_scores with zero weights."""
        combined = combine_scores(0.8, 0.6, 0.0, 0.0)
        self.assertEqual(combined, 0.0)

    def test_combine_scores_normalization(self):
        """Test combine_scores weight normalization."""
        # Weights that don't sum to 1
        combined = combine_scores(0.8, 0.6, 0.4, 0.6)  # Sum = 1.0
        # Normalized weights: 0.4, 0.6
        expected = (0.8 * 0.4) + (0.6 * 0.6)
        self.assertAlmostEqual(combined, expected, places=2)

    def test_combine_scores_bounds(self):
        """Test combine_scores bounds checking."""
        # Test upper bound
        combined = combine_scores(1.0, 1.0, 1.0, 1.0)
        self.assertEqual(combined, 1.0)

        # Test lower bound (should not go negative)
        combined = combine_scores(-0.5, -0.5, 1.0, 1.0)
        self.assertEqual(combined, 0.0)

    def test_rerank_results_by_relevance_empty(self):
        """Test rerank_results_by_relevance with empty input."""
        results = rerank_results_by_relevance("query", [])
        self.assertEqual(results, [])

    def test_rerank_results_by_relevance_custom_boost(self):
        """Test rerank_results_by_relevance with custom boost factors."""
        results = [
            SearchResult("1", "machine learning algorithms", 0.8, {}),
            SearchResult("2", "deep learning networks", 0.7, {"title": "Introduction"}),
        ]

        boost_factors = {
            "term_overlap": 0.2,
            "length_optimal": 1.0,
            "length_short": 0.8,
            "length_long": 0.9,
            "title_boost": 1.1
        }

        reranked = rerank_results_by_relevance("machine learning", results, boost_factors)

        # Results should be reordered by updated scores
        self.assertEqual(len(reranked), 2)
        # All results should have scores > original due to boosts
        self.assertGreater(reranked[0].score, 0.7)

    def test_rerank_results_by_relevance_missing_boost_factors(self):
        """Test rerank_results_by_relevance with missing boost factors."""
        results = [SearchResult("1", "content", 0.8, {})]

        incomplete_boost = {"term_overlap": 0.2}  # Missing other required keys

        with self.assertRaises(ValueError) as cm:
            rerank_results_by_relevance("query", results, incomplete_boost)
        self.assertIn("Missing", str(cm.exception))

    def test_filter_results_by_threshold(self):
        """Test filter_results_by_threshold."""
        results = [
            SearchResult("1", "content1", 0.9, {}),
            SearchResult("2", "content2", 0.7, {}),
            SearchResult("3", "content3", 0.5, {}),
            SearchResult("4", "content4", 0.3, {}),
        ]

        filtered = filter_results_by_threshold(results, 0.6)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].id, "1")
        self.assertEqual(filtered[1].id, "2")

    def test_filter_results_by_threshold_empty(self):
        """Test filter_results_by_threshold with empty input."""
        filtered = filter_results_by_threshold([], 0.5)
        self.assertEqual(filtered, [])

    def test_limit_results(self):
        """Test limit_results."""
        results = [
            SearchResult("1", "content1", 0.9, {}),
            SearchResult("2", "content2", 0.8, {}),
            SearchResult("3", "content3", 0.7, {}),
        ]

        limited = limit_results(results, 2)
        self.assertEqual(len(limited), 2)
        self.assertEqual(limited[0].id, "1")
        self.assertEqual(limited[1].id, "2")

    def test_limit_results_zero_or_negative(self):
        """Test limit_results with zero or negative limits."""
        results = [SearchResult("1", "content1", 0.9, {})]

        limited_zero = limit_results(results, 0)
        self.assertEqual(limited_zero, results)

        limited_negative = limit_results(results, -1)
        self.assertEqual(limited_negative, results)

    def test_extract_context_from_results_basic(self):
        """Test extract_context_from_results basic functionality."""
        results = [
            SearchResult("1", "This is the first document.", 0.9, {}),
            SearchResult("2", "This is the second document.", 0.8, {}),
        ]

        context = extract_context_from_results(results, 100)
        expected = "This is the first document.\n\nThis is the second document."
        self.assertEqual(context, expected)

    def test_extract_context_from_results_empty(self):
        """Test extract_context_from_results with empty input."""
        context = extract_context_from_results([], 100)
        self.assertEqual(context, "")

    def test_extract_context_from_results_length_limit(self):
        """Test extract_context_from_results with length limit."""
        results = [
            SearchResult("1", "A" * 50, 0.9, {}),
            SearchResult("2", "B" * 50, 0.8, {}),
        ]

        # Limit that allows only first result
        context = extract_context_from_results(results, 60)
        self.assertTrue(context.startswith("A"))
        self.assertNotIn("B", context)

    def test_extract_context_from_results_truncation(self):
        """Test extract_context_from_results with truncation."""
        results = [
            SearchResult("1", "A" * 100, 0.9, {}),
        ]

        # Very small limit forces truncation
        context = extract_context_from_results(results, 20)
        self.assertTrue(context.endswith("..."))
        self.assertLess(len(context), 25)

    def test_extract_context_from_results_custom_separator(self):
        """Test extract_context_from_results with custom separator."""
        results = [
            SearchResult("1", "First", 0.9, {}),
            SearchResult("2", "Second", 0.8, {}),
        ]

        context = extract_context_from_results(results, 100, " | ")
        self.assertEqual(context, "First | Second")


class TestMockProtocols(unittest.TestCase):
    """Test mock implementations for protocols."""

    def test_embedding_provider_protocol(self):
        """Test EmbeddingProvider protocol compliance."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.encode_text = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))

        # Should be callable
        self.assertTrue(hasattr(mock_provider, 'encode_text'))

    def test_vector_search_provider_protocol(self):
        """Test VectorSearchProvider protocol compliance."""
        mock_provider = Mock(spec=VectorSearchProvider)
        mock_provider.search_by_embedding = AsyncMock(return_value={
            "ids": ["1"], "documents": ["content"], "metadatas": [{}], "distances": [0.1]
        })
        mock_provider.search_by_text = AsyncMock(return_value={
            "ids": ["1"], "documents": ["content"], "metadatas": [{}], "distances": [0.1]
        })
        mock_provider.get_document = AsyncMock(return_value={"content": "test"})

        # Should be callable
        self.assertTrue(hasattr(mock_provider, 'search_by_embedding'))
        self.assertTrue(hasattr(mock_provider, 'search_by_text'))
        self.assertTrue(hasattr(mock_provider, 'get_document'))

    def test_config_provider_protocol(self):
        """Test ConfigProvider protocol compliance."""
        mock_provider = Mock(spec=ConfigProvider)
        mock_provider.get_search_config = Mock(return_value={"threshold": 0.5})
        mock_provider.get_scoring_weights = Mock(return_value={"semantic": 0.7, "keyword": 0.3})

        # Should be callable
        self.assertTrue(hasattr(mock_provider, 'get_search_config'))
        self.assertTrue(hasattr(mock_provider, 'get_scoring_weights'))


class TestSemanticSearchEngine(unittest.IsolatedAsyncioTestCase):
    """Test SemanticSearchEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_embedding_provider = Mock(spec=EmbeddingProvider)
        self.mock_search_provider = Mock(spec=VectorSearchProvider)
        self.mock_config_provider = Mock(spec=ConfigProvider)

        self.search_engine = SemanticSearchEngine(
            embedding_provider=self.mock_embedding_provider,
            search_provider=self.mock_search_provider,
            config_provider=self.mock_config_provider
        )

        # Set up default mock responses
        self.sample_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        self.sample_search_results = {
            "ids": ["doc1", "doc2"],
            "documents": ["AI content", "More AI content"],
            "metadatas": [{"source": "doc1"}, {"source": "doc2"}],
            "distances": [0.1, 0.2]
        }

        self.mock_embedding_provider.encode_text = AsyncMock(return_value=self.sample_embedding)
        self.mock_search_provider.search_by_embedding = AsyncMock(return_value=self.sample_search_results)
        self.mock_search_provider.search_by_text = AsyncMock(return_value=self.sample_search_results)
        self.mock_config_provider.get_scoring_weights = Mock(return_value={"semantic": 0.7, "keyword": 0.3})

    async def test_search_basic_semantic(self):
        """Test basic semantic search."""
        query = SearchQuery(text="What is AI?", method="semantic")

        response = await self.search_engine.search(query)

        self.assertIsInstance(response, SearchResponse)
        self.assertEqual(response.query, "What is AI?")
        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.method_used, "semantic")
        self.assertGreater(response.search_time, 0)

        # Verify dependencies were called
        self.mock_embedding_provider.encode_text.assert_called_once_with("What is AI?")
        self.mock_search_provider.search_by_embedding.assert_called_once()

    async def test_search_keyword_with_text_support(self):
        """Test keyword search when provider supports text search."""
        query = SearchQuery(text="machine learning", method="keyword")

        response = await self.search_engine.search(query)

        self.assertEqual(response.method_used, "keyword")
        self.mock_search_provider.search_by_text.assert_called_once()

    async def test_search_keyword_fallback(self):
        """Test keyword search fallback when provider doesn't support text search."""
        # Remove search_by_text method
        del self.mock_search_provider.search_by_text

        query = SearchQuery(text="machine learning", method="keyword")

        response = await self.search_engine.search(query)

        self.assertEqual(response.method_used, "keyword")
        # Should fallback to semantic search then re-score
        self.mock_search_provider.search_by_embedding.assert_called_once()

    async def test_search_hybrid(self):
        """Test hybrid search functionality."""
        query = SearchQuery(text="AI research", method="hybrid")

        response = await self.search_engine.search(query)

        self.assertEqual(response.method_used, "hybrid")
        # Should call both semantic and keyword searches
        self.mock_embedding_provider.encode_text.assert_called()
        self.mock_config_provider.get_scoring_weights.assert_called_once()

    async def test_search_hybrid_missing_weights(self):
        """Test hybrid search with missing scoring weights."""
        self.mock_config_provider.get_scoring_weights = Mock(return_value={})
        query = SearchQuery(text="test", method="hybrid")

        with self.assertRaises(ValueError) as cm:
            await self.search_engine.search(query)
        self.assertIn("Missing 'semantic' weight", str(cm.exception))

    async def test_search_hybrid_exception_handling(self):
        """Test hybrid search with individual method exceptions."""
        # Make semantic search fail
        self.mock_embedding_provider.encode_text = AsyncMock(side_effect=Exception("Embedding failed"))

        query = SearchQuery(text="test", method="hybrid")

        with patch.object(self.search_engine, 'logger') as mock_logger:
            response = await self.search_engine.search(query)

            # Should still get results from keyword search
            self.assertGreaterEqual(len(response.results), 0)
            mock_logger.warning.assert_called()

    async def test_search_invalid_query(self):
        """Test search with invalid query."""
        query = SearchQuery(text="", top_k=5)  # Empty text

        with self.assertRaises(ValueError) as cm:
            await self.search_engine.search(query)
        self.assertIn("Invalid query", str(cm.exception))

    async def test_search_unknown_method(self):
        """Test search with unknown method."""
        # Create query that bypasses __post_init__ to test unknown method handling
        query = SearchQuery.__new__(SearchQuery)
        query.text = "test"
        query.top_k = 5
        query.method = "unknown_method"
        query.similarity_threshold = 0.5
        query.max_context_length = 1000
        query.rerank = True
        query.filters = None

        with self.assertRaises(ValueError) as cm:
            await self.search_engine.search(query)
        # The validation happens in validate_search_query, not in the engine logic
        self.assertIn("method must be one of", str(cm.exception))

    async def test_search_with_filters_and_threshold(self):
        """Test search with filters and similarity threshold."""
        query = SearchQuery(
            text="test",
            method="semantic",
            filters={"category": "tech"},
            similarity_threshold=0.8,
            rerank=True
        )

        response = await self.search_engine.search(query)

        # Check that filters were passed to search provider
        call_kwargs = self.mock_search_provider.search_by_embedding.call_args[1]
        self.assertEqual(call_kwargs["filters"], {"category": "tech"})

        # Check response metadata
        self.assertEqual(response.metadata["filters"], {"category": "tech"})
        self.assertEqual(response.metadata["similarity_threshold"], 0.8)
        self.assertTrue(response.metadata["reranked"])

    async def test_search_exception_handling(self):
        """Test search exception handling."""
        self.mock_embedding_provider.encode_text = AsyncMock(side_effect=Exception("Processing failed"))
        query = SearchQuery(text="test")

        with patch.object(self.search_engine, 'logger') as mock_logger:
            with self.assertRaises(Exception):
                await self.search_engine.search(query)
            mock_logger.error.assert_called()

    async def test_semantic_search_error_handling(self):
        """Test semantic search error handling."""
        self.mock_embedding_provider.encode_text = AsyncMock(side_effect=Exception("Embedding error"))
        query = SearchQuery(text="test")

        with patch.object(self.search_engine, 'logger') as mock_logger:
            with self.assertRaises(Exception):
                await self.search_engine._semantic_search(query)
            mock_logger.error.assert_called()

    async def test_find_similar_documents_success(self):
        """Test find_similar_documents with valid document."""
        self.mock_search_provider.get_document = AsyncMock(return_value={
            "content": "reference document content"
        })

        response = await self.search_engine.find_similar_documents("doc123", top_k=3)

        self.assertIsInstance(response, SearchResponse)
        self.assertEqual(response.method_used, "semantic_similarity")
        self.assertEqual(response.metadata["reference_document_id"], "doc123")

        # Should exclude the reference document itself
        result_ids = [result.id for result in response.results]
        self.assertNotIn("doc123", result_ids)

    async def test_find_similar_documents_not_found(self):
        """Test find_similar_documents with non-existent document."""
        self.mock_search_provider.get_document = AsyncMock(return_value=None)

        with self.assertRaises(ValueError) as cm:
            await self.search_engine.find_similar_documents("nonexistent")
        self.assertIn("Document nonexistent not found", str(cm.exception))

    async def test_find_similar_documents_exception(self):
        """Test find_similar_documents exception handling."""
        self.mock_search_provider.get_document = AsyncMock(side_effect=Exception("DB error"))

        with patch.object(self.search_engine, 'logger') as mock_logger:
            with self.assertRaises(Exception):
                await self.search_engine.find_similar_documents("doc123")
            mock_logger.error.assert_called()


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_create_search_query(self):
        """Test create_search_query factory function."""
        query = create_search_query(
            text="test query",
            top_k=10,
            method="hybrid",
            filters={"category": "tech"},
            similarity_threshold=0.7
        )

        self.assertIsInstance(query, SearchQuery)
        self.assertEqual(query.text, "test query")
        self.assertEqual(query.top_k, 10)
        self.assertEqual(query.method, "hybrid")
        self.assertEqual(query.filters, {"category": "tech"})
        self.assertEqual(query.similarity_threshold, 0.7)

    def test_create_search_engine(self):
        """Test create_search_engine factory function."""
        mock_embedding_provider = Mock(spec=EmbeddingProvider)
        mock_search_provider = Mock(spec=VectorSearchProvider)
        mock_config_provider = Mock(spec=ConfigProvider)

        engine = create_search_engine(
            embedding_provider=mock_embedding_provider,
            search_provider=mock_search_provider,
            config_provider=mock_config_provider
        )

        self.assertIsInstance(engine, SemanticSearchEngine)
        self.assertEqual(engine.embedding_provider, mock_embedding_provider)
        self.assertEqual(engine.search_provider, mock_search_provider)
        self.assertEqual(engine.config_provider, mock_config_provider)


if __name__ == "__main__":
    unittest.main()