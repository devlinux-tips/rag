"""
Comprehensive tests for DocumentRetriever module.

This module tests the multilingual document retrieval system with pure functions,
dependency injection, and sophisticated retrieval strategies.

Test Coverage:
- Enums (RetrievalStrategy, QueryType)
- Data structures (RetrievalQuery, RetrievedDocument, RetrievalResult)
- Protocol definitions
- Pure utility functions
- DocumentRetriever class with all strategies
- Factory functions
- Error handling and edge cases

Author: Test Suite for RAG System
"""

import asyncio
import time
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

from src.retrieval.retriever import (
    RetrievalStrategy,
    QueryType,
    RetrievalQuery,
    RetrievedDocument,
    RetrievalResult,
    QueryProcessor,
    SearchEngine,
    ResultRanker,
    RetrievalConfig,
    select_retrieval_strategy,
    calculate_adaptive_top_k,
    merge_retrieval_results,
    filter_results_by_threshold,
    limit_results,
    add_query_match_analysis,
    calculate_diversity_score,
    DocumentRetriever,
    create_retrieval_query,
    create_document_retriever,
)


class TestRetrievalStrategy(unittest.TestCase):
    """Test RetrievalStrategy enum."""

    def test_retrieval_strategy_values(self):
        """Test RetrievalStrategy enum values."""
        self.assertEqual(RetrievalStrategy.SEMANTIC.value, "semantic")
        self.assertEqual(RetrievalStrategy.KEYWORD.value, "keyword")
        self.assertEqual(RetrievalStrategy.HYBRID.value, "hybrid")
        self.assertEqual(RetrievalStrategy.ADAPTIVE.value, "adaptive")
        self.assertEqual(RetrievalStrategy.MULTI_PASS.value, "multi_pass")

    def test_retrieval_strategy_membership(self):
        """Test RetrievalStrategy membership."""
        strategies = list(RetrievalStrategy)
        self.assertEqual(len(strategies), 5)
        self.assertIn(RetrievalStrategy.SEMANTIC, strategies)
        self.assertIn(RetrievalStrategy.KEYWORD, strategies)


class TestQueryType(unittest.TestCase):
    """Test QueryType enum."""

    def test_query_type_values(self):
        """Test QueryType enum values."""
        self.assertEqual(QueryType.FACTUAL.value, "factual")
        self.assertEqual(QueryType.CONCEPTUAL.value, "conceptual")
        self.assertEqual(QueryType.PROCEDURAL.value, "procedural")
        self.assertEqual(QueryType.COMPARATIVE.value, "comparative")
        self.assertEqual(QueryType.EXPLORATORY.value, "exploratory")

    def test_query_type_membership(self):
        """Test QueryType membership."""
        types = list(QueryType)
        self.assertEqual(len(types), 5)
        self.assertIn(QueryType.FACTUAL, types)
        self.assertIn(QueryType.EXPLORATORY, types)


class TestRetrievalQuery(unittest.TestCase):
    """Test RetrievalQuery dataclass."""

    def test_retrieval_query_creation(self):
        """Test creating a valid RetrievalQuery."""
        query = RetrievalQuery(
            original_text="What is AI?",
            processed_text="ai artificial intelligence",
            query_type=QueryType.FACTUAL,
            language="en",
            keywords=["ai", "artificial", "intelligence"],
            max_results=5,
            similarity_threshold=0.7
        )

        self.assertEqual(query.original_text, "What is AI?")
        self.assertEqual(query.processed_text, "ai artificial intelligence")
        self.assertEqual(query.query_type, QueryType.FACTUAL)
        self.assertEqual(query.language, "en")
        self.assertEqual(query.keywords, ["ai", "artificial", "intelligence"])
        self.assertIsNone(query.filters)
        self.assertEqual(query.max_results, 5)
        self.assertEqual(query.similarity_threshold, 0.7)

    def test_retrieval_query_defaults(self):
        """Test RetrievalQuery with default values."""
        query = RetrievalQuery(
            original_text="test",
            processed_text="test",
            query_type=QueryType.FACTUAL,
            language="en",
            keywords=["test"]
        )

        self.assertIsNone(query.filters)
        self.assertEqual(query.max_results, 10)
        self.assertEqual(query.similarity_threshold, 0.1)
        self.assertIsNone(query.strategy_override)

    def test_retrieval_query_validation_invalid_max_results(self):
        """Test RetrievalQuery validation with invalid max_results."""
        with self.assertRaises(ValueError) as cm:
            RetrievalQuery(
                original_text="test",
                processed_text="test",
                query_type=QueryType.FACTUAL,
                language="en",
                keywords=["test"],
                max_results=0
            )
        self.assertIn("max_results must be positive", str(cm.exception))

    def test_retrieval_query_validation_invalid_threshold(self):
        """Test RetrievalQuery validation with invalid similarity_threshold."""
        with self.assertRaises(ValueError) as cm:
            RetrievalQuery(
                original_text="test",
                processed_text="test",
                query_type=QueryType.FACTUAL,
                language="en",
                keywords=["test"],
                similarity_threshold=1.5
            )
        self.assertIn("similarity_threshold must be between 0 and 1", str(cm.exception))

    def test_retrieval_query_validation_negative_threshold(self):
        """Test RetrievalQuery validation with negative threshold."""
        with self.assertRaises(ValueError) as cm:
            RetrievalQuery(
                original_text="test",
                processed_text="test",
                query_type=QueryType.FACTUAL,
                language="en",
                keywords=["test"],
                similarity_threshold=-0.1
            )
        self.assertIn("similarity_threshold must be between 0 and 1", str(cm.exception))


class TestRetrievedDocument(unittest.TestCase):
    """Test RetrievedDocument dataclass."""

    def test_retrieved_document_creation(self):
        """Test creating a valid RetrievedDocument."""
        doc = RetrievedDocument(
            id="doc_1",
            content="This is test content",
            score=0.85,
            metadata={"source": "test.txt", "category": "technical"},
            retrieval_method="semantic",
            rank=1,
            query_match_info={"keywords": ["test"]}
        )

        self.assertEqual(doc.id, "doc_1")
        self.assertEqual(doc.content, "This is test content")
        self.assertEqual(doc.score, 0.85)
        self.assertEqual(doc.metadata, {"source": "test.txt", "category": "technical"})
        self.assertEqual(doc.retrieval_method, "semantic")
        self.assertEqual(doc.rank, 1)
        self.assertEqual(doc.query_match_info, {"keywords": ["test"]})

    def test_retrieved_document_defaults(self):
        """Test RetrievedDocument with default values."""
        doc = RetrievedDocument(
            id="doc_1",
            content="content",
            score=0.5,
            metadata={},
            retrieval_method="keyword"
        )

        self.assertIsNone(doc.rank)
        self.assertIsNone(doc.query_match_info)

    def test_retrieved_document_to_dict(self):
        """Test RetrievedDocument to_dict conversion."""
        doc = RetrievedDocument(
            id="doc_1",
            content="content",
            score=0.7,
            metadata={"type": "test"},
            retrieval_method="hybrid",
            rank=2,
            query_match_info={"matches": 3}
        )

        result = doc.to_dict()

        expected = {
            "id": "doc_1",
            "content": "content",
            "score": 0.7,
            "metadata": {"type": "test"},
            "retrieval_method": "hybrid",
            "rank": 2,
            "query_match_info": {"matches": 3}
        }

        self.assertEqual(result, expected)


class TestRetrievalResult(unittest.TestCase):
    """Test RetrievalResult dataclass."""

    def test_retrieval_result_creation(self):
        """Test creating a valid RetrievalResult."""
        docs = [
            RetrievedDocument("1", "content1", 0.9, {}, "semantic"),
            RetrievedDocument("2", "content2", 0.8, {}, "semantic")
        ]

        result = RetrievalResult(
            query="test query",
            documents=docs,
            total_found=2,
            retrieval_time=0.5,
            strategy_used=RetrievalStrategy.SEMANTIC,
            query_type=QueryType.FACTUAL,
            language="en",
            metadata={"processed": True}
        )

        self.assertEqual(result.query, "test query")
        self.assertEqual(len(result.documents), 2)
        self.assertEqual(result.total_found, 2)
        self.assertEqual(result.retrieval_time, 0.5)
        self.assertEqual(result.strategy_used, RetrievalStrategy.SEMANTIC)
        self.assertEqual(result.query_type, QueryType.FACTUAL)
        self.assertEqual(result.language, "en")
        self.assertEqual(result.metadata, {"processed": True})

    def test_retrieval_result_rank_assignment(self):
        """Test RetrievalResult automatic rank assignment."""
        docs = [
            RetrievedDocument("1", "content1", 0.9, {}, "semantic"),
            RetrievedDocument("2", "content2", 0.8, {}, "semantic", rank=None),
        ]

        result = RetrievalResult(
            query="test",
            documents=docs,
            total_found=2,
            retrieval_time=0.1,
            strategy_used=RetrievalStrategy.SEMANTIC,
            query_type=QueryType.FACTUAL,
            language="en"
        )

        # Ranks should be assigned in post_init
        self.assertEqual(result.documents[0].rank, 1)
        self.assertEqual(result.documents[1].rank, 2)

    def test_retrieval_result_preserve_existing_ranks(self):
        """Test RetrievalResult preserves existing ranks."""
        docs = [
            RetrievedDocument("1", "content1", 0.9, {}, "semantic", rank=5),
            RetrievedDocument("2", "content2", 0.8, {}, "semantic", rank=3),
        ]

        result = RetrievalResult(
            query="test",
            documents=docs,
            total_found=2,
            retrieval_time=0.1,
            strategy_used=RetrievalStrategy.SEMANTIC,
            query_type=QueryType.FACTUAL,
            language="en"
        )

        # Existing ranks should be preserved
        self.assertEqual(result.documents[0].rank, 5)
        self.assertEqual(result.documents[1].rank, 3)


class TestPureFunctions(unittest.TestCase):
    """Test pure utility functions."""

    def test_select_retrieval_strategy_override(self):
        """Test select_retrieval_strategy with override."""
        query = RetrievalQuery(
            original_text="test",
            processed_text="test",
            query_type=QueryType.FACTUAL,
            language="en",
            keywords=["test"],
            strategy_override=RetrievalStrategy.HYBRID
        )

        result = select_retrieval_strategy(query)
        self.assertEqual(result, RetrievalStrategy.HYBRID)

    def test_select_retrieval_strategy_by_type(self):
        """Test select_retrieval_strategy based on query type."""
        # Test factual -> keyword
        query_factual = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.FACTUAL, language="en", keywords=["test"]
        )
        self.assertEqual(select_retrieval_strategy(query_factual), RetrievalStrategy.KEYWORD)

        # Test conceptual -> semantic
        query_conceptual = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.CONCEPTUAL, language="en", keywords=["test"]
        )
        self.assertEqual(select_retrieval_strategy(query_conceptual), RetrievalStrategy.SEMANTIC)

        # Test procedural -> hybrid
        query_procedural = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.PROCEDURAL, language="en", keywords=["test"]
        )
        self.assertEqual(select_retrieval_strategy(query_procedural), RetrievalStrategy.HYBRID)

    def test_select_retrieval_strategy_keyword_adjustment(self):
        """Test select_retrieval_strategy keyword count adjustment."""
        query = RetrievalQuery(
            original_text="test",
            processed_text="test",
            query_type=QueryType.CONCEPTUAL,  # Would normally be SEMANTIC
            language="en",
            keywords=["word1", "word2", "word3", "word4"]  # >= 3 keywords
        )

        result = select_retrieval_strategy(query)
        # Should be adjusted to HYBRID due to many keywords
        self.assertEqual(result, RetrievalStrategy.HYBRID)

    def test_select_retrieval_strategy_default(self):
        """Test select_retrieval_strategy with default fallback."""
        # Create a query type that's not in the strategy map
        query = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.EXPLORATORY, language="en", keywords=["test"]
        )

        # Should return the mapped strategy for EXPLORATORY
        result = select_retrieval_strategy(query)
        self.assertEqual(result, RetrievalStrategy.SEMANTIC)

    def test_calculate_adaptive_top_k_base(self):
        """Test calculate_adaptive_top_k with base functionality."""
        query = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.PROCEDURAL, language="en", keywords=["test"]
        )

        result = calculate_adaptive_top_k(query, base_top_k=10)
        # PROCEDURAL has multiplier 1.0, keywords count adjustment = 0.9 (<=2 keywords)
        # Expected: 10 * 1.0 * 0.9 = 9
        self.assertEqual(result, 9)

    def test_calculate_adaptive_top_k_bounds(self):
        """Test calculate_adaptive_top_k bounds checking."""
        # Test lower bound
        query_low = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.FACTUAL, language="en", keywords=["test"]
        )
        result_low = calculate_adaptive_top_k(query_low, base_top_k=1)
        self.assertGreaterEqual(result_low, 5)  # Should be clamped to minimum 5

        # Test upper bound
        query_high = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.EXPLORATORY, language="en",
            keywords=["w1", "w2", "w3", "w4", "w5", "w6"]  # >5 keywords
        )
        result_high = calculate_adaptive_top_k(query_high, base_top_k=100)
        self.assertLessEqual(result_high, 50)  # Should be clamped to maximum 50

    def test_merge_retrieval_results_empty(self):
        """Test merge_retrieval_results with empty inputs."""
        # Both empty
        result = merge_retrieval_results([], [])
        self.assertEqual(result, [])

        # One empty
        docs = [RetrievedDocument("1", "content", 0.5, {}, "semantic")]
        result1 = merge_retrieval_results(docs, [])
        self.assertEqual(result1, docs)

        result2 = merge_retrieval_results([], docs)
        self.assertEqual(result2, docs)

    def test_merge_retrieval_results_no_overlap(self):
        """Test merge_retrieval_results with no document overlap."""
        semantic_docs = [RetrievedDocument("1", "content1", 0.8, {}, "semantic")]
        keyword_docs = [RetrievedDocument("2", "content2", 0.6, {}, "keyword")]

        result = merge_retrieval_results(semantic_docs, keyword_docs, 0.7, 0.3)

        self.assertEqual(len(result), 2)
        # Check that scores are weighted properly
        # First doc: 0.8 * 0.7 = 0.56, second doc: 0.6 * 0.3 = 0.18
        # Should be sorted by score (desc), so first doc should have higher score
        self.assertGreater(result[0].score, result[1].score)
        self.assertEqual(result[0].retrieval_method, "hybrid")
        self.assertEqual(result[1].retrieval_method, "hybrid")

    def test_merge_retrieval_results_with_overlap(self):
        """Test merge_retrieval_results with document overlap."""
        semantic_docs = [RetrievedDocument("1", "content1", 0.8, {}, "semantic")]
        keyword_docs = [RetrievedDocument("1", "content1", 0.6, {}, "keyword")]

        result = merge_retrieval_results(semantic_docs, keyword_docs, 0.7, 0.3)

        self.assertEqual(len(result), 1)
        # Combined score: 0.8 * 0.7 + 0.6 * 0.3 = 0.56 + 0.18 = 0.74
        self.assertAlmostEqual(result[0].score, 0.74, places=2)
        self.assertEqual(result[0].retrieval_method, "hybrid")
        self.assertIsNotNone(result[0].query_match_info)
        self.assertEqual(result[0].query_match_info["semantic_score"], 0.8)
        self.assertEqual(result[0].query_match_info["keyword_score"], 0.6)

    def test_merge_retrieval_results_zero_weights(self):
        """Test merge_retrieval_results with zero weights."""
        semantic_docs = [RetrievedDocument("1", "content1", 0.8, {}, "semantic")]
        keyword_docs = [RetrievedDocument("2", "content2", 0.6, {}, "keyword")]

        result = merge_retrieval_results(semantic_docs, keyword_docs, 0.0, 0.0)
        # Should fallback to semantic_docs
        self.assertEqual(result, semantic_docs)

    def test_filter_results_by_threshold(self):
        """Test filter_results_by_threshold."""
        docs = [
            RetrievedDocument("1", "content1", 0.9, {}, "semantic"),
            RetrievedDocument("2", "content2", 0.7, {}, "semantic"),
            RetrievedDocument("3", "content3", 0.5, {}, "semantic"),
            RetrievedDocument("4", "content4", 0.3, {}, "semantic"),
        ]

        result = filter_results_by_threshold(docs, 0.6)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "1")
        self.assertEqual(result[1].id, "2")

    def test_filter_results_by_threshold_empty(self):
        """Test filter_results_by_threshold with empty input."""
        result = filter_results_by_threshold([], 0.5)
        self.assertEqual(result, [])

    def test_limit_results(self):
        """Test limit_results."""
        docs = [
            RetrievedDocument("1", "content1", 0.9, {}, "semantic"),
            RetrievedDocument("2", "content2", 0.8, {}, "semantic"),
            RetrievedDocument("3", "content3", 0.7, {}, "semantic"),
        ]

        result = limit_results(docs, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "1")
        self.assertEqual(result[1].id, "2")

    def test_limit_results_zero_or_negative(self):
        """Test limit_results with zero or negative limits."""
        docs = [RetrievedDocument("1", "content1", 0.9, {}, "semantic")]

        result_zero = limit_results(docs, 0)
        self.assertEqual(result_zero, docs)

        result_negative = limit_results(docs, -1)
        self.assertEqual(result_negative, docs)

    def test_add_query_match_analysis(self):
        """Test add_query_match_analysis."""
        query = RetrievalQuery(
            original_text="test machine learning",
            processed_text="test machine learning",
            query_type=QueryType.FACTUAL,
            language="en",
            keywords=["test", "machine", "learning"]
        )

        docs = [
            RetrievedDocument("1", "This is a test for machine learning algorithms", 0.9, {}, "semantic"),
            RetrievedDocument("2", "Neural networks and deep learning", 0.8, {}, "semantic"),
        ]

        result = add_query_match_analysis(query, docs)

        # First document should have good keyword matches
        self.assertIsNotNone(result[0].query_match_info)
        self.assertIn("keyword_matches", result[0].query_match_info)
        self.assertIn("keyword_match_ratio", result[0].query_match_info)
        self.assertIn("content_length", result[0].query_match_info)

        # Check that keywords are matched correctly
        matches = set(result[0].query_match_info["keyword_matches"])
        expected_matches = {"test", "machine", "learning"}
        self.assertEqual(matches, expected_matches)

    def test_add_query_match_analysis_existing_info(self):
        """Test add_query_match_analysis with existing query_match_info."""
        query = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.FACTUAL, language="en", keywords=["test"]
        )

        docs = [RetrievedDocument("1", "test content", 0.9, {}, "semantic",
                                query_match_info={"existing": "info"})]

        result = add_query_match_analysis(query, docs)

        self.assertIn("existing", result[0].query_match_info)
        self.assertIn("keyword_matches", result[0].query_match_info)

    def test_calculate_diversity_score_single_doc(self):
        """Test calculate_diversity_score with single document."""
        docs = [RetrievedDocument("1", "content", 0.9, {}, "semantic")]
        score = calculate_diversity_score(docs)
        self.assertEqual(score, 1.0)

    def test_calculate_diversity_score_empty(self):
        """Test calculate_diversity_score with empty input."""
        score = calculate_diversity_score([])
        self.assertEqual(score, 1.0)

    def test_calculate_diversity_score_multiple(self):
        """Test calculate_diversity_score with multiple documents."""
        docs = [
            RetrievedDocument("1", "artificial intelligence machine learning", 0.9, {}, "semantic"),
            RetrievedDocument("2", "deep learning neural networks", 0.8, {}, "semantic"),
            RetrievedDocument("3", "computer vision image processing", 0.7, {}, "semantic"),
        ]

        score = calculate_diversity_score(docs)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestMockProtocols(unittest.TestCase):
    """Test mock implementations for protocols."""

    def test_query_processor_protocol(self):
        """Test QueryProcessor protocol compliance."""
        mock_processor = Mock(spec=QueryProcessor)
        mock_processor.process_query = AsyncMock(return_value=RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.FACTUAL, language="en", keywords=["test"]
        ))

        # Should be callable
        self.assertTrue(hasattr(mock_processor, 'process_query'))

    def test_search_engine_protocol(self):
        """Test SearchEngine protocol compliance."""
        mock_engine = Mock(spec=SearchEngine)
        mock_engine.search_by_text = AsyncMock(return_value={
            "ids": [["1"]], "documents": [["test"]], "metadatas": [[{}]], "distances": [[0.2]]
        })

        # Should be callable
        self.assertTrue(hasattr(mock_engine, 'search_by_text'))

    def test_result_ranker_protocol(self):
        """Test ResultRanker protocol compliance."""
        mock_ranker = Mock(spec=ResultRanker)
        docs = [RetrievedDocument("1", "content", 0.8, {}, "semantic")]
        mock_ranker.rank_documents = Mock(return_value=[{"id": "1", "content": "content", "score": 0.8, "metadata": {}, "retrieval_method": "semantic"}])

        # Should be callable
        self.assertTrue(hasattr(mock_ranker, 'rank_documents'))

    def test_retrieval_config_protocol(self):
        """Test RetrievalConfig protocol compliance."""
        mock_config = Mock(spec=RetrievalConfig)
        mock_config.get_strategy_config = Mock(return_value={"semantic_weight": 0.7, "keyword_weight": 0.3})
        mock_config.get_adaptive_config = Mock(return_value={"threshold": 0.5})

        # Should be callable
        self.assertTrue(hasattr(mock_config, 'get_strategy_config'))
        self.assertTrue(hasattr(mock_config, 'get_adaptive_config'))


class TestDocumentRetriever(unittest.IsolatedAsyncioTestCase):
    """Test DocumentRetriever class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_query_processor = Mock(spec=QueryProcessor)
        self.mock_search_engine = Mock(spec=SearchEngine)
        self.mock_result_ranker = Mock(spec=ResultRanker)
        self.mock_config = Mock(spec=RetrievalConfig)

        self.retriever = DocumentRetriever(
            query_processor=self.mock_query_processor,
            search_engine=self.mock_search_engine,
            result_ranker=self.mock_result_ranker,
            config=self.mock_config
        )

        # Set up default mock responses
        self.sample_query = RetrievalQuery(
            original_text="What is AI?",
            processed_text="ai artificial intelligence",
            query_type=QueryType.FACTUAL,
            language="en",
            keywords=["ai", "artificial", "intelligence"]
        )

        self.sample_search_results = {
            "ids": [["1", "2"]],
            "documents": [["AI content", "More AI content"]],
            "metadatas": [[{"source": "doc1"}, {"source": "doc2"}]],
            "distances": [[0.1, 0.2]]
        }

        self.mock_query_processor.process_query = AsyncMock(return_value=self.sample_query)
        self.mock_search_engine.search_by_text = AsyncMock(return_value={
            "ids": [["1", "2"]],
            "documents": [["artificial intelligence content", "machine learning content"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]]
        })
        self.mock_result_ranker.rank_documents = Mock(side_effect=lambda docs, q, context=None: docs)
        self.mock_config.get_strategy_config = Mock(return_value={
            "semantic_weight": 0.7,
            "keyword_weight": 0.3
        })
        self.mock_config.get_adaptive_config = Mock(return_value={"threshold": 0.5})

    async def test_retrieve_documents_basic(self):
        """Test basic document retrieval."""
        result = await self.retriever.retrieve_documents("What is AI?", "en")

        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.query, "What is AI?")
        self.assertEqual(len(result.documents), 2)
        self.assertEqual(result.total_found, 2)
        self.assertGreater(result.retrieval_time, 0)
        self.assertEqual(result.language, "en")

        # Verify dependencies were called
        self.mock_query_processor.process_query.assert_called_once_with("What is AI?", "en")
        self.mock_search_engine.search_by_text.assert_called_once()

    async def test_retrieve_documents_empty_query(self):
        """Test retrieve_documents with empty query."""
        with self.assertRaises(ValueError) as cm:
            await self.retriever.retrieve_documents("", "en")
        self.assertIn("Query cannot be empty", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            await self.retriever.retrieve_documents("   ", "en")
        self.assertIn("Query cannot be empty", str(cm.exception))

    async def test_retrieve_documents_with_overrides(self):
        """Test retrieve_documents with parameter overrides."""
        await self.retriever.retrieve_documents(
            "test query",
            language="hr",
            max_results=5,
            strategy=RetrievalStrategy.HYBRID,
            filters={"category": "tech"}
        )

        # Check that the processed query was modified with overrides
        call_args = self.mock_query_processor.process_query.call_args
        self.assertEqual(call_args[0], ("test query", "hr"))

    async def test_semantic_retrieval(self):
        """Test semantic retrieval strategy."""
        result = await self.retriever._semantic_retrieval(self.sample_query)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].retrieval_method, "semantic")
        self.assertEqual(result[1].retrieval_method, "semantic")

        # Verify search was called with correct parameters
        self.mock_search_engine.search_by_text.assert_called_once()
        call_args = self.mock_search_engine.search_by_text.call_args[1]
        self.assertIn("query_text", call_args)
        self.assertIn("top_k", call_args)

    async def test_semantic_retrieval_missing_fields(self):
        """Test semantic retrieval with empty Chroma results."""
        # Test empty results
        self.mock_search_engine.search_by_text = AsyncMock(return_value={
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        })

        result = await self.retriever._semantic_retrieval(self.sample_query)
        self.assertEqual(len(result), 0)

        # Test missing fields in Chroma format - should handle gracefully
        self.mock_search_engine.search_by_text = AsyncMock(return_value={
            "ids": [["1"]],
            "documents": [[]],  # Missing document content
            "metadatas": [[]],   # Missing metadata
            "distances": [[]]    # Missing distances
        })

        result = await self.retriever._semantic_retrieval(self.sample_query)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "1")
        self.assertEqual(result[0].content, "")  # Should default to empty
        self.assertEqual(result[0].score, 0.0)   # Should default to 0.0

    async def test_keyword_retrieval(self):
        """Test keyword retrieval strategy."""
        result = await self.retriever._keyword_retrieval(self.sample_query)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].retrieval_method, "keyword")
        self.assertEqual(result[1].retrieval_method, "keyword")

        # Verify search was called with correct parameters
        call_args = self.mock_search_engine.search_by_text.call_args[1]
        self.assertIn("query_text", call_args)
        self.assertIn("top_k", call_args)

    async def test_hybrid_retrieval_success(self):
        """Test hybrid retrieval strategy."""
        self.mock_config.get_strategy_config = Mock(return_value={
            "semantic_weight": 0.7,
            "keyword_weight": 0.3
        })

        result = await self.retriever._hybrid_retrieval(self.sample_query)

        self.assertGreater(len(result), 0)
        # Should have called search twice (semantic and keyword)
        self.assertEqual(self.mock_search_engine.search_by_text.call_count, 2)

    async def test_hybrid_retrieval_missing_config(self):
        """Test hybrid retrieval with missing configuration."""
        self.mock_config.get_strategy_config = Mock(return_value={})

        with self.assertRaises(ValueError) as cm:
            await self.retriever._hybrid_retrieval(self.sample_query)
        self.assertIn("Missing 'semantic_weight'", str(cm.exception))

    async def test_hybrid_retrieval_exception_handling(self):
        """Test hybrid retrieval with search exceptions."""
        self.mock_config.get_strategy_config = Mock(return_value={
            "semantic_weight": 0.7,
            "keyword_weight": 0.3
        })

        # Make one search fail
        self.mock_search_engine.search_by_text = AsyncMock(side_effect=[
            Exception("Semantic search failed"),
            self.sample_search_results
        ])

        with patch('src.retrieval.retriever.get_system_logger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            result = await self.retriever._hybrid_retrieval(self.sample_query)

            # Should still get results from keyword search
            self.assertGreater(len(result), 0)
            mock_logger.warning.assert_called()

    async def test_adaptive_retrieval(self):
        """Test adaptive retrieval strategy."""
        # Test factual query -> keyword
        factual_query = RetrievalQuery(
            original_text="test", processed_text="test",
            query_type=QueryType.FACTUAL, language="en", keywords=["test"]
        )

        self.mock_config.get_adaptive_config = Mock(return_value={"threshold": 0.5})
        result = await self.retriever._adaptive_retrieval(factual_query)

        self.assertGreater(len(result), 0)
        # Should call keyword retrieval for factual queries
        call_args = self.mock_search_engine.search_by_text.call_args[1]
        self.assertIn("query_text", call_args)
        self.assertIn("top_k", call_args)

    async def test_multi_pass_retrieval(self):
        """Test multi-pass retrieval strategy."""
        # Set up search to return different results for different calls in Chroma format
        self.mock_search_engine.search_by_text = AsyncMock(side_effect=[
            {"ids": [["1"]], "documents": [["high quality"]], "metadatas": [[{}]], "distances": [[0.1]]},
            {"ids": [["2"]], "documents": [["additional"]], "metadatas": [[{}]], "distances": [[0.3]]}
        ])

        result = await self.retriever._multi_pass_retrieval(self.sample_query)

        self.assertGreater(len(result), 0)
        # Should call search twice (broad semantic, then keyword)
        self.assertEqual(self.mock_search_engine.search_by_text.call_count, 2)

    async def test_execute_retrieval_strategy_unknown(self):
        """Test execute_retrieval_strategy with unknown strategy."""
        # Create a mock unknown strategy
        unknown_strategy = "unknown_strategy"

        with patch('src.retrieval.retriever.get_system_logger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            # This should call _semantic_retrieval as fallback
            result = await self.retriever._execute_retrieval_strategy(
                self.sample_query, unknown_strategy
            )

            mock_logger.warning.assert_called()
            self.assertGreater(len(result), 0)

    async def test_post_process_results_empty(self):
        """Test _post_process_results with empty documents."""
        result = self.retriever._post_process_results(self.sample_query, [])
        self.assertEqual(result, [])

    async def test_post_process_results_complete(self):
        """Test _post_process_results with full pipeline."""
        docs = [
            RetrievedDocument("1", "test content", 0.9, {}, "semantic"),
            RetrievedDocument("2", "more content", 0.8, {}, "semantic"),
        ]

        result = self.retriever._post_process_results(self.sample_query, docs)

        # Should have added query match analysis
        self.assertIsNotNone(result[0].query_match_info) if result else self.fail("No results returned")
        # Should have called ranker
        self.mock_result_ranker.rank_documents.assert_called_once()

    async def test_retrieve_documents_exception_handling(self):
        """Test retrieve_documents exception handling."""
        self.mock_query_processor.process_query = AsyncMock(
            side_effect=Exception("Processing failed")
        )

        with patch('src.retrieval.retriever.log_error_context') as mock_log_error:
            with self.assertRaises(Exception):
                await self.retriever.retrieve_documents("test query", "en")

            mock_log_error.assert_called()


class TestAsyncDocumentRetriever(unittest.IsolatedAsyncioTestCase):
    """Test DocumentRetriever with actual async execution."""

    async def test_retrieve_documents_timing(self):
        """Test that retrieval timing is recorded."""
        # Set up mocks
        mock_query_processor = Mock(spec=QueryProcessor)
        mock_search_engine = Mock(spec=SearchEngine)
        mock_result_ranker = Mock(spec=ResultRanker)
        mock_config = Mock(spec=RetrievalConfig)

        retriever = DocumentRetriever(
            query_processor=mock_query_processor,
            search_engine=mock_search_engine,
            result_ranker=mock_result_ranker,
            config=mock_config
        )

        # Set up async mocks with small delay
        async def delayed_process_query(query, language):
            await asyncio.sleep(0.01)  # Small delay
            return RetrievalQuery(
                original_text=query, processed_text=query,
                query_type=QueryType.FACTUAL, language=language, keywords=[query]
            )

        async def delayed_search(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay
            return [{"id": "1", "content": "test", "score": 0.8, "metadata": {}}]

        mock_query_processor.process_query = delayed_process_query
        mock_search_engine.search_by_text = delayed_search
        mock_result_ranker.rank_documents = Mock(side_effect=lambda q, docs: docs)

        result = await retriever.retrieve_documents("test query", "en")

        # Should have measured some time
        self.assertGreater(result.retrieval_time, 0.0)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_create_retrieval_query(self):
        """Test create_retrieval_query factory function."""
        query = create_retrieval_query(
            original_text="test",
            processed_text="test processed",
            query_type=QueryType.FACTUAL,
            language="en",
            keywords=["test"],
            max_results=5,
            similarity_threshold=0.8
        )

        self.assertIsInstance(query, RetrievalQuery)
        self.assertEqual(query.original_text, "test")
        self.assertEqual(query.processed_text, "test processed")
        self.assertEqual(query.query_type, QueryType.FACTUAL)
        self.assertEqual(query.language, "en")
        self.assertEqual(query.keywords, ["test"])
        self.assertEqual(query.max_results, 5)
        self.assertEqual(query.similarity_threshold, 0.8)

    def test_create_document_retriever(self):
        """Test create_document_retriever factory function."""
        mock_query_processor = Mock(spec=QueryProcessor)
        mock_search_engine = Mock(spec=SearchEngine)
        mock_result_ranker = Mock(spec=ResultRanker)
        mock_config = Mock(spec=RetrievalConfig)

        retriever = create_document_retriever(
            query_processor=mock_query_processor,
            search_engine=mock_search_engine,
            result_ranker=mock_result_ranker,
            config=mock_config
        )

        self.assertIsInstance(retriever, DocumentRetriever)
        self.assertEqual(retriever.query_processor, mock_query_processor)
        self.assertEqual(retriever.search_engine, mock_search_engine)
        self.assertEqual(retriever.result_ranker, mock_result_ranker)
        self.assertEqual(retriever.config, mock_config)


if __name__ == "__main__":
    unittest.main()