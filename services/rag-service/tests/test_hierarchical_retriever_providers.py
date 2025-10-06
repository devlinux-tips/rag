"""
Tests for hierarchical retriever provider implementations.
Comprehensive testing of mock and production components for dependency injection.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any

from src.retrieval.categorization import CategoryMatch, CategoryType, QueryComplexity
from src.retrieval.hierarchical_retriever import ProcessedQuery, RetrievalConfig, SearchResult
from src.retrieval.hierarchical_retriever_providers import (
    QueryProcessor,
    Categorizer,
    SearchEngineAdapter,
    RerankerAdapter,
    create_hierarchical_retriever,
)
from tests.conftest import (
    MockQueryProcessor,
    MockCategorizer,
    MockSearchEngine,
    MockReranker,
    MockLoggerProvider,
    create_mock_retriever_setup,
    create_test_config,
    create_test_retrieval_config,
)


class TestMockQueryProcessor(unittest.TestCase):
    """Test mock query processor implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = MockQueryProcessor()

    def test_initialization_default(self):
        """Test default initialization."""
        processor = MockQueryProcessor()
        self.assertEqual(processor.mock_responses, {})
        self.assertEqual(processor.call_history, [])

    def test_initialization_with_responses(self):
        """Test initialization with mock responses."""
        mock_response = ProcessedQuery(
            original="test",
            processed="test processed",
            query_type="technical",
            keywords=["test"],
            expanded_terms=["test_expanded"],
            metadata={"test": True}
        )
        responses = {"test": mock_response}
        processor = MockQueryProcessor(responses)
        self.assertEqual(processor.mock_responses, responses)

    def test_set_mock_response(self):
        """Test setting mock response."""
        mock_response = ProcessedQuery(
            original="query",
            processed="processed",
            query_type="general",
            keywords=["query"],
            expanded_terms=["expanded"],
            metadata={}
        )
        self.processor.set_mock_response("test query", mock_response)
        self.assertEqual(self.processor.mock_responses["test query"], mock_response)

    def test_process_query_with_mock_response(self):
        """Test query processing with predefined mock response."""
        mock_response = ProcessedQuery(
            original="test",
            processed="mock processed",
            query_type="cultural",
            keywords=["mock"],
            expanded_terms=["mock_expanded"],
            metadata={"mock": True}
        )
        self.processor.set_mock_response("test", mock_response)

        result = self.processor.process_query("test")
        self.assertEqual(result, mock_response)
        self.assertEqual(len(self.processor.call_history), 1)
        self.assertEqual(self.processor.call_history[0]["query"], "test")

    def test_process_query_default_response(self):
        """Test query processing with default response."""
        result = self.processor.process_query("sample query")

        self.assertEqual(result.original, "sample query")
        self.assertEqual(result.processed, "sample query")
        self.assertEqual(result.query_type, "general")
        self.assertEqual(result.keywords, ["sample", "query"])
        self.assertEqual(result.expanded_terms, ["expanded_sample", "expanded_query"])
        self.assertTrue(result.metadata["mock"])

    def test_process_query_with_context(self):
        """Test query processing with context."""
        context = {"language": "hr", "user": "test"}
        result = self.processor.process_query("test", context)

        self.assertEqual(len(self.processor.call_history), 1)
        self.assertEqual(self.processor.call_history[0]["context"], context)

    def test_process_query_expanded_terms_limit(self):
        """Test that expanded terms are limited to 3."""
        long_query = "one two three four five six"
        result = self.processor.process_query(long_query)

        self.assertEqual(len(result.expanded_terms), 3)
        self.assertEqual(result.expanded_terms, ["expanded_one", "expanded_two", "expanded_three"])


class TestMockCategorizer(unittest.TestCase):
    """Test mock categorizer implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.categorizer = MockCategorizer()

    def test_initialization_default(self):
        """Test default initialization."""
        categorizer = MockCategorizer()
        self.assertEqual(categorizer.mock_responses, {})
        self.assertEqual(categorizer.call_history, [])

    def test_initialization_with_responses(self):
        """Test initialization with mock responses."""
        mock_response = CategoryMatch(
            category=CategoryType.TECHNICAL,
            confidence=0.9,
            matched_patterns=["api"],
            cultural_indicators=[],
            complexity=QueryComplexity.SIMPLE,
            retrieval_strategy="dense"
        )
        responses = {"api query": mock_response}
        categorizer = MockCategorizer(responses)
        self.assertEqual(categorizer.mock_responses, responses)

    def test_set_mock_response(self):
        """Test setting mock response."""
        mock_response = CategoryMatch(
            category=CategoryType.CULTURAL,
            confidence=0.95,
            matched_patterns=["culture"],
            cultural_indicators=["cultural_context"],
            complexity=QueryComplexity.COMPLEX,
            retrieval_strategy="cultural_context"
        )
        self.categorizer.set_mock_response("culture query", mock_response)
        self.assertEqual(self.categorizer.mock_responses["culture query"], mock_response)

    def test_categorize_query_with_mock_response(self):
        """Test categorization with predefined mock response."""
        mock_response = CategoryMatch(
            category=CategoryType.ACADEMIC,
            confidence=0.85,
            matched_patterns=["research"],
            cultural_indicators=[],
            complexity=QueryComplexity.ANALYTICAL,
            retrieval_strategy="hierarchical"
        )
        self.categorizer.set_mock_response("research", mock_response)

        result = self.categorizer.categorize_query("research")
        self.assertEqual(result, mock_response)
        self.assertEqual(len(self.categorizer.call_history), 1)

    def test_categorize_query_technical_default(self):
        """Test default categorization for technical queries."""
        result = self.categorizer.categorize_query("api documentation")

        self.assertEqual(result.category, CategoryType.TECHNICAL)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.retrieval_strategy, "dense")
        self.assertEqual(result.complexity, QueryComplexity.MODERATE)

    def test_categorize_query_cultural_default(self):
        """Test default categorization for cultural queries."""
        result = self.categorizer.categorize_query("hrvatska kultura")

        self.assertEqual(result.category, CategoryType.CULTURAL)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.retrieval_strategy, "cultural_context")

    def test_categorize_query_general_default(self):
        """Test default categorization for general queries."""
        result = self.categorizer.categorize_query("general information")

        self.assertEqual(result.category, CategoryType.GENERAL)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.retrieval_strategy, "hybrid")

    def test_categorize_query_with_context(self):
        """Test categorization with context."""
        context = {"language": "hr", "domain": "technical"}
        result = self.categorizer.categorize_query("test", context)

        self.assertEqual(len(self.categorizer.call_history), 1)
        self.assertEqual(self.categorizer.call_history[0]["scope_context"], context)


class TestMockSearchEngine(unittest.IsolatedAsyncioTestCase):
    """Test mock search engine implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.search_engine = MockSearchEngine()

    def test_initialization_default(self):
        """Test default initialization."""
        engine = MockSearchEngine()
        self.assertEqual(engine.mock_results, [])
        self.assertEqual(engine.call_history, [])
        self.assertEqual(engine.delay_seconds, 0.0)

    def test_initialization_with_results(self):
        """Test initialization with mock results."""
        results = [
            SearchResult(
                content="test content",
                metadata={"source": "test"},
                similarity_score=0.8,
                final_score=0.8,
                boosts={}
            )
        ]
        engine = MockSearchEngine(results)
        self.assertEqual(engine.mock_results, results)

    def test_set_mock_results(self):
        """Test setting mock results."""
        results = [
            SearchResult(
                content="new content",
                metadata={"new": True},
                similarity_score=0.9,
                final_score=0.9,
                boosts={}
            )
        ]
        self.search_engine.set_mock_results(results)
        self.assertEqual(self.search_engine.mock_results, results)

    def test_set_delay(self):
        """Test setting artificial delay."""
        self.search_engine.set_delay(0.5)
        self.assertEqual(self.search_engine.delay_seconds, 0.5)

    async def test_search_basic(self):
        """Test basic search operation."""
        # Create mock results
        self.search_engine.create_mock_results(count=3, base_similarity=0.9)

        results = await self.search_engine.search("test query", k=2)

        self.assertEqual(len(results), 2)  # Limited by k
        self.assertEqual(len(self.search_engine.call_history), 1)
        self.assertEqual(self.search_engine.call_history[0]["query_text"], "test query")
        self.assertEqual(self.search_engine.call_history[0]["k"], 2)

    async def test_search_with_threshold(self):
        """Test search with similarity threshold."""
        # Create results with varying similarities
        results = [
            SearchResult(content="high", metadata={}, similarity_score=0.9, final_score=0.9, boosts={}),
            SearchResult(content="medium", metadata={}, similarity_score=0.5, final_score=0.5, boosts={}),
            SearchResult(content="low", metadata={}, similarity_score=0.2, final_score=0.2, boosts={})
        ]
        self.search_engine.set_mock_results(results)

        filtered_results = await self.search_engine.search("test", k=5, similarity_threshold=0.4)

        self.assertEqual(len(filtered_results), 2)  # Only high and medium pass threshold
        self.assertGreaterEqual(filtered_results[0].similarity_score, 0.4)
        self.assertGreaterEqual(filtered_results[1].similarity_score, 0.4)

    async def test_search_with_delay(self):
        """Test search with artificial delay."""
        self.search_engine.set_delay(0.01)  # Small delay for testing
        self.search_engine.create_mock_results(count=1)

        import time
        start_time = time.time()
        results = await self.search_engine.search("test")
        end_time = time.time()

        self.assertGreaterEqual(end_time - start_time, 0.01)
        self.assertEqual(len(results), 1)

    def test_create_mock_results_default(self):
        """Test creating default mock results."""
        self.search_engine.create_mock_results()

        self.assertEqual(len(self.search_engine.mock_results), 5)
        for i, result in enumerate(self.search_engine.mock_results):
            self.assertIn(f"Mock document {i + 1}", result.content)
            self.assertEqual(result.metadata["source"], f"doc_{i + 1}")
            self.assertTrue(result.metadata["mock"])
            self.assertGreater(result.similarity_score, 0)

    def test_create_mock_results_custom(self):
        """Test creating custom mock results."""
        self.search_engine.create_mock_results(count=3, base_similarity=0.6)

        self.assertEqual(len(self.search_engine.mock_results), 3)
        # Check descending similarity scores
        for i in range(len(self.search_engine.mock_results) - 1):
            self.assertGreaterEqual(
                self.search_engine.mock_results[i].similarity_score,
                self.search_engine.mock_results[i + 1].similarity_score
            )


class TestMockReranker(unittest.IsolatedAsyncioTestCase):
    """Test mock reranker implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.reranker = MockReranker()

    def test_initialization_default(self):
        """Test default initialization."""
        reranker = MockReranker()
        self.assertTrue(reranker.rerank_enabled)
        self.assertEqual(reranker.call_history, [])
        self.assertEqual(reranker.delay_seconds, 0.0)

    def test_initialization_disabled(self):
        """Test initialization with reranking disabled."""
        reranker = MockReranker(rerank_enabled=False)
        self.assertFalse(reranker.rerank_enabled)

    def test_set_delay(self):
        """Test setting artificial delay."""
        self.reranker.set_delay(0.5)
        self.assertEqual(self.reranker.delay_seconds, 0.5)

    async def test_rerank_enabled(self):
        """Test reranking when enabled."""
        documents = [
            {"content": "doc1", "final_score": 0.8},
            {"content": "doc2", "final_score": 0.6},
            {"content": "doc3", "final_score": 0.7}
        ]

        reranked = await self.reranker.rerank("test query", documents, "technical")

        # Should be reversed order
        self.assertEqual(reranked[0]["content"], "doc3")
        self.assertEqual(reranked[1]["content"], "doc2")
        self.assertEqual(reranked[2]["content"], "doc1")

        # Should have updated scores and reranked flag
        for doc in reranked:
            self.assertTrue(doc["reranked"])
            self.assertGreater(doc["final_score"], 0.1)

        # Check call history
        self.assertEqual(len(self.reranker.call_history), 1)
        self.assertEqual(self.reranker.call_history[0]["query"], "test query")
        self.assertEqual(self.reranker.call_history[0]["document_count"], 3)
        self.assertEqual(self.reranker.call_history[0]["category"], "technical")

    async def test_rerank_disabled(self):
        """Test reranking when disabled."""
        reranker = MockReranker(rerank_enabled=False)
        documents = [
            {"content": "doc1", "final_score": 0.8},
            {"content": "doc2", "final_score": 0.6}
        ]

        result = await reranker.rerank("test", documents)

        # Should return unchanged documents
        self.assertEqual(result, documents)
        self.assertEqual(len(reranker.call_history), 1)

    async def test_rerank_with_delay(self):
        """Test reranking with artificial delay."""
        self.reranker.set_delay(0.01)
        documents = [{"content": "test", "final_score": 0.5}]

        import time
        start_time = time.time()
        result = await self.reranker.rerank("test", documents)
        end_time = time.time()

        self.assertGreaterEqual(end_time - start_time, 0.01)
        self.assertEqual(len(result), 1)


class TestMockLoggerProvider(unittest.TestCase):
    """Test mock logger provider implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = MockLoggerProvider()

    def test_initialization(self):
        """Test logger initialization."""
        logger = MockLoggerProvider()
        expected_levels = {"info", "debug", "warning", "error"}
        self.assertEqual(set(logger.messages.keys()), expected_levels)
        for level in expected_levels:
            self.assertEqual(logger.messages[level], [])

    def test_info_logging(self):
        """Test info message logging."""
        self.logger.info("Test info message")
        self.assertEqual(self.logger.messages["info"], ["Test info message"])
        self.assertEqual(len(self.logger.messages["debug"]), 0)
        self.assertEqual(len(self.logger.messages["error"]), 0)

    def test_debug_logging(self):
        """Test debug message logging."""
        self.logger.debug("Test debug message")
        self.assertEqual(self.logger.messages["debug"], ["Test debug message"])
        self.assertEqual(len(self.logger.messages["info"]), 0)

    def test_error_logging(self):
        """Test error message logging."""
        self.logger.error("Test error message")
        self.assertEqual(self.logger.messages["error"], ["Test error message"])
        self.assertEqual(len(self.logger.messages["info"]), 0)

    def test_multiple_messages(self):
        """Test logging multiple messages."""
        self.logger.info("Info 1")
        self.logger.info("Info 2")
        self.logger.debug("Debug 1")
        self.logger.error("Error 1")

        self.assertEqual(len(self.logger.messages["info"]), 2)
        self.assertEqual(len(self.logger.messages["debug"]), 1)
        self.assertEqual(len(self.logger.messages["error"]), 1)

    def test_clear_messages(self):
        """Test clearing all messages."""
        self.logger.info("Test message")
        self.logger.debug("Test debug")
        self.logger.clear_messages()

        for level in self.logger.messages:
            self.assertEqual(len(self.logger.messages[level]), 0)

    def test_get_messages_by_level(self):
        """Test getting messages by specific level."""
        self.logger.info("Info message")
        self.logger.debug("Debug message")

        info_messages = self.logger.get_messages("info")
        self.assertEqual(info_messages, ["Info message"])

        debug_messages = self.logger.get_messages("debug")
        self.assertEqual(debug_messages, ["Debug message"])

    def test_get_all_messages(self):
        """Test getting all messages."""
        self.logger.info("Info")
        self.logger.error("Error")

        all_messages = self.logger.get_messages()
        self.assertIsInstance(all_messages, dict)
        self.assertEqual(len(all_messages["info"]), 1)
        self.assertEqual(len(all_messages["error"]), 1)

    def test_get_messages_nonexistent_level(self):
        """Test getting messages for non-existent level."""
        result = self.logger.get_messages("warning")
        self.assertEqual(result, [])


class TestQueryProcessor(unittest.TestCase):
    """Test production query processor wrapper."""

    @patch('src.retrieval.query_processor.create_query_processor')
    def test_initialization_with_factory(self, mock_create):
        """Test initialization using factory function."""
        mock_processor = Mock()
        mock_create.return_value = mock_processor

        processor = QueryProcessor("hr")

        mock_create.assert_called_once_with(main_config=unittest.mock.ANY, language="hr", config_provider=unittest.mock.ANY)
        self.assertEqual(processor._processor, mock_processor)

    @patch('src.retrieval.query_processor.create_query_processor')
    @patch('src.retrieval.query_processor_providers.create_default_config')
    @patch('src.retrieval.query_processor.MultilingualQueryProcessor')
    def test_initialization_fallback_to_default_config(self, mock_processor_class, mock_create_config, mock_create):
        """Test fallback to default config when factory fails."""
        mock_create.side_effect = ImportError("Factory not available")
        mock_config = Mock()
        mock_create_config.return_value = mock_config
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        processor = QueryProcessor("en")

        mock_create_config.assert_called_once_with(language="en")
        mock_processor_class.assert_called_once_with(config=mock_config, filter_config=unittest.mock.ANY)
        self.assertEqual(processor._processor, mock_processor)

    @patch('src.retrieval.query_processor.create_query_processor')
    @patch('src.retrieval.query_processor_providers.create_default_config')
    @patch('src.retrieval.query_processor.MultilingualQueryProcessor')
    @patch('src.utils.config_models.QueryProcessingConfig')
    def test_initialization_minimal_fallback(self, mock_config_class, mock_processor_class, mock_create_config, mock_create):
        """Test minimal fallback when both factory and default config fail."""
        mock_create.side_effect = ImportError("Factory not available")
        mock_create_config.side_effect = ImportError("Default config not available")
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        processor = QueryProcessor("hr")

        # Verify minimal config creation
        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args[1]
        self.assertEqual(call_kwargs["language"], "hr")
        self.assertFalse(call_kwargs["expand_synonyms"])
        self.assertTrue(call_kwargs["normalize_case"])
        self.assertFalse(call_kwargs["enable_morphological_analysis"])

    def test_process_query_success(self):
        """Test successful query processing."""
        mock_processor = Mock()
        mock_result = Mock()
        mock_result.processed = "processed query"
        mock_result.query_type = "technical"
        mock_result.keywords = ["test", "query"]
        mock_result.expanded_terms = ["expanded_test"]
        mock_result.metadata = {"source": "test"}
        mock_processor.process_query.return_value = mock_result

        with patch('src.retrieval.query_processor.create_query_processor') as mock_create:
            mock_create.return_value = mock_processor
            processor = QueryProcessor()

            result = processor.process_query("test query", {"context": "test"})

            self.assertIsInstance(result, ProcessedQuery)
            self.assertEqual(result.original, "test query")
            self.assertEqual(result.processed, "processed query")
            self.assertEqual(result.query_type, "technical")
            self.assertEqual(result.keywords, ["test", "query"])
            self.assertEqual(result.expanded_terms, ["expanded_test"])
            self.assertEqual(result.metadata, {"source": "test"})

    def test_process_query_processor_none(self):
        """Test query processing when processor is None."""
        # Create processor with working initialization but force processor to None
        mock_processor = Mock()
        with patch('src.retrieval.query_processor.create_query_processor') as mock_create:
            mock_create.return_value = mock_processor
            processor = QueryProcessor()
            # Force processor to None to test error handling
            processor._processor = None

            with self.assertRaises(Exception) as cm:
                processor.process_query("test")

            self.assertIn("Query processor not available", str(cm.exception))

    def test_process_query_missing_attributes(self):
        """Test query processing with missing result attributes."""
        mock_processor = Mock()
        mock_result = Mock(spec=[])  # Empty spec, no attributes
        mock_processor.process_query.return_value = mock_result

        with patch('src.retrieval.query_processor.create_query_processor') as mock_create:
            mock_create.return_value = mock_processor
            processor = QueryProcessor()

            result = processor.process_query("test query")

            # Should use defaults when attributes missing
            self.assertEqual(result.original, "test query")
            self.assertEqual(result.processed, "test query")
            self.assertEqual(result.query_type, "general")
            self.assertEqual(result.keywords, ["test", "query"])
            self.assertEqual(result.expanded_terms, [])
            self.assertEqual(result.metadata, {})


class TestCategorizer(unittest.TestCase):
    """Test production categorizer wrapper."""

    @patch('src.retrieval.categorization_providers.create_config_provider')
    @patch('src.retrieval.categorization.QueryCategorizer')
    def test_initialization(self, mock_categorizer_class, mock_create_provider):
        """Test production categorizer initialization."""
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider
        mock_categorizer = Mock()
        mock_categorizer_class.return_value = mock_categorizer

        categorizer = Categorizer("hr")

        mock_create_provider.assert_called_once()
        mock_categorizer_class.assert_called_once_with("hr", mock_provider)
        self.assertEqual(categorizer._categorizer, mock_categorizer)

    @patch('src.retrieval.categorization_providers.create_config_provider')
    @patch('src.retrieval.categorization.QueryCategorizer')
    def test_categorize_query(self, mock_categorizer_class, mock_create_provider):
        """Test query categorization."""
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider
        mock_categorizer = Mock()
        mock_categorizer_class.return_value = mock_categorizer

        expected_result = CategoryMatch(
            category=CategoryType.TECHNICAL,
            confidence=0.9,
            matched_patterns=["api"],
            cultural_indicators=[],
            complexity=QueryComplexity.SIMPLE,
            retrieval_strategy="dense"
        )
        mock_categorizer.categorize_query.return_value = expected_result

        categorizer = Categorizer("en")
        result = categorizer.categorize_query("api documentation", {"context": "test"})

        mock_categorizer.categorize_query.assert_called_once_with("api documentation", {"context": "test"})
        self.assertEqual(result, expected_result)


class TestSearchEngineAdapter(unittest.IsolatedAsyncioTestCase):
    """Test production search engine adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_search_engine = AsyncMock()
        self.adapter = SearchEngineAdapter(self.mock_search_engine)

    async def test_search_with_search_result_objects(self):
        """Test search adaptation with SearchResult-like objects."""
        # Mock ChromaDB-style response format
        mock_raw_results = {
            "documents": [["test content"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.15]]  # Distance, will be converted to similarity
        }
        self.mock_search_engine.search_by_text.return_value = mock_raw_results

        results = await self.adapter.search("test query", k=3, similarity_threshold=0.5)

        self.mock_search_engine.search_by_text.assert_called_once_with(
            query_text="test query", top_k=3, filters=None, include_metadata=True
        )
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].content, "test content")
        self.assertEqual(results[0].metadata, {"source": "test"})
        self.assertEqual(results[0].similarity_score, 0.85)
        self.assertEqual(results[0].final_score, 0.85)

    async def test_search_with_dict_results(self):
        """Test search adaptation with dictionary results."""
        # Mock ChromaDB-style response format
        mock_raw_results = {
            "documents": [["dict content"]],
            "metadatas": [[{"type": "dict"}]],
            "distances": [[0.25]]  # Distance, will be converted to similarity 0.75
        }
        self.mock_search_engine.search_by_text.return_value = mock_raw_results

        results = await self.adapter.search("test")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "dict content")
        self.assertEqual(results[0].metadata, {"type": "dict"})
        self.assertEqual(results[0].similarity_score, 0.75)

    async def test_search_with_string_results(self):
        """Test search adaptation with string results."""
        # Mock ChromaDB-style response format
        mock_raw_results = {
            "documents": [["string result"]],
            "metadatas": [[{}]],
            "distances": [[0.5]]  # Distance, will be converted to similarity 0.5
        }
        self.mock_search_engine.search_by_text.return_value = mock_raw_results

        results = await self.adapter.search("test")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "string result")
        self.assertEqual(results[0].metadata, {})
        self.assertEqual(results[0].similarity_score, 0.5)

    async def test_search_missing_attributes(self):
        """Test search adaptation with objects missing some attributes."""
        # Mock ChromaDB-style response with minimal metadata
        mock_raw_results = {
            "documents": [["minimal content"]],
            "metadatas": [[{}]],  # Empty metadata
            "distances": [[0.5]]  # Distance, will be converted to similarity 0.5
        }
        self.mock_search_engine.search_by_text.return_value = mock_raw_results

        results = await self.adapter.search("test")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "minimal content")
        self.assertEqual(results[0].metadata, {})
        self.assertEqual(results[0].similarity_score, 0.5)

    async def test_search_empty_results(self):
        """Test search adaptation with empty results."""
        # Mock ChromaDB-style response with empty results
        mock_raw_results = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        self.mock_search_engine.search_by_text.return_value = mock_raw_results

        results = await self.adapter.search("test")

        self.assertEqual(len(results), 0)


class TestRerankerAdapter(unittest.IsolatedAsyncioTestCase):
    """Test production reranker adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_reranker = AsyncMock()
        self.adapter = RerankerAdapter(self.mock_reranker, "hr")

    async def test_rerank_delegation(self):
        """Test that rerank calls are properly delegated."""
        expected_result = [{"content": "reranked", "score": 0.9}]
        self.mock_reranker.rerank.return_value = expected_result

        documents = [{"content": "original", "score": 0.7}]
        result = await self.adapter.rerank("test query", documents, "technical")

        self.mock_reranker.rerank.assert_called_once_with(
            query="test query",
            documents=documents,
            category="technical"
        )
        self.assertEqual(result, expected_result)

    async def test_rerank_with_none_category(self):
        """Test reranking with None category."""
        expected_result = [{"content": "test"}]
        self.mock_reranker.rerank.return_value = expected_result

        result = await self.adapter.rerank("query", [], None)

        self.mock_reranker.rerank.assert_called_once_with(
            query="query",
            documents=[],
            category=None
        )
        self.assertEqual(result, expected_result)

    def test_initialization_with_language(self):
        """Test adapter initialization with language."""
        adapter = RerankerAdapter(self.mock_reranker, "en")
        self.assertEqual(adapter._reranker, self.mock_reranker)
        self.assertEqual(adapter.language, "en")


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating provider setups."""

    def test_create_mock_retriever_setup_default(self):
        """Test creating default mock setup."""
        result = create_mock_retriever_setup()

        self.assertEqual(len(result), 6)
        query_processor, categorizer, search_engine, reranker, logger, config = result

        self.assertIsInstance(query_processor, MockQueryProcessor)
        self.assertIsInstance(categorizer, MockCategorizer)
        self.assertIsInstance(search_engine, MockSearchEngine)
        self.assertIsInstance(reranker, MockReranker)
        self.assertIsInstance(logger, MockLoggerProvider)
        self.assertIsInstance(config, RetrievalConfig)

        # Should have default search results
        self.assertEqual(len(search_engine.mock_results), 5)

    def test_create_mock_retriever_setup_with_parameters(self):
        """Test creating mock setup with custom parameters."""
        query_responses = {"test": ProcessedQuery("test", "processed", "general", [], [], {})}
        category_responses = {"test": CategoryMatch(CategoryType.TECHNICAL, 0.9, [], [], QueryComplexity.SIMPLE, "dense")}
        search_results = [SearchResult("test", {}, 0.8, 0.8, {})]

        result = create_mock_retriever_setup(
            query_responses=query_responses,
            category_responses=category_responses,
            search_results=search_results,
            enable_reranking=False,
            search_delay=0.1
        )

        query_processor, categorizer, search_engine, reranker, logger, config = result

        self.assertEqual(query_processor.mock_responses, query_responses)
        self.assertEqual(categorizer.mock_responses, category_responses)
        self.assertEqual(search_engine.mock_results, search_results)
        self.assertFalse(reranker.rerank_enabled)
        self.assertEqual(search_engine.delay_seconds, 0.1)
        self.assertEqual(reranker.delay_seconds, 0.1)

    def test_create_mock_retriever_setup_config_structure(self):
        """Test that created mock setup has proper config structure."""
        _, _, _, _, _, config = create_mock_retriever_setup()

        self.assertEqual(config.default_max_results, 5)
        self.assertIn("semantic_focused", config.similarity_thresholds)
        self.assertIn("keyword", config.boost_weights)
        self.assertTrue(config.performance_tracking)

    @patch('src.retrieval.hierarchical_retriever.HierarchicalRetriever')
    @patch('src.retrieval.hierarchical_retriever_providers.QueryProcessor')
    @patch('src.retrieval.hierarchical_retriever_providers.Categorizer')
    @patch('src.retrieval.hierarchical_retriever_providers.SearchEngineAdapter')
    @patch('src.retrieval.hierarchical_retriever_providers.RerankerAdapter')
    def test_create_hierarchical_retriever_with_reranker(self, mock_reranker_class, mock_search_class, mock_cat_class, mock_query_class, mock_hierarchical_class):
        """Test creating production setup with reranker."""
        mock_search_engine = Mock()
        mock_reranker = Mock()
        mock_retriever = Mock()
        mock_hierarchical_class.return_value = mock_retriever

        mock_query_processor = Mock()
        mock_categorizer = Mock()
        mock_search_adapter = Mock()
        mock_reranker_adapter = Mock()

        mock_query_class.return_value = mock_query_processor
        mock_cat_class.return_value = mock_categorizer
        mock_search_class.return_value = mock_search_adapter
        mock_reranker_class.return_value = mock_reranker_adapter

        result = create_hierarchical_retriever(mock_search_engine, mock_reranker, "hr")

        # Should return HierarchicalRetriever instance
        self.assertEqual(result, mock_retriever)

        # Verify component creation
        mock_query_class.assert_called_once_with(language="hr")
        mock_cat_class.assert_called_once_with(language="hr")
        mock_search_class.assert_called_once_with(mock_search_engine)
        mock_reranker_class.assert_called_once_with(mock_reranker, language="hr")

        # Verify HierarchicalRetriever creation
        mock_hierarchical_class.assert_called_once()
        call_kwargs = mock_hierarchical_class.call_args[1]
        self.assertEqual(call_kwargs['query_processor'], mock_query_processor)
        self.assertEqual(call_kwargs['categorizer'], mock_categorizer)
        self.assertEqual(call_kwargs['search_engine'], mock_search_adapter)
        self.assertEqual(call_kwargs['reranker'], mock_reranker_adapter)

    @patch('src.retrieval.hierarchical_retriever.HierarchicalRetriever')
    @patch('src.retrieval.hierarchical_retriever_providers.QueryProcessor')
    @patch('src.retrieval.hierarchical_retriever_providers.Categorizer')
    @patch('src.retrieval.hierarchical_retriever_providers.SearchEngineAdapter')
    def test_create_hierarchical_retriever_without_reranker(self, mock_search_class, mock_cat_class, mock_query_class, mock_hierarchical_class):
        """Test creating production setup without reranker."""
        mock_search_engine = Mock()
        mock_retriever = Mock()
        mock_hierarchical_class.return_value = mock_retriever

        result = create_hierarchical_retriever(mock_search_engine, None, "en")

        # Should return HierarchicalRetriever instance
        self.assertEqual(result, mock_retriever)

        # Verify HierarchicalRetriever creation with None reranker and performance_tracking=False
        mock_hierarchical_class.assert_called_once()
        call_kwargs = mock_hierarchical_class.call_args[1]
        self.assertIsNone(call_kwargs['reranker'])
        # Config should have performance_tracking=False
        config = call_kwargs['config']
        self.assertFalse(config.performance_tracking)

    def test_create_test_config_default(self):
        """Test creating default test configuration."""
        config = create_test_retrieval_config()

        self.assertIsInstance(config, RetrievalConfig)
        self.assertEqual(config.default_max_results, 5)
        self.assertTrue(config.performance_tracking)
        self.assertIn("semantic_focused", config.similarity_thresholds)
        self.assertEqual(config.similarity_thresholds["default"], 0.3)

    def test_create_test_config_custom(self):
        """Test creating custom test configuration."""
        config = create_test_retrieval_config(max_results=10, performance_tracking=False)

        self.assertEqual(config.default_max_results, 10)
        self.assertFalse(config.performance_tracking)

    def test_config_consistency_across_factories(self):
        """Test that all factory functions create consistent configurations."""
        mock_config = create_mock_retriever_setup()[5]
        test_config = create_test_retrieval_config()

        # Should have same structure
        self.assertEqual(mock_config.similarity_thresholds, test_config.similarity_thresholds)
        self.assertEqual(mock_config.boost_weights, test_config.boost_weights)


if __name__ == "__main__":
    unittest.main()