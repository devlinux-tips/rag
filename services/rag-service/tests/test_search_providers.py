"""
Tests for vectordb.search_providers module.
Validates provider implementations for search system dependencies.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any

import numpy as np
import pytest

from src.vectordb.search_providers import (
    # Mock Providers
    MockEmbeddingProvider,
    MockVectorSearchProvider,
    MockConfigProvider,
    # Production Providers
    SentenceTransformerEmbeddingProvider,
    ChromaDBSearchProvider,
    DefaultConfigProvider,
    # Factory Functions
    create_mock_embedding_provider,
    create_mock_search_provider,
    create_mock_config_provider,
    create_embedding_provider,
    create_vector_search_provider,
    create_config_provider,
)
from src.vectordb.search import ConfigProvider, EmbeddingProvider, VectorSearchProvider


class TestMockEmbeddingProvider(unittest.TestCase):
    """Test MockEmbeddingProvider functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockEmbeddingProvider(dimension=384, deterministic=True)

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        provider = MockEmbeddingProvider()
        self.assertEqual(provider.dimension, 384)
        self.assertTrue(provider.deterministic)
        self.assertEqual(provider._embedding_cache, {})

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        provider = MockEmbeddingProvider(dimension=512, deterministic=False)
        self.assertEqual(provider.dimension, 512)
        self.assertFalse(provider.deterministic)

    def test_encode_text_deterministic_first_call(self):
        """Test encoding text deterministically on first call."""
        text = "test text"

        async def run_test():
            embedding = await self.provider.encode_text(text)

            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (384,))
            self.assertEqual(embedding.dtype, np.float32)

            # Check normalization (unit vector)
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=5)

            # Should be cached
            self.assertIn(text, self.provider._embedding_cache)

        asyncio.run(run_test())

    def test_encode_text_deterministic_cached(self):
        """Test encoding same text returns cached result."""
        text = "test text"

        async def run_test():
            embedding1 = await self.provider.encode_text(text)
            embedding2 = await self.provider.encode_text(text)

            np.testing.assert_array_equal(embedding1, embedding2)

        asyncio.run(run_test())

    def test_encode_text_deterministic_different_texts(self):
        """Test different texts produce different embeddings."""
        async def run_test():
            embedding1 = await self.provider.encode_text("text1")
            embedding2 = await self.provider.encode_text("text2")

            # Should be different but same shape/type
            self.assertFalse(np.array_equal(embedding1, embedding2))
            self.assertEqual(embedding1.shape, embedding2.shape)
            self.assertEqual(embedding1.dtype, embedding2.dtype)

        asyncio.run(run_test())

    def test_encode_text_non_deterministic(self):
        """Test non-deterministic encoding produces different results."""
        provider = MockEmbeddingProvider(deterministic=False)

        async def run_test():
            embedding1 = await provider.encode_text("test")
            embedding2 = await provider.encode_text("test")

            # Should be different (very unlikely to be equal)
            self.assertFalse(np.array_equal(embedding1, embedding2))

        asyncio.run(run_test())

    def test_encode_text_custom_dimension(self):
        """Test encoding with custom dimension."""
        provider = MockEmbeddingProvider(dimension=768)

        async def run_test():
            embedding = await provider.encode_text("test")

            self.assertEqual(embedding.shape, (768,))
            self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)

        asyncio.run(run_test())

    def test_encode_text_empty_string(self):
        """Test encoding empty string."""
        async def run_test():
            embedding = await self.provider.encode_text("")

            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (384,))
            self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)

        asyncio.run(run_test())


class TestMockVectorSearchProvider(unittest.TestCase):
    """Test MockVectorSearchProvider functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockVectorSearchProvider()
        self.sample_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.sample_embedding = self.sample_embedding / np.linalg.norm(self.sample_embedding)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.provider.documents, {})

    def test_add_document_basic(self):
        """Test adding a document."""
        self.provider.add_document(
            "doc1", "test content", self.sample_embedding, {"type": "test"}
        )

        self.assertIn("doc1", self.provider.documents)
        doc = self.provider.documents["doc1"]
        self.assertEqual(doc["content"], "test content")
        np.testing.assert_array_equal(doc["embedding"], self.sample_embedding)
        self.assertEqual(doc["metadata"], {"type": "test"})

    def test_add_document_no_metadata(self):
        """Test adding document without metadata."""
        self.provider.add_document("doc1", "content", self.sample_embedding)

        doc = self.provider.documents["doc1"]
        self.assertEqual(doc["metadata"], {})

    def test_search_by_embedding_empty_collection(self):
        """Test search with no documents."""
        query_embedding = np.array([0.5, 0.5, 0.0], dtype=np.float32)

        async def run_test():
            result = await self.provider.search_by_embedding(query_embedding, top_k=5)

            expected = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            self.assertEqual(result, expected)

        asyncio.run(run_test())

    def test_search_by_embedding_single_document(self):
        """Test search with single document."""
        self.provider.add_document("doc1", "test content", self.sample_embedding)

        async def run_test():
            # Query with same embedding (should have distance ~0)
            result = await self.provider.search_by_embedding(self.sample_embedding, top_k=1)

            self.assertEqual(result["ids"], [["doc1"]])
            self.assertEqual(result["documents"], [["test content"]])
            self.assertEqual(result["metadatas"], [[{}]])

            # Distance should be very small (cosine distance = 1 - similarity)
            distance = result["distances"][0][0]
            self.assertLess(distance, 0.001)

        asyncio.run(run_test())

    def test_search_by_embedding_multiple_documents(self):
        """Test search with multiple documents."""
        # Add documents with different embeddings
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        emb3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self.provider.add_document("doc1", "content1", emb1, {"type": "A"})
        self.provider.add_document("doc2", "content2", emb2, {"type": "B"})
        self.provider.add_document("doc3", "content3", emb3, {"type": "C"})

        async def run_test():
            # Query with embedding similar to emb1
            query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
            query = query / np.linalg.norm(query)

            result = await self.provider.search_by_embedding(query, top_k=3)

            # Should return all 3, sorted by distance
            self.assertEqual(len(result["ids"][0]), 3)
            self.assertEqual(result["ids"][0][0], "doc1")  # Closest to query

        asyncio.run(run_test())

    def test_search_by_embedding_with_filters(self):
        """Test search with metadata filters."""
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.provider.add_document("doc1", "content1", emb1, {"type": "A"})
        self.provider.add_document("doc2", "content2", emb2, {"type": "B"})

        async def run_test():
            query = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            result = await self.provider.search_by_embedding(
                query, top_k=5, filters={"type": "A"}
            )

            # Should only return doc1
            self.assertEqual(result["ids"], [["doc1"]])
            self.assertEqual(result["documents"], [["content1"]])

        asyncio.run(run_test())

    def test_search_by_embedding_top_k_limit(self):
        """Test top_k parameter limits results."""
        for i in range(5):
            emb = np.random.normal(0, 1, 3).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            self.provider.add_document(f"doc{i}", f"content{i}", emb)

        async def run_test():
            query = np.random.normal(0, 1, 3).astype(np.float32)
            query = query / np.linalg.norm(query)

            result = await self.provider.search_by_embedding(query, top_k=2)

            self.assertEqual(len(result["ids"][0]), 2)
            self.assertEqual(len(result["documents"][0]), 2)

        asyncio.run(run_test())

    def test_search_by_text_empty_collection(self):
        """Test text search with no documents."""
        async def run_test():
            result = await self.provider.search_by_text("query", top_k=5)

            expected = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            self.assertEqual(result, expected)

        asyncio.run(run_test())

    def test_search_by_text_keyword_matching(self):
        """Test text search with keyword matching."""
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.provider.add_document("doc1", "python programming language", emb)
        self.provider.add_document("doc2", "java development framework", emb)
        self.provider.add_document("doc3", "python data science", emb)

        async def run_test():
            result = await self.provider.search_by_text("python", top_k=5)

            # Should return docs containing "python" with better scores
            ids = result["ids"][0]
            self.assertIn("doc1", ids)
            self.assertIn("doc3", ids)

        asyncio.run(run_test())

    def test_search_by_text_phrase_boost(self):
        """Test text search phrase matching boost."""
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.provider.add_document("doc1", "machine learning algorithms", emb)
        self.provider.add_document("doc2", "learning machine tools", emb)

        async def run_test():
            result = await self.provider.search_by_text("machine learning", top_k=2)

            # doc1 should rank higher due to exact phrase match
            self.assertEqual(result["ids"][0][0], "doc1")

        asyncio.run(run_test())

    def test_search_by_text_with_filters(self):
        """Test text search with filters."""
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.provider.add_document("doc1", "python code", emb, {"lang": "python"})
        self.provider.add_document("doc2", "python tutorial", emb, {"lang": "java"})

        async def run_test():
            result = await self.provider.search_by_text(
                "python", top_k=5, filters={"lang": "python"}
            )

            self.assertEqual(result["ids"], [["doc1"]])

        asyncio.run(run_test())

    def test_get_document_exists(self):
        """Test getting existing document."""
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.provider.add_document("doc1", "content", emb, {"key": "value"})

        async def run_test():
            result = await self.provider.get_document("doc1")

            expected = {
                "id": "doc1",
                "content": "content",
                "metadata": {"key": "value"}
            }
            self.assertEqual(result, expected)

        asyncio.run(run_test())

    def test_get_document_not_exists(self):
        """Test getting non-existent document."""
        async def run_test():
            result = await self.provider.get_document("nonexistent")
            self.assertIsNone(result)

        asyncio.run(run_test())

    def test_matches_filters_success(self):
        """Test filter matching success."""
        metadata = {"type": "doc", "category": "test"}
        filters = {"type": "doc"}

        result = self.provider._matches_filters(metadata, filters)
        self.assertTrue(result)

    def test_matches_filters_failure_missing_key(self):
        """Test filter matching failure - missing key."""
        metadata = {"type": "doc"}
        filters = {"category": "test"}

        result = self.provider._matches_filters(metadata, filters)
        self.assertFalse(result)

    def test_matches_filters_failure_wrong_value(self):
        """Test filter matching failure - wrong value."""
        metadata = {"type": "doc"}
        filters = {"type": "image"}

        result = self.provider._matches_filters(metadata, filters)
        self.assertFalse(result)

    def test_matches_filters_multiple_criteria(self):
        """Test filter matching with multiple criteria."""
        metadata = {"type": "doc", "category": "test", "lang": "en"}
        filters = {"type": "doc", "category": "test"}

        result = self.provider._matches_filters(metadata, filters)
        self.assertTrue(result)


class TestMockConfigProvider(unittest.TestCase):
    """Test MockConfigProvider functionality."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        provider = MockConfigProvider()

        config = provider.config
        self.assertIn("search", config)
        self.assertIn("scoring", config)
        self.assertEqual(config["search"]["default_method"], "semantic")

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        custom_config = {
            "search": {"default_method": "keyword", "top_k": 10},
            "scoring": {"weights": {"semantic": 0.5, "keyword": 0.5}}
        }
        provider = MockConfigProvider(custom_config)

        self.assertEqual(provider.config, custom_config)

    def test_get_search_config(self):
        """Test getting search configuration."""
        provider = MockConfigProvider()
        search_config = provider.get_search_config()

        self.assertIn("default_method", search_config)
        self.assertIn("top_k", search_config)
        self.assertEqual(search_config["default_method"], "semantic")

    def test_get_scoring_weights(self):
        """Test getting scoring weights."""
        provider = MockConfigProvider()
        weights = provider.get_scoring_weights()

        self.assertIn("semantic", weights)
        self.assertIn("keyword", weights)
        self.assertEqual(weights["semantic"], 0.7)
        self.assertEqual(weights["keyword"], 0.3)

    def test_default_config_structure(self):
        """Test default config structure is complete."""
        provider = MockConfigProvider()
        config = provider._default_config()

        # Verify search section
        search = config["search"]
        required_search_keys = [
            "default_method", "top_k", "similarity_threshold",
            "max_context_length", "rerank", "include_metadata", "include_distances"
        ]
        for key in required_search_keys:
            self.assertIn(key, search)

        # Verify scoring section
        scoring = config["scoring"]
        self.assertIn("weights", scoring)
        self.assertIn("boost_factors", scoring)

        weights = scoring["weights"]
        self.assertIn("semantic", weights)
        self.assertIn("keyword", weights)


class TestSentenceTransformerEmbeddingProvider(unittest.TestCase):
    """Test SentenceTransformerEmbeddingProvider functionality."""

    @patch('sentence_transformers.SentenceTransformer')
    def test_init_default_params(self, mock_sentence_transformer):
        """Test initialization with default parameters."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_sentence_transformer.return_value = mock_model

        provider = SentenceTransformerEmbeddingProvider()

        mock_sentence_transformer.assert_called_once_with("BAAI/bge-m3", device="cpu")
        self.assertEqual(provider.model, mock_model)
        self.assertEqual(provider.model_name, "BAAI/bge-m3")
        self.assertEqual(provider.device, "cpu")

    @patch('sentence_transformers.SentenceTransformer')
    def test_init_custom_params(self, mock_sentence_transformer):
        """Test initialization with custom parameters."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        provider = SentenceTransformerEmbeddingProvider(
            model_name="custom/model", device="cuda"
        )

        mock_sentence_transformer.assert_called_once_with("custom/model", device="cuda")
        self.assertEqual(provider.model_name, "custom/model")
        self.assertEqual(provider.device, "cuda")

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_text_success(self, mock_sentence_transformer):
        """Test successful text encoding."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model

        provider = SentenceTransformerEmbeddingProvider()

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(
                    return_value=np.array([0.1, 0.2, 0.3], dtype=np.float32)
                )

                result = await provider.encode_text("test text")

                mock_loop.run_in_executor.assert_called_once()
                call_args = mock_loop.run_in_executor.call_args
                self.assertIsNone(call_args[0][0])  # executor=None

                np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32))

        asyncio.run(run_test())

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_text_2d_to_1d_conversion(self, mock_sentence_transformer):
        """Test 2D array conversion to 1D."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model

        provider = SentenceTransformerEmbeddingProvider()

        # Mock 2D array with single row
        mock_2d_embedding = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_2d_embedding)

                result = await provider.encode_text("test")

                self.assertEqual(result.shape, (3,))
                np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3], dtype=np.float32))

        asyncio.run(run_test())

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_text_error_handling(self, mock_sentence_transformer):
        """Test error handling during encoding."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model

        provider = SentenceTransformerEmbeddingProvider()

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(side_effect=RuntimeError("Model error"))

                with self.assertRaises(RuntimeError):
                    await provider.encode_text("test")

        asyncio.run(run_test())


class TestChromaDBSearchProvider(unittest.TestCase):
    """Test ChromaDBSearchProvider functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_collection = Mock()
        self.provider = ChromaDBSearchProvider(self.mock_collection)

    def test_search_by_embedding_success(self):
        """Test successful embedding search."""
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [[{"type": "A"}, {"type": "B"}]],
            "distances": [[0.1, 0.2]]
        }

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_results)

                result = await self.provider.search_by_embedding(
                    query_embedding, top_k=2, include_metadata=True
                )

                self.assertEqual(result, mock_results)
                mock_loop.run_in_executor.assert_called_once()

        asyncio.run(run_test())

    def test_search_by_embedding_with_filters(self):
        """Test embedding search with filters."""
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        filters = {"type": "document"}

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value={"ids": [[]]})

                await self.provider.search_by_embedding(
                    query_embedding, top_k=5, filters=filters
                )

                # Verify the lambda function was called (can't easily inspect its args)
                mock_loop.run_in_executor.assert_called_once()

        asyncio.run(run_test())

    def test_search_by_embedding_without_metadata(self):
        """Test embedding search without metadata."""
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value={"ids": [[]]})

                await self.provider.search_by_embedding(
                    query_embedding, top_k=5, include_metadata=False
                )

                mock_loop.run_in_executor.assert_called_once()

        asyncio.run(run_test())

    def test_search_by_embedding_error(self):
        """Test embedding search error handling."""
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(side_effect=Exception("ChromaDB error"))

                with self.assertRaises(Exception):
                    await self.provider.search_by_embedding(query_embedding, top_k=5)

        asyncio.run(run_test())

    def test_search_by_text_success(self):
        """Test successful text search."""
        mock_results = {
            "ids": [["doc1"]],
            "documents": [["content1"]],
            "metadatas": [[{"type": "A"}]],
            "distances": [[0.1]]
        }

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_results)

                result = await self.provider.search_by_text(
                    "query text", top_k=5, include_metadata=True
                )

                self.assertEqual(result, mock_results)

        asyncio.run(run_test())

    def test_search_by_text_with_filters(self):
        """Test text search with filters."""
        filters = {"category": "docs"}

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value={"ids": [[]]})

                await self.provider.search_by_text(
                    "query", top_k=5, filters=filters
                )

                mock_loop.run_in_executor.assert_called_once()

        asyncio.run(run_test())

    def test_search_by_text_error(self):
        """Test text search error handling."""
        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(side_effect=Exception("ChromaDB error"))

                with self.assertRaises(Exception):
                    await self.provider.search_by_text("query", top_k=5)

        asyncio.run(run_test())

    def test_get_document_success(self):
        """Test successful document retrieval."""
        mock_results = {
            "ids": ["doc1"],
            "documents": ["content1"],
            "metadatas": [{"type": "A"}]
        }

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_results)

                result = await self.provider.get_document("doc1")

                expected = {
                    "id": "doc1",
                    "content": "content1",
                    "metadata": {"type": "A"}
                }
                self.assertEqual(result, expected)

        asyncio.run(run_test())

    def test_get_document_not_found(self):
        """Test document not found."""
        mock_results = {"ids": [], "documents": [], "metadatas": []}

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_results)

                result = await self.provider.get_document("nonexistent")

                self.assertIsNone(result)

        asyncio.run(run_test())

    def test_get_document_empty_results(self):
        """Test document retrieval with empty results."""
        mock_results = {"ids": [], "documents": [], "metadatas": []}

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=mock_results)

                result = await self.provider.get_document("doc1")

                self.assertIsNone(result)

        asyncio.run(run_test())

    def test_get_document_error(self):
        """Test document retrieval error handling."""
        async def run_test():
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(side_effect=Exception("ChromaDB error"))

                result = await self.provider.get_document("doc1")

                self.assertIsNone(result)

        asyncio.run(run_test())


class TestConfigProvider(unittest.TestCase):
    """Test ConfigProvider functionality."""

    def test_init_with_custom_loader(self):
        """Test initialization with custom config loader."""
        mock_loader = Mock()
        provider = DefaultConfigProvider(config_loader=mock_loader)

        self.assertEqual(provider.config_loader, mock_loader)
        # When custom loader provided, get_search_config_func is set from loader
        self.assertTrue(hasattr(provider, 'get_search_config_func'))

    @patch('src.utils.config_loader.get_search_config')
    @patch('src.utils.config_loader.get_shared_config')
    def test_init_with_default_loader(self, mock_get_shared, mock_get_search):
        """Test initialization with default config loader."""
        provider = DefaultConfigProvider()

        self.assertEqual(provider.get_search_config_func, mock_get_search)
        self.assertEqual(provider.get_shared_config_func, mock_get_shared)

    def test_get_search_config_with_custom_loader_uses_fallback(self):
        """Test that custom loader actually uses fallback config."""
        mock_loader = Mock()
        provider = DefaultConfigProvider(config_loader=mock_loader)

        # Manually set get_search_config_func to None since it's not set in init when custom loader provided
        provider.get_search_config_func = None

        # The custom loader is stored as config_loader, not get_search_config_func
        # So the get_search_config method uses fallback since get_search_config_func is None
        result = provider.get_search_config()

        # Should return fallback config since get_search_config_func is None
        self.assertEqual(result["default_method"], "semantic")
        self.assertEqual(result["top_k"], 5)
        self.assertEqual(result["similarity_threshold"], 0.0)

    @patch('src.utils.config_loader.get_search_config')
    def test_get_search_config_with_default_loader(self, mock_get_search):
        """Test getting search config with default loader."""
        mock_config = {"default_method": "hybrid"}
        mock_get_search.return_value = mock_config

        with patch('src.utils.config_loader.get_shared_config'):
            provider = DefaultConfigProvider()
            result = provider.get_search_config()

            self.assertEqual(result, mock_config)
            mock_get_search.assert_called_once()

    def test_get_search_config_fallback(self):
        """Test getting search config with fallback."""
        provider = DefaultConfigProvider()
        provider.get_search_config_func = None

        result = provider.get_search_config()

        # Should return fallback config
        self.assertEqual(result["default_method"], "semantic")
        self.assertEqual(result["top_k"], 5)

    def test_get_scoring_weights_success(self):
        """Test getting scoring weights successfully."""
        provider = DefaultConfigProvider()
        provider.get_search_config_func = Mock(return_value={
            "weights": {
                "semantic_weight": 0.8,
                "keyword_weight": 0.2
            }
        })

        result = provider.get_scoring_weights()

        expected = {"semantic": 0.8, "keyword": 0.2}
        self.assertEqual(result, expected)

    def test_get_scoring_weights_missing_weights_section(self):
        """Test error when weights section is missing."""
        provider = DefaultConfigProvider()
        provider.get_search_config_func = Mock(return_value={
            "default_method": "semantic"
        })

        with self.assertRaises(ValueError) as cm:
            provider.get_scoring_weights()

        self.assertIn("Missing 'weights' section", str(cm.exception))

    def test_get_scoring_weights_missing_semantic_weight(self):
        """Test error when semantic_weight is missing."""
        provider = DefaultConfigProvider()
        provider.get_search_config_func = Mock(return_value={
            "weights": {"keyword_weight": 0.3}
        })

        with self.assertRaises(ValueError) as cm:
            provider.get_scoring_weights()

        self.assertIn("Missing 'semantic_weight'", str(cm.exception))

    def test_get_scoring_weights_missing_keyword_weight(self):
        """Test error when keyword_weight is missing."""
        provider = DefaultConfigProvider()
        provider.get_search_config_func = Mock(return_value={
            "weights": {"semantic_weight": 0.7}
        })

        with self.assertRaises(ValueError) as cm:
            provider.get_scoring_weights()

        self.assertIn("Missing 'keyword_weight'", str(cm.exception))


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating providers."""

    def test_create_mock_embedding_provider(self):
        """Test creating mock embedding provider."""
        provider = create_mock_embedding_provider(dimension=512)

        self.assertIsInstance(provider, MockEmbeddingProvider)
        self.assertEqual(provider.dimension, 512)
        self.assertTrue(provider.deterministic)

    def test_create_mock_embedding_provider_default(self):
        """Test creating mock embedding provider with defaults."""
        provider = create_mock_embedding_provider()

        self.assertIsInstance(provider, MockEmbeddingProvider)
        self.assertEqual(provider.dimension, 384)

    def test_create_mock_search_provider(self):
        """Test creating mock search provider."""
        provider = create_mock_search_provider()

        self.assertIsInstance(provider, MockVectorSearchProvider)
        self.assertEqual(provider.documents, {})

    def test_create_mock_config_provider(self):
        """Test creating mock config provider."""
        custom_config = {"search": {"top_k": 3}}
        provider = create_mock_config_provider(custom_config)

        self.assertIsInstance(provider, MockConfigProvider)
        self.assertEqual(provider.config, custom_config)

    def test_create_mock_config_provider_default(self):
        """Test creating mock config provider with defaults."""
        provider = create_mock_config_provider()

        self.assertIsInstance(provider, MockConfigProvider)
        self.assertIn("search", provider.config)

    @patch('sentence_transformers.SentenceTransformer')
    def test_create_embedding_provider(self, mock_sentence_transformer):
        """Test creating production embedding provider."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        provider = create_embedding_provider(
            model_name="custom/model", device="cuda"
        )

        self.assertIsInstance(provider, SentenceTransformerEmbeddingProvider)
        self.assertEqual(provider.model_name, "custom/model")
        self.assertEqual(provider.device, "cuda")

    @patch('sentence_transformers.SentenceTransformer')
    def test_create_embedding_provider_default(self, mock_sentence_transformer):
        """Test creating production embedding provider with defaults."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_sentence_transformer.return_value = mock_model

        provider = create_embedding_provider()

        self.assertIsInstance(provider, SentenceTransformerEmbeddingProvider)
        self.assertEqual(provider.model_name, "BAAI/bge-m3")
        self.assertEqual(provider.device, "cpu")

    @patch('src.utils.config_loader.get_config_section')
    def test_create_vector_search_provider(self, mock_get_config):
        """Test creating ChromaDB search provider."""
        # Mock config to return chromadb as provider
        mock_get_config.return_value = {"provider": "chromadb"}

        # Create a mock collection without a .collection attribute
        mock_collection = Mock(spec=['name', 'count', 'query'])
        provider = create_vector_search_provider(mock_collection)

        self.assertIsInstance(provider, ChromaDBSearchProvider)
        self.assertEqual(provider.collection, mock_collection)

    def test_create_config_provider(self):
        """Test creating production config provider."""
        mock_loader = Mock()
        provider = create_config_provider(mock_loader)

        self.assertIsInstance(provider, ConfigProvider)
        self.assertEqual(provider.config_loader, mock_loader)

    def test_create_config_provider_default(self):
        """Test creating production config provider with defaults."""
        provider = create_config_provider()

        self.assertIsInstance(provider, ConfigProvider)


class TestProviderInterfaces(unittest.TestCase):
    """Test that providers implement required interfaces correctly."""

    def test_mock_embedding_provider_interface(self):
        """Test MockEmbeddingProvider has required methods."""
        provider = MockEmbeddingProvider()

        # Check required methods exist
        self.assertTrue(hasattr(provider, 'encode_text'))
        self.assertTrue(callable(getattr(provider, 'encode_text')))

    def test_mock_search_provider_interface(self):
        """Test MockVectorSearchProvider has required methods."""
        provider = MockVectorSearchProvider()

        # Check required methods exist
        self.assertTrue(hasattr(provider, 'search_by_embedding'))
        self.assertTrue(hasattr(provider, 'search_by_text'))
        self.assertTrue(hasattr(provider, 'get_document'))
        self.assertTrue(callable(getattr(provider, 'search_by_embedding')))

    def test_mock_config_provider_interface(self):
        """Test MockConfigProvider has required methods."""
        provider = MockConfigProvider()

        # Check required methods exist
        self.assertTrue(hasattr(provider, 'get_search_config'))
        self.assertTrue(hasattr(provider, 'get_scoring_weights'))
        self.assertTrue(callable(getattr(provider, 'get_search_config')))

    @patch('sentence_transformers.SentenceTransformer')
    def test_production_embedding_provider_interface(self, mock_sentence_transformer):
        """Test SentenceTransformerEmbeddingProvider has required methods."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_sentence_transformer.return_value = mock_model
        provider = SentenceTransformerEmbeddingProvider()

        # Check required methods exist
        self.assertTrue(hasattr(provider, 'encode_text'))
        self.assertTrue(callable(getattr(provider, 'encode_text')))

    def test_chromadb_search_provider_interface(self):
        """Test ChromaDBSearchProvider has required methods."""
        mock_collection = Mock()
        provider = ChromaDBSearchProvider(mock_collection)

        # Check required methods exist
        self.assertTrue(hasattr(provider, 'search_by_embedding'))
        self.assertTrue(hasattr(provider, 'search_by_text'))
        self.assertTrue(hasattr(provider, 'get_document'))

    def test_production_config_provider_interface(self):
        """Test ConfigProvider has required methods."""
        provider = DefaultConfigProvider()

        # Check required methods exist
        self.assertTrue(hasattr(provider, 'get_search_config'))
        self.assertTrue(hasattr(provider, 'get_scoring_weights'))
        self.assertTrue(callable(getattr(provider, 'get_search_config')))


if __name__ == "__main__":
    unittest.main()