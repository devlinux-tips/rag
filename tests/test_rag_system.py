"""
Integration tests for complete Croatian RAG system.
Tests end-to-end pipeline functionality and component integration.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.pipeline.config import OllamaConfig, ProcessingConfig, RAGConfig
from src.pipeline.rag_system import CroatianRAGSystem, RAGQuery, RAGResponse, create_rag_system


class TestRAGQuery:
    """Test RAG query structure."""

    def test_rag_query_creation(self):
        """Test RAG query creation."""
        query = RAGQuery(
            text="Što je Zagreb?",
            query_id="test-001",
            user_id="user-123",
            context_filters={"language": "hr"},
            max_results=5,
            metadata={"test": True},
        )

        assert query.text == "Što je Zagreb?"
        assert query.query_id == "test-001"
        assert query.user_id == "user-123"
        assert query.context_filters == {"language": "hr"}
        assert query.max_results == 5
        assert query.metadata == {"test": True}


class TestRAGResponse:
    """Test RAG response structure."""

    def test_rag_response_creation(self):
        """Test RAG response creation."""
        response = RAGResponse(
            answer="Zagreb je glavni grad Hrvatske.",
            query="Što je Zagreb?",
            retrieved_chunks=[{"content": "Zagreb info", "similarity_score": 0.9}],
            confidence=0.85,
            generation_time=2.5,
            retrieval_time=1.0,
            total_time=3.5,
            sources=["document1.pdf"],
            metadata={"test": True},
        )

        assert response.answer == "Zagreb je glavni grad Hrvatske."
        assert response.query == "Što je Zagreb?"
        assert len(response.retrieved_chunks) == 1
        assert response.confidence == 0.85
        assert response.has_high_confidence is True
        assert response.total_time == 3.5
        assert response.sources == ["document1.pdf"]

    def test_high_confidence_property(self):
        """Test high confidence detection."""
        # High confidence
        high_response = RAGResponse(
            answer="Test",
            query="Test",
            retrieved_chunks=[],
            confidence=0.85,
            generation_time=1.0,
            retrieval_time=1.0,
            total_time=2.0,
            sources=[],
            metadata={},
        )
        assert high_response.has_high_confidence is True

        # Low confidence
        low_response = RAGResponse(
            answer="Test",
            query="Test",
            retrieved_chunks=[],
            confidence=0.75,
            generation_time=1.0,
            retrieval_time=1.0,
            total_time=2.0,
            sources=[],
            metadata={},
        )
        assert low_response.has_high_confidence is False


class TestCroatianRAGSystem:
    """Test complete Croatian RAG system."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = RAGConfig()
        config.processing = ProcessingConfig(max_chunk_size=256, chunk_overlap=25)
        config.ollama = OllamaConfig(model="llama3.1:8b", timeout=30.0)
        return config

    @pytest.fixture
    def rag_system(self, config):
        """Create test RAG system."""
        return CroatianRAGSystem(config)

    def test_system_creation(self, rag_system, config):
        """Test RAG system creation."""
        assert rag_system.config == config
        assert rag_system._initialized is False
        assert rag_system._document_count == 0
        assert rag_system._query_count == 0

    @pytest.mark.asyncio
    async def test_component_initialization(self, rag_system):
        """Test component initialization."""
        # Mock all component initializations
        with patch.multiple(
            rag_system,
            _initialize_preprocessing=AsyncMock(),
            _initialize_vectordb=AsyncMock(),
            _initialize_retrieval=AsyncMock(),
            _initialize_generation=AsyncMock(),
        ):
            await rag_system.initialize()

            assert rag_system._initialized is True
            rag_system._initialize_preprocessing.assert_called_once()
            rag_system._initialize_vectordb.assert_called_once()
            rag_system._initialize_retrieval.assert_called_once()
            rag_system._initialize_generation.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_initialization(self, rag_system):
        """Test that double initialization is handled gracefully."""
        with patch.multiple(
            rag_system,
            _initialize_preprocessing=AsyncMock(),
            _initialize_vectordb=AsyncMock(),
            _initialize_retrieval=AsyncMock(),
            _initialize_generation=AsyncMock(),
        ):
            # First initialization
            await rag_system.initialize()

            # Second initialization should be skipped
            await rag_system.initialize()

            # Components should only be initialized once
            rag_system._initialize_preprocessing.assert_called_once()

    @pytest.mark.asyncio
    async def test_preprocessing_initialization(self, rag_system):
        """Test preprocessing component initialization."""
        await rag_system._initialize_preprocessing()

        assert rag_system._document_extractor is not None
        assert rag_system._text_cleaner is not None
        assert rag_system._chunker is not None

    @pytest.mark.asyncio
    async def test_vectordb_initialization(self, rag_system):
        """Test vector database initialization."""
        # Mock the embedding model and storage
        with (
            patch("src.pipeline.rag_system.EmbeddingModel") as mock_embedding,
            patch("src.pipeline.rag_system.ChromaVectorStorage") as mock_storage,
            patch("src.pipeline.rag_system.VectorSearchEngine") as mock_search,
        ):
            mock_embedding_instance = AsyncMock()
            mock_embedding.return_value = mock_embedding_instance

            mock_storage_instance = AsyncMock()
            mock_storage.return_value = mock_storage_instance

            await rag_system._initialize_vectordb()

            # Check that components were created
            mock_embedding.assert_called_once()
            mock_storage.assert_called_once()
            mock_search.assert_called_once()

            # Check that async methods were called
            mock_embedding_instance.load_model.assert_called_once()
            mock_storage_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieval_initialization(self, rag_system):
        """Test retrieval component initialization."""
        # Mock search engine dependency
        rag_system._search_engine = Mock()

        await rag_system._initialize_retrieval()

        assert rag_system._query_processor is not None
        assert rag_system._ranker is not None
        assert rag_system._retriever is not None

    @pytest.mark.asyncio
    async def test_generation_initialization(self, rag_system):
        """Test generation component initialization."""
        with (
            patch("src.pipeline.rag_system.OllamaClient") as mock_client,
            patch("src.pipeline.rag_system.create_response_parser") as mock_parser,
        ):
            mock_client_instance = Mock()
            mock_client_instance.health_check.return_value = True
            mock_client_instance.get_available_models.return_value = ["llama3.1:8b"]
            mock_client.return_value = mock_client_instance

            await rag_system._initialize_generation()

            assert rag_system._ollama_client is not None
            assert rag_system._response_parser is not None
            mock_client_instance.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_success(self, rag_system):
        """Test successful document addition."""
        # Mock all required components
        await self._mock_all_components(rag_system)

        # Mock document processing
        rag_system._document_extractor.extract_text_async = AsyncMock(
            return_value="Zagreb je glavni grad Hrvatske."
        )
        rag_system._text_cleaner.clean_text = Mock(return_value="zagreb je glavni grad hrvatske")

        # Mock chunking
        mock_chunk = Mock()
        mock_chunk.content = "zagreb je glavni grad hrvatske"
        mock_chunk.chunk_type = "paragraph"
        mock_chunk.metadata = {"confidence": 0.9}
        rag_system._chunker.create_chunks = Mock(return_value=[mock_chunk])

        # Mock embedding and storage
        rag_system._embedding_model.generate_embedding_async = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )
        rag_system._vector_storage.store_embedding_async = AsyncMock()

        # Test document addition
        doc_paths = ["test_doc.txt"]
        result = await rag_system.add_documents(doc_paths)

        assert result["processed_documents"] == 1
        assert result["failed_documents"] == 0
        assert result["total_chunks"] == 1
        assert rag_system._document_count == 1

    @pytest.mark.asyncio
    async def test_add_documents_failure(self, rag_system):
        """Test document addition with failures."""
        await self._mock_all_components(rag_system)

        # Mock extraction failure
        rag_system._document_extractor.extract_text_async = AsyncMock(
            side_effect=Exception("Extraction failed")
        )

        doc_paths = ["bad_doc.txt"]
        result = await rag_system.add_documents(doc_paths)

        assert result["processed_documents"] == 0
        assert result["failed_documents"] == 1
        assert result["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_query_success(self, rag_system):
        """Test successful query processing."""
        await self._mock_all_components(rag_system)

        # Mock query processing
        mock_processed_query = Mock()
        mock_processed_query.original = "Što je Zagreb?"
        mock_processed_query.processed = "što je zagreb"
        mock_processed_query.query_type.value = "factual"
        mock_processed_query.keywords = ["što", "zagreb"]
        mock_processed_query.confidence = 0.8

        rag_system._query_processor.process_query = Mock(return_value=mock_processed_query)

        # Mock retrieval
        mock_retrieval_result = Mock()
        mock_retrieval_result.results = [
            Mock(
                content="Zagreb je glavni grad Hrvatske.",
                similarity_score=0.9,
                final_score=0.85,
                metadata={"source": "test.pdf", "chunk_index": 0},
            )
        ]
        mock_retrieval_result.strategy_used = "semantic"

        rag_system._retriever.retrieve_async = AsyncMock(return_value=mock_retrieval_result)

        # Mock generation
        mock_generation_response = Mock()
        mock_generation_response.text = "Zagreb je glavni i najveći grad Republike Hrvatske."
        mock_generation_response.model = "llama3.1:8b"
        mock_generation_response.tokens_used = 50
        mock_generation_response.confidence = 0.85
        mock_generation_response.metadata = {}

        rag_system._ollama_client.generate_text_async = AsyncMock(
            return_value=mock_generation_response
        )

        # Mock response parsing
        mock_parsed_response = Mock()
        mock_parsed_response.content = "Zagreb je glavni i najveći grad Republike Hrvatske."
        mock_parsed_response.confidence = 0.85
        mock_parsed_response.language = "hr"
        mock_parsed_response.sources_mentioned = ["test.pdf"]

        rag_system._response_parser.parse_response = Mock(return_value=mock_parsed_response)

        # Test query
        query = RAGQuery(text="Što je Zagreb?", query_id="test-001")
        response = await rag_system.query(query)

        assert response.answer == "Zagreb je glavni i najveći grad Republike Hrvatske."
        assert response.query == "Što je Zagreb?"
        assert response.confidence == 0.85
        assert len(response.retrieved_chunks) == 1
        assert "test.pdf" in response.sources
        assert rag_system._query_count == 1

    @pytest.mark.asyncio
    async def test_query_failure(self, rag_system):
        """Test query processing with failure."""
        await self._mock_all_components(rag_system)

        # Mock query processor failure
        rag_system._query_processor.process_query = Mock(
            side_effect=Exception("Query processing failed")
        )

        query = RAGQuery(text="Test query")
        response = await rag_system.query(query)

        assert "greška" in response.answer.lower()
        assert response.confidence == 0.0
        assert len(response.retrieved_chunks) == 0
        assert "error" in response.metadata

    @pytest.mark.asyncio
    async def test_health_check(self, rag_system):
        """Test system health check."""
        await self._mock_all_components(rag_system)

        # Mock component health
        rag_system._vector_storage.get_document_count = AsyncMock(return_value=100)
        rag_system._ollama_client.health_check = Mock(return_value=True)
        rag_system._ollama_client.get_available_models = Mock(return_value=["llama3.1:8b"])

        health = await rag_system.health_check()

        assert health["system_status"] == "healthy"
        assert "preprocessing" in health["components"]
        assert "vectordb" in health["components"]
        assert "retrieval" in health["components"]
        assert "generation" in health["components"]
        assert health["metrics"]["total_chunks"] == 100

    @pytest.mark.asyncio
    async def test_get_system_stats(self, rag_system):
        """Test system statistics."""
        await self._mock_all_components(rag_system)

        rag_system._document_count = 10
        rag_system._query_count = 25
        rag_system._vector_storage.get_document_count = AsyncMock(return_value=150)

        stats = await rag_system.get_system_stats()

        assert stats["documents"] == 10
        assert stats["queries"] == 25
        assert stats["chunks"] == 150
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_system_close(self, rag_system):
        """Test system shutdown."""
        await self._mock_all_components(rag_system)

        rag_system._ollama_client.close = AsyncMock()
        rag_system._vector_storage.close = AsyncMock()

        await rag_system.close()

        rag_system._ollama_client.close.assert_called_once()
        rag_system._vector_storage.close.assert_called_once()

    async def _mock_all_components(self, rag_system):
        """Helper to mock all system components."""
        # Mock preprocessing
        rag_system._document_extractor = Mock()
        rag_system._text_cleaner = Mock()
        rag_system._chunker = Mock()

        # Mock vectordb
        rag_system._embedding_model = Mock()
        rag_system._vector_storage = Mock()
        rag_system._search_engine = Mock()

        # Mock retrieval
        rag_system._query_processor = Mock()
        rag_system._retriever = Mock()
        rag_system._ranker = Mock()

        # Mock generation
        rag_system._ollama_client = Mock()
        rag_system._response_parser = Mock()

        rag_system._initialized = True


class TestRAGSystemIntegration:
    """Test RAG system integration scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline with mocked components."""
        # This would be a comprehensive integration test
        # For now, we'll test the initialization and basic structure

        config = RAGConfig()
        config.ollama.timeout = 10.0  # Shorter timeout for tests

        system = CroatianRAGSystem(config)

        # Mock initialization to avoid external dependencies
        with patch.multiple(
            system,
            _initialize_preprocessing=AsyncMock(),
            _initialize_vectordb=AsyncMock(),
            _initialize_retrieval=AsyncMock(),
            _initialize_generation=AsyncMock(),
        ):
            await system.initialize()

            assert system._initialized is True

            # Test that we can create queries and responses
            query = RAGQuery(text="Test Croatian query", query_id="integration-test")
            assert query.text == "Test Croatian query"

            # Mock a complete response
            response = RAGResponse(
                answer="Mocked Croatian response",
                query=query.text,
                retrieved_chunks=[],
                confidence=0.8,
                generation_time=1.0,
                retrieval_time=0.5,
                total_time=1.5,
                sources=[],
                metadata={"test": True},
            )

            assert response.has_high_confidence is True
            assert response.total_time == 1.5

            await system.close()

    @pytest.mark.asyncio
    async def test_create_rag_system_factory(self):
        """Test RAG system factory function."""
        # Test with default config
        with patch.multiple(
            "src.pipeline.rag_system.CroatianRAGSystem", initialize=AsyncMock()
        ) as mock_methods:
            with patch("src.pipeline.rag_system.CroatianRAGSystem") as mock_class:
                mock_instance = Mock()
                mock_instance.initialize = AsyncMock()
                mock_class.return_value = mock_instance

                system = await create_rag_system()

                mock_class.assert_called_once()
                mock_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_rag_system_with_config_file(self):
        """Test RAG system creation with config file."""
        # Create temporary config file
        config_data = {
            "ollama": {"model": "custom-model", "temperature": 0.5},
            "processing": {"max_chunk_size": 1024},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.pipeline.rag_system.CroatianRAGSystem") as mock_class:
                mock_instance = Mock()
                mock_instance.initialize = AsyncMock()
                mock_class.return_value = mock_instance

                system = await create_rag_system(config_path)

                # Verify that system was created and initialized
                mock_class.assert_called_once()
                mock_instance.initialize.assert_called_once()

        finally:
            Path(config_path).unlink()

    def test_croatian_specific_features(self):
        """Test Croatian-specific features in the system."""
        config = RAGConfig()
        system = CroatianRAGSystem(config)

        # Test Croatian config defaults
        assert config.croatian.enable_morphological_expansion is True
        assert config.croatian.enable_cultural_context is True
        assert config.ollama.preserve_diacritics is True
        assert config.ollama.include_cultural_context is True

        # Test Croatian language settings
        assert config.ollama.model == "llama3.1:8b"  # Good for Croatian
        assert config.embedding.model_name == "bge-m3"


class TestCLIInterface:
    """Test CLI interface functionality."""

    @pytest.mark.asyncio
    async def test_main_function_health_check(self):
        """Test main function health check."""
        # This would require mocking argparse and the entire system
        # For now, we just verify the function structure exists
        from src.pipeline.rag_system import main

        assert callable(main)

    def test_query_and_response_json_serialization(self):
        """Test that queries and responses can be JSON serialized."""
        # Test query serialization
        query = RAGQuery(text="Što je Zagreb?", query_id="test-001", metadata={"test": True})

        # Should be able to convert to dict for JSON
        query_dict = {
            "text": query.text,
            "query_id": query.query_id,
            "metadata": query.metadata,
        }

        json_str = json.dumps(query_dict, ensure_ascii=False)
        assert "Zagreb" in json_str

        # Test response serialization
        response = RAGResponse(
            answer="Zagreb je glavni grad Hrvatske.",
            query="Što je Zagreb?",
            retrieved_chunks=[{"content": "test", "score": 0.9}],
            confidence=0.85,
            generation_time=1.0,
            retrieval_time=0.5,
            total_time=1.5,
            sources=["test.pdf"],
            metadata={"test": True},
        )

        response_dict = {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources,
            "total_time": response.total_time,
        }

        json_str = json.dumps(response_dict, ensure_ascii=False)
        assert "Zagreb" in json_str


if __name__ == "__main__":
    pytest.main([__file__])
