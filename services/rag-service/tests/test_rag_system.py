"""
Comprehensive tests for rag_system.py demonstrating 100% testability.
Tests pure functions, dependency injection, and integration scenarios.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.pipeline.rag_system import (  # Pure functions; Data classes; Main class; Mock implementations
    ComponentHealth, DocumentProcessingResult, MockDocumentChunker,
    MockDocumentExtractor, MockEmbeddingModel, MockGenerationClient,
    MockPromptBuilder, MockResponseParser, MockRetriever, MockTextCleaner,
    MockVectorStorage, RAGQuery, RAGResponse, RAGSystemV2, SystemHealth,
    SystemStats, build_response_metadata, calculate_overall_health,
    calculate_processing_metrics, create_chunk_metadata, create_error_response,
    create_language_collection_name, create_mock_rag_system,
    create_rag_system_v2, evaluate_component_health, evaluate_ollama_health,
    extract_sources_from_chunks, prepare_chunk_info, validate_document_paths,
    validate_language_code, validate_query)


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_validate_language_code_valid(self):
        """Test language code validation with valid inputs."""
        assert validate_language_code("hr") == "hr"
        assert validate_language_code("en") == "en"
        assert validate_language_code("multilingual") == "multilingual"

        # Test normalization
        assert validate_language_code("HR") == "hr"
        assert validate_language_code("  en  ") == "en"

    def test_validate_language_code_invalid(self):
        """Test language code validation with invalid inputs."""
        with pytest.raises(
            ValueError, match="Language code must be a non-empty string"
        ):
            validate_language_code("")

        with pytest.raises(
            ValueError, match="Language code must be a non-empty string"
        ):
            validate_language_code(None)

        with pytest.raises(ValueError, match="Unsupported language"):
            validate_language_code("fr")

        with pytest.raises(ValueError, match="Unsupported language"):
            validate_language_code("invalid")

    def test_create_language_collection_name(self):
        """Test language collection name creation."""
        assert create_language_collection_name("hr") == "croatian_documents"
        assert create_language_collection_name("en") == "english_documents"
        assert (
            create_language_collection_name("multilingual") == "multilingual_documents"
        )
        assert create_language_collection_name("unknown") == "unknown_documents"

    def test_validate_document_paths_valid(self):
        """Test document path validation with valid paths."""
        # Create temporary files for testing
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "doc1.txt"
            file2 = Path(tmpdir) / "doc2.txt"
            file1.write_text("content1")
            file2.write_text("content2")

            paths = validate_document_paths([str(file1), str(file2)])
            assert len(paths) == 2
            assert all(isinstance(p, Path) for p in paths)
            assert all(p.exists() for p in paths)

    def test_validate_document_paths_invalid(self):
        """Test document path validation with invalid inputs."""
        with pytest.raises(ValueError, match="Document paths list cannot be empty"):
            validate_document_paths([])

        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_document_paths(["", "valid.txt"])

        with pytest.raises(ValueError, match="Document path does not exist"):
            validate_document_paths(["/nonexistent/file.txt"])

    def test_validate_query_valid(self):
        """Test query validation with valid inputs."""
        query = RAGQuery(text="Test query", language="hr")
        validated = validate_query(query)

        assert validated.text == "Test query"
        assert validated.language == "hr"
        assert validated.max_results == 5  # Default value

    def test_validate_query_invalid(self):
        """Test query validation with invalid inputs."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            validate_query(RAGQuery(text="", language="hr"))

        with pytest.raises(ValueError, match="Query text cannot be empty"):
            validate_query(RAGQuery(text="   ", language="hr"))

        with pytest.raises(ValueError, match="Query language must be specified"):
            validate_query(RAGQuery(text="Valid query", language=""))

        with pytest.raises(ValueError, match="max_results must be positive"):
            validate_query(RAGQuery(text="Valid query", language="hr", max_results=-1))

    def test_calculate_processing_metrics(self):
        """Test processing metrics calculation."""
        metrics = calculate_processing_metrics(10, 5.0, 50)

        assert metrics["documents_per_second"] == 2.0
        assert metrics["chunks_per_second"] == 10.0
        assert metrics["average_chunks_per_document"] == 5.0

        # Test edge case with zero time
        metrics_zero = calculate_processing_metrics(10, 0.0, 50)
        assert metrics_zero["documents_per_second"] == 0.0

    def test_create_chunk_metadata(self):
        """Test chunk metadata creation."""

        class MockChunk:
            def __init__(self):
                self.chunk_id = "test_chunk_1"
                self.start_char = 0
                self.end_char = 100
                self.word_count = 20
                self.content = "Test content"

        chunk = MockChunk()
        metadata = create_chunk_metadata("test.txt", 0, chunk, "hr", 1234567890.0)

        assert metadata["source"] == "test.txt"
        assert metadata["chunk_index"] == 0
        assert metadata["language"] == "hr"
        assert metadata["chunk_id"] == "test_chunk_1"
        assert metadata["processing_timestamp"] == 1234567890.0

    def test_extract_sources_from_chunks(self):
        """Test source extraction from chunks."""
        chunks = [
            {"metadata": {"source": "doc1.txt", "other": "data"}},
            {"metadata": {"source": "doc2.txt"}},
            {"metadata": {"source": "doc1.txt"}},  # Duplicate
            {"metadata": {"other": "no_source"}},  # No source
        ]

        sources = extract_sources_from_chunks(chunks)
        assert set(sources) == {"doc1.txt", "doc2.txt"}

    def test_prepare_chunk_info(self):
        """Test chunk information preparation."""
        chunk_result = {
            "content": "Test chunk content",
            "similarity_score": 0.9,
            "final_score": 0.95,
            "metadata": {
                "source": "test.txt",
                "chunk_index": 2,
                "ranking_signals": {"signal1": "value1"},
            },
        }

        # Without debug info
        chunk_info = prepare_chunk_info(chunk_result, return_debug_info=False)
        expected_keys = {
            "content",
            "similarity_score",
            "final_score",
            "source",
            "chunk_index",
        }
        assert set(chunk_info.keys()) == expected_keys

        # With debug info
        chunk_info_debug = prepare_chunk_info(chunk_result, return_debug_info=True)
        assert "metadata" in chunk_info_debug
        assert "signals" in chunk_info_debug

    def test_create_error_response(self):
        """Test error response creation."""
        query = RAGQuery(text="Test query", language="hr", query_id="test123")
        error = ValueError("Test error")
        start_time = time.time()

        response = create_error_response(query, error, start_time)

        assert response.query == "Test query"
        assert response.confidence == 0.0
        assert "greška" in response.answer  # Croatian error message
        assert response.metadata["error"] == "Test error"
        assert response.metadata["query_id"] == "test123"

        # Test English error message
        query_en = RAGQuery(text="Test query", language="en")
        response_en = create_error_response(query_en, error, start_time)
        assert "error occurred" in response_en.answer

    def test_evaluate_component_health(self):
        """Test component health evaluation."""
        # All healthy components
        healthy_components = [Mock(), Mock(), Mock()]
        health = evaluate_component_health("test", healthy_components)
        assert health.status == "healthy"
        assert "All test components loaded" in health.details

        # Some components missing
        unhealthy_components = [Mock(), None, Mock()]
        health_bad = evaluate_component_health("test", unhealthy_components)
        assert health_bad.status == "unhealthy"

    def test_evaluate_ollama_health(self):
        """Test Ollama service health evaluation."""
        # No client
        health_no_client = evaluate_ollama_health(None, "test-model")
        assert health_no_client.status == "unhealthy"
        assert "not initialized" in health_no_client.details

        # Unhealthy service
        mock_client = Mock()
        mock_client.health_check.return_value = False
        health_unhealthy = evaluate_ollama_health(mock_client, "test-model")
        assert health_unhealthy.status == "unhealthy"

        # Healthy service, missing model
        mock_client.health_check.return_value = True
        mock_client.get_available_models.return_value = ["other-model"]
        health_degraded = evaluate_ollama_health(mock_client, "test-model")
        assert health_degraded.status == "degraded"

        # Healthy service, available model
        mock_client.get_available_models.return_value = ["test-model"]
        health_healthy = evaluate_ollama_health(mock_client, "test-model")
        assert health_healthy.status == "healthy"

    def test_calculate_overall_health(self):
        """Test overall health calculation."""
        # All healthy
        all_healthy = {
            "comp1": ComponentHealth("healthy", "ok"),
            "comp2": ComponentHealth("healthy", "ok"),
        }
        assert calculate_overall_health(all_healthy) == "healthy"

        # Mixed health
        mixed_health = {
            "comp1": ComponentHealth("healthy", "ok"),
            "comp2": ComponentHealth("degraded", "warning"),
        }
        assert calculate_overall_health(mixed_health) == "degraded"

        # All unhealthy
        all_unhealthy = {
            "comp1": ComponentHealth("unhealthy", "error"),
            "comp2": ComponentHealth("unhealthy", "error"),
        }
        assert calculate_overall_health(all_unhealthy) == "unhealthy"


class TestDataClasses:
    """Test data class functionality."""

    def test_rag_query_creation(self):
        """Test RAGQuery data class."""
        query = RAGQuery(
            text="Test query", language="hr", query_id="123", max_results=10
        )
        assert query.text == "Test query"
        assert query.language == "hr"
        assert query.query_id == "123"
        assert query.max_results == 10

    def test_rag_response_creation(self):
        """Test RAGResponse data class."""
        response = RAGResponse(
            answer="Test answer",
            query="Test query",
            retrieved_chunks=[],
            confidence=0.9,
            generation_time=1.0,
            retrieval_time=0.5,
            total_time=1.5,
            sources=["doc1.txt"],
            metadata={"key": "value"},
        )

        assert response.answer == "Test answer"
        assert response.confidence == 0.9
        assert response.has_high_confidence is True  # >= 0.8

        # Test low confidence
        low_conf_response = RAGResponse(
            answer="Test",
            query="Test",
            retrieved_chunks=[],
            confidence=0.5,
            generation_time=0,
            retrieval_time=0,
            total_time=0,
            sources=[],
            metadata={},
        )
        assert low_conf_response.has_high_confidence is False


class TestMockImplementations:
    """Test mock implementations for dependency injection."""

    def test_mock_document_extractor(self):
        """Test mock document extractor."""
        extractor = MockDocumentExtractor()
        result = extractor.extract_text(Path("test.txt"))
        assert "test.txt" in result
        assert "Mock content" in result

    def test_mock_text_cleaner(self):
        """Test mock text cleaner."""
        cleaner = MockTextCleaner()
        result = cleaner.clean_text("  test text  ")
        assert result == "test text"

        # Test setup method doesn't raise
        cleaner.setup_language_environment()

    def test_mock_document_chunker(self):
        """Test mock document chunker."""
        chunker = MockDocumentChunker()
        content = "A" * 250  # 250 characters
        chunks = chunker.chunk_document(content, "test.txt")

        # Should create multiple chunks for long content
        assert len(chunks) > 1
        assert all(hasattr(chunk, "content") for chunk in chunks)
        assert all(hasattr(chunk, "chunk_id") for chunk in chunks)

    def test_mock_embedding_model(self):
        """Test mock embedding model."""
        model = MockEmbeddingModel()
        model.load_model()  # Should not raise

        embedding = model.encode_text("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 10  # Mock returns 10-dim embeddings
        assert all(0 <= val <= 1 for val in embedding)

    def test_mock_vector_storage(self):
        """Test mock vector storage."""
        storage = MockVectorStorage()
        storage.create_collection()  # Should not raise

        # Test adding documents
        storage.add_documents(
            documents=["doc1", "doc2"],
            metadatas=[{"key": "val1"}, {"key": "val2"}],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )

        assert storage.get_document_count() == 2

    @pytest.mark.asyncio
    async def test_mock_generation_client(self):
        """Test mock generation client."""
        client = MockGenerationClient()

        assert client.health_check() is True
        assert "mock-model" in client.get_available_models()

        response = await client.generate_text_async({"prompt": "test"})
        assert hasattr(response, "text")
        assert hasattr(response, "model")

        await client.close()  # Should not raise

    def test_mock_response_parser(self):
        """Test mock response parser."""
        parser = MockResponseParser()
        result = parser.parse_response("test response", "test query", ["context"])

        assert hasattr(result, "content")
        assert hasattr(result, "confidence")
        assert result.content == "test response"

    def test_mock_prompt_builder(self):
        """Test mock prompt builder."""
        builder = MockPromptBuilder()
        system_prompt, user_prompt = builder.build_prompt(
            query="test query", context_chunks=["chunk1", "chunk2"]
        )

        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert "test query" in user_prompt

    @pytest.mark.asyncio
    async def test_mock_retriever(self):
        """Test mock retriever."""
        retriever = MockRetriever()
        results = await retriever.retrieve("test query", max_results=3)

        assert hasattr(results, "category")
        assert hasattr(results, "documents")
        assert len(results.documents) > 0


class TestRAGSystemV2WithMocks:
    """Test RAGSystemV2 with mock dependencies."""

    @pytest.fixture
    def mock_rag_system(self):
        """Create RAG system with all mocks."""
        return create_mock_rag_system("hr", {"test": "config"})

    @pytest.mark.asyncio
    async def test_initialization(self, mock_rag_system):
        """Test system initialization."""
        assert not mock_rag_system._initialized

        await mock_rag_system.initialize()

        assert mock_rag_system._initialized is True
        assert mock_rag_system.language == "hr"

    @pytest.mark.asyncio
    async def test_add_documents_success(self, mock_rag_system):
        """Test successful document processing."""
        # Create temporary test files
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            doc1 = Path(tmpdir) / "doc1.txt"
            doc2 = Path(tmpdir) / "doc2.txt"
            doc1.write_text("Test document 1 content")
            doc2.write_text("Test document 2 content")

            result = await mock_rag_system.add_documents([str(doc1), str(doc2)])

            assert isinstance(result, DocumentProcessingResult)
            assert result.processed_documents == 2
            assert result.failed_documents == 0
            assert result.total_chunks > 0
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_add_documents_validation_error(self, mock_rag_system):
        """Test document processing with validation errors."""
        with pytest.raises(ValueError, match="Document paths list cannot be empty"):
            await mock_rag_system.add_documents([])

    @pytest.mark.asyncio
    async def test_query_success(self, mock_rag_system):
        """Test successful query execution."""
        query = RAGQuery(text="Test query", language="hr")

        response = await mock_rag_system.query(query)

        assert isinstance(response, RAGResponse)
        assert response.query == "Test query"
        assert response.answer  # Should have content
        assert response.confidence > 0
        assert response.total_time > 0

    @pytest.mark.asyncio
    async def test_query_with_invalid_input(self, mock_rag_system):
        """Test query with invalid input."""
        invalid_query = RAGQuery(text="", language="hr")  # Empty text

        response = await mock_rag_system.query(invalid_query)

        # Should return error response, not raise exception
        assert isinstance(response, RAGResponse)
        assert response.confidence == 0.0
        assert "greška" in response.answer  # Croatian error message

    @pytest.mark.asyncio
    async def test_health_check(self, mock_rag_system):
        """Test system health check."""
        health = await mock_rag_system.health_check()

        assert isinstance(health, SystemHealth)
        assert health.system_status in ["healthy", "degraded", "unhealthy"]
        assert "preprocessing" in health.components
        assert "vectordb" in health.components
        assert "retrieval" in health.components
        assert "generation" in health.components

    @pytest.mark.asyncio
    async def test_get_system_stats(self, mock_rag_system):
        """Test system statistics."""
        stats = await mock_rag_system.get_system_stats()

        assert isinstance(stats, SystemStats)
        assert stats.system["language"] == "hr"
        assert "active_collection" in stats.collections
        assert "embedding_model" in stats.models

    @pytest.mark.asyncio
    async def test_close(self, mock_rag_system):
        """Test system cleanup."""
        await mock_rag_system.initialize()
        await mock_rag_system.close()  # Should not raise


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_rag_workflow(self):
        """Test complete RAG workflow from document processing to query."""
        rag_system = create_mock_rag_system("hr")

        # Initialize system
        await rag_system.initialize()

        # Add documents
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Path(tmpdir) / "croatian_doc.txt"
            doc.write_text(
                "Ovo je hrvatski dokument o tehnologiji. RAG sustav je napredna AI tehnologija."
            )

            processing_result = await rag_system.add_documents([str(doc)])
            assert processing_result.processed_documents == 1

        # Query the system
        query = RAGQuery(text="Što je RAG sustav?", language="hr", max_results=3)
        response = await rag_system.query(
            query, return_sources=True, return_debug_info=True
        )

        assert isinstance(response, RAGResponse)
        assert response.query == "Što je RAG sustav?"
        assert len(response.retrieved_chunks) > 0
        assert response.total_time > 0
        assert "debug" in response.metadata or "categorization" in response.metadata

        # Health check
        health = await rag_system.health_check()
        assert health.system_status in ["healthy", "degraded"]

        # Clean shutdown
        await rag_system.close()

    @pytest.mark.asyncio
    async def test_multilingual_workflow(self):
        """Test multilingual document processing."""
        # Test with English system
        rag_en = create_mock_rag_system("en")
        await rag_en.initialize()

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            doc_en = Path(tmpdir) / "english_doc.txt"
            doc_en.write_text("This is an English document about RAG technology.")

            result_en = await rag_en.add_documents([str(doc_en)])
            assert result_en.processed_documents == 1

        query_en = RAGQuery(text="What is RAG?", language="en")
        response_en = await rag_en.query(query_en)
        assert response_en.query == "What is RAG?"

        # Test with multilingual system
        rag_multi = create_mock_rag_system("multilingual")
        await rag_multi.initialize()

        stats = await rag_multi.get_system_stats()
        assert stats.collections["collection_type"] == "multilingual_documents"

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error handling and recovery."""
        # Create system with failing components
        failing_client = MockGenerationClient(healthy=False)

        rag_system = create_rag_system_v2(
            language="hr",
            document_extractor=MockDocumentExtractor(),
            text_cleaner=MockTextCleaner(),
            document_chunker=MockDocumentChunker(),
            embedding_model=MockEmbeddingModel(),
            vector_storage=MockVectorStorage(),
            search_engine=None,
            query_processor=None,
            retriever=MockRetriever(),
            hierarchical_retriever=MockRetriever(),
            ranker=None,
            generation_client=failing_client,
            response_parser=MockResponseParser(),
            prompt_builder=MockPromptBuilder(),
        )

        # Health check should detect unhealthy generation component
        health = await rag_system.health_check()
        assert health.components["generation"].status == "unhealthy"
        assert health.system_status in ["degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent RAG operations."""
        rag_system = create_mock_rag_system("hr")
        await rag_system.initialize()

        # Create multiple queries
        queries = [RAGQuery(text=f"Test query {i}", language="hr") for i in range(5)]

        # Execute queries concurrently
        tasks = [rag_system.query(query) for query in queries]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(isinstance(r, RAGResponse) for r in responses)
        assert all(r.confidence > 0 for r in responses)

    @pytest.mark.asyncio
    async def test_large_document_batch(self):
        """Test processing large batches of documents."""
        rag_system = create_mock_rag_system("hr")
        await rag_system.initialize()

        # Create many test documents
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            doc_paths = []
            for i in range(20):
                doc = Path(tmpdir) / f"doc_{i}.txt"
                doc.write_text(
                    f"Document {i} content with Croatian text: tehnologija, RAG sustav, analiza."
                )
                doc_paths.append(str(doc))

            # Process in small batches
            result = await rag_system.add_documents(doc_paths, batch_size=5)

            assert result.processed_documents == 20
            assert result.failed_documents == 0
            assert result.documents_per_second > 0

    def test_factory_function(self):
        """Test factory function for creating RAG system."""
        rag_system = create_rag_system_v2(
            language="en",
            document_extractor=MockDocumentExtractor(),
            text_cleaner=MockTextCleaner(),
            document_chunker=MockDocumentChunker(),
            embedding_model=MockEmbeddingModel(),
            vector_storage=MockVectorStorage(),
            search_engine=None,
            query_processor=None,
            retriever=MockRetriever(),
            hierarchical_retriever=MockRetriever(),
            ranker=None,
            generation_client=MockGenerationClient(),
            response_parser=MockResponseParser(),
            prompt_builder=MockPromptBuilder(),
            config={"test": "config"},
        )

        assert isinstance(rag_system, RAGSystemV2)
        assert rag_system.language == "en"
        assert rag_system.config == {"test": "config"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
