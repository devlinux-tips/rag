"""
Comprehensive tests for RAG system pipeline with protocol validation.
Tests pure functions, data structures, protocols, and RAG system orchestration.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.pipeline.rag_system import (
    RAGQuery,
    RAGResponse,
    RAGSystem,
    ComponentHealth,
    SystemHealth,
    SystemStats,
    DocumentProcessingResult,
    # Pure functions
    validate_language_code,
    create_language_collection_name,
    validate_document_paths,
    validate_query,
    calculate_processing_metrics,
    create_chunk_metadata,
    extract_sources_from_chunks,
    prepare_chunk_info,
    build_response_metadata,
    create_error_response,
    evaluate_component_health,
    evaluate_ollama_health,
    calculate_overall_health,
    create_rag_system,
)
from src.retrieval.categorization import CategoryType
from tests.fixtures.mock_rag_system import (
    MockDocumentExtractor,
    MockTextCleaner,
    MockDocumentChunker,
    MockEmbeddingModel,
    MockVectorStorage,
    MockRetriever,
    MockGenerationClient,
    MockResponseParser,
    MockPromptBuilder,
)


# Test Data Classes and Structures
class TestDataStructures:
    """Test pure data classes and their methods."""

    def test_rag_query_creation(self):
        """Test RAGQuery creation with various parameters."""
        query = RAGQuery(
            text="What is machine learning?",
            language="en",
            query_id="q123",
            user_id="u456",
            context_filters={"category": "technical"},
            max_results=10,
            metadata={"priority": "high"}
        )

        assert query.text == "What is machine learning?"
        assert query.language == "en"
        assert query.query_id == "q123"
        assert query.user_id == "u456"
        assert query.context_filters == {"category": "technical"}
        assert query.max_results == 10
        assert query.metadata == {"priority": "high"}

    def test_rag_query_minimal_creation(self):
        """Test RAGQuery with minimal required fields."""
        query = RAGQuery(text="Test query", language="hr")

        assert query.text == "Test query"
        assert query.language == "hr"
        assert query.query_id is None
        assert query.user_id is None
        assert query.context_filters is None
        assert query.max_results is None
        assert query.metadata is None

    def test_rag_response_properties(self):
        """Test RAGResponse and its properties."""
        response = RAGResponse(
            answer="Test answer",
            query="Test query",
            retrieved_chunks=[{"content": "chunk1"}],
            confidence=0.85,
            generation_time=1.5,
            retrieval_time=0.8,
            total_time=2.3,
            sources=["doc1.txt", "doc2.txt"],
            metadata={"category": "technical"}
        )

        assert response.answer == "Test answer"
        assert response.query == "Test query"
        assert response.confidence == 0.85
        assert response.has_high_confidence is True
        assert response.generation_time == 1.5
        assert response.retrieval_time == 0.8
        assert response.total_time == 2.3
        assert response.sources == ["doc1.txt", "doc2.txt"]

    def test_rag_response_low_confidence(self):
        """Test RAGResponse with low confidence."""
        response = RAGResponse(
            answer="Uncertain answer",
            query="Complex query",
            retrieved_chunks=[],
            confidence=0.6,
            generation_time=1.0,
            retrieval_time=0.5,
            total_time=1.5,
            sources=[],
            metadata={}
        )

        assert response.has_high_confidence is False
        assert response.confidence == 0.6

    def test_component_health(self):
        """Test ComponentHealth data class."""
        health = ComponentHealth(
            status="healthy",
            details="All systems operational",
            metadata={"uptime": 99.9}
        )

        assert health.status == "healthy"
        assert health.details == "All systems operational"
        assert health.metadata == {"uptime": 99.9}

    def test_system_health(self):
        """Test SystemHealth data class."""
        comp_health = ComponentHealth(status="healthy", details="OK")
        health = SystemHealth(
            system_status="healthy",
            components={"db": comp_health},
            metrics={"docs": 100},
            timestamp=1234567890.0,
            error=None
        )

        assert health.system_status == "healthy"
        assert "db" in health.components
        assert health.metrics == {"docs": 100}
        assert health.timestamp == 1234567890.0
        assert health.error is None

    def test_document_processing_result(self):
        """Test DocumentProcessingResult data class."""
        result = DocumentProcessingResult(
            processed_documents=10,
            failed_documents=2,
            total_chunks=150,
            processing_time=30.5,
            documents_per_second=0.33,
            errors=["Error 1", "Error 2"]
        )

        assert result.processed_documents == 10
        assert result.failed_documents == 2
        assert result.total_chunks == 150
        assert result.processing_time == 30.5
        assert result.documents_per_second == 0.33
        assert result.errors == ["Error 1", "Error 2"]


# Test Pure Functions
class TestPureFunctions:
    """Test all pure business logic functions."""

    def test_validate_language_code_valid(self):
        """Test language code validation with valid codes."""
        assert validate_language_code("hr") == "hr"
        assert validate_language_code("en") == "en"
        assert validate_language_code("multilingual") == "multilingual"
        assert validate_language_code("HR") == "hr"  # Case normalization
        assert validate_language_code(" en ") == "en"  # Whitespace handling

    def test_validate_language_code_invalid(self):
        """Test language code validation with invalid codes."""
        with pytest.raises(ValueError, match="Unsupported language"):
            validate_language_code("fr")

        with pytest.raises(ValueError, match="Language code must be a non-empty string"):
            validate_language_code("")

        with pytest.raises(ValueError, match="Language code must be a non-empty string"):
            validate_language_code(None)

    def test_create_language_collection_name(self):
        """Test language collection name creation."""
        assert create_language_collection_name("hr") == "croatian_documents"
        assert create_language_collection_name("en") == "english_documents"
        assert create_language_collection_name("multilingual") == "multilingual_documents"

    def test_create_language_collection_name_invalid(self):
        """Test collection name creation with invalid language."""
        with pytest.raises(Exception, match="Unsupported language"):
            create_language_collection_name("fr")

    @patch('src.pipeline.rag_system.Path')
    def test_validate_document_paths_valid(self, mock_path):
        """Test document path validation with valid paths."""
        # Setup mocks
        mock_path1 = Mock()
        mock_path1.exists.return_value = True
        mock_path1.is_file.return_value = True
        mock_path2 = Mock()
        mock_path2.exists.return_value = True
        mock_path2.is_file.return_value = True

        mock_path.side_effect = [mock_path1, mock_path2]

        result = validate_document_paths(["doc1.txt", "doc2.txt"])
        assert len(result) == 2
        assert result[0] == mock_path1
        assert result[1] == mock_path2

    def test_validate_document_paths_empty(self):
        """Test document path validation with empty list."""
        with pytest.raises(ValueError, match="Document paths list cannot be empty"):
            validate_document_paths([])

    def test_validate_document_paths_invalid_string(self):
        """Test document path validation with invalid string types."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_document_paths(["", "doc2.txt"])

    @patch('src.pipeline.rag_system.Path')
    def test_validate_document_paths_nonexistent(self, mock_path):
        """Test document path validation with nonexistent files."""
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj

        with pytest.raises(ValueError, match="Document path does not exist"):
            validate_document_paths(["nonexistent.txt"])

    @patch('src.pipeline.rag_system.Path')
    def test_validate_document_paths_not_file(self, mock_path):
        """Test document path validation with directories instead of files."""
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        mock_path.return_value = mock_path_obj

        with pytest.raises(ValueError, match="Document path is not a file"):
            validate_document_paths(["somedir/"])

    def test_validate_query_valid(self):
        """Test query validation with valid inputs."""
        query = RAGQuery(text="What is AI?", language="en", max_results=10)
        result = validate_query(query)

        assert result.text == "What is AI?"
        assert result.language == "en"
        assert result.max_results == 10

    def test_validate_query_sets_defaults(self):
        """Test query validation sets default values."""
        query = RAGQuery(text="Test query", language="hr")
        result = validate_query(query)

        assert result.max_results == 5  # Default value

    def test_validate_query_empty_text(self):
        """Test query validation with empty text."""
        query = RAGQuery(text="", language="en")
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            validate_query(query)

        query = RAGQuery(text="   ", language="en")
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            validate_query(query)

    def test_validate_query_no_language(self):
        """Test query validation without language."""
        query = RAGQuery(text="Test query", language="")
        with pytest.raises(ValueError, match="Query language must be specified"):
            validate_query(query)

    def test_validate_query_invalid_max_results(self):
        """Test query validation with invalid max_results."""
        query = RAGQuery(text="Test", language="en", max_results=0)
        with pytest.raises(ValueError, match="max_results must be positive"):
            validate_query(query)

        query = RAGQuery(text="Test", language="en", max_results=-1)
        with pytest.raises(ValueError, match="max_results must be positive"):
            validate_query(query)

    def test_calculate_processing_metrics_normal(self):
        """Test processing metrics calculation with normal values."""
        result = calculate_processing_metrics(10, 5.0, 100)

        assert result["documents_per_second"] == 2.0
        assert result["chunks_per_second"] == 20.0
        assert result["average_chunks_per_document"] == 10.0

    def test_calculate_processing_metrics_zero_time(self):
        """Test processing metrics calculation with zero time."""
        result = calculate_processing_metrics(10, 0.0, 100)

        assert result["documents_per_second"] == 0.0
        assert result["chunks_per_second"] == 0.0
        assert result["average_chunks_per_document"] == 0.0

    def test_calculate_processing_metrics_zero_docs(self):
        """Test processing metrics calculation with zero documents."""
        result = calculate_processing_metrics(0, 5.0, 0)

        assert result["documents_per_second"] == 0.0
        assert result["chunks_per_second"] == 0.0
        assert result["average_chunks_per_document"] == 0.0

    def test_create_chunk_metadata(self):
        """Test chunk metadata creation."""
        # Mock chunk object with explicit char_count
        chunk = Mock()
        chunk.chunk_id = "chunk_123"
        chunk.start_char = 0
        chunk.end_char = 100
        chunk.word_count = 20
        chunk.char_count = 25  # Explicit value
        chunk.content = "This is chunk content."

        result = create_chunk_metadata("doc.txt", 5, chunk, "hr", 1234567890.0)

        assert result["source"] == "doc.txt"
        assert result["chunk_index"] == 5
        assert result["language"] == "hr"
        assert result["chunk_id"] == "chunk_123"
        assert result["start_char"] == 0
        assert result["end_char"] == 100
        assert result["word_count"] == 20
        assert result["char_count"] == 25
        assert result["processing_timestamp"] == 1234567890.0

    def test_create_chunk_metadata_missing_attrs(self):
        """Test chunk metadata creation with missing attributes."""
        # Create a simple object with only content
        class MinimalChunk:
            def __init__(self):
                self.content = "Content"

        chunk = MinimalChunk()
        result = create_chunk_metadata("doc.txt", 0, chunk, "en", 1234567890.0)

        assert result["chunk_id"] == "doc.txt_0"
        assert result["start_char"] == 0
        assert result["end_char"] == 0
        assert result["word_count"] == 0
        assert result["char_count"] == 7

    def test_extract_sources_from_chunks_valid(self):
        """Test source extraction from chunks with valid data."""
        chunks = [
            {"metadata": {"source": "doc1.txt"}},
            {"metadata": {"source": "doc2.txt"}},
            {"metadata": {"source": "doc1.txt"}},  # Duplicate
            {"metadata": {"source": "doc3.txt"}},
        ]

        result = extract_sources_from_chunks(chunks)
        assert len(result) == 3
        assert "doc1.txt" in result
        assert "doc2.txt" in result
        assert "doc3.txt" in result

    def test_extract_sources_from_chunks_filters_unknown(self):
        """Test source extraction filters out Unknown sources."""
        chunks = [
            {"metadata": {"source": "doc1.txt"}},
            {"metadata": {"source": "Unknown"}},
            {"metadata": {"source": ""}},
            {"metadata": {"source": "doc2.txt"}},
        ]

        result = extract_sources_from_chunks(chunks)
        assert len(result) == 2
        assert "doc1.txt" in result
        assert "doc2.txt" in result
        assert "Unknown" not in result
        assert "" not in result

    def test_extract_sources_from_chunks_missing_metadata(self):
        """Test source extraction with missing metadata."""
        chunks = [{"content": "text", "score": 0.9}]  # Missing metadata

        with pytest.raises(ValueError, match="Chunk missing required 'metadata' field"):
            extract_sources_from_chunks(chunks)

    def test_extract_sources_from_chunks_missing_source(self):
        """Test source extraction with missing source in metadata."""
        chunks = [{"metadata": {"chunk_index": 0}}]  # Missing source

        with pytest.raises(ValueError, match="Chunk metadata missing required 'source' field"):
            extract_sources_from_chunks(chunks)

    def test_prepare_chunk_info_basic(self):
        """Test chunk info preparation with basic data."""
        chunk_result = {
            "content": "This is chunk content",
            "similarity_score": 0.85,
            "final_score": 0.9,
            "metadata": {"source": "doc1.txt", "chunk_index": 3}
        }

        result = prepare_chunk_info(chunk_result)

        assert result["content"] == "This is chunk content"
        assert result["similarity_score"] == 0.85
        assert result["final_score"] == 0.9
        assert result["source"] == "doc1.txt"
        assert result["chunk_index"] == 3
        assert "metadata" not in result  # Not in debug mode

    def test_prepare_chunk_info_debug_mode(self):
        """Test chunk info preparation with debug information."""
        chunk_result = {
            "content": "Debug content",
            "similarity_score": 0.75,
            "final_score": 0.8,
            "metadata": {
                "source": "debug.txt",
                "chunk_index": 1,
                "ranking_signals": {"keyword_match": 0.9}
            }
        }

        result = prepare_chunk_info(chunk_result, return_debug_info=True)

        assert result["content"] == "Debug content"
        assert result["metadata"] == chunk_result["metadata"]
        assert result["signals"] == {"keyword_match": 0.9}

    def test_prepare_chunk_info_missing_fields(self):
        """Test chunk info preparation with missing optional fields."""
        chunk_result = {
            "content": "Minimal content",
            "similarity_score": 0.6,
            "final_score": 0.65,
            "metadata": {}  # Missing source and chunk_index
        }

        result = prepare_chunk_info(chunk_result)

        assert result["source"] == "Unknown"
        assert result["chunk_index"] == 0

    def test_build_response_metadata_basic(self):
        """Test response metadata building with basic components."""
        query = RAGQuery(text="Test query", language="en", query_id="q1", user_id="u1")

        # Mock hierarchical results
        hierarchical_results = Mock()
        hierarchical_results.category = Mock()
        hierarchical_results.category.value = "technical"
        hierarchical_results.strategy_used = Mock()
        hierarchical_results.strategy_used.value = "hybrid"
        hierarchical_results.confidence = 0.8
        hierarchical_results.routing_metadata = {"route": "expert"}
        hierarchical_results.documents = [{"content": "doc1"}, {"content": "doc2"}]
        hierarchical_results.retrieval_time = 0.5

        # Mock generation response
        generation_response = Mock()
        generation_response.model = "gpt-4"
        generation_response.tokens_used = 150
        generation_response.confidence = 0.9

        # Mock parsed response
        parsed_response = Mock()
        parsed_response.language = "en"
        parsed_response.sources_mentioned = ["source1", "source2"]

        result = build_response_metadata(
            query, hierarchical_results, generation_response, parsed_response,
            0.5, 1.2, 1.7, return_debug_info=False
        )

        assert result["query_id"] == "q1"
        assert result["user_id"] == "u1"
        assert result["categorization"]["detected_category"] == "technical"
        assert result["categorization"]["strategy_used"] == "hybrid"
        assert result["retrieval"]["total_results"] == 2
        assert result["generation"]["model"] == "gpt-4"
        assert result["generation"]["tokens_used"] == 150
        assert result["parsing"]["language_detected"] == "en"
        assert result["performance"]["retrieval_time"] == 0.5
        assert result["performance"]["generation_time"] == 1.2
        assert result["performance"]["total_time"] == 1.7

    def test_build_response_metadata_debug_mode(self):
        """Test response metadata building with debug information."""
        query = RAGQuery(text="Debug query", language="hr")
        hierarchical_results = Mock()
        hierarchical_results.category = Mock()
        hierarchical_results.category.value = "general"
        hierarchical_results.strategy_used = Mock()
        hierarchical_results.strategy_used.value = "standard"
        hierarchical_results.documents = []
        generation_response = Mock()
        generation_response.text = "Generated text"
        generation_response.metadata = {"temp": 0.7}
        parsed_response = Mock()
        parsed_response.language = "hr"

        result = build_response_metadata(
            query, hierarchical_results, generation_response, parsed_response,
            0.3, 0.8, 1.1, return_debug_info=True,
            system_prompt="System prompt", user_prompt="User prompt"
        )

        assert "debug" in result
        assert result["debug"]["system_prompt"] == "System prompt"
        assert result["debug"]["user_prompt"] == "User prompt"
        assert result["debug"]["raw_generation"] == "Generated text"
        assert result["debug"]["generation_metadata"] == {"temp": 0.7}

    def test_create_error_response_english(self):
        """Test error response creation in English."""
        query = RAGQuery(text="Error test", language="en", query_id="q1")
        error = ValueError("Test error")
        start_time = time.time() - 1.0

        result = create_error_response(query, error, start_time)

        assert "error occurred while processing" in result.answer
        assert "Test error" in result.answer
        assert result.query == "Error test"
        assert result.confidence == 0.0
        assert result.generation_time == 0.0
        assert result.retrieval_time == 0.0
        assert result.total_time > 0.0  # Should be calculated from start_time
        assert result.sources == []
        assert result.metadata["error"] == "Test error"
        assert result.metadata["query_id"] == "q1"

    def test_create_error_response_croatian(self):
        """Test error response creation in Croatian."""
        query = RAGQuery(text="Test greške", language="hr", user_id="u1")
        error = RuntimeError("Runtime greška")
        start_time = time.time() - 0.5

        result = create_error_response(query, error, start_time)

        assert "greška pri obradi pitanja" in result.answer
        assert "Runtime greška" in result.answer
        assert result.metadata["user_id"] == "u1"

    def test_evaluate_component_health_all_healthy(self):
        """Test component health evaluation with all components healthy."""
        components = [Mock(), Mock(), Mock()]  # All non-None

        result = evaluate_component_health("test", components)

        assert result.status == "healthy"
        assert "All test components loaded" in result.details

    def test_evaluate_component_health_some_unhealthy(self):
        """Test component health evaluation with some unhealthy components."""
        components = [Mock(), None, Mock()]  # One is None

        result = evaluate_component_health("database", components, "Custom details")

        assert result.status == "unhealthy"
        assert result.details == "Custom details"

    def test_evaluate_ollama_health_not_initialized(self):
        """Test Ollama health evaluation when client not initialized."""
        result = evaluate_ollama_health(None, "test-model")

        assert result.status == "unhealthy"
        assert "not initialized" in result.details

    def test_evaluate_ollama_health_service_down(self):
        """Test Ollama health evaluation when service is down."""
        client = Mock()
        client.health_check.return_value = False

        result = evaluate_ollama_health(client, "test-model")

        assert result.status == "unhealthy"
        assert "not available" in result.details

    def test_evaluate_ollama_health_model_missing(self):
        """Test Ollama health evaluation when model is missing."""
        client = Mock()
        client.health_check.return_value = True
        client.get_available_models.return_value = ["other-model"]

        result = evaluate_ollama_health(client, "missing-model")

        assert result.status == "degraded"
        assert "missing-model: ❌" in result.details
        assert result.metadata["available_models"] == ["other-model"]

    def test_evaluate_ollama_health_all_good(self):
        """Test Ollama health evaluation when everything is healthy."""
        client = Mock()
        client.health_check.return_value = True
        client.get_available_models.return_value = ["target-model", "other-model"]

        result = evaluate_ollama_health(client, "target-model")

        assert result.status == "healthy"
        assert "target-model: ✅" in result.details
        assert "Ollama: ✅" in result.details

    def test_calculate_overall_health_all_healthy(self):
        """Test overall health calculation when all components are healthy."""
        components = {
            "db": ComponentHealth(status="healthy", details="OK"),
            "api": ComponentHealth(status="healthy", details="OK"),
        }

        result = calculate_overall_health(components)
        assert result == "healthy"

    def test_calculate_overall_health_mixed(self):
        """Test overall health calculation with mixed component statuses."""
        components = {
            "db": ComponentHealth(status="healthy", details="OK"),
            "api": ComponentHealth(status="unhealthy", details="Down"),
        }

        result = calculate_overall_health(components)
        assert result == "degraded"

    def test_calculate_overall_health_all_unhealthy(self):
        """Test overall health calculation when all components are unhealthy."""
        components = {
            "db": ComponentHealth(status="unhealthy", details="Down"),
            "api": ComponentHealth(status="unhealthy", details="Error"),
        }

        result = calculate_overall_health(components)
        assert result == "unhealthy"


# Test Factory Functions
class TestFactoryFunctions:
    """Test factory function for creating RAG system instances."""

    def create_mock_configs(self):
        """Create mock configuration objects."""
        embedding_config = Mock()
        embedding_config.model_name = "test-embedding"
        embedding_config.device = "cpu"

        ollama_config = Mock()
        ollama_config.model = "test-llm"
        ollama_config.timeout = 30

        processing_config = Mock()
        processing_config.enable_smart_chunking = True

        retrieval_config = Mock()
        retrieval_config.max_k = 10
        retrieval_config.similarity_threshold = 0.7

        return embedding_config, ollama_config, processing_config, retrieval_config

    def test_create_rag_system_factory(self):
        """Test RAG system factory function."""
        # Create mock components
        document_extractor = MockDocumentExtractor()
        text_cleaner = MockTextCleaner()
        document_chunker = MockDocumentChunker()
        embedding_model = MockEmbeddingModel()
        vector_storage = MockVectorStorage()
        search_engine = Mock()
        query_processor = Mock()
        retriever = MockRetriever()
        hierarchical_retriever = MockRetriever()
        ranker = Mock()
        generation_client = MockGenerationClient()
        response_parser = MockResponseParser()
        prompt_builder = MockPromptBuilder()

        # Create mock configs
        embedding_config, ollama_config, processing_config, retrieval_config = self.create_mock_configs()

        # Create RAG system via factory
        rag_system = create_rag_system(
            language="hr",
            document_extractor=document_extractor,
            text_cleaner=text_cleaner,
            document_chunker=document_chunker,
            embedding_model=embedding_model,
            vector_storage=vector_storage,
            search_engine=search_engine,
            query_processor=query_processor,
            retriever=retriever,
            hierarchical_retriever=hierarchical_retriever,
            ranker=ranker,
            generation_client=generation_client,
            response_parser=response_parser,
            prompt_builder=prompt_builder,
            embedding_config=embedding_config,
            ollama_config=ollama_config,
            processing_config=processing_config,
            retrieval_config=retrieval_config,
        )

        assert isinstance(rag_system, RAGSystem)
        assert rag_system.language == "hr"
        assert rag_system.embedding_config == embedding_config
        assert rag_system.ollama_config == ollama_config
        assert rag_system.processing_config == processing_config
        assert rag_system.retrieval_config == retrieval_config


# Test RAG System Integration
class TestRAGSystemIntegration:
    """Test the main RAGSystem class with comprehensive integration scenarios."""

    def setup_method(self):
        """Set up test fixtures for each test."""
        # Create mock components
        self.document_extractor = MockDocumentExtractor()
        self.text_cleaner = MockTextCleaner()
        self.document_chunker = MockDocumentChunker()
        self.embedding_model = MockEmbeddingModel()
        self.vector_storage = MockVectorStorage()
        self.search_engine = Mock()
        self.query_processor = Mock()
        self.retriever = MockRetriever()
        self.hierarchical_retriever = MockRetriever()
        self.ranker = Mock()
        self.generation_client = MockGenerationClient(available_models=["test-llm"])  # Match ollama_config.model
        self.response_parser = MockResponseParser()
        self.prompt_builder = MockPromptBuilder()

        # Fix async methods for close operations
        self.generation_client.close = AsyncMock()
        self.vector_storage.close = AsyncMock()

        # Fix mock methods for testing
        self.text_cleaner.setup_language_environment = Mock()
        self.embedding_model.load_model = Mock()
        self.vector_storage.create_collection = Mock()

        # Create mock configs
        self.embedding_config = Mock()
        self.embedding_config.model_name = "test-embedding"
        self.embedding_config.device = "cpu"

        self.ollama_config = Mock()
        self.ollama_config.model = "test-llm"
        self.ollama_config.timeout = 30

        self.processing_config = Mock()
        self.processing_config.enable_smart_chunking = True

        self.retrieval_config = Mock()
        self.retrieval_config.max_k = 10
        self.retrieval_config.similarity_threshold = 0.7

        # Create RAG system
        self.rag_system = RAGSystem(
            language="hr",
            document_extractor=self.document_extractor,
            text_cleaner=self.text_cleaner,
            document_chunker=self.document_chunker,
            embedding_model=self.embedding_model,
            vector_storage=self.vector_storage,
            search_engine=self.search_engine,
            query_processor=self.query_processor,
            retriever=self.retriever,
            hierarchical_retriever=self.hierarchical_retriever,
            ranker=self.ranker,
            generation_client=self.generation_client,
            response_parser=self.response_parser,
            prompt_builder=self.prompt_builder,
            embedding_config=self.embedding_config,
            ollama_config=self.ollama_config,
            processing_config=self.processing_config,
            retrieval_config=self.retrieval_config,
        )

    def test_rag_system_initialization(self):
        """Test RAG system initialization."""
        assert self.rag_system.language == "hr"
        assert self.rag_system._initialized is False
        assert self.rag_system._document_count == 0
        assert self.rag_system._query_count == 0
        assert self.rag_system.embedding_config == self.embedding_config
        assert self.rag_system.ollama_config == self.ollama_config

    def test_rag_system_invalid_language(self):
        """Test RAG system initialization with invalid language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            RAGSystem(
                language="invalid",
                document_extractor=self.document_extractor,
                text_cleaner=self.text_cleaner,
                document_chunker=self.document_chunker,
                embedding_model=self.embedding_model,
                vector_storage=self.vector_storage,
                search_engine=self.search_engine,
                query_processor=self.query_processor,
                retriever=self.retriever,
                hierarchical_retriever=self.hierarchical_retriever,
                ranker=self.ranker,
                generation_client=self.generation_client,
                response_parser=self.response_parser,
                prompt_builder=self.prompt_builder,
                embedding_config=self.embedding_config,
                ollama_config=self.ollama_config,
                processing_config=self.processing_config,
                retrieval_config=self.retrieval_config,
            )

    @pytest.mark.asyncio
    async def test_initialize_system(self):
        """Test system initialization process."""
        with patch('src.utils.config_loader.load_config') as mock_load_config, \
             patch('src.utils.config_loader.get_language_config') as mock_lang_config, \
             patch('src.utils.config_validator.ConfigValidator') as mock_validator:

            mock_load_config.return_value = {"test": "config"}
            mock_lang_config.return_value = {"lang": "hr"}

            await self.rag_system.initialize()

            assert self.rag_system._initialized is True
            # Verify component initialization was called
            assert self.text_cleaner.setup_language_environment.called
            assert self.embedding_model.load_model.called
            assert self.vector_storage.create_collection.called

    @pytest.mark.asyncio
    async def test_initialize_system_twice(self):
        """Test that initialize doesn't run twice."""
        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config'), \
             patch('src.utils.config_validator.ConfigValidator'):

            await self.rag_system.initialize()
            assert self.rag_system._initialized is True

            # Reset mocks and initialize again
            self.text_cleaner.setup_language_environment.reset_mock()
            await self.rag_system.initialize()

            # Should not call setup again
            assert not self.text_cleaner.setup_language_environment.called

    @pytest.mark.asyncio
    @patch('src.pipeline.rag_system.Path')
    async def test_add_documents_success(self, mock_path):
        """Test successful document addition."""
        # Setup path mocks
        mock_path1 = Mock()
        mock_path1.exists.return_value = True
        mock_path1.is_file.return_value = True
        mock_path1.name = "doc1.txt"
        mock_path.return_value = mock_path1

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config'), \
             patch('src.utils.config_validator.ConfigValidator'):

            result = await self.rag_system.add_documents(["doc1.txt"])

            assert result.processed_documents == 1
            assert result.failed_documents == 0
            assert result.total_chunks > 0
            assert result.processing_time > 0
            assert self.rag_system._document_count == 1

    @pytest.mark.asyncio
    @patch('src.pipeline.rag_system.Path')
    async def test_add_documents_extraction_failure(self, mock_path):
        """Test document addition with extraction failure."""
        # Setup path mock
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj

        # Make extractor return empty text
        self.document_extractor.extract_text = Mock(return_value="")

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config'), \
             patch('src.utils.config_validator.ConfigValidator'):

            result = await self.rag_system.add_documents(["empty_doc.txt"])

            assert result.processed_documents == 0
            assert result.failed_documents == 1
            assert len(result.errors) == 1
            assert "No text extracted" in result.errors[0]

    @pytest.mark.asyncio
    @patch('src.pipeline.rag_system.Path')
    async def test_add_documents_chunking_failure(self, mock_path):
        """Test document addition with chunking failure."""
        # Setup path mock
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj

        # Make chunker return empty list
        self.document_chunker.chunk_document = Mock(return_value=[])

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config'), \
             patch('src.utils.config_validator.ConfigValidator'):

            result = await self.rag_system.add_documents(["no_chunks.txt"])

            assert result.processed_documents == 0
            assert result.failed_documents == 1
            assert "No chunks created" in result.errors[0]

    @pytest.mark.asyncio
    async def test_query_success(self):
        """Test successful query execution."""
        query = RAGQuery(text="What is machine learning?", language="hr")

        # Mock language config to return minimal required structure
        mock_config = {
            'prompts': {
                'system_base': 'You are a helpful assistant.',
                'context_intro': 'Based on the following context:',
                'answer_intro': 'Answer:',
                'no_context_response': 'Not enough information to answer.',
                'question_answering_system': 'You are a helpful assistant that answers questions in Croatian.',
                'question_answering_user': 'Question: {query}\n\nProvide a detailed answer:',
                'question_answering_context': 'Contextual information:\n{context}\n\n',
            },
            'shared': {
                'response_language': 'croatian',
                'error_message': 'Error processing query'
            }
        }

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config', return_value=mock_config), \
             patch('src.utils.config_validator.ConfigValidator'):

            response = await self.rag_system.query(query)

            assert isinstance(response, RAGResponse)
            assert response.answer == "Mock generated response"
            assert response.query == "What is machine learning?"
            assert response.confidence == 0.8
            assert response.total_time > 0
            assert self.rag_system._query_count == 1

    @pytest.mark.asyncio
    async def test_query_with_sources(self):
        """Test query execution with source extraction."""
        query = RAGQuery(text="Test query", language="en")

        # Setup mock retriever to return chunks with sources
        mock_results = Mock()
        mock_results.documents = [
            {
                "content": "Content 1",
                "metadata": {"source": "doc1.txt"},
                "similarity_score": 0.9,
                "final_score": 0.9
            },
            {
                "content": "Content 2",
                "metadata": {"source": "doc2.txt"},
                "similarity_score": 0.8,
                "final_score": 0.8
            },
        ]
        mock_results.category = CategoryType.GENERAL  # Use real CategoryType enum
        self.hierarchical_retriever.retrieve = AsyncMock(return_value=mock_results)

        # Mock language config to return minimal required structure
        mock_config = {
            'prompts': {
                'system_base': 'You are a helpful assistant.',
                'context_intro': 'Based on the following context:',
                'answer_intro': 'Answer:',
                'no_context_response': 'Not enough information to answer.',
                'question_answering_system': 'You are a helpful assistant that answers questions.',
                'question_answering_user': 'Question: {query}\n\nProvide a detailed answer:',
                'question_answering_context': 'Contextual information:\n{context}\n\n',
            },
            'shared': {
                'response_language': 'english',
                'error_message': 'Error processing query'
            }
        }

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config', return_value=mock_config), \
             patch('src.utils.config_validator.ConfigValidator'):

            response = await self.rag_system.query(query, return_sources=True)

            assert len(response.sources) == 2
            assert "doc1.txt" in response.sources
            assert "doc2.txt" in response.sources

    @pytest.mark.asyncio
    async def test_query_error_handling(self):
        """Test query execution with error handling."""
        query = RAGQuery(text="Error query", language="hr")

        # Make hierarchical retriever raise exception
        self.hierarchical_retriever.retrieve = AsyncMock(side_effect=RuntimeError("Retrieval error"))

        # Mock language config to return Croatian error message
        mock_config = {
            'prompts': {
                'system_base': 'Ti si pomoćni asistent.',
                'context_intro': 'Na temelju sljedećeg konteksta:',
                'answer_intro': 'Odgovor:',
                'no_context_response': 'Nema dovoljno informacija u kontekstu.',
                'question_answering_system': 'Ti si pomoćni asistent koji odgovara na hrvatskom jeziku.',
                'question_answering_user': 'Pitanje: {query}\n\nDaj mi detaljan odgovor:',
                'question_answering_context': 'Kontekstualne informacije:\n{context}\n\n',
            },
            'shared': {
                'response_language': 'croatian',
                'error_message': 'Žao mi je, dogodila se greška pri obradi pitanja'
            }
        }

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config', return_value=mock_config), \
             patch('src.utils.config_validator.ConfigValidator'):

            response = await self.rag_system.query(query)

            assert "greška pri obradi pitanja" in response.answer
            assert "Retrieval error" in response.answer
            assert response.confidence == 0.0
            assert response.sources == []

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self):
        """Test health check when all components are healthy."""
        # Ensure generation client returns healthy status (must be async)
        self.generation_client.health_check = AsyncMock(return_value=True)
        self.generation_client.get_available_models = AsyncMock(return_value=["test-llm"])

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config'), \
             patch('src.utils.config_validator.ConfigValidator'):

            health = await self.rag_system.health_check()

            assert health.system_status == "healthy"
            assert "preprocessing" in health.components
            assert "vectordb" in health.components
            assert "retrieval" in health.components
            assert "generation" in health.components
            assert health.components["generation"].status == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check with degraded components."""
        # Make Ollama healthy but model missing
        self.generation_client.health_check = AsyncMock(return_value=True)
        self.generation_client.get_available_models = AsyncMock(return_value=["other-model"])

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config'), \
             patch('src.utils.config_validator.ConfigValidator'):

            health = await self.rag_system.health_check()

            assert health.system_status == "degraded"
            assert health.components["generation"].status == "degraded"

    @pytest.mark.asyncio
    async def test_health_check_error(self):
        """Test health check with system error."""
        # Make vector storage raise exception
        self.vector_storage.get_document_count = Mock(side_effect=RuntimeError("DB error"))

        with patch('src.utils.config_loader.load_config'), \
             patch('src.utils.config_loader.get_language_config'), \
             patch('src.utils.config_validator.ConfigValidator'):

            health = await self.rag_system.health_check()

            assert health.system_status == "error"
            assert health.error == "DB error"

    @pytest.mark.asyncio
    async def test_get_system_stats(self):
        """Test system statistics retrieval."""
        self.rag_system._initialized = True
        self.rag_system._document_count = 10
        self.rag_system._query_count = 25
        self.vector_storage.get_document_count = Mock(return_value=100)

        stats = await self.rag_system.get_system_stats()

        assert stats.system["language"] == "hr"
        assert stats.system["initialized"] is True
        assert stats.system["documents_processed"] == 10
        assert stats.system["queries_processed"] == 25
        assert stats.system["total_chunks"] == 100
        assert stats.models["embedding_model"] == "test-embedding"
        assert stats.models["llm_model"] == "test-llm"
        assert stats.performance["max_retrieval"] == 10

    @pytest.mark.asyncio
    async def test_get_system_stats_uninitialized(self):
        """Test system statistics when not initialized."""
        stats = await self.rag_system.get_system_stats()

        assert "error" in stats.system
        assert stats.system["error"] == "System not initialized"

    @pytest.mark.asyncio
    async def test_close_system(self):
        """Test system shutdown."""
        await self.rag_system.close()

        # Verify components were closed
        self.generation_client.close.assert_called_once()
        self.vector_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_system_with_none_components(self):
        """Test system shutdown with None components."""
        # Create system with None components
        rag_system = RAGSystem(
            language="en",
            document_extractor=self.document_extractor,
            text_cleaner=self.text_cleaner,
            document_chunker=self.document_chunker,
            embedding_model=self.embedding_model,
            vector_storage=None,  # None component
            search_engine=self.search_engine,
            query_processor=self.query_processor,
            retriever=self.retriever,
            hierarchical_retriever=self.hierarchical_retriever,
            ranker=self.ranker,
            generation_client=None,  # None component
            response_parser=self.response_parser,
            prompt_builder=self.prompt_builder,
            embedding_config=self.embedding_config,
            ollama_config=self.ollama_config,
            processing_config=self.processing_config,
            retrieval_config=self.retrieval_config,
        )

        # Should not raise exception
        await rag_system.close()