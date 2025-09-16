"""
Comprehensive tests for generation/ollama_client.py - Level 3 module
Tests all data classes, pure functions, and dependency injection patterns.
"""

import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass
from typing import Any

from src.generation.ollama_client import (
    # Data classes
    OllamaConfig,
    GenerationRequest,
    GenerationResponse,
    HttpResponse,
    HttpError,
    ConnectionError,

    # Protocols
    HttpClient,
    LanguageConfigProvider,

    # Pure functions
    build_complete_prompt,
    enhance_prompt_with_formality,
    create_ollama_request,
    parse_streaming_response,
    parse_non_streaming_response,
    calculate_generation_confidence,
    estimate_token_count,
    extract_model_list,
    check_model_availability,
    create_error_response,

    # Main class
    MultilingualOllamaClient,

    # Factory function
    create_ollama_client,
)


# ===== DATA CLASS TESTS =====

class TestOllamaConfig:
    """Test OllamaConfig data class."""

    def test_ollama_config_defaults(self):
        """Test OllamaConfig creation with defaults."""
        config = OllamaConfig()

        assert config.base_url == "http://localhost:11434"
        assert config.timeout == 120.0
        assert config.model == "llama3.1:8b"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.top_p == 0.9
        assert config.top_k == 64
        assert config.preserve_diacritics is True
        assert config.prefer_formal_style is True
        assert config.streaming is True
        assert config.confidence_threshold == 0.5
        assert len(config.fallback_models) == 3

    def test_ollama_config_custom_values(self):
        """Test OllamaConfig creation with custom values."""
        config = OllamaConfig(
            base_url="http://custom:8080",
            timeout=60.0,
            model="custom:7b",
            temperature=0.3,
            max_tokens=1000,
            streaming=False
        )

        assert config.base_url == "http://custom:8080"
        assert config.timeout == 60.0
        assert config.model == "custom:7b"
        assert config.temperature == 0.3
        assert config.max_tokens == 1000
        assert config.streaming is False


class TestGenerationRequest:
    """Test GenerationRequest data class."""

    def test_generation_request_creation(self):
        """Test GenerationRequest creation."""
        request = GenerationRequest(
            prompt="Test prompt",
            context=["context1", "context2"],
            query="What is AI?",
            query_type="factual",
            language="en",
            metadata={"test": "value"}
        )

        assert request.prompt == "Test prompt"
        assert request.context == ["context1", "context2"]
        assert request.query == "What is AI?"
        assert request.query_type == "factual"
        assert request.language == "en"
        assert request.metadata == {"test": "value"}

    def test_generation_request_defaults(self):
        """Test GenerationRequest with default values."""
        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Test query"
        )

        assert request.query_type == "general"
        assert request.language == "hr"
        assert request.metadata is None


class TestGenerationResponse:
    """Test GenerationResponse data class."""

    def test_generation_response_creation(self):
        """Test GenerationResponse creation."""
        response = GenerationResponse(
            text="Generated text",
            model="llama3.1:8b",
            tokens_used=50,
            generation_time=1.5,
            confidence=0.85,
            metadata={"test": "data"},
            language="hr"
        )

        assert response.text == "Generated text"
        assert response.model == "llama3.1:8b"
        assert response.tokens_used == 50
        assert response.generation_time == 1.5
        assert response.confidence == 0.85
        assert response.metadata == {"test": "data"}
        assert response.language == "hr"

    def test_generation_response_default_language(self):
        """Test GenerationResponse with default language."""
        response = GenerationResponse(
            text="Test",
            model="test:7b",
            tokens_used=10,
            generation_time=1.0,
            confidence=0.5,
            metadata={}
        )

        assert response.language == "hr"


class TestHttpResponse:
    """Test HttpResponse data class."""

    def test_http_response_creation(self):
        """Test HttpResponse creation."""
        response = HttpResponse(
            status_code=200,
            content=b'{"test": "data"}',
            json_data={"test": "data"}
        )

        assert response.status_code == 200
        assert response.content == b'{"test": "data"}'
        assert response.json_data == {"test": "data"}

    def test_http_response_json_from_content(self):
        """Test JSON parsing from content."""
        response = HttpResponse(
            status_code=200,
            content=b'{"key": "value"}'
        )

        json_data = response.json()
        assert json_data == {"key": "value"}

    def test_http_response_json_from_data(self):
        """Test JSON return from existing data."""
        json_data = {"existing": "data"}
        response = HttpResponse(
            status_code=200,
            content=b'{}',
            json_data=json_data
        )

        assert response.json() == json_data

    def test_http_response_raise_for_status_success(self):
        """Test raise_for_status with successful response."""
        response = HttpResponse(status_code=200, content=b'{}')
        response.raise_for_status()  # Should not raise

    def test_http_response_raise_for_status_error(self):
        """Test raise_for_status with error response."""
        response = HttpResponse(status_code=404, content=b'{}')

        with pytest.raises(HttpError) as exc_info:
            response.raise_for_status()

        assert exc_info.value.status_code == 404


class TestHttpError:
    """Test HttpError exception."""

    def test_http_error_creation(self):
        """Test HttpError creation."""
        error = HttpError("Not found", 404)

        assert str(error) == "Not found"
        assert error.status_code == 404


class TestConnectionError:
    """Test ConnectionError exception."""

    def test_connection_error_creation(self):
        """Test ConnectionError creation."""
        error = ConnectionError("Connection failed")

        assert str(error) == "Connection failed"


# ===== PURE FUNCTION TESTS =====

class TestBuildCompletePrompt:
    """Test build_complete_prompt function."""

    def test_build_complete_prompt_basic(self):
        """Test basic prompt building."""
        query = "What is AI?"
        context = ["AI is artificial intelligence", "Machine learning is a subset"]
        system_prompt = "You are a helpful assistant"

        result = build_complete_prompt(query, context, system_prompt)

        assert "System: You are a helpful assistant" in result
        assert "Context:" in result
        assert "AI is artificial intelligence" in result
        assert "Machine learning is a subset" in result
        assert "Question: What is AI?" in result
        assert "Answer:" in result

    def test_build_complete_prompt_no_context(self):
        """Test prompt building without context."""
        query = "What is AI?"
        system_prompt = "You are helpful"

        result = build_complete_prompt(query, None, system_prompt)

        assert "System: You are helpful" in result
        assert "Context:" not in result
        assert "Question: What is AI?" in result
        assert "Answer:" in result

    def test_build_complete_prompt_no_system(self):
        """Test prompt building without system prompt."""
        query = "What is AI?"
        context = ["AI context"]

        result = build_complete_prompt(query, context, None)

        assert "System:" not in result
        assert "Context:" in result
        assert "AI context" in result
        assert "Question: What is AI?" in result
        assert "Answer:" in result

    def test_build_complete_prompt_minimal(self):
        """Test minimal prompt building."""
        query = "Test query"

        result = build_complete_prompt(query)

        assert "Question: Test query" in result
        assert "Answer:" in result
        assert "System:" not in result
        assert "Context:" not in result


class TestEnhancePromptWithFormality:
    """Test enhance_prompt_with_formality function."""

    def test_enhance_prompt_with_formality_enabled(self):
        """Test prompt enhancement with formality enabled."""
        prompt = "What is AI?"
        formal_prompts = {"formal_instruction": "Please respond formally and professionally."}

        result = enhance_prompt_with_formality(prompt, formal_prompts, use_formal_style=True)

        assert "Please respond formally and professionally." in result
        assert "What is AI?" in result
        assert result.startswith("Please respond formally and professionally.")

    def test_enhance_prompt_with_formality_disabled(self):
        """Test prompt enhancement with formality disabled."""
        prompt = "What is AI?"
        formal_prompts = {"formal_instruction": "Please respond formally and professionally."}

        result = enhance_prompt_with_formality(prompt, formal_prompts, use_formal_style=False)

        assert result == prompt
        assert "Please respond formally and professionally." not in result

    def test_enhance_prompt_missing_formal_instruction(self):
        """Test prompt enhancement with missing formal instruction."""
        prompt = "What is AI?"
        formal_prompts = {}

        result = enhance_prompt_with_formality(prompt, formal_prompts, use_formal_style=True)

        assert result == prompt


class TestCreateOllamaRequest:
    """Test create_ollama_request function."""

    def test_create_ollama_request(self):
        """Test Ollama request creation."""
        prompt = "Test prompt"
        config = OllamaConfig(
            model="llama3.1:8b",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.8,
            top_k=50,
            streaming=True
        )

        result = create_ollama_request(prompt, config)

        assert result["model"] == "llama3.1:8b"
        assert result["prompt"] == "Test prompt"
        assert result["stream"] is True
        assert result["options"]["temperature"] == 0.5
        assert result["options"]["num_predict"] == 1000
        assert result["options"]["top_p"] == 0.8
        assert result["options"]["top_k"] == 50


class TestParseStreamingResponse:
    """Test parse_streaming_response function."""

    def test_parse_streaming_response_valid(self):
        """Test parsing valid streaming response."""
        json_lines = [
            '{"response": "Hello"}',
            '{"response": " world"}',
            '{"response": "!"}',
            '{"done": true}'
        ]

        result = parse_streaming_response(json_lines)

        assert result == "Hello world!"

    def test_parse_streaming_response_empty_lines(self):
        """Test parsing streaming response with empty lines."""
        json_lines = [
            '{"response": "Hello"}',
            '',
            '{"response": " world"}',
            '   ',
            '{"response": "!"}'
        ]

        result = parse_streaming_response(json_lines)

        assert result == "Hello world!"

    def test_parse_streaming_response_invalid_json(self):
        """Test parsing streaming response with invalid JSON."""
        json_lines = [
            '{"response": "Hello"}',
            'invalid json',
            '{"response": " world"}'
        ]

        result = parse_streaming_response(json_lines)

        assert result == "Hello world"

    def test_parse_streaming_response_missing_response(self):
        """Test parsing streaming response with missing response field."""
        json_lines = [
            '{"response": "Hello"}',
            '{"other": "data"}',
            '{"response": " world"}'
        ]

        result = parse_streaming_response(json_lines)

        assert result == "Hello world"

    def test_parse_streaming_response_empty(self):
        """Test parsing empty streaming response."""
        json_lines = []

        result = parse_streaming_response(json_lines)

        assert result == ""


class TestParseNonStreamingResponse:
    """Test parse_non_streaming_response function."""

    def test_parse_non_streaming_response_valid(self):
        """Test parsing valid non-streaming response."""
        response_data = {"response": "Generated text response"}

        result = parse_non_streaming_response(response_data)

        assert result == "Generated text response"

    def test_parse_non_streaming_response_missing_response(self):
        """Test parsing non-streaming response without response field."""
        response_data = {"other": "data"}

        with pytest.raises(ValueError, match="Missing 'response' in response data"):
            parse_non_streaming_response(response_data)


class TestCalculateGenerationConfidence:
    """Test calculate_generation_confidence function."""

    def test_calculate_generation_confidence_basic(self):
        """Test basic confidence calculation."""
        generated_text = "This is a comprehensive answer about artificial intelligence and machine learning."
        request = GenerationRequest(
            prompt="Tell me about AI",
            context=["AI is artificial intelligence", "Machine learning is important"],
            query="What is artificial intelligence?",
            language="en"
        )
        confidence_settings = {"error_phrases": ["error", "cannot", "unable"]}

        confidence = calculate_generation_confidence(generated_text, request, confidence_settings)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be above base

    def test_calculate_generation_confidence_short_text(self):
        """Test confidence calculation with short text."""
        generated_text = "Short"
        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Test query",
            language="en"
        )
        confidence_settings = {"error_phrases": []}

        confidence = calculate_generation_confidence(generated_text, request, confidence_settings)

        assert confidence < 0.5  # Should be penalized for shortness

    def test_calculate_generation_confidence_with_errors(self):
        """Test confidence calculation with error phrases."""
        generated_text = "I cannot provide an answer to this question because I cannot understand."
        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Test query",
            language="en"
        )
        confidence_settings = {"error_phrases": ["cannot", "unable", "error"]}

        with patch('src.generation.ollama_client.detect_language_content_with_config') as mock_detect:
            mock_detect.return_value = 0.0  # No language boost
            confidence = calculate_generation_confidence(generated_text, request, confidence_settings)

        assert confidence < 0.4  # Should be heavily penalized for multiple error phrases

    def test_calculate_generation_confidence_missing_error_phrases(self):
        """Test confidence calculation with missing error_phrases in settings."""
        generated_text = "Good answer"
        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Test query",
            language="en"
        )
        confidence_settings = {}  # Missing error_phrases

        with pytest.raises(ValueError, match="Missing 'error_phrases' in confidence_settings"):
            calculate_generation_confidence(generated_text, request, confidence_settings)


class TestEstimateTokenCount:
    """Test estimate_token_count function."""

    def test_estimate_token_count_basic(self):
        """Test basic token count estimation."""
        text = "This is a test sentence with multiple words"

        result = estimate_token_count(text)

        assert result == 8  # Number of words

    def test_estimate_token_count_empty(self):
        """Test token count estimation with empty text."""
        text = ""

        result = estimate_token_count(text)

        assert result == 0

    def test_estimate_token_count_none(self):
        """Test token count estimation with None."""
        text = None

        result = estimate_token_count(text)

        assert result == 0


class TestExtractModelList:
    """Test extract_model_list function."""

    def test_extract_model_list_valid(self):
        """Test extracting model list from valid API response."""
        api_response = {
            "models": [
                {"name": "llama3.1:8b", "size": 123456},
                {"name": "mistral:7b", "size": 654321},
                {"name": "qwen2.5:7b-instruct", "size": 987654}
            ]
        }

        result = extract_model_list(api_response)

        assert result == ["llama3.1:8b", "mistral:7b", "qwen2.5:7b-instruct"]

    def test_extract_model_list_missing_models(self):
        """Test extracting model list from response without models."""
        api_response = {"other": "data"}

        with pytest.raises(ValueError, match="Missing 'models' in API response"):
            extract_model_list(api_response)

    def test_extract_model_list_missing_name(self):
        """Test extracting model list with missing name field."""
        api_response = {
            "models": [
                {"name": "llama3.1:8b"},
                {"size": 123456},  # Missing name
                {"name": "mistral:7b"}
            ]
        }

        result = extract_model_list(api_response)

        assert result == ["llama3.1:8b", "mistral:7b"]  # Skip entry without name


class TestCheckModelAvailability:
    """Test check_model_availability function."""

    def test_check_model_availability_available(self):
        """Test checking availability of available model."""
        model_name = "llama3.1:8b"
        available_models = ["llama3.1:8b", "mistral:7b", "qwen2.5:7b-instruct"]

        result = check_model_availability(model_name, available_models)

        assert result is True

    def test_check_model_availability_unavailable(self):
        """Test checking availability of unavailable model."""
        model_name = "gpt-4"
        available_models = ["llama3.1:8b", "mistral:7b", "qwen2.5:7b-instruct"]

        result = check_model_availability(model_name, available_models)

        assert result is False


class TestCreateErrorResponse:
    """Test create_error_response function."""

    def test_create_error_response(self):
        """Test creating error response."""
        error_message = "Connection failed"
        model = "llama3.1:8b"
        start_time = time.time() - 1.0  # 1 second ago
        error_template = "Error occurred: {error}"
        query_type = "factual"

        response = create_error_response(error_message, model, start_time, error_template, query_type)

        assert response.text == "Error occurred: Connection failed"
        assert response.model == model
        assert response.tokens_used == 0
        assert response.generation_time >= 1.0
        assert response.confidence == 0.0
        assert response.metadata["error"] == error_message
        assert response.metadata["query_type"] == query_type
        assert response.language == "hr"


# ===== MAIN CLASS TESTS =====

class TestMultilingualOllamaClient:
    """Test MultilingualOllamaClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_http_client = AsyncMock(spec=HttpClient)
        self.mock_language_provider = Mock(spec=LanguageConfigProvider)
        self.mock_logger = Mock()

        # Configure mock language provider
        self.mock_language_provider.get_formal_prompts.return_value = {
            "formal_instruction": "Please respond formally."
        }
        self.mock_language_provider.get_confidence_settings.return_value = {
            "error_phrases": ["error", "cannot", "unable"]
        }

        self.config = OllamaConfig(
            base_url="http://test:11434",
            model="test:7b",
            streaming=False
        )

    def test_multilingual_ollama_client_initialization(self):
        """Test MultilingualOllamaClient initialization."""
        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider,
            logger=self.mock_logger
        )

        assert client.config == self.config
        assert client.http_client == self.mock_http_client
        assert client.language_config_provider == self.mock_language_provider
        assert client.logger == self.mock_logger

    @patch('src.generation.ollama_client.detect_language_content_with_config')
    @patch('src.generation.ollama_client.preserve_text_encoding')
    @patch('src.utils.config_loader.get_language_specific_config')
    async def test_generate_text_async_success(self, mock_get_config, mock_preserve_encoding, mock_detect_language):
        """Test successful text generation."""
        # Configure mocks
        self.mock_http_client.get.return_value = HttpResponse(200, b'{}')
        self.mock_http_client.post.return_value = HttpResponse(
            200, b'{}', {"response": "Generated response text"}
        )
        mock_preserve_encoding.return_value = "Generated response text"
        mock_detect_language.return_value = 0.8  # Mock language detection score
        mock_get_config.return_value = {
            "keywords": {"factual": ["what", "how", "why"]},
            "question_answering_system": "You are a helpful assistant."
        }

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider,
            logger=self.mock_logger
        )

        request = GenerationRequest(
            prompt="Test prompt",
            context=["Test context"],
            query="What is AI?",
            language="hr"
        )

        response = await client.generate_text_async(request)

        assert isinstance(response, GenerationResponse)
        assert response.text == "Generated response text"
        assert response.model == "test:7b"
        assert response.tokens_used > 0
        assert response.confidence >= 0.0
        assert response.language == "hr"

        # Verify HTTP calls
        self.mock_http_client.get.assert_called_once()
        self.mock_http_client.post.assert_called_once()

    async def test_generate_text_async_service_unavailable(self):
        """Test text generation when service is unavailable."""
        self.mock_http_client.get.return_value = HttpResponse(500, b'{}')

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider,
            logger=self.mock_logger
        )

        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Test query",
            language="hr"
        )

        with pytest.raises(ConnectionError, match="Ollama service is not available"):
            await client.generate_text_async(request)

    async def test_generate_text_async_streaming(self):
        """Test text generation with streaming."""
        streaming_config = OllamaConfig(streaming=True)

        self.mock_http_client.get.return_value = HttpResponse(200, b'{}')
        self.mock_http_client.stream_post.return_value = [
            '{"response": "Hello"}',
            '{"response": " world"}',
            '{"done": true}'
        ]

        client = MultilingualOllamaClient(
            config=streaming_config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider,
            logger=self.mock_logger
        )

        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Test query",
            language="hr"
        )

        with patch('src.utils.config_loader.get_language_specific_config') as mock_get_config, \
             patch('src.generation.ollama_client.detect_language_content_with_config') as mock_detect_language:
            mock_get_config.return_value = {
                "keywords": {"factual": ["what"]},
                "question_answering_system": "You are helpful."
            }
            mock_detect_language.return_value = 0.8

            response = await client.generate_text_async(request)

        assert response.text == "Hello world"
        self.mock_http_client.stream_post.assert_called_once()

    async def test_is_service_available_success(self):
        """Test service availability check success."""
        self.mock_http_client.get.return_value = HttpResponse(200, b'{}')

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        result = await client.is_service_available()

        assert result is True
        self.mock_http_client.get.assert_called_with(
            "http://test:11434/api/tags", timeout=5.0
        )

    async def test_is_service_available_failure(self):
        """Test service availability check failure."""
        self.mock_http_client.get.return_value = HttpResponse(500, b'{}')

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        result = await client.is_service_available()

        assert result is False

    async def test_get_available_models(self):
        """Test getting available models."""
        models_response = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "mistral:7b"}
            ]
        }
        self.mock_http_client.get.return_value = HttpResponse(
            200, b'{}', models_response
        )

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        models = await client.get_available_models()

        assert models == ["llama3.1:8b", "mistral:7b"]

    async def test_is_model_available_success(self):
        """Test model availability check success."""
        models_response = {
            "models": [
                {"name": "test:7b"},
                {"name": "other:7b"}
            ]
        }
        self.mock_http_client.get.return_value = HttpResponse(
            200, b'{}', models_response
        )

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        result = await client.is_model_available()  # Uses configured model

        assert result is True

    async def test_is_model_available_specific_model(self):
        """Test model availability check for specific model."""
        models_response = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "mistral:7b"}
            ]
        }
        self.mock_http_client.get.return_value = HttpResponse(
            200, b'{}', models_response
        )

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        result = await client.is_model_available("mistral:7b")

        assert result is True

    async def test_pull_model_already_available(self):
        """Test pulling model that's already available."""
        models_response = {
            "models": [{"name": "test:7b"}]
        }
        self.mock_http_client.get.return_value = HttpResponse(
            200, b'{}', models_response
        )

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        result = await client.pull_model()

        assert result is True
        # Should not call POST since model is already available
        self.mock_http_client.post.assert_not_called()

    async def test_pull_model_new_model(self):
        """Test pulling new model."""
        # First call returns no models, second call (after pull) returns the model
        models_response_empty = {"models": []}
        self.mock_http_client.get.return_value = HttpResponse(
            200, b'{}', models_response_empty
        )
        self.mock_http_client.post.return_value = HttpResponse(200, b'{}')

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider,
            logger=self.mock_logger
        )

        result = await client.pull_model("new:7b")

        assert result is True
        self.mock_http_client.post.assert_called_once()

    def test_generate_response_sync_wrapper(self):
        """Test synchronous wrapper for generate_text_async."""
        # Mock the async method
        async def mock_generate_text_async(request):
            return GenerationResponse(
                text="Sync response",
                model="test:7b",
                tokens_used=10,
                generation_time=1.0,
                confidence=0.8,
                metadata={},
                language="hr"
            )

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        # Patch the async method
        client.generate_text_async = mock_generate_text_async

        result = client.generate_response("Test prompt", language="hr")

        assert result == "Sync response"

    async def test_health_check(self):
        """Test health check method."""
        self.mock_http_client.get.return_value = HttpResponse(200, b'{}')

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        result = await client.health_check()

        assert result is True

    async def test_close(self):
        """Test client close method."""
        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        await client.close()

        self.mock_http_client.close.assert_called_once()

    async def test_async_context_manager(self):
        """Test async context manager."""
        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        async with client as ctx_client:
            assert ctx_client is client

        self.mock_http_client.close.assert_called_once()

    @patch('src.utils.config_loader.get_language_specific_config')
    def test_get_system_prompt_for_query_factual(self, mock_get_config):
        """Test system prompt selection for factual query."""
        mock_get_config.return_value = {
            "keywords": {
                "factual": ["what", "how", "why"],
                "tourism": ["visit", "travel"]
            },
            "factual_qa_system": "You are a factual assistant.",
            "tourism_system": "You are a tourism assistant."
        }

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="What is the capital?",  # Contains "what"
            language="hr"
        )

        result = client._get_system_prompt_for_query(request, {})

        assert result == "You are a factual assistant."

    @patch('src.utils.config_loader.get_language_specific_config')
    def test_get_system_prompt_for_query_default(self, mock_get_config):
        """Test system prompt selection with default fallback."""
        mock_get_config.return_value = {
            "keywords": {
                "factual": ["what", "how"],
                "tourism": ["visit", "travel"]
            },
            "question_answering_system": "You are a general assistant.",
            "factual_qa_system": "You are a factual assistant."
        }

        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Tell me about something",  # No matching keywords
            language="hr"
        )

        result = client._get_system_prompt_for_query(request, {})

        assert result == "You are a general assistant."

    def test_get_system_prompt_key_valid_category(self):
        """Test system prompt key mapping for valid category."""
        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        result = client._get_system_prompt_key("tourism")

        assert result == "tourism_system"

    def test_get_system_prompt_key_invalid_category(self):
        """Test system prompt key mapping for invalid category."""
        client = MultilingualOllamaClient(
            config=self.config,
            http_client=self.mock_http_client,
            language_config_provider=self.mock_language_provider
        )

        with pytest.raises(ValueError, match="Unsupported query category: invalid"):
            client._get_system_prompt_key("invalid")


# ===== FACTORY FUNCTION TESTS =====

class TestFactoryFunction:
    """Test factory function."""

    def test_create_ollama_client_defaults(self):
        """Test creating client with default configuration."""
        client = create_ollama_client()

        assert isinstance(client, MultilingualOllamaClient)
        assert isinstance(client.config, OllamaConfig)
        assert client.config.model == "llama3.1:8b"  # Default value

    def test_create_ollama_client_custom_config(self):
        """Test creating client with custom configuration."""
        config = OllamaConfig(model="custom:7b", timeout=60.0)
        mock_http_client = AsyncMock(spec=HttpClient)
        mock_language_provider = Mock(spec=LanguageConfigProvider)
        mock_logger = Mock()

        client = create_ollama_client(
            config=config,
            http_client=mock_http_client,
            language_config_provider=mock_language_provider,
            logger=mock_logger
        )

        assert isinstance(client, MultilingualOllamaClient)
        assert client.config.model == "custom:7b"
        assert client.config.timeout == 60.0
        assert client.http_client == mock_http_client
        assert client.language_config_provider == mock_language_provider
        assert client.logger == mock_logger