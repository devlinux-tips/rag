"""
Tests for Ollama client functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import json
import asyncio
from dataclasses import dataclass

from src.generation.ollama_client import (
    MultilingualOllamaClient,
    GenerationRequest,
    GenerationResponse,
    HttpResponse,
    HttpError,
    ConnectionError,
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
    create_ollama_client,
)
from src.utils.config_models import OllamaConfig


class TestDataClasses:
    """Test data classes for generation."""

    def test_generation_request(self):
        """Test GenerationRequest creation."""
        request = GenerationRequest(
            prompt="Test prompt",
            context=["context1", "context2"],
            query="What is RAG?",
            query_type="general",
            language="en",
            metadata={"key": "value"}
        )

        assert request.prompt == "Test prompt"
        assert len(request.context) == 2
        assert request.query == "What is RAG?"
        assert request.language == "en"

    def test_generation_response(self):
        """Test GenerationResponse creation."""
        response = GenerationResponse(
            text="Generated text",
            model="llama2",
            tokens_used=50,
            generation_time=1.5,
            confidence=0.95,
            metadata={"key": "value"},
            language="en"
        )

        assert response.text == "Generated text"
        assert response.model == "llama2"
        assert response.tokens_used == 50
        assert response.confidence == 0.95

    def test_http_response(self):
        """Test HttpResponse creation and methods."""
        response = HttpResponse(
            status_code=200,
            content=b'{"test": "data"}',
            json_data=None
        )

        assert response.status_code == 200
        assert response.json() == {"test": "data"}

    def test_http_response_with_json_data(self):
        """Test HttpResponse with pre-parsed JSON."""
        response = HttpResponse(
            status_code=200,
            content=b'',
            json_data={"test": "data"}
        )

        assert response.json() == {"test": "data"}

    def test_http_response_raise_for_status(self):
        """Test HttpResponse.raise_for_status."""
        # Success response
        response = HttpResponse(status_code=200, content=b'')
        response.raise_for_status()  # Should not raise

        # Error response
        error_response = HttpResponse(status_code=404, content=b'')
        with pytest.raises(HttpError) as exc_info:
            error_response.raise_for_status()
        assert exc_info.value.status_code == 404

    def test_http_error(self):
        """Test HttpError creation."""
        error = HttpError("Not found", 404)
        assert str(error) == "Not found"
        assert error.status_code == 404

    def test_connection_error(self):
        """Test ConnectionError creation."""
        error = ConnectionError("Connection refused")
        assert str(error) == "Connection refused"


class TestPromptFunctions:
    """Test prompt building functions."""

    def test_build_complete_prompt(self):
        """Test building complete prompt."""
        result = build_complete_prompt(
            query="What is RAG?",
            context=["Context about RAG"]
        )

        assert "What is RAG?" in result
        assert "Context about RAG" in result

    def test_build_complete_prompt_with_system_prompt(self):
        """Test building prompt with system prompt."""
        result = build_complete_prompt(
            query="Explain AI",
            context=["AI context"],
            system_prompt="Be concise"
        )

        assert "Explain AI" in result
        assert "AI context" in result

    def test_enhance_prompt_with_formality(self):
        """Test prompt enhancement."""
        formal_prompts = {"formal": "Please provide a formal response."}
        result = enhance_prompt_with_formality(
            prompt="Tell me about AI",
            formal_prompts=formal_prompts,
            use_formal_style=True
        )

        assert "Tell me about AI" in result

    def test_enhance_prompt_with_formality_informal(self):
        """Test informal prompt enhancement."""
        formal_prompts = {"informal": "Keep it casual."}
        result = enhance_prompt_with_formality(
            prompt="What's up with AI?",
            formal_prompts=formal_prompts,
            use_formal_style=False
        )

        assert result  # Should return something


class TestOllamaRequest:
    """Test Ollama request creation."""

    @pytest.fixture
    def mock_ollama_config(self):
        """Create mock OllamaConfig."""
        config = Mock(spec=OllamaConfig)
        config.model = "llama2"
        config.temperature = 0.7
        config.max_tokens = 100
        config.stream = False
        config.ollama_num_thread = 4
        config.num_predict = 100
        config.top_p = 0.9
        config.top_k = 40
        return config

    def test_create_ollama_request(self, mock_ollama_config):
        """Test creating Ollama request."""
        result = create_ollama_request(
            prompt="Test prompt",
            config=mock_ollama_config
        )

        assert result["prompt"] == "Test prompt"
        assert result["model"] == "llama2"
        assert "options" in result
        assert result["options"]["temperature"] == 0.7

    def test_create_ollama_request_with_stream(self, mock_ollama_config):
        """Test creating streaming request."""
        mock_ollama_config.stream = True
        result = create_ollama_request(
            prompt="Test",
            config=mock_ollama_config
        )

        assert result["stream"] is True


class TestResponseParsing:
    """Test response parsing functions."""

    def test_parse_streaming_response(self):
        """Test parsing streaming response."""
        json_lines = [
            '{"response": "Hello", "done": false}',
            '{"response": " World", "done": false}',
            '{"response": "", "done": true}'
        ]
        result = parse_streaming_response(json_lines)

        assert "Hello World" in result or result == "Hello World"

    def test_parse_streaming_response_invalid(self):
        """Test parsing invalid streaming response."""
        result = parse_streaming_response(["invalid json"])
        assert result == "" or result == "invalid json"  # May return empty or raw data

    def test_parse_non_streaming_response(self):
        """Test parsing non-streaming response."""
        response = {
            "response": "Generated text",
            "model": "llama2",
            "total_duration": 2000000000
        }
        result = parse_non_streaming_response(response)

        assert result == "Generated text"

    def test_parse_non_streaming_response_minimal(self):
        """Test parsing minimal response."""
        response = {"response": "Text"}
        result = parse_non_streaming_response(response)

        assert result == "Text"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_generation_confidence(self):
        """Test confidence calculation."""
        request = Mock(spec=GenerationRequest)
        request.query = "Test query"
        request.context = ["some context"]
        request.language = "en"
        confidence_settings = {
            "base_confidence": 0.5,
            "temperature_weight": 0.2,
            "error_phrases": ["I cannot", "I don't know", "Sorry"]
        }

        confidence = calculate_generation_confidence(
            generated_text="This is a good response.",
            request=request,
            confidence_settings=confidence_settings
        )

        assert 0.0 <= confidence <= 1.0

    def test_calculate_generation_confidence_empty(self):
        """Test confidence for empty response."""
        request = Mock(spec=GenerationRequest)
        request.query = "Test query"
        request.context = []
        request.language = "en"
        confidence_settings = {
            "error_phrases": ["I cannot", "I don't know", "Sorry"]
        }

        confidence = calculate_generation_confidence(
            generated_text="",
            request=request,
            confidence_settings=confidence_settings
        )

        # Empty text: base 0.5 - 0.3 (short text penalty) = 0.2
        assert confidence == 0.2

    def test_estimate_token_count(self):
        """Test token estimation."""
        count = estimate_token_count("This is a test.")
        assert count > 0

    def test_estimate_token_count_empty(self):
        """Test token estimation for empty string."""
        count = estimate_token_count("")
        assert count == 0

    def test_extract_model_list(self):
        """Test model list extraction."""
        response = {
            "models": [
                {"name": "llama2"},
                {"name": "mistral"}
            ]
        }

        models = extract_model_list(response)
        assert len(models) == 2
        assert "llama2" in models

    def test_extract_model_list_missing_key(self):
        """Test model extraction with missing key."""
        # Function should raise ValueError for missing 'models' key (fail-fast pattern)
        with pytest.raises(ValueError, match="Missing 'models' in API response"):
            extract_model_list({})

    def test_check_model_availability(self):
        """Test model availability check."""
        models = ["llama2", "mistral"]

        assert check_model_availability("llama2", models) is True
        assert check_model_availability("gpt4", models) is False

    def test_create_error_response(self):
        """Test error response creation."""
        response = create_error_response(
            error_message="Test error",
            model="llama2",
            start_time=0.0,
            error_template="Error: {error}",
            query_type="general"
        )

        assert "Test error" in response.text
        assert response.model == "llama2"
        assert response.confidence == 0.0


class TestMultilingualOllamaClient:
    """Test MultilingualOllamaClient."""

    @pytest.fixture
    def mock_config(self):
        """Create mock Ollama config."""
        config = Mock(spec=OllamaConfig)
        config.base_url = "http://localhost:11434"
        config.model = "llama2"
        config.temperature = 0.7
        config.max_tokens = 2000
        config.timeout = 30
        config.stream = False
        config.ollama_num_thread = 4
        config.num_predict = 2000
        config.top_p = 0.9
        config.top_k = 40
        return config

    @pytest.fixture
    def mock_http_client(self):
        """Create mock HTTP client."""
        client = Mock()
        client.get = AsyncMock()
        client.post = AsyncMock()
        client.stream_post = AsyncMock()
        return client

    @pytest.fixture
    def ollama_client(self, mock_config, mock_http_client):
        """Create Ollama client with mocks."""
        return MultilingualOllamaClient(
            config=mock_config,
            http_client=mock_http_client
        )

    def test_client_init(self, mock_config):
        """Test client initialization."""
        client = MultilingualOllamaClient(config=mock_config)
        assert client.config == mock_config

    @pytest.mark.asyncio
    async def test_generate_text_async(self, ollama_client, mock_http_client):
        """Test async text generation."""
        # Mock service availability check to return True
        mock_http_client.get.return_value = HttpResponse(
            status_code=200,
            content=b'{"models": []}',
            json_data={"models": []}
        )

        mock_http_client.post.return_value = HttpResponse(
            status_code=200,
            content=b'{"response": "Generated text", "model": "llama2"}',
            json_data={"response": "Generated text", "model": "llama2"}
        )

        request = GenerationRequest(
            prompt="Test",
            context=["context"],
            query="Query",
            language="en"
        )

        response = await ollama_client.generate_text_async(request)

        assert response.text == "Generated text"
        assert response.model == "llama2"

    @pytest.mark.asyncio
    async def test_generate_text_async_error(self, ollama_client, mock_http_client):
        """Test async generation with error."""
        # Mock service as available
        mock_http_client.get.return_value = HttpResponse(
            status_code=200,
            content=b'{"models": []}',
            json_data={"models": []}
        )

        # But POST fails
        mock_http_client.post.side_effect = ConnectionError("Connection failed")

        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Query",
            language="en"
        )

        # The current implementation raises exceptions, not error responses
        with pytest.raises(ConnectionError) as exc_info:
            await ollama_client.generate_text_async(request)

        assert str(exc_info.value) == "Connection failed"

    @pytest.mark.asyncio
    async def test_is_service_available(self, ollama_client, mock_http_client):
        """Test service availability check."""
        mock_http_client.get.return_value = HttpResponse(
            status_code=200,
            content=b'{"status": "ok"}',
            json_data={"status": "ok"}
        )

        available = await ollama_client.is_service_available()
        assert available is True

    @pytest.mark.asyncio
    async def test_is_service_unavailable(self, ollama_client, mock_http_client):
        """Test service unavailability."""
        mock_http_client.get.side_effect = ConnectionError("Connection refused")

        available = await ollama_client.is_service_available()
        assert available is False

    @pytest.mark.asyncio
    async def test_get_available_models(self, ollama_client, mock_http_client):
        """Test getting available models."""
        mock_http_client.get.return_value = HttpResponse(
            status_code=200,
            content=b'{"models": [{"name": "llama2"}, {"name": "mistral"}]}',
            json_data={"models": [{"name": "llama2"}, {"name": "mistral"}]}
        )

        models = await ollama_client.get_available_models()
        assert len(models) == 2
        assert "llama2" in models

    @pytest.mark.asyncio
    async def test_is_model_available(self, ollama_client, mock_http_client):
        """Test model availability check."""
        mock_http_client.get.return_value = HttpResponse(
            status_code=200,
            content=b'{"models": [{"name": "llama2"}]}',
            json_data={"models": [{"name": "llama2"}]}
        )

        available = await ollama_client.is_model_available("llama2")
        assert available is True

    @pytest.mark.asyncio
    async def test_pull_model(self, ollama_client, mock_http_client):
        """Test pulling model."""
        # Mock service as available
        mock_http_client.get.return_value = HttpResponse(
            status_code=200,
            content=b'{"models": []}',
            json_data={"models": []}
        )

        mock_http_client.post.return_value = HttpResponse(
            status_code=200,
            content=b'{"status": "success"}',
            json_data={"status": "success"}
        )

        success = await ollama_client.pull_model("llama2")
        assert success is True

    @pytest.mark.asyncio
    async def test_health_check(self, ollama_client, mock_http_client):
        """Test health check."""
        mock_http_client.get.return_value = HttpResponse(
            status_code=200,
            content=b'{"status": "ok"}',
            json_data={"status": "ok"}
        )

        health = await ollama_client.health_check()
        assert health is True  # health_check returns a boolean, not dict

    def test_generate_response_sync(self, ollama_client, mock_http_client):
        """Test synchronous generate_response."""
        # Mock service as available
        mock_http_client.get.return_value = HttpResponse(
            status_code=200,
            content=b'{"models": []}',
            json_data={"models": []}
        )

        mock_http_client.post.return_value = HttpResponse(
            status_code=200,
            content=b'{"response": "Sync response", "model": "llama2"}',
            json_data={"response": "Sync response", "model": "llama2"}
        )

        # The sync method should handle async internally
        # Note: generate_response takes prompt, not query
        response = ollama_client.generate_response(
            prompt="Test",
            context=["context"],
            language="en"
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_close(self, ollama_client, mock_http_client):
        """Test closing client."""
        # Mock async close method
        mock_http_client.close = AsyncMock()

        await ollama_client.close()

        # Verify close was called on http_client
        mock_http_client.close.assert_called_once()


class TestOllamaClientFactory:
    """Test create_ollama_client factory."""

    @patch('src.utils.config_loader.load_config')
    def test_create_ollama_client_default(self, mock_load_config):
        """Test creating client with defaults."""
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        mock_ollama_config = Mock(spec=OllamaConfig)
        mock_ollama_config.model = "llama2"
        mock_ollama_config.stream = False
        mock_ollama_config.base_url = "http://localhost:11434"
        mock_ollama_config.timeout = 30
        mock_ollama_config.top_p = 0.9
        mock_ollama_config.top_k = 40

        with patch.object(OllamaConfig, 'from_validated_config', return_value=mock_ollama_config):
            client = create_ollama_client()
            assert client is not None
            assert isinstance(client, MultilingualOllamaClient)

    def test_create_ollama_client_with_config(self):
        """Test creating client with config."""
        config = Mock(spec=OllamaConfig)
        config.model = "mistral"
        config.stream = False
        config.base_url = "http://localhost:11434"
        config.timeout = 30
        config.top_p = 0.9
        config.top_k = 40

        client = create_ollama_client(config=config)
        assert client is not None
        assert client.config == config


class TestStreamingResponse:
    """Test streaming response handling."""

    @pytest.fixture
    def ollama_client(self):
        """Create client for streaming tests."""
        config = Mock(spec=OllamaConfig)
        config.base_url = "http://localhost:11434"
        config.model = "llama2"
        config.stream = True
        config.temperature = 0.7
        config.max_tokens = 2000
        config.timeout = 30
        config.ollama_num_thread = 4
        config.num_predict = 2000
        config.top_p = 0.9
        config.top_k = 40

        return MultilingualOllamaClient(config=config)

    @pytest.mark.asyncio
    async def test_streaming_response(self, ollama_client):
        """Test handling streaming responses."""
        mock_http = Mock()

        # Mock service availability check
        mock_http.get = AsyncMock(return_value=HttpResponse(
            status_code=200,
            content=b'{"status": "ok"}',
            json_data={"status": "ok"}
        ))

        # Mock streaming response
        async def mock_stream(url, request, timeout):
            return [
                '{"response": "Part 1", "done": false}',
                '{"response": " Part 2", "done": false}',
                '{"response": "", "done": true, "total_duration": 1000000000}'
            ]

        mock_http.stream_post = mock_stream
        ollama_client.http_client = mock_http

        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Query",
            language="en"
        )

        response = await ollama_client.generate_text_async(request)
        assert response is not None
        assert "Part 1 Part 2" in response.text


class TestErrorHandling:
    """Test error handling."""

    def test_http_error_handling(self):
        """Test HTTP error handling."""
        response = HttpResponse(status_code=500, content=b'Server error')

        with pytest.raises(HttpError) as exc_info:
            response.raise_for_status()

        assert exc_info.value.status_code == 500

    def test_connection_error_handling(self):
        """Test connection error."""
        error = ConnectionError("Connection timeout")
        assert str(error) == "Connection timeout"

    @pytest.mark.asyncio
    async def test_client_connection_error(self):
        """Test client handling connection errors."""
        config = Mock(spec=OllamaConfig)
        config.base_url = "http://localhost:11434"
        config.model = "llama2"
        config.timeout = 30
        config.stream = False
        config.top_p = 0.9
        config.top_k = 40
        config.temperature = 0.7
        config.num_predict = 2000

        mock_http = Mock()
        # Mock service as available first
        mock_http.get = AsyncMock(return_value=HttpResponse(
            status_code=200,
            content=b'{"status": "ok"}',
            json_data={"status": "ok"}
        ))
        # But POST fails
        mock_http.post = AsyncMock(side_effect=ConnectionError("Connection refused"))

        client = MultilingualOllamaClient(config=config, http_client=mock_http)

        request = GenerationRequest(
            prompt="Test",
            context=[],
            query="Query"
        )

        # The client raises exceptions for connection errors
        with pytest.raises(ConnectionError) as exc_info:
            await client.generate_text_async(request)

        assert "Connection refused" in str(exc_info.value)