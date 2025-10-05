"""
Test suite for generation.http_clients module.

Tests HTTP client implementations for dependency injection,
including async operations, exception handling, and mocking capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Any

from src.generation.http_clients import (
    AsyncHttpxClient,
    FallbackAsyncClient,
)
from tests.conftest import (
    MockHttpClient,
)

# Import the exceptions and HttpResponse from ollama_client
from src.generation.ollama_client import ConnectionError, HttpError, HttpResponse


class TestAsyncHttpxClient:
    """Test cases for AsyncHttpxClient."""

    @pytest.fixture
    def client(self):
        """Create AsyncHttpxClient instance for testing."""
        return AsyncHttpxClient()

    def test_async_httpx_client_initialization(self, client):
        """Test AsyncHttpxClient initializes correctly."""
        assert client._client is None

    def test_ensure_client_creates_httpx_client(self, client):
        """Test _ensure_client creates httpx.AsyncClient on first call."""
        with patch('src.generation.http_clients.httpx.AsyncClient') as mock_httpx:
            mock_instance = Mock()
            mock_httpx.return_value = mock_instance

            result = client._ensure_client()

            mock_httpx.assert_called_once()
            assert client._client is mock_instance
            assert result is mock_instance

    def test_ensure_client_reuses_existing_client(self, client):
        """Test _ensure_client reuses existing client."""
        with patch('src.generation.http_clients.httpx.AsyncClient') as mock_httpx:
            mock_instance = Mock()
            mock_httpx.return_value = mock_instance

            # First call creates client
            result1 = client._ensure_client()
            # Second call reuses client
            result2 = client._ensure_client()

            mock_httpx.assert_called_once()  # Only called once
            assert result1 is result2
            assert result1 is mock_instance

    @pytest.mark.asyncio
    async def test_get_success(self, client):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"test": "data"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"test": "data"}

        with patch.object(client, '_ensure_client') as mock_ensure:
            mock_httpx_client = AsyncMock()
            mock_httpx_client.get.return_value = mock_response
            mock_ensure.return_value = mock_httpx_client

            result = await client.get("http://test.com", timeout=30.0)

            mock_httpx_client.get.assert_called_once_with("http://test.com", timeout=30.0)
            assert isinstance(result, HttpResponse)
            assert result.status_code == 200
            assert result.content == b'{"test": "data"}'
            assert result.json_data == {"test": "data"}

    @pytest.mark.asyncio
    async def test_get_without_json_content_type(self, client):
        """Test GET request without JSON content-type."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"plain text"
        mock_response.headers = {"content-type": "text/plain"}

        with patch.object(client, '_ensure_client') as mock_ensure:
            mock_httpx_client = AsyncMock()
            mock_httpx_client.get.return_value = mock_response
            mock_ensure.return_value = mock_httpx_client

            result = await client.get("http://test.com")

            assert result.json_data is None
            assert result.content == b"plain text"

    @pytest.mark.asyncio
    async def test_get_connection_error(self, client):
        """Test GET request with connection error."""
        import httpx

        with patch.object(client, '_ensure_client') as mock_ensure:
            mock_httpx_client = AsyncMock()
            mock_httpx_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_ensure.return_value = mock_httpx_client

            with pytest.raises(ConnectionError, match="Connection failed"):
                await client.get("http://test.com")

    @pytest.mark.asyncio
    async def test_get_http_status_error(self, client):
        """Test GET request with HTTP status error."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 404

        with patch.object(client, '_ensure_client') as mock_ensure:
            mock_httpx_client = AsyncMock()
            mock_httpx_client.get.side_effect = httpx.HTTPStatusError(
                "Not found", request=Mock(), response=mock_response
            )
            mock_ensure.return_value = mock_httpx_client

            with pytest.raises(HttpError, match="HTTP error"):
                await client.get("http://test.com")

    @pytest.mark.asyncio
    async def test_get_timeout_error(self, client):
        """Test GET request with timeout error."""
        import httpx

        with patch.object(client, '_ensure_client') as mock_ensure:
            mock_httpx_client = AsyncMock()
            mock_httpx_client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_ensure.return_value = mock_httpx_client

            with pytest.raises(TimeoutError, match="Request timeout"):
                await client.get("http://test.com")

    @pytest.mark.asyncio
    async def test_post_success(self, client):
        """Test successful POST request."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.content = b'{"created": "success"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"created": "success"}

        json_data = {"input": "test"}

        with patch.object(client, '_ensure_client') as mock_ensure:
            mock_httpx_client = AsyncMock()
            mock_httpx_client.post.return_value = mock_response
            mock_ensure.return_value = mock_httpx_client

            result = await client.post("http://test.com/api", json_data, timeout=60.0)

            mock_httpx_client.post.assert_called_once_with(
                "http://test.com/api", json=json_data, timeout=60.0, headers=None
            )
            assert result.status_code == 201
            assert result.json_data == {"created": "success"}

    @pytest.mark.asyncio
    async def test_post_exception_handling(self, client):
        """Test POST request exception handling."""
        import httpx

        with patch.object(client, '_ensure_client') as mock_ensure:
            mock_httpx_client = AsyncMock()
            mock_httpx_client.post.side_effect = httpx.ConnectError("No connection")
            mock_ensure.return_value = mock_httpx_client

            with pytest.raises(ConnectionError, match="Connection failed"):
                await client.post("http://test.com", {"data": "test"})

    @pytest.mark.asyncio
    async def test_stream_post_success(self, client):
        """Test successful streaming POST request."""
        # Make aiter_lines an async generator
        async def mock_aiter_lines():
            lines = [
                '{"response": "Hello", "done": false}',
                '{"response": " World", "done": true}',
                '',  # Empty line should be filtered
            ]
            for line in lines:
                yield line

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.aiter_lines = Mock(return_value=mock_aiter_lines())

        json_data = {"prompt": "test"}

        with patch.object(client, '_ensure_client') as mock_ensure:
            # Create async context manager that properly handles httpx.stream
            class MockStreamContextManager:
                async def __aenter__(self):
                    return mock_response
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None

            mock_httpx_client = Mock()
            mock_httpx_client.stream = Mock(return_value=MockStreamContextManager())
            mock_ensure.return_value = mock_httpx_client

            result = await client.stream_post("http://test.com/stream", json_data)

            mock_httpx_client.stream.assert_called_once_with(
                "POST", "http://test.com/stream", json=json_data, timeout=30.0, headers=None
            )
            assert result == [
                '{"response": "Hello", "done": false}',
                '{"response": " World", "done": true}',
            ]

    @pytest.mark.asyncio
    async def test_stream_post_exception_handling(self, client):
        """Test streaming POST request exception handling."""
        import httpx

        with patch.object(client, '_ensure_client') as mock_ensure:
            mock_httpx_client = Mock()
            mock_httpx_client.stream.side_effect = httpx.TimeoutException("Stream timeout")
            mock_ensure.return_value = mock_httpx_client

            with pytest.raises(TimeoutError, match="Streaming request timeout"):
                await client.stream_post("http://test.com", {"data": "test"})

    @pytest.mark.asyncio
    async def test_close_with_client(self, client):
        """Test close method when client exists."""
        mock_httpx_client = AsyncMock()
        client._client = mock_httpx_client

        await client.close()

        mock_httpx_client.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, client):
        """Test close method when no client exists."""
        assert client._client is None

        # Should not raise exception
        await client.close()

        assert client._client is None


class TestMockHttpClient:
    """Test cases for MockHttpClient."""

    @pytest.fixture
    def mock_client(self):
        """Create MockHttpClient instance for testing."""
        return MockHttpClient()

    def test_mock_http_client_initialization(self, mock_client):
        """Test MockHttpClient initializes correctly."""
        assert mock_client.responses == {}
        assert mock_client.streaming_responses == {}
        assert mock_client.call_log == []
        assert mock_client.should_raise is None

    def test_set_response(self, mock_client):
        """Test set_response method."""
        response = HttpResponse(200, b"test", {"test": "data"})
        mock_client.set_response("GET", "http://test.com", response)

        assert mock_client.responses["GET:http://test.com"] == response

    def test_set_streaming_response(self, mock_client):
        """Test set_streaming_response method."""
        lines = ["line1", "line2", "line3"]
        mock_client.set_streaming_response("http://stream.com", lines)

        assert mock_client.streaming_responses["http://stream.com"] == lines

    def test_set_exception(self, mock_client):
        """Test set_exception method."""
        exception = ConnectionError("Test error")
        mock_client.set_exception(exception)

        assert mock_client.should_raise == exception

    def test_get_calls_and_clear_calls(self, mock_client):
        """Test get_calls and clear_calls methods."""
        # Initially empty
        assert mock_client.get_calls() == []

        # Add some calls
        mock_client.call_log.append({"method": "GET", "url": "test"})
        mock_client.call_log.append({"method": "POST", "url": "test2"})

        calls = mock_client.get_calls()
        assert len(calls) == 2
        assert calls[0]["method"] == "GET"
        assert calls[1]["method"] == "POST"

        # Verify it returns a copy
        calls.append({"method": "DELETE"})
        assert len(mock_client.call_log) == 2

        # Clear calls
        mock_client.clear_calls()
        assert mock_client.call_log == []

    @pytest.mark.asyncio
    async def test_mock_get_with_preset_response(self, mock_client):
        """Test mock GET with preset response."""
        expected_response = HttpResponse(404, b"Not found", None)
        mock_client.set_response("GET", "http://test.com", expected_response)

        result = await mock_client.get("http://test.com", timeout=15.0)

        assert result == expected_response
        calls = mock_client.get_calls()
        assert len(calls) == 1
        assert calls[0]["method"] == "GET"
        assert calls[0]["url"] == "http://test.com"
        assert calls[0]["timeout"] == 15.0

    @pytest.mark.asyncio
    async def test_mock_get_with_default_response(self, mock_client):
        """Test mock GET with default response."""
        result = await mock_client.get("http://unknown.com")

        assert result.status_code == 200
        assert result.json_data == {"models": []}
        calls = mock_client.get_calls()
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_mock_get_with_exception(self, mock_client):
        """Test mock GET with exception."""
        test_exception = HttpError("Test error", 500)
        mock_client.set_exception(test_exception)

        with pytest.raises(HttpError, match="Test error"):
            await mock_client.get("http://test.com")

        # Exception should be cleared after use
        assert mock_client.should_raise is None

    @pytest.mark.asyncio
    async def test_mock_post_with_preset_response(self, mock_client):
        """Test mock POST with preset response."""
        expected_response = HttpResponse(201, b"Created", {"id": 123})
        mock_client.set_response("POST", "http://api.com", expected_response)

        json_data = {"name": "test"}
        result = await mock_client.post("http://api.com", json_data, timeout=45.0)

        assert result == expected_response
        calls = mock_client.get_calls()
        assert len(calls) == 1
        assert calls[0]["method"] == "POST"
        assert calls[0]["json_data"] == json_data
        assert calls[0]["timeout"] == 45.0

    @pytest.mark.asyncio
    async def test_mock_post_with_default_response(self, mock_client):
        """Test mock POST with default response."""
        result = await mock_client.post("http://api.com", {"data": "test"})

        assert result.status_code == 200
        assert result.json_data == {"response": "Mock response"}

    @pytest.mark.asyncio
    async def test_mock_stream_post_with_preset_response(self, mock_client):
        """Test mock streaming POST with preset response."""
        expected_lines = ["stream1", "stream2", "stream3"]
        mock_client.set_streaming_response("http://stream.com", expected_lines)

        result = await mock_client.stream_post("http://stream.com", {"prompt": "test"})

        assert result == expected_lines
        calls = mock_client.get_calls()
        assert len(calls) == 1
        assert calls[0]["method"] == "STREAM_POST"

    @pytest.mark.asyncio
    async def test_mock_stream_post_with_default_response(self, mock_client):
        """Test mock streaming POST with default response."""
        result = await mock_client.stream_post("http://unknown.com", {"data": "test"})

        expected_default = [
            '{"response": "Mock ", "done": false}',
            '{"response": "streaming ", "done": false}',
            '{"response": "response", "done": true}',
        ]
        assert result == expected_default

    @pytest.mark.asyncio
    async def test_mock_stream_post_with_exception(self, mock_client):
        """Test mock streaming POST with exception."""
        test_exception = ConnectionError("Stream failed")
        mock_client.set_exception(test_exception)

        with pytest.raises(ConnectionError, match="Stream failed"):
            await mock_client.stream_post("http://stream.com", {"prompt": "test"})

    @pytest.mark.asyncio
    async def test_mock_close(self, mock_client):
        """Test mock close method."""
        await mock_client.close()

        calls = mock_client.get_calls()
        assert len(calls) == 1
        assert calls[0]["method"] == "CLOSE"


class TestFallbackAsyncClient:
    """Test cases for FallbackAsyncClient."""

    @pytest.fixture
    def fallback_client(self):
        """Create FallbackAsyncClient instance for testing."""
        return FallbackAsyncClient()

    def test_fallback_client_initialization(self, fallback_client):
        """Test FallbackAsyncClient initializes correctly."""
        assert fallback_client._session is None

    @pytest.mark.asyncio
    async def test_fallback_get_success(self, fallback_client):
        """Test fallback GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fallback content"

        with patch('requests.get') as mock_get:
            mock_get.return_value = mock_response

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_response)

                result = await fallback_client.get("http://test.com", timeout=30.0)

                assert isinstance(result, HttpResponse)
                assert result.status_code == 200
                assert result.content == b"fallback content"

    @pytest.mark.asyncio
    async def test_fallback_post_success(self, fallback_client):
        """Test fallback POST request."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.content = b"created"

        json_data = {"test": "data"}

        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_response)

                result = await fallback_client.post("http://api.com", json_data, timeout=60.0)

                assert result.status_code == 201
                assert result.content == b"created"

    @pytest.mark.asyncio
    async def test_fallback_stream_post_success(self, fallback_client):
        """Test fallback streaming POST request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Fallback stream response"}

        json_data = {"prompt": "test"}

        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_response)

                result = await fallback_client.stream_post("http://stream.com", json_data)

                expected = ['{"response": "Fallback stream response", "done": true}']
                assert result == expected

    @pytest.mark.asyncio
    async def test_fallback_stream_post_missing_response_key(self, fallback_client):
        """Test fallback streaming POST with missing response key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "no response key"}

        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_response)

                with pytest.raises(ValueError, match="Missing 'response' in JSON response"):
                    await fallback_client.stream_post("http://stream.com", {"prompt": "test"})

    @pytest.mark.asyncio
    async def test_fallback_close(self, fallback_client):
        """Test fallback close method."""
        # Should not raise exception
        await fallback_client.close()


# Integration tests
class TestHttpClientsIntegration:
    """Integration tests for the http_clients module."""

    def test_module_imports_successfully(self):
        """Test that the module can be imported without errors."""
        import src.generation.http_clients
        assert hasattr(src.generation.http_clients, 'AsyncHttpxClient')
        # MockHttpClient is in tests.conftest, not production module
        assert hasattr(src.generation.http_clients, 'FallbackAsyncClient')

    def test_all_clients_have_same_interface(self):
        """Test that all client classes have the same interface."""
        clients = [AsyncHttpxClient(), MockHttpClient(), FallbackAsyncClient()]

        for client in clients:
            assert hasattr(client, 'get')
            assert hasattr(client, 'post')
            assert hasattr(client, 'stream_post')
            assert hasattr(client, 'close')

    def test_http_response_import_works(self):
        """Test that HttpResponse import from ollama_client works."""
        from src.generation.http_clients import HttpResponse
        response = HttpResponse(200, b"test", {"test": "data"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_mock_client_realistic_workflow(self):
        """Test MockHttpClient in a realistic workflow."""
        mock_client = MockHttpClient()

        # Set up responses
        mock_client.set_response("GET", "http://api.com/models",
                                HttpResponse(200, b'{"models": ["model1", "model2"]}',
                                           {"models": ["model1", "model2"]}))

        mock_client.set_response("POST", "http://api.com/generate",
                                HttpResponse(201, b'{"id": "gen123"}', {"id": "gen123"}))

        mock_client.set_streaming_response("http://api.com/stream",
                                         ['{"chunk": "Hello"', '{"chunk": " World"}'])

        # Simulate API calls
        models = await mock_client.get("http://api.com/models")
        generation = await mock_client.post("http://api.com/generate", {"prompt": "test"})
        stream = await mock_client.stream_post("http://api.com/stream", {"data": "stream"})

        # Verify responses
        assert models.json_data["models"] == ["model1", "model2"]
        assert generation.json_data["id"] == "gen123"
        assert len(stream) == 2

        # Verify call logging
        calls = mock_client.get_calls()
        assert len(calls) == 3
        assert calls[0]["method"] == "GET"
        assert calls[1]["method"] == "POST"
        assert calls[2]["method"] == "STREAM_POST"