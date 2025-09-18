"""
HTTP client implementations for Ollama client dependency injection.
Provides testable HTTP abstraction layer.
"""

import asyncio
from typing import Any

import httpx

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_error_context,
    log_performance_metric,
)
from .ollama_client import ConnectionError, HttpError, HttpResponse


class AsyncHttpxClient:
    """
    Production HTTP client using httpx library.
    """

    def __init__(self):
        logger = get_system_logger()
        log_component_start("http_client", "init", client_type="AsyncHttpxClient")
        self._client = None
        logger.debug("http_client", "init", "HTTP client initialized without connection")
        log_component_end("http_client", "init", "AsyncHttpxClient ready")

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        logger = get_system_logger()
        logger.trace("http_client", "_ensure_client", "Checking client connection state")

        if self._client is None:
            logger.debug("http_client", "_ensure_client", "Creating new httpx.AsyncClient")
            self._client = httpx.AsyncClient()
            logger.info("http_client", "_ensure_client", "HTTP client connection established")
        else:
            logger.trace("http_client", "_ensure_client", "Reusing existing HTTP client")

        return self._client

    async def get(self, url: str, timeout: float = 30.0) -> HttpResponse:
        """Make GET request."""
        logger = get_system_logger()
        log_component_start("http_client", "get", url=url, timeout=timeout)

        client = self._ensure_client()

        try:
            logger.debug("http_client", "get", f"Executing GET request to {url}")
            response = await client.get(url, timeout=timeout)

            logger.trace("http_client", "get", f"Response status: {response.status_code}")
            log_performance_metric("http_client", "get", "response_status_code", response.status_code)

            json_data = None
            content_type = response.headers.get("content-type", "")
            logger.trace("http_client", "get", f"Response content-type: {content_type}")

            if content_type.startswith("application/json"):
                logger.debug("http_client", "get", "Parsing JSON response data")
                json_data = response.json()
                logger.trace("http_client", "get", f"JSON data keys: {list(json_data.keys()) if json_data else 'None'}")

            log_performance_metric("http_client", "get", "response_content_length", len(response.content))
            response_obj = HttpResponse(status_code=response.status_code, content=response.content, json_data=json_data)

            log_component_end("http_client", "get", f"GET request completed: {response.status_code}")
            return response_obj

        except httpx.ConnectError as e:
            log_error_context("http_client", "get", e, {"url": url, "timeout": timeout, "error_type": "connection"})
            raise ConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPStatusError as e:
            log_error_context(
                "http_client",
                "get",
                e,
                {"url": url, "status_code": e.response.status_code, "error_type": "http_status"},
            )
            raise HttpError(f"HTTP error: {e}", e.response.status_code) from e
        except httpx.TimeoutException as e:
            log_error_context("http_client", "get", e, {"url": url, "timeout": timeout, "error_type": "timeout"})
            raise TimeoutError(f"Request timeout: {e}") from e

    async def post(
        self, url: str, json_data: dict[str, Any], timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        """Make POST request."""
        logger = get_system_logger()
        log_component_start("http_client", "post", url=url, timeout=timeout, data_keys=list(json_data.keys()))

        client = self._ensure_client()

        try:
            logger.debug("http_client", "post", f"Executing POST request to {url}")
            logger.trace("http_client", "post", f"Request data size: {len(str(json_data))} chars")
            response = await client.post(url, json=json_data, timeout=timeout, headers=headers)

            logger.trace("http_client", "post", f"Response status: {response.status_code}")
            log_performance_metric("http_client", "post", "response_status_code", response.status_code)

            json_data_response = None
            content_type = response.headers.get("content-type", "")
            logger.trace("http_client", "post", f"Response content-type: {content_type}")

            if content_type.startswith("application/json"):
                logger.debug("http_client", "post", "Parsing JSON response data")
                json_data_response = response.json()
                logger.trace(
                    "http_client",
                    "post",
                    f"JSON response keys: {list(json_data_response.keys()) if json_data_response else 'None'}",
                )

            log_performance_metric("http_client", "post", "response_content_length", len(response.content))
            response_obj = HttpResponse(
                status_code=response.status_code, content=response.content, json_data=json_data_response
            )

            log_component_end("http_client", "post", f"POST request completed: {response.status_code}")
            return response_obj

        except httpx.ConnectError as e:
            log_error_context("http_client", "post", e, {"url": url, "timeout": timeout, "error_type": "connection"})
            raise ConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPStatusError as e:
            log_error_context(
                "http_client",
                "post",
                e,
                {"url": url, "status_code": e.response.status_code, "error_type": "http_status"},
            )
            raise HttpError(f"HTTP error: {e}", e.response.status_code) from e
        except httpx.TimeoutException as e:
            log_error_context("http_client", "post", e, {"url": url, "timeout": timeout, "error_type": "timeout"})
            raise TimeoutError(f"Request timeout: {e}") from e

    async def stream_post(
        self, url: str, json_data: dict[str, Any], timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> list[str]:
        """Make streaming POST request, return lines."""
        logger = get_system_logger()
        log_component_start("http_client", "stream_post", url=url, timeout=timeout, data_keys=list(json_data.keys()))

        client = self._ensure_client()
        lines = []

        try:
            logger.debug("http_client", "stream_post", f"Executing streaming POST request to {url}")
            logger.trace("http_client", "stream_post", f"Request data size: {len(str(json_data))} chars")

            async with client.stream("POST", url, json=json_data, timeout=timeout, headers=headers) as response:
                logger.trace("http_client", "stream_post", f"Stream response status: {response.status_code}")
                response.raise_for_status()
                log_performance_metric("http_client", "stream_post", "response_status_code", response.status_code)

                logger.debug("http_client", "stream_post", "Processing streaming response lines")
                async for line in response.aiter_lines():
                    if line.strip():
                        lines.append(line)
                        logger.trace("http_client", "stream_post", f"Received line: {len(line)} chars")

            log_performance_metric("http_client", "stream_post", "total_lines_received", len(lines))
            logger.debug("http_client", "stream_post", f"Streaming completed with {len(lines)} lines")

            log_component_end("http_client", "stream_post", f"Streaming POST completed: {len(lines)} lines")
            return lines

        except httpx.ConnectError as e:
            log_error_context(
                "http_client", "stream_post", e, {"url": url, "timeout": timeout, "error_type": "connection"}
            )
            raise ConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPStatusError as e:
            log_error_context(
                "http_client",
                "stream_post",
                e,
                {"url": url, "status_code": e.response.status_code, "error_type": "http_status"},
            )
            raise HttpError(f"HTTP error: {e}", e.response.status_code) from e
        except httpx.TimeoutException as e:
            log_error_context(
                "http_client", "stream_post", e, {"url": url, "timeout": timeout, "error_type": "timeout"}
            )
            raise TimeoutError(f"Streaming request timeout: {e}") from e

    async def close(self) -> None:
        """Close HTTP client."""
        logger = get_system_logger()
        log_component_start("http_client", "close")

        if self._client:
            logger.debug("http_client", "close", "Closing active HTTP client connection")
            await self._client.aclose()
            self._client = None
            logger.info("http_client", "close", "HTTP client connection closed")
        else:
            logger.trace("http_client", "close", "No active HTTP client to close")

        log_component_end("http_client", "close", "HTTP client cleanup completed")


class MockHttpClient:
    """
    Mock HTTP client for testing.
    Allows complete control over responses for unit tests.
    """

    def __init__(self):
        self.responses = {}
        self.streaming_responses = {}
        self.call_log = []
        self.should_raise = None

    def set_response(self, method: str, url: str, response: HttpResponse):
        """Set response for specific method and URL."""
        key = f"{method.upper()}:{url}"
        self.responses[key] = response

    def set_streaming_response(self, url: str, lines: list[str]):
        """Set streaming response for URL."""
        self.streaming_responses[url] = lines

    def set_exception(self, exception: Exception):
        """Set exception to raise on next request."""
        self.should_raise = exception

    def get_calls(self) -> list[dict[str, Any]]:
        """Get log of all API calls made."""
        return self.call_log.copy()

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    async def get(self, url: str, timeout: float = 30.0) -> HttpResponse:
        """Mock GET request."""
        self.call_log.append({"method": "GET", "url": url, "timeout": timeout})

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        key = f"GET:{url}"
        if key in self.responses:
            return self.responses[key]

        # Default response
        return HttpResponse(status_code=200, content=b'{"models": []}', json_data={"models": []})

    async def post(
        self, url: str, json_data: dict[str, Any], timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        """Mock POST request."""
        self.call_log.append({"method": "POST", "url": url, "json_data": json_data, "timeout": timeout})

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        key = f"POST:{url}"
        if key in self.responses:
            return self.responses[key]

        # Default response
        return HttpResponse(
            status_code=200, content=b'{"response": "Mock response"}', json_data={"response": "Mock response"}
        )

    async def stream_post(
        self, url: str, json_data: dict[str, Any], timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> list[str]:
        """Mock streaming POST request."""
        self.call_log.append({"method": "STREAM_POST", "url": url, "json_data": json_data, "timeout": timeout})

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        if url in self.streaming_responses:
            return self.streaming_responses[url]

        # Default streaming response
        return [
            '{"response": "Mock ", "done": false}',
            '{"response": "streaming ", "done": false}',
            '{"response": "response", "done": true}',
        ]

    async def close(self) -> None:
        """Mock close operation."""
        self.call_log.append({"method": "CLOSE"})


# Fallback HTTP client for environments without httpx
class FallbackAsyncClient:
    """
    Fallback HTTP client using built-in libraries.
    Used when httpx is not available.
    """

    def __init__(self):
        self._session = None

    async def get(self, url: str, timeout: float = 30.0) -> HttpResponse:
        """Fallback GET using requests in executor."""
        import requests

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=timeout))

        return HttpResponse(status_code=response.status_code, content=response.content)

    async def post(
        self, url: str, json_data: dict[str, Any], timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        """Fallback POST using requests in executor."""
        import requests

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: requests.post(url, json=json_data, timeout=timeout, headers=headers)
        )

        return HttpResponse(status_code=response.status_code, content=response.content)

    async def stream_post(
        self, url: str, json_data: dict[str, Any], timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> list[str]:
        """Fallback streaming (simulated) using requests in executor."""
        import requests

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: requests.post(url, json=json_data, timeout=timeout, headers=headers)
        )

        # Simulate streaming by wrapping response
        response_json = response.json()
        if "response" not in response_json:
            raise ValueError("Missing 'response' in JSON response")
        response_text = response_json["response"]
        return [f'{{"response": "{response_text}", "done": true}}']

    async def close(self) -> None:
        """Nothing to close for fallback client."""
        pass
