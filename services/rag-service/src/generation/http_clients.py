"""
HTTP client implementations for Ollama client dependency injection.
Provides testable HTTP abstraction layer.
"""

import asyncio
import json
from typing import Any, Dict, List

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .ollama_client import ConnectionError, HttpClient, HttpError, HttpResponse


class AsyncHttpxClient:
    """
    Production HTTP client using httpx library.
    """

    def __init__(self):
        self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            if not HTTPX_AVAILABLE:
                raise ImportError("httpx is required for AsyncHttpxClient")
            self._client = httpx.AsyncClient()
        return self._client

    async def get(self, url: str, timeout: float = 30.0) -> HttpResponse:
        """Make GET request."""
        client = self._ensure_client()

        try:
            response = await client.get(url, timeout=timeout)

            json_data = None
            if "content-type" in response.headers and response.headers[
                "content-type"
            ].startswith("application/json"):
                json_data = response.json()

            return HttpResponse(
                status_code=response.status_code,
                content=response.content,
                json_data=json_data,
            )
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}")
        except httpx.HTTPStatusError as e:
            raise HttpError(f"HTTP error: {e}", e.response.status_code)
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timeout: {e}")

    async def post(
        self, url: str, json_data: Dict[str, Any], timeout: float = 30.0
    ) -> HttpResponse:
        """Make POST request."""
        client = self._ensure_client()

        try:
            response = await client.post(url, json=json_data, timeout=timeout)

            json_data_response = None
            if "content-type" in response.headers and response.headers[
                "content-type"
            ].startswith("application/json"):
                json_data_response = response.json()

            return HttpResponse(
                status_code=response.status_code,
                content=response.content,
                json_data=json_data_response,
            )
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}")
        except httpx.HTTPStatusError as e:
            raise HttpError(f"HTTP error: {e}", e.response.status_code)
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timeout: {e}")

    async def stream_post(
        self, url: str, json_data: Dict[str, Any], timeout: float = 30.0
    ) -> List[str]:
        """Make streaming POST request, return lines."""
        client = self._ensure_client()
        lines = []

        try:
            async with client.stream(
                "POST", url, json=json_data, timeout=timeout
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        lines.append(line)
            return lines
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}")
        except httpx.HTTPStatusError as e:
            raise HttpError(f"HTTP error: {e}", e.response.status_code)
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Streaming request timeout: {e}")

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


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

    def set_streaming_response(self, url: str, lines: List[str]):
        """Set streaming response for URL."""
        self.streaming_responses[url] = lines

    def set_exception(self, exception: Exception):
        """Set exception to raise on next request."""
        self.should_raise = exception

    def get_calls(self) -> List[Dict[str, Any]]:
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
        return HttpResponse(
            status_code=200, content=b'{"models": []}', json_data={"models": []}
        )

    async def post(
        self, url: str, json_data: Dict[str, Any], timeout: float = 30.0
    ) -> HttpResponse:
        """Mock POST request."""
        self.call_log.append(
            {"method": "POST", "url": url, "json_data": json_data, "timeout": timeout}
        )

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        key = f"POST:{url}"
        if key in self.responses:
            return self.responses[key]

        # Default response
        return HttpResponse(
            status_code=200,
            content=b'{"response": "Mock response"}',
            json_data={"response": "Mock response"},
        )

    async def stream_post(
        self, url: str, json_data: Dict[str, Any], timeout: float = 30.0
    ) -> List[str]:
        """Mock streaming POST request."""
        self.call_log.append(
            {
                "method": "STREAM_POST",
                "url": url,
                "json_data": json_data,
                "timeout": timeout,
            }
        )

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
        response = await loop.run_in_executor(
            None, lambda: requests.get(url, timeout=timeout)
        )

        return HttpResponse(status_code=response.status_code, content=response.content)

    async def post(
        self, url: str, json_data: Dict[str, Any], timeout: float = 30.0
    ) -> HttpResponse:
        """Fallback POST using requests in executor."""
        import requests

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: requests.post(url, json=json_data, timeout=timeout)
        )

        return HttpResponse(status_code=response.status_code, content=response.content)

    async def stream_post(
        self, url: str, json_data: Dict[str, Any], timeout: float = 30.0
    ) -> List[str]:
        """Fallback streaming (simulated) using requests in executor."""
        import requests

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: requests.post(url, json=json_data, timeout=timeout)
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
