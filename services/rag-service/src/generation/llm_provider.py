"""
Universal LLM Provider Implementation
Support for Ollama (local) and OpenRouter (remote) with chat message compatibility.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from ..utils.error_handler import ConfigurationError
from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start

# ============================================================================
# LLM Request/Response Logging
# ============================================================================


def dump_llm_request_response(timestamp: str, request_data: dict, response_data: dict, provider: str):
    """Dump LLM request and response data to timestamped JSON files."""
    try:
        os.makedirs("services/rag-service/logs", exist_ok=True)

        # Request dump
        request_filename = f"services/rag-service/logs/{timestamp}_llm_{provider}_request.json"
        with open(request_filename, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False, default=str)

        # Response dump
        response_filename = f"services/rag-service/logs/{timestamp}_llm_{provider}_response.json"
        with open(response_filename, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"INFO: Logged LLM {provider} request/response: {timestamp}")

    except Exception as e:
        print(f"ERROR: Failed to dump LLM logs: {str(e)}")


# ============================================================================
# Core Data Models
# ============================================================================


class MessageRole(Enum):
    """Standardized message roles across all providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FinishReason(Enum):
    """Standardized finish reasons across all providers."""

    COMPLETED = "completed"
    MAX_TOKENS = "max_tokens"
    STOPPED = "stopped"
    FILTERED = "filtered"


class ProviderType(Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


@dataclass
class ChatMessage:
    """Chat message with role and content."""

    role: MessageRole
    content: str


@dataclass
class ChatRequest:
    """Request for chat completion."""

    messages: list[ChatMessage]
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None


@dataclass
class TokenUsage:
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    """Response from chat completion."""

    id: str
    content: str
    finish_reason: FinishReason
    usage: TokenUsage
    provider: ProviderType
    model: str


@dataclass
class StreamChunk:
    """Streaming chunk."""

    content: str
    finish_reason: FinishReason | None = None


# ============================================================================
# Provider Interfaces
# ============================================================================


class LLMProvider(Protocol):
    """Protocol defining interface for all LLM providers."""

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion."""
        ...

    def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Generate streaming chat completion."""
        ...

    def get_provider_type(self) -> ProviderType:
        """Return provider type."""
        ...


class HttpClient(Protocol):
    """HTTP client interface."""

    async def post_json(
        self, url: str, headers: dict[str, str], json_data: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Make POST request and return JSON."""
        ...

    def stream_post_lines(
        self, url: str, headers: dict[str, str], json_data: dict[str, Any], timeout: float
    ) -> AsyncIterator[str]:
        """Make streaming POST request."""
        ...


# ============================================================================
# Base Provider Implementation
# ============================================================================


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: dict[str, Any], http_client: HttpClient | None = None):
        self._validate_config(config)
        self.config = config
        self.http_client = http_client or self._create_default_http_client()
        self.logger = get_system_logger()

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate required configuration keys - fail fast if missing."""
        required_keys = self._get_required_config_keys()
        for key in required_keys:
            if key not in config:
                raise ConfigurationError(f"Missing required config key: {key}")

    @abstractmethod
    def _get_required_config_keys(self) -> list[str]:
        """Return list of required configuration keys for this provider."""
        pass

    def _create_default_http_client(self) -> HttpClient:
        """Create default HTTP client."""
        from .http_clients import AsyncHttpxClient

        return AsyncHttpxClient()

    @abstractmethod
    def _transform_request(self, request: ChatRequest) -> dict[str, Any]:
        """Transform universal request to provider-specific format."""
        pass

    @abstractmethod
    def _transform_response(self, response: dict[str, Any]) -> ChatResponse:
        """Transform provider response to universal format."""
        pass

    @abstractmethod
    def _extract_stream_chunk(self, chunk: str) -> StreamChunk | None:
        """Extract content from streaming chunk."""
        pass

    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """Return provider type."""
        pass

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}

        # Add authentication for OpenRouter
        if self.get_provider_type() == ProviderType.OPENROUTER:
            api_key = self.config["api_key"]
            if not api_key or len(api_key) < 25:
                self.logger.error("openrouter_provider", "_get_headers", "CRITICAL: API key missing or invalid")
                raise ValueError("OpenRouter API key is missing or invalid")
            # DO NOT LOG API KEY - security risk for GitHub secret scanning
            self.logger.debug("openrouter_provider", "_get_headers", "API key configured")
            headers["Authorization"] = f"Bearer {api_key}"

        return headers


# ============================================================================
# Ollama Provider
# ============================================================================


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider with chat compatibility."""

    def _get_required_config_keys(self) -> list[str]:
        """Return required configuration keys for Ollama provider."""
        return ["base_url", "timeout", "model", "temperature", "num_predict", "top_p", "top_k"]

    def get_provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion with Ollama."""
        log_component_start(
            "ollama_provider", "chat_completion", messages_count=len(request.messages), model=request.model
        )

        start_time = time.time()

        # Transform to Ollama format
        ollama_request = self._transform_request(request)

        # Make request
        url = f"{self.config['base_url']}/api/generate"
        headers = self._get_headers()

        try:
            response_data = await self.http_client.post_json(url, headers, ollama_request, self.config["timeout"])

            # Transform response
            chat_response = self._transform_response(response_data)

            generation_time = time.time() - start_time
            self.logger.info(
                "ollama_provider",
                "chat_completion",
                f"Generated {len(chat_response.content)} chars in {generation_time:.3f}s",
            )

            log_component_end("ollama_provider", "chat_completion", "Chat completion successful")
            return chat_response

        except Exception as e:
            self.logger.error("ollama_provider", "chat_completion", f"Request failed: {e}")
            raise

    async def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Generate streaming chat completion with Ollama."""
        log_component_start(
            "ollama_provider", "stream_chat_completion", messages_count=len(request.messages), model=request.model
        )

        # Transform to Ollama format with streaming
        ollama_request = self._transform_request(request)
        ollama_request["stream"] = True

        url = f"{self.config['base_url']}/api/generate"
        headers = self._get_headers()

        try:
            async for line in self.http_client.stream_post_lines(url, headers, ollama_request, self.config["timeout"]):
                chunk = self._extract_stream_chunk(line)
                if chunk:
                    yield chunk

        except Exception as e:
            self.logger.error("ollama_provider", "stream_chat_completion", f"Streaming failed: {e}")
            raise

    def _transform_request(self, request: ChatRequest) -> dict[str, Any]:
        """Transform ChatRequest to Ollama format."""
        # Convert chat messages to single prompt for Ollama
        prompt_parts = []

        for message in request.messages:
            if message.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {message.content}")
            elif message.role == MessageRole.USER:
                prompt_parts.append(f"Human: {message.content}")
            elif message.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {message.content}")

        # Add continuation for next assistant response
        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)

        ollama_request: dict[str, Any] = {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature or self.config["temperature"],
                "num_predict": request.max_tokens or self.config["num_predict"],
                "top_p": self.config["top_p"],
                "top_k": self.config["top_k"],
            },
        }

        if request.stop_sequences:
            ollama_request["options"]["stop"] = request.stop_sequences

        return ollama_request

    def _transform_response(self, response: dict[str, Any]) -> ChatResponse:
        """Transform Ollama response to ChatResponse."""
        content = response.get("response", "")

        # Estimate token usage (Ollama doesn't provide exact counts)
        input_tokens = len(response.get("prompt", "").split())
        output_tokens = len(content.split())

        return ChatResponse(
            id=f"ollama-{int(time.time() * 1000)}",
            content=content,
            finish_reason=FinishReason.COMPLETED,  # Ollama doesn't provide finish reason
            usage=TokenUsage(
                input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens
            ),
            provider=ProviderType.OLLAMA,
            model=response.get("model", self.config["model"]),
        )

    def _extract_stream_chunk(self, chunk: str) -> StreamChunk | None:
        """Extract content from Ollama streaming chunk."""
        if not chunk.strip():
            return None

        try:
            data = json.loads(chunk)
            content = data.get("response", "")

            finish_reason = None
            if data.get("done", False):
                finish_reason = FinishReason.COMPLETED

            return StreamChunk(content=content, finish_reason=finish_reason)

        except json.JSONDecodeError:
            return None


# ============================================================================
# OpenRouter Provider
# ============================================================================


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter API provider with chat compatibility."""

    def _get_required_config_keys(self) -> list[str]:
        """Return required configuration keys for OpenRouter provider."""
        return ["base_url", "timeout", "api_key"]

    def get_provider_type(self) -> ProviderType:
        return ProviderType.OPENROUTER

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion with OpenRouter."""
        log_component_start(
            "openrouter_provider", "chat_completion", messages_count=len(request.messages), model=request.model
        )

        start_time = time.time()

        # Transform to OpenAI-compatible format
        openai_request = self._transform_request(request)

        # Make request
        url = f"{self.config['base_url']}/chat/completions"
        headers = self._get_headers()

        try:
            response_data = await self.http_client.post_json(url, headers, openai_request, self.config["timeout"])

            # Transform response
            chat_response = self._transform_response(response_data)

            generation_time = time.time() - start_time
            self.logger.info(
                "openrouter_provider",
                "chat_completion",
                f"Generated {len(chat_response.content)} chars in {generation_time:.3f}s",
            )

            log_component_end("openrouter_provider", "chat_completion", "Chat completion successful")
            return chat_response

        except Exception as e:
            self.logger.error("openrouter_provider", "chat_completion", f"Request failed: {e}")
            raise

    async def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Generate streaming chat completion with OpenRouter."""
        log_component_start(
            "openrouter_provider", "stream_chat_completion", messages_count=len(request.messages), model=request.model
        )

        # Transform to OpenAI format with streaming
        openai_request = self._transform_request(request)
        openai_request["stream"] = True

        url = f"{self.config['base_url']}/chat/completions"
        headers = self._get_headers()

        try:
            async for line in self.http_client.stream_post_lines(url, headers, openai_request, self.config["timeout"]):
                chunk = self._extract_stream_chunk(line)
                if chunk:
                    yield chunk

        except Exception as e:
            self.logger.error("openrouter_provider", "stream_chat_completion", f"Streaming failed: {e}")
            raise

    def _transform_request(self, request: ChatRequest) -> dict[str, Any]:
        """Transform ChatRequest to OpenAI-compatible format."""
        messages = []

        for message in request.messages:
            messages.append({"role": message.role.value, "content": message.content})

        openai_request = {"model": request.model, "messages": messages, "stream": request.stream}

        if request.max_tokens:
            openai_request["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            openai_request["temperature"] = request.temperature
        if request.stop_sequences:
            openai_request["stop"] = request.stop_sequences

        return openai_request

    def _transform_response(self, response: dict[str, Any]) -> ChatResponse:
        """Transform OpenAI-compatible response to ChatResponse."""
        choice = response["choices"][0]
        usage = response.get("usage", {})

        # Map finish reason
        finish_reason_map = {
            "stop": FinishReason.COMPLETED,
            "length": FinishReason.MAX_TOKENS,
            "content_filter": FinishReason.FILTERED,
        }

        return ChatResponse(
            id=response["id"],
            content=choice["message"]["content"],
            finish_reason=finish_reason_map.get(choice["finish_reason"], FinishReason.COMPLETED),
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            provider=ProviderType.OPENROUTER,
            model=response["model"],
        )

    def _extract_stream_chunk(self, chunk: str) -> StreamChunk | None:
        """Extract content from OpenAI-compatible streaming chunk."""
        if not chunk.startswith("data: "):
            return None

        data = chunk[6:].strip()
        if data == "[DONE]":
            return StreamChunk(content="", finish_reason=FinishReason.COMPLETED)

        try:
            parsed = json.loads(data)
            delta = parsed["choices"][0].get("delta", {})
            content = delta.get("content", "")

            finish_reason = None
            if parsed["choices"][0].get("finish_reason"):
                finish_reason_map = {
                    "stop": FinishReason.COMPLETED,
                    "length": FinishReason.MAX_TOKENS,
                    "content_filter": FinishReason.FILTERED,
                }
                finish_reason = finish_reason_map.get(parsed["choices"][0]["finish_reason"])

            return StreamChunk(content=content, finish_reason=finish_reason)

        except (json.JSONDecodeError, KeyError):
            return None


# ============================================================================
# Provider Factory
# ============================================================================


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(
        provider_type: str, config: dict[str, Any], http_client: HttpClient | None = None
    ) -> LLMProvider:
        """Create provider based on type and configuration."""
        logger = get_system_logger()

        if provider_type == "ollama":
            logger.info("llm_factory", "create_provider", f"Creating Ollama provider for {config['model']}")
            return OllamaProvider(config, http_client)
        elif provider_type == "openrouter":
            logger.info("llm_factory", "create_provider", "Creating OpenRouter provider")
            return OpenRouterProvider(config, http_client)
        else:
            raise ConfigurationError(f"Unsupported LLM provider: {provider_type}")


# ============================================================================
# LLM Manager
# ============================================================================


class LLMManager:
    """
    Manager for multiple LLM providers with automatic failover.
    Supports both Ollama (local) and OpenRouter (remote) providers.
    """

    def __init__(self, config: dict[str, Any]):
        self._validate_manager_config(config)
        self.config = config
        self.providers: dict[str, LLMProvider] = {}
        self.primary_provider = config["primary_provider"]
        self.fallback_order = config["fallback_order"]
        self.logger = get_system_logger()

        self._initialize_providers()

    def _validate_manager_config(self, config: dict[str, Any]) -> None:
        """Validate LLM manager configuration - fail fast if missing."""
        required_keys = ["primary_provider", "fallback_order"]
        for key in required_keys:
            if key not in config:
                raise ConfigurationError(f"Missing required LLM manager config key: {key}")

    def _initialize_providers(self):
        """Initialize all configured providers."""
        log_component_start(
            "llm_manager", "initialize_providers", primary=self.primary_provider, fallbacks=len(self.fallback_order)
        )

        # Initialize Ollama if configured
        if "ollama" in self.config:
            self.providers["ollama"] = LLMProviderFactory.create_provider("ollama", self.config["ollama"])

        # Initialize OpenRouter if configured
        if "openrouter" in self.config:
            self.providers["openrouter"] = LLMProviderFactory.create_provider("openrouter", self.config["openrouter"])

        available_providers = list(self.providers.keys())
        self.logger.info("llm_manager", "initialize_providers", f"Initialized providers: {available_providers}")

        # Validate primary provider exists
        if self.primary_provider not in self.providers:
            raise ConfigurationError(f"Primary provider '{self.primary_provider}' not configured")

        log_component_end("llm_manager", "initialize_providers", f"Initialized {len(self.providers)} providers")

    async def chat_completion(self, messages: list[ChatMessage], model: str | None = None, **kwargs) -> ChatResponse:
        """Generate chat completion with automatic failover."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds

        request = ChatRequest(
            messages=messages,
            model=model or self._get_default_model(),
            max_tokens=kwargs.get("max_tokens"),  # Optional parameter - None is valid
            temperature=kwargs.get("temperature"),  # Optional parameter - None is valid
            stream=kwargs.get("stream", False),  # Has logical default
            stop_sequences=kwargs.get("stop_sequences"),  # Optional parameter - None is valid
        )

        # Log LLM request
        request_data = {
            "timestamp": timestamp,
            "provider_order": [self.primary_provider] + self.fallback_order,
            "request": {
                "messages": [{"role": msg.role.value, "content": msg.content} for msg in messages],
                "model": request.model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": request.stream,
                "stop_sequences": request.stop_sequences,
            },
        }

        # Try primary provider first, then fallbacks
        providers_to_try = [self.primary_provider] + self.fallback_order

        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue

            try:
                provider = self.providers[provider_name]
                response = await provider.chat_completion(request)

                # Log successful LLM response
                response_data = {
                    "timestamp": timestamp,
                    "successful_provider": provider_name,
                    "response": {
                        "content": response.content,
                        "finish_reason": response.finish_reason.value if response.finish_reason else None,
                        "model": response.model,
                        "provider": response.provider.value,
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.total_tokens,
                        },
                    },
                }

                dump_llm_request_response(timestamp, request_data, response_data, provider_name)

                return response

            except Exception as e:
                self.logger.warning("llm_manager", "chat_completion", f"Provider {provider_name} failed: {e}")
                continue

        raise RuntimeError("All LLM providers failed")

    async def stream_chat_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming chat completion with automatic failover."""
        request = ChatRequest(
            messages=messages,
            model=model or self._get_default_model(),
            max_tokens=kwargs.get("max_tokens"),  # Optional parameter - None is valid
            temperature=kwargs.get("temperature"),  # Optional parameter - None is valid
            stream=True,
            stop_sequences=kwargs.get("stop_sequences"),  # Optional parameter - None is valid
        )

        # Try primary provider first, then fallbacks
        providers_to_try = [self.primary_provider] + self.fallback_order

        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue

            try:
                provider = self.providers[provider_name]
                async for chunk in provider.stream_chat_completion(request):
                    yield chunk
                return

            except Exception as e:
                self.logger.warning("llm_manager", "stream_chat_completion", f"Provider {provider_name} failed: {e}")
                continue

        raise RuntimeError("All LLM providers failed for streaming")

    def _get_default_model(self) -> str:
        """Get default model from primary provider."""
        primary_config = self.config[self.primary_provider]
        return primary_config["model"]

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())


# ============================================================================
# Factory Function
# ============================================================================


def create_llm_manager(config: dict[str, Any]) -> LLMManager:
    """Create LLM manager from configuration."""
    return LLMManager(config)
