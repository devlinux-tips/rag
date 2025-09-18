"""
Universal LLM Provider Implementation Design
==========================================

This module defines the interfaces and implementation strategy for a universal
LLM provider system that can work with OpenAI GPT, Google Gemini, and Anthropic Claude.

The design follows the adapter pattern to abstract provider differences while
maintaining maximum flexibility and vendor independence.
"""

from typing import Protocol, Dict, Any, List, Optional, Union, AsyncIterator
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod


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
    COMPLETED = "completed"      # Natural completion
    MAX_TOKENS = "max_tokens"    # Token limit reached
    STOPPED = "stopped"          # Stop sequence encountered
    FILTERED = "filtered"        # Content filtered/safety


class ProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"


@dataclass
class UniversalMessage:
    """Standardized message format across all providers."""
    role: MessageRole
    content: str


@dataclass
class UniversalRequest:
    """Standardized request format for all LLM providers."""
    model: str
    messages: List[UniversalMessage]
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    stop_sequences: Optional[List[str]] = None


@dataclass
class TokenUsage:
    """Token usage statistics."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class UniversalResponse:
    """Standardized response format from all providers."""
    id: str
    content: str
    finish_reason: FinishReason
    usage: TokenUsage
    provider: ProviderType
    model: str


@dataclass
class StreamChunk:
    """Standardized streaming chunk."""
    content: str
    finish_reason: Optional[FinishReason] = None


# ============================================================================
# Provider Configuration Models
# ============================================================================

@dataclass
class ProviderConfig:
    """Base configuration for LLM providers."""
    api_key: str
    base_url: str
    model: str
    endpoint: str
    auth_header: str
    auth_prefix: str
    response_format: str


@dataclass
class OpenAIConfig(ProviderConfig):
    """OpenAI-specific configuration."""
    def __post_init__(self):
        if not self.endpoint:
            self.endpoint = "/v1/chat/completions"
        if not self.auth_header:
            self.auth_header = "Authorization"
        if not self.auth_prefix:
            self.auth_prefix = "Bearer "


@dataclass
class GeminiConfig(ProviderConfig):
    """Google Gemini-specific configuration."""
    def __post_init__(self):
        if not self.endpoint:
            self.endpoint = f"/v1beta/models/{self.model}:generateContent"
        if not self.auth_header:
            self.auth_header = "x-goog-api-key"
        if not self.auth_prefix:
            self.auth_prefix = ""


@dataclass
class ClaudeConfig(ProviderConfig):
    """Anthropic Claude-specific configuration."""
    def __post_init__(self):
        if not self.endpoint:
            self.endpoint = "/v1/messages"
        if not self.auth_header:
            self.auth_header = "x-api-key"
        if not self.auth_prefix:
            self.auth_prefix = ""


# ============================================================================
# Core Interfaces
# ============================================================================

class LLMProvider(Protocol):
    """Protocol defining the interface for all LLM providers."""

    async def generate(self, request: UniversalRequest) -> UniversalResponse:
        """Generate a response from the LLM provider."""
        ...

    async def stream_generate(self, request: UniversalRequest) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response from the LLM provider."""
        ...

    def get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        ...


# ============================================================================
# Abstract Base Adapter
# ============================================================================

class BaseLLMAdapter(ABC):
    """Abstract base class for LLM provider adapters."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def _transform_request(self, request: UniversalRequest) -> Dict[str, Any]:
        """Transform universal request to provider-specific format."""
        pass

    @abstractmethod
    def _transform_response(self, response: Dict[str, Any]) -> UniversalResponse:
        """Transform provider-specific response to universal format."""
        pass

    @abstractmethod
    def _extract_stream_chunk(self, chunk: str) -> Optional[StreamChunk]:
        """Extract content from streaming chunk."""
        pass

    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for the provider."""
        return {
            self.config.auth_header: f"{self.config.auth_prefix}{self.config.api_key}",
            "Content-Type": "application/json"
        }


# ============================================================================
# Concrete Adapter Implementations
# ============================================================================

class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI GPT adapter implementation."""

    def get_provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def _transform_request(self, request: UniversalRequest) -> Dict[str, Any]:
        """Transform to OpenAI chat completions format."""
        messages = []

        # Add system prompt as first message if provided
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })

        # Add conversation messages
        for msg in request.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream
        }

        # Add optional parameters
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        return payload

    def _transform_response(self, response: Dict[str, Any]) -> UniversalResponse:
        """Transform OpenAI response to universal format."""
        choice = response["choices"][0]
        usage = response.get("usage", {})

        # Map finish reason
        finish_reason_map = {
            "stop": FinishReason.COMPLETED,
            "length": FinishReason.MAX_TOKENS,
            "content_filter": FinishReason.FILTERED
        }

        return UniversalResponse(
            id=response["id"],
            content=choice["message"]["content"],
            finish_reason=finish_reason_map.get(choice["finish_reason"], FinishReason.COMPLETED),
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            ),
            provider=ProviderType.OPENAI,
            model=response["model"]
        )

    def _extract_stream_chunk(self, chunk: str) -> Optional[StreamChunk]:
        """Extract content from OpenAI streaming chunk."""
        import json

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
                    "content_filter": FinishReason.FILTERED
                }
                finish_reason = finish_reason_map.get(parsed["choices"][0]["finish_reason"])

            return StreamChunk(content=content, finish_reason=finish_reason)
        except (json.JSONDecodeError, KeyError):
            return None


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini adapter implementation."""

    def get_provider_type(self) -> ProviderType:
        return ProviderType.GEMINI

    def _transform_request(self, request: UniversalRequest) -> Dict[str, Any]:
        """Transform to Gemini generateContent format."""
        contents = []

        # Transform messages (Gemini uses 'user' and 'model' roles)
        for msg in request.messages:
            role = "user" if msg.role == MessageRole.USER else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })

        payload = {
            "contents": contents
        }

        # Add system instruction if provided
        if request.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": request.system_prompt}]
            }

        # Add generation config
        generation_config = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens:
            generation_config["maxOutputTokens"] = request.max_tokens

        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    def _transform_response(self, response: Dict[str, Any]) -> UniversalResponse:
        """Transform Gemini response to universal format."""
        candidate = response["candidates"][0]
        content = candidate["content"]["parts"][0]["text"]
        usage = response.get("usageMetadata", {})

        # Map finish reason
        finish_reason_map = {
            "STOP": FinishReason.COMPLETED,
            "MAX_TOKENS": FinishReason.MAX_TOKENS,
            "SAFETY": FinishReason.FILTERED,
            "RECITATION": FinishReason.FILTERED
        }

        return UniversalResponse(
            id=f"gemini-{hash(content)}",  # Gemini doesn't provide response ID
            content=content,
            finish_reason=finish_reason_map.get(candidate.get("finishReason"), FinishReason.COMPLETED),
            usage=TokenUsage(
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
                total_tokens=usage.get("totalTokenCount", 0)
            ),
            provider=ProviderType.GEMINI,
            model=self.config.model
        )

    def _extract_stream_chunk(self, chunk: str) -> Optional[StreamChunk]:
        """Extract content from Gemini streaming chunk."""
        import json

        if not chunk.startswith("data: "):
            return None

        data = chunk[6:].strip()

        try:
            parsed = json.loads(data)
            candidate = parsed["candidates"][0]
            content = candidate["content"]["parts"][0]["text"]

            finish_reason = None
            if candidate.get("finishReason"):
                finish_reason_map = {
                    "STOP": FinishReason.COMPLETED,
                    "MAX_TOKENS": FinishReason.MAX_TOKENS,
                    "SAFETY": FinishReason.FILTERED
                }
                finish_reason = finish_reason_map.get(candidate["finishReason"])

            return StreamChunk(content=content, finish_reason=finish_reason)
        except (json.JSONDecodeError, KeyError):
            return None


class ClaudeAdapter(BaseLLMAdapter):
    """Anthropic Claude adapter implementation."""

    def get_provider_type(self) -> ProviderType:
        return ProviderType.CLAUDE

    def _transform_request(self, request: UniversalRequest) -> Dict[str, Any]:
        """Transform to Claude messages format."""
        messages = []

        # Transform messages (Claude doesn't include system in messages)
        for msg in request.messages:
            if msg.role != MessageRole.SYSTEM:  # System handled separately
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024  # Required for Claude
        }

        # Add system prompt if provided
        if request.system_prompt:
            payload["system"] = request.system_prompt

        # Add optional parameters
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stream:
            payload["stream"] = request.stream
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        return payload

    def _transform_response(self, response: Dict[str, Any]) -> UniversalResponse:
        """Transform Claude response to universal format."""
        content = response["content"][0]["text"]
        usage = response.get("usage", {})

        # Map finish reason
        finish_reason_map = {
            "end_turn": FinishReason.COMPLETED,
            "max_tokens": FinishReason.MAX_TOKENS,
            "stop_sequence": FinishReason.STOPPED
        }

        return UniversalResponse(
            id=response["id"],
            content=content,
            finish_reason=finish_reason_map.get(response.get("stop_reason"), FinishReason.COMPLETED),
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            ),
            provider=ProviderType.CLAUDE,
            model=response["model"]
        )

    def _extract_stream_chunk(self, chunk: str) -> Optional[StreamChunk]:
        """Extract content from Claude streaming chunk."""
        import json

        if not chunk.startswith("data: "):
            return None

        data = chunk[6:].strip()

        try:
            parsed = json.loads(data)

            # Handle different event types
            if parsed.get("type") == "content_block_delta":
                delta = parsed.get("delta", {})
                content = delta.get("text", "")
                return StreamChunk(content=content)

            elif parsed.get("type") == "message_stop":
                return StreamChunk(content="", finish_reason=FinishReason.COMPLETED)

            return None
        except (json.JSONDecodeError, KeyError):
            return None


# ============================================================================
# Provider Factory
# ============================================================================

class LLMProviderFactory:
    """Factory for creating LLM provider adapters."""

    @staticmethod
    def create_provider(provider_type: ProviderType, config: ProviderConfig) -> LLMProvider:
        """Create a provider adapter based on type and configuration."""
        if provider_type == ProviderType.OPENAI:
            return OpenAIAdapter(config)
        elif provider_type == ProviderType.GEMINI:
            return GeminiAdapter(config)
        elif provider_type == ProviderType.CLAUDE:
            return ClaudeAdapter(config)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")


# ============================================================================
# Universal LLM Manager
# ============================================================================

class UniversalLLMManager:
    """
    Main manager class that provides a unified interface to multiple LLM providers
    with automatic failover and provider selection.
    """

    def __init__(self, provider_configs: Dict[str, ProviderConfig], primary_provider: str, fallback_order: List[str]):
        self.provider_configs = provider_configs
        self.primary_provider = primary_provider
        self.fallback_order = fallback_order
        self.providers = {}

        # Initialize all configured providers
        for name, config in provider_configs.items():
            provider_type = ProviderType(name)
            self.providers[name] = LLMProviderFactory.create_provider(provider_type, config)

    async def generate(self, request: UniversalRequest, language: str = "en") -> UniversalResponse:
        """
        Generate response using the configured provider chain.
        Automatically handles failover if primary provider fails.
        """
        # Add language-specific system prompt
        if language and not request.system_prompt:
            request.system_prompt = self._get_language_system_prompt(language)

        # Try primary provider first
        providers_to_try = [self.primary_provider] + self.fallback_order

        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue

            try:
                provider = self.providers[provider_name]
                return await provider.generate(request)
            except Exception as e:
                print(f"Provider {provider_name} failed: {e}")
                continue

        raise RuntimeError("All LLM providers failed")

    async def stream_generate(self, request: UniversalRequest, language: str = "en") -> AsyncIterator[StreamChunk]:
        """Generate streaming response with automatic failover."""
        # Add language-specific system prompt
        if language and not request.system_prompt:
            request.system_prompt = self._get_language_system_prompt(language)

        # Try primary provider first
        providers_to_try = [self.primary_provider] + self.fallback_order

        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue

            try:
                provider = self.providers[provider_name]
                async for chunk in provider.stream_generate(request):
                    yield chunk
                return
            except Exception as e:
                print(f"Provider {provider_name} failed: {e}")
                continue

        raise RuntimeError("All LLM providers failed")

    def _get_language_system_prompt(self, language: str) -> str:
        """Get language-specific system prompt."""
        language_prompts = {
            "hr": "Ti si pomoćni asistent koji odgovara ISKLJUČIVO na hrvatskom jeziku. Bez obzira na kontekst, uvijek odgovori na hrvatskom. Koristi dane informacije da daš precizan odgovor.",
            "en": "You are a helpful assistant who responds EXCLUSIVELY in English. Regardless of context, always respond in English. Use the given information to provide a precise answer."
        }
        return language_prompts.get(language, language_prompts["en"])


# ============================================================================
# Configuration Example
# ============================================================================

def create_universal_llm_manager_from_config(config: Dict[str, Any]) -> UniversalLLMManager:
    """Create UniversalLLMManager from configuration dictionary."""
    providers_config = config["providers"]

    # Create provider configurations
    provider_configs = {}

    if "openai" in providers_config:
        openai_config = providers_config["openai"]
        provider_configs["openai"] = OpenAIConfig(
            api_key=openai_config["api_key"],
            base_url=openai_config["base_url"],
            model=openai_config["model"],
            endpoint=openai_config.get("endpoint", "/v1/chat/completions"),
            auth_header=openai_config.get("auth_header", "Authorization"),
            auth_prefix=openai_config.get("auth_prefix", "Bearer "),
            response_format=openai_config.get("response_format", "openai")
        )

    if "gemini" in providers_config:
        gemini_config = providers_config["gemini"]
        provider_configs["gemini"] = GeminiConfig(
            api_key=gemini_config["api_key"],
            base_url=gemini_config["base_url"],
            model=gemini_config["model"],
            endpoint=gemini_config.get("endpoint", f"/v1beta/models/{gemini_config['model']}:generateContent"),
            auth_header=gemini_config.get("auth_header", "x-goog-api-key"),
            auth_prefix=gemini_config.get("auth_prefix", ""),
            response_format=gemini_config.get("response_format", "gemini")
        )

    if "claude" in providers_config:
        claude_config = providers_config["claude"]
        provider_configs["claude"] = ClaudeConfig(
            api_key=claude_config["api_key"],
            base_url=claude_config["base_url"],
            model=claude_config["model"],
            endpoint=claude_config.get("endpoint", "/v1/messages"),
            auth_header=claude_config.get("auth_header", "x-api-key"),
            auth_prefix=claude_config.get("auth_prefix", ""),
            response_format=claude_config.get("response_format", "anthropic")
        )

    return UniversalLLMManager(
        provider_configs=provider_configs,
        primary_provider=providers_config["primary"],
        fallback_order=providers_config.get("fallback_order", [])
    )


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example configuration
    config = {
        "providers": {
            "primary": "claude",
            "fallback_order": ["openai", "gemini"],
            "claude": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "base_url": "https://api.anthropic.com",
                "model": "claude-3-5-sonnet-20241022"
            },
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "base_url": "https://api.openai.com",
                "model": "gpt-4o-2024-08-06"
            },
            "gemini": {
                "api_key": "${GEMINI_API_KEY}",
                "base_url": "https://generativelanguage.googleapis.com",
                "model": "gemini-2.5-flash"
            }
        }
    }

    # Create manager
    llm_manager = create_universal_llm_manager_from_config(config)

    # Create request
    request = UniversalRequest(
        model="claude-3-5-sonnet-20241022",  # Will be handled by primary provider
        messages=[
            UniversalMessage(role=MessageRole.USER, content="Što je glavni grad Hrvatske?")
        ]
    )

    # Generate response (with automatic Croatian language handling)
    # response = await llm_manager.generate(request, language="hr")
    # print(f"Response: {response.content}")
    # print(f"Provider used: {response.provider}")

    print("Universal LLM Provider system designed successfully!")