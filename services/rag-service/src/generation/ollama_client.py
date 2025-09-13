"""
Ollama client for local LLM integration providing text generation and streaming.
Handles HTTP communication with Ollama API for language model inference
with configurable parameters and error handling.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from ..preprocessing.cleaners import detect_language_content, preserve_text_encoding
from ..utils.config_loader import ConfigError


@dataclass
class OllamaConfig:
    """Configuration for Ollama client - pure data structure."""

    # Server settings
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0

    # Model settings
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    top_k: int = 64

    # Language-specific settings
    preserve_diacritics: bool = True
    prefer_formal_style: bool = True

    # Generation settings
    streaming: bool = True
    confidence_threshold: float = 0.5

    # Fallback models
    fallback_models: List[str] = field(
        default_factory=lambda: ["qwen2.5:7b-instruct", "llama3.1:8b", "mistral:7b"]
    )


@dataclass
class GenerationRequest:
    """Request for text generation - pure data structure."""

    prompt: str
    context: List[str]
    query: str
    query_type: str = "general"
    language: str = "hr"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Response from text generation - pure data structure."""

    text: str
    model: str
    tokens_used: int
    generation_time: float
    confidence: float
    metadata: Dict[str, Any]
    language: str = "hr"


@dataclass
class HttpResponse:
    """HTTP response wrapper for testing."""

    status_code: int
    content: bytes
    json_data: Optional[Dict[str, Any]] = None

    def json(self) -> Dict[str, Any]:
        """Get JSON data from response."""
        if self.json_data is not None:
            return self.json_data
        return json.loads(self.content.decode())

    def raise_for_status(self) -> None:
        """Raise exception for bad status codes."""
        if self.status_code >= 400:
            raise HttpError(f"HTTP {self.status_code}", self.status_code)


class HttpError(Exception):
    """HTTP error wrapper."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class ConnectionError(Exception):
    """Connection error wrapper."""

    pass


# Protocols for dependency injection
class HttpClient(Protocol):
    """HTTP client interface for dependency injection."""

    async def get(self, url: str, timeout: float = 30.0) -> HttpResponse:
        """Make GET request."""
        ...

    async def post(
        self, url: str, json_data: Dict[str, Any], timeout: float = 30.0
    ) -> HttpResponse:
        """Make POST request."""
        ...

    async def stream_post(
        self, url: str, json_data: Dict[str, Any], timeout: float = 30.0
    ) -> List[str]:
        """Make streaming POST request, return lines."""
        ...

    async def close(self) -> None:
        """Close client."""
        ...


class LanguageConfigProvider(Protocol):
    """Language-specific configuration provider."""

    def get_formal_prompts(self, language: str) -> Dict[str, str]:
        """Get formal prompt templates for language."""
        ...

    def get_error_template(self, language: str) -> str:
        """Get error message template for language."""
        ...

    def get_confidence_settings(self, language: str) -> Dict[str, Any]:
        """Get confidence calculation settings for language."""
        ...


# Pure functions for business logic
def build_complete_prompt(
    query: str, context: Optional[List[str]] = None, system_prompt: Optional[str] = None
) -> str:
    """
    Build complete prompt from components.

    Args:
        query: User query
        context: Retrieved document chunks
        system_prompt: System instructions

    Returns:
        Complete formatted prompt
    """
    parts = []

    # Add system prompt if provided
    if system_prompt:
        parts.append(f"System: {system_prompt}")

    # Add context if provided
    if context:
        context_text = "\n\n".join(context)
        parts.append(f"Context:\n{context_text}")

    # Add the main query
    parts.append(f"Question: {query}")
    parts.append("Answer:")

    return "\n\n".join(parts)


def enhance_prompt_with_formality(
    prompt: str, formal_prompts: Dict[str, str], use_formal_style: bool = True
) -> str:
    """
    Enhance prompt with formal language instructions.

    Args:
        prompt: Original prompt
        formal_prompts: Formal prompt templates from config
        use_formal_style: Whether to add formal instructions

    Returns:
        Enhanced prompt
    """
    if not use_formal_style or "formal_instruction" not in formal_prompts:
        return prompt

    formal_instruction = formal_prompts["formal_instruction"]
    return f"{formal_instruction}\n\n{prompt}"


def create_ollama_request(prompt: str, config: OllamaConfig) -> Dict[str, Any]:
    """
    Create Ollama API request from prompt and config.

    Args:
        prompt: Complete prompt text
        config: Ollama configuration

    Returns:
        Request dictionary for Ollama API
    """
    return {
        "model": config.model,
        "prompt": prompt,
        "stream": config.streaming,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
            "top_p": config.top_p,
            "top_k": config.top_k,
        },
    }


def parse_streaming_response(json_lines: List[str]) -> str:
    """
    Parse streaming JSON lines into complete text.

    Args:
        json_lines: List of JSON line strings

    Returns:
        Complete generated text
    """
    generated_chunks = []

    for line in json_lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if "response" not in data:
                continue
            chunk = data["response"]
            if chunk:
                generated_chunks.append(chunk)
        except json.JSONDecodeError:
            # Skip invalid JSON lines
            continue

    return "".join(generated_chunks)


def parse_non_streaming_response(response_data: Dict[str, Any]) -> str:
    """
    Parse non-streaming response into text.

    Args:
        response_data: Response JSON data

    Returns:
        Generated text
    """
    if "response" not in response_data:
        raise ValueError("Missing 'response' in response data")
    return response_data["response"]


def calculate_generation_confidence(
    generated_text: str, request: GenerationRequest, confidence_settings: Dict[str, Any]
) -> float:
    """
    Calculate confidence score for generated text.

    Args:
        generated_text: Generated text to evaluate
        request: Original generation request
        confidence_settings: Language-specific confidence settings

    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 0.5  # Base confidence

    # Length check
    text_length = len(generated_text.strip())
    if text_length < 10:
        confidence -= 0.3
    elif text_length > 50:
        confidence += 0.1

    # Language content check
    if request.language:
        language_score = detect_language_content(generated_text, request.language)
        confidence += language_score * 0.3

    # Query relevance (simple keyword check)
    query_words = set(request.query.lower().split())
    text_words = set(generated_text.lower().split())
    overlap = len(query_words.intersection(text_words))
    if len(query_words) > 0:
        relevance = overlap / len(query_words)
        confidence += relevance * 0.2

    # Context usage check
    if request.context:
        context_text = " ".join(request.context).lower()
        context_words = set(context_text.split())
        context_overlap = len(context_words.intersection(text_words))
        if len(context_words) > 0:
            context_usage = min(context_overlap / len(context_words), 0.3)
            confidence += context_usage

    # Error indicators from config
    if "error_phrases" not in confidence_settings:
        raise ValueError("Missing 'error_phrases' in confidence_settings")
    error_phrases = confidence_settings["error_phrases"]
    if any(phrase in generated_text.lower() for phrase in error_phrases):
        confidence -= 0.2

    return max(0.0, min(1.0, confidence))


def estimate_token_count(text: str) -> int:
    """
    Estimate token count from text (simple word-based approximation).

    Args:
        text: Text to analyze

    Returns:
        Estimated token count
    """
    return len(text.split()) if text else 0


def extract_model_list(api_response: Dict[str, Any]) -> List[str]:
    """
    Extract model names from Ollama API tags response.

    Args:
        api_response: Response from /api/tags endpoint

    Returns:
        List of model names
    """
    if "models" not in api_response:
        raise ValueError("Missing 'models' in API response")
    models_data = api_response["models"]
    result = []
    for model in models_data:
        if "name" not in model:
            continue
        result.append(model["name"])
    return result


def check_model_availability(model_name: str, available_models: List[str]) -> bool:
    """
    Check if a model is available in the list.

    Args:
        model_name: Model to check
        available_models: List of available models

    Returns:
        True if model is available
    """
    return model_name in available_models


def create_error_response(
    error_message: str,
    model: str,
    start_time: float,
    error_template: str,
    query_type: str = "general",
) -> GenerationResponse:
    """
    Create error response with proper formatting.

    Args:
        error_message: Error message to include
        model: Model name being used
        start_time: Request start time
        error_template: Language-specific error template
        query_type: Type of query that failed

    Returns:
        Error response object
    """
    error_text = error_template.format(error=error_message)

    return GenerationResponse(
        text=error_text,
        model=model,
        tokens_used=0,
        generation_time=time.time() - start_time,
        confidence=0.0,
        metadata={"error": error_message, "query_type": query_type},
        language="hr",
    )


class MultilingualOllamaClient:
    """
    Testable Ollama client with dependency injection for HTTP operations.

    All business logic is in pure functions, HTTP operations are injected.
    This enables flexible and configurable client architecture.
    """

    def __init__(
        self,
        config: OllamaConfig,
        http_client: Optional[HttpClient] = None,
        language_config_provider: Optional[LanguageConfigProvider] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize client with injected dependencies.

        Args:
            config: Ollama configuration
            http_client: HTTP client for API calls
            language_config_provider: Language-specific configuration
            logger: Logger instance
        """
        self.config = config
        self.http_client = http_client or self._create_default_http_client()
        self.language_config_provider = (
            language_config_provider or self._create_default_language_provider()
        )
        self.logger = logger or logging.getLogger(__name__)

    def _create_default_http_client(self) -> HttpClient:
        """Create default HTTP client implementation."""
        from .http_clients import AsyncHttpxClient

        return AsyncHttpxClient()

    def _create_default_language_provider(self) -> LanguageConfigProvider:
        """Create default language config provider."""
        from .language_providers import DefaultLanguageProvider

        return DefaultLanguageProvider()

    async def generate_text_async(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using Ollama with async support.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        start_time = time.time()

        # Step 1: Check if service is available
        if not await self.is_service_available():
            raise ConnectionError("Ollama service is not available")

        # Step 2: Get language-specific configuration
        formal_prompts = self.language_config_provider.get_formal_prompts(request.language)

        # Get appropriate system prompt based on query type
        system_prompt = self._get_system_prompt_for_query(request, formal_prompts)

        # Step 3: Build and enhance prompt
        base_prompt = build_complete_prompt(request.query, request.context, system_prompt)

        enhanced_prompt = enhance_prompt_with_formality(
            base_prompt, formal_prompts, self.config.prefer_formal_style
        )

        # Step 4: Create API request
        ollama_request = create_ollama_request(enhanced_prompt, self.config)

        # Step 5: Make API call (streaming or non-streaming)
        url = f"{self.config.base_url}/api/generate"

        if self.config.streaming:
            json_lines = await self.http_client.stream_post(
                url, ollama_request, self.config.timeout
            )
            generated_text = parse_streaming_response(json_lines)
        else:
            response = await self.http_client.post(url, ollama_request, self.config.timeout)
            generated_text = parse_non_streaming_response(response.json())

        # Step 6: Apply text preservation if needed
        if self.config.preserve_diacritics:
            generated_text = preserve_text_encoding(generated_text)

        # Step 7: Calculate confidence
        confidence_settings = self.language_config_provider.get_confidence_settings(
            request.language
        )
        confidence = calculate_generation_confidence(generated_text, request, confidence_settings)

        # Step 8: Build response
        metadata = {
            "query_type": request.query_type,
            "language": request.language,
            "context_length": len(" ".join(request.context)),
            "streaming": self.config.streaming,
            "formal_style": self.config.prefer_formal_style,
        }
        if request.metadata:
            metadata.update(request.metadata)

        return GenerationResponse(
            text=generated_text,
            model=self.config.model,
            tokens_used=estimate_token_count(generated_text),
            generation_time=time.time() - start_time,
            confidence=confidence,
            metadata=metadata,
            language=request.language,
        )

    async def is_service_available(self) -> bool:
        """
        Check if Ollama service is available.

        Returns:
            True if service is available
        """
        url = f"{self.config.base_url}/api/tags"
        response = await self.http_client.get(url, timeout=5.0)
        return response.status_code == 200

    async def get_available_models(self) -> List[str]:
        """
        Get list of available models.

        Returns:
            List of model names
        """
        url = f"{self.config.base_url}/api/tags"
        response = await self.http_client.get(url, self.config.timeout)
        return extract_model_list(response.json())

    async def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """
        Check if a specific model is available.

        Args:
            model_name: Model to check (defaults to configured model)

        Returns:
            True if model is available
        """
        target_model = model_name or self.config.model
        available_models = await self.get_available_models()
        return check_model_availability(target_model, available_models)

    async def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model_name: Model to pull (defaults to configured model)

        Returns:
            True if model was pulled successfully
        """
        target_model = model_name or self.config.model

        # Check if already available
        if await self.is_model_available(target_model):
            return True

        self.logger.info(f"Pulling model {target_model}...")

        url = f"{self.config.base_url}/api/pull"
        payload = {"name": target_model}

        await self.http_client.post(url, payload, self.config.timeout)
        self.logger.info(f"Successfully pulled {target_model}")
        return True

    def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        language: str = "hr",
    ) -> str:
        """
        Synchronous wrapper for generate_text_async.

        Args:
            prompt: User prompt
            context: Context documents
            system_prompt: System instructions
            language: Response language

        Returns:
            Generated response text
        """
        # Build complete prompt
        complete_prompt = build_complete_prompt(prompt, context, system_prompt)

        request = GenerationRequest(
            prompt=complete_prompt,
            context=context or [],
            query=prompt,
            query_type="general",
            language=language,
        )

        # Run async function in sync context
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(self.generate_text_async(request))
        return response.text

    async def health_check(self) -> bool:
        """
        Perform health check on Ollama service.

        Returns:
            True if service is healthy
        """
        return await self.is_service_available()

    def _get_system_prompt_for_query(
        self, request: GenerationRequest, formal_prompts: Dict[str, str]
    ) -> str:
        """
        Select appropriate system prompt based on query type and language using configuration.

        Args:
            request: Generation request with query and language
            formal_prompts: Language-specific formal prompts configuration

        Returns:
            System prompt optimized for the query type
        """
        try:
            # Get language-specific prompts configuration
            from ..utils.config_loader import get_language_specific_config

            prompts_config = get_language_specific_config("prompts", request.language)

            # Get keyword categories from configuration
            keywords_config = prompts_config["keywords"]
            query_lower = request.query.lower()

            # Check each keyword category from configuration
            for category, keywords in keywords_config.items():
                if isinstance(keywords, list) and any(
                    keyword in query_lower for keyword in keywords
                ):
                    # Map category to system prompt name
                    system_prompt_key = self._get_system_prompt_key(category)
                    if system_prompt_key in prompts_config:
                        return prompts_config[system_prompt_key]

            # Default: Use question answering system for general queries
            return prompts_config["question_answering_system"]

        except Exception as e:
            self.logger.error(
                f"Failed to get language-specific system prompt for language '{request.language}': {e}"
            )
            raise ConfigError(
                f"System prompt configuration missing or invalid for language '{request.language}'"
            ) from e

    def _get_system_prompt_key(self, category: str) -> str:
        """
        Map keyword category to system prompt configuration key.

        Args:
            category: Keyword category from configuration

        Returns:
            System prompt configuration key
        """
        # Standard mapping from category to system prompt key
        category_mapping = {
            "tourism": "tourism_system",
            "comparison": "comparison_system",
            "explanation": "explanatory_system",
            "factual": "factual_qa_system",
            "summary": "summarization_system",
            "business": "business_system",
        }
        return category_mapping.get(category, "question_answering_system")

    async def close(self) -> None:
        """Close HTTP client."""
        await self.http_client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Factory function for convenient creation
def create_ollama_client(
    config: Optional[OllamaConfig] = None,
    http_client: Optional[HttpClient] = None,
    language_config_provider: Optional[LanguageConfigProvider] = None,
    logger: Optional[logging.Logger] = None,
) -> MultilingualOllamaClient:
    """
    Factory function to create configured Ollama client.

    Args:
        config: Ollama configuration (creates default if None)
        http_client: HTTP client implementation
        language_config_provider: Language configuration provider
        logger: Logger instance

    Returns:
        Configured MultilingualOllamaClient
    """
    if config is None:
        config = OllamaConfig()

    return MultilingualOllamaClient(
        config=config,
        http_client=http_client,
        language_config_provider=language_config_provider,
        logger=logger,
    )
