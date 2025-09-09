"""
Ollama client for local LLM integration in multilingual RAG system.
Handles communication with Ollama API for answer generation with language support.
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

if TYPE_CHECKING:
    from ..utils.config_protocol import ConfigProvider

import httpx
import requests

from ..preprocessing.cleaners import (detect_language_content,
                                      preserve_text_encoding)
from ..utils.config_loader import (get_generation_config, get_language_shared,
                                   get_language_specific_config,
                                   get_ollama_config)
from ..utils.error_handler import create_config_loader

# Create specialized config loaders
load_generation_config = create_config_loader("config/config.toml", __name__)
# Language-specific config will be loaded dynamically based on language parameter


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""

    # Server settings
    base_url: str = field(default="http://localhost:11434")
    timeout: float = field(default=120.0)

    # Model settings
    model: str = field(default="llama3.1:8b")
    temperature: float = field(default=0.7)
    max_tokens: int = field(default=2000)
    top_p: float = field(default=0.9)
    top_k: int = field(default=64)

    # Language-specific settings
    preserve_diacritics: bool = field(default=True)
    prefer_formal_style: bool = field(default=True)
    # Note: Cultural context handled by language-specific prompts

    # Generation settings
    streaming: bool = field(default=True)
    confidence_threshold: float = field(default=0.5)

    # Fallback models
    fallback_models: List[str] = field(
        default_factory=lambda: ["qwen2.5:7b-instruct", "llama3.1:8b", "mistral:7b"]
    )

    @classmethod
    def from_config(
        cls,
        config_path: Optional[str] = None,
        language: str = "hr",
        config_dict: Optional[Dict[str, Any]] = None,
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "OllamaConfig":
        """Load configuration from dictionary or config provider."""
        if config_dict:
            # Direct config provided
            ollama_config = config_dict.get("ollama", config_dict)
            language_config = config_dict.get("language_specific", {})
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get configs through provider
            full_config = config_provider.load_config("config")
            ollama_config = full_config["ollama"]

            # Get language-specific config
            language_config = config_provider.get_language_specific_config(
                "generation", language
            )

        return cls(
            base_url=ollama_config["base_url"],
            timeout=ollama_config["timeout"],
            model=ollama_config["model"],
            temperature=ollama_config["temperature"],
            max_tokens=ollama_config["max_tokens"],
            top_p=ollama_config["top_p"],
            top_k=ollama_config["top_k"],
            preserve_diacritics=ollama_config["preserve_diacritics"],
            prefer_formal_style=language_config.get("formality_level", "casual")
            == "polite",
            streaming=ollama_config["stream"],
            confidence_threshold=0.5,
            fallback_models=[ollama_config["fallback_model"]],
        )


@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    context: List[str]
    query: str
    query_type: str = "general"
    language: str = "hr"  # Default language, can be overridden
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    model: str
    tokens_used: int
    generation_time: float
    confidence: float
    metadata: Dict[str, Any]
    language: str = "hr"  # Language of the response

    def has_language_content(self, language: str = None) -> bool:
        """Check if response contains content in specified language."""
        target_language = language or self.language
        return detect_language_content(self.text, target_language) > 0.3


class OllamaClient:
    """Enhanced client for interacting with Ollama API for multilingual text generation."""

    def __init__(self, config: Optional[OllamaConfig] = None, language: str = "hr"):
        """
        Initialize Ollama client with language support.

        Args:
            config: Configuration object for Ollama settings. If None, loads from TOML.
            language: Language code for language-specific behavior
        """
        self.language = language
        self.config = config or OllamaConfig.from_config(language=language)
        self.logger = logging.getLogger(__name__)
        self._async_client = None
        self._model_info = None

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API.

        Args:
            endpoint: API endpoint path
            payload: Request payload

        Returns:
            API response as dictionary

        Raises:
            ConnectionError: If Ollama is not running
            requests.RequestException: For other API errors
        """
        url = f"{self.config.base_url}/{endpoint}"

        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()

        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Make sure Ollama is running with: ollama serve"
            )
        except requests.RequestException as e:
            self.logger.error(f"Ollama API request failed: {e}")
            raise

    def is_model_available(self) -> bool:
        """
        Check if the configured model is available in Ollama.

        Returns:
            True if model is available, False otherwise
        """
        try:
            response = self._make_request("api/tags", {})
            models = [model["name"] for model in response["models"]]
            return self.config.model in models

        except Exception as e:
            self.logger.error(f"Failed to check model availability: {e}")
            return False

    def pull_model(self) -> bool:
        """
        Download the configured model if not available.

        Returns:
            True if model was pulled successfully, False otherwise
        """
        if self.is_model_available():
            return True

        try:
            self.logger.info(f"Pulling model {self.config.model}...")
            payload = {"name": self.config.model}
            self._make_request("api/pull", payload)
            self.logger.info(f"Successfully pulled {self.config.model}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to pull model {self.config.model}: {e}")
            return False

    async def generate_text_async(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """Generate text using Ollama with async support and language optimization.
        Uses streaming mode based on configuration settings.
        """
        start_time = time.time()

        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.config.timeout)

        try:
            # Step 1: Availability check
            if not await self._is_available_async():
                raise ConnectionError("Ollama service is not available")

            # Step 2: Prepare prompt with config-driven enrichments
            prompt = request.prompt
            if self.config.prefer_formal_style:
                language_config = get_language_specific_config("prompts", self.language)
                formal_prompts = language_config["formal"]
                formal_instruction = formal_prompts["formal_instruction"]
                if formal_instruction:
                    prompt = f"{formal_instruction}\n\n{prompt}"
            # Note: Cultural context handled by language-specific prompt templates

            # Step 3: Prepare request (streaming based on config)
            ollama_request = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": self.config.streaming,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                },
            }
            self.logger.debug(f"Sending request to Ollama: {ollama_request}")

            # Step 4: Handle response based on streaming setting
            if self.config.streaming:
                generated_text = await self._handle_streaming_response(ollama_request)
            else:
                generated_text = await self._handle_non_streaming_response(
                    ollama_request
                )

            # Step 5: Preserve encoding if needed
            if self.config.preserve_diacritics and request.language:
                generated_text = preserve_text_encoding(generated_text)

            # Step 6: Confidence calculation
            confidence = self._calculate_confidence(generated_text, request)

            # Step 7: Metadata
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
                tokens_used=len(generated_text.split())
                if generated_text
                else 0,  # rough estimate
                generation_time=time.time() - start_time,
                confidence=confidence,
                metadata=metadata,
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Generation failed ({type(e).__name__}): {e}")
            self.logger.debug(f"Traceback:\n{tb}")
            error_msg = f"{type(e).__name__}: {str(e)}"

            # Get error message template from language config
            language_config = get_language_specific_config("prompts", self.language)
            error_template = language_config.get(
                "error_message_template", "An error occurred: {error}"
            )
            error_text = error_template.format(error=error_msg)

            return GenerationResponse(
                text=error_text,
                model=self.config.model,
                tokens_used=0,
                generation_time=time.time() - start_time,
                confidence=0.0,
                metadata={"error": error_msg, "query_type": request.query_type},
            )

    def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper for generate_text_async (backward compatibility)."""
        request = GenerationRequest(
            prompt=self._build_prompt(prompt, context, system_prompt),
            context=context or [],
            query=prompt,
            query_type="general",
            language=self.language,
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

    def _build_prompt(
        self,
        query: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build complete prompt from query, context, and system instructions.

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

    def get_available_models(self) -> List[str]:
        """
        Get list of available models in Ollama.

        Returns:
            List of model names
        """
        try:
            # Use GET request for /api/tags endpoint (not POST like other endpoints)
            url = f"{self.config.base_url}/api/tags"
            response = requests.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data["models"]]

        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []

    async def _is_available_async(self) -> bool:
        """Check if Ollama service is available (async)."""
        try:
            if self._async_client is None:
                self._async_client = httpx.AsyncClient(timeout=self.config.timeout)

            response = await self._async_client.get(f"{self.config.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Ollama service not available: {e}")
            return False

    async def _handle_streaming_response(self, ollama_request: Dict[str, Any]) -> str:
        """Handle streaming response from Ollama."""
        generated_chunks = []
        async with self._async_client.stream(
            "POST",
            f"{self.config.base_url}/api/generate",
            json=ollama_request,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    chunk = data["response"]
                    if chunk:
                        generated_chunks.append(chunk)
                except Exception as parse_err:
                    self.logger.warning(
                        f"Failed to parse stream chunk: {parse_err} | raw={line}"
                    )
        return "".join(generated_chunks)

    async def _handle_non_streaming_response(
        self, ollama_request: Dict[str, Any]
    ) -> str:
        """Handle non-streaming response from Ollama."""
        response = await self._async_client.post(
            f"{self.config.base_url}/api/generate", json=ollama_request
        )
        response.raise_for_status()
        data = response.json()
        return data["response"]

    def _calculate_confidence(
        self, generated_text: str, request: GenerationRequest
    ) -> float:
        """Calculate confidence score for generated text."""
        confidence = 0.5  # Base confidence

        # Length check
        if len(generated_text.strip()) < 10:
            confidence -= 0.3
        elif len(generated_text.strip()) > 50:
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
        language_config = get_language_specific_config("confidence", self.language)
        error_phrases = language_config.get(
            "error_phrases", ["error", "failed", "sorry"]
        )
        if any(phrase in generated_text.lower() for phrase in error_phrases):
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def health_check(self) -> bool:
        """
        Check if Ollama service is healthy.

        Returns:
            True if service is running, False otherwise
        """
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            return response.status_code == 200

        except Exception:
            return False

    async def close(self):
        """Close async client."""
        if self._async_client:
            await self._async_client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


def create_ollama_client(
    config_path: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    language: str = "hr",
) -> OllamaClient:
    """
    Factory function to create configured Ollama client.

    Args:
        config_path: Unused, kept for backwards compatibility
        model: Model name to use (overrides config)
        base_url: Ollama API base URL (overrides config)
        temperature: Generation temperature (overrides config)
        language: Language code for language-specific behavior

    Returns:
        Configured OllamaClient instance
    """
    # Load config from TOML files
    config = OllamaConfig.from_config(language=language)

    # Override with provided parameters
    if model is not None:
        config.model = model
    if base_url is not None:
        config.base_url = base_url
    if temperature is not None:
        config.temperature = temperature

    return OllamaClient(config, language=language)
