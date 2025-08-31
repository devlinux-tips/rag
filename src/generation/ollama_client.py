"""
Ollama client for local LLM integration in Croatian RAG system.
Handles communication with Ollama API for answer generation with Croatian language support.
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import requests

from ..utils.croatian_utils import detect_croatian_content, preserve_croatian_encoding


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""

    base_url: str = "http://localhost:11434"
    model: str = "jobautomation/openeurollm-croatian:latest"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    top_k: int = 64  # 40
    timeout: float = 120.0

    # Croatian-specific settings
    preserve_diacritics: bool = True
    prefer_formal_style: bool = True
    include_cultural_context: bool = True


@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    context: List[str]
    query: str
    query_type: str = "general"
    language: str = "hr"
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

    @property
    def has_croatian_content(self) -> bool:
        """Check if response contains Croatian content."""
        return detect_croatian_content(self.text) > 0.3


class OllamaClient:
    """Enhanced client for interacting with Ollama API for Croatian text generation."""

    def __init__(self, config: OllamaConfig = None):
        """
        Initialize Ollama client with Croatian language support.

        Args:
            config: Configuration object for Ollama settings
        """
        self.config = config or OllamaConfig()
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
            models = [model["name"] for model in response.get("models", [])]
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

    async def generate_text_async(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Ollama with async support and Croatian optimization (streaming).
        TODO: maybe add toggle between streaming/non-streaming via a flag in request or config (instead of hardcoding)? That way you can choose per-use-case.
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
                prompt = f"Odgovaraj uvijek u formalnom stilu.\n\n{prompt}"
            if self.config.include_cultural_context:
                prompt = f"{prompt}\n\nAko je prikladno, uključi kulturni kontekst u odgovoru."

            # Step 3: Prepare request for streaming
            ollama_request = {
                "model": self.config.model,  # "jobautomation/openeurollm-croatian:latest"
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                },
            }
            self.logger.debug(f"Sending streaming request to Ollama: {ollama_request}")

            # Step 4: Collect streamed response
            generated_chunks = []
            async with self._async_client.stream(
                "POST",
                f"http://127.0.0.1:11434/api/generate",  # use 127.0.0.1 instead of localhost
                json=ollama_request,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            generated_chunks.append(chunk)
                    except Exception as parse_err:
                        self.logger.warning(
                            f"Failed to parse stream chunk: {parse_err} | raw={line}"
                        )

            generated_text = "".join(generated_chunks)

            # Step 5: Preserve Croatian encoding if needed
            if self.config.preserve_diacritics and request.language == "hr":
                generated_text = preserve_croatian_encoding(generated_text)

            # Step 6: Confidence calculation
            confidence = self._calculate_confidence(generated_text, request)

            # Step 7: Metadata (streaming mode gives limited stats)
            metadata = {
                "query_type": request.query_type,
                "language": request.language,
                "context_length": len(" ".join(request.context)),
                "streaming": True,
                "formal_style": self.config.prefer_formal_style,
                "cultural_context": self.config.include_cultural_context,
            }
            if request.metadata:
                metadata.update(request.metadata)

            return GenerationResponse(
                text=generated_text,
                model=self.config.model,
                tokens_used=len(generated_chunks),  # rough estimate
                generation_time=time.time() - start_time,
                confidence=confidence,
                metadata=metadata,
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Generation failed ({type(e).__name__}): {e}")
            self.logger.debug(f"Traceback:\n{tb}")
            error_msg = f"{type(e).__name__}: {str(e)}"

            return GenerationResponse(
                text=f"Greška u generiranju odgovora: {error_msg}",
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
            language="hr",
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
            response = self._make_request("api/tags", {})
            return [model["name"] for model in response.get("models", [])]

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

    def _calculate_confidence(self, generated_text: str, request: GenerationRequest) -> float:
        """Calculate confidence score for generated text."""
        confidence = 0.5  # Base confidence

        # Length check
        if len(generated_text.strip()) < 10:
            confidence -= 0.3
        elif len(generated_text.strip()) > 50:
            confidence += 0.1

        # Croatian content check
        if request.language == "hr":
            croatian_score = detect_croatian_content(generated_text)
            confidence += croatian_score * 0.3

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

        # Error indicators
        error_phrases = ["greška", "error", "ne znam", "ne mogu", "sorry"]
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
    model: str = "qwen2.5:7b-instruct",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
) -> OllamaClient:
    """
    Factory function to create configured Ollama client.

    Args:
        model: Model name to use
        base_url: Ollama API base URL
        temperature: Generation temperature

    Returns:
        Configured OllamaClient instance
    """
    config = OllamaConfig(model=model, base_url=base_url, temperature=temperature)
    return OllamaClient(config)
