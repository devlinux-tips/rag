"""
Ollama client for local LLM integration providing text generation and streaming.
Handles HTTP communication with Ollama API for language model inference
with configurable parameters and error handling.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..preprocessing.cleaners import detect_language_content_with_config
from ..utils.config_loader import ConfigError
from ..utils.config_models import OllamaConfig
from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_data_transformation,
    log_decision_point,
    log_error_context,
    log_performance_metric,
)

# OllamaConfig imported from config_models.py


@dataclass
class GenerationRequest:
    """Request for text generation - pure data structure."""

    prompt: str
    context: list[str]
    query: str
    query_type: str = "general"
    language: str = "hr"
    metadata: dict[str, Any] | None = None


@dataclass
class GenerationResponse:
    """Response from text generation - pure data structure."""

    text: str
    model: str
    tokens_used: int
    generation_time: float
    confidence: float
    metadata: dict[str, Any]
    language: str = "hr"


@dataclass
class HttpResponse:
    """HTTP response wrapper for testing."""

    status_code: int
    content: bytes
    json_data: dict[str, Any] | None = None

    def json(self) -> dict[str, Any]:
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

    async def post(self, url: str, json_data: dict[str, Any], timeout: float = 30.0) -> HttpResponse:
        """Make POST request."""
        ...

    async def stream_post(self, url: str, json_data: dict[str, Any], timeout: float = 30.0) -> list[str]:
        """Make streaming POST request, return lines."""
        ...

    async def close(self) -> None:
        """Close client."""
        ...


class LanguageConfigProvider(Protocol):
    """Language-specific configuration provider."""

    def get_formal_prompts(self, language: str) -> dict[str, str]:
        """Get formal prompt templates for language."""
        ...

    def get_error_template(self, language: str) -> str:
        """Get error message template for language."""
        ...

    def get_confidence_settings(self, language: str) -> dict[str, Any]:
        """Get confidence calculation settings for language."""
        ...


# Pure functions for business logic
def build_complete_prompt(query: str, context: list[str] | None = None, system_prompt: str | None = None) -> str:
    """
    Build complete prompt from components.

    Args:
        query: User query
        context: Retrieved document chunks
        system_prompt: System instructions

    Returns:
        Complete formatted prompt
    """
    logger = get_system_logger()
    log_component_start(
        "prompt_builder",
        "build_complete_prompt",
        query_length=len(query),
        context_count=len(context) if context else 0,
        has_system=system_prompt is not None,
    )

    parts = []

    # Add system prompt if provided
    if system_prompt:
        parts.append(f"System: {system_prompt}")
        logger.trace("prompt_builder", "build_complete_prompt", f"Added system prompt: {len(system_prompt)} chars")

    # Add context if provided
    if context:
        context_text = "\n\n".join(context)
        parts.append(f"Context:\n{context_text}")
        logger.debug(
            "prompt_builder",
            "build_complete_prompt",
            f"Added context: {len(context)} chunks, {len(context_text)} chars",
        )

    # Add the main query
    parts.append(f"Question: {query}")
    parts.append("Answer:")
    logger.trace("prompt_builder", "build_complete_prompt", f"Added query: {len(query)} chars")

    complete_prompt = "\n\n".join(parts)
    log_data_transformation(
        "prompt_builder", "assemble_parts", f"parts[{len(parts)}]", f"prompt[{len(complete_prompt)}]"
    )
    log_component_end("prompt_builder", "build_complete_prompt", f"Built prompt: {len(complete_prompt)} chars")
    return complete_prompt


def enhance_prompt_with_formality(prompt: str, formal_prompts: dict[str, str], use_formal_style: bool = True) -> str:
    """
    Enhance prompt with formal language instructions.

    Args:
        prompt: Original prompt
        formal_prompts: Formal prompt templates from config
        use_formal_style: Whether to add formal instructions

    Returns:
        Enhanced prompt
    """
    logger = get_system_logger()
    log_component_start(
        "prompt_enhancer",
        "enhance_formality",
        prompt_length=len(prompt),
        use_formal=use_formal_style,
        has_formal_templates=len(formal_prompts) > 0,
    )

    if not use_formal_style:
        logger.debug("prompt_enhancer", "enhance_formality", "Formal style disabled, returning original")
        log_component_end("prompt_enhancer", "enhance_formality", "No enhancement applied")
        return prompt

    if "formal_instruction" not in formal_prompts:
        logger.debug("prompt_enhancer", "enhance_formality", "No formal_instruction found in templates")
        log_component_end("prompt_enhancer", "enhance_formality", "No formal template available")
        return prompt

    formal_instruction = formal_prompts["formal_instruction"]
    enhanced = f"{formal_instruction}\n\n{prompt}"

    log_data_transformation("prompt_enhancer", "add_formality", f"prompt[{len(prompt)}]", f"enhanced[{len(enhanced)}]")
    logger.debug("prompt_enhancer", "enhance_formality", f"Added formal instruction: {len(formal_instruction)} chars")
    log_component_end("prompt_enhancer", "enhance_formality", f"Enhanced prompt: {len(enhanced)} chars")
    return enhanced


def create_ollama_request(prompt: str, config: OllamaConfig) -> dict[str, Any]:
    """
    Create Ollama API request from prompt and config.

    Args:
        prompt: Complete prompt text
        config: Ollama configuration

    Returns:
        Request dictionary for Ollama API
    """
    logger = get_system_logger()
    log_component_start(
        "request_builder",
        "create_ollama_request",
        model=config.model,
        prompt_length=len(prompt),
        streaming=config.stream,
    )

    # Standard Ollama format
    request_data = {
        "model": config.model,
        "prompt": prompt,
        "stream": config.stream,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
            "top_p": config.top_p,
            "top_k": config.top_k,
        },
    }
    logger.debug(
        "request_builder",
        "create_ollama_request",
        f"Ollama format: model={config.model}, temp={config.temperature}, "
        f"tokens={config.num_predict}, stream={config.stream}",
    )

    log_component_end("request_builder", "create_ollama_request", f"Created request for {config.model}")
    return request_data


def parse_streaming_response(json_lines: list[str]) -> str:
    """
    Parse streaming JSON lines into complete text for Ollama format.

    Args:
        json_lines: List of JSON line strings

    Returns:
        Complete generated text
    """
    logger = get_system_logger()
    log_component_start("response_parser", "parse_streaming", lines_count=len(json_lines))

    generated_chunks = []
    invalid_lines = 0
    empty_lines = 0

    for i, line in enumerate(json_lines):
        if not line.strip():
            empty_lines += 1
            continue

        try:
            data = json.loads(line)
            if "response" not in data:
                logger.trace("response_parser", "parse_streaming", f"Ollama line {i}: no 'response' field")
                continue

            chunk = data["response"]
            if chunk:
                generated_chunks.append(chunk)
                logger.trace("response_parser", "parse_streaming", f"Ollama chunk {i}: '{chunk[:50]}...'")

        except json.JSONDecodeError as e:
            invalid_lines += 1
            logger.debug("response_parser", "parse_streaming", f"Ollama line {i}: invalid JSON - {str(e)}")
            continue

    complete_text = "".join(generated_chunks)
    log_data_transformation(
        "response_parser", "combine_chunks", f"chunks[{len(generated_chunks)}]", f"text[{len(complete_text)}]"
    )

    logger.debug(
        "response_parser",
        "parse_streaming",
        f"Generated chunks: {len(generated_chunks)}, Total chars: {len(complete_text)}",
    )

    if invalid_lines > 0 or empty_lines > 0:
        logger.debug(
            "response_parser",
            "parse_streaming",
            f"Processing stats: {len(generated_chunks)} chunks, skipped: {invalid_lines} invalid, {empty_lines} empty",
        )

    if len(generated_chunks) == 0:
        logger.error("response_parser", "parse_streaming", f"NO CHUNKS EXTRACTED! Total lines: {len(json_lines)}")

    log_component_end(
        "response_parser", "parse_streaming", f"Parsed {len(generated_chunks)} chunks into {len(complete_text)} chars"
    )
    return complete_text


def parse_non_streaming_response(response_data: dict[str, Any]) -> str:
    """
    Parse non-streaming response into text for Ollama format.

    Args:
        response_data: Response JSON data

    Returns:
        Generated text
    """
    logger = get_system_logger()
    log_component_start("response_parser", "parse_non_streaming", response_keys=list(response_data.keys()))

    # Ollama format: {"response": "text"}
    if "response" not in response_data:
        logger.error(
            "response_parser",
            "parse_non_streaming",
            f"Missing 'response' field in Ollama response. Available keys: {list(response_data.keys())}",
        )
        raise ValueError("Missing 'response' in Ollama response data")

    text = response_data["response"]

    logger.debug("response_parser", "parse_non_streaming", f"Extracted response text: {len(text)} chars")
    log_component_end("response_parser", "parse_non_streaming", f"Parsed non-streaming response: {len(text)} chars")
    return text


def calculate_generation_confidence(
    generated_text: str, request: GenerationRequest, confidence_settings: dict[str, Any]
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
    logger = get_system_logger()
    log_component_start(
        "confidence_calculator",
        "calculate_confidence",
        text_length=len(generated_text),
        language=request.language,
        has_context=len(request.context) > 0,
    )

    confidence = 0.5  # Base confidence
    logger.debug("confidence_calculator", "calculate_confidence", f"Starting with base confidence: {confidence}")

    # Length check
    text_length = len(generated_text.strip())
    if text_length < 10:
        confidence -= 0.3
        log_decision_point("confidence_calculator", "calculate_confidence", f"short_text={text_length}", "penalty=-0.3")
    elif text_length > 50:
        confidence += 0.1
        log_decision_point(
            "confidence_calculator", "calculate_confidence", f"adequate_length={text_length}", "bonus=+0.1"
        )

    # Language content check
    if request.language:
        language_score = detect_language_content_with_config(generated_text, request.language)
        language_bonus = language_score * 0.3
        confidence += language_bonus
        log_decision_point(
            "confidence_calculator",
            "calculate_confidence",
            f"language_score={language_score:.3f}",
            f"bonus=+{language_bonus:.3f}",
        )

    # Query relevance (simple keyword check)
    query_words = set(request.query.lower().split())
    text_words = set(generated_text.lower().split())
    overlap = len(query_words.intersection(text_words))
    if len(query_words) > 0:
        relevance = overlap / len(query_words)
        relevance_bonus = relevance * 0.2
        confidence += relevance_bonus
        log_decision_point(
            "confidence_calculator",
            "calculate_confidence",
            f"query_relevance={relevance:.3f}",
            f"bonus=+{relevance_bonus:.3f}",
        )

    # Context usage check
    if request.context:
        context_text = " ".join(request.context).lower()
        context_words = set(context_text.split())
        context_overlap = len(context_words.intersection(text_words))
        if len(context_words) > 0:
            context_usage = min(context_overlap / len(context_words), 0.3)
            confidence += context_usage
            log_decision_point(
                "confidence_calculator",
                "calculate_confidence",
                f"context_usage={context_usage:.3f}",
                f"bonus=+{context_usage:.3f}",
            )

    # Error indicators from config
    if "error_phrases" not in confidence_settings:
        logger.error("confidence_calculator", "calculate_confidence", "Missing 'error_phrases' in confidence_settings")
        raise ValueError("Missing 'error_phrases' in confidence_settings")

    error_phrases = confidence_settings["error_phrases"]
    error_found = any(phrase in generated_text.lower() for phrase in error_phrases)
    if error_found:
        confidence -= 0.2
        log_decision_point("confidence_calculator", "calculate_confidence", "error_phrases_detected", "penalty=-0.2")

    final_confidence = max(0.0, min(1.0, confidence))
    log_performance_metric("confidence_calculator", "calculate_confidence", "final_score", final_confidence)
    log_component_end("confidence_calculator", "calculate_confidence", f"Calculated confidence: {final_confidence:.3f}")
    return final_confidence


def estimate_token_count(text: str) -> int:
    """
    Estimate token count from text (simple word-based approximation).

    Args:
        text: Text to analyze

    Returns:
        Estimated token count
    """
    return len(text.split()) if text else 0


def extract_model_list(api_response: dict[str, Any]) -> list[str]:
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


def check_model_availability(model_name: str, available_models: list[str]) -> bool:
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
    error_message: str, model: str, start_time: float, error_template: str, query_type: str = "general"
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
        http_client: HttpClient | None = None,
        language_config_provider: LanguageConfigProvider | None = None,
        logger=None,
    ):
        """
        Initialize client with injected dependencies.

        Args:
            config: Ollama configuration
            http_client: HTTP client for API calls
            language_config_provider: Language-specific configuration
            logger: Logger instance
        """
        get_system_logger()
        log_component_start(
            "ollama_client",
            "init",
            model=config.model,
            base_url=config.base_url,
            streaming=config.stream,
            has_http_client=http_client is not None,
        )

        self.config = config
        self.http_client = http_client or self._create_default_http_client()
        self.language_config_provider = language_config_provider or self._create_default_language_provider()
        self.logger = logger or get_system_logger()

        log_component_end("ollama_client", "init", f"Initialized Ollama client for {config.model}")

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
        logger = get_system_logger()
        log_component_start(
            "ollama_client",
            "generate_text_async",
            query_length=len(request.query),
            context_count=len(request.context),
            language=request.language,
            query_type=request.query_type,
        )

        start_time = time.time()

        try:
            # Step 1: Check if service is available
            logger.debug("ollama_client", "generate_text_async", "Checking service availability")
            if not await self.is_service_available():
                logger.error("ollama_client", "generate_text_async", "Ollama service is not available")
                raise ConnectionError("Ollama service is not available")

            # Step 2: Get language-specific configuration
            logger.debug("ollama_client", "generate_text_async", f"Getting formal prompts for {request.language}")
            formal_prompts = self.language_config_provider.get_formal_prompts(request.language)

            # Get appropriate system prompt based on query type
            system_prompt = self._get_system_prompt_for_query(request, formal_prompts)
            logger.debug(
                "ollama_client",
                "generate_text_async",
                f"Selected system prompt for {request.query_type}: {len(system_prompt)} chars",
            )

            # Step 3: Build and enhance prompt
            base_prompt = build_complete_prompt(request.query, request.context, system_prompt)

            enhanced_prompt = enhance_prompt_with_formality(
                base_prompt,
                formal_prompts,
                True,  # Default to formal style
            )
            log_performance_metric("ollama_client", "generate_text_async", "final_prompt_length", len(enhanced_prompt))

            # Step 4: Create API request
            ollama_request = create_ollama_request(enhanced_prompt, self.config)

            # Step 5: Make API call (streaming or non-streaming)
            url = f"{self.config.base_url}/api/generate"
            logger.info(
                "ollama_client",
                "generate_text_async",
                f"Generating with {self.config.model}, streaming={self.config.stream}",
            )

            if self.config.stream:
                logger.debug("ollama_client", "generate_text_async", "Making streaming request")
                json_lines = await self.http_client.stream_post(url, ollama_request, self.config.timeout)
                generated_text = parse_streaming_response(json_lines)
            else:
                logger.debug("ollama_client", "generate_text_async", "Making non-streaming request")
                http_response = await self.http_client.post(url, ollama_request, self.config.timeout)
                generated_text = parse_non_streaming_response(http_response.json())

            generation_time = time.time() - start_time
            log_performance_metric("ollama_client", "generate_text_async", "generation_time_sec", generation_time)
            logger.debug(
                "ollama_client",
                "generate_text_async",
                f"Generated {len(generated_text)} chars in {generation_time:.3f}s",
            )

            # Step 7: Calculate confidence
            confidence_settings = self.language_config_provider.get_confidence_settings(request.language)
            confidence = calculate_generation_confidence(generated_text, request, confidence_settings)

            # Step 8: Build response
            context_length = len(" ".join(request.context))
            metadata = {
                "query_type": request.query_type,
                "language": request.language,
                "context_length": context_length,
                "streaming": self.config.stream,
                "formal_style": True,
            }
            if request.metadata:
                metadata.update(request.metadata)

            tokens_used = estimate_token_count(generated_text)
            log_performance_metric("ollama_client", "generate_text_async", "tokens_generated", tokens_used)

            response = GenerationResponse(
                text=generated_text,
                model=self.config.model,
                tokens_used=tokens_used,
                generation_time=generation_time,
                confidence=confidence,
                metadata=metadata,
                language=request.language,
            )

            log_component_end(
                "ollama_client",
                "generate_text_async",
                f"Generated {len(generated_text)} chars, confidence={confidence:.3f}",
                tokens=tokens_used,
                confidence=confidence,
                generation_time=generation_time,
            )
            return response

        except Exception as e:
            log_error_context(
                "ollama_client",
                "generate_text_async",
                e,
                {
                    "query_length": len(request.query),
                    "context_count": len(request.context),
                    "language": request.language,
                    "model": self.config.model,
                    "base_url": self.config.base_url,
                },
            )
            raise

    async def is_service_available(self) -> bool:
        """
        Check if Ollama service is available.

        Returns:
            True if service is available
        """
        logger = get_system_logger()
        log_component_start("ollama_client", "is_service_available", base_url=self.config.base_url)

        try:
            url = f"{self.config.base_url}/api/version"
            logger.trace("ollama_client", "is_service_available", f"Checking service at {url}")
            response = await self.http_client.get(url, timeout=5.0)
            available = response.status_code == 200

            log_decision_point(
                "ollama_client", "is_service_available", f"status_code={response.status_code}", f"available={available}"
            )
            log_component_end(
                "ollama_client", "is_service_available", f"Service {'available' if available else 'unavailable'}"
            )
            return available

        except Exception as e:
            logger.debug("ollama_client", "is_service_available", f"Service check failed: {str(e)}")
            log_component_end("ollama_client", "is_service_available", "Service unavailable (error)")
            return False

    async def get_available_models(self) -> list[str]:
        """
        Get list of available models.

        Returns:
            List of model names
        """
        url = f"{self.config.base_url}/api/tags"
        response = await self.http_client.get(url, self.config.timeout)
        return extract_model_list(response.json())

    async def is_model_available(self, model_name: str | None = None) -> bool:
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

    async def pull_model(self, model_name: str | None = None) -> bool:
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

        self.logger.info("ollama_client", "pull_model", f"Pulling model {target_model}...")

        url = f"{self.config.base_url}/api/pull"
        payload = {"name": target_model}

        await self.http_client.post(url, payload, self.config.timeout)
        self.logger.info("ollama_client", "pull_model", f"Successfully pulled {target_model}")
        return True

    def generate_response(
        self, prompt: str, context: list[str] | None = None, system_prompt: str | None = None, language: str = "hr"
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
            prompt=complete_prompt, context=context or [], query=prompt, query_type="general", language=language
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

    def _get_system_prompt_for_query(self, request: GenerationRequest, formal_prompts: dict[str, str]) -> str:
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
                if isinstance(keywords, list) and any(keyword in query_lower for keyword in keywords):
                    # Map category to system prompt name
                    system_prompt_key = self._get_system_prompt_key(category)
                    if system_prompt_key in prompts_config:
                        return prompts_config[system_prompt_key]

            # Default: Use question answering system for general queries
            return prompts_config["question_answering_system"]

        except Exception as e:
            self.logger.error(
                "ollama_client",
                "get_system_prompt",
                f"Failed to get language-specific system prompt for language '{request.language}': {e}",
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
            "question_answering": "question_answering_system",  # explicit default
        }

        if category not in category_mapping:
            raise ValueError(
                f"Unsupported query category: {category}. Supported categories: {list(category_mapping.keys())}"
            )

        return category_mapping[category]

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
    config: OllamaConfig | None = None,
    http_client: HttpClient | None = None,
    language_config_provider: LanguageConfigProvider | None = None,
    logger=None,
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
    system_logger = get_system_logger()
    log_component_start(
        "ollama_factory",
        "create_client",
        has_config=config is not None,
        has_http=http_client is not None,
        has_lang_provider=language_config_provider is not None,
    )

    if config is None:
        system_logger.debug("ollama_factory", "create_client", "Creating default OllamaConfig")
        from ..utils.config_loader import load_config

        main_config = load_config("config")
        config = OllamaConfig.from_validated_config(main_config)

    system_logger.debug("ollama_factory", "create_client", f"Creating client for model: {config.model}")

    client = MultilingualOllamaClient(
        config=config, http_client=http_client, language_config_provider=language_config_provider, logger=logger
    )

    log_component_end("ollama_factory", "create_client", f"Created Ollama client for {config.model}")
    return client
