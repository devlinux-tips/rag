"""
Prompt templates for RAG system with local LLM integration.
Clean architecture with dependency injection and pure functions.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_data_transformation,
    log_decision_point,
    log_error_context,
    log_performance_metric,
)

logger = logging.getLogger(__name__)


# ===== PURE FUNCTIONS =====


def validate_query_for_prompt(query: str) -> str:
    """
    Validate and normalize query text for prompt generation.
    Pure function - no side effects, deterministic output.

    Args:
        query: Raw query text

    Returns:
        Validated and normalized query text

    Raises:
        ValueError: If query is invalid
    """
    logger = get_system_logger()
    log_component_start(
        "query_validator",
        "validate_for_prompt",
        input_type=type(query).__name__,
        input_length=len(query) if isinstance(query, str) else 0,
    )

    if not query:
        logger.error("query_validator", "validate_for_prompt", "Query cannot be empty")
        raise ValueError("Query cannot be empty")

    if not isinstance(query, str):
        logger.error("query_validator", "validate_for_prompt", f"Query must be string, got {type(query)}")
        raise ValueError(f"Query must be string, got {type(query)}")

    # Normalize whitespace and strip
    normalized = " ".join(query.strip().split())
    logger.trace(
        "query_validator", "validate_for_prompt", f"Normalized whitespace: {len(query)} → {len(normalized)} chars"
    )

    if not normalized:
        logger.error("query_validator", "validate_for_prompt", "Query cannot be only whitespace")
        raise ValueError("Query cannot be only whitespace")

    if len(normalized) > 10000:
        logger.error("query_validator", "validate_for_prompt", f"Query too long: {len(normalized)} > 10000 chars")
        raise ValueError("Query too long (max 10000 characters)")

    log_data_transformation(
        "query_validator", "normalize_query", f"raw[{len(query)}]", f"normalized[{len(normalized)}]"
    )
    log_component_end("query_validator", "validate_for_prompt", f"Validated query: {len(normalized)} chars")
    return normalized


def truncate_context_chunks(chunks: list[str], max_total_length: int, chunk_separator: str = "\n---\n") -> list[str]:
    """
    Truncate context chunks to fit within length limit.
    Pure function - no side effects, deterministic output.

    Args:
        chunks: List of context chunks
        max_total_length: Maximum total character length
        chunk_separator: Separator between chunks in final text

    Returns:
        List of chunks that fit within length limit

    Raises:
        ValueError: If max_total_length is invalid
    """
    logger = get_system_logger()
    log_component_start(
        "context_truncator",
        "truncate_chunks",
        input_chunks=len(chunks),
        max_length=max_total_length,
        separator_length=len(chunk_separator),
    )

    if max_total_length <= 0:
        logger.error("context_truncator", "truncate_chunks", f"Invalid max_total_length: {max_total_length}")
        raise ValueError("max_total_length must be positive")

    if not chunks:
        logger.debug("context_truncator", "truncate_chunks", "No chunks provided, returning empty list")
        log_component_end("context_truncator", "truncate_chunks", "No chunks to process")
        return []

    if not all(isinstance(chunk, str) for chunk in chunks):
        logger.error("context_truncator", "truncate_chunks", "All chunks must be strings")
        raise ValueError("All chunks must be strings")

    result_chunks: list[str] = []
    current_length = 0
    separator_length = len(chunk_separator)
    total_input_length = sum(len(chunk.strip()) for chunk in chunks if chunk.strip())

    logger.debug(
        "context_truncator",
        "truncate_chunks",
        f"Processing {len(chunks)} chunks, total input: {total_input_length} chars",
    )

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            logger.trace("context_truncator", "truncate_chunks", f"Skipping empty chunk {i}")
            continue

        # Calculate length if we add this chunk
        chunk_length = len(chunk)
        # Add separator length if not the first chunk
        total_addition = chunk_length + (separator_length if result_chunks else 0)

        if current_length + total_addition <= max_total_length:
            result_chunks.append(chunk)
            current_length += total_addition
            logger.trace(
                "context_truncator",
                "truncate_chunks",
                f"Added chunk {i}: {chunk_length} chars, total: {current_length}",
            )
        else:
            # Try to fit a truncated version if this is the first chunk
            if not result_chunks:
                max_first_chunk = max_total_length - 3  # Leave space for "..."
                if max_first_chunk > 0:
                    truncated = chunk[:max_first_chunk] + "..."
                    result_chunks.append(truncated)
                    current_length = len(truncated)
                    log_decision_point(
                        "context_truncator",
                        "truncate_chunks",
                        f"first_chunk_too_long={chunk_length}",
                        f"truncated_to={len(truncated)}",
                    )
                else:
                    logger.warning(
                        "context_truncator",
                        "truncate_chunks",
                        f"First chunk too large for limit: {chunk_length} > {max_total_length}",
                    )
            else:
                log_decision_point(
                    "context_truncator",
                    "truncate_chunks",
                    f"chunk_{i}_exceeds_limit",
                    f"stopping_at_{len(result_chunks)}_chunks",
                )
            break

    log_data_transformation(
        "context_truncator", "filter_chunks", f"input[{len(chunks)}]", f"output[{len(result_chunks)}]"
    )
    log_performance_metric("context_truncator", "truncate_chunks", "final_length", current_length)
    log_component_end(
        "context_truncator", "truncate_chunks", f"Truncated to {len(result_chunks)} chunks, {current_length} chars"
    )
    return result_chunks


def format_context_with_headers(
    chunks: list[str], header_template: str = "Document {index}:", chunk_separator: str = "\n\n"
) -> str:
    """
    Format context chunks with headers and separators.
    Pure function - no side effects, deterministic output.

    Args:
        chunks: List of context chunks
        header_template: Template for chunk headers (must contain {index})
        chunk_separator: Separator between chunks

    Returns:
        Formatted context string

    Raises:
        ValueError: If header_template is invalid
    """
    if not chunks:
        return ""

    if "{index}" not in header_template:
        raise ValueError("header_template must contain {index} placeholder")

    formatted_parts = []

    for i, chunk in enumerate(chunks, 1):
        if not chunk or not chunk.strip():
            continue

        try:
            header = header_template.format(index=i)
        except KeyError as e:
            raise ValueError(f"Invalid header_template: {e}") from e

        formatted_part = f"{header}\n{chunk.strip()}"
        formatted_parts.append(formatted_part)

    return chunk_separator.join(formatted_parts)


def classify_query_type(query: str, keyword_patterns: dict[str, list[str]]) -> str:
    """
    Classify query type based on keyword patterns.
    Pure function - no side effects, deterministic output.

    Args:
        query: Query text to classify
        keyword_patterns: Dict mapping query types to keyword lists

    Returns:
        Query type string (or "default" if no match)

    Raises:
        ValueError: If inputs are invalid
    """
    logger = get_system_logger()
    log_component_start(
        "query_classifier", "classify_type", query_length=len(query), pattern_count=len(keyword_patterns)
    )

    if not query:
        logger.error("query_classifier", "classify_type", "Query cannot be empty")
        raise ValueError("Query cannot be empty")

    if not isinstance(keyword_patterns, dict):
        logger.error("query_classifier", "classify_type", "keyword_patterns must be dict")
        raise ValueError("keyword_patterns must be dict")

    query_lower = query.lower()
    logger.trace(
        "query_classifier",
        "classify_type",
        f"Analyzing query: '{query_lower[:50]}{'...' if len(query_lower) > 50 else ''}'",
    )

    # Check patterns in priority order
    priority_order = ["tourism", "cultural", "summarization", "comparison", "explanatory", "factual"]
    logger.debug("query_classifier", "classify_type", f"Checking priority patterns: {priority_order}")

    for query_type in priority_order:
        if query_type in keyword_patterns:
            keywords = keyword_patterns[query_type]
            if isinstance(keywords, list):
                matched_keywords = [
                    keyword for keyword in keywords if isinstance(keyword, str) and keyword.lower() in query_lower
                ]
                if matched_keywords:
                    log_decision_point(
                        "query_classifier",
                        "classify_type",
                        f"matched_keywords={matched_keywords}",
                        f"type={query_type}",
                    )
                    log_component_end(
                        "query_classifier", "classify_type", f"Classified as {query_type} (priority match)"
                    )
                    return query_type

    # Check remaining patterns
    logger.debug("query_classifier", "classify_type", "Checking remaining patterns")
    for query_type, keywords in keyword_patterns.items():
        if query_type not in priority_order and isinstance(keywords, list):
            matched_keywords = [
                keyword for keyword in keywords if isinstance(keyword, str) and keyword.lower() in query_lower
            ]
            if matched_keywords:
                log_decision_point(
                    "query_classifier", "classify_type", f"matched_keywords={matched_keywords}", f"type={query_type}"
                )
                log_component_end("query_classifier", "classify_type", f"Classified as {query_type} (secondary match)")
                return query_type

    log_decision_point("query_classifier", "classify_type", "no_keywords_matched", "type=default")
    log_component_end("query_classifier", "classify_type", "Classified as default (no matches)")
    return "default"


def build_complete_prompt(
    system_prompt: str, user_template: str, context_template: str, query: str, context_text: str = ""
) -> tuple[str, str]:
    """
    Build complete prompt from components.
    Pure function - no side effects, deterministic output.

    Args:
        system_prompt: System prompt text
        user_template: User prompt template with {query} placeholder
        context_template: Context template with {context} placeholder
        query: User query
        context_text: Formatted context text

    Returns:
        Tuple of (system_prompt, formatted_user_prompt)

    Raises:
        ValueError: If templates are invalid
    """
    # Validate all inputs first
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be string")

    if not isinstance(user_template, str):
        raise ValueError("user_template must be string")

    if not isinstance(query, str):
        raise ValueError("query must be string")

    if context_text is not None and not isinstance(context_text, str):
        raise ValueError("context_text must be string or None")

    if context_template is not None and not isinstance(context_template, str):
        raise ValueError("context_template must be string or None")

    if "{query}" not in user_template:
        raise ValueError("user_template must contain {query} placeholder")

    if context_text and context_template and "{context}" not in context_template:
        raise ValueError("context_template must contain {context} placeholder")

    # Log after validation
    logger = get_system_logger()
    log_component_start(
        "prompt_builder",
        "build_complete",
        system_length=len(system_prompt),
        query_length=len(query),
        context_length=len(context_text) if context_text else 0,
        has_context=bool(context_text),
    )

    # Build user prompt
    user_prompt_parts = []
    logger.debug("prompt_builder", "build_complete", "Building user prompt components")

    # Add context if provided
    if context_text:
        logger.debug("prompt_builder", "build_complete", f"Adding context: {len(context_text)} chars")

        try:
            formatted_context = context_template.format(context=context_text)
            user_prompt_parts.append(formatted_context)
            logger.trace("prompt_builder", "build_complete", f"Formatted context: {len(formatted_context)} chars")
        except KeyError as e:
            logger.error("prompt_builder", "build_complete", f"Invalid context_template: {e}")
            raise ValueError(f"Invalid context_template: {e}") from e

    # Add query
    logger.trace("prompt_builder", "build_complete", f"Adding query: {len(query)} chars")
    try:
        formatted_query = user_template.format(query=query)
        user_prompt_parts.append(formatted_query)
        logger.trace("prompt_builder", "build_complete", f"Formatted query: {len(formatted_query)} chars")
    except KeyError as e:
        logger.error("prompt_builder", "build_complete", f"Invalid user_template: {e}")
        raise ValueError(f"Invalid user_template: {e}") from e

    user_prompt = "".join(user_prompt_parts)
    log_data_transformation(
        "prompt_builder", "combine_parts", f"parts[{len(user_prompt_parts)}]", f"prompt[{len(user_prompt)}]"
    )
    log_component_end(
        "prompt_builder",
        "build_complete",
        f"Built complete prompt: system={len(system_prompt)}, user={len(user_prompt)}",
    )
    return system_prompt, user_prompt


# ===== DATA STRUCTURES =====


@dataclass
class PromptTemplate:
    """Template for generating prompts."""

    system_prompt: str
    user_template: str
    context_template: str = "Context:\n{context}\n\n"

    def __post_init__(self):
        """Validate template after initialization."""
        if not self.system_prompt:
            raise ValueError("system_prompt cannot be empty")

        if not self.user_template:
            raise ValueError("user_template cannot be empty")

        if "{query}" not in self.user_template:
            raise ValueError("user_template must contain {query} placeholder")

        if self.context_template and "{context}" not in self.context_template:
            raise ValueError("context_template must contain {context} placeholder")


@dataclass
class PromptConfig:
    """Configuration for prompt templates and formatting."""

    templates: dict[str, PromptTemplate]
    keyword_patterns: dict[str, list[str]]
    formatting: dict[str, str]

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.templates, dict):
            raise ValueError("templates must be dict")

        if not isinstance(self.keyword_patterns, dict):
            raise ValueError("keyword_patterns must be dict")

        if not isinstance(self.formatting, dict):
            raise ValueError("formatting must be dict")

        # Validate required formatting keys
        required_formatting = ["header_template", "chunk_separator", "context_separator"]
        for key in required_formatting:
            if key not in self.formatting:
                raise ValueError(f"formatting must contain {key}")


# ===== PROTOCOLS =====


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""

    def get_prompt_config(self, language: str) -> PromptConfig:
        """Get prompt configuration for language."""
        ...


class PromptTemplateProvider(Protocol):
    """Protocol for prompt template providers."""

    def get_template(self, template_name: str) -> PromptTemplate:
        """Get specific prompt template."""
        ...

    def get_templates(self) -> dict[str, PromptTemplate]:
        """Get all available templates."""
        ...


# ===== CORE CLASSES =====


class MultilingualRAGPrompts:
    """Collection of prompt templates for multilingual RAG system."""

    def __init__(self, config_provider: ConfigProvider, language: str = "hr"):
        """
        Initialize prompts with dependency injection.

        Args:
            config_provider: Provider for prompt configuration
            language: Language code
        """
        logger = get_system_logger()
        log_component_start("multilingual_prompts", "init", language=language)

        self.language = language
        self.config_provider = config_provider
        self.logger = logging.getLogger(__name__)

        try:
            logger.debug("multilingual_prompts", "init", f"Loading prompt config for {language}")
            self._config = config_provider.get_prompt_config(language)

            template_count = len(self._config.templates)
            pattern_count = len(self._config.keyword_patterns)
            log_performance_metric("multilingual_prompts", "init", "template_count", template_count)
            log_performance_metric("multilingual_prompts", "init", "pattern_count", pattern_count)

            log_component_end("multilingual_prompts", "init", f"Initialized {template_count} templates for {language}")
        except Exception as e:
            log_error_context(
                "multilingual_prompts",
                "init",
                e,
                {"language": language, "config_provider": type(config_provider).__name__},
            )
            raise

    @property
    def templates(self) -> dict[str, PromptTemplate]:
        """Get all available templates."""
        return self._config.templates

    def get_template(self, template_name: str) -> PromptTemplate:
        """
        Get specific template by name.

        Args:
            template_name: Name of template

        Returns:
            PromptTemplate instance

        Raises:
            KeyError: If template not found
        """
        if template_name not in self._config.templates:
            available = list(self._config.templates.keys())
            raise KeyError(f"Template '{template_name}' not found. Available: {available}")

        return self._config.templates[template_name]

    def classify_query(self, query: str) -> str:
        """
        Classify query type using configured patterns.

        Args:
            query: Query text

        Returns:
            Query type string
        """
        get_system_logger()
        log_component_start("multilingual_prompts", "classify_query", query_length=len(query), language=self.language)

        validated_query = validate_query_for_prompt(query)
        query_type = classify_query_type(validated_query, self._config.keyword_patterns)

        log_component_end("multilingual_prompts", "classify_query", f"Classified '{query[:50]}...' as {query_type}")
        return query_type

    def get_template_for_query(self, query: str) -> PromptTemplate:
        """
        Get appropriate template for query.

        Args:
            query: Query text

        Returns:
            Most suitable PromptTemplate
        """
        logger = get_system_logger()
        log_component_start("multilingual_prompts", "get_template_for_query", query_length=len(query))

        query_type = self.classify_query(query)

        # Map query types to template names
        template_mapping = {
            "cultural": "explanatory",
            "tourism": "tourism",
            "summarization": "summarization",
            "comparison": "comparison",
            "explanatory": "explanatory",
            "factual": "factual_qa",
            "question": "question_answering",  # Question queries use Q&A template
            "default": "question_answering",
        }

        if query_type not in template_mapping:
            logger.error(
                "multilingual_prompts",
                "get_template_for_query",
                f"Unknown query type '{query_type}'. Supported: {list(template_mapping.keys())}",
            )
            raise ValueError(f"Unknown query type '{query_type}'. Supported types: {list(template_mapping.keys())}")

        template_name = template_mapping[query_type]
        log_decision_point(
            "multilingual_prompts", "get_template_for_query", f"query_type={query_type}", f"template={template_name}"
        )

        try:
            template = self.get_template(template_name)
            log_component_end("multilingual_prompts", "get_template_for_query", f"Selected template: {template_name}")
            return template
        except KeyError:
            logger.warning(
                "multilingual_prompts", "get_template_for_query", f"Template '{template_name}' not found, using default"
            )
            fallback_template = self.get_template("question_answering")
            log_component_end(
                "multilingual_prompts", "get_template_for_query", "Using fallback template: question_answering"
            )
            return fallback_template


class PromptBuilder:
    """Builder class for constructing prompts from templates and context."""

    def __init__(self, template: PromptTemplate, prompt_config: PromptConfig):
        """
        Initialize prompt builder with template and configuration.

        Args:
            template: PromptTemplate to use
            prompt_config: Configuration for formatting
        """
        self.template = template
        self.config = prompt_config
        self.logger = logging.getLogger(__name__)

    def build_prompt(
        self, query: str, context: list[str] | None = None, max_context_length: int = 2000
    ) -> tuple[str, str]:
        """
        Build complete prompt from query and context.

        Args:
            query: User query
            context: List of context chunks
            max_context_length: Maximum length of context text

        Returns:
            Tuple of (system_prompt, user_prompt)

        Raises:
            ValueError: If inputs are invalid
        """
        logger = get_system_logger()
        log_component_start(
            "prompt_builder",
            "build_prompt",
            query_length=len(query),
            context_count=len(context) if context else 0,
            max_context_length=max_context_length,
        )

        # Validate query
        validated_query = validate_query_for_prompt(query)

        # Process context
        context_text = ""
        if context:
            logger.debug("prompt_builder", "build_prompt", f"Formatting {len(context)} context chunks")
            context_text = self._format_context(context, max_context_length)
            log_performance_metric("prompt_builder", "build_prompt", "context_text_length", len(context_text))

        # Build complete prompt
        system_prompt, user_prompt = build_complete_prompt(
            system_prompt=self.template.system_prompt,
            user_template=self.template.user_template,
            context_template=self.template.context_template,
            query=validated_query,
            context_text=context_text,
        )

        log_performance_metric("prompt_builder", "build_prompt", "system_prompt_length", len(system_prompt))
        log_performance_metric("prompt_builder", "build_prompt", "user_prompt_length", len(user_prompt))
        log_component_end(
            "prompt_builder", "build_prompt", f"Built prompt: system={len(system_prompt)}, user={len(user_prompt)}"
        )
        return system_prompt, user_prompt

    def _format_context(self, context: list[str], max_length: int) -> str:
        """
        Format context chunks into single text with length limit.

        Args:
            context: List of context chunks
            max_length: Maximum total length

        Returns:
            Formatted context text
        """
        if not context:
            return ""

        # Get formatting configuration
        header_template = self.config.formatting["header_template"]
        chunk_separator = self.config.formatting["chunk_separator"]

        # Truncate chunks to fit length limit
        truncated_chunks = truncate_context_chunks(
            chunks=context, max_total_length=max_length, chunk_separator=chunk_separator
        )

        # Format with headers
        formatted_context = format_context_with_headers(
            chunks=truncated_chunks, header_template=header_template, chunk_separator=chunk_separator
        )

        return formatted_context


# ===== FACTORY FUNCTIONS =====


def create_multilingual_prompts(config_provider: ConfigProvider, language: str = "hr") -> MultilingualRAGPrompts:
    """
    Factory function to create multilingual prompts.

    Args:
        config_provider: Configuration provider
        language: Language code

    Returns:
        MultilingualRAGPrompts instance
    """
    return MultilingualRAGPrompts(config_provider, language)


def create_prompt_builder_for_query(config_provider: ConfigProvider, query: str, language: str = "hr") -> PromptBuilder:
    """
    Factory function to create prompt builder for specific query.

    Args:
        config_provider: Configuration provider
        query: User query
        language: Language code

    Returns:
        PromptBuilder with appropriate template
    """
    prompts = create_multilingual_prompts(config_provider, language)
    template = prompts.get_template_for_query(query)
    config = prompts._config

    return PromptBuilder(template, config)
