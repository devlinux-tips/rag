"""
Prompt templates for RAG system with local LLM integration.
Clean architecture with dependency injection and pure functions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

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
    if not query:
        raise ValueError("Query cannot be empty")

    if not isinstance(query, str):
        raise ValueError(f"Query must be string, got {type(query)}")

    # Normalize whitespace and strip
    normalized = " ".join(query.strip().split())

    if not normalized:
        raise ValueError("Query cannot be only whitespace")

    if len(normalized) > 10000:
        raise ValueError("Query too long (max 10000 characters)")

    return normalized


def truncate_context_chunks(
    chunks: List[str], max_total_length: int, chunk_separator: str = "\n---\n"
) -> List[str]:
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
    if max_total_length <= 0:
        raise ValueError("max_total_length must be positive")

    if not chunks:
        return []

    if not all(isinstance(chunk, str) for chunk in chunks):
        raise ValueError("All chunks must be strings")

    result_chunks = []
    current_length = 0
    separator_length = len(chunk_separator)

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        # Calculate length if we add this chunk
        chunk_length = len(chunk)
        # Add separator length if not the first chunk
        total_addition = chunk_length + (separator_length if result_chunks else 0)

        if current_length + total_addition <= max_total_length:
            result_chunks.append(chunk)
            current_length += total_addition
        else:
            # Try to fit a truncated version if this is the first chunk
            if not result_chunks:
                max_first_chunk = max_total_length - 3  # Leave space for "..."
                if max_first_chunk > 0:
                    truncated = chunk[:max_first_chunk] + "..."
                    result_chunks.append(truncated)
            break

    return result_chunks


def format_context_with_headers(
    chunks: List[str],
    header_template: str = "Document {index}:",
    chunk_separator: str = "\n\n",
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
            raise ValueError(f"Invalid header_template: {e}")

        formatted_part = f"{header}\n{chunk.strip()}"
        formatted_parts.append(formatted_part)

    return chunk_separator.join(formatted_parts)


def classify_query_type(query: str, keyword_patterns: Dict[str, List[str]]) -> str:
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
    if not query:
        raise ValueError("Query cannot be empty")

    if not isinstance(keyword_patterns, dict):
        raise ValueError("keyword_patterns must be dict")

    query_lower = query.lower()

    # Check patterns in priority order
    priority_order = [
        "tourism",
        "cultural",
        "summarization",
        "comparison",
        "explanatory",
        "factual",
    ]

    for query_type in priority_order:
        if query_type in keyword_patterns:
            keywords = keyword_patterns[query_type]
            if isinstance(keywords, list) and any(
                keyword.lower() in query_lower for keyword in keywords if isinstance(keyword, str)
            ):
                return query_type

    # Check remaining patterns
    for query_type, keywords in keyword_patterns.items():
        if query_type not in priority_order and isinstance(keywords, list):
            if any(
                keyword.lower() in query_lower for keyword in keywords if isinstance(keyword, str)
            ):
                return query_type

    return "default"


def build_complete_prompt(
    system_prompt: str,
    user_template: str,
    context_template: str,
    query: str,
    context_text: str = "",
) -> Tuple[str, str]:
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
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be string")

    if not isinstance(user_template, str):
        raise ValueError("user_template must be string")

    if "{query}" not in user_template:
        raise ValueError("user_template must contain {query} placeholder")

    # Build user prompt
    user_prompt_parts = []

    # Add context if provided
    if context_text:
        if not isinstance(context_template, str):
            raise ValueError("context_template must be string")

        if "{context}" not in context_template:
            raise ValueError("context_template must contain {context} placeholder")

        try:
            formatted_context = context_template.format(context=context_text)
            user_prompt_parts.append(formatted_context)
        except KeyError as e:
            raise ValueError(f"Invalid context_template: {e}")

    # Add query
    try:
        formatted_query = user_template.format(query=query)
        user_prompt_parts.append(formatted_query)
    except KeyError as e:
        raise ValueError(f"Invalid user_template: {e}")

    user_prompt = "".join(user_prompt_parts)

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

    templates: Dict[str, PromptTemplate]
    keyword_patterns: Dict[str, List[str]]
    formatting: Dict[str, str]

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.templates, dict):
            raise ValueError("templates must be dict")

        if not isinstance(self.keyword_patterns, dict):
            raise ValueError("keyword_patterns must be dict")

        if not isinstance(self.formatting, dict):
            raise ValueError("formatting must be dict")

        # Validate required formatting keys
        required_formatting = [
            "header_template",
            "chunk_separator",
            "context_separator",
        ]
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

    def get_templates(self) -> Dict[str, PromptTemplate]:
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
        self.language = language
        self.config_provider = config_provider
        self.logger = logging.getLogger(__name__)

        try:
            self._config = config_provider.get_prompt_config(language)
            self.logger.debug(f"Initialized prompts for language: {language}")
        except Exception as e:
            self.logger.error(f"Failed to initialize prompts for {language}: {e}")
            raise

    @property
    def templates(self) -> Dict[str, PromptTemplate]:
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
        validated_query = validate_query_for_prompt(query)
        return classify_query_type(validated_query, self._config.keyword_patterns)

    def get_template_for_query(self, query: str) -> PromptTemplate:
        """
        Get appropriate template for query.

        Args:
            query: Query text

        Returns:
            Most suitable PromptTemplate
        """
        query_type = self.classify_query(query)

        # Map query types to template names
        template_mapping = {
            "cultural": "explanatory",
            "tourism": "tourism",
            "summarization": "summarization",
            "comparison": "comparison",
            "explanatory": "explanatory",
            "factual": "factual_qa",
            "default": "question_answering",
        }

        if query_type not in template_mapping:
            raise ValueError(
                f"Unknown query type '{query_type}'. Supported types: {list(template_mapping.keys())}"
            )
        template_name = template_mapping[query_type]

        try:
            return self.get_template(template_name)
        except KeyError:
            self.logger.warning(f"Template '{template_name}' not found, using default")
            return self.get_template("question_answering")


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
        self,
        query: str,
        context: Optional[List[str]] = None,
        max_context_length: int = 2000,
    ) -> Tuple[str, str]:
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
        # Validate query
        validated_query = validate_query_for_prompt(query)

        # Process context
        context_text = ""
        if context:
            context_text = self._format_context(context, max_context_length)

        # Build complete prompt
        system_prompt, user_prompt = build_complete_prompt(
            system_prompt=self.template.system_prompt,
            user_template=self.template.user_template,
            context_template=self.template.context_template,
            query=validated_query,
            context_text=context_text,
        )

        self.logger.debug(f"Built prompt for query length: {len(validated_query)}")
        return system_prompt, user_prompt

    def _format_context(self, context: List[str], max_length: int) -> str:
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
            chunks=context,
            max_total_length=max_length,
            chunk_separator=chunk_separator,
        )

        # Format with headers
        formatted_context = format_context_with_headers(
            chunks=truncated_chunks,
            header_template=header_template,
            chunk_separator=chunk_separator,
        )

        return formatted_context


# ===== FACTORY FUNCTIONS =====


def create_multilingual_prompts(
    config_provider: ConfigProvider, language: str = "hr"
) -> MultilingualRAGPrompts:
    """
    Factory function to create multilingual prompts.

    Args:
        config_provider: Configuration provider
        language: Language code

    Returns:
        MultilingualRAGPrompts instance
    """
    return MultilingualRAGPrompts(config_provider, language)


def create_prompt_builder_for_query(
    config_provider: ConfigProvider, query: str, language: str = "hr"
) -> PromptBuilder:
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


def create_mock_config_provider(
    templates: Optional[Dict[str, PromptTemplate]] = None,
    keyword_patterns: Optional[Dict[str, List[str]]] = None,
    formatting: Optional[Dict[str, str]] = None,
) -> ConfigProvider:
    """
    Factory function to create mock configuration provider.

    Args:
        templates: Custom templates (uses defaults if None)
        keyword_patterns: Custom keyword patterns (uses defaults if None)
        formatting: Custom formatting (uses defaults if None)

    Returns:
        Mock ConfigProvider
    """

    class MockConfigProvider:
        def get_prompt_config(self, language: str) -> PromptConfig:
            default_templates = {
                "question_answering": PromptTemplate(
                    system_prompt="You are a helpful assistant.",
                    user_template="Question: {query}\n\nAnswer:",
                    context_template="Context:\n{context}\n\n",
                ),
                "summarization": PromptTemplate(
                    system_prompt="You are a helpful assistant that summarizes text.",
                    user_template="Summarize: {query}",
                    context_template="Text to summarize:\n{context}\n\n",
                ),
                "factual_qa": PromptTemplate(
                    system_prompt="You answer factual questions accurately.",
                    user_template="Question: {query}\n\nAnswer:",
                    context_template="Facts:\n{context}\n\n",
                ),
                "explanatory": PromptTemplate(
                    system_prompt="You provide detailed explanations.",
                    user_template="Explain: {query}",
                    context_template="Information:\n{context}\n\n",
                ),
                "comparison": PromptTemplate(
                    system_prompt="You compare and contrast topics.",
                    user_template="Compare: {query}",
                    context_template="Information to compare:\n{context}\n\n",
                ),
                "tourism": PromptTemplate(
                    system_prompt="You are a tourism expert.",
                    user_template="Tourism question: {query}",
                    context_template="Tourism information:\n{context}\n\n",
                ),
            }

            default_patterns = {
                "cultural": ["culture", "history", "tradition"],
                "tourism": ["hotel", "restaurant", "attraction", "travel"],
                "summarization": ["summarize", "summary", "brief"],
                "comparison": ["compare", "difference", "versus"],
                "explanatory": ["explain", "how", "why", "what"],
                "factual": ["when", "where", "who"],
            }

            default_formatting = {
                "header_template": "Document {index}:",
                "chunk_separator": "\n\n",
                "context_separator": "\n---\n",
            }

            return PromptConfig(
                templates=templates or default_templates,
                keyword_patterns=keyword_patterns or default_patterns,
                formatting=formatting or default_formatting,
            )

    return MockConfigProvider()
