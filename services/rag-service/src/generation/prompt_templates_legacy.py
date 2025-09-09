"""
Prompt templates for RAG system using local LLM.
Contains system prompts and templates for different query types.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..utils.config_loader import (get_generation_config, get_language_shared,
                                   get_language_specific_config)


@dataclass
class PromptTemplate:
    """Template for generating prompts."""

    system_prompt: str
    user_template: str
    context_template: str = "Context:\n{context}\n\n"


class MultilingualRAGPrompts:
    """Collection of prompt templates for multilingual RAG system."""

    def __init__(self, language: str = "hr"):
        """Initialize prompts from language-specific configuration."""
        self.language = language

        language_config = get_language_specific_config("prompts", self.language)

        # Extract prompts section from language config
        self._language_prompts = language_config
        self._generation_prompts = get_generation_config()

    @property
    def BASE_SYSTEM_PROMPT(self) -> str:
        """Get base system prompt from config."""
        return self._language_prompts["base_system_prompt"]

    @property
    def QUESTION_ANSWERING(self) -> "PromptTemplate":
        """Get question answering template."""
        return PromptTemplate(
            system_prompt=self._language_prompts.get(
                "question_answering_system", "You are a helpful assistant."
            ),
            user_template=self._language_prompts.get(
                "question_answering_user", "Question: {query}\n\nAnswer:"
            ),
            context_template=self._language_prompts.get(
                "question_answering_context", "Context:\n{context}\n\n"
            ),
        )

    @property
    def SUMMARIZATION(self) -> "PromptTemplate":
        """Get summarization template."""
        return PromptTemplate(
            system_prompt=self._language_prompts.get(
                "summarization_system", "You are a helpful assistant."
            ),
            user_template=self._language_prompts["summarization_user"],
            context_template=self._language_prompts.get(
                "summarization_context", "Text to summarize:\n{context}\n\n"
            ),
        )

    @property
    def FACTUAL_QA(self) -> "PromptTemplate":
        """Get factual Q&A template."""
        return PromptTemplate(
            system_prompt=self._language_prompts.get(
                "factual_qa_system", "You are a helpful assistant."
            ),
            user_template=self._language_prompts.get(
                "factual_qa_user", "Question: {query}\n\nAnswer:"
            ),
            context_template=self._language_prompts.get(
                "factual_qa_context", "Facts:\n{context}\n\n"
            ),
        )

    @property
    def EXPLANATORY(self) -> "PromptTemplate":
        """Get explanatory template."""
        return PromptTemplate(
            system_prompt=self._language_prompts.get(
                "explanatory_system", "You are a helpful assistant."
            ),
            user_template=self._language_prompts["explanatory_user"],
            context_template=self._language_prompts.get(
                "explanatory_context", "Information:\n{context}\n\n"
            ),
        )

    @property
    def COMPARISON(self) -> "PromptTemplate":
        """Get comparison template."""
        return PromptTemplate(
            system_prompt=self._language_prompts.get(
                "comparison_system", "You are a helpful assistant."
            ),
            user_template=self._language_prompts["comparison_user"],
            context_template=self._language_prompts.get(
                "comparison_context", "Information to compare:\n{context}\n\n"
            ),
        )

    # Note: Cultural context handled by language-specific prompt templates

    @property
    def TOURISM(self) -> "PromptTemplate":
        """Get tourism template."""
        return PromptTemplate(
            system_prompt=self._language_prompts.get(
                "tourism_system", "You are a helpful assistant."
            ),
            user_template=self._language_prompts["tourism_user"],
            context_template=self._language_prompts.get(
                "tourism_context", "Tourism information:\n{context}\n\n"
            ),
        )


class PromptBuilder:
    """Builder class for constructing prompts from templates and context."""

    def __init__(self, template: PromptTemplate, language: str = "hr"):
        """
        Initialize prompt builder with template.

        Args:
            template: PromptTemplate to use for building prompts
            language: Language code for the prompts
        """
        self.template = template
        self.language = language
        # Use language-specific config for formatting templates
        language_config = get_language_specific_config("prompts", self.language)
        self._generation_config = {"prompts": language_config}

    def build_prompt(
        self,
        query: str,
        context: Optional[List[str]] = None,
        max_context_length: int = 2000,
    ) -> tuple[str, str]:
        """
        Build complete prompt from query and context.

        Args:
            query: User query
            context: List of context chunks
            max_context_length: Maximum length of context text

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Build context string if provided
        context_text = ""
        if context:
            context_text = self._format_context(context, max_context_length)

        # Build user prompt
        user_prompt = ""
        if context_text:
            user_prompt += self.template.context_template.format(context=context_text)

        user_prompt += self.template.user_template.format(query=query)

        return self.template.system_prompt, user_prompt

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

        formatted_chunks = []
        total_length = 0

        # Get formatting templates from config
        chunk_header_template = self._generation_config["prompts"][
            "chunk_header_template"
        ]
        context_separator = self._generation_config["prompts"]["context_separator"]

        for i, chunk in enumerate(context, 1):
            # Add chunk header using config template
            chunk_header = chunk_header_template.format(index=i) + "\n"
            chunk_text = chunk_header + chunk.strip() + "\n"

            # Check if adding this chunk exceeds limit
            if total_length + len(chunk_text) > max_length:
                if not formatted_chunks:  # At least include first chunk
                    # Truncate the chunk to fit
                    remaining_length = max_length - len(chunk_header) - 10
                    truncated_chunk = chunk[:remaining_length] + "..."
                    formatted_chunks.append(chunk_header + truncated_chunk)
                break

            formatted_chunks.append(chunk_text)
            total_length += len(chunk_text)

        return context_separator.join(formatted_chunks)


def get_prompt_for_query_type(query: str, language: str = "hr") -> PromptTemplate:
    """
    Select appropriate prompt template based on query characteristics.

    Args:
        query: User query text
        language: Language code

    Returns:
        Most suitable PromptTemplate
    """
    query_lower = query.lower()

    # Load keywords from language-specific config
    language_config = get_language_specific_config("prompts", language)
    keywords = language_config["keywords"]

    # Create templates instance
    templates = MultilingualRAGPrompts(language)

    # Get shared question patterns for consistent matching
    shared_config = get_language_shared(language)
    question_patterns = shared_config["question_patterns"]

    # Check for cultural/historical context - use EXPLANATORY template
    if any(keyword in query_lower for keyword in keywords["cultural"]):
        return templates.EXPLANATORY

    # Check for tourism queries
    if any(keyword in query_lower for keyword in keywords["tourism"]):
        return templates.TOURISM

    # Check for summary request (using shared patterns)
    if any(keyword in query_lower for keyword in question_patterns["summarization"]):
        return templates.SUMMARIZATION

    # Check for comparison request (using shared patterns)
    if any(keyword in query_lower for keyword in question_patterns["comparison"]):
        return templates.COMPARISON

    # Check for explanation request (using shared patterns)
    if any(keyword in query_lower for keyword in question_patterns["explanatory"]):
        return templates.EXPLANATORY

    # Check for factual questions (using shared config)
    if any(keyword in query_lower for keyword in question_patterns["factual"]):
        return templates.FACTUAL_QA

    # Default to general question answering
    return templates.QUESTION_ANSWERING


def create_prompt_builder(query: str, language: str = "hr") -> PromptBuilder:
    """
    Factory function to create prompt builder for specific query.

    Args:
        query: User query
        language: Language code

    Returns:
        PromptBuilder with appropriate template
    """
    template = get_prompt_for_query_type(query, language)
    return PromptBuilder(template, language)
