"""
Prompt templates for Croatian RAG system using local LLM.
Contains system prompts and templates for different query types.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..utils.config_loader import (
    get_croatian_prompts,
    get_croatian_settings,
    get_croatian_shared,
    get_generation_config,
    get_generation_prompts_config,
)
from ..utils.error_handler import handle_config_error


@dataclass
class PromptTemplate:
    """Template for generating prompts."""

    system_prompt: str
    user_template: str
    context_template: str = "Context:\n{context}\n\n"


class CroatianRAGPrompts:
    """Collection of prompt templates for Croatian RAG system."""

    def __init__(self):
        """Initialize prompts from configuration."""
        croatian_config = handle_config_error(
            operation=lambda: get_croatian_settings(),
            fallback_value={"prompts": {}, "generation": {}},
            config_file="config/croatian.toml",
            section="[prompts]",
        )

        # Extract prompts section from Croatian config
        self._croatian_prompts = croatian_config.get("prompts", {})
        self._generation_prompts = handle_config_error(
            operation=lambda: get_generation_config(),
            fallback_value={
                "question_answering": "Answer the question: {question}",
                "summarization": "Summarize the text: {text}",
            },
            config_file="config/config.toml",
            section="[prompts]",
        )

    @property
    def BASE_SYSTEM_PROMPT(self) -> str:
        """Get base system prompt from config."""
        return self._croatian_prompts["base_system_prompt"]

    @property
    def QUESTION_ANSWERING(self) -> "PromptTemplate":
        """Get question answering template."""
        return PromptTemplate(
            system_prompt=self._croatian_prompts["question_answering_system"],
            user_template=self._croatian_prompts["question_answering_user"],
            context_template=self._croatian_prompts["question_answering_context"],
        )

    @property
    def SUMMARIZATION(self) -> "PromptTemplate":
        """Get summarization template."""
        return PromptTemplate(
            system_prompt=self._croatian_prompts["summarization_system"],
            user_template=self._croatian_prompts["summarization_user"],
            context_template=self._croatian_prompts["summarization_context"],
        )

    @property
    def FACTUAL_QA(self) -> "PromptTemplate":
        """Get factual Q&A template."""
        return PromptTemplate(
            system_prompt=self._croatian_prompts["factual_qa_system"],
            user_template=self._croatian_prompts["factual_qa_user"],
            context_template=self._croatian_prompts["factual_qa_context"],
        )

    @property
    def EXPLANATORY(self) -> "PromptTemplate":
        """Get explanatory template."""
        return PromptTemplate(
            system_prompt=self._croatian_prompts["explanatory_system"],
            user_template=self._croatian_prompts["explanatory_user"],
            context_template=self._croatian_prompts["explanatory_context"],
        )

    @property
    def COMPARISON(self) -> "PromptTemplate":
        """Get comparison template."""
        return PromptTemplate(
            system_prompt=self._croatian_prompts["comparison_system"],
            user_template=self._croatian_prompts["comparison_user"],
            context_template=self._croatian_prompts["comparison_context"],
        )

    @property
    def CULTURAL_CONTEXT(self) -> "PromptTemplate":
        """Get cultural context template."""
        return PromptTemplate(
            system_prompt=self._croatian_prompts["cultural_context_system"],
            user_template=self._croatian_prompts["cultural_context_user"],
            context_template=self._croatian_prompts["cultural_context_context"],
        )

    @property
    def TOURISM(self) -> "PromptTemplate":
        """Get tourism template."""
        return PromptTemplate(
            system_prompt=self._croatian_prompts["tourism_system"],
            user_template=self._croatian_prompts["tourism_user"],
            context_template=self._croatian_prompts["tourism_context"],
        )


class PromptBuilder:
    """Builder class for constructing prompts from templates and context."""

    def __init__(self, template: PromptTemplate):
        """
        Initialize prompt builder with template.

        Args:
            template: PromptTemplate to use for building prompts
        """
        self.template = template
        # Use Croatian config for formatting templates
        self._generation_config = {"prompts": get_croatian_settings()["prompts"]}

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
        chunk_header_template = self._generation_config["prompts"]["chunk_header_template"]
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


def get_prompt_for_query_type(query: str) -> PromptTemplate:
    """
    Select appropriate prompt template based on query characteristics.

    Args:
        query: User query text

    Returns:
        Most suitable PromptTemplate
    """
    query_lower = query.lower()

    # Load keywords from Croatian config
    croatian_config = get_croatian_settings()
    keywords = croatian_config["prompts"]["keywords"]  # Create templates instance
    templates = CroatianRAGPrompts()

    # Get shared question patterns for consistent matching
    shared_config = get_croatian_shared()
    question_patterns = shared_config.get("question_patterns", {})

    # Check for cultural/historical context
    if any(keyword in query_lower for keyword in keywords["cultural"]):
        return templates.CULTURAL_CONTEXT

    # Check for tourism queries
    if any(keyword in query_lower for keyword in keywords["tourism"]):
        return templates.TOURISM

    # Check for summary request (using shared patterns)
    if any(keyword in query_lower for keyword in question_patterns.get("summarization", [])):
        return templates.SUMMARIZATION

    # Check for comparison request (using shared patterns)
    if any(keyword in query_lower for keyword in question_patterns.get("comparison", [])):
        return templates.COMPARISON

    # Check for explanation request (using shared patterns)
    if any(keyword in query_lower for keyword in question_patterns.get("explanatory", [])):
        return templates.EXPLANATORY

    # Check for factual questions (using shared config)
    if any(keyword in query_lower for keyword in question_patterns.get("factual", [])):
        return templates.FACTUAL_QA

    # Default to general question answering
    return templates.QUESTION_ANSWERING


def create_prompt_builder(query: str) -> PromptBuilder:
    """
    Factory function to create prompt builder for specific query.

    Args:
        query: User query

    Returns:
        PromptBuilder with appropriate template
    """
    template = get_prompt_for_query_type(query)
    return PromptBuilder(template)
