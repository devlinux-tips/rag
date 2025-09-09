"""
Enhanced Prompt Template System with Category-Specific Templates.

This module provides specialized prompt templates for different document categories,
supporting multilingual content through configuration-driven templates only.
NO HARDCODED DEFAULTS - all templates come from language-specific configuration files.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..retrieval.categorization import DocumentCategory
from ..utils.config_loader import get_language_specific_config


class PromptType(Enum):
    """Types of prompt templates."""

    SYSTEM = "system"
    USER = "user"
    CONTEXT = "context"
    FOLLOWUP = "followup"


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""

    template: str
    category: DocumentCategory
    prompt_type: PromptType
    language: str
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logging.warning(f"Missing template variable: {e}")
            return self.template


class EnhancedPromptBuilder:
    """Enhanced prompt builder with category-specific templates from configuration only."""

    def __init__(self, language: str):
        """Initialize prompt builder with language support."""
        self.language = language
        self.logger = logging.getLogger(__name__)

        # Load category-specific prompt templates from config ONLY
        self._load_prompt_templates()

    def _load_prompt_templates(self) -> None:
        """Load prompt templates from configuration - fail if not found."""
        # Load prompts config - will raise exception if not found (fail-fast)
        prompts_config = get_language_specific_config("prompts", self.language)

        # Initialize template storage
        self.templates = {}

        # Parse and organize templates
        self._parse_config_templates(prompts_config)

    def _parse_config_templates(self, config: Dict[str, Any]) -> None:
        """Parse configuration templates into PromptTemplate objects."""
        for category_name, category_templates in config.items():
            try:
                category = DocumentCategory(category_name)
            except ValueError:
                self.logger.warning(f"Unknown category: {category_name}")
                continue

            if category not in self.templates:
                self.templates[category] = {}

            for template_type, template_text in category_templates.items():
                try:
                    prompt_type = PromptType(template_type)

                    template = PromptTemplate(
                        template=template_text,
                        category=category,
                        prompt_type=prompt_type,
                        language=self.language,
                    )

                    self.templates[category][prompt_type] = template

                except ValueError:
                    self.logger.warning(f"Unknown prompt type: {template_type}")
                    continue

    def build_prompt(
        self,
        query: str,
        context_chunks: List[str],
        category: DocumentCategory = DocumentCategory.GENERAL,
        max_context_length: int = 2000,
        include_source_attribution: bool = True,
    ) -> Tuple[str, str]:
        """Build category-specific system and user prompts."""

        # Get templates for the category - fail if not found
        category_templates = self.templates[category]  # Will raise KeyError if missing

        # Get system prompt template - fail if not found
        system_template = category_templates[
            PromptType.SYSTEM
        ]  # Will raise KeyError if missing
        system_prompt = system_template.template

        # Build context from chunks
        formatted_context = self._format_context_chunks(
            context_chunks,
            max_length=max_context_length,
            include_attribution=include_source_attribution,
        )

        # Get user prompt template - fail if not found
        user_template = category_templates[
            PromptType.USER
        ]  # Will raise KeyError if missing
        user_prompt = user_template.format(query=query, context=formatted_context)

        return system_prompt, user_prompt

    def _format_context_chunks(
        self,
        chunks: List[str],
        max_length: int = 2000,
        include_attribution: bool = True,
    ) -> str:
        """Format context chunks with length limitation."""
        if not chunks:
            # Get no-context message from config
            messages_config = get_language_specific_config("messages", self.language)
            return messages_config["no_context"]

        # Get formatting config
        formatting_config = get_language_specific_config("formatting", self.language)
        source_label = formatting_config["source_label"]

        formatted_chunks = []
        current_length = 0

        for i, chunk in enumerate(chunks):
            # Add chunk numbering for clarity
            if include_attribution:
                chunk_header = f"\n[{source_label} {i+1}]:\n"
                formatted_chunk = f"{chunk_header}{chunk.strip()}"
            else:
                formatted_chunk = chunk.strip()

            # Check if adding this chunk would exceed the limit
            if current_length + len(formatted_chunk) > max_length:
                # Truncate the last chunk to fit within the limit
                remaining_space = max_length - current_length - 50  # Leave some buffer
                if remaining_space > 100:  # Only add if there's meaningful space
                    truncated_chunk = formatted_chunk[:remaining_space].rsplit(" ", 1)[
                        0
                    ]  # Break at word boundary
                    ellipsis = formatting_config.get("truncation_indicator", "...")
                    formatted_chunks.append(truncated_chunk + ellipsis)
                break

            formatted_chunks.append(formatted_chunk)
            current_length += len(formatted_chunk)

        return "\n\n".join(formatted_chunks)

    def get_followup_prompt(
        self,
        original_query: str,
        original_answer: str,
        followup_query: str,
        category: DocumentCategory = DocumentCategory.GENERAL,
    ) -> str:
        """Generate a follow-up prompt that maintains conversation context."""

        # Get follow-up template - fail if not found
        category_templates = self.templates[category]
        followup_template = category_templates[
            PromptType.FOLLOWUP
        ]  # Will raise KeyError if missing

        return followup_template.format(
            original_query=original_query,
            original_answer=original_answer,
            followup_query=followup_query,
        )

    def suggest_template_improvements(
        self, category: DocumentCategory, usage_stats: Dict[str, Any]
    ) -> List[str]:
        """Suggest improvements for prompt templates based on usage statistics."""
        suggestions = []

        # Check if category has templates
        if category not in self.templates:
            suggestions.append(
                f"Add specialized templates for {category.value} category"
            )

        # Check template completeness
        category_templates = self.templates.get(category, {})
        missing_types = []

        for prompt_type in PromptType:
            if prompt_type not in category_templates:
                missing_types.append(prompt_type.value)

        if missing_types:
            suggestions.append(
                f"Add missing template types: {', '.join(missing_types)}"
            )

        # Performance-based suggestions
        if usage_stats.get("avg_confidence", 1.0) < 0.7:
            suggestions.append(
                "Consider more specific system prompts to improve response quality"
            )

        avg_length = usage_stats.get("avg_response_length", 200)
        if avg_length < 100:
            suggestions.append("Templates might be generating too brief responses")
        elif avg_length > 500:
            suggestions.append("Templates might be generating verbose responses")

        return suggestions

    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded templates."""
        stats = {
            "total_categories": len(self.templates),
            "language": self.language,
            "categories": {},
        }

        for category, templates in self.templates.items():
            stats["categories"][category.value] = {
                "template_count": len(templates),
                "template_types": [t.value for t in templates.keys()],
                "avg_template_length": sum(len(t.template) for t in templates.values())
                / len(templates)
                if templates
                else 0,
            }

        return stats

    def validate_templates(self) -> Dict[str, Any]:
        """Validate all loaded templates for common issues."""
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
        }

        for category, templates in self.templates.items():
            for prompt_type, template in templates.items():
                # Check for required placeholders
                template_text = template.template

                if prompt_type == PromptType.USER:
                    if "{query}" not in template_text:
                        validation_results["issues"].append(
                            f"{category.value}/{prompt_type.value}: Missing {{query}} placeholder"
                        )
                        validation_results["valid"] = False

                    if "{context}" not in template_text:
                        validation_results["issues"].append(
                            f"{category.value}/{prompt_type.value}: Missing {{context}} placeholder"
                        )
                        validation_results["valid"] = False

                # Check template length
                if len(template_text) < 20:
                    validation_results["warnings"].append(
                        f"{category.value}/{prompt_type.value}: Very short template (might be insufficient)"
                    )
                elif len(template_text) > 500:
                    validation_results["warnings"].append(
                        f"{category.value}/{prompt_type.value}: Very long template (might be verbose)"
                    )

        return validation_results


def create_enhanced_prompt_builder(language: str) -> EnhancedPromptBuilder:
    """Factory function to create enhanced prompt builder."""
    return EnhancedPromptBuilder(language=language)
