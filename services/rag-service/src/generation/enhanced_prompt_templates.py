"""
Enhanced Prompt Template System with dependency injection.
100% testable version with pure functions and dependency injection architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.retrieval.categorization import DocumentCategory


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
            # Return template with missing placeholder info instead of raising
            return f"{self.template} [MISSING: {e}]"


@dataclass
class PromptConfig:
    """Complete prompt configuration."""

    category_templates: Dict[DocumentCategory, Dict[PromptType, str]]
    messages: Dict[str, str]
    formatting: Dict[str, str]
    language: str


@dataclass
class ContextFormattingOptions:
    """Options for context chunk formatting."""

    max_length: int = 2000
    include_attribution: bool = True
    source_label: str = "Source"
    truncation_indicator: str = "..."
    min_chunk_size: int = 100


@dataclass
class BuildPromptResult:
    """Result of building a prompt."""

    system_prompt: str
    user_prompt: str
    context_used: str
    chunks_included: int
    truncated: bool


@dataclass
class ValidationResult:
    """Template validation result."""

    valid: bool
    issues: List[str]
    warnings: List[str]


@dataclass
class TemplateStats:
    """Statistics about loaded templates."""

    total_categories: int
    language: str
    categories: Dict[str, Dict[str, Any]]


# ================================
# DEPENDENCY INJECTION PROTOCOLS
# ================================


class ConfigProvider(Protocol):
    """Protocol for configuration access."""

    def get_prompt_config(self, language: str) -> PromptConfig:
        """Get prompt configuration for language."""
        ...


class LoggerProvider(Protocol):
    """Protocol for logging operations."""

    def info(self, message: str) -> None:
        """Log info message."""
        ...

    def debug(self, message: str) -> None:
        """Log debug message."""
        ...

    def warning(self, message: str) -> None:
        """Log warning message."""
        ...

    def error(self, message: str) -> None:
        """Log error message."""
        ...


# ================================
# PURE BUSINESS LOGIC FUNCTIONS
# ================================


def parse_config_templates(
    config_data: Dict[str, Dict[str, str]], language: str
) -> Dict[DocumentCategory, Dict[PromptType, PromptTemplate]]:
    """Pure function to parse configuration templates into PromptTemplate objects."""
    templates = {}

    for category_name, category_templates in config_data.items():
        try:
            category = DocumentCategory(category_name)
        except ValueError:
            # Skip unknown categories
            continue

        if category not in templates:
            templates[category] = {}

        for template_type, template_text in category_templates.items():
            try:
                prompt_type = PromptType(template_type)

                template = PromptTemplate(
                    template=template_text,
                    category=category,
                    prompt_type=prompt_type,
                    language=language,
                )

                templates[category][prompt_type] = template

            except ValueError:
                # Skip unknown prompt types
                continue

    return templates


def format_context_chunks(
    chunks: List[str],
    options: ContextFormattingOptions,
    no_context_message: str = "No relevant context available.",
) -> Tuple[str, bool]:
    """Pure function to format context chunks with length limitation."""
    if not chunks:
        return no_context_message, False

    formatted_chunks = []
    current_length = 0
    truncated = False

    for i, chunk in enumerate(chunks):
        # Add chunk numbering for clarity
        if options.include_attribution:
            chunk_header = f"\n[{options.source_label} {i+1}]:\n"
            formatted_chunk = f"{chunk_header}{chunk.strip()}"
        else:
            formatted_chunk = chunk.strip()

        # Check if adding this chunk would exceed the limit
        if current_length + len(formatted_chunk) > options.max_length:
            # Truncate the last chunk to fit within the limit
            remaining_space = (
                options.max_length - current_length - 50
            )  # Leave some buffer
            if (
                remaining_space > options.min_chunk_size
            ):  # Only add if there's meaningful space
                truncated_chunk = formatted_chunk[:remaining_space].rsplit(" ", 1)[
                    0
                ]  # Break at word boundary
                formatted_chunks.append(truncated_chunk + options.truncation_indicator)
            truncated = True
            break

        formatted_chunks.append(formatted_chunk)
        current_length += len(formatted_chunk)

    return "\n\n".join(formatted_chunks), truncated


def build_category_prompt(
    query: str,
    context_chunks: List[str],
    category: DocumentCategory,
    templates: Dict[DocumentCategory, Dict[PromptType, PromptTemplate]],
    formatting_options: ContextFormattingOptions,
    no_context_message: str = "No relevant context available.",
) -> BuildPromptResult:
    """Pure function to build category-specific system and user prompts."""

    # Get templates for the category
    if category not in templates:
        raise KeyError(f"No templates found for category: {category.value}")

    category_templates = templates[category]

    # Get system prompt template
    if PromptType.SYSTEM not in category_templates:
        raise KeyError(f"No system template found for category: {category.value}")

    system_template = category_templates[PromptType.SYSTEM]
    system_prompt = system_template.template

    # Build context from chunks
    formatted_context, truncated = format_context_chunks(
        chunks=context_chunks,
        options=formatting_options,
        no_context_message=no_context_message,
    )

    # Get user prompt template
    if PromptType.USER not in category_templates:
        raise KeyError(f"No user template found for category: {category.value}")

    user_template = category_templates[PromptType.USER]
    user_prompt = user_template.format(query=query, context=formatted_context)

    return BuildPromptResult(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_used=formatted_context,
        chunks_included=len(context_chunks)
        if not truncated
        else len(context_chunks) - 1,
        truncated=truncated,
    )


def build_followup_prompt(
    original_query: str,
    original_answer: str,
    followup_query: str,
    category: DocumentCategory,
    templates: Dict[DocumentCategory, Dict[PromptType, PromptTemplate]],
) -> str:
    """Pure function to generate a follow-up prompt that maintains conversation context."""

    # Get templates for the category
    if category not in templates:
        raise KeyError(f"No templates found for category: {category.value}")

    category_templates = templates[category]

    # Get follow-up template
    if PromptType.FOLLOWUP not in category_templates:
        raise KeyError(f"No followup template found for category: {category.value}")

    followup_template = category_templates[PromptType.FOLLOWUP]

    return followup_template.format(
        original_query=original_query,
        original_answer=original_answer,
        followup_query=followup_query,
    )


def calculate_template_stats(
    templates: Dict[DocumentCategory, Dict[PromptType, PromptTemplate]], language: str
) -> TemplateStats:
    """Pure function to calculate statistics about loaded templates."""
    categories = {}

    for category, template_dict in templates.items():
        categories[category.value] = {
            "template_count": len(template_dict),
            "template_types": [t.value for t in template_dict.keys()],
            "avg_template_length": sum(len(t.template) for t in template_dict.values())
            / len(template_dict)
            if template_dict
            else 0,
        }

    return TemplateStats(
        total_categories=len(templates), language=language, categories=categories
    )


def validate_templates(
    templates: Dict[DocumentCategory, Dict[PromptType, PromptTemplate]]
) -> ValidationResult:
    """Pure function to validate all templates for common issues."""
    issues = []
    warnings = []

    for category, template_dict in templates.items():
        for prompt_type, template in template_dict.items():
            template_text = template.template

            # Check for required placeholders
            if prompt_type == PromptType.USER:
                if "{query}" not in template_text:
                    issues.append(
                        f"{category.value}/{prompt_type.value}: Missing {{query}} placeholder"
                    )

                if "{context}" not in template_text:
                    issues.append(
                        f"{category.value}/{prompt_type.value}: Missing {{context}} placeholder"
                    )

            if prompt_type == PromptType.FOLLOWUP:
                required_placeholders = [
                    "{original_query}",
                    "{original_answer}",
                    "{followup_query}",
                ]
                for placeholder in required_placeholders:
                    if placeholder not in template_text:
                        issues.append(
                            f"{category.value}/{prompt_type.value}: Missing {placeholder} placeholder"
                        )

            # Check template length
            if len(template_text) < 20:
                warnings.append(
                    f"{category.value}/{prompt_type.value}: Very short template (might be insufficient)"
                )
            elif len(template_text) > 500:
                warnings.append(
                    f"{category.value}/{prompt_type.value}: Very long template (might be verbose)"
                )

    return ValidationResult(valid=len(issues) == 0, issues=issues, warnings=warnings)


def suggest_template_improvements(
    templates: Dict[DocumentCategory, Dict[PromptType, PromptTemplate]],
    category: DocumentCategory,
    usage_stats: Dict[str, Any],
) -> List[str]:
    """Pure function to suggest improvements for prompt templates based on usage statistics."""
    suggestions = []

    # Check if category has templates
    if category not in templates:
        suggestions.append(f"Add specialized templates for {category.value} category")
        return suggestions

    # Check template completeness
    category_templates = templates[category]
    missing_types = []

    for prompt_type in PromptType:
        if prompt_type not in category_templates:
            missing_types.append(prompt_type.value)

    if missing_types:
        suggestions.append(f"Add missing template types: {', '.join(missing_types)}")

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


def get_missing_templates(
    templates: Dict[DocumentCategory, Dict[PromptType, PromptTemplate]],
    required_categories: List[DocumentCategory],
    required_types: List[PromptType],
) -> Dict[str, List[str]]:
    """Pure function to identify missing templates for given categories and types."""
    missing = {"categories": [], "templates": []}

    for category in required_categories:
        if category not in templates:
            missing["categories"].append(category.value)
            continue

        category_templates = templates[category]
        for prompt_type in required_types:
            if prompt_type not in category_templates:
                missing["templates"].append(f"{category.value}/{prompt_type.value}")

    return missing


def find_template_by_content(
    templates: Dict[DocumentCategory, Dict[PromptType, PromptTemplate]],
    search_text: str,
) -> List[Tuple[DocumentCategory, PromptType]]:
    """Pure function to find templates containing specific text."""
    matches = []

    for category, template_dict in templates.items():
        for prompt_type, template in template_dict.items():
            if search_text.lower() in template.template.lower():
                matches.append((category, prompt_type))

    return matches


# ================================
# DEPENDENCY INJECTION ORCHESTRATION
# ================================


class _EnhancedPromptBuilder:
    """100% testable enhanced prompt builder with dependency injection."""

    def __init__(
        self,
        language: str,
        config_provider: ConfigProvider,
        logger_provider: Optional[LoggerProvider] = None,
    ):
        """Initialize with injected dependencies."""
        self.language = language
        self._config_provider = config_provider
        self._logger = logger_provider

        # Load configuration
        self._prompt_config = self._config_provider.get_prompt_config(language)
        self._templates = parse_config_templates(
            self._prompt_config.category_templates, language
        )

        self._log_info(
            f"Loaded {len(self._templates)} template categories for {language}"
        )

    def _log_info(self, message: str) -> None:
        """Log info message if logger available."""
        if self._logger:
            self._logger.info(message)

    def _log_debug(self, message: str) -> None:
        """Log debug message if logger available."""
        if self._logger:
            self._logger.debug(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message if logger available."""
        if self._logger:
            self._logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log error message if logger available."""
        if self._logger:
            self._logger.error(message)

    def build_prompt(
        self,
        query: str,
        context_chunks: List[str],
        category: DocumentCategory = DocumentCategory.GENERAL,
        max_context_length: int = 2000,
        include_source_attribution: bool = True,
    ) -> Tuple[str, str]:
        """Build category-specific system and user prompts."""

        try:
            formatting_options = ContextFormattingOptions(
                max_length=max_context_length,
                include_attribution=include_source_attribution,
                source_label=self._prompt_config.formatting.get(
                    "source_label", "Source"
                ),
                truncation_indicator=self._prompt_config.formatting.get(
                    "truncation_indicator", "..."
                ),
                min_chunk_size=self._prompt_config.formatting.get(
                    "min_chunk_size", 100
                ),
            )

            no_context_message = self._prompt_config.messages.get(
                "no_context", "No relevant context available."
            )

            result = build_category_prompt(
                query=query,
                context_chunks=context_chunks,
                category=category,
                templates=self._templates,
                formatting_options=formatting_options,
                no_context_message=no_context_message,
            )

            if result.truncated:
                self._log_debug(
                    f"Context truncated for {category.value} prompt: {result.chunks_included} chunks included"
                )

            return result.system_prompt, result.user_prompt

        except KeyError as e:
            self._log_error(f"Template not found: {e}")
            raise

    def get_followup_prompt(
        self,
        original_query: str,
        original_answer: str,
        followup_query: str,
        category: DocumentCategory = DocumentCategory.GENERAL,
    ) -> str:
        """Generate a follow-up prompt that maintains conversation context."""

        try:
            return build_followup_prompt(
                original_query=original_query,
                original_answer=original_answer,
                followup_query=followup_query,
                category=category,
                templates=self._templates,
            )
        except KeyError as e:
            self._log_error(f"Followup template not found: {e}")
            raise

    def suggest_template_improvements(
        self, category: DocumentCategory, usage_stats: Dict[str, Any]
    ) -> List[str]:
        """Suggest improvements for prompt templates based on usage statistics."""
        suggestions = suggest_template_improvements(
            templates=self._templates, category=category, usage_stats=usage_stats
        )

        if suggestions:
            self._log_info(
                f"Generated {len(suggestions)} improvement suggestions for {category.value}"
            )

        return suggestions

    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded templates."""
        stats = calculate_template_stats(self._templates, self.language)
        return {
            "total_categories": stats.total_categories,
            "language": stats.language,
            "categories": stats.categories,
        }

    def validate_templates(self) -> Dict[str, Any]:
        """Validate all loaded templates for common issues."""
        result = validate_templates(self._templates)

        validation_dict = {
            "valid": result.valid,
            "issues": result.issues,
            "warnings": result.warnings,
        }

        if not result.valid:
            self._log_warning(
                f"Template validation failed with {len(result.issues)} issues"
            )

        if result.warnings:
            self._log_debug(
                f"Template validation found {len(result.warnings)} warnings"
            )

        return validation_dict

    def get_missing_templates(
        self,
        required_categories: List[DocumentCategory] = None,
        required_types: List[PromptType] = None,
    ) -> Dict[str, List[str]]:
        """Get missing templates for specified categories and types."""
        if required_categories is None:
            required_categories = list(DocumentCategory)

        if required_types is None:
            required_types = [PromptType.SYSTEM, PromptType.USER]

        return get_missing_templates(
            self._templates, required_categories, required_types
        )

    def find_templates_by_content(self, search_text: str) -> List[Tuple[str, str]]:
        """Find templates containing specific text."""
        matches = find_template_by_content(self._templates, search_text)
        return [
            (category.value, prompt_type.value) for category, prompt_type in matches
        ]


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_enhanced_prompt_builder(
    language: str,
    config_provider: ConfigProvider,
    logger_provider: Optional[LoggerProvider] = None,
) -> _EnhancedPromptBuilder:
    """Factory function to create configured enhanced prompt builder."""
    return _EnhancedPromptBuilder(
        language=language,
        config_provider=config_provider,
        logger_provider=logger_provider,
    )


# ================================
# BACKWARD COMPATIBILITY LAYER
# ================================


class LegacyEnhancedPromptBuilder:
    """Legacy enhanced prompt builder for backward compatibility."""

    def __init__(self, language: str):
        """Initialize with legacy interface pattern."""
        from .enhanced_prompt_templates_providers import \
            create_production_setup

        # Create DI components
        config_provider, logger_provider = create_production_setup()

        # Create modern builder
        self._builder = _EnhancedPromptBuilder(
            language=language,
            config_provider=config_provider,
            logger_provider=logger_provider,
        )

        # Store language for compatibility
        self.language = language

        # Expose legacy attributes for compatibility
        self.templates = self._builder._templates

    def build_prompt(
        self,
        query: str,
        context_chunks: List[str],
        category: DocumentCategory = DocumentCategory.GENERAL,
        max_context_length: int = 2000,
        include_source_attribution: bool = True,
    ) -> Tuple[str, str]:
        """Legacy method interface."""
        return self._builder.build_prompt(
            query=query,
            context_chunks=context_chunks,
            category=category,
            max_context_length=max_context_length,
            include_source_attribution=include_source_attribution,
        )

    def get_followup_prompt(
        self,
        original_query: str,
        original_answer: str,
        followup_query: str,
        category: DocumentCategory = DocumentCategory.GENERAL,
    ) -> str:
        """Legacy method interface."""
        return self._builder.get_followup_prompt(
            original_query=original_query,
            original_answer=original_answer,
            followup_query=followup_query,
            category=category,
        )

    def suggest_template_improvements(
        self, category: DocumentCategory, usage_stats: Dict[str, Any]
    ) -> List[str]:
        """Legacy method interface."""
        return self._builder.suggest_template_improvements(category, usage_stats)

    def get_template_stats(self) -> Dict[str, Any]:
        """Legacy method interface."""
        return self._builder.get_template_stats()

    def validate_templates(self) -> Dict[str, Any]:
        """Legacy method interface."""
        return self._builder.validate_templates()


# ================================
# DUAL INTERFACE SUPPORT
# ================================


def EnhancedPromptBuilder(
    language: str,
    config_provider: Optional[ConfigProvider] = None,
    logger_provider: Optional[LoggerProvider] = None,
):
    """
    Unified EnhancedPromptBuilder factory supporting both interfaces.

    Legacy interface (for backward compatibility):
        EnhancedPromptBuilder(language="hr")

    New DI interface:
        EnhancedPromptBuilder(language="hr", config_provider=provider, logger_provider=provider)
    """
    # New DI interface - config provider provided
    if config_provider is not None:
        return _EnhancedPromptBuilder(
            language=language,
            config_provider=config_provider,
            logger_provider=logger_provider,
        )

    # Legacy interface - only language provided
    else:
        return LegacyEnhancedPromptBuilder(language)


# Legacy factory function for backward compatibility
def create_enhanced_prompt_builder_legacy(language: str) -> LegacyEnhancedPromptBuilder:
    """Create enhanced prompt builder using legacy interface for backward compatibility."""
    return LegacyEnhancedPromptBuilder(language)
