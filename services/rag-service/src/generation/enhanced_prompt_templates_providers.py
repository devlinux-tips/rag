"""
Provider implementations for enhanced prompt templates dependency injection.
Standard and mock providers for configurable prompt template system.
"""

import logging

from src.generation.enhanced_prompt_templates import PromptConfig, PromptType
from src.retrieval.categorization import CategoryType

from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start, log_data_transformation

# ================================
# STANDARD PROVIDERS
# ================================


class ConfigProvider:
    """Configuration provider using config system."""

    def __init__(self):
        """Initialize config provider."""
        self._config_cache: dict[str, PromptConfig] = {}

    def get_prompt_config(self, language: str) -> PromptConfig:
        """Get prompt configuration from real config system."""
        if language not in self._config_cache:
            self._config_cache[language] = self._load_config_from_system(language)
        return self._config_cache[language]

    def _load_config_from_system(self, language: str) -> PromptConfig:
        """Load configuration from the real system."""
        try:
            # Import at runtime to avoid circular dependencies
            from ..utils.config_loader import get_language_specific_config

            # Load prompts, messages, and formatting configurations
            prompts_config = get_language_specific_config("prompts", language)
            messages_config = get_language_specific_config("messages", language)
            formatting_config = get_language_specific_config("formatting", language)

            # Parse category templates
            category_templates: dict[CategoryType, dict[PromptType, str]] = {}

            for category_name, category_data in prompts_config.items():
                try:
                    category = CategoryType(category_name)
                    category_templates[category] = {}

                    for prompt_type_name, template_text in category_data.items():
                        try:
                            prompt_type = PromptType(prompt_type_name)
                            category_templates[category][prompt_type] = template_text
                        except ValueError:
                            # Skip unknown prompt types
                            continue

                except ValueError:
                    # Skip unknown categories
                    continue

            return PromptConfig(
                category_templates=category_templates,
                messages=messages_config,
                formatting=formatting_config,
                language=language,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load prompt configuration for language '{language}': {e}") from e


class StandardLoggerProvider:
    """Standard logger provider using Python's logging system."""

    def __init__(self, logger_name: str = __name__):
        """Initialize with logger."""
        self.logger = logging.getLogger(logger_name)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_prompt_builder(logger_name: str | None = None) -> tuple:
    """
    Create prompt builder with real components.

    Args:
        logger_name: Optional logger name override

    Returns:
        Tuple of (config_provider, logger_provider)
    """
    config_provider = ConfigProvider()
    logger_provider = StandardLoggerProvider(logger_name or __name__)

    return config_provider, logger_provider


def create_test_config(
    language: str = "hr", include_followup: bool = True, include_technical: bool = True, include_cultural: bool = True
) -> PromptConfig:
    """Create test configuration with customizable parameters."""

    category_templates = {}

    # Always include GENERAL
    category_templates[CategoryType.GENERAL] = {
        PromptType.SYSTEM: "You are a helpful assistant. Answer based on context.",
        PromptType.USER: "Q: {query}\nContext: {context}\nA:",
    }

    if include_followup:
        category_templates[CategoryType.GENERAL][PromptType.FOLLOWUP] = (
            "Previous: {original_query} -> {original_answer}\nNew: {followup_query}\nAnswer:"
        )

    if include_technical:
        category_templates[CategoryType.TECHNICAL] = {
            PromptType.SYSTEM: "Technical expert. Provide detailed answers.",
            PromptType.USER: "Tech Q: {query}\nDocs: {context}\nTech A:",
        }

    if include_cultural:
        category_templates[CategoryType.CULTURAL] = {
            PromptType.SYSTEM: "Cultural expert. Provide culturally aware answers.",
            PromptType.USER: "Cultural Q: {query}\nContext: {context}\nCultural A:",
        }

    messages = {
        "no_context": f"No context available ({language})",
        "error_template_missing": f"Template missing ({language})",
    }

    formatting = {
        "source_label": "Source" if language == "en" else "Izvor",
        "truncation_indicator": "..." if language == "en" else "...",
        "min_chunk_size": "100",
    }

    return PromptConfig(
        category_templates=category_templates, messages=messages, formatting=formatting, language=language
    )


def create_minimal_config(language: str = "hr") -> PromptConfig:
    """Create minimal configuration for basic testing."""
    category_templates = {
        CategoryType.GENERAL: {PromptType.SYSTEM: "Basic assistant.", PromptType.USER: "{query} - {context}"}
    }

    return PromptConfig(
        category_templates=category_templates,
        messages={"no_context": "No context."},
        formatting={"source_label": "Source"},
        language=language,
    )


def create_invalid_config(language: str = "hr") -> PromptConfig:
    """Create configuration with missing required templates for testing error scenarios."""
    # Missing USER template for GENERAL category
    category_templates = {
        CategoryType.GENERAL: {
            PromptType.SYSTEM: "You are a helpful assistant."
            # Missing PromptType.USER intentionally
        }
    }

    return PromptConfig(
        category_templates=category_templates,
        messages={"no_context": "No context."},
        formatting={"source_label": "Source"},
        language=language,
    )


# ================================
# INTEGRATION HELPERS
# ================================


def create_development_prompt_builder():
    """Create prompt builder configured for development/testing."""
    get_system_logger()
    log_component_start(
        "prompt_templates_providers", "create_development_prompt_builder", language="hr", environment="development"
    )

    from .enhanced_prompt_templates import create_enhanced_prompt_builder

    config_provider, logger_provider = create_prompt_builder()

    log_data_transformation(
        "prompt_templates_providers",
        "builder_creation",
        "Input: development environment request",
        "Output: Croatian prompt builder with mock providers",
        language="hr",
        provider_type="mock",
    )

    builder = create_enhanced_prompt_builder(
        language="hr", config_provider=config_provider, logger_provider=logger_provider
    )

    log_component_end(
        "prompt_templates_providers",
        "create_development_prompt_builder",
        "Successfully created development prompt builder",
        language="hr",
        builder_type="enhanced",
    )

    return builder


# ================================
# TEMPLATE BUILDING HELPERS
# ================================


def build_category_templates(templates: dict[str, str]) -> dict[CategoryType, dict[PromptType, str]]:
    """Helper to build category templates from flat dictionary."""
    result: dict[CategoryType, dict[PromptType, str]] = {}

    for key, template in templates.items():
        if "." not in key:
            continue

        category_name, prompt_type_name = key.split(".", 1)

        try:
            category = CategoryType(category_name)
            prompt_type = PromptType(prompt_type_name)

            if category not in result:
                result[category] = {}

            result[category][prompt_type] = template

        except ValueError:
            continue

    return result


def create_template_variants(base_template: str, variants: dict[str, str]) -> dict[str, str]:
    """Create template variants by substituting parts of base template."""
    result = {"base": base_template}

    for variant_name, substitution in variants.items():
        result[variant_name] = base_template.replace("{variant}", substitution)

    return result
