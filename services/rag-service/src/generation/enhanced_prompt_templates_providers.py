"""
Provider implementations for enhanced prompt templates dependency injection.
Production and mock providers for configurable prompt template system.
"""

import logging
from typing import Dict, List, Optional

from src.generation.enhanced_prompt_templates import (
    ConfigProvider,
    LoggerProvider,
    PromptConfig,
    PromptType,
)
from src.retrieval.categorization import DocumentCategory

# ================================
# MOCK PROVIDERS FOR TESTING
# ================================


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self, config: Optional[PromptConfig] = None):
        """Initialize with optional mock configuration."""
        self.config = config or self._create_default_config()
        self.call_history: List[str] = []

    def _create_default_config(self) -> PromptConfig:
        """Create default test configuration."""
        category_templates = {
            DocumentCategory.GENERAL: {
                PromptType.SYSTEM: "You are a helpful assistant. Answer questions based on the provided context.",
                PromptType.USER: "Question: {query}\n\nContext: {context}\n\nAnswer:",
                PromptType.FOLLOWUP: "Previous question: {original_query}\nPrevious answer: {original_answer}\n\nFollow-up question: {followup_query}\n\nAnswer:",
            },
            DocumentCategory.TECHNICAL: {
                PromptType.SYSTEM: "You are a technical expert. Provide detailed technical answers based on the context.",
                PromptType.USER: "Technical question: {query}\n\nTechnical documentation: {context}\n\nDetailed answer:",
                PromptType.FOLLOWUP: "Previous technical question: {original_query}\nPrevious answer: {original_answer}\n\nFollow-up: {followup_query}\n\nTechnical answer:",
            },
            DocumentCategory.CULTURAL: {
                PromptType.SYSTEM: "You are a cultural expert. Provide culturally sensitive answers based on context.",
                PromptType.USER: "Cultural question: {query}\n\nCultural context: {context}\n\nCulturally aware answer:",
                PromptType.FOLLOWUP: "Previous cultural question: {original_query}\nPrevious answer: {original_answer}\n\nFollow-up: {followup_query}\n\nCultural answer:",
            },
        }

        messages = {
            "no_context": "No relevant context available.",
            "error_template_missing": "Template not found for this category.",
            "truncation_notice": "Some content was truncated due to length limits.",
        }

        formatting = {
            "source_label": "Source",
            "truncation_indicator": "...",
            "min_chunk_size": "100",
            "max_context_length": "2000",
        }

        return PromptConfig(
            category_templates=category_templates,
            messages=messages,
            formatting=formatting,
            language="hr",
        )

    def set_config(self, config: PromptConfig) -> None:
        """Set mock configuration."""
        self.config = config

    def add_category_template(
        self, category: DocumentCategory, prompt_type: PromptType, template: str
    ) -> None:
        """Add a template for specific category and type."""
        if category not in self.config.category_templates:
            self.config.category_templates[category] = {}
        self.config.category_templates[category][prompt_type] = template

    def remove_template(
        self, category: DocumentCategory, prompt_type: PromptType
    ) -> None:
        """Remove a template (for testing missing template scenarios)."""
        if category in self.config.category_templates:
            if prompt_type in self.config.category_templates[category]:
                del self.config.category_templates[category][prompt_type]

    def get_prompt_config(self, language: str) -> PromptConfig:
        """Get prompt configuration for language."""
        self.call_history.append(f"get_prompt_config({language})")

        # Return config with requested language
        config_copy = PromptConfig(
            category_templates=self.config.category_templates,
            messages=self.config.messages,
            formatting=self.config.formatting,
            language=language,
        )
        return config_copy


class MockLoggerProvider:
    """Mock logger provider that captures messages for testing."""

    def __init__(self):
        """Initialize message capture."""
        self.messages: Dict[str, List[str]] = {
            "info": [],
            "debug": [],
            "warning": [],
            "error": [],
        }

    def info(self, message: str) -> None:
        """Capture info message."""
        self.messages["info"].append(message)

    def debug(self, message: str) -> None:
        """Capture debug message."""
        self.messages["debug"].append(message)

    def warning(self, message: str) -> None:
        """Capture warning message."""
        self.messages["warning"].append(message)

    def error(self, message: str) -> None:
        """Capture error message."""
        self.messages["error"].append(message)

    def clear_messages(self) -> None:
        """Clear all captured messages."""
        for level in self.messages:
            self.messages[level].clear()

    def get_messages(self, level: str = None) -> Dict[str, List[str]] | List[str]:
        """Get captured messages by level or all messages."""
        if level:
            if level not in self.messages:
                return []
            return self.messages[level]
        return self.messages


# ================================
# PRODUCTION PROVIDERS
# ================================


class ProductionConfigProvider:
    """Production configuration provider using real config system."""

    def __init__(self):
        """Initialize production config provider."""
        self._config_cache: Dict[str, PromptConfig] = {}

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
            category_templates = {}

            for category_name, category_data in prompts_config.items():
                try:
                    category = DocumentCategory(category_name)
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
            raise RuntimeError(
                f"Failed to load prompt configuration for language '{language}': {e}"
            )


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


def create_mock_setup(
    config: Optional[PromptConfig] = None,
    custom_templates: Optional[Dict[DocumentCategory, Dict[PromptType, str]]] = None,
    custom_messages: Optional[Dict[str, str]] = None,
    custom_formatting: Optional[Dict[str, str]] = None,
    language: str = "hr",
) -> tuple:
    """
    Create complete mock setup for testing.

    Args:
        config: Optional mock configuration
        custom_templates: Optional custom category templates
        custom_messages: Optional custom messages
        custom_formatting: Optional custom formatting options
        language: Language for the configuration

    Returns:
        Tuple of (config_provider, logger_provider)
    """
    # Create mock components
    config_provider = MockConfigProvider(config)
    logger_provider = MockLoggerProvider()

    # Customize configuration if provided
    if custom_templates or custom_messages or custom_formatting:
        base_config = config_provider.config

        updated_config = PromptConfig(
            category_templates=custom_templates or base_config.category_templates,
            messages=custom_messages or base_config.messages,
            formatting=custom_formatting or base_config.formatting,
            language=language,
        )
        config_provider.set_config(updated_config)

    return config_provider, logger_provider


def create_production_setup(logger_name: Optional[str] = None) -> tuple:
    """
    Create production setup with real components.

    Args:
        logger_name: Optional logger name override

    Returns:
        Tuple of (config_provider, logger_provider)
    """
    config_provider = ProductionConfigProvider()
    logger_provider = StandardLoggerProvider(logger_name or __name__)

    return config_provider, logger_provider


def create_test_config(
    language: str = "hr",
    include_followup: bool = True,
    include_technical: bool = True,
    include_cultural: bool = True,
) -> PromptConfig:
    """Create test configuration with customizable parameters."""

    category_templates = {}

    # Always include GENERAL
    category_templates[DocumentCategory.GENERAL] = {
        PromptType.SYSTEM: "You are a helpful assistant. Answer based on context.",
        PromptType.USER: "Q: {query}\nContext: {context}\nA:",
    }

    if include_followup:
        category_templates[DocumentCategory.GENERAL][
            PromptType.FOLLOWUP
        ] = "Previous: {original_query} -> {original_answer}\nNew: {followup_query}\nAnswer:"

    if include_technical:
        category_templates[DocumentCategory.TECHNICAL] = {
            PromptType.SYSTEM: "Technical expert. Provide detailed answers.",
            PromptType.USER: "Tech Q: {query}\nDocs: {context}\nTech A:",
        }

    if include_cultural:
        category_templates[DocumentCategory.CULTURAL] = {
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
        category_templates=category_templates,
        messages=messages,
        formatting=formatting,
        language=language,
    )


def create_minimal_config(language: str = "hr") -> PromptConfig:
    """Create minimal configuration for basic testing."""
    category_templates = {
        DocumentCategory.GENERAL: {
            PromptType.SYSTEM: "Basic assistant.",
            PromptType.USER: "{query} - {context}",
        }
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
        DocumentCategory.GENERAL: {
            PromptType.SYSTEM: "You are a helpful assistant.",
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
    from .enhanced_prompt_templates import create_enhanced_prompt_builder

    config_provider, logger_provider = create_production_setup()
    return create_enhanced_prompt_builder(
        language="hr", config_provider=config_provider, logger_provider=logger_provider
    )


def create_test_prompt_builder(
    language: str = "hr",
    config: Optional[PromptConfig] = None,
    custom_templates: Optional[Dict[DocumentCategory, Dict[PromptType, str]]] = None,
    custom_messages: Optional[Dict[str, str]] = None,
    custom_formatting: Optional[Dict[str, str]] = None,
):
    """Create prompt builder configured for testing."""
    from .enhanced_prompt_templates import create_enhanced_prompt_builder

    config_provider, logger_provider = create_mock_setup(
        config=config,
        custom_templates=custom_templates,
        custom_messages=custom_messages,
        custom_formatting=custom_formatting,
        language=language,
    )

    return create_enhanced_prompt_builder(
        language=language,
        config_provider=config_provider,
        logger_provider=logger_provider,
    ), (config_provider, logger_provider)


# ================================
# TEMPLATE BUILDING HELPERS
# ================================


def build_category_templates(
    templates: Dict[str, str]
) -> Dict[DocumentCategory, Dict[PromptType, str]]:
    """Helper to build category templates from flat dictionary."""
    result = {}

    for key, template in templates.items():
        if "." not in key:
            continue

        category_name, prompt_type_name = key.split(".", 1)

        try:
            category = DocumentCategory(category_name)
            prompt_type = PromptType(prompt_type_name)

            if category not in result:
                result[category] = {}

            result[category][prompt_type] = template

        except ValueError:
            continue

    return result


def create_template_variants(
    base_template: str, variants: Dict[str, str]
) -> Dict[str, str]:
    """Create template variants by substituting parts of base template."""
    result = {"base": base_template}

    for variant_name, substitution in variants.items():
        result[variant_name] = base_template.replace("{variant}", substitution)

    return result
