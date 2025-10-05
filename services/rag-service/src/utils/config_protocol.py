"""
Configuration Protocol for Dependency Injection.

This module defines the ConfigProvider protocol to enable dependency injection
for configuration loading, making the system testable while maintaining clean APIs.
"""

from typing import Any, Protocol, cast, runtime_checkable


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration providers to enable dependency injection."""

    def load_config(self, config_name: str, use_cache: bool = True) -> dict[str, Any]:
        """Load a configuration file."""
        ...

    def get_config_section(self, config_name: str, section: str) -> dict[str, Any]:
        """Get a specific section from a config file."""
        ...

    def get_shared_config(self) -> dict[str, Any]:
        """Get shared configuration."""
        ...

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get configuration for specified language."""
        ...

    def get_language_specific_config(self, section: str, language: str) -> dict[str, Any]:
        """Get specific configuration section for specified language."""
        ...

    def get_cleaning_config(self) -> dict[str, Any]:
        """Get general cleaning configuration."""
        ...

    def get_document_cleaning_config(self, language: str) -> dict[str, Any]:
        """Get document cleaning configuration for specified language."""
        ...

    def get_chunking_config(self, language: str) -> dict[str, Any]:
        """Get chunking configuration for specified language."""
        ...

    def get_shared_language_config(self, language: str) -> dict[str, Any]:
        """Get shared configuration for specified language."""
        ...

    def get_categorization_config(self, language: str) -> dict[str, Any]:
        """Get categorization configuration for specified language."""
        ...

    def get_parsing_config(self, language: str) -> dict[str, Any]:
        """Get parsing configuration for specified language."""
        ...

    def get_prompt_config(self, language: str) -> Any:
        """Get prompt configuration for specified language."""
        ...


class DefaultConfigProvider:
    """Default configuration provider using TOML files."""

    def __init__(self):
        # Import at runtime to avoid circular dependencies
        from . import config_loader

        self._config_loader = config_loader

    def load_config(self, config_name: str, use_cache: bool = True) -> dict[str, Any]:
        """Load a configuration file."""
        return self._config_loader.load_config(config_name, use_cache)

    def get_config_section(self, config_name: str, section: str) -> dict[str, Any]:
        """Get a specific section from a config file."""
        return self._config_loader.get_config_section(config_name, section)

    def get_shared_config(self) -> dict[str, Any]:
        """Get shared configuration."""
        return self._config_loader.get_shared_config()

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get configuration for specified language."""
        config = self._config_loader.get_language_config(language)

        # Flatten text_processing section for cleaners compatibility
        if "text_processing" in config:
            text_proc = config["text_processing"]

            # Move word_char_pattern to root level
            if "word_char_pattern" in text_proc:
                config["word_char_pattern"] = text_proc["word_char_pattern"]

            # Move diacritic_map to root level
            if "diacritic_map" in text_proc:
                config["diacritic_map"] = text_proc["diacritic_map"]

            # Move locale to root level
            if "locale" in text_proc:
                config["locale"] = text_proc["locale"]

        return config

    def get_language_specific_config(self, section: str, language: str) -> dict[str, Any]:
        """Get specific configuration section for specified language."""
        return self._config_loader.get_language_specific_config(section, language)

    def get_cleaning_config(self) -> dict[str, Any]:
        """Get general cleaning configuration from main config."""
        main_config = self._config_loader.load_config("config")
        if "cleaning" not in main_config:
            raise KeyError("Missing 'cleaning' section in main configuration")
        return cast(dict[str, Any], main_config["cleaning"])

    def get_document_cleaning_config(self, language: str) -> dict[str, Any]:
        """Get document cleaning configuration for specified language."""
        language_config = self.get_language_config(language)
        if "document_cleaning" not in language_config:
            raise KeyError(f"Missing 'document_cleaning' section in {language} configuration")
        return cast(dict[str, Any], language_config["document_cleaning"])

    def get_chunking_config(self, language: str) -> dict[str, Any]:
        """Get chunking configuration for specified language, merged with main config."""
        # Get base chunking config from main config
        main_config = self._config_loader.load_config("config")
        base_chunking = main_config["chunking"]

        # Get language-specific chunking config
        language_config = self.get_language_config(language)
        lang_chunking = language_config["chunking"]

        # Merge configs (language-specific overrides main config)
        merged_config = {**base_chunking, **lang_chunking}
        return cast(dict[str, Any], merged_config)

    def get_shared_language_config(self, language: str) -> dict[str, Any]:
        """Get shared configuration for specified language."""
        language_config = self.get_language_config(language)
        if "shared" not in language_config:
            raise KeyError(f"Missing 'shared' section in {language} configuration")
        return cast(dict[str, Any], language_config["shared"])

    def get_categorization_config(self, language: str) -> dict[str, Any]:
        """Get categorization configuration for specified language."""
        language_config = self.get_language_config(language)

        # Get shared categorization data
        shared_config = language_config["shared"]
        shared_categorization = shared_config["categorization"]
        shared_patterns = shared_config["patterns"]

        # Get language-specific categorization data
        categorization_config = language_config["categorization"]

        # Merge shared and language-specific data
        merged_config = {
            **shared_categorization,  # Shared categories, cultural_keywords, complexity_thresholds
            **categorization_config,  # Language-specific indicators (overrides shared)
            "patterns": shared_patterns,  # Shared patterns
        }

        return cast(dict[str, Any], merged_config)

    def get_parsing_config(self, language: str) -> dict[str, Any]:
        """Get parsing configuration for specified language."""
        language_config = self.get_language_config(language)
        if "response_parsing" not in language_config:
            raise KeyError(f"Missing 'response_parsing' section in {language} configuration")
        return cast(dict[str, Any], language_config["response_parsing"])

    def get_prompt_config(self, language: str) -> Any:
        """Get prompt configuration for specified language."""
        language_config = self.get_language_config(language)
        if "prompts" not in language_config:
            raise KeyError(f"Missing 'prompts' section in {language} configuration")

        prompts_config = language_config["prompts"]

        # Import PromptConfig at runtime to avoid circular dependencies
        from ..generation.enhanced_prompt_templates import PromptConfig

        # Create a proper PromptConfig object
        return PromptConfig(
            category_templates={},  # Empty for now since not in config
            messages={k: v for k, v in prompts_config.items() if isinstance(v, str)},
            formatting={},
            language=language,
        )


# Global default provider - can be overridden for testing
_default_provider: ConfigProvider = DefaultConfigProvider()


def set_config_provider(provider: ConfigProvider) -> None:
    """Set the global configuration provider (mainly for testing)."""
    global _default_provider
    _default_provider = provider


def get_config_provider() -> ConfigProvider:
    """Get the current configuration provider."""
    return _default_provider


def reset_config_provider() -> None:
    """Reset to default configuration provider."""
    global _default_provider
    _default_provider = DefaultConfigProvider()
