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


class ProductionConfigProvider:
    """Production configuration provider using real TOML files."""

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
        base_chunking = main_config.get("chunking", {})

        # Get language-specific chunking config
        language_config = self.get_language_config(language)
        lang_chunking = language_config.get("chunking", {})

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
        shared_config = language_config.get("shared", {})
        shared_categorization = shared_config.get("categorization", {})
        shared_patterns = shared_config.get("patterns", {})

        # Get language-specific categorization data
        categorization_config = language_config.get("categorization", {})

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


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self, mock_configs: dict[str, dict[str, Any]] | None = None):
        """Initialize with mock configuration data."""
        self.mock_configs = mock_configs or {}
        self.mock_language_configs: dict[str, dict[str, Any]] = {}
        self.mock_shared_config: dict[str, Any] = {}

    def set_config(self, config_name: str, config_data: dict[str, Any]) -> None:
        """Set mock configuration data."""
        self.mock_configs[config_name] = config_data

    def set_language_config(self, language: str, config_data: dict[str, Any]) -> None:
        """Set mock language configuration data."""
        self.mock_language_configs[language] = config_data

    def set_shared_config(self, config_data: dict[str, Any]) -> None:
        """Set mock shared configuration data."""
        self.mock_shared_config = config_data

    def load_config(self, config_name: str, use_cache: bool = True) -> dict[str, Any]:
        """Load mock configuration."""
        if config_name not in self.mock_configs:
            raise KeyError(f"Mock config '{config_name}' not found")
        return self.mock_configs[config_name]

    def get_config_section(self, config_name: str, section: str) -> dict[str, Any]:
        """Get mock configuration section."""
        config = self.load_config(config_name)
        if section not in config:
            raise KeyError(f"Mock section '{section}' not found in '{config_name}'")
        return cast(dict[str, Any], config[section])

    def get_shared_config(self) -> dict[str, Any]:
        """Get mock shared configuration."""
        return self.mock_shared_config

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get mock language configuration."""
        if language not in self.mock_language_configs:
            raise KeyError(f"Mock language config '{language}' not found")
        return cast(dict[str, Any], self.mock_language_configs[language])

    def get_language_specific_config(self, section: str, language: str) -> dict[str, Any]:
        """Get mock language-specific configuration section."""
        # Handle patterns section which would be in shared
        if section == "patterns":
            return {"general": [".*"], "cultural": ["culture.*"], "technical": ["tech.*"]}

        language_config = self.get_language_config(language)
        if section not in language_config:
            raise KeyError(f"Mock section '{section}' not found in language '{language}'")
        return cast(dict[str, Any], language_config[section])

    def get_cleaning_config(self) -> dict[str, Any]:
        """Get mock cleaning configuration."""
        return {
            "multiple_whitespace": True,
            "multiple_linebreaks": True,
            "min_meaningful_words": 3,
            "min_word_char_ratio": 0.7,
        }

    def get_document_cleaning_config(self, language: str) -> dict[str, Any]:
        """Get mock document cleaning configuration."""
        return {"header_footer_patterns": []}

    def get_chunking_config(self, language: str) -> dict[str, Any]:
        """Get mock chunking configuration."""
        return {
            "sentence_endings": [".", "!", "?"],
            "abbreviations": [],
            "min_sentence_length": 10,
            "sentence_ending_pattern": "[.!?]+",
        }

    def get_shared_language_config(self, language: str) -> dict[str, Any]:
        """Get mock shared language configuration."""
        return {"chars_pattern": "[^\\w\\s]", "stopwords": {"words": []}}

    def get_categorization_config(self, language: str) -> dict[str, Any]:
        """Get mock categorization configuration."""
        return {
            "categories": ["general", "technical", "cultural"],
            "patterns": {"general": [".*"], "cultural": ["culture.*"], "technical": ["tech.*"]},
            "cultural_keywords": ["culture", "traditional"],
            "complexity_thresholds": {"simple": 0.3, "medium": 0.6, "complex": 0.8},
            "retrieval_strategies": {"general": "semantic", "cultural": "semantic", "technical": "dense"},
        }

    def get_parsing_config(self, language: str) -> dict[str, Any]:
        """Get mock parsing configuration."""
        return {
            "validate_responses": True,
            "extract_confidence_scores": True,
            "parse_citations": True,
            "handle_incomplete_responses": True,
            "max_response_length": 3000,
            "min_response_length": 50,
            "filter_hallucinations": True,
            "require_source_grounding": True,
            "confidence_threshold": 0.7,
            "response_format": "markdown",
            "include_metadata": True,
        }

    def get_prompt_config(self, language: str) -> Any:
        """Get mock prompt configuration."""
        # Import PromptConfig at runtime to avoid circular dependencies
        from ..generation.enhanced_prompt_templates import PromptConfig

        return PromptConfig(
            category_templates={},
            messages={
                "system_base": "You are an AI assistant.",
                "context_intro": "Based on the following context:",
                "answer_intro": "Answer:",
            },
            formatting={},
            language=language,
        )


# Global default provider - can be overridden for testing
_default_provider: ConfigProvider = ProductionConfigProvider()


def set_config_provider(provider: ConfigProvider) -> None:
    """Set the global configuration provider (mainly for testing)."""
    global _default_provider
    _default_provider = provider


def get_config_provider() -> ConfigProvider:
    """Get the current configuration provider."""
    return _default_provider


def reset_config_provider() -> None:
    """Reset to production configuration provider."""
    global _default_provider
    _default_provider = ProductionConfigProvider()
