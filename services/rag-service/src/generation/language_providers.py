"""
Language configuration providers for Ollama client dependency injection.
Provides testable language configuration abstraction layer.
"""

from typing import Any, cast

from ..utils.config_loader import ConfigError as ConfigurationError


class DefaultLanguageProvider:
    """
    Production language config provider using actual configuration files.
    """

    def get_formal_prompts(self, language: str) -> dict[str, str]:
        """Get formal prompt templates for language.

        Raises:
            ConfigurationError: If language configuration is missing or invalid
        """
        try:
            from ..utils.config_loader import get_language_specific_config

            language_config = get_language_specific_config("prompts", language)
            if "formal" not in language_config:
                raise ConfigurationError(f"Missing 'formal' prompts section in {language}.toml configuration")
            return cast(dict[str, str], language_config["formal"])
        except Exception as e:
            raise ConfigurationError(f"Failed to load formal prompts for {language}: {e}") from e

    def get_error_template(self, language: str) -> str:
        """Get error message template for language.

        Raises:
            ConfigurationError: If language configuration is missing or invalid
        """
        try:
            from ..utils.config_loader import get_language_specific_config

            language_config = get_language_specific_config("prompts", language)
            if "error_message_template" not in language_config:
                raise ConfigurationError(f"Missing 'error_message_template' in {language}.toml prompts configuration")
            return cast(str, language_config["error_message_template"])
        except Exception as e:
            raise ConfigurationError(f"Failed to load error template for {language}: {e}") from e

    def get_confidence_settings(self, language: str) -> dict[str, Any]:
        """Get confidence calculation settings for language.

        Raises:
            ConfigurationError: If language configuration is missing or invalid
        """
        try:
            from ..utils.config_loader import get_language_specific_config

            return cast(dict[str, Any], get_language_specific_config("confidence", language))
        except Exception as e:
            raise ConfigurationError(f"Failed to load confidence settings for {language}: {e}") from e


class MockLanguageProvider:
    """
    Mock language config provider for testing.
    Allows complete control over language configuration for unit tests.
    """

    def __init__(self):
        self.formal_prompts = {}
        self.error_templates = {}
        self.confidence_settings = {}
        self.call_log = []

    def set_formal_prompts(self, language: str, prompts: dict[str, str]):
        """Set formal prompts for language."""
        self.formal_prompts[language] = prompts

    def set_error_template(self, language: str, template: str):
        """Set error template for language."""
        self.error_templates[language] = template

    def set_confidence_settings(self, language: str, settings: dict[str, Any]):
        """Set confidence settings for language."""
        self.confidence_settings[language] = settings

    def get_calls(self) -> list:
        """Get log of all method calls made."""
        return self.call_log.copy()

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    def get_formal_prompts(self, language: str) -> dict[str, str]:
        """Get formal prompt templates for language."""
        self.call_log.append({"method": "get_formal_prompts", "language": language})

        if language not in self.formal_prompts:
            return {"formal_instruction": f"Mock formal instruction for {language}"}
        return cast(dict[str, str], self.formal_prompts[language])

    def get_error_template(self, language: str) -> str:
        """Get error message template for language."""
        self.call_log.append({"method": "get_error_template", "language": language})

        if language not in self.error_templates:
            return f"Mock error template for {language}: {{error}}"
        return cast(str, self.error_templates[language])

    def get_confidence_settings(self, language: str) -> dict[str, Any]:
        """Get confidence calculation settings for language."""
        self.call_log.append({"method": "get_confidence_settings", "language": language})

        if language not in self.confidence_settings:
            return {"error_phrases": [f"mock_error_{language}", "test_failure"]}
        return cast(dict[str, Any], self.confidence_settings[language])
