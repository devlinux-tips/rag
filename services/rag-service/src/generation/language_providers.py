"""
Language configuration providers for Ollama client dependency injection.
Provides testable language configuration abstraction layer.
"""

from typing import Any, Dict, Optional

from .ollama_client import LanguageConfigProvider


class DefaultLanguageProvider:
    """
    Production language config provider using actual configuration files.
    """

    def get_formal_prompts(self, language: str) -> Dict[str, str]:
        """Get formal prompt templates for language."""
        try:
            from ..utils.config_loader import get_language_specific_config

            language_config = get_language_specific_config("prompts", language)
            return language_config.get("formal", {})
        except Exception:
            # Fallback for missing configuration
            if language == "hr":
                return {
                    "formal_instruction": "Molim te odgovori ljubazno i profesionalno na hrvatskom jeziku."
                }
            elif language == "en":
                return {
                    "formal_instruction": "Please respond politely and professionally in English."
                }
            else:
                return {}

    def get_error_template(self, language: str) -> str:
        """Get error message template for language."""
        try:
            from ..utils.config_loader import get_language_specific_config

            language_config = get_language_specific_config("prompts", language)
            return language_config.get(
                "error_message_template", "An error occurred: {error}"
            )
        except Exception:
            # Fallback templates
            if language == "hr":
                return "Dogodila se greška: {error}"
            elif language == "en":
                return "An error occurred: {error}"
            else:
                return "Error: {error}"

    def get_confidence_settings(self, language: str) -> Dict[str, Any]:
        """Get confidence calculation settings for language."""
        try:
            from ..utils.config_loader import get_language_specific_config

            return get_language_specific_config("confidence", language)
        except Exception:
            # Fallback confidence settings
            if language == "hr":
                return {
                    "error_phrases": [
                        "greška",
                        "problem",
                        "ne mogu",
                        "žao mi je",
                        "nema podataka",
                    ]
                }
            elif language == "en":
                return {
                    "error_phrases": ["error", "failed", "sorry", "cannot", "no data"]
                }
            else:
                return {"error_phrases": ["error", "failed", "sorry"]}


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

    def set_formal_prompts(self, language: str, prompts: Dict[str, str]):
        """Set formal prompts for language."""
        self.formal_prompts[language] = prompts

    def set_error_template(self, language: str, template: str):
        """Set error template for language."""
        self.error_templates[language] = template

    def set_confidence_settings(self, language: str, settings: Dict[str, Any]):
        """Set confidence settings for language."""
        self.confidence_settings[language] = settings

    def get_calls(self) -> list:
        """Get log of all method calls made."""
        return self.call_log.copy()

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    def get_formal_prompts(self, language: str) -> Dict[str, str]:
        """Get formal prompt templates for language."""
        self.call_log.append({"method": "get_formal_prompts", "language": language})

        return self.formal_prompts.get(
            language, {"formal_instruction": f"Mock formal instruction for {language}"}
        )

    def get_error_template(self, language: str) -> str:
        """Get error message template for language."""
        self.call_log.append({"method": "get_error_template", "language": language})

        return self.error_templates.get(
            language, f"Mock error template for {language}: {{error}}"
        )

    def get_confidence_settings(self, language: str) -> Dict[str, Any]:
        """Get confidence calculation settings for language."""
        self.call_log.append(
            {"method": "get_confidence_settings", "language": language}
        )

        return self.confidence_settings.get(
            language, {"error_phrases": [f"mock_error_{language}", "test_failure"]}
        )


class StaticLanguageProvider:
    """
    Static language provider with predefined configurations.
    Useful for testing and simple deployments.
    """

    def __init__(self, static_config: Optional[Dict[str, Dict[str, Any]]] = None):
        self.config = static_config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default static configuration."""
        return {
            "hr": {
                "formal_prompts": {
                    "formal_instruction": "Molim te odgovori ljubazno i profesionalno na hrvatskom jeziku."
                },
                "error_template": "Dogodila se greška: {error}",
                "confidence_settings": {
                    "error_phrases": [
                        "greška",
                        "problem",
                        "ne mogu",
                        "žao mi je",
                        "nema podataka",
                    ]
                },
            },
            "en": {
                "formal_prompts": {
                    "formal_instruction": "Please respond politely and professionally in English."
                },
                "error_template": "An error occurred: {error}",
                "confidence_settings": {
                    "error_phrases": ["error", "failed", "sorry", "cannot", "no data"]
                },
            },
        }

    def get_formal_prompts(self, language: str) -> Dict[str, str]:
        """Get formal prompt templates for language."""
        language_config = self.config.get(language, {})
        return language_config.get("formal_prompts", {})

    def get_error_template(self, language: str) -> str:
        """Get error message template for language."""
        language_config = self.config.get(language, {})
        return language_config.get("error_template", "Error: {error}")

    def get_confidence_settings(self, language: str) -> Dict[str, Any]:
        """Get confidence calculation settings for language."""
        language_config = self.config.get(language, {})
        return language_config.get("confidence_settings", {"error_phrases": ["error"]})
