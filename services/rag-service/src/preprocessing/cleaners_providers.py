"""
Provider implementations for multilingual text cleaning dependency injection.
Standard and mock providers for configurable text cleaning operations.
"""

import locale
import logging
import os
from typing import Any, cast

# ================================
# STANDARD PROVIDERS
# ================================


class ConfigProvider:
    """Standard provider using real TOML files."""

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get language-specific text processing configuration."""
        from ..utils.config_loader import get_language_specific_config

        return get_language_specific_config("text_processing", language)

    def get_cleaning_config(self) -> dict[str, Any]:
        """Get general cleaning configuration."""
        from ..utils.config_loader import get_cleaning_config

        return get_cleaning_config()

    def get_document_cleaning_config(self, language: str) -> dict[str, Any]:
        """Get document cleaning configuration."""
        from ..utils.config_loader import get_language_specific_config

        return get_language_specific_config("document_cleaning", language)

    def get_chunking_config(self, language: str) -> dict[str, Any]:
        """Get chunking configuration (merged)."""
        from ..utils.config_loader import get_chunking_config, get_language_specific_config

        # Load main chunking config
        main_chunking_config = get_chunking_config()

        # Load language-specific overrides
        language_chunking_config = get_language_specific_config("chunking", language)

        # Merge configs: language-specific overrides main config
        merged_config = {**main_chunking_config, **language_chunking_config}

        return merged_config

    def get_shared_language_config(self, language: str) -> dict[str, Any]:
        """Get shared language configuration."""
        from ..utils.config_loader import get_language_shared

        return get_language_shared(language)


class LoggerProvider:
    """Standard provider using actual Python logging."""

    def __init__(self, logger_name: str | None = None):
        """Initialize with logger name."""
        self.logger = logging.getLogger(logger_name or __name__)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


class EnvironmentProvider:
    """Standard provider using real system operations."""

    def set_environment_variable(self, key: str, value: str) -> None:
        """Set environment variable."""
        os.environ[key] = value

    def set_locale(self, category: int, locale_name: str) -> None:
        """Set locale."""
        locale.setlocale(category, locale_name)


# ================================
# MOCK PROVIDERS FOR TESTING
# ================================


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self):
        """Initialize with mock configurations."""
        self.language_configs: dict[str, dict[str, Any]] = {}
        self.cleaning_config: dict[str, Any] = {}
        self.document_cleaning_configs: dict[str, dict[str, Any]] = {}
        self.chunking_configs: dict[str, dict[str, Any]] = {}
        self.shared_language_configs: dict[str, dict[str, Any]] = {}

    def set_language_config(self, language: str, config: dict[str, Any]) -> None:
        """Set mock language configuration."""
        self.language_configs[language] = config

    def set_cleaning_config(self, config: dict[str, Any]) -> None:
        """Set mock cleaning configuration."""
        self.cleaning_config = config

    def set_document_cleaning_config(self, language: str, config: dict[str, Any]) -> None:
        """Set mock document cleaning configuration."""
        self.document_cleaning_configs[language] = config

    def set_chunking_config(self, language: str, config: dict[str, Any]) -> None:
        """Set mock chunking configuration."""
        self.chunking_configs[language] = config

    def set_shared_language_config(self, language: str, config: dict[str, Any]) -> None:
        """Set mock shared language configuration."""
        self.shared_language_configs[language] = config

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get mock language configuration."""
        if language not in self.language_configs:
            raise KeyError(f"Mock language config '{language}' not found")
        return self.language_configs[language]

    def get_cleaning_config(self) -> dict[str, Any]:
        """Get mock cleaning configuration."""
        if not self.cleaning_config:
            raise KeyError("Mock cleaning config not set")
        return self.cleaning_config

    def get_document_cleaning_config(self, language: str) -> dict[str, Any]:
        """Get mock document cleaning configuration."""
        if language not in self.document_cleaning_configs:
            raise KeyError(f"Mock document cleaning config '{language}' not found")
        return self.document_cleaning_configs[language]

    def get_chunking_config(self, language: str) -> dict[str, Any]:
        """Get mock chunking configuration."""
        if language not in self.chunking_configs:
            raise KeyError(f"Mock chunking config '{language}' not found")
        return self.chunking_configs[language]

    def get_shared_language_config(self, language: str) -> dict[str, Any]:
        """Get mock shared language configuration."""
        if language not in self.shared_language_configs:
            raise KeyError(f"Mock shared language config '{language}' not found")
        return self.shared_language_configs[language]


class MockLoggerProvider:
    """Mock logger provider for testing."""

    def __init__(self):
        """Initialize mock logger."""
        self.debug_messages = []
        self.info_messages = []
        self.error_messages = []

    def debug(self, message: str) -> None:
        """Record debug message."""
        self.debug_messages.append(message)

    def info(self, message: str) -> None:
        """Record info message."""
        self.info_messages.append(message)

    def error(self, message: str) -> None:
        """Record error message."""
        self.error_messages.append(message)

    def get_all_messages(self) -> dict[str, list]:
        """Get all logged messages for testing."""
        return {"debug": self.debug_messages, "info": self.info_messages, "error": self.error_messages}

    def clear_messages(self) -> None:
        """Clear all logged messages."""
        self.debug_messages.clear()
        self.info_messages.clear()
        self.error_messages.clear()


class MockEnvironmentProvider:
    """Mock environment provider for testing."""

    def __init__(self):
        """Initialize mock environment."""
        self.environment_variables: dict[str, str] = {}
        self.locale_calls: list[tuple] = []

    def set_environment_variable(self, key: str, value: str) -> None:
        """Record environment variable setting."""
        self.environment_variables[key] = value

    def set_locale(self, category: int, locale_name: str) -> None:
        """Record locale setting."""
        self.locale_calls.append((category, locale_name))

    def get_environment_variables(self) -> dict[str, str]:
        """Get recorded environment variables."""
        return self.environment_variables.copy()

    def get_locale_calls(self) -> list[tuple]:
        """Get recorded locale calls."""
        return self.locale_calls.copy()

    def clear_records(self) -> None:
        """Clear all recorded operations."""
        self.environment_variables.clear()
        self.locale_calls.clear()


# ================================
# FACTORY FUNCTIONS
# ================================


def create_config_provider(mock_data: dict[str, Any] | None = None):
    """Create configuration provider (real or mock)."""
    if mock_data is not None:
        provider = MockConfigProvider()

        # Set up mock configurations if provided
        if "language_configs" in mock_data:
            for language, config in mock_data["language_configs"].items():
                provider.set_language_config(language, config)

        if "cleaning_config" in mock_data:
            provider.set_cleaning_config(mock_data["cleaning_config"])

        if "document_cleaning_configs" in mock_data:
            for language, config in mock_data["document_cleaning_configs"].items():
                provider.set_document_cleaning_config(language, config)

        if "chunking_configs" in mock_data:
            for language, config in mock_data["chunking_configs"].items():
                provider.set_chunking_config(language, config)

        if "shared_language_configs" in mock_data:
            for language, config in mock_data["shared_language_configs"].items():
                provider.set_shared_language_config(language, config)

        return provider

    return ConfigProvider()


def create_logger_provider(logger_name: str | None = None, mock: bool = False):
    """Create logger provider (real or mock)."""
    if mock:
        return MockLoggerProvider()
    return LoggerProvider(logger_name)


def create_environment_provider(mock: bool = False):
    """Create environment provider (real or mock)."""
    if mock:
        return MockEnvironmentProvider()
    return EnvironmentProvider()


# ================================
# CONVENIENCE PROVIDER BUILDERS
# ================================


def create_test_providers(
    language: str = "hr",
    mock_configs: dict[str, Any] | None = None,
    mock_logging: bool = True,
    mock_environment: bool = True,
):
    """
    Create full set of test providers for testing.

    Args:
        language: Target language for default configurations
        mock_configs: Custom mock configurations
        mock_logging: Whether to use mock logging
        mock_environment: Whether to use mock environment

    Returns:
        Tuple of (config_provider, logger_provider, environment_provider)
    """
    # Default test configurations for Croatian
    default_configs = {
        "language_configs": {
            language: {
                "diacritic_map": {"č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "d"},
                "word_char_pattern": r"[a-zA-ZčćšžđĆČŠŽĐ]",
                "locale": {"primary": "hr_HR.UTF-8", "fallback": "C.UTF-8"},
            }
        },
        "cleaning_config": {
            "multiple_whitespace": r"\s+",
            "multiple_linebreaks": r"\n\s*\n\s*\n+",
            "min_meaningful_words": 3,
            "min_word_char_ratio": 0.5,
        },
        "document_cleaning_configs": {
            language: {
                "header_footer_patterns": [
                    r"^\s*Stranica\s+\d+\s*$",  # Croatian page numbers
                    r"^\s*\d+\s*$",  # Simple page numbers
                ],
                "ocr_corrections": {
                    r"\bl\b": "i",  # Common OCR error
                    r"\b0\b": "o",  # Zero to o
                },
            }
        },
        "chunking_configs": {language: {"sentence_ending_pattern": r"[.!?]+\s+", "min_sentence_length": 10}},
        "shared_language_configs": {
            language: {
                "stopwords": {"words": ["i", "je", "da", "se", "na", "za", "od", "do", "u", "s"]},
                "chars_pattern": r"[^\w\s.,!?:;()-]",
            }
        },
    }

    # Merge with provided configurations
    if mock_configs:
        # Deep merge the configurations
        for key, value in mock_configs.items():
            if key in default_configs and isinstance(value, dict) and isinstance(default_configs[key], dict):
                cast(dict[str, Any], default_configs[key]).update(value)
            else:
                default_configs[key] = value

    config_provider = create_config_provider(default_configs)
    logger_provider = create_logger_provider(mock=mock_logging)
    environment_provider = create_environment_provider(mock=mock_environment)

    return config_provider, logger_provider, environment_provider


def create_providers(logger_name: str | None = None):
    """
    Create full set of providers.

    Args:
        logger_name: Logger name for logging

    Returns:
        Tuple of (config_provider, logger_provider, environment_provider)
    """
    config_provider = ConfigProvider()
    logger_provider = LoggerProvider(logger_name)
    environment_provider = EnvironmentProvider()

    return config_provider, logger_provider, environment_provider


# ================================
# SPECIALIZED TEST CONFIGURATIONS
# ================================


def create_multilingual_test_providers():
    """Create test providers with configurations for multiple languages."""
    multilingual_configs = {
        "language_configs": {
            "hr": {  # Croatian
                "diacritic_map": {"č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "d"},
                "word_char_pattern": r"[a-zA-ZčćšžđĆČŠŽĐ]",
                "locale": {"primary": "hr_HR.UTF-8", "fallback": "C.UTF-8"},
            },
            "en": {  # English
                "diacritic_map": {},
                "word_char_pattern": r"[a-zA-Z]",
                "locale": {"primary": "en_US.UTF-8", "fallback": "C.UTF-8"},
            },
            "de": {  # German
                "diacritic_map": {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"},
                "word_char_pattern": r"[a-zA-ZäöüßÄÖÜ]",
                "locale": {"primary": "de_DE.UTF-8", "fallback": "C.UTF-8"},
            },
        },
        "shared_language_configs": {
            "hr": {"stopwords": {"words": ["i", "je", "da", "se", "na", "za"]}, "chars_pattern": r"[^\w\s.,!?:;()-]"},
            "en": {
                "stopwords": {"words": ["the", "and", "or", "but", "in", "on"]},
                "chars_pattern": r"[^\w\s.,!?:;()-]",
            },
            "de": {
                "stopwords": {"words": ["der", "die", "das", "und", "oder", "aber"]},
                "chars_pattern": r"[^\w\s.,!?:;()-]",
            },
        },
    }

    return create_test_providers(mock_configs=multilingual_configs)


def create_minimal_test_providers():
    """Create test providers with minimal configurations for fast testing."""
    minimal_configs = {
        "language_configs": {"test": {"diacritic_map": {"ç": "c"}, "word_char_pattern": r"[a-zA-Z]"}},
        "cleaning_config": {
            "multiple_whitespace": r"\s+",
            "multiple_linebreaks": r"\n+",
            "min_meaningful_words": 1,
            "min_word_char_ratio": 0.1,
        },
        "document_cleaning_configs": {
            "test": {"header_footer_patterns": [r"^\s*Page\s+\d+\s*$"], "ocr_corrections": {}}
        },
        "chunking_configs": {"test": {"sentence_ending_pattern": r"[.!?]+\s+", "min_sentence_length": 1}},
        "shared_language_configs": {"test": {"stopwords": {"words": ["the", "and"]}, "chars_pattern": r"[^\w\s]"}},
    }

    return create_test_providers(language="test", mock_configs=minimal_configs)
