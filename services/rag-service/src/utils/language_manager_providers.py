"""
Provider implementations for language manager dependency injection.
Production and mock providers for testable language management system.
"""

import logging
from typing import Dict, List, Optional, Set

from .config_validator import ConfigurationError
from .language_manager import (
    ConfigProvider,
    LanguagePatterns,
    LanguageSettings,
    LoggerProvider,
    PatternProvider,
)

# ================================
# MOCK PROVIDERS FOR TESTING
# ================================


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self, settings: LanguageSettings | None = None):
        """Initialize with optional mock settings."""
        self.settings = settings or self._create_default_settings()
        self.call_history: list[str] = []

    def _create_default_settings(self) -> LanguageSettings:
        """Create default test settings."""
        return LanguageSettings(
            supported_languages=["hr", "en", "multilingual"],
            default_language="hr",
            auto_detect=True,
            fallback_language="hr",
            language_names={
                "hr": "Croatian",
                "en": "English",
                "multilingual": "Multilingual",
            },
            embedding_model="BAAI/bge-m3",
            chunk_size=512,
            chunk_overlap=50,
        )

    def set_settings(self, settings: LanguageSettings) -> None:
        """Set mock settings."""
        self.settings = settings

    def get_language_settings(self) -> LanguageSettings:
        """Get language settings configuration."""
        self.call_history.append("get_language_settings")
        return self.settings


class MockPatternProvider:
    """Mock pattern provider for testing."""

    def __init__(self, patterns: LanguagePatterns | None = None):
        """Initialize with optional mock patterns."""
        self.patterns = patterns or self._create_default_patterns()
        self.call_history: list[str] = []

    def _create_default_patterns(self) -> LanguagePatterns:
        """Create default test patterns."""
        return LanguagePatterns(
            detection_patterns={
                "hr": ["što", "kako", "gdje", "kada", "zašto", "koji", "koja", "koje"],
                "en": ["what", "how", "where", "when", "why", "which", "who", "that"],
                "multilingual": ["i", "in", "of", "to", "and", "the", "is", "for"],
            },
            stopwords={
                "hr": {"i", "u", "na", "za", "je", "se", "da", "od", "do", "sa"},
                "en": {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from"},
                "multilingual": {
                    "the",
                    "of",
                    "and",
                    "to",
                    "a",
                    "in",
                    "is",
                    "it",
                    "you",
                    "that",
                },
            },
        )

    def set_patterns(self, patterns: LanguagePatterns) -> None:
        """Set mock patterns."""
        self.patterns = patterns

    def add_detection_pattern(self, language_code: str, patterns: list[str]) -> None:
        """Add detection patterns for language."""
        self.patterns.detection_patterns[language_code] = patterns

    def add_stopwords(self, language_code: str, stopwords: set[str]) -> None:
        """Add stopwords for language."""
        self.patterns.stopwords[language_code] = stopwords

    def get_language_patterns(self) -> LanguagePatterns:
        """Get language detection patterns and stopwords."""
        self.call_history.append("get_language_patterns")
        return self.patterns


class MockLoggerProvider:
    """Mock logger provider that captures messages for testing."""

    def __init__(self):
        """Initialize message capture."""
        self.messages: dict[str, list[str]] = {
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

    def get_messages(self, level: str = None) -> dict[str, list[str]] | list[str]:
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
        self._settings_cache: LanguageSettings | None = None

    def get_language_settings(self) -> LanguageSettings:
        """Get language settings from real config system."""
        if self._settings_cache is None:
            self._settings_cache = self._load_settings_from_system()
        return self._settings_cache

    def _load_settings_from_system(self) -> LanguageSettings:
        """Load settings from the real configuration system."""
        try:
            # Import at runtime to avoid circular dependencies
            from .config_loader import (
                get_shared_config,
                get_supported_languages,
                load_config,
            )

            # Load configurations
            supported_langs = get_supported_languages()
            main_config = load_config("config")
            shared_config = get_shared_config()

            # Extract language settings
            languages_config = main_config["languages"]
            embeddings_config = main_config["embeddings"]

            return LanguageSettings(
                supported_languages=supported_langs,
                default_language=languages_config["default"],
                auto_detect=languages_config["auto_detect"],
                fallback_language=languages_config["default"],
                language_names=languages_config["names"],
                embedding_model=embeddings_config["model_name"],
                chunk_size=shared_config["default_chunk_size"],
                chunk_overlap=shared_config["default_chunk_overlap"],
            )
        except Exception as e:
            # FAIL FAST: No fallbacks - configuration must be valid
            raise ConfigurationError(
                f"Failed to load language settings: {e}. "
                f"Please check your configuration files."
            )


class ProductionPatternProvider:
    """Production pattern provider using real config system."""

    def __init__(self):
        """Initialize production pattern provider."""
        self._patterns_cache: LanguagePatterns | None = None

    def get_language_patterns(self) -> LanguagePatterns:
        """Get language patterns from real config system."""
        if self._patterns_cache is None:
            self._patterns_cache = self._load_patterns_from_system()
        return self._patterns_cache

    def _load_patterns_from_system(self) -> LanguagePatterns:
        """Load patterns from the real configuration system."""
        try:
            # Import at runtime to avoid circular dependencies
            from .config_loader import get_language_config, get_supported_languages

            supported_langs = get_supported_languages()
            detection_patterns = {}
            stopwords = {}

            # Load patterns for each supported language
            for lang_code in supported_langs:
                try:
                    # Load language-specific config (e.g., croatian.toml, english.toml)
                    lang_config = get_language_config(lang_code)

                    # Extract question patterns for detection
                    if (
                        "shared" not in lang_config
                        or "question_patterns" not in lang_config["shared"]
                    ):
                        continue  # Skip languages without question patterns
                    question_patterns = lang_config["shared"]["question_patterns"]
                    if question_patterns:
                        # Combine different pattern types for detection
                        detection_words = []
                        for pattern_list in question_patterns.values():
                            if isinstance(pattern_list, list):
                                detection_words.extend(pattern_list)

                        if detection_words:
                            # Use first 8 words for detection (limit pattern size)
                            detection_patterns[lang_code] = detection_words[:8]

                    # Extract stopwords
                    if (
                        "shared" in lang_config
                        and "stopwords" in lang_config["shared"]
                        and "words" in lang_config["shared"]["stopwords"]
                    ):
                        stopwords_list = lang_config["shared"]["stopwords"]["words"]
                    else:
                        stopwords_list = []
                    if stopwords_list:
                        # Use first 20 stopwords for efficiency
                        stopwords[lang_code] = set(stopwords_list[:20])

                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to load patterns for language '{lang_code}': {e}. "
                        f"Please check configuration for {lang_code}"
                    )

            return LanguagePatterns(
                detection_patterns=detection_patterns, stopwords=stopwords
            )

        except Exception as e:
            # FAIL FAST: No fallbacks - configuration must be valid
            raise ConfigurationError(
                f"Failed to load language patterns: {e}. "
                f"Please check your configuration files."
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
    settings: LanguageSettings | None = None,
    patterns: LanguagePatterns | None = None,
    custom_patterns: dict[str, list[str]] | None = None,
    custom_stopwords: dict[str, set[str]] | None = None,
) -> tuple:
    """
    Create complete mock setup for testing.

    Args:
        settings: Optional mock language settings
        patterns: Optional mock language patterns
        custom_patterns: Optional custom detection patterns per language
        custom_stopwords: Optional custom stopwords per language

    Returns:
        Tuple of (config_provider, pattern_provider, logger_provider)
    """
    # Create mock components
    config_provider = MockConfigProvider(settings)
    pattern_provider = MockPatternProvider(patterns)
    logger_provider = MockLoggerProvider()

    # Add custom patterns if provided
    if custom_patterns:
        for lang_code, patterns_list in custom_patterns.items():
            pattern_provider.add_detection_pattern(lang_code, patterns_list)

    # Add custom stopwords if provided
    if custom_stopwords:
        for lang_code, stopwords_set in custom_stopwords.items():
            pattern_provider.add_stopwords(lang_code, stopwords_set)

    return config_provider, pattern_provider, logger_provider


def create_production_setup(logger_name: str | None = None) -> tuple:
    """
    Create production setup with real components.

    Args:
        logger_name: Optional logger name override

    Returns:
        Tuple of (config_provider, pattern_provider, logger_provider)
    """
    config_provider = ProductionConfigProvider()
    pattern_provider = ProductionPatternProvider()
    logger_provider = StandardLoggerProvider(logger_name or __name__)

    return config_provider, pattern_provider, logger_provider


def create_test_settings(
    supported_languages: list[str] = None,
    default_language: str = "hr",
    auto_detect: bool = True,
    embedding_model: str = "BAAI/bge-m3",
) -> LanguageSettings:
    """Create test settings with customizable parameters."""
    if supported_languages is None:
        supported_languages = ["hr", "en", "multilingual"]

    language_names = {}
    for lang in supported_languages:
        if lang == "hr":
            language_names[lang] = "Croatian"
        elif lang == "en":
            language_names[lang] = "English"
        elif lang == "multilingual":
            language_names[lang] = "Multilingual"
        else:
            language_names[lang] = lang.upper()

    return LanguageSettings(
        supported_languages=supported_languages,
        default_language=default_language,
        auto_detect=auto_detect,
        fallback_language=default_language,
        language_names=language_names,
        embedding_model=embedding_model,
        chunk_size=512,
        chunk_overlap=50,
    )


def create_test_patterns(
    detection_patterns: dict[str, list[str]] | None = None,
    stopwords: dict[str, set[str]] | None = None,
) -> LanguagePatterns:
    """Create test patterns with customizable parameters."""
    default_detection = {
        "hr": ["što", "kako", "gdje", "kada", "zašto"],
        "en": ["what", "how", "where", "when", "why"],
        "multilingual": ["what", "kako", "where", "gdje"],
    }

    default_stopwords = {
        "hr": {"i", "u", "na", "za", "je"},
        "en": {"a", "and", "the", "of", "to"},
        "multilingual": {"i", "and", "u", "the"},
    }

    return LanguagePatterns(
        detection_patterns=detection_patterns or default_detection,
        stopwords=stopwords or default_stopwords,
    )


# ================================
# INTEGRATION HELPERS
# ================================


def create_development_language_manager():
    """Create language manager configured for development/testing."""
    from .language_manager import create_language_manager

    config_provider, pattern_provider, logger_provider = create_production_setup()
    return create_language_manager(
        config_provider=config_provider,
        pattern_provider=pattern_provider,
        logger_provider=logger_provider,
    )


def create_test_language_manager(
    settings: LanguageSettings | None = None,
    patterns: LanguagePatterns | None = None,
    custom_patterns: dict[str, list[str]] | None = None,
    custom_stopwords: dict[str, set[str]] | None = None,
):
    """Create language manager configured for testing."""
    from .language_manager import create_language_manager

    config_provider, pattern_provider, logger_provider = create_mock_setup(
        settings=settings,
        patterns=patterns,
        custom_patterns=custom_patterns,
        custom_stopwords=custom_stopwords,
    )

    return create_language_manager(
        config_provider=config_provider,
        pattern_provider=pattern_provider,
        logger_provider=logger_provider,
    ), (config_provider, pattern_provider, logger_provider)
