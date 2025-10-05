"""
Provider implementations for language manager dependency injection.
Production and mock providers for testable language management system.
"""

import logging

from .config_validator import ConfigurationError
from .language_manager import LanguagePatterns, LanguageSettings
from .logging_factory import get_system_logger, log_component_end, log_component_start, log_data_transformation

# ================================
# PRODUCTION PROVIDERS
# ================================


class DefaultConfigProvider:
    """Production configuration provider using real config system."""

    def __init__(self):
        """Initialize production config provider."""
        self._settings_cache: LanguageSettings | None = None

    def get_language_settings(self) -> LanguageSettings:
        """Get language settings from real config system."""
        get_system_logger()
        log_component_start(
            "language_manager_providers", "get_language_settings", cache_available=self._settings_cache is not None
        )

        if self._settings_cache is None:
            self._settings_cache = self._load_settings_from_system()

        log_component_end(
            "language_manager_providers",
            "get_language_settings",
            "Language settings retrieved",
            supported_languages=len(self._settings_cache.supported_languages),
            default_language=self._settings_cache.default_language,
        )
        return self._settings_cache

    def _load_settings_from_system(self) -> LanguageSettings:
        """Load settings from the real configuration system."""
        system_logger = get_system_logger()
        log_component_start("language_manager_providers", "_load_settings_from_system")

        try:
            # Import at runtime to avoid circular dependencies
            from .config_loader import get_shared_config, get_supported_languages, load_config

            # Load configurations
            supported_langs = get_supported_languages()
            main_config = load_config("config")
            shared_config = get_shared_config()

            # Extract language settings
            languages_config = main_config["languages"]
            embeddings_config = main_config["embeddings"]

            log_data_transformation(
                "language_manager_providers",
                "settings_loading",
                f"Input: config files (main, shared, {len(supported_langs)} languages)",
                f"Output: LanguageSettings with {len(supported_langs)} supported languages",
                default_language=languages_config["default"],
                auto_detect=languages_config["auto_detect"],
                embedding_model=embeddings_config["model_name"],
            )

            settings = LanguageSettings(
                supported_languages=supported_langs,
                default_language=languages_config["default"],
                auto_detect=languages_config["auto_detect"],
                fallback_language=languages_config["default"],
                language_names=languages_config["names"],
                embedding_model=embeddings_config["model_name"],
                chunk_size=shared_config["default_chunk_size"],
                chunk_overlap=shared_config["default_chunk_overlap"],
            )

            log_component_end(
                "language_manager_providers",
                "_load_settings_from_system",
                "Successfully loaded language settings",
                languages_count=len(settings.supported_languages),
                default_language=settings.default_language,
            )

            return settings
        except Exception as e:
            system_logger.error(
                "language_manager_providers",
                "_load_settings_from_system",
                "FAILED: Configuration loading error",
                error_type=type(e).__name__,
                stack_trace=str(e),
            )
            # FAIL FAST: No fallbacks - configuration must be valid
            raise ConfigurationError(
                f"Failed to load language settings: {e}. Please check your configuration files."
            ) from e


class DefaultPatternProvider:
    """Production pattern provider using real config system."""

    def __init__(self):
        """Initialize production pattern provider."""
        self._patterns_cache: LanguagePatterns | None = None

    def get_language_patterns(self) -> LanguagePatterns:
        """Get language patterns from real config system."""
        get_system_logger()
        log_component_start(
            "language_manager_providers", "get_language_patterns", cache_available=self._patterns_cache is not None
        )

        if self._patterns_cache is None:
            self._patterns_cache = self._load_patterns_from_system()

        log_component_end(
            "language_manager_providers",
            "get_language_patterns",
            "Language patterns retrieved",
            detection_patterns_count=len(self._patterns_cache.detection_patterns),
            stopwords_languages=len(self._patterns_cache.stopwords),
        )
        return self._patterns_cache

    def _load_patterns_from_system(self) -> LanguagePatterns:
        """Load patterns from the real configuration system."""
        system_logger = get_system_logger()
        log_component_start("language_manager_providers", "_load_patterns_from_system")

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
                    if "shared" not in lang_config or "question_patterns" not in lang_config["shared"]:
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

                    log_data_transformation(
                        "language_manager_providers",
                        "pattern_loading",
                        f"Input: {lang_code} config file",
                        f"Output: {len(detection_patterns.get(lang_code, []))} detection patterns, {len(stopwords.get(lang_code, set()))} stopwords",
                        language=lang_code,
                        detection_patterns_count=len(detection_patterns.get(lang_code, [])),
                        stopwords_count=len(stopwords.get(lang_code, set())),
                    )

                except Exception as e:
                    system_logger.error(
                        "language_manager_providers",
                        "_load_patterns_from_system",
                        f"FAILED: Pattern loading for language {lang_code}",
                        error_type=type(e).__name__,
                        stack_trace=str(e),
                        metadata={"language": lang_code},
                    )
                    raise ConfigurationError(
                        f"Failed to load patterns for language '{lang_code}': {e}. "
                        f"Please check configuration for {lang_code}"
                    ) from e

            patterns = LanguagePatterns(detection_patterns=detection_patterns, stopwords=stopwords)

            log_component_end(
                "language_manager_providers",
                "_load_patterns_from_system",
                "Successfully loaded language patterns",
                languages_processed=len(supported_langs),
                detection_languages=len(detection_patterns),
                stopwords_languages=len(stopwords),
            )

            return patterns

        except Exception as e:
            system_logger.error(
                "language_manager_providers",
                "_load_patterns_from_system",
                "FAILED: Pattern system loading error",
                error_type=type(e).__name__,
                stack_trace=str(e),
            )
            # FAIL FAST: No fallbacks - configuration must be valid
            raise ConfigurationError(
                f"Failed to load language patterns: {e}. Please check your configuration files."
            ) from e


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


def create_default_setup(logger_name: str | None = None) -> tuple:
    """
    Create production setup with real components.

    Args:
        logger_name: Optional logger name override

    Returns:
        Tuple of (config_provider, pattern_provider, logger_provider)
    """
    config_provider = DefaultConfigProvider()
    pattern_provider = DefaultPatternProvider()
    logger_provider = StandardLoggerProvider(logger_name or __name__)

    return config_provider, pattern_provider, logger_provider


# Backward compatibility aliases
ProductionConfigProvider = DefaultConfigProvider
ProductionPatternProvider = DefaultPatternProvider
create_production_setup = create_default_setup


def create_test_settings(
    supported_languages: list[str] | None = None,
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
    detection_patterns: dict[str, list[str]] | None = None, stopwords: dict[str, set[str]] | None = None
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
        detection_patterns=detection_patterns or default_detection, stopwords=stopwords or default_stopwords
    )


# ================================
# INTEGRATION HELPERS
# ================================


def create_development_language_manager():
    """Create language manager configured for development/testing."""
    from .language_manager import create_language_manager

    config_provider, pattern_provider, logger_provider = create_production_setup()
    return create_language_manager(
        config_provider=config_provider, pattern_provider=pattern_provider, logger_provider=logger_provider
    )
