"""
Provider implementations for query processor dependency injection.
Production and mock providers for testable query processing system.
"""

from typing import Any, Protocol

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_error_context,
    log_performance_metric,
)
from .query_processor import QueryProcessingConfig

# ================================
# PROTOCOLS
# ================================


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""

    def load_config(self, config_name: str) -> dict[str, Any]:
        """Load configuration by name."""
        ...

    def get_language_specific_config(self, section: str, language: str) -> dict[str, Any]:
        """Get language-specific configuration."""
        ...


# ================================
# STANDARD PROVIDERS
# ================================


class LanguageDataProvider:
    """Language data provider using configuration files."""

    def __init__(self, config_provider: ConfigProvider):
        """Initialize with configuration provider."""
        logger = get_system_logger()
        log_component_start("language_data_provider", "init")

        self.config_provider = config_provider
        self._cache: dict[str, Any] = {}

        logger.debug("language_data_provider", "init", "Language data provider initialized with cache")
        log_component_end("language_data_provider", "init", "Language data provider ready")

    def get_stop_words(self, language: str) -> set[str]:
        """Get stop words for language."""
        logger = get_system_logger()
        log_component_start("language_data_provider", "get_stop_words", language=language)

        cache_key = f"stop_words_{language}"

        try:
            if cache_key not in self._cache:
                logger.debug("language_data_provider", "get_stop_words", f"Cache miss for {language} stop words")
                lang_config = self.config_provider.get_language_specific_config("language_data", language)

                if "stop_words" not in lang_config:
                    error_msg = f"Missing 'stop_words' in language configuration for {language}"
                    logger.error("language_data_provider", "get_stop_words", error_msg)
                    raise ValueError(error_msg)

                stop_words = lang_config["stop_words"]
                self._cache[cache_key] = set(stop_words)

                logger.debug(
                    "language_data_provider", "get_stop_words", f"Cached {len(stop_words)} stop words for {language}"
                )
            else:
                logger.trace("language_data_provider", "get_stop_words", f"Cache hit for {language} stop words")

            result = self._cache[cache_key]
            log_performance_metric("language_data_provider", "get_stop_words", "stop_words_count", len(result))

            log_component_end(
                "language_data_provider", "get_stop_words", f"Retrieved {len(result)} stop words for {language}"
            )
            return result

        except Exception as e:
            log_error_context(
                "language_data_provider", "get_stop_words", e, {"language": language, "cache_key": cache_key}
            )
            raise

    def get_question_patterns(self, language: str) -> list[str]:
        """Get question patterns for language."""
        logger = get_system_logger()
        log_component_start("language_data_provider", "get_question_patterns", language=language)

        cache_key = f"question_patterns_{language}"

        try:
            if cache_key not in self._cache:
                logger.debug(
                    "language_data_provider", "get_question_patterns", f"Cache miss for {language} question patterns"
                )
                lang_config = self.config_provider.get_language_specific_config("language_data", language)

                if "question_patterns" not in lang_config:
                    error_msg = f"Missing 'question_patterns' in language configuration for {language}"
                    logger.error("language_data_provider", "get_question_patterns", error_msg)
                    raise ValueError(error_msg)

                patterns = lang_config["question_patterns"]
                self._cache[cache_key] = patterns

                logger.debug(
                    "language_data_provider",
                    "get_question_patterns",
                    f"Cached {len(patterns)} question patterns for {language}",
                )
            else:
                logger.trace(
                    "language_data_provider", "get_question_patterns", f"Cache hit for {language} question patterns"
                )

            result = self._cache[cache_key]
            log_performance_metric("language_data_provider", "get_question_patterns", "patterns_count", len(result))

            log_component_end(
                "language_data_provider",
                "get_question_patterns",
                f"Retrieved {len(result)} question patterns for {language}",
            )
            return result

        except Exception as e:
            log_error_context(
                "language_data_provider", "get_question_patterns", e, {"language": language, "cache_key": cache_key}
            )
            raise

    def get_synonym_groups(self, language: str) -> dict[str, list[str]]:
        """Get synonym groups for language."""
        cache_key = f"synonym_groups_{language}"
        if cache_key not in self._cache:
            lang_config = self.config_provider.get_language_specific_config("language_data", language)
            if "synonym_groups" not in lang_config:
                raise ValueError(f"Missing 'synonym_groups' in language configuration for {language}")
            synonyms = lang_config["synonym_groups"]
            self._cache[cache_key] = synonyms

        return self._cache[cache_key]

    def get_morphological_patterns(self, language: str) -> dict[str, list[str]]:
        """Get morphological patterns for language."""
        cache_key = f"morphological_patterns_{language}"
        if cache_key not in self._cache:
            lang_config = self.config_provider.get_language_specific_config("language_data", language)
            if "morphological_patterns" not in lang_config:
                raise ValueError(f"Missing 'morphological_patterns' in language configuration for {language}")
            patterns = lang_config["morphological_patterns"]
            self._cache[cache_key] = patterns

        return self._cache[cache_key]


# ================================
# FACTORY FUNCTIONS
# ================================


def create_default_config(language: str = "hr", config_provider: ConfigProvider | None = None) -> QueryProcessingConfig:
    """
    Create default query processing configuration.

    Args:
        language: Language code for configuration
        config_provider: Optional configuration provider

    Returns:
        QueryProcessingConfig instance with default values
    """
    # Default configuration values
    default_config = {
        "query_processing": {
            "expand_synonyms": True,
            "normalize_case": True,
            "remove_stopwords": True,
            "min_query_length": 3,
            "max_query_length": 50,
            "max_expanded_terms": 10,
            "enable_morphological_analysis": False,
            "use_query_classification": True,
            "enable_spell_check": False,
        }
    }

    # Use config provider if available
    if config_provider:
        config_dict = config_provider.load_config("config")
        if not config_dict:
            raise ValueError("Failed to load configuration from config provider")
    else:
        config_dict = default_config

    return QueryProcessingConfig.from_validated_config(main_config=config_dict, language=language)


def create_language_provider(config_provider: ConfigProvider) -> LanguageDataProvider:
    """Create language data provider."""
    return LanguageDataProvider(config_provider)


def create_providers(language: str = "hr") -> tuple:
    """
    Create providers for query processing.

    Args:
        language: Language for providers

    Returns:
        Tuple of (config, language_provider, config_provider)
    """
    # Import real config provider
    from ..utils.config_protocol import get_config_provider

    config_provider = get_config_provider()

    # Create language data provider
    language_provider = create_language_provider(config_provider)

    # Create configuration
    config = create_default_config(language, config_provider)

    return config, language_provider, config_provider


def create_query_processor(language: str = "hr"):
    """Create query processor with all dependencies."""
    from .query_processor import MultilingualQueryProcessor

    config, language_provider, config_provider = create_providers(language)

    return MultilingualQueryProcessor(config=config, language_data_provider=language_provider)
