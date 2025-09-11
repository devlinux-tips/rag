"""
Provider implementations for query processor dependency injection.
Production and mock providers for testable query processing system.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, Set

from .query_processor import LanguageDataProvider, QueryProcessingConfig

# ================================
# PROTOCOLS
# ================================


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration by name."""
        ...

    def get_language_specific_config(self, section: str, language: str) -> Dict[str, Any]:
        """Get language-specific configuration."""
        ...


# ================================
# PRODUCTION PROVIDERS
# ================================


class ProductionLanguageDataProvider:
    """Production language data provider using real configuration files."""

    def __init__(self, config_provider: ConfigProvider):
        """Initialize with configuration provider."""
        self.config_provider = config_provider
        self._cache = {}

    def get_stop_words(self, language: str) -> Set[str]:
        """Get stop words for language."""
        cache_key = f"stop_words_{language}"
        if cache_key not in self._cache:
            lang_config = self.config_provider.get_language_specific_config(
                "language_data", language
            )
            if "stop_words" not in lang_config:
                raise ValueError(f"Missing 'stop_words' in language configuration for {language}")
            stop_words = lang_config["stop_words"]
            self._cache[cache_key] = set(stop_words)

        return self._cache[cache_key]

    def get_question_patterns(self, language: str) -> List[str]:
        """Get question patterns for language."""
        cache_key = f"question_patterns_{language}"
        if cache_key not in self._cache:
            lang_config = self.config_provider.get_language_specific_config(
                "language_data", language
            )
            if "question_patterns" not in lang_config:
                raise ValueError(
                    f"Missing 'question_patterns' in language configuration for {language}"
                )
            patterns = lang_config["question_patterns"]
            self._cache[cache_key] = patterns

        return self._cache[cache_key]

    def get_synonym_groups(self, language: str) -> Dict[str, List[str]]:
        """Get synonym groups for language."""
        cache_key = f"synonym_groups_{language}"
        if cache_key not in self._cache:
            lang_config = self.config_provider.get_language_specific_config(
                "language_data", language
            )
            if "synonym_groups" not in lang_config:
                raise ValueError(
                    f"Missing 'synonym_groups' in language configuration for {language}"
                )
            synonyms = lang_config["synonym_groups"]
            self._cache[cache_key] = synonyms

        return self._cache[cache_key]

    def get_morphological_patterns(self, language: str) -> Dict[str, List[str]]:
        """Get morphological patterns for language."""
        cache_key = f"morphological_patterns_{language}"
        if cache_key not in self._cache:
            lang_config = self.config_provider.get_language_specific_config(
                "language_data", language
            )
            if "morphological_patterns" not in lang_config:
                raise ValueError(
                    f"Missing 'morphological_patterns' in language configuration for {language}"
                )
            patterns = lang_config["morphological_patterns"]
            self._cache[cache_key] = patterns

        return self._cache[cache_key]


# ================================
# MOCK PROVIDERS
# ================================


class MockLanguageDataProvider:
    """Mock language data provider for testing."""

    def __init__(self):
        """Initialize with empty mock data."""
        self.stop_words = {}
        self.question_patterns = {}
        self.synonym_groups = {}
        self.morphological_patterns = {}

    def set_stop_words(self, language: str, stop_words: Set[str]) -> None:
        """Set mock stop words for language."""
        self.stop_words[language] = stop_words

    def set_question_patterns(self, language: str, patterns: List[str]) -> None:
        """Set mock question patterns for language."""
        self.question_patterns[language] = patterns

    def set_synonym_groups(self, language: str, synonyms: Dict[str, List[str]]) -> None:
        """Set mock synonym groups for language."""
        self.synonym_groups[language] = synonyms

    def set_morphological_patterns(self, language: str, patterns: Dict[str, List[str]]) -> None:
        """Set mock morphological patterns for language."""
        self.morphological_patterns[language] = patterns

    def get_stop_words(self, language: str) -> Set[str]:
        """Get mock stop words for language."""
        return self.stop_words.get(language, set())

    def get_question_patterns(self, language: str) -> List[str]:
        """Get mock question patterns for language."""
        return self.question_patterns.get(language, [])

    def get_synonym_groups(self, language: str) -> Dict[str, List[str]]:
        """Get mock synonym groups for language."""
        return self.synonym_groups.get(language, {})

    def get_morphological_patterns(self, language: str) -> Dict[str, List[str]]:
        """Get mock morphological patterns for language."""
        return self.morphological_patterns.get(language, {})


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self):
        """Initialize with empty mock configuration."""
        self.configs = {}
        self.language_configs = {}

    def set_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """Set mock configuration."""
        self.configs[config_name] = config

    def set_language_config(self, section: str, language: str, config: Dict[str, Any]) -> None:
        """Set mock language-specific configuration."""
        key = f"{section}_{language}"
        self.language_configs[key] = config

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load mock configuration by name."""
        return self.configs.get(config_name, {})

    def get_language_specific_config(self, section: str, language: str) -> Dict[str, Any]:
        """Get mock language-specific configuration."""
        key = f"{section}_{language}"
        return self.language_configs.get(key, {})


# ================================
# FACTORY FUNCTIONS
# ================================


def create_default_config(
    language: str = "hr", config_provider: Optional[ConfigProvider] = None
) -> QueryProcessingConfig:
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
            "max_expanded_terms": 10,
            "enable_spell_check": False,
            "min_word_length": 2,
        }
    }

    # Use config provider if available
    if config_provider:
        config_dict = config_provider.load_config("config")
        if not config_dict:
            raise ValueError("Failed to load configuration from config provider")
    else:
        config_dict = default_config

    return QueryProcessingConfig.from_config(
        config_dict=config_dict, config_provider=config_provider, language=language
    )


def create_production_language_provider(
    config_provider: ConfigProvider,
) -> ProductionLanguageDataProvider:
    """Create production language data provider."""
    return ProductionLanguageDataProvider(config_provider)


def create_mock_language_provider(
    language: str = "hr", custom_data: Optional[Dict[str, Any]] = None
) -> MockLanguageDataProvider:
    """
    Create mock language data provider with optional custom data.

    Args:
        language: Language for default data
        custom_data: Optional custom language data

    Returns:
        MockLanguageDataProvider instance
    """
    provider = MockLanguageDataProvider()

    # Set default data for language
    if language == "hr":
        provider.set_stop_words(
            "hr", {"i", "a", "u", "na", "za", "od", "do", "iz", "s", "sa", "se"}
        )
        provider.set_question_patterns(
            "hr", [r"^što\s", r"^kako\s", r"^kada\s", r"^gdje\s", r"^zašto\s"]
        )
        provider.set_synonym_groups(
            "hr",
            {
                "brz": ["brži", "brzo", "hitno"],
                "dobro": ["odlično", "izvrsno", "super"],
            },
        )
    else:  # English
        provider.set_stop_words(
            "en", {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"}
        )
        provider.set_question_patterns(
            "en", [r"^what\s", r"^how\s", r"^when\s", r"^where\s", r"^why\s"]
        )
        provider.set_synonym_groups(
            "en",
            {
                "fast": ["quick", "rapid", "swift"],
                "good": ["great", "excellent", "awesome"],
            },
        )

    # Apply custom data if provided
    if custom_data:
        for key, value in custom_data.items():
            if key == "stop_words":
                provider.set_stop_words(language, set(value))
            elif key == "question_patterns":
                provider.set_question_patterns(language, value)
            elif key == "synonym_groups":
                provider.set_synonym_groups(language, value)
            elif key == "morphological_patterns":
                provider.set_morphological_patterns(language, value)

    return provider


def create_test_providers(
    language: str = "hr",
    custom_config: Optional[Dict[str, Any]] = None,
    custom_language_data: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Create complete set of test providers for query processing.

    Args:
        language: Language for providers
        custom_config: Optional custom configuration
        custom_language_data: Optional custom language data

    Returns:
        Tuple of (config, language_provider, config_provider)
    """
    # Create mock config provider
    config_provider = MockConfigProvider()

    # Set default or custom config
    if custom_config:
        config_provider.set_config("config", custom_config)
    else:
        config_provider.set_config(
            "config",
            {
                "query_processing": {
                    "expand_synonyms": True,
                    "normalize_case": True,
                    "remove_stopwords": True,
                    "min_query_length": 3,
                    "max_expanded_terms": 10,
                    "enable_spell_check": False,
                    "min_word_length": 2,
                }
            },
        )

    # Create language data provider
    language_provider = create_mock_language_provider(language, custom_language_data)

    # Create configuration
    config = create_default_config(language, config_provider)

    return config, language_provider, config_provider


def create_production_providers(language: str = "hr") -> tuple:
    """
    Create production providers for query processing.

    Args:
        language: Language for providers

    Returns:
        Tuple of (config, language_provider, config_provider)
    """
    # Try to import real config provider
    try:
        from ..utils.config_loader import get_config_provider

        config_provider = get_config_provider()
    except ImportError:
        # Fallback to mock for development
        config_provider = MockConfigProvider()
        config_provider.set_config(
            "config",
            {
                "query_processing": {
                    "expand_synonyms": True,
                    "normalize_case": True,
                    "remove_stopwords": True,
                    "min_query_length": 3,
                    "max_expanded_terms": 10,
                    "enable_spell_check": False,
                    "min_word_length": 2,
                }
            },
        )

    # Create language data provider
    language_provider = create_production_language_provider(config_provider)

    # Create configuration
    config = create_default_config(language, config_provider)

    return config, language_provider, config_provider
