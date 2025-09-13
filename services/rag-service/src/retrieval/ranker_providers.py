"""
Provider implementations for document ranker dependencies.
Includes production implementations and mock providers for testing.
"""

import logging
from typing import Any, Dict, List, Set

from .ranker import ConfigProvider, LanguageFeatures, LanguageProvider

logger = logging.getLogger(__name__)


# ===== MOCK PROVIDERS FOR TESTING =====


class MockConfigProvider(ConfigProvider):
    """Mock configuration provider for testing."""

    def __init__(self, config_dict: dict[str, Any]):
        """
        Initialize with test configuration.

        Args:
            config_dict: Configuration dictionary for testing
        """
        self.config_dict = config_dict
        self.language_configs = {
            "hr": {
                "morphology": {
                    "important_words": [
                        "zagreb",
                        "hrvatska",
                        "dubrovnik",
                        "važan",
                        "značajan",
                    ],
                    "quality_positive": [
                        "detaljno",
                        "sveobuhvatno",
                        "temeljito",
                        "precizno",
                    ],
                    "quality_negative": ["možda", "vjerojatno", "nejasno", "približno"],
                    "cultural_patterns": [
                        "biser jadrana",
                        "perla jadrana",
                        "adriatic",
                        "unesco",
                    ],
                    "grammar_patterns": [
                        "\\w+ić\\b",
                        "\\w+ović\\b",
                        "\\w+ski\\b",
                        "\\w+nja\\b",
                    ],
                }
            },
            "en": {
                "morphology": {
                    "important_words": [
                        "important",
                        "significant",
                        "major",
                        "primary",
                        "essential",
                    ],
                    "quality_positive": [
                        "detailed",
                        "comprehensive",
                        "thorough",
                        "precise",
                    ],
                    "quality_negative": [
                        "maybe",
                        "probably",
                        "unclear",
                        "approximately",
                    ],
                    "cultural_patterns": ["United States", "UK", "Britain", "American"],
                    "grammar_patterns": [
                        "\\w+ing\\b",
                        "\\w+ly\\b",
                        "\\w+tion\\b",
                        "\\w+ness\\b",
                    ],
                }
            },
        }

    def get_ranking_config(self) -> dict[str, Any]:
        """Get ranking configuration for testing."""
        return self.config_dict.get(
            "ranking",
            {
                "method": "language_enhanced",
                "enable_diversity": True,
                "diversity_threshold": 0.8,
                "boost_recent": False,
                "boost_authoritative": True,
                "content_length_factor": True,
                "keyword_density_factor": True,
                "language_specific_boost": True,
            },
        )

    def get_language_specific_config(
        self, section: str, language: str
    ) -> dict[str, Any]:
        """Get language-specific configuration for testing."""
        return self.language_configs.get(language, {})


class MockLanguageProvider(LanguageProvider):
    """Mock language provider for testing."""

    def __init__(self):
        """Initialize mock language provider."""
        self.language_features_cache = {}

    def get_language_features(self, language: str) -> LanguageFeatures:
        """Get language features for testing."""
        if language in self.language_features_cache:
            return self.language_features_cache[language]

        if language == "hr":
            features = LanguageFeatures(
                importance_words={
                    "zagreb",
                    "hrvatska",
                    "dubrovnik",
                    "split",
                    "rijeka",
                    "osijek",
                    "glavni",
                    "važan",
                    "značajan",
                    "poznati",
                    "tradicionalni",
                    "historijski",
                    "kulturni",
                    "turistički",
                    "nacionalni",
                },
                quality_indicators={
                    "positive": [
                        r"\b(detaljno|sveobuhvatno|temeljito|precizno)\b",
                        r"\b(službeno|autoritetno|provjereno|pouzdano)\b",
                        r"\b(suvremeno|aktualno|novo|nedavno)\b",
                    ],
                    "negative": [
                        r"\b(možda|vjerojatno|nejasno|približno)\b",
                        r"\b(staro|zastarjelo|neprovjereno|sumnjivo)\b",
                        r"\b(kratko|površno|nepotpuno|fragmentarno)\b",
                    ],
                },
                cultural_patterns=[
                    r"\b(biser jadrana|perla jadrana)\b",
                    r"\b(hrvatski?\w* kralj|hrvatska povijest)\b",
                    r"\b(adriatic|jadransko more)\b",
                    r"\b(unesco|svjetska baština)\b",
                ],
                grammar_patterns=[
                    r"\b\w+ić\b",
                    r"\b\w+ović\b",
                    r"\b\w+ski\b",
                    r"\b\w+nja\b",
                ],
                type_weights={
                    "encyclopedia": 1.2,
                    "academic": 1.1,
                    "news": 1.0,
                    "blog": 0.9,
                    "forum": 0.8,
                    "social": 0.7,
                },
            )

        elif language == "en":
            features = LanguageFeatures(
                importance_words={
                    "important",
                    "significant",
                    "major",
                    "primary",
                    "essential",
                    "key",
                    "main",
                    "crucial",
                    "critical",
                    "fundamental",
                    "notable",
                    "prominent",
                    "leading",
                    "advanced",
                    "innovative",
                },
                quality_indicators={
                    "positive": [
                        r"\b(detailed|comprehensive|thorough|precise)\b",
                        r"\b(official|authoritative|verified|reliable)\b",
                        r"\b(current|recent|new|updated)\b",
                    ],
                    "negative": [
                        r"\b(maybe|probably|unclear|approximately)\b",
                        r"\b(old|outdated|unverified|questionable)\b",
                        r"\b(brief|superficial|incomplete|fragmentary)\b",
                    ],
                },
                cultural_patterns=[
                    r"\b(United States|UK|Britain|England|American|British)\b",
                    r"\b(technology|science|research|development|innovation)\b",
                ],
                grammar_patterns=[
                    r"\b\w+ing\b",
                    r"\b\w+ly\b",
                    r"\b\w+tion\b",
                    r"\b\w+ness\b",
                ],
                type_weights={
                    "encyclopedia": 1.2,
                    "academic": 1.1,
                    "news": 1.0,
                    "blog": 0.9,
                    "forum": 0.8,
                    "social": 0.7,
                },
            )

        else:
            # Default/fallback features
            features = LanguageFeatures(
                importance_words=set(),
                quality_indicators={"positive": [], "negative": []},
                cultural_patterns=[],
                grammar_patterns=[],
                type_weights={"default": 1.0},
            )

        self.language_features_cache[language] = features
        return features

    def detect_language_content(self, text: str) -> dict[str, Any]:
        """Mock language detection for testing."""
        # Simple mock detection based on character patterns
        if any(char in text.lower() for char in "čćšžđ"):
            return {"language": "hr", "confidence": 0.9}
        else:
            return {"language": "en", "confidence": 0.8}


# ===== PRODUCTION PROVIDERS =====


class ProductionConfigProvider(ConfigProvider):
    """Production configuration provider using config_loader."""

    def __init__(self):
        """Initialize production config provider."""
        # Import here to avoid circular imports
        from ..utils.config_loader import (
            get_language_specific_config,
            get_ranking_config,
        )

        self._get_ranking_config = get_ranking_config
        self._get_language_specific_config = get_language_specific_config
        self.logger = logging.getLogger(__name__)

    def get_ranking_config(self) -> dict[str, Any]:
        """Get ranking configuration from config files."""
        config = self._get_ranking_config()
        if not config:
            raise ValueError("Missing ranking configuration in config files")
        return config

    def get_language_specific_config(
        self, section: str, language: str
    ) -> dict[str, Any]:
        """Get language-specific configuration from config files."""
        config = self._get_language_specific_config(section, language)
        if not config:
            raise ValueError(
                f"Missing {section} configuration for language '{language}'"
            )
        return config


class ProductionLanguageProvider(LanguageProvider):
    """Production language provider using config system."""

    def __init__(self, config_provider: ConfigProvider):
        """
        Initialize with config provider.

        Args:
            config_provider: Configuration provider for language-specific settings
        """
        self.config_provider = config_provider
        self.logger = logging.getLogger(__name__)
        self.features_cache = {}

    def get_language_features(self, language: str) -> LanguageFeatures:
        """Get language features from configuration."""
        if language in self.features_cache:
            return self.features_cache[language]

        try:
            # Get language-specific configuration
            language_config = self.config_provider.get_language_specific_config(
                "retrieval", language
            )

            if "morphology" not in language_config:
                raise ValueError(
                    f"Missing 'morphology' section in language config for '{language}'"
                )
            morphology = language_config["morphology"]

            # Build language features from configuration
            features = self._build_language_features(language, morphology)

            self.features_cache[language] = features
            return features

        except Exception as e:
            self.logger.error(f"Failed to load language features for {language}: {e}")
            raise

    def _build_language_features(
        self, language: str, morphology: dict[str, Any]
    ) -> LanguageFeatures:
        """Build language features from configuration."""

        # Importance words from morphology config
        importance_words = set()
        for word_group in morphology.values():
            if isinstance(word_group, list):
                importance_words.update(word_group)

        # Add language-specific default words if config is empty
        if not importance_words:
            importance_words = self._get_default_importance_words(language)

        # Quality indicators from configuration
        quality_indicators = {
            "positive": (
                morphology["quality_positive"]
                if "quality_positive" in morphology
                else self._get_default_quality_positive(language)
            ),
            "negative": (
                morphology["quality_negative"]
                if "quality_negative" in morphology
                else self._get_default_quality_negative(language)
            ),
        }

        # Cultural patterns from configuration
        cultural_patterns = (
            morphology["cultural_patterns"]
            if "cultural_patterns" in morphology
            else self._get_default_cultural_patterns(language)
        )

        # Grammar patterns from configuration
        grammar_patterns = (
            morphology["grammar_patterns"]
            if "grammar_patterns" in morphology
            else self._get_default_grammar_patterns(language)
        )

        # Type weights
        type_weights = {
            "encyclopedia": 1.2,
            "academic": 1.1,
            "news": 1.0,
            "blog": 0.9,
            "forum": 0.8,
            "social": 0.7,
        }

        return LanguageFeatures(
            importance_words=importance_words,
            quality_indicators=quality_indicators,
            cultural_patterns=cultural_patterns,
            grammar_patterns=grammar_patterns,
            type_weights=type_weights,
        )

    def _get_default_importance_words(self, language: str) -> set[str]:
        """Get default importance words for language."""
        defaults = {
            "hr": {
                "zagreb",
                "hrvatska",
                "dubrovnik",
                "split",
                "rijeka",
                "osijek",
                "glavni",
                "važan",
                "značajan",
                "poznati",
                "tradicionalni",
                "historijski",
                "kulturni",
                "turistički",
                "nacionalni",
            },
            "en": {
                "important",
                "significant",
                "major",
                "primary",
                "essential",
                "key",
                "main",
                "crucial",
                "critical",
                "fundamental",
                "notable",
                "prominent",
                "leading",
                "advanced",
                "innovative",
            },
        }
        return defaults.get(language, set())

    def _get_default_quality_positive(self, language: str) -> list[str]:
        """Get default positive quality indicators."""
        defaults = {
            "hr": [
                r"\b(detaljno|sveobuhvatno|temeljito|precizno)\b",
                r"\b(službeno|autoritetno|provjereno|pouzdano)\b",
                r"\b(suvremeno|aktualno|novo|nedavno)\b",
            ],
            "en": [
                r"\b(detailed|comprehensive|thorough|precise)\b",
                r"\b(official|authoritative|verified|reliable)\b",
                r"\b(current|recent|new|updated)\b",
            ],
        }
        return defaults.get(language, [])

    def _get_default_quality_negative(self, language: str) -> list[str]:
        """Get default negative quality indicators."""
        defaults = {
            "hr": [
                r"\b(možda|vjerojatno|nejasno|približno)\b",
                r"\b(staro|zastarjelo|neprovjereno|sumnjivo)\b",
                r"\b(kratko|površno|nepotpuno|fragmentarno)\b",
            ],
            "en": [
                r"\b(maybe|probably|unclear|approximately)\b",
                r"\b(old|outdated|unverified|questionable)\b",
                r"\b(brief|superficial|incomplete|fragmentary)\b",
            ],
        }
        return defaults.get(language, [])

    def _get_default_cultural_patterns(self, language: str) -> list[str]:
        """Get default cultural patterns."""
        defaults = {
            "hr": [
                r"\b(biser jadrana|perla jadrana)\b",
                r"\b(hrvatski?\w* kralj|hrvatska povijest)\b",
                r"\b(adriatic|jadransko more)\b",
                r"\b(unesco|svjetska baština)\b",
            ],
            "en": [
                r"\b(United States|UK|Britain|England|American|British)\b",
                r"\b(technology|science|research|development|innovation)\b",
            ],
        }
        return defaults.get(language, [])

    def _get_default_grammar_patterns(self, language: str) -> list[str]:
        """Get default grammar patterns."""
        defaults = {
            "hr": [r"\b\w+ić\b", r"\b\w+ović\b", r"\b\w+ski\b", r"\b\w+nja\b"],
            "en": [r"\b\w+ing\b", r"\b\w+ly\b", r"\b\w+tion\b", r"\b\w+ness\b"],
        }
        return defaults.get(language, [])

    def detect_language_content(self, text: str) -> dict[str, Any]:
        """Detect language from content."""
        # Import here to avoid circular imports
        try:
            from ..preprocessing.cleaners import detect_language_content

            return detect_language_content(text)
        except ImportError:
            # Fallback detection
            if any(char in text.lower() for char in "čćšžđ"):
                return {"language": "hr", "confidence": 0.9}
            else:
                return {"language": "en", "confidence": 0.8}


# ===== FACTORY FUNCTIONS =====


def create_config_provider() -> ConfigProvider:
    """Create production configuration provider."""
    return ProductionConfigProvider()


def create_language_provider(
    config_provider: ConfigProvider = None,
) -> LanguageProvider:
    """Create production language provider."""
    if config_provider is None:
        config_provider = create_config_provider()
    return ProductionLanguageProvider(config_provider)


def create_mock_config_provider(config_dict: dict[str, Any] = None) -> ConfigProvider:
    """Create mock configuration provider for testing."""
    return MockConfigProvider(config_dict or {})


def create_mock_language_provider() -> LanguageProvider:
    """Create mock language provider for testing."""
    return MockLanguageProvider()
