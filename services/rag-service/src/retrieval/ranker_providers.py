"""
Provider implementations for document ranker dependencies.
Includes production implementations and mock providers for testing.
"""

import logging
from typing import Any, cast

from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start, log_data_transformation
from .ranker import ConfigProvider, LanguageFeatures, LanguageProvider

logger = logging.getLogger(__name__)


# ===== PRODUCTION PROVIDERS =====


class DefaultConfigProvider(ConfigProvider):
    """Default configuration provider using config_loader."""

    def __init__(self):
        """Initialize default config provider."""
        # Import here to avoid circular imports
        from ..utils.config_loader import get_language_specific_config, get_ranking_config

        self._get_ranking_config = get_ranking_config
        self._get_language_specific_config = get_language_specific_config
        self.logger = logging.getLogger(__name__)

    def get_ranking_config(self) -> dict[str, Any]:
        """Get ranking configuration from config files."""
        system_logger = get_system_logger()
        log_component_start("ranker_providers", "get_ranking_config")

        config = self._get_ranking_config()
        if not config:
            system_logger.error(
                "ranker_providers",
                "get_ranking_config",
                "FAILED: Missing ranking configuration",
                error_type="ConfigurationError",
                stack_trace="No ranking configuration found in config files",
            )
            raise ValueError("Missing ranking configuration in config files")

        log_data_transformation(
            "ranker_providers",
            "config_loading",
            "Input: ranking config from files",
            f"Output: ranking configuration with {len(config)} keys",
            config_keys=list(config.keys()),
            config_size=len(config),
        )

        log_component_end(
            "ranker_providers",
            "get_ranking_config",
            "Successfully loaded ranking configuration",
            config_keys_count=len(config),
        )
        return config

    def get_language_specific_config(self, section: str, language: str) -> dict[str, Any]:
        """Get language-specific configuration from config files."""
        system_logger = get_system_logger()
        log_component_start("ranker_providers", "get_language_specific_config", section=section, language=language)

        config = self._get_language_specific_config(section, language)
        if not config:
            system_logger.error(
                "ranker_providers",
                "get_language_specific_config",
                f"FAILED: Missing {section} configuration for {language}",
                error_type="ConfigurationError",
                stack_trace=f"No {section} configuration found for language '{language}'",
                metadata={"section": section, "language": language},
            )
            raise ValueError(f"Missing {section} configuration for language '{language}'")

        log_data_transformation(
            "ranker_providers",
            "language_config_loading",
            f"Input: {section} config for {language}",
            f"Output: language configuration with {len(config)} keys",
            section=section,
            language=language,
            config_keys=list(config.keys()),
            config_size=len(config),
        )

        log_component_end(
            "ranker_providers",
            "get_language_specific_config",
            "Successfully loaded language-specific configuration",
            section=section,
            language=language,
            config_keys_count=len(config),
        )
        return config


class DefaultLanguageProvider(LanguageProvider):
    """Default language provider using config system."""

    def __init__(self, config_provider: ConfigProvider):
        """
        Initialize with config provider.

        Args:
            config_provider: Configuration provider for language-specific settings
        """
        self.config_provider = config_provider
        self.logger = logging.getLogger(__name__)
        self.features_cache: dict[str, LanguageFeatures] = {}

    def get_language_features(self, language: str) -> LanguageFeatures:
        """Get language features from configuration."""
        system_logger = get_system_logger()
        log_component_start(
            "ranker_providers", "get_language_features", language=language, cache_hit=language in self.features_cache
        )

        if language in self.features_cache:
            log_component_end(
                "ranker_providers",
                "get_language_features",
                "Language features retrieved from cache",
                language=language,
                cache_hit=True,
            )
            return self.features_cache[language]

        try:
            # Get language-specific configuration
            language_config = self.config_provider.get_language_specific_config("retrieval", language)

            if "morphology" not in language_config:
                system_logger.error(
                    "ranker_providers",
                    "get_language_features",
                    f"FAILED: Missing morphology section for {language}",
                    error_type="ConfigurationError",
                    stack_trace=f"Missing 'morphology' section in language config for '{language}'",
                    metadata={"language": language},
                )
                raise ValueError(f"Missing 'morphology' section in language config for '{language}'")
            morphology = language_config["morphology"]

            # Build language features from configuration
            features = self._build_language_features(language, morphology)

            log_data_transformation(
                "ranker_providers",
                "features_building",
                f"Input: morphology config for {language}",
                f"Output: LanguageFeatures with {len(features.importance_words)} importance words",
                language=language,
                importance_words_count=len(features.importance_words),
                cultural_patterns_count=len(features.cultural_patterns),
                grammar_patterns_count=len(features.grammar_patterns),
            )

            self.features_cache[language] = features

            log_component_end(
                "ranker_providers",
                "get_language_features",
                "Successfully built and cached language features",
                language=language,
                importance_words_count=len(features.importance_words),
            )
            return features

        except Exception as e:
            system_logger.error(
                "ranker_providers",
                "get_language_features",
                f"FAILED: Language features loading error for {language}",
                error_type=type(e).__name__,
                stack_trace=str(e),
                metadata={"language": language},
            )
            raise

    def _build_language_features(self, language: str, morphology: dict[str, Any]) -> LanguageFeatures:
        """Build language features from configuration."""
        get_system_logger()
        log_component_start(
            "ranker_providers", "_build_language_features", language=language, morphology_keys=list(morphology.keys())
        )

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
        type_weights = {"encyclopedia": 1.2, "academic": 1.1, "news": 1.0, "blog": 0.9, "forum": 0.8, "social": 0.7}

        features = LanguageFeatures(
            importance_words=importance_words,
            quality_indicators=quality_indicators,
            cultural_patterns=cultural_patterns,
            grammar_patterns=grammar_patterns,
            type_weights=type_weights,
        )

        log_component_end(
            "ranker_providers",
            "_build_language_features",
            "Successfully built language features",
            language=language,
            importance_words_count=len(importance_words),
            quality_positive_count=len(quality_indicators["positive"]),
            cultural_patterns_count=len(cultural_patterns),
            grammar_patterns_count=len(grammar_patterns),
            type_weights_count=len(type_weights),
        )

        return features

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
        from ..preprocessing.cleaners import detect_language_content_with_config

        # Try to detect language using config-based approach
        try:
            # Cast config provider to expected type
            from ..preprocessing.cleaners import ConfigProvider as CleanersConfigProvider

            cleaners_provider = cast(CleanersConfigProvider, self.config_provider)
            hr_confidence = detect_language_content_with_config(text, "hr", cleaners_provider)
            en_confidence = detect_language_content_with_config(text, "en", cleaners_provider)

            if hr_confidence > en_confidence:
                return {"language": "hr", "confidence": hr_confidence}
            else:
                return {"language": "en", "confidence": en_confidence}
        except Exception:
            # Fallback to simple heuristic
            if any(char in text.lower() for char in "čćšžđ"):
                return {"language": "hr", "confidence": 0.9}
            else:
                return {"language": "en", "confidence": 0.8}


# ===== FACTORY FUNCTIONS =====


def create_config_provider() -> ConfigProvider:
    """Create default configuration provider."""
    from ..utils.config_protocol import DefaultConfigProvider as FullDefaultConfigProvider

    return FullDefaultConfigProvider()


def create_language_provider(config_provider: ConfigProvider | None = None) -> LanguageProvider:
    """Create default language provider."""
    if config_provider is None:
        config_provider = create_config_provider()
    return DefaultLanguageProvider(config_provider)
