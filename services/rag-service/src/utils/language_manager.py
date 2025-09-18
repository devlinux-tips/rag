"""
Language management system with dependency injection.
Testable version with pure functions and dependency injection architecture.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from .logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_config_usage,
    log_data_transformation,
    log_decision_point,
)

# ================================
# DATA CLASSES & CONFIGURATION
# ================================


@dataclass
class LanguageConfig:
    """Configuration for a single language."""

    code: str
    name: str
    native_name: str
    enabled: bool
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self):
        """Validate language configuration."""
        logger = get_system_logger()
        logger.trace("language_config", "validate", f"Validating config for {self.code}")

        if not self.code or len(self.code) < 2:
            logger.error("language_config", "validate", f"Invalid language code: {self.code}")
            raise ValueError(f"Invalid language code: {self.code}")
        if not self.name:
            logger.error("language_config", "validate", f"Language name required for {self.code}")
            raise ValueError(f"Language name required for {self.code}")

        logger.debug("language_config", "validate", f"Validation passed for {self.code} ({self.name})")


@dataclass
class LanguageSettings:
    """Complete language settings configuration."""

    supported_languages: list[str]
    default_language: str
    auto_detect: bool
    fallback_language: str
    language_names: dict[str, str]
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


@dataclass
class LanguagePatterns:
    """Language detection and processing patterns."""

    detection_patterns: dict[str, list[str]]
    stopwords: dict[str, set[str]]


@dataclass
class DetectionResult:
    """Language detection result with confidence scores."""

    detected_language: str
    confidence: float
    scores: dict[str, float]
    fallback_used: bool = False


# ================================
# DEPENDENCY INJECTION PROTOCOLS
# ================================


class ConfigProvider(Protocol):
    """Protocol for configuration access."""

    def get_language_settings(self) -> LanguageSettings:
        """Get language settings configuration."""
        ...


class PatternProvider(Protocol):
    """Protocol for language pattern access."""

    def get_language_patterns(self) -> LanguagePatterns:
        """Get language detection patterns and stopwords."""
        ...


class LoggerProvider(Protocol):
    """Protocol for logging operations."""

    def info(self, message: str) -> None:
        """Log info message."""
        ...

    def debug(self, message: str) -> None:
        """Log debug message."""
        ...

    def warning(self, message: str) -> None:
        """Log warning message."""
        ...

    def error(self, message: str) -> None:
        """Log error message."""
        ...


# ================================
# PURE BUSINESS LOGIC FUNCTIONS
# ================================


def create_language_config(
    code: str, name: str, native_name: str, enabled: bool, embedding_model: str, chunk_size: int, chunk_overlap: int
) -> LanguageConfig:
    """Pure function to create validated language configuration."""
    return LanguageConfig(
        code=code,
        name=name,
        native_name=native_name,
        enabled=enabled,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_languages_dict(settings: LanguageSettings) -> dict[str, LanguageConfig]:
    """Pure function to build languages dictionary from settings."""
    logger = get_system_logger()
    log_component_start("language_manager", "build_languages_dict", supported_count=len(settings.supported_languages))

    languages = {}

    for lang_code in settings.supported_languages:
        logger.trace("language_manager", "build_languages_dict", f"Processing language: {lang_code}")

        if lang_code not in settings.language_names:
            logger.error("language_manager", "build_languages_dict", f"Language name missing for {lang_code}")
            raise ValueError(f"Language name missing for {lang_code} in configuration")

        name = settings.language_names[lang_code]
        logger.debug("language_manager", "build_languages_dict", f"Creating config for {lang_code}: {name}")

        languages[lang_code] = create_language_config(
            code=lang_code,
            name=name,
            native_name=name,
            enabled=True,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    log_component_end(
        "language_manager",
        "build_languages_dict",
        f"Built {len(languages)} language configs",
        created_languages=list(languages.keys()),
    )
    return languages


def normalize_text_for_detection(text: str) -> list[str]:
    """Pure function to normalize text for language detection."""
    if not text:
        return []

    # Clean and normalize text
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    words = clean_text.split()

    return words


def calculate_pattern_scores(words: list[str], detection_patterns: dict[str, list[str]]) -> dict[str, float]:
    """Pure function to calculate pattern match scores for languages."""
    if not words:
        return {}

    word_set = set(words)
    language_scores = {}

    for lang_code, patterns in detection_patterns.items():
        if not patterns:
            continue

        # Count pattern matches
        matches = sum(1 for pattern in patterns if pattern in word_set)
        if matches > 0:
            language_scores[lang_code] = matches / len(patterns)

    return language_scores


def detect_language_from_text(
    text: str,
    detection_patterns: dict[str, list[str]],
    auto_detect: bool = True,
    default_language: str = "hr",
    min_words: int = 3,
) -> DetectionResult:
    """Pure function to detect language from text using pattern matching."""
    logger = get_system_logger()
    log_component_start(
        "language_detection",
        "detect_language_from_text",
        auto_detect=auto_detect,
        text_length=len(text),
        pattern_count=len(detection_patterns),
    )

    if not auto_detect or not text:
        log_decision_point(
            "language_detection",
            "detect_language_from_text",
            f"auto_detect={auto_detect}, text_empty={not text}",
            f"fallback to {default_language}",
        )
        return DetectionResult(detected_language=default_language, confidence=0.0, scores={}, fallback_used=True)

    words = normalize_text_for_detection(text)
    log_data_transformation("language_detection", "normalize_text", f"text[{len(text)}]", f"words[{len(words)}]")

    if len(words) < min_words:
        log_decision_point(
            "language_detection",
            "detect_language_from_text",
            f"words_count={len(words)} < min_words={min_words}",
            f"fallback to {default_language}",
        )
        return DetectionResult(detected_language=default_language, confidence=0.0, scores={}, fallback_used=True)

    language_scores = calculate_pattern_scores(words, detection_patterns)
    logger.debug("language_detection", "calculate_scores", f"Language scores: {language_scores}")

    if language_scores:
        detected_lang, confidence = max(language_scores.items(), key=lambda x: x[1])
        log_decision_point(
            "language_detection",
            "detect_language_from_text",
            f"scores={language_scores}",
            f"detected {detected_lang} confidence={confidence:.3f}",
        )
        log_component_end(
            "language_detection",
            "detect_language_from_text",
            f"Detected {detected_lang}",
            confidence=confidence,
            all_scores=language_scores,
        )
        return DetectionResult(
            detected_language=detected_lang, confidence=confidence, scores=language_scores, fallback_used=False
        )

    log_decision_point(
        "language_detection", "detect_language_from_text", "no_patterns_matched", f"fallback to {default_language}"
    )
    log_component_end(
        "language_detection", "detect_language_from_text", f"No detection, fallback to {default_language}"
    )
    return DetectionResult(
        detected_language=default_language, confidence=0.0, scores=language_scores, fallback_used=True
    )


def remove_stopwords_from_text(text: str, stopwords: set[str]) -> str:
    """Pure function to remove stopwords from text."""
    if not stopwords or not text:
        return text

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return " ".join(filtered_words)


def calculate_collection_suffix(language_code: str, supported_languages: list[str], fallback_language: str) -> str:
    """Pure function to calculate collection suffix for language."""
    if language_code == "auto":
        return fallback_language
    elif language_code == "multilingual":
        return "multi"
    elif language_code in supported_languages:
        return language_code
    else:
        return fallback_language


def normalize_language_code_pure(
    language_code: str, supported_languages: list[str], default_language: str
) -> tuple[str, bool]:
    """Pure function to normalize language code - returns (normalized_code, is_valid)."""
    if not language_code or language_code == "auto":
        return default_language, True

    # Handle common variations
    normalized = language_code.lower().replace("-", "_").split("_")[0]

    if normalized in supported_languages:
        return normalized, True
    else:
        # Language not supported
        return default_language, False


def validate_languages_list(
    language_codes: list[str], supported_languages: list[str], default_language: str
) -> list[str]:
    """Pure function to validate and normalize list of language codes."""
    validated = []

    for code in language_codes:
        normalized, is_valid = normalize_language_code_pure(code, supported_languages, default_language)
        if normalized not in validated:
            validated.append(normalized)

    if not validated:
        validated.append(default_language)

    return validated


def get_chunk_config_for_language(
    languages: dict[str, LanguageConfig], language_code: str, default_chunk_size: int = 512, default_overlap: int = 50
) -> tuple[int, int]:
    """Pure function to get chunk configuration for language."""
    if language_code not in languages:
        raise ValueError(f"Language {language_code} not supported")
    config = languages[language_code]
    return config.chunk_size, config.chunk_overlap


def get_embedding_model_for_language(
    languages: dict[str, LanguageConfig], language_code: str, default_model: str = "BAAI/bge-m3"
) -> str:
    """Pure function to get embedding model for language."""
    if language_code not in languages:
        raise ValueError(f"Language {language_code} not supported")
    config = languages[language_code]
    return config.embedding_model


def get_display_name_for_language(languages: dict[str, LanguageConfig], language_code: str) -> str:
    """Pure function to get human-readable language name."""
    if language_code not in languages:
        raise ValueError(f"Language {language_code} not supported")
    config = languages[language_code]
    return config.native_name


# ================================
# DEPENDENCY INJECTION ORCHESTRATION
# ================================


class _LanguageManager:
    """Testable language manager with dependency injection."""

    def __init__(
        self,
        config_provider: ConfigProvider,
        pattern_provider: PatternProvider,
        logger_provider: LoggerProvider | None = None,
    ):
        """Initialize with injected dependencies."""
        logger = get_system_logger()
        log_component_start("language_manager", "init", has_logger=logger_provider is not None)

        self._config_provider = config_provider
        self._pattern_provider = pattern_provider
        self._logger = logger_provider

        logger.debug("language_manager", "init", "Loading language settings")
        self._settings = self._config_provider.get_language_settings()
        log_config_usage(
            "language_manager",
            "settings_loaded",
            {
                "supported_languages": self._settings.supported_languages,
                "default_language": self._settings.default_language,
                "auto_detect": self._settings.auto_detect,
                "fallback_language": self._settings.fallback_language,
            },
        )

        logger.debug("language_manager", "init", "Loading language patterns")
        self._patterns = self._pattern_provider.get_language_patterns()
        logger.trace(
            "language_manager", "init", f"Loaded patterns for: {list(self._patterns.detection_patterns.keys())}"
        )

        logger.debug("language_manager", "init", "Building languages dictionary")
        self._languages = build_languages_dict(self._settings)

        log_component_end(
            "language_manager",
            "init",
            f"Loaded {len(self._languages)} supported languages",
            supported_languages=list(self._languages.keys()),
        )
        self._log_info(f"Loaded {len(self._languages)} supported languages: {list(self._languages.keys())}")

    def _log_info(self, message: str) -> None:
        """Log info message if logger available."""
        if self._logger:
            self._logger.info(message)

    def _log_debug(self, message: str) -> None:
        """Log debug message if logger available."""
        if self._logger:
            self._logger.debug(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message if logger available."""
        if self._logger:
            self._logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log error message if logger available."""
        if self._logger:
            self._logger.error(message)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return list(self._languages.keys())

    def get_language_config(self, language_code: str) -> LanguageConfig:
        """Get configuration for specific language."""
        if language_code not in self._languages:
            raise ValueError(f"Language {language_code} not supported")
        return self._languages[language_code]

    def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported and enabled."""
        return language_code in self._languages

    def get_default_language(self) -> str:
        """Get default language code."""
        return self._settings.default_language

    def get_fallback_language(self) -> str:
        """Get fallback language code."""
        return self._settings.fallback_language

    def detect_language(self, text: str) -> str:
        """Detect language from text using pattern matching."""
        logger = get_system_logger()
        log_component_start("language_manager", "detect_language", text_length=len(text))

        result = detect_language_from_text(
            text=text,
            detection_patterns=self._patterns.detection_patterns,
            auto_detect=self._settings.auto_detect,
            default_language=self._settings.default_language,
        )

        if result.scores:
            logger.debug("language_manager", "detect_language", f"Detected language: {result.detected_language}")
            self._log_debug(f"Detected language: {result.detected_language} (scores: {result.scores})")
        elif result.fallback_used:
            logger.debug(
                "language_manager",
                "detect_language",
                f"No language detected, using default: {result.detected_language}",
            )
            self._log_debug(f"No language detected, using default: {result.detected_language}")

        log_component_end(
            "language_manager",
            "detect_language",
            f"Result: {result.detected_language}",
            confidence=result.confidence,
            fallback_used=result.fallback_used,
        )
        return result.detected_language

    def detect_language_detailed(self, text: str) -> DetectionResult:
        """Detect language with detailed confidence information."""
        return detect_language_from_text(
            text=text,
            detection_patterns=self._patterns.detection_patterns,
            auto_detect=self._settings.auto_detect,
            default_language=self._settings.default_language,
        )

    def get_stopwords(self, language_code: str) -> set[str]:
        """Get stopwords for specific language."""
        if language_code not in self._patterns.stopwords:
            raise ValueError(f"Stopwords not available for language {language_code}")
        return self._patterns.stopwords[language_code]

    def remove_stopwords(self, text: str, language_code: str) -> str:
        """Remove stopwords from text for specific language."""
        stopwords = self.get_stopwords(language_code)
        return remove_stopwords_from_text(text, stopwords)

    def get_collection_suffix(self, language_code: str) -> str:
        """Get collection suffix for language (validated)."""
        suffix = calculate_collection_suffix(
            language_code=language_code,
            supported_languages=self.get_supported_languages(),
            fallback_language=self._settings.fallback_language,
        )

        if suffix != language_code and language_code not in ["auto", "multilingual"]:
            self._log_warning(f"Unsupported language {language_code}, using fallback {suffix}")

        return suffix

    def normalize_language_code(self, language_code: str) -> str:
        """Normalize language code to supported format."""
        normalized, is_valid = normalize_language_code_pure(
            language_code=language_code,
            supported_languages=self.get_supported_languages(),
            default_language=self._settings.default_language,
        )

        if not is_valid and language_code != "auto":
            self._log_warning(f"Language {language_code} not supported, using default {normalized}")

        return normalized

    def get_language_display_name(self, language_code: str) -> str:
        """Get human-readable language name."""
        return get_display_name_for_language(self._languages, language_code)

    def validate_languages(self, language_codes: list[str]) -> list[str]:
        """Validate and normalize list of language codes."""
        return validate_languages_list(
            language_codes=language_codes,
            supported_languages=self.get_supported_languages(),
            default_language=self._settings.default_language,
        )

    def get_embedding_model(self, language_code: str) -> str:
        """Get embedding model for specific language."""
        return get_embedding_model_for_language(
            languages=self._languages, language_code=language_code, default_model=self._settings.embedding_model
        )

    def get_chunk_config(self, language_code: str) -> tuple[int, int]:
        """Get chunk size and overlap for specific language."""
        return get_chunk_config_for_language(
            languages=self._languages,
            language_code=language_code,
            default_chunk_size=self._settings.chunk_size,
            default_overlap=self._settings.chunk_overlap,
        )

    def add_language_runtime(
        self,
        code: str,
        name: str,
        native_name: str,
        patterns: list[str] | None = None,
        stopwords: list[str] | None = None,
    ) -> None:
        """Add language support at runtime."""
        logger = get_system_logger()
        log_component_start(
            "language_manager",
            "add_language_runtime",
            code=code,
            name=name,
            has_patterns=patterns is not None,
            has_stopwords=stopwords is not None,
        )

        self._log_info(f"Adding runtime language support: {code} ({name})")

        logger.debug("language_manager", "add_language_runtime", f"Creating language config for {code}")
        self._languages[code] = create_language_config(
            code=code,
            name=name,
            native_name=native_name,
            enabled=True,
            embedding_model=self._settings.embedding_model,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )

        if patterns:
            logger.debug(
                "language_manager", "add_language_runtime", f"Adding {len(patterns)} detection patterns for {code}"
            )
            self._patterns.detection_patterns[code] = patterns

        if stopwords:
            logger.debug("language_manager", "add_language_runtime", f"Adding {len(stopwords)} stopwords for {code}")
            self._patterns.stopwords[code] = set(stopwords)

        log_component_end(
            "language_manager",
            "add_language_runtime",
            f"Language {code} added successfully",
            total_languages=len(self._languages),
        )
        self._log_info(f"Language {code} added successfully")


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_language_manager(
    config_provider: ConfigProvider, pattern_provider: PatternProvider, logger_provider: LoggerProvider | None = None
) -> _LanguageManager:
    """Factory function to create configured language manager."""
    return _LanguageManager(
        config_provider=config_provider, pattern_provider=pattern_provider, logger_provider=logger_provider
    )


# ================================
# PUBLIC INTERFACE
# ================================


def LanguageManager(
    config_provider: ConfigProvider | None = None,
    pattern_provider: PatternProvider | None = None,
    logger_provider: LoggerProvider | None = None,
):
    """
    Create a language manager with dependency injection.

    Args:
        config_provider: Configuration provider for language settings
        pattern_provider: Pattern provider for detection patterns and stopwords
        logger_provider: Logger provider for debugging

    Returns:
        Configured _LanguageManager instance
    """
    if not config_provider or not pattern_provider:
        from .language_manager_providers import create_production_setup

        config_provider, pattern_provider, logger_provider = create_production_setup()

    # Ensure providers are not None after potential defaults
    assert config_provider is not None, "ConfigProvider must not be None"
    assert pattern_provider is not None, "PatternProvider must not be None"

    return _LanguageManager(
        config_provider=config_provider, pattern_provider=pattern_provider, logger_provider=logger_provider
    )
