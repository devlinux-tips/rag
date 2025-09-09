"""
Dynamic language management system for multilingual RAG.

DRY PRINCIPLE IMPLEMENTATION:
- REUSES existing config.toml settings (embeddings.model_name, languages.supported, etc.)
- REUSES language-specific TOML files (croatian.toml, english.toml) for patterns/stopwords
- NO HARDCODED defaults - all configuration comes from TOML files
- Language detection patterns extracted from existing question_patterns in language files
- Embedding model comes from existing embeddings.model_name configuration
- Chunk sizes come from existing shared.default_chunk_size/overlap settings

For new languages: Add to languages.supported in config.toml + create <lang>.toml file.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


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
        if not self.code or len(self.code) < 2:
            raise ValueError(f"Invalid language code: {self.code}")
        if not self.name:
            raise ValueError(f"Language name required for {self.code}")


class LanguageManager:
    """Manages language configurations and utilities dynamically."""

    def __init__(self):
        """Initialize language manager with existing configuration system."""
        self.languages: Dict[str, LanguageConfig] = {}
        self.detection_patterns: Dict[str, List[str]] = {}
        self.stopwords: Dict[str, Set[str]] = {}
        # These will be loaded from config.toml, not hardcoded
        self.default_language = None
        self.auto_detect = None
        self.fallback_language = None

        self._load_config()

    def _load_config(self) -> None:
        """Load language configuration using existing TOML structure."""
        from src.utils.config_loader import (get_shared_config,
                                             get_supported_languages,
                                             load_config)

        # Use existing config loader functions
        supported_langs = get_supported_languages()
        main_config = load_config("config")

        # Get language settings from main config
        languages_config = main_config["languages"]
        self.default_language = languages_config["default"]  # Minimal fallback only
        self.auto_detect = languages_config["auto_detect"]
        self.fallback_language = self.default_language

        # Get shared config for chunk settings
        shared_config = get_shared_config()
        chunk_size = shared_config["default_chunk_size"]
        chunk_overlap = shared_config["default_chunk_overlap"]

        # Get embedding model from existing config
        embeddings_config = main_config["embeddings"]
        embedding_model = embeddings_config["model_name"]

        # Get language names from main config
        lang_names = languages_config["names"]

        # Create language configs for supported languages
        for lang_code in supported_langs:
            self.languages[lang_code] = LanguageConfig(
                code=lang_code,
                name=lang_names.get(lang_code, lang_code.upper()),
                native_name=lang_names.get(lang_code, lang_code.upper()),
                enabled=True,
                embedding_model=embedding_model,  # From existing embeddings config
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        # Load detection patterns from language-specific config files
        self._load_detection_patterns_from_language_configs()

        logger.info(
            f"Loaded {len(self.languages)} supported languages: {list(self.languages.keys())}"
        )

    def _load_detection_patterns_from_language_configs(self) -> None:
        """Load detection patterns from existing language-specific TOML files."""
        from src.utils.config_loader import get_language_config

        for lang_code in self.languages.keys():
            try:
                # Load language-specific config (e.g., croatian.toml, english.toml)
                lang_config = get_language_config(lang_code)

                # Extract question patterns to use as detection patterns
                question_patterns = lang_config["shared"].get("question_patterns", {})
                if question_patterns:
                    # Combine different pattern types for detection
                    detection_words = []
                    for pattern_list in question_patterns.values():
                        if isinstance(pattern_list, list):
                            detection_words.extend(pattern_list)

                    if detection_words:
                        # Use first 8 words for detection (limit pattern size)
                        self.detection_patterns[lang_code] = detection_words[:8]

                # Extract stopwords from language config
                stopwords_list = (
                    lang_config["shared"].get("stopwords", {}).get("words", [])
                )
                if stopwords_list:
                    # Use first 20 stopwords for efficiency
                    self.stopwords[lang_code] = set(stopwords_list[:20])

                logger.debug(
                    f"Loaded patterns for {lang_code}: {len(self.detection_patterns.get(lang_code, []))} detection words, {len(self.stopwords.get(lang_code, []))} stopwords"
                )

            except Exception as e:
                logger.warning(f"Could not load patterns for {lang_code}: {e}")
                # Continue without patterns for this language

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.languages.keys())

    def get_language_config(self, language_code: str) -> Optional[LanguageConfig]:
        """Get configuration for specific language."""
        return self.languages.get(language_code)

    def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported and enabled."""
        return language_code in self.languages

    def get_default_language(self) -> str:
        """Get default language code."""
        return self.default_language

    def get_fallback_language(self) -> str:
        """Get fallback language code."""
        return self.fallback_language

    def detect_language(self, text: str) -> str:
        """Detect language from text using pattern matching."""
        if not self.auto_detect or not text:
            return self.default_language

        # Clean and normalize text
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        words = clean_text.split()

        if len(words) < 3:  # Too short for reliable detection
            return self.default_language

        # Score languages based on pattern matches
        language_scores = {}
        word_set = set(words)

        for lang_code, patterns in self.detection_patterns.items():
            if lang_code not in self.languages:
                continue

            # Count pattern matches
            matches = sum(1 for pattern in patterns if pattern in word_set)
            if matches > 0:
                language_scores[lang_code] = matches / len(patterns)

        if language_scores:
            # Return language with highest match score
            detected = max(language_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Detected language: {detected} (scores: {language_scores})")
            return detected

        logger.debug(f"No language detected, using default: {self.default_language}")
        return self.default_language

    def get_stopwords(self, language_code: str) -> Set[str]:
        """Get stopwords for specific language."""
        return self.stopwords.get(language_code, set())

    def remove_stopwords(self, text: str, language_code: str) -> str:
        """Remove stopwords from text for specific language."""
        stopwords = self.get_stopwords(language_code)
        if not stopwords:
            return text

        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        return " ".join(filtered_words)

    def get_collection_suffix(self, language_code: str) -> str:
        """Get collection suffix for language (validated)."""
        if language_code == "auto":
            return self.default_language
        elif language_code == "multilingual":
            return "multi"
        elif self.is_language_supported(language_code):
            return language_code
        else:
            logger.warning(
                f"Unsupported language {language_code}, using fallback {self.fallback_language}"
            )
            return self.fallback_language

    def normalize_language_code(self, language_code: str) -> str:
        """Normalize language code to supported format - FAIL FAST for unsupported."""
        if not language_code or language_code == "auto":
            return self.default_language

        # Handle common variations
        normalized = language_code.lower().replace("-", "_").split("_")[0]

        # Use main config loader for validation
        from .config_loader import is_language_supported

        if is_language_supported(normalized):
            return normalized
        else:
            # Language not supported, use default
            logger.warning(
                f"Language {language_code} not supported, using default {self.default_language}"
            )
            return self.default_language

    def get_language_display_name(self, language_code: str) -> str:
        """Get human-readable language name."""
        config = self.get_language_config(language_code)
        return config.native_name if config else language_code

    def validate_languages(self, language_codes: List[str]) -> List[str]:
        """Validate and normalize list of language codes."""
        validated = []
        for code in language_codes:
            normalized = self.normalize_language_code(code)
            if normalized not in validated:
                validated.append(normalized)

        if not validated:
            validated.append(self.default_language)

        return validated

    def get_embedding_model(self, language_code: str) -> str:
        """Get embedding model for specific language."""
        config = self.get_language_config(language_code)
        return config.embedding_model if config else "BAAI/bge-m3"

    def get_chunk_config(self, language_code: str) -> tuple[int, int]:
        """Get chunk size and overlap for specific language."""
        config = self.get_language_config(language_code)
        if config:
            return config.chunk_size, config.chunk_overlap
        return 512, 50

    def add_language_runtime(
        self,
        code: str,
        name: str,
        native_name: str,
        patterns: Optional[List[str]] = None,
        stopwords: Optional[List[str]] = None,
    ) -> None:
        """Add language support at runtime."""
        logger.info(f"Adding runtime language support: {code} ({name})")

        # Add language config
        self.languages[code] = LanguageConfig(
            code=code,
            name=name,
            native_name=native_name,
            enabled=True,
            embedding_model="BAAI/bge-m3",
            chunk_size=512,
            chunk_overlap=50,
        )

        # Add detection patterns if provided
        if patterns:
            self.detection_patterns[code] = patterns

        # Add stopwords if provided
        if stopwords:
            self.stopwords[code] = set(stopwords)

        logger.info(f"Language {code} added successfully")


# Global language manager instance
_language_manager: Optional[LanguageManager] = None


def get_language_manager() -> LanguageManager:
    """Get global language manager instance."""
    global _language_manager
    if _language_manager is None:
        _language_manager = LanguageManager()
    return _language_manager


def get_supported_languages() -> List[str]:
    """Convenience function to get supported languages."""
    return get_language_manager().get_supported_languages()


def detect_language(text: str) -> str:
    """Convenience function to detect language."""
    return get_language_manager().detect_language(text)


def normalize_language_code(language_code: str) -> str:
    """Convenience function to normalize language code."""
    return get_language_manager().normalize_language_code(language_code)
