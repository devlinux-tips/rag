"""
Multilingual text cleaning system with language-aware preprocessing.
Provides deterministic text normalization and cleaning with configurable
language-specific rules and encoding preservation.
"""

import locale
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass
class CleaningResult:
    """Result of text cleaning operation."""

    text: str
    original_length: int
    cleaned_length: int
    operations_performed: List[str]
    language_score: Optional[float] = None
    is_meaningful: Optional[bool] = None


@dataclass
class LanguageConfig:
    """Language-specific configuration for text processing."""

    diacritic_map: Dict[str, str]
    word_char_pattern: str
    locale_primary: Optional[str] = None
    locale_fallback: str = "C.UTF-8"


@dataclass
class CleaningConfig:
    """General cleaning configuration."""

    multiple_whitespace: str
    multiple_linebreaks: str
    min_meaningful_words: int
    min_word_char_ratio: float


@dataclass
class DocumentCleaningConfig:
    """Document-specific cleaning configuration."""

    header_footer_patterns: List[str]
    ocr_corrections: Dict[str, str]


@dataclass
class ChunkingConfig:
    """Chunking-related configuration."""

    sentence_ending_pattern: str
    min_sentence_length: int


@dataclass
class SharedLanguageConfig:
    """Shared language configuration."""

    stopwords: List[str]
    chars_pattern: str


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration access."""

    def get_language_config(self, language: str) -> Dict[str, any]:
        """Get language-specific text processing configuration."""
        ...

    def get_cleaning_config(self) -> Dict[str, any]:
        """Get general cleaning configuration."""
        ...

    def get_document_cleaning_config(self, language: str) -> Dict[str, any]:
        """Get document cleaning configuration."""
        ...

    def get_chunking_config(self, language: str) -> Dict[str, any]:
        """Get chunking configuration."""
        ...

    def get_shared_language_config(self, language: str) -> Dict[str, any]:
        """Get shared language configuration."""
        ...


@runtime_checkable
class LoggerProvider(Protocol):
    """Protocol for logging operations."""

    def debug(self, message: str) -> None:
        """Log debug message."""
        ...

    def info(self, message: str) -> None:
        """Log info message."""
        ...

    def error(self, message: str) -> None:
        """Log error message."""
        ...


@runtime_checkable
class EnvironmentProvider(Protocol):
    """Protocol for environment operations."""

    def set_environment_variable(self, key: str, value: str) -> None:
        """Set environment variable."""
        ...

    def set_locale(self, category: int, locale_name: str) -> None:
        """Set locale."""
        ...


# ================================
# PURE BUSINESS LOGIC FUNCTIONS
# ================================


def normalize_whitespace(text: str, preserve_structure: bool = True) -> str:
    """
    Normalize whitespace while optionally preserving paragraph structure.
    Pure function with no side effects.

    Args:
        text: Input text
        preserve_structure: Whether to preserve paragraph breaks

    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""

    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    if preserve_structure:
        # Preserve paragraph breaks but normalize other line breaks
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Max 2 line breaks
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # Single line breaks to space
    else:
        # Convert all line breaks to spaces
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_formatting_artifacts(text: str) -> str:
    """
    Remove document formatting artifacts.
    Pure function with no side effects.

    Args:
        text: Input text

    Returns:
        Text with formatting artifacts removed
    """
    if not text:
        return ""

    # Remove excessive punctuation
    text = re.sub(r"[.]{3,}", "...", text)  # Multiple dots to ellipsis
    text = re.sub(r"[-]{3,}", "---", text)  # Multiple dashes

    # Remove isolated formatting characters
    text = re.sub(r"\s+[_*-]\s+", " ", text)

    # Remove standalone special characters that are formatting artifacts
    text = re.sub(r"^\s*[_*-]+\s*$", "", text, flags=re.MULTILINE)

    return text


def remove_headers_footers(text: str, header_footer_patterns: List[str]) -> str:
    """
    Remove common document headers and footers.
    Pure function with no side effects.

    Args:
        text: Input text
        header_footer_patterns: List of regex patterns to match headers/footers

    Returns:
        Text with headers and footers removed
    """
    if not text or not header_footer_patterns:
        return text

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        is_artifact = False
        for pattern in header_footer_patterns:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                is_artifact = True
                break

        if not is_artifact:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def fix_ocr_errors(text: str, ocr_corrections: Dict[str, str]) -> str:
    """
    Fix common OCR errors using correction mapping.
    Pure function with no side effects.

    Args:
        text: Input text
        ocr_corrections: Dictionary mapping error patterns to corrections

    Returns:
        Text with OCR errors corrected
    """
    if not text or not ocr_corrections:
        return text

    for pattern, replacement in ocr_corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def extract_sentences(
    text: str, sentence_ending_pattern: str, min_sentence_length: int
) -> List[str]:
    """
    Extract sentences from text using language-specific patterns.
    Pure function with no side effects.

    Args:
        text: Input text
        sentence_ending_pattern: Regex pattern for sentence endings
        min_sentence_length: Minimum sentence length to include

    Returns:
        List of sentences
    """
    if not text:
        return []

    sentences = re.split(sentence_ending_pattern, text)

    # Clean and filter sentences
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > min_sentence_length:
            clean_sentences.append(sentence)

    return clean_sentences


def normalize_diacritics(text: str, diacritic_map: Dict[str, str]) -> str:
    """
    Normalize diacritics using character mapping.
    Pure function with no side effects.

    Args:
        text: Text with diacritics
        diacritic_map: Mapping of diacritic to normalized characters

    Returns:
        Text with normalized characters
    """
    if not text or not diacritic_map:
        return text

    for diacritic, normalized in diacritic_map.items():
        text = text.replace(diacritic, normalized)

    return text


def is_meaningful_text(
    text: str, min_words: int, word_char_pattern: str, min_word_char_ratio: float
) -> bool:
    """
    Check if text contains meaningful content.
    Pure function with no side effects.

    Args:
        text: Text to check
        min_words: Minimum number of words required
        word_char_pattern: Regex pattern for word characters
        min_word_char_ratio: Minimum ratio of word characters to total

    Returns:
        True if text is meaningful
    """
    if not text or not text.strip():
        return False

    words = text.split()

    # Check minimum word count
    if len(words) < min_words:
        return False

    # Check if text is mostly numbers or special characters
    word_chars = sum(len(re.findall(word_char_pattern, word)) for word in words)
    total_chars = len(re.sub(r"\s", "", text))

    if total_chars == 0 or word_chars / total_chars < min_word_char_ratio:
        return False

    return True


def detect_language_content(text: str, language_words: List[str]) -> float:
    """
    Detect how much content matches the target language using Unicode analysis.
    Pure function with no side effects.

    Args:
        text: Text to analyze
        language_words: List of common words in the target language

    Returns:
        Score between 0.0 and 1.0 indicating language content ratio
    """
    if not text:
        return 0.0

    # Unicode-based diacritic detection - works for ALL languages
    def has_diacritics(char):
        return unicodedata.normalize("NFD", char) != char

    # Convert language words to set for faster lookup
    language_word_set = set(word.lower() for word in language_words)

    # Count Unicode diacritics (works for Croatian, French, German, etc.)
    char_score = sum(1 for char in text if has_diacritics(char))
    word_score = sum(1 for word in language_word_set if word in text.lower().split())

    # Normalize scores (language-agnostic approach)
    total_chars = len(text)
    total_words = len(text.split())

    # Dynamic diacritic expectation based on actual text
    char_ratio = char_score / max(total_chars * 0.05, 1) if char_score > 0 else 0  # 5% threshold
    word_ratio = word_score / max(total_words * 0.2, 1)  # 20% common words threshold

    # Language-agnostic scoring
    score = min((char_ratio + word_ratio) / 2, 1.0) if char_score > 0 else min(word_ratio, 1.0)

    return score


def preserve_text_encoding(text: any) -> str:
    """
    Preserve proper text encoding.
    Pure function with no side effects.

    Args:
        text: Text to process (str or bytes)

    Returns:
        Text with proper encoding
    """
    # If text is already properly encoded, return as-is
    if isinstance(text, str):
        return text

    # Try to decode if it's bytes
    if isinstance(text, bytes):
        try:
            return text.decode("utf-8")
        except UnicodeDecodeError:
            return text.decode("latin1")

    return str(text)


def clean_text_comprehensive(
    text: str,
    language_config: LanguageConfig,
    cleaning_config: CleaningConfig,
    document_cleaning_config: DocumentCleaningConfig,
    preserve_structure: bool = True,
) -> CleaningResult:
    """
    Comprehensive text cleaning using pure functions.
    Pure function with no side effects.

    Args:
        text: Input text
        language_config: Language-specific configuration
        cleaning_config: General cleaning configuration
        document_cleaning_config: Document cleaning configuration
        preserve_structure: Whether to preserve paragraph structure

    Returns:
        CleaningResult with cleaned text and metadata
    """
    if not text or not text.strip():
        return CleaningResult(
            text="",
            original_length=len(text) if text else 0,
            cleaned_length=0,
            operations_performed=[],
        )

    original_length = len(text)
    operations = []

    # Start with the original text
    cleaned = text

    # Remove document header/footer artifacts
    cleaned = remove_headers_footers(cleaned, document_cleaning_config.header_footer_patterns)
    if len(cleaned) != len(text):
        operations.append("header_footer_removal")

    # Normalize whitespace and line breaks
    old_cleaned = cleaned
    cleaned = normalize_whitespace(cleaned, preserve_structure)
    if cleaned != old_cleaned:
        operations.append("whitespace_normalization")

    # Remove formatting artifacts
    old_cleaned = cleaned
    cleaned = remove_formatting_artifacts(cleaned)
    if cleaned != old_cleaned:
        operations.append("formatting_artifact_removal")

    # Fix common OCR errors
    old_cleaned = cleaned
    cleaned = fix_ocr_errors(cleaned, document_cleaning_config.ocr_corrections)
    if cleaned != old_cleaned:
        operations.append("ocr_error_correction")

    # Final cleanup
    cleaned = cleaned.strip()

    return CleaningResult(
        text=cleaned,
        original_length=original_length,
        cleaned_length=len(cleaned),
        operations_performed=operations,
    )


# ================================
# DEPENDENCY INJECTION ORCHESTRATION
# ================================


class MultilingualTextCleaner:
    """Multilingual text cleaner with configurable language-aware preprocessing."""

    def __init__(
        self,
        language: str,
        config_provider: ConfigProvider,
        logger_provider: Optional[LoggerProvider] = None,
        environment_provider: Optional[EnvironmentProvider] = None,
    ):
        """Initialize cleaner with injected dependencies."""
        self.language = language
        self._config_provider = config_provider
        self._logger = logger_provider
        self._environment = environment_provider

        # Load all configurations once during initialization
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load all required configurations."""
        # Language-specific configuration
        language_data = self._config_provider.get_language_config(self.language)
        # Validate required language configuration keys
        if "word_char_pattern" not in language_data:
            raise ValueError("Missing 'word_char_pattern' in language configuration")
        if "locale" not in language_data:
            raise ValueError("Missing 'locale' in language configuration")
        if "primary" not in language_data["locale"]:
            raise ValueError("Missing 'primary' in locale configuration")
        if "fallback" not in language_data["locale"]:
            raise ValueError("Missing 'fallback' in locale configuration")

        self.language_config = LanguageConfig(
            diacritic_map=language_data["diacritic_map"],
            word_char_pattern=language_data["word_char_pattern"],
            locale_primary=language_data["locale"]["primary"],
            locale_fallback=language_data["locale"]["fallback"],
        )

        # General cleaning configuration
        cleaning_data = self._config_provider.get_cleaning_config()
        self.cleaning_config = CleaningConfig(
            multiple_whitespace=cleaning_data["multiple_whitespace"],
            multiple_linebreaks=cleaning_data["multiple_linebreaks"],
            min_meaningful_words=cleaning_data["min_meaningful_words"],
            min_word_char_ratio=cleaning_data["min_word_char_ratio"],
        )

        # Document cleaning configuration
        doc_cleaning_data = self._config_provider.get_document_cleaning_config(self.language)
        self.document_cleaning_config = DocumentCleaningConfig(
            header_footer_patterns=doc_cleaning_data["header_footer_patterns"],
            ocr_corrections=doc_cleaning_data["ocr_corrections"],
        )

        # Chunking configuration (merged)
        chunking_data = self._config_provider.get_chunking_config(self.language)
        self.chunking_config = ChunkingConfig(
            sentence_ending_pattern=chunking_data["sentence_ending_pattern"],
            min_sentence_length=chunking_data["min_sentence_length"],
        )

        # Shared language configuration
        shared_data = self._config_provider.get_shared_language_config(self.language)
        # Validate shared language configuration
        if "chars_pattern" not in shared_data:
            raise ValueError("Missing 'chars_pattern' in shared language configuration")

        self.shared_config = SharedLanguageConfig(
            stopwords=shared_data["stopwords"]["words"],
            chars_pattern=shared_data["chars_pattern"],
        )

    def clean_text(self, text: str, preserve_structure: bool = True) -> CleaningResult:
        """
        Clean and normalize text for the specified language.

        Args:
            text: Raw text to clean
            preserve_structure: Whether to preserve paragraph structure

        Returns:
            CleaningResult with cleaned text and metadata
        """
        self._log_debug(f"Cleaning text of length {len(text) if text else 0}")

        result = clean_text_comprehensive(
            text=text,
            language_config=self.language_config,
            cleaning_config=self.cleaning_config,
            document_cleaning_config=self.document_cleaning_config,
            preserve_structure=preserve_structure,
        )

        self._log_debug(f"Cleaned text length: {result.cleaned_length}")

        return result

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text for the specified language."""
        return extract_sentences(
            text=text,
            sentence_ending_pattern=self.chunking_config.sentence_ending_pattern,
            min_sentence_length=self.chunking_config.min_sentence_length,
        )

    def normalize_diacritics(self, text: str) -> str:
        """Normalize diacritics for the specified language."""
        return normalize_diacritics(text, self.language_config.diacritic_map)

    def is_meaningful_text(self, text: str, min_words: int = None) -> bool:
        """Check if text contains meaningful content."""
        if min_words is None:
            min_words = self.cleaning_config.min_meaningful_words

        return is_meaningful_text(
            text=text,
            min_words=min_words,
            word_char_pattern=self.language_config.word_char_pattern,
            min_word_char_ratio=self.cleaning_config.min_word_char_ratio,
        )

    def detect_language_content(self, text: str) -> float:
        """Detect how much content matches the current language."""
        if not text:
            return 0.0

        # Simple language detection based on character patterns and stopwords
        words = text.lower().split()
        if not words:
            return 0.0

        # Check for language-specific characters
        language_chars = 0
        total_chars = len(text.replace(" ", ""))

        if (
            hasattr(self.language_config, "word_char_pattern")
            and self.language_config.word_char_pattern
        ):
            import re

            matches = re.findall(self.language_config.word_char_pattern, text)
            language_chars = len("".join(matches))

        char_ratio = language_chars / total_chars if total_chars > 0 else 0.0

        # Check for stopwords if available
        stopword_ratio = 0.0
        if hasattr(self.shared_config, "stopwords") and self.shared_config.stopwords:
            matching_stopwords = sum(1 for word in words if word in self.shared_config.stopwords)
            stopword_ratio = matching_stopwords / len(words) if words else 0.0

        # Combined score
        return (char_ratio * 0.6) + (stopword_ratio * 0.4)

    def preserve_encoding(self, text: any) -> str:
        """Preserve proper text encoding for the language."""
        if not isinstance(text, str):
            text = str(text)

        # Apply language-specific diacritic preservation
        if hasattr(self.language_config, "diacritic_map") and self.language_config.diacritic_map:
            # For languages with diacritics, ensure proper UTF-8 encoding
            text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

        return text

    def setup_language_environment(self) -> None:
        """Setup language-specific environment from configuration."""
        if not self._environment:
            return

        # Set UTF-8 encoding
        self._environment.set_environment_variable("PYTHONIOENCODING", "utf-8")

        # Set language-specific locale (fail-fast approach)
        if self.language_config.locale_primary:
            self._environment.set_locale(locale.LC_ALL, self.language_config.locale_primary)

    def _log_debug(self, message: str) -> None:
        """Log debug message if logger available."""
        if self._logger:
            self._logger.debug(message)


# ================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# ================================


def clean_text(
    text: str,
    language: str,
    preserve_structure: bool = True,
    config_provider: Optional[ConfigProvider] = None,
) -> str:
    """
    Clean and normalize text for the specified language.

    Args:
        text: Raw text to clean
        language: Language code
        preserve_structure: Whether to preserve paragraph structure
        config_provider: Optional config provider (uses production if None)

    Returns:
        Cleaned text
    """
    from .cleaners_providers import create_config_provider

    provider = config_provider or create_config_provider()
    cleaner = MultilingualTextCleaner(language, provider)
    result = cleaner.clean_text(text, preserve_structure)

    return result.text


def detect_language_content_with_config(
    text: str, language: str, config_provider: Optional[ConfigProvider] = None
) -> float:
    """
    Detect how much content matches the specified language.

    Args:
        text: Text to analyze
        language: Language code
        config_provider: Optional config provider (uses production if None)

    Returns:
        Score between 0.0 and 1.0 indicating language content ratio
    """
    from .cleaners_providers import create_config_provider

    provider = config_provider or create_config_provider()
    cleaner = MultilingualTextCleaner(language, provider)

    return cleaner.detect_language_content(text)


def preserve_text_encoding_with_config(
    text: any, language: str, config_provider: Optional[ConfigProvider] = None
) -> str:
    """
    Preserve proper text encoding for the specified language.

    Args:
        text: Text to process
        language: Language code
        config_provider: Optional config provider (uses production if None)

    Returns:
        Text with proper encoding
    """
    from .cleaners_providers import create_config_provider

    provider = config_provider or create_config_provider()
    cleaner = MultilingualTextCleaner(language, provider)

    return cleaner.preserve_encoding(text)


def setup_language_environment(
    language: str, config_provider: Optional[ConfigProvider] = None
) -> None:
    """
    Setup language-specific environment.

    Args:
        language: Language code
        config_provider: Optional config provider (uses production if None)
    """
    from .cleaners_providers import create_config_provider, create_environment_provider

    config_prov = config_provider or create_config_provider()
    env_prov = create_environment_provider()
    cleaner = MultilingualTextCleaner(language, config_prov, environment_provider=env_prov)

    cleaner.setup_language_environment()
