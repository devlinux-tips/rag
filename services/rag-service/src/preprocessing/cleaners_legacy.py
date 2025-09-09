"""
Multilingual text cleaning and normalization utilities.
Handles language-specific text processing challenges through configuration-driven approach.
NO HARDCODED DEFAULTS - all settings come from language-specific configuration files.
"""

import locale
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config_loader import (get_chunking_config, get_cleaning_config,
                                   get_language_shared,
                                   get_language_specific_config)

logger = logging.getLogger(__name__)


class MultilingualTextCleaner:
    """Clean and normalize multilingual text for RAG processing through configuration only."""

    def __init__(self, language: str):
        """
        Initialize the multilingual text cleaner.

        Args:
            language: Language code - must be supported in configuration
        """
        self.language = language

        # Load ALL configuration from language-specific files - fail if not found
        self._language_config = get_language_specific_config(
            "text_processing", self.language
        )
        self._cleaning_config = get_cleaning_config()
        self._document_cleaning_config = get_language_specific_config(
            "document_cleaning", self.language
        )

        # Load chunking config: merge main config with language-specific overrides
        main_chunking_config = get_chunking_config()
        language_chunking_config = get_language_specific_config(
            "chunking", self.language
        )
        # Merge configs: language-specific overrides main config
        self._chunking_config = {**main_chunking_config, **language_chunking_config}

        # Language-specific diacritic mappings for normalization (from config)
        self.diacritic_map = self._language_config["diacritic_map"]

        # Language-specific stopwords (from shared config)
        shared_config = get_language_shared(self.language)
        self.stopwords = set(shared_config["stopwords"]["words"])

        # Document formatting artifacts to remove (from config)
        char_pattern = shared_config.get("chars_pattern", r"[^\w\s.,!?:;()-]")

        self.formatting_patterns = [
            self._cleaning_config["multiple_whitespace"],  # Multiple whitespaces
            self._cleaning_config["multiple_linebreaks"],  # Multiple line breaks
            char_pattern,  # Language-specific character patterns
            r"^\s*\d+\s*$",  # Standalone page numbers
            r"^\s*[IVX]+\s*$",  # Roman numerals
            r"^\s*[a-z]\)\s*$",  # List markers like a), b)
        ]

    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        Clean and normalize text for the specified language.

        Args:
            text: Raw text to clean
            preserve_structure: Whether to preserve paragraph structure

        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""

        logger.debug(f"Cleaning text of length {len(text)}")

        # Start with the original text
        cleaned = text

        # Remove document header/footer artifacts
        cleaned = self._remove_headers_footers(cleaned)

        # Normalize whitespace and line breaks
        cleaned = self._normalize_whitespace(cleaned, preserve_structure)

        # Remove formatting artifacts
        cleaned = self._remove_formatting_artifacts(cleaned)

        # Fix common OCR errors for the specified language
        cleaned = self._fix_ocr_errors(cleaned)

        # Final cleanup
        cleaned = cleaned.strip()

        logger.debug(f"Cleaned text length: {len(cleaned)}")
        return cleaned

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common document headers and footers."""
        # Remove page headers with page numbers and dates
        patterns = self._document_cleaning_config["header_footer_patterns"]

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            is_artifact = False
            for pattern in patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_artifact = True
                    break

            if not is_artifact:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _normalize_whitespace(self, text: str, preserve_structure: bool) -> str:
        """Normalize whitespace while optionally preserving paragraph structure."""
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

    def _remove_formatting_artifacts(self, text: str) -> str:
        """Remove document formatting artifacts."""
        # Remove excessive punctuation
        text = re.sub(r"[.]{3,}", "...", text)  # Multiple dots to ellipsis
        text = re.sub(r"[-]{3,}", "---", text)  # Multiple dashes

        # Remove isolated formatting characters
        text = re.sub(r"\s+[_*-]\s+", " ", text)

        # Remove standalone special characters that are formatting artifacts
        text = re.sub(r"^\s*[_*-]+\s*$", "", text, flags=re.MULTILINE)

        return text

    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors for the specified language."""
        # Common OCR mistakes for the current language (from config)
        ocr_fixes = self._document_cleaning_config["ocr_corrections"]

        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text for the specified language.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Language-specific sentence ending patterns from config
        sentence_endings = self._chunking_config["sentence_ending_pattern"]

        sentences = re.split(sentence_endings, text)

        # Clean and filter sentences
        clean_sentences = []
        min_length = self._chunking_config["min_sentence_length"]
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > min_length:
                clean_sentences.append(sentence)

        return clean_sentences

    def normalize_diacritics(self, text: str) -> str:
        """
        Normalize diacritics for the specified language (optional - use carefully as it loses information).

        Args:
            text: Text with diacritics

        Returns:
            Text with normalized characters
        """
        for diacritic, normalized in self.diacritic_map.items():
            text = text.replace(diacritic, normalized)

        return text

    def is_meaningful_text(self, text: str, min_words: int = None) -> bool:
        """
        Check if text contains meaningful content.

        Args:
            text: Text to check
            min_words: Minimum number of words required (uses config if None)

        Returns:
            True if text is meaningful
        """
        if not text or not text.strip():
            return False

        if min_words is None:
            min_words = self._cleaning_config["min_meaningful_words"]

        words = text.split()

        # Check minimum word count
        if len(words) < min_words:
            return False

        # Get language-specific word character pattern from config
        word_char_pattern = self._language_config.get("word_char_pattern", r"[a-zA-Z]")

        # Check if text is mostly numbers or special characters
        word_chars = sum(len(re.findall(word_char_pattern, word)) for word in words)
        total_chars = len(re.sub(r"\s", "", text))

        min_ratio = self._cleaning_config["min_word_char_ratio"]
        if total_chars == 0 or word_chars / total_chars < min_ratio:
            return False

        return True

    def detect_language_content(self, text: str) -> float:
        """
        Detect how much content matches the current language.

        Args:
            text: Text to analyze

        Returns:
            Score between 0.0 and 1.0 indicating language content ratio
        """
        if not text:
            return 0.0

        # Get shared config for language detection
        shared_config = get_language_shared(self.language)

        # Unicode-based diacritic detection - works for ALL languages
        def has_diacritics(char):
            return unicodedata.normalize("NFD", char) != char

        # Get common words from stopwords config (language-agnostic approach)
        language_words = set(shared_config["stopwords"].get("words", []))

        # Count Unicode diacritics (works for Croatian, French, German, etc.)
        char_score = sum(1 for char in text if has_diacritics(char))
        word_score = sum(1 for word in language_words if word in text.lower().split())

        # Normalize scores (language-agnostic approach)
        total_chars = len(text)
        total_words = len(text.split())

        # Dynamic diacritic expectation based on actual text
        char_ratio = (
            char_score / max(total_chars * 0.05, 1) if char_score > 0 else 0
        )  # 5% threshold
        word_ratio = word_score / max(
            total_words * 0.2, 1
        )  # 20% common words threshold

        # Language-agnostic scoring
        score = (
            min((char_ratio + word_ratio) / 2, 1.0)
            if char_score > 0
            else min(word_ratio, 1.0)
        )

        return score

    def preserve_encoding(self, text: str) -> str:
        """
        Preserve proper text encoding for the language.

        Args:
            text: Text to process

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
                return text.decode("latin-1")

        return str(text)

    def setup_language_environment(self):
        """
        Setup language-specific environment from configuration.
        """
        # Set UTF-8 encoding
        os.environ["PYTHONIOENCODING"] = "utf-8"

        # Get locale from language config
        locale_config = self._language_config.get("locale", {})
        primary_locale = locale_config.get("primary")
        fallback_locale = locale_config.get("fallback", "C.UTF-8")

        # Try to set language-specific locale
        if primary_locale:
            try:
                locale.setlocale(locale.LC_ALL, primary_locale)
            except locale.Error:
                try:
                    locale.setlocale(locale.LC_ALL, fallback_locale)
                except locale.Error:
                    pass  # Use default locale


# Convenience functions for backward compatibility and easier imports
def clean_text(text: str, language: str, preserve_structure: bool = True) -> str:
    """
    Clean and normalize text for the specified language.

    Args:
        text: Raw text to clean
        language: Language code - must be supported in configuration
        preserve_structure: Whether to preserve paragraph structure

    Returns:
        Cleaned text
    """
    cleaner = MultilingualTextCleaner(language=language)
    return cleaner.clean_text(text, preserve_structure=preserve_structure)


def detect_language_content(text: str, language: str) -> float:
    """
    Detect how much content matches the specified language.

    Args:
        text: Text to analyze
        language: Language code - must be supported in configuration

    Returns:
        Score between 0.0 and 1.0 indicating language content ratio
    """
    cleaner = MultilingualTextCleaner(language=language)
    return cleaner.detect_language_content(text)


def preserve_text_encoding(text: str, language: str) -> str:
    """
    Preserve proper text encoding for the specified language.

    Args:
        text: Text to process
        language: Language code - must be supported in configuration

    Returns:
        Text with proper encoding
    """
    cleaner = MultilingualTextCleaner(language=language)
    return cleaner.preserve_encoding(text)


def setup_language_environment(language: str):
    """
    Setup language-specific environment.

    Args:
        language: Language code - must be supported in configuration
    """
    cleaner = MultilingualTextCleaner(language=language)
    return cleaner.setup_language_environment()
