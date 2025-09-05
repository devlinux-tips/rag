"""
Multilingual text cleaning and normalization utilities.
Handles language-specific text processing challenges including diacritics,
morphology, and document formatting artifacts for Croatian and English.
"""

import locale
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config_loader import (
    get_cleaning_config,
    get_language_shared,
    get_language_specific_config,
)
from ..utils.error_handler import handle_config_error

logger = logging.getLogger(__name__)


class MultilingualTextCleaner:
    """Clean and normalize multilingual text for RAG processing."""

    def __init__(self, language: str = "hr"):
        """
        Initialize the multilingual text cleaner.

        Args:
            language: Language code ('hr' for Croatian, 'en' for English)
        """
        self.language = language

        self._language_config = handle_config_error(
            operation=lambda: get_language_specific_config("text_processing", self.language),
            fallback_value={
                "diacritic_map": (
                    {"ć": "c", "č": "c", "š": "s", "ž": "z", "đ": "d"} if language == "hr" else {}
                ),
                "preserve_diacritics": True if language == "hr" else False,
            },
            config_file=f"config/{self.language}.toml",
            section="[text_processing]",
        )
        self._cleaning_config = handle_config_error(
            operation=lambda: get_cleaning_config(),
            fallback_value={"remove_extra_whitespace": True, "normalize_unicode": True},
            config_file="config/config.toml",
            section="[cleaning]",
        )
        self._document_cleaning_config = handle_config_error(
            operation=lambda: get_language_specific_config("document_cleaning", self.language),
            fallback_value={"remove_headers": True, "remove_footers": True},
            config_file=f"config/{self.language}.toml",
            section="[document_cleaning]",
        )
        self._chunking_config = handle_config_error(
            operation=lambda: get_language_specific_config("chunking", self.language),
            fallback_value={"sentence_endings": [".", "!", "?"], "preserve_paragraphs": True},
            config_file=f"config/{self.language}.toml",
            section="[chunking]",
        )

        # Language-specific diacritic mappings for normalization (optional)
        self.diacritic_map = self._language_config.get("diacritic_map", {})

        # Language-specific stopwords (from shared config)
        shared_config = handle_config_error(
            operation=lambda: get_language_shared(self.language),
            fallback_value={
                "stopwords": {
                    "words": (
                        ["i", "u", "na", "za", "se", "je"]
                        if language == "hr"
                        else ["the", "a", "an", "and", "or"]
                    )
                }
            },
            config_file=f"config/{self.language}.toml",
            section="[shared]",
        )
        self.stopwords = set(shared_config["stopwords"]["words"])

        # Document formatting artifacts to remove
        char_pattern_key = "croatian_chars_pattern" if language == "hr" else "english_chars_pattern"
        char_pattern = shared_config.get(char_pattern_key, r"[^\\w\\s.,!?:;()-]")

        self.formatting_patterns = [
            self._cleaning_config.get("multiple_whitespace", r"\s+"),  # Multiple whitespaces
            self._cleaning_config.get("multiple_linebreaks", r"\n\s*\n"),  # Multiple line breaks
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
        # Common OCR mistakes for the current language
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

        # Check if text is mostly numbers or special characters
        word_chars = sum(len(re.findall(r"[a-zA-ZčćžšđČĆŽŠĐ]", word)) for word in words)
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

        # Get shared config
        shared_config = handle_config_error(
            operation=lambda: get_language_shared(self.language),
            fallback_value={
                "stopwords": {
                    "words": (
                        ["i", "u", "na", "za", "se", "je"]
                        if self.language == "hr"
                        else ["the", "a", "an", "and", "or"]
                    )
                }
            },
            config_file=f"config/{self.language}.toml",
            section="[shared]",
        )

        # Get language-specific characters and words from config
        if self.language == "hr":
            # Croatian diacritics
            language_chars = set("čćšžđČĆŠŽĐ")

            # Croatian common words from config or fallback
            language_words = {
                "je",
                "se",
                "na",
                "za",
                "da",
                "su",
                "ili",
                "ako",
                "kad",
                "što",
                "biti",
                "imati",
                "moći",
                "htjeti",
                "trebati",
                "doći",
                "vidjeti",
                "zagreb",
                "hrvatska",
                "dubrovnik",
                "split",
                "rijeka",
                "grad",
                "glavni",
                "veliki",
                "lijep",
                "važan",
                "poznaj",
                "hrvatski",
            }
        else:  # English or other languages
            # English typically doesn't have special diacritics
            language_chars = set()

            # English common words from config
            language_words = set(shared_config.get("stopwords", {}).get("words", []))

        # Count language indicators
        char_score = sum(1 for char in text if char in language_chars)
        word_score = sum(1 for word in language_words if word in text.lower().split())

        # Normalize scores
        total_chars = len(text)
        total_words = len(text.split())

        if self.language == "hr":
            char_ratio = char_score / max(total_chars * 0.02, 1)  # Expected 2% diacritics
            word_ratio = word_score / max(total_words * 0.1, 1)  # Expected 10% common words
        else:
            # For English, rely more on common words
            char_ratio = 0
            word_ratio = word_score / max(total_words * 0.3, 1)  # Expected 30% common words

        # Combine scores
        score = (
            min((char_ratio + word_ratio) / 2, 1.0)
            if self.language == "hr"
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
        Setup language-specific environment.
        This function ensures proper locale and encoding settings.
        """
        # Set UTF-8 encoding
        os.environ["PYTHONIOENCODING"] = "utf-8"

        # Try to set language-specific locale
        if self.language == "hr":
            try:
                locale.setlocale(locale.LC_ALL, "hr_HR.UTF-8")
            except locale.Error:
                try:
                    locale.setlocale(locale.LC_ALL, "C.UTF-8")
                except locale.Error:
                    pass  # Use default locale
        elif self.language == "en":
            try:
                locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
            except locale.Error:
                try:
                    locale.setlocale(locale.LC_ALL, "C.UTF-8")
                except locale.Error:
                    pass  # Use default locale


# Convenience functions for backward compatibility and easier imports
def clean_text(text: str, language: str = "hr", preserve_structure: bool = True) -> str:
    """
    Clean and normalize text for the specified language.

    Args:
        text: Raw text to clean
        language: Language code ('hr' for Croatian, 'en' for English)
        preserve_structure: Whether to preserve paragraph structure

    Returns:
        Cleaned text
    """
    cleaner = MultilingualTextCleaner(language=language)
    return cleaner.clean_text(text, preserve_structure=preserve_structure)


def detect_language_content(text: str, language: str = "hr") -> float:
    """
    Detect how much content matches the specified language.

    Args:
        text: Text to analyze
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Score between 0.0 and 1.0 indicating language content ratio
    """
    cleaner = MultilingualTextCleaner(language=language)
    return cleaner.detect_language_content(text)


def preserve_text_encoding(text: str, language: str = "hr") -> str:
    """
    Preserve proper text encoding for the specified language.

    Args:
        text: Text to process
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Text with proper encoding
    """
    cleaner = MultilingualTextCleaner(language=language)
    return cleaner.preserve_encoding(text)


def setup_language_environment(language: str = "hr"):
    """
    Setup language-specific environment.

    Args:
        language: Language code ('hr' for Croatian, 'en' for English)
    """
    cleaner = MultilingualTextCleaner(language=language)
    return cleaner.setup_language_environment()
