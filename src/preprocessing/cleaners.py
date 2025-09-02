"""
Croatianfrom ..utils.error_handler import handle_config_errort cleaning and normalization utilities.
Handles Croatian-specific text processing challenges including diacritics,
morphology, and document formatting artifacts.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config_loader import (
    get_cleaning_config,
    get_croatian_chunking,
    get_croatian_document_cleaning,
    get_croatian_shared,
    get_croatian_text_processing,
)
from ..utils.error_handler import handle_config_error

logger = logging.getLogger(__name__)


class CroatianTextCleaner:
    """Clean and normalize Croatian text for RAG processing."""

    def __init__(self):
        """Initialize the Croatian text cleaner."""
        self._croatian_config = handle_config_error(
            operation=lambda: get_croatian_text_processing(),
            fallback_value={
                "diacritic_map": {"ć": "c", "č": "c", "š": "s", "ž": "z", "đ": "d"},
                "preserve_diacritics": True,
            },
            config_file="config/croatian.toml",
            section="[text_processing]",
        )
        self._cleaning_config = handle_config_error(
            operation=lambda: get_cleaning_config(),
            fallback_value={"remove_extra_whitespace": True, "normalize_unicode": True},
            config_file="config/config.toml",
            section="[cleaning]",
        )
        self._document_cleaning_config = handle_config_error(
            operation=lambda: get_croatian_document_cleaning(),
            fallback_value={"remove_headers": True, "remove_footers": True},
            config_file="config/croatian.toml",
            section="[document_cleaning]",
        )
        self._chunking_config = handle_config_error(
            operation=lambda: get_croatian_chunking(),
            fallback_value={"sentence_endings": [".", "!", "?"], "preserve_paragraphs": True},
            config_file="config/croatian.toml",
            section="[chunking]",
        )

        # Croatian diacritic mappings for normalization (optional)
        self.diacritic_map = self._croatian_config["diacritic_map"]

        # Common Croatian stopwords (from shared config)
        shared_config = handle_config_error(
            operation=lambda: get_croatian_shared(),
            fallback_value={"stopwords": {"words": ["i", "u", "na", "za", "se", "je"]}},
            config_file="config/croatian.toml",
            section="[shared]",
        )
        self.stopwords = set(shared_config["stopwords"]["words"])

        # Document formatting artifacts to remove
        self.formatting_patterns = [
            self._cleaning_config.get("multiple_whitespace", r"\s+"),  # Multiple whitespaces
            self._cleaning_config.get("multiple_linebreaks", r"\n\s*\n"),  # Multiple line breaks
            self._croatian_config[
                "croatian_chars_pattern"
            ],  # Non-standard chars (preserve Croatian)
            r"^\s*\d+\s*$",  # Standalone page numbers
            r"^\s*[IVX]+\s*$",  # Roman numerals
            r"^\s*[a-z]\)\s*$",  # List markers like a), b)
        ]

    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        Clean and normalize Croatian text.

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

        # Fix common OCR errors in Croatian text
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
        """Fix common OCR errors in Croatian text."""
        # Common OCR mistakes in Croatian
        ocr_fixes = self._document_cleaning_config["ocr_corrections"]

        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from Croatian text.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Croatian sentence ending patterns from config
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
        Normalize Croatian diacritics (optional - use carefully as it loses information).

        Args:
            text: Text with Croatian diacritics

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


def clean_croatian_text(text: str, preserve_structure: bool = True) -> str:
    """
    Convenience function to clean Croatian text.

    Args:
        text: Raw text to clean
        preserve_structure: Whether to preserve paragraph structure

    Returns:
        Cleaned text
    """
    cleaner = CroatianTextCleaner()
    return cleaner.clean_text(text, preserve_structure)
