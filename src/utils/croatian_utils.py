"""
Croatian language utilities for text processing.
"""

import re
from typing import List


class CroatianTextProcessor:
    """Utilities for processing Croatian text."""

    CROATIAN_CHARS = "ČčĆćŠšŽžĐđ"

    def __init__(self):
        self.diacritic_map = {
            "č": "c",
            "ć": "c",
            "š": "s",
            "ž": "z",
            "đ": "d",
            "Č": "C",
            "Ć": "C",
            "Š": "S",
            "Ž": "Z",
            "Đ": "D",
        }

    def normalize_text(self, text: str) -> str:
        """Normalize Croatian text while preserving diacritics."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def has_croatian_chars(self, text: str) -> bool:
        """Check if text contains Croatian-specific characters."""
        return any(char in text for char in self.CROATIAN_CHARS)

    def remove_diacritics(self, text: str) -> str:
        """Remove Croatian diacritics (use sparingly for search)."""
        for croatian, latin in self.diacritic_map.items():
            text = text.replace(croatian, latin)
        return text


def preserve_croatian_encoding(text: str) -> str:
    """
    Ensure Croatian text encoding is preserved.

    Args:
        text: Text that might have encoding issues

    Returns:
        Text with proper Croatian encoding
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


def detect_croatian_content(text: str) -> float:
    """
    Detect how much Croatian content is in the text.

    Args:
        text: Text to analyze

    Returns:
        Score between 0.0 and 1.0 indicating Croatian content ratio
    """
    if not text:
        return 0.0

    # Croatian diacritics
    croatian_chars = set("čćšžđČĆŠŽĐ")

    # Croatian common words
    croatian_words = {
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

    # Count Croatian indicators
    char_score = sum(1 for char in text if char in croatian_chars)
    word_score = sum(1 for word in croatian_words if word in text.lower().split())

    # Normalize scores
    total_chars = len(text)
    total_words = len(text.split())

    char_ratio = char_score / max(total_chars * 0.02, 1)  # Expected 2% diacritics
    word_ratio = word_score / max(total_words * 0.1, 1)  # Expected 10% Croatian words

    # Combine scores
    score = min((char_ratio + word_ratio) / 2, 1.0)

    return score


def setup_croatian_environment():
    """
    Setup Croatian language environment.
    This function ensures proper locale and encoding settings.
    """
    import locale
    import os

    # Set UTF-8 encoding
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Try to set Croatian locale
    try:
        locale.setlocale(locale.LC_ALL, "hr_HR.UTF-8")
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, "C.UTF-8")
        except locale.Error:
            pass  # Use default locale


# Create default instance for module-level functions
_processor = CroatianTextProcessor()


def normalize_croatian_text(text: str) -> str:
    """Normalize Croatian text (module-level function)."""
    return _processor.normalize_text(text)


def has_croatian_content_old(text: str) -> bool:
    """Check if text has Croatian content (module-level function)."""
    return _processor.has_croatian_chars(text)
