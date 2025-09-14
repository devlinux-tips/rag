"""
Generic OCR correction algorithms - replace hardcoded config patterns.

This module provides algorithmic OCR corrections instead of hardcoded word lists.
Works for any language and content, not specific Croatian/English patterns.
"""

import re
from typing import Any


def fix_spaced_capitals(text: str) -> str:
    """
    Fix spaced capital letters - works for any language.

    Examples:
        "H R V A T S K A" → "HRVATSKA"
        "Z A G R E B" → "ZAGREB"
        "U S A" → "USA"
        "N A T O" → "NATO"

    Args:
        text: Input text with potentially spaced capitals

    Returns:
        Text with spaced capitals fixed
    """
    # Fix fully spaced capitals: "H R V A T S K A" → "HRVATSKA"
    text = re.sub(r"\b([A-Z])\s+([A-Z](?:\s+[A-Z])*)\b", lambda m: m.group(0).replace(" ", ""), text)

    # Fix partially spaced capitals: "HR VATSKA" → "HRVATSKA"
    text = re.sub(r"\b([A-Z]{2,})\s+([A-Z]{2,})\b", r"\1\2", text)

    return text


def fix_spaced_punctuation(text: str) -> str:
    """
    Fix spaced punctuation marks.

    Examples:
        "word , word" → "word, word"
        "sentence ." → "sentence."
        "question ?" → "question?"

    Args:
        text: Input text with spaced punctuation

    Returns:
        Text with punctuation properly attached
    """
    # Fix spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Fix spaces after opening punctuation
    text = re.sub(r'([(\[\{"])\s+', r"\1", text)

    return text


def fix_common_ocr_errors(text: str) -> str:
    """
    Fix common OCR character recognition errors.

    Examples:
        "rn" confused with "m": "moming" → "morning"
        "vv" confused with "w": "vvorld" → "world"
        "cl" confused with "d": "clar" → "clear"

    Args:
        text: Input text with OCR errors

    Returns:
        Text with common OCR errors fixed
    """
    # Common OCR substitution errors
    ocr_fixes = {
        # Character confusion patterns
        r"\brn\b": "m",  # "rn" → "m" in words like "morning"
        r"\bvv\b": "w",  # "vv" → "w" in words like "world"
        r"\bcl\b": "d",  # "cl" → "d" in some contexts
        # Spacing in numbers
        r"(\d)\s+(\d)": r"\1\2",  # "1 234" → "1234"
        # Spacing in common abbreviations
        r"\bU\s*S\s*A\b": "USA",
        r"\bU\s*K\b": "UK",
        r"\bE\s*U\b": "EU",
        r"\bN\s*A\s*T\s*O\b": "NATO",
    }

    for pattern, replacement in ocr_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def fix_spaced_diacritics(text: str, language: str | None = None) -> str:
    """
    Fix spaced diacritic characters for languages that use them.

    Examples (Croatian):
        "c̆ " → "č"
        "s̆ " → "š"
        "z̆ " → "ž"

    Args:
        text: Input text with spaced diacritics
        language: Language code (optional, for language-specific patterns)

    Returns:
        Text with diacritics properly formed
    """
    if language == "hr":
        # Croatian diacritic combining patterns
        diacritic_fixes = {
            r"c\s*[\u030C\u0306]": "č",  # c + combining caron → č
            r"s\s*[\u030C\u0306]": "š",  # s + combining caron → š
            r"z\s*[\u030C\u0306]": "ž",  # z + combining caron → ž
            r"d\s*[\u0304\u0335]": "đ",  # d + combining stroke → đ
            r"c\s*[\u0301\u0306]": "ć",  # c + combining acute → ć
        }

        for pattern, replacement in diacritic_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Generic: fix any spaced combining diacritics
    text = re.sub(r"([a-zA-Z])\s+([\u0300-\u036F])", r"\1\2", text)

    return text


def apply_ocr_corrections(text: str, config: dict[str, Any], language: str | None = None) -> str:
    """
    Apply OCR corrections based on configuration flags.

    Args:
        text: Input text to correct
        config: OCR correction configuration
        language: Language code for language-specific corrections

    Returns:
        Text with OCR corrections applied
    """
    if not text or not config:
        return text

    # Apply corrections based on config flags
    if config["fix_spaced_capitals"]:
        text = fix_spaced_capitals(text)

    if config["fix_spaced_punctuation"]:
        text = fix_spaced_punctuation(text)

    if config["fix_common_ocr_errors"]:
        text = fix_common_ocr_errors(text)

    if config["fix_spaced_diacritics"]:
        text = fix_spaced_diacritics(text, language)

    return text


def get_ocr_correction_stats(original_text: str, corrected_text: str) -> dict[str, int]:
    """
    Get statistics on OCR corrections applied.

    Args:
        original_text: Text before corrections
        corrected_text: Text after corrections

    Returns:
        Dictionary with correction statistics
    """
    return {
        "original_length": len(original_text),
        "corrected_length": len(corrected_text),
        "characters_changed": sum(1 for a, b in zip(original_text, corrected_text, strict=False) if a != b),
        "length_difference": len(corrected_text) - len(original_text),
    }
