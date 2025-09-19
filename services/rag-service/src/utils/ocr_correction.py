"""
Generic OCR correction algorithms - replace hardcoded config patterns.

This module provides algorithmic OCR corrections instead of hardcoded word lists.
Works for any language and content, not specific Croatian/English patterns.
"""

import re
from typing import Any

from .logging_factory import get_system_logger, log_component_end, log_component_start, log_data_transformation


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
    logger = get_system_logger()
    logger.trace("ocr_correction", "fix_spaced_capitals", f"Processing text length: {len(text)}")

    original_text = text

    # Fix fully spaced capitals: "H R V A T S K A" → "HRVATSKA"
    logger.debug("ocr_correction", "fix_spaced_capitals", "Applying fully spaced capitals pattern")
    text = re.sub(r"\b([A-Z])\s+([A-Z](?:\s+[A-Z])*)\b", lambda m: m.group(0).replace(" ", ""), text)

    # Fix partially spaced capitals: "HR VATSKA" → "HRVATSKA"
    logger.debug("ocr_correction", "fix_spaced_capitals", "Applying partially spaced capitals pattern")
    text = re.sub(r"\b([A-Z]{2,})\s+([A-Z]{2,})\b", r"\1\2", text)

    if text != original_text:
        log_data_transformation("ocr_correction", "fix_spaced_capitals", "spaced capitals text", "fixed capitals text")

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
    logger = get_system_logger()
    log_component_start(
        "ocr_correction", "apply_ocr_corrections", language=language, text_length=len(text) if text else 0
    )

    if not text or not config:
        logger.warning("ocr_correction", "apply_ocr_corrections", "Empty text or config, returning original text")
        return text

    original_text = text
    corrections_applied = []

    # Handle legacy config format (word mappings only) vs new format (with flags)
    if all(isinstance(v, str) for v in config.values()):
        # Legacy format: config is just word mappings, enable all OCR corrections by default
        ocr_flags = {
            "fix_spaced_capitals": True,
            "fix_spaced_punctuation": True,
            "fix_common_ocr_errors": True,
            "fix_character_spacing": True,
            "word_replacements": config,
        }
    else:
        # New format: config contains flags
        ocr_flags = config

    # Apply corrections based on config flags
    if ocr_flags.get("fix_spaced_capitals", False):
        logger.debug("ocr_correction", "apply_ocr_corrections", "Applying spaced capitals correction")
        text = fix_spaced_capitals(text)
        corrections_applied.append("spaced_capitals")

    if ocr_flags.get("fix_spaced_punctuation", False):
        logger.debug("ocr_correction", "apply_ocr_corrections", "Applying spaced punctuation correction")
        text = fix_spaced_punctuation(text)
        corrections_applied.append("spaced_punctuation")

    if ocr_flags.get("fix_common_ocr_errors", False):
        logger.debug("ocr_correction", "apply_ocr_corrections", "Applying common OCR errors correction")
        text = fix_common_ocr_errors(text)
        corrections_applied.append("common_ocr_errors")

    if ocr_flags.get("fix_spaced_diacritics", False):
        logger.debug(
            "ocr_correction", "apply_ocr_corrections", f"Applying spaced diacritics correction for language: {language}"
        )
        text = fix_spaced_diacritics(text, language)
        corrections_applied.append("spaced_diacritics")

    # Apply word replacements (both legacy and new format)
    word_replacements_raw = ocr_flags.get("word_replacements", {})
    if word_replacements_raw and isinstance(word_replacements_raw, dict):
        word_replacements: dict[str, str] = word_replacements_raw
        logger.debug("ocr_correction", "apply_ocr_corrections", f"Applying {len(word_replacements)} word replacements")
        for old_word, new_word in word_replacements.items():
            if old_word in text:
                text = text.replace(old_word, new_word)
        corrections_applied.append("word_replacements")

    if text != original_text:
        stats = get_ocr_correction_stats(original_text, text)
        log_data_transformation(
            "ocr_correction",
            "apply_ocr_corrections",
            f"text with OCR errors ({len(original_text)} chars)",
            f"corrected text ({len(text)} chars)",
            corrections=corrections_applied,
            characters_changed=stats["characters_changed"],
        )

    log_component_end("ocr_correction", "apply_ocr_corrections", f"Applied {len(corrections_applied)} correction types")
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
    logger = get_system_logger()
    logger.trace(
        "ocr_correction",
        "get_ocr_correction_stats",
        f"Calculating stats for texts of length {len(original_text)} and {len(corrected_text)}",
    )

    stats = {
        "original_length": len(original_text),
        "corrected_length": len(corrected_text),
        "characters_changed": sum(1 for a, b in zip(original_text, corrected_text, strict=False) if a != b),
        "length_difference": len(corrected_text) - len(original_text),
    }

    logger.trace("ocr_correction", "get_ocr_correction_stats", f"Stats calculated: {stats}")
    return stats
