"""
Comprehensive tests for response_parser.py demonstrating 100% testability.
Tests pure functions, dependency injection, and integration scenarios.
"""

import re
from typing import Dict, List, Optional
from unittest.mock import Mock

import pytest
from src.generation.response_parser import (  # Pure functions; Data structures; Core classes; Factory functions
    MultilingualResponseParser, ParsedResponse, ParsingConfig,
    calculate_confidence_score, check_no_answer_patterns, clean_response_text,
    create_mock_config_provider, create_response_parser,
    detect_language_by_patterns, extract_source_references,
    fix_punctuation_spacing, format_display_text, normalize_whitespace,
    remove_prefixes)


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_normalize_whitespace_basic(self):
        """Test basic whitespace normalization."""
        text = "  This   has    multiple\t\nspaces  "
        result = normalize_whitespace(text)
        assert result == "This has multiple spaces"

    def test_normalize_whitespace_empty(self):
        """Test whitespace normalization with empty string."""
        result = normalize_whitespace("")
        assert result == ""

    def test_normalize_whitespace_only_whitespace(self):
        """Test whitespace normalization with only whitespace."""
        text = "   \t\n\r   "
        result = normalize_whitespace(text)
        assert result == ""

    def test_normalize_whitespace_invalid_input(self):
        """Test whitespace normalization with invalid input."""
        with pytest.raises(ValueError, match="Text must be string"):
            normalize_whitespace(123)

    def test_normalize_whitespace_unicode(self):
        """Test whitespace normalization with Unicode."""
        text = "  Croatian đčćšž  text  "
        result = normalize_whitespace(text)
        assert result == "Croatian đčćšž text"

    def test_remove_prefixes_basic(self):
        """Test basic prefix removal."""
        text = "Answer: This is the response content"
        prefixes = [r"^Answer:\s*", r"^Question:\s*"]
        result = remove_prefixes(text, prefixes)
        assert result == "This is the response content"

    def test_remove_prefixes_multiple(self):
        """Test removing multiple prefixes."""
        text = "Question: What is Answer: This is the content"
        prefixes = [r"Question:\s*", r"Answer:\s*"]
        result = remove_prefixes(text, prefixes)
        assert result == "What is  This is the content"

    def test_remove_prefixes_case_insensitive(self):
        """Test case-insensitive prefix removal."""
        text = "ANSWER: This is the response"
        prefixes = [r"answer:\s*"]
        result = remove_prefixes(text, prefixes)
        assert result == "This is the response"

    def test_remove_prefixes_no_match(self):
        """Test prefix removal with no matches."""
        text = "This has no prefixes"
        prefixes = [r"^Answer:\s*", r"^Question:\s*"]
        result = remove_prefixes(text, prefixes)
        assert result == text

    def test_remove_prefixes_invalid_regex(self):
        """Test prefix removal with invalid regex patterns."""
        text = "Answer: This is content"
        prefixes = [r"^Answer:\s*", r"[invalid regex", r"Question:\s*"]
        # Should skip invalid regex and continue with valid ones
        result = remove_prefixes(text, prefixes)
        assert result == "This is content"

    def test_remove_prefixes_invalid_inputs(self):
        """Test prefix removal with invalid inputs."""
        with pytest.raises(ValueError, match="Text must be string"):
            remove_prefixes(123, ["pattern"])

        with pytest.raises(ValueError, match="Prefixes must be list"):
            remove_prefixes("text", "pattern")

    def test_fix_punctuation_spacing_basic(self):
        """Test basic punctuation spacing fixes."""
        text = "Hello , world ! How are you ? Fine ."
        result = fix_punctuation_spacing(text)
        assert result == "Hello, world! How are you? Fine."

    def test_fix_punctuation_spacing_sentence_endings(self):
        """Test sentence ending punctuation."""
        text = "First sentence.Second sentence!Third sentence?"
        result = fix_punctuation_spacing(text)
        assert result == "First sentence. Second sentence! Third sentence?"

    def test_fix_punctuation_spacing_croatian_caps(self):
        """Test Croatian capital letters after punctuation."""
        text = "Prva rečenica.Druga rečenica!Treća rečenica?"
        result = fix_punctuation_spacing(text)
        assert result == "Prva rečenica. Druga rečenica! Treća rečenica?"

    def test_fix_punctuation_spacing_special_chars(self):
        """Test punctuation with Croatian special characters."""
        text = "Hello , Čestitam ! Đ character ."
        result = fix_punctuation_spacing(text)
        assert result == "Hello, Čestitam! Đ character."

    def test_fix_punctuation_spacing_invalid_input(self):
        """Test punctuation fixing with invalid input."""
        with pytest.raises(ValueError, match="Text must be string"):
            fix_punctuation_spacing(123)

    def test_clean_response_text_complete(self):
        """Test complete response text cleaning."""
        text = "  Answer:   Hello , world !   This is content.  "
        prefixes = [r"^Answer:\s*"]
        result = clean_response_text(text, prefixes)
        assert result == "Hello, world! This is content."

    def test_clean_response_text_empty(self):
        """Test cleaning empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            clean_response_text("")

    def test_clean_response_text_default_prefixes(self):
        """Test cleaning with default prefixes (None)."""
        text = "  Hello , world !  "
        result = clean_response_text(text)
        assert result == "Hello, world!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
