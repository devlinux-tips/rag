"""
Comprehensive tests for generation/response_parser.py

Tests all response parsing functionality including:
- Pure functions for text cleaning and analysis
- Data structures (ParsedResponse, ParsingConfig)
- Protocol implementation
- MultilingualResponseParser class
- Factory functions
"""

import unittest
from unittest.mock import MagicMock, patch

from src.generation.response_parser import (
    ConfigProvider,
    MultilingualResponseParser,
    ParsedResponse,
    ParsingConfig,
    calculate_confidence_score,
    check_no_answer_patterns,
    clean_response_text,
    create_mock_config_provider,
    create_response_parser,
    detect_language_by_patterns,
    extract_source_references,
    fix_punctuation_spacing,
    format_display_text,
    normalize_whitespace,
    remove_prefixes,
)


class TestPureFunctions(unittest.TestCase):
    """Test pure functions for text processing and analysis."""

    def test_normalize_whitespace_valid(self):
        """Test normalizing valid whitespace."""
        self.assertEqual(normalize_whitespace("  Hello   world  "), "Hello world")
        self.assertEqual(normalize_whitespace("Text\n\twith\r\nvarious   spaces"), "Text with various spaces")
        self.assertEqual(normalize_whitespace("Normal text"), "Normal text")
        self.assertEqual(normalize_whitespace(""), "")

    def test_normalize_whitespace_invalid_type(self):
        """Test normalizing non-string input."""
        with self.assertRaises(ValueError) as cm:
            normalize_whitespace(123)
        self.assertIn("Text must be string", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            normalize_whitespace(None)
        self.assertIn("Text must be string", str(cm.exception))

    def test_remove_prefixes_valid(self):
        """Test removing valid prefixes."""
        text = "Answer: This is the answer"
        prefixes = [r"^Answer:\s*", r"^Question:\s*"]
        result = remove_prefixes(text, prefixes)
        self.assertEqual(result, "This is the answer")

    def test_remove_prefixes_no_match(self):
        """Test removing prefixes with no matches."""
        text = "Regular text"
        prefixes = [r"^Answer:\s*", r"^Question:\s*"]
        result = remove_prefixes(text, prefixes)
        self.assertEqual(result, "Regular text")

    def test_remove_prefixes_multiple_patterns(self):
        """Test removing multiple matching prefixes."""
        text = "Question: What is AI? Answer: AI is artificial intelligence"
        prefixes = [r"Question:\s*", r"Answer:\s*"]
        result = remove_prefixes(text, prefixes)
        self.assertEqual(result, "What is AI? AI is artificial intelligence")

    def test_remove_prefixes_invalid_inputs(self):
        """Test removing prefixes with invalid inputs."""
        # Invalid text type
        with self.assertRaises(ValueError) as cm:
            remove_prefixes(123, ["pattern"])
        self.assertIn("Text must be string", str(cm.exception))

        # Invalid prefixes type
        with self.assertRaises(ValueError) as cm:
            remove_prefixes("text", "not a list")
        self.assertIn("Prefixes must be list", str(cm.exception))

    def test_remove_prefixes_invalid_patterns(self):
        """Test removing prefixes with invalid regex patterns."""
        text = "Some text"
        prefixes = ["valid_pattern", 123, "[invalid_regex"]  # Mixed valid/invalid
        # Should not raise error, just skip invalid patterns
        result = remove_prefixes(text, prefixes)
        self.assertIsInstance(result, str)

    def test_fix_punctuation_spacing_valid(self):
        """Test fixing valid punctuation spacing."""
        text = "Hello , world ! How are you ?"
        result = fix_punctuation_spacing(text)
        self.assertEqual(result, "Hello, world! How are you?")

    def test_fix_punctuation_spacing_sentence_endings(self):
        """Test fixing sentence-ending punctuation."""
        text = "First sentence.Second sentence!Third sentence?"
        result = fix_punctuation_spacing(text)
        self.assertEqual(result, "First sentence. Second sentence! Third sentence?")

    def test_fix_punctuation_spacing_croatian_characters(self):
        """Test fixing punctuation with Croatian characters."""
        text = "Zagreb.Čakovec!Šibenik?"
        result = fix_punctuation_spacing(text)
        self.assertEqual(result, "Zagreb. Čakovec! Šibenik?")

    def test_fix_punctuation_spacing_invalid_type(self):
        """Test fixing punctuation with invalid input."""
        with self.assertRaises(ValueError) as cm:
            fix_punctuation_spacing(123)
        self.assertIn("Text must be string", str(cm.exception))

    def test_clean_response_text_valid(self):
        """Test cleaning valid response text."""
        text = "Answer:  Hello   world ! This is clean text ."
        prefixes = [r"^Answer:\s*"]
        result = clean_response_text(text, prefixes)
        self.assertEqual(result, "Hello world! This is clean text.")

    def test_clean_response_text_empty(self):
        """Test cleaning empty text."""
        with self.assertRaises(ValueError) as cm:
            clean_response_text("")
        self.assertIn("Text cannot be empty", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            clean_response_text(None)
        self.assertIn("Text cannot be empty", str(cm.exception))

    def test_clean_response_text_no_prefixes(self):
        """Test cleaning text without prefixes."""
        text = "  Regular text with spaces  "
        result = clean_response_text(text)
        self.assertEqual(result, "Regular text with spaces")

    def test_check_no_answer_patterns_positive(self):
        """Test checking text that contains no-answer patterns."""
        text = "Ne znam odgovor na to pitanje"
        patterns = ["ne znam", "ne mogu", "nema podataka"]
        result = check_no_answer_patterns(text, patterns)
        self.assertTrue(result)

    def test_check_no_answer_patterns_negative(self):
        """Test checking text that doesn't contain no-answer patterns."""
        text = "Ovo je odgovor na pitanje"
        patterns = ["ne znam", "ne mogu", "nema podataka"]
        result = check_no_answer_patterns(text, patterns)
        self.assertFalse(result)

    def test_check_no_answer_patterns_case_insensitive(self):
        """Test no-answer pattern matching is case insensitive."""
        text = "NE ZNAM odgovor"
        patterns = ["ne znam"]
        result = check_no_answer_patterns(text, patterns)
        self.assertTrue(result)

    def test_check_no_answer_patterns_invalid_inputs(self):
        """Test checking no-answer patterns with invalid inputs."""
        # Invalid text type
        with self.assertRaises(ValueError) as cm:
            check_no_answer_patterns(123, ["pattern"])
        self.assertIn("Text must be string", str(cm.exception))

        # Invalid patterns type
        with self.assertRaises(ValueError) as cm:
            check_no_answer_patterns("text", "not a list")
        self.assertIn("No answer patterns must be list", str(cm.exception))

    def test_check_no_answer_patterns_invalid_regex(self):
        """Test no-answer patterns with invalid regex."""
        text = "Some text"
        patterns = ["valid", "[invalid_regex", 123]  # Mixed valid/invalid
        # Should not raise error, just skip invalid patterns
        result = check_no_answer_patterns(text, patterns)
        self.assertIsInstance(result, bool)

    def test_extract_source_references_found(self):
        """Test extracting source references that are found."""
        text = "According to source: document1.pdf and source: document2.txt"
        patterns = [r"source:\s*(\w+\.\w+)"]
        result = extract_source_references(text, patterns)
        # Note: this captures the entire match, not just the group
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_extract_source_references_not_found(self):
        """Test extracting source references when none found."""
        text = "Regular text without sources"
        patterns = [r"source:\s*(\w+\.\w+)"]
        result = extract_source_references(text, patterns)
        self.assertEqual(result, [])

    def test_extract_source_references_duplicates(self):
        """Test extracting source references with duplicates."""
        text = "Source: doc1.pdf and source: doc1.pdf again"
        patterns = [r"source:\s*\w+\.\w+"]
        result = extract_source_references(text, patterns)
        # Should remove duplicates
        unique_sources = list(dict.fromkeys(result))
        self.assertEqual(len(result), len(unique_sources))

    def test_extract_source_references_invalid_inputs(self):
        """Test extracting sources with invalid inputs."""
        # Invalid text type
        with self.assertRaises(ValueError) as cm:
            extract_source_references(123, ["pattern"])
        self.assertIn("Text must be string", str(cm.exception))

        # Invalid patterns type
        with self.assertRaises(ValueError) as cm:
            extract_source_references("text", "not a list")
        self.assertIn("Source patterns must be list", str(cm.exception))

    def test_extract_source_references_invalid_regex(self):
        """Test extracting sources with invalid regex patterns."""
        text = "Some text"
        patterns = ["valid", "[invalid_regex", 123]  # Mixed valid/invalid
        # Should not raise error, just skip invalid patterns
        result = extract_source_references(text, patterns)
        self.assertIsInstance(result, list)

    def test_calculate_confidence_score_high(self):
        """Test calculating high confidence score."""
        text = "I am certain that this is correct and definitive"
        indicators = {
            "high": ["certain", "definitive"],
            "medium": ["likely", "probably"],
            "low": ["uncertain", "maybe"]
        }
        result = calculate_confidence_score(text, indicators)
        self.assertGreater(result, 0.8)

    def test_calculate_confidence_score_medium(self):
        """Test calculating medium confidence score."""
        text = "This is probably correct and likely true"
        indicators = {
            "high": ["certain", "definitive"],
            "medium": ["probably", "likely"],
            "low": ["uncertain", "maybe"]
        }
        result = calculate_confidence_score(text, indicators)
        self.assertGreaterEqual(result, 0.5)
        self.assertLess(result, 0.8)

    def test_calculate_confidence_score_low(self):
        """Test calculating low confidence score."""
        text = "I am uncertain and maybe this is correct"
        indicators = {
            "high": ["certain", "definitive"],
            "medium": ["probably", "likely"],
            "low": ["uncertain", "maybe"]
        }
        result = calculate_confidence_score(text, indicators)
        self.assertLess(result, 0.5)

    def test_calculate_confidence_score_neutral(self):
        """Test calculating neutral confidence score."""
        text = "This is neutral text without indicators"
        indicators = {
            "high": ["certain", "definitive"],
            "medium": ["probably", "likely"],
            "low": ["uncertain", "maybe"]
        }
        result = calculate_confidence_score(text, indicators)
        self.assertEqual(result, 0.5)  # Neutral confidence

    def test_calculate_confidence_score_invalid_inputs(self):
        """Test calculating confidence with invalid inputs."""
        # Invalid text type
        with self.assertRaises(ValueError) as cm:
            calculate_confidence_score(123, {})
        self.assertIn("Text must be string", str(cm.exception))

        # Invalid indicators type
        with self.assertRaises(ValueError) as cm:
            calculate_confidence_score("text", "not a dict")
        self.assertIn("Confidence indicators must be dict", str(cm.exception))

        # Missing required keys
        incomplete_indicators = {"high": ["certain"], "medium": ["probably"]}  # Missing "low"
        with self.assertRaises(ValueError) as cm:
            calculate_confidence_score("text", incomplete_indicators)
        self.assertIn("Missing 'low' in confidence indicators", str(cm.exception))

    def test_detect_language_by_patterns_croatian(self):
        """Test detecting Croatian language."""
        text = "Ovo je hrvatski tekst sa dijakritičnim znakovima č ć š ž đ"
        patterns = {
            "hr": ["je", "sa", "hrvatski"],
            "en": ["is", "with", "english"]
        }
        result = detect_language_by_patterns(text, patterns)
        self.assertEqual(result, "hr")

    def test_detect_language_by_patterns_english(self):
        """Test detecting English language."""
        text = "This is english text with common words"
        patterns = {
            "hr": ["je", "sa", "hrvatski"],
            "en": ["is", "with", "english"]
        }
        result = detect_language_by_patterns(text, patterns)
        self.assertEqual(result, "en")

    def test_detect_language_by_patterns_default(self):
        """Test language detection falling back to default."""
        text = "Texto en español sin patrones"
        patterns = {
            "hr": ["je", "sa", "hrvatski"],
            "en": ["is", "with", "english"]
        }
        result = detect_language_by_patterns(text, patterns, default_language="es")
        self.assertEqual(result, "es")

    def test_detect_language_by_patterns_diacritics(self):
        """Test language detection considering diacritics."""
        text = "čćšžđ"  # Croatian diacritics
        patterns = {
            "hr": [],  # No word patterns but has diacritics
            "en": []
        }
        result = detect_language_by_patterns(text, patterns, default_language="unknown")
        # Should detect due to diacritics weight
        self.assertIn(result, ["hr", "unknown"])  # Depends on threshold

    def test_detect_language_by_patterns_invalid_inputs(self):
        """Test language detection with invalid inputs."""
        # Invalid text type
        with self.assertRaises(ValueError) as cm:
            detect_language_by_patterns(123, {})
        self.assertIn("Text must be string", str(cm.exception))

        # Invalid patterns type
        with self.assertRaises(ValueError) as cm:
            detect_language_by_patterns("text", "not a dict")
        self.assertIn("Language patterns must be dict", str(cm.exception))

    def test_format_display_text_content_only(self):
        """Test formatting display text with content only."""
        result = format_display_text("This is the main content")
        self.assertEqual(result, "This is the main content")

    def test_format_display_text_with_confidence(self):
        """Test formatting display text with confidence."""
        result = format_display_text("Content", confidence=0.9)
        self.assertIn("Content", result)
        self.assertIn("High Confidence", result)

        result = format_display_text("Content", confidence=0.6)
        self.assertIn("Medium Confidence", result)

        result = format_display_text("Content", confidence=0.3)
        self.assertIn("Low Confidence", result)

    def test_format_display_text_with_sources(self):
        """Test formatting display text with sources."""
        sources = ["doc1.pdf", "doc2.txt"]
        result = format_display_text("Content", sources=sources)
        self.assertIn("Content", result)
        self.assertIn("Sources:", result)
        self.assertIn("doc1.pdf", result)
        self.assertIn("doc2.txt", result)

    def test_format_display_text_custom_labels(self):
        """Test formatting display text with custom confidence labels."""
        custom_labels = {
            "high": "Very Confident",
            "medium": "Somewhat Confident",
            "low": "Not Confident"
        }
        result = format_display_text("Content", confidence=0.9, confidence_labels=custom_labels)
        self.assertIn("Very Confident", result)

    def test_format_display_text_custom_sources_prefix(self):
        """Test formatting display text with custom sources prefix."""
        sources = ["doc1.pdf"]
        result = format_display_text("Content", sources=sources, sources_prefix="References")
        self.assertIn("References:", result)

    def test_format_display_text_invalid_inputs(self):
        """Test formatting display text with invalid inputs."""
        # Invalid content type
        with self.assertRaises(ValueError) as cm:
            format_display_text(123)
        self.assertIn("Content must be string", str(cm.exception))

        # Invalid confidence type
        with self.assertRaises(ValueError) as cm:
            format_display_text("content", confidence="not numeric")
        self.assertIn("Confidence must be numeric", str(cm.exception))

        # Invalid confidence range
        with self.assertRaises(ValueError) as cm:
            format_display_text("content", confidence=1.5)
        self.assertIn("Confidence must be between 0.0 and 1.0", str(cm.exception))

        # Invalid sources type
        with self.assertRaises(ValueError) as cm:
            format_display_text("content", sources="not a list")
        self.assertIn("Sources must be list", str(cm.exception))

    def test_format_display_text_missing_confidence_labels(self):
        """Test formatting with missing confidence labels."""
        incomplete_labels = {"high": "High", "medium": "Medium"}  # Missing "low"
        with self.assertRaises(ValueError) as cm:
            format_display_text("content", confidence=0.3, confidence_labels=incomplete_labels)
        self.assertIn("Missing 'low' confidence label", str(cm.exception))


class TestDataStructures(unittest.TestCase):
    """Test data structure classes."""

    def test_parsed_response_valid(self):
        """Test creating valid ParsedResponse."""
        response = ParsedResponse(
            content="This is the response",
            confidence=0.8,
            sources_mentioned=["doc1.pdf", "doc2.txt"],
            has_answer=True,
            language="hr",
            metadata={"key": "value"}
        )

        self.assertEqual(response.content, "This is the response")
        self.assertEqual(response.confidence, 0.8)
        self.assertEqual(response.sources_mentioned, ["doc1.pdf", "doc2.txt"])
        self.assertTrue(response.has_answer)
        self.assertEqual(response.language, "hr")
        self.assertEqual(response.metadata, {"key": "value"})

    def test_parsed_response_defaults(self):
        """Test ParsedResponse with default values."""
        response = ParsedResponse(content="Content only")

        self.assertEqual(response.content, "Content only")
        self.assertIsNone(response.confidence)
        self.assertEqual(response.sources_mentioned, [])
        self.assertTrue(response.has_answer)
        self.assertEqual(response.language, "unknown")
        self.assertEqual(response.metadata, {})

    def test_parsed_response_invalid_content(self):
        """Test ParsedResponse with invalid content."""
        with self.assertRaises(ValueError) as cm:
            ParsedResponse(content=123)
        self.assertIn("Content must be string", str(cm.exception))

    def test_parsed_response_invalid_confidence(self):
        """Test ParsedResponse with invalid confidence."""
        with self.assertRaises(ValueError) as cm:
            ParsedResponse(content="test", confidence="not numeric")
        self.assertIn("Confidence must be numeric or None", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            ParsedResponse(content="test", confidence=1.5)
        self.assertIn("Confidence must be between 0.0 and 1.0", str(cm.exception))

    def test_parsed_response_invalid_types(self):
        """Test ParsedResponse with invalid field types."""
        # Invalid sources_mentioned
        with self.assertRaises(ValueError) as cm:
            ParsedResponse(content="test", sources_mentioned="not a list")
        self.assertIn("Sources mentioned must be list", str(cm.exception))

        # Invalid has_answer
        with self.assertRaises(ValueError) as cm:
            ParsedResponse(content="test", has_answer="not bool")
        self.assertIn("Has answer must be boolean", str(cm.exception))

        # Invalid language
        with self.assertRaises(ValueError) as cm:
            ParsedResponse(content="test", language=123)
        self.assertIn("Language must be string", str(cm.exception))

        # Invalid metadata
        with self.assertRaises(ValueError) as cm:
            ParsedResponse(content="test", metadata="not dict")
        self.assertIn("Metadata must be dict", str(cm.exception))

    def test_parsing_config_valid(self):
        """Test creating valid ParsingConfig."""
        config = ParsingConfig(
            no_answer_patterns=["no answer"],
            source_patterns=["source:"],
            confidence_indicators={"high": ["certain"]},
            language_patterns={"en": ["is", "the"]},
            cleaning_prefixes=["answer:"],
            display_settings={"key": "value"}
        )

        self.assertEqual(config.no_answer_patterns, ["no answer"])
        self.assertEqual(config.source_patterns, ["source:"])
        self.assertEqual(config.confidence_indicators, {"high": ["certain"]})
        self.assertEqual(config.language_patterns, {"en": ["is", "the"]})
        self.assertEqual(config.cleaning_prefixes, ["answer:"])
        self.assertEqual(config.display_settings, {"key": "value"})

    def test_parsing_config_defaults(self):
        """Test ParsingConfig with default values."""
        config = ParsingConfig()

        self.assertEqual(config.no_answer_patterns, [])
        self.assertEqual(config.source_patterns, [])
        self.assertEqual(config.confidence_indicators, {})
        self.assertEqual(config.language_patterns, {})
        self.assertEqual(config.cleaning_prefixes, [])
        self.assertEqual(config.display_settings, {})

    def test_parsing_config_invalid_types(self):
        """Test ParsingConfig with invalid types."""
        # Invalid no_answer_patterns
        with self.assertRaises(ValueError) as cm:
            ParsingConfig(no_answer_patterns="not a list")
        self.assertIn("No answer patterns must be list", str(cm.exception))

        # Invalid source_patterns
        with self.assertRaises(ValueError) as cm:
            ParsingConfig(source_patterns="not a list")
        self.assertIn("Source patterns must be list", str(cm.exception))

        # Invalid confidence_indicators
        with self.assertRaises(ValueError) as cm:
            ParsingConfig(confidence_indicators="not a dict")
        self.assertIn("Confidence indicators must be dict", str(cm.exception))

        # Invalid language_patterns
        with self.assertRaises(ValueError) as cm:
            ParsingConfig(language_patterns="not a dict")
        self.assertIn("Language patterns must be dict", str(cm.exception))

        # Invalid cleaning_prefixes
        with self.assertRaises(ValueError) as cm:
            ParsingConfig(cleaning_prefixes="not a list")
        self.assertIn("Cleaning prefixes must be list", str(cm.exception))

        # Invalid display_settings
        with self.assertRaises(ValueError) as cm:
            ParsingConfig(display_settings="not a dict")
        self.assertIn("Display settings must be dict", str(cm.exception))


class TestMultilingualResponseParser(unittest.TestCase):
    """Test MultilingualResponseParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config_provider = create_mock_config_provider()
        self.parser = MultilingualResponseParser(self.mock_config_provider, "hr")

    def test_initialization_valid(self):
        """Test valid initialization."""
        self.assertEqual(self.parser.language, "hr")
        self.assertIsNotNone(self.parser._config)

    def test_initialization_config_error(self):
        """Test initialization with config provider error."""
        mock_provider = MagicMock()
        mock_provider.get_parsing_config.side_effect = Exception("Config error")

        with self.assertRaises(Exception) as cm:
            MultilingualResponseParser(mock_provider, "hr")
        self.assertIn("Config error", str(cm.exception))

    def test_parse_response_valid(self):
        """Test parsing valid response."""
        raw_response = "Answer: This is a good response with high confidence"
        result = self.parser.parse_response(raw_response)

        self.assertIsInstance(result, ParsedResponse)
        self.assertIn("good response", result.content)
        self.assertTrue(result.has_answer)
        self.assertIsInstance(result.confidence, float)
        self.assertEqual(result.language, "hr")  # Default language

    def test_parse_response_empty(self):
        """Test parsing empty response."""
        result = self.parser.parse_response("")

        self.assertIsInstance(result, ParsedResponse)
        self.assertEqual(result.content, "No answer available.")
        self.assertFalse(result.has_answer)
        self.assertEqual(result.confidence, 0.0)

    def test_parse_response_whitespace_only(self):
        """Test parsing whitespace-only response."""
        result = self.parser.parse_response("   \n\t   ")

        self.assertIsInstance(result, ParsedResponse)
        self.assertEqual(result.content, "No answer available.")
        self.assertFalse(result.has_answer)

    def test_parse_response_no_answer_detected(self):
        """Test parsing response with no-answer patterns."""
        raw_response = "Ne znam odgovor na ovo pitanje"
        result = self.parser.parse_response(raw_response)

        self.assertIsInstance(result, ParsedResponse)
        self.assertFalse(result.has_answer)

    def test_parse_response_with_sources(self):
        """Test parsing response with source references."""
        raw_response = "Answer: According to source: document.pdf, this is correct"
        result = self.parser.parse_response(raw_response)

        self.assertIsInstance(result, ParsedResponse)
        self.assertTrue(len(result.sources_mentioned) > 0)

    def test_parse_response_with_context(self):
        """Test parsing response with query and context."""
        raw_response = "This is the answer"
        query = "What is the question?"
        context_chunks = ["chunk1", "chunk2"]

        result = self.parser.parse_response(raw_response, query, context_chunks)

        self.assertIsInstance(result, ParsedResponse)
        self.assertEqual(result.metadata["query_length"], len(query))
        self.assertEqual(result.metadata["context_chunks_count"], 2)

    def test_format_for_display_valid(self):
        """Test formatting valid ParsedResponse for display."""
        parsed_response = ParsedResponse(
            content="This is the content",
            confidence=0.8,
            sources_mentioned=["doc1.pdf"],
            has_answer=True,
            language="hr"
        )

        result = self.parser.format_for_display(parsed_response)

        self.assertIn("This is the content", result)
        self.assertIn("High Confidence", result)
        self.assertIn("Sources", result)
        self.assertIn("doc1.pdf", result)

    def test_format_for_display_missing_settings(self):
        """Test formatting with missing display settings."""
        # Create parser with incomplete display settings
        incomplete_settings = {
            "no_answer_message": "No answer",
            "high_confidence_label": "High",
            # Missing medium and low labels
            "sources_prefix": "Sources"
        }
        provider = create_mock_config_provider(display_settings=incomplete_settings)
        parser = MultilingualResponseParser(provider, "hr")

        parsed_response = ParsedResponse(content="test", confidence=0.6)

        with self.assertRaises(ValueError) as cm:
            parser.format_for_display(parsed_response)
        self.assertIn("Missing 'medium_confidence_label'", str(cm.exception))


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_create_response_parser(self):
        """Test creating response parser."""
        config_provider = create_mock_config_provider()
        parser = create_response_parser(config_provider, "en")

        self.assertIsInstance(parser, MultilingualResponseParser)
        self.assertEqual(parser.language, "en")

    def test_create_mock_config_provider_default(self):
        """Test creating mock config provider with defaults."""
        provider = create_mock_config_provider()
        config = provider.get_parsing_config("hr")

        self.assertIsInstance(config, ParsingConfig)
        self.assertIn("ne znam", config.no_answer_patterns)
        self.assertIn("high", config.confidence_indicators)
        self.assertIn("hr", config.language_patterns)

    def test_create_mock_config_provider_custom(self):
        """Test creating mock config provider with custom values."""
        custom_no_answer = ["custom no answer"]
        custom_sources = ["custom source:"]
        custom_confidence = {"high": ["custom certain"]}
        custom_language = {"test": ["test word"]}
        custom_prefixes = ["custom prefix:"]
        custom_display = {"custom_key": "custom_value"}

        provider = create_mock_config_provider(
            no_answer_patterns=custom_no_answer,
            source_patterns=custom_sources,
            confidence_indicators=custom_confidence,
            language_patterns=custom_language,
            cleaning_prefixes=custom_prefixes,
            display_settings=custom_display
        )
        config = provider.get_parsing_config("test")

        self.assertEqual(config.no_answer_patterns, custom_no_answer)
        self.assertEqual(config.source_patterns, custom_sources)
        self.assertEqual(config.confidence_indicators, custom_confidence)
        self.assertEqual(config.language_patterns, custom_language)
        self.assertEqual(config.cleaning_prefixes, custom_prefixes)
        self.assertEqual(config.display_settings, custom_display)


class TestProtocols(unittest.TestCase):
    """Test protocol implementations."""

    def test_config_provider_protocol(self):
        """Test ConfigProvider protocol implementation."""
        provider = create_mock_config_provider()

        # Should implement the protocol
        self.assertTrue(hasattr(provider, 'get_parsing_config'))
        self.assertTrue(callable(getattr(provider, 'get_parsing_config')))

        # Should return proper type
        config = provider.get_parsing_config("hr")
        self.assertIsInstance(config, ParsingConfig)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""

    def test_end_to_end_parsing(self):
        """Test complete parsing workflow."""
        # Create components
        config_provider = create_mock_config_provider()
        parser = create_response_parser(config_provider, "hr")

        # Parse response
        raw_response = "Answer: Ovo je odgovor sa velikim povjerenjem. Source: document.pdf"
        query = "Što je to?"
        context = ["chunk1", "chunk2"]

        parsed = parser.parse_response(raw_response, query, context)

        # Verify parsing
        self.assertIsInstance(parsed, ParsedResponse)
        self.assertTrue(parsed.has_answer)
        self.assertIsInstance(parsed.confidence, float)

        # Format for display
        formatted = parser.format_for_display(parsed)

        self.assertIsInstance(formatted, str)
        self.assertIn("Ovo je odgovor", formatted)

    def test_multilingual_support(self):
        """Test multilingual parsing support."""
        config_provider = create_mock_config_provider()

        # Test different languages
        for language in ["hr", "en"]:
            parser = create_response_parser(config_provider, language)
            self.assertEqual(parser.language, language)

            # Should work with same response
            parsed = parser.parse_response("This is a test response")
            self.assertIsInstance(parsed, ParsedResponse)

    def test_error_handling_chain(self):
        """Test error handling propagates correctly through the chain."""
        config_provider = create_mock_config_provider()
        parser = create_response_parser(config_provider, "hr")

        # Empty response should not raise error but return proper ParsedResponse
        parsed = parser.parse_response("")
        self.assertIsInstance(parsed, ParsedResponse)
        self.assertFalse(parsed.has_answer)

    def test_dependency_injection_isolation(self):
        """Test that different instances are properly isolated."""
        provider1 = create_mock_config_provider()
        provider2 = create_mock_config_provider()

        parser1 = create_response_parser(provider1, "hr")
        parser2 = create_response_parser(provider2, "en")

        # Should be independent instances
        self.assertNotEqual(parser1, parser2)
        self.assertEqual(parser1.language, "hr")
        self.assertEqual(parser2.language, "en")

    def test_configuration_inheritance(self):
        """Test configuration is properly inherited from provider."""
        custom_confidence = {
            "high": ["absolutely certain"],
            "medium": ["quite likely"],
            "low": ["somewhat uncertain"]
        }
        provider = create_mock_config_provider(confidence_indicators=custom_confidence)
        parser = create_response_parser(provider, "hr")

        # Should use custom configuration
        response = "I am absolutely certain this is correct"
        parsed = parser.parse_response(response)
        self.assertGreater(parsed.confidence, 0.8)  # Should detect high confidence


if __name__ == "__main__":
    unittest.main()