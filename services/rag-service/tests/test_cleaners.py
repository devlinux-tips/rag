"""
Tests for multilingual text cleaning system.

Tests all data classes, pure functions, and the MultilingualTextCleaner class
with proper dependency injection patterns.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Any

from src.preprocessing.cleaners import (
    # Data Classes
    CleaningResult,
    LanguageConfig,
    CleaningConfig,
    DocumentCleaningConfig,
    ChunkingConfig,
    SharedLanguageConfig,

    # Protocols
    ConfigProvider,
    LoggerProvider,
    EnvironmentProvider,

    # Pure Functions
    normalize_whitespace,
    remove_formatting_artifacts,
    remove_headers_footers,
    fix_ocr_errors,
    extract_sentences,
    normalize_diacritics,
    is_meaningful_text,
    detect_language_content,
    preserve_text_encoding,
    clean_text_comprehensive,

    # Main Class
    MultilingualTextCleaner,

    # Convenience Functions
    clean_text,
    detect_language_content_with_config,
    preserve_text_encoding_with_config,
    setup_language_environment,
)


class TestCleaningResult:
    """Test CleaningResult data class."""

    def test_cleaning_result_creation(self):
        """Test basic cleaning result creation."""
        result = CleaningResult(
            text="Cleaned text",
            original_length=100,
            cleaned_length=80,
            operations_performed=["whitespace_normalization", "ocr_correction"]
        )

        assert result.text == "Cleaned text"
        assert result.original_length == 100
        assert result.cleaned_length == 80
        assert result.operations_performed == ["whitespace_normalization", "ocr_correction"]
        assert result.language_score is None
        assert result.is_meaningful is None

    def test_cleaning_result_with_optional_fields(self):
        """Test cleaning result with optional fields."""
        result = CleaningResult(
            text="Test",
            original_length=10,
            cleaned_length=8,
            operations_performed=[],
            language_score=0.85,
            is_meaningful=True
        )

        assert result.language_score == 0.85
        assert result.is_meaningful is True


class TestLanguageConfig:
    """Test LanguageConfig data class."""

    def test_language_config_creation(self):
        """Test language configuration creation."""
        diacritic_map = {"ć": "c", "č": "c", "š": "s", "ž": "z"}
        config = LanguageConfig(
            diacritic_map=diacritic_map,
            word_char_pattern=r"[a-zA-ZčćžšđĆČŽŠĐ]",
            locale_primary="hr_HR.UTF-8",
            locale_fallback="C.UTF-8"
        )

        assert config.diacritic_map == diacritic_map
        assert config.word_char_pattern == r"[a-zA-ZčćžšđĆČŽŠĐ]"
        assert config.locale_primary == "hr_HR.UTF-8"
        assert config.locale_fallback == "C.UTF-8"

    def test_language_config_defaults(self):
        """Test language configuration with defaults."""
        config = LanguageConfig(
            diacritic_map={},
            word_char_pattern=r"[a-zA-Z]"
        )

        assert config.locale_primary is None
        assert config.locale_fallback == "C.UTF-8"


class TestCleaningConfig:
    """Test CleaningConfig data class."""

    def test_cleaning_config_creation(self):
        """Test cleaning configuration creation."""
        config = CleaningConfig(
            multiple_whitespace=r"\s+",
            multiple_linebreaks=r"\n\s*\n\s*\n+",
            min_meaningful_words=3,
            min_word_char_ratio=0.6
        )

        assert config.multiple_whitespace == r"\s+"
        assert config.multiple_linebreaks == r"\n\s*\n\s*\n+"
        assert config.min_meaningful_words == 3
        assert config.min_word_char_ratio == 0.6


class TestDocumentCleaningConfig:
    """Test DocumentCleaningConfig data class."""

    def test_document_cleaning_config_creation(self):
        """Test document cleaning configuration creation."""
        patterns = [r"^\s*Page \d+", r"^\s*Copyright"]
        corrections = {"rn": "m", "nn": "n"}

        config = DocumentCleaningConfig(
            header_footer_patterns=patterns,
            ocr_corrections=corrections
        )

        assert config.header_footer_patterns == patterns
        assert config.ocr_corrections == corrections


class TestChunkingConfig:
    """Test ChunkingConfig data class."""

    def test_chunking_config_creation(self):
        """Test chunking configuration creation."""
        config = ChunkingConfig(
            sentence_ending_pattern=r"[.!?]+",
            min_sentence_length=10
        )

        assert config.sentence_ending_pattern == r"[.!?]+"
        assert config.min_sentence_length == 10


class TestSharedLanguageConfig:
    """Test SharedLanguageConfig data class."""

    def test_shared_language_config_creation(self):
        """Test shared language configuration creation."""
        stopwords = ["i", "je", "u", "na", "se"]
        config = SharedLanguageConfig(
            stopwords=stopwords,
            chars_pattern=r"[čćžšđ]"
        )

        assert config.stopwords == stopwords
        assert config.chars_pattern == r"[čćžšđ]"


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_normalize_whitespace_basic(self):
        """Test basic whitespace normalization."""
        text = "This  has   multiple    spaces"
        result = normalize_whitespace(text)

        assert result == "This has multiple spaces"

    def test_normalize_whitespace_preserve_structure(self):
        """Test whitespace normalization preserving structure."""
        text = "Paragraph 1\n\nParagraph 2\n\n\nParagraph 3"
        result = normalize_whitespace(text, preserve_structure=True)

        assert "Paragraph 1\n\nParagraph 2\n\nParagraph 3" in result

    def test_normalize_whitespace_no_preserve_structure(self):
        """Test whitespace normalization without preserving structure."""
        text = "Line 1\nLine 2\n\nLine 3"
        result = normalize_whitespace(text, preserve_structure=False)

        assert result == "Line 1 Line 2 Line 3"

    def test_normalize_whitespace_empty(self):
        """Test normalizing empty text."""
        result = normalize_whitespace("")
        assert result == ""

        result = normalize_whitespace("   ")
        assert result == ""

    def test_remove_formatting_artifacts_dots(self):
        """Test removing excessive dots."""
        text = "End of sentence..... Start of next"
        result = remove_formatting_artifacts(text)

        assert result == "End of sentence... Start of next"

    def test_remove_formatting_artifacts_dashes(self):
        """Test removing excessive dashes."""
        text = "Section break----------next section"
        result = remove_formatting_artifacts(text)

        assert result == "Section break---next section"

    def test_remove_formatting_artifacts_isolated_chars(self):
        """Test removing isolated formatting characters."""
        text = "Normal text _ isolated underscore _ more text"
        result = remove_formatting_artifacts(text)

        # The function removes " _ " (space-underscore-space) patterns
        assert result == "Normal text isolated underscore more text"

    def test_remove_formatting_artifacts_empty(self):
        """Test removing artifacts from empty text."""
        result = remove_formatting_artifacts("")
        assert result == ""

    def test_remove_headers_footers_basic(self):
        """Test removing headers and footers."""
        text = "Page 1\nActual content here\nCopyright 2023"
        patterns = [r"^Page \d+", r"^Copyright"]

        result = remove_headers_footers(text, patterns)

        lines = result.split("\n")
        assert "Actual content here" in lines
        assert not any("Page" in line for line in lines)
        assert not any("Copyright" in line for line in lines)

    def test_remove_headers_footers_case_insensitive(self):
        """Test case insensitive header/footer removal."""
        text = "COPYRIGHT 2023\nContent\npage 1"
        patterns = [r"^page \d+", r"^copyright"]

        result = remove_headers_footers(text, patterns)

        assert "Content" in result
        assert "COPYRIGHT" not in result
        assert "page 1" not in result

    def test_remove_headers_footers_empty_patterns(self):
        """Test header/footer removal with empty patterns."""
        text = "Some text content"
        result = remove_headers_footers(text, [])

        assert result == text

    def test_remove_headers_footers_empty_text(self):
        """Test header/footer removal with empty text."""
        result = remove_headers_footers("", ["pattern"])
        assert result == ""

    def test_fix_ocr_errors_basic(self):
        """Test fixing basic OCR errors."""
        text = "The rnan walked to the house"
        corrections = {"rnan": "man"}

        result = fix_ocr_errors(text, corrections)

        assert result == "The man walked to the house"

    def test_fix_ocr_errors_case_insensitive(self):
        """Test case insensitive OCR error correction."""
        text = "The RNAN walked"
        corrections = {"rnan": "man"}

        result = fix_ocr_errors(text, corrections)

        assert result == "The man walked"

    def test_fix_ocr_errors_multiple(self):
        """Test fixing multiple OCR errors."""
        text = "The rnan and wornan walked"
        corrections = {"rnan": "man", "wornan": "woman"}

        result = fix_ocr_errors(text, corrections)

        assert result == "The man and woman walked"

    def test_fix_ocr_errors_empty(self):
        """Test OCR correction with empty inputs."""
        result = fix_ocr_errors("", {"error": "fix"})
        assert result == ""

        result = fix_ocr_errors("text", {})
        assert result == "text"

    def test_extract_sentences_basic(self):
        """Test basic sentence extraction."""
        text = "First sentence. Second sentence! Third sentence?"
        pattern = r"[.!?]+"
        min_length = 5

        result = extract_sentences(text, pattern, min_length)

        assert len(result) == 3
        assert "First sentence" in result
        assert "Second sentence" in result
        assert "Third sentence" in result

    def test_extract_sentences_min_length_filter(self):
        """Test sentence extraction with minimum length filter."""
        text = "Long sentence here. Hi."
        pattern = r"[.!?]+"
        min_length = 10

        result = extract_sentences(text, pattern, min_length)

        assert len(result) == 1
        assert "Long sentence here" in result
        assert "Hi" not in result

    def test_extract_sentences_empty(self):
        """Test sentence extraction from empty text."""
        result = extract_sentences("", r"[.!?]+", 1)
        assert result == []

    def test_normalize_diacritics_basic(self):
        """Test basic diacritic normalization."""
        text = "čučorko šećer žaba đačić"
        diacritic_map = {"č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "d"}

        result = normalize_diacritics(text, diacritic_map)

        assert result == "cucorko secer zaba dacic"

    def test_normalize_diacritics_case_sensitive(self):
        """Test case sensitive diacritic normalization."""
        text = "Čuček ŠEĆER"
        diacritic_map = {"č": "c", "Č": "C", "š": "s", "Š": "S", "ć": "c", "Ć": "C"}

        result = normalize_diacritics(text, diacritic_map)

        assert result == "Cucek SECER"

    def test_normalize_diacritics_empty(self):
        """Test diacritic normalization with empty inputs."""
        result = normalize_diacritics("", {"č": "c"})
        assert result == ""

        result = normalize_diacritics("text", {})
        assert result == "text"

    def test_is_meaningful_text_basic(self):
        """Test basic meaningful text detection."""
        text = "This is a meaningful sentence with enough words"
        min_words = 3
        word_pattern = r"[a-zA-Z]"
        min_ratio = 0.7

        result = is_meaningful_text(text, min_words, word_pattern, min_ratio)

        assert result is True

    def test_is_meaningful_text_too_few_words(self):
        """Test meaningful text detection with too few words."""
        text = "Only two"
        min_words = 3
        word_pattern = r"[a-zA-Z]"
        min_ratio = 0.7

        result = is_meaningful_text(text, min_words, word_pattern, min_ratio)

        assert result is False

    def test_is_meaningful_text_too_few_characters(self):
        """Test meaningful text detection with too few word characters."""
        text = "123 456 789 !@# $%^"  # Mostly numbers and symbols
        min_words = 2
        word_pattern = r"[a-zA-Z]"
        min_ratio = 0.5

        result = is_meaningful_text(text, min_words, word_pattern, min_ratio)

        assert result is False

    def test_is_meaningful_text_empty(self):
        """Test meaningful text detection with empty text."""
        result = is_meaningful_text("", 1, r"[a-zA-Z]", 0.5)
        assert result is False

        result = is_meaningful_text("   ", 1, r"[a-zA-Z]", 0.5)
        assert result is False

    def test_detect_language_content_basic(self):
        """Test basic language content detection."""
        text = "This is an English text with common words"
        language_words = ["this", "is", "an", "with", "the", "and"]

        result = detect_language_content(text, language_words)

        assert 0.0 <= result <= 1.0
        assert result > 0  # Should detect some English content

    def test_detect_language_content_croatian(self):
        """Test Croatian language content detection."""
        text = "Ovo je hrvatski tekst sa dijakritičnim znakovima: čćžšđ"
        language_words = ["ovo", "je", "sa", "i", "u", "na"]

        result = detect_language_content(text, language_words)

        assert 0.0 <= result <= 1.0
        assert result > 0  # Should detect Croatian content due to diacritics and words

    def test_detect_language_content_empty(self):
        """Test language content detection with empty text."""
        result = detect_language_content("", ["word"])
        assert result == 0.0

    def test_detect_language_content_no_matches(self):
        """Test language content detection with no matches."""
        text = "Random text without target language features"
        language_words = ["ovo", "je", "ili"]  # Croatian words

        result = detect_language_content(text, language_words)

        assert result >= 0.0  # Should be low but not necessarily 0

    def test_preserve_text_encoding_string(self):
        """Test preserving encoding for string input."""
        text = "Normal string text"
        result = preserve_text_encoding(text)

        assert result == text
        assert isinstance(result, str)

    def test_preserve_text_encoding_bytes_utf8(self):
        """Test preserving encoding for UTF-8 bytes."""
        text = "Croatian: čćžšđ".encode("utf-8")
        result = preserve_text_encoding(text)

        assert result == "Croatian: čćžšđ"
        assert isinstance(result, str)

    def test_preserve_text_encoding_bytes_latin1(self):
        """Test preserving encoding for Latin-1 bytes."""
        text = "Test text".encode("latin1")
        result = preserve_text_encoding(text)

        assert result == "Test text"
        assert isinstance(result, str)

    def test_preserve_text_encoding_other_types(self):
        """Test preserving encoding for other types."""
        result = preserve_text_encoding(123)
        assert result == "123"

        result = preserve_text_encoding(None)
        assert result == "None"

    def test_clean_text_comprehensive_basic(self):
        """Test comprehensive text cleaning."""
        language_config = LanguageConfig(
            diacritic_map={"ć": "c"},
            word_char_pattern=r"[a-zA-Z]"
        )
        cleaning_config = CleaningConfig(
            multiple_whitespace=r"\s+",
            multiple_linebreaks=r"\n+",
            min_meaningful_words=2,
            min_word_char_ratio=0.5
        )
        doc_config = DocumentCleaningConfig(
            header_footer_patterns=[r"^Page \d+"],
            ocr_corrections={"rnan": "man"}
        )

        text = "Page 1\nThe rnan   walked    slowly.\n\nEnd"

        result = clean_text_comprehensive(text, language_config, cleaning_config, doc_config)

        assert isinstance(result, CleaningResult)
        assert result.original_length == len(text)
        assert result.cleaned_length > 0
        assert "man walked" in result.text
        assert "Page 1" not in result.text
        assert len(result.operations_performed) > 0

    def test_clean_text_comprehensive_empty(self):
        """Test comprehensive cleaning with empty text."""
        language_config = LanguageConfig(diacritic_map={}, word_char_pattern=r"[a-zA-Z]")
        cleaning_config = CleaningConfig(
            multiple_whitespace=r"\s+",
            multiple_linebreaks=r"\n+",
            min_meaningful_words=1,
            min_word_char_ratio=0.5
        )
        doc_config = DocumentCleaningConfig(header_footer_patterns=[], ocr_corrections={})

        result = clean_text_comprehensive("", language_config, cleaning_config, doc_config)

        assert result.text == ""
        assert result.original_length == 0
        assert result.cleaned_length == 0
        assert result.operations_performed == []

    def test_clean_text_comprehensive_only_whitespace(self):
        """Test comprehensive cleaning with only whitespace."""
        language_config = LanguageConfig(diacritic_map={}, word_char_pattern=r"[a-zA-Z]")
        cleaning_config = CleaningConfig(
            multiple_whitespace=r"\s+",
            multiple_linebreaks=r"\n+",
            min_meaningful_words=1,
            min_word_char_ratio=0.5
        )
        doc_config = DocumentCleaningConfig(header_footer_patterns=[], ocr_corrections={})

        result = clean_text_comprehensive("   \n\n   ", language_config, cleaning_config, doc_config)

        assert result.text == ""
        assert result.original_length == 8
        assert result.cleaned_length == 0


class TestMultilingualTextCleaner:
    """Test MultilingualTextCleaner class."""

    def create_test_providers(self):
        """Create mock providers for testing."""
        config_provider = Mock(spec=ConfigProvider)
        logger_provider = Mock(spec=LoggerProvider)
        environment_provider = Mock(spec=EnvironmentProvider)

        # Mock language configuration
        language_data = {
            "diacritic_map": {"ć": "c", "č": "c", "š": "s", "ž": "z", "đ": "d"},
            "word_char_pattern": r"[a-zA-ZčćžšđĆČŽŠĐ]",
            "locale": {
                "primary": "hr_HR.UTF-8",
                "fallback": "C.UTF-8"
            }
        }
        config_provider.get_language_config.return_value = language_data

        # Mock cleaning configuration
        cleaning_data = {
            "multiple_whitespace": r"\s+",
            "multiple_linebreaks": r"\n\s*\n\s*\n+",
            "min_meaningful_words": 3,
            "min_word_char_ratio": 0.6
        }
        config_provider.get_cleaning_config.return_value = cleaning_data

        # Mock document cleaning configuration
        doc_cleaning_data = {
            "header_footer_patterns": [r"^\s*Page \d+", r"^\s*Copyright"],
            "ocr_corrections": {"rnan": "man", "wornan": "woman"}
        }
        config_provider.get_document_cleaning_config.return_value = doc_cleaning_data

        # Mock chunking configuration
        chunking_data = {
            "sentence_ending_pattern": r"[.!?]+",
            "min_sentence_length": 10
        }
        config_provider.get_chunking_config.return_value = chunking_data

        # Mock shared language configuration
        shared_data = {
            "stopwords": {"words": ["i", "je", "u", "na", "se", "da"]},
            "chars_pattern": r"[čćžšđ]"
        }
        config_provider.get_shared_language_config.return_value = shared_data

        return config_provider, logger_provider, environment_provider

    def test_multilingual_text_cleaner_initialization(self):
        """Test text cleaner initialization."""
        config_provider, logger_provider, environment_provider = self.create_test_providers()

        cleaner = MultilingualTextCleaner("hr", config_provider, logger_provider, environment_provider)

        assert cleaner.language == "hr"
        assert cleaner._config_provider == config_provider
        assert cleaner._logger == logger_provider
        assert cleaner._environment == environment_provider

        # Verify all configurations were loaded
        config_provider.get_language_config.assert_called_once_with("hr")
        config_provider.get_cleaning_config.assert_called_once()
        config_provider.get_document_cleaning_config.assert_called_once_with("hr")
        config_provider.get_chunking_config.assert_called_once_with("hr")
        config_provider.get_shared_language_config.assert_called_once_with("hr")

    def test_multilingual_text_cleaner_without_optional_providers(self):
        """Test cleaner initialization without optional providers."""
        config_provider, _, _ = self.create_test_providers()

        cleaner = MultilingualTextCleaner("en", config_provider)

        assert cleaner.language == "en"
        assert cleaner._logger is None
        assert cleaner._environment is None

    def test_cleaner_missing_language_config_word_char_pattern(self):
        """Test cleaner with missing word_char_pattern in language config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_language_config.return_value = {
            "diacritic_map": {},
            "locale": {"primary": "en_US.UTF-8", "fallback": "C.UTF-8"}
            # Missing word_char_pattern
        }

        with pytest.raises(ValueError, match="Missing 'word_char_pattern'"):
            MultilingualTextCleaner("en", config_provider)

    def test_cleaner_missing_language_config_locale(self):
        """Test cleaner with missing locale in language config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_language_config.return_value = {
            "diacritic_map": {},
            "word_char_pattern": r"[a-zA-Z]"
            # Missing locale
        }

        with pytest.raises(ValueError, match="Missing 'locale'"):
            MultilingualTextCleaner("en", config_provider)

    def test_cleaner_missing_language_config_locale_primary(self):
        """Test cleaner with missing locale primary in language config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_language_config.return_value = {
            "diacritic_map": {},
            "word_char_pattern": r"[a-zA-Z]",
            "locale": {"fallback": "C.UTF-8"}  # Missing primary
        }

        with pytest.raises(ValueError, match="Missing 'primary' in locale"):
            MultilingualTextCleaner("en", config_provider)

    def test_cleaner_missing_language_config_locale_fallback(self):
        """Test cleaner with missing locale fallback in language config."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_language_config.return_value = {
            "diacritic_map": {},
            "word_char_pattern": r"[a-zA-Z]",
            "locale": {"primary": "en_US.UTF-8"}  # Missing fallback
        }

        with pytest.raises(ValueError, match="Missing 'fallback' in locale"):
            MultilingualTextCleaner("en", config_provider)

    def test_cleaner_missing_shared_config_chars_pattern(self):
        """Test cleaner with missing chars_pattern in shared config."""
        config_provider, _, _ = self.create_test_providers()

        # Override shared config to missing chars_pattern
        shared_data = {
            "stopwords": {"words": ["test"]},
            # Missing chars_pattern
        }
        config_provider.get_shared_language_config.return_value = shared_data

        with pytest.raises(ValueError, match="Missing 'chars_pattern'"):
            MultilingualTextCleaner("hr", config_provider)

    def test_clean_text_success(self):
        """Test successful text cleaning."""
        config_provider, logger_provider, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider, logger_provider)

        text = "Page 1\nOvo je   test  tekst  sa  višak  razmaka."
        result = cleaner.clean_text(text)

        assert isinstance(result, CleaningResult)
        assert "test tekst sa višak razmaka" in result.text
        assert "Page 1" not in result.text
        assert result.original_length > result.cleaned_length
        logger_provider.debug.assert_called()

    def test_clean_text_preserve_structure(self):
        """Test text cleaning with structure preservation."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        text = "Paragraph 1\n\nParagraph 2"
        result = cleaner.clean_text(text, preserve_structure=True)

        assert "\n\n" in result.text  # Structure preserved

    def test_clean_text_no_preserve_structure(self):
        """Test text cleaning without structure preservation."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        text = "Line 1\nLine 2"
        result = cleaner.clean_text(text, preserve_structure=False)

        assert "\n" not in result.text  # Structure not preserved

    def test_extract_sentences_success(self):
        """Test successful sentence extraction."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        text = "First sentence. Second sentence! Third sentence?"
        sentences = cleaner.extract_sentences(text)

        assert len(sentences) >= 2  # At least some sentences extracted
        assert all(isinstance(s, str) for s in sentences)

    def test_normalize_diacritics_success(self):
        """Test successful diacritic normalization."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        text = "čuvaj ćevape šećer žaba đak"
        result = cleaner.normalize_diacritics(text)

        assert "c" in result  # ć and č should be normalized to c
        assert "s" in result  # š should be normalized to s
        assert "z" in result  # ž should be normalized to z
        assert "d" in result  # đ should be normalized to d

    def test_is_meaningful_text_success(self):
        """Test meaningful text detection."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        meaningful_text = "Ovo je značajan tekst sa dovoljno riječi"
        result = cleaner.is_meaningful_text(meaningful_text)

        assert result is True

    def test_is_meaningful_text_custom_min_words(self):
        """Test meaningful text detection with custom minimum words."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        short_text = "Kratak tekst"
        result = cleaner.is_meaningful_text(short_text, min_words=1)

        assert result is True  # Should pass with lower threshold

    def test_detect_language_content_success(self):
        """Test language content detection."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        croatian_text = "Ovo je hrvatski tekst sa stopwords riječi"
        score = cleaner.detect_language_content(croatian_text)

        assert 0.0 <= score <= 1.0
        assert score > 0  # Should detect some Croatian content

    def test_detect_language_content_empty(self):
        """Test language content detection with empty text."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        score = cleaner.detect_language_content("")

        assert score == 0.0

    def test_preserve_encoding_string(self):
        """Test encoding preservation for string."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        text = "Croatian: čćžšđ"
        result = cleaner.preserve_encoding(text)

        assert result == text
        assert isinstance(result, str)

    def test_preserve_encoding_non_string(self):
        """Test encoding preservation for non-string."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        result = cleaner.preserve_encoding(123)

        assert result == "123"
        assert isinstance(result, str)

    def test_setup_language_environment_with_provider(self):
        """Test language environment setup with provider."""
        config_provider, _, environment_provider = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider, environment_provider=environment_provider)

        cleaner.setup_language_environment()

        environment_provider.set_environment_variable.assert_called_with("PYTHONIOENCODING", "utf-8")
        # Note: set_locale is called with locale.LC_ALL which is an integer

    def test_setup_language_environment_without_provider(self):
        """Test language environment setup without provider."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        # Should not raise exception
        cleaner.setup_language_environment()

    def test_logging_methods(self):
        """Test logging methods."""
        config_provider, logger_provider, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider, logger_provider)

        cleaner._log_debug("Test debug message")

        logger_provider.debug.assert_called()

    def test_logging_methods_without_logger(self):
        """Test logging methods without logger provider."""
        config_provider, _, _ = self.create_test_providers()
        cleaner = MultilingualTextCleaner("hr", config_provider)

        # Should not raise exceptions
        cleaner._log_debug("Test message")


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('src.preprocessing.cleaners_providers.create_config_provider')
    def test_clean_text_with_provider(self, mock_create_provider):
        """Test clean_text convenience function with provider."""
        config_provider = Mock(spec=ConfigProvider)

        # Mock the provider creation
        mock_create_provider.return_value = config_provider

        # Mock all required configurations
        config_provider.get_language_config.return_value = {
            "diacritic_map": {},
            "word_char_pattern": r"[a-zA-Z]",
            "locale": {"primary": "en_US.UTF-8", "fallback": "C.UTF-8"}
        }
        config_provider.get_cleaning_config.return_value = {
            "multiple_whitespace": r"\s+",
            "multiple_linebreaks": r"\n+",
            "min_meaningful_words": 1,
            "min_word_char_ratio": 0.5
        }
        config_provider.get_document_cleaning_config.return_value = {
            "header_footer_patterns": [],
            "ocr_corrections": {}
        }
        config_provider.get_chunking_config.return_value = {
            "sentence_ending_pattern": r"[.!?]+",
            "min_sentence_length": 1
        }
        config_provider.get_shared_language_config.return_value = {
            "stopwords": {"words": []},
            "chars_pattern": r"[a-zA-Z]"
        }

        result = clean_text("Test  text  with  spaces", "en", config_provider=config_provider)

        assert isinstance(result, str)
        assert "Test text with spaces" in result

    @patch('src.preprocessing.cleaners_providers.create_config_provider')
    def test_detect_language_content_with_config_function(self, mock_create_provider):
        """Test detect_language_content_with_config convenience function."""
        config_provider = Mock(spec=ConfigProvider)
        mock_create_provider.return_value = config_provider

        # Mock all required configurations
        config_provider.get_language_config.return_value = {
            "diacritic_map": {},
            "word_char_pattern": r"[a-zA-Z]",
            "locale": {"primary": "en_US.UTF-8", "fallback": "C.UTF-8"}
        }
        config_provider.get_cleaning_config.return_value = {
            "multiple_whitespace": r"\s+",
            "multiple_linebreaks": r"\n+",
            "min_meaningful_words": 1,
            "min_word_char_ratio": 0.5
        }
        config_provider.get_document_cleaning_config.return_value = {
            "header_footer_patterns": [],
            "ocr_corrections": {}
        }
        config_provider.get_chunking_config.return_value = {
            "sentence_ending_pattern": r"[.!?]+",
            "min_sentence_length": 1
        }
        config_provider.get_shared_language_config.return_value = {
            "stopwords": {"words": ["the", "and", "of"]},
            "chars_pattern": r"[a-zA-Z]"
        }

        result = detect_language_content_with_config("The text and content", "en")

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch('src.preprocessing.cleaners_providers.create_config_provider')
    def test_preserve_text_encoding_with_config_function(self, mock_create_provider):
        """Test preserve_text_encoding_with_config convenience function."""
        config_provider = Mock(spec=ConfigProvider)
        mock_create_provider.return_value = config_provider

        # Mock all required configurations
        config_provider.get_language_config.return_value = {
            "diacritic_map": {"ć": "c"},
            "word_char_pattern": r"[a-zA-Z]",
            "locale": {"primary": "hr_HR.UTF-8", "fallback": "C.UTF-8"}
        }
        config_provider.get_cleaning_config.return_value = {
            "multiple_whitespace": r"\s+",
            "multiple_linebreaks": r"\n+",
            "min_meaningful_words": 1,
            "min_word_char_ratio": 0.5
        }
        config_provider.get_document_cleaning_config.return_value = {
            "header_footer_patterns": [],
            "ocr_corrections": {}
        }
        config_provider.get_chunking_config.return_value = {
            "sentence_ending_pattern": r"[.!?]+",
            "min_sentence_length": 1
        }
        config_provider.get_shared_language_config.return_value = {
            "stopwords": {"words": []},
            "chars_pattern": r"[čćžšđ]"
        }

        result = preserve_text_encoding_with_config("Test text", "hr")

        assert isinstance(result, str)
        assert result == "Test text"

    @patch('src.preprocessing.cleaners_providers.create_environment_provider')
    @patch('src.preprocessing.cleaners_providers.create_config_provider')
    def test_setup_language_environment_function(self, mock_create_config, mock_create_env):
        """Test setup_language_environment convenience function."""
        config_provider = Mock(spec=ConfigProvider)
        environment_provider = Mock(spec=EnvironmentProvider)

        mock_create_config.return_value = config_provider
        mock_create_env.return_value = environment_provider

        # Mock all required configurations
        config_provider.get_language_config.return_value = {
            "diacritic_map": {},
            "word_char_pattern": r"[a-zA-Z]",
            "locale": {"primary": "en_US.UTF-8", "fallback": "C.UTF-8"}
        }
        config_provider.get_cleaning_config.return_value = {
            "multiple_whitespace": r"\s+",
            "multiple_linebreaks": r"\n+",
            "min_meaningful_words": 1,
            "min_word_char_ratio": 0.5
        }
        config_provider.get_document_cleaning_config.return_value = {
            "header_footer_patterns": [],
            "ocr_corrections": {}
        }
        config_provider.get_chunking_config.return_value = {
            "sentence_ending_pattern": r"[.!?]+",
            "min_sentence_length": 1
        }
        config_provider.get_shared_language_config.return_value = {
            "stopwords": {"words": []},
            "chars_pattern": r"[a-zA-Z]"
        }

        # Should not raise exceptions
        setup_language_environment("en")

        # Verify providers were created
        mock_create_config.assert_called_once()
        mock_create_env.assert_called_once()