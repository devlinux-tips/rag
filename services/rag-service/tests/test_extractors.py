"""
Tests for document text extraction system.

Tests all data classes, pure functions, and the DocumentExtractor class
with proper dependency injection patterns.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from typing import Any

from src.preprocessing.extractors import (
    # Data Classes
    ExtractionResult,
    ExtractionConfig,

    # Protocols
    FileSystemProvider,
    ConfigProvider,
    LoggerProvider,

    # Pure Functions
    validate_file_format,
    extract_text_from_pdf_binary,
    extract_text_from_docx_binary,
    extract_text_with_encoding_fallback,
    post_process_extracted_text,

    # Main Class
    DocumentExtractor,

    # Convenience Functions
    extract_document_text,
)


class TestExtractionResult:
    """Test ExtractionResult data class."""

    def test_extraction_result_creation(self):
        """Test basic extraction result creation."""
        result = ExtractionResult(
            text="Extracted text content",
            character_count=20,
            extraction_method="PDF"
        )

        assert result.text == "Extracted text content"
        assert result.character_count == 20
        assert result.extraction_method == "PDF"
        assert result.encoding_used is None
        assert result.pages_processed is None
        assert result.error_details is None

    def test_extraction_result_with_optional_fields(self):
        """Test extraction result with optional fields."""
        result = ExtractionResult(
            text="Test content",
            character_count=12,
            extraction_method="TXT",
            encoding_used="utf-8",
            pages_processed=5,
            error_details="Minor warning"
        )

        assert result.encoding_used == "utf-8"
        assert result.pages_processed == 5
        assert result.error_details == "Minor warning"


class TestExtractionConfig:
    """Test ExtractionConfig data class."""

    def test_extraction_config_creation(self):
        """Test extraction configuration creation."""
        config = ExtractionConfig(
            supported_formats=[".pdf", ".docx", ".txt"],
            text_encodings=["utf-8", "latin1"],
            max_file_size_mb=100,
            enable_logging=True
        )

        assert config.supported_formats == [".pdf", ".docx", ".txt"]
        assert config.text_encodings == ["utf-8", "latin1"]
        assert config.max_file_size_mb == 100
        assert config.enable_logging is True

    def test_extraction_config_defaults(self):
        """Test extraction configuration with defaults."""
        config = ExtractionConfig(
            supported_formats=[".pdf"],
            text_encodings=["utf-8"]
        )

        assert config.max_file_size_mb == 50  # Default value
        assert config.enable_logging is True  # Default value


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_validate_file_format_supported(self):
        """Test file format validation with supported format."""
        file_path = Path("document.pdf")
        supported_formats = [".pdf", ".docx", ".txt"]

        is_valid, error_msg = validate_file_format(file_path, supported_formats)

        assert is_valid is True
        assert error_msg is None

    def test_validate_file_format_unsupported(self):
        """Test file format validation with unsupported format."""
        file_path = Path("document.xlsx")
        supported_formats = [".pdf", ".docx", ".txt"]

        is_valid, error_msg = validate_file_format(file_path, supported_formats)

        assert is_valid is False
        assert "Unsupported format: .xlsx" in error_msg
        assert str(supported_formats) in error_msg

    def test_validate_file_format_case_insensitive(self):
        """Test file format validation is case insensitive."""
        file_path = Path("document.PDF")
        supported_formats = [".pdf", ".docx", ".txt"]

        is_valid, error_msg = validate_file_format(file_path, supported_formats)

        assert is_valid is True
        assert error_msg is None

    def test_extract_text_from_pdf_binary_success(self):
        """Test successful PDF text extraction."""
        # Create a simple PDF binary mock
        pdf_content = b"%PDF-1.4 simple test content"

        with patch('src.preprocessing.extractors.PdfReader') as mock_pdf_reader:
            # Mock page object
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test PDF content"

            # Mock PDF reader
            mock_reader_instance = Mock()
            mock_reader_instance.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader_instance

            text, pages_processed = extract_text_from_pdf_binary(pdf_content)

            assert text == "Test PDF content"
            assert pages_processed == 1

    def test_extract_text_from_pdf_binary_multiple_pages(self):
        """Test PDF text extraction with multiple pages."""
        pdf_content = b"%PDF-1.4 multi-page content"

        with patch('src.preprocessing.extractors.PdfReader') as mock_pdf_reader:
            # Mock multiple pages
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1 content"

            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Page 2 content"

            # Mock PDF reader
            mock_reader_instance = Mock()
            mock_reader_instance.pages = [mock_page1, mock_page2]
            mock_pdf_reader.return_value = mock_reader_instance

            text, pages_processed = extract_text_from_pdf_binary(pdf_content)

            assert "Page 1 content" in text
            assert "Page 2 content" in text
            assert pages_processed == 2

    def test_extract_text_from_pdf_binary_empty_pages(self):
        """Test PDF text extraction with empty pages."""
        pdf_content = b"%PDF-1.4 empty pages"

        with patch('src.preprocessing.extractors.PdfReader') as mock_pdf_reader:
            # Mock pages with empty content
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "   "  # Whitespace only

            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Valid content"

            # Mock PDF reader
            mock_reader_instance = Mock()
            mock_reader_instance.pages = [mock_page1, mock_page2]
            mock_pdf_reader.return_value = mock_reader_instance

            text, pages_processed = extract_text_from_pdf_binary(pdf_content)

            assert text == "Valid content"  # Only non-empty content included
            assert pages_processed == 2

    def test_extract_text_from_docx_binary_success(self):
        """Test successful DOCX text extraction."""
        docx_content = b"DOCX binary content"

        with patch('src.preprocessing.extractors.Document') as mock_document:
            # Mock paragraphs
            mock_paragraph1 = Mock()
            mock_paragraph1.text = "First paragraph"

            mock_paragraph2 = Mock()
            mock_paragraph2.text = "Second paragraph"

            # Mock document
            mock_doc_instance = Mock()
            mock_doc_instance.paragraphs = [mock_paragraph1, mock_paragraph2]
            mock_document.return_value = mock_doc_instance

            text = extract_text_from_docx_binary(docx_content)

            assert "First paragraph" in text
            assert "Second paragraph" in text

    def test_extract_text_from_docx_binary_empty_paragraphs(self):
        """Test DOCX text extraction with empty paragraphs."""
        docx_content = b"DOCX binary content"

        with patch('src.preprocessing.extractors.Document') as mock_document:
            # Mock paragraphs with some empty
            mock_paragraph1 = Mock()
            mock_paragraph1.text = "   "  # Empty paragraph

            mock_paragraph2 = Mock()
            mock_paragraph2.text = "Valid content"

            mock_paragraph3 = Mock()
            mock_paragraph3.text = ""  # Empty paragraph

            # Mock document
            mock_doc_instance = Mock()
            mock_doc_instance.paragraphs = [mock_paragraph1, mock_paragraph2, mock_paragraph3]
            mock_document.return_value = mock_doc_instance

            text = extract_text_from_docx_binary(docx_content)

            assert text == "Valid content"  # Only non-empty content included

    def test_extract_text_with_encoding_fallback_success_first(self):
        """Test text extraction with successful first encoding."""
        text_content = "Test content with utf-8 encoding"
        text_binary = text_content.encode("utf-8")
        encodings = ["utf-8", "latin1", "cp1252"]

        result_text, encoding_used = extract_text_with_encoding_fallback(text_binary, encodings)

        assert result_text == text_content
        assert encoding_used == "utf-8"

    def test_extract_text_with_encoding_fallback_success_second(self):
        """Test text extraction with successful second encoding."""
        # Create bytes that are invalid UTF-8 but valid latin1
        text_binary = b'\xe9\xe0\xf1'  # Invalid UTF-8, valid latin1 (éàñ)
        encodings = ["utf-8", "latin1", "cp1252"]

        result_text, encoding_used = extract_text_with_encoding_fallback(text_binary, encodings)

        assert result_text == "éàñ"
        assert encoding_used == "latin1"

    def test_extract_text_with_encoding_fallback_failure(self):
        """Test text extraction with all encodings failing."""
        # Create binary data that can't be decoded with common encodings
        text_binary = b"\xff\xfe\x00\x00invalid"
        encodings = ["utf-8", "ascii"]

        with pytest.raises(ValueError, match="Could not decode text with any supported encoding"):
            extract_text_with_encoding_fallback(text_binary, encodings)

    def test_post_process_extracted_text_basic(self):
        """Test basic text post-processing."""
        raw_text = "  First paragraph  \n\n  Second paragraph  \n\n  "
        result = post_process_extracted_text(raw_text)

        assert result == "First paragraph\n\nSecond paragraph"

    def test_post_process_extracted_text_empty_lines(self):
        """Test text post-processing with empty lines."""
        raw_text = "Paragraph 1\n\n\n\nParagraph 2\n   \nParagraph 3"
        result = post_process_extracted_text(raw_text)

        assert result == "Paragraph 1\n\nParagraph 2\n\nParagraph 3"

    def test_post_process_extracted_text_empty(self):
        """Test text post-processing with empty input."""
        result = post_process_extracted_text("")
        assert result == ""

        result = post_process_extracted_text("   ")
        assert result == ""


class TestDocumentExtractor:
    """Test DocumentExtractor class."""

    def create_test_providers(self):
        """Create mock providers for testing."""
        config_provider = Mock(spec=ConfigProvider)
        file_system_provider = Mock(spec=FileSystemProvider)
        logger_provider = Mock(spec=LoggerProvider)

        # Mock extraction configuration
        config_data = {
            "supported_formats": [".pdf", ".docx", ".txt"],
            "text_encodings": ["utf-8", "latin1", "cp1252"],
            "max_file_size_mb": 50,
            "enable_logging": True
        }
        config_provider.get_extraction_config.return_value = config_data

        return config_provider, file_system_provider, logger_provider

    def test_document_extractor_initialization(self):
        """Test document extractor initialization."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()

        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        assert extractor._config_provider == config_provider
        assert extractor._file_system == file_system_provider
        assert extractor._logger == logger_provider
        config_provider.get_extraction_config.assert_called_once()

    def test_document_extractor_without_logger(self):
        """Test extractor initialization without logger."""
        config_provider, file_system_provider, _ = self.create_test_providers()

        extractor = DocumentExtractor(config_provider, file_system_provider)

        assert extractor._logger is None

    def test_extractor_missing_config_max_file_size_mb(self):
        """Test extractor with missing max_file_size_mb config."""
        config_provider = Mock(spec=ConfigProvider)
        file_system_provider = Mock(spec=FileSystemProvider)

        config_data = {
            "supported_formats": [".pdf"],
            "text_encodings": ["utf-8"],
            "enable_logging": True
            # Missing max_file_size_mb
        }
        config_provider.get_extraction_config.return_value = config_data

        with pytest.raises(ValueError, match="Missing 'max_file_size_mb'"):
            DocumentExtractor(config_provider, file_system_provider)

    def test_extractor_missing_config_enable_logging(self):
        """Test extractor with missing enable_logging config."""
        config_provider = Mock(spec=ConfigProvider)
        file_system_provider = Mock(spec=FileSystemProvider)

        config_data = {
            "supported_formats": [".pdf"],
            "text_encodings": ["utf-8"],
            "max_file_size_mb": 50
            # Missing enable_logging
        }
        config_provider.get_extraction_config.return_value = config_data

        with pytest.raises(ValueError, match="Missing 'enable_logging'"):
            DocumentExtractor(config_provider, file_system_provider)

    def test_extract_text_file_not_found(self):
        """Test text extraction with non-existent file."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()
        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        file_path = Path("nonexistent.pdf")
        file_system_provider.file_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="File not found"):
            extractor.extract_text(file_path)

    def test_extract_text_file_too_large(self):
        """Test text extraction with file too large."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()
        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        file_path = Path("large.pdf")
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 100.0  # Exceeds 50MB limit

        with pytest.raises(ValueError, match="File too large"):
            extractor.extract_text(file_path)

    def test_extract_text_unsupported_format(self):
        """Test text extraction with unsupported format."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()
        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        file_path = Path("document.xlsx")  # Not in supported formats
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 10.0

        with pytest.raises(ValueError, match="Unsupported format"):
            extractor.extract_text(file_path)

    def test_extract_pdf_success(self):
        """Test successful PDF extraction."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()
        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        file_path = Path("document.pdf")
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 10.0
        file_system_provider.open_binary.return_value = b"PDF content"

        with patch('src.preprocessing.extractors.extract_text_from_pdf_binary') as mock_extract:
            mock_extract.return_value = ("Extracted PDF text", 3)

            result = extractor.extract_text(file_path)

            assert isinstance(result, ExtractionResult)
            assert result.text == "Extracted PDF text"
            assert result.extraction_method == "PDF"
            assert result.pages_processed == 3
            logger_provider.info.assert_called()

    def test_extract_docx_success(self):
        """Test successful DOCX extraction."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()
        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        file_path = Path("document.docx")
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 5.0
        file_system_provider.open_binary.return_value = b"DOCX content"

        with patch('src.preprocessing.extractors.extract_text_from_docx_binary') as mock_extract:
            mock_extract.return_value = "Extracted DOCX text"

            result = extractor.extract_text(file_path)

            assert isinstance(result, ExtractionResult)
            assert result.text == "Extracted DOCX text"
            assert result.extraction_method == "DOCX"
            assert result.pages_processed is None
            logger_provider.info.assert_called()

    def test_extract_txt_success(self):
        """Test successful TXT extraction."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()
        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        file_path = Path("document.txt")
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 1.0
        file_system_provider.open_binary.return_value = b"Text content"

        with patch('src.preprocessing.extractors.extract_text_with_encoding_fallback') as mock_extract:
            mock_extract.return_value = ("Extracted TXT text", "utf-8")

            result = extractor.extract_text(file_path)

            assert isinstance(result, ExtractionResult)
            assert result.text == "Extracted TXT text"
            assert result.extraction_method == "TXT"
            assert result.encoding_used == "utf-8"
            logger_provider.info.assert_called()

    def test_extract_text_logging_disabled(self):
        """Test text extraction with logging disabled."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()

        # Override config to disable logging
        config_data = {
            "supported_formats": [".txt"],
            "text_encodings": ["utf-8"],
            "max_file_size_mb": 50,
            "enable_logging": False
        }
        config_provider.get_extraction_config.return_value = config_data

        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        file_path = Path("document.txt")
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 1.0
        file_system_provider.open_binary.return_value = b"Text content"

        with patch('src.preprocessing.extractors.extract_text_with_encoding_fallback') as mock_extract:
            mock_extract.return_value = ("Extracted text", "utf-8")

            result = extractor.extract_text(file_path)

            assert isinstance(result, ExtractionResult)
            # Logger should not be called when logging is disabled
            logger_provider.info.assert_not_called()

    def test_logging_methods(self):
        """Test logging methods."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()
        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        extractor._log_info("Test info message")
        extractor._log_error("Test error message")

        logger_provider.info.assert_called_with("Test info message")
        logger_provider.error.assert_called_with("Test error message")

    def test_logging_methods_without_logger(self):
        """Test logging methods without logger provider."""
        config_provider, file_system_provider, _ = self.create_test_providers()
        extractor = DocumentExtractor(config_provider, file_system_provider)

        # Should not raise exceptions
        extractor._log_info("Test message")
        extractor._log_error("Test message")

    def test_logging_methods_with_logging_disabled(self):
        """Test logging methods with logging disabled in config."""
        config_provider, file_system_provider, logger_provider = self.create_test_providers()

        # Override config to disable logging
        config_data = {
            "supported_formats": [".txt"],
            "text_encodings": ["utf-8"],
            "max_file_size_mb": 50,
            "enable_logging": False
        }
        config_provider.get_extraction_config.return_value = config_data

        extractor = DocumentExtractor(config_provider, file_system_provider, logger_provider)

        extractor._log_info("Test message")
        extractor._log_error("Test message")

        # Logger should not be called when logging is disabled
        logger_provider.info.assert_not_called()
        logger_provider.error.assert_not_called()


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('src.preprocessing.extractors_providers.create_file_system_provider')
    @patch('src.preprocessing.extractors_providers.create_config_provider')
    def test_extract_document_text_with_string_path(self, mock_create_config, mock_create_fs):
        """Test extract_document_text convenience function with string path."""
        config_provider = Mock(spec=ConfigProvider)
        file_system_provider = Mock(spec=FileSystemProvider)

        mock_create_config.return_value = config_provider
        mock_create_fs.return_value = file_system_provider

        # Mock configuration
        config_data = {
            "supported_formats": [".txt"],
            "text_encodings": ["utf-8"],
            "max_file_size_mb": 50,
            "enable_logging": True
        }
        config_provider.get_extraction_config.return_value = config_data

        # Mock file system operations
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 1.0
        file_system_provider.open_binary.return_value = b"Test content"

        with patch('src.preprocessing.extractors.extract_text_with_encoding_fallback') as mock_extract:
            mock_extract.return_value = ("Extracted text", "utf-8")

            result = extract_document_text("document.txt")

            assert result == "Extracted text"
            assert isinstance(result, str)

    @patch('src.preprocessing.extractors_providers.create_file_system_provider')
    @patch('src.preprocessing.extractors_providers.create_config_provider')
    def test_extract_document_text_with_path_object(self, mock_create_config, mock_create_fs):
        """Test extract_document_text convenience function with Path object."""
        config_provider = Mock(spec=ConfigProvider)
        file_system_provider = Mock(spec=FileSystemProvider)

        mock_create_config.return_value = config_provider
        mock_create_fs.return_value = file_system_provider

        # Mock configuration
        config_data = {
            "supported_formats": [".pdf"],
            "text_encodings": ["utf-8"],
            "max_file_size_mb": 50,
            "enable_logging": True
        }
        config_provider.get_extraction_config.return_value = config_data

        # Mock file system operations
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 5.0
        file_system_provider.open_binary.return_value = b"PDF content"

        with patch('src.preprocessing.extractors.extract_text_from_pdf_binary') as mock_extract:
            mock_extract.return_value = ("PDF text content", 2)

            result = extract_document_text(Path("document.pdf"))

            assert result == "PDF text content"

    def test_extract_document_text_with_explicit_providers(self):
        """Test extract_document_text with explicit providers."""
        config_provider = Mock(spec=ConfigProvider)
        file_system_provider = Mock(spec=FileSystemProvider)

        # Mock configuration
        config_data = {
            "supported_formats": [".txt"],
            "text_encodings": ["utf-8"],
            "max_file_size_mb": 50,
            "enable_logging": True
        }
        config_provider.get_extraction_config.return_value = config_data

        # Mock file system operations
        file_system_provider.file_exists.return_value = True
        file_system_provider.get_file_size_mb.return_value = 1.0
        file_system_provider.open_binary.return_value = b"Custom content"

        with patch('src.preprocessing.extractors.extract_text_with_encoding_fallback') as mock_extract:
            mock_extract.return_value = ("Custom extracted text", "utf-8")

            result = extract_document_text(
                "custom.txt",
                config_provider=config_provider,
                file_system_provider=file_system_provider
            )

            assert result == "Custom extracted text"