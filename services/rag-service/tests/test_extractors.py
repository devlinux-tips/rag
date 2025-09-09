"""
Comprehensive test suite for extractors_v2.py demonstrating 100% testability.
Tests pure functions and dependency injection with complete isolation.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import pytest
from src.preprocessing.extractors import (DocumentExtractor, ExtractionConfig,
                                          ExtractionResult,
                                          extract_document_text,
                                          extract_text_from_docx_binary,
                                          extract_text_from_pdf_binary,
                                          extract_text_with_encoding_fallback,
                                          post_process_extracted_text,
                                          validate_file_format)
from src.preprocessing.extractors_providers import (MockConfigProvider,
                                                    MockFileSystemProvider,
                                                    MockLoggerProvider,
                                                    create_test_providers)


class TestPureFunctions:
    """Test pure business logic functions with no side effects."""

    def test_validate_file_format_supported(self):
        """Test file format validation for supported formats."""
        file_path = Path("/test/document.pdf")
        supported_formats = [".pdf", ".docx", ".txt"]

        is_valid, error_msg = validate_file_format(file_path, supported_formats)

        assert is_valid is True
        assert error_msg is None

    def test_validate_file_format_unsupported(self):
        """Test file format validation for unsupported formats."""
        file_path = Path("/test/document.xlsx")
        supported_formats = [".pdf", ".docx", ".txt"]

        is_valid, error_msg = validate_file_format(file_path, supported_formats)

        assert is_valid is False
        assert "Unsupported format: .xlsx" in error_msg
        assert "Supported: ['.pdf', '.docx', '.txt']" in error_msg

    def test_extract_text_with_encoding_fallback_utf8_success(self):
        """Test encoding fallback with UTF-8 success."""
        text_content = "Test Croatian text with diacritics: čćšžđ"
        text_binary = text_content.encode("utf-8")
        encodings = ["utf-8", "latin1", "cp1252"]

        extracted_text, encoding_used = extract_text_with_encoding_fallback(
            text_binary, encodings
        )

        assert extracted_text == text_content
        assert encoding_used == "utf-8"

    def test_extract_text_with_encoding_fallback_latin1_success(self):
        """Test encoding fallback with latin1 fallback."""
        text_content = "Test text with special chars"
        text_binary = text_content.encode("latin1")
        encodings = ["utf-8", "latin1", "cp1252"]

        # This will fail UTF-8 but succeed with latin1
        extracted_text, encoding_used = extract_text_with_encoding_fallback(
            text_binary, encodings
        )

        assert extracted_text == text_content
        assert encoding_used == "latin1"

    def test_extract_text_with_encoding_fallback_all_fail(self):
        """Test encoding fallback when all encodings fail."""
        # Create binary that can't be decoded with common encodings
        text_binary = b"\xff\xfe\x00\x00invalid"
        encodings = ["utf-8", "ascii"]

        with pytest.raises(ValueError) as exc_info:
            extract_text_with_encoding_fallback(text_binary, encodings)

        assert "Could not decode text with any supported encoding" in str(
            exc_info.value
        )

    def test_post_process_extracted_text_normal(self):
        """Test post-processing of normal text."""
        raw_text = "  Line 1  \n\n  Line 2  \n  \n  Line 3  "
        expected = "Line 1\n\nLine 2\n\nLine 3"

        processed = post_process_extracted_text(raw_text)

        assert processed == expected

    def test_post_process_extracted_text_empty(self):
        """Test post-processing of empty text."""
        assert post_process_extracted_text("") == ""
        assert post_process_extracted_text("   ") == ""
        assert post_process_extracted_text("\n\n\n") == ""

    def test_post_process_extracted_text_single_line(self):
        """Test post-processing of single line text."""
        raw_text = "  Single line with spaces  "
        expected = "Single line with spaces"

        processed = post_process_extracted_text(raw_text)

        assert processed == expected


class TestMockProviders:
    """Test mock providers for dependency injection."""

    def test_mock_config_provider(self):
        """Test mock configuration provider."""
        test_config = {
            "supported_formats": [".pdf", ".txt"],
            "text_encodings": ["utf-8"],
            "max_file_size_mb": 25,
        }

        provider = MockConfigProvider(test_config)
        config = provider.get_extraction_config()

        assert config == test_config

    def test_mock_file_system_provider_binary_file(self):
        """Test mock file system provider with binary file."""
        provider = MockFileSystemProvider()
        test_content = b"Binary test content"
        file_path = "/test/file.bin"

        provider.add_file(file_path, test_content, size_mb=0.1)

        assert provider.file_exists(Path(file_path))
        assert provider.get_file_size_mb(Path(file_path)) == 0.1
        assert provider.open_binary(Path(file_path)) == test_content

    def test_mock_file_system_provider_text_file(self):
        """Test mock file system provider with text file."""
        provider = MockFileSystemProvider()
        test_content = "Croatian text: čćšžđ"
        file_path = "/test/file.txt"

        provider.add_text_file(file_path, test_content, encoding="utf-8", size_mb=0.05)

        assert provider.file_exists(Path(file_path))
        assert provider.get_file_size_mb(Path(file_path)) == 0.05
        assert provider.open_text(Path(file_path), "utf-8") == test_content

    def test_mock_file_system_provider_nonexistent_file(self):
        """Test mock file system provider with nonexistent file."""
        provider = MockFileSystemProvider()

        assert not provider.file_exists(Path("/nonexistent.txt"))

        with pytest.raises(FileNotFoundError):
            provider.get_file_size_mb(Path("/nonexistent.txt"))

        with pytest.raises(FileNotFoundError):
            provider.open_binary(Path("/nonexistent.txt"))

    def test_mock_logger_provider(self):
        """Test mock logger provider."""
        logger = MockLoggerProvider()

        logger.info("Test info message")
        logger.error("Test error message")

        messages = logger.get_all_messages()
        assert messages["info"] == ["Test info message"]
        assert messages["error"] == ["Test error message"]

        logger.clear_messages()
        messages = logger.get_all_messages()
        assert messages["info"] == []
        assert messages["error"] == []


class TestDocumentExtractorWithDependencyInjection:
    """Test DocumentExtractor with complete dependency injection."""

    def setup_method(self):
        """Set up test dependencies."""
        self.test_config = {
            "supported_formats": [".pdf", ".docx", ".txt"],
            "text_encodings": ["utf-8", "latin1", "cp1252"],
            "max_file_size_mb": 10,
            "enable_logging": True,
        }

        self.config_provider = MockConfigProvider(self.test_config)
        self.file_system = MockFileSystemProvider()
        self.logger = MockLoggerProvider()

    def test_initialization_loads_config(self):
        """Test extractor initialization loads configuration correctly."""
        extractor = DocumentExtractor(
            self.config_provider, self.file_system, self.logger
        )

        assert extractor._config.supported_formats == [".pdf", ".docx", ".txt"]
        assert extractor._config.text_encodings == ["utf-8", "latin1", "cp1252"]
        assert extractor._config.max_file_size_mb == 10
        assert extractor._config.enable_logging is True

    def test_extract_nonexistent_file(self):
        """Test extraction of nonexistent file."""
        extractor = DocumentExtractor(
            self.config_provider, self.file_system, self.logger
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            extractor.extract_text(Path("/nonexistent.pdf"))

        assert "File not found: /nonexistent.pdf" in str(exc_info.value)

    def test_extract_file_too_large(self):
        """Test extraction of file exceeding size limit."""
        self.file_system.add_file(
            "/large.pdf", b"content", size_mb=15.0
        )  # Over 10MB limit

        extractor = DocumentExtractor(
            self.config_provider, self.file_system, self.logger
        )

        with pytest.raises(ValueError) as exc_info:
            extractor.extract_text(Path("/large.pdf"))

        assert "File too large: 15.0MB > 10MB" in str(exc_info.value)

    def test_extract_unsupported_format(self):
        """Test extraction of unsupported file format."""
        self.file_system.add_file("/test.xlsx", b"content", size_mb=1.0)

        extractor = DocumentExtractor(
            self.config_provider, self.file_system, self.logger
        )

        with pytest.raises(ValueError) as exc_info:
            extractor.extract_text(Path("/test.xlsx"))

        assert "Unsupported format: .xlsx" in str(exc_info.value)

    def test_extract_txt_file_success(self):
        """Test successful extraction of TXT file."""
        test_content = "Croatian text with diacritics: čćšžđ\nSecond line of text"
        self.file_system.add_text_file(
            "/test.txt", test_content, encoding="utf-8", size_mb=0.1
        )

        extractor = DocumentExtractor(
            self.config_provider, self.file_system, self.logger
        )

        result = extractor.extract_text(Path("/test.txt"))

        assert isinstance(result, ExtractionResult)
        assert (
            result.text == "Croatian text with diacritics: čćšžđ\n\nSecond line of text"
        )
        assert result.character_count == len(result.text)
        assert result.extraction_method == "TXT"
        assert result.encoding_used == "utf-8"
        assert result.error_details is None

        # Check logging
        messages = self.logger.get_all_messages()
        assert any("Extracting text from /test.txt" in msg for msg in messages["info"])
        assert any(
            "Successfully read TXT file with utf-8 encoding" in msg
            for msg in messages["info"]
        )

    def test_extract_txt_file_encoding_fallback(self):
        """Test TXT file extraction with encoding fallback."""
        test_content = "Test text with special characters"
        # Encode with latin1 to test fallback
        self.file_system.add_file(
            "/test.txt", test_content.encode("latin1"), size_mb=0.1
        )

        extractor = DocumentExtractor(
            self.config_provider, self.file_system, self.logger
        )

        result = extractor.extract_text(Path("/test.txt"))

        assert result.text == test_content
        assert result.extraction_method == "TXT"
        assert result.encoding_used == "latin1"  # Should fallback to latin1

    def test_extract_with_logging_disabled(self):
        """Test extraction with logging disabled."""
        # Disable logging in config
        config_no_logging = {**self.test_config, "enable_logging": False}
        config_provider = MockConfigProvider(config_no_logging)

        test_content = "Test content"
        self.file_system.add_text_file("/test.txt", test_content, size_mb=0.1)

        extractor = DocumentExtractor(config_provider, self.file_system, self.logger)

        result = extractor.extract_text(Path("/test.txt"))

        assert result.text == test_content
        # Should have no log messages due to disabled logging
        messages = self.logger.get_all_messages()
        assert len(messages["info"]) == 0

    def test_extract_with_no_logger_provider(self):
        """Test extraction with no logger provider."""
        test_content = "Test content"
        self.file_system.add_text_file("/test.txt", test_content, size_mb=0.1)

        # No logger provider
        extractor = DocumentExtractor(
            self.config_provider, self.file_system, logger_provider=None
        )

        result = extractor.extract_text(Path("/test.txt"))

        assert result.text == test_content
        # Should not raise any errors despite no logger


class TestConvenienceFunctions:
    """Test convenience functions and backward compatibility."""

    def test_extract_document_text_with_string_path(self):
        """Test convenience function with string path."""
        # This tests the convenience function but requires mocking the providers
        # We'll test the path conversion logic
        from unittest.mock import MagicMock, patch

        mock_extractor = MagicMock()
        mock_result = ExtractionResult(
            text="test content", character_count=12, extraction_method="TXT"
        )
        mock_extractor.extract_text.return_value = mock_result

        with patch(
            "src.preprocessing.extractors.DocumentExtractor",
            return_value=mock_extractor,
        ):
            with patch("src.preprocessing.extractors.create_config_provider"):
                with patch("src.preprocessing.extractors.create_file_system_provider"):
                    result = extract_document_text("/test/file.txt")

                    assert result == "test content"
                    # Verify Path conversion
                    mock_extractor.extract_text.assert_called_once()
                    called_path = mock_extractor.extract_text.call_args[0][0]
                    assert isinstance(called_path, Path)
                    assert str(called_path) == "/test/file.txt"


class TestFactoryFunctions:
    """Test provider factory functions."""

    def test_create_test_providers_defaults(self):
        """Test creating test providers with defaults."""
        config_provider, file_system_provider, logger_provider = create_test_providers()

        # Test default configuration
        config = config_provider.get_extraction_config()
        assert ".pdf" in config["supported_formats"]
        assert ".docx" in config["supported_formats"]
        assert ".txt" in config["supported_formats"]
        assert "utf-8" in config["text_encodings"]
        assert config["max_file_size_mb"] == 50

        # Test providers are mock types
        assert isinstance(config_provider, MockConfigProvider)
        assert isinstance(file_system_provider, MockFileSystemProvider)
        assert isinstance(logger_provider, MockLoggerProvider)

    def test_create_test_providers_custom_config(self):
        """Test creating test providers with custom configuration."""
        custom_config = {
            "supported_formats": [".pdf"],
            "text_encodings": ["utf-8"],
            "max_file_size_mb": 25,
            "enable_logging": False,
        }

        config_provider, _, _ = create_test_providers(config=custom_config)

        config = config_provider.get_extraction_config()
        assert config == custom_config

    def test_create_test_providers_with_mock_files(self):
        """Test creating test providers with mock files."""
        mock_files = {"/test1.txt": b"content1", "/test2.txt": b"content2"}

        _, file_system_provider, _ = create_test_providers(files=mock_files)

        assert file_system_provider.file_exists(Path("/test1.txt"))
        assert file_system_provider.file_exists(Path("/test2.txt"))
        assert not file_system_provider.file_exists(Path("/test3.txt"))


class TestIntegrationScenarios:
    """Integration tests demonstrating complete testable workflows."""

    def test_full_txt_extraction_workflow(self):
        """Test complete TXT extraction workflow from start to finish."""
        # Setup test environment
        test_config = {
            "supported_formats": [".txt"],
            "text_encodings": ["utf-8", "latin1"],
            "max_file_size_mb": 5,
            "enable_logging": True,
        }

        config_provider, file_system_provider, logger_provider = create_test_providers(
            config=test_config, mock_logging=True
        )

        # Add test file
        test_content = "Croatian document\nWith multiple lines\nAnd diacritics: čćšžđ"
        file_system_provider.add_text_file("/document.txt", test_content, size_mb=0.1)

        # Create extractor and process
        extractor = DocumentExtractor(
            config_provider, file_system_provider, logger_provider
        )
        result = extractor.extract_text(Path("/document.txt"))

        # Verify results
        expected_processed = (
            "Croatian document\n\nWith multiple lines\n\nAnd diacritics: čćšžđ"
        )
        assert result.text == expected_processed
        assert result.character_count == len(expected_processed)
        assert result.extraction_method == "TXT"
        assert result.encoding_used == "utf-8"
        assert result.error_details is None

        # Verify logging occurred
        messages = logger_provider.get_all_messages()
        info_messages = " ".join(messages["info"])
        assert "Extracting text from /document.txt" in info_messages
        assert "Successfully read TXT file with utf-8 encoding" in info_messages
        assert (
            f"Extracted {len(expected_processed)} characters from TXT" in info_messages
        )

    def test_error_handling_workflow(self):
        """Test error handling workflow with graceful degradation."""
        config_provider, file_system_provider, logger_provider = create_test_providers()

        # Add file that will cause extraction error (invalid binary for PDF)
        file_system_provider.add_file("/invalid.pdf", b"not a real pdf", size_mb=0.1)

        extractor = DocumentExtractor(
            config_provider, file_system_provider, logger_provider
        )
        result = extractor.extract_text(Path("/invalid.pdf"))

        # Should return error result, not raise exception
        assert result.text == ""
        assert result.character_count == 0
        assert result.extraction_method == "PDF"
        assert result.error_details is not None
        assert "Error extracting from PDF" in " ".join(
            logger_provider.get_all_messages()["error"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
