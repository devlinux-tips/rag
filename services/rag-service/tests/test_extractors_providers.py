"""
Tests for preprocessing/extractors_providers.py module.
Tests document extraction providers for dependency injection patterns.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.preprocessing.extractors_providers import (
    ConfigProvider,
    FileSystemProvider,
    LoggerProvider,
    create_config_provider,
    create_file_system_provider,
    create_logger_provider,
    create_providers,
)
from tests.conftest import (
    MockConfigProvider,
    MockFileSystemProvider,
    MockLoggerProvider,
    create_test_providers,
)


class TestConfigProvider(unittest.TestCase):
    """Test production configuration provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = ConfigProvider()

    def test_provider_creation(self):
        """Test provider can be created."""
        provider = ConfigProvider()
        self.assertIsInstance(provider, ConfigProvider)

    def test_provider_attributes(self):
        """Test provider has no attributes by default (stateless)."""
        self.assertEqual(len(self.provider.__dict__), 0)

    @patch("src.utils.config_loader.get_extraction_config")
    def test_get_extraction_config_success(self, mock_get_config):
        """Test successful retrieval of extraction configuration."""
        mock_config = {
            "supported_formats": [".pdf", ".txt"],
            "max_file_size_mb": 10
        }
        mock_get_config.return_value = mock_config

        result = self.provider.get_extraction_config()

        mock_get_config.assert_called_once()
        self.assertEqual(result, mock_config)

    @patch("src.utils.config_loader.get_extraction_config")
    def test_get_extraction_config_error(self, mock_get_config):
        """Test error handling when config loading fails."""
        mock_get_config.side_effect = Exception("Config file not found")

        with self.assertRaises(Exception) as cm:
            self.provider.get_extraction_config()

        self.assertIn("Config file not found", str(cm.exception))

    def test_provider_immutability(self):
        """Test provider has no mutable state."""
        provider1 = ConfigProvider()
        provider2 = ConfigProvider()

        # Both instances should be functionally identical
        self.assertEqual(type(provider1), type(provider2))
        self.assertEqual(provider1.__dict__, provider2.__dict__)


class TestFileSystemProvider(unittest.TestCase):
    """Test production file system provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = FileSystemProvider()

    def test_provider_creation(self):
        """Test provider can be created."""
        provider = FileSystemProvider()
        self.assertIsInstance(provider, FileSystemProvider)

    def test_provider_attributes(self):
        """Test provider has no attributes by default (stateless)."""
        self.assertEqual(len(self.provider.__dict__), 0)

    def test_file_exists_existing_file(self):
        """Test file_exists with existing file."""
        # Use a file that should exist
        test_file = Path(__file__)  # This test file should exist
        result = self.provider.file_exists(test_file)
        self.assertTrue(result)

    def test_file_exists_non_existing_file(self):
        """Test file_exists with non-existing file."""
        test_file = Path("/non/existing/file.txt")
        result = self.provider.file_exists(test_file)
        self.assertFalse(result)

    def test_get_file_size_mb_existing_file(self):
        """Test get_file_size_mb with existing file."""
        test_file = Path(__file__)  # This test file should exist
        result = self.provider.get_file_size_mb(test_file)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # File should have some size

    def test_get_file_size_mb_non_existing_file(self):
        """Test get_file_size_mb with non-existing file."""
        test_file = Path("/non/existing/file.txt")

        with self.assertRaises(FileNotFoundError):
            self.provider.get_file_size_mb(test_file)

    def test_open_binary_existing_file(self):
        """Test open_binary with existing file."""
        test_file = Path(__file__)  # This test file should exist
        result = self.provider.open_binary(test_file)

        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)  # File should have some content

    def test_open_binary_non_existing_file(self):
        """Test open_binary with non-existing file."""
        test_file = Path("/non/existing/file.txt")

        with self.assertRaises(FileNotFoundError):
            self.provider.open_binary(test_file)

    def test_open_text_existing_file(self):
        """Test open_text with existing file."""
        test_file = Path(__file__)  # This test file should exist
        result = self.provider.open_text(test_file, "utf-8")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)  # File should have some content
        self.assertIn("test", result.lower())  # Should contain test-related content

    def test_open_text_non_existing_file(self):
        """Test open_text with non-existing file."""
        test_file = Path("/non/existing/file.txt")

        with self.assertRaises(FileNotFoundError):
            self.provider.open_text(test_file, "utf-8")

    def test_open_text_invalid_encoding(self):
        """Test open_text with invalid encoding."""
        test_file = Path(__file__)  # This test file should exist

        with self.assertRaises(LookupError):
            self.provider.open_text(test_file, "invalid-encoding")


class TestLoggerProvider(unittest.TestCase):
    """Test production logger provider."""

    def test_provider_creation_default_name(self):
        """Test provider creation with default logger name."""
        provider = LoggerProvider()
        self.assertIsInstance(provider, LoggerProvider)
        self.assertEqual(provider.logger.name, "src.preprocessing.extractors_providers")

    def test_provider_creation_custom_name(self):
        """Test provider creation with custom logger name."""
        custom_name = "test.extractor.logger"
        provider = LoggerProvider(custom_name)
        self.assertEqual(provider.logger.name, custom_name)

    def test_provider_creation_none_name(self):
        """Test provider creation with None name uses module name."""
        provider = LoggerProvider(None)
        self.assertEqual(provider.logger.name, "src.preprocessing.extractors_providers")

    @patch("logging.getLogger")
    def test_info_logging(self, mock_get_logger):
        """Test info message logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        provider = LoggerProvider("test")
        provider.info("Test info message")

        mock_logger.info.assert_called_once_with("Test info message")

    @patch("logging.getLogger")
    def test_error_logging(self, mock_get_logger):
        """Test error message logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        provider = LoggerProvider("test")
        provider.error("Test error message")

        mock_logger.error.assert_called_once_with("Test error message")

    def test_logger_persistence(self):
        """Test that logger is stored in provider."""
        provider = LoggerProvider("test")
        self.assertTrue(hasattr(provider, "logger"))
        self.assertEqual(provider.logger.name, "test")


class TestMockConfigProvider(unittest.TestCase):
    """Test mock configuration provider."""

    def test_provider_creation(self):
        """Test provider creation with config."""
        config = {"test_key": "test_value"}
        provider = MockConfigProvider(config)
        self.assertIsInstance(provider, MockConfigProvider)
        self.assertEqual(provider.extraction_config, config)

    def test_get_extraction_config(self):
        """Test getting extraction configuration."""
        config = {
            "supported_formats": [".pdf", ".txt"],
            "max_file_size_mb": 20
        }
        provider = MockConfigProvider(config)

        result = provider.get_extraction_config()
        self.assertEqual(result, config)

    def test_config_isolation(self):
        """Test that different providers have isolated configs."""
        config1 = {"key": "value1"}
        config2 = {"key": "value2"}

        provider1 = MockConfigProvider(config1)
        provider2 = MockConfigProvider(config2)

        self.assertEqual(provider1.get_extraction_config(), config1)
        self.assertEqual(provider2.get_extraction_config(), config2)

    def test_config_mutability(self):
        """Test that config can be modified."""
        config = {"key": "original"}
        provider = MockConfigProvider(config)

        # Modify the original config
        config["key"] = "modified"

        # Provider should reflect the change
        self.assertEqual(provider.get_extraction_config()["key"], "modified")


class TestMockFileSystemProvider(unittest.TestCase):
    """Test mock file system provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockFileSystemProvider()

    def test_provider_creation(self):
        """Test provider creation."""
        provider = MockFileSystemProvider()
        self.assertIsInstance(provider, MockFileSystemProvider)
        self.assertEqual(provider.files, {})
        self.assertEqual(provider.file_sizes_mb, {})
        self.assertEqual(provider.existing_files, set())

    def test_add_file(self):
        """Test adding file to mock file system."""
        content = b"test content"
        self.provider.add_file("test.txt", content)

        self.assertIn("test.txt", self.provider.files)
        self.assertEqual(self.provider.files["test.txt"], content)
        self.assertIn("test.txt", self.provider.existing_files)

    def test_add_file_with_custom_size(self):
        """Test adding file with custom size."""
        content = b"small"
        custom_size = 10.5
        self.provider.add_file("test.txt", content, custom_size)

        self.assertEqual(self.provider.file_sizes_mb["test.txt"], custom_size)

    def test_add_file_auto_size(self):
        """Test adding file with automatic size calculation."""
        content = b"test content"
        self.provider.add_file("test.txt", content)

        expected_size = len(content) / (1024 * 1024)
        self.assertEqual(self.provider.file_sizes_mb["test.txt"], expected_size)

    def test_add_text_file(self):
        """Test adding text file."""
        content = "test content"
        self.provider.add_text_file("test.txt", content)

        self.assertIn("test.txt", self.provider.files)
        self.assertEqual(self.provider.files["test.txt"], content.encode("utf-8"))

    def test_add_text_file_custom_encoding(self):
        """Test adding text file with custom encoding."""
        content = "test content"
        encoding = "latin1"
        self.provider.add_text_file("test.txt", content, encoding)

        self.assertEqual(self.provider.files["test.txt"], content.encode(encoding))

    def test_file_exists_true(self):
        """Test file_exists returns True for existing file."""
        self.provider.add_file("test.txt", b"content")

        result = self.provider.file_exists(Path("test.txt"))
        self.assertTrue(result)

    def test_file_exists_false(self):
        """Test file_exists returns False for non-existing file."""
        result = self.provider.file_exists(Path("non_existing.txt"))
        self.assertFalse(result)

    def test_get_file_size_mb_success(self):
        """Test getting file size for existing file."""
        size = 5.5
        self.provider.add_file("test.txt", b"content", size)

        result = self.provider.get_file_size_mb(Path("test.txt"))
        self.assertEqual(result, size)

    def test_get_file_size_mb_file_not_found(self):
        """Test getting file size for non-existing file."""
        with self.assertRaises(FileNotFoundError) as cm:
            self.provider.get_file_size_mb(Path("non_existing.txt"))

        self.assertIn("Mock file not found", str(cm.exception))

    def test_open_binary_success(self):
        """Test opening file in binary mode."""
        content = b"binary content"
        self.provider.add_file("test.bin", content)

        result = self.provider.open_binary(Path("test.bin"))
        self.assertEqual(result, content)

    def test_open_binary_file_not_found(self):
        """Test opening non-existing file in binary mode."""
        with self.assertRaises(FileNotFoundError) as cm:
            self.provider.open_binary(Path("non_existing.bin"))

        self.assertIn("Mock file not found", str(cm.exception))

    def test_open_text_success(self):
        """Test opening file in text mode."""
        content = "text content"
        self.provider.add_text_file("test.txt", content)

        result = self.provider.open_text(Path("test.txt"), "utf-8")
        self.assertEqual(result, content)

    def test_open_text_different_encoding(self):
        """Test opening file with different encoding."""
        content = "text content"
        encoding = "latin1"
        self.provider.add_text_file("test.txt", content, encoding)

        result = self.provider.open_text(Path("test.txt"), encoding)
        self.assertEqual(result, content)

    def test_open_text_file_not_found(self):
        """Test opening non-existing file in text mode."""
        with self.assertRaises(FileNotFoundError):
            self.provider.open_text(Path("non_existing.txt"), "utf-8")

    def test_path_handling(self):
        """Test that Path objects are handled correctly."""
        content = b"test"
        file_path = "dir/subdir/file.txt"
        self.provider.add_file(file_path, content)

        # Should work with Path object
        result = self.provider.file_exists(Path(file_path))
        self.assertTrue(result)


class TestMockLoggerProvider(unittest.TestCase):
    """Test mock logger provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockLoggerProvider()

    def test_provider_creation(self):
        """Test provider creation."""
        provider = MockLoggerProvider()
        self.assertIsInstance(provider, MockLoggerProvider)
        self.assertEqual(provider.info_messages, [])
        self.assertEqual(provider.error_messages, [])

    def test_info_logging(self):
        """Test info message logging."""
        message = "Test info message"
        self.provider.info(message)

        self.assertEqual(len(self.provider.info_messages), 1)
        self.assertEqual(self.provider.info_messages[0], message)

    def test_error_logging(self):
        """Test error message logging."""
        message = "Test error message"
        self.provider.error(message)

        self.assertEqual(len(self.provider.error_messages), 1)
        self.assertEqual(self.provider.error_messages[0], message)

    def test_multiple_messages(self):
        """Test logging multiple messages."""
        self.provider.info("Info 1")
        self.provider.error("Error 1")
        self.provider.info("Info 2")

        self.assertEqual(len(self.provider.info_messages), 2)
        self.assertEqual(len(self.provider.error_messages), 1)

    def test_get_all_messages(self):
        """Test getting all logged messages."""
        self.provider.info("Info message")
        self.provider.error("Error message")

        result = self.provider.get_all_messages()

        expected = {
            "info": ["Info message"],
            "error": ["Error message"]
        }
        self.assertEqual(result, expected)

    def test_clear_messages(self):
        """Test clearing all messages."""
        self.provider.info("Info message")
        self.provider.error("Error message")

        self.provider.clear_messages()

        self.assertEqual(len(self.provider.info_messages), 0)
        self.assertEqual(len(self.provider.error_messages), 0)

    def test_message_order_preservation(self):
        """Test that message order is preserved."""
        messages = ["First", "Second", "Third"]

        for msg in messages:
            self.provider.info(msg)

        self.assertEqual(self.provider.info_messages, messages)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating providers."""

    def test_create_config_provider_production(self):
        """Test creating production config provider."""
        provider = create_config_provider()
        self.assertIsInstance(provider, ConfigProvider)

    def test_create_config_provider_mock(self):
        """Test creating mock config provider."""
        config = {"test": "value"}
        provider = create_config_provider(config)
        self.assertIsInstance(provider, MockConfigProvider)
        self.assertEqual(provider.extraction_config, config)

    def test_create_file_system_provider_production(self):
        """Test creating production file system provider."""
        provider = create_file_system_provider()
        self.assertIsInstance(provider, FileSystemProvider)

    def test_create_file_system_provider_mock(self):
        """Test creating mock file system provider."""
        files = {"test.txt": b"content"}
        provider = create_file_system_provider(files)
        self.assertIsInstance(provider, MockFileSystemProvider)
        self.assertIn("test.txt", provider.files)

    def test_create_logger_provider_production(self):
        """Test creating production logger provider."""
        provider = create_logger_provider()
        self.assertIsInstance(provider, LoggerProvider)

    def test_create_logger_provider_production_with_name(self):
        """Test creating production logger provider with name."""
        logger_name = "test.logger"
        provider = create_logger_provider(logger_name)
        self.assertEqual(provider.logger.name, logger_name)

    def test_create_logger_provider_mock(self):
        """Test creating mock logger provider."""
        provider = create_logger_provider(mock=True)
        self.assertIsInstance(provider, MockLoggerProvider)


class TestConvenienceBuilders(unittest.TestCase):
    """Test convenience provider builder functions."""

    def test_create_test_providers_defaults(self):
        """Test creating test providers with defaults."""
        config_provider, fs_provider, logger_provider = create_test_providers()

        self.assertIsInstance(config_provider, MockConfigProvider)
        # When files=None, creates production provider (as per implementation)
        self.assertIsInstance(fs_provider, FileSystemProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)

    def test_create_test_providers_custom_config(self):
        """Test creating test providers with custom config."""
        custom_config = {"custom": "value"}
        config_provider, fs_provider, logger_provider = create_test_providers(config=custom_config)

        self.assertEqual(config_provider.extraction_config, custom_config)

    def test_create_test_providers_custom_files(self):
        """Test creating test providers with custom files."""
        custom_files = {"file1.txt": b"content1", "file2.txt": b"content2"}
        config_provider, fs_provider, logger_provider = create_test_providers(files=custom_files)

        self.assertIn("file1.txt", fs_provider.files)
        self.assertIn("file2.txt", fs_provider.files)

    def test_create_test_providers_production_logging(self):
        """Test creating test providers with production logging."""
        config_provider, fs_provider, logger_provider = create_test_providers(mock_logging=False)

        self.assertIsInstance(logger_provider, LoggerProvider)

    def test_create_test_providers_default_config_content(self):
        """Test that default config has expected content."""
        config_provider, _, _ = create_test_providers()
        config = config_provider.get_extraction_config()

        expected_keys = {"supported_formats", "text_encodings", "max_file_size_mb", "enable_logging"}
        self.assertTrue(expected_keys.issubset(config.keys()))

    def test_create_providers(self):
        """Test creating production providers."""
        config_provider, fs_provider, logger_provider = create_providers()

        self.assertIsInstance(config_provider, ConfigProvider)
        self.assertIsInstance(fs_provider, FileSystemProvider)
        self.assertIsInstance(logger_provider, LoggerProvider)

    def test_create_providers_custom_logger_name(self):
        """Test creating production providers with custom logger name."""
        logger_name = "custom.production.logger"
        config_provider, fs_provider, logger_provider = create_providers(logger_name)

        self.assertEqual(logger_provider.logger.name, logger_name)

    def test_provider_isolation(self):
        """Test that different provider sets are isolated."""
        providers1 = create_test_providers()
        providers2 = create_test_providers()

        # Should be different instances
        self.assertIsNot(providers1[0], providers2[0])
        self.assertIsNot(providers1[1], providers2[1])
        self.assertIsNot(providers1[2], providers2[2])


class TestProviderInterfaces(unittest.TestCase):
    """Test that all providers implement expected interfaces."""

    def test_config_provider_interface(self):
        """Test that config providers have consistent interface."""
        production = ConfigProvider()
        mock = MockConfigProvider({})

        # Both should have get_extraction_config method
        self.assertTrue(hasattr(production, "get_extraction_config"))
        self.assertTrue(hasattr(mock, "get_extraction_config"))
        self.assertTrue(callable(production.get_extraction_config))
        self.assertTrue(callable(mock.get_extraction_config))

    def test_file_system_provider_interface(self):
        """Test that file system providers have consistent interface."""
        production = FileSystemProvider()
        mock = MockFileSystemProvider()

        # Both should have required methods
        required_methods = ["file_exists", "get_file_size_mb", "open_binary", "open_text"]

        for method in required_methods:
            self.assertTrue(hasattr(production, method))
            self.assertTrue(hasattr(mock, method))
            self.assertTrue(callable(getattr(production, method)))
            self.assertTrue(callable(getattr(mock, method)))

    def test_logger_provider_interface(self):
        """Test that logger providers have consistent interface."""
        production = LoggerProvider()
        mock = MockLoggerProvider()

        # Both should have logging methods
        required_methods = ["info", "error"]

        for method in required_methods:
            self.assertTrue(hasattr(production, method))
            self.assertTrue(hasattr(mock, method))
            self.assertTrue(callable(getattr(production, method)))
            self.assertTrue(callable(getattr(mock, method)))


if __name__ == "__main__":
    unittest.main()