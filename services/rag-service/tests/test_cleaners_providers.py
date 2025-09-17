"""
Tests for preprocessing/cleaners_providers.py module.
Tests multilingual text cleaning providers for dependency injection patterns.
"""

import locale
import os
import unittest
from unittest.mock import Mock, patch

from src.preprocessing.cleaners_providers import (
    ConfigProvider,
    EnvironmentProvider,
    LoggerProvider,
    MockConfigProvider,
    MockEnvironmentProvider,
    MockLoggerProvider,
    create_config_provider,
    create_environment_provider,
    create_logger_provider,
    create_minimal_test_providers,
    create_multilingual_test_providers,
    create_providers,
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

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_language_config_success(self, mock_get_config):
        """Test successful retrieval of language configuration."""
        mock_config = {
            "diacritic_map": {"č": "c", "ć": "c"},
            "word_char_pattern": r"[a-zA-ZčćšžđĆČŠŽĐ]"
        }
        mock_get_config.return_value = mock_config

        result = self.provider.get_language_config("hr")

        mock_get_config.assert_called_once_with("text_processing", "hr")
        self.assertEqual(result, mock_config)

    @patch("src.utils.config_loader.get_cleaning_config")
    def test_get_cleaning_config_success(self, mock_get_config):
        """Test successful retrieval of cleaning configuration."""
        mock_config = {
            "multiple_whitespace": r"\s+",
            "min_meaningful_words": 3
        }
        mock_get_config.return_value = mock_config

        result = self.provider.get_cleaning_config()

        mock_get_config.assert_called_once()
        self.assertEqual(result, mock_config)

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_document_cleaning_config_success(self, mock_get_config):
        """Test successful retrieval of document cleaning configuration."""
        mock_config = {
            "header_footer_patterns": [r"^\s*Stranica\s+\d+\s*$"],
            "ocr_corrections": {"l": "i"}
        }
        mock_get_config.return_value = mock_config

        result = self.provider.get_document_cleaning_config("hr")

        mock_get_config.assert_called_once_with("document_cleaning", "hr")
        self.assertEqual(result, mock_config)

    @patch("src.utils.config_loader.get_language_specific_config")
    @patch("src.utils.config_loader.get_chunking_config")
    def test_get_chunking_config_success(self, mock_get_chunking, mock_get_language):
        """Test successful retrieval and merging of chunking configuration."""
        mock_main_config = {"sentence_ending_pattern": r"[.!?]+\s+", "min_sentence_length": 10}
        mock_language_config = {"min_sentence_length": 15}  # Override

        mock_get_chunking.return_value = mock_main_config
        mock_get_language.return_value = mock_language_config

        result = self.provider.get_chunking_config("hr")

        mock_get_chunking.assert_called_once()
        mock_get_language.assert_called_once_with("chunking", "hr")

        # Should merge with language-specific override
        expected = {"sentence_ending_pattern": r"[.!?]+\s+", "min_sentence_length": 15}
        self.assertEqual(result, expected)

    @patch("src.utils.config_loader.get_language_shared")
    def test_get_shared_language_config_success(self, mock_get_shared):
        """Test successful retrieval of shared language configuration."""
        mock_config = {
            "stopwords": {"words": ["i", "je", "da"]},
            "chars_pattern": r"[^\w\s.,!?:;()-]"
        }
        mock_get_shared.return_value = mock_config

        result = self.provider.get_shared_language_config("hr")

        mock_get_shared.assert_called_once_with("hr")
        self.assertEqual(result, mock_config)

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_config_error_handling(self, mock_get_config):
        """Test error handling when config loading fails."""
        mock_get_config.side_effect = Exception("Config file not found")

        with self.assertRaises(Exception) as cm:
            self.provider.get_language_config("hr")

        self.assertIn("Config file not found", str(cm.exception))

    def test_provider_immutability(self):
        """Test provider has no mutable state."""
        provider1 = ConfigProvider()
        provider2 = ConfigProvider()

        # Both instances should be functionally identical
        self.assertEqual(type(provider1), type(provider2))
        self.assertEqual(provider1.__dict__, provider2.__dict__)


class TestLoggerProvider(unittest.TestCase):
    """Test production logger provider."""

    def test_provider_creation_default_name(self):
        """Test provider creation with default logger name."""
        provider = LoggerProvider()
        self.assertIsInstance(provider, LoggerProvider)
        self.assertEqual(provider.logger.name, "src.preprocessing.cleaners_providers")

    def test_provider_creation_custom_name(self):
        """Test provider creation with custom logger name."""
        custom_name = "test.cleaner.logger"
        provider = LoggerProvider(custom_name)
        self.assertEqual(provider.logger.name, custom_name)

    def test_provider_creation_none_name(self):
        """Test provider creation with None name uses module name."""
        provider = LoggerProvider(None)
        self.assertEqual(provider.logger.name, "src.preprocessing.cleaners_providers")

    @patch("logging.getLogger")
    def test_debug_logging(self, mock_get_logger):
        """Test debug message logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        provider = LoggerProvider("test")
        provider.debug("Test debug message")

        mock_logger.debug.assert_called_once_with("Test debug message")

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


class TestEnvironmentProvider(unittest.TestCase):
    """Test production environment provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = EnvironmentProvider()
        # Store original values to restore after tests
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_provider_creation(self):
        """Test provider can be created."""
        provider = EnvironmentProvider()
        self.assertIsInstance(provider, EnvironmentProvider)

    def test_provider_attributes(self):
        """Test provider has no attributes by default (stateless)."""
        self.assertEqual(len(self.provider.__dict__), 0)

    def test_set_environment_variable(self):
        """Test setting environment variable."""
        key = "TEST_VAR"
        value = "test_value"

        # Ensure variable is not set initially
        if key in os.environ:
            del os.environ[key]

        self.provider.set_environment_variable(key, value)

        self.assertEqual(os.environ[key], value)

    def test_set_environment_variable_overwrite(self):
        """Test overwriting existing environment variable."""
        key = "TEST_VAR"
        original_value = "original"
        new_value = "new"

        os.environ[key] = original_value
        self.provider.set_environment_variable(key, new_value)

        self.assertEqual(os.environ[key], new_value)

    @patch("locale.setlocale")
    def test_set_locale(self, mock_setlocale):
        """Test setting locale."""
        category = locale.LC_ALL
        locale_name = "hr_HR.UTF-8"

        self.provider.set_locale(category, locale_name)

        mock_setlocale.assert_called_once_with(category, locale_name)

    @patch("locale.setlocale")
    def test_set_locale_error_handling(self, mock_setlocale):
        """Test locale setting error handling."""
        mock_setlocale.side_effect = locale.Error("Unsupported locale")

        with self.assertRaises(locale.Error):
            self.provider.set_locale(locale.LC_ALL, "invalid_locale")


class TestMockConfigProvider(unittest.TestCase):
    """Test mock configuration provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockConfigProvider()

    def test_provider_creation(self):
        """Test provider creation."""
        provider = MockConfigProvider()
        self.assertIsInstance(provider, MockConfigProvider)
        self.assertEqual(provider.language_configs, {})
        self.assertEqual(provider.cleaning_config, {})
        self.assertEqual(provider.document_cleaning_configs, {})
        self.assertEqual(provider.chunking_configs, {})
        self.assertEqual(provider.shared_language_configs, {})

    def test_set_and_get_language_config(self):
        """Test setting and getting language configuration."""
        language = "hr"
        config = {"diacritic_map": {"č": "c"}, "word_char_pattern": r"[a-zA-Z]"}

        self.provider.set_language_config(language, config)
        result = self.provider.get_language_config(language)

        self.assertEqual(result, config)

    def test_get_language_config_not_found(self):
        """Test getting non-existent language configuration."""
        with self.assertRaises(KeyError) as cm:
            self.provider.get_language_config("non_existent")

        self.assertIn("Mock language config 'non_existent' not found", str(cm.exception))

    def test_set_and_get_cleaning_config(self):
        """Test setting and getting cleaning configuration."""
        config = {"multiple_whitespace": r"\s+", "min_meaningful_words": 3}

        self.provider.set_cleaning_config(config)
        result = self.provider.get_cleaning_config()

        self.assertEqual(result, config)

    def test_get_cleaning_config_not_set(self):
        """Test getting cleaning configuration when not set."""
        with self.assertRaises(KeyError) as cm:
            self.provider.get_cleaning_config()

        self.assertIn("Mock cleaning config not set", str(cm.exception))

    def test_set_and_get_document_cleaning_config(self):
        """Test setting and getting document cleaning configuration."""
        language = "hr"
        config = {"header_footer_patterns": [r"^\s*Stranica\s+\d+\s*$"]}

        self.provider.set_document_cleaning_config(language, config)
        result = self.provider.get_document_cleaning_config(language)

        self.assertEqual(result, config)

    def test_get_document_cleaning_config_not_found(self):
        """Test getting non-existent document cleaning configuration."""
        with self.assertRaises(KeyError) as cm:
            self.provider.get_document_cleaning_config("non_existent")

        self.assertIn("Mock document cleaning config 'non_existent' not found", str(cm.exception))

    def test_set_and_get_chunking_config(self):
        """Test setting and getting chunking configuration."""
        language = "hr"
        config = {"sentence_ending_pattern": r"[.!?]+\s+", "min_sentence_length": 10}

        self.provider.set_chunking_config(language, config)
        result = self.provider.get_chunking_config(language)

        self.assertEqual(result, config)

    def test_get_chunking_config_not_found(self):
        """Test getting non-existent chunking configuration."""
        with self.assertRaises(KeyError) as cm:
            self.provider.get_chunking_config("non_existent")

        self.assertIn("Mock chunking config 'non_existent' not found", str(cm.exception))

    def test_set_and_get_shared_language_config(self):
        """Test setting and getting shared language configuration."""
        language = "hr"
        config = {"stopwords": {"words": ["i", "je"]}, "chars_pattern": r"[^\w\s]"}

        self.provider.set_shared_language_config(language, config)
        result = self.provider.get_shared_language_config(language)

        self.assertEqual(result, config)

    def test_get_shared_language_config_not_found(self):
        """Test getting non-existent shared language configuration."""
        with self.assertRaises(KeyError) as cm:
            self.provider.get_shared_language_config("non_existent")

        self.assertIn("Mock shared language config 'non_existent' not found", str(cm.exception))

    def test_multiple_language_isolation(self):
        """Test that different languages have isolated configurations."""
        hr_config = {"diacritic_map": {"č": "c"}}
        en_config = {"diacritic_map": {}}

        self.provider.set_language_config("hr", hr_config)
        self.provider.set_language_config("en", en_config)

        self.assertEqual(self.provider.get_language_config("hr"), hr_config)
        self.assertEqual(self.provider.get_language_config("en"), en_config)

    def test_config_mutability(self):
        """Test that configurations can be modified."""
        original_config = {"key": "original"}
        self.provider.set_language_config("test", original_config)

        # Modify the original config
        original_config["key"] = "modified"

        # Provider should reflect the change
        self.assertEqual(self.provider.get_language_config("test")["key"], "modified")


class TestMockLoggerProvider(unittest.TestCase):
    """Test mock logger provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockLoggerProvider()

    def test_provider_creation(self):
        """Test provider creation."""
        provider = MockLoggerProvider()
        self.assertIsInstance(provider, MockLoggerProvider)
        self.assertEqual(provider.debug_messages, [])
        self.assertEqual(provider.info_messages, [])
        self.assertEqual(provider.error_messages, [])

    def test_debug_logging(self):
        """Test debug message logging."""
        message = "Test debug message"
        self.provider.debug(message)

        self.assertEqual(len(self.provider.debug_messages), 1)
        self.assertEqual(self.provider.debug_messages[0], message)

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

    def test_multiple_messages_different_levels(self):
        """Test logging multiple messages at different levels."""
        self.provider.debug("Debug 1")
        self.provider.info("Info 1")
        self.provider.error("Error 1")
        self.provider.debug("Debug 2")

        self.assertEqual(len(self.provider.debug_messages), 2)
        self.assertEqual(len(self.provider.info_messages), 1)
        self.assertEqual(len(self.provider.error_messages), 1)

    def test_get_all_messages(self):
        """Test getting all logged messages."""
        self.provider.debug("Debug message")
        self.provider.info("Info message")
        self.provider.error("Error message")

        result = self.provider.get_all_messages()

        expected = {
            "debug": ["Debug message"],
            "info": ["Info message"],
            "error": ["Error message"]
        }
        self.assertEqual(result, expected)

    def test_clear_messages(self):
        """Test clearing all messages."""
        self.provider.debug("Debug message")
        self.provider.info("Info message")
        self.provider.error("Error message")

        self.provider.clear_messages()

        self.assertEqual(len(self.provider.debug_messages), 0)
        self.assertEqual(len(self.provider.info_messages), 0)
        self.assertEqual(len(self.provider.error_messages), 0)

    def test_message_order_preservation(self):
        """Test that message order is preserved within each level."""
        messages = ["First", "Second", "Third"]

        for msg in messages:
            self.provider.info(msg)

        self.assertEqual(self.provider.info_messages, messages)


class TestMockEnvironmentProvider(unittest.TestCase):
    """Test mock environment provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockEnvironmentProvider()

    def test_provider_creation(self):
        """Test provider creation."""
        provider = MockEnvironmentProvider()
        self.assertIsInstance(provider, MockEnvironmentProvider)
        self.assertEqual(provider.environment_variables, {})
        self.assertEqual(provider.locale_calls, [])

    def test_set_environment_variable(self):
        """Test setting environment variable."""
        key = "TEST_VAR"
        value = "test_value"

        self.provider.set_environment_variable(key, value)

        self.assertEqual(self.provider.environment_variables[key], value)

    def test_multiple_environment_variables(self):
        """Test setting multiple environment variables."""
        variables = {"VAR1": "value1", "VAR2": "value2", "VAR3": "value3"}

        for key, value in variables.items():
            self.provider.set_environment_variable(key, value)

        for key, value in variables.items():
            self.assertEqual(self.provider.environment_variables[key], value)

    def test_set_locale(self):
        """Test setting locale."""
        category = locale.LC_ALL
        locale_name = "hr_HR.UTF-8"

        self.provider.set_locale(category, locale_name)

        self.assertEqual(len(self.provider.locale_calls), 1)
        self.assertEqual(self.provider.locale_calls[0], (category, locale_name))

    def test_multiple_locale_calls(self):
        """Test multiple locale calls."""
        calls = [
            (locale.LC_ALL, "hr_HR.UTF-8"),
            (locale.LC_TIME, "en_US.UTF-8"),
            (locale.LC_NUMERIC, "C")
        ]

        for category, locale_name in calls:
            self.provider.set_locale(category, locale_name)

        self.assertEqual(self.provider.locale_calls, calls)

    def test_get_environment_variables_returns_copy(self):
        """Test that get_environment_variables returns a copy."""
        self.provider.set_environment_variable("TEST", "value")

        vars1 = self.provider.get_environment_variables()
        vars2 = self.provider.get_environment_variables()

        # Should be equal but different objects
        self.assertEqual(vars1, vars2)
        self.assertIsNot(vars1, vars2)

    def test_get_locale_calls_returns_copy(self):
        """Test that get_locale_calls returns a copy."""
        self.provider.set_locale(locale.LC_ALL, "hr_HR.UTF-8")

        calls1 = self.provider.get_locale_calls()
        calls2 = self.provider.get_locale_calls()

        # Should be equal but different objects
        self.assertEqual(calls1, calls2)
        self.assertIsNot(calls1, calls2)

    def test_clear_records(self):
        """Test clearing all recorded operations."""
        self.provider.set_environment_variable("TEST", "value")
        self.provider.set_locale(locale.LC_ALL, "hr_HR.UTF-8")

        self.provider.clear_records()

        self.assertEqual(len(self.provider.environment_variables), 0)
        self.assertEqual(len(self.provider.locale_calls), 0)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating providers."""

    def test_create_config_provider_production(self):
        """Test creating production config provider."""
        provider = create_config_provider()
        self.assertIsInstance(provider, ConfigProvider)

    def test_create_config_provider_mock_empty(self):
        """Test creating mock config provider with empty data."""
        provider = create_config_provider({})
        self.assertIsInstance(provider, MockConfigProvider)

    def test_create_config_provider_mock_with_data(self):
        """Test creating mock config provider with data."""
        mock_data = {
            "language_configs": {"hr": {"diacritic_map": {"č": "c"}}},
            "cleaning_config": {"multiple_whitespace": r"\s+"}
        }
        provider = create_config_provider(mock_data)

        self.assertIsInstance(provider, MockConfigProvider)
        self.assertEqual(provider.get_language_config("hr"), {"diacritic_map": {"č": "c"}})
        self.assertEqual(provider.get_cleaning_config(), {"multiple_whitespace": r"\s+"})

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

    def test_create_environment_provider_production(self):
        """Test creating production environment provider."""
        provider = create_environment_provider()
        self.assertIsInstance(provider, EnvironmentProvider)

    def test_create_environment_provider_mock(self):
        """Test creating mock environment provider."""
        provider = create_environment_provider(mock=True)
        self.assertIsInstance(provider, MockEnvironmentProvider)


class TestConvenienceBuilders(unittest.TestCase):
    """Test convenience provider builder functions."""

    def test_create_test_providers_defaults(self):
        """Test creating test providers with defaults."""
        config_provider, logger_provider, environment_provider = create_test_providers()

        self.assertIsInstance(config_provider, MockConfigProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)
        self.assertIsInstance(environment_provider, MockEnvironmentProvider)

    def test_create_test_providers_custom_language(self):
        """Test creating test providers with custom language."""
        config_provider, _, _ = create_test_providers(language="en")

        # Should have English configuration
        result = config_provider.get_language_config("en")
        self.assertIn("diacritic_map", result)

    def test_create_test_providers_custom_configs(self):
        """Test creating test providers with custom configurations."""
        custom_configs = {
            "language_configs": {"test": {"custom": "value"}},
            "cleaning_config": {"custom_key": "custom_value"}
        }
        config_provider, _, _ = create_test_providers(mock_configs=custom_configs)

        # Should merge with defaults
        result = config_provider.get_language_config("test")
        self.assertEqual(result["custom"], "value")

        cleaning_result = config_provider.get_cleaning_config()
        self.assertEqual(cleaning_result["custom_key"], "custom_value")

    def test_create_test_providers_production_logging(self):
        """Test creating test providers with production logging."""
        _, logger_provider, _ = create_test_providers(mock_logging=False)
        self.assertIsInstance(logger_provider, LoggerProvider)

    def test_create_test_providers_production_environment(self):
        """Test creating test providers with production environment."""
        _, _, environment_provider = create_test_providers(mock_environment=False)
        self.assertIsInstance(environment_provider, EnvironmentProvider)

    def test_create_providers(self):
        """Test creating production providers."""
        config_provider, logger_provider, environment_provider = create_providers()

        self.assertIsInstance(config_provider, ConfigProvider)
        self.assertIsInstance(logger_provider, LoggerProvider)
        self.assertIsInstance(environment_provider, EnvironmentProvider)

    def test_create_providers_custom_logger_name(self):
        """Test creating production providers with custom logger name."""
        logger_name = "custom.production.logger"
        _, logger_provider, _ = create_providers(logger_name)

        self.assertEqual(logger_provider.logger.name, logger_name)

    def test_create_multilingual_test_providers(self):
        """Test creating multilingual test providers."""
        config_provider, logger_provider, environment_provider = create_multilingual_test_providers()

        self.assertIsInstance(config_provider, MockConfigProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)
        self.assertIsInstance(environment_provider, MockEnvironmentProvider)

        # Should have multiple languages
        hr_config = config_provider.get_language_config("hr")
        en_config = config_provider.get_language_config("en")
        de_config = config_provider.get_language_config("de")

        self.assertIn("diacritic_map", hr_config)
        self.assertIn("diacritic_map", en_config)
        self.assertIn("diacritic_map", de_config)

    def test_create_minimal_test_providers(self):
        """Test creating minimal test providers."""
        config_provider, logger_provider, environment_provider = create_minimal_test_providers()

        self.assertIsInstance(config_provider, MockConfigProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)
        self.assertIsInstance(environment_provider, MockEnvironmentProvider)

        # Should have minimal test configuration
        test_config = config_provider.get_language_config("test")
        self.assertEqual(test_config["diacritic_map"], {"ç": "c"})

        cleaning_config = config_provider.get_cleaning_config()
        self.assertEqual(cleaning_config["min_meaningful_words"], 1)

    def test_provider_isolation(self):
        """Test that different provider sets are isolated."""
        providers1 = create_test_providers()
        providers2 = create_test_providers()

        # Should be different instances
        for i in range(3):
            self.assertIsNot(providers1[i], providers2[i])


class TestProviderInterfaces(unittest.TestCase):
    """Test that all providers implement expected interfaces."""

    def test_config_provider_interface(self):
        """Test that config providers have consistent interface."""
        production = ConfigProvider()
        mock = MockConfigProvider()

        # Both should have required methods
        required_methods = [
            "get_language_config",
            "get_cleaning_config",
            "get_document_cleaning_config",
            "get_chunking_config",
            "get_shared_language_config"
        ]

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
        required_methods = ["debug", "info", "error"]

        for method in required_methods:
            self.assertTrue(hasattr(production, method))
            self.assertTrue(hasattr(mock, method))
            self.assertTrue(callable(getattr(production, method)))
            self.assertTrue(callable(getattr(mock, method)))

    def test_environment_provider_interface(self):
        """Test that environment providers have consistent interface."""
        production = EnvironmentProvider()
        mock = MockEnvironmentProvider()

        # Both should have environment methods
        required_methods = ["set_environment_variable", "set_locale"]

        for method in required_methods:
            self.assertTrue(hasattr(production, method))
            self.assertTrue(hasattr(mock, method))
            self.assertTrue(callable(getattr(production, method)))
            self.assertTrue(callable(getattr(mock, method)))


class TestComplexScenarios(unittest.TestCase):
    """Test complex scenarios and edge cases."""

    def test_config_merging_behavior(self):
        """Test configuration merging scenarios."""
        override_configs = {
            "language_configs": {"en": {"override": "new", "additional": "extra"}},
            "cleaning_config": {"new_clean": "value"}
        }

        config_provider, _, _ = create_test_providers(language="hr", mock_configs=override_configs)

        # Should have merged configurations
        hr_config = config_provider.get_language_config("hr")  # Default language from create_test_providers
        en_config = config_provider.get_language_config("en")  # Added by override
        cleaning_config = config_provider.get_cleaning_config()

        # Check default language config (hr) from defaults
        self.assertIn("diacritic_map", hr_config)  # From defaults
        self.assertIn("word_char_pattern", hr_config)  # From defaults

        # Check override language config (en)
        self.assertEqual(en_config.get("override"), "new")
        self.assertEqual(en_config.get("additional"), "extra")

        # Check merged cleaning config (shallow merge of top-level keys)
        self.assertIn("multiple_whitespace", cleaning_config)  # From defaults
        self.assertIn("min_meaningful_words", cleaning_config)  # From defaults
        self.assertEqual(cleaning_config.get("new_clean"), "value")  # From override

    def test_multilingual_provider_completeness(self):
        """Test that multilingual providers have complete configurations."""
        config_provider, _, _ = create_multilingual_test_providers()

        languages = ["hr", "en", "de"]

        for language in languages:
            # Each language should have language config
            lang_config = config_provider.get_language_config(language)
            self.assertIn("diacritic_map", lang_config)
            self.assertIn("word_char_pattern", lang_config)
            self.assertIn("locale", lang_config)

            # Each language should have shared config
            shared_config = config_provider.get_shared_language_config(language)
            self.assertIn("stopwords", shared_config)
            self.assertIn("chars_pattern", shared_config)

    def test_provider_state_independence(self):
        """Test that providers maintain independent state."""
        provider1 = MockConfigProvider()
        provider2 = MockConfigProvider()

        # Set different configs
        provider1.set_language_config("test", {"provider": "1"})
        provider2.set_language_config("test", {"provider": "2"})

        # Should be independent
        self.assertEqual(provider1.get_language_config("test")["provider"], "1")
        self.assertEqual(provider2.get_language_config("test")["provider"], "2")


if __name__ == "__main__":
    unittest.main()