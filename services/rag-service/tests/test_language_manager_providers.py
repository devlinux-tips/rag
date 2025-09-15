"""
Comprehensive tests for language manager provider implementations.
Tests production and mock providers, factory functions, and configuration management.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Any

from src.utils.language_manager_providers import (
    MockConfigProvider,
    MockPatternProvider,
    MockLoggerProvider,
    ProductionConfigProvider,
    ProductionPatternProvider,
    StandardLoggerProvider,
    create_mock_setup,
    create_production_setup,
    create_test_settings,
    create_test_patterns,
    create_development_language_manager,
    create_test_language_manager,
)
from src.utils.language_manager import LanguageSettings, LanguagePatterns
from src.utils.config_validator import ConfigurationError


class TestMockConfigProvider(unittest.TestCase):
    """Test mock configuration provider functionality."""

    def test_init_with_no_settings(self):
        """Test initialization with no settings creates defaults."""
        provider = MockConfigProvider()

        self.assertIsInstance(provider.settings, LanguageSettings)
        self.assertIn("hr", provider.settings.supported_languages)
        self.assertIn("en", provider.settings.supported_languages)
        self.assertEqual(provider.settings.default_language, "hr")
        self.assertEqual(provider.call_history, [])

    def test_init_with_custom_settings(self):
        """Test initialization with custom settings."""
        custom_settings = LanguageSettings(
            supported_languages=["en", "de"],
            default_language="en",
            auto_detect=False,
            fallback_language="en",
            language_names={"en": "English", "de": "German"},
            embedding_model="custom-model",
            chunk_size=256,
            chunk_overlap=25,
        )

        provider = MockConfigProvider(custom_settings)

        self.assertEqual(provider.settings, custom_settings)
        self.assertEqual(provider.settings.supported_languages, ["en", "de"])
        self.assertEqual(provider.settings.default_language, "en")

    def test_set_settings(self):
        """Test setting new settings."""
        provider = MockConfigProvider()

        new_settings = LanguageSettings(
            supported_languages=["es"],
            default_language="es",
            auto_detect=True,
            fallback_language="es",
            language_names={"es": "Spanish"},
            embedding_model="spanish-model",
            chunk_size=1024,
            chunk_overlap=100,
        )

        provider.set_settings(new_settings)

        self.assertEqual(provider.settings, new_settings)
        self.assertEqual(provider.settings.default_language, "es")

    def test_get_language_settings_records_call(self):
        """Test get_language_settings records call history."""
        provider = MockConfigProvider()

        result = provider.get_language_settings()

        self.assertEqual(provider.call_history, ["get_language_settings"])
        self.assertIsInstance(result, LanguageSettings)

    def test_get_language_settings_multiple_calls(self):
        """Test multiple calls are recorded in history."""
        provider = MockConfigProvider()

        provider.get_language_settings()
        provider.get_language_settings()
        provider.get_language_settings()

        self.assertEqual(len(provider.call_history), 3)
        self.assertEqual(provider.call_history, ["get_language_settings"] * 3)

    def test_create_default_settings_structure(self):
        """Test default settings have expected structure."""
        provider = MockConfigProvider()
        settings = provider.settings

        # Check required fields
        self.assertIsInstance(settings.supported_languages, list)
        self.assertIsInstance(settings.language_names, dict)
        self.assertIsInstance(settings.embedding_model, str)
        self.assertIsInstance(settings.chunk_size, int)
        self.assertIsInstance(settings.chunk_overlap, int)

        # Check specific values
        self.assertEqual(settings.embedding_model, "BAAI/bge-m3")
        self.assertEqual(settings.chunk_size, 512)
        self.assertEqual(settings.chunk_overlap, 50)


class TestMockPatternProvider(unittest.TestCase):
    """Test mock pattern provider functionality."""

    def test_init_with_no_patterns(self):
        """Test initialization with no patterns creates defaults."""
        provider = MockPatternProvider()

        self.assertIsInstance(provider.patterns, LanguagePatterns)
        self.assertIn("hr", provider.patterns.detection_patterns)
        self.assertIn("en", provider.patterns.detection_patterns)
        self.assertEqual(provider.call_history, [])

    def test_init_with_custom_patterns(self):
        """Test initialization with custom patterns."""
        custom_patterns = LanguagePatterns(
            detection_patterns={"de": ["was", "wie", "wo"]},
            stopwords={"de": {"der", "die", "das"}}
        )

        provider = MockPatternProvider(custom_patterns)

        self.assertEqual(provider.patterns, custom_patterns)
        self.assertIn("de", provider.patterns.detection_patterns)

    def test_set_patterns(self):
        """Test setting new patterns."""
        provider = MockPatternProvider()

        new_patterns = LanguagePatterns(
            detection_patterns={"fr": ["que", "comment", "où"]},
            stopwords={"fr": {"le", "la", "les"}}
        )

        provider.set_patterns(new_patterns)

        self.assertEqual(provider.patterns, new_patterns)
        self.assertIn("fr", provider.patterns.detection_patterns)

    def test_add_detection_pattern(self):
        """Test adding detection patterns for a language."""
        provider = MockPatternProvider()

        provider.add_detection_pattern("de", ["was", "wie", "wo"])

        self.assertEqual(provider.patterns.detection_patterns["de"], ["was", "wie", "wo"])

    def test_add_stopwords(self):
        """Test adding stopwords for a language."""
        provider = MockPatternProvider()

        provider.add_stopwords("de", {"der", "die", "das"})

        self.assertEqual(provider.patterns.stopwords["de"], {"der", "die", "das"})

    def test_get_language_patterns_records_call(self):
        """Test get_language_patterns records call history."""
        provider = MockPatternProvider()

        result = provider.get_language_patterns()

        self.assertEqual(provider.call_history, ["get_language_patterns"])
        self.assertIsInstance(result, LanguagePatterns)

    def test_create_default_patterns_structure(self):
        """Test default patterns have expected structure."""
        provider = MockPatternProvider()
        patterns = provider.patterns

        # Check required fields
        self.assertIsInstance(patterns.detection_patterns, dict)
        self.assertIsInstance(patterns.stopwords, dict)

        # Check Croatian patterns
        self.assertIn("što", patterns.detection_patterns["hr"])
        self.assertIn("kako", patterns.detection_patterns["hr"])
        self.assertIn("i", patterns.stopwords["hr"])

        # Check English patterns
        self.assertIn("what", patterns.detection_patterns["en"])
        self.assertIn("how", patterns.detection_patterns["en"])
        self.assertIn("a", patterns.stopwords["en"])


class TestMockLoggerProvider(unittest.TestCase):
    """Test mock logger provider functionality."""

    def test_init_creates_empty_message_storage(self):
        """Test initialization creates empty message storage."""
        logger = MockLoggerProvider()

        self.assertEqual(logger.messages["info"], [])
        self.assertEqual(logger.messages["debug"], [])
        self.assertEqual(logger.messages["warning"], [])
        self.assertEqual(logger.messages["error"], [])

    def test_info_captures_message(self):
        """Test info logging captures message."""
        logger = MockLoggerProvider()

        logger.info("test info message")

        self.assertEqual(logger.messages["info"], ["test info message"])
        self.assertEqual(logger.messages["debug"], [])

    def test_debug_captures_message(self):
        """Test debug logging captures message."""
        logger = MockLoggerProvider()

        logger.debug("test debug message")

        self.assertEqual(logger.messages["debug"], ["test debug message"])
        self.assertEqual(logger.messages["info"], [])

    def test_warning_captures_message(self):
        """Test warning logging captures message."""
        logger = MockLoggerProvider()

        logger.warning("test warning message")

        self.assertEqual(logger.messages["warning"], ["test warning message"])

    def test_error_captures_message(self):
        """Test error logging captures message."""
        logger = MockLoggerProvider()

        logger.error("test error message")

        self.assertEqual(logger.messages["error"], ["test error message"])

    def test_multiple_messages_captured_in_order(self):
        """Test multiple messages are captured in order."""
        logger = MockLoggerProvider()

        logger.info("first info")
        logger.debug("first debug")
        logger.info("second info")
        logger.error("first error")

        self.assertEqual(logger.messages["info"], ["first info", "second info"])
        self.assertEqual(logger.messages["debug"], ["first debug"])
        self.assertEqual(logger.messages["error"], ["first error"])

    def test_clear_messages_removes_all(self):
        """Test clear_messages removes all captured messages."""
        logger = MockLoggerProvider()

        # Add some messages
        logger.info("test info")
        logger.debug("test debug")
        logger.warning("test warning")
        logger.error("test error")

        # Clear all
        logger.clear_messages()

        # Verify all cleared
        self.assertEqual(logger.messages["info"], [])
        self.assertEqual(logger.messages["debug"], [])
        self.assertEqual(logger.messages["warning"], [])
        self.assertEqual(logger.messages["error"], [])

    def test_get_messages_returns_all_when_no_level(self):
        """Test get_messages returns all messages when no level specified."""
        logger = MockLoggerProvider()

        logger.info("info msg")
        logger.debug("debug msg")
        logger.error("error msg")

        result = logger.get_messages()

        expected = {
            "info": ["info msg"],
            "debug": ["debug msg"],
            "warning": [],
            "error": ["error msg"]
        }
        self.assertEqual(result, expected)

    def test_get_messages_returns_specific_level(self):
        """Test get_messages returns specific level messages."""
        logger = MockLoggerProvider()

        logger.info("info msg 1")
        logger.info("info msg 2")
        logger.debug("debug msg")

        result = logger.get_messages("info")

        self.assertEqual(result, ["info msg 1", "info msg 2"])

    def test_get_messages_returns_empty_list_for_unknown_level(self):
        """Test get_messages returns empty list for unknown level."""
        logger = MockLoggerProvider()

        result = logger.get_messages("unknown")

        self.assertEqual(result, [])


class TestProductionConfigProvider(unittest.TestCase):
    """Test production configuration provider functionality."""

    def test_init_creates_cache(self):
        """Test initialization creates settings cache."""
        provider = ProductionConfigProvider()

        self.assertIsNone(provider._settings_cache)

    def test_get_language_settings_caches_result(self):
        """Test get_language_settings caches the result."""
        # Mock the config loading system
        mock_get_supported_languages = MagicMock(return_value=["hr", "en"])
        mock_load_config = MagicMock(return_value={
            "languages": {"default": "hr", "auto_detect": True, "names": {"hr": "Croatian", "en": "English"}},
            "embeddings": {"model_name": "test-model"}
        })
        mock_get_shared_config = MagicMock(return_value={"default_chunk_size": 512, "default_chunk_overlap": 50})

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_supported_languages=mock_get_supported_languages,
                load_config=mock_load_config,
                get_shared_config=mock_get_shared_config
            )
        }):
            provider = ProductionConfigProvider()

            # First call should load and cache
            result1 = provider.get_language_settings()
            # Second call should return cached result
            result2 = provider.get_language_settings()

            # Should be same object (cached)
            self.assertIs(result1, result2)
            # Mock should only be called once due to caching
            mock_load_config.assert_called_once()

    def test_load_settings_from_system_success(self):
        """Test successful loading of settings from system."""
        # Mock the config loading system
        mock_get_supported_languages = MagicMock(return_value=["hr", "en"])
        mock_load_config = MagicMock(return_value={
            "languages": {
                "default": "hr",
                "auto_detect": True,
                "names": {"hr": "Croatian", "en": "English"}
            },
            "embeddings": {"model_name": "BAAI/bge-m3"}
        })
        mock_get_shared_config = MagicMock(return_value={
            "default_chunk_size": 512,
            "default_chunk_overlap": 50
        })

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_supported_languages=mock_get_supported_languages,
                load_config=mock_load_config,
                get_shared_config=mock_get_shared_config
            )
        }):
            provider = ProductionConfigProvider()
            result = provider.get_language_settings()

            self.assertIsInstance(result, LanguageSettings)
            self.assertEqual(result.supported_languages, ["hr", "en"])
            self.assertEqual(result.default_language, "hr")
            self.assertEqual(result.embedding_model, "BAAI/bge-m3")

    def test_load_settings_from_system_handles_exceptions(self):
        """Test that exceptions from config system are properly handled."""
        mock_config_loader = MagicMock()
        mock_config_loader.get_supported_languages.side_effect = ValueError("Config error")

        with patch.dict('sys.modules', {'src.utils.config_loader': mock_config_loader}):
            provider = ProductionConfigProvider()

            with self.assertRaises(ConfigurationError) as context:
                provider.get_language_settings()

            self.assertIn("Failed to load language settings", str(context.exception))


class TestProductionPatternProvider(unittest.TestCase):
    """Test production pattern provider functionality."""

    def test_init_creates_cache(self):
        """Test initialization creates patterns cache."""
        provider = ProductionPatternProvider()

        self.assertIsNone(provider._patterns_cache)

    def test_get_language_patterns_caches_result(self):
        """Test get_language_patterns caches the result."""
        # Mock the config loading system
        mock_get_supported_languages = MagicMock(return_value=["hr"])
        mock_get_language_config = MagicMock(return_value={
            "shared": {
                "question_patterns": {"wh": ["što", "kako"]},
                "stopwords": {"words": ["i", "u", "na"]}
            }
        })

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_supported_languages=mock_get_supported_languages,
                get_language_config=mock_get_language_config
            )
        }):
            provider = ProductionPatternProvider()

            # First call should load and cache
            result1 = provider.get_language_patterns()
            # Second call should return cached result
            result2 = provider.get_language_patterns()

            # Should be same object (cached)
            self.assertIs(result1, result2)

    def test_load_patterns_from_system_success(self):
        """Test successful loading of patterns from system."""
        mock_get_supported_languages = MagicMock(return_value=["hr", "en"])
        mock_get_language_config = MagicMock()

        def mock_lang_config(lang_code):
            if lang_code == "hr":
                return {
                    "shared": {
                        "question_patterns": {"wh": ["što", "kako", "gdje"]},
                        "stopwords": {"words": ["i", "u", "na", "za", "je"]}
                    }
                }
            elif lang_code == "en":
                return {
                    "shared": {
                        "question_patterns": {"wh": ["what", "how", "where"]},
                        "stopwords": {"words": ["a", "and", "the", "of", "to"]}
                    }
                }
            return {}

        mock_get_language_config.side_effect = mock_lang_config

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_supported_languages=mock_get_supported_languages,
                get_language_config=mock_get_language_config
            )
        }):
            provider = ProductionPatternProvider()
            result = provider.get_language_patterns()

            self.assertIsInstance(result, LanguagePatterns)
            self.assertIn("hr", result.detection_patterns)
            self.assertIn("en", result.detection_patterns)
            self.assertIn("što", result.detection_patterns["hr"])
            self.assertIn("what", result.detection_patterns["en"])

    def test_load_patterns_handles_missing_language_config(self):
        """Test handling of missing language configuration."""
        mock_get_supported_languages = MagicMock(return_value=["hr"])
        mock_get_language_config = MagicMock(side_effect=ValueError("Missing config"))

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_supported_languages=mock_get_supported_languages,
                get_language_config=mock_get_language_config
            )
        }):
            provider = ProductionPatternProvider()

            with self.assertRaises(ConfigurationError) as context:
                provider.get_language_patterns()

            self.assertIn("Failed to load patterns for language 'hr'", str(context.exception))

    def test_load_patterns_handles_system_exceptions(self):
        """Test that exceptions from pattern system are properly handled."""
        mock_get_supported_languages = MagicMock(side_effect=RuntimeError("System error"))

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_supported_languages=mock_get_supported_languages
            )
        }):
            provider = ProductionPatternProvider()

            with self.assertRaises(ConfigurationError) as context:
                provider.get_language_patterns()

            self.assertIn("Failed to load language patterns", str(context.exception))


class TestStandardLoggerProvider(unittest.TestCase):
    """Test standard logger provider functionality."""

    def test_init_with_default_logger_name(self):
        """Test initialization with default logger name."""
        provider = StandardLoggerProvider()

        self.assertIsNotNone(provider.logger)
        # Logger name should include the module name
        self.assertIn("language_manager_providers", provider.logger.name)

    def test_init_with_custom_logger_name(self):
        """Test initialization with custom logger name."""
        provider = StandardLoggerProvider("custom.logger.name")

        self.assertEqual(provider.logger.name, "custom.logger.name")

    def test_logging_methods_call_logger(self):
        """Test that logging methods properly call the underlying logger."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            provider = StandardLoggerProvider("test.logger")

            # Test all logging methods
            provider.info("info message")
            provider.debug("debug message")
            provider.warning("warning message")
            provider.error("error message")

            # Verify calls to underlying logger
            mock_logger.info.assert_called_once_with("info message")
            mock_logger.debug.assert_called_once_with("debug message")
            mock_logger.warning.assert_called_once_with("warning message")
            mock_logger.error.assert_called_once_with("error message")


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for provider creation."""

    def test_create_mock_setup_with_defaults(self):
        """Test create_mock_setup with default parameters."""
        config_provider, pattern_provider, logger_provider = create_mock_setup()

        self.assertIsInstance(config_provider, MockConfigProvider)
        self.assertIsInstance(pattern_provider, MockPatternProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)

    def test_create_mock_setup_with_custom_settings(self):
        """Test create_mock_setup with custom settings."""
        custom_settings = LanguageSettings(
            supported_languages=["de"],
            default_language="de",
            auto_detect=False,
            fallback_language="de",
            language_names={"de": "German"},
            embedding_model="german-model",
            chunk_size=256,
            chunk_overlap=25,
        )

        config_provider, pattern_provider, logger_provider = create_mock_setup(
            settings=custom_settings
        )

        self.assertEqual(config_provider.settings, custom_settings)

    def test_create_mock_setup_with_custom_patterns(self):
        """Test create_mock_setup with custom patterns."""
        custom_patterns = LanguagePatterns(
            detection_patterns={"fr": ["que", "comment"]},
            stopwords={"fr": {"le", "la"}}
        )

        config_provider, pattern_provider, logger_provider = create_mock_setup(
            patterns=custom_patterns
        )

        self.assertEqual(pattern_provider.patterns, custom_patterns)

    def test_create_mock_setup_with_custom_detection_patterns(self):
        """Test create_mock_setup with custom detection patterns."""
        custom_patterns = {"es": ["qué", "cómo", "dónde"]}

        config_provider, pattern_provider, logger_provider = create_mock_setup(
            custom_patterns=custom_patterns
        )

        self.assertEqual(pattern_provider.patterns.detection_patterns["es"], ["qué", "cómo", "dónde"])

    def test_create_mock_setup_with_custom_stopwords(self):
        """Test create_mock_setup with custom stopwords."""
        custom_stopwords = {"es": {"el", "la", "los", "las"}}

        config_provider, pattern_provider, logger_provider = create_mock_setup(
            custom_stopwords=custom_stopwords
        )

        self.assertEqual(pattern_provider.patterns.stopwords["es"], {"el", "la", "los", "las"})

    def test_create_production_setup_with_defaults(self):
        """Test create_production_setup with default logger name."""
        config_provider, pattern_provider, logger_provider = create_production_setup()

        self.assertIsInstance(config_provider, ProductionConfigProvider)
        self.assertIsInstance(pattern_provider, ProductionPatternProvider)
        self.assertIsInstance(logger_provider, StandardLoggerProvider)

    def test_create_production_setup_with_custom_logger_name(self):
        """Test create_production_setup with custom logger name."""
        config_provider, pattern_provider, logger_provider = create_production_setup(
            logger_name="custom.logger"
        )

        self.assertEqual(logger_provider.logger.name, "custom.logger")

    def test_create_test_settings_with_defaults(self):
        """Test create_test_settings with default parameters."""
        settings = create_test_settings()

        self.assertEqual(settings.supported_languages, ["hr", "en", "multilingual"])
        self.assertEqual(settings.default_language, "hr")
        self.assertTrue(settings.auto_detect)
        self.assertEqual(settings.embedding_model, "BAAI/bge-m3")

    def test_create_test_settings_with_custom_parameters(self):
        """Test create_test_settings with custom parameters."""
        settings = create_test_settings(
            supported_languages=["de", "fr"],
            default_language="de",
            auto_detect=False,
            embedding_model="custom-model"
        )

        self.assertEqual(settings.supported_languages, ["de", "fr"])
        self.assertEqual(settings.default_language, "de")
        self.assertFalse(settings.auto_detect)
        self.assertEqual(settings.embedding_model, "custom-model")

    def test_create_test_settings_language_names_mapping(self):
        """Test create_test_settings properly maps language names."""
        settings = create_test_settings(supported_languages=["hr", "en", "fr"])

        self.assertEqual(settings.language_names["hr"], "Croatian")
        self.assertEqual(settings.language_names["en"], "English")
        self.assertEqual(settings.language_names["fr"], "FR")  # Unknown languages get uppercased

    def test_create_test_patterns_with_defaults(self):
        """Test create_test_patterns with default parameters."""
        patterns = create_test_patterns()

        self.assertIn("hr", patterns.detection_patterns)
        self.assertIn("en", patterns.detection_patterns)
        self.assertIn("multilingual", patterns.detection_patterns)

        self.assertIn("što", patterns.detection_patterns["hr"])
        self.assertIn("what", patterns.detection_patterns["en"])

    def test_create_test_patterns_with_custom_parameters(self):
        """Test create_test_patterns with custom parameters."""
        custom_detection = {"de": ["was", "wie"]}
        custom_stopwords = {"de": {"der", "die"}}

        patterns = create_test_patterns(
            detection_patterns=custom_detection,
            stopwords=custom_stopwords
        )

        self.assertEqual(patterns.detection_patterns, custom_detection)
        self.assertEqual(patterns.stopwords, custom_stopwords)


class TestIntegrationHelpers(unittest.TestCase):
    """Test integration helper functions."""

    def test_create_development_language_manager(self):
        """Test create_development_language_manager creates manager with production setup."""
        # Mock the language manager creation
        mock_manager = MagicMock()

        with patch('src.utils.language_manager.create_language_manager') as mock_create:
            mock_create.return_value = mock_manager

            result = create_development_language_manager()

            self.assertEqual(result, mock_manager)
            mock_create.assert_called_once()

            # Check that it was called with production providers
            call_kwargs = mock_create.call_args[1]
            self.assertIsInstance(call_kwargs["config_provider"], ProductionConfigProvider)
            self.assertIsInstance(call_kwargs["pattern_provider"], ProductionPatternProvider)
            self.assertIsInstance(call_kwargs["logger_provider"], StandardLoggerProvider)

    def test_create_test_language_manager_with_defaults(self):
        """Test create_test_language_manager with default parameters."""
        # Mock the language manager creation
        mock_manager = MagicMock()

        with patch('src.utils.language_manager.create_language_manager') as mock_create:
            mock_create.return_value = mock_manager

            result, providers = create_test_language_manager()

            self.assertEqual(result, mock_manager)
            self.assertEqual(len(providers), 3)  # config, pattern, logger providers

            config_provider, pattern_provider, logger_provider = providers
            self.assertIsInstance(config_provider, MockConfigProvider)
            self.assertIsInstance(pattern_provider, MockPatternProvider)
            self.assertIsInstance(logger_provider, MockLoggerProvider)

    def test_create_test_language_manager_with_custom_settings(self):
        """Test create_test_language_manager with custom settings."""
        custom_settings = LanguageSettings(
            supported_languages=["de"],
            default_language="de",
            auto_detect=False,
            fallback_language="de",
            language_names={"de": "German"},
            embedding_model="german-model",
            chunk_size=256,
            chunk_overlap=25,
        )

        mock_manager = MagicMock()

        with patch('src.utils.language_manager.create_language_manager') as mock_create:
            mock_create.return_value = mock_manager

            result, providers = create_test_language_manager(settings=custom_settings)

            config_provider, pattern_provider, logger_provider = providers
            self.assertEqual(config_provider.settings, custom_settings)

    def test_create_test_language_manager_with_custom_patterns(self):
        """Test create_test_language_manager with custom patterns."""
        custom_patterns = LanguagePatterns(
            detection_patterns={"it": ["che", "come"]},
            stopwords={"it": {"il", "la"}}
        )

        mock_manager = MagicMock()

        with patch('src.utils.language_manager.create_language_manager') as mock_create:
            mock_create.return_value = mock_manager

            result, providers = create_test_language_manager(patterns=custom_patterns)

            config_provider, pattern_provider, logger_provider = providers
            self.assertEqual(pattern_provider.patterns, custom_patterns)


if __name__ == "__main__":
    unittest.main()