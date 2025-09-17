"""
Comprehensive tests for categorization provider implementations.
Tests production and mock providers, factory functions, and configuration management.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Any

from src.retrieval.categorization_providers import (
    ConfigProvider,
    MockConfigProvider,
    NoOpLoggerProvider,
    MockLoggerProvider,
    create_config_provider,
    create_test_categorization_setup,
    create_minimal_test_config,
    create_complex_test_config,
)


class TestConfigProvider(unittest.TestCase):
    """Test production configuration provider functionality."""

    def test_init_imports_config_loader(self):
        """Test initialization properly imports config_loader."""
        provider = ConfigProvider()
        # Verify that _config_loader is set and has the expected method
        self.assertTrue(hasattr(provider._config_loader, 'get_language_specific_config'))

    def test_get_categorization_config_success(self):
        """Test successful retrieval of categorization configuration."""
        # Setup
        expected_config = {
            "categories": {"general": {"priority": 1}},
            "patterns": {"general": ["test"]}
        }

        provider = ConfigProvider()

        # Mock the config loader after initialization
        mock_config_loader = MagicMock()
        mock_config_loader.get_language_specific_config.return_value = expected_config
        provider._config_loader = mock_config_loader

        # Execute
        result = provider.get_categorization_config("hr")

        # Verify
        self.assertEqual(result, expected_config)
        mock_config_loader.get_language_specific_config.assert_called_once_with("categorization", "hr")

    def test_get_categorization_config_different_languages(self):
        """Test configuration retrieval for different languages."""
        # Setup
        hr_config = {"patterns": {"cultural": ["hrvatski"]}}
        en_config = {"patterns": {"cultural": ["english"]}}

        def mock_get_config(section, language):
            if language == "hr":
                return hr_config
            elif language == "en":
                return en_config
            return {}

        provider = ConfigProvider()

        # Mock the config loader after initialization
        mock_config_loader = MagicMock()
        mock_config_loader.get_language_specific_config.side_effect = mock_get_config
        provider._config_loader = mock_config_loader

        # Execute and verify
        self.assertEqual(provider.get_categorization_config("hr"), hr_config)
        self.assertEqual(provider.get_categorization_config("en"), en_config)

    def test_get_categorization_config_handles_exceptions(self):
        """Test that exceptions from config_loader are propagated."""
        provider = ConfigProvider()

        # Mock the config loader after initialization
        mock_config_loader = MagicMock()
        mock_config_loader.get_language_specific_config.side_effect = ValueError("Config error")
        provider._config_loader = mock_config_loader

        # Execute and verify exception is raised
        with self.assertRaises(ValueError) as context:
            provider.get_categorization_config("hr")

        self.assertEqual(str(context.exception), "Config error")


class TestMockConfigProvider(unittest.TestCase):
    """Test mock configuration provider functionality."""

    def test_init_with_no_mock_configs(self):
        """Test initialization with no mock configurations."""
        provider = MockConfigProvider()
        self.assertEqual(provider.mock_configs, {})
        self.assertIsInstance(provider._default_config, dict)
        self.assertIn("categories", provider._default_config)
        self.assertIn("patterns", provider._default_config)

    def test_init_with_mock_configs(self):
        """Test initialization with provided mock configurations."""
        custom_configs = {"categorization_hr": {"test": "value"}}
        provider = MockConfigProvider(custom_configs)
        self.assertEqual(provider.mock_configs, custom_configs)

    def test_set_categorization_config(self):
        """Test setting categorization configuration for language."""
        provider = MockConfigProvider()
        config_data = {"categories": {"test": {"priority": 1}}}

        provider.set_categorization_config("hr", config_data)

        self.assertEqual(provider.mock_configs["categorization_hr"], config_data)

    def test_get_categorization_config_with_set_config(self):
        """Test getting configuration when specific config is set."""
        provider = MockConfigProvider()
        config_data = {"test_key": "test_value"}

        provider.set_categorization_config("hr", config_data)
        result = provider.get_categorization_config("hr")

        self.assertEqual(result, config_data)

    def test_get_categorization_config_falls_back_to_language_default(self):
        """Test getting configuration falls back to language default."""
        provider = MockConfigProvider()

        # No config set, should get language default
        result = provider.get_categorization_config("hr")

        self.assertIsInstance(result, dict)
        self.assertIn("categories", result)
        self.assertIn("patterns", result)
        # Should have Croatian-specific additions
        self.assertIn("hrvatska", result["patterns"]["cultural"])

    def test_get_categorization_config_english_default(self):
        """Test getting configuration for English gets English defaults."""
        provider = MockConfigProvider()

        result = provider.get_categorization_config("en")

        self.assertIsInstance(result, dict)
        # Should have English-specific additions
        self.assertIn("england", result["patterns"]["cultural"])
        self.assertIn("english_test", result["cultural_keywords"])

    def test_get_categorization_config_unknown_language_gets_base_default(self):
        """Test getting configuration for unknown language gets base default."""
        provider = MockConfigProvider()

        result = provider.get_categorization_config("unknown")

        self.assertIsInstance(result, dict)
        self.assertIn("categories", result)
        self.assertIn("patterns", result)
        # Should not have language-specific additions
        self.assertNotIn("hrvatska", result["patterns"]["cultural"])
        self.assertNotIn("england", result["patterns"]["cultural"])

    def test_create_default_test_config_structure(self):
        """Test default test configuration has expected structure."""
        provider = MockConfigProvider()
        config = provider._default_config

        # Check all required sections exist
        self.assertIn("categories", config)
        self.assertIn("patterns", config)
        self.assertIn("cultural_keywords", config)
        self.assertIn("complexity_thresholds", config)
        self.assertIn("retrieval_strategies", config)

        # Check categories structure
        self.assertIn("general", config["categories"])
        self.assertIn("priority", config["categories"]["general"])

        # Check patterns structure
        self.assertIn("general", config["patterns"])
        self.assertIsInstance(config["patterns"]["general"], list)

        # Check complexity thresholds
        self.assertIn("simple", config["complexity_thresholds"])
        self.assertIn("complex", config["complexity_thresholds"])

    def test_get_language_default_config_croatian_extensions(self):
        """Test Croatian language default gets Croatian-specific extensions."""
        provider = MockConfigProvider()

        result = provider._get_language_default_config("hr")

        # Should have Croatian cultural patterns
        cultural_patterns = result["patterns"]["cultural"]
        self.assertIn("hrvatska", cultural_patterns)
        self.assertIn("dubrovnik", cultural_patterns)
        self.assertIn("split", cultural_patterns)
        self.assertIn("zagreb", cultural_patterns)

        # Should have Croatian cultural keywords
        self.assertIn("croatian_test", result["cultural_keywords"])
        croatian_keywords = result["cultural_keywords"]["croatian_test"]
        self.assertIn("test_hr", croatian_keywords)
        self.assertIn("test_croatia", croatian_keywords)

    def test_get_language_default_config_english_extensions(self):
        """Test English language default gets English-specific extensions."""
        provider = MockConfigProvider()

        result = provider._get_language_default_config("en")

        # Should have English cultural patterns
        cultural_patterns = result["patterns"]["cultural"]
        self.assertIn("england", cultural_patterns)
        self.assertIn("london", cultural_patterns)
        self.assertIn("british", cultural_patterns)
        self.assertIn("american", cultural_patterns)

        # Should have English cultural keywords
        self.assertIn("english_test", result["cultural_keywords"])
        english_keywords = result["cultural_keywords"]["english_test"]
        self.assertIn("test_en", english_keywords)
        self.assertIn("test_uk", english_keywords)

    def test_get_language_default_config_preserves_base_config(self):
        """Test language defaults preserve base configuration."""
        provider = MockConfigProvider()

        hr_config = provider._get_language_default_config("hr")
        en_config = provider._get_language_default_config("en")

        # Both should have same base structure
        self.assertEqual(hr_config["categories"], en_config["categories"])
        self.assertEqual(hr_config["complexity_thresholds"], en_config["complexity_thresholds"])
        self.assertEqual(hr_config["retrieval_strategies"], en_config["retrieval_strategies"])


class TestNoOpLoggerProvider(unittest.TestCase):
    """Test no-operation logger provider functionality."""

    def test_info_does_nothing(self):
        """Test info logging does nothing."""
        logger = NoOpLoggerProvider()
        # Should not raise any exceptions
        logger.info("test message")

    def test_debug_does_nothing(self):
        """Test debug logging does nothing."""
        logger = NoOpLoggerProvider()
        # Should not raise any exceptions
        logger.debug("test message")

    def test_warning_does_nothing(self):
        """Test warning logging does nothing."""
        logger = NoOpLoggerProvider()
        # Should not raise any exceptions
        logger.warning("test message")

    def test_all_methods_accept_any_string(self):
        """Test all logging methods accept any string message."""
        logger = NoOpLoggerProvider()
        messages = ["", "test", "very long message with special chars !@#$%"]

        for message in messages:
            logger.info(message)
            logger.debug(message)
            logger.warning(message)


class TestMockLoggerProvider(unittest.TestCase):
    """Test mock logger provider functionality."""

    def test_init_creates_empty_message_storage(self):
        """Test initialization creates empty message storage."""
        logger = MockLoggerProvider()

        self.assertEqual(logger.messages["info"], [])
        self.assertEqual(logger.messages["debug"], [])
        self.assertEqual(logger.messages["warning"], [])

    def test_info_captures_message(self):
        """Test info logging captures message."""
        logger = MockLoggerProvider()

        logger.info("test info message")

        self.assertEqual(logger.messages["info"], ["test info message"])
        self.assertEqual(logger.messages["debug"], [])
        self.assertEqual(logger.messages["warning"], [])

    def test_debug_captures_message(self):
        """Test debug logging captures message."""
        logger = MockLoggerProvider()

        logger.debug("test debug message")

        self.assertEqual(logger.messages["info"], [])
        self.assertEqual(logger.messages["debug"], ["test debug message"])
        self.assertEqual(logger.messages["warning"], [])

    def test_warning_captures_message(self):
        """Test warning logging captures message."""
        logger = MockLoggerProvider()

        logger.warning("test warning message")

        self.assertEqual(logger.messages["info"], [])
        self.assertEqual(logger.messages["debug"], [])
        self.assertEqual(logger.messages["warning"], ["test warning message"])

    def test_multiple_messages_captured_in_order(self):
        """Test multiple messages are captured in order."""
        logger = MockLoggerProvider()

        logger.info("first info")
        logger.debug("first debug")
        logger.info("second info")
        logger.warning("first warning")

        self.assertEqual(logger.messages["info"], ["first info", "second info"])
        self.assertEqual(logger.messages["debug"], ["first debug"])
        self.assertEqual(logger.messages["warning"], ["first warning"])

    def test_clear_messages_removes_all(self):
        """Test clear_messages removes all captured messages."""
        logger = MockLoggerProvider()

        # Add some messages
        logger.info("test info")
        logger.debug("test debug")
        logger.warning("test warning")

        # Clear all
        logger.clear_messages()

        # Verify all cleared
        self.assertEqual(logger.messages["info"], [])
        self.assertEqual(logger.messages["debug"], [])
        self.assertEqual(logger.messages["warning"], [])

    def test_get_messages_returns_all_when_no_level(self):
        """Test get_messages returns all messages when no level specified."""
        logger = MockLoggerProvider()

        logger.info("info msg")
        logger.debug("debug msg")

        result = logger.get_messages()

        expected = {
            "info": ["info msg"],
            "debug": ["debug msg"],
            "warning": []
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

    def test_get_messages_after_clear(self):
        """Test get_messages after clearing returns empty."""
        logger = MockLoggerProvider()

        logger.info("test message")
        logger.clear_messages()

        result = logger.get_messages("info")

        self.assertEqual(result, [])


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for provider creation."""

    def test_create_config_provider_mock_mode(self):
        """Test create_config_provider with mock mode."""
        provider = create_config_provider(use_mock=True)

        self.assertIsInstance(provider, MockConfigProvider)

    @patch('src.retrieval.categorization_providers.ConfigProvider')
    def test_create_config_provider_production_mode(self, mock_production_class):
        """Test create_config_provider with production mode."""
        mock_instance = MagicMock()
        mock_production_class.return_value = mock_instance

        provider = create_config_provider(use_mock=False)

        self.assertEqual(provider, mock_instance)
        mock_production_class.assert_called_once()

    def test_create_config_provider_default_is_production(self):
        """Test create_config_provider defaults to production mode."""
        with patch('src.retrieval.categorization_providers.ConfigProvider') as mock_production:
            mock_instance = MagicMock()
            mock_production.return_value = mock_instance

            provider = create_config_provider()

            self.assertEqual(provider, mock_instance)

    def test_create_test_categorization_setup_default_language(self):
        """Test create_test_categorization_setup with default language."""
        config_provider, logger_provider = create_test_categorization_setup()

        self.assertIsInstance(config_provider, MockConfigProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)

    def test_create_test_categorization_setup_custom_language(self):
        """Test create_test_categorization_setup with custom language."""
        config_provider, logger_provider = create_test_categorization_setup(language="en")

        self.assertIsInstance(config_provider, MockConfigProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)

    def test_create_test_categorization_setup_with_custom_config(self):
        """Test create_test_categorization_setup with custom configuration."""
        custom_config = {"test_key": "test_value"}

        config_provider, logger_provider = create_test_categorization_setup(
            language="hr", custom_config=custom_config
        )

        # Verify custom config was set
        result = config_provider.get_categorization_config("hr")
        self.assertEqual(result, custom_config)

    def test_create_test_categorization_setup_without_custom_config(self):
        """Test create_test_categorization_setup without custom configuration."""
        config_provider, logger_provider = create_test_categorization_setup(language="hr")

        # Should get default configuration
        result = config_provider.get_categorization_config("hr")
        self.assertIsInstance(result, dict)
        self.assertIn("categories", result)

    def test_create_minimal_test_config_structure(self):
        """Test create_minimal_test_config returns proper structure."""
        config = create_minimal_test_config()

        # Check all required sections
        self.assertIn("categories", config)
        self.assertIn("patterns", config)
        self.assertIn("cultural_keywords", config)
        self.assertIn("complexity_thresholds", config)
        self.assertIn("retrieval_strategies", config)

        # Check it's minimal (only 2 categories)
        self.assertEqual(len(config["categories"]), 2)
        self.assertIn("general", config["categories"])
        self.assertIn("technical", config["categories"])

    def test_create_minimal_test_config_values(self):
        """Test create_minimal_test_config has expected values."""
        config = create_minimal_test_config()

        # Check specific values
        self.assertEqual(config["categories"]["general"]["priority"], 1)
        self.assertEqual(config["categories"]["technical"]["priority"], 2)
        self.assertIn("test", config["patterns"]["general"])
        self.assertIn("API", config["patterns"]["technical"])
        self.assertEqual(config["complexity_thresholds"]["simple"], 2.0)
        self.assertEqual(config["retrieval_strategies"]["default"], "hybrid")

    def test_create_complex_test_config_structure(self):
        """Test create_complex_test_config returns comprehensive structure."""
        config = create_complex_test_config()

        # Check all required sections
        self.assertIn("categories", config)
        self.assertIn("patterns", config)
        self.assertIn("cultural_keywords", config)
        self.assertIn("complexity_thresholds", config)
        self.assertIn("retrieval_strategies", config)

        # Check it's complex (6 categories)
        self.assertEqual(len(config["categories"]), 6)
        expected_categories = ["general", "technical", "cultural", "academic", "legal", "medical"]
        for category in expected_categories:
            self.assertIn(category, config["categories"])

    def test_create_complex_test_config_comprehensive_patterns(self):
        """Test create_complex_test_config has comprehensive patterns."""
        config = create_complex_test_config()

        # Check pattern coverage
        patterns = config["patterns"]
        self.assertIn("API", patterns["technical"])
        self.assertIn("programming", patterns["technical"])
        self.assertIn("kultura", patterns["cultural"])
        self.assertIn("tradition", patterns["cultural"])
        self.assertIn("research", patterns["academic"])
        self.assertIn("law", patterns["legal"])
        self.assertIn("medicine", patterns["medical"])

    def test_create_complex_test_config_cultural_keywords(self):
        """Test create_complex_test_config has comprehensive cultural keywords."""
        config = create_complex_test_config()

        cultural_keywords = config["cultural_keywords"]

        # Check Croatian cultural categories
        self.assertIn("croatian_culture", cultural_keywords)
        self.assertIn("croatian_history", cultural_keywords)
        self.assertIn("croatian_language", cultural_keywords)

        # Check specific Croatian keywords
        self.assertIn("zagreb", cultural_keywords["croatian_culture"])
        self.assertIn("dubrovnik", cultural_keywords["croatian_culture"])
        self.assertIn("domovinski rat", cultural_keywords["croatian_history"])
        self.assertIn("štokavski", cultural_keywords["croatian_language"])

        # Check English cultural keywords
        self.assertIn("english_culture", cultural_keywords)
        self.assertIn("london", cultural_keywords["english_culture"])

    def test_create_complex_test_config_retrieval_strategies(self):
        """Test create_complex_test_config has comprehensive retrieval strategies."""
        config = create_complex_test_config()

        strategies = config["retrieval_strategies"]

        # Check category-specific strategies
        self.assertEqual(strategies["category_technical"], "dense")
        self.assertEqual(strategies["category_cultural"], "cultural_context")
        self.assertEqual(strategies["category_academic"], "hierarchical")
        self.assertEqual(strategies["category_legal"], "precise")
        self.assertEqual(strategies["category_medical"], "specialized")

        # Check complexity-based strategies
        self.assertEqual(strategies["complexity_simple"], "sparse")
        self.assertEqual(strategies["complexity_moderate"], "hybrid")
        self.assertEqual(strategies["complexity_complex"], "dense")
        self.assertEqual(strategies["complexity_analytical"], "hierarchical")


class TestConfigProviderIntegration(unittest.TestCase):
    """Test integration between different provider components."""

    def test_mock_config_provider_with_factory_function(self):
        """Test mock config provider works with factory function."""
        config_provider = create_config_provider(use_mock=True)

        # Set test configuration
        test_config = {"test": "integration"}
        config_provider.set_categorization_config("hr", test_config)

        # Retrieve and verify
        result = config_provider.get_categorization_config("hr")
        self.assertEqual(result, test_config)

    def test_test_setup_integration(self):
        """Test complete test setup integration."""
        config_provider, logger_provider = create_test_categorization_setup(
            language="en",
            custom_config={"integration": "test"}
        )

        # Test config provider
        config = config_provider.get_categorization_config("en")
        self.assertEqual(config, {"integration": "test"})

        # Test logger provider
        logger_provider.info("integration test")
        messages = logger_provider.get_messages("info")
        self.assertEqual(messages, ["integration test"])

    def test_complex_config_with_mock_provider(self):
        """Test complex configuration with mock provider."""
        complex_config = create_complex_test_config()

        config_provider = MockConfigProvider()
        config_provider.set_categorization_config("hr", complex_config)

        result = config_provider.get_categorization_config("hr")

        # Verify complex structure is preserved
        self.assertEqual(len(result["categories"]), 6)
        self.assertIn("croatian_culture", result["cultural_keywords"])
        self.assertEqual(result["retrieval_strategies"]["category_medical"], "specialized")


if __name__ == "__main__":
    unittest.main()