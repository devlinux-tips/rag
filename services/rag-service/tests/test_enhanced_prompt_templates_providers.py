"""
Tests for enhanced prompt templates provider implementations.
Comprehensive testing of mock and production components for dependency injection.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch
from typing import Any

from src.generation.enhanced_prompt_templates import PromptConfig, PromptType
from src.retrieval.categorization import CategoryType
from src.generation.enhanced_prompt_templates_providers import (
    MockConfigProvider,
    MockLoggerProvider,
    ProductionConfigProvider,
    StandardLoggerProvider,
    create_mock_setup,
    create_production_setup,
    create_test_config,
    create_minimal_config,
    create_invalid_config,
    create_development_prompt_builder,
    create_test_prompt_builder,
    build_category_templates,
    create_template_variants,
)


class TestMockConfigProvider(unittest.TestCase):
    """Test mock configuration provider implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockConfigProvider()

    def test_initialization_default(self):
        """Test default initialization creates proper config."""
        provider = MockConfigProvider()

        self.assertIsInstance(provider.config, PromptConfig)
        self.assertEqual(provider.config.language, "hr")
        self.assertEqual(provider.call_history, [])

        # Check that all required categories are present
        self.assertIn(CategoryType.GENERAL, provider.config.category_templates)
        self.assertIn(CategoryType.TECHNICAL, provider.config.category_templates)
        self.assertIn(CategoryType.CULTURAL, provider.config.category_templates)

        # Check that required prompt types are present
        general_templates = provider.config.category_templates[CategoryType.GENERAL]
        self.assertIn(PromptType.SYSTEM, general_templates)
        self.assertIn(PromptType.USER, general_templates)
        self.assertIn(PromptType.FOLLOWUP, general_templates)

    def test_initialization_with_config(self):
        """Test initialization with provided config."""
        custom_config = PromptConfig(
            category_templates={CategoryType.GENERAL: {PromptType.SYSTEM: "Custom system"}},
            messages={"test": "message"},
            formatting={"test": "format"},
            language="en"
        )

        provider = MockConfigProvider(custom_config)
        self.assertEqual(provider.config, custom_config)

    def test_create_default_config_structure(self):
        """Test that default config has proper structure."""
        config = self.provider._create_default_config()

        # Test category templates structure
        self.assertIsInstance(config.category_templates, dict)
        for category, templates in config.category_templates.items():
            self.assertIsInstance(category, CategoryType)
            self.assertIsInstance(templates, dict)
            for prompt_type, template in templates.items():
                self.assertIsInstance(prompt_type, PromptType)
                self.assertIsInstance(template, str)
                self.assertGreater(len(template), 0)

        # Test messages structure
        self.assertIsInstance(config.messages, dict)
        self.assertIn("no_context", config.messages)
        self.assertIn("error_template_missing", config.messages)
        self.assertIn("truncation_notice", config.messages)

        # Test formatting structure
        self.assertIsInstance(config.formatting, dict)
        self.assertIn("source_label", config.formatting)
        self.assertIn("truncation_indicator", config.formatting)
        self.assertIn("min_chunk_size", config.formatting)
        self.assertIn("max_context_length", config.formatting)

    def test_set_config(self):
        """Test setting new configuration."""
        new_config = PromptConfig(
            category_templates={CategoryType.TECHNICAL: {PromptType.USER: "New template"}},
            messages={"new": "message"},
            formatting={"new": "format"},
            language="en"
        )

        self.provider.set_config(new_config)
        self.assertEqual(self.provider.config, new_config)

    def test_add_category_template_new_category(self):
        """Test adding template for new category."""
        # Use ACADEMIC category if available, otherwise create a test scenario
        if hasattr(CategoryType, 'ACADEMIC'):
            test_category = CategoryType.ACADEMIC
        else:
            # Skip if ACADEMIC not available
            self.skipTest("ACADEMIC category not available")

        template = "Academic template: {query}"
        self.provider.add_category_template(test_category, PromptType.USER, template)

        self.assertIn(test_category, self.provider.config.category_templates)
        self.assertEqual(
            self.provider.config.category_templates[test_category][PromptType.USER],
            template
        )

    def test_add_category_template_existing_category(self):
        """Test adding template to existing category."""
        template = "New general template: {query}"
        self.provider.add_category_template(CategoryType.GENERAL, PromptType.USER, template)

        self.assertEqual(
            self.provider.config.category_templates[CategoryType.GENERAL][PromptType.USER],
            template
        )

    def test_remove_template_existing(self):
        """Test removing existing template."""
        # Ensure template exists first
        self.assertIn(PromptType.USER, self.provider.config.category_templates[CategoryType.GENERAL])

        self.provider.remove_template(CategoryType.GENERAL, PromptType.USER)

        self.assertNotIn(PromptType.USER, self.provider.config.category_templates[CategoryType.GENERAL])

    def test_remove_template_nonexistent_type(self):
        """Test removing non-existent template type."""
        # Should not raise error
        self.provider.remove_template(CategoryType.GENERAL, PromptType.FOLLOWUP)
        self.provider.remove_template(CategoryType.GENERAL, PromptType.FOLLOWUP)  # Second call

    def test_remove_template_nonexistent_category(self):
        """Test removing template from non-existent category."""
        # Should not raise error for non-existent category
        if hasattr(CategoryType, 'ACADEMIC'):
            test_category = CategoryType.ACADEMIC
            # Remove the category first if it exists
            if test_category in self.provider.config.category_templates:
                del self.provider.config.category_templates[test_category]
            self.provider.remove_template(test_category, PromptType.USER)

    def test_get_prompt_config_basic(self):
        """Test getting prompt config for language."""
        result = self.provider.get_prompt_config("en")

        self.assertIsInstance(result, PromptConfig)
        self.assertEqual(result.language, "en")
        self.assertEqual(len(self.provider.call_history), 1)
        self.assertEqual(self.provider.call_history[0], "get_prompt_config(en)")

    def test_get_prompt_config_preserves_templates(self):
        """Test that get_prompt_config preserves template structure."""
        result = self.provider.get_prompt_config("hr")

        # Should have same category templates
        self.assertEqual(result.category_templates, self.provider.config.category_templates)
        self.assertEqual(result.messages, self.provider.config.messages)
        self.assertEqual(result.formatting, self.provider.config.formatting)

    def test_get_prompt_config_multiple_calls(self):
        """Test multiple calls to get_prompt_config."""
        self.provider.get_prompt_config("hr")
        self.provider.get_prompt_config("en")
        self.provider.get_prompt_config("hr")

        self.assertEqual(len(self.provider.call_history), 3)
        self.assertEqual(self.provider.call_history[0], "get_prompt_config(hr)")
        self.assertEqual(self.provider.call_history[1], "get_prompt_config(en)")
        self.assertEqual(self.provider.call_history[2], "get_prompt_config(hr)")


class TestMockLoggerProvider(unittest.TestCase):
    """Test mock logger provider implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = MockLoggerProvider()

    def test_initialization(self):
        """Test logger initialization."""
        logger = MockLoggerProvider()
        expected_levels = {"info", "debug", "warning", "error"}
        self.assertEqual(set(logger.messages.keys()), expected_levels)
        for level in expected_levels:
            self.assertEqual(logger.messages[level], [])

    def test_info_logging(self):
        """Test info message logging."""
        self.logger.info("Test info message")
        self.assertEqual(self.logger.messages["info"], ["Test info message"])
        self.assertEqual(len(self.logger.messages["debug"]), 0)
        self.assertEqual(len(self.logger.messages["warning"]), 0)
        self.assertEqual(len(self.logger.messages["error"]), 0)

    def test_debug_logging(self):
        """Test debug message logging."""
        self.logger.debug("Test debug message")
        self.assertEqual(self.logger.messages["debug"], ["Test debug message"])
        self.assertEqual(len(self.logger.messages["info"]), 0)

    def test_warning_logging(self):
        """Test warning message logging."""
        self.logger.warning("Test warning message")
        self.assertEqual(self.logger.messages["warning"], ["Test warning message"])
        self.assertEqual(len(self.logger.messages["info"]), 0)

    def test_error_logging(self):
        """Test error message logging."""
        self.logger.error("Test error message")
        self.assertEqual(self.logger.messages["error"], ["Test error message"])
        self.assertEqual(len(self.logger.messages["info"]), 0)

    def test_multiple_messages_same_level(self):
        """Test logging multiple messages to same level."""
        self.logger.info("Info 1")
        self.logger.info("Info 2")
        self.logger.info("Info 3")

        self.assertEqual(len(self.logger.messages["info"]), 3)
        self.assertEqual(self.logger.messages["info"], ["Info 1", "Info 2", "Info 3"])

    def test_multiple_messages_different_levels(self):
        """Test logging messages to different levels."""
        self.logger.info("Info message")
        self.logger.debug("Debug message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")

        self.assertEqual(len(self.logger.messages["info"]), 1)
        self.assertEqual(len(self.logger.messages["debug"]), 1)
        self.assertEqual(len(self.logger.messages["warning"]), 1)
        self.assertEqual(len(self.logger.messages["error"]), 1)

    def test_clear_messages(self):
        """Test clearing all messages."""
        self.logger.info("Test message")
        self.logger.debug("Test debug")
        self.logger.warning("Test warning")
        self.logger.error("Test error")

        self.logger.clear_messages()

        for level in self.logger.messages:
            self.assertEqual(len(self.logger.messages[level]), 0)

    def test_get_messages_by_level(self):
        """Test getting messages by specific level."""
        self.logger.info("Info message")
        self.logger.debug("Debug message")
        self.logger.warning("Warning message")

        info_messages = self.logger.get_messages("info")
        self.assertEqual(info_messages, ["Info message"])

        debug_messages = self.logger.get_messages("debug")
        self.assertEqual(debug_messages, ["Debug message"])

        warning_messages = self.logger.get_messages("warning")
        self.assertEqual(warning_messages, ["Warning message"])

    def test_get_all_messages(self):
        """Test getting all messages."""
        self.logger.info("Info")
        self.logger.error("Error")

        all_messages = self.logger.get_messages()
        self.assertIsInstance(all_messages, dict)
        self.assertEqual(len(all_messages["info"]), 1)
        self.assertEqual(len(all_messages["error"]), 1)
        self.assertEqual(len(all_messages["debug"]), 0)
        self.assertEqual(len(all_messages["warning"]), 0)

    def test_get_messages_nonexistent_level(self):
        """Test getting messages for non-existent level."""
        result = self.logger.get_messages("nonexistent")
        self.assertEqual(result, [])

    def test_get_messages_none_level(self):
        """Test getting messages with None level returns all."""
        self.logger.info("Info")
        self.logger.error("Error")

        result = self.logger.get_messages(None)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result["info"]), 1)
        self.assertEqual(len(result["error"]), 1)


class TestProductionConfigProvider(unittest.TestCase):
    """Test production configuration provider implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = ProductionConfigProvider()

    def test_initialization(self):
        """Test provider initialization."""
        provider = ProductionConfigProvider()
        self.assertEqual(provider._config_cache, {})

    @patch('src.utils.config_loader.get_language_specific_config')
    def test_get_prompt_config_success(self, mock_get_config):
        """Test successful configuration loading."""
        # Mock configuration data
        mock_prompts = {
            "general": {
                "system": "System prompt",
                "user": "User prompt: {query}",
                "followup": "Followup: {followup_query}"
            },
            "technical": {
                "system": "Technical system",
                "user": "Tech: {query}"
            }
        }

        mock_messages = {
            "no_context": "No context available",
            "error_template_missing": "Template missing"
        }

        mock_formatting = {
            "source_label": "Source",
            "truncation_indicator": "..."
        }

        # Configure mock to return different data for different calls
        def mock_get_config_side_effect(section, language):
            if section == "prompts":
                return mock_prompts
            elif section == "messages":
                return mock_messages
            elif section == "formatting":
                return mock_formatting
            return {}

        mock_get_config.side_effect = mock_get_config_side_effect

        result = self.provider.get_prompt_config("hr")

        self.assertIsInstance(result, PromptConfig)
        self.assertEqual(result.language, "hr")

        # Verify category templates were parsed correctly
        self.assertIn(CategoryType.GENERAL, result.category_templates)
        self.assertIn(CategoryType.TECHNICAL, result.category_templates)

        general_templates = result.category_templates[CategoryType.GENERAL]
        self.assertEqual(general_templates[PromptType.SYSTEM], "System prompt")
        self.assertEqual(general_templates[PromptType.USER], "User prompt: {query}")
        self.assertEqual(general_templates[PromptType.FOLLOWUP], "Followup: {followup_query}")

        technical_templates = result.category_templates[CategoryType.TECHNICAL]
        self.assertEqual(technical_templates[PromptType.SYSTEM], "Technical system")
        self.assertEqual(technical_templates[PromptType.USER], "Tech: {query}")

        # Verify messages and formatting
        self.assertEqual(result.messages, mock_messages)
        self.assertEqual(result.formatting, mock_formatting)

        # Verify caching
        self.assertIn("hr", self.provider._config_cache)

    @patch('src.utils.config_loader.get_language_specific_config')
    def test_get_prompt_config_caching(self, mock_get_config):
        """Test that configurations are cached."""
        mock_get_config.return_value = {}

        # First call
        result1 = self.provider.get_prompt_config("en")
        # Second call
        result2 = self.provider.get_prompt_config("en")

        # Should only load once
        self.assertEqual(mock_get_config.call_count, 3)  # prompts, messages, formatting

        # Should return same object (cached)
        self.assertIs(result1, result2)

    @patch('src.utils.config_loader.get_language_specific_config')
    def test_get_prompt_config_different_languages(self, mock_get_config):
        """Test loading different languages."""
        mock_get_config.return_value = {}

        result_hr = self.provider.get_prompt_config("hr")
        result_en = self.provider.get_prompt_config("en")

        # Should be different objects
        self.assertIsNot(result_hr, result_en)
        self.assertEqual(result_hr.language, "hr")
        self.assertEqual(result_en.language, "en")

    @patch('src.utils.config_loader.get_language_specific_config')
    def test_get_prompt_config_invalid_category(self, mock_get_config):
        """Test handling of invalid categories."""
        mock_prompts = {
            "general": {"system": "Valid"},
            "invalid_category": {"system": "Should be skipped"}
        }

        def mock_get_config_side_effect(section, language):
            if section == "prompts":
                return mock_prompts
            return {}

        mock_get_config.side_effect = mock_get_config_side_effect

        result = self.provider.get_prompt_config("hr")

        # Should only contain valid categories
        self.assertIn(CategoryType.GENERAL, result.category_templates)
        self.assertEqual(len(result.category_templates), 1)

    @patch('src.utils.config_loader.get_language_specific_config')
    def test_get_prompt_config_invalid_prompt_type(self, mock_get_config):
        """Test handling of invalid prompt types."""
        mock_prompts = {
            "general": {
                "system": "Valid system",
                "invalid_type": "Should be skipped",
                "user": "Valid user"
            }
        }

        def mock_get_config_side_effect(section, language):
            if section == "prompts":
                return mock_prompts
            return {}

        mock_get_config.side_effect = mock_get_config_side_effect

        result = self.provider.get_prompt_config("hr")

        # Should only contain valid prompt types
        general_templates = result.category_templates[CategoryType.GENERAL]
        self.assertIn(PromptType.SYSTEM, general_templates)
        self.assertIn(PromptType.USER, general_templates)
        self.assertEqual(len(general_templates), 2)

    @patch('src.utils.config_loader.get_language_specific_config')
    def test_load_config_from_system_failure(self, mock_get_config):
        """Test handling of configuration loading failure."""
        mock_get_config.side_effect = Exception("Configuration error")

        with self.assertRaises(RuntimeError) as cm:
            self.provider.get_prompt_config("hr")

        self.assertIn("Failed to load prompt configuration", str(cm.exception))
        self.assertIn("hr", str(cm.exception))
        self.assertIn("Configuration error", str(cm.exception))


class TestStandardLoggerProvider(unittest.TestCase):
    """Test standard logger provider implementation."""

    @patch('logging.getLogger')
    def test_initialization_default(self, mock_get_logger):
        """Test initialization with default logger name."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        provider = StandardLoggerProvider()

        mock_get_logger.assert_called_once_with('src.generation.enhanced_prompt_templates_providers')
        self.assertEqual(provider.logger, mock_logger)

    @patch('logging.getLogger')
    def test_initialization_custom_name(self, mock_get_logger):
        """Test initialization with custom logger name."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        provider = StandardLoggerProvider("custom_logger")

        mock_get_logger.assert_called_once_with("custom_logger")
        self.assertEqual(provider.logger, mock_logger)

    def test_logging_methods(self):
        """Test all logging methods delegate to logger."""
        mock_logger = Mock()
        provider = StandardLoggerProvider()
        provider.logger = mock_logger

        provider.info("Info message")
        provider.debug("Debug message")
        provider.warning("Warning message")
        provider.error("Error message")

        mock_logger.info.assert_called_once_with("Info message")
        mock_logger.debug.assert_called_once_with("Debug message")
        mock_logger.warning.assert_called_once_with("Warning message")
        mock_logger.error.assert_called_once_with("Error message")


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating provider setups."""

    def test_create_mock_setup_default(self):
        """Test creating default mock setup."""
        config_provider, logger_provider = create_mock_setup()

        self.assertIsInstance(config_provider, MockConfigProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)

        # Should have default language
        self.assertEqual(config_provider.config.language, "hr")

    def test_create_mock_setup_with_config(self):
        """Test creating mock setup with custom config."""
        custom_config = PromptConfig(
            category_templates={CategoryType.GENERAL: {PromptType.SYSTEM: "Custom"}},
            messages={"test": "message"},
            formatting={"test": "format"},
            language="en"
        )

        config_provider, logger_provider = create_mock_setup(config=custom_config)

        self.assertEqual(config_provider.config, custom_config)

    def test_create_mock_setup_with_custom_templates(self):
        """Test creating mock setup with custom templates."""
        custom_templates = {
            CategoryType.TECHNICAL: {
                PromptType.SYSTEM: "Custom technical system",
                PromptType.USER: "Custom technical user"
            }
        }

        config_provider, logger_provider = create_mock_setup(
            custom_templates=custom_templates,
            language="en"
        )

        self.assertEqual(config_provider.config.category_templates, custom_templates)
        self.assertEqual(config_provider.config.language, "en")

    def test_create_mock_setup_with_custom_messages(self):
        """Test creating mock setup with custom messages."""
        custom_messages = {"custom_message": "Custom value"}

        config_provider, logger_provider = create_mock_setup(custom_messages=custom_messages)

        self.assertEqual(config_provider.config.messages, custom_messages)

    def test_create_mock_setup_with_custom_formatting(self):
        """Test creating mock setup with custom formatting."""
        custom_formatting = {"custom_format": "Custom format value"}

        config_provider, logger_provider = create_mock_setup(custom_formatting=custom_formatting)

        self.assertEqual(config_provider.config.formatting, custom_formatting)

    def test_create_mock_setup_combined_customizations(self):
        """Test creating mock setup with multiple customizations."""
        custom_templates = {CategoryType.GENERAL: {PromptType.SYSTEM: "Custom"}}
        custom_messages = {"msg": "value"}
        custom_formatting = {"fmt": "value"}

        config_provider, logger_provider = create_mock_setup(
            custom_templates=custom_templates,
            custom_messages=custom_messages,
            custom_formatting=custom_formatting,
            language="en"
        )

        config = config_provider.config
        self.assertEqual(config.category_templates, custom_templates)
        self.assertEqual(config.messages, custom_messages)
        self.assertEqual(config.formatting, custom_formatting)
        self.assertEqual(config.language, "en")

    @patch('src.generation.enhanced_prompt_templates_providers.StandardLoggerProvider')
    def test_create_production_setup_default(self, mock_logger_class):
        """Test creating production setup with defaults."""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        config_provider, logger_provider = create_production_setup()

        self.assertIsInstance(config_provider, ProductionConfigProvider)
        mock_logger_class.assert_called_once_with('src.generation.enhanced_prompt_templates_providers')
        self.assertEqual(logger_provider, mock_logger)

    @patch('src.generation.enhanced_prompt_templates_providers.StandardLoggerProvider')
    def test_create_production_setup_custom_logger(self, mock_logger_class):
        """Test creating production setup with custom logger name."""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        config_provider, logger_provider = create_production_setup("custom_logger")

        self.assertIsInstance(config_provider, ProductionConfigProvider)
        mock_logger_class.assert_called_once_with("custom_logger")


class TestConfigBuilders(unittest.TestCase):
    """Test configuration builder functions."""

    def test_create_test_config_default(self):
        """Test creating default test configuration."""
        config = create_test_config()

        self.assertIsInstance(config, PromptConfig)
        self.assertEqual(config.language, "hr")

        # Should include all categories by default
        self.assertIn(CategoryType.GENERAL, config.category_templates)
        self.assertIn(CategoryType.TECHNICAL, config.category_templates)
        self.assertIn(CategoryType.CULTURAL, config.category_templates)

        # Should include followup for general
        self.assertIn(PromptType.FOLLOWUP, config.category_templates[CategoryType.GENERAL])

    def test_create_test_config_custom_language(self):
        """Test creating test config with custom language."""
        config = create_test_config(language="en")

        self.assertEqual(config.language, "en")
        # Messages should reflect language
        self.assertIn("(en)", config.messages["no_context"])

    def test_create_test_config_no_followup(self):
        """Test creating test config without followup."""
        config = create_test_config(include_followup=False)

        general_templates = config.category_templates[CategoryType.GENERAL]
        self.assertNotIn(PromptType.FOLLOWUP, general_templates)

    def test_create_test_config_no_technical(self):
        """Test creating test config without technical category."""
        config = create_test_config(include_technical=False)

        self.assertNotIn(CategoryType.TECHNICAL, config.category_templates)

    def test_create_test_config_no_cultural(self):
        """Test creating test config without cultural category."""
        config = create_test_config(include_cultural=False)

        self.assertNotIn(CategoryType.CULTURAL, config.category_templates)

    def test_create_test_config_minimal(self):
        """Test creating minimal test config."""
        config = create_test_config(
            include_followup=False,
            include_technical=False,
            include_cultural=False
        )

        # Should only have general category
        self.assertEqual(len(config.category_templates), 1)
        self.assertIn(CategoryType.GENERAL, config.category_templates)

        general_templates = config.category_templates[CategoryType.GENERAL]
        self.assertIn(PromptType.SYSTEM, general_templates)
        self.assertIn(PromptType.USER, general_templates)
        self.assertNotIn(PromptType.FOLLOWUP, general_templates)

    def test_create_test_config_language_specific_formatting(self):
        """Test that language affects formatting."""
        config_hr = create_test_config(language="hr")
        config_en = create_test_config(language="en")

        self.assertEqual(config_hr.formatting["source_label"], "Izvor")
        self.assertEqual(config_en.formatting["source_label"], "Source")

    def test_create_minimal_config(self):
        """Test creating minimal configuration."""
        config = create_minimal_config()

        self.assertEqual(config.language, "hr")
        self.assertEqual(len(config.category_templates), 1)
        self.assertIn(CategoryType.GENERAL, config.category_templates)

        general_templates = config.category_templates[CategoryType.GENERAL]
        self.assertEqual(len(general_templates), 2)
        self.assertIn(PromptType.SYSTEM, general_templates)
        self.assertIn(PromptType.USER, general_templates)

    def test_create_minimal_config_custom_language(self):
        """Test creating minimal config with custom language."""
        config = create_minimal_config("en")
        self.assertEqual(config.language, "en")

    def test_create_invalid_config(self):
        """Test creating invalid configuration for error testing."""
        config = create_invalid_config()

        self.assertEqual(config.language, "hr")
        general_templates = config.category_templates[CategoryType.GENERAL]
        self.assertIn(PromptType.SYSTEM, general_templates)
        self.assertNotIn(PromptType.USER, general_templates)  # Intentionally missing

    def test_create_invalid_config_custom_language(self):
        """Test creating invalid config with custom language."""
        config = create_invalid_config("en")
        self.assertEqual(config.language, "en")


class TestIntegrationHelpers(unittest.TestCase):
    """Test integration helper functions."""

    @patch('src.generation.enhanced_prompt_templates.create_enhanced_prompt_builder')
    @patch('src.generation.enhanced_prompt_templates_providers.create_production_setup')
    def test_create_development_prompt_builder(self, mock_create_setup, mock_create_builder):
        """Test creating development prompt builder."""
        mock_config_provider = Mock()
        mock_logger_provider = Mock()
        mock_create_setup.return_value = (mock_config_provider, mock_logger_provider)
        mock_builder = Mock()
        mock_create_builder.return_value = mock_builder

        result = create_development_prompt_builder()

        mock_create_setup.assert_called_once()
        mock_create_builder.assert_called_once_with(
            language="hr",
            config_provider=mock_config_provider,
            logger_provider=mock_logger_provider
        )
        self.assertEqual(result, mock_builder)

    @patch('src.generation.enhanced_prompt_templates.create_enhanced_prompt_builder')
    @patch('src.generation.enhanced_prompt_templates_providers.create_mock_setup')
    def test_create_test_prompt_builder_default(self, mock_create_setup, mock_create_builder):
        """Test creating test prompt builder with defaults."""
        mock_config_provider = Mock()
        mock_logger_provider = Mock()
        mock_create_setup.return_value = (mock_config_provider, mock_logger_provider)
        mock_builder = Mock()
        mock_create_builder.return_value = mock_builder

        result = create_test_prompt_builder()

        mock_create_setup.assert_called_once_with(
            config=None,
            custom_templates=None,
            custom_messages=None,
            custom_formatting=None,
            language="hr"
        )
        mock_create_builder.assert_called_once_with(
            language="hr",
            config_provider=mock_config_provider,
            logger_provider=mock_logger_provider
        )
        # Should return tuple
        self.assertEqual(result[0], mock_builder)
        self.assertEqual(result[1], (mock_config_provider, mock_logger_provider))

    @patch('src.generation.enhanced_prompt_templates.create_enhanced_prompt_builder')
    @patch('src.generation.enhanced_prompt_templates_providers.create_mock_setup')
    def test_create_test_prompt_builder_custom(self, mock_create_setup, mock_create_builder):
        """Test creating test prompt builder with custom parameters."""
        custom_config = Mock()
        custom_templates = Mock()
        custom_messages = Mock()
        custom_formatting = Mock()

        mock_config_provider = Mock()
        mock_logger_provider = Mock()
        mock_create_setup.return_value = (mock_config_provider, mock_logger_provider)
        mock_builder = Mock()
        mock_create_builder.return_value = mock_builder

        result = create_test_prompt_builder(
            language="en",
            config=custom_config,
            custom_templates=custom_templates,
            custom_messages=custom_messages,
            custom_formatting=custom_formatting
        )

        mock_create_setup.assert_called_once_with(
            config=custom_config,
            custom_templates=custom_templates,
            custom_messages=custom_messages,
            custom_formatting=custom_formatting,
            language="en"
        )
        mock_create_builder.assert_called_once_with(
            language="en",
            config_provider=mock_config_provider,
            logger_provider=mock_logger_provider
        )


class TestTemplateHelpers(unittest.TestCase):
    """Test template building helper functions."""

    def test_build_category_templates_valid(self):
        """Test building category templates from flat dictionary."""
        templates = {
            "general.system": "General system prompt",
            "general.user": "General user prompt",
            "technical.system": "Technical system prompt",
            "technical.user": "Technical user prompt",
        }

        result = build_category_templates(templates)

        self.assertEqual(len(result), 2)
        self.assertIn(CategoryType.GENERAL, result)
        self.assertIn(CategoryType.TECHNICAL, result)

        self.assertEqual(result[CategoryType.GENERAL][PromptType.SYSTEM], "General system prompt")
        self.assertEqual(result[CategoryType.GENERAL][PromptType.USER], "General user prompt")
        self.assertEqual(result[CategoryType.TECHNICAL][PromptType.SYSTEM], "Technical system prompt")
        self.assertEqual(result[CategoryType.TECHNICAL][PromptType.USER], "Technical user prompt")

    def test_build_category_templates_invalid_keys(self):
        """Test building templates with invalid keys."""
        templates = {
            "general.system": "Valid",
            "invalid_format": "Should be skipped",
            "invalid_category.system": "Should be skipped",
            "general.invalid_type": "Should be skipped",
            "": "Empty key",
        }

        result = build_category_templates(templates)

        self.assertEqual(len(result), 1)
        self.assertIn(CategoryType.GENERAL, result)
        self.assertEqual(len(result[CategoryType.GENERAL]), 1)
        self.assertEqual(result[CategoryType.GENERAL][PromptType.SYSTEM], "Valid")

    def test_build_category_templates_empty(self):
        """Test building templates with empty input."""
        result = build_category_templates({})
        self.assertEqual(result, {})

    def test_create_template_variants_basic(self):
        """Test creating template variants."""
        base_template = "Base template with {variant} placeholder"
        variants = {
            "formal": "formal language",
            "casual": "casual tone",
            "technical": "technical details"
        }

        result = create_template_variants(base_template, variants)

        self.assertEqual(len(result), 4)  # base + 3 variants
        self.assertEqual(result["base"], base_template)
        self.assertEqual(result["formal"], "Base template with formal language placeholder")
        self.assertEqual(result["casual"], "Base template with casual tone placeholder")
        self.assertEqual(result["technical"], "Base template with technical details placeholder")

    def test_create_template_variants_no_placeholder(self):
        """Test creating variants when base template has no placeholder."""
        base_template = "Template without placeholder"
        variants = {"variant1": "replacement"}

        result = create_template_variants(base_template, variants)

        self.assertEqual(result["base"], base_template)
        self.assertEqual(result["variant1"], base_template)  # No change

    def test_create_template_variants_empty_variants(self):
        """Test creating variants with empty variants dict."""
        base_template = "Base template"

        result = create_template_variants(base_template, {})

        self.assertEqual(len(result), 1)
        self.assertEqual(result["base"], base_template)


if __name__ == "__main__":
    unittest.main()