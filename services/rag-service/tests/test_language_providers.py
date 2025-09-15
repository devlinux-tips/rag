"""
Tests for generation/language_providers.py module.
Tests language configuration providers for dependency injection patterns.
"""

import unittest
from unittest.mock import Mock, patch

from src.generation.language_providers import DefaultLanguageProvider, MockLanguageProvider
from src.utils.config_loader import ConfigError as ConfigurationError


class TestDefaultLanguageProvider(unittest.TestCase):
    """Test production language configuration provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = DefaultLanguageProvider()

    def test_provider_creation(self):
        """Test provider can be created."""
        provider = DefaultLanguageProvider()
        self.assertIsInstance(provider, DefaultLanguageProvider)

    def test_provider_attributes(self):
        """Test provider has no attributes by default (stateless)."""
        self.assertEqual(len(self.provider.__dict__), 0)

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_formal_prompts_success(self, mock_get_config):
        """Test successful retrieval of formal prompts."""
        mock_get_config.return_value = {
            "formal": {
                "instruction": "Formal instruction template",
                "response": "Formal response template"
            }
        }

        result = self.provider.get_formal_prompts("hr")

        mock_get_config.assert_called_once_with("prompts", "hr")
        self.assertEqual(result, {
            "instruction": "Formal instruction template",
            "response": "Formal response template"
        })

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_formal_prompts_missing_formal_section(self, mock_get_config):
        """Test error when formal section is missing."""
        mock_get_config.return_value = {
            "informal": {"some": "data"}
        }

        with self.assertRaises(ConfigurationError) as cm:
            self.provider.get_formal_prompts("hr")

        self.assertIn("Missing 'formal' prompts section in hr.toml configuration", str(cm.exception))

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_formal_prompts_config_error(self, mock_get_config):
        """Test error handling when config loading fails."""
        mock_get_config.side_effect = Exception("Config file not found")

        with self.assertRaises(ConfigurationError) as cm:
            self.provider.get_formal_prompts("hr")

        self.assertIn("Failed to load formal prompts for hr: Config file not found", str(cm.exception))

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_error_template_success(self, mock_get_config):
        """Test successful retrieval of error template."""
        mock_get_config.return_value = {
            "error_message_template": "Error: {error}"
        }

        result = self.provider.get_error_template("en")

        mock_get_config.assert_called_once_with("prompts", "en")
        self.assertEqual(result, "Error: {error}")

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_error_template_missing_template(self, mock_get_config):
        """Test error when error template is missing."""
        mock_get_config.return_value = {
            "formal": {"some": "data"}
        }

        with self.assertRaises(ConfigurationError) as cm:
            self.provider.get_error_template("en")

        self.assertIn("Missing 'error_message_template' in en.toml prompts configuration", str(cm.exception))

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_error_template_config_error(self, mock_get_config):
        """Test error handling when config loading fails."""
        mock_get_config.side_effect = Exception("Network error")

        with self.assertRaises(ConfigurationError) as cm:
            self.provider.get_error_template("en")

        self.assertIn("Failed to load error template for en: Network error", str(cm.exception))

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_confidence_settings_success(self, mock_get_config):
        """Test successful retrieval of confidence settings."""
        mock_get_config.return_value = {
            "error_phrases": ["greška", "pogreška"],
            "threshold": 0.8
        }

        result = self.provider.get_confidence_settings("hr")

        mock_get_config.assert_called_once_with("confidence", "hr")
        self.assertEqual(result, {
            "error_phrases": ["greška", "pogreška"],
            "threshold": 0.8
        })

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_get_confidence_settings_config_error(self, mock_get_config):
        """Test error handling when config loading fails."""
        mock_get_config.side_effect = Exception("Invalid TOML")

        with self.assertRaises(ConfigurationError) as cm:
            self.provider.get_confidence_settings("hr")

        self.assertIn("Failed to load confidence settings for hr: Invalid TOML", str(cm.exception))

    @patch("src.utils.config_loader.get_language_specific_config")
    def test_multiple_language_requests(self, mock_get_config):
        """Test handling multiple language requests."""
        mock_get_config.side_effect = [
            {"formal": {"instruction": "HR instruction"}},
            {"formal": {"instruction": "EN instruction"}}
        ]

        hr_result = self.provider.get_formal_prompts("hr")
        en_result = self.provider.get_formal_prompts("en")

        self.assertEqual(hr_result, {"instruction": "HR instruction"})
        self.assertEqual(en_result, {"instruction": "EN instruction"})
        self.assertEqual(mock_get_config.call_count, 2)

    def test_provider_immutability(self):
        """Test provider has no mutable state."""
        provider1 = DefaultLanguageProvider()
        provider2 = DefaultLanguageProvider()

        # Both instances should be functionally identical
        self.assertEqual(type(provider1), type(provider2))
        self.assertEqual(provider1.__dict__, provider2.__dict__)


class TestMockLanguageProvider(unittest.TestCase):
    """Test mock language configuration provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockLanguageProvider()

    def test_provider_creation(self):
        """Test provider can be created."""
        provider = MockLanguageProvider()
        self.assertIsInstance(provider, MockLanguageProvider)

    def test_initial_state(self):
        """Test provider initial state."""
        self.assertEqual(self.provider.formal_prompts, {})
        self.assertEqual(self.provider.error_templates, {})
        self.assertEqual(self.provider.confidence_settings, {})
        self.assertEqual(self.provider.call_log, [])

    def test_set_formal_prompts(self):
        """Test setting formal prompts."""
        prompts = {"instruction": "Test instruction", "response": "Test response"}
        self.provider.set_formal_prompts("hr", prompts)

        self.assertEqual(self.provider.formal_prompts["hr"], prompts)

    def test_set_error_template(self):
        """Test setting error template."""
        template = "Error: {error}"
        self.provider.set_error_template("en", template)

        self.assertEqual(self.provider.error_templates["en"], template)

    def test_set_confidence_settings(self):
        """Test setting confidence settings."""
        settings = {"error_phrases": ["error", "fail"], "threshold": 0.7}
        self.provider.set_confidence_settings("en", settings)

        self.assertEqual(self.provider.confidence_settings["en"], settings)

    def test_get_calls_returns_copy(self):
        """Test that get_calls returns a copy of call log."""
        self.provider.get_formal_prompts("hr")

        calls1 = self.provider.get_calls()
        calls2 = self.provider.get_calls()

        # Should be equal but different objects
        self.assertEqual(calls1, calls2)
        self.assertIsNot(calls1, calls2)

    def test_clear_calls(self):
        """Test clearing call log."""
        self.provider.get_formal_prompts("hr")
        self.assertEqual(len(self.provider.call_log), 1)

        self.provider.clear_calls()
        self.assertEqual(len(self.provider.call_log), 0)

    def test_get_formal_prompts_with_preset_data(self):
        """Test getting formal prompts with preset data."""
        prompts = {"instruction": "Custom instruction"}
        self.provider.set_formal_prompts("hr", prompts)

        result = self.provider.get_formal_prompts("hr")

        self.assertEqual(result, prompts)
        self.assertEqual(len(self.provider.call_log), 1)
        self.assertEqual(self.provider.call_log[0], {
            "method": "get_formal_prompts",
            "language": "hr"
        })

    def test_get_formal_prompts_default_fallback(self):
        """Test getting formal prompts with default fallback."""
        result = self.provider.get_formal_prompts("hr")

        expected = {"formal_instruction": "Mock formal instruction for hr"}
        self.assertEqual(result, expected)
        self.assertEqual(len(self.provider.call_log), 1)

    def test_get_error_template_with_preset_data(self):
        """Test getting error template with preset data."""
        template = "Custom error: {error}"
        self.provider.set_error_template("en", template)

        result = self.provider.get_error_template("en")

        self.assertEqual(result, template)
        self.assertEqual(len(self.provider.call_log), 1)
        self.assertEqual(self.provider.call_log[0], {
            "method": "get_error_template",
            "language": "en"
        })

    def test_get_error_template_default_fallback(self):
        """Test getting error template with default fallback."""
        result = self.provider.get_error_template("en")

        expected = "Mock error template for en: {error}"
        self.assertEqual(result, expected)
        self.assertEqual(len(self.provider.call_log), 1)

    def test_get_confidence_settings_with_preset_data(self):
        """Test getting confidence settings with preset data."""
        settings = {"error_phrases": ["custom_error"], "threshold": 0.9}
        self.provider.set_confidence_settings("hr", settings)

        result = self.provider.get_confidence_settings("hr")

        self.assertEqual(result, settings)
        self.assertEqual(len(self.provider.call_log), 1)
        self.assertEqual(self.provider.call_log[0], {
            "method": "get_confidence_settings",
            "language": "hr"
        })

    def test_get_confidence_settings_default_fallback(self):
        """Test getting confidence settings with default fallback."""
        result = self.provider.get_confidence_settings("hr")

        expected = {"error_phrases": ["mock_error_hr", "test_failure"]}
        self.assertEqual(result, expected)
        self.assertEqual(len(self.provider.call_log), 1)

    def test_call_logging_order(self):
        """Test that calls are logged in correct order."""
        self.provider.get_formal_prompts("hr")
        self.provider.get_error_template("en")
        self.provider.get_confidence_settings("hr")

        calls = self.provider.get_calls()
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["method"], "get_formal_prompts")
        self.assertEqual(calls[1]["method"], "get_error_template")
        self.assertEqual(calls[2]["method"], "get_confidence_settings")

    def test_multiple_language_calls(self):
        """Test calls with multiple languages."""
        self.provider.get_formal_prompts("hr")
        self.provider.get_formal_prompts("en")
        self.provider.get_error_template("hr")

        calls = self.provider.get_calls()
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["language"], "hr")
        self.assertEqual(calls[1]["language"], "en")
        self.assertEqual(calls[2]["language"], "hr")

    def test_call_logging_persistence(self):
        """Test that call log persists across multiple method calls."""
        self.provider.get_formal_prompts("hr")
        self.assertEqual(len(self.provider.call_log), 1)

        self.provider.get_error_template("en")
        self.assertEqual(len(self.provider.call_log), 2)

        self.provider.get_confidence_settings("hr")
        self.assertEqual(len(self.provider.call_log), 3)

    def test_data_isolation_between_languages(self):
        """Test that data is properly isolated between languages."""
        self.provider.set_formal_prompts("hr", {"instruction": "HR instruction"})
        self.provider.set_formal_prompts("en", {"instruction": "EN instruction"})

        hr_result = self.provider.get_formal_prompts("hr")
        en_result = self.provider.get_formal_prompts("en")

        self.assertEqual(hr_result["instruction"], "HR instruction")
        self.assertEqual(en_result["instruction"], "EN instruction")

    def test_provider_mutability(self):
        """Test provider maintains mutable state for testing."""
        # Unlike production provider, mock provider has mutable state
        original_state = len(self.provider.formal_prompts)

        self.provider.set_formal_prompts("test", {"key": "value"})

        new_state = len(self.provider.formal_prompts)
        self.assertEqual(new_state, original_state + 1)

    def test_type_casting_consistency(self):
        """Test that return types are properly cast."""
        # Test dict return type
        prompts = self.provider.get_formal_prompts("hr")
        self.assertIsInstance(prompts, dict)

        # Test str return type
        template = self.provider.get_error_template("en")
        self.assertIsInstance(template, str)

        # Test dict return type
        settings = self.provider.get_confidence_settings("hr")
        self.assertIsInstance(settings, dict)


class TestProviderInterfaces(unittest.TestCase):
    """Test that both providers implement the same interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_provider = DefaultLanguageProvider()
        self.mock_provider = MockLanguageProvider()

    def test_interface_compatibility(self):
        """Test that both providers have the same public interface."""
        default_methods = {name for name in dir(self.default_provider)
                          if not name.startswith('_') and callable(getattr(self.default_provider, name))}

        mock_methods = {name for name in dir(self.mock_provider)
                       if not name.startswith('_') and callable(getattr(self.mock_provider, name))}

        # Mock provider has additional test methods, but should include all default methods
        core_methods = {"get_formal_prompts", "get_error_template", "get_confidence_settings"}
        self.assertTrue(core_methods.issubset(default_methods))
        self.assertTrue(core_methods.issubset(mock_methods))

    def test_method_signatures_compatible(self):
        """Test that core methods have compatible signatures."""
        import inspect

        core_methods = ["get_formal_prompts", "get_error_template", "get_confidence_settings"]

        for method_name in core_methods:
            default_method = getattr(self.default_provider, method_name)
            mock_method = getattr(self.mock_provider, method_name)

            default_sig = inspect.signature(default_method)
            mock_sig = inspect.signature(mock_method)

            # Should have same parameter names (excluding self)
            default_params = list(default_sig.parameters.keys())
            mock_params = list(mock_sig.parameters.keys())

            self.assertEqual(default_params, mock_params,
                           f"Method {method_name} has different signatures")


if __name__ == "__main__":
    unittest.main()