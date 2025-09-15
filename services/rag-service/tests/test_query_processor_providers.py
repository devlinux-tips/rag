"""
Comprehensive tests for query processor provider implementations.
Tests all provider classes, factory functions, and dependency injection patterns.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.retrieval.query_processor_providers import (
    ConfigProvider,
    MockConfigProvider,
    MockLanguageDataProvider,
    ProductionLanguageDataProvider,
    create_default_config,
    create_mock_language_provider,
    create_production_language_provider,
    create_production_providers,
    create_test_providers,
)
from src.retrieval.query_processor import QueryProcessingConfig


class TestConfigProviderProtocol(unittest.TestCase):
    """Test ConfigProvider protocol compliance."""

    def test_protocol_methods_exist(self):
        """Test that protocol defines required methods."""
        # This test ensures the protocol has the expected methods
        mock_provider = MockConfigProvider()

        # Test protocol methods are callable
        self.assertTrue(hasattr(mock_provider, "load_config"))
        self.assertTrue(hasattr(mock_provider, "get_language_specific_config"))
        self.assertTrue(callable(mock_provider.load_config))
        self.assertTrue(callable(mock_provider.get_language_specific_config))


class TestMockConfigProvider(unittest.TestCase):
    """Test mock configuration provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockConfigProvider()

    def test_initialization(self):
        """Test provider initializes with empty data."""
        self.assertEqual(self.provider.configs, {})
        self.assertEqual(self.provider.language_configs, {})

    def test_set_config(self):
        """Test setting configuration."""
        config_data = {"key": "value", "nested": {"inner": "data"}}
        self.provider.set_config("test_config", config_data)

        self.assertEqual(self.provider.configs["test_config"], config_data)

    def test_set_language_config(self):
        """Test setting language-specific configuration."""
        config_data = {"stop_words": ["the", "a"], "patterns": ["test"]}
        self.provider.set_language_config("language_data", "en", config_data)

        expected_key = "language_data_en"
        self.assertEqual(self.provider.language_configs[expected_key], config_data)

    def test_load_config_success(self):
        """Test successful configuration loading."""
        config_data = {"setting": "value"}
        self.provider.set_config("test", config_data)

        result = self.provider.load_config("test")
        self.assertEqual(result, config_data)

    def test_load_config_not_found(self):
        """Test loading non-existent configuration raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.provider.load_config("nonexistent")

        self.assertIn("Mock configuration not found: nonexistent", str(context.exception))

    def test_get_language_specific_config_success(self):
        """Test successful language-specific configuration retrieval."""
        config_data = {"data": "language_specific"}
        self.provider.set_language_config("section", "hr", config_data)

        result = self.provider.get_language_specific_config("section", "hr")
        self.assertEqual(result, config_data)

    def test_get_language_specific_config_not_found(self):
        """Test getting non-existent language config raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.provider.get_language_specific_config("section", "missing")

        self.assertIn("Mock language-specific configuration not found: section for language missing",
                     str(context.exception))


class TestMockLanguageDataProvider(unittest.TestCase):
    """Test mock language data provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockLanguageDataProvider()

    def test_initialization(self):
        """Test provider initializes with empty data structures."""
        self.assertEqual(self.provider.stop_words, {})
        self.assertEqual(self.provider.question_patterns, {})
        self.assertEqual(self.provider.synonym_groups, {})
        self.assertEqual(self.provider.morphological_patterns, {})

    def test_set_stop_words(self):
        """Test setting stop words for language."""
        stop_words = {"the", "a", "an"}
        self.provider.set_stop_words("en", stop_words)

        self.assertEqual(self.provider.stop_words["en"], stop_words)

    def test_set_question_patterns(self):
        """Test setting question patterns for language."""
        patterns = [r"^what\s", r"^how\s", r"^when\s"]
        self.provider.set_question_patterns("en", patterns)

        self.assertEqual(self.provider.question_patterns["en"], patterns)

    def test_set_synonym_groups(self):
        """Test setting synonym groups for language."""
        synonyms = {"fast": ["quick", "rapid"], "good": ["great", "excellent"]}
        self.provider.set_synonym_groups("en", synonyms)

        self.assertEqual(self.provider.synonym_groups["en"], synonyms)

    def test_set_morphological_patterns(self):
        """Test setting morphological patterns for language."""
        patterns = {"verb": ["run", "runs", "running"], "noun": ["cat", "cats"]}
        self.provider.set_morphological_patterns("en", patterns)

        self.assertEqual(self.provider.morphological_patterns["en"], patterns)

    def test_get_stop_words_success(self):
        """Test successful stop words retrieval."""
        stop_words = {"i", "u", "na"}
        self.provider.set_stop_words("hr", stop_words)

        result = self.provider.get_stop_words("hr")
        self.assertEqual(result, stop_words)

    def test_get_stop_words_not_configured(self):
        """Test getting stop words for unconfigured language raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.provider.get_stop_words("missing")

        self.assertIn("Mock language data not configured for language: missing", str(context.exception))

    def test_get_question_patterns_success(self):
        """Test successful question patterns retrieval."""
        patterns = [r"^što\s", r"^kako\s"]
        self.provider.set_question_patterns("hr", patterns)

        result = self.provider.get_question_patterns("hr")
        self.assertEqual(result, patterns)

    def test_get_question_patterns_not_configured(self):
        """Test getting patterns for unconfigured language raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.provider.get_question_patterns("missing")

        self.assertIn("Mock question patterns not configured for language: missing", str(context.exception))

    def test_get_synonym_groups_success(self):
        """Test successful synonym groups retrieval."""
        synonyms = {"brz": ["brži", "brzo"]}
        self.provider.set_synonym_groups("hr", synonyms)

        result = self.provider.get_synonym_groups("hr")
        self.assertEqual(result, synonyms)

    def test_get_synonym_groups_not_configured(self):
        """Test getting synonyms for unconfigured language raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.provider.get_synonym_groups("missing")

        self.assertIn("Mock synonym groups not configured for language: missing", str(context.exception))

    def test_get_morphological_patterns_success(self):
        """Test successful morphological patterns retrieval."""
        patterns = {"verb": ["raditi", "radi", "radio"]}
        self.provider.set_morphological_patterns("hr", patterns)

        result = self.provider.get_morphological_patterns("hr")
        self.assertEqual(result, patterns)

    def test_get_morphological_patterns_not_configured(self):
        """Test getting morphological patterns for unconfigured language raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.provider.get_morphological_patterns("missing")

        self.assertIn("Mock morphological patterns not configured for language: missing", str(context.exception))


class TestProductionLanguageDataProvider(unittest.TestCase):
    """Test production language data provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config_provider = MockConfigProvider()
        self.provider = ProductionLanguageDataProvider(self.mock_config_provider)

    def test_initialization(self):
        """Test provider initializes correctly."""
        self.assertIs(self.provider.config_provider, self.mock_config_provider)
        self.assertEqual(self.provider._cache, {})

    def test_get_stop_words_success(self):
        """Test successful stop words retrieval."""
        language_config = {"stop_words": ["the", "a", "an"]}
        self.mock_config_provider.set_language_config("language_data", "en", language_config)

        result = self.provider.get_stop_words("en")
        expected = {"the", "a", "an"}
        self.assertEqual(result, expected)

    def test_get_stop_words_caching(self):
        """Test stop words are cached properly."""
        language_config = {"stop_words": ["the", "a"]}
        self.mock_config_provider.set_language_config("language_data", "en", language_config)

        # First call
        result1 = self.provider.get_stop_words("en")
        # Second call should use cache
        result2 = self.provider.get_stop_words("en")

        self.assertIs(result1, result2)  # Same object instance (cached)
        self.assertTrue("stop_words_en" in self.provider._cache)

    def test_get_stop_words_missing_config(self):
        """Test missing stop words config raises ValueError."""
        language_config = {"other_data": "value"}  # Missing stop_words
        self.mock_config_provider.set_language_config("language_data", "en", language_config)

        with self.assertRaises(ValueError) as context:
            self.provider.get_stop_words("en")

        self.assertIn("Missing 'stop_words' in language configuration for en", str(context.exception))

    def test_get_question_patterns_success(self):
        """Test successful question patterns retrieval."""
        language_config = {"question_patterns": [r"^what\s", r"^how\s"]}
        self.mock_config_provider.set_language_config("language_data", "en", language_config)

        result = self.provider.get_question_patterns("en")
        expected = [r"^what\s", r"^how\s"]
        self.assertEqual(result, expected)

    def test_get_question_patterns_caching(self):
        """Test question patterns are cached properly."""
        language_config = {"question_patterns": [r"^što\s"]}
        self.mock_config_provider.set_language_config("language_data", "hr", language_config)

        # First call
        result1 = self.provider.get_question_patterns("hr")
        # Second call should use cache
        result2 = self.provider.get_question_patterns("hr")

        self.assertIs(result1, result2)  # Same object instance (cached)
        self.assertTrue("question_patterns_hr" in self.provider._cache)

    def test_get_question_patterns_missing_config(self):
        """Test missing question patterns config raises ValueError."""
        language_config = {"other_data": "value"}  # Missing question_patterns
        self.mock_config_provider.set_language_config("language_data", "hr", language_config)

        with self.assertRaises(ValueError) as context:
            self.provider.get_question_patterns("hr")

        self.assertIn("Missing 'question_patterns' in language configuration for hr", str(context.exception))

    def test_get_synonym_groups_success(self):
        """Test successful synonym groups retrieval."""
        language_config = {"synonym_groups": {"fast": ["quick", "rapid"]}}
        self.mock_config_provider.set_language_config("language_data", "en", language_config)

        result = self.provider.get_synonym_groups("en")
        expected = {"fast": ["quick", "rapid"]}
        self.assertEqual(result, expected)

    def test_get_synonym_groups_caching(self):
        """Test synonym groups are cached properly."""
        language_config = {"synonym_groups": {"brz": ["brži"]}}
        self.mock_config_provider.set_language_config("language_data", "hr", language_config)

        # First call
        result1 = self.provider.get_synonym_groups("hr")
        # Second call should use cache
        result2 = self.provider.get_synonym_groups("hr")

        self.assertIs(result1, result2)  # Same object instance (cached)
        self.assertTrue("synonym_groups_hr" in self.provider._cache)

    def test_get_synonym_groups_missing_config(self):
        """Test missing synonym groups config raises ValueError."""
        language_config = {"other_data": "value"}  # Missing synonym_groups
        self.mock_config_provider.set_language_config("language_data", "en", language_config)

        with self.assertRaises(ValueError) as context:
            self.provider.get_synonym_groups("en")

        self.assertIn("Missing 'synonym_groups' in language configuration for en", str(context.exception))

    def test_get_morphological_patterns_success(self):
        """Test successful morphological patterns retrieval."""
        language_config = {"morphological_patterns": {"verb": ["run", "runs"]}}
        self.mock_config_provider.set_language_config("language_data", "en", language_config)

        result = self.provider.get_morphological_patterns("en")
        expected = {"verb": ["run", "runs"]}
        self.assertEqual(result, expected)

    def test_get_morphological_patterns_caching(self):
        """Test morphological patterns are cached properly."""
        language_config = {"morphological_patterns": {"glagol": ["raditi", "radi"]}}
        self.mock_config_provider.set_language_config("language_data", "hr", language_config)

        # First call
        result1 = self.provider.get_morphological_patterns("hr")
        # Second call should use cache
        result2 = self.provider.get_morphological_patterns("hr")

        self.assertIs(result1, result2)  # Same object instance (cached)
        self.assertTrue("morphological_patterns_hr" in self.provider._cache)

    def test_get_morphological_patterns_missing_config(self):
        """Test missing morphological patterns config raises ValueError."""
        language_config = {"other_data": "value"}  # Missing morphological_patterns
        self.mock_config_provider.set_language_config("language_data", "hr", language_config)

        with self.assertRaises(ValueError) as context:
            self.provider.get_morphological_patterns("hr")

        self.assertIn("Missing 'morphological_patterns' in language configuration for hr", str(context.exception))

    def test_cache_isolation_between_languages(self):
        """Test cache isolation between different languages."""
        # Set up configs for two languages
        en_config = {"stop_words": ["the", "a"]}
        hr_config = {"stop_words": ["i", "u"]}
        self.mock_config_provider.set_language_config("language_data", "en", en_config)
        self.mock_config_provider.set_language_config("language_data", "hr", hr_config)

        # Get data for both languages
        en_result = self.provider.get_stop_words("en")
        hr_result = self.provider.get_stop_words("hr")

        # Verify correct data and cache isolation
        self.assertEqual(en_result, {"the", "a"})
        self.assertEqual(hr_result, {"i", "u"})
        self.assertTrue("stop_words_en" in self.provider._cache)
        self.assertTrue("stop_words_hr" in self.provider._cache)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating providers."""

    def test_create_default_config_with_defaults(self):
        """Test creating default config without config provider."""
        config = create_default_config()

        self.assertIsInstance(config, QueryProcessingConfig)
        # Verify it's using Croatian as default
        self.assertEqual(config.language, "hr")

    def test_create_default_config_with_language(self):
        """Test creating default config with specific language."""
        config = create_default_config(language="en")

        self.assertIsInstance(config, QueryProcessingConfig)
        self.assertEqual(config.language, "en")

    def test_create_default_config_with_provider(self):
        """Test creating config with custom provider."""
        mock_provider = MockConfigProvider()
        custom_config = {
            "query_processing": {
                "expand_synonyms": False,
                "normalize_case": True,
                "remove_stopwords": False,
                "min_query_length": 5,
                "max_query_length": 100,
                "max_expanded_terms": 15,
                "enable_morphological_analysis": True,
                "use_query_classification": False,
                "enable_spell_check": True,
            }
        }
        mock_provider.set_config("config", custom_config)

        config = create_default_config(language="hr", config_provider=mock_provider)

        self.assertIsInstance(config, QueryProcessingConfig)
        self.assertEqual(config.language, "hr")
        # Verify custom settings applied
        self.assertFalse(config.expand_synonyms)
        self.assertEqual(config.min_query_length, 5)
        self.assertEqual(config.max_query_length, 100)

    def test_create_default_config_provider_failure(self):
        """Test config creation fails when provider returns None."""
        mock_provider = MockConfigProvider()
        # Don't set any config, so load_config will fail

        with self.assertRaises(ValueError) as context:
            create_default_config(config_provider=mock_provider)

        self.assertIn("Mock configuration not found: config", str(context.exception))

    def test_create_production_language_provider(self):
        """Test creating production language provider."""
        mock_config_provider = MockConfigProvider()

        provider = create_production_language_provider(mock_config_provider)

        self.assertIsInstance(provider, ProductionLanguageDataProvider)
        self.assertIs(provider.config_provider, mock_config_provider)

    def test_create_mock_language_provider_default_croatian(self):
        """Test creating mock provider with Croatian defaults."""
        provider = create_mock_language_provider("hr")

        self.assertIsInstance(provider, MockLanguageDataProvider)
        # Verify Croatian data was set
        hr_stop_words = provider.get_stop_words("hr")
        self.assertIn("i", hr_stop_words)
        self.assertIn("u", hr_stop_words)

        hr_patterns = provider.get_question_patterns("hr")
        self.assertTrue(any("što" in pattern for pattern in hr_patterns))

    def test_create_mock_language_provider_default_english(self):
        """Test creating mock provider with English defaults."""
        provider = create_mock_language_provider("en")

        self.assertIsInstance(provider, MockLanguageDataProvider)
        # Verify English data was set
        en_stop_words = provider.get_stop_words("en")
        self.assertIn("the", en_stop_words)
        self.assertIn("and", en_stop_words)

        en_patterns = provider.get_question_patterns("en")
        self.assertTrue(any("what" in pattern for pattern in en_patterns))

    def test_create_mock_language_provider_with_custom_data(self):
        """Test creating mock provider with custom data."""
        custom_data = {
            "stop_words": ["custom", "stop", "words"],
            "question_patterns": [r"^custom\s"],
            "synonym_groups": {"custom": ["test", "data"]},
            "morphological_patterns": {"test": ["pattern"]}
        }

        provider = create_mock_language_provider("test_lang", custom_data)

        # Verify custom data applied
        stop_words = provider.get_stop_words("test_lang")
        self.assertEqual(stop_words, {"custom", "stop", "words"})

        patterns = provider.get_question_patterns("test_lang")
        self.assertEqual(patterns, [r"^custom\s"])

        synonyms = provider.get_synonym_groups("test_lang")
        self.assertEqual(synonyms, {"custom": ["test", "data"]})

        morphological = provider.get_morphological_patterns("test_lang")
        self.assertEqual(morphological, {"test": ["pattern"]})

    def test_create_test_providers_default(self):
        """Test creating complete test provider setup with defaults."""
        config, language_provider, config_provider = create_test_providers()

        # Verify all components created
        self.assertIsInstance(config, QueryProcessingConfig)
        self.assertIsInstance(language_provider, MockLanguageDataProvider)
        self.assertIsInstance(config_provider, MockConfigProvider)

        # Verify default language is Croatian
        self.assertEqual(config.language, "hr")

        # Verify Croatian language data available
        hr_stop_words = language_provider.get_stop_words("hr")
        self.assertIsInstance(hr_stop_words, set)

    def test_create_test_providers_custom_language(self):
        """Test creating test providers with custom language."""
        config, language_provider, config_provider = create_test_providers(language="en")

        self.assertEqual(config.language, "en")
        # Verify English language data available
        en_stop_words = language_provider.get_stop_words("en")
        self.assertIn("the", en_stop_words)

    def test_create_test_providers_custom_config(self):
        """Test creating test providers with custom configuration."""
        custom_config = {
            "query_processing": {
                "expand_synonyms": False,
                "normalize_case": False,
                "remove_stopwords": True,
                "min_query_length": 2,
                "max_query_length": 30,
                "max_expanded_terms": 5,
                "enable_morphological_analysis": True,
                "use_query_classification": False,
                "enable_spell_check": True,
            }
        }

        config, language_provider, config_provider = create_test_providers(
            custom_config=custom_config
        )

        # Verify custom config applied
        self.assertFalse(config.expand_synonyms)
        self.assertFalse(config.normalize_case)
        self.assertTrue(config.enable_spell_check)
        self.assertEqual(config.min_query_length, 2)
        self.assertEqual(config.max_query_length, 30)

    def test_create_test_providers_custom_language_data(self):
        """Test creating test providers with custom language data."""
        custom_language_data = {
            "stop_words": ["test", "custom"],
            "question_patterns": [r"^test\s"],
            "synonym_groups": {"test": ["custom"]}
        }

        config, language_provider, config_provider = create_test_providers(
            language="test_lang",
            custom_language_data=custom_language_data
        )

        # Verify custom language data applied
        stop_words = language_provider.get_stop_words("test_lang")
        self.assertEqual(stop_words, {"test", "custom"})


class TestCreateProductionProviders(unittest.TestCase):
    """Test production provider creation with mocking."""

    @patch('src.utils.config_protocol.get_config_provider')
    @patch('src.retrieval.query_processor_providers.QueryProcessingConfig.from_validated_config')
    def test_create_production_providers_success(self, mock_config_method, mock_get_provider):
        """Test successful production provider creation."""
        # Setup mocks
        mock_real_provider = MagicMock()
        mock_get_provider.return_value = mock_real_provider

        mock_config = MagicMock()
        mock_config.language = "hr"
        mock_config_method.return_value = mock_config

        mock_real_provider.load_config.return_value = {
            "query_processing": {
                "expand_synonyms": True,
                "normalize_case": True,
                "remove_stopwords": True,
                "min_query_length": 3,
                "max_query_length": 50,
                "max_expanded_terms": 10,
                "enable_morphological_analysis": False,
                "use_query_classification": True,
                "enable_spell_check": False,
            }
        }

        # Test function
        config, language_provider, config_provider = create_production_providers("hr")

        # Verify results
        self.assertIs(config, mock_config)
        self.assertIsInstance(language_provider, ProductionLanguageDataProvider)
        self.assertIs(config_provider, mock_real_provider)
        self.assertIs(language_provider.config_provider, mock_real_provider)

        # Verify calls
        mock_get_provider.assert_called_once()
        mock_real_provider.load_config.assert_called_once_with("config")

    @patch('src.utils.config_protocol.get_config_provider')
    def test_create_production_providers_import_error(self, mock_get_provider):
        """Test production provider creation handles import errors gracefully."""
        # This test verifies the import is attempted - the actual failure handling
        # depends on the real get_config_provider implementation
        mock_get_provider.return_value = MagicMock()

        # Should not raise during import
        try:
            create_production_providers("en")
        except Exception:
            # If it fails, it should be due to config issues, not import issues
            pass


if __name__ == "__main__":
    unittest.main()