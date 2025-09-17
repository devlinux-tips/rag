"""
Comprehensive tests for ranker provider implementations.
Tests all provider classes, factory functions, and dependency injection patterns.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.retrieval.ranker_providers import (
    MockConfigProvider,
    MockLanguageProvider,
    ConfigProvider,
    ProductionLanguageProvider,
    create_config_provider,
    create_language_provider,
    create_mock_config_provider,
    create_mock_language_provider,
)
from src.retrieval.ranker import LanguageFeatures


class TestMockConfigProvider(unittest.TestCase):
    """Test mock configuration provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "ranking": {
                "method": "test_method",
                "enable_diversity": False,
                "diversity_threshold": 0.5,
            }
        }
        self.provider = MockConfigProvider(self.test_config)

    def test_initialization_with_config(self):
        """Test initialization with configuration."""
        self.assertEqual(self.provider.config_dict, self.test_config)
        self.assertIn("hr", self.provider.language_configs)
        self.assertIn("en", self.provider.language_configs)

    def test_initialization_default_language_configs(self):
        """Test default language configurations are created."""
        # Croatian config
        hr_config = self.provider.language_configs["hr"]["morphology"]
        self.assertIn("important_words", hr_config)
        self.assertIn("zagreb", hr_config["important_words"])
        self.assertIn("quality_positive", hr_config)
        self.assertIn("detaljno", hr_config["quality_positive"])

        # English config
        en_config = self.provider.language_configs["en"]["morphology"]
        self.assertIn("important_words", en_config)
        self.assertIn("important", en_config["important_words"])
        self.assertIn("quality_positive", en_config)
        self.assertIn("detailed", en_config["quality_positive"])

    def test_get_ranking_config_with_custom_config(self):
        """Test ranking config retrieval with custom configuration."""
        result = self.provider.get_ranking_config()

        self.assertEqual(result["method"], "test_method")
        self.assertFalse(result["enable_diversity"])
        self.assertEqual(result["diversity_threshold"], 0.5)

    def test_get_ranking_config_with_defaults(self):
        """Test ranking config retrieval with default configuration."""
        provider = MockConfigProvider({})
        result = provider.get_ranking_config()

        # Should return default configuration
        self.assertEqual(result["method"], "language_enhanced")
        self.assertTrue(result["enable_diversity"])
        self.assertEqual(result["diversity_threshold"], 0.8)
        self.assertFalse(result["boost_recent"])
        self.assertTrue(result["boost_authoritative"])

    def test_get_language_specific_config_croatian(self):
        """Test language-specific config for Croatian."""
        result = self.provider.get_language_specific_config("morphology", "hr")

        expected_config = self.provider.language_configs["hr"]
        self.assertEqual(result, expected_config)

    def test_get_language_specific_config_english(self):
        """Test language-specific config for English."""
        result = self.provider.get_language_specific_config("morphology", "en")

        expected_config = self.provider.language_configs["en"]
        self.assertEqual(result, expected_config)

    def test_get_language_specific_config_unknown_language(self):
        """Test language-specific config for unknown language."""
        result = self.provider.get_language_specific_config("morphology", "unknown")

        # Should return empty dict for unknown language
        self.assertEqual(result, {})

    def test_morphology_structures(self):
        """Test that morphology structures contain expected components."""
        hr_morphology = self.provider.language_configs["hr"]["morphology"]

        # Test Croatian morphology components
        self.assertIsInstance(hr_morphology["important_words"], list)
        self.assertIsInstance(hr_morphology["quality_positive"], list)
        self.assertIsInstance(hr_morphology["quality_negative"], list)
        self.assertIsInstance(hr_morphology["cultural_patterns"], list)
        self.assertIsInstance(hr_morphology["grammar_patterns"], list)

        # Test specific Croatian content
        self.assertIn("hrvatska", hr_morphology["important_words"])
        self.assertIn("dubrovnik", hr_morphology["important_words"])
        self.assertIn("temeljito", hr_morphology["quality_positive"])
        self.assertIn("nejasno", hr_morphology["quality_negative"])
        self.assertIn("biser jadrana", hr_morphology["cultural_patterns"])


class TestMockLanguageProvider(unittest.TestCase):
    """Test mock language provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = MockLanguageProvider()

    def test_initialization(self):
        """Test provider initializes with empty cache."""
        self.assertEqual(self.provider.language_features_cache, {})

    def test_get_language_features_croatian(self):
        """Test Croatian language features retrieval."""
        features = self.provider.get_language_features("hr")

        self.assertIsInstance(features, LanguageFeatures)
        self.assertIn("zagreb", features.importance_words)
        self.assertIn("hrvatska", features.importance_words)
        self.assertIn("dubrovnik", features.importance_words)

        # Test quality indicators structure
        self.assertIn("positive", features.quality_indicators)
        self.assertIn("negative", features.quality_indicators)
        self.assertIsInstance(features.quality_indicators["positive"], list)

        # Test cultural patterns
        self.assertIsInstance(features.cultural_patterns, list)
        self.assertTrue(any("biser jadrana" in pattern for pattern in features.cultural_patterns))

        # Test grammar patterns
        self.assertIsInstance(features.grammar_patterns, list)
        self.assertIn(r"\b\w+ić\b", features.grammar_patterns)

        # Test type weights
        self.assertIn("encyclopedia", features.type_weights)
        self.assertEqual(features.type_weights["encyclopedia"], 1.2)

    def test_get_language_features_english(self):
        """Test English language features retrieval."""
        features = self.provider.get_language_features("en")

        self.assertIsInstance(features, LanguageFeatures)
        self.assertIn("important", features.importance_words)
        self.assertIn("significant", features.importance_words)
        self.assertIn("essential", features.importance_words)

        # Test quality indicators
        positive_patterns = features.quality_indicators["positive"]
        self.assertTrue(any("detailed" in pattern for pattern in positive_patterns))
        self.assertTrue(any("comprehensive" in pattern for pattern in positive_patterns))

        # Test cultural patterns for English
        cultural_patterns = features.cultural_patterns
        self.assertTrue(any("United States" in pattern for pattern in cultural_patterns))
        self.assertTrue(any("technology" in pattern for pattern in cultural_patterns))

        # Test English grammar patterns
        self.assertIn(r"\b\w+ing\b", features.grammar_patterns)
        self.assertIn(r"\b\w+tion\b", features.grammar_patterns)

    def test_get_language_features_unknown_language(self):
        """Test features for unknown language returns defaults."""
        features = self.provider.get_language_features("unknown")

        self.assertIsInstance(features, LanguageFeatures)
        self.assertEqual(len(features.importance_words), 0)
        self.assertEqual(features.quality_indicators["positive"], [])
        self.assertEqual(features.quality_indicators["negative"], [])
        self.assertEqual(features.cultural_patterns, [])
        self.assertEqual(features.grammar_patterns, [])
        self.assertEqual(features.type_weights, {"default": 1.0})

    def test_language_features_caching(self):
        """Test that language features are cached properly."""
        # First call
        features1 = self.provider.get_language_features("hr")

        # Second call should return cached version
        features2 = self.provider.get_language_features("hr")

        # Should be the same object (cached)
        self.assertIs(features1, features2)
        self.assertIn("hr", self.provider.language_features_cache)

    def test_detect_language_content_croatian(self):
        """Test Croatian language detection."""
        croatian_text = "Ovo je tekst sa čćšžđ slovima"
        result = self.provider.detect_language_content(croatian_text)

        self.assertEqual(result["language"], "hr")
        self.assertEqual(result["confidence"], 0.9)

    def test_detect_language_content_english(self):
        """Test English language detection."""
        english_text = "This is English text without special characters"
        result = self.provider.detect_language_content(english_text)

        self.assertEqual(result["language"], "en")
        self.assertEqual(result["confidence"], 0.8)

    def test_detect_language_content_mixed(self):
        """Test language detection with mixed content."""
        mixed_text = "This text has some č characters"
        result = self.provider.detect_language_content(mixed_text)

        # Should detect Croatian due to special characters
        self.assertEqual(result["language"], "hr")
        self.assertEqual(result["confidence"], 0.9)


class TestConfigProvider(unittest.TestCase):
    """Test production configuration provider."""

    def test_initialization(self):
        """Test provider initializes correctly."""
        provider = ConfigProvider()

        # Mock the methods after initialization
        mock_get_ranking = MagicMock()
        mock_get_language = MagicMock()
        provider._get_ranking_config = mock_get_ranking
        provider._get_language_specific_config = mock_get_language

        # Verify attributes are set
        self.assertTrue(hasattr(provider, '_get_ranking_config'))
        self.assertTrue(hasattr(provider, '_get_language_specific_config'))
        self.assertTrue(hasattr(provider, 'logger'))

    def test_get_ranking_config_success(self):
        """Test successful ranking config retrieval."""
        provider = ConfigProvider()

        expected_config = {
            "method": "language_enhanced",
            "enable_diversity": True,
            "diversity_threshold": 0.8,
        }

        # Mock the config loader after initialization
        mock_get_ranking = MagicMock()
        mock_get_ranking.return_value = expected_config
        provider._get_ranking_config = mock_get_ranking

        result = provider.get_ranking_config()

        self.assertEqual(result, expected_config)
        mock_get_ranking.assert_called_once()

    def test_get_ranking_config_missing_config(self):
        """Test ranking config retrieval when config is missing."""
        provider = ConfigProvider()

        # Mock to return None/empty config
        mock_get_ranking = MagicMock()
        mock_get_ranking.return_value = None
        provider._get_ranking_config = mock_get_ranking

        with self.assertRaises(ValueError) as context:
            provider.get_ranking_config()

        self.assertIn("Missing ranking configuration", str(context.exception))

    def test_get_language_specific_config_success(self):
        """Test successful language-specific config retrieval."""
        provider = ConfigProvider()

        expected_config = {
            "morphology": {
                "important_words": ["test", "words"],
                "quality_positive": ["good", "excellent"],
            }
        }

        # Mock the config loader after initialization
        mock_get_language = MagicMock()
        mock_get_language.return_value = expected_config
        provider._get_language_specific_config = mock_get_language

        result = provider.get_language_specific_config("retrieval", "hr")

        self.assertEqual(result, expected_config)
        mock_get_language.assert_called_once_with("retrieval", "hr")

    def test_get_language_specific_config_missing_config(self):
        """Test language-specific config when config is missing."""
        provider = ConfigProvider()

        # Mock to return None/empty config
        mock_get_language = MagicMock()
        mock_get_language.return_value = None
        provider._get_language_specific_config = mock_get_language

        with self.assertRaises(ValueError) as context:
            provider.get_language_specific_config("retrieval", "hr")

        self.assertIn("Missing retrieval configuration for language 'hr'", str(context.exception))


class TestProductionLanguageProvider(unittest.TestCase):
    """Test production language provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config_provider = MagicMock()
        self.provider = ProductionLanguageProvider(self.mock_config_provider)

    def test_initialization(self):
        """Test provider initializes correctly."""
        self.assertIs(self.provider.config_provider, self.mock_config_provider)
        self.assertEqual(self.provider.features_cache, {})
        self.assertTrue(hasattr(self.provider, 'logger'))

    def test_get_language_features_success(self):
        """Test successful language features retrieval."""
        language_config = {
            "morphology": {
                "important_words": ["test", "words"],
                "quality_positive": ["good", "excellent"],
                "quality_negative": ["bad", "poor"],
                "cultural_patterns": ["pattern1", "pattern2"],
                "grammar_patterns": ["\\w+ing", "\\w+ed"],
            }
        }

        self.mock_config_provider.get_language_specific_config.return_value = language_config

        features = self.provider.get_language_features("en")

        self.assertIsInstance(features, LanguageFeatures)
        self.assertIn("test", features.importance_words)
        self.assertIn("words", features.importance_words)
        self.assertEqual(features.quality_indicators["positive"], ["good", "excellent"])
        self.assertEqual(features.quality_indicators["negative"], ["bad", "poor"])
        self.assertEqual(features.cultural_patterns, ["pattern1", "pattern2"])
        self.assertEqual(features.grammar_patterns, ["\\w+ing", "\\w+ed"])

    def test_get_language_features_caching(self):
        """Test language features are cached properly."""
        language_config = {
            "morphology": {
                "important_words": ["test"],
                "quality_positive": ["good"],
                "quality_negative": ["bad"],
            }
        }

        self.mock_config_provider.get_language_specific_config.return_value = language_config

        # First call
        features1 = self.provider.get_language_features("hr")
        # Second call should use cache
        features2 = self.provider.get_language_features("hr")

        self.assertIs(features1, features2)  # Same object (cached)
        # Config provider should only be called once
        self.mock_config_provider.get_language_specific_config.assert_called_once()

    def test_get_language_features_missing_morphology(self):
        """Test language features when morphology section is missing."""
        language_config = {"other_section": {"data": "value"}}

        self.mock_config_provider.get_language_specific_config.return_value = language_config

        with self.assertRaises(ValueError) as context:
            self.provider.get_language_features("hr")

        self.assertIn("Missing 'morphology' section", str(context.exception))

    def test_get_language_features_config_exception(self):
        """Test language features when config provider raises exception."""
        self.mock_config_provider.get_language_specific_config.side_effect = Exception("Config error")

        with self.assertRaises(Exception) as context:
            self.provider.get_language_features("hr")

        self.assertIn("Config error", str(context.exception))

    def test_build_language_features_with_defaults(self):
        """Test building features when config has missing sections."""
        minimal_morphology = {}  # Empty morphology config

        language_config = {"morphology": minimal_morphology}
        self.mock_config_provider.get_language_specific_config.return_value = language_config

        features = self.provider.get_language_features("hr")

        # Should use default Croatian values
        self.assertIn("zagreb", features.importance_words)
        self.assertIn("hrvatska", features.importance_words)

        # Should have default quality indicators
        positive_patterns = features.quality_indicators["positive"]
        self.assertTrue(any("detaljno" in pattern for pattern in positive_patterns))

        # Should have default cultural patterns
        cultural_patterns = features.cultural_patterns
        self.assertTrue(any("biser jadrana" in pattern for pattern in cultural_patterns))

    def test_default_functions_croatian(self):
        """Test default functions return Croatian values."""
        # Test default importance words
        words = self.provider._get_default_importance_words("hr")
        self.assertIn("zagreb", words)
        self.assertIn("hrvatska", words)
        self.assertIn("dubrovnik", words)

        # Test default quality positive
        positive = self.provider._get_default_quality_positive("hr")
        self.assertTrue(any("detaljno" in pattern for pattern in positive))
        self.assertTrue(any("sveobuhvatno" in pattern for pattern in positive))

        # Test default quality negative
        negative = self.provider._get_default_quality_negative("hr")
        self.assertTrue(any("možda" in pattern for pattern in negative))
        self.assertTrue(any("nejasno" in pattern for pattern in negative))

        # Test default cultural patterns
        cultural = self.provider._get_default_cultural_patterns("hr")
        self.assertTrue(any("biser jadrana" in pattern for pattern in cultural))
        self.assertTrue(any("unesco" in pattern for pattern in cultural))

        # Test default grammar patterns
        grammar = self.provider._get_default_grammar_patterns("hr")
        self.assertIn(r"\b\w+ić\b", grammar)
        self.assertIn(r"\b\w+ović\b", grammar)

    def test_default_functions_english(self):
        """Test default functions return English values."""
        # Test default importance words
        words = self.provider._get_default_importance_words("en")
        self.assertIn("important", words)
        self.assertIn("significant", words)
        self.assertIn("essential", words)

        # Test default quality positive
        positive = self.provider._get_default_quality_positive("en")
        self.assertTrue(any("detailed" in pattern for pattern in positive))
        self.assertTrue(any("comprehensive" in pattern for pattern in positive))

        # Test default quality negative
        negative = self.provider._get_default_quality_negative("en")
        self.assertTrue(any("maybe" in pattern for pattern in negative))
        self.assertTrue(any("unclear" in pattern for pattern in negative))

        # Test default cultural patterns
        cultural = self.provider._get_default_cultural_patterns("en")
        self.assertTrue(any("United States" in pattern for pattern in cultural))
        self.assertTrue(any("technology" in pattern for pattern in cultural))

        # Test default grammar patterns
        grammar = self.provider._get_default_grammar_patterns("en")
        self.assertIn(r"\b\w+ing\b", grammar)
        self.assertIn(r"\b\w+tion\b", grammar)

    def test_default_functions_unknown_language(self):
        """Test default functions return empty for unknown language."""
        self.assertEqual(self.provider._get_default_importance_words("unknown"), set())
        self.assertEqual(self.provider._get_default_quality_positive("unknown"), [])
        self.assertEqual(self.provider._get_default_quality_negative("unknown"), [])
        self.assertEqual(self.provider._get_default_cultural_patterns("unknown"), [])
        self.assertEqual(self.provider._get_default_grammar_patterns("unknown"), [])

    @patch('src.preprocessing.cleaners.detect_language_content_with_config')
    def test_detect_language_content_success(self, mock_detect):
        """Test successful language detection."""
        mock_detect.side_effect = [0.9, 0.7]  # hr_confidence, en_confidence

        result = self.provider.detect_language_content("test text")

        self.assertEqual(result["language"], "hr")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(mock_detect.call_count, 2)

    @patch('src.preprocessing.cleaners.detect_language_content_with_config')
    def test_detect_language_content_english_higher(self, mock_detect):
        """Test language detection when English has higher confidence."""
        mock_detect.side_effect = [0.6, 0.8]  # hr_confidence, en_confidence

        result = self.provider.detect_language_content("test text")

        self.assertEqual(result["language"], "en")
        self.assertEqual(result["confidence"], 0.8)

    @patch('src.preprocessing.cleaners.detect_language_content_with_config')
    def test_detect_language_content_fallback(self, mock_detect):
        """Test language detection fallback when detection fails."""
        mock_detect.side_effect = Exception("Detection failed")

        # Test Croatian fallback
        result_hr = self.provider.detect_language_content("tekst sa č karakterima")
        self.assertEqual(result_hr["language"], "hr")
        self.assertEqual(result_hr["confidence"], 0.9)

        # Test English fallback
        result_en = self.provider.detect_language_content("text without special chars")
        self.assertEqual(result_en["language"], "en")
        self.assertEqual(result_en["confidence"], 0.8)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    @patch('src.utils.config_protocol.ConfigProvider')
    def test_create_config_provider(self, mock_full_provider):
        """Test config provider creation."""
        mock_instance = MagicMock()
        mock_full_provider.return_value = mock_instance

        result = create_config_provider()

        self.assertIs(result, mock_instance)
        mock_full_provider.assert_called_once()

    def test_create_language_provider_with_config(self):
        """Test language provider creation with provided config."""
        mock_config = MagicMock()

        result = create_language_provider(mock_config)

        self.assertIsInstance(result, ProductionLanguageProvider)
        self.assertIs(result.config_provider, mock_config)

    @patch('src.retrieval.ranker_providers.create_config_provider')
    def test_create_language_provider_without_config(self, mock_create_config):
        """Test language provider creation without provided config."""
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        result = create_language_provider()

        self.assertIsInstance(result, ProductionLanguageProvider)
        self.assertIs(result.config_provider, mock_config)
        mock_create_config.assert_called_once()

    @patch('src.utils.config_protocol.MockConfigProvider')
    def test_create_mock_config_provider_without_config(self, mock_full_provider):
        """Test mock config provider creation without custom config."""
        mock_instance = MagicMock()
        mock_full_provider.return_value = mock_instance

        result = create_mock_config_provider()

        self.assertIs(result, mock_instance)
        mock_full_provider.assert_called_once()
        # Should not call set_config
        mock_instance.set_config.assert_not_called()

    @patch('src.utils.config_protocol.MockConfigProvider')
    def test_create_mock_config_provider_with_config(self, mock_full_provider):
        """Test mock config provider creation with custom config."""
        mock_instance = MagicMock()
        mock_full_provider.return_value = mock_instance

        config_dict = {"key1": "value1", "key2": "value2"}
        result = create_mock_config_provider(config_dict)

        self.assertIs(result, mock_instance)
        mock_full_provider.assert_called_once()
        # Should call set_config for each key-value pair
        mock_instance.set_config.assert_any_call("key1", "value1")
        mock_instance.set_config.assert_any_call("key2", "value2")
        self.assertEqual(mock_instance.set_config.call_count, 2)

    def test_create_mock_language_provider(self):
        """Test mock language provider creation."""
        result = create_mock_language_provider()

        self.assertIsInstance(result, MockLanguageProvider)
        self.assertEqual(result.language_features_cache, {})


if __name__ == "__main__":
    unittest.main()