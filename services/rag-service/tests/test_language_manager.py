"""
Comprehensive test suite for language_manager_v2.py dependency injection implementation.
Tests pure functions, DI orchestration, mock providers, and integration scenarios.
"""

from typing import Dict, List, Set
from unittest.mock import Mock, patch

import pytest
from src.utils.language_manager import (  # Data classes; Pure functions; DI classes
    DetectionResult, LanguageConfig, LanguagePatterns, LanguageSettings,
    _LanguageManager, build_languages_dict, calculate_collection_suffix,
    calculate_pattern_scores, create_language_config, create_language_manager,
    detect_language_from_text, get_chunk_config_for_language,
    get_display_name_for_language, get_embedding_model_for_language,
    normalize_language_code_pure, normalize_text_for_detection,
    remove_stopwords_from_text, validate_languages_list)
from src.utils.language_manager_providers import (MockConfigProvider,
                                                  MockLoggerProvider,
                                                  MockPatternProvider,
                                                  create_mock_setup,
                                                  create_test_language_manager,
                                                  create_test_patterns,
                                                  create_test_settings)

# ================================
# PURE FUNCTION TESTS
# ================================


class TestPureFunctions:
    """Test all pure business logic functions."""

    def test_create_language_config(self):
        """Test language configuration creation and validation."""
        # Test valid config
        config = create_language_config(
            code="hr",
            name="Croatian",
            native_name="Hrvatski",
            enabled=True,
            embedding_model="BAAI/bge-m3",
            chunk_size=512,
            chunk_overlap=50,
        )

        assert config.code == "hr"
        assert config.name == "Croatian"
        assert config.native_name == "Hrvatski"
        assert config.enabled is True
        assert config.embedding_model == "BAAI/bge-m3"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

        # Test validation errors
        with pytest.raises(ValueError, match="Invalid language code"):
            create_language_config("", "Test", "Test", True, "model", 512, 50)

        with pytest.raises(ValueError, match="Language name required"):
            create_language_config("hr", "", "Test", True, "model", 512, 50)

    def test_build_languages_dict(self):
        """Test building languages dictionary from settings."""
        settings = create_test_settings(
            supported_languages=["hr", "en"], default_language="hr"
        )

        languages = build_languages_dict(settings)

        assert len(languages) == 2
        assert "hr" in languages
        assert "en" in languages
        assert languages["hr"].code == "hr"
        assert languages["hr"].name == "Croatian"
        assert languages["hr"].embedding_model == "BAAI/bge-m3"
        assert languages["en"].code == "en"
        assert languages["en"].name == "English"

    def test_normalize_text_for_detection(self):
        """Test text normalization for language detection."""
        # Test normal text
        text = "Što je ovo? How are you!"
        words = normalize_text_for_detection(text)
        assert words == ["što", "je", "ovo", "how", "are", "you"]

        # Test empty text
        assert normalize_text_for_detection("") == []
        assert normalize_text_for_detection(None) == []

        # Test text with special characters
        text = "Hello, world! This is a test... 123"
        words = normalize_text_for_detection(text)
        assert "hello" in words
        assert "world" in words
        assert "," not in words
        assert "!" not in words

    def test_calculate_pattern_scores(self):
        """Test pattern matching score calculation."""
        words = ["what", "is", "this", "kako", "je"]
        patterns = {"en": ["what", "how", "where"], "hr": ["što", "kako", "gdje"]}

        scores = calculate_pattern_scores(words, patterns)

        assert "en" in scores
        assert "hr" in scores
        assert scores["en"] == 1 / 3  # "what" matches
        assert scores["hr"] == 1 / 3  # "kako" matches

        # Test empty words
        scores = calculate_pattern_scores([], patterns)
        assert scores == {}

    def test_detect_language_from_text(self):
        """Test language detection from text."""
        patterns = {"hr": ["što", "kako", "gdje"], "en": ["what", "how", "where"]}

        # Test Croatian text
        result = detect_language_from_text(
            "Što je ovo kako funkcionira",
            patterns,
            auto_detect=True,
            default_language="hr",
        )

        assert result.detected_language == "hr"
        assert result.confidence > 0
        assert not result.fallback_used
        assert "hr" in result.scores

        # Test English text
        result = detect_language_from_text(
            "What is this how does it work",
            patterns,
            auto_detect=True,
            default_language="hr",
        )

        assert result.detected_language == "en"
        assert result.confidence > 0
        assert not result.fallback_used

        # Test auto_detect disabled
        result = detect_language_from_text(
            "What is this", patterns, auto_detect=False, default_language="hr"
        )

        assert result.detected_language == "hr"
        assert result.fallback_used

        # Test too short text
        result = detect_language_from_text(
            "hi", patterns, auto_detect=True, default_language="hr"
        )

        assert result.detected_language == "hr"
        assert result.fallback_used

    def test_remove_stopwords_from_text(self):
        """Test stopword removal."""
        text = "This is a test with some stopwords"
        stopwords = {"is", "a", "with", "some"}

        result = remove_stopwords_from_text(text, stopwords)
        assert result == "This test stopwords"

        # Test empty stopwords
        result = remove_stopwords_from_text(text, set())
        assert result == text

        # Test empty text
        result = remove_stopwords_from_text("", stopwords)
        assert result == ""

    def test_calculate_collection_suffix(self):
        """Test collection suffix calculation."""
        supported = ["hr", "en", "multilingual"]
        fallback = "hr"

        # Test supported language
        assert calculate_collection_suffix("hr", supported, fallback) == "hr"
        assert calculate_collection_suffix("en", supported, fallback) == "en"

        # Test special cases
        assert calculate_collection_suffix("auto", supported, fallback) == "hr"
        assert (
            calculate_collection_suffix("multilingual", supported, fallback) == "multi"
        )

        # Test unsupported language
        assert calculate_collection_suffix("de", supported, fallback) == "hr"

    def test_normalize_language_code_pure(self):
        """Test pure language code normalization."""
        supported = ["hr", "en", "multilingual"]
        default = "hr"

        # Test supported languages
        result, is_valid = normalize_language_code_pure("hr", supported, default)
        assert result == "hr"
        assert is_valid is True

        result, is_valid = normalize_language_code_pure("en", supported, default)
        assert result == "en"
        assert is_valid is True

        # Test auto
        result, is_valid = normalize_language_code_pure("auto", supported, default)
        assert result == "hr"
        assert is_valid is True

        # Test variations
        result, is_valid = normalize_language_code_pure("HR", supported, default)
        assert result == "hr"
        assert is_valid is True

        result, is_valid = normalize_language_code_pure("hr-HR", supported, default)
        assert result == "hr"
        assert is_valid is True

        # Test unsupported language
        result, is_valid = normalize_language_code_pure("de", supported, default)
        assert result == "hr"
        assert is_valid is False

    def test_validate_languages_list(self):
        """Test language list validation."""
        supported = ["hr", "en", "multilingual"]
        default = "hr"

        # Test valid languages
        result = validate_languages_list(["hr", "en"], supported, default)
        assert result == ["hr", "en"]

        # Test duplicates
        result = validate_languages_list(["hr", "hr", "en"], supported, default)
        assert result == ["hr", "en"]

        # Test with invalid languages
        result = validate_languages_list(["hr", "de", "fr"], supported, default)
        assert result == ["hr"]  # Only valid one kept

        # Test empty list
        result = validate_languages_list([], supported, default)
        assert result == ["hr"]  # Default added

    def test_get_chunk_config_for_language(self):
        """Test chunk configuration retrieval."""
        languages = {
            "hr": create_language_config(
                "hr", "Croatian", "Hrvatski", True, "model", 1024, 100
            ),
            "en": create_language_config(
                "en", "English", "English", True, "model", 512, 50
            ),
        }

        # Test existing language
        chunk_size, overlap = get_chunk_config_for_language(languages, "hr")
        assert chunk_size == 1024
        assert overlap == 100

        # Test non-existing language
        chunk_size, overlap = get_chunk_config_for_language(languages, "de", 256, 25)
        assert chunk_size == 256
        assert overlap == 25

    def test_get_embedding_model_for_language(self):
        """Test embedding model retrieval."""
        languages = {
            "hr": create_language_config(
                "hr", "Croatian", "Hrvatski", True, "model-hr", 512, 50
            ),
            "en": create_language_config(
                "en", "English", "English", True, "model-en", 512, 50
            ),
        }

        # Test existing language
        model = get_embedding_model_for_language(languages, "hr")
        assert model == "model-hr"

        # Test non-existing language
        model = get_embedding_model_for_language(languages, "de", "default-model")
        assert model == "default-model"

    def test_get_display_name_for_language(self):
        """Test display name retrieval."""
        languages = {
            "hr": create_language_config(
                "hr", "Croatian", "Hrvatski", True, "model", 512, 50
            ),
            "en": create_language_config(
                "en", "English", "English", True, "model", 512, 50
            ),
        }

        # Test existing language
        name = get_display_name_for_language(languages, "hr")
        assert name == "Hrvatski"

        # Test non-existing language
        name = get_display_name_for_language(languages, "de")
        assert name == "de"


# ================================
# MOCK PROVIDER TESTS
# ================================


class TestMockProviders:
    """Test mock providers for complete test isolation."""

    def test_mock_config_provider(self):
        """Test mock configuration provider."""
        # Test default settings
        provider = MockConfigProvider()
        settings = provider.get_language_settings()

        assert isinstance(settings, LanguageSettings)
        assert "hr" in settings.supported_languages
        assert "en" in settings.supported_languages
        assert settings.default_language == "hr"
        assert len(provider.call_history) == 1

        # Test custom settings
        custom_settings = create_test_settings(
            supported_languages=["de", "fr"], default_language="de"
        )
        provider.set_settings(custom_settings)
        settings = provider.get_language_settings()

        assert settings.supported_languages == ["de", "fr"]
        assert settings.default_language == "de"

    def test_mock_pattern_provider(self):
        """Test mock pattern provider."""
        # Test default patterns
        provider = MockPatternProvider()
        patterns = provider.get_language_patterns()

        assert isinstance(patterns, LanguagePatterns)
        assert "hr" in patterns.detection_patterns
        assert "en" in patterns.detection_patterns
        assert "što" in patterns.detection_patterns["hr"]
        assert "what" in patterns.detection_patterns["en"]
        assert len(provider.call_history) == 1

        # Test custom patterns
        provider.add_detection_pattern("de", ["was", "wie", "wo"])
        provider.add_stopwords("de", {"der", "die", "das"})
        patterns = provider.get_language_patterns()

        assert "de" in patterns.detection_patterns
        assert "de" in patterns.stopwords
        assert "was" in patterns.detection_patterns["de"]
        assert "der" in patterns.stopwords["de"]

    def test_mock_logger_provider(self):
        """Test mock logger provider message capture."""
        logger = MockLoggerProvider()

        # Test message capture
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        assert len(logger.get_messages("info")) == 1
        assert len(logger.get_messages("debug")) == 1
        assert len(logger.get_messages("warning")) == 1
        assert len(logger.get_messages("error")) == 1

        assert logger.get_messages("info")[0] == "Test info message"
        assert logger.get_messages("debug")[0] == "Test debug message"

        # Test clear messages
        logger.clear_messages()
        assert len(logger.get_messages("info")) == 0
        assert len(logger.get_messages("debug")) == 0


# ================================
# DEPENDENCY INJECTION TESTS
# ================================


class TestDependencyInjection:
    """Test dependency injection orchestration."""

    def test_language_manager_initialization(self):
        """Test language manager initialization with DI."""
        config_provider, pattern_provider, logger_provider = create_mock_setup()

        manager = create_language_manager(
            config_provider=config_provider,
            pattern_provider=pattern_provider,
            logger_provider=logger_provider,
        )

        assert isinstance(manager, _LanguageManager)

        # Verify configuration was loaded
        assert len(config_provider.call_history) >= 1
        assert len(pattern_provider.call_history) >= 1

        # Verify languages were built
        languages = manager.get_supported_languages()
        assert "hr" in languages
        assert "en" in languages

        # Verify logger captured initialization
        init_messages = logger_provider.get_messages("info")
        assert len(init_messages) >= 1
        assert "supported languages" in init_messages[0]

    def test_language_manager_methods(self):
        """Test language manager methods with mocked dependencies."""
        config_provider, pattern_provider, logger_provider = create_mock_setup()
        manager = create_language_manager(
            config_provider=config_provider,
            pattern_provider=pattern_provider,
            logger_provider=logger_provider,
        )

        # Test basic methods
        assert manager.is_language_supported("hr")
        assert manager.is_language_supported("en")
        assert not manager.is_language_supported("de")

        assert manager.get_default_language() == "hr"
        assert manager.get_fallback_language() == "hr"

        # Test language configuration
        config = manager.get_language_config("hr")
        assert config is not None
        assert config.code == "hr"
        assert config.name == "Croatian"

        # Test chunk configuration
        chunk_size, overlap = manager.get_chunk_config("hr")
        assert chunk_size == 512
        assert overlap == 50

        # Test embedding model
        model = manager.get_embedding_model("hr")
        assert model == "BAAI/bge-m3"

    def test_language_detection_with_di(self):
        """Test language detection with dependency injection."""
        # Set up custom patterns for testing
        custom_patterns = {
            "hr": ["što", "kako", "gdje", "hrvatski"],
            "en": ["what", "how", "where", "english"],
        }

        config_provider, pattern_provider, logger_provider = create_mock_setup(
            custom_patterns=custom_patterns
        )

        manager = create_language_manager(
            config_provider=config_provider,
            pattern_provider=pattern_provider,
            logger_provider=logger_provider,
        )

        # Test Croatian detection
        detected = manager.detect_language("Što je ovo kako funkcionira hrvatski")
        assert detected == "hr"

        # Test English detection
        detected = manager.detect_language("What is this how does english work")
        assert detected == "en"

        # Test detailed detection
        result = manager.detect_language_detailed("What is this")
        assert isinstance(result, DetectionResult)
        assert result.detected_language == "en"
        assert result.confidence > 0

        # Verify debug logging
        debug_messages = logger_provider.get_messages("debug")
        assert len(debug_messages) >= 2  # At least 2 detection attempts

    def test_stopwords_with_di(self):
        """Test stopword functionality with dependency injection."""
        custom_stopwords = {
            "hr": {"i", "u", "na", "za"},
            "en": {"a", "and", "the", "of"},
        }

        config_provider, pattern_provider, logger_provider = create_mock_setup(
            custom_stopwords=custom_stopwords
        )

        manager = create_language_manager(
            config_provider=config_provider,
            pattern_provider=pattern_provider,
            logger_provider=logger_provider,
        )

        # Test stopword retrieval
        hr_stopwords = manager.get_stopwords("hr")
        assert "i" in hr_stopwords
        assert "u" in hr_stopwords

        # Test stopword removal
        text = "Ovo je i test u Hrvatskoj za sve"
        filtered = manager.remove_stopwords(text, "hr")
        expected_words = [
            "Ovo",
            "je",
            "test",
            "hrvatskoj",
            "sve",
        ]  # "i", "u", "za" removed
        assert all(word in filtered for word in expected_words)
        assert "i" not in filtered.split()
        assert "u" not in filtered.split()

    def test_collection_suffix_and_normalization(self):
        """Test collection suffix and language normalization."""
        config_provider, pattern_provider, logger_provider = create_mock_setup()
        manager = create_language_manager(
            config_provider=config_provider,
            pattern_provider=pattern_provider,
            logger_provider=logger_provider,
        )

        # Test collection suffix
        assert manager.get_collection_suffix("hr") == "hr"
        assert manager.get_collection_suffix("en") == "en"
        assert manager.get_collection_suffix("auto") == "hr"
        assert manager.get_collection_suffix("multilingual") == "multi"
        assert manager.get_collection_suffix("unknown") == "hr"

        # Test normalization
        assert manager.normalize_language_code("HR") == "hr"
        assert manager.normalize_language_code("hr-HR") == "hr"
        assert manager.normalize_language_code("auto") == "hr"
        assert manager.normalize_language_code("unknown") == "hr"

        # Verify warning logged for unknown language
        warning_messages = logger_provider.get_messages("warning")
        assert len(warning_messages) >= 1
        assert "unknown" in warning_messages[0] or "Unsupported" in warning_messages[0]

    def test_runtime_language_addition(self):
        """Test adding languages at runtime."""
        config_provider, pattern_provider, logger_provider = create_mock_setup()
        manager = create_language_manager(
            config_provider=config_provider,
            pattern_provider=pattern_provider,
            logger_provider=logger_provider,
        )

        # Add new language
        manager.add_language_runtime(
            code="de",
            name="German",
            native_name="Deutsch",
            patterns=["was", "wie", "wo"],
            stopwords=["der", "die", "das"],
        )

        # Verify language was added
        assert manager.is_language_supported("de")
        config = manager.get_language_config("de")
        assert config.name == "German"
        assert config.native_name == "Deutsch"

        # Verify patterns were added
        de_stopwords = manager.get_stopwords("de")
        assert "der" in de_stopwords

        # Verify info logging
        info_messages = logger_provider.get_messages("info")
        add_messages = [
            msg
            for msg in info_messages
            if "Adding runtime" in msg or "added successfully" in msg
        ]
        assert len(add_messages) >= 2  # Adding + added successfully


# ================================
# INTEGRATION TESTS
# ================================


class TestIntegration:
    """Test complete integration scenarios."""

    def test_factory_functions(self):
        """Test factory function integration."""
        # Test mock setup factory
        manager, (
            config_provider,
            pattern_provider,
            logger_provider,
        ) = create_test_language_manager()

        assert isinstance(manager, _LanguageManager)
        assert isinstance(config_provider, MockConfigProvider)
        assert isinstance(pattern_provider, MockPatternProvider)
        assert isinstance(logger_provider, MockLoggerProvider)

        # Test functionality works
        languages = manager.get_supported_languages()
        assert len(languages) >= 2

        detected = manager.detect_language("What is this test")
        assert detected in languages

    def test_custom_configuration_integration(self):
        """Test integration with custom configuration."""
        # Create custom settings
        custom_settings = create_test_settings(
            supported_languages=["de", "fr", "es"],
            default_language="de",
            auto_detect=False,
        )

        custom_patterns = create_test_patterns(
            detection_patterns={
                "de": ["was", "wie", "wo"],
                "fr": ["quoi", "comment", "où"],
                "es": ["qué", "cómo", "dónde"],
            },
            stopwords={
                "de": {"der", "die", "das"},
                "fr": {"le", "la", "les"},
                "es": {"el", "la", "los"},
            },
        )

        # Create manager with custom configuration
        manager, providers = create_test_language_manager(
            settings=custom_settings, patterns=custom_patterns
        )

        # Verify custom configuration
        languages = manager.get_supported_languages()
        assert set(languages) == {"de", "fr", "es"}
        assert manager.get_default_language() == "de"

        # Test detection with auto_detect disabled
        detected = manager.detect_language("Quoi est-ce que c'est")
        assert detected == "de"  # Should use default because auto_detect=False

        # Test stopwords
        de_stopwords = manager.get_stopwords("de")
        assert "der" in de_stopwords
