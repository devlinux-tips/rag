"""
Tests for language manager module.
Comprehensive testing of pure functions and dependency injection architecture.
"""

import pytest
from unittest.mock import Mock

from src.utils.language_manager import (
    # Data classes
    LanguageConfig,
    LanguageSettings,
    LanguagePatterns,
    DetectionResult,
    # Protocols
    ConfigProvider,
    PatternProvider,
    LoggerProvider,
    # Pure functions
    create_language_config,
    build_languages_dict,
    normalize_text_for_detection,
    calculate_pattern_scores,
    detect_language_from_text,
    remove_stopwords_from_text,
    calculate_collection_suffix,
    normalize_language_code_pure,
    validate_languages_list,
    get_chunk_config_for_language,
    get_embedding_model_for_language,
    get_display_name_for_language,
    # Main class and factory
    _LanguageManager,
    create_language_manager,
    LanguageManager,
)


class TestLanguageConfig:
    """Test LanguageConfig data class."""

    def test_valid_config_creation(self):
        """Test creating valid language configuration."""
        config = LanguageConfig(
            code="hr",
            name="Croatian",
            native_name="Hrvatski",
            enabled=True,
            embedding_model="classla/bcms-bertic",
            chunk_size=512,
            chunk_overlap=50
        )

        assert config.code == "hr"
        assert config.name == "Croatian"
        assert config.native_name == "Hrvatski"
        assert config.enabled is True
        assert config.embedding_model == "classla/bcms-bertic"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_config_validation_empty_code(self):
        """Test configuration validation with empty code."""
        with pytest.raises(ValueError, match="Invalid language code"):
            LanguageConfig(
                code="",
                name="Test",
                native_name="Test",
                enabled=True,
                embedding_model="test",
                chunk_size=512,
                chunk_overlap=50
            )

    def test_config_validation_short_code(self):
        """Test configuration validation with too short code."""
        with pytest.raises(ValueError, match="Invalid language code: x"):
            LanguageConfig(
                code="x",
                name="Test",
                native_name="Test",
                enabled=True,
                embedding_model="test",
                chunk_size=512,
                chunk_overlap=50
            )

    def test_config_validation_empty_name(self):
        """Test configuration validation with empty name."""
        with pytest.raises(ValueError, match="Language name required for hr"):
            LanguageConfig(
                code="hr",
                name="",
                native_name="Hrvatski",
                enabled=True,
                embedding_model="test",
                chunk_size=512,
                chunk_overlap=50
            )


class TestLanguageSettings:
    """Test LanguageSettings data class."""

    def test_language_settings_creation(self):
        """Test creating language settings."""
        settings = LanguageSettings(
            supported_languages=["hr", "en"],
            default_language="hr",
            auto_detect=True,
            fallback_language="en",
            language_names={"hr": "Croatian", "en": "English"},
            embedding_model="BAAI/bge-m3",
            chunk_size=512,
            chunk_overlap=50
        )

        assert settings.supported_languages == ["hr", "en"]
        assert settings.default_language == "hr"
        assert settings.auto_detect is True
        assert settings.fallback_language == "en"
        assert settings.language_names == {"hr": "Croatian", "en": "English"}


class TestLanguagePatterns:
    """Test LanguagePatterns data class."""

    def test_language_patterns_creation(self):
        """Test creating language patterns."""
        patterns = LanguagePatterns(
            detection_patterns={
                "hr": ["je", "da", "i", "u"],
                "en": ["the", "and", "of", "to"]
            },
            stopwords={
                "hr": {"je", "da", "i"},
                "en": {"the", "and", "of"}
            }
        )

        assert "hr" in patterns.detection_patterns
        assert "en" in patterns.detection_patterns
        assert len(patterns.stopwords["hr"]) == 3
        assert len(patterns.stopwords["en"]) == 3


class TestDetectionResult:
    """Test DetectionResult data class."""

    def test_detection_result_basic(self):
        """Test basic detection result."""
        result = DetectionResult(
            detected_language="hr",
            confidence=0.85,
            scores={"hr": 0.85, "en": 0.15}
        )

        assert result.detected_language == "hr"
        assert result.confidence == 0.85
        assert result.scores == {"hr": 0.85, "en": 0.15}
        assert result.fallback_used is False

    def test_detection_result_with_fallback(self):
        """Test detection result with fallback flag."""
        result = DetectionResult(
            detected_language="hr",
            confidence=0.0,
            scores={},
            fallback_used=True
        )

        assert result.fallback_used is True
        assert result.confidence == 0.0
        assert result.scores == {}


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_create_language_config(self):
        """Test pure language config creation function."""
        config = create_language_config(
            code="hr",
            name="Croatian",
            native_name="Hrvatski",
            enabled=True,
            embedding_model="classla/bcms-bertic",
            chunk_size=512,
            chunk_overlap=50
        )

        assert isinstance(config, LanguageConfig)
        assert config.code == "hr"
        assert config.name == "Croatian"

    def test_build_languages_dict_success(self):
        """Test building languages dictionary from settings."""
        settings = LanguageSettings(
            supported_languages=["hr", "en"],
            default_language="hr",
            auto_detect=True,
            fallback_language="en",
            language_names={"hr": "Croatian", "en": "English"},
            embedding_model="BAAI/bge-m3",
            chunk_size=512,
            chunk_overlap=50
        )

        languages = build_languages_dict(settings)

        assert len(languages) == 2
        assert "hr" in languages
        assert "en" in languages
        assert languages["hr"].name == "Croatian"
        assert languages["en"].name == "English"
        assert languages["hr"].embedding_model == "BAAI/bge-m3"

    def test_build_languages_dict_missing_name(self):
        """Test building languages dictionary with missing name."""
        settings = LanguageSettings(
            supported_languages=["hr", "en"],
            default_language="hr",
            auto_detect=True,
            fallback_language="en",
            language_names={"hr": "Croatian"},  # Missing "en"
            embedding_model="BAAI/bge-m3",
            chunk_size=512,
            chunk_overlap=50
        )

        with pytest.raises(ValueError, match="Language name missing for en"):
            build_languages_dict(settings)

    def test_normalize_text_for_detection_basic(self):
        """Test text normalization for detection."""
        text = "Ovo je hrvatski tekst!"
        words = normalize_text_for_detection(text)

        assert words == ["ovo", "je", "hrvatski", "tekst"]

    def test_normalize_text_for_detection_empty(self):
        """Test text normalization with empty input."""
        assert normalize_text_for_detection("") == []
        assert normalize_text_for_detection(None) == []

    def test_normalize_text_for_detection_special_chars(self):
        """Test text normalization with special characters."""
        text = "Test! @#$ 123 [brackets] (parens)"
        words = normalize_text_for_detection(text)

        # Should remove special characters and normalize
        assert "test" in words
        assert "123" in words
        assert "brackets" in words
        assert "parens" in words
        # Special characters should be removed
        assert "@#$" not in " ".join(words)

    def test_calculate_pattern_scores_basic(self):
        """Test pattern score calculation."""
        words = ["ovo", "je", "hrvatski", "tekst"]
        patterns = {
            "hr": ["je", "ovo", "i", "da"],
            "en": ["the", "and", "of", "to"]
        }

        scores = calculate_pattern_scores(words, patterns)

        assert "hr" in scores
        assert "en" not in scores  # No English patterns matched
        assert scores["hr"] == 0.5  # 2 out of 4 patterns matched

    def test_calculate_pattern_scores_empty_words(self):
        """Test pattern scores with empty words."""
        scores = calculate_pattern_scores([], {"hr": ["je"]})
        assert scores == {}

    def test_calculate_pattern_scores_empty_patterns(self):
        """Test pattern scores with empty patterns."""
        words = ["test"]
        patterns = {"hr": []}

        scores = calculate_pattern_scores(words, patterns)
        assert scores == {}

    def test_detect_language_from_text_success(self):
        """Test successful language detection."""
        text = "Ovo je hrvatski tekst koji sadrži mnoge hrvatske riječi"
        patterns = {
            "hr": ["je", "ovo", "koji", "hrvatske"],
            "en": ["the", "and", "of", "to"]
        }

        result = detect_language_from_text(text, patterns, auto_detect=True)

        assert result.detected_language == "hr"
        assert result.confidence > 0
        assert result.fallback_used is False
        assert "hr" in result.scores

    def test_detect_language_from_text_fallback_short(self):
        """Test language detection fallback for short text."""
        text = "Hi"  # Too short
        patterns = {"hr": ["je"], "en": ["the"]}

        result = detect_language_from_text(text, patterns, default_language="hr")

        assert result.detected_language == "hr"
        assert result.confidence == 0.0
        assert result.fallback_used is True

    def test_detect_language_from_text_no_patterns_match(self):
        """Test language detection when no patterns match."""
        text = "xyz abc def ghi"  # No known patterns
        patterns = {"hr": ["je"], "en": ["the"]}

        result = detect_language_from_text(text, patterns, default_language="hr")

        assert result.detected_language == "hr"
        assert result.confidence == 0.0
        assert result.fallback_used is True

    def test_detect_language_from_text_auto_detect_disabled(self):
        """Test language detection with auto-detect disabled."""
        text = "This is English text"
        patterns = {"en": ["this", "is"]}

        result = detect_language_from_text(text, patterns, auto_detect=False, default_language="hr")

        assert result.detected_language == "hr"
        assert result.fallback_used is True

    def test_remove_stopwords_from_text_basic(self):
        """Test stopword removal."""
        text = "This is a test text"
        stopwords = {"is", "a"}

        result = remove_stopwords_from_text(text, stopwords)

        assert result == "This test text"

    def test_remove_stopwords_from_text_empty(self):
        """Test stopword removal with empty inputs."""
        assert remove_stopwords_from_text("", {"test"}) == ""
        assert remove_stopwords_from_text("test", set()) == "test"

    def test_remove_stopwords_from_text_case_insensitive(self):
        """Test stopword removal is case insensitive."""
        text = "This Is A Test"
        stopwords = {"is", "a"}  # lowercase

        result = remove_stopwords_from_text(text, stopwords)

        # "Is" and "A" should be removed (case insensitive matching)
        assert result == "This Test"
        assert "This" in result
        assert "Test" in result
        assert "Is" not in result  # Removed because "Is".lower() == "is"
        assert "A" not in result   # Removed because "A".lower() == "a"

    def test_calculate_collection_suffix_basic(self):
        """Test collection suffix calculation."""
        supported = ["hr", "en"]

        assert calculate_collection_suffix("hr", supported, "en") == "hr"
        assert calculate_collection_suffix("en", supported, "en") == "en"
        assert calculate_collection_suffix("fr", supported, "en") == "en"  # fallback

    def test_calculate_collection_suffix_special_cases(self):
        """Test collection suffix for special cases."""
        supported = ["hr", "en"]

        assert calculate_collection_suffix("auto", supported, "en") == "en"
        assert calculate_collection_suffix("multilingual", supported, "en") == "multi"

    def test_normalize_language_code_pure_basic(self):
        """Test pure language code normalization."""
        supported = ["hr", "en"]

        # Valid codes
        normalized, valid = normalize_language_code_pure("hr", supported, "hr")
        assert normalized == "hr"
        assert valid is True

        # Auto code
        normalized, valid = normalize_language_code_pure("auto", supported, "hr")
        assert normalized == "hr"
        assert valid is True

        # Invalid code
        normalized, valid = normalize_language_code_pure("fr", supported, "hr")
        assert normalized == "hr"
        assert valid is False

    def test_normalize_language_code_pure_variations(self):
        """Test language code normalization with variations."""
        supported = ["hr", "en"]

        # With region codes
        normalized, valid = normalize_language_code_pure("en-US", supported, "hr")
        assert normalized == "en"
        assert valid is True

        # With underscores
        normalized, valid = normalize_language_code_pure("en_GB", supported, "hr")
        assert normalized == "en"
        assert valid is True

    def test_validate_languages_list_basic(self):
        """Test languages list validation."""
        input_codes = ["hr", "en", "fr"]  # "fr" not supported
        supported = ["hr", "en"]

        result = validate_languages_list(input_codes, supported, "hr")

        assert "hr" in result
        assert "en" in result
        assert len(result) == 2  # "fr" normalized to default "hr", but "hr" already present

    def test_validate_languages_list_empty_input(self):
        """Test languages list validation with empty input."""
        result = validate_languages_list([], ["hr", "en"], "hr")

        assert result == ["hr"]  # Should add default

    def test_validate_languages_list_duplicates(self):
        """Test languages list validation removes duplicates."""
        input_codes = ["hr", "hr", "en"]
        supported = ["hr", "en"]

        result = validate_languages_list(input_codes, supported, "hr")

        assert len(result) == 2
        assert result.count("hr") == 1  # No duplicates

    def test_get_chunk_config_for_language_success(self):
        """Test getting chunk configuration for language."""
        languages = {
            "hr": LanguageConfig("hr", "Croatian", "Hrvatski", True, "model", 512, 50),
            "en": LanguageConfig("en", "English", "English", True, "model", 1024, 100)
        }

        chunk_size, overlap = get_chunk_config_for_language(languages, "hr")

        assert chunk_size == 512
        assert overlap == 50

    def test_get_chunk_config_for_language_not_supported(self):
        """Test getting chunk config for unsupported language."""
        languages = {"hr": LanguageConfig("hr", "Croatian", "Hrvatski", True, "model", 512, 50)}

        with pytest.raises(ValueError, match="Language en not supported"):
            get_chunk_config_for_language(languages, "en")

    def test_get_embedding_model_for_language_success(self):
        """Test getting embedding model for language."""
        languages = {
            "hr": LanguageConfig("hr", "Croatian", "Hrvatski", True, "classla/bcms-bertic", 512, 50)
        }

        model = get_embedding_model_for_language(languages, "hr")

        assert model == "classla/bcms-bertic"

    def test_get_embedding_model_for_language_not_supported(self):
        """Test getting embedding model for unsupported language."""
        languages = {"hr": LanguageConfig("hr", "Croatian", "Hrvatski", True, "model", 512, 50)}

        with pytest.raises(ValueError, match="Language en not supported"):
            get_embedding_model_for_language(languages, "en")

    def test_get_display_name_for_language_success(self):
        """Test getting display name for language."""
        languages = {
            "hr": LanguageConfig("hr", "Croatian", "Hrvatski", True, "model", 512, 50)
        }

        name = get_display_name_for_language(languages, "hr")

        assert name == "Hrvatski"

    def test_get_display_name_for_language_not_supported(self):
        """Test getting display name for unsupported language."""
        languages = {"hr": LanguageConfig("hr", "Croatian", "Hrvatski", True, "model", 512, 50)}

        with pytest.raises(ValueError, match="Language en not supported"):
            get_display_name_for_language(languages, "en")


class TestLanguageManager:
    """Test _LanguageManager class."""

    def create_test_providers(self):
        """Create test providers for testing."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_language_settings.return_value = LanguageSettings(
            supported_languages=["hr", "en"],
            default_language="hr",
            auto_detect=True,
            fallback_language="en",
            language_names={"hr": "Croatian", "en": "English"},
            embedding_model="BAAI/bge-m3",
            chunk_size=512,
            chunk_overlap=50
        )

        pattern_provider = Mock(spec=PatternProvider)
        pattern_provider.get_language_patterns.return_value = LanguagePatterns(
            detection_patterns={
                "hr": ["je", "da", "i", "u"],
                "en": ["the", "and", "of", "to"]
            },
            stopwords={
                "hr": {"je", "da", "i"},
                "en": {"the", "and", "of"}
            }
        )

        logger_provider = Mock(spec=LoggerProvider)

        return config_provider, pattern_provider, logger_provider

    def test_language_manager_initialization(self):
        """Test language manager initialization."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()

        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        assert len(manager.get_supported_languages()) == 2
        assert "hr" in manager.get_supported_languages()
        assert "en" in manager.get_supported_languages()
        config_provider.get_language_settings.assert_called_once()
        pattern_provider.get_language_patterns.assert_called_once()

    def test_language_manager_without_logger(self):
        """Test language manager initialization without logger."""
        config_provider, pattern_provider, _ = self.create_test_providers()

        manager = _LanguageManager(config_provider, pattern_provider, None)

        assert manager._logger is None
        assert len(manager.get_supported_languages()) == 2

    def test_get_language_config(self):
        """Test getting language configuration."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        config = manager.get_language_config("hr")

        assert config.code == "hr"
        assert config.name == "Croatian"
        assert config.embedding_model == "BAAI/bge-m3"

    def test_get_language_config_not_supported(self):
        """Test getting configuration for unsupported language."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        with pytest.raises(ValueError, match="Language fr not supported"):
            manager.get_language_config("fr")

    def test_is_language_supported(self):
        """Test language support checking."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        assert manager.is_language_supported("hr") is True
        assert manager.is_language_supported("en") is True
        assert manager.is_language_supported("fr") is False

    def test_get_default_language(self):
        """Test getting default language."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        assert manager.get_default_language() == "hr"

    def test_get_fallback_language(self):
        """Test getting fallback language."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        assert manager.get_fallback_language() == "en"

    def test_detect_language(self):
        """Test language detection."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        # Croatian text
        result = manager.detect_language("Ovo je hrvatski tekst")

        assert result == "hr"

    def test_detect_language_detailed(self):
        """Test detailed language detection."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        result = manager.detect_language_detailed("Ovo je hrvatski tekst")

        assert isinstance(result, DetectionResult)
        assert result.detected_language == "hr"

    def test_get_stopwords(self):
        """Test getting stopwords for language."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        stopwords = manager.get_stopwords("hr")

        assert isinstance(stopwords, set)
        assert "je" in stopwords

    def test_get_stopwords_not_available(self):
        """Test getting stopwords for language without stopwords."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        with pytest.raises(ValueError, match="Stopwords not available for language fr"):
            manager.get_stopwords("fr")

    def test_remove_stopwords(self):
        """Test removing stopwords from text."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        result = manager.remove_stopwords("je da test", "hr")

        assert result == "test"  # "je" and "da" should be removed

    def test_get_collection_suffix(self):
        """Test getting collection suffix."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        assert manager.get_collection_suffix("hr") == "hr"
        assert manager.get_collection_suffix("auto") == "en"  # fallback
        assert manager.get_collection_suffix("fr") == "en"  # unsupported -> fallback

    def test_normalize_language_code(self):
        """Test language code normalization."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        assert manager.normalize_language_code("hr") == "hr"
        assert manager.normalize_language_code("en-US") == "en"
        assert manager.normalize_language_code("fr") == "hr"  # unsupported -> default

    def test_get_language_display_name(self):
        """Test getting language display name."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        assert manager.get_language_display_name("hr") == "Croatian"

    def test_validate_languages(self):
        """Test validating languages list."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        result = manager.validate_languages(["hr", "en", "fr"])

        assert "hr" in result
        assert "en" in result
        assert len(result) == 2  # "fr" normalized to default, but no duplicates

    def test_get_embedding_model(self):
        """Test getting embedding model for language."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        model = manager.get_embedding_model("hr")

        assert model == "BAAI/bge-m3"

    def test_get_chunk_config(self):
        """Test getting chunk configuration for language."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        chunk_size, overlap = manager.get_chunk_config("hr")

        assert chunk_size == 512
        assert overlap == 50

    def test_add_language_runtime(self):
        """Test adding language support at runtime."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        # Before adding
        assert not manager.is_language_supported("de")

        # Add German support
        manager.add_language_runtime("de", "German", "Deutsch", ["der", "die", "das"], ["der", "die"])

        # After adding
        assert manager.is_language_supported("de")
        assert manager.get_language_display_name("de") == "Deutsch"
        stopwords = manager.get_stopwords("de")
        assert "der" in stopwords

    def test_logging_methods(self):
        """Test logging methods work correctly."""
        config_provider, pattern_provider, logger_provider = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, logger_provider)

        # These should not raise errors
        manager._log_info("test info")
        manager._log_debug("test debug")
        manager._log_warning("test warning")
        manager._log_error("test error")

        # Verify logger was called
        assert logger_provider.info.call_count >= 1  # At least initialization log
        logger_provider.debug.assert_called_with("test debug")
        logger_provider.warning.assert_called_with("test warning")
        logger_provider.error.assert_called_with("test error")

    def test_logging_methods_without_logger(self):
        """Test logging methods work without logger."""
        config_provider, pattern_provider, _ = self.create_test_providers()
        manager = _LanguageManager(config_provider, pattern_provider, None)

        # These should not raise errors even without logger
        manager._log_info("test info")
        manager._log_debug("test debug")
        manager._log_warning("test warning")
        manager._log_error("test error")


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_language_manager(self):
        """Test language manager factory function."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_language_settings.return_value = LanguageSettings(
            supported_languages=["hr"],
            default_language="hr",
            auto_detect=True,
            fallback_language="hr",
            language_names={"hr": "Croatian"},
            embedding_model="test",
            chunk_size=512,
            chunk_overlap=50
        )

        pattern_provider = Mock(spec=PatternProvider)
        pattern_provider.get_language_patterns.return_value = LanguagePatterns(
            detection_patterns={"hr": ["je"]},
            stopwords={"hr": {"je"}}
        )

        logger_provider = Mock(spec=LoggerProvider)

        manager = create_language_manager(config_provider, pattern_provider, logger_provider)

        assert isinstance(manager, _LanguageManager)
        assert manager.is_language_supported("hr")

    def test_language_manager_public_interface_with_providers(self):
        """Test public LanguageManager interface with explicit providers."""
        config_provider = Mock(spec=ConfigProvider)
        config_provider.get_language_settings.return_value = LanguageSettings(
            supported_languages=["hr"],
            default_language="hr",
            auto_detect=True,
            fallback_language="hr",
            language_names={"hr": "Croatian"},
            embedding_model="test",
            chunk_size=512,
            chunk_overlap=50
        )

        pattern_provider = Mock(spec=PatternProvider)
        pattern_provider.get_language_patterns.return_value = LanguagePatterns(
            detection_patterns={"hr": ["je"]},
            stopwords={"hr": {"je"}}
        )

        logger_provider = Mock(spec=LoggerProvider)

        manager = LanguageManager(config_provider, pattern_provider, logger_provider)

        assert isinstance(manager, _LanguageManager)
        assert manager.is_language_supported("hr")


if __name__ == "__main__":
    pytest.main([__file__])