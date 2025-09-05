#!/usr/bin/env python3
"""
Test suite for multilingual configuration loader system.
Tests Croatian and English language configurations using the new multilingual APIs.
"""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import (
    get_language_config,
    get_language_shared,
    get_language_specific_config,
    get_supported_languages,
    is_language_supported,
    load_config,
)


class TestMultilingualConfig:
    """Test multilingual configuration loading."""

    def test_supported_languages(self):
        """Test that both Croatian and English are supported."""
        supported = get_supported_languages()
        assert "hr" in supported, "Croatian should be supported"
        assert "en" in supported, "English should be supported"

        assert is_language_supported("hr"), "Croatian should be supported"
        assert is_language_supported("en"), "English should be supported"
        assert not is_language_supported("fr"), "French should not be supported"

    def test_croatian_language_config(self):
        """Test Croatian language configuration loading."""
        hr_config = get_language_config("hr")

        # Verify basic structure
        assert "language" in hr_config
        assert "shared" in hr_config
        assert "prompts" in hr_config

        # Verify Croatian-specific values
        assert hr_config["language"]["code"] == "hr"
        assert hr_config["language"]["name"] == "Croatian"

        # Test shared config
        hr_shared = get_language_shared("hr")
        assert "stopwords" in hr_shared
        assert "words" in hr_shared["stopwords"]

        # Verify Croatian stopwords contain expected words
        stopwords = hr_shared["stopwords"]["words"]
        assert "i" in stopwords
        assert "u" in stopwords
        assert "na" in stopwords

    def test_english_language_config(self):
        """Test English language configuration loading."""
        en_config = get_language_config("en")

        # Verify basic structure
        assert "language" in en_config
        assert "shared" in en_config
        assert "prompts" in en_config

        # Verify English-specific values
        assert en_config["language"]["code"] == "en"
        assert en_config["language"]["name"] == "English"

        # Test shared config
        en_shared = get_language_shared("en")
        assert "stopwords" in en_shared
        assert "words" in en_shared["stopwords"]

        # Verify English stopwords contain expected words
        stopwords = en_shared["stopwords"]["words"]
        assert "the" in stopwords
        assert "and" in stopwords
        assert "that" in stopwords

    def test_language_specific_sections(self):
        """Test language-specific section retrieval."""
        # Test Croatian prompts
        hr_prompts = get_language_specific_config("prompts", "hr")
        assert "system_base" in hr_prompts
        assert "question_answering_system" in hr_prompts

        # Test English prompts
        en_prompts = get_language_specific_config("prompts", "en")
        assert "system_base" in en_prompts
        assert "question_answering_system" in en_prompts

        # Verify they're different
        assert hr_prompts["system_base"] != en_prompts["system_base"]

    def test_text_processing_config(self):
        """Test text processing configuration."""
        hr_text = get_language_specific_config("text_processing", "hr")
        en_text = get_language_specific_config("text_processing", "en")

        # Both should have text processing settings
        assert hr_text is not None
        assert en_text is not None

        # Croatian should preserve diacritics
        assert hr_text["remove_diacritics"] is False  # Croatian keeps diacritics

        # English also preserves diacritics (for foreign words)
        assert en_text["remove_diacritics"] is False  # English also keeps diacritics

        # Both should normalize case
        assert hr_text["normalize_case"] is True
        assert en_text["normalize_case"] is True

    def test_retrieval_config(self):
        """Test retrieval configuration for both languages."""
        # Croatian retrieval
        hr_retrieval = get_language_specific_config("retrieval", "hr")
        assert "synonyms" in hr_retrieval
        assert "morphology" in hr_retrieval

        # English retrieval
        en_retrieval = get_language_specific_config("retrieval", "en")
        assert "synonyms" in en_retrieval
        assert "morphology" in en_retrieval

    def test_vectordb_config(self):
        """Test vectordb configuration for both languages."""
        # Croatian vectordb
        hr_vectordb = get_language_specific_config("vectordb", "hr")
        assert "embeddings" in hr_vectordb
        assert "collection_name" in hr_vectordb

        # English vectordb
        en_vectordb = get_language_specific_config("vectordb", "en")
        assert "embeddings" in en_vectordb
        assert "collection_name" in en_vectordb

    def test_pipeline_config(self):
        """Test pipeline configuration for both languages."""
        # Croatian pipeline
        hr_pipeline = get_language_specific_config("pipeline", "hr")
        assert "processing" in hr_pipeline

        # English pipeline
        en_pipeline = get_language_specific_config("pipeline", "en")
        assert "processing" in en_pipeline

    def test_unsupported_language(self):
        """Test error handling for unsupported languages."""
        with pytest.raises(Exception):  # Should raise ConfigError
            get_language_config("fr")  # French not supported

    def test_main_config_loading(self):
        """Test main configuration files still work."""
        # Test main config
        main_config = load_config("config")
        assert "languages" in main_config
        assert "system" in main_config

        # Test generation config
        generation_config = load_config("generation")
        assert "ollama" in generation_config
        assert "prompts" in generation_config


if __name__ == "__main__":
    # Run tests manually
    test_config = TestMultilingualConfig()

    print("🧪 Testing Multilingual Configuration")
    print("=" * 50)

    try:
        test_config.test_supported_languages()
        print("✅ Supported languages test passed")

        test_config.test_croatian_language_config()
        print("✅ Croatian config test passed")

        test_config.test_english_language_config()
        print("✅ English config test passed")

        test_config.test_language_specific_sections()
        print("✅ Language-specific sections test passed")

        test_config.test_text_processing_config()
        print("✅ Text processing config test passed")

        test_config.test_retrieval_config()
        print("✅ Retrieval config test passed")

        test_config.test_vectordb_config()
        print("✅ Vectordb config test passed")

        test_config.test_pipeline_config()
        print("✅ Pipeline config test passed")

        test_config.test_main_config_loading()
        print("✅ Main config loading test passed")

        print("\n🎉 All multilingual configuration tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
