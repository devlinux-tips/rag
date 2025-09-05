#!/usr/bin/env python3
"""
Test suite for multilingual text cleaning functionality.
Tests both Croatian and English text processing using MultilingualTextCleaner.
"""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.cleaners import (
    MultilingualTextCleaner,
    clean_text,
    detect_language_content,
    preserve_text_encoding,
    setup_language_environment,
)


class TestMultilingualTextCleaner:
    """Test multilingual text cleaning functionality."""

    def test_croatian_text_cleaner(self):
        """Test Croatian text cleaning."""
        cleaner = MultilingualTextCleaner(language="hr")

        # Test Croatian text with diacritics
        croatian_text = (
            "Ovo je  test   tekst sa  hrvat­skim   znakovima: čćšžđ.    \n\n\nDrugi paragraf."
        )
        cleaned = cleaner.clean_text(croatian_text)

        # Should preserve Croatian characters
        assert "č" in cleaned or "ć" in cleaned or "š" in cleaned
        # Should normalize whitespace
        assert "  " not in cleaned  # No double spaces
        # Should preserve structure
        assert len(cleaned) > 0

    def test_english_text_cleaner(self):
        """Test English text cleaning."""
        cleaner = MultilingualTextCleaner(language="en")

        # Test English text
        english_text = (
            "This is  a  test   text    with multiple    spaces.   \n\n\nSecond paragraph."
        )
        cleaned = cleaner.clean_text(english_text)

        # Should normalize whitespace
        assert "  " not in cleaned  # No double spaces
        # Should preserve basic structure
        assert len(cleaned) > 0
        assert "test" in cleaned

    def test_language_detection(self):
        """Test language content detection."""
        # Croatian text detection
        croatian_text = "Zagreb je glavni grad Hrvatske. Dubrovnik je biser Jadrana."
        hr_score = detect_language_content(croatian_text, language="hr")
        assert hr_score > 0.1, "Should detect Croatian content"

        # English text detection
        english_text = "The quick brown fox jumps over the lazy dog."
        en_score = detect_language_content(english_text, language="en")
        assert en_score > 0.1, "Should detect English content"

    def test_convenience_functions(self):
        """Test convenience functions for text processing."""
        # Test clean_text function
        test_text = "This   is   a   test   with   multiple   spaces."

        # Croatian cleaning
        hr_cleaned = clean_text(test_text, language="hr")
        assert len(hr_cleaned) > 0
        assert "  " not in hr_cleaned

        # English cleaning
        en_cleaned = clean_text(test_text, language="en")
        assert len(en_cleaned) > 0
        assert "  " not in en_cleaned

    def test_diacritic_handling(self):
        """Test diacritic handling for different languages."""
        # Croatian should preserve diacritics
        hr_cleaner = MultilingualTextCleaner(language="hr")
        croatian_text = "Čovijek čita časopis."
        hr_cleaned = hr_cleaner.clean_text(croatian_text)
        assert "č" in hr_cleaned.lower()  # Should preserve Croatian diacritics

        # English cleaner (no diacritics expected)
        en_cleaner = MultilingualTextCleaner(language="en")
        english_text = "The man reads a magazine."
        en_cleaned = en_cleaner.clean_text(english_text)
        assert "man" in en_cleaned.lower()

    def test_sentence_extraction(self):
        """Test sentence extraction for both languages."""
        # Croatian sentences
        hr_cleaner = MultilingualTextCleaner(language="hr")
        croatian_text = "Ovo je prva rečenica. Ovo je druga rečenica! Ovo je treća rečenica?"
        hr_sentences = hr_cleaner.extract_sentences(croatian_text)
        assert len(hr_sentences) >= 2, "Should extract multiple sentences"

        # English sentences
        en_cleaner = MultilingualTextCleaner(language="en")
        english_text = (
            "This is the first sentence. This is the second sentence! This is the third sentence?"
        )
        en_sentences = en_cleaner.extract_sentences(english_text)
        assert len(en_sentences) >= 2, "Should extract multiple sentences"

    def test_meaningful_text_detection(self):
        """Test meaningful text detection for both languages."""
        # Croatian meaningful text
        hr_cleaner = MultilingualTextCleaner(language="hr")
        assert hr_cleaner.is_meaningful_text("Ovo je smisleni tekst o važnoj temi.")
        assert not hr_cleaner.is_meaningful_text("123 !@# $$$")
        assert not hr_cleaner.is_meaningful_text("")

        # English meaningful text
        en_cleaner = MultilingualTextCleaner(language="en")
        assert en_cleaner.is_meaningful_text("This is meaningful text about an important topic.")
        assert not en_cleaner.is_meaningful_text("123 !@# $$$")
        assert not en_cleaner.is_meaningful_text("")

    def test_encoding_preservation(self):
        """Test text encoding preservation."""
        # Test with various input types
        test_text = "Test text with special characters: čćšžđ"

        # Should handle string input
        preserved = preserve_text_encoding(test_text, language="hr")
        assert isinstance(preserved, str)
        assert len(preserved) > 0

    def test_environment_setup(self):
        """Test language environment setup."""
        # Should not raise errors
        setup_language_environment(language="hr")
        setup_language_environment(language="en")

    def test_different_text_types(self):
        """Test cleaning different types of text content."""
        # Test document-like text
        document_text = """
        Naslov Dokumenta

        Ovo je prvi paragraf dokumenta sa važnim informacijama.
        Drugi paragraf sadrži dodatne detalje.

        • Prvi element liste
        • Drugi element liste

        Zaključak dokumenta.
        """

        hr_cleaner = MultilingualTextCleaner(language="hr")
        cleaned = hr_cleaner.clean_text(document_text)

        assert len(cleaned) > 0
        assert "Naslov" in cleaned
        assert "paragraf" in cleaned

    def test_error_handling(self):
        """Test error handling for edge cases."""
        hr_cleaner = MultilingualTextCleaner(language="hr")

        # Empty text
        assert hr_cleaner.clean_text("") == ""
        assert hr_cleaner.clean_text(None) == ""

        # Whitespace only
        whitespace_cleaned = hr_cleaner.clean_text("   \n\n   \t\t   ")
        assert whitespace_cleaned == ""


if __name__ == "__main__":
    # Run tests manually
    test_cleaner = TestMultilingualTextCleaner()

    print("🧪 Testing Multilingual Text Cleaner")
    print("=" * 50)

    try:
        test_cleaner.test_croatian_text_cleaner()
        print("✅ Croatian text cleaner test passed")

        test_cleaner.test_english_text_cleaner()
        print("✅ English text cleaner test passed")

        test_cleaner.test_language_detection()
        print("✅ Language detection test passed")

        test_cleaner.test_convenience_functions()
        print("✅ Convenience functions test passed")

        test_cleaner.test_diacritic_handling()
        print("✅ Diacritic handling test passed")

        test_cleaner.test_sentence_extraction()
        print("✅ Sentence extraction test passed")

        test_cleaner.test_meaningful_text_detection()
        print("✅ Meaningful text detection test passed")

        test_cleaner.test_encoding_preservation()
        print("✅ Encoding preservation test passed")

        test_cleaner.test_environment_setup()
        print("✅ Environment setup test passed")

        test_cleaner.test_different_text_types()
        print("✅ Different text types test passed")

        test_cleaner.test_error_handling()
        print("✅ Error handling test passed")

        print("\n🎉 All multilingual text cleaner tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
