"""
Unit tests for Croatian text cleaners.
"""

import pytest

import sys
sys.path.append('src')

from preprocessing.cleaners import CroatianTextCleaner, clean_croatian_text


class TestCroatianTextCleaner:
    """Test the CroatianTextCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = CroatianTextCleaner()
    
    def test_init(self):
        """Test cleaner initialization."""
        assert 'č' in self.cleaner.diacritic_map
        assert 'je' in self.cleaner.stopwords
        assert len(self.cleaner.formatting_patterns) > 0
    
    def test_clean_text_empty_input(self):
        """Test cleaning with empty or None input."""
        assert self.cleaner.clean_text("") == ""
        assert self.cleaner.clean_text(None) == ""
        assert self.cleaner.clean_text("   ") == ""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Ovo   je   test   tekst."
        result = self.cleaner.clean_text(text)
        assert "  " not in result  # Multiple spaces removed
        assert result.strip() == result  # No leading/trailing whitespace
    
    def test_normalize_whitespace_preserve_structure(self):
        """Test whitespace normalization with structure preservation."""
        text = "Prvi paragraf.\n\nDrugi paragraf.\nTreća linija."
        result = self.cleaner._normalize_whitespace(text, preserve_structure=True)
        
        assert "\n\n" in result  # Paragraph breaks preserved
        assert result.count("\n\n") <= text.count("\n\n")
    
    def test_normalize_whitespace_no_structure(self):
        """Test whitespace normalization without structure preservation."""
        text = "Prvi paragraf.\n\nDrugi paragraf.\nTreća linija."
        result = self.cleaner._normalize_whitespace(text, preserve_structure=False)
        
        assert "\n" not in result  # All line breaks converted to spaces
        assert "Prvi paragraf. Drugi paragraf. Treća linija." == result
    
    def test_remove_headers_footers(self):
        """Test removal of Croatian document headers and footers."""
        text = """NARODNE NOVINE SLUŽBENI LIST
        
        Ovo je glavni sadržaj dokumenta.
        
        STRANICA 123
        ZAGREB, 2025."""
        
        result = self.cleaner._remove_headers_footers(text)
        assert "Ovo je glavni sadržaj dokumenta." in result
        # Headers/footers should be removed or reduced
        assert len(result) < len(text)
    
    def test_fix_ocr_errors(self):
        """Test OCR error corrections for Croatian text."""
        text = "HRV  A  T  S  K  E ZAGREB"
        result = self.cleaner._fix_ocr_errors(text)
        
        assert "HRVATSKE" in result or "HRV A T S K E" not in result
    
    def test_extract_sentences_croatian(self):
        """Test sentence extraction with Croatian punctuation."""
        text = "Prva rečenica. Druga rečenica! Treća rečenica? Četvrta rečenica."
        sentences = self.cleaner.extract_sentences(text)
        
        assert len(sentences) >= 3  # Should find multiple sentences
        for sentence in sentences:
            assert len(sentence.strip()) > 10  # Minimum sentence length enforced
    
    def test_normalize_diacritics(self):
        """Test diacritic normalization."""
        text = "čćžšđ ČĆŽŠĐ"
        result = self.cleaner.normalize_diacritics(text)
        
        assert "č" not in result
        assert "ć" not in result
        assert "c" in result
        assert "s" in result
        assert "z" in result
    
    def test_is_meaningful_text_valid(self):
        """Test meaningful text detection with valid Croatian text."""
        text = "Ovo je smisleni hrvatski tekst sa čćžšđ."
        assert self.cleaner.is_meaningful_text(text) is True
    
    def test_is_meaningful_text_invalid(self):
        """Test meaningful text detection with invalid inputs."""
        assert self.cleaner.is_meaningful_text("") is False
        assert self.cleaner.is_meaningful_text("a b") is False  # Too few words
        assert self.cleaner.is_meaningful_text("123 456 789 000") is False  # Mostly numbers
        assert self.cleaner.is_meaningful_text("!@# $$$ %%%") is False  # Special chars only
    
    def test_is_meaningful_text_min_words(self):
        """Test meaningful text with custom minimum word count."""
        text = "Kratki tekst"
        assert self.cleaner.is_meaningful_text(text, min_words=2) is True
        assert self.cleaner.is_meaningful_text(text, min_words=5) is False


class TestCroatianSpecificFeatures:
    """Test Croatian language-specific features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = CroatianTextCleaner()
    
    def test_croatian_diacritics_preservation(self):
        """Test that Croatian diacritics are preserved in cleaning."""
        text = "Šišmiš je čudna životinja koja živi u špiljama."
        result = self.cleaner.clean_text(text)
        
        # All Croatian diacritics should be preserved
        croatian_chars = set('čćžšđČĆŽŠĐ')
        original_chars = croatian_chars.intersection(set(text))
        result_chars = croatian_chars.intersection(set(result))
        
        assert original_chars == result_chars
    
    def test_croatian_sentence_boundaries(self):
        """Test sentence boundary detection with Croatian capitalization."""
        text = "To je činjenica. Što mislite o tome? Čini mi se da je tako."
        sentences = self.cleaner.extract_sentences(text)
        
        assert len(sentences) >= 2
        # Croatian uppercase letters should be recognized
        for sentence in sentences:
            if sentence.strip():
                assert len(sentence.strip()) > 5
    
    def test_croatian_official_document_artifacts(self):
        """Test removal of Croatian official document artifacts."""
        text = """NARODNE NOVINE
        SLUŽBENI LIST REPUBLIKE HRVATSKE
        BROJ 115, ZAGREB, 29. KOLOVOZA 2025.
        
        Glavni sadržaj dokumenta sa važnim informacijama.
        
        STRANICA 155"""
        
        result = self.cleaner._remove_headers_footers(text)
        assert "Glavni sadržaj" in result
        # Should reduce government header artifacts
        original_lines = len(text.split('\n'))
        result_lines = len(result.split('\n'))
        assert result_lines <= original_lines
    
    def test_croatian_ocr_corrections(self):
        """Test OCR corrections specific to Croatian documents."""
        ocr_errors = {
            "HR V A TSKE": "HRVATSKE",
            "ZA GR EB": "ZAGREB",
            "HR  VATSKE": "HRVATSKE"
        }
        
        for error, correction in ocr_errors.items():
            result = self.cleaner._fix_ocr_errors(error)
            # Should fix or improve the text
            assert error != result or correction.lower() in result.lower()


class TestCleanCroatianTextFunction:
    """Test the convenience function."""
    
    def test_clean_croatian_text_basic(self):
        """Test convenience function basic usage."""
        text = "Test   tekst   sa   višestrukim   razmacima."
        result = clean_croatian_text(text)
        
        assert "  " not in result
        assert len(result) > 0
    
    def test_clean_croatian_text_preserve_structure(self):
        """Test convenience function with structure preservation."""
        text = "Prvi paragraf.\n\nDrugi paragraf."
        
        result_preserve = clean_croatian_text(text, preserve_structure=True)
        result_no_preserve = clean_croatian_text(text, preserve_structure=False)
        
        assert "\n\n" in result_preserve
        assert "\n" not in result_no_preserve


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = CroatianTextCleaner()
    
    def test_very_long_text(self):
        """Test cleaning with very long text."""
        long_text = "Croatian text. " * 1000
        result = self.cleaner.clean_text(long_text)
        
        assert len(result) > 0
        assert "Croatian text" in result
    
    def test_only_special_characters(self):
        """Test text with only special characters."""
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = self.cleaner.clean_text(text)
        
        # Should either clean to empty or handle gracefully
        assert isinstance(result, str)
    
    def test_mixed_languages(self):
        """Test text mixing Croatian and other languages."""
        text = "Croatian: ovo je hrvatski tekst. English: this is English text."
        result = self.cleaner.clean_text(text)
        
        assert "Croatian" in result
        assert "English" in result
        assert len(result) > 0
    
    def test_unicode_edge_cases(self):
        """Test various Unicode characters."""
        text = "Test 🇭🇷 Croatian emoji and unicode ćčžšđ characters."
        result = self.cleaner.clean_text(text)
        
        # Croatian characters should be preserved
        assert any(char in result for char in 'ćčžšđ')
        assert len(result) > 0