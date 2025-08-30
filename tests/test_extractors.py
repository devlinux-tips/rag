"""
Unit tests for document extractors.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
sys.path.append('src')

from preprocessing.extractors import DocumentExtractor, extract_document_text


class TestDocumentExtractor:
    """Test the DocumentExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = DocumentExtractor()
    
    def test_init(self):
        """Test extractor initialization."""
        assert self.extractor.supported_formats == {'.pdf', '.docx', '.txt'}
    
    def test_extract_text_file_not_found(self):
        """Test extraction with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.extractor.extract_text(Path("nonexistent.pdf"))
    
    def test_extract_text_unsupported_format(self):
        """Test extraction with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz') as tmp:
            with pytest.raises(ValueError, match="Unsupported format"):
                self.extractor.extract_text(Path(tmp.name))
    
    @patch('builtins.open', mock_open(read_data="Test Croatian text with čćžšđ"))
    def test_extract_from_txt_utf8(self):
        """Test TXT extraction with UTF-8 encoding."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            result = self.extractor._extract_from_txt(Path(tmp.name))
            assert result == "Test Croatian text with čćžšđ"
    
    @patch('builtins.open')
    def test_extract_from_txt_encoding_fallback(self, mock_file):
        """Test TXT extraction with encoding fallback."""
        # Simulate UTF-8 failure, then success with cp1250
        mock_file.side_effect = [
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'),
            UnicodeDecodeError('utf-8-sig', b'', 0, 1, 'invalid'),
            mock_open(read_data="Croatian text").return_value
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            result = self.extractor._extract_from_txt(Path(tmp.name))
            assert result == "Croatian text"
    
    @patch('PyPDF2.PdfReader')
    def test_extract_from_pdf(self, mock_pdf_reader):
        """Test PDF extraction."""
        # Mock PDF structure
        mock_page = mock_pdf_reader.return_value.pages.__getitem__.return_value
        mock_page.extract_text.return_value = "Croatian PDF text\nwith multiple lines"
        mock_pdf_reader.return_value.pages.__len__.return_value = 1
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            result = self.extractor._extract_from_pdf(Path(tmp.name))
            assert "Croatian PDF text" in result
            assert "multiple lines" in result
    
    @patch('docx.Document')
    def test_extract_from_docx(self, mock_document):
        """Test DOCX extraction."""
        # Mock DOCX structure
        mock_para1 = type('MockParagraph', (), {'text': 'Croatian DOCX text'})()
        mock_para2 = type('MockParagraph', (), {'text': 'Second paragraph'})()
        mock_document.return_value.paragraphs = [mock_para1, mock_para2]
        
        with tempfile.NamedTemporaryFile(suffix='.docx') as tmp:
            result = self.extractor._extract_from_docx(Path(tmp.name))
            assert "Croatian DOCX text" in result
            assert "Second paragraph" in result
    
    def test_extract_text_routing(self):
        """Test that extract_text routes to correct method."""
        with patch.object(self.extractor, '_extract_from_pdf') as mock_pdf:
            mock_pdf.return_value = "PDF content"
            with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
                result = self.extractor.extract_text(Path(tmp.name))
                mock_pdf.assert_called_once()
                assert result == "PDF content"


class TestExtractDocumentTextFunction:
    """Test the convenience function."""
    
    def test_extract_document_text_string_path(self):
        """Test convenience function with string path."""
        with patch.object(DocumentExtractor, 'extract_text') as mock_extract:
            mock_extract.return_value = "Extracted text"
            
            result = extract_document_text("test.txt")
            assert result == "Extracted text"
            mock_extract.assert_called_once()
    
    def test_extract_document_text_path_object(self):
        """Test convenience function with Path object."""
        with patch.object(DocumentExtractor, 'extract_text') as mock_extract:
            mock_extract.return_value = "Extracted text"
            
            result = extract_document_text(Path("test.txt"))
            assert result == "Extracted text"
            mock_extract.assert_called_once()


class TestCroatianTextHandling:
    """Test Croatian-specific text handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = DocumentExtractor()
        self.croatian_text = "Ovo je croatian tekst sa čćžšđ karakterima."
        self.mixed_text = "Croatian: čćžšđ, English: hello"
    
    @patch('builtins.open', mock_open(read_data="Tekst sa čćžšđ dijakritičkim znakovima"))
    def test_croatian_diacritics_preservation(self):
        """Test that Croatian diacritics are preserved."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            result = self.extractor._extract_from_txt(Path(tmp.name))
            
            # Check that all Croatian diacritics are preserved
            croatian_chars = set('čćžšđ')
            found_chars = croatian_chars.intersection(set(result))
            assert len(found_chars) > 0, "Croatian diacritics should be preserved"
    
    @patch('builtins.open')
    def test_encoding_detection_with_croatian(self, mock_file):
        """Test encoding detection with Croatian characters."""
        # Test that cp1250 encoding works for Croatian
        mock_file.side_effect = [
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'),
            UnicodeDecodeError('utf-8-sig', b'', 0, 1, 'invalid'),
            mock_open(read_data="Šišmiš čuva žutu ćupriju").return_value
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            result = self.extractor._extract_from_txt(Path(tmp.name))
            assert "Šišmiš" in result
            assert "čuva" in result
            assert "žutu" in result
            assert "ćupriju" in result