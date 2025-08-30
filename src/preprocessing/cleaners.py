"""
Croatian text cleaning and normalization utilities.
Handles Croatian-specific text processing challenges including diacritics,
morphology, and document formatting artifacts.
"""

import re
import logging
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class CroatianTextCleaner:
    """Clean and normalize Croatian text for RAG processing."""
    
    def __init__(self):
        """Initialize the Croatian text cleaner."""
        # Croatian diacritic mappings for normalization (optional)
        self.diacritic_map = {
            'č': 'c', 'ć': 'c', 'đ': 'd', 'š': 's', 'ž': 'z',
            'Č': 'C', 'Ć': 'C', 'Đ': 'D', 'Š': 'S', 'Ž': 'Z'
        }
        
        # Common Croatian stopwords (basic set)
        self.stopwords = {
            'a', 'ali', 'am', 'as', 'at', 'ba', 'be', 'bi', 'bo', 'bu',
            'by', 'da', 'do', 'eh', 'el', 'en', 'es', 'et', 'ga', 'go',
            'ha', 'hi', 'ho', 'i', 'ic', 'id', 'ie', 'if', 'il', 'in',
            'ir', 'is', 'it', 'ja', 'je', 'ka', 'ko', 'la', 'li', 'lo',
            'ma', 'me', 'mi', 'mo', 'mu', 'my', 'na', 'ne', 'ni', 'no',
            'nu', 'od', 'oh', 'ok', 'ol', 'om', 'on', 'oo', 'op', 'or',
            'os', 'ot', 'ov', 'pa', 'po', 'ra', 're', 'sa', 'se', 'si',
            'so', 'ta', 'te', 'ti', 'to', 'tu', 'u', 'up', 'uz', 've',
            'za', 'ze'
        }
        
        # Document formatting artifacts to remove
        self.formatting_patterns = [
            r'\s+',  # Multiple whitespaces
            r'\n\s*\n',  # Multiple line breaks
            r'[^\w\sčćžšđČĆŽŠĐ.,!?:;()-]',  # Non-standard chars (preserve Croatian)
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'^\s*[IVX]+\s*$',  # Roman numerals
            r'^\s*[a-z]\)\s*$',  # List markers like a), b)
        ]
    
    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        Clean and normalize Croatian text.
        
        Args:
            text: Raw text to clean
            preserve_structure: Whether to preserve paragraph structure
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        logger.debug(f"Cleaning text of length {len(text)}")
        
        # Start with the original text
        cleaned = text
        
        # Remove document header/footer artifacts
        cleaned = self._remove_headers_footers(cleaned)
        
        # Normalize whitespace and line breaks
        cleaned = self._normalize_whitespace(cleaned, preserve_structure)
        
        # Remove formatting artifacts
        cleaned = self._remove_formatting_artifacts(cleaned)
        
        # Fix common OCR errors in Croatian text
        cleaned = self._fix_ocr_errors(cleaned)
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        logger.debug(f"Cleaned text length: {len(cleaned)}")
        return cleaned
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common document headers and footers."""
        # Remove page headers with page numbers and dates
        patterns = [
            r'^\s*\d+\s*$',  # Standalone page numbers
            r'^\s*STRANICA\s*\d+.*$',  # Croatian "PAGE X"
            r'^\s*BROJ\s*\d+.*ZAGREB.*$',  # Document headers
            r'^\s*NARODNE\s*NOVINE.*$',  # Croatian Official Gazette header
            r'^\s*SLUŽBENI\s*LIST.*$',  # Official list header
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            is_artifact = False
            for pattern in patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_artifact = True
                    break
            
            if not is_artifact:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_whitespace(self, text: str, preserve_structure: bool) -> str:
        """Normalize whitespace while optionally preserving paragraph structure."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        if preserve_structure:
            # Preserve paragraph breaks but normalize other line breaks
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 line breaks
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single line breaks to space
        else:
            # Convert all line breaks to spaces
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _remove_formatting_artifacts(self, text: str) -> str:
        """Remove document formatting artifacts."""
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'[-]{3,}', '---', text)  # Multiple dashes
        
        # Remove isolated formatting characters
        text = re.sub(r'\s+[_*-]\s+', ' ', text)
        
        # Remove standalone special characters that are formatting artifacts
        text = re.sub(r'^\s*[_*-]+\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in Croatian text."""
        # Common OCR mistakes in Croatian
        ocr_fixes = {
            r'\bHR\s+V\s+A\s+TSKE\b': 'HRVATSKE',  # Split "HRVATSKE"
            r'\bHR\s+VATSKE\b': 'HRVATSKE',
            r'\bZAGR\s+EB\b': 'ZAGREB',
            r'\bHR\s+V\s+A\s+T\s+S\s+K\s+E\b': 'HRVATSKE',
            r'\s+([čćžšđČĆŽŠĐ])\s+': r'\1',  # Fix spaced diacritics
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from Croatian text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Croatian sentence ending patterns
        sentence_endings = r'[.!?]+(?=\s+[A-ZČĆŽŠĐ]|\s*$)'
        
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def normalize_diacritics(self, text: str) -> str:
        """
        Normalize Croatian diacritics (optional - use carefully as it loses information).
        
        Args:
            text: Text with Croatian diacritics
            
        Returns:
            Text with normalized characters
        """
        for diacritic, normalized in self.diacritic_map.items():
            text = text.replace(diacritic, normalized)
        
        return text
    
    def is_meaningful_text(self, text: str, min_words: int = 3) -> bool:
        """
        Check if text contains meaningful content.
        
        Args:
            text: Text to check
            min_words: Minimum number of words required
            
        Returns:
            True if text is meaningful
        """
        if not text or not text.strip():
            return False
        
        words = text.split()
        
        # Check minimum word count
        if len(words) < min_words:
            return False
        
        # Check if text is mostly numbers or special characters
        word_chars = sum(len(re.findall(r'[a-zA-ZčćžšđČĆŽŠĐ]', word)) for word in words)
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0 or word_chars / total_chars < 0.5:
            return False
        
        return True


def clean_croatian_text(text: str, preserve_structure: bool = True) -> str:
    """
    Convenience function to clean Croatian text.
    
    Args:
        text: Raw text to clean
        preserve_structure: Whether to preserve paragraph structure
        
    Returns:
        Cleaned text
    """
    cleaner = CroatianTextCleaner()
    return cleaner.clean_text(text, preserve_structure)