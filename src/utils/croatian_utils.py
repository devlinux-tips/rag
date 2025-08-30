"""
Croatian language utilities for text processing.
"""
import re
from typing import List


class CroatianTextProcessor:
    """Utilities for processing Croatian text."""
    
    CROATIAN_CHARS = "ČčĆćŠšŽžĐđ"
    
    def __init__(self):
        self.diacritic_map = {
            'č': 'c', 'ć': 'c', 'š': 's', 'ž': 'z', 'đ': 'd',
            'Č': 'C', 'Ć': 'C', 'Š': 'S', 'Ž': 'Z', 'Đ': 'D'
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize Croatian text while preserving diacritics."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def has_croatian_chars(self, text: str) -> bool:
        """Check if text contains Croatian-specific characters."""
        return any(char in text for char in self.CROATIAN_CHARS)
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Croatian diacritics (use sparingly for search)."""
        for croatian, latin in self.diacritic_map.items():
            text = text.replace(croatian, latin)
        return text
