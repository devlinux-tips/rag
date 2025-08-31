"""
Response parser for processing LLM outputs in Croatian RAG system.
Handles post-processing, validation, and formatting of generated responses.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    """Structured representation of parsed LLM response."""
    content: str
    confidence: Optional[float] = None
    sources_mentioned: List[str] = None
    has_answer: bool = True
    language: str = "hr"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sources_mentioned is None:
            self.sources_mentioned = []
        if self.metadata is None:
            self.metadata = {}


class CroatianResponseParser:
    """Parser for Croatian language LLM responses."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns for Croatian language processing
        self.no_answer_patterns = [
            r"ne mogu pronaći",
            r"nije dostupno",
            r"nema informacija",
            r"ne znam",
            r"nije jasno",
            r"nedostaju podaci"
        ]
        
        # Patterns for source references
        self.source_patterns = [
            r"\[dokument \d+\]",
            r"prema dokumentu",
            r"u tekstu se navodi",
            r"dokument spominje"
        ]
        
        # Confidence indicators
        self.confidence_indicators = {
            "high": [
                r"jasno je da", r"definitvno", r"sigurno", 
                r"bez sumnje", r"očigledno"
            ],
            "medium": [
                r"vjerovatno", r"moguće je", r"čini se", 
                r"indicira", r"sugerira"
            ],
            "low": [
                r"možda", r"nije sigurno", r"teško je reći",
                r"nedovoljno informacija", r"ograničeno"
            ]
        }
    
    def parse_response(
        self, 
        raw_response: str, 
        query: str = "",
        context_chunks: Optional[List[str]] = None
    ) -> ParsedResponse:
        """
        Parse and analyze LLM response.
        
        Args:
            raw_response: Raw response text from LLM
            query: Original user query
            context_chunks: Context chunks used for generation
            
        Returns:
            ParsedResponse object with analyzed content
        """
        if not raw_response or not raw_response.strip():
            return ParsedResponse(
                content="Žao mi je, nisam mogao generirati odgovor.",
                has_answer=False,
                confidence=0.0
            )
        
        # Clean and normalize response
        cleaned_response = self._clean_response(raw_response)
        
        # Check if response indicates no answer found
        has_answer = not self._indicates_no_answer(cleaned_response)
        
        # Extract source mentions
        sources = self._extract_source_mentions(cleaned_response)
        
        # Estimate confidence
        confidence = self._estimate_confidence(cleaned_response)
        
        # Detect language
        language = self._detect_language(cleaned_response)
        
        # Build metadata
        metadata = {
            "original_length": len(raw_response),
            "cleaned_length": len(cleaned_response),
            "query_length": len(query),
            "context_chunks_count": len(context_chunks) if context_chunks else 0
        }
        
        return ParsedResponse(
            content=cleaned_response,
            confidence=confidence,
            sources_mentioned=sources,
            has_answer=has_answer,
            language=language,
            metadata=metadata
        )
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and normalize response text.
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', response.strip())
        
        # Remove common prefixes that models might add
        prefixes_to_remove = [
            r'^(odgovor|answer):\s*',
            r'^(rezultat|result):\s*',
            r'^(zaključak|conclusion):\s*'
        ]
        
        for pattern in prefixes_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Fix common punctuation issues
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
        cleaned = re.sub(r'([.!?])\s*([A-ZČĆŠŽĐ])', r'\1 \2', cleaned)
        
        return cleaned.strip()
    
    def _indicates_no_answer(self, response: str) -> bool:
        """
        Check if response indicates no answer was found.
        
        Args:
            response: Response text to check
            
        Returns:
            True if response indicates no answer available
        """
        response_lower = response.lower()
        return any(
            re.search(pattern, response_lower) 
            for pattern in self.no_answer_patterns
        )
    
    def _extract_source_mentions(self, response: str) -> List[str]:
        """
        Extract source references from response.
        
        Args:
            response: Response text
            
        Returns:
            List of source references found
        """
        sources = []
        
        for pattern in self.source_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                sources.append(match.group())
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(sources))
    
    def _estimate_confidence(self, response: str) -> float:
        """
        Estimate confidence level based on language indicators.
        
        Args:
            response: Response text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        response_lower = response.lower()
        
        # Count confidence indicators
        high_count = sum(
            1 for pattern in self.confidence_indicators["high"]
            if re.search(pattern, response_lower)
        )
        
        medium_count = sum(
            1 for pattern in self.confidence_indicators["medium"] 
            if re.search(pattern, response_lower)
        )
        
        low_count = sum(
            1 for pattern in self.confidence_indicators["low"]
            if re.search(pattern, response_lower)
        )
        
        # Calculate weighted confidence score
        total_indicators = high_count + medium_count + low_count
        
        if total_indicators == 0:
            return 0.5  # Neutral confidence
        
        weighted_score = (
            high_count * 1.0 + 
            medium_count * 0.6 + 
            low_count * 0.2
        ) / total_indicators
        
        return min(max(weighted_score, 0.0), 1.0)
    
    def _detect_language(self, response: str) -> str:
        """
        Detect response language.
        
        Args:
            response: Response text
            
        Returns:
            Language code ('hr' for Croatian, 'en' for English, etc.)
        """
        # Simple heuristic based on Croatian-specific characters and words
        croatian_chars = 'čćšžđČĆŠŽĐ'
        croatian_words = [
            'je', 'se', 'na', 'za', 'da', 'su', 'ili', 'ako', 'kad', 'što',
            'biti', 'imati', 'moći', 'htjeti', 'trebati'
        ]
        
        # Count Croatian indicators
        char_score = sum(1 for char in response if char in croatian_chars)
        word_score = sum(1 for word in croatian_words if word in response.lower())
        
        total_score = char_score + word_score
        
        # Simple threshold-based detection
        if total_score > 2:
            return "hr"
        elif any(word in response.lower() for word in ['the', 'and', 'or', 'but', 'with']):
            return "en"
        else:
            return "unknown"
    
    def format_for_display(self, parsed_response: ParsedResponse) -> str:
        """
        Format parsed response for user display.
        
        Args:
            parsed_response: ParsedResponse object
            
        Returns:
            Formatted response text
        """
        formatted = parsed_response.content
        
        # Add confidence indicator if available
        if parsed_response.confidence is not None:
            if parsed_response.confidence >= 0.8:
                confidence_label = "visoka pouzdanost"
            elif parsed_response.confidence >= 0.5:
                confidence_label = "umjerena pouzdanost"
            else:
                confidence_label = "niska pouzdanost"
            
            formatted += f"\n\n[{confidence_label}]"
        
        # Add source information if available
        if parsed_response.sources_mentioned:
            formatted += f"\n\nIzvori: {', '.join(parsed_response.sources_mentioned)}"
        
        return formatted


def create_response_parser() -> CroatianResponseParser:
    """
    Factory function to create Croatian response parser.
    
    Returns:
        Configured CroatianResponseParser instance
    """
    return CroatianResponseParser()