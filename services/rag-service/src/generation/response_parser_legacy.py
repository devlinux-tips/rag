"""
Response parser for processing LLM outputs in multilingual RAG system.
Handles post-processing, validation, and formatting of generated responses.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.config_loader import (get_language_specific_config,
                                   get_response_parsing_config)


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


class MultilingualResponseParser:
    """Parser for multilingual LLM responses with language-specific behavior."""

    def __init__(self, language: str = "hr"):
        self.language = language
        self.logger = logging.getLogger(__name__)

        # Load language-specific configuration first, fall back to general config
        language_config = get_language_specific_config(
            "response_parsing", self.language
        )

        # Load general configuration as fallback
        general_config = get_response_parsing_config()

        # Merge language config with general config (language-specific takes priority)
        self._config = {**general_config, **language_config}

        # Load patterns from merged config
        self.no_answer_patterns = self._config.get(
            "no_answer_patterns", ["ne znam", "ne mogu", "nema podataka"]
        )
        self.source_patterns = self._config.get(
            "source_patterns", ["izvor:", "prema:", "dokumentu:"]
        )
        self.confidence_indicators = self._config.get(
            "confidence_indicators",
            {"high": ["sigurno"], "medium": ["možda"], "low": ["nisam siguran"]},
        )

    def parse_response(
        self,
        raw_response: str,
        query: str = "",
        context_chunks: Optional[List[str]] = None,
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
                content=self._config["display"]["no_answer_message"],
                has_answer=False,
                confidence=0.0,
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
            "context_chunks_count": len(context_chunks) if context_chunks else 0,
        }

        return ParsedResponse(
            content=cleaned_response,
            confidence=confidence,
            sources_mentioned=sources,
            has_answer=has_answer,
            language=language,
            metadata=metadata,
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
        cleaned = re.sub(r"\s+", " ", response.strip())

        # Remove common prefixes that models might add
        prefixes_to_remove = self._config["cleaning"]["prefixes_to_remove"]

        for pattern in prefixes_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Fix common punctuation issues
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        cleaned = re.sub(r"([.!?])\s*([A-ZČĆŠŽĐ])", r"\1 \2", cleaned)

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
            re.search(pattern, response_lower) for pattern in self.no_answer_patterns
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
            1
            for pattern in self.confidence_indicators["high"]
            if re.search(pattern, response_lower)
        )

        medium_count = sum(
            1
            for pattern in self.confidence_indicators["medium"]
            if re.search(pattern, response_lower)
        )

        low_count = sum(
            1
            for pattern in self.confidence_indicators["low"]
            if re.search(pattern, response_lower)
        )

        # Calculate weighted confidence score
        total_indicators = high_count + medium_count + low_count

        if total_indicators == 0:
            return 0.5  # Neutral confidence

        weighted_score = (
            high_count * 1.0 + medium_count * 0.6 + low_count * 0.2
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
        import unicodedata

        # Use Unicode-based diacritic detection - works for ALL languages
        def has_diacritics(char):
            return unicodedata.normalize("NFD", char) != char

        language_words = self._config["language_detection"].get(
            f"{self.language}_words", []
        )

        # Count Unicode diacritics (language-agnostic)
        char_score = sum(1 for char in response if has_diacritics(char))
        word_score = sum(1 for word in language_words if word in response.lower())

        total_score = char_score + word_score

        # Return configured language if indicators are found, otherwise try to detect
        if total_score > 2:
            return self.language
        elif any(
            word in response.lower()
            for word in self._config["language_detection"].get("english_words", [])
        ):
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
            display_config = self._config["display"]
            if parsed_response.confidence >= 0.8:
                confidence_label = display_config["high_confidence_label"]
            elif parsed_response.confidence >= 0.5:
                confidence_label = display_config["medium_confidence_label"]
            else:
                confidence_label = display_config["low_confidence_label"]

            formatted += f"\n\n[{confidence_label}]"

        # Add source information if available
        if parsed_response.sources_mentioned:
            sources_prefix = self._config["display"]["sources_prefix"]
            formatted += (
                f"\n\n{sources_prefix}: {', '.join(parsed_response.sources_mentioned)}"
            )

        return formatted


def create_response_parser(language: str = "hr") -> MultilingualResponseParser:
    """
    Factory function to create multilingual response parser.

    Args:
        language: Language code for language-specific behavior

    Returns:
        Configured MultilingualResponseParser instance
    """
    return MultilingualResponseParser(language=language)
