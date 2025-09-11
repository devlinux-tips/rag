"""
Response parser for processing LLM outputs in multilingual RAG system.
Clean architecture with dependency injection and pure functions.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ===== PURE FUNCTIONS =====


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    Pure function - no side effects, deterministic output.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace

    Raises:
        ValueError: If text is not string
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be string, got {type(text)}")

    # Replace multiple whitespace with single space
    normalized = re.sub(r"\s+", " ", text.strip())
    return normalized


def remove_prefixes(text: str, prefixes: List[str]) -> str:
    """
    Remove specified prefixes from text.
    Pure function - no side effects, deterministic output.

    Args:
        text: Input text
        prefixes: List of regex patterns to remove

    Returns:
        Text with prefixes removed

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be string, got {type(text)}")

    if not isinstance(prefixes, list):
        raise ValueError(f"Prefixes must be list, got {type(prefixes)}")

    result = text

    for pattern in prefixes:
        if not isinstance(pattern, str):
            continue  # Skip invalid patterns

        try:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
        except re.error:
            # Skip invalid regex patterns
            continue

    return result


def fix_punctuation_spacing(text: str) -> str:
    """
    Fix common punctuation spacing issues.
    Pure function - no side effects, deterministic output.

    Args:
        text: Input text

    Returns:
        Text with fixed punctuation spacing
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be string, got {type(text)}")

    # Remove spaces before punctuation
    fixed = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Add space after sentence-ending punctuation followed by capital letter
    fixed = re.sub(r"([.!?])\s*([A-ZČĆŠŽĐ])", r"\1 \2", fixed)

    return fixed


def clean_response_text(text: str, prefixes_to_remove: List[str] = None) -> str:
    """
    Clean and normalize response text.
    Pure function - no side effects, deterministic output.

    Args:
        text: Raw response text
        prefixes_to_remove: Regex patterns to remove

    Returns:
        Cleaned response text

    Raises:
        ValueError: If text is invalid
    """
    if not text:
        raise ValueError("Text cannot be empty")

    prefixes_to_remove = prefixes_to_remove or []

    # Apply cleaning steps
    cleaned = normalize_whitespace(text)
    cleaned = remove_prefixes(cleaned, prefixes_to_remove)
    cleaned = fix_punctuation_spacing(cleaned)

    return cleaned.strip()


def check_no_answer_patterns(text: str, no_answer_patterns: List[str]) -> bool:
    """
    Check if text contains patterns indicating no answer.
    Pure function - no side effects, deterministic output.

    Args:
        text: Text to check
        no_answer_patterns: List of regex patterns indicating no answer

    Returns:
        True if text indicates no answer available

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be string, got {type(text)}")

    if not isinstance(no_answer_patterns, list):
        raise ValueError(
            f"No answer patterns must be list, got {type(no_answer_patterns)}"
        )

    text_lower = text.lower()

    for pattern in no_answer_patterns:
        if not isinstance(pattern, str):
            continue  # Skip invalid patterns

        try:
            if re.search(pattern, text_lower):
                return True
        except re.error:
            # Skip invalid regex patterns
            continue

    return False


def extract_source_references(text: str, source_patterns: List[str]) -> List[str]:
    """
    Extract source references from text.
    Pure function - no side effects, deterministic output.

    Args:
        text: Text to search for sources
        source_patterns: List of regex patterns for source matching

    Returns:
        List of unique source references found

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be string, got {type(text)}")

    if not isinstance(source_patterns, list):
        raise ValueError(f"Source patterns must be list, got {type(source_patterns)}")

    sources = []

    for pattern in source_patterns:
        if not isinstance(pattern, str):
            continue  # Skip invalid patterns

        try:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                sources.append(match.group())
        except re.error:
            # Skip invalid regex patterns
            continue

    # Remove duplicates while preserving order
    return list(dict.fromkeys(sources))


def calculate_confidence_score(
    text: str, confidence_indicators: Dict[str, List[str]]
) -> float:
    """
    Calculate confidence score based on language indicators.
    Pure function - no side effects, deterministic output.

    Args:
        text: Text to analyze
        confidence_indicators: Dict mapping confidence levels to patterns

    Returns:
        Confidence score between 0.0 and 1.0

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be string, got {type(text)}")

    if not isinstance(confidence_indicators, dict):
        raise ValueError(
            f"Confidence indicators must be dict, got {type(confidence_indicators)}"
        )

    text_lower = text.lower()

    # Count indicators for each confidence level
    indicator_counts = {}

    for level, patterns in confidence_indicators.items():
        if not isinstance(patterns, list):
            continue

        count = 0
        for pattern in patterns:
            if not isinstance(pattern, str):
                continue

            try:
                if re.search(pattern, text_lower):
                    count += 1
            except re.error:
                continue  # Skip invalid patterns

        indicator_counts[level] = count

    # Calculate weighted score
    if "high" not in indicator_counts:
        raise ValueError("Missing 'high' in confidence indicators")
    if "medium" not in indicator_counts:
        raise ValueError("Missing 'medium' in confidence indicators")
    if "low" not in indicator_counts:
        raise ValueError("Missing 'low' in confidence indicators")

    high_count = indicator_counts["high"]
    medium_count = indicator_counts["medium"]
    low_count = indicator_counts["low"]

    total_indicators = high_count + medium_count + low_count

    if total_indicators == 0:
        return 0.5  # Neutral confidence

    # Weight: high=1.0, medium=0.6, low=0.2
    weighted_score = (
        high_count * 1.0 + medium_count * 0.6 + low_count * 0.2
    ) / total_indicators

    return min(max(weighted_score, 0.0), 1.0)


def detect_language_by_patterns(
    text: str,
    language_patterns: Dict[str, List[str]],
    default_language: str = "unknown",
) -> str:
    """
    Detect language based on word and character patterns.
    Pure function - no side effects, deterministic output.

    Args:
        text: Text to analyze
        language_patterns: Dict mapping language codes to word lists
        default_language: Default language if no patterns match

    Returns:
        Detected language code

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be string, got {type(text)}")

    if not isinstance(language_patterns, dict):
        raise ValueError(
            f"Language patterns must be dict, got {type(language_patterns)}"
        )

    text_lower = text.lower()

    # Count diacritics (Unicode-based detection)
    diacritic_count = sum(
        1 for char in text if unicodedata.normalize("NFD", char) != char
    )

    # Score each language
    language_scores = {}

    for lang_code, word_patterns in language_patterns.items():
        if not isinstance(word_patterns, list):
            continue

        word_score = sum(
            1
            for pattern in word_patterns
            if isinstance(pattern, str) and pattern.lower() in text_lower
        )

        language_scores[lang_code] = word_score

    # Find best match
    if language_scores:
        best_language = max(language_scores.items(), key=lambda x: x[1])

        # Require minimum score (words + diacritics)
        total_score = best_language[1] + (diacritic_count / 10)  # Weighted diacritics

        if total_score > 0.5:  # Minimum threshold
            return best_language[0]

    return default_language


def format_display_text(
    content: str,
    confidence: Optional[float] = None,
    sources: Optional[List[str]] = None,
    confidence_labels: Optional[Dict[str, str]] = None,
    sources_prefix: str = "Sources",
) -> str:
    """
    Format response content for display.
    Pure function - no side effects, deterministic output.

    Args:
        content: Main response content
        confidence: Confidence score (0.0-1.0)
        sources: List of source references
        confidence_labels: Labels for confidence levels
        sources_prefix: Prefix for sources section

    Returns:
        Formatted display text

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(content, str):
        raise ValueError(f"Content must be string, got {type(content)}")

    result_parts = [content]

    # Add confidence indicator
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"Confidence must be numeric, got {type(confidence)}")

        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {confidence}"
            )

        confidence_labels = confidence_labels or {
            "high": "High Confidence",
            "medium": "Medium Confidence",
            "low": "Low Confidence",
        }

        if confidence >= 0.8:
            if "high" not in confidence_labels:
                raise ValueError("Missing 'high' confidence label")
            label = confidence_labels["high"]
        elif confidence >= 0.5:
            if "medium" not in confidence_labels:
                raise ValueError("Missing 'medium' confidence label")
            label = confidence_labels["medium"]
        else:
            if "low" not in confidence_labels:
                raise ValueError("Missing 'low' confidence label")
            label = confidence_labels["low"]

        result_parts.append(f"\n\n[{label}]")

    # Add sources
    if sources:
        if not isinstance(sources, list):
            raise ValueError(f"Sources must be list, got {type(sources)}")

        valid_sources = [s for s in sources if isinstance(s, str) and s.strip()]

        if valid_sources:
            sources_text = f"\n\n{sources_prefix}: {', '.join(valid_sources)}"
            result_parts.append(sources_text)

    return "".join(result_parts)


# ===== DATA STRUCTURES =====


@dataclass
class ParsedResponse:
    """Structured representation of parsed LLM response."""

    content: str
    confidence: Optional[float] = None
    sources_mentioned: List[str] = field(default_factory=list)
    has_answer: bool = True
    language: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate fields after initialization."""
        if not isinstance(self.content, str):
            raise ValueError("Content must be string")

        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise ValueError("Confidence must be numeric or None")
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError("Confidence must be between 0.0 and 1.0")

        if not isinstance(self.sources_mentioned, list):
            raise ValueError("Sources mentioned must be list")

        if not isinstance(self.has_answer, bool):
            raise ValueError("Has answer must be boolean")

        if not isinstance(self.language, str):
            raise ValueError("Language must be string")

        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be dict")


@dataclass
class ParsingConfig:
    """Configuration for response parsing."""

    no_answer_patterns: List[str] = field(default_factory=list)
    source_patterns: List[str] = field(default_factory=list)
    confidence_indicators: Dict[str, List[str]] = field(default_factory=dict)
    language_patterns: Dict[str, List[str]] = field(default_factory=dict)
    cleaning_prefixes: List[str] = field(default_factory=list)
    display_settings: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.no_answer_patterns, list):
            raise ValueError("No answer patterns must be list")

        if not isinstance(self.source_patterns, list):
            raise ValueError("Source patterns must be list")

        if not isinstance(self.confidence_indicators, dict):
            raise ValueError("Confidence indicators must be dict")

        if not isinstance(self.language_patterns, dict):
            raise ValueError("Language patterns must be dict")

        if not isinstance(self.cleaning_prefixes, list):
            raise ValueError("Cleaning prefixes must be list")

        if not isinstance(self.display_settings, dict):
            raise ValueError("Display settings must be dict")


# ===== PROTOCOLS =====


class ConfigProvider(Protocol):
    """Protocol for parsing configuration providers."""

    def get_parsing_config(self, language: str) -> ParsingConfig:
        """Get parsing configuration for language."""
        ...


# ===== CORE CLASSES =====


class MultilingualResponseParser:
    """Parser for multilingual LLM responses with language-specific behavior."""

    def __init__(self, config_provider: ConfigProvider, language: str = "hr"):
        """
        Initialize parser with dependency injection.

        Args:
            config_provider: Provider for parsing configuration
            language: Language code
        """
        self.language = language
        self.config_provider = config_provider
        self.logger = logging.getLogger(__name__)

        try:
            self._config = config_provider.get_parsing_config(language)
            self.logger.debug(f"Initialized parser for language: {language}")
        except Exception as e:
            self.logger.error(f"Failed to initialize parser for {language}: {e}")
            raise

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

        Raises:
            ValueError: If raw_response is invalid
        """
        # Handle empty response
        if not raw_response or not raw_response.strip():
            if "no_answer_message" not in self._config.display_settings:
                raise ValueError("Missing 'no_answer_message' in display settings")
            no_answer_message = self._config.display_settings["no_answer_message"]
            return ParsedResponse(
                content=no_answer_message,
                has_answer=False,
                confidence=0.0,
                language=self.language,
            )

        # Clean response text
        cleaned_content = clean_response_text(
            raw_response, prefixes_to_remove=self._config.cleaning_prefixes
        )

        # Analyze response characteristics
        has_answer = not check_no_answer_patterns(
            cleaned_content, self._config.no_answer_patterns
        )

        sources = extract_source_references(
            cleaned_content, self._config.source_patterns
        )

        confidence = calculate_confidence_score(
            cleaned_content, self._config.confidence_indicators
        )

        detected_language = detect_language_by_patterns(
            cleaned_content,
            self._config.language_patterns,
            default_language=self.language,
        )

        # Build metadata
        metadata = {
            "original_length": len(raw_response),
            "cleaned_length": len(cleaned_content),
            "query_length": len(query),
            "context_chunks_count": len(context_chunks) if context_chunks else 0,
            "processing_language": self.language,
        }

        return ParsedResponse(
            content=cleaned_content,
            confidence=confidence,
            sources_mentioned=sources,
            has_answer=has_answer,
            language=detected_language,
            metadata=metadata,
        )

    def format_for_display(self, parsed_response: ParsedResponse) -> str:
        """
        Format parsed response for user display.

        Args:
            parsed_response: ParsedResponse object

        Returns:
            Formatted response text

        Raises:
            ValueError: If parsed_response is invalid
        """
        # Validate display settings
        required_labels = [
            "high_confidence_label",
            "medium_confidence_label",
            "low_confidence_label",
        ]
        for label in required_labels:
            if label not in self._config.display_settings:
                raise ValueError(f"Missing '{label}' in display settings")

        confidence_labels = {
            "high": self._config.display_settings["high_confidence_label"],
            "medium": self._config.display_settings["medium_confidence_label"],
            "low": self._config.display_settings["low_confidence_label"],
        }

        if "sources_prefix" not in self._config.display_settings:
            raise ValueError("Missing 'sources_prefix' in display settings")
        sources_prefix = self._config.display_settings["sources_prefix"]

        return format_display_text(
            content=parsed_response.content,
            confidence=parsed_response.confidence,
            sources=parsed_response.sources_mentioned,
            confidence_labels=confidence_labels,
            sources_prefix=sources_prefix,
        )


# ===== FACTORY FUNCTIONS =====


def create_response_parser(
    config_provider: ConfigProvider, language: str = "hr"
) -> MultilingualResponseParser:
    """
    Factory function to create multilingual response parser.

    Args:
        config_provider: Configuration provider
        language: Language code for language-specific behavior

    Returns:
        Configured MultilingualResponseParser instance
    """
    return MultilingualResponseParser(config_provider, language)


def create_mock_config_provider(
    no_answer_patterns: Optional[List[str]] = None,
    source_patterns: Optional[List[str]] = None,
    confidence_indicators: Optional[Dict[str, List[str]]] = None,
    language_patterns: Optional[Dict[str, List[str]]] = None,
    cleaning_prefixes: Optional[List[str]] = None,
    display_settings: Optional[Dict[str, str]] = None,
) -> ConfigProvider:
    """
    Factory function to create mock configuration provider.

    Args:
        no_answer_patterns: Custom no-answer patterns
        source_patterns: Custom source patterns
        confidence_indicators: Custom confidence indicators
        language_patterns: Custom language patterns
        cleaning_prefixes: Custom cleaning prefixes
        display_settings: Custom display settings

    Returns:
        Mock ConfigProvider
    """

    class MockConfigProvider:
        def get_parsing_config(self, language: str) -> ParsingConfig:
            default_no_answer = [
                "ne znam",
                "ne mogu",
                "nema podataka",
                "don't know",
                "no information",
            ]
            default_source = [
                r"izvor:",
                r"prema:",
                r"dokumentu:",
                r"source:",
                r"according to:",
            ]
            default_confidence = {
                "high": ["sigurno", "certainly", "definitive"],
                "medium": ["možda", "probably", "likely"],
                "low": ["nisam siguran", "not sure", "uncertain"],
            }
            default_language = {
                "hr": ["je", "su", "ima", "biti", "kao", "da", "se", "i", "u", "na"],
                "en": [
                    "is",
                    "are",
                    "the",
                    "and",
                    "of",
                    "to",
                    "in",
                    "that",
                    "have",
                    "for",
                ],
            }
            default_prefixes = [r"^(odgovor|answer):\s*", r"^(pitanje|question):\s*"]
            default_display = {
                "no_answer_message": "No answer available.",
                "high_confidence_label": "High Confidence",
                "medium_confidence_label": "Medium Confidence",
                "low_confidence_label": "Low Confidence",
                "sources_prefix": "Sources",
            }

            return ParsingConfig(
                no_answer_patterns=no_answer_patterns or default_no_answer,
                source_patterns=source_patterns or default_source,
                confidence_indicators=confidence_indicators or default_confidence,
                language_patterns=language_patterns or default_language,
                cleaning_prefixes=cleaning_prefixes or default_prefixes,
                display_settings=display_settings or default_display,
            )

    return MockConfigProvider()
