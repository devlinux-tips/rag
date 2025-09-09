"""
Multilingual query preprocessing for intelligent document retrieval.
Handles query analysis, expansion, and language-specific processing for Croatian and English.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..utils.config_protocol import ConfigProvider

from ..utils.config_loader import (get_language_shared,
                                   get_language_specific_config,
                                   get_query_processing_config)


class QueryType(Enum):
    """Types of queries the system can handle."""

    FACTUAL = "factual"  # Who, what, when, where
    EXPLANATORY = "explanatory"  # How, why, explain
    COMPARISON = "comparison"  # Compare, difference, similar
    SUMMARIZATION = "summarization"  # Summarize, overview
    GENERAL = "general"  # General questions


@dataclass
class QueryProcessingConfig:
    """Configuration for query processing."""

    language: str
    expand_synonyms: bool = True
    normalize_case: bool = True
    remove_stopwords: bool = True
    min_query_length: int = 3
    max_expanded_terms: int = 10
    enable_spell_check: bool = False  # Could add later

    @classmethod
    def from_config(
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "QueryProcessingConfig":
        """Load configuration from dictionary or config provider."""
        if config_dict:
            config = config_dict
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get query processing config through provider
            full_config = config_provider.load_config("config")
            config = full_config["query_processing"]

        return cls(
            language=config["language"],
            expand_synonyms=config["expand_synonyms"],
            normalize_case=config["normalize_case"],
            remove_stopwords=config["remove_stopwords"],
            min_query_length=config["min_query_length"],
            max_expanded_terms=config["max_expanded_terms"],
            enable_spell_check=config["enable_spell_check"],
        )


@dataclass
class ProcessedQuery:
    """Result of query preprocessing."""

    original: str
    processed: str
    query_type: QueryType
    keywords: List[str]
    expanded_terms: List[str]
    filters: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]


class MultilingualQueryProcessor:
    """Preprocessor for multilingual queries."""

    def __init__(self, language: str, config: QueryProcessingConfig = None):
        """
        Initialize multilingual query processor.

        Args:
            language: Language code ('hr' for Croatian, 'en' for English)
            config: Query processing configuration
        """
        self.language = language
        self.config = config or QueryProcessingConfig.from_config()
        self.logger = logging.getLogger(__name__)

        # Load language-specific configurations
        self.shared_config = get_language_shared(self.language)

        # Load stop words from shared config
        self.stop_words = set(self.shared_config["stopwords"]["words"])

        # Load question patterns from shared config
        self.question_patterns = self.shared_config["question_patterns"]

        # Load morphological patterns if available
        self.morphological_patterns = self.shared_config.get(
            "morphological_patterns", {}
        )

        # Load synonym groups if available
        self.synonym_groups = self.shared_config.get("synonym_groups", {})

    def process_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ProcessedQuery:
        """
        Process a query for multilingual retrieval.

        Args:
            query: Raw query string
            context: Additional context for processing

        Returns:
            ProcessedQuery with extracted information
        """
        context = context or {}
        original_query = query

        try:
            # Validate query length
            if len(query.strip()) < self.config.min_query_length:
                return self._create_empty_result(query)

            # Step 1: Preprocess text
            processed_query = self._preprocess_text(query)

            # Step 2: Classify query type
            query_type = self._classify_query_type(processed_query)

            # Step 3: Extract keywords
            keywords = self._extract_keywords(processed_query)

            # Step 4: Expand terms
            expanded_terms = []
            if self.config.expand_synonyms:
                expanded_terms = self._expand_terms(keywords)

            # Step 5: Generate filters
            filters = self._generate_filters(processed_query, context)

            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(
                processed_query, keywords, query_type
            )

            # Create metadata
            metadata = {
                "processing_steps": [
                    "preprocess",
                    "classify",
                    "extract",
                    "expand",
                    "filter",
                ],
                "original_length": len(original_query),
                "processed_length": len(processed_query),
                "keyword_count": len(keywords),
                "expanded_count": len(expanded_terms),
                "language": self.config.language,
            }

            return ProcessedQuery(
                original=original_query,
                processed=processed_query,
                query_type=query_type,
                keywords=keywords,
                expanded_terms=expanded_terms,
                filters=filters,
                confidence=confidence,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return self._create_error_result(original_query, str(e))

    def _preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing for multilingual queries.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Normalize case if configured
        if self.config.normalize_case:
            text = text.lower()

        # Normalize quotation marks
        text = text.replace("„", '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")

        # Remove excessive punctuation
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        return text

    def _classify_query_type(self, query: str) -> QueryType:
        """
        Classify the type of query based on language patterns.

        Args:
            query: Preprocessed query text

        Returns:
            Detected QueryType
        """
        query_lower = query.lower()

        # Check each query type pattern
        for query_type_str, patterns in self.question_patterns.items():
            query_type = QueryType(query_type_str)
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return query_type

        # Default to general if no specific pattern found
        return QueryType.GENERAL

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query.

        Args:
            query: Preprocessed query text

        Returns:
            List of extracted keywords
        """
        # Split into words
        words = re.findall(r"\b\w+\b", query.lower())

        # Remove stop words if configured
        if self.config.remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        # Remove very short words
        words = [word for word in words if len(word) >= 2]

        # Remove duplicates while preserving order
        keywords = []
        seen = set()
        for word in words:
            if word not in seen:
                keywords.append(word)
                seen.add(word)

        return keywords

    def _expand_terms(self, keywords: List[str]) -> List[str]:
        """
        Expand keywords with synonyms and morphological variations.

        Args:
            keywords: List of keywords to expand

        Returns:
            List of expanded terms
        """
        expanded = []

        for keyword in keywords:
            # Add morphological variations
            if keyword in self.morphological_patterns:
                expanded.extend(self.morphological_patterns[keyword])

            # Add synonyms
            for base_word, synonyms in self.synonym_groups.items():
                if keyword == base_word or keyword in synonyms:
                    expanded.extend(synonyms)

        # Remove duplicates and original keywords
        expanded_unique = []
        original_set = set(keywords)
        seen = set(keywords)  # Include original keywords in seen set

        for term in expanded:
            if term not in seen and term not in original_set:
                expanded_unique.append(term)
                seen.add(term)

        # Limit expansion
        return expanded_unique[: self.config.max_expanded_terms]

    def _generate_filters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata filters based on query content and context.

        Args:
            query: Processed query text
            context: Additional context information

        Returns:
            Dictionary of filters to apply
        """
        filters = {"language": self.language}

        # Load topic filters from language-specific config
        try:
            topic_config = get_language_specific_config("topic_filters", self.language)

            # Apply topic filters based on patterns
            for topic, patterns in topic_config.items():
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        filters["topic"] = topic
                        break
                if "topic" in filters:
                    break
        except Exception:
            # No topic filters available for this language
            pass

        # Add context-based filters
        if "user_preferences" in context:
            prefs = context["user_preferences"]
            if "region" in prefs:
                filters["region"] = prefs["region"]
            if "content_type" in prefs:
                filters["content_type"] = prefs["content_type"]

        return filters

    def _calculate_confidence(
        self, query: str, keywords: List[str], query_type: QueryType
    ) -> float:
        """
        Calculate confidence score for query processing quality.

        Args:
            query: Processed query
            keywords: Extracted keywords
            query_type: Detected query type

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        # Boost for reasonable query length
        if 10 <= len(query) <= 200:
            confidence += 0.2

        # Boost for good keyword extraction
        if 2 <= len(keywords) <= 8:
            confidence += 0.2

        # Boost for language-specific indicators
        language_config = get_language_specific_config(
            "language_indicators", self.language
        )
        language_indicators = language_config.get("indicators", [])
        if any(char in query for char in language_indicators):
            confidence += 0.1

        # Boost for specific query types (not general)
        if query_type != QueryType.GENERAL:
            confidence += 0.1

        # Penalize very short or very long queries
        if len(query) < 5 or len(query) > 300:
            confidence -= 0.2

        # Penalize if no meaningful keywords
        if len(keywords) == 0:
            confidence -= 0.3

        # Clamp to valid range
        return max(0.0, min(1.0, confidence))

    def _create_empty_result(self, query: str) -> ProcessedQuery:
        """Create result for empty/invalid query."""
        return ProcessedQuery(
            original=query,
            processed="",
            query_type=QueryType.GENERAL,
            keywords=[],
            expanded_terms=[],
            filters={"language": self.language},
            confidence=0.0,
            metadata={"error": "Query too short or empty"},
        )

    def _create_error_result(self, query: str, error: str) -> ProcessedQuery:
        """Create result for processing error."""
        return ProcessedQuery(
            original=query,
            processed=query,
            query_type=QueryType.GENERAL,
            keywords=[],
            expanded_terms=[],
            filters={"language": self.language},
            confidence=0.0,
            metadata={"error": error},
        )

    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity for debugging and optimization.

        Args:
            query: Query to analyze

        Returns:
            Dictionary with complexity analysis
        """
        analysis = {
            "character_count": len(query),
            "word_count": len(query.split()),
            "sentence_count": query.count(".")
            + query.count("!")
            + query.count("?")
            + 1,
            "has_question_mark": "?" in query,
            "complexity_score": 0.0,
        }

        # Add language-specific indicators
        try:
            language_config = get_language_specific_config(
                "language_indicators", self.language
            )
            indicators = language_config.get("indicators", [])
            analysis["has_language_indicators"] = any(c in query for c in indicators)
        except Exception:
            analysis["has_language_indicators"] = False

        # Calculate complexity score
        score = 0.0

        # Length complexity
        if analysis["word_count"] > 10:
            score += 0.3
        elif analysis["word_count"] > 5:
            score += 0.1

        # Multiple sentences
        if analysis["sentence_count"] > 1:
            score += 0.2

        # Question complexity
        if analysis["has_question_mark"]:
            score += 0.1

        # Language-specific complexity
        if analysis["has_language_indicators"]:
            score += 0.1

        analysis["complexity_score"] = min(1.0, score)

        return analysis

    def suggest_query_improvements(self, processed_query: ProcessedQuery) -> List[str]:
        """
        Suggest improvements for better query results.

        Args:
            processed_query: Result of query processing

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Load suggestions from language-specific config
        try:
            suggestions_config = get_language_specific_config(
                "suggestions", self.language
            )

            # Low confidence suggestions
            if processed_query.confidence < 0.5:
                if len(processed_query.keywords) < 2:
                    suggestions.extend(suggestions_config.get("more_keywords", []))
                if len(processed_query.original) < 10:
                    suggestions.extend(suggestions_config.get("expand_query", []))

            # Query type specific suggestions
            if processed_query.query_type == QueryType.GENERAL:
                suggestions.extend(suggestions_config.get("be_specific", []))

            # No expanded terms
            if self.config.expand_synonyms and not processed_query.expanded_terms:
                suggestions.extend(suggestions_config.get("try_synonyms", []))

            # No filters applied
            if len(processed_query.filters) <= 1:  # Only language filter
                suggestions.extend(suggestions_config.get("add_context", []))

        except Exception:
            # Fallback to basic suggestions
            suggestions = ["Add more specific terms for better results"]

        return suggestions


def create_query_processor(
    language: str, expand_synonyms: bool = True
) -> MultilingualQueryProcessor:
    """
    Factory function to create query processor.

    Args:
        language: Processing language
        expand_synonyms: Whether to expand with synonyms

    Returns:
        Configured MultilingualQueryProcessor
    """
    config = QueryProcessingConfig(language=language, expand_synonyms=expand_synonyms)
    return MultilingualQueryProcessor(language=language, config=config)
