"""
Fully testable query processing with dependency injection.
Clean slate recreation with mockable architecture for reliable testing.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast

from ..utils.config_models import QueryProcessingConfig

if TYPE_CHECKING:
    from ..utils.config_protocol import ConfigProvider

# Pure function algorithms for query processing


def normalize_text(
    text: str, normalize_case: bool = True, normalize_quotes: bool = True, normalize_punctuation: bool = True
) -> str:
    """
    Normalize text using pure transformations (pure function).

    Args:
        text: Text to normalize
        normalize_case: Convert to lowercase
        normalize_quotes: Normalize quote characters
        normalize_punctuation: Clean excessive punctuation

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove extra whitespace
    result = re.sub(r"\s+", " ", text.strip())

    # Normalize case if requested
    if normalize_case:
        result = result.lower()

    # Normalize quotation marks if requested
    if normalize_quotes:
        result = result.replace("„", '"').replace('"', '"')
        result = result.replace(""", "'").replace(""", "'")

    # Remove excessive punctuation if requested
    if normalize_punctuation:
        result = re.sub(r"[!]{2,}", "!", result)
        result = re.sub(r"[?]{2,}", "?", result)

    return result


def classify_query_by_patterns(query: str, pattern_config: dict[str, list[str]]) -> str:
    """
    Classify query type based on pattern matching (pure function).

    Args:
        query: Query text to classify
        pattern_config: Dictionary mapping query types to regex patterns

    Returns:
        Query type string ("factual", "explanatory", "comparison", etc.)
    """
    if not query or not pattern_config:
        return "general"

    query_lower = query.lower()

    # Check each query type pattern
    for query_type, patterns in pattern_config.items():
        for pattern in patterns:
            try:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return query_type
            except re.error:
                # Skip invalid regex patterns
                continue

    return "general"


def extract_keywords_from_text(
    text: str, stop_words: set[str] | None = None, min_word_length: int = 2, remove_stopwords: bool = True
) -> list[str]:
    """
    Extract keywords from text (pure function).

    Args:
        text: Text to extract keywords from
        stop_words: Set of stop words to filter out
        min_word_length: Minimum length for keywords
        remove_stopwords: Whether to remove stop words

    Returns:
        List of extracted keywords
    """
    if not text:
        return []

    stop_words = stop_words or set()

    # Extract words using regex
    words = re.findall(r"\b\w+\b", text.lower())

    # Remove stop words if configured
    if remove_stopwords:
        words = [word for word in words if word not in stop_words]

    # Remove short words
    words = [word for word in words if len(word) >= min_word_length]

    # Remove duplicates while preserving order
    keywords = []
    seen = set()
    for word in words:
        if word not in seen:
            keywords.append(word)
            seen.add(word)

    return keywords


def expand_terms_with_synonyms(
    keywords: list[str], synonym_groups: dict[str, list[str]], max_expanded_terms: int = 10
) -> list[str]:
    """
    Expand keywords with synonyms (pure function).

    Args:
        keywords: Original keywords
        synonym_groups: Dictionary mapping words to synonym lists
        max_expanded_terms: Maximum number of expanded terms to return

    Returns:
        List of expanded terms including synonyms
    """
    if not keywords or not synonym_groups:
        return keywords

    expanded = []

    for keyword in keywords:
        # Add original keyword
        if keyword not in expanded:
            expanded.append(keyword)

        # Add synonyms if available
        if keyword in synonym_groups:
            for synonym in synonym_groups[keyword]:
                if synonym not in expanded and len(expanded) < max_expanded_terms:
                    expanded.append(synonym)

    return expanded[:max_expanded_terms]


def calculate_query_confidence(
    original_query: str, processed_query: str, keywords: list[str], query_type: str, patterns_matched: int = 0
) -> float:
    """
    Calculate confidence score for query processing (pure function).

    Args:
        original_query: Original query text
        processed_query: Processed query text
        keywords: Extracted keywords
        query_type: Detected query type
        patterns_matched: Number of patterns matched

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not original_query:
        return 0.0

    confidence = 0.5  # Base confidence

    # Boost for having keywords
    if keywords:
        confidence += min(len(keywords) * 0.1, 0.3)

    # Boost for specific query type
    if query_type != "general":
        confidence += 0.2

    # Boost for pattern matches
    confidence += min(patterns_matched * 0.05, 0.1)

    # Penalty for very short queries
    if len(original_query.split()) < 3:
        confidence -= 0.2

    # Clamp to valid range
    return max(0.0, min(1.0, confidence))


def generate_query_filters(
    query: str, context: dict[str, Any], filter_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Generate retrieval filters from query and context (pure function).

    Args:
        query: Processed query text
        context: Additional context information
        filter_config: Configuration for filter generation

    Returns:
        Dictionary of filters
    """
    filters = {}
    filter_config = filter_config or {}

    # Language filter from context
    if "language" in context:
        filters["language"] = context["language"]

    # Date range filters
    if "date_range" in context:
        filters["date_range"] = context["date_range"]

    # Document type filters
    if "document_types" in context:
        filters["document_types"] = context["document_types"]

    # Topic filters based on keywords
    if "topic_patterns" not in filter_config:
        raise ValueError("Missing 'topic_patterns' in filter configuration")
    topic_patterns = filter_config["topic_patterns"]
    query_lower = query.lower()

    for topic, patterns in topic_patterns.items():
        for pattern in patterns:
            try:
                if re.search(pattern, query_lower):
                    filters.setdefault("topics", []).append(topic)
                    break
            except re.error:
                continue

    return filters


# Configuration and data classes


class QueryType(Enum):
    """Types of queries the system can handle."""

    FACTUAL = "factual"
    EXPLANATORY = "explanatory"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


# Note: QueryProcessingConfig is now imported from config_models.py


@dataclass
class ProcessedQuery:
    """Result of query preprocessing."""

    original: str
    processed: str
    query_type: QueryType
    keywords: list[str]
    expanded_terms: list[str]
    filters: dict[str, Any]
    confidence: float
    metadata: dict[str, Any]


# Protocol definitions for dependency injection


class LanguageDataProvider(Protocol):
    """Protocol for language-specific data providers."""

    def get_stop_words(self, language: str) -> set[str]:
        """Get stop words for language."""
        ...

    def get_question_patterns(self, language: str) -> dict[str, list[str]]:
        """Get question classification patterns for language."""
        ...

    def get_synonym_groups(self, language: str) -> dict[str, list[str]]:
        """Get synonym groups for language."""
        ...

    def get_morphological_patterns(self, language: str) -> dict[str, Any]:
        """Get morphological patterns for language."""
        ...


class SpellChecker(Protocol):
    """Protocol for spell checking services."""

    def check_and_correct(self, text: str, language: str) -> str:
        """Check and correct spelling in text."""
        ...


# Main testable query processor class


class MultilingualQueryProcessor:
    """Fully testable query processor with dependency injection."""

    def __init__(
        self,
        config: QueryProcessingConfig,
        language_data_provider: LanguageDataProvider | None = None,
        spell_checker: SpellChecker | None = None,
        filter_config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize query processor with injectable dependencies.

        Args:
            config: Query processing configuration
            language_data_provider: Language-specific data provider (optional)
            spell_checker: Spell checking service (optional)
            filter_config: Filter generation configuration (optional)
            logger: Logger instance (optional)
        """
        self.config = config
        self.language_data_provider = language_data_provider
        self.spell_checker = spell_checker
        self.filter_config = filter_config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Cache language data if provider available
        self._stop_words: set[str] | None = None
        self._question_patterns: dict[str, list[str]] | None = None
        self._synonym_groups: dict[str, list[str]] | None = None
        self._morphological_patterns: dict[str, list[str]] | None = None

    def process_query(self, query: str, context: dict[str, Any] | None = None) -> ProcessedQuery:
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

        # Validate query length
        if len(query.strip()) < self.config.min_query_length:
            return self._create_empty_result(query)

        # Step 1: Preprocess text using pure function
        processed_query = normalize_text(
            query, normalize_case=self.config.normalize_case, normalize_quotes=True, normalize_punctuation=True
        )

        # Step 2: Spell check if enabled and available
        if self.config.enable_spell_check and self.spell_checker:
            processed_query = self.spell_checker.check_and_correct(processed_query, self.config.language)

        # Step 3: Classify query type using pure function
        question_patterns = self._get_question_patterns()
        query_type_str = classify_query_by_patterns(processed_query, question_patterns)
        query_type = QueryType(query_type_str)

        # Step 4: Extract keywords using pure function
        stop_words = self._get_stop_words()
        keywords = extract_keywords_from_text(
            processed_query,
            stop_words=stop_words,
            min_word_length=2,  # Default value since min_word_length not in validated config
            remove_stopwords=self.config.remove_stopwords,
        )

        # Step 5: Expand terms using pure function
        expanded_terms = []
        if self.config.expand_synonyms:
            synonym_groups = self._get_synonym_groups()
            expanded_terms = expand_terms_with_synonyms(
                keywords, synonym_groups=synonym_groups, max_expanded_terms=self.config.max_expanded_terms
            )

        # Step 6: Generate filters using pure function
        filters = generate_query_filters(processed_query, context, filter_config=self.filter_config)

        # Step 7: Calculate confidence using pure function
        confidence = calculate_query_confidence(
            original_query=original_query,
            processed_query=processed_query,
            keywords=keywords,
            query_type=query_type_str,
            patterns_matched=1 if query_type != QueryType.GENERAL else 0,
        )

        # Create metadata
        metadata = {
            "processing_steps": ["preprocess", "classify", "extract", "expand", "filter"],
            "language": self.config.language,
            "original_length": len(original_query),
            "processed_length": len(processed_query),
            "keyword_count": len(keywords),
            "expanded_count": len(expanded_terms),
        }

        result = ProcessedQuery(
            original=original_query,
            processed=processed_query,
            query_type=query_type,
            keywords=keywords,
            expanded_terms=expanded_terms,
            filters=filters,
            confidence=confidence,
            metadata=metadata,
        )

        self.logger.info(
            f"Processed query: '{original_query[:50]}...' "
            f"-> {len(keywords)} keywords, type: {query_type.value}, "
            f"confidence: {confidence:.2f}"
        )

        return result

    def _get_stop_words(self) -> set[str]:
        """Get stop words with caching."""
        if self._stop_words is None:
            if self.language_data_provider:
                self._stop_words = self.language_data_provider.get_stop_words(self.config.language)
            else:
                # Fallback to common stop words
                self._stop_words = {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                }

        return self._stop_words

    def _get_question_patterns(self) -> dict[str, list[str]]:
        """Get question patterns with caching."""
        if self._question_patterns is None:
            if self.language_data_provider:
                self._question_patterns = self.language_data_provider.get_question_patterns(self.config.language)
            else:
                # Fallback to basic patterns
                self._question_patterns = {
                    "factual": [r"\b(what|who|when|where|which)\b"],
                    "explanatory": [r"\b(how|why|explain)\b"],
                    "comparison": [r"\b(compare|difference|versus|vs)\b"],
                    "summarization": [r"\b(summarize|summary|overview)\b"],
                }

        return self._question_patterns

    def _get_synonym_groups(self) -> dict[str, list[str]]:
        """Get synonym groups with caching."""
        if self._synonym_groups is None:
            if self.language_data_provider:
                self._synonym_groups = self.language_data_provider.get_synonym_groups(self.config.language)
            else:
                # Fallback to empty synonyms
                self._synonym_groups = {}

        return self._synonym_groups

    def _create_empty_result(self, query: str) -> ProcessedQuery:
        """Create empty result for invalid queries."""
        return ProcessedQuery(
            original=query,
            processed="",
            query_type=QueryType.GENERAL,
            keywords=[],
            expanded_terms=[],
            filters={},
            confidence=0.0,
            metadata={"error": "Query too short", "language": self.config.language},
        )

    def _create_error_result(self, query: str, error: str) -> ProcessedQuery:
        """Create error result for failed processing."""
        return ProcessedQuery(
            original=query,
            processed=query,
            query_type=QueryType.GENERAL,
            keywords=[],
            expanded_terms=[],
            filters={},
            confidence=0.0,
            metadata={"error": error, "language": self.config.language},
        )


# Factory function for convenient creation


def create_query_processor(
    main_config: dict[str, Any], language: str = "hr", config_provider: Optional["ConfigProvider"] = None
) -> MultilingualQueryProcessor:
    """
    Create a MultilingualQueryProcessor with validated configuration.

    Args:
        main_config: Validated main configuration dictionary
        language: Language code for language-specific behavior
        config_provider: Configuration provider (optional, for language data)

    Returns:
        Configured MultilingualQueryProcessor instance
    """
    # Create configuration from validated config - no fallbacks needed
    config = QueryProcessingConfig.from_validated_config(main_config=main_config, language=language)

    # Create language data provider (can be mocked in tests)
    language_data_provider = None
    if config_provider:
        language_data_provider = ProductionLanguageDataProvider(config_provider)

    # Get filter configuration
    filter_config = {}
    if config_provider:
        try:
            language_config = config_provider.get_language_specific_config("query_filters", language)
            if "filters" not in language_config:
                raise ValueError("Missing 'filters' in language configuration")
            filter_config = language_config["filters"]
        except (KeyError, AttributeError):
            pass
    elif main_config and "query_filters" in main_config:
        filter_config = main_config["query_filters"]

    return MultilingualQueryProcessor(
        config=config, language_data_provider=language_data_provider, filter_config=filter_config
    )


# Production implementation of language data provider (optional)


class ProductionLanguageDataProvider:
    """Production implementation of LanguageDataProvider."""

    def __init__(self, config_provider: "ConfigProvider"):
        """Initialize with configuration provider."""
        self.config_provider = config_provider
        self._cache: dict[str, Any] = {}

    def get_stop_words(self, language: str) -> set[str]:
        """Get stop words for language."""
        cache_key = f"stop_words_{language}"
        if cache_key not in self._cache:
            try:
                shared_config = self.config_provider.get_language_config(language)
                # Direct access - ConfigValidator guarantees existence
                stopwords_config = shared_config["stopwords"]
                words = stopwords_config["words"]
                self._cache[cache_key] = set(words)
            except (KeyError, AttributeError):
                self._cache[cache_key] = set()

        return cast(set[str], self._cache[cache_key])

    def get_question_patterns(self, language: str) -> dict[str, list[str]]:
        """Get question classification patterns for language."""
        cache_key = f"question_patterns_{language}"
        if cache_key not in self._cache:
            try:
                shared_config = self.config_provider.get_language_config(language)
                # Direct access - ConfigValidator guarantees existence
                self._cache[cache_key] = shared_config["question_patterns"]
            except (KeyError, AttributeError):
                self._cache[cache_key] = {}

        return cast(dict[str, list[str]], self._cache[cache_key])

    def get_synonym_groups(self, language: str) -> dict[str, list[str]]:
        """Get synonym groups for language."""
        cache_key = f"synonym_groups_{language}"
        if cache_key not in self._cache:
            try:
                shared_config = self.config_provider.get_language_config(language)
                # Direct access - ConfigValidator guarantees existence if synonym_groups exists in config
                if "synonym_groups" not in shared_config:
                    raise ValueError("Missing 'synonym_groups' in shared configuration")
                self._cache[cache_key] = shared_config["synonym_groups"]
            except (KeyError, AttributeError):
                self._cache[cache_key] = {}

        return cast(dict[str, list[str]], self._cache[cache_key])

    def get_morphological_patterns(self, language: str) -> dict[str, Any]:
        """Get morphological patterns for language."""
        cache_key = f"morphological_patterns_{language}"
        if cache_key not in self._cache:
            try:
                shared_config = self.config_provider.get_language_config(language)
                # Direct access - ConfigValidator guarantees existence if morphological_patterns exists in config
                if "morphological_patterns" not in shared_config:
                    raise ValueError("Missing 'morphological_patterns' in shared configuration")
                self._cache[cache_key] = shared_config["morphological_patterns"]
            except (KeyError, AttributeError):
                self._cache[cache_key] = {}

        return cast(dict[str, Any], self._cache[cache_key])
