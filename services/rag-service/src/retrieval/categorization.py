"""
Pure function categorization system with dependency injection.
Clean architecture with no side effects and deterministic output.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable


class CategoryType(Enum):
    """Enumeration of query category types."""

    GENERAL = "general"
    TECHNICAL = "technical"
    CULTURAL = "cultural"
    HISTORICAL = "historical"
    ACADEMIC = "academic"
    LEGAL = "legal"
    MEDICAL = "medical"
    BUSINESS = "business"
    TOURISM = "tourism"
    EDUCATION = "education"


class QueryComplexity(Enum):
    """Enumeration of query complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ANALYTICAL = "analytical"


@dataclass
class CategoryMatch:
    """Result of category pattern matching."""

    category: CategoryType
    confidence: float
    matched_patterns: List[str]
    cultural_indicators: List[str]
    complexity: QueryComplexity
    retrieval_strategy: str


@dataclass
class CategorizationConfig:
    """Configuration for query categorization."""

    categories: Dict[str, Dict[str, Any]]
    patterns: Dict[str, List[str]]
    cultural_keywords: Dict[str, List[str]]
    complexity_thresholds: Dict[str, float]
    retrieval_strategies: Dict[str, str]


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration access to enable testing."""

    def get_categorization_config(self, language: str) -> Dict[str, Any]:
        """Get categorization configuration for specified language."""
        ...


@runtime_checkable
class LoggerProvider(Protocol):
    """Protocol for logging operations."""

    def info(self, message: str) -> None:
        """Log info message."""
        ...

    def debug(self, message: str) -> None:
        """Log debug message."""
        ...

    def warning(self, message: str) -> None:
        """Log warning message."""
        ...


# ================================
# PURE BUSINESS LOGIC FUNCTIONS
# ================================


def normalize_query_text(query: str) -> str:
    """
    Normalize query text for consistent pattern matching.
    Pure function with no side effects.

    Args:
        query: Raw query text

    Returns:
        Normalized query text
    """
    if not query:
        return ""

    # Convert to lowercase for consistent matching
    normalized = query.lower().strip()

    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized)

    # Remove punctuation for pattern matching (but preserve for cultural analysis)
    normalized = re.sub(r"[^\w\s\u0100-\u017F\u1E00-\u1EFF]", "", normalized)

    return normalized


def extract_cultural_indicators(query: str, cultural_keywords: Dict[str, List[str]]) -> List[str]:
    """
    Extract cultural indicators from query text.
    Pure function with no side effects.

    Args:
        query: Query text to analyze
        cultural_keywords: Dictionary of cultural keyword categories

    Returns:
        List of matched cultural indicators
    """
    indicators = []
    normalized_query = normalize_query_text(query)

    for category, keywords in cultural_keywords.items():
        for keyword in keywords:
            if keyword.lower() in normalized_query:
                indicators.append(f"{category}:{keyword}")

    return indicators


def calculate_query_complexity(
    query: str, complexity_thresholds: Dict[str, float]
) -> QueryComplexity:
    """
    Calculate query complexity based on various linguistic features.
    Pure function with no side effects.

    Args:
        query: Query text to analyze
        complexity_thresholds: Thresholds for complexity classification

    Returns:
        Query complexity level
    """
    if not query:
        return QueryComplexity.SIMPLE

    # Calculate complexity factors
    word_count = len(query.split())
    sentence_count = len(re.split(r"[.!?]+", query))
    avg_word_length = sum(len(word) for word in query.split()) / max(word_count, 1)

    # Check for complex linguistic patterns
    question_words = len(
        re.findall(
            r"\b(što|kako|kada|gdje|zašto|tko|koji|what|how|when|where|why|who|which)\b",
            query.lower(),
        )
    )
    conditional_words = len(re.findall(r"\b(ako|ukoliko|provided|assuming|given)\b", query.lower()))
    comparative_words = len(
        re.findall(r"\b(više|manje|bolji|gori|better|worse|more|less)\b", query.lower())
    )

    # Calculate complexity score
    complexity_score = (
        word_count * 0.1
        + sentence_count * 0.2
        + avg_word_length * 0.1
        + question_words * 0.3
        + conditional_words * 0.4
        + comparative_words * 0.3
    )

    # Apply thresholds - validate required keys
    if "analytical" not in complexity_thresholds:
        raise ValueError("Missing 'analytical' threshold in complexity_thresholds")
    if "complex" not in complexity_thresholds:
        raise ValueError("Missing 'complex' threshold in complexity_thresholds")
    if "moderate" not in complexity_thresholds:
        raise ValueError("Missing 'moderate' threshold in complexity_thresholds")

    # Apply thresholds
    if complexity_score >= complexity_thresholds["analytical"]:
        return QueryComplexity.ANALYTICAL
    elif complexity_score >= complexity_thresholds["complex"]:
        return QueryComplexity.COMPLEX
    elif complexity_score >= complexity_thresholds["moderate"]:
        return QueryComplexity.MODERATE
    else:
        return QueryComplexity.SIMPLE


def match_category_patterns(
    query: str, category_patterns: Dict[str, List[str]]
) -> List[Tuple[CategoryType, float, List[str]]]:
    """
    Match query against category patterns and calculate confidence scores.
    Pure function with no side effects.

    Args:
        query: Query text to analyze
        category_patterns: Dictionary of category patterns

    Returns:
        List of (category, confidence, matched_patterns) tuples
    """
    normalized_query = normalize_query_text(query)
    category_matches = []

    for category_name, patterns in category_patterns.items():
        matched_patterns = []
        total_matches = 0

        for pattern in patterns:
            # Convert pattern to regex for flexible matching
            regex_pattern = pattern.lower().replace("*", ".*")
            if re.search(regex_pattern, normalized_query):
                matched_patterns.append(pattern)
                total_matches += 1

        if matched_patterns:
            # Calculate confidence based on pattern matches and pattern specificity
            confidence = min(total_matches / len(patterns) * 1.2, 1.0)

            try:
                category_type = CategoryType(category_name.lower())
                category_matches.append((category_type, confidence, matched_patterns))
            except ValueError:
                # Skip invalid category names
                continue

    # Sort by confidence (highest first)
    category_matches.sort(key=lambda x: x[1], reverse=True)

    return category_matches


def determine_retrieval_strategy(
    category: CategoryType,
    complexity: QueryComplexity,
    cultural_indicators: List[str],
    strategy_config: Dict[str, str],
) -> str:
    """
    Determine optimal retrieval strategy based on categorization results.
    Pure function with no side effects.

    Args:
        category: Primary query category
        complexity: Query complexity level
        cultural_indicators: List of cultural context indicators
        strategy_config: Configuration mapping categories to strategies

    Returns:
        Optimal retrieval strategy name
    """
    # Strategy priority: specific > cultural > complexity > default

    # Check for category-specific strategy
    category_key = f"category_{category.value}"
    if category_key in strategy_config:
        return strategy_config[category_key]

    # Check for cultural context strategy
    if cultural_indicators and "cultural_context" in strategy_config:
        return strategy_config["cultural_context"]

    # Check for complexity-based strategy
    complexity_key = f"complexity_{complexity.value}"
    if complexity_key in strategy_config:
        return strategy_config[complexity_key]

    # Default strategy - validate required key
    if "default" not in strategy_config:
        raise ValueError("Missing 'default' strategy in retrieval_strategies configuration")
    return strategy_config["default"]


def categorize_query_pure(query: str, config: CategorizationConfig) -> CategoryMatch:
    """
    Categorize query using pure business logic.
    Pure function with no side effects and deterministic output.

    Args:
        query: Query text to categorize
        config: Categorization configuration

    Returns:
        CategoryMatch with categorization results
    """
    if not query or not config:
        return CategoryMatch(
            category=CategoryType.GENERAL,
            confidence=0.0,
            matched_patterns=[],
            cultural_indicators=[],
            complexity=QueryComplexity.SIMPLE,
            retrieval_strategy="default",
        )

    # Extract cultural indicators
    cultural_indicators = extract_cultural_indicators(query, config.cultural_keywords)

    # Calculate complexity
    complexity = calculate_query_complexity(query, config.complexity_thresholds)

    # Match category patterns
    category_matches = match_category_patterns(query, config.patterns)

    # Select best category match considering both confidence and priority
    if category_matches:
        # Find the match with highest priority among high-confidence matches
        best_match = None
        best_priority = -1

        for category, confidence, patterns in category_matches:
            # Get category priority from config
            category_priority = 0  # Default priority
            if category.value in config.categories:
                category_settings = config.categories[category.value]
                if "priority" in category_settings:
                    category_priority = category_settings["priority"]

            # Choose this match if it has higher priority, or same priority but higher confidence
            if category_priority > best_priority or (
                category_priority == best_priority and best_match is None
            ):
                best_match = (category, confidence, patterns)
                best_priority = category_priority

        if best_match:
            best_category, confidence, matched_patterns = best_match
        else:
            best_category, confidence, matched_patterns = category_matches[0]
    else:
        best_category = CategoryType.GENERAL
        confidence = 0.1  # Low confidence for general category
        matched_patterns = []

    # Determine retrieval strategy
    retrieval_strategy = determine_retrieval_strategy(
        best_category, complexity, cultural_indicators, config.retrieval_strategies
    )

    return CategoryMatch(
        category=best_category,
        confidence=confidence,
        matched_patterns=matched_patterns,
        cultural_indicators=cultural_indicators,
        complexity=complexity,
        retrieval_strategy=retrieval_strategy,
    )


# ================================
# DEPENDENCY INJECTION ORCHESTRATION
# ================================


class QueryCategorizer:
    """Query categorizer with dependency injection for testability."""

    def __init__(
        self,
        language: str,
        config_provider: ConfigProvider,
        logger_provider: Optional[LoggerProvider] = None,
    ):
        """Initialize categorizer with injected dependencies."""
        self.language = language
        self._config_provider = config_provider
        self._logger = logger_provider

        # Load configuration once during initialization
        config_data = config_provider.get_categorization_config(language)
        # Validate required configuration keys
        if "categories" not in config_data:
            raise ValueError("Missing 'categories' in categorization configuration")
        if "patterns" not in config_data:
            raise ValueError("Missing 'patterns' in categorization configuration")
        if "cultural_keywords" not in config_data:
            raise ValueError("Missing 'cultural_keywords' in categorization configuration")
        if "complexity_thresholds" not in config_data:
            raise ValueError("Missing 'complexity_thresholds' in categorization configuration")
        if "retrieval_strategies" not in config_data:
            raise ValueError("Missing 'retrieval_strategies' in categorization configuration")

        self._config = CategorizationConfig(
            categories=config_data["categories"],
            patterns=config_data["patterns"],
            cultural_keywords=config_data["cultural_keywords"],
            complexity_thresholds=config_data["complexity_thresholds"],
            retrieval_strategies=config_data["retrieval_strategies"],
        )

    def categorize_query(self, query: str) -> CategoryMatch:
        """
        Categorize query using dependency injection.

        Args:
            query: Query text to categorize

        Returns:
            CategoryMatch with categorization results
        """
        self._log_debug(f"Categorizing query: {query[:50]}...")

        result = categorize_query_pure(query, self._config)

        self._log_info(
            f"Query categorized as {result.category.value} "
            f"(confidence: {result.confidence:.2f}, strategy: {result.retrieval_strategy})"
        )

        return result

    def _log_info(self, message: str) -> None:
        """Log info message if logger available."""
        if self._logger:
            self._logger.info(message)

    def _log_debug(self, message: str) -> None:
        """Log debug message if logger available."""
        if self._logger:
            self._logger.debug(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message if logger available."""
        if self._logger:
            self._logger.warning(message)


# ================================
# CONVENIENCE FUNCTIONS (Backward Compatibility)
# ================================


def categorize_query(
    query: str, language: str, config_provider: Optional[ConfigProvider] = None
) -> CategoryMatch:
    """
    Convenience function for backward compatibility.

    Args:
        query: Query text to categorize
        language: Language for categorization
        config_provider: Optional config provider (uses production if None)

    Returns:
        CategoryMatch with categorization results
    """
    from .categorization_providers import create_config_provider

    # Use injected provider or create default
    config = config_provider or create_config_provider()

    categorizer = QueryCategorizer(language, config)
    return categorizer.categorize_query(query)


# ================================
# BACKWARD COMPATIBILITY ALIASES
# ================================


# Legacy class name for compatibility
# Removed simple alias - full implementation below at line 530


# Legacy aliases for classes that other files are importing
class DocumentCategory(Enum):
    """Legacy enum for backward compatibility with hierarchical_retriever.py and other files."""

    CULTURAL = "cultural"
    TOURISM = "tourism"
    TECHNICAL = "technical"
    LEGAL = "legal"
    BUSINESS = "business"
    EDUCATIONAL = "educational"
    NEWS = "news"
    REFERENCE = "reference"
    FAQ = "faq"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    HISTORICAL = "historical"
    GENERAL = "general"


class RetrievalStrategy(Enum):
    """Legacy enum for backward compatibility."""

    SEMANTIC_FOCUSED = "semantic_focused"
    KEYWORD_HYBRID = "keyword_hybrid"
    TECHNICAL_PRECISE = "technical_precise"
    TEMPORAL_AWARE = "temporal_aware"
    FAQ_OPTIMIZED = "faq_optimized"
    COMPARATIVE_STRUCTURED = "comparative_structured"
    CULTURAL_CONTEXT = "cultural_context"
    DEFAULT = "default"


@dataclass
class CategorizationResult:
    """Legacy dataclass for backward compatibility."""

    primary_category: DocumentCategory
    secondary_categories: List[DocumentCategory]
    confidence: float
    signals: Dict[str, float]
    suggested_strategy: RetrievalStrategy
    filters: Dict[str, Any]
    metadata: Dict[str, Any]


# Legacy class alias - the main backward compatibility class
class EnhancedQueryCategorizer(QueryCategorizer):
    """
    Legacy class for backward compatibility.
    Maps old interface to new dependency injection interface.
    """

    def __init__(self, language: str):
        """Initialize with legacy interface pattern."""
        from .categorization_providers import create_config_provider

        # Use production config provider by default
        config_provider = create_config_provider(use_mock=False)

        # Initialize parent with dependency injection
        super().__init__(language, config_provider, None)

    def categorize(self, query: str) -> CategorizationResult:
        """
        Legacy categorize method that returns old CategorizationResult format.
        Maps new CategoryMatch to old CategorizationResult.
        """
        # Use new categorization method
        modern_result = self.categorize_query(query)

        # Map to legacy format
        legacy_primary = DocumentCategory.GENERAL
        try:
            # Map new CategoryType to legacy DocumentCategory
            category_mapping = {
                CategoryType.GENERAL: DocumentCategory.GENERAL,
                CategoryType.TECHNICAL: DocumentCategory.TECHNICAL,
                CategoryType.CULTURAL: DocumentCategory.CULTURAL,
                CategoryType.HISTORICAL: DocumentCategory.HISTORICAL,
                CategoryType.ACADEMIC: DocumentCategory.EDUCATIONAL,
                CategoryType.LEGAL: DocumentCategory.LEGAL,
                CategoryType.MEDICAL: DocumentCategory.REFERENCE,  # Map medical to reference
                CategoryType.BUSINESS: DocumentCategory.BUSINESS,
                CategoryType.TOURISM: DocumentCategory.TOURISM,
                CategoryType.EDUCATION: DocumentCategory.EDUCATIONAL,
            }
            if modern_result.category not in category_mapping:
                raise ValueError(f"Missing category mapping for '{modern_result.category}'")
            legacy_primary = category_mapping[modern_result.category]
        except (KeyError, AttributeError):
            legacy_primary = DocumentCategory.GENERAL

        # Map retrieval strategy
        legacy_strategy = RetrievalStrategy.DEFAULT
        try:
            strategy_mapping = {
                "hybrid": RetrievalStrategy.KEYWORD_HYBRID,
                "dense": RetrievalStrategy.SEMANTIC_FOCUSED,
                "cultural_context": RetrievalStrategy.CULTURAL_CONTEXT,
                "cultural_aware": RetrievalStrategy.CULTURAL_CONTEXT,
                "sparse": RetrievalStrategy.TECHNICAL_PRECISE,
                "hierarchical": RetrievalStrategy.COMPARATIVE_STRUCTURED,
                "default": RetrievalStrategy.DEFAULT,
            }
            if modern_result.retrieval_strategy not in strategy_mapping:
                raise ValueError(
                    f"Missing strategy mapping for '{modern_result.retrieval_strategy}'"
                )
            legacy_strategy = strategy_mapping[modern_result.retrieval_strategy]
        except (KeyError, AttributeError):
            legacy_strategy = RetrievalStrategy.DEFAULT

        # Create legacy result
        return CategorizationResult(
            primary_category=legacy_primary,
            secondary_categories=[],  # New system doesn't have secondary categories
            confidence=modern_result.confidence,
            signals={
                "complexity": modern_result.complexity.value,
                "cultural_indicators": len(modern_result.cultural_indicators),
                "matched_patterns": len(modern_result.matched_patterns),
            },
            suggested_strategy=legacy_strategy,
            filters={},
            metadata={
                "matched_patterns": modern_result.matched_patterns,
                "cultural_indicators": modern_result.cultural_indicators,
                "complexity": modern_result.complexity.value,
            },
        )
