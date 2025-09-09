"""
Enhanced Document Categorization System with Hierarchical Router Pattern.

This module implements smart query classification and category-specific retrieval
strategies for the multilingual RAG system, supporting Croatian and English content
with cultural context awareness.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.config_loader import get_language_specific_config


class DocumentCategory(Enum):
    """Enhanced document categories for hierarchical retrieval."""

    # Primary Categories (Research-based)
    CULTURAL = "cultural"  # Croatian culture, traditions, history
    TOURISM = "tourism"  # Travel, destinations, accommodations
    TECHNICAL = "technical"  # IT, engineering, technical documentation
    LEGAL = "legal"  # Laws, regulations, legal documents
    BUSINESS = "business"  # Economy, commerce, business processes
    EDUCATIONAL = "educational"  # Academic content, courses, tutorials
    NEWS = "news"  # Current events, news articles
    REFERENCE = "reference"  # Dictionaries, encyclopedias, fact sheets

    # Specialized Categories
    FAQ = "faq"  # Frequently asked questions
    PROCEDURAL = "procedural"  # Step-by-step instructions
    COMPARATIVE = "comparative"  # Comparison content (A vs B)
    HISTORICAL = "historical"  # Historical documents, archives

    # Default
    GENERAL = "general"  # Uncategorized content


class RetrievalStrategy(Enum):
    """Retrieval strategies for different content types."""

    SEMANTIC_FOCUSED = "semantic_focused"  # High semantic similarity
    KEYWORD_HYBRID = "keyword_hybrid"  # Balance semantic + keyword
    TECHNICAL_PRECISE = "technical_precise"  # Exact term matching priority
    TEMPORAL_AWARE = "temporal_aware"  # Time-sensitive content
    FAQ_OPTIMIZED = "faq_optimized"  # Question-answer pairs
    COMPARATIVE_STRUCTURED = "comparative_structured"  # Side-by-side comparisons
    CULTURAL_CONTEXT = "cultural_context"  # Cultural context aware

    DEFAULT = "default"  # Standard retrieval


@dataclass
class CategoryConfig:
    """Configuration for a document category."""

    name: str
    retrieval_strategy: RetrievalStrategy
    similarity_threshold: float
    max_results: int
    boost_keywords: List[str]
    penalty_keywords: List[str]
    cultural_weight: float = 1.0
    temporal_weight: float = 1.0
    enable_expansion: bool = True


@dataclass
class CategorizationResult:
    """Result of query categorization."""

    primary_category: DocumentCategory
    secondary_categories: List[DocumentCategory]
    confidence: float
    signals: Dict[str, float]
    suggested_strategy: RetrievalStrategy
    filters: Dict[str, Any]
    metadata: Dict[str, Any]


class EnhancedQueryCategorizer:
    """Enhanced query categorization with hierarchical routing."""

    def __init__(self, language: str):
        """Initialize categorizer with language-specific patterns."""
        self.language = language
        self.logger = logging.getLogger(__name__)

        # Load language-specific categorization patterns
        self._load_categorization_config()

        # Initialize category configurations
        self._initialize_category_configs()

        # Load multilingual keywords and patterns
        self._load_category_patterns()

    def _load_categorization_config(self) -> None:
        """Load categorization configuration from language files."""
        self.config = get_language_specific_config("categorization", self.language)

    def _initialize_category_configs(self) -> None:
        """Initialize category-specific configurations."""
        self.category_configs = {
            DocumentCategory.CULTURAL: CategoryConfig(
                name="cultural",
                retrieval_strategy=RetrievalStrategy.CULTURAL_CONTEXT,
                similarity_threshold=0.4,
                max_results=8,
                boost_keywords=self.config["cultural_indicators"],
                penalty_keywords=[],
                cultural_weight=1.5,  # Boost Croatian cultural content
                enable_expansion=True,
            ),
            DocumentCategory.TOURISM: CategoryConfig(
                name="tourism",
                retrieval_strategy=RetrievalStrategy.SEMANTIC_FOCUSED,
                similarity_threshold=0.5,
                max_results=6,
                boost_keywords=self.config["tourism_indicators"],
                penalty_keywords=[],
                cultural_weight=1.2,
                enable_expansion=True,
            ),
            DocumentCategory.TECHNICAL: CategoryConfig(
                name="technical",
                retrieval_strategy=RetrievalStrategy.TECHNICAL_PRECISE,
                similarity_threshold=0.6,
                max_results=5,
                boost_keywords=self.config["technical_indicators"],
                penalty_keywords=[],
                cultural_weight=1.0,  # Language-neutral
                enable_expansion=False,  # Preserve technical precision
            ),
            DocumentCategory.LEGAL: CategoryConfig(
                name="legal",
                retrieval_strategy=RetrievalStrategy.TECHNICAL_PRECISE,
                similarity_threshold=0.65,
                max_results=4,
                boost_keywords=self.config["legal_indicators"],
                penalty_keywords=[],
                cultural_weight=1.3,  # Croatian legal context important
                enable_expansion=False,
            ),
            DocumentCategory.BUSINESS: CategoryConfig(
                name="business",
                retrieval_strategy=RetrievalStrategy.KEYWORD_HYBRID,
                similarity_threshold=0.5,
                max_results=6,
                boost_keywords=self.config["business_indicators"],
                penalty_keywords=[],
                cultural_weight=1.1,
                enable_expansion=True,
            ),
            DocumentCategory.FAQ: CategoryConfig(
                name="faq",
                retrieval_strategy=RetrievalStrategy.FAQ_OPTIMIZED,
                similarity_threshold=0.3,  # Lower threshold for FAQ matching
                max_results=3,
                boost_keywords=self.config["faq_indicators"],
                penalty_keywords=[],
                cultural_weight=1.0,
                enable_expansion=True,
            ),
            DocumentCategory.EDUCATIONAL: CategoryConfig(
                name="educational",
                retrieval_strategy=RetrievalStrategy.SEMANTIC_FOCUSED,
                similarity_threshold=0.45,
                max_results=7,
                boost_keywords=self.config["educational_indicators"],
                penalty_keywords=[],
                cultural_weight=1.1,
                enable_expansion=True,
            ),
            DocumentCategory.NEWS: CategoryConfig(
                name="news",
                retrieval_strategy=RetrievalStrategy.TEMPORAL_AWARE,
                similarity_threshold=0.4,
                max_results=8,
                boost_keywords=self.config["news_indicators"],
                penalty_keywords=[],
                cultural_weight=1.2,
                temporal_weight=1.5,  # Recent content priority
                enable_expansion=True,
            ),
            DocumentCategory.GENERAL: CategoryConfig(
                name="general",
                retrieval_strategy=RetrievalStrategy.DEFAULT,
                similarity_threshold=0.3,
                max_results=5,
                boost_keywords=[],
                penalty_keywords=[],
                cultural_weight=1.0,
                enable_expansion=True,
            ),
        }

    def _load_category_patterns(self) -> None:
        """Load regex patterns for category detection."""
        # Load patterns from language-specific configuration
        patterns_config = get_language_specific_config("patterns", self.language)
        self.category_patterns = {
            DocumentCategory.CULTURAL: patterns_config["cultural"],
            DocumentCategory.TOURISM: patterns_config["tourism"],
            DocumentCategory.TECHNICAL: patterns_config["technical"],
            DocumentCategory.LEGAL: patterns_config["legal"],
            DocumentCategory.BUSINESS: patterns_config["business"],
            DocumentCategory.FAQ: patterns_config["faq"],
            DocumentCategory.EDUCATIONAL: patterns_config["educational"],
            DocumentCategory.NEWS: patterns_config["news"],
        }

    def categorize_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> CategorizationResult:
        """Categorize query using hierarchical pattern matching."""
        context = context or {}
        query_lower = query.lower()

        # Calculate category scores
        category_scores = {}
        detection_signals = {}

        for category, patterns in self.category_patterns.items():
            score = 0.0
            matches = []

            # Pattern matching score
            for pattern in patterns:
                match_count = len(re.findall(pattern, query_lower, re.IGNORECASE))
                if match_count > 0:
                    matches.append(pattern)
                    score += match_count * 0.3  # Each match adds to score

            # Keyword boost score
            config = self.category_configs.get(category)
            if config and config.boost_keywords:
                for keyword in config.boost_keywords:
                    if keyword.lower() in query_lower:
                        score += 0.2  # Boost for category-specific keywords

            # Context boost
            if context.get("user_category_preference") == category.value:
                score += 0.4  # Strong user preference boost

            # Croatian language and cultural context boost
            if category == DocumentCategory.CULTURAL and self.language == "hr":
                croatian_markers = ["ž", "č", "ć", "š", "đ"]
                if any(marker in query for marker in croatian_markers):
                    score += 0.2  # Boost for Croatian cultural queries

            category_scores[category] = score
            detection_signals[category.value] = {
                "pattern_matches": len(matches),
                "keyword_matches": matches,
                "base_score": score,
            }

        # Determine primary category
        if not any(score > 0.1 for score in category_scores.values()):
            primary_category = DocumentCategory.GENERAL
            confidence = 0.5
        else:
            # Sort categories by score
            sorted_categories = sorted(
                category_scores.items(), key=lambda x: x[1], reverse=True
            )
            primary_category = sorted_categories[0][0]
            confidence = min(0.95, max(0.3, sorted_categories[0][1]))

        # Determine secondary categories (score > 0.2 and not primary)
        secondary_categories = [
            cat
            for cat, score in category_scores.items()
            if score > 0.2 and cat != primary_category
        ][
            :2
        ]  # Limit to top 2 secondary categories

        # Select retrieval strategy
        primary_config = self.category_configs.get(primary_category)
        suggested_strategy = (
            primary_config.retrieval_strategy
            if primary_config
            else RetrievalStrategy.DEFAULT
        )

        # Build filters
        filters = {
            "category": primary_category.value,
            "language": self.language,
        }

        # Add category-specific filters
        if primary_config:
            if primary_config.boost_keywords:
                filters["boost_terms"] = primary_config.boost_keywords
            if primary_config.cultural_weight != 1.0:
                filters["cultural_weight"] = primary_config.cultural_weight
            if (
                hasattr(primary_config, "temporal_weight")
                and primary_config.temporal_weight != 1.0
            ):
                filters["temporal_weight"] = primary_config.temporal_weight

        # Add context filters
        if context.get("region"):
            filters["region"] = context["region"]
        if context.get("time_period"):
            filters["time_period"] = context["time_period"]

        # Build metadata
        metadata = {
            "categorization_method": "pattern_based_hierarchical",
            "language": self.language,
            "all_scores": {cat.value: score for cat, score in category_scores.items()},
            "detection_signals": detection_signals,
            "query_length": len(query),
            "query_words": len(query.split()),
            "has_question_words": any(
                word in query_lower
                for word in [
                    "što",
                    "kako",
                    "zašto",
                    "gdje",
                    "kada",
                    "tko",
                    "what",
                    "how",
                    "why",
                    "where",
                    "when",
                    "who",
                ]
            ),
        }

        self.logger.debug(
            f"Query categorized as {primary_category.value} (confidence: {confidence:.3f})"
        )

        return CategorizationResult(
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            confidence=confidence,
            signals=detection_signals,
            suggested_strategy=suggested_strategy,
            filters=filters,
            metadata=metadata,
        )

    def get_category_config(
        self, category: DocumentCategory
    ) -> Optional[CategoryConfig]:
        """Get configuration for a specific category."""
        return self.category_configs.get(category)

    def suggest_query_improvements(self, result: CategorizationResult) -> List[str]:
        """Suggest query improvements based on categorization results."""
        suggestions = []

        # Low confidence suggestions
        if result.confidence < 0.6:
            language_config = get_language_specific_config("suggestions", self.language)
            suggestions.extend(language_config["low_confidence"])

        # Category-specific suggestions
        if result.primary_category == DocumentCategory.GENERAL:
            language_config = get_language_specific_config("suggestions", self.language)
            suggestions.extend(language_config["general_category"])

        # FAQ optimization
        if result.primary_category == DocumentCategory.FAQ:
            if not result.metadata["has_question_words"]:
                language_config = get_language_specific_config(
                    "suggestions", self.language
                )
                suggestions.extend(language_config["faq_optimization"])

        return suggestions

    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query for complexity and categorization hints."""
        analysis = {
            "character_count": len(query),
            "word_count": len(query.split()),
            "sentence_count": query.count(".")
            + query.count("!")
            + query.count("?")
            + 1,
            "has_question_mark": "?" in query,
            "has_cultural_markers": any(c in query for c in "čćšžđČĆŠŽĐ")
            if self.language == "hr"
            else False,
            "category_indicators": {},
            "complexity_score": 0.0,
        }

        # Detect category indicators
        query_lower = query.lower()
        for category, patterns in self.category_patterns.items():
            indicator_count = sum(
                len(re.findall(pattern, query_lower, re.IGNORECASE))
                for pattern in patterns
            )
            if indicator_count > 0:
                analysis["category_indicators"][category.value] = indicator_count

        # Calculate complexity score
        score = 0.0

        # Length complexity
        if analysis["word_count"] > 15:
            score += 0.4
        elif analysis["word_count"] > 8:
            score += 0.2

        # Multiple categories complexity
        if len(analysis["category_indicators"]) > 1:
            score += 0.3

        # Question complexity
        if analysis["has_question_mark"]:
            score += 0.1

        # Cultural context complexity
        if analysis["has_cultural_markers"]:
            score += 0.2

        analysis["complexity_score"] = min(1.0, score)

        return analysis


def create_query_categorizer(language: str) -> EnhancedQueryCategorizer:
    """Factory function to create query categorizer."""
    return EnhancedQueryCategorizer(language=language)
