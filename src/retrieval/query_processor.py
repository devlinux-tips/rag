"""
Croatian query preprocessing for intelligent document retrieval.
Handles query analysis, expansion, and Croatian language-specific processing.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set


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

    language: str = "hr"
    expand_synonyms: bool = True
    normalize_case: bool = True
    remove_stop_words: bool = True
    min_query_length: int = 3
    max_expanded_terms: int = 10
    enable_spell_check: bool = False  # Could add later


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


class CroatianQueryProcessor:
    """Preprocessor for Croatian language queries."""

    def __init__(self, config: QueryProcessingConfig = None):
        """
        Initialize Croatian query processor.

        Args:
            config: Query processing configuration
        """
        self.config = config or QueryProcessingConfig()
        self.logger = logging.getLogger(__name__)

        # Croatian stop words
        self.croatian_stop_words = {
            "i",
            "u",
            "na",
            "za",
            "se",
            "je",
            "da",
            "su",
            "od",
            "do",
            "po",
            "sa",
            "te",
            "ta",
            "to",
            "ti",
            "tu",
            "iz",
            "ni",
            "li",
            "ga",
            "mu",
            "pa",
            "ne",
            "si",
            "me",
            "mi",
            "ih",
            "im",
            "ju",
            "jo",
            "ja",
            "ma",
            "ah",
            "oh",
            "al",
            "el",
            "ol",
            "ul",
            "ej",
            "oj",
            "aj",
            "uj",
            "uh",
        }

        # Croatian question words and their types
        self.question_patterns = {
            QueryType.FACTUAL: [
                r"\b(tko|što|kada|gdje|koji|koja|koje|koliko|čiji|čija|čije)\b",
                r"\b(who|what|when|where|which|how many|whose)\b",  # Mixed queries
            ],
            QueryType.EXPLANATORY: [
                r"\b(kako|zašto|zbog čega|objasni|opisi)\b",
                r"\b(how|why|explain|describe)\b",
            ],
            QueryType.COMPARISON: [
                r"\b(usporedi|razlika|sličnost|bolje|gore|nasuprot)\b",
                r"\b(compare|difference|similarity|better|worse|versus)\b",
            ],
            QueryType.SUMMARIZATION: [
                r"\b(sažmi|sažetak|ukratko|pregled|overview)\b",
                r"\b(summarize|summary|briefly|overview)\b",
            ],
        }

        # Croatian synonyms for query expansion
        self.synonym_groups = {
            "grad": ["mjesto", "grad", "gradić", "metropola", "centar"],
            "veliki": ["velik", "ogroman", "znatan", "značajan", "važan"],
            "glavni": ["glavni", "centralni", "primarni", "osnovni"],
            "lijep": ["lijep", "prekrasan", "krasan", "divan", "predivan"],
            "stari": ["star", "drevni", "antički", "povijesni", "tradicionalni"],
            "novi": ["nov", "moderan", "suvremeni", "sadašnji", "trenutni"],
            "poznati": ["poznat", "slavan", "čuven", "proslavljeni", "znamenit"],
            "more": ["more", "ocean", "jadran", "obala", "morsko"],
            "planina": ["planina", "brdo", "vrh", "planinský", "gorski"],
            "rijeka": ["rijeka", "potok", "voda", "vodeni tok", "riječni"],
        }

        # Croatian morphological variations
        self.morphological_patterns = {
            "zagreb": ["zagreb", "zagreba", "zagrebu", "zagrebom", "zagrebe"],
            "hrvatska": ["hrvatska", "hrvatske", "hrvatskoj", "hrvatsku", "hrvatskom"],
            "dubrovnik": ["dubrovnik", "dubrovnika", "dubrovniku", "dubrovnikom"],
            "grad": ["grad", "grada", "gradu", "gradom", "gradovi", "gradova"],
            "more": ["more", "mora", "moru", "morem", "morski", "morska", "morsko"],
        }

    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> ProcessedQuery:
        """
        Process Croatian query for retrieval.

        Args:
            query: Raw query string
            context: Optional context information

        Returns:
            ProcessedQuery object with analysis results
        """
        if not query or len(query.strip()) < self.config.min_query_length:
            return self._create_empty_result(query)

        original_query = query
        context = context or {}

        try:
            # Step 1: Basic preprocessing
            processed_query = self._preprocess_text(query)

            # Step 2: Determine query type
            query_type = self._classify_query_type(processed_query)

            # Step 3: Extract keywords
            keywords = self._extract_keywords(processed_query)

            # Step 4: Expand terms
            expanded_terms = self._expand_terms(keywords) if self.config.expand_synonyms else []

            # Step 5: Generate filters
            filters = self._generate_filters(processed_query, context)

            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(processed_query, keywords, query_type)

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
        Basic text preprocessing for Croatian queries.

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

        # Normalize Croatian quotation marks
        text = text.replace("„", '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")

        # Remove excessive punctuation
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        return text

    def _classify_query_type(self, query: str) -> QueryType:
        """
        Classify the type of query based on Croatian patterns.

        Args:
            query: Preprocessed query text

        Returns:
            Detected QueryType
        """
        query_lower = query.lower()

        # Check each query type pattern
        for query_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return query_type

        # Default to general if no specific pattern found
        return QueryType.GENERAL

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from Croatian query.

        Args:
            query: Preprocessed query text

        Returns:
            List of extracted keywords
        """
        # Split into words
        words = re.findall(r"\b\w+\b", query.lower())

        # Remove stop words if configured
        if self.config.remove_stop_words:
            words = [word for word in words if word not in self.croatian_stop_words]

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
        Expand keywords with Croatian synonyms and morphological variations.

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
        filters = {}

        # Language filter (always Croatian for our system)
        filters["language"] = "hr"

        # Geographic filters
        if re.search(
            r"\b(zagreb|dubrovnik|split|rijeka|osijek|zadar|pula)\b",
            query,
            re.IGNORECASE,
        ):
            # Could add city-specific filtering
            pass

        # Topic filters based on keywords
        if re.search(r"\b(povijest|historij|stari|drevni)\b", query, re.IGNORECASE):
            filters["topic"] = "history"
        elif re.search(r"\b(turizam|putovanje|odmor|plaža)\b", query, re.IGNORECASE):
            filters["topic"] = "tourism"
        elif re.search(r"\b(priroda|park|nacionalni|šuma|planine)\b", query, re.IGNORECASE):
            filters["topic"] = "nature"
        elif re.search(r"\b(hrana|jelo|kuhinja|restoran)\b", query, re.IGNORECASE):
            filters["topic"] = "food"
        elif re.search(r"\b(sport|nogomet|košarka|tenis)\b", query, re.IGNORECASE):
            filters["topic"] = "sports"

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

        # Boost for Croatian-specific words
        croatian_indicators = ["ž", "č", "ć", "š", "đ"]
        if any(char in query for char in croatian_indicators):
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
            filters={"language": "hr"},
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
            filters={"language": "hr"},
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
            "sentence_count": query.count(".") + query.count("!") + query.count("?") + 1,
            "has_question_mark": "?" in query,
            "has_croatian_diacritics": any(c in query for c in "čćšžđČĆŠŽĐ"),
            "complexity_score": 0.0,
        }

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

        # Croatian language complexity
        if analysis["has_croatian_diacritics"]:
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

        # Low confidence suggestions
        if processed_query.confidence < 0.5:
            if len(processed_query.keywords) < 2:
                suggestions.append("Dodajte više ključnih riječi za bolju pretragu")

            if len(processed_query.original) < 10:
                suggestions.append("Proširite upit s više detalja")

        # Query type specific suggestions
        if processed_query.query_type == QueryType.GENERAL:
            suggestions.append("Budite specifičniji - koristite 'što', 'kako', 'gdje', itd.")

        # No expanded terms
        if self.config.expand_synonyms and not processed_query.expanded_terms:
            suggestions.append("Probajte koristiti sinonime za bolje rezultate")

        # No filters applied
        if len(processed_query.filters) <= 1:  # Only language filter
            suggestions.append("Dodajte kontekst (grad, tema, vremenski period)")

        return suggestions


def create_query_processor(
    language: str = "hr", expand_synonyms: bool = True
) -> CroatianQueryProcessor:
    """
    Factory function to create query processor.

    Args:
        language: Processing language
        expand_synonyms: Whether to expand with synonyms

    Returns:
        Configured CroatianQueryProcessor
    """
    config = QueryProcessingConfig(language=language, expand_synonyms=expand_synonyms)
    return CroatianQueryProcessor(config)
