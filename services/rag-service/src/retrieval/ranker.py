"""
Modern, fully testable result ranking system for multilingual RAG.
Clean architecture with dependency injection and pure functions.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ..utils.config_models import RankingConfig
from ..utils.config_protocol import ConfigProvider

logger = logging.getLogger(__name__)


# ===== PURE DATA STRUCTURES =====


# Note: RankingConfig is now imported from config_models.py


@dataclass
class RankingSignal:
    """Individual ranking signal - pure data structure."""

    name: str
    score: float
    weight: float
    metadata: dict[str, Any] | None = None


@dataclass
class RankedDocument:
    """Document with ranking information - pure data structure."""

    id: str
    content: str
    metadata: dict[str, Any]
    original_score: float
    final_score: float
    rank: int
    ranking_signals: list[RankingSignal]
    ranking_metadata: dict[str, Any]


@dataclass
class LanguageFeatures:
    """Language-specific features - pure data structure."""

    importance_words: set[str]
    quality_indicators: dict[str, list[str]]  # positive/negative patterns
    cultural_patterns: list[str]
    grammar_patterns: list[str]
    type_weights: dict[str, float]


@dataclass
class ProcessedQuery:
    """Processed query information - pure data structure."""

    text: str
    keywords: list[str]
    query_type: str
    language: str
    metadata: dict[str, Any] | None = None


class LanguageProvider(Protocol):
    """Protocol for language-specific features."""

    def get_language_features(self, language: str) -> LanguageFeatures: ...

    def detect_language_content(self, text: str) -> dict[str, Any]: ...


# ===== PURE FUNCTIONS =====


# validate_ranking_config function removed - now handled by ConfigValidator and RankingConfig.from_validated_config()


def calculate_keyword_relevance_score(
    content: str, keywords: list[str], boost_unique_coverage: bool = True
) -> tuple[float, dict[str, Any]]:
    """
    Calculate keyword relevance score.
    Pure function - no side effects, deterministic output.

    Args:
        content: Document content
        keywords: List of query keywords
        boost_unique_coverage: Whether to boost unique keyword coverage

    Returns:
        Tuple of (score, metadata)
    """
    if not keywords:
        return 0.0, {"matches": 0, "unique_keywords": 0, "coverage": 0.0}

    content_lower = content.lower()
    content_words = re.findall(r"\b\w+\b", content_lower)
    content_word_count = len(content_words)

    if content_word_count == 0:
        return 0.0, {"matches": 0, "unique_keywords": 0, "coverage": 0.0}

    # Count keyword occurrences
    keyword_matches = 0
    unique_keywords_found = set()

    for keyword in keywords:
        keyword_lower = keyword.lower()
        count = content_lower.count(keyword_lower)
        if count > 0:
            keyword_matches += count
            unique_keywords_found.add(keyword_lower)

    # Calculate TF-like score
    tf_score = keyword_matches / content_word_count

    # Coverage boost
    coverage_ratio = len(unique_keywords_found) / len(keywords)

    # Combined score
    if boost_unique_coverage:
        score = min(1.0, tf_score * 100 + coverage_ratio * 0.3)
    else:
        score = min(1.0, tf_score * 100)

    metadata = {
        "matches": keyword_matches,
        "unique_keywords": len(unique_keywords_found),
        "coverage": coverage_ratio,
        "tf_score": tf_score,
    }

    return score, metadata


def calculate_content_quality_score(
    content: str,
    quality_indicators: dict[str, list[str]],
    metadata: dict[str, Any],
    structured_content_boost: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Calculate content quality score.
    Pure function - no side effects, deterministic output.

    Args:
        content: Document content
        quality_indicators: Dict with 'positive' and 'negative' indicator patterns
        metadata: Document metadata
        structured_content_boost: Whether to boost structured content

    Returns:
        Tuple of (score, metadata)
    """
    score = 0.5  # Base score
    content_lower = content.lower()

    # Check for quality indicators
    positive_matches = sum(
        len(re.findall(pattern, content_lower))
        for pattern in quality_indicators.get(
            "positive", []
        )  # Keep .get() - quality_indicators structure not validated
    )
    negative_matches = sum(
        len(re.findall(pattern, content_lower))
        for pattern in quality_indicators.get(
            "negative", []
        )  # Keep .get() - quality_indicators structure not validated
    )

    # Adjust score based on indicators
    score += min(0.3, positive_matches * 0.1)
    score -= min(0.2, negative_matches * 0.1)

    # Boost for having title
    has_title = bool(metadata.get("title"))  # Keep .get() - document metadata from external sources
    if has_title:
        score += 0.1

    # Boost for appropriate length (not too short, not too long)
    content_length = len(content)
    if 100 <= content_length <= 1000:
        score += 0.1
    elif content_length < 50:
        score -= 0.2

    # Check for structured content (lists, headers)
    has_structure = False
    if structured_content_boost:
        if re.search(r"[•\-\*]\s+\w+", content) or re.search(r"^[A-ZČĆŠŽĐ][^.]*:$", content, re.MULTILINE):
            score += 0.1
            has_structure = True

    score = max(0.0, min(1.0, score))

    result_metadata = {
        "positive_indicators": positive_matches,
        "negative_indicators": negative_matches,
        "has_title": has_title,
        "content_length": content_length,
        "has_structure": has_structure,
    }

    return score, result_metadata


def calculate_language_relevance_score(
    content: str, language: str, language_features: LanguageFeatures
) -> tuple[float, dict[str, Any]]:
    """
    Calculate language-specific relevance boost using configuration-driven approach.

    This function replaces hardcoded language logic with configurable features,
    allowing easy addition of new languages without code changes.

    Args:
        content: Document content to analyze
        language: Language code (hr, en, etc.)
        language_features: Language-specific feature patterns (for compatibility)

    Returns:
        Tuple of (score, metadata) where score is 0.0-1.0 and metadata contains
        detailed breakdown of feature detection
    """
    try:
        # Import here to avoid circular imports
        from ..utils.config_loader import get_language_ranking_features

        # Get configuration-driven language features
        ranking_features = cast(dict[str, Any], get_language_ranking_features(language))

    except Exception as e:
        # Simple fallback - return minimal score when config unavailable
        logger.warning(f"Failed to load ranking features for '{language}': {e}. Using fallback logic.")
        return 0.0, {
            "language": language,
            "features_detected": {},
            "config_driven": False,
            "fallback_reason": "Configuration system unavailable",
            "language_features_detected": False,
        }

    # Initialize scoring
    total_score = 0.0
    content_lower = content.lower()
    word_count = len(content.split())
    metadata: dict[str, Any] = {"language": language, "features_detected": {}, "config_driven": True}

    # 1. Special Characters (diacritics, etc.)
    special_chars_config = ranking_features["special_characters"]
    if special_chars_config["enabled"]:
        special_chars = special_chars_config["characters"]
        char_count = sum(content_lower.count(char) for char in special_chars)
        density_factor = special_chars_config["density_factor"]
        max_score = special_chars_config["max_score"]

        special_score = 0.0
        if char_count > 0 and len(content) > 0:
            special_score = min(max_score, char_count / len(content) * density_factor)
            total_score += special_score

        metadata["features_detected"]["special_characters"] = {
            "score": special_score,
            "count": char_count,
            "characters_found": special_chars if char_count > 0 else [],
        }

    # 2. Importance Words
    importance_config = ranking_features["importance_words"]
    if importance_config["enabled"]:
        importance_words = importance_config["words"]
        word_boost = importance_config["word_boost"]
        max_score = importance_config["max_score"]

        importance_matches = sum(1 for word in importance_words if word in content_lower)
        importance_score = min(max_score, importance_matches * word_boost)
        total_score += importance_score

        metadata["features_detected"]["importance_words"] = {
            "score": importance_score,
            "matches": importance_matches,
            "matched_words": [word for word in importance_words if word in content_lower],
        }

    # 3. Cultural Patterns
    cultural_config = ranking_features["cultural_patterns"]
    if cultural_config["enabled"]:
        cultural_patterns = cultural_config["patterns"]
        pattern_boost = cultural_config["pattern_boost"]
        max_score = cultural_config["max_score"]

        cultural_matches = sum(len(re.findall(pattern, content_lower, re.IGNORECASE)) for pattern in cultural_patterns)
        cultural_score = min(max_score, cultural_matches * pattern_boost)
        total_score += cultural_score

        metadata["features_detected"]["cultural_patterns"] = {"score": cultural_score, "matches": cultural_matches}

    # 4. Grammar Patterns
    grammar_config = ranking_features["grammar_patterns"]
    if grammar_config["enabled"]:
        grammar_patterns = grammar_config["patterns"]
        density_factor = grammar_config["density_factor"]
        max_score = grammar_config["max_score"]

        grammar_matches = sum(len(re.findall(pattern, content_lower)) for pattern in grammar_patterns)
        grammar_score = 0.0
        if word_count > 0:
            grammar_score = min(max_score, grammar_matches / word_count * density_factor)
        total_score += grammar_score

        metadata["features_detected"]["grammar_patterns"] = {"score": grammar_score, "matches": grammar_matches}

    # 5. Capitalization Patterns
    capitalization_config = ranking_features["capitalization"]
    if capitalization_config["enabled"]:
        proper_nouns = capitalization_config["proper_nouns"]
        capitalization_boost = capitalization_config["capitalization_boost"]
        max_score = capitalization_config["max_score"]

        # Check for proper noun matches
        proper_noun_matches = sum(1 for noun in proper_nouns if noun in content)

        # Analyze general capitalization patterns
        sentence_count = len(re.findall(r"[.!?]+", content))
        capital_starts = len(re.findall(r"(?:^|[.!?]\\s+)([A-Z][a-z])", content, re.MULTILINE))

        capitalization_score = 0.0
        if sentence_count > 0:
            capitalization_ratio = capital_starts / sentence_count
            capitalization_score = min(max_score, capitalization_ratio * 0.5)

        # Add proper noun bonus
        capitalization_score += min(max_score * 0.5, proper_noun_matches * capitalization_boost)
        capitalization_score = min(max_score, capitalization_score)

        total_score += capitalization_score

        metadata["features_detected"]["capitalization"] = {
            "score": capitalization_score,
            "proper_noun_matches": proper_noun_matches,
            "sentence_capitalization_ratio": (capitalization_ratio if sentence_count > 0 else 0.0),
        }

    # 6. Vocabulary Patterns
    vocabulary_config = ranking_features["vocabulary_patterns"]
    if vocabulary_config["enabled"]:
        vocab_patterns = vocabulary_config["patterns"]
        pattern_boost = vocabulary_config["pattern_boost"]
        max_score = vocabulary_config["max_score"]

        vocab_matches = sum(len(re.findall(pattern, content_lower, re.IGNORECASE)) for pattern in vocab_patterns)
        vocab_score = 0.0
        if word_count > 0:
            vocab_score = min(max_score, vocab_matches / word_count * 50)  # Scale factor for readability
        total_score += vocab_score

        metadata["features_detected"]["vocabulary_patterns"] = {"score": vocab_score, "matches": vocab_matches}

    # Normalize final score
    final_score = max(0.0, min(1.0, total_score))
    metadata["total_score"] = final_score
    metadata["language_features_detected"] = final_score > 0.1

    return final_score, metadata


def calculate_authority_score(
    metadata: dict[str, Any], type_weights: dict[str, float], authoritative_sources: list[str]
) -> tuple[float, dict[str, Any]]:
    """
    Calculate document authority score.
    Pure function - no side effects, deterministic output.

    Args:
        metadata: Document metadata
        type_weights: Weights for different document types
        authoritative_sources: List of authoritative source patterns

    Returns:
        Tuple of (score, metadata)
    """
    score = 0.5  # Base score

    # Source-based authority
    source = metadata.get("source", "").lower()  # Keep .get() - document metadata from external sources
    source_authority = 0.0
    for auth_source in authoritative_sources:
        if auth_source in source:
            if any(term in source for term in ["wikipedia", "gov.hr", "akademija"]):
                source_authority = 0.3
            elif any(term in source for term in [".edu", "sveučilište", "fakultet"]):
                source_authority = 0.2
            elif any(term in source for term in ["news", "vijesti", "novosti"]):
                source_authority = 0.1
            break

    score += source_authority

    # Document type authority
    doc_type = metadata.get(
        "content_type", metadata.get("type", "")
    )  # Keep .get() - document metadata from external sources
    type_multiplier = type_weights[doc_type] if doc_type in type_weights else 1.0
    score *= type_multiplier

    # Metadata completeness (more complete = more authoritative)
    metadata_fields = ["title", "author", "date", "source", "language"]
    completeness = sum(
        1 for metadata_field in metadata_fields if metadata.get(metadata_field)
    )  # Keep .get() - document metadata from external sources
    completeness_score = completeness / len(metadata_fields) * 0.2
    score += completeness_score

    score = max(0.0, min(1.0, score))

    result_metadata = {
        "source_authority": source_authority,
        "type_multiplier": type_multiplier,
        "metadata_completeness": completeness / len(metadata_fields),
        "source_type": "authoritative" if score > 0.7 else "standard",
    }

    return score, result_metadata


def calculate_length_appropriateness_score(
    content: str, query_type: str, optimal_ranges: dict[str, tuple[int, int]]
) -> tuple[float, dict[str, Any]]:
    """
    Calculate how appropriate content length is for query type.
    Pure function - no side effects, deterministic output.

    Args:
        content: Document content
        query_type: Type of query (factual, explanatory, etc.)
        optimal_ranges: Optimal length ranges for each query type

    Returns:
        Tuple of (score, metadata)
    """
    content_length = len(content)
    score = 0.5

    # Get optimal range for query type
    optimal_min, optimal_max = optimal_ranges[query_type] if query_type in optimal_ranges else (100, 500)

    if optimal_min <= content_length <= optimal_max:
        score = 1.0
    elif content_length < optimal_min:
        # Too short
        score = max(0.3, content_length / optimal_min)
    else:
        # Too long
        score = max(0.3, optimal_max / content_length)

    score = max(0.0, min(1.0, score))

    metadata = {
        "content_length": content_length,
        "query_type": query_type,
        "optimal_range": f"{optimal_min}-{optimal_max} chars",
        "length_category": (
            "optimal"
            if optimal_min <= content_length <= optimal_max
            else "too_short"
            if content_length < optimal_min
            else "too_long"
        ),
    }

    return score, metadata


def calculate_query_type_match_score(
    content: str, query_type: str, type_patterns: dict[str, list[str]]
) -> tuple[float, dict[str, Any]]:
    """
    Calculate how well content matches the query type.
    Pure function - no side effects, deterministic output.

    Args:
        content: Document content
        query_type: Type of query
        type_patterns: Patterns that indicate content matches query type

    Returns:
        Tuple of (score, metadata)
    """
    content_lower = content.lower()
    score = 0.5  # Base score

    # Check type-specific patterns
    patterns = type_patterns[query_type] if query_type in type_patterns else []
    pattern_matches = sum(len(re.findall(pattern, content_lower)) for pattern in patterns)
    pattern_score = min(0.3, pattern_matches * 0.1)
    score += pattern_score

    # Check for structural indicators
    structural_score = 0.0
    if query_type == "comparison":
        # Look for comparative structures
        if re.search(r"[•\-]\s*\w+.*vs.*\w+", content, re.IGNORECASE):
            structural_score = 0.2
    elif query_type == "explanatory":
        # Look for step-by-step or process indicators
        if re.search(r"\b\d+\.\s+\w+", content) or re.search(r"prvo|drugo|treće", content_lower):
            structural_score = 0.2

    score += structural_score
    score = max(0.0, min(1.0, score))

    metadata = {
        "query_type": query_type,
        "pattern_matches": pattern_matches,
        "structural_indicators": structural_score > 0,
    }

    return score, metadata


def combine_ranking_signals(signals: list[RankingSignal]) -> float:
    """
    Combine multiple ranking signals into final score.
    Pure function - no side effects, deterministic output.

    Args:
        signals: List of ranking signals

    Returns:
        Final combined score (0.0 to 1.0)
    """
    if not signals:
        return 0.0

    # Weighted average
    total_weighted_score = sum(signal.score * signal.weight for signal in signals)
    total_weight = sum(signal.weight for signal in signals)

    if total_weight == 0:
        return 0.0

    final_score = total_weighted_score / total_weight

    # Apply normalization to keep scores in reasonable range
    return max(0.0, min(1.0, final_score))


def apply_diversity_filtering(
    ranked_docs: list[RankedDocument], diversity_threshold: float = 0.8, min_results: int = 2
) -> list[RankedDocument]:
    """
    Apply diversity filtering to avoid too similar results.
    Pure function - no side effects, deterministic output.

    Args:
        ranked_docs: List of ranked documents
        diversity_threshold: Similarity threshold for filtering
        min_results: Minimum number of results to keep

    Returns:
        Filtered list with improved diversity
    """
    if len(ranked_docs) <= min_results:
        return ranked_docs

    diverse_docs: list[RankedDocument] = []
    used_content_hashes: set[frozenset[str]] = set()

    for doc in ranked_docs:
        # Create a simple content hash for similarity detection
        content_words = set(re.findall(r"\b\w+\b", doc.content.lower()))

        # Check similarity with already selected documents
        is_similar = False
        for existing_hash in used_content_hashes:
            if len(content_words) == 0 or len(existing_hash) == 0:
                continue
            similarity = len(content_words & existing_hash) / len(content_words | existing_hash)
            if similarity > diversity_threshold:
                is_similar = True
                break

        if not is_similar or len(diverse_docs) < min_results:  # Always keep minimum
            diverse_docs.append(doc)
            used_content_hashes.add(frozenset[str](content_words))

    return diverse_docs


def create_ranking_explanation(ranked_doc: RankedDocument) -> str:
    """
    Generate human-readable explanation of document ranking.
    Pure function - no side effects, deterministic output.

    Args:
        ranked_doc: Ranked document to explain

    Returns:
        Explanation string
    """
    lines = []
    lines.append(f"🏆 Rank #{ranked_doc.rank} (Score: {ranked_doc.final_score:.3f})")
    lines.append(f"📄 Document: {ranked_doc.id}")
    lines.append("")
    lines.append("📊 Ranking Signals:")

    # Sort signals by contribution
    signals_by_contribution = sorted(ranked_doc.ranking_signals, key=lambda s: s.score * s.weight, reverse=True)

    for signal in signals_by_contribution:
        contribution = signal.score * signal.weight
        bar_length = int(contribution * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        lines.append(f"   {signal.name:<20}: {contribution:.3f} {bar}")

    lines.append("")
    lines.append(f"📈 Original search score: {ranked_doc.original_score:.3f}")
    lines.append(f"🎯 Final ranking score: {ranked_doc.final_score:.3f}")

    return "\n".join(lines)


# ===== MAIN RANKER CLASS =====


class DocumentRanker:
    """Document ranker with dependency injection for testability."""

    def __init__(self, config_provider: ConfigProvider, language_provider: LanguageProvider, language: str):
        """Initialize with all dependencies injected."""
        self.language = language
        self.config_provider = config_provider
        self.language_provider = language_provider
        self.logger = logging.getLogger(__name__)

        # Load configuration through providers
        self.config = self._load_configuration()
        self.language_features = self.language_provider.get_language_features(language)

    def _load_configuration(self) -> RankingConfig:
        """Load ranking configuration through provider with validated config."""
        # Get full main config from provider to access ranking section
        main_config = self.config_provider.load_config('config')
        return RankingConfig.from_validated_config(main_config)

    def rank_documents(
        self, documents: list[dict[str, Any]], query: ProcessedQuery, context: dict[str, Any] | None = None
    ) -> list[RankedDocument]:
        """
        Rank and re-order documents for better relevance.

        Args:
            documents: List of retrieved documents
            query: Processed query information
            context: Additional context for ranking

        Returns:
            List of RankedDocument objects, ranked by relevance
        """
        if not documents:
            return []

        context = context or {}

        self.logger.info(f"Ranking {len(documents)} documents using {self.config.method.value}")

        # Step 1: Calculate ranking signals for each document
        ranked_docs: list[RankedDocument] = []
        for document in documents:
            ranked_doc = self._rank_single_document(document, query, context)
            ranked_docs.append(ranked_doc)

        # Step 2: Apply diversity filtering if enabled
        if self.config.enable_diversity:
            ranked_docs = apply_diversity_filtering(ranked_docs, self.config.diversity_threshold)

        # Step 3: Final sort by score
        ranked_docs.sort(key=lambda x: x.final_score, reverse=True)

        # Step 4: Update ranks
        for i, doc in enumerate(ranked_docs):
            doc.rank = i + 1

        self.logger.info(f"Ranking complete: {len(ranked_docs)} documents ranked")
        return ranked_docs

    def _rank_single_document(
        self, document: dict[str, Any], query: ProcessedQuery, context: dict[str, Any]
    ) -> RankedDocument:
        """Calculate ranking signals for a single document."""
        content = document.get("content", "")  # Keep .get() - document data from external sources
        metadata = document.get("metadata", {})  # Keep .get() - document data from external sources
        original_score = document.get("relevance_score", 0.0)  # Keep .get() - document data from external sources

        ranking_signals = []

        # Signal 1: Semantic similarity (from original search)
        semantic_signal = RankingSignal(
            name="semantic_similarity", score=original_score, weight=0.3, metadata={"source": "vector_search"}
        )
        ranking_signals.append(semantic_signal)

        # Signal 2: Keyword relevance
        keyword_score, keyword_metadata = calculate_keyword_relevance_score(content, query.keywords)
        keyword_signal = RankingSignal(
            name="keyword_relevance", score=keyword_score, weight=0.25, metadata=keyword_metadata
        )
        ranking_signals.append(keyword_signal)

        # Signal 3: Content quality
        quality_score, quality_metadata = calculate_content_quality_score(
            content, self.language_features.quality_indicators, metadata
        )
        quality_signal = RankingSignal(
            name="content_quality", score=quality_score, weight=0.15, metadata=quality_metadata
        )
        ranking_signals.append(quality_signal)

        # Signal 4: Language-specific features
        if self.config.language_specific_boost:
            lang_score, lang_metadata = calculate_language_relevance_score(
                content, self.language, self.language_features
            )
            lang_signal = RankingSignal(
                name=f"{self.language}_relevance", score=lang_score, weight=0.2, metadata=lang_metadata
            )
            ranking_signals.append(lang_signal)

        # Signal 5: Document authority
        if self.config.boost_authoritative:
            auth_score, auth_metadata = calculate_authority_score(
                metadata,
                self.language_features.type_weights,
                ["wikipedia", "gov.hr", "akademija", ".edu", "sveučilište", "fakultet"],
            )
            auth_signal = RankingSignal(name="authority_score", score=auth_score, weight=0.1, metadata=auth_metadata)
            ranking_signals.append(auth_signal)

        # Calculate final score
        final_score = combine_ranking_signals(ranking_signals)

        # Create ranking metadata
        ranking_metadata = {
            "ranking_method": self.config.method.value,
            "signal_count": len(ranking_signals),
            "top_signals": sorted(
                [(s.name, s.score * s.weight) for s in ranking_signals], key=lambda x: x[1], reverse=True
            )[:3],
        }

        return RankedDocument(
            id=document.get("id", "unknown"),  # Keep .get() - document data from external sources
            content=content,
            metadata=metadata,
            original_score=original_score,
            final_score=final_score,
            rank=0,  # Will be set later
            ranking_signals=ranking_signals,
            ranking_metadata=ranking_metadata,
        )

    def explain_ranking(self, ranked_doc: RankedDocument) -> str:
        """Generate human-readable explanation of document ranking."""
        return create_ranking_explanation(ranked_doc)


# ===== FACTORY FUNCTIONS =====


def create_document_ranker(
    language: str, config_provider: ConfigProvider | None = None, language_provider: LanguageProvider | None = None
) -> DocumentRanker:
    """
    Factory function to create document ranker.

    Args:
        language: Language code for ranking
        config_provider: Configuration provider (uses default if None)
        language_provider: Language provider (uses default if None)

    Returns:
        Configured DocumentRanker
    """
    # Import here to avoid circular imports
    from .ranker_providers import create_config_provider, create_language_provider

    if config_provider is None:
        config_provider = create_config_provider()

    if language_provider is None:
        language_provider = create_language_provider()

    return DocumentRanker(config_provider, language_provider, language)


def create_mock_ranker(language: str = "hr", config_dict: dict[str, Any] | None = None) -> DocumentRanker:
    """
    Factory function to create mock ranker for testing.

    Args:
        language: Language code
        config_dict: Optional configuration override

    Returns:
        DocumentRanker with mock providers
    """
    # Import here to avoid circular imports
    from .ranker_providers import create_mock_config_provider, create_mock_language_provider

    config_provider = create_mock_config_provider(config_dict or {})
    language_provider = create_mock_language_provider()

    return DocumentRanker(config_provider, language_provider, language)
