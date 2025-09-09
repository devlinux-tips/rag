"""
Result ranking and filtering for multilingual RAG system.
Advanced ranking algorithms to improve retrieval quality.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .query_processor import ProcessedQuery, QueryType


class RankingMethod(Enum):
    """Available ranking methods."""

    BM25 = "bm25"  # Best Matching 25 algorithm
    TF_IDF = "tf_idf"  # Term Frequency - Inverse Document Frequency
    SEMANTIC = "semantic"  # Pure semantic similarity
    HYBRID = "hybrid"  # Combine multiple signals
    LANGUAGE_ENHANCED = "language_enhanced"  # Language-specific enhancements


@dataclass
class RankingConfig:
    """Configuration for result ranking."""

    method: RankingMethod = RankingMethod.LANGUAGE_ENHANCED
    enable_diversity: bool = True
    diversity_threshold: float = 0.8
    boost_recent: bool = False
    boost_authoritative: bool = True
    content_length_factor: bool = True
    keyword_density_factor: bool = True
    language_specific_boost: bool = True

    @classmethod
    def from_config(
        cls, config_dict: Optional[Dict[str, Any]] = None
    ) -> "RankingConfig":
        """Create config from dictionary with DRY error handling."""
        from ..utils.config_loader import get_ranking_config
        from ..utils.error_handler import handle_config_error

        # Load config with fallback
        ranking_config = config_dict or handle_config_error(
            operation=get_ranking_config,
            fallback_value={},
            config_file="config/config.toml",
            section="ranking",
        )

        # Handle method enum conversion
        method_str = ranking_config.get("method", "language_enhanced")
        try:
            method = RankingMethod(method_str)
        except ValueError:
            method = RankingMethod.LANGUAGE_ENHANCED

        return cls(
            method=method,
            enable_diversity=ranking_config.get("enable_diversity", True),
            diversity_threshold=ranking_config.get("diversity_threshold", 0.8),
            boost_recent=ranking_config.get("boost_recent", False),
            boost_authoritative=ranking_config.get("boost_authoritative", True),
            content_length_factor=ranking_config.get("content_length_factor", True),
            keyword_density_factor=ranking_config.get("keyword_density_factor", True),
            language_specific_boost=ranking_config.get("language_specific_boost", True),
        )


@dataclass
class RankingSignal:
    """Individual ranking signal."""

    name: str
    score: float
    weight: float
    metadata: Dict[str, Any] = None


@dataclass
class RankedDocument:
    """Document with ranking information."""

    id: str
    content: str
    metadata: Dict[str, Any]
    original_score: float
    final_score: float
    rank: int
    ranking_signals: List[RankingSignal]
    ranking_metadata: Dict[str, Any]


class ResultRanker:
    """Advanced result ranker for multilingual content."""

    def __init__(self, config: RankingConfig = None, language: str = "hr"):
        """
        Initialize result ranker.

        Args:
            config: Ranking configuration
            language: Language code for ranking
        """
        from ..utils.config_loader import get_language_specific_config
        from ..utils.error_handler import handle_config_error

        self.language = language
        self.config = config or RankingConfig.from_config()
        self.logger = logging.getLogger(__name__)

        # Load language-specific settings
        language_config = handle_config_error(
            operation=lambda: get_language_specific_config("retrieval", self.language),
            fallback_value={"morphology": {}},
            config_file=f"config/{self.language}.toml",
            section=f"{self.language} retrieval",
        )

        # Language-specific features from config
        morphology = language_config.get("morphology", {})
        # Language-specific importance words from config or defaults
        if self.language == "hr":
            default_words = [
                "zagreb",
                "hrvatska",
                "dubrovnik",
                "split",
                "rijeka",
                "osijek",
                "glavni",
                "važan",
                "značajan",
                "poznati",
                "tradicionalni",
                "historijski",
                "kulturni",
                "turistički",
                "nacionalni",
            ]
        elif self.language == "en":
            default_words = [
                "important",
                "significant",
                "major",
                "primary",
                "essential",
                "key",
                "main",
                "crucial",
                "critical",
                "fundamental",
                "notable",
                "prominent",
                "leading",
                "advanced",
                "innovative",
            ]
        else:
            default_words = []

        # Combine configured morphological variations
        self.language_importance_words = set(default_words)
        for word_group in morphology.values():
            if isinstance(word_group, list):
                self.language_importance_words.update(word_group)

        # Content quality indicators (language-specific)
        if self.language == "hr":
            self.quality_indicators = {
                "positive": [
                    r"\b(detaljno|sveobuhvatno|temeljito|precizno)\b",
                    r"\b(službeno|autoritetno|provjereno|pouzdano)\b",
                    r"\b(suvremeno|aktualno|novo|nedavno)\b",
                ],
                "negative": [
                    r"\b(možda|vjerojatno|nejasno|približno)\b",
                    r"\b(staro|zastarjelo|neprovjereno|sumnjivo)\b",
                    r"\b(kratko|površno|nepotpuno|fragmentarno)\b",
                ],
            }
        elif self.language == "en":
            self.quality_indicators = {
                "positive": [
                    r"\b(detailed|comprehensive|thorough|precise)\b",
                    r"\b(official|authoritative|verified|reliable)\b",
                    r"\b(current|recent|new|updated)\b",
                ],
                "negative": [
                    r"\b(maybe|probably|unclear|approximately)\b",
                    r"\b(old|outdated|unverified|questionable)\b",
                    r"\b(brief|superficial|incomplete|fragmentary)\b",
                ],
            }
        else:
            self.quality_indicators = {
                "positive": [],
                "negative": [],
            }

        # Document type preferences
        self.type_weights = {
            "encyclopedia": 1.2,
            "academic": 1.1,
            "news": 1.0,
            "blog": 0.9,
            "forum": 0.8,
            "social": 0.7,
        }

    def rank_documents(
        self,
        documents: List[Dict[str, Any]],
        query: ProcessedQuery,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RankedDocument]:
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

        try:
            self.logger.info(
                f"Ranking {len(documents)} documents using {self.config.method.value}"
            )

            # Step 1: Calculate ranking signals for each document
            ranked_docs = []
            for doc in documents:
                ranked_doc = self._rank_single_document(doc, query, documents, context)
                ranked_docs.append(ranked_doc)

            # Step 2: Apply diversity filtering if enabled
            if self.config.enable_diversity:
                ranked_docs = self._apply_diversity_filtering(ranked_docs)

            # Step 3: Final sort by score
            ranked_docs.sort(key=lambda x: x.final_score, reverse=True)

            # Step 4: Update ranks
            for i, doc in enumerate(ranked_docs):
                doc.rank = i + 1

            self.logger.info(f"Ranking complete: {len(ranked_docs)} documents ranked")

            return ranked_docs

        except Exception as e:
            self.logger.error(f"Ranking failed: {e}")
            # Fallback: return original order
            return self._create_fallback_ranking(documents)

    def _rank_single_document(
        self,
        document: Dict[str, Any],
        query: ProcessedQuery,
        all_documents: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> RankedDocument:
        """
        Calculate ranking signals for a single document.

        Args:
            document: Document to rank
            query: Processed query
            all_documents: All documents for corpus statistics
            context: Additional context

        Returns:
            RankedDocument with calculated scores
        """
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        original_score = document.get("relevance_score", 0.0)

        ranking_signals = []

        # Signal 1: Semantic similarity (from original search)
        semantic_signal = RankingSignal(
            name="semantic_similarity",
            score=original_score,
            weight=0.3,
            metadata={"source": "vector_search"},
        )
        ranking_signals.append(semantic_signal)

        # Signal 2: Keyword relevance
        keyword_signal = self._calculate_keyword_relevance(content, query)
        ranking_signals.append(keyword_signal)

        # Signal 3: Content quality
        quality_signal = self._calculate_content_quality(content, metadata)
        ranking_signals.append(quality_signal)

        # Signal 4: Language-specific features
        if self.config.language_specific_boost:
            language_signal = self._calculate_language_relevance(content, query)
            ranking_signals.append(language_signal)

        # Signal 5: Document authority
        if self.config.boost_authoritative:
            authority_signal = self._calculate_authority_score(metadata)
            ranking_signals.append(authority_signal)

        # Signal 6: Content length appropriateness
        if self.config.content_length_factor:
            length_signal = self._calculate_length_appropriateness(
                content, query.query_type
            )
            ranking_signals.append(length_signal)

        # Signal 7: Query type matching
        query_match_signal = self._calculate_query_type_match(content, query.query_type)
        ranking_signals.append(query_match_signal)

        # Calculate final score
        final_score = self._combine_ranking_signals(ranking_signals)

        # Create ranking metadata
        ranking_metadata = {
            "ranking_method": self.config.method.value,
            "signal_count": len(ranking_signals),
            "top_signals": sorted(
                [(s.name, s.score * s.weight) for s in ranking_signals],
                key=lambda x: x[1],
                reverse=True,
            )[:3],
        }

        return RankedDocument(
            id=document.get("id", "unknown"),
            content=content,
            metadata=metadata,
            original_score=original_score,
            final_score=final_score,
            rank=0,  # Will be set later
            ranking_signals=ranking_signals,
            ranking_metadata=ranking_metadata,
        )

    def _calculate_keyword_relevance(
        self, content: str, query: ProcessedQuery
    ) -> RankingSignal:
        """
        Calculate keyword relevance score.

        Args:
            content: Document content
            query: Processed query

        Returns:
            Keyword relevance ranking signal
        """
        if not query.keywords:
            return RankingSignal("keyword_relevance", 0.0, 0.2)

        content_lower = content.lower()
        content_words = re.findall(r"\b\w+\b", content_lower)
        content_word_count = len(content_words)

        if content_word_count == 0:
            return RankingSignal("keyword_relevance", 0.0, 0.2)

        # Count keyword occurrences
        keyword_matches = 0
        unique_keywords_found = set()

        for keyword in query.keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            if count > 0:
                keyword_matches += count
                unique_keywords_found.add(keyword_lower)

        # Calculate TF-like score
        tf_score = keyword_matches / content_word_count

        # Boost for unique keyword coverage
        coverage_boost = len(unique_keywords_found) / len(query.keywords)

        # Combined score
        score = min(1.0, tf_score * 100 + coverage_boost * 0.3)

        return RankingSignal(
            name="keyword_relevance",
            score=score,
            weight=0.25,
            metadata={
                "matches": keyword_matches,
                "unique_keywords": len(unique_keywords_found),
                "coverage": coverage_boost,
            },
        )

    def _calculate_content_quality(
        self, content: str, metadata: Dict[str, Any]
    ) -> RankingSignal:
        """
        Calculate content quality score.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            Content quality ranking signal
        """
        score = 0.5  # Base score

        content_lower = content.lower()

        # Check for quality indicators
        positive_matches = sum(
            len(re.findall(pattern, content_lower))
            for pattern in self.quality_indicators["positive"]
        )
        negative_matches = sum(
            len(re.findall(pattern, content_lower))
            for pattern in self.quality_indicators["negative"]
        )

        # Adjust score based on indicators
        score += min(0.3, positive_matches * 0.1)
        score -= min(0.2, negative_matches * 0.1)

        # Boost for having title
        if metadata.get("title"):
            score += 0.1

        # Boost for appropriate length (not too short, not too long)
        content_length = len(content)
        if 100 <= content_length <= 1000:
            score += 0.1
        elif content_length < 50:
            score -= 0.2

        # Check for structured content (lists, headers)
        if re.search(r"[•\-\*]\s+\w+", content) or re.search(
            r"^[A-ZČĆŠŽĐ][^.]*:$", content, re.MULTILINE
        ):
            score += 0.1

        score = max(0.0, min(1.0, score))

        return RankingSignal(
            name="content_quality",
            score=score,
            weight=0.15,
            metadata={
                "positive_indicators": positive_matches,
                "negative_indicators": negative_matches,
                "has_title": bool(metadata.get("title")),
                "content_length": content_length,
            },
        )

    def _calculate_language_relevance(
        self, content: str, query: ProcessedQuery
    ) -> RankingSignal:
        """
        Calculate language-specific relevance boost.

        Args:
            content: Document content
            query: Processed query

        Returns:
            Language relevance ranking signal
        """
        score = 0.0
        content_lower = content.lower()

        if self.language == "hr":
            # Croatian-specific features
            # Boost for Croatian diacritics (indicates proper Croatian content)
            croatian_chars = "čćšžđ"
            diacritic_count = sum(content_lower.count(char) for char in croatian_chars)
            if diacritic_count > 0:
                score += min(0.3, diacritic_count / len(content) * 1000)

            # Boost for Croatian importance words
            importance_matches = sum(
                1 for word in self.language_importance_words if word in content_lower
            )
            score += min(0.4, importance_matches * 0.1)

            # Boost for Croatian cultural references
            cultural_patterns = [
                r"\b(biser jadrana|perla jadrana)\b",
                r"\b(hrvatski?\w* kralj|hrvatska povijest)\b",
                r"\b(adriatic|jadransko more)\b",
                r"\b(unesco|svjetska baština)\b",
            ]

            cultural_matches = sum(
                len(re.findall(pattern, content_lower, re.IGNORECASE))
                for pattern in cultural_patterns
            )
            score += min(0.2, cultural_matches * 0.1)

            # Boost for proper Croatian grammar indicators
            grammar_patterns = [
                r"\b\w+ić\b",  # Common Croatian surname ending
                r"\b\w+ović\b",  # Common Croatian surname ending
                r"\b\w+ski\b",  # Croatian adjective ending
                r"\b\w+nja\b",  # Croatian place name ending
            ]

            grammar_matches = sum(
                len(re.findall(pattern, content_lower)) for pattern in grammar_patterns
            )
            score += min(0.1, grammar_matches / len(content.split()) * 10)

        elif self.language == "en":
            # English-specific features
            # Boost for proper English capitalization patterns
            sentence_count = len(re.findall(r"[.!?]+", content))
            capital_starts = len(
                re.findall(r"(?:^|[.!?]\s+)([A-Z][a-z])", content, re.MULTILINE)
            )
            if sentence_count > 0:
                capitalization_ratio = capital_starts / sentence_count
                score += min(0.2, capitalization_ratio * 0.5)

            # Boost for English-specific vocabulary indicators
            english_indicators = [
                r"\b(the|and|that|have|for|not|with|you|this|but|his|from|they)\b",
                r"\b(United States|UK|Britain|England|American|British)\b",
                r"\b(technology|science|research|development|innovation)\b",
            ]

            indicator_matches = sum(
                len(re.findall(pattern, content_lower, re.IGNORECASE))
                for pattern in english_indicators
            )
            score += min(0.4, indicator_matches / len(content.split()) * 5)

            # Boost for proper English grammar patterns
            grammar_patterns = [
                r"\b(ing\b)",  # Common English gerund ending
                r"\b\w+ly\b",  # Common English adverb ending
                r"\b\w+tion\b",  # Common English noun ending
                r"\b\w+ness\b",  # Common English noun ending
            ]

            grammar_matches = sum(
                len(re.findall(pattern, content_lower)) for pattern in grammar_patterns
            )
            score += min(0.3, grammar_matches / len(content.split()) * 10)

        score = max(0.0, min(1.0, score))

        return RankingSignal(
            name=f"{self.language}_relevance",
            score=score,
            weight=0.2,
            metadata={
                "language": self.language,
                "language_features_detected": score > 0.1,
            },
        )

    def _calculate_authority_score(self, metadata: Dict[str, Any]) -> RankingSignal:
        """
        Calculate document authority score.

        Args:
            metadata: Document metadata

        Returns:
            Authority ranking signal
        """
        score = 0.5  # Base score

        # Source-based authority
        source = metadata.get("source", "").lower()
        if any(term in source for term in ["wikipedia", "gov.hr", "akademija"]):
            score += 0.3
        elif any(term in source for term in [".edu", "sveučilište", "fakultet"]):
            score += 0.2
        elif any(term in source for term in ["news", "vijesti", "novosti"]):
            score += 0.1

        # Document type authority
        doc_type = metadata.get("content_type", metadata.get("type", ""))
        if doc_type in self.type_weights:
            score *= self.type_weights[doc_type]

        # Metadata completeness (more complete = more authoritative)
        metadata_fields = ["title", "author", "date", "source", "language"]
        completeness = sum(1 for field in metadata_fields if metadata.get(field))
        score += completeness / len(metadata_fields) * 0.2

        score = max(0.0, min(1.0, score))

        return RankingSignal(
            name="authority_score",
            score=score,
            weight=0.1,
            metadata={
                "source_type": "authoritative" if score > 0.7 else "standard",
                "metadata_completeness": completeness / len(metadata_fields),
            },
        )

    def _calculate_length_appropriateness(
        self, content: str, query_type: QueryType
    ) -> RankingSignal:
        """
        Calculate how appropriate content length is for query type.

        Args:
            content: Document content
            query_type: Type of query

        Returns:
            Length appropriateness ranking signal
        """
        content_length = len(content)
        score = 0.5

        # Different optimal lengths for different query types
        if query_type == QueryType.FACTUAL:
            # Factual queries prefer concise answers
            if 50 <= content_length <= 300:
                score = 1.0
            elif 300 < content_length <= 500:
                score = 0.8
            else:
                score = max(0.3, 1.0 - (content_length - 300) / 1000)

        elif query_type == QueryType.EXPLANATORY:
            # Explanatory queries prefer medium-length detailed content
            if 200 <= content_length <= 800:
                score = 1.0
            elif 100 <= content_length < 200:
                score = 0.7
            elif 800 < content_length <= 1200:
                score = 0.8
            else:
                score = 0.4

        elif query_type == QueryType.SUMMARIZATION:
            # Summarization can handle longer content
            if content_length >= 300:
                score = min(1.0, content_length / 1000)
            else:
                score = 0.3

        elif query_type == QueryType.COMPARISON:
            # Comparison queries benefit from comprehensive content
            if 300 <= content_length <= 1000:
                score = 1.0
            elif content_length > 1000:
                score = 0.9
            else:
                score = 0.6

        score = max(0.0, min(1.0, score))

        return RankingSignal(
            name="length_appropriateness",
            score=score,
            weight=0.1,
            metadata={
                "content_length": content_length,
                "query_type": query_type.value,
                "optimal_range": self._get_optimal_range(query_type),
            },
        )

    def _get_optimal_range(self, query_type: QueryType) -> str:
        """Get optimal length range description for query type."""
        ranges = {
            QueryType.FACTUAL: "50-300 chars",
            QueryType.EXPLANATORY: "200-800 chars",
            QueryType.SUMMARIZATION: "300+ chars",
            QueryType.COMPARISON: "300-1000 chars",
            QueryType.GENERAL: "100-500 chars",
        }
        return ranges.get(query_type, "100-500 chars")

    def _calculate_query_type_match(
        self, content: str, query_type: QueryType
    ) -> RankingSignal:
        """
        Calculate how well content matches the query type.

        Args:
            content: Document content
            query_type: Type of query

        Returns:
            Query type match ranking signal
        """
        content_lower = content.lower()
        score = 0.5  # Base score

        # Patterns that indicate content type
        patterns = {
            QueryType.FACTUAL: [
                r"\b(broj|količina|datum|godine?|kad[ae]?|gdje|tko)\b",
                r"\b(\d{4}\.?\s*god|\d+\s*%|\d+\s*km)\b",
            ],
            QueryType.EXPLANATORY: [
                r"\b(zato što|zbog toga|objašnjenje|razlog)\b",
                r"\b(proces|postupak|način|metod[ae])\b",
            ],
            QueryType.COMPARISON: [
                r"\b(za razliku od|s druge strane|bolje|gore)\b",
                r"\b(usporedba|razlika|sličnost|nasuprot)\b",
            ],
            QueryType.SUMMARIZATION: [
                r"\b(ukratko|sažeto|glavne točke|pregled)\b",
                r"\b(ukupno|općenito|sveukupno)\b",
            ],
        }

        if query_type in patterns:
            matches = sum(
                len(re.findall(pattern, content_lower))
                for pattern in patterns[query_type]
            )
            score += min(0.3, matches * 0.1)

        # Check for structural indicators
        if query_type == QueryType.COMPARISON:
            # Look for comparative structures
            if re.search(r"[•\-]\s*\w+.*vs.*\w+", content, re.IGNORECASE):
                score += 0.2

        elif query_type == QueryType.EXPLANATORY:
            # Look for step-by-step or process indicators
            if re.search(r"\b\d+\.\s+\w+", content) or re.search(
                r"prvo|drugo|treće", content_lower
            ):
                score += 0.2

        score = max(0.0, min(1.0, score))

        return RankingSignal(
            name="query_type_match",
            score=score,
            weight=0.15,
            metadata={"query_type": query_type.value},
        )

    def _combine_ranking_signals(self, signals: List[RankingSignal]) -> float:
        """
        Combine multiple ranking signals into final score.

        Args:
            signals: List of ranking signals

        Returns:
            Final combined score
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

    def _apply_diversity_filtering(
        self, ranked_docs: List[RankedDocument]
    ) -> List[RankedDocument]:
        """
        Apply diversity filtering to avoid too similar results.

        Args:
            ranked_docs: List of ranked documents

        Returns:
            Filtered list with improved diversity
        """
        if len(ranked_docs) <= 3:
            return ranked_docs

        diverse_docs = []
        used_content_hashes = set()

        for doc in ranked_docs:
            # Create a simple content hash for similarity detection
            content_words = set(re.findall(r"\b\w+\b", doc.content.lower()))

            # Check similarity with already selected documents
            is_similar = False
            for existing_hash in used_content_hashes:
                similarity = len(content_words & existing_hash) / len(
                    content_words | existing_hash
                )
                if similarity > self.config.diversity_threshold:
                    is_similar = True
                    break

            if not is_similar or len(diverse_docs) < 2:  # Always keep top 2
                diverse_docs.append(doc)
                used_content_hashes.add(frozenset(content_words))

        return diverse_docs

    def _create_fallback_ranking(
        self, documents: List[Dict[str, Any]]
    ) -> List[RankedDocument]:
        """
        Create fallback ranking when main ranking fails.

        Args:
            documents: Original documents

        Returns:
            Basic ranked documents
        """
        ranked_docs = []

        for i, doc in enumerate(documents):
            ranked_doc = RankedDocument(
                id=doc.get("id", f"fallback_{i}"),
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                original_score=doc.get("relevance_score", 0.0),
                final_score=doc.get("relevance_score", 0.0),
                rank=i + 1,
                ranking_signals=[],
                ranking_metadata={"fallback": True},
            )
            ranked_docs.append(ranked_doc)

        return ranked_docs

    def explain_ranking(self, ranked_doc: RankedDocument) -> str:
        """
        Generate human-readable explanation of document ranking.

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
        signals_by_contribution = sorted(
            ranked_doc.ranking_signals, key=lambda s: s.score * s.weight, reverse=True
        )

        for signal in signals_by_contribution:
            contribution = signal.score * signal.weight
            bar_length = int(contribution * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"   {signal.name:<20}: {contribution:.3f} {bar}")

        lines.append("")
        lines.append(f"📈 Original search score: {ranked_doc.original_score:.3f}")
        lines.append(f"🎯 Final ranking score: {ranked_doc.final_score:.3f}")

        return "\n".join(lines)


def create_result_ranker(
    method: RankingMethod = RankingMethod.LANGUAGE_ENHANCED,
    enable_diversity: bool = True,
    language: str = "hr",
) -> ResultRanker:
    """
    Factory function to create result ranker.

    Args:
        method: Ranking method to use
        enable_diversity: Whether to enable diversity filtering
        language: Language code for ranking

    Returns:
        Configured ResultRanker
    """
    config = RankingConfig(method=method, enable_diversity=enable_diversity)
    return ResultRanker(config, language=language)
