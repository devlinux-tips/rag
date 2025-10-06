"""
Simple query type classifier for determining data source routing.
Determines whether a query should use user documents or feature sources
like Narodne Novine.
"""

import re
from enum import Enum
from typing import List
from dataclasses import dataclass

from ..utils.logging_factory import get_system_logger


class QueryType(Enum):
    """Query types for data source routing."""
    USER_DOCUMENTS = "user"  # Query user's personal documents
    NARODNE_NOVINE = "narodne_novine"  # Query Croatian Official Gazette
    MIXED = "mixed"  # Query both sources


@dataclass
class QueryClassification:
    """Result of query classification."""
    primary_type: QueryType
    confidence: float
    reasoning: str
    detected_keywords: List[str]
    should_include_user_docs: bool = True
    should_include_narodne_novine: bool = False


class SimpleQueryClassifier:
    """Simple rule-based classifier for query type detection."""

    def __init__(self, language: str = "hr"):
        self.language = language
        self.logger = get_system_logger()

        # Croatian legal keywords that indicate Narodne Novine queries
        self.legal_keywords = {
            # Direct references
            "narodne novine", "nn", "službeni glasnik", "službeno glasilo",

            # Legal document types
            "zakon", "uredba", "pravilnik", "odluka", "rješenje", "propis",
            "direktiva", "regulativa", "norma", "normativ",

            # Legal concepts
            "pravni", "pravno", "pravna", "zakonski", "zakonska", "zakonsko",
            "regulatorni", "regulatorna", "regulatorno",

            # Government entities
            "ministarstvo", "vlada", "sabor", "parlament", "državni",
            "republic", "republika", "upravni", "javni sektor",

            # Legal procedures
            "stupanje na snagu", "objava", "izmjena", "dopuna",
            "prestanak važenja", "implementacija", "primjena zakona",

            # Common legal phrases
            "članak", "stavak", "točka", "odredba", "odredbe", "propisi",
            "zakonska obveza", "pravna osnova", "pravni okvir"
        }

        # Keywords that indicate user document queries
        self.personal_keywords = {
            "moj", "moja", "moje", "naš", "naša", "naše",
            "privatni", "osobni", "interni", "interna", "interno",
            "kompanijski", "tvrtka", "organizacija", "projekt",
            "dokument", "dokumenti", "datoteke", "datoteka",
            "bilješke", "zapis", "zapisi", "izvještaj", "izvještaji"
        }

    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify query to determine appropriate data sources.

        Args:
            query: User's query text

        Returns:
            QueryClassification with routing decision
        """
        query_lower = query.lower().strip()

        # Find legal keywords
        legal_matches = []
        for keyword in self.legal_keywords:
            if keyword in query_lower:
                legal_matches.append(keyword)

        # Find personal keywords
        personal_matches = []
        for keyword in self.personal_keywords:
            if keyword in query_lower:
                personal_matches.append(keyword)

        # Check for specific patterns
        has_nn_reference = self._check_nn_reference(query_lower)
        has_legal_entities = self._check_legal_entities(query_lower)
        has_year_reference = self._check_year_reference(query_lower)

        # Calculate confidence and make decision
        legal_score = (
            len(legal_matches)
            + (2 if has_nn_reference else 0)
            + (1 if has_legal_entities else 0)
            + (0.5 if has_year_reference else 0)
        )
        personal_score = len(personal_matches)

        self.logger.debug(
            "query_classifier", "classify",
            f"Legal score: {legal_score}, "
            f"Personal score: {personal_score}, "
            f"Legal matches: {legal_matches}, "
            f"Personal matches: {personal_matches}"
        )

        # Decision logic
        if legal_score >= 2.0:  # Strong legal indicators
            if personal_score > 0:
                # Mixed query - check both sources
                primary_type = QueryType.MIXED
                confidence = 0.8
                reasoning = (
                    f"Mixed query: legal keywords ({legal_matches}) + "
                    f"personal keywords ({personal_matches})"
                )
                should_include_user_docs = True
                should_include_nn = True
            else:
                # Pure legal query
                primary_type = QueryType.NARODNE_NOVINE
                confidence = 0.9
                reasoning = (
                    f"Legal query: detected legal keywords "
                    f"({legal_matches})"
                )
                should_include_user_docs = False
                should_include_nn = True

        elif legal_score >= 1.0:  # Some legal indicators
            # Include both sources with lower confidence
            primary_type = QueryType.MIXED
            confidence = 0.6
            reasoning = (
                f"Possible legal query: some legal keywords "
                f"({legal_matches})"
            )
            should_include_user_docs = True
            should_include_nn = True

        else:
            # Default to user documents
            primary_type = QueryType.USER_DOCUMENTS
            confidence = 0.8 if personal_score > 0 else 0.5
            default_indicator = (
                personal_matches if personal_matches
                else 'no specific indicators'
            )
            reasoning = f"Personal/general query: {default_indicator}"
            should_include_user_docs = True
            should_include_nn = False

        all_detected = legal_matches + personal_matches

        result = QueryClassification(
            primary_type=primary_type,
            confidence=confidence,
            reasoning=reasoning,
            detected_keywords=all_detected,
            should_include_user_docs=should_include_user_docs,
            should_include_narodne_novine=should_include_nn
        )

        self.logger.info(
            "query_classifier", "classify",
            f"Query classified as {primary_type.value} "
            f"(confidence: {confidence:.2f}): {reasoning}"
        )

        return result

    def _check_nn_reference(self, query: str) -> bool:
        """Check for explicit Narodne Novine references."""
        patterns = [
            r'\bnn\s+\d+/\d+',          # "NN 85/2009"
            r'narodne novine\s+\d+',    # "Narodne novine 85"
            r'\d+/\d+',                 # "85/2009" (generic year pattern)
        ]

        return any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in patterns
        )

    def _check_legal_entities(self, query: str) -> bool:
        """Check for legal entity patterns like law names."""
        patterns = [
            r'zakon\s+o\s+\w+',         # "Zakon o radu"
            r'uredba\s+o\s+\w+',        # "Uredba o ..."
            r'pravilnik\s+o\s+\w+',     # "Pravilnik o ..."
            r'\b[A-ZČĆŽŠĐ]{2,5}\b',     # Law abbreviations like "ZOR"
        ]

        return any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in patterns
        )

    def _check_year_reference(self, query: str) -> bool:
        """Check for year references that might indicate legal documents."""
        pattern = r'\b(19\d{2}|20\d{2})\b'  # Years from 1900-2099
        return bool(re.search(pattern, query))


def create_query_classifier(language: str = "hr") -> SimpleQueryClassifier:
    """Factory function to create query classifier."""
    return SimpleQueryClassifier(language=language)
