"""
Enhanced chunking system with configurable strategies for large document processing.
Optimized for 300K document scale with smart legal chunking and batch processing support.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_data_transformation,
    log_decision_point,
)


class ChunkingStrategy(Enum):
    """Enhanced chunking strategies for different document types."""

    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SMART_LEGAL = "smart_legal"


@dataclass
class EnhancedChunkingConfig:
    """Configuration for enhanced chunking system."""

    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_chunk_size: int
    preserve_sentence_boundaries: bool
    respect_paragraph_breaks: bool
    preserve_document_structure: bool

    # Smart legal chunking
    smart_legal_enabled: bool
    preserve_section_boundaries: bool
    preserve_paragraph_structure: bool
    merge_short_paragraphs: bool
    section_min_length: int
    paragraph_min_length: int
    legal_section_indicators: list[str]

    # Advanced chunking
    enable_semantic_chunking: bool
    semantic_threshold: float
    max_chunks_per_document: int

    # Sliding window specific
    sentence_search_range: int
    overlap_strategy: str

    # Sentence chunking specific
    min_sentences_per_chunk: int
    max_sentences_per_chunk: int
    sentence_overlap_count: int

    # Paragraph chunking specific
    min_paragraphs_per_chunk: int
    max_paragraphs_per_chunk: int
    paragraph_overlap_count: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EnhancedChunkingConfig":
        """Create config from configuration dictionary with fail-fast validation."""
        logger = get_system_logger()
        log_component_start("enhanced_chunking_config", "from_config", config_keys=list(config.keys()))

        # Direct access - fail if keys missing (no fallbacks)
        chunking_config = config["chunking"]

        strategy = ChunkingStrategy(chunking_config["strategy"])

        # Base chunking parameters
        chunk_size = chunking_config["chunk_size"]
        chunk_overlap = chunking_config["chunk_overlap"]
        min_chunk_size = chunking_config["min_chunk_size"]
        max_chunk_size = chunking_config["max_chunk_size"]

        # Boundary settings
        preserve_sentence_boundaries = chunking_config["preserve_sentence_boundaries"]
        respect_paragraph_breaks = chunking_config["respect_paragraph_breaks"]
        preserve_document_structure = chunking_config["preserve_document_structure"]

        # Smart legal chunking
        smart_legal = chunking_config["smart_legal"]
        smart_legal_enabled = smart_legal["enabled"]
        preserve_section_boundaries = smart_legal["preserve_section_boundaries"]
        preserve_paragraph_structure = smart_legal["preserve_paragraph_structure"]
        merge_short_paragraphs = smart_legal["merge_short_paragraphs"]
        section_min_length = smart_legal["section_min_length"]
        paragraph_min_length = smart_legal["paragraph_min_length"]
        legal_section_indicators = smart_legal["legal_section_indicators"]

        # Advanced chunking
        enable_semantic_chunking = chunking_config["enable_semantic_chunking"]
        semantic_threshold = chunking_config["semantic_threshold"]
        max_chunks_per_document = chunking_config["max_chunks_per_document"]

        # Strategy-specific settings
        sliding_window = chunking_config["sliding_window"]
        sentence_search_range = sliding_window["sentence_search_range"]
        overlap_strategy = sliding_window["overlap_strategy"]

        sentence_config = chunking_config["sentence"]
        min_sentences_per_chunk = sentence_config["min_sentences_per_chunk"]
        max_sentences_per_chunk = sentence_config["max_sentences_per_chunk"]
        sentence_overlap_count = sentence_config["sentence_overlap_count"]

        paragraph_config = chunking_config["paragraph"]
        min_paragraphs_per_chunk = paragraph_config["min_paragraphs_per_chunk"]
        max_paragraphs_per_chunk = paragraph_config["max_paragraphs_per_chunk"]
        paragraph_overlap_count = paragraph_config["paragraph_overlap_count"]

        logger.debug("enhanced_chunking_config", "from_config", f"Strategy: {strategy.value}")
        logger.debug("enhanced_chunking_config", "from_config", f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

        log_component_end("enhanced_chunking_config", "from_config", "Configuration loaded successfully")

        return cls(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            preserve_sentence_boundaries=preserve_sentence_boundaries,
            respect_paragraph_breaks=respect_paragraph_breaks,
            preserve_document_structure=preserve_document_structure,
            smart_legal_enabled=smart_legal_enabled,
            preserve_section_boundaries=preserve_section_boundaries,
            preserve_paragraph_structure=preserve_paragraph_structure,
            merge_short_paragraphs=merge_short_paragraphs,
            section_min_length=section_min_length,
            paragraph_min_length=paragraph_min_length,
            legal_section_indicators=legal_section_indicators,
            enable_semantic_chunking=enable_semantic_chunking,
            semantic_threshold=semantic_threshold,
            max_chunks_per_document=max_chunks_per_document,
            sentence_search_range=sentence_search_range,
            overlap_strategy=overlap_strategy,
            min_sentences_per_chunk=min_sentences_per_chunk,
            max_sentences_per_chunk=max_sentences_per_chunk,
            sentence_overlap_count=sentence_overlap_count,
            min_paragraphs_per_chunk=min_paragraphs_per_chunk,
            max_paragraphs_per_chunk=max_paragraphs_per_chunk,
            paragraph_overlap_count=paragraph_overlap_count,
        )


@dataclass
class EnhancedTextChunk:
    """Enhanced text chunk with comprehensive metadata."""

    content: str
    chunk_id: str
    source_file: str
    start_char: int
    end_char: int
    chunk_index: int
    word_count: int
    char_count: int
    strategy_used: str
    section_type: str | None
    metadata: dict[str, Any]


def smart_legal_chunk_positions(
    text: str, config: EnhancedChunkingConfig, legal_indicators: list[str]
) -> list[tuple[int, int, str]]:
    """
    Calculate chunk positions for smart legal chunking strategy (pure function).

    Identifies legal sections and creates chunks respecting document structure.

    Args:
        text: Text to chunk
        config: Enhanced chunking configuration
        legal_indicators: Legal section indicators

    Returns:
        List of (start, end, section_type) tuples
    """
    logger = get_system_logger()
    log_component_start(
        "smart_legal_chunker", "calculate_positions", text_length=len(text), indicators_count=len(legal_indicators)
    )

    if not text.strip():
        return []

    positions: list[tuple[int, int, str]] = []
    current_pos = 0

    # Split by paragraphs first
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Find paragraph start position in original text
        para_start = text.find(paragraph, current_pos)
        if para_start == -1:
            para_start = current_pos

        para_end = para_start + len(paragraph)

        # Determine section type
        section_type = "content"
        for indicator in legal_indicators:
            if indicator.lower() in paragraph.lower()[:100]:  # Check first 100 chars
                section_type = "legal_section"
                break

        # Check paragraph length and merge if needed
        if (
            config.merge_short_paragraphs
            and len(paragraph) < config.paragraph_min_length
            and positions
            and positions[-1][2] == section_type
        ):
            # Extend previous chunk
            last_start, _, last_type = positions.pop()
            positions.append((last_start, para_end, section_type))
        else:
            # Create new chunk
            positions.append((para_start, para_end, section_type))

        current_pos = para_end

        logger.trace(
            "smart_legal_chunker", "process_paragraph", f"Para {i}: {len(paragraph)} chars, type: {section_type}"
        )

    # Post-process to respect max chunk size
    final_positions = []
    for start, end, section_type in positions:
        chunk_length = end - start

        if chunk_length <= config.max_chunk_size:
            final_positions.append((start, end, section_type))
        else:
            # Split large chunks using sliding window
            chunk_text = text[start:end]
            sub_positions = sliding_window_chunk_positions(
                text_length=len(chunk_text),
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                preserve_boundaries=config.preserve_sentence_boundaries,
            )

            for sub_start, sub_end in sub_positions:
                final_positions.append((start + sub_start, start + sub_end, section_type))

    log_component_end(
        "smart_legal_chunker",
        "calculate_positions",
        f"Generated {len(final_positions)} positions",
        positions_count=len(final_positions),
    )

    return final_positions


def sliding_window_chunk_positions(
    text_length: int,
    chunk_size: int,
    overlap: int,
    sentence_boundaries: list[int] | None = None,
    preserve_boundaries: bool = True,
) -> list[tuple[int, int]]:
    """
    Calculate chunk positions for sliding window strategy (pure function).
    Enhanced version with better boundary detection.
    """
    get_system_logger()

    if text_length == 0:
        return []

    positions = []
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Adjust to sentence boundaries if requested
        if end < text_length and preserve_boundaries and sentence_boundaries:
            search_range = min(200, text_length - end)
            best_end = end

            for boundary in sentence_boundaries:
                if end <= boundary <= end + search_range:
                    best_end = boundary
                    break

            end = best_end

        if start < end:
            positions.append((start, end))

        if end >= text_length:
            break

        # Calculate next start position
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1

        start = next_start

    return positions


def find_enhanced_sentence_boundaries(text: str, language_patterns: dict[str, Any] | None = None) -> list[int]:
    """
    Enhanced sentence boundary detection (pure function).

    Args:
        text: Text to analyze
        language_patterns: Language-specific patterns

    Returns:
        List of sentence boundary positions
    """
    if not text:
        return []

    boundaries = []
    sentence_endings = ".!?"

    # Use language-specific patterns if available
    if language_patterns and "sentence_endings" in language_patterns:
        sentence_endings = "".join(language_patterns["sentence_endings"])

    # Enhanced sentence detection with abbreviation handling
    abbreviations = {"dr", "prof", "mr", "mrs", "ms", "vs", "etc", "jr", "sr"}
    if language_patterns and "abbreviations" in language_patterns:
        abbreviations.update(language_patterns["abbreviations"])

    for i, char in enumerate(text):
        if char in sentence_endings:
            # Check for abbreviations
            if char == "." and i > 0:
                # Look back for potential abbreviation
                word_start = i - 1
                while word_start >= 0 and text[word_start].isalnum():
                    word_start -= 1
                word_start += 1

                if word_start < i:
                    word = text[word_start:i].lower()
                    if word in abbreviations:
                        continue

            # Check if next character (after spaces) is uppercase
            next_pos = i + 1
            while next_pos < len(text) and text[next_pos].isspace():
                next_pos += 1

            if next_pos >= len(text) or text[next_pos].isupper():
                boundaries.append(i + 1)

    return boundaries


def calculate_enhanced_chunk_metadata(content: str, strategy: str, section_type: str | None = None) -> dict[str, Any]:
    """Calculate enhanced chunk metadata (pure function)."""
    words = content.split()
    lines = content.count("\n") + 1

    # Calculate readability metrics
    sentences = len([s for s in re.split(r"[.!?]+", content) if s.strip()])
    avg_words_per_sentence = len(words) / max(sentences, 1)

    # Legal document specific metrics
    legal_indicators = 0
    if section_type == "legal_section":
        legal_terms = ["članak", "stavak", "točka", "podtočka", "odjeljak"]
        legal_indicators = sum(1 for term in legal_terms if term in content.lower())

    return {
        "word_count": len(words),
        "char_count": len(content),
        "line_count": lines,
        "sentence_count": sentences,
        "avg_words_per_sentence": round(avg_words_per_sentence, 2),
        "strategy_used": strategy,
        "section_type": section_type or "content",
        "legal_indicators": legal_indicators,
        "readability_score": min(100, max(0, 206.835 - 1.015 * avg_words_per_sentence)),
    }


class EnhancedDocumentChunker:
    """Enhanced document chunker with multiple strategies and optimization."""

    def __init__(self, config: EnhancedChunkingConfig):
        """Initialize enhanced document chunker."""
        self.config = config
        self.logger = get_system_logger()

        log_component_start(
            "enhanced_document_chunker",
            "init",
            strategy=config.strategy.value,
            chunk_size=config.chunk_size,
            max_chunks=config.max_chunks_per_document,
        )

        self.logger.info("enhanced_chunker", "init", f"Initialized with {config.strategy.value} strategy")

    def chunk_document(
        self, text: str, source_file: str, strategy_override: ChunkingStrategy | None = None
    ) -> list[EnhancedTextChunk]:
        """
        Chunk document using enhanced strategies.

        Args:
            text: Document text to chunk
            source_file: Source file path
            strategy_override: Override default strategy

        Returns:
            List of enhanced text chunks
        """
        logger = get_system_logger()
        log_component_start(
            "enhanced_document_chunker",
            "chunk_document",
            source_file=source_file,
            text_length=len(text),
            strategy_override=strategy_override.value if strategy_override else None,
        )

        if not text or not text.strip():
            logger.debug("enhanced_document_chunker", "chunk_document", "Empty text, returning no chunks")
            return []

        strategy = strategy_override or self.config.strategy
        log_decision_point(
            "enhanced_document_chunker", "chunk_document", "strategy_selection", f"using {strategy.value}"
        )

        # Apply chunking strategy
        if strategy == ChunkingStrategy.SMART_LEGAL:
            chunks = self._smart_legal_chunking(text, source_file)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            chunks = self._sliding_window_chunking(text, source_file)
        elif strategy == ChunkingStrategy.SENTENCE:
            chunks = self._sentence_chunking(text, source_file)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._paragraph_chunking(text, source_file)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Apply document limits
        if len(chunks) > self.config.max_chunks_per_document:
            logger.warning(
                "enhanced_document_chunker",
                "chunk_document",
                f"Document has {len(chunks)} chunks, limiting to {self.config.max_chunks_per_document}",
            )
            chunks = chunks[: self.config.max_chunks_per_document]

        # Filter meaningful chunks
        meaningful_chunks = self._filter_meaningful_chunks(chunks)

        log_data_transformation(
            "enhanced_document_chunker",
            "filter_chunks",
            f"chunks[{len(chunks)}]",
            f"meaningful[{len(meaningful_chunks)}]",
        )

        log_component_end(
            "enhanced_document_chunker",
            "chunk_document",
            f"Created {len(meaningful_chunks)} meaningful chunks",
            total_chunks=len(chunks),
            meaningful_chunks=len(meaningful_chunks),
            strategy=strategy.value,
        )

        return meaningful_chunks

    def _smart_legal_chunking(self, text: str, source_file: str) -> list[EnhancedTextChunk]:
        """Execute smart legal chunking strategy."""
        get_system_logger()
        log_component_start("smart_legal_chunker", "chunk_text", text_length=len(text), source_file=source_file)

        positions = smart_legal_chunk_positions(
            text=text, config=self.config, legal_indicators=self.config.legal_section_indicators
        )

        chunks = []
        for i, (start, end, section_type) in enumerate(positions):
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = self._create_enhanced_chunk(
                    content=chunk_text,
                    source_file=source_file,
                    start_char=start,
                    end_char=end,
                    chunk_index=i,
                    strategy="smart_legal",
                    section_type=section_type,
                )
                chunks.append(chunk)

        log_component_end(
            "smart_legal_chunker", "chunk_text", f"Created {len(chunks)} legal chunks", chunks_created=len(chunks)
        )

        return chunks

    def _sliding_window_chunking(self, text: str, source_file: str) -> list[EnhancedTextChunk]:
        """Execute sliding window chunking strategy."""
        sentence_boundaries = None
        if self.config.preserve_sentence_boundaries:
            sentence_boundaries = find_enhanced_sentence_boundaries(text)

        positions = sliding_window_chunk_positions(
            text_length=len(text),
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            sentence_boundaries=sentence_boundaries,
            preserve_boundaries=self.config.preserve_sentence_boundaries,
        )

        chunks = []
        for i, (start, end) in enumerate(positions):
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = self._create_enhanced_chunk(
                    content=chunk_text,
                    source_file=source_file,
                    start_char=start,
                    end_char=end,
                    chunk_index=i,
                    strategy="sliding_window",
                )
                chunks.append(chunk)

        return chunks

    def _sentence_chunking(self, text: str, source_file: str) -> list[EnhancedTextChunk]:
        """Execute sentence-based chunking strategy."""
        # Simple sentence splitting - can be enhanced with NLP
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        chunks = []
        current_content: list[str] = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if we should create a chunk
            if (
                current_length + sentence_length > self.config.chunk_size
                and current_content
                and len(current_content) >= self.config.min_sentences_per_chunk
            ):
                # Create chunk
                chunk_text = ". ".join(current_content) + "."
                chunk = self._create_enhanced_chunk(
                    content=chunk_text.strip(),
                    source_file=source_file,
                    start_char=0,  # Would need proper calculation
                    end_char=len(chunk_text),
                    chunk_index=chunk_index,
                    strategy="sentence",
                )
                chunks.append(chunk)
                chunk_index += 1

                # Handle overlap
                overlap_sentences = max(0, self.config.sentence_overlap_count)
                current_content = current_content[-overlap_sentences:] if overlap_sentences > 0 else []
                current_length = sum(len(s) for s in current_content)

            current_content.append(sentence)
            current_length += sentence_length

        # Handle remaining sentences
        if current_content:
            chunk_text = ". ".join(current_content) + "."
            chunk = self._create_enhanced_chunk(
                content=chunk_text.strip(),
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_text),
                chunk_index=chunk_index,
                strategy="sentence",
            )
            chunks.append(chunk)

        return chunks

    def _paragraph_chunking(self, text: str, source_file: str) -> list[EnhancedTextChunk]:
        """Execute paragraph-based chunking strategy."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        chunks = []
        current_content: list[str] = []
        current_length = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)

            # Check if we should create a chunk
            if (
                current_length + paragraph_length > self.config.chunk_size
                and current_content
                and len(current_content) >= self.config.min_paragraphs_per_chunk
            ):
                # Create chunk
                chunk_text = "\n\n".join(current_content)
                chunk = self._create_enhanced_chunk(
                    content=chunk_text.strip(),
                    source_file=source_file,
                    start_char=0,  # Would need proper calculation
                    end_char=len(chunk_text),
                    chunk_index=chunk_index,
                    strategy="paragraph",
                )
                chunks.append(chunk)
                chunk_index += 1

                # Handle overlap
                overlap_count = self.config.paragraph_overlap_count
                current_content = current_content[-overlap_count:] if overlap_count > 0 else []
                current_length = sum(len(p) for p in current_content)

            current_content.append(paragraph)
            current_length += paragraph_length

        # Handle remaining paragraphs
        if current_content:
            chunk_text = "\n\n".join(current_content)
            chunk = self._create_enhanced_chunk(
                content=chunk_text.strip(),
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_text),
                chunk_index=chunk_index,
                strategy="paragraph",
            )
            chunks.append(chunk)

        return chunks

    def _filter_meaningful_chunks(self, chunks: list[EnhancedTextChunk]) -> list[EnhancedTextChunk]:
        """Filter chunks to keep only meaningful ones."""
        meaningful_chunks = []

        for chunk in chunks:
            content_length = len(chunk.content.strip())
            word_count = chunk.word_count

            # Apply size filters
            if (
                content_length >= self.config.min_chunk_size
                and content_length <= self.config.max_chunk_size
                and word_count >= 3
            ):  # Minimum word count for meaningful content
                meaningful_chunks.append(chunk)

        return meaningful_chunks

    def _create_enhanced_chunk(
        self,
        content: str,
        source_file: str,
        start_char: int,
        end_char: int,
        chunk_index: int,
        strategy: str,
        section_type: str | None = None,
    ) -> EnhancedTextChunk:
        """Create an enhanced text chunk with comprehensive metadata."""
        metadata = calculate_enhanced_chunk_metadata(content, strategy, section_type)
        chunk_id = f"{Path(source_file).stem}_{chunk_index:04d}_{strategy}"

        return EnhancedTextChunk(
            content=content,
            chunk_id=chunk_id,
            source_file=source_file,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            word_count=metadata["word_count"],
            char_count=metadata["char_count"],
            strategy_used=strategy,
            section_type=section_type,
            metadata=metadata,
        )


def create_enhanced_chunker(config: dict[str, Any]) -> EnhancedDocumentChunker:
    """
    Factory function to create enhanced document chunker.

    Args:
        config: Configuration dictionary

    Returns:
        Configured EnhancedDocumentChunker instance
    """
    chunking_config = EnhancedChunkingConfig.from_config(config)
    return EnhancedDocumentChunker(chunking_config)
