"""
Document chunking system with configurable strategies and language awareness.
Provides intelligent text segmentation for multilingual documents with
configurable chunk sizes and overlap handling.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from ..utils.config_protocol import ConfigProvider
    from .cleaners import CleaningResult

# Pure function algorithms that can be tested in complete isolation


def sliding_window_chunk_positions(
    text_length: int,
    chunk_size: int,
    overlap: int,
    sentence_boundaries: list[int] | None = None,
    respect_sentences: bool = True,
) -> list[tuple[int, int]]:
    """
    Calculate chunk positions for sliding window strategy (pure function).

    Args:
        text_length: Length of text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        sentence_boundaries: List of sentence boundary positions
        respect_sentences: Whether to adjust positions to sentence boundaries

    Returns:
        List of (start, end) positions
    """
    if text_length == 0:
        return []

    positions = []
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Adjust to sentence boundary if requested and boundaries available
        if end < text_length and respect_sentences and sentence_boundaries and sentence_boundaries:
            # Find nearest sentence boundary within reasonable range
            search_range = min(200, text_length - end)  # Configurable range
            best_end = end

            for boundary in sentence_boundaries:
                if end <= boundary <= end + search_range:
                    best_end = boundary
                    break

            end = best_end

        if start < end:  # Ensure valid chunk
            positions.append((start, end))

        # If we've reached the end, stop
        if end >= text_length:
            break

        # Calculate next start with overlap
        next_start = end - overlap
        if next_start <= start:  # Prevent infinite loop - ensure progress
            next_start = start + 1

        start = next_start

    return positions


def sentence_chunk_positions(
    sentences: list[str], chunk_size: int, overlap: int, min_chunk_size: int
) -> list[tuple[int, int, list[str]]]:
    """
    Calculate chunk positions for sentence-based strategy (pure function).

    Args:
        sentences: List of sentences
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size

    Returns:
        List of (start_idx, end_idx, sentence_group) tuples
    """
    if not sentences:
        return []

    chunks = []
    current_sentences: list[str] = []
    current_length = 0
    start_idx = 0

    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)

        # Check if we should create a chunk
        should_chunk = (
            current_length + sentence_length > chunk_size and current_sentences and current_length >= min_chunk_size
        )

        if should_chunk:
            chunks.append((start_idx, i, current_sentences.copy()))

            # Handle overlap
            if overlap > 0 and current_sentences:
                overlap_sentences: list[str] = []
                overlap_length = 0

                for sent in reversed(current_sentences):
                    if overlap_length + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break

                current_sentences = overlap_sentences
                current_length = overlap_length
                start_idx = i - len(overlap_sentences)
            else:
                current_sentences = []
                current_length = 0
                start_idx = i

        current_sentences.append(sentence)
        current_length += sentence_length

    # Handle remaining sentences
    if current_sentences:
        chunks.append((start_idx, len(sentences), current_sentences))

    return chunks


def paragraph_chunk_positions(
    paragraphs: list[str], chunk_size: int, overlap: int, min_chunk_size: int
) -> list[tuple[int, int, list[str]]]:
    """
    Calculate chunk positions for paragraph-based strategy (pure function).

    Args:
        paragraphs: List of paragraphs
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size

    Returns:
        List of (start_idx, end_idx, paragraph_group) tuples
    """
    if not paragraphs:
        return []

    chunks = []
    current_paragraphs: list[str] = []
    current_length = 0
    start_idx = 0

    for i, paragraph in enumerate(paragraphs):
        paragraph_length = len(paragraph)

        # Handle oversized paragraphs by marking them for splitting
        if paragraph_length > chunk_size * 1.5:
            # Finalize current chunk first
            if current_paragraphs:
                chunks.append((start_idx, i, current_paragraphs.copy()))
                current_paragraphs = []
                current_length = 0

            # Mark oversized paragraph for sliding window treatment
            chunks.append((i, i + 1, [paragraph]))
            start_idx = i + 1
            continue

        # Check if we should create a chunk
        should_chunk = (
            current_length + paragraph_length > chunk_size and current_paragraphs and current_length >= min_chunk_size
        )

        if should_chunk:
            chunks.append((start_idx, i, current_paragraphs.copy()))

            # Handle overlap
            if overlap > 0 and current_paragraphs:
                last_para_len = len(current_paragraphs[-1])
                if last_para_len <= overlap:
                    current_paragraphs = [current_paragraphs[-1]]
                    current_length = last_para_len
                    start_idx = i - 1
                else:
                    current_paragraphs = []
                    current_length = 0
                    start_idx = i
            else:
                current_paragraphs = []
                current_length = 0
                start_idx = i

        current_paragraphs.append(paragraph)
        current_length += paragraph_length + 2  # +2 for paragraph separator

    # Handle remaining paragraphs
    if current_paragraphs:
        chunks.append((start_idx, len(paragraphs), current_paragraphs))

    return chunks


def find_sentence_boundaries(text: str, language_patterns: dict[str, Any] | None = None) -> list[int]:
    """
    Find sentence boundaries in text (pure function).

    Args:
        text: Text to analyze
        language_patterns: Language-specific sentence patterns

    Returns:
        List of sentence boundary positions
    """
    if not text:
        return []

    boundaries = []

    # Default sentence endings - works for most languages
    sentence_endings = ".!?"

    # Add language-specific patterns if provided
    if language_patterns and "sentence_endings" in language_patterns:
        sentence_endings = language_patterns["sentence_endings"]

    for i, char in enumerate(text):
        if char in sentence_endings:
            # Check if next character (after optional spaces) is uppercase
            next_pos = i + 1
            while next_pos < len(text) and text[next_pos].isspace():
                next_pos += 1

            if next_pos >= len(text) or text[next_pos].isupper():
                boundaries.append(i + 1)

    return boundaries


def extract_paragraphs(text: str) -> list[str]:
    """Extract paragraphs from text (pure function)."""
    if not text:
        return []

    # Split by double line breaks
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def calculate_chunk_metadata(content: str) -> dict[str, int]:
    """Calculate chunk metadata (pure function)."""
    words = content.split()
    return {"word_count": len(words), "char_count": len(content), "line_count": content.count("\n") + 1}


# Configuration and protocols for dependency injection


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int
    overlap: int
    min_chunk_size: int
    respect_sentences: bool
    sentence_search_range: int
    strategy: ChunkingStrategy

    @classmethod
    def from_config(
        cls, config_dict: dict[str, Any] | None = None, config_provider: Optional["ConfigProvider"] = None
    ) -> "ChunkingConfig":
        """Create config from dictionary or provider with DRY error handling."""
        if config_dict:
            if "chunking" not in config_dict:
                chunking_config = config_dict
            else:
                chunking_config = config_dict["chunking"]

            if "shared" not in config_dict:
                raise ValueError("Missing 'shared' section in configuration")
            shared_config = config_dict["shared"]
        else:
            # Use dependency injection
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            full_config = config_provider.load_config("config")
            chunking_config = full_config["chunking"]
            shared_config = config_provider.get_shared_config()

        # Validate required configuration keys
        required_chunking_keys = [
            "chunk_size",
            "overlap_size",
            "min_chunk_size",
            "preserve_sentence_boundaries",
            "sentence_search_range",
            "strategy",
        ]
        for key in required_chunking_keys:
            if key not in chunking_config:
                if key == "chunk_size" and "default_chunk_size" in shared_config:
                    continue  # Will use shared value
                elif key == "overlap_size" and "default_chunk_overlap" in shared_config:
                    continue  # Will use shared value
                elif key == "min_chunk_size" and "min_chunk_size" in shared_config:
                    continue  # Will use shared value
                else:
                    raise ValueError(f"Missing '{key}' in chunking configuration")

        # Get chunk size
        if "chunk_size" in chunking_config:
            chunk_size = chunking_config["chunk_size"]
        elif "default_chunk_size" in shared_config:
            chunk_size = shared_config["default_chunk_size"]
        else:
            raise ValueError("Missing 'chunk_size' or 'default_chunk_size'")

        # Get overlap
        if "overlap_size" in chunking_config:
            overlap = chunking_config["overlap_size"]
        elif "default_chunk_overlap" in shared_config:
            overlap = shared_config["default_chunk_overlap"]
        else:
            raise ValueError("Missing 'overlap_size' or 'default_chunk_overlap'")

        # Get min chunk size
        if "min_chunk_size" in chunking_config:
            min_chunk_size = chunking_config["min_chunk_size"]
        elif "min_chunk_size" in shared_config:
            min_chunk_size = shared_config["min_chunk_size"]
        else:
            raise ValueError("Missing 'min_chunk_size' in configuration")

        return cls(
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            respect_sentences=chunking_config["preserve_sentence_boundaries"],
            sentence_search_range=chunking_config["sentence_search_range"],
            strategy=ChunkingStrategy(chunking_config["strategy"]),
        )


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""

    content: str
    chunk_id: str
    source_file: str
    start_char: int
    end_char: int
    chunk_index: int
    word_count: int
    char_count: int
    metadata: dict[str, Any]


class TextCleaner(Protocol):
    """Protocol for text cleaning dependencies."""

    def clean_text(self, text: str, preserve_structure: bool = True) -> "CleaningResult":
        """Clean text while optionally preserving structure."""
        ...

    def extract_sentences(self, text: str) -> list[str]:
        """Extract sentences from text."""
        ...


class SentenceExtractor(Protocol):
    """Protocol for sentence extraction dependencies."""

    def extract_sentences(self, text: str) -> list[str]:
        """Extract sentences from text."""
        ...


# Main testable chunker class with full dependency injection


class DocumentChunker:
    """Fully testable document chunker with dependency injection."""

    def __init__(
        self,
        config: ChunkingConfig,
        text_cleaner: TextCleaner | None = None,
        sentence_extractor: SentenceExtractor | None = None,
        language_patterns: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize document chunker with injectable dependencies.

        Args:
            config: Chunking configuration
            text_cleaner: Text cleaning service (optional)
            sentence_extractor: Sentence extraction service (optional)
            language_patterns: Language-specific patterns (optional)
            logger: Logger instance (optional)
        """
        self.config = config
        self.text_cleaner = text_cleaner
        self.sentence_extractor = sentence_extractor
        self.language_patterns = language_patterns or {}
        self.logger = logger or logging.getLogger(__name__)

    def chunk_document(self, text: str, source_file: str, strategy: ChunkingStrategy | None = None) -> list[TextChunk]:
        """
        Chunk document using specified strategy.

        Args:
            text: Document text to chunk
            source_file: Source file path
            strategy: Chunking strategy (uses config default if None)

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        strategy = strategy or self.config.strategy
        self.logger.info(f"Chunking document {source_file} with strategy '{strategy.value}'")

        # Clean text if cleaner available
        cleaned_text = text
        if self.text_cleaner:
            cleaning_result = self.text_cleaner.clean_text(text, preserve_structure=True)
            cleaned_text = cleaning_result.text

        # Execute strategy
        if strategy == ChunkingStrategy.SLIDING_WINDOW:
            chunks = self._sliding_window_chunking(cleaned_text, source_file)
        elif strategy == ChunkingStrategy.SENTENCE:
            chunks = self._sentence_based_chunking(cleaned_text, source_file)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._paragraph_based_chunking(cleaned_text, source_file)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Filter meaningful chunks
        meaningful_chunks = self._filter_meaningful_chunks(chunks)

        self.logger.info(f"Created {len(meaningful_chunks)} meaningful chunks from {len(chunks)} total")
        return meaningful_chunks

    def _sliding_window_chunking(self, text: str, source_file: str) -> list[TextChunk]:
        """Execute sliding window chunking using pure functions."""
        # Get sentence boundaries if needed
        sentence_boundaries = None
        if self.config.respect_sentences:
            sentence_boundaries = find_sentence_boundaries(text, self.language_patterns)

        # Calculate positions using pure function
        positions = sliding_window_chunk_positions(
            text_length=len(text),
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            sentence_boundaries=sentence_boundaries,
            respect_sentences=self.config.respect_sentences,
        )

        # Create chunks from positions
        chunks = []
        for i, (start, end) in enumerate(positions):
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = self._create_chunk(
                    content=chunk_text, source_file=source_file, start_char=start, end_char=end, chunk_index=i
                )
                chunks.append(chunk)

        return chunks

    def _sentence_based_chunking(self, text: str, source_file: str) -> list[TextChunk]:
        """Execute sentence-based chunking using pure functions."""
        # Extract sentences
        if self.sentence_extractor:
            sentences = self.sentence_extractor.extract_sentences(text)
        elif self.text_cleaner:
            sentences = self.text_cleaner.extract_sentences(text)
        else:
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        # Calculate chunk positions using pure function
        chunk_positions = sentence_chunk_positions(
            sentences=sentences,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            min_chunk_size=self.config.min_chunk_size,
        )

        # Create chunks
        chunks = []
        char_offset = 0

        for chunk_idx, (_start_idx, _end_idx, sentence_group) in enumerate(chunk_positions):
            chunk_text = " ".join(sentence_group).strip()
            start_char = char_offset
            end_char = char_offset + len(chunk_text)

            if chunk_text:
                chunk = self._create_chunk(
                    content=chunk_text,
                    source_file=source_file,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_index=chunk_idx,
                )
                chunks.append(chunk)

            char_offset = end_char

        return chunks

    def _paragraph_based_chunking(self, text: str, source_file: str) -> list[TextChunk]:
        """Execute paragraph-based chunking using pure functions."""
        paragraphs = extract_paragraphs(text)

        # Calculate chunk positions using pure function
        chunk_positions = paragraph_chunk_positions(
            paragraphs=paragraphs,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            min_chunk_size=self.config.min_chunk_size,
        )

        # Create chunks
        chunks: list[TextChunk] = []
        char_offset = 0

        for chunk_idx, (_start_idx, _end_idx, paragraph_group) in enumerate(chunk_positions):
            # Handle oversized paragraph (needs sliding window)
            if len(paragraph_group) == 1 and len(paragraph_group[0]) > self.config.chunk_size * 1.5:
                # Use sliding window for this paragraph
                para_chunks = self._sliding_window_chunking(paragraph_group[0], source_file)
                for para_chunk in para_chunks:
                    para_chunk.chunk_index = len(chunks)
                    chunks.append(para_chunk)
                continue

            chunk_text = "\n\n".join(paragraph_group).strip()
            start_char = char_offset
            end_char = char_offset + len(chunk_text)

            if chunk_text:
                chunk = self._create_chunk(
                    content=chunk_text,
                    source_file=source_file,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_index=chunk_idx,
                )
                chunks.append(chunk)

            char_offset = end_char

        return chunks

    def _filter_meaningful_chunks(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """Filter chunks to keep only meaningful ones."""
        meaningful_chunks = []
        for chunk in chunks:
            if len(chunk.content.strip()) >= self.config.min_chunk_size:
                meaningful_chunks.append(chunk)
        return meaningful_chunks

    def _create_chunk(
        self, content: str, source_file: str, start_char: int, end_char: int, chunk_index: int
    ) -> TextChunk:
        """Create a TextChunk with metadata using pure functions."""
        metadata = calculate_chunk_metadata(content)
        chunk_id = f"{Path(source_file).stem}_{chunk_index:04d}"

        return TextChunk(
            content=content,
            chunk_id=chunk_id,
            source_file=source_file,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            word_count=metadata["word_count"],
            char_count=metadata["char_count"],
            metadata=metadata,
        )


# Factory function for convenient creation


def create_document_chunker(
    config_dict: dict[str, Any] | None = None, config_provider: Optional["ConfigProvider"] = None, language: str = "hr"
) -> DocumentChunker:
    """
    Create a DocumentChunker with default dependencies.

    Args:
        config_dict: Configuration dictionary (optional)
        config_provider: Configuration provider (optional)
        language: Language code for language-specific behavior

    Returns:
        Configured DocumentChunker instance
    """
    # Create configuration
    config = ChunkingConfig.from_config(config_dict, config_provider)

    # Create dependencies (can be mocked in tests)
    from typing import cast

    from .cleaners import ConfigProvider as CleanersConfigProvider
    from .cleaners import MultilingualTextCleaner

    cleaners_provider = cast(CleanersConfigProvider, config_provider)
    text_cleaner = MultilingualTextCleaner(language=language, config_provider=cleaners_provider)

    # Get language patterns if available
    language_patterns = None
    if config_provider:
        try:
            language_patterns = config_provider.get_language_specific_config("patterns", language)
        except (KeyError, AttributeError):
            pass
    elif config_dict and "language_specific" in config_dict:
        if "patterns" not in config_dict["language_specific"]:
            raise ValueError("Missing 'patterns' in language_specific configuration")
        language_patterns = config_dict["language_specific"]["patterns"]

    return DocumentChunker(
        config=config,
        text_cleaner=text_cleaner,
        sentence_extractor=text_cleaner,  # Same object implements both protocols
        language_patterns=language_patterns,
    )
