"""
Document chunking strategies for multilingual text.
Implements various chunking approaches optimized for RAG retrieval.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config_loader import (get_chunking_config,
                                   get_language_specific_config,
                                   get_shared_config)
from .cleaners import MultilingualTextCleaner

logger = logging.getLogger(__name__)


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


class DocumentChunker:
    """Chunk multilingual documents for optimal RAG retrieval."""

    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        min_chunk_size: int = None,
        respect_sentences: bool = None,
        language: str = "hr",
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Target chunk size in characters (uses config if None)
            overlap: Overlap between chunks in characters (uses config if None)
            min_chunk_size: Minimum chunk size to keep (uses config if None)
            respect_sentences: Whether to try to keep sentences intact (uses config if None)
            language: Language code for language-specific behavior
        """
        self.language = language
        self._chunking_config = get_chunking_config()
        self._language_config = get_language_specific_config(
            "text_processing", self.language
        )

        # Load shared config for chunk size defaults
        shared_config = get_shared_config()

        self.chunk_size = chunk_size or self._chunking_config.get(
            "chunk_size", shared_config["default_chunk_size"]
        )
        self.overlap = overlap or self._chunking_config.get(
            "overlap_size", shared_config["default_chunk_overlap"]
        )
        self.min_chunk_size = min_chunk_size or shared_config["min_chunk_size"]
        self.respect_sentences = (
            respect_sentences
            if respect_sentences is not None
            else self._chunking_config["preserve_sentence_boundaries"]
        )
        self.cleaner = MultilingualTextCleaner(language=self.language)

        logger.info(f"Initialized chunker: size={chunk_size}, overlap={overlap}")

    def chunk_document(
        self, text: str, source_file: str, strategy: str = "sliding_window"
    ) -> List[TextChunk]:
        """
        Chunk a document using specified strategy.

        Args:
            text: Document text to chunk
            source_file: Source file path
            strategy: Chunking strategy ('sliding_window', 'sentence', 'paragraph')

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        logger.info(f"Chunking document {source_file} with strategy '{strategy}'")

        # Clean text first
        cleaned_text = self.cleaner.clean_text(text, preserve_structure=True)

        if strategy == "sliding_window":
            chunks = self._sliding_window_chunking(cleaned_text, source_file)
        elif strategy == "sentence":
            chunks = self._sentence_based_chunking(cleaned_text, source_file)
        elif strategy == "paragraph":
            chunks = self._paragraph_based_chunking(cleaned_text, source_file)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        # Filter out chunks that are too small or not meaningful
        meaningful_chunks = []
        for chunk in chunks:
            # Simple check - just verify minimum size for now
            if len(chunk.content.strip()) >= self.min_chunk_size:
                meaningful_chunks.append(chunk)

        logger.info(
            f"Created {len(meaningful_chunks)} meaningful chunks from {len(chunks)} total"
        )
        return meaningful_chunks

    def _sliding_window_chunking(self, text: str, source_file: str) -> List[TextChunk]:
        """Implement sliding window chunking with language-specific text awareness."""
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))

            # If we're not at the end and respect_sentences is True, try to end at sentence boundary
            if end < len(text) and self.respect_sentences:
                end = self._find_sentence_boundary(text, end, start)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = self._create_chunk(
                    content=chunk_text,
                    source_file=source_file,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            start = max(end - self.overlap, start + 1)

            # Prevent infinite loop
            if start >= end:
                break

        return chunks

    def _sentence_based_chunking(self, text: str, source_file: str) -> List[TextChunk]:
        """Chunk by sentences, grouping sentences to reach target size."""
        sentences = self.cleaner.extract_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0
        start_char = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk size, create current chunk
            if (
                current_length + sentence_length > self.chunk_size
                and current_chunk
                and current_length >= self.min_chunk_size
            ):
                chunk_text = " ".join(current_chunk).strip()
                end_char = start_char + len(chunk_text)

                chunk = self._create_chunk(
                    content=chunk_text,
                    source_file=source_file,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Handle overlap - keep last sentence if overlap is enabled
                if self.overlap > 0 and current_chunk:
                    overlap_sentences = []
                    overlap_length = 0
                    for sent in reversed(current_chunk):
                        if overlap_length + len(sent) <= self.overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_length += len(sent)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                    start_char = end_char - overlap_length
                else:
                    current_chunk = []
                    current_length = 0
                    start_char = end_char

            current_chunk.append(sentence)
            current_length += sentence_length

        # Handle remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            end_char = start_char + len(chunk_text)

            chunk = self._create_chunk(
                content=chunk_text,
                source_file=source_file,
                start_char=start_char,
                end_char=end_char,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)

        return chunks

    def _paragraph_based_chunking(self, text: str, source_file: str) -> List[TextChunk]:
        """Chunk by paragraphs, combining paragraphs to reach target size."""
        # Split by double line breaks to get paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        chunks = []
        current_chunk = []
        current_length = 0
        start_char = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)

            # If this paragraph alone exceeds chunk size, split it
            if paragraph_length > self.chunk_size * 1.5:
                # First, finalize current chunk if it has content
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk).strip()
                    end_char = start_char + len(chunk_text)

                    chunk = self._create_chunk(
                        content=chunk_text,
                        source_file=source_file,
                        start_char=start_char,
                        end_char=end_char,
                        chunk_index=chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_char = end_char
                    current_chunk = []
                    current_length = 0

                # Split the large paragraph using sliding window
                paragraph_chunks = self._sliding_window_chunking(paragraph, source_file)
                for para_chunk in paragraph_chunks:
                    para_chunk.chunk_index = chunk_index
                    chunks.append(para_chunk)
                    chunk_index += 1

                continue

            # If adding this paragraph would exceed chunk size, create current chunk
            if (
                current_length + paragraph_length > self.chunk_size
                and current_chunk
                and current_length >= self.min_chunk_size
            ):
                chunk_text = "\n\n".join(current_chunk).strip()
                end_char = start_char + len(chunk_text)

                chunk = self._create_chunk(
                    content=chunk_text,
                    source_file=source_file,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Handle overlap
                if self.overlap > 0 and current_chunk:
                    # Keep last paragraph if it fits in overlap
                    last_para_len = len(current_chunk[-1])
                    if last_para_len <= self.overlap:
                        current_chunk = [current_chunk[-1]]
                        current_length = last_para_len
                        start_char = end_char - last_para_len
                    else:
                        current_chunk = []
                        current_length = 0
                        start_char = end_char
                else:
                    current_chunk = []
                    current_length = 0
                    start_char = end_char

            current_chunk.append(paragraph)
            current_length += paragraph_length + 2  # +2 for paragraph separator

        # Handle remaining paragraphs
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk).strip()
            end_char = start_char + len(chunk_text)

            chunk = self._create_chunk(
                content=chunk_text,
                source_file=source_file,
                start_char=start_char,
                end_char=end_char,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)

        return chunks

    def _find_sentence_boundary(self, text: str, target_pos: int, min_pos: int) -> int:
        """Find the nearest sentence boundary near target position."""
        # Look for sentence endings within a reasonable range
        search_range = min(
            self._chunking_config["sentence_search_range"], len(text) - target_pos
        )

        # Search forward for sentence ending
        for i in range(target_pos, min(target_pos + search_range, len(text))):
            if text[i] in ".!?" and (i + 1 >= len(text) or text[i + 1].isspace()):
                # Check if next character (after space) is uppercase
                next_char_pos = i + 1
                while next_char_pos < len(text) and text[next_char_pos].isspace():
                    next_char_pos += 1

                # Use Unicode uppercase detection - works for ALL languages automatically
                if next_char_pos >= len(text) or text[next_char_pos].isupper():
                    return i + 1

        # If no sentence boundary found, return original position
        return target_pos

    def _create_chunk(
        self,
        content: str,
        source_file: str,
        start_char: int,
        end_char: int,
        chunk_index: int,
    ) -> TextChunk:
        """Create a TextChunk object with metadata."""
        words = content.split()

        chunk_id = f"{Path(source_file).stem}_{chunk_index:04d}"

        return TextChunk(
            content=content,
            chunk_id=chunk_id,
            source_file=source_file,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            word_count=len(words),
            char_count=len(content),
        )


def chunk_document(
    text: str,
    source_file: str,
    chunk_size: int = None,
    overlap: int = None,
    strategy: str = None,
    language: str = "hr",
) -> List[TextChunk]:
    """
    Chunk documents for optimal RAG retrieval with language support.

    Args:
        text: Document text
        source_file: Source file path
        chunk_size: Target chunk size in characters (uses config if None)
        overlap: Overlap between chunks (uses config if None)
        strategy: Chunking strategy (uses config if None)
        language: Language code for language-specific behavior

    Returns:
        List of text chunks
    """
    if strategy is None:
        strategy = "sentence_aware"  # Default strategy

    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap, language=language)
    return chunker.chunk_document(text, source_file, strategy)
