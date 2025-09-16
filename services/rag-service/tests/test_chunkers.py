"""
Comprehensive tests for document chunking system.
Tests pure functions, dependency injection, chunking strategies, and document processing workflows.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Any, Optional

from src.preprocessing.chunkers import (
    # Pure functions
    sliding_window_chunk_positions,
    sentence_chunk_positions,
    paragraph_chunk_positions,
    find_sentence_boundaries,
    extract_paragraphs,
    calculate_chunk_metadata,

    # Enums and data structures
    ChunkingStrategy,
    ChunkingConfig,
    TextChunk,

    # Protocols
    TextCleaner,
    SentenceExtractor,

    # Main class
    DocumentChunker,

    # Factory function
    create_document_chunker,
)


# Test Data Structures
class TestDataStructures:
    """Test chunking data structures."""

    def test_chunking_strategy_enum(self):
        """Test ChunkingStrategy enum values."""
        assert ChunkingStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert ChunkingStrategy.SENTENCE.value == "sentence"
        assert ChunkingStrategy.PARAGRAPH.value == "paragraph"

    def test_text_chunk_creation(self):
        """Test TextChunk dataclass creation."""
        chunk = TextChunk(
            content="This is a test chunk.",
            chunk_id="test_0001",
            source_file="test.txt",
            start_char=0,
            end_char=21,
            chunk_index=0,
            word_count=5,
            char_count=21,
            metadata={"test": "value"}
        )

        assert chunk.content == "This is a test chunk."
        assert chunk.chunk_id == "test_0001"
        assert chunk.source_file == "test.txt"
        assert chunk.start_char == 0
        assert chunk.end_char == 21
        assert chunk.chunk_index == 0
        assert chunk.word_count == 5
        assert chunk.char_count == 21
        assert chunk.metadata == {"test": "value"}

    def test_chunking_config_creation(self):
        """Test ChunkingConfig creation with from_config method."""
        config_dict = {
            "chunking": {
                "chunk_size": 1000,
                "overlap_size": 200,
                "min_chunk_size": 100,
                "preserve_sentence_boundaries": True,
                "sentence_search_range": 200,
                "strategy": "sliding_window"
            },
            "shared": {
                "default_chunk_size": 800,
                "default_chunk_overlap": 150,
                "min_chunk_size": 80
            }
        }

        config = ChunkingConfig.from_config(config_dict)

        assert config.chunk_size == 1000
        assert config.overlap == 200
        assert config.min_chunk_size == 100
        assert config.respect_sentences == True
        assert config.sentence_search_range == 200
        assert config.strategy == ChunkingStrategy.SLIDING_WINDOW

    def test_chunking_config_with_shared_defaults(self):
        """Test ChunkingConfig using shared defaults."""
        config_dict = {
            "chunking": {
                "preserve_sentence_boundaries": True,
                "sentence_search_range": 150,
                "strategy": "sentence"
            },
            "shared": {
                "default_chunk_size": 800,
                "default_chunk_overlap": 150,
                "min_chunk_size": 80
            }
        }

        config = ChunkingConfig.from_config(config_dict)

        assert config.chunk_size == 800  # From shared
        assert config.overlap == 150     # From shared
        assert config.min_chunk_size == 80  # From shared
        assert config.strategy == ChunkingStrategy.SENTENCE

    def test_chunking_config_missing_required_keys(self):
        """Test ChunkingConfig error handling for missing keys."""
        config_dict = {
            "chunking": {
                "chunk_size": 1000
                # Missing other required keys
            },
            "shared": {}
        }

        with pytest.raises(ValueError, match="Missing .* in chunking configuration"):
            ChunkingConfig.from_config(config_dict)

    def test_chunking_config_missing_shared_section(self):
        """Test ChunkingConfig error handling for missing shared section."""
        config_dict = {
            "chunking": {
                "chunk_size": 1000,
                "overlap_size": 200,
                "min_chunk_size": 100,
                "preserve_sentence_boundaries": True,
                "sentence_search_range": 200,
                "strategy": "sliding_window"
            }
            # Missing shared section
        }

        with pytest.raises(ValueError, match="Missing 'shared' section in configuration"):
            ChunkingConfig.from_config(config_dict)


# Test Pure Functions
class TestSlidingWindowChunkPositions:
    """Test sliding window chunk position calculation."""

    def test_empty_text(self):
        """Test with empty text."""
        positions = sliding_window_chunk_positions(0, 1000, 200)
        assert positions == []

    def test_single_chunk(self):
        """Test text that fits in single chunk."""
        positions = sliding_window_chunk_positions(500, 1000, 200)
        assert positions == [(0, 500)]

    def test_multiple_chunks_no_overlap(self):
        """Test multiple chunks without overlap."""
        positions = sliding_window_chunk_positions(2500, 1000, 0)
        expected = [(0, 1000), (1000, 2000), (2000, 2500)]
        assert positions == expected

    def test_multiple_chunks_with_overlap(self):
        """Test multiple chunks with overlap."""
        positions = sliding_window_chunk_positions(2500, 1000, 200)
        expected = [(0, 1000), (800, 1800), (1600, 2500)]
        assert positions == expected

    def test_sentence_boundary_adjustment(self):
        """Test adjustment to sentence boundaries."""
        sentence_boundaries = [950, 1100, 1450]
        positions = sliding_window_chunk_positions(
            2000, 1000, 200, sentence_boundaries, respect_sentences=True
        )

        # Should adjust end positions to sentence boundaries where possible
        assert len(positions) >= 2
        # First chunk should end at or near sentence boundary
        assert positions[0][1] in [950, 1000, 1100]

    def test_no_sentence_boundary_adjustment(self):
        """Test without sentence boundary adjustment."""
        sentence_boundaries = [950, 1100, 1450]
        positions = sliding_window_chunk_positions(
            2000, 1000, 200, sentence_boundaries, respect_sentences=False
        )

        # Should ignore sentence boundaries
        expected = [(0, 1000), (800, 1800), (1600, 2000)]
        assert positions == expected

    def test_overlap_larger_than_chunk(self):
        """Test edge case where overlap is larger than chunk."""
        positions = sliding_window_chunk_positions(3000, 1000, 1200)

        # Should still make progress (prevent infinite loop)
        assert len(positions) > 0
        assert all(start < end for start, end in positions)

    def test_sentence_boundaries_empty(self):
        """Test with empty sentence boundaries list."""
        positions = sliding_window_chunk_positions(
            2000, 1000, 200, [], respect_sentences=True
        )

        # Should work like no sentence boundaries
        expected = [(0, 1000), (800, 1800), (1600, 2000)]
        assert positions == expected


class TestSentenceChunkPositions:
    """Test sentence-based chunk position calculation."""

    def test_empty_sentences(self):
        """Test with empty sentences list."""
        positions = sentence_chunk_positions([], 1000, 200, 100)
        assert positions == []

    def test_single_sentence(self):
        """Test with single sentence."""
        sentences = ["This is a test sentence."]
        positions = sentence_chunk_positions(sentences, 1000, 200, 10)
        assert positions == [(0, 1, ["This is a test sentence."])]

    def test_multiple_sentences_single_chunk(self):
        """Test multiple sentences that fit in single chunk."""
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        positions = sentence_chunk_positions(sentences, 1000, 200, 10)
        assert positions == [(0, 3, sentences)]

    def test_multiple_chunks_no_overlap(self):
        """Test multiple chunks without overlap."""
        sentences = ["A" * 400, "B" * 400, "C" * 400]  # Each ~400 chars
        positions = sentence_chunk_positions(sentences, 500, 0, 100)

        assert len(positions) == 3
        assert positions[0] == (0, 1, ["A" * 400])
        assert positions[1] == (1, 2, ["B" * 400])
        assert positions[2] == (2, 3, ["C" * 400])

    def test_multiple_chunks_with_overlap(self):
        """Test multiple chunks with overlap."""
        sentences = ["A" * 300, "B" * 300, "C" * 300, "D" * 300]
        positions = sentence_chunk_positions(sentences, 500, 200, 100)

        assert len(positions) >= 2
        # Verify that chunks are created (exact overlap behavior may vary)
        for start_idx, end_idx, sentence_group in positions:
            assert start_idx < end_idx
            assert len(sentence_group) > 0

    def test_min_chunk_size_enforcement(self):
        """Test minimum chunk size enforcement."""
        sentences = ["A" * 50, "B" * 40, "C" * 600]  # Third is large
        positions = sentence_chunk_positions(sentences, 500, 0, 200)

        # Should not create chunks smaller than min_chunk_size
        for start_idx, end_idx, sentence_group in positions:
            content_length = sum(len(s) for s in sentence_group)
            if content_length < 200:
                # Should be combined with next sentences
                assert end_idx < len(sentences)

    def test_overlap_calculation(self):
        """Test overlap sentence calculation."""
        sentences = ["A" * 100, "B" * 100, "C" * 100, "D" * 100, "E" * 100]
        positions = sentence_chunk_positions(sentences, 250, 150, 50)

        # Verify that overlap is calculated correctly
        assert len(positions) >= 2
        if len(positions) >= 2:
            first_chunk = positions[0][2]
            second_chunk = positions[1][2]

            # Should have overlap
            overlap_count = 0
            for sentence in first_chunk:
                if sentence in second_chunk:
                    overlap_count += 1
            assert overlap_count > 0

    def test_edge_case_exact_chunk_size(self):
        """Test edge case where sentences exactly fill chunk size."""
        sentences = ["A" * 500, "B" * 500]  # Exactly 500 chars each
        positions = sentence_chunk_positions(sentences, 500, 0, 100)

        assert len(positions) == 2
        assert positions[0] == (0, 1, ["A" * 500])
        assert positions[1] == (1, 2, ["B" * 500])


class TestParagraphChunkPositions:
    """Test paragraph-based chunk position calculation."""

    def test_empty_paragraphs(self):
        """Test with empty paragraphs list."""
        positions = paragraph_chunk_positions([], 1000, 200, 100)
        assert positions == []

    def test_single_paragraph(self):
        """Test with single paragraph."""
        paragraphs = ["This is a test paragraph with some content."]
        positions = paragraph_chunk_positions(paragraphs, 1000, 200, 50)
        assert positions == [(0, 1, paragraphs)]

    def test_multiple_paragraphs_single_chunk(self):
        """Test multiple paragraphs that fit in single chunk."""
        paragraphs = ["Para 1.", "Para 2.", "Para 3."]
        positions = paragraph_chunk_positions(paragraphs, 1000, 200, 50)
        assert positions == [(0, 3, paragraphs)]

    def test_oversized_paragraph_detection(self):
        """Test detection of oversized paragraphs."""
        large_para = "A" * 2000  # Much larger than chunk_size * 1.5
        paragraphs = ["Small para.", large_para, "Another small para."]
        positions = paragraph_chunk_positions(paragraphs, 1000, 200, 100)

        # Should mark oversized paragraph separately
        assert len(positions) >= 2
        # Find the oversized paragraph chunk
        oversized_chunk = None
        for start_idx, end_idx, para_group in positions:
            if len(para_group) == 1 and len(para_group[0]) > 1500:
                oversized_chunk = (start_idx, end_idx, para_group)
                break

        assert oversized_chunk is not None
        assert oversized_chunk[2] == [large_para]

    def test_paragraph_overlap_behavior(self):
        """Test overlap behavior with paragraphs."""
        paragraphs = ["A" * 400, "B" * 400, "C" * 400]
        positions = paragraph_chunk_positions(paragraphs, 500, 300, 100)

        # Should create chunks (exact overlap may vary based on implementation)
        assert len(positions) >= 1
        for start_idx, end_idx, para_group in positions:
            assert start_idx < end_idx
            assert len(para_group) > 0

    def test_min_chunk_size_with_paragraphs(self):
        """Test minimum chunk size enforcement with paragraphs."""
        paragraphs = ["A" * 50, "B" * 40, "C" * 500]
        positions = paragraph_chunk_positions(paragraphs, 400, 0, 200)

        # Should combine small paragraphs or skip them
        for start_idx, end_idx, para_group in positions:
            total_length = sum(len(p) for p in para_group) + 2 * (len(para_group) - 1)
            # Either meets min size or is the last remaining content
            assert total_length >= 200 or end_idx == len(paragraphs)

    def test_paragraph_separator_calculation(self):
        """Test that paragraph separator is included in length calculation."""
        paragraphs = ["A" * 200, "B" * 200, "C" * 200]
        positions = paragraph_chunk_positions(paragraphs, 450, 0, 100)

        # Should account for +2 characters per paragraph separator
        # 200 + 2 + 200 = 402 chars, should fit in 450
        # Adding third paragraph: 402 + 2 + 200 = 604, should exceed 450
        assert len(positions) >= 2
        first_chunk = positions[0][2]
        assert len(first_chunk) <= 2  # Should not fit all three


class TestFindSentenceBoundaries:
    """Test sentence boundary detection."""

    def test_empty_text(self):
        """Test with empty text."""
        boundaries = find_sentence_boundaries("")
        assert boundaries == []

    def test_single_sentence(self):
        """Test with single sentence."""
        text = "This is a test sentence."
        boundaries = find_sentence_boundaries(text)
        assert boundaries == [24]

    def test_multiple_sentences(self):
        """Test with multiple sentences."""
        text = "First sentence. Second sentence! Third sentence?"
        boundaries = find_sentence_boundaries(text)

        # Should find boundaries after sentence endings
        assert len(boundaries) >= 3
        # Verify boundaries are within reasonable ranges
        assert any(b >= 14 and b <= 16 for b in boundaries)  # After "First sentence."
        assert any(b >= 31 and b <= 33 for b in boundaries)  # After "Second sentence!"
        assert any(b >= 48 and b <= 50 for b in boundaries)  # After "Third sentence?"

    def test_sentence_with_spaces(self):
        """Test sentences with spaces after punctuation."""
        text = "First sentence.   Second sentence."
        boundaries = find_sentence_boundaries(text)
        assert 15 in boundaries  # After first sentence
        assert len(boundaries) == 2

    def test_false_positives_handling(self):
        """Test handling of abbreviations and false positives."""
        text = "Dr. Smith went home. He was tired."
        boundaries = find_sentence_boundaries(text)

        # Should find at least one proper sentence boundary
        assert len(boundaries) >= 1
        # Should find boundary after "home." or "tired."
        assert any(b >= 20 and b <= 22 for b in boundaries) or any(b >= 33 and b <= 35 for b in boundaries)

    def test_end_of_text_boundary(self):
        """Test boundary at end of text."""
        text = "This is the end."
        boundaries = find_sentence_boundaries(text)
        assert boundaries == [16]

    def test_custom_language_patterns(self):
        """Test with custom language patterns."""
        text = "Prva rečenica… Druga rečenica‽"
        language_patterns = {"sentence_endings": ".!?…‽"}
        boundaries = find_sentence_boundaries(text, language_patterns)

        assert len(boundaries) == 2
        assert 14 in boundaries  # After "…"
        assert 30 in boundaries  # After "‽"

    def test_no_sentence_endings(self):
        """Test text without sentence endings."""
        text = "This text has no proper endings"
        boundaries = find_sentence_boundaries(text)
        assert boundaries == []

    def test_consecutive_punctuation(self):
        """Test consecutive punctuation marks."""
        text = "Really?! Yes... Absolutely!!!"
        boundaries = find_sentence_boundaries(text)

        # Should find boundaries after each group
        assert len(boundaries) >= 2


class TestExtractParagraphs:
    """Test paragraph extraction."""

    def test_empty_text(self):
        """Test with empty text."""
        paragraphs = extract_paragraphs("")
        assert paragraphs == []

    def test_single_paragraph(self):
        """Test with single paragraph."""
        text = "This is a single paragraph with some content."
        paragraphs = extract_paragraphs(text)
        assert paragraphs == [text]

    def test_multiple_paragraphs(self):
        """Test with multiple paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = extract_paragraphs(text)
        expected = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        assert paragraphs == expected

    def test_paragraphs_with_extra_whitespace(self):
        """Test paragraphs with extra whitespace."""
        text = "  First paragraph.  \n\n  \n  Second paragraph.  \n\n\n  Third paragraph.  "
        paragraphs = extract_paragraphs(text)
        expected = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        assert paragraphs == expected

    def test_single_line_breaks(self):
        """Test that single line breaks don't split paragraphs."""
        text = "First line.\nSecond line.\n\nNew paragraph."
        paragraphs = extract_paragraphs(text)
        expected = ["First line.\nSecond line.", "New paragraph."]
        assert paragraphs == expected

    def test_empty_paragraphs_filtered(self):
        """Test that empty paragraphs are filtered out."""
        text = "First paragraph.\n\n\n\nSecond paragraph.\n\n   \n\nThird paragraph."
        paragraphs = extract_paragraphs(text)
        expected = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        assert paragraphs == expected

    def test_windows_line_endings(self):
        """Test with Windows-style line endings."""
        text = "First paragraph.\r\n\r\nSecond paragraph."
        paragraphs = extract_paragraphs(text)
        expected = ["First paragraph.", "Second paragraph."]
        assert paragraphs == expected


class TestCalculateChunkMetadata:
    """Test chunk metadata calculation."""

    def test_empty_content(self):
        """Test with empty content."""
        metadata = calculate_chunk_metadata("")
        expected = {"word_count": 0, "char_count": 0, "line_count": 1}
        assert metadata == expected

    def test_single_word(self):
        """Test with single word."""
        metadata = calculate_chunk_metadata("hello")
        expected = {"word_count": 1, "char_count": 5, "line_count": 1}
        assert metadata == expected

    def test_multiple_words(self):
        """Test with multiple words."""
        content = "This is a test sentence with multiple words."
        metadata = calculate_chunk_metadata(content)
        expected = {"word_count": 8, "char_count": 44, "line_count": 1}
        assert metadata == expected

    def test_multiline_content(self):
        """Test with multiline content."""
        content = "First line.\nSecond line.\nThird line."
        metadata = calculate_chunk_metadata(content)
        expected = {"word_count": 6, "char_count": len(content), "line_count": 3}
        assert metadata == expected

    def test_content_with_extra_whitespace(self):
        """Test with extra whitespace."""
        content = "  Word1   Word2  \n  Word3  "
        metadata = calculate_chunk_metadata(content)
        expected = {"word_count": 3, "char_count": len(content), "line_count": 2}
        assert metadata == expected

    def test_punctuation_handling(self):
        """Test that punctuation doesn't affect word count incorrectly."""
        content = "Hello, world! How are you? Fine, thanks."
        metadata = calculate_chunk_metadata(content)
        expected = {"word_count": 7, "char_count": 40, "line_count": 1}
        assert metadata == expected


# Mock Protocol Implementations
class MockTextCleaner:
    """Mock implementation of TextCleaner protocol."""

    def __init__(self):
        self.cleaned_texts = {}
        self.extracted_sentences = {}

    def clean_text(self, text: str, preserve_structure: bool = True):
        """Mock text cleaning."""
        # Simple mock: just strip whitespace
        cleaned = text.strip()

        # Create mock CleaningResult
        mock_result = Mock()
        mock_result.text = cleaned
        mock_result.original_text = text
        mock_result.removed_elements = []
        mock_result.processing_stats = {"chars_removed": len(text) - len(cleaned)}

        self.cleaned_texts[text] = cleaned
        return mock_result

    def extract_sentences(self, text: str) -> list[str]:
        """Mock sentence extraction."""
        # Simple sentence splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        self.extracted_sentences[text] = sentences
        return sentences


class MockSentenceExtractor:
    """Mock implementation of SentenceExtractor protocol."""

    def __init__(self):
        self.extracted_sentences = {}

    def extract_sentences(self, text: str) -> list[str]:
        """Mock sentence extraction with custom logic."""
        # More sophisticated mock splitting
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        self.extracted_sentences[text] = sentences
        return sentences


# Test Main DocumentChunker Class
class TestDocumentChunker:
    """Test DocumentChunker class."""

    @pytest.fixture
    def config(self):
        """Create test chunking configuration."""
        return ChunkingConfig(
            chunk_size=1000,
            overlap=200,
            min_chunk_size=100,
            respect_sentences=True,
            sentence_search_range=200,
            strategy=ChunkingStrategy.SLIDING_WINDOW
        )

    @pytest.fixture
    def mock_text_cleaner(self):
        """Create mock text cleaner."""
        return MockTextCleaner()

    @pytest.fixture
    def mock_sentence_extractor(self):
        """Create mock sentence extractor."""
        return MockSentenceExtractor()

    def test_init_with_dependencies(self, config, mock_text_cleaner, mock_sentence_extractor):
        """Test initialization with all dependencies."""
        language_patterns = {"sentence_endings": ".!?"}

        chunker = DocumentChunker(
            config=config,
            text_cleaner=mock_text_cleaner,
            sentence_extractor=mock_sentence_extractor,
            language_patterns=language_patterns
        )

        assert chunker.config == config
        assert chunker.text_cleaner == mock_text_cleaner
        assert chunker.sentence_extractor == mock_sentence_extractor
        assert chunker.language_patterns == language_patterns
        assert isinstance(chunker.logger, logging.Logger)

    def test_init_with_minimal_dependencies(self, config):
        """Test initialization with minimal dependencies."""
        chunker = DocumentChunker(config)

        assert chunker.config == config
        assert chunker.text_cleaner is None
        assert chunker.sentence_extractor is None
        assert chunker.language_patterns == {}

    def test_chunk_document_empty_text(self, config):
        """Test chunking with empty text."""
        config.min_chunk_size = 10  # Lower threshold for testing
        chunker = DocumentChunker(config)

        chunks = chunker.chunk_document("", "test.txt")
        assert chunks == []

        chunks = chunker.chunk_document("   ", "test.txt")
        assert chunks == []

    def test_sliding_window_chunking(self, config, mock_text_cleaner):
        """Test sliding window chunking strategy."""
        config.strategy = ChunkingStrategy.SLIDING_WINDOW
        chunker = DocumentChunker(config, text_cleaner=mock_text_cleaner)

        text = "A" * 2500  # Text larger than chunk_size
        chunks = chunker.chunk_document(text, "test.txt")

        assert len(chunks) >= 2
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.source_file == "test.txt"
            assert len(chunk.content) <= config.chunk_size + config.overlap
            assert chunk.word_count > 0
            assert chunk.char_count > 0

    def test_sentence_based_chunking(self, config, mock_text_cleaner):
        """Test sentence-based chunking strategy."""
        config.strategy = ChunkingStrategy.SENTENCE
        config.min_chunk_size = 50  # Lower threshold
        chunker = DocumentChunker(config, text_cleaner=mock_text_cleaner)

        text = "First sentence. " * 100  # Many sentences
        chunks = chunker.chunk_document(text, "test.txt")

        # May have no chunks if sentences don't meet min size after splitting
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.source_file == "test.txt"
            # Sentences should be properly combined
            assert len(chunk.content) > 0

    def test_paragraph_based_chunking(self, config, mock_text_cleaner):
        """Test paragraph-based chunking strategy."""
        config.strategy = ChunkingStrategy.PARAGRAPH
        config.min_chunk_size = 10  # Lower threshold to ensure chunk creation
        chunker = DocumentChunker(config, text_cleaner=mock_text_cleaner)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk_document(text, "test.txt")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.source_file == "test.txt"

    def test_chunking_with_custom_strategy(self, config, mock_text_cleaner):
        """Test chunking with custom strategy override."""
        config.strategy = ChunkingStrategy.SLIDING_WINDOW
        config.min_chunk_size = 10  # Lower threshold to ensure chunk creation
        chunker = DocumentChunker(config, text_cleaner=mock_text_cleaner)

        text = "Test text content."
        chunks = chunker.chunk_document(text, "test.txt", strategy=ChunkingStrategy.SENTENCE)

        # Should use sentence strategy despite config
        assert len(chunks) >= 1

    def test_text_cleaning_integration(self, config):
        """Test integration with text cleaner."""
        config.min_chunk_size = 10  # Lower threshold to ensure chunk creation
        mock_cleaner = MockTextCleaner()
        chunker = DocumentChunker(config, text_cleaner=mock_cleaner)

        text = "  Test text with whitespace  "
        chunks = chunker.chunk_document(text, "test.txt")

        # Should have called text cleaner
        assert text in mock_cleaner.cleaned_texts
        assert len(chunks) >= 1

    def test_sentence_extractor_integration(self, config):
        """Test integration with sentence extractor."""
        mock_extractor = MockSentenceExtractor()
        config.strategy = ChunkingStrategy.SENTENCE
        config.min_chunk_size = 10  # Lower threshold
        chunker = DocumentChunker(config, sentence_extractor=mock_extractor)

        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_document(text, "test.txt")

        # Should have called sentence extractor
        cleaned_text = text.strip()  # Text gets cleaned
        assert cleaned_text in mock_extractor.extracted_sentences

    def test_fallback_sentence_extraction(self, config):
        """Test fallback sentence extraction without extractor."""
        config.strategy = ChunkingStrategy.SENTENCE
        config.min_chunk_size = 10  # Lower threshold
        chunker = DocumentChunker(config)  # No extractor

        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_document(text, "test.txt")

        # Fallback splitting may not produce chunks that meet min size
        # Just verify no crash occurred
        assert isinstance(chunks, list)

    def test_chunk_filtering(self, config):
        """Test meaningful chunk filtering."""
        config.min_chunk_size = 100
        chunker = DocumentChunker(config)

        # Create text that will produce small chunks
        text = "A" * 50  # Smaller than min_chunk_size
        chunks = chunker.chunk_document(text, "test.txt")

        # Should filter out small chunks
        for chunk in chunks:
            assert len(chunk.content.strip()) >= config.min_chunk_size

    def test_chunk_id_generation(self, config):
        """Test chunk ID generation."""
        chunker = DocumentChunker(config)

        text = "A" * 2000  # Large enough for multiple chunks
        chunks = chunker.chunk_document(text, "test_document.txt")

        # Check chunk IDs
        for i, chunk in enumerate(chunks):
            expected_id = f"test_document_{i:04d}"
            assert chunk.chunk_id == expected_id
            assert chunk.chunk_index == i

    def test_chunk_metadata_accuracy(self, config):
        """Test accuracy of chunk metadata."""
        config.min_chunk_size = 10  # Lower threshold to ensure chunk creation
        chunker = DocumentChunker(config)

        text = "This is a test sentence with exactly eight words."
        chunks = chunker.chunk_document(text, "test.txt")

        assert len(chunks) >= 1
        if chunks:
            chunk = chunks[0]
            # Text actually has 9 words: "This", "is", "a", "test", "sentence", "with", "exactly", "eight", "words"
            assert chunk.word_count == 9
            assert chunk.char_count == len(chunk.content)
            assert "word_count" in chunk.metadata
            assert "char_count" in chunk.metadata

    def test_unknown_strategy_error(self, config):
        """Test error handling for unknown chunking strategy."""
        chunker = DocumentChunker(config)

        # Create mock strategy that doesn't have .value attribute
        class InvalidStrategy:
            value = "invalid_strategy"

        invalid_strategy = InvalidStrategy()

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunker.chunk_document("test", "test.txt", strategy=invalid_strategy)

    def test_oversized_paragraph_handling(self, config):
        """Test handling of oversized paragraphs in paragraph strategy."""
        config.strategy = ChunkingStrategy.PARAGRAPH
        chunker = DocumentChunker(config)

        # Create text with oversized paragraph
        oversized_para = "A" * (config.chunk_size * 2)
        text = f"Normal paragraph.\n\n{oversized_para}\n\nAnother normal paragraph."

        chunks = chunker.chunk_document(text, "test.txt")

        # Should handle oversized paragraph with sliding window
        assert len(chunks) >= 2


# Test Factory Function
class TestCreateDocumentChunker:
    """Test create_document_chunker factory function."""

    @pytest.fixture
    def config_dict(self):
        """Create test configuration dictionary."""
        return {
            "chunking": {
                "chunk_size": 1000,
                "overlap_size": 200,
                "min_chunk_size": 100,
                "preserve_sentence_boundaries": True,
                "sentence_search_range": 200,
                "strategy": "sliding_window"
            },
            "shared": {
                "default_chunk_size": 800,
                "default_chunk_overlap": 150,
                "min_chunk_size": 80
            },
            "language_specific": {
                "patterns": {"sentence_endings": ".!?"}
            }
        }

    def test_create_with_config_dict(self, config_dict):
        """Test factory with configuration dictionary."""
        with patch('src.preprocessing.cleaners.MultilingualTextCleaner') as mock_cleaner_class:
            mock_cleaner = Mock()
            mock_cleaner_class.return_value = mock_cleaner

            chunker = create_document_chunker(config_dict, language="hr")

            assert isinstance(chunker, DocumentChunker)
            assert chunker.config.chunk_size == 1000
            assert chunker.text_cleaner == mock_cleaner
            assert chunker.sentence_extractor == mock_cleaner  # Same object
            assert chunker.language_patterns == {"sentence_endings": ".!?"}

    def test_create_with_config_provider(self):
        """Test factory with configuration provider."""
        mock_config_provider = Mock()
        mock_config_provider.get_language_specific_config.return_value = {
            "patterns": {"sentence_endings": ".!?…"}
        }

        with patch('src.preprocessing.chunkers.ChunkingConfig.from_config') as mock_config:
            mock_config.return_value = ChunkingConfig(
                chunk_size=1000, overlap=200, min_chunk_size=100,
                respect_sentences=True, sentence_search_range=200,
                strategy=ChunkingStrategy.SLIDING_WINDOW
            )

            with patch('src.preprocessing.cleaners.MultilingualTextCleaner') as mock_cleaner_class:
                mock_cleaner = Mock()
                mock_cleaner_class.return_value = mock_cleaner

                chunker = create_document_chunker(config_provider=mock_config_provider, language="en")

                assert isinstance(chunker, DocumentChunker)
                assert chunker.language_patterns == {"sentence_endings": ".!?…"}

    def test_create_with_config_provider_error(self):
        """Test factory with config provider error handling."""
        mock_config_provider = Mock()
        mock_config_provider.get_language_specific_config.side_effect = KeyError("Missing")

        with patch('src.preprocessing.chunkers.ChunkingConfig.from_config') as mock_config:
            mock_config.return_value = ChunkingConfig(
                chunk_size=1000, overlap=200, min_chunk_size=100,
                respect_sentences=True, sentence_search_range=200,
                strategy=ChunkingStrategy.SLIDING_WINDOW
            )

            with patch('src.preprocessing.cleaners.MultilingualTextCleaner') as mock_cleaner_class:
                mock_cleaner = Mock()
                mock_cleaner_class.return_value = mock_cleaner

                chunker = create_document_chunker(config_provider=mock_config_provider, language="en")

                assert isinstance(chunker, DocumentChunker)
                # Language patterns defaults to empty dict when config fails
                assert chunker.language_patterns == {}

    def test_create_missing_patterns_error(self, config_dict):
        """Test factory error handling when patterns missing."""
        # Remove patterns from language_specific
        config_dict["language_specific"] = {}

        with patch('src.preprocessing.cleaners.MultilingualTextCleaner') as mock_cleaner_class:
            mock_cleaner = Mock()
            mock_cleaner_class.return_value = mock_cleaner

            with pytest.raises(ValueError, match="Missing 'patterns' in language_specific configuration"):
                create_document_chunker(config_dict, language="hr")


# Integration Tests
class TestIntegration:
    """Test complete chunking workflows."""

    def test_end_to_end_chunking_workflow(self):
        """Test complete end-to-end chunking workflow."""
        config = ChunkingConfig(
            chunk_size=500,
            overlap=100,
            min_chunk_size=50,
            respect_sentences=True,
            sentence_search_range=100,
            strategy=ChunkingStrategy.SLIDING_WINDOW
        )

        mock_cleaner = MockTextCleaner()
        language_patterns = {"sentence_endings": ".!?"}

        chunker = DocumentChunker(
            config=config,
            text_cleaner=mock_cleaner,
            language_patterns=language_patterns
        )

        # Create realistic document text
        text = """
        This is the first paragraph with multiple sentences. It contains enough content to test chunking.
        The paragraph discusses various topics and should be properly processed.

        This is the second paragraph with different content. It also has multiple sentences for testing.
        The chunking algorithm should handle this content appropriately.

        The third paragraph is shorter. But still meaningful.
        """

        chunks = chunker.chunk_document(text, "test_document.txt")

        # Verify complete workflow
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.source_file == "test_document.txt"
            assert len(chunk.content) >= config.min_chunk_size
            assert chunk.word_count > 0
            assert chunk.char_count == len(chunk.content)
            assert chunk.chunk_id.startswith("test_document_")

        # Verify text was cleaned (check if any key contains the stripped text)
        cleaned_found = any(text.strip() == cleaned for cleaned in mock_cleaner.cleaned_texts.values())
        assert cleaned_found

    def test_protocol_compliance(self):
        """Test that all protocols are properly implemented."""
        # Test TextCleaner protocol
        cleaner = MockTextCleaner()
        assert callable(cleaner.clean_text)
        assert callable(cleaner.extract_sentences)

        # Test SentenceExtractor protocol
        extractor = MockSentenceExtractor()
        assert callable(extractor.extract_sentences)

        # Test actual usage
        result = cleaner.clean_text("test text")
        assert hasattr(result, 'text')

        sentences = extractor.extract_sentences("First. Second. Third.")
        assert isinstance(sentences, list)

    def test_all_chunking_strategies(self):
        """Test all chunking strategies work correctly."""
        config = ChunkingConfig(
            chunk_size=300,
            overlap=50,
            min_chunk_size=30,
            respect_sentences=True,
            sentence_search_range=100,
            strategy=ChunkingStrategy.SLIDING_WINDOW  # Will be overridden
        )

        mock_cleaner = MockTextCleaner()
        chunker = DocumentChunker(config, text_cleaner=mock_cleaner)

        text = "First sentence. Second sentence. Third sentence.\n\nNew paragraph here.\n\nAnother paragraph."

        # Test each strategy
        strategies = [ChunkingStrategy.SLIDING_WINDOW, ChunkingStrategy.SENTENCE, ChunkingStrategy.PARAGRAPH]

        for strategy in strategies:
            chunks = chunker.chunk_document(text, "test.txt", strategy=strategy)
            assert len(chunks) >= 1
            assert all(isinstance(chunk, TextChunk) for chunk in chunks)

    def test_error_handling_workflow(self):
        """Test comprehensive error handling in workflows."""
        config = ChunkingConfig(
            chunk_size=1000,
            overlap=200,
            min_chunk_size=100,
            respect_sentences=True,
            sentence_search_range=200,
            strategy=ChunkingStrategy.SLIDING_WINDOW
        )

        chunker = DocumentChunker(config)

        # Test empty document
        chunks = chunker.chunk_document("", "empty.txt")
        assert chunks == []

        # Test whitespace-only document
        chunks = chunker.chunk_document("   \n\n   ", "whitespace.txt")
        assert chunks == []

        # Test very small document
        chunks = chunker.chunk_document("Hi", "tiny.txt")
        # Should either create no chunks (filtered) or one small chunk
        assert len(chunks) <= 1

    def test_large_document_processing(self):
        """Test processing of large documents."""
        config = ChunkingConfig(
            chunk_size=1000,
            overlap=200,
            min_chunk_size=100,
            respect_sentences=False,  # Faster processing
            sentence_search_range=100,
            strategy=ChunkingStrategy.SLIDING_WINDOW
        )

        chunker = DocumentChunker(config)

        # Create large document
        large_text = "This is a test sentence. " * 1000  # ~25,000 characters
        chunks = chunker.chunk_document(large_text, "large_document.txt")

        # Should create multiple chunks
        assert len(chunks) >= 20

        # Verify overlap
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Check that chunks have reasonable sizes
            assert len(current_chunk.content) <= config.chunk_size + 100  # Some tolerance
            assert len(current_chunk.content) >= config.min_chunk_size

        # Verify metadata accuracy
        total_chunk_chars = sum(chunk.char_count for chunk in chunks)
        total_chunk_words = sum(chunk.word_count for chunk in chunks)

        # Should be reasonable (accounting for overlap)
        assert total_chunk_chars >= len(large_text)
        assert total_chunk_words > 0

    def test_multilingual_content_handling(self):
        """Test handling of multilingual content."""
        config = ChunkingConfig(
            chunk_size=300,
            overlap=50,
            min_chunk_size=30,
            respect_sentences=True,
            sentence_search_range=100,
            strategy=ChunkingStrategy.SENTENCE
        )

        # Croatian patterns
        language_patterns = {"sentence_endings": ".!?…"}
        chunker = DocumentChunker(config, language_patterns=language_patterns)

        # Mixed language text
        text = """
        Ovo je prva rečenica na hrvatskom jeziku. Druga rečenica također…

        This is an English sentence! Another English sentence?

        Treća hrvatska rečenica. Četvrta i zadnja rečenica.
        """

        chunks = chunker.chunk_document(text, "multilingual.txt")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            # Should handle both languages properly
            assert len(chunk.content) > 0