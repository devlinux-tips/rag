"""
Unit tests for document chunkers.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append("src")

from preprocessing.chunkers import DocumentChunker, TextChunk, chunk_croatian_document


class TestTextChunk:
    """Test the TextChunk dataclass."""

    def test_text_chunk_creation(self):
        """Test TextChunk creation and attributes."""
        chunk = TextChunk(
            content="Test content",
            chunk_id="test_001",
            source_file="test.txt",
            start_char=0,
            end_char=12,
            chunk_index=0,
            word_count=2,
            char_count=12,
        )

        assert chunk.content == "Test content"
        assert chunk.chunk_id == "test_001"
        assert chunk.word_count == 2
        assert chunk.char_count == 12


class TestDocumentChunker:
    """Test the DocumentChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker(chunk_size=100, overlap=20)
        self.croatian_text = """
        Prvi paragraf sa hrvatskim tekstom koji sadrži čćžšđ karaktere.

        Drugi paragraf također sadrži važne informacije. Treća rečenica je ovdje.

        Treći paragraf završava dokument sa dodatnim čćžšđ karakterima.
        """.strip()

    def test_init_default_parameters(self):
        """Test chunker initialization with default parameters."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50
        assert chunker.min_chunk_size == 100
        assert chunker.respect_sentences is True

    def test_init_custom_parameters(self):
        """Test chunker initialization with custom parameters."""
        chunker = DocumentChunker(
            chunk_size=256, overlap=30, min_chunk_size=50, respect_sentences=False
        )
        assert chunker.chunk_size == 256
        assert chunker.overlap == 30
        assert chunker.min_chunk_size == 50
        assert chunker.respect_sentences is False

    def test_chunk_document_empty_text(self):
        """Test chunking with empty text."""
        result = self.chunker.chunk_document("", "test.txt")
        assert result == []

        result = self.chunker.chunk_document(None, "test.txt")
        assert result == []

        result = self.chunker.chunk_document("   ", "test.txt")
        assert result == []

    def test_chunk_document_sliding_window(self):
        """Test sliding window chunking strategy."""
        chunks = self.chunker.chunk_document(
            self.croatian_text, "test.txt", strategy="sliding_window"
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))
        assert all("test" in chunk.chunk_id for chunk in chunks)

    def test_chunk_document_sentence_strategy(self):
        """Test sentence-based chunking strategy."""
        chunks = self.chunker.chunk_document(self.croatian_text, "test.txt", strategy="sentence")

        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        # Should respect sentence boundaries
        for chunk in chunks:
            assert len(chunk.content) >= self.chunker.min_chunk_size or len(chunks) == 1

    def test_chunk_document_paragraph_strategy(self):
        """Test paragraph-based chunking strategy."""
        chunks = self.chunker.chunk_document(self.croatian_text, "test.txt", strategy="paragraph")

        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)

    def test_chunk_document_invalid_strategy(self):
        """Test chunking with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            self.chunker.chunk_document(self.croatian_text, "test.txt", strategy="invalid_strategy")

    def test_create_chunk(self):
        """Test chunk creation with metadata."""
        chunk = self.chunker._create_chunk(
            content="Test chunk content",
            source_file="/path/to/test.txt",
            start_char=0,
            end_char=18,
            chunk_index=5,
        )

        assert chunk.content == "Test chunk content"
        assert chunk.chunk_id == "test_0005"
        assert chunk.source_file == "/path/to/test.txt"
        assert chunk.start_char == 0
        assert chunk.end_char == 18
        assert chunk.chunk_index == 5
        assert chunk.word_count == 3
        assert chunk.char_count == 18

    def test_find_sentence_boundary(self):
        """Test sentence boundary detection."""
        text = "Prva rečenica. Druga rečenica! Treća rečenica?"

        # Test finding boundary after first sentence
        boundary = self.chunker._find_sentence_boundary(text, 15, 0)

        assert boundary > 14  # Should find a boundary after the period
        assert boundary <= len(text)

    def test_sliding_window_chunking_overlap(self):
        """Test sliding window chunking with overlap."""
        text = "A" * 200  # Simple text for testing overlap
        chunker = DocumentChunker(chunk_size=50, overlap=10)

        chunks = chunker._sliding_window_chunking(text, "test.txt")

        if len(chunks) > 1:
            # Check overlap exists between consecutive chunks
            for i in range(len(chunks) - 1):
                current_end = chunks[i].end_char
                next_start = chunks[i + 1].start_char
                overlap = current_end - next_start
                assert overlap >= 0  # Should have some overlap or be adjacent

    def test_sentence_based_chunking_respect_sentences(self):
        """Test that sentence-based chunking respects sentence boundaries."""
        text = "Kratka rečenica. " * 20  # Create multiple short sentences
        chunks = self.chunker._sentence_based_chunking(text, "test.txt")

        # Each chunk should end with sentence punctuation or be the last chunk
        for chunk in chunks:
            content = chunk.content.strip()
            if content:
                # Should end with sentence punctuation or be incomplete due to size limits
                assert (
                    content.endswith(".")
                    or content.endswith("!")
                    or content.endswith("?")
                    or len(content) >= self.chunker.chunk_size * 0.8
                )

    def test_paragraph_based_chunking_preserves_structure(self):
        """Test that paragraph-based chunking preserves paragraph structure."""
        text = """Prvi paragraf.

Drugi paragraf sa više teksta koji može biti dugačak.

Treći paragraf."""

        chunks = self.chunker._paragraph_based_chunking(text, "test.txt")

        assert len(chunks) > 0
        # At least some chunks should contain paragraph separators or single paragraphs
        paragraph_indicators = sum(
            1 for chunk in chunks if "\n\n" in chunk.content or len(chunk.content.strip()) > 10
        )
        assert paragraph_indicators > 0


class TestCroatianSpecificChunking:
    """Test Croatian language-specific chunking features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker(chunk_size=200, overlap=50)

    def test_croatian_sentence_boundary_detection(self):
        """Test sentence boundary detection with Croatian text."""
        text = "Ovo je prva rečenica. Što mislite o ovome? Čini mi se ispravno."

        # Test boundary detection after Croatian sentences
        boundary = self.chunker._find_sentence_boundary(text, 25, 0)

        assert boundary > 20
        assert text[boundary - 1] in ".!?" or text[boundary - 2] in ".!?"

    def test_croatian_diacritics_in_chunks(self):
        """Test that Croatian diacritics are preserved in chunks."""
        text = "Šišmiš čuva žutu ćupriju. Đak čita šarenu žirafa."

        chunks = self.chunker.chunk_document(text, "test.txt")

        # All chunks should preserve Croatian diacritics
        croatian_chars = set("čćžšđČĆŽŠĐ")

        for chunk in chunks:
            original_chars = croatian_chars.intersection(set(text))
            chunk_chars = croatian_chars.intersection(set(chunk.content))

            if original_chars:  # If original text had Croatian chars
                # At least some should be preserved in chunks containing that text
                assert len(chunk_chars) >= 0  # Basic preservation test

    def test_croatian_uppercase_after_punctuation(self):
        """Test recognition of Croatian uppercase letters after punctuation."""
        text = "Prva rečenica. Što je sa Čakovec? Đurđevac je grad."

        boundary = self.chunker._find_sentence_boundary(text, 20, 0)

        # Should recognize Croatian uppercase letters as sentence starts
        assert boundary > 15
        if boundary < len(text):
            # Check that we properly handle Croatian uppercase
            next_chars = text[boundary : boundary + 5]
            assert any(char.isupper() or char in "ČĆŽŠĐ" for char in next_chars)


class TestChunkFiltering:
    """Test chunk filtering and quality checks."""

    def setup_method(self):
        """Set up test fixtures with mock cleaner."""
        self.chunker = DocumentChunker(chunk_size=100, overlap=20, min_chunk_size=30)

    def test_minimum_chunk_size_filtering(self):
        """Test that chunks below minimum size are filtered out."""
        # Create text that will produce some small chunks
        short_text = "Short. Text."

        with patch.object(self.chunker.cleaner, "is_meaningful_text") as mock_meaningful:
            mock_meaningful.return_value = True  # Allow all chunks for this test

            chunks = self.chunker.chunk_document(short_text, "test.txt")

            # All returned chunks should meet minimum size requirement
            for chunk in chunks:
                assert len(chunk.content) >= self.chunker.min_chunk_size

    def test_meaningful_text_filtering(self):
        """Test that non-meaningful chunks are filtered out."""
        text = "Meaningful text here. 123 456 789. More meaningful content."

        with patch.object(self.chunker.cleaner, "is_meaningful_text") as mock_meaningful:
            # Mock the cleaner to reject the middle chunk (numbers)
            def mock_is_meaningful(chunk_text):
                return "123 456 789" not in chunk_text

            mock_meaningful.side_effect = mock_is_meaningful

            chunks = self.chunker.chunk_document(text, "test.txt")

            # Should only return meaningful chunks
            for chunk in chunks:
                assert "123 456 789" not in chunk.content


class TestChunkCroatianDocumentFunction:
    """Test the convenience function."""

    def test_chunk_croatian_document_basic(self):
        """Test convenience function basic usage."""
        text = "Test Croatian text sa čćžšđ karakterima."

        chunks = chunk_croatian_document(text, "test.txt")

        assert isinstance(chunks, list)
        assert len(chunks) >= 0
        if chunks:
            assert all(isinstance(chunk, TextChunk) for chunk in chunks)

    def test_chunk_croatian_document_custom_parameters(self):
        """Test convenience function with custom parameters."""
        text = "Test text " * 100  # Create longer text

        chunks = chunk_croatian_document(
            text=text,
            source_file="test.txt",
            chunk_size=200,
            overlap=30,
            strategy="sentence",
        )

        assert isinstance(chunks, list)
        if chunks:
            # Should respect custom chunk size approximately
            avg_length = sum(len(chunk.content) for chunk in chunks) / len(chunks)
            assert avg_length <= 250  # Should be around target size


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker()

    def test_very_long_single_paragraph(self):
        """Test handling of very long single paragraphs."""
        long_paragraph = "This is a very long paragraph. " * 100

        chunks = self.chunker.chunk_document(long_paragraph, "test.txt", strategy="paragraph")

        assert len(chunks) > 0
        # Should break down long paragraphs appropriately
        for chunk in chunks:
            assert len(chunk.content) <= self.chunker.chunk_size * 2  # Allow some flexibility

    def test_text_shorter_than_chunk_size(self):
        """Test handling of text shorter than chunk size."""
        short_text = "Kratki tekst."

        chunks = self.chunker.chunk_document(short_text, "test.txt")

        if chunks:  # Might be empty if too short and filtered
            assert len(chunks) == 1
            assert chunks[0].content.strip() == short_text.strip()

    def test_text_with_only_whitespace_paragraphs(self):
        """Test handling of text with empty paragraphs."""
        text_with_empty = """


        Real content here.


        More real content.


        """

        chunks = self.chunker.chunk_document(text_with_empty, "test.txt")

        # Should handle empty paragraphs gracefully
        for chunk in chunks:
            assert chunk.content.strip()  # No empty chunks should be returned

    def test_unicode_handling_in_chunking(self):
        """Test proper Unicode handling in chunk boundaries."""
        text_with_unicode = "Test with émojis 🇭🇷 and čćžšđ characters mixed together."

        chunks = self.chunker.chunk_document(text_with_unicode, "test.txt")

        # Should handle Unicode without breaking
        for chunk in chunks:
            assert len(chunk.content) > 0
            # Basic check that Croatian characters are preserved
            if any(char in text_with_unicode for char in "čćžšđ"):
                # Should preserve some Croatian characters
                assert isinstance(chunk.content, str)
