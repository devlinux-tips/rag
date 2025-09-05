"""
Multilingual document preprocessing pipeline for RAG.
"""

from .chunkers import DocumentChunker, TextChunk, chunk_document
from .cleaners import (
    MultilingualTextCleaner,
    detect_language_content,
    preserve_text_encoding,
    setup_language_environment,
)
from .extractors import DocumentExtractor, extract_document_text

__all__ = [
    # Primary multilingual API
    "DocumentExtractor",
    "extract_document_text",
    "MultilingualTextCleaner",
    "detect_language_content",
    "preserve_text_encoding",
    "setup_language_environment",
    "DocumentChunker",
    "TextChunk",
    "chunk_document",
]
