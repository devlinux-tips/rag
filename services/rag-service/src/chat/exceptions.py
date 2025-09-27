"""
RAG Chat Service Exceptions
Custom exceptions for distinguishing between pipeline errors and empty results.
"""


class RAGChatException(Exception):
    """Base exception for RAG chat operations."""
    pass


class RAGPipelineError(RAGChatException):
    """
    Critical RAG pipeline error that should fail hard.

    Raised when there are errors in:
    - Document chunking
    - Vectorization/embedding generation
    - Vector database operations
    - Retrieval system failures
    - RAG system initialization failures
    """
    pass


class RAGEmptyResultsError(RAGChatException):
    """
    RAG query completed successfully but found no matching documents.

    This should return a generic message to the user, not fail hard.
    The query itself worked, but no relevant documents were found.
    """
    pass