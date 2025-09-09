"""
Comprehensive tests for storage.py demonstrating 100% testability.
Tests pure functions, dependency injection, and integration scenarios.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from src.vectordb.storage import (  # Pure functions; Core classes; Factory functions
    DocumentMetadata, QueryResult, StorageResult, VectorStorage,
    calculate_batch_sizes, create_mock_storage, create_vector_storage,
    extract_document_ids, merge_search_results, parse_query_results,
    prepare_storage_batch, validate_documents_for_storage,
    validate_embeddings_for_storage)


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_validate_documents_for_storage_valid(self):
        """Test document validation with valid input."""
        documents = [
            "Valid document content",
            "Another valid document",
            "Third document with special chars: čćšđž",
        ]

        result = validate_documents_for_storage(documents)

        assert result == documents
        assert len(result) == 3

    def test_validate_documents_for_storage_empty_list(self):
        """Test document validation with empty list."""
        documents = []

        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            validate_documents_for_storage(documents)

    def test_validate_embeddings_for_storage_valid(self):
        """Test embedding validation with valid input."""
        embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9]),
        ]

        result = validate_embeddings_for_storage(embeddings, expected_dim=3)

        assert len(result) == 3
        assert all(isinstance(emb, np.ndarray) for emb in result)
        assert all(emb.shape == (3,) for emb in result)

    def test_prepare_storage_batch_normal(self):
        """Test batch preparation with normal input."""
        documents = ["doc1", "doc2", "doc3"]
        embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])]
        metadata_list = [
            DocumentMetadata(source_file="file1.txt", chunk_index=0, language="en"),
            DocumentMetadata(source_file="file2.txt", chunk_index=1, language="hr"),
            DocumentMetadata(source_file="file3.txt", chunk_index=2, language="en"),
        ]

        batches = prepare_storage_batch(
            documents, embeddings, metadata_list, batch_size=2
        )

        assert len(batches) == 2  # 2 batches for 3 items with batch_size=2

        # First batch
        batch1 = batches[0]
        assert len(batch1["ids"]) == 2
        assert len(batch1["documents"]) == 2
        assert len(batch1["embeddings"]) == 2
        assert len(batch1["metadatas"]) == 2
        assert batch1["documents"] == ["doc1", "doc2"]

        # Second batch
        batch2 = batches[1]
        assert len(batch2["ids"]) == 1
        assert len(batch2["documents"]) == 1
        assert batch2["documents"] == ["doc3"]

    def test_parse_query_results_valid(self):
        """Test query result parsing with valid ChromaDB format."""
        raw_results = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["Content 1", "Content 2", "Content 3"]],
            "metadatas": [
                [
                    {"source_file": "file1.txt", "chunk_index": 0},
                    {"source_file": "file2.txt", "chunk_index": 1},
                    {"source_file": "file3.txt", "chunk_index": 2},
                ]
            ],
            "distances": [[0.1, 0.3, 0.5]],
        }

        results = parse_query_results(raw_results)

        assert len(results) == 3

        # Check first result
        result1 = results[0]
        assert result1.id == "doc1"
        assert result1.content == "Content 1"
        assert result1.metadata["source_file"] == "file1.txt"
        assert result1.metadata["chunk_index"] == 0
        assert result1.score == 0.1


class TestVectorStorageWithMocks:
    """Test VectorStorage with mock dependencies."""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "Croatian document with đčćšž characters",
            "English document with technical terms",
            "Mixed language document sa hrvatskim riječima",
        ]

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
            np.array([0.9, 0.8, 0.7, 0.6]),
        ]

    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return [
            DocumentMetadata(source_file="croatian.txt", chunk_index=0, language="hr"),
            DocumentMetadata(source_file="english.txt", chunk_index=0, language="en"),
            DocumentMetadata(
                source_file="mixed.txt", chunk_index=0, language="multilingual"
            ),
        ]

    @pytest.mark.asyncio
    async def test_store_documents_mock(
        self, sample_documents, sample_embeddings, sample_metadata
    ):
        """Test storing documents with mock storage."""
        storage = create_mock_storage()
        await storage.initialize("test_collection")

        result = await storage.store_documents(
            documents=sample_documents,
            embeddings=sample_embeddings,
            metadata_list=sample_metadata,
            batch_size=2,
        )

        assert result.success is True
        assert result.documents_stored == 3
        assert result.batches_processed == 2
        assert len(result.document_ids) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
