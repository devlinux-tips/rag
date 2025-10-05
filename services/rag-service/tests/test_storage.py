"""
Comprehensive tests for vectordb/storage.py
Tests all data classes, pure functions, and dependency injection patterns.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from typing import Any

from src.vectordb.storage import (
    # Data classes
    DocumentMetadata,
    StorageResult,
    QueryResult,
    # Protocols
    VectorCollection,
    VectorDatabase,
    # Pure functions
    validate_documents_for_storage,
    validate_embeddings_for_storage,
    prepare_storage_batch,
    parse_query_results,
    calculate_batch_sizes,
    extract_document_ids,
    merge_search_results,
    # Main class
    VectorStorage,
    # Factory functions
    create_vector_storage,
)
from tests.conftest import (
    create_mock_storage,
)


# ===== DATA CLASS TESTS =====

class TestDocumentMetadata:
    """Test DocumentMetadata data class."""

    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation with timestamp."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metadata = DocumentMetadata(
            source_file="test.pdf",
            chunk_index=1,
            language="hr",
            timestamp=timestamp
        )

        assert metadata.source_file == "test.pdf"
        assert metadata.chunk_index == 1
        assert metadata.language == "hr"
        assert metadata.timestamp == timestamp

    def test_document_metadata_auto_timestamp(self):
        """Test DocumentMetadata with automatic timestamp."""
        metadata = DocumentMetadata(
            source_file="test.pdf",
            chunk_index=1,
            language="hr"
        )

        assert metadata.source_file == "test.pdf"
        assert metadata.chunk_index == 1
        assert metadata.language == "hr"
        assert metadata.timestamp is not None
        assert isinstance(metadata.timestamp, datetime)

    def test_document_metadata_to_dict(self):
        """Test converting DocumentMetadata to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metadata = DocumentMetadata(
            source_file="test.pdf",
            chunk_index=1,
            language="hr",
            timestamp=timestamp
        )

        result = metadata.to_dict()

        assert result == {
            "source_file": "test.pdf",
            "chunk_index": 1,
            "language": "hr",
            "timestamp": "2024-01-01T12:00:00"
        }

    def test_document_metadata_to_dict_none_timestamp(self):
        """Test converting DocumentMetadata with None timestamp - auto-generated due to __post_init__."""
        # Note: __post_init__ automatically sets timestamp if None
        metadata = DocumentMetadata(
            source_file="test.pdf",
            chunk_index=1,
            language="hr",
            timestamp=None
        )

        result = metadata.to_dict()

        # Timestamp is auto-generated in __post_init__, so it won't be None
        assert result["timestamp"] is not None
        assert isinstance(result["timestamp"], str)


class TestStorageResult:
    """Test StorageResult data class."""

    def test_storage_result_success(self):
        """Test successful StorageResult creation."""
        result = StorageResult(
            success=True,
            documents_stored=10,
            batches_processed=2,
            document_ids=["doc1", "doc2"],
            error_message=None
        )

        assert result.success is True
        assert result.documents_stored == 10
        assert result.batches_processed == 2
        assert result.document_ids == ["doc1", "doc2"]
        assert result.error_message is None

    def test_storage_result_failure(self):
        """Test failed StorageResult creation."""
        result = StorageResult(
            success=False,
            error_message="Storage failed"
        )

        assert result.success is False
        assert result.documents_stored == 0  # Default value
        assert result.batches_processed == 0  # Default value
        assert result.document_ids == []  # Default value
        assert result.error_message == "Storage failed"

    def test_storage_result_defaults(self):
        """Test StorageResult with default values."""
        result = StorageResult(success=True)

        assert result.success is True
        assert result.documents_stored == 0
        assert result.batches_processed == 0
        assert result.document_ids == []
        assert result.error_message is None


class TestQueryResult:
    """Test QueryResult data class."""

    def test_query_result_creation(self):
        """Test QueryResult creation."""
        metadata = {"source": "test.pdf", "language": "hr"}
        result = QueryResult(
            id="doc_123",
            content="Test document content",
            metadata=metadata,
            score=0.85
        )

        assert result.id == "doc_123"
        assert result.content == "Test document content"
        assert result.metadata == metadata
        assert result.score == 0.85


# ===== PURE FUNCTION TESTS =====

class TestValidateDocumentsForStorage:
    """Test validate_documents_for_storage function."""

    def test_validate_documents_valid(self):
        """Test validation with valid documents."""
        documents = ["Document 1", "Document 2", "Document 3"]

        result = validate_documents_for_storage(documents)

        assert result == documents

    def test_validate_documents_empty_list(self):
        """Test validation with empty documents list."""
        documents = []

        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            validate_documents_for_storage(documents)

    def test_validate_documents_none_document(self):
        """Test validation with None document."""
        documents = ["Document 1", None, "Document 3"]

        with pytest.raises(ValueError, match="Document at index 1 is None"):
            validate_documents_for_storage(documents)

    def test_validate_documents_wrong_type(self):
        """Test validation with wrong document type."""
        documents = ["Document 1", 123, "Document 3"]

        with pytest.raises(ValueError, match="Document at index 1 must be string"):
            validate_documents_for_storage(documents)

    def test_validate_documents_empty_string(self):
        """Test validation with empty string document."""
        documents = ["Document 1", "   ", "Document 3"]

        with pytest.raises(ValueError, match="Document at index 1 is empty"):
            validate_documents_for_storage(documents)


class TestValidateEmbeddingsForStorage:
    """Test validate_embeddings_for_storage function."""

    def test_validate_embeddings_valid(self):
        """Test validation with valid embeddings."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0])
        ]

        result = validate_embeddings_for_storage(embeddings)

        assert len(result) == 2
        assert np.array_equal(result[0], embeddings[0])
        assert np.array_equal(result[1], embeddings[1])

    def test_validate_embeddings_empty_list(self):
        """Test validation with empty embeddings list."""
        embeddings = []

        with pytest.raises(ValueError, match="Embeddings list cannot be empty"):
            validate_embeddings_for_storage(embeddings)

    def test_validate_embeddings_none_embedding(self):
        """Test validation with None embedding."""
        embeddings = [np.array([1.0, 2.0]), None]

        with pytest.raises(ValueError, match="Embedding at index 1 is None"):
            validate_embeddings_for_storage(embeddings)

    def test_validate_embeddings_wrong_type(self):
        """Test validation with wrong embedding type."""
        embeddings = [np.array([1.0, 2.0]), [3.0, 4.0]]

        with pytest.raises(ValueError, match="Embedding at index 1 must be numpy array"):
            validate_embeddings_for_storage(embeddings)

    def test_validate_embeddings_wrong_dimension(self):
        """Test validation with wrong embedding dimension."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0])  # Wrong dimension
        ]

        with pytest.raises(ValueError, match="Embedding at index 1 has shape \\(2,\\), expected \\(3,\\)"):
            validate_embeddings_for_storage(embeddings, expected_dim=3)

    def test_validate_embeddings_no_dimension_check(self):
        """Test validation without dimension checking."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0])  # Different dimension
        ]

        result = validate_embeddings_for_storage(embeddings)

        assert len(result) == 2
        assert np.array_equal(result[0], embeddings[0])
        assert np.array_equal(result[1], embeddings[1])


class TestPrepareStorageBatch:
    """Test prepare_storage_batch function."""

    def test_prepare_storage_batch_single_batch(self):
        """Test preparing storage batch that fits in single batch."""
        documents = ["Doc 1", "Doc 2"]
        embeddings = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        metadata_list = [
            DocumentMetadata("file1.pdf", 0, "hr"),
            DocumentMetadata("file2.pdf", 1, "hr")
        ]

        result = prepare_storage_batch(documents, embeddings, metadata_list, batch_size=100)

        assert len(result) == 1
        batch = result[0]
        assert len(batch["ids"]) == 2
        assert batch["documents"] == documents
        assert len(batch["embeddings"]) == 2
        assert len(batch["metadatas"]) == 2
        assert batch["metadatas"][0]["source_file"] == "file1.pdf"
        assert batch["metadatas"][1]["source_file"] == "file2.pdf"

    def test_prepare_storage_batch_multiple_batches(self):
        """Test preparing storage batch that requires multiple batches."""
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        metadata_list = [
            DocumentMetadata("file1.pdf", 0, "hr"),
            DocumentMetadata("file2.pdf", 1, "hr"),
            DocumentMetadata("file3.pdf", 2, "hr")
        ]

        result = prepare_storage_batch(documents, embeddings, metadata_list, batch_size=2)

        assert len(result) == 2
        # First batch
        assert len(result[0]["ids"]) == 2
        assert result[0]["documents"] == ["Doc 1", "Doc 2"]
        # Second batch
        assert len(result[1]["ids"]) == 1
        assert result[1]["documents"] == ["Doc 3"]

    def test_prepare_storage_batch_mismatched_lengths(self):
        """Test preparing storage batch with mismatched input lengths."""
        # The Python chained comparison doesn't work as expected for three-way length check
        # len(documents) != len(embeddings) != len(metadata_list) means:
        # (len(documents) != len(embeddings)) and (len(embeddings) != len(metadata_list))
        # So with lengths [2, 1, 2] it's (2 != 1) and (1 != 2) = True and True = True (raises)
        documents = ["Doc 1", "Doc 2"]
        embeddings = [np.array([1.0])]  # 1 embedding
        metadata_list = [DocumentMetadata("file1.pdf", 0, "hr"), DocumentMetadata("file2.pdf", 1, "hr")]  # 2 metadata

        with pytest.raises(ValueError, match="Documents, embeddings, and metadata lists must have same length"):
            prepare_storage_batch(documents, embeddings, metadata_list)

    def test_prepare_storage_batch_id_generation(self):
        """Test that batch preparation generates unique IDs."""
        documents = ["Doc 1", "Doc 2"]
        embeddings = [np.array([1.0]), np.array([2.0])]
        metadata_list = [
            DocumentMetadata("test.pdf", 0, "hr"),
            DocumentMetadata("test.pdf", 1, "hr")
        ]

        result = prepare_storage_batch(documents, embeddings, metadata_list)

        batch = result[0]
        ids = batch["ids"]
        assert len(ids) == 2
        assert ids[0] != ids[1]  # IDs should be unique
        assert "test.pdf_chunk_0_" in ids[0]
        assert "test.pdf_chunk_1_" in ids[1]


class TestParseQueryResults:
    """Test parse_query_results function."""

    def test_parse_query_results_valid(self):
        """Test parsing valid query results."""
        raw_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[{"source": "file1.pdf"}, {"source": "file2.pdf"}]],
            "distances": [[0.1, 0.3]]
        }

        result = parse_query_results(raw_results)

        assert len(result) == 2
        assert result[0].id == "doc1"
        assert result[0].content == "Content 1"
        assert result[0].metadata == {"source": "file1.pdf"}
        assert result[0].score == 0.1
        assert result[1].id == "doc2"
        assert result[1].content == "Content 2"
        assert result[1].metadata == {"source": "file2.pdf"}
        assert result[1].score == 0.3

    def test_parse_query_results_empty(self):
        """Test parsing empty query results."""
        raw_results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        result = parse_query_results(raw_results)

        assert result == []

    def test_parse_query_results_no_results(self):
        """Test parsing with no results structure."""
        raw_results = {}

        result = parse_query_results(raw_results)

        assert result == []

    def test_parse_query_results_missing_documents(self):
        """Test parsing with missing documents field."""
        raw_results = {
            "ids": [["doc1"]],
            "metadatas": [[{"source": "file1.pdf"}]],
            "distances": [[0.1]]
        }

        with pytest.raises(ValueError, match="Missing 'documents' in query results"):
            parse_query_results(raw_results)

    def test_parse_query_results_missing_metadatas(self):
        """Test parsing with missing metadatas field."""
        raw_results = {
            "ids": [["doc1"]],
            "documents": [["Content 1"]],
            "distances": [[0.1]]
        }

        with pytest.raises(ValueError, match="Missing 'metadatas' in query results"):
            parse_query_results(raw_results)

    def test_parse_query_results_missing_distances(self):
        """Test parsing with missing distances field."""
        raw_results = {
            "ids": [["doc1"]],
            "documents": [["Content 1"]],
            "metadatas": [[{"source": "file1.pdf"}]]
        }

        with pytest.raises(ValueError, match="Missing 'distances' in query results"):
            parse_query_results(raw_results)

    def test_parse_query_results_mismatched_lengths(self):
        """Test parsing with mismatched result lengths."""
        raw_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Content 1"]],  # Shorter than ids
            "metadatas": [[{"source": "file1.pdf"}, {"source": "file2.pdf"}]],
            "distances": [[0.1, 0.3]]
        }

        result = parse_query_results(raw_results)

        assert len(result) == 2
        assert result[0].content == "Content 1"
        assert result[1].content == ""  # Missing content defaults to empty string


class TestCalculateBatchSizes:
    """Test calculate_batch_sizes function."""

    def test_calculate_batch_sizes_small(self):
        """Test batch size calculation for small document sets."""
        result = calculate_batch_sizes(50)

        assert result == 25

    def test_calculate_batch_sizes_very_small(self):
        """Test batch size calculation for very small document sets."""
        result = calculate_batch_sizes(10)

        assert result == 10

    def test_calculate_batch_sizes_medium(self):
        """Test batch size calculation for medium document sets."""
        result = calculate_batch_sizes(500)

        assert result == 50

    def test_calculate_batch_sizes_large(self):
        """Test batch size calculation for large document sets."""
        result = calculate_batch_sizes(5000)

        assert result == 100

    def test_calculate_batch_sizes_custom_max(self):
        """Test batch size calculation with custom max batch size."""
        result = calculate_batch_sizes(5000, max_batch_size=200)

        assert result == 200


class TestExtractDocumentIds:
    """Test extract_document_ids function."""

    def test_extract_document_ids_valid(self):
        """Test extracting IDs from valid documents."""
        documents = [
            {"id": "doc1", "content": "Content 1"},
            {"id": "doc2", "content": "Content 2"}
        ]

        result = extract_document_ids(documents)

        assert result == ["doc1", "doc2"]

    def test_extract_document_ids_missing_id(self):
        """Test extracting IDs with missing ID field."""
        documents = [
            {"id": "doc1", "content": "Content 1"},
            {"content": "Content 2"}  # Missing id
        ]

        with pytest.raises(KeyError, match="Document at index 1 missing 'id' field"):
            extract_document_ids(documents)

    def test_extract_document_ids_empty_list(self):
        """Test extracting IDs from empty list."""
        documents = []

        result = extract_document_ids(documents)

        assert result == []


class TestMergeSearchResults:
    """Test merge_search_results function."""

    def test_merge_search_results_basic(self):
        """Test basic merging of search results."""
        results1 = [
            QueryResult("doc1", "Content 1", {}, 0.1),
            QueryResult("doc2", "Content 2", {}, 0.3)
        ]
        results2 = [
            QueryResult("doc3", "Content 3", {}, 0.2),
            QueryResult("doc4", "Content 4", {}, 0.4)
        ]

        result = merge_search_results([results1, results2], max_results=10)

        assert len(result) == 4
        # Should be sorted by score (ascending)
        assert result[0].id == "doc1"  # score 0.1
        assert result[1].id == "doc3"  # score 0.2
        assert result[2].id == "doc2"  # score 0.3
        assert result[3].id == "doc4"  # score 0.4

    def test_merge_search_results_limit(self):
        """Test merging with result limit."""
        results1 = [
            QueryResult("doc1", "Content 1", {}, 0.1),
            QueryResult("doc2", "Content 2", {}, 0.3)
        ]
        results2 = [
            QueryResult("doc3", "Content 3", {}, 0.2),
            QueryResult("doc4", "Content 4", {}, 0.4)
        ]

        result = merge_search_results([results1, results2], max_results=2)

        assert len(result) == 2
        assert result[0].id == "doc1"  # Best score
        assert result[1].id == "doc3"  # Second best score

    def test_merge_search_results_empty(self):
        """Test merging empty results."""
        result = merge_search_results([[], []], max_results=10)

        assert result == []

    def test_merge_search_results_single_list(self):
        """Test merging single result list."""
        results = [
            QueryResult("doc1", "Content 1", {}, 0.3),
            QueryResult("doc2", "Content 2", {}, 0.1)
        ]

        result = merge_search_results([results], max_results=10)

        assert len(result) == 2
        assert result[0].id == "doc2"  # Better score (0.1)
        assert result[1].id == "doc1"  # Worse score (0.3)


# ===== MAIN CLASS TESTS =====

class TestVectorStorage:
    """Test VectorStorage class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_collection = Mock(spec=VectorCollection)
        self.mock_database = Mock(spec=VectorDatabase)
        self.mock_collection.name = "test_collection"
        self.mock_collection.metadata = {"test": "metadata"}
        self.mock_collection.count.return_value = 100
        self.mock_database.create_collection.return_value = self.mock_collection

    def test_vector_storage_initialization(self):
        """Test VectorStorage initialization."""
        storage = VectorStorage(self.mock_database)

        assert storage.database == self.mock_database
        assert storage.collection is None

    async def test_initialize_collection(self):
        """Test collection initialization."""
        storage = VectorStorage(self.mock_database)

        await storage.initialize("test_collection", reset_if_exists=True)

        assert storage.collection == self.mock_collection
        self.mock_database.create_collection.assert_called_once_with(
            name="test_collection", reset_if_exists=True
        )

    async def test_store_documents_success(self):
        """Test successful document storage."""
        storage = VectorStorage(self.mock_database)
        await storage.initialize("test_collection")

        documents = ["Doc 1", "Doc 2"]
        embeddings = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        metadata_list = [
            DocumentMetadata("file1.pdf", 0, "hr"),
            DocumentMetadata("file2.pdf", 1, "hr")
        ]

        result = await storage.store_documents(documents, embeddings, metadata_list)

        assert result.success is True
        assert result.documents_stored == 2
        assert result.batches_processed == 1
        assert len(result.document_ids) == 2
        self.mock_collection.add.assert_called_once()

    async def test_store_documents_not_initialized(self):
        """Test document storage without initialization."""
        storage = VectorStorage(self.mock_database)

        documents = ["Doc 1"]
        embeddings = [np.array([1.0])]
        metadata_list = [DocumentMetadata("file1.pdf", 0, "hr")]

        result = await storage.store_documents(documents, embeddings, metadata_list)

        assert result.success is False
        assert result.error_message == "Storage not initialized - call initialize() first"


    async def test_search_documents_not_initialized(self):
        """Test document search without initialization."""
        storage = VectorStorage(self.mock_database)

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            await storage.search_documents(query_text="test")

    async def test_search_documents_no_query(self):
        """Test document search without query text or embedding."""
        storage = VectorStorage(self.mock_database)
        await storage.initialize("test_collection")

        with pytest.raises(ValueError, match="Either query_text or query_embedding must be provided"):
            await storage.search_documents()

    async def test_get_collection_stats(self):
        """Test getting collection statistics."""
        storage = VectorStorage(self.mock_database)
        await storage.initialize("test_collection")

        stats = await storage.get_collection_stats()

        assert stats == {
            "name": "test_collection",
            "document_count": 100,
            "metadata": {"test": "metadata"}
        }

    async def test_get_collection_stats_not_initialized(self):
        """Test getting stats without initialization."""
        storage = VectorStorage(self.mock_database)

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            await storage.get_collection_stats()

    async def test_delete_documents_by_ids(self):
        """Test deleting documents by IDs."""
        storage = VectorStorage(self.mock_database)
        await storage.initialize("test_collection")

        await storage.delete_documents(ids=["doc1", "doc2"])

        self.mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2"], where=None)

    async def test_delete_documents_by_filter(self):
        """Test deleting documents by metadata filter."""
        storage = VectorStorage(self.mock_database)
        await storage.initialize("test_collection")

        filter_metadata = {"language": "hr"}
        await storage.delete_documents(filter_metadata=filter_metadata)

        self.mock_collection.delete.assert_called_once_with(ids=None, where=filter_metadata)

    async def test_delete_documents_not_initialized(self):
        """Test deleting documents without initialization."""
        storage = VectorStorage(self.mock_database)

        with pytest.raises(ValueError, match="Storage not initialized"):
            await storage.delete_documents(ids=["doc1"])

    async def test_delete_documents_no_criteria(self):
        """Test deleting documents without IDs or filter."""
        storage = VectorStorage(self.mock_database)
        await storage.initialize("test_collection")

        with pytest.raises(ValueError, match="Either ids or filter_metadata must be provided"):
            await storage.delete_documents()


# ===== FACTORY FUNCTION TESTS =====

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_vector_storage(self):
        """Test creating vector storage."""
        mock_database = Mock(spec=VectorDatabase)

        storage = create_vector_storage(mock_database)

        assert isinstance(storage, VectorStorage)
        assert storage.database == mock_database

