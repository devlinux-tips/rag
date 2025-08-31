"""
Unit tests for storage module.
Tests ChromaDB operations with Croatian document storage and retrieval.
"""

import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.vectordb.storage import (
    ChromaDBStorage,
    DocumentMetadata,
    StorageConfig,
    create_storage_client,
)


class TestStorageConfig:
    """Test storage configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StorageConfig()

        assert config.db_path == "./data/chromadb"
        assert config.collection_name == "croatian_documents"
        assert config.distance_metric == "cosine"
        assert config.persist is True
        assert config.allow_reset is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = StorageConfig(
            db_path="/tmp/test_db",
            collection_name="test_collection",
            distance_metric="l2",
            persist=False,
            allow_reset=False,
        )

        assert config.db_path == "/tmp/test_db"
        assert config.collection_name == "test_collection"
        assert config.distance_metric == "l2"
        assert config.persist is False
        assert config.allow_reset is False


class TestDocumentMetadata:
    """Test document metadata structure."""

    def test_metadata_creation(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            source="test.txt",
            title="Test Document",
            chunk_index=1,
            total_chunks=5,
            language="hr",
            content_type="text/plain",
        )

        assert metadata.source == "test.txt"
        assert metadata.title == "Test Document"
        assert metadata.chunk_index == 1
        assert metadata.total_chunks == 5
        assert metadata.language == "hr"
        assert metadata.content_type == "text/plain"

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = DocumentMetadata(
            source="test.txt", title="Test Document", chunk_index=1, total_chunks=5
        )

        metadata_dict = metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["source"] == "test.txt"
        assert metadata_dict["title"] == "Test Document"
        assert metadata_dict["chunk_index"] == 1
        assert metadata_dict["total_chunks"] == 5
        assert metadata_dict["language"] == "hr"  # Default value

    def test_metadata_to_dict_excludes_none(self):
        """Test that None values are excluded from dictionary."""
        metadata = DocumentMetadata(
            source="test.txt", title=None, chunk_index=1  # This should be excluded
        )

        metadata_dict = metadata.to_dict()

        assert "source" in metadata_dict
        assert "chunk_index" in metadata_dict
        assert "title" not in metadata_dict  # Should be excluded


class TestChromaDBStorage:
    """Test ChromaDB storage operations."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def storage_config(self, temp_db_path):
        """Create test storage configuration."""
        return StorageConfig(
            db_path=temp_db_path,
            collection_name="test_collection",
            persist=True,
            allow_reset=True,
        )

    @pytest.fixture
    def mock_client(self):
        """Create mock ChromaDB client."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        return mock_client, mock_collection

    def test_initialization(self, storage_config):
        """Test storage initialization."""
        with patch("src.vectordb.storage.chromadb.PersistentClient") as mock_persistent:
            mock_persistent.return_value = MagicMock()

            storage = ChromaDBStorage(storage_config)

            assert storage.config == storage_config
            assert Path(storage_config.db_path).exists()
            mock_persistent.assert_called_once()

    def test_create_collection(self, storage_config, mock_client):
        """Test collection creation."""
        mock_client_obj, mock_collection = mock_client

        with patch(
            "src.vectordb.storage.chromadb.PersistentClient",
            return_value=mock_client_obj,
        ):
            storage = ChromaDBStorage(storage_config)
            collection = storage.create_collection()

            assert collection == mock_collection
            mock_client_obj.get_or_create_collection.assert_called_with(
                name="test_collection", metadata={"hnsw:space": "cosine"}
            )

    def test_add_documents(self, storage_config, mock_client):
        """Test adding documents."""
        mock_client_obj, mock_collection = mock_client

        with patch(
            "src.vectordb.storage.chromadb.PersistentClient",
            return_value=mock_client_obj,
        ):
            storage = ChromaDBStorage(storage_config)
            storage.collection = mock_collection

            documents = [
                "Zagreb je glavni grad Hrvatske.",
                "Hrvatska ima prekrasnu obalu.",
            ]
            metadatas = [
                {"source": "doc1.txt", "title": "O Zagrebu"},
                {"source": "doc2.txt", "title": "O obali"},
            ]

            ids = storage.add_documents(documents, metadatas)

            assert len(ids) == 2
            mock_collection.add.assert_called_once()

            # Check call arguments
            call_kwargs = mock_collection.add.call_args.kwargs
            assert call_kwargs["documents"] == documents
            assert call_kwargs["metadatas"] == metadatas
            assert "ids" in call_kwargs

    def test_add_documents_with_embeddings(self, storage_config, mock_client):
        """Test adding documents with pre-computed embeddings."""
        mock_client_obj, mock_collection = mock_client

        with patch(
            "src.vectordb.storage.chromadb.PersistentClient",
            return_value=mock_client_obj,
        ):
            storage = ChromaDBStorage(storage_config)
            storage.collection = mock_collection

            documents = ["Test document"]
            metadatas = [{"source": "test.txt"}]
            embeddings = [[0.1, 0.2, 0.3]]

            storage.add_documents(documents, metadatas, embeddings)

            call_kwargs = mock_collection.add.call_args.kwargs
            assert call_kwargs["embeddings"] == embeddings


class TestCroatianDocumentOperations:
    """Test Croatian-specific document operations."""

    @pytest.fixture
    def croatian_documents(self):
        """Sample Croatian documents with metadata."""
        return [
            {
                "content": "Zagreb je glavni i najveći grad Republike Hrvatske.",
                "metadata": {
                    "source": "zagreb.txt",
                    "title": "O Zagrebu",
                    "language": "hr",
                    "region": "Središnja Hrvatska",
                },
            },
            {
                "content": "Dubrovnik se često naziva 'biser Jadrana' zbog svoje ljepote.",
                "metadata": {
                    "source": "dubrovnik.txt",
                    "title": "Dubrovnik - biser Jadrana",
                    "language": "hr",
                    "region": "Dalmacija",
                },
            },
            {
                "content": "Plitvička jezera su najpoznatiji nacionalni park u Hrvatskoj.",
                "metadata": {
                    "source": "plitvice.txt",
                    "title": "Plitvička jezera",
                    "language": "hr",
                    "region": "Lika",
                },
            },
        ]

    @pytest.fixture
    def storage(self):
        """Create storage instance with mocked ChromaDB."""
        config = StorageConfig(db_path="/tmp/test_croatian_db", collection_name="croatian_test")

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "croatian_test"
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("src.vectordb.storage.chromadb.PersistentClient", return_value=mock_client):
            storage = ChromaDBStorage(config)
            storage.collection = mock_collection
            return storage, mock_collection

    def test_add_croatian_document_chunks(self, croatian_documents, storage):
        """Test adding Croatian document chunks."""
        storage_obj, mock_collection = storage

        chunks = []
        for doc in croatian_documents:
            chunk = {
                "id": str(uuid.uuid4()),
                "content": doc["content"],
                **doc["metadata"],
            }
            chunks.append(chunk)

        ids = storage_obj.add_document_chunks(chunks)

        assert len(ids) == len(croatian_documents)
        mock_collection.add.assert_called_once()

        # Verify Croatian content is preserved
        call_kwargs = mock_collection.add.call_args.kwargs
        documents = call_kwargs["documents"]

        # Check that Croatian diacritics are preserved
        assert "č" in documents[1] or "ć" in documents[1]  # Dubrovnik content
        assert "Plitvička" in documents[2]  # Plitvice content with č

    def test_query_croatian_documents(self, storage):
        """Test querying Croatian documents."""
        storage_obj, mock_collection = storage

        # Mock query results
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Zagreb je glavni grad Hrvatske.", "Dubrovnik je biser Jadrana."]],
            "metadatas": [
                [
                    {"source": "zagreb.txt", "language": "hr"},
                    {"source": "dubrovnik.txt", "language": "hr"},
                ]
            ],
            "distances": [[0.1, 0.3]],
        }

        results = storage_obj.query_similar(
            query_texts=["Koji je glavni grad Hrvatske?"], n_results=2
        )

        assert "ids" in results
        assert "documents" in results
        mock_collection.query.assert_called_once()

        # Check query parameters
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["query_texts"] == ["Koji je glavni grad Hrvatske?"]
        assert call_kwargs["n_results"] == 2

    def test_filter_by_language(self, storage):
        """Test filtering documents by language."""
        storage_obj, mock_collection = storage

        # Mock filtered results
        mock_collection.query.return_value = {
            "ids": [["hr_doc1", "hr_doc2"]],
            "documents": [["Croatian doc 1", "Croatian doc 2"]],
            "metadatas": [[{"language": "hr"}, {"language": "hr"}]],
            "distances": [[0.1, 0.2]],
        }

        results = storage_obj.query_similar(
            query_texts=["test query"], where={"language": "hr"}, n_results=5
        )

        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"language": "hr"}

    def test_filter_by_region(self, storage):
        """Test filtering Croatian documents by region."""
        storage_obj, mock_collection = storage

        results = storage_obj.query_similar(
            query_texts=["Dalmatian coast"], where={"region": "Dalmacija"}, n_results=3
        )

        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"region": "Dalmacija"}


class TestDocumentCRUDOperations:
    """Test Create, Read, Update, Delete operations."""

    @pytest.fixture
    def storage_with_mock(self):
        """Create storage with mock collection."""
        config = StorageConfig(db_path="/tmp/test_crud", collection_name="crud_test")

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "crud_test"
        mock_collection.count.return_value = 3
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("src.vectordb.storage.chromadb.PersistentClient", return_value=mock_client):
            storage = ChromaDBStorage(config)
            storage.collection = mock_collection
            return storage, mock_collection

    def test_get_documents_by_ids(self, storage_with_mock):
        """Test retrieving documents by IDs."""
        storage, mock_collection = storage_with_mock

        mock_collection.get.return_value = {
            "ids": ["doc1", "doc2"],
            "documents": ["Document 1", "Document 2"],
            "metadatas": [{"source": "1.txt"}, {"source": "2.txt"}],
        }

        results = storage.get_documents(ids=["doc1", "doc2"])

        assert "ids" in results
        mock_collection.get.assert_called_once()

        call_kwargs = mock_collection.get.call_args.kwargs
        assert call_kwargs["ids"] == ["doc1", "doc2"]

    def test_get_documents_with_filter(self, storage_with_mock):
        """Test retrieving documents with metadata filter."""
        storage, mock_collection = storage_with_mock

        results = storage.get_documents(where={"language": "hr"}, limit=5)

        call_kwargs = mock_collection.get.call_args.kwargs
        assert call_kwargs["where"] == {"language": "hr"}
        assert call_kwargs["limit"] == 5

    def test_update_documents(self, storage_with_mock):
        """Test updating existing documents."""
        storage, mock_collection = storage_with_mock

        ids = ["doc1", "doc2"]
        new_documents = ["Updated doc 1", "Updated doc 2"]
        new_metadatas = [{"updated": True}, {"updated": True}]

        storage.update_documents(ids=ids, documents=new_documents, metadatas=new_metadatas)

        mock_collection.update.assert_called_once()

        call_kwargs = mock_collection.update.call_args.kwargs
        assert call_kwargs["ids"] == ids
        assert call_kwargs["documents"] == new_documents
        assert call_kwargs["metadatas"] == new_metadatas

    def test_delete_documents_by_ids(self, storage_with_mock):
        """Test deleting documents by IDs."""
        storage, mock_collection = storage_with_mock

        ids_to_delete = ["doc1", "doc2"]
        storage.delete_documents(ids=ids_to_delete)

        mock_collection.delete.assert_called_once()
        call_kwargs = mock_collection.delete.call_args.kwargs
        assert call_kwargs["ids"] == ids_to_delete

    def test_delete_documents_by_filter(self, storage_with_mock):
        """Test deleting documents by metadata filter."""
        storage, mock_collection = storage_with_mock

        filter_condition = {"source": "old_source.txt"}
        storage.delete_documents(where=filter_condition)

        mock_collection.delete.assert_called_once()
        call_kwargs = mock_collection.delete.call_args.kwargs
        assert call_kwargs["where"] == filter_condition

    def test_delete_documents_requires_filter_or_ids(self, storage_with_mock):
        """Test that delete requires either IDs or filter."""
        storage, mock_collection = storage_with_mock

        with pytest.raises(ValueError, match="Either ids or where filter must be provided"):
            storage.delete_documents()


class TestCollectionManagement:
    """Test collection management operations."""

    @pytest.fixture
    def storage_manager(self):
        """Create storage for collection management tests."""
        config = StorageConfig(db_path="/tmp/collection_test")

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 10
        mock_collection.metadata = {"hnsw:space": "cosine"}

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.list_collections.return_value = [mock_collection]

        with patch("src.vectordb.storage.chromadb.PersistentClient", return_value=mock_client):
            storage = ChromaDBStorage(config)
            storage.collection = mock_collection
            return storage, mock_client, mock_collection

    def test_get_collection_info(self, storage_manager):
        """Test getting collection information."""
        storage, mock_client, mock_collection = storage_manager

        info = storage.get_collection_info()

        assert info["name"] == "test_collection"
        assert info["count"] == 10
        assert info["metadata"] == {"hnsw:space": "cosine"}
        assert info["distance_metric"] == "cosine"

    def test_list_collections(self, storage_manager):
        """Test listing all collections."""
        storage, mock_client, mock_collection = storage_manager

        collections = storage.list_collections()

        assert len(collections) == 1
        assert collections[0] == "test_collection"
        mock_client.list_collections.assert_called_once()

    def test_reset_collection(self, storage_manager):
        """Test resetting (deleting and recreating) collection."""
        storage, mock_client, mock_collection = storage_manager

        storage.reset_collection("test_collection")

        mock_client.delete_collection.assert_called_with("test_collection")
        # get_or_create_collection called twice: once in init, once in reset
        assert mock_client.get_or_create_collection.call_count == 2


class TestUtilityFunctions:
    """Test utility and factory functions."""

    def test_create_storage_client(self):
        """Test storage client factory function."""
        with patch("src.vectordb.storage.chromadb.PersistentClient"):
            client = create_storage_client(
                db_path="/tmp/test_factory",
                collection_name="factory_test",
                persist=True,
            )

            assert isinstance(client, ChromaDBStorage)
            assert client.config.db_path == "/tmp/test_factory"
            assert client.config.collection_name == "factory_test"
            assert client.config.persist is True


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_inputs_add_documents(self):
        """Test error handling for invalid inputs in add_documents."""
        config = StorageConfig(db_path="/tmp/error_test")

        with patch("src.vectordb.storage.chromadb.PersistentClient"):
            storage = ChromaDBStorage(config)
            storage.collection = MagicMock()

            # Mismatched document and metadata counts
            with pytest.raises(ValueError, match="Number of documents and metadatas must match"):
                storage.add_documents(
                    documents=["doc1", "doc2"],
                    metadatas=[{"meta": "1"}],  # Only one metadata for two docs
                )

            # Mismatched embeddings and documents
            with pytest.raises(ValueError, match="Number of embeddings and documents must match"):
                storage.add_documents(
                    documents=["doc1"],
                    metadatas=[{"meta": "1"}],
                    embeddings=[[0.1, 0.2], [0.3, 0.4]],  # Two embeddings for one doc
                )

    def test_no_collection_error(self):
        """Test error when no collection is available."""
        config = StorageConfig(db_path="/tmp/no_collection_test")

        with patch("src.vectordb.storage.chromadb.PersistentClient"):
            storage = ChromaDBStorage(config)
            storage.collection = None  # No collection set

            with pytest.raises(ValueError, match="No collection available"):
                storage.query_similar(query_texts=["test"])

    def test_query_without_parameters(self):
        """Test error when query has no parameters."""
        config = StorageConfig()

        with patch("src.vectordb.storage.chromadb.PersistentClient"):
            storage = ChromaDBStorage(config)
            storage.collection = MagicMock()

            with pytest.raises(
                ValueError,
                match="Either query_texts or query_embeddings must be provided",
            ):
                storage.query_similar()  # No query parameters


if __name__ == "__main__":
    pytest.main([__file__])
