"""
Comprehensive tests for storage factory implementations.
Tests ChromaDB and mock implementations of storage protocols.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from src.vectordb.storage_factories import (
    # ChromaDB implementations
    ChromaDBCollection,
    ChromaDBDatabase,

    # Mock implementations
    MockCollection,
    MockDatabase,

    # Factory functions
    create_chromadb_database,
    create_mock_database,
)


# ===== CHROMADB COLLECTION TESTS =====

class TestChromaDBCollection:
    """Test ChromaDBCollection implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock ChromaDB collection
        self.mock_chroma_collection = Mock()
        self.mock_chroma_collection.name = "test_collection"
        self.mock_chroma_collection.metadata = {"hnsw:space": "cosine"}
        self.collection = ChromaDBCollection(self.mock_chroma_collection)

    def test_add_with_embeddings(self):
        """Test adding documents with embeddings."""
        ids = ["doc1", "doc2"]
        documents = ["Document 1", "Document 2"]
        metadatas = [{"type": "test"}, {"type": "test"}]
        embeddings = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]

        self.collection.add(ids, documents, metadatas, embeddings)

        self.mock_chroma_collection.add.assert_called_once()
        args, kwargs = self.mock_chroma_collection.add.call_args

        assert kwargs["ids"] == ids
        assert kwargs["documents"] == documents
        assert kwargs["metadatas"] == metadatas
        # Check embeddings converted to lists
        assert kwargs["embeddings"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_add_without_embeddings(self):
        """Test adding documents without embeddings."""
        ids = ["doc1", "doc2"]
        documents = ["Document 1", "Document 2"]
        metadatas = [{"type": "test"}, {"type": "test"}]

        self.collection.add(ids, documents, metadatas)

        self.mock_chroma_collection.add.assert_called_once_with(
            ids=ids, documents=documents, metadatas=metadatas
        )

    def test_add_handles_list_embeddings(self):
        """Test adding documents with embeddings already as lists."""
        ids = ["doc1"]
        documents = ["Document 1"]
        metadatas = [{"type": "test"}]
        embeddings = [[1.0, 2.0, 3.0]]  # Already lists

        self.collection.add(ids, documents, metadatas, embeddings)

        args, kwargs = self.mock_chroma_collection.add.call_args
        assert kwargs["embeddings"] == [[1.0, 2.0, 3.0]]

    def test_add_error_handling(self):
        """Test add method error handling."""
        self.mock_chroma_collection.add.side_effect = Exception("ChromaDB error")

        with pytest.raises(Exception, match="ChromaDB error"):
            self.collection.add(["doc1"], ["Document 1"], [{"type": "test"}])

    def test_query_with_embeddings(self):
        """Test querying with embeddings."""
        query_embeddings = [np.array([1.0, 2.0, 3.0])]
        expected_results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [[{"type": "test"}, {"type": "test"}]],
            "distances": [[0.1, 0.2]]
        }
        self.mock_chroma_collection.query.return_value = expected_results

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=5,
            where={"type": "test"}
        )

        self.mock_chroma_collection.query.assert_called_once()
        args, kwargs = self.mock_chroma_collection.query.call_args

        assert kwargs["query_embeddings"] == [[1.0, 2.0, 3.0]]
        assert kwargs["n_results"] == 5
        assert kwargs["where"] == {"type": "test"}
        assert kwargs["include"] == ["documents", "metadatas", "distances"]
        assert results == expected_results

    def test_query_with_texts(self):
        """Test querying with text queries."""
        query_texts = ["search query"]
        expected_results = {"ids": [["doc1"]], "documents": [["Document 1"]]}
        self.mock_chroma_collection.query.return_value = expected_results

        results = self.collection.query(
            query_texts=query_texts,
            n_results=3,
            include=["documents"]
        )

        self.mock_chroma_collection.query.assert_called_once_with(
            query_texts=query_texts,
            n_results=3,
            where=None,
            where_document=None,
            include=["documents"]
        )
        assert results == expected_results

    def test_query_default_include(self):
        """Test query with default include parameters."""
        self.mock_chroma_collection.query.return_value = {}

        self.collection.query(query_texts=["test"])

        args, kwargs = self.mock_chroma_collection.query.call_args
        assert kwargs["include"] == ["documents", "metadatas", "distances"]

    def test_query_error_handling(self):
        """Test query method error handling."""
        self.mock_chroma_collection.query.side_effect = Exception("Query failed")

        with pytest.raises(Exception, match="Query failed"):
            self.collection.query(query_texts=["test"])

    def test_get_with_ids(self):
        """Test getting documents by IDs."""
        ids = ["doc1", "doc2"]
        expected_results = {"ids": ["doc1", "doc2"], "documents": ["Doc 1", "Doc 2"]}
        self.mock_chroma_collection.get.return_value = expected_results

        results = self.collection.get(ids=ids, include=["documents"])

        self.mock_chroma_collection.get.assert_called_once_with(
            ids=ids, where=None, limit=None, offset=None, include=["documents"]
        )
        assert results == expected_results

    def test_get_default_include(self):
        """Test get with default include parameters."""
        self.mock_chroma_collection.get.return_value = {}

        self.collection.get()

        args, kwargs = self.mock_chroma_collection.get.call_args
        assert kwargs["include"] == ["documents", "metadatas"]

    def test_get_error_handling(self):
        """Test get method error handling."""
        self.mock_chroma_collection.get.side_effect = Exception("Get failed")

        with pytest.raises(Exception, match="Get failed"):
            self.collection.get(ids=["doc1"])

    def test_update_with_embeddings(self):
        """Test updating documents with embeddings."""
        ids = ["doc1", "doc2"]
        documents = ["Updated Doc 1", "Updated Doc 2"]
        metadatas = [{"updated": True}, {"updated": True}]
        embeddings = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        self.collection.update(ids, documents, metadatas, embeddings)

        self.mock_chroma_collection.update.assert_called_once()
        args, kwargs = self.mock_chroma_collection.update.call_args

        assert kwargs["ids"] == ids
        assert kwargs["documents"] == documents
        assert kwargs["metadatas"] == metadatas
        assert kwargs["embeddings"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_update_without_embeddings(self):
        """Test updating documents without embeddings."""
        ids = ["doc1"]
        documents = ["Updated Doc 1"]
        metadatas = [{"updated": True}]

        self.collection.update(ids, documents, metadatas)

        self.mock_chroma_collection.update.assert_called_once_with(
            ids=ids, documents=documents, metadatas=metadatas
        )

    def test_update_error_handling(self):
        """Test update method error handling."""
        self.mock_chroma_collection.update.side_effect = Exception("Update failed")

        with pytest.raises(Exception, match="Update failed"):
            self.collection.update(["doc1"], ["Updated"])

    def test_delete_with_ids(self):
        """Test deleting documents by IDs."""
        ids = ["doc1", "doc2"]

        self.collection.delete(ids=ids)

        self.mock_chroma_collection.delete.assert_called_once_with(ids=ids, where=None)

    def test_delete_with_where(self):
        """Test deleting documents with where filter."""
        where = {"type": "test"}

        self.collection.delete(where=where)

        self.mock_chroma_collection.delete.assert_called_once_with(ids=None, where=where)

    def test_delete_validation_error(self):
        """Test delete method validation."""
        with pytest.raises(ValueError, match="Either ids or where filter must be provided"):
            self.collection.delete()

    def test_delete_error_handling(self):
        """Test delete method error handling."""
        self.mock_chroma_collection.delete.side_effect = Exception("Delete failed")

        with pytest.raises(Exception, match="Delete failed"):
            self.collection.delete(ids=["doc1"])

    def test_count(self):
        """Test getting document count."""
        self.mock_chroma_collection.count.return_value = 42

        count = self.collection.count()

        assert count == 42
        self.mock_chroma_collection.count.assert_called_once()

    def test_count_error_handling(self):
        """Test count method error handling."""
        self.mock_chroma_collection.count.side_effect = Exception("Count failed")

        with pytest.raises(Exception, match="Count failed"):
            self.collection.count()

    def test_name_property(self):
        """Test collection name property."""
        assert self.collection.name == "test_collection"

    def test_metadata_property(self):
        """Test collection metadata property."""
        assert self.collection.metadata == {"hnsw:space": "cosine"}

    def test_metadata_property_none(self):
        """Test collection metadata property when None."""
        self.mock_chroma_collection.metadata = None

        assert self.collection.metadata == {}


# ===== CHROMADB DATABASE TESTS =====

class TestChromaDBDatabase:
    """Test ChromaDBDatabase implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test_db")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.vectordb.storage_factories.chromadb.PersistentClient')
    @patch('src.vectordb.storage_factories.Settings')
    def test_init_persistent(self, mock_settings, mock_client):
        """Test database initialization with persistent client."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        db = ChromaDBDatabase(
            db_path=self.db_path,
            distance_metric="l2",
            persist=True,
            allow_reset=True
        )

        assert db.db_path == self.db_path
        assert db.distance_metric == "l2"
        assert db.persist is True
        assert db.allow_reset is True

        # Check that directory was created
        assert Path(self.db_path).exists()

        # Verify client creation
        mock_settings.assert_called_once_with(
            persist_directory=self.db_path,
            allow_reset=True,
            anonymized_telemetry=False,
        )
        mock_client.assert_called_once_with(path=self.db_path, settings=mock_settings.return_value)

    @patch('src.vectordb.storage_factories.chromadb.EphemeralClient')
    @patch('src.vectordb.storage_factories.Settings')
    def test_init_ephemeral(self, mock_settings, mock_client):
        """Test database initialization with ephemeral client."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        db = ChromaDBDatabase(
            db_path=self.db_path,
            persist=False
        )

        assert db.persist is False

        mock_settings.assert_called_once_with(
            persist_directory=None,
            allow_reset=False,
            anonymized_telemetry=False,
        )
        mock_client.assert_called_once_with(settings=mock_settings.return_value)

    @patch('src.vectordb.storage_factories.chromadb.PersistentClient')
    def test_create_client_error(self, mock_client):
        """Test client creation error handling."""
        mock_client.side_effect = Exception("Client creation failed")

        with pytest.raises(Exception, match="Client creation failed"):
            ChromaDBDatabase(self.db_path)

    def test_create_collection(self):
        """Test creating a collection."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)
            collection = db.create_collection("test_collection")

        mock_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={"hnsw:space": "cosine"}
        )
        assert isinstance(collection, ChromaDBCollection)

    def test_create_collection_with_reset(self):
        """Test creating a collection with reset."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)
            collection = db.create_collection("test_collection", reset_if_exists=True)

        # Should try to delete first
        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.get_or_create_collection.assert_called_once()

    def test_create_collection_reset_nonexistent(self):
        """Test creating a collection with reset when collection doesn't exist."""
        mock_client = Mock()
        mock_client.delete_collection.side_effect = Exception("Collection not found")
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)
            collection = db.create_collection("test_collection", reset_if_exists=True)

        # Should handle delete error gracefully
        mock_client.delete_collection.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()

    def test_create_collection_error(self):
        """Test create collection error handling."""
        mock_client = Mock()
        mock_client.get_or_create_collection.side_effect = Exception("Creation failed")

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)

            with pytest.raises(Exception, match="Creation failed"):
                db.create_collection("test_collection")

    def test_get_collection(self):
        """Test getting an existing collection."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "existing_collection"
        mock_client.get_collection.return_value = mock_collection

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)
            collection = db.get_collection("existing_collection")

        mock_client.get_collection.assert_called_once_with("existing_collection")
        assert isinstance(collection, ChromaDBCollection)

    def test_get_collection_error(self):
        """Test get collection error handling."""
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)

            with pytest.raises(Exception, match="Collection not found"):
                db.get_collection("nonexistent")

    def test_delete_collection(self):
        """Test deleting a collection."""
        mock_client = Mock()

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)
            db.delete_collection("test_collection")

        mock_client.delete_collection.assert_called_once_with("test_collection")

    def test_delete_collection_error(self):
        """Test delete collection error handling."""
        mock_client = Mock()
        mock_client.delete_collection.side_effect = Exception("Delete failed")

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)

            with pytest.raises(Exception, match="Delete failed"):
                db.delete_collection("test_collection")

    def test_list_collections(self):
        """Test listing collections."""
        mock_client = Mock()
        mock_col1 = Mock()
        mock_col1.name = "collection1"
        mock_col2 = Mock()
        mock_col2.name = "collection2"
        mock_client.list_collections.return_value = [mock_col1, mock_col2]

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)
            collections = db.list_collections()

        assert collections == ["collection1", "collection2"]
        mock_client.list_collections.assert_called_once()

    def test_list_collections_error(self):
        """Test list collections error handling."""
        mock_client = Mock()
        mock_client.list_collections.side_effect = Exception("List failed")

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path)

            with pytest.raises(Exception, match="List failed"):
                db.list_collections()

    def test_reset_allowed(self):
        """Test database reset when allowed."""
        mock_client = Mock()

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path, allow_reset=True)
            db.reset()

        mock_client.reset.assert_called_once()

    def test_reset_not_allowed(self):
        """Test database reset when not allowed."""
        mock_client = Mock()

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path, allow_reset=False)

            with pytest.raises(ValueError, match="Database reset not allowed"):
                db.reset()

    def test_reset_error(self):
        """Test reset error handling."""
        mock_client = Mock()
        mock_client.reset.side_effect = Exception("Reset failed")

        with patch.object(ChromaDBDatabase, '_create_client', return_value=mock_client):
            db = ChromaDBDatabase(self.db_path, allow_reset=True)

            with pytest.raises(Exception, match="Reset failed"):
                db.reset()


# ===== MOCK COLLECTION TESTS =====

class TestMockCollection:
    """Test MockCollection implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collection = MockCollection("test_mock_collection")

    def test_init(self):
        """Test mock collection initialization."""
        assert self.collection.name == "test_mock_collection"
        assert self.collection.metadata == {"hnsw:space": "cosine"}
        assert len(self.collection._documents) == 0

    def test_add_documents(self):
        """Test adding documents to mock collection."""
        ids = ["doc1", "doc2"]
        documents = ["Document 1", "Document 2"]
        metadatas = [{"type": "test"}, {"type": "sample"}]
        embeddings = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        self.collection.add(ids, documents, metadatas, embeddings)

        assert len(self.collection._documents) == 2
        assert self.collection._documents["doc1"]["document"] == "Document 1"
        assert self.collection._documents["doc1"]["metadata"] == {"type": "test"}
        assert np.array_equal(self.collection._documents["doc1"]["embedding"], np.array([1.0, 2.0]))

    def test_add_without_embeddings(self):
        """Test adding documents without embeddings."""
        ids = ["doc1"]
        documents = ["Document 1"]
        metadatas = [{"type": "test"}]

        self.collection.add(ids, documents, metadatas)

        assert self.collection._documents["doc1"]["embedding"] is None

    def test_query_basic(self):
        """Test basic query functionality."""
        # Add test documents
        self.collection.add(
            ["doc1", "doc2", "doc3"],
            ["Document 1", "Document 2", "Document 3"],
            [{"type": "test"}, {"type": "sample"}, {"type": "test"}]
        )

        results = self.collection.query(
            query_texts=["test query"],
            n_results=2
        )

        assert len(results["ids"][0]) == 2
        assert len(results["documents"][0]) == 2
        assert len(results["metadatas"][0]) == 2
        assert len(results["distances"][0]) == 2
        # Mock returns first n_results documents
        assert results["ids"][0] == ["doc1", "doc2"]
        assert results["documents"][0] == ["Document 1", "Document 2"]

    def test_query_with_include(self):
        """Test query with specific include parameters."""
        self.collection.add(["doc1"], ["Document 1"], [{"type": "test"}])

        results = self.collection.query(
            query_texts=["test"],
            n_results=1,
            include=["documents"]
        )

        assert "documents" in results
        assert "metadatas" not in results
        assert "distances" not in results

    def test_query_empty_collection(self):
        """Test querying empty collection."""
        results = self.collection.query(query_texts=["test"], n_results=5)

        assert results["ids"] == [[]]
        assert results["documents"] == [[]]
        assert results["metadatas"] == [[]]
        assert results["distances"] == [[]]

    def test_get_by_ids(self):
        """Test getting documents by IDs."""
        self.collection.add(
            ["doc1", "doc2", "doc3"],
            ["Document 1", "Document 2", "Document 3"],
            [{"type": "test"}, {"type": "sample"}, {"type": "test"}]
        )

        results = self.collection.get(ids=["doc1", "doc3"])

        assert results["ids"] == ["doc1", "doc3"]
        assert results["documents"] == ["Document 1", "Document 3"]
        assert results["metadatas"] == [{"type": "test"}, {"type": "test"}]

    def test_get_all(self):
        """Test getting all documents."""
        self.collection.add(
            ["doc1", "doc2"],
            ["Document 1", "Document 2"],
            [{"type": "test"}, {"type": "sample"}]
        )

        results = self.collection.get()

        assert len(results["ids"]) == 2
        assert set(results["ids"]) == {"doc1", "doc2"}

    def test_get_nonexistent_ids(self):
        """Test getting nonexistent documents."""
        self.collection.add(["doc1"], ["Document 1"], [{"type": "test"}])

        results = self.collection.get(ids=["doc1", "nonexistent"])

        assert results["ids"] == ["doc1"]
        assert results["documents"] == ["Document 1"]

    def test_update_documents(self):
        """Test updating documents."""
        self.collection.add(
            ["doc1", "doc2"],
            ["Document 1", "Document 2"],
            [{"type": "test"}, {"type": "sample"}]
        )

        self.collection.update(
            ids=["doc1"],
            documents=["Updated Document 1"],
            metadatas=[{"type": "updated"}],
            embeddings=[np.array([5.0, 6.0])]
        )

        assert self.collection._documents["doc1"]["document"] == "Updated Document 1"
        assert self.collection._documents["doc1"]["metadata"] == {"type": "updated"}
        assert np.array_equal(self.collection._documents["doc1"]["embedding"], np.array([5.0, 6.0]))
        # doc2 should remain unchanged
        assert self.collection._documents["doc2"]["document"] == "Document 2"

    def test_update_nonexistent_document(self):
        """Test updating nonexistent document."""
        self.collection.update(
            ids=["nonexistent"],
            documents=["New Document"]
        )

        # Should not add new documents
        assert "nonexistent" not in self.collection._documents

    def test_delete_by_ids(self):
        """Test deleting documents by IDs."""
        self.collection.add(
            ["doc1", "doc2", "doc3"],
            ["Document 1", "Document 2", "Document 3"],
            [{"type": "test"}, {"type": "sample"}, {"type": "test"}]
        )

        self.collection.delete(ids=["doc1", "doc3"])

        assert "doc1" not in self.collection._documents
        assert "doc2" in self.collection._documents
        assert "doc3" not in self.collection._documents

    def test_delete_all(self):
        """Test deleting all documents."""
        self.collection.add(
            ["doc1", "doc2"],
            ["Document 1", "Document 2"],
            [{"type": "test"}, {"type": "sample"}]
        )

        self.collection.delete()  # No IDs or where clause

        assert len(self.collection._documents) == 0

    def test_count(self):
        """Test document count."""
        assert self.collection.count() == 0

        self.collection.add(
            ["doc1", "doc2"],
            ["Document 1", "Document 2"],
            [{"type": "test"}, {"type": "sample"}]
        )

        assert self.collection.count() == 2

    def test_name_property(self):
        """Test collection name property."""
        assert self.collection.name == "test_mock_collection"

    def test_metadata_property(self):
        """Test collection metadata property."""
        assert self.collection.metadata == {"hnsw:space": "cosine"}


# ===== MOCK DATABASE TESTS =====

class TestMockDatabase:
    """Test MockDatabase implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.database = MockDatabase()

    def test_init(self):
        """Test mock database initialization."""
        assert self.database.db_path == "/tmp/mock_db"
        assert self.database.distance_metric == "cosine"
        assert self.database.persist is True
        assert self.database.allow_reset is True
        assert len(self.database._collections) == 0

    def test_init_custom_params(self):
        """Test mock database initialization with custom parameters."""
        db = MockDatabase(
            db_path="/custom/path",
            distance_metric="l2",
            persist=False,
            allow_reset=False
        )

        assert db.db_path == "/custom/path"
        assert db.distance_metric == "l2"
        assert db.persist is False
        assert db.allow_reset is False

    def test_create_collection_new(self):
        """Test creating a new collection."""
        collection = self.database.create_collection("test_collection")

        assert isinstance(collection, MockCollection)
        assert collection.name == "test_collection"
        assert "test_collection" in self.database._collections

    def test_create_collection_existing(self):
        """Test creating an existing collection without reset."""
        collection1 = self.database.create_collection("test_collection")
        collection2 = self.database.create_collection("test_collection")

        # Should return the same collection
        assert collection1 is collection2

    def test_create_collection_with_reset(self):
        """Test creating collection with reset."""
        collection1 = self.database.create_collection("test_collection")
        collection1.add(["doc1"], ["Document 1"], [{"type": "test"}])

        collection2 = self.database.create_collection("test_collection", reset_if_exists=True)

        # Should be a new collection
        assert collection1 is not collection2
        assert collection2.count() == 0

    def test_get_collection_existing(self):
        """Test getting an existing collection."""
        created_collection = self.database.create_collection("test_collection")
        retrieved_collection = self.database.get_collection("test_collection")

        assert created_collection is retrieved_collection

    def test_get_collection_nonexistent(self):
        """Test getting a nonexistent collection."""
        with pytest.raises(ValueError, match="Collection nonexistent does not exist"):
            self.database.get_collection("nonexistent")

    def test_delete_collection_existing(self):
        """Test deleting an existing collection."""
        self.database.create_collection("test_collection")
        assert "test_collection" in self.database._collections

        self.database.delete_collection("test_collection")
        assert "test_collection" not in self.database._collections

    def test_delete_collection_nonexistent(self):
        """Test deleting a nonexistent collection."""
        # Should not raise an error
        self.database.delete_collection("nonexistent")

    def test_list_collections_empty(self):
        """Test listing collections when empty."""
        collections = self.database.list_collections()
        assert collections == []

    def test_list_collections_with_data(self):
        """Test listing collections with data."""
        self.database.create_collection("collection1")
        self.database.create_collection("collection2")

        collections = self.database.list_collections()
        assert set(collections) == {"collection1", "collection2"}

    def test_reset(self):
        """Test resetting the database."""
        self.database.create_collection("collection1")
        self.database.create_collection("collection2")
        assert len(self.database._collections) == 2

        self.database.reset()
        assert len(self.database._collections) == 0


# ===== FACTORY FUNCTION TESTS =====

class TestFactoryFunctions:
    """Test factory functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test_db")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.vectordb.storage_factories.ChromaDBDatabase')
    def test_create_chromadb_database(self, mock_chromadb_class):
        """Test ChromaDB database factory function."""
        mock_instance = Mock()
        mock_chromadb_class.return_value = mock_instance

        result = create_chromadb_database(
            db_path=self.db_path,
            distance_metric="l2",
            persist=False,
            allow_reset=True
        )

        mock_chromadb_class.assert_called_once_with(
            db_path=self.db_path,
            distance_metric="l2",
            persist=False,
            allow_reset=True
        )
        assert result is mock_instance

    def test_create_chromadb_database_defaults(self):
        """Test ChromaDB database factory with default parameters."""
        with patch('src.vectordb.storage_factories.ChromaDBDatabase') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            result = create_chromadb_database(self.db_path)

            mock_class.assert_called_once_with(
                db_path=self.db_path,
                distance_metric="cosine",
                persist=True,
                allow_reset=False
            )

    def test_create_mock_database(self):
        """Test mock database factory function."""
        result = create_mock_database(
            db_path="/custom/path",
            distance_metric="l2",
            persist=False,
            allow_reset=False
        )

        assert isinstance(result, MockDatabase)
        assert result.db_path == "/custom/path"
        assert result.distance_metric == "l2"
        assert result.persist is False
        assert result.allow_reset is False

    def test_create_mock_database_defaults(self):
        """Test mock database factory with default parameters."""
        result = create_mock_database()

        assert isinstance(result, MockDatabase)
        assert result.db_path == "/tmp/mock_db"
        assert result.distance_metric == "cosine"
        assert result.persist is True
        assert result.allow_reset is True


# ===== INTEGRATION TESTS =====

class TestIntegration:
    """Integration tests for storage factories."""

    def test_chromadb_collection_protocol_compliance(self):
        """Test that ChromaDBCollection implements VectorCollection protocol."""
        mock_chroma_collection = Mock()
        mock_chroma_collection.name = "test"
        mock_chroma_collection.metadata = {}

        collection = ChromaDBCollection(mock_chroma_collection)

        # Test all protocol methods exist and are callable
        assert hasattr(collection, 'add')
        assert hasattr(collection, 'query')
        assert hasattr(collection, 'get')
        assert hasattr(collection, 'update')
        assert hasattr(collection, 'delete')
        assert hasattr(collection, 'count')
        assert hasattr(collection, 'name')
        assert hasattr(collection, 'metadata')

    def test_mock_collection_protocol_compliance(self):
        """Test that MockCollection implements VectorCollection protocol."""
        collection = MockCollection()

        # Test all protocol methods exist and are callable
        assert hasattr(collection, 'add')
        assert hasattr(collection, 'query')
        assert hasattr(collection, 'get')
        assert hasattr(collection, 'update')
        assert hasattr(collection, 'delete')
        assert hasattr(collection, 'count')
        assert hasattr(collection, 'name')
        assert hasattr(collection, 'metadata')

    def test_database_protocol_compliance(self):
        """Test that database implementations follow VectorDatabase protocol."""
        mock_db = MockDatabase()

        # Test all protocol methods exist
        assert hasattr(mock_db, 'create_collection')
        assert hasattr(mock_db, 'get_collection')
        assert hasattr(mock_db, 'delete_collection')
        assert hasattr(mock_db, 'list_collections')
        assert hasattr(mock_db, 'reset')

    def test_end_to_end_mock_workflow(self):
        """Test end-to-end workflow with mock implementations."""
        db = create_mock_database()

        # Create collection
        collection = db.create_collection("test_collection")

        # Add documents
        collection.add(
            ids=["doc1", "doc2"],
            documents=["Document 1", "Document 2"],
            metadatas=[{"type": "test"}, {"type": "sample"}],
            embeddings=[np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        )

        # Query documents
        results = collection.query(query_texts=["test"], n_results=2)
        assert len(results["ids"][0]) == 2

        # Update document
        collection.update(ids=["doc1"], documents=["Updated Document 1"])

        # Get document
        get_results = collection.get(ids=["doc1"])
        assert get_results["documents"][0] == "Updated Document 1"

        # Count documents
        assert collection.count() == 2

        # Delete document
        collection.delete(ids=["doc2"])
        assert collection.count() == 1

        # List collections
        collections = db.list_collections()
        assert "test_collection" in collections

        # Delete collection
        db.delete_collection("test_collection")
        assert len(db.list_collections()) == 0

    def test_numpy_embedding_conversion(self):
        """Test numpy array to list conversion in ChromaDB operations."""
        mock_chroma_collection = Mock()
        collection = ChromaDBCollection(mock_chroma_collection)

        # Test with numpy arrays
        embeddings = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]

        collection.add(
            ids=["doc1", "doc2"],
            documents=["Doc 1", "Doc 2"],
            metadatas=[{}, {}],
            embeddings=embeddings
        )

        # Verify conversion to lists
        args, kwargs = mock_chroma_collection.add.call_args
        assert kwargs["embeddings"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_error_propagation(self):
        """Test that errors are properly propagated from underlying implementations."""
        mock_chroma_collection = Mock()
        mock_chroma_collection.add.side_effect = RuntimeError("Underlying error")

        collection = ChromaDBCollection(mock_chroma_collection)

        with pytest.raises(RuntimeError, match="Underlying error"):
            collection.add(["doc1"], ["Document 1"], [{}])