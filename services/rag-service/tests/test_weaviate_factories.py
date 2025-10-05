"""
Tests for Weaviate factory functions and classes.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import numpy as np
from typing import Dict, Any, List
import uuid
import time

from src.vectordb.weaviate_factories import (
    WeaviateCollection,
    WeaviateDatabase,
    create_weaviate_client,
    create_weaviate_database,
    create_weaviate_collection,
)
from src.vectordb.weaviate_config import (
    WeaviateConfiguration,
    WeaviateConnectionConfig,
    WeaviateHNSWConfig,
    WeaviateCompressionConfig,
    WeaviateGeneralConfig,
    WeaviateBackupConfig,
    create_weaviate_configuration,
)
from src.vectordb.storage import (
    VectorSearchResults,
    VectorSearchResult,
)


class TestWeaviateCollection:
    """Test WeaviateCollection implementation."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Weaviate client."""
        client = Mock()

        # Setup collections mock
        collections = Mock()
        client.collections = collections

        # Setup collection mock
        collection = Mock()
        collections.get.return_value = collection

        # Setup data operations
        collection.data = Mock()
        collection.query = Mock()
        collection.aggregate = Mock()

        return client

    @pytest.fixture
    def weaviate_config(self):
        """Create a test Weaviate configuration."""
        return WeaviateConfiguration(
            connection=WeaviateConnectionConfig(
                host="localhost",
                port=8080,
                grpc_port=50051,
                scheme="http",
                timeout=30.0,
                startup_period=5,
            ),
            hnsw=WeaviateHNSWConfig(
                ef_construction=200,
                ef=100,
                max_connections=16,
                ef_dynamic=100,
                cleanup_interval_seconds=300,
                vector_cache_max_objects=1000000,
            ),
            compression=WeaviateCompressionConfig(
                enabled=False,
                type="pq",
                rescore_limit=200,
                cache=True,
            ),
            general=WeaviateGeneralConfig(
                batch_size=100,
                timeout=30.0,
                max_retries=3,
            ),
            backup=WeaviateBackupConfig(
                enabled=False,
            ),
        )

    def test_collection_init(self, mock_client, weaviate_config):
        """Test WeaviateCollection initialization."""
        # Mock the exists check
        mock_client.collections.get.side_effect = [
            Exception("Collection not found"),  # First call - doesn't exist
            Mock(),  # Second call after creation
        ]

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        assert collection.class_name == "test_collection"
        assert collection.client == mock_client
        assert collection.config == weaviate_config
        assert collection.name == "test_collection"

    def test_collection_metadata(self, mock_client, weaviate_config):
        """Test collection metadata property."""
        mock_client.collections.get.return_value = Mock()

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        metadata = collection.metadata
        assert metadata["class_name"] == "test_collection"
        assert "config" in metadata
        assert metadata["config"]["hnsw"]["ef"] == 100
        assert metadata["config"]["compression"]["enabled"] is False

    def test_collection_add_documents(self, mock_client, weaviate_config):
        """Test adding documents to collection."""
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.data.insert_many = Mock()

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        # Test data
        ids = ["doc1", "doc2"]
        documents = ["Document 1 content", "Document 2 content"]
        metadatas = [
            {"source_file": "file1.txt", "chunk_index": 0},
            {"source_file": "file2.txt", "chunk_index": 1}
        ]
        embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]

        # Add documents
        collection.add(ids, documents, metadatas, embeddings)

        # Verify insert_many was called
        mock_collection.data.insert_many.assert_called()

        # Check that objects were created with proper format
        call_args = mock_collection.data.insert_many.call_args[0][0]
        assert len(call_args) == 2  # Two documents

        # Verify first object properties
        first_obj = call_args[0]
        assert first_obj.properties["content"] == "Document 1 content"
        assert first_obj.properties["source_file"] == "file1.txt"
        assert first_obj.properties["chunk_index"] == 0

    def test_collection_query_with_embeddings(self, mock_client, weaviate_config):
        """Test querying collection with embeddings."""
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection

        # Setup mock response
        mock_obj = Mock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {
            "content": "Test content",
            "source_file": "test.txt",
            "chunk_index": 0
        }
        mock_obj.metadata = Mock()
        mock_obj.metadata.distance = 0.5
        mock_obj.metadata.certainty = 0.75

        mock_response = Mock()
        mock_response.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_response

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        # Query with embeddings
        query_embedding = np.array([0.1, 0.2, 0.3])
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        # Verify results
        assert isinstance(results, VectorSearchResults)
        assert len(results.results) == 1
        assert results.results[0].content == "Test content"
        assert abs(results.results[0].distance - 0.5) < 0.0001

        # Verify near_vector was called with correct parameters
        mock_collection.query.near_vector.assert_called_once()
        call_args = mock_collection.query.near_vector.call_args
        assert call_args[1]["limit"] == 5

    def test_collection_query_with_text(self, mock_client, weaviate_config):
        """Test querying collection with text."""
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection

        # Setup mock response
        mock_obj = Mock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {"content": "Test content"}
        mock_obj.metadata = Mock()
        mock_obj.metadata.certainty = 0.9
        mock_obj.metadata.distance = None  # Set distance to None so it uses certainty

        mock_response = Mock()
        mock_response.objects = [mock_obj]
        mock_collection.query.near_text.return_value = mock_response

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        # Query with text
        results = collection.query(
            query_texts=["test query"],
            n_results=10
        )

        # Verify results
        assert isinstance(results, VectorSearchResults)
        assert len(results.results) == 1
        # Distance should be 1.0 - certainty
        assert abs(results.results[0].distance - 0.1) < 0.0001  # Use approximate comparison for floats

        # Verify near_text was called
        mock_collection.query.near_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_collection_search(self, mock_client, weaviate_config):
        """Test async search method."""
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection

        # Setup mock response for query
        mock_obj = Mock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {"content": "Test content"}
        mock_obj.metadata = Mock()
        mock_obj.metadata.distance = 0.3  # Low distance = high similarity

        mock_response = Mock()
        mock_response.objects = [mock_obj]
        mock_collection.query.near_text.return_value = mock_response

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        # Search with threshold
        results = await collection.search(
            query_text="test query",
            k=5,
            similarity_threshold=0.5
        )

        # Verify results
        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        assert "similarity_score" in results[0]
        # Score should be 1 - (distance/2) = 1 - (0.3/2) = 0.85
        assert results[0]["similarity_score"] > 0.5

    def test_collection_get_documents(self, mock_client, weaviate_config):
        """Test getting documents by IDs."""
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection

        # Setup mock response
        mock_obj = Mock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {
            "content": "Test content",
            "source_file": "test.txt"
        }

        mock_response = Mock()
        mock_response.objects = [mock_obj]
        mock_collection.query.fetch_objects_by_ids.return_value = mock_response

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        # Get documents by IDs
        result = collection.get(ids=["doc1", "doc2"])

        # Verify result format (ChromaDB compatibility)
        assert "documents" in result
        assert "metadatas" in result
        assert "ids" in result
        assert len(result["documents"][0]) == 1

    def test_collection_update_documents(self, mock_client, weaviate_config):
        """Test updating documents."""
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.data.update = Mock()

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        # Update documents
        ids = ["doc1", "doc2"]
        documents = ["Updated content 1", "Updated content 2"]
        metadatas = [
            {"source_file": "updated1.txt"},
            {"source_file": "updated2.txt"}
        ]

        collection.update(ids, documents, metadatas)

        # Verify update was called for each document
        assert mock_collection.data.update.call_count == 2

    def test_collection_delete_documents(self, mock_client, weaviate_config):
        """Test deleting documents."""
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.data.delete_by_id = Mock()

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        # Delete documents by IDs
        ids = ["doc1", "doc2", "doc3"]
        collection.delete(ids=ids)

        # Verify delete was called for each ID
        assert mock_collection.data.delete_by_id.call_count == 3

    def test_collection_count(self, mock_client, weaviate_config):
        """Test counting documents."""
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection

        mock_response = Mock()
        mock_response.total_count = 42
        mock_collection.aggregate.over_all.return_value = mock_response

        collection = WeaviateCollection(
            client=mock_client,
            class_name="test_collection",
            config=weaviate_config
        )

        # Count documents
        count = collection.count()

        assert count == 42
        mock_collection.aggregate.over_all.assert_called_once_with(total_count=True)


class TestWeaviateDatabase:
    """Test WeaviateDatabase implementation."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Weaviate client."""
        client = Mock()
        client.collections = Mock()
        client.is_ready = Mock(return_value=True)  # For _test_connection
        return client

    @pytest.fixture
    def weaviate_config(self):
        """Create a test Weaviate configuration."""
        return WeaviateConfiguration(
            connection=WeaviateConnectionConfig(
                host="localhost",
                port=8080,
                grpc_port=50051,
                scheme="http",
                timeout=30.0,
                startup_period=5,
            ),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(),
            backup=WeaviateBackupConfig(),
        )

    def test_database_init(self, mock_client, weaviate_config):
        """Test WeaviateDatabase initialization."""
        database = WeaviateDatabase(mock_client, weaviate_config)

        assert database.client == mock_client
        assert database.config == weaviate_config
        assert database._collections == {}

    def test_database_get_collection(self, mock_client, weaviate_config):
        """Test getting a collection."""
        database = WeaviateDatabase(mock_client, weaviate_config)

        # Get collection (should create it)
        collection = database.get_collection("test_collection")

        assert isinstance(collection, WeaviateCollection)
        assert collection.class_name == "test_collection"

        # Get same collection again (should return cached)
        collection2 = database.get_collection("test_collection")
        assert collection is collection2

    def test_database_create_collection(self, mock_client, weaviate_config):
        """Test creating a collection explicitly."""
        database = WeaviateDatabase(mock_client, weaviate_config)

        collection = database.create_collection("new_collection")

        assert isinstance(collection, WeaviateCollection)
        assert collection.class_name == "new_collection"
        assert "new_collection" in database._collections

    def test_database_list_collections(self, mock_client, weaviate_config):
        """Test listing collections."""
        # Setup mock collections - list_all returns collection names as strings
        mock_client.collections.list_all.return_value = [
            "collection1",
            "collection2"
        ]

        database = WeaviateDatabase(mock_client, weaviate_config)

        collections = database.list_collections()

        assert len(collections) == 2
        assert "collection1" in collections
        assert "collection2" in collections

    def test_database_delete_collection(self, mock_client, weaviate_config):
        """Test deleting a collection."""
        mock_client.collections.delete = Mock()

        database = WeaviateDatabase(mock_client, weaviate_config)

        # Create and then delete a collection
        database.get_collection("test_collection")
        assert "test_collection" in database._collections

        database.delete_collection("test_collection")

        mock_client.collections.delete.assert_called_once_with("test_collection")
        assert "test_collection" not in database._collections



class TestFactoryFunctions:
    """Test factory functions for creating Weaviate components."""

    @patch('src.vectordb.weaviate_factories.weaviate.connect_to_local')
    def test_create_weaviate_client(self, mock_connect):
        """Test creating a Weaviate client."""
        mock_client = Mock()
        mock_client.is_ready.return_value = True
        mock_connect.return_value = mock_client

        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(
                host="localhost",
                port=8080,
                grpc_port=50051,
                scheme="http",
                timeout=30.0,
                startup_period=5,
            ),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(),
            backup=WeaviateBackupConfig(),
        )

        client = create_weaviate_client(config)

        assert client == mock_client
        mock_connect.assert_called_once_with(
            host="localhost",
            port=8080,
            grpc_port=50051
        )

    @patch('src.vectordb.weaviate_factories.weaviate.connect_to_local')
    def test_create_weaviate_client_timeout(self, mock_connect):
        """Test client creation timeout."""
        mock_client = Mock()
        mock_client.is_ready.return_value = False
        mock_connect.return_value = mock_client

        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(
                host="localhost",
                port=8080,
                grpc_port=50051,
                scheme="http",
                timeout=1.0,  # Short timeout for test
                startup_period=1,
            ),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(),
            backup=WeaviateBackupConfig(),
        )

        with pytest.raises(ConnectionError, match="Failed to create Weaviate client"):
            create_weaviate_client(config)

    @patch('src.vectordb.weaviate_factories.create_weaviate_client')
    @patch('src.vectordb.weaviate_factories.create_weaviate_configuration')
    def test_create_weaviate_database(self, mock_create_config, mock_create_client):
        """Test creating a Weaviate database."""
        mock_config = Mock()
        mock_create_config.return_value = mock_config

        mock_client = Mock()
        mock_create_client.return_value = mock_client

        config_dict = {
            "weaviate": {
                "connection": {
                    "url": "http://localhost:8080"
                }
            }
        }

        database = create_weaviate_database(config_dict, "en")

        assert isinstance(database, WeaviateDatabase)
        assert database.client == mock_client
        assert database.config == mock_config

        mock_create_config.assert_called_once_with(config_dict)
        mock_create_client.assert_called_once_with(mock_config)

    @patch('src.vectordb.weaviate_factories.WeaviateCollection')
    def test_create_weaviate_collection(self, mock_collection_class):
        """Test creating a Weaviate collection."""
        mock_client = Mock()
        mock_config = Mock()
        mock_collection = Mock()
        mock_collection_class.return_value = mock_collection

        collection = create_weaviate_collection(
            client=mock_client,
            collection_name="test_collection",
            config=mock_config
        )

        assert collection == mock_collection
        mock_collection_class.assert_called_once_with(
            mock_client,
            "test_collection",
            mock_config
        )


class TestWeaviateCollectionIntegration:
    """Integration tests for WeaviateCollection with more complex scenarios."""

    @pytest.fixture
    def mock_client_with_responses(self):
        """Create a mock client with predefined responses."""
        client = Mock()
        collections = Mock()
        client.collections = collections

        # Setup collection mock with more detailed responses
        collection = Mock()
        collections.get.return_value = collection

        # Setup data operations
        collection.data = Mock()
        collection.query = Mock()
        collection.aggregate = Mock()

        # Setup batch response
        collection.data.insert_many = Mock()

        return client

    def test_add_documents_with_validation(self, mock_client_with_responses):
        """Test adding documents with input validation."""
        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(batch_size=2),  # Small batch for testing
            backup=WeaviateBackupConfig(),
        )

        collection = WeaviateCollection(
            client=mock_client_with_responses,
            class_name="test",
            config=config
        )

        # Test with mismatched lengths
        with pytest.raises(ValueError, match="Length mismatch"):
            collection.add(
                ids=["id1", "id2"],
                documents=["doc1"],  # Wrong length
                metadatas=[{}, {}],
                embeddings=None
            )

    def test_query_distance_extraction_scenarios(self, mock_client_with_responses):
        """Test various distance extraction scenarios."""
        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(),
            backup=WeaviateBackupConfig(),
        )

        collection = WeaviateCollection(
            client=mock_client_with_responses,
            class_name="test",
            config=config
        )

        # Test scenario 1: Has distance
        mock_obj1 = Mock()
        mock_obj1.uuid = uuid.uuid4()
        mock_obj1.properties = {"content": "Test 1"}
        mock_obj1.metadata = Mock()
        mock_obj1.metadata.distance = 0.25
        mock_obj1.metadata.certainty = None

        # Test scenario 2: Has certainty only
        mock_obj2 = Mock()
        mock_obj2.uuid = uuid.uuid4()
        mock_obj2.properties = {"content": "Test 2"}
        mock_obj2.metadata = Mock()
        mock_obj2.metadata.distance = None
        mock_obj2.metadata.certainty = 0.8

        # Test scenario 3: Has neither
        mock_obj3 = Mock()
        mock_obj3.uuid = uuid.uuid4()
        mock_obj3.properties = {"content": "Test 3"}
        mock_obj3.metadata = Mock()
        mock_obj3.metadata.distance = None
        mock_obj3.metadata.certainty = None

        mock_response = Mock()
        mock_response.objects = [mock_obj1, mock_obj2, mock_obj3]

        mock_client_with_responses.collections.get.return_value.query.near_text.return_value = mock_response

        results = collection.query(query_texts=["test"], n_results=3)

        assert len(results.results) == 3
        assert abs(results.results[0].distance - 0.25) < 0.0001  # Direct distance
        assert abs(results.results[1].distance - 0.2) < 0.0001   # 1.0 - 0.8
        assert abs(results.results[2].distance - 1.0) < 0.0001   # Default

    def test_upsert_operation(self, mock_client_with_responses):
        """Test upsert operation."""
        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(),
            backup=WeaviateBackupConfig(),
        )

        collection = WeaviateCollection(
            client=mock_client_with_responses,
            class_name="test",
            config=config
        )

        # Test upsert
        ids = ["doc1", "doc2"]
        documents = ["Content 1", "Content 2"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        # Should call add internally
        mock_client_with_responses.collections.get.return_value.data.insert_many.assert_called()

        # Test upsert without documents (should raise error)
        with pytest.raises(ValueError, match="Documents are required"):
            collection.upsert(ids=ids, metadatas=metadatas)


class TestErrorHandling:
    """Test error handling in Weaviate components."""

    def test_collection_creation_failure(self):
        """Test handling of collection creation failures."""
        mock_client = Mock()
        # Setup collections mock
        mock_client.collections = Mock()
        # First call to get() should raise (collection doesn't exist)
        # Second call to create() should also raise (connection failed)
        mock_client.collections.get.side_effect = Exception("Not found")
        mock_client.collections.create.side_effect = Exception("Connection failed")

        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(),
            backup=WeaviateBackupConfig(),
        )

        with pytest.raises(Exception, match="Connection failed"):
            WeaviateCollection(
                client=mock_client,
                class_name="test",
                config=config
            )

    def test_database_operation_failure(self):
        """Test handling of database operation failures."""
        mock_client = Mock()
        mock_client.collections.list_all.side_effect = Exception("List failed")

        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(),
            backup=WeaviateBackupConfig(),
        )

        database = WeaviateDatabase(mock_client, config)

        # list_collections catches exceptions and returns empty list
        result = database.list_collections()
        assert result == []