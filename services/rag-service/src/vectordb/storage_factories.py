"""
Factory implementations for vector database components.
Provides production ChromaDB implementations of storage protocols.
"""

import logging
from pathlib import Path
from typing import Any, cast

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata, WhereDocument
from chromadb.config import Settings

from .storage import VectorCollection, VectorDatabase

logger = logging.getLogger(__name__)


class ChromaDBCollection(VectorCollection):
    """ChromaDB implementation of VectorCollection protocol."""

    def __init__(self, collection: Collection):
        self._collection = collection
        self.logger = logging.getLogger(__name__)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """Add documents to collection."""
        try:
            if embeddings is not None:
                # Convert numpy arrays to lists for ChromaDB
                embedding_lists = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]

                self._collection.add(
                    ids=ids, documents=documents, metadatas=cast(list[Metadata], metadatas), embeddings=embedding_lists
                )
            else:
                self._collection.add(ids=ids, documents=documents, metadatas=cast(list[Metadata], metadatas))
            self.logger.debug(f"Added {len(documents)} documents to collection")

        except Exception as e:
            self.logger.error(f"Failed to add documents to collection: {e}")
            raise

    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[np.ndarray] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Query collection for similar documents."""
        try:
            final_include = include if include is not None else ["documents", "metadatas", "distances"]

            if query_embeddings is not None:
                # Convert numpy arrays to lists for ChromaDB
                embedding_lists = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in query_embeddings]

                results = self._collection.query(
                    query_embeddings=embedding_lists,
                    n_results=n_results,
                    where=where,
                    where_document=cast(WhereDocument, where_document),
                    include=final_include,  # type: ignore[arg-type]
                )
            else:
                results = self._collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    where_document=cast(WhereDocument, where_document),
                    include=final_include,  # type: ignore[arg-type]
                )
            self.logger.debug("Query returned results for collection")

            # Convert ChromaDB QueryResult to dict for compatibility
            return dict(results)

        except Exception as e:
            self.logger.error(f"Failed to query collection: {e}")
            raise

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get documents from collection."""
        try:
            final_include = include if include is not None else ["documents", "metadatas"]

            # Convert ChromaDB GetResult to dict for compatibility
            results = self._collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                include=final_include,  # type: ignore[arg-type]
            )
            return dict(results)

        except Exception as e:
            self.logger.error(f"Failed to get documents from collection: {e}")
            raise

    def update(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """Update documents in collection."""
        try:
            if embeddings is not None:
                # Convert numpy arrays to lists for ChromaDB
                embedding_lists = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
                self._collection.update(
                    ids=ids, documents=documents, metadatas=cast(list[Metadata], metadatas), embeddings=embedding_lists
                )
            else:
                self._collection.update(ids=ids, documents=documents, metadatas=cast(list[Metadata], metadatas))
            self.logger.debug(f"Updated {len(ids)} documents in collection")

        except Exception as e:
            self.logger.error(f"Failed to update documents in collection: {e}")
            raise

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        """Delete documents from collection."""
        try:
            if ids is None and where is None:
                raise ValueError("Either ids or where filter must be provided")

            self._collection.delete(ids=ids, where=where)
            self.logger.debug("Deleted documents from collection")

        except Exception as e:
            self.logger.error(f"Failed to delete documents from collection: {e}")
            raise

    def count(self) -> int:
        """Get document count in collection."""
        try:
            return self._collection.count()
        except Exception as e:
            self.logger.error(f"Failed to get collection count: {e}")
            raise

    @property
    def name(self) -> str:
        """Get collection name."""
        return self._collection.name

    @property
    def metadata(self) -> dict[str, Any]:
        """Get collection metadata."""
        return self._collection.metadata or {}


class ChromaDBDatabase(VectorDatabase):
    """ChromaDB implementation of VectorDatabase protocol."""

    def __init__(self, db_path: str, distance_metric: str = "cosine", persist: bool = True, allow_reset: bool = False):
        self.db_path = db_path
        self.distance_metric = distance_metric
        self.persist = persist
        self.allow_reset = allow_reset
        self.logger = logging.getLogger(__name__)

        # Create database directory
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Create ChromaDB client with appropriate settings."""
        try:
            settings = Settings(
                persist_directory=self.db_path if self.persist else None,
                allow_reset=self.allow_reset,
                anonymized_telemetry=False,
            )

            if self.persist:
                client = chromadb.PersistentClient(path=self.db_path, settings=settings)
                self.logger.info(f"Created persistent ChromaDB client at {self.db_path}")
            else:
                client = chromadb.EphemeralClient(settings=settings)
                self.logger.info("Created ephemeral ChromaDB client")

            return client

        except Exception as e:
            self.logger.error(f"Failed to create ChromaDB client: {e}")
            raise

    def create_collection(self, name: str, reset_if_exists: bool = False) -> VectorCollection:
        """Create or get collection."""
        try:
            if reset_if_exists:
                try:
                    self._client.delete_collection(name)
                    self.logger.info(f"Deleted existing collection: {name}")
                except Exception as e:
                    self.logger.debug(f"Collection {name} did not exist for deletion: {e}")

            # Create or get collection
            collection = self._client.get_or_create_collection(name=name, metadata={"hnsw:space": self.distance_metric})

            self.logger.info(f"Collection '{name}' ready with {self.distance_metric} metric")
            return ChromaDBCollection(collection)

        except Exception as e:
            self.logger.error(f"Failed to create collection {name}: {e}")
            raise

    def get_collection(self, name: str) -> VectorCollection:
        """Get existing collection."""
        try:
            collection = self._client.get_collection(name)
            self.logger.debug(f"Retrieved collection: {name}")
            return ChromaDBCollection(collection)

        except Exception as e:
            self.logger.error(f"Failed to get collection {name}: {e}")
            raise

    def delete_collection(self, name: str) -> None:
        """Delete collection."""
        try:
            self._client.delete_collection(name)
            self.logger.info(f"Deleted collection: {name}")

        except Exception as e:
            self.logger.error(f"Failed to delete collection {name}: {e}")
            raise

    def list_collections(self) -> list[str]:
        """List all collections."""
        try:
            collections = self._client.list_collections()
            collection_names = [col.name for col in collections]
            self.logger.debug(f"Found {len(collection_names)} collections")
            return collection_names

        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            raise

    def reset(self) -> None:
        """Reset entire database."""
        try:
            if not self.allow_reset:
                raise ValueError("Database reset not allowed - check allow_reset setting")

            self._client.reset()
            self.logger.warning("Database reset - all collections deleted")

        except Exception as e:
            self.logger.error(f"Failed to reset database: {e}")
            raise


class MockCollection(VectorCollection):
    """Mock collection for testing."""

    def __init__(self, name: str = "test_collection"):
        self._name = name
        self._metadata = {"hnsw:space": "cosine"}
        self._documents: dict[str, dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """Add documents to mock collection."""
        for i, doc_id in enumerate(ids):
            self._documents[doc_id] = {
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i] if embeddings else None,
            }
        self.logger.debug(f"Added {len(ids)} documents to mock collection")

    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[np.ndarray] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Mock query returns first n_results documents."""
        include = include or ["documents", "metadatas", "distances"]

        # Simple mock: return first n_results documents
        doc_items = list(self._documents.items())[:n_results]

        results: dict[str, Any] = {"ids": [[]]}
        if "documents" in include:
            results["documents"] = [[]]
        if "metadatas" in include:
            results["metadatas"] = [[]]
        if "distances" in include:
            results["distances"] = [[]]

        for doc_id, doc_data in doc_items:
            results["ids"][0].append(doc_id)
            if "documents" in include:
                results["documents"][0].append(doc_data["document"])
            if "metadatas" in include:
                results["metadatas"][0].append(doc_data["metadata"])
            if "distances" in include:
                results["distances"][0].append(0.5)  # Mock distance

        return results

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get documents from mock collection."""
        include = include or ["documents", "metadatas"]

        if ids:
            doc_items = [(doc_id, self._documents[doc_id]) for doc_id in ids if doc_id in self._documents]
        else:
            doc_items = list(self._documents.items())

        results: dict[str, Any] = {"ids": []}
        if "documents" in include:
            results["documents"] = []
        if "metadatas" in include:
            results["metadatas"] = []

        for doc_id, doc_data in doc_items:
            results["ids"].append(doc_id)
            if "documents" in include:
                results["documents"].append(doc_data["document"])
            if "metadatas" in include:
                results["metadatas"].append(doc_data["metadata"])

        return results

    def update(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """Update documents in mock collection."""
        for i, doc_id in enumerate(ids):
            if doc_id in self._documents:
                if documents:
                    self._documents[doc_id]["document"] = documents[i]
                if metadatas:
                    self._documents[doc_id]["metadata"] = metadatas[i]
                if embeddings:
                    self._documents[doc_id]["embedding"] = embeddings[i]

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        """Delete documents from mock collection."""
        if ids:
            for doc_id in ids:
                self._documents.pop(doc_id, None)
        else:
            # Mock: delete all if no specific filter
            self._documents.clear()

    def count(self) -> int:
        """Get document count."""
        return len(self._documents)

    @property
    def name(self) -> str:
        """Get collection name."""
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        """Get collection metadata."""
        return self._metadata


class MockDatabase(VectorDatabase):
    """Mock database for testing."""

    def __init__(
        self,
        db_path: str = "/tmp/mock_db",
        distance_metric: str = "cosine",
        persist: bool = True,
        allow_reset: bool = True,
    ):
        self.db_path = db_path
        self.distance_metric = distance_metric
        self.persist = persist
        self.allow_reset = allow_reset
        self._collections: dict[str, MockCollection] = {}
        self.logger = logging.getLogger(__name__)

    def create_collection(self, name: str, reset_if_exists: bool = False) -> VectorCollection:
        """Create or get mock collection."""
        if reset_if_exists and name in self._collections:
            del self._collections[name]

        if name not in self._collections:
            self._collections[name] = MockCollection(name)

        return self._collections[name]

    def get_collection(self, name: str) -> VectorCollection:
        """Get existing mock collection."""
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete mock collection."""
        if name in self._collections:
            del self._collections[name]

    def list_collections(self) -> list[str]:
        """List all mock collections."""
        return list(self._collections.keys())

    def reset(self) -> None:
        """Reset mock database."""
        self._collections.clear()


def create_chromadb_database(
    db_path: str, distance_metric: str = "cosine", persist: bool = True, allow_reset: bool = False
) -> VectorDatabase:
    """Factory function to create ChromaDB database."""
    return ChromaDBDatabase(db_path=db_path, distance_metric=distance_metric, persist=persist, allow_reset=allow_reset)


def create_vector_database(
    db_path: str, distance_metric: str = "cosine", persist: bool = True, allow_reset: bool = False
) -> VectorDatabase:
    """Factory function to create vector database."""
    return create_chromadb_database(
        db_path=db_path, distance_metric=distance_metric, persist=persist, allow_reset=allow_reset
    )


def create_mock_database(
    db_path: str = "/tmp/mock_db", distance_metric: str = "cosine", persist: bool = True, allow_reset: bool = True
) -> VectorDatabase:
    """Factory function to create mock database."""
    return MockDatabase(db_path=db_path, distance_metric=distance_metric, persist=persist, allow_reset=allow_reset)
