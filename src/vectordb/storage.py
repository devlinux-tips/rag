"""
ChromaDB storage operations for Croatian RAG system.
Handles vector database operations, collections, and persistence.
"""

import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from ..utils.config_loader import get_croatian_vectordb, get_search_config, get_storage_config
from ..utils.error_handler import create_config_loader, handle_config_error

logger = logging.getLogger(__name__)

# Create specialized config loaders
load_vectordb_config = create_config_loader("config/config.toml", __name__)
load_croatian_config = create_config_loader("config/croatian.toml", __name__)


@dataclass
class StorageConfig:
    """Configuration for ChromaDB storage."""

    db_path: str
    collection_name: str
    distance_metric: str
    persist: bool
    allow_reset: bool

    @classmethod
    def from_config(cls) -> "StorageConfig":
        """Load configuration from TOML files."""
        return load_vectordb_config(
            operation=lambda: cls(
                db_path=get_storage_config()["db_path"],
                collection_name=get_storage_config()["collection_name"],
                distance_metric=get_storage_config()["distance_metric"],
                persist=get_storage_config()["persist"],
                allow_reset=get_storage_config()["allow_reset"],
            ),
            fallback_value=cls(
                db_path="./data/chromadb",
                collection_name="croatian_documents",
                distance_metric="cosine",
                persist=True,
                allow_reset=True,
            ),
            section="[storage]",
            error_level="error",
        )


@dataclass
class DocumentMetadata:
    """Metadata structure for stored documents."""

    source: str
    title: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    language: str = None
    processed_at: Optional[str] = None
    content_type: Optional[str] = None
    file_size: Optional[int] = None

    def __post_init__(self):
        """Set default language from config if not provided."""
        if self.language is None:
            try:
                # Use Croatian language code from Croatian config
                from ..utils.config_loader import get_croatian_language_code

                self.language = get_croatian_language_code()
                # Note: No logging here as this is normal operation
            except Exception as e:
                self.language = "hr"
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load Croatian language code, using fallback 'hr': {e}")
                logger.warning("Check your config/croatian.toml file")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class ChromaDBStorage:
    """ChromaDB client for vector storage and retrieval."""

    def __init__(self, config: StorageConfig = None):
        """
        Initialize ChromaDB storage.

        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig.from_config()
        self.logger = logging.getLogger(__name__)

        # Create database directory
        Path(self.config.db_path).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and settings."""
        try:
            settings = Settings(
                persist_directory=self.config.db_path if self.config.persist else None,
                allow_reset=self.config.allow_reset,
                anonymized_telemetry=False,
            )

            if self.config.persist:
                self.client = chromadb.PersistentClient(path=self.config.db_path, settings=settings)
            else:
                self.client = chromadb.EphemeralClient(settings=settings)

            self.logger.info(f"ChromaDB client initialized with persist={self.config.persist}")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise

    def create_collection(
        self, collection_name: Optional[str] = None, reset_if_exists: bool = False
    ) -> Collection:
        """
        Create or get collection.

        Args:
            collection_name: Name of collection (uses config default if None)
            reset_if_exists: Whether to reset collection if it exists

        Returns:
            ChromaDB Collection object
        """
        collection_name = collection_name or self.config.collection_name

        try:
            if reset_if_exists:
                try:
                    self.client.delete_collection(collection_name)
                    self.logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    pass  # Collection might not exist

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )

            self.logger.info(f"Collection '{collection_name}' ready")
            return self.collection

        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            raise

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the collection.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            embeddings: Pre-computed embeddings (optional)
            ids: Document IDs (auto-generated if not provided)

        Returns:
            List of document IDs
        """
        if not self.collection:
            self.create_collection()

        # Validate inputs
        if len(documents) != len(metadatas):
            raise ValueError("Number of documents and metadatas must match")

        if embeddings and len(embeddings) != len(documents):
            raise ValueError("Number of embeddings and documents must match")

        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in documents]

        try:
            add_kwargs = {"documents": documents, "metadatas": metadatas, "ids": ids}

            if embeddings:
                add_kwargs["embeddings"] = embeddings

            self.collection.add(**add_kwargs)

            self.logger.info(f"Added {len(documents)} documents to collection")
            return ids

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise

    def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add document chunks with structured metadata.

        Args:
            chunks: List of document chunks with content and metadata
            embeddings: Pre-computed embeddings for chunks

        Returns:
            List of chunk IDs
        """
        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            # Extract content
            content = chunk.get("content", "")
            if not content:
                continue

            documents.append(content)

            # Prepare metadata
            metadata = chunk.copy()
            metadata.pop("content", None)  # Remove content from metadata

            # Ensure required metadata fields
            if "id" not in metadata:
                metadata["id"] = str(uuid.uuid4())

            metadatas.append(metadata)
            ids.append(metadata["id"])

        return self.add_documents(documents, metadatas, embeddings, ids)

    def query_similar(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Query for similar documents.

        Args:
            query_texts: Query texts (will be embedded automatically)
            query_embeddings: Pre-computed query embeddings
            n_results: Number of results to return (defaults from config)
            where: Metadata filter conditions
            where_document: Document content filter conditions
            include: What to include in results ("embeddings", "documents", "metadatas", "distances")

        Returns:
            Query results from ChromaDB
        """
        if not self.collection:
            raise ValueError("No collection available. Create collection first.")

        if not query_texts and not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided")

        # Use config default for n_results if not provided
        if n_results is None:
            try:
                search_config = get_search_config()
                n_results = search_config["top_k"]
                self.logger.debug(f"Using search config top_k: {n_results}")
            except Exception as e:
                n_results = 5
                self.logger.warning(
                    f"Failed to load search config for n_results, using fallback value {n_results}: {e}"
                )
                self.logger.warning("Check your config/vectordb.toml and config/search.toml files")

        # Default include list
        if include is None:
            include = ["documents", "metadatas", "distances"]

        try:
            query_kwargs = {"n_results": n_results, "include": include}

            if query_texts:
                query_kwargs["query_texts"] = query_texts

            if query_embeddings:
                query_kwargs["query_embeddings"] = query_embeddings

            if where:
                query_kwargs["where"] = where

            if where_document:
                query_kwargs["where_document"] = where_document

            results = self.collection.query(**query_kwargs)

            result_count = (
                len(results.get("ids", [[]])[0]) if results.get("ids") and results["ids"] else 0
            )
            self.logger.debug(f"Query returned {result_count} results")
            return results

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Get documents by IDs or metadata filters.

        Args:
            ids: Specific document IDs to retrieve
            where: Metadata filter conditions
            limit: Maximum number of results
            offset: Number of results to skip
            include: What to include in results

        Returns:
            Retrieved documents
        """
        if not self.collection:
            raise ValueError("No collection available")

        if include is None:
            include = ["documents", "metadatas"]

        try:
            get_kwargs = {"include": include}

            if ids:
                get_kwargs["ids"] = ids

            if where:
                get_kwargs["where"] = where

            if limit:
                get_kwargs["limit"] = limit

            if offset:
                get_kwargs["offset"] = offset

            return self.collection.get(**get_kwargs)

        except Exception as e:
            self.logger.error(f"Failed to get documents: {e}")
            raise

    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """
        Update existing documents.

        Args:
            ids: Document IDs to update
            documents: New document texts
            metadatas: New metadata
            embeddings: New embeddings
        """
        if not self.collection:
            raise ValueError("No collection available")

        try:
            update_kwargs = {"ids": ids}

            if documents:
                update_kwargs["documents"] = documents

            if metadatas:
                update_kwargs["metadatas"] = metadatas

            if embeddings:
                update_kwargs["embeddings"] = embeddings

            self.collection.update(**update_kwargs)

            self.logger.info(f"Updated {len(ids)} documents")

        except Exception as e:
            self.logger.error(f"Failed to update documents: {e}")
            raise

    def delete_documents(
        self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete documents by IDs or metadata filters.

        Args:
            ids: Document IDs to delete
            where: Metadata filter for deletion
        """
        if not self.collection:
            raise ValueError("No collection available")

        if not ids and not where:
            raise ValueError("Either ids or where filter must be provided")

        try:
            delete_kwargs = {}

            if ids:
                delete_kwargs["ids"] = ids

            if where:
                delete_kwargs["where"] = where

            self.collection.delete(**delete_kwargs)

            self.logger.info(f"Deleted documents with filter: {delete_kwargs}")

        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Collection statistics and metadata
        """
        if not self.collection:
            return {"error": "No collection available"}

        try:
            count = self.collection.count()

            return {
                "name": self.collection.name,
                "count": count,
                "metadata": self.collection.metadata,
                "distance_metric": self.config.distance_metric,
                "persist_directory": (self.config.db_path if self.config.persist else None),
            }

        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}

    def reset_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Reset (delete and recreate) collection.

        Args:
            collection_name: Collection name to reset
        """
        collection_name = collection_name or self.config.collection_name

        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")

            self.create_collection(collection_name)
            self.logger.info(f"Recreated collection: {collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            raise

    def list_collections(self) -> List[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []

    def get_document_count(self, collection_name: Optional[str] = None) -> int:
        """
        Get the total number of documents in the collection.

        Args:
            collection_name: Optional collection name, uses default if None

        Returns:
            Number of documents in the collection
        """
        try:
            collection_name = collection_name or self.config.collection_name
            collection = self.client.get_collection(collection_name)
            return collection.count()
        except Exception as e:
            self.logger.error(f"Failed to get document count: {e}")
            return 0

    async def close(self) -> None:
        """
        Close the ChromaDB storage connection gracefully.

        Note: ChromaDB persistent client doesn't require explicit closing,
        but this method provides a standard interface for cleanup.
        """
        try:
            self.logger.info("Closing ChromaDB storage connection...")
            # ChromaDB persistent client automatically handles cleanup
            # No explicit close needed for the client
            self.logger.info("ChromaDB storage connection closed successfully")
        except Exception as e:
            self.logger.error(f"Error during ChromaDB storage cleanup: {e}")


def create_storage_client(
    db_path: str = None,
    collection_name: str = None,
    persist: bool = None,
) -> ChromaDBStorage:
    """
    Factory function to create ChromaDB storage client.

    Args:
        db_path: Database storage path (defaults from config)
        collection_name: Default collection name (defaults from config)
        persist: Whether to persist data to disk (defaults from config)

    Returns:
        Configured ChromaDBStorage instance
    """
    # Use config defaults if not provided
    if any(param is None for param in [db_path, collection_name, persist]):
        try:
            storage_config = get_storage_config()
            db_path = db_path or storage_config["db_path"]
            collection_name = collection_name or storage_config["collection_name"]
            persist = persist if persist is not None else storage_config["persist"]
        except Exception as e:
            # Fallback to hardcoded defaults
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to load storage config for factory function, using hardcoded defaults: {e}"
            )
            logger.warning("Check your config/vectordb.toml file")
            db_path = db_path or "./data/chromadb"
            collection_name = collection_name or "croatian_documents"
            persist = persist if persist is not None else True

    config = StorageConfig(
        db_path=db_path,
        collection_name=collection_name,
        distance_metric="cosine",
        persist=persist,
        allow_reset=True,
    )
    return ChromaDBStorage(config)
