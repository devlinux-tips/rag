"""
ChromaDB storage system for document vectors and metadata.
Implements pure functions and dependency injection for robust storage operations.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np


@dataclass
class DocumentMetadata:
    """Document metadata - pure data structure."""

    source_file: str
    chunk_index: int
    language: str
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "language": self.language,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class StorageResult:
    """Result from storage operation - pure data structure."""

    success: bool
    documents_stored: int = 0
    batches_processed: int = 0
    document_ids: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class QueryResult:
    """Query result from vector database - pure data structure."""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float


# Protocols for dependency injection
class VectorCollection(Protocol):
    """Vector collection interface."""

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: Optional[list[np.ndarray]] = None,
    ) -> None:
        ...

    def query(
        self,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[np.ndarray]] = None,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        include: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        ...

    def get(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        ...

    def update(
        self,
        ids: list[str],
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        embeddings: Optional[list[np.ndarray]] = None,
    ) -> None:
        ...

    def delete(
        self, ids: Optional[list[str]] = None, where: Optional[dict[str, Any]] = None
    ) -> None:
        ...

    def count(self) -> int:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        ...


class VectorDatabase(Protocol):
    """Vector database interface."""

    def create_collection(
        self, name: str, reset_if_exists: bool = False
    ) -> VectorCollection:
        ...

    def get_collection(self, name: str) -> VectorCollection:
        ...

    def delete_collection(self, name: str) -> None:
        ...

    def list_collections(self) -> list[str]:
        ...

    def reset(self) -> None:
        ...


# Pure functions for business logic
def validate_documents_for_storage(documents: list[str]) -> list[str]:
    """Validate documents for storage - pure function."""
    if not documents:
        raise ValueError("Documents list cannot be empty")

    validated_docs = []
    for i, doc in enumerate(documents):
        if doc is None:
            raise ValueError(f"Document at index {i} is None")
        if not isinstance(doc, str):
            raise ValueError(f"Document at index {i} must be string, got {type(doc)}")
        if not doc.strip():
            raise ValueError(f"Document at index {i} is empty")
        validated_docs.append(doc)

    return validated_docs


def validate_embeddings_for_storage(
    embeddings: list[np.ndarray], expected_dim: Optional[int] = None
) -> list[np.ndarray]:
    """Validate embeddings for storage - pure function."""
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty")

    validated_embeddings = []
    for i, emb in enumerate(embeddings):
        if emb is None:
            raise ValueError(f"Embedding at index {i} is None")
        if not isinstance(emb, np.ndarray):
            raise ValueError(
                f"Embedding at index {i} must be numpy array, got {type(emb)}"
            )

        if expected_dim is not None and emb.shape[0] != expected_dim:
            raise ValueError(
                f"Embedding at index {i} has shape {emb.shape}, expected ({expected_dim},)"
            )

        validated_embeddings.append(emb)

    return validated_embeddings


def prepare_storage_batch(
    documents: list[str],
    embeddings: list[np.ndarray],
    metadata_list: list[DocumentMetadata],
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    """Prepare documents for batch storage - pure function."""
    if len(documents) != len(embeddings) != len(metadata_list):
        raise ValueError(
            "Documents, embeddings, and metadata lists must have same length"
        )

    batches = []
    total_items = len(documents)

    for i in range(0, total_items, batch_size):
        end_idx = min(i + batch_size, total_items)

        batch_documents = documents[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadata = metadata_list[i:end_idx]

        # Generate IDs from metadata
        batch_ids = []
        batch_metadatas = []

        for metadata in batch_metadata:
            doc_id = f"{metadata.source_file}_chunk_{metadata.chunk_index}_{uuid.uuid4().hex[:8]}"
            batch_ids.append(doc_id)
            batch_metadatas.append(metadata.to_dict())

        batch = {
            "ids": batch_ids,
            "documents": batch_documents,
            "embeddings": batch_embeddings,
            "metadatas": batch_metadatas,
        }

        batches.append(batch)

    return batches


def parse_query_results(raw_results: dict[str, Any]) -> list[QueryResult]:
    """Parse raw ChromaDB query results - pure function."""
    if not raw_results or "ids" not in raw_results:
        return []

    ids_list = raw_results["ids"][0] if raw_results["ids"] else []
    if "documents" not in raw_results:
        raise ValueError("Missing 'documents' in query results")
    if "metadatas" not in raw_results:
        raise ValueError("Missing 'metadatas' in query results")
    if "distances" not in raw_results:
        raise ValueError("Missing 'distances' in query results")

    documents_list = raw_results["documents"][0]
    metadatas_list = raw_results["metadatas"][0]
    distances_list = raw_results["distances"][0]

    results = []
    for i, doc_id in enumerate(ids_list):
        content = documents_list[i] if i < len(documents_list) else ""
        metadata = metadatas_list[i] if i < len(metadatas_list) else {}
        score = distances_list[i] if i < len(distances_list) else 1.0

        result = QueryResult(id=doc_id, content=content, metadata=metadata, score=score)
        results.append(result)

    return results


def calculate_batch_sizes(num_documents: int, max_batch_size: int = 100) -> int:
    """Calculate optimal batch size - pure function."""
    if num_documents <= 100:
        return min(25, num_documents)
    elif num_documents <= 1000:
        return 50
    else:
        return max_batch_size


def extract_document_ids(documents: list[dict[str, Any]]) -> list[str]:
    """Extract document IDs - pure function."""
    ids = []
    for i, doc in enumerate(documents):
        if "id" not in doc:
            raise KeyError(f"Document at index {i} missing 'id' field")
        ids.append(doc["id"])
    return ids


def merge_search_results(
    results_list: list[list[QueryResult]], max_results: int = 10
) -> list[QueryResult]:
    """Merge and sort search results from multiple sources - pure function."""
    all_results = []
    for results in results_list:
        all_results.extend(results)

    # Sort by score (ascending - lower distance is better)
    all_results.sort(key=lambda r: r.score)

    return all_results[:max_results]


# Main vector storage class
class VectorStorage:
    """Vector storage system with dependency injection."""

    def __init__(self, database: VectorDatabase):
        self.database = database
        self.collection: Optional[VectorCollection] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(
        self, collection_name: str, reset_if_exists: bool = False
    ) -> None:
        """Initialize storage with collection."""
        self.collection = self.database.create_collection(
            name=collection_name, reset_if_exists=reset_if_exists
        )
        self.logger.info(f"Initialized storage with collection: {collection_name}")

    async def store_documents(
        self,
        documents: list[str],
        embeddings: list[np.ndarray],
        metadata_list: list[DocumentMetadata],
        batch_size: int = 100,
    ) -> StorageResult:
        """Store documents in batches."""
        if not self.collection:
            return StorageResult(
                success=False,
                error_message="Storage not initialized - call initialize() first",
            )

        # Validate inputs using pure functions
        validated_docs = validate_documents_for_storage(documents)
        validated_embeddings = validate_embeddings_for_storage(embeddings)

        # Prepare batches using pure function
        batches = prepare_storage_batch(
            validated_docs, validated_embeddings, metadata_list, batch_size
        )

        # Store each batch
        all_doc_ids = []
        for batch in batches:
            self.collection.add(**batch)
            all_doc_ids.extend(batch["ids"])

        self.logger.info(
            f"Stored {len(validated_docs)} documents in {len(batches)} batches"
        )

        return StorageResult(
            success=True,
            documents_stored=len(validated_docs),
            batches_processed=len(batches),
            document_ids=all_doc_ids,
        )

    async def search_documents(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[QueryResult]:
        """Search documents by text or embedding."""
        if not self.collection:
            raise RuntimeError("Storage not initialized - call initialize() first")

        query_kwargs = {
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if query_text:
            query_kwargs["query_texts"] = [query_text]
        elif query_embedding is not None:
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            raise ValueError("Either query_text or query_embedding must be provided")

        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        raw_results = self.collection.query(**query_kwargs)
        results = parse_query_results(raw_results)

        self.logger.debug(f"Search returned {len(results)} results")
        return results

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            raise RuntimeError("Storage not initialized - call initialize() first")

        count = self.collection.count()
        return {
            "name": self.collection.name,
            "document_count": count,
            "metadata": self.collection.metadata,
        }

    async def delete_documents(
        self,
        ids: Optional[list[str]] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Delete documents from collection."""
        if not self.collection:
            raise ValueError("Storage not initialized")

        delete_kwargs = {}
        if ids:
            delete_kwargs["ids"] = ids
        if filter_metadata:
            delete_kwargs["where"] = filter_metadata

        if not delete_kwargs:
            raise ValueError("Either ids or filter_metadata must be provided")

        self.collection.delete(**delete_kwargs)
        self.logger.info("Documents deleted successfully")


# Factory functions
def create_vector_storage(database: VectorDatabase) -> VectorStorage:
    """Factory function to create vector storage."""
    return VectorStorage(database)


def create_mock_storage() -> VectorStorage:
    """Factory function to create mock storage for testing."""
    from .storage_factories import create_mock_database

    mock_db = create_mock_database()
    return VectorStorage(mock_db)
