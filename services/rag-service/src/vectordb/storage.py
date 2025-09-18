"""
ChromaDB storage system for document vectors and metadata.
Implements pure functions and dependency injection for robust storage operations.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_data_transformation,
    log_decision_point,
    log_error_context,
    log_performance_metric,
)


@dataclass
class DocumentMetadata:
    """Document metadata - pure data structure."""

    source_file: str
    chunk_index: int
    language: str
    timestamp: datetime | None = None

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
    error_message: str | None = None


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
        embeddings: list[np.ndarray] | None = None,
    ) -> None: ...

    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[np.ndarray] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def update(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[np.ndarray] | None = None,
    ) -> None: ...

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None: ...

    def count(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


class VectorDatabase(Protocol):
    """Vector database interface."""

    def create_collection(self, name: str, reset_if_exists: bool = False) -> VectorCollection: ...

    def get_collection(self, name: str) -> VectorCollection: ...

    def delete_collection(self, name: str) -> None: ...

    def list_collections(self) -> list[str]: ...

    def reset(self) -> None: ...


# Pure functions for business logic
def validate_documents_for_storage(documents: list[str]) -> list[str]:
    """Validate documents for storage - pure function."""
    logger = get_system_logger()
    log_component_start("document_validator", "validate_for_storage", input_count=len(documents))

    if not documents:
        logger.error("document_validator", "validate_for_storage", "Documents list cannot be empty")
        raise ValueError("Documents list cannot be empty")

    validated_docs = []
    for i, doc in enumerate(documents):
        logger.trace("document_validator", "validate_for_storage", f"Validating document {i}")

        if doc is None:
            logger.error("document_validator", "validate_for_storage", f"Document at index {i} is None")
            raise ValueError(f"Document at index {i} is None")

        if not isinstance(doc, str):
            logger.error(
                "document_validator", "validate_for_storage", f"Document at index {i} must be string, got {type(doc)}"
            )
            raise ValueError(f"Document at index {i} must be string, got {type(doc)}")

        if not doc.strip():
            logger.error("document_validator", "validate_for_storage", f"Document at index {i} is empty")
            raise ValueError(f"Document at index {i} is empty")

        validated_docs.append(doc)
        logger.trace("document_validator", "validate_for_storage", f"Document {i} validated: {len(doc)} chars")

    log_data_transformation(
        "document_validator", "validate_documents", f"raw[{len(documents)}]", f"validated[{len(validated_docs)}]"
    )
    log_component_end("document_validator", "validate_for_storage", f"Validated {len(validated_docs)} documents")
    return validated_docs


def validate_embeddings_for_storage(embeddings: list[np.ndarray], expected_dim: int | None = None) -> list[np.ndarray]:
    """Validate embeddings for storage - pure function."""
    logger = get_system_logger()
    log_component_start(
        "embedding_validator", "validate_for_storage", input_count=len(embeddings), expected_dim=expected_dim
    )

    if not embeddings:
        logger.error("embedding_validator", "validate_for_storage", "Embeddings list cannot be empty")
        raise ValueError("Embeddings list cannot be empty")

    validated_embeddings = []
    for i, emb in enumerate(embeddings):
        logger.trace("embedding_validator", "validate_for_storage", f"Validating embedding {i}")

        if emb is None:
            logger.error("embedding_validator", "validate_for_storage", f"Embedding at index {i} is None")
            raise ValueError(f"Embedding at index {i} is None")

        if not isinstance(emb, np.ndarray):
            logger.error(
                "embedding_validator",
                "validate_for_storage",
                f"Embedding at index {i} must be numpy array, got {type(emb)}",
            )
            raise ValueError(f"Embedding at index {i} must be numpy array, got {type(emb)}")

        if expected_dim is not None and emb.shape[0] != expected_dim:
            logger.error(
                "embedding_validator",
                "validate_for_storage",
                f"Embedding at index {i} has shape {emb.shape}, expected ({expected_dim},)",
            )
            raise ValueError(f"Embedding at index {i} has shape {emb.shape}, expected ({expected_dim},)")

        validated_embeddings.append(emb)
        logger.trace("embedding_validator", "validate_for_storage", f"Embedding {i} validated: shape {emb.shape}")

    log_data_transformation(
        "embedding_validator",
        "validate_embeddings",
        f"raw[{len(embeddings)}]",
        f"validated[{len(validated_embeddings)}]",
    )
    log_component_end(
        "embedding_validator", "validate_for_storage", f"Validated {len(validated_embeddings)} embeddings"
    )
    return validated_embeddings


def prepare_storage_batch(
    documents: list[str], embeddings: list[np.ndarray], metadata_list: list[DocumentMetadata], batch_size: int = 100
) -> list[dict[str, Any]]:
    """Prepare documents for batch storage - pure function."""
    logger = get_system_logger()
    log_component_start(
        "batch_preparer",
        "prepare_storage_batch",
        documents_count=len(documents),
        embeddings_count=len(embeddings),
        metadata_count=len(metadata_list),
        batch_size=batch_size,
    )

    if len(documents) != len(embeddings) != len(metadata_list):
        logger.error(
            "batch_preparer",
            "prepare_storage_batch",
            f"Length mismatch: docs={len(documents)}, emb={len(embeddings)}, meta={len(metadata_list)}",
        )
        raise ValueError("Documents, embeddings, and metadata lists must have same length")

    batches = []
    total_items = len(documents)
    expected_batches = (total_items + batch_size - 1) // batch_size
    logger.debug(
        "batch_preparer", "prepare_storage_batch", f"Preparing {total_items} items in ~{expected_batches} batches"
    )

    for i in range(0, total_items, batch_size):
        end_idx = min(i + batch_size, total_items)
        batch_num = len(batches) + 1

        logger.trace(
            "batch_preparer", "prepare_storage_batch", f"Processing batch {batch_num}: items {i}-{end_idx - 1}"
        )

        batch_documents = documents[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadata = metadata_list[i:end_idx]

        # Generate IDs from metadata
        batch_ids = []
        batch_metadatas = []

        for j, metadata in enumerate(batch_metadata):
            doc_id = f"{metadata.source_file}_chunk_{metadata.chunk_index}_{uuid.uuid4().hex[:8]}"
            batch_ids.append(doc_id)
            batch_metadatas.append(metadata.to_dict())
            logger.trace("batch_preparer", "prepare_storage_batch", f"Generated ID {j}: {doc_id}")

        batch = {
            "ids": batch_ids,
            "documents": batch_documents,
            "embeddings": batch_embeddings,
            "metadatas": batch_metadatas,
        }

        batches.append(batch)
        logger.debug(
            "batch_preparer", "prepare_storage_batch", f"Batch {batch_num} prepared: {len(batch_documents)} items"
        )

    log_data_transformation("batch_preparer", "create_batches", f"items[{total_items}]", f"batches[{len(batches)}]")
    log_component_end(
        "batch_preparer", "prepare_storage_batch", f"Prepared {len(batches)} batches from {total_items} items"
    )
    return batches


def parse_query_results(raw_results: dict[str, Any]) -> list[QueryResult]:
    """Parse raw ChromaDB query results - pure function."""
    logger = get_system_logger()
    log_component_start(
        "result_parser",
        "parse_query_results",
        has_results=bool(raw_results),
        result_keys=list(raw_results.keys()) if raw_results else [],
    )

    if not raw_results or "ids" not in raw_results:
        logger.debug("result_parser", "parse_query_results", "No results or missing 'ids' field")
        log_component_end("result_parser", "parse_query_results", "No results to parse")
        return []

    ids_list = raw_results["ids"][0] if raw_results["ids"] else []
    logger.debug("result_parser", "parse_query_results", f"Processing {len(ids_list)} result IDs")

    # Validate required fields
    required_fields = ["documents", "metadatas", "distances"]
    for field_name in required_fields:
        if field_name not in raw_results:
            logger.error("result_parser", "parse_query_results", f"Missing '{field_name}' in query results")
            raise ValueError(f"Missing '{field_name}' in query results")

    documents_list = raw_results["documents"][0]
    metadatas_list = raw_results["metadatas"][0]
    distances_list = raw_results["distances"][0]

    logger.debug(
        "result_parser",
        "parse_query_results",
        f"Result arrays: ids={len(ids_list)}, docs={len(documents_list)}, "
        f"meta={len(metadatas_list)}, dist={len(distances_list)}",
    )

    results = []
    for i, doc_id in enumerate(ids_list):
        content = documents_list[i] if i < len(documents_list) else ""
        metadata = metadatas_list[i] if i < len(metadatas_list) else {}
        score = distances_list[i] if i < len(distances_list) else 1.0

        result = QueryResult(id=doc_id, content=content, metadata=metadata, score=score)
        results.append(result)
        logger.trace(
            "result_parser",
            "parse_query_results",
            f"Result {i}: id={doc_id[:20]}..., score={score:.4f}, content={len(content)} chars",
        )

    log_data_transformation(
        "result_parser", "convert_results", f"raw_results[{len(ids_list)}]", f"query_results[{len(results)}]"
    )
    log_component_end("result_parser", "parse_query_results", f"Parsed {len(results)} query results")
    return results


def calculate_batch_sizes(num_documents: int, max_batch_size: int = 100) -> int:
    """Calculate optimal batch size - pure function."""
    get_system_logger()
    log_component_start(
        "batch_calculator", "calculate_batch_size", num_documents=num_documents, max_batch_size=max_batch_size
    )

    if num_documents <= 100:
        batch_size = min(25, num_documents)
        log_decision_point(
            "batch_calculator", "calculate_batch_size", f"small_dataset={num_documents}", f"batch_size={batch_size}"
        )
    elif num_documents <= 1000:
        batch_size = 50
        log_decision_point(
            "batch_calculator", "calculate_batch_size", f"medium_dataset={num_documents}", f"batch_size={batch_size}"
        )
    else:
        batch_size = max_batch_size
        log_decision_point(
            "batch_calculator", "calculate_batch_size", f"large_dataset={num_documents}", f"batch_size={batch_size}"
        )

    log_component_end(
        "batch_calculator", "calculate_batch_size", f"Calculated batch size: {batch_size} for {num_documents} documents"
    )
    return batch_size


def extract_document_ids(documents: list[dict[str, Any]]) -> list[str]:
    """Extract document IDs - pure function."""
    ids = []
    for i, doc in enumerate(documents):
        if "id" not in doc:
            raise KeyError(f"Document at index {i} missing 'id' field")
        ids.append(doc["id"])
    return ids


def merge_search_results(results_list: list[list[QueryResult]], max_results: int = 10) -> list[QueryResult]:
    """Merge and sort search results from multiple sources - pure function."""
    logger = get_system_logger()
    log_component_start(
        "result_merger", "merge_search_results", source_count=len(results_list), max_results=max_results
    )

    all_results = []
    for i, results in enumerate(results_list):
        all_results.extend(results)
        logger.trace("result_merger", "merge_search_results", f"Source {i}: added {len(results)} results")

    logger.debug("result_merger", "merge_search_results", f"Total results before sorting: {len(all_results)}")

    # Sort by score (ascending - lower distance is better)
    all_results.sort(key=lambda r: r.score)
    logger.debug(
        "result_merger",
        "merge_search_results",
        f"Sorted results by score, best: {all_results[0].score:.4f}" if all_results else "No results",
    )

    final_results = all_results[:max_results]
    log_data_transformation(
        "result_merger", "merge_and_limit", f"all[{len(all_results)}]", f"final[{len(final_results)}]"
    )
    log_component_end(
        "result_merger", "merge_search_results", f"Merged {len(results_list)} sources into {len(final_results)} results"
    )
    return final_results


# Main vector storage class
class VectorStorage:
    """Vector storage system with dependency injection."""

    def __init__(self, database: VectorDatabase):
        get_system_logger()
        log_component_start("vector_storage", "init", database_type=type(database).__name__)

        self.database = database
        self.collection: VectorCollection | None = None
        self.logger = get_system_logger()

        log_component_end("vector_storage", "init", "Vector storage initialized")

    async def initialize(self, collection_name: str, reset_if_exists: bool = False) -> None:
        """Initialize storage with collection."""
        logger = get_system_logger()
        log_component_start(
            "vector_storage", "initialize", collection_name=collection_name, reset_if_exists=reset_if_exists
        )

        try:
            logger.debug("vector_storage", "initialize", f"Creating collection: {collection_name}")
            self.collection = self.database.create_collection(name=collection_name, reset_if_exists=reset_if_exists)

            # Get collection stats for logging
            try:
                count = self.collection.count()
                log_performance_metric("vector_storage", "initialize", "collection_count", count)
            except Exception:
                logger.trace("vector_storage", "initialize", "Could not get collection count")

            log_component_end("vector_storage", "initialize", f"Initialized collection: {collection_name}")

        except Exception as e:
            log_error_context(
                "vector_storage",
                "initialize",
                e,
                {"collection_name": collection_name, "reset_if_exists": reset_if_exists},
            )
            raise

    async def store_documents(
        self,
        documents: list[str],
        embeddings: list[np.ndarray],
        metadata_list: list[DocumentMetadata],
        batch_size: int = 100,
    ) -> StorageResult:
        """Store documents in batches."""
        logger = get_system_logger()
        log_component_start(
            "vector_storage",
            "store_documents",
            documents_count=len(documents),
            embeddings_count=len(embeddings),
            metadata_count=len(metadata_list),
            batch_size=batch_size,
        )

        if not self.collection:
            error_msg = "Storage not initialized - call initialize() first"
            logger.error("vector_storage", "store_documents", error_msg)
            return StorageResult(success=False, error_message=error_msg)

        try:
            # Validate inputs using pure functions
            logger.debug("vector_storage", "store_documents", "Validating input documents and embeddings")
            validated_docs = validate_documents_for_storage(documents)
            validated_embeddings = validate_embeddings_for_storage(embeddings)

            # Prepare batches using pure function
            logger.debug("vector_storage", "store_documents", "Preparing storage batches")
            batches = prepare_storage_batch(validated_docs, validated_embeddings, metadata_list, batch_size)
            log_performance_metric("vector_storage", "store_documents", "batch_count", len(batches))

            # Store each batch
            all_doc_ids = []
            logger.info(
                "vector_storage",
                "store_documents",
                f"Storing {len(validated_docs)} documents in {len(batches)} batches",
            )

            for i, batch in enumerate(batches):
                logger.trace(
                    "vector_storage",
                    "store_documents",
                    f"Storing batch {i + 1}/{len(batches)}: {len(batch['ids'])} documents",
                )
                self.collection.add(**batch)
                all_doc_ids.extend(batch["ids"])
                logger.debug("vector_storage", "store_documents", f"Batch {i + 1} stored successfully")

            result = StorageResult(
                success=True,
                documents_stored=len(validated_docs),
                batches_processed=len(batches),
                document_ids=all_doc_ids,
            )

            log_performance_metric("vector_storage", "store_documents", "documents_stored", len(validated_docs))
            log_component_end(
                "vector_storage", "store_documents", f"Successfully stored {len(validated_docs)} documents"
            )
            return result

        except Exception as e:
            log_error_context(
                "vector_storage",
                "store_documents",
                e,
                {
                    "documents_count": len(documents),
                    "embeddings_count": len(embeddings),
                    "metadata_count": len(metadata_list),
                    "batch_size": batch_size,
                },
            )
            return StorageResult(success=False, error_message=str(e))

    async def search_documents(
        self,
        query_text: str | None = None,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """Search documents by text or embedding."""
        logger = get_system_logger()
        log_component_start(
            "vector_storage",
            "search_documents",
            has_text=query_text is not None,
            has_embedding=query_embedding is not None,
            top_k=top_k,
            has_filter=filter_metadata is not None,
        )

        if not self.collection:
            logger.error("vector_storage", "search_documents", "Storage not initialized")
            raise RuntimeError("Storage not initialized - call initialize() first")

        try:
            if query_text:
                logger.debug(
                    "vector_storage", "search_documents", f"Searching by text: '{query_text[:50]}...' (top_k={top_k})"
                )
                raw_results = self.collection.query(
                    query_texts=[query_text],
                    n_results=top_k,
                    where=filter_metadata,
                    include=["documents", "metadatas", "distances"],
                )
                log_decision_point("vector_storage", "search_documents", "search_type=text", "query_executed")
            elif query_embedding is not None:
                logger.debug(
                    "vector_storage",
                    "search_documents",
                    f"Searching by embedding: shape={query_embedding.shape} (top_k={top_k})",
                )
                raw_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter_metadata,
                    include=["documents", "metadatas", "distances"],
                )
                log_decision_point("vector_storage", "search_documents", "search_type=embedding", "query_executed")
            else:
                logger.error("vector_storage", "search_documents", "Neither query_text nor query_embedding provided")
                raise ValueError("Either query_text or query_embedding must be provided")

            results = parse_query_results(raw_results)
            log_performance_metric("vector_storage", "search_documents", "results_count", len(results))

            if results:
                best_score = results[0].score
                worst_score = results[-1].score
                logger.debug(
                    "vector_storage",
                    "search_documents",
                    f"Search scores: best={best_score:.4f}, worst={worst_score:.4f}",
                )

            log_component_end("vector_storage", "search_documents", f"Search returned {len(results)} results")
            return results

        except Exception as e:
            log_error_context(
                "vector_storage",
                "search_documents",
                e,
                {
                    "has_text": query_text is not None,
                    "has_embedding": query_embedding is not None,
                    "top_k": top_k,
                    "has_filter": filter_metadata is not None,
                },
            )
            raise

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            raise RuntimeError("Storage not initialized - call initialize() first")

        count = self.collection.count()
        return {"name": self.collection.name, "document_count": count, "metadata": self.collection.metadata}

    async def delete_documents(
        self, ids: list[str] | None = None, filter_metadata: dict[str, Any] | None = None
    ) -> None:
        """Delete documents from collection."""
        if not self.collection:
            raise ValueError("Storage not initialized")

        if not ids and not filter_metadata:
            raise ValueError("Either ids or filter_metadata must be provided")

        self.collection.delete(ids=ids, where=filter_metadata)
        self.logger.info("Documents deleted successfully")


# Factory functions
def create_vector_storage(database: VectorDatabase) -> VectorStorage:
    """Factory function to create vector storage."""
    get_system_logger()
    log_component_start("storage_factory", "create_vector_storage", database_type=type(database).__name__)

    storage = VectorStorage(database)

    log_component_end(
        "storage_factory", "create_vector_storage", f"Created vector storage with {type(database).__name__}"
    )
    return storage


def create_mock_storage() -> VectorStorage:
    """Factory function to create mock storage for testing."""
    from .storage_factories import create_mock_database

    mock_db = create_mock_database()
    return VectorStorage(mock_db)
