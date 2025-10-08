"""
Weaviate factory implementations for vector database components.
Provides production Weaviate implementations of storage protocols with full configuration support.
"""

import logging
import time
from typing import Any

import numpy as np
import weaviate  # type: ignore[import-not-found]
from weaviate.classes.config import Configure, VectorDistances  # type: ignore[import-not-found]
from weaviate.classes.query import Filter  # type: ignore[import-not-found]

from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start, log_error_context
from .storage import VectorCollection, VectorDatabase, VectorSearchResult, VectorSearchResults
from .weaviate_config import WeaviateConfiguration, create_weaviate_configuration


class WeaviateCollection(VectorCollection):
    """Weaviate implementation of VectorCollection protocol."""

    def __init__(self, client: weaviate.WeaviateClient, class_name: str, config: WeaviateConfiguration):
        """Initialize Weaviate collection."""
        get_system_logger()
        log_component_start("weaviate_collection", "init", class_name=class_name)

        self.client = client
        self.class_name = class_name
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize collection if it doesn't exist
        self._ensure_class_exists()

        log_component_end("weaviate_collection", "init", f"Weaviate collection initialized: {class_name}")

    @property
    def name(self) -> str:
        """Get collection name."""
        return self.class_name

    @property
    def metadata(self) -> dict[str, Any]:
        """Get collection metadata."""
        return {
            "class_name": self.class_name,
            "config": {
                "hnsw": {
                    "ef_construction": self.config.hnsw.ef_construction,
                    "ef": self.config.hnsw.ef,
                    "max_connections": self.config.hnsw.max_connections,
                },
                "compression": {"enabled": self.config.compression.enabled, "type": self.config.compression.type},
            },
        }

    def _ensure_class_exists(self) -> None:
        """Ensure the Weaviate class/collection exists."""
        logger = get_system_logger()

        try:
            # Check if class exists using collections API
            collections = self.client.collections
            try:
                # Try to get the collection - will raise exception if not exists
                collections.get(self.class_name)
                logger.debug("weaviate_collection", "ensure_class", f"Weaviate class already exists: {self.class_name}")
                return
            except Exception:
                # Collection doesn't exist, create it
                logger.info("weaviate_collection", "ensure_class", f"Creating Weaviate class: {self.class_name}")

                # Get HNSW configuration
                hnsw_config = self.config.hnsw
                compression_config = self.config.compression.to_weaviate_config()

                # Create collection with new v4 API
                collection_config = Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef_construction=hnsw_config.ef_construction,
                    ef=hnsw_config.ef,
                    max_connections=hnsw_config.max_connections,
                    dynamic_ef_factor=hnsw_config.ef_dynamic,
                    cleanup_interval_seconds=hnsw_config.cleanup_interval_seconds,
                    vector_cache_max_objects=hnsw_config.vector_cache_max_objects,
                )

                # Add compression if enabled
                if compression_config and self.config.compression.enabled:
                    if self.config.compression.type.lower() == "sq":
                        vector_index_config = Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE,
                            ef_construction=hnsw_config.ef_construction,
                            ef=hnsw_config.ef,
                            max_connections=hnsw_config.max_connections,
                            dynamic_ef_factor=hnsw_config.ef_dynamic,
                            cleanup_interval_seconds=hnsw_config.cleanup_interval_seconds,
                            vector_cache_max_objects=hnsw_config.vector_cache_max_objects,
                            quantizer=Configure.VectorIndex.Quantizer.sq(
                                rescore_limit=self.config.compression.rescore_limit, cache=self.config.compression.cache
                            ),
                        )
                    else:
                        vector_index_config = collection_config
                else:
                    vector_index_config = collection_config

                # Create the collection
                collections.create(
                    name=self.class_name,
                    description=f"Document collection for {self.class_name}",
                    vector_index_config=vector_index_config,
                    vectorizer_config=None,  # We provide our own embeddings
                )

                logger.info(
                    "weaviate_collection", "ensure_class", f"Successfully created Weaviate class: {self.class_name}"
                )

        except Exception as e:
            logger.error("weaviate_collection", "ensure_class", f"Failed to ensure class exists: {str(e)}")
            raise

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """Add documents to Weaviate collection."""
        logger = get_system_logger()
        log_component_start(
            "weaviate_collection", "add", doc_count=len(documents), has_embeddings=embeddings is not None
        )

        if len(ids) != len(documents) or len(ids) != len(metadatas):
            raise ValueError("Length mismatch between ids, documents, and metadatas")

        try:
            # Get the collection
            collection = self.client.collections.get(self.class_name)

            # Prepare batch data
            objects_to_insert = []
            logger.debug("weaviate_collection", "add", f"Starting to process {len(documents)} documents")

            for i, (doc_id, document, metadata) in enumerate(zip(ids, documents, metadatas, strict=False)):
                logger.trace("weaviate_collection", "add", f"Processing document {i}: {doc_id}")

                # Prepare properties - start with standard fields
                properties = {
                    "content": document,
                    "source_file": metadata.get("source_file", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "language": metadata.get("language", ""),
                    "timestamp": metadata.get("timestamp", ""),
                }

                # Add all additional metadata fields (feature-specific metadata like nn_metadata)
                # Weaviate auto-creates properties, so we can safely add any dict/list/str/number fields
                for key, value in metadata.items():
                    if key not in properties:  # Don't overwrite standard fields
                        # Weaviate supports: text, number, boolean, object (dict), and array
                        # Complex objects need to be serialized to JSON strings
                        if isinstance(value, dict):
                            import json

                            properties[key] = json.dumps(value, ensure_ascii=False)
                        elif value is not None:  # Skip None values
                            properties[key] = value

                # Add vector if provided - handle both numpy arrays and lists
                if embeddings and i < len(embeddings):
                    if hasattr(embeddings[i], "tolist"):
                        vector = embeddings[i].tolist()
                    elif isinstance(embeddings[i], (list, tuple)):
                        vector = list(embeddings[i])
                    else:
                        vector = embeddings[i]
                    logger.trace(
                        "weaviate_collection",
                        "add",
                        f"Vector prepared for doc {i}: shape/length = {len(vector) if vector else 'None'}",
                    )
                else:
                    vector = None
                    logger.trace("weaviate_collection", "add", f"No vector for doc {i}")

                import uuid as uuid_lib

                from weaviate.classes.data import DataObject  # type: ignore[import-not-found]

                try:
                    # Generate a proper UUID if doc_id is not valid UUID format
                    try:
                        # Try to parse the doc_id as UUID to see if it's valid
                        uuid_lib.UUID(doc_id)
                        weaviate_uuid = doc_id
                    except ValueError:
                        # If doc_id is not a valid UUID, generate one from it
                        weaviate_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, doc_id))

                    obj = DataObject(properties=properties, uuid=weaviate_uuid, vector=vector)
                    objects_to_insert.append(obj)
                    logger.trace(
                        "weaviate_collection",
                        "add",
                        f"Successfully created and added DataObject for doc {i}: UUID={weaviate_uuid}",
                    )
                except Exception as obj_error:
                    logger.error("weaviate_collection", "add", f"Failed to create DataObject for doc {i}: {obj_error}")
                    raise

                if i % 100 == 0:
                    logger.trace("weaviate_collection", "add", f"Prepared {i + 1}/{len(documents)} documents for batch")

            logger.info("weaviate_collection", "add", f"Created {len(objects_to_insert)} objects for insertion")

            # Insert in batches - but check if we have objects first
            if not objects_to_insert:
                logger.error("weaviate_collection", "add", "No objects to insert - batch is empty!")
                raise ValueError("No objects were created for insertion")

            batch_size = self.config.general.batch_size
            logger.debug(
                "weaviate_collection", "add", f"Inserting {len(objects_to_insert)} objects in batches of {batch_size}"
            )

            for i in range(0, len(objects_to_insert), batch_size):
                batch = objects_to_insert[i : i + batch_size]
                logger.debug(
                    "weaviate_collection", "add", f"Inserting batch {i // batch_size + 1}: {len(batch)} objects"
                )
                logger.trace("weaviate_collection", "add", f"Batch contents: {[type(obj).__name__ for obj in batch]}")
                logger.trace(
                    "weaviate_collection",
                    "add",
                    f"First object properties: {batch[0].properties if batch else 'No objects'}",
                )
                collection.data.insert_many(batch)

            logger.info("weaviate_collection", "add", f"Successfully added {len(documents)} documents")

        except Exception as e:
            error_msg = f"Failed to add documents to Weaviate: {str(e)}"
            logger.error("weaviate_collection", "add", error_msg)
            log_error_context(
                "weaviate_collection", "add", e, {"doc_count": len(documents), "class_name": self.class_name}
            )
            raise

        log_component_end("weaviate_collection", "add", f"Added {len(documents)} documents successfully")

    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[np.ndarray] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> VectorSearchResults:
        """Query Weaviate collection and return standard VectorSearchResults."""
        import time

        start_time = time.time()

        logger = get_system_logger()
        log_component_start(
            "weaviate_collection",
            "query",
            n_results=n_results,
            has_query_texts=query_texts is not None,
            has_embeddings=query_embeddings is not None,
        )

        try:
            collection = self.client.collections.get(self.class_name)

            # Prepare query
            if query_embeddings is not None and len(query_embeddings) > 0:
                # Vector similarity search
                # Handle both numpy arrays and lists for query embeddings
                if hasattr(query_embeddings[0], "tolist"):
                    vector = query_embeddings[0].tolist()
                elif isinstance(query_embeddings[0], (list, tuple)):
                    vector = list(query_embeddings[0])
                else:
                    vector = query_embeddings[0]

                response = collection.query.near_vector(
                    near_vector=vector, limit=n_results, return_metadata=["certainty", "distance"]
                )

            elif query_texts is not None and len(query_texts) > 0:
                # Text-based search (requires vectorizer)
                query_text = query_texts[0]

                response = collection.query.near_text(
                    query=query_text, limit=n_results, return_metadata=["certainty", "distance"]
                )

            else:
                # Get all documents (limited)
                response = collection.query.fetch_objects(limit=n_results, return_metadata=["certainty", "distance"])

            # Convert Weaviate response directly to VectorSearchResults
            results = []
            for obj in response.objects:
                # Extract content and metadata from Weaviate object
                content = obj.properties.get("content", "") or obj.properties.get("text", "")
                metadata = dict(obj.properties)  # All properties become metadata
                metadata.pop("content", None)  # Remove content from metadata
                metadata.pop("text", None)  # Remove text from metadata

                # Extract distance from Weaviate metadata (lower = more similar)
                # AI DEBUGGING: Comprehensive trace logging for distance extraction
                logger = get_system_logger()

                # Log raw metadata structure for AI debugging
                metadata_structure = {
                    "has_distance": hasattr(obj.metadata, "distance"),
                    "distance_value": getattr(obj.metadata, "distance", "NOT_FOUND"),
                    "distance_type": type(getattr(obj.metadata, "distance", None)).__name__,
                    "has_certainty": hasattr(obj.metadata, "certainty"),
                    "certainty_value": getattr(obj.metadata, "certainty", "NOT_FOUND"),
                    "certainty_type": type(getattr(obj.metadata, "certainty", None)).__name__,
                    "metadata_attrs": [attr for attr in dir(obj.metadata) if not attr.startswith("_")],
                    "metadata_dict": vars(obj.metadata) if hasattr(obj.metadata, "__dict__") else "NO_DICT",
                }
                logger.trace(
                    "weaviate_distance_extraction",
                    "metadata_analysis",
                    f"METADATA_STRUCTURE | obj_id={str(obj.uuid)[:8]} | "
                    f"has_distance={metadata_structure['has_distance']} | "
                    f"distance_val={metadata_structure['distance_value']} | "
                    f"distance_type={metadata_structure['distance_type']} | "
                    f"has_certainty={metadata_structure['has_certainty']} | "
                    f"certainty_val={metadata_structure['certainty_value']} | "
                    f"certainty_type={metadata_structure['certainty_type']} | "
                    f"available_attrs={metadata_structure['metadata_attrs']}",
                )

                # Distance extraction with comprehensive logging
                distance = None
                extraction_method = None

                if hasattr(obj.metadata, "distance") and obj.metadata.distance is not None:
                    raw_distance = obj.metadata.distance
                    distance = float(raw_distance)
                    extraction_method = "direct_distance"
                    logger.trace(
                        "weaviate_distance_extraction",
                        "distance_conversion",
                        f"DISTANCE_DIRECT | obj_id={str(obj.uuid)[:8]} | "
                        f"raw_distance={raw_distance} | raw_type={type(raw_distance).__name__} | "
                        f"converted_distance={distance} | extraction_method={extraction_method}",
                    )
                elif hasattr(obj.metadata, "certainty") and obj.metadata.certainty is not None:
                    raw_certainty = obj.metadata.certainty
                    distance = 1.0 - float(raw_certainty)
                    extraction_method = "certainty_conversion"
                    logger.trace(
                        "weaviate_distance_extraction",
                        "certainty_conversion",
                        f"CERTAINTY_CONVERT | obj_id={str(obj.uuid)[:8]} | "
                        f"raw_certainty={raw_certainty} | raw_type={type(raw_certainty).__name__} | "
                        f"converted_distance={distance} | formula=1.0-certainty | extraction_method={extraction_method}",
                    )
                else:
                    distance = 1.0
                    extraction_method = "default_fallback"
                    logger.warning(
                        "weaviate_distance_extraction",
                        "fallback_distance",
                        f"FALLBACK_DISTANCE | obj_id={str(obj.uuid)[:8]} | "
                        f"no_distance_or_certainty=true | default_distance={distance} | extraction_method={extraction_method}",
                    )

                # Log final distance result for AI debugging
                logger.trace(
                    "weaviate_distance_extraction",
                    "final_result",
                    f"DISTANCE_FINAL | obj_id={str(obj.uuid)[:8]} | "
                    f"final_distance={distance} | extraction_method={extraction_method} | "
                    f"distance_range_ok={0.0 <= distance <= 2.0} | "
                    f"similarity_preview={1.0 - (distance / 2.0)}",
                )

                result = VectorSearchResult(
                    id=str(obj.uuid), content=str(content), metadata=metadata, distance=distance
                )
                results.append(result)

            search_time_ms = (time.time() - start_time) * 1000

            vector_results = VectorSearchResults(
                results=results, total_count=len(results), search_time_ms=search_time_ms
            )

            log_component_end("weaviate_collection", "query", f"Query returned {len(results)} results")

            return vector_results

        except Exception as e:
            error_msg = f"Failed to query Weaviate: {str(e)}"
            logger.error("weaviate_collection", "query", error_msg)
            log_error_context(
                "weaviate_collection", "query", e, {"n_results": n_results, "class_name": self.class_name}
            )
            raise

    async def search(self, query_text: str, k: int = 5, similarity_threshold: float = 0.3) -> list[dict[str, Any]]:
        """
        Search wrapper method for SearchEngineAdapter compatibility.

        This method provides a simplified interface that matches what SearchEngineAdapter expects.
        """
        logger = get_system_logger()
        log_component_start(
            "weaviate_collection", "search", query_text=query_text[:50], k=k, similarity_threshold=similarity_threshold
        )

        try:
            # Use the existing query method with text-based search
            vector_results = self.query(query_texts=[query_text], n_results=k)

            # Convert VectorSearchResults to list of SearchResult-compatible dicts
            results = []
            for result in vector_results.results:
                # Convert distance to similarity score (distance: lower = more similar)
                similarity = result.score  # Use the score property which converts distance to similarity

                # Log similarity values for debugging
                logger.debug(
                    "weaviate_collection",
                    "search",
                    f"Document similarity: {similarity:.4f} | threshold: {similarity_threshold} | "
                    f"distance: {result.distance:.4f} | content_preview: {result.content[:50]}...",
                )

                # Skip results below threshold
                if similarity < similarity_threshold:
                    logger.debug(
                        "weaviate_collection",
                        "search",
                        f"FILTERED OUT: similarity {similarity:.4f} < threshold {similarity_threshold}",
                    )
                    continue

                results.append({"content": result.content, "metadata": result.metadata, "similarity_score": similarity})

            logger.info(
                "weaviate_collection",
                "search",
                f"Search returned {len(results)} results above threshold {similarity_threshold}",
            )

            log_component_end("weaviate_collection", "search", f"Returned {len(results)} results")

            return results

        except Exception as e:
            error_msg = f"Failed to search Weaviate: {str(e)}"
            logger.error("weaviate_collection", "search", error_msg)
            log_error_context(
                "weaviate_collection",
                "search",
                e,
                {"query_text": query_text[:50], "k": k, "class_name": self.class_name},
            )
            raise

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get documents from Weaviate collection."""
        logger = get_system_logger()
        log_component_start("weaviate_collection", "get", has_ids=ids is not None, limit=limit, offset=offset)

        try:
            collection = self.client.collections.get(self.class_name)

            if ids:
                # Get specific documents by ID
                response = collection.query.fetch_objects_by_ids(ids)
            else:
                # Get documents with optional filtering
                response = collection.query.fetch_objects(limit=limit or 100, offset=offset or 0)

            # Convert Weaviate response directly to VectorSearchResults
            results = []
            for obj in response.objects:
                # Extract content and metadata from Weaviate object
                content = obj.properties.get("content", "") or obj.properties.get("text", "")
                metadata = dict(obj.properties)  # All properties become metadata
                metadata.pop("content", None)  # Remove content from metadata
                metadata.pop("text", None)  # Remove text from metadata

                result = VectorSearchResult(
                    id=str(obj.uuid),
                    content=str(content),
                    metadata=metadata,
                    distance=0.0,  # No distance for get operations
                )
                results.append(result)

            vector_results = VectorSearchResults(results=results, total_count=len(results), search_time_ms=0.0)

            # Convert to ChromaDB format for backward compatibility
            formatted_result: dict[str, Any] = {"documents": [], "metadatas": [], "distances": [], "ids": []}
            chunks_data = vector_results.to_chunks_format()
            if chunks_data:
                formatted_result["documents"] = [[chunk["content"] for chunk in chunks_data]]
                formatted_result["metadatas"] = [[chunk["metadata"] for chunk in chunks_data]]
                formatted_result["distances"] = [[chunk["distance"] for chunk in chunks_data]]
                formatted_result["ids"] = [[chunk["document_id"] for chunk in chunks_data]]

            log_component_end("weaviate_collection", "get", f"Retrieved {len(results)} documents")

            return formatted_result

        except Exception as e:
            error_msg = f"Failed to get documents from Weaviate: {str(e)}"
            logger.error("weaviate_collection", "get", error_msg)
            log_error_context(
                "weaviate_collection",
                "get",
                e,
                {"has_ids": ids is not None, "limit": limit, "class_name": self.class_name},
            )
            raise

    def update(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """Update documents in Weaviate collection."""
        logger = get_system_logger()
        log_component_start("weaviate_collection", "update", doc_count=len(ids))

        try:
            collection = self.client.collections.get(self.class_name)

            for i, doc_id in enumerate(ids):
                update_data = {}

                if documents and i < len(documents):
                    update_data["content"] = documents[i]

                if metadatas and i < len(metadatas):
                    metadata = metadatas[i]
                    update_data.update(
                        {
                            "source_file": metadata.get("source_file") or "",
                            "chunk_index": metadata.get("chunk_index") or "",
                            "language": metadata.get("language") or "",
                            "timestamp": metadata.get("timestamp") or "",
                        }
                    )

                if update_data:
                    collection.data.update(
                        uuid=doc_id,
                        properties=update_data,
                        vector=embeddings[i].tolist() if embeddings and i < len(embeddings) else None,
                    )

            logger.info("weaviate_collection", "update", f"Successfully updated {len(ids)} documents")

        except Exception as e:
            error_msg = f"Failed to update documents in Weaviate: {str(e)}"
            logger.error("weaviate_collection", "update", error_msg)
            log_error_context(
                "weaviate_collection", "update", e, {"doc_count": len(ids), "class_name": self.class_name}
            )
            raise

        log_component_end("weaviate_collection", "update", f"Updated {len(ids)} documents successfully")

    def upsert(
        self,
        ids: list[str],
        embeddings: list[np.ndarray] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Upsert documents in Weaviate collection."""
        get_system_logger()
        log_component_start("weaviate_collection", "upsert", doc_count=len(ids))

        # For Weaviate, upsert is handled by the add operation with UUID
        if documents is None:
            raise ValueError("Documents are required for upsert operation")

        self.add(ids, documents, metadatas or [{} for _ in ids], embeddings)

        log_component_end("weaviate_collection", "upsert", f"Upserted {len(ids)} documents successfully")

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        """Delete documents from Weaviate collection."""
        logger = get_system_logger()
        log_component_start("weaviate_collection", "delete", has_ids=ids is not None, has_where=where is not None)

        try:
            collection = self.client.collections.get(self.class_name)

            if ids:
                # Delete specific documents
                for doc_id in ids:
                    collection.data.delete_by_id(doc_id)
                logger.info("weaviate_collection", "delete", f"Deleted {len(ids)} documents by ID")

            elif where:
                # Delete by filter (convert where clause to Weaviate filter)
                # This is a simplified implementation
                collection.data.delete_many(where=Filter.by_property("language").equal(where.get("language", "")))
                logger.info("weaviate_collection", "delete", "Deleted documents by filter")

            else:
                raise ValueError("Either ids or where clause must be provided for deletion")

        except Exception as e:
            error_msg = f"Failed to delete documents from Weaviate: {str(e)}"
            logger.error("weaviate_collection", "delete", error_msg)
            log_error_context(
                "weaviate_collection", "delete", e, {"has_ids": ids is not None, "class_name": self.class_name}
            )
            raise

        log_component_end("weaviate_collection", "delete", "Delete operation completed")

    def count(self) -> int:
        """Count documents in Weaviate collection."""
        try:
            collection = self.client.collections.get(self.class_name)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count or 0

        except Exception as e:
            self.logger.error(f"Failed to count documents in Weaviate: {str(e)}")
            return 0


class WeaviateDatabase(VectorDatabase):
    """Weaviate implementation of VectorDatabase protocol."""

    def __init__(self, client: weaviate.WeaviateClient, config: WeaviateConfiguration):
        """Initialize Weaviate database."""
        get_system_logger()
        log_component_start("weaviate_database", "init")

        self.client = client
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._collections: dict[str, WeaviateCollection] = {}

        # Test connection
        self._test_connection()

        log_component_end("weaviate_database", "init", "Weaviate database initialized")

    def _test_connection(self) -> None:
        """Test Weaviate connection."""
        logger = get_system_logger()

        try:
            # Test basic connectivity
            ready = self.client.is_ready()
            if not ready:
                raise ConnectionError("Weaviate is not ready")

            logger.info("weaviate_database", "test_connection", "Weaviate connection successful")

        except Exception as e:
            error_msg = f"Failed to connect to Weaviate: {str(e)}"
            logger.error("weaviate_database", "test_connection", error_msg)
            log_error_context("weaviate_database", "test_connection", e, {"url": self.config.connection.url})
            raise ConnectionError(error_msg) from e

    def get_collection(self, collection_name: str) -> WeaviateCollection:
        """Get or create Weaviate collection."""
        logger = get_system_logger()

        if collection_name not in self._collections:
            logger.debug("weaviate_database", "get_collection", f"Creating new collection: {collection_name}")

            collection = WeaviateCollection(client=self.client, class_name=collection_name, config=self.config)

            self._collections[collection_name] = collection

        return self._collections[collection_name]

    def create_collection(self, name: str, reset_if_exists: bool = False) -> VectorCollection:
        """Create or get Weaviate collection."""
        logger = get_system_logger()

        if reset_if_exists and name in self._collections:
            logger.debug("weaviate_database", "create_collection", f"Resetting existing collection: {name}")
            # Remove from cache - will be recreated in get_collection
            del self._collections[name]

            # Delete the collection from Weaviate if it exists
            try:
                self.client.collections.delete(name)
                logger.debug("weaviate_database", "create_collection", f"Deleted existing Weaviate collection: {name}")
            except Exception as e:
                logger.debug(
                    "weaviate_database", "create_collection", f"Collection {name} did not exist for deletion: {e}"
                )

        # Delegate to get_collection which creates if needed
        return self.get_collection(name)

    def list_collections(self) -> list[str]:
        """List all Weaviate classes/collections."""
        try:
            # Use collections API to list all collections
            collections_list = []
            for collection_name in self.client.collections.list_all():
                collections_list.append(collection_name)
            return collections_list

        except Exception as e:
            self.logger.error(f"Failed to list Weaviate collections: {str(e)}")
            return []

    def delete_collection(self, collection_name: str) -> None:
        """Delete Weaviate class/collection."""
        logger = get_system_logger()
        log_component_start("weaviate_database", "delete_collection", collection_name=collection_name)

        try:
            # Use collections API to delete collection
            self.client.collections.delete(collection_name)

            # Remove from cache
            if collection_name in self._collections:
                del self._collections[collection_name]

            logger.info("weaviate_database", "delete_collection", f"Successfully deleted collection: {collection_name}")

            log_component_end("weaviate_database", "delete_collection", f"Collection {collection_name} deleted")

        except Exception as e:
            error_msg = f"Failed to delete Weaviate collection {collection_name}: {str(e)}"
            logger.error("weaviate_database", "delete_collection", error_msg)
            log_error_context("weaviate_database", "delete_collection", e, {"collection_name": collection_name})
            raise

    def reset(self) -> None:
        """Reset entire Weaviate database."""
        logger = get_system_logger()
        log_component_start("weaviate_database", "reset_database")

        try:
            # Get all classes and delete them
            collections = self.list_collections()
            for collection_name in collections:
                self.delete_collection(collection_name)

            # Clear collection cache
            self._collections.clear()

            logger.info(
                "weaviate_database",
                "reset_database",
                f"Successfully reset database, deleted {len(collections)} collections",
            )

            log_component_end(
                "weaviate_database", "reset_database", f"Database reset, {len(collections)} collections deleted"
            )

        except Exception as e:
            error_msg = f"Failed to reset Weaviate database: {str(e)}"
            logger.error("weaviate_database", "reset_database", error_msg)
            log_error_context("weaviate_database", "reset_database", e, {})
            raise


def create_weaviate_client(config: WeaviateConfiguration) -> weaviate.WeaviateClient:
    """Create Weaviate client with configuration."""
    logger = get_system_logger()
    log_component_start("weaviate_client_factory", "create_client", url=config.connection.url)

    try:
        # Create client with configuration
        client = weaviate.connect_to_local(
            host=config.connection.host, port=config.connection.port, grpc_port=config.connection.grpc_port
        )

        # Wait for startup
        start_time = time.time()
        timeout = config.connection.startup_period

        while time.time() - start_time < timeout:
            if client.is_ready():
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"Weaviate not ready after {timeout} seconds")

        logger.info(
            "weaviate_client_factory",
            "create_client",
            f"Weaviate client connected successfully to {config.connection.url}",
        )

        log_component_end("weaviate_client_factory", "create_client", "Client created successfully")

        return client

    except Exception as e:
        error_msg = f"Failed to create Weaviate client: {str(e)}"
        logger.error("weaviate_client_factory", "create_client", error_msg)
        log_error_context(
            "weaviate_client_factory",
            "create_client",
            e,
            {"url": config.connection.url, "host": config.connection.host, "port": config.connection.port},
        )
        raise ConnectionError(error_msg) from e


def create_weaviate_database(config: dict[str, Any], language: str) -> WeaviateDatabase:
    """
    Factory function to create Weaviate database with full configuration.

    Args:
        config: Configuration dictionary
        language: Language code for tenant setup

    Returns:
        WeaviateDatabase instance

    Raises:
        ValueError: If configuration is invalid
        ConnectionError: If connection fails
    """
    logger = get_system_logger()
    log_component_start("weaviate_database_factory", "create_database", language=language)

    try:
        # Create Weaviate configuration
        weaviate_config = create_weaviate_configuration(config)

        # Create client
        client = create_weaviate_client(weaviate_config)

        # Create database
        database = WeaviateDatabase(client, weaviate_config)

        logger.info(
            "weaviate_database_factory",
            "create_database",
            f"Weaviate database created successfully for language {language}",
        )

        log_component_end("weaviate_database_factory", "create_database", f"Database created for {language}")

        return database

    except Exception as e:
        error_msg = f"Failed to create Weaviate database for language {language}: {str(e)}"
        logger.error("weaviate_database_factory", "create_database", error_msg)
        log_error_context("weaviate_database_factory", "create_database", e, {"language": language})
        raise


# Convenience function for backward compatibility
def create_weaviate_collection(
    client: weaviate.WeaviateClient, collection_name: str, config: WeaviateConfiguration
) -> WeaviateCollection:
    """Create a Weaviate collection."""
    return WeaviateCollection(client, collection_name, config)
