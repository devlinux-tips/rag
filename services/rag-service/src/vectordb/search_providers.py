"""
Provider implementations for search system dependencies.
Includes production implementations and mock providers for testing.
"""

import asyncio
from typing import Any

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
from .search import ConfigProvider, EmbeddingProvider, VectorSearchProvider


# Mock Providers for Testing
class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384, deterministic: bool = True):
        """
        Initialize mock embedding provider.

        Args:
            dimension: Embedding vector dimension
            deterministic: If True, same text produces same embedding
        """
        self.dimension = dimension
        self.deterministic = deterministic
        self._embedding_cache: dict[str, np.ndarray] = {}
        self.logger = get_system_logger()

    async def encode_text(self, text: str) -> np.ndarray:
        """Generate mock embedding for text."""
        if self.deterministic and text in self._embedding_cache:
            return self._embedding_cache[text]

        # Generate deterministic or random embedding
        if self.deterministic:
            # Use hash of text for deterministic embedding
            text_hash = hash(text) % (2**31)  # Ensure positive
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            self._embedding_cache[text] = embedding
        else:
            embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

        self.logger.debug(
            "search_providers", "text_to_embedding", f"Generated embedding for text (length: {len(text)})"
        )
        return embedding


class MockVectorSearchProvider(VectorSearchProvider):
    """Mock vector search provider for testing."""

    def __init__(self):
        """Initialize mock search provider."""
        self.documents = {}  # id -> {"content": str, "embedding": np.ndarray, "metadata": dict}
        self.logger = get_system_logger()

    def add_document(self, doc_id: str, content: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None):
        """Add document for testing."""
        self.documents[doc_id] = {"content": content, "embedding": embedding, "metadata": metadata or {}}

    async def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Mock embedding-based search."""
        if not self.documents:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Calculate similarities
        similarities = []
        for doc_id, doc_data in self.documents.items():
            # Skip if filters don't match
            if filters and not self._matches_filters(doc_data["metadata"], filters):
                continue

            doc_embedding = doc_data["embedding"]
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            distance = 1.0 - similarity  # Convert to distance

            similarities.append(
                {
                    "id": doc_id,
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "distance": max(0.0, distance),  # Clamp to non-negative
                }
            )

        # Sort by distance (ascending - lower is better)
        similarities.sort(key=lambda x: x["distance"])

        # Limit results
        similarities = similarities[:top_k]

        # Format as ChromaDB-style results
        ids = [[item["id"] for item in similarities]]
        documents = [[item["content"] for item in similarities]]
        metadatas = [[item["metadata"] for item in similarities]]
        distances = [[item["distance"] for item in similarities]]

        return {"ids": ids, "documents": documents, "metadatas": metadatas, "distances": distances}

    async def search_by_text(
        self, query_text: str, top_k: int, filters: dict[str, Any] | None = None, include_metadata: bool = True
    ) -> dict[str, Any]:
        """Mock text-based search using simple keyword matching."""
        if not self.documents:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_terms = set(query_text.lower().split())
        scores = []

        for doc_id, doc_data in self.documents.items():
            # Skip if filters don't match
            if filters and not self._matches_filters(doc_data["metadata"], filters):
                continue

            content = doc_data["content"].lower()
            doc_terms = set(content.split())

            # Simple keyword overlap score
            if query_terms:
                overlap = len(query_terms.intersection(doc_terms))
                score = overlap / len(query_terms)
            else:
                score = 0.0

            # Boost for exact phrase matches
            if query_text.lower() in content:
                score *= 1.5

            distance = 1.0 - min(1.0, score)  # Convert to distance

            scores.append(
                {"id": doc_id, "content": doc_data["content"], "metadata": doc_data["metadata"], "distance": distance}
            )

        # Sort by distance (ascending)
        scores.sort(key=lambda x: x["distance"])

        # Limit results
        scores = scores[:top_k]

        # Format as ChromaDB-style results
        ids = [[item["id"] for item in scores]]
        documents = [[item["content"] for item in scores]]
        metadatas = [[item["metadata"] for item in scores]]
        distances = [[item["distance"] for item in scores]]

        return {"ids": ids, "documents": documents, "metadatas": metadatas, "distances": distances}

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get document by ID."""
        if document_id in self.documents:
            return {
                "id": document_id,
                "content": self.documents[document_id]["content"],
                "metadata": self.documents[document_id]["metadata"],
            }
        return None

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True


class MockConfigProvider(ConfigProvider):
    """Mock configuration provider for testing."""

    def __init__(self, custom_config: dict[str, Any] | None = None):
        """
        Initialize mock config provider.

        Args:
            custom_config: Optional custom configuration
        """
        self.config = custom_config or self._default_config()
        self.logger = get_system_logger()

    def get_search_config(self) -> dict[str, Any]:
        """Get search configuration."""
        return self.config["search"]

    def get_scoring_weights(self) -> dict[str, float]:
        """Get scoring weights for hybrid search."""
        return self.config["scoring"]["weights"]

    def _default_config(self) -> dict[str, Any]:
        """Default test configuration."""
        return {
            "search": {
                "default_method": "semantic",
                "top_k": 5,
                "similarity_threshold": 0.0,
                "max_context_length": 2000,
                "rerank": True,
                "include_metadata": True,
                "include_distances": True,
            },
            "scoring": {
                "weights": {"semantic": 0.7, "keyword": 0.3},
                "boost_factors": {
                    "term_overlap": 0.2,
                    "length_optimal": 1.0,
                    "length_short": 0.8,
                    "length_long": 0.9,
                    "title_boost": 1.1,
                    "phrase_match_boost": 1.5,
                },
            },
        }


# Production Providers (Adapters for existing components)
class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Production embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        """
        Initialize sentence-transformers embedding provider.

        Args:
            model_name: HuggingFace model name
            device: Device to run model on (cpu, cuda, mps)
        """
        get_system_logger()
        log_component_start("embedding_provider", "init", model=model_name, device=device)

        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            self.device = device
            self.logger = get_system_logger()

            embedding_dim = self.model.get_sentence_embedding_dimension()
            log_performance_metric(
                "embedding_provider",
                "init",
                "embedding_dimension",
                float(embedding_dim) if embedding_dim is not None else 0.0,
            )
            log_component_end("embedding_provider", "init", f"Loaded {model_name} on {device} ({embedding_dim}D)")

        except Exception as e:
            log_error_context("embedding_provider", "init", e, {"model_name": model_name, "device": device})
            raise

    async def encode_text(self, text: str) -> np.ndarray:
        """Encode text using sentence-transformers model."""
        logger = get_system_logger()
        log_component_start("embedding_provider", "encode_text", text_length=len(text), model=self.model_name)

        try:
            logger.trace("embedding_provider", "encode_text", f"Encoding text: '{text[:100]}...' ({len(text)} chars)")

            # Run encoding in thread pool to avoid blocking async loop
            loop = asyncio.get_event_loop()
            raw_embedding = await loop.run_in_executor(None, lambda: self.model.encode(text, normalize_embeddings=True))
            embedding: np.ndarray = np.asarray(raw_embedding)

            logger.trace("embedding_provider", "encode_text", f"Raw embedding shape: {embedding.shape}")

            # Ensure numpy array format
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
                logger.trace("embedding_provider", "encode_text", "Converted to numpy array")

            # Ensure 1D array (some models return 2D with single row)
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]
                logger.trace("embedding_provider", "encode_text", "Flattened 2D embedding to 1D")

            final_embedding = embedding.astype(np.float32)
            log_data_transformation(
                "embedding_provider", "encode_text", f"text ({len(text)} chars)", f"embedding {final_embedding.shape}"
            )
            log_component_end("embedding_provider", "encode_text", f"Generated {final_embedding.shape} embedding")
            return final_embedding

        except Exception as e:
            log_error_context(
                "embedding_provider", "encode_text", e, {"text_length": len(text), "model": self.model_name}
            )
            raise


class ChromaDBSearchProvider(VectorSearchProvider):
    """Production search provider using ChromaDB."""

    def __init__(self, collection, embedding_provider=None):
        """
        Initialize ChromaDB search provider.

        Args:
            collection: ChromaDB collection instance
            embedding_provider: Provider to generate embeddings for text queries
        """
        get_system_logger()
        log_component_start("search_provider", "init", provider_type="ChromaDB")

        self.collection = collection
        self.embedding_provider = embedding_provider
        self.logger = get_system_logger()

        collection_name = getattr(collection, "name", "unknown")
        log_component_end("search_provider", "init", f"ChromaDB provider initialized: {collection_name}")

    async def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Search using embedding vector."""
        logger = get_system_logger()
        log_component_start(
            "search_provider",
            "search_by_embedding",
            embedding_shape=query_embedding.shape,
            top_k=top_k,
            has_filters=bool(filters),
        )

        try:
            logger.trace("search_provider", "search_by_embedding", f"Query embedding shape: {query_embedding.shape}")
            logger.debug("search_provider", "search_by_embedding", f"Searching for top {top_k} results")

            # Run ChromaDB query in thread pool
            loop = asyncio.get_event_loop()

            query_kwargs = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k,
                "include": (
                    ["documents", "metadatas", "distances"] if include_metadata else ["documents", "distances"]
                ),
            }

            if filters:
                query_kwargs["where"] = filters
                logger.debug("search_provider", "search_by_embedding", f"Applied filters: {filters}")

            log_decision_point(
                "search_provider", "search_by_embedding", f"include_metadata={include_metadata}", "executing_query"
            )

            results = await loop.run_in_executor(None, lambda: self.collection.query(**query_kwargs))

            num_results = len(results["ids"][0]) if results and "ids" in results and results["ids"] else 0
            log_performance_metric("search_provider", "search_by_embedding", "results_returned", float(num_results))
            log_component_end("search_provider", "search_by_embedding", f"ChromaDB returned {num_results} results")
            return results

        except Exception as e:
            log_error_context(
                "search_provider",
                "search_by_embedding",
                e,
                {"embedding_shape": query_embedding.shape, "top_k": top_k, "filters": filters},
            )
            raise

    async def search_by_text(
        self, query_text: str, top_k: int, filters: dict[str, Any] | None = None, include_metadata: bool = True
    ) -> dict[str, Any]:
        """Search using query text with proper embedding model."""
        try:
            if self.embedding_provider:
                # Use provided embedding provider to generate query embedding
                query_embedding = await self.embedding_provider.encode_text(query_text)
                return await self.search_by_embedding(query_embedding, top_k, filters, include_metadata)
            else:
                # Fallback to ChromaDB's built-in embedding (may cause dimension mismatch)
                loop = asyncio.get_event_loop()
                query_kwargs = {
                    "query_texts": [query_text],
                    "n_results": top_k,
                    "include": (
                        ["documents", "metadatas", "distances"] if include_metadata else ["documents", "distances"]
                    ),
                }

                if filters:
                    query_kwargs["where"] = filters

                results = await loop.run_in_executor(None, lambda: self.collection.query(**query_kwargs))
                self.logger.debug(
                    "search_providers",
                    "text_search",
                    f"ChromaDB text search returned {len(results.get('ids', [[]])[0])} results",
                )
                return results

        except Exception as e:
            self.logger.error("search_providers", "text_search", f"ChromaDB text search failed: {e}")
            raise

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get document by ID from ChromaDB."""
        try:
            loop = asyncio.get_event_loop()

            results = await loop.run_in_executor(
                None, lambda: self.collection.get(ids=[document_id], include=["documents", "metadatas"])
            )

            if results and results.get("ids") and results["ids"]:
                get_system_logger().debug("search_provider", "get_document", f"Document found: {document_id}")
                document = {
                    "id": document_id,
                    "content": (results["documents"][0] if results.get("documents") else ""),
                    "metadata": (results["metadatas"][0] if results.get("metadatas") else {}),
                }
                log_component_end("search_provider", "get_document", f"Retrieved document: {document_id}")
                return document

            get_system_logger().debug("search_provider", "get_document", f"Document not found: {document_id}")
            log_component_end("search_provider", "get_document", f"Document not found: {document_id}")
            return None

        except Exception as e:
            log_error_context("search_provider", "get_document", e, {"document_id": document_id})
            return None


class DefaultConfigProvider(ConfigProvider):
    """Default configuration provider using config files."""

    def __init__(self, config_loader=None):
        """Initialize default config provider."""
        # Use provided config loader or default
        if config_loader is not None:
            self.config_loader = config_loader
            self.get_search_config_func = getattr(config_loader, "get_search_config", None)
            self.get_shared_config_func = getattr(config_loader, "get_shared_config", None)
        else:
            # Use default config loader
            from ..utils.config_loader import get_search_config, get_shared_config

            self.config_loader = None
            self.get_search_config_func = get_search_config
            self.get_shared_config_func = get_shared_config

        self.logger = get_system_logger()

    def get_search_config(self) -> dict[str, Any]:
        """Get search configuration from config files."""
        if self.get_search_config_func is not None:
            return self.get_search_config_func()
        else:
            # Fallback default configuration
            return {
                "default_method": "semantic",
                "top_k": 5,
                "similarity_threshold": 0.0,
                "max_context_length": 2000,
                "rerank": True,
                "include_metadata": True,
                "include_distances": True,
            }

    def get_scoring_weights(self) -> dict[str, float]:
        """Get scoring weights from configuration."""
        search_config = self.get_search_config()
        if "weights" not in search_config:
            raise ValueError("Missing 'weights' section in search configuration")
        weights = search_config["weights"]

        # Validate required weight keys
        if "semantic_weight" not in weights:
            raise ValueError("Missing 'semantic_weight' in search weights configuration")
        if "keyword_weight" not in weights:
            raise ValueError("Missing 'keyword_weight' in search weights configuration")

        return {"semantic": weights["semantic_weight"], "keyword": weights["keyword_weight"]}


# Factory Functions
def create_mock_embedding_provider(dimension: int = 384) -> EmbeddingProvider:
    """Create mock embedding provider for testing."""
    return MockEmbeddingProvider(dimension=dimension, deterministic=True)


def create_mock_search_provider() -> VectorSearchProvider:
    """Create mock search provider for testing."""
    return MockVectorSearchProvider()


def create_mock_config_provider(custom_config: dict[str, Any] | None = None) -> ConfigProvider:
    """Create mock config provider for testing."""
    return MockConfigProvider(custom_config=custom_config)


def create_embedding_provider(model_name: str = "BAAI/bge-m3", device: str = "cpu") -> EmbeddingProvider:
    """Create production embedding provider."""
    return SentenceTransformerEmbeddingProvider(model_name=model_name, device=device)


def create_vector_search_provider(collection, embedding_provider=None) -> VectorSearchProvider:
    """Create vector search provider."""
    return ChromaDBSearchProvider(collection, embedding_provider)


def create_config_provider(config_loader_func=None) -> ConfigProvider:
    """Create default configuration provider."""
    return DefaultConfigProvider(config_loader=config_loader_func)
