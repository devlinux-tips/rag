"""
Provider implementations for search system dependencies.
Includes production implementations and mock providers for testing.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .search import ConfigProvider, EmbeddingProvider, VectorSearchProvider

logger = logging.getLogger(__name__)


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
        self._embedding_cache = {}
        self.logger = logging.getLogger(__name__)

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

        self.logger.debug(f"Generated embedding for text (length: {len(text)})")
        return embedding


class MockVectorSearchProvider(VectorSearchProvider):
    """Mock vector search provider for testing."""

    def __init__(self):
        """Initialize mock search provider."""
        self.documents = (
            {}
        )  # id -> {"content": str, "embedding": np.ndarray, "metadata": dict}
        self.logger = logging.getLogger(__name__)

    def add_document(
        self,
        doc_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add document for testing."""
        self.documents[doc_id] = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
        }

    async def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Mock embedding-based search."""
        if not self.documents:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

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

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }

    async def search_by_text(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Mock text-based search using simple keyword matching."""
        if not self.documents:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

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
                {
                    "id": doc_id,
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "distance": distance,
                }
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

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        if document_id in self.documents:
            return {
                "id": document_id,
                "content": self.documents[document_id]["content"],
                "metadata": self.documents[document_id]["metadata"],
            }
        return None

    def _matches_filters(
        self, metadata: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True


class MockConfigProvider(ConfigProvider):
    """Mock configuration provider for testing."""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize mock config provider.

        Args:
            custom_config: Optional custom configuration
        """
        self.config = custom_config or self._default_config()
        self.logger = logging.getLogger(__name__)

    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration."""
        return self.config["search"]

    def get_scoring_weights(self) -> Dict[str, float]:
        """Get scoring weights for hybrid search."""
        return self.config["scoring"]["weights"]

    def _default_config(self) -> Dict[str, Any]:
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
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            self.device = device
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Loaded embedding model {model_name} on {device}")
        except ImportError:
            raise ImportError(
                "sentence-transformers package required for production embedding provider"
            )

    async def encode_text(self, text: str) -> np.ndarray:
        """Encode text using sentence-transformers model."""
        try:
            # Run encoding in thread pool to avoid blocking async loop
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.model.encode(text, normalize_embeddings=True)
            )

            # Ensure numpy array format
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            # Ensure 1D array (some models return 2D with single row)
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]

            self.logger.debug(f"Encoded text to {embedding.shape} embedding")
            return embedding.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            raise


class ChromaDBSearchProvider(VectorSearchProvider):
    """Production search provider using ChromaDB."""

    def __init__(self, collection):
        """
        Initialize ChromaDB search provider.

        Args:
            collection: ChromaDB collection instance
        """
        self.collection = collection
        self.logger = logging.getLogger(__name__)

    async def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Search using embedding vector."""
        try:
            # Run ChromaDB query in thread pool
            loop = asyncio.get_event_loop()

            query_kwargs = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
                if include_metadata
                else ["documents", "distances"],
            }

            if filters:
                query_kwargs["where"] = filters

            results = await loop.run_in_executor(
                None, lambda: self.collection.query(**query_kwargs)
            )

            self.logger.debug(
                f"ChromaDB embedding search returned {len(results.get('ids', [[]])[0])} results"
            )
            return results

        except Exception as e:
            self.logger.error(f"ChromaDB embedding search failed: {e}")
            raise

    async def search_by_text(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Search using query text (relies on ChromaDB's built-in embedding)."""
        try:
            loop = asyncio.get_event_loop()

            query_kwargs = {
                "query_texts": [query_text],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
                if include_metadata
                else ["documents", "distances"],
            }

            if filters:
                query_kwargs["where"] = filters

            results = await loop.run_in_executor(
                None, lambda: self.collection.query(**query_kwargs)
            )

            self.logger.debug(
                f"ChromaDB text search returned {len(results.get('ids', [[]])[0])} results"
            )
            return results

        except Exception as e:
            self.logger.error(f"ChromaDB text search failed: {e}")
            raise

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID from ChromaDB."""
        try:
            loop = asyncio.get_event_loop()

            results = await loop.run_in_executor(
                None,
                lambda: self.collection.get(
                    ids=[document_id], include=["documents", "metadatas"]
                ),
            )

            if results and results.get("ids") and results["ids"]:
                return {
                    "id": document_id,
                    "content": results["documents"][0]
                    if results.get("documents")
                    else "",
                    "metadata": results["metadatas"][0]
                    if results.get("metadatas")
                    else {},
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            return None


class ProductionConfigProvider(ConfigProvider):
    """Production configuration provider using config files."""

    def __init__(self, config_loader_func=None):
        """
        Initialize production config provider.

        Args:
            config_loader_func: Function to load configuration
        """
        if config_loader_func:
            self.config_loader = config_loader_func
        else:
            # Use default config loader if available
            try:
                from ..utils.config_loader import (get_search_config,
                                                   get_shared_config)

                self.get_search_config_func = get_search_config
                self.get_shared_config_func = get_shared_config
            except ImportError:
                self.get_search_config_func = None
                self.get_shared_config_func = None

        self.logger = logging.getLogger(__name__)

    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration from config files."""
        if self.get_search_config_func:
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

    def get_scoring_weights(self) -> Dict[str, float]:
        """Get scoring weights from configuration."""
        search_config = self.get_search_config()
        weights = search_config.get("weights", {})
        return {
            "semantic": weights.get("semantic_weight", 0.7),
            "keyword": weights.get("keyword_weight", 0.3),
        }


# Factory Functions
def create_mock_embedding_provider(dimension: int = 384) -> EmbeddingProvider:
    """Create mock embedding provider for testing."""
    return MockEmbeddingProvider(dimension=dimension, deterministic=True)


def create_mock_search_provider() -> VectorSearchProvider:
    """Create mock search provider for testing."""
    return MockVectorSearchProvider()


def create_mock_config_provider(
    custom_config: Optional[Dict[str, Any]] = None
) -> ConfigProvider:
    """Create mock config provider for testing."""
    return MockConfigProvider(custom_config=custom_config)


def create_production_embedding_provider(
    model_name: str = "BAAI/bge-m3", device: str = "cpu"
) -> EmbeddingProvider:
    """Create production embedding provider."""
    return SentenceTransformerEmbeddingProvider(model_name=model_name, device=device)


def create_chromadb_search_provider(collection) -> VectorSearchProvider:
    """Create production ChromaDB search provider."""
    return ChromaDBSearchProvider(collection)


def create_production_config_provider(config_loader_func=None) -> ConfigProvider:
    """Create production configuration provider."""
    return ProductionConfigProvider(config_loader_func=config_loader_func)
