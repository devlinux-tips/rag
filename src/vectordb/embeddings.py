"""
Embedding model management for Croatian RAG system.
Handles multilingual sentence-transformers models optimized for Croatian language.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model_name: str = "sentence-transformers/LaBSE"
    cache_dir: str = "./models/embeddings"
    device: str = "auto"  # "auto", "cpu", "cuda"
    max_seq_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True


class CroatianEmbeddingModel:
    """Multilingual embedding model optimized for Croatian text."""

    # Recommended models for Croatian language
    RECOMMENDED_MODELS = {
        "multilingual-minilm": "paraphrase-multilingual-MiniLM-L12-v2",
        "multilingual-mpnet": "paraphrase-multilingual-mpnet-base-v2",
        "distiluse-multilingual": "distiluse-base-multilingual-cased",
        "labse": "sentence-transformers/LaBSE",
    }

    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize embedding model.

        Args:
            config: Configuration for embedding model
        """
        self.config = config or EmbeddingConfig()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._model_loaded = False

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Set device
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device

    def load_model(self) -> None:
        """Load the sentence transformer model."""
        if self._model_loaded:
            return

        try:
            self.logger.info(f"Loading embedding model: {self.config.model_name}")

            # Load model with caching
            self.model = SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_dir,
                device=self.device,
            )

            # Configure model settings
            self.model.max_seq_length = self.config.max_seq_length

            self._model_loaded = True
            self.logger.info("Embedding model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing (uses config default if None)
            show_progress: Show progress bar for large batches

        Returns:
            Numpy array of embeddings
        """
        if not self._model_loaded:
            self.load_model()

        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        batch_size = batch_size or self.config.batch_size

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Failed to encode texts: {e}")
            raise

    def encode_documents(
        self, documents: List[Dict[str, Any]], text_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Encode a list of document dictionaries.

        Args:
            documents: List of document dictionaries
            text_field: Field name containing text to embed

        Returns:
            List of documents with added embeddings
        """
        if not documents:
            return []

        # Extract texts
        texts = [doc.get(text_field, "") for doc in documents]

        # Generate embeddings
        self.logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.encode_text(texts, show_progress=True)

        # Add embeddings to documents
        enriched_docs = []
        for doc, embedding in zip(documents, embeddings):
            doc_copy = doc.copy()
            doc_copy["embedding"] = embedding
            doc_copy["embedding_model"] = self.config.model_name
            enriched_docs.append(doc_copy)

        return enriched_docs

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ("cosine", "dot", "euclidean")

        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)

        elif metric == "dot":
            # Dot product similarity
            return np.dot(embedding1, embedding2)

        elif metric == "euclidean":
            # Negative euclidean distance (higher is more similar)
            return -np.linalg.norm(embedding1 - embedding2)

        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings to a query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top results to return
            metric: Similarity metric to use

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate, metric)
            similarities.append((i, similarity))

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self._model_loaded:
            self.load_model()

        return {
            "model_name": self.config.model_name,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "device": str(self.device),
            "normalize_embeddings": self.config.normalize_embeddings,
            "pooling_mode": str(self.model._modules.get("1", "unknown")),
        }

    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], filepath: str) -> None:
        """
        Save embeddings and metadata to file.

        Args:
            embeddings: Embedding vectors
            metadata: Corresponding metadata
            filepath: Path to save file
        """
        data = {
            "embeddings": embeddings,
            "metadata": metadata,
            "model_name": self.config.model_name,
            "model_info": self.get_model_info(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        self.logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")

    def load_embeddings(self, filepath: str) -> tuple[np.ndarray, List[Dict]]:
        """
        Load embeddings and metadata from file.

        Args:
            filepath: Path to embedding file

        Returns:
            Tuple of (embeddings, metadata)
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        embeddings = data["embeddings"]
        metadata = data["metadata"]

        self.logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")
        return embeddings, metadata


class EmbeddingCache:
    """Cache for storing and retrieving embeddings."""

    def __init__(self, cache_dir: str = "./cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model combination."""
        import hashlib

        combined = f"{model_name}:{text}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for text.

        Args:
            text: Input text
            model_name: Model name used for embedding

        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.npy"

        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {e}")

        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """
        Cache embedding for text.

        Args:
            text: Input text
            model_name: Model name used for embedding
            embedding: Embedding vector to cache
        """
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.npy"

        try:
            np.save(cache_file, embedding)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {e}")


def create_embedding_model(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    device: str = "auto",
    cache_dir: str = "./models/embeddings",
) -> CroatianEmbeddingModel:
    """
    Factory function to create embedding model.

    Args:
        model_name: Sentence transformer model name
        device: Device to use ("auto", "cpu", "cuda")
        cache_dir: Directory for model cache

    Returns:
        Configured CroatianEmbeddingModel instance
    """
    config = EmbeddingConfig(model_name=model_name, device=device, cache_dir=cache_dir)
    return CroatianEmbeddingModel(config)


def get_recommended_model(use_case: str = "general") -> str:
    """
    Get recommended model for specific use case.

    Args:
        use_case: Use case type ("general", "fast", "accurate", "cross-lingual")

    Returns:
        Recommended model name
    """
    models = CroatianEmbeddingModel.RECOMMENDED_MODELS

    recommendations = {
        "general": models["multilingual-minilm"],  # Good balance
        "fast": models["multilingual-minilm"],  # Fastest option
        "accurate": models["multilingual-mpnet"],  # Most accurate
        "cross-lingual": models["labse"],  # Best for multiple languages
    }

    return recommendations.get(use_case, models["multilingual-minilm"])
