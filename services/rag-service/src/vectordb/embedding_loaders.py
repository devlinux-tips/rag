"""
Embedding model loaders for dependency injection.
Provides testable model loading abstraction layer.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .embeddings import EmbeddingModel, ModelLoader


class SentenceTransformerLoader:
    """
    Production model loader using sentence-transformers library.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_model(
        self, model_name: str, cache_dir: str, device: str, **kwargs
    ) -> EmbeddingModel:
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            # Create cache directory
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            # Load model with configuration
            model = SentenceTransformer(
                model_name, cache_folder=cache_dir, device=device, **kwargs
            )

            self.logger.info(f"Loaded model {model_name} on device {device}")
            return SentenceTransformerAdapter(model)

        except ImportError:
            raise ImportError(
                "sentence-transformers library is required for SentenceTransformerLoader"
            )
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available for loading."""
        try:
            # Try to access model info without downloading
            from huggingface_hub import model_info
            from sentence_transformers import SentenceTransformer

            # Check if it's a valid Hugging Face model
            info = model_info(model_name)
            return info is not None

        except Exception:
            # If any error occurs, assume model is not available
            return False


class SentenceTransformerAdapter:
    """
    Adapter to make SentenceTransformer compatible with EmbeddingModel protocol.
    """

    def __init__(self, model):
        self._model = model

    def encode(
        self, texts, batch_size: int = 32, normalize_embeddings: bool = True, **kwargs
    ):
        """Generate embeddings for texts."""
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )

    @property
    def device(self) -> str:
        """Get current device."""
        return str(self._model.device)

    @property
    def max_seq_length(self) -> int:
        """Get maximum sequence length."""
        return getattr(self._model, "max_seq_length", 512)

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._model.get_sentence_embedding_dimension()


class MockModelLoader:
    """
    Mock model loader for testing.
    Allows complete control over model behavior for unit tests.
    """

    def __init__(self):
        self.models = {}
        self.available_models = set()
        self.call_log = []
        self.should_raise = None

    def set_model(self, model_name: str, model: EmbeddingModel):
        """Set mock model for specific name."""
        self.models[model_name] = model
        self.available_models.add(model_name)

    def set_available_models(self, model_names):
        """Set list of available models."""
        self.available_models = set(model_names)

    def set_exception(self, exception: Exception):
        """Set exception to raise on next load."""
        self.should_raise = exception

    def get_calls(self):
        """Get log of all method calls made."""
        return self.call_log.copy()

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    def load_model(
        self, model_name: str, cache_dir: str, device: str, **kwargs
    ) -> EmbeddingModel:
        """Mock model loading."""
        self.call_log.append(
            {
                "method": "load_model",
                "model_name": model_name,
                "cache_dir": cache_dir,
                "device": device,
                "kwargs": kwargs,
            }
        )

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        if model_name in self.models:
            return self.models[model_name]

        # Return default mock model
        return MockEmbeddingModel(model_name, device)

    def is_model_available(self, model_name: str) -> bool:
        """Mock model availability check."""
        self.call_log.append({"method": "is_model_available", "model_name": model_name})

        return model_name in self.available_models


class MockEmbeddingModel:
    """
    Mock embedding model for testing.
    """

    def __init__(self, model_name: str = "mock-model", device: str = "cpu"):
        self.model_name = model_name
        self._device = device
        self._max_seq_length = 512
        self._embedding_dim = 1024
        self.call_log = []

    def encode(
        self, texts, batch_size: int = 32, normalize_embeddings: bool = True, **kwargs
    ):
        """Mock embedding generation."""
        import numpy as np

        self.call_log.append(
            {
                "method": "encode",
                "num_texts": len(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "kwargs": kwargs,
            }
        )

        # Generate mock embeddings
        embeddings = np.random.rand(len(texts), self._embedding_dim).astype(np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)

        return embeddings

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device

    @property
    def max_seq_length(self) -> int:
        """Get maximum sequence length."""
        return self._max_seq_length

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim

    def set_embedding_dimension(self, dim: int):
        """Set embedding dimension for testing."""
        self._embedding_dim = dim

    def get_calls(self):
        """Get call log."""
        return self.call_log.copy()

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()
