"""
Embedding model loaders for dependency injection.
Provides testable model loading abstraction layer.
"""

import logging
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_decision_point,
    log_error_context,
    log_performance_metric,
)
from .embeddings import EmbeddingModel


class SentenceTransformerLoader:
    """
    Production model loader using sentence-transformers library.
    """

    def __init__(self):
        get_system_logger()
        log_component_start("model_loader", "init", loader_type="SentenceTransformer")
        self.logger = logging.getLogger(__name__)
        log_component_end("model_loader", "init", "SentenceTransformer loader initialized")

    def load_model(self, model_name: str, cache_dir: str, device: str, **kwargs) -> EmbeddingModel:
        """Load sentence-transformers model."""
        logger = get_system_logger()
        log_component_start("model_loader", "load_model", model=model_name, device=device, cache_dir=cache_dir)

        try:
            # Create cache directory
            cache_path = Path(cache_dir)
            logger.debug("model_loader", "load_model", f"Creating cache directory: {cache_path}")
            cache_path.mkdir(parents=True, exist_ok=True)

            # Filter out parameters not supported by SentenceTransformer
            supported_params = {}
            unsupported_params = []

            for key, value in kwargs.items():
                if key == "trust_remote_code":
                    supported_params[key] = value
                    logger.trace("model_loader", "load_model", f"Added supported param: {key}={value}")
                else:
                    unsupported_params.append(key)

            if unsupported_params:
                logger.debug("model_loader", "load_model", f"Ignored unsupported params: {unsupported_params}")

            log_decision_point(
                "model_loader", "load_model", f"params_filtered={len(supported_params)}", f"loading_{model_name}"
            )

            # Load model with only supported configuration
            logger.info("model_loader", "load_model", f"Loading {model_name} on {device}")
            model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device, **supported_params)

            # Get model properties for logging
            embedding_dim = model.get_sentence_embedding_dimension()
            max_seq_len = getattr(model, "max_seq_length", "unknown")

            log_performance_metric(
                "model_loader",
                "load_model",
                "embedding_dimension",
                float(embedding_dim) if embedding_dim is not None else 0.0,
            )
            log_performance_metric(
                "model_loader",
                "load_model",
                "max_sequence_length",
                float(max_seq_len) if isinstance(max_seq_len, (int, float)) else 0.0,
            )

            adapter = SentenceTransformerAdapter(model)
            log_component_end(
                "model_loader",
                "load_model",
                f"Loaded {model_name} ({embedding_dim}D) on {device}",
                model=model_name,
                device=device,
                dimension=embedding_dim,
            )
            return adapter

        except Exception as e:
            log_error_context(
                "model_loader",
                "load_model",
                e,
                {"model_name": model_name, "device": device, "cache_dir": cache_dir, "kwargs": kwargs},
            )
            raise RuntimeError(f"Model loading failed: {e}") from e

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available for loading."""
        logger = get_system_logger()
        log_component_start("model_loader", "is_model_available", model=model_name)

        try:
            # Try to access model info without downloading
            from huggingface_hub import model_info

            logger.debug("model_loader", "is_model_available", f"Checking HuggingFace model: {model_name}")

            # Check if it's a valid Hugging Face model
            info = model_info(model_name)
            available = info is not None

            log_decision_point("model_loader", "is_model_available", f"model={model_name}", f"available={available}")
            log_component_end(
                "model_loader", "is_model_available", f"Model {model_name}: {'available' if available else 'not found'}"
            )
            return available

        except Exception as e:
            logger.debug("model_loader", "is_model_available", f"Model {model_name} not available: {str(e)}")
            log_component_end("model_loader", "is_model_available", f"Model {model_name}: unavailable (error)")
            return False


class SentenceTransformerAdapter:
    """
    Adapter to make SentenceTransformer compatible with EmbeddingModel protocol.
    """

    def __init__(self, model):
        get_system_logger()
        log_component_start("model_adapter", "init", model_type=type(model).__name__, device=str(model.device))
        self._model = model
        log_component_end("model_adapter", "init", "SentenceTransformer adapter created")

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True, **kwargs):
        """Generate embeddings for texts."""
        logger = get_system_logger()
        log_component_start(
            "model_adapter", "encode", text_count=len(texts), batch_size=batch_size, normalize=normalize_embeddings
        )

        try:
            logger.trace("model_adapter", "encode", f"Encoding {len(texts)} texts with batch_size={batch_size}")
            embeddings = self._model.encode(
                texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings, **kwargs
            )

            log_performance_metric(
                "model_adapter",
                "encode",
                "output_dim",
                float(embeddings.shape[1]) if len(embeddings.shape) > 1 else float(len(embeddings)),
            )
            log_component_end(
                "model_adapter",
                "encode",
                f"Generated embeddings: {embeddings.shape}",
                text_count=len(texts),
                output_shape=embeddings.shape,
            )
            return embeddings

        except Exception as e:
            log_error_context(
                "model_adapter",
                "encode",
                e,
                {
                    "text_count": len(texts),
                    "batch_size": batch_size,
                    "normalize_embeddings": normalize_embeddings,
                    "kwargs": kwargs,
                },
            )
            raise

    @property
    def device(self) -> str:
        """Get current device."""
        device = str(self._model.device)
        logger = get_system_logger()
        logger.trace("model_adapter", "device", f"Current device: {device}")
        return device

    @property
    def max_seq_length(self) -> int:
        """Get maximum sequence length."""
        max_len = getattr(self._model, "max_seq_length", 512)
        logger = get_system_logger()
        logger.trace("model_adapter", "max_seq_length", f"Max sequence length: {max_len}")
        return max_len

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        dimension = self._model.get_sentence_embedding_dimension()
        logger = get_system_logger()
        logger.trace("model_adapter", "get_sentence_embedding_dimension", f"Embedding dimension: {dimension}")
        return dimension


class MockModelLoader:
    """
    Mock model loader for testing.
    Allows complete control over model behavior for unit tests.
    """

    def __init__(self):
        get_system_logger()
        log_component_start("mock_loader", "init", loader_type="Mock")
        self.models = {}
        self.available_models = set()
        self.call_log = []
        self.should_raise = None
        log_component_end("mock_loader", "init", "Mock loader initialized")

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

    def load_model(self, model_name: str, cache_dir: str, device: str, **kwargs) -> EmbeddingModel:
        """Mock model loading."""
        logger = get_system_logger()
        log_component_start("mock_loader", "load_model", model=model_name, device=device, is_mock=True)

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
            logger.debug("mock_loader", "load_model", f"Raising configured exception: {exception}")
            raise exception

        if model_name in self.models:
            logger.debug("mock_loader", "load_model", f"Returning pre-configured mock model: {model_name}")
            log_component_end("mock_loader", "load_model", f"Loaded pre-configured mock: {model_name}")
            return self.models[model_name]

        # Return default mock model
        logger.debug("mock_loader", "load_model", f"Creating default mock model: {model_name}")
        mock_model = MockEmbeddingModel(model_name, device)
        log_component_end("mock_loader", "load_model", f"Created default mock: {model_name}")
        return mock_model

    def is_model_available(self, model_name: str) -> bool:
        """Mock model availability check."""
        get_system_logger()
        log_component_start("mock_loader", "is_model_available", model=model_name, is_mock=True)

        self.call_log.append({"method": "is_model_available", "model_name": model_name})
        available = model_name in self.available_models

        log_decision_point("mock_loader", "is_model_available", f"model={model_name}", f"mock_available={available}")
        log_component_end("mock_loader", "is_model_available", f"Mock model {model_name}: {available}")
        return available


class MockEmbeddingModel:
    """
    Mock embedding model for testing.
    """

    def __init__(self, model_name: str = "mock-model", device: str = "cpu"):
        self.model_name = model_name
        self._device = device
        self._max_seq_length = 512
        self._embedding_dim = 1024
        self.call_log: list[dict[str, Any]] = []

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True, **kwargs):
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
