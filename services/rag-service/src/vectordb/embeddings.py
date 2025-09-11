"""
Multilingual embedding system for RAG applications.
Implements dependency injection and pure functions for robust embedding generation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models - pure data structure."""

    model_name: str = "BAAI/bge-m3"
    cache_dir: str = "models/embeddings"
    device: str = "auto"
    max_seq_length: int = 8192
    batch_size: int = 32
    normalize_embeddings: bool = True
    use_safetensors: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "auto"


@dataclass
class EmbeddingResult:
    """Result from embedding generation - pure data structure."""

    embeddings: np.ndarray
    input_texts: List[str]
    model_name: str
    embedding_dim: int
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class DeviceInfo:
    """Device information for embedding computation."""

    device_type: str  # "cuda", "mps", "cpu"
    device_name: str
    available_memory: Optional[int] = None
    device_properties: Optional[Dict[str, Any]] = None


# Protocols for dependency injection
class EmbeddingModel(Protocol):
    """Embedding model interface for dependency injection."""

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Generate embeddings for texts."""
        ...

    @property
    def device(self) -> str:
        """Get current device."""
        ...

    @property
    def max_seq_length(self) -> int:
        """Get maximum sequence length."""
        ...

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        ...


class ModelLoader(Protocol):
    """Model loader interface for dependency injection."""

    def load_model(self, model_name: str, cache_dir: str, device: str, **kwargs) -> EmbeddingModel:
        """Load embedding model."""
        ...

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available."""
        ...


class DeviceDetector(Protocol):
    """Device detection interface for dependency injection."""

    def detect_best_device(self, preferred_device: str = "auto") -> DeviceInfo:
        """Detect best available device."""
        ...

    def is_device_available(self, device: str) -> bool:
        """Check if device is available."""
        ...


# Pure functions for business logic
def validate_texts_for_embedding(texts: List[str]) -> List[str]:
    """
    Validate and clean texts for embedding generation.

    Args:
        texts: Input texts to validate

    Returns:
        Cleaned and validated texts

    Raises:
        ValueError: If texts are invalid
    """
    if not texts:
        raise ValueError("Cannot generate embeddings for empty text list")

    if not isinstance(texts, list):
        raise ValueError("Texts must be provided as a list")

    cleaned_texts = []
    for i, text in enumerate(texts):
        if text is None:
            raise ValueError(f"Text at index {i} is None")

        if not isinstance(text, str):
            raise ValueError(f"Text at index {i} is not a string: {type(text)}")

        # Clean whitespace but preserve content
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError(f"Text at index {i} is empty after cleaning")

        cleaned_texts.append(cleaned_text)

    return cleaned_texts


def calculate_optimal_batch_size(
    num_texts: int,
    available_memory: Optional[int],
    base_batch_size: int = 32,
    max_batch_size: int = 256,
) -> int:
    """
    Calculate optimal batch size for embedding generation.

    Args:
        num_texts: Number of texts to process
        available_memory: Available memory in MB
        base_batch_size: Base batch size to use
        max_batch_size: Maximum allowed batch size

    Returns:
        Optimal batch size
    """
    if num_texts <= 0:
        return base_batch_size

    # Start with base batch size
    optimal_batch = base_batch_size

    # Adjust based on available memory if known
    if available_memory is not None and available_memory > 0:
        # Rough heuristic: 1GB allows ~64 batch size for typical models
        memory_based_batch = min(int(available_memory / 16), max_batch_size)
        optimal_batch = min(optimal_batch, memory_based_batch)

    # Don't use batch larger than total texts
    optimal_batch = min(optimal_batch, num_texts)

    # Ensure minimum batch size of 1
    return max(1, optimal_batch)


def split_texts_into_batches(texts: List[str], batch_size: int) -> List[List[str]]:
    """
    Split texts into batches for processing.

    Args:
        texts: Texts to split
        batch_size: Size of each batch

    Returns:
        List of text batches
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batches.append(batch)

    return batches


def normalize_embeddings_array(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length.

    Args:
        embeddings: Raw embeddings array [num_texts, embedding_dim]

    Returns:
        Normalized embeddings array
    """
    if embeddings.size == 0:
        return embeddings

    # L2 normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    return embeddings / norms


def combine_batch_embeddings(batch_embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Combine embeddings from multiple batches.

    Args:
        batch_embeddings: List of embedding arrays from batches

    Returns:
        Combined embeddings array
    """
    if not batch_embeddings:
        return np.array([])

    return np.vstack(batch_embeddings)


def validate_embedding_dimensions(
    embeddings: np.ndarray,
    expected_dim: Optional[int] = None,
    num_texts: Optional[int] = None,
) -> None:
    """
    Validate embedding array dimensions.

    Args:
        embeddings: Embedding array to validate
        expected_dim: Expected embedding dimension
        num_texts: Expected number of texts

    Raises:
        ValueError: If dimensions are invalid
    """
    if embeddings.size == 0:
        raise ValueError("Embedding array is empty")

    if len(embeddings.shape) != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")

    if num_texts is not None and embeddings.shape[0] != num_texts:
        raise ValueError(f"Expected {num_texts} embeddings, got {embeddings.shape[0]}")

    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValueError(f"Expected embedding dimension {expected_dim}, got {embeddings.shape[1]}")


def calculate_embedding_statistics(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for embedding array.

    Args:
        embeddings: Embedding array to analyze

    Returns:
        Dictionary with embedding statistics
    """
    if embeddings.size == 0:
        return {"empty": True}

    return {
        "num_embeddings": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "min_value": float(np.min(embeddings)),
        "max_value": float(np.max(embeddings)),
        "mean_value": float(np.mean(embeddings)),
        "std_value": float(np.std(embeddings)),
    }


def get_recommended_models() -> Dict[str, str]:
    """
    Get dictionary of recommended embedding models.

    Returns:
        Dictionary mapping model keys to model names
    """
    return {
        "bge_m3": "BAAI/bge-m3",
        "bge_large": "BAAI/bge-large-en-v1.5",
        "labse": "sentence-transformers/LaBSE",
        "multilingual_minilm": "paraphrase-multilingual-MiniLM-L12-v2",
        "multilingual_mpnet": "paraphrase-multilingual-mpnet-base-v2",
        "distiluse_multilingual": "distiluse-base-multilingual-cased",
        "croatian_electra": "classla/bcms-bertic",
    }


def choose_model_for_language(language: str) -> str:
    """
    Choose optimal model for specific language.

    Args:
        language: Language code (e.g., 'hr', 'en', 'multilingual')

    Returns:
        Recommended model name
    """
    models = get_recommended_models()

    if language.lower() in ["hr", "croatian", "serbian", "bosnian"]:
        # For Croatian and related languages, prefer BGE-M3 or specific BCMS models
        return models["bge_m3"]  # Best multilingual performance for Croatian
    elif language.lower() in ["en", "english"]:
        # For English, BGE-large or BGE-M3 work well
        return models["bge_m3"]  # Consistent choice for multilingual systems
    else:
        # For other languages or multilingual, use BGE-M3
        return models["bge_m3"]


class MultilingualEmbeddingGenerator:
    """
    Multilingual embedding generator with dependency injection.

    All external dependencies (model loading, device detection) are injected,
    enabling modular and testable architecture.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        model_loader: Optional[ModelLoader] = None,
        device_detector: Optional[DeviceDetector] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize embedding generator with injected dependencies.

        Args:
            config: Embedding configuration
            model_loader: Model loading implementation
            device_detector: Device detection implementation
            logger: Logger instance
        """
        self.config = config
        self.model_loader = model_loader or self._create_default_model_loader()
        self.device_detector = device_detector or self._create_default_device_detector()
        self.logger = logger or logging.getLogger(__name__)

        self._model: Optional[EmbeddingModel] = None
        self._device_info: Optional[DeviceInfo] = None
        self._is_initialized = False

    def _create_default_model_loader(self) -> ModelLoader:
        """Create default model loader implementation."""
        from .embedding_loaders import SentenceTransformerLoader

        return SentenceTransformerLoader()

    def _create_default_device_detector(self) -> DeviceDetector:
        """Create default device detector implementation."""
        from .embedding_devices import TorchDeviceDetector

        return TorchDeviceDetector()

    def initialize(self) -> None:
        """
        Initialize the embedding system.

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Detect device
            self._device_info = self.device_detector.detect_best_device(self.config.device)
            self.logger.info(f"Using device: {self._device_info.device_type}")

            # Load model
            self._model = self.model_loader.load_model(
                model_name=self.config.model_name,
                cache_dir=self.config.cache_dir,
                device=self._device_info.device_type,
                max_seq_length=self.config.max_seq_length,
                use_safetensors=self.config.use_safetensors,
                trust_remote_code=self.config.trust_remote_code,
            )

            self._is_initialized = True
            self.logger.info(f"Embedding system initialized with model: {self.config.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize embedding system: {e}")
            raise RuntimeError(f"Embedding system initialization failed: {e}")

    def generate_embeddings(
        self,
        texts: List[str],
        normalize: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for input texts.

        Args:
            texts: Input texts to embed
            normalize: Whether to normalize embeddings (overrides config)
            batch_size: Batch size to use (overrides config)

        Returns:
            Embedding result with embeddings and metadata

        Raises:
            RuntimeError: If system not initialized or generation fails
            ValueError: If input texts are invalid
        """
        if not self._is_initialized:
            raise RuntimeError("Embedding system not initialized. Call initialize() first.")

        import time

        start_time = time.time()

        try:
            # Validate and clean input texts
            cleaned_texts = validate_texts_for_embedding(texts)

            # Determine processing parameters
            use_normalize = normalize if normalize is not None else self.config.normalize_embeddings
            use_batch_size = batch_size if batch_size is not None else self.config.batch_size

            # Calculate optimal batch size
            available_memory = None
            if self._device_info and self._device_info.available_memory:
                available_memory = self._device_info.available_memory

            optimal_batch_size = calculate_optimal_batch_size(
                len(cleaned_texts), available_memory, use_batch_size
            )

            self.logger.info(
                f"Processing {len(cleaned_texts)} texts with batch size {optimal_batch_size}"
            )

            # Generate embeddings in batches
            text_batches = split_texts_into_batches(cleaned_texts, optimal_batch_size)
            batch_embeddings = []

            for batch in text_batches:
                batch_emb = self._model.encode(
                    batch,
                    batch_size=optimal_batch_size,
                    normalize_embeddings=False,  # We'll normalize at the end if needed
                )
                batch_embeddings.append(batch_emb)

            # Combine all batch embeddings
            all_embeddings = combine_batch_embeddings(batch_embeddings)

            # Normalize if requested
            if use_normalize:
                all_embeddings = normalize_embeddings_array(all_embeddings)

            # Validate results
            embedding_dim = self._model.get_sentence_embedding_dimension()
            validate_embedding_dimensions(
                all_embeddings, expected_dim=embedding_dim, num_texts=len(cleaned_texts)
            )

            # Calculate processing time and statistics
            processing_time = time.time() - start_time
            statistics = calculate_embedding_statistics(all_embeddings)

            # Create result
            metadata = {
                "processing_time": processing_time,
                "batch_size_used": optimal_batch_size,
                "num_batches": len(text_batches),
                "normalized": use_normalize,
                "device": self._device_info.device_type,
                "statistics": statistics,
            }

            return EmbeddingResult(
                embeddings=all_embeddings,
                input_texts=cleaned_texts,
                model_name=self.config.model_name,
                embedding_dim=embedding_dim,
                processing_time=processing_time,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the loaded model.

        Returns:
            Embedding dimension

        Raises:
            RuntimeError: If system not initialized
        """
        if not self._is_initialized or not self._model:
            raise RuntimeError("Embedding system not initialized")

        return self._model.get_sentence_embedding_dimension()

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """
        Check if a model is available for loading.

        Args:
            model_name: Model name to check (defaults to configured model)

        Returns:
            True if model is available
        """
        check_model = model_name or self.config.model_name
        return self.model_loader.is_model_available(check_model)

    def get_device_info(self) -> Optional[DeviceInfo]:
        """
        Get information about the current device.

        Returns:
            Device information if initialized
        """
        return self._device_info

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary

        Raises:
            RuntimeError: If system not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Embedding system not initialized")

        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_seq_length": self.config.max_seq_length,
            "device": self._device_info.device_type if self._device_info else "unknown",
            "normalized_by_default": self.config.normalize_embeddings,
        }


# Factory function for convenient creation
def create_embedding_generator(
    config: Optional[EmbeddingConfig] = None,
    model_loader: Optional[ModelLoader] = None,
    device_detector: Optional[DeviceDetector] = None,
    logger: Optional[logging.Logger] = None,
) -> MultilingualEmbeddingGenerator:
    """
    Factory function to create configured embedding generator.

    Args:
        config: Embedding configuration (creates default if None)
        model_loader: Model loader implementation
        device_detector: Device detector implementation
        logger: Logger instance

    Returns:
        Configured MultilingualEmbeddingGenerator
    """
    if config is None:
        config = EmbeddingConfig()

    return MultilingualEmbeddingGenerator(
        config=config,
        model_loader=model_loader,
        device_detector=device_detector,
        logger=logger,
    )


# Backward compatibility aliases
EmbeddingModel = MultilingualEmbeddingGenerator
