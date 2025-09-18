"""
Multilingual embedding system for RAG applications.
Implements dependency injection and pure functions for robust embedding generation.
"""

import logging
from dataclasses import dataclass
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
    input_texts: list[str]
    model_name: str
    embedding_dim: int
    processing_time: float
    metadata: dict[str, Any]


@dataclass
class DeviceInfo:
    """Device information for embedding computation."""

    device_type: str  # "cuda", "mps", "cpu"
    device_name: str
    available_memory: int | None = None
    device_properties: dict[str, Any] | None = None


# Protocols for dependency injection
class EmbeddingModel(Protocol):
    """Embedding model interface for dependency injection."""

    def encode(self, texts: list[str], batch_size: int = 32, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
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
def validate_texts_for_embedding(texts: list[str]) -> list[str]:
    """
    Validate and clean texts for embedding generation.

    Args:
        texts: Input texts to validate

    Returns:
        Cleaned and validated texts

    Raises:
        ValueError: If texts are invalid
    """
    logger = get_system_logger()
    log_component_start(
        "embedding_validator", "validate_texts", input_count=len(texts) if isinstance(texts, list) else 0
    )

    if not texts:
        logger.error("embedding_validator", "validate_texts", "Cannot generate embeddings for empty text list")
        raise ValueError("Cannot generate embeddings for empty text list")

    if not isinstance(texts, list):
        logger.error("embedding_validator", "validate_texts", f"Texts must be provided as a list, got {type(texts)}")
        raise ValueError("Texts must be provided as a list")

    cleaned_texts = []
    for i, text in enumerate(texts):
        if text is None:
            logger.error("embedding_validator", "validate_texts", f"Text at index {i} is None")
            raise ValueError(f"Text at index {i} is None")

        if not isinstance(text, str):
            logger.error("embedding_validator", "validate_texts", f"Text at index {i} is not a string: {type(text)}")
            raise ValueError(f"Text at index {i} is not a string: {type(text)}")

        cleaned_text = text.strip()
        if not cleaned_text:
            logger.error("embedding_validator", "validate_texts", f"Text at index {i} is empty after cleaning")
            raise ValueError(f"Text at index {i} is empty after cleaning")

        cleaned_texts.append(cleaned_text)
        logger.trace("embedding_validator", "validate_texts", f"Validated text {i}: {len(cleaned_text)} chars")

    log_data_transformation(
        "embedding_validator", "clean_texts", f"raw[{len(texts)}]", f"cleaned[{len(cleaned_texts)}]"
    )
    log_component_end("embedding_validator", "validate_texts", f"Validated {len(cleaned_texts)} texts")
    return cleaned_texts


def calculate_optimal_batch_size(
    num_texts: int, available_memory: int | None, base_batch_size: int = 32, max_batch_size: int = 256
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
    logger = get_system_logger()
    log_component_start(
        "batch_optimizer",
        "calculate_optimal_batch_size",
        num_texts=num_texts,
        available_memory=available_memory,
        base_batch_size=base_batch_size,
    )

    if num_texts <= 0:
        logger.debug(
            "batch_optimizer",
            "calculate_optimal_batch_size",
            f"No texts to process, returning base batch size: {base_batch_size}",
        )
        return base_batch_size

    optimal_batch = base_batch_size
    logger.debug("batch_optimizer", "calculate_optimal_batch_size", f"Starting with base batch size: {optimal_batch}")

    if available_memory is not None and available_memory > 0:
        memory_based_batch = min(int(available_memory / 16), max_batch_size)
        optimal_batch = min(optimal_batch, memory_based_batch)
        log_decision_point(
            "batch_optimizer",
            "calculate_optimal_batch_size",
            f"memory={available_memory}MB",
            f"adjusted to {optimal_batch}",
        )

    optimal_batch = min(optimal_batch, num_texts)
    log_decision_point(
        "batch_optimizer", "calculate_optimal_batch_size", f"num_texts={num_texts}", f"final batch size {optimal_batch}"
    )

    optimal_batch = max(1, optimal_batch)
    log_component_end("batch_optimizer", "calculate_optimal_batch_size", f"Optimal batch size: {optimal_batch}")
    return optimal_batch


def split_texts_into_batches(texts: list[str], batch_size: int) -> list[list[str]]:
    """
    Split texts into batches for processing.

    Args:
        texts: Texts to split
        batch_size: Size of each batch

    Returns:
        List of text batches
    """
    logger = get_system_logger()
    log_component_start("batch_splitter", "split_texts_into_batches", total_texts=len(texts), batch_size=batch_size)

    if batch_size <= 0:
        logger.error("batch_splitter", "split_texts_into_batches", f"Invalid batch size: {batch_size}")
        raise ValueError("Batch size must be positive")

    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batches.append(batch)
        logger.trace("batch_splitter", "split_texts_into_batches", f"Created batch {len(batches)}: {len(batch)} texts")

    log_data_transformation("batch_splitter", "split_texts", f"texts[{len(texts)}]", f"batches[{len(batches)}]")
    log_component_end("batch_splitter", "split_texts_into_batches", f"Created {len(batches)} batches")
    return batches


def normalize_embeddings_array(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length.

    Args:
        embeddings: Raw embeddings array [num_texts, embedding_dim]

    Returns:
        Normalized embeddings array
    """
    logger = get_system_logger()
    log_component_start(
        "embedding_normalizer",
        "normalize_embeddings",
        embedding_shape=embeddings.shape if embeddings.size > 0 else "empty",
    )

    if embeddings.size == 0:
        logger.debug("embedding_normalizer", "normalize_embeddings", "Empty embeddings array, returning as-is")
        return embeddings

    # L2 normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    logger.debug(
        "embedding_normalizer",
        "normalize_embeddings",
        f"Calculated L2 norms: mean={np.mean(norms):.6f}, std={np.std(norms):.6f}",
    )

    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    normalized = embeddings / norms

    log_performance_metric(
        "embedding_normalizer", "normalize_embeddings", "norm_stability", f"min_norm={np.min(norms):.6f}"
    )
    log_component_end("embedding_normalizer", "normalize_embeddings", f"Normalized {embeddings.shape[0]} embeddings")
    return normalized


def combine_batch_embeddings(batch_embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Combine embeddings from multiple batches.

    Args:
        batch_embeddings: List of embedding arrays from batches

    Returns:
        Combined embeddings array
    """
    logger = get_system_logger()
    log_component_start("embedding_combiner", "combine_batch_embeddings", num_batches=len(batch_embeddings))

    if not batch_embeddings:
        logger.debug("embedding_combiner", "combine_batch_embeddings", "No batch embeddings to combine")
        return np.array([])

    batch_shapes = [batch.shape for batch in batch_embeddings]
    total_embeddings = sum(shape[0] for shape in batch_shapes)
    embedding_dim = batch_shapes[0][1] if batch_shapes else 0

    logger.debug(
        "embedding_combiner",
        "combine_batch_embeddings",
        f"Combining {len(batch_embeddings)} batches into {total_embeddings}x{embedding_dim}",
    )

    combined = np.vstack(batch_embeddings)

    log_data_transformation(
        "embedding_combiner", "vstack_batches", f"batches[{len(batch_embeddings)}]", f"combined[{combined.shape}]"
    )
    log_component_end("embedding_combiner", "combine_batch_embeddings", f"Combined to shape {combined.shape}")
    return combined


def validate_embedding_dimensions(
    embeddings: np.ndarray, expected_dim: int | None = None, num_texts: int | None = None
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
    logger = get_system_logger()
    log_component_start(
        "embedding_validator",
        "validate_dimensions",
        shape=embeddings.shape,
        expected_dim=expected_dim,
        num_texts=num_texts,
    )

    if embeddings.size == 0:
        logger.error("embedding_validator", "validate_dimensions", "Embedding array is empty")
        raise ValueError("Embedding array is empty")

    if len(embeddings.shape) != 2:
        logger.error(
            "embedding_validator", "validate_dimensions", f"Invalid shape: {embeddings.shape}, expected 2D array"
        )
        raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")

    actual_count, actual_dim = embeddings.shape
    logger.debug(
        "embedding_validator",
        "validate_dimensions",
        f"Validating shape: {actual_count} embeddings x {actual_dim} dimensions",
    )

    if num_texts is not None and actual_count != num_texts:
        logger.error(
            "embedding_validator", "validate_dimensions", f"Count mismatch: expected {num_texts}, got {actual_count}"
        )
        raise ValueError(f"Expected {num_texts} embeddings, got {actual_count}")

    if expected_dim is not None and actual_dim != expected_dim:
        logger.error(
            "embedding_validator",
            "validate_dimensions",
            f"Dimension mismatch: expected {expected_dim}, got {actual_dim}",
        )
        raise ValueError(f"Expected embedding dimension {expected_dim}, got {actual_dim}")

    log_component_end(
        "embedding_validator", "validate_dimensions", f"Validation passed for {actual_count}x{actual_dim}"
    )


def calculate_embedding_statistics(embeddings: np.ndarray) -> dict[str, Any]:
    """
    Calculate statistics for embedding array.

    Args:
        embeddings: Embedding array to analyze

    Returns:
        Dictionary with embedding statistics
    """
    logger = get_system_logger()
    log_component_start(
        "embedding_analyzer", "calculate_statistics", shape=embeddings.shape if embeddings.size > 0 else "empty"
    )

    if embeddings.size == 0:
        logger.debug("embedding_analyzer", "calculate_statistics", "Empty embeddings, returning minimal stats")
        return {"empty": True}

    norms = np.linalg.norm(embeddings, axis=1)
    logger.trace("embedding_analyzer", "calculate_statistics", "Calculated L2 norms for statistics")

    stats = {
        "num_embeddings": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_value": float(np.min(embeddings)),
        "max_value": float(np.max(embeddings)),
        "mean_value": float(np.mean(embeddings)),
        "std_value": float(np.std(embeddings)),
    }

    logger.debug(
        "embedding_analyzer",
        "calculate_statistics",
        f"Stats: {stats['num_embeddings']}x{stats['embedding_dim']}, "
        f"norm={stats['mean_norm']:.4f}±{stats['std_norm']:.4f}",
    )
    log_component_end("embedding_analyzer", "calculate_statistics", "Statistics calculated")
    return stats


# Model selection is now configuration-driven via language-specific TOML files
# Croatian: Uses classla/bcms-bertic (Croatian-optimized ELECTRA model)
# English: Uses BAAI/bge-large-en-v1.5 (English-optimized model)
# Fallback: BAAI/bge-m3 (multilingual) configured in main config.toml


class MultilingualEmbeddingGenerator:
    """
    Multilingual embedding generator with dependency injection.

    All external dependencies (model loading, device detection) are injected,
    enabling modular and testable architecture.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        model_loader: ModelLoader | None = None,
        device_detector: DeviceDetector | None = None,
        logger: logging.Logger | None = None,
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
        # Dependency injection - explicit defaults for optional dependencies
        self.model_loader = model_loader if model_loader is not None else self._create_default_model_loader()
        self.device_detector = (
            device_detector if device_detector is not None else self._create_default_device_detector()
        )
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self._model: EmbeddingModel | None = None
        self._device_info: DeviceInfo | None = None
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
        logger = get_system_logger()
        log_component_start(
            "embedding_generator", "initialize", model=self.config.model_name, device=self.config.device
        )

        try:
            # Detect device
            logger.debug("embedding_generator", "initialize", f"Detecting device for: {self.config.device}")
            self._device_info = self.device_detector.detect_best_device(self.config.device)
            log_decision_point(
                "embedding_generator",
                "initialize",
                f"device_preference={self.config.device}",
                f"selected={self._device_info.device_type}",
            )

            logger.info("embedding_generator", "initialize", f"Device selected: {self._device_info.device_type}")
            if self._device_info.available_memory:
                log_performance_metric(
                    "embedding_generator", "initialize", "available_memory_mb", self._device_info.available_memory
                )

            # Load model
            logger.debug("embedding_generator", "initialize", f"Loading model: {self.config.model_name}")
            self._model = self.model_loader.load_model(
                model_name=self.config.model_name,
                cache_dir=self.config.cache_dir,
                device=self._device_info.device_type,
                max_seq_length=self.config.max_seq_length,
                use_safetensors=self.config.use_safetensors,
                trust_remote_code=self.config.trust_remote_code,
            )

            embedding_dim = self._model.get_sentence_embedding_dimension()
            log_performance_metric("embedding_generator", "initialize", "embedding_dimension", embedding_dim)

            self._is_initialized = True
            log_component_end(
                "embedding_generator",
                "initialize",
                f"Initialized {self.config.model_name} on {self._device_info.device_type}",
                model=self.config.model_name,
                device=self._device_info.device_type,
                dimension=embedding_dim,
            )

        except Exception as e:
            log_error_context(
                "embedding_generator",
                "initialize",
                e,
                {
                    "model_name": self.config.model_name,
                    "device_requested": self.config.device,
                    "cache_dir": self.config.cache_dir,
                },
            )
            raise RuntimeError(f"Embedding system initialization failed: {e}") from e

    def generate_embeddings(
        self, texts: list[str], normalize: bool | None = None, batch_size: int | None = None
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
        logger = get_system_logger()
        log_component_start(
            "embedding_generator",
            "generate_embeddings",
            input_count=len(texts) if isinstance(texts, list) else 0,
            normalize=normalize,
            batch_size=batch_size,
        )

        if not self._is_initialized:
            logger.error("embedding_generator", "generate_embeddings", "System not initialized")
            raise RuntimeError("Embedding system not initialized. Call initialize() first.")

        import time

        start_time = time.time()

        try:
            # Ensure model is initialized
            if self._model is None:
                logger.error("embedding_generator", "generate_embeddings", "Model not loaded")
                raise RuntimeError("Embedding model not initialized. Call initialize() first.")
            if self._device_info is None:
                logger.error("embedding_generator", "generate_embeddings", "Device info not available")
                raise RuntimeError("Device info not initialized. Call initialize() first.")

            # Validate and clean input texts
            cleaned_texts = validate_texts_for_embedding(texts)
            logger.debug("embedding_generator", "generate_embeddings", f"Validated {len(cleaned_texts)} input texts")

            # Determine processing parameters
            use_normalize = normalize if normalize is not None else self.config.normalize_embeddings
            use_batch_size = batch_size if batch_size is not None else self.config.batch_size
            log_decision_point(
                "embedding_generator",
                "generate_embeddings",
                f"normalize={use_normalize}, batch_size={use_batch_size}",
                "parameters_set",
            )

            # Calculate optimal batch size
            available_memory = None
            if self._device_info and self._device_info.available_memory:
                available_memory = self._device_info.available_memory

            optimal_batch_size = calculate_optimal_batch_size(len(cleaned_texts), available_memory, use_batch_size)
            log_performance_metric(
                "embedding_generator", "generate_embeddings", "optimal_batch_size", optimal_batch_size
            )

            # Generate embeddings in batches
            text_batches = split_texts_into_batches(cleaned_texts, optimal_batch_size)
            batch_embeddings = []
            logger.info(
                "embedding_generator",
                "generate_embeddings",
                f"Processing {len(cleaned_texts)} texts in {len(text_batches)} batches",
            )

            for batch_idx, batch in enumerate(text_batches):
                logger.trace(
                    "embedding_generator",
                    "generate_embeddings",
                    f"Processing batch {batch_idx + 1}/{len(text_batches)}: {len(batch)} texts",
                )

                batch_emb = self._model.encode(
                    batch,
                    batch_size=optimal_batch_size,
                    normalize_embeddings=False,  # We'll normalize at the end if needed
                )
                batch_embeddings.append(batch_emb)
                logger.trace(
                    "embedding_generator", "generate_embeddings", f"Batch {batch_idx + 1} generated: {batch_emb.shape}"
                )

            # Combine all batch embeddings
            all_embeddings = combine_batch_embeddings(batch_embeddings)
            logger.debug(
                "embedding_generator", "generate_embeddings", f"Combined embeddings shape: {all_embeddings.shape}"
            )

            # Normalize if requested
            if use_normalize:
                logger.debug("embedding_generator", "generate_embeddings", "Normalizing embeddings")
                all_embeddings = normalize_embeddings_array(all_embeddings)

            # Validate results
            embedding_dim = self._model.get_sentence_embedding_dimension()
            validate_embedding_dimensions(all_embeddings, expected_dim=embedding_dim, num_texts=len(cleaned_texts))

            # Calculate processing time and statistics
            processing_time = time.time() - start_time
            log_performance_metric("embedding_generator", "generate_embeddings", "processing_time_sec", processing_time)

            statistics = calculate_embedding_statistics(all_embeddings)
            logger.debug(
                "embedding_generator",
                "generate_embeddings",
                f"Generated embeddings: {statistics['num_embeddings']}x{statistics['embedding_dim']}",
            )

            # Create result
            metadata = {
                "processing_time": processing_time,
                "batch_size_used": optimal_batch_size,
                "num_batches": len(text_batches),
                "normalized": use_normalize,
                "device": self._device_info.device_type,
                "statistics": statistics,
            }

            result = EmbeddingResult(
                embeddings=all_embeddings,
                input_texts=cleaned_texts,
                model_name=self.config.model_name,
                embedding_dim=embedding_dim,
                processing_time=processing_time,
                metadata=metadata,
            )

            log_component_end(
                "embedding_generator",
                "generate_embeddings",
                f"Generated {len(cleaned_texts)} embeddings in {processing_time:.3f}s",
                texts_count=len(cleaned_texts),
                embedding_dim=embedding_dim,
                processing_time=processing_time,
            )
            return result

        except Exception as e:
            log_error_context(
                "embedding_generator",
                "generate_embeddings",
                e,
                {
                    "input_count": len(texts) if isinstance(texts, list) else 0,
                    "model_name": self.config.model_name,
                    "device": self._device_info.device_type if self._device_info else "unknown",
                    "normalize": normalize,
                    "batch_size": batch_size,
                },
            )
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the loaded model.

        Returns:
            Embedding dimension

        Raises:
            RuntimeError: If system not initialized
        """
        logger = get_system_logger()
        logger.trace("embedding_generator", "get_embedding_dimension", "Retrieving model dimension")

        if not self._is_initialized or not self._model:
            logger.error("embedding_generator", "get_embedding_dimension", "System not initialized")
            raise RuntimeError("Embedding system not initialized")

        dimension = self._model.get_sentence_embedding_dimension()
        logger.debug("embedding_generator", "get_embedding_dimension", f"Model dimension: {dimension}")
        return dimension

    def is_model_available(self, model_name: str | None = None) -> bool:
        """
        Check if a model is available for loading.

        Args:
            model_name: Model name to check (defaults to configured model)

        Returns:
            True if model is available
        """
        get_system_logger()
        check_model = model_name or self.config.model_name
        log_component_start("embedding_generator", "is_model_available", model=check_model)

        available = self.model_loader.is_model_available(check_model)
        log_decision_point(
            "embedding_generator", "is_model_available", f"model={check_model}", f"available={available}"
        )
        log_component_end("embedding_generator", "is_model_available", f"Model {check_model}: {available}")
        return available

    def get_device_info(self) -> DeviceInfo | None:
        """
        Get information about the current device.

        Returns:
            Device information if initialized
        """
        return self._device_info

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary

        Raises:
            RuntimeError: If system not initialized
        """
        logger = get_system_logger()
        log_component_start("embedding_generator", "get_model_info", initialized=self._is_initialized)

        if not self._is_initialized:
            logger.error("embedding_generator", "get_model_info", "System not initialized")
            raise RuntimeError("Embedding system not initialized")

        info = {
            "model_name": self.config.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_seq_length": self.config.max_seq_length,
            "device": self._device_info.device_type if self._device_info else "unknown",
            "normalized_by_default": self.config.normalize_embeddings,
        }

        logger.debug(
            "embedding_generator",
            "get_model_info",
            f"Model: {info['model_name']}, dim: {info['embedding_dimension']}, device: {info['device']}",
        )
        log_component_end("embedding_generator", "get_model_info", "Model info retrieved")
        return info


# Factory function for convenient creation
def create_embedding_generator(
    config: EmbeddingConfig | None = None,
    model_loader: ModelLoader | None = None,
    device_detector: DeviceDetector | None = None,
    logger: logging.Logger | None = None,
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
    system_logger = get_system_logger()
    log_component_start(
        "embedding_factory",
        "create_generator",
        has_config=config is not None,
        has_loader=model_loader is not None,
        has_detector=device_detector is not None,
    )

    if config is None:
        system_logger.debug("embedding_factory", "create_generator", "Creating default configuration")
        config = EmbeddingConfig()

    system_logger.debug("embedding_factory", "create_generator", f"Creating generator with model: {config.model_name}")

    generator = MultilingualEmbeddingGenerator(
        config=config, model_loader=model_loader, device_detector=device_detector, logger=logger
    )

    log_component_end("embedding_factory", "create_generator", f"Created generator for {config.model_name}")
    return generator
