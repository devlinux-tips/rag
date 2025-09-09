"""
Embedding model management for multilingual RAG system.
Handles multilingual sentence-transformers models optimized for various languages.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ..utils.config_protocol import ConfigProvider

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

from ..utils.config_loader import get_embeddings_config, get_vectordb_config


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model_name: str
    cache_dir: str
    device: str
    max_seq_length: int
    batch_size: int
    normalize_embeddings: bool
    use_safetensors: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "auto"

    @classmethod
    def from_config(
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "EmbeddingConfig":
        """Load configuration from dictionary or config provider."""
        if config_dict:
            config = config_dict
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get embeddings config through provider
            full_config = config_provider.load_config("config")
            config = full_config["embeddings"]

        return cls(
            model_name=config["model_name"],
            cache_dir=config["cache_dir"],
            device=config["device"],
            max_seq_length=config["max_seq_length"],
            batch_size=config["batch_size"],
            normalize_embeddings=config["normalize_embeddings"],
            use_safetensors=config.get("use_safetensors", True),
            trust_remote_code=config.get("trust_remote_code", False),
            torch_dtype=config.get("torch_dtype", "auto"),
        )


class MultilingualEmbeddingModel:
    """Multilingual embedding model optimized for various languages."""

    @property
    def recommended_models(self) -> Dict[str, str]:
        """Get recommended models from config."""
        config = get_embeddings_config()
        return config.get(
            "recommended_models",
            {
                "bge_m3": "BAAI/bge-m3",
                "labse": "sentence-transformers/LaBSE",
                "multilingual_minilm": "paraphrase-multilingual-MiniLM-L12-v2",
                "multilingual_mpnet": "paraphrase-multilingual-mpnet-base-v2",
                "distiluse_multilingual": "distiluse-base-multilingual-cased",
            },
        )

    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize embedding model.

        Args:
            config: Configuration for embedding model
        """
        self.config = config or EmbeddingConfig.from_config()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._model_loaded = False

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Set device
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device with detailed logging."""
        if self.config.device == "auto":
            # Auto-detect best available device
            # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                self.logger.info(
                    "Apple Silicon Metal Performance Shaders (MPS) detected"
                )

                # Try to get Apple Silicon chip info
                try:
                    import platform

                    machine = platform.machine()
                    if machine == "arm64":
                        # Running on Apple Silicon
                        self.logger.info(f"Apple Silicon detected: {machine}")
                        if (
                            "M4" in platform.processor()
                            or torch.backends.mps.is_built()
                        ):
                            self.logger.info("Optimized for M4 Pro/Max performance")
                    self.logger.info(f"Using device: {device}")
                except:
                    self.logger.info(f"Using device: {device}")

                return device

            # Check CUDA with graceful error handling
            cuda_available = False
            try:
                # Suppress CUDA initialization warnings temporarily
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
                    cuda_available = torch.cuda.is_available()
            except Exception as e:
                self.logger.debug(f"CUDA check failed: {e}")
                cuda_available = False

            if cuda_available:
                try:
                    device = "cuda"
                    # Log CUDA info
                    cuda_count = torch.cuda.device_count()
                    current_device = torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(current_device)
                    memory_info = torch.cuda.get_device_properties(current_device)
                    total_memory = memory_info.total_memory / 1024**3  # GB

                    self.logger.info(f"CUDA detected: {device_name}")
                    self.logger.info(f"CUDA memory: {total_memory:.1f}GB total")
                    self.logger.info(f"Using device: cuda (device {current_device})")

                    return device
                except Exception as e:
                    self.logger.warning(f"CUDA detected but initialization failed: {e}")
                    self.logger.info(
                        "Falling back to CPU (restart terminal to fix CUDA)"
                    )
                    device = "cpu"
            else:
                device = "cpu"
                self.logger.info("No GPU acceleration available")
                self.logger.info(f"Using device: {device}")
                return device
                self.logger.info(f"Using device: {device}")
                return device
        else:
            # Use specified device
            device = self.config.device

            # Validate specified device
            if device == "mps":
                if not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    self.logger.warning(
                        "MPS requested but not available, falling back to CPU"
                    )
                    return "cpu"
                self.logger.info(f"Using specified device: {device}")

            elif device.startswith("cuda"):
                if not torch.cuda.is_available():
                    self.logger.warning(
                        f"CUDA requested but not available, falling back to CPU"
                    )
                    return "cpu"

                # Check specific CUDA device if specified (e.g., "cuda:1")
                if ":" in device:
                    device_id = int(device.split(":")[1])
                    if device_id >= torch.cuda.device_count():
                        self.logger.warning(
                            f"CUDA device {device_id} not available, using cuda:0"
                        )
                        device = "cuda:0"

                self.logger.info(f"Using specified device: {device}")

            elif device == "cpu":
                self.logger.info(f"Using specified device: {device}")

            else:
                self.logger.warning(f"Unknown device '{device}', falling back to CPU")
                return "cpu"

            return device

    def load_model(self) -> None:
        """Load the sentence transformer model with safetensors support."""
        if self._model_loaded:
            return

        try:
            self.logger.info(f"Loading embedding model: {self.config.model_name}")

            # Get safetensors preference from config
            embeddings_config = get_embeddings_config()
            use_safetensors = embeddings_config["use_safetensors"]
            trust_remote_code = embeddings_config["trust_remote_code"]
            torch_dtype = embeddings_config["torch_dtype"]

            # Prepare model loading kwargs
            model_kwargs = {
                "cache_folder": self.config.cache_dir,
                "device": self.device,
                "trust_remote_code": trust_remote_code,
            }

            # Add torch_dtype if specified
            if torch_dtype != "auto":
                model_kwargs["torch_dtype"] = getattr(torch, torch_dtype, None)

            # For modern multilingual models, prefer safetensors
            if use_safetensors:
                try:
                    # SentenceTransformer doesn't directly support use_safetensors parameter
                    # but will use safetensors automatically if available and secure
                    self.model = SentenceTransformer(
                        self.config.model_name, **model_kwargs
                    )
                    self.logger.info("Model loaded (safetensors preferred)")
                except Exception as safetensors_error:
                    self.logger.warning(
                        f"Model loading with safetensors preference failed: {safetensors_error}"
                    )
                    # Fallback to regular loading
                    self.model = SentenceTransformer(
                        self.config.model_name, **model_kwargs
                    )
                    self.logger.info("Model loaded with standard format")
            else:
                # Standard loading
                self.model = SentenceTransformer(self.config.model_name, **model_kwargs)

            # Configure model settings
            self.model.max_seq_length = self.config.max_seq_length

            self._model_loaded = True
            self.logger.info("Embedding model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            # Check if it's the PyTorch security error
            if "torch.load" in str(e) and "vulnerability" in str(e).lower():
                self.logger.error("PyTorch security vulnerability detected!")
                self.logger.error("Solutions:")
                self.logger.error("1. Upgrade PyTorch: pip install torch>=2.6.0")
                self.logger.error(
                    "2. Install safetensors: pip install safetensors>=0.4.0"
                )
                self.logger.error("3. Set use_safetensors=true in config.toml")
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
        top_k: int = None,
        metric: str = None,
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
        # Use config defaults if not provided
        if top_k is None:
            embeddings_config = get_embeddings_config()
            top_k = embeddings_config["similarity"]["top_k_default"]

        if metric is None:
            embeddings_config = get_embeddings_config()
            metric = embeddings_config["similarity"]["default_metric"]

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

        info = {
            "model_name": self.config.model_name,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "device": str(self.device),
            "normalize_embeddings": self.config.normalize_embeddings,
            "pooling_mode": str(self.model._modules["1"]),
            "pytorch_version": torch.__version__,
        }

        # Add device-specific information
        if self.device.startswith("cuda"):
            if torch.cuda.is_available():
                device_id = 0
                if ":" in self.device:
                    device_id = int(self.device.split(":")[1])

                info.update(
                    {
                        "cuda_device_name": torch.cuda.get_device_name(device_id),
                        "cuda_memory_total": f"{torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f}GB",
                        "cuda_memory_allocated": f"{torch.cuda.memory_allocated(device_id) / 1024**3:.1f}GB",
                        "cuda_memory_cached": f"{torch.cuda.memory_reserved(device_id) / 1024**3:.1f}GB",
                        "cuda_version": torch.version.cuda,
                    }
                )
        elif self.device == "mps":
            info["mps_available"] = torch.backends.mps.is_available()
            if torch.backends.mps.is_available():
                info["mps_built"] = torch.backends.mps.is_built()

                # Apple Silicon specific info
                try:
                    import platform

                    import psutil

                    machine = platform.machine()
                    if machine == "arm64":
                        info["apple_silicon"] = True
                        info["platform_machine"] = machine

                        # Get system memory (unified memory on Apple Silicon)
                        memory = psutil.virtual_memory()
                        info["unified_memory_total"] = f"{memory.total / 1024**3:.1f}GB"
                        info[
                            "unified_memory_available"
                        ] = f"{memory.available / 1024**3:.1f}GB"

                        # Try to detect specific M-series chip
                        try:
                            processor = platform.processor()
                            if "M4" in processor:
                                info["apple_chip"] = "M4 Pro/Max/Ultra"
                                info["performance_cores"] = "10-16"  # M4 Pro/Max range
                                info["gpu_cores"] = "16-40"  # M4 Pro/Max range
                            elif any(m in processor for m in ["M1", "M2", "M3"]):
                                info["apple_chip"] = processor
                            else:
                                info["apple_chip"] = "Apple Silicon (M-series)"
                        except:
                            info["apple_chip"] = "Apple Silicon"
                except ImportError:
                    info["apple_silicon"] = "unknown (install psutil for detailed info)"
                except:
                    info["apple_silicon"] = "unknown"

        return info

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed information about available devices.

        Returns:
            Dictionary with device capabilities
        """
        info = {
            "current_device": str(self.device),
            "pytorch_version": torch.__version__,
            "cpu_available": True,
        }

        # CUDA information with graceful error handling
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
                cuda_available = torch.cuda.is_available()

            info["cuda_available"] = cuda_available

            if cuda_available:
                try:
                    info["cuda_version"] = torch.version.cuda
                    info["cuda_device_count"] = torch.cuda.device_count()
                    info["cuda_devices"] = []

                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        device_info = {
                            "id": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_total": f"{props.total_memory / 1024**3:.1f}GB",
                            "memory_allocated": f"{torch.cuda.memory_allocated(i) / 1024**3:.1f}GB",
                            "memory_cached": f"{torch.cuda.memory_reserved(i) / 1024**3:.1f}GB",
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                        info["cuda_devices"].append(device_info)
                except Exception as e:
                    info["cuda_error"] = f"CUDA detected but initialization failed: {e}"
                    info[
                        "cuda_solution"
                    ] = "Restart terminal to fix CUDA initialization"
        except Exception as e:
            info["cuda_available"] = False
            info["cuda_error"] = str(e)

        # MPS information (Apple Silicon)
        if hasattr(torch.backends, "mps"):
            info["mps_available"] = torch.backends.mps.is_available()
            if torch.backends.mps.is_available():
                info["mps_built"] = torch.backends.mps.is_built()

                # Try to detect Apple Silicon chip info
                try:
                    import platform

                    machine = platform.machine()
                    if machine == "arm64":
                        info["apple_silicon"] = True
                        info["platform_machine"] = machine

                        # Try to detect M-series chip
                        try:
                            processor = platform.processor()
                            if any(m in processor for m in ["M1", "M2", "M3", "M4"]):
                                info["apple_chip"] = processor
                            else:
                                info["apple_chip"] = "Apple Silicon (M-series)"
                        except:
                            info["apple_chip"] = "Apple Silicon"
                    else:
                        info["apple_silicon"] = False
                except:
                    info["apple_silicon"] = "unknown"
        else:
            info["mps_available"] = False

        return info

    def switch_device(self, new_device: str) -> None:
        """
        Switch the model to a different device.

        Args:
            new_device: Target device ("cpu", "cuda", "cuda:0", etc.)
        """
        if not self._model_loaded:
            self.logger.warning("Model not loaded yet, updating device configuration")
            self.config.device = new_device
            self.device = self._get_device()
            return

        old_device = self.device
        self.config.device = new_device
        self.device = self._get_device()

        if old_device != self.device:
            self.logger.info(f"Switching model from {old_device} to {self.device}")
            try:
                # Move model to new device
                self.model = self.model.to(self.device)
                self.logger.info(f"Model successfully moved to {self.device}")
            except Exception as e:
                self.logger.error(f"Failed to move model to {self.device}: {e}")
                # Revert to old device
                self.config.device = old_device
                self.device = old_device
                raise
        else:
            self.logger.info(f"Device unchanged: {self.device}")

    def save_embeddings(
        self, embeddings: np.ndarray, metadata: List[Dict], filepath: str
    ) -> None:
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

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            vectordb_config = get_vectordb_config()
            cache_dir = vectordb_config["factory"]["cache_embeddings_dir"]

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
    model_name: str = None,
    device: str = None,
    cache_dir: str = None,
) -> MultilingualEmbeddingModel:
    """
    Factory function to create embedding model.

    Args:
        model_name: Sentence transformer model name (defaults from config)
        device: Device to use (defaults from config)
        cache_dir: Directory for model cache (defaults from config)

    Returns:
        Configured MultilingualEmbeddingModel instance
    """
    # Use config defaults if not provided
    vectordb_config = get_vectordb_config()
    factory_config = vectordb_config["factory"]

    model_name = model_name or factory_config["default_model"]
    device = device or factory_config["default_device"]
    cache_dir = cache_dir or factory_config["default_cache_dir"]

    config = EmbeddingConfig(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        max_seq_length=512,
        batch_size=32,
        normalize_embeddings=True,
    )
    return MultilingualEmbeddingModel(config)


def get_recommended_model(use_case: str = "general") -> str:
    """
    Get recommended model for specific use case.

    Args:
        use_case: Use case type ("general", "fast", "accurate", "cross-lingual")

    Returns:
        Recommended model name
    """
    embeddings_config = get_embeddings_config()
    models = embeddings_config["recommended_models"]

    recommendations = {
        "general": models["bge_m3"],  # Primary multilingual model
        "fast": models.get(
            "multilingual_minilm", "paraphrase-multilingual-MiniLM-L12-v2"
        ),  # Fastest option
        "accurate": models["bge_m3"],  # Most accurate multilingual model
        "cross-lingual": models.get(
            "bge_m3", "BAAI/bge-m3"
        ),  # Excellent multilingual performance
    }

    return recommendations.get(use_case, models["bge_m3"])
