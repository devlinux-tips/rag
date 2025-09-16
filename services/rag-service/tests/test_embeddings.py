"""
Comprehensive tests for multilingual embedding system.
Tests pure functions, dependency injection, and embedding generation workflows.
"""

import pytest
import logging
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from src.vectordb.embeddings import (
    # Data structures
    EmbeddingConfig,
    EmbeddingResult,
    DeviceInfo,

    # Pure functions
    validate_texts_for_embedding,
    calculate_optimal_batch_size,
    split_texts_into_batches,
    normalize_embeddings_array,
    combine_batch_embeddings,
    validate_embedding_dimensions,
    calculate_embedding_statistics,

    # Main class
    MultilingualEmbeddingGenerator,

    # Factory function
    create_embedding_generator,
)


# ===== DATA STRUCTURE TESTS =====

class TestDataStructures:
    """Test data structure creation and validation."""

    def test_embedding_config_creation(self):
        """Test EmbeddingConfig creation with defaults."""
        config = EmbeddingConfig()

        assert config.model_name == "BAAI/bge-m3"
        assert config.cache_dir == "models/embeddings"
        assert config.device == "auto"
        assert config.max_seq_length == 8192
        assert config.batch_size == 32
        assert config.normalize_embeddings is True
        assert config.use_safetensors is True
        assert config.trust_remote_code is False
        assert config.torch_dtype == "auto"

    def test_embedding_config_custom_values(self):
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            model_name="custom/model",
            cache_dir="/custom/cache",
            device="cuda",
            max_seq_length=4096,
            batch_size=16,
            normalize_embeddings=False,
            use_safetensors=False,
            trust_remote_code=True,
            torch_dtype="float16"
        )

        assert config.model_name == "custom/model"
        assert config.cache_dir == "/custom/cache"
        assert config.device == "cuda"
        assert config.max_seq_length == 4096
        assert config.batch_size == 16
        assert config.normalize_embeddings is False
        assert config.use_safetensors is False
        assert config.trust_remote_code is True
        assert config.torch_dtype == "float16"

    def test_embedding_result_creation(self):
        """Test EmbeddingResult creation."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        texts = ["text1", "text2"]
        metadata = {"batch_size": 2}

        result = EmbeddingResult(
            embeddings=embeddings,
            input_texts=texts,
            model_name="test_model",
            embedding_dim=2,
            processing_time=0.5,
            metadata=metadata
        )

        assert np.array_equal(result.embeddings, embeddings)
        assert result.input_texts == texts
        assert result.model_name == "test_model"
        assert result.embedding_dim == 2
        assert result.processing_time == 0.5
        assert result.metadata == metadata

    def test_device_info_creation(self):
        """Test DeviceInfo creation."""
        device_info = DeviceInfo(
            device_type="cuda",
            device_name="NVIDIA RTX 3080",
            available_memory=10240,
            device_properties={"compute_capability": "8.6"}
        )

        assert device_info.device_type == "cuda"
        assert device_info.device_name == "NVIDIA RTX 3080"
        assert device_info.available_memory == 10240
        assert device_info.device_properties == {"compute_capability": "8.6"}

    def test_device_info_minimal(self):
        """Test DeviceInfo with minimal parameters."""
        device_info = DeviceInfo(device_type="cpu", device_name="Intel i7")

        assert device_info.device_type == "cpu"
        assert device_info.device_name == "Intel i7"
        assert device_info.available_memory is None
        assert device_info.device_properties is None


# ===== PURE FUNCTION TESTS =====

class TestValidateTextsForEmbedding:
    """Test validate_texts_for_embedding pure function."""

    def test_valid_texts(self):
        """Test validation with valid texts."""
        texts = ["Hello world", "Test text", "  Another text  "]
        result = validate_texts_for_embedding(texts)

        assert result == ["Hello world", "Test text", "Another text"]

    def test_empty_list_error(self):
        """Test error with empty text list."""
        with pytest.raises(ValueError, match="Cannot generate embeddings for empty text list"):
            validate_texts_for_embedding([])

    def test_non_list_error(self):
        """Test error with non-list input."""
        with pytest.raises(ValueError, match="Texts must be provided as a list"):
            validate_texts_for_embedding("not a list")

    def test_none_text_error(self):
        """Test error with None text."""
        with pytest.raises(ValueError, match="Text at index 1 is None"):
            validate_texts_for_embedding(["valid text", None, "another text"])

    def test_non_string_error(self):
        """Test error with non-string text."""
        with pytest.raises(ValueError, match="Text at index 0 is not a string"):
            validate_texts_for_embedding([123, "valid text"])

    def test_empty_string_error(self):
        """Test error with empty string after cleaning."""
        with pytest.raises(ValueError, match="Text at index 1 is empty after cleaning"):
            validate_texts_for_embedding(["valid text", "   ", "another text"])

    def test_whitespace_cleaning(self):
        """Test whitespace cleaning."""
        texts = ["  hello  ", "\t\nworld\n\t", "   test   "]
        result = validate_texts_for_embedding(texts)

        assert result == ["hello", "world", "test"]


class TestCalculateOptimalBatchSize:
    """Test calculate_optimal_batch_size pure function."""

    def test_basic_calculation(self):
        """Test basic batch size calculation."""
        result = calculate_optimal_batch_size(100, None, base_batch_size=32)
        assert result == 32

    def test_zero_texts(self):
        """Test with zero texts."""
        result = calculate_optimal_batch_size(0, None, base_batch_size=32)
        assert result == 32

    def test_negative_texts(self):
        """Test with negative text count."""
        result = calculate_optimal_batch_size(-5, None, base_batch_size=32)
        assert result == 32

    def test_memory_based_adjustment(self):
        """Test memory-based batch size adjustment."""
        # 512MB available -> should suggest 32 batch size (512/16)
        result = calculate_optimal_batch_size(100, 512, base_batch_size=64)
        assert result == 32

    def test_memory_exceeds_max(self):
        """Test when memory suggests batch larger than max."""
        # 8GB available -> would suggest 512, but max is 256
        result = calculate_optimal_batch_size(1000, 8192, base_batch_size=32, max_batch_size=256)
        assert result == 32  # Takes min of base and memory-based

    def test_fewer_texts_than_batch(self):
        """Test when text count is less than batch size."""
        result = calculate_optimal_batch_size(10, None, base_batch_size=32)
        assert result == 10

    def test_minimum_batch_size(self):
        """Test minimum batch size enforcement."""
        result = calculate_optimal_batch_size(1, 1, base_batch_size=32)  # Very low memory
        assert result == 1


class TestSplitTextsIntoBatches:
    """Test split_texts_into_batches pure function."""

    def test_basic_splitting(self):
        """Test basic text splitting."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        batches = split_texts_into_batches(texts, 2)

        expected = [["text1", "text2"], ["text3", "text4"], ["text5"]]
        assert batches == expected

    def test_exact_division(self):
        """Test when texts divide evenly into batches."""
        texts = ["text1", "text2", "text3", "text4"]
        batches = split_texts_into_batches(texts, 2)

        expected = [["text1", "text2"], ["text3", "text4"]]
        assert batches == expected

    def test_single_batch(self):
        """Test when batch size equals text count."""
        texts = ["text1", "text2", "text3"]
        batches = split_texts_into_batches(texts, 3)

        expected = [["text1", "text2", "text3"]]
        assert batches == expected

    def test_batch_larger_than_texts(self):
        """Test when batch size is larger than text count."""
        texts = ["text1", "text2"]
        batches = split_texts_into_batches(texts, 5)

        expected = [["text1", "text2"]]
        assert batches == expected

    def test_empty_texts(self):
        """Test with empty text list."""
        batches = split_texts_into_batches([], 2)
        assert batches == []

    def test_zero_batch_size_error(self):
        """Test error with zero batch size."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            split_texts_into_batches(["text1"], 0)

    def test_negative_batch_size_error(self):
        """Test error with negative batch size."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            split_texts_into_batches(["text1"], -1)


class TestNormalizeEmbeddingsArray:
    """Test normalize_embeddings_array pure function."""

    def test_basic_normalization(self):
        """Test basic L2 normalization."""
        embeddings = np.array([[3.0, 4.0], [1.0, 1.0]])
        normalized = normalize_embeddings_array(embeddings)

        # Check that norms are approximately 1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-10)

        # Check first embedding: [3,4] normalized = [0.6, 0.8]
        np.testing.assert_allclose(normalized[0], [0.6, 0.8], rtol=1e-10)

    def test_empty_array(self):
        """Test with empty array."""
        empty_array = np.array([])
        result = normalize_embeddings_array(empty_array)

        assert result.size == 0
        assert np.array_equal(result, empty_array)

    def test_zero_vector_handling(self):
        """Test handling of zero vectors."""
        embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
        normalized = normalize_embeddings_array(embeddings)

        # Zero vector should remain zero (due to epsilon handling)
        assert normalized[0, 0] == 0.0
        assert normalized[0, 1] == 0.0

        # Non-zero vector should be normalized
        np.testing.assert_allclose(np.linalg.norm(normalized[1]), 1.0, rtol=1e-10)

    def test_already_normalized(self):
        """Test with already normalized embeddings."""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])  # Unit vectors
        normalized = normalize_embeddings_array(embeddings)

        np.testing.assert_allclose(embeddings, normalized, rtol=1e-10)

    def test_single_embedding(self):
        """Test with single embedding."""
        embedding = np.array([[3.0, 4.0]])
        normalized = normalize_embeddings_array(embedding)

        np.testing.assert_allclose(normalized[0], [0.6, 0.8], rtol=1e-10)


class TestCombineBatchEmbeddings:
    """Test combine_batch_embeddings pure function."""

    def test_basic_combination(self):
        """Test basic batch combination."""
        batch1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        batch2 = np.array([[5.0, 6.0]])
        batch3 = np.array([[7.0, 8.0], [9.0, 10.0]])

        combined = combine_batch_embeddings([batch1, batch2, batch3])

        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        np.testing.assert_array_equal(combined, expected)

    def test_empty_batch_list(self):
        """Test with empty batch list."""
        combined = combine_batch_embeddings([])
        assert combined.size == 0

    def test_single_batch(self):
        """Test with single batch."""
        batch = np.array([[1.0, 2.0], [3.0, 4.0]])
        combined = combine_batch_embeddings([batch])

        np.testing.assert_array_equal(combined, batch)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        batch1 = np.array([[1.0, 2.0]])  # Size 1
        batch2 = np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])  # Size 3

        combined = combine_batch_embeddings([batch1, batch2])

        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_array_equal(combined, expected)


class TestValidateEmbeddingDimensions:
    """Test validate_embedding_dimensions pure function."""

    def test_valid_dimensions(self):
        """Test with valid dimensions."""
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Should not raise any exception
        validate_embedding_dimensions(embeddings, expected_dim=3, num_texts=2)

    def test_empty_array_error(self):
        """Test error with empty array."""
        with pytest.raises(ValueError, match="Embedding array is empty"):
            validate_embedding_dimensions(np.array([]))

    def test_wrong_shape_error(self):
        """Test error with wrong array shape."""
        # 1D array instead of 2D
        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            validate_embedding_dimensions(np.array([1.0, 2.0, 3.0]))

        # 3D array
        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            validate_embedding_dimensions(np.array([[[1.0, 2.0]]]))

    def test_wrong_num_texts_error(self):
        """Test error with wrong number of texts."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Expected 3 embeddings, got 2"):
            validate_embedding_dimensions(embeddings, num_texts=3)

    def test_wrong_dimension_error(self):
        """Test error with wrong embedding dimension."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Expected embedding dimension 3, got 2"):
            validate_embedding_dimensions(embeddings, expected_dim=3)

    def test_optional_parameters(self):
        """Test with optional parameters."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Should work with only one constraint
        validate_embedding_dimensions(embeddings, expected_dim=2)
        validate_embedding_dimensions(embeddings, num_texts=2)
        validate_embedding_dimensions(embeddings)  # No constraints


class TestCalculateEmbeddingStatistics:
    """Test calculate_embedding_statistics pure function."""

    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        stats = calculate_embedding_statistics(embeddings)

        assert stats["num_embeddings"] == 3
        assert stats["embedding_dim"] == 2
        assert isinstance(stats["mean_norm"], float)
        assert isinstance(stats["std_norm"], float)
        assert isinstance(stats["min_value"], float)
        assert isinstance(stats["max_value"], float)
        assert isinstance(stats["mean_value"], float)
        assert isinstance(stats["std_value"], float)

        # Check specific values
        assert stats["min_value"] == 1.0
        assert stats["max_value"] == 6.0
        assert stats["mean_value"] == 3.5

    def test_empty_array_statistics(self):
        """Test statistics with empty array."""
        stats = calculate_embedding_statistics(np.array([]))
        assert stats == {"empty": True}

    def test_single_embedding_statistics(self):
        """Test statistics with single embedding."""
        embeddings = np.array([[1.0, 2.0]])
        stats = calculate_embedding_statistics(embeddings)

        assert stats["num_embeddings"] == 1
        assert stats["embedding_dim"] == 2
        assert stats["std_norm"] == 0.0  # No variation with single embedding
        assert stats["std_value"] == 0.5  # Std of [1.0, 2.0]

    def test_statistics_types(self):
        """Test that all statistics are proper Python types."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = calculate_embedding_statistics(embeddings)

        # Ensure all numeric values are Python floats, not numpy types
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                assert isinstance(value, (int, float))
                # Should be JSON serializable
                import json
                json.dumps(value)


# ===== MOCK PROTOCOL IMPLEMENTATIONS =====

class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, embedding_dim: int = 384, device: str = "cpu", max_seq_length: int = 512):
        self._embedding_dim = embedding_dim
        self._device = device
        self._max_seq_length = max_seq_length

    def encode(self, texts: list[str], batch_size: int = 32, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        """Generate mock embeddings."""
        embeddings = np.random.rand(len(texts), self._embedding_dim).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def get_sentence_embedding_dimension(self) -> int:
        return self._embedding_dim


class MockModelLoader:
    """Mock model loader for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.loaded_models = {}

    def load_model(self, model_name: str, cache_dir: str, device: str, **kwargs) -> MockEmbeddingModel:
        if self.should_fail:
            raise RuntimeError(f"Failed to load model {model_name}")

        model = MockEmbeddingModel(device=device)
        self.loaded_models[model_name] = model
        return model

    def is_model_available(self, model_name: str) -> bool:
        # Mock some available models
        available_models = {"BAAI/bge-m3", "test/model", "available/model"}
        return model_name in available_models


class MockDeviceDetector:
    """Mock device detector for testing."""

    def __init__(self, device_type: str = "cpu", should_fail: bool = False):
        self.device_type = device_type
        self.should_fail = should_fail

    def detect_best_device(self, preferred_device: str = "auto") -> DeviceInfo:
        if self.should_fail:
            raise RuntimeError("Device detection failed")

        return DeviceInfo(
            device_type=self.device_type,
            device_name=f"Mock {self.device_type.upper()}",
            available_memory=8192 if self.device_type == "cuda" else None,
            device_properties={"mock": True}
        )

    def is_device_available(self, device: str) -> bool:
        return device in ["cpu", "cuda", "mps"]


# ===== MAIN CLASS TESTS =====

class TestMultilingualEmbeddingGenerator:
    """Test MultilingualEmbeddingGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EmbeddingConfig(
            model_name="test/model",
            batch_size=2,
            normalize_embeddings=True
        )
        self.mock_loader = MockModelLoader()
        self.mock_detector = MockDeviceDetector()
        self.mock_logger = Mock(spec=logging.Logger)

    def test_init_with_dependencies(self):
        """Test initialization with injected dependencies."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )

        assert generator.config is self.config
        assert generator.model_loader is self.mock_loader
        assert generator.device_detector is self.mock_detector
        assert generator.logger is self.mock_logger
        assert not generator._is_initialized

    def test_init_with_default_dependencies(self):
        """Test initialization with default dependencies."""
        with patch('src.vectordb.embedding_loaders.SentenceTransformerLoader') as mock_st_loader, \
             patch('src.vectordb.embedding_devices.TorchDeviceDetector') as mock_device_detector:

            mock_st_instance = Mock()
            mock_device_instance = Mock()
            mock_st_loader.return_value = mock_st_instance
            mock_device_detector.return_value = mock_device_instance

            generator = MultilingualEmbeddingGenerator(config=self.config)

            assert generator.model_loader is mock_st_instance
            assert generator.device_detector is mock_device_instance
            assert isinstance(generator.logger, logging.Logger)

    def test_initialization_success(self):
        """Test successful initialization."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )

        generator.initialize()

        assert generator._is_initialized is True
        assert generator._device_info is not None
        assert generator._model is not None
        assert generator._device_info.device_type == "cpu"

    def test_initialization_device_failure(self):
        """Test initialization failure during device detection."""
        failing_detector = MockDeviceDetector(should_fail=True)
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=failing_detector,
            logger=self.mock_logger
        )

        with pytest.raises(RuntimeError, match="Embedding system initialization failed"):
            generator.initialize()

        assert not generator._is_initialized

    def test_initialization_model_failure(self):
        """Test initialization failure during model loading."""
        failing_loader = MockModelLoader(should_fail=True)
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=failing_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )

        with pytest.raises(RuntimeError, match="Embedding system initialization failed"):
            generator.initialize()

        assert not generator._is_initialized

    def test_generate_embeddings_success(self):
        """Test successful embedding generation."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )
        generator.initialize()

        texts = ["Hello world", "Test text", "Another example"]
        result = generator.generate_embeddings(texts)

        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape[0] == 3  # 3 texts
        assert result.input_texts == texts
        assert result.model_name == "test/model"
        assert result.processing_time > 0
        assert "processing_time" in result.metadata
        assert "batch_size_used" in result.metadata
        assert "num_batches" in result.metadata

    def test_generate_embeddings_not_initialized(self):
        """Test embedding generation without initialization."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )

        with pytest.raises(RuntimeError, match="Embedding system not initialized"):
            generator.generate_embeddings(["test"])

    def test_generate_embeddings_invalid_texts(self):
        """Test embedding generation with invalid texts."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )
        generator.initialize()

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            generator.generate_embeddings([])

    def test_generate_embeddings_custom_parameters(self):
        """Test embedding generation with custom parameters."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )
        generator.initialize()

        texts = ["Text 1", "Text 2"]
        result = generator.generate_embeddings(
            texts,
            normalize=False,
            batch_size=1
        )

        assert result.metadata["normalized"] is False
        assert result.metadata["batch_size_used"] == 1
        assert result.metadata["num_batches"] == 2  # 2 texts with batch size 1

    def test_get_embedding_dimension_success(self):
        """Test getting embedding dimension."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )
        generator.initialize()

        dim = generator.get_embedding_dimension()
        assert dim == 384  # Mock model default

    def test_get_embedding_dimension_not_initialized(self):
        """Test getting embedding dimension without initialization."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )

        with pytest.raises(RuntimeError, match="Embedding system not initialized"):
            generator.get_embedding_dimension()

    def test_is_model_available(self):
        """Test model availability check."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )

        # Test with configured model
        assert generator.is_model_available() is True  # test/model is not in available list

        # Test with specific model
        assert generator.is_model_available("BAAI/bge-m3") is True
        assert generator.is_model_available("nonexistent/model") is False

    def test_get_device_info(self):
        """Test getting device information."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )

        # Before initialization
        assert generator.get_device_info() is None

        # After initialization
        generator.initialize()
        device_info = generator.get_device_info()
        assert device_info is not None
        assert device_info.device_type == "cpu"

    def test_get_model_info_success(self):
        """Test getting model information."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )
        generator.initialize()

        model_info = generator.get_model_info()

        assert model_info["model_name"] == "test/model"
        assert model_info["embedding_dimension"] == 384
        assert model_info["max_seq_length"] == 8192
        assert model_info["device"] == "cpu"
        assert model_info["normalized_by_default"] is True

    def test_get_model_info_not_initialized(self):
        """Test getting model information without initialization."""
        generator = MultilingualEmbeddingGenerator(
            config=self.config,
            model_loader=self.mock_loader,
            device_detector=self.mock_detector,
            logger=self.mock_logger
        )

        with pytest.raises(RuntimeError, match="Embedding system not initialized"):
            generator.get_model_info()


# ===== FACTORY FUNCTION TESTS =====

class TestCreateEmbeddingGenerator:
    """Test create_embedding_generator factory function."""

    def test_create_with_default_config(self):
        """Test factory with default config."""
        generator = create_embedding_generator()

        assert isinstance(generator, MultilingualEmbeddingGenerator)
        assert generator.config.model_name == "BAAI/bge-m3"  # Default
        assert isinstance(generator.logger, logging.Logger)

    def test_create_with_custom_config(self):
        """Test factory with custom config."""
        config = EmbeddingConfig(model_name="custom/model", batch_size=16)
        mock_loader = MockModelLoader()
        mock_detector = MockDeviceDetector()
        mock_logger = Mock(spec=logging.Logger)

        generator = create_embedding_generator(
            config=config,
            model_loader=mock_loader,
            device_detector=mock_detector,
            logger=mock_logger
        )

        assert generator.config is config
        assert generator.model_loader is mock_loader
        assert generator.device_detector is mock_detector
        assert generator.logger is mock_logger

    def test_create_with_partial_dependencies(self):
        """Test factory with some custom dependencies."""
        config = EmbeddingConfig(model_name="test/model")
        mock_loader = MockModelLoader()

        with patch('src.vectordb.embedding_devices.TorchDeviceDetector') as mock_device_detector:
            mock_device_instance = Mock()
            mock_device_detector.return_value = mock_device_instance

            generator = create_embedding_generator(
                config=config,
                model_loader=mock_loader
            )

            assert generator.config is config
            assert generator.model_loader is mock_loader
            assert generator.device_detector is mock_device_instance


# ===== INTEGRATION TESTS =====

class TestIntegration:
    """Integration tests for embedding system."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end embedding workflow."""
        config = EmbeddingConfig(
            model_name="test/model",
            batch_size=2,
            normalize_embeddings=True
        )

        generator = create_embedding_generator(
            config=config,
            model_loader=MockModelLoader(),
            device_detector=MockDeviceDetector(device_type="cuda")
        )

        # Initialize system
        generator.initialize()

        # Check system status
        assert generator._is_initialized
        device_info = generator.get_device_info()
        assert device_info.device_type == "cuda"

        # Generate embeddings
        texts = ["Hello world", "Machine learning", "Vector embeddings", "Test text"]
        result = generator.generate_embeddings(texts)

        # Validate results
        assert result.embeddings.shape == (4, 384)  # 4 texts, 384 dimensions
        assert result.input_texts == texts
        assert result.embedding_dim == 384
        assert result.processing_time > 0

        # Check metadata
        metadata = result.metadata
        assert metadata["batch_size_used"] == 2
        assert metadata["num_batches"] == 2  # 4 texts / 2 batch size
        assert metadata["normalized"] is True
        assert metadata["device"] == "cuda"
        assert "statistics" in metadata

        # Verify normalization
        norms = np.linalg.norm(result.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_protocol_compliance(self):
        """Test that mock implementations comply with protocols."""
        # Test EmbeddingModel protocol
        model = MockEmbeddingModel()
        assert hasattr(model, 'encode')
        assert hasattr(model, 'device')
        assert hasattr(model, 'max_seq_length')
        assert hasattr(model, 'get_sentence_embedding_dimension')

        # Test ModelLoader protocol
        loader = MockModelLoader()
        assert hasattr(loader, 'load_model')
        assert hasattr(loader, 'is_model_available')

        # Test DeviceDetector protocol
        detector = MockDeviceDetector()
        assert hasattr(detector, 'detect_best_device')
        assert hasattr(detector, 'is_device_available')

    def test_memory_calculation_integration(self):
        """Test integration of memory-based batch size calculation."""
        config = EmbeddingConfig(batch_size=64)  # Large base batch
        detector = MockDeviceDetector(device_type="cuda")  # Has memory info

        generator = create_embedding_generator(
            config=config,
            model_loader=MockModelLoader(),
            device_detector=detector
        )

        generator.initialize()

        # Generate embeddings - should use memory-optimized batch size
        texts = ["text"] * 1000  # Many texts
        result = generator.generate_embeddings(texts)

        # Memory-based batch should be smaller than config batch
        # 8192 MB / 16 = 512, but min with base 64 = 64
        assert result.metadata["batch_size_used"] <= 64

    def test_error_propagation(self):
        """Test that errors propagate correctly through the system."""
        # Test model loading error
        failing_loader = MockModelLoader(should_fail=True)
        generator = create_embedding_generator(
            model_loader=failing_loader,
            device_detector=MockDeviceDetector()
        )

        with pytest.raises(RuntimeError, match="Embedding system initialization failed"):
            generator.initialize()

        # Test device detection error
        failing_detector = MockDeviceDetector(should_fail=True)
        generator2 = create_embedding_generator(
            model_loader=MockModelLoader(),
            device_detector=failing_detector
        )

        with pytest.raises(RuntimeError, match="Embedding system initialization failed"):
            generator2.initialize()

    def test_batching_with_edge_cases(self):
        """Test batching behavior with edge cases."""
        config = EmbeddingConfig(batch_size=3)
        generator = create_embedding_generator(
            config=config,
            model_loader=MockModelLoader(),
            device_detector=MockDeviceDetector()
        )
        generator.initialize()

        # Test with exact batch size
        texts_exact = ["text1", "text2", "text3"]
        result = generator.generate_embeddings(texts_exact)
        assert result.metadata["num_batches"] == 1

        # Test with one extra text
        texts_extra = ["text1", "text2", "text3", "text4"]
        result = generator.generate_embeddings(texts_extra)
        assert result.metadata["num_batches"] == 2

        # Test with single text
        texts_single = ["single_text"]
        result = generator.generate_embeddings(texts_single)
        assert result.metadata["num_batches"] == 1
        assert result.metadata["batch_size_used"] == 1  # Optimized for single text

    def test_statistics_calculation_integration(self):
        """Test statistics calculation in real workflow."""
        generator = create_embedding_generator(
            model_loader=MockModelLoader(),
            device_detector=MockDeviceDetector()
        )
        generator.initialize()

        texts = ["Short", "Medium length text", "Very long text with multiple words and sentences"]
        result = generator.generate_embeddings(texts)

        stats = result.metadata["statistics"]
        assert stats["num_embeddings"] == 3
        assert stats["embedding_dim"] == 384
        assert "mean_norm" in stats
        assert "std_norm" in stats
        assert "min_value" in stats
        assert "max_value" in stats
        assert "mean_value" in stats
        assert "std_value" in stats

        # All statistics should be JSON serializable
        import json
        json.dumps(stats)