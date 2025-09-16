"""
Comprehensive tests for vectordb/embedding_loaders.py
Tests all model loaders, adapters, and mock implementations.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from src.vectordb.embedding_loaders import (
    SentenceTransformerLoader,
    SentenceTransformerAdapter,
    MockModelLoader,
    MockEmbeddingModel,
)


class TestSentenceTransformerLoader:
    """Test SentenceTransformerLoader class."""

    def setup_method(self):
        """Set up test instance."""
        self.loader = SentenceTransformerLoader()

    @patch('src.vectordb.embedding_loaders.SentenceTransformer')
    @patch('src.vectordb.embedding_loaders.Path')
    def test_load_model_success(self, mock_path, mock_sentence_transformer):
        """Test successful model loading."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance

        # Call load_model
        model_name = "test-model"
        cache_dir = "/tmp/cache"
        device = "cpu"
        kwargs = {"trust_remote_code": True}

        result = self.loader.load_model(model_name, cache_dir, device, **kwargs)

        # Verify Path operations
        mock_path.assert_called_once_with(cache_dir)
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify SentenceTransformer call
        mock_sentence_transformer.assert_called_once_with(
            model_name,
            cache_folder=cache_dir,
            device=device,
            trust_remote_code=True
        )

        # Verify result is wrapped in adapter
        assert isinstance(result, SentenceTransformerAdapter)
        assert result._model == mock_model

    @patch('src.vectordb.embedding_loaders.SentenceTransformer')
    @patch('src.vectordb.embedding_loaders.Path')
    def test_load_model_failure(self, mock_path, mock_sentence_transformer):
        """Test model loading failure."""
        # Setup mocks to raise exception
        mock_sentence_transformer.side_effect = Exception("Model not found")

        # Call load_model and expect RuntimeError
        with pytest.raises(RuntimeError, match="Model loading failed: Model not found"):
            self.loader.load_model("invalid-model", "/tmp/cache", "cpu")

    @patch('huggingface_hub.model_info')
    def test_is_model_available_success(self, mock_model_info):
        """Test successful model availability check."""
        # Setup mock to return model info
        mock_info = Mock()
        mock_model_info.return_value = mock_info

        result = self.loader.is_model_available("valid-model")

        assert result is True
        mock_model_info.assert_called_once_with("valid-model")

    @patch('huggingface_hub.model_info')
    def test_is_model_available_failure(self, mock_model_info):
        """Test model availability check failure."""
        # Setup mock to raise exception
        mock_model_info.side_effect = Exception("Model not found")

        result = self.loader.is_model_available("invalid-model")

        assert result is False
        mock_model_info.assert_called_once_with("invalid-model")

    @patch('huggingface_hub.model_info')
    def test_is_model_available_none_result(self, mock_model_info):
        """Test model availability check with None result."""
        # Setup mock to return None
        mock_model_info.return_value = None

        result = self.loader.is_model_available("model-with-none-info")

        assert result is False


class TestSentenceTransformerAdapter:
    """Test SentenceTransformerAdapter class."""

    def setup_method(self):
        """Set up test instance."""
        self.mock_model = Mock()
        self.adapter = SentenceTransformerAdapter(self.mock_model)

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter._model == self.mock_model

    def test_encode_method(self):
        """Test encode method delegation."""
        # Setup mock return value
        expected_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.mock_model.encode.return_value = expected_embeddings

        # Call encode
        texts = ["text1", "text2"]
        batch_size = 16
        normalize_embeddings = False
        kwargs = {"convert_to_numpy": True}

        result = self.adapter.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )

        # Verify delegation
        self.mock_model.encode.assert_called_once_with(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, expected_embeddings)

    def test_encode_default_parameters(self):
        """Test encode with default parameters."""
        texts = ["text1"]
        self.adapter.encode(texts)

        self.mock_model.encode.assert_called_once_with(
            texts,
            batch_size=32,
            normalize_embeddings=True
        )

    def test_device_property(self):
        """Test device property."""
        # Mock device as different types
        self.mock_model.device = "cuda:0"
        assert self.adapter.device == "cuda:0"

        # Test with torch device object
        mock_device = Mock()
        mock_device.__str__ = Mock(return_value="cuda:1")
        self.mock_model.device = mock_device
        assert self.adapter.device == "cuda:1"

    def test_max_seq_length_property(self):
        """Test max_seq_length property."""
        # Test with explicit max_seq_length
        self.mock_model.max_seq_length = 256
        assert self.adapter.max_seq_length == 256

        # Test with default value (no max_seq_length attribute)
        del self.mock_model.max_seq_length
        assert self.adapter.max_seq_length == 512

    def test_get_sentence_embedding_dimension(self):
        """Test get_sentence_embedding_dimension method."""
        expected_dim = 768
        self.mock_model.get_sentence_embedding_dimension.return_value = expected_dim

        result = self.adapter.get_sentence_embedding_dimension()

        assert result == expected_dim
        self.mock_model.get_sentence_embedding_dimension.assert_called_once()


class TestMockModelLoader:
    """Test MockModelLoader class."""

    def setup_method(self):
        """Set up test instance."""
        self.loader = MockModelLoader()

    def test_initialization(self):
        """Test mock loader initialization."""
        assert self.loader.models == {}
        assert self.loader.available_models == set()
        assert self.loader.call_log == []
        assert self.loader.should_raise is None

    def test_set_model(self):
        """Test setting mock model."""
        mock_model = Mock()
        model_name = "test-model"

        self.loader.set_model(model_name, mock_model)

        assert self.loader.models[model_name] == mock_model
        assert model_name in self.loader.available_models

    def test_set_available_models(self):
        """Test setting available models."""
        models = ["model1", "model2", "model3"]
        self.loader.set_available_models(models)

        assert self.loader.available_models == set(models)

    def test_set_exception(self):
        """Test setting exception to raise."""
        exception = RuntimeError("Test error")
        self.loader.set_exception(exception)

        assert self.loader.should_raise == exception

    def test_get_calls(self):
        """Test getting call log."""
        # Add some calls
        self.loader.call_log.append({"method": "test", "param": "value"})

        calls = self.loader.get_calls()

        assert calls == [{"method": "test", "param": "value"}]
        # Verify it returns a copy
        calls.append({"method": "new"})
        assert len(self.loader.call_log) == 1

    def test_clear_calls(self):
        """Test clearing call log."""
        self.loader.call_log.append({"method": "test"})
        assert len(self.loader.call_log) == 1

        self.loader.clear_calls()

        assert self.loader.call_log == []

    def test_load_model_with_set_model(self):
        """Test loading model that was explicitly set."""
        mock_model = Mock()
        model_name = "preset-model"
        self.loader.set_model(model_name, mock_model)

        result = self.loader.load_model(model_name, "/cache", "cpu")

        assert result == mock_model
        assert len(self.loader.call_log) == 1
        call = self.loader.call_log[0]
        assert call["method"] == "load_model"
        assert call["model_name"] == model_name
        assert call["cache_dir"] == "/cache"
        assert call["device"] == "cpu"

    def test_load_model_with_kwargs(self):
        """Test loading model with additional kwargs."""
        kwargs = {"trust_remote_code": True, "revision": "main"}

        result = self.loader.load_model("test-model", "/cache", "gpu", **kwargs)

        assert isinstance(result, MockEmbeddingModel)
        call = self.loader.call_log[0]
        assert call["kwargs"] == kwargs

    def test_load_model_default_mock(self):
        """Test loading model returns default mock when not set."""
        result = self.loader.load_model("unknown-model", "/cache", "cpu")

        assert isinstance(result, MockEmbeddingModel)
        assert result.model_name == "unknown-model"
        assert result.device == "cpu"

    def test_load_model_with_exception(self):
        """Test loading model raises set exception."""
        exception = ValueError("Custom error")
        self.loader.set_exception(exception)

        with pytest.raises(ValueError, match="Custom error"):
            self.loader.load_model("any-model", "/cache", "cpu")

        # Exception should be cleared after raising
        assert self.loader.should_raise is None

    def test_is_model_available_true(self):
        """Test model availability check returns True for available models."""
        self.loader.set_available_models(["available-model"])

        result = self.loader.is_model_available("available-model")

        assert result is True
        assert len(self.loader.call_log) == 1
        call = self.loader.call_log[0]
        assert call["method"] == "is_model_available"
        assert call["model_name"] == "available-model"

    def test_is_model_available_false(self):
        """Test model availability check returns False for unavailable models."""
        self.loader.set_available_models(["other-model"])

        result = self.loader.is_model_available("unavailable-model")

        assert result is False

    def test_call_logging_integration(self):
        """Test that all methods properly log calls."""
        self.loader.set_available_models(["test-model"])

        # Make various calls
        self.loader.is_model_available("test-model")
        self.loader.load_model("test-model", "/cache", "cpu", param="value")
        self.loader.is_model_available("other-model")

        calls = self.loader.get_calls()
        assert len(calls) == 3

        assert calls[0]["method"] == "is_model_available"
        assert calls[1]["method"] == "load_model"
        assert calls[2]["method"] == "is_model_available"


class TestMockEmbeddingModel:
    """Test MockEmbeddingModel class."""

    def setup_method(self):
        """Set up test instance."""
        self.model = MockEmbeddingModel()

    def test_initialization_defaults(self):
        """Test mock model initialization with defaults."""
        assert self.model.model_name == "mock-model"
        assert self.model.device == "cpu"
        assert self.model.max_seq_length == 512
        assert self.model.get_sentence_embedding_dimension() == 1024
        assert self.model.call_log == []

    def test_initialization_custom(self):
        """Test mock model initialization with custom values."""
        model = MockEmbeddingModel("custom-model", "cuda")

        assert model.model_name == "custom-model"
        assert model.device == "cuda"

    def test_encode_basic(self):
        """Test basic encode functionality."""
        texts = ["text1", "text2", "text3"]

        embeddings = self.model.encode(texts)

        assert embeddings.shape == (3, 1024)
        assert embeddings.dtype == np.float32

        # Check call logging
        assert len(self.model.call_log) == 1
        call = self.model.call_log[0]
        assert call["method"] == "encode"
        assert call["num_texts"] == 3
        assert call["batch_size"] == 32
        assert call["normalize_embeddings"] is True

    def test_encode_custom_parameters(self):
        """Test encode with custom parameters."""
        texts = ["text1"]
        batch_size = 16
        normalize_embeddings = False
        kwargs = {"convert_to_numpy": True}

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )

        assert embeddings.shape == (1, 1024)

        call = self.model.call_log[0]
        assert call["batch_size"] == batch_size
        assert call["normalize_embeddings"] is False
        assert call["kwargs"] == kwargs

    def test_encode_normalization(self):
        """Test embedding normalization."""
        texts = ["text1"]

        # Test with normalization (default)
        embeddings_normalized = self.model.encode(texts, normalize_embeddings=True)
        norms_normalized = np.linalg.norm(embeddings_normalized, axis=1)
        np.testing.assert_array_almost_equal(norms_normalized, 1.0, decimal=5)

        # Clear call log
        self.model.clear_calls()

        # Test without normalization
        embeddings_raw = self.model.encode(texts, normalize_embeddings=False)
        # Raw embeddings should not be normalized (norms likely != 1.0)
        norms_raw = np.linalg.norm(embeddings_raw, axis=1)
        # Most random vectors won't have norm exactly 1.0
        assert not np.allclose(norms_raw, 1.0, atol=0.1)

    def test_encode_different_dimensions(self):
        """Test encode with different embedding dimensions."""
        self.model.set_embedding_dimension(512)

        texts = ["text1", "text2"]
        embeddings = self.model.encode(texts)

        assert embeddings.shape == (2, 512)

    def test_device_property(self):
        """Test device property."""
        assert self.model.device == "cpu"

        # Test with custom device
        model = MockEmbeddingModel(device="cuda:1")
        assert model.device == "cuda:1"

    def test_max_seq_length_property(self):
        """Test max_seq_length property."""
        assert self.model.max_seq_length == 512

        # Test modification
        self.model._max_seq_length = 256
        assert self.model.max_seq_length == 256

    def test_get_sentence_embedding_dimension(self):
        """Test get_sentence_embedding_dimension method."""
        assert self.model.get_sentence_embedding_dimension() == 1024

        # Test after dimension change
        self.model.set_embedding_dimension(768)
        assert self.model.get_sentence_embedding_dimension() == 768

    def test_set_embedding_dimension(self):
        """Test setting embedding dimension."""
        self.model.set_embedding_dimension(512)

        assert self.model._embedding_dim == 512
        assert self.model.get_sentence_embedding_dimension() == 512

        # Test encode uses new dimension
        embeddings = self.model.encode(["text"])
        assert embeddings.shape == (1, 512)

    def test_get_calls(self):
        """Test getting call log."""
        self.model.encode(["text1"])
        self.model.encode(["text2"])

        calls = self.model.get_calls()
        assert len(calls) == 2
        assert all(call["method"] == "encode" for call in calls)

        # Verify it returns a copy
        calls.append({"method": "fake"})
        assert len(self.model.call_log) == 2

    def test_clear_calls(self):
        """Test clearing call log."""
        self.model.encode(["text"])
        assert len(self.model.call_log) == 1

        self.model.clear_calls()
        assert self.model.call_log == []

    def test_reproducible_behavior(self):
        """Test that mock model behavior is consistent for testing."""
        texts = ["same text"]

        # Multiple calls should produce different random embeddings
        embeddings1 = self.model.encode(texts)
        embeddings2 = self.model.encode(texts)

        # Should be different (random generation)
        assert not np.array_equal(embeddings1, embeddings2)

        # But both should have correct shape and properties
        assert embeddings1.shape == embeddings2.shape == (1, 1024)
        assert embeddings1.dtype == embeddings2.dtype == np.float32

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty texts list
        embeddings = self.model.encode([])
        assert embeddings.shape == (0, 1024)

        # Single text
        embeddings = self.model.encode(["single"])
        assert embeddings.shape == (1, 1024)

        # Large number of texts
        large_texts = [f"text_{i}" for i in range(100)]
        embeddings = self.model.encode(large_texts)
        assert embeddings.shape == (100, 1024)

    def test_normalization_edge_cases(self):
        """Test normalization with edge cases."""
        # Set very small dimension to test normalization edge case
        self.model.set_embedding_dimension(2)

        texts = ["text"]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        # Check that normalization handled potential division by zero
        assert np.isfinite(embeddings).all()
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-5)


class TestIntegration:
    """Test integration scenarios with multiple components."""

    def test_mock_loader_with_mock_model(self):
        """Test MockModelLoader with MockEmbeddingModel integration."""
        loader = MockModelLoader()
        custom_model = MockEmbeddingModel("custom-model", "gpu")
        custom_model.set_embedding_dimension(256)

        loader.set_model("custom-model", custom_model)
        loader.set_available_models(["custom-model", "other-model"])

        # Test availability
        assert loader.is_model_available("custom-model") is True
        assert loader.is_model_available("other-model") is True
        assert loader.is_model_available("unknown-model") is False

        # Test loading
        loaded_model = loader.load_model("custom-model", "/cache", "cpu")
        assert loaded_model == custom_model

        # Test encoding through loaded model
        embeddings = loaded_model.encode(["test text"])
        assert embeddings.shape == (1, 256)

        # Check call logs
        loader_calls = loader.get_calls()
        model_calls = loaded_model.get_calls()

        assert len(loader_calls) == 4  # 3 availability checks + 1 load
        assert len(model_calls) == 1   # 1 encode call

    def test_sentence_transformer_adapter_integration(self):
        """Test SentenceTransformerAdapter with mock SentenceTransformer."""
        mock_st_model = Mock()
        mock_st_model.device = "cuda:0"
        mock_st_model.max_seq_length = 384
        mock_st_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)

        adapter = SentenceTransformerAdapter(mock_st_model)

        # Test all adapter methods
        assert adapter.device == "cuda:0"
        assert adapter.max_seq_length == 384
        assert adapter.get_sentence_embedding_dimension() == 768

        texts = ["text1", "text2"]
        embeddings = adapter.encode(texts, batch_size=16)
        assert embeddings.shape == (2, 768)

        # Verify underlying model was called correctly
        mock_st_model.encode.assert_called_once_with(
            texts, batch_size=16, normalize_embeddings=True
        )

    @patch('src.vectordb.embedding_loaders.SentenceTransformer')
    @patch('src.vectordb.embedding_loaders.Path')
    def test_full_loading_workflow(self, mock_path, mock_sentence_transformer):
        """Test complete model loading workflow."""
        # Setup mocks
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.max_seq_length = 512
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_sentence_transformer.return_value = mock_model

        loader = SentenceTransformerLoader()

        # Load model
        adapter = loader.load_model("test-model", "/cache", "cpu", trust_remote_code=True)

        # Verify it's properly wrapped
        assert isinstance(adapter, SentenceTransformerAdapter)
        assert adapter._model == mock_model

        # Test usage through adapter
        assert adapter.device == "cpu"
        assert adapter.max_seq_length == 512
        assert adapter.get_sentence_embedding_dimension() == 1024

    def test_error_handling_integration(self):
        """Test error handling across components."""
        loader = MockModelLoader()
        error = RuntimeError("Model loading failed")
        loader.set_exception(error)

        # Test exception propagation
        with pytest.raises(RuntimeError, match="Model loading failed"):
            loader.load_model("any-model", "/cache", "cpu")

        # Test that exception is cleared
        default_model = loader.load_model("any-model", "/cache", "cpu")
        assert isinstance(default_model, MockEmbeddingModel)

    def test_call_logging_comprehensive(self):
        """Test comprehensive call logging across all components."""
        loader = MockModelLoader()
        loader.set_available_models(["model1", "model2"])

        # Test multiple operations
        loader.is_model_available("model1")
        model = loader.load_model("model1", "/cache", "cpu", param="value")
        model.encode(["text1", "text2"], batch_size=16)
        loader.is_model_available("model3")

        # Check loader calls
        loader_calls = loader.get_calls()
        assert len(loader_calls) == 3

        assert loader_calls[0]["method"] == "is_model_available"
        assert loader_calls[1]["method"] == "load_model"
        assert loader_calls[2]["method"] == "is_model_available"

        # Check model calls
        model_calls = model.get_calls()
        assert len(model_calls) == 1
        assert model_calls[0]["method"] == "encode"
        assert model_calls[0]["num_texts"] == 2
        assert model_calls[0]["batch_size"] == 16