"""
Unit tests for embeddings module.
Tests multilingual sentence-transformers with Croatian language support.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.vectordb.embeddings import (
    CroatianEmbeddingModel,
    EmbeddingCache,
    EmbeddingConfig,
    create_embedding_model,
    get_recommended_model,
)


class TestEmbeddingConfig:
    """Test embedding configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_name == "bge-m3"
        assert config.device == "auto"
        assert config.max_seq_length == 512
        assert config.batch_size == 32
        assert config.normalize_embeddings is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_name="test-model",
            device="cpu",
            max_seq_length=256,
            batch_size=16,
            normalize_embeddings=False,
        )

        assert config.model_name == "test-model"
        assert config.device == "cpu"
        assert config.max_seq_length == 256
        assert config.batch_size == 16
        assert config.normalize_embeddings is False


class TestCroatianEmbeddingModel:
    """Test Croatian embedding model functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def config(self, temp_cache_dir):
        """Create test configuration."""
        return EmbeddingConfig(
            model_name="bge-m3", cache_dir=temp_cache_dir, device="cpu", batch_size=2
        )

    @pytest.fixture
    def mock_model(self):
        """Create mock sentence transformer model."""
        mock = MagicMock()
        mock.max_seq_length = 512
        mock.get_sentence_embedding_dimension.return_value = 384
        mock.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return mock

    def test_initialization(self, config):
        """Test model initialization."""
        model = CroatianEmbeddingModel(config)

        assert model.config == config
        assert model.model is None
        assert model._model_loaded is False
        assert Path(config.cache_dir).exists()

    def test_device_detection_cpu(self, config):
        """Test CPU device detection."""
        with patch("torch.cuda.is_available", return_value=False):
            model = CroatianEmbeddingModel(config)
            assert model.device == "cpu"

    def test_device_detection_cuda(self, config):
        """Test CUDA device detection."""
        config.device = "auto"
        with patch("torch.cuda.is_available", return_value=True):
            model = CroatianEmbeddingModel(config)
            assert model.device == "cuda"

    def test_load_model(self, config, mock_model):
        """Test model loading."""
        with patch("src.vectordb.embeddings.SentenceTransformer", return_value=mock_model):
            model = CroatianEmbeddingModel(config)
            model.load_model()

            assert model._model_loaded is True
            assert model.model == mock_model
            assert model.model.max_seq_length == config.max_seq_length

    def test_encode_single_text(self, config, mock_model):
        """Test encoding single text."""
        with patch("src.vectordb.embeddings.SentenceTransformer", return_value=mock_model):
            model = CroatianEmbeddingModel(config)

            text = "Dobar dan, kako ste?"
            result = model.encode_text(text)

            assert isinstance(result, np.ndarray)
            mock_model.encode.assert_called_once()

    def test_encode_multiple_texts(self, config, mock_model):
        """Test encoding multiple texts."""
        with patch("src.vectordb.embeddings.SentenceTransformer", return_value=mock_model):
            model = CroatianEmbeddingModel(config)

            texts = ["Dobar dan", "Kako ste?", "Hvala vam"]
            result = model.encode_text(texts)

            assert isinstance(result, np.ndarray)
            mock_model.encode.assert_called_once()

    def test_encode_empty_input(self, config, mock_model):
        """Test encoding empty input."""
        with patch("src.vectordb.embeddings.SentenceTransformer", return_value=mock_model):
            model = CroatianEmbeddingModel(config)

            result = model.encode_text([])
            assert result.size == 0


class TestCroatianLanguageSupport:
    """Test Croatian language specific functionality."""

    @pytest.fixture
    def croatian_texts(self):
        """Croatian test texts with diacritics and specific words."""
        return [
            "Željko želi žuti žeton.",  # Ž diacritic
            "Čovjek čita časopis o čudima.",  # Č diacritic
            "Ćuprija je stara hrvatska građevina.",  # Ć diacritic
            "Šuma šumi, a šaran šeta.",  # Š diacritic
            "Đavo đipa oko đumbira.",  # Đ diacritic
            "Zagreb je glavni grad Hrvatske.",  # Standard Croatian
            "Hvala vam na pomoći!",  # Polite Croatian
            "Doviđenja i laku noć.",  # Common goodbye
        ]

    @pytest.fixture
    def model(self):
        """Create embedding model for Croatian tests."""
        config = EmbeddingConfig(
            model_name="bge-m3", cache_dir="./test_cache", device="cpu", batch_size=4
        )
        return CroatianEmbeddingModel(config)

    def test_croatian_diacritics_encoding(self, model, croatian_texts):
        """Test that Croatian diacritics are properly encoded."""
        # Mock the actual model to avoid downloading
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(len(croatian_texts), 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.max_seq_length = 512
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch("src.vectordb.embeddings.SentenceTransformer", return_value=mock_model):
            embeddings = model.encode_text(croatian_texts)

            assert len(embeddings) == len(croatian_texts)
            assert embeddings.shape[1] == 384

            # Verify that all texts were passed correctly (with diacritics)
            call_args = mock_model.encode.call_args[0][0]
            assert all(text in call_args for text in croatian_texts)

    def test_similarity_computation(self, model):
        """Test similarity computation between Croatian texts."""
        # Similar Croatian texts should have higher similarity
        text1_embedding = np.array([1.0, 0.0, 0.0])
        text2_embedding = np.array([0.9, 0.1, 0.0])
        text3_embedding = np.array([0.0, 0.0, 1.0])

        # High similarity
        sim_high = model.compute_similarity(text1_embedding, text2_embedding, "cosine")
        # Low similarity
        sim_low = model.compute_similarity(text1_embedding, text3_embedding, "cosine")

        assert sim_high > sim_low
        assert 0 <= sim_high <= 1
        assert 0 <= sim_low <= 1

    def test_find_most_similar_croatian(self, model):
        """Test finding most similar Croatian texts."""
        query_embedding = np.array([1.0, 0.0, 0.0])
        candidates = np.array(
            [
                [0.9, 0.1, 0.0],  # Very similar
                [0.1, 0.9, 0.0],  # Less similar
                [0.0, 0.0, 1.0],  # Least similar
            ]
        )

        results = model.find_most_similar(query_embedding, candidates, top_k=2)

        assert len(results) == 2
        assert results[0][0] == 0  # First candidate should be most similar
        assert results[0][1] > results[1][1]  # Scores should be descending


class TestEmbeddingCache:
    """Test embedding caching functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create embedding cache."""
        return EmbeddingCache(temp_cache_dir)

    def test_cache_set_and_get(self, cache):
        """Test setting and getting cached embeddings."""
        text = "Test Croatian text: Ovo je test."
        model_name = "test-model"
        embedding = np.array([0.1, 0.2, 0.3])

        # Set cache
        cache.set(text, model_name, embedding)

        # Get from cache
        cached_embedding = cache.get(text, model_name)

        assert cached_embedding is not None
        np.testing.assert_array_equal(cached_embedding, embedding)

    def test_cache_miss(self, cache):
        """Test cache miss scenario."""
        text = "Non-existent text"
        model_name = "test-model"

        cached_embedding = cache.get(text, model_name)
        assert cached_embedding is None

    def test_cache_key_generation(self, cache):
        """Test that different texts generate different cache keys."""
        text1 = "Prvi tekst"
        text2 = "Drugi tekst"
        model_name = "test-model"

        key1 = cache._get_cache_key(text1, model_name)
        key2 = cache._get_cache_key(text2, model_name)

        assert key1 != key2
        assert len(key1) == 32  # MD5 hash length
        assert len(key2) == 32


class TestUtilityFunctions:
    """Test utility and factory functions."""

    def test_create_embedding_model(self):
        """Test embedding model factory function."""
        model = create_embedding_model(
            model_name="test-model", device="cpu", cache_dir="./test_cache"
        )

        assert isinstance(model, CroatianEmbeddingModel)
        assert model.config.model_name == "test-model"
        assert model.config.device == "cpu"
        assert model.config.cache_dir == "./test_cache"

    def test_get_recommended_model_general(self):
        """Test getting recommended model for general use."""
        model_name = get_recommended_model("general")
        assert model_name == "bge-m3"

    def test_get_recommended_model_fast(self):
        """Test getting recommended model for fast processing."""
        model_name = get_recommended_model("fast")
        assert model_name == "bge-m3"

    def test_get_recommended_model_accurate(self):
        """Test getting recommended model for accuracy."""
        model_name = get_recommended_model("accurate")
        assert model_name == "paraphrase-multilingual-mpnet-base-v2"

    def test_get_recommended_model_cross_lingual(self):
        """Test getting recommended model for cross-lingual tasks."""
        model_name = get_recommended_model("cross-lingual")
        assert model_name == "sentence-transformers/LaBSE"

    def test_get_recommended_model_unknown(self):
        """Test getting recommended model for unknown use case."""
        model_name = get_recommended_model("unknown")
        assert model_name == "bge-m3"  # Default


class TestDocumentEncoding:
    """Test document encoding functionality."""

    @pytest.fixture
    def sample_documents(self):
        """Sample Croatian documents."""
        return [
            {
                "id": "doc1",
                "content": "Zagreb je glavni grad Hrvatske. Nalazi se u sjeverozapadnom dijelu zemlje.",
                "title": "O Zagrebu",
                "source": "test_doc_1.txt",
            },
            {
                "id": "doc2",
                "content": "Hrvatska ima prekrasnu obalu uz Jadransko more s mnogo otoka.",
                "title": "Hrvatska obala",
                "source": "test_doc_2.txt",
            },
            {
                "id": "doc3",
                "content": "Dubrovnik je poznat kao 'biser Jadrana' zbog svoje ljepote.",
                "title": "Dubrovnik",
                "source": "test_doc_3.txt",
            },
        ]

    def test_encode_documents(self, sample_documents):
        """Test encoding multiple documents."""
        config = EmbeddingConfig(device="cpu", batch_size=2)
        model = CroatianEmbeddingModel(config)

        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(len(sample_documents), 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.max_seq_length = 512
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch("src.vectordb.embeddings.SentenceTransformer", return_value=mock_model):
            enriched_docs = model.encode_documents(sample_documents)

            assert len(enriched_docs) == len(sample_documents)

            for i, doc in enumerate(enriched_docs):
                # Original fields preserved
                assert doc["id"] == sample_documents[i]["id"]
                assert doc["content"] == sample_documents[i]["content"]
                assert doc["title"] == sample_documents[i]["title"]

                # New fields added
                assert "embedding" in doc
                assert "embedding_model" in doc
                assert doc["embedding_model"] == config.model_name
                assert isinstance(doc["embedding"], np.ndarray)

    def test_encode_documents_empty(self):
        """Test encoding empty document list."""
        config = EmbeddingConfig(device="cpu")
        model = CroatianEmbeddingModel(config)

        result = model.encode_documents([])
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__])
