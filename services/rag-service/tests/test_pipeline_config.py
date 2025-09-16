"""
Test configuration for RAG pipeline components.
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from src.pipeline.config import load_yaml_config


class TestLoadYamlConfig(unittest.TestCase):
    """Test YAML configuration loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_yaml_data = {
            "log_level": "INFO",
            "enable_caching": True,
            "cache_dir": "/tmp/cache",
            "processing": {"chunk_size": 1000, "chunk_overlap": 200},
            "embedding": {"model_name": "test-model", "batch_size": 32},
        }

    def test_load_existing_yaml_config(self):
        """Test loading existing YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(self.test_yaml_data, f)
            config_path = f.name

        try:
            result = load_yaml_config(config_path)
            assert result == self.test_yaml_data
            assert result["log_level"] == "INFO"
            assert result["enable_caching"] is True
            assert "processing" in result
        finally:
            Path(config_path).unlink()

    def test_load_nonexistent_yaml_config(self):
        """Test loading non-existent YAML file."""
        result = load_yaml_config("/nonexistent/path/config.yaml")
        assert result == {}

    def test_load_empty_yaml_config(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            config_path = f.name

        try:
            result = load_yaml_config(config_path)
            assert result is None or result == {}
        finally:
            Path(config_path).unlink()

    def test_load_invalid_yaml_config(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")
            config_path = f.name

        try:
            with self.assertRaises(yaml.YAMLError):
                load_yaml_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_load_yaml_with_unicode(self):
        """Test loading YAML file with Unicode characters."""
        unicode_data = {
            "log_level": "INFO",
            "language": "hrvatski",
            "message": "Čitaj konfiguraciju",
            "settings": {"encoding": "UTF-8", "locale": "hr_HR"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.dump(unicode_data, f, allow_unicode=True)
            config_path = f.name

        try:
            result = load_yaml_config(config_path)
            assert result == unicode_data
            assert result["language"] == "hrvatski"
            assert result["message"] == "Čitaj konfiguraciju"
        finally:
            Path(config_path).unlink()


class TestRAGConfigComponents(unittest.TestCase):
    """Test individual RAG configuration components."""

    def test_config_models_import(self):
        """Test that config model imports work correctly."""
        from src.utils.config_models import (
            ChromaConfig,
            EmbeddingConfig,
            LanguageConfig,
            OllamaConfig,
            ProcessingConfig,
            RetrievalConfig,
        )

        # Test that all classes are importable
        assert ChromaConfig is not None
        assert EmbeddingConfig is not None
        assert LanguageConfig is not None
        assert OllamaConfig is not None
        assert ProcessingConfig is not None
        assert RetrievalConfig is not None

    def test_processing_config_creation(self):
        """Test ProcessingConfig creation."""
        from src.utils.config_models import ProcessingConfig

        config = ProcessingConfig(
            sentence_chunk_overlap=2,
            preserve_paragraphs=True,
            enable_smart_chunking=True,
            respect_document_structure=True,
        )

        assert config.sentence_chunk_overlap == 2
        assert config.preserve_paragraphs is True
        assert config.enable_smart_chunking is True
        assert config.respect_document_structure is True

    def test_embedding_config_creation(self):
        """Test EmbeddingConfig creation."""
        from src.utils.config_models import EmbeddingConfig

        config = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            device="auto",
            max_seq_length=8192,
            batch_size=32,
            normalize_embeddings=True,
            use_safetensors=True,
            trust_remote_code=False,
            torch_dtype="float32",
        )

        assert config.model_name == "BAAI/bge-m3"
        assert config.device == "auto"
        assert config.max_seq_length == 8192
        assert config.batch_size == 32
        assert config.normalize_embeddings is True

    def test_chroma_config_creation(self):
        """Test ChromaConfig creation."""
        from src.utils.config_models import ChromaConfig

        config = ChromaConfig(
            db_path="/data/chroma",
            collection_name="test_collection",
            distance_metric="cosine",
            chunk_size=1000,
            ef_construction=200,
            m=16,
            persist=True,
            allow_reset=False,
        )

        assert config.db_path == "/data/chroma"
        assert config.collection_name == "test_collection"
        assert config.distance_metric == "cosine"
        assert config.chunk_size == 1000

    def test_retrieval_config_creation(self):
        """Test RetrievalConfig creation."""
        from src.utils.config_models import RetrievalConfig

        config = RetrievalConfig(
            default_k=5,
            max_k=20,
            similarity_threshold=0.5,
            adaptive_retrieval=True,
            enable_reranking=False,
            diversity_lambda=0.6,
            use_hybrid_search=True,
            enable_query_expansion=False,
        )

        assert config.default_k == 5
        assert config.max_k == 20
        assert config.similarity_threshold == 0.5
        assert config.adaptive_retrieval is True

    def test_ollama_config_creation(self):
        """Test OllamaConfig creation."""
        from src.utils.config_models import OllamaConfig

        config = OllamaConfig(
            base_url="http://localhost:11434",
            model="qwen2.5:7b-instruct",
            timeout=30.0,
            temperature=0.3,
            max_tokens=2048,
            top_p=0.9,
            top_k=40,
            stream=False,
            keep_alive="5m",
            num_predict=2048,
            repeat_penalty=1.1,
            seed=-1,
        )

        assert config.base_url == "http://localhost:11434"
        assert config.model == "qwen2.5:7b-instruct"
        assert config.timeout == 30.0
        assert config.temperature == 0.3

    def test_language_config_creation(self):
        """Test LanguageConfig creation."""
        from src.utils.config_models import LanguageConfig

        config = LanguageConfig(
            language_code="hr",
            enable_morphological_expansion=True,
            enable_synonym_expansion=False,
            use_language_query_processing=True,
            language_priority=True,
            stop_words_file="config/stopwords/hr.txt",
            morphology_patterns_file="config/morphology/hr.json",
        )

        assert config.language_code == "hr"
        assert config.enable_morphological_expansion is True
        assert config.enable_synonym_expansion is False
        assert config.use_language_query_processing is True

    def test_config_serialization(self):
        """Test that config objects can be serialized to dict."""
        from src.utils.config_models import ProcessingConfig

        config = ProcessingConfig(
            sentence_chunk_overlap=3,
            preserve_paragraphs=False,
            enable_smart_chunking=True,
            respect_document_structure=False,
        )

        # Test that it has __dict__ attribute
        assert hasattr(config, "__dict__")
        config_dict = config.__dict__

        # Test that all values are preserved
        assert config_dict["sentence_chunk_overlap"] == 3
        assert config_dict["preserve_paragraphs"] is False
        assert config_dict["enable_smart_chunking"] is True
        assert config_dict["respect_document_structure"] is False


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration components."""

    def test_yaml_config_loading_workflow(self):
        """Test complete YAML configuration loading workflow."""
        # Create test YAML data
        test_config = {
            "system": {"log_level": "DEBUG", "enable_caching": False, "max_concurrent_requests": 5},
            "processing": {
                "sentence_chunk_overlap": 1,
                "preserve_paragraphs": False,
                "enable_smart_chunking": False,
                "respect_document_structure": True,
            },
            "embedding": {"model_name": "test-model", "device": "cpu", "batch_size": 8},
        }

        # Test YAML round-trip
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        try:
            loaded_config = load_yaml_config(config_path)
            assert loaded_config == test_config
            assert loaded_config["system"]["log_level"] == "DEBUG"
            assert loaded_config["processing"]["enable_smart_chunking"] is False
            assert loaded_config["embedding"]["model_name"] == "test-model"
        finally:
            Path(config_path).unlink()

    def test_config_object_creation_workflow(self):
        """Test creating config objects from loaded data."""
        from src.utils.config_models import EmbeddingConfig, ProcessingConfig

        # Simulate data loaded from YAML
        processing_data = {
            "sentence_chunk_overlap": 2,
            "preserve_paragraphs": True,
            "enable_smart_chunking": True,
            "respect_document_structure": True,
        }

        embedding_data = {
            "model_name": "BAAI/bge-m3",
            "device": "auto",
            "max_seq_length": 4096,
            "batch_size": 16,
            "normalize_embeddings": True,
            "use_safetensors": True,
            "trust_remote_code": False,
            "torch_dtype": "float32",
        }

        # Test object creation
        processing_config = ProcessingConfig(**processing_data)
        embedding_config = EmbeddingConfig(**embedding_data)

        assert processing_config.sentence_chunk_overlap == 2
        assert embedding_config.model_name == "BAAI/bge-m3"

        # Test serialization back to dict
        assert processing_config.__dict__ == processing_data
        assert embedding_config.__dict__ == embedding_data

    def test_config_validation_workflow(self):
        """Test configuration validation and error handling."""
        from src.utils.config_models import ProcessingConfig

        # Test valid configuration
        valid_data = {
            "sentence_chunk_overlap": 1,
            "preserve_paragraphs": True,
            "enable_smart_chunking": True,
            "respect_document_structure": False,
        }

        config = ProcessingConfig(**valid_data)
        assert isinstance(config, ProcessingConfig)

        # Test that required fields are enforced
        with self.assertRaises((TypeError, ValueError)):
            ProcessingConfig()  # Missing required fields

    def test_config_component_interaction(self):
        """Test interaction between different config components."""
        from src.utils.config_models import EmbeddingConfig, OllamaConfig, RetrievalConfig

        # Create related configs
        embedding_config = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            device="cuda",
            max_seq_length=8192,
            batch_size=32,
            normalize_embeddings=True,
            use_safetensors=True,
            trust_remote_code=False,
            torch_dtype="float32",
        )

        retrieval_config = RetrievalConfig(
            default_k=5,
            max_k=20,
            similarity_threshold=0.7,
            adaptive_retrieval=True,
            enable_reranking=True,
            diversity_lambda=0.5,
            use_hybrid_search=True,
            enable_query_expansion=False,
        )

        ollama_config = OllamaConfig(
            base_url="http://localhost:11434",
            model="qwen2.5:7b-instruct",
            timeout=30.0,
            temperature=0.3,
            max_tokens=2048,
            top_p=0.9,
            top_k=40,
            stream=False,
            keep_alive="5m",
            num_predict=2048,
            repeat_penalty=1.1,
            seed=-1,
        )

        # Test that configs work together logically
        assert embedding_config.batch_size <= retrieval_config.max_k * 4  # Reasonable relationship
        assert retrieval_config.similarity_threshold < 1.0
        assert ollama_config.temperature <= 1.0
        assert ollama_config.top_p <= 1.0

        # Test config modification
        original_threshold = retrieval_config.similarity_threshold
        retrieval_config.similarity_threshold = 0.8
        assert retrieval_config.similarity_threshold == 0.8
        assert retrieval_config.similarity_threshold != original_threshold


if __name__ == "__main__":
    unittest.main()
