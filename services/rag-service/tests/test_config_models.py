"""
Tests for utils/config_models.py module.
Tests configuration dataclass models with fail-fast validation patterns.
"""

import unittest
from unittest.mock import Mock, patch

from src.utils.config_models import (
    ChunkingConfig,
    ChunkingStrategy,
    ChromaConfig,
    DeviceType,
    EmbeddingConfig,
    HybridRetrievalConfig,
    LanguageConfig,
    LanguageSpecificConfig,
    OllamaConfig,
    ProcessingConfig,
    QueryProcessingConfig,
    RankingConfig,
    RankingMethod,
    ReRankingConfig,
    ResponseParsingConfig,
    RetrievalConfig,
    SearchConfig,
    StorageConfig,
    SystemConfig,
)


class TestEnums(unittest.TestCase):
    """Test enum definitions."""

    def test_device_type_enum(self):
        """Test DeviceType enum values."""
        self.assertEqual(DeviceType.CPU.value, "cpu")
        self.assertEqual(DeviceType.CUDA.value, "cuda")
        self.assertEqual(DeviceType.MPS.value, "mps")
        self.assertEqual(DeviceType.AUTO.value, "auto")

    def test_chunking_strategy_enum(self):
        """Test ChunkingStrategy enum values."""
        self.assertEqual(ChunkingStrategy.SLIDING_WINDOW.value, "sliding_window")
        self.assertEqual(ChunkingStrategy.SENTENCE.value, "sentence")
        self.assertEqual(ChunkingStrategy.PARAGRAPH.value, "paragraph")

    def test_ranking_method_enum(self):
        """Test RankingMethod enum values."""
        self.assertEqual(RankingMethod.BASIC.value, "basic")
        self.assertEqual(RankingMethod.LANGUAGE_ENHANCED.value, "language_enhanced")
        self.assertEqual(RankingMethod.SEMANTIC_BOOST.value, "semantic_boost")

    def test_enum_string_conversion(self):
        """Test enum string conversion."""
        self.assertEqual(str(DeviceType.CPU), "DeviceType.CPU")
        self.assertEqual(str(ChunkingStrategy.SENTENCE), "ChunkingStrategy.SENTENCE")
        self.assertEqual(str(RankingMethod.BASIC), "RankingMethod.BASIC")


class TestQueryProcessingConfig(unittest.TestCase):
    """Test QueryProcessingConfig dataclass."""

    def test_dataclass_creation(self):
        """Test creating config instance."""
        config = QueryProcessingConfig(
            language="hr",
            expand_synonyms=True,
            normalize_case=True,
            remove_stopwords=False,
            min_query_length=3,
            max_query_length=200,
            max_expanded_terms=5,
            enable_morphological_analysis=True,
            use_query_classification=False,
            enable_spell_check=True,
        )

        self.assertEqual(config.language, "hr")
        self.assertTrue(config.expand_synonyms)
        self.assertTrue(config.normalize_case)
        self.assertFalse(config.remove_stopwords)
        self.assertEqual(config.min_query_length, 3)
        self.assertEqual(config.max_query_length, 200)
        self.assertEqual(config.max_expanded_terms, 5)
        self.assertTrue(config.enable_morphological_analysis)
        self.assertFalse(config.use_query_classification)
        self.assertTrue(config.enable_spell_check)

    def test_from_validated_config(self):
        """Test creating config from validated configuration."""
        main_config = {
            "query_processing": {
                "expand_synonyms": True,
                "normalize_case": False,
                "remove_stopwords": True,
                "min_query_length": 5,
                "max_query_length": 150,
                "max_expanded_terms": 3,
                "enable_morphological_analysis": False,
                "use_query_classification": True,
                "enable_spell_check": False,
            }
        }

        config = QueryProcessingConfig.from_validated_config(main_config, "en")

        self.assertEqual(config.language, "en")
        self.assertTrue(config.expand_synonyms)
        self.assertFalse(config.normalize_case)
        self.assertTrue(config.remove_stopwords)
        self.assertEqual(config.min_query_length, 5)
        self.assertEqual(config.max_query_length, 150)
        self.assertEqual(config.max_expanded_terms, 3)
        self.assertFalse(config.enable_morphological_analysis)
        self.assertTrue(config.use_query_classification)
        self.assertFalse(config.enable_spell_check)

    def test_from_validated_config_missing_section(self):
        """Test error when query_processing section is missing."""
        main_config = {"other_section": {}}

        with self.assertRaises(KeyError):
            QueryProcessingConfig.from_validated_config(main_config, "hr")

    def test_from_validated_config_missing_key(self):
        """Test error when required key is missing."""
        main_config = {
            "query_processing": {
                "expand_synonyms": True,
                # Missing other required keys
            }
        }

        with self.assertRaises(KeyError):
            QueryProcessingConfig.from_validated_config(main_config, "hr")


class TestEmbeddingConfig(unittest.TestCase):
    """Test EmbeddingConfig dataclass."""

    def test_dataclass_creation(self):
        """Test creating config instance."""
        config = EmbeddingConfig(
            model_name="BAAI/bge-large-en-v1.5",
            device="cuda",
            max_seq_length=512,
            batch_size=32,
            normalize_embeddings=True,
            use_safetensors=True,
            trust_remote_code=False,
            torch_dtype="float32",
        )

        self.assertEqual(config.model_name, "BAAI/bge-large-en-v1.5")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.max_seq_length, 512)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.normalize_embeddings)
        self.assertTrue(config.use_safetensors)
        self.assertFalse(config.trust_remote_code)
        self.assertEqual(config.torch_dtype, "float32")

    def test_from_validated_config_without_language_override(self):
        """Test creating config from main configuration only."""
        main_config = {
            "embeddings": {
                "model_name": "test-model",
                "device": "cpu",
                "max_seq_length": 256,
                "batch_size": 16,
                "normalize_embeddings": False,
                "use_safetensors": False,
                "trust_remote_code": True,
                "torch_dtype": "float16",
            }
        }

        config = EmbeddingConfig.from_validated_config(main_config)

        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.max_seq_length, 256)
        self.assertEqual(config.batch_size, 16)
        self.assertFalse(config.normalize_embeddings)
        self.assertFalse(config.use_safetensors)
        self.assertTrue(config.trust_remote_code)
        self.assertEqual(config.torch_dtype, "float16")

    def test_from_validated_config_with_language_override(self):
        """Test creating config with language-specific overrides."""
        main_config = {
            "embeddings": {
                "model_name": "main-model",
                "device": "cpu",
                "max_seq_length": 256,
                "batch_size": 16,
                "normalize_embeddings": False,
                "use_safetensors": False,
                "trust_remote_code": True,
                "torch_dtype": "float16",
            }
        }

        language_config = {
            "embeddings": {
                "model_name": "lang-specific-model",
                "device": "cuda",
                "batch_size": 32,
            }
        }

        config = EmbeddingConfig.from_validated_config(main_config, language_config)

        # Should use language overrides where provided
        self.assertEqual(config.model_name, "lang-specific-model")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.batch_size, 32)
        # Should use main config for non-overridden values
        self.assertEqual(config.max_seq_length, 256)
        self.assertFalse(config.normalize_embeddings)


class TestRetrievalConfig(unittest.TestCase):
    """Test RetrievalConfig dataclass."""

    def test_dataclass_creation(self):
        """Test creating config instance."""
        config = RetrievalConfig(
            default_k=5,
            max_k=20,
            similarity_threshold=0.7,
            adaptive_retrieval=True,
            enable_reranking=False,
            diversity_lambda=0.5,
            use_hybrid_search=True,
            enable_query_expansion=False,
        )

        self.assertEqual(config.default_k, 5)
        self.assertEqual(config.max_k, 20)
        self.assertEqual(config.similarity_threshold, 0.7)
        self.assertTrue(config.adaptive_retrieval)
        self.assertFalse(config.enable_reranking)
        self.assertEqual(config.diversity_lambda, 0.5)
        self.assertTrue(config.use_hybrid_search)
        self.assertFalse(config.enable_query_expansion)

    def test_from_validated_config(self):
        """Test creating config from validated configuration."""
        main_config = {
            "retrieval": {
                "default_k": 10,
                "max_k": 50,
                "adaptive_retrieval": False,
                "enable_reranking": True,
                "diversity_lambda": 0.3,
                "use_hybrid_search": False,
                "enable_query_expansion": True,
            },
            "similarity_threshold": 0.8,
        }

        config = RetrievalConfig.from_validated_config(main_config)

        self.assertEqual(config.default_k, 10)
        self.assertEqual(config.max_k, 50)
        self.assertEqual(config.similarity_threshold, 0.8)
        self.assertFalse(config.adaptive_retrieval)
        self.assertTrue(config.enable_reranking)
        self.assertEqual(config.diversity_lambda, 0.3)
        self.assertFalse(config.use_hybrid_search)
        self.assertTrue(config.enable_query_expansion)

    def test_float_conversion(self):
        """Test that float values are properly converted."""
        main_config = {
            "retrieval": {
                "default_k": 5,
                "max_k": 20,
                "adaptive_retrieval": True,
                "enable_reranking": False,
                "diversity_lambda": "0.5",  # String that should be converted
                "use_hybrid_search": True,
                "enable_query_expansion": False,
            },
            "similarity_threshold": "0.7",  # String that should be converted
        }

        config = RetrievalConfig.from_validated_config(main_config)

        self.assertEqual(config.similarity_threshold, 0.7)
        self.assertEqual(config.diversity_lambda, 0.5)
        self.assertIsInstance(config.similarity_threshold, float)
        self.assertIsInstance(config.diversity_lambda, float)


class TestRankingConfig(unittest.TestCase):
    """Test RankingConfig dataclass."""

    def test_dataclass_creation(self):
        """Test creating config instance."""
        config = RankingConfig(
            method=RankingMethod.LANGUAGE_ENHANCED,
            enable_diversity=True,
            diversity_threshold=0.6,
            boost_recent=False,
            boost_authoritative=True,
            content_length_factor=False,
            keyword_density_factor=True,
            language_specific_boost=True,
        )

        self.assertEqual(config.method, RankingMethod.LANGUAGE_ENHANCED)
        self.assertTrue(config.enable_diversity)
        self.assertEqual(config.diversity_threshold, 0.6)
        self.assertFalse(config.boost_recent)
        self.assertTrue(config.boost_authoritative)
        self.assertFalse(config.content_length_factor)
        self.assertTrue(config.keyword_density_factor)
        self.assertTrue(config.language_specific_boost)

    def test_from_validated_config_valid_method(self):
        """Test creating config with valid ranking method."""
        main_config = {
            "ranking": {
                "method": "basic",
                "enable_diversity": False,
                "diversity_threshold": "0.4",
                "boost_recent": True,
                "boost_authoritative": False,
                "content_length_factor": True,
                "keyword_density_factor": False,
                "language_specific_boost": False,
            }
        }

        config = RankingConfig.from_validated_config(main_config)

        self.assertEqual(config.method, RankingMethod.BASIC)
        self.assertFalse(config.enable_diversity)
        self.assertEqual(config.diversity_threshold, 0.4)
        self.assertTrue(config.boost_recent)
        self.assertFalse(config.boost_authoritative)
        self.assertTrue(config.content_length_factor)
        self.assertFalse(config.keyword_density_factor)
        self.assertFalse(config.language_specific_boost)

    def test_from_validated_config_invalid_method(self):
        """Test error with invalid ranking method."""
        main_config = {
            "ranking": {
                "method": "invalid_method",
                "enable_diversity": False,
                "diversity_threshold": 0.4,
                "boost_recent": True,
                "boost_authoritative": False,
                "content_length_factor": True,
                "keyword_density_factor": False,
                "language_specific_boost": False,
            }
        }

        with self.assertRaises(ValueError) as cm:
            RankingConfig.from_validated_config(main_config)

        self.assertIn("Invalid ranking method: invalid_method", str(cm.exception))


class TestChunkingConfig(unittest.TestCase):
    """Test ChunkingConfig dataclass."""

    def test_dataclass_creation(self):
        """Test creating config instance."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE,
            max_chunk_size=512,
            preserve_sentence_boundaries=True,
            respect_paragraph_breaks=False,
            enable_smart_splitting=True,
            sentence_search_range=100,
            paragraph_separators=["\n\n", "\r\n\r\n"],
            min_sentence_length=10,
        )

        self.assertEqual(config.strategy, ChunkingStrategy.SENTENCE)
        self.assertEqual(config.max_chunk_size, 512)
        self.assertTrue(config.preserve_sentence_boundaries)
        self.assertFalse(config.respect_paragraph_breaks)
        self.assertTrue(config.enable_smart_splitting)
        self.assertEqual(config.sentence_search_range, 100)
        self.assertEqual(config.paragraph_separators, ["\n\n", "\r\n\r\n"])
        self.assertEqual(config.min_sentence_length, 10)

    def test_from_validated_config_valid_strategy(self):
        """Test creating config with valid chunking strategy."""
        main_config = {
            "chunking": {
                "strategy": "sliding_window",
                "max_chunk_size": 256,
                "preserve_sentence_boundaries": False,
                "respect_paragraph_breaks": True,
                "enable_smart_splitting": False,
                "sentence_search_range": 50,
                "paragraph_separators": ["\n"],
                "min_sentence_length": 5,
            }
        }

        config = ChunkingConfig.from_validated_config(main_config)

        self.assertEqual(config.strategy, ChunkingStrategy.SLIDING_WINDOW)
        self.assertEqual(config.max_chunk_size, 256)
        self.assertFalse(config.preserve_sentence_boundaries)
        self.assertTrue(config.respect_paragraph_breaks)
        self.assertFalse(config.enable_smart_splitting)
        self.assertEqual(config.sentence_search_range, 50)
        self.assertEqual(config.paragraph_separators, ["\n"])
        self.assertEqual(config.min_sentence_length, 5)

    def test_from_validated_config_invalid_strategy(self):
        """Test error with invalid chunking strategy."""
        main_config = {
            "chunking": {
                "strategy": "invalid_strategy",
                "max_chunk_size": 256,
                "preserve_sentence_boundaries": False,
                "respect_paragraph_breaks": True,
                "enable_smart_splitting": False,
                "sentence_search_range": 50,
                "paragraph_separators": ["\n"],
                "min_sentence_length": 5,
            }
        }

        with self.assertRaises(ValueError) as cm:
            ChunkingConfig.from_validated_config(main_config)

        self.assertIn("Invalid chunking strategy: invalid_strategy", str(cm.exception))


class TestComplexConfigs(unittest.TestCase):
    """Test more complex configuration classes."""

    def test_ollama_config_creation(self):
        """Test OllamaConfig creation."""
        main_config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "qwen2.5:7b-instruct",
                "timeout": "30.0",
                "temperature": "0.7",
                "max_tokens": 2048,
                "top_p": "0.9",
                "top_k": 50,
                "stream": True,
                "keep_alive": "5m",
                "num_predict": 512,
                "repeat_penalty": "1.1",
                "seed": 42,
            }
        }

        config = OllamaConfig.from_validated_config(main_config)

        self.assertEqual(config.base_url, "http://localhost:11434")
        self.assertEqual(config.model, "qwen2.5:7b-instruct")
        self.assertEqual(config.timeout, 30.0)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 50)
        self.assertTrue(config.stream)
        self.assertEqual(config.keep_alive, "5m")
        self.assertEqual(config.num_predict, 512)
        self.assertEqual(config.repeat_penalty, 1.1)
        self.assertEqual(config.seed, 42)

    def test_hybrid_retrieval_config_creation(self):
        """Test HybridRetrievalConfig creation."""
        main_config = {
            "hybrid_retrieval": {
                "dense_weight": "0.7",
                "sparse_weight": "0.3",
                "fusion_method": "rrf",
                "bm25_k1": "1.2",
                "bm25_b": "0.75",
            }
        }

        config = HybridRetrievalConfig.from_validated_config(main_config)

        self.assertEqual(config.dense_weight, 0.7)
        self.assertEqual(config.sparse_weight, 0.3)
        self.assertEqual(config.fusion_method, "rrf")
        self.assertEqual(config.bm25_k1, 1.2)
        self.assertEqual(config.bm25_b, 0.75)

    def test_search_config_creation(self):
        """Test SearchConfig creation."""
        main_config = {
            "search": {
                "default_method": "hybrid",
                "max_context_length": 4096,
                "rerank": True,
                "include_metadata": False,
                "include_distances": True,
                "weights": {
                    "semantic_weight": "0.6",
                    "keyword_weight": "0.4",
                },
            }
        }

        config = SearchConfig.from_validated_config(main_config)

        self.assertEqual(config.default_method, "hybrid")
        self.assertEqual(config.max_context_length, 4096)
        self.assertTrue(config.rerank)
        self.assertFalse(config.include_metadata)
        self.assertTrue(config.include_distances)
        self.assertEqual(config.semantic_weight, 0.6)
        self.assertEqual(config.keyword_weight, 0.4)

    def test_response_parsing_config_creation(self):
        """Test ResponseParsingConfig creation."""
        main_config = {
            "response_parsing": {
                "validate_responses": True,
                "extract_confidence_scores": False,
                "parse_citations": True,
                "handle_incomplete_responses": False,
                "max_response_length": 2048,
                "min_response_length": 50,
                "filter_hallucinations": True,
                "require_source_grounding": False,
                "confidence_threshold": "0.8",
                "response_format": "json",
                "include_metadata": True,
            }
        }

        config = ResponseParsingConfig.from_validated_config(main_config)

        self.assertTrue(config.validate_responses)
        self.assertFalse(config.extract_confidence_scores)
        self.assertTrue(config.parse_citations)
        self.assertFalse(config.handle_incomplete_responses)
        self.assertEqual(config.max_response_length, 2048)
        self.assertEqual(config.min_response_length, 50)
        self.assertTrue(config.filter_hallucinations)
        self.assertFalse(config.require_source_grounding)
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertEqual(config.response_format, "json")
        self.assertTrue(config.include_metadata)


class TestLanguageSpecificConfig(unittest.TestCase):
    """Test LanguageSpecificConfig dataclass."""

    def test_from_validated_config(self):
        """Test creating config from validated language configuration."""
        language_config = {
            "language": {
                "code": "hr",
                "name": "Croatian",
                "family": "South Slavic",
            },
            "shared": {
                "preserve_diacritics": True,
                "response_language": "hr",
                "stopwords": {"words": ["i", "je", "da", "se"]},
                "question_patterns": {
                    "what": ["što", "šta"],
                    "when": ["kada", "kad"],
                },
            },
            "categorization": {"cultural_indicators": ["Croatian", "Zagreb", "Hrvat"]},
            "generation": {
                "system_prompt_language": "hr",
                "formality_level": "formal",
            },
        }

        config = LanguageSpecificConfig.from_validated_config(language_config)

        self.assertEqual(config.language_code, "hr")
        self.assertEqual(config.language_name, "Croatian")
        self.assertEqual(config.language_family, "South Slavic")
        self.assertTrue(config.preserve_diacritics)
        self.assertEqual(config.response_language, "hr")
        self.assertEqual(config.stopwords, ["i", "je", "da", "se"])
        self.assertEqual(config.question_patterns, {"what": ["što", "šta"], "when": ["kada", "kad"]})
        self.assertEqual(config.cultural_indicators, ["Croatian", "Zagreb", "Hrvat"])
        self.assertEqual(config.system_prompt_language, "hr")
        self.assertEqual(config.formality_level, "formal")


class TestSystemConfig(unittest.TestCase):
    """Test SystemConfig master configuration."""

    def test_from_validated_configs(self):
        """Test creating complete system configuration."""
        # Create minimal valid configs for all components
        main_config = {
            "query_processing": {
                "expand_synonyms": True,
                "normalize_case": True,
                "remove_stopwords": False,
                "min_query_length": 3,
                "max_query_length": 200,
                "max_expanded_terms": 5,
                "enable_morphological_analysis": True,
                "use_query_classification": False,
                "enable_spell_check": True,
            },
            "embeddings": {
                "model_name": "test-model",
                "device": "cpu",
                "max_seq_length": 256,
                "batch_size": 16,
                "normalize_embeddings": False,
                "use_safetensors": False,
                "trust_remote_code": True,
                "torch_dtype": "float16",
            },
            "retrieval": {
                "default_k": 5,
                "max_k": 20,
                "adaptive_retrieval": True,
                "enable_reranking": False,
                "diversity_lambda": 0.5,
                "use_hybrid_search": True,
                "enable_query_expansion": False,
            },
            "similarity_threshold": 0.7,
            "ranking": {
                "method": "basic",
                "enable_diversity": False,
                "diversity_threshold": 0.4,
                "boost_recent": True,
                "boost_authoritative": False,
                "content_length_factor": True,
                "keyword_density_factor": False,
                "language_specific_boost": False,
            },
            "reranking": {
                "enabled": True,
                "model_name": "test-reranker",
                "max_length": 512,
                "batch_size": 8,
                "top_k": 10,
                "use_fp16": True,
                "normalize": False,
            },
            "hybrid_retrieval": {
                "dense_weight": 0.7,
                "sparse_weight": 0.3,
                "fusion_method": "rrf",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "test-model",
                "timeout": 30.0,
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "top_k": 50,
                "stream": True,
                "keep_alive": "5m",
                "num_predict": 512,
                "repeat_penalty": 1.1,
                "seed": 42,
            },
            "processing": {
                "sentence_chunk_overlap": 50,
                "preserve_paragraphs": True,
                "enable_smart_chunking": False,
                "respect_document_structure": True,
            },
            "chunking": {
                "strategy": "sentence",
                "max_chunk_size": 512,
                "preserve_sentence_boundaries": True,
                "respect_paragraph_breaks": False,
                "enable_smart_splitting": True,
                "sentence_search_range": 100,
                "paragraph_separators": ["\n\n"],
                "min_sentence_length": 10,
            },
            "storage": {
                "db_path_template": "/data/{tenant}_{user}.db",
                "collection_name_template": "{tenant}_{user}_{language}_documents",
                "distance_metric": "cosine",
                "persist": True,
                "allow_reset": False,
            },
            "search": {
                "default_method": "hybrid",
                "max_context_length": 4096,
                "rerank": True,
                "include_metadata": False,
                "include_distances": True,
                "weights": {"semantic_weight": 0.6, "keyword_weight": 0.4},
            },
            "response_parsing": {
                "validate_responses": True,
                "extract_confidence_scores": False,
                "parse_citations": True,
                "handle_incomplete_responses": False,
                "max_response_length": 2048,
                "min_response_length": 50,
                "filter_hallucinations": True,
                "require_source_grounding": False,
                "confidence_threshold": 0.8,
                "response_format": "json",
                "include_metadata": True,
            },
        }

        language_config = {
            "language": {"code": "hr", "name": "Croatian", "family": "South Slavic"},
            "shared": {
                "preserve_diacritics": True,
                "response_language": "hr",
                "stopwords": {"words": ["i", "je"]},
                "question_patterns": {"what": ["što"]},
            },
            "categorization": {"cultural_indicators": ["Croatian"]},
            "generation": {"system_prompt_language": "hr", "formality_level": "formal"},
        }

        with patch("src.utils.config_models.logger") as mock_logger:
            system_config = SystemConfig.from_validated_configs(main_config, language_config, "hr")

        # Verify logging
        mock_logger.info.assert_called_once_with("Creating system configuration for language: hr")

        # Verify all component configs were created
        self.assertIsInstance(system_config.query_processing, QueryProcessingConfig)
        self.assertIsInstance(system_config.embedding, EmbeddingConfig)
        self.assertIsInstance(system_config.retrieval, RetrievalConfig)
        self.assertIsInstance(system_config.ranking, RankingConfig)
        self.assertIsInstance(system_config.reranking, ReRankingConfig)
        self.assertIsInstance(system_config.hybrid_retrieval, HybridRetrievalConfig)
        self.assertIsInstance(system_config.ollama, OllamaConfig)
        self.assertIsInstance(system_config.processing, ProcessingConfig)
        self.assertIsInstance(system_config.chunking, ChunkingConfig)
        self.assertIsInstance(system_config.storage, StorageConfig)
        self.assertIsInstance(system_config.search, SearchConfig)
        self.assertIsInstance(system_config.response_parsing, ResponseParsingConfig)
        self.assertIsInstance(system_config.language_specific, LanguageSpecificConfig)

        # Verify some specific values
        self.assertEqual(system_config.query_processing.language, "hr")
        self.assertEqual(system_config.embedding.model_name, "test-model")
        self.assertEqual(system_config.language_specific.language_code, "hr")


class TestLanguageConfig(unittest.TestCase):
    """Test LanguageConfig dataclass."""

    @patch("src.utils.config_protocol.get_config_provider")
    def test_from_validated_config(self, mock_get_provider):
        """Test creating config from validated configuration."""
        mock_provider = Mock()
        mock_provider.get_language_specific_config.return_value = {
            "enable_morphological_expansion": True,
            "enable_synonym_expansion": False,
            "use_language_query_processing": True,
            "language_priority": False,
            "stop_words_file": "stopwords_hr.txt",
            "morphology_patterns_file": "morphology_hr.json",
        }
        mock_get_provider.return_value = mock_provider

        main_config = {}  # Not used in this method
        config = LanguageConfig.from_validated_config(main_config, "hr")

        mock_provider.get_language_specific_config.assert_called_once_with("pipeline", "hr")

        self.assertEqual(config.language_code, "hr")
        self.assertTrue(config.enable_morphological_expansion)
        self.assertFalse(config.enable_synonym_expansion)
        self.assertTrue(config.use_language_query_processing)
        self.assertFalse(config.language_priority)
        self.assertEqual(config.stop_words_file, "stopwords_hr.txt")
        self.assertEqual(config.morphology_patterns_file, "morphology_hr.json")


class TestConfigDataclassProperties(unittest.TestCase):
    """Test dataclass properties and behavior."""

    def test_dataclass_immutability_patterns(self):
        """Test that configs follow immutability patterns."""
        config = QueryProcessingConfig(
            language="test",
            expand_synonyms=True,
            normalize_case=True,
            remove_stopwords=False,
            min_query_length=3,
            max_query_length=200,
            max_expanded_terms=5,
            enable_morphological_analysis=True,
            use_query_classification=False,
            enable_spell_check=True,
        )

        # Dataclass should be mutable by default, but we can test field access
        self.assertTrue(hasattr(config, "__dataclass_fields__"))
        self.assertIn("language", config.__dataclass_fields__)
        self.assertIn("expand_synonyms", config.__dataclass_fields__)

    def test_dataclass_equality(self):
        """Test dataclass equality comparison."""
        config1 = QueryProcessingConfig(
            language="hr",
            expand_synonyms=True,
            normalize_case=True,
            remove_stopwords=False,
            min_query_length=3,
            max_query_length=200,
            max_expanded_terms=5,
            enable_morphological_analysis=True,
            use_query_classification=False,
            enable_spell_check=True,
        )

        config2 = QueryProcessingConfig(
            language="hr",
            expand_synonyms=True,
            normalize_case=True,
            remove_stopwords=False,
            min_query_length=3,
            max_query_length=200,
            max_expanded_terms=5,
            enable_morphological_analysis=True,
            use_query_classification=False,
            enable_spell_check=True,
        )

        config3 = QueryProcessingConfig(
            language="en",  # Different language
            expand_synonyms=True,
            normalize_case=True,
            remove_stopwords=False,
            min_query_length=3,
            max_query_length=200,
            max_expanded_terms=5,
            enable_morphological_analysis=True,
            use_query_classification=False,
            enable_spell_check=True,
        )

        self.assertEqual(config1, config2)
        self.assertNotEqual(config1, config3)

    def test_dataclass_repr(self):
        """Test dataclass string representation."""
        config = DeviceType.CPU
        repr_str = repr(config)
        self.assertIn("DeviceType.CPU", repr_str)

    def test_type_annotations(self):
        """Test that dataclasses have proper type annotations."""
        self.assertTrue(hasattr(QueryProcessingConfig, "__annotations__"))
        annotations = QueryProcessingConfig.__annotations__

        self.assertEqual(annotations["language"], str)
        self.assertEqual(annotations["expand_synonyms"], bool)
        self.assertEqual(annotations["min_query_length"], int)


if __name__ == "__main__":
    unittest.main()