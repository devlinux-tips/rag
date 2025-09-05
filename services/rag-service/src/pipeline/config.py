"""
Configuration management for RAG system.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from pydantic import ConfigDict, Field
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings, Field

    ConfigDict = None


@dataclass
class ProcessingConfig:
    """Document processing configuration."""

    max_chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    sentence_chunk_overlap: int
    preserve_paragraphs: bool
    enable_smart_chunking: bool = True
    respect_document_structure: bool = True

    @classmethod
    def from_config(
        cls, config_dict: Optional[Dict[str, Any]] = None, language: str = "hr"
    ) -> "ProcessingConfig":
        """Create config from dictionary with DRY error handling."""
        from ..utils.config_loader import get_language_specific_config, get_processing_config
        from ..utils.error_handler import handle_config_error

        # Load main processing config
        processing_config = config_dict or handle_config_error(
            operation=get_processing_config,
            fallback_value={},
            config_file="config/config.toml",
            section="processing",
        )

        # Load language-specific settings
        language_config = handle_config_error(
            operation=lambda: get_language_specific_config("pipeline", language),
            fallback_value={"processing": {}},
            config_file=f"config/{language}.toml",
            section="pipeline",
        )
        language_processing = language_config.get("processing", {})

        return cls(
            max_chunk_size=processing_config.get("max_chunk_size", 512),
            chunk_overlap=processing_config.get("chunk_overlap", 50),
            min_chunk_size=processing_config.get("min_chunk_size", 100),
            sentence_chunk_overlap=processing_config.get("sentence_chunk_overlap", 2),
            preserve_paragraphs=processing_config.get("preserve_paragraphs", True),
            enable_smart_chunking=processing_config.get("enable_smart_chunking", True),
            respect_document_structure=processing_config.get("respect_document_structure", True),
        )


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name: str
    cache_folder: str
    batch_size: int
    max_seq_length: int
    device: str = "auto"
    normalize_embeddings: bool = True

    @classmethod
    def from_config(cls, config_dict: Optional[Dict[str, Any]] = None) -> "EmbeddingConfig":
        """Create config from dictionary with DRY error handling."""
        from ..utils.config_loader import get_pipeline_config
        from ..utils.error_handler import handle_config_error

        # Load pipeline embedding config (this references vectordb config)
        pipeline_config = config_dict or handle_config_error(
            operation=get_pipeline_config,
            fallback_value={},
            config_file="config/config.toml",
            section="pipeline",
        )
        embedding_config = pipeline_config.get("embedding", {})

        return cls(
            model_name=embedding_config.get("model_name", "BAAI/bge-m3"),
            cache_folder=embedding_config.get("cache_folder", "./models/embeddings"),
            batch_size=embedding_config.get("batch_size", 32),
            max_seq_length=embedding_config.get("max_seq_length", 512),
            device=embedding_config.get("device", "auto"),
            normalize_embeddings=embedding_config.get("normalize_embeddings", True),
        )


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""

    db_path: str
    collection_name: str
    distance_metric: str
    ef_construction: int
    m: int
    persist: bool = True
    allow_reset: bool = False

    @classmethod
    def from_config(cls, config_dict: Optional[Dict[str, Any]] = None) -> "ChromaConfig":
        """Create config from dictionary with DRY error handling."""
        from ..utils.config_loader import get_chroma_config
        from ..utils.error_handler import handle_config_error

        # Load chroma config
        chroma_config = config_dict or handle_config_error(
            operation=get_chroma_config,
            fallback_value={},
            config_file="config/config.toml",
            section="chroma",
        )

        return cls(
            db_path=chroma_config.get("db_path", "./data/chromadb"),
            collection_name=chroma_config.get("collection_name", "multilingual_documents"),
            distance_metric=chroma_config.get("distance_metric", "cosine"),
            ef_construction=chroma_config.get("ef_construction", 200),
            m=chroma_config.get("m", 16),
            persist=chroma_config.get("persist", True),
            allow_reset=chroma_config.get("allow_reset", False),
        )


@dataclass
class RetrievalConfig:
    """Retrieval system configuration."""

    default_k: int
    max_k: int
    min_similarity_score: float
    adaptive_retrieval: bool
    enable_reranking: bool
    diversity_lambda: float
    use_hybrid_search: bool = True
    enable_query_expansion: bool = True

    @classmethod
    def from_config(
        cls, config_dict: Optional[Dict[str, Any]] = None, language: str = "hr"
    ) -> "RetrievalConfig":
        """Create config from dictionary with DRY error handling."""
        from ..utils.config_loader import get_language_specific_config, get_pipeline_config
        from ..utils.error_handler import handle_config_error

        # Load main retrieval config
        pipeline_config = config_dict or handle_config_error(
            operation=get_pipeline_config,
            fallback_value={},
            config_file="config/config.toml",
            section="pipeline",
        )
        retrieval_config = pipeline_config.get("retrieval", {})

        # Load language-specific settings
        language_config = handle_config_error(
            operation=lambda: get_language_specific_config("pipeline", language),
            fallback_value={"retrieval": {}},
            config_file=f"config/{language}.toml",
            section="pipeline",
        )
        language_retrieval = language_config.get("retrieval", {})

        return cls(
            default_k=retrieval_config.get("default_k", 5),
            max_k=retrieval_config.get("max_k", 10),
            min_similarity_score=retrieval_config.get("min_similarity_score", 0.3),
            adaptive_retrieval=retrieval_config.get("adaptive_retrieval", True),
            enable_reranking=retrieval_config.get("enable_reranking", True),
            diversity_lambda=retrieval_config.get("diversity_lambda", 0.3),
            use_hybrid_search=retrieval_config.get("use_hybrid_search", True),
            enable_query_expansion=retrieval_config.get("enable_query_expansion", True),
        )


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""

    base_url: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    timeout: float
    preserve_diacritics: bool
    prefer_formal_style: bool
    include_cultural_context: bool
    stream: bool = True
    keep_alive: str = "5m"

    @classmethod
    def from_config(
        cls, config_dict: Optional[Dict[str, Any]] = None, language: str = "hr"
    ) -> "OllamaConfig":
        """Create config from dictionary with DRY error handling."""
        from ..utils.config_loader import (
            get_language_specific_config,
            get_ollama_config,
            get_shared_config,
        )
        from ..utils.error_handler import handle_config_error

        # Load main ollama config
        ollama_config = config_dict or handle_config_error(
            operation=get_ollama_config,
            fallback_value={},
            config_file="config/config.toml",
            section="ollama",
        )

        # Load language-specific generation settings
        language_config = handle_config_error(
            operation=lambda: get_language_specific_config("pipeline", language),
            fallback_value={"generation": {}},
            config_file=f"config/{language}.toml",
            section="pipeline",
        )
        language_generation = language_config.get("generation", {})
        language_shared = language_config.get("shared", {})
        pipeline_generation = language_config.get("pipeline", {}).get("generation", {})

        return cls(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            model=ollama_config.get("model", "llama3.1:8b"),
            temperature=ollama_config.get("temperature", 0.7),
            max_tokens=ollama_config.get("max_tokens", 2000),
            top_p=ollama_config.get("top_p", 0.9),
            top_k=ollama_config.get("top_k", 40),
            timeout=ollama_config.get("timeout", 60.0),
            preserve_diacritics=pipeline_generation.get(
                "preserve_diacritics",
                language_shared.get(
                    "preserve_diacritics", language_generation.get("preserve_diacritics", True)
                ),
            ),
            prefer_formal_style=pipeline_generation.get(
                "prefer_formal_style", language_generation.get("prefer_formal_style", True)
            ),
            include_cultural_context=pipeline_generation.get(
                "include_cultural_context",
                language_generation.get("include_cultural_context", True),
            ),
            stream=ollama_config.get("stream", True),
            keep_alive=ollama_config.get("keep_alive", "5m"),
        )


@dataclass
class LanguageConfig:
    """Language-specific configuration for multilingual support."""

    enable_morphological_expansion: bool
    enable_synonym_expansion: bool
    enable_cultural_context: bool
    use_language_query_processing: bool
    language_priority: bool
    language_code: str = "hr"
    stop_words_file: str = "config/hr_stop_words.txt"
    morphology_patterns_file: str = "config/hr_morphology.json"
    cultural_context_file: str = "config/hr_cultural_context.json"

    @classmethod
    def from_config(
        cls, config_dict: Optional[Dict[str, Any]] = None, language: str = "hr"
    ) -> "LanguageConfig":
        """Create config from dictionary with language support."""
        from ..utils.config_loader import get_language_specific_config
        from ..utils.error_handler import handle_config_error

        # Load language-specific pipeline config
        language_config = config_dict or handle_config_error(
            operation=lambda: get_language_specific_config("pipeline", language),
            fallback_value={},
            config_file=f"config/{language}.toml",
            section="pipeline",
        )

        return cls(
            language_code=language,
            enable_morphological_expansion=language_config.get(
                "enable_morphological_expansion", True
            ),
            enable_synonym_expansion=language_config.get("enable_synonym_expansion", True),
            enable_cultural_context=language_config.get("enable_cultural_context", True),
            use_language_query_processing=language_config.get(
                "use_language_query_processing", True
            ),
            language_priority=language_config.get("language_priority", True),
            stop_words_file=language_config.get(
                "stop_words_file", f"config/{language}_stop_words.txt"
            ),
            morphology_patterns_file=language_config.get(
                "morphology_patterns_file", f"config/{language}_morphology.json"
            ),
            cultural_context_file=language_config.get(
                "cultural_context_file", f"config/{language}_cultural_context.json"
            ),
        )


class RAGConfig(BaseSettings):
    """Main configuration for multilingual RAG system."""

    # Component configurations
    processing: ProcessingConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig
    ollama: OllamaConfig
    language: LanguageConfig

    # System settings
    log_level: str
    enable_caching: bool
    cache_dir: str

    # Performance settings
    max_concurrent_requests: int
    request_timeout: float
    enable_metrics: bool
    metrics_dir: str

    # Data paths
    documents_dir: str
    processed_dir: str
    test_data_dir: str

    # Use ConfigDict for Pydantic v2 compatibility
    if ConfigDict:
        model_config = ConfigDict(env_file=".env", extra="allow")
    else:

        class Config:
            env_file = ".env"
            extra = "allow"

    def __init__(self, language: str = "hr", **data):
        """Initialize with DRY config loading."""
        # If no data provided, load from config files
        if not data:
            from ..utils.config_loader import (
                get_paths_config,
                get_performance_config,
                get_shared_config,
                get_system_config,
            )
            from ..utils.error_handler import handle_config_error

            # Load shared config for common settings
            shared_config = handle_config_error(
                operation=get_shared_config,
                fallback_value={},
                config_file="config/config.toml",
                section="shared",
            )

            # Load system config
            system_config = handle_config_error(
                operation=get_system_config,
                fallback_value={},
                config_file="config/config.toml",
                section="system",
            )

            # Load paths config
            paths_config = handle_config_error(
                operation=get_paths_config,
                fallback_value={},
                config_file="config/config.toml",
                section="paths",
            )

            # Load performance config
            performance_config = handle_config_error(
                operation=get_performance_config,
                fallback_value={},
                config_file="config/config.toml",
                section="performance",
            )

            # Create component configs
            data = {
                "processing": ProcessingConfig.from_config(language=language),
                "embedding": EmbeddingConfig.from_config(),
                "chroma": ChromaConfig.from_config(),
                "retrieval": RetrievalConfig.from_config(language=language),
                "ollama": OllamaConfig.from_config(language=language),
                "language": LanguageConfig.from_config(language=language),
                "log_level": system_config.get("log_level", "INFO"),
                "enable_caching": system_config.get("enable_caching", True),
                "cache_dir": shared_config.get("cache_dir", "./data/cache"),
                "max_concurrent_requests": system_config.get("max_concurrent_requests", 5),
                "request_timeout": shared_config.get("request_timeout", 120.0),
                "enable_metrics": system_config.get("enable_metrics", True),
                "metrics_dir": shared_config.get("metrics_dir", "./data/metrics"),
                "documents_dir": paths_config.get("documents_dir", "./data/raw"),
                "processed_dir": paths_config.get("processed_dir", "./data/processed"),
                "test_data_dir": paths_config.get("test_data_dir", "./data/test"),
            }

        super().__init__(**data)
        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "__dict__"):
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary."""
        config = cls()

        # Update nested configs
        if "processing" in data:
            config.processing = ProcessingConfig(**data["processing"])
        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        if "chroma" in data:
            config.chroma = ChromaConfig(**data["chroma"])
        if "retrieval" in data:
            config.retrieval = RetrievalConfig(**data["retrieval"])
        if "ollama" in data:
            config.ollama = OllamaConfig(**data["ollama"])
        if "language" in data:
            config.language = LanguageConfig(**data["language"])

        # Update other fields
        for key, value in data.items():
            if key not in [
                "processing",
                "embedding",
                "chroma",
                "retrieval",
                "ollama",
                "language",
            ]:
                setattr(config, key, value)

        return config


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}
