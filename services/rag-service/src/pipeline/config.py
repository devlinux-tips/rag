"""
Configuration management for RAG system.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..utils.config_protocol import ConfigProvider

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
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        language: str = "hr",
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "ProcessingConfig":
        """Create config from dictionary with DRY error handling."""
        if config_dict:
            # Direct config provided
            processing_config = config_dict
            language_config = config_dict.get("language_specific", {})
            language_processing = language_config.get("processing", {})
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get configs through provider
            full_config = config_provider.load_config("config")
            processing_config = full_config["processing"]
            chunking_config = full_config["chunking"]

            # Get language-specific config
            language_config = config_provider.get_language_specific_config(
                "pipeline", language
            )
            language_processing = language_config.get("pipeline", {}).get(
                "processing", {}
            )

        return cls(
            max_chunk_size=chunking_config["max_chunk_size"],
            chunk_overlap=chunking_config.get(
                "chunk_overlap", 100
            ),  # Default if not present
            min_chunk_size=chunking_config.get(
                "min_chunk_size", 100
            ),  # Default if not present
            sentence_chunk_overlap=processing_config["sentence_chunk_overlap"],
            preserve_paragraphs=processing_config["preserve_paragraphs"],
            enable_smart_chunking=processing_config["enable_smart_chunking"],
            respect_document_structure=processing_config["respect_document_structure"],
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
    def from_config(
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "EmbeddingConfig":
        """Create config from dictionary with DRY error handling."""
        if config_dict:
            # Direct config provided
            embedding_config = config_dict
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get config through provider
            full_config = config_provider.load_config("config")
            embedding_config = full_config["embeddings"]

        return cls(
            model_name=embedding_config["model_name"],
            cache_folder=embedding_config.get(
                "cache_folder", "models/embeddings"
            ),  # Optional with default
            batch_size=embedding_config["batch_size"],
            max_seq_length=embedding_config["max_seq_length"],
            device=embedding_config["device"],
            normalize_embeddings=embedding_config["normalize_embeddings"],
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
    def from_config(
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "ChromaConfig":
        """Create config from dictionary with DRY error handling."""
        if config_dict:
            # Direct config provided
            chroma_config = config_dict
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get config through provider
            full_config = config_provider.load_config("config")
            chroma_config = full_config["chroma"]

        return cls(
            db_path=chroma_config["db_path"],
            collection_name=chroma_config["collection_name"],
            distance_metric=chroma_config["distance_metric"],
            ef_construction=chroma_config["ef_construction"],
            m=chroma_config["m"],
            persist=chroma_config["persist"],
            allow_reset=chroma_config["allow_reset"],
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
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        language: str = "hr",
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "RetrievalConfig":
        """Create config from dictionary with DRY error handling."""
        if config_dict:
            # Direct config provided
            retrieval_config = config_dict.get("retrieval", config_dict)
            language_config = config_dict.get("language_specific", {})
            language_retrieval = language_config.get("retrieval", {})
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get configs through provider
            full_config = config_provider.load_config("config")
            retrieval_config = full_config["retrieval"]

            # Get language-specific config
            language_config = config_provider.get_language_specific_config(
                "pipeline", language
            )
            language_retrieval = language_config.get("pipeline", {}).get(
                "retrieval", {}
            )

        return cls(
            default_k=retrieval_config["default_k"],
            max_k=retrieval_config["max_k"],
            min_similarity_score=retrieval_config["min_similarity_score"],
            adaptive_retrieval=retrieval_config["adaptive_retrieval"],
            enable_reranking=retrieval_config["enable_reranking"],
            diversity_lambda=retrieval_config["diversity_lambda"],
            use_hybrid_search=retrieval_config["use_hybrid_search"],
            enable_query_expansion=retrieval_config["enable_query_expansion"],
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
    stream: bool = True
    keep_alive: str = "5m"

    @classmethod
    def from_config(
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        language: str = "hr",
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "OllamaConfig":
        """Create config from dictionary with DRY error handling."""
        if config_dict:
            # Direct config provided
            ollama_config = config_dict.get("ollama", config_dict)
            language_config = config_dict.get("language_specific", {})
            language_generation = language_config.get("generation", {})
            language_shared = language_config.get("shared", {})
            pipeline_generation = language_config.get("pipeline", {}).get(
                "generation", {}
            )
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get configs through provider
            full_config = config_provider.load_config("config")
            ollama_config = full_config["ollama"]

            # Get language-specific config
            language_config = config_provider.get_language_specific_config(
                "pipeline", language
            )
            language_generation = language_config.get("generation", {})
            language_shared = language_config.get("shared", {})
            pipeline_generation = language_config.get("pipeline", {}).get(
                "generation", {}
            )

        return cls(
            base_url=ollama_config["base_url"],
            model=ollama_config["model"],
            temperature=ollama_config["temperature"],
            max_tokens=ollama_config["max_tokens"],
            top_p=ollama_config["top_p"],
            top_k=ollama_config["top_k"],
            timeout=ollama_config["timeout"],
            preserve_diacritics=pipeline_generation.get(
                "preserve_diacritics",
                language_shared.get(
                    "preserve_diacritics", language_generation["preserve_diacritics"]
                ),
            ),
            prefer_formal_style=pipeline_generation.get(
                "prefer_formal_style", language_generation["prefer_formal_style"]
            ),
            stream=ollama_config["stream"],
            keep_alive=ollama_config["keep_alive"],
        )


@dataclass
class LanguageConfig:
    """Language-specific configuration for multilingual support."""

    enable_morphological_expansion: bool
    enable_synonym_expansion: bool
    use_language_query_processing: bool
    language_priority: bool
    language_code: str = "hr"
    stop_words_file: str = "config/hr_stop_words.txt"
    morphology_patterns_file: str = "config/hr_morphology.json"

    @classmethod
    def from_config(
        cls,
        config_dict: Optional[Dict[str, Any]] = None,
        language: str = "hr",
        config_provider: Optional["ConfigProvider"] = None,
    ) -> "LanguageConfig":
        """Create config from dictionary with language support."""
        if config_dict:
            # Direct config provided
            language_config = config_dict
        else:
            # Use dependency injection - falls back to production provider
            if config_provider is None:
                from ..utils.config_protocol import get_config_provider

                config_provider = get_config_provider()

            # Get language-specific config through provider
            language_config = config_provider.get_language_specific_config(
                "pipeline", language
            )

        return cls(
            language_code=language,
            enable_morphological_expansion=language_config.get(
                "enable_morphological_expansion", True
            ),
            enable_synonym_expansion=language_config["enable_synonym_expansion"],
            use_language_query_processing=language_config.get(
                "use_language_query_processing", True
            ),
            language_priority=language_config["language_priority"],
            stop_words_file=language_config.get(
                "stop_words_file", f"config/{language}_stop_words.txt"
            ),
            morphology_patterns_file=language_config.get(
                "morphology_patterns_file", f"config/{language}_morphology.json"
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
            from ..utils.config_loader import (get_paths_config,
                                               get_performance_config,
                                               get_shared_config,
                                               get_system_config)

            # Load shared config for common settings
            shared_config = get_shared_config()

            # Load system config
            system_config = get_system_config()

            # Load paths config
            paths_config = get_paths_config()

            # Load performance config
            performance_config = get_performance_config()

            # Create component configs
            data = {
                "processing": ProcessingConfig.from_config(language=language),
                "embedding": EmbeddingConfig.from_config(),
                "chroma": ChromaConfig.from_config(),
                "retrieval": RetrievalConfig.from_config(language=language),
                "ollama": OllamaConfig.from_config(language=language),
                "language": LanguageConfig.from_config(language=language),
                "log_level": system_config["log_level"],
                "enable_caching": system_config["enable_caching"],
                "cache_dir": shared_config["cache_dir"],
                "max_concurrent_requests": system_config["max_concurrent_requests"],
                "request_timeout": shared_config["request_timeout"],
                "enable_metrics": system_config["enable_metrics"],
                "metrics_dir": shared_config["metrics_dir"],
                "documents_dir": paths_config["documents_dir"],
                "processed_dir": paths_config["processed_dir"],
                "test_data_dir": paths_config["test_data_dir"],
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
