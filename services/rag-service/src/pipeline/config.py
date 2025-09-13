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
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

from ..utils.config_models import (
    ChromaConfig,
    EmbeddingConfig,
    LanguageConfig,
    OllamaConfig,
    ProcessingConfig,
    RetrievalConfig,
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

    # Use ConfigDict for modern Pydantic compatibility
    if ConfigDict:
        model_config = ConfigDict(env_file=".env", extra="allow")
    else:

        class Config:
            env_file = ".env"
            extra = "allow"

    def __init__(self, language: str = "hr", **data):
        """Initialize with validated config loading."""
        # If no data provided, load from config files
        if not data:
            from ..utils.config_loader import (
                get_paths_config,
                get_performance_config,
                get_shared_config,
                get_system_config,
            )
            from ..utils.config_protocol import get_config_provider

            # Get validated config through provider
            config_provider = get_config_provider()
            main_config = config_provider.get_main_config()

            # Get language-specific config for embeddings
            language_config = config_provider.get_language_config(language)

            # Load additional configs for direct system settings
            shared_config = get_shared_config()
            system_config = get_system_config()
            paths_config = get_paths_config()

            # Create component configs using validated approach
            data = {
                "processing": ProcessingConfig.from_validated_config(main_config),
                "embedding": EmbeddingConfig.from_validated_config(
                    main_config, language_config
                ),
                "chroma": ChromaConfig.from_validated_config(main_config),
                "retrieval": RetrievalConfig.from_validated_config(main_config),
                "ollama": OllamaConfig.from_validated_config(main_config),
                "language": LanguageConfig.from_validated_config(main_config, language),
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

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "__dict__"):
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], language: str = "hr") -> "RAGConfig":
        """Create config from dictionary using validated approach."""
        config = cls(language=language)

        # Update nested configs - direct instantiation since data is already validated
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


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}
