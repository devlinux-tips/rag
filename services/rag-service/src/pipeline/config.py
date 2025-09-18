"""
Configuration management for RAG system.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..utils.config_models import (
    ChromaConfig,
    EmbeddingConfig,
    LanguageConfig,
    OllamaConfig,
    ProcessingConfig,
    RetrievalConfig,
)
from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_data_transformation,
    log_error_context,
    log_performance_metric,
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

    # Use SettingsConfigDict for modern Pydantic settings compatibility
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def __init__(self, language: str = "hr", **data):
        """Initialize with validated config loading."""
        # If no data provided, load from config files
        if not data:
            from ..utils.config_loader import get_paths_config, get_shared_config, get_system_config
            from ..utils.config_protocol import get_config_provider

            # Get validated config through provider
            config_provider = get_config_provider()
            main_config = config_provider.load_config("config")  # Get full structured config

            # Get language-specific config for embeddings
            language_config = config_provider.get_language_config(language)

            # Load additional configs for direct system settings
            shared_config = get_shared_config()
            system_config = get_system_config()
            paths_config = get_paths_config()

            # Build compatible chroma config from storage and vectordb sections
            storage_config = main_config["storage"]
            vectordb_config = main_config["vectordb"]["factory"]

            # Create a chroma section that matches ChromaConfig expectations
            main_config["chroma"] = {
                "db_path": "./data/vectordb",  # Default path, will be updated per tenant
                "collection_name": f"{language}_documents",  # Default, will be updated per tenant
                "distance_metric": storage_config["distance_metric"],
                "chunk_size": vectordb_config["chunk_size"],
                "ef_construction": 200,  # HNSW default
                "m": 16,  # HNSW default
                "persist": storage_config["persist"],
                "allow_reset": storage_config["allow_reset"],
            }

            # Promote shared config values to root level for config models that expect them there
            for key, value in shared_config.items():
                if key not in main_config:  # Don't override existing sections
                    main_config[key] = value

            # Create component configs using validated approach
            data = {
                "processing": ProcessingConfig.from_validated_config(main_config),
                "embedding": EmbeddingConfig.from_validated_config(main_config, language_config),
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
            if key not in ["processing", "embedding", "chroma", "retrieval", "ollama", "language"]:
                setattr(config, key, value)

        return config


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    logger = get_system_logger()
    log_component_start(
        "config_loader", "load_yaml_config", config_path=config_path, file_exists=Path(config_path).exists()
    )

    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            config_size = len(str(config_data)) if config_data else 0
            config_keys = len(config_data) if isinstance(config_data, dict) else 0

            log_performance_metric("config_loader", "load_yaml_config", "config_keys_count", config_keys)
            log_performance_metric("config_loader", "load_yaml_config", "config_size_chars", config_size)

            log_data_transformation(
                "config_loader",
                "yaml_parsing",
                f"Input: YAML file at {config_path}",
                f"Loaded configuration with {config_keys} top-level keys",
                file_path=config_path,
                keys_loaded=config_keys,
                config_size=config_size,
            )

            logger.info(
                "config_loader",
                "load_yaml_config",
                f"Successfully loaded YAML config from {config_path}: {config_keys} keys",
            )

            log_component_end(
                "config_loader",
                "load_yaml_config",
                f"Configuration loaded successfully: {config_keys} keys",
                config_keys=config_keys,
                file_path=config_path,
            )

            return config_data

        except yaml.YAMLError as e:
            log_error_context(
                "config_loader", "load_yaml_config", e, {"file_path": config_path, "error_type": "yaml_error"}
            )
            logger.error("config_loader", "load_yaml_config", f"Failed to parse YAML file {config_path}: {e}")
            log_component_end("config_loader", "load_yaml_config", f"Failed to parse YAML: {e}")
            raise
        except Exception as e:
            log_error_context(
                "config_loader", "load_yaml_config", e, {"file_path": config_path, "error_type": "file_error"}
            )
            logger.error("config_loader", "load_yaml_config", f"Failed to read config file {config_path}: {e}")
            log_component_end("config_loader", "load_yaml_config", f"Failed to read file: {e}")
            raise
    else:
        logger.warning(
            "config_loader", "load_yaml_config", f"Config file not found: {config_path}, returning empty config"
        )
        log_component_end(
            "config_loader",
            "load_yaml_config",
            "File not found, returning empty config",
            file_path=config_path,
            file_exists=False,
        )
        return {}
