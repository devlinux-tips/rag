"""
Weaviate configuration classes for enhanced vector database setup.
Provides comprehensive configuration models for Weaviate with HNSW optimization and compression.
"""

from dataclasses import dataclass
from typing import Any

from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start


@dataclass
class WeaviateHNSWConfig:
    """HNSW index configuration for Weaviate."""

    type: str = "hnsw"
    ef_construction: int = 128
    ef: int = -1
    max_connections: int = 32
    ef_dynamic: int = 100
    cleanup_interval_seconds: int = 300
    vector_cache_max_objects: int = 1000000

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WeaviateHNSWConfig":
        """Create HNSW config from configuration dictionary."""
        logger = get_system_logger()
        log_component_start("weaviate_hnsw_config", "from_config")

        # Direct access - fail if keys missing (no fallbacks)
        index_config = config["vectordb"]["weaviate"]["index"]

        hnsw_config = cls(
            type=index_config["type"],
            ef_construction=index_config["ef_construction"],
            ef=index_config["ef"],
            max_connections=index_config["max_connections"],
            ef_dynamic=index_config["ef_dynamic"],
            cleanup_interval_seconds=index_config["cleanup_interval_seconds"],
            vector_cache_max_objects=index_config["vector_cache_max_objects"],
        )

        logger.debug(
            "weaviate_hnsw_config",
            "from_config",
            f"HNSW: ef_construction={hnsw_config.ef_construction}, max_connections={hnsw_config.max_connections}",
        )

        log_component_end("weaviate_hnsw_config", "from_config", "HNSW configuration created")
        return hnsw_config

    def to_weaviate_config(self) -> dict[str, Any]:
        """Convert to Weaviate client configuration format."""
        return {
            "vectorIndexType": self.type,
            "vectorIndexConfig": {
                "efConstruction": self.ef_construction,
                "ef": self.ef,
                "maxConnections": self.max_connections,
                "dynamicEfMin": self.ef_dynamic,
                "cleanupIntervalSeconds": self.cleanup_interval_seconds,
                "vectorCacheMaxObjects": self.vector_cache_max_objects,
            },
        }


@dataclass
class WeaviateCompressionConfig:
    """Vector compression configuration for Weaviate."""

    enabled: bool = False
    type: str = "pq"
    rescore_limit: int = 100
    training_limit: int = 100000
    cache: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WeaviateCompressionConfig":
        """Create compression config from configuration dictionary."""
        logger = get_system_logger()
        log_component_start("weaviate_compression_config", "from_config")

        # Direct access - fail if keys missing (no fallbacks)
        compression_config = config["vectordb"]["weaviate"]["compression"]

        config_obj = cls(
            enabled=compression_config["enabled"],
            type=compression_config["type"],
            rescore_limit=compression_config["rescore_limit"],
            training_limit=compression_config["training_limit"],
            cache=compression_config["cache"],
        )

        logger.debug(
            "weaviate_compression_config",
            "from_config",
            f"Compression: {config_obj.type}, enabled={config_obj.enabled}",
        )

        log_component_end("weaviate_compression_config", "from_config", "Compression configuration created")
        return config_obj

    def to_weaviate_config(self) -> dict[str, Any] | None:
        """Convert to Weaviate client configuration format."""
        if not self.enabled:
            return None

        return {
            "vectorCompressionType": self.type.upper(),  # SQ or PQ
            "rescoreLimit": self.rescore_limit,
            "trainingLimit": self.training_limit,
            "cache": self.cache,
        }


@dataclass
class WeaviateBackupConfig:
    """Backup configuration for Weaviate."""

    enabled: bool = False
    backend: str = "filesystem"
    backup_id: str = "default"
    include_meta: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WeaviateBackupConfig":
        """Create backup config from configuration dictionary."""
        logger = get_system_logger()
        log_component_start("weaviate_backup_config", "from_config")

        # Direct access - fail if keys missing (no fallbacks)
        backup_config = config["vectordb"]["weaviate"]["backup"]

        config_obj = cls(
            enabled=backup_config["enabled"],
            backend=backup_config["backend"],
            backup_id=backup_config["backup_id"],
            include_meta=backup_config["include_meta"],
        )

        logger.debug(
            "weaviate_backup_config",
            "from_config",
            f"Backup: enabled={config_obj.enabled}, backend={config_obj.backend}",
        )

        log_component_end("weaviate_backup_config", "from_config", "Backup configuration created")
        return config_obj


@dataclass
class WeaviateConnectionConfig:
    """Connection configuration for Weaviate."""

    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051
    scheme: str = "http"
    timeout: float = 30.0
    startup_period: int = 5
    additional_headers: dict[str, str] | None = None

    def __post_init__(self):
        """Initialize additional headers if None."""
        if self.additional_headers is None:
            self.additional_headers = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WeaviateConnectionConfig":
        """Create connection config from configuration dictionary."""
        logger = get_system_logger()
        log_component_start("weaviate_connection_config", "from_config")

        # Direct access - fail if keys missing (no fallbacks)
        weaviate_config = config["vectordb"]["weaviate"]

        # Additional headers are optional
        additional_headers = {}
        if "additional_headers" in weaviate_config:
            additional_headers = weaviate_config["additional_headers"]

        config_obj = cls(
            host=weaviate_config["host"],
            port=weaviate_config["port"],
            grpc_port=weaviate_config["grpc_port"],
            scheme=weaviate_config["scheme"],
            timeout=float(weaviate_config["timeout"]),
            startup_period=weaviate_config["startup_period"],
            additional_headers=additional_headers,
        )

        logger.debug(
            "weaviate_connection_config",
            "from_config",
            f"Connection: {config_obj.scheme}://{config_obj.host}:{config_obj.port}",
        )

        log_component_end("weaviate_connection_config", "from_config", "Connection configuration created")
        return config_obj

    @property
    def url(self) -> str:
        """Get the full Weaviate URL."""
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def grpc_url(self) -> str:
        """Get the gRPC URL."""
        return f"{self.host}:{self.grpc_port}"


@dataclass
class WeaviateGeneralConfig:
    """General Weaviate configuration."""

    vectorizer: str = "none"
    collection_name_template: str = "{tenant}_{language}_collection"
    distance_metric: str = "cosine"
    batch_size: int = 100
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WeaviateGeneralConfig":
        """Create general config from configuration dictionary."""
        logger = get_system_logger()
        log_component_start("weaviate_general_config", "from_config")

        # Direct access - fail if keys missing (no fallbacks)
        vectordb_config = config["vectordb"]
        weaviate_config = vectordb_config["weaviate"]

        config_obj = cls(
            vectorizer=weaviate_config["vectorizer"],
            collection_name_template=vectordb_config["collection_name_template"],
            distance_metric=vectordb_config["distance_metric"],
            batch_size=vectordb_config["batch_size"],
            timeout=float(vectordb_config["timeout"]),
            max_retries=vectordb_config["max_retries"],
            retry_delay=float(vectordb_config["retry_delay"]),
        )

        logger.debug(
            "weaviate_general_config",
            "from_config",
            f"Vectorizer: {config_obj.vectorizer}, distance_metric: {config_obj.distance_metric}",
        )

        log_component_end("weaviate_general_config", "from_config", "General configuration created")
        return config_obj

    def format_collection_name(self, tenant: str, user: str, language: str) -> str:
        """Format collection name using template."""
        return self.collection_name_template.format(tenant=tenant, user=user, language=language)


@dataclass
class WeaviateConfiguration:
    """Complete Weaviate configuration."""

    connection: WeaviateConnectionConfig
    general: WeaviateGeneralConfig
    hnsw: WeaviateHNSWConfig
    compression: WeaviateCompressionConfig
    backup: WeaviateBackupConfig

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WeaviateConfiguration":
        """Create complete Weaviate configuration from config dictionary."""
        logger = get_system_logger()
        log_component_start("weaviate_configuration", "from_config", provider=config["vectordb"]["provider"])

        # Validate that Weaviate is selected as provider
        provider = config["vectordb"]["provider"]
        if provider != "weaviate":
            raise ValueError(f"Expected vectordb.provider='weaviate', got '{provider}'")

        config_obj = cls(
            connection=WeaviateConnectionConfig.from_config(config),
            general=WeaviateGeneralConfig.from_config(config),
            hnsw=WeaviateHNSWConfig.from_config(config),
            compression=WeaviateCompressionConfig.from_config(config),
            backup=WeaviateBackupConfig.from_config(config),
        )

        logger.info(
            "weaviate_configuration", "from_config", f"Weaviate configuration loaded: {config_obj.connection.url}"
        )

        log_component_end("weaviate_configuration", "from_config", "Complete Weaviate configuration created")

        return config_obj

    def get_client_config(self) -> dict[str, Any]:
        """Get configuration suitable for Weaviate client initialization."""
        client_config = {
            "url": self.connection.url,
            "grpc_port": self.connection.grpc_port,
            "timeout_config": self.connection.timeout,
            "startup_period": self.connection.startup_period,
            "additional_headers": self.connection.additional_headers,
        }

        return client_config

    def get_class_config(self, class_name: str) -> dict[str, Any]:
        """Get configuration for creating a Weaviate class/collection."""
        class_config = {
            "class": class_name,
            "vectorizer": self.general.vectorizer,
            "vectorIndexType": self.hnsw.type,
            "vectorIndexConfig": self.hnsw.to_weaviate_config()["vectorIndexConfig"],
        }

        # Add compression if enabled
        compression_config = self.compression.to_weaviate_config()
        if compression_config:
            class_config.update(compression_config)

        return class_config

    def validate_configuration(self) -> list[str]:
        """
        Validate the configuration and return any issues found.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate connection settings
        if not self.connection.host:
            errors.append("Weaviate host cannot be empty")

        if self.connection.port <= 0 or self.connection.port > 65535:
            errors.append(f"Invalid Weaviate port: {self.connection.port}")

        if self.connection.grpc_port <= 0 or self.connection.grpc_port > 65535:
            errors.append(f"Invalid Weaviate gRPC port: {self.connection.grpc_port}")

        if self.connection.scheme not in ["http", "https"]:
            errors.append(f"Invalid scheme: {self.connection.scheme}")

        # Validate vectorizer
        if self.general.vectorizer not in ["none", "text2vec-contextionary", "text2vec-transformers"]:
            errors.append(f"Unsupported vectorizer: {self.general.vectorizer}")

        # Validate HNSW settings
        if self.hnsw.ef_construction < 16:
            errors.append("ef_construction should be at least 16 for good performance")

        if self.hnsw.max_connections < 16:
            errors.append("max_connections should be at least 16")

        # Validate compression settings
        if self.compression.enabled and self.compression.type not in ["sq", "pq"]:
            errors.append(f"Invalid compression type: {self.compression.type}")

        # Validate distance metric
        valid_metrics = ["cosine", "dot", "l2-squared", "manhattan", "hamming"]
        if self.general.distance_metric not in valid_metrics:
            errors.append(f"Invalid distance metric: {self.general.distance_metric}")

        return errors


def create_weaviate_configuration(config: dict[str, Any]) -> WeaviateConfiguration:
    """
    Factory function to create Weaviate configuration with validation.

    Args:
        config: Configuration dictionary

    Returns:
        Validated WeaviateConfiguration instance

    Raises:
        ValueError: If configuration is invalid
    """
    logger = get_system_logger()
    log_component_start("weaviate_config_factory", "create_configuration")

    weaviate_config = WeaviateConfiguration.from_config(config)

    # Validate configuration
    validation_errors = weaviate_config.validate_configuration()
    if validation_errors:
        error_msg = "Weaviate configuration validation failed:\n" + "\n".join(
            f"- {error}" for error in validation_errors
        )
        logger.error("weaviate_config_factory", "create_configuration", error_msg)
        raise ValueError(error_msg)

    log_component_end("weaviate_config_factory", "create_configuration", "Weaviate configuration created and validated")

    return weaviate_config
