"""
Vector database factory with config-driven provider selection.
Supports both ChromaDB and Weaviate with unified interface.
"""

from typing import Any, Protocol

from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start, log_decision_point
from .storage import VectorDatabase


class DatabaseFactory(Protocol):
    """Protocol for vector database factory."""

    def create_database(self, config: dict[str, Any], language: str) -> VectorDatabase:
        """Create vector database instance."""
        ...


def create_vector_database(config: dict[str, Any], language: str) -> VectorDatabase:
    """
    Create vector database instance based on configuration.

    Args:
        config: Configuration dictionary
        language: Language code for multi-tenant setup

    Returns:
        VectorDatabase instance

    Raises:
        ValueError: If provider is not supported or configuration is invalid
    """
    logger = get_system_logger()
    log_component_start("database_factory", "create_vector_database", language=language)

    # Direct access - fail if key missing (no fallbacks)
    provider = config["vectordb"]["provider"]

    log_decision_point("database_factory", "provider_selection", "provider", provider)

    if provider == "chromadb":
        database = _create_chromadb_database(config, language)
    elif provider == "weaviate":
        database = _create_weaviate_database(config, language)
    else:
        error_msg = f"Unsupported vector database provider: {provider}. Supported: chromadb, weaviate"
        logger.error("database_factory", "create_vector_database", error_msg)
        raise ValueError(error_msg)

    log_component_end(
        "database_factory", "create_vector_database", f"Created {provider} database for language {language}"
    )

    return database


def _create_chromadb_database(config: dict[str, Any], language: str) -> VectorDatabase:
    """Create ChromaDB database instance."""
    logger = get_system_logger()
    log_component_start("chromadb_factory", "create_database", language=language)

    # Import here to avoid circular imports
    from .chromadb_factories import create_chromadb_database

    database = create_chromadb_database(config, language)

    logger.debug("chromadb_factory", "create_database", "ChromaDB database created successfully")

    log_component_end("chromadb_factory", "create_database", "ChromaDB database instance created")

    return database


def _create_weaviate_database(config: dict[str, Any], language: str) -> VectorDatabase:
    """Create Weaviate database instance."""
    logger = get_system_logger()
    log_component_start("weaviate_factory", "create_database", language=language)

    # Import here to avoid circular imports
    from .weaviate_factories import create_weaviate_database

    database = create_weaviate_database(config, language)

    logger.debug("weaviate_factory", "create_database", "Weaviate database created successfully")

    log_component_end("weaviate_factory", "create_database", "Weaviate database instance created")

    return database


def validate_database_config(config: dict[str, Any]) -> list[str]:
    """
    Validate vector database configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    logger = get_system_logger()
    log_component_start("database_factory", "validate_config")

    errors = []

    try:
        # Check if vectordb section exists
        if "vectordb" not in config:
            errors.append("Missing 'vectordb' section in configuration")
            return errors

        vectordb_config = config["vectordb"]

        # Check provider
        if "provider" not in vectordb_config:
            errors.append("Missing 'vectordb.provider' in configuration")
        else:
            provider = vectordb_config["provider"]
            if provider not in ["chromadb", "weaviate"]:
                errors.append(f"Unsupported provider '{provider}'. Supported: chromadb, weaviate")

            # Provider-specific validation
            if provider == "chromadb":
                errors.extend(_validate_chromadb_config(config))
            elif provider == "weaviate":
                errors.extend(_validate_weaviate_config(config))

        # Common configuration validation
        required_common_keys = [
            "collection_name_template",
            "distance_metric",
            "batch_size",
            "timeout",
            "max_retries",
            "retry_delay",
        ]

        for key in required_common_keys:
            if key not in vectordb_config:
                errors.append(f"Missing 'vectordb.{key}' in configuration")

    except Exception as e:
        logger.error("database_factory", "validate_config", f"Validation error: {str(e)}")
        errors.append(f"Configuration validation failed: {str(e)}")

    if errors:
        logger.warning("database_factory", "validate_config", f"Found {len(errors)} validation errors")
    else:
        logger.debug("database_factory", "validate_config", "Configuration validation passed")

    log_component_end("database_factory", "validate_config", f"Validation completed with {len(errors)} errors")

    return errors


def _validate_chromadb_config(config: dict[str, Any]) -> list[str]:
    """Validate ChromaDB-specific configuration."""
    errors = []

    try:
        chromadb_config = config["vectordb"]["chromadb"]

        required_keys = [
            "db_path_template",
            "persist",
            "allow_reset",
            "anonymized_telemetry",
            "heartbeat_interval",
            "max_batch_size",
        ]

        for key in required_keys:
            if key not in chromadb_config:
                errors.append(f"Missing 'vectordb.chromadb.{key}' in configuration")

    except KeyError:
        errors.append("Missing 'vectordb.chromadb' section in configuration")

    return errors


def _validate_weaviate_config(config: dict[str, Any]) -> list[str]:
    """Validate Weaviate-specific configuration."""
    errors = []

    try:
        weaviate_config = config["vectordb"]["weaviate"]

        # Connection settings
        required_connection_keys = ["host", "port", "grpc_port", "scheme", "timeout", "startup_period"]

        for key in required_connection_keys:
            if key not in weaviate_config:
                errors.append(f"Missing 'vectordb.weaviate.{key}' in configuration")

        # HNSW index settings
        if "index" not in weaviate_config:
            errors.append("Missing 'vectordb.weaviate.index' section in configuration")
        else:
            index_config = weaviate_config["index"]
            required_index_keys = [
                "type",
                "ef_construction",
                "ef",
                "max_connections",
                "ef_dynamic",
                "cleanup_interval_seconds",
                "vector_cache_max_objects",
            ]

            for key in required_index_keys:
                if key not in index_config:
                    errors.append(f"Missing 'vectordb.weaviate.index.{key}' in configuration")

        # Compression settings
        if "compression" not in weaviate_config:
            errors.append("Missing 'vectordb.weaviate.compression' section in configuration")
        else:
            compression_config = weaviate_config["compression"]
            required_compression_keys = ["enabled", "type", "rescore_limit", "training_limit", "cache"]

            for key in required_compression_keys:
                if key not in compression_config:
                    errors.append(f"Missing 'vectordb.weaviate.compression.{key}' in configuration")

        # Backup settings
        if "backup" not in weaviate_config:
            errors.append("Missing 'vectordb.weaviate.backup' section in configuration")

    except KeyError:
        errors.append("Missing 'vectordb.weaviate' section in configuration")

    return errors


class VectorDatabaseFactory:
    """Factory class for creating vector database instances."""

    @staticmethod
    def create(config: dict[str, Any], language: str) -> VectorDatabase:
        """
        Create vector database instance.

        Args:
            config: Configuration dictionary
            language: Language code

        Returns:
            VectorDatabase instance

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration first
        validation_errors = validate_database_config(config)
        if validation_errors:
            error_msg = "Vector database configuration validation failed:\n" + "\n".join(
                f"- {error}" for error in validation_errors
            )
            raise ValueError(error_msg)

        return create_vector_database(config, language)

    @staticmethod
    def validate_config(config: dict[str, Any]) -> bool:
        """
        Validate configuration without creating database.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        validation_errors = validate_database_config(config)
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
            raise ValueError(error_msg)

        return True

    @staticmethod
    def get_supported_providers() -> list[str]:
        """Get list of supported database providers."""
        return ["chromadb", "weaviate"]


# Factory instance for easy access
vector_database_factory = VectorDatabaseFactory()
