"""
Database Provider Factory for RAG System
Creates appropriate database provider based on configuration.

Supports fail-fast validation and AI-friendly logging.
"""

from typing import Dict, Any

from .protocols import DatabaseProvider
from .providers.postgresql_provider import PostgreSQLProvider
from .providers.surrealdb_provider import SurrealDBProvider
from .providers.supabase_provider import SupabaseProvider
from ..utils.error_handler import ConfigurationError
from ..utils.logging_factory import get_system_logger, log_component_start, log_component_end


def create_database_provider(config: Dict[str, Any]) -> DatabaseProvider:
    """
    Factory function to create database provider based on configuration.

    Args:
        config: Database configuration dictionary

    Returns:
        DatabaseProvider: Initialized database provider

    Raises:
        ConfigurationError: If provider type invalid or config missing
    """
    logger = get_system_logger()

    log_component_start(
        "database_factory", "create_provider",
        config_keys=list(config.keys())
    )

    # Validate provider type exists
    provider_type = config.get("provider")
    if not provider_type:
        error_msg = "Database provider must be specified in config.database.provider"
        logger.error("database_factory", "create_provider", error_msg)
        raise ConfigurationError(error_msg)

    # Provider registry
    providers = {
        "postgresql": PostgreSQLProvider,
        "surrealdb": SurrealDBProvider,
        "supabase": SupabaseProvider
    }

    if provider_type not in providers:
        supported = list(providers.keys())
        error_msg = f"Unsupported database provider: {provider_type}. Supported: {supported}"
        logger.error("database_factory", "create_provider", error_msg)
        raise ConfigurationError(error_msg)

    # Create provider instance
    provider_class = providers[provider_type]
    provider = provider_class()

    logger.info(
        "database_factory", "create_provider",
        f"Created {provider_type} provider: {provider_class.__name__}"
    )

    log_component_end(
        "database_factory", "create_provider",
        f"Provider created: {provider_type}"
    )

    return provider


def validate_provider_config(config: Dict[str, Any], provider_type: str) -> None:
    """
    Validate provider-specific configuration.

    Args:
        config: Database configuration
        provider_type: Provider type to validate

    Raises:
        ConfigurationError: If required config missing
    """
    logger = get_system_logger()

    if provider_type == "postgresql":
        required_keys = ["host", "database", "user", "password"]
        provider_config = config.get("postgresql", {})
    elif provider_type == "surrealdb":
        required_keys = ["url", "namespace", "database", "username", "password"]
        provider_config = config.get("surrealdb", {})
    elif provider_type == "supabase":
        required_keys = ["url", "service_role_key", "anon_key"]
        provider_config = config.get("supabase", {})
    else:
        raise ConfigurationError(f"Unknown provider type for validation: {provider_type}")

    missing_keys = [key for key in required_keys if key not in provider_config]
    if missing_keys:
        error_msg = f"Missing required {provider_type} config keys: {missing_keys}"
        logger.error("database_factory", "validate_config", error_msg)
        raise ConfigurationError(error_msg)

    logger.debug(
        "database_factory", "validate_config",
        f"{provider_type} configuration validated: {len(required_keys)} keys"
    )