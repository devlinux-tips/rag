"""
Centralized Configuration Loader for Croatian RAG Project

This module provides a unified interface for loading TOML configuration files
with Croatian language settings integration and environment-specific overrides.

Usage:
    from src.utils.config_loader import load_config, get_croatian_settings

    # Load specific config
    ollama_config = load_config('ollama')

    # Get Croatian settings
    croatian = get_croatian_settings()

    # Load with environment override
    config = load_config('main', environment='production')
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions

logger = logging.getLogger(__name__)

# Configuration directory path
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class ConfigError(Exception):
    """Raised when configuration loading fails."""

    pass


class ConfigLoader:
    """Centralized configuration loader for TOML files."""

    def __init__(self, config_dir: Path = CONFIG_DIR):
        """Initialize config loader with directory path."""
        self.config_dir = config_dir
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Validate config directory exists
        if not self.config_dir.exists():
            raise ConfigError(f"Configuration directory not found: {self.config_dir}")

    def load(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a TOML configuration file.

        Args:
            config_name: Name of config file (without .toml extension)
            use_cache: Whether to use cached config if available

        Returns:
            Dictionary containing configuration data

        Raises:
            ConfigError: If config file not found or invalid TOML
        """
        if use_cache and config_name in self._cache:
            logger.debug(f"Using cached config: {config_name}")
            return self._cache[config_name]

        # Use consolidated config.toml for most configurations, croatian.toml for Croatian-specific
        if config_name == "croatian":
            config_path = self.config_dir / "croatian.toml"
        else:
            config_path = self.config_dir / "config.toml"

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)

            # For non-Croatian configs, we need to return the specific section from the consolidated config
            if config_name != "croatian":
                # Map config names to their sections in the consolidated config
                section_mapping = {
                    "vectordb": {
                        "embeddings": config_data.get("embeddings", {}),
                        "storage": config_data.get("storage", {}),
                        "search": config_data.get("search", {}),
                        "factory": config_data.get("vectordb", {}).get("factory", {}),
                    },
                    "generation": {
                        "ollama": config_data.get("ollama", {}),
                        "prompts": config_data.get("prompts", {}),
                        "response_parsing": config_data.get("response_parsing", {}),
                    },
                    "preprocessing": {
                        "processing": config_data.get("processing", {}),
                        "extraction": config_data.get("extraction", {}),
                        "chunking": config_data.get("chunking", {}),
                        "cleaning": config_data.get("cleaning", {}),
                        "formatting_patterns": config_data.get("formatting_patterns", {}),
                    },
                    "retrieval": {
                        "query_processing": config_data.get("query_processing", {}),
                        "retrieval": config_data.get("retrieval", {}),
                        "ranking": config_data.get("ranking", {}),
                        "reranking": config_data.get("reranking", {}),
                        "hybrid_retrieval": config_data.get("hybrid_retrieval", {}),
                    },
                    "pipeline": {
                        "processing": config_data.get("processing", {}),
                        "embedding": config_data.get("embeddings", {}),
                        "chroma": config_data.get("storage", {}),
                        "retrieval": config_data.get("retrieval", {}),
                        "ollama": config_data.get("ollama", {}),
                        "system": config_data.get("system", {}),
                        "paths": config_data.get("paths", {}),
                        "performance": config_data.get("performance", {}),
                        "pipeline": config_data.get("pipeline", {}),
                    },
                    "main": None,  # Return full config for main
                    "config": None,  # Return full config for config
                }

                if config_name in section_mapping and section_mapping[config_name] is not None:
                    config_data = section_mapping[config_name]
                # If config_name is not mapped or maps to None, return the full config

            self._cache[config_name] = config_data
            logger.info(f"Loaded configuration: {config_name}")
            return config_data

        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"Invalid TOML in {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load {config_path}: {e}")

    def get_section(self, config_name: str, section: str) -> Dict[str, Any]:
        """
        Get a specific section from a config file.

        Args:
            config_name: Name of config file
            section: Section name to extract

        Returns:
            Dictionary containing section data

        Raises:
            ConfigError: If section not found
        """
        config = self.load(config_name)

        if section not in config:
            raise ConfigError(f"Section '{section}' not found in {config_name}.toml")

        return config[section]

    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configuration files.

        Args:
            config_names: Names of config files to merge

        Returns:
            Merged configuration dictionary
        """
        merged = {}

        for config_name in config_names:
            config = self.load(config_name)
            merged.update(config)

        return merged

    def clear_cache(self):
        """Clear the configuration cache."""
        self._cache.clear()
        logger.debug("Configuration cache cleared")


# Global config loader instance
_config_loader = ConfigLoader()


def load_config(config_name: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Load a configuration file using the global config loader.

    Args:
        config_name: Name of config file (without .toml extension)
        use_cache: Whether to use cached config if available

    Returns:
        Dictionary containing configuration data
    """
    return _config_loader.load(config_name, use_cache=use_cache)


def get_config_section(config_name: str, section: str) -> Dict[str, Any]:
    """
    Get a specific section from a config file.

    Args:
        config_name: Name of config file
        section: Section name to extract

    Returns:
        Dictionary containing section data
    """
    return _config_loader.get_section(config_name, section)


def get_croatian_settings() -> Dict[str, Any]:
    """
    Get Croatian language settings.

    Returns:
        Dictionary containing Croatian configuration
    """
    return load_config("croatian")


def get_croatian_prompts() -> Dict[str, str]:
    """
    Get Croatian prompt templates.

    Returns:
        Dictionary containing Croatian prompts
    """
    return get_config_section("croatian", "prompts")


def get_croatian_language_code() -> str:
    """
    Get Croatian language code.

    Returns:
        Language code string (e.g., 'hr')
    """
    return get_config_section("croatian", "language")["code"]


def get_croatian_confidence_settings() -> Dict[str, Any]:
    """
    Get Croatian confidence calculation settings.

    Returns:
        Dictionary containing confidence settings
    """
    return get_config_section("croatian", "confidence")


def get_croatian_formal_prompts() -> Dict[str, str]:
    """
    Get Croatian formal prompt templates.

    Returns:
        Dictionary containing formal prompts
    """
    return get_config_section("croatian", "formal_prompts")


def get_generation_config() -> Dict[str, Any]:
    """
    Get generation module configuration.

    Returns:
        Dictionary containing generation config
    """
    return load_config("generation")


def get_ollama_config() -> Dict[str, Any]:
    """
    Get Ollama-specific configuration.

    Returns:
        Dictionary containing Ollama config
    """
    return get_config_section("generation", "ollama")


def get_response_parsing_config() -> Dict[str, Any]:
    """Get response parsing configuration."""
    generation_config = get_generation_config()
    return generation_config["response_parsing"]


def get_preprocessing_config() -> Dict[str, Any]:
    """Get preprocessing configuration."""
    return _config_loader.load("preprocessing")


def get_extraction_config() -> Dict[str, Any]:
    """Get extraction configuration."""
    preprocessing_config = get_preprocessing_config()
    return preprocessing_config["extraction"]


def get_chunking_config() -> Dict[str, Any]:
    """Get chunking configuration."""
    preprocessing_config = get_preprocessing_config()
    return preprocessing_config["chunking"]


def get_cleaning_config() -> Dict[str, Any]:
    """Get cleaning configuration."""
    preprocessing_config = get_preprocessing_config()
    return preprocessing_config["cleaning"]


def get_croatian_text_processing() -> Dict[str, Any]:
    """Get Croatian text processing configuration."""
    croatian_config = get_croatian_settings()
    return croatian_config["text_processing"]


def get_croatian_document_cleaning() -> Dict[str, Any]:
    """Get Croatian document cleaning configuration."""
    croatian_config = get_croatian_settings()
    return croatian_config["document_cleaning"]


def get_croatian_chunking() -> Dict[str, Any]:
    """Get Croatian chunking configuration."""
    croatian_config = get_croatian_settings()
    return croatian_config["chunking"]


def get_generation_prompts_config() -> Dict[str, Any]:
    """
    Get generation prompts configuration.

    Returns:
        Dictionary containing prompts config
    """
    return get_config_section("generation", "prompts")


def merge_configs(*config_names: str) -> Dict[str, Any]:
    """
    Merge multiple configuration files.

    Args:
        config_names: Names of config files to merge

    Returns:
        Merged configuration dictionary
    """
    return _config_loader.merge_configs(*config_names)


def reload_config(config_name: str) -> Dict[str, Any]:
    """
    Force reload a configuration file (bypass cache).

    Args:
        config_name: Name of config file to reload

    Returns:
        Reloaded configuration dictionary
    """
    return _config_loader.load(config_name, use_cache=False)


def get_project_info() -> Dict[str, Any]:
    """
    Get main project information.

    Returns:
        Dictionary containing project configuration
    """
    return get_config_section("main", "project")


def get_paths_config() -> Dict[str, str]:
    """
    Get project paths configuration.

    Returns:
        Dictionary containing path configurations
    """
    return get_config_section("main", "paths")


# Vector Database Configuration Functions
def get_vectordb_config() -> Dict[str, Any]:
    """Get complete vectordb configuration."""
    return load_config("vectordb")


def get_embeddings_config() -> Dict[str, Any]:
    """Get embeddings configuration."""
    vectordb_config = get_vectordb_config()
    return vectordb_config["embeddings"]


def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration."""
    vectordb_config = get_vectordb_config()
    return vectordb_config["storage"]


def get_search_config() -> Dict[str, Any]:
    """Get search configuration."""
    vectordb_config = get_vectordb_config()
    return vectordb_config["search"]


def get_croatian_vectordb() -> Dict[str, Any]:
    """Get Croatian vectordb configuration."""
    croatian_config = get_croatian_settings()
    return croatian_config["vectordb"]


# Retrieval Configuration Functions
def get_retrieval_config() -> Dict[str, Any]:
    """Get main retrieval configuration."""
    return _config_loader.load("retrieval")


def get_query_processing_config() -> Dict[str, Any]:
    """Get query processing configuration."""
    retrieval_config = get_retrieval_config()
    return retrieval_config["query_processing"]


def get_ranking_config() -> Dict[str, Any]:
    """Get ranking configuration."""
    retrieval_config = get_retrieval_config()
    return retrieval_config["ranking"]


def get_reranking_config() -> Dict[str, Any]:
    """Get reranking configuration."""
    retrieval_config = get_retrieval_config()
    return retrieval_config["reranking"]


def get_hybrid_retrieval_config() -> Dict[str, Any]:
    """Get hybrid retrieval configuration."""
    retrieval_config = get_retrieval_config()
    return retrieval_config["hybrid_retrieval"]


def get_croatian_retrieval() -> Dict[str, Any]:
    """Get Croatian retrieval configuration."""
    croatian_config = get_croatian_settings()
    return croatian_config.get("retrieval", {})


# Pipeline configuration functions
def get_pipeline_config() -> Dict[str, Any]:
    """Get complete pipeline configuration."""
    return load_config("pipeline")


def get_processing_config() -> Dict[str, Any]:
    """Get document processing configuration."""
    pipeline_config = get_pipeline_config()
    return pipeline_config.get("processing", {})


def get_chroma_config() -> Dict[str, Any]:
    """Get ChromaDB configuration."""
    pipeline_config = get_pipeline_config()
    return pipeline_config.get("chroma", {})


def get_croatian_response_parsing_config() -> Dict[str, Any]:
    """Get Croatian response parsing configuration."""
    croatian_config = get_croatian_settings()
    return croatian_config.get("response_parsing", {})


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    pipeline_config = get_pipeline_config()
    return pipeline_config.get("performance", {})


def get_system_config() -> Dict[str, Any]:
    """Get system configuration."""
    main_config = load_config("config")
    return main_config.get("system", {})


def get_croatian_pipeline() -> Dict[str, Any]:
    """Get Croatian pipeline configuration."""
    croatian_config = get_croatian_settings()
    return croatian_config.get("pipeline", {})
