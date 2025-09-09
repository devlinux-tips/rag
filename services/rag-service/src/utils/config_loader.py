"""
Centralized Configuration Loader for Multilingual RAG Project

This module provides a unified interface for loading TOML configuration files
with multilingual language settings (Croatian and English) and environment-specific overrides.

Usage:
    from src.utils.config_loader import load_config, get_language_config

    # Load specific config
    ollama_config = load_config('ollama')

    # Get language-specific settings
    croatian = get_language_config('hr')
    english = get_language_config('en')

    # Get shared language configuration
    croatian_shared = get_language_shared('hr')
    english_shared = get_language_shared('en')

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

    def _get_language_config_files(self) -> Dict[str, str]:
        """Get available language config files dynamically."""
        try:
            # Load main config to get language mappings
            main_config_path = self.config_dir / "config.toml"
            if main_config_path.exists():
                with open(main_config_path, "rb") as f:
                    main_config = tomllib.load(f)
                    return main_config["languages"].get("config_files", {})
        except Exception:
            pass

        # Fallback: return known language files
        return {"croatian": "croatian.toml", "english": "english.toml"}

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

        # Use dynamic language config file mapping
        language_config_files = self._get_language_config_files()
        language_file_names = [
            f.replace(".toml", "") for f in language_config_files.values()
        ]

        if config_name in language_file_names:
            # This is a language-specific config (croatian, english, etc.)
            config_path = self.config_dir / f"{config_name}.toml"
        else:
            # This is a main system config (config, main, etc.)
            config_path = self.config_dir / "config.toml"

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)

            # All configs return their full data - no special section mapping needed

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


def get_shared_config() -> Dict[str, Any]:
    """Get shared configuration from main config (constants and common settings)."""
    main_config = load_config("config")
    return main_config["shared"]


# ============================================================================
# MULTILINGUAL CONFIGURATION FUNCTIONS
# ============================================================================


def get_language_config(language: str) -> Dict[str, Any]:
    """
    Get configuration for specified language.

    Args:
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Dictionary containing language-specific configuration

    Raises:
        ConfigError: If language is not supported
    """
    if not is_language_supported(language):
        supported = get_supported_languages()
        raise ConfigError(
            f"Unsupported language: {language}. Supported languages: {', '.join(supported)}"
        )

    # Get config file name for language
    config_file = get_language_config_file(language)
    config_name = config_file.replace(".toml", "")

    return load_config(config_name)


def get_language_shared(language: str) -> Dict[str, Any]:
    """
    Get shared configuration for specified language.

    Args:
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Dictionary containing language-specific shared configuration
    """
    config = get_language_config(language)
    return config["shared"]


def get_language_specific_config(section: str, language: str) -> Dict[str, Any]:
    """
    Get specific configuration section for specified language.

    Args:
        section: Configuration section name (e.g., 'prompts', 'retrieval')
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Dictionary containing language-specific section configuration
    """
    config = get_language_config(language)
    return config.get(section, {})


def get_supported_languages() -> list[str]:
    """
    Get list of supported languages from configuration.

    Returns:
        List of supported language codes
    """
    try:
        main_config = load_config("config")
        return main_config["languages"].get("supported", ["hr"])
    except Exception:
        # Fallback to Croatian only
        return ["hr"]


def get_language_config_file(language: str) -> str:
    """
    Get config file name for language using convention-based naming.

    Args:
        language: Language code (e.g., 'hr', 'en')

    Returns:
        Config file name (e.g., 'hr.toml', 'en.toml')
    """
    # Convention: language code directly maps to filename
    return f"{language}.toml"


def is_language_supported(language: str) -> bool:
    """
    Check if language is supported.

    Args:
        language: Language code to check

    Returns:
        True if language is supported, False otherwise
    """
    supported = get_supported_languages()
    return language.lower() in [lang.lower() for lang in supported]


def discover_available_languages() -> list[str]:
    """
    Scan config directory for available language files.

    Returns:
        List of available language config file names (without .toml)
    """
    language_files = []
    for file in CONFIG_DIR.glob("*.toml"):
        if file.name not in ["config.toml"]:
            language_files.append(file.stem)
    return language_files


def validate_language_configuration() -> Dict[str, str]:
    """
    Validate that all supported languages have config files.

    Returns:
        Dictionary mapping languages to their config files

    Raises:
        ConfigError: If any supported language is missing its config file
    """
    supported = get_supported_languages()
    available = discover_available_languages()

    missing = []
    valid_mapping = {}

    for lang in supported:
        config_file = get_language_config_file(lang)
        config_name = config_file.replace(".toml", "")

        if config_name not in available:
            missing.append(f"{lang} -> {config_file}")
        else:
            valid_mapping[lang] = config_file

    if missing:
        raise ConfigError(f"Missing config files for languages: {missing}")

    return valid_mapping


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


# Pipeline configuration functions
def get_pipeline_config() -> Dict[str, Any]:
    """Get complete pipeline configuration."""
    return load_config("pipeline")


def get_processing_config() -> Dict[str, Any]:
    """Get document processing configuration."""
    pipeline_config = get_pipeline_config()
    return pipeline_config["processing"]


def get_chroma_config() -> Dict[str, Any]:
    """Get ChromaDB configuration."""
    pipeline_config = get_pipeline_config()
    return pipeline_config["chroma"]


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    pipeline_config = get_pipeline_config()
    return pipeline_config["performance"]


def get_system_config() -> Dict[str, Any]:
    """Get system configuration."""
    main_config = load_config("config")
    return main_config["system"]
