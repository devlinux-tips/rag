"""
Centralized Configuration Loader for Multilingual RAG Project

This module provides a unified interface for loading TOML configuration files
with multilingual language settings and environment-specific overrides.

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
import os
import re
import tomllib
from pathlib import Path
from typing import Any, cast

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
        self._cache: dict[str, dict[str, Any]] = {}

        # Validate config directory exists
        if not self.config_dir.exists():
            raise ConfigError(f"Configuration directory not found: {self.config_dir}")

    def _get_language_config_files(self) -> dict[str, str]:
        """Get available language config files dynamically."""
        # Convention-based approach: language code maps to filename (hr -> hr.toml, en -> en.toml)
        # This matches the actual architecture used in the project
        return {}

    def load(self, config_name: str, use_cache: bool = True) -> dict[str, Any]:
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

        # Determine config file path based on naming convention
        # Language codes (hr, en, etc.) map directly to language-specific config files
        # Everything else uses the main config.toml

        # Check if this is a supported language code
        supported_languages = ["hr", "en"]  # Could make this dynamic if needed

        if config_name in supported_languages:
            # This is a language-specific config (hr.toml, en.toml, etc.)
            config_path = self.config_dir / f"{config_name}.toml"
        else:
            # This is a main system config - always use config.toml
            config_path = self.config_dir / "config.toml"

        if not config_path.exists():
            available_files = [f.name for f in self.config_dir.glob("*.toml")]
            logger.error(
                f"Configuration file not found: {config_path}",
                extra={
                    "config_name": config_name,
                    "expected_path": str(config_path),
                    "available_files": available_files,
                },
            )
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)

            # Expand environment variables in config values (security: avoid hardcoded secrets)
            config_data = self._expand_env_vars(config_data)

            self._cache[config_name] = config_data
            logger.info(f"Loaded configuration: {config_name} from {config_path}, sections: {list(config_data.keys())}")
            return config_data

        except tomllib.TOMLDecodeError as e:
            logger.error(
                f"Invalid TOML syntax in {config_path}: {e}",
                extra={"config_path": str(config_path), "toml_error": str(e)},
            )
            raise ConfigError(f"Invalid TOML in {config_path}: {e}") from e
        except Exception as e:
            logger.error(
                f"Failed to load {config_path}: {e}",
                extra={"config_path": str(config_path), "error_type": type(e).__name__},
            )
            raise ConfigError(f"Failed to load {config_path}: {e}") from e

    def get_section(self, config_name: str, section: str) -> dict[str, Any]:
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

        return cast(dict[str, Any], config[section])

    def merge_configs(self, *config_names: str) -> dict[str, Any]:
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

    def _expand_env_vars(self, config: Any) -> Any:
        """
        Recursively expand environment variables in config values.

        Supports ${VAR_NAME} syntax for environment variable substitution.
        SECURITY: Prevents hardcoded secrets in git-tracked config files.

        Args:
            config: Configuration dict, list, or primitive value

        Returns:
            Configuration with expanded environment variables
        """
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Match ${VAR_NAME} pattern
            def replace_env_var(match):
                var_name = match.group(1)
                value = os.environ.get(var_name, "")
                if not value:
                    logger.warning(f"Environment variable {var_name} not set, using empty string")
                return value

            return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", replace_env_var, config)
        else:
            return config


# Global config loader instance
_config_loader = ConfigLoader()


def load_config(config_name: str, use_cache: bool = True) -> dict[str, Any]:
    """
    Load a configuration file using the global config loader.

    Args:
        config_name: Name of config file (without .toml extension)
        use_cache: Whether to use cached config if available

    Returns:
        Dictionary containing configuration data
    """
    return _config_loader.load(config_name, use_cache=use_cache)


def get_config_section(config_name: str, section: str) -> dict[str, Any]:
    """
    Get a specific section from a config file.

    Args:
        config_name: Name of config file
        section: Section name to extract

    Returns:
        Dictionary containing section data
    """
    return _config_loader.get_section(config_name, section)


def get_shared_config() -> dict[str, Any]:
    """Get shared configuration from main config (constants and common settings)."""
    main_config = load_config("config")
    return cast(dict[str, Any], main_config["shared"])


# ============================================================================
# MULTILINGUAL CONFIGURATION FUNCTIONS
# ============================================================================


def get_language_config(language: str) -> dict[str, Any]:
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
        raise ConfigError(f"Unsupported language: {language}. Supported languages: {', '.join(supported)}")

    # Get config file name for language
    config_file = get_language_config_file(language)
    config_name = config_file.replace(".toml", "")

    return load_config(config_name)


def get_language_shared(language: str) -> dict[str, Any]:
    """
    Get shared configuration for specified language.

    Args:
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Dictionary containing language-specific shared configuration
    """
    config = get_language_config(language)
    return cast(dict[str, Any], config["shared"])


def get_language_specific_config(section: str, language: str) -> dict[str, Any]:
    """
    Get specific configuration section for specified language.

    Args:
        section: Configuration section name (e.g., 'prompts', 'retrieval', 'patterns')
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Dictionary containing language-specific section configuration

    Raises:
        ConfigError: If section not found in language configuration
    """
    config = get_language_config(language)

    # Handle patterns section which is now in shared.patterns
    if section == "patterns":
        if "shared" in config and "patterns" in config["shared"]:
            return cast(dict[str, Any], config["shared"]["patterns"])
        else:
            raise ConfigError(f"Patterns not found in shared section of {language}.toml")

    if section not in config:
        raise ConfigError(f"Section '{section}' not found in {language}.toml")
    return cast(dict[str, Any], config[section])


def get_supported_languages() -> list[str]:
    """
    Get list of supported languages from configuration.

    Returns:
        List of supported language codes

    Raises:
        ConfigError: If languages configuration is invalid
    """
    try:
        main_config = load_config("config")
        # FAIL FAST: No fallback defaults - configuration must be complete
        return cast(list[str], main_config["languages"]["supported"])
    except Exception as e:
        # FAIL FAST: Language configuration must be valid
        raise ConfigError(f"Failed to load supported languages: {e}") from e


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


def validate_language_configuration() -> dict[str, str]:
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


def get_generation_config() -> dict[str, Any]:
    """
    Get generation module configuration.

    Returns:
        Dictionary containing generation config
    """
    return load_config("generation")


def get_ollama_config() -> dict[str, Any]:
    """
    Get Ollama-specific configuration.

    Returns:
        Dictionary containing Ollama config
    """
    return get_config_section("generation", "ollama")


def get_response_parsing_config() -> dict[str, Any]:
    """Get response parsing configuration."""
    generation_config = get_generation_config()
    return cast(dict[str, Any], generation_config["response_parsing"])


def get_preprocessing_config() -> dict[str, Any]:
    """Get preprocessing configuration."""
    # Try to load a dedicated preprocessing config file
    try:
        return _config_loader.load("preprocessing", use_cache=True)
    except Exception:
        # If no preprocessing.toml exists, return empty dict
        # This allows tests to work with mocked configs
        return {}


def get_extraction_config() -> dict[str, Any]:
    """Get extraction configuration."""
    preprocessing_config = get_preprocessing_config()
    return cast(dict[str, Any], preprocessing_config["extraction"])


def get_chunking_config() -> dict[str, Any]:
    """Get chunking configuration."""
    preprocessing_config = get_preprocessing_config()
    return cast(dict[str, Any], preprocessing_config["chunking"])


def get_cleaning_config() -> dict[str, Any]:
    """Get cleaning configuration."""
    preprocessing_config = get_preprocessing_config()
    return cast(dict[str, Any], preprocessing_config["cleaning"])


def get_generation_prompts_config() -> dict[str, Any]:
    """
    Get generation prompts configuration.

    Returns:
        Dictionary containing prompts config
    """
    return get_config_section("generation", "prompts")


def merge_configs(*config_names: str) -> dict[str, Any]:
    """
    Merge multiple configuration files.

    Args:
        config_names: Names of config files to merge

    Returns:
        Merged configuration dictionary
    """
    return _config_loader.merge_configs(*config_names)


def reload_config(config_name: str) -> dict[str, Any]:
    """
    Force reload a configuration file (bypass cache).

    Args:
        config_name: Name of config file to reload

    Returns:
        Reloaded configuration dictionary
    """
    return _config_loader.load(config_name, use_cache=False)


def get_project_info() -> dict[str, Any]:
    """
    Get main project information.

    Returns:
        Dictionary containing project configuration
    """
    return get_config_section("main", "project")


def get_paths_config() -> dict[str, str]:
    """
    Get project paths configuration.

    Returns:
        Dictionary containing path configurations
    """
    return get_config_section("main", "paths")


# Vector Database Configuration Functions
def get_vectordb_config() -> dict[str, Any]:
    """Get complete vectordb configuration."""
    return load_config("vectordb")


def get_embeddings_config() -> dict[str, Any]:
    """Get embeddings configuration."""
    vectordb_config = get_vectordb_config()
    return cast(dict[str, Any], vectordb_config["embeddings"])


def get_storage_config() -> dict[str, Any]:
    """Get storage configuration."""
    vectordb_config = get_vectordb_config()
    if "storage" not in vectordb_config:
        raise ValueError("Missing required config: vectordb.storage")
    return cast(dict[str, Any], vectordb_config["storage"])


def get_search_config() -> dict[str, Any]:
    """Get search configuration."""
    vectordb_config = get_vectordb_config()
    return cast(dict[str, Any], vectordb_config["search"])


# Retrieval Configuration Functions
def get_retrieval_config() -> dict[str, Any]:
    """Get main retrieval configuration."""
    return _config_loader.load("retrieval")


def get_query_processing_config() -> dict[str, Any]:
    """Get query processing configuration."""
    retrieval_config = get_retrieval_config()
    return cast(dict[str, Any], retrieval_config["query_processing"])


def get_ranking_config() -> dict[str, Any]:
    """Get ranking configuration."""
    retrieval_config = get_retrieval_config()
    return cast(dict[str, Any], retrieval_config["ranking"])


def get_language_ranking_features(language: str) -> dict[str, Any]:
    """
    Get language-specific ranking features configuration.

    This function retrieves the ranking.language_features section from the
    language-specific configuration file (hr.toml, en.toml, etc.).

    Args:
        language: Language code ('hr' for Croatian, 'en' for English)

    Returns:
        Dictionary containing language-specific ranking features configuration
        with structure:
        {
            "special_characters": {"enabled": bool, "characters": list, ...},
            "importance_words": {"enabled": bool, "words": list, ...},
            "cultural_patterns": {"enabled": bool, "patterns": list, ...},
            "grammar_patterns": {"enabled": bool, "patterns": list, ...},
            "capitalization": {"enabled": bool, "proper_nouns": list, ...},
            "vocabulary_patterns": {"enabled": bool, "patterns": list, ...}
        }

    Raises:
        ConfigError: If language not supported or ranking features not found

    Example:
        >>> features = get_language_ranking_features("hr")
        >>> croatian_chars = features["special_characters"]["characters"]
        >>> print(croatian_chars)  # ["č", "ć", "š", "ž", "đ"]
    """
    if not is_language_supported(language):
        raise ConfigError(f"Language '{language}' not supported")

    try:
        ranking_config = get_language_specific_config("ranking", language)
        if "language_features" not in ranking_config:
            raise ConfigError(f"Missing 'language_features' section in ranking configuration for language '{language}'")
        return cast(dict[str, Any], ranking_config["language_features"])
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Failed to load language ranking features for '{language}': {e}") from e


def get_reranking_config() -> dict[str, Any]:
    """Get reranking configuration."""
    retrieval_config = get_retrieval_config()
    return cast(dict[str, Any], retrieval_config["reranking"])


def get_hybrid_retrieval_config() -> dict[str, Any]:
    """Get hybrid retrieval configuration."""
    retrieval_config = get_retrieval_config()
    return cast(dict[str, Any], retrieval_config["hybrid_retrieval"])


# Pipeline configuration functions
def get_pipeline_config() -> dict[str, Any]:
    """Get complete pipeline configuration."""
    return load_config("pipeline")


def get_processing_config() -> dict[str, Any]:
    """Get document processing configuration."""
    pipeline_config = get_pipeline_config()
    return cast(dict[str, Any], pipeline_config["processing"])


def get_chroma_config() -> dict[str, Any]:
    """Get ChromaDB configuration."""
    pipeline_config = get_pipeline_config()
    return cast(dict[str, Any], pipeline_config["chroma"])


def get_performance_config() -> dict[str, Any]:
    """Get performance configuration."""
    pipeline_config = get_pipeline_config()
    return cast(dict[str, Any], pipeline_config["performance"])


def get_system_config() -> dict[str, Any]:
    """Get system configuration."""
    main_config = load_config("config")
    return cast(dict[str, Any], main_config["system"])


def get_logging_config() -> dict[str, Any]:
    """Get logging configuration with fallback defaults."""
    try:
        config = get_shared_config()
        logging_config = config["logging"]

        # Set sensible defaults
        if "backends" not in logging_config:
            logging_config["backends"] = ["console"]

        if "level" not in logging_config:
            logging_config["level"] = "INFO"

        # Elasticsearch defaults
        if "elasticsearch" not in logging_config:
            logging_config["elasticsearch"] = {
                "hosts": ["localhost:9200"],
                "index_prefix": "rag-logs",
                "timeout": 30,
                "verify_certs": False,
            }

        return logging_config
    except Exception:
        # Fallback to console-only logging
        return {
            "backends": ["console"],
            "level": "INFO",
            "elasticsearch": {
                "hosts": ["localhost:9200"],
                "index_prefix": "rag-logs",
                "timeout": 30,
                "verify_certs": False,
            },
        }
