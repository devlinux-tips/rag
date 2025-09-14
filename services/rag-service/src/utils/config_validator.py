"""
Configuration Validator for RAG System
Two-phase validation system to eliminate silent fallbacks and ensure fail-fast behavior.

This module implements the ConfigValidator as defined in CONFIG_ARCHITECTURE.md
- Phase 1: Startup validation ensures ALL required keys exist
- Phase 2: Enables clean DI components with direct dictionary access

Author: RAG System Architecture
Status: Production Implementation
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass
class ConfigValidationResult:
    """Result of configuration validation with detailed error information."""

    is_valid: bool
    missing_keys: list[str]
    invalid_types: list[str]
    config_file: str

    def __str__(self) -> str:
        if self.is_valid:
            return f"✅ {self.config_file}: Valid"

        errors = []
        if self.missing_keys:
            errors.append(f"Missing keys: {', '.join(self.missing_keys)}")
        if self.invalid_types:
            errors.append(f"Invalid types: {', '.join(self.invalid_types)}")

        return f"❌ {self.config_file}: {' | '.join(errors)}"


class ConfigValidator:
    """
    Two-phase configuration validator that eliminates need for .get() fallbacks.

    PHASE 1: validate_startup_config() - Validates ALL required keys exist at startup
    PHASE 2: Enables clean DI components with guaranteed valid configuration

    This validator follows the fail-fast philosophy:
    - System won't start with invalid/missing configuration
    - Components can use direct dictionary access after validation
    - No silent fallbacks or magic defaults in business logic
    """

    # Main config.toml required keys (shared across system)
    MAIN_CONFIG_SCHEMA: dict[str, type | tuple[Any, ...]] = {
        # Shared settings
        "shared.cache_dir": str,
        "shared.default_timeout": (int, float),
        "shared.default_device": str,
        "shared.default_batch_size": int,
        "shared.default_chunk_size": int,
        "shared.default_chunk_overlap": int,
        "shared.min_chunk_size": int,
        "shared.default_top_k": int,
        "shared.similarity_threshold": (int, float),
        # Language configuration
        "languages.supported": list,
        "languages.default": str,
        # Embeddings configuration
        "embeddings.model_name": str,
        "embeddings.device": str,
        "embeddings.max_seq_length": int,
        "embeddings.batch_size": int,
        "embeddings.normalize_embeddings": bool,
        "embeddings.use_safetensors": bool,
        "embeddings.trust_remote_code": bool,
        # Query processing configuration
        "query_processing.language": str,
        "query_processing.expand_synonyms": bool,
        "query_processing.normalize_case": bool,
        "query_processing.remove_stopwords": bool,
        "query_processing.min_query_length": int,
        "query_processing.max_query_length": int,
        "query_processing.max_expanded_terms": int,
        "query_processing.enable_morphological_analysis": bool,
        "query_processing.use_query_classification": bool,
        "query_processing.enable_spell_check": bool,
        # Retrieval configuration
        "retrieval.default_k": int,
        "retrieval.max_k": int,
        "retrieval.adaptive_retrieval": bool,
        "retrieval.enable_reranking": bool,
        "retrieval.diversity_lambda": (int, float),
        "retrieval.use_hybrid_search": bool,
        "retrieval.enable_query_expansion": bool,
        # Ranking configuration
        "ranking.method": str,
        "ranking.enable_diversity": bool,
        "ranking.diversity_threshold": (int, float),
        "ranking.boost_recent": bool,
        "ranking.boost_authoritative": bool,
        "ranking.content_length_factor": bool,
        "ranking.keyword_density_factor": bool,
        "ranking.language_specific_boost": bool,
        # Reranking configuration
        "reranking.enabled": bool,
        "reranking.model_name": str,
        "reranking.max_length": int,
        "reranking.batch_size": int,
        "reranking.top_k": int,
        "reranking.use_fp16": bool,
        "reranking.normalize": bool,
        # Hybrid retrieval configuration
        "hybrid_retrieval.dense_weight": (int, float),
        "hybrid_retrieval.sparse_weight": (int, float),
        "hybrid_retrieval.fusion_method": str,
        "hybrid_retrieval.bm25_k1": (int, float),
        "hybrid_retrieval.bm25_b": (int, float),
        # Ollama configuration
        "ollama.base_url": str,
        "ollama.model": str,
        "ollama.temperature": (int, float),
        "ollama.max_tokens": int,
        "ollama.top_p": (int, float),
        "ollama.top_k": int,
        "ollama.stream": bool,
        "ollama.keep_alive": str,
        "ollama.num_predict": int,
        "ollama.repeat_penalty": (int, float),
        "ollama.seed": int,
        # Processing configuration
        "processing.sentence_chunk_overlap": int,
        "processing.preserve_paragraphs": bool,
        "processing.enable_smart_chunking": bool,
        "processing.respect_document_structure": bool,
        # Chunking configuration
        "chunking.strategy": str,
        "chunking.max_chunk_size": int,
        "chunking.preserve_sentence_boundaries": bool,
        "chunking.respect_paragraph_breaks": bool,
        "chunking.enable_smart_splitting": bool,
        "chunking.sentence_search_range": int,
        "chunking.paragraph_separators": list,
        "chunking.min_sentence_length": int,
        # Storage configuration
        "storage.db_path_template": str,
        "storage.collection_name_template": str,
        "storage.distance_metric": str,
        "storage.persist": bool,
        "storage.allow_reset": bool,
        # Search configuration
        "search.default_method": str,
        "search.max_context_length": int,
        "search.rerank": bool,
        "search.include_metadata": bool,
        "search.include_distances": bool,
        "search.weights.semantic_weight": (int, float),
        "search.weights.keyword_weight": (int, float),
        # Response parsing configuration
        "response_parsing.validate_responses": bool,
        "response_parsing.extract_confidence_scores": bool,
        "response_parsing.parse_citations": bool,
        "response_parsing.handle_incomplete_responses": bool,
        "response_parsing.max_response_length": int,
        "response_parsing.min_response_length": int,
        "response_parsing.filter_hallucinations": bool,
        "response_parsing.require_source_grounding": bool,
        "response_parsing.confidence_threshold": (int, float),
        "response_parsing.response_format": str,
        "response_parsing.include_metadata": bool,
    }

    # Ranking features schema for language-specific configuration files
    RANKING_FEATURES_SCHEMA: dict[str, type | tuple[Any, ...]] = {
        # Special characters configuration (Croatian diacritics, etc.)
        "ranking.language_features.special_characters.enabled": bool,
        "ranking.language_features.special_characters.characters": list,
        "ranking.language_features.special_characters.max_score": (int, float),
        "ranking.language_features.special_characters.density_factor": (int, float),
        # Importance words configuration
        "ranking.language_features.importance_words.enabled": bool,
        "ranking.language_features.importance_words.words": list,
        "ranking.language_features.importance_words.max_score": (int, float),
        "ranking.language_features.importance_words.word_boost": (int, float),
        # Cultural patterns configuration
        "ranking.language_features.cultural_patterns.enabled": bool,
        "ranking.language_features.cultural_patterns.patterns": list,
        "ranking.language_features.cultural_patterns.max_score": (int, float),
        "ranking.language_features.cultural_patterns.pattern_boost": (int, float),
        # Grammar patterns configuration
        "ranking.language_features.grammar_patterns.enabled": bool,
        "ranking.language_features.grammar_patterns.patterns": list,
        "ranking.language_features.grammar_patterns.max_score": (int, float),
        "ranking.language_features.grammar_patterns.pattern_boost": (int, float),
        # Capitalization configuration
        "ranking.language_features.capitalization.enabled": bool,
        "ranking.language_features.capitalization.proper_nouns": list,
        "ranking.language_features.capitalization.max_score": (int, float),
        "ranking.language_features.capitalization.capitalization_boost": (int, float),
        # Vocabulary patterns configuration
        "ranking.language_features.vocabulary_patterns.enabled": bool,
        "ranking.language_features.vocabulary_patterns.patterns": list,
        "ranking.language_features.vocabulary_patterns.max_score": (int, float),
        "ranking.language_features.vocabulary_patterns.pattern_boost": (int, float),
    }

    # Language config schema - includes main config + ranking features for language-specific files
    LANGUAGE_CONFIG_SCHEMA: dict[str, type | tuple[Any, ...]] = {
        **MAIN_CONFIG_SCHEMA,  # All main config keys
        **RANKING_FEATURES_SCHEMA,  # Plus language-specific ranking features
        # Language-specific metadata
        "language.code": str,
        "language.name": str,
    }

    @classmethod
    def validate_startup_config(cls, main_config: dict, language_configs: dict[str, dict]) -> None:
        """
        PHASE 1: Validate ALL configuration at system startup.

        This method performs comprehensive validation of both main config and all
        language-specific configs. System will fail to start if ANY required key
        is missing or has wrong type.

        Args:
            main_config: Dictionary loaded from config/config.toml
            language_configs: Dictionary of {language_code: config_dict} from language files

        Raises:
            ConfigurationError: If any validation fails, with detailed error information
        """
        logger.info("🔍 Starting comprehensive configuration validation...")

        # Validate main configuration
        main_result = cls._validate_config_section(
            config=main_config, schema=cls.MAIN_CONFIG_SCHEMA, config_file="config/config.toml"
        )

        if not main_result.is_valid:
            logger.error(f"❌ Main configuration validation failed: {main_result}")
            raise ConfigurationError(
                f"Invalid main configuration in {main_result.config_file}:\n"
                f"Missing keys: {main_result.missing_keys}\n"
                f"Invalid types: {main_result.invalid_types}\n\n"
                f"Please ensure all required keys exist in config/config.toml"
            )

        logger.info("✅ Main configuration validation passed")

        # Validate each language configuration
        for lang_code, lang_config in language_configs.items():
            lang_result = cls._validate_config_section(
                config=lang_config, schema=cls.LANGUAGE_CONFIG_SCHEMA, config_file=f"config/{lang_code}.toml"
            )

            if not lang_result.is_valid:
                logger.error(f"❌ Language configuration validation failed: {lang_result}")
                raise ConfigurationError(
                    f"Invalid language configuration in {lang_result.config_file}:\n"
                    f"Missing keys: {lang_result.missing_keys}\n"
                    f"Invalid types: {lang_result.invalid_types}\n\n"
                    f"Please ensure all required keys exist in config/{lang_code}.toml"
                )

        logger.info(f"✅ All language configurations validated: {list(language_configs.keys())}")

        # Cross-config consistency validation
        cls._validate_cross_config_consistency(main_config, language_configs)

        # Validate ranking features consistency between languages
        cls._validate_ranking_features_consistency(language_configs)

        logger.info("🎯 Configuration validation completed successfully - all keys exist and are valid")

    @classmethod
    def _validate_config_section(
        cls, config: dict, schema: dict[str, type | tuple], config_file: str
    ) -> ConfigValidationResult:
        """
        Validate individual config section against schema.

        Args:
            config: Configuration dictionary to validate
            schema: Schema defining required keys and their types
            config_file: Name of config file for error reporting

        Returns:
            ConfigValidationResult with validation status and detailed errors
        """
        missing_keys = []
        invalid_types = []

        for key_path, expected_type in schema.items():
            try:
                # Navigate nested dictionary structure using key path
                current = config
                keys = key_path.split(".")

                for key in keys:
                    if not isinstance(current, dict):
                        raise KeyError(f"Expected dict at {key} but got {type(current)}")
                    current = current[key]  # Direct access - no .get() fallbacks

                # Type validation - handle union types (e.g., (int, float))
                if isinstance(expected_type, tuple):
                    if not isinstance(current, expected_type):
                        invalid_types.append(f"{key_path}: expected {expected_type}, got {type(current).__name__}")
                else:
                    if not isinstance(current, expected_type):
                        invalid_types.append(
                            f"{key_path}: expected {expected_type.__name__}, got {type(current).__name__}"
                        )

            except (KeyError, TypeError):
                missing_keys.append(key_path)

        return ConfigValidationResult(
            is_valid=(len(missing_keys) == 0 and len(invalid_types) == 0),
            missing_keys=missing_keys,
            invalid_types=invalid_types,
            config_file=config_file,
        )

    @classmethod
    def _validate_cross_config_consistency(cls, main_config: dict, language_configs: dict[str, dict]) -> None:
        """
        Validate consistency across configuration files.

        Ensures that language references in main config match available language files,
        and that language-specific settings are coherent.

        Args:
            main_config: Main configuration dictionary
            language_configs: Language configuration dictionaries

        Raises:
            ConfigurationError: If cross-config inconsistencies found
        """
        # Validate supported languages match available language configs
        supported_languages = main_config["languages"]["supported"]
        available_languages = set(language_configs.keys())
        declared_languages = set(supported_languages)

        if declared_languages != available_languages:
            missing_configs = declared_languages - available_languages
            extra_configs = available_languages - declared_languages

            error_parts = []
            if missing_configs:
                error_parts.append(f"Missing language config files: {list(missing_configs)}")
            if extra_configs:
                error_parts.append(f"Extra language config files: {list(extra_configs)}")

            raise ConfigurationError(
                f"Language configuration mismatch:\n"
                f"{'; '.join(error_parts)}\n\n"
                f"Declared in config.toml: {supported_languages}\n"
                f"Available config files: {list(available_languages)}"
            )

        # Validate default language exists
        default_language = main_config["languages"]["default"]
        if default_language not in language_configs:
            raise ConfigurationError(
                f"Default language '{default_language}' not found in available language configs: "
                f"{list(language_configs.keys())}"
            )

        # Validate language codes in language configs match their filenames
        for lang_code, lang_config in language_configs.items():
            config_lang_code = lang_config["language"]["code"]
            if config_lang_code != lang_code:
                raise ConfigurationError(
                    f"Language code mismatch in {lang_code}.toml: "
                    f"filename says '{lang_code}' but config says '{config_lang_code}'"
                )

    @classmethod
    def _validate_ranking_features_consistency(cls, language_configs: dict[str, dict]) -> None:
        """
        Validate that ranking features have consistent structure across all languages.

        Ensures that all language configurations have the same ranking feature keys
        even if some features are disabled for certain languages.

        Args:
            language_configs: Dictionary of language configuration dictionaries

        Raises:
            ConfigurationError: If ranking features structure inconsistencies found
        """
        if len(language_configs) < 2:
            return  # Nothing to compare

        # Get first language config as reference
        reference_lang = list(language_configs.keys())[0]
        reference_config = language_configs[reference_lang]

        try:
            reference_features = reference_config["ranking"]["language_features"]
            reference_feature_keys = cls._get_nested_keys(reference_features)
        except KeyError as e:
            raise ConfigurationError(f"Missing ranking.language_features section in {reference_lang}.toml") from e

        # Validate all other language configs have same structure
        for lang_code, lang_config in language_configs.items():
            if lang_code == reference_lang:
                continue

            try:
                lang_features = lang_config["ranking"]["language_features"]
                lang_feature_keys = cls._get_nested_keys(lang_features)
            except KeyError as e:
                raise ConfigurationError(f"Missing ranking.language_features section in {lang_code}.toml") from e

            # Check for missing or extra keys
            missing_keys = reference_feature_keys - lang_feature_keys
            extra_keys = lang_feature_keys - reference_feature_keys

            if missing_keys or extra_keys:
                error_parts = []
                if missing_keys:
                    error_parts.append(f"Missing feature keys: {sorted(missing_keys)}")
                if extra_keys:
                    error_parts.append(f"Extra feature keys: {sorted(extra_keys)}")

                raise ConfigurationError(
                    f"Ranking features structure mismatch between {reference_lang}.toml and {lang_code}.toml:\n"
                    f"{'; '.join(error_parts)}\n\n"
                    f"All language configs must have identical ranking.language_features structure"
                )

    @classmethod
    def _get_nested_keys(cls, config_dict: dict, prefix: str = "") -> set:
        """
        Get all nested keys from a configuration dictionary.

        Args:
            config_dict: Dictionary to extract keys from
            prefix: Prefix for nested keys

        Returns:
            Set of all nested key paths
        """
        keys = set()
        for key, value in config_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)

            if isinstance(value, dict):
                keys.update(cls._get_nested_keys(value, full_key))

        return keys

    @classmethod
    def get_main_config_schema(cls) -> dict[str, type | tuple[Any, ...]]:
        """Get the main config schema for external validation."""
        return cls.MAIN_CONFIG_SCHEMA.copy()

    @classmethod
    def get_language_config_schema(cls) -> dict[str, type | tuple[Any, ...]]:
        """Get the language config schema for external validation."""
        return cls.LANGUAGE_CONFIG_SCHEMA.copy()

    @classmethod
    def get_ranking_features_schema(cls) -> dict[str, type | tuple[Any, ...]]:
        """Get the ranking features schema for external validation."""
        return cls.RANKING_FEATURES_SCHEMA.copy()

    @classmethod
    def validate_single_config_key(
        cls, config: dict, key_path: str, expected_type: type | tuple, config_file: str = "unknown"
    ) -> bool:
        """
        Utility method to validate a single configuration key.

        Useful for component-specific validation or debugging.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., "embeddings.model_name")
            expected_type: Expected type or tuple of types
            config_file: Config file name for error reporting

        Returns:
            bool: True if key exists and has correct type

        Raises:
            ConfigurationError: If key missing or wrong type
        """
        try:
            current = config
            keys = key_path.split(".")

            for key in keys:
                current = current[key]  # Direct access - no .get()

            # Type validation
            if isinstance(expected_type, tuple):
                if not isinstance(current, expected_type):
                    raise ConfigurationError(
                        f"Invalid type for {key_path} in {config_file}: expected {expected_type}, got {type(current)}"
                    )
            else:
                if not isinstance(current, expected_type):
                    raise ConfigurationError(
                        f"Invalid type for {key_path} in {config_file}: expected {expected_type}, got {type(current)}"
                    )

            return True

        except KeyError as e:
            raise ConfigurationError(f"Missing required configuration key: {key_path} in {config_file}") from e


# Convenience functions for common validation scenarios
def validate_main_config(config: dict) -> None:
    """Validate main configuration only."""
    ConfigValidator.validate_startup_config(config, {})


def validate_language_config(config: dict, language_code: str) -> None:
    """Validate single language configuration."""
    ConfigValidator.validate_startup_config({}, {language_code: config})


def ensure_config_key_exists(
    config: dict, key_path: str, expected_type: type | tuple = str, config_file: str = "config"
) -> Any:
    """
    Ensure a configuration key exists and return its value.

    Replacement for .get() patterns - fails fast with clear error message.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        expected_type: Expected type(s) for the value
        config_file: Config file name for error reporting

    Returns:
        The configuration value

    Raises:
        ConfigurationError: If key missing or wrong type
    """
    ConfigValidator.validate_single_config_key(config, key_path, expected_type, config_file)

    # Navigate to the key and return value
    current = config
    for key in key_path.split("."):
        current = current[key]

    return current
