"""
Pytest configuration and shared test fixtures.
All mock classes and test helpers belong here, NOT in production code.
"""

import pytest
import numpy as np
from typing import Any, cast
from pathlib import Path

from src.utils.language_manager import LanguageSettings, LanguagePatterns, ConfigProvider
from src.utils.language_manager_providers import DefaultConfigProvider, DefaultPatternProvider, StandardLoggerProvider
from src.utils.folder_manager import FolderConfig, FolderStats
from src.utils.config_models import RetrievalConfig
from src.generation.enhanced_prompt_templates import PromptConfig, PromptType
from src.generation.prompt_templates import PromptTemplate
from src.generation.response_parser import ParsingConfig, ParsingConfigProvider
from src.generation.ollama_client import HttpResponse
from src.cli.rag_cli import OutputWriterProtocol, MultiTenantRAGCLI
from src.retrieval.hierarchical_retriever import ProcessedQuery, HierarchicalRetriever, SearchResult
from src.retrieval.categorization import CategoryMatch, CategoryType
from src.retrieval.ranker import DocumentRanker, LanguageFeatures, LanguageProvider
from src.retrieval.reranker import ModelLoader, ScoreCalculator
from src.retrieval.hybrid_retriever import StopWordsProvider
from src.vectordb.embeddings import DeviceInfo, EmbeddingModel
from src.vectordb.search import EmbeddingProvider, VectorSearchProvider
from src.vectordb.storage import VectorCollection, VectorSearchResults, VectorStorage, VectorDatabase
from src.utils.logging_factory import get_system_logger

# This file contains all mock classes and test helpers
# Consolidated from duplicates - each class appears only once

# ============================================================================
# CONFIGURATION MOCKS
# ============================================================================

class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self, config_dict: dict[str, Any] | None = None):
        """
        Initialize with test configuration.

        Args:
            config_dict: Configuration dictionary for testing (optional)
        """
        # Support multiple initialization patterns from different test versions
        self.config_dict = config_dict or {}
        self.config = config_dict or {}  # Alias for compatibility
        self.mock_configs = {}  # For set_config/load_config pattern
        self.mock_language_configs: dict[str, dict[str, Any]] = {}
        self.mock_shared_config: dict[str, Any] = {}
        self.configs = {}
        self.settings = None
        self.extraction_config = {}
        self.chunking_configs = {}
        self.document_cleaning_configs = {}
        self.cleaning_config = {}
        self.shared_language_configs = {}
        self.call_history = []

        self.language_configs = {
            "hr": {
                "morphology": {
                    "important_words": ["zagreb", "hrvatska", "dubrovnik", "važan", "značajan"],
                    "quality_positive": ["detaljno", "sveobuhvatno", "temeljito", "precizno"],
                    "quality_negative": ["možda", "vjerojatno", "nejasno", "približno"],
                    "cultural_patterns": ["biser jadrana", "perla jadrana", "adriatic", "unesco"],
                    "grammar_patterns": ["\\w+ić\\b", "\\w+ović\\b", "\\w+ski\\b", "\\w+nja\\b"],
                }
            },
            "en": {
                "morphology": {
                    "important_words": ["important", "significant", "major", "primary", "essential"],
                    "quality_positive": ["detailed", "comprehensive", "thorough", "precise"],
                    "quality_negative": ["maybe", "probably", "unclear", "approximately"],
                    "cultural_patterns": ["United States", "UK", "Britain", "American"],
                    "grammar_patterns": ["\\w+ing\\b", "\\w+ly\\b", "\\w+tion\\b", "\\w+ness\\b"],
                }
            },
        }

    def _create_default_config(self) -> PromptConfig:
        """Create default test configuration."""
        category_templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: "You are a helpful assistant. Answer questions based on the provided context.",
                PromptType.USER: "Question: {query}\n\nContext: {context}\n\nAnswer:",
                PromptType.FOLLOWUP: "Previous question: {original_query}\nPrevious answer: {original_answer}\n\nFollow-up question: {followup_query}\n\nAnswer:",
            },
            CategoryType.TECHNICAL: {
                PromptType.SYSTEM: "You are a technical expert. Provide detailed technical answers based on the context.",
                PromptType.USER: "Technical question: {query}\n\nTechnical documentation: {context}\n\nDetailed answer:",
                PromptType.FOLLOWUP: "Previous technical question: {original_query}\nPrevious answer: {original_answer}\n\nFollow-up: {followup_query}\n\nTechnical answer:",
            },
            CategoryType.CULTURAL: {
                PromptType.SYSTEM: "You are a cultural expert. Provide culturally sensitive answers based on context.",
                PromptType.USER: "Cultural question: {query}\n\nCultural context: {context}\n\nCulturally aware answer:",
                PromptType.FOLLOWUP: "Previous cultural question: {original_query}\nPrevious answer: {original_answer}\n\nFollow-up: {followup_query}\n\nCultural answer:",
            },
        }

        messages = {
            "no_context": "No relevant context available.",
            "error_template_missing": "Template not found for this category.",
            "truncation_notice": "Some content was truncated due to length limits.",
        }

        formatting = {
            "source_label": "Source",
            "truncation_indicator": "...",
            "min_chunk_size": "100",
            "max_context_length": "2000",
        }

        return PromptConfig(
            category_templates=category_templates, messages=messages, formatting=formatting, language="hr"
        )

    def _create_default_settings(self) -> LanguageSettings:
        """Create default test settings."""
        return LanguageSettings(
            supported_languages=["hr", "en", "multilingual"],
            default_language="hr",
            auto_detect=True,
            fallback_language="hr",
            language_names={"hr": "Croatian", "en": "English", "multilingual": "Multilingual"},
            embedding_model="BAAI/bge-m3",
            chunk_size=512,
            chunk_overlap=50,
        )

    def _create_default_test_config(self) -> dict[str, Any]:
        """Create default test configuration."""
        return {
            "categories": {
                "general": {"priority": 1},
                "technical": {"priority": 2},
                "cultural": {"priority": 3},
                "academic": {"priority": 4},
            },
            "patterns": {
                "general": ["test", "example"],
                "technical": ["API", "database", "server", "kod", "programming"],
                "cultural": ["kultura", "tradicija", "culture", "tradition"],
                "academic": ["research", "study", "istraživanje", "studij"],
            },
            "cultural_keywords": {
                "test_croatian": ["test_hrvatski", "test_zagreb"],
                "test_english": ["test_english", "test_london"],
            },
            "complexity_thresholds": {"simple": 1.0, "moderate": 3.0, "complex": 6.0, "analytical": 10.0},
            "retrieval_strategies": {
                "default": "test_hybrid",
                "category_technical": "test_dense",
                "category_cultural": "test_cultural",
                "complexity_simple": "test_sparse",
                "cultural_context": "test_cultural_aware",
            },
        }

    def _default_config(self) -> dict[str, Any]:
        """Default test configuration."""
        return {
            "search": {
                "default_method": "semantic",
                "top_k": 5,
                "similarity_threshold": 0.0,
                "max_context_length": 2000,
                "rerank": True,
                "include_metadata": True,
                "include_distances": True,
            },
            "scoring": {
                "weights": {"semantic": 0.7, "keyword": 0.3},
                "boost_factors": {
                    "term_overlap": 0.2,
                    "length_optimal": 1.0,
                    "length_short": 0.8,
                    "length_long": 0.9,
                    "title_boost": 1.1,
                    "phrase_match_boost": 1.5,
                },
            },
        }


# Production Providers (Adapters for existing components)

    def _get_language_default_config(self, language: str) -> dict[str, Any]:
        """Get language-specific default configuration for testing."""
        base_config = self._default_config.copy()

        if language == "hr":
            # Croatian-specific test patterns
            base_config["patterns"]["cultural"].extend(["hrvatska", "dubrovnik", "split", "zagreb"])
            base_config["cultural_keywords"]["croatian_test"] = ["test_hr", "test_croatia", "test_jadran"]
        elif language == "en":
            # English-specific test patterns
            base_config["patterns"]["cultural"].extend(["england", "london", "british", "american"])
            base_config["cultural_keywords"]["english_test"] = ["test_en", "test_uk", "test_usa"]

        return base_config



    def add_category_template(self, category: CategoryType, prompt_type: PromptType, template: str) -> None:
        """Add a template for specific category and type."""
        if category not in self.config.category_templates:
            self.config.category_templates[category] = {}
        self.config.category_templates[category][prompt_type] = template

    def get_categorization_config(self, language: str) -> dict[str, Any]:
        """Get mock categorization configuration."""
        return {
            "categories": ["general", "technical", "cultural"],
            "patterns": {"general": [".*"], "cultural": ["culture.*"], "technical": ["tech.*"]},
            "cultural_keywords": ["culture", "traditional"],
            "complexity_thresholds": {"simple": 0.3, "medium": 0.6, "complex": 0.8},
            "retrieval_strategies": {"general": "semantic", "cultural": "semantic", "technical": "dense"},
        }

    def get_chunking_config(self, language: str) -> dict[str, Any]:
        """Get mock chunking configuration."""
        return {
            "sentence_endings": [".", "!", "?"],
            "abbreviations": [],
            "min_sentence_length": 10,
            "sentence_ending_pattern": "[.!?]+",
        }

    def get_cleaning_config(self) -> dict[str, Any]:
        """Get mock cleaning configuration."""
        return {
            "multiple_whitespace": True,
            "multiple_linebreaks": True,
            "min_meaningful_words": 3,
            "min_word_char_ratio": 0.7,
        }

    def get_config_section(self, config_name: str, section: str) -> dict[str, Any]:
        """Get mock configuration section."""
        config = self.load_config(config_name)
        if section not in config:
            raise KeyError(f"Mock section '{section}' not found in '{config_name}'")
        return cast(dict[str, Any], config[section])

    def get_document_cleaning_config(self, language: str) -> dict[str, Any]:
        """Get mock document cleaning configuration."""
        if language not in self.document_cleaning_configs:
            raise KeyError(f"Mock document cleaning config '{language}' not found")
        return self.document_cleaning_configs[language]

    def get_extraction_config(self) -> dict[str, Any]:
        """Get mock extraction configuration."""
        return self.extraction_config


    def get_folder_config(self) -> FolderConfig:
        """Get folder configuration."""
        self.call_history.append("get_folder_config")
        return self.config


    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get mock language configuration."""
        if language not in self.mock_language_configs:
            raise KeyError(f"Mock language config '{language}' not found")
        return cast(dict[str, Any], self.mock_language_configs[language])

    def get_language_settings(self) -> LanguageSettings:
        """Get language settings configuration."""
        self.call_history.append("get_language_settings")
        return self.settings


    def get_language_specific_config(self, section: str, language: str) -> dict[str, Any]:
        """Get mock language-specific configuration section."""
        # Handle patterns section which would be in shared
        if section == "patterns":
            return {"general": [".*"], "cultural": ["culture.*"], "technical": ["tech.*"]}

        language_config = self.get_language_config(language)
        if section not in language_config:
            raise KeyError(f"Mock section '{section}' not found in language '{language}'")
        return cast(dict[str, Any], language_config[section])

    def get_parsing_config(self, language: str) -> dict[str, Any]:
        """Get mock parsing configuration."""
        return {
            "validate_responses": True,
            "extract_confidence_scores": True,
            "parse_citations": True,
            "handle_incomplete_responses": True,
            "max_response_length": 3000,
            "min_response_length": 50,
            "filter_hallucinations": True,
            "require_source_grounding": True,
            "confidence_threshold": 0.7,
            "response_format": "markdown",
            "include_metadata": True,
        }

    def get_prompt_config(self, language: str) -> Any:
        """Get mock prompt configuration."""
        # Import PromptConfig at runtime to avoid circular dependencies
        from src.generation.enhanced_prompt_templates import PromptConfig

        return PromptConfig(
            category_templates={},
            messages={
                "system_base": "You are an AI assistant.",
                "context_intro": "Based on the following context:",
                "answer_intro": "Answer:",
            },
            formatting={},
            language=language,
        )


# ============================================================================
# CLI MOCKS  
# ============================================================================

# Will be extracted from cli/rag_cli.py


# ============================================================================
# GENERATION MOCKS
# ============================================================================

# Will be extracted from generation/*.py files


# ============================================================================
# PREPROCESSING MOCKS
# ============================================================================

# Will be extracted from preprocessing/*.py files


# ============================================================================
# RETRIEVAL MOCKS
# ============================================================================

# Will be extracted from retrieval/*.py files


# ============================================================================
# VECTORDB MOCKS
# ============================================================================

# Will be extracted from vectordb/*.py files


# ============================================================================
# UTILITY MOCKS
# ============================================================================

# Will be extracted from utils/*.py files


# ============================================================================
# UTILS MOCKS
# ============================================================================

    def get_ranking_config(self) -> dict[str, Any]:
        """Get ranking configuration for testing."""
        return cast(
            dict[str, Any],
            self.config_dict.get(
                "ranking",
                {
                    "method": "language_enhanced",
                    "enable_diversity": True,
                    "diversity_threshold": 0.8,
                    "boost_recent": False,
                    "boost_authoritative": True,
                    "content_length_factor": True,
                    "keyword_density_factor": True,
                    "language_specific_boost": True,
                },
            ),
        )

    def get_scoring_weights(self) -> dict[str, float]:
        """Get scoring weights for hybrid search."""
        return self.config["scoring"]["weights"]

    def get_search_config(self) -> dict[str, Any]:
        """Get search configuration."""
        return self.config["search"]

    def get_shared_config(self) -> dict[str, Any]:
        """Get mock shared configuration."""
        return self.mock_shared_config

    def get_shared_language_config(self, language: str) -> dict[str, Any]:
        """Get mock shared language configuration."""
        if language not in self.shared_language_configs:
            raise KeyError(f"Mock shared language config '{language}' not found")
        return self.shared_language_configs[language]


    def load_config(self, config_name: str, use_cache: bool = True) -> dict[str, Any]:
        """Load mock configuration."""
        if config_name not in self.mock_configs:
            raise KeyError(f"Mock config '{config_name}' not found")
        return self.mock_configs[config_name]

    def remove_template(self, category: CategoryType, prompt_type: PromptType) -> None:
        """Remove a template (for testing missing template scenarios)."""
        if category in self.config.category_templates:
            if prompt_type in self.config.category_templates[category]:
                del self.config.category_templates[category][prompt_type]

    def set_categorization_config(self, language: str, config_data: dict[str, Any]) -> None:
        """Set mock categorization configuration for specified language."""
        self.mock_configs[f"categorization_{language}"] = config_data

    def set_chunking_config(self, language: str, config: dict[str, Any]) -> None:
        """Set mock chunking configuration."""
        self.chunking_configs[language] = config

    def set_cleaning_config(self, config: dict[str, Any]) -> None:
        """Set mock cleaning configuration."""
        self.cleaning_config = config

    def set_config(self, config_name: str, config_data: dict[str, Any]) -> None:
        """Set mock configuration data."""
        self.mock_configs[config_name] = config_data

    def set_document_cleaning_config(self, language: str, config: dict[str, Any]) -> None:
        """Set mock document cleaning configuration."""
        self.document_cleaning_configs[language] = config

    def set_language_config(self, section: str, language: str, config: dict[str, Any]) -> None:
        """Set mock language-specific configuration."""
        key = f"{section}_{language}"
        self.language_configs[key] = config

    def set_settings(self, settings: LanguageSettings) -> None:
        """Set mock settings."""
        self.settings = settings

    def set_shared_config(self, config_data: dict[str, Any]) -> None:
        """Set mock shared configuration data."""
        self.mock_shared_config = config_data

    def set_shared_language_config(self, language: str, config: dict[str, Any]) -> None:
        """Set mock shared language configuration."""
        self.shared_language_configs[language] = config


class MockPatternProvider:
    """Mock pattern provider for testing."""

    def __init__(self, patterns: LanguagePatterns | None = None):
        """Initialize with optional mock patterns."""
        self.patterns = patterns or self._create_default_patterns()
        self.call_history: list[str] = []

    def _create_default_patterns(self) -> LanguagePatterns:
        """Create default test patterns."""
        return LanguagePatterns(
            detection_patterns={
                "hr": ["što", "kako", "gdje", "kada", "zašto", "koji", "koja", "koje"],
                "en": ["what", "how", "where", "when", "why", "which", "who", "that"],
                "multilingual": ["i", "in", "of", "to", "and", "the", "is", "for"],
            },
            stopwords={
                "hr": {"i", "u", "na", "za", "je", "se", "da", "od", "do", "sa"},
                "en": {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from"},
                "multilingual": {"the", "of", "and", "to", "a", "in", "is", "it", "you", "that"},
            },
        )

    def add_detection_pattern(self, language_code: str, patterns: list[str]) -> None:
        """Add detection patterns for language."""
        self.patterns.detection_patterns[language_code] = patterns

    def add_stopwords(self, language_code: str, stopwords: set[str]) -> None:
        """Add stopwords for language."""
        self.patterns.stopwords[language_code] = stopwords

    def get_language_patterns(self) -> LanguagePatterns:
        """Get language detection patterns and stopwords."""
        self.call_history.append("get_language_patterns")
        return self.patterns


    def set_patterns(self, patterns: LanguagePatterns) -> None:
        """Set mock patterns."""
        self.patterns = patterns



# ============================================================================
# LOGGING MOCKS
# ============================================================================

class MockLoggerProvider:
    """Mock logger provider for testing."""

    def __init__(self):
        """Initialize message capture."""
        self.messages: dict[str, list[str]] = {"info": [], "debug": [], "warning": [], "error": []}

    def clear_messages(self) -> None:
        """Clear all logged messages."""
        self.debug_messages.clear()
        self.info_messages.clear()
        self.error_messages.clear()


    def debug(self, message: str) -> None:
        """Capture debug message."""
        self.messages["debug"].append(message)

    def error(self, message: str) -> None:
        """Capture error message."""
        self.messages["error"].append(message)

    def get_all_messages(self) -> dict[str, list]:
        """Get all logged messages for testing."""
        return {"debug": self.debug_messages, "info": self.info_messages, "error": self.error_messages}

    def get_messages(self, level: str | None = None) -> dict[str, list[str]] | list[str]:
        """Get captured messages by level or all messages."""
        if level:
            if level not in self.messages:
                raise ValueError(f"Unknown log level: {level}")
            return self.messages[level]
        return self.messages



    def info(self, message: str) -> None:
        """Capture info message."""
        self.messages["info"].append(message)

    def warning(self, message: str) -> None:
        """Capture warning message."""
        self.messages["warning"].append(message)


class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.logs = []

    def error(self, message: str) -> None:
        self.logs.append(("ERROR", message))

    def exception(self, message: str) -> None:
        self.logs.append(("EXCEPTION", message))


    def info(self, message: str) -> None:
        self.logs.append(("INFO", message))



# ============================================================================
# FILESYSTEM MOCKS
# ============================================================================

class MockFileSystemProvider:
    """Mock filesystemprovider for testing."""

    def __init__(self):
        """Initialize with in-memory filesystem simulation."""
        self.created_folders: list[str] = []
        self.existing_folders: dict[str, bool] = {}
        self.folder_stats: dict[str, FolderStats] = {}
        self.call_history: list[dict] = []
        self.should_fail: dict[str, bool] = {}

    def add_file(self, file_path: str, content: bytes, size_mb: float | None = None) -> None:
        """Add a mock file to the file system."""
        self.files[file_path] = content
        self.file_sizes_mb[file_path] = size_mb or len(content) / (1024 * 1024)
        self.existing_files.add(file_path)

    def add_text_file(
        self, file_path: str, content: str, encoding: str = "utf-8", size_mb: float | None = None
    ) -> None:
        """Add a mock text file to the file system."""
        binary_content = content.encode(encoding)
        self.add_file(file_path, binary_content, size_mb)

    def clear_history(self) -> None:
        """Clear operation history."""
        self.call_history.clear()

    def create_folder(self, folder_path: Path) -> bool:
        """Mock folder creation."""
        self.call_history.append({"operation": "create_folder", "path": str(folder_path)})

        if "create_folder" in self.should_fail and self.should_fail["create_folder"]:
            return False

        path_str = str(folder_path)
        if path_str not in [str(p) for p in self.created_folders]:
            self.created_folders.append(path_str)
            self.existing_folders[path_str] = True
            return True
        return False

    def file_exists(self, file_path: Path) -> bool:
        """Check if mock file exists."""
        return str(file_path) in self.existing_files

    def folder_exists(self, folder_path: Path) -> bool:
        """Mock folder existence check."""
        self.call_history.append({"operation": "folder_exists", "path": str(folder_path)})
        path_str = str(folder_path)
        if path_str not in self.existing_folders:
            raise ValueError(f"Mock folder existence not configured for {folder_path}")
        return self.existing_folders[path_str]

    def get_created_folders(self) -> list[str]:
        """Get list of folders that were created."""
        return self.created_folders.copy()


    def get_file_size_mb(self, file_path: Path) -> float:
        """Get mock file size in MB."""
        path_str = str(file_path)
        if path_str not in self.file_sizes_mb:
            raise FileNotFoundError(f"Mock file not found: {file_path}")
        return self.file_sizes_mb[path_str]

    def get_folder_stats(self, folder_path: Path) -> FolderStats:
        """Mock folder statistics."""
        self.call_history.append({"operation": "get_folder_stats", "path": str(folder_path)})

        path_str = str(folder_path)
        if path_str not in self.folder_stats:
            raise ValueError(f"Mock folder stats not configured for {folder_path}")
        return self.folder_stats[path_str]

    def open_binary(self, file_path: Path) -> bytes:
        """Open mock file in binary mode."""
        path_str = str(file_path)
        if path_str not in self.files:
            raise FileNotFoundError(f"Mock file not found: {file_path}")
        return self.files[path_str]

    def open_text(self, file_path: Path, encoding: str) -> str:
        """Open mock file in text mode with specified encoding."""
        binary_content = self.open_binary(file_path)
        return binary_content.decode(encoding)


    def remove_folder(self, folder_path: Path) -> bool:
        """Mock folder removal."""
        self.call_history.append({"operation": "remove_folder", "path": str(folder_path)})

        if "remove_folder" in self.should_fail and self.should_fail["remove_folder"]:
            return False

        path_str = str(folder_path)
        if path_str in self.existing_folders:
            del self.existing_folders[path_str]
            if path_str in self.created_folders:
                self.created_folders.remove(path_str)
        return True

    def set_folder_exists(self, folder_path: Path, exists: bool = True) -> None:
        """Set whether a folder should be considered to exist."""
        self.existing_folders[str(folder_path)] = exists

    def set_folder_stats(self, folder_path: Path, stats: FolderStats) -> None:
        """Set mock statistics for a folder."""
        self.folder_stats[str(folder_path)] = stats

    def set_should_fail(self, operation: str, should_fail: bool = True) -> None:
        """Set whether an operation should fail."""
        self.should_fail[operation] = should_fail


class MockFolderManager:
    """Mock foldermanager for testing."""

    def create_tenant_folder_structure(self, tenant: Any, user: Any, languages: list[str]) -> tuple[bool, list[str]]:
        # Access tenant_slug properly based on Tenant structure
        tenant_slug = tenant.slug if hasattr(tenant, "slug") else getattr(tenant, "tenant_slug", "unknown")
        created_folders = [f"/mock/{tenant_slug}/{lang}" for lang in languages]
        return True, created_folders


    def ensure_context_folders(self, context: Any, language: str) -> bool:
        return True

    def get_collection_storage_paths(self, context: Any, language: str) -> Any:
        from pathlib import Path

        from ..utils.folder_manager import CollectionPaths

        # Access context attributes properly based on TenantUserContext structure
        tenant_slug = context.tenant.slug if hasattr(context, "tenant") else getattr(context, "tenant_slug", "unknown")
        user_username = (
            context.user.username if hasattr(context, "user") else getattr(context, "user_username", "unknown")
        )

        return CollectionPaths(
            user_collection_name=f"user_{user_username}_{language}",
            tenant_collection_name=f"tenant_{tenant_slug}_{language}",
            user_collection_path=Path(f"/mock/path/{tenant_slug}/user"),
            tenant_collection_path=Path(f"/mock/path/{tenant_slug}/tenant"),
            base_path=Path(f"/mock/path/{tenant_slug}"),
        )

    def get_tenant_folder_structure(self, tenant: Any, user: Any, language: str) -> dict[str, Any]:
        # Access tenant_slug properly based on Tenant structure
        tenant_slug = tenant.slug if hasattr(tenant, "slug") else getattr(tenant, "tenant_slug", "unknown")

        return {
            "data_folder": Path(f"/mock/{tenant_slug}/data"),
            "models_folder": Path(f"/mock/{tenant_slug}/models"),
            "config_folder": Path(f"/mock/{tenant_slug}/config"),
        }



# ============================================================================
# ENVIRONMENT MOCKS
# ============================================================================

class MockEnvironmentProvider:
    """Mock environmentprovider for testing."""

    def __init__(self):
        """Initialize mock environment."""
        self.environment_variables: dict[str, str] = {}
        self.locale_calls: list[tuple] = []

    def clear_records(self) -> None:
        """Clear all recorded operations."""
        self.environment_variables.clear()
        self.locale_calls.clear()




# ============================================================================
# RETRIEVAL MOCKS
# ============================================================================

    def get_environment_variables(self) -> dict[str, str]:
        """Get recorded environment variables."""
        return self.environment_variables.copy()

    def get_locale_calls(self) -> list[tuple]:
        """Get recorded locale calls."""
        return self.locale_calls.copy()

    def set_environment_variable(self, key: str, value: str) -> None:
        """Record environment variable setting."""
        self.environment_variables[key] = value

    def set_locale(self, category: int, locale_name: str) -> None:
        """Record locale setting."""
        self.locale_calls.append((category, locale_name))



# ============================================================================
# LANGUAGE MOCKS
# ============================================================================

class MockLanguageProvider:
    """Mock languageprovider for testing."""

    def __init__(self):
        self.formal_prompts = {}
        self.error_templates = {}
        self.confidence_settings = {}
        self.call_log = []

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    def detect_language_content(self, text: str) -> dict[str, Any]:
        """Mock language detection for testing."""
        # Simple mock detection based on character patterns
        if any(char in text.lower() for char in "čćšžđ"):
            return {"language": "hr", "confidence": 0.9}
        else:
            return {"language": "en", "confidence": 0.8}



    def get_calls(self) -> list:
        """Get log of all method calls made."""
        return self.call_log.copy()

    def get_confidence_settings(self, language: str) -> dict[str, Any]:
        """Get confidence calculation settings for language."""
        self.call_log.append({"method": "get_confidence_settings", "language": language})

        if language not in self.confidence_settings:
            return {"error_phrases": [f"mock_error_{language}", "test_failure"]}
        return cast(dict[str, Any], self.confidence_settings[language])



# ============================================================================
# PREPROCESSING MOCKS
# ============================================================================

    def get_error_template(self, language: str) -> str:
        """Get error message template for language."""
        self.call_log.append({"method": "get_error_template", "language": language})

        if language not in self.error_templates:
            return f"Mock error template for {language}: {{error}}"
        return cast(str, self.error_templates[language])

    def get_formal_prompts(self, language: str) -> dict[str, str]:
        """Get formal prompt templates for language."""
        self.call_log.append({"method": "get_formal_prompts", "language": language})

        if language not in self.formal_prompts:
            return {"formal_instruction": f"Mock formal instruction for {language}"}
        return cast(dict[str, str], self.formal_prompts[language])

    def get_language_features(self, language: str) -> LanguageFeatures:
        """Get language features for testing."""
        if language in self.language_features_cache:
            return cast(LanguageFeatures, self.language_features_cache[language])

        if language == "hr":
            features = LanguageFeatures(
                importance_words={
                    "zagreb",
                    "hrvatska",
                    "dubrovnik",
                    "split",
                    "rijeka",
                    "osijek",
                    "glavni",
                    "važan",
                    "značajan",
                    "poznati",
                    "tradicionalni",
                    "historijski",
                    "kulturni",
                    "turistički",
                    "nacionalni",
                },
                quality_indicators={
                    "positive": [
                        r"\b(detaljno|sveobuhvatno|temeljito|precizno)\b",
                        r"\b(službeno|autoritetno|provjereno|pouzdano)\b",
                        r"\b(suvremeno|aktualno|novo|nedavno)\b",
                    ],
                    "negative": [
                        r"\b(možda|vjerojatno|nejasno|približno)\b",
                        r"\b(staro|zastarjelo|neprovjereno|sumnjivo)\b",
                        r"\b(kratko|površno|nepotpuno|fragmentarno)\b",
                    ],
                },
                cultural_patterns=[
                    r"\b(biser jadrana|perla jadrana)\b",
                    r"\b(hrvatski?\w* kralj|hrvatska povijest)\b",
                    r"\b(adriatic|jadransko more)\b",
                    r"\b(unesco|svjetska baština)\b",
                ],
                grammar_patterns=[r"\b\w+ić\b", r"\b\w+ović\b", r"\b\w+ski\b", r"\b\w+nja\b"],
                type_weights={
                    "encyclopedia": 1.2,
                    "academic": 1.1,
                    "news": 1.0,
                    "blog": 0.9,
                    "forum": 0.8,
                    "social": 0.7,
                },
            )

        elif language == "en":
            features = LanguageFeatures(
                importance_words={
                    "important",
                    "significant",
                    "major",
                    "primary",
                    "essential",
                    "key",
                    "main",
                    "crucial",
                    "critical",
                    "fundamental",
                    "notable",
                    "prominent",
                    "leading",
                    "advanced",
                    "innovative",
                },
                quality_indicators={
                    "positive": [
                        r"\b(detailed|comprehensive|thorough|precise)\b",
                        r"\b(official|authoritative|verified|reliable)\b",
                        r"\b(current|recent|new|updated)\b",
                    ],
                    "negative": [
                        r"\b(maybe|probably|unclear|approximately)\b",
                        r"\b(old|outdated|unverified|questionable)\b",
                        r"\b(brief|superficial|incomplete|fragmentary)\b",
                    ],
                },
                cultural_patterns=[
                    r"\b(United States|UK|Britain|England|American|British)\b",
                    r"\b(technology|science|research|development|innovation)\b",
                ],
                grammar_patterns=[r"\b\w+ing\b", r"\b\w+ly\b", r"\b\w+tion\b", r"\b\w+ness\b"],
                type_weights={
                    "encyclopedia": 1.2,
                    "academic": 1.1,
                    "news": 1.0,
                    "blog": 0.9,
                    "forum": 0.8,
                    "social": 0.7,
                },
            )

        else:
            # Default/fallback features
            features = LanguageFeatures(
                importance_words=set(),
                quality_indicators={"positive": [], "negative": []},
                cultural_patterns=[],
                grammar_patterns=[],
                type_weights={"default": 1.0},
            )

        self.language_features_cache[language] = features
        return features

    def set_confidence_settings(self, language: str, settings: dict[str, Any]):
        """Set confidence settings for language."""
        self.confidence_settings[language] = settings

    def set_error_template(self, language: str, template: str):
        """Set error template for language."""
        self.error_templates[language] = template

    def set_formal_prompts(self, language: str, prompts: dict[str, str]):
        """Set formal prompts for language."""
        self.formal_prompts[language] = prompts


class MockLanguageDataProvider:
    """Mock languagedataprovider for testing."""

    def __init__(self):
        """Initialize with empty mock data."""
        self.stop_words = {}
        self.question_patterns = {}
        self.synonym_groups = {}
        self.morphological_patterns = {}

    def get_morphological_patterns(self, language: str) -> dict[str, list[str]]:
        """Get mock morphological patterns for language."""
        if language not in self.morphological_patterns:
            raise ValueError(f"Mock morphological patterns not configured for language: {language}")
        return self.morphological_patterns[language]


    def get_question_patterns(self, language: str) -> list[str]:
        """Get mock question patterns for language."""
        if language not in self.question_patterns:
            raise ValueError(f"Mock question patterns not configured for language: {language}")
        return self.question_patterns[language]

    def get_stop_words(self, language: str) -> set[str]:
        """Get mock stop words for language."""
        if language not in self.stop_words:
            raise ValueError(f"Mock language data not configured for language: {language}")
        return self.stop_words[language]

    def get_synonym_groups(self, language: str) -> dict[str, list[str]]:
        """Get mock synonym groups for language."""
        if language not in self.synonym_groups:
            raise ValueError(f"Mock synonym groups not configured for language: {language}")
        return self.synonym_groups[language]

    def set_morphological_patterns(self, language: str, patterns: dict[str, list[str]]) -> None:
        """Set mock morphological patterns for language."""
        self.morphological_patterns[language] = patterns

    def set_question_patterns(self, language: str, patterns: list[str]) -> None:
        """Set mock question patterns for language."""
        self.question_patterns[language] = patterns

    def set_stop_words(self, language: str, stop_words: set[str]) -> None:
        """Set mock stop words for language."""
        self.stop_words[language] = stop_words

    def set_synonym_groups(self, language: str, synonyms: dict[str, list[str]]) -> None:
        """Set mock synonym groups for language."""
        self.synonym_groups[language] = synonyms



# ============================================================================
# HTTP MOCKS
# ============================================================================

class MockHttpClient:
    """Mock httpclient for testing."""

    def __init__(self):
        self.responses = {}
        self.streaming_responses = {}
        self.call_log = []
        self.should_raise = None

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    async def get(self, url: str, timeout: float = 30.0) -> HttpResponse:
        """Mock GET request."""
        self.call_log.append({"method": "GET", "url": url, "timeout": timeout})

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        key = f"GET:{url}"
        if key in self.responses:
            return self.responses[key]

        # Default response
        return HttpResponse(status_code=200, content=b'{"models": []}', json_data={"models": []})

    async def post(
        self, url: str, json_data: dict[str, Any], timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        """Mock POST request."""
        self.call_log.append({"method": "POST", "url": url, "json_data": json_data, "timeout": timeout})

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        key = f"POST:{url}"
        if key in self.responses:
            return self.responses[key]

        # Default response
        return HttpResponse(
            status_code=200, content=b'{"response": "Mock response"}', json_data={"response": "Mock response"}
        )

    async def stream_post(
        self, url: str, json_data: dict[str, Any], timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> list[str]:
        """Mock streaming POST request."""
        self.call_log.append({"method": "STREAM_POST", "url": url, "json_data": json_data, "timeout": timeout})

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        if url in self.streaming_responses:
            return self.streaming_responses[url]

        # Default streaming response
        return [
            '{"response": "Mock ", "done": false}',
            '{"response": "streaming ", "done": false}',
            '{"response": "response", "done": true}',
        ]

    async def close(self) -> None:
        """Mock close operation."""
        self.call_log.append({"method": "CLOSE"})


# Fallback HTTP client for environments without httpx

    def get_calls(self) -> list[dict[str, Any]]:
        """Get log of all API calls made."""
        return self.call_log.copy()

    def set_exception(self, exception: Exception):
        """Set exception to raise on next request."""
        self.should_raise = exception

    def set_response(self, method: str, url: str, response: HttpResponse):
        """Set response for specific method and URL."""
        key = f"{method.upper()}:{url}"
        self.responses[key] = response

    def set_streaming_response(self, url: str, lines: list[str]):
        """Set streaming response for URL."""
        self.streaming_responses[url] = lines



# ============================================================================
# RETRIEVAL MOCKS
# ============================================================================

class MockQueryProcessor:
    """Mock queryprocessor for testing."""

    def __init__(self, mock_responses: dict[str, ProcessedQuery] | None | None = None):
        """Initialize with optional mock responses."""
        self.mock_responses = mock_responses or {}
        self.call_history: list[dict[str, Any]] = []

    def process_query(self, query: str, context: dict[str, Any] | None = None) -> ProcessedQuery:
        """Mock query processing."""
        self.call_history.append({"query": query, "context": context})

        if query in self.mock_responses:
            return self.mock_responses[query]

        # Default mock response
        return ProcessedQuery(
            original=query,
            processed=query.lower(),
            query_type="general",
            keywords=query.split(),
            expanded_terms=[f"expanded_{word}" for word in query.split()[:3]],
            metadata={"mock": True},
        )


    def set_mock_response(self, query: str, response: ProcessedQuery) -> None:
        """Set mock response for specific query."""
        self.mock_responses[query] = response


class MockCategorizer:
    """Mock categorizer for testing."""

    def __init__(self, mock_responses: dict[str, CategoryMatch] | None = None):
        """Initialize with optional mock responses."""
        self.mock_responses = mock_responses or {}
        self.call_history: list[dict[str, Any]] = []

    def categorize_query(self, query: str, scope_context: dict[str, Any] | None = None) -> CategoryMatch:
        """Mock query categorization."""
        self.call_history.append({"query": query, "scope_context": scope_context})

        if query in self.mock_responses:
            return self.mock_responses[query]

        # Default mock response
        if "api" in query.lower() or "kod" in query.lower():
            category = CategoryType.TECHNICAL
            strategy = "dense"
        elif "kultura" in query.lower() or "culture" in query.lower():
            category = CategoryType.CULTURAL
            strategy = "cultural_context"
        else:
            category = CategoryType.GENERAL
            strategy = "hybrid"

        return CategoryMatch(
            category=category,
            confidence=0.8,
            matched_patterns=[],
            cultural_indicators=[],
            complexity=QueryComplexity.MODERATE,
            retrieval_strategy=strategy,
        )


    def set_mock_response(self, query: str, response: CategoryMatch) -> None:
        """Set mock response for specific query."""
        self.mock_responses[query] = response


class MockSearchEngine:
    """Mock searchengine for testing."""

    def __init__(self, mock_results: list[SearchResult] | None = None):
        """Initialize with optional mock results."""
        self.mock_results = mock_results or []
        self.call_history: list[dict[str, Any]] = []
        self.delay_seconds = 0.0  # Simulate search delay

    def create_mock_results(self, count: int = 5, base_similarity: float = 0.8) -> None:
        """Create mock search results for testing."""
        results = []
        for i in range(count):
            similarity = max(0.1, base_similarity - (i * 0.1))
            results.append(
                SearchResult(
                    content=f"Mock document {i + 1} content with relevant information",
                    metadata={"source": f"doc_{i + 1}", "mock": True},
                    similarity_score=similarity,
                    final_score=similarity,
                    boosts={},
                )
            )

        self.mock_results = results


    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for performance testing."""
        self.delay_seconds = seconds

    async def search(self, query_text: str, k: int = 5, similarity_threshold: float = 0.3) -> list[SearchResult]:
        """Mock search operation."""
        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        self.call_history.append({"query_text": query_text, "k": k, "similarity_threshold": similarity_threshold})

        # Filter mock results by threshold and limit
        filtered_results = [result for result in self.mock_results if result.similarity_score >= similarity_threshold]

        return filtered_results[:k]

    def set_mock_results(self, results: list[SearchResult]) -> None:
        """Set mock search results."""
        self.mock_results = results


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self, rerank_enabled: bool = True):
        """Initialize mock reranker."""
        self.rerank_enabled = rerank_enabled
        self.call_history: list[dict[str, Any]] = []
        self.delay_seconds = 0.0

    def set_delay(self, seconds: float) -> None:
        """Set artificial delay for performance testing."""
        self.delay_seconds = seconds

    async def rerank(
        self, query: str, documents: list[dict[str, Any]], category: str | None = None
    ) -> list[dict[str, Any]]:
        """Mock reranking operation."""
        if self.delay_seconds > 0:
            import asyncio

            await asyncio.sleep(self.delay_seconds)

        self.call_history.append({"query": query, "document_count": len(documents), "category": category})

        if not self.rerank_enabled:
            return documents

        # Simple mock reranking - reverse order to simulate change
        reranked = documents.copy()
        reranked.reverse()

        # Update final scores to show reranking effect
        for i, doc in enumerate(reranked):
            doc["final_score"] = max(0.1, doc.get("final_score", 0.5) + (0.1 * (len(reranked) - i)))
            doc["reranked"] = True

        return reranked



class MockModelLoader:
    """Mock modelloader for testing."""

    def __init__(self):
        get_system_logger()
        log_component_start("mock_loader", "init", loader_type="Mock")
        self.models = {}
        self.available_models = set()
        self.call_log = []
        self.should_raise = None
        log_component_end("mock_loader", "init", "Mock loader initialized")

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    def get_calls(self):
        """Get log of all method calls made."""
        return self.call_log.copy()

    def is_model_available(self, model_name: str) -> bool:
        """Mock model availability check."""
        get_system_logger()
        log_component_start("mock_loader", "is_model_available", model=model_name, is_mock=True)

        self.call_log.append({"method": "is_model_available", "model_name": model_name})
        available = model_name in self.available_models

        log_decision_point("mock_loader", "is_model_available", f"model={model_name}", f"mock_available={available}")
        log_component_end("mock_loader", "is_model_available", f"Mock model {model_name}: {available}")
        return available


    def load_model(self, model_name: str, cache_dir: str, device: str, **kwargs) -> EmbeddingModel:
        """Mock model loading."""
        logger = get_system_logger()
        log_component_start("mock_loader", "load_model", model=model_name, device=device, is_mock=True)

        self.call_log.append(
            {
                "method": "load_model",
                "model_name": model_name,
                "cache_dir": cache_dir,
                "device": device,
                "kwargs": kwargs,
            }
        )

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            logger.debug("mock_loader", "load_model", f"Raising configured exception: {exception}")
            raise exception

        if model_name in self.models:
            logger.debug("mock_loader", "load_model", f"Returning pre-configured mock model: {model_name}")
            log_component_end("mock_loader", "load_model", f"Loaded pre-configured mock: {model_name}")
            return self.models[model_name]

        # Return default mock model
        logger.debug("mock_loader", "load_model", f"Creating default mock model: {model_name}")
        mock_model = MockEmbeddingModel(model_name, device)
        log_component_end("mock_loader", "load_model", f"Created default mock: {model_name}")
        return mock_model

    def set_available_models(self, model_names):
        """Set list of available models."""
        self.available_models = set(model_names)

    def set_exception(self, exception: Exception):
        """Set exception to raise on next load."""
        self.should_raise = exception

    def set_model(self, model_name: str, model: EmbeddingModel):
        """Set mock model for specific name."""
        self.models[model_name] = model
        self.available_models.add(model_name)



# ============================================================================
# VECTORDB MOCKS
# ============================================================================

class MockEmbeddingModel:
    """Mock embeddingmodel for testing."""

    def __init__(self, model_name: str = "mock-model", device: str = "cpu"):
        self.model_name = model_name
        self._device = device
        self._max_seq_length = 512
        self._embedding_dim = 1024
        self.call_log: list[dict[str, Any]] = []

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()


    def device(self) -> str:
        """Get current device."""
        return self._device

    @property
    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True, **kwargs):
        """Mock embedding generation."""
        import numpy as np

        self.call_log.append(
            {
                "method": "encode",
                "num_texts": len(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "kwargs": kwargs,
            }
        )

        # Generate mock embeddings
        embeddings = np.random.rand(len(texts), self._embedding_dim).astype(np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)

        return embeddings

    @property
    def get_calls(self):
        """Get call log."""
        return self.call_log.copy()

    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim

    def max_seq_length(self) -> int:
        """Get maximum sequence length."""
        return self._max_seq_length

    def set_embedding_dimension(self, dim: int):
        """Set embedding dimension for testing."""
        self._embedding_dim = dim


class MockDeviceDetector:
    """Mock devicedetector for testing."""

    def __init__(self):
        self.available_devices = {"cpu"}
        self.device_infos = {
            "cpu": DeviceInfo(
                device_type="cpu", device_name="Mock CPU", available_memory=8192, device_properties={"cores": 4}
            )
        }
        self.best_device = "cpu"
        self.call_log = []
        self.should_raise = None

    def clear_calls(self):
        """Clear call log."""
        self.call_log.clear()

    def detect_best_device(self, preferred_device: str = "auto") -> DeviceInfo:
        """Mock device detection."""
        self.call_log.append({"method": "detect_best_device", "preferred_device": preferred_device})

        if self.should_raise:
            exception = self.should_raise
            self.should_raise = None
            raise exception

        if preferred_device != "auto" and preferred_device in self.device_infos:
            return self.device_infos[preferred_device]

        return self.device_infos[self.best_device]

    def get_calls(self):
        """Get log of all method calls made."""
        return self.call_log.copy()

    def is_device_available(self, device: str) -> bool:
        """Mock device availability check."""
        self.call_log.append({"method": "is_device_available", "device": device})

        return device in self.available_devices



# ============================================================================
# CLI MOCKS
# ============================================================================

    def set_available_devices(self, devices):
        """Set list of available devices."""
        self.available_devices = set(devices)

    def set_best_device(self, device: str):
        """Set the device that should be returned as best."""
        self.best_device = device

    def set_device_info(self, device: str, info: DeviceInfo):
        """Set device info for specific device."""
        self.device_infos[device] = info

    def set_exception(self, exception: Exception):
        """Set exception to raise on next detection."""
        self.should_raise = exception


class MockEmbeddingProvider:
    """Mock embeddingprovider for testing."""

    def __init__(self, dimension: int = 384, deterministic: bool = True):
        """
        Initialize mock embedding provider.

        Args:
            dimension: Embedding vector dimension
            deterministic: If True, same text produces same embedding
        """
        self.dimension = dimension
        self.deterministic = deterministic
        self._embedding_cache: dict[str, np.ndarray] = {}
        self.logger = get_system_logger()

    async def encode_text(self, text: str) -> np.ndarray:
        """Generate mock embedding for text."""
        if self.deterministic and text in self._embedding_cache:
            return self._embedding_cache[text]

        # Generate deterministic or random embedding
        if self.deterministic:
            # Use hash of text for deterministic embedding
            text_hash = hash(text) % (2**31)  # Ensure positive
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            self._embedding_cache[text] = embedding
        else:
            embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

        self.logger.debug(
            "search_providers", "text_to_embedding", f"Generated embedding for text (length: {len(text)})"
        )
        return embedding



class MockVectorSearchProvider:
    """Mock vectorsearchprovider for testing."""

    def __init__(self):
        """Initialize mock search provider."""
        self.documents = {}  # id -> {"content": str, "embedding": np.ndarray, "metadata": dict}
        self.logger = get_system_logger()

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True


    def add_document(self, doc_id: str, content: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None):
        """Add document for testing."""
        self.documents[doc_id] = {"content": content, "embedding": embedding, "metadata": metadata or {}}

    async def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Mock embedding-based search."""
        if not self.documents:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Calculate similarities
        similarities = []
        for doc_id, doc_data in self.documents.items():
            # Skip if filters don't match
            if filters and not self._matches_filters(doc_data["metadata"], filters):
                continue

            doc_embedding = doc_data["embedding"]
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            distance = 1.0 - similarity  # Convert to distance

            similarities.append(
                {
                    "id": doc_id,
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "distance": max(0.0, distance),  # Clamp to non-negative
                }
            )

        # Sort by distance (ascending - lower is better)
        similarities.sort(key=lambda x: x["distance"])

        # Limit results
        similarities = similarities[:top_k]

        # Format as ChromaDB-style results
        ids = [[item["id"] for item in similarities]]
        documents = [[item["content"] for item in similarities]]
        metadatas = [[item["metadata"] for item in similarities]]
        distances = [[item["distance"] for item in similarities]]

        return {"ids": ids, "documents": documents, "metadatas": metadatas, "distances": distances}

    async def search_by_text(
        self, query_text: str, top_k: int, filters: dict[str, Any] | None = None, include_metadata: bool = True
    ) -> dict[str, Any]:
        """Mock text-based search using simple keyword matching."""
        if not self.documents:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_terms = set(query_text.lower().split())
        scores = []

        for doc_id, doc_data in self.documents.items():
            # Skip if filters don't match
            if filters and not self._matches_filters(doc_data["metadata"], filters):
                continue

            content = doc_data["content"].lower()
            doc_terms = set(content.split())

            # Simple keyword overlap score
            if query_terms:
                overlap = len(query_terms.intersection(doc_terms))
                score = overlap / len(query_terms)
            else:
                score = 0.0

            # Boost for exact phrase matches
            if query_text.lower() in content:
                score *= 1.5

            distance = 1.0 - min(1.0, score)  # Convert to distance

            scores.append(
                {"id": doc_id, "content": doc_data["content"], "metadata": doc_data["metadata"], "distance": distance}
            )

        # Sort by distance (ascending)
        scores.sort(key=lambda x: x["distance"])

        # Limit results
        scores = scores[:top_k]

        # Format as ChromaDB-style results
        ids = [[item["id"] for item in scores]]
        documents = [[item["content"] for item in scores]]
        metadatas = [[item["metadata"] for item in scores]]
        distances = [[item["distance"] for item in scores]]

        return {"ids": ids, "documents": documents, "metadatas": metadatas, "distances": distances}

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get document by ID."""
        if document_id in self.documents:
            return {
                "id": document_id,
                "content": self.documents[document_id]["content"],
                "metadata": self.documents[document_id]["metadata"],
            }
        return None


class MockCollection:
    """Mock collection for testing."""

    def __init__(self, name: str = "test_collection"):
        self._name = name
        self._metadata = {"hnsw:space": "cosine"}
        self._documents: dict[str, dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """Add documents to mock collection."""
        for i, doc_id in enumerate(ids):
            self._documents[doc_id] = {
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i] if embeddings else None,
            }
        self.logger.debug(f"Added {len(ids)} documents to mock collection")

    def count(self) -> int:
        """Get document count."""
        return len(self._documents)

    @property
    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        """Delete documents from mock collection."""
        if ids:
            for doc_id in ids:
                self._documents.pop(doc_id, None)
        else:
            # Mock: delete all if no specific filter
            self._documents.clear()

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get documents from mock collection."""
        include = include or ["documents", "metadatas"]

        if ids:
            doc_items = [(doc_id, self._documents[doc_id]) for doc_id in ids if doc_id in self._documents]
        else:
            doc_items = list(self._documents.items())

        results: dict[str, Any] = {"ids": []}
        if "documents" in include:
            results["documents"] = []
        if "metadatas" in include:
            results["metadatas"] = []

        for doc_id, doc_data in doc_items:
            results["ids"].append(doc_id)
            if "documents" in include:
                results["documents"].append(doc_data["document"])
            if "metadatas" in include:
                results["metadatas"].append(doc_data["metadata"])

        return results

    def metadata(self) -> dict[str, Any]:
        """Get collection metadata."""
        return self._metadata


    def name(self) -> str:
        """Get collection name."""
        return self._name

    @property
    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[np.ndarray] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> VectorSearchResults:
        """Mock query returns first n_results documents."""
        include = include or ["documents", "metadatas", "distances"]

        # Simple mock: return first n_results documents
        doc_items = list(self._documents.items())[:n_results]

        search_results = []
        for doc_id, doc_data in doc_items:
            result = VectorSearchResult(
                id=str(doc_id),
                content=doc_data.get("document", ""),
                metadata=doc_data.get("metadata", {}),
                distance=0.5,  # Mock distance
            )
            search_results.append(result)

        return VectorSearchResults(
            results=search_results,
            total_count=len(search_results),
            search_time_ms=0.0,  # Mock timing
        )

    def update(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        """Update documents in mock collection."""
        for i, doc_id in enumerate(ids):
            if doc_id in self._documents:
                if documents:
                    self._documents[doc_id]["document"] = documents[i]
                if metadatas:
                    self._documents[doc_id]["metadata"] = metadatas[i]
                if embeddings:
                    self._documents[doc_id]["embedding"] = embeddings[i]


class MockDatabase:
    """Mock database for testing."""

    def __init__(
        self,
        db_path: str = "/tmp/mock_db",
        distance_metric: str = "cosine",
        persist: bool = True,
        allow_reset: bool = True,
    ):
        self.db_path = db_path
        self.distance_metric = distance_metric
        self.persist = persist
        self.allow_reset = allow_reset
        self._collections: dict[str, MockCollection] = {}
        self.logger = logging.getLogger(__name__)

    def create_collection(self, name: str, reset_if_exists: bool = False) -> VectorCollection:
        """Create or get mock collection."""
        if reset_if_exists and name in self._collections:
            del self._collections[name]

        if name not in self._collections:
            self._collections[name] = MockCollection(name)

        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete mock collection."""
        if name in self._collections:
            del self._collections[name]

    def get_collection(self, name: str) -> VectorCollection:
        """Get existing mock collection."""
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        return self._collections[name]

    def list_collections(self) -> list[str]:
        """List all mock collections."""
        return list(self._collections.keys())

    def reset(self) -> None:
        """Reset mock database."""
        self._collections.clear()



class MockStorage:
    """Mock storage for testing."""

    def get_document_count(self, collection_name: str) -> int:
        return 42  # Mock count


    def list_collections(self) -> list[str]:
        return ["collection1", "collection2", "user_dev_user_hr"]



# ============================================================================
# CLI MOCKS
# ============================================================================

class MockOutputWriter:
    """Mock outputwriter for testing."""

    def __init__(self):
        self.written_lines = []

    def flush(self) -> None:
        pass


    def write(self, text: str) -> None:
        self.written_lines.append(text.rstrip("\n"))


class MockRAGSystem:
    """Mock ragsystem for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.initialized = False

    async def initialize(self) -> None:
        if self.should_fail:
            raise Exception("Mock initialization failure")
        self.initialized = True

    async def query(self, query: Any) -> Any:
        if self.should_fail:
            raise Exception("Mock query failure")

        class MockResponse:
            answer = f"Mock answer for: {query['text']}"
            sources = ["mock_doc1.txt", "mock_doc2.txt"]
            retrieved_chunks = [
                {"content": "Mock chunk 1", "similarity_score": 0.9, "final_score": 0.9, "source": "mock_doc1.txt"},
                {"content": "Mock chunk 2", "similarity_score": 0.8, "final_score": 0.8, "source": "mock_doc2.txt"},
            ]

        return MockResponse()

    async def add_documents(self, document_paths: list[str]) -> dict[str, Any]:
        if self.should_fail:
            raise Exception("Mock processing failure")

        return {
            "processed_documents": len(document_paths),
            "failed_documents": 0,
            "total_chunks": len(document_paths) * 5,
            "processing_time": 1.0,
            "documents_per_second": len(document_paths),
        }



class MockConfigLoader:
    """Mock configloader for testing."""

    def get_shared_config(self) -> dict[str, Any]:
        return {"key": "value"}

    def get_storage_config(self) -> dict[str, Any]:
        return {"storage": "config"}




# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_chromadb_database(
    db_path: str, distance_metric: str = "cosine", persist: bool = True, allow_reset: bool = False
) -> VectorDatabase:
    """Factory function to create ChromaDB database."""
    return ChromaDBDatabase(db_path=db_path, distance_metric=distance_metric, persist=persist, allow_reset=allow_reset)

def create_complex_test_config():
    """Create complex test configuration."""
    config_provider = MockConfigProvider()
    logger_provider = MockLoggerProvider()
    return config_provider, logger_provider

def create_config_provider(config_loader_func=None) -> ConfigProvider:
    """Create default configuration provider."""
    return DefaultConfigProvider(config_loader=config_loader_func)

def create_default_setup(logger_name: str | None = None) -> tuple:
    """
    Create production setup with real components.

    Args:
        logger_name: Optional logger name override

    Returns:
        Tuple of (config_provider, pattern_provider, logger_provider)
    """
    config_provider = DefaultConfigProvider()
    pattern_provider = DefaultPatternProvider()
    logger_provider = StandardLoggerProvider(logger_name or __name__)

    return config_provider, pattern_provider, logger_provider


# Backward compatibility aliases
ProductionConfigProvider = DefaultConfigProvider
ProductionPatternProvider = DefaultPatternProvider
create_production_setup = create_default_setup

def create_embedding_provider(model_name: str = "BAAI/bge-m3", device: str = "cpu") -> EmbeddingProvider:
    """Create production embedding provider."""
    return SentenceTransformerEmbeddingProvider(model_name=model_name, device=device)

def create_hierarchical_retriever(
    search_engine, language: str = "hr", reranker=None, enable_performance_tracking: bool = True
):
    """
    Create hierarchical retriever with real components.

    Args:
        search_engine: Search engine instance
        language: Language for processing components
        reranker: Optional reranker
        enable_performance_tracking: Whether to track performance

    Returns:
        HierarchicalRetriever instance
    """
    get_system_logger()
    log_component_start(
        "hierarchical_retriever_providers",
        "create_hierarchical_retriever",
        language=language,
        has_reranker=reranker is not None,
        performance_tracking=enable_performance_tracking,
    )

    # Import the actual HierarchicalRetriever class
    from .hierarchical_retriever import HierarchicalRetriever

    # Create components
    query_processor = QueryProcessor(language)
    categorizer = Categorizer(language)
    search_engine_adapter = SearchEngineAdapter(search_engine)
    reranker_adapter = RerankerAdapter(reranker, language) if reranker else None

    # Use Python's standard logger for HierarchicalRetriever (different from structured logger)
    import logging

    class StandardLoggerProvider:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def info(self, message: str) -> None:
            self.logger.info(message)

        def debug(self, message: str) -> None:
            self.logger.debug(message)

        def error(self, message: str) -> None:
            self.logger.error(message)

    hierarchical_logger = StandardLoggerProvider()

    # Create production configuration
    config = RetrievalConfig(
        default_max_results=5,
        similarity_thresholds=get_similarity_thresholds(),
        boost_weights={
            "keyword": 0.2,
            "technical": 0.1,
            "exact_match": 0.2,
            "temporal": 0.15,
            "faq": 0.1,
            "comparative": 0.1,
        },
        strategy_mappings={},
        performance_tracking=enable_performance_tracking,
    )

    log_data_transformation(
        "hierarchical_retriever_providers",
        "component_assembly",
        f"Input: search engine + language '{language}'",
        f"Output: HierarchicalRetriever with {len(config.similarity_thresholds)} strategies",
        language=language,
        strategies_count=len(config.similarity_thresholds),
        has_reranker=reranker is not None,
    )

    log_performance_metric(
        "hierarchical_retriever_providers", "create_hierarchical_retriever", "max_results", config.default_max_results
    )

    # Create and return the actual HierarchicalRetriever instance
    retriever = HierarchicalRetriever(
        query_processor=query_processor,
        categorizer=categorizer,
        search_engine=search_engine_adapter,
        config=config,
        reranker=reranker_adapter,
        logger_provider=hierarchical_logger,
    )

    log_component_end(
        "hierarchical_retriever_providers",
        "create_hierarchical_retriever",
        f"Successfully created HierarchicalRetriever for language '{language}'",
        language=language,
        component_type="production",
    )

    return retriever

def create_invalid_config(language: str = "hr") -> PromptConfig:
    """Create configuration with missing required templates for testing error scenarios."""
    # Missing USER template for GENERAL category
    category_templates = {
        CategoryType.GENERAL: {
            PromptType.SYSTEM: "You are a helpful assistant."
            # Missing PromptType.USER intentionally
        }
    }

    return PromptConfig(
        category_templates=category_templates,
        messages={"no_context": "No context."},
        formatting={"source_label": "Source"},
        language=language,
    )




# ============================================================================
# PREPROCESSING MOCKS
# ============================================================================

def create_minimal_config(language: str = "hr") -> PromptConfig:
    """Create minimal configuration for basic testing."""
    category_templates = {
        CategoryType.GENERAL: {PromptType.SYSTEM: "Basic assistant.", PromptType.USER: "{query} - {context}"}
    }

    return PromptConfig(
        category_templates=category_templates,
        messages={"no_context": "No context."},
        formatting={"source_label": "Source"},
        language=language,
    )

def create_minimal_test_config():
    """Create minimal test configuration."""
    config_provider = MockConfigProvider()
    logger_provider = MockLoggerProvider()
    return config_provider, logger_provider

def create_minimal_test_providers():
    """Create minimal test providers for cleaning."""
    config_provider = MockConfigProvider()
    logger_provider = MockLoggerProvider()
    environment_provider = MockEnvironmentProvider()
    return config_provider, logger_provider, environment_provider


# ============================================================================
# RETRIEVAL MOCKS
# ============================================================================

def create_mock_cli(should_fail: bool = False, output_writer: OutputWriterProtocol | None = None) -> MultiTenantRAGCLI:
    """Create a fully mocked CLI for testing."""
    output_writer = output_writer or MockOutputWriter()

    def mock_rag_factory(language: str, tenant_context=None, scope: str = "user", feature_name: str | None = None):
        return MockRAGSystem(should_fail=should_fail)

    return MultiTenantRAGCLI(
        output_writer=output_writer,
        logger=MockLogger(),
        rag_system_factory=mock_rag_factory,
        folder_manager=MockFolderManager(),
        storage=MockStorage(),
        config_loader=MockConfigLoader(),
    )


# Main entry point
async def main():
    args = parse_cli_arguments(sys.argv[1:])

    # Setup new logging system with configurable backends
    backend_kwargs = {"console": {"level": args.log_level, "colored": True}}

    # Add file logging in debug mode
    if args.log_level.upper() == "DEBUG":
        backend_kwargs["file"] = {"log_file": "logs/rag_debug.log", "format_type": "text"}

    setup_system_logging(["console"], **backend_kwargs)
    log_workflow_event("cli", "STARTUP", "STARTED", command=args.command, language=args.language)

    class StandardOutputWriter:
        def write(self, text: str) -> None:
            sys.stdout.write(text)

        def flush(self) -> None:
            sys.stdout.flush()

    class StandardLogger:
        def info(self, message: str) -> None:
            log_component_info("cli", "OPERATION", message)

        def error(self, message: str) -> None:
            log_component_error("cli", "OPERATION", message)

        def exception(self, message: str) -> None:
            log_component_error("cli", "EXCEPTION", message)

    def real_rag_factory(language: str, tenant_context=None, scope: str = "user", feature_name: str | None = None):
        log_component_info(
            "cli", "RAG_FACTORY", f"Creating system for language={language}, scope={scope}, feature={feature_name}"
        )

        try:
            # Import Tenant and User from factories (where they're now defined)
            from ..utils.factories import Tenant, User, create_complete_rag_system

            tenant_obj = None
            user_obj = None
            if tenant_context and scope in ["user", "tenant"]:
                tenant_obj = Tenant(
                    id=tenant_context.tenant_id, name=tenant_context.tenant_name, slug=tenant_context.tenant_slug
                )
                user_obj = User(
                    id=tenant_context.user_id,
                    tenant_id=tenant_obj.id,
                    email=tenant_context.user_email,
                    username=tenant_context.user_username,
                    full_name=tenant_context.user_full_name,
                )

            rag_system = create_complete_rag_system(
                language=language,
                tenant=tenant_obj if scope in ["user", "tenant"] else None,
                user=user_obj if scope == "user" else None,
                scope=scope,
                feature_name=feature_name,
            )

            log_component_info("cli", "RAG_FACTORY", f"System created successfully for {language}, scope={scope}")
            return rag_system

        except Exception as e:
            log_component_error("cli", "RAG_FACTORY", f"Failed to create system for {language}", e)
            raise e

    from ..utils.folder_manager import TenantFolderManager
    from ..utils.folder_manager_providers import (
        ProductionConfigProvider,
        ProductionFileSystemProvider,
        StandardLoggerProvider,
    )

    real_folder_manager = TenantFolderManager(
        config_provider=ProductionConfigProvider(),
        filesystem_provider=ProductionFileSystemProvider(),
        logger_provider=StandardLoggerProvider(),
    )

    class VectorDatabaseStorage:
        def __init__(self):
            self._vector_db = None

        def _ensure_initialized(self):
            if self._vector_db is None:
                from ..utils.config_loader import get_paths_config
                from ..vectordb.database_factory import create_vector_database

                paths_config = get_paths_config()
                db_path_template = paths_config["chromadb_path_template"]
                data_base_dir = paths_config["data_base_dir"]
                db_path = db_path_template.format(data_base_dir=data_base_dir, tenant_slug="development")
                self._vector_db = create_vector_database(db_path=db_path)

        def list_collections(self) -> list[str]:
            try:
                self._ensure_initialized()
                return self._vector_db.list_collections()
            except Exception:
                return []

        def get_document_count(self, collection_name: str) -> int:
            try:
                self._ensure_initialized()
                return self._vector_db.get_collection_size(collection_name)
            except Exception:
                return 0

    class TomlConfigLoader:
        def get_shared_config(self) -> dict[str, Any]:
            from ..utils.config_loader import get_shared_config

            return get_shared_config()

        def get_storage_config(self) -> dict[str, Any]:
            from ..utils.config_loader import load_config

            config = load_config("config")
            return config["vectordb"]

    cli = MultiTenantRAGCLI(
        output_writer=StandardOutputWriter(),
        logger=StandardLogger(),
        rag_system_factory=real_rag_factory,
        folder_manager=real_folder_manager,
        storage=VectorDatabaseStorage(),
        config_loader=TomlConfigLoader(),
    )

    await cli.execute_command(args)
    log_workflow_event("cli", "STARTUP", "COMPLETED")

def create_mock_config_provider(
    templates: dict[str, PromptTemplate] | None = None,
    keyword_patterns: dict[str, list[str]] | None = None,
    formatting: dict[str, str] | None = None,
) -> ConfigProvider:
    """
    Factory function to create mock configuration provider.

    Args:
        templates: Custom templates (uses defaults if None)
        keyword_patterns: Custom keyword patterns (uses defaults if None)
        formatting: Custom formatting (uses defaults if None)

    Returns:
        Mock ConfigProvider instance (uses global MockConfigProvider class)
    """
    # Configure and return global MockConfigProvider
    provider = MockConfigProvider()
    # Mock data will be set by test if needed
    return provider

def create_mock_database(
    db_path: str = "/tmp/mock_db", distance_metric: str = "cosine", persist: bool = True, allow_reset: bool = True
) -> VectorDatabase:
    """Factory function to create mock database."""
    return MockDatabase(db_path=db_path, distance_metric=distance_metric, persist=persist, allow_reset=allow_reset)

def create_mock_embedding_provider(dimension: int = 384) -> EmbeddingProvider:
    """Create mock embedding provider for testing."""
    return MockEmbeddingProvider(dimension=dimension, deterministic=True)

def create_mock_language_provider(
    language: str = "hr", custom_data: dict[str, Any] | None = None
) -> MockLanguageDataProvider:
    """
    Create mock language data provider with optional custom data.

    Args:
        language: Language for default data
        custom_data: Optional custom language data

    Returns:
        MockLanguageDataProvider instance
    """
    provider = MockLanguageDataProvider()

    # Set default data for language
    if language == "hr":
        provider.set_stop_words("hr", {"i", "a", "u", "na", "za", "od", "do", "iz", "s", "sa", "se"})
        provider.set_question_patterns("hr", [r"^što\s", r"^kako\s", r"^kada\s", r"^gdje\s", r"^zašto\s"])
        provider.set_synonym_groups("hr", {"brz": ["brži", "brzo", "hitno"], "dobro": ["odlično", "izvrsno", "super"]})
    else:  # English
        provider.set_stop_words("en", {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"})
        provider.set_question_patterns("en", [r"^what\s", r"^how\s", r"^when\s", r"^where\s", r"^why\s"])
        provider.set_synonym_groups(
            "en", {"fast": ["quick", "rapid", "swift"], "good": ["great", "excellent", "awesome"]}
        )

    # Apply custom data if provided
    if custom_data:
        for key, value in custom_data.items():
            if key == "stop_words":
                provider.set_stop_words(language, set(value))
            elif key == "question_patterns":
                provider.set_question_patterns(language, value)
            elif key == "synonym_groups":
                provider.set_synonym_groups(language, value)
            elif key == "morphological_patterns":
                provider.set_morphological_patterns(language, value)

    return provider

def create_mock_model_loader(should_load_successfully: bool = True, is_loaded: bool = True) -> ModelLoader:
    """
    Factory function to create mock model loader.

    Args:
        should_load_successfully: Whether load_model should succeed
        is_loaded: Whether model should report as loaded

    Returns:
        Mock ModelLoader
    """

    class MockModelLoader:
        def __init__(self):
            self._is_loaded = False
            self._should_succeed = should_load_successfully
            self._target_loaded_state = is_loaded

        def load_model(self, model_name: str, device: str) -> Any:
            if not self._should_succeed:
                raise ValueError(f"Mock failure loading {model_name}")
            self._is_loaded = self._target_loaded_state
            return "mock_model"

        def is_model_loaded(self) -> bool:
            return self._is_loaded

    return MockModelLoader()

def create_mock_ranker(language: str = "hr", config_dict: dict[str, Any] | None = None) -> DocumentRanker:
    """
    Factory function to create mock ranker for testing.

    Args:
        language: Language code
        config_dict: Optional configuration override

    Returns:
        DocumentRanker with mock providers
    """
    # Use mock providers from conftest
    config_provider = create_mock_config_provider(config_dict)
    language_provider = create_mock_language_provider()

    return DocumentRanker(config_provider, language_provider, language)

def create_mock_score_calculator(base_scores: list[float] | None = None, add_noise: bool = False) -> ScoreCalculator:
    """
    Factory function to create mock score calculator.

    Args:
        base_scores: Base scores to return (generates if None)
        add_noise: Whether to add random noise to scores

    Returns:
        Mock ScoreCalculator
    """

    class MockScoreCalculator:
        def __init__(self):
            self.base_scores = base_scores
            self.add_noise = add_noise

        def calculate_scores(self, query_document_pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
            n_pairs = len(query_document_pairs)

            if self.base_scores:
                # Use provided scores, cycling if necessary
                scores = [self.base_scores[i % len(self.base_scores)] for i in range(n_pairs)]
            else:
                # Generate mock scores based on query-document similarity
                scores = []
                for query, doc in query_document_pairs:
                    # Simple mock scoring based on common words
                    query_words = set(query.lower().split())
                    doc_words = set(doc.lower().split())

                    if query_words and doc_words:
                        overlap = len(query_words & doc_words)
                        total_unique = len(query_words | doc_words)
                        score = overlap / total_unique if total_unique > 0 else 0.0
                    else:
                        score = 0.0

                    scores.append(score)

            # Add noise if requested
            if self.add_noise:
                import random

                scores = [max(0.0, min(1.0, score + random.uniform(-0.1, 0.1))) for score in scores]

            return scores

    return MockScoreCalculator()



# ============================================================================
# VECTORDB MOCKS
# ============================================================================

def create_mock_search_provider() -> VectorSearchProvider:
    """Create mock search provider for testing."""
    return MockVectorSearchProvider()

def create_mock_setup(
    query_responses: dict[str, ProcessedQuery] | None = None,
    category_responses: dict[str, CategoryMatch] | None = None,
    search_results: list[SearchResult] | None = None,
    enable_reranking: bool = True,
    search_delay: float = 0.0,
) -> tuple:
    """
    Create complete mock setup for testing.

    Args:
        query_responses: Mock query processor responses
        category_responses: Mock categorizer responses
        search_results: Mock search results
        enable_reranking: Whether to enable mock reranking
        search_delay: Artificial delay for search operations

    Returns:
        Tuple of (query_processor, categorizer, search_engine, reranker, logger, config)
    """
    # Create mock components
    query_processor = MockQueryProcessor(query_responses)
    categorizer = MockCategorizer(category_responses)
    search_engine = MockSearchEngine(search_results)
    reranker = MockReranker(enable_reranking)
    logger = MockLoggerProvider()

    # Set delays if specified
    search_engine.set_delay(search_delay)
    reranker.set_delay(search_delay)

    # Create default search results if none provided
    if search_results is None:
        search_engine.create_mock_results(count=5)

    # Create test configuration
    config = RetrievalConfig(
        default_max_results=5,
        similarity_thresholds=get_similarity_thresholds(),
        boost_weights={
            "keyword": 0.2,
            "technical": 0.1,
            "exact_match": 0.2,
            "temporal": 0.15,
            "faq": 0.1,
            "comparative": 0.1,
        },
        strategy_mappings={},
        performance_tracking=True,
    )

    return query_processor, categorizer, search_engine, reranker, logger, config

def create_mock_stop_words_provider(
    croatian_stop_words: set | None = None, english_stop_words: set | None = None
) -> StopWordsProvider:
    """
    Factory function to create mock stop words provider.

    Args:
        croatian_stop_words: Custom Croatian stop words
        english_stop_words: Custom English stop words

    Returns:
        Mock StopWordsProvider
    """

    class MockStopWordsProvider:
        def __init__(self):
            self.default_croatian = croatian_stop_words or {
                "i",
                "je",
                "su",
                "da",
                "se",
                "na",
                "u",
                "za",
                "od",
                "do",
                "s",
                "sa",
                "kao",
                "ima",
                "biti",
                "bilo",
                "mogu",
                "možete",
                "ili",
                "ako",
                "kada",
                "gdje",
            }
            self.default_english = english_stop_words or {
                "the",
                "is",
                "are",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "up",
                "about",
                "into",
                "through",
                "during",
            }

        def get_stop_words(self, language: str) -> set:
            if language == "hr":
                return self.default_croatian
            elif language == "en":
                return self.default_english
            else:
                return set()  # Unknown language

    return MockStopWordsProvider()

def create_mock_storage() -> VectorStorage:
    """Factory function to create mock storage for testing."""
    # TODO: Implement mock database for testing
    # from .storage_factories import create_mock_database
    # mock_db = create_mock_database()
    raise NotImplementedError("Mock storage not implemented yet")
    # return VectorStorage(mock_db)

def create_multilingual_test_providers():
    """Create multilingual test providers for cleaning."""
    config_provider = MockConfigProvider()
    logger_provider = MockLoggerProvider()
    environment_provider = MockEnvironmentProvider()
    return config_provider, logger_provider, environment_provider

def create_prompt_builder(logger_name: str | None = None) -> tuple:
    """
    Create prompt builder with real components.

    Args:
        logger_name: Optional logger name override

    Returns:
        Tuple of (config_provider, logger_provider)
    """
    config_provider = ConfigProvider()
    logger_provider = StandardLoggerProvider(logger_name or __name__)

    return config_provider, logger_provider

def create_providers(language: str = "hr") -> tuple:
    """
    Create providers for query processing.

    Args:
        language: Language for providers

    Returns:
        Tuple of (config, language_provider, config_provider)
    """
    # Import real config provider
    from ..utils.config_protocol import get_config_provider

    config_provider = get_config_provider()

    # Create language data provider
    language_provider = create_language_provider(config_provider)

    # Create configuration
    config = create_default_config(language, config_provider)

    return config, language_provider, config_provider

def create_query_processor(language: str = "hr"):
    """Create query processor with all dependencies."""
    from .query_processor import MultilingualQueryProcessor

    config, language_provider, config_provider = create_providers(language)

    return MultilingualQueryProcessor(config=config, language_data_provider=language_provider)



# ============================================================================
# VECTORDB MOCKS
# ============================================================================

def create_test_categorization_setup():
    """Create test categorization setup."""
    config_provider = MockConfigProvider()
    logger_provider = MockLoggerProvider()
    return config_provider, logger_provider

def create_test_config(
    language: str = "hr", include_followup: bool = True, include_technical: bool = True, include_cultural: bool = True
) -> PromptConfig:
    """Create test configuration with customizable parameters."""

    category_templates = {}

    # Always include GENERAL
    category_templates[CategoryType.GENERAL] = {
        PromptType.SYSTEM: "You are a helpful assistant. Answer based on context.",
        PromptType.USER: "Q: {query}\nContext: {context}\nA:",
    }

    if include_followup:
        category_templates[CategoryType.GENERAL][PromptType.FOLLOWUP] = (
            "Previous: {original_query} -> {original_answer}\nNew: {followup_query}\nAnswer:"
        )

    if include_technical:
        category_templates[CategoryType.TECHNICAL] = {
            PromptType.SYSTEM: "Technical expert. Provide detailed answers.",
            PromptType.USER: "Tech Q: {query}\nDocs: {context}\nTech A:",
        }

    if include_cultural:
        category_templates[CategoryType.CULTURAL] = {
            PromptType.SYSTEM: "Cultural expert. Provide culturally aware answers.",
            PromptType.USER: "Cultural Q: {query}\nContext: {context}\nCultural A:",
        }

    messages = {
        "no_context": f"No context available ({language})",
        "error_template_missing": f"Template missing ({language})",
    }

    formatting = {
        "source_label": "Source" if language == "en" else "Izvor",
        "truncation_indicator": "..." if language == "en" else "...",
        "min_chunk_size": "100",
    }

    return PromptConfig(
        category_templates=category_templates, messages=messages, formatting=formatting, language=language
    )

def create_test_folder_manager(config: FolderConfig | None = None):
    """Create test folder manager setup."""
    from src.utils.folder_manager import FolderManager
    from src.utils.folder_manager_providers import DefaultFileSystemProvider
    
    config_provider = MockConfigProvider() 
    file_system_provider = DefaultFileSystemProvider()
    logger_provider = MockLoggerProvider()
    
    folder_manager = FolderManager(
        config=config,
        config_provider=config_provider,
        file_system_provider=file_system_provider,
        logger_provider=logger_provider
    )
    return folder_manager, config_provider, file_system_provider, logger_provider

def create_test_language_manager(
    settings: LanguageSettings | None = None,
    patterns: LanguagePatterns | None = None,
) -> tuple:
    """Create test language manager setup."""
    from src.utils.language_manager import LanguageManager
    config_provider, pattern_provider, logger_provider = create_mock_setup(settings, patterns)
    language_manager = LanguageManager(
        config_provider=config_provider,
        pattern_provider=pattern_provider,
        logger_provider=logger_provider
    )
    return language_manager, config_provider, pattern_provider, logger_provider

def create_test_patterns(
    detection_patterns: dict[str, list[str]] | None = None, stopwords: dict[str, set[str]] | None = None
) -> LanguagePatterns:
    """Create test patterns with customizable parameters."""
    default_detection = {
        "hr": ["što", "kako", "gdje", "kada", "zašto"],
        "en": ["what", "how", "where", "when", "why"],
        "multilingual": ["what", "kako", "where", "gdje"],
    }

    default_stopwords = {
        "hr": {"i", "u", "na", "za", "je"},
        "en": {"a", "and", "the", "of", "to"},
        "multilingual": {"i", "and", "u", "the"},
    }

    return LanguagePatterns(
        detection_patterns=detection_patterns or default_detection, stopwords=stopwords or default_stopwords
    )

def create_test_prompt_builder(language: str = "hr"):
    """Create test prompt builder."""
    from src.generation.enhanced_prompt_templates import EnhancedPromptBuilder
    config_provider = MockConfigProvider()
    logger_provider = MockLoggerProvider()
    
    builder = EnhancedPromptBuilder(
        language=language,
        config_provider=config_provider,
        logger_provider=logger_provider
    )
    return builder, config_provider, logger_provider

def create_test_providers(
    language: str = "hr",
    custom_config: dict[str, Any] | None = None,
    custom_language_data: dict[str, Any] | None = None,
) -> tuple:
    """
    Create complete set of test providers for query processing.

    Args:
        language: Language for providers
        custom_config: Optional custom configuration
        custom_language_data: Optional custom language data

    Returns:
        Tuple of (config, language_provider, config_provider)
    """
    # Create mock config provider
    config_provider = MockConfigProvider()

    # Set default or custom config
    if custom_config:
        config_provider.set_config("config", custom_config)
    else:
        config_provider.set_config(
            "config",
            {
                "query_processing": {
                    "expand_synonyms": True,
                    "normalize_case": True,
                    "remove_stopwords": True,
                    "min_query_length": 3,
                    "max_query_length": 50,
                    "max_expanded_terms": 10,
                    "enable_morphological_analysis": False,
                    "use_query_classification": True,
                    "enable_spell_check": False,
                }
            },
        )

    # Create language data provider
    language_provider = create_mock_language_provider(language, custom_language_data)

    # Create configuration
    config = create_default_config(language, config_provider)

    return config, language_provider, config_provider

def create_test_settings(
    supported_languages: list[str] | None = None,
    default_language: str = "hr",
    auto_detect: bool = True,
    embedding_model: str = "BAAI/bge-m3",
) -> LanguageSettings:
    """Create test settings with customizable parameters."""
    if supported_languages is None:
        supported_languages = ["hr", "en", "multilingual"]

    language_names = {}
    for lang in supported_languages:
        if lang == "hr":
            language_names[lang] = "Croatian"
        elif lang == "en":
            language_names[lang] = "English"
        elif lang == "multilingual":
            language_names[lang] = "Multilingual"
        else:
            language_names[lang] = lang.upper()

    return LanguageSettings(
        supported_languages=supported_languages,
        default_language=default_language,
        auto_detect=auto_detect,
        fallback_language=default_language,
        language_names=language_names,
        embedding_model=embedding_model,
        chunk_size=512,
        chunk_overlap=50,
    )

def create_vector_database(
    db_path: str, distance_metric: str = "cosine", persist: bool = True, allow_reset: bool = False
) -> VectorDatabase:
    """Factory function to create vector database."""
    return create_chromadb_database(
        db_path=db_path, distance_metric=distance_metric, persist=persist, allow_reset=allow_reset
    )

def create_vector_search_provider(vector_storage_or_collection, embedding_provider=None) -> VectorSearchProvider:
    """Create vector search provider based on configured vector database."""
    from ..utils.config_loader import get_config_section
    from ..utils.logging_factory import get_system_logger

    logger = get_system_logger()
    vectordb_config = get_config_section("config", "vectordb")
    provider = vectordb_config["provider"]

    logger.info(
        "search_providers",
        "create_vector_search_provider",
        f"FACTORY DEBUG: provider={provider} | input_type={type(vector_storage_or_collection).__name__} | input_none={vector_storage_or_collection is None}",
    )

    # Handle both VectorStorage objects and raw collections
    if hasattr(vector_storage_or_collection, "collection"):
        # This is a VectorStorage object, get the collection
        collection = vector_storage_or_collection.collection
        logger.info(
            "search_providers",
            "create_vector_search_provider",
            f"FACTORY DEBUG: VectorStorage.collection={type(collection).__name__ if collection else 'None'} | collection_none={collection is None}",
        )
    else:
        # This is already a collection
        collection = vector_storage_or_collection
        logger.info(
            "search_providers",
            "create_vector_search_provider",
            f"FACTORY DEBUG: Direct collection={type(collection).__name__ if collection else 'None'} | collection_none={collection is None}",
        )

    if collection is None:
        logger.error(
            "search_providers",
            "create_vector_search_provider",
            "FACTORY ERROR: Collection is None - cannot create search provider",
        )
        raise ValueError("Collection is None - cannot create search provider")

    search_provider: VectorSearchProvider
    if provider == "weaviate":
        # For Weaviate, create a search provider wrapper that handles text-to-embedding conversion
        search_provider = WeaviateSearchProvider(collection, embedding_provider)
        logger.info(
            "search_providers",
            "create_vector_search_provider",
            f"FACTORY DEBUG: Created Weaviate provider {type(search_provider).__name__}",
        )
        return search_provider
    elif provider == "chromadb":
        search_provider = ChromaDBSearchProvider(collection, embedding_provider)
        logger.info(
            "search_providers",
            "create_vector_search_provider",
            f"FACTORY DEBUG: Created ChromaDB provider {type(search_provider).__name__}",
        )
        return search_provider
    else:
        raise ValueError(f"Unsupported vector database provider: {provider}")

def log_component_start(component: str, operation: str, **kwargs) -> None:
    """Log component start for AI debugging."""
    logger = get_system_logger()
    details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug("component_start", f"{component}.{operation}", details if details else "Starting operation")

def log_component_end(component: str, operation: str, message: str = "") -> None:
    """Log component end for AI debugging."""
    logger = get_system_logger()
    logger.debug("component_end", f"{component}.{operation}", message)

def log_component_info(component: str, operation: str, message: str) -> None:
    """Log component info for AI debugging."""
    logger = get_system_logger()
    logger.info(component, operation, message)

def log_component_error(component: str, operation: str, message: str, error: Exception | None = None) -> None:
    """Log component error for AI debugging."""
    logger = get_system_logger()
    error_msg = f"{message} | error={error}" if error else message
    logger.error(component, operation, error_msg)

