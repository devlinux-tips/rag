"""
Comprehensive tests for utils/config_loader.py

Tests all configuration loading functionality including:
- Basic config loading and caching
- Language-specific configuration handling
- Error handling and validation
- Global function interface
- Multilingual configuration functions
"""

import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, mock_open, patch

try:
    import toml
except ImportError:
    import tomli_w as toml

from src.utils.config_loader import (
    CONFIG_DIR,
    ConfigError,
    ConfigLoader,
    _config_loader,
    discover_available_languages,
    get_chunking_config,
    get_chroma_config,
    get_cleaning_config,
    get_config_section,
    get_embeddings_config,
    get_extraction_config,
    get_generation_config,
    get_generation_prompts_config,
    get_hybrid_retrieval_config,
    get_language_config,
    get_language_config_file,
    get_language_ranking_features,
    get_language_shared,
    get_language_specific_config,
    get_ollama_config,
    get_paths_config,
    get_performance_config,
    get_pipeline_config,
    get_preprocessing_config,
    get_processing_config,
    get_project_info,
    get_query_processing_config,
    get_ranking_config,
    get_reranking_config,
    get_response_parsing_config,
    get_retrieval_config,
    get_search_config,
    get_shared_config,
    get_storage_config,
    get_supported_languages,
    get_system_config,
    get_vectordb_config,
    is_language_supported,
    load_config,
    merge_configs,
    reload_config,
    validate_language_configuration,
)


class TestConfigError(unittest.TestCase):
    """Test ConfigError exception class."""

    def test_config_error_creation(self):
        """Test ConfigError can be created with message."""
        error = ConfigError("Test error message")
        self.assertEqual(str(error), "Test error message")

    def test_config_error_inheritance(self):
        """Test ConfigError inherits from Exception."""
        error = ConfigError("Test")
        self.assertIsInstance(error, Exception)


class TestConfigLoader(unittest.TestCase):
    """Test ConfigLoader class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)
        self.loader = ConfigLoader(self.config_dir)

        # Create mock config files
        self.mock_config = {
            "shared": {"version": "1.0.0"},
            "languages": {"supported": ["hr", "en"]},
            "system": {"debug": True}
        }
        self.mock_hr_config = {
            "shared": {"language": "hr"},
            "prompts": {"greeting": "Pozdrav"},
            "ranking": {
                "language_features": {
                    "special_characters": {"enabled": True, "characters": ["č", "ć", "š", "ž", "đ"]}
                }
            }
        }
        self.mock_en_config = {
            "shared": {"language": "en"},
            "prompts": {"greeting": "Hello"},
            "ranking": {"language_features": {"special_characters": {"enabled": False}}}
        }

        # Write mock config files
        config_toml = self.config_dir / "config.toml"
        hr_toml = self.config_dir / "hr.toml"
        en_toml = self.config_dir / "en.toml"

        with open(config_toml, "w") as f:
            toml.dump(self.mock_config, f)
        with open(hr_toml, "w") as f:
            toml.dump(self.mock_hr_config, f)
        with open(en_toml, "w") as f:
            toml.dump(self.mock_en_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_init_with_valid_directory(self):
        """Test ConfigLoader initialization with valid directory."""
        loader = ConfigLoader(self.config_dir)
        self.assertEqual(loader.config_dir, self.config_dir)
        self.assertEqual(loader._cache, {})

    def test_init_with_invalid_directory(self):
        """Test ConfigLoader initialization with invalid directory."""
        invalid_dir = Path("/nonexistent/directory")
        with self.assertRaises(ConfigError) as cm:
            ConfigLoader(invalid_dir)
        self.assertIn("Configuration directory not found", str(cm.exception))

    def test_load_main_config(self):
        """Test loading main config file."""
        config = self.loader.load("config")
        self.assertEqual(config["shared"]["version"], "1.0.0")
        self.assertEqual(config["languages"]["supported"], ["hr", "en"])

    def test_load_language_config(self):
        """Test loading language-specific config."""
        hr_config = self.loader.load("hr")
        self.assertEqual(hr_config["shared"]["language"], "hr")
        self.assertEqual(hr_config["prompts"]["greeting"], "Pozdrav")

    def test_load_with_caching(self):
        """Test configuration caching functionality."""
        # First load
        config1 = self.loader.load("config")
        self.assertIn("config", self.loader._cache)

        # Second load should use cache
        config2 = self.loader.load("config")
        self.assertIs(config1, config2)

    def test_load_without_caching(self):
        """Test loading without using cache."""
        config1 = self.loader.load("config")
        config2 = self.loader.load("config", use_cache=False)
        # Should be equal but not the same object
        self.assertEqual(config1, config2)
        self.assertIsNot(config1, config2)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent config file."""
        # Remove config.toml to make this test work
        (self.config_dir / "config.toml").unlink()

        with self.assertRaises(ConfigError) as cm:
            self.loader.load("nonexistent")
        self.assertIn("Configuration file not found", str(cm.exception))

    def test_load_invalid_toml(self):
        """Test loading invalid TOML file."""
        # Replace config.toml with invalid content
        config_toml = self.config_dir / "config.toml"
        config_toml.write_text("invalid toml content [")

        with self.assertRaises(ConfigError) as cm:
            self.loader.load("invalid")  # This will load config.toml due to hardcoded logic
        self.assertIn("Invalid TOML", str(cm.exception))

    def test_get_section_valid(self):
        """Test getting valid section from config."""
        shared = self.loader.get_section("config", "shared")
        self.assertEqual(shared["version"], "1.0.0")

    def test_get_section_invalid(self):
        """Test getting invalid section from config."""
        with self.assertRaises(ConfigError) as cm:
            self.loader.get_section("config", "nonexistent")
        self.assertIn("Section 'nonexistent' not found", str(cm.exception))

    def test_merge_configs(self):
        """Test merging multiple configurations."""
        merged = self.loader.merge_configs("config", "hr")

        # Should contain data from both configs
        # hr config will override shared section, so we check the hr greeting
        self.assertEqual(merged["languages"]["supported"], ["hr", "en"])  # From config
        self.assertEqual(merged["prompts"]["greeting"], "Pozdrav")  # From hr

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        self.loader.load("config")
        self.assertIn("config", self.loader._cache)

        self.loader.clear_cache()
        self.assertEqual(self.loader._cache, {})


class TestGlobalFunctions(unittest.TestCase):
    """Test global configuration loading functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)

        # Mock the global config loader
        self.patcher = patch('src.utils.config_loader._config_loader')
        self.mock_loader = self.patcher.start()

        self.mock_config = {
            "shared": {"version": "1.0.0"},
            "languages": {"supported": ["hr", "en"]},
            "main": {"project": {"name": "RAG Service"}},
            "generation": {"ollama": {"model": "qwen2.5:7b"}},
            "system": {"debug": True}
        }

        self.mock_loader.load.return_value = self.mock_config
        self.mock_loader.get_section.return_value = {"name": "RAG Service"}

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_load_config(self):
        """Test global load_config function."""
        config = load_config("config")
        self.mock_loader.load.assert_called_once_with("config", use_cache=True)
        self.assertEqual(config, self.mock_config)

    def test_get_config_section(self):
        """Test global get_config_section function."""
        section = get_config_section("config", "shared")
        self.mock_loader.get_section.assert_called_once_with("config", "shared")

    def test_get_shared_config(self):
        """Test get_shared_config function."""
        self.mock_loader.load.return_value = {"shared": {"version": "1.0.0"}}

        shared = get_shared_config()
        self.mock_loader.load.assert_called_once_with("config", use_cache=True)
        self.assertEqual(shared, {"version": "1.0.0"})


class TestLanguageFunctions(unittest.TestCase):
    """Test language-specific configuration functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.patcher = patch('src.utils.config_loader._config_loader')
        self.mock_loader = self.patcher.start()

        self.mock_main_config = {
            "languages": {"supported": ["hr", "en"]}
        }
        self.mock_hr_config = {
            "shared": {"language": "hr"},
            "prompts": {"greeting": "Pozdrav"},
            "ranking": {
                "language_features": {
                    "special_characters": {"enabled": True, "characters": ["č", "ć", "š", "ž", "đ"]}
                }
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher.stop()

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        self.mock_loader.load.return_value = self.mock_main_config

        languages = get_supported_languages()
        self.assertEqual(languages, ["hr", "en"])
        self.mock_loader.load.assert_called_with("config", use_cache=True)

    def test_get_supported_languages_error(self):
        """Test getting supported languages with error."""
        self.mock_loader.load.side_effect = Exception("Config error")

        with self.assertRaises(ConfigError) as cm:
            get_supported_languages()
        self.assertIn("Failed to load supported languages", str(cm.exception))

    def test_is_language_supported_valid(self):
        """Test checking if valid language is supported."""
        self.mock_loader.load.return_value = self.mock_main_config

        self.assertTrue(is_language_supported("hr"))
        self.assertTrue(is_language_supported("HR"))  # Case insensitive
        self.assertTrue(is_language_supported("en"))

    def test_is_language_supported_invalid(self):
        """Test checking if invalid language is supported."""
        self.mock_loader.load.return_value = self.mock_main_config

        self.assertFalse(is_language_supported("fr"))
        self.assertFalse(is_language_supported("invalid"))

    def test_get_language_config_file(self):
        """Test getting language config file name."""
        self.assertEqual(get_language_config_file("hr"), "hr.toml")
        self.assertEqual(get_language_config_file("en"), "en.toml")

    def test_get_language_config_valid(self):
        """Test getting valid language configuration."""
        self.mock_loader.load.side_effect = [self.mock_main_config, self.mock_hr_config]

        config = get_language_config("hr")
        self.assertEqual(config, self.mock_hr_config)

    def test_get_language_config_invalid(self):
        """Test getting invalid language configuration."""
        self.mock_loader.load.return_value = self.mock_main_config

        with self.assertRaises(ConfigError) as cm:
            get_language_config("fr")
        self.assertIn("Unsupported language: fr", str(cm.exception))

    def test_get_language_shared(self):
        """Test getting language shared configuration."""
        self.mock_loader.load.side_effect = [self.mock_main_config, self.mock_hr_config]

        shared = get_language_shared("hr")
        self.assertEqual(shared, {"language": "hr"})

    def test_get_language_specific_config_valid(self):
        """Test getting valid language-specific section."""
        self.mock_loader.load.side_effect = [self.mock_main_config, self.mock_hr_config]

        prompts = get_language_specific_config("prompts", "hr")
        self.assertEqual(prompts, {"greeting": "Pozdrav"})

    def test_get_language_specific_config_invalid_section(self):
        """Test getting invalid language-specific section."""
        self.mock_loader.load.side_effect = [self.mock_main_config, self.mock_hr_config]

        with self.assertRaises(ConfigError) as cm:
            get_language_specific_config("nonexistent", "hr")
        self.assertIn("Section 'nonexistent' not found", str(cm.exception))

    def test_get_language_ranking_features_valid(self):
        """Test getting valid language ranking features."""
        # Reset mock to control the call sequence better
        self.mock_loader.reset_mock()

        # First call to get_supported_languages, second to get_language_config, third for actual config load
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self.mock_main_config  # For get_supported_languages
            elif call_count == 2:
                return self.mock_main_config  # For is_language_supported check
            else:
                return self.mock_hr_config    # For actual config load

        self.mock_loader.load.side_effect = side_effect

        features = get_language_ranking_features("hr")
        expected = {
            "special_characters": {"enabled": True, "characters": ["č", "ć", "š", "ž", "đ"]}
        }
        self.assertEqual(features, expected)

    def test_get_language_ranking_features_unsupported_language(self):
        """Test getting ranking features for unsupported language."""
        self.mock_loader.load.return_value = self.mock_main_config

        with self.assertRaises(ConfigError) as cm:
            get_language_ranking_features("fr")
        self.assertIn("Language 'fr' not supported", str(cm.exception))

    def test_get_language_ranking_features_missing_section(self):
        """Test getting ranking features with missing section."""
        # Reset mock to control the call sequence better
        self.mock_loader.reset_mock()

        config_without_features = {
            "shared": {"language": "hr"},
            "ranking": {}  # Missing language_features
        }

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self.mock_main_config  # For get_supported_languages
            elif call_count == 2:
                return self.mock_main_config  # For is_language_supported check
            else:
                return config_without_features  # For actual config load

        self.mock_loader.load.side_effect = side_effect

        with self.assertRaises(ConfigError) as cm:
            get_language_ranking_features("hr")
        self.assertIn("Missing 'language_features' section", str(cm.exception))


class TestDiscoveryFunctions(unittest.TestCase):
    """Test configuration discovery functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)

        # Create test config files
        (self.config_dir / "config.toml").touch()
        (self.config_dir / "hr.toml").touch()
        (self.config_dir / "en.toml").touch()
        (self.config_dir / "fr.toml").touch()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch('src.utils.config_loader.CONFIG_DIR')
    def test_discover_available_languages(self, mock_config_dir):
        """Test discovering available language configurations."""
        mock_config_dir.glob.return_value = [
            Path("hr.toml"),
            Path("en.toml"),
            Path("config.toml"),
            Path("fr.toml")
        ]

        languages = discover_available_languages()
        expected = ["hr", "en", "fr"]  # config.toml excluded
        self.assertEqual(sorted(languages), sorted(expected))

    @patch('src.utils.config_loader.get_supported_languages')
    @patch('src.utils.config_loader.discover_available_languages')
    def test_validate_language_configuration_valid(self, mock_discover, mock_supported):
        """Test validating valid language configuration."""
        mock_supported.return_value = ["hr", "en"]
        mock_discover.return_value = ["hr", "en", "fr"]

        mapping = validate_language_configuration()
        expected = {"hr": "hr.toml", "en": "en.toml"}
        self.assertEqual(mapping, expected)

    @patch('src.utils.config_loader.get_supported_languages')
    @patch('src.utils.config_loader.discover_available_languages')
    def test_validate_language_configuration_missing(self, mock_discover, mock_supported):
        """Test validating language configuration with missing files."""
        mock_supported.return_value = ["hr", "en", "fr"]
        mock_discover.return_value = ["hr", "en"]  # fr missing

        with self.assertRaises(ConfigError) as cm:
            validate_language_configuration()
        self.assertIn("Missing config files for languages", str(cm.exception))
        self.assertIn("fr -> fr.toml", str(cm.exception))


class TestSpecificConfigFunctions(unittest.TestCase):
    """Test specific configuration getter functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.patcher = patch('src.utils.config_loader._config_loader')
        self.mock_loader = self.patcher.start()

        self.mock_configs = {
            "generation": {
                "ollama": {"model": "qwen2.5:7b"},
                "prompts": {"system": "You are helpful"},
                "response_parsing": {"format": "json"}
            },
            "preprocessing": {
                "extraction": {"pdf_enabled": True},
                "chunking": {"size": 512},
                "cleaning": {"enabled": True}
            },
            "vectordb": {
                "embeddings": {"model": "sentence-transformers"},
                "storage": {"persist_directory": "./chroma"},
                "search": {"k": 5}
            },
            "retrieval": {
                "query_processing": {"enabled": True},
                "ranking": {"algorithm": "bm25"},
                "reranking": {"enabled": False},
                "hybrid_retrieval": {"alpha": 0.5}
            },
            "pipeline": {
                "processing": {"batch_size": 32},
                "chroma": {"collection_name": "documents"},
                "performance": {"max_workers": 4}
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher.stop()

    def test_get_generation_config(self):
        """Test getting generation configuration."""
        self.mock_loader.load.return_value = self.mock_configs["generation"]

        config = get_generation_config()
        self.mock_loader.load.assert_called_with("generation", use_cache=True)
        self.assertEqual(config, self.mock_configs["generation"])

    def test_get_ollama_config(self):
        """Test getting Ollama configuration."""
        self.mock_loader.get_section.return_value = self.mock_configs["generation"]["ollama"]

        config = get_ollama_config()
        self.mock_loader.get_section.assert_called_with("generation", "ollama")
        self.assertEqual(config, {"model": "qwen2.5:7b"})

    def test_get_response_parsing_config(self):
        """Test getting response parsing configuration."""
        self.mock_loader.load.return_value = self.mock_configs["generation"]

        config = get_response_parsing_config()
        self.assertEqual(config, {"format": "json"})

    def test_get_preprocessing_config(self):
        """Test getting preprocessing configuration."""
        self.mock_loader.load.return_value = self.mock_configs["preprocessing"]

        config = get_preprocessing_config()
        self.assertEqual(config, self.mock_configs["preprocessing"])

    def test_get_extraction_config(self):
        """Test getting extraction configuration."""
        self.mock_loader.load.return_value = self.mock_configs["preprocessing"]

        config = get_extraction_config()
        self.assertEqual(config, {"pdf_enabled": True})

    def test_get_chunking_config(self):
        """Test getting chunking configuration."""
        self.mock_loader.load.return_value = self.mock_configs["preprocessing"]

        config = get_chunking_config()
        self.assertEqual(config, {"size": 512})

    def test_get_cleaning_config(self):
        """Test getting cleaning configuration."""
        self.mock_loader.load.return_value = self.mock_configs["preprocessing"]

        config = get_cleaning_config()
        self.assertEqual(config, {"enabled": True})

    def test_get_generation_prompts_config(self):
        """Test getting generation prompts configuration."""
        self.mock_loader.get_section.return_value = self.mock_configs["generation"]["prompts"]

        config = get_generation_prompts_config()
        self.mock_loader.get_section.assert_called_with("generation", "prompts")
        self.assertEqual(config, {"system": "You are helpful"})

    def test_get_vectordb_config(self):
        """Test getting vectordb configuration."""
        self.mock_loader.load.return_value = self.mock_configs["vectordb"]

        config = get_vectordb_config()
        self.assertEqual(config, self.mock_configs["vectordb"])

    def test_get_embeddings_config(self):
        """Test getting embeddings configuration."""
        self.mock_loader.load.return_value = self.mock_configs["vectordb"]

        config = get_embeddings_config()
        self.assertEqual(config, {"model": "sentence-transformers"})

    def test_get_storage_config(self):
        """Test getting storage configuration."""
        self.mock_loader.load.return_value = self.mock_configs["vectordb"]

        config = get_storage_config()
        self.assertEqual(config, {"persist_directory": "./chroma"})

    def test_get_search_config(self):
        """Test getting search configuration."""
        self.mock_loader.load.return_value = self.mock_configs["vectordb"]

        config = get_search_config()
        self.assertEqual(config, {"k": 5})

    def test_get_retrieval_config(self):
        """Test getting retrieval configuration."""
        self.mock_loader.load.return_value = self.mock_configs["retrieval"]

        config = get_retrieval_config()
        self.assertEqual(config, self.mock_configs["retrieval"])

    def test_get_query_processing_config(self):
        """Test getting query processing configuration."""
        self.mock_loader.load.return_value = self.mock_configs["retrieval"]

        config = get_query_processing_config()
        self.assertEqual(config, {"enabled": True})

    def test_get_ranking_config(self):
        """Test getting ranking configuration."""
        self.mock_loader.load.return_value = self.mock_configs["retrieval"]

        config = get_ranking_config()
        self.assertEqual(config, {"algorithm": "bm25"})

    def test_get_reranking_config(self):
        """Test getting reranking configuration."""
        self.mock_loader.load.return_value = self.mock_configs["retrieval"]

        config = get_reranking_config()
        self.assertEqual(config, {"enabled": False})

    def test_get_hybrid_retrieval_config(self):
        """Test getting hybrid retrieval configuration."""
        self.mock_loader.load.return_value = self.mock_configs["retrieval"]

        config = get_hybrid_retrieval_config()
        self.assertEqual(config, {"alpha": 0.5})

    def test_get_pipeline_config(self):
        """Test getting pipeline configuration."""
        self.mock_loader.load.return_value = self.mock_configs["pipeline"]

        config = get_pipeline_config()
        self.assertEqual(config, self.mock_configs["pipeline"])

    def test_get_processing_config(self):
        """Test getting processing configuration."""
        self.mock_loader.load.return_value = self.mock_configs["pipeline"]

        config = get_processing_config()
        self.assertEqual(config, {"batch_size": 32})

    def test_get_chroma_config(self):
        """Test getting ChromaDB configuration."""
        self.mock_loader.load.return_value = self.mock_configs["pipeline"]

        config = get_chroma_config()
        self.assertEqual(config, {"collection_name": "documents"})

    def test_get_performance_config(self):
        """Test getting performance configuration."""
        self.mock_loader.load.return_value = self.mock_configs["pipeline"]

        config = get_performance_config()
        self.assertEqual(config, {"max_workers": 4})


class TestUtilityFunctions(unittest.TestCase):
    """Test utility configuration functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.patcher = patch('src.utils.config_loader._config_loader')
        self.mock_loader = self.patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher.stop()

    def test_merge_configs(self):
        """Test merging configurations."""
        merged_config = {"section1": "value1", "section2": "value2"}
        self.mock_loader.merge_configs.return_value = merged_config

        result = merge_configs("config1", "config2")
        self.mock_loader.merge_configs.assert_called_with("config1", "config2")
        self.assertEqual(result, merged_config)

    def test_reload_config(self):
        """Test reloading configuration."""
        reloaded_config = {"reloaded": True}
        self.mock_loader.load.return_value = reloaded_config

        result = reload_config("config")
        self.mock_loader.load.assert_called_with("config", use_cache=False)
        self.assertEqual(result, reloaded_config)

    def test_get_project_info(self):
        """Test getting project info."""
        self.mock_loader.get_section.return_value = {"name": "RAG Service", "version": "1.0.0"}

        info = get_project_info()
        self.mock_loader.get_section.assert_called_with("main", "project")
        self.assertEqual(info, {"name": "RAG Service", "version": "1.0.0"})

    def test_get_paths_config(self):
        """Test getting paths configuration."""
        self.mock_loader.get_section.return_value = {"data": "./data", "logs": "./logs"}

        paths = get_paths_config()
        self.mock_loader.get_section.assert_called_with("main", "paths")
        self.assertEqual(paths, {"data": "./data", "logs": "./logs"})

    def test_get_system_config(self):
        """Test getting system configuration."""
        main_config = {"system": {"debug": True, "log_level": "INFO"}}
        self.mock_loader.load.return_value = main_config

        system = get_system_config()
        self.mock_loader.load.assert_called_with("config", use_cache=True)
        self.assertEqual(system, {"debug": True, "log_level": "INFO"})


if __name__ == "__main__":
    unittest.main()