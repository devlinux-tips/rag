"""
Comprehensive tests for folder manager provider implementations.
Tests production and mock providers, factory functions, and filesystem operations.
"""

import sys
import unittest
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Any

from src.utils.folder_manager_providers import (
    MockFileSystemProvider,
    MockConfigProvider,
    MockLoggerProvider,
    ProductionFileSystemProvider,
    ProductionConfigProvider,
    StandardLoggerProvider,
    create_mock_setup,
    create_production_setup,
    create_test_config,
    create_development_folder_manager,
    create_test_folder_manager,
)
from src.utils.folder_manager import FolderConfig, FolderStats


class TestMockFileSystemProvider(unittest.TestCase):
    """Test mock filesystem provider functionality."""

    def test_init_creates_empty_state(self):
        """Test initialization creates empty state."""
        provider = MockFileSystemProvider()

        self.assertEqual(provider.created_folders, [])
        self.assertEqual(provider.existing_folders, {})
        self.assertEqual(provider.folder_stats, {})
        self.assertEqual(provider.call_history, [])
        self.assertEqual(provider.should_fail, {})

    def test_set_folder_exists(self):
        """Test setting folder existence state."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/folder")

        provider.set_folder_exists(folder_path, True)

        self.assertTrue(provider.existing_folders[str(folder_path)])

        provider.set_folder_exists(folder_path, False)

        self.assertFalse(provider.existing_folders[str(folder_path)])

    def test_set_folder_stats(self):
        """Test setting folder statistics."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/folder")
        stats = FolderStats(count=5, size_bytes=1024)

        provider.set_folder_stats(folder_path, stats)

        self.assertEqual(provider.folder_stats[str(folder_path)], stats)

    def test_set_should_fail(self):
        """Test setting operation failure flags."""
        provider = MockFileSystemProvider()

        provider.set_should_fail("create_folder", True)

        self.assertTrue(provider.should_fail["create_folder"])

        provider.set_should_fail("create_folder", False)

        self.assertFalse(provider.should_fail["create_folder"])

    def test_create_folder_success(self):
        """Test successful folder creation."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/new_folder")

        result = provider.create_folder(folder_path)

        self.assertTrue(result)
        self.assertIn(str(folder_path), provider.created_folders)
        self.assertTrue(provider.existing_folders[str(folder_path)])
        self.assertEqual(len(provider.call_history), 1)
        self.assertEqual(provider.call_history[0]["operation"], "create_folder")
        self.assertEqual(provider.call_history[0]["path"], str(folder_path))

    def test_create_folder_already_exists(self):
        """Test creating folder that already exists."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/existing_folder")

        # Create folder first time
        result1 = provider.create_folder(folder_path)
        # Try to create again
        result2 = provider.create_folder(folder_path)

        self.assertTrue(result1)
        self.assertFalse(result2)  # Should return False for already existing
        # Should only appear once in created_folders
        self.assertEqual(provider.created_folders.count(str(folder_path)), 1)

    def test_create_folder_with_failure_flag(self):
        """Test folder creation with failure flag set."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/fail_folder")

        provider.set_should_fail("create_folder", True)
        result = provider.create_folder(folder_path)

        self.assertFalse(result)
        self.assertNotIn(str(folder_path), provider.created_folders)

    def test_folder_exists_configured_true(self):
        """Test folder_exists when configured to exist."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/existing")

        provider.set_folder_exists(folder_path, True)
        result = provider.folder_exists(folder_path)

        self.assertTrue(result)
        self.assertEqual(len(provider.call_history), 1)
        self.assertEqual(provider.call_history[0]["operation"], "folder_exists")

    def test_folder_exists_configured_false(self):
        """Test folder_exists when configured not to exist."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/nonexistent")

        provider.set_folder_exists(folder_path, False)
        result = provider.folder_exists(folder_path)

        self.assertFalse(result)

    def test_folder_exists_not_configured_raises_error(self):
        """Test folder_exists raises error when not configured."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/unconfigured")

        with self.assertRaises(ValueError) as context:
            provider.folder_exists(folder_path)

        self.assertIn("Mock folder existence not configured", str(context.exception))

    def test_remove_folder_success(self):
        """Test successful folder removal."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/to_remove")

        # Set up folder as existing
        provider.set_folder_exists(folder_path, True)
        provider.created_folders.append(str(folder_path))

        result = provider.remove_folder(folder_path)

        self.assertTrue(result)
        self.assertNotIn(str(folder_path), provider.existing_folders)
        self.assertNotIn(str(folder_path), provider.created_folders)

    def test_remove_folder_with_failure_flag(self):
        """Test folder removal with failure flag set."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/fail_remove")

        provider.set_should_fail("remove_folder", True)
        result = provider.remove_folder(folder_path)

        self.assertFalse(result)

    def test_get_folder_stats_configured(self):
        """Test getting folder stats when configured."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/stats_folder")
        expected_stats = FolderStats(count=10, size_bytes=2048)

        provider.set_folder_stats(folder_path, expected_stats)
        result = provider.get_folder_stats(folder_path)

        self.assertEqual(result, expected_stats)
        self.assertEqual(len(provider.call_history), 1)
        self.assertEqual(provider.call_history[0]["operation"], "get_folder_stats")

    def test_get_folder_stats_not_configured_raises_error(self):
        """Test get_folder_stats raises error when not configured."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/unconfigured_stats")

        with self.assertRaises(ValueError) as context:
            provider.get_folder_stats(folder_path)

        self.assertIn("Mock folder stats not configured", str(context.exception))

    def test_clear_history(self):
        """Test clearing operation history."""
        provider = MockFileSystemProvider()
        folder_path = Path("/test/folder")

        # Generate some history
        provider.set_folder_exists(folder_path, True)
        provider.folder_exists(folder_path)
        provider.create_folder(folder_path)

        # Clear history
        provider.clear_history()

        self.assertEqual(provider.call_history, [])

    def test_get_created_folders(self):
        """Test getting list of created folders."""
        provider = MockFileSystemProvider()
        folder1 = Path("/test/folder1")
        folder2 = Path("/test/folder2")

        provider.create_folder(folder1)
        provider.create_folder(folder2)

        created = provider.get_created_folders()

        self.assertEqual(len(created), 2)
        self.assertIn(str(folder1), created)
        self.assertIn(str(folder2), created)
        # Should be a copy, not the original list
        self.assertIsNot(created, provider.created_folders)


class TestMockConfigProvider(unittest.TestCase):
    """Test mock configuration provider functionality."""

    def test_init_with_no_config(self):
        """Test initialization with no configuration creates defaults."""
        provider = MockConfigProvider()

        self.assertIsInstance(provider.config, FolderConfig)
        self.assertEqual(provider.config.data_base_dir, "/mock/data")
        self.assertEqual(provider.config.models_base_dir, "/mock/models")
        self.assertEqual(provider.call_history, [])

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = FolderConfig(
            data_base_dir="/custom/data",
            models_base_dir="/custom/models",
            system_dir="/custom/system",
            tenant_root_template="{data_base_dir}/tenants/{tenant_slug}",
            user_documents_template="{data_base_dir}/users/{user_id}",
            tenant_shared_template="{data_base_dir}/shared",
            user_processed_template="{data_base_dir}/processed/{user_id}",
            tenant_processed_template="{data_base_dir}/processed/shared",
            chromadb_path_template="{data_base_dir}/chroma",
            models_path_template="{models_base_dir}/{language}",
            collection_name_template="{tenant_slug}_{language}",
        )

        provider = MockConfigProvider(custom_config)

        self.assertEqual(provider.config, custom_config)
        self.assertEqual(provider.config.data_base_dir, "/custom/data")

    def test_set_config(self):
        """Test setting new configuration."""
        provider = MockConfigProvider()

        new_config = FolderConfig(
            data_base_dir="/new/data",
            models_base_dir="/new/models",
            system_dir="/new/system",
            tenant_root_template="{data_base_dir}/tenants/{tenant_slug}",
            user_documents_template="{data_base_dir}/users/{user_id}",
            tenant_shared_template="{data_base_dir}/shared",
            user_processed_template="{data_base_dir}/processed/{user_id}",
            tenant_processed_template="{data_base_dir}/processed/shared",
            chromadb_path_template="{data_base_dir}/chroma",
            models_path_template="{models_base_dir}/{language}",
            collection_name_template="{tenant_slug}_{language}",
        )

        provider.set_config(new_config)

        self.assertEqual(provider.config, new_config)

    def test_get_folder_config_records_call(self):
        """Test get_folder_config records call history."""
        provider = MockConfigProvider()

        result = provider.get_folder_config()

        self.assertEqual(provider.call_history, ["get_folder_config"])
        self.assertIsInstance(result, FolderConfig)

    def test_create_default_config_structure(self):
        """Test default configuration has expected structure."""
        provider = MockConfigProvider()
        config = provider.config

        # Check required fields
        self.assertIsNotNone(config.data_base_dir)
        self.assertIsNotNone(config.models_base_dir)
        self.assertIsNotNone(config.system_dir)
        self.assertIsNotNone(config.tenant_root_template)
        self.assertIsNotNone(config.user_documents_template)
        self.assertIsNotNone(config.collection_name_template)

        # Check template placeholders
        self.assertIn("{tenant_slug}", config.tenant_root_template)
        self.assertIn("{user_id}", config.user_documents_template)
        self.assertIn("{language}", config.models_path_template)


class TestMockLoggerProvider(unittest.TestCase):
    """Test mock logger provider functionality."""

    def test_init_creates_empty_message_storage(self):
        """Test initialization creates empty message storage."""
        logger = MockLoggerProvider()

        self.assertEqual(logger.messages["info"], [])
        self.assertEqual(logger.messages["debug"], [])
        self.assertEqual(logger.messages["warning"], [])
        self.assertEqual(logger.messages["error"], [])

    def test_info_captures_message(self):
        """Test info logging captures message."""
        logger = MockLoggerProvider()

        logger.info("test info message")

        self.assertEqual(logger.messages["info"], ["test info message"])

    def test_debug_captures_message(self):
        """Test debug logging captures message."""
        logger = MockLoggerProvider()

        logger.debug("test debug message")

        self.assertEqual(logger.messages["debug"], ["test debug message"])

    def test_warning_captures_message(self):
        """Test warning logging captures message."""
        logger = MockLoggerProvider()

        logger.warning("test warning message")

        self.assertEqual(logger.messages["warning"], ["test warning message"])

    def test_error_captures_message(self):
        """Test error logging captures message."""
        logger = MockLoggerProvider()

        logger.error("test error message")

        self.assertEqual(logger.messages["error"], ["test error message"])

    def test_clear_messages_removes_all(self):
        """Test clear_messages removes all captured messages."""
        logger = MockLoggerProvider()

        logger.info("test info")
        logger.debug("test debug")
        logger.warning("test warning")
        logger.error("test error")

        logger.clear_messages()

        self.assertEqual(logger.messages["info"], [])
        self.assertEqual(logger.messages["debug"], [])
        self.assertEqual(logger.messages["warning"], [])
        self.assertEqual(logger.messages["error"], [])

    def test_get_messages_returns_all_when_no_level(self):
        """Test get_messages returns all messages when no level specified."""
        logger = MockLoggerProvider()

        logger.info("info msg")
        logger.debug("debug msg")

        result = logger.get_messages()

        expected = {
            "info": ["info msg"],
            "debug": ["debug msg"],
            "warning": [],
            "error": []
        }
        self.assertEqual(result, expected)

    def test_get_messages_returns_specific_level(self):
        """Test get_messages returns specific level messages."""
        logger = MockLoggerProvider()

        logger.info("info msg 1")
        logger.info("info msg 2")
        logger.debug("debug msg")

        result = logger.get_messages("info")

        self.assertEqual(result, ["info msg 1", "info msg 2"])

    def test_get_messages_raises_error_for_unknown_level(self):
        """Test get_messages raises error for unknown level."""
        logger = MockLoggerProvider()

        with self.assertRaises(ValueError) as context:
            logger.get_messages("unknown")

        self.assertIn("Unknown log level: unknown", str(context.exception))


class TestProductionFileSystemProvider(unittest.TestCase):
    """Test production filesystem provider functionality."""

    def test_init_creates_logger(self):
        """Test initialization creates logger."""
        provider = ProductionFileSystemProvider()

        self.assertIsNotNone(provider.logger)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_create_folder_success(self, mock_mkdir, mock_exists):
        """Test successful folder creation."""
        mock_exists.return_value = False
        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/new_folder")

        result = provider.create_folder(folder_path)

        self.assertTrue(result)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.exists')
    def test_create_folder_already_exists(self, mock_exists):
        """Test creating folder that already exists."""
        mock_exists.return_value = True
        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/existing_folder")

        result = provider.create_folder(folder_path)

        self.assertFalse(result)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_folder_exists_true(self, mock_is_dir, mock_exists):
        """Test folder_exists returns True for existing directory."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/existing")

        result = provider.folder_exists(folder_path)

        self.assertTrue(result)

    @patch('pathlib.Path.exists')
    def test_folder_exists_false(self, mock_exists):
        """Test folder_exists returns False for non-existing path."""
        mock_exists.return_value = False
        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/nonexistent")

        result = provider.folder_exists(folder_path)

        self.assertFalse(result)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_folder_exists_false_for_file(self, mock_is_dir, mock_exists):
        """Test folder_exists returns False for existing file (not directory)."""
        mock_exists.return_value = True
        mock_is_dir.return_value = False
        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/file.txt")

        result = provider.folder_exists(folder_path)

        self.assertFalse(result)

    @patch('pathlib.Path.exists')
    @patch('shutil.rmtree')
    def test_remove_folder_success(self, mock_rmtree, mock_exists):
        """Test successful folder removal."""
        mock_exists.return_value = True
        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/to_remove")

        result = provider.remove_folder(folder_path)

        self.assertTrue(result)
        mock_rmtree.assert_called_once_with(folder_path)

    @patch('pathlib.Path.exists')
    def test_remove_folder_not_exists(self, mock_exists):
        """Test removing non-existing folder."""
        mock_exists.return_value = False
        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/nonexistent")

        result = provider.remove_folder(folder_path)

        self.assertFalse(result)

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.rglob')
    def test_get_folder_stats_success(self, mock_rglob, mock_exists):
        """Test successful folder statistics gathering."""
        mock_exists.return_value = True

        # Mock file objects
        mock_file1 = MagicMock()
        mock_file1.is_file.return_value = True
        mock_file1.stat.return_value.st_size = 100

        mock_file2 = MagicMock()
        mock_file2.is_file.return_value = True
        mock_file2.stat.return_value.st_size = 200

        mock_dir = MagicMock()
        mock_dir.is_file.return_value = False

        mock_rglob.return_value = [mock_file1, mock_file2, mock_dir]

        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/stats_folder")

        result = provider.get_folder_stats(folder_path)

        self.assertEqual(result.count, 2)  # Only files counted
        self.assertEqual(result.size_bytes, 300)  # Sum of file sizes

    @patch('pathlib.Path.exists')
    def test_get_folder_stats_not_exists(self, mock_exists):
        """Test getting stats for non-existing folder."""
        mock_exists.return_value = False
        provider = ProductionFileSystemProvider()
        folder_path = Path("/test/nonexistent")

        result = provider.get_folder_stats(folder_path)

        self.assertEqual(result.count, 0)
        self.assertEqual(result.size_bytes, 0)


class TestConfigProvider(unittest.TestCase):
    """Test production configuration provider functionality."""

    def test_init_creates_cache(self):
        """Test initialization creates config cache."""
        provider = ProductionConfigProvider()

        self.assertIsNone(provider._config_cache)

    def test_get_folder_config_caches_result(self):
        """Test get_folder_config caches the result."""
        mock_get_paths_config = MagicMock(return_value={
            "data_base_dir": "/prod/data",
            "models_base_dir": "/prod/models",
            "system_dir": "/prod/system",
            "tenant_root_template": "{data_base_dir}/tenants/{tenant_slug}",
            "user_documents_template": "{data_base_dir}/users/{user_id}",
            "tenant_shared_template": "{data_base_dir}/shared",
            "user_processed_template": "{data_base_dir}/processed/{user_id}",
            "tenant_processed_template": "{data_base_dir}/processed/shared",
            "chromadb_path_template": "{data_base_dir}/chroma",
            "models_path_template": "{models_base_dir}/{language}",
            "collection_name_template": "{tenant_slug}_{language}",
        })

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_paths_config=mock_get_paths_config
            )
        }):
            provider = ProductionConfigProvider()

            # First call should load and cache
            result1 = provider.get_folder_config()
            # Second call should return cached result
            result2 = provider.get_folder_config()

            # Should be same object (cached)
            self.assertIs(result1, result2)
            # Mock should only be called once due to caching
            mock_get_paths_config.assert_called_once()

    def test_load_config_from_system_success(self):
        """Test successful loading of config from system."""
        mock_get_paths_config = MagicMock(return_value={
            "data_base_dir": "/prod/data",
            "models_base_dir": "/prod/models",
            "system_dir": "/prod/system",
            "tenant_root_template": "{data_base_dir}/tenants/{tenant_slug}",
            "user_documents_template": "{data_base_dir}/users/{user_id}",
            "tenant_shared_template": "{data_base_dir}/shared",
            "user_processed_template": "{data_base_dir}/processed/{user_id}",
            "tenant_processed_template": "{data_base_dir}/processed/shared",
            "chromadb_path_template": "{data_base_dir}/chroma",
            "models_path_template": "{models_base_dir}/{language}",
            "collection_name_template": "{tenant_slug}_{language}",
        })

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_paths_config=mock_get_paths_config
            )
        }):
            provider = ProductionConfigProvider()
            result = provider.get_folder_config()

            self.assertIsInstance(result, FolderConfig)
            self.assertEqual(result.data_base_dir, "/prod/data")
            self.assertEqual(result.models_base_dir, "/prod/models")
            self.assertEqual(result.system_dir, "/prod/system")

    def test_load_config_from_system_handles_exceptions(self):
        """Test that exceptions from config system are properly handled."""
        mock_get_paths_config = MagicMock(side_effect=RuntimeError("Config error"))

        with patch.dict('sys.modules', {
            'src.utils.config_loader': MagicMock(
                get_paths_config=mock_get_paths_config
            )
        }):
            provider = ProductionConfigProvider()

            with self.assertRaises(RuntimeError) as context:
                provider.get_folder_config()

            self.assertIn("Failed to load folder configuration from system", str(context.exception))


class TestStandardLoggerProvider(unittest.TestCase):
    """Test standard logger provider functionality."""

    def test_init_with_default_logger_name(self):
        """Test initialization with default logger name."""
        provider = StandardLoggerProvider()

        self.assertIsNotNone(provider.logger)
        # Logger name should include the module name
        self.assertIn("folder_manager_providers", provider.logger.name)

    def test_init_with_custom_logger_name(self):
        """Test initialization with custom logger name."""
        provider = StandardLoggerProvider("custom.logger.name")

        self.assertEqual(provider.logger.name, "custom.logger.name")

    def test_logging_methods_call_logger(self):
        """Test that logging methods properly call the underlying logger."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            provider = StandardLoggerProvider("test.logger")

            # Test all logging methods
            provider.info("info message")
            provider.debug("debug message")
            provider.warning("warning message")
            provider.error("error message")

            # Verify calls to underlying logger
            mock_logger.info.assert_called_once_with("info message")
            mock_logger.debug.assert_called_once_with("debug message")
            mock_logger.warning.assert_called_once_with("warning message")
            mock_logger.error.assert_called_once_with("error message")


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for provider creation."""

    def test_create_mock_setup_with_defaults(self):
        """Test create_mock_setup with default parameters."""
        config_provider, filesystem_provider, logger_provider = create_mock_setup()

        self.assertIsInstance(config_provider, MockConfigProvider)
        self.assertIsInstance(filesystem_provider, MockFileSystemProvider)
        self.assertIsInstance(logger_provider, MockLoggerProvider)

    def test_create_mock_setup_with_custom_config(self):
        """Test create_mock_setup with custom configuration."""
        custom_config = FolderConfig(
            data_base_dir="/custom/data",
            models_base_dir="/custom/models",
            system_dir="/custom/system",
            tenant_root_template="{data_base_dir}/tenants/{tenant_slug}",
            user_documents_template="{data_base_dir}/users/{user_id}",
            tenant_shared_template="{data_base_dir}/shared",
            user_processed_template="{data_base_dir}/processed/{user_id}",
            tenant_processed_template="{data_base_dir}/processed/shared",
            chromadb_path_template="{data_base_dir}/chroma",
            models_path_template="{models_base_dir}/{language}",
            collection_name_template="{tenant_slug}_{language}",
        )

        config_provider, filesystem_provider, logger_provider = create_mock_setup(
            config=custom_config
        )

        self.assertEqual(config_provider.config, custom_config)

    def test_create_mock_setup_with_existing_folders(self):
        """Test create_mock_setup with existing folder configuration."""
        existing_folders = {
            "/test/folder1": True,
            "/test/folder2": False
        }

        config_provider, filesystem_provider, logger_provider = create_mock_setup(
            existing_folders=existing_folders
        )

        self.assertTrue(filesystem_provider.existing_folders["/test/folder1"])
        self.assertFalse(filesystem_provider.existing_folders["/test/folder2"])

    def test_create_mock_setup_with_folder_stats(self):
        """Test create_mock_setup with folder statistics configuration."""
        folder_stats = {
            "/test/stats1": FolderStats(count=5, size_bytes=1024),
            "/test/stats2": FolderStats(count=10, size_bytes=2048)
        }

        config_provider, filesystem_provider, logger_provider = create_mock_setup(
            folder_stats=folder_stats
        )

        self.assertEqual(filesystem_provider.folder_stats["/test/stats1"].count, 5)
        self.assertEqual(filesystem_provider.folder_stats["/test/stats2"].size_bytes, 2048)

    def test_create_mock_setup_with_filesystem_failures(self):
        """Test create_mock_setup with filesystem failure configuration."""
        filesystem_failures = {
            "create_folder": True,
            "remove_folder": False
        }

        config_provider, filesystem_provider, logger_provider = create_mock_setup(
            filesystem_failures=filesystem_failures
        )

        self.assertTrue(filesystem_provider.should_fail["create_folder"])
        self.assertFalse(filesystem_provider.should_fail["remove_folder"])

    def test_create_production_setup_with_defaults(self):
        """Test create_production_setup with default logger name."""
        config_provider, filesystem_provider, logger_provider = create_production_setup()

        self.assertIsInstance(config_provider, ProductionConfigProvider)
        self.assertIsInstance(filesystem_provider, ProductionFileSystemProvider)
        self.assertIsInstance(logger_provider, StandardLoggerProvider)

    def test_create_production_setup_with_custom_logger_name(self):
        """Test create_production_setup with custom logger name."""
        config_provider, filesystem_provider, logger_provider = create_production_setup(
            logger_name="custom.logger"
        )

        self.assertEqual(logger_provider.logger.name, "custom.logger")

    def test_create_test_config_with_defaults(self):
        """Test create_test_config with default parameters."""
        config = create_test_config()

        self.assertEqual(config.data_base_dir, "/test/data")
        self.assertEqual(config.models_base_dir, "/test/models")
        self.assertEqual(config.system_dir, "/test/system")

    def test_create_test_config_with_custom_parameters(self):
        """Test create_test_config with custom parameters."""
        config = create_test_config(
            data_base_dir="/custom/data",
            models_base_dir="/custom/models",
            system_dir="/custom/system"
        )

        self.assertEqual(config.data_base_dir, "/custom/data")
        self.assertEqual(config.models_base_dir, "/custom/models")
        self.assertEqual(config.system_dir, "/custom/system")

    def test_create_test_config_template_structure(self):
        """Test create_test_config has proper template structure."""
        config = create_test_config()

        # Check all templates are present
        self.assertIsNotNone(config.tenant_root_template)
        self.assertIsNotNone(config.user_documents_template)
        self.assertIsNotNone(config.tenant_shared_template)
        self.assertIsNotNone(config.user_processed_template)
        self.assertIsNotNone(config.tenant_processed_template)
        self.assertIsNotNone(config.chromadb_path_template)
        self.assertIsNotNone(config.models_path_template)
        self.assertIsNotNone(config.collection_name_template)

        # Check template placeholders
        self.assertIn("{tenant_slug}", config.tenant_root_template)
        self.assertIn("{user_id}", config.user_documents_template)
        self.assertIn("{language}", config.models_path_template)


class TestIntegrationHelpers(unittest.TestCase):
    """Test integration helper functions."""

    def test_create_development_folder_manager(self):
        """Test create_development_folder_manager creates manager with production setup."""
        mock_manager = MagicMock()

        with patch('src.utils.folder_manager.create_tenant_folder_manager') as mock_create:
            mock_create.return_value = mock_manager

            result = create_development_folder_manager()

            self.assertEqual(result, mock_manager)
            mock_create.assert_called_once()

            # Check that it was called with production providers
            call_kwargs = mock_create.call_args[1]
            self.assertIsInstance(call_kwargs["config_provider"], ProductionConfigProvider)
            self.assertIsInstance(call_kwargs["filesystem_provider"], ProductionFileSystemProvider)
            self.assertIsInstance(call_kwargs["logger_provider"], StandardLoggerProvider)

    def test_create_test_folder_manager_with_defaults(self):
        """Test create_test_folder_manager with default parameters."""
        mock_manager = MagicMock()

        with patch('src.utils.folder_manager.create_tenant_folder_manager') as mock_create:
            mock_create.return_value = mock_manager

            result, providers = create_test_folder_manager()

            self.assertEqual(result, mock_manager)
            self.assertEqual(len(providers), 3)  # config, filesystem, logger providers

            config_provider, filesystem_provider, logger_provider = providers
            self.assertIsInstance(config_provider, MockConfigProvider)
            self.assertIsInstance(filesystem_provider, MockFileSystemProvider)
            self.assertIsInstance(logger_provider, MockLoggerProvider)

    def test_create_test_folder_manager_with_custom_config(self):
        """Test create_test_folder_manager with custom configuration."""
        custom_config = FolderConfig(
            data_base_dir="/custom/data",
            models_base_dir="/custom/models",
            system_dir="/custom/system",
            tenant_root_template="{data_base_dir}/tenants/{tenant_slug}",
            user_documents_template="{data_base_dir}/users/{user_id}",
            tenant_shared_template="{data_base_dir}/shared",
            user_processed_template="{data_base_dir}/processed/{user_id}",
            tenant_processed_template="{data_base_dir}/processed/shared",
            chromadb_path_template="{data_base_dir}/chroma",
            models_path_template="{models_base_dir}/{language}",
            collection_name_template="{tenant_slug}_{language}",
        )

        mock_manager = MagicMock()

        with patch('src.utils.folder_manager.create_tenant_folder_manager') as mock_create:
            mock_create.return_value = mock_manager

            result, providers = create_test_folder_manager(config=custom_config)

            config_provider, filesystem_provider, logger_provider = providers
            self.assertEqual(config_provider.config, custom_config)

    def test_create_test_folder_manager_with_existing_folders(self):
        """Test create_test_folder_manager with existing folder configuration."""
        existing_folders = {"/test/existing": True}

        mock_manager = MagicMock()

        with patch('src.utils.folder_manager.create_tenant_folder_manager') as mock_create:
            mock_create.return_value = mock_manager

            result, providers = create_test_folder_manager(existing_folders=existing_folders)

            config_provider, filesystem_provider, logger_provider = providers
            self.assertTrue(filesystem_provider.existing_folders["/test/existing"])

    def test_create_test_folder_manager_with_folder_stats(self):
        """Test create_test_folder_manager with folder statistics configuration."""
        folder_stats = {"/test/stats": FolderStats(count=5, size_bytes=1024)}

        mock_manager = MagicMock()

        with patch('src.utils.folder_manager.create_tenant_folder_manager') as mock_create:
            mock_create.return_value = mock_manager

            result, providers = create_test_folder_manager(folder_stats=folder_stats)

            config_provider, filesystem_provider, logger_provider = providers
            self.assertEqual(filesystem_provider.folder_stats["/test/stats"].count, 5)

    def test_create_test_folder_manager_with_filesystem_failures(self):
        """Test create_test_folder_manager with filesystem failure configuration."""
        filesystem_failures = {"create_folder": True}

        mock_manager = MagicMock()

        with patch('src.utils.folder_manager.create_tenant_folder_manager') as mock_create:
            mock_create.return_value = mock_manager

            result, providers = create_test_folder_manager(filesystem_failures=filesystem_failures)

            config_provider, filesystem_provider, logger_provider = providers
            self.assertTrue(filesystem_provider.should_fail["create_folder"])


if __name__ == "__main__":
    unittest.main()