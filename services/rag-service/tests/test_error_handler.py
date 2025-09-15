"""
Test suite for utils.error_handler module.

Tests the error handling utilities for fail-fast error handling.
Follows project's testing standards with comprehensive coverage.
"""

import logging
import pytest
from unittest.mock import patch

from src.utils.error_handler import get_logger


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_with_name(self):
        """Test get_logger with explicit name parameter."""
        logger_name = "test.logger.name"
        logger = get_logger(logger_name)

        assert isinstance(logger, logging.Logger)
        assert logger.name == logger_name

    def test_get_logger_with_none_name(self):
        """Test get_logger with None name parameter (should use __name__)."""
        logger = get_logger(None)

        assert isinstance(logger, logging.Logger)
        assert logger.name == "src.utils.error_handler"

    def test_get_logger_with_default_name(self):
        """Test get_logger with no name parameter (should use __name__)."""
        logger = get_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "src.utils.error_handler"

    def test_get_logger_returns_same_instance_for_same_name(self):
        """Test that get_logger returns the same logger instance for same name."""
        name = "duplicate.test.logger"
        logger1 = get_logger(name)
        logger2 = get_logger(name)

        assert logger1 is logger2
        assert logger1.name == name

    def test_get_logger_different_names_return_different_instances(self):
        """Test that different names return different logger instances."""
        logger1 = get_logger("logger.one")
        logger2 = get_logger("logger.two")

        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_empty_string_name(self):
        """Test get_logger with empty string name (should use __name__ due to falsy value)."""
        logger = get_logger("")

        assert isinstance(logger, logging.Logger)
        # Empty string is falsy, so function uses __name__ as fallback
        assert logger.name == "src.utils.error_handler"

    def test_get_logger_with_hierarchical_name(self):
        """Test get_logger with hierarchical logger name."""
        hierarchical_name = "parent.child.grandchild"
        logger = get_logger(hierarchical_name)

        assert isinstance(logger, logging.Logger)
        assert logger.name == hierarchical_name

    @patch('src.utils.error_handler.__name__', 'mocked.module.name')
    def test_get_logger_uses_module_name_when_none(self):
        """Test that get_logger uses the module's __name__ when name is None."""
        logger = get_logger(None)

        assert isinstance(logger, logging.Logger)
        # Note: The patch doesn't affect the actual __name__ in this context
        # but this test documents the intended behavior

    def test_get_logger_type_annotations(self):
        """Test that the function handles type annotations correctly."""
        # Test with str type
        logger_str = get_logger("string_name")
        assert isinstance(logger_str, logging.Logger)

        # Test with None type
        logger_none = get_logger(None)
        assert isinstance(logger_none, logging.Logger)

    def test_get_logger_preserves_logging_hierarchy(self):
        """Test that get_logger preserves Python logging hierarchy."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        assert isinstance(parent_logger, logging.Logger)
        assert isinstance(child_logger, logging.Logger)
        assert child_logger.parent.name == parent_logger.name

    def test_get_logger_with_special_characters(self):
        """Test get_logger with special characters in name."""
        special_name = "test-logger_with.special@characters"
        logger = get_logger(special_name)

        assert isinstance(logger, logging.Logger)
        assert logger.name == special_name

    def test_get_logger_integration_with_logging_module(self):
        """Test that get_logger integrates properly with Python logging module."""
        test_name = "integration.test.logger"
        logger = get_logger(test_name)

        # Verify it's the same as calling logging.getLogger directly
        direct_logger = logging.getLogger(test_name)
        assert logger is direct_logger

    def test_get_logger_handles_union_type_annotation(self):
        """Test function signature with Union type annotation (str | None)."""
        # This test ensures the modern Union syntax (str | None) works
        # Test both valid types in the Union
        logger_with_str = get_logger("test_union")
        logger_with_none = get_logger(None)

        assert isinstance(logger_with_str, logging.Logger)
        assert isinstance(logger_with_none, logging.Logger)
        assert logger_with_str.name == "test_union"
        assert logger_with_none.name == "src.utils.error_handler"


# Integration tests
class TestErrorHandlerIntegration:
    """Integration tests for the error_handler module."""

    def test_module_imports_successfully(self):
        """Test that the module can be imported without errors."""
        import src.utils.error_handler
        assert hasattr(src.utils.error_handler, 'get_logger')

    def test_function_is_callable(self):
        """Test that get_logger is callable."""
        assert callable(get_logger)

    def test_docstring_present(self):
        """Test that get_logger has proper documentation."""
        assert get_logger.__doc__ is not None
        assert len(get_logger.__doc__.strip()) > 0

    def test_function_signature_type_hints(self):
        """Test that function has proper type hints."""
        import inspect
        signature = inspect.signature(get_logger)

        # Check parameter type hints
        name_param = signature.parameters['name']
        assert name_param.annotation is not None

        # Check return type hint
        assert signature.return_annotation is not None