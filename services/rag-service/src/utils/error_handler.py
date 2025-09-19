"""
Error handling utilities for the RAG system.
Provides fail-fast error handling following system governance.
"""

import logging
import sys
import traceback
from typing import Any

from .logging_factory import get_system_logger, log_error_context


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name, if None uses this module's __name__

    Returns:
        Logger instance
    """
    if not name:
        # Use this module's __name__ as default
        name = __name__

    return logging.getLogger(name)


class RAGError(Exception):
    """Base exception for RAG system errors."""

    def __init__(
        self,
        message: str,
        component: str = "unknown",
        operation: str = "unknown",
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.component = component
        self.operation = operation
        self.context = context or {}

        logger = get_system_logger()
        logger.trace("error_handler", "RAGError.__init__", f"Creating RAG error: {message}")
        log_error_context(component, operation, self, self.context)


class ConfigurationError(RAGError):
    """Configuration validation or loading error."""

    pass


class ValidationError(RAGError):
    """Data validation error."""

    pass


class ProcessingError(RAGError):
    """Document or query processing error."""

    pass


class StorageError(RAGError):
    """Storage backend error."""

    pass


class ModelError(RAGError):
    """LLM or embedding model error."""

    pass


def handle_critical_error(
    error: Exception, component: str, operation: str, context: dict[str, Any] | None = None
) -> None:
    """Handle critical system errors with comprehensive logging and graceful shutdown."""
    logger = get_system_logger()

    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "component": component,
        "operation": operation,
        **(context or {}),
    }

    logger.error(
        component,
        operation,
        f"CRITICAL FAILURE: {str(error)}",
        error_type=type(error).__name__,
        stack_trace=traceback.format_exc(),
        metadata=error_context,
    )

    logger.trace("error_handler", "handle_critical_error", "System shutdown initiated")
    sys.exit(1)


def log_and_raise(
    error_class: type[RAGError], message: str, component: str, operation: str, context: dict[str, Any] | None = None
) -> None:
    """Log error context and raise RAG exception."""
    logger = get_system_logger()
    logger.trace("error_handler", "log_and_raise", f"Raising {error_class.__name__}: {message}")

    error = error_class(message, component, operation, context)
    logger.debug("error_handler", "log_and_raise", f"Error created with context: {context or {}}")
    raise error


def validate_required(value: Any, name: str, component: str, operation: str) -> Any:
    """Validate required value with fail-fast error handling."""
    logger = get_system_logger()
    logger.trace("error_handler", "validate_required", f"Validating required value: {name}")

    if value is None:
        logger.debug("error_handler", "validate_required", f"VALIDATION FAILED: {name} is None")
        log_and_raise(ValidationError, f"Required value '{name}' is None", component, operation, {"value_name": name})

    if isinstance(value, str) and not value.strip():
        logger.debug("error_handler", "validate_required", f"VALIDATION FAILED: {name} is empty string")
        log_and_raise(
            ValidationError,
            f"Required value '{name}' is empty",
            component,
            operation,
            {"value_name": name, "value": value},
        )

    logger.trace("error_handler", "validate_required", f"VALIDATION PASSED: {name}")
    return value
