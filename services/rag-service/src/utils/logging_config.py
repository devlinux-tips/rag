"""
Legacy logging configuration - DEPRECATED.
Use logging_factory.py for new implementations.

Provides structured logging with appropriate levels:
- INFO: Major workflow steps (vectorization, ranking, LLM responses)
- DEBUG: Detailed step-by-step processing information
- ERROR: Exception handling and error conditions
- WARNING: Potentially problematic conditions
"""

import logging
import sys
from pathlib import Path

from .logging_factory import get_system_logger


def setup_logging(level: str = "INFO", log_file: str | None = None, include_debug: bool = False) -> logging.Logger:
    """
    Setup centralized logging configuration for RAG system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        include_debug: Whether to include debug logging

    Returns:
        Configured logger instance
    """
    # Create root logger for RAG system
    logger = logging.getLogger("rag_system")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter with structured output
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set specific loggers for components
    component_loggers = [
        "rag_system.pipeline",
        "rag_system.extraction",
        "rag_system.cleaning",
        "rag_system.chunking",
        "rag_system.embedding",
        "rag_system.vectordb",
        "rag_system.retrieval",
        "rag_system.ranking",
        "rag_system.generation",
        "rag_system.parsing",
        "rag_system.config",
    ]

    for component_name in component_loggers:
        component_logger = logging.getLogger(component_name)
        component_logger.setLevel(getattr(logging, level.upper()))

    # Suppress third-party noise unless debug enabled
    if not include_debug:
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info(f"Logging initialized - Level: {level}, Debug: {include_debug}")
    return logger


def get_component_logger(component: str) -> logging.Logger:
    """DEPRECATED: Get logger for specific RAG component."""
    return logging.getLogger(f"rag_system.{component}")


def log_exception(logger: logging.Logger, operation: str, exception: Exception) -> None:
    """DEPRECATED: Standardized exception logging."""
    logger.error(f"FAILED: {operation} - {type(exception).__name__}: {exception}")


def log_performance(logger: logging.Logger, operation: str, duration: float, **metrics) -> None:
    """DEPRECATED: Log performance metrics for operations."""
    metric_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(f"PERFORMANCE: {operation} completed in {duration:.2f}s | {metric_str}")


def log_workflow_step(logger: logging.Logger, step: str, status: str = "STARTED", **details) -> None:
    """DEPRECATED: Log major workflow steps."""
    detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
    message = f"WORKFLOW: {step} {status}"
    if detail_str:
        message += f" | {detail_str}"
    logger.info(message)


# New factory-based logging helpers
def log_component_info(component: str, operation: str, message: str, **metadata) -> None:
    """Log INFO level message using logging factory."""
    logger = get_system_logger()
    logger.info(component, operation, message, metadata=metadata if metadata else None)


def log_component_debug(component: str, operation: str, message: str, **metadata) -> None:
    """Log DEBUG level message using logging factory."""
    logger = get_system_logger()
    logger.debug(component, operation, message, metadata=metadata if metadata else None)


def log_component_warning(component: str, operation: str, message: str, **metadata) -> None:
    """Log WARNING level message using logging factory."""
    logger = get_system_logger()
    logger.warning(component, operation, message, metadata=metadata if metadata else None)


def log_component_error(
    component: str, operation: str, message: str, exception: Exception | None = None, **metadata
) -> None:
    """Log ERROR level message using logging factory."""
    logger = get_system_logger()
    error_type = type(exception).__name__ if exception else None
    stack_trace = str(exception) if exception else None
    logger.error(
        component,
        operation,
        message,
        error_type=error_type,
        stack_trace=stack_trace,
        metadata=metadata if metadata else None,
    )


def log_performance_metric(component: str, operation: str, duration: float, **metrics) -> None:
    """Log performance metrics using logging factory."""
    logger = get_system_logger()
    message = f"completed in {duration:.2f}s"
    logger.info(component, operation, message, duration=duration, metadata=metrics if metrics else None)


def log_workflow_event(component: str, workflow: str, status: str = "STARTED", **details) -> None:
    """Log workflow events using logging factory."""
    logger = get_system_logger()
    message = f"{workflow} {status}"
    logger.info(component, f"WORKFLOW_{workflow.upper()}", message, metadata=details if details else None)
