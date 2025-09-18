"""
Logging factory for swappable logging backends.

Supports multiple logging destinations:
- Console/stdout (for development and AI terminal usage)
- Elasticsearch (for production monitoring and analysis)
- File (for persistent local logging)
- Structured JSON (for external log aggregation)
"""

import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_loader import get_logging_config

# Add TRACE level (5) - below DEBUG (10) for finest-grained logging
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def trace_method(self, message, *args, **kwargs):
    """Add trace method to Logger class."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


# Add trace method to Logger class - ignore mypy type checking for this extension
logging.Logger.trace = trace_method  # type: ignore[attr-defined]


@dataclass
class LogEntry:
    """Structured log entry for consistent logging across backends."""

    timestamp: str
    level: str
    component: str
    operation: str
    message: str
    duration: float | None = None
    metadata: dict[str, Any] | None = None
    error_type: str | None = None
    stack_trace: str | None = None


class LoggingBackend(ABC):
    """Abstract base class for logging backends."""

    @abstractmethod
    def log(self, entry: LogEntry) -> None:
        """Log a structured entry."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close backend resources."""
        pass


class ConsoleLoggingBackend(LoggingBackend):
    """Standard console/stdout logging backend for development."""

    def __init__(self, level: str = "INFO", colored: bool = True):
        # Handle custom TRACE level
        if level.upper() == "TRACE":
            self.level = TRACE_LEVEL
        else:
            self.level = getattr(logging, level.upper())
        self.colored = colored

        # Setup Python logging for console
        self.logger = logging.getLogger("rag_console")
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log(self, entry: LogEntry) -> None:
        # Handle custom TRACE level
        if entry.level.upper() == "TRACE":
            level_num = TRACE_LEVEL
        else:
            level_num = getattr(logging, entry.level.upper())
        if level_num >= self.level:
            # Format message with metadata
            message = entry.message
            if entry.duration:
                message += f" | duration={entry.duration:.2f}s"
            if entry.metadata:
                metadata_str = " | ".join(f"{k}={v}" for k, v in entry.metadata.items())
                message += f" | {metadata_str}"

            # Use the main console logger that has the handler configured
            formatted_message = f"{entry.component}.{entry.operation}: {message}"
            self.logger.log(level_num, formatted_message)

    def close(self) -> None:
        for handler in self.logger.handlers:
            handler.close()


class ElasticsearchLoggingBackend(LoggingBackend):
    """Elasticsearch logging backend for production monitoring."""

    def __init__(self, es_config: dict[str, Any]):
        self.es_config = es_config
        self.index_prefix = es_config.get("index_prefix", "rag-logs")
        self.es_client = None
        self._init_elasticsearch()

    def _init_elasticsearch(self):
        """Initialize Elasticsearch client."""
        try:
            from elasticsearch import Elasticsearch

            self.es_client = Elasticsearch(
                hosts=self.es_config.get("hosts", ["localhost:9200"]),
                http_auth=self.es_config.get("auth"),
                verify_certs=self.es_config.get("verify_certs", True),
                request_timeout=self.es_config.get("timeout", 30),
            )

            # Test connection
            if not self.es_client.ping():
                raise ConnectionError("Cannot connect to Elasticsearch")

        except ImportError as e:
            raise ImportError("elasticsearch package required for ElasticsearchLoggingBackend") from e
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Elasticsearch: {e}") from e

    def log(self, entry: LogEntry) -> None:
        if not self.es_client:
            return

        try:
            # Create index name with date
            date_str = datetime.now().strftime("%Y.%m.%d")
            index_name = f"{self.index_prefix}-{date_str}"

            # Convert entry to document
            doc = asdict(entry)
            doc["@timestamp"] = entry.timestamp

            # Index document
            self.es_client.index(index=index_name, document=doc)

        except Exception as e:
            # Fallback to console logging if ES fails
            print(f"Elasticsearch logging failed: {e}")
            console_backend = ConsoleLoggingBackend()
            console_backend.log(entry)

    def close(self) -> None:
        if self.es_client:
            self.es_client.close()


class FileLoggingBackend(LoggingBackend):
    """File-based logging backend for persistent local logs."""

    def __init__(self, log_file: str, format_type: str = "text"):
        self.log_file = Path(log_file)
        self.format_type = format_type  # "text" or "json"

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: LogEntry) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                if self.format_type == "json":
                    json.dump(asdict(entry), f, ensure_ascii=False)
                    f.write("\n")
                else:
                    # Text format
                    message = (
                        f"{entry.timestamp} | {entry.level:8} | {entry.component} | {entry.operation}: {entry.message}"
                    )
                    if entry.duration:
                        message += f" | duration={entry.duration:.2f}s"
                    if entry.metadata:
                        metadata_str = " | ".join(f"{k}={v}" for k, v in entry.metadata.items())
                        message += f" | {metadata_str}"
                    f.write(message + "\n")
        except Exception as e:
            print(f"File logging failed: {e}")

    def close(self) -> None:
        pass  # File handles are closed automatically


class LoggingFactory:
    """Factory for creating and managing logging backends."""

    def __init__(self):
        self.backends: list[LoggingBackend] = []
        self.config = get_logging_config()

    def create_backend(self, backend_type: str, **kwargs) -> LoggingBackend:
        """Create a logging backend of specified type."""
        if backend_type == "console":
            return ConsoleLoggingBackend(**kwargs)
        elif backend_type == "elasticsearch":
            es_config = self.config.get("elasticsearch", {})
            es_config.update(kwargs)
            return ElasticsearchLoggingBackend(es_config)
        elif backend_type == "file":
            return FileLoggingBackend(**kwargs)
        else:
            raise ValueError(f"Unknown logging backend type: {backend_type}")

    def setup_logging(self, backend_types: list[str], **backend_kwargs) -> "MultiBackendLogger":
        """Setup logging with multiple backends."""
        self.backends.clear()

        for backend_type in backend_types:
            try:
                backend = self.create_backend(backend_type, **backend_kwargs.get(backend_type, {}))
                self.backends.append(backend)
            except Exception as e:
                print(f"Failed to setup {backend_type} backend: {e}")
                # Fallback to console if primary backend fails
                if backend_type != "console":
                    console_backend = self.create_backend("console")
                    self.backends.append(console_backend)

        return MultiBackendLogger(self.backends)

    def close_all(self) -> None:
        """Close all logging backends."""
        for backend in self.backends:
            backend.close()


class MultiBackendLogger:
    """Logger that sends to multiple backends simultaneously."""

    def __init__(self, backends: list[LoggingBackend]):
        self.backends = backends

    def log(
        self,
        level: str,
        component: str,
        operation: str,
        message: str,
        duration: float | None = None,
        metadata: dict[str, Any] | None = None,
        error_type: str | None = None,
        stack_trace: str | None = None,
    ) -> None:
        """Log to all configured backends."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.upper(),
            component=component,
            operation=operation,
            message=message,
            duration=duration,
            metadata=metadata,
            error_type=error_type,
            stack_trace=stack_trace,
        )

        for backend in self.backends:
            try:
                backend.log(entry)
            except Exception as e:
                print(f"Backend logging failed: {e}")

    def info(self, component: str, operation: str, message: str, **kwargs) -> None:
        self.log("INFO", component, operation, message, **kwargs)

    def debug(self, component: str, operation: str, message: str, **kwargs) -> None:
        self.log("DEBUG", component, operation, message, **kwargs)

    def warning(self, component: str, operation: str, message: str, **kwargs) -> None:
        self.log("WARNING", component, operation, message, **kwargs)

    def error(self, component: str, operation: str, message: str, **kwargs) -> None:
        self.log("ERROR", component, operation, message, **kwargs)

    def trace(self, component: str, operation: str, message: str, **kwargs) -> None:
        self.log("TRACE", component, operation, message, **kwargs)


# Global logging factory instance
_logging_factory = LoggingFactory()
_current_logger: MultiBackendLogger | None = None


def setup_system_logging(backend_types: list[str] | None = None, **kwargs) -> MultiBackendLogger:
    """Setup system-wide logging with configurable backends."""
    global _current_logger

    if backend_types is None:
        # Default based on config or environment
        config = get_logging_config()
        backend_types = config.get("backends", ["console"])

    _current_logger = _logging_factory.setup_logging(backend_types, **kwargs)
    return _current_logger


def get_system_logger() -> MultiBackendLogger:
    """Get the current system logger."""
    global _current_logger
    if _current_logger is None:
        _current_logger = setup_system_logging()
    return _current_logger


def close_system_logging() -> None:
    """Close all logging backends."""
    global _current_logger
    if _current_logger:
        _logging_factory.close_all()
        _current_logger = None


# AI-Debugging Optimized Utility Functions
def log_component_start(component: str, operation: str, **context) -> None:
    """Log component operation start with context for AI debugging."""
    logger = get_system_logger()
    logger.info(component, operation, "STARTED", metadata=context)


def log_component_end(
    component: str, operation: str, result_summary: str, duration: float | None = None, **context
) -> None:
    """Log component operation completion with result for AI debugging."""
    logger = get_system_logger()
    logger.info(component, operation, f"COMPLETED: {result_summary}", duration=duration, metadata=context)


def log_data_transformation(component: str, operation: str, input_desc: str, output_desc: str, **context) -> None:
    """Log data transformation for AI debugging - what went in, what came out."""
    logger = get_system_logger()
    logger.debug(component, operation, f"TRANSFORM: {input_desc} → {output_desc}", metadata=context)


def log_decision_point(component: str, operation: str, condition: str, chosen_path: str, **context) -> None:
    """Log decision points for AI debugging - why did system choose path A vs B."""
    logger = get_system_logger()
    logger.debug(component, operation, f"DECISION: {condition} → chose {chosen_path}", metadata=context)


def log_config_usage(component: str, operation: str, config_values: dict) -> None:
    """Log configuration values affecting behavior for AI debugging."""
    logger = get_system_logger()
    logger.trace(component, operation, "CONFIG_ACTIVE", metadata=config_values)


def log_performance_metric(component: str, operation: str, metric_name: str, value: float, **context) -> None:
    """Log performance metrics for AI debugging."""
    logger = get_system_logger()
    logger.debug(component, operation, f"METRIC: {metric_name}={value}", metadata=context)


def log_error_context(component: str, operation: str, error: Exception, context: dict) -> None:
    """Log error with full context for AI debugging - what state was system in when it failed."""
    logger = get_system_logger()
    logger.error(
        component,
        operation,
        f"FAILED: {str(error)}",
        error_type=type(error).__name__,
        stack_trace=str(error),
        metadata=context,
    )
