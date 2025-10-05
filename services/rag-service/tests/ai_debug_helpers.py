"""
AI Debugging Helpers for Tests
Provides comprehensive trace logging and mock detection for AI-assisted debugging
"""

import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
from unittest.mock import Mock, MagicMock, AsyncMock
import logging

# Configure AI-friendly logging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("PYTEST_AI_TRACE") else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ai_debug")


# ============ MOCK DETECTION ============


class MockDetector:
    """Detect and log mock objects for AI debugging."""

    @staticmethod
    def is_mock(obj: Any) -> bool:
        """Check if object is a mock."""
        return (
            isinstance(obj, (Mock, MagicMock, AsyncMock))
            or "Mock" in str(type(obj))
            or hasattr(obj, "_mock_name")
            or hasattr(obj, "_spec_set")
        )

    @staticmethod
    def log_mock_detection(
        component: str, operation: str, obj: Any, expected_type: Optional[str] = None
    ):
        """Log mock detection for AI pattern recognition."""
        is_mock = MockDetector.is_mock(obj)
        obj_type = type(obj).__name__
        obj_id = id(obj)

        log_msg = (
            f"MOCK_DETECT | component={component} | op={operation} | "
            f"obj_type={obj_type} | is_mock={is_mock} | "
            f"expected={expected_type or 'any'} | obj_id={obj_id}"
        )

        logger.debug(log_msg)

        # Also write to trace file if enabled
        trace_file = os.environ.get("TRACE_LOG_FILE")
        if trace_file:
            with open(trace_file, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] {log_msg}\n")

        return is_mock

    @staticmethod
    def analyze_test_mocks(test_function: Callable) -> Dict[str, Any]:
        """Analyze mock usage in a test function."""
        source = inspect.getsource(test_function)
        mock_count = (
            source.count("Mock(")
            + source.count("MagicMock(")
            + source.count("AsyncMock(")
        )
        patch_count = source.count("@patch") + source.count("patch(")

        return {
            "function": test_function.__name__,
            "mock_count": mock_count,
            "patch_count": patch_count,
            "is_mock_heavy": mock_count + patch_count > 5,
        }


# ============ ASSERTION CONTEXT LOGGING ============


class AssertionLogger:
    """Log assertion context for AI debugging."""

    @staticmethod
    def log_assertion_context(
        test_name: str,
        component: str,
        expected: Any,
        actual: Any,
        assertion_type: str = "equality",
    ):
        """Log detailed assertion context."""
        expected_type = type(expected).__name__
        actual_type = type(actual).__name__
        expected_repr = repr(expected)[:200]
        actual_repr = repr(actual)[:200]

        log_msg = (
            f"ASSERTION | test={test_name} | component={component} | "
            f"type={assertion_type} | "
            f"expected_type={expected_type} | actual_type={actual_type} | "
            f"expected_value={expected_repr} | actual_value={actual_repr}"
        )

        logger.error(log_msg)

        # Detailed mismatch analysis
        if expected_type != actual_type:
            logger.error(
                f"TYPE_MISMATCH | expected={expected_type} | got={actual_type}"
            )

        if hasattr(expected, "__dict__") and hasattr(actual, "__dict__"):
            # Object attribute comparison
            expected_attrs = (
                set(vars(expected).keys()) if hasattr(expected, "__dict__") else set()
            )
            actual_attrs = (
                set(vars(actual).keys()) if hasattr(actual, "__dict__") else set()
            )

            missing = expected_attrs - actual_attrs
            extra = actual_attrs - expected_attrs

            if missing:
                logger.error(f"MISSING_ATTRS | attrs={missing}")
            if extra:
                logger.error(f"EXTRA_ATTRS | attrs={extra}")

    @staticmethod
    def assert_with_context(
        test_name: str,
        component: str,
        expected: Any,
        actual: Any,
        assertion_type: str = "equality",
    ):
        """Enhanced assertion with logging."""
        try:
            if assertion_type == "equality":
                assert expected == actual
            elif assertion_type == "type":
                assert type(expected) == type(actual)
            elif assertion_type == "contains":
                assert expected in actual
            else:
                assert expected == actual
        except AssertionError:
            AssertionLogger.log_assertion_context(
                test_name, component, expected, actual, assertion_type
            )
            raise


# ============ API CONTRACT VALIDATION ============


class APIContractValidator:
    """Validate API contracts and log mismatches."""

    @staticmethod
    def log_api_mismatch(
        component: str,
        method: str,
        expected_signature: List[str],
        actual_params: Dict[str, Any],
    ):
        """Log API signature mismatches."""
        actual_keys = set(actual_params.keys())
        expected_keys = set(expected_signature)

        missing_params = expected_keys - actual_keys
        extra_params = actual_keys - expected_keys

        log_msg = (
            f"API_MISMATCH | component={component} | method={method} | "
            f"expected={expected_signature} | actual={list(actual_keys)} | "
            f"missing={list(missing_params)} | extra={list(extra_params)}"
        )

        logger.error(log_msg)

        return {
            "component": component,
            "method": method,
            "missing": list(missing_params),
            "extra": list(extra_params),
            "valid": len(missing_params) == 0 and len(extra_params) == 0,
        }

    @staticmethod
    def validate_provider_interface(
        provider: Any, expected_methods: Dict[str, List[str]]
    ):
        """Validate provider implements expected interface."""
        results = {}

        for method_name, expected_params in expected_methods.items():
            if hasattr(provider, method_name):
                method = getattr(provider, method_name)
                if callable(method):
                    # Get actual parameters
                    sig = inspect.signature(method)
                    actual_params = list(sig.parameters.keys())

                    # Remove 'self' if present
                    if "self" in actual_params:
                        actual_params.remove("self")

                    results[method_name] = {
                        "exists": True,
                        "expected_params": expected_params,
                        "actual_params": actual_params,
                        "matches": actual_params == expected_params,
                    }
                else:
                    results[method_name] = {"exists": True, "error": "Not callable"}
            else:
                results[method_name] = {
                    "exists": False,
                    "expected_params": expected_params,
                }

        # Log validation results
        logger.info(
            f"PROVIDER_VALIDATION | provider={type(provider).__name__} | "
            f"results={json.dumps(results, indent=2)}"
        )

        return results


# ============ MIGRATION TRACKING ============


class MigrationTracker:
    """Track ChromaDB to Weaviate migration status."""

    @staticmethod
    def log_migration_issue(
        component: str,
        old_impl: str,
        new_impl: str,
        status: str,
        details: Optional[str] = None,
    ):
        """Log migration issues for systematic fixing."""
        log_msg = (
            f"MIGRATION | component={component} | "
            f"old={old_impl} | new={new_impl} | "
            f"status={status}"
        )

        if details:
            log_msg += f" | details={details}"

        logger.warning(log_msg)

    @staticmethod
    @contextmanager
    def track_migration(test_name: str, from_system: str, to_system: str):
        """Context manager to track migration progress."""
        start_time = time.time()
        logger.info(
            f"MIGRATION_START | test={test_name} | from={from_system} | to={to_system}"
        )

        try:
            yield
            duration = time.time() - start_time
            logger.info(
                f"MIGRATION_SUCCESS | test={test_name} | duration_ms={duration*1000:.1f}"
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"MIGRATION_FAIL | test={test_name} | duration_ms={duration*1000:.1f} | "
                f"error={type(e).__name__} | msg={str(e)}"
            )
            raise


# ============ CONFIG EVOLUTION TRACKING ============


class ConfigEvolutionTracker:
    """Track configuration schema changes."""

    @staticmethod
    def log_config_evolution(
        key_path: str, old_format: Any, new_format: Any, compatibility_action: str
    ):
        """Track config schema evolution."""
        log_msg = (
            f"CONFIG_EVOLUTION | path={key_path} | "
            f"old_type={type(old_format).__name__} | "
            f"new_type={type(new_format).__name__} | "
            f"action={compatibility_action} | "
            f"migration_needed={old_format != new_format}"
        )

        logger.info(log_msg)

    @staticmethod
    def validate_config_migration(old_config: Dict, new_config: Dict) -> Dict[str, Any]:
        """Validate configuration migration completeness."""
        results: Dict[str, Any] = {
            "removed_keys": [],
            "added_keys": [],
            "type_changes": [],
            "value_changes": [],
        }

        def flatten_dict(d: Dict, parent_key: str = "") -> Dict:
            """Flatten nested dictionary for comparison."""
            items: List[Tuple[str, Any]] = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        old_flat = flatten_dict(old_config)
        new_flat = flatten_dict(new_config)

        # Find differences
        old_keys = set(old_flat.keys())
        new_keys = set(new_flat.keys())

        results["removed_keys"] = list(old_keys - new_keys)
        results["added_keys"] = list(new_keys - old_keys)

        # Check type and value changes
        for key in old_keys & new_keys:
            old_val = old_flat[key]
            new_val = new_flat[key]

            if type(old_val) != type(new_val):
                results["type_changes"].append(
                    {
                        "key": key,
                        "old_type": type(old_val).__name__,
                        "new_type": type(new_val).__name__,
                    }
                )
            elif old_val != new_val:
                results["value_changes"].append(
                    {
                        "key": key,
                        "old_value": str(old_val)[:50],
                        "new_value": str(new_val)[:50],
                    }
                )

        logger.info(
            f"CONFIG_MIGRATION_VALIDATION | results={json.dumps(results, indent=2)}"
        )
        return results


# ============ TEST DECORATORS ============


def ai_debug_trace(component: str = "test"):
    """Decorator to add AI debug tracing to test functions."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            test_name = func.__name__
            start_time = time.time()

            logger.info(f"TEST_START | component={component} | test={test_name}")

            # Analyze mocks
            mock_analysis = MockDetector.analyze_test_mocks(func)
            if mock_analysis["is_mock_heavy"]:
                logger.warning(
                    f"MOCK_HEAVY_TEST | test={test_name} | "
                    f"mocks={mock_analysis['mock_count']} | "
                    f"patches={mock_analysis['patch_count']}"
                )

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(
                    f"TEST_SUCCESS | component={component} | test={test_name} | "
                    f"duration_ms={duration*1000:.1f}"
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                error_type = type(e).__name__
                error_msg = str(e)[:200]

                logger.error(
                    f"TEST_FAIL | component={component} | test={test_name} | "
                    f"duration_ms={duration*1000:.1f} | "
                    f"error_type={error_type} | error_msg={error_msg}"
                )

                # Log stack trace for AI analysis
                import traceback

                tb = traceback.format_exc()
                logger.error(f"STACK_TRACE | test={test_name} | trace={tb[:500]}")

                raise

        return wrapper if not inspect.iscoroutinefunction(func) else wrapper

    return decorator


def skip_chromadb(reason: str = "ChromaDB deprecated, migrating to Weaviate"):
    """Decorator to skip ChromaDB tests."""
    import pytest

    def decorator(func: Callable):
        return pytest.mark.skip(reason=reason)(func)

    return decorator


# ============ TEST FIXTURES ============


class AIDebugFixtures:
    """Common fixtures for AI debugging."""

    @staticmethod
    def create_traced_mock(name: str, spec: Optional[Any] = None) -> Mock:
        """Create a mock with trace logging."""
        mock = Mock(spec=spec, name=name)

        # Log mock creation
        MockDetector.log_mock_detection(
            component="fixture",
            operation=f"create_mock_{name}",
            obj=mock,
            expected_type=str(spec) if spec else None,
        )

        return mock

    @staticmethod
    def create_traced_async_mock(name: str, spec: Optional[Any] = None) -> AsyncMock:
        """Create an async mock with trace logging."""
        mock = AsyncMock(spec=spec, name=name)

        MockDetector.log_mock_detection(
            component="fixture",
            operation=f"create_async_mock_{name}",
            obj=mock,
            expected_type=str(spec) if spec else None,
        )

        return mock


# ============ REPORTING ============


class AITestReporter:
    """Generate AI-friendly test reports."""

    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def add_result(
        self,
        test_name: str,
        status: str,
        duration: float,
        error: Optional[str] = None,
        mocks_used: int = 0,
    ):
        """Add test result."""
        self.results.append(
            {
                "test": test_name,
                "status": status,
                "duration_ms": duration * 1000,
                "error": error,
                "mocks_used": mocks_used,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        total_duration = time.time() - self.start_time

        passed = sum(1 for r in self.results if r["status"] == "passed")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        skipped = sum(1 for r in self.results if r["status"] == "skipped")

        # Error pattern analysis
        error_patterns: Dict[str, int] = {}
        for result in self.results:
            if result["error"]:
                error_type = (
                    result["error"].split(":")[0]
                    if ":" in result["error"]
                    else "Unknown"
                )
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

        # Mock usage analysis
        mock_heavy = [r["test"] for r in self.results if r.get("mocks_used", 0) > 5]
        avg_mocks = (
            sum(r.get("mocks_used", 0) for r in self.results) / len(self.results)
            if self.results
            else 0
        )

        report = {
            "summary": {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "duration_seconds": total_duration,
                "success_rate": (
                    (passed / len(self.results) * 100) if self.results else 0
                ),
            },
            "error_patterns": error_patterns,
            "mock_analysis": {
                "average_mocks_per_test": avg_mocks,
                "mock_heavy_tests": mock_heavy,
                "mock_heavy_count": len(mock_heavy),
            },
            "slowest_tests": sorted(
                self.results, key=lambda x: x["duration_ms"], reverse=True
            )[:5],
            "failed_tests": [r for r in self.results if r["status"] == "failed"],
            "timestamp": datetime.now().isoformat(),
        }

        # Save to file
        report_file = Path("test_logs") / "ai_test_report.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"AI_REPORT_GENERATED | file={report_file}")

        return report


# Export all helpers
__all__ = [
    "MockDetector",
    "AssertionLogger",
    "APIContractValidator",
    "MigrationTracker",
    "ConfigEvolutionTracker",
    "ai_debug_trace",
    "skip_chromadb",
    "AIDebugFixtures",
    "AITestReporter",
]
