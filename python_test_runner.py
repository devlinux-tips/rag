#!/usr/bin/env python3
"""
Python Test Runner for Multilingual RAG System
Orchestrates testing for all Python services with AI-friendly trace logging
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict


# ============ AI DEBUGGING INFRASTRUCTURE ============


@dataclass
class TestExecutionContext:
    """Track test execution context for AI debugging."""

    test_file: str
    test_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    mock_objects_detected: int = 0
    assertions_failed: int = 0

    def to_trace_log(self) -> str:
        """Convert to AI-parseable trace log format."""
        duration = (self.end_time - self.start_time) if self.end_time else 0
        return (
            f"TEST_EXEC | file={self.test_file} | test={self.test_name} | "
            f"status={self.status} | duration_ms={duration*1000:.1f} | "
            f"mocks={self.mock_objects_detected} | assertions_failed={self.assertions_failed} | "
            f"error={self.error_type or 'none'}"
        )


class AITraceLogger:
    """AI-friendly trace logging for test debugging."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.contexts: Dict[str, TestExecutionContext] = {}
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def trace(self, component: str, operation: str, details: str):
        """Log trace event for AI pattern recognition."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] TRACE | component={component} | op={operation} | {details}\n"

        with open(self.log_file, "a") as f:
            f.write(log_entry)

        # Also print to console if verbose
        if os.environ.get("TRACE_VERBOSE"):
            print(f"🔍 {log_entry.strip()}")

    def test_started(self, test_file: str, test_name: str):
        """Log test start."""
        key = f"{test_file}::{test_name}"
        self.contexts[key] = TestExecutionContext(
            test_file=test_file, test_name=test_name, start_time=time.time()
        )
        self.trace("test_runner", "test_start", f"test={test_name} | file={test_file}")

    def test_completed(
        self, test_file: str, test_name: str, status: str, error: Optional[str] = None
    ):
        """Log test completion."""
        key = f"{test_file}::{test_name}"
        if key in self.contexts:
            ctx = self.contexts[key]
            ctx.end_time = time.time()
            ctx.status = status
            if error:
                ctx.error_type = (
                    type(error).__name__
                    if hasattr(error, "__class__")
                    else "UnknownError"
                )
                ctx.error_message = str(error)[:200]

            self.trace("test_runner", "test_complete", ctx.to_trace_log())

    def mock_detected(self, test_name: str, obj_type: str, is_mock: bool):
        """Log mock object detection."""
        self.trace(
            "mock_detection", test_name, f"obj_type={obj_type} | is_mock={is_mock}"
        )

    def migration_issue(
        self, component: str, old_impl: str, new_impl: str, migration_status: str
    ):
        """Log migration issues (ChromaDB -> Weaviate)."""
        self.trace(
            "migration",
            component,
            f"old={old_impl} | new={new_impl} | status={migration_status}",
        )

    def config_evolution(self, key_path: str, old_format: str, new_format: str):
        """Log configuration changes."""
        self.trace(
            "config_evolution",
            "schema_change",
            f"path={key_path} | old={old_format} | new={new_format}",
        )

    def generate_report(self) -> Dict:
        """Generate AI-friendly test report."""
        return {
            "total_tests": len(self.contexts),
            "passed": sum(1 for c in self.contexts.values() if c.status == "passed"),
            "failed": sum(1 for c in self.contexts.values() if c.status == "failed"),
            "skipped": sum(1 for c in self.contexts.values() if c.status == "skipped"),
            "mock_heavy_tests": [
                k for k, c in self.contexts.items() if c.mock_objects_detected > 5
            ],
            "error_patterns": self._analyze_error_patterns(),
        }

    def _analyze_error_patterns(self) -> Dict[str, int]:
        """Analyze error patterns for AI debugging."""
        patterns: Dict[str, int] = {}
        for ctx in self.contexts.values():
            if ctx.error_type:
                patterns[ctx.error_type] = patterns.get(ctx.error_type, 0) + 1
        return patterns


# ============ TEST DISCOVERY & ORGANIZATION ============


class TestDiscovery:
    """Discover and categorize tests across services."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.trace_logger = AITraceLogger(repo_root / "test_logs" / "trace.log")

    def discover_all_tests(self) -> Dict[str, List[Path]]:
        """Discover all Python tests in the repository."""
        test_categories: Dict[str, List[Path]] = {
            "rag-service": [],
            "storage": [],
            "llm": [],
            "integration": [],
            "disabled": [],
        }

        # Scan services/rag-service/tests
        rag_tests_dir = self.repo_root / "services" / "rag-service" / "tests"
        if rag_tests_dir.exists():
            for test_file in rag_tests_dir.glob("test_*.py"):
                category = self._categorize_test(test_file)
                test_categories[category].append(test_file)

        self.trace_logger.trace(
            "test_discovery",
            "scan_complete",
            f"found={sum(len(v) for v in test_categories.values())} tests",
        )

        return test_categories

    def _categorize_test(self, test_file: Path) -> str:
        """Categorize test based on filename and content."""
        name = test_file.name

        # Check for ChromaDB tests to disable
        if "chromadb" in name.lower() or "chroma" in name.lower():
            return "disabled"

        # Check content for ChromaDB imports
        try:
            content = test_file.read_text()
            if "import chromadb" in content or "from chromadb" in content:
                self.trace_logger.migration_issue(
                    test_file.name, "chromadb", "weaviate", "needs_migration"
                )
                return "disabled"
        except Exception as e:
            self.trace_logger.trace(
                "test_discovery", "read_error", f"file={test_file} | error={e}"
            )

        # Categorize by name patterns
        if "storage" in name or "vector" in name:
            return "storage"
        elif "llm" in name or "generation" in name or "ollama" in name:
            return "llm"
        elif "integration" in name or "e2e" in name:
            return "integration"
        else:
            return "rag-service"

    def get_chromadb_tests(self) -> List[Path]:
        """Find all tests that use ChromaDB."""
        chromadb_tests = []

        rag_tests_dir = self.repo_root / "services" / "rag-service" / "tests"
        if rag_tests_dir.exists():
            for test_file in rag_tests_dir.glob("test_*.py"):
                try:
                    content = test_file.read_text()
                    if "chromadb" in content.lower():
                        chromadb_tests.append(test_file)
                except:
                    pass

        return chromadb_tests


# ============ TEST RUNNER ============


class PythonTestRunner:
    """Main test runner with AI-friendly logging."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.trace_logger = AITraceLogger(repo_root / "test_logs" / "trace.log")
        self.discovery = TestDiscovery(repo_root)

    def run_tests(
        self,
        categories: Optional[List[str]] = None,
        coverage: bool = False,
        verbose: bool = False,
        markers: Optional[str] = None,
        trace: bool = True,
    ) -> bool:
        """Run tests with specified options."""

        # Discover tests
        all_tests = self.discovery.discover_all_tests()

        # Filter categories
        if categories:
            test_files = []
            for cat in categories:
                test_files.extend(all_tests.get(cat, []))
        else:
            # Run all except disabled
            test_files = []
            for cat, files in all_tests.items():
                if cat != "disabled":
                    test_files.extend(files)

        if not test_files:
            print("❌ No tests found to run")
            return False

        print(f"\n🧪 PYTHON TEST RUNNER - Multilingual RAG System")
        print("=" * 60)
        print(f"📁 Repository: {self.repo_root}")
        print(f"🔍 Tests found: {len(test_files)}")
        print(f"🏷️  Categories: {', '.join(categories) if categories else 'all'}")

        if all_tests.get("disabled"):
            print(f"⚠️  Disabled tests (ChromaDB): {len(all_tests['disabled'])}")
            for test in all_tests["disabled"][:5]:
                print(f"   - {test.name}")

        print("\n" + "=" * 60)

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]

        # Add test paths
        for test_file in test_files:
            cmd.append(str(test_file))

        # Verbosity
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")

        # Coverage
        if coverage:
            cmd.extend(
                [
                    "--cov=services/rag-service/src",
                    "--cov-report=term-missing",
                    "--cov-report=html:test_logs/coverage_html",
                ]
            )

        # Markers
        if markers:
            cmd.extend(["-m", markers])

        # Trace logging via environment
        if trace:
            os.environ["PYTEST_AI_TRACE"] = "1"
            os.environ["TRACE_LOG_FILE"] = str(self.trace_logger.log_file)

        # Color output
        cmd.append("--color=yes")

        # Capture for AI analysis
        cmd.extend(
            [
                "--tb=short",
                "--capture=no",
                f"--junit-xml={self.repo_root}/test_logs/junit.xml",
            ]
        )

        print("🚀 Executing: " + " ".join(cmd[:3]) + " ...")
        print()

        # Run tests
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.repo_root)
        duration = time.time() - start_time

        # Generate report
        if trace:
            report = self.trace_logger.generate_report()
            report_file = self.repo_root / "test_logs" / "ai_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            print(f"\n📊 AI Debug Report: {report_file}")
            print(f"   - Total: {report['total_tests']} tests")
            print(f"   - Passed: {report['passed']}")
            print(f"   - Failed: {report['failed']}")
            if report["error_patterns"]:
                print("   - Error patterns:")
                for error, count in report["error_patterns"].items():
                    print(f"     • {error}: {count}")

        print(f"\n⏱️  Total time: {duration:.2f} seconds")

        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Tests failed (exit code: {result.returncode})")
            print("\n💡 Debug hints:")
            print("   1. Check test_logs/trace.log for AI-friendly debugging info")
            print("   2. Review disabled ChromaDB tests for migration needs")
            print("   3. Use --verbose for detailed output")

        return result.returncode == 0

    def migrate_chromadb_tests(self, dry_run: bool = True):
        """Identify and help migrate ChromaDB tests to Weaviate."""
        print("\n🔄 CHROMADB TO WEAVIATE TEST MIGRATION")
        print("=" * 60)

        chromadb_tests = self.discovery.get_chromadb_tests()

        if not chromadb_tests:
            print("✅ No ChromaDB tests found!")
            return

        print(f"Found {len(chromadb_tests)} tests using ChromaDB:\n")

        for test_file in chromadb_tests:
            print(f"📄 {test_file.relative_to(self.repo_root)}")

            if not dry_run:
                # Create migration stub
                new_file = test_file.parent / f"{test_file.stem}_weaviate.py"

                migration_template = '''"""
Migrated from ChromaDB to Weaviate
Original: {original}
Migration Date: {date}
"""

import pytest
from unittest.mock import Mock, AsyncMock
import weaviate

# TODO: Migrate test implementation from ChromaDB to Weaviate

@pytest.mark.skip(reason="Awaiting Weaviate migration implementation")
def test_placeholder():
    """Placeholder for migrated test."""
    pass
'''.format(
                    original=test_file.name, date=datetime.now().isoformat()
                )

                print(f"   → Creating migration stub: {new_file.name}")
                new_file.write_text(migration_template)

                # Add skip marker to original
                content = test_file.read_text()
                if "@pytest.mark.skip" not in content:
                    lines = content.splitlines()
                    lines.insert(0, "import pytest")
                    lines.insert(1, '@pytest.mark.skip(reason="Migrating to Weaviate")')
                    test_file.write_text("\n".join(lines))
                    print(f"   → Added skip marker to original")

        if dry_run:
            print("\n✨ Dry run complete. Use --migrate to apply changes.")


# ============ MAIN ENTRY POINT ============


def main():
    """Main entry point for Python test runner."""
    parser = argparse.ArgumentParser(
        description="Python Test Runner for Multilingual RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python python_test_runner.py                     # Run all tests
  python python_test_runner.py --coverage          # With coverage report
  python python_test_runner.py --category storage  # Run storage tests only
  python python_test_runner.py --migrate-chromadb  # Migrate ChromaDB tests
  python python_test_runner.py --list              # List all tests
  python python_test_runner.py -vv --trace         # Verbose with AI trace logging
        """,
    )

    parser.add_argument(
        "--category",
        "-c",
        choices=["rag-service", "storage", "llm", "integration", "all"],
        nargs="+",
        help="Test categories to run",
    )

    parser.add_argument(
        "--coverage", "--cov", action="store_true", help="Enable coverage reporting"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-vv for very verbose)",
    )

    parser.add_argument(
        "--markers", "-m", help="Pytest markers to filter tests (e.g., 'not slow')"
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        default=True,
        help="Enable AI-friendly trace logging (default: True)",
    )

    parser.add_argument(
        "--no-trace", action="store_true", help="Disable AI trace logging"
    )

    parser.add_argument(
        "--list", action="store_true", help="List all available tests by category"
    )

    parser.add_argument(
        "--migrate-chromadb",
        action="store_true",
        help="Migrate ChromaDB tests to Weaviate",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run for migration (default: True)",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply migration changes (disables dry-run)",
    )

    args = parser.parse_args()

    # Setup repo root
    repo_root = Path(__file__).parent.absolute()
    runner = PythonTestRunner(repo_root)

    # Handle list command
    if args.list:
        discovery = TestDiscovery(repo_root)
        all_tests = discovery.discover_all_tests()

        print("\n📋 AVAILABLE PYTHON TESTS")
        print("=" * 60)

        for category, tests in all_tests.items():
            if tests:
                print(f"\n📦 {category.upper()} ({len(tests)} tests)")
                for test in sorted(tests)[:10]:
                    print(f"   • {test.name}")
                if len(tests) > 10:
                    print(f"   ... and {len(tests) - 10} more")

        return 0

    # Handle migration command
    if args.migrate_chromadb:
        runner.migrate_chromadb_tests(dry_run=not args.apply)
        return 0

    # Run tests
    trace = args.trace and not args.no_trace
    success = runner.run_tests(
        categories=args.category,
        coverage=args.coverage,
        verbose=args.verbose > 0,
        markers=args.markers,
        trace=trace,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
