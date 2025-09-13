#!/usr/bin/env python3
"""
Test Runner for Multilingual RAG System
Runs all tests with optional coverage reporting using pytest
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"], check=True)

    # Check for coverage if --cov flag will be used
    try:
        import coverage
    except ImportError:
        print(
            "ℹ️  coverage not installed. Use 'pip install coverage pytest-cov' for coverage reports"
        )


def run_tests(coverage_enabled=False, verbose=False):
    """Run all tests with optional coverage"""

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", "tests/"]

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    if coverage_enabled:
        try:
            import pytest_cov

            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        except ImportError:
            print("⚠️  pytest-cov not installed. Running tests without coverage.")
            print("   Install with: pip install pytest-cov")
            coverage_enabled = False

    # Add colored output
    cmd.append("--color=yes")

    print("🧪 MULTILINGUAL RAG TEST SUITE")
    print("=" * 50)

    if coverage_enabled:
        print("Running all tests with coverage analysis...")
    else:
        print("Running all tests...")
    print()

    try:
        # Run pytest
        result = subprocess.run(cmd, cwd=os.getcwd())

        if result.returncode == 0:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def discover_tests():
    """Discover and list all available tests"""
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("❌ Tests directory not found")
        return

    test_files = list(tests_dir.glob("test_*.py"))

    print("📋 AVAILABLE TESTS")
    print("=" * 30)
    for test_file in sorted(test_files):
        print(f"  • {test_file}")
    print(f"\nTotal: {len(test_files)} test files")


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="Run multilingual RAG system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py                    # Run all tests
  python test_runner.py --cov             # Run tests with coverage
  python test_runner.py --list            # List available tests
  python test_runner.py -v --cov          # Verbose output with coverage
        """,
    )

    parser.add_argument(
        "--cov",
        "--coverage",
        action="store_true",
        help="Enable test coverage reporting",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose test output")

    parser.add_argument("--list", action="store_true", help="List available test files")

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    if args.list:
        discover_tests()
        return

    # Run tests
    success = run_tests(coverage_enabled=args.cov, verbose=args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
