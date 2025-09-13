#!/usr/bin/env python3
"""
Script to format all Python files in the project using modern tooling.
Uses ruff (fast linter/formatter) + black + mypy for comprehensive code quality.
Run this to ensure consistent formatting across the codebase.
Works from any directory by finding repository root and using root configurations.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, allow_failure=False):
    """Run a command and report its status."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return allow_failure


def check_tool_availability():
    """Check if required tools are available."""
    tools = {"ruff": "ruff", "black": "black", "mypy": "mypy"}
    available_tools = {}

    for name, command in tools.items():
        if shutil.which(command):
            available_tools[name] = command
            print(f"✅ {name} found: {command}")
        else:
            print(f"❌ {name} not found in PATH")

    return available_tools


def find_repo_root():
    """Find repository root by looking for .git directory."""
    current_path = Path.cwd()
    repo_root = current_path

    while repo_root != repo_root.parent:
        if (repo_root / ".git").exists():
            return repo_root
        repo_root = repo_root.parent

    # If no .git found, use current directory
    print("⚠️ Git repository root not found, using current directory")
    return current_path


def get_python_targets(repo_root, current_path):
    """Determine which Python files/directories to format."""
    python_targets = []

    # If we're in rag-service directory or subdirectory, format local files
    if "rag-service" in str(current_path):
        # Find the rag-service directory relative to repo root
        rag_service_path = None
        for part_path in [current_path] + list(current_path.parents):
            if part_path.name == "rag-service":
                rag_service_path = part_path.relative_to(repo_root)
                break

        if rag_service_path:
            # Format rag-service specific paths
            for target in ["src/", "tests/", "*.py"]:
                target_path = rag_service_path / target
                if target.endswith(".py"):
                    # Check for individual Python files
                    if (repo_root / target_path).exists():
                        python_targets.append(str(target_path))
                else:
                    # Check for directories
                    if (repo_root / target_path).exists():
                        python_targets.append(str(target_path))

        # Also include Python files in rag-service root
        rag_service_root = (
            repo_root / rag_service_path.parts[0] / rag_service_path.parts[1]
            if len(rag_service_path.parts) >= 2
            else repo_root
        )
        for py_file in rag_service_root.glob("*.py"):
            rel_path = py_file.relative_to(repo_root)
            python_targets.append(str(rel_path))
    else:
        # If we're in repository root, format the entire services directory
        services_path = repo_root / "services"
        if services_path.exists():
            python_targets.append("services/")

        # Also include root-level Python files
        for py_file in repo_root.glob("*.py"):
            python_targets.append(str(py_file.relative_to(repo_root)))

    return python_targets


def main():
    """Format all Python files in the project."""
    print("🔍 Checking tool availability...")
    available_tools = check_tool_availability()

    if not available_tools:
        print("❌ No formatting tools available. Please install them:")
        print("pip install ruff black mypy")
        sys.exit(1)

    # Find repository root and determine working context
    repo_root = find_repo_root()
    current_path = Path.cwd()

    print(f"📂 Repository root: {repo_root}")
    print(f"📂 Current directory: {current_path}")

    # Get Python targets based on current location
    python_targets = get_python_targets(repo_root, current_path)

    if not python_targets:
        print("❌ No Python files or directories found to format")
        sys.exit(1)

    print(f"🎯 Targets: {python_targets}")
    targets_str = " ".join(python_targets)

    print("\n🚀 Formatting Python files in the project...")

    success_count = 0
    total_operations = 0

    # Change to repository root for consistent configuration file discovery
    original_cwd = Path.cwd()
    if repo_root != original_cwd:
        print(f"🔄 Changing to repository root: {repo_root}")
        os.chdir(repo_root)

    try:
        # Ruff: fix issues and format code (if available)
        if "ruff" in available_tools:
            total_operations += 1
            # Ruff fix: auto-fix import sorting and style issues
            if run_command(f"ruff check --fix {targets_str}", "Ruff auto-fix"):
                success_count += 1

            total_operations += 1
            # Ruff format: code formatting
            if run_command(f"ruff format {targets_str}", "Ruff formatting"):
                success_count += 1

        # Format with black (if available and ruff not available)
        elif "black" in available_tools:
            total_operations += 1
            # Black will use root pyproject.toml or default settings
            if run_command(f"black {targets_str}", "Black formatting"):
                success_count += 1

        # Type checking with mypy (if available)
        if "mypy" in available_tools:
            total_operations += 1
            # mypy will use root configuration
            if run_command(
                f"mypy {targets_str}", "MyPy type checking", allow_failure=True
            ):
                success_count += 1
    finally:
        # Restore original working directory
        if repo_root != original_cwd:
            os.chdir(original_cwd)

    # Report results
    print(
        f"\n📊 Results: {success_count}/{total_operations} operations completed successfully"
    )

    if success_count == total_operations:
        print("🎉 All formatting and type checking completed successfully!")
    elif success_count >= total_operations - 1:  # Allow mypy to fail
        print("✅ Formatting completed successfully!")
        if "mypy" in available_tools and success_count < total_operations:
            print("⚠️  Some type checking issues remain - check output above")
    else:
        print("❌ Some critical formatting operations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
