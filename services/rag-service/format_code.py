#!/usr/bin/env python3
"""
AI-Friendly Python Code Formatter

Uses Ruff for both linting and formatting to preserve AI-friendly patterns.
Configured to maintain readable one-liners and clean coding patterns.

Workflow:
1. Ruff check --fix: Import sorting, unused code removal, linting fixes
2. Ruff format: Code formatting that preserves clean one-liners
3. MyPy: Type checking focused on core system (excludes CLI)

This tool MAKES CHANGES to your code.
Pre-commit hooks only CHECK - they won't modify files during commit.

Use this tool before committing to ensure code passes pre-commit checks.
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
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
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
    """Check if required tools are available, prioritizing Ruff."""
    # Prioritize Ruff as primary formatter
    tools = {"ruff": "ruff", "mypy": "mypy", "black": "black"}
    available_tools = {}

    for name, command in tools.items():
        if shutil.which(command):
            available_tools[name] = command
            print(f"✅ {name} found: {command}")
        else:
            print(f"❌ {name} not found in PATH")

    # Ruff is essential for AI-friendly formatting
    if "ruff" not in available_tools:
        print("⚠️  Ruff not found - this is the preferred AI-friendly formatter")
        print("   Install with: pip install ruff")

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

    # If we're in rag-service directory or subdirectory, format ONLY core src subdirectories
    if "rag-service" in str(current_path):
        # Find the rag-service directory relative to repo root
        rag_service_path = None
        for part_path in [current_path] + list(current_path.parents):
            if part_path.name == "rag-service":
                rag_service_path = part_path.relative_to(repo_root)
                break

        if rag_service_path:
            # WHITELIST: Only include core src subdirectories
            core_dirs = ["utils", "generation", "preprocessing", "retrieval", "models", "pipeline", "vectordb", "cli"]

            src_path = rag_service_path / "src"
            if (repo_root / src_path).exists():
                src_full_path = repo_root / src_path
                for core_dir in core_dirs:
                    core_dir_path = src_full_path / core_dir
                    if core_dir_path.exists() and core_dir_path.is_dir():
                        rel_path = core_dir_path.relative_to(repo_root)
                        python_targets.append(str(rel_path))
    else:
        # If we're in repository root, format both rag-service and rag-api
        rag_service_path = repo_root / "services" / "rag-service"
        if rag_service_path.exists():
            core_dirs = ["utils", "generation", "preprocessing", "retrieval", "models", "pipeline", "vectordb", "cli"]

            src_path = rag_service_path / "src"
            if src_path.exists():
                for core_dir in core_dirs:
                    core_dir_path = src_path / core_dir
                    if core_dir_path.exists() and core_dir_path.is_dir():
                        rel_path = core_dir_path.relative_to(repo_root)
                        python_targets.append(str(rel_path))

        # Add rag-api service
        rag_api_path = repo_root / "services" / "rag-api"
        if rag_api_path.exists():
            rel_path = rag_api_path.relative_to(repo_root)
            python_targets.append(str(rel_path))

    return python_targets


def main():
    """Format all Python files in the project."""
    print("🔍 Checking tool availability...")
    available_tools = check_tool_availability()

    if not available_tools:
        print("❌ No formatting tools available. Please install them:")
        print("   pip install ruff mypy      # Recommended for AI-friendly formatting")
        print("   pip install black          # Optional fallback formatter")
        sys.exit(1)

    # Warn if Ruff is not available (preferred tool)
    if "ruff" not in available_tools:
        print("⚠️  Ruff not available - this is the preferred AI-friendly formatter")
        print("   Consider installing: pip install ruff")

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

    print("\n🚀 AI-Friendly Python Code Formatting")
    print("    Focus: Core system with consistent best practices")
    print("    Strategy: Preserve readable one-liners, fix imports, maintain clean code")
    print("    Tools: Ruff (linting & imports) → Ruff format → MyPy (type checking)")
    print("    Exclusions: cache directories, hidden directories")
    print("")

    success_count = 0
    total_operations = 0

    # Change to repository root for consistent configuration file discovery
    original_cwd = Path.cwd()
    if repo_root != original_cwd:
        print(f"🔄 Changing to repository root: {repo_root}")
        os.chdir(repo_root)

    try:
        # Ruff-only approach: Both linting and formatting for consistency
        if "ruff" in available_tools:
            total_operations += 1
            # Ruff check: Fix import sorting, unused imports, and other issues
            # Uses pyproject.toml configuration for AI-friendly rules
            if run_command(
                f"ruff check --fix --unsafe-fixes {targets_str}", "Ruff auto-fix (imports, unused code, style)"
            ):
                success_count += 1

        # Use Ruff format - better at preserving AI-friendly one-liners than Black
        if "ruff" in available_tools:
            print("🎯 Using Ruff format - Superior at preserving AI-friendly one-liners")
            total_operations += 1
            # Ruff format: Code formatting that preserves readable one-liners better than Black
            # Respects pyproject.toml settings to preserve readable patterns
            if run_command(f"ruff format {targets_str}", "Ruff formatting (preserves clean one-liners)"):
                success_count += 1

        # Type checking with MyPy (focuses on core system, excludes CLI)
        if "mypy" in available_tools:
            total_operations += 1
            # MyPy uses pyproject.toml exclusion rules to focus on core system
            if run_command(f"mypy {targets_str}", "MyPy type checking (core system focus)", allow_failure=True):
                success_count += 1
    finally:
        # Restore original working directory
        if repo_root != original_cwd:
            os.chdir(original_cwd)

    # Report results with AI-friendly focus
    print(f"\n📊 Results: {success_count}/{total_operations} operations completed successfully")

    if success_count == total_operations:
        print("🎉 All AI-friendly formatting and type checking completed successfully!")
        print("    - Core system follows Python best practices")
        print("    - Clean one-liners preserved with Ruff formatting")
        print("    - Import sorting and unused code removal completed with Ruff")
    elif success_count >= total_operations - 1:  # Allow mypy to fail
        print("✅ AI-friendly formatting completed successfully!")
        print("    - Ruff formatting preserves clean one-liner patterns")
        print("    - Ruff linting and import optimization completed")
        if "mypy" in available_tools and success_count < total_operations:
            print("⚠️  Some type checking issues remain")
    else:
        print("❌ Critical formatting operations failed")
        print("    Check output above for specific issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
