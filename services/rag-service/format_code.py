#!/usr/bin/env python3
"""
Script to format all Python files in the project using black and isort.
Run this to ensure consistent formatting across the codebase.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
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
        return False


def main():
    """Format all Python files in the project."""
    project_root = Path(__file__).parent.parent.parent
    venv_python = project_root / "venv" / "bin" / "python"
    venv_black = project_root / "venv" / "bin" / "black"
    venv_isort = project_root / "venv" / "bin" / "isort"

    # Check if virtual environment exists
    if not venv_python.exists():
        print("❌ Virtual environment not found. Please run:")
        print("python3 -m venv venv")
        print("source venv/bin/activate")
        print("pip install -r requirements.txt")
        sys.exit(1)

    print("🚀 Formatting Python files in the project...")

    # Format with black
    success1 = run_command(f"{venv_black} src/ tests/ *.py", "Black formatting")

    # Sort imports with isort
    success2 = run_command(f"{venv_isort} src/ tests/ *.py", "Import sorting")

    # Run flake8 to check for any remaining issues
    success3 = run_command(f"{project_root}/venv/bin/flake8 src/ tests/", "Flake8 linting")

    if success1 and success2:
        print("\n✅ All formatting completed successfully!")
        if not success3:
            print("⚠️  Some linting issues remain (see above)")
    else:
        print("\n❌ Some formatting operations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
