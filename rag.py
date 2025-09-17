#!/usr/bin/env python3
"""
Multi-tenant RAG System - Main Entry Point

This is the main entry point for the RAG system. It delegates to the actual
implementation in services/rag-service/ while maintaining proper working directory.

Usage:
    python rag.py --help
    python rag.py --tenant development --user dev_user --language hr query "Što je RAG?"
    python rag.py --tenant development --user dev_user --language en status
"""

import os
import sys
from pathlib import Path


def main():
    """Main entry point that delegates to the real RAG CLI."""
    # Get the directory where this script is located (project root)
    project_root = Path(__file__).parent.absolute()

    # Path to the real RAG service
    rag_service_dir = project_root / "services" / "rag-service"
    rag_service_script = rag_service_dir / "rag.py"

    # Fail-fast validation: Ensure the RAG service exists
    if not rag_service_dir.exists():
        print("❌ ERROR: RAG service directory not found!")
        print(f"Expected: {rag_service_dir}")
        print("This indicates a corrupted project structure.")
        sys.exit(1)

    if not rag_service_script.exists():
        print("❌ ERROR: RAG service script not found!")
        print(f"Expected: {rag_service_script}")
        print("This indicates missing RAG implementation.")
        sys.exit(1)

    # Change to the RAG service directory (required for proper imports)
    original_cwd = os.getcwd()

    try:
        os.chdir(rag_service_dir)

        # Add the RAG service directory to Python path for imports
        if str(rag_service_dir) not in sys.path:
            sys.path.insert(0, str(rag_service_dir))

        # Import and execute the real RAG CLI
        from src.cli.rag_cli import main as rag_main
        import asyncio

        # Execute the real CLI with original arguments (it's an async function)
        asyncio.run(rag_main())

    except ImportError as e:
        print("❌ IMPORT ERROR: Failed to import RAG CLI components!")
        print(f"Error: {e}")
        print("This indicates missing dependencies or broken imports.")
        print("\nTroubleshooting steps:")
        print("1. Ensure you're in the correct project directory")
        print("2. Check that all dependencies are installed")
        print("3. Verify the virtual environment is activated")
        sys.exit(1)

    except Exception as e:
        print(f"❌ EXECUTION ERROR: {e}")
        print("The RAG CLI encountered an unexpected error.")
        sys.exit(1)

    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()