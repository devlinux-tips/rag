#!/usr/bin/env python3
"""
Multi-tenant RAG System CLI - Main Entry Point

This is the primary interface for the multilingual RAG system with Croatian and English support.
Provides document processing, semantic search, and answer generation capabilities.

Usage:
    python rag.py --tenant development --user dev_user --language hr query "Što je RAG sustav?"
    python rag.py --tenant acme --user john --language en process-docs ./docs/
    python rag.py --language hr status
    python rag.py --help

Best Practice: Always use this script as the main entry point rather than module imports.
"""

import sys
from pathlib import Path

# Add src to Python path for proper module resolution
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli.rag_cli import cli_main

if __name__ == "__main__":
    cli_main()
