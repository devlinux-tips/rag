#!/usr/bin/env python3
"""
Multi-tenant RAG System CLI entry point.

This script provides a convenient way to run the RAG system CLI with full
multi-tenant support, including --tenant and --user switches.

Usage:
    python rag_new.py --tenant development --user dev_user --language hr query "Što je RAG sustav?"
    python rag_new.py --tenant acme --user john --language en process-docs ./docs/
    python rag_new.py --help
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli.rag_cli import main

if __name__ == "__main__":
    main()
