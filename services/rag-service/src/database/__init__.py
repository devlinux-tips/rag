"""
Database Protocol Layer for RAG System
Provides swappable database backends: SurrealDB for local/on-premise, Supabase for cloud deployment.

This module follows fail-fast philosophy with comprehensive AI-friendly logging.
"""

from .factory import create_database_provider
from .protocols import DatabaseProvider

__all__ = ["DatabaseProvider", "create_database_provider"]