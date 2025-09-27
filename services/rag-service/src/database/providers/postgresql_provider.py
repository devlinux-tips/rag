"""
PostgreSQL Provider Implementation
Local/cloud database implementation with AI-friendly logging.

PostgreSQL Configuration:
- Uses connection pooling for performance
- Supports multi-tenant data isolation
- Standard SQL with JSON support for metadata
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import json
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from ...utils.error_handler import StorageError as DatabaseError
from ...utils.logging_factory import (
    get_system_logger, log_component_start, log_component_end,
    log_decision_point, log_error_context
)
from ...models.multitenant_models import (
    Tenant, User, Document, Chunk, SearchQuery,
    CategorizationTemplate, SystemConfig
)


class PostgreSQLProvider:
    """
    PostgreSQL implementation of DatabaseProvider protocol.

    Supports connection pooling and multi-tenant data isolation
    with standard SQL and JSON support for flexible metadata.
    """

    def __init__(self):
        self.pool: Optional[ThreadedConnectionPool] = None
        self.config: Dict[str, Any] = {}
        self.logger = get_system_logger()

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize PostgreSQL connection with fail-fast validation."""
        log_component_start(
            "postgresql_provider", "initialize",
            config_keys=list(config.keys())
        )

        # Extract PostgreSQL-specific config
        postgresql_config = config.get("postgresql", {})
        if not postgresql_config:
            raise DatabaseError("Missing PostgreSQL configuration section")

        # Fail-fast validation - CLAUDE.md compliant
        if "host" not in postgresql_config:
            raise DatabaseError("Missing required config: postgresql.host")
        if "database" not in postgresql_config:
            raise DatabaseError("Missing required config: postgresql.database")
        if "user" not in postgresql_config:
            raise DatabaseError("Missing required config: postgresql.user")
        if "password" not in postgresql_config:
            raise DatabaseError("Missing required config: postgresql.password")

        self.config = postgresql_config

        # Create connection pool
        try:
            self.pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.config["host"],
                port=self.config.get("port", 5432),
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
                cursor_factory=RealDictCursor
            )

            # Test connection
            if self.pool is None:
                raise DatabaseError("Connection pool not initialized")


            conn = self.pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()
                    self.logger.info("postgresql_provider", "initialize", f"PostgreSQL connected: {version['version']}")
            finally:
                self.pool.putconn(conn)

            # Create schema if needed
            await self._ensure_schema()

        except Exception as e:
            log_error_context(
                "postgresql_provider", "initialize", e, {"connection_config": "hidden"}
            )
            raise DatabaseError(f"Failed to initialize PostgreSQL: {str(e)}")

        log_component_end("postgresql_provider", "initialize", "PostgreSQL initialization completed successfully")

    async def _ensure_schema(self) -> None:
        """Create database tables if they don't exist."""
        schema_sql = """
        -- Conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            conversation_id VARCHAR(255) UNIQUE NOT NULL,
            tenant_slug VARCHAR(100) NOT NULL,
            user_id VARCHAR(100) NOT NULL,
            title VARCHAR(500),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}'::jsonb
        );

        -- Messages table
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            message_id VARCHAR(255) UNIQUE NOT NULL,
            conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'::jsonb
        );

        -- Documents table
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            document_id VARCHAR(255) UNIQUE NOT NULL,
            tenant_slug VARCHAR(100) NOT NULL,
            user_id VARCHAR(100),
            title VARCHAR(500),
            content TEXT,
            file_path VARCHAR(1000),
            file_type VARCHAR(50),
            language VARCHAR(10),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'::jsonb
        );

        -- Chunks table
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            chunk_id VARCHAR(255) UNIQUE NOT NULL,
            document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            metadata JSONB DEFAULT '{}'::jsonb
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_conversations_tenant_user ON conversations(tenant_slug, user_id);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_documents_tenant_user ON documents(tenant_slug, user_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        """

        if self.pool is None:
            raise DatabaseError("Connection pool not initialized")



        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(schema_sql)
                conn.commit()
                self.logger.info("postgresql_provider", "_ensure_schema", "PostgreSQL schema ensured")
        finally:
            self.pool.putconn(conn)

    async def store_conversation(self, conversation_id: str, tenant_slug: str,
                                user_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Store a new conversation."""
        log_component_start(
            "postgresql_provider", "store_conversation",
            conversation_id=conversation_id, tenant_slug=tenant_slug, user_id=user_id
        )

        if self.pool is None:
            raise DatabaseError("Connection pool not initialized")



        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO conversations (conversation_id, tenant_slug, user_id, title)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (conversation_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                """, (conversation_id, tenant_slug, user_id, title))

                result = cursor.fetchone()
                conn.commit()

                log_component_end("postgresql_provider", "store_conversation", "conversation stored successfully", conversation_id=conversation_id)

                return dict(result)

        except Exception as e:
            conn.rollback()
            log_error_context(
                "postgresql_provider", "store_conversation", e,
                {"conversation_id": conversation_id, "tenant": tenant_slug}
            )
            raise DatabaseError(f"Failed to store conversation: {str(e)}")
        finally:
            self.pool.putconn(conn)

    async def store_message(self, message_id: str, conversation_id: str,
                           role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store a message in a conversation."""
        log_component_start(
            "postgresql_provider", "store_message",
            message_id=message_id, conversation_id=conversation_id, role=role
        )

        if self.pool is None:
            raise DatabaseError("Connection pool not initialized")

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO messages (message_id, conversation_id, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING *
                """, (message_id, conversation_id, role, content, json.dumps(metadata or {})))

                result = cursor.fetchone()
                conn.commit()

                log_component_end("postgresql_provider", "store_message", "operation completed", success=True, message_id=message_id)

                return dict(result)

        except Exception as e:
            conn.rollback()
            log_error_context("postgresql_provider", "store_message", e, {"error_details": "Error storing message: {str(e)}"})
            raise DatabaseError(f"Failed to store message: {str(e)}")
        finally:
            self.pool.putconn(conn)

    async def get_conversation_history(self, conversation_id: str,
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation message history."""
        log_component_start(
            "postgresql_provider", "get_conversation_history",
            conversation_id=conversation_id, limit=limit
        )

        if self.pool is None:
            raise DatabaseError("Connection pool not initialized")

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM messages
                    WHERE conversation_id = %s
                    ORDER BY timestamp ASC
                    LIMIT %s
                """, (conversation_id, limit))

                results = [dict(row) for row in cursor.fetchall()]

                log_component_end("postgresql_provider", "get_conversation_history", "operation completed", success=True, message_count=len(results))

                return results

        except Exception as e:
            log_error_context("postgresql_provider", "get_conversation_history", e, {"error_details": "Error retrieving history: {str(e)}"})
            raise DatabaseError(f"Failed to get conversation history: {str(e)}")
        finally:
            self.pool.putconn(conn)

    async def store_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Store a document with metadata."""
        log_component_start(
            "postgresql_provider", "store_document",
            document_id=document.get("document_id")
        )

        if self.pool is None:
            raise DatabaseError("Connection pool not initialized")

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO documents
                    (document_id, tenant_slug, user_id, title, content, file_path, file_type, language, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata
                    RETURNING *
                """, (
                    document["document_id"],
                    document["tenant_slug"],
                    document.get("user_id"),
                    document.get("title"),
                    document.get("content"),
                    document.get("file_path"),
                    document.get("file_type"),
                    document.get("language"),
                    json.dumps(document.get("metadata", {}))
                ))

                result = cursor.fetchone()
                conn.commit()

                log_component_end("postgresql_provider", "store_document", "operation completed", success=True, document_id=document["document_id"])

                return dict(result)

        except Exception as e:
            conn.rollback()
            log_error_context("postgresql_provider", "store_document", e, {"error_details": "Error storing document: {str(e)}"})
            raise DatabaseError(f"Failed to store document: {str(e)}")
        finally:
            self.pool.putconn(conn)

    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results."""
        log_component_start(
            "postgresql_provider", "execute_query",
            query_preview=query[:100] + "..." if len(query) > 100 else query
        )

        if self.pool is None:
            raise DatabaseError("Connection pool not initialized")

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Convert $1, $2, etc. to %s for psycopg2
                psycopg2_query = query
                if params:
                    # Replace PostgreSQL positional parameters with psycopg2 format
                    for i in range(len(params)):
                        psycopg2_query = psycopg2_query.replace(f'${i+1}', '%s')
                    cursor.execute(psycopg2_query, params)
                else:
                    cursor.execute(psycopg2_query)

                # For SELECT queries, fetch results
                if query.strip().upper().startswith('SELECT'):
                    results = [dict(row) for row in cursor.fetchall()]
                    log_component_end("postgresql_provider", "execute_query", f"Query executed successfully, {len(results)} rows returned")
                    return results
                else:
                    # For INSERT/UPDATE/DELETE, commit and return empty list
                    conn.commit()
                    log_component_end("postgresql_provider", "execute_query", "Query executed successfully")
                    return []
        except Exception as e:
            conn.rollback()
            log_error_context(
                "postgresql_provider", "execute_query", e,
                {"query": query[:100], "params": params}
            )
            raise DatabaseError(f"Failed to execute query: {str(e)}")
        finally:
            self.pool.putconn(conn)

    async def fetch_one(self, query: str, params: Optional[list] = None) -> Optional[Dict[str, Any]]:
        """Fetch single row from query (SurrealDB compatibility method)."""
        log_component_start(
            "postgresql_provider", "fetch_one",
            query_preview=query[:100] + "..." if len(query) > 100 else query
        )

        if self.pool is None:
            raise DatabaseError("Connection pool not initialized")

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Convert $1, $2, etc. to %s for psycopg2
                psycopg2_query = query
                if params:
                    # Replace PostgreSQL positional parameters with psycopg2 format
                    for i in range(len(params)):
                        psycopg2_query = psycopg2_query.replace(f'${i+1}', '%s')
                    cursor.execute(psycopg2_query, params)
                else:
                    cursor.execute(psycopg2_query)

                result = cursor.fetchone()
                if result:
                    result_dict = dict(result)
                    log_component_end("postgresql_provider", "fetch_one", "Row fetched successfully")
                    return result_dict
                else:
                    log_component_end("postgresql_provider", "fetch_one", "No row found")
                    return None

        except Exception as e:
            log_error_context(
                "postgresql_provider", "fetch_one", e,
                {"query": query[:100], "params": params}
            )
            raise DatabaseError(f"Failed to fetch row: {str(e)}")
        finally:
            self.pool.putconn(conn)

    async def fetch_all(self, query: str, params: Optional[list] = None) -> List[Dict[str, Any]]:
        """Fetch all rows from query (SurrealDB compatibility method)."""
        log_component_start(
            "postgresql_provider", "fetch_all",
            query_preview=query[:100] + "..." if len(query) > 100 else query
        )

        if self.pool is None:
            raise DatabaseError("Connection pool not initialized")

        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Convert $1, $2, etc. to %s for psycopg2
                psycopg2_query = query
                if params:
                    # Replace PostgreSQL positional parameters with psycopg2 format
                    for i in range(len(params)):
                        psycopg2_query = psycopg2_query.replace(f'${i+1}', '%s')
                    cursor.execute(psycopg2_query, params)
                else:
                    cursor.execute(psycopg2_query)

                results = cursor.fetchall()
                result_dicts = [dict(row) for row in results]

                log_component_end("postgresql_provider", "fetch_all", f"Fetched {len(result_dicts)} rows successfully")
                return result_dicts

        except Exception as e:
            log_error_context(
                "postgresql_provider", "fetch_all", e,
                {"query": query[:100], "params": params}
            )
            raise DatabaseError(f"Failed to fetch rows: {str(e)}")
        finally:
            self.pool.putconn(conn)

    async def close(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            self.pool.closeall()
            self.pool = None
            self.logger.info("postgresql_provider", "close", "PostgreSQL connection pool closed")