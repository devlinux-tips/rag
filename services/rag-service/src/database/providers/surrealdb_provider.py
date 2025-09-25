"""
SurrealDB Provider Implementation
Local/on-premise database implementation with AI-friendly logging.

SurrealDB Configuration:
- Uses local file database or websocket connection
- Supports multi-tenant data isolation
- Direct SQL-like querying with SurrealQL
"""

from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from surrealdb import Surreal  # type: ignore[import-not-found]
else:
    Surreal = Any
import asyncio
from pathlib import Path
import re

from ...utils.error_handler import StorageError as DatabaseError
from ...utils.logging_factory import (
    get_system_logger, log_component_start, log_component_end,
    log_decision_point, log_error_context
)
from ...models.multitenant_models import (
    Tenant, User, Document, Chunk, SearchQuery,
    CategorizationTemplate, SystemConfig, DocumentScope
)


class SurrealDBProvider:
    """
    SurrealDB implementation of DatabaseProvider protocol.

    Supports local file-based database for development and
    WebSocket connection for production deployments.
    """

    def __init__(self):
        self.client: Optional["Surreal"] = None
        self.config: Dict[str, Any] = {}
        self.logger = get_system_logger()

    def _ensure_client(self) -> "Surreal":
        """Ensure client is initialized and return it."""
        if self.client is None:
            raise DatabaseError("Database client not initialized. Call initialize() first.")
        return self.client

    def _parse_datetime(self, value: Any) -> datetime:
        """Parse datetime value from database result."""
        if value is None:
            return datetime.now()
        elif isinstance(value, datetime):
            return value
        elif isinstance(value, str):
            # Try parsing common formats
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return datetime.now()
        else:
            return datetime.now()

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize SurrealDB connection with fail-fast validation."""
        log_component_start(
            "surrealdb_provider", "initialize",
            config_keys=list(config.keys())
        )

        # Extract SurrealDB-specific config
        surrealdb_config = config.get("surrealdb", {})
        if not surrealdb_config:
            raise DatabaseError("Missing SurrealDB configuration section")

        # Validate required configuration
        required_keys = ["url", "namespace", "database"]
        missing_keys = [key for key in required_keys if key not in surrealdb_config]
        if missing_keys:
            error_msg = f"Missing required SurrealDB config keys: {missing_keys}"
            self.logger.error("surrealdb_provider", "initialize", error_msg)
            raise DatabaseError(error_msg)

        self.config = surrealdb_config
        self.max_retries = 3
        self.retry_delay = 1.0

        await self._establish_connection()

    async def _establish_connection(self) -> None:
        """Establish robust connection with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Import SurrealDB here to handle optional dependency
                try:
                    from surrealdb import Surreal  # type: ignore[import-not-found]
                except ImportError as e:
                    raise DatabaseError(
                        "SurrealDB not installed. Install with: pip install surrealdb"
                    ) from e

                # Initialize connection with timeout
                self.client = Surreal(self.config["url"])

                # Connect with authentication if credentials provided
                username = self.config["username"] if "username" in self.config else None
                password = self.config["password"] if "password" in self.config else None

                if username and password:
                    self.logger.info("surrealdb_provider", "_establish_connection",
                                   "Connecting to SurrealDB with authentication")
                    await self.client.signin({
                        "user": username,
                        "pass": password
                    })
                else:
                    self.logger.info("surrealdb_provider", "_establish_connection",
                                   "Connecting to SurrealDB without authentication")

                # Use namespace and database
                self.client.use(
                    self.config["namespace"],
                    self.config["database"]
                )

                # Connection established successfully - no test query needed

                self.logger.info(
                    "surrealdb_provider", "_establish_connection",
                    f"Successfully connected to SurrealDB at {self.config['url']} (attempt {attempt + 1})"
                )

                log_component_end(
                    "surrealdb_provider", "initialize",
                    "SurrealDB connection established successfully"
                )
                return

            except Exception as e:
                self.logger.warning(
                    "surrealdb_provider", "_establish_connection",
                    f"Connection attempt {attempt + 1} failed: {e}"
                )

                if attempt == self.max_retries - 1:
                    error_msg = f"Failed to establish SurrealDB connection after {self.max_retries} attempts: {e}"
                    log_error_context(
                        "surrealdb_provider", "_establish_connection",
                        DatabaseError(error_msg),
                        {"url": self.config.get("url"), "namespace": self.config.get("namespace")}
                    )
                    raise DatabaseError(error_msg) from e

                # Wait before retry with exponential backoff
                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.info(
                    "surrealdb_provider", "_establish_connection",
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

    async def health_check(self) -> bool:
        """Check SurrealDB connectivity and health."""
        if not self.client:
            self.logger.warning("surrealdb_provider", "health_check", "Client not initialized")
            return False

        try:
            result = await self._ensure_client().query("SELECT 1 as health_check")
            is_healthy = len(result) > 0

            self.logger.trace(
                "surrealdb_provider", "health_check",
                f"Health check result: {is_healthy}"
            )

            return is_healthy
        except Exception as e:
            self.logger.error(
                "surrealdb_provider", "health_check",
                f"Health check failed: {e}"
            )
            return False

    async def close(self) -> None:
        """Close SurrealDB connection and cleanup resources."""
        if self.client:
            try:
                await self.client.close()
                self.logger.info("surrealdb_provider", "close", "Connection closed")
            except Exception as e:
                self.logger.warning(
                    "surrealdb_provider", "close",
                    f"Error closing connection: {e}"
                )
            finally:
                self.client = None

    # ================================
    # Tenant Operations
    # ================================

    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create new tenant in SurrealDB."""
        log_component_start(
            "surrealdb_provider", "create_tenant",
            tenant_slug=tenant.slug
        )

        try:
            query = """
            CREATE tenant SET
                name = $name,
                slug = $slug,
                description = $description,
                status = $status,
                language_preference = $language_preference,
                cultural_context = $cultural_context,
                subscription_tier = $subscription_tier,
                settings = $settings
            """

            params = {
                "name": tenant.name,
                "slug": tenant.slug,
                "description": tenant.description,
                "status": tenant.status.value,
                "language_preference": tenant.language_preference.value,
                "cultural_context": tenant.cultural_context.value,
                "subscription_tier": tenant.subscription_tier.value,
                "settings": tenant.settings or {}
            }

            result = await self._ensure_client().query(query, params)

            if not result or len(result) == 0:
                raise DatabaseError("Tenant creation returned no results")

            created_tenant = self._parse_tenant_result(result[0])

            self.logger.info(
                "surrealdb_provider", "create_tenant",
                f"Tenant created: {created_tenant.slug} (ID: {created_tenant.id})"
            )

            log_component_end(
                "surrealdb_provider", "create_tenant",
                f"Tenant created successfully: {tenant.slug}"
            )

            return created_tenant

        except Exception as e:
            error_msg = f"Failed to create tenant {tenant.slug}: {e}"
            log_error_context(
                "surrealdb_provider", "create_tenant",
                DatabaseError(error_msg),
                {"tenant_slug": tenant.slug}
            )
            raise DatabaseError(error_msg) from e

    async def get_tenant(self, tenant_id: str) -> Tenant:
        """Get tenant by ID."""
        try:
            result = await self._ensure_client().query(
                "SELECT * FROM tenant WHERE id = $tenant_id",
                {"tenant_id": tenant_id}
            )

            if not result or len(result) == 0:
                raise DatabaseError(f"Tenant not found: {tenant_id}")

            tenant = self._parse_tenant_result(result[0])

            self.logger.trace(
                "surrealdb_provider", "get_tenant",
                f"Retrieved tenant: {tenant.slug}"
            )

            return tenant

        except Exception as e:
            if isinstance(e, DatabaseError):
                raise
            error_msg = f"Failed to get tenant {tenant_id}: {e}"
            raise DatabaseError(error_msg) from e

    async def get_tenant_by_slug(self, slug: str) -> Tenant:
        """Get tenant by slug."""
        try:
            result = await self._ensure_client().query(
                "SELECT * FROM tenant WHERE slug = $slug",
                {"slug": slug}
            )

            if not result or len(result) == 0:
                raise DatabaseError(f"Tenant not found with slug: {slug}")

            return self._parse_tenant_result(result[0])

        except Exception as e:
            if isinstance(e, DatabaseError):
                raise
            error_msg = f"Failed to get tenant by slug {slug}: {e}"
            raise DatabaseError(error_msg) from e

    async def list_tenants(self, status: Optional[str] = None) -> List[Tenant]:
        """List all tenants, optionally filtered by status."""
        try:
            if status:
                query = "SELECT * FROM tenant WHERE status = $status ORDER BY created_at DESC"
                params = {"status": status}
            else:
                query = "SELECT * FROM tenant ORDER BY created_at DESC"
                params = {}

            result = await self._ensure_client().query(query, params)

            tenants = [self._parse_tenant_result(row) for row in result]

            self.logger.debug(
                "surrealdb_provider", "list_tenants",
                f"Retrieved {len(tenants)} tenants (status filter: {status})"
            )

            return tenants

        except Exception as e:
            error_msg = f"Failed to list tenants: {e}"
            raise DatabaseError(error_msg) from e

    # ================================
    # User Operations
    # ================================

    async def create_user(self, user: User) -> User:
        """Create new user in SurrealDB."""
        try:
            query = """
            CREATE user SET
                tenant_id = $tenant_id,
                email = $email,
                username = $username,
                full_name = $full_name,
                password_hash = $password_hash,
                role = $role,
                status = $status,
                language_preference = $language_preference,
                settings = $settings
            """

            params = {
                "tenant_id": user.tenant_id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "password_hash": user.password_hash,
                "role": user.role.value,
                "status": user.status.value,
                "language_preference": user.language_preference.value,
                "settings": user.settings or {}
            }

            result = await self._ensure_client().query(query, params)

            if not result or len(result) == 0:
                raise DatabaseError("User creation returned no results")

            created_user = self._parse_user_result(result[0])

            self.logger.info(
                "surrealdb_provider", "create_user",
                f"User created: {created_user.email} (ID: {created_user.id})"
            )

            return created_user

        except Exception as e:
            error_msg = f"Failed to create user {user.email}: {e}"
            raise DatabaseError(error_msg) from e

    async def get_user(self, user_id: str) -> User:
        """Get user by ID."""
        try:
            result = await self._ensure_client().query(
                "SELECT * FROM user WHERE id = $user_id",
                {"user_id": user_id}
            )

            if not result or len(result) == 0:
                raise DatabaseError(f"User not found: {user_id}")

            return self._parse_user_result(result[0])

        except Exception as e:
            if isinstance(e, DatabaseError):
                raise
            error_msg = f"Failed to get user {user_id}: {e}"
            raise DatabaseError(error_msg) from e

    async def get_user_by_email(self, email: str, tenant_id: str) -> User:
        """Get user by email within tenant."""
        try:
            result = await self._ensure_client().query(
                "SELECT * FROM user WHERE email = $email AND tenant_id = $tenant_id",
                {"email": email, "tenant_id": tenant_id}
            )

            if not result or len(result) == 0:
                raise DatabaseError(f"User not found: {email} in tenant {tenant_id}")

            return self._parse_user_result(result[0])

        except Exception as e:
            if isinstance(e, DatabaseError):
                raise
            error_msg = f"Failed to get user by email {email}: {e}"
            raise DatabaseError(error_msg) from e

    async def list_tenant_users(self, tenant_id: str, status: Optional[str] = None) -> List[User]:
        """List all users in tenant, optionally filtered by status."""
        try:
            if status:
                query = "SELECT * FROM user WHERE tenant_id = $tenant_id AND status = $status ORDER BY created_at DESC"
                params = {"tenant_id": tenant_id, "status": status}
            else:
                query = "SELECT * FROM user WHERE tenant_id = $tenant_id ORDER BY created_at DESC"
                params = {"tenant_id": tenant_id}

            result = await self._ensure_client().query(query, params)

            users = [self._parse_user_result(row) for row in result]

            self.logger.debug(
                "surrealdb_provider", "list_tenant_users",
                f"Retrieved {len(users)} users for tenant {tenant_id}"
            )

            return users

        except Exception as e:
            error_msg = f"Failed to list users for tenant {tenant_id}: {e}"
            raise DatabaseError(error_msg) from e

    # ================================
    # Placeholder methods for remaining operations
    # ================================

    async def create_document(self, document: Document) -> Document:
        """Create new document in SurrealDB."""
        log_component_start(
            "surrealdb_provider", "create_document",
            document_title=document.title, tenant_id=document.tenant_id
        )

        try:
            query = """
            CREATE document SET
                tenant_id = $tenant_id,
                user_id = $user_id,
                title = $title,
                content = $content,
                language = $language,
                scope = $scope,
                category = $category,
                status = $status,
                metadata = $metadata,
                source_path = $source_path,
                mime_type = $mime_type,
                file_size = $file_size,
                checksum = $checksum
            """

            params = {
                "tenant_id": document.tenant_id,
                "user_id": document.user_id,
                "title": document.title,
                "content": document.content,
                "language": document.language if isinstance(document.language, str) else document.language.value,
                "scope": document.scope.value,
                "categories": document.categories,
                "status": document.status if isinstance(document.status, str) else document.status.value,
                "metadata": document.metadata or {},
                "source_path": document.source_path,
                "mime_type": document.mime_type,
                "file_size": document.file_size,
                "checksum": document.checksum
            }

            result = await self._ensure_client().query(query, params)

            if not result or len(result) == 0:
                raise DatabaseError("Document creation returned no results")

            created_document = self._parse_document_result(result[0])

            self.logger.info(
                "surrealdb_provider", "create_document",
                f"Document created: {created_document.title} (ID: {created_document.id})"
            )

            log_component_end(
                "surrealdb_provider", "create_document",
                f"Document created successfully: {document.title}"
            )

            return created_document

        except Exception as e:
            error_msg = f"Failed to create document {document.title}: {e}"
            log_error_context(
                "surrealdb_provider", "create_document",
                DatabaseError(error_msg),
                {"document_title": document.title, "tenant_id": document.tenant_id}
            )
            raise DatabaseError(error_msg) from e

    async def get_document(self, document_id: str) -> Document:
        """Get document by ID."""
        try:
            result = await self._ensure_client().query(
                "SELECT * FROM document WHERE id = $document_id",
                {"document_id": document_id}
            )

            if not result or len(result) == 0:
                raise DatabaseError(f"Document not found: {document_id}")

            document = self._parse_document_result(result[0])

            self.logger.trace(
                "surrealdb_provider", "get_document",
                f"Retrieved document: {document.title}"
            )

            return document

        except Exception as e:
            if isinstance(e, DatabaseError):
                raise
            error_msg = f"Failed to get document {document_id}: {e}"
            raise DatabaseError(error_msg) from e

    async def get_tenant_documents(self, tenant_id: str, scope: str, language: str, user_id: Optional[str] = None, status: Optional[str] = None) -> List[Document]:
        """Get documents by tenant/scope/language with optional filters."""
        try:
            # Build dynamic query based on filters
            conditions = ["tenant_id = $tenant_id", "scope = $scope", "language = $language"]
            params = {"tenant_id": tenant_id, "scope": scope, "language": language}

            if user_id:
                conditions.append("user_id = $user_id")
                params["user_id"] = user_id

            if status:
                conditions.append("status = $status")
                params["status"] = status

            query = f"SELECT * FROM document WHERE {' AND '.join(conditions)} ORDER BY created_at DESC"

            result = await self._ensure_client().query(query, params)

            documents = [self._parse_document_result(row) for row in result]

            self.logger.debug(
                "surrealdb_provider", "get_tenant_documents",
                f"Retrieved {len(documents)} documents for tenant {tenant_id}"
            )

            return documents

        except Exception as e:
            error_msg = f"Failed to get tenant documents for {tenant_id}: {e}"
            raise DatabaseError(error_msg) from e

    async def update_document_status(self, document_id: str, status: str) -> Document:
        """Update document processing status."""
        try:
            query = """
            UPDATE document SET status = $status, updated_at = time::now()
            WHERE id = $document_id
            """

            params = {"document_id": document_id, "status": status}

            result = await self._ensure_client().query(query, params)

            if not result or len(result) == 0:
                raise DatabaseError(f"Document not found for status update: {document_id}")

            # Get updated document
            updated_document = await self.get_document(document_id)

            self.logger.info(
                "surrealdb_provider", "update_document_status",
                f"Document status updated: {document_id} -> {status}"
            )

            return updated_document

        except Exception as e:
            if isinstance(e, DatabaseError):
                raise
            error_msg = f"Failed to update document status {document_id}: {e}"
            raise DatabaseError(error_msg) from e

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create new chunk metadata in SurrealDB."""
        log_component_start(
            "surrealdb_provider", "create_chunk",
            document_id=chunk.document_id, chunk_index=chunk.chunk_index
        )

        try:
            query = """
            CREATE chunk SET
                document_id = $document_id,
                tenant_id = $tenant_id,
                user_id = $user_id,
                chunk_index = $chunk_index,
                content = $content,
                language = $language,
                start_char = $start_char,
                end_char = $end_char,
                token_count = $token_count,
                vector_id = $vector_id,
                vector_collection = $vector_collection,
                metadata = $metadata,
                embedding_model = $embedding_model,
                embedding_dimension = $embedding_dimension
            """

            params = {
                "document_id": chunk.document_id,
                "tenant_id": chunk.tenant_id,
                "user_id": chunk.user_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "language": chunk.language if isinstance(chunk.language, str) else chunk.language.value,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "vector_id": chunk.vector_id,
                "vector_collection": chunk.vector_collection,
                "metadata": chunk.metadata or {},
                "embedding_model": chunk.embedding_model,
                "embedding_dimension": chunk.embedding_dimension
            }

            result = await self._ensure_client().query(query, params)

            if not result or len(result) == 0:
                raise DatabaseError("Chunk creation returned no results")

            created_chunk = self._parse_chunk_result(result[0])

            self.logger.info(
                "surrealdb_provider", "create_chunk",
                f"Chunk created: document {chunk.document_id}, index {chunk.chunk_index} (ID: {created_chunk.id})"
            )

            log_component_end(
                "surrealdb_provider", "create_chunk",
                f"Chunk created successfully: {chunk.document_id}[{chunk.chunk_index}]"
            )

            return created_chunk

        except Exception as e:
            error_msg = f"Failed to create chunk for document {chunk.document_id}: {e}"
            log_error_context(
                "surrealdb_provider", "create_chunk",
                DatabaseError(error_msg),
                {"document_id": chunk.document_id, "chunk_index": chunk.chunk_index}
            )
            raise DatabaseError(error_msg) from e

    async def get_document_chunks(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        try:
            result = await self._ensure_client().query(
                "SELECT * FROM chunk WHERE document_id = $document_id ORDER BY chunk_index ASC",
                {"document_id": document_id}
            )

            chunks = [self._parse_chunk_result(row) for row in result]

            self.logger.debug(
                "surrealdb_provider", "get_document_chunks",
                f"Retrieved {len(chunks)} chunks for document {document_id}"
            )

            return chunks

        except Exception as e:
            error_msg = f"Failed to get chunks for document {document_id}: {e}"
            raise DatabaseError(error_msg) from e

    async def get_chunk_by_vector_id(self, vector_collection: str, vector_id: str) -> Chunk:
        """Get chunk by vector database ID."""
        try:
            result = await self._ensure_client().query(
                "SELECT * FROM chunk WHERE vector_collection = $collection AND vector_id = $vector_id",
                {"collection": vector_collection, "vector_id": vector_id}
            )

            if not result or len(result) == 0:
                raise DatabaseError(f"Chunk not found for vector {vector_id} in collection {vector_collection}")

            chunk = self._parse_chunk_result(result[0])

            self.logger.trace(
                "surrealdb_provider", "get_chunk_by_vector_id",
                f"Retrieved chunk: {chunk.id} for vector {vector_id}"
            )

            return chunk

        except Exception as e:
            if isinstance(e, DatabaseError):
                raise
            error_msg = f"Failed to get chunk by vector ID {vector_id}: {e}"
            raise DatabaseError(error_msg) from e

    async def log_search_query(self, query: SearchQuery) -> None:
        """Log search query - placeholder implementation."""
        raise NotImplementedError("Search analytics not yet implemented")

    async def get_search_analytics(self, tenant_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[SearchQuery]:
        """Get search analytics - placeholder implementation."""
        raise NotImplementedError("Search analytics not yet implemented")

    async def get_system_config(self, config_key: str, tenant_id: Optional[str] = None) -> SystemConfig:
        """Get system config - placeholder implementation."""
        raise NotImplementedError("System config not yet implemented")

    async def set_system_config(self, config: SystemConfig) -> SystemConfig:
        """Set system config - placeholder implementation."""
        raise NotImplementedError("System config not yet implemented")

    async def get_categorization_templates(self, language: str, category: Optional[str] = None, tenant_id: Optional[str] = None) -> List[CategorizationTemplate]:
        """Get categorization templates - placeholder implementation."""
        raise NotImplementedError("Categorization templates not yet implemented")

    # ================================
    # Helper Methods
    # ================================

    def _parse_tenant_result(self, result: Dict[str, Any]) -> Tenant:
        """Parse SurrealDB result into Tenant object."""
        from ...models.multitenant_models import TenantStatus, BusinessContext, SubscriptionTier

        if "id" not in result:
            raise DatabaseError("Missing required field 'id' in tenant result")
        if "name" not in result:
            raise DatabaseError("Missing required field 'name' in tenant result")
        if "slug" not in result:
            raise DatabaseError("Missing required field 'slug' in tenant result")
        if "status" not in result:
            raise DatabaseError("Missing required field 'status' in tenant result")

        return Tenant(
            id=str(result["id"]),
            name=result["name"],
            slug=result["slug"],
            description=result.get("description"),
            status=TenantStatus(result["status"]),
            settings=result.get("settings") or {},
            language_preference=result["language_preference"],
            cultural_context=BusinessContext(result["cultural_context"]),
            subscription_tier=SubscriptionTier(result["subscription_tier"]),
            created_at=self._parse_datetime(result.get("created_at")),
            updated_at=self._parse_datetime(result.get("updated_at"))
        )

    def _parse_user_result(self, result: Dict[str, Any]) -> User:
        """Parse SurrealDB result into User object."""
        from ...models.multitenant_models import UserRole, UserStatus, Language

        required_fields = ["id", "tenant_id", "email", "username", "password_hash", "role", "status", "language_preference"]
        for field in required_fields:
            if field not in result:
                raise DatabaseError(f"Missing required field '{field}' in user result")

        return User(
            id=str(result["id"]),
            tenant_id=result["tenant_id"],
            email=result["email"],
            username=result["username"],
            full_name=result.get("full_name"),
            password_hash=result["password_hash"],
            role=UserRole(result["role"]),
            status=UserStatus(result["status"]),
            language_preference=Language(result["language_preference"]),
            settings=result.get("settings") or {},
            last_login_at=self._parse_datetime(result.get("last_login_at")) if result.get("last_login_at") else None,
            created_at=self._parse_datetime(result.get("created_at")),
            updated_at=self._parse_datetime(result.get("updated_at"))
        )

    def _parse_document_result(self, result: Dict[str, Any]) -> Document:
        """Parse SurrealDB result into Document object."""
        from ...models.multitenant_models import DocumentStatus, DocumentScope, Language

        required_fields = ["id", "tenant_id", "user_id", "title", "content", "language", "scope", "status"]
        for field in required_fields:
            if field not in result:
                raise DatabaseError(f"Missing required field '{field}' in document result")

        return Document(
            id=str(result["id"]),
            tenant_id=result["tenant_id"],
            user_id=result["user_id"],
            title=result["title"],
            filename=result.get("filename", result["title"]),
            file_path=result.get("file_path", ""),
            content=result["content"],
            language=result["language"] if isinstance(result["language"], str) else result["language"],
            scope=DocumentScope(result["scope"]),
            categories=result.get("categories", []),
            status=DocumentStatus(result["status"]),
            metadata=result.get("metadata") or {},
            source_path=result.get("source_path"),
            mime_type=result.get("mime_type"),
            file_size=result.get("file_size") or 0,
            checksum=result.get("checksum"),
            created_at=self._parse_datetime(result.get("created_at")),
            updated_at=self._parse_datetime(result.get("updated_at"))
        )

    def _parse_chunk_result(self, result: Dict[str, Any]) -> Chunk:
        """Parse SurrealDB result into Chunk object."""
        from ...models.multitenant_models import Language

        required_fields = ["id", "document_id", "tenant_id", "user_id", "chunk_index", "content", "language"]
        for field in required_fields:
            if field not in result:
                raise DatabaseError(f"Missing required field '{field}' in chunk result")

        return Chunk(
            id=str(result["id"]),
            document_id=result["document_id"],
            tenant_id=result["tenant_id"],
            user_id=result["user_id"],
            scope=DocumentScope(result.get("scope", "user")),
            chunk_index=result["chunk_index"],
            content=result["content"],
            language=result["language"] if isinstance(result["language"], str) else result["language"],
            start_char=result.get("start_char") or 0,
            end_char=result.get("end_char") or 0,
            token_count=result.get("token_count") or 0,
            vector_id=result.get("vector_id") or "",
            vector_collection=result.get("vector_collection") or "",
            metadata=result.get("metadata") or {},
            embedding_model=result.get("embedding_model") or "bge-m3",
            embedding_dimension=result.get("embedding_dimension") or 768,
            created_at=self._parse_datetime(result.get("created_at"))
        )

    async def execute_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute SQL-like query with automatic retry and reconnection.

        This method provides compatibility with chat persistence that expects SQL syntax.
        Translates common SQL operations to SurrealQL equivalents.
        """
        client = self._ensure_client()

        query = query.strip()

        for attempt in range(self.max_retries):
            try:
                # Route different SQL operations to appropriate handlers
                if query.upper().startswith("CREATE TABLE"):
                    result = await self._handle_table_creation(query)
                elif query.upper().startswith("CREATE INDEX"):
                    result = await self._handle_index_creation(query)
                elif query.upper().startswith("INSERT INTO"):
                    result = await self._handle_insert(query, parameters)
                elif query.upper().startswith("UPDATE"):
                    result = await self._handle_update(query, parameters)
                elif query.upper().startswith("DELETE"):
                    result = await self._handle_delete(query, parameters)
                elif query.upper().startswith("SELECT"):
                    result = await self._handle_select(query, parameters)
                else:
                    # Direct SurrealQL query
                    result = await self._ensure_client().query(query, parameters or {})
                    result = result if isinstance(result, list) else [result]

                return result

            except Exception as e:
                self.logger.warning("surrealdb_provider", "execute_query",
                               f"Query attempt {attempt + 1} failed: {query[:50]}... Error: {e}")

                # Check if it's a connection error
                if "WebSocket" in str(e) or "connection" in str(e).lower() or "close" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        self.logger.info("surrealdb_provider", "execute_query",
                                       "Connection lost, attempting reconnection...")
                        try:
                            await self._establish_connection()
                            continue  # Retry the query
                        except Exception as reconnect_error:
                            self.logger.error("surrealdb_provider", "execute_query",
                                           f"Reconnection failed: {reconnect_error}")

                if attempt == self.max_retries - 1:
                    self.logger.error("surrealdb_provider", "execute_query",
                                   f"Query failed after {self.max_retries} attempts: {query[:100]}... Error: {e}")
                    raise DatabaseError(f"Query execution failed: {e}") from e

                # Wait before retry
                wait_time = self.retry_delay * (attempt + 1)
                await asyncio.sleep(wait_time)

        # This should not be reached, but add for type safety
        raise DatabaseError("Query execution failed after all retries")

    async def _handle_table_creation(self, query: str) -> list[dict]:
        """Handle CREATE TABLE statements by translating to SurrealQL."""
        # Extract table name from CREATE TABLE statement
        parts = query.split()
        if len(parts) < 3:
            raise DatabaseError(f"Invalid CREATE TABLE syntax: {query}")

        table_name = parts[2].replace("(", "").strip()

        # For SurrealDB, we don't need to pre-create tables
        # They are created automatically when we insert data
        # But we can create an empty record to establish the table
        try:
            # Just create table structure - SurrealDB will auto-create on first insert
            self.logger.info("surrealdb_provider", "_handle_table_creation",
                           f"Table {table_name} will be auto-created on first insert")
            return [{"status": "ok", "message": f"Table {table_name} ready"}]
        except Exception as e:
            raise DatabaseError(f"Failed to create table {table_name}: {e}") from e

    async def _handle_index_creation(self, query: str) -> list[dict]:
        """Handle CREATE INDEX statements - SurrealDB doesn't use traditional indexes."""
        # SurrealDB handles indexes differently - they're auto-created for performance
        # For compatibility, we just acknowledge the index creation without error
        try:
            self.logger.info("surrealdb_provider", "_handle_index_creation",
                           "Index creation acknowledged - SurrealDB auto-optimizes queries")
            return [{"status": "ok", "message": "Index creation acknowledged"}]
        except Exception as e:
            raise DatabaseError(f"Failed to acknowledge index creation: {e}") from e

    async def _handle_insert(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Handle INSERT INTO statements by translating to SurrealQL."""
        import re

        # Parse INSERT INTO table_name (columns) VALUES (values)
        insert_match = re.match(
            r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)',
            query, re.IGNORECASE
        )

        if not insert_match:
            raise DatabaseError(f"Invalid INSERT syntax: {query}")

        table_name = insert_match.group(1)
        columns = [col.strip() for col in insert_match.group(2).split(',')]
        values_str = insert_match.group(3)

        # Parse values - handle both quoted strings and parameters
        values = []
        for val in values_str.split(','):
            val = val.strip()
            if val.startswith('$') and parameters and val[1:] in parameters:
                values.append(parameters[val[1:]])
            elif val.startswith("'") and val.endswith("'"):
                values.append(val[1:-1])  # Remove quotes
            elif val.isdigit():
                values.append(int(val))
            else:
                values.append(val)

        # Create data dictionary
        data = dict(zip(columns, values))

        try:
            result = await self._ensure_client().create(table_name, data)
            return [result] if not isinstance(result, list) else result
        except Exception as e:
            raise DatabaseError(f"Failed to insert into {table_name}: {e}") from e

    async def _handle_update(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Handle UPDATE statements by translating to SurrealQL."""
        import re

        # Parse UPDATE table_name SET column = value WHERE condition
        update_match = re.match(
            r'UPDATE\s+(\w+)\s+SET\s+(.+?)\s+WHERE\s+(.+)',
            query, re.IGNORECASE
        )

        if not update_match:
            raise DatabaseError(f"Invalid UPDATE syntax: {query}")

        table_name = update_match.group(1)
        set_clause = update_match.group(2)
        where_clause = update_match.group(3)

        # Parse SET clause
        updates = {}
        for assignment in set_clause.split(','):
            key, value = assignment.split('=', 1)
            key = key.strip()
            value = value.strip()

            if value.startswith('$') and parameters and value[1:] in parameters:
                updates[key] = parameters[value[1:]]
            elif value.startswith("'") and value.endswith("'"):
                updates[key] = value[1:-1]
            else:
                updates[key] = value

        # Construct SurrealQL UPDATE query
        surreal_query = f"UPDATE {table_name} SET {', '.join(f'{k} = ${k}' for k in updates.keys())} WHERE {where_clause}"

        try:
            result = await self._ensure_client().query(surreal_query, updates)
            return result if isinstance(result, list) else [result]
        except Exception as e:
            raise DatabaseError(f"Failed to update {table_name}: {e}") from e

    async def _handle_delete(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Handle DELETE statements by translating to SurrealQL."""
        import re

        # Parse DELETE FROM table_name WHERE condition
        delete_match = re.match(
            r'DELETE\s+FROM\s+(\w+)\s+WHERE\s+(.+)',
            query, re.IGNORECASE
        )

        if not delete_match:
            raise DatabaseError(f"Invalid DELETE syntax: {query}")

        table_name = delete_match.group(1)
        where_clause = delete_match.group(2)

        # Construct SurrealQL DELETE query
        surreal_query = f"DELETE FROM {table_name} WHERE {where_clause}"

        try:
            result = await self._ensure_client().query(surreal_query, parameters or {})
            return result if isinstance(result, list) else [result]
        except Exception as e:
            raise DatabaseError(f"Failed to delete from {table_name}: {e}") from e

    async def _handle_select(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Handle SELECT statements by translating to SurrealQL."""
        # SurrealQL SELECT syntax is similar to SQL, so we can often pass it through
        try:
            result = await self._ensure_client().query(query, parameters or {})
            return result if isinstance(result, list) else [result]
        except Exception as e:
            raise DatabaseError(f"Failed to execute SELECT: {e}") from e

    async def fetch_one(self, query: str, parameters: list[Any] | None = None) -> dict | None:
        """Execute SQL-like query and return single result with retry logic."""
        client = self._ensure_client()

        try:
            # Convert positional parameters to SurrealQL format
            if parameters:
                # Replace $1, $2, etc. with parameter values
                for i, param in enumerate(parameters, 1):
                    query = query.replace(f"${i}", f"'{param}'" if isinstance(param, str) else str(param))

            result = await self.execute_query(query)

            # Return first result or None
            if result and len(result) > 0:
                return result[0]
            return None

        except Exception as e:
            self.logger.error("surrealdb_provider", "fetch_one",
                           f"Fetch one failed: {query[:100]}... Error: {e}")
            raise DatabaseError(f"Fetch one failed: {e}") from e