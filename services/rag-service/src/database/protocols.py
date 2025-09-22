"""
Database Protocol Definitions for RAG System
Defines the contract for swappable database backends (SurrealDB/Supabase).

Following fail-fast philosophy with comprehensive type checking.
"""

from typing import Protocol, List, Optional, Dict, Any, runtime_checkable
from abc import abstractmethod

from ..models.multitenant_models import (
    Tenant, User, Document, Chunk, SearchQuery,
    CategorizationTemplate, SystemConfig, TenantUserContext
)


@runtime_checkable
class DatabaseProvider(Protocol):
    """
    Protocol for swappable database backends.

    Implementations:
    - SurrealDBProvider: Local/on-premise deployments
    - SupabaseProvider: Cloud/external deployments

    All methods MUST fail-fast with explicit exceptions.
    No silent fallbacks or .get() patterns allowed.
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize database connection with fail-fast validation.

        Args:
            config: Database configuration dictionary

        Raises:
            DatabaseError: If initialization fails or config invalid
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check database connectivity and health.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close database connection and cleanup resources."""
        ...

    # ================================
    # Tenant Operations
    # ================================

    @abstractmethod
    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """
        Create new tenant.

        Args:
            tenant: Tenant object to create

        Returns:
            Tenant: Created tenant with assigned ID

        Raises:
            DatabaseError: If tenant creation fails or tenant exists
        """
        ...

    @abstractmethod
    async def get_tenant(self, tenant_id: str) -> Tenant:
        """
        Get tenant by ID.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant: Tenant object

        Raises:
            DatabaseError: If tenant not found
        """
        ...

    @abstractmethod
    async def get_tenant_by_slug(self, slug: str) -> Tenant:
        """
        Get tenant by slug.

        Args:
            slug: Tenant slug

        Returns:
            Tenant: Tenant object

        Raises:
            DatabaseError: If tenant not found
        """
        ...

    @abstractmethod
    async def list_tenants(self, status: Optional[str] = None) -> List[Tenant]:
        """
        List all tenants, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List[Tenant]: List of tenants
        """
        ...

    # ================================
    # User Operations
    # ================================

    @abstractmethod
    async def create_user(self, user: User) -> User:
        """
        Create new user.

        Args:
            user: User object to create

        Returns:
            User: Created user with assigned ID

        Raises:
            DatabaseError: If user creation fails or user exists
        """
        ...

    @abstractmethod
    async def get_user(self, user_id: str) -> User:
        """
        Get user by ID.

        Args:
            user_id: User identifier

        Returns:
            User: User object

        Raises:
            DatabaseError: If user not found
        """
        ...

    @abstractmethod
    async def get_user_by_email(self, email: str, tenant_id: str) -> User:
        """
        Get user by email within tenant.

        Args:
            email: User email
            tenant_id: Tenant identifier

        Returns:
            User: User object

        Raises:
            DatabaseError: If user not found
        """
        ...

    @abstractmethod
    async def list_tenant_users(self, tenant_id: str, status: Optional[str] = None) -> List[User]:
        """
        List all users in tenant, optionally filtered by status.

        Args:
            tenant_id: Tenant identifier
            status: Optional status filter

        Returns:
            List[User]: List of users
        """
        ...

    # ================================
    # Document Operations
    # ================================

    @abstractmethod
    async def create_document(self, document: Document) -> Document:
        """
        Create new document.

        Args:
            document: Document object to create

        Returns:
            Document: Created document with assigned ID

        Raises:
            DatabaseError: If document creation fails
        """
        ...

    @abstractmethod
    async def get_document(self, document_id: str) -> Document:
        """
        Get document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document: Document object

        Raises:
            DatabaseError: If document not found
        """
        ...

    @abstractmethod
    async def get_tenant_documents(
        self,
        tenant_id: str,
        scope: str,
        language: str,
        user_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Document]:
        """
        Get documents by tenant/scope/language with optional filters.

        Args:
            tenant_id: Tenant identifier
            scope: Document scope ('user' or 'tenant')
            language: Language code
            user_id: Optional user filter (for 'user' scope)
            status: Optional status filter

        Returns:
            List[Document]: List of documents
        """
        ...

    @abstractmethod
    async def update_document_status(self, document_id: str, status: str) -> Document:
        """
        Update document processing status.

        Args:
            document_id: Document identifier
            status: New status

        Returns:
            Document: Updated document

        Raises:
            DatabaseError: If document not found or update fails
        """
        ...

    # ================================
    # Chunk Operations (Vector Metadata)
    # ================================

    @abstractmethod
    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """
        Create new chunk metadata.

        Args:
            chunk: Chunk object to create

        Returns:
            Chunk: Created chunk with assigned ID

        Raises:
            DatabaseError: If chunk creation fails
        """
        ...

    @abstractmethod
    async def get_document_chunks(self, document_id: str) -> List[Chunk]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            List[Chunk]: List of chunks
        """
        ...

    @abstractmethod
    async def get_chunk_by_vector_id(self, vector_collection: str, vector_id: str) -> Chunk:
        """
        Get chunk by vector database ID.

        Args:
            vector_collection: Vector collection name
            vector_id: Vector ID

        Returns:
            Chunk: Chunk object

        Raises:
            DatabaseError: If chunk not found
        """
        ...

    # ================================
    # Search Analytics
    # ================================

    @abstractmethod
    async def log_search_query(self, query: SearchQuery) -> None:
        """
        Log search query for analytics.

        Args:
            query: Search query to log

        Raises:
            DatabaseError: If logging fails
        """
        ...

    @abstractmethod
    async def get_search_analytics(
        self,
        tenant_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[SearchQuery]:
        """
        Get search analytics for tenant.

        Args:
            tenant_id: Tenant identifier
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List[SearchQuery]: List of search queries
        """
        ...

    # ================================
    # System Configuration
    # ================================

    @abstractmethod
    async def get_system_config(self, config_key: str, tenant_id: Optional[str] = None) -> SystemConfig:
        """
        Get system configuration value.

        Args:
            config_key: Configuration key
            tenant_id: Optional tenant-specific config

        Returns:
            SystemConfig: Configuration object

        Raises:
            DatabaseError: If config not found
        """
        ...

    @abstractmethod
    async def set_system_config(self, config: SystemConfig) -> SystemConfig:
        """
        Set system configuration value.

        Args:
            config: Configuration object to save

        Returns:
            SystemConfig: Saved configuration

        Raises:
            DatabaseError: If save fails
        """
        ...

    # ================================
    # Categorization Templates
    # ================================

    @abstractmethod
    async def get_categorization_templates(
        self,
        language: str,
        category: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[CategorizationTemplate]:
        """
        Get categorization templates for language/category.

        Args:
            language: Language code
            category: Optional category filter
            tenant_id: Optional tenant-specific templates

        Returns:
            List[CategorizationTemplate]: List of templates
        """
        ...