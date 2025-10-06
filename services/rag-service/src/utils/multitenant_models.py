"""
Multitenant data models for RAG system.

These dataclasses support multi-tenant, multi-user document scoping.
They are used for:
- Collection naming (tenant_user_language_documents)
- Access control context
- Document scope determination (user/tenant/feature)
"""

from dataclasses import dataclass
from enum import Enum


class DocumentScope(Enum):
    """Document visibility scope in multi-tenant system."""

    USER = "user"  # Personal documents for specific user
    TENANT = "tenant"  # Shared documents for all users in tenant
    FEATURE = "feature"  # Global feature datasets (e.g., narodne-novine)


@dataclass
class Tenant:
    """
    Tenant entity for multi-tenant RAG system.

    Examples:
        >>> tenant = Tenant(id="tenant:acme", name="ACME Corp", slug="acme")
        >>> tenant.slug
        'acme'
    """

    id: str  # Unique identifier (e.g., "tenant:acme")
    name: str  # Display name (e.g., "ACME Corporation")
    slug: str  # URL-friendly identifier (e.g., "acme")

    def __post_init__(self):
        """Validate required fields."""
        if not self.id:
            raise ValueError("Tenant.id cannot be empty")
        if not self.slug:
            raise ValueError("Tenant.slug cannot be empty")


@dataclass
class User:
    """
    User entity for multi-tenant RAG system.

    Examples:
        >>> user = User(
        ...     id="user:john",
        ...     tenant_id="tenant:acme",
        ...     email="john@acme.com",
        ...     username="john",
        ...     full_name="John Doe"
        ... )
        >>> user.username
        'john'
    """

    id: str  # Unique identifier (e.g., "user:john")
    tenant_id: str  # Parent tenant (e.g., "tenant:acme")
    email: str  # User email
    username: str  # Username
    full_name: str  # Display name

    def __post_init__(self):
        """Validate required fields."""
        if not self.id:
            raise ValueError("User.id cannot be empty")
        if not self.tenant_id:
            raise ValueError("User.tenant_id cannot be empty")
        if not self.username:
            raise ValueError("User.username cannot be empty")


@dataclass
class TenantUserContext:
    """
    Combined tenant + user context for scoped operations.

    Used by RAG CLI and pipeline for:
    - Determining correct collection name
    - Access control decisions
    - Document scoping
    """

    tenant: Tenant
    user: User
    scope: DocumentScope = DocumentScope.USER

    def get_collection_name(self, language: str) -> str:
        """
        Generate ChromaDB collection name based on scope.

        Args:
            language: Language code (e.g., "hr", "en")

        Returns:
            Collection name in format:
            - USER scope: "{tenant_slug}_{username}_{language}_documents"
            - TENANT scope: "{tenant_slug}_shared_{language}_documents"
            - FEATURE scope: Not handled here (use feature-specific naming)

        Examples:
            >>> ctx = TenantUserContext(
            ...     tenant=Tenant(id="tenant:dev", name="Dev", slug="development"),
            ...     user=User(id="user:john", tenant_id="tenant:dev", email="john@dev.com",
            ...               username="dev_user", full_name="John"),
            ...     scope=DocumentScope.USER
            ... )
            >>> ctx.get_collection_name("hr")
            'development_dev_user_hr_documents'
        """
        if self.scope == DocumentScope.USER:
            return f"{self.tenant.slug}_{self.user.username}_{language}_documents"
        elif self.scope == DocumentScope.TENANT:
            return f"{self.tenant.slug}_shared_{language}_documents"
        else:
            raise ValueError(f"Collection naming for scope {self.scope} must be handled separately")

    def __post_init__(self):
        """Validate context integrity."""
        if self.user.tenant_id != self.tenant.id:
            raise ValueError(f"User tenant_id ({self.user.tenant_id}) does not match Tenant id ({self.tenant.id})")
