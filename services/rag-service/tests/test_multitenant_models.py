"""
Tests for simplified multitenant models.
Only tests the essential dataclasses used in production.
"""

import pytest
from src.utils.multitenant_models import (
    DocumentScope,
    Tenant,
    User,
    TenantUserContext,
)


class TestDocumentScope:
    """Test DocumentScope enum."""

    def test_scope_values(self):
        """Test that all scope values are defined."""
        assert DocumentScope.USER.value == "user"
        assert DocumentScope.TENANT.value == "tenant"
        assert DocumentScope.FEATURE.value == "feature"


class TestTenant:
    """Test Tenant dataclass."""

    def test_create_valid_tenant(self):
        """Test creating a valid tenant."""
        tenant = Tenant(id="tenant:acme", name="ACME Corp", slug="acme")
        assert tenant.id == "tenant:acme"
        assert tenant.name == "ACME Corp"
        assert tenant.slug == "acme"

    def test_tenant_empty_id_raises_error(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="Tenant.id cannot be empty"):
            Tenant(id="", name="Test", slug="test")

    def test_tenant_empty_slug_raises_error(self):
        """Test that empty slug raises ValueError."""
        with pytest.raises(ValueError, match="Tenant.slug cannot be empty"):
            Tenant(id="tenant:test", name="Test", slug="")


class TestUser:
    """Test User dataclass."""

    def test_create_valid_user(self):
        """Test creating a valid user."""
        user = User(
            id="user:john",
            tenant_id="tenant:acme",
            email="john@acme.com",
            username="john",
            full_name="John Doe",
        )
        assert user.id == "user:john"
        assert user.tenant_id == "tenant:acme"
        assert user.email == "john@acme.com"
        assert user.username == "john"
        assert user.full_name == "John Doe"

    def test_user_empty_id_raises_error(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="User.id cannot be empty"):
            User(
                id="",
                tenant_id="tenant:test",
                email="test@test.com",
                username="test",
                full_name="Test User",
            )

    def test_user_empty_tenant_id_raises_error(self):
        """Test that empty tenant_id raises ValueError."""
        with pytest.raises(ValueError, match="User.tenant_id cannot be empty"):
            User(
                id="user:test",
                tenant_id="",
                email="test@test.com",
                username="test",
                full_name="Test User",
            )

    def test_user_empty_username_raises_error(self):
        """Test that empty username raises ValueError."""
        with pytest.raises(ValueError, match="User.username cannot be empty"):
            User(
                id="user:test",
                tenant_id="tenant:test",
                email="test@test.com",
                username="",
                full_name="Test User",
            )


class TestTenantUserContext:
    """Test TenantUserContext dataclass."""

    def test_create_valid_context(self):
        """Test creating a valid context."""
        tenant = Tenant(id="tenant:dev", name="Dev", slug="development")
        user = User(
            id="user:john",
            tenant_id="tenant:dev",
            email="john@dev.com",
            username="dev_user",
            full_name="John Doe",
        )
        context = TenantUserContext(tenant=tenant, user=user, scope=DocumentScope.USER)

        assert context.tenant == tenant
        assert context.user == user
        assert context.scope == DocumentScope.USER

    def test_context_default_scope_is_user(self):
        """Test that default scope is USER."""
        tenant = Tenant(id="tenant:dev", name="Dev", slug="development")
        user = User(
            id="user:john",
            tenant_id="tenant:dev",
            email="john@dev.com",
            username="dev_user",
            full_name="John",
        )
        context = TenantUserContext(tenant=tenant, user=user)

        assert context.scope == DocumentScope.USER

    def test_context_mismatched_tenant_raises_error(self):
        """Test that mismatched tenant IDs raise ValueError."""
        tenant = Tenant(id="tenant:dev", name="Dev", slug="development")
        user = User(
            id="user:john",
            tenant_id="tenant:other",  # Mismatched!
            email="john@dev.com",
            username="dev_user",
            full_name="John",
        )

        with pytest.raises(ValueError, match="does not match"):
            TenantUserContext(tenant=tenant, user=user)

    def test_get_collection_name_user_scope(self):
        """Test collection naming for USER scope."""
        tenant = Tenant(id="tenant:dev", name="Dev", slug="development")
        user = User(
            id="user:john",
            tenant_id="tenant:dev",
            email="john@dev.com",
            username="dev_user",
            full_name="John",
        )
        context = TenantUserContext(tenant=tenant, user=user, scope=DocumentScope.USER)

        collection_name = context.get_collection_name("hr")
        assert collection_name == "development_dev_user_hr_documents"

    def test_get_collection_name_tenant_scope(self):
        """Test collection naming for TENANT scope."""
        tenant = Tenant(id="tenant:dev", name="Dev", slug="development")
        user = User(
            id="user:john",
            tenant_id="tenant:dev",
            email="john@dev.com",
            username="dev_user",
            full_name="John",
        )
        context = TenantUserContext(
            tenant=tenant, user=user, scope=DocumentScope.TENANT
        )

        collection_name = context.get_collection_name("hr")
        assert collection_name == "development_shared_hr_documents"

    def test_get_collection_name_feature_scope_raises_error(self):
        """Test that FEATURE scope raises ValueError (must be handled separately)."""
        tenant = Tenant(id="tenant:dev", name="Dev", slug="development")
        user = User(
            id="user:john",
            tenant_id="tenant:dev",
            email="john@dev.com",
            username="dev_user",
            full_name="John",
        )
        context = TenantUserContext(
            tenant=tenant, user=user, scope=DocumentScope.FEATURE
        )

        with pytest.raises(
            ValueError, match="must be handled separately"
        ):
            context.get_collection_name("hr")

    def test_get_collection_name_different_languages(self):
        """Test collection naming with different language codes."""
        tenant = Tenant(id="tenant:acme", name="ACME", slug="acme")
        user = User(
            id="user:bob",
            tenant_id="tenant:acme",
            email="bob@acme.com",
            username="bob",
            full_name="Bob Smith",
        )
        context = TenantUserContext(tenant=tenant, user=user, scope=DocumentScope.USER)

        assert context.get_collection_name("hr") == "acme_bob_hr_documents"
        assert context.get_collection_name("en") == "acme_bob_en_documents"
        assert context.get_collection_name("de") == "acme_bob_de_documents"
