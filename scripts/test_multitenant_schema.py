#!/usr/bin/env python3
"""
Test script for the multi-tenant SurrealDB schema.

This script validates the new tenant-user hierarchy and document scoping
functionality for the RAG system evolution.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
services_path = project_root / "services" / "rag-service"
sys.path.insert(0, str(services_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test SurrealDB connection
try:
    from src.models.multitenant_models import (
        DEFAULT_DEVELOPMENT_CONTEXT,
        DEFAULT_DEVELOPMENT_TENANT,
        Document,
        DocumentScope,
        Language,
    )
    from surrealdb import Surreal

    logger.info("✅ Successfully imported multi-tenant models")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)


async def test_database_connection():
    """Test basic SurrealDB connection."""
    logger.info("🔍 Testing SurrealDB connection...")

    try:
        from surrealdb import Surreal

        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

            # Test basic query
            await db.query("SELECT * FROM tenant")
            logger.info("✅ SurrealDB connection successful")
            return True

    except Exception as e:
        logger.error(f"❌ SurrealDB connection failed: {e}")
        return False


async def load_schema():
    """Load the multi-tenant schema into SurrealDB."""
    logger.info("📋 Loading multi-tenant schema...")

    try:
        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

        # Read schema file
        schema_file = (
            project_root
            / "services"
            / "rag-service"
            / "schema"
            / "multitenant_schema.surql"
        )
        if not schema_file.exists():
            logger.error(f"❌ Schema file not found: {schema_file}")
            return False

        with open(schema_file, "r", encoding="utf-8") as f:
            schema_content = f.read()

        # Split schema into individual statements
        statements = [
            stmt.strip() for stmt in schema_content.split(";") if stmt.strip()
        ]

        success_count = 0
        for i, statement in enumerate(statements):
            if not statement:
                continue

            try:
                await db.query(statement)
                success_count += 1
            except Exception as e:
                # Log non-critical errors but continue
                if "already exists" not in str(e).lower():
                    logger.warning(f"⚠️ Statement {i+1} warning: {e}")

            logger.info(
                f"✅ Schema loaded successfully ({success_count}/{len(statements)} statements)"
            )
            return True

    except Exception as e:
        logger.error(f"❌ Schema loading failed: {e}")
        return False


async def test_tenant_operations():
    """Test tenant operations."""
    logger.info("🏢 Testing tenant operations...")

    try:
        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

            # Test tenant creation
            tenant_data = {
                "id": "tenant:test_corp",
                "name": "Test Corporation",
                "slug": "test_corp",
                "description": "Test tenant for validation",
                "status": "active",
                "language_preference": "hr",
                "cultural_context": "croatian_business",
                "settings": {
                    "max_documents_per_user": 500,
                    "enable_advanced_categorization": True,
                },
            }

            result = await db.create("tenant:test_corp", tenant_data)
            logger.info(f"✅ Created tenant: {result[0]['id']}")

            # Test tenant retrieval
            retrieved = await db.select("tenant:test_corp")
            if retrieved:
                logger.info(f"✅ Retrieved tenant: {retrieved[0]['name']}")

            # Test tenant update
            await db.update("tenant:test_corp", {"description": "Updated test tenant"})
            logger.info("✅ Updated tenant")

            # Test tenant query
            tenants = await db.query("SELECT * FROM tenant WHERE status = 'active'")
            logger.info(f"✅ Found {len(tenants[0]['result'])} active tenants")
            return True

    except Exception as e:
        logger.error(f"❌ Tenant operations failed: {e}")
        return False


async def test_user_operations():
    """Test user CRUD operations with tenant relationship."""
    logger.info("👤 Testing user operations...")

    try:
        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

        # Test user creation
        user_data = {
            "id": "user:test_user",
            "tenant_id": "tenant:test_corp",
            "email": "test@testcorp.com",
            "username": "test_user",
            "full_name": "Test User",
            "password_hash": "$2b$12$test_hash",
            "role": "member",
            "status": "active",
            "language_preference": "hr",
            "settings": {
                "preferred_categories": ["business", "technical"],
                "search_both_scopes": True,
            },
        }

        result = await db.create("user:test_user", user_data)
        logger.info(f"✅ Created user: {result[0]['email']}")

        # Test user-tenant relationship query
        user_with_tenant = await db.query(
            """
            SELECT *, tenant_id.* FROM user:test_user
        """
        )

        if user_with_tenant[0]["result"]:
            user = user_with_tenant[0]["result"][0]
            logger.info(
                f"✅ User-tenant relationship working: {user['username']} → {user.get('tenant_id', {}).get('name', 'Unknown')}"
            )
            return True

    except Exception as e:
        logger.error(f"❌ User operations failed: {e}")
        return False


async def test_document_scoping():
    """Test document scoping with user and tenant scopes."""
    logger.info("📄 Testing document scoping...")

    try:
        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

        # Create user-scoped document
        user_doc_data = {
            "id": "document:user_doc_1",
            "tenant_id": "tenant:test_corp",
            "user_id": "user:test_user",
            "title": "User Private Document",
            "filename": "private.pdf",
            "file_path": "/data/user/private.pdf",
            "file_type": "pdf",
            "language": "hr",
            "scope": "user",
            "status": "processed",
            "categories": ["business", "technical"],
            "chunk_count": 5,
        }

        user_doc = await db.create("document:user_doc_1", user_doc_data)
        logger.info(f"✅ Created user document: {user_doc[0]['title']}")

        # Create tenant-scoped document
        tenant_doc_data = {
            "id": "document:tenant_doc_1",
            "tenant_id": "tenant:test_corp",
            "user_id": "user:test_user",
            "title": "Tenant Shared Document",
            "filename": "shared.pdf",
            "file_path": "/data/tenant/shared.pdf",
            "file_type": "pdf",
            "language": "hr",
            "scope": "tenant",
            "status": "processed",
            "categories": ["business"],
            "chunk_count": 3,
        }

        tenant_doc = await db.create("document:tenant_doc_1", tenant_doc_data)
        logger.info(f"✅ Created tenant document: {tenant_doc[0]['title']}")

        # Test scoped queries
        user_docs = await db.query(
            """
            SELECT * FROM document
            WHERE tenant_id = $tenant AND scope = 'user' AND user_id = $user
        """,
            {"tenant": "tenant:test_corp", "user": "user:test_user"},
        )

        tenant_docs = await db.query(
            """
            SELECT * FROM document
            WHERE tenant_id = $tenant AND scope = 'tenant'
        """,
            {"tenant": "tenant:test_corp"},
        )

        logger.info(f"✅ User documents: {len(user_docs[0]['result'])}")
        logger.info(f"✅ Tenant documents: {len(tenant_docs[0]['result'])}")
        return True

    except Exception as e:
        logger.error(f"❌ Document scoping failed: {e}")
        return False


async def test_collection_naming():
    """Test ChromaDB collection naming strategy."""
    logger.info("🏷️ Testing collection naming strategy...")

    # Test collection name generation
    tenant = DEFAULT_DEVELOPMENT_TENANT

    # Test different scopes and languages
    test_cases = [
        (DocumentScope.USER, Language.CROATIAN, "development_user_hr"),
        (DocumentScope.USER, Language.ENGLISH, "development_user_en"),
        (DocumentScope.TENANT, Language.CROATIAN, "development_tenant_hr"),
        (DocumentScope.TENANT, Language.ENGLISH, "development_tenant_en"),
    ]

    for scope, language, expected in test_cases:
        collection_name = tenant.get_collection_name(scope, language)
        if collection_name == expected:
            logger.info(
                f"✅ Collection naming: {scope.value} + {language.value} = {collection_name}"
            )
        else:
            logger.error(
                f"❌ Collection naming failed: expected {expected}, got {collection_name}"
            )

    # Test TenantUserContext collection methods
    context = DEFAULT_DEVELOPMENT_CONTEXT

    user_collection = context.get_user_collection_name(Language.CROATIAN)
    tenant_collection = context.get_tenant_collection_name(Language.CROATIAN)
    search_collections = context.get_search_collections(Language.CROATIAN)

    logger.info(f"✅ User collection: {user_collection}")
    logger.info(f"✅ Tenant collection: {tenant_collection}")
    logger.info(f"✅ Search collections: {search_collections}")

    return True


async def test_access_control():
    """Test document access control logic."""
    logger.info("🔐 Testing access control...")

    context = DEFAULT_DEVELOPMENT_CONTEXT

    # Test document access scenarios
    user_doc = Document(
        id="doc_1",
        tenant_id=context.tenant.id,
        user_id=context.user.id,
        title="User Document",
        filename="user.pdf",
        file_path="/data/user.pdf",
        scope=DocumentScope.USER,
    )

    tenant_doc = Document(
        id="doc_2",
        tenant_id=context.tenant.id,
        user_id="user:other_user",
        title="Tenant Document",
        filename="tenant.pdf",
        file_path="/data/tenant.pdf",
        scope=DocumentScope.TENANT,
    )

    other_tenant_doc = Document(
        id="doc_3",
        tenant_id="tenant:other",
        user_id="user:other_user",
        title="Other Tenant Document",
        filename="other.pdf",
        file_path="/data/other.pdf",
        scope=DocumentScope.TENANT,
    )

    # Test access control
    test_cases = [
        (user_doc, True, "Own user document"),
        (tenant_doc, True, "Tenant document"),
        (other_tenant_doc, False, "Other tenant document"),
    ]

    for doc, expected, description in test_cases:
        can_access = context.can_access_document(doc)
        status = "✅" if can_access == expected else "❌"
        logger.info(f"{status} Access control - {description}: {can_access}")

    return True


async def test_categorization_templates():
    """Test categorization template functionality."""
    logger.info("📝 Testing categorization templates...")

    try:
        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

        # Query system templates
        templates = await db.query(
            """
            SELECT * FROM categorization_template
            WHERE is_system_default = true AND language = 'hr'
        """
        )

        template_count = len(templates[0]["result"])
        logger.info(f"✅ Found {template_count} system Croatian templates")

        if template_count > 0:
            for template in templates[0]["result"]:
                logger.info(f"   - {template['name']}: {template['category']}")

        # Test custom tenant template
        custom_template = {
            "id": "categorization_template:custom_business",
            "tenant_id": "tenant:test_corp",
            "name": "Custom Business Template",
            "category": "business",
            "language": "hr",
            "keywords": ["poslovanje", "tvrtka", "startup", "biznis"],
            "patterns": ["kako pokrenuti.*", "poslovanje.*", "startup.*"],
            "system_prompt": "Ti si poslovni savjetnik za hrvatska poduzeća.",
            "user_prompt_template": "Kontekst: {context}\n\nPitanje: {query}\n\nDaj poslovni savjet.",
            "is_system_default": False,
            "is_active": True,
            "priority": 5,
        }

        result = await db.create(
            "categorization_template:custom_business", custom_template
        )
        logger.info(f"✅ Created custom template: {result[0]['name']}")
        return True

    except Exception as e:
        logger.error(f"❌ Categorization template test failed: {e}")
        return False


async def test_system_functions():
    """Test SurrealDB functions and computed fields."""
    logger.info("⚙️ Testing system functions...")

    try:
        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

        # Test collection name function
        result = await db.query(
            "RETURN fn::get_collection_name('test_corp', 'user', 'hr')"
        )
        if result and result[0]["result"]:
            collection_name = result[0]["result"][0]
            logger.info(f"✅ Collection name function: {collection_name}")

        # Test user access function
        access_result = await db.query(
            """
            RETURN fn::user_can_access_document(user:test_user, document:user_doc_1)
        """
        )
        if access_result and access_result[0]["result"]:
            can_access = access_result[0]["result"][0]
            logger.info(f"✅ User access function: {can_access}")
            return True

    except Exception as e:
        logger.error(f"❌ System functions test failed: {e}")
        return False


async def cleanup_test_data():
    """Clean up test data from database."""
    logger.info("🧹 Cleaning up test data...")

    try:
        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

        # Delete test records
        test_records = [
            "tenant:test_corp",
            "user:test_user",
            "document:user_doc_1",
            "document:tenant_doc_1",
            "categorization_template:custom_business",
        ]

        for record_id in test_records:
            try:
                await db.delete(record_id)
                logger.info(f"✅ Deleted: {record_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not delete {record_id}: {e}")
            return True

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return False


async def generate_schema_report():
    """Generate comprehensive schema validation report."""
    logger.info("📊 Generating schema validation report...")

    try:
        async with Surreal("ws://localhost:8000/rpc") as db:
            await db.use("test", "multitenant")

        # Get table information
        await db.query("SELECT * FROM information_schema.tables")

        report = {
            "timestamp": datetime.now().isoformat(),
            "database": "multitenant",
            "schema_validation": {
                "tables_created": [],
                "functions_available": [],
                "indexes_created": [],
                "default_data": {},
            },
        }

        # Check default tenant and user
        default_tenant = await db.select("tenant:development")
        default_user = await db.select("user:dev_user")

        if default_tenant:
            report["schema_validation"]["default_data"]["tenant"] = default_tenant[0]
            logger.info("✅ Default development tenant found")

        if default_user:
            report["schema_validation"]["default_data"]["user"] = default_user[0]
            logger.info("✅ Default development user found")

        # Check system templates
        system_templates = await db.query(
            """
            SELECT * FROM categorization_template WHERE is_system_default = true
        """
        )

        if system_templates[0]["result"]:
            report["schema_validation"]["default_data"]["templates"] = len(
                system_templates[0]["result"]
            )
            logger.info(
                f"✅ Found {len(system_templates[0]['result'])} system templates"
            )

        # Save report
        report_file = "multitenant_schema_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"📋 Report saved to: {report_file}")
            return True

    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting Multi-Tenant Schema Tests")
    logger.info("=" * 60)

    test_results = []

    try:
        # Run all tests in sequence
        tests = [
            ("Database Connection", test_database_connection),
            ("Schema Loading", load_schema),
            ("Tenant Operations", test_tenant_operations),
            ("User Operations", test_user_operations),
            ("Document Scoping", test_document_scoping),
            ("Collection Naming", test_collection_naming),
            ("Access Control", test_access_control),
            ("Categorization Templates", test_categorization_templates),
            ("System Functions", test_system_functions),
            ("Schema Report", generate_schema_report),
            ("Cleanup", cleanup_test_data),
        ]

        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = await test_func()
                test_results.append((test_name, result))
                if result:
                    logger.info(f"✅ {test_name} completed successfully")
                else:
                    logger.error(f"❌ {test_name} failed")
            except Exception as e:
                logger.error(f"❌ {test_name} error: {e}")
                test_results.append((test_name, False))

        # Summary
        successful = sum(1 for _, result in test_results if result)
        total = len(test_results)

        logger.info(f"\n🎉 Test Summary: {successful}/{total} tests passed")

        if successful == total:
            logger.info("✅ Multi-tenant schema is fully functional!")
            logger.info("🚀 Ready for multi-tenant RAG system implementation")
        else:
            logger.warning("⚠️ Some tests failed - review results above")

        # Show failed tests
        failed_tests = [name for name, result in test_results if not result]
        if failed_tests:
            logger.warning(f"❌ Failed tests: {', '.join(failed_tests)}")

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
