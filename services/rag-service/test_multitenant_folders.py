#!/usr/bin/env python3
"""
Test script for multi-tenant folder management.

Tests the TenantFolderManager with development tenant/user context
to verify folder structure creation and template rendering.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.multitenant_models import (DEFAULT_DEVELOPMENT_TENANT,
                                           DEFAULT_DEVELOPMENT_USER,
                                           TenantUserContext)
from src.utils.folder_manager import (TenantFolderManager,
                                      create_development_structure)


def test_template_rendering():
    """Test template rendering with development context."""
    print("🔧 Testing template rendering...")

    folder_manager = TenantFolderManager()

    # Test template rendering
    test_params = {
        "tenant_slug": "development",
        "user_id": "dev_user",
        "language": "hr",
    }

    templates_to_test = [
        "tenant_root",
        "user_documents",
        "tenant_shared",
        "user_processed",
        "tenant_chromadb",
        "tenant_models",
        "collection_name",
    ]

    print("Template rendering results:")
    for template_name in templates_to_test:
        try:
            if template_name == "collection_name":
                # Collection name needs scope parameter
                result = folder_manager._render_template(
                    template_name, scope="user", **test_params
                )
            else:
                result = folder_manager._render_template(template_name, **test_params)
            print(f"  ✅ {template_name}: {result}")
        except Exception as e:
            print(f"  ❌ {template_name}: Error - {e}")

    print()


def test_folder_structure_generation():
    """Test folder structure generation."""
    print("📁 Testing folder structure generation...")

    folder_manager = TenantFolderManager()
    context = TenantUserContext(
        tenant=DEFAULT_DEVELOPMENT_TENANT, user=DEFAULT_DEVELOPMENT_USER
    )

    # Test structure generation for Croatian
    paths = folder_manager.get_tenant_folder_structure(
        tenant=context.tenant, user=context.user, language="hr"
    )

    print("Generated folder structure:")
    for path_type, path in paths.items():
        print(f"  📂 {path_type}: {path}")

    print()


def test_collection_names():
    """Test ChromaDB collection name generation."""
    print("🗄️  Testing collection names...")

    folder_manager = TenantFolderManager()
    context = TenantUserContext(
        tenant=DEFAULT_DEVELOPMENT_TENANT, user=DEFAULT_DEVELOPMENT_USER
    )

    # Test collection names for different languages
    for language in ["hr", "en", "multilingual"]:
        collection_info = folder_manager.get_collection_storage_paths(context, language)
        print(f"  Language: {language}")
        print(f"    👤 User collection: {collection_info['user_collection_name']}")
        print(f"    🏢 Tenant collection: {collection_info['tenant_collection_name']}")
        print(f"    📁 Base path: {collection_info['base_path']}")

    print()


def test_folder_creation():
    """Test actual folder creation (dry run)."""
    print("🏗️  Testing folder creation...")

    # This will create the actual folder structure
    success = create_development_structure()

    if success:
        print("  ✅ Development structure created successfully")

        # Verify some key folders exist
        key_paths = [
            Path("./data/development"),
            Path("./data/development/users/dev_user"),
            Path("./data/development/users/dev_user/documents/hr"),
            Path("./data/development/users/dev_user/documents/en"),
            Path("./data/development/shared/documents/hr"),
            Path("./data/development/vectordb"),
            Path("./models/development/hr/embeddings"),
            Path("./system/logs"),
        ]

        print("  Verifying key folders:")
        for path in key_paths:
            if path.exists():
                print(f"    ✅ {path}")
            else:
                print(f"    ❌ {path} (not found)")

    else:
        print("  ❌ Failed to create development structure")

    print()


def test_context_methods():
    """Test TenantUserContext integration."""
    print("🔗 Testing context integration...")

    context = TenantUserContext(
        tenant=DEFAULT_DEVELOPMENT_TENANT, user=DEFAULT_DEVELOPMENT_USER
    )

    # Test context methods for collection naming
    language = "hr"
    user_collection = context.get_user_collection_name(language)
    tenant_collection = context.get_tenant_collection_name(language)
    search_collections = context.get_search_collections(language)

    print(f"  Context methods for language '{language}':")
    print(f"    👤 User collection: {user_collection}")
    print(f"    🏢 Tenant collection: {tenant_collection}")
    print(f"    🔍 Search collections: {search_collections}")

    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Multi-Tenant Folder Management Tests")
    print("=" * 60)
    print()

    try:
        test_template_rendering()
        test_folder_structure_generation()
        test_collection_names()
        test_context_methods()
        test_folder_creation()

        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
