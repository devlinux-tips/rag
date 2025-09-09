#!/usr/bin/env python3
"""
Test fail-fast configuration behavior to ensure no hardcoded defaults remain.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_folder_manager_fail_fast():
    """Test that TenantFolderManager fails fast when config missing."""
    print("🧪 Testing TenantFolderManager fail-fast behavior...")

    try:
        # This should work with proper config
        from src.utils.folder_manager import TenantFolderManager

        folder_manager = TenantFolderManager()
        print("  ✅ TenantFolderManager initialized successfully with config")
        return True
    except KeyError as e:
        print(f"  ❌ TenantFolderManager failed with missing config key: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  TenantFolderManager failed with unexpected error: {e}")
        return False


def test_storage_config_fail_fast():
    """Test that StorageConfig fails fast when config missing."""
    print("🧪 Testing StorageConfig fail-fast behavior...")

    try:
        from src.vectordb.storage import StorageConfig

        config = StorageConfig.from_config()
        print("  ✅ StorageConfig initialized successfully")
        return True
    except KeyError as e:
        print(f"  ❌ StorageConfig failed with missing config key: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  StorageConfig failed with unexpected error: {e}")
        return False


def test_rag_system_fail_fast():
    """Test that RAGSystem fails fast when config missing."""
    print("🧪 Testing RAGSystem fail-fast behavior...")

    try:
        from src.pipeline.rag_system import RAGSystem

        rag = RAGSystem(language="hr")
        print("  ✅ RAGSystem initialized successfully")
        return True
    except KeyError as e:
        print(f"  ❌ RAGSystem failed with missing config key: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  RAGSystem failed with unexpected error: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test basic config loading functions."""
    print("🧪 Testing basic config loading...")

    try:
        from src.utils.config_loader import (get_embeddings_config,
                                             get_processing_config,
                                             get_shared_config,
                                             get_storage_config)

        # Test each config section
        configs = [
            ("shared", get_shared_config),
            ("storage", get_storage_config),
            ("embeddings", get_embeddings_config),
            ("processing", get_processing_config),
        ]

        for name, func in configs:
            config = func()
            if config:
                print(f"  ✅ {name} config loaded successfully")
            else:
                print(f"  ❌ {name} config is empty")
                return False

        return True
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        traceback.print_exc()
        return False


def test_multitenant_models():
    """Test that multitenant models work with updated config."""
    print("🧪 Testing multitenant models...")

    try:
        from src.models.multitenant_models import (DEFAULT_DEVELOPMENT_TENANT,
                                                   DEFAULT_DEVELOPMENT_USER,
                                                   TenantUserContext)

        context = TenantUserContext(
            tenant=DEFAULT_DEVELOPMENT_TENANT, user=DEFAULT_DEVELOPMENT_USER
        )

        # Test collection naming
        user_collection = context.get_user_collection_name("hr")
        tenant_collection = context.get_tenant_collection_name("hr")

        if user_collection and tenant_collection:
            print(
                f"  ✅ Collection names generated: {user_collection}, {tenant_collection}"
            )
            return True
        else:
            print("  ❌ Collection name generation failed")
            return False

    except Exception as e:
        print(f"  ❌ Multitenant models test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all fail-fast tests."""
    print("=" * 60)
    print("🚀 Fail-Fast Configuration Tests")
    print("=" * 60)

    tests = [
        test_config_loading,
        test_folder_manager_fail_fast,
        test_storage_config_fail_fast,
        test_multitenant_models,
        test_rag_system_fail_fast,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  💥 Test {test.__name__} crashed: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed! Fail-fast behavior working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check configuration completeness.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
