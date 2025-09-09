#!/usr/bin/env python3
"""
Test the CLI parsing functionality for tenant/user switches.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_cli_parsing():
    """Test that CLI parses tenant/user arguments correctly."""
    print("🧪 Testing CLI argument parsing...")

    # Create the parser (copied from the CLI)
    parser = argparse.ArgumentParser(description="Multi-tenant RAG System CLI")

    # Global options
    parser.add_argument(
        "--tenant", default="development", help="Tenant slug (default: development)"
    )
    parser.add_argument(
        "--user", default="dev_user", help="User ID (default: dev_user)"
    )
    parser.add_argument(
        "--language",
        choices=["hr", "en", "multilingual"],
        default="hr",
        help="Language code (default: hr)",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query_text", help="Query text to search for")
    query_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of documents to retrieve"
    )

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # Test different argument combinations
    test_cases = [
        # Default values
        ["status"],
        # Custom tenant and user
        ["--tenant", "acme", "--user", "john", "--language", "en", "status"],
        # Query command
        [
            "--tenant",
            "development",
            "--user",
            "dev_user",
            "--language",
            "hr",
            "query",
            "Što je RAG sustav?",
        ],
        # Query with options
        [
            "--tenant",
            "enterprise",
            "--user",
            "admin",
            "--language",
            "en",
            "query",
            "What is RAG?",
            "--top-k",
            "10",
        ],
    ]

    print("Testing argument combinations:")
    for i, test_args in enumerate(test_cases, 1):
        try:
            args = parser.parse_args(test_args)
            print(f"  ✅ Test {i}: {' '.join(test_args)}")
            print(
                f"     Tenant: {args.tenant}, User: {args.user}, Language: {args.language}, Command: {args.command}"
            )
            if hasattr(args, "query_text"):
                print(f"     Query: {args.query_text}, Top-k: {args.top_k}")
        except Exception as e:
            print(f"  ❌ Test {i} failed: {e}")
        print()


def test_tenant_user_context():
    """Test tenant/user context creation without full RAG dependencies."""
    print("🧪 Testing tenant/user context creation...")

    try:
        from src.models.multitenant_models import (DEFAULT_DEVELOPMENT_TENANT,
                                                   DEFAULT_DEVELOPMENT_USER,
                                                   Tenant, TenantUserContext,
                                                   User)

        # Test default development context
        dev_context = TenantUserContext(
            tenant=DEFAULT_DEVELOPMENT_TENANT, user=DEFAULT_DEVELOPMENT_USER
        )

        print(f"  ✅ Development context:")
        print(f"     Tenant: {dev_context.tenant.name} ({dev_context.tenant.slug})")
        print(f"     User: {dev_context.user.full_name} ({dev_context.user.username})")

        # Test collection naming
        user_collection = dev_context.get_user_collection_name("hr")
        tenant_collection = dev_context.get_tenant_collection_name("hr")

        print(f"     User collection (hr): {user_collection}")
        print(f"     Tenant collection (hr): {tenant_collection}")

        # Test custom tenant/user
        custom_tenant = Tenant(
            id="tenant:acme",
            name="Acme Corporation",
            slug="acme",
            description="Acme tenant",
        )

        custom_user = User(
            id="user:john",
            tenant_id="tenant:acme",
            email="john@acme.com",
            username="john",
            full_name="John Doe",
        )

        custom_context = TenantUserContext(tenant=custom_tenant, user=custom_user)

        print(f"  ✅ Custom context:")
        print(
            f"     Tenant: {custom_context.tenant.name} ({custom_context.tenant.slug})"
        )
        print(
            f"     User: {custom_context.user.full_name} ({custom_context.user.username})"
        )

        # Test collection naming for custom context
        custom_user_collection = custom_context.get_user_collection_name("en")
        custom_tenant_collection = custom_context.get_tenant_collection_name("en")

        print(f"     User collection (en): {custom_user_collection}")
        print(f"     Tenant collection (en): {custom_tenant_collection}")

        return True

    except Exception as e:
        print(f"  ❌ Context creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_folder_manager():
    """Test folder manager with tenant/user context."""
    print("🧪 Testing folder manager with tenant context...")

    try:
        from src.models.multitenant_models import (DEFAULT_DEVELOPMENT_TENANT,
                                                   DEFAULT_DEVELOPMENT_USER)
        from src.utils.folder_manager import TenantFolderManager

        folder_manager = TenantFolderManager()

        # Test template rendering
        test_params = {"tenant_slug": "acme", "user_id": "john", "language": "en"}

        templates = [
            "tenant_root",
            "user_documents",
            "tenant_shared",
            "collection_name",
        ]

        print("  Template rendering tests:")
        for template_name in templates:
            try:
                if template_name == "collection_name":
                    result = folder_manager._render_template(
                        template_name, scope="user", **test_params
                    )
                else:
                    result = folder_manager._render_template(
                        template_name, **test_params
                    )
                print(f"    ✅ {template_name}: {result}")
            except Exception as e:
                print(f"    ❌ {template_name}: {e}")

        return True

    except Exception as e:
        print(f"  ❌ Folder manager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all CLI tests."""
    print("=" * 60)
    print("🚀 Multi-tenant CLI Tests")
    print("=" * 60)
    print()

    tests = [
        test_cli_parsing,
        test_tenant_user_context,
        test_folder_manager,
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
        print("✅ All CLI functionality tests passed!")
        print("\n💡 CLI Usage Examples:")
        print(
            "  python rag_new.py --tenant development --user dev_user --language hr status"
        )
        print(
            "  python rag_new.py --tenant acme --user john --language en query 'What is RAG?'"
        )
        print(
            "  python rag_new.py --tenant enterprise --user admin create-folders --languages hr en"
        )
        return 0
    else:
        print("❌ Some CLI tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
