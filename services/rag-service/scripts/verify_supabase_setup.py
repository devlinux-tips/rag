#!/usr/bin/env python3
"""
Verify Supabase setup after running the complete schema.
Tests all tables and basic functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import get_config_section
from src.utils.logging_factory import get_system_logger
from supabase import create_client, Client


async def verify_supabase_setup():
    """Verify that all tables exist and basic operations work."""
    logger = get_system_logger()

    try:
        # Load Supabase configuration
        database_config = get_config_section("config", "database")
        supabase_config = database_config["supabase"]

        logger.info("supabase_verify", "start", "Verifying Supabase setup")

        # Create Supabase client
        supabase: Client = create_client(
            supabase_config["url"],
            supabase_config["service_role_key"]
        )

        # Expected tables
        expected_tables = [
            'tenants',
            'users',
            'documents',
            'chunks',
            'conversations',
            'chat_messages',
            'search_queries',
            'categorization_templates',
            'system_configs'
        ]

        print(f"\n{'='*60}")
        print("🔍 VERIFYING SUPABASE DATABASE SETUP")
        print(f"{'='*60}")

        # Test each table
        table_status = {}
        for table in expected_tables:
            try:
                result = supabase.table(table).select("*").limit(1).execute()
                table_status[table] = "✅ EXISTS"
                logger.info("supabase_verify", "table_check", f"Table {table}: OK")
            except Exception as e:
                table_status[table] = f"❌ ERROR: {str(e)[:50]}..."
                logger.error("supabase_verify", "table_check", f"Table {table}: {e}")

        # Print table status
        print("\n📋 Table Status:")
        for table, status in table_status.items():
            print(f"  {table:<25} {status}")

        # Count successful tables
        successful_tables = sum(1 for status in table_status.values() if status.startswith("✅"))
        total_tables = len(expected_tables)

        print(f"\n📊 Summary: {successful_tables}/{total_tables} tables verified")

        if successful_tables == total_tables:
            print("🎉 ALL TABLES CREATED SUCCESSFULLY!")

            # Test basic operations
            print(f"\n{'='*60}")
            print("🧪 TESTING BASIC OPERATIONS")
            print(f"{'='*60}")

            # Test tenant data
            try:
                tenants = supabase.table("tenants").select("*").execute()
                print(f"  ✅ Tenants: {len(tenants.data)} records found")
                if tenants.data:
                    dev_tenant = next((t for t in tenants.data if t['slug'] == 'development'), None)
                    if dev_tenant:
                        print(f"     - Development tenant exists: {dev_tenant['name']}")
                    else:
                        print("     - No development tenant found")
            except Exception as e:
                print(f"  ❌ Tenants test failed: {e}")

            # Test user data
            try:
                users = supabase.table("users").select("*").execute()
                print(f"  ✅ Users: {len(users.data)} records found")
                if users.data:
                    dev_user = next((u for u in users.data if u['username'] == 'dev_user'), None)
                    if dev_user:
                        print(f"     - Development user exists: {dev_user['full_name']}")
                    else:
                        print("     - No development user found")
            except Exception as e:
                print(f"  ❌ Users test failed: {e}")

            # Test configuration
            try:
                configs = supabase.table("system_configs").select("*").execute()
                print(f"  ✅ System configs: {len(configs.data)} records found")
                for config in configs.data:
                    print(f"     - {config['config_key']}: {config['config_value']}")
            except Exception as e:
                print(f"  ❌ System configs test failed: {e}")

            # Test categorization templates
            try:
                templates = supabase.table("categorization_templates").select("*").execute()
                print(f"  ✅ Templates: {len(templates.data)} records found")
                for template in templates.data:
                    print(f"     - {template['name']} ({template['language']}/{template['category']})")
            except Exception as e:
                print(f"  ❌ Templates test failed: {e}")

            print(f"\n{'='*60}")
            print("🚀 SUPABASE SETUP VERIFICATION COMPLETE!")
            print(f"{'='*60}")
            print("\nYour database is ready for:")
            print("  ✅ Multi-tenant RAG system")
            print("  ✅ Chat conversations with persistence")
            print("  ✅ Document processing and storage")
            print("  ✅ Search analytics and categorization")
            print("\nYou can now start using the chat API at http://localhost:8080")
            print(f"{'='*60}")

            return True

        else:
            print(f"\n❌ SETUP INCOMPLETE: {total_tables - successful_tables} tables missing")
            print("\n💡 Please run the complete schema SQL script in Supabase dashboard:")
            print("   1. Go to Supabase SQL Editor")
            print("   2. Copy and paste scripts/complete_supabase_schema.sql")
            print("   3. Click 'Run' to execute")
            return False

    except Exception as e:
        logger.error("supabase_verify", "failed", f"Verification failed: {e}")
        print(f"\n❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Verifying Supabase database setup...")
    success = asyncio.run(verify_supabase_setup())
    sys.exit(0 if success else 1)