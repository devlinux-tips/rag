#!/usr/bin/env python3
"""
Supabase Chat Database Setup Script
Sets up the complete database schema for the chat system with proper RLS policies.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import get_config_section
from src.utils.logging_factory import get_system_logger
from supabase import create_client, Client


async def setup_chat_database():
    """Set up the complete chat database schema in Supabase."""
    logger = get_system_logger()

    try:
        # Load Supabase configuration
        database_config = get_config_section("config", "database")
        supabase_config = database_config["supabase"]

        logger.info("supabase_setup", "start", "Setting up Supabase chat database")

        # Create Supabase client with service role key (admin access)
        supabase: Client = create_client(
            supabase_config["url"],
            supabase_config["service_role_key"]
        )

        # 1. Delete the old 'rag' table if it exists
        logger.info("supabase_setup", "cleanup", "Removing old 'rag' table if exists")
        try:
            supabase.table("rag").delete().neq("id", "none").execute()
            logger.info("supabase_setup", "cleanup", "Old 'rag' table cleaned up")
        except Exception as e:
            logger.info("supabase_setup", "cleanup", f"No 'rag' table to clean: {e}")

        # 2. Create conversations table
        logger.info("supabase_setup", "create_table", "Creating conversations table")
        conversations_sql = """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            tenant_slug TEXT NOT NULL,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            message_count INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        """

        # Execute via RPC call (Supabase SQL editor equivalent)
        result = supabase.rpc("exec_sql", {"sql": conversations_sql}).execute()
        logger.info("supabase_setup", "create_table", "Conversations table created")

        # 3. Create chat_messages table
        logger.info("supabase_setup", "create_table", "Creating chat_messages table")
        messages_sql = """
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            order_index INTEGER NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        """

        result = supabase.rpc("exec_sql", {"sql": messages_sql}).execute()
        logger.info("supabase_setup", "create_table", "Chat messages table created")

        # 4. Create indexes for performance
        logger.info("supabase_setup", "create_indexes", "Creating performance indexes")
        indexes_sql = """
        CREATE INDEX IF NOT EXISTS idx_conversations_tenant_user ON conversations(tenant_slug, user_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON chat_messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_messages_order ON chat_messages(conversation_id, order_index);
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON chat_messages(timestamp DESC);
        """

        result = supabase.rpc("exec_sql", {"sql": indexes_sql}).execute()
        logger.info("supabase_setup", "create_indexes", "Performance indexes created")

        # 5. Enable Row Level Security (RLS)
        logger.info("supabase_setup", "enable_rls", "Enabling Row Level Security")
        rls_sql = """
        ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
        ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
        """

        result = supabase.rpc("exec_sql", {"sql": rls_sql}).execute()
        logger.info("supabase_setup", "enable_rls", "RLS enabled on tables")

        # 6. Create RLS policies for multi-tenant isolation
        logger.info("supabase_setup", "create_policies", "Creating RLS policies")
        policies_sql = """
        -- Conversations policies
        CREATE POLICY IF NOT EXISTS "Users can only access their own conversations"
        ON conversations FOR ALL
        USING (tenant_slug = current_setting('app.current_tenant', true)
               AND user_id = current_setting('app.current_user', true));

        -- Messages policies
        CREATE POLICY IF NOT EXISTS "Users can only access messages from their conversations"
        ON chat_messages FOR ALL
        USING (conversation_id IN (
            SELECT conversation_id FROM conversations
            WHERE tenant_slug = current_setting('app.current_tenant', true)
            AND user_id = current_setting('app.current_user', true)
        ));
        """

        result = supabase.rpc("exec_sql", {"sql": policies_sql}).execute()
        logger.info("supabase_setup", "create_policies", "RLS policies created")

        # 7. Test the setup
        logger.info("supabase_setup", "test", "Testing database setup")

        # Test conversations table
        test_result = supabase.table("conversations").select("*").limit(1).execute()
        logger.info("supabase_setup", "test", f"Conversations table accessible: {len(test_result.data)} rows")

        # Test messages table
        test_result = supabase.table("chat_messages").select("*").limit(1).execute()
        logger.info("supabase_setup", "test", f"Chat messages table accessible: {len(test_result.data)} rows")

        logger.info("supabase_setup", "complete", "✅ Supabase chat database setup completed successfully!")

        print("\n" + "="*60)
        print("🎉 SUPABASE CHAT DATABASE SETUP COMPLETE!")
        print("="*60)
        print("\nTables created:")
        print("  ✅ conversations - Stores chat conversation metadata")
        print("  ✅ chat_messages - Stores individual chat messages")
        print("\nFeatures configured:")
        print("  ✅ Row Level Security (RLS) enabled")
        print("  ✅ Multi-tenant isolation policies")
        print("  ✅ Performance indexes")
        print("  ✅ Foreign key constraints")
        print("\nYour chat API server can now persist conversations!")
        print("="*60)

        return True

    except Exception as e:
        logger.error("supabase_setup", "failed", f"Database setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Setting up Supabase database for chat system...")
    success = asyncio.run(setup_chat_database())
    sys.exit(0 if success else 1)