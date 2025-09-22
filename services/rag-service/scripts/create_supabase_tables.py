#!/usr/bin/env python3
"""
Simple Supabase table creation script using REST API
Creates the necessary tables for chat functionality.
"""

import asyncio
import httpx
import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import get_config_section
from src.utils.logging_factory import get_system_logger


async def create_tables_via_rest():
    """Create tables using Supabase REST API and manual SQL execution."""
    logger = get_system_logger()

    try:
        # Load Supabase configuration
        database_config = get_config_section("config", "database")
        supabase_config = database_config["supabase"]

        url = supabase_config["url"]
        service_key = supabase_config["service_role_key"]

        logger.info("supabase_create", "start", "Creating tables via REST API")

        # First, let's create the tables by inserting into them
        # This will auto-create the tables if they don't exist

        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
                "apikey": service_key
            }

            # Test if conversations table exists by trying to query it
            logger.info("supabase_create", "test", "Testing if conversations table exists")

            try:
                response = await client.get(
                    f"{url}/rest/v1/conversations",
                    headers=headers,
                    params={"limit": "1"}
                )

                if response.status_code == 200:
                    logger.info("supabase_create", "exists", "Conversations table already exists")
                    conversations_exist = True
                else:
                    logger.info("supabase_create", "missing", f"Conversations table doesn't exist: {response.status_code}")
                    conversations_exist = False

            except Exception as e:
                logger.info("supabase_create", "missing", f"Conversations table doesn't exist: {e}")
                conversations_exist = False

            # Test if chat_messages table exists
            try:
                response = await client.get(
                    f"{url}/rest/v1/chat_messages",
                    headers=headers,
                    params={"limit": "1"}
                )

                if response.status_code == 200:
                    logger.info("supabase_create", "exists", "Chat messages table already exists")
                    messages_exist = True
                else:
                    logger.info("supabase_create", "missing", f"Chat messages table doesn't exist: {response.status_code}")
                    messages_exist = False

            except Exception as e:
                logger.info("supabase_create", "missing", f"Chat messages table doesn't exist: {e}")
                messages_exist = False

            if conversations_exist and messages_exist:
                logger.info("supabase_create", "complete", "✅ All tables already exist!")
                print("\n✅ Tables already exist and are ready for use!")
                return True

            print(f"\n{'='*60}")
            print("📋 SUPABASE TABLE CREATION INSTRUCTIONS")
            print(f"{'='*60}")
            print("\nSince Supabase doesn't allow programmatic table creation via REST API,")
            print("you need to create the tables manually in the Supabase dashboard.")
            print(f"\n1. Go to: https://supabase.com/dashboard/project/vdmizraansyjcblabuqp/sql")
            print("2. Click on 'SQL Editor' in the left sidebar")
            print("3. Copy and paste the following SQL:")
            print(f"\n{'-'*40}")
            print("-- Create conversations table")
            print("""CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    tenant_slug TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    message_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create chat_messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    order_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_conversations_tenant_user ON conversations(tenant_slug, user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_order ON chat_messages(conversation_id, order_index);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON chat_messages(timestamp DESC);

-- Enable Row Level Security (optional for testing)
-- ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;""")
            print(f"{'-'*40}")
            print("\n4. Click 'Run' to execute the SQL")
            print("5. Verify the tables were created in the 'Table Editor'")
            print(f"\n{'='*60}")

            return False

    except Exception as e:
        logger.error("supabase_create", "failed", f"Table creation check failed: {e}")
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Checking Supabase tables for chat system...")
    success = asyncio.run(create_tables_via_rest())
    sys.exit(0 if success else 1)