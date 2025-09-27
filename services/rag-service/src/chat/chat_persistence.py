"""
Chat persistence system for storing and retrieving chat conversations.
Integrates with the database provider system for multi-tenant storage.
"""

import time
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from uuid import uuid4
from datetime import datetime

from ..database.protocols import DatabaseProvider
from ..generation.llm_provider import ChatMessage, MessageRole
from ..utils.logging_factory import get_system_logger, log_component_start, log_component_end


@dataclass
class ChatConversation:
    """Chat conversation metadata."""
    conversation_id: str
    tenant_slug: str
    user_id: str
    title: Optional[str] = None
    created_at: Union[float, datetime, None] = None
    updated_at: Union[float, datetime, None] = None
    message_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = int(time.time())
        if self.updated_at is None:
            self.updated_at = int(time.time())

    def get_created_at_timestamp(self) -> float:
        """Get created_at as Unix timestamp for JSON serialization."""
        if self.created_at is None:
            return float(int(time.time()))
        if isinstance(self.created_at, datetime):
            return self.created_at.timestamp()
        return float(self.created_at)

    def get_updated_at_timestamp(self) -> float:
        """Get updated_at as Unix timestamp for JSON serialization."""
        if self.updated_at is None:
            return float(int(time.time()))
        if isinstance(self.updated_at, datetime):
            return self.updated_at.timestamp()
        return float(self.updated_at)


@dataclass
class ChatMessageRecord:
    """Chat message record for database storage."""
    message_id: str
    conversation_id: str
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float
    order_index: int
    metadata: Optional[Dict[str, Any]] = None


class ChatPersistenceManager:
    """
    Manages chat conversation persistence using database providers.
    Provides CRUD operations for conversations and messages.
    """

    def __init__(self, database_provider: DatabaseProvider):
        self.db = database_provider
        self.logger = get_system_logger()

    async def create_conversation(self, tenant_slug: str, user_id: str, title: Optional[str] = None) -> ChatConversation:
        """Create new chat conversation."""
        log_component_start("chat_persistence", "create_conversation",
                          tenant=tenant_slug, user=user_id, has_title=title is not None)

        conversation_id = str(uuid4())
        conversation = ChatConversation(
            conversation_id=conversation_id,
            tenant_slug=tenant_slug,
            user_id=user_id,
            title=title or "New Chat",
            created_at=int(time.time()),
            updated_at=int(time.time()),
            message_count=0
        )

        # Store in database (assuming conversations table exists)
        await self.db.execute_query(
            """
            INSERT INTO conversations (
                conversation_id, tenant_slug, user_id, title, message_count
            ) VALUES ($1, $2, $3, $4, $5)
            """,
            [
                conversation.conversation_id,
                conversation.tenant_slug,
                conversation.user_id,
                conversation.title,
                conversation.message_count
            ]
        )

        self.logger.info("chat_persistence", "create_conversation",
                        f"Created conversation {conversation_id} for {user_id}")
        log_component_end("chat_persistence", "create_conversation", f"Conversation {conversation_id} created")

        return conversation

    async def add_message(self, conversation_id: str, role: MessageRole, content: str,
                         metadata: Optional[Dict[str, Any]] = None) -> ChatMessageRecord:
        """Add message to conversation."""
        log_component_start("chat_persistence", "add_message",
                          conversation_id=conversation_id, role=role.value, content_length=len(content))

        # Get current message count for ordering
        result = await self.db.fetch_one(
            "SELECT message_count FROM conversations WHERE conversation_id = $1",
            [conversation_id]
        )

        if not result:
            raise ValueError(f"Conversation {conversation_id} not found")

        order_index = result["message_count"]
        message_id = str(uuid4())
        timestamp = int(time.time())

        message = ChatMessageRecord(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role.value,
            content=content,
            timestamp=timestamp,
            order_index=order_index,
            metadata=metadata
        )

        # Insert message
        await self.db.execute_query(
            """
            INSERT INTO chat_messages (
                message_id, conversation_id, role, content,
                timestamp, order_index, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            [
                message.message_id,
                message.conversation_id,
                message.role,
                message.content,
                message.timestamp,
                message.order_index,
                json.dumps(message.metadata) if message.metadata else None
            ]
        )

        # Update conversation message count and updated_at
        await self.db.execute_query(
            """
            UPDATE conversations
            SET message_count = message_count + 1, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = $1
            """,
            [conversation_id]
        )

        self.logger.debug("chat_persistence", "add_message",
                         f"Added message {message_id} to conversation {conversation_id}")
        log_component_end("chat_persistence", "add_message", f"Message {message_id} added")

        return message

    async def get_conversation_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get messages from conversation in chronological order."""
        log_component_start("chat_persistence", "get_conversation_messages",
                          conversation_id=conversation_id, limit=limit)

        query = """
            SELECT role, content, order_index
            FROM chat_messages
            WHERE conversation_id = $1
            ORDER BY order_index ASC
        """
        params: List[Any] = [conversation_id]

        if limit:
            query += " LIMIT $2"
            params.append(limit)

        results = await self.db.fetch_all(query, params)

        messages = []
        for row in results:
            role = MessageRole(row["role"])
            messages.append(ChatMessage(role=role, content=row["content"]))

        self.logger.debug("chat_persistence", "get_conversation_messages",
                         f"Retrieved {len(messages)} messages from conversation {conversation_id}")
        log_component_end("chat_persistence", "get_conversation_messages", f"Retrieved {len(messages)} messages")

        return messages

    async def get_recent_messages(self, conversation_id: str, count: int = 10) -> List[ChatMessage]:
        """Get recent messages from conversation (for context)."""
        log_component_start("chat_persistence", "get_recent_messages",
                          conversation_id=conversation_id, count=count)

        query = """
            SELECT role, content, order_index
            FROM chat_messages
            WHERE conversation_id = $1
            ORDER BY order_index DESC
            LIMIT $2
        """

        results = await self.db.fetch_all(query, [conversation_id, count])

        # Reverse to get chronological order
        messages = []
        for row in reversed(results):
            role = MessageRole(row["role"])
            messages.append(ChatMessage(role=role, content=row["content"]))

        self.logger.debug("chat_persistence", "get_recent_messages",
                         f"Retrieved {len(messages)} recent messages from conversation {conversation_id}")
        log_component_end("chat_persistence", "get_recent_messages", f"Retrieved {len(messages)} recent messages")

        return messages

    async def list_user_conversations(self, tenant_slug: str, user_id: str, limit: int = 50) -> List[ChatConversation]:
        """List conversations for a user."""
        log_component_start("chat_persistence", "list_user_conversations",
                          tenant=tenant_slug, user=user_id, limit=limit)

        query = """
            SELECT conversation_id, tenant_slug, user_id, title,
                   created_at, updated_at, message_count
            FROM conversations
            WHERE tenant_slug = $1 AND user_id = $2
            ORDER BY updated_at DESC
            LIMIT $3
        """

        results = await self.db.fetch_all(query, [tenant_slug, user_id, limit])

        conversations = []
        for row in results:
            conversation = ChatConversation(
                conversation_id=row["conversation_id"],
                tenant_slug=row["tenant_slug"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"]
            )
            conversations.append(conversation)

        self.logger.debug("chat_persistence", "list_user_conversations",
                         f"Retrieved {len(conversations)} conversations for user {user_id}")
        log_component_end("chat_persistence", "list_user_conversations", f"Retrieved {len(conversations)} conversations")

        return conversations

    async def update_conversation_title(self, conversation_id: str, title: str) -> None:
        """Update conversation title."""
        log_component_start("chat_persistence", "update_conversation_title",
                          conversation_id=conversation_id, title=title)

        await self.db.execute_query(
            """
            UPDATE conversations
            SET title = $1, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = $2
            """,
            [title, conversation_id]
        )

        self.logger.debug("chat_persistence", "update_conversation_title",
                         f"Updated title for conversation {conversation_id}")
        log_component_end("chat_persistence", "update_conversation_title", "Title updated")

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete conversation and all its messages."""
        log_component_start("chat_persistence", "delete_conversation",
                          conversation_id=conversation_id)

        # Delete messages first (foreign key constraint)
        await self.db.execute_query(
            "DELETE FROM chat_messages WHERE conversation_id = $1",
            [conversation_id]
        )

        # Delete conversation
        await self.db.execute_query(
            "DELETE FROM conversations WHERE conversation_id = $1",
            [conversation_id]
        )

        self.logger.info("chat_persistence", "delete_conversation",
                        f"Deleted conversation {conversation_id} and all messages")
        log_component_end("chat_persistence", "delete_conversation", "Conversation deleted")

    async def get_conversation(self, conversation_id: str) -> Optional[ChatConversation]:
        """Get conversation metadata by ID."""
        result = await self.db.fetch_one(
            """
            SELECT conversation_id, tenant_slug, user_id, title,
                   created_at, updated_at, message_count
            FROM conversations
            WHERE conversation_id = $1
            """,
            [conversation_id]
        )

        if not result:
            return None

        return ChatConversation(
            conversation_id=result["conversation_id"],
            tenant_slug=result["tenant_slug"],
            user_id=result["user_id"],
            title=result["title"],
            created_at=result["created_at"],
            updated_at=result["updated_at"],
            message_count=result["message_count"]
        )

    async def initialize_schema(self) -> None:
        """Initialize database schema for chat persistence."""
        log_component_start("chat_persistence", "initialize_schema")

        # Create conversations table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                tenant_slug TEXT NOT NULL,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                metadata TEXT -- JSON metadata
            )
        """)

        # Create chat_messages table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL, -- 'system', 'user', 'assistant'
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                order_index INTEGER NOT NULL,
                metadata TEXT, -- JSON metadata
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)

        # Create indexes for performance
        await self.db.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(tenant_slug, user_id, updated_at DESC)
        """)

        await self.db.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON chat_messages(conversation_id, order_index)
        """)

        self.logger.info("chat_persistence", "initialize_schema", "Chat persistence schema initialized")
        log_component_end("chat_persistence", "initialize_schema", "Schema ready")


# Factory function
def create_chat_persistence_manager(database_provider: DatabaseProvider) -> ChatPersistenceManager:
    """Create chat persistence manager with database provider."""
    return ChatPersistenceManager(database_provider)