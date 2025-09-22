# Multi-Tenant Chat API Design

## Overview

This document outlines the REST API design for a multi-tenant RAG-enabled chat system supporting both user-private chats and tenant-shared collaborative chats.

## 1. Tenant Architecture Patterns

### Hierarchical Resource Structure
```
Tenant (Organization)
├── Users (Members of the organization)
├── Shared Chats (Accessible to all tenant users)
└── User-Specific Chats (Private to individual users)
```

### Resource Ownership Model
```json
{
  "tenant_id": "acme_corp",
  "owner_type": "user|tenant",
  "owner_id": "user_123|tenant_acme_corp",
  "visibility": "private|tenant_shared|public",
  "permissions": {
    "read": ["user_123", "user_456"],
    "write": ["user_123"],
    "admin": ["user_123"]
  }
}
```

## 2. REST API Design for Multi-Tenant Chats

### Recommended Approach: Separate Endpoints

```json
# User-scoped chats (private to user)
GET    /tenants/{tenant_id}/users/{user_id}/chats
POST   /tenants/{tenant_id}/users/{user_id}/chats
GET    /tenants/{tenant_id}/users/{user_id}/chats/{chat_id}
DELETE /tenants/{tenant_id}/users/{user_id}/chats/{chat_id}

# Tenant-scoped chats (shared among tenant users)
GET    /tenants/{tenant_id}/chats
POST   /tenants/{tenant_id}/chats
GET    /tenants/{tenant_id}/chats/{chat_id}
DELETE /tenants/{tenant_id}/chats/{chat_id}

# Messages work the same for both scopes
POST   /tenants/{tenant_id}/chats/{chat_id}/messages
GET    /tenants/{tenant_id}/chats/{chat_id}/messages
```

**Advantages:**
- **Clear Ownership**: URL structure immediately shows scope
- **Permission Simplicity**: Different endpoints = different permission models
- **API Clarity**: No ambiguity about resource access level
- **Scaling**: Can apply different rate limits/quotas per scope

### Alternative: Unified with Scope Parameter
```json
# Unified endpoint with scope distinction
GET /tenants/{tenant_id}/chats?scope=user&user_id=123
GET /tenants/{tenant_id}/chats?scope=tenant

POST /tenants/{tenant_id}/chats
{
  "title": "Team Discussion",
  "scope": "tenant|user",
  "visibility": "private|tenant_shared"
}
```

## 3. Database Schema for Multi-Tenant

### PostgreSQL Schema
```sql
CREATE TABLE tenants (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(50) UNIQUE NOT NULL,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE tenant_users (
    id UUID PRIMARY KEY,
    tenant_id VARCHAR(50) REFERENCES tenants(id),
    user_id VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    role VARCHAR(50) DEFAULT 'member', -- 'admin', 'member', 'viewer'
    permissions JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, user_id)
);

CREATE TABLE chats (
    id UUID PRIMARY KEY,
    tenant_id VARCHAR(50) REFERENCES tenants(id),

    -- Ownership fields
    owner_type VARCHAR(20) NOT NULL, -- 'user' or 'tenant'
    owner_id VARCHAR(100) NOT NULL,  -- user_id or tenant_id

    -- Metadata
    title TEXT NOT NULL,
    visibility VARCHAR(20) DEFAULT 'private', -- 'private', 'tenant_shared'

    -- RAG configuration per chat
    rag_config JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chat_permissions (
    chat_id UUID REFERENCES chats(id),
    user_id VARCHAR(100) NOT NULL,
    permission VARCHAR(20) NOT NULL, -- 'read', 'write', 'admin'
    granted_by VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (chat_id, user_id, permission)
);

-- Indexes for performance
CREATE INDEX idx_chats_tenant_owner ON chats(tenant_id, owner_type, owner_id);
CREATE INDEX idx_chats_visibility ON chats(tenant_id, visibility);
CREATE INDEX idx_chat_permissions_user ON chat_permissions(user_id);
```

### Permission Resolution Logic
```python
def get_user_accessible_chats(tenant_id: str, user_id: str) -> List[Chat]:
    """Get all chats accessible to a user in a tenant."""

    # 1. User's own private chats
    user_chats = db.query("""
        SELECT * FROM chats
        WHERE tenant_id = %s
        AND owner_type = 'user'
        AND owner_id = %s
    """, tenant_id, user_id)

    # 2. Tenant shared chats
    tenant_chats = db.query("""
        SELECT * FROM chats
        WHERE tenant_id = %s
        AND visibility = 'tenant_shared'
    """, tenant_id)

    # 3. Explicitly shared chats
    shared_chats = db.query("""
        SELECT c.* FROM chats c
        JOIN chat_permissions cp ON c.id = cp.chat_id
        WHERE c.tenant_id = %s
        AND cp.user_id = %s
        AND cp.permission IN ('read', 'write', 'admin')
    """, tenant_id, user_id)

    return deduplicate(user_chats + tenant_chats + shared_chats)
```

## 4. API Examples with Permission Models

### Creating Chats
```json
# User creates private chat
POST /tenants/acme_corp/users/john_doe/chats
{
  "title": "My Personal Research",
  "rag_config": {
    "language": "hr",
    "max_documents": 5
  }
}

# User creates tenant-shared chat
POST /tenants/acme_corp/chats
{
  "title": "Team Brainstorming",
  "visibility": "tenant_shared",
  "rag_config": {
    "language": "en",
    "max_documents": 10
  }
}

# Admin creates restricted shared chat
POST /tenants/acme_corp/chats
{
  "title": "Executive Planning",
  "visibility": "private",
  "permissions": [
    {"user_id": "ceo", "permission": "admin"},
    {"user_id": "cto", "permission": "write"},
    {"user_id": "secretary", "permission": "read"}
  ]
}
```

### Querying Chats
```json
# Get all user's accessible chats
GET /tenants/acme_corp/users/john_doe/accessible-chats
Response: {
  "user_chats": [...],      // User's private chats
  "tenant_chats": [...],    // Shared tenant chats
  "shared_chats": [...],    // Explicitly shared chats
  "total": 15
}

# Get only tenant-shared chats
GET /tenants/acme_corp/chats?visibility=tenant_shared

# Get chats by topic/RAG content
GET /tenants/acme_corp/chats?search=machine%20learning&scope=accessible
```

## 5. Permission & Security Considerations

### Role-Based Access Control
```python
class TenantRole(Enum):
    ADMIN = "admin"      # Can manage tenant, create shared chats
    MEMBER = "member"    # Can participate in shared chats, create private
    VIEWER = "viewer"    # Read-only access to shared chats

class ChatPermission(Enum):
    ADMIN = "admin"      # Delete, modify permissions
    WRITE = "write"      # Send messages, modify chat settings
    READ = "read"        # View messages only
```

### Middleware for Permission Checking
```python
async def check_chat_access(
    tenant_id: str,
    user_id: str,
    chat_id: str,
    required_permission: ChatPermission
) -> bool:
    """Verify user has required permission for chat."""

    chat = await get_chat(chat_id)

    # Check tenant membership first
    if not await is_tenant_member(tenant_id, user_id):
        return False

    # Owner always has admin access
    if chat.owner_type == "user" and chat.owner_id == user_id:
        return True

    # Tenant shared chats - check tenant role
    if chat.visibility == "tenant_shared":
        tenant_role = await get_user_tenant_role(tenant_id, user_id)
        return tenant_role in [TenantRole.ADMIN, TenantRole.MEMBER]

    # Explicit permissions
    permissions = await get_user_chat_permissions(chat_id, user_id)
    return required_permission in permissions
```

## 6. RAG Considerations for Multi-Tenant

### Document Isolation
```python
# Separate document collections per tenant/user scope
tenant_collection = f"{tenant_id}_shared_documents"
user_collection = f"{tenant_id}_{user_id}_private_documents"

# Chat-specific RAG config
chat_rag_config = {
    "document_sources": [
        {"collection": tenant_collection, "weight": 0.7},
        {"collection": user_collection, "weight": 0.3}
    ],
    "language": "hr",
    "max_results": 5
}
```

### Knowledge Sharing Patterns
```json
{
  "tenant_knowledge": {
    "public_docs": "All tenant users can search",
    "dept_docs": "Department-specific document access",
    "role_docs": "Role-based document visibility"
  },
  "user_knowledge": {
    "private_docs": "User's personal document uploads",
    "shared_docs": "User explicitly shares with others"
  }
}
```

## 7. Message API Design

### Send Message
```json
POST /tenants/{tenant_id}/chats/{chat_id}/messages
{
  "content": "What are the key features of RAG?",
  "rag_config": {
    "max_documents": 5,
    "min_confidence": 0.7,
    "language": "hr"
  },
  "model_config": {
    "model": "qwen2.5:7b",
    "temperature": 0.7,
    "max_tokens": 2000
  }
}

Response: {
  "message_id": "msg_ulid_456",
  "content": "RAG combines retrieval and generation...",
  "rag_results": {
    "documents_found": 3,
    "confidence_score": 0.89,
    "search_time_ms": 234
  },
  "usage": {
    "input_tokens": 150,
    "output_tokens": 300,
    "total_tokens": 450
  }
}
```

### Get Message History
```json
GET /tenants/{tenant_id}/chats/{chat_id}/messages?limit=50&after=msg_123

Response: {
  "messages": [
    {
      "id": "msg_123",
      "role": "user",
      "content": "What is machine learning?",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "id": "msg_124",
      "role": "assistant",
      "content": "Machine learning is...",
      "rag_context": {
        "documents_used": 3,
        "confidence": 0.85
      },
      "model": "qwen2.5:7b",
      "timestamp": "2024-01-01T12:00:15Z"
    }
  ],
  "has_more": false,
  "next_cursor": null
}
```

## 8. Authentication & Authorization

### Headers for all requests
```json
{
  "Authorization": "Bearer jwt_token_here",
  "X-Tenant-Slug": "acme_corp",
  "Content-Type": "application/json"
}
```

### JWT payload should include
```json
{
  "user_id": "john_doe",
  "tenant_id": "acme_corp",
  "tenant_role": "member",
  "permissions": ["chat:read", "chat:write", "tenant:shared_chat"],
  "exp": 1704067200
}
```

## 9. Complete API Reference

### Chat Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tenants/{tenant_id}/users/{user_id}/chats` | List user's private chats |
| POST | `/tenants/{tenant_id}/users/{user_id}/chats` | Create user private chat |
| GET | `/tenants/{tenant_id}/chats` | List tenant shared chats |
| POST | `/tenants/{tenant_id}/chats` | Create tenant shared chat |
| GET | `/tenants/{tenant_id}/chats/{chat_id}` | Get chat details |
| PUT | `/tenants/{tenant_id}/chats/{chat_id}` | Update chat settings |
| DELETE | `/tenants/{tenant_id}/chats/{chat_id}` | Delete chat |

### Messages
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tenants/{tenant_id}/chats/{chat_id}/messages` | Send message with RAG |
| GET | `/tenants/{tenant_id}/chats/{chat_id}/messages` | Get message history |
| GET | `/tenants/{tenant_id}/chats/{chat_id}/messages/{message_id}` | Get specific message |

### Utility Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tenants/{tenant_id}/users/{user_id}/accessible-chats` | All accessible chats for user |
| GET | `/tenants/{tenant_id}/chats/{chat_id}/permissions` | Get chat permissions |
| PUT | `/tenants/{tenant_id}/chats/{chat_id}/permissions` | Update chat permissions |

## 10. Implementation Notes

### Recommended Architecture
- **Separate Endpoints**: Use Option A (separate endpoints) for clearer ownership model
- **PostgreSQL**: Recommended database for ACID compliance and JSONB support
- **Permission Middleware**: Implement centralized permission checking
- **RAG Integration**: Chat-specific RAG configurations with tenant/user document isolation

### Future Considerations
- **Real-time**: WebSocket support for live chat features
- **Search**: Full-text search across chat history with proper permission filtering
- **Analytics**: Usage metrics and conversation analytics per tenant
- **Export**: Chat export functionality with privacy controls
- **Integrations**: Slack/Teams integration for tenant shared chats

---

*This design provides a modern, scalable foundation for AI chat with RAG that follows 2024 best practices while maintaining flexibility for future evolution.*