# 2025-09-25 Web API Specification

**Version**: 1.0.0
**Date**: 2025-09-25
**Status**: Development Specification

## Overview

Unified Web API for multi-tenant RAG platform with chat capabilities, user management, tenant administration, and feature-based scoping. All tenant/user context is derived from authentication tokens rather than URL parameters.

### API Architecture

This is a **modular monolith** API that handles all user-facing operations including:
- Authentication & Authorization
- User Management
- Tenant Management
- Chat & Messaging (with **full Markdown preservation**)
- Feature Flags & Permissions
- Settings & Preferences

The API is built with TypeScript/Node.js and communicates with the Python RAG service for AI/ML operations.

## Base URL

```
Development: http://localhost:3000/api/v1
Production: https://api.{domain}/v1
```

## Authentication

### Mock Authentication (Development)
```bash
# Set in .env.development
AUTH_MODE=mock

# Mock context automatically provided:
{
  "user": { "id": "dev_user_123", "email": "dev@example.com" },
  "tenant": { "id": "dev_tenant", "slug": "development" },
  "features": ["narodne-novine", "financial-reports"],
  "permissions": ["chat:create", "chat:read", "chat:write", "chat:delete"]
}
```

### JWT Authentication (Production)
```bash
Authorization: Bearer <jwt_token>

# JWT Payload Structure:
{
  "userId": "user_123",
  "email": "user@example.com",
  "tenantId": "tenant_456",
  "tenantSlug": "acme-corp",
  "features": ["narodne-novine"],
  "permissions": ["chat:create", "chat:read", "chat:write"],
  "iat": 1699564800,
  "exp": 1699568400
}
```

## Core Resources

### Chat Object
```typescript
{
  "id": "chat_abc123",
  "title": "Narodne Novine Research",
  "feature": "narodne-novine",           // Feature scope
  "visibility": "private",                // "private" | "tenant_shared"
  "tenantId": "tenant_456",
  "userId": "user_123",
  "ragConfig": {
    "language": "hr",
    "maxDocuments": 5,
    "minConfidence": 0.7,
    "temperature": 0.7
  },
  "metadata": {
    "tags": ["research", "2024"],
    "description": "Research on latest regulations"
  },
  "createdAt": "2024-01-01T12:00:00Z",
  "updatedAt": "2024-01-01T12:00:00Z",
  "lastMessageAt": "2024-01-01T13:30:00Z",
  "messageCount": 42
}
```

### Message Object

**IMPORTANT**: The `content` field contains **raw Markdown text** that must be preserved exactly as received from the LLM. This includes:
- Headers (`###`), bold (`**`), italic (`*`), links
- Tables, lists (numbered and bulleted)
- Code blocks and inline code
- Emojis and UTF-8 characters (📌, ✅, etc.)
- Line breaks and formatting

```typescript
{
  "id": "msg_xyz789",
  "chatId": "chat_abc123",
  "role": "user" | "assistant",
  "content": "### Header\n**Bold text** with *italics*\n- Bullet point\n- Another point\n\n| Column | Data |\n|--------|------|\n| Value | 123 |\n\n✅ Complete!",  // Raw Markdown preserved
  "feature": "narodne-novine",
  "metadata": {
    // For user messages
    "edited": false,
    "editedAt": null,

    // For assistant messages
    "ragContext": {
      "documentsRetrieved": 5,
      "documentsUsed": 3,
      "confidence": 0.89,
      "searchTimeMs": 234,
      "sources": [
        {
          "documentId": "doc_123",
          "title": "NN 45/2024",
          "relevance": 0.92,
          "chunk": "Relevant excerpt..."
        }
      ]
    },
    "model": "qwen2.5:7b",
    "tokensUsed": {
      "input": 150,
      "output": 300,
      "total": 450
    },
    "responseTimeMs": 1250,
    "processingTaskId": "task_abc123"
  },
  "createdAt": "2024-01-01T13:30:00Z"
}
```

## API Endpoints

### 1. Chat Management

#### List Chats
```http
GET /api/v1/chats
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| feature | string | No | Filter by feature (e.g., "narodne-novine") |
| visibility | string | No | "private" \| "tenant_shared" \| "all" |
| search | string | No | Search in chat titles and descriptions |
| limit | number | No | Max results per page (default: 20, max: 100) |
| cursor | string | No | Pagination cursor |
| sortBy | string | No | "createdAt" \| "updatedAt" \| "lastMessageAt" |
| sortOrder | string | No | "asc" \| "desc" (default: "desc") |

**Response:**
```json
{
  "chats": [
    {
      "id": "chat_abc123",
      "title": "Narodne Novine Research",
      "feature": "narodne-novine",
      "visibility": "private",
      "lastMessageAt": "2024-01-01T13:30:00Z",
      "messageCount": 42,
      "createdAt": "2024-01-01T12:00:00Z"
    }
  ],
  "pagination": {
    "hasMore": true,
    "nextCursor": "eyJpZCI6ImNoYXRfMTIzIn0=",
    "totalCount": 156
  }
}
```

#### Create Chat
```http
POST /api/v1/chats
```

**Request Body:**
```json
{
  "title": "New Research Chat",
  "feature": "narodne-novine",
  "visibility": "private",
  "ragConfig": {
    "language": "hr",
    "maxDocuments": 5,
    "minConfidence": 0.7,
    "temperature": 0.7
  },
  "metadata": {
    "tags": ["research", "2024"],
    "description": "Research on latest regulations"
  }
}
```

**Response:** `201 Created`
```json
{
  "id": "chat_new456",
  "title": "New Research Chat",
  "feature": "narodne-novine",
  "visibility": "private",
  "tenantId": "tenant_456",
  "userId": "user_123",
  "ragConfig": {...},
  "metadata": {...},
  "createdAt": "2024-01-01T14:00:00Z",
  "updatedAt": "2024-01-01T14:00:00Z",
  "messageCount": 0
}
```

**Error Responses:**
- `403 Forbidden`: Feature not enabled for user
- `400 Bad Request`: Invalid feature or configuration

#### Get Chat Details
```http
GET /api/v1/chats/{chatId}
```

**Response:**
```json
{
  "id": "chat_abc123",
  "title": "Narodne Novine Research",
  "feature": "narodne-novine",
  "visibility": "private",
  "tenantId": "tenant_456",
  "userId": "user_123",
  "ragConfig": {...},
  "metadata": {...},
  "permissions": {
    "canEdit": true,
    "canDelete": true,
    "canShare": true
  },
  "stats": {
    "messageCount": 42,
    "documentsProcessed": 15,
    "tokensUsed": 12500
  },
  "createdAt": "2024-01-01T12:00:00Z",
  "updatedAt": "2024-01-01T13:30:00Z",
  "lastMessageAt": "2024-01-01T13:30:00Z"
}
```

#### Update Chat
```http
PUT /api/v1/chats/{chatId}
```

**Request Body:**
```json
{
  "title": "Updated Title",
  "visibility": "tenant_shared",
  "ragConfig": {
    "maxDocuments": 10
  },
  "metadata": {
    "tags": ["research", "2024", "regulations"]
  }
}
```

**Response:** `200 OK` (Returns updated chat object)

#### Delete Chat
```http
DELETE /api/v1/chats/{chatId}
```

**Response:** `204 No Content`

### 2. Message Management

#### Send Message
```http
POST /api/v1/chats/{chatId}/messages
```

**Request Body:**
```json
{
  "content": "What are the latest changes in the regulations?",
  "ragConfig": {
    "maxDocuments": 5,
    "minConfidence": 0.8,
    "language": "hr"
  },
  "modelConfig": {
    "model": "qwen2.5:7b",
    "temperature": 0.7,
    "maxTokens": 2000
  },
  "stream": false
}
```

**Response (Non-streaming):** `201 Created`
```json
{
  "userMessage": {
    "id": "msg_user123",
    "role": "user",
    "content": "What are the latest changes in the regulations?",
    "createdAt": "2024-01-01T14:00:00Z"
  },
  "assistantMessage": {
    "id": "msg_asst124",
    "role": "assistant",
    "content": "Based on the latest Narodne Novine publications...",
    "metadata": {
      "ragContext": {
        "documentsRetrieved": 5,
        "documentsUsed": 3,
        "confidence": 0.89,
        "sources": [...]
      },
      "model": "qwen2.5:7b",
      "tokensUsed": {
        "input": 150,
        "output": 300,
        "total": 450
      },
      "responseTimeMs": 1250
    },
    "createdAt": "2024-01-01T14:00:01Z"
  },
  "processingTask": {
    "id": "task_abc123",
    "status": "completed",
    "duration": 1250
  }
}
```

**Response (Streaming):** `200 OK`
```
data: {"type":"start","taskId":"task_abc123"}
data: {"type":"progress","progress":10,"message":"Retrieving documents"}
data: {"type":"progress","progress":50,"message":"Processing with LLM"}
data: {"type":"content","delta":"Based on the latest"}
data: {"type":"content","delta":" Narodne Novine publications"}
data: {"type":"metadata","ragContext":{...}}
data: {"type":"complete","messageId":"msg_asst124"}
```

#### Get Messages
```http
GET /api/v1/chats/{chatId}/messages
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| limit | number | No | Messages per page (default: 50, max: 100) |
| cursor | string | No | Pagination cursor |
| order | string | No | "asc" \| "desc" (default: "desc") |
| role | string | No | Filter by role: "user" \| "assistant" |
| search | string | No | Search in message content |

**Response:**
```json
{
  "messages": [
    {
      "id": "msg_asst124",
      "role": "assistant",
      "content": "Based on the latest Narodne Novine publications...",
      "metadata": {...},
      "createdAt": "2024-01-01T14:00:01Z"
    },
    {
      "id": "msg_user123",
      "role": "user",
      "content": "What are the latest changes?",
      "createdAt": "2024-01-01T14:00:00Z"
    }
  ],
  "pagination": {
    "hasMore": true,
    "nextCursor": "eyJpZCI6Im1zZ18xMjMifQ==",
    "totalCount": 42
  }
}
```

#### Get Message Details
```http
GET /api/v1/chats/{chatId}/messages/{messageId}
```

**Response:**
```json
{
  "id": "msg_asst124",
  "chatId": "chat_abc123",
  "role": "assistant",
  "content": "Based on the latest Narodne Novine publications...",
  "metadata": {
    "ragContext": {
      "documentsRetrieved": 5,
      "documentsUsed": 3,
      "confidence": 0.89,
      "sources": [
        {
          "documentId": "doc_123",
          "title": "NN 45/2024",
          "relevance": 0.92,
          "chunk": "Article 5 states that..."
        }
      ]
    },
    "model": "qwen2.5:7b",
    "tokensUsed": {...},
    "responseTimeMs": 1250
  },
  "createdAt": "2024-01-01T14:00:01Z"
}
```

### 3. Feature & Permission Queries

#### Get User Features
```http
GET /api/v1/user/features
```

**Response:**
```json
{
  "features": [
    {
      "id": "narodne-novine",
      "name": "Narodne Novine",
      "description": "Croatian Official Gazette search and analysis",
      "enabled": true,
      "config": {
        "maxDocuments": 10,
        "languages": ["hr", "en"]
      }
    },
    {
      "id": "financial-reports",
      "name": "Financial Reports",
      "description": "Financial document analysis",
      "enabled": false,
      "enabledAt": null
    }
  ]
}
```

#### Get User Permissions
```http
GET /api/v1/user/permissions
```

**Response:**
```json
{
  "permissions": [
    "chat:create",
    "chat:read",
    "chat:write",
    "chat:delete",
    "chat:share",
    "documents:upload",
    "documents:read"
  ],
  "tenant": {
    "id": "tenant_456",
    "slug": "acme-corp",
    "role": "member"
  }
}
```

### 4. Real-time Subscriptions (WebSocket)

#### Subscribe to Chat Updates
```javascript
// WebSocket connection
ws://localhost:3000/api/v1/ws

// Subscribe to chat
{
  "type": "subscribe",
  "channel": "chat:chat_abc123"
}

// Receive updates
{
  "type": "message:new",
  "data": {
    "messageId": "msg_125",
    "role": "assistant",
    "preview": "Based on the analysis..."
  }
}

{
  "type": "message:progress",
  "data": {
    "taskId": "task_xyz",
    "progress": 50,
    "status": "Processing with LLM"
  }
}
```

### 5. Utility Endpoints

#### Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "redis": "connected",
    "pythonRag": "connected"
  },
  "timestamp": "2024-01-01T14:00:00Z"
}
```

#### Get API Info
```http
GET /api/v1/info
```

**Response:**
```json
{
  "version": "1.0.0",
  "environment": "development",
  "features": {
    "streaming": true,
    "websockets": true,
    "maxMessageLength": 10000,
    "maxFileSizeMB": 10
  },
  "rateLimits": {
    "messagesPerMinute": 30,
    "messagesPerHour": 500,
    "chatsPerDay": 100
  }
}
```

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "FEATURE_NOT_ENABLED",
    "message": "The 'financial-reports' feature is not enabled for your account",
    "details": {
      "feature": "financial-reports",
      "availableFeatures": ["narodne-novine"]
    }
  },
  "timestamp": "2024-01-01T14:00:00Z",
  "requestId": "req_abc123"
}
```

### Common Error Codes
| Status | Code | Description |
|--------|------|-------------|
| 400 | INVALID_REQUEST | Malformed request or invalid parameters |
| 401 | UNAUTHORIZED | Missing or invalid authentication |
| 403 | FORBIDDEN | Insufficient permissions |
| 403 | FEATURE_NOT_ENABLED | Feature not available for user |
| 404 | NOT_FOUND | Resource does not exist |
| 409 | CONFLICT | Resource already exists |
| 429 | RATE_LIMITED | Too many requests |
| 500 | INTERNAL_ERROR | Server error |
| 503 | SERVICE_UNAVAILABLE | Temporary service issue |

## Rate Limiting

### Headers
```http
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1699568400
```

### Limits by Endpoint
| Endpoint | Limit | Window |
|----------|-------|--------|
| POST /chats/*/messages | 30 | 1 minute |
| GET /chats | 100 | 1 minute |
| POST /chats | 10 | 1 minute |
| All others | 60 | 1 minute |

## Pagination

### Cursor-based Pagination
```json
// Request
GET /api/v1/chats?limit=20&cursor=eyJpZCI6ImNoYXRfMTIzIn0=

// Response
{
  "data": [...],
  "pagination": {
    "hasMore": true,
    "nextCursor": "eyJpZCI6ImNoYXRfMTQzIn0=",
    "totalCount": 156
  }
}
```

## Webhooks (Future)

### Event Types
- `chat.created`
- `chat.deleted`
- `message.created`
- `message.processed`
- `feature.enabled`
- `feature.disabled`

### Webhook Payload
```json
{
  "id": "evt_abc123",
  "type": "message.created",
  "data": {
    "chatId": "chat_abc123",
    "messageId": "msg_125"
  },
  "timestamp": "2024-01-01T14:00:00Z"
}
```

## Testing Examples

### cURL Examples

```bash
# Create a chat
curl -X POST http://localhost:3000/api/v1/chats \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Chat",
    "feature": "narodne-novine",
    "visibility": "private"
  }'

# Send a message
curl -X POST http://localhost:3000/api/v1/chats/chat_abc123/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What are the latest regulations?",
    "stream": false
  }'

# List chats with filters
curl -X GET "http://localhost:3000/api/v1/chats?feature=narodne-novine&visibility=private&limit=10" \
  -H "Authorization: Bearer $TOKEN"
```

### TypeScript Client Example

```typescript
// Using tRPC client
const client = createTRPCClient<AppRouter>({
  links: [
    httpBatchLink({
      url: 'http://localhost:3000/api/trpc',
      headers: {
        authorization: `Bearer ${token}`,
      },
    }),
  ],
});

// Create chat
const chat = await client.chats.create.mutate({
  title: 'Research Chat',
  feature: 'narodne-novine',
  visibility: 'private',
});

// Send message
const response = await client.messages.send.mutate({
  chatId: chat.id,
  content: 'What are the latest regulations?',
});

// Subscribe to updates
const unsubscribe = client.messages.onUpdate.subscribe(
  { chatId: chat.id },
  {
    onData: (update) => {
      console.log('Message update:', update);
    },
  }
);
```

## Implementation Notes

1. **Auth Context**: All user/tenant info comes from auth token, not URL parameters
2. **Feature Validation**: Backend validates feature access on every request
3. **Async Processing**: RAG queries processed asynchronously with task tracking
4. **Real-time Updates**: WebSocket/SSE for live message streaming
5. **Type Safety**: Auto-generated TypeScript types from Pydantic models
6. **Mock Development**: Use `AUTH_MODE=mock` for development without real auth

## Security Considerations

1. **Token Validation**: Verify JWT signature and expiration
2. **Tenant Isolation**: Automatic filtering by tenant ID from token
3. **Feature Access**: Validate feature availability before processing
4. **Rate Limiting**: Prevent abuse with per-endpoint limits
5. **Input Validation**: Strict schema validation on all inputs
6. **CORS**: Configure allowed origins properly
7. **HTTPS Only**: Enforce TLS in production

---

*This specification follows the simplified, context-aware API design pattern with features as properties rather than URL paths.*