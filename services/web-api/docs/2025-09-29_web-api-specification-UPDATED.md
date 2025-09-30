# Web API Specification v2.0 - UPDATED
**Date**: 2025-09-29
**Version**: 2.0.0 (Updated to reflect actual implementation)
**Status**: Implementation Complete (Backend), Frontend UI Required

## Overview

Unified Web API for multi-tenant RAG platform with **JWT authentication**, chat capabilities, user management, tenant administration, and feature-based scoping. All tenant/user context is derived from JWT tokens.

### Key Changes from v1.0
- ❌ **REMOVED**: Mock authentication mode (JWT only)
- ✅ **ADDED**: Full user registration and authentication system
- ✅ **ADDED**: Profile management endpoints
- ✅ **ADDED**: Refresh token mechanism
- ✅ **ADDED**: Password change endpoint
- ⚠️ **MISSING**: Signup/login UI pages (API exists, UI does not)

### API Architecture

This is a **modular monolith** API built with:
- **TypeScript/Node.js** (Express + tRPC)
- **Prisma ORM** (PostgreSQL)
- **JWT Authentication** (bcrypt + jsonwebtoken)
- **Redis** (configured but WebSocket not yet implemented)
- **Docker** (full containerization)

---

## Base URL

```
Development: http://localhost:3000/api/v1
Production: https://api.{domain}/v1
```

---

## Authentication (JWT ONLY)

### JWT Authentication Structure

```http
Authorization: Bearer <jwt_token>
```

### JWT Payload Structure
```json
{
  "userId": "user_123",
  "email": "user@example.com",
  "name": "User Name",
  "role": "user",
  "tenantId": "tenant_456",
  "tenantSlug": "acme-corp",
  "tenantName": "Acme Corporation",
  "language": "hr",
  "features": ["narodne-novine"],
  "permissions": ["chat:create", "chat:read", "chat:write"],
  "iat": 1699564800,
  "exp": 1699565700,
  "iss": "rag-web-api",
  "aud": "rag-client"
}
```

### Token Expiration
- **Access Token**: 15 minutes
- **Refresh Token**: 7 days
- **Refresh tokens are rotated** on each refresh

---

## Core Data Models

### User Model
```typescript
{
  "id": "user_abc123",
  "email": "user@example.com",
  "name": "John Doe",
  "role": "user",  // "user" | "admin" | "owner"
  "tenantId": "tenant_456",
  "features": ["narodne-novine", "financial-reports"],
  "permissions": ["chat:create", "chat:read", "chat:write", "chat:delete"],
  "language": "hr",
  "timezone": "Europe/Zagreb",
  "settings": {},
  "emailVerified": false,
  "emailVerifiedAt": null,
  "lastLoginAt": "2024-01-01T13:30:00Z",
  "passwordChangedAt": "2024-01-01T12:00:00Z",
  "createdAt": "2024-01-01T12:00:00Z",
  "updatedAt": "2024-01-01T13:30:00Z"
}
```

### Tenant Model
```typescript
{
  "id": "tenant_abc123",
  "name": "Acme Corporation",
  "slug": "acme-corp",
  "status": "active",  // "active" | "suspended" | "trial"
  "features": ["narodne-novine", "financial-reports"],
  "settings": {},
  "createdAt": "2024-01-01T12:00:00Z",
  "updatedAt": "2024-01-01T13:30:00Z"
}
```

### Chat Model
```typescript
{
  "id": "chat_abc123",
  "title": "Narodne Novine Research",
  "feature": "narodne-novine",
  "visibility": "private",  // "private" | "tenant_shared"
  "tenantId": "tenant_456",
  "userId": "user_123",
  "language": "hr",
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

### Message Model
**IMPORTANT**: `content` field preserves **raw Markdown** exactly as received from LLM.

```typescript
{
  "id": "msg_xyz789",
  "chatId": "chat_abc123",
  "role": "user" | "assistant",
  "content": "### Header\n**Bold** with *italics*\n- Bullet\n\n| Col | Data |\n|-----|------|\n| Val | 123 |\n\n✅ Done!",
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
    "provider": "ollama",
    "tokensUsed": {
      "input": 150,
      "output": 300,
      "total": 450
    },
    "responseTimeMs": 1250,
    "processingTaskId": "task_abc123"
  },
  "status": "completed",  // "pending" | "processing" | "completed" | "failed"
  "errorMessage": null,
  "createdAt": "2024-01-01T13:30:00Z",
  "updatedAt": "2024-01-01T13:30:00Z"
}
```

---

## API Endpoints

## 1. Authentication Endpoints

### 1.1 Register New User
```http
POST /api/v1/auth/register
Content-Type: application/json
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecureP@ssw0rd!",
  "name": "John Doe",
  "tenantSlug": "acme-corp"  // Optional: join existing tenant
}
```

**Password Requirements:**
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

**Response:** `201 Created`
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "user_abc123",
      "email": "user@example.com",
      "name": "John Doe"
    },
    "tenant": {
      "id": "tenant_xyz789",
      "slug": "john-doe",
      "name": "John Doe's Organization"
    },
    "tokens": {
      "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "refreshToken": "1a2b3c4d5e6f7g8h9i0j..."
    }
  }
}
```

**Error Responses:**
- `400 Bad Request`: Validation errors
- `409 Conflict`: User already exists
- `404 Not Found`: Tenant slug not found (if joining existing tenant)

### 1.2 Login
```http
POST /api/v1/auth/login
Content-Type: application/json
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecureP@ssw0rd!",
  "deviceInfo": "Mozilla/5.0 ..."  // Optional
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "user": {
      "id": "user_abc123",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "user",
      "language": "hr"
    },
    "tenant": {
      "id": "tenant_xyz789",
      "slug": "acme-corp",
      "name": "Acme Corporation"
    },
    "tokens": {
      "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "refreshToken": "1a2b3c4d5e6f7g8h9i0j..."
    }
  }
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid credentials
- `403 Forbidden`: Account suspended/inactive

### 1.3 Refresh Access Token
```http
POST /api/v1/auth/refresh
Content-Type: application/json
```

**Request Body:**
```json
{
  "refreshToken": "1a2b3c4d5e6f7g8h9i0j..."
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "tokens": {
      "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "refreshToken": "9j8i7h6g5f4e3d2c1b0a..."  // New refresh token
    }
  }
}
```

**Notes:**
- Old refresh token is automatically revoked
- Refresh tokens are rotated on each use

**Error Responses:**
- `401 Unauthorized`: Invalid, expired, or revoked token

### 1.4 Logout (Single Device)
```http
POST /api/v1/auth/logout
Content-Type: application/json
```

**Request Body:**
```json
{
  "refreshToken": "1a2b3c4d5e6f7g8h9i0j..."
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

### 1.5 Logout All Devices
```http
POST /api/v1/auth/logout-all
Authorization: Bearer <access_token>
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Logged out from all devices successfully"
}
```

### 1.6 Get Profile
```http
GET /api/v1/auth/profile
Authorization: Bearer <access_token>
```

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "id": "user_abc123",
    "email": "user@example.com",
    "name": "John Doe",
    "role": "user",
    "language": "hr",
    "timezone": "Europe/Zagreb",
    "settings": {},
    "features": ["narodne-novine"],
    "permissions": ["chat:create", "chat:read", "chat:write", "chat:delete"],
    "emailVerified": false,
    "emailVerifiedAt": null,
    "lastLoginAt": "2024-01-01T13:30:00Z",
    "createdAt": "2024-01-01T12:00:00Z",
    "updatedAt": "2024-01-01T13:30:00Z"
  }
}
```

### 1.7 Update Profile
```http
PUT /api/v1/auth/profile
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "John Updated",
  "language": "en",
  "timezone": "America/New_York",
  "settings": {
    "theme": "dark",
    "notifications": true
  }
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "id": "user_abc123",
    "email": "user@example.com",
    "name": "John Updated",
    "role": "user",
    "language": "en",
    "timezone": "America/New_York",
    "settings": {
      "theme": "dark",
      "notifications": true
    },
    "updatedAt": "2024-01-01T14:00:00Z"
  }
}
```

**Validation:**
- `language`: Must be "hr" or "en"
- `timezone`: Must be valid IANA timezone

### 1.8 Change Password
```http
POST /api/v1/auth/change-password
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "currentPassword": "OldP@ssw0rd!",
  "newPassword": "NewSecureP@ssw0rd!"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Password changed successfully. Please login again on all devices."
}
```

**Notes:**
- All refresh tokens are revoked after password change
- User must re-authenticate on all devices

**Error Responses:**
- `401 Unauthorized`: Current password incorrect
- `400 Bad Request`: New password doesn't meet requirements

---

## 2. User Info Endpoint

### Get Current User Context
```http
GET /api/v1/user
Authorization: Bearer <access_token>
```

**Response:** `200 OK`
```json
{
  "user": {
    "id": "user_abc123",
    "email": "user@example.com",
    "name": "John Doe",
    "role": "user"
  },
  "tenant": {
    "id": "tenant_xyz789",
    "slug": "acme-corp",
    "name": "Acme Corporation"
  },
  "language": "hr",
  "features": ["narodne-novine", "financial-reports"],
  "permissions": ["chat:create", "chat:read", "chat:write", "chat:delete"]
}
```

---

## 3. Chat Management (tRPC)

**Note**: All chat endpoints use tRPC. Access via `/api/trpc`

### 3.1 List Chats
```typescript
// tRPC query
client.chats.list.query({
  feature?: string,
  visibility?: "private" | "tenant_shared" | "all",
  search?: string,
  limit?: number,  // default 20, max 100
  cursor?: string,
  sortBy?: "createdAt" | "updatedAt" | "lastMessageAt",
  sortOrder?: "asc" | "desc"  // default "desc"
})
```

**Response:**
```json
{
  "chats": [
    {
      "id": "chat_abc123",
      "title": "Narodne Novine Research",
      "feature": "narodne-novine",
      "visibility": "private",
      "tenantId": "tenant_456",
      "userId": "user_123",
      "language": "hr",
      "ragConfig": {...},
      "metadata": {...},
      "createdAt": "2024-01-01T12:00:00Z",
      "updatedAt": "2024-01-01T13:30:00Z",
      "lastMessageAt": "2024-01-01T13:30:00Z",
      "messageCount": 42
    }
  ],
  "pagination": {
    "hasMore": true,
    "nextCursor": "chat_xyz789",
    "totalCount": 156
  }
}
```

### 3.2 Create Chat
```typescript
// tRPC mutation
client.chats.create.mutate({
  title: string,
  feature: string,
  visibility?: "private" | "tenant_shared",  // default "private"
  ragConfig?: {
    language: string,
    maxDocuments: number,  // 1-20
    minConfidence: number,  // 0-1
    temperature: number  // 0-2
  },
  metadata?: {
    tags?: string[],
    description?: string
  }
})
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
  "language": "hr",
  "ragConfig": {
    "language": "hr",
    "maxDocuments": 5,
    "minConfidence": 0.7,
    "temperature": 0.7
  },
  "metadata": {
    "tags": ["research"],
    "description": ""
  },
  "createdAt": "2024-01-01T14:00:00Z",
  "updatedAt": "2024-01-01T14:00:00Z",
  "lastMessageAt": null,
  "messageCount": 0
}
```

**Error Responses:**
- `FORBIDDEN`: Feature not enabled for user

### 3.3 Get Chat Details
```typescript
// tRPC query
client.chats.getById.query({
  chatId: string
})
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
  "language": "hr",
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
  "lastMessageAt": "2024-01-01T13:30:00Z",
  "messageCount": 42
}
```

### 3.4 Update Chat
```typescript
// tRPC mutation
client.chats.update.mutate({
  chatId: string,
  updates: {
    title?: string,
    visibility?: "private" | "tenant_shared",
    ragConfig?: {
      language?: string,
      maxDocuments?: number,
      minConfidence?: number,
      temperature?: number
    },
    metadata?: {
      tags?: string[],
      description?: string
    }
  }
})
```

**Response:** Updated chat object

**Error Responses:**
- `NOT_FOUND`: Chat not found
- `FORBIDDEN`: Only owner can update

### 3.5 Delete Chat
```typescript
// tRPC mutation
client.chats.delete.mutate({
  chatId: string
})
```

**Response:**
```json
{
  "success": true
}
```

**Notes:**
- Messages cascade delete automatically
- Only owner can delete

---

## 4. Message Management (tRPC)

### 4.1 List Messages
```typescript
// tRPC query
client.messages.list.query({
  chatId: string,
  limit?: number,  // default 50, max 100
  cursor?: string,
  order?: "asc" | "desc"  // default "desc"
})
```

**Response:**
```json
{
  "messages": [
    {
      "id": "msg_asst124",
      "chatId": "chat_abc123",
      "role": "assistant",
      "content": "Based on the latest Narodne Novine...",
      "feature": "narodne-novine",
      "metadata": {
        "ragContext": {...},
        "model": "qwen2.5:7b",
        "tokensUsed": {...}
      },
      "status": "completed",
      "createdAt": "2024-01-01T14:00:01Z",
      "updatedAt": "2024-01-01T14:00:01Z"
    }
  ],
  "pagination": {
    "hasMore": true,
    "nextCursor": "msg_xyz789",
    "totalCount": 42
  }
}
```

### 4.2 Send Message
```typescript
// tRPC mutation
client.messages.send.mutate({
  chatId: string,
  content: string,  // 1-10000 chars
  ragConfig?: {
    maxDocuments?: number,  // 1-20
    minConfidence?: number,  // 0-1
    temperature?: number,  // 0-2
    language?: string
  }
})
```

**Response:**
```json
{
  "userMessage": {
    "id": "msg_user123",
    "chatId": "chat_abc123",
    "role": "user",
    "content": "What are the latest regulations?",
    "metadata": {
      "edited": false
    },
    "status": "completed",
    "createdAt": "2024-01-01T14:00:00Z"
  },
  "assistantMessage": {
    "id": "msg_asst124",
    "chatId": "chat_abc123",
    "role": "assistant",
    "content": "### Latest Regulations\n\nBased on...",
    "feature": "narodne-novine",
    "metadata": {
      "ragContext": {
        "documentsRetrieved": 5,
        "documentsUsed": 3,
        "confidence": 0.89,
        "searchTimeMs": 234,
        "sources": [...]
      },
      "model": "qwen2.5:7b",
      "provider": "ollama",
      "tokensUsed": {
        "input": 150,
        "output": 300,
        "total": 450
      },
      "responseTimeMs": 1250,
      "processingTaskId": "task_abc123"
    },
    "status": "completed",
    "createdAt": "2024-01-01T14:00:01Z"
  },
  "processingTask": {
    "id": "task_abc123",
    "status": "completed",
    "duration": 1250
  }
}
```

**Notes:**
- RAG processing happens synchronously
- Processing task tracks execution
- Automatic markdown preservation
- Error messages created on failure

### 4.3 Edit Message
```typescript
// tRPC mutation
client.messages.edit.mutate({
  messageId: string,
  content: string  // 1-10000 chars
})
```

**Response:** Updated message object

**Restrictions:**
- Only user messages can be edited
- Only owner can edit

### 4.4 Delete Message
```typescript
// tRPC mutation
client.messages.delete.mutate({
  messageId: string
})
```

**Response:**
```json
{
  "success": true
}
```

**Restrictions:**
- Only owner can delete

---

## 5. Utility Endpoints

### Health Check
```http
GET /api/v1/health
```

**Response:** `200 OK` or `503 Service Unavailable`
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

### API Info
```http
GET /api/v1/info
```

**Response:** `200 OK`
```json
{
  "version": "1.0.0",
  "environment": "development",
  "authMode": "jwt",
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

---

## Error Response Format

### Standard Error Structure
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid access token",
    "details": {
      "reason": "Token expired"
    }
  },
  "timestamp": "2024-01-01T14:00:00Z",
  "requestId": "req_abc123"
}
```

### Common Error Codes

| Status | Code | Description |
|--------|------|-------------|
| 400 | VALIDATION_ERROR | Invalid request parameters |
| 401 | UNAUTHORIZED | Missing or invalid authentication |
| 403 | FORBIDDEN | Insufficient permissions |
| 403 | FEATURE_NOT_ENABLED | Feature not available for user |
| 404 | NOT_FOUND | Resource does not exist |
| 409 | CONFLICT | Resource already exists |
| 429 | RATE_LIMITED | Too many requests |
| 500 | INTERNAL_ERROR | Server error |
| 503 | SERVICE_UNAVAILABLE | Temporary service issue |

---

## Security Considerations

### 1. JWT Token Security
- ✅ 15-minute access token expiration
- ✅ 7-day refresh token expiration
- ✅ Refresh token rotation on use
- ✅ Token revocation on password change
- ✅ Token revocation on logout

### 2. Password Security
- ✅ bcrypt hashing (12 rounds)
- ✅ Strong password requirements enforced
- ✅ Password change invalidates all sessions

### 3. Multi-tenant Isolation
- ✅ Automatic tenant filtering on all queries
- ✅ Tenant ID derived from JWT token
- ✅ No cross-tenant data leakage

### 4. Feature & Permission Control
- ✅ Feature access validated on every request
- ✅ Permission middleware for endpoint protection
- ✅ Detailed error responses for access denial

### 5. API Security Headers
- ⚠️ CORS configured (needs production review)
- ⚠️ Rate limiting defined (not yet implemented)
- ⚠️ HTTPS enforcement needed in production

---

## Implementation Notes

### Backend (IMPLEMENTED)
1. ✅ **JWT Authentication**: Full implementation with refresh tokens
2. ✅ **User Management**: Registration, login, profile, password change
3. ✅ **Multi-tenant**: Database isolation, feature/permission control
4. ✅ **Chat System**: Full CRUD with tRPC
5. ✅ **Message System**: RAG integration, markdown preservation
6. ✅ **Docker**: Full containerization with docker-compose

### Frontend (NOT IMPLEMENTED)
1. ❌ **Signup Page**: NO UI component exists
2. ❌ **Login Page**: NO UI component exists
3. ❌ **Profile Page**: NO UI component exists
4. ❌ **Password Reset**: NO UI component exists
5. ❌ **Email Verification**: NO UI component exists

### Planned Features (NOT IMPLEMENTED)
1. ❌ **WebSocket/Real-time**: Redis configured but not implemented
2. ❌ **Rate Limiting**: Config defined but not implemented
3. ❌ **Email Service**: Required for verification/reset flows
4. ❌ **Testing**: Framework installed but no tests written

---

## Docker Deployment

### Docker Compose Services
```yaml
services:
  postgres:      # PostgreSQL database
  redis:         # Redis for real-time (not yet used)
  weaviate:      # Vector database
  rag-service:   # Python RAG processing
  rag-api:       # Python RAG API
  web-api:       # Node.js API (THIS SERVICE)
  web-ui:        # React frontend
  nginx:         # Reverse proxy
```

### Environment Variables (web-api)
```bash
# Required
AUTH_MODE=jwt  # Must be "jwt"
JWT_SECRET=<secure-secret-min-32-chars>
DATABASE_URL=postgresql://user:pass@host:port/db
PYTHON_RAG_URL=http://rag_api:8082

# Optional
REDIS_URL=redis://redis:6379
CORS_ORIGIN=http://localhost:3001,http://localhost:5173
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX_REQUESTS=60
```

### Running with Docker
```bash
# Start all services
docker-compose up -d

# Apply database migrations
docker-compose exec web-api npx prisma migrate deploy

# View logs
docker-compose logs -f web-api

# Restart after changes
docker-compose restart web-api
```

---

## Migration from Mock to JWT

### Changes Required
1. ✅ Remove `AUTH_MODE=mock` from environment
2. ✅ Set `AUTH_MODE=jwt` in all environments
3. ⚠️ Remove mock authentication code (still exists)
4. ⚠️ Update documentation to remove mock references

### Breaking Changes
- ❌ Mock authentication no longer available
- ✅ Users must register/login to get JWT tokens
- ✅ All API calls require `Authorization: Bearer <token>` header

---

## Next Steps

### CRITICAL
1. **Remove mock authentication code completely**
2. **Create signup/login UI pages**
3. **Implement database migration strategy**

### HIGH PRIORITY
4. **Email verification flow**
5. **Password reset flow**
6. **Rate limiting implementation**

### MEDIUM PRIORITY
7. **WebSocket/real-time updates**
8. **Testing infrastructure**
9. **Tenant management UI**

---

**This specification reflects the ACTUAL implementation as of 2025-09-29. All endpoints are functional via API, but NO USER INTERFACE exists for authentication flows.**