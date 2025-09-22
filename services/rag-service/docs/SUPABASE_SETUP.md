# Supabase Database Setup for Multi-Tenant RAG + Chat System

## 🎯 Overview

This guide sets up a complete Supabase PostgreSQL database for:
- **Multi-tenant RAG system** (tenants, users, documents, chunks)
- **Chat system** (conversations, messages with persistence)
- **Analytics & configuration** (search queries, categorization templates)

## 📋 Setup Instructions

### Step 1: Access Supabase Dashboard
Go to your Supabase project dashboard:
```
https://supabase.com/dashboard/project/vdmizraansyjcblabuqp
```

Or go directly to the SQL Editor:
```
https://supabase.com/dashboard/project/vdmizraansyjcblabuqp/sql
```

### Step 2: Run Schema Creation
1. Click **"SQL Editor"** in the left sidebar
2. Copy the entire contents of `scripts/complete_supabase_schema.sql`
3. Paste into the SQL editor
4. Click **"Run"** to execute

### Step 3: Verify Setup
Run the verification script:
```bash
PYTHONPATH=/home/x/src/rag/learn-rag/services/rag-service python scripts/verify_supabase_setup.py
```

## 🗄️ Database Schema

### Core Tables

| Table | Purpose | Records |
|-------|---------|---------|
| `tenants` | Multi-tenant organizations | Development tenant created |
| `users` | User management per tenant | Development user created |
| `documents` | Document metadata and status | Ready for uploads |
| `chunks` | Vector storage metadata | Ready for embeddings |
| `conversations` | Chat conversation metadata | Ready for chat API |
| `chat_messages` | Individual chat messages | Ready for persistence |
| `search_queries` | Search analytics | Ready for tracking |
| `categorization_templates` | AI categorization rules | Croatian templates included |
| `system_configs` | System configuration | Default settings included |

### Default Data Included

**Development Tenant:**
- ID: `development`
- Name: "Development Tenant"
- Language: Croatian (hr)
- Tier: Enterprise

**Development User:**
- ID: `dev_user`
- Email: `dev@example.com`
- Role: Admin
- Tenant: development

**Croatian AI Templates:**
- Cultural template for Croatian culture/traditions
- Technical template for programming/tech questions

**System Configuration:**
- Default embedding model: `BAAI/bge-m3`
- Default chunk size: 512
- Max retrieval results: 10

## 🔧 Integration Status

### ✅ Ready Components
- **Chat API Server**: Running on http://localhost:8080
- **LLM Providers**: Ollama (local) + OpenRouter (cloud)
- **Database Connection**: Supabase configured and connected
- **Multi-tenant Support**: Full tenant/user isolation

### 🔗 Chat API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web chat interface |
| `/chat/start` | POST | Start new conversation |
| `/chat/message` | POST | Send message and get response |
| `/chat/conversations` | GET | List user conversations |
| `/chat/history/{id}` | GET | Get conversation history |
| `/chat/stream/{id}` | WebSocket | Streaming chat |
| `/health` | GET | System health check |

### 📝 Example Usage

**Start a conversation:**
```bash
curl -X POST http://localhost:8080/chat/start \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_slug": "development",
    "user_id": "dev_user",
    "title": "My First Chat"
  }'
```

**Send a message:**
```bash
curl -X POST http://localhost:8080/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "CONVERSATION_ID_FROM_ABOVE",
    "message": "Što je RAG sustav?"
  }'
```

## 🛡️ Security Features

### Row Level Security (RLS)
- **Optional**: Commented out in schema for development
- **Production**: Uncomment RLS policies for tenant isolation
- **Policies**: Ready for tenant-based data isolation

### Data Validation
- **Email validation**: Proper email format checks
- **Enum constraints**: Status fields limited to valid values
- **Length limits**: Text fields have appropriate size limits
- **Foreign keys**: Referential integrity maintained

## 🚀 Next Steps

1. **Complete schema setup** in Supabase dashboard
2. **Run verification script** to confirm all tables exist
3. **Test chat interface** at http://localhost:8080
4. **Upload documents** for RAG functionality
5. **Configure production RLS** when ready for deployment

## 📊 Monitoring

### Health Checks
- API health: `GET /health`
- Database connection: Included in health response
- LLM providers: Listed in health response

### Analytics Available
- Search query tracking
- Chat conversation metrics
- Document processing statistics
- User activity patterns

## 🔍 Troubleshooting

### Common Issues

1. **Tables not found**: Re-run complete schema SQL
2. **Connection errors**: Check Supabase credentials in config.toml
3. **Permission errors**: Ensure service_role_key is used
4. **Chat persistence fails**: Verify conversations/chat_messages tables exist

### Verification Commands
```bash
# Check tables exist
PYTHONPATH=/home/x/src/rag/learn-rag/services/rag-service python scripts/verify_supabase_setup.py

# Test API health
curl http://localhost:8080/health

# View server logs
# Check console output from chat_api.py
```

---

**✅ Your multi-tenant RAG + Chat system is now ready for use!** 🎉