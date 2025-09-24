# Incremental Build Roadmap: Simple to Advanced Multi-Tenant RAG Platform

**Date**: 2025-09-23
**Status**: Phased Implementation Strategy
**Authors**: Architecture Planning
**Context**: Start simple, build incrementally - RAG → UI/API → Elixir → Multi-tenancy

## Overview

This document outlines a pragmatic incremental approach to building the tri-language RAG platform. We start with the existing RAG system, add a simple UI/API layer, then introduce Elixir orchestration, and finally implement full multi-tenancy. Each phase is production-ready while being future-proof for the next evolution.

## Philosophy: Simple First, Future-Proof Always

### Core Principles
- **Start Simple**: Single user, basic functionality, minimal complexity
- **Future-Proof**: Architecture decisions support multi-tenancy without rebuilding
- **Incremental Value**: Each phase delivers working, valuable functionality
- **Non-Disruptive**: Never break what's already working
- **Tenant-Ready**: Behind-the-scenes preparation for multi-tenancy

## Phase 1: RAG Foundation Complete ✅

**Status**: Mostly Done
**Timeline**: Current
**Complexity**: ⭐⭐☆☆☆

### What You Have
- Python RAG system with BGE-M3 embeddings
- qwen2.5:7b-instruct for generation
- ChromaDB for vector storage
- Multi-language support (Croatian/English)
- Document processing and chunking

### What's Needed
- Ensure HTTP API endpoints for external integration
- Basic health check endpoint
- Simple configuration management
- Docker containerization for easy deployment

### Simple API Wrapper for RAG
```python
# Enhanced minimal API for Phase 2 integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid

app = FastAPI(title="RAG Service", version="1.0.0")

# Simple models (tenant-ready but not enforced yet)
class QueryRequest(BaseModel):
    text: str
    language: str = "auto"
    max_results: int = 5
    user_id: str = "default_user"  # Simple user ID
    # tenant_id hidden but stored for future

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    query_id: str
    language_detected: str

class HealthStatus(BaseModel):
    status: str
    version: str
    models_loaded: bool
    vector_db_connected: bool

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Simple RAG query - user-focused, tenant-ready"""
    query_id = str(uuid.uuid4())

    # Process with existing RAG system
    result = await process_rag_query(
        text=request.text,
        language=request.language,
        max_results=request.max_results,
        # Store with hidden tenant for future
        context={"user_id": request.user_id, "tenant_id": "default_tenant"}
    )

    return QueryResponse(
        answer=result.answer,
        confidence=result.confidence,
        sources=result.sources,
        query_id=query_id,
        language_detected=result.language
    )

@app.get("/api/v1/health", response_model=HealthStatus)
async def health_check():
    """Health endpoint for monitoring"""
    return HealthStatus(
        status="healthy",
        version="1.0.0",
        models_loaded=True,  # Check your models
        vector_db_connected=True  # Check ChromaDB
    )

@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile, user_id: str = "default_user"):
    """Simple document upload"""
    # Process document with hidden tenant structure
    result = await process_document(
        file=file,
        context={"user_id": user_id, "tenant_id": "default_tenant"}
    )
    return {"status": "success", "document_id": result.id}
```

## Phase 2: UI and API for Chat RAG Messaging

**Timeline**: Weeks 1-4
**Complexity**: ⭐⭐⭐☆☆
**Value**: Immediate usable chat interface

### Objective
Create a simple, polished chat interface for RAG interactions with a TypeScript API layer. Single user experience but architecturally ready for multi-user.

### 2.1: TypeScript API Layer (Week 1-2)

#### Simple tRPC Router
```typescript
// Simple chat-focused tRPC router
import { router, procedure } from '@trpc/server';
import { z } from 'zod';

// Simple schemas (future-proof but not complex)
const QuerySchema = z.object({
  text: z.string(),
  language: z.string().default('auto'),
  chatId: z.string().optional()
});

const CreateChatSchema = z.object({
  title: z.string(),
  userId: z.string().default('default_user') // Simple user concept
});

export const appRouter = router({
  // Chat management
  chat: router({
    create: procedure
      .input(CreateChatSchema)
      .mutation(async ({ input }) => {
        const chat = await prisma.chat.create({
          data: {
            title: input.title,
            userId: input.userId,
            tenantId: 'default_tenant', // Hidden but future-ready
            visibility: 'private'
          }
        });
        return chat;
      }),

    list: procedure
      .input(z.object({ userId: z.string().default('default_user') }))
      .query(async ({ input }) => {
        return await prisma.chat.findMany({
          where: { userId: input.userId },
          orderBy: { updatedAt: 'desc' }
        });
      }),

    get: procedure
      .input(z.object({ chatId: z.string() }))
      .query(async ({ input }) => {
        return await prisma.chat.findUnique({
          where: { id: input.chatId },
          include: {
            messages: {
              orderBy: { createdAt: 'asc' }
            }
          }
        });
      })
  }),

  // RAG operations
  rag: router({
    query: procedure
      .input(QuerySchema)
      .mutation(async ({ input }) => {
        // Call Python RAG service
        const ragResponse = await fetch('http://python-rag:8000/api/v1/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: input.text,
            language: input.language,
            user_id: 'default_user'
          })
        });

        const result = await ragResponse.json();

        // Store messages if chatId provided
        if (input.chatId) {
          // Store user message
          await prisma.message.create({
            data: {
              chatId: input.chatId,
              role: 'user',
              content: input.text
            }
          });

          // Store AI response
          await prisma.message.create({
            data: {
              chatId: input.chatId,
              role: 'assistant',
              content: result.answer,
              metadata: {
                confidence: result.confidence,
                sources: result.sources,
                language_detected: result.language_detected
              }
            }
          });
        }

        return result;
      }),

    uploadDocument: procedure
      .input(z.object({
        fileName: z.string(),
        content: z.string(),
        userId: z.string().default('default_user')
      }))
      .mutation(async ({ input }) => {
        // Forward to Python service
        const response = await fetch('http://python-rag:8000/api/v1/documents/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(input)
        });

        return await response.json();
      })
  })
});

export type AppRouter = typeof appRouter;
```

#### Simple Database Schema
```prisma
// Simple but future-proof Prisma schema
model Chat {
  id         String   @id @default(cuid())
  title      String
  userId     String   @map("user_id")     // Simple user concept
  tenantId   String   @map("tenant_id")   // Hidden but ready
  visibility String   @default("private") // Ready for sharing
  createdAt  DateTime @default(now()) @map("created_at")
  updatedAt  DateTime @updatedAt @map("updated_at")

  messages Message[]

  @@map("chats")
}

model Message {
  id        String   @id @default(cuid())
  chatId    String   @map("chat_id")
  role      String   // 'user' | 'assistant'
  content   String
  metadata  Json?    // RAG sources, confidence, etc.
  createdAt DateTime @default(now()) @map("created_at")

  chat Chat @relation(fields: [chatId], references: [id])

  @@map("messages")
}

// Ready for future phases but not enforced yet
model User {
  id       String @id @default(cuid())
  name     String @default("Default User")
  email    String @default("user@example.com")
  tenantId String @map("tenant_id") @default("default_tenant")

  @@map("users")
}

model Tenant {
  id       String @id @default(cuid())
  name     String @default("Default Tenant")
  slug     String @default("default") @unique
  settings Json?

  @@map("tenants")
}
```

### 2.2: React Chat Interface (Week 2-3)

#### Simple Chat Component
```typescript
// Simple but polished chat interface
import React, { useState, useEffect } from 'react';
import { trpc } from '../utils/trpc';

export function ChatInterface() {
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);

  const { data: chatList } = trpc.chat.list.useQuery({ userId: 'default_user' });
  const createChatMutation = trpc.chat.create.useMutation();
  const ragQueryMutation = trpc.rag.query.useMutation();

  const createNewChat = async () => {
    const chat = await createChatMutation.mutateAsync({
      title: 'New Chat',
      userId: 'default_user'
    });
    setActiveChat(chat.id);
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    try {
      await ragQueryMutation.mutateAsync({
        text: inputText,
        language: 'auto',
        chatId: activeChat
      });

      setInputText('');
      // Refresh chat to show new messages
      refetchActiveChat();
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      {/* Sidebar with chat list */}
      <div className="chat-sidebar">
        <button onClick={createNewChat} className="new-chat-btn">
          New Chat
        </button>

        <div className="chat-list">
          {chatList?.map(chat => (
            <div
              key={chat.id}
              className={`chat-item ${activeChat === chat.id ? 'active' : ''}`}
              onClick={() => setActiveChat(chat.id)}
            >
              {chat.title}
            </div>
          ))}
        </div>
      </div>

      {/* Main chat area */}
      <div className="chat-main">
        {activeChat ? (
          <>
            <ChatMessages chatId={activeChat} />

            <div className="chat-input">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Ask anything..."
                disabled={loading}
              />
              <button onClick={sendMessage} disabled={loading || !inputText.trim()}>
                {loading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </>
        ) : (
          <div className="empty-state">
            <p>Select a chat or create a new one to start</p>
          </div>
        )}
      </div>
    </div>
  );
}

function ChatMessages({ chatId }: { chatId: string }) {
  const { data: chat } = trpc.chat.get.useQuery({ chatId });

  return (
    <div className="messages">
      {chat?.messages.map(message => (
        <div key={message.id} className={`message ${message.role}`}>
          <div className="message-content">
            {message.content}
          </div>

          {message.role === 'assistant' && message.metadata && (
            <div className="message-metadata">
              <div className="confidence">
                Confidence: {Math.round(message.metadata.confidence * 100)}%
              </div>
              {message.metadata.sources && (
                <div className="sources">
                  Sources: {message.metadata.sources.join(', ')}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
```

### 2.3: Document Upload Interface (Week 3-4)

```typescript
// Simple document upload component
export function DocumentUpload() {
  const [uploading, setUploading] = useState(false);
  const uploadMutation = trpc.rag.uploadDocument.useMutation();

  const handleFileUpload = async (file: File) => {
    setUploading(true);

    try {
      const content = await file.text();
      await uploadMutation.mutateAsync({
        fileName: file.name,
        content: content,
        userId: 'default_user'
      });

      alert('Document uploaded successfully!');
    } catch (error) {
      alert('Upload failed: ' + error.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="document-upload">
      <h3>Upload Documents</h3>

      <div className="upload-area">
        <input
          type="file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFileUpload(file);
          }}
          accept=".txt,.md,.pdf"
          disabled={uploading}
        />

        {uploading && <p>Uploading...</p>}
      </div>
    </div>
  );
}
```

### Phase 2 Deliverables
- ✅ Working chat interface for RAG queries
- ✅ Chat history and conversation management
- ✅ Document upload capability
- ✅ TypeScript API with tRPC
- ✅ Simple database schema (future-proof)
- ✅ Responsive React interface
- ✅ Integration with existing Python RAG

## Phase 3: Elixir Orchestration Layer (Simple)

**Timeline**: Weeks 5-8
**Complexity**: ⭐⭐⭐⭐☆
**Value**: Fault tolerance, jobs, rate limiting, admin monitoring

### Objective
Add Elixir OTP as orchestration layer for reliability, background jobs, rate limiting, and feature flags. Keep it simple - no multi-tenancy yet, but future-ready.

### 3.1: Basic Elixir Phoenix Wrapper (Week 5-6)

#### Simple Phoenix Application
```elixir
# lib/rag_platform/application.ex
defmodule RagPlatform.Application do
  use Application

  def start(_type, _args) do
    children = [
      # Database and PubSub
      RagPlatform.Repo,
      {Phoenix.PubSub, name: RagPlatform.PubSub},

      # Core services (simple, no tenancy yet)
      RagPlatform.Services.RagHealthMonitor,
      RagPlatform.Services.JobCoordinator,
      RagPlatform.Services.RateLimiter,
      RagPlatform.Services.FeatureFlags,

      # Job processing
      {Oban, Application.fetch_env!(:rag_platform, Oban)},

      # Web endpoint
      RagPlatformWeb.Endpoint
    ]

    opts = [strategy: :one_for_one, name: RagPlatform.Supervisor]
    Supervisor.start_link(children, opts)
  end
end

# Simple health monitor for Python RAG service
defmodule RagPlatform.Services.RagHealthMonitor do
  use GenServer
  require Logger

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    schedule_health_check()
    {:ok, %{healthy: false, last_check: nil}}
  end

  def handle_info(:health_check, state) do
    health_status = check_python_rag_health()

    if health_status != state.healthy do
      Logger.info("RAG service health changed: #{health_status}")
      broadcast_health_change(health_status)
    end

    schedule_health_check()
    {:noreply, %{healthy: health_status, last_check: DateTime.utc_now()}}
  end

  defp check_python_rag_health do
    case HTTPoison.get("http://python-rag:8000/api/v1/health", [], timeout: 5000) do
      {:ok, %{status_code: 200}} -> true
      _ -> false
    end
  end

  defp broadcast_health_change(healthy) do
    Phoenix.PubSub.broadcast(
      RagPlatform.PubSub,
      "system:health",
      {:rag_health_changed, healthy}
    )
  end

  defp schedule_health_check do
    Process.send_after(self(), :health_check, 30_000) # Every 30 seconds
  end
end
```

#### Simple Job Coordination
```elixir
# Background job for document processing
defmodule RagPlatform.Workers.DocumentProcessor do
  use Oban.Worker, queue: :documents

  @impl Oban.Worker
  def perform(%Oban.Job{args: args}) do
    %{
      "user_id" => user_id,
      "file_name" => file_name,
      "content" => content
    } = args

    # Send to Python service for processing
    case process_document(user_id, file_name, content) do
      {:ok, result} ->
        broadcast_completion(user_id, result)
        :ok

      {:error, reason} ->
        broadcast_error(user_id, reason)
        {:error, reason}
    end
  end

  defp process_document(user_id, file_name, content) do
    payload = %{
      fileName: file_name,
      content: content,
      userId: user_id
    }

    case HTTPoison.post(
      "http://python-rag:8000/api/v1/documents/upload",
      Jason.encode!(payload),
      [{"Content-Type", "application/json"}],
      timeout: 60_000
    ) do
      {:ok, %{status_code: 200, body: body}} ->
        {:ok, Jason.decode!(body)}
      {:ok, %{status_code: status}} ->
        {:error, "HTTP #{status}"}
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp broadcast_completion(user_id, result) do
    Phoenix.PubSub.broadcast(
      RagPlatform.PubSub,
      "user:#{user_id}:documents",
      {:document_processed, result}
    )
  end

  defp broadcast_error(user_id, reason) do
    Phoenix.PubSub.broadcast(
      RagPlatform.PubSub,
      "user:#{user_id}:documents",
      {:document_failed, reason}
    )
  end
end

# Simple job scheduling
defmodule RagPlatform.Jobs do
  alias RagPlatform.Workers.DocumentProcessor

  def process_document(user_id, file_name, content) do
    %{
      user_id: user_id,
      file_name: file_name,
      content: content
    }
    |> DocumentProcessor.new()
    |> Oban.insert()
  end
end
```

### 3.2: Rate Limiting and Feature Flags (Week 6-7)

#### Simple Rate Limiter
```elixir
defmodule RagPlatform.Services.RateLimiter do
  use GenServer

  # Simple rate limiting - no tenancy yet, just per user
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def check_rate_limit(user_id, operation) do
    GenServer.call(__MODULE__, {:check_limit, user_id, operation})
  end

  def init(_opts) do
    # Simple ETS table for rate limiting
    table = :ets.new(:rate_limits, [:set, :public, :named_table])
    {:ok, %{table: table}}
  end

  def handle_call({:check_limit, user_id, operation}, _from, state) do
    key = {user_id, operation}
    current_time = System.monotonic_time(:second)

    limit_config = get_limit_config(operation)

    case :ets.lookup(:rate_limits, key) do
      [{^key, count, window_start}] ->
        if current_time - window_start >= limit_config.window do
          # Reset window
          :ets.insert(:rate_limits, {key, 1, current_time})
          {:reply, :ok, state}
        else
          if count >= limit_config.max_requests do
            {:reply, {:error, :rate_limited}, state}
          else
            :ets.insert(:rate_limits, {key, count + 1, window_start})
            {:reply, :ok, state}
          end
        end

      [] ->
        # First request in window
        :ets.insert(:rate_limits, {key, 1, current_time})
        {:reply, :ok, state}
    end
  end

  defp get_limit_config(:rag_query), do: %{max_requests: 100, window: 3600} # 100/hour
  defp get_limit_config(:document_upload), do: %{max_requests: 10, window: 3600} # 10/hour
  defp get_limit_config(_), do: %{max_requests: 1000, window: 3600} # Default
end
```

#### Simple Feature Flags
```elixir
defmodule RagPlatform.Services.FeatureFlags do
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def enabled?(flag_name, user_id \\ "default_user") do
    GenServer.call(__MODULE__, {:check_flag, flag_name, user_id})
  end

  def toggle_flag(flag_name, enabled) do
    GenServer.cast(__MODULE__, {:toggle_flag, flag_name, enabled})
  end

  def init(_opts) do
    # Simple in-memory flags - will move to DB later
    flags = %{
      "document_upload" => true,
      "advanced_rag" => false,
      "real_time_updates" => true,
      "debug_mode" => false
    }

    {:ok, %{flags: flags}}
  end

  def handle_call({:check_flag, flag_name, _user_id}, _from, state) do
    # Simple global flags for now - user-specific later
    enabled = Map.get(state.flags, flag_name, false)
    {:reply, enabled, state}
  end

  def handle_cast({:toggle_flag, flag_name, enabled}, state) do
    new_flags = Map.put(state.flags, flag_name, enabled)

    # Broadcast change
    Phoenix.PubSub.broadcast(
      RagPlatform.PubSub,
      "system:feature_flags",
      {:flag_changed, flag_name, enabled}
    )

    {:noreply, %{state | flags: new_flags}}
  end
end
```

### 3.3: Admin Dashboard with Phoenix LiveView (Week 7-8)

#### Simple Admin Dashboard
```elixir
defmodule RagPlatformWeb.AdminLive do
  use RagPlatformWeb, :live_view

  def mount(_params, _session, socket) do
    if connected?(socket) do
      # Subscribe to system updates
      Phoenix.PubSub.subscribe(RagPlatform.PubSub, "system:health")
      Phoenix.PubSub.subscribe(RagPlatform.PubSub, "system:feature_flags")
    end

    {:ok, assign(socket,
      rag_healthy: false,
      active_users: 0,
      feature_flags: %{},
      recent_jobs: []
    )}
  end

  def handle_info({:rag_health_changed, healthy}, socket) do
    {:noreply, assign(socket, rag_healthy: healthy)}
  end

  def handle_info({:flag_changed, flag_name, enabled}, socket) do
    new_flags = Map.put(socket.assigns.feature_flags, flag_name, enabled)
    {:noreply, assign(socket, feature_flags: new_flags)}
  end

  def handle_event("toggle_flag", %{"flag" => flag_name}, socket) do
    current_state = Map.get(socket.assigns.feature_flags, flag_name, false)
    RagPlatform.Services.FeatureFlags.toggle_flag(flag_name, !current_state)
    {:noreply, socket}
  end

  def render(assigns) do
    ~H"""
    <div class="admin-dashboard">
      <h1>RAG Platform Admin</h1>

      <!-- System Health -->
      <div class="health-section">
        <h2>System Health</h2>
        <div class="health-indicator">
          <span class={if @rag_healthy, do: "status-healthy", else: "status-unhealthy"}>
            RAG Service: <%= if @rag_healthy, do: "Healthy", else: "Unhealthy" %>
          </span>
        </div>
      </div>

      <!-- Feature Flags -->
      <div class="feature-flags-section">
        <h2>Feature Flags</h2>
        <%= for {flag_name, enabled} <- @feature_flags do %>
          <div class="flag-control">
            <label>
              <input
                type="checkbox"
                checked={enabled}
                phx-click="toggle_flag"
                phx-value-flag={flag_name}
              />
              <%= flag_name %>
            </label>
          </div>
        <% end %>
      </div>

      <!-- Recent Activity -->
      <div class="activity-section">
        <h2>Recent Jobs</h2>
        <div class="jobs-list">
          <%= for job <- @recent_jobs do %>
            <div class="job-item">
              <%= job.type %> - <%= job.status %>
            </div>
          <% end %>
        </div>
      </div>
    </div>
    """
  end
end
```

### 3.4: Integration with TypeScript API

#### Enhanced TypeScript API with Elixir
```typescript
// Enhanced tRPC router with Elixir integration
export const appRouter = router({
  chat: chatRouter, // Existing chat routes

  rag: router({
    query: procedure
      .input(QuerySchema)
      .mutation(async ({ input }) => {
        // Check rate limit via Elixir
        const rateLimitCheck = await fetch('http://elixir-platform:4000/api/rate-limit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: 'default_user',
            operation: 'rag_query'
          })
        });

        if (!rateLimitCheck.ok) {
          throw new TRPCError({
            code: 'TOO_MANY_REQUESTS',
            message: 'Rate limit exceeded'
          });
        }

        // Check feature flag
        const featureFlagResponse = await fetch('http://elixir-platform:4000/api/feature-flags/advanced_rag');
        const advancedRagEnabled = await featureFlagResponse.json();

        // Process query (existing logic)
        const result = await processRAGQuery(input, advancedRagEnabled.enabled);

        return result;
      }),

    uploadDocument: procedure
      .input(DocumentUploadSchema)
      .mutation(async ({ input }) => {
        // Queue via Elixir instead of direct processing
        const jobResponse = await fetch('http://elixir-platform:4000/api/jobs/document-processing', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: 'default_user',
            file_name: input.fileName,
            content: input.content
          })
        });

        const jobResult = await jobResponse.json();
        return { jobId: jobResult.job_id, status: 'queued' };
      })
  }),

  // Admin endpoints
  admin: router({
    health: procedure
      .query(async () => {
        const response = await fetch('http://elixir-platform:4000/api/admin/health');
        return await response.json();
      }),

    featureFlags: procedure
      .query(async () => {
        const response = await fetch('http://elixir-platform:4000/api/admin/feature-flags');
        return await response.json();
      })
  })
});
```

### Phase 3 Deliverables
- ✅ Elixir Phoenix orchestration layer
- ✅ Background job processing with Oban
- ✅ Simple rate limiting (per user)
- ✅ Feature flags system
- ✅ Phoenix LiveView admin dashboard
- ✅ Health monitoring and alerting
- ✅ Integration with existing TypeScript API
- ✅ Fault tolerance and automatic recovery

## Phase 4: Multi-Tenancy and Advanced Orchestration

**Timeline**: Weeks 9-16
**Complexity**: ⭐⭐⭐⭐⭐
**Value**: Enterprise-ready multi-tenant platform

### Objective
Transform the simple single-user system into a fully multi-tenant platform with advanced orchestration, user management, and tenant isolation.

### 4.1: Multi-Tenant Database Migration (Week 9-10)

#### Enhanced Database Schema
```sql
-- Enhanced multi-tenant schema
ALTER TABLE tenants DROP CONSTRAINT IF EXISTS tenants_slug_key;
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS plan VARCHAR(50) DEFAULT 'free';
ALTER TABLE tenants ADD COLUMN IF NOT EXISTS limits JSONB DEFAULT '{}';

-- Enhanced users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'member';
ALTER TABLE users ADD COLUMN IF NOT EXISTS permissions JSONB DEFAULT '[]';
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMPTZ;

-- Chat permissions for sharing
CREATE TABLE IF NOT EXISTS chat_permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permission VARCHAR(20) NOT NULL, -- 'read', 'write', 'admin'
    granted_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chat_id, user_id, permission)
);

-- Rate limit tracking per tenant
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    operation VARCHAR(100) NOT NULL,
    count INTEGER DEFAULT 0,
    window_start TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature flags per tenant
CREATE TABLE IF NOT EXISTS feature_flags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id),
    flag_name VARCHAR(100) NOT NULL,
    enabled BOOLEAN DEFAULT false,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, flag_name)
);

-- Enable Row Level Security
ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY tenant_isolation_chats ON chats
    FOR ALL TO authenticated
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

CREATE POLICY user_access_chats ON chats
    FOR ALL TO authenticated
    USING (
        user_id = current_setting('app.current_user_id')::UUID OR
        visibility = 'tenant_shared' OR
        EXISTS (
            SELECT 1 FROM chat_permissions cp
            WHERE cp.chat_id = chats.id
            AND cp.user_id = current_setting('app.current_user_id')::UUID
        )
    );
```

### 4.2: Advanced Elixir Multi-Tenant Orchestration (Week 11-12)

#### Tenant-Aware Services
```elixir
defmodule RagPlatform.Services.TenantManager do
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_tenant_context(tenant_id) do
    GenServer.call(__MODULE__, {:get_tenant, tenant_id})
  end

  def create_tenant(tenant_params) do
    GenServer.call(__MODULE__, {:create_tenant, tenant_params})
  end

  def init(_opts) do
    # Load tenant configurations
    tenants = load_tenant_configs()
    {:ok, %{tenants: tenants}}
  end

  def handle_call({:get_tenant, tenant_id}, _from, state) do
    tenant = Map.get(state.tenants, tenant_id)
    {:reply, tenant, state}
  end

  def handle_call({:create_tenant, params}, _from, state) do
    # Create tenant in database
    {:ok, tenant} = create_tenant_in_db(params)

    # Setup tenant resources
    setup_tenant_resources(tenant)

    # Update state
    new_tenants = Map.put(state.tenants, tenant.id, tenant)

    {:reply, {:ok, tenant}, %{state | tenants: new_tenants}}
  end

  defp setup_tenant_resources(tenant) do
    # Setup ChromaDB collections
    setup_tenant_vector_collections(tenant.id)

    # Initialize feature flags
    initialize_tenant_feature_flags(tenant.id)

    # Setup rate limiting
    initialize_tenant_rate_limits(tenant.id)
  end
end

# Enhanced multi-tenant rate limiter
defmodule RagPlatform.Services.TenantRateLimiter do
  use GenServer

  def check_tenant_rate_limit(tenant_id, user_id, operation) do
    GenServer.call(__MODULE__, {:check_limit, tenant_id, user_id, operation})
  end

  def handle_call({:check_limit, tenant_id, user_id, operation}, _from, state) do
    # Get tenant limits
    tenant_limits = get_tenant_limits(tenant_id)
    user_limits = get_user_limits(tenant_id, user_id)

    # Check both tenant and user limits
    with :ok <- check_limit(tenant_limits, operation, "tenant_#{tenant_id}"),
         :ok <- check_limit(user_limits, operation, "user_#{user_id}") do
      {:reply, :ok, state}
    else
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  defp get_tenant_limits(tenant_id) do
    # Query tenant plan and get limits
    case RagPlatform.Tenants.get_tenant_plan(tenant_id) do
      %{plan: "free"} -> %{
        rag_query: %{max: 100, window: 3600},
        document_upload: %{max: 5, window: 3600}
      }
      %{plan: "pro"} -> %{
        rag_query: %{max: 1000, window: 3600},
        document_upload: %{max: 50, window: 3600}
      }
      _ -> %{} # Default limits
    end
  end
end
```

### 4.3: Advanced TypeScript Multi-Tenant API (Week 12-13)

#### Multi-Tenant tRPC Router
```typescript
// Enhanced multi-tenant tRPC router
import { initTRPC, TRPCError } from '@trpc/server';
import { CreateExpressContextOptions } from '@trpc/server/adapters/express';

// Create context with tenant/user info
export const createContext = async ({ req }: CreateExpressContextOptions) => {
  const token = req.headers.authorization?.replace('Bearer ', '');

  if (!token) {
    throw new TRPCError({
      code: 'UNAUTHORIZED',
      message: 'No authorization token provided'
    });
  }

  const decoded = verifyJWT(token);

  return {
    user: {
      id: decoded.userId,
      tenantId: decoded.tenantId,
      role: decoded.role
    },
    prisma: prisma,
    elixir: elixirClient
  };
};

const t = initTRPC.context<typeof createContext>().create();

export const appRouter = t.router({
  // Tenant management
  tenant: t.router({
    get: t.procedure
      .query(async ({ ctx }) => {
        return await ctx.prisma.tenant.findUnique({
          where: { id: ctx.user.tenantId }
        });
      }),

    update: t.procedure
      .input(UpdateTenantSchema)
      .mutation(async ({ input, ctx }) => {
        // Check admin permissions
        if (ctx.user.role !== 'admin') {
          throw new TRPCError({
            code: 'FORBIDDEN',
            message: 'Admin access required'
          });
        }

        return await ctx.prisma.tenant.update({
          where: { id: ctx.user.tenantId },
          data: input
        });
      }),

    members: t.procedure
      .query(async ({ ctx }) => {
        return await ctx.prisma.user.findMany({
          where: { tenantId: ctx.user.tenantId }
        });
      }),

    inviteUser: t.procedure
      .input(InviteUserSchema)
      .mutation(async ({ input, ctx }) => {
        // Send invitation via Elixir
        await ctx.elixir.inviteUser({
          tenantId: ctx.user.tenantId,
          email: input.email,
          role: input.role,
          invitedBy: ctx.user.id
        });

        return { status: 'invited' };
      })
  }),

  // Enhanced chat with multi-tenancy
  chat: t.router({
    create: t.procedure
      .input(CreateChatSchema)
      .mutation(async ({ input, ctx }) => {
        const chat = await ctx.prisma.chat.create({
          data: {
            title: input.title,
            userId: ctx.user.id,
            tenantId: ctx.user.tenantId,
            visibility: input.visibility || 'private'
          }
        });

        return chat;
      }),

    list: t.procedure
      .input(z.object({
        scope: z.enum(['my', 'shared', 'all']).default('my')
      }))
      .query(async ({ input, ctx }) => {
        const where: any = {
          tenantId: ctx.user.tenantId
        };

        if (input.scope === 'my') {
          where.userId = ctx.user.id;
        } else if (input.scope === 'shared') {
          where.visibility = 'tenant_shared';
        }
        // 'all' includes both my and shared

        return await ctx.prisma.chat.findMany({
          where,
          orderBy: { updatedAt: 'desc' },
          include: {
            user: { select: { id: true, name: true } }
          }
        });
      }),

    share: t.procedure
      .input(ShareChatSchema)
      .mutation(async ({ input, ctx }) => {
        // Check ownership
        const chat = await ctx.prisma.chat.findFirst({
          where: {
            id: input.chatId,
            userId: ctx.user.id,
            tenantId: ctx.user.tenantId
          }
        });

        if (!chat) {
          throw new TRPCError({
            code: 'NOT_FOUND',
            message: 'Chat not found or access denied'
          });
        }

        // Add permissions
        await ctx.prisma.chatPermission.createMany({
          data: input.userIds.map(userId => ({
            chatId: input.chatId,
            userId: userId,
            permission: input.permission,
            grantedBy: ctx.user.id
          }))
        });

        return { status: 'shared' };
      })
  }),

  // Enhanced RAG with tenant context
  rag: t.router({
    query: t.procedure
      .input(QuerySchema)
      .mutation(async ({ input, ctx }) => {
        // Check rate limits via Elixir
        const rateLimitResult = await ctx.elixir.checkRateLimit({
          tenantId: ctx.user.tenantId,
          userId: ctx.user.id,
          operation: 'rag_query'
        });

        if (!rateLimitResult.allowed) {
          throw new TRPCError({
            code: 'TOO_MANY_REQUESTS',
            message: `Rate limit exceeded: ${rateLimitResult.message}`
          });
        }

        // Process with tenant context
        const result = await processRAGQuery({
          ...input,
          tenantId: ctx.user.tenantId,
          userId: ctx.user.id
        });

        // Store messages if chatId provided
        if (input.chatId) {
          await storeMessages(input.chatId, input.text, result, ctx.user.id);
        }

        return result;
      })
  })
});
```

### 4.4: Advanced Admin Interface (Week 14-15)

#### Multi-Tenant Admin Dashboard
```elixir
defmodule RagPlatformWeb.AdminLive.TenantsLive do
  use RagPlatformWeb, :live_view

  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(RagPlatform.PubSub, "admin:tenants")
    end

    tenants = list_tenants_with_stats()

    {:ok, assign(socket,
      tenants: tenants,
      selected_tenant: nil,
      tenant_stats: %{}
    )}
  end

  def handle_event("select_tenant", %{"tenant_id" => tenant_id}, socket) do
    tenant = Enum.find(socket.assigns.tenants, &(&1.id == tenant_id))
    stats = get_tenant_detailed_stats(tenant_id)

    {:noreply, assign(socket,
      selected_tenant: tenant,
      tenant_stats: stats
    )}
  end

  def handle_event("suspend_tenant", %{"tenant_id" => tenant_id}, socket) do
    case RagPlatform.Tenants.suspend_tenant(tenant_id) do
      {:ok, _tenant} ->
        updated_tenants = update_tenant_in_list(socket.assigns.tenants, tenant_id, :suspended)
        {:noreply, assign(socket, tenants: updated_tenants)}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to suspend tenant: #{reason}")}
    end
  end

  def render(assigns) do
    ~H"""
    <div class="admin-tenants">
      <h1>Tenant Management</h1>

      <div class="tenants-overview">
        <div class="tenant-list">
          <%= for tenant <- @tenants do %>
            <div
              class={["tenant-card", (if @selected_tenant && @selected_tenant.id == tenant.id, do: "selected", else: "")]}
              phx-click="select_tenant"
              phx-value-tenant_id={tenant.id}
            >
              <h3><%= tenant.name %></h3>
              <p>Plan: <%= tenant.plan %></p>
              <p>Status: <%= tenant.status %></p>
              <p>Users: <%= tenant.user_count %></p>
              <p>Usage: <%= tenant.monthly_queries %> queries</p>
            </div>
          <% end %>
        </div>

        <%= if @selected_tenant do %>
          <div class="tenant-details">
            <h2><%= @selected_tenant.name %></h2>

            <div class="tenant-actions">
              <button
                phx-click="suspend_tenant"
                phx-value-tenant_id={@selected_tenant.id}
                data-confirm="Are you sure?"
              >
                Suspend Tenant
              </button>
            </div>

            <div class="tenant-stats">
              <h3>Statistics</h3>
              <div class="stats-grid">
                <div class="stat">
                  <span class="stat-value"><%= @tenant_stats.total_queries || 0 %></span>
                  <span class="stat-label">Total Queries</span>
                </div>

                <div class="stat">
                  <span class="stat-value"><%= @tenant_stats.active_users || 0 %></span>
                  <span class="stat-label">Active Users</span>
                </div>

                <div class="stat">
                  <span class="stat-value"><%= @tenant_stats.documents_processed || 0 %></span>
                  <span class="stat-label">Documents</span>
                </div>
              </div>
            </div>

            <div class="feature-flags">
              <h3>Feature Flags</h3>
              <%= for {flag_name, config} <- @tenant_stats.feature_flags || %{} do %>
                <div class="flag-control">
                  <label>
                    <input
                      type="checkbox"
                      checked={config.enabled}
                      phx-click="toggle_tenant_flag"
                      phx-value-tenant_id={@selected_tenant.id}
                      phx-value-flag={flag_name}
                    />
                    <%= flag_name %>
                  </label>
                </div>
              <% end %>
            </div>
          </div>
        <% end %>
      </div>
    </div>
    """
  end

  defp list_tenants_with_stats do
    # Query tenants with basic stats
    RagPlatform.Repo.all(
      from t in RagPlatform.Tenants.Tenant,
      left_join: u in assoc(t, :users),
      group_by: t.id,
      select: %{
        id: t.id,
        name: t.name,
        plan: t.plan,
        status: t.status,
        user_count: count(u.id),
        created_at: t.created_at
      }
    )
  end

  defp get_tenant_detailed_stats(tenant_id) do
    # Get detailed statistics for a tenant
    %{
      total_queries: get_tenant_query_count(tenant_id),
      active_users: get_tenant_active_users(tenant_id),
      documents_processed: get_tenant_document_count(tenant_id),
      feature_flags: get_tenant_feature_flags(tenant_id)
    }
  end
end
```

### 4.5: Production Deployment (Week 15-16)

#### Kubernetes Deployment
```yaml
# k8s/production/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-platform-prod

---
# k8s/production/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: rag-platform-prod
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: rag_platform_prod
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
# k8s/production/elixir-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elixir-platform
  namespace: rag-platform-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: elixir-platform
  template:
    metadata:
      labels:
        app: elixir-platform
    spec:
      containers:
      - name: elixir-platform
        image: rag-platform/elixir:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: SECRET_KEY_BASE
          valueFrom:
            secretKeyRef:
              name: elixir-secret
              key: secret_key_base
        - name: PYTHON_RAG_URL
          value: "http://python-rag-service:8000"
        ports:
        - containerPort: 4000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 4000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 4000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/production/python-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: python-rag
  namespace: rag-platform-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: python-rag
  template:
    metadata:
      labels:
        app: python-rag
    spec:
      containers:
      - name: python-rag
        image: rag-platform/python-rag:latest
        env:
        - name: CHROMA_HOST
          value: "chromadb-service"
        - name: ELIXIR_API_URL
          value: "http://elixir-platform-service:4000"
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc

---
# k8s/production/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-platform-ingress
  namespace: rag-platform-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.ragplatform.com
    secretName: rag-platform-tls
  rules:
  - host: api.ragplatform.com
    http:
      paths:
      - path: /api/v1/rag
        pathType: Prefix
        backend:
          service:
            name: python-rag-service
            port:
              number: 8000
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: elixir-platform-service
            port:
              number: 4000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: typescript-api-service
            port:
              number: 3000
```

### Phase 4 Deliverables
- ✅ Complete multi-tenant database schema with RLS
- ✅ Advanced Elixir tenant management and orchestration
- ✅ Multi-tenant TypeScript API with proper authorization
- ✅ Advanced admin dashboard for tenant management
- ✅ Production-ready Kubernetes deployment
- ✅ Comprehensive monitoring and alerting
- ✅ User invitation and management system
- ✅ Advanced feature flags per tenant
- ✅ Enterprise-grade rate limiting and quotas

## Development Timeline Summary

| Phase | Duration | Complexity | Key Value |
|-------|----------|------------|-----------|
| Phase 1: RAG Foundation | Current | ⭐⭐☆☆☆ | Working RAG system |
| Phase 2: UI and API | 4 weeks | ⭐⭐⭐☆☆ | Usable chat interface |
| Phase 3: Elixir Orchestration | 4 weeks | ⭐⭐⭐⭐☆ | Fault tolerance & jobs |
| Phase 4: Multi-Tenancy | 8 weeks | ⭐⭐⭐⭐⭐ | Enterprise platform |

**Total Timeline**: 16 weeks (4 months) from current state to full enterprise platform

## Technical Stack Evolution

### Phase 2 Stack
```
Frontend: React + TypeScript + tRPC
API: Node.js + TypeScript + tRPC + Prisma
RAG: Python + FastAPI + ChromaDB
Database: PostgreSQL (simple schema)
```

### Phase 3 Stack
```
Frontend: React + TypeScript + tRPC
API: Node.js + TypeScript + tRPC + Prisma
Orchestration: Elixir + Phoenix + Oban
RAG: Python + FastAPI + ChromaDB
Database: PostgreSQL
Cache/Jobs: Redis
```

### Phase 4 Stack
```
Frontend: React + TypeScript + tRPC
API: Node.js + TypeScript + tRPC + Prisma
Orchestration: Elixir + Phoenix + Oban (Multi-tenant)
RAG: Python + FastAPI + ChromaDB (Tenant-isolated)
Database: PostgreSQL (RLS + Multi-tenant)
Cache/Jobs: Redis (Tenant-aware)
Monitoring: Elixir Telemetry + Grafana
```

## Key Benefits of This Approach

### 1. Incremental Value Delivery
- **Phase 2**: Immediate usable chat interface
- **Phase 3**: Production reliability and monitoring
- **Phase 4**: Enterprise-ready multi-tenant platform

### 2. Risk Mitigation
- **Non-disruptive**: Each phase builds on the previous
- **Working system**: Always have a functioning product
- **Rollback capability**: Can revert to previous phase if needed

### 3. Future-Proof Architecture
- **Hidden multi-tenancy**: Prepared from Phase 2
- **Scalable design**: Can handle enterprise load
- **Technology flexibility**: Can swap components as needed

### 4. Development Efficiency
- **AI-friendly**: Clear patterns for AI assistance
- **Incremental complexity**: Learn and adapt as you build
- **Proven patterns**: Each phase uses established best practices

## Implementation Recommendations

### Start with Phase 2 Immediately
- Build on your existing RAG system
- Create simple but polished chat interface
- Establish TypeScript API patterns
- Set up basic database schema (future-proof)

### Phase 3 When Ready for Production
- Add Elixir when you need fault tolerance
- Implement when background jobs become necessary
- Include when monitoring becomes critical

### Phase 4 When Scaling
- Multi-tenancy when you have multiple customers
- Advanced features when business demands them
- Enterprise features when revenue justifies complexity

This approach gives you a clear path from where you are now to a world-class enterprise platform, with each step delivering immediate value while building toward the ultimate vision.

**Ready to start with Phase 2?** 🚀