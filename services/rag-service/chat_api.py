"""
Chat Web API Server
Separate FastAPI server for testing the new chat system.
Independent of the existing RAG system for clean testing.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.chat.chat_service import create_chat_service
from src.database.factory import create_database_provider
from src.utils.config_loader import get_shared_config
from src.utils.logging_factory import get_system_logger
from src.query.query_classifier import create_query_classifier


from src.utils.json_logging import write_debug_json


# Request/Response Models
class StartChatRequest(BaseModel):
    tenant_slug: str = "development"
    user_id: str = "web_user"
    title: Optional[str] = None


class SendMessageRequest(BaseModel):
    conversation_id: str
    message: str
    language: str = "hr"
    tenant_slug: str = "development"
    user_id: str = "dev_user"
    max_rag_results: int = 3
    system_prompt: Optional[str] = None
    rag_context: Optional[List[str]] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # New scope parameter for different data sources
    scope: str = "user"  # "user", "narodne_novine", "tenant" (future)


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    content: str
    model: str
    provider: str
    usage: Dict[str, int]


class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


class QueryClassifyRequest(BaseModel):
    query: str
    language: str = "hr"


class QueryClassifyResponse(BaseModel):
    query_type: str
    confidence: float
    reasoning: str
    detected_keywords: List[str]
    should_include_user_docs: bool
    should_include_narodne_novine: bool


# Global chat service instance
chat_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup chat service."""
    global chat_service

    logger = get_system_logger()
    logger.info("chat_api", "startup", "Initializing chat API server")

    try:
        # Load configuration sections manually
        from src.utils.config_loader import get_config_section

        # Load required config sections
        database_config = get_config_section("config", "database")
        ollama_config = get_config_section("config", "ollama")
        openrouter_config = get_config_section("config", "openrouter")
        llm_config_section = get_config_section("config", "llm")

        # Create database provider
        db_provider = create_database_provider(database_config)
        await db_provider.initialize(database_config)

        # Create LLM configuration
        llm_config = {
            "ollama": ollama_config,
            "openrouter": openrouter_config,
            "primary_provider": llm_config_section["primary_provider"],
            "fallback_order": llm_config_section["fallback_order"]
        }

        # Create chat service
        chat_service = create_chat_service(llm_config, db_provider)
        await chat_service.initialize()

        logger.info("chat_api", "startup", "Chat API server ready")
        yield

    except Exception as e:
        logger.error("chat_api", "startup", f"Failed to initialize: {e}")
        raise
    finally:
        logger.info("chat_api", "shutdown", "Chat API server shutdown")


# Create FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="Chat API with LLM providers and persistence",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_chat_interface():
    """Serve simple chat HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Chat Interface</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
            .message { margin-bottom: 10px; padding: 8px; border-radius: 4px; }
            .user-message { background-color: #e3f2fd; text-align: right; }
            .assistant-message { background-color: #f5f5f5; }
            .system-message { background-color: #fff3e0; font-style: italic; }
            .input-area { display: flex; gap: 10px; }
            .input-area input { flex: 1; padding: 8px; }
            .input-area button { padding: 8px 16px; }
            .conversation-list { margin-bottom: 20px; }
            .conversation-item { padding: 8px; border: 1px solid #ddd; margin-bottom: 5px; cursor: pointer; }
            .conversation-item:hover { background-color: #f0f0f0; }
            .controls { margin-bottom: 20px; }
            .status { margin-top: 10px; font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <h1>RAG Chat Interface</h1>

        <div class="controls">
            <button onclick="startNewConversation()">New Conversation</button>
            <button onclick="loadConversations()">Load Conversations</button>
            <span class="status" id="status">Ready</span>
        </div>

        <div id="conversations" class="conversation-list" style="display: none;">
            <h3>Your Conversations</h3>
            <div id="conversation-list"></div>
        </div>

        <div id="chat-area" style="display: none;">
            <h3 id="chat-title">New Chat</h3>

            <div class="scope-selector" style="margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;">
                <label for="scopeSelect">Data Source:</label>
                <select id="scopeSelect" style="margin-left: 8px; padding: 4px;">
                    <option value="user">Personal Documents</option>
                    <option value="narodne_novine">Narodne Novine</option>
                    <option value="tenant">Tenant Documents (Future)</option>
                </select>
                <span id="scopeStatus" style="margin-left: 8px; font-size: 12px; color: #666;">Using: Personal Documents</span>
            </div>

            <div id="messages" class="chat-container"></div>
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Type your message..."
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            let currentConversationId = null;

            function setStatus(message) {
                document.getElementById('status').textContent = message;
            }

            function updateScopeStatus() {
                const scopeSelect = document.getElementById('scopeSelect');
                const scopeStatus = document.getElementById('scopeStatus');
                const scopeTexts = {
                    'user': 'Personal Documents',
                    'narodne_novine': 'Narodne Novine',
                    'tenant': 'Tenant Documents (Future)'
                };
                scopeStatus.textContent = 'Using: ' + scopeTexts[scopeSelect.value];
            }

            // Add event listener for scope changes
            document.addEventListener('DOMContentLoaded', function() {
                const scopeSelect = document.getElementById('scopeSelect');
                if (scopeSelect) {
                    scopeSelect.addEventListener('change', updateScopeStatus);
                }
            });

            async function startNewConversation() {
                setStatus('Starting new conversation...');
                try {
                    const response = await fetch('/chat/start', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            tenant_slug: 'development',
                            user_id: 'web_user',
                            title: 'New Chat'
                        })
                    });

                    const conversation = await response.json();
                    currentConversationId = conversation.conversation_id;

                    document.getElementById('chat-title').textContent = conversation.title;
                    document.getElementById('messages').innerHTML = '';
                    document.getElementById('chat-area').style.display = 'block';
                    document.getElementById('conversations').style.display = 'none';

                    setStatus('Ready');
                } catch (error) {
                    setStatus('Error: ' + error.message);
                }
            }

            async function sendMessage() {
                if (!currentConversationId) {
                    alert('Please start a conversation first');
                    return;
                }

                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;

                // Add user message to chat
                addMessage('user', message);
                input.value = '';
                setStatus('Generating response...');

                try {
                    const scopeSelect = document.getElementById('scopeSelect');
                    const currentScope = scopeSelect ? scopeSelect.value : 'user';

                    const response = await fetch('/chat/message', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            conversation_id: currentConversationId,
                            message: message,
                            scope: currentScope
                        })
                    });

                    const result = await response.json();
                    addMessage('assistant', result.content, `${result.model} (${result.provider})`);
                    setStatus('Ready');

                } catch (error) {
                    addMessage('system', 'Error: ' + error.message);
                    setStatus('Error occurred');
                }
            }

            function addMessage(role, content, meta = '') {
                const messages = document.getElementById('messages');
                const div = document.createElement('div');
                div.className = `message ${role}-message`;

                const roleLabel = role === 'user' ? 'You' :
                                role === 'assistant' ? 'Assistant' : 'System';
                const metaText = meta ? ` (${meta})` : '';

                div.innerHTML = `<strong>${roleLabel}${metaText}:</strong><br>${content}`;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }

            async function loadConversations() {
                setStatus('Loading conversations...');
                try {
                    const response = await fetch('/chat/conversations?tenant_slug=development&user_id=web_user');
                    const conversations = await response.json();

                    const listDiv = document.getElementById('conversation-list');
                    listDiv.innerHTML = '';

                    conversations.forEach(conv => {
                        const div = document.createElement('div');
                        div.className = 'conversation-item';
                        div.onclick = () => openConversation(conv.conversation_id, conv.title);
                        div.innerHTML = `
                            <strong>${conv.title}</strong><br>
                            <small>${conv.message_count} messages, ${new Date(conv.updated_at * 1000).toLocaleString()}</small>
                        `;
                        listDiv.appendChild(div);
                    });

                    document.getElementById('conversations').style.display = 'block';
                    document.getElementById('chat-area').style.display = 'none';
                    setStatus('Ready');

                } catch (error) {
                    setStatus('Error loading conversations: ' + error.message);
                }
            }

            async function openConversation(conversationId, title) {
                currentConversationId = conversationId;
                document.getElementById('chat-title').textContent = title;
                document.getElementById('messages').innerHTML = '';

                // Load conversation history
                try {
                    const response = await fetch(`/chat/history/${conversationId}`);
                    const messages = await response.json();

                    messages.forEach(msg => {
                        addMessage(msg.role, msg.content);
                    });

                    document.getElementById('chat-area').style.display = 'block';
                    document.getElementById('conversations').style.display = 'none';
                    setStatus('Ready');

                } catch (error) {
                    setStatus('Error loading history: ' + error.message);
                }
            }

            // Load conversations on page load
            window.onload = () => {
                setStatus('Chat interface ready');
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/chat/start")
async def start_conversation(request: StartChatRequest):
    """Start new chat conversation."""
    logger = get_system_logger()
    logger.info("chat_api", "start_conversation", f"Starting chat for {request.user_id}")

    try:
        conversation = await chat_service.start_conversation(
            tenant_slug=request.tenant_slug,
            user_id=request.user_id,
            title=request.title
        )

        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.get_created_at_timestamp()
        }

    except Exception as e:
        logger.error("chat_api", "start_conversation", f"Failed to start conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/message")
async def send_message(request: SendMessageRequest):
    """Send message with RAG document search enabled by default."""
    import json
    import os
    from datetime import datetime

    logger = get_system_logger()
    logger.info("chat_api", "send_message", f"RAG message to {request.conversation_id}: {len(request.message)} chars")

    # Log and store Chat API request details
    chat_api_request = {
        "timestamp": datetime.now().isoformat(),
        "type": "CHAT_API_REQUEST",
        "conversation_id": request.conversation_id,
        "message": request.message,
        "message_length": len(request.message),
        "language": request.language,
        "tenant_slug": request.tenant_slug,
        "user_id": request.user_id,
        "max_rag_results": request.max_rag_results,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "model": request.model,
        "system_prompt": request.system_prompt
    }

    # Store Chat API request to file
    os.makedirs("./logs/chat_debug", exist_ok=True)
    chat_request_file = f"./logs/chat_debug/{datetime.now().strftime('%Y%m%d_%H%M%S')}_chat_api_request.json"
    with open(chat_request_file, 'w', encoding='utf-8') as f:
        json.dump(chat_api_request, f, indent=2, ensure_ascii=False)

    # Store pure API request (clean format as per docs)
    pure_api_request = {
        "conversation_id": request.conversation_id,
        "message": request.message,
        "tenant_slug": request.tenant_slug,
        "user_id": request.user_id
    }
    pure_request_file = f"./logs/chat_debug/{datetime.now().strftime('%Y%m%d_%H%M%S')}_pure_api_request.json"
    with open(pure_request_file, 'w', encoding='utf-8') as f:
        json.dump(pure_api_request, f, indent=2, ensure_ascii=False)

    logger.trace("chat_api", "request", f"CHAT_API_REQUEST: {json.dumps(chat_api_request, indent=2, ensure_ascii=False)}")

    try:
        # Build llm_kwargs with RAG parameters (RAG is now always enabled)
        llm_kwargs = {
            "language": request.language,
            "rag_top_k": request.max_rag_results,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        # Remove None values
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

        # Call the enhanced chat service (RAG is now built-in)
        response = await chat_service.send_message(
            conversation_id=request.conversation_id,
            user_message=request.message,
            system_prompt=request.system_prompt,
            model=request.model,
            scope=request.scope,
            **llm_kwargs
        )

        # Prepare response
        chat_api_response = {
            "conversation_id": request.conversation_id,
            "message_id": response.id,
            "content": response.content,
            "model": response.model,
            "provider": response.provider.value,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

        # Log and store Chat API response details
        chat_api_response_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "CHAT_API_RESPONSE",
            "conversation_id": chat_api_response["conversation_id"],
            "message_id": chat_api_response["message_id"],
            "content": chat_api_response["content"],
            "content_length": len(chat_api_response["content"]),
            "model": chat_api_response["model"],
            "provider": chat_api_response["provider"],
            "usage": chat_api_response["usage"],
            "request_file": chat_request_file
        }

        # Store Chat API response to file with readable markdown formatting
        chat_response_file = f"./logs/chat_debug/{datetime.now().strftime('%Y%m%d_%H%M%S')}_chat_api_response.json"
        write_debug_json(chat_response_file, chat_api_response_log)

        # Store pure API response (clean format as per docs)
        pure_api_response = {
            "conversation_id": chat_api_response["conversation_id"],
            "message_id": chat_api_response["message_id"],
            "content": chat_api_response["content"],
            "model": chat_api_response["model"],
            "provider": chat_api_response["provider"],
            "usage": chat_api_response["usage"]
        }
        pure_response_file = f"./logs/chat_debug/{datetime.now().strftime('%Y%m%d_%H%M%S')}_pure_api_response.json"
        write_debug_json(pure_response_file, pure_api_response)

        logger.trace("chat_api", "response", f"CHAT_API_RESPONSE: {json.dumps(chat_api_response_log, indent=2, ensure_ascii=False)}")

        return chat_api_response

    except Exception as e:
        logger.error("chat_api", "send_message", f"Failed to send RAG message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/conversations")
async def list_conversations(tenant_slug: str = "development", user_id: str = "web_user", limit: int = 50):
    """List user conversations."""
    try:
        conversations = await chat_service.list_user_conversations(tenant_slug, user_id, limit)

        return [
            {
                "conversation_id": conv.conversation_id,
                "title": conv.title,
                "created_at": conv.get_created_at_timestamp(),
                "updated_at": conv.get_updated_at_timestamp(),
                "message_count": conv.message_count
            }
            for conv in conversations
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation message history."""
    try:
        messages = await chat_service.get_conversation_history(conversation_id)

        return [
            {
                "role": msg.role.value,
                "content": msg.content
            }
            for msg in messages
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/chat/stream/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for streaming chat."""
    await websocket.accept()
    logger = get_system_logger()
    logger.info("chat_api", "websocket_connect", f"WebSocket connected for {conversation_id}")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")

            if not message:
                continue

            # Send typing indicator
            await websocket.send_json({"type": "typing", "status": "started"})

            try:
                # Stream response
                response_chunks = []
                async for chunk in chat_service.send_message_streaming(
                    conversation_id=conversation_id,
                    user_message=message,
                    system_prompt=data.get("system_prompt"),
                    rag_context=data.get("rag_context"),
                    model=data.get("model")
                ):
                    response_chunks.append(chunk)
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk
                    })

                # Send completion signal
                await websocket.send_json({
                    "type": "complete",
                    "full_response": "".join(response_chunks)
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        logger.info("chat_api", "websocket_disconnect", f"WebSocket disconnected for {conversation_id}")


@app.post("/query/classify")
async def classify_query(request: QueryClassifyRequest):
    """Classify query to determine data source routing."""
    logger = get_system_logger()
    logger.info("query_classify", "request", f"Classifying query: '{request.query[:50]}...'")

    try:
        # Create classifier for specified language
        classifier = create_query_classifier(language=request.language)

        # Classify the query
        classification = classifier.classify_query(request.query)

        response = QueryClassifyResponse(
            query_type=classification.primary_type.value,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            detected_keywords=classification.detected_keywords,
            should_include_user_docs=classification.should_include_user_docs,
            should_include_narodne_novine=classification.should_include_narodne_novine
        )

        logger.info("query_classify", "response",
                   f"Classified as {response.query_type} (confidence: {response.confidence:.2f})")

        return response

    except Exception as e:
        logger.error("query_classify", "error", f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/scopes")
async def get_available_scopes(tenant_slug: str = "development", user_id: str = "web_user", language: str = "hr"):
    """Get available data source scopes for the user."""
    try:
        from src.chat.rag_integration import create_rag_chat_service

        # Create temporary RAG service to check available scopes
        rag_service = create_rag_chat_service(language=language, tenant_slug=tenant_slug, user_id=user_id)
        await rag_service.initialize()

        available_scopes = rag_service.get_available_scopes()
        await rag_service.close()

        return {
            "available_scopes": available_scopes,
            "scope_descriptions": {
                "user": "Personal Documents",
                "narodne_novine": "Narodne Novine (Croatian Official Gazette)",
                "tenant": "Tenant-wide Documents"
            }
        }
    except Exception as e:
        logger = get_system_logger()
        logger.error("chat_api", "get_scopes", f"Failed to get scopes: {e}")
        # Return default scopes if RAG service fails
        return {
            "available_scopes": ["user"],
            "scope_descriptions": {
                "user": "Personal Documents"
            }
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "chat-api",
        "providers": chat_service.llm_manager.get_available_providers() if chat_service else []
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting RAG Chat API Server...")
    print("📝 Web Interface: http://localhost:8080")
    print("🔧 API Docs: http://localhost:8080/docs")
    print("🛑 Press Ctrl+C to stop")

    uvicorn.run(
        "chat_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )