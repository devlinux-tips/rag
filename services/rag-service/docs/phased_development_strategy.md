# Phased Development Strategy: Simple Web → Full Elixir Platform

## Phase 1: Minimal Web Interface (Week 1-2)
**Goal**: Working web chat interface for users, hiding all complexity

### 1.1 Simple Python FastAPI Backend
```python
# src/web/simple_api.py
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Serve static files (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Hard-coded defaults (hidden from user)
DEFAULT_TENANT = "development"
DEFAULT_USER = "web_user"
DEFAULT_LANGUAGE = "hr"

@app.get("/")
async def chat_interface():
    """Serve simple chat HTML interface."""
    return HTMLResponse(open("web/templates/chat.html").read())

@app.post("/chat")
async def chat_query(request: ChatRequest):
    """Simple chat endpoint - user just sends message, gets response."""
    # Hide all complexity from user
    rag_system = RAGSystem(language=DEFAULT_LANGUAGE)

    response = await rag_system.query(
        tenant=DEFAULT_TENANT,
        user=DEFAULT_USER,
        query=request.message,
        language=DEFAULT_LANGUAGE
    )

    return {"response": response.answer}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for streaming responses (optional enhancement)."""
    await websocket.accept()

    while True:
        message = await websocket.receive_text()

        # Stream response chunks
        async for chunk in rag_system.query_streaming(
            tenant=DEFAULT_TENANT,
            user=DEFAULT_USER,
            query=message,
            language=DEFAULT_LANGUAGE
        ):
            await websocket.send_text(chunk)
```

### 1.2 Simple HTML Chat Interface
```html
<!-- web/templates/chat.html -->
<!DOCTYPE html>
<html>
<head>
    <title>RAG Chat</title>
    <style>
        .chat-container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .messages { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; }
        .input-area { display: flex; gap: 10px; margin-top: 10px; }
        input[type="text"] { flex: 1; padding: 10px; }
        button { padding: 10px 20px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div id="messages" class="messages"></div>
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Postavite pitanje...">
            <button onclick="sendMessage()">Pošalji</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;

            // Add user message to chat
            addMessage('user', message);
            input.value = '';

            // Send to API
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            });

            const data = await response.json();
            addMessage('assistant', data.response);
        }

        function addMessage(role, content) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.innerHTML = `<strong>${role}:</strong> ${content}`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollTop;
        }
    </script>
</body>
</html>
```

### 1.3 Launch Script
```python
# launch_simple.py
import uvicorn
from src.web.simple_api import app

if __name__ == "__main__":
    print("🚀 Starting Simple RAG Web Interface...")
    print("📝 Access at: http://localhost:8080")
    print("🛑 Admin CLI still available: python rag.py ...")

    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Phase 2: Strategic Architecture Preparation (Week 2-4)
**Goal**: Prepare components for Elixir migration while simple web works

### 2.1 Refactor Python for Service Architecture
```python
# src/services/rag_service.py - Prepared for Elixir integration
class RAGService:
    """Service layer ready for Elixir communication."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_service = QueryService(config)
        self.document_service = DocumentService(config)

    async def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Command handler ready for Elixir JSON-RPC."""
        cmd_type = command["type"]
        params = command["params"]

        if cmd_type == "query":
            return await self.query_service.execute(params)
        elif cmd_type == "process_documents":
            return await self.document_service.execute(params)
        # More commands as needed
```

### 2.2 Create Basic Elixir Phoenix App
```elixir
# Initialize Phoenix project (parallel development)
mix phx.new rag_platform --no-ecto
cd rag_platform

# Add basic tenant/user models (planning for future)
defmodule RagPlatform.Accounts.User do
  use Ecto.Schema

  schema "users" do
    field :email, :string
    field :name, :string
    belongs_to :tenant, RagPlatform.Accounts.Tenant
    timestamps()
  end
end
```

## Phase 3: Elixir Integration (Week 4-6)
**Goal**: Replace simple Python API with Phoenix, maintain user experience

### 3.1 Phoenix Web Interface
```elixir
# lib/rag_platform_web/live/chat_live.ex
defmodule RagPlatformWeb.ChatLive do
  use Phoenix.LiveView

  def mount(_params, _session, socket) do
    # For now, use same defaults as Python API
    socket = assign(socket,
      tenant: "development",
      user: "web_user",
      language: "hr",
      messages: []
    )

    {:ok, socket}
  end

  def handle_event("send_message", %{"message" => message}, socket) do
    # Call Python service (same backend, different frontend)
    response = call_python_rag_service(message, socket.assigns)

    messages = socket.assigns.messages ++ [
      %{role: "user", content: message},
      %{role: "assistant", content: response}
    ]

    {:noreply, assign(socket, messages: messages)}
  end

  defp call_python_rag_service(message, assigns) do
    # Initially HTTP call to Python, later GenServer
    HTTPoison.post!("http://localhost:8080/chat",
      Jason.encode!(%{message: message}),
      [{"Content-Type", "application/json"}]
    )
    |> Map.get(:body)
    |> Jason.decode!()
    |> Map.get("response")
  end
end
```

### 3.2 User Migration Strategy
```bash
# Week 4: Users access Phoenix instead of Python directly
# http://localhost:4000 (Phoenix) -> calls -> http://localhost:8080 (Python)

# Week 5: Replace HTTP with GenServer communication
# Phoenix LiveView -> GenServer -> Python Port

# Week 6: Add streaming, maintain same user interface
```

## Phase 4: Advanced Features (Week 6-8)
**Goal**: Add real-time streaming while maintaining simple UX

### 4.1 Streaming Integration
```elixir
def handle_event("send_message", %{"message" => message}, socket) do
  # Start async streaming
  pid = self()

  Task.start_link(fn ->
    RagPlatform.PythonService.query_streaming(message, fn chunk ->
      send(pid, {:rag_chunk, chunk})
    end)
  end)

  messages = socket.assigns.messages ++ [
    %{role: "user", content: message},
    %{role: "assistant", content: "", streaming: true}
  ]

  {:noreply, assign(socket, messages: messages)}
end

def handle_info({:rag_chunk, chunk}, socket) do
  # Update last message with new chunk
  messages = update_streaming_message(socket.assigns.messages, chunk)
  {:noreply, push_event(socket, "append_chunk", %{chunk: chunk})}
end
```

## Phase 5: Platform Features (Week 8-12)
**Goal**: Add multi-tenancy, user management, advanced features

### 5.1 Gradual Complexity Introduction
```elixir
# Week 8: Add user registration (still single tenant)
# Week 9: Add document upload interface
# Week 10: Add conversation history
# Week 11: Add multi-tenant admin interface
# Week 12: Add advanced analytics
```

## Strategic Benefits of This Approach

### ✅ Immediate Value
- **Week 1**: Users have working web chat interface
- **Week 2**: Streaming responses, better UX
- **Week 3**: Phoenix replaces Python frontend (same experience)

### ✅ Risk Mitigation
- **Always Working**: Never break user experience during migration
- **Parallel Development**: Elixir features built while Python works
- **Gradual Migration**: Replace components one by one
- **Rollback Strategy**: Can always revert to previous phase

### ✅ Strategic Planning
- **Architecture Ready**: All complex features planned from start
- **Team Efficiency**: Frontend/backend teams work independently
- **Future-Proof**: Foundation supports all advanced features

## Development Commands

### Phase 1 Commands
```bash
# Start simple web interface
python launch_simple.py

# Admin still uses CLI
python rag.py --tenant development --user admin --language hr status
```

### Phase 3 Commands
```bash
# Start Phoenix (calls Python backend)
cd rag_platform && mix phx.server

# Python service runs as backend
python launch_simple.py  # Same backend, new frontend
```

### Phase 5 Commands
```bash
# Full platform
cd rag_platform && mix phx.server

# Admin CLI for system management
python rag.py admin create-tenant --slug "enterprise"
python rag.py admin manage-users --tenant "enterprise"
```

## Migration Safety Net

### Always Maintain Working State
1. **Phase 1**: Python web works
2. **Phase 2**: Python web + Elixir development in parallel
3. **Phase 3**: Phoenix calls Python (same functionality)
4. **Phase 4**: Phoenix + streaming (enhanced functionality)
5. **Phase 5**: Full platform features

### Rollback Strategy
- Each phase can run independently
- Previous phase always available as backup
- Database migrations planned and reversible
- Configuration switches between modes

This approach gives you immediate value while strategically building toward the full platform vision!