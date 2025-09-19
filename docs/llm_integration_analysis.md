# LLM Client Modernization Plan

## Current State Overview

### Ollama Client (`services/rag-service/src/generation/ollama_client.py`)
- Focused on local Ollama (`/api/generate`) with JSON streaming lines processed via `response` field.
- Request schema uses `model`, `prompt`, `stream`, and `options` with temperature/top-p/top-k.
- Formal, non-chat prompt builder: constructs single-turn prompt (System, Context, Question, Answer stub).
- Language prompts keyed to query categories via configuration (`prompts` section).
- Confidence heuristics, logging, and dependency injection already implemented.
- No abstraction layer for other providers; response parser is Ollama-specific.
- Only supports single request lifecycle (`GenerationRequest` → `GenerationResponse`). No chat history structure.

### Configuration (`services/rag-service/config/config.toml`)
- `[ollama]` section merges base configuration and endpoints.
- `response_format` enumerates `ollama` vs `openai`; no specific sections for OpenAI/Claude/Gemini.
- `[ollama.endpoints]` contains path data for local or alternative API hosts; limited to one provider per config.
- Prompts set for question answering; no conversational multi-turn settings (e.g., memory length, roles).

## Goals
1. Support both local Ollama and hosted APIs (OpenAI, Claude, Gemini) with a unified abstraction.
2. Enable conversational chat flows with maintained history, multi-turn contexts, and structured role messages.
3. Keep compatibility with vector RAG retrieval pipeline while enabling follow-up questions referencing prior answers.
4. Fail-fast configuration with provider-specific validation; extend logging/metrics consistently.
5. Preserve default prompts in versioned config while enabling tenant-specific overrides stored in SurrealDB.

## Key Gaps & Required Changes

### 1. Abstraction Layer & Client Architecture
- Introduce provider-agnostic interfaces:
  - `LLMClient` protocol with methods like `generate_reply(history: ChatHistory, request: GenerationRequest)`.
  - Keep existing `GenerationRequest/Response`, but extend with chat metadata (conversation_id, parent_message_id, etc.).
  - Define a `ChatMessage` dataclass with `role` (system/user/assistant), `content`, optional `name/tool_calls`.
- Implement provider adapters:
  - `OllamaChatClient` (wrap current logic, adapt to chat format).
  - `OpenAIChatClient`, `ClaudeChatClient`, `GeminiChatClient` using respective SDKs or HTTP endpoints.
- Provide factory that selects client based on config (`provider = "ollama"|"openai"|"claude"|"gemini"`).

### 2. Configuration Restructure
- Split `[ollama]` into provider-specific sections (`[llm.ollama]`, `[llm.openai]`, `[llm.claude]`, `[llm.gemini]`).
- Add root `[llm]` with fields:
  - `provider` (enum), `default_language`, `default_model`, `chat_mode` options, `max_history_turns`.
  - API settings: `base_url`, `api_key`, `timeout`, `streaming`, `model_capabilities`.
- Introduce `[llm.formats]` to map provider to response parsers (already enumerated). Possibly rename `ResponseFormat` -> `LLMResponseFormat`.
- Add `[llm.prompts]` for system prompts, chat templates, fallback instructions.
- Support provider-specific overrides via `[llm.providers.<provider>.options]`.
- Keep base language prompts (e.g., `en.toml`, `hr.toml`) under version control and layer tenant/workflow overrides from SurrealDB; capture metadata (tenant, language, version, author, last_modified) for governance.

### 3. Chat Conversation Support
- Modify prompt builder to handle history: assemble `messages` list with alternating user/assistant roles.
- Add memory manager interfacing with storage (optional: in-memory per request). Provide config for history length and summarization.
- Extend `GenerationRequest` to include `history: list[ChatMessage]`, `session_id`, `tenant_id` for retrieval context.
- Update `GenerationResponse` with `message_id`, `stream_tokens`, `usage` (tokens prompt/completion).
- Ensure retrieval pipeline can pass follow-up question context + previous answers into LLM.

### 4. Provider-Specific Handling
- Response parsing per provider:
  - Ollama: JSON lines with `response` / `done`.
  - OpenAI/Claude: SSE or JSON with `choices[].delta.content`.
  - Gemini: `candidates[0].content.parts[].text` etc.
- Temperature/top_p meaning differs; create normalized config mapping to provider options (e.g., `temperature` -> `topP` for Gemini).
- Streaming protocols vary; adapters should handle event loops and convert to unified `GenerationResponse`.
- Add health checks for each provider endpoint.

### 5. Logging & Metrics
- Reuse `log_component_*` but consider provider dimension (tag logs with `provider=model` for metrics).
- Track token usage for billing when using hosted APIs.
- Add structured errors for API quota/timeouts.

### 6. Security & Secrets
- Add config for storing API keys securely (env var references). Provide instructions in docs.
- Avoid bundling API keys in TOML; use `env:` placeholders.
- Support optional proxy settings, SSL options for remote APIs.

## Proposed Refactoring Steps
1. **Introduce new data structures**
   - `ChatMessage`, `ChatHistory`, `LLMProviderConfig`, `LLMClientSettings` dataclasses.
   - Extend `GenerationRequest/Response` for conversation data.

2. **Configuration overhaul**
   - Create new `[llm]` section; migrate existing `[ollama]` to `[llm.ollama]`.
   - Update `config_validator` & `config_models` to parse provider sections.
   - Support environment variable interpolation for secrets.

3. **Client architecture**
   - Define `LLMClient` protocol; implement `OllamaChatClient` adapting existing functions.
   - Implement stubs for `OpenAIChatClient`, `ClaudeChatClient`, `GeminiChatClient` with detailed TODO comments.
   - Provide `LLMClientFactory` selecting and instantiating the correct adapter based on config.

4. **Prompt & History handling**
   - Update `build_complete_prompt` to produce structured message list rather than single string.
   - Add `MessageFormatter` module to convert history to provider-specific format (e.g., `[{role, content}]`).
   - Support system role prompts per provider.
   - Implement layered prompt lookup (TOML defaults → SurrealDB overrides) and optionally expose admin tooling for runtime edits with validation of placeholders/tokens.

5. **Response parsing**
   - Extract current parsing functions into `ollama_parser.py` and add analogous modules for other providers.
   - Provide unified interface returning `GenerationResponse`.

6. **Conversation state management** (optional milestone)
   - Introduce `ConversationManager` storing histories keyed by session (in-memory or SurrealDB).
   - Provide config for max history length, summarization triggers.
   - Ensure retrieval step can fetch previously generated answers to include in context.

7. **Documentation & onboarding**
   - Update README/docs with provider setup instructions, sample config sections, and environment variable usage.
   - Document how to run with local Ollama vs OpenAI/Claude/Gemini (including rate limits).

## Research / Open Questions
- Precise API contract differences:
  - OpenAI ChatCompletions vs Responses API (gpt-4.1) vs legacy completions.
  - Claude’s anthropic API streaming details (SSE vs chunked JSON).
  - Gemini API message format (multi-part content, safety categories).
- How to persist chat history: SurrealDB vs in-memory vs vectordb (for summarization).
- Tool/function calling: Do we anticipate invoking external tools? Need to plan for structured outputs.
- Cost/billing tracking: Should we log token usage for remote providers?
- Model switching per request: Should API allow specifying target provider/model at runtime?

## Immediate Next Steps
1. Draft new `llm_config.py` dataclasses for provider-agnostic config.
2. Map existing `ollama_client` logic into `OllamaChatClient` implementing `LLMClient`.
3. Prototype conversation-aware prompt builder and message format module.
4. Design configuration migration strategy (backwards-compatible default to `ollama`).
5. Evaluate SDKs/HTTP clients for OpenAI, Claude, Gemini (consider async support compatibility).

## Suggested Renaming
- Rename `ollama_client.py` to `llm_clients/ollama_client.py` to emphasize modularity.
- Introduce package structure `src/generation/llm_clients/` and `src/generation/llm_parsers/`.
- Update `config_models.OllamaConfig` to `LLMProviderConfig` with provider-specific subclasses (Ollama/OpenAI/Claude/Gemini).

## Summary
By introducing a provider-agnostic LLM client interface, reworking configuration to support multiple providers, and adding conversation-aware prompt/message handling, the RAG system can flexibly use local Ollama models and remote APIs like OpenAI, Claude, and Gemini. The recommended roadmap covers configuration, adapter implementations, response parsing, and conversation state management to achieve a full chat-oriented experience across providers.
