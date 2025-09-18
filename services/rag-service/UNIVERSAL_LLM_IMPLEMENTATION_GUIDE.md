# Universal LLM Provider Implementation Guide

## Overview

This document provides a comprehensive guide for implementing a universal LLM provider system that supports **OpenAI GPT**, **Google Gemini**, and **Anthropic Claude** APIs through a single, unified interface. The design eliminates vendor lock-in while providing automatic failover and language-specific response control.

## Key Benefits

✅ **Vendor Independence** - No lock-in to any single provider
✅ **Automatic Failover** - Seamless switching if primary provider fails
✅ **Language Control** - Clean language handling without hard-coded patterns
✅ **Cost Optimization** - Use different providers for different use cases
✅ **Feature Access** - Leverage unique capabilities of different providers
✅ **Configuration-Driven** - All provider differences handled through config

## Architecture Overview

```
┌─────────────────────┐
│   RAG System       │
├─────────────────────┤
│ UniversalLLMManager │  ← Single interface for all providers
├─────────────────────┤
│   Provider Factory  │  ← Creates appropriate adapters
├─────────────────────┤
│ OpenAI  │ Gemini    │  ← Provider-specific adapters
│ Adapter │ Adapter   │
│         │ Claude    │
│         │ Adapter   │
└─────────────────────┘
```

## API Differences Summary

| Feature | OpenAI | Gemini | Claude |
|---------|--------|---------|---------|
| **Authentication** | `Authorization: Bearer $KEY` | `x-goog-api-key: $KEY` | `x-api-key: $KEY` |
| **System Prompts** | Via messages array | `systemInstruction` field | `system` field |
| **Message Roles** | system, user, assistant | user, model | user, assistant |
| **Response Path** | `choices[0].message.content` | `candidates[0].content.parts[0].text` | `content[0].text` |
| **Streaming** | `choices[0].delta.content` | `candidates[0].content.parts[0].text` | `delta.text` |
| **Max Tokens** | Optional | Optional | **Required** |

## Implementation Files Structure

```
src/llm/
├── universal_provider.py          # Core interfaces and protocols
├── provider_factory.py            # Factory for creating adapters
├── provider_models.py             # Configuration models
├── adapters/
│   ├── __init__.py
│   ├── base_adapter.py            # Abstract base adapter
│   ├── openai_adapter.py          # OpenAI GPT adapter
│   ├── gemini_adapter.py          # Google Gemini adapter
│   └── claude_adapter.py          # Anthropic Claude adapter
└── manager.py                     # UniversalLLMManager
```

## Configuration Structure

### providers.toml
```toml
[providers]
primary = "claude"
fallback_order = ["openai", "gemini"]

[providers.claude]
api_key = "${ANTHROPIC_API_KEY}"
base_url = "https://api.anthropic.com"
model = "claude-3-5-sonnet-20241022"
endpoint = "/v1/messages"
auth_header = "x-api-key"
auth_prefix = ""
response_format = "anthropic"

[providers.openai]
api_key = "${OPENAI_API_KEY}"
base_url = "https://api.openai.com"
model = "gpt-4o-2024-08-06"
endpoint = "/v1/chat/completions"
auth_header = "Authorization"
auth_prefix = "Bearer "
response_format = "openai"

[providers.gemini]
api_key = "${GEMINI_API_KEY}"
base_url = "https://generativelanguage.googleapis.com"
model = "gemini-2.5-flash"
endpoint = "/v1beta/models/{model}:generateContent"
auth_header = "x-goog-api-key"
auth_prefix = ""
response_format = "gemini"
```

## Universal Request/Response Format

### Standardized Request
```python
@dataclass
class UniversalRequest:
    model: str
    messages: List[UniversalMessage]
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
```

### Standardized Response
```python
@dataclass
class UniversalResponse:
    id: str
    content: str
    finish_reason: FinishReason
    usage: TokenUsage
    provider: ProviderType
    model: str
```

## Language Control Implementation

### System Prompt Templates
```python
LANGUAGE_PROMPTS = {
    "hr": "Ti si pomoćni asistent koji odgovara ISKLJUČIVO na hrvatskom jeziku. "
          "Bez obzira na kontekst, uvijek odgovori na hrvatskom. "
          "Koristi dane informacije da daš precizan odgovor.",

    "en": "You are a helpful assistant who responds EXCLUSIVELY in English. "
          "Regardless of context, always respond in English. "
          "Use the given information to provide a precise answer."
}
```

### Provider-Specific Implementation
- **OpenAI**: Add system message to messages array
- **Gemini**: Use `systemInstruction` field
- **Claude**: Use `system` field

## Integration with Existing RAG System

### 1. Update OllamaClient to use UniversalLLMManager

```python
# Before (ollama_client.py)
class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config

# After (universal_client.py)
class UniversalLLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.llm_manager = create_universal_llm_manager_from_config(config)
```

### 2. Update RAG System Integration

```python
# In rag_system.py
class RAGSystem:
    def __init__(self, language: str = "en"):
        self.language = language
        config = get_unified_config()
        self.llm_client = UniversalLLMClient(config["providers"])

    async def generate_response(self, query: str, context: str) -> str:
        request = UniversalRequest(
            model=self.config["model"],  # Will be handled by primary provider
            messages=[
                UniversalMessage(role=MessageRole.USER, content=f"Context: {context}\n\nQuestion: {query}")
            ]
        )

        response = await self.llm_client.generate(request, language=self.language)
        return response.content
```

## Migration Strategy

### Phase 1: Implement Universal Interface
1. Create core interfaces and data models
2. Implement provider adapters
3. Create factory and manager classes
4. Add configuration support

### Phase 2: Provider Integration
1. Implement OpenAI adapter
2. Implement Claude adapter
3. Implement Gemini adapter
4. Test each adapter independently

### Phase 3: System Integration
1. Replace OllamaClient with UniversalLLMClient
2. Update configuration loading
3. Update RAG system integration
4. Test end-to-end functionality

### Phase 4: Language Control
1. Implement language-specific system prompts
2. Test Croatian/English response consistency
3. Remove any hard-coded language patterns
4. Validate fail-fast principles

## Error Handling & Failover

### Automatic Provider Switching
```python
async def generate(self, request: UniversalRequest, language: str = "en") -> UniversalResponse:
    providers_to_try = [self.primary_provider] + self.fallback_order

    for provider_name in providers_to_try:
        try:
            provider = self.providers[provider_name]
            return await provider.generate(request)
        except Exception as e:
            logger.warning(f"Provider {provider_name} failed: {e}")
            continue

    raise RuntimeError("All LLM providers failed")
```

### Rate Limiting Handling
- Exponential backoff for each provider
- Automatic switching to next provider on rate limits
- Retry logic with configurable attempts

## Testing Strategy

### Unit Tests
- Test each adapter independently
- Mock provider APIs for consistent testing
- Validate request/response transformations

### Integration Tests
- Test provider failover scenarios
- Validate language control across providers
- Test streaming functionality

### End-to-End Tests
- Full RAG pipeline with different providers
- Croatian/English language consistency tests
- Performance and reliability testing

## Performance Considerations

### Request Optimization
- Connection pooling for HTTP clients
- Request batching where supported
- Caching for repeated similar requests

### Response Processing
- Streaming support for real-time responses
- Efficient JSON parsing
- Memory management for large responses

## Security Considerations

### API Key Management
- Environment variable configuration
- No hard-coded credentials
- Secure key rotation support

### Request Validation
- Input sanitization
- Request size limits
- Rate limiting protection

## Monitoring & Observability

### Metrics Collection
- Response times per provider
- Error rates and types
- Token usage tracking
- Failover frequency

### Logging
- Provider selection decisions
- Request/response debugging (configurable)
- Error conditions and recovery

## Cost Optimization

### Provider Selection Strategy
- Use cost-effective providers for simple queries
- Reserve premium providers for complex tasks
- Monitor usage patterns and costs

### Token Management
- Optimize prompt sizes
- Implement response caching
- Track token usage per provider

## Conclusion

The Universal LLM Provider system provides:

1. **Complete vendor independence** - Easy switching between providers
2. **Robust failover** - Automatic recovery from provider issues
3. **Clean language control** - No hard-coded extraction patterns
4. **Configuration-driven** - All differences handled through config
5. **Future-proof** - Easy addition of new providers

This implementation solves the original language response problem while providing a solid foundation for multi-provider LLM usage in the RAG system.

## Next Steps

1. **Implement the core interfaces** as defined in `universal_llm_implementation_design.py`
2. **Create the adapter classes** for each provider
3. **Update configuration system** to support multiple providers
4. **Integrate with existing RAG system** by replacing OllamaClient
5. **Test language control** with Croatian/English inputs
6. **Remove hard-coded patterns** from OpenRouter implementation

The design is complete and ready for implementation!