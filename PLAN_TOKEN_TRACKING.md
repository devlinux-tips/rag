# Token Tracking Implementation Plan

## Objective
Enable accurate token usage tracking for rate limiting and cost control by integrating the RAG system with the LLM manager architecture.

## Current Architecture Problem

### What's Wrong:
```python
# rag_system.py:1382 (CURRENT - BROKEN)
from ..generation.ollama_client import GenerationRequest

generation_request = GenerationRequest(
    prompt=user_prompt,
    context=context_chunks,
    query=validated_query.text,
    # ...
)

# Direct Ollama API call - NO TOKEN TRACKING
response = await self.ollama_client.generate(generation_request)
```

### What's Right (but unused):
```python
# llm_manager.py:595 (EXISTS - NOT USED BY RAG)
async def chat_completion(
    self,
    messages: list[ChatMessage],
    model: str | None = None,
    **kwargs
) -> ChatResponse:
    # Returns ChatResponse with:
    # - usage: TokenUsage(input_tokens, output_tokens, total_tokens)
    # - model: actual model name
    # - provider: ProviderType.OPENROUTER
```

## Implementation Steps

### Step 1: Add Token Usage to RAGResponse
**File:** `services/rag-service/src/pipeline/rag_system.py:42`

**Before:**
```python
@dataclass
class RAGResponse:
    answer: str
    query: str
    retrieved_chunks: list[dict[str, Any]]
    confidence: float
    generation_time: float
    retrieval_time: float
    total_time: float
    sources: list[str]
    metadata: dict[str, Any]
    nn_sources: list[dict[str, Any]] | None = None
```

**After:**
```python
from ..generation.llm_provider import TokenUsage  # ADD IMPORT

@dataclass
class RAGResponse:
    answer: str
    query: str
    retrieved_chunks: list[dict[str, Any]]
    confidence: float
    generation_time: float
    retrieval_time: float
    total_time: float
    sources: list[str]
    metadata: dict[str, Any]
    nn_sources: list[dict[str, Any]] | None = None
    token_usage: TokenUsage | None = None  # ADD THIS
    model_used: str = "unknown"  # ADD THIS
```

### Step 2: Update RAGSystem Constructor
**File:** `services/rag-service/src/pipeline/rag_system.py` (constructor)

**Add:**
```python
from ..generation.llm_provider import LLMManager, ChatMessage, MessageRole

class RAGSystem:
    def __init__(
        self,
        # ... existing params
        llm_manager: LLMManager | None = None,  # ADD THIS
    ):
        # ... existing code
        self.llm_manager = llm_manager  # ADD THIS
```

### Step 3: Replace Ollama Client with LLM Manager
**File:** `services/rag-service/src/pipeline/rag_system.py:1350-1400`

**Before:**
```python
# Lines 1363-1389 (CURRENT)
system_prompt = prompts_config["question_answering_system"] + citation_instruction
user_prompt = prompts_config["question_answering_user"].format(
    query=validated_query.text, context=context_text
)

from ..generation.ollama_client import GenerationRequest

generation_request = GenerationRequest(
    prompt=user_prompt,
    context=context_chunks,
    query=validated_query.text,
    query_type=category_value,
    language=validated_query.language,
    system_prompt=system_prompt,
)

answer = await self.ollama_client.generate(generation_request)
```

**After:**
```python
# Build chat messages for LLM manager
system_prompt = prompts_config["question_answering_system"] + citation_instruction
user_prompt = prompts_config["question_answering_user"].format(
    query=validated_query.text, context=context_text
)

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
    ChatMessage(role=MessageRole.USER, content=user_prompt)
]

# Call LLM manager (supports OpenRouter with token tracking)
chat_response = await self.llm_manager.chat_completion(
    messages=messages,
    temperature=self.ollama_config.get("temperature", 0.7),
    max_tokens=self.ollama_config.get("num_predict", 2048)
)

answer = chat_response.content
token_usage = chat_response.usage
model_used = chat_response.model
```

### Step 4: Update RAGResponse Creation
**File:** `services/rag-service/src/pipeline/rag_system.py:1460-1466`

**Before:**
```python
return RAGResponse(
    answer=answer,
    query=validated_query.text,
    retrieved_chunks=retrieved_chunks,
    confidence=rag_response.confidence,
    generation_time=generation_time,
    retrieval_time=retrieval_time,
    total_time=total_time,
    sources=sources,
    metadata=response_metadata,
    nn_sources=nn_sources if nn_sources else None,
)
```

**After:**
```python
return RAGResponse(
    answer=answer,
    query=validated_query.text,
    retrieved_chunks=retrieved_chunks,
    confidence=rag_response.confidence,
    generation_time=generation_time,
    retrieval_time=retrieval_time,
    total_time=total_time,
    sources=sources,
    metadata=response_metadata,
    nn_sources=nn_sources if nn_sources else None,
    token_usage=token_usage,  # ADD THIS
    model_used=model_used,  # ADD THIS
)
```

### Step 5: Initialize LLM Manager in RAG-API
**File:** `services/rag-api/main.py` (startup section)

**Add:**
```python
from src.generation.llm_provider import LLMManager

# During initialization (around line 250)
llm_config = get_unified_config()
llm_manager = LLMManager(llm_config)

# Pass to RAGSystem constructor
rag_system = RAGSystem(
    # ... existing params
    llm_manager=llm_manager,  # ADD THIS
)
```

### Step 6: Update Token Usage in API Response
**File:** `services/rag-api/main.py:469-473`

**Before:**
```python
if rag_response.answer and not is_error_response:
    final_response = rag_response.answer
    model_used = "rag-generated"
    llm_tokens = TokenUsage(input=0, output=0, total=0)
```

**After:**
```python
if rag_response.answer and not is_error_response:
    final_response = rag_response.answer
    model_used = rag_response.model_used

    # Use actual token counts from LLM response
    if rag_response.token_usage:
        llm_tokens = TokenUsage(
            input=rag_response.token_usage.input_tokens,
            output=rag_response.token_usage.output_tokens,
            total=rag_response.token_usage.total_tokens
        )
    else:
        # Fallback if no token tracking available
        llm_tokens = TokenUsage(input=0, output=0, total=0)
```

## Testing Plan

### 1. Unit Tests
```bash
# Test RAGResponse with token_usage
python -m pytest services/rag-service/tests/test_rag_system.py -k token

# Test LLM manager integration
python -m pytest services/rag-service/tests/test_llm_manager.py
```

### 2. Integration Test
```bash
# Query with OpenRouter and check tokens
curl -X POST http://localhost:8082/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "tenant": "development",
    "user": "dev_user",
    "language": "hr",
    "scope": "feature",
    "feature": "narodne-novine"
  }' | jq '.tokensUsed'

# Expected output:
# {
#   "input": 1234,
#   "output": 567,
#   "total": 1801
# }
```

### 3. UI Verification
- Open web UI
- Send a query
- Check that token count appears in metadata (🎯 X tokens)
- Verify it's > 0

## Migration Considerations

### Backward Compatibility
- `token_usage` is optional (`| None`) - won't break existing code
- Fallback to 0 tokens if not available
- Gradual migration: old ollama_client can stay for fallback

### Configuration
- No config changes needed - already have `llm_manager` config
- Uses existing `${OPENROUTER_API_KEY}` environment variable

### Performance
- LLM manager has built-in failover
- Same latency as direct Ollama calls
- OpenRouter may be slightly slower but provides token tracking

## Rate Limiting Design (Future)

Once token tracking works:

```python
# Future: In web-api/src/middleware/rate_limit.ts
async function checkTokenLimit(userId: string, tokensUsed: number) {
  const usage = await redis.get(`tokens:${userId}:${today()}`);
  const limit = await getUserTokenLimit(userId);

  if (usage + tokensUsed > limit) {
    throw new RateLimitError("Daily token limit exceeded");
  }

  await redis.incrby(`tokens:${userId}:${today()}`, tokensUsed);
}
```

## Success Criteria

- ✅ Token counts appear in API responses (not 0)
- ✅ Token counts appear in UI metadata
- ✅ Model name shows actual model (not "rag-generated")
- ✅ OpenRouter integration working end-to-end
- ✅ No regression in query response quality
- ✅ All tests passing

## Estimated Effort

- **Step 1-2:** 30 minutes (dataclass updates)
- **Step 3-4:** 1-2 hours (core refactoring + testing)
- **Step 5-6:** 30 minutes (API integration)
- **Testing:** 1 hour (manual + automated)

**Total:** 3-4 hours

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing queries | High | Add token_usage as optional field, fallback to 0 |
| LLM manager not initialized | High | Add validation at startup, fail fast |
| OpenRouter API errors | Medium | Keep ollama_client as fallback option |
| Token counts still 0 | Low | OpenRouter provider already tested, should work |

## Next Steps

1. Create feature branch: `git checkout -b feature/token-tracking`
2. Implement Step 1-2 (RAGResponse updates)
3. Implement Step 3-4 (LLM manager integration)
4. Test with curl
5. Implement Step 5-6 (API updates)
6. Run full test suite
7. Manual UI testing
8. Commit and document
9. Update IMPROVEMENTS.md to mark as complete
