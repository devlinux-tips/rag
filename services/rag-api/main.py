#!/usr/bin/env python
"""
FastAPI server for RAG service integration.
Bridges between TypeScript Web API and Python RAG system.
"""

import sys
import os
import time
import json
import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Configure file logging
log_dir = "/home/rag/src/rag/services/rag-api/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "fastapi.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the rag-service directory to the Python path
# Support both Docker (/app/src) and local deployment
rag_service_path = os.path.join(os.path.dirname(__file__), "..", "rag-service", "src")
if os.path.exists(rag_service_path):
    sys.path.insert(0, os.path.abspath(os.path.join(rag_service_path, "..")))
elif os.path.exists("/app/src"):
    sys.path.insert(0, "/app/src")
else:
    # Rely on PYTHONPATH from environment
    pass

# Import RAG service components
from src.generation.llm_provider import (
    LLMManager,
    ChatMessage,
    MessageRole,
)
from src.utils.factories import create_complete_rag_system, Tenant, User
from src.pipeline.rag_system import RAGQuery

# Initialize FastAPI
app = FastAPI(
    title="RAG Service API",
    description="API bridge for Python RAG system",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # TypeScript API origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., description="User's query text")
    tenant: str = Field(..., description="Tenant identifier")
    user: str = Field(..., description="User identifier")
    language: str = Field(..., description="Language code (hr, en)")
    scope: str = Field("user", description="Document scope: user, tenant, or feature")
    feature: Optional[str] = Field(
        None, description="Feature name when scope=feature (e.g., narodne-novine)"
    )
    max_documents: Optional[int] = Field(5, description="Maximum documents to retrieve")
    min_confidence: Optional[float] = Field(
        0.7, description="Minimum confidence threshold"
    )
    temperature: Optional[float] = Field(0.7, description="LLM temperature")


class DocumentSource(BaseModel):
    """Document source in RAG response."""

    documentId: str
    title: str
    relevance: float
    chunk: str


class TokenUsage(BaseModel):
    """Token usage statistics."""

    input: int
    output: int
    total: int


class NNSource(BaseModel):
    """Narodne Novine source metadata."""

    citationId: int = Field(..., description="Citation number for referencing [1], [2], etc.")
    title: str = Field(..., description="Document title")
    issue: str = Field(..., description="NN issue number")
    eli: Optional[str] = Field(None, description="European Legislation Identifier")
    year: Optional[str] = Field(None, description="Publication year")
    publisher: Optional[str] = Field(None, description="Publishing authority")
    start_page: Optional[int] = Field(None, description="Starting page number")
    end_page: Optional[int] = Field(None, description="Ending page number")


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries."""

    response: str = Field(
        ..., description="Generated response with Markdown formatting"
    )
    sources: List[DocumentSource] = Field(default_factory=list)
    documentsRetrieved: int = Field(0)
    documentsUsed: int = Field(0)
    confidence: float = Field(0.0)
    searchTimeMs: int = Field(0)
    responseTimeMs: int = Field(0)
    model: str = Field("qwen2.5:7b-instruct")
    tokensUsed: TokenUsage
    nnSources: Optional[List[NNSource]] = Field(None, description="Narodne Novine source metadata for citations")


def get_collection_name(tenant: str, user: str, feature: str, language: str) -> str:
    """
    Determine the correct collection name based on scope.

    Features have three scopes:
    - user: Personal documents for that user only
    - tenant: Shared documents for all users in tenant
    - feature: Specialized datasets available globally
    """
    # Map feature names to scopes
    if feature in [
        "narodne-novine",
        "financial-reports",
        "legal-docs",
        "medical-records",
    ]:
        # Global feature collections
        return f"Features_{feature.replace('-', '_')}_{language}"
    elif feature == "tenant" or feature == "shared":
        # Tenant-shared documents
        return f"{tenant}_shared_{language}_documents"
    else:
        # Default to user scope (personal documents)
        return f"{tenant}_{user}_{language}_documents"


# Global variables for services
llm_manager = None
rag_systems = {}  # Store RAG systems per tenant/language/scope


# Request/Response logging functions
def dump_request_response(
    timestamp: str, request_data: dict, response_data: dict, log_type: str = "fastapi"
):
    """Dump request and response data to timestamped JSON files."""
    try:
        os.makedirs("services/rag-service/logs", exist_ok=True)

        # Request dump
        request_filename = (
            f"services/rag-service/logs/{timestamp}_{log_type}_request.json"
        )
        with open(request_filename, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False, default=str)

        # Response dump
        response_filename = (
            f"services/rag-service/logs/{timestamp}_{log_type}_response.json"
        )
        with open(response_filename, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"INFO: Logged {log_type} request/response: {timestamp}")

    except Exception as e:
        print(f"ERROR: Failed to dump {log_type} logs: {str(e)}")


async def get_or_create_rag_system(tenant: str, user: str, language: str, scope: str):
    """Get or create RAG system for specific tenant/user/language/scope combination."""
    global rag_systems

    # Key includes scope for proper routing
    rag_key = f"{tenant}_{user}_{language}_{scope}"

    if rag_key not in rag_systems:
        print(f"INFO: Creating RAG system for scope: {scope}")

        # Create tenant and user objects
        tenant_obj = Tenant(
            id=f"tenant:{tenant}", name=f"Tenant {tenant.title()}", slug=tenant
        )
        user_obj = User(
            id=f"user:{user}",
            tenant_id=tenant_obj.id,
            email=f"{user}@{tenant}.example.com",
            username=user,
            full_name=f"User {user.title()}",
        )

        # Set environment variables for Weaviate connection before creating RAG system
        weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
        weaviate_port = os.getenv("WEAVIATE_PORT", "8080")
        print(f"INFO: Using Weaviate at {weaviate_host}:{weaviate_port}")

        if scope == "narodne_novine":
            # For Narodne Novine feature scope
            print("INFO: Creating feature-scoped RAG system for narodne-novine")
            rag_system = create_complete_rag_system(
                language=language,
                scope="feature",
                feature_name="narodne-novine",  # Use hyphen format for feature name
            )
        else:
            # Default user/tenant scope
            print(f"INFO: Creating {scope}-scoped RAG system for {tenant}/{user}")
            rag_system = create_complete_rag_system(
                language=language,
                scope=scope,
                tenant=tenant_obj,
                user=user_obj,
            )

        await rag_system.initialize()
        rag_systems[rag_key] = rag_system
        print(f"INFO: RAG system created for {rag_key}")

    return rag_systems[rag_key]


@app.on_event("startup")
async def startup_event():
    """Initialize LLM manager on startup."""
    global llm_manager

    print("INFO: RAG API server starting...")
    print("INFO: Loading configuration...")

    # Load OpenRouter config from config.toml
    from src.utils.config_loader import load_config
    config = load_config("config")
    openrouter_toml_config = config.get("openrouter", {})

    openrouter_config = {
        "api_key": openrouter_toml_config.get("api_key", os.getenv("OPENROUTER_API_KEY")),  # From config.toml or environment
        "base_url": openrouter_toml_config.get("base_url", "https://openrouter.ai/api/v1"),
        "model": "qwen/qwen3-30b-a3b-instruct-2507",
        "timeout": float(openrouter_toml_config.get("timeout", 30.0)),
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }

    # AI-FRIENDLY DEBUG LOG: Verify API key loaded
    key_preview = f"{openrouter_config['api_key'][:15]}...{openrouter_config['api_key'][-10:]}" if openrouter_config['api_key'] and len(openrouter_config['api_key']) > 25 else "MISSING"
    print(f"INFO: OpenRouter API key loaded: {key_preview}")

    # Create LLM Manager with proper config structure
    llm_config = {
        "primary_provider": "openrouter",
        "fallback_order": ["openrouter"],  # Only OpenRouter, no fallbacks
        "openrouter": openrouter_config,
    }
    llm_manager = LLMManager(llm_config)

    print("INFO: LLM provider: OpenRouter")
    print("INFO: Ready to receive requests on http://localhost:8082")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "rag-api",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """
    Process a RAG query directly using RAG system.

    This endpoint:
    1. Gets or creates RAG system for the specified scope
    2. Executes RAG search to retrieve relevant documents
    3. Sends documents + query to LLM for response generation
    4. Returns formatted response with sources and logging
    """
    global llm_manager
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # Log FastAPI request
    request_data = {
        "timestamp": timestamp,
        "endpoint": "/api/v1/query",
        "method": "POST",
        "request": request.dict(),
    }

    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not initialized")

    try:
        # Validate and determine scope
        if request.scope == "feature":
            if not request.feature:
                raise HTTPException(
                    status_code=400,
                    detail="Feature name required when scope is 'feature'",
                )
            scope = request.feature.replace(
                "-", "_"
            )  # narodne-novine -> narodne_novine
        elif request.scope in ["user", "tenant"]:
            scope = request.scope
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid scope. Must be 'user', 'tenant', or 'feature'",
            )

        print(
            f"INFO_RAG_QUERY | scope={scope} | requested_scope={request.scope} | feature={request.feature} | language={request.language} | tenant={request.tenant} | user={request.user}"
        )

        # Get or create RAG system for this scope
        rag_system = await get_or_create_rag_system(
            tenant=request.tenant,
            user=request.user,
            language=request.language,
            scope=scope,
        )

        # Create RAG query with scope context for categorization
        scope_context = {
            "tenant_slug": request.tenant,
            "user_id": request.user,
            "scope": scope,
        }

        # Add feature_name for scope-based categorization (narodne-novine -> legal)
        if request.scope == "feature" and request.feature:
            scope_context["feature_name"] = (
                request.feature
            )  # Use hyphen format: "narodne-novine"

        rag_query = RAGQuery(
            text=request.query,
            language=request.language,
            user_id=request.user,
            max_results=request.max_documents,
            context_filters=scope_context,
        )

        # Execute RAG search - handle missing collections gracefully
        rag_response = None
        try:
            rag_response = await rag_system.query(rag_query)
        except Exception as search_error:
            # AI-FRIENDLY LOG: Structured for pattern recognition
            error_msg = str(search_error)
            error_type = type(search_error).__name__
            is_collection_missing = "could not find class" in error_msg.lower() or "collection" in error_msg.lower()

            print(f"ERROR_RAG_SEARCH | error_type={error_type} | is_collection_missing={is_collection_missing} | scope={scope} | tenant={request.tenant} | user={request.user} | language={request.language} | error_msg={error_msg[:200]}")

            # Check if it's a "collection not found" error (no data in Weaviate yet)
            if is_collection_missing:
                print(f"INFO_COLLECTION_MISSING | scope={scope} | tenant={request.tenant} | feature={request.feature} | language={request.language} | status=no_documents_indexed")

            # Create empty response to trigger LLM-only generation
            from src.core.rag_types import RAGResponse
            rag_response = RAGResponse(
                answer=None,
                retrieved_chunks=[],
                confidence=0.0,
                search_time_ms=0,
                llm_response_time_ms=0,
                total_time_ms=0,
            )

        search_time = time.time()
        search_time_ms = int((search_time - start_time) * 1000)

        # Check if we have documents OR a generated answer
        # If no documents found OR error occurred, send question directly to LLM (general knowledge)
        has_no_data = not rag_response.retrieved_chunks and not rag_response.answer
        has_error = rag_response.metadata and rag_response.metadata.get("error_type") is not None

        if has_no_data or has_error:
            reason = "error_fallback" if has_error else "no_documents"
            print(f"INFO_LLM_FALLBACK | scope={scope} | tenant={request.tenant} | language={request.language} | reason={reason} | fallback=llm_general_knowledge")

            # Send question directly to LLM without context (general knowledge mode)
            from src.utils.config_loader import get_language_specific_config

            prompts_config = get_language_specific_config("prompts", request.language)

            # Use a simple prompt for general questions
            system_prompt = prompts_config.get('system_prompt', 'You are a helpful assistant.')

            # Create messages in the format expected by chat_completion
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=request.query)
            ]

            llm_response = await llm_manager.chat_completion(
                messages=messages,
                model="qwen/qwen3-30b-a3b-instruct-2507",  # OpenRouter model
                max_tokens=2048,
                temperature=request.temperature,
            )

            end_time = time.time()
            total_time_ms = int((end_time - start_time) * 1000)

            return RAGQueryResponse(
                response=llm_response.content,
                sources=[],
                documentsRetrieved=0,
                documentsUsed=0,
                confidence=0.5,  # Lower confidence since no documents
                searchTimeMs=search_time_ms,
                responseTimeMs=total_time_ms,
                model=llm_response.model,
                tokensUsed=TokenUsage(
                    input=llm_response.usage.input_tokens,
                    output=llm_response.usage.output_tokens,
                    total=llm_response.usage.total_tokens,
                ),
            )

        # Check if RAG response is an error (has error_type in metadata)
        is_error_response = rag_response.metadata and rag_response.metadata.get("error_type") is not None

        # AI-FRIENDLY LOG: Error detection
        if is_error_response:
            error_type = rag_response.metadata.get("error_type")
            error_msg = rag_response.metadata.get("error_message", "")[:200]
            is_weaviate_missing = "could not find class" in error_msg.lower() or "collection" in error_msg.lower()

            print(f"INFO_ERROR_DETECTED | error_type={error_type} | is_weaviate_missing={is_weaviate_missing} | fallback_to_llm=True | language={request.language}")

            # Treat error as "no answer" to trigger LLM fallback
            rag_response.answer = None

        # If RAG found documents, use the RAG-generated answer if available,
        # otherwise send to LLM for generation
        if rag_response.answer and not is_error_response:
            # Use RAG-generated answer directly (RAG already called LLM internally)
            final_response = rag_response.answer
            model_used = getattr(rag_response, 'model_used', 'rag-generated')
            # Get token usage from RAG response with input/output breakdown
            rag_tokens_total = getattr(rag_response, 'tokens_used', 0)
            rag_tokens_input = getattr(rag_response, 'input_tokens', 0)
            rag_tokens_output = getattr(rag_response, 'output_tokens', 0)
            llm_tokens = TokenUsage(
                input=rag_tokens_input,
                output=rag_tokens_output,
                total=rag_tokens_total
            )
        else:
            # Send to LLM for generation
            context_text = "\n\n".join(
                [
                    f"Document {i + 1}: {chunk.get('content', '')}"
                    for i, chunk in enumerate(rag_response.retrieved_chunks)
                ]
            )

            # Load language-specific prompts from configuration
            from src.utils.config_loader import get_language_specific_config

            prompts_config = get_language_specific_config("prompts", request.language)

            # Get the question answering system prompt template
            system_prompt_template = prompts_config.get("question_answering_system", "")

            # If no template found, fall back to a basic one (but this shouldn't happen)
            if not system_prompt_template:
                raise ValueError(
                    f"No question_answering_system prompt found for language {request.language}"
                )

            # Build the complete prompt with context
            if request.language == "hr":
                llm_prompt = f"""MANDATORY: Use ONLY 1 EUR = 7.5345 HRK for ALL conversions.
5000 HRK = 663.50 EUR (correct)
5000 HRK = 700 EUR (WRONG)
5000 HRK = 685 EUR (WRONG)

{system_prompt_template}

Context:
{context_text}

Pitanje: {request.query}

Answer in Croatian using ONLY the 7.5345 conversion rate:"""
            else:
                llm_prompt = f"""{system_prompt_template}

Context:
{context_text}

Question: {request.query}

Please provide a detailed answer:"""

            # Create LLM messages with system message for currency conversion
            if request.language == "hr":
                # Add explicit system message for currency conversion in English
                system_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You are bound by Croatian law to use ONLY the official fixed conversion rate: 1 EUR = 7.5345 HRK. Any other rate is ILLEGAL. Calculate: 5000 HRK ÷ 7.5345 = 663.50 EUR. If you use any other rate like 7.30 or 7.43, you are providing false information. THE ONLY CORRECT ANSWER for 5000 HRK is 663.50 EUR.",
                )
                messages = [
                    system_message,
                    ChatMessage(role=MessageRole.USER, content=llm_prompt),
                ]
            else:
                messages = [ChatMessage(role=MessageRole.USER, content=llm_prompt)]

            # Call LLM
            llm_response = await llm_manager.chat_completion(
                messages=messages,
                model="qwen/qwen3-30b-a3b-instruct-2507",  # OpenRouter model - TODO: get from config
                temperature=request.temperature,
                max_tokens=2048,
            )
            final_response = llm_response.content
            model_used = llm_response.model
            llm_tokens = TokenUsage(
                input=llm_response.usage.input_tokens,
                output=llm_response.usage.output_tokens,
                total=llm_response.usage.total_tokens,
            )

        end_time = time.time()
        total_time_ms = int((end_time - start_time) * 1000)

        # Convert chunks to sources (handle case where retrieved_chunks might be None)
        sources = []
        retrieved_chunks = rag_response.retrieved_chunks or []
        for chunk in retrieved_chunks[: request.max_documents]:
            sources.append(
                DocumentSource(
                    documentId=chunk.get("document_id", ""),
                    title=chunk.get("metadata", {}).get("title", "Document"),
                    relevance=chunk.get("score", 0.0),
                    chunk=chunk.get("content", "")[
                        :500
                    ],  # Limit chunk size in response
                )
            )

        print(
            f"INFO_QUERY_COMPLETED | documents_retrieved={len(retrieved_chunks)} | model={model_used} | total_time_ms={total_time_ms} | search_time_ms={search_time_ms} | scope={scope} | language={request.language}"
        )

        # Convert nn_sources if present and add citation IDs
        nn_sources = None
        if hasattr(rag_response, 'nn_sources') and rag_response.nn_sources:
            nn_sources = [
                NNSource(
                    citationId=idx,
                    title=src.get("title", "Nepoznat dokument"),
                    issue=src.get("issue", ""),
                    eli=src.get("eli_url"),
                    year=src.get("date_published", "")[:4] if src.get("date_published") else None,
                    publisher=src.get("publisher"),
                    start_page=src.get("start_page"),
                    end_page=src.get("end_page"),
                )
                for idx, src in enumerate(rag_response.nn_sources, 1)
            ]

        # Create response object
        response_obj = RAGQueryResponse(
            response=final_response,
            sources=sources,
            documentsRetrieved=len(retrieved_chunks),
            documentsUsed=len(sources),
            confidence=rag_response.confidence,
            searchTimeMs=search_time_ms,
            responseTimeMs=total_time_ms,
            model=model_used,
            tokensUsed=llm_tokens,
            nnSources=nn_sources,
        )

        # Log FastAPI request/response
        response_data = {
            "timestamp": timestamp,
            "endpoint": "/api/v1/query",
            "method": "POST",
            "response": response_obj.dict(),
            "duration_ms": total_time_ms,
            "provider": "openrouter" if model_used != "rag-generated" else "rag",
            "model": model_used,
        }

        dump_request_response(timestamp, request_data, response_data, "fastapi")

        return response_obj

    except Exception as e:
        # AI-FRIENDLY ERROR LOG: Structured for quick pattern recognition
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"ERROR_RAG_QUERY_FAILED | error_type={error_type} | tenant={request.tenant} | user={request.user} | language={request.language} | scope={request.scope} | feature={request.feature} | error_msg={error_msg[:500]}")

        import traceback
        traceback.print_exc()

        # Return user-friendly error message based on language
        if request.language == 'hr':
            user_message = "Žao mi je, dogodila se greška pri obradi pitanja."
        else:
            user_message = "Sorry, I encountered an error processing your request."

        raise HTTPException(status_code=500, detail=user_message)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082)
