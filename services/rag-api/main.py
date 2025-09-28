#!/usr/bin/env python
"""
FastAPI server for RAG service integration.
Bridges between TypeScript Web API and Python RAG system.
"""

import sys
import os
import time
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Add the rag-service directory to the Python path
sys.path.insert(0, "/app/src")

# Import RAG service components
from src.database.factory import create_database_provider
from src.generation.llm_provider import (
    LLMManager,
    ChatMessage,
    MessageRole,
)
from src.utils.factories import create_complete_rag_system
from src.models.multitenant_models import Tenant, User
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
    """Initialize LLM manager and database on startup."""
    global llm_manager

    print("INFO: RAG API server starting...")
    print("INFO: Loading configuration...")

    # Use config.toml values with environment variables for secrets
    database_config = {
        "provider": "postgresql",
        "postgresql": {
            "host": os.getenv("POSTGRES_HOST", "postgres"),  # Docker container name
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "ragdb"),
            "user": os.getenv("POSTGRES_USER", "raguser"),
            "password": os.getenv("POSTGRES_PASSWORD", "ragpass"),
        },
    }

    openrouter_config = {
        "api_key": os.getenv("OPENROUTER_API_KEY"),  # From environment
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3-30b-a3b-instruct-2507",
        "timeout": 30.0,
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }

    # Initialize database provider
    db_provider = create_database_provider(database_config)
    await db_provider.initialize(database_config)

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
            f"INFO: RAG query | scope: {scope} | requested_scope: {request.scope} | feature: {request.feature} | lang: {request.language}"
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

        # Execute RAG search
        rag_response = await rag_system.query(rag_query)

        search_time = time.time()
        search_time_ms = int((search_time - start_time) * 1000)

        # Check if we have documents OR a generated answer
        # The system can return an answer even if retrieved_chunks is None/empty
        if not rag_response.retrieved_chunks and not rag_response.answer:
            print(f"INFO: No documents found for query in scope: {scope}")

            # Return empty result response
            no_data_message = "I don't have relevant information in my knowledge base to answer your question. Please try rephrasing your question or asking about a different topic."

            return RAGQueryResponse(
                response=no_data_message,
                sources=[],
                documentsRetrieved=0,
                documentsUsed=0,
                confidence=0.7,
                searchTimeMs=search_time_ms,
                responseTimeMs=search_time_ms,
                model="system-generic",
                tokensUsed=TokenUsage(
                    input=len(request.query.split()),
                    output=len(no_data_message.split()),
                    total=len(request.query.split()) + len(no_data_message.split()),
                ),
            )

        # If RAG found documents, use the RAG-generated answer if available,
        # otherwise send to LLM for generation
        if rag_response.answer:
            # Use RAG-generated answer directly
            final_response = rag_response.answer
            model_used = "rag-generated"
            llm_tokens = TokenUsage(input=0, output=0, total=0)
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
            f"INFO: Query completed | documents: {len(retrieved_chunks)} | model: {model_used} | time: {total_time_ms}ms"
        )

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
        # All errors should fail with 500 status
        print(f"ERROR: RAG query failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082)
