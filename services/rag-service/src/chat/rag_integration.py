"""
RAG Integration Service for Chat System
Bridges chat conversations with document search functionality.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..pipeline.rag_system import RAGQuery, RAGResponse
from ..utils.config_loader import get_shared_config
from ..utils.logging_factory import get_system_logger, log_component_start, log_component_end
from ..utils.factories import create_complete_rag_system
from ..models.multitenant_models import Tenant, User


@dataclass
class RAGChatContext:
    """Context from RAG search for chat conversation."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    chunks: List[Dict[str, Any]]
    query_id: str


class RAGChatService:
    """
    Service that integrates RAG document search with chat conversations.
    Provides context-aware responses based on document retrieval.
    """

    def __init__(self, language: str = "hr", tenant_slug: str = "development", user_id: str = "web_user"):
        self.language = language
        self.tenant_slug = tenant_slug
        self.user_id = user_id
        self.rag_system = None
        self.logger = get_system_logger()
        self.config = get_shared_config()

    async def initialize(self) -> None:
        """Initialize the RAG system with proper configuration."""
        log_component_start("rag_chat_service", "initialize",
                           language=self.language, tenant=self.tenant_slug)

        try:
            # Create tenant and user objects for proper RAG initialization
            tenant = Tenant(
                id=f"tenant:{self.tenant_slug}",
                name=f"Tenant {self.tenant_slug.title()}",
                slug=self.tenant_slug
            )
            user = User(
                id=f"user:{self.user_id}",
                tenant_id=tenant.id,
                email=f"{self.user_id}@{self.tenant_slug}.example.com",
                username=self.user_id,
                full_name=f"User {self.user_id.title()}"
            )

            # Initialize RAG system using the same factory that works in rag.py CLI
            self.rag_system = create_complete_rag_system(
                language=self.language,
                tenant=tenant,
                user=user
            )
            await self.rag_system.initialize()

            self.logger.info("rag_chat_service", "initialize",
                           f"RAG system initialized for language: {self.language}")
            log_component_end("rag_chat_service", "initialize", "RAG system ready")

        except Exception as e:
            self.logger.error("rag_chat_service", "initialize", f"Failed to initialize RAG: {e}")
            raise

    async def search_documents(self, query_text: str, max_results: int = 5) -> Optional[RAGChatContext]:
        """
        Search documents using RAG system and return chat-ready context.

        Args:
            query_text: User query to search for
            max_results: Maximum number of document chunks to retrieve

        Returns:
            RAGChatContext with search results or None if no results
        """
        if not self.rag_system:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        log_component_start("rag_chat_service", "search_documents",
                           query_length=len(query_text), max_results=max_results)

        try:
            # Create RAG query
            rag_query = RAGQuery(
                text=query_text,
                language=self.language,
                user_id=self.user_id,
                max_results=max_results,
                context_filters={
                    "tenant_slug": self.tenant_slug,
                    "user_id": self.user_id
                }
            )

            # Execute RAG search
            rag_response: RAGResponse = await self.rag_system.query(rag_query)

            if not rag_response.retrieved_chunks:
                self.logger.info("rag_chat_service", "search_documents", "No documents found for query")
                log_component_end("rag_chat_service", "search_documents", "No results")
                return None

            # Convert to chat context
            context = RAGChatContext(
                answer=rag_response.answer,
                sources=rag_response.sources,
                confidence=rag_response.confidence,
                chunks=[
                    {
                        "content": chunk.get("content", ""),
                        "document_id": chunk.get("document_id", ""),
                        "score": chunk.get("score", 0.0),
                        "metadata": chunk.get("metadata", {})
                    }
                    for chunk in rag_response.retrieved_chunks
                ],
                query_id=rag_response.metadata.get("query_id", "")
            )

            self.logger.info("rag_chat_service", "search_documents",
                           f"Found {len(context.chunks)} chunks with confidence {context.confidence:.3f}")
            log_component_end("rag_chat_service", "search_documents",
                             f"Retrieved {len(context.chunks)} chunks")

            return context

        except Exception as e:
            self.logger.error("rag_chat_service", "search_documents", f"RAG search failed: {e}")
            log_component_end("rag_chat_service", "search_documents", f"Error: {e}")
            return None

    def should_use_rag(self, user_message: str) -> bool:
        """
        Determine if a user message should trigger RAG document search.

        Args:
            user_message: User's chat message

        Returns:
            True if message should use RAG search
        """
        # RAG is now the default behavior for all messages
        # Only skip RAG for very short messages or greetings
        message_lower = user_message.lower().strip()

        # Skip RAG for very short greetings or simple responses
        skip_rag_patterns = [
            "hi", "hello", "hey", "ok", "yes", "no", "thanks", "thank you"
        ]

        # Skip RAG if message is too short or is a simple greeting
        if len(message_lower) <= 3 or message_lower in skip_rag_patterns:
            return False

        # Use RAG for all other messages (this is now the default)
        return True

    def format_rag_context_for_llm(self, context: RAGChatContext, user_message: str) -> str:
        """
        Format RAG context into a prompt section for the LLM.

        Args:
            context: RAG search context
            user_message: Original user message

        Returns:
            Formatted context string for LLM prompt
        """
        if not context or not context.chunks:
            return ""

        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(context.chunks[:3], 1):  # Use top 3 chunks
            context_parts.append(f"Document {i}: {chunk['content'][:500]}...")

        context_text = "\n\n".join(context_parts)

        # Create structured prompt
        rag_prompt = f"""
Based on the following document context, please answer the user's question: "{user_message}"

DOCUMENT CONTEXT:
{context_text}

Please provide a helpful answer based on the documents above. If the documents don't contain relevant information, say so clearly.
"""

        return rag_prompt

    async def close(self) -> None:
        """Close the RAG system and cleanup resources."""
        if self.rag_system:
            await self.rag_system.close()
            self.logger.info("rag_chat_service", "close", "RAG system closed")


# Factory function
def create_rag_chat_service(language: str = "hr", tenant_slug: str = "development",
                           user_id: str = "web_user") -> RAGChatService:
    """Create RAG chat service with specified configuration."""
    return RAGChatService(language=language, tenant_slug=tenant_slug, user_id=user_id)