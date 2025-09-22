"""
Unified chat service integrating LLM providers with chat persistence and RAG.
Provides high-level chat operations with conversation management and document retrieval.
"""

import json
import time
from typing import AsyncIterator, Dict, List, Optional, Any

from ..database.protocols import DatabaseProvider
from ..generation.llm_provider import (
    ChatMessage, ChatRequest, ChatResponse, MessageRole,
    UnifiedLLMManager, StreamChunk
)
from .chat_persistence import ChatPersistenceManager, ChatConversation
from .rag_integration import RAGChatService, RAGChatContext
from ..models.multitenant_models import Tenant, User, DocumentScope
from ..utils.logging_factory import get_system_logger, log_component_start, log_component_end


from ..utils.json_logging import write_debug_json


class ChatService:
    """
    High-level chat service that combines LLM generation with persistence and RAG.
    Manages conversations, message history, and document retrieval for enhanced responses.
    """

    def __init__(self, llm_manager: UnifiedLLMManager, db_provider: DatabaseProvider):
        self.llm_manager = llm_manager
        self.chat_persistence = ChatPersistenceManager(db_provider)
        self.logger = get_system_logger()

        # RAG chat services will be initialized per tenant/user/language
        self._rag_services: Dict[str, RAGChatService] = {}

    async def initialize(self) -> None:
        """Initialize chat service and database schema."""
        log_component_start("chat_service", "initialize")
        await self.chat_persistence.initialize_schema()
        log_component_end("chat_service", "initialize", "Chat service ready")

    def _get_user_language_preference(self, tenant_slug: str, user_id: str) -> str:
        """Hard-coded Croatian language as requested."""
        return "hr"

    async def _get_rag_service(self, tenant_slug: str, user_id: str, language: str) -> Optional[RAGChatService]:
        """Get or create RAG chat service for tenant/user/language combination."""
        rag_key = f"{tenant_slug}_{user_id}_{language}"

        if rag_key not in self._rag_services:
            try:
                # Create RAG chat service
                rag_service = RAGChatService(
                    language=language,
                    tenant_slug=tenant_slug,
                    user_id=user_id
                )
                await rag_service.initialize()

                self._rag_services[rag_key] = rag_service
                self.logger.info("chat_service", "_get_rag_service",
                               f"Created RAG service for {tenant_slug}/{user_id}/{language}")

            except Exception as e:
                self.logger.error("chat_service", "_get_rag_service",
                                f"Failed to create RAG service: {e}")
                return None

        return self._rag_services.get(rag_key)

    async def start_conversation(self, tenant_slug: str, user_id: str,
                                title: Optional[str] = None) -> ChatConversation:
        """Start new chat conversation."""
        log_component_start("chat_service", "start_conversation",
                          tenant=tenant_slug, user=user_id, has_title=title is not None)

        conversation = await self.chat_persistence.create_conversation(
            tenant_slug=tenant_slug,
            user_id=user_id,
            title=title
        )

        self.logger.info("chat_service", "start_conversation",
                        f"Started conversation {conversation.conversation_id}")
        log_component_end("chat_service", "start_conversation", f"Conversation {conversation.conversation_id} started")

        return conversation

    async def send_message(self, conversation_id: str, user_message: str,
                          system_prompt: Optional[str] = None,
                          rag_context: Optional[List[str]] = None,
                          model: Optional[str] = None,
                          enable_rag: bool = True,  # RAG enabled by default
                          **llm_kwargs) -> ChatResponse:
        """
        Send message and get response with conversation persistence.

        Args:
            conversation_id: ID of the conversation
            user_message: User's message
            system_prompt: Optional system prompt for this exchange
            rag_context: Optional RAG context documents
            model: Optional specific model to use
            **llm_kwargs: Additional LLM parameters (temperature, max_tokens, etc.)
        """
        log_component_start("chat_service", "send_message",
                          conversation_id=conversation_id, message_length=len(user_message),
                          has_context=rag_context is not None, has_system=system_prompt is not None)

        # Get conversation metadata to determine tenant/user/language
        conversation = await self.chat_persistence.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Get user's preferred language from database or use explicit override
        if "language" in llm_kwargs:
            # Explicit language override provided
            language = llm_kwargs["language"]
        else:
            # Hard-coded Croatian language
            language = self._get_user_language_preference(
                tenant_slug=conversation.tenant_slug,
                user_id=conversation.user_id
            )

        # Get or create RAG service for this tenant/user/language
        rag_service = await self._get_rag_service(
            tenant_slug=conversation.tenant_slug,
            user_id=conversation.user_id,
            language=language
        )

        # Perform RAG retrieval if no explicit context provided and RAG service available
        retrieved_context = rag_context
        rag_context_obj: Optional[RAGChatContext] = None

        if enable_rag and not retrieved_context and rag_service and rag_service.should_use_rag(user_message):
            try:
                # Execute RAG search
                rag_context_obj = await rag_service.search_documents(
                    query_text=user_message,
                    max_results=llm_kwargs.get("rag_top_k", 3)
                )

                # If RAG found results and has an answer, use it directly (like CLI)
                if rag_context_obj and rag_context_obj.answer:
                    # Store user message
                    await self.chat_persistence.add_message(
                        conversation_id=conversation_id,
                        role=MessageRole.USER,
                        content=user_message,
                        metadata={"timestamp": time.time()}
                    )

                    # Store RAG-generated response
                    await self.chat_persistence.add_message(
                        conversation_id=conversation_id,
                        role=MessageRole.ASSISTANT,
                        content=rag_context_obj.answer,
                        metadata={
                            "model": "rag-generated",
                            "provider": "rag",
                            "rag_confidence": rag_context_obj.confidence,
                            "rag_sources": len(rag_context_obj.sources),
                            "rag_chunks": len(rag_context_obj.chunks),
                            "query_id": rag_context_obj.query_id
                        }
                    )

                    # Create ChatResponse object to match expected return type
                    from ..generation.llm_provider import ChatResponse, ProviderType, FinishReason, TokenUsage
                    import uuid

                    # Get the actual provider from the LLM manager config
                    provider_name = self.llm_manager.primary_provider
                    provider_enum = ProviderType.OPENROUTER if provider_name == "openrouter" else ProviderType.OLLAMA

                    # Get the actual model from the provider config
                    provider_config = self.llm_manager.config[provider_name]
                    actual_model = provider_config.get("model", "unknown")

                    response = ChatResponse(
                        id=f"rag-{uuid.uuid4().hex[:8]}",
                        content=rag_context_obj.answer,
                        model=actual_model,
                        provider=provider_enum,
                        finish_reason=FinishReason.COMPLETED,
                        usage=TokenUsage(
                            input_tokens=len(user_message.split()),
                            output_tokens=len(rag_context_obj.answer.split()),
                            total_tokens=len(user_message.split()) + len(rag_context_obj.answer.split())
                        )
                    )

                    # Log and store RAG response details as JSON
                    import json
                    import os
                    from datetime import datetime

                    rag_response_details = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "RAG_DIRECT_ANSWER",
                        "id": response.id,
                        "content": response.content,
                        "content_length": len(response.content),
                        "model": response.model,
                        "provider": response.provider.value if hasattr(response.provider, 'value') else str(response.provider),
                        "finish_reason": response.finish_reason.value if hasattr(response.finish_reason, 'value') else str(response.finish_reason),
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.total_tokens
                        },
                        "conversation_id": conversation_id,
                        "rag_confidence": rag_context_obj.confidence,
                        "rag_sources_count": len(rag_context_obj.sources),
                        "rag_chunks_count": len(rag_context_obj.chunks),
                        "query_id": rag_context_obj.query_id,
                        "user_message": user_message
                    }

                    # Store RAG response to file with readable markdown formatting
                    os.makedirs("./logs/chat_debug", exist_ok=True)
                    rag_response_file = f"./logs/chat_debug/{datetime.now().strftime('%Y%m%d_%H%M%S')}_rag_response_{conversation_id[:8]}.json"
                    write_debug_json(rag_response_file, rag_response_details)

                    self.logger.trace("chat_service", "rag_response", f"RAG_RESPONSE_JSON: {json.dumps(rag_response_details, indent=2, ensure_ascii=False)}")

                    self.logger.info("chat_service", "send_message",
                                   f"Used RAG answer directly: {len(rag_context_obj.answer)} chars, confidence: {rag_context_obj.confidence:.3f}")
                    log_component_end("chat_service", "send_message", "RAG answer used directly")

                    return response

                # Extract context from retrieved chunks for fallback
                if rag_context_obj:
                    retrieved_context = [chunk["content"] for chunk in rag_context_obj.chunks]
                    self.logger.info("chat_service", "send_message",
                                   f"Retrieved {len(retrieved_context)} chunks from RAG (confidence: {rag_context_obj.confidence:.3f})")
                else:
                    self.logger.info("chat_service", "send_message", "No relevant documents found")

            except Exception as e:
                self.logger.error("chat_service", "send_message", f"RAG retrieval failed: {e}")
                # Continue without RAG context if retrieval fails

        # Get conversation history
        history = await self.chat_persistence.get_recent_messages(
            conversation_id=conversation_id,
            count=llm_kwargs.get("context_messages", 10)
        )

        # Build message list for LLM
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

        # Add RAG context as system message if provided
        if retrieved_context and rag_service:
            context_content = rag_service.format_rag_context_for_llm(rag_context_obj, user_message)
            if context_content:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=context_content))
        elif retrieved_context:
            # Fallback for manually provided context
            context_content = "Context information:\n\n" + "\n\n".join(retrieved_context)
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=context_content))

        # Add conversation history
        messages.extend(history)

        # Add current user message
        messages.append(ChatMessage(role=MessageRole.USER, content=user_message))

        # Generate response
        try:
            # Log and store full LLM request details as JSON
            import json
            import os
            from datetime import datetime

            request_details = {
                "timestamp": datetime.now().isoformat(),
                "messages": [{"role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role), "content": msg.content} for msg in messages],
                "model": model,
                "primary_provider": self.llm_manager.primary_provider,
                "llm_kwargs": llm_kwargs,
                "conversation_id": conversation_id,
                "enable_rag": enable_rag,
                "rag_context_available": retrieved_context is not None,
                "rag_chunks_count": len(retrieved_context) if retrieved_context else 0
            }

            # Store request to file
            os.makedirs("./logs/chat_debug", exist_ok=True)
            request_file = f"./logs/chat_debug/request_{conversation_id[:8]}_{datetime.now().strftime('%H%M%S')}.json"
            with open(request_file, 'w', encoding='utf-8') as f:
                json.dump(request_details, f, indent=2, ensure_ascii=False)

            self.logger.trace("chat_service", "llm_request", f"LLM_REQUEST_JSON: {json.dumps(request_details, indent=2, ensure_ascii=False)}")

            response = await self.llm_manager.chat_completion(
                messages=messages,
                model=model,
                **llm_kwargs
            )

            # Log and store full LLM response details as JSON
            response_details = {
                "timestamp": datetime.now().isoformat(),
                "id": response.id,
                "content": response.content,
                "content_length": len(response.content),
                "model": response.model,
                "provider": response.provider.value if hasattr(response.provider, 'value') else str(response.provider),
                "finish_reason": response.finish_reason.value if hasattr(response.finish_reason, 'value') else str(response.finish_reason),
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "conversation_id": conversation_id,
                "request_file": request_file
            }

            # Store response to file
            response_file = f"./logs/chat_debug/response_{conversation_id[:8]}_{datetime.now().strftime('%H%M%S')}.json"
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response_details, f, indent=2, ensure_ascii=False)

            self.logger.trace("chat_service", "llm_response", f"LLM_RESPONSE_JSON: {json.dumps(response_details, indent=2, ensure_ascii=False)}")

            # Store user message
            await self.chat_persistence.add_message(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_message,
                metadata={"timestamp": time.time()}
            )

            # Store assistant response
            await self.chat_persistence.add_message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=response.content,
                metadata={
                    "model": response.model,
                    "provider": response.provider.value,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "finish_reason": response.finish_reason.value
                }
            )

            self.logger.info("chat_service", "send_message",
                           f"Generated {len(response.content)} chars using {response.model}")
            log_component_end("chat_service", "send_message",
                             f"Message exchange completed: {response.usage.total_tokens} tokens")

            return response

        except Exception as e:
            self.logger.error("chat_service", "send_message", f"Message generation failed: {e}")
            # Still store the user message even if generation fails
            await self.chat_persistence.add_message(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_message,
                metadata={"timestamp": time.time(), "error": str(e)}
            )
            raise

    async def send_message_streaming(self, conversation_id: str, user_message: str,
                                   system_prompt: Optional[str] = None,
                                   rag_context: Optional[List[str]] = None,
                                   model: Optional[str] = None,
                                   enable_rag: bool = True,  # RAG enabled by default
                                   **llm_kwargs) -> AsyncIterator[str]:
        """
        Send message and get streaming response with conversation persistence.
        """
        log_component_start("chat_service", "send_message_streaming",
                          conversation_id=conversation_id, message_length=len(user_message))

        # Get conversation metadata to determine tenant/user/language
        conversation = await self.chat_persistence.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Get user's preferred language from database or use explicit override
        if "language" in llm_kwargs:
            # Explicit language override provided
            language = llm_kwargs["language"]
        else:
            # Hard-coded Croatian language
            language = self._get_user_language_preference(
                tenant_slug=conversation.tenant_slug,
                user_id=conversation.user_id
            )

        # Get or create RAG service for this tenant/user/language
        rag_service = await self._get_rag_service(
            tenant_slug=conversation.tenant_slug,
            user_id=conversation.user_id,
            language=language
        )

        # Perform RAG retrieval if no explicit context provided and RAG service available
        retrieved_context = rag_context
        rag_context_obj: Optional[RAGChatContext] = None

        if enable_rag and not retrieved_context and rag_service and rag_service.should_use_rag(user_message):
            try:
                # Execute RAG search
                rag_context_obj = await rag_service.search_documents(
                    query_text=user_message,
                    max_results=llm_kwargs.get("rag_top_k", 3)
                )

                # If RAG found results and has an answer, use it directly (like CLI)
                if rag_context_obj and rag_context_obj.answer:
                    # Store user message
                    await self.chat_persistence.add_message(
                        conversation_id=conversation_id,
                        role=MessageRole.USER,
                        content=user_message,
                        metadata={"timestamp": time.time()}
                    )

                    # Store RAG-generated response
                    await self.chat_persistence.add_message(
                        conversation_id=conversation_id,
                        role=MessageRole.ASSISTANT,
                        content=rag_context_obj.answer,
                        metadata={
                            "model": "rag-generated",
                            "provider": "rag",
                            "rag_confidence": rag_context_obj.confidence,
                            "rag_sources": len(rag_context_obj.sources),
                            "rag_chunks": len(rag_context_obj.chunks),
                            "query_id": rag_context_obj.query_id,
                            "streaming": True
                        }
                    )

                    self.logger.info("chat_service", "send_message_streaming",
                                   f"Using RAG answer directly: {len(rag_context_obj.answer)} chars, confidence: {rag_context_obj.confidence:.3f}")

                    # Yield the RAG answer as streaming chunks
                    answer_words = rag_context_obj.answer.split()
                    for i, word in enumerate(answer_words):
                        yield word + (" " if i < len(answer_words) - 1 else "")

                    log_component_end("chat_service", "send_message_streaming", "RAG answer streamed directly")
                    return

                # Extract context from retrieved chunks for fallback
                if rag_context_obj:
                    retrieved_context = [chunk["content"] for chunk in rag_context_obj.chunks]
                    self.logger.info("chat_service", "send_message_streaming",
                                   f"Retrieved {len(retrieved_context)} chunks from RAG (confidence: {rag_context_obj.confidence:.3f})")
                else:
                    self.logger.info("chat_service", "send_message_streaming", "No relevant documents found")

            except Exception as e:
                self.logger.error("chat_service", "send_message_streaming", f"RAG retrieval failed: {e}")
                # Continue without RAG context if retrieval fails

        # Get conversation history
        history = await self.chat_persistence.get_recent_messages(
            conversation_id=conversation_id,
            count=llm_kwargs.get("context_messages", 10)
        )

        # Build message list (same as non-streaming)
        messages = []

        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

        # Add RAG context as system message if provided (streaming version)
        if retrieved_context and rag_service:
            context_content = rag_service.format_rag_context_for_llm(rag_context_obj, user_message)
            if context_content:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=context_content))
        elif retrieved_context:
            # Fallback for manually provided context
            context_content = "Context information:\n\n" + "\n\n".join(retrieved_context)
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=context_content))

        messages.extend(history)
        messages.append(ChatMessage(role=MessageRole.USER, content=user_message))

        # Store user message immediately
        await self.chat_persistence.add_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=user_message,
            metadata={"timestamp": time.time()}
        )

        # Stream response and collect content
        collected_content = []
        model_used = model or self.llm_manager._get_default_model()

        try:
            async for chunk in self.llm_manager.stream_chat_completion(
                messages=messages,
                model=model,
                stream=True,
                **llm_kwargs
            ):
                if chunk.content:
                    collected_content.append(chunk.content)
                    yield chunk.content

                # If streaming is complete, store the full response
                if chunk.finish_reason:
                    full_response = "".join(collected_content)
                    await self.chat_persistence.add_message(
                        conversation_id=conversation_id,
                        role=MessageRole.ASSISTANT,
                        content=full_response,
                        metadata={
                            "model": model_used,
                            "streaming": True,
                            "finish_reason": chunk.finish_reason.value,
                            "timestamp": time.time()
                        }
                    )
                    break

        except Exception as e:
            self.logger.error("chat_service", "send_message_streaming", f"Streaming failed: {e}")
            # Store partial response if any content was generated
            if collected_content:
                partial_response = "".join(collected_content)
                await self.chat_persistence.add_message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=partial_response,
                    metadata={
                        "model": model_used,
                        "streaming": True,
                        "error": str(e),
                        "partial": True,
                        "timestamp": time.time()
                    }
                )
            raise

        log_component_end("chat_service", "send_message_streaming", "Streaming completed")

    async def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """Get conversation message history."""
        return await self.chat_persistence.get_conversation_messages(conversation_id, limit)

    async def list_user_conversations(self, tenant_slug: str, user_id: str, limit: int = 50) -> List[ChatConversation]:
        """List conversations for a user."""
        return await self.chat_persistence.list_user_conversations(tenant_slug, user_id, limit)

    async def update_conversation_title(self, conversation_id: str, title: str) -> None:
        """Update conversation title."""
        await self.chat_persistence.update_conversation_title(conversation_id, title)

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete conversation and all messages."""
        await self.chat_persistence.delete_conversation(conversation_id)

    async def get_conversation(self, conversation_id: str) -> Optional[ChatConversation]:
        """Get conversation metadata."""
        return await self.chat_persistence.get_conversation(conversation_id)


# Factory function
def create_chat_service(llm_config: Dict[str, Any], db_provider: DatabaseProvider) -> ChatService:
    """Create chat service with LLM manager and database provider."""
    from ..generation.llm_provider import UnifiedLLMManager

    llm_manager = UnifiedLLMManager(llm_config)
    return ChatService(llm_manager, db_provider)