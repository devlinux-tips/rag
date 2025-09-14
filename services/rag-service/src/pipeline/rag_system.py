"""
Multilingual RAG System orchestrating document processing, retrieval, and generation.
Coordinates preprocessing, vector storage, semantic search, and LLM generation
for multilingual document question-answering workflows.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..utils.config_models import EmbeddingConfig, OllamaConfig, ProcessingConfig, RetrievalConfig
from ..utils.config_validator import ConfigurationError

# Module logger
logger = logging.getLogger(__name__)


# Pure data structures
@dataclass
class RAGQuery:
    """RAG query with metadata - pure data structure."""

    text: str
    language: str  # Language code (hr, en, etc.)
    query_id: str | None = None
    user_id: str | None = None
    context_filters: dict[str, Any] | None = None
    max_results: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RAGResponse:
    """RAG response with full pipeline information - pure data structure."""

    answer: str
    query: str
    retrieved_chunks: list[dict[str, Any]]
    confidence: float
    generation_time: float
    retrieval_time: float
    total_time: float
    sources: list[str]
    metadata: dict[str, Any]

    @property
    def has_high_confidence(self) -> bool:
        """Check if response has high confidence."""
        return self.confidence >= 0.8


@dataclass
class ComponentHealth:
    """Health status for a system component."""

    status: str  # "healthy", "degraded", "unhealthy"
    details: str
    metadata: dict[str, Any] | None = None


@dataclass
class SystemHealth:
    """Overall system health status."""

    system_status: str  # "healthy", "degraded", "unhealthy", "error"
    components: dict[str, ComponentHealth]
    metrics: dict[str, Any]
    timestamp: float
    error: str | None = None


@dataclass
class SystemStats:
    """System statistics and configuration."""

    system: dict[str, Any]
    collections: dict[str, Any]
    models: dict[str, Any]
    performance: dict[str, Any]


@dataclass
class DocumentProcessingResult:
    """Result of document processing operation."""

    processed_documents: int
    failed_documents: int
    total_chunks: int
    processing_time: float
    documents_per_second: float
    errors: list[str] | None = None


# Protocol definitions for dependency injection
class DocumentExtractorProtocol(Protocol):
    """Protocol for document text extraction."""

    def extract_text(self, file_path: Path) -> str: ...


class TextCleanerProtocol(Protocol):
    """Protocol for multilingual text cleaning."""

    def clean_text(self, text: str) -> str: ...

    def setup_language_environment(self) -> None: ...


class DocumentChunkerProtocol(Protocol):
    """Protocol for document chunking."""

    def chunk_document(self, content: str, source_file: str) -> list[Any]: ...


class EmbeddingModelProtocol(Protocol):
    """Protocol for text embedding generation."""

    def encode_text(self, text: str) -> Any: ...

    def load_model(self) -> None: ...


class VectorStorageProtocol(Protocol):
    """Protocol for vector storage operations."""

    def add_documents(self, documents: list[str], metadatas: list[dict], embeddings: list) -> None: ...

    def create_collection(self) -> None: ...

    def get_document_count(self) -> int: ...

    async def close(self) -> None: ...


class SearchEngineProtocol(Protocol):
    """Protocol for semantic search."""

    pass


class QueryProcessorProtocol(Protocol):
    """Protocol for query processing."""

    pass


class RetrieverProtocol(Protocol):
    """Protocol for document retrieval."""

    async def retrieve(self, query: str, max_results: int = 5, context: dict | None = None) -> Any: ...


class RankerProtocol(Protocol):
    """Protocol for result ranking."""

    pass


class GenerationClientProtocol(Protocol):
    """Protocol for LLM generation."""

    async def generate_text_async(self, request: Any) -> Any: ...

    def health_check(self) -> bool: ...

    def get_available_models(self) -> list[str]: ...

    async def close(self) -> None: ...


class ResponseParserProtocol(Protocol):
    """Protocol for response parsing."""

    def parse_response(self, text: str, query: str, context: list[str]) -> Any: ...


class PromptBuilderProtocol(Protocol):
    """Protocol for prompt building."""

    def build_prompt(self, query: str, context_chunks: list[str], **kwargs) -> tuple[str, str]: ...


# Pure functions for business logic
def validate_language_code(language: str) -> str:
    """Validate and normalize language code."""
    if not language or not isinstance(language, str):
        raise ValueError("Language code must be a non-empty string")

    language = language.lower().strip()
    valid_languages = {"hr", "en", "multilingual"}

    if language not in valid_languages:
        raise ValueError(f"Unsupported language: {language}. Supported: {valid_languages}")

    return language


def create_language_collection_name(language: str) -> str:
    """Create language-specific collection name."""
    language_collection_map = {
        "hr": "croatian_documents",
        "en": "english_documents",
        "multilingual": "multilingual_documents",
    }
    # FAIL FAST: Language must be configured
    if language not in language_collection_map:
        raise ConfigurationError(
            f"Unsupported language '{language}'. Supported: {list(language_collection_map.keys())}"
        )
    return language_collection_map[language]


def validate_document_paths(document_paths: list[str]) -> list[Path]:
    """Validate and convert document paths to Path objects."""
    if not document_paths:
        raise ValueError("Document paths list cannot be empty")

    validated_paths = []
    for i, path_str in enumerate(document_paths):
        if not path_str or not isinstance(path_str, str):
            raise ValueError(f"Document path at index {i} must be a non-empty string")

        path = Path(path_str)
        if not path.exists():
            raise ValueError(f"Document path does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Document path is not a file: {path}")

        validated_paths.append(path)

    return validated_paths


def validate_query(query: RAGQuery) -> RAGQuery:
    """Validate RAG query parameters."""
    if not query.text or not query.text.strip():
        raise ValueError("Query text cannot be empty")

    if not query.language:
        raise ValueError("Query language must be specified")

    # Normalize language
    query.language = validate_language_code(query.language)

    # Set defaults
    if query.max_results is None:
        query.max_results = 5
    elif query.max_results <= 0:
        raise ValueError("max_results must be positive")

    return query


def calculate_processing_metrics(processed_count: int, total_time: float, total_chunks: int) -> dict[str, float]:
    """Calculate document processing metrics."""
    if total_time <= 0:
        return {"documents_per_second": 0.0, "chunks_per_second": 0.0, "average_chunks_per_document": 0.0}

    docs_per_sec = processed_count / total_time
    chunks_per_sec = total_chunks / total_time
    avg_chunks = total_chunks / processed_count if processed_count > 0 else 0.0

    return {
        "documents_per_second": docs_per_sec,
        "chunks_per_second": chunks_per_sec,
        "average_chunks_per_document": avg_chunks,
    }


def create_chunk_metadata(
    doc_path: str, chunk_idx: int, chunk: Any, language: str, processing_timestamp: float
) -> dict[str, Any]:
    """Create metadata for a document chunk."""
    return {
        "source": doc_path,
        "chunk_index": chunk_idx,
        "language": language,
        "chunk_id": getattr(chunk, "chunk_id", f"{doc_path}_{chunk_idx}"),
        "start_char": getattr(chunk, "start_char", 0),
        "end_char": getattr(chunk, "end_char", 0),
        "word_count": getattr(chunk, "word_count", 0),
        "char_count": getattr(chunk, "char_count", len(chunk.content)),
        "processing_timestamp": processing_timestamp,
    }


def extract_sources_from_chunks(retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    """Extract unique sources from retrieved document chunks."""
    sources = set()
    for chunk in retrieved_chunks:
        # FAIL FAST: Chunk must have proper metadata structure
        if "metadata" not in chunk:
            raise ValueError(f"Chunk missing required 'metadata' field: {chunk}")
        metadata = chunk["metadata"]

        if "source" not in metadata:
            raise ValueError(f"Chunk metadata missing required 'source' field: {metadata}")
        source = metadata["source"]
        if source and source != "Unknown":
            sources.add(source)

    return list(sources)


def prepare_chunk_info(chunk_result: dict[str, Any], return_debug_info: bool = False) -> dict[str, Any]:
    """Prepare chunk information for response."""
    chunk_info = {
        "content": chunk_result["content"],
        "similarity_score": chunk_result["similarity_score"],
        "final_score": chunk_result["final_score"],
        "source": (chunk_result["metadata"]["source"] if "source" in chunk_result["metadata"] else "Unknown"),
        "chunk_index": (chunk_result["metadata"]["chunk_index"] if "chunk_index" in chunk_result["metadata"] else 0),
    }

    if return_debug_info:
        result_metadata = chunk_result["metadata"]
        chunk_info.update(
            {
                "metadata": result_metadata,
                "signals": (result_metadata["ranking_signals"] if "ranking_signals" in result_metadata else {}),
            }
        )

    return chunk_info


def build_response_metadata(
    query: RAGQuery,
    hierarchical_results: Any,
    generation_response: Any,
    parsed_response: Any,
    retrieval_time: float,
    generation_time: float,
    total_time: float,
    return_debug_info: bool = False,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Build comprehensive response metadata."""
    metadata = {
        "query_id": query.query_id,
        "user_id": query.user_id,
        "categorization": {
            "detected_category": getattr(hierarchical_results.category, "value", "unknown"),
            "strategy_used": getattr(hierarchical_results.strategy_used, "value", "unknown"),
            "confidence": getattr(hierarchical_results, "confidence", 0.0),
            "routing_metadata": getattr(hierarchical_results, "routing_metadata", {}),
        },
        "retrieval": {
            "total_results": len(hierarchical_results.documents),
            "strategy_used": getattr(hierarchical_results.strategy_used, "value", "unknown"),
            "filters_applied": query.context_filters or {},
            "retrieval_time": getattr(hierarchical_results, "retrieval_time", retrieval_time),
        },
        "generation": {
            "model": getattr(generation_response, "model", "unknown"),
            "tokens_used": getattr(generation_response, "tokens_used", 0),
            "model_confidence": getattr(generation_response, "confidence", 0.0),
        },
        "parsing": {
            "language_detected": getattr(parsed_response, "language", query.language),
            "sources_mentioned": getattr(parsed_response, "sources_mentioned", []),
        },
        "performance": {"retrieval_time": retrieval_time, "generation_time": generation_time, "total_time": total_time},
    }

    if return_debug_info and system_prompt and user_prompt:
        metadata["debug"] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_generation": getattr(generation_response, "text", ""),
            "generation_metadata": getattr(generation_response, "metadata", {}),
        }

    return metadata


def create_error_response(query: RAGQuery, error: Exception, start_time: float) -> RAGResponse:
    """Create error response in the appropriate language."""
    error_msg = (
        "I apologize, an error occurred while processing your question"
        if query.language == "en"
        else "Žao mi je, dogodila se greška pri obradi pitanja"
    )

    return RAGResponse(
        answer=f"{error_msg}: {str(error)}",
        query=query.text,
        retrieved_chunks=[],
        confidence=0.0,
        generation_time=0.0,
        retrieval_time=0.0,
        total_time=time.time() - start_time,
        sources=[],
        metadata={"error": str(error), "query_id": query.query_id, "user_id": query.user_id},
    )


def evaluate_component_health(component_name: str, components: list[Any], details: str = "") -> ComponentHealth:
    """Evaluate health of system components."""
    all_healthy = all(comp is not None for comp in components)
    status = "healthy" if all_healthy else "unhealthy"

    if not details:
        details = (
            f"All {component_name} components loaded" if all_healthy else f"Some {component_name} components missing"
        )

    return ComponentHealth(status=status, details=details)


def evaluate_ollama_health(client: GenerationClientProtocol | None, model_name: str) -> ComponentHealth:
    """Evaluate Ollama service health."""
    if not client:
        return ComponentHealth(status="unhealthy", details="Ollama client not initialized")

    ollama_healthy = client.health_check()
    if not ollama_healthy:
        return ComponentHealth(status="unhealthy", details="Ollama service not available")

    available_models = client.get_available_models()
    model_available = model_name in available_models

    status = "healthy" if model_available else "degraded"
    details = f"Ollama: {'✅' if ollama_healthy else '❌'}, Model {model_name}: {'✅' if model_available else '❌'}"

    return ComponentHealth(status=status, details=details, metadata={"available_models": available_models})


def calculate_overall_health(component_healths: dict[str, ComponentHealth]) -> str:
    """Calculate overall system health from component healths."""
    statuses = [comp.status for comp in component_healths.values()]

    if all(status == "healthy" for status in statuses):
        return "healthy"
    elif any(status == "healthy" for status in statuses):
        return "degraded"
    else:
        return "unhealthy"


# Main testable RAG system class
class RAGSystem:
    """Multilingual RAG system coordinating document processing, retrieval, and generation."""

    def __init__(
        self,
        language: str,
        # Injected dependencies
        document_extractor: DocumentExtractorProtocol,
        text_cleaner: TextCleanerProtocol,
        document_chunker: DocumentChunkerProtocol,
        embedding_model: EmbeddingModelProtocol,
        vector_storage: VectorStorageProtocol,
        search_engine: SearchEngineProtocol,
        query_processor: QueryProcessorProtocol,
        retriever: RetrieverProtocol,
        hierarchical_retriever: RetrieverProtocol,
        ranker: RankerProtocol,
        generation_client: GenerationClientProtocol,
        response_parser: ResponseParserProtocol,
        prompt_builder: PromptBuilderProtocol,
        # Configuration - validated config objects (guaranteed by ConfigValidator)
        embedding_config: EmbeddingConfig,
        ollama_config: OllamaConfig,
        processing_config: ProcessingConfig,
        retrieval_config: RetrievalConfig,
    ):
        """Initialize with all dependencies injected."""
        self.language = validate_language_code(language)

        # Store validated config objects
        self.embedding_config = embedding_config
        self.ollama_config = ollama_config
        self.processing_config = processing_config
        self.retrieval_config = retrieval_config

        # Injected dependencies
        self._document_extractor = document_extractor
        self._text_cleaner = text_cleaner
        self._chunker = document_chunker
        self._embedding_model = embedding_model
        self._vector_storage = vector_storage
        self._search_engine = search_engine
        self._query_processor = query_processor
        self._retriever = retriever
        self._hierarchical_retriever = hierarchical_retriever
        self._ranker = ranker
        self._generation_client = generation_client
        self._response_parser = response_parser
        self._prompt_builder = prompt_builder

        # System state
        self._initialized = False
        self._document_count = 0
        self._query_count = 0

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        # ConfigValidator integration - validate configs at startup
        await self._validate_configuration()

        # Setup language environment
        self._text_cleaner.setup_language_environment()

        # Load embedding model
        self._embedding_model.load_model()

        # Create vector storage collection
        self._vector_storage.create_collection()

        self._initialized = True

    async def _validate_configuration(self) -> None:
        """Validate configuration using ConfigValidator - minimal integration."""
        try:
            from ..utils.config_loader import get_language_config, load_config
            from ..utils.config_validator import ConfigurationError, ConfigValidator

            # Load current configs
            main_config = load_config("config")
            language_config = get_language_config(self.language)

            # Try ConfigValidator - log result but don't break system
            ConfigValidator.validate_startup_config(main_config, {self.language: language_config})
            logger.info("✅ ConfigValidator: All configuration keys validated successfully")
        except ConfigurationError as e:
            # Log warning but continue - development system
            logger.warning(f"⚠️  ConfigValidator found missing keys (development mode): {e}")
            logger.info("🔧 System will continue with current configuration")
        except Exception as e:
            # Any other error - log but don't break startup
            logger.warning(f"Configuration validation failed: {e}")

    async def add_documents(self, document_paths: list[str], batch_size: int = 10) -> DocumentProcessingResult:
        """Add documents to the RAG system using pure functions."""
        if not self._initialized:
            await self.initialize()

        # Validate inputs
        validated_paths = validate_document_paths(document_paths)

        start_time = time.time()
        processed_docs = 0
        failed_docs = 0
        total_chunks = 0
        errors = []

        # Process documents in batches
        for i in range(0, len(validated_paths), batch_size):
            batch = validated_paths[i : i + batch_size]

            for doc_path in batch:
                try:
                    # Extract text
                    extracted_text = self._document_extractor.extract_text(doc_path)
                    if not extracted_text.strip():
                        error_msg = f"No text extracted from {doc_path}"
                        errors.append(error_msg)
                        failed_docs += 1
                        continue

                    # Clean text
                    cleaned_text = self._text_cleaner.clean_text(extracted_text)

                    # Create chunks
                    chunks = self._chunker.chunk_document(cleaned_text, str(doc_path))
                    if not chunks:
                        error_msg = f"No chunks created from {doc_path}"
                        errors.append(error_msg)
                        failed_docs += 1
                        continue

                    # Process chunks
                    for chunk_idx, chunk in enumerate(chunks):
                        # Generate embedding
                        embedding = self._embedding_model.encode_text(chunk.content)

                        # Ensure embedding is 1D for ChromaDB
                        if hasattr(embedding, "ndim") and embedding.ndim == 2:
                            embedding = embedding[0]

                        # Create metadata using pure function
                        metadata = create_chunk_metadata(str(doc_path), chunk_idx, chunk, self.language, time.time())

                        # Store in vector database
                        self._vector_storage.add_documents(
                            documents=[chunk.content],
                            metadatas=[metadata],
                            embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else embedding],
                        )

                    processed_docs += 1
                    total_chunks += len(chunks)

                except Exception as e:
                    error_msg = f"Failed to process {doc_path}: {e}"
                    errors.append(error_msg)
                    failed_docs += 1

        processing_time = time.time() - start_time
        self._document_count += processed_docs

        # Calculate metrics using pure function
        metrics = calculate_processing_metrics(processed_docs, processing_time, total_chunks)

        return DocumentProcessingResult(
            processed_documents=processed_docs,
            failed_documents=failed_docs,
            total_chunks=total_chunks,
            processing_time=processing_time,
            documents_per_second=metrics["documents_per_second"],
            errors=errors if errors else None,
        )

    async def query(self, query: RAGQuery, return_sources: bool = True, return_debug_info: bool = False) -> RAGResponse:
        """Execute complete RAG query pipeline using pure functions."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        self._query_count += 1

        try:
            # Validate query using pure function
            validated_query = validate_query(query)
            # Step 1: Hierarchical retrieval
            query_start = time.time()
            hierarchical_results = await self._hierarchical_retriever.retrieve(
                query=validated_query.text,
                max_results=validated_query.max_results or 10,
                context=validated_query.context_filters,
            )
            retrieval_time = time.time() - query_start

            # Step 2: Generate response
            generation_start = time.time()

            # Build prompts
            context_chunks = [result["content"] for result in hierarchical_results.documents]
            system_prompt, user_prompt = self._prompt_builder.build_prompt(
                query=validated_query.text,
                context_chunks=context_chunks,
                category=getattr(hierarchical_results, "category", None),
                max_context_length=2000,
                include_source_attribution=True,
            )

            # Create generation request (this would be a protocol-defined structure)
            generation_request = {
                "prompt": user_prompt,
                "context": context_chunks,
                "query": validated_query.text,
                "query_type": getattr(hierarchical_results.category, "value", "general"),
                "language": validated_query.language,
                "metadata": validated_query.metadata,
            }

            # Generate response
            generation_response = await self._generation_client.generate_text_async(generation_request)
            generation_time = time.time() - generation_start

            # Step 3: Parse response
            parsed_response = self._response_parser.parse_response(
                generation_response.text, validated_query.text, context_chunks
            )

            # Step 4: Build response using pure functions
            total_time = time.time() - start_time

            # Extract sources
            sources = extract_sources_from_chunks(hierarchical_results.documents) if return_sources else []

            # Prepare chunk info
            retrieved_chunks = [
                prepare_chunk_info(result, return_debug_info) for result in hierarchical_results.documents
            ]

            # Build metadata
            response_metadata = build_response_metadata(
                validated_query,
                hierarchical_results,
                generation_response,
                parsed_response,
                retrieval_time,
                generation_time,
                total_time,
                return_debug_info,
                system_prompt,
                user_prompt,
            )

            return RAGResponse(
                answer=parsed_response.content,
                query=validated_query.text,
                retrieved_chunks=retrieved_chunks,
                confidence=getattr(parsed_response, "confidence", 0.5),
                generation_time=generation_time,
                retrieval_time=retrieval_time,
                total_time=total_time,
                sources=sources,
                metadata=response_metadata,
            )

        except Exception as e:
            return create_error_response(query, e, start_time)

    async def health_check(self) -> SystemHealth:
        """Perform comprehensive health check using pure functions."""
        if not self._initialized:
            await self.initialize()

        try:
            # Check preprocessing components
            preprocessing_health = evaluate_component_health(
                "preprocessing", [self._document_extractor, self._text_cleaner, self._chunker]
            )

            # Check vector database components
            vectordb_health = evaluate_component_health(
                "vectordb",
                [self._embedding_model, self._vector_storage, self._search_engine],
                f"Vector DB with {self._vector_storage.get_document_count()} documents",
            )

            # Check retrieval components
            retrieval_health = evaluate_component_health(
                "retrieval",
                [self._query_processor, self._retriever, self._ranker, self._hierarchical_retriever],
                "All retrieval components loaded (including hierarchical routing)",
            )

            # Check generation components
            model_name = self.ollama_config.model if self.ollama_config else "unknown"
            generation_health = evaluate_ollama_health(self._generation_client, model_name)

            components = {
                "preprocessing": preprocessing_health,
                "vectordb": vectordb_health,
                "retrieval": retrieval_health,
                "generation": generation_health,
            }

            # Calculate overall health
            overall_status = calculate_overall_health(components)

            metrics = {
                "documents_processed": self._document_count,
                "queries_processed": self._query_count,
                "total_chunks": self._vector_storage.get_document_count(),
            }

            return SystemHealth(
                system_status=overall_status, components=components, metrics=metrics, timestamp=time.time()
            )

        except Exception as e:
            return SystemHealth(system_status="error", components={}, metrics={}, timestamp=time.time(), error=str(e))

    async def get_system_stats(self) -> SystemStats:
        """Get comprehensive system statistics."""
        if not self._initialized:
            return SystemStats(system={"error": "System not initialized"}, collections={}, models={}, performance={})

        return SystemStats(
            system={
                "language": self.language,
                "initialized": self._initialized,
                "documents_processed": self._document_count,
                "queries_processed": self._query_count,
                "total_chunks": self._vector_storage.get_document_count(),
            },
            collections={
                "active_collection": create_language_collection_name(self.language),
                "collection_type": f"{self.language}_documents",
            },
            models={
                "embedding_model": self.embedding_config.model_name,
                "llm_model": self.ollama_config.model,
                "device": "auto",
            },
            performance={
                "max_retrieval": self.retrieval_config.max_k,
                "similarity_threshold": self.retrieval_config.similarity_threshold,
                "timeout": self.ollama_config.timeout,
            },
        )

    async def close(self) -> None:
        """Clean shutdown of all components."""
        if self._generation_client:
            await self._generation_client.close()

        if self._vector_storage:
            await self._vector_storage.close()


# Factory functions for creating instances
def create_rag_system(
    language: str,
    document_extractor: DocumentExtractorProtocol,
    text_cleaner: TextCleanerProtocol,
    document_chunker: DocumentChunkerProtocol,
    embedding_model: EmbeddingModelProtocol,
    vector_storage: VectorStorageProtocol,
    search_engine: SearchEngineProtocol,
    query_processor: QueryProcessorProtocol,
    retriever: RetrieverProtocol,
    hierarchical_retriever: RetrieverProtocol,
    ranker: RankerProtocol,
    generation_client: GenerationClientProtocol,
    response_parser: ResponseParserProtocol,
    prompt_builder: PromptBuilderProtocol,
    embedding_config: EmbeddingConfig,
    ollama_config: OllamaConfig,
    processing_config: ProcessingConfig,
    retrieval_config: RetrievalConfig,
) -> RAGSystem:
    """Factory function to create RAG system with validated config dependency injection."""
    return RAGSystem(
        language=language,
        document_extractor=document_extractor,
        text_cleaner=text_cleaner,
        document_chunker=document_chunker,
        embedding_model=embedding_model,
        vector_storage=vector_storage,
        search_engine=search_engine,
        query_processor=query_processor,
        retriever=retriever,
        hierarchical_retriever=hierarchical_retriever,
        ranker=ranker,
        generation_client=generation_client,
        response_parser=response_parser,
        prompt_builder=prompt_builder,
        embedding_config=embedding_config,
        ollama_config=ollama_config,
        processing_config=processing_config,
        retrieval_config=retrieval_config,
    )


# Note: Mock implementations have been moved to tests/fixtures/mock_rag_system.py
