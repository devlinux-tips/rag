"""
Multilingual RAG System orchestrating document processing, retrieval, and generation.
Coordinates preprocessing, vector storage, semantic search, and LLM generation
for multilingual document question-answering workflows.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..utils.config_models import EmbeddingConfig, OllamaConfig, ProcessingConfig, RetrievalConfig
from ..utils.config_validator import ConfigurationError
from ..utils.logging_factory import (
    get_system_logger,
    log_component_end,
    log_component_start,
    log_config_usage,
    log_data_transformation,
    log_decision_point,
    log_error_context,
    log_performance_metric,
)


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
    require_rag_documents: bool = True  # Fail if no documents found (no LLM fallback)


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
    nn_sources: list[dict[str, Any]] | None = None  # Narodne Novine metadata for citations
    tokens_used: int = 0  # LLM tokens consumed (total, for backward compatibility)
    input_tokens: int = 0  # LLM input tokens (prompt + context)
    output_tokens: int = 0  # LLM output tokens (generated response)
    model_used: str = "unknown"  # Actual model name used for generation

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

    def extract_text(self, file_path: Path) -> Any: ...


class TextCleanerProtocol(Protocol):
    """Protocol for multilingual text cleaning."""

    def clean_text(self, text: str, preserve_structure: bool = ...) -> Any: ...

    def setup_language_environment(self) -> None: ...


class DocumentChunkerProtocol(Protocol):
    """Protocol for document chunking."""

    def chunk_document(self, content: str, source_file: str) -> list[Any]: ...


class EmbeddingModelProtocol(Protocol):
    """Protocol for text embedding generation."""

    def encode_text(self, text: str) -> Any: ...

    def generate_embeddings(self, texts: list[str]) -> Any: ...

    def initialize(self) -> None: ...


class VectorStorageProtocol(Protocol):
    """Protocol for vector storage operations."""

    collection: Any  # Vector collection object (varies by provider)

    def add_documents(self, documents: list[str], metadatas: list[dict], embeddings: list) -> None: ...

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict], embeddings: list) -> None: ...

    def get_document_count(self) -> int: ...

    async def initialize(self, collection_name: str, reset_if_exists: bool = False) -> None: ...

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

    async def health_check(self) -> bool: ...

    async def get_available_models(self) -> list[str]: ...

    async def close(self) -> None: ...


class ResponseParserProtocol(Protocol):
    """Protocol for response parsing."""

    def parse_response(self, text: str, query: str, context: list[str]) -> Any: ...


class PromptBuilderProtocol(Protocol):
    """Protocol for prompt building."""

    def build_prompt(
        self,
        query: str,
        context_chunks: list[str],
        category: Any = ...,
        max_context_length: int = ...,
        include_source_attribution: bool = ...,
    ) -> tuple[str, str]: ...


# Pure functions for business logic
def validate_language_code(language: str) -> str:
    """Validate and normalize language code."""
    logger = get_system_logger()
    logger.trace("validation", "validate_language_code", f"Input: '{language}'")

    if not language or not isinstance(language, str):
        error_msg = "Language code must be a non-empty string"
        logger.error("validation", "validate_language_code", f"VALIDATION_FAILED: {error_msg}")
        raise ValueError(error_msg)

    original_language = language
    language = language.lower().strip()
    valid_languages = {"hr", "en", "multilingual"}

    logger.debug("validation", "validate_language_code", f"Normalized '{original_language}' → '{language}'")

    if language not in valid_languages:
        error_msg = f"Unsupported language: {language}. Supported: {valid_languages}"
        logger.error("validation", "validate_language_code", f"VALIDATION_FAILED: {error_msg}")
        raise ValueError(error_msg)

    logger.debug("validation", "validate_language_code", f"VALIDATION_SUCCESS: '{language}' is valid")
    return language


def create_language_collection_name(language: str) -> str:
    """Create language-specific collection name."""
    logger = get_system_logger()
    logger.trace("collection", "create_collection_name", f"Input language: '{language}'")

    language_collection_map = {
        "hr": "croatian_documents",
        "en": "english_documents",
        "multilingual": "multilingual_documents",
    }

    if language not in language_collection_map:
        {"input_language": language, "available_mappings": list(language_collection_map.keys())}
        error_msg = f"Unsupported language '{language}'. Supported: {list(language_collection_map.keys())}"
        logger.error("collection", "create_collection_name", f"MAPPING_FAILED: {error_msg}")
        raise ConfigurationError(error_msg)

    collection_name = language_collection_map[language]
    log_decision_point(
        "collection", "create_collection_name", f"language='{language}'", f"collection='{collection_name}'"
    )
    return collection_name


def validate_document_paths(document_paths: list[str]) -> list[Path]:
    """Validate and convert document paths to Path objects."""
    logger = get_system_logger()
    log_component_start("validation", "validate_document_paths", count=len(document_paths) if document_paths else 0)

    if not document_paths:
        error_msg = "Document paths list cannot be empty"
        logger.error("validation", "validate_document_paths", f"EMPTY_LIST: {error_msg}")
        raise ValueError(error_msg)

    validated_paths = []
    for i, path_str in enumerate(document_paths):
        logger.trace(
            "validation", "validate_document_paths", f"Validating path {i + 1}/{len(document_paths)}: '{path_str}'"
        )

        if not path_str or not isinstance(path_str, str):
            error_msg = f"Document path at index {i} must be a non-empty string"
            logger.error("validation", "validate_document_paths", f"INVALID_PATH: {error_msg}")
            raise ValueError(error_msg)

        path = Path(path_str)
        if not path.exists():
            error_msg = f"Document path does not exist: {path}"
            logger.error("validation", "validate_document_paths", f"PATH_NOT_FOUND: {error_msg}")
            raise ValueError(error_msg)

        if not path.is_file():
            error_msg = f"Document path is not a file: {path}"
            logger.error("validation", "validate_document_paths", f"NOT_A_FILE: {error_msg}")
            raise ValueError(error_msg)

        validated_paths.append(path)
        logger.debug("validation", "validate_document_paths", f"Path {i + 1} valid: {path}")

    log_component_end("validation", "validate_document_paths", f"validated {len(validated_paths)} paths")
    return validated_paths


def validate_query(query: RAGQuery) -> RAGQuery:
    """Validate RAG query parameters."""
    logger = get_system_logger()
    log_component_start(
        "validation",
        "validate_query",
        query_id=query.query_id,
        language=query.language,
        text_length=len(query.text) if query.text else 0,
    )

    if not query.text or not query.text.strip():
        error_msg = "Query text cannot be empty"
        logger.error("validation", "validate_query", f"EMPTY_QUERY: {error_msg}")
        raise ValueError(error_msg)

    if not query.language:
        error_msg = "Query language must be specified"
        logger.error("validation", "validate_query", f"MISSING_LANGUAGE: {error_msg}")
        raise ValueError(error_msg)

    original_language = query.language
    query.language = validate_language_code(query.language)
    if original_language != query.language:
        log_data_transformation(
            "validation", "validate_query", f"language '{original_language}'", f"normalized '{query.language}'"
        )

    original_max_results = query.max_results
    if query.max_results is None:
        query.max_results = 5
        logger.debug("validation", "validate_query", "Applied default max_results=5")
    elif query.max_results <= 0:
        error_msg = "max_results must be positive"
        logger.error("validation", "validate_query", f"INVALID_MAX_RESULTS: {error_msg}, got: {query.max_results}")
        raise ValueError(error_msg)

    if original_max_results != query.max_results:
        log_data_transformation(
            "validation", "validate_query", f"max_results {original_max_results}", f"set to {query.max_results}"
        )

    log_component_end(
        "validation",
        "validate_query",
        "query validated successfully",
        final_language=query.language,
        final_max_results=query.max_results,
    )
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
    doc_path: str,
    chunk_idx: int,
    chunk: Any,
    language: str,
    processing_timestamp: float,
    nn_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create metadata for a document chunk.

    Args:
        doc_path: Path to source document
        chunk_idx: Index of chunk in document
        chunk: Chunk object with content and attributes
        language: Language code
        processing_timestamp: Processing timestamp
        nn_metadata: Optional Narodne Novine metadata (feature-specific)

    Returns:
        Metadata dict for chunk storage
    """
    metadata = {
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

    # Add feature-specific metadata if present (narodne-novine documents)
    if nn_metadata:
        metadata["nn_metadata"] = nn_metadata

    return metadata


def extract_sources_from_chunks(retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    """Extract unique sources from retrieved document chunks."""
    from src.utils.logging_config import get_system_logger

    logger = get_system_logger()

    sources = set()
    for i, chunk in enumerate(retrieved_chunks):
        # AI DEBUGGING: Log actual chunk structure
        logger.error(
            "chunk_debug",
            f"extract_sources_chunk_{i}",
            f"CHUNK_STRUCTURE | chunk_keys={list(chunk.keys())} | "
            f"has_metadata={('metadata' in chunk)} | "
            f"chunk_sample={str(chunk)[:500]}...",
        )

        # FAIL FAST: Chunk must have proper metadata structure
        if "metadata" not in chunk:
            raise ValueError(f"Chunk missing required 'metadata' field: {chunk}")
        metadata = chunk["metadata"]

        # AI DEBUGGING: Log metadata structure
        logger.error(
            "metadata_debug",
            f"extract_sources_metadata_{i}",
            f"METADATA_STRUCTURE | metadata_keys={list(metadata.keys())} | "
            f"has_source={('source' in metadata)} | "
            f"metadata_content={metadata}",
        )

        # ENHANCED: Try multiple source field names
        source = None
        source_candidates = ["source", "document_id", "file_path", "filename", "title", "document_title"]

        for field_name in source_candidates:
            if field_name in metadata:
                source = metadata[field_name]
                logger.info(
                    "source_mapping", f"found_source_field_{i}", f"FOUND_SOURCE | field={field_name} | value={source}"
                )
                break

        if source is None:
            # Try to construct source from available metadata
            if "detected_category" in metadata:
                source = f"Document (Category: {metadata['detected_category']})"
                logger.info("source_fallback", f"constructed_source_{i}", f"CONSTRUCTED_SOURCE | value={source}")
            else:
                logger.error(
                    "source_missing",
                    f"no_source_found_{i}",
                    f"NO_SOURCE_AVAILABLE | metadata_keys={list(metadata.keys())}",
                )
                source = "Unknown Document"

        if source and source != "Unknown":
            sources.add(source)

    return list(sources)


def prepare_chunk_info(chunk_result: dict[str, Any], return_debug_info: bool = False) -> dict[str, Any]:
    """Prepare chunk information for response."""
    # ENHANCED: Extract source using same logic as extract_sources_from_chunks
    metadata = chunk_result["metadata"]
    source = None
    source_candidates = ["source", "document_id", "file_path", "filename", "title", "document_title"]

    for field_name in source_candidates:
        if field_name in metadata:
            source = metadata[field_name]
            break

    if source is None:
        if "detected_category" in metadata:
            source = f"Document (Category: {metadata['detected_category']})"
        else:
            source = "Unknown"

    chunk_info = {
        "content": chunk_result["content"],
        "similarity_score": chunk_result["similarity_score"],
        "final_score": chunk_result["final_score"],
        "source": source,
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


def get_language_message(language: str, message_key: str) -> str:
    """Get language-specific message from configuration."""
    try:
        from ..utils.config_loader import get_language_config

        language_config = get_language_config(language)
        message = language_config["shared"][message_key]
        if not message:
            raise ConfigurationError(f"Empty message for key '{message_key}' in language '{language}'")
        return message
    except KeyError as e:
        raise ConfigurationError(f"Missing message key '{message_key}' in language '{language}' configuration") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to load message '{message_key}' for language '{language}': {e}") from e


def create_empty_response(query: RAGQuery, retrieval_time: float, start_time: float) -> RAGResponse:
    """Create empty response when no documents found - configuration-driven multilingual."""
    no_results_msg = get_language_message(query.language, "no_results_message")

    total_time = time.time() - start_time
    return RAGResponse(
        answer=no_results_msg,
        query=query.text,
        retrieved_chunks=[],
        confidence=0.0,
        generation_time=0.0,
        retrieval_time=retrieval_time,
        total_time=total_time,
        sources=[],
        metadata={
            "query": query.text,
            "language": query.language,
            "results_found": 0,
            "category": "unknown",
            "strategy": "none",
        },
    )


def create_error_response(query: RAGQuery, error: Exception, start_time: float) -> RAGResponse:
    """Create error response in the appropriate language - configuration-driven.

    CRITICAL: Only user-friendly message in answer field. Technical details in metadata for AI debugging.
    """
    # Get clean user-friendly error message (e.g., "Žao mi je, dogodila se greška pri obradi pitanja")
    error_msg = get_language_message(query.language, "error_message")

    # AI-FRIENDLY ERROR METADATA: All technical details for debugging
    error_metadata = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_details": repr(error),
        "query_id": query.query_id,
        "user_id": query.user_id,
        "language": query.language,
        "timestamp": time.time(),
    }

    return RAGResponse(
        answer=error_msg,  # CLEAN message only - no technical details
        query=query.text,
        retrieved_chunks=[],
        confidence=0.0,
        generation_time=0.0,
        retrieval_time=0.0,
        total_time=time.time() - start_time,
        sources=[],
        metadata=error_metadata,  # Technical details here for AI debugging
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


async def evaluate_ollama_health(client: GenerationClientProtocol | None, model_name: str) -> ComponentHealth:
    """Evaluate Ollama service health."""
    if not client:
        return ComponentHealth(status="unhealthy", details="Ollama client not initialized")

    ollama_healthy = await client.health_check()
    if not ollama_healthy:
        return ComponentHealth(status="unhealthy", details="Ollama service not available")

    available_models = await client.get_available_models()
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
        batch_config: dict[str, Any],  # Raw batch config dict for BatchProcessingConfig.from_config()
        # Feature context (for feature-specific behavior)
        scope: str = "user",
        feature_name: str | None = None,
    ):
        """Initialize with all dependencies injected.

        Args:
            language: Language code (hr, en, etc.)
            scope: Data scope (user, tenant, feature, global)
            feature_name: Feature name when scope='feature' (e.g., 'narodne-novine')
            ... (other dependencies)
        """
        system_logger = get_system_logger()
        log_component_start("rag_system", "initialize", target_language=language, scope=scope, feature=feature_name)

        self.language = validate_language_code(language)
        self.scope = scope
        self.feature_name = feature_name
        system_logger.info(
            "rag_system",
            "initialize",
            f"Language validated: {self.language} | Scope: {scope} | Feature: {feature_name or 'N/A'}",
        )

        self.embedding_config = embedding_config
        self.ollama_config = ollama_config
        self.processing_config = processing_config
        self.retrieval_config = retrieval_config

        # Initialize batch processing config
        from ..preprocessing.batch_processor import BatchProcessingConfig

        self.batch_config = BatchProcessingConfig.from_config(batch_config)

        log_config_usage(
            "rag_system",
            "initialize",
            {
                "embedding_model": embedding_config.model_name,
                "ollama_model": ollama_config.model,
                "sentence_chunk_overlap": processing_config.sentence_chunk_overlap,
                "max_k": retrieval_config.max_k,
            },
        )

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
        self._collection_name = None  # Will be set during initialization

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        system_logger = get_system_logger()

        if self._initialized:
            system_logger.debug("rag_system", "initialize", "Already initialized, skipping")
            return

        log_component_start("rag_system", "post_init", language=self.language)

        await self._validate_configuration()
        system_logger.info("rag_system", "post_init", "Configuration validation completed")

        self._text_cleaner.setup_language_environment()
        system_logger.debug("rag_system", "post_init", f"Language environment setup for: {self.language}")

        self._embedding_model.initialize()
        system_logger.info(
            "rag_system", "post_init", f"Embedding model initialized: {self.embedding_config.model_name}"
        )

        # Initialize vector storage with the pending collection name (if not already initialized)
        if self._vector_storage.collection is None:
            if hasattr(self._vector_storage, "_pending_collection_name"):
                self._collection_name = self._vector_storage._pending_collection_name
                await self._vector_storage.initialize(
                    collection_name=self._vector_storage._pending_collection_name, reset_if_exists=False
                )
                system_logger.debug(
                    "rag_system",
                    "post_init",
                    f"Vector storage collection initialized: {self._vector_storage._pending_collection_name}",
                )
            else:
                # Error case - no collection name available
                error_msg = "Vector storage initialization failed: no collection name specified"
                system_logger.error("rag_system", "post_init", error_msg)
                raise RuntimeError(error_msg)
        else:
            # Collection already initialized, get the collection name for tracking
            if hasattr(self._vector_storage.collection, "class_name"):
                self._collection_name = self._vector_storage.collection.class_name
            system_logger.debug(
                "rag_system",
                "post_init",
                f"Vector storage already initialized with collection: {self._collection_name}",
            )

        # Initialize search engine with now-initialized VectorStorage
        # This fixes the issue where search engine was created with uninitialized VectorStorage
        system_logger.info(
            "rag_system",
            "post_init",
            f"SEARCH ENGINE DEBUG | has_hierarchical_retriever={self._hierarchical_retriever is not None}",
        )
        if self._hierarchical_retriever:
            has_search_engine_attr = hasattr(self._hierarchical_retriever, "_search_engine")
            search_engine_value = getattr(self._hierarchical_retriever, "_search_engine", "NO_ATTR")
            search_engine_type = type(search_engine_value).__name__
            # Check the actual search engine inside the adapter
            actual_search_engine = None
            if hasattr(search_engine_value, "_search_engine"):
                actual_search_engine = search_engine_value._search_engine
            actual_search_engine_type = type(actual_search_engine).__name__
            system_logger.info(
                "rag_system",
                "post_init",
                f"SEARCH ENGINE DEBUG | has_attr={has_search_engine_attr} | adapter_type={search_engine_type} | actual_search_engine_type={actual_search_engine_type} | actual_none={actual_search_engine is None}",
            )

            # Check if the SearchEngineAdapter contains None as the actual search engine
            if (
                hasattr(self._hierarchical_retriever, "_search_engine")
                and hasattr(self._hierarchical_retriever._search_engine, "_search_engine")
                and self._hierarchical_retriever._search_engine._search_engine is None
            ):
                from ..vectordb.search_providers import create_vector_search_provider

                search_engine = create_vector_search_provider(self._vector_storage, self._embedding_model)
                system_logger.info(
                    "rag_system", "post_init", f"LATE SEARCH ENGINE INIT | search_engine_none={search_engine is None}"
                )
                # Update the SearchEngineAdapter with the properly initialized search engine
                self._hierarchical_retriever._search_engine._search_engine = search_engine
                system_logger.info(
                    "rag_system", "post_init", "Search engine initialized after VectorStorage initialization"
                )
            else:
                system_logger.info(
                    "rag_system", "post_init", "SEARCH ENGINE DEBUG | Lazy initialization condition not met"
                )

        self._initialized = True
        log_component_end("rag_system", "post_init", "All components initialized successfully")

    def _get_feature_scope(self) -> str | None:
        """Get current feature name from scope context.

        Returns feature name if scope='feature', otherwise None.
        Enables feature-specific behavior without configuration files.

        Returns:
            Feature name (e.g., 'narodne-novine') or None
        """
        return self.feature_name if self.scope == "feature" else None

    def _is_nn_feature(self) -> bool:
        """Check if processing narodne-novine feature.

        Convention-based detection:
        - No configuration needed
        - Based on --scope feature --feature narodne-novine
        - Enables NN-specific metadata extraction automatically

        Returns:
            True if narodne-novine feature scope is active
        """
        return self._get_feature_scope() == "narodne-novine"

    def _should_extract_nn_metadata(self, file_path: Path, html_content: str) -> bool:
        """Determine if NN metadata extraction should run.

        Triple-check ensures safety and efficiency:
        1. Must be in narodne-novine feature scope
        2. Must be HTML file
        3. Must contain ELI metadata tags

        Args:
            file_path: Path to document file
            html_content: Raw HTML content

        Returns:
            True if all conditions met for NN metadata extraction

        This ensures:
        - No metadata extraction for non-NN features (zero overhead)
        - No wasted processing on non-HTML files
        - No errors on non-NN HTML files
        """
        if not self._is_nn_feature():
            return False

        if file_path.suffix.lower() != ".html":
            return False

        # Quick check for ELI metadata presence
        from ..extraction.nn_metadata_extractor import is_nn_document

        return is_nn_document(html_content)

    async def _validate_configuration(self) -> None:
        """Validate configuration using ConfigValidator - minimal integration."""
        from ..utils.config_loader import get_language_config, load_config
        from ..utils.config_validator import ConfigValidator

        # Load current configs
        main_config = load_config("config")
        language_config = get_language_config(self.language)

        ConfigValidator.validate_startup_config(
            main_config, {self.language: language_config}, current_language=self.language
        )
        get_system_logger().info(
            "config_validator", "validate_startup_config", "All configuration keys validated successfully"
        )

    async def add_documents(self, document_paths: list[str], batch_size: int = 10) -> DocumentProcessingResult:
        """Process documents using efficient batch processing for embeddings and vector storage."""
        system_logger = get_system_logger()
        log_component_start(
            "rag_system", "add_documents", doc_count=len(document_paths), batch_size=batch_size, language=self.language
        )

        if not self._initialized:
            system_logger.info("rag_system", "add_documents", "System not initialized, initializing now")
            await self.initialize()

        validated_paths = validate_document_paths(document_paths)
        system_logger.info(
            "pipeline",
            "BATCH_PROCESSING",
            f"STARTED: {len(validated_paths)} documents | embedding_batch_size={self.batch_config.embedding_batch_size} | vector_batch_size={self.batch_config.vector_insert_batch_size}",
        )

        start_time = time.time()
        processed_docs = 0
        failed_docs = 0
        total_chunks = 0
        errors = []

        # PHASE 1: Collect all chunks from all documents (no embeddings yet)
        all_chunks_data = []  # List of (chunk_content, chunk_id, metadata) tuples
        system_logger.info("batch_processing", "phase_1", "Extracting and chunking all documents...")

        doc_batch_size = self.batch_config.document_batch_size

        for i in range(0, len(validated_paths), doc_batch_size):
            batch = validated_paths[i : i + doc_batch_size]
            system_logger.debug(
                "batch_processing",
                "document_batch",
                f"Processing document batch {i // doc_batch_size + 1}/{(len(validated_paths) - 1) // doc_batch_size + 1}: {len(batch)} documents",
            )

            for doc_path in batch:
                time.time()
                system_logger.debug("document_processing", "extract_and_chunk", f"Processing: {doc_path}")

                try:
                    # Extract text
                    system_logger.trace("document_processing", "extract_text", f"Starting extraction: {doc_path}")
                    extraction_result = self._document_extractor.extract_text(doc_path)
                    extracted_text = (
                        extraction_result.text if hasattr(extraction_result, "text") else str(extraction_result)
                    )
                    if not extracted_text.strip():
                        error_msg = f"No text extracted from {doc_path}"
                        system_logger.warning("document_processing", "extract_text", error_msg)
                        errors.append(error_msg)
                        failed_docs += 1
                        continue

                    # Extract Narodne Novine metadata (feature-specific, fail-safe)
                    nn_metadata = None
                    is_html = doc_path.suffix.lower() == ".html"
                    is_nn_feat = self._is_nn_feature()
                    system_logger.debug(
                        "nn_metadata",
                        "check",
                        f"doc={doc_path.name} | is_html={is_html} | is_nn_feature={is_nn_feat} | scope={self.scope} | feature_name={self.feature_name}",
                    )
                    if is_html and is_nn_feat:
                        try:
                            with open(doc_path, encoding="utf-8") as f:
                                html_content = f.read()
                            if self._should_extract_nn_metadata(doc_path, html_content):
                                from ..extraction.nn_metadata_extractor import extract_nn_metadata

                                nn_metadata = extract_nn_metadata(html_content, doc_path)
                                if nn_metadata:
                                    title_preview = nn_metadata.get("title", "")[:50]
                                    system_logger.info(
                                        "nn_metadata",
                                        "extracted",
                                        f"title={title_preview} | eli={nn_metadata.get('eli_url', 'N/A')} | issue={nn_metadata.get('issue', 'N/A')}",
                                    )
                        except Exception as e:
                            # NN metadata extraction is optional - don't fail document processing
                            system_logger.warning(
                                "nn_metadata",
                                "extraction_failed",
                                f"Failed to extract NN metadata from {doc_path}: {e}",
                            )

                    # Clean text
                    system_logger.trace("document_processing", "clean_text", f"Cleaning: {len(extracted_text)} chars")
                    cleaning_result = self._text_cleaner.clean_text(extracted_text)
                    cleaned_text = cleaning_result.text if hasattr(cleaning_result, "text") else str(cleaning_result)

                    # Chunk document
                    system_logger.trace("document_processing", "chunk_document", f"Chunking: {len(cleaned_text)} chars")
                    chunks = self._chunker.chunk_document(cleaned_text, str(doc_path))
                    if not chunks:
                        error_msg = f"No chunks created from {doc_path}"
                        system_logger.warning("document_processing", "chunk_document", error_msg)
                        errors.append(error_msg)
                        failed_docs += 1
                        continue

                    # Collect chunk data (but don't generate embeddings yet)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_id = f"{doc_path.stem}_{chunk_idx}_{hash(chunk.content) % 1000000}"
                        metadata = create_chunk_metadata(
                            str(doc_path), chunk_idx, chunk, self.language, time.time(), nn_metadata
                        )
                        all_chunks_data.append((chunk.content, chunk_id, metadata))

                    processed_docs += 1
                    total_chunks += len(chunks)
                    system_logger.debug(
                        "document_processing",
                        "extract_and_chunk",
                        f"Collected {len(chunks)} chunks from {doc_path.name}",
                    )

                except Exception as e:
                    error_msg = f"Failed to process {doc_path}: {e}"
                    system_logger.error("document_processing", "extract_and_chunk", f"ERROR: {error_msg}")
                    errors.append(error_msg)
                    failed_docs += 1

        system_logger.info(
            "batch_processing",
            "phase_1_complete",
            f"Collected {len(all_chunks_data)} chunks from {processed_docs} documents",
        )

        # PHASE 2: Generate embeddings in batches
        if not all_chunks_data:
            system_logger.warning("batch_processing", "phase_2", "No chunks to process - skipping embedding generation")
        else:
            system_logger.info(
                "batch_processing",
                "phase_2",
                f"Generating embeddings for {len(all_chunks_data)} chunks in batches of {self.batch_config.embedding_batch_size}...",
            )

            chunk_contents = [chunk_data[0] for chunk_data in all_chunks_data]
            all_embeddings = []

            embed_batch_size = self.batch_config.embedding_batch_size
            total_batches = (len(chunk_contents) - 1) // embed_batch_size + 1

            for i in range(0, len(chunk_contents), embed_batch_size):
                batch_contents = chunk_contents[i : i + embed_batch_size]
                batch_start = time.time()
                current_batch = i // embed_batch_size + 1

                # AI-FRIENDLY PROGRESS LOG: Track embedding generation progress
                system_logger.info(
                    "batch_processing",
                    "embedding_batch",
                    f"EMBEDDING_PROGRESS | batch={current_batch}/{total_batches} | chunks={len(batch_contents)} | processed={i}/{len(chunk_contents)} | progress={(i / len(chunk_contents) * 100):.1f}%",
                )

                # Generate embeddings for entire batch at once
                embedding_result = self._embedding_model.generate_embeddings(batch_contents)
                batch_embeddings = embedding_result.embeddings

                # Handle 2D embeddings (batch_size x embedding_dim)
                if hasattr(batch_embeddings, "ndim") and batch_embeddings.ndim == 2:
                    for embedding in batch_embeddings:
                        all_embeddings.append(embedding)
                else:
                    # Single embedding or list
                    all_embeddings.extend(batch_embeddings)

                batch_time = time.time() - batch_start
                system_logger.debug(
                    "batch_processing",
                    "embedding_batch",
                    f"Generated {len(batch_contents)} embeddings in {batch_time:.2f}s",
                )

            system_logger.info(
                "batch_processing",
                "phase_2_complete",
                f"Generated {len(all_embeddings)} embeddings for {len(all_chunks_data)} chunks",
            )

            # PHASE 3: Store in vector database in batches
            system_logger.info(
                "batch_processing",
                "phase_3",
                f"Storing chunks in vector DB in batches of {self.batch_config.vector_insert_batch_size}...",
            )

            vector_batch_size = self.batch_config.vector_insert_batch_size
            stored_chunks = 0

            for i in range(0, len(all_chunks_data), vector_batch_size):
                batch_chunk_data = all_chunks_data[i : i + vector_batch_size]
                batch_embeddings = all_embeddings[i : i + vector_batch_size]

                system_logger.debug(
                    "batch_processing",
                    "vector_batch",
                    f"Storing batch {i // vector_batch_size + 1}/{(len(all_chunks_data) - 1) // vector_batch_size + 1}: {len(batch_chunk_data)} chunks",
                )

                # Prepare batch data
                batch_ids = [chunk_data[1] for chunk_data in batch_chunk_data]
                batch_documents = [chunk_data[0] for chunk_data in batch_chunk_data]
                batch_metadatas = [chunk_data[2] for chunk_data in batch_chunk_data]

                # Convert embeddings to proper format for Weaviate
                batch_embedding_data = []
                for embedding in batch_embeddings:
                    if hasattr(embedding, "tolist"):
                        batch_embedding_data.append(embedding.tolist())
                    elif isinstance(embedding, (list, tuple)):
                        batch_embedding_data.append(list(embedding))
                    else:
                        batch_embedding_data.append(embedding)

                # Store entire batch at once
                self._vector_storage.add(
                    ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas, embeddings=batch_embedding_data
                )

                stored_chunks += len(batch_chunk_data)
                system_logger.debug("batch_processing", "vector_batch", f"Stored {len(batch_chunk_data)} chunks")

            system_logger.info(
                "batch_processing", "phase_3_complete", f"Stored {stored_chunks} chunks in vector database"
            )

        # Calculate final metrics
        processing_time = time.time() - start_time
        self._document_count += processed_docs

        metrics = calculate_processing_metrics(processed_docs, processing_time, total_chunks)

        log_component_end(
            "rag_system",
            "add_documents",
            f"BATCH_PROCESSED: {processed_docs}/{len(document_paths)} documents, {total_chunks} chunks in {processing_time:.2f}s",
            duration=processing_time,
            processed_docs=processed_docs,
            failed_docs=failed_docs,
            total_chunks=total_chunks,
        )

        system_logger.info(
            "pipeline",
            "BATCH_PROCESSING_COMPLETE",
            f"✅ {processed_docs} docs → {total_chunks} chunks → {len(all_embeddings) if all_chunks_data else 0} embeddings → stored in {processing_time:.2f}s",
        )

        return DocumentProcessingResult(
            processed_documents=processed_docs,
            failed_documents=failed_docs,
            total_chunks=total_chunks,
            processing_time=processing_time,
            documents_per_second=metrics["documents_per_second"],
            errors=errors if errors else None,
        )

    async def query(self, query: RAGQuery, return_sources: bool = True, return_debug_info: bool = False) -> RAGResponse:
        system_logger = get_system_logger()
        log_component_start(
            "rag_system",
            "query",
            query_id=query.query_id,
            language=query.language,
            query_length=len(query.text) if query.text else 0,
            return_sources=return_sources,
            return_debug_info=return_debug_info,
        )

        if not self._initialized:
            system_logger.info("rag_system", "query", "System not initialized, initializing now")
            await self.initialize()

        start_time = time.time()
        self._query_count += 1

        try:
            validated_query = validate_query(query)
            system_logger.info(
                "pipeline",
                "QUERY_PROCESSING",
                f"STARTED: '{validated_query.text[:100]}...' lang={validated_query.language}",
            )

            query_start = time.time()
            system_logger.info("retrieval", "hierarchical_retrieval", "Starting hierarchical retrieval for query")
            system_logger.trace(
                "query_processing",
                "hierarchical_retrieval",
                f"Query: '{validated_query.text[:50]}...', max_results={validated_query.max_results}",
            )

            hierarchical_results = await self._hierarchical_retriever.retrieve(
                query=validated_query.text,
                max_results=validated_query.max_results or 10,
                context=validated_query.context_filters,
            )

            retrieval_time = time.time() - query_start
            log_performance_metric(
                "retrieval",
                "hierarchical_retrieval",
                "duration",
                retrieval_time,
                results_found=len(hierarchical_results.documents) if hierarchical_results.documents else 0,
            )

            if not hierarchical_results or not hierarchical_results.documents:
                system_logger.warning(
                    "retrieval", "hierarchical_retrieval", "No documents found - returning no-context response"
                )
                log_decision_point(
                    "query_processing", "no_results_found", "no documents retrieved", "returning empty response"
                )
                # AI-friendly TRACE logging for empty results
                system_logger.trace(
                    "rag_system",
                    "query",
                    f"RETRIEVAL_EMPTY | query='{validated_query.text[:100]}' | "
                    f"collection={self._collection_name} | chunks_found=0",
                )
                return create_empty_response(validated_query, time.time() - query_start, start_time)

            try:
                generation_start = time.time()
                system_logger.info(
                    "generation",
                    "llm_processing",
                    f"Processing {len(hierarchical_results.documents)} retrieved documents",
                )

                context_chunks = [result["content"] for result in hierarchical_results.documents]
                total_context_chars = sum(len(c) for c in context_chunks)
                log_data_transformation(
                    "query_processing",
                    "prepare_context",
                    f"{len(hierarchical_results.documents)} documents",
                    f"{len(context_chunks)} chunks, {total_context_chars} chars",
                )

                # AI-friendly TRACE logging for chunk retrieval
                system_logger.trace(
                    "rag_system",
                    "query",
                    f"CHUNKS_RETRIEVED | query='{validated_query.text[:100]}' | "
                    f"collection={self._collection_name} | chunks_count={len(context_chunks)} | "
                    f"total_chars={total_context_chars} | "
                    f"first_chunk_preview='{context_chunks[0][:200] if context_chunks else 'None'}'",
                )

                from ..retrieval.categorization import CategoryType

                category_str = getattr(hierarchical_results, "category", None)
                if category_str is None:
                    error_msg = f"No category found in hierarchical_results: {hierarchical_results}"
                    log_error_context(
                        "query_processing",
                        "get_category",
                        ValueError(error_msg),
                        {"hierarchical_results_type": type(hierarchical_results).__name__},
                    )
                    raise ValueError(error_msg)

                category = CategoryType(category_str)
                log_decision_point(
                    "query_processing", "categorize_query", "query analysis", f"category: {category_str}"
                )

                from ..utils.config_loader import get_language_specific_config

                prompts_config = get_language_specific_config("prompts", self.language)
                log_config_usage("query_processing", "load_prompts", {"language": self.language})

                # Format context with citations for NN documents
                formatted_chunks = []
                nn_sources = []  # Track sources for citation list

                for idx, (chunk, doc) in enumerate(
                    zip(context_chunks, hierarchical_results.documents, strict=False), 1
                ):
                    # Check if document has NN metadata
                    metadata = doc.get("metadata", {})

                    nn_metadata = metadata.get("nn_metadata") if isinstance(metadata, dict) else None

                    if nn_metadata and isinstance(nn_metadata, dict):
                        # NN document: add numbered citation
                        title = nn_metadata.get("title", "Nepoznat dokument")
                        issue = nn_metadata.get("issue", "")
                        formatted_chunks.append(f"[{idx}] {chunk}")
                        nn_sources.append(nn_metadata)
                        system_logger.debug("rag_citation", "nn_source", f"[{idx}] {title} ({issue})")
                    else:
                        # Non-NN document: no citation
                        formatted_chunks.append(chunk)

                context_text = "\n\n".join(formatted_chunks) if formatted_chunks else "Nema dostupnih informacija."

                # Deduplicate nn_sources by eli_url (multiple chunks from same document)
                if nn_sources:
                    seen_eli = set()
                    unique_nn_sources = []
                    for source in nn_sources:
                        eli_url = source.get("eli_url")
                        if eli_url and eli_url not in seen_eli:
                            seen_eli.add(eli_url)
                            unique_nn_sources.append(source)
                    nn_sources = unique_nn_sources
                    system_logger.debug(
                        "rag_citation",
                        "deduplicate",
                        f"Deduplicated {len(seen_eli)} unique sources from {len(nn_sources)} chunks",
                    )

                # Add citation instructions to system prompt if we have NN sources
                if nn_sources:
                    citation_instruction = (
                        "\n\nVAŽNO: U kontekstu su dokumenti označeni brojevima [1], [2], itd. "
                        "Kada koristiš informacije iz dokumenta, DODAJ oznaku broja izvora odmah nakon tvrdnje. "
                        "Primjer: 'Najviša cijena goriva je 1,50 EUR/l [1].'"
                    )
                else:
                    citation_instruction = ""

                # AI-friendly TRACE logging for context being sent to LLM
                system_logger.trace(
                    "rag_system",
                    "query",
                    f"CONTEXT_TO_LLM | chunks_used={len(context_chunks)} | "
                    f"context_length={len(context_text)} | "
                    f"sending_to_model={self.ollama_config.model} | "
                    f"has_context={len(context_chunks) > 0} | "
                    f"nn_sources={len(nn_sources)}",
                )

                system_prompt = prompts_config["question_answering_system"] + citation_instruction
                user_prompt = prompts_config["question_answering_user"].format(
                    query=validated_query.text, context=context_text
                )

                log_data_transformation(
                    "query_processing",
                    "build_prompts",
                    f"context: {len(context_text)} chars + query",
                    f"system: {len(system_prompt)}, user: {len(user_prompt)} chars",
                )

                category_value = "general"
                if hasattr(hierarchical_results, "category"):
                    if hasattr(hierarchical_results.category, "value"):
                        category_value = hierarchical_results.category.value
                    elif isinstance(hierarchical_results.category, str):
                        category_value = hierarchical_results.category

                from ..generation.ollama_client import GenerationRequest

                generation_request = GenerationRequest(
                    prompt=user_prompt,
                    context=context_chunks,
                    query=validated_query.text,
                    query_type=category_value,
                    language=validated_query.language,
                    metadata=validated_query.metadata,
                )

                system_logger.trace(
                    "query_processing", "ollama_request", f"Sending to model: {self.ollama_config.model}"
                )

                # AI-friendly TRACE logging for LLM request
                system_logger.trace(
                    "rag_system",
                    "query",
                    f"LLM_REQUEST | model={self.ollama_config.model} | "
                    f"context_chunks={len(context_chunks)} | "
                    f"context_chars={len(context_text)} | "
                    f"query='{validated_query.text[:100]}' | "
                    f"RAG_ENABLED=true",
                )

                system_logger.info("generation", "ollama_request", "Sending request to Ollama")
                generation_response = await self._generation_client.generate_text_async(generation_request)
                generation_time = time.time() - generation_start

                log_data_transformation(
                    "query_processing",
                    "llm_generation",
                    f"prompt: {len(user_prompt)} chars",
                    f"response: {len(generation_response.text)} chars in {generation_time:.3f}s",
                )

                # AI-friendly TRACE logging for LLM response
                system_logger.trace(
                    "rag_system",
                    "query",
                    f"LLM_RESPONSE | response_length={len(generation_response.text)} | "
                    f"generation_time={generation_time:.3f}s | "
                    f"used_context=true | chunks_in_context={len(context_chunks)} | "
                    f"response_preview='{generation_response.text[:200]}'",
                )

                log_performance_metric(
                    "generation",
                    "llm_generation",
                    "duration",
                    generation_time,
                    response_chars=len(generation_response.text),
                    model=getattr(generation_response, "model", "unknown"),
                )

                system_logger.info("parsing", "process_response", "Processing LLM response")
                parsed_response = self._response_parser.parse_response(
                    generation_response.text, validated_query.text, context_chunks
                )
                system_logger.debug(
                    "parsing", "process_response", f"Parsed response length: {len(parsed_response.content)}"
                )

                total_time = time.time() - start_time
                sources = extract_sources_from_chunks(hierarchical_results.documents) if return_sources else []
                retrieved_chunks = [
                    prepare_chunk_info(result, return_debug_info) for result in hierarchical_results.documents
                ]

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

                # Extract token usage from generation response
                tokens_used = getattr(generation_response, "tokens_used", 0)
                model_used = getattr(generation_response, "model", "unknown")

                # Extract input/output token breakdown from metadata
                gen_metadata = getattr(generation_response, "metadata", {})
                input_tokens = gen_metadata.get("input_tokens", 0)
                output_tokens = gen_metadata.get("output_tokens", 0)

                system_logger.info(
                    "pipeline",
                    "QUERY_PROCESSING",
                    f"COMPLETED: retrieval={retrieval_time:.2f}s, generation={generation_time:.2f}s, total={total_time:.2f}s, sources={len(sources)}, tokens={tokens_used} (in={input_tokens}, out={output_tokens})",
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
                    nn_sources=nn_sources if nn_sources else None,
                    tokens_used=tokens_used,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_used=model_used,
                )

            except Exception as e:
                system_logger.error(
                    "generation",
                    "pipeline_error",
                    "GENERATION/PARSING pipeline failed",
                    error_type=type(e).__name__,
                    stack_trace=str(e),
                )
                return create_error_response(validated_query, e, start_time)

        except Exception as e:
            # Import traceback to get full stack trace
            import traceback

            system_logger.error(
                "query_processing",
                "pipeline_error",
                f"QUERY_PROCESSING main pipeline failed: {type(e).__name__}: {str(e)}",
            )
            # Log the full traceback separately for complete debugging info
            system_logger.error("query_processing", "full_traceback", traceback.format_exc())
            return create_error_response(validated_query if "validated_query" in locals() else query, e, start_time)

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
            generation_health = await evaluate_ollama_health(self._generation_client, model_name)

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
    vector_storage: Any,
    search_engine: SearchEngineProtocol,
    query_processor: QueryProcessorProtocol,
    retriever: Any,
    hierarchical_retriever: RetrieverProtocol,
    ranker: RankerProtocol | None,
    generation_client: GenerationClientProtocol,
    response_parser: ResponseParserProtocol,
    prompt_builder: PromptBuilderProtocol,
    embedding_config: EmbeddingConfig,
    ollama_config: OllamaConfig,
    processing_config: ProcessingConfig,
    retrieval_config: RetrievalConfig,
    batch_config: dict[str, Any] | None = None,
    scope: str = "user",
    feature_name: str | None = None,
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
        batch_config=batch_config or {},
        scope=scope,
        feature_name=feature_name,
    )


# Note: Mock implementations have been moved to tests/fixtures/mock_rag_system.py
