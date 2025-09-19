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

    def add_documents(self, documents: list[str], metadatas: list[dict], embeddings: list) -> None: ...

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict], embeddings: list) -> None: ...

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
    """Create error response in the appropriate language - configuration-driven."""
    error_msg = get_language_message(query.language, "error_message")

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
    ):
        """Initialize with all dependencies injected."""
        system_logger = get_system_logger()
        log_component_start("rag_system", "initialize", target_language=language)

        self.language = validate_language_code(language)
        system_logger.info("rag_system", "initialize", f"Language validated: {self.language}")

        self.embedding_config = embedding_config
        self.ollama_config = ollama_config
        self.processing_config = processing_config
        self.retrieval_config = retrieval_config

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

        self._vector_storage.create_collection()
        system_logger.debug("rag_system", "post_init", "Vector storage collection created")

        self._initialized = True
        log_component_end("rag_system", "post_init", "All components initialized successfully")

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
        system_logger = get_system_logger()
        log_component_start(
            "rag_system", "add_documents", doc_count=len(document_paths), batch_size=batch_size, language=self.language
        )

        if not self._initialized:
            system_logger.info("rag_system", "add_documents", "System not initialized, initializing now")
            await self.initialize()

        validated_paths = validate_document_paths(document_paths)
        system_logger.info(
            "pipeline", "DOCUMENT_PROCESSING", f"STARTED: {len(validated_paths)} documents, batch_size={batch_size}"
        )

        start_time = time.time()
        processed_docs = 0
        failed_docs = 0
        total_chunks = 0
        errors = []

        for i in range(0, len(validated_paths), batch_size):
            batch = validated_paths[i : i + batch_size]
            system_logger.debug(
                "pipeline",
                "process_batch",
                f"Batch {i // batch_size + 1}/{(len(validated_paths) - 1) // batch_size + 1}: {len(batch)} documents",
            )

            for doc_path in batch:
                doc_start = time.time()
                system_logger.debug("document_processing", "process_document", f"Starting: {doc_path}")

                try:
                    system_logger.trace("document_processing", "extract_text", f"Starting extraction: {doc_path}")
                    extraction_result = self._document_extractor.extract_text(doc_path)
                    extracted_text = (
                        extraction_result.text if hasattr(extraction_result, "text") else str(extraction_result)
                    )
                    if not extracted_text.strip():
                        error_msg = f"No text extracted from {doc_path}"
                        system_logger.warning("document_processing", "extract_text", error_msg)
                        log_error_context(
                            "document_processing",
                            "extract_text",
                            ValueError(error_msg),
                            {"doc_path": str(doc_path), "result_type": type(extraction_result).__name__},
                        )
                        errors.append(error_msg)
                        failed_docs += 1
                        continue

                    log_data_transformation(
                        "document_processing",
                        "extract_text",
                        f"document: {doc_path.name}",
                        f"text: {len(extracted_text)} chars",
                    )

                    system_logger.trace(
                        "document_processing", "clean_text", f"Starting cleaning: {len(extracted_text)} chars"
                    )
                    cleaning_result = self._text_cleaner.clean_text(extracted_text)
                    cleaned_text = cleaning_result.text if hasattr(cleaning_result, "text") else str(cleaning_result)
                    log_data_transformation(
                        "document_processing",
                        "clean_text",
                        f"{len(extracted_text)} chars",
                        f"{len(cleaned_text)} chars",
                    )

                    system_logger.trace(
                        "document_processing", "chunk_document", f"Starting chunking: {len(cleaned_text)} chars"
                    )
                    chunks = self._chunker.chunk_document(cleaned_text, str(doc_path))
                    if not chunks:
                        error_msg = f"No chunks created from {doc_path}"
                        system_logger.warning("document_processing", "chunk_document", error_msg)
                        log_error_context(
                            "document_processing",
                            "chunk_document",
                            ValueError(error_msg),
                            {"doc_path": str(doc_path), "text_length": len(cleaned_text)},
                        )
                        errors.append(error_msg)
                        failed_docs += 1
                        continue

                    log_data_transformation(
                        "document_processing", "chunk_document", f"{len(cleaned_text)} chars", f"{len(chunks)} chunks"
                    )

                    for chunk_idx, chunk in enumerate(chunks):
                        system_logger.trace(
                            "document_processing",
                            "process_chunk",
                            f"Chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk.content)} chars",
                        )

                        embedding_result = self._embedding_model.generate_embeddings([chunk.content])
                        embedding = embedding_result.embeddings

                        if hasattr(embedding, "ndim") and embedding.ndim == 2:
                            embedding = embedding[0]
                            system_logger.trace(
                                "document_processing",
                                "process_chunk",
                                f"Embedding shape normalized: {embedding.shape if hasattr(embedding, 'shape') else len(embedding)}",
                            )

                        metadata = create_chunk_metadata(str(doc_path), chunk_idx, chunk, self.language, time.time())
                        chunk_id = f"{doc_path.stem}_{chunk_idx}_{hash(chunk.content) % 1000000}"

                        self._vector_storage.add(
                            ids=[chunk_id],
                            documents=[chunk.content],
                            metadatas=[metadata],
                            embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else embedding],
                        )
                        system_logger.trace("document_processing", "vector_storage", f"Stored chunk {chunk_id}")

                    processed_docs += 1
                    total_chunks += len(chunks)

                    doc_time = time.time() - doc_start
                    log_performance_metric(
                        "document_processing",
                        "process_document",
                        "duration",
                        doc_time,
                        chunks=len(chunks),
                        chars=len(extracted_text),
                        doc_name=doc_path.name,
                    )

                except Exception as e:
                    import traceback

                    stack_trace = traceback.format_exc()
                    error_msg = f"Failed to process {doc_path}: {e}"
                    log_error_context(
                        "document_processing",
                        "process_document",
                        e,
                        {
                            "doc_path": str(doc_path),
                            "language": self.language,
                            "batch_size": batch_size,
                            "stack_trace": stack_trace,
                        },
                    )
                    system_logger.error(
                        "document_processing",
                        "process_document",
                        f"FAILED: {doc_path}",
                        error_type=type(e).__name__,
                        stack_trace=str(e),
                    )
                    errors.append(error_msg)
                    failed_docs += 1

        processing_time = time.time() - start_time
        self._document_count += processed_docs

        metrics = calculate_processing_metrics(processed_docs, processing_time, total_chunks)

        log_component_end(
            "rag_system",
            "add_documents",
            f"Processed {processed_docs}/{len(document_paths)} documents, {total_chunks} chunks",
            duration=processing_time,
            processed_docs=processed_docs,
            failed_docs=failed_docs,
        )

        system_logger.info(
            "pipeline",
            "DOCUMENT_PROCESSING",
            f"COMPLETED: {processed_docs} processed, {failed_docs} failed, {total_chunks} chunks in {processing_time:.2f}s",
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

                context_text = "\n\n".join(context_chunks) if context_chunks else "Nema dostupnih informacija."
                system_prompt = prompts_config["question_answering_system"]
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
                system_logger.info("generation", "ollama_request", "Sending request to Ollama")
                generation_response = await self._generation_client.generate_text_async(generation_request)
                generation_time = time.time() - generation_start

                log_data_transformation(
                    "query_processing",
                    "llm_generation",
                    f"prompt: {len(user_prompt)} chars",
                    f"response: {len(generation_response.text)} chars in {generation_time:.3f}s",
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

                system_logger.info(
                    "pipeline",
                    "QUERY_PROCESSING",
                    f"COMPLETED: retrieval={retrieval_time:.2f}s, generation={generation_time:.2f}s, total={total_time:.2f}s, sources={len(sources)}",
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
                system_logger.error(
                    "generation",
                    "pipeline_error",
                    "GENERATION/PARSING pipeline failed",
                    error_type=type(e).__name__,
                    stack_trace=str(e),
                )
                return create_error_response(validated_query, e, start_time)

        except Exception as e:
            system_logger.error(
                "query_processing",
                "pipeline_error",
                "QUERY_PROCESSING main pipeline failed",
                error_type=type(e).__name__,
                stack_trace=str(e),
            )
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
