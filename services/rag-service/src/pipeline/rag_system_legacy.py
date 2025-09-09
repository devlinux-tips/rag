"""
Complete Multilingual RAG System - End-to-End Pipeline
Orchestrates all components: preprocessing, vector storage, retrieval, and generation.

Supports multiple languages through configurable components and language-specific
settings. All components are language-agnostic and configured through TOML files.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..generation.enhanced_prompt_templates import \
    create_enhanced_prompt_builder
from ..generation.ollama_client import GenerationRequest, OllamaClient
from ..generation.prompt_templates import create_prompt_builder
from ..generation.response_parser import create_response_parser
from ..preprocessing.chunkers import DocumentChunker
from ..preprocessing.cleaners import MultilingualTextCleaner
from ..preprocessing.extractors import DocumentExtractor
from ..retrieval.hierarchical_retriever import \
    LegacyHierarchicalRetriever as HierarchicalRetriever
from ..retrieval.query_processor import MultilingualQueryProcessor
from ..retrieval.ranker import ResultRanker
from ..retrieval.retriever import IntelligentRetriever
from ..vectordb.embeddings import EmbeddingConfig
from ..vectordb.embeddings import MultilingualEmbeddingModel as EmbeddingModel
from ..vectordb.search import SemanticSearchEngine
from ..vectordb.storage import ChromaDBStorage
# Import all pipeline components
from .config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass
@dataclass
class RAGQuery:
    """RAG query with metadata."""

    text: str
    language: str  # Language code (hr, en, etc.)
    query_id: Optional[str] = None
    user_id: Optional[str] = None
    context_filters: Optional[Dict[str, Any]] = None
    max_results: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RAGResponse:
    """RAG response with full pipeline information."""

    answer: str
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    confidence: float
    generation_time: float
    retrieval_time: float
    total_time: float
    sources: List[str]
    metadata: Dict[str, Any]

    @property
    def has_high_confidence(self) -> bool:
        """Check if response has high confidence."""
        return self.confidence >= 0.8


class RAGSystem:
    """Complete multilingual RAG system orchestrating all pipeline components.

    Supports multiple languages through configurable components.
    """

    def __init__(self, language: str, config: Optional[RAGConfig] = None):
        """Initialize RAG system with language support."""

        # Store language preference
        self.language = language

        # Load config - fail fast if config missing
        if config is None:
            self.config = RAGConfig()
        else:
            self.config = config

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self._document_extractor = None
        self._text_cleaner = None
        self._chunker = None
        self._embedding_model = None
        self._vector_storage = None
        self._search_engine = None
        self._query_processor = None
        self._retriever = None
        self._hierarchical_retriever = None  # New enhanced retriever
        self._ranker = None
        self._ollama_client = None
        self._response_parser = None
        self._enhanced_prompt_builder = None  # New enhanced prompt builder

        # System state
        self._initialized = False
        self._document_count = 0
        self._query_count = 0

        self.logger.info("Multilingual RAG System created")

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            self.logger.info("System already initialized")
            return

        self.logger.info("Initializing Multilingual RAG System components...")

        try:
            # Setup language environment
            self._text_cleaner = MultilingualTextCleaner(language=self.language)
            self._text_cleaner.setup_language_environment()

            # Initialize preprocessing components
            await self._initialize_preprocessing()

            # Initialize vector database components
            await self._initialize_vectordb()

            # Initialize retrieval components
            await self._initialize_retrieval()

            # Initialize generation components
            await self._initialize_generation()

            self._initialized = True
            self.logger.info("✅ Multilingual RAG System initialization complete")

        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            raise

    async def _initialize_preprocessing(self) -> None:
        """Initialize document preprocessing components."""
        self.logger.info("🔧 Initializing preprocessing components...")

        self._document_extractor = DocumentExtractor()
        self._text_cleaner = MultilingualTextCleaner(language=self.language)
        self._chunker = DocumentChunker(
            chunk_size=self.config.processing.max_chunk_size,
            overlap=self.config.processing.chunk_overlap,
        )

        self.logger.info("✅ Preprocessing components initialized")

    async def _initialize_vectordb(self) -> None:
        """Initialize vector database components."""
        self.logger.info("🔧 Initializing vector database components...")

        # Initialize embedding model with language-specific cache
        language_cache_dir = f"{self.config.embedding.cache_folder}/{self.language}"

        embedding_config = EmbeddingConfig(
            model_name=self.config.embedding.model_name,
            cache_dir=language_cache_dir,
            device="auto",  # Auto-detect best device
            max_seq_length=self.config.embedding.max_seq_length,
            batch_size=self.config.embedding.batch_size,
            normalize_embeddings=True,  # Standard for RAG
            use_safetensors=True,  # Security setting
            trust_remote_code=False,  # Security setting
            torch_dtype="auto",  # Auto-detect best dtype
        )
        self._embedding_model = EmbeddingModel(embedding_config)
        self._embedding_model.load_model()

        # Initialize ChromaDB storage with language-specific collection
        from ..vectordb.storage import StorageConfig

        # Create language-specific collection name
        language_collection_map = {
            "hr": "croatian_documents",
            "en": "english_documents",
            "multilingual": "multilingual_documents",
        }
        collection_name = language_collection_map.get(
            self.language, f"{self.language}_documents"
        )

        storage_config = StorageConfig(
            db_path=self.config.chroma.db_path,
            collection_name=collection_name,
            distance_metric=self.config.chroma.distance_metric,
            persist=True,  # Always persist for production
            allow_reset=False,  # Prevent accidental data loss
        )
        self._vector_storage = ChromaDBStorage(storage_config)
        self._vector_storage.create_collection()

        # Initialize search engine
        self._search_engine = SemanticSearchEngine(
            storage=self._vector_storage, embedding_model=self._embedding_model
        )

        self.logger.info("✅ Vector database components initialized")

    async def _initialize_retrieval(self) -> None:
        """Initialize retrieval components."""
        self.logger.info("🔧 Initializing retrieval components...")

        self._query_processor = MultilingualQueryProcessor(language=self.language)

        self._ranker = ResultRanker(language=self.language)

        # Initialize both standard and hierarchical retrievers
        self._retriever = IntelligentRetriever(
            query_processor=self._query_processor, search_engine=self._search_engine
        )

        # Initialize new hierarchical retriever with enhanced categorization
        self._hierarchical_retriever = HierarchicalRetriever(
            search_engine=self._search_engine,
            language=self.language,
            enable_reranking=self.config.retrieval.enable_reranking,
        )

        self.logger.info(
            "✅ Retrieval components initialized (including hierarchical routing)"
        )

    async def _initialize_generation(self) -> None:
        """Initialize generation components."""
        self.logger.info("🔧 Initializing generation components...")

        # Initialize Ollama client with language config
        from ..generation.ollama_client import \
            OllamaConfig as OllamaClientConfig

        ollama_config = OllamaClientConfig(
            model=self.config.ollama.model,
            base_url=self.config.ollama.base_url,
            temperature=self.config.ollama.temperature,
            max_tokens=self.config.ollama.max_tokens,
            preserve_diacritics=self.config.ollama.preserve_diacritics,
            prefer_formal_style=self.config.ollama.prefer_formal_style,
        )

        self._ollama_client = OllamaClient(ollama_config)

        # Check Ollama availability
        if not self._ollama_client.health_check():
            self.logger.warning(
                "⚠️  Ollama service not available. Generation will fail."
            )
            self.logger.info("💡 To start Ollama: ollama serve")
            self.logger.info(f"💡 To pull model: ollama pull {self.config.ollama.model}")
        else:
            # Check if model is available
            available_models = self._ollama_client.get_available_models()
            if self.config.ollama.model not in available_models:
                self.logger.warning(f"⚠️  Model {self.config.ollama.model} not found")
                self.logger.info(
                    f"💡 Pull model: ollama pull {self.config.ollama.model}"
                )

        self._response_parser = create_response_parser()

        # Initialize enhanced prompt builder with category support
        self._enhanced_prompt_builder = create_enhanced_prompt_builder(
            language=self.language
        )

        self.logger.info(
            "✅ Generation components initialized (including enhanced prompts)"
        )

    async def add_documents(
        self, document_paths: List[str], batch_size: int = 10
    ) -> Dict[str, Any]:
        """Add documents to the RAG system."""
        if not self._initialized:
            await self.initialize()

        self.logger.info(f"📄 Processing {len(document_paths)} documents...")
        start_time = time.time()

        processed_docs = 0
        failed_docs = 0
        total_chunks = 0

        # Process documents in batches
        for i in range(0, len(document_paths), batch_size):
            batch = document_paths[i : i + batch_size]
            self.logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(document_paths) - 1) // batch_size + 1}"
            )

            for doc_path in batch:
                try:
                    # Extract text
                    from pathlib import Path

                    extracted_text = self._document_extractor.extract_text(
                        Path(doc_path)
                    )
                    if not extracted_text.strip():
                        self.logger.warning(f"⚠️  No text extracted from {doc_path}")
                        failed_docs += 1
                        continue

                    # Clean text with language-specific processing
                    cleaned_text = self._text_cleaner.clean_text(extracted_text)

                    # Create chunks
                    chunks = self._chunker.chunk_document(cleaned_text, doc_path)
                    if not chunks:
                        self.logger.warning(f"⚠️  No chunks created from {doc_path}")
                        failed_docs += 1
                        continue

                    # Create embeddings and store
                    for chunk_idx, chunk in enumerate(chunks):
                        # Generate embedding
                        embedding = self._embedding_model.encode_text(chunk.content)

                        # Ensure embedding is 1D for ChromaDB (some models return 2D arrays)
                        if embedding.ndim == 2:
                            embedding = embedding[0]  # Take first (and only) embedding

                        # Prepare metadata
                        metadata = {
                            "source": doc_path,
                            "chunk_index": chunk_idx,
                            "language": self.language,
                            "chunk_id": chunk.chunk_id,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "word_count": chunk.word_count,
                            "char_count": chunk.char_count,
                            "processing_timestamp": time.time(),
                        }

                        # Store in vector database
                        self._vector_storage.add_documents(
                            documents=[chunk.content],
                            metadatas=[metadata],
                            embeddings=[embedding.tolist()],
                        )

                    processed_docs += 1
                    total_chunks += len(chunks)
                    self.logger.info(f"✅ Processed {doc_path}: {len(chunks)} chunks")

                except Exception as e:
                    self.logger.error(f"❌ Failed to process {doc_path}: {e}")
                    failed_docs += 1

        processing_time = time.time() - start_time
        self._document_count += processed_docs

        result = {
            "processed_documents": processed_docs,
            "failed_documents": failed_docs,
            "total_chunks": total_chunks,
            "processing_time": processing_time,
            "documents_per_second": (
                processed_docs / processing_time if processing_time > 0 else 0
            ),
        }

        self.logger.info(f"📊 Document processing complete: {result}")
        return result

    async def query(
        self,
        query: RAGQuery,
        return_sources: bool = True,
        return_debug_info: bool = False,
    ) -> RAGResponse:
        """Execute complete RAG query pipeline."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        self._query_count += 1

        self.logger.info(f"🔍 Processing query: {query.text[:100]}...")

        try:
            # Step 1: Enhanced hierarchical retrieval with categorization
            query_start = time.time()

            # Use hierarchical retriever for intelligent routing
            hierarchical_results = await self._hierarchical_retriever.retrieve(
                query=query.text,
                max_results=query.max_results or 5,
                context=query.context_filters,
            )
            retrieval_time = time.time() - query_start

            self.logger.info(
                f"📂 Query categorized as: {hierarchical_results.category.value} "
                f"using {hierarchical_results.strategy_used.value} strategy"
            )

            # Step 2: Generate response with category-specific prompts
            generation_start = time.time()

            # Use enhanced prompt builder with category awareness
            context_chunks = [
                result["content"] for result in hierarchical_results.documents
            ]
            system_prompt, user_prompt = self._enhanced_prompt_builder.build_prompt(
                query=query.text,
                context_chunks=context_chunks,
                category=hierarchical_results.category,
                max_context_length=2000,
                include_source_attribution=True,
            )

            # Create generation request
            generation_request = GenerationRequest(
                prompt=user_prompt,
                context=context_chunks,
                query=query.text,
                query_type=hierarchical_results.category.value,  # Use detected category
                language=query.language,
                metadata=query.metadata,
            )

            # Generate response
            generation_response = await self._ollama_client.generate_text_async(
                generation_request
            )
            generation_time = time.time() - generation_start

            # Step 4: Parse and validate response
            parsed_response = self._response_parser.parse_response(
                generation_response.text, query.text, context_chunks
            )

            # Step 5: Build final response
            total_time = time.time() - start_time

            # Extract sources
            sources = []
            if return_sources:
                sources = list(
                    set(
                        [
                            result["metadata"].get("source", "Unknown")
                            for result in hierarchical_results.documents
                        ]
                    )
                )

            # Prepare retrieved chunks info
            retrieved_chunks = []
            for result in hierarchical_results.documents:
                chunk_info = {
                    "content": result["content"],
                    "similarity_score": result["similarity_score"],
                    "final_score": result["final_score"],
                    "source": result["metadata"].get("source", "Unknown"),
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                }

                if return_debug_info:
                    result_metadata = result["metadata"]
                    chunk_info.update(
                        {
                            "metadata": result_metadata,
                            "signals": result_metadata["ranking_signals"],
                        }
                    )

                retrieved_chunks.append(chunk_info)

            # Build metadata
            response_metadata = {
                "query_id": query.query_id,
                "user_id": query.user_id,
                "categorization": {
                    "detected_category": hierarchical_results.category.value,
                    "strategy_used": hierarchical_results.strategy_used.value,
                    "confidence": hierarchical_results.confidence,
                    "routing_metadata": hierarchical_results.routing_metadata,
                },
                "retrieval": {
                    "total_results": len(hierarchical_results.documents),
                    "strategy_used": hierarchical_results.strategy_used.value,
                    "filters_applied": query.context_filters or {},
                    "retrieval_time": hierarchical_results.retrieval_time,
                },
                "generation": {
                    "model": generation_response.model,
                    "tokens_used": generation_response.tokens_used,
                    "model_confidence": generation_response.confidence,
                },
                "parsing": {
                    "language_detected": parsed_response.language,
                    "sources_mentioned": parsed_response.sources_mentioned,
                },
                "performance": {
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                },
            }

            if return_debug_info:
                response_metadata["debug"] = {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_generation": generation_response.text,
                    "generation_metadata": generation_response.metadata,
                }

            response = RAGResponse(
                answer=parsed_response.content,
                query=query.text,
                retrieved_chunks=retrieved_chunks,
                confidence=parsed_response.confidence or 0.5,
                generation_time=generation_time,
                retrieval_time=retrieval_time,
                total_time=total_time,
                sources=sources,
                metadata=response_metadata,
            )

            self.logger.info(
                f"✅ Query processed in {total_time:.2f}s (confidence: {response.confidence:.3f})"
            )
            return response

        except Exception as e:
            self.logger.error(f"❌ Query processing failed: {e}")

            # Return error response in the query language
            error_msg = (
                "I apologize, an error occurred while processing your question"
                if query.language == "en"
                else "Žao mi je, dogodila se greška pri obradi pitanja"
            )
            return RAGResponse(
                answer=f"{error_msg}: {str(e)}",
                query=query.text,
                retrieved_chunks=[],
                confidence=0.0,
                generation_time=0.0,
                retrieval_time=0.0,
                total_time=time.time() - start_time,
                sources=[],
                metadata={
                    "error": str(e),
                    "query_id": query.query_id,
                    "user_id": query.user_id,
                },
            )

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            "system_status": "unknown",
            "components": {},
            "metrics": {},
            "timestamp": time.time(),
        }

        try:
            if not self._initialized:
                await self.initialize()

            # Check each component
            health_status["components"]["preprocessing"] = {
                "status": (
                    "healthy"
                    if all(
                        [self._document_extractor, self._text_cleaner, self._chunker]
                    )
                    else "unhealthy"
                ),
                "details": "All preprocessing components loaded",
            }

            health_status["components"]["vectordb"] = {
                "status": (
                    "healthy"
                    if all(
                        [
                            self._embedding_model,
                            self._vector_storage,
                            self._search_engine,
                        ]
                    )
                    else "unhealthy"
                ),
                "details": f"Vector DB with {self._vector_storage.get_document_count()} documents",
            }

            health_status["components"]["retrieval"] = {
                "status": (
                    "healthy"
                    if all(
                        [
                            self._query_processor,
                            self._retriever,
                            self._ranker,
                            self._hierarchical_retriever,
                        ]
                    )
                    else "unhealthy"
                ),
                "details": "All retrieval components loaded (including hierarchical routing)",
            }

            # Check Ollama service
            ollama_healthy = (
                self._ollama_client.health_check() if self._ollama_client else False
            )
            available_models = (
                self._ollama_client.get_available_models() if ollama_healthy else []
            )
            model_available = self.config.ollama.model in available_models

            health_status["components"]["generation"] = {
                "status": (
                    "healthy" if ollama_healthy and model_available else "degraded"
                ),
                "details": f"Ollama: {'✅' if ollama_healthy else '❌'}, Model {self.config.ollama.model}: {'✅' if model_available else '❌'}",
                "available_models": available_models,
            }

            # System metrics
            health_status["metrics"] = {
                "documents_processed": self._document_count,
                "queries_processed": self._query_count,
                "total_chunks": (
                    self._vector_storage.get_document_count()
                    if self._vector_storage
                    else 0
                ),
            }

            # Overall status
            component_statuses = [
                comp["status"] for comp in health_status["components"].values()
            ]
            if all(status == "healthy" for status in component_statuses):
                health_status["system_status"] = "healthy"
            elif any(status == "healthy" for status in component_statuses):
                health_status["system_status"] = "degraded"
            else:
                health_status["system_status"] = "unhealthy"

        except Exception as e:
            health_status["system_status"] = "error"
            health_status["error"] = str(e)

        return health_status

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self._initialized:
            return {"error": "System not initialized"}

        # Get config and update with actual runtime values
        config_dict = self.config.to_dict()

        # Update with actual runtime collection name
        if self._vector_storage and hasattr(self._vector_storage, "config"):
            config_dict["chroma"][
                "collection_name"
            ] = self._vector_storage.config.collection_name
            config_dict["chroma"]["actual_language"] = self.language

        stats = {
            "system": {
                "language": self.language,
                "initialized": self._initialized,
                "documents_processed": self._document_count,
                "queries_processed": self._query_count,
                "total_chunks": (
                    self._vector_storage.get_document_count()
                    if self._vector_storage
                    else 0
                ),
            },
            "collections": {
                "active_collection": (
                    self._vector_storage.config.collection_name
                    if self._vector_storage
                    else "none"
                ),
                "collection_type": f"{self.language}_documents"
                if self.language != "multilingual"
                else "multilingual_documents",
            },
            "models": {
                "embedding_model": self.config.embedding.model_name,
                "embedding_cache": f"{self.config.embedding.cache_folder}/{self.language}",
                "llm_model": self.config.ollama.model,
                "device": "cpu",  # TODO: get actual device from embedding model
            },
            "performance": {
                "chunk_size": self.config.processing.max_chunk_size,
                "max_retrieval": self.config.retrieval.max_k,
                "similarity_threshold": self.config.retrieval.min_similarity_score,
                "timeout": self.config.ollama.timeout,
            },
        }

        return stats

    async def close(self):
        """Clean shutdown of all components."""
        self.logger.info("🔄 Shutting down Multilingual RAG System...")

        if self._ollama_client:
            await self._ollama_client.close()

        if self._vector_storage:
            await self._vector_storage.close()

        self.logger.info("✅ Multilingual RAG System shutdown complete")


async def create_rag_system(
    language: str, config_path: Optional[str] = None
) -> RAGSystem:
    """Factory function to create and initialize RAG system."""
    config = RAGConfig()

    if config_path and Path(config_path).exists():
        # Load from YAML config
        from .config import load_yaml_config

        yaml_config = load_yaml_config(config_path)
        config = RAGConfig.from_dict(yaml_config)

    system = RAGSystem(config, language=language)
    await system.initialize()

    return system


# CLI interface for testing
async def main():
    """Main function for CLI testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Multilingual RAG System")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--add-docs", nargs="+", help="Add documents to the system")
    parser.add_argument("--query", help="Query the system")
    parser.add_argument("--lang", default="hr", help="Language code (hr, en, etc.)")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--stats", action="store_true", help="Show system stats")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create system
    system = await create_rag_system(args.config, language=args.lang)

    try:
        if args.health:
            health = await system.health_check()
            print("🏥 System Health Check:")
            print(json.dumps(health, indent=2, ensure_ascii=False))

        if args.stats:
            stats = await system.get_system_stats()
            print("📊 System Statistics:")
            print(json.dumps(stats, indent=2, ensure_ascii=False))

        if args.add_docs:
            print(f"📄 Adding {len(args.add_docs)} documents...")
            result = await system.add_documents(args.add_docs)
            print("✅ Document processing complete:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.query:
            print(f"🔍 Querying: {args.query}")
            query = RAGQuery(text=args.query, language=args.lang, query_id="cli-test")
            response = await system.query(query, return_debug_info=True)

            print("💬 Response:")
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Sources: {response.sources}")
            print(f"Retrieved chunks: {len(response.retrieved_chunks)}")
            print(f"Generation time: {response.generation_time:.2f}s")
            print(f"Retrieval time: {response.retrieval_time:.2f}s")
            print(f"Total time: {response.total_time:.2f}s")

            print("\n📊 Detailed Metadata:")

            # Convert metadata to JSON-serializable format
            def make_json_serializable(obj):
                """Convert objects to JSON-serializable format."""
                if hasattr(obj, "value"):  # Enum objects
                    return obj.value
                elif hasattr(obj, "__dict__"):  # Custom objects
                    return obj.__dict__
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                else:
                    return obj

            serializable_metadata = make_json_serializable(response.metadata)
            print(json.dumps(serializable_metadata, indent=2, ensure_ascii=False))

    finally:
        await system.close()


if __name__ == "__main__":
    asyncio.run(main())
