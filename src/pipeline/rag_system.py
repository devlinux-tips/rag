"""
Complete Croatian RAG System - End-to-End Pipeline
Orchestrates all components: preprocessing, vector storage, retrieval, and generation.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..generation.ollama_client import GenerationRequest, OllamaClient
from ..generation.prompt_templates import create_prompt_builder
from ..generation.response_parser import create_response_parser
from ..preprocessing.chunkers import DocumentChunker
from ..preprocessing.cleaners import CroatianTextCleaner
from ..preprocessing.extractors import DocumentExtractor
from ..retrieval.query_processor import CroatianQueryProcessor
from ..retrieval.ranker import CroatianResultRanker
from ..retrieval.retriever import IntelligentRetriever
from ..utils.croatian_utils import setup_croatian_environment
from ..vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from ..vectordb.search import SemanticSearchEngine
from ..vectordb.storage import ChromaDBStorage

# Import all pipeline components
from .config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGQuery:
    """RAG query with metadata."""

    text: str
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


class CroatianRAGSystem:
    """Complete Croatian RAG system orchestrating all pipeline components."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize Croatian RAG system."""
        self.config = config or RAGConfig()
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
        self._ranker = None
        self._ollama_client = None
        self._response_parser = None

        # System state
        self._initialized = False
        self._document_count = 0
        self._query_count = 0

        self.logger.info("Croatian RAG System created")

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            self.logger.info("System already initialized")
            return

        self.logger.info("Initializing Croatian RAG System components...")

        try:
            # Setup Croatian environment
            setup_croatian_environment()

            # Initialize preprocessing components
            await self._initialize_preprocessing()

            # Initialize vector database components
            await self._initialize_vectordb()

            # Initialize retrieval components
            await self._initialize_retrieval()

            # Initialize generation components
            await self._initialize_generation()

            self._initialized = True
            self.logger.info("✅ Croatian RAG System initialization complete")

        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            raise

    async def _initialize_preprocessing(self) -> None:
        """Initialize document preprocessing components."""
        self.logger.info("🔧 Initializing preprocessing components...")

        self._document_extractor = DocumentExtractor()
        self._text_cleaner = CroatianTextCleaner()
        self._chunker = DocumentChunker(
            chunk_size=self.config.processing.max_chunk_size,
            overlap=self.config.processing.chunk_overlap,
        )

        self.logger.info("✅ Preprocessing components initialized")

    async def _initialize_vectordb(self) -> None:
        """Initialize vector database components."""
        self.logger.info("🔧 Initializing vector database components...")

        # Initialize embedding model
        embedding_config = EmbeddingConfig(
            model_name=self.config.embedding.model_name,
            cache_dir=self.config.embedding.cache_folder,
            batch_size=self.config.embedding.batch_size,
            max_seq_length=self.config.embedding.max_seq_length,
        )
        self._embedding_model = CroatianEmbeddingModel(embedding_config)
        self._embedding_model.load_model()

        # Initialize ChromaDB storage
        from ..vectordb.storage import StorageConfig

        storage_config = StorageConfig(
            db_path=self.config.chroma.db_path,
            collection_name=self.config.chroma.collection_name,
            distance_metric=self.config.chroma.distance_metric,
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

        self._query_processor = CroatianQueryProcessor()

        self._ranker = CroatianResultRanker()

        self._retriever = IntelligentRetriever(
            query_processor=self._query_processor, search_engine=self._search_engine
        )

        self.logger.info("✅ Retrieval components initialized")

    async def _initialize_generation(self) -> None:
        """Initialize generation components."""
        self.logger.info("🔧 Initializing generation components...")

        # Initialize Ollama client with Croatian config
        from ..generation.ollama_client import OllamaConfig as OllamaClientConfig

        ollama_config = OllamaClientConfig(
            model=self.config.ollama.model,
            base_url=self.config.ollama.base_url,
            temperature=self.config.ollama.temperature,
            max_tokens=self.config.ollama.max_tokens,
            preserve_diacritics=self.config.ollama.preserve_diacritics,
            prefer_formal_style=self.config.ollama.prefer_formal_style,
            include_cultural_context=self.config.ollama.include_cultural_context,
        )

        self._ollama_client = OllamaClient(ollama_config)

        # Check Ollama availability
        if not self._ollama_client.health_check():
            self.logger.warning("⚠️  Ollama service not available. Generation will fail.")
            self.logger.info("💡 To start Ollama: ollama serve")
            self.logger.info(f"💡 To pull model: ollama pull {self.config.ollama.model}")
        else:
            # Check if model is available
            available_models = self._ollama_client.get_available_models()
            if self.config.ollama.model not in available_models:
                self.logger.warning(f"⚠️  Model {self.config.ollama.model} not found")
                self.logger.info(f"💡 Pull model: ollama pull {self.config.ollama.model}")

        self._response_parser = create_response_parser()

        self.logger.info("✅ Generation components initialized")

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
                f"Processing batch {i//batch_size + 1}/{(len(document_paths)-1)//batch_size + 1}"
            )

            for doc_path in batch:
                try:
                    # Extract text
                    from pathlib import Path

                    extracted_text = self._document_extractor.extract_text(Path(doc_path))
                    if not extracted_text.strip():
                        self.logger.warning(f"⚠️  No text extracted from {doc_path}")
                        failed_docs += 1
                        continue

                    # Clean Croatian text
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
                        embedding = await self._embedding_model.generate_embedding_async(
                            chunk.content
                        )

                        # Prepare metadata
                        metadata = {
                            "source": doc_path,
                            "chunk_index": chunk_idx,
                            "language": "hr",
                            "chunk_id": chunk.chunk_id,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "word_count": chunk.word_count,
                            "char_count": chunk.char_count,
                            "processing_timestamp": time.time(),
                        }

                        # Store in vector database
                        await self._vector_storage.store_embedding_async(
                            embedding=embedding, text=chunk.content, metadata=metadata
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
            # Step 1: Process query
            query_start = time.time()
            processed_query = self._query_processor.process_query(query.text)

            # Step 2: Retrieve relevant chunks
            retrieval_results = await self._retriever.retrieve_async(
                processed_query=processed_query,
                k=query.max_results or self.config.retrieval.default_k,
                filters=query.context_filters,
            )
            retrieval_time = time.time() - query_start

            # Step 3: Generate response
            generation_start = time.time()

            # Build prompt
            prompt_builder = create_prompt_builder(query.text)
            context_chunks = [result.content for result in retrieval_results.results]
            system_prompt, user_prompt = prompt_builder.build_prompt(query.text, context_chunks)

            # Create generation request
            generation_request = GenerationRequest(
                prompt=user_prompt,
                context=context_chunks,
                query=query.text,
                query_type=processed_query.query_type.value,
                language="hr",
                metadata=query.metadata,
            )

            # Generate response
            generation_response = await self._ollama_client.generate_text_async(generation_request)
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
                            result.metadata.get("source", "Unknown")
                            for result in retrieval_results.results
                        ]
                    )
                )

            # Prepare retrieved chunks info
            retrieved_chunks = []
            for result in retrieval_results.results:
                chunk_info = {
                    "content": result.content,
                    "similarity_score": result.similarity_score,
                    "final_score": result.final_score,
                    "source": result.metadata.get("source", "Unknown"),
                    "chunk_index": result.metadata.get("chunk_index", 0),
                }

                if return_debug_info:
                    chunk_info.update(
                        {
                            "metadata": result.metadata,
                            "signals": result.metadata.get("ranking_signals", {}),
                        }
                    )

                retrieved_chunks.append(chunk_info)

            # Build metadata
            response_metadata = {
                "query_id": query.query_id,
                "user_id": query.user_id,
                "processed_query": {
                    "original": processed_query.original,
                    "processed": processed_query.processed,
                    "query_type": processed_query.query_type.value,
                    "keywords": processed_query.keywords,
                    "confidence": processed_query.confidence,
                },
                "retrieval": {
                    "total_results": len(retrieval_results.results),
                    "strategy_used": retrieval_results.strategy_used,
                    "filters_applied": query.context_filters or {},
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

            # Return error response
            return RAGResponse(
                answer=f"Žao mi je, dogodila se greška pri obradi pitanja: {str(e)}",
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
                    if all([self._document_extractor, self._text_cleaner, self._chunker])
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
                "details": f"Vector DB with {await self._vector_storage.get_document_count()} documents",
            }

            health_status["components"]["retrieval"] = {
                "status": (
                    "healthy"
                    if all([self._query_processor, self._retriever, self._ranker])
                    else "unhealthy"
                ),
                "details": "All retrieval components loaded",
            }

            # Check Ollama service
            ollama_healthy = self._ollama_client.health_check() if self._ollama_client else False
            available_models = self._ollama_client.get_available_models() if ollama_healthy else []
            model_available = self.config.ollama.model in available_models

            health_status["components"]["generation"] = {
                "status": ("healthy" if ollama_healthy and model_available else "degraded"),
                "details": f"Ollama: {'✅' if ollama_healthy else '❌'}, Model {self.config.ollama.model}: {'✅' if model_available else '❌'}",
                "available_models": available_models,
            }

            # System metrics
            health_status["metrics"] = {
                "documents_processed": self._document_count,
                "queries_processed": self._query_count,
                "total_chunks": (
                    await self._vector_storage.get_document_count() if self._vector_storage else 0
                ),
            }

            # Overall status
            component_statuses = [comp["status"] for comp in health_status["components"].values()]
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

        stats = {
            "documents": self._document_count,
            "queries": self._query_count,
            "chunks": (
                await self._vector_storage.get_document_count() if self._vector_storage else 0
            ),
            "config": self.config.to_dict(),
        }

        return stats

    async def close(self):
        """Clean shutdown of all components."""
        self.logger.info("🔄 Shutting down Croatian RAG System...")

        if self._ollama_client:
            await self._ollama_client.close()

        if self._vector_storage:
            await self._vector_storage.close()

        self.logger.info("✅ Croatian RAG System shutdown complete")


async def create_rag_system(config_path: Optional[str] = None) -> CroatianRAGSystem:
    """Factory function to create and initialize RAG system."""
    config = RAGConfig()

    if config_path and Path(config_path).exists():
        # Load from YAML config
        from .config import load_yaml_config

        yaml_config = load_yaml_config(config_path)
        config = RAGConfig.from_dict(yaml_config)

    system = CroatianRAGSystem(config)
    await system.initialize()

    return system


# CLI interface for testing
async def main():
    """Main function for CLI testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Croatian RAG System")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--add-docs", nargs="+", help="Add documents to the system")
    parser.add_argument("--query", help="Query the system")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--stats", action="store_true", help="Show system stats")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create system
    system = await create_rag_system(args.config)

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
            query = RAGQuery(text=args.query, query_id="cli-test")
            response = await system.query(query, return_debug_info=True)

            print("💬 Response:")
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Sources: {response.sources}")
            print(f"Time: {response.total_time:.2f}s")

    finally:
        await system.close()


if __name__ == "__main__":
    asyncio.run(main())
