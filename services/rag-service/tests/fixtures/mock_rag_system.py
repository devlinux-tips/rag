"""
Mock RAG system components for testing.
Moved from src/pipeline/rag_system.py to proper testing location.
"""

import hashlib
from pathlib import Path
from typing import Any


class MockDocumentExtractor:
    """Mock document extractor for testing."""

    def extract_text(self, file_path: Path) -> str:
        return f"Mock content from {file_path.name}"


class MockTextCleaner:
    """Mock text cleaner for testing."""

    def clean_text(self, text: str) -> str:
        return text.strip()

    def setup_language_environment(self) -> None:
        pass


class MockDocumentChunker:
    """Mock document chunker for testing."""

    def chunk_document(self, content: str, source_file: str) -> list[Any]:
        # Create simple mock chunks
        class MockChunk:
            def __init__(self, content: str, idx: int):
                self.content = content
                self.chunk_id = f"chunk_{idx}"
                self.start_char = 0
                self.end_char = len(content)
                self.word_count = len(content.split())
                self.char_count = len(content)

        # Split content into chunks of ~100 chars
        chunks = []
        chunk_size = 100
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i : i + chunk_size]
            chunks.append(MockChunk(chunk_content, i // chunk_size))

        return chunks


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def encode_text(self, text: str) -> list[float]:
        # Return deterministic mock embeddings based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to numbers
        numbers = [ord(c) / 255.0 for c in text_hash[:10]]  # 10-dim embedding
        return numbers

    def load_model(self) -> None:
        pass


class MockVectorStorage:
    """Mock vector storage for testing."""

    def __init__(self):
        self._documents = []

    def add_documents(
        self, documents: list[str], metadatas: list[dict], embeddings: list
    ) -> None:
        for doc, meta, emb in zip(documents, metadatas, embeddings, strict=False):
            self._documents.append({"content": doc, "metadata": meta, "embedding": emb})

    def create_collection(self) -> None:
        pass

    def get_document_count(self) -> int:
        return len(self._documents)

    async def close(self) -> None:
        pass


class MockGenerationClient:
    """Mock generation client for testing."""

    def __init__(self, healthy: bool = True, available_models: list[str] = None):
        self.healthy = healthy
        self.available_models = available_models or ["mock-model"]

    async def generate_text_async(self, request: Any) -> Any:
        class MockResponse:
            text = "Mock generated response"
            model = "mock-model"
            tokens_used = 50
            confidence = 0.8
            metadata = {}

        return MockResponse()

    def health_check(self) -> bool:
        return self.healthy

    def get_available_models(self) -> list[str]:
        return self.available_models

    async def close(self) -> None:
        pass


class MockResponseParser:
    """Mock response parser for testing."""

    def parse_response(self, text: str, query: str, context: list[str]) -> Any:
        class MockParsedResponse:
            content = text
            confidence = 0.8
            language = "en"
            sources_mentioned = []

        return MockParsedResponse()


class MockPromptBuilder:
    """Mock prompt builder for testing."""

    def build_prompt(
        self, query: str, context_chunks: list[str], **kwargs
    ) -> tuple[str, str]:
        system_prompt = "You are a helpful assistant."
        user_prompt = f"Query: {query}\nContext: {' '.join(context_chunks[:2])}"
        return system_prompt, user_prompt


class MockRetriever:
    """Mock retriever for testing."""

    async def retrieve(
        self, query: str, max_results: int = 5, context: dict | None = None
    ) -> Any:
        class MockCategory:
            value = "general"

        class MockStrategy:
            value = "standard"

        class MockResults:
            category = MockCategory()
            strategy_used = MockStrategy()
            confidence = 0.8
            routing_metadata = {}
            retrieval_time = 0.1
            documents = [
                {
                    "content": f"Mock retrieved content for query: {query}",
                    "similarity_score": 0.9,
                    "final_score": 0.9,
                    "metadata": {"source": "mock_doc.txt", "chunk_index": 0},
                }
            ]

        return MockResults()


def create_mock_rag_system(language: str = "hr", config: dict[str, Any] | None = None):
    """Create a fully mocked RAG system for testing."""
    from src.pipeline.rag_system import create_rag_system

    # Create mock config objects for the updated RAGSystem interface
    class MockEmbeddingConfig:
        model_name = "mock-embedding-model"
        device = "cpu"

    class MockOllamaConfig:
        model = "mock-llm-model"
        timeout = 30

    class MockProcessingConfig:
        enable_smart_chunking = True

    class MockRetrievalConfig:
        max_k = 5
        default_k = 3

    return create_rag_system(
        language=language,
        document_extractor=MockDocumentExtractor(),
        text_cleaner=MockTextCleaner(),
        document_chunker=MockDocumentChunker(),
        embedding_model=MockEmbeddingModel(),
        vector_storage=MockVectorStorage(),
        search_engine=None,  # Not used directly in current implementation
        query_processor=None,  # Not used directly in current implementation
        retriever=MockRetriever(),
        hierarchical_retriever=MockRetriever(),
        ranker=None,  # Not used directly in current implementation
        generation_client=MockGenerationClient(),
        response_parser=MockResponseParser(),
        prompt_builder=MockPromptBuilder(),
        embedding_config=MockEmbeddingConfig(),
        ollama_config=MockOllamaConfig(),
        processing_config=MockProcessingConfig(),
        retrieval_config=MockRetrievalConfig(),
    )
