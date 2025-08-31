#!/usr/bin/env python3
"""
Simple test to verify basic imports and components work.
"""

import asyncio
import sys

# Test basic imports
try:
    print("🔧 Testing imports...")

    # Test config
    from src.pipeline.config import RAGConfig

    print("✅ Config import successful")

    # Test individual components
    from src.preprocessing.extractors import DocumentExtractor

    print("✅ DocumentExtractor import successful")

    from src.preprocessing.cleaners import CroatianTextCleaner

    print("✅ CroatianTextCleaner import successful")

    from src.preprocessing.chunkers import DocumentChunker

    print("✅ DocumentChunker import successful")

    from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig

    print("✅ Embedding imports successful")

    from src.vectordb.storage import ChromaDBStorage

    print("✅ ChromaVectorStorage import successful")

    from src.retrieval.query_processor import CroatianQueryProcessor

    print("✅ CroatianQueryProcessor import successful")

    from src.generation.ollama_client import OllamaClient, OllamaConfig

    print("✅ Ollama imports successful")

    print("\n🎉 All individual imports successful!")

    # Test basic functionality
    print("\n🧪 Testing basic functionality...")

    # Test config creation
    config = RAGConfig()
    print(f"✅ RAGConfig created - embedding model: {config.embedding.model_name}")

    # Test document extractor
    extractor = DocumentExtractor()
    print("✅ DocumentExtractor created")

    # Test text cleaner
    cleaner = CroatianTextCleaner()
    test_text = "Ovo je test Croatian tekst sa dijakritičkim znakovima: čćšžđ"
    cleaned = cleaner.clean_text(test_text)
    print(f"✅ Text cleaning works: '{cleaned[:50]}...'")

    # Test chunker
    chunker = DocumentChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_document(cleaned, "test.txt")
    print(f"✅ Chunking works: created {len(chunks)} chunks")

    # Test Ollama client
    ollama_config = OllamaConfig()
    ollama_client = OllamaClient(ollama_config)
    print(f"✅ OllamaClient created with model: {ollama_config.model}")

    # Test if Ollama is available
    if ollama_client.health_check():
        print("✅ Ollama service is running!")
    else:
        print("⚠️  Ollama service not available")

    print("\n🎉 Basic functionality tests passed!")

except Exception as e:
    print(f"❌ Import/test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# Test async functionality
async def test_async_components():
    """Test async components."""
    try:
        print("\n🔄 Testing async components...")

        # Test embedding model
        embedding_config = EmbeddingConfig(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir="./temp_cache",
            batch_size=16,
        )

        embedding_model = CroatianEmbeddingModel(embedding_config)
        print("✅ CroatianEmbeddingModel created")

        # Load model
        await embedding_model.load_model()
        print("✅ Embedding model loaded")

        # Test embedding generation
        test_text = "Zagreb je glavni grad Hrvatske."
        embedding = await embedding_model.generate_embedding_async(test_text)
        print(f"✅ Embedding generated: {len(embedding)} dimensions")

        # Test vector storage
        storage = ChromaDBStorage()
        await storage.initialize()
        print("✅ ChromaDBStorage initialized")

        # Test storing embedding
        await storage.store_embedding_async(
            embedding=embedding,
            text=test_text,
            metadata={"test": True, "language": "hr"},
        )
        print("✅ Embedding stored successfully")

        # Test search
        results = await storage.similarity_search_async(embedding, k=1)
        print(f"✅ Search works: found {len(results)} results")

        print("\n🎉 Async components test passed!")

        # Cleanup
        await storage.close()

        return True

    except Exception as e:
        print(f"❌ Async test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_async_components())
    if success:
        print("\n🎯 All tests passed! System components are working correctly.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
