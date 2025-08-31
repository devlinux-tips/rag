#!/usr/bin/env python3
"""
Test Croatian RAG system with CPU-only processing.
"""

import asyncio
import logging
import time
from pathlib import Path
import os

# Force CPU usage for torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from src.pipeline.config import RAGConfig
from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig
from src.preprocessing.extractors import DocumentExtractor
from src.preprocessing.cleaners import CroatianTextCleaner
from src.preprocessing.chunkers import DocumentChunker
from src.generation.ollama_client import OllamaClient, OllamaConfig
from src.retrieval.query_processor import CroatianQueryProcessor

logging.basicConfig(level=logging.INFO)

async def test_cpu_pipeline():
    """Test basic pipeline functionality with CPU-only processing."""
    
    print("🧪 Testing Croatian RAG Components (CPU Only)")
    print("=" * 60)
    
    try:
        # 1. Test document processing
        print("\n📄 Testing Document Processing...")
        extractor = DocumentExtractor()
        cleaner = CroatianTextCleaner()
        chunker = DocumentChunker(chunk_size=200, overlap=50)
        
        # Test with sample Croatian text
        sample_text = """
        Zagreb je glavni i najveći grad Republike Hrvatske, te ujedno i glavno političko, 
        gospodarsko i kulturno središte zemlje. Zagreb se prostire na 641 km² te broji 
        792.875 stanovnika (2021.), dok zagrebačka urbana aglomeracija ima 1.113.111 
        stanovnika.
        
        Grad se nalazi na sjeverozapadu Hrvatske, na južnim obroncima Medvednice, uz rijeku 
        Savu na prosječnoj nadmorskoj visini od 158 metara.
        """
        
        cleaned_text = cleaner.clean_text(sample_text)
        chunks = chunker.chunk_document(cleaned_text, "sample.txt")
        
        print(f"✅ Created {len(chunks)} text chunks")
        if chunks:
            print(f"   First chunk: {chunks[0].content[:100]}...")
        
        # 2. Test embeddings with CPU
        print("\n🔤 Testing Embeddings (CPU)...")
        embedding_config = EmbeddingConfig(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu",  # Force CPU
            cache_dir="./temp_cache"
        )
        
        embedding_model = CroatianEmbeddingModel(embedding_config)
        embedding_model.load_model()
        print("✅ Embedding model loaded on CPU")
        
        # Generate embedding for first chunk
        if chunks:
            embedding = embedding_model.encode_text(chunks[0].content)
            if len(embedding.shape) > 1:
                embedding = embedding[0]  # Get first embedding if batch
            print(f"✅ Generated embedding: {len(embedding)} dimensions")
        
        # 3. Test vector storage
        print("\n🗄️ Testing Vector Storage...")
        storage_config = StorageConfig(
            collection_name="test_collection",
            db_path="./temp_chroma_db"
        )
        storage = ChromaDBStorage(storage_config)
        # ChromaDB initializes automatically
        print("✅ ChromaDB storage initialized")
        
        # Store embeddings
        documents = []
        metadatas = []
        embeddings_list = []
        
        for i, chunk in enumerate(chunks[:3]):  # Only first 3 chunks
            embedding = embedding_model.encode_text(chunk.content)
            if len(embedding.shape) > 1:
                embedding = embedding[0]  # Get first embedding if batch
            
            documents.append(chunk.content)
            metadatas.append({
                "chunk_id": chunk.chunk_id,
                "source": "sample.txt",
                "language": "hr"
            })
            embeddings_list.append(embedding.tolist())
        
        # Create collection and add documents
        storage.create_collection()
        storage.add_documents(documents, metadatas, embeddings_list)
        
        print(f"✅ Stored {min(len(chunks), 3)} embeddings")
        
        # Test search
        if chunks:
            query_embedding = embedding_model.encode_text("Zagreb glavni grad")
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]  # Get first embedding if batch
            
            results = storage.query_similar(
                query_embeddings=[query_embedding.tolist()],
                n_results=2
            )
            
            if results and 'documents' in results and results['documents'][0]:
                print(f"✅ Search found {len(results['documents'][0])} similar chunks")
                
                for i, (doc, distance) in enumerate(zip(results['documents'][0][:1], results['distances'][0][:1])):
                    similarity = 1 - distance  # Convert distance to similarity
                    print(f"   Result {i+1}: {doc[:100]}... (score: {similarity:.3f})")
            else:
                print("✅ Search completed but no results found")
        
        # 4. Test query processor
        print("\n🔍 Testing Query Processing...")
        query_processor = CroatianQueryProcessor()
        
        test_queries = [
            "Što je Zagreb?",
            "Koliko stanovnika ima Zagreb?",
            "Gdje se nalazi Zagreb?"
        ]
        
        for query in test_queries:
            processed = query_processor.process_query(query)
            print(f"✅ Query: '{query}' -> Type: {processed.query_type}, Confidence: {processed.confidence:.3f}")
        
        # 5. Test Ollama client
        print("\n🤖 Testing Ollama Client...")
        ollama_config = OllamaConfig()
        ollama_client = OllamaClient(ollama_config)
        
        if ollama_client.health_check():
            print("✅ Ollama service available")
            
            # Simple generation test
            test_context = [chunks[0].content] if chunks else ["Zagreb je glavni grad Hrvatske."]
            
            # Use async method directly
            from src.generation.ollama_client import GenerationRequest
            request = GenerationRequest(
                prompt="Što je Zagreb?",
                context=test_context,
                query="Što je Zagreb?",
                query_type="factual",
                language="hr"
            )
            
            response = await ollama_client.generate_text_async(request)
            print(f"✅ Generated response: {response.text[:150]}...")
        else:
            print("⚠️  Ollama service not available")
        
        print(f"\n🎉 All CPU-only tests completed successfully!")
        
        # Cleanup
        # Note: ChromaDBStorage doesn't require async cleanup
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    success = asyncio.run(test_cpu_pipeline())
    if success:
        print("\n✅ Croatian RAG system components working correctly on CPU!")
        print("You can now run the full system test.")
    else:
        print("\n❌ Some issues found. Check the output above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())