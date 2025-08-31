#!/usr/bin/env python3
"""
Demo Croatian RAG Queries - Show system working with example questions
"""
import asyncio
import os
from pathlib import Path

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig
from src.generation.ollama_client import OllamaClient, OllamaConfig, GenerationRequest
from src.retrieval.query_processor import CroatianQueryProcessor

async def demo_croatian_rag():
    """Demonstrate Croatian RAG with example queries."""
    
    print("🚀 Croatian RAG System Demo")
    print("=" * 50)
    
    # Load existing system (assumes data is already processed)
    print("🔧 Loading Croatian RAG components...")
    
    # Embedding model
    embedding_config = EmbeddingConfig(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        cache_dir="./temp_cache"
    )
    embedding_model = CroatianEmbeddingModel(embedding_config)
    embedding_model.load_model()
    
    # Vector storage (existing collection)
    storage_config = StorageConfig(
        collection_name="croatian_test",
        db_path="./temp_chroma_db"
    )
    storage = ChromaDBStorage(storage_config)
    
    # Check if collection exists and has data
    try:
        collection_info = storage.get_collection_info()
        if collection_info.get('count', 0) == 0:
            print("❌ No documents found in database. Please run the processing first.")
            return
        print(f"✅ Found {collection_info['count']} document chunks in database")
    except:
        print("❌ Database not found. Please run the processing first.")
        return
    
    # Query processor
    query_processor = CroatianQueryProcessor()
    
    # Ollama client
    ollama_config = OllamaConfig()
    ollama_client = OllamaClient(ollama_config)
    ollama_available = ollama_client.health_check()
    
    print(f"🤖 Ollama status: {'✅ Available' if ollama_available else '⚠️ Not available'}")
    print()
    
    # Example Croatian queries
    test_queries = [
        "O čemu govore dokumenti?",
        "Što je najvažnije u dokumentima?", 
        "Koji su glavni dijelovi?",
        "Kakve su teme obrađene?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"🇭🇷 Query {i}: '{query}'")
        print("-" * 30)
        
        # Process query
        processed_query = query_processor.process_query(query)
        print(f"📋 Query type: {processed_query.query_type} (confidence: {processed_query.confidence:.3f})")
        
        # Search for relevant documents
        query_embedding = embedding_model.encode_text(query)
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]
        
        results = storage.query_similar(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        
        if results and results.get('documents') and results['documents'][0]:
            print(f"📄 Found {len(results['documents'][0])} relevant chunks:")
            
            context = []
            for j, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                similarity = 1 - distance
                print(f"   {j+1}. (similarity: {similarity:.3f}) {doc[:150]}...")
                context.append(doc)
            
            # Generate answer if Ollama available
            if ollama_available:
                print("\n🤖 Generated answer:")
                try:
                    request = GenerationRequest(
                        prompt=query,
                        context=context,
                        query=query,
                        query_type="factual",
                        language="hr"
                    )
                    
                    response = await ollama_client.generate_text_async(request)
                    print(f"💬 {response.text}")
                except Exception as e:
                    print(f"❌ Generation error: {e}")
            else:
                print("⚠️ Ollama not available - showing retrieved context only")
        else:
            print("❌ No relevant documents found")
        
        print()
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(demo_croatian_rag())