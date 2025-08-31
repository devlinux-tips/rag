#!/usr/bin/env python3
"""
Debug what context the RAG system actually retrieved
"""
import asyncio
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig

async def debug_context():
    """Debug what context was actually retrieved."""
    
    # Set up same components
    embedding_config = EmbeddingConfig(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        cache_dir="./temp_cache"
    )
    embedding_model = CroatianEmbeddingModel(embedding_config)
    embedding_model.load_model()
    
    storage_config = StorageConfig(
        collection_name="normal_rag_system",
        db_path="./temp_chroma_db"
    )
    storage = ChromaDBStorage(storage_config)
    
    # Same query
    question = "Koje odluke su donesene 1. srpnja 2025, zanimaju nas samo iznosi u EURima?"
    
    query_embedding = embedding_model.encode_text(question)
    if len(query_embedding.shape) > 1:
        query_embedding = query_embedding[0]
    
    results = storage.query_similar(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )
    
    print("🔍 DEBUGGING: What context was actually retrieved?")
    print("=" * 60)
    print(f"Query: {question}")
    print()
    
    if results and results.get('documents') and results['documents'][0]:
        for j, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0], 
            results['distances'][0],
            results['metadatas'][0]
        )):
            similarity = 1 - distance
            print(f"📄 Chunk {j+1} (similarity: {similarity:.3f}) from {metadata.get('source')}:")
            print(f"Content: {doc}")
            print()
            
            # Check if EUR amounts are in the context
            if "EUR" in doc or "eur" in doc.lower():
                print("✅ This chunk contains EUR amounts!")
            else:
                print("❌ No EUR amounts in this chunk")
            print("-" * 50)
    
    print("\n🤔 ANALYSIS:")
    print("1. Did we retrieve chunks with EUR amounts?")
    print("2. Why did the LLM not mention them in the answer?")
    print("3. Is this a retrieval problem or generation problem?")

if __name__ == "__main__":
    asyncio.run(debug_context())