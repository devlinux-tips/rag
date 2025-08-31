#!/usr/bin/env python3
"""
Quick test to see the exact retrieved content and LLM response
"""
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from src.preprocessing.extractors import DocumentExtractor
from src.preprocessing.cleaners import CroatianTextCleaner
from src.preprocessing.chunkers import DocumentChunker
from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig
from src.generation.ollama_client import OllamaClient, OllamaConfig, GenerationRequest

async def quick_test():
    print("🔬 Quick RAG Test - Debug Retrieved Context")
    
    # Extract some Croatian text with EUR amounts
    extractor = DocumentExtractor()
    doc_file = "data/raw/NN - 2025 - 116 - 1683.pdf"  # This one has EUR amounts
    from pathlib import Path
    text = extractor.extract_text(Path(doc_file))
    
    # Find parts with EUR amounts
    lines = text.split('\n')
    eur_lines = [line for line in lines if 'EUR' in line or '15,32' in line or '331,23' in line]
    
    print(f"📄 Raw text with EUR amounts from {doc_file}:")
    for line in eur_lines[:5]:
        print(f"   {line.strip()}")
    
    # Set up minimal RAG components
    cleaner = CroatianTextCleaner()
    chunker = DocumentChunker(chunk_size=512, overlap=50)
    
    embedding_config = EmbeddingConfig(model_name="distiluse-base-multilingual-cased", device="cpu", cache_dir="./temp_cache")
    embedding_model = CroatianEmbeddingModel(embedding_config)
    embedding_model.load_model()
    
    storage_config = StorageConfig(collection_name="distiluse_test", db_path="./temp_chroma_db")
    storage = ChromaDBStorage(storage_config)
    storage.create_collection()
    
    # Process and store the document
    cleaned_text = cleaner.clean_text(text)
    chunks = chunker.chunk_document(cleaned_text, doc_file)
    
    # Store chunks that contain EUR amounts
    eur_chunks = []
    for chunk in chunks:
        if 'EUR' in chunk.content or '15,32' in chunk.content or '331,23' in chunk.content:
            eur_chunks.append(chunk)
    
    print(f"\n📊 Found {len(eur_chunks)} chunks with EUR amounts")
    
    documents_list = []
    metadatas_list = []
    embeddings_list = []
    
    for i, chunk in enumerate(eur_chunks[:5]):  # First 5 EUR chunks
        embedding = embedding_model.encode_text(chunk.content)
        if len(embedding.shape) > 1:
            embedding = embedding[0]
        
        documents_list.append(chunk.content)
        metadatas_list.append({"source": doc_file, "chunk_id": chunk.chunk_id})
        embeddings_list.append(embedding.tolist())
        
        print(f"💰 EUR Chunk {i+1}: {chunk.content[:200]}...")
    
    storage.add_documents(documents_list, metadatas_list, embeddings_list)
    
    # Now test the query
    question = "Koje odluke su donesene 1. srpnja 2025, zanimaju nas samo iznosi u EURima?"
    
    query_embedding = embedding_model.encode_text(question)
    if len(query_embedding.shape) > 1:
        query_embedding = query_embedding[0]
    
    results = storage.query_similar(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    
    print(f"\n🔍 Retrieved context for: '{question}'")
    print("="*60)
    
    context = []
    for j, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0], 
        results['distances'][0],
        results['metadatas'][0]
    )):
        similarity = 1 - distance
        print(f"\n📄 Context {j+1} (similarity: {similarity:.3f}):")
        print(f"Content: {doc}")
        context.append(doc)
        
        # Check for EUR amounts
        import re
        eur_amounts = re.findall(r'(\d+[.,]\d+)\s*EUR', doc)
        if eur_amounts:
            print(f"💰 EUR amounts in this context: {eur_amounts}")
    
    # Generate answer
    print(f"\n🤖 Generating answer with this context...")
    
    ollama_config = OllamaConfig(model="qwen2.5:7b-instruct")
    ollama_client = OllamaClient(ollama_config)
    
    if ollama_client.health_check():
        # Build explicit prompt that forces model to use context
        explicit_prompt = f"""Ti si pomoćnik koji mora odgovoriti na pitanja koristeći SAMO informacije iz priloženih dokumenata.

DOKUMENTI:
{chr(10).join([f"Dokument {i+1}: {doc}" for i, doc in enumerate(context)])}

PITANJE: {question}

PRAVILA:
1. Koristi SAMO informacije iz gore navedenih dokumenata
2. Izvuci točne EUR iznose spomenute u dokumentima
3. Ne dodavaj informacije koje nema u dokumentima
4. Odgovori na hrvatskom jeziku
5. Ako nema informacija u dokumentima, reci "Nema informacija u priloženim dokumentima"

ODGOVOR:"""

        request = GenerationRequest(
            prompt=explicit_prompt,
            context=context,
            query=question,
            query_type="factual",
            language="hr"
        )
        
        response = await ollama_client.generate_text_async(request)
        print(f"\n💬 LLM Response:")
        print(f"{response.text}")
        
        # Check if EUR amounts are in the response
        response_eur = re.findall(r'(\d+[.,]\d+)\s*(?:EUR|eur|eura)', response.text)
        if response_eur:
            print(f"\n✅ EUR amounts found in response: {response_eur}")
        else:
            print(f"\n❌ No EUR amounts in the response!")
            print("🔧 This indicates the LLM is not extracting EUR amounts from the context properly")

if __name__ == "__main__":
    asyncio.run(quick_test())