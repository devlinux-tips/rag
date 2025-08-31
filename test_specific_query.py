#!/usr/bin/env python3
"""
Test specific Croatian query about EUR amounts from July 1, 2025 decisions
"""
import asyncio
import os
from pathlib import Path

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig
from src.preprocessing.extractors import DocumentExtractor
from src.preprocessing.cleaners import CroatianTextCleaner
from src.preprocessing.chunkers import DocumentChunker
from src.generation.ollama_client import OllamaClient, OllamaConfig, GenerationRequest
from src.retrieval.query_processor import CroatianQueryProcessor

async def test_specific_query():
    """Test the specific Croatian query about EUR amounts."""
    
    print("🇭🇷 Testing Croatian Query about EUR amounts from July 1, 2025 decisions")
    print("=" * 70)
    
    # Initialize components
    extractor = DocumentExtractor()
    cleaner = CroatianTextCleaner()
    chunker = DocumentChunker(chunk_size=512, overlap=50)
    
    embedding_config = EmbeddingConfig(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        cache_dir="./temp_cache"
    )
    embedding_model = CroatianEmbeddingModel(embedding_config)
    embedding_model.load_model()
    
    storage_config = StorageConfig(
        collection_name="specific_query_test",
        db_path="./temp_chroma_db"
    )
    storage = ChromaDBStorage(storage_config)
    storage.create_collection()
    
    query_processor = CroatianQueryProcessor()
    
    ollama_config = OllamaConfig()
    ollama_client = OllamaClient(ollama_config)
    
    # Process documents looking for July 2025 content
    print("📄 Processing documents for July 1, 2025 content...")
    
    doc_path = Path("./data/raw")
    documents = list(doc_path.glob("*.pdf")) + list(doc_path.glob("*.docx"))
    
    documents_list = []
    metadatas_list = []
    embeddings_list = []
    
    for doc_file in documents[:3]:  # Process first 3 documents
        try:
            print(f"📄 Processing {doc_file.name}...")
            
            text = extractor.extract_text(doc_file)
            cleaned_text = cleaner.clean_text(text)
            
            # Look for July 2025 and EUR content
            if "2025" in text and ("EUR" in text or "euro" in text.lower() or "eur" in text.lower()):
                chunks = chunker.chunk_document(cleaned_text, doc_file.name)
                print(f"   Found July 2025 content with {len(chunks)} chunks")
                
                # Store relevant chunks (those containing 2025, srpnja, EUR)
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk.content.lower()
                    if any(keyword in chunk_text for keyword in ["2025", "srpnja", "eur", "euro"]):
                        embedding = embedding_model.encode_text(chunk.content)
                        if len(embedding.shape) > 1:
                            embedding = embedding[0]
                        
                        documents_list.append(chunk.content)
                        metadatas_list.append({
                            "source": doc_file.name,
                            "chunk_id": chunk.chunk_id,
                            "language": "hr",
                            "contains_eur": "eur" in chunk_text,
                            "contains_2025": "2025" in chunk_text
                        })
                        embeddings_list.append(embedding.tolist())
                        
                        if len(documents_list) >= 10:  # Limit to 10 most relevant chunks
                            break
                    
                    if len(documents_list) >= 10:
                        break
            else:
                print(f"   No July 2025/EUR content found in {doc_file.name}")
                
        except Exception as e:
            print(f"   ❌ Error processing {doc_file.name}: {e}")
    
    if documents_list:
        storage.add_documents(documents_list, metadatas_list, embeddings_list)
        print(f"✅ Stored {len(documents_list)} relevant chunks")
    else:
        print("❌ No relevant chunks found")
        return
    
    # Test the specific query
    query = "Koje odluke su donesene 1. srpnja 2025, zanimaju nas samo iznosi u EURima?"
    
    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    
    # Process query
    processed_query = query_processor.process_query(query)
    print(f"📋 Query analysis:")
    print(f"   Type: {processed_query.query_type}")
    print(f"   Confidence: {processed_query.confidence:.3f}")
    print(f"   Keywords: {processed_query.keywords}")
    
    # Search for relevant content
    query_embedding = embedding_model.encode_text(query)
    if len(query_embedding.shape) > 1:
        query_embedding = query_embedding[0]
    
    results = storage.query_similar(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )
    
    if results and results.get('documents') and results['documents'][0]:
        print(f"\n📄 Found {len(results['documents'][0])} relevant chunks:")
        
        context = []
        for j, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0], 
            results['distances'][0],
            results['metadatas'][0]
        )):
            similarity = 1 - distance
            print(f"\n   Result {j+1} (similarity: {similarity:.3f}):")
            print(f"   Source: {metadata.get('source', 'Unknown')}")
            print(f"   Contains EUR: {metadata.get('contains_eur', False)}")
            print(f"   Contains 2025: {metadata.get('contains_2025', False)}")
            print(f"   Text: {doc[:300]}...")
            context.append(doc)
        
        # Generate comprehensive answer
        print(f"\n🤖 Generating detailed Croatian answer about EUR amounts...")
        
        if ollama_client.health_check():
            try:
                request = GenerationRequest(
                    prompt=f"Na temelju danog konteksta, odgovori na hrvatskom jeziku: {query}. Fokusiraj se na konkretne iznose u eurima i datume odluka.",
                    context=context,
                    query=query,
                    query_type="factual",
                    language="hr"
                )
                
                response = await ollama_client.generate_text_async(request)
                print(f"\n💬 Croatian Answer:")
                print(f"{response.text}")
                
            except Exception as e:
                print(f"❌ Generation error: {e}")
        
        # Also extract EUR amounts directly from context
        print(f"\n💰 Direct EUR amounts found in documents:")
        eur_amounts = []
        for doc in context:
            import re
            # Look for EUR amounts (various formats)
            eur_patterns = [
                r'(\d+(?:[.,]\d+)?)\s*(?:EUR|eur|euro|eura)',
                r'(?:EUR|eur|euro|eura)\s*(\d+(?:[.,]\d+)?)',
                r'(\d+(?:[.,]\d+)?)\s*€'
            ]
            
            for pattern in eur_patterns:
                matches = re.findall(pattern, doc)
                for match in matches:
                    eur_amounts.append(match)
        
        if eur_amounts:
            print("   Found EUR amounts:")
            for amount in set(eur_amounts):  # Remove duplicates
                print(f"   • {amount} EUR")
        else:
            print("   No specific EUR amounts found in retrieved text")
            
    else:
        print("❌ No relevant documents found for this query")

if __name__ == "__main__":
    asyncio.run(test_specific_query())