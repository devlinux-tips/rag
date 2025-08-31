#!/usr/bin/env python3
"""
Interactive Croatian RAG System Test - Simplified Version
"""
import asyncio
import os
from pathlib import Path
from typing import List

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from src.pipeline.config import RAGConfig
from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig
from src.preprocessing.extractors import DocumentExtractor
from src.preprocessing.cleaners import CroatianTextCleaner
from src.preprocessing.chunkers import DocumentChunker
from src.generation.ollama_client import OllamaClient, OllamaConfig, GenerationRequest
from src.retrieval.query_processor import CroatianQueryProcessor

async def create_simple_rag_system():
    """Create a simple RAG system for testing."""
    
    print("🔧 Setting up Croatian RAG System...")
    
    # Initialize components
    extractor = DocumentExtractor()
    cleaner = CroatianTextCleaner()
    chunker = DocumentChunker(chunk_size=512, overlap=50)
    
    # Embedding model
    embedding_config = EmbeddingConfig(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        cache_dir="./temp_cache"
    )
    embedding_model = CroatianEmbeddingModel(embedding_config)
    embedding_model.load_model()
    print("✅ Embedding model loaded")
    
    # Vector storage
    storage_config = StorageConfig(
        collection_name="croatian_test",
        db_path="./temp_chroma_db"
    )
    storage = ChromaDBStorage(storage_config)
    storage.create_collection()
    print("✅ Vector storage ready")
    
    # Query processor
    query_processor = CroatianQueryProcessor()
    
    # Ollama client
    ollama_config = OllamaConfig()
    ollama_client = OllamaClient(ollama_config)
    
    return {
        'extractor': extractor,
        'cleaner': cleaner,
        'chunker': chunker,
        'embedding_model': embedding_model,
        'storage': storage,
        'query_processor': query_processor,
        'ollama_client': ollama_client
    }

async def process_documents(components, doc_dir: str = "./data/raw"):
    """Process documents and add to vector database."""
    
    print(f"\n📄 Processing documents from {doc_dir}...")
    
    doc_path = Path(doc_dir)
    if not doc_path.exists():
        print("❌ Document directory not found!")
        return
    
    documents = list(doc_path.glob("*.pdf")) + list(doc_path.glob("*.docx")) + list(doc_path.glob("*.txt"))
    
    if not documents:
        print("❌ No documents found!")
        return
    
    print(f"📁 Found {len(documents)} documents")
    
    total_chunks = 0
    
    for doc_file in documents[:2]:  # Process first 2 documents for speed
        try:
            print(f"📄 Processing {doc_file.name}...")
            
            # Extract text
            text = components['extractor'].extract_text(doc_file)
            if not text.strip():
                print(f"   ⚠️ No text extracted from {doc_file.name}")
                continue
            
            # Clean text
            cleaned_text = components['cleaner'].clean_text(text)
            
            # Create chunks
            chunks = components['chunker'].chunk_document(cleaned_text, doc_file.name)
            print(f"   ✅ Created {len(chunks)} chunks")
            
            # Generate embeddings and store
            documents_list = []
            metadatas_list = []
            embeddings_list = []
            
            for i, chunk in enumerate(chunks[:5]):  # Only first 5 chunks per document
                embedding = components['embedding_model'].encode_text(chunk.content)
                if len(embedding.shape) > 1:
                    embedding = embedding[0]
                
                documents_list.append(chunk.content)
                metadatas_list.append({
                    "source": doc_file.name,
                    "chunk_id": chunk.chunk_id,
                    "language": "hr"
                })
                embeddings_list.append(embedding.tolist())
            
            if documents_list:
                components['storage'].add_documents(documents_list, metadatas_list, embeddings_list)
                print(f"   ✅ Stored {len(documents_list)} chunks")
                total_chunks += len(documents_list)
        
        except Exception as e:
            print(f"   ❌ Error processing {doc_file.name}: {e}")
    
    print(f"\n📊 Processing complete! Total chunks stored: {total_chunks}")

async def interactive_query(components):
    """Run interactive queries."""
    
    print("\n🎯 Interactive Croatian Query System")
    print("=" * 50)
    print("Enter Croatian questions about your documents.")
    print("Type 'exit' to quit, 'help' for examples.")
    print()
    
    # Example queries
    examples = [
        "Što je Zagreb?",
        "Koliko stanovnika ima Hrvatska?",
        "Koji su glavni gradovi?",
        "O čemu govore dokumenti?"
    ]
    
    while True:
        try:
            query = input("🇭🇷 Pitanje: ").strip()
            
            if query.lower() in ['exit', 'quit', 'izlaz']:
                print("👋 Do viđenja!")
                break
            
            if query.lower() == 'help':
                print("💡 Primjeri pitanja:")
                for example in examples:
                    print(f"   • {example}")
                continue
            
            if not query:
                continue
            
            print(f"\n🔍 Tražim: '{query}'...")
            
            # Process query
            processed_query = components['query_processor'].process_query(query)
            print(f"📋 Tip pitanja: {processed_query.query_type}")
            
            # Search for relevant documents
            query_embedding = components['embedding_model'].encode_text(query)
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding[0]
            
            results = components['storage'].query_similar(
                query_embeddings=[query_embedding.tolist()],
                n_results=3
            )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                print("❌ Nema relevantnih dokumenata.")
                continue
            
            # Display search results
            print(f"📄 Pronašao {len(results['documents'][0])} relevantnih dokumenata:")
            context = []
            
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                similarity = 1 - distance
                print(f"   {i+1}. (sličnost: {similarity:.3f}) {doc[:100]}...")
                context.append(doc)
            
            # Generate answer with Ollama
            print("\n🤖 Generiram odgovor...")
            
            if components['ollama_client'].health_check():
                request = GenerationRequest(
                    prompt=query,
                    context=context,
                    query=query,
                    query_type="factual",
                    language="hr"
                )
                
                response = await components['ollama_client'].generate_text_async(request)
                print(f"\n💬 Odgovor:")
                print(f"{response.text}")
            else:
                print("⚠️ Ollama servis nije dostupan. Evo konteksta:")
                for i, doc in enumerate(context):
                    print(f"{i+1}. {doc[:200]}...")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Do viđenja!")
            break
        except Exception as e:
            print(f"❌ Greška: {e}")
            continue

async def main():
    """Main function."""
    
    print("🚀 Croatian RAG Interactive Test")
    print("=" * 50)
    
    try:
        # Set up system
        components = await create_simple_rag_system()
        
        # Process documents
        await process_documents(components)
        
        # Start interactive session
        await interactive_query(components)
        
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))