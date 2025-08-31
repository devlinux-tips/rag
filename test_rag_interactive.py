#!/usr/bin/env python3
"""
Test the RAG system with simulated interactions
"""
import asyncio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig
from src.preprocessing.extractors import DocumentExtractor
from src.preprocessing.cleaners import CroatianTextCleaner
from src.preprocessing.chunkers import DocumentChunker
from src.generation.ollama_client import OllamaClient, OllamaConfig, GenerationRequest
from src.retrieval.query_processor import CroatianQueryProcessor

class CroatianRAG:
    def __init__(self):
        self.embedding_model = None
        self.storage = None
        self.query_processor = None
        self.ollama_client = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize all RAG components."""
        print("🔧 Initializing Croatian RAG System...")
        
        # Embedding model
        embedding_config = EmbeddingConfig(
            model_name="distiluse-base-multilingual-cased",
            device="cpu",
            cache_dir="./temp_cache"
        )
        self.embedding_model = CroatianEmbeddingModel(embedding_config)
        self.embedding_model.load_model()
        print("✅ Embedding model loaded")
        
        # Vector storage
        storage_config = StorageConfig(
            collection_name="croatian_rag",
            db_path="./data/chromadb"
        )
        self.storage = ChromaDBStorage(storage_config)
        self.storage.create_collection()
        
        # Check if we have documents
        info = self.storage.get_collection_info()
        doc_count = info.get('count', 0)
        print(f"📊 Vector database: {doc_count} document chunks")
        
        # Query processor
        self.query_processor = CroatianQueryProcessor()
        
        # Ollama client
        ollama_config = OllamaConfig()  # Uses qwen2.5:32b by default
        self.ollama_client = OllamaClient(ollama_config)
        
        # Check Ollama status
        if self.ollama_client.health_check():
            print(f"✅ Ollama ready with {ollama_config.model}")
        else:
            print(f"⚠️  Ollama not available - answers will be context-only")
        
        self.initialized = True
        print("🎯 Croatian RAG System ready!\n")
    
    async def process_documents(self, doc_dir="./data/raw"):
        """Process documents and add to vector database."""
        print(f"📄 Processing documents from {doc_dir}...")
        
        # Initialize components if not done already
        if not self.initialized:
            await self.initialize()
        
        from pathlib import Path
        extractor = DocumentExtractor()
        cleaner = CroatianTextCleaner()
        chunker = DocumentChunker(chunk_size=512, overlap=50)
        
        doc_path = Path(doc_dir)
        if not doc_path.exists():
            print(f"❌ Directory {doc_dir} not found!")
            return
        
        documents = list(doc_path.glob("*.pdf")) + list(doc_path.glob("*.docx")) + list(doc_path.glob("*.txt"))
        
        if not documents:
            print("❌ No documents found!")
            return
        
        print(f"📁 Found {len(documents)} documents")
        
        total_chunks = 0
        documents_list = []
        metadatas_list = []
        embeddings_list = []
        
        for doc_file in documents:
            try:
                print(f"   Processing {doc_file.name}...")
                
                # Extract and clean text
                text = extractor.extract_text(doc_file)
                cleaned_text = cleaner.clean_text(text)
                
                # Create chunks
                chunks = chunker.chunk_document(cleaned_text, doc_file.name)
                
                # Process chunks (limit per document)
                for chunk in chunks[:20]:  # First 20 chunks per document
                    embedding = self.embedding_model.encode_text(chunk.content)
                    if len(embedding.shape) > 1:
                        embedding = embedding[0]
                    
                    documents_list.append(chunk.content)
                    metadatas_list.append({
                        "source": doc_file.name,
                        "chunk_id": chunk.chunk_id,
                        "language": "hr"
                    })
                    embeddings_list.append(embedding.tolist())
                
                print(f"   ✅ Created {len(chunks)} chunks")
                total_chunks += len(chunks[:20])
                
            except Exception as e:
                print(f"   ❌ Error processing {doc_file.name}: {e}")
        
        if documents_list:
            self.storage.add_documents(documents_list, metadatas_list, embeddings_list)
            print(f"✅ Stored {len(documents_list)} chunks total")
        
        print(f"📊 Processing complete!\n")
    
    async def query(self, question):
        """Query the RAG system."""
        if not self.initialized:
            await self.initialize()
        
        print(f"🔍 Searching: '{question}'")
        
        # Process query
        processed_query = self.query_processor.process_query(question)
        print(f"📋 Query type: {processed_query.query_type}")
        
        # Search for relevant content
        query_embedding = self.embedding_model.encode_text(question)
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]
        
        results = self.storage.query_similar(
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            print("❌ No relevant documents found")
            return
        
        print(f"📄 Found {len(results['documents'][0])} relevant chunks")
        
        context = []
        for j, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0], 
            results['distances'][0],
            results['metadatas'][0]
        )):
            similarity = 1 - distance
            source = metadata.get('source', 'Unknown')
            print(f"   {j+1}. {source} (similarity: {similarity:.3f})")
            context.append(doc)
        
        # Generate answer
        print(f"\n🤖 Generating answer...")
        
        if self.ollama_client.health_check():
            try:
                # Improved prompt for Croatian
                prompt = f"""Na temelju sljedećih hrvatskih dokumenata odgovori na pitanje na hrvatskom jeziku.

DOKUMENTI:
{chr(10).join([f"Dokument {i+1}: {doc[:500]}..." for i, doc in enumerate(context)])}

PITANJE: {question}

VAŽNO: Koristi SAMO informacije iz dokumenata. Izvuci konkretne brojeve, datume i iznose ako postoje.

ODGOVOR:"""

                request = GenerationRequest(
                    prompt=prompt,
                    context=context,
                    query=question,
                    query_type="factual",
                    language="hr"
                )
                
                response = await self.ollama_client.generate_text_async(request)
                print(f"💬 {response.text}")
                
            except Exception as e:
                print(f"❌ Generation error: {e}")
                print("📝 Context summary:")
                for i, chunk in enumerate(context[:2]):
                    print(f"   {i+1}. {chunk[:200]}...")
        else:
            print("⚠️ Ollama not available. Context preview:")
            for i, chunk in enumerate(context[:2]):
                print(f"   {i+1}. {chunk[:200]}...")
        
        print("-" * 60)

async def test_rag():
    """Test the RAG system with Croatian queries."""
    print("🇭🇷 Testing Croatian RAG System")
    print("=" * 40)
    
    rag = CroatianRAG()
    
    # Test 1: Process documents first
    print("STEP 1: Process documents")
    await rag.process_documents()
    
    # Test 2: Query the system
    print("\nSTEP 2: Test query")
    question = "Koje odluke su donesene 1. srpnja 2025, zanimaju nas samo iznosi u EURima?"
    await rag.query(question)
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_rag())