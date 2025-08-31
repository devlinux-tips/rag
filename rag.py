#!/usr/bin/env python3
"""
Croatian RAG System - Interactive Command Line Interface
Usage: python rag.py
"""
import asyncio
import os
import sys
from pathlib import Path

from src.generation.ollama_client import GenerationRequest, OllamaClient, OllamaConfig
from src.preprocessing.chunkers import DocumentChunker
from src.preprocessing.cleaners import CroatianTextCleaner
from src.preprocessing.extractors import DocumentExtractor
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_processor import CroatianQueryProcessor
from src.retrieval.reranker import MultilingualReranker
from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig

# Force CPU usage
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


class CroatianRAG:
    def __init__(self):
        self.embedding_model = None
        self.storage = None
        self.query_processor = None
        self.ollama_client = None
        self.hybrid_retriever = None
        self.reranker = None
        self.initialized = False

    async def initialize(self):
        """Initialize all RAG components."""
        print("🔧 Initializing Croatian RAG System...")

        # Embedding model
        embedding_config = EmbeddingConfig(
            # model_name="LaBSE",
            device="cuda",
            cache_dir="./temp_cache",
        )
        print(f"Embedding Config: {embedding_config}")
        self.embedding_model = CroatianEmbeddingModel(embedding_config)
        self.embedding_model.load_model()
        print("✅ Embedding model loaded")

        # Vector storage
        storage_config = StorageConfig(collection_name="croatian_rag", db_path="./data/chromadb")
        self.storage = ChromaDBStorage(storage_config)
        self.storage.create_collection()

        # Check if we have documents
        info = self.storage.get_collection_info()
        doc_count = info.get("count", 0)
        print(f"📊 Vector database: {doc_count} document chunks")

        # Query processor
        self.query_processor = CroatianQueryProcessor()

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(dense_weight=0.7, sparse_weight=0.3)

        # Reranker (lightweight, loads on first use)
        self.reranker = MultilingualReranker(device="cuda", batch_size=4)

        # Ollama client
        ollama_config = OllamaConfig()  # Uses qwen2.5:7b-instruct by default
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

        extractor = DocumentExtractor()
        cleaner = CroatianTextCleaner()
        chunker = DocumentChunker(chunk_size=512, overlap=50)

        doc_path = Path(doc_dir)
        if not doc_path.exists():
            print(f"❌ Directory {doc_dir} not found!")
            return

        documents = (
            list(doc_path.glob("*.pdf"))
            + list(doc_path.glob("*.docx"))
            + list(doc_path.glob("*.txt"))
        )

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
                    metadatas_list.append(
                        {
                            "source": doc_file.name,
                            "chunk_id": chunk.chunk_id,
                            "language": "hr",
                        }
                    )
                    embeddings_list.append(embedding.tolist())

                print(f"   ✅ Created {len(chunks)} chunks")
                total_chunks += len(chunks[:20])

            except Exception as e:
                print(f"   ❌ Error processing {doc_file.name}: {e}")

        if documents_list:
            self.storage.add_documents(documents_list, metadatas_list, embeddings_list)
            print(f"✅ Stored {len(documents_list)} chunks total")

            # Index documents for hybrid retrieval
            print("🔧 Indexing documents for hybrid retrieval...")
            self.hybrid_retriever.index_documents(documents_list, metadatas_list)
            print("✅ Hybrid retriever indexed")

        print(f"📊 Processing complete!\n")

    async def query(self, question):
        """Query the RAG system."""
        if not self.initialized:
            await self.initialize()

        print(f"🔍 Searching: '{question}'")

        # Process query
        processed_query = self.query_processor.process_query(question)
        print(f"📋 Query type: {processed_query.query_type}")

        # Search for relevant content using hybrid retrieval
        print("🔍 Dense search...")
        query_embedding = self.embedding_model.encode_text(question)
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]

        dense_results = self.storage.query_similar(
            query_embeddings=[query_embedding.tolist()],
            n_results=20,  # Get more for reranking
        )

        if (
            not dense_results
            or not dense_results.get("documents")
            or not dense_results["documents"][0]
        ):
            print("❌ No relevant documents found")
            return

        # Prepare dense results for hybrid retrieval
        dense_formatted = []
        for doc, distance, metadata in zip(
            dense_results["documents"][0],
            dense_results["distances"][0],
            dense_results["metadatas"][0],
        ):
            dense_formatted.append({"content": doc, "distance": distance, "metadata": metadata})

        # Apply hybrid retrieval (dense + BM25)
        print("⚡ Hybrid retrieval (dense + BM25)...")
        hybrid_results = self.hybrid_retriever.search(question, dense_formatted, n_results=10)

        # Apply reranking
        print("🎯 Reranking results...")
        documents_to_rerank = [r.content for r in hybrid_results]
        metadatas_to_rerank = [r.metadata for r in hybrid_results]

        # Load reranker if not loaded (only on first use)
        if not self.reranker.is_loaded:
            self.reranker.load_model()

        reranked_results = self.reranker.rerank(
            query=question,
            documents=documents_to_rerank,
            metadatas=metadatas_to_rerank,
            top_k=5,
        )

        print(f"📄 Final results: {len(reranked_results)} chunks")

        context = []
        for j, result in enumerate(reranked_results):
            source = result.metadata.get("source", "Unknown")
            print(f"   {j+1}. {source} (score: {result.score:.3f})")
            context.append(result.content)

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
                    language="hr",
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


async def main():
    """Main interactive loop."""
    rag = CroatianRAG()

    print("🇭🇷 Croatian RAG System")
    print("=" * 40)
    print("Commands:")
    print("  process - Process documents from data/raw/")
    print("  status  - Show system status")
    print("  help    - Show example questions")
    print("  exit    - Quit")
    print("  Or just ask a Croatian question!")
    print()

    while True:
        try:
            user_input = input("RAG> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "izlaz"]:
                print("👋 Do viđenja!")
                break

            elif user_input.lower() == "process":
                await rag.process_documents()

            elif user_input.lower() == "status":
                if not rag.initialized:
                    await rag.initialize()
                info = rag.storage.get_collection_info()
                print(f"📊 Documents in database: {info.get('count', 0)}")
                print(f"🤖 Model: qwen2.5:7b-instruct")
                print(f"🔤 Embeddings: distiluse-base-multilingual-cased")

            elif user_input.lower() == "help":
                print("💡 Example Croatian questions:")
                print("  • Koje odluke su donesene 1. srpnja 2025?")
                print("  • Kolika je najniža mirovina?")
                print("  • O čemu govore dokumenti?")
                print("  • Koji su glavni iznosi u eurima?")

            else:
                # Treat as query
                await rag.query(user_input)

        except KeyboardInterrupt:
            print("\n👋 Do viđenja!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
