#!/usr/bin/env python3
"""
Simple RAG query - just ask the system directly
"""
import asyncio
import os
from pathlib import Path

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.generation.ollama_client import GenerationRequest, OllamaClient, OllamaConfig
from src.preprocessing.chunkers import DocumentChunker
from src.preprocessing.cleaners import CroatianTextCleaner
from src.preprocessing.extractors import DocumentExtractor
from src.retrieval.query_processor import CroatianQueryProcessor
from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig
from src.vectordb.storage import ChromaDBStorage, StorageConfig


async def setup_rag_system():
    """Set up RAG system by processing all documents normally."""

    print("🔧 Setting up Croatian RAG System...")

    # Initialize components
    extractor = DocumentExtractor()
    cleaner = CroatianTextCleaner()
    chunker = DocumentChunker(chunk_size=512, overlap=50)

    embedding_config = EmbeddingConfig(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        cache_dir="./temp_cache",
    )
    embedding_model = CroatianEmbeddingModel(embedding_config)
    embedding_model.load_model()

    storage_config = StorageConfig(collection_name="normal_rag_system", db_path="./temp_chroma_db")
    storage = ChromaDBStorage(storage_config)
    storage.create_collection()

    query_processor = CroatianQueryProcessor()

    ollama_config = OllamaConfig()
    ollama_client = OllamaClient(ollama_config)

    # Process documents normally (no special filtering)
    print("📄 Processing all documents normally...")

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

            # Create chunks from ALL content (no filtering)
            chunks = chunker.chunk_document(cleaned_text, doc_file.name)
            print(f"   Created {len(chunks)} chunks")

            # Store chunks (limit to reasonable number per document)
            for i, chunk in enumerate(chunks[:20]):  # First 20 chunks per document
                embedding = embedding_model.encode_text(chunk.content)
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

        except Exception as e:
            print(f"   ❌ Error processing {doc_file.name}: {e}")

    if documents_list:
        storage.add_documents(documents_list, metadatas_list, embeddings_list)
        print(f"✅ Stored {len(documents_list)} chunks total")

    return {
        "embedding_model": embedding_model,
        "storage": storage,
        "query_processor": query_processor,
        "ollama_client": ollama_client,
    }


async def ask_question(components, question):
    """Ask any question to the RAG system."""

    print(f"\n🇭🇷 Question: '{question}'")
    print("-" * 50)

    # Process query
    processed_query = components["query_processor"].process_query(question)
    print(f"📋 Query type: {processed_query.query_type}")

    # Search for relevant content
    query_embedding = components["embedding_model"].encode_text(question)
    if len(query_embedding.shape) > 1:
        query_embedding = query_embedding[0]

    results = components["storage"].query_similar(
        query_embeddings=[query_embedding.tolist()], n_results=5
    )

    if results and results.get("documents") and results["documents"][0]:
        print(f"📄 Found {len(results['documents'][0])} relevant chunks")

        context = []
        for j, (doc, distance, metadata) in enumerate(
            zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ):
            similarity = 1 - distance
            print(f"   {j+1}. Source: {metadata.get('source')} (similarity: {similarity:.3f})")
            context.append(doc)

        # Generate answer
        print(f"\n🤖 Generating answer...")

        if components["ollama_client"].health_check():
            try:
                request = GenerationRequest(
                    prompt=question,
                    context=context,
                    query=question,
                    query_type="factual",
                    language="hr",
                )

                response = await components["ollama_client"].generate_text_async(request)
                print(f"\n💬 Answer:")
                print(f"{response.text}")

            except Exception as e:
                print(f"❌ Generation error: {e}")
                print("📝 Context preview:")
                for i, chunk in enumerate(context[:2]):
                    print(f"   {i+1}. {chunk[:200]}...")
        else:
            print("⚠️ Ollama not available. Context preview:")
            for i, chunk in enumerate(context[:2]):
                print(f"   {i+1}. {chunk[:200]}...")
    else:
        print("❌ No relevant documents found")


async def main():
    """Main function - set up system and ask questions."""

    # Set up RAG system
    components = await setup_rag_system()

    # Ask your specific question
    await ask_question(
        components,
        "Koje odluke su donesene 1. srpnja 2025, zanimaju nas samo iznosi u EURima?",
    )

    # Ask other questions to show it works generally
    print("\n" + "=" * 60)
    await ask_question(components, "Kolika je najniža mirovina?")

    print("\n" + "=" * 60)
    await ask_question(components, "Što se odlučilo o nekretninama?")


if __name__ == "__main__":
    asyncio.run(main())
