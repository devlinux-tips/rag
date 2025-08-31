#!/usr/bin/env python3
"""
Test script for complete Croatian RAG system.
This script will initialize the system, process documents, and allow interactive querying.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from src.pipeline.config import RAGConfig
from src.pipeline.rag_system import CroatianRAGSystem, RAGQuery, create_rag_system

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_complete_rag_system():
    """Test the complete RAG system with real Croatian documents."""

    print("🚀 Testing Complete Croatian RAG System")
    print("=" * 60)

    # Create optimized config for testing
    config = RAGConfig()
    config.processing.max_chunk_size = 512  # Good balance for testing
    config.processing.chunk_overlap = 50
    config.retrieval.default_k = 3  # Fewer results for cleaner output
    config.ollama.timeout = 60.0  # Longer timeout for generation

    print(f"📋 Configuration:")
    print(f"  Embedding Model: {config.embedding.model_name}")
    print(f"  LLM Model: {config.ollama.model}")
    print(f"  Max Chunk Size: {config.processing.max_chunk_size}")
    print(f"  Default K: {config.retrieval.default_k}")

    try:
        # Initialize system
        print(f"\n🔧 Initializing Croatian RAG System...")
        system = CroatianRAGSystem(config)
        await system.initialize()

        # Health check
        print(f"\n🏥 Performing Health Check...")
        health = await system.health_check()
        print(f"System Status: {health.get('system_status', 'unknown').upper()}")

        if health.get("system_status") != "healthy":
            print("⚠️  System not fully healthy, but continuing with test...")
            for comp, status in health.get("components", {}).items():
                if status.get("status") != "healthy":
                    print(f"  {comp}: {status.get('details', 'No details')}")

        # Process documents from data/raw
        print(f"\n📄 Processing Documents from data/raw/...")
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            print(f"❌ Directory {raw_dir} does not exist!")
            return

        # Get all document files
        doc_extensions = {".pdf", ".docx", ".txt"}
        doc_files = [
            str(f) for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in doc_extensions
        ]

        if not doc_files:
            print(f"❌ No documents found in {raw_dir}")
            return

        print(f"📁 Found {len(doc_files)} documents:")
        for doc in doc_files:
            print(f"  • {Path(doc).name}")

        # Process documents
        start_time = time.time()
        result = await system.add_documents(doc_files, batch_size=2)
        processing_time = time.time() - start_time

        print(f"\n✅ Document Processing Results:")
        print(f"  ✅ Processed: {result['processed_documents']} documents")
        print(f"  ❌ Failed: {result['failed_documents']} documents")
        print(f"  📄 Total Chunks: {result['total_chunks']} chunks")
        print(f"  ⏱️  Processing Time: {result['processing_time']:.2f}s")
        print(f"  📊 Rate: {result['documents_per_second']:.1f} docs/sec")

        if result["failed_documents"] > 0:
            print(f"⚠️  Some documents failed to process. Check logs for details.")

        # Interactive query loop
        print(f"\n🔍 Ready for Interactive Queries!")
        print(f"=" * 60)
        print(f"Enter Croatian queries to search the processed documents.")
        print(f"Type 'quit', 'exit', or 'kraj' to end the session.")
        print(f"Type 'stats' to see system statistics.")
        print(f"Type 'health' to check system health.")
        print(f"=" * 60)

        query_count = 0

        while True:
            try:
                # Get user input
                user_input = input(f"\n🇭🇷 Croatian Query > ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "kraj", "q"]:
                    print(f"👋 Ending session. Processed {query_count} queries total.")
                    break

                if user_input.lower() == "stats":
                    stats = await system.get_system_stats()
                    print(f"\n📊 System Statistics:")
                    print(f"  Documents: {stats.get('documents', 0)}")
                    print(f"  Queries: {stats.get('queries', 0)}")
                    print(f"  Chunks: {stats.get('chunks', 0)}")
                    continue

                if user_input.lower() == "health":
                    health = await system.health_check()
                    print(f"\n🏥 System Health: {health.get('system_status', 'unknown').upper()}")
                    for comp, status in health.get("components", {}).items():
                        status_icon = {
                            "healthy": "✅",
                            "degraded": "⚠️",
                            "unhealthy": "❌",
                        }.get(status.get("status"), "❓")
                        print(f"  {status_icon} {comp}: {status.get('status', 'unknown')}")
                    continue

                # Process query
                query_count += 1
                print(f"\n🔄 Processing Query {query_count}: '{user_input}'")

                query = RAGQuery(
                    text=user_input,
                    query_id=f"interactive-{query_count:03d}",
                    max_results=3,
                )

                start_time = time.time()
                response = await system.query(query, return_sources=True, return_debug_info=False)
                query_time = time.time() - start_time

                # Display results
                print(f"\n💬 Response:")
                print(f"{'─' * 80}")
                print(response.answer)
                print(f"{'─' * 80}")

                print(f"\n📊 Query Metrics:")
                print(f"  ⏱️  Total Time: {response.total_time:.2f}s")
                print(f"  🔍 Retrieval: {response.retrieval_time:.2f}s")
                print(f"  🤖 Generation: {response.generation_time:.2f}s")
                print(f"  🎯 Confidence: {response.confidence:.3f}")

                print(f"\n📄 Retrieved Information:")
                print(f"  📚 Sources: {len(response.sources)}")
                print(f"  📄 Chunks: {len(response.retrieved_chunks)}")

                if response.sources:
                    print(f"  📋 Source Files:")
                    for source in response.sources:
                        source_name = Path(source).name if source != "Unknown" else "Unknown"
                        print(f"    • {source_name}")

                # Show retrieved chunks summary
                if response.retrieved_chunks:
                    print(f"  🔍 Top Retrieved Chunks:")
                    for i, chunk in enumerate(response.retrieved_chunks[:2], 1):
                        chunk_preview = (
                            chunk["content"][:100] + "..."
                            if len(chunk["content"]) > 100
                            else chunk["content"]
                        )
                        similarity = chunk.get("similarity_score", 0)
                        print(f"    {i}. Score: {similarity:.3f} | {chunk_preview}")

                # Quality indicators
                quality_indicators = []
                if response.confidence >= 0.8:
                    quality_indicators.append("🟢 High Confidence")
                elif response.confidence >= 0.6:
                    quality_indicators.append("🟡 Medium Confidence")
                else:
                    quality_indicators.append("🔴 Low Confidence")

                if len(response.retrieved_chunks) >= 2:
                    quality_indicators.append("🟢 Good Context")

                # Check for Croatian diacritics
                if any(char in response.answer for char in "čćšžđČĆŠŽĐ"):
                    quality_indicators.append("🟢 Croatian Diacritics")

                print(f"  📊 Quality: {', '.join(quality_indicators)}")

            except KeyboardInterrupt:
                print(f"\n\n👋 Session interrupted. Processed {query_count} queries.")
                break
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                logging.error(f"Query processing error: {e}")

        # Final statistics
        print(f"\n📊 Final Session Statistics:")
        final_stats = await system.get_system_stats()
        print(f"  Total Documents: {final_stats.get('documents', 0)}")
        print(f"  Total Queries: {final_stats.get('queries', 0)}")
        print(f"  Total Chunks: {final_stats.get('chunks', 0)}")

        # Clean shutdown
        print(f"\n🔄 Shutting down system...")
        await system.close()
        print(f"✅ System shutdown complete")

    except Exception as e:
        print(f"❌ System error: {e}")
        logging.error(f"System error: {e}")
        raise


def main():
    """Main function to run the test."""
    try:
        asyncio.run(test_complete_rag_system())
    except KeyboardInterrupt:
        print(f"\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
