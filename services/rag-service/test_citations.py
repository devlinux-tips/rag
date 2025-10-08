"""Test script to verify citation system works end-to-end."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.factories import create_complete_rag_system
from src.pipeline.rag_system import RAGQuery


async def test_citations():
    """Test that citations appear in responses."""
    print("=" * 80)
    print("TESTING NARODNE NOVINE CITATION SYSTEM")
    print("=" * 80)
    print()

    # Create RAG system for narodne-novine feature
    print("1. Creating RAG system for narodne-novine feature...")
    rag_system = create_complete_rag_system(
        language="hr",
        scope="feature",
        feature_name="narodne-novine",
    )
    await rag_system.initialize()
    print("   ✅ RAG system initialized")
    print()

    # Test query
    query_text = "Što je uredba o cijenama goriva?"
    print(f"2. Testing query: '{query_text}'")
    print()

    query = RAGQuery(
        text=query_text,
        language="hr",
        max_results=5,
    )

    print("3. Executing query...")
    response = await rag_system.query(query)
    print("   ✅ Query completed")
    print()

    # Check response
    print("4. RESPONSE ANALYSIS:")
    print("-" * 80)
    print(f"Answer length: {len(response.answer)} chars")
    print(f"Retrieved chunks: {len(response.retrieved_chunks)}")
    print(f"NN sources: {len(response.nn_sources) if response.nn_sources else 0}")
    print()

    # Check for citations in answer
    has_citations = "[1]" in response.answer or "[2]" in response.answer
    print(f"Citations in answer: {'✅ YES' if has_citations else '❌ NO'}")
    print()

    # Show answer
    print("5. GENERATED ANSWER:")
    print("-" * 80)
    print(response.answer)
    print("-" * 80)
    print()

    # Show sources
    if response.nn_sources:
        print("6. NN SOURCES:")
        print("-" * 80)
        for i, source in enumerate(response.nn_sources, 1):
            print(f"[{i}] {source.get('title', 'No title')}")
            print(f"    Issue: {source.get('issue', 'N/A')}")
            if source.get('eli'):
                print(f"    ELI: {source['eli']}")
            print()
    else:
        print("6. NN SOURCES: None found")

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_citations())
