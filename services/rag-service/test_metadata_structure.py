"""Test to understand metadata structure in retrieved documents."""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.factories import create_complete_rag_system
from src.pipeline.rag_system import RAGQuery


async def test_metadata_structure():
    """Debug metadata structure in retrieved documents."""
    print("=" * 80)
    print("DEBUGGING METADATA STRUCTURE IN RETRIEVED DOCUMENTS")
    print("=" * 80)
    print()

    # Create RAG system
    print("1. Creating RAG system...")
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
        max_results=3,  # Just 3 for debugging
    )

    print("3. Executing retrieval...")
    # Access the internal retrieval to see the raw documents
    hierarchical_results = await rag_system._document_retriever.retrieve_with_hierarchy(
        query_text, max_results=3
    )
    print(f"   ✅ Retrieved {len(hierarchical_results.documents)} documents")
    print()

    # Inspect each document's structure
    print("4. DOCUMENT METADATA STRUCTURE:")
    print("-" * 80)
    for i, doc in enumerate(hierarchical_results.documents, 1):
        print(f"\n[Document #{i}]")
        print(f"Type: {type(doc)}")
        print(f"Keys: {list(doc.keys()) if isinstance(doc, dict) else 'Not a dict'}")
        print()

        # Check metadata field
        if isinstance(doc, dict) and "metadata" in doc:
            metadata = doc["metadata"]
            print(f"  metadata type: {type(metadata)}")
            print(f"  metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'Not a dict'}")
            print()

            # Check for nn_metadata
            if isinstance(metadata, dict) and "nn_metadata" in metadata:
                nn_meta = metadata["nn_metadata"]
                print(f"  ✅ nn_metadata type: {type(nn_meta)}")
                if isinstance(nn_meta, dict):
                    print(f"  ✅ nn_metadata keys: {list(nn_meta.keys())}")
                    print(f"  ✅ nn_metadata content:")
                    print(f"      {json.dumps(nn_meta, indent=6, ensure_ascii=False)}")
                elif isinstance(nn_meta, str):
                    print(f"  ⚠️  nn_metadata is STRING (not deserialized!)")
                    print(f"      {nn_meta[:200]}...")
                else:
                    print(f"  ❌ nn_metadata is unexpected type: {nn_meta}")
            else:
                print("  ❌ No nn_metadata in metadata dict")
        else:
            print("  ❌ No metadata field in document")

        print("-" * 80)

    print()
    print("=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_metadata_structure())
