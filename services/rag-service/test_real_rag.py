#!/usr/bin/env python3
"""
Test real RAG system with actual components on our real document.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing.chunkers import create_document_chunker
from src.preprocessing.cleaners import clean_text
from src.preprocessing.extractors import extract_document_text


async def test_real_document_processing():
    """Test real document processing with actual components."""
    print("🚀 Testing REAL RAG components with actual document")
    print("=" * 60)

    # Document path
    doc_path = Path("data/development/users/dev_user/documents/en/sample_rag_info.txt")
    if not doc_path.exists():
        print(f"❌ Document not found: {doc_path}")
        return

    print(f"📄 Processing: {doc_path}")

    try:
        # 1. Real Document Extraction
        print("\n1️⃣ DOCUMENT EXTRACTION")
        print("-" * 30)
        extracted_text = extract_document_text(doc_path)
        print(f"✅ Extracted {len(extracted_text)} characters")
        print(f"📝 Preview: {extracted_text[:200]}...")

        # 2. Real Text Cleaning
        print("\n2️⃣ TEXT CLEANING")
        print("-" * 30)
        cleaned_text = clean_text(extracted_text, language="en")
        print(f"✅ Cleaned text: {len(cleaned_text)} characters")
        print(f"📝 Preview: {cleaned_text[:200]}...")

        # 3. Real Document Chunking
        print("\n3️⃣ DOCUMENT CHUNKING")
        print("-" * 30)
        chunker = create_document_chunker(language="en")
        chunks = chunker.chunk_document(cleaned_text, str(doc_path))
        print(f"✅ Created {len(chunks)} chunks")

        # Show first few chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"📦 Chunk {i+1}: {len(chunk.content)} chars")
            print(f"   Preview: {chunk.content[:100]}...")

        print(f"\n🎉 SUCCESS! Real document processed with actual components")
        print(f"📊 Final result: {len(chunks)} chunks from {len(extracted_text)} characters")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_real_document_processing())
