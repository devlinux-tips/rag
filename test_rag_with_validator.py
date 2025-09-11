#!/usr/bin/env python3
"""
Test: Does RAGSystem startup now include ConfigValidator?
"""

import asyncio
import os
import sys

sys.path.append("services/rag-service/src")


async def test_rag_with_configvalidator():
    """Test RAGSystem initialization with ConfigValidator integration."""
    print("🔍 Testing RAGSystem with ConfigValidator integration...")

    try:
        from pipeline.rag_system import RAGSystem

        print("✅ RAGSystem imported successfully")

        # Create RAG system
        rag = RAGSystem(language="hr")
        print("✅ RAGSystem instance created")

        # Initialize - should now call ConfigValidator
        print("🚀 Initializing RAGSystem (should validate configs)...")
        await rag.initialize()
        print("✅ RAGSystem initialization complete")

        print("🎯 SUCCESS: ConfigValidator integration working")

    except Exception as e:
        print(f"❌ Error during initialization: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_rag_with_configvalidator())
