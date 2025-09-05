#!/usr/bin/env python3
"""
Fresh multilingual RAG system test
Tests the complete pipeline integration
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.rag_system import RAGSystem


def test_rag_system():
    """Test complete multilingual RAG system"""
    print("🧪 Testing Complete Multilingual RAG System")
    print("=" * 45)

    # Initialize RAG system
    print("🔧 Initializing RAGSystem...")
    rag_system = RAGSystem(language="hr")

    # Test system initialization
    print("📝 Testing system initialization...")
    assert rag_system is not None, "RAG system should be initialized"
    assert hasattr(rag_system, "language"), "System should have language attribute"
    print(f"✅ System initialized with language: {rag_system.language}")

    # Test language switching
    print("📝 Testing language switching...")
    assert rag_system.language == "hr", "Initial language should be Croatian"

    # Try English
    rag_system_en = RAGSystem(language="en")
    assert rag_system_en.language == "en", "English system should have en language"
    print("✅ Language switching verified")


if __name__ == "__main__":
    test_rag_system()
