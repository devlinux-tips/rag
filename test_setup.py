#!/usr/bin/env python3
"""
Test script to verify Croatian RAG setup is working correctly.
"""
import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    try:
        import anthropic
        import chromadb
        import sentence_transformers
        import spacy

        print("✅ Core packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_croatian_model():
    """Test Croatian spaCy model."""
    try:
        import spacy

        nlp = spacy.load("hr_core_news_sm")
        doc = nlp("Zagreb je glavni grad Hrvatske.")
        print(f"✅ Croatian spaCy model loaded: {len(doc)} tokens")
        return True
    except Exception as e:
        print(f"❌ Croatian model error: {e}")
        return False


def test_embeddings():
    """Test embedding model loading."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Test with Croatian text
        text = "Ovo je test rečenica na hrvatskom jeziku."
        embedding = model.encode(text)
        print(f"✅ Embedding model working: {len(embedding)} dimensions")
        return True
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return False


def test_file_structure():
    """Test that project structure is created correctly."""
    required_dirs = [
        "src/preprocessing",
        "src/vectordb",
        "src/retrieval",
        "src/claude_api",
        "src/pipeline",
        "data/raw",
        "data/processed",
    ]

    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)

    if missing:
        print(f"❌ Missing directories: {missing}")
        return False
    else:
        print("✅ Project structure created correctly")
        return True


if __name__ == "__main__":
    print("🧪 Testing Croatian RAG setup...\n")

    tests = [
        test_file_structure,
        test_imports,
        test_croatian_model,
        test_embeddings,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 Setup complete! Ready to start building your Croatian RAG system.")
        print("\nNext steps:")
        print("1. Add your Claude API key to .env file")
        print("2. Place Croatian documents in data/raw/")
        print("3. Follow the learning path in CLAUDE.md")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
