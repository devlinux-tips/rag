#!/usr/bin/env python3
"""
Simple multilingual chunking test.
Tests the actual chunk_document function with Croatian and English.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.chunkers import chunk_document


def test_multilingual_chunking():
    """Test chunking with both Croatian and English."""

    # Croatian text
    croatian_text = """
    Zagreb je glavni grad Republike Hrvatske. Grad ima bogatu povijest koja seže u rimsko doba.
    Zagreb se sastoji od dva dijela: Gornji grad i Donji grad. Gornji grad je povijesni dio grada.
    U Gornjem gradu nalaze se važne institucije poput Sabora i Vlade. Donji grad je moderniji dio.
    Zagreb ima oko 800.000 stanovnika. To ga čini najvećim gradom u hrvatskoj.
    """

    # English text
    english_text = """
    London is the capital city of England and the United Kingdom. The city has a rich history dating back to Roman times.
    London consists of multiple boroughs and districts. The City of London is the historic core of the metropolis.
    Important institutions like Parliament and government offices are located in Westminster. The East End is known for its diversity.
    London has approximately 9 million inhabitants. This makes it the largest city in the United Kingdom.
    """

    print("🧪 Testing Multilingual Chunking")
    print("=" * 40)

    # Test Croatian chunking
    print("📝 Testing Croatian chunking...")
    hr_chunks = chunk_document(
        croatian_text, "test_croatian.txt", strategy="sentence", language="hr"
    )
    assert len(hr_chunks) > 0, "Croatian chunking should produce chunks"
    assert any("Zagreb" in chunk.content for chunk in hr_chunks), "Should contain Croatian content"
    print(f"✅ Croatian: {len(hr_chunks)} chunks created")

    # Test English chunking
    print("📝 Testing English chunking...")
    en_chunks = chunk_document(english_text, "test_english.txt", strategy="sentence", language="en")
    assert len(en_chunks) > 0, "English chunking should produce chunks"
    assert any("London" in chunk.content for chunk in en_chunks), "Should contain English content"
    print(f"✅ English: {len(en_chunks)} chunks created")

    # Test empty text
    print("📝 Testing empty text...")
    empty_chunks = chunk_document("", "empty.txt", strategy="sentence", language="hr")
    assert len(empty_chunks) == 0, "Empty text should produce no chunks"
    print("✅ Empty text handled correctly")

    print("\n🎉 All chunking tests passed!")


if __name__ == "__main__":
    try:
        test_multilingual_chunking()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
