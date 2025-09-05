#!/usr/bin/env python3
"""
Fresh multilingual query processing test
Tests the actual MultilingualQueryProcessor API
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.query_processor import MultilingualQueryProcessor


def test_query_processing():
    """Test multilingual query processing with actual API"""
    print("🧪 Testing Multilingual Query Processing")
    print("=" * 40)

    # Test Croatian query processing
    print("� Testing Croatian query...")
    hr_processor = MultilingualQueryProcessor(language="hr")
    hr_query = "Koliko košta stanovanje u Zagrebu?"
    hr_processed = hr_processor.process_query(hr_query)
    assert hr_processed is not None, "Croatian query should be processed"
    assert hasattr(hr_processed, "keywords"), "Should have keywords attribute"
    print(f"✅ Croatian query processed with {len(hr_processed.keywords)} keywords")

    # Test English query processing
    print("📝 Testing English query...")
    en_processor = MultilingualQueryProcessor(language="en")
    en_query = "What is the cost of living in London?"
    en_processed = en_processor.process_query(en_query)
    assert en_processed is not None, "English query should be processed"
    assert hasattr(en_processed, "keywords"), "Should have keywords attribute"
    print(f"✅ English query processed with {len(en_processed.keywords)} keywords")

    # Test empty query
    print("📝 Testing empty query...")
    empty_processed = hr_processor.process_query("")
    assert empty_processed is not None, "Empty query should return result object"
    print("✅ Empty query handled correctly")

    print("\n🎉 All query processing tests passed!")


if __name__ == "__main__":
    test_query_processing()
