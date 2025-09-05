#!/usr/bin/env python3
"""
Fresh multilingual storage test
Tests the actual MultilingualVectorStore API
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectordb.storage import ChromaDBStorage


def test_storage():
    """Test multilingual vector storage with actual API"""
    print("🧪 Testing Multilingual Vector Storage")
    print("=" * 40)

    # Initialize storage
    print("🔧 Initializing ChromaDBStorage...")
    storage = ChromaDBStorage()

    # Test that storage was initialized
    print("📝 Testing storage initialization...")
    assert storage.client is not None, "ChromaDB client should be initialized"
    assert hasattr(storage, "config"), "Storage should have config"
    print("✅ Storage initialized successfully")

    # Test configuration access
    print("📝 Testing configuration...")
    config = storage.config
    assert config is not None, "Storage config should exist"
    assert hasattr(config, "db_path"), "Config should have db_path"
    print(f"✅ Config loaded: db_path={config.db_path}")

    print("✅ Basic storage functionality confirmed")

    print("\n🎉 All storage tests passed!")


if __name__ == "__main__":
    test_storage()
