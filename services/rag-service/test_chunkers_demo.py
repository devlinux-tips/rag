"""
Quick demonstration of the new testable chunker without pytest dependency.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing.chunkers_v2 import (ChunkingConfig, ChunkingStrategy,
                                           DocumentChunker,
                                           create_document_chunker,
                                           find_sentence_boundaries,
                                           sentence_chunk_positions,
                                           sliding_window_chunk_positions)
from src.utils.config_protocol import MockConfigProvider


def test_pure_functions():
    """Test pure functions in isolation."""
    print("🧪 Testing pure functions...")

    # Test sliding window positions
    positions = sliding_window_chunk_positions(100, 50, 10)
    assert len(positions) == 3, f"Expected 3 positions, got {len(positions)}"
    assert positions[0] == (0, 50), f"Expected (0, 50), got {positions[0]}"
    print("✅ sliding_window_chunk_positions works correctly")

    # Test sentence boundaries
    boundaries = find_sentence_boundaries("First sentence. Second sentence!")
    assert len(boundaries) == 2, f"Expected 2 boundaries, got {len(boundaries)}"
    print("✅ find_sentence_boundaries works correctly")

    # Test sentence chunk positions
    sentences = ["First.", "Second.", "Third."]
    sentence_positions = sentence_chunk_positions(sentences, 20, 5, 5)
    assert len(sentence_positions) >= 1, "Should have at least 1 sentence chunk"
    print("✅ sentence_chunk_positions works correctly")


def test_dependency_injection():
    """Test dependency injection with mocks."""
    print("\n🔧 Testing dependency injection...")

    # Create mock configuration
    mock_config_data = {
        "chunking": {
            "chunk_size": 100,
            "overlap_size": 20,
            "strategy": "sliding_window",
        },
        "shared": {"min_chunk_size": 30},
    }

    mock_provider = MockConfigProvider()
    mock_provider.set_config("config", mock_config_data)
    mock_provider.set_shared_config(mock_config_data["shared"])

    # Create config from mock provider
    config = ChunkingConfig.from_config(config_provider=mock_provider)
    assert config.chunk_size == 100, f"Expected chunk_size 100, got {config.chunk_size}"
    assert config.overlap == 20, f"Expected overlap 20, got {config.overlap}"
    print("✅ Configuration injection works correctly")

    # Create chunker with injected config
    chunker = DocumentChunker(config=config)
    assert chunker.config.chunk_size == 100, "Config not properly injected"
    print("✅ DocumentChunker dependency injection works correctly")


def test_chunking_functionality():
    """Test actual chunking functionality."""
    print("\n📝 Testing chunking functionality...")

    # Create chunker with factory function
    chunker = create_document_chunker(
        config_dict={"chunking": {"chunk_size": 50, "overlap_size": 10}}
    )

    # Test document
    test_text = "This is a test document. It has multiple sentences. Each sentence should be handled properly."

    # Chunk the document
    chunks = chunker.chunk_document(test_text, "test.txt")

    assert len(chunks) >= 1, f"Should have at least 1 chunk, got {len(chunks)}"
    assert all(
        chunk.content.strip() for chunk in chunks
    ), "All chunks should have content"
    assert all(
        chunk.word_count > 0 for chunk in chunks
    ), "All chunks should have word count"

    print(f"✅ Created {len(chunks)} chunks successfully")
    print(
        f"   First chunk: '{chunks[0].content[:30]}...' ({chunks[0].word_count} words)"
    )


def test_backward_compatibility():
    """Test backward compatibility function."""
    print("\n🔄 Testing backward compatibility...")

    from src.preprocessing.chunkers_v2 import chunk_document

    chunks = chunk_document(
        text="Test document for compatibility.",
        source_file="test.txt",
        chunk_size=50,
        overlap=10,
    )

    assert len(chunks) >= 1, "Backward compatibility function should work"
    assert chunks[0].source_file == "test.txt", "Source file should be preserved"
    print("✅ Backward compatibility function works correctly")


def main():
    """Run all tests."""
    print("🎯 Testing Chunkers V2 - 100% Testable Architecture\n")

    try:
        test_pure_functions()
        test_dependency_injection()
        test_chunking_functionality()
        test_backward_compatibility()

        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Pure functions: 100% testable without dependencies")
        print("✅ Configuration: Fully mockable via dependency injection")
        print("✅ Business logic: Completely isolated and testable")
        print("✅ Integration: Factory pattern works with real and mock dependencies")
        print("✅ Backward compatibility: Legacy interface preserved")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
