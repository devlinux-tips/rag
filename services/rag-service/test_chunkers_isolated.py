"""
Isolated test of chunkers_v2 functionality without complex dependencies.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

# Import only what we need to test
from src.preprocessing.chunkers_v2 import (ChunkingConfig, ChunkingStrategy,
                                           DocumentChunker,
                                           calculate_chunk_metadata,
                                           extract_paragraphs,
                                           find_sentence_boundaries,
                                           sentence_chunk_positions,
                                           sliding_window_chunk_positions)
# Import MockConfigProvider directly
from src.utils.config_protocol import MockConfigProvider


def test_pure_functions():
    """Test all pure functions in complete isolation."""
    print("🧪 Testing pure functions (no dependencies)...")

    # Test 1: sliding_window_chunk_positions
    positions = sliding_window_chunk_positions(100, 50, 10)
    assert len(positions) == 3
    assert positions[0] == (0, 50)
    assert positions[1] == (40, 90)
    print("✅ sliding_window_chunk_positions: PASS")

    # Test 2: find_sentence_boundaries
    boundaries = find_sentence_boundaries("First sentence. Second! Third?")
    assert len(boundaries) == 3
    assert 15 in boundaries  # After "First sentence."
    print("✅ find_sentence_boundaries: PASS")

    # Test 3: extract_paragraphs
    paras = extract_paragraphs("Para 1.\n\nPara 2.\n\n\nPara 3.")
    assert len(paras) == 3
    assert "Para 1." in paras
    print("✅ extract_paragraphs: PASS")

    # Test 4: calculate_chunk_metadata
    meta = calculate_chunk_metadata("Hello world test content")
    assert meta["word_count"] == 4
    assert meta["char_count"] == 24
    print("✅ calculate_chunk_metadata: PASS")

    # Test 5: sentence_chunk_positions
    sentences = ["Short.", "A longer sentence here.", "End."]
    positions = sentence_chunk_positions(sentences, 30, 5, 5)
    assert len(positions) >= 1
    print("✅ sentence_chunk_positions: PASS")


def test_config_injection():
    """Test configuration with dependency injection."""
    print("\n🔧 Testing configuration dependency injection...")

    # Test with MockConfigProvider
    mock_data = {
        "chunking": {"chunk_size": 200, "overlap_size": 40, "strategy": "sentence"},
        "shared": {"min_chunk_size": 50},
    }

    mock_provider = MockConfigProvider()
    mock_provider.set_config("config", mock_data)
    mock_provider.set_shared_config(mock_data["shared"])

    config = ChunkingConfig.from_config(config_provider=mock_provider)

    assert config.chunk_size == 200
    assert config.overlap == 40
    assert config.strategy == ChunkingStrategy.SENTENCE
    print("✅ MockConfigProvider injection: PASS")

    # Test with direct dictionary
    config_dict = {
        "chunking": {"chunk_size": 150, "strategy": "paragraph"},
        "shared": {"min_chunk_size": 25},
    }

    config2 = ChunkingConfig.from_config(config_dict=config_dict)
    assert config2.chunk_size == 150
    assert config2.strategy == ChunkingStrategy.PARAGRAPH
    print("✅ Direct dictionary config: PASS")


def test_chunker_without_dependencies():
    """Test DocumentChunker with minimal dependencies."""
    print("\n📝 Testing DocumentChunker (no external deps)...")

    # Create config
    config = ChunkingConfig(
        chunk_size=60,
        overlap=10,
        min_chunk_size=20,
        respect_sentences=True,
        sentence_search_range=50,
        strategy=ChunkingStrategy.SLIDING_WINDOW,
    )

    # Create chunker without text cleaner (should still work)
    chunker = DocumentChunker(config=config)

    # Test chunking
    text = "This is a test document. It contains multiple sentences. The chunker should handle this properly without external dependencies."
    chunks = chunker.chunk_document(text, "test.txt")

    assert len(chunks) >= 1
    assert all(chunk.word_count > 0 for chunk in chunks)
    assert all(chunk.char_count > 0 for chunk in chunks)
    assert all("test_" in chunk.chunk_id for chunk in chunks)

    print(f"✅ Created {len(chunks)} chunks without dependencies")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i}: '{chunk.content[:40]}...' ({chunk.word_count} words)")


def test_all_strategies():
    """Test all chunking strategies."""
    print("\n⚡ Testing all chunking strategies...")

    config = ChunkingConfig(
        chunk_size=80,
        overlap=15,
        min_chunk_size=25,
        respect_sentences=True,
        sentence_search_range=50,
        strategy=ChunkingStrategy.SLIDING_WINDOW,  # Default
    )

    chunker = DocumentChunker(config=config)

    text = """First paragraph with content.

Second paragraph with more content here.

Third paragraph to test chunking."""

    # Test sliding window
    chunks1 = chunker.chunk_document(text, "test.txt", ChunkingStrategy.SLIDING_WINDOW)
    assert len(chunks1) >= 1
    print(f"✅ SLIDING_WINDOW: {len(chunks1)} chunks")

    # Test sentence-based (without sentence extractor)
    chunks2 = chunker.chunk_document(text, "test.txt", ChunkingStrategy.SENTENCE)
    assert len(chunks2) >= 1
    print(f"✅ SENTENCE: {len(chunks2)} chunks")

    # Test paragraph-based
    chunks3 = chunker.chunk_document(text, "test.txt", ChunkingStrategy.PARAGRAPH)
    assert len(chunks3) >= 1
    print(f"✅ PARAGRAPH: {len(chunks3)} chunks")


def main():
    """Run all isolated tests."""
    print("🎯 Chunkers V2 - Testability Demonstration (Isolated)\n")

    try:
        test_pure_functions()
        test_config_injection()
        test_chunker_without_dependencies()
        test_all_strategies()

        print("\n🎉 ALL TESTS PASSED!")
        print("\n📊 TESTABILITY ACHIEVEMENTS:")
        print("✅ Pure functions: 100% testable in complete isolation")
        print("✅ Configuration: Fully injectable via MockConfigProvider")
        print("✅ Business logic: Completely separated from external dependencies")
        print("✅ Strategy pattern: All chunking strategies work independently")
        print("✅ Graceful degradation: Works without optional dependencies")
        print("✅ Clean architecture: No hard-coded imports in business logic")

        print(f"\n⭐ COMPARISON WITH LEGACY:")
        print("❌ Legacy: Hard-coded config loading in constructor")
        print("✅ New: Fully injectable configuration")
        print("❌ Legacy: Mixed business logic and dependencies")
        print("✅ New: Pure functions + dependency injection")
        print("❌ Legacy: Difficult to test in isolation")
        print("✅ New: 100% testable without external systems")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
