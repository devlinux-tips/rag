#!/usr/bin/env python3
"""
Device Testing and Switching Demo for Croatian RAG System

This script demonstrates device detection, switching, and performance testing
for the Croatian embedding model with BGE-M3.
"""

import logging
import time
from typing import List

import torch

from src.vectordb.embeddings import CroatianEmbeddingModel, EmbeddingConfig


def setup_logging():
    """Configure logging for the demo."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def print_device_capabilities():
    """Print detailed device capabilities."""
    print("=" * 60)
    print("DEVICE CAPABILITIES ANALYSIS")
    print("=" * 60)

    # Create model to access device info
    config = EmbeddingConfig.from_config()
    model = CroatianEmbeddingModel(config)
    device_info = model.get_device_info()

    print(f"PyTorch Version: {device_info['pytorch_version']}")
    print(f"Current Device: {device_info['current_device']}")
    print()

    # CPU Info
    print("📱 CPU:")
    print(f"  Available: {device_info['cpu_available']}")
    print()

    # CUDA Info
    print("🚀 CUDA:")
    print(f"  Available: {device_info['cuda_available']}")
    if device_info["cuda_available"]:
        print(f"  Version: {device_info['cuda_version']}")
        print(f"  Device Count: {device_info['cuda_device_count']}")
        for device in device_info["cuda_devices"]:
            print(f"    Device {device['id']}: {device['name']}")
            print(f"      Memory: {device['memory_total']} total")
            print(f"      Compute: {device['compute_capability']}")
    else:
        print("  No CUDA devices available")
    print()

    # MPS Info (Apple Silicon)
    print("🍎 MPS (Apple Silicon):")
    print(f"  Available: {device_info['mps_available']}")
    if device_info["mps_available"]:
        if device_info.get("mps_built"):
            print(f"  Built: {device_info['mps_built']}")
        if device_info.get("apple_silicon"):
            print(f"  Apple Silicon: {device_info['apple_silicon']}")
        if device_info.get("apple_chip"):
            print(f"  Chip: {device_info['apple_chip']}")
        if device_info.get("unified_memory_total"):
            print(f"  Unified Memory: {device_info['unified_memory_total']}")
    print()


def test_device_performance(device: str, test_texts: List[str]) -> dict:
    """Test embedding performance on a specific device."""
    print(f"Testing device: {device}")

    try:
        # Create config with specific device
        config = EmbeddingConfig.from_config()
        config.device = device
        config.batch_size = 16  # Smaller batch for testing

        model = CroatianEmbeddingModel(config)

        # Load model and measure time
        start_time = time.time()
        model.load_model()
        load_time = time.time() - start_time

        print(f"  ✓ Model loaded in {load_time:.2f}s")
        print(f"  ✓ Resolved device: {model.device}")

        # Get model info
        model_info = model.get_model_info()
        print(f"  ✓ Embedding dimension: {model_info['embedding_dimension']}")

        # Test encoding performance
        start_time = time.time()
        embeddings = model.encode_text(test_texts)
        encode_time = time.time() - start_time

        texts_per_second = len(test_texts) / encode_time

        print(f"  ✓ Encoded {len(test_texts)} texts in {encode_time:.2f}s")
        print(f"  ✓ Performance: {texts_per_second:.1f} texts/second")

        # Memory info for CUDA
        if device.startswith("cuda") and torch.cuda.is_available():
            memory_used = model_info.get("cuda_memory_allocated", "N/A")
            print(f"  ✓ GPU memory used: {memory_used}")

        return {
            "success": True,
            "device": model.device,
            "load_time": load_time,
            "encode_time": encode_time,
            "texts_per_second": texts_per_second,
            "embedding_shape": embeddings.shape,
            "model_info": model_info,
        }

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return {"success": False, "device": device, "error": str(e)}


def test_device_switching():
    """Test dynamic device switching."""
    print("=" * 60)
    print("DEVICE SWITCHING TEST")
    print("=" * 60)

    config = EmbeddingConfig.from_config()
    model = CroatianEmbeddingModel(config)

    # Load model first
    model.load_model()
    original_device = model.device
    print(f"Original device: {original_device}")

    # Test switching to CPU
    print("\nSwitching to CPU...")
    try:
        model.switch_device("cpu")

        # Test encoding after switch
        test_text = "Test nakon prebacivanja na CPU."
        embedding = model.encode_text([test_text])
        print(f"✓ CPU encoding successful: {embedding.shape}")

    except Exception as e:
        print(f"✗ CPU switch failed: {e}")

    # Test switching to MPS (if available)
    if torch.backends.mps.is_available():
        print("\nSwitching to MPS (Apple Silicon)...")
        try:
            model.switch_device("mps")

            # Test encoding after switch
            test_text = "Test nakon prebacivanja na MPS (Apple Silicon)."
            embedding = model.encode_text([test_text])
            print(f"✓ MPS encoding successful: {embedding.shape}")

        except Exception as e:
            print(f"✗ MPS switch failed: {e}")

    # Test switching to CUDA (if available)
    if torch.cuda.is_available():
        print("\nSwitching to CUDA...")
        try:
            model.switch_device("cuda")

            # Test encoding after switch
            test_text = "Test nakon prebacivanja na CUDA."
            embedding = model.encode_text([test_text])
            print(f"✓ CUDA encoding successful: {embedding.shape}")

        except Exception as e:
            print(f"✗ CUDA switch failed: {e}")

    print(f"\nFinal device: {model.device}")


def main():
    """Run the complete device testing demo."""
    setup_logging()

    print("🇭🇷 Croatian RAG System - Device Testing Demo")
    print()

    # Print device capabilities
    print_device_capabilities()

    # Test texts in multiple languages
    test_texts = [
        "Ovo je test tekst na hrvatskom jeziku za testiranje embedding modela.",
        "This is a test text in English for embedding model testing.",
        "Dies ist ein Testtext auf Deutsch für das Testen des Embedding-Modells.",
        "Este es un texto de prueba en español para probar el modelo de embeddings.",
        "Това е тестов текст на български език за тестване на модела за embeddings.",
    ]

    print("=" * 60)
    print("PERFORMANCE TESTING")
    print("=" * 60)
    print(f"Test corpus: {len(test_texts)} multilingual texts")
    print()

    # Test available devices
    devices_to_test = ["cpu"]
    if torch.backends.mps.is_available():
        devices_to_test.append("mps")
    if torch.cuda.is_available():
        devices_to_test.append("cuda")

    results = {}
    for device in devices_to_test:
        print(f"Testing {device.upper()}:")
        print("-" * 40)
        results[device] = test_device_performance(device, test_texts)
        print()

    # Performance comparison
    if len(results) > 1:
        print("=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)

        successful_results = {k: v for k, v in results.items() if v["success"]}

        if len(successful_results) > 1:
            # Sort by performance
            sorted_results = sorted(
                successful_results.items(), key=lambda x: x[1]["texts_per_second"], reverse=True
            )

            print("Performance ranking (texts/second):")
            for i, (device, result) in enumerate(sorted_results, 1):
                speedup = result["texts_per_second"] / sorted_results[-1][1]["texts_per_second"]
                print(
                    f"{i}. {device.upper()}: {result['texts_per_second']:.1f} texts/s (🚀 {speedup:.1f}x)"
                )

            print()

            # Memory comparison
            print("Memory usage:")
            for device, result in successful_results.items():
                if "cuda_memory_allocated" in result["model_info"]:
                    print(
                        f"  {device.upper()}: {result['model_info']['cuda_memory_allocated']} GPU"
                    )
                else:
                    print(f"  {device.upper()}: System RAM")

    # Test device switching
    test_device_switching()

    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) available - Excellent for development")
        print("   • Great performance with unified memory")
        print("   • Set device='auto' for automatic detection")
        print("   • Ideal for M1/M2/M3/M4 Pro/Max development")
    elif torch.cuda.is_available():
        print("✅ CUDA available - Recommended for production use")
        print("   • Fastest performance for large batches")
        print("   • Set device='auto' in config for automatic detection")
        print("   • Monitor GPU memory with nvidia-smi")
    else:
        print("⚠️  GPU acceleration not available - CPU only")
        print("   • Install CUDA-enabled PyTorch (NVIDIA) or use Apple Silicon")
        print("   • See docs/pytorch_device_setup.md for installation guide")

    print()
    print("Configuration options:")
    print("  • device='auto'  - Automatic detection (recommended)")
    print("  • device='mps'   - Force Apple Silicon (M1/M2/M3/M4)")
    print("  • device='cuda'  - Force CUDA (fails if not available)")
    print("  • device='cpu'   - Force CPU (always works)")
    print("  • device='cuda:0'- Specific CUDA device")

    print()
    print("🎉 Device testing complete!")


if __name__ == "__main__":
    main()
