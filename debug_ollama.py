#!/usr/bin/env python3
"""
Debug Ollama generation issue
"""
import asyncio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.generation.ollama_client import GenerationRequest, OllamaClient, OllamaConfig


async def debug_ollama():
    """Debug Ollama generation directly."""
    print("🔧 Testing Ollama directly...")

    # Test configuration
    config = OllamaConfig()
    print(f"Using model: {config.model}")

    client = OllamaClient(config)

    # Health check
    if client.health_check():
        print("✅ Ollama health check passed")
    else:
        print("❌ Ollama health check failed")
        return

    # Simple test
    test_context = [
        "Dokument sadrži informacije o odlukama donesenim 1. srpnja 2025, uključujući iznos od 15,32 EUR i 331,23 EUR."
    ]
    test_question = "Koje odluke su donesene 1. srpnja 2025, zanimaju nas samo iznosi u EURima?"

    # Build prompt
    prompt = f"""Na temelju sljedećih hrvatskih dokumenata odgovori na pitanje na hrvatskom jeziku.

DOKUMENTI:
Dokument 1: {test_context[0]}

PITANJE: {test_question}

VAŽNO: Koristi SAMO informacije iz dokumenata. Izvuci konkretne brojeve, datume i iznose ako postoje.

ODGOVOR:"""

    print(f"\n📝 Test prompt:\n{prompt}")
    print("\n" + "=" * 60)

    # Test generation
    request = GenerationRequest(
        prompt=prompt,
        context=test_context,
        query=test_question,
        query_type="factual",
        language="hr",
    )

    try:
        print("🤖 Generating response...")
        response = await client.generate_text_async(request)
        print(f"✅ Generation successful!")
        print(f"💬 Response: {response.text}")
        print(f"🔢 Tokens used: {response.tokens_used}")
        print(f"⏱️  Generation time: {response.generation_time:.2f}s")
        print(f"🎯 Confidence: {response.confidence:.2f}")

    except Exception as e:
        print(f"❌ Generation failed with error: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(debug_ollama())
