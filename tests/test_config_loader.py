#!/usr/bin/env python3
"""
Test suite for the centralized configuration loader system.
Validates config loading, Croatian settings integration, and OllamaConfig functionality.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config_loading():
    """Test configuration loader functionality."""
    print("🧪 Testing Configuration Loader")
    print("=" * 50)

    try:
        # Test config loader import
        from src.utils.config_loader import (
            get_croatian_prompts,
            get_croatian_settings,
            get_project_info,
            load_config,
        )

        print("✅ Config loader imported successfully")

        # Test main config
        main_config = load_config("main")
        print(f"✅ Main config loaded: {main_config['project']['name']}")

        # Test Croatian config
        croatian_config = get_croatian_settings()
        print(f"✅ Croatian config loaded: {croatian_config['language']['name']}")

        # Test Croatian prompts
        prompts = get_croatian_prompts()
        print(f"✅ Croatian prompts loaded: {prompts['system_base'][:50]}...")

        # Test Ollama config
        ollama_config = load_config("generation")
        print(f"✅ Generation config loaded: {ollama_config['ollama']['primary_model']}")

        # Test project info
        project_info = get_project_info()
        print(f"✅ Project info: {project_info['name']} v{project_info['version']}")

        print("\n🎉 All configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ollama_config():
    """Test OllamaConfig with new TOML loading."""
    print("\n🔧 Testing OllamaConfig Integration")
    print("=" * 50)

    try:
        from src.generation.ollama_client import OllamaConfig

        # Test config loading
        config = OllamaConfig.from_config()
        print(f"✅ OllamaConfig loaded successfully")
        print(f"   • Model: {config.model}")
        print(f"   • Base URL: {config.base_url}")
        print(f"   • Temperature: {config.temperature}")
        print(f"   • Croatian diacritics: {config.preserve_diacritics}")
        print(f"   • Formal style: {config.prefer_formal_style}")
        print(f"   • Cultural context: {config.include_cultural_context}")

        return True

    except Exception as e:
        print(f"❌ OllamaConfig test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_croatian_integration():
    """Test Croatian language integration with config."""
    print("\n🇭🇷 Testing Croatian Language Integration")
    print("=" * 50)

    try:
        from src.utils.config_loader import (
            get_croatian_confidence_settings,
            get_croatian_formal_prompts,
            get_croatian_language_code,
            get_croatian_text_processing,
        )

        # Test Croatian text processing config
        text_config = get_croatian_text_processing()
        print(f"✅ Croatian text processing config loaded")
        print(f"   • Remove diacritics: {text_config['remove_diacritics']}")
        print(f"   • Normalize case: {text_config['normalize_case']}")
        print(f"   • Special chars: {', '.join(text_config['preserve_special_chars'][:5])}...")

        # Test language code
        lang_code = get_croatian_language_code()
        print(f"✅ Croatian language code: {lang_code}")

        # Test confidence settings
        confidence_config = get_croatian_confidence_settings()
        print(f"✅ Croatian confidence settings loaded")
        print(f"   • Error phrases: {', '.join(confidence_config['error_phrases'][:3])}...")
        print(f"   • Confidence threshold: {confidence_config['confidence_threshold']}")

        # Test formal prompts
        formal_prompts = get_croatian_formal_prompts()
        print(f"✅ Croatian formal prompts loaded")
        print(f"   • Formal instruction: {formal_prompts['formal_instruction'][:30]}...")

        return True

    except Exception as e:
        print(f"❌ Croatian integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Croatian RAG Configuration Loader Test Suite")
    print("Validating centralized configuration system")
    print()

    success = True
    success &= test_config_loading()
    success &= test_ollama_config()
    success &= test_croatian_integration()

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Configuration system is working correctly.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED. Check output above for details.")
        sys.exit(1)
