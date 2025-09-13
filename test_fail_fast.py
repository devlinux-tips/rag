#!/usr/bin/env python3
"""
Test: Does system fail fast now that fallbacks are removed?
"""

import sys

sys.path.append("services/rag-service/src")


def test_fail_fast_behavior():
    """Test that production providers fail fast without fallbacks."""
    print("🔍 Testing fail-fast behavior after fallback removal...")

    try:
        from utils.config_validator import ConfigurationError
        from utils.language_manager_providers import (
            ProductionConfigProvider,
            ProductionPatternProvider,
        )

        print("✅ Imports successful")

        # Test config provider - should fail if config is bad
        print("🧪 Testing ProductionConfigProvider...")
        config_provider = ProductionConfigProvider()

        try:
            settings = config_provider.get_language_settings()
            print("⚠️  Config provider succeeded - config might be working")
            print(
                f"   Got: {settings.supported_languages}, model: {settings.embedding_model}"
            )
        except ConfigurationError as e:
            print(f"✅ GOOD: Config provider failed fast: {e}")
        except Exception as e:
            print(f"🤔 Config provider failed with: {type(e).__name__}: {e}")

        # Test pattern provider - should fail if config is bad
        print("🧪 Testing ProductionPatternProvider...")
        pattern_provider = ProductionPatternProvider()

        try:
            patterns = pattern_provider.get_language_patterns()
            print("⚠️  Pattern provider succeeded - config might be working")
            print(
                f"   Got detection patterns for: {list(patterns.detection_patterns.keys())}"
            )
        except ConfigurationError as e:
            print(f"✅ GOOD: Pattern provider failed fast: {e}")
        except Exception as e:
            print(f"🤔 Pattern provider failed with: {type(e).__name__}: {e}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")


if __name__ == "__main__":
    test_fail_fast_behavior()
