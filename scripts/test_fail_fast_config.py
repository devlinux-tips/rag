#!/usr/bin/env python3
"""
Test script for fail-fast configuration system.
Validates that system fails immediately when essential config is missing.
"""

import shutil
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
services_path = project_root / "services" / "rag-service"
sys.path.insert(0, str(services_path))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_fail_fast_with_valid_config():
    """Test fail-fast config loader with valid configuration."""
    logger.info("🧪 Testing fail-fast config with valid configuration...")

    try:
        from src.utils.fail_fast_config import (get_default_language_strict,
                                                get_supported_languages_strict,
                                                validate_essential_config)

        # This should work with existing config.toml
        validate_essential_config()

        languages = get_supported_languages_strict()
        default_lang = get_default_language_strict()

        logger.info(f"✅ Valid config test passed")
        logger.info(f"  Languages: {languages}")
        logger.info(f"  Default: {default_lang}")

        return True

    except Exception as e:
        logger.error(f"❌ Valid config test failed: {e}")
        return False


def test_fail_fast_with_missing_config():
    """Test fail-fast behavior when config is missing."""
    logger.info("🧪 Testing fail-fast behavior with missing config...")

    # Create temporary empty config directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config_dir = Path(temp_dir) / "config"
        temp_config_dir.mkdir()

        # Create invalid config.toml (missing required sections)
        invalid_config = temp_config_dir / "config.toml"
        invalid_config.write_text(
            """
# Invalid config - missing languages section
[project]
name = "Test"

[shared]
cache_dir = "./data/cache"
"""
        )

        try:
            # Import and test with temp config directory
            from src.utils.fail_fast_config import (ConfigurationError,
                                                    FailFastConfigLoader)

            loader = FailFastConfigLoader(config_dir=temp_config_dir)

            # This should fail fast
            try:
                loader.get_supported_languages()
                logger.error("❌ Expected ConfigurationError but got success")
                return False
            except ConfigurationError as e:
                logger.info(f"✅ Correctly failed fast: {e}")
                return True

        except Exception as e:
            logger.error(f"❌ Unexpected error in fail-fast test: {e}")
            return False


def test_language_manager_fail_fast():
    """Test that language manager fails fast with invalid config."""
    logger.info("🧪 Testing language manager fail-fast behavior...")

    # Create temporary invalid config
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config_dir = Path(temp_dir) / "config"
        temp_config_dir.mkdir()

        invalid_config = temp_config_dir / "config.toml"
        invalid_config.write_text(
            """
# Invalid config - missing languages.supported
[languages]
default = "en"
# missing: supported = ["en", "hr"]

[shared]
default_chunk_size = 512
default_chunk_overlap = 50
"""
        )

        try:
            # Monkey-patch the CONFIG_DIR temporarily
            import src.utils.fail_fast_config

            original_config_dir = src.utils.fail_fast_config.CONFIG_DIR
            src.utils.fail_fast_config.CONFIG_DIR = temp_config_dir

            # Clear any cached loader
            src.utils.fail_fast_config._fail_fast_loader = None

            # Now try to create language manager - should fail
            from src.utils.language_manager import LanguageManager

            try:
                manager = LanguageManager()  # Should fail
                logger.error("❌ Expected LanguageManager to fail but it succeeded")
                return False
            except Exception as e:
                logger.info(f"✅ LanguageManager correctly failed fast: {e}")
                return True
            finally:
                # Restore original config
                src.utils.fail_fast_config.CONFIG_DIR = original_config_dir
                src.utils.fail_fast_config._fail_fast_loader = None

        except Exception as e:
            logger.error(f"❌ Unexpected error in language manager test: {e}")
            return False


def test_config_validation_patterns():
    """Test different configuration validation patterns."""
    logger.info("🧪 Testing configuration validation patterns...")

    results = []

    # Test cases with different missing config scenarios
    test_cases = [
        (
            "Missing languages section",
            """
[project]
name = "Test"
""",
        ),
        (
            "Missing supported languages",
            """
[languages]
default = "en"
# missing supported array
""",
        ),
        (
            "Empty supported languages",
            """
[languages]
default = "en"
supported = []
""",
        ),
        (
            "Missing default language",
            """
[languages]
supported = ["en", "hr"]
# missing default
""",
        ),
        (
            "Invalid default language",
            """
[languages]
supported = ["en", "hr"]
default = "fr"  # not in supported
""",
        ),
    ]

    for test_name, config_content in test_cases:
        logger.info(f"  Testing: {test_name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_dir = Path(temp_dir) / "config"
            temp_config_dir.mkdir()

            config_file = temp_config_dir / "config.toml"
            config_file.write_text(config_content)

            try:
                from src.utils.fail_fast_config import (ConfigurationError,
                                                        FailFastConfigLoader)

                loader = FailFastConfigLoader(config_dir=temp_config_dir)

                try:
                    # Test different methods
                    loader.get_supported_languages()
                    loader.get_default_language()
                    results.append(f"❌ {test_name}: Expected failure but succeeded")
                except ConfigurationError as e:
                    results.append(f"✅ {test_name}: Correctly failed")
                except Exception as e:
                    results.append(f"⚠️ {test_name}: Unexpected error type: {e}")

            except Exception as e:
                results.append(f"❌ {test_name}: Setup failed: {e}")

    # Print results
    for result in results:
        logger.info(f"    {result}")

    # Return True if all tests showed expected behavior (failure)
    success_count = sum(1 for r in results if r.startswith("✅"))
    return success_count == len(test_cases)


def main():
    """Run all fail-fast configuration tests."""
    logger.info("🚀 Fail-Fast Configuration Testing")
    logger.info("=" * 50)

    tests = [
        ("Valid Config Test", test_fail_fast_with_valid_config),
        ("Missing Config Test", test_fail_fast_with_missing_config),
        ("Language Manager Test", test_language_manager_fail_fast),
        ("Validation Patterns Test", test_config_validation_patterns),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅" if result else "❌"
            logger.info(f"{status} {test_name} completed")
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)

    logger.info(f"\n🎉 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✅ All fail-fast configuration tests PASSED!")
        logger.info("🎯 System correctly fails fast when essential config is missing")
        logger.info("🛡️ No more silent fallbacks to extensive hardcoded defaults")
    else:
        failed = [name for name, result in results if not result]
        logger.warning(f"❌ Failed tests: {', '.join(failed)}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
