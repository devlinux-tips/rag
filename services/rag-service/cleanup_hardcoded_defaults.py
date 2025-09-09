#!/usr/bin/env python3
"""
Script to systematically remove hardcoded defaults and implement fail-fast behavior.
This ensures all configuration is properly loaded from TOML files.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


def find_and_replace_in_file(
    file_path: Path, patterns_and_replacements: List[Tuple[str, str]]
) -> bool:
    """Find and replace patterns in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        modified = False
        for pattern, replacement in patterns_and_replacements:
            old_content = content
            content = re.sub(
                pattern, replacement, content, flags=re.MULTILINE | re.DOTALL
            )
            if content != old_content:
                modified = True
                print(f"  ✅ Applied pattern in {file_path.name}")

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False


def cleanup_config_files():
    """Clean up configuration loading patterns."""
    print("🔧 Cleaning up configuration files...")

    src_path = Path("src")
    if not src_path.exists():
        print("❌ src/ directory not found")
        return

    # Patterns to replace fallback_value with fail-fast
    patterns = [
        # Remove handle_config_error with fallback_value
        (
            r"handle_config_error\(\s*operation=[^,]+,\s*fallback_value=[^,]+,\s*config_file=[^,]+,\s*section=[^,)]+\)",
            r"operation()",
        ),
        # Replace .get() with hardcoded defaults with direct access
        (r'(\w+)\.get\(["\'](\w+)["\'],\s*([^,)]+)\)', r'\1["\2"]'),
        # Remove fallback_value parameters completely
        (r",\s*fallback_value=[^,)]+", r""),
    ]

    # Find all Python files in src
    python_files = list(src_path.rglob("*.py"))
    print(f"Found {len(python_files)} Python files to process")

    modified_files = []

    for py_file in python_files:
        if "test" in str(py_file).lower():
            continue  # Skip test files for now

        print(f"Processing {py_file.relative_to(src_path)}")
        if find_and_replace_in_file(py_file, patterns):
            modified_files.append(py_file)

    print(f"\n✅ Modified {len(modified_files)} files:")
    for f in modified_files:
        print(f"  - {f.relative_to(src_path)}")


def cleanup_specific_patterns():
    """Clean up specific hardcoded default patterns we identified."""
    print("\n🎯 Cleaning up specific hardcoded patterns...")

    specific_fixes = [
        # vectordb/embeddings.py - remove fallback cache folder
        (
            Path("src/vectordb/embeddings.py"),
            [
                (r'fallback_value="[^"]*cache[^"]*"', ""),
                (
                    r'embedding_config\.get\("cache_folder"[^)]+\)',
                    'embedding_config["cache_folder"]',
                ),
            ],
        ),
        # vectordb/search.py - remove top_k fallbacks
        (
            Path("src/vectordb/search.py"),
            [
                (
                    r'top_k=search_config\.get\("top_k"[^)]+\)',
                    'top_k=search_config["top_k"]',
                ),
                (
                    r"similarity_threshold[^)]+shared_config\.get[^)]+\)",
                    'similarity_threshold=search_config["similarity_threshold"]',
                ),
            ],
        ),
    ]

    for file_path, patterns in specific_fixes:
        if file_path.exists():
            print(f"Processing {file_path.name}")
            find_and_replace_in_file(file_path, patterns)


def validate_config_requirements():
    """Validate that all required config keys are present in config.toml"""
    print("\n🔍 Validating config.toml requirements...")

    config_file = Path("config/config.toml")
    if not config_file.exists():
        print("❌ config/config.toml not found")
        return

    with open(config_file, "r") as f:
        config_content = f.read()

    # Required keys for fail-fast operation
    required_keys = [
        # Basic paths
        "data_base_dir",
        "models_base_dir",
        "system_dir",
        # Path templates
        "tenant_root_template",
        "user_documents_template",
        "tenant_shared_template",
        "user_processed_template",
        "tenant_processed_template",
        "chromadb_path_template",
        "models_path_template",
        "collection_name_template",
        # Processing settings
        "max_chunk_size",
        "chunk_overlap",
        "min_chunk_size",
        # Embeddings
        "model_name",
        "batch_size",
        "max_seq_length",
        "device",
        "normalize_embeddings",
        # Storage
        "db_path",
        "collection_name",
        "distance_metric",
        "persist",
        "allow_reset",
    ]

    missing_keys = []
    for key in required_keys:
        if key not in config_content:
            missing_keys.append(key)

    if missing_keys:
        print(f"⚠️  Missing required config keys: {missing_keys}")
        print("These keys must be present in config.toml for fail-fast operation")
    else:
        print("✅ All required config keys are present")


def main():
    """Main cleanup function."""
    print("=" * 60)
    print("🚀 Fail-Fast Configuration Cleanup")
    print("=" * 60)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Run cleanup steps
    cleanup_config_files()
    cleanup_specific_patterns()
    validate_config_requirements()

    print("\n" + "=" * 60)
    print("✅ Cleanup completed!")
    print("💡 Next steps:")
    print("   1. Test configuration loading with python test_config_loading.py")
    print("   2. Run unit tests to ensure no regressions")
    print("   3. Update any missing config keys identified above")
    print("=" * 60)


if __name__ == "__main__":
    main()
