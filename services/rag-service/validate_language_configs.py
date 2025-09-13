#!/usr/bin/env python3
"""
Language Configuration Structure Validator

Ensures that all language configuration files have identical
key structures to prevent configuration drift and maintain language equality.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def load_toml_file(file_path: Path) -> Dict[str, Any]:
    """Load TOML file and return parsed content."""
    try:
        with open(file_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}


def get_nested_keys(data: Dict[str, Any], prefix: str = "") -> Set[str]:
    """Extract all nested keys from a dictionary with dot notation."""
    keys = set()

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        keys.add(full_key)

        if isinstance(value, dict):
            keys.update(get_nested_keys(value, full_key))

    return keys


def compare_config_structures(lang1_path: Path, lang2_path: Path) -> Tuple[bool, List[str]]:
    """Compare two language config files and return differences."""
    lang1_name = lang1_path.stem
    lang2_name = lang2_path.stem

    # Load configurations
    config1 = load_toml_file(lang1_path)
    config2 = load_toml_file(lang2_path)

    if not config1 or not config2:
        return False, [f"Failed to load one or both config files"]

    # Extract all keys
    keys1 = get_nested_keys(config1)
    keys2 = get_nested_keys(config2)

    # Find differences
    missing_in_lang2 = keys1 - keys2
    missing_in_lang1 = keys2 - keys1
    common_keys = keys1 & keys2

    issues = []

    # Report missing keys
    if missing_in_lang2:
        issues.append(f"❌ Keys in {lang1_name}.toml but missing in {lang2_name}.toml:")
        for key in sorted(missing_in_lang2):
            issues.append(f"   - {key}")

    if missing_in_lang1:
        issues.append(f"❌ Keys in {lang2_name}.toml but missing in {lang1_name}.toml:")
        for key in sorted(missing_in_lang1):
            issues.append(f"   - {key}")

    # Check value types for common keys
    type_mismatches = []
    for key in common_keys:
        # Navigate to the nested value
        value1 = config1
        value2 = config2

        try:
            for part in key.split("."):
                value1 = value1[part]
                value2 = value2[part]

            # Check if types match
            if not isinstance(value1, type(value2)) and not isinstance(value2, type(value1)):
                type_mismatches.append(
                    f"   - {key}: {lang1_name}={type(value1).__name__}, {lang2_name}={type(value2).__name__}"
                )

        except (KeyError, TypeError):
            # Skip if we can't navigate to the value
            pass

    if type_mismatches:
        issues.append(f"⚠️  Type mismatches for common keys:")
        issues.extend(type_mismatches)

    # Summary
    all_good = len(issues) == 0
    if all_good:
        issues.append(f"✅ {lang1_name}.toml and {lang2_name}.toml have identical structures!")
        issues.append(f"   📊 Total keys: {len(common_keys)}")

    return all_good, issues


def validate_all_language_configs() -> bool:
    """Validate all language configuration files."""
    config_dir = Path(__file__).parent / "config"

    # Find all language config files (excluding main config.toml)
    language_files = []
    for toml_file in config_dir.glob("*.toml"):
        if toml_file.name != "config.toml":  # Skip main config
            language_files.append(toml_file)

    if len(language_files) < 2:
        print("⚠️  Need at least 2 language config files to compare")
        return True

    print(f"🔍 Found {len(language_files)} language config files:")
    for file in language_files:
        print(f"   📄 {file.name}")
    print()

    # Compare each pair of language configs
    all_valid = True
    comparisons_made = 0

    for i, file1 in enumerate(language_files):
        for file2 in language_files[i + 1 :]:
            print(f"🔄 Comparing {file1.name} ↔ {file2.name}")
            print("=" * 50)

            is_valid, issues = compare_config_structures(file1, file2)

            for issue in issues:
                print(issue)

            print()
            comparisons_made += 1

            if not is_valid:
                all_valid = False

    # Final summary
    print("=" * 60)
    if all_valid:
        print(f"🎉 SUCCESS: All {len(language_files)} language configs have identical structures!")
        print(f"📊 Completed {comparisons_made} pairwise comparisons")
    else:
        print(f"💥 FAILURE: Language configs have structural differences!")
        print(f"🔧 Fix the missing keys to maintain language equality")

    return all_valid


if __name__ == "__main__":
    print("🏗️  Language Configuration Structure Validator")
    print("=" * 60)

    success = validate_all_language_configs()
    sys.exit(0 if success else 1)
