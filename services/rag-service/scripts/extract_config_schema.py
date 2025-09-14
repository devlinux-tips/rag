#!/usr/bin/env python3
"""
Extract actual schema from existing config files.
Generate ConfigValidator schema that matches reality.
"""

import sys
from typing import Any

import toml

sys.path.insert(0, "../src")


def extract_keys_from_config(
    config: dict[str, Any], prefix: str = ""
) -> dict[str, type]:
    """Extract all keys from config with their types."""
    schema = {}

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Nested section - recurse
            schema.update(extract_keys_from_config(value, full_key))
        else:
            # Leaf value - determine type
            if isinstance(value, int | float):
                schema[full_key] = type(value)  # Use actual type
            elif isinstance(value, bool):
                schema[full_key] = bool
            elif isinstance(value, str):
                schema[full_key] = str
            elif isinstance(value, list):
                schema[full_key] = list
            else:
                schema[full_key] = str  # Default to string

    return schema


def main():
    print("🔍 Extracting actual schema from config files...")

    # Load actual configs
    with open("config/config.toml") as f:
        main_config = toml.load(f)

    with open("config/hr.toml") as f:
        hr_config = toml.load(f)

    print(
        f"✅ Loaded configs: main ({len(main_config)} sections), hr ({len(hr_config)} sections)"
    )

    # Extract schemas
    main_schema = extract_keys_from_config(main_config)
    language_schema = extract_keys_from_config(hr_config)

    print(
        f"📊 Extracted schemas: main ({len(main_schema)} keys), language ({len(language_schema)} keys)"
    )

    # Generate ConfigValidator schema
    print("\n🔧 Generated MAIN_CONFIG_SCHEMA:")
    for key, typ in sorted(main_schema.items()):
        print(
            f'        "{key}": {typ.__name__ if not isinstance(typ, tuple) else typ},'
        )

    print("\n🔧 Generated LANGUAGE_CONFIG_SCHEMA:")
    for key, typ in sorted(language_schema.items()):
        print(
            f'        "{key}": {typ.__name__ if not isinstance(typ, tuple) else typ},'
        )

    print("\n🎯 SUMMARY:")
    print(
        f"   Main config keys: {len(main_schema)} (was expecting 140+ in ConfigValidator)"
    )
    print(
        f"   Language config keys: {len(language_schema)} (was expecting 120+ in ConfigValidator)"
    )
    print(
        f"   Total: {len(main_schema) + len(language_schema)} keys (ConfigValidator expected 189)"
    )


if __name__ == "__main__":
    main()
