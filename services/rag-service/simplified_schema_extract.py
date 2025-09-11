#!/usr/bin/env python3
"""
Extract actual config structure from existing files using the config loader.
"""

import sys

sys.path.insert(0, "src")


def extract_actual_schema():
    """Extract schema from actual working config loader."""
    print("🔍 Testing actual config loading...")

    try:
        from utils.config_loader import get_language_config, load_config

        # Load configs the way the system does
        main_config = load_config("config")
        hr_config = get_language_config("hr")
        en_config = get_language_config("en")

        print(f"✅ Loaded configs successfully")
        print(f"   Main config sections: {list(main_config.keys())}")
        print(f"   HR config sections: {list(hr_config.keys())}")
        print(f"   EN config sections: {list(en_config.keys())}")

        def count_keys(config, prefix=""):
            count = 0
            keys = []
            for key, value in config.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    sub_count, sub_keys = count_keys(value, full_key)
                    count += sub_count
                    keys.extend(sub_keys)
                else:
                    count += 1
                    keys.append(full_key)
            return count, keys

        main_count, main_keys = count_keys(main_config)
        hr_count, hr_keys = count_keys(hr_config)

        print(f"\n📊 ACTUAL CONFIG STRUCTURE:")
        print(f"   Main config: {main_count} keys")
        print(f"   Language config: {hr_count} keys")
        print(f"   Total: {main_count + hr_count} keys")

        print(f"\n🎯 ConfigValidator was expecting 189 keys")
        print(f"   Reality: {main_count + hr_count} keys")
        print(f"   Difference: {189 - (main_count + hr_count)} extra keys in schema")

        print(f"\n📋 SAMPLE MAIN CONFIG KEYS:")
        for key in sorted(main_keys)[:20]:
            print(f"   {key}")

        print(f"\n📋 SAMPLE LANGUAGE CONFIG KEYS:")
        for key in sorted(hr_keys)[:20]:
            print(f"   {key}")

        return main_keys, hr_keys

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return [], []


if __name__ == "__main__":
    extract_actual_schema()
