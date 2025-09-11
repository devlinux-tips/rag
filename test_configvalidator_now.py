#!/usr/bin/env python3
"""
Quick test: Does ConfigValidator work with existing configs?
"""

import os
import sys

sys.path.append("services/rag-service/src")

try:
    from utils.config_loader import get_language_config, load_config
    from utils.config_validator import ConfigurationError, ConfigValidator

    print("🔍 Testing ConfigValidator against existing configs...")

    # Load actual configs
    main_config = load_config("config")
    print(f"✅ Loaded main config: {len(main_config)} sections")

    # Load language configs
    language_configs = {}
    for lang in ["hr", "en"]:
        try:
            lang_config = get_language_config(lang)
            language_configs[lang] = lang_config
            print(f"✅ Loaded {lang} config: {len(lang_config)} sections")
        except Exception as e:
            print(f"⚠️  Could not load {lang} config: {e}")

    # Test ConfigValidator
    if language_configs:
        ConfigValidator.validate_startup_config(main_config, language_configs)
        print("🎯 SUCCESS: ConfigValidator passed with existing configs!")
        print("👉 All keys are present, no need to add anything")
    else:
        print("❌ No language configs found to test")

except ImportError as e:
    print(f"❌ Import error: {e}")
except ConfigurationError as e:
    print(f"❌ ConfigValidator FAILED: {e}")
    print("👉 Some keys are missing, need to investigate")
except Exception as e:
    print(f"💥 Unexpected error: {e}")
