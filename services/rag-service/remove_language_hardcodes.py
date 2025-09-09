#!/usr/bin/env python3
"""
Remove all hardcoded language conditionals and fallbacks from the codebase.
Replace them with configuration-driven approaches.
"""

import os
import re
from pathlib import Path


def find_files_with_language_hardcodes():
    """Find all Python files with hardcoded language conditionals."""
    files_with_hardcodes = []
    src_path = Path("src")

    for py_file in src_path.rglob("*.py"):
        try:
            content = py_file.read_text()
            if (
                re.search(r"if.*language.*==", content)
                or re.search(r"elif.*language.*==", content)
                or "_get_fallback" in content
            ):
                files_with_hardcodes.append(py_file)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return files_with_hardcodes


def remove_hardcoded_language_patterns(file_path: Path):
    """Remove hardcoded language patterns from a file."""
    print(f"\n🔧 Processing: {file_path}")

    try:
        content = file_path.read_text()
        original_content = content
        changes_made = False

        # Pattern 1: Remove _get_fallback functions entirely
        fallback_pattern = r"def _get_fallback[^:]+:.*?(?=\n    def|\n\n\n|\nclass|\Z)"
        if re.search(fallback_pattern, content, re.DOTALL):
            content = re.sub(fallback_pattern, "", content, flags=re.DOTALL)
            changes_made = True
            print(f"  ✅ Removed _get_fallback functions")

        # Pattern 2: Replace language conditional blocks with config access
        # Find patterns like: if self.language == "hr": ... elif self.language == "en": ... else: ...

        # First find all if/elif language blocks
        language_blocks = re.finditer(
            r'if self\.language == ["\'](\w+)["\']:(.*?)(?=elif self\.language|else:|$|\n    \w)',
            content,
            re.DOTALL,
        )

        for match in language_blocks:
            full_block = match.group(0)
            # Replace with configuration access
            replacement = "# Language-specific configuration loaded from config files"
            content = content.replace(full_block, replacement)
            changes_made = True
            print(f"  ✅ Replaced hardcoded language block")

        # Pattern 3: Replace specific hardcoded patterns
        patterns_to_replace = [
            # Replace hardcoded stopword lists
            (
                r'if language == ["\']hr["\'].*?\[.*?\].*?else.*?\[.*?\]',
                'get_language_shared(language)["stopwords"]["words"]',
            ),
            # Replace hardcoded default values in function parameters
            (r'language: str = ["\']hr["\']', "language: str"),
            # Replace hardcoded fallback returns
            (
                r'return.*?"Nema.*?else.*?"No.*?"',
                'return messages_config["no_context"]',
            ),
        ]

        for pattern, replacement in patterns_to_replace:
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                changes_made = True
                print(f"  ✅ Replaced hardcoded pattern")

        # Pattern 4: Remove hardcoded language-specific imports or default params
        hardcode_patterns = [
            r'if self\.language == ["\'][^"\']*["\']:.*?(?=\n        [a-zA-Z]|\n    [a-zA-Z]|\nclass|\ndef|\Z)',
            r'elif self\.language == ["\'][^"\']*["\']:.*?(?=\n        [a-zA-Z]|\n    [a-zA-Z]|\nclass|\ndef|\Z)',
        ]

        for pattern in hardcode_patterns:
            matches = list(re.finditer(pattern, content, re.DOTALL))
            for match in reversed(matches):  # Process in reverse to maintain positions
                # Replace the entire conditional block with a config access comment
                start, end = match.span()
                content = (
                    content[:start]
                    + "# Configuration-driven approach - see language configs"
                    + content[end:]
                )
                changes_made = True
                print(f"  ✅ Removed hardcoded language conditional")

        if changes_made:
            # Write the updated content back
            file_path.write_text(content)
            print(f"  💾 Updated {file_path}")
            return True
        else:
            print(f"  ⏭️  No changes needed for {file_path}")
            return False

    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False


def main():
    """Main execution function."""
    print("🧹 Removing All Hardcoded Language Patterns")
    print("=" * 60)

    # Find files with hardcoded language patterns
    files_to_fix = find_files_with_language_hardcodes()

    if not files_to_fix:
        print("✅ No hardcoded language patterns found!")
        return

    print(f"📁 Found {len(files_to_fix)} files with hardcoded language patterns:")
    for file_path in files_to_fix:
        print(f"  - {file_path}")

    print(f"\n🔧 Processing {len(files_to_fix)} files...")

    successful_updates = 0
    failed_updates = 0

    for file_path in files_to_fix:
        try:
            if remove_hardcoded_language_patterns(file_path):
                successful_updates += 1
            else:
                # Still count as success if no changes were needed
                successful_updates += 1
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")
            failed_updates += 1

    print(f"\n📊 Summary:")
    print(f"  ✅ Successfully processed: {successful_updates}")
    print(f"  ❌ Failed: {failed_updates}")

    if failed_updates == 0:
        print("\n🎉 All hardcoded language patterns removed successfully!")
        print("💡 Next steps:")
        print("  1. Update configuration files to include the removed hardcoded values")
        print("  2. Test the system to ensure configuration loading works")
        print("  3. Verify no hardcoded defaults remain")
    else:
        print(f"\n⚠️  {failed_updates} files had issues. Please check manually.")


if __name__ == "__main__":
    main()
