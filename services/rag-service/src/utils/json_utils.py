"""
Shared JSON utilities to eliminate code duplication.
Centralized logging functions with proper markdown indentation preservation.
"""

import json
from typing import Any


def write_json_with_readable_content(file_path: str, data: dict[str, Any], content_field: str = "content") -> None:
    """
    Write JSON file with human-readable formatting for content fields.
    FIXED VERSION - preserves original markdown indentation.

    Args:
        file_path: Path to write JSON file
        data: Dictionary containing data to write
        content_field: Field name containing content to format specially
    """
    with open(file_path, "w", encoding="utf-8") as f:
        # Handle the case where the content field might not exist
        if content_field not in data:
            json.dump(data, f, indent=2, ensure_ascii=False)
            return

        content = data[content_field]
        data_copy = data.copy()

        # Write JSON structure manually to preserve readable content formatting
        f.write("{\n")

        # Write all fields except content first
        fields_written = 0
        for key, value in data_copy.items():
            if key == content_field:
                continue

            if fields_written > 0:
                f.write(",\n")

            if isinstance(value, str):
                f.write(f'  "{key}": "{value}"')
            elif isinstance(value, dict):
                formatted_dict = json.dumps(value, indent=4).replace("\n", "\n    ")
                f.write(f'  "{key}": {formatted_dict}')
            elif isinstance(value, (int, float)):
                f.write(f'  "{key}": {value}')
            else:
                f.write(f'  "{key}": {json.dumps(value)}')
            fields_written += 1

        # Write content field with preserved formatting - FIXED VERSION
        if fields_written > 0:
            f.write(",\n")
        f.write(f'  "{content_field}": "')

        # FIXED: Preserve original indentation while ensuring valid JSON
        escaped_content = content.replace("\\", "\\\\").replace('"', '\\"')

        # Split by lines and preserve original indentation
        lines = escaped_content.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                f.write("\\n\n  ")  # Add JSON indentation only
            f.write(line)  # Preserve original line indentation

        f.write('"\n')
        f.write("}\n")


def write_debug_json(file_path: str, data: dict[str, Any], content_field: str = "content") -> None:
    """
    Alias for write_json_with_readable_content for debug logging.
    Maintains backward compatibility while centralizing the implementation.
    """
    write_json_with_readable_content(file_path, data, content_field)
