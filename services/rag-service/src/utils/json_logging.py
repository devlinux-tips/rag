"""
Centralized JSON logging utilities.
Single source of truth for debug JSON writing.
"""

import json
from typing import Any


def write_debug_json(file_path: str, data: dict[str, Any]) -> None:
    """
    Write debug JSON file with proper formatting.

    Args:
        file_path: Path to write JSON file
        data: Dictionary to write as JSON
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
