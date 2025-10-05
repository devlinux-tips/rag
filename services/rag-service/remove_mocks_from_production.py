#!/usr/bin/env python3
"""
Remove all mock classes and functions from production files.
This removes the blocks that were extracted to conftest.py.
"""

import re
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path("/home/x/src/rag/learn-rag")
SERVICE_ROOT = REPO_ROOT / "services/rag-service"


def find_mock_section_boundaries(lines: List[str]) -> Tuple[int, int]:
    """
    Find the start and end of the MOCK PROVIDERS section.
    Returns (start_index, end_index) or (-1, -1) if not found.
    """
    start_idx = -1
    end_idx = -1

    for i, line in enumerate(lines):
        # Look for section markers
        if "MOCK PROVIDERS" in line or "MOCK" in line and "====" in line:
            start_idx = i
        elif start_idx != -1 and ("PRODUCTION" in line or "STANDARD" in line) and "====" in line:
            end_idx = i
            break

    return (start_idx, end_idx)


def remove_mock_section(file_path: Path) -> bool:
    """
    Remove the entire MOCK PROVIDERS section from a file.
    Returns True if section was removed, False if not found.
    """
    content = file_path.read_text()
    lines = content.split("\n")

    start_idx, end_idx = find_mock_section_boundaries(lines)

    if start_idx == -1:
        # No mock section found, try to find individual mock items
        return remove_individual_mocks(file_path)

    # Remove section
    print(f"    Removing MOCK section: lines {start_idx + 1} to {end_idx}")
    new_lines = lines[:start_idx] + lines[end_idx:]

    file_path.write_text("\n".join(new_lines))
    return True


def remove_individual_mocks(file_path: Path) -> bool:
    """
    Remove individual mock classes and functions (for files without section markers).
    """
    content = file_path.read_text()
    lines = content.split("\n")

    # Find all mock blocks to remove
    mocks_to_remove = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if re.match(r"^class Mock[A-Z]", line) or re.match(r"^def create_mock_", line):
            start_idx = i

            # Find end of block
            i += 1
            while i < len(lines):
                next_line = lines[i]

                if (
                    next_line.startswith("class ")
                    and not next_line.startswith("class Mock")
                    or next_line.startswith("def ")
                    and not next_line.startswith("def create_mock")
                    and not next_line.strip().startswith("def ")
                    or next_line.startswith("# ===")
                ):
                    break

                i += 1

            end_idx = i
            mocks_to_remove.append((start_idx, end_idx))
        else:
            i += 1

    if not mocks_to_remove:
        print(f"    No mocks found")
        return False

    # Remove mocks in reverse order (to preserve indices)
    for start, end in reversed(mocks_to_remove):
        print(f"    Removing mock: lines {start + 1} to {end}")
        del lines[start:end]

    file_path.write_text("\n".join(lines))
    return True


def main():
    """Main removal process."""
    print("=" * 60)
    print("REMOVING MOCKS FROM PRODUCTION FILES")
    print("=" * 60)

    files_to_process = [
        # Utils
        "src/utils/language_manager_providers.py",
        "src/utils/folder_manager_providers.py",
        # Generation
        "src/generation/prompt_templates.py",
        "src/generation/response_parser.py",
        "src/generation/enhanced_prompt_templates_providers.py",
        "src/generation/http_clients.py",
        "src/generation/language_providers.py",
        # Preprocessing
        "src/preprocessing/extractors_providers.py",
        "src/preprocessing/cleaners_providers.py",
        # Retrieval
        "src/retrieval/hybrid_retriever.py",
        "src/retrieval/ranker.py",
        "src/retrieval/query_processor_providers.py",
        "src/retrieval/hierarchical_retriever_providers.py",
        "src/retrieval/categorization_providers.py",
        "src/retrieval/ranker_providers.py",
        "src/retrieval/reranker.py",
        # Vectordb
        "src/vectordb/search_providers.py",
        "src/vectordb/storage.py",
        "src/vectordb/chromadb_factories.py",
        "src/vectordb/embedding_loaders.py",
        "src/vectordb/embedding_devices.py",
        # CLI
        "src/cli/rag_cli.py",
    ]

    removed_count = 0

    for file_rel_path in files_to_process:
        file_path = SERVICE_ROOT / file_rel_path
        print(f"\n  Processing: {file_rel_path}")

        if remove_mock_section(file_path):
            removed_count += 1

    print(f"\n  ✓ Removed mocks from {removed_count} files")

    print("\n" + "=" * 60)
    print("REMOVAL COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify production files are clean")
    print("2. Update test imports")
    print("3. Run tests")


if __name__ == "__main__":
    main()
