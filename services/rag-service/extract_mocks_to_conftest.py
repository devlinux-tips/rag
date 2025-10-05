#!/usr/bin/env python3
"""
Extract all mock classes and functions from production files to conftest.py.
This script systematically moves ALL test mocks out of production code.
"""

import re
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path("/home/x/src/rag/learn-rag")
SERVICE_ROOT = REPO_ROOT / "services/rag-service"
CONFTEST_PATH = SERVICE_ROOT / "tests/conftest.py"


def find_mock_blocks(file_path: Path) -> List[Tuple[int, int, str]]:
    """
    Find all mock class/function blocks in a file.
    Returns list of (start_line, end_line, block_text) tuples.
    """
    content = file_path.read_text()
    lines = content.split("\n")

    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Match mock class or mock function
        if re.match(r"^class Mock[A-Z]", line) or re.match(r"^def create_mock_", line):
            start_line = i + 1  # 1-indexed
            block_lines = [line]

            # Find end of block (next class/function or end of file)
            i += 1
            while i < len(lines):
                next_line = lines[i]

                # Stop at next top-level definition or section marker
                if (
                    next_line.startswith("class ")
                    and not next_line.startswith("class Mock")
                    or next_line.startswith("def ")
                    and not next_line.startswith("def create_mock")
                    and not next_line.strip().startswith("def ")  # Not indented method
                    or next_line.startswith("# ===")  # Section marker
                ):
                    break

                block_lines.append(next_line)
                i += 1

            end_line = i  # Last line number (1-indexed)
            block_text = "\n".join(block_lines)
            blocks.append((start_line, end_line, block_text))
        else:
            i += 1

    return blocks


def extract_file_mocks(file_path: Path) -> List[str]:
    """Extract all mock blocks from a file and return as list of code strings."""
    print(f"\n  Extracting from: {file_path.relative_to(SERVICE_ROOT)}")

    blocks = find_mock_blocks(file_path)

    if not blocks:
        print(f"    No mocks found")
        return []

    mock_code = []
    for start, end, block in blocks:
        # Extract mock name from first line
        first_line = block.split("\n")[0]
        if "class" in first_line:
            name = re.search(r"class (Mock\w+)", first_line).group(1)
        else:
            name = re.search(r"def (create_mock_\w+)", first_line).group(1)

        print(f"    Found: {name} (lines {start}-{end})")
        mock_code.append(block)

    return mock_code


def main():
    """Main extraction process."""
    print("=" * 60)
    print("MOCK EXTRACTION TO CONFTEST.PY")
    print("=" * 60)

    # Files to process
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

    # Collect all mock code by category
    mocks_by_category = {
        "UTILS": [],
        "GENERATION": [],
        "PREPROCESSING": [],
        "RETRIEVAL": [],
        "VECTORDB": [],
        "CLI": [],
    }

    total_mocks = 0

    for file_rel_path in files_to_process:
        file_path = SERVICE_ROOT / file_rel_path

        # Determine category
        if "utils" in file_rel_path:
            category = "UTILS"
        elif "generation" in file_rel_path:
            category = "GENERATION"
        elif "preprocessing" in file_rel_path:
            category = "PREPROCESSING"
        elif "retrieval" in file_rel_path:
            category = "RETRIEVAL"
        elif "vectordb" in file_rel_path:
            category = "VECTORDB"
        elif "cli" in file_rel_path:
            category = "CLI"
        else:
            category = "OTHER"

        mocks = extract_file_mocks(file_path)
        mocks_by_category[category].extend(mocks)
        total_mocks += len(mocks)

    print(f"\n  Total mocks found: {total_mocks}")

    # Read current conftest
    conftest_content = CONFTEST_PATH.read_text()

    # Append mocks to conftest by category
    print("\n  Appending to conftest.py...")

    new_sections = []

    for category, mocks in mocks_by_category.items():
        if not mocks:
            continue

        new_sections.append(f"\n# ============================================================================")
        new_sections.append(f"# {category} MOCKS")
        new_sections.append(f"# ============================================================================")
        new_sections.append("")

        for mock_code in mocks:
            new_sections.append(mock_code)
            new_sections.append("")  # Blank line between mocks

    # Write updated conftest
    updated_conftest = conftest_content.rstrip() + "\n\n" + "\n".join(new_sections)
    CONFTEST_PATH.write_text(updated_conftest)

    print(f"  ✓ Appended {total_mocks} mocks to conftest.py")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review conftest.py")
    print("2. Remove mocks from production files")
    print("3. Update test imports")
    print("4. Run tests")


if __name__ == "__main__":
    main()
