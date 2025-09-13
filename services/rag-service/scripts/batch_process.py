#!/usr/bin/env python3
"""
Multilingual Batch Processing Script

This script demonstrates how to process documents using the new language-based
folder structure for the multilingual RAG system.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rag_system import RAGSystem, create_rag_system


async def process_language_documents(
    language: str,
    source_dir: Optional[str] = None,
    file_patterns: Optional[list[str]] = None,
) -> dict:
    """
    Process all documents for a specific language.

    Args:
        language: Language code (hr, en, etc.)
        source_dir: Custom source directory (defaults to data/raw/{language}/)
        file_patterns: File patterns to include (defaults to all supported formats)

    Returns:
        Processing results dictionary
    """
    # Default source directory based on language
    if source_dir is None:
        source_dir = f"data/raw/{language}"

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")

    # Find all supported document files
    supported_extensions = [".pdf", ".docx", ".txt", ".html", ".md"]
    if file_patterns:
        # Use custom patterns
        files = []
        for pattern in file_patterns:
            files.extend(source_path.glob(pattern))
    else:
        # Use default extensions
        files = []
        for ext in supported_extensions:
            files.extend(source_path.glob(f"*{ext}"))

    if not files:
        print(f"⚠️  No documents found in {source_dir}")
        return {"processed": 0, "failed": 0, "files": []}

    print(f"📁 Found {len(files)} documents in {source_dir}")
    print(f"🗣️  Processing with language: {language}")

    # Create RAG system for this language
    system = await create_rag_system(language=language)

    try:
        # Convert file paths to strings
        file_paths = [str(f) for f in files]

        # Process documents
        print(f"🔄 Processing {len(file_paths)} documents...")
        result = await system.add_documents(file_paths)

        print("✅ Processing complete!")
        print(f"   📄 Processed: {result.get('processed_documents', 0)}")
        print(f"   ❌ Failed: {result.get('failed_documents', 0)}")
        print(f"   🔢 Total chunks: {result.get('total_chunks', 0)}")
        print(f"   ⏱️  Time: {result.get('processing_time', 0):.2f}s")

        return {
            "processed": result.get("processed_documents", 0),
            "failed": result.get("failed_documents", 0),
            "files": file_paths,
            "result": result,
        }

    finally:
        await system.close()


async def batch_process_all_languages(languages: list[str]) -> dict:
    """
    Process documents for multiple languages.

    Args:
        languages: List of language codes to process

    Returns:
        Combined processing results
    """
    all_results = {}
    total_processed = 0
    total_failed = 0

    for language in languages:
        print(f"\n🌍 Processing language: {language.upper()}")
        print("=" * 50)

        try:
            result = await process_language_documents(language)
            all_results[language] = result
            total_processed += result["processed"]
            total_failed += result["failed"]

        except Exception as e:
            print(f"❌ Error processing {language}: {e}")
            all_results[language] = {"error": str(e)}

    print("\n📊 SUMMARY")
    print("=" * 50)
    print(f"Languages processed: {len(languages)}")
    print(f"Total documents processed: {total_processed}")
    print(f"Total documents failed: {total_failed}")

    return all_results


async def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(description="Multilingual RAG Batch Processor")
    parser.add_argument(
        "--language",
        "-l",
        help="Process documents for specific language (hr, en, etc.)",
    )
    parser.add_argument(
        "--all-languages",
        "-a",
        action="store_true",
        help="Process documents for all supported languages",
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        help="Custom source directory (overrides default language-based path)",
    )
    parser.add_argument(
        "--patterns",
        "-p",
        nargs="+",
        help="File patterns to include (e.g., '*.pdf' '*.docx')",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🚀 Multilingual RAG Batch Processor")
    print("=" * 50)

    try:
        if args.all_languages:
            # Process all supported languages
            supported_languages = ["hr", "en"]  # Can be loaded from config
            await batch_process_all_languages(supported_languages)

        elif args.language:
            # Process specific language
            await process_language_documents(
                language=args.language,
                source_dir=args.source_dir,
                file_patterns=args.patterns,
            )

        else:
            print("❌ Please specify --language or --all-languages")
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        return 1

    print("\n✅ Batch processing completed successfully!")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
