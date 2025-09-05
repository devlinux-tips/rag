#!/usr/bin/env python3
"""
Script to clear generated data from multilingual RAG system
- ChromaDB databases (all or language-specific)
- Temporary cache files
- Embedding caches
- Language-specific processed data

Usage:
    python clear_data.py [all|hr|en|multilingual]

Examples:
    python clear_data.py          # Clear all data (default)
    python clear_data.py all      # Clear all data
    python clear_data.py hr       # Clear only Croatian data
    python clear_data.py en       # Clear only English data
    python clear_data.py multilingual  # Clear only multilingual data
"""
import argparse
import os
import shutil
import sys
from pathlib import Path


def clear_chromadb(language=None):
    """Clear ChromaDB databases.

    Args:
        language: If specified, clear only language-specific collections.
                 If None, clear all databases.
    """
    db_paths = ["./data/chromadb", "./temp_chroma_db", "./chromadb"]

    cleared = []
    for db_path in db_paths:
        path = Path(db_path)
        if path.exists():
            if language and language != "all":
                # For language-specific clearing, we would need to clear specific collections
                # Since ChromaDB stores collections in complex folder structures,
                # we'll note this and clear the entire DB for now
                print(f"⚠️  ChromaDB language-specific clearing not implemented yet")
                print(
                    f"   To clear {language} data, consider clearing all ChromaDB and re-processing"
                )
                continue

            try:
                shutil.rmtree(path)
                cleared.append(db_path)
                print(f"✅ Cleared ChromaDB: {db_path}")
            except Exception as e:
                print(f"❌ Error clearing {db_path}: {e}")
        else:
            print(f"ℹ️  ChromaDB not found: {db_path}")

    return cleared


def clear_temp_cache(language=None):
    """Clear temporary embedding caches.

    Args:
        language: If specified, clear only language-specific caches.
    """
    cache_paths = ["./temp_cache", "./.cache", "./cache", "./test_cache"]

    cleared = []
    for cache_path in cache_paths:
        path = Path(cache_path)
        if path.exists():
            if language and language != "all":
                # Look for language-specific cache folders
                lang_cache = path / language
                if lang_cache.exists():
                    try:
                        shutil.rmtree(lang_cache)
                        cleared.append(str(lang_cache))
                        print(f"✅ Cleared {language} cache: {lang_cache}")
                    except Exception as e:
                        print(f"❌ Error clearing {lang_cache}: {e}")
                else:
                    print(f"ℹ️  {language} cache not found: {lang_cache}")
            else:
                try:
                    shutil.rmtree(path)
                    cleared.append(cache_path)
                    print(f"✅ Cleared cache: {cache_path}")
                except Exception as e:
                    print(f"❌ Error clearing {cache_path}: {e}")
        else:
            print(f"ℹ️  Cache not found: {cache_path}")

    return cleared


def clear_processed_data(language=None):
    """Clear processed document data.

    Args:
        language: If specified, clear only language-specific processed data.
    """
    processed_paths = ["./data/processed", "./processed_docs"]

    cleared = []
    for proc_path in processed_paths:
        path = Path(proc_path)
        if path.exists():
            if language and language != "all":
                # Clear language-specific processed data
                lang_path = path / language
                if lang_path.exists():
                    try:
                        shutil.rmtree(lang_path)
                        cleared.append(str(lang_path))
                        print(f"✅ Cleared {language} processed data: {lang_path}")
                    except Exception as e:
                        print(f"❌ Error clearing {lang_path}: {e}")
                else:
                    print(f"ℹ️  {language} processed data not found: {lang_path}")
            else:
                try:
                    shutil.rmtree(path)
                    cleared.append(proc_path)
                    print(f"✅ Cleared processed data: {proc_path}")
                except Exception as e:
                    print(f"❌ Error clearing {proc_path}: {e}")
        else:
            print(f"ℹ️  Processed data not found: {proc_path}")

    return cleared


def clear_vectordb_data(language=None):
    """Clear vector database data.

    Args:
        language: If specified, clear only language-specific vector data.
    """
    vectordb_paths = ["./data/vectordb", "./vectordb"]

    cleared = []
    for vdb_path in vectordb_paths:
        path = Path(vdb_path)
        if path.exists():
            if language and language != "all":
                # Clear language-specific vector data
                lang_path = path / language
                if lang_path.exists():
                    try:
                        shutil.rmtree(lang_path)
                        cleared.append(str(lang_path))
                        print(f"✅ Cleared {language} vector data: {lang_path}")
                    except Exception as e:
                        print(f"❌ Error clearing {lang_path}: {e}")
                else:
                    print(f"ℹ️  {language} vector data not found: {lang_path}")
            else:
                try:
                    shutil.rmtree(path)
                    cleared.append(vdb_path)
                    print(f"✅ Cleared vector database: {vdb_path}")
                except Exception as e:
                    print(f"❌ Error clearing {vdb_path}: {e}")
        else:
            print(f"ℹ️  Vector database not found: {vdb_path}")

    return cleared


def clear_huggingface_cache():
    """Clear HuggingFace transformers cache."""
    hf_cache_paths = [
        Path.home() / ".cache" / "huggingface" / "transformers",
        Path.home() / ".cache" / "torch" / "sentence_transformers",
    ]

    cleared = []
    for cache_path in hf_cache_paths:
        if cache_path.exists():
            try:
                # Only clear sentence transformer models we might have downloaded
                if "sentence_transformers" in str(cache_path):
                    for model_dir in cache_path.iterdir():
                        if model_dir.is_dir() and any(
                            x in model_dir.name.lower()
                            for x in ["distiluse", "multilingual", "bge-m3"]
                        ):
                            shutil.rmtree(model_dir)
                            cleared.append(str(model_dir))
                            print(f"✅ Cleared HF model cache: {model_dir.name}")
                else:
                    print(f"ℹ️  Skipping large HF cache: {cache_path}")
            except Exception as e:
                print(f"❌ Error clearing HF cache {cache_path}: {e}")
        else:
            print(f"ℹ️  HF cache not found: {cache_path}")

    return cleared


def get_supported_languages():
    """Get list of supported languages from data/raw structure."""
    raw_path = Path("./data/raw")
    languages = ["all"]

    if raw_path.exists():
        for item in raw_path.iterdir():
            if item.is_dir() and item.name not in [".git", "__pycache__"]:
                languages.append(item.name)

    return languages


def main():
    """Main cleanup function with language-specific support."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Clear generated data from multilingual RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Clear all data (default)
  %(prog)s all          Clear all data
  %(prog)s hr           Clear only Croatian data
  %(prog)s en           Clear only English data
  %(prog)s multilingual Clear only multilingual data
        """,
    )

    supported_langs = get_supported_languages()
    parser.add_argument(
        "language",
        nargs="?",
        default="all",
        choices=supported_langs,
        help=f"Language to clear data for. Choices: {', '.join(supported_langs)}",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleared without actually clearing it",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No data will actually be cleared")
        print("=" * 50)

    # Display what we're clearing
    if args.language == "all":
        print("🧹 Multilingual RAG Data Cleanup - ALL LANGUAGES")
        print("🌍 This will clear data for all languages (hr, en, multilingual)")
    else:
        print(f"🧹 Multilingual RAG Data Cleanup - {args.language.upper()}")
        print(f"🎯 This will clear data only for: {args.language}")

    print("=" * 60)

    total_cleared = 0

    # Clear ChromaDB
    print(f"\n📊 Clearing ChromaDB databases ({args.language})...")
    if not args.dry_run:
        cleared = clear_chromadb(args.language)
        total_cleared += len(cleared)
    else:
        print(f"   Would clear ChromaDB for: {args.language}")

    # Clear temporary caches
    print(f"\n💾 Clearing temporary caches ({args.language})...")
    if not args.dry_run:
        cleared = clear_temp_cache(args.language)
        total_cleared += len(cleared)
    else:
        print(f"   Would clear temp caches for: {args.language}")

    # Clear processed data
    print(f"\n📄 Clearing processed document data ({args.language})...")
    if not args.dry_run:
        cleared = clear_processed_data(args.language)
        total_cleared += len(cleared)
    else:
        print(f"   Would clear processed data for: {args.language}")

    # Clear vector database
    print(f"\n🔍 Clearing vector database ({args.language})...")
    if not args.dry_run:
        cleared = clear_vectordb_data(args.language)
        total_cleared += len(cleared)
    else:
        print(f"   Would clear vector database for: {args.language}")

    # Clear HuggingFace cache (only for 'all')
    if args.language == "all":
        print("\n🤗 Checking HuggingFace caches...")
        if not args.dry_run:
            cleared = clear_huggingface_cache()
            total_cleared += len(cleared)
        else:
            print("   Would clear HuggingFace model caches")

    if args.dry_run:
        print(f"\n🔍 DRY RUN COMPLETE")
        print("   Use without --dry-run to actually clear the data")
    else:
        print(f"\n🎯 Cleanup complete! Cleared {total_cleared} directories/files")

    print(f"\nThe following data is preserved:")
    if args.language == "all":
        print("  • Raw documents in data/raw/ (all languages)")
    else:
        print(f"  • Raw documents in data/raw/ (including {args.language})")
        print(f"  • All data for other languages")
    print("  • Source code in src/")
    print("  • Configuration files")

    print(f"\n💡 Next steps:")
    if args.language == "all":
        print("  • Run your RAG system to regenerate fresh embeddings for all languages")
        print("  • Process documents again with: python rag.py")
    else:
        print(f"  • Run your RAG system to regenerate {args.language} embeddings")
        print(f"  • Process {args.language} documents again")

    print(f"\n🌍 Available languages: {', '.join(get_supported_languages())}")


if __name__ == "__main__":
    main()
