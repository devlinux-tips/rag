#!/usr/bin/env python3
"""
Script to clear all generated data from Croatian RAG system
- ChromaDB databases
- Temporary cache files
- Embedding caches
"""
import os
import shutil
from pathlib import Path

def clear_chromadb():
    """Clear all ChromaDB databases."""
    db_paths = [
        "./data/chromadb",
        "./temp_chroma_db", 
        "./chromadb"
    ]
    
    cleared = []
    for db_path in db_paths:
        path = Path(db_path)
        if path.exists():
            try:
                shutil.rmtree(path)
                cleared.append(db_path)
                print(f"✅ Cleared ChromaDB: {db_path}")
            except Exception as e:
                print(f"❌ Error clearing {db_path}: {e}")
        else:
            print(f"ℹ️  ChromaDB not found: {db_path}")
    
    return cleared

def clear_temp_cache():
    """Clear temporary embedding caches."""
    cache_paths = [
        "./temp_cache",
        "./.cache",
        "./cache"
    ]
    
    cleared = []
    for cache_path in cache_paths:
        path = Path(cache_path)
        if path.exists():
            try:
                shutil.rmtree(path)
                cleared.append(cache_path)
                print(f"✅ Cleared cache: {cache_path}")
            except Exception as e:
                print(f"❌ Error clearing {cache_path}: {e}")
        else:
            print(f"ℹ️  Cache not found: {cache_path}")
    
    return cleared

def clear_processed_data():
    """Clear processed document data."""
    processed_paths = [
        "./data/processed",
        "./processed_docs"
    ]
    
    cleared = []
    for proc_path in processed_paths:
        path = Path(proc_path)
        if path.exists():
            try:
                shutil.rmtree(path)
                cleared.append(proc_path)
                print(f"✅ Cleared processed data: {proc_path}")
            except Exception as e:
                print(f"❌ Error clearing {proc_path}: {e}")
        else:
            print(f"ℹ️  Processed data not found: {proc_path}")
    
    return cleared

def clear_huggingface_cache():
    """Clear HuggingFace transformers cache."""
    hf_cache_paths = [
        Path.home() / ".cache" / "huggingface" / "transformers",
        Path.home() / ".cache" / "torch" / "sentence_transformers"
    ]
    
    cleared = []
    for cache_path in hf_cache_paths:
        if cache_path.exists():
            try:
                # Only clear sentence transformer models we might have downloaded
                if "sentence_transformers" in str(cache_path):
                    for model_dir in cache_path.iterdir():
                        if model_dir.is_dir() and any(x in model_dir.name.lower() for x in ["distiluse", "multilingual", "bge-m3"]):
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

def main():
    """Main cleanup function."""
    print("🧹 Croatian RAG Data Cleanup")
    print("=" * 40)
    
    total_cleared = 0
    
    # Clear ChromaDB
    print("\n📊 Clearing ChromaDB databases...")
    cleared = clear_chromadb()
    total_cleared += len(cleared)
    
    # Clear temporary caches
    print("\n💾 Clearing temporary caches...")
    cleared = clear_temp_cache()
    total_cleared += len(cleared)
    
    # Clear processed data
    print("\n📄 Clearing processed document data...")
    cleared = clear_processed_data()
    total_cleared += len(cleared)
    
    # Clear HuggingFace cache (optional)
    print("\n🤗 Checking HuggingFace caches...")
    cleared = clear_huggingface_cache()
    total_cleared += len(cleared)
    
    print(f"\n🎯 Cleanup complete! Cleared {total_cleared} directories/files")
    print("\nThe following data is preserved:")
    print("  • Raw documents in data/raw/")
    print("  • Source code in src/")
    print("  • Configuration files")
    
    print("\n💡 Next steps:")
    print("  • Run your RAG system to regenerate fresh embeddings")
    print("  • Process documents again with: python rag.py -> process")

if __name__ == "__main__":
    main()