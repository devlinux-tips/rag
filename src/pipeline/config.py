"""
Configuration management for Croatian RAG system.
"""
import os
import yaml
from pathlib import Path
from pydantic import BaseSettings
from typing import Dict, Any


class RAGConfig(BaseSettings):
    """Main configuration for RAG system."""
    
    # API Configuration
    anthropic_api_key: str = ""
    claude_model: str = "claude-3-sonnet-20240229"
    
    # Database Configuration
    chroma_db_path: str = "./data/vectordb"
    chroma_collection_name: str = "croatian_documents"
    
    # Embedding Configuration
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Processing Configuration
    max_chunk_size: int = 512
    chunk_overlap: int = 50
    
    class Config:
        env_file = ".env"


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}
