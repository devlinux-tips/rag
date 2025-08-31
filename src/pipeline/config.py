"""
Configuration management for Croatian RAG system.
"""
import os
import yaml
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    max_chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    sentence_chunk_overlap: int = 2
    preserve_paragraphs: bool = True


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    cache_folder: str = "./models/embeddings"
    batch_size: int = 32
    max_seq_length: int = 512


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""
    db_path: str = "./data/vectordb"
    collection_name: str = "croatian_documents"
    distance_metric: str = "cosine"
    ef_construction: int = 200
    m: int = 16


@dataclass
class RetrievalConfig:
    """Retrieval system configuration."""
    default_k: int = 5
    max_k: int = 10
    min_similarity_score: float = 0.3
    adaptive_retrieval: bool = True
    enable_reranking: bool = True
    diversity_lambda: float = 0.3


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    top_k: int = 40
    timeout: float = 60.0
    preserve_diacritics: bool = True
    prefer_formal_style: bool = True
    include_cultural_context: bool = True


@dataclass
class CroatianConfig:
    """Croatian language specific configuration."""
    enable_morphological_expansion: bool = True
    enable_synonym_expansion: bool = True
    enable_cultural_context: bool = True
    stop_words_file: str = "config/croatian_stop_words.txt"
    morphology_patterns_file: str = "config/croatian_morphology.json"
    cultural_context_file: str = "config/croatian_cultural_context.json"


class RAGConfig(BaseSettings):
    """Main configuration for Croatian RAG system."""
    
    # Component configurations
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    croatian: CroatianConfig = Field(default_factory=CroatianConfig)
    
    # System settings
    log_level: str = "INFO"
    enable_caching: bool = True
    cache_dir: str = "./data/cache"
    
    # Performance settings
    max_concurrent_requests: int = 5
    request_timeout: float = 120.0
    enable_metrics: bool = True
    metrics_dir: str = "./data/metrics"
    
    # Data paths
    documents_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    test_data_dir: str = "./data/test"
    
    class Config:
        env_file = ".env"
        extra = "allow"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
        """Create config from dictionary."""
        config = cls()
        
        # Update nested configs
        if 'processing' in data:
            config.processing = ProcessingConfig(**data['processing'])
        if 'embedding' in data:
            config.embedding = EmbeddingConfig(**data['embedding'])
        if 'chroma' in data:
            config.chroma = ChromaConfig(**data['chroma'])
        if 'retrieval' in data:
            config.retrieval = RetrievalConfig(**data['retrieval'])
        if 'ollama' in data:
            config.ollama = OllamaConfig(**data['ollama'])
        if 'croatian' in data:
            config.croatian = CroatianConfig(**data['croatian'])
        
        # Update other fields
        for key, value in data.items():
            if key not in ['processing', 'embedding', 'chroma', 'retrieval', 'ollama', 'croatian']:
                setattr(config, key, value)
        
        return config


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}
