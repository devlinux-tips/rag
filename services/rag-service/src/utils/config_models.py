"""
Clean Configuration Models for RAG System
Dataclasses that use validated configuration - no .get() fallbacks needed.

These models represent Phase 2 of the ConfigValidator system:
- ConfigValidator (Phase 1) guarantees all keys exist
- These models (Phase 2) use direct dictionary access safely

Author: RAG System Architecture
Status: Production Implementation
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types for model execution."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class ChunkingStrategy(Enum):
    """Text chunking strategies."""

    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


class RankingMethod(Enum):
    """Ranking methods for retrieval results."""

    BASIC = "basic"
    LANGUAGE_ENHANCED = "language_enhanced"
    SEMANTIC_BOOST = "semantic_boost"


@dataclass
class QueryProcessingConfig:
    """
    Configuration for query processing - no fallbacks needed.
    ConfigValidator guarantees all keys exist before this is created.
    """

    language: str
    expand_synonyms: bool
    normalize_case: bool
    remove_stopwords: bool
    min_query_length: int
    max_query_length: int
    max_expanded_terms: int
    enable_morphological_analysis: bool
    use_query_classification: bool
    enable_spell_check: bool

    @classmethod
    def from_validated_config(cls, main_config: dict, language: str) -> "QueryProcessingConfig":
        """
        Create config from validated configuration.
        Uses direct dictionary access - ConfigValidator guarantees existence.
        """
        query_config = main_config["query_processing"]  # Direct access - guaranteed to exist

        return cls(
            language=language,
            expand_synonyms=query_config["expand_synonyms"],
            normalize_case=query_config["normalize_case"],
            remove_stopwords=query_config["remove_stopwords"],
            min_query_length=query_config["min_query_length"],
            max_query_length=query_config["max_query_length"],
            max_expanded_terms=query_config["max_expanded_terms"],
            enable_morphological_analysis=query_config["enable_morphological_analysis"],
            use_query_classification=query_config["use_query_classification"],
            enable_spell_check=query_config["enable_spell_check"],
        )


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding models - clean DI implementation.
    """

    model_name: str
    device: str
    max_seq_length: int
    batch_size: int
    normalize_embeddings: bool
    use_safetensors: bool
    trust_remote_code: bool
    torch_dtype: str
    cache_dir: str

    @classmethod
    def from_validated_config(
        cls, main_config: dict, language_config: dict[Any, Any] | None = None
    ) -> "EmbeddingConfig":
        """Create config from validated main configuration with optional language-specific overrides."""
        embed_config = main_config["embeddings"]  # Direct access

        # Language-specific overrides if provided
        if language_config and "embeddings" in language_config:
            lang_embed_config = language_config["embeddings"]
            # Merge with language-specific overrides taking precedence
            merged_config = {**embed_config}
            merged_config.update(lang_embed_config)
            embed_config = merged_config

        return cls(
            model_name=embed_config["model_name"],
            device=embed_config["device"],
            max_seq_length=embed_config["max_seq_length"],
            batch_size=embed_config["batch_size"],
            normalize_embeddings=embed_config["normalize_embeddings"],
            use_safetensors=embed_config["use_safetensors"],
            trust_remote_code=embed_config["trust_remote_code"],
            torch_dtype=embed_config["torch_dtype"],
            cache_dir=main_config["shared"]["cache_dir"],
        )


@dataclass
class RetrievalConfig:
    """
    Configuration for document retrieval - fail-fast implementation.
    """

    default_k: int
    max_k: int
    similarity_threshold: float
    adaptive_retrieval: bool
    enable_reranking: bool
    diversity_lambda: float
    use_hybrid_search: bool
    enable_query_expansion: bool

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "RetrievalConfig":
        """Create config from validated main configuration."""
        retrieval_config = main_config["retrieval"]  # Direct access

        return cls(
            default_k=retrieval_config["default_k"],
            max_k=retrieval_config["max_k"],
            similarity_threshold=float(main_config["similarity_threshold"]),  # Root level
            adaptive_retrieval=retrieval_config["adaptive_retrieval"],
            enable_reranking=retrieval_config["enable_reranking"],
            diversity_lambda=float(retrieval_config["diversity_lambda"]),
            use_hybrid_search=retrieval_config["use_hybrid_search"],
            enable_query_expansion=retrieval_config["enable_query_expansion"],
        )


@dataclass
class RankingConfig:
    """
    Configuration for result ranking - explicit validation.
    """

    method: RankingMethod
    enable_diversity: bool
    diversity_threshold: float
    boost_recent: bool
    boost_authoritative: bool
    content_length_factor: bool
    keyword_density_factor: bool
    language_specific_boost: bool

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "RankingConfig":
        """Create config from validated main configuration."""
        ranking_config = main_config["ranking"]  # Direct access

        # Convert string to enum - fail fast if invalid
        method_str = ranking_config["method"]
        try:
            method = RankingMethod(method_str)
        except ValueError as e:
            raise ValueError(f"Invalid ranking method: {method_str}") from e

        return cls(
            method=method,
            enable_diversity=ranking_config["enable_diversity"],
            diversity_threshold=float(ranking_config["diversity_threshold"]),
            boost_recent=ranking_config["boost_recent"],
            boost_authoritative=ranking_config["boost_authoritative"],
            content_length_factor=ranking_config["content_length_factor"],
            keyword_density_factor=ranking_config["keyword_density_factor"],
            language_specific_boost=ranking_config["language_specific_boost"],
        )


@dataclass
class ReRankingConfig:
    """
    Configuration for cross-encoder reranking.
    """

    enabled: bool
    model_name: str
    max_length: int
    batch_size: int
    top_k: int
    use_fp16: bool
    normalize: bool

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "ReRankingConfig":
        """Create config from validated main configuration."""
        reranking_config = main_config["reranking"]  # Direct access

        return cls(
            enabled=reranking_config["enabled"],
            model_name=reranking_config["model_name"],
            max_length=reranking_config["max_length"],
            batch_size=reranking_config["batch_size"],
            top_k=reranking_config["top_k"],
            use_fp16=reranking_config["use_fp16"],
            normalize=reranking_config["normalize"],
        )


@dataclass
class HybridRetrievalConfig:
    """
    Configuration for hybrid retrieval (dense + sparse).
    """

    dense_weight: float
    sparse_weight: float
    fusion_method: str
    bm25_k1: float
    bm25_b: float

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "HybridRetrievalConfig":
        """Create config from validated main configuration."""
        hybrid_config = main_config["hybrid_retrieval"]  # Direct access

        return cls(
            dense_weight=float(hybrid_config["dense_weight"]),
            sparse_weight=float(hybrid_config["sparse_weight"]),
            fusion_method=hybrid_config["fusion_method"],
            bm25_k1=float(hybrid_config["bm25_k1"]),
            bm25_b=float(hybrid_config["bm25_b"]),
        )


@dataclass
class OllamaConfig:
    """
    Configuration for Ollama LLM client.
    """

    base_url: str
    model: str
    timeout: float
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    stream: bool
    keep_alive: str
    num_predict: int
    repeat_penalty: float
    seed: int

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "OllamaConfig":
        """Create config from validated main configuration."""
        ollama_config = main_config["ollama"]  # Direct access

        return cls(
            base_url=ollama_config["base_url"],
            model=ollama_config["model"],
            timeout=float(ollama_config["timeout"]),
            temperature=float(ollama_config["temperature"]),
            max_tokens=ollama_config["max_tokens"],
            top_p=float(ollama_config["top_p"]),
            top_k=ollama_config["top_k"],
            stream=ollama_config["stream"],
            keep_alive=ollama_config["keep_alive"],
            num_predict=ollama_config["num_predict"],
            repeat_penalty=float(ollama_config["repeat_penalty"]),
            seed=ollama_config["seed"],
        )


@dataclass
class ProcessingConfig:
    """
    Configuration for document processing.
    """

    sentence_chunk_overlap: int
    preserve_paragraphs: bool
    enable_smart_chunking: bool
    respect_document_structure: bool

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "ProcessingConfig":
        """Create config from validated main configuration."""
        processing_config = main_config["processing"]  # Direct access

        return cls(
            sentence_chunk_overlap=processing_config["sentence_chunk_overlap"],
            preserve_paragraphs=processing_config["preserve_paragraphs"],
            enable_smart_chunking=processing_config["enable_smart_chunking"],
            respect_document_structure=processing_config["respect_document_structure"],
        )


@dataclass
class ChunkingConfig:
    """
    Configuration for text chunking.
    """

    strategy: ChunkingStrategy
    max_chunk_size: int
    preserve_sentence_boundaries: bool
    respect_paragraph_breaks: bool
    enable_smart_splitting: bool
    sentence_search_range: int
    paragraph_separators: list[str]
    min_sentence_length: int

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "ChunkingConfig":
        """Create config from validated main configuration."""
        chunking_config = main_config["chunking"]  # Direct access

        # Convert string to enum
        strategy_str = chunking_config["strategy"]
        try:
            strategy = ChunkingStrategy(strategy_str)
        except ValueError as e:
            raise ValueError(f"Invalid chunking strategy: {strategy_str}") from e

        return cls(
            strategy=strategy,
            max_chunk_size=chunking_config["max_chunk_size"],
            preserve_sentence_boundaries=chunking_config["preserve_sentence_boundaries"],
            respect_paragraph_breaks=chunking_config["respect_paragraph_breaks"],
            enable_smart_splitting=chunking_config["enable_smart_splitting"],
            sentence_search_range=chunking_config["sentence_search_range"],
            paragraph_separators=chunking_config["paragraph_separators"],
            min_sentence_length=chunking_config["min_sentence_length"],
        )


@dataclass
class StorageConfig:
    """
    Configuration for vector database storage.
    """

    db_path_template: str
    collection_name_template: str
    distance_metric: str
    persist: bool
    allow_reset: bool

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "StorageConfig":
        """Create config from validated main configuration."""
        storage_config = main_config["storage"]  # Direct access

        return cls(
            db_path_template=storage_config["db_path_template"],
            collection_name_template=storage_config["collection_name_template"],
            distance_metric=storage_config["distance_metric"],
            persist=storage_config["persist"],
            allow_reset=storage_config["allow_reset"],
        )


@dataclass
class SearchConfig:
    """
    Configuration for search operations.
    """

    default_method: str
    max_context_length: int
    rerank: bool
    include_metadata: bool
    include_distances: bool
    semantic_weight: float
    keyword_weight: float

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "SearchConfig":
        """Create config from validated main configuration."""
        search_config = main_config["search"]  # Direct access
        weights_config = search_config["weights"]  # Direct access

        return cls(
            default_method=search_config["default_method"],
            max_context_length=search_config["max_context_length"],
            rerank=search_config["rerank"],
            include_metadata=search_config["include_metadata"],
            include_distances=search_config["include_distances"],
            semantic_weight=float(weights_config["semantic_weight"]),
            keyword_weight=float(weights_config["keyword_weight"]),
        )


@dataclass
class ResponseParsingConfig:
    """
    Configuration for response parsing and validation.
    """

    validate_responses: bool
    extract_confidence_scores: bool
    parse_citations: bool
    handle_incomplete_responses: bool
    max_response_length: int
    min_response_length: int
    filter_hallucinations: bool
    require_source_grounding: bool
    confidence_threshold: float
    response_format: str
    include_metadata: bool

    @classmethod
    def from_validated_config(cls, main_config: dict) -> "ResponseParsingConfig":
        """Create config from validated main configuration."""
        response_config = main_config["response_parsing"]  # Direct access

        return cls(
            validate_responses=response_config["validate_responses"],
            extract_confidence_scores=response_config["extract_confidence_scores"],
            parse_citations=response_config["parse_citations"],
            handle_incomplete_responses=response_config["handle_incomplete_responses"],
            max_response_length=response_config["max_response_length"],
            min_response_length=response_config["min_response_length"],
            filter_hallucinations=response_config["filter_hallucinations"],
            require_source_grounding=response_config["require_source_grounding"],
            confidence_threshold=float(response_config["confidence_threshold"]),
            response_format=response_config["response_format"],
            include_metadata=response_config["include_metadata"],
        )


@dataclass
class LanguageSpecificConfig:
    """
    Configuration for language-specific settings.
    Combines common language settings used across components.
    """

    language_code: str
    language_name: str
    language_family: str
    preserve_diacritics: bool
    response_language: str
    stopwords: list[str]
    question_patterns: dict[str, list[str]]
    cultural_indicators: list[str]
    system_prompt_language: str
    formality_level: str

    @classmethod
    def from_validated_config(cls, language_config: dict) -> "LanguageSpecificConfig":
        """Create config from validated language configuration."""
        return cls(
            language_code=language_config["language"]["code"],
            language_name=language_config["language"]["name"],
            language_family=language_config["language"]["family"],
            preserve_diacritics=language_config["shared"]["preserve_diacritics"],
            response_language=language_config["shared"]["response_language"],
            stopwords=language_config["shared"]["stopwords"]["words"],
            question_patterns=language_config["shared"]["question_patterns"],
            cultural_indicators=language_config["categorization"]["cultural_indicators"],
            system_prompt_language=language_config["generation"]["system_prompt_language"],
            formality_level=language_config["generation"]["formality_level"],
        )


@dataclass
class SystemConfig:
    """
    Master configuration combining all component configs.
    Created after ConfigValidator ensures all keys exist.
    """

    query_processing: QueryProcessingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    ranking: RankingConfig
    reranking: ReRankingConfig
    hybrid_retrieval: HybridRetrievalConfig
    ollama: OllamaConfig
    processing: ProcessingConfig
    chunking: ChunkingConfig
    storage: StorageConfig
    search: SearchConfig
    response_parsing: ResponseParsingConfig
    language_specific: LanguageSpecificConfig

    @classmethod
    def from_validated_configs(cls, main_config: dict, language_config: dict, language_code: str) -> "SystemConfig":
        """
        Create complete system configuration from validated configs.

        This is the main entry point for creating system configuration.
        ConfigValidator must be run first to guarantee all keys exist.
        """
        logger.info(f"Creating system configuration for language: {language_code}")

        return cls(
            query_processing=QueryProcessingConfig.from_validated_config(main_config, language_code),
            embedding=EmbeddingConfig.from_validated_config(main_config),
            retrieval=RetrievalConfig.from_validated_config(main_config),
            ranking=RankingConfig.from_validated_config(main_config),
            reranking=ReRankingConfig.from_validated_config(main_config),
            hybrid_retrieval=HybridRetrievalConfig.from_validated_config(main_config),
            ollama=OllamaConfig.from_validated_config(main_config),
            processing=ProcessingConfig.from_validated_config(main_config),
            chunking=ChunkingConfig.from_validated_config(main_config),
            storage=StorageConfig.from_validated_config(main_config),
            search=SearchConfig.from_validated_config(main_config),
            response_parsing=ResponseParsingConfig.from_validated_config(main_config),
            language_specific=LanguageSpecificConfig.from_validated_config(language_config),
        )


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""

    db_path: str
    collection_name: str
    distance_metric: str
    chunk_size: int
    ef_construction: int
    m: int
    persist: bool
    allow_reset: bool

    @classmethod
    def from_validated_config(cls, main_config: dict, tenant_slug: str = "development") -> "ChromaConfig":
        """Create config from validated configuration."""
        storage_config = main_config["storage"]  # Direct access - guaranteed by validation

        # Generate tenant-specific db_path from template
        db_path_template = storage_config["db_path_template"]
        data_base_dir = main_config["paths"]["data_base_dir"]
        db_path = db_path_template.format(data_base_dir=data_base_dir, tenant_slug=tenant_slug)

        return cls(
            db_path=db_path,
            collection_name=storage_config.get("collection_name_template", "default_collection"),
            distance_metric=storage_config["distance_metric"],
            chunk_size=storage_config.get("chunk_size", 1000),
            ef_construction=storage_config.get("ef_construction", 200),
            m=storage_config.get("m", 16),
            persist=storage_config["persist"],
            allow_reset=storage_config["allow_reset"],
        )


@dataclass
class LanguageConfig:
    """Language-specific configuration for multilingual support."""

    language_code: str
    enable_morphological_expansion: bool
    enable_synonym_expansion: bool
    use_language_query_processing: bool
    language_priority: bool
    stop_words_file: str
    morphology_patterns_file: str

    @classmethod
    def from_validated_config(cls, main_config: dict, language: str) -> "LanguageConfig":
        """Create config from validated configuration."""
        from ..utils.config_protocol import get_config_provider

        # Get language-specific config from provider
        config_provider = get_config_provider()
        language_config = config_provider.get_language_specific_config("pipeline", language)

        return cls(
            language_code=language,
            enable_morphological_expansion=language_config["enable_morphological_expansion"],
            enable_synonym_expansion=language_config["enable_synonym_expansion"],
            use_language_query_processing=language_config["use_language_query_processing"],
            language_priority=language_config["language_priority"],
            stop_words_file=language_config["stop_words_file"],
            morphology_patterns_file=language_config["morphology_patterns_file"],
        )
