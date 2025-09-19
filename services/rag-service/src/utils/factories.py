"""
Component factory utilities for RAG system dependency injection.
"""

from pathlib import Path

from ..models.multitenant_models import DocumentScope, Tenant, User
from ..pipeline.rag_system import create_rag_system
from .logging_factory import get_system_logger, log_component_end, log_component_start, log_error_context


def create_complete_rag_system(
    language: str, tenant: Tenant | None = None, user: User | None = None, collection_name: str | None = None
):
    """Create a complete RAG system using real components for production use."""
    logger = get_system_logger()
    log_component_start(
        "rag_factory",
        "create_complete_rag_system",
        language=language,
        tenant_slug=tenant.slug if tenant else None,
        user_id=user.username if user else None,
    )

    try:
        # Load configurations using the same approach as RAGConfig
        from ..utils.config_loader import load_config
        from ..utils.config_models import ChromaConfig, EmbeddingConfig, OllamaConfig, ProcessingConfig, RetrievalConfig
        from ..utils.config_protocol import get_config_provider

        logger.debug("rag_factory", "create_complete_rag_system", "Loading configuration provider and configs")
        config_provider = get_config_provider()
        main_config = load_config("config")  # Get full structured config
        language_config = config_provider.get_language_config(language)

        logger.debug("rag_factory", "create_complete_rag_system", f"Loaded configs for {language}")
        logger.trace("rag_factory", "create_complete_rag_system", f"Main config sections: {list(main_config.keys())}")
        logger.trace(
            "rag_factory", "create_complete_rag_system", f"Language config sections: {list(language_config.keys())}"
        )

        # Build compatible chroma config from storage and vectordb sections
        storage_config = main_config["storage"]
        logger.trace("rag_factory", "create_complete_rag_system", f"storage_config keys: {list(storage_config.keys())}")

        vectordb_config = main_config["vectordb"]["factory"]
        logger.trace(
            "rag_factory", "create_complete_rag_system", f"vectordb_config keys: {list(vectordb_config.keys())}"
        )

        shared_config = main_config["shared"]

        # Create a chroma section that matches ChromaConfig expectations
        main_config["chroma"] = {
            "db_path": "./data/vectordb",  # Default path, will be updated per tenant
            "collection_name": f"{language}_documents",  # Default, will be updated per tenant
            "distance_metric": storage_config["distance_metric"],
            "chunk_size": vectordb_config["chunk_size"],
            "ef_construction": 200,  # HNSW default
            "m": 16,  # HNSW default
            "persist": storage_config["persist"],
            "allow_reset": storage_config["allow_reset"],
        }

        # Promote shared config values to root level for config models that expect them there
        for key, value in shared_config.items():
            if key not in main_config:  # Don't override existing sections
                main_config[key] = value

        # Create component configs using validated approach
        processing_config = ProcessingConfig.from_validated_config(main_config)
        embedding_config = EmbeddingConfig.from_validated_config(main_config, language_config)
        tenant_slug = tenant.slug if tenant else "development"
        chroma_config = ChromaConfig.from_validated_config(main_config, tenant_slug)
        retrieval_config = RetrievalConfig.from_validated_config(main_config)
        ollama_config = OllamaConfig.from_validated_config(main_config)

        # Create individual components using their factories
        from ..generation.enhanced_prompt_templates import create_enhanced_prompt_builder
        from ..generation.ollama_client import create_ollama_client
        from ..generation.response_parser import create_response_parser
        from ..preprocessing.chunkers import create_document_chunker
        from ..preprocessing.cleaners import MultilingualTextCleaner
        from ..preprocessing.cleaners_providers import create_providers as create_cleaning_providers
        from ..preprocessing.extractors import DocumentExtractor
        from ..preprocessing.extractors_providers import create_providers
        from ..retrieval.hierarchical_retriever_providers import create_hierarchical_retriever
        from ..retrieval.query_processor_providers import create_query_processor
        from ..retrieval.ranker import create_document_ranker
        from ..retrieval.reranker import create_multilingual_reranker
        from ..retrieval.retriever import create_document_retriever
        from ..vectordb.embeddings import create_embedding_generator
        from ..vectordb.search_providers import create_vector_search_provider
        from ..vectordb.storage_factories import create_vector_database

        # Document extractor with dependency injection
        extractor_config_provider, file_system_provider, logger_provider = create_providers()
        document_extractor = DocumentExtractor(extractor_config_provider, file_system_provider)

        # Text cleaner with language support and dependency injection
        cleaning_config_provider, cleaning_logger_provider, cleaning_env_provider = create_cleaning_providers()
        text_cleaner = MultilingualTextCleaner(
            language=language,
            config_provider=cleaning_config_provider,
            logger_provider=cleaning_logger_provider,
            environment_provider=cleaning_env_provider,
        )

        # Document chunker with config provider
        from ..utils.config_protocol import get_config_provider

        chunker_config_provider = get_config_provider()
        document_chunker = create_document_chunker(
            config_dict=main_config, config_provider=chunker_config_provider, language=language
        )

        # Embedding model
        embedding_model = create_embedding_generator(config=embedding_config)

        # Vector storage with proper collection name
        if collection_name is None and tenant and user:
            scope = DocumentScope.USER
            collection_name = tenant.get_collection_name(scope, language)

        vector_database = create_vector_database(
            db_path=chroma_config.db_path, distance_metric=chroma_config.distance_metric
        )
        # Try to get existing collection, create if it doesn't exist
        try:
            vector_storage = vector_database.get_collection(name=collection_name or f"{language}_documents")
        except Exception:
            # Collection doesn't exist, create it
            vector_storage = vector_database.create_collection(name=collection_name or f"{language}_documents")

        # Create embedding provider for search engine with same model as document processing
        from ..vectordb.search_providers import create_embedding_provider

        embedding_provider = create_embedding_provider(
            model_name=embedding_config.model_name, device=embedding_config.device
        )

        # Search engine
        search_engine = create_vector_search_provider(vector_storage, embedding_provider)

        # Query processor
        query_processor = create_query_processor(language=language)

        # Document ranker
        document_ranker = create_document_ranker(language=language)

        # Retriever
        retriever = create_document_retriever(
            query_processor=query_processor,
            search_engine=search_engine,
            result_ranker=document_ranker,
            config=retrieval_config,
        )

        # Hierarchical retriever
        hierarchical_retriever = create_hierarchical_retriever(search_engine=search_engine, language=language)

        # Ranker (using mock components for now)
        from ..retrieval.reranker import create_mock_model_loader, create_mock_score_calculator

        mock_model_loader = create_mock_model_loader()
        mock_score_calculator = create_mock_score_calculator()
        ranker = create_multilingual_reranker(model_loader=mock_model_loader, score_calculator=mock_score_calculator)

        # Generation client
        generation_client = create_ollama_client(config=ollama_config)

        # Response parser
        response_parser = create_response_parser(config_provider=config_provider, language=language)

        # Prompt builder
        prompt_builder = create_enhanced_prompt_builder(language=language, config_provider=config_provider)

        logger.debug("rag_factory", "create_complete_rag_system", "Creating complete RAG system with all components")

        # Create the complete RAG system using the factory
        rag_system = create_rag_system(
            language=language,
            document_extractor=document_extractor,
            text_cleaner=text_cleaner,
            document_chunker=document_chunker,
            embedding_model=embedding_model,
            vector_storage=vector_storage,
            search_engine=search_engine,
            query_processor=query_processor,
            retriever=retriever,
            hierarchical_retriever=hierarchical_retriever,
            ranker=ranker,
            generation_client=generation_client,
            response_parser=response_parser,
            prompt_builder=prompt_builder,
            embedding_config=embedding_config,
            ollama_config=ollama_config,
            processing_config=processing_config,
            retrieval_config=retrieval_config,
        )

        logger.info("rag_factory", "create_complete_rag_system", f"RAG system created successfully for {language}")
        log_component_end("rag_factory", "create_complete_rag_system", f"Complete RAG system ready for {language}")
        return rag_system

    except Exception as e:
        log_error_context(
            "rag_factory",
            "create_complete_rag_system",
            e,
            {
                "language": language,
                "tenant_slug": tenant.slug if tenant else None,
                "user_id": user.username if user else None,
            },
        )
        raise


def create_tenant_and_user_from_cli_args(tenant: str, user: str) -> tuple[Tenant, User]:
    """Create Tenant and User objects from CLI arguments."""
    tenant_obj = Tenant(id=f"tenant_{tenant}", name=tenant.replace("_", " ").title(), slug=tenant)

    user_obj = User(id=f"user_{user}", username=user, tenant_id=tenant_obj.id, email=f"{user}@{tenant}.local")

    return tenant_obj, user_obj


def create_documents_path(tenant: str, user: str, language: str) -> Path:
    """Create the documents path for a given tenant, user, and language."""
    return Path("data") / tenant / "users" / user / "documents" / language


def create_processed_path(tenant: str, user: str, language: str) -> Path:
    """Create the processed documents path for a given tenant, user, and language."""
    return Path("data") / tenant / "users" / user / "processed" / language
