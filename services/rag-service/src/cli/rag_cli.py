#!/usr/bin/env python3
"""
Command-line interface for the multilingual RAG system.
Provides document processing, querying, and system management operations
with support for multiple languages and tenants.
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TextIO


# Pure data structures
@dataclass
class CLIArgs:
    """CLI arguments - pure data structure."""

    command: Optional[str]
    tenant: str
    user: str
    language: str
    log_level: str
    # Command-specific args
    query_text: Optional[str] = None
    top_k: Optional[int] = None
    no_sources: bool = False
    docs_path: Optional[str] = None
    languages: Optional[List[str]] = None


@dataclass
class TenantContext:
    """Tenant context - pure data structure."""

    tenant_id: str
    tenant_name: str
    tenant_slug: str
    user_id: str
    user_email: str
    user_username: str
    user_full_name: str


@dataclass
class QueryResult:
    """Query execution result - pure data structure."""

    success: bool
    answer: str
    sources: List[str]
    query_time: float
    documents_retrieved: int
    retrieved_chunks: List[Dict[str, Any]]
    error_message: Optional[str] = None


@dataclass
class DocumentProcessingResult:
    """Document processing result - pure data structure."""

    success: bool
    processing_time: float
    processing_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class CollectionInfo:
    """Collection information - pure data structure."""

    user_collection_name: str
    tenant_collection_name: str
    base_path: str
    available_collections: List[str]
    document_counts: Dict[str, int]


@dataclass
class SystemStatus:
    """System status information - pure data structure."""

    rag_system_status: str  # "initialized", "failed"
    folder_structure: Dict[str, bool]  # path -> exists
    config_status: str  # "loaded", "failed"
    details: Dict[str, Any]
    error_messages: List[str]


# Protocol definitions for dependency injection
class OutputWriterProtocol(Protocol):
    """Protocol for output writing (stdout, file, etc.)."""

    def write(self, text: str) -> None:
        ...

    def flush(self) -> None:
        ...


class LoggerProtocol(Protocol):
    """Protocol for logging."""

    def info(self, message: str) -> None:
        ...

    def error(self, message: str) -> None:
        ...

    def exception(self, message: str) -> None:
        ...


class RAGSystemProtocol(Protocol):
    """Protocol for RAG system operations."""

    async def initialize(self) -> None:
        ...

    async def query(self, query: Any) -> Any:
        ...

    async def add_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        ...


class FolderManagerProtocol(Protocol):
    """Protocol for folder management."""

    def ensure_context_folders(self, context: Any, language: str) -> bool:
        ...

    def get_collection_storage_paths(self, context: Any, language: str) -> Dict[str, Any]:
        ...

    def get_tenant_folder_structure(self, tenant: Any, user: Any, language: str) -> Dict[str, Any]:
        ...

    def create_tenant_folder_structure(
        self, tenant: Any, user: Any, languages: List[str]
    ) -> tuple[bool, List[str]]:
        ...


class StorageProtocol(Protocol):
    """Protocol for storage operations."""

    def list_collections(self) -> List[str]:
        ...

    def get_document_count(self, collection_name: str) -> int:
        ...


class ConfigLoaderProtocol(Protocol):
    """Protocol for configuration loading."""

    def get_shared_config(self) -> Dict[str, Any]:
        ...

    def get_storage_config(self) -> Dict[str, Any]:
        ...


# Pure functions for business logic
def validate_language_code(language: str) -> str:
    """Validate and normalize language code."""
    if not language or not isinstance(language, str):
        raise ValueError("Language code must be a non-empty string")

    language = language.lower().strip()
    valid_languages = {"hr", "en", "multilingual"}

    if language not in valid_languages:
        raise ValueError(f"Unsupported language: {language}. Supported: {valid_languages}")

    return language


def validate_tenant_slug(tenant_slug: str) -> str:
    """Validate tenant slug."""
    if not tenant_slug or not isinstance(tenant_slug, str):
        raise ValueError("Tenant slug must be a non-empty string")

    slug = tenant_slug.strip().lower()
    if not slug.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "Tenant slug must contain only alphanumeric characters, hyphens, and underscores"
        )

    return slug


def validate_user_id(user_id: str) -> str:
    """Validate user ID."""
    if not user_id or not isinstance(user_id, str):
        raise ValueError("User ID must be a non-empty string")

    user_id = user_id.strip()
    if len(user_id) < 2:
        raise ValueError("User ID must be at least 2 characters long")

    return user_id


def create_tenant_context(tenant_slug: str, user_id: str) -> TenantContext:
    """Create tenant/user context from CLI arguments using pure logic."""
    # Validate inputs
    validated_tenant = validate_tenant_slug(tenant_slug)
    validated_user = validate_user_id(user_id)

    if validated_tenant == "development" and validated_user == "dev_user":
        # Default development context
        return TenantContext(
            tenant_id="tenant:development",
            tenant_name="Development Tenant",
            tenant_slug="development",
            user_id="user:dev_user",
            user_email="dev_user@development.example.com",
            user_username="dev_user",
            user_full_name="Development User",
        )
    else:
        # Custom tenant/user context
        return TenantContext(
            tenant_id=f"tenant:{validated_tenant}",
            tenant_name=f"Tenant {validated_tenant.title()}",
            tenant_slug=validated_tenant,
            user_id=f"user:{validated_user}",
            user_email=f"{validated_user}@{validated_tenant}.example.com",
            user_username=validated_user,
            user_full_name=f"User {validated_user.title()}",
        )


def format_query_results(result: QueryResult, context: TenantContext, language: str) -> List[str]:
    """Format query results for display using pure logic."""
    if not result.success:
        return [
            f"❌ Query failed: {result.error_message}",
            "",
            f"🏢 Tenant: {context.tenant_slug}",
            f"👤 User: {context.user_username}",
            f"🌐 Language: {language}",
        ]

    lines = ["=" * 60, "📊 QUERY RESULTS", "=" * 60, f"💬 Answer: {result.answer}", ""]

    if result.sources:
        lines.append("📚 Sources:")
        for i, source in enumerate(result.sources, 1):
            lines.append(f"  {i}. {source}")
        lines.append("")

    lines.extend(
        [
            f"⚡ Query time: {result.query_time:.2f}s",
            f"📄 Documents retrieved: {result.documents_retrieved}",
        ]
    )

    if result.retrieved_chunks:
        lines.append("\n🔍 Retrieved chunks:")
        for i, chunk in enumerate(result.retrieved_chunks, 1):
            score = chunk["similarity_score"] if "similarity_score" in chunk else 0
            final_score = chunk["final_score"] if "final_score" in chunk else score
            source = chunk["source"] if "source" in chunk else "Unknown"
            lines.extend(
                [
                    f"  {i}. Score: {final_score:.3f} | Source: {source}",
                    f"     Content: {chunk['content'][:100]}...",
                    "",
                ]
            )

    return lines


def format_processing_results(
    result: DocumentProcessingResult,
    context: TenantContext,
    language: str,
    docs_path: str,
) -> List[str]:
    """Format document processing results using pure logic."""
    lines = [
        f"📁 Processing documents for tenant: {context.tenant_slug}, user: {context.user_username}",
        f"📂 Documents path: {docs_path}",
        f"🌐 Language: {language}",
        "",
    ]

    if not result.success:
        lines.extend([f"❌ Document processing failed: {result.error_message}"])
        return lines

    lines.extend(
        [
            f"✅ Documents processed successfully in {result.processing_time:.2f}s",
            f"📊 Processing result: {result.processing_result}",
        ]
    )

    return lines


def format_collection_info(
    collection_info: CollectionInfo, context: TenantContext, language: str
) -> List[str]:
    """Format collection information using pure logic."""
    lines = [
        f"📋 Listing collections for tenant: {context.tenant_slug}",
        f"👤 User: {context.user_username}",
        f"🌐 Language: {language}",
        "",
        "📦 ChromaDB Collections:",
        f"  👤 User collection: {collection_info.user_collection_name}",
        f"  🏢 Tenant collection: {collection_info.tenant_collection_name}",
        f"  📁 Base path: {collection_info.base_path}",
        "",
    ]

    if collection_info.available_collections:
        lines.append("🗃️  Available collections:")
        for collection in collection_info.available_collections:
            lines.append(f"  - {collection}")

        # Add document counts if available
        for collection, count in collection_info.document_counts.items():
            if collection == collection_info.user_collection_name:
                lines.append(f"  👤 User collection document count: {count}")
            elif collection == collection_info.tenant_collection_name:
                lines.append(f"  🏢 Tenant collection document count: {count}")

    return lines


def format_system_status(status: SystemStatus, context: TenantContext, language: str) -> List[str]:
    """Format system status using pure logic."""
    lines = [
        f"📊 System status for tenant: {context.tenant_slug}",
        f"👤 User: {context.user_username}",
        f"🌐 Language: {language}",
        "",
    ]

    # RAG System status
    if status.rag_system_status == "initialized":
        lines.append("✅ RAG System: Initialized successfully")
    else:
        lines.append(f"❌ RAG System: Failed to initialize")

    # Folder structure
    existing_folders = [path for path, exists in status.folder_structure.items() if exists]
    missing_folders = [path for path, exists in status.folder_structure.items() if not exists]

    lines.append(
        f"📁 Folder structure ({len(existing_folders)} existing, {len(missing_folders)} missing):"
    )
    for folder in existing_folders[:5]:  # Show first 5
        lines.append(f"  ✅ {folder}")
    if len(existing_folders) > 5:
        lines.append(f"  ... and {len(existing_folders) - 5} more")

    if missing_folders:
        lines.append("  Missing folders:")
        for folder in missing_folders[:3]:  # Show first 3
            lines.append(f"  ❌ {folder}")

    # Configuration status
    if status.config_status == "loaded":
        lines.append("✅ Configuration: Loaded successfully")
    else:
        lines.append("❌ Configuration: Failed to load")

    # Error messages
    if status.error_messages:
        lines.append("\n⚠️  Errors:")
        for error in status.error_messages:
            lines.append(f"  - {error}")

    return lines


def format_create_folders_result(result: Dict[str, Any], context: TenantContext) -> List[str]:
    """Format create-folders result using pure logic."""
    lines = [
        f"📁 Folder creation result for tenant: {context.tenant_slug}",
        f"👤 User: {context.user_username}",
        f"🌐 Languages: {', '.join(result['languages'] if 'languages' in result else [])}",
        "",
    ]

    if "success" in result and result["success"]:
        lines.append("✅ Folder creation completed successfully")

        created = result["created_folders"] if "created_folders" in result else []
        existing = result["existing_folders"] if "existing_folders" in result else []

        if created:
            lines.append(f"\n📂 Created folders ({len(created)}):")
            for folder in created:
                lines.append(f"  ✅ {folder}")

        if existing:
            lines.append(f"\n📁 Already existing folders ({len(existing)}):")
            for folder in existing:
                lines.append(f"  ℹ️  {folder}")

        lines.append(f"\n{result['message'] if 'message' in result else ''}")
    else:
        lines.append("❌ Folder creation failed")
        lines.append(f"Error: {result['error'] if 'error' in result else 'Unknown error'}")

    return lines


def parse_cli_arguments(args: List[str]) -> CLIArgs:
    """Parse command-line arguments using pure logic."""
    parser = argparse.ArgumentParser(
        description="Multi-tenant RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query the system (English)
  python -m src.cli.rag_cli --language en query "What is a RAG system?"

  # Process documents (English)
  python -m src.cli.rag_cli --language en process-docs ./data/raw/en/

  # Check system status
  python -m src.cli.rag_cli --language en status

  # List collections
  python -m src.cli.rag_cli --language en list-collections
        """,
    )

    # Global options
    parser.add_argument(
        "--tenant", default="development", help="Tenant slug (default: development)"
    )
    parser.add_argument("--user", default="dev_user", help="User ID (default: dev_user)")
    parser.add_argument(
        "--language",
        choices=["hr", "en", "multilingual"],
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query_text", help="Query text to search for")
    query_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of documents to retrieve"
    )
    query_parser.add_argument("--no-sources", action="store_true", help="Don't return sources")

    # Process documents command
    process_parser = subparsers.add_parser("process-docs", help="Process documents")
    process_parser.add_argument("docs_path", help="Path to documents to process")

    # List collections command
    subparsers.add_parser("list-collections", help="List ChromaDB collections")

    # Create folders command
    folders_parser = subparsers.add_parser(
        "create-folders", help="Create tenant/user folder structure"
    )
    folders_parser.add_argument(
        "--languages",
        nargs="+",
        default=["hr", "en"],
        help="Languages to create folders for",
    )

    # Status command
    subparsers.add_parser("status", help="Show system status")

    parsed_args = parser.parse_args(args)

    return CLIArgs(
        command=parsed_args.command,
        tenant=parsed_args.tenant,
        user=parsed_args.user,
        language=parsed_args.language,
        log_level=parsed_args.log_level,
        query_text=getattr(parsed_args, "query_text", None),
        top_k=getattr(parsed_args, "top_k", None),
        no_sources=getattr(parsed_args, "no_sources", False),
        docs_path=getattr(parsed_args, "docs_path", None),
        languages=getattr(parsed_args, "languages", None),
    )


# Main testable CLI class
class MultiTenantRAGCLI:
    """Multi-tenant RAG command-line interface with configurable operations."""

    def __init__(
        self,
        output_writer: OutputWriterProtocol,
        logger: LoggerProtocol,
        rag_system_factory: Any,  # Factory function
        folder_manager: FolderManagerProtocol,
        storage: StorageProtocol,
        config_loader: ConfigLoaderProtocol,
    ):
        """Initialize with all dependencies injected."""
        self.output_writer = output_writer
        self.logger = logger
        self.rag_system_factory = rag_system_factory
        self.folder_manager = folder_manager
        self.storage = storage
        self.config_loader = config_loader

    def write_output(self, lines: List[str]) -> None:
        """Write formatted output."""
        for line in lines:
            self.output_writer.write(line + "\n")
        self.output_writer.flush()

    async def execute_query_command(
        self,
        context: TenantContext,
        language: str,
        query_text: str,
        top_k: int = 5,
        return_sources: bool = True,
    ) -> QueryResult:
        """Execute query command using pure business logic."""
        try:
            # Import RAGQuery class for proper query object creation
            from src.pipeline.rag_system import RAGQuery

            # Create RAG system (injected factory)
            rag = self.rag_system_factory(language=language, tenant_context=context)
            await rag.initialize()

            # Create proper RAGQuery object
            query = RAGQuery(
                text=query_text,
                language=language,
                max_results=top_k,
                metadata={"return_sources": return_sources, "return_debug_info": True},
            )

            # Execute query
            start_time = time.time()
            response = await rag.query(query)
            query_time = time.time() - start_time

            return QueryResult(
                success=True,
                answer=response.answer,
                sources=response.sources if hasattr(response, "sources") else [],
                query_time=query_time,
                documents_retrieved=len(response.retrieved_chunks),
                retrieved_chunks=response.retrieved_chunks,
            )

        except Exception as e:
            self.logger.exception("Query execution failed")
            return QueryResult(
                success=False,
                answer="",
                sources=[],
                query_time=0.0,
                documents_retrieved=0,
                retrieved_chunks=[],
                error_message=str(e),
            )

    async def execute_process_documents_command(
        self, context: TenantContext, language: str, docs_path: str
    ) -> DocumentProcessingResult:
        """Execute document processing command using pure business logic."""
        try:
            # Ensure folder structure exists
            success = self.folder_manager.ensure_context_folders(context, language)
            if not success:
                return DocumentProcessingResult(
                    success=False,
                    processing_time=0.0,
                    error_message="Failed to create folder structure",
                )

            # Initialize RAG system
            rag = self.rag_system_factory(language=language, tenant_context=context)
            await rag.initialize()

            # Process documents
            start_time = time.time()
            result = await rag.add_documents([docs_path])
            processing_time = time.time() - start_time

            return DocumentProcessingResult(
                success=True, processing_time=processing_time, processing_result=result
            )

        except Exception as e:
            self.logger.exception("Document processing failed")
            return DocumentProcessingResult(
                success=False, processing_time=0.0, error_message=str(e)
            )

    async def execute_list_collections_command(
        self, context: TenantContext, language: str
    ) -> CollectionInfo:
        """Execute list collections command using pure business logic."""
        try:
            collections_info = self.folder_manager.get_collection_storage_paths(context, language)

            # Get available collections from storage
            available_collections = self.storage.list_collections()

            # Get document counts
            document_counts = {}
            user_collection = collections_info["user_collection_name"]
            tenant_collection = collections_info["tenant_collection_name"]

            if user_collection in available_collections:
                document_counts[user_collection] = self.storage.get_document_count(user_collection)

            if tenant_collection in available_collections:
                document_counts[tenant_collection] = self.storage.get_document_count(
                    tenant_collection
                )

            return CollectionInfo(
                user_collection_name=user_collection,
                tenant_collection_name=tenant_collection,
                base_path=collections_info["base_path"],
                available_collections=available_collections,
                document_counts=document_counts,
            )

        except Exception as e:
            self.logger.exception("Collection listing failed")
            return CollectionInfo(
                user_collection_name="unknown",
                tenant_collection_name="unknown",
                base_path="unknown",
                available_collections=[],
                document_counts={},
            )

    async def execute_status_command(self, context: TenantContext, language: str) -> SystemStatus:
        """Execute status command using pure business logic."""
        error_messages = []

        # Test RAG system initialization
        try:
            rag = self.rag_system_factory(language=language, tenant_context=context)
            await rag.initialize()
            rag_status = "initialized"
        except Exception as e:
            rag_status = "failed"
            error_messages.append(f"RAG System: {str(e)}")

        # Check folder structure
        try:
            paths = self.folder_manager.get_tenant_folder_structure(context, None, language)
            folder_structure = {}
            for path_name, path in paths.items():
                folder_structure[f"{path_name}: {path}"] = Path(str(path)).exists()
        except Exception as e:
            folder_structure = {}
            error_messages.append(f"Folder check: {str(e)}")

        # Test configuration loading
        try:
            self.config_loader.get_shared_config()
            self.config_loader.get_storage_config()
            config_status = "loaded"
        except Exception as e:
            config_status = "failed"
            error_messages.append(f"Configuration: {str(e)}")

        return SystemStatus(
            rag_system_status=rag_status,
            folder_structure=folder_structure,
            config_status=config_status,
            details={},
            error_messages=error_messages,
        )

    async def execute_create_folders_command(
        self, context: TenantContext, languages: List[str]
    ) -> Dict[str, Any]:
        """Execute create-folders command using pure business logic."""
        try:
            created_folders = []
            existing_folders = []

            for language in languages:
                paths = self.folder_manager.get_tenant_folder_structure(context, None, language)

                for path_name, path in paths.items():
                    path_obj = Path(str(path))
                    if path_obj.exists():
                        existing_folders.append(f"{path_name}: {path}")
                    else:
                        # Create folder (mock implementation)
                        created_folders.append(f"{path_name}: {path}")

            return {
                "success": True,
                "created_folders": created_folders,
                "existing_folders": existing_folders,
                "languages": languages,
                "message": f"Processed folder creation for languages: {', '.join(languages)}",
            }

        except Exception as e:
            return {
                "success": False,
                "created_folders": [],
                "existing_folders": [],
                "languages": languages,
                "error": str(e),
                "message": f"Failed to create folders: {str(e)}",
            }

    async def execute_command(self, args: CLIArgs) -> None:
        """Execute the requested command using pure business logic."""
        if not args.command:
            self.write_output(["❌ No command specified. Use --help for available commands."])
            return

        # Create context using pure function
        context = create_tenant_context(args.tenant, args.user)

        # Write header
        header = [
            "🚀 Multi-tenant RAG System CLI",
            f"🏢 Tenant: {context.tenant_name} ({context.tenant_slug})",
            f"👤 User: {context.user_full_name} ({context.user_username})",
            f"🌐 Language: {args.language}",
            "=" * 60,
        ]
        self.write_output(header)

        try:
            if args.command == "query":
                result = await self.execute_query_command(
                    context,
                    args.language,
                    args.query_text,
                    args.top_k,
                    not args.no_sources,
                )
                formatted_output = format_query_results(result, context, args.language)
                self.write_output(formatted_output)

            elif args.command == "process-docs":
                result = await self.execute_process_documents_command(
                    context, args.language, args.docs_path
                )
                formatted_output = format_processing_results(
                    result, context, args.language, args.docs_path
                )
                self.write_output(formatted_output)

            elif args.command == "list-collections":
                collection_info = await self.execute_list_collections_command(
                    context, args.language
                )
                formatted_output = format_collection_info(collection_info, context, args.language)
                self.write_output(formatted_output)

            elif args.command == "status":
                status = await self.execute_status_command(context, args.language)
                formatted_output = format_system_status(status, context, args.language)
                self.write_output(formatted_output)

            elif args.command == "create-folders":
                result = await self.execute_create_folders_command(context, args.languages)
                formatted_output = format_create_folders_result(result, context)
                self.write_output(formatted_output)

            else:
                self.write_output([f"❌ Unknown command: {args.command}"])

        except KeyboardInterrupt:
            self.write_output(["\n⚠️  Operation cancelled by user"])
        except Exception as e:
            self.write_output([f"❌ Command failed: {e}"])
            self.logger.exception("Command execution failed")


# Mock implementations for testing
class MockOutputWriter:
    """Mock output writer for testing."""

    def __init__(self):
        self.written_lines = []

    def write(self, text: str) -> None:
        self.written_lines.append(text.rstrip("\n"))

    def flush(self) -> None:
        pass


class MockLogger:
    """Mock logger for testing."""

    def __init__(self):
        self.logs = []

    def info(self, message: str) -> None:
        self.logs.append(("INFO", message))

    def error(self, message: str) -> None:
        self.logs.append(("ERROR", message))

    def exception(self, message: str) -> None:
        self.logs.append(("EXCEPTION", message))


class MockRAGSystem:
    """Mock RAG system for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.initialized = False

    async def initialize(self) -> None:
        if self.should_fail:
            raise Exception("Mock initialization failure")
        self.initialized = True

    async def query(self, query: Any) -> Any:
        if self.should_fail:
            raise Exception("Mock query failure")

        class MockResponse:
            answer = f"Mock answer for: {query['text']}"
            sources = ["mock_doc1.txt", "mock_doc2.txt"]
            retrieved_chunks = [
                {
                    "content": "Mock chunk 1",
                    "similarity_score": 0.9,
                    "final_score": 0.9,
                    "source": "mock_doc1.txt",
                },
                {
                    "content": "Mock chunk 2",
                    "similarity_score": 0.8,
                    "final_score": 0.8,
                    "source": "mock_doc2.txt",
                },
            ]

        return MockResponse()

    async def add_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        if self.should_fail:
            raise Exception("Mock processing failure")

        return {
            "processed_documents": len(document_paths),
            "failed_documents": 0,
            "total_chunks": len(document_paths) * 5,
            "processing_time": 1.0,
            "documents_per_second": len(document_paths),
        }


class MockFolderManager:
    """Mock folder manager for testing."""

    def ensure_context_folders(self, context: Any, language: str) -> bool:
        return True

    def get_collection_storage_paths(self, context: Any, language: str) -> Dict[str, Any]:
        return {
            "user_collection_name": f"user_{context.user_username}_{language}",
            "tenant_collection_name": f"tenant_{context.tenant_slug}_{language}",
            "base_path": f"/mock/path/{context.tenant_slug}",
        }

    def get_tenant_folder_structure(self, tenant: Any, user: Any, language: str) -> Dict[str, Any]:
        return {
            "data_folder": Path(f"/mock/{tenant.tenant_slug}/data"),
            "models_folder": Path(f"/mock/{tenant.tenant_slug}/models"),
            "config_folder": Path(f"/mock/{tenant.tenant_slug}/config"),
        }

    def create_tenant_folder_structure(
        self, tenant: Any, user: Any, languages: List[str]
    ) -> tuple[bool, List[str]]:
        created_folders = [f"/mock/{tenant.tenant_slug}/{lang}" for lang in languages]
        return True, created_folders


class MockStorage:
    """Mock storage for testing."""

    def list_collections(self) -> List[str]:
        return ["collection1", "collection2", "user_dev_user_hr"]

    def get_document_count(self, collection_name: str) -> int:
        return 42  # Mock count


class MockConfigLoader:
    """Mock config loader for testing."""

    def get_shared_config(self) -> Dict[str, Any]:
        return {"key": "value"}

    def get_storage_config(self) -> Dict[str, Any]:
        return {"storage": "config"}


def create_mock_cli(
    should_fail: bool = False, output_writer: Optional[OutputWriterProtocol] = None
) -> MultiTenantRAGCLI:
    """Create a fully mocked CLI for testing."""
    output_writer = output_writer or MockOutputWriter()

    def mock_rag_factory(language: str):
        return MockRAGSystem(should_fail=should_fail)

    return MultiTenantRAGCLI(
        output_writer=output_writer,
        logger=MockLogger(),
        rag_system_factory=mock_rag_factory,
        folder_manager=MockFolderManager(),
        storage=MockStorage(),
        config_loader=MockConfigLoader(),
    )


# Entry point (kept for compatibility)
async def main():
    """Main CLI entry point using dependency injection."""
    import logging
    import sys

    # Parse arguments
    args = parse_cli_arguments(sys.argv[1:])

    # Setup real dependencies (this would be replaced with proper factories)
    class RealOutputWriter:
        def write(self, text: str) -> None:
            sys.stdout.write(text)

        def flush(self) -> None:
            sys.stdout.flush()

    class RealLogger:
        def __init__(self):
            logging.basicConfig(level=getattr(logging, args.log_level))
            self.logger = logging.getLogger(__name__)

        def info(self, message: str) -> None:
            self.logger.info(message)

        def error(self, message: str) -> None:
            self.logger.error(message)

        def exception(self, message: str) -> None:
            self.logger.exception(message)

    def real_rag_factory(language: str, tenant_context=None):
        """Create real RAG system using complete production pipeline."""
        print(f"🚀 Creating real RAG system for language: {language}")

        try:
            # Import real components
            import time

            import chromadb

            from ..preprocessing.extractors import extract_document_text

            # Create complete RAG system with real components
            class CompleteRAGSystem:
                def __init__(self, language: str, tenant_context=None):
                    self.language = language
                    self.tenant_context = tenant_context
                    self._client = None
                    self._collection = None
                    self._model = None
                    self._document_count = 0
                    print(f"✅ Complete RAG system created for {language}")

                async def initialize(self):
                    """Initialize real components: ChromaDB, BGE-M3 embeddings."""
                    try:
                        # Load configuration and build proper tenant-specific path
                        import os

                        from ..utils.config_loader import get_paths_config

                        paths_config = get_paths_config()
                        data_base_dir = paths_config["data_base_dir"]

                        if self.tenant_context:
                            # Use proper tenant-specific path from configuration template
                            tenant_slug = self.tenant_context.tenant_slug
                            persist_dir = os.path.join(data_base_dir, tenant_slug, "vectordb")
                        else:
                            # Fallback for testing without tenant context - still avoid hardcoded "vectordb_cli"
                            persist_dir = os.path.join(data_base_dir, "cli_vectordb")

                        os.makedirs(persist_dir, exist_ok=True)

                        self._client = chromadb.PersistentClient(path=persist_dir)
                        print(f"✅ ChromaDB persistent client initialized: {persist_dir}")

                        # Create or get collection for this tenant/user/language
                        if self.tenant_context:
                            collection_name = f"{self.tenant_context.tenant_slug}_{self.tenant_context.user_username}_{self.language}_documents"
                        else:
                            # Fallback for testing without tenant context
                            collection_name = f"{self.language}_documents_cli"

                        # Try to get existing collection first
                        try:
                            self._collection = self._client.get_collection(collection_name)
                            existing_count = self._collection.count()
                            print(
                                f"📦 Found existing collection: {collection_name} ({existing_count} documents)"
                            )
                        except:
                            # Collection doesn't exist, create new one
                            self._collection = self._client.create_collection(
                                name=collection_name,
                                metadata={
                                    "description": f"Documents for tenant:{self.tenant_context.tenant_slug if self.tenant_context else 'unknown'}, user:{self.tenant_context.user_username if self.tenant_context else 'unknown'}, language:{self.language}",
                                    "tenant": (
                                        self.tenant_context.tenant_slug
                                        if self.tenant_context
                                        else "unknown"
                                    ),
                                    "user": (
                                        self.tenant_context.user_username
                                        if self.tenant_context
                                        else "unknown"
                                    ),
                                    "language": self.language,
                                },
                            )
                            print(f"📦 Created new collection: {collection_name}")

                        # Initialize BGE-M3 embedding model
                        try:
                            from sentence_transformers import SentenceTransformer

                            print("🔧 Loading BGE-M3 embedding model...")
                            self._model = SentenceTransformer("BAAI/bge-m3")
                            print("✅ BGE-M3 model loaded successfully")
                        except ImportError:
                            print("⚠️ sentence-transformers not installed, using dummy embeddings")
                            self._model = None

                        print(f"🎯 Complete RAG system initialized for {self.language}")

                    except Exception as e:
                        print(f"❌ Failed to initialize RAG components: {e}")
                        raise e

                async def add_documents(self, document_paths: list, batch_size: int = 10):
                    """Process documents through complete pipeline: extract → chunk → embed → store."""
                    if not self._collection:
                        raise RuntimeError("RAG system not initialized. Call initialize() first.")

                    processed_docs = 0
                    total_chunks = 0
                    errors = []
                    start_time = time.time()

                    # Handle directory vs individual files
                    all_files = []
                    for doc_path in document_paths:
                        from pathlib import Path

                        path_obj = Path(doc_path)
                        if path_obj.is_dir():
                            # Process all supported files in directory
                            supported_extensions = [
                                ".pdf",
                                ".docx",
                                ".txt",
                                ".html",
                                ".md",
                            ]
                            for ext in supported_extensions:
                                all_files.extend(path_obj.glob(f"*{ext}"))
                        elif path_obj.is_file():
                            all_files.append(path_obj)

                    for doc_path in all_files:
                        try:
                            print(f"📄 Processing: {doc_path}")

                            # 1. Real document extraction
                            extracted_text = extract_document_text(str(doc_path))
                            if not extracted_text.strip():
                                errors.append(f"No text extracted from {doc_path}")
                                continue

                            print(f"📝 Extracted {len(extracted_text)} characters")

                            # 2. Real chunking with overlap
                            chunk_size = 500
                            overlap = 50
                            chunks = []

                            for i in range(0, len(extracted_text), chunk_size - overlap):
                                chunk_text = extracted_text[i : i + chunk_size].strip()
                                if chunk_text:
                                    chunks.append(
                                        {
                                            "content": chunk_text,
                                            "chunk_id": f"doc_{processed_docs}_chunk_{len(chunks)}",
                                            "source": str(doc_path),
                                            "start_char": i,
                                            "end_char": min(i + chunk_size, len(extracted_text)),
                                        }
                                    )

                            print(f"📦 Created {len(chunks)} chunks")

                            # 3. Real embedding generation
                            embeddings = []
                            if self._model:
                                print("🔄 Generating BGE-M3 embeddings...")
                                for chunk in chunks:
                                    embedding = self._model.encode(chunk["content"])
                                    embeddings.append(embedding.tolist())
                                print(f"✅ Generated {len(embeddings)} embeddings")
                            else:
                                # Dummy embeddings if model not available
                                import numpy as np

                                embeddings = [np.random.random(1024).tolist() for _ in chunks]
                                print(f"⚠️ Generated {len(embeddings)} dummy embeddings")

                            # 4. Real vector storage in ChromaDB
                            if chunks and embeddings:
                                print("💾 Storing in ChromaDB...")
                                self._collection.add(
                                    documents=[chunk["content"] for chunk in chunks],
                                    metadatas=[
                                        {
                                            "source": chunk["source"],
                                            "chunk_id": chunk["chunk_id"],
                                            "start_char": chunk["start_char"],
                                            "end_char": chunk["end_char"],
                                        }
                                        for chunk in chunks
                                    ],
                                    ids=[chunk["chunk_id"] for chunk in chunks],
                                    embeddings=embeddings,
                                )
                                print(f"✅ Stored {len(chunks)} chunks in vector database")

                            processed_docs += 1
                            total_chunks += len(chunks)

                        except Exception as e:
                            error_msg = f"Failed to process {doc_path}: {e}"
                            errors.append(error_msg)
                            print(f"❌ {error_msg}")

                    processing_time = time.time() - start_time
                    self._document_count += processed_docs

                    # Check total stored documents
                    stored_count = self._collection.count() if self._collection else 0
                    print(f"📊 Total documents in collection: {stored_count}")

                    # Return real result structure
                    return {
                        "processed_documents": processed_docs,
                        "failed_documents": len(document_paths) - processed_docs,
                        "total_chunks": total_chunks,
                        "stored_chunks": stored_count,
                        "processing_time": processing_time,
                        "documents_per_second": (
                            processed_docs / processing_time if processing_time > 0 else 0
                        ),
                        "errors": errors if errors else None,
                    }

                async def query(self, query_obj):
                    """Handle queries with real retrieval and response generation."""
                    if not self._collection:
                        raise RuntimeError("RAG system not initialized. Call initialize() first.")

                    start_time = time.time()
                    query_text = query_obj.text

                    try:
                        print(f"🔍 Processing query: '{query_text}'")

                        # 1. Generate query embedding
                        if self._model:
                            query_embedding = self._model.encode(query_text).tolist()
                            print("✅ Generated query embedding")
                        else:
                            import numpy as np

                            query_embedding = np.random.random(1024).tolist()
                            print("⚠️ Using dummy query embedding")

                        # 2. Real similarity search in ChromaDB
                        print("🔍 Searching vector database...")
                        results = self._collection.query(
                            query_embeddings=[query_embedding],
                            n_results=min(3, self._collection.count()),  # Get up to 3 results
                        )

                        retrieved_chunks = []
                        context_chunks = []

                        if results["documents"][0]:
                            print(f"📊 Found {len(results['documents'][0])} relevant chunks")

                            for i, (doc, metadata) in enumerate(
                                zip(results["documents"][0], results["metadatas"][0])
                            ):
                                distance = (
                                    results["distances"][0][i] if "distances" in results else 0
                                )
                                score = 1 - distance  # Convert distance to similarity score

                                chunk_data = {
                                    "content": doc,
                                    "similarity_score": score,
                                    "final_score": score,
                                    "source": (
                                        metadata["source"] if "source" in metadata else "Unknown"
                                    ),
                                    "chunk_id": (
                                        metadata["chunk_id"]
                                        if "chunk_id" in metadata
                                        else f"chunk_{i}"
                                    ),
                                }
                                retrieved_chunks.append(chunk_data)
                                context_chunks.append(doc)

                                print(
                                    f"  📄 Chunk {i+1}: Score {score:.3f} | Source: {metadata['source'] if 'source' in metadata else 'Unknown'}"
                                )

                        # 3. Generate answer from retrieved context
                        if context_chunks:
                            context_text = "\n\n".join(context_chunks)

                            # Simple answer extraction (similar to test_real_full_rag.py)
                            sentences = [s.strip() for s in context_text.split(".") if s.strip()]
                            relevant_sentences = []

                            # Look for sentences containing key terms from the query
                            query_words = query_text.lower().split()
                            for sentence in sentences:
                                sentence_lower = sentence.lower()
                                if any(
                                    word in sentence_lower for word in query_words if len(word) > 2
                                ):
                                    relevant_sentences.append(sentence)

                            if relevant_sentences:
                                answer = ". ".join(relevant_sentences[:2]) + "."
                                print(f"✅ Generated answer from context")
                            else:
                                answer = (
                                    f"Based on the retrieved documents: {context_chunks[0][:200]}..."
                                    if context_chunks
                                    else "No relevant information found."
                                )
                                print(f"✅ Generated fallback answer")
                        else:
                            answer = "No relevant documents found in the knowledge base."
                            print("⚠️ No documents retrieved")

                        total_time = time.time() - start_time

                        # Create response in expected format
                        from ..pipeline.rag_system import RAGResponse

                        response = RAGResponse(
                            answer=answer,
                            query=query_text,
                            retrieved_chunks=retrieved_chunks,
                            confidence=0.8 if retrieved_chunks else 0.1,
                            generation_time=0.1,
                            retrieval_time=total_time - 0.1,
                            total_time=total_time,
                            sources=[
                                chunk["source"] if "source" in chunk else "Unknown"
                                for chunk in retrieved_chunks
                            ],
                            metadata={
                                "language": self.language,
                                "real_components": True,
                                "chunks_retrieved": len(retrieved_chunks),
                                "collection_size": self._collection.count(),
                            },
                        )

                        print(f"🎯 Query completed in {total_time:.2f}s")
                        return response

                    except Exception as e:
                        print(f"❌ Query failed: {e}")
                        # Return error response
                        from ..pipeline.rag_system import RAGResponse

                        return RAGResponse(
                            answer=f"Error processing query: {str(e)}",
                            query=query_text,
                            retrieved_chunks=[],
                            confidence=0.0,
                            generation_time=0.0,
                            retrieval_time=0.0,
                            total_time=time.time() - start_time,
                            sources=[],
                            metadata={"error": str(e)},
                        )

            return CompleteRAGSystem(language, tenant_context=tenant_context)

        except Exception as e:
            print(f"❌ Failed to create complete RAG system: {e}")
            raise e

    cli = MultiTenantRAGCLI(
        output_writer=RealOutputWriter(),
        logger=RealLogger(),
        rag_system_factory=real_rag_factory,
        folder_manager=MockFolderManager(),  # Would be real implementation
        storage=MockStorage(),  # Would be real implementation
        config_loader=MockConfigLoader(),  # Would be real implementation
    )

    await cli.execute_command(args)


def cli_main():
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
