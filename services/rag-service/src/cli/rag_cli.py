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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..utils.logging_factory import log_component_start, log_error_context, setup_system_logging


# Pure data structures
@dataclass
class CLIArgs:
    """CLI arguments - pure data structure."""

    command: str | None
    tenant: str
    user: str
    language: str
    log_level: str
    # Scope-related args
    scope: str = "user"  # user, tenant, feature, global
    feature_name: str | None = None  # For feature scope (e.g., "narodne-novine")
    # Command-specific args
    query_text: str | None = None
    top_k: int | None = None
    no_sources: bool = False
    docs_path: str | None = None
    languages: list[str] | None = None
    # Data management args
    confirm: bool = False
    dry_run: bool = False


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
    sources: list[str]
    query_time: float
    documents_retrieved: int
    retrieved_chunks: list[dict[str, Any]]
    error_message: str | None = None


@dataclass
class DocumentProcessingResult:
    """Document processing result - pure data structure."""

    success: bool
    processing_time: float
    processing_result: dict[str, Any] | None = None
    error_message: str | None = None


@dataclass
class CollectionInfo:
    """Collection information - pure data structure."""

    user_collection_name: str
    tenant_collection_name: str
    base_path: str
    available_collections: list[str]
    document_counts: dict[str, int]


@dataclass
class SystemStatus:
    """System status information - pure data structure."""

    rag_system_status: str  # "initialized", "failed"
    folder_structure: dict[str, bool]  # path -> exists
    config_status: str  # "loaded", "failed"
    details: dict[str, Any]
    error_messages: list[str] = field(default_factory=list)


@dataclass
class DataClearResult:
    """Result of data clearing operation - pure data structure."""

    success: bool
    cleared_paths: list[str]
    preserved_paths: list[str]
    errors: list[str]
    message: str


# Protocol definitions for dependency injection
class OutputWriterProtocol(Protocol):
    """Protocol for output writing (stdout, file, etc.)."""

    def write(self, text: str) -> None: ...

    def flush(self) -> None: ...


class LoggerProtocol(Protocol):
    """Protocol for logging."""

    def info(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...

    def exception(self, message: str) -> None: ...


class RAGSystemProtocol(Protocol):
    """Protocol for RAG system operations."""

    async def initialize(self) -> None: ...

    async def query(self, query: Any) -> Any: ...

    async def add_documents(self, document_paths: list[str]) -> dict[str, Any]: ...


# FolderManagerProtocol removed - Weaviate doesn't need local folder management
# Collection names come from TenantUserContext.get_collection_name(language)


class StorageProtocol(Protocol):
    """Protocol for storage operations."""

    def list_collections(self) -> list[str]: ...

    def get_document_count(self, collection_name: str) -> int: ...


class ConfigLoaderProtocol(Protocol):
    """Protocol for configuration loading."""

    def get_shared_config(self) -> dict[str, Any]: ...

    def get_storage_config(self) -> dict[str, Any]: ...


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
        raise ValueError("Tenant slug must contain only alphanumeric characters, hyphens, and underscores")

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


def format_query_results(result: QueryResult, context: TenantContext, language: str) -> list[str]:
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

    lines.extend([f"⚡ Query time: {result.query_time:.2f}s", f"📄 Documents retrieved: {result.documents_retrieved}"])

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
    result: DocumentProcessingResult, context: TenantContext, language: str, docs_path: str
) -> list[str]:
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


def format_collection_info(collection_info: CollectionInfo, context: TenantContext, language: str) -> list[str]:
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


def format_system_status(status: SystemStatus, context: TenantContext, language: str) -> list[str]:
    """Format system status using pure logic."""
    lines = [
        f"📊 System status for tenant: {context.tenant_slug}",
        f"👤 User: {context.user_username}",
        f"🌐 Language: {language}",
        "",
    ]

    # RAG System status
    if status.rag_system_status == "created":
        lines.append("✅ RAG System: Created successfully (ready for initialization)")
    elif status.rag_system_status == "initialized":
        lines.append("✅ RAG System: Fully initialized")
    else:
        lines.append("❌ RAG System: Failed to create")

    # Folder structure
    existing_folders = [path for path, exists in status.folder_structure.items() if exists]
    missing_folders = [path for path, exists in status.folder_structure.items() if not exists]

    lines.append(f"📁 Folder structure ({len(existing_folders)} existing, {len(missing_folders)} missing):")
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


def format_create_folders_result(result: dict[str, Any], context: TenantContext) -> list[str]:
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


def format_clear_data_result(result: DataClearResult, context: TenantContext, language: str) -> list[str]:
    """Format clear-data result using pure logic."""
    lines = [
        "🧹 Data Clearing Result",
        f"🏢 Tenant: {context.tenant_slug}",
        f"👤 User: {context.user_username}",
        f"🌐 Language: {language}",
        "=" * 50,
    ]

    if result.success:
        lines.extend(["✅ Data clearing completed successfully", f"📝 {result.message}", ""])

        if result.cleared_paths:
            lines.append(f"🗑️  Cleared paths ({len(result.cleared_paths)}):")
            for path in result.cleared_paths[:10]:  # Show first 10
                lines.append(f"  ✅ {path}")
            if len(result.cleared_paths) > 10:
                lines.append(f"  ... and {len(result.cleared_paths) - 10} more")
            lines.append("")

        if result.preserved_paths:
            lines.append(f"🔒 Preserved paths ({len(result.preserved_paths)}):")
            for path in result.preserved_paths[:5]:  # Show first 5
                lines.append(f"  💾 {path}")
            if len(result.preserved_paths) > 5:
                lines.append(f"  ... and {len(result.preserved_paths) - 5} more")
            lines.append("")

    else:
        lines.extend(["❌ Data clearing failed", f"📝 {result.message}", ""])

        if result.errors:
            lines.append("❌ Errors encountered:")
            for error in result.errors:
                lines.append(f"  • {error}")

    return lines


def format_reprocess_result(result: dict[str, Any], context: TenantContext, language: str) -> list[str]:
    """Format reprocess result using pure logic."""
    lines = [
        "🔄 Document Reprocessing Result",
        f"🏢 Tenant: {context.tenant_slug}",
        f"👤 User: {context.user_username}",
        f"🌐 Language: {language}",
        "=" * 50,
    ]

    success = result["success"]
    message = result["message"]

    if success:
        lines.extend(["✅ Document reprocessing completed successfully", f"📝 {message}", ""])

        # Show clear results
        if "clear_result" not in result:
            raise RuntimeError("Missing 'clear_result' in reprocessing operation result")
        clear_result = result["clear_result"]
        if clear_result:
            lines.extend(
                [
                    "🧹 Data Clearing Phase:",
                    f"  ✅ Cleared {len(clear_result.cleared_paths)} paths",
                    f"  🔒 Preserved {len(clear_result.preserved_paths)} paths",
                    "",
                ]
            )

        # Show process results
        if "process_result" not in result:
            raise RuntimeError("Missing 'process_result' in reprocessing operation result")
        process_result = result["process_result"]
        if process_result:
            lines.extend(
                ["📄 Document Processing Phase:", f"  📝 Processing time: {process_result.processing_time:.2f}s", ""]
            )

    else:
        lines.extend(["❌ Document reprocessing failed", f"📝 {message}", ""])

        # Show any partial results or errors
        if "clear_result" in result:
            clear_result = result["clear_result"]
            if clear_result and clear_result.errors:
                lines.extend(["❌ Clear phase errors:", *[f"  • {error}" for error in clear_result.errors], ""])

    lines.extend(
        [
            "💡 Next steps:",
            f"  • Query the system to verify reprocessing: --language {language} query 'test'",
            f"  • Check collections: --language {language} list-collections",
        ]
    )

    return lines


def parse_cli_arguments(args: list[str]) -> CLIArgs:
    """Parse command-line arguments using pure logic."""
    parser = argparse.ArgumentParser(
        description="Multi-tenant RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query the system (Croatian)
  python rag.py --tenant development --user dev_user --language hr query "Što je RAG sustav?"

  # Process documents (English)
  python rag.py --tenant development --user dev_user --language en process-docs ./data/development/users/dev_user/documents/en/

  # Check system status
  python rag.py --tenant development --user dev_user --language hr status

  # Clear data (dry-run first to see what would be cleared)
  python rag.py --tenant development --user dev_user --language hr clear-data --dry-run
  python rag.py --tenant development --user dev_user --language hr clear-data --confirm

  # Reprocess all documents from scratch
  python rag.py --tenant development --user dev_user --language en reprocess --confirm

  # List collections
  python rag.py --tenant development --user dev_user --language hr list-collections
        """,
    )

    # Global options
    parser.add_argument("--tenant", default="development", help="Tenant slug (default: development)")
    parser.add_argument("--user", default="dev_user", help="User ID (default: dev_user)")
    parser.add_argument(
        "--language", choices=["hr", "en", "multilingual"], default="en", help="Language code (default: en)"
    )
    parser.add_argument(
        "--scope",
        choices=["user", "tenant", "feature", "global"],
        default="user",
        help="Data scope: user (default), tenant, feature, or global",
    )
    parser.add_argument("--feature", help="Feature name when scope is 'feature' (e.g., narodne-novine)")
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query_text", help="Query text to search for")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    query_parser.add_argument("--no-sources", action="store_true", help="Don't return sources")

    # Process documents command
    process_parser = subparsers.add_parser("process-docs", help="Process documents")
    process_parser.add_argument("docs_path", help="Path to documents to process")

    # List collections command
    subparsers.add_parser("list-collections", help="List ChromaDB collections")

    # Create folders command
    folders_parser = subparsers.add_parser("create-folders", help="Create tenant/user folder structure")
    folders_parser.add_argument("--languages", nargs="+", default=["hr", "en"], help="Languages to create folders for")

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # Data management commands
    clear_parser = subparsers.add_parser(
        "clear-data", help="Clear processed data, vectors, and caches for tenant/user/language"
    )
    clear_parser.add_argument("--confirm", action="store_true", help="Confirm data deletion (required for safety)")
    clear_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be cleared without actually clearing"
    )

    reprocess_parser = subparsers.add_parser(
        "reprocess", help="Clear data and reprocess all documents for tenant/user/language"
    )
    reprocess_parser.add_argument(
        "--confirm", action="store_true", help="Confirm data deletion and reprocessing (required for safety)"
    )
    reprocess_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without actually doing it"
    )

    parsed_args = parser.parse_args(args)

    # Validate scope and feature_name combination
    if parsed_args.scope == "feature" and not parsed_args.feature:
        parser.error("When --scope is 'feature', --feature must be specified (e.g., --feature narodne-novine)")

    if parsed_args.feature and parsed_args.scope != "feature":
        parser.error(f"--feature can only be used with --scope feature (current scope: {parsed_args.scope})")

    return CLIArgs(
        command=parsed_args.command,
        tenant=parsed_args.tenant,
        user=parsed_args.user,
        language=parsed_args.language,
        scope=parsed_args.scope,
        feature_name=parsed_args.feature if hasattr(parsed_args, "feature") else None,
        log_level=parsed_args.log_level,
        query_text=getattr(parsed_args, "query_text", None),
        top_k=getattr(parsed_args, "top_k", None),
        no_sources=getattr(parsed_args, "no_sources", False),
        docs_path=getattr(parsed_args, "docs_path", None),
        languages=getattr(parsed_args, "languages", None),
        confirm=getattr(parsed_args, "confirm", False),
        dry_run=getattr(parsed_args, "dry_run", False),
    )


# Main testable CLI class
class MultiTenantRAGCLI:
    """Multi-tenant RAG command-line interface with configurable operations."""

    def __init__(
        self,
        output_writer: OutputWriterProtocol,
        logger: LoggerProtocol,
        rag_system_factory: Any,  # Factory function
        storage: StorageProtocol,
        config_loader: ConfigLoaderProtocol,
    ):
        """Initialize with all dependencies injected."""
        self.output_writer = output_writer
        self.logger = logger
        self.rag_system_factory = rag_system_factory
        self.storage = storage
        self.config_loader = config_loader

    def write_output(self, lines: list[str]) -> None:
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
        scope: str = "user",
        feature_name: str | None = None,
    ) -> QueryResult:
        """Execute query command using pure business logic."""
        try:
            # Import RAGQuery class for proper query object creation
            from src.pipeline.rag_system import RAGQuery

            # Create RAG system (injected factory)
            rag = self.rag_system_factory(
                language=language, tenant_context=context, scope=scope, feature_name=feature_name
            )
            await rag.initialize()

            # Create proper RAGQuery object with scope context
            scope_context = {}
            if feature_name:
                scope_context["feature_name"] = feature_name
            if scope:
                scope_context["scope"] = scope

            query = RAGQuery(
                text=query_text,
                language=language,
                max_results=top_k,
                context_filters=scope_context if scope_context else None,
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
        self,
        context: TenantContext,
        language: str,
        docs_path: str,
        scope: str = "user",
        feature_name: str | None = None,
    ) -> DocumentProcessingResult:
        """Execute document processing command using pure business logic."""
        try:
            from ..utils.multitenant_models import Tenant, TenantUserContext, User

            # Create tenant user context from CLI context
            tenant_obj = Tenant(id=context.tenant_id, name=context.tenant_name, slug=context.tenant_slug)
            user_obj = User(
                id=context.user_id,
                tenant_id=tenant_obj.id,
                email=context.user_email,
                username=context.user_username,
                full_name=context.user_full_name,
            )
            TenantUserContext(tenant=tenant_obj, user=user_obj)

            # Initialize RAG system
            rag = self.rag_system_factory(
                language=language, tenant_context=context, scope=scope, feature_name=feature_name
            )
            await rag.initialize()

            # Handle both file and directory paths
            from pathlib import Path

            path = Path(docs_path)

            if path.is_file():
                # Single file processing
                document_paths = [str(path)]
            elif path.is_dir():
                # Directory processing - find all supported document files recursively
                supported_extensions = {".pdf", ".docx", ".txt", ".md", ".html"}
                document_paths = []
                for ext in supported_extensions:
                    document_paths.extend([str(p) for p in path.rglob(f"*{ext}")])

                if not document_paths:
                    return DocumentProcessingResult(
                        success=False,
                        processing_time=0.0,
                        error_message=f"No supported documents found in directory (searched recursively): {docs_path}",
                    )
            else:
                return DocumentProcessingResult(
                    success=False,
                    processing_time=0.0,
                    error_message=f"Path does not exist or is neither file nor directory: {docs_path}",
                )

            # Process documents
            start_time = time.time()
            result = await rag.add_documents(document_paths)
            processing_time = time.time() - start_time

            return DocumentProcessingResult(success=True, processing_time=processing_time, processing_result=result)

        except Exception as e:
            self.logger.exception("Document processing failed")
            return DocumentProcessingResult(success=False, processing_time=0.0, error_message=str(e))

    async def execute_list_collections_command(self, context: TenantContext, language: str) -> CollectionInfo:
        """Execute list collections command using pure business logic."""
        try:
            from ..utils.multitenant_models import DocumentScope, Tenant, TenantUserContext, User

            # Create tenant user context from CLI context
            tenant_obj = Tenant(id=context.tenant_id, name=context.tenant_name, slug=context.tenant_slug)
            user_obj = User(
                id=context.user_id,
                tenant_id=tenant_obj.id,
                email=context.user_email,
                username=context.user_username,
                full_name=context.user_full_name,
            )
            tenant_user_context = TenantUserContext(tenant=tenant_obj, user=user_obj)

            user_collection = tenant_user_context.get_collection_name(language)
            tenant_scope_context = TenantUserContext(tenant=tenant_obj, user=user_obj, scope=DocumentScope.TENANT)
            tenant_collection = tenant_scope_context.get_collection_name(language)

            available_collections = self.storage.list_collections()

            document_counts = {}
            if user_collection in available_collections:
                document_counts[user_collection] = self.storage.get_document_count(user_collection)

            if tenant_collection in available_collections:
                document_counts[tenant_collection] = self.storage.get_document_count(tenant_collection)

            return CollectionInfo(
                user_collection_name=user_collection,
                tenant_collection_name=tenant_collection,
                base_path="N/A",
                available_collections=available_collections,
                document_counts=document_counts,
            )

        except Exception:
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

        # Test RAG system creation (without full initialization)
        try:
            self.rag_system_factory(language=language, tenant_context=context)
            # Skip full initialization for status check - just test creation
            rag_status = "created"
        except Exception as e:
            rag_status = "failed"
            error_messages.append(f"RAG System: {str(e)}")

        # Weaviate uses cloud-based storage, no local folders needed
        folder_structure: dict[str, Any] = {}

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

    async def execute_create_folders_command(self, context: TenantContext, languages: list[str]) -> dict[str, Any]:
        """Execute create-folders command - not needed for Weaviate (cloud-based)."""
        return {
            "success": True,
            "created_folders": [],
            "existing_folders": [],
            "languages": languages,
            "message": "Weaviate manages collections automatically, no local folder creation needed",
        }

    async def execute_clear_data_command(
        self, context: TenantContext, language: str, dry_run: bool, confirm: bool
    ) -> DataClearResult:
        """Execute clear-data command for tenant/user/language context."""
        import shutil

        if not dry_run and not confirm:
            return DataClearResult(
                success=False,
                cleared_paths=[],
                preserved_paths=[],
                errors=["Must provide --confirm flag to actually clear data"],
                message="Data clearing cancelled - safety confirmation required",
            )

        cleared_paths = []
        preserved_paths = []
        errors = []

        try:
            # Get tenant-specific paths
            tenant_slug = context.tenant_slug
            user_id = context.user_id if hasattr(context, "user_id") else "dev_user"

            # Paths to clear (tenant/user/language specific)
            base_path = Path("./data") / tenant_slug
            paths_to_clear = [
                base_path / "users" / user_id / "processed" / language,
                base_path / "users" / user_id / "cache" / language,
                base_path / "shared" / "processed" / language,
                base_path / "vectordb",  # Clear entire vectordb for tenant
                Path("./models") / tenant_slug / language,  # Language-specific models
                Path("./temp") / tenant_slug / user_id / language,  # Temp files
            ]

            # Preserve paths (show what won't be cleared)
            preserve_paths = [
                base_path / "users" / user_id / "documents",  # Original documents
                base_path / "shared" / "documents",  # Shared documents
                Path("./config"),  # Configuration files
                Path("./src"),  # Source code
            ]

            for path in preserve_paths:
                if path.exists():
                    preserved_paths.append(str(path))

            # Clear data
            for path in paths_to_clear:
                if path.exists():
                    if dry_run:
                        cleared_paths.append(f"[DRY-RUN] Would clear: {path}")
                    else:
                        try:
                            if path.is_file():
                                path.unlink()
                            else:
                                shutil.rmtree(path)
                            cleared_paths.append(str(path))
                        except Exception as e:
                            errors.append(f"Failed to clear {path}: {e}")

            success = len(errors) == 0
            action = "Would clear" if dry_run else "Cleared"
            message = f"{action} {len(cleared_paths)} paths for {tenant_slug}/{user_id}/{language}"

            return DataClearResult(
                success=success,
                cleared_paths=cleared_paths,
                preserved_paths=preserved_paths,
                errors=errors,
                message=message,
            )

        except Exception as e:
            return DataClearResult(
                success=False,
                cleared_paths=cleared_paths,
                preserved_paths=preserved_paths,
                errors=[f"Clear operation failed: {e}"],
                message=f"Failed to clear data for {context.tenant_slug}/{language}",
            )

    async def execute_reprocess_command(
        self,
        context: TenantContext,
        language: str,
        dry_run: bool,
        confirm: bool,
        scope: str = "user",
        feature_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute reprocess command - clear data then process documents."""
        if not dry_run and not confirm:
            return {
                "success": False,
                "clear_result": None,
                "process_result": None,
                "message": "Reprocessing cancelled - safety confirmation required",
            }

        try:
            # Step 1: Clear existing data
            clear_result = await self.execute_clear_data_command(context, language, dry_run, confirm)

            if dry_run:
                return {
                    "success": True,
                    "clear_result": clear_result,
                    "process_result": {"message": "[DRY-RUN] Would reprocess documents after clearing"},
                    "message": f"[DRY-RUN] Would reprocess all documents for {context.tenant_slug}/{language}",
                }

            if not clear_result.success:
                return {
                    "success": False,
                    "clear_result": clear_result,
                    "process_result": None,
                    "message": "Reprocessing failed during clear step",
                }

            # Step 2: Determine document path based on scope
            if scope == "feature" and feature_name:
                # Use config to get the feature document path
                from ..utils.config_loader import get_shared_config

                config = get_shared_config()
                data_base_dir = config["paths"]["data_base_dir"]
                docs_path = f"{data_base_dir}/features/{feature_name}/{language}"
            elif scope == "tenant":
                tenant_slug = context.tenant_slug
                docs_path = f"./data/{tenant_slug}/{language}"
            else:  # Default to user scope
                tenant_slug = context.tenant_slug
                user_username = context.user_username
                docs_path = f"./data/{tenant_slug}/users/{user_username}/{language}"

            process_result = await self.execute_process_documents_command(
                context, language, docs_path, scope, feature_name
            )

            scope_desc = f"{scope}:{feature_name}" if feature_name else scope
            return {
                "success": process_result.success,
                "clear_result": clear_result,
                "process_result": process_result,
                "message": f"Reprocessed documents for {scope_desc}/{language}",
            }

        except Exception as e:
            return {
                "success": False,
                "clear_result": None,
                "process_result": None,
                "message": f"Reprocessing failed: {e}",
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
            f"📊 Scope: {args.scope}" + (f" (feature: {args.feature_name})" if args.feature_name else ""),
            "=" * 60,
        ]
        self.write_output(header)

        try:
            if args.command == "query":
                # Handle None arguments with proper defaults
                query_text = args.query_text or ""
                top_k = args.top_k or 5
                query_result = await self.execute_query_command(
                    context, args.language, query_text, top_k, not args.no_sources, args.scope, args.feature_name
                )
                formatted_output = format_query_results(query_result, context, args.language)
                self.write_output(formatted_output)

            elif args.command == "process-docs":
                # Handle None argument with proper default
                docs_path = args.docs_path or ""
                process_result = await self.execute_process_documents_command(
                    context, args.language, docs_path, args.scope, args.feature_name
                )
                formatted_output = format_processing_results(process_result, context, args.language, docs_path)
                self.write_output(formatted_output)

            elif args.command == "list-collections":
                collection_info = await self.execute_list_collections_command(context, args.language)
                formatted_output = format_collection_info(collection_info, context, args.language)
                self.write_output(formatted_output)

            elif args.command == "status":
                status = await self.execute_status_command(context, args.language)
                formatted_output = format_system_status(status, context, args.language)
                self.write_output(formatted_output)

            elif args.command == "create-folders":
                # Handle None argument with proper default
                languages = args.languages or ["hr", "en"]
                folders_result = await self.execute_create_folders_command(context, languages)
                formatted_output = format_create_folders_result(folders_result, context)
                self.write_output(formatted_output)

            elif args.command == "clear-data":
                clear_result = await self.execute_clear_data_command(context, args.language, args.dry_run, args.confirm)
                formatted_output = format_clear_data_result(clear_result, context, args.language)
                self.write_output(formatted_output)

            elif args.command == "reprocess":
                reprocess_result = await self.execute_reprocess_command(
                    context, args.language, args.dry_run, args.confirm, args.scope, args.feature_name
                )
                formatted_output = format_reprocess_result(reprocess_result, context, args.language)
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
                {"content": "Mock chunk 1", "similarity_score": 0.9, "final_score": 0.9, "source": "mock_doc1.txt"},
                {"content": "Mock chunk 2", "similarity_score": 0.8, "final_score": 0.8, "source": "mock_doc2.txt"},
            ]

        return MockResponse()

    async def add_documents(self, document_paths: list[str]) -> dict[str, Any]:
        if self.should_fail:
            raise Exception("Mock processing failure")

        return {
            "processed_documents": len(document_paths),
            "failed_documents": 0,
            "total_chunks": len(document_paths) * 5,
            "processing_time": 1.0,
            "documents_per_second": len(document_paths),
        }


class MockStorage:
    """Mock storage for testing."""

    def list_collections(self) -> list[str]:
        return ["collection1", "collection2", "user_dev_user_hr"]

    def get_document_count(self, collection_name: str) -> int:
        return 42  # Mock count


class MockConfigLoader:
    """Mock config loader for testing."""

    def get_shared_config(self) -> dict[str, Any]:
        return {"key": "value"}

    def get_storage_config(self) -> dict[str, Any]:
        return {"storage": "config"}


def create_mock_cli(should_fail: bool = False, output_writer: OutputWriterProtocol | None = None) -> MultiTenantRAGCLI:
    """Create a fully mocked CLI for testing."""
    output_writer = output_writer or MockOutputWriter()

    def mock_rag_factory(language: str, tenant_context=None, scope: str = "user", feature_name: str | None = None):
        return MockRAGSystem(should_fail=should_fail)

    return MultiTenantRAGCLI(
        output_writer=output_writer,
        logger=MockLogger(),
        rag_system_factory=mock_rag_factory,
        storage=MockStorage(),
        config_loader=MockConfigLoader(),
    )


# Main entry point
async def main():
    args = parse_cli_arguments(sys.argv[1:])

    # Setup new logging system with configurable backends
    backend_kwargs = {"console": {"level": args.log_level, "colored": True}}

    # Add file logging in debug mode
    if args.log_level.upper() == "DEBUG":
        backend_kwargs["file"] = {"log_file": "logs/rag_debug.log", "format_type": "text"}

    setup_system_logging(["console"], **backend_kwargs)
    log_component_start("cli", "STARTUP", command=args.command, language=args.language)

    class StandardOutputWriter:
        def write(self, text: str) -> None:
            sys.stdout.write(text)

        def flush(self) -> None:
            sys.stdout.flush()

    class StandardLogger:
        def info(self, message: str) -> None:
            log_component_start("cli", "OPERATION", message=message)

        def error(self, message: str) -> None:
            from ..utils.logging_factory import get_system_logger

            logger = get_system_logger()
            logger.error("cli", "OPERATION", message)

        def exception(self, message: str) -> None:
            from ..utils.logging_factory import get_system_logger

            logger = get_system_logger()
            logger.error("cli", "EXCEPTION", message)

    def real_rag_factory(language: str, tenant_context=None, scope: str = "user", feature_name: str | None = None):
        log_component_start("cli", "RAG_FACTORY", language=language, scope=scope, feature_name=feature_name)

        try:
            from ..utils.factories import Tenant, User, create_complete_rag_system

            tenant_obj = None
            user_obj = None
            if tenant_context and scope in ["user", "tenant"]:
                tenant_obj = Tenant(
                    id=tenant_context.tenant_id, name=tenant_context.tenant_name, slug=tenant_context.tenant_slug
                )
                user_obj = User(
                    id=tenant_context.user_id,
                    tenant_id=tenant_obj.id,
                    email=tenant_context.user_email,
                    username=tenant_context.user_username,
                    full_name=tenant_context.user_full_name,
                )

            rag_system = create_complete_rag_system(
                language=language,
                tenant=tenant_obj if scope in ["user", "tenant"] else None,
                user=user_obj if scope == "user" else None,
                scope=scope,
                feature_name=feature_name,
            )

            log_component_start("cli", "RAG_FACTORY_SUCCESS", language=language, scope=scope)
            return rag_system

        except Exception as e:
            log_error_context("cli", "RAG_FACTORY", e, {"language": language, "scope": scope})
            raise e

    class VectorDatabaseStorage:
        def __init__(self):
            self._vector_db = None

        def _ensure_initialized(self):
            if self._vector_db is None:
                from ..utils.config_loader import get_paths_config
                from ..vectordb.database_factory import create_vector_database

                paths_config = get_paths_config()
                db_path_template = paths_config["chromadb_path_template"]
                data_base_dir = paths_config["data_base_dir"]
                db_path = db_path_template.format(data_base_dir=data_base_dir, tenant_slug="development")
                self._vector_db = create_vector_database(db_path=db_path)

        def list_collections(self) -> list[str]:
            try:
                self._ensure_initialized()
                return self._vector_db.list_collections()
            except Exception:
                return []

        def get_document_count(self, collection_name: str) -> int:
            try:
                self._ensure_initialized()
                return self._vector_db.get_collection_size(collection_name)
            except Exception:
                return 0

    class TomlConfigLoader:
        def get_shared_config(self) -> dict[str, Any]:
            from ..utils.config_loader import get_shared_config

            return get_shared_config()

        def get_storage_config(self) -> dict[str, Any]:
            from ..utils.config_loader import load_config

            config = load_config("config")
            return config["vectordb"]

    cli = MultiTenantRAGCLI(
        output_writer=StandardOutputWriter(),
        logger=StandardLogger(),
        rag_system_factory=real_rag_factory,
        storage=VectorDatabaseStorage(),
        config_loader=TomlConfigLoader(),
    )

    await cli.execute_command(args)
    log_component_start("cli", "SHUTDOWN", status="completed")


def cli_main():
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
