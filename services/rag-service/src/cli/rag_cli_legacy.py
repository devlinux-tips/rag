#!/usr/bin/env python3
"""
Multi-tenant RAG System CLI with tenant/user context support.

Usage:
    python -m src.cli.rag_cli --tenant development --user dev_user --language hr query "Što je RAG sustav?"
    python -m src.cli.rag_cli --tenant development --user dev_user --language hr process-docs ./docs/
    python -m src.cli.rag_cli --tenant development --user dev_user list-collections
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models.multitenant_models import (DEFAULT_DEVELOPMENT_TENANT,
                                           DEFAULT_DEVELOPMENT_USER, Tenant,
                                           TenantUserContext, User)
from src.pipeline.rag_system import RAGQuery, RAGSystem
from src.utils.folder_manager import TenantFolderManager
from src.vectordb.storage import ChromaDBStorage


class MultiTenantRAGCLI:
    """Multi-tenant RAG System CLI."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def setup_logging(self, level: str = "INFO"):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    def create_tenant_context(
        self, tenant_slug: str, user_id: str
    ) -> TenantUserContext:
        """Create tenant/user context from CLI arguments."""
        if tenant_slug == "development" and user_id == "dev_user":
            # Use default development context
            return TenantUserContext(
                tenant=DEFAULT_DEVELOPMENT_TENANT, user=DEFAULT_DEVELOPMENT_USER
            )
        else:
            # Create custom tenant/user (for production use)
            tenant = Tenant(
                id=f"tenant:{tenant_slug}",
                name=f"Tenant {tenant_slug.title()}",
                slug=tenant_slug,
                description=f"Tenant for {tenant_slug}",
            )
            user = User(
                id=f"user:{user_id}",
                tenant_id=tenant.id,
                email=f"{user_id}@{tenant_slug}.example.com",
                username=user_id,
                full_name=f"User {user_id}",
            )
            return TenantUserContext(tenant=tenant, user=user)

    async def query_command(
        self,
        context: TenantUserContext,
        language: str,
        query_text: str,
        top_k: int = 5,
        return_sources: bool = True,
    ) -> None:
        """Execute a query against the RAG system."""
        print(
            f"🔍 Querying RAG system (tenant: {context.tenant.slug}, user: {context.user.username})"
        )
        print(f"📝 Query: {query_text}")
        print(f"🌐 Language: {language}")
        print()

        try:
            # Initialize RAG system with multi-tenant support
            rag = RAGSystem(language=language)
            await rag.initialize()

            # Create query
            query = RAGQuery(
                text=query_text,
                top_k=top_k,
                return_sources=return_sources,
                return_debug_info=True,
            )

            # Execute query with tenant context
            start_time = time.time()
            response = await rag.query(query)
            query_time = time.time() - start_time

            # Display results
            print("=" * 60)
            print("📊 QUERY RESULTS")
            print("=" * 60)
            print(f"💬 Answer: {response.answer}")
            print()

            if response.sources:
                print("📚 Sources:")
                for i, source in enumerate(response.sources, 1):
                    print(f"  {i}. {source}")
                print()

            print(f"⚡ Query time: {query_time:.2f}s")
            print(f"📄 Documents retrieved: {len(response.retrieved_chunks)}")

            if response.retrieved_chunks:
                print("\n🔍 Retrieved chunks:")
                for i, chunk in enumerate(response.retrieved_chunks, 1):
                    score = chunk.get("similarity_score", 0)
                    final_score = chunk.get("final_score", score)
                    source = chunk.get("source", "Unknown")
                    print(f"  {i}. Score: {final_score:.3f} | Source: {source}")
                    print(f"     Content: {chunk['content'][:100]}...")
                    print()

        except Exception as e:
            print(f"❌ Query failed: {e}")
            self.logger.exception("Query execution failed")

    async def process_documents_command(
        self, context: TenantUserContext, language: str, docs_path: str
    ) -> None:
        """Process documents for tenant/user."""
        print(
            f"📁 Processing documents for tenant: {context.tenant.slug}, user: {context.user.username}"
        )
        print(f"📂 Documents path: {docs_path}")
        print(f"🌐 Language: {language}")

        try:
            # Ensure folder structure exists
            folder_manager = TenantFolderManager()
            success = folder_manager.ensure_context_folders(context, language)

            if not success:
                print("❌ Failed to create folder structure")
                return

            # Initialize RAG system
            rag = RAGSystem(language=language)
            await rag.initialize()

            # Process documents
            start_time = time.time()
            result = await rag.process_documents(docs_path)
            process_time = time.time() - start_time

            if result:
                print(f"✅ Documents processed successfully in {process_time:.2f}s")
                print(f"📊 Processing result: {result}")
            else:
                print("⚠️  Document processing completed with warnings")

        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            self.logger.exception("Document processing failed")

    async def list_collections_command(
        self, context: TenantUserContext, language: str
    ) -> None:
        """List collections for tenant/user."""
        print(f"📋 Listing collections for tenant: {context.tenant.slug}")
        print(f"👤 User: {context.user.username}")
        print(f"🌐 Language: {language}")
        print()

        try:
            folder_manager = TenantFolderManager()
            collections_info = folder_manager.get_collection_storage_paths(
                context, language
            )

            print("📦 ChromaDB Collections:")
            print(f"  👤 User collection: {collections_info['user_collection_name']}")
            print(
                f"  🏢 Tenant collection: {collections_info['tenant_collection_name']}"
            )
            print(f"  📁 Base path: {collections_info['base_path']}")
            print()

            # Try to connect to ChromaDB and get actual stats
            try:
                storage = ChromaDBStorage()
                collections = storage.list_collections()

                print("🗃️  Available collections:")
                for collection in collections:
                    print(f"  - {collection}")

                if collections_info["user_collection_name"] in collections:
                    count = storage.get_document_count(
                        collections_info["user_collection_name"]
                    )
                    print(f"  👤 User collection document count: {count}")

                if collections_info["tenant_collection_name"] in collections:
                    count = storage.get_document_count(
                        collections_info["tenant_collection_name"]
                    )
                    print(f"  🏢 Tenant collection document count: {count}")

            except Exception as e:
                print(f"⚠️  Could not connect to ChromaDB: {e}")

        except Exception as e:
            print(f"❌ Failed to list collections: {e}")
            self.logger.exception("Collection listing failed")

    async def create_folders_command(
        self, context: TenantUserContext, languages: List[str]
    ) -> None:
        """Create folder structure for tenant/user."""
        print(f"🏗️  Creating folder structure for tenant: {context.tenant.slug}")
        print(f"👤 User: {context.user.username}")
        print(f"🌐 Languages: {', '.join(languages)}")

        try:
            folder_manager = TenantFolderManager()

            success, created_folders = folder_manager.create_tenant_folder_structure(
                context.tenant, context.user, languages
            )

            if success:
                print(f"✅ Created {len(created_folders)} folders successfully")
                print("\n📁 Created folders:")
                for folder in created_folders:
                    print(f"  - {folder}")
            else:
                print("❌ Failed to create folder structure")

        except Exception as e:
            print(f"❌ Folder creation failed: {e}")
            self.logger.exception("Folder creation failed")

    async def status_command(self, context: TenantUserContext, language: str) -> None:
        """Show system status for tenant/user."""
        print(f"📊 System status for tenant: {context.tenant.slug}")
        print(f"👤 User: {context.user.username}")
        print(f"🌐 Language: {language}")
        print()

        try:
            # Test RAG system initialization
            try:
                rag = RAGSystem(language=language)
                await rag.initialize()
                print("✅ RAG System: Initialized successfully")
            except Exception as e:
                print(f"❌ RAG System: Failed to initialize ({e})")

            # Check folder structure
            folder_manager = TenantFolderManager()
            paths = folder_manager.get_tenant_folder_structure(
                context.tenant, context.user, language
            )

            existing_folders = []
            missing_folders = []

            for path_name, path in paths.items():
                if path.exists():
                    existing_folders.append(f"{path_name}: {path}")
                else:
                    missing_folders.append(f"{path_name}: {path}")

            print(
                f"📁 Folder structure ({len(existing_folders)} existing, {len(missing_folders)} missing):"
            )
            for folder in existing_folders[:5]:  # Show first 5
                print(f"  ✅ {folder}")
            if len(existing_folders) > 5:
                print(f"  ... and {len(existing_folders) - 5} more")

            if missing_folders:
                print("  Missing folders:")
                for folder in missing_folders[:3]:  # Show first 3
                    print(f"  ❌ {folder}")

            # Test configuration loading
            try:
                from src.utils.config_loader import (get_shared_config,
                                                     get_storage_config)

                shared_config = get_shared_config()
                storage_config = get_storage_config()
                print("✅ Configuration: Loaded successfully")
            except Exception as e:
                print(f"❌ Configuration: Failed to load ({e})")

        except Exception as e:
            print(f"❌ Status check failed: {e}")
            self.logger.exception("Status check failed")

    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Multi-tenant RAG System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Query the system
  python -m src.cli.rag_cli --tenant development --user dev_user --language hr query "Što je RAG sustav?"

  # Process documents
  python -m src.cli.rag_cli --tenant development --user dev_user --language hr process-docs ./data/documents/

  # List collections
  python -m src.cli.rag_cli --tenant acme --user john --language en list-collections

  # Create folder structure
  python -m src.cli.rag_cli --tenant acme --user john create-folders --languages hr en

  # Check system status
  python -m src.cli.rag_cli --tenant development --user dev_user --language hr status
            """,
        )

        # Global options
        parser.add_argument(
            "--tenant",
            default="development",
            help="Tenant slug (default: development)",
        )
        parser.add_argument(
            "--user", default="dev_user", help="User ID (default: dev_user)"
        )
        parser.add_argument(
            "--language",
            choices=["hr", "en", "multilingual"],
            default="hr",
            help="Language code (default: hr)",
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
        query_parser.add_argument(
            "--no-sources", action="store_true", help="Don't return sources"
        )

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

        return parser

    async def main(self):
        """Main CLI entry point."""
        parser = self.create_parser()
        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Setup logging
        self.setup_logging(args.log_level)

        # Create tenant/user context
        context = self.create_tenant_context(args.tenant, args.user)

        print(f"🚀 Multi-tenant RAG System CLI")
        print(f"🏢 Tenant: {context.tenant.name} ({context.tenant.slug})")
        print(f"👤 User: {context.user.full_name} ({context.user.username})")
        print(f"🌐 Language: {args.language}")
        print("=" * 60)

        # Execute command
        try:
            if args.command == "query":
                await self.query_command(
                    context,
                    args.language,
                    args.query_text,
                    args.top_k,
                    not args.no_sources,
                )
            elif args.command == "process-docs":
                await self.process_documents_command(
                    context, args.language, args.docs_path
                )
            elif args.command == "list-collections":
                await self.list_collections_command(context, args.language)
            elif args.command == "create-folders":
                await self.create_folders_command(context, args.languages)
            elif args.command == "status":
                await self.status_command(context, args.language)
            else:
                print(f"❌ Unknown command: {args.command}")
                parser.print_help()

        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
        except Exception as e:
            print(f"❌ Command failed: {e}")
            self.logger.exception("Command execution failed")


def main():
    """Entry point for CLI."""
    cli = MultiTenantRAGCLI()
    asyncio.run(cli.main())


if __name__ == "__main__":
    main()
