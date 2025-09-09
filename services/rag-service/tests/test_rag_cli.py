"""
Comprehensive tests for rag_cli.py demonstrating 100% testability.
Tests pure functions, dependency injection, and CLI scenarios.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
from src.cli.rag_cli import (  # Pure functions; Data classes; Main class; Mock implementations
    CLIArgs, CollectionInfo, DocumentProcessingResult, MockConfigLoader,
    MockFolderManager, MockLogger, MockOutputWriter, MockRAGSystem,
    MockStorage, MultiTenantRAGCLIV2, QueryResult, SystemStatus, TenantContext,
    create_mock_cli, create_tenant_context, format_collection_info,
    format_processing_results, format_query_results, format_system_status,
    parse_cli_arguments, validate_language_code, validate_tenant_slug,
    validate_user_id)


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_validate_language_code_valid(self):
        """Test language code validation with valid inputs."""
        assert validate_language_code("hr") == "hr"
        assert validate_language_code("en") == "en"
        assert validate_language_code("multilingual") == "multilingual"

        # Test normalization
        assert validate_language_code("HR") == "hr"
        assert validate_language_code("  en  ") == "en"

    def test_validate_language_code_invalid(self):
        """Test language code validation with invalid inputs."""
        with pytest.raises(
            ValueError, match="Language code must be a non-empty string"
        ):
            validate_language_code("")

        with pytest.raises(
            ValueError, match="Language code must be a non-empty string"
        ):
            validate_language_code(None)

        with pytest.raises(ValueError, match="Unsupported language"):
            validate_language_code("fr")

    def test_validate_tenant_slug_valid(self):
        """Test tenant slug validation with valid inputs."""
        assert validate_tenant_slug("development") == "development"
        assert validate_tenant_slug("ACME-Corp") == "acme-corp"
        assert validate_tenant_slug("  test_tenant  ") == "test_tenant"
        assert validate_tenant_slug("tenant123") == "tenant123"

    def test_validate_tenant_slug_invalid(self):
        """Test tenant slug validation with invalid inputs."""
        with pytest.raises(ValueError, match="Tenant slug must be a non-empty string"):
            validate_tenant_slug("")

        with pytest.raises(ValueError, match="Tenant slug must be a non-empty string"):
            validate_tenant_slug(None)

        with pytest.raises(ValueError, match="alphanumeric characters"):
            validate_tenant_slug("tenant@invalid")

        with pytest.raises(ValueError, match="alphanumeric characters"):
            validate_tenant_slug("tenant with spaces")

    def test_validate_user_id_valid(self):
        """Test user ID validation with valid inputs."""
        assert validate_user_id("dev_user") == "dev_user"
        assert validate_user_id("  john123  ") == "john123"
        assert validate_user_id("admin") == "admin"

    def test_validate_user_id_invalid(self):
        """Test user ID validation with invalid inputs."""
        with pytest.raises(ValueError, match="User ID must be a non-empty string"):
            validate_user_id("")

        with pytest.raises(ValueError, match="User ID must be a non-empty string"):
            validate_user_id(None)

        with pytest.raises(ValueError, match="at least 2 characters"):
            validate_user_id("a")

    def test_create_tenant_context_development(self):
        """Test creating development tenant context."""
        context = create_tenant_context("development", "dev_user")

        assert context.tenant_slug == "development"
        assert context.tenant_name == "Development Tenant"
        assert context.user_username == "dev_user"
        assert context.user_full_name == "Development User"
        assert "development" in context.user_email

    def test_create_tenant_context_custom(self):
        """Test creating custom tenant context."""
        context = create_tenant_context("acme-corp", "john")

        assert context.tenant_slug == "acme-corp"
        assert context.tenant_name == "Tenant Acme-Corp"
        assert context.user_username == "john"
        assert context.user_full_name == "User John"
        assert "john@acme-corp.example.com" == context.user_email

    def test_format_query_results_success(self):
        """Test formatting successful query results."""
        context = TenantContext(
            tenant_id="tenant:test",
            tenant_name="Test Tenant",
            tenant_slug="test",
            user_id="user:testuser",
            user_email="test@test.com",
            user_username="testuser",
            user_full_name="Test User",
        )

        result = QueryResult(
            success=True,
            answer="Test answer about RAG systems",
            sources=["doc1.txt", "doc2.txt"],
            query_time=1.5,
            documents_retrieved=2,
            retrieved_chunks=[
                {
                    "content": "Chunk 1 content",
                    "similarity_score": 0.9,
                    "final_score": 0.9,
                    "source": "doc1.txt",
                },
                {
                    "content": "Chunk 2 content",
                    "similarity_score": 0.8,
                    "final_score": 0.8,
                    "source": "doc2.txt",
                },
            ],
        )

        formatted = format_query_results(result, context, "hr")

        # Check key elements are present
        formatted_text = "\n".join(formatted)
        assert "QUERY RESULTS" in formatted_text
        assert "Test answer about RAG systems" in formatted_text
        assert "doc1.txt" in formatted_text
        assert "Query time: 1.50s" in formatted_text
        assert "Documents retrieved: 2" in formatted_text

    def test_format_query_results_failure(self):
        """Test formatting failed query results."""
        context = create_tenant_context("test", "testuser")

        result = QueryResult(
            success=False,
            answer="",
            sources=[],
            query_time=0.0,
            documents_retrieved=0,
            retrieved_chunks=[],
            error_message="Test error occurred",
        )

        formatted = format_query_results(result, context, "hr")
        formatted_text = "\n".join(formatted)

        assert "Query failed" in formatted_text
        assert "Test error occurred" in formatted_text
        assert "Tenant: test" in formatted_text

    def test_format_processing_results_success(self):
        """Test formatting successful processing results."""
        context = create_tenant_context("test", "testuser")

        result = DocumentProcessingResult(
            success=True,
            processing_time=5.2,
            processing_result={
                "processed_documents": 10,
                "failed_documents": 0,
                "total_chunks": 50,
            },
        )

        formatted = format_processing_results(result, context, "hr", "/path/to/docs")
        formatted_text = "\n".join(formatted)

        assert "Documents processed successfully" in formatted_text
        assert "5.2s" in formatted_text
        assert "/path/to/docs" in formatted_text

    def test_format_collection_info(self):
        """Test formatting collection information."""
        context = create_tenant_context("test", "testuser")

        collection_info = CollectionInfo(
            user_collection_name="user_testuser_hr",
            tenant_collection_name="tenant_test_hr",
            base_path="/mock/test",
            available_collections=["collection1", "collection2", "user_testuser_hr"],
            document_counts={"user_testuser_hr": 42, "collection1": 15},
        )

        formatted = format_collection_info(collection_info, context, "hr")
        formatted_text = "\n".join(formatted)

        assert "user_testuser_hr" in formatted_text
        assert "tenant_test_hr" in formatted_text
        assert "User collection document count: 42" in formatted_text

    def test_format_system_status(self):
        """Test formatting system status."""
        context = create_tenant_context("test", "testuser")

        status = SystemStatus(
            rag_system_status="initialized",
            folder_structure={"/path1": True, "/path2": True, "/path3": False},
            config_status="loaded",
            details={},
            error_messages=["Warning: Some issue"],
        )

        formatted = format_system_status(status, context, "hr")
        formatted_text = "\n".join(formatted)

        assert "RAG System: Initialized successfully" in formatted_text
        assert "Folder structure (2 existing, 1 missing)" in formatted_text
        assert "Configuration: Loaded successfully" in formatted_text
        assert "Errors:" in formatted_text
        assert "Warning: Some issue" in formatted_text

    def test_parse_cli_arguments_query_command(self):
        """Test parsing query command arguments."""
        args = parse_cli_arguments(
            [
                "--tenant",
                "acme",
                "--user",
                "john",
                "--language",
                "en",
                "query",
                "What is RAG?",
                "--top-k",
                "3",
                "--no-sources",
            ]
        )

        assert args.command == "query"
        assert args.tenant == "acme"
        assert args.user == "john"
        assert args.language == "en"
        assert args.query_text == "What is RAG?"
        assert args.top_k == 3
        assert args.no_sources is True

    def test_parse_cli_arguments_process_docs_command(self):
        """Test parsing process-docs command arguments."""
        args = parse_cli_arguments(
            ["--tenant", "development", "process-docs", "/path/to/documents"]
        )

        assert args.command == "process-docs"
        assert args.tenant == "development"
        assert args.docs_path == "/path/to/documents"

    def test_parse_cli_arguments_defaults(self):
        """Test CLI argument parsing with defaults."""
        args = parse_cli_arguments(["status"])

        assert args.command == "status"
        assert args.tenant == "development"  # default
        assert args.user == "dev_user"  # default
        assert args.language == "hr"  # default
        assert args.log_level == "INFO"  # default


class TestDataClasses:
    """Test data class functionality."""

    def test_cli_args_creation(self):
        """Test CLIArgs data class."""
        args = CLIArgs(
            command="query",
            tenant="test",
            user="testuser",
            language="hr",
            log_level="INFO",
            query_text="Test query",
            top_k=5,
        )

        assert args.command == "query"
        assert args.query_text == "Test query"
        assert args.top_k == 5

    def test_tenant_context_creation(self):
        """Test TenantContext data class."""
        context = TenantContext(
            tenant_id="tenant:test",
            tenant_name="Test Tenant",
            tenant_slug="test",
            user_id="user:testuser",
            user_email="test@test.com",
            user_username="testuser",
            user_full_name="Test User",
        )

        assert context.tenant_slug == "test"
        assert context.user_username == "testuser"

    def test_query_result_creation(self):
        """Test QueryResult data class."""
        result = QueryResult(
            success=True,
            answer="Test answer",
            sources=["doc.txt"],
            query_time=1.0,
            documents_retrieved=1,
            retrieved_chunks=[],
        )

        assert result.success is True
        assert result.answer == "Test answer"
        assert len(result.sources) == 1


class TestMockImplementations:
    """Test mock implementations for dependency injection."""

    def test_mock_output_writer(self):
        """Test mock output writer."""
        writer = MockOutputWriter()
        writer.write("Test line 1\n")
        writer.write("Test line 2")
        writer.flush()

        assert len(writer.written_lines) == 2
        assert writer.written_lines[0] == "Test line 1"
        assert writer.written_lines[1] == "Test line 2"

    def test_mock_logger(self):
        """Test mock logger."""
        logger = MockLogger()
        logger.info("Info message")
        logger.error("Error message")
        logger.exception("Exception message")

        assert len(logger.logs) == 3
        assert logger.logs[0] == ("INFO", "Info message")
        assert logger.logs[1] == ("ERROR", "Error message")
        assert logger.logs[2] == ("EXCEPTION", "Exception message")

    @pytest.mark.asyncio
    async def test_mock_rag_system_success(self):
        """Test mock RAG system successful operations."""
        rag = MockRAGSystem(should_fail=False)

        await rag.initialize()
        assert rag.initialized is True

        # Test query
        query = {"text": "test query"}
        response = await rag.query(query)
        assert hasattr(response, "answer")
        assert "test query" in response.answer

        # Test document processing
        result = await rag.add_documents(["/path/doc.txt"])
        assert result["processed_documents"] == 1
        assert result["failed_documents"] == 0

    @pytest.mark.asyncio
    async def test_mock_rag_system_failure(self):
        """Test mock RAG system failure scenarios."""
        rag = MockRAGSystem(should_fail=True)

        with pytest.raises(Exception, match="Mock initialization failure"):
            await rag.initialize()

        with pytest.raises(Exception, match="Mock query failure"):
            await rag.query({"text": "test"})

        with pytest.raises(Exception, match="Mock processing failure"):
            await rag.add_documents(["/path/doc.txt"])

    def test_mock_folder_manager(self):
        """Test mock folder manager."""
        manager = MockFolderManager()
        context = create_tenant_context("test", "testuser")

        # Test ensure folders
        success = manager.ensure_context_folders(context, "hr")
        assert success is True

        # Test get collection paths
        paths = manager.get_collection_storage_paths(context, "hr")
        assert "user_collection_name" in paths
        assert "tenant_collection_name" in paths

        # Test folder structure
        structure = manager.get_tenant_folder_structure(context, None, "hr")
        assert len(structure) > 0
        assert all(isinstance(path, Path) for path in structure.values())

    def test_mock_storage(self):
        """Test mock storage."""
        storage = MockStorage()

        collections = storage.list_collections()
        assert isinstance(collections, list)
        assert len(collections) > 0

        count = storage.get_document_count("test_collection")
        assert isinstance(count, int)
        assert count > 0

    def test_mock_config_loader(self):
        """Test mock config loader."""
        loader = MockConfigLoader()

        shared_config = loader.get_shared_config()
        assert isinstance(shared_config, dict)

        storage_config = loader.get_storage_config()
        assert isinstance(storage_config, dict)


class TestMultiTenantRAGCLIV2WithMocks:
    """Test CLI with mock dependencies."""

    @pytest.fixture
    def mock_output_writer(self):
        """Mock output writer fixture."""
        return MockOutputWriter()

    @pytest.fixture
    def mock_cli(self, mock_output_writer):
        """Mock CLI fixture."""
        return create_mock_cli(should_fail=False, output_writer=mock_output_writer)

    def test_write_output(self, mock_cli, mock_output_writer):
        """Test output writing."""
        lines = ["Line 1", "Line 2", "Line 3"]
        mock_cli.write_output(lines)

        assert len(mock_output_writer.written_lines) == 3
        assert mock_output_writer.written_lines == lines

    @pytest.mark.asyncio
    async def test_execute_query_command_success(self, mock_cli):
        """Test successful query command execution."""
        context = create_tenant_context("test", "testuser")

        result = await mock_cli.execute_query_command(
            context=context,
            language="hr",
            query_text="Test query",
            top_k=3,
            return_sources=True,
        )

        assert isinstance(result, QueryResult)
        assert result.success is True
        assert result.answer  # Should have content
        assert len(result.sources) > 0
        assert result.query_time > 0

    @pytest.mark.asyncio
    async def test_execute_query_command_failure(self):
        """Test failed query command execution."""
        failing_cli = create_mock_cli(should_fail=True)
        context = create_tenant_context("test", "testuser")

        result = await failing_cli.execute_query_command(
            context=context,
            language="hr",
            query_text="Test query",
            top_k=3,
            return_sources=True,
        )

        assert isinstance(result, QueryResult)
        assert result.success is False
        assert result.error_message  # Should have error message

    @pytest.mark.asyncio
    async def test_execute_process_documents_command_success(self, mock_cli):
        """Test successful document processing command."""
        context = create_tenant_context("test", "testuser")

        result = await mock_cli.execute_process_documents_command(
            context=context, language="hr", docs_path="/path/to/docs"
        )

        assert isinstance(result, DocumentProcessingResult)
        assert result.success is True
        assert result.processing_time > 0
        assert result.processing_result is not None

    @pytest.mark.asyncio
    async def test_execute_list_collections_command(self, mock_cli):
        """Test list collections command."""
        context = create_tenant_context("test", "testuser")

        collection_info = await mock_cli.execute_list_collections_command(
            context=context, language="hr"
        )

        assert isinstance(collection_info, CollectionInfo)
        assert collection_info.user_collection_name
        assert collection_info.tenant_collection_name
        assert len(collection_info.available_collections) > 0

    @pytest.mark.asyncio
    async def test_execute_status_command(self, mock_cli):
        """Test status command execution."""
        context = create_tenant_context("test", "testuser")

        status = await mock_cli.execute_status_command(context=context, language="hr")

        assert isinstance(status, SystemStatus)
        assert status.rag_system_status == "initialized"  # Mock should succeed
        assert status.config_status == "loaded"
        assert isinstance(status.folder_structure, dict)

    @pytest.mark.asyncio
    async def test_execute_command_query(self, mock_cli, mock_output_writer):
        """Test executing query command end-to-end."""
        args = CLIArgs(
            command="query",
            tenant="test",
            user="testuser",
            language="hr",
            log_level="INFO",
            query_text="Test query",
            top_k=3,
            no_sources=False,
        )

        await mock_cli.execute_command(args)

        # Check that output was written
        output_text = "\n".join(mock_output_writer.written_lines)
        assert "Multi-tenant RAG System CLI" in output_text
        assert "Test Tenant" in output_text
        assert "QUERY RESULTS" in output_text or "Mock answer" in output_text

    @pytest.mark.asyncio
    async def test_execute_command_process_docs(self, mock_cli, mock_output_writer):
        """Test executing process-docs command."""
        args = CLIArgs(
            command="process-docs",
            tenant="test",
            user="testuser",
            language="hr",
            log_level="INFO",
            docs_path="/path/to/docs",
        )

        await mock_cli.execute_command(args)

        output_text = "\n".join(mock_output_writer.written_lines)
        assert "Processing documents" in output_text
        assert "/path/to/docs" in output_text

    @pytest.mark.asyncio
    async def test_execute_command_list_collections(self, mock_cli, mock_output_writer):
        """Test executing list-collections command."""
        args = CLIArgs(
            command="list-collections",
            tenant="test",
            user="testuser",
            language="hr",
            log_level="INFO",
        )

        await mock_cli.execute_command(args)

        output_text = "\n".join(mock_output_writer.written_lines)
        assert "Listing collections" in output_text
        assert "ChromaDB Collections" in output_text

    @pytest.mark.asyncio
    async def test_execute_command_status(self, mock_cli, mock_output_writer):
        """Test executing status command."""
        args = CLIArgs(
            command="status",
            tenant="test",
            user="testuser",
            language="hr",
            log_level="INFO",
        )

        await mock_cli.execute_command(args)

        output_text = "\n".join(mock_output_writer.written_lines)
        assert "System status" in output_text
        assert "RAG System:" in output_text

    @pytest.mark.asyncio
    async def test_execute_command_unknown(self, mock_cli, mock_output_writer):
        """Test executing unknown command."""
        args = CLIArgs(
            command="unknown-command",
            tenant="test",
            user="testuser",
            language="hr",
            log_level="INFO",
        )

        await mock_cli.execute_command(args)

        output_text = "\n".join(mock_output_writer.written_lines)
        assert "Unknown command" in output_text


class TestIntegrationScenarios:
    """Test realistic CLI integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_cli_workflow(self):
        """Test complete CLI workflow."""
        output_writer = MockOutputWriter()
        cli = create_mock_cli(should_fail=False, output_writer=output_writer)

        # Test status command
        status_args = CLIArgs(
            command="status",
            tenant="acme",
            user="john",
            language="en",
            log_level="INFO",
        )
        await cli.execute_command(status_args)

        # Test query command
        query_args = CLIArgs(
            command="query",
            tenant="acme",
            user="john",
            language="en",
            log_level="INFO",
            query_text="What is RAG technology?",
            top_k=5,
        )
        await cli.execute_command(query_args)

        # Verify both commands produced output
        assert len(output_writer.written_lines) > 10  # Should have substantial output
        output_text = "\n".join(output_writer.written_lines)
        assert "System status" in output_text
        assert "QUERY RESULTS" in output_text or "Mock answer" in output_text

    @pytest.mark.asyncio
    async def test_multilingual_cli_usage(self):
        """Test CLI with different languages."""
        output_writer = MockOutputWriter()
        cli = create_mock_cli(output_writer=output_writer)

        # Test Croatian query
        hr_args = CLIArgs(
            command="query",
            tenant="development",
            user="dev_user",
            language="hr",
            log_level="INFO",
            query_text="Što je RAG sustav?",
            top_k=3,
        )
        await cli.execute_command(hr_args)

        # Test English query
        en_args = CLIArgs(
            command="query",
            tenant="development",
            user="dev_user",
            language="en",
            log_level="INFO",
            query_text="What is a RAG system?",
            top_k=3,
        )
        await cli.execute_command(en_args)

        output_text = "\n".join(output_writer.written_lines)
        assert "Language: hr" in output_text
        assert "Language: en" in output_text

    @pytest.mark.asyncio
    async def test_error_handling_in_cli(self):
        """Test CLI error handling."""
        output_writer = MockOutputWriter()
        failing_cli = create_mock_cli(should_fail=True, output_writer=output_writer)

        # Try to execute query that will fail
        args = CLIArgs(
            command="query",
            tenant="test",
            user="testuser",
            language="hr",
            log_level="INFO",
            query_text="Test query",
        )

        await failing_cli.execute_command(args)

        output_text = "\n".join(output_writer.written_lines)
        assert "Query failed" in output_text or "Mock" in output_text

    def test_cli_argument_validation_edge_cases(self):
        """Test CLI with edge case arguments."""
        # Test minimum valid arguments
        args = parse_cli_arguments(["query", "test"])
        assert args.command == "query"
        assert args.query_text == "test"

        # Test with all optional parameters
        args_full = parse_cli_arguments(
            [
                "--tenant",
                "edge-case_tenant123",
                "--user",
                "user_123",
                "--language",
                "multilingual",
                "--log-level",
                "DEBUG",
                "process-docs",
                "/very/long/path/to/documents",
            ]
        )
        assert args_full.tenant == "edge-case_tenant123"
        assert args_full.user == "user_123"
        assert args_full.language == "multilingual"
        assert args_full.log_level == "DEBUG"

    def test_tenant_context_edge_cases(self):
        """Test tenant context creation with edge cases."""
        # Test with special characters that should be normalized
        context = create_tenant_context("ACME-Corp_2024", "admin_user")
        assert context.tenant_slug == "acme-corp_2024"
        assert context.user_username == "admin_user"

        # Test with minimum length user ID
        context_min = create_tenant_context("test", "ab")
        assert context_min.user_username == "ab"
        assert context_min.user_full_name == "User Ab"


class TestFactoryFunction:
    """Test factory function for creating CLI."""

    def test_create_mock_cli_success_mode(self):
        """Test creating CLI in success mode."""
        cli = create_mock_cli(should_fail=False)

        assert isinstance(cli, MultiTenantRAGCLIV2)
        assert hasattr(cli, "output_writer")
        assert hasattr(cli, "logger")
        assert hasattr(cli, "rag_system_factory")

    def test_create_mock_cli_failure_mode(self):
        """Test creating CLI in failure mode."""
        output_writer = MockOutputWriter()
        cli = create_mock_cli(should_fail=True, output_writer=output_writer)

        assert isinstance(cli, MultiTenantRAGCLIV2)
        assert cli.output_writer is output_writer

    @pytest.mark.asyncio
    async def test_factory_created_cli_functionality(self):
        """Test that factory-created CLI works correctly."""
        output_writer = MockOutputWriter()
        cli = create_mock_cli(output_writer=output_writer)

        # Test basic functionality
        context = create_tenant_context("test", "user")
        result = await cli.execute_query_command(context, "hr", "test query")

        assert isinstance(result, QueryResult)
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
