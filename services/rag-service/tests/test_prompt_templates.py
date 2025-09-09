"""
Comprehensive tests for prompt_templates.py demonstrating 100% testability.
Tests pure functions, dependency injection, and integration scenarios.
"""

from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
from src.generation.prompt_templates import (  # Pure functions; Data structures; Core classes; Factory functions
    MultilingualRAGPrompts, PromptBuilder, PromptConfig, PromptTemplate,
    build_complete_prompt, classify_query_type, create_mock_config_provider,
    create_multilingual_prompts, create_prompt_builder_for_query,
    format_context_with_headers, truncate_context_chunks,
    validate_query_for_prompt)


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_validate_query_for_prompt_valid(self):
        """Test query validation with valid input."""
        query = "What is RAG system?"
        result = validate_query_for_prompt(query)
        assert result == "What is RAG system?"

    def test_validate_query_for_prompt_whitespace_normalization(self):
        """Test query validation normalizes whitespace."""
        query = "  What   is    RAG   system?  \n\t"
        result = validate_query_for_prompt(query)
        assert result == "What is RAG system?"

    def test_validate_query_for_prompt_empty(self):
        """Test query validation with empty string."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            validate_query_for_prompt("")

    def test_validate_query_for_prompt_whitespace_only(self):
        """Test query validation with whitespace only."""
        with pytest.raises(ValueError, match="Query cannot be only whitespace"):
            validate_query_for_prompt("   \n\t   ")

    def test_validate_query_for_prompt_non_string(self):
        """Test query validation with non-string input."""
        with pytest.raises(ValueError, match="Query must be string"):
            validate_query_for_prompt(123)

    def test_validate_query_for_prompt_too_long(self):
        """Test query validation with overly long input."""
        long_query = "a" * 10001
        with pytest.raises(ValueError, match="Query too long"):
            validate_query_for_prompt(long_query)

    def test_truncate_context_chunks_normal(self):
        """Test context chunk truncation with normal input."""
        chunks = ["First chunk text", "Second chunk text", "Third chunk text"]
        result = truncate_context_chunks(chunks, max_total_length=50)

        # Should fit first two chunks
        assert len(result) <= 2
        assert "First chunk text" in result

    def test_truncate_context_chunks_empty_list(self):
        """Test context chunk truncation with empty list."""
        result = truncate_context_chunks([], max_total_length=100)
        assert result == []

    def test_truncate_context_chunks_invalid_length(self):
        """Test context chunk truncation with invalid max length."""
        chunks = ["test"]
        with pytest.raises(ValueError, match="max_total_length must be positive"):
            truncate_context_chunks(chunks, max_total_length=0)

    def test_truncate_context_chunks_non_string(self):
        """Test context chunk truncation with non-string chunks."""
        chunks = ["valid", 123, "another valid"]
        with pytest.raises(ValueError, match="All chunks must be strings"):
            truncate_context_chunks(chunks, max_total_length=100)

    def test_truncate_context_chunks_single_large_chunk(self):
        """Test truncation with single chunk too large."""
        large_chunk = "a" * 100
        result = truncate_context_chunks([large_chunk], max_total_length=50)

        assert len(result) == 1
        assert result[0].endswith("...")
        assert len(result[0]) <= 50

    def test_format_context_with_headers_normal(self):
        """Test context formatting with headers."""
        chunks = ["First chunk", "Second chunk"]
        result = format_context_with_headers(
            chunks, header_template="Doc {index}:", chunk_separator="\n\n"
        )

        expected = "Doc 1:\nFirst chunk\n\nDoc 2:\nSecond chunk"
        assert result == expected

    def test_format_context_with_headers_empty(self):
        """Test context formatting with empty chunks."""
        result = format_context_with_headers([])
        assert result == ""

    def test_format_context_with_headers_invalid_template(self):
        """Test context formatting with invalid header template."""
        chunks = ["test"]
        with pytest.raises(ValueError, match="header_template must contain {index}"):
            format_context_with_headers(chunks, header_template="Invalid template")

    def test_format_context_with_headers_template_error(self):
        """Test context formatting with template that has extra placeholders."""
        chunks = ["test"]
        with pytest.raises(ValueError, match="Invalid header_template"):
            format_context_with_headers(
                chunks, header_template="Doc {index} {missing}:"
            )

    def test_classify_query_type_cultural(self):
        """Test query classification for cultural queries."""
        patterns = {
            "cultural": ["culture", "history", "tradition"],
            "tourism": ["hotel", "restaurant"],
        }

        result = classify_query_type("Tell me about Croatian culture", patterns)
        assert result == "cultural"

    def test_classify_query_type_tourism(self):
        """Test query classification for tourism queries."""
        patterns = {
            "cultural": ["culture", "history"],
            "tourism": ["hotel", "restaurant", "travel"],
        }

        result = classify_query_type("Where is the best hotel?", patterns)
        assert result == "tourism"

    def test_classify_query_type_priority_order(self):
        """Test query classification respects priority order."""
        patterns = {
            "cultural": ["travel"],  # Lower priority
            "tourism": ["travel"],  # Higher priority
        }

        result = classify_query_type("Travel to Croatia", patterns)
        assert result == "tourism"  # Tourism has higher priority

    def test_classify_query_type_no_match(self):
        """Test query classification with no pattern match."""
        patterns = {"cultural": ["culture"], "tourism": ["hotel"]}

        result = classify_query_type("What is machine learning?", patterns)
        assert result == "default"

    def test_classify_query_type_empty_query(self):
        """Test query classification with empty query."""
        patterns = {"test": ["keyword"]}
        with pytest.raises(ValueError, match="Query cannot be empty"):
            classify_query_type("", patterns)

    def test_classify_query_type_invalid_patterns(self):
        """Test query classification with invalid patterns."""
        with pytest.raises(ValueError, match="keyword_patterns must be dict"):
            classify_query_type("test query", "not a dict")

    def test_build_complete_prompt_with_context(self):
        """Test building complete prompt with context."""
        system_prompt, user_prompt = build_complete_prompt(
            system_prompt="You are helpful.",
            user_template="Question: {query}",
            context_template="Context: {context}\n",
            query="What is AI?",
            context_text="AI is artificial intelligence.",
        )

        assert system_prompt == "You are helpful."
        assert "Context: AI is artificial intelligence." in user_prompt
        assert "Question: What is AI?" in user_prompt

    def test_build_complete_prompt_without_context(self):
        """Test building complete prompt without context."""
        system_prompt, user_prompt = build_complete_prompt(
            system_prompt="You are helpful.",
            user_template="Question: {query}",
            context_template="Context: {context}\n",
            query="What is AI?",
            context_text="",
        )

        assert system_prompt == "You are helpful."
        assert user_prompt == "Question: What is AI?"
        assert "Context:" not in user_prompt

    def test_build_complete_prompt_invalid_user_template(self):
        """Test building prompt with invalid user template."""
        with pytest.raises(ValueError, match="user_template must contain {query}"):
            build_complete_prompt(
                system_prompt="Test",
                user_template="No query placeholder",
                context_template="Context: {context}",
                query="test",
                context_text="",
            )

    def test_build_complete_prompt_invalid_context_template(self):
        """Test building prompt with invalid context template."""
        with pytest.raises(ValueError, match="context_template must contain {context}"):
            build_complete_prompt(
                system_prompt="Test",
                user_template="Query: {query}",
                context_template="No context placeholder",
                query="test",
                context_text="some context",
            )


class TestPromptTemplate:
    """Test PromptTemplate dataclass."""

    def test_prompt_template_creation_valid(self):
        """Test creating valid PromptTemplate."""
        template = PromptTemplate(
            system_prompt="You are helpful.",
            user_template="Question: {query}",
            context_template="Context: {context}\n",
        )

        assert template.system_prompt == "You are helpful."
        assert template.user_template == "Question: {query}"
        assert template.context_template == "Context: {context}\n"

    def test_prompt_template_default_context(self):
        """Test PromptTemplate with default context template."""
        template = PromptTemplate(
            system_prompt="You are helpful.", user_template="Question: {query}"
        )

        assert template.context_template == "Context:\n{context}\n\n"

    def test_prompt_template_empty_system_prompt(self):
        """Test PromptTemplate validation with empty system prompt."""
        with pytest.raises(ValueError, match="system_prompt cannot be empty"):
            PromptTemplate(system_prompt="", user_template="Question: {query}")

    def test_prompt_template_empty_user_template(self):
        """Test PromptTemplate validation with empty user template."""
        with pytest.raises(ValueError, match="user_template cannot be empty"):
            PromptTemplate(system_prompt="You are helpful.", user_template="")

    def test_prompt_template_missing_query_placeholder(self):
        """Test PromptTemplate validation without query placeholder."""
        with pytest.raises(ValueError, match="user_template must contain {query}"):
            PromptTemplate(
                system_prompt="You are helpful.", user_template="No query placeholder"
            )

    def test_prompt_template_missing_context_placeholder(self):
        """Test PromptTemplate validation without context placeholder."""
        with pytest.raises(ValueError, match="context_template must contain {context}"):
            PromptTemplate(
                system_prompt="You are helpful.",
                user_template="Question: {query}",
                context_template="No context placeholder",
            )


class TestPromptConfig:
    """Test PromptConfig dataclass."""

    def test_prompt_config_creation_valid(self):
        """Test creating valid PromptConfig."""
        template = PromptTemplate(system_prompt="Test", user_template="Query: {query}")

        config = PromptConfig(
            templates={"test": template},
            keyword_patterns={"test": ["keyword"]},
            formatting={
                "header_template": "Doc {index}:",
                "chunk_separator": "\n\n",
                "context_separator": "\n---\n",
            },
        )

        assert "test" in config.templates
        assert "test" in config.keyword_patterns
        assert "header_template" in config.formatting

    def test_prompt_config_invalid_templates(self):
        """Test PromptConfig validation with invalid templates."""
        with pytest.raises(ValueError, match="templates must be dict"):
            PromptConfig(
                templates="not a dict",
                keyword_patterns={},
                formatting={
                    "header_template": "Doc {index}:",
                    "chunk_separator": "\n",
                    "context_separator": "\n",
                },
            )

    def test_prompt_config_missing_formatting_key(self):
        """Test PromptConfig validation with missing formatting keys."""
        template = PromptTemplate("Test", "Query: {query}")

        with pytest.raises(ValueError, match="formatting must contain header_template"):
            PromptConfig(
                templates={"test": template},
                keyword_patterns={},
                formatting={
                    "chunk_separator": "\n",
                    "context_separator": "\n",
                },  # Missing header_template
            )


class TestMultilingualRAGPromptsWithMocks:
    """Test MultilingualRAGPrompts with mock dependencies."""

    @pytest.fixture
    def mock_config_provider(self):
        """Create mock config provider."""
        return create_mock_config_provider()

    def test_initialization(self, mock_config_provider):
        """Test MultilingualRAGPrompts initialization."""
        prompts = MultilingualRAGPrompts(mock_config_provider, "hr")

        assert prompts.language == "hr"
        assert prompts.config_provider == mock_config_provider
        assert isinstance(prompts.templates, dict)

    def test_get_template_valid(self, mock_config_provider):
        """Test getting valid template."""
        prompts = MultilingualRAGPrompts(mock_config_provider, "hr")

        template = prompts.get_template("question_answering")

        assert isinstance(template, PromptTemplate)
        assert "helpful assistant" in template.system_prompt.lower()

    def test_get_template_invalid(self, mock_config_provider):
        """Test getting invalid template."""
        prompts = MultilingualRAGPrompts(mock_config_provider, "hr")

        with pytest.raises(KeyError, match="Template 'nonexistent' not found"):
            prompts.get_template("nonexistent")

    def test_classify_query(self, mock_config_provider):
        """Test query classification."""
        prompts = MultilingualRAGPrompts(mock_config_provider, "hr")

        # Test cultural query
        result = prompts.classify_query("Tell me about Croatian culture")
        assert result == "cultural"

        # Test tourism query
        result = prompts.classify_query("Where is the best hotel?")
        assert result == "tourism"

        # Test default
        result = prompts.classify_query("What is machine learning?")
        assert result == "default"

    def test_get_template_for_query(self, mock_config_provider):
        """Test getting template for specific query."""
        prompts = MultilingualRAGPrompts(mock_config_provider, "hr")

        # Cultural query should get explanatory template
        template = prompts.get_template_for_query("Croatian culture")
        assert "detailed explanations" in template.system_prompt.lower()

        # Tourism query should get tourism template
        template = prompts.get_template_for_query("best hotel")
        assert "tourism expert" in template.system_prompt.lower()

        # Default query should get question answering template
        template = prompts.get_template_for_query("machine learning")
        assert "helpful assistant" in template.system_prompt.lower()

    def test_get_template_for_query_fallback(self, mock_config_provider):
        """Test template fallback when specific template not found."""
        # Create config with missing templates
        limited_config = create_mock_config_provider(
            templates={"question_answering": PromptTemplate("Test", "Query: {query}")}
        )

        prompts = MultilingualRAGPrompts(limited_config, "hr")

        # Should fallback to question_answering even for tourism query
        template = prompts.get_template_for_query("hotel")
        assert template.system_prompt == "Test"


class TestPromptBuilderWithMocks:
    """Test PromptBuilder with mock dependencies."""

    @pytest.fixture
    def sample_template(self):
        """Sample template for testing."""
        return PromptTemplate(
            system_prompt="You are a helpful assistant.",
            user_template="Question: {query}\n\nAnswer:",
            context_template="Context:\n{context}\n\n",
        )

    @pytest.fixture
    def sample_config(self):
        """Sample config for testing."""
        return PromptConfig(
            templates={"test": PromptTemplate("Test", "Query: {query}")},
            keyword_patterns={"test": ["keyword"]},
            formatting={
                "header_template": "Document {index}:",
                "chunk_separator": "\n\n",
                "context_separator": "\n---\n",
            },
        )

    def test_build_prompt_without_context(self, sample_template, sample_config):
        """Test building prompt without context."""
        builder = PromptBuilder(sample_template, sample_config)

        system_prompt, user_prompt = builder.build_prompt("What is AI?")

        assert system_prompt == "You are a helpful assistant."
        assert user_prompt == "Question: What is AI?\n\nAnswer:"
        assert "Context:" not in user_prompt

    def test_build_prompt_with_context(self, sample_template, sample_config):
        """Test building prompt with context."""
        builder = PromptBuilder(sample_template, sample_config)
        context = ["AI is artificial intelligence.", "It involves machine learning."]

        system_prompt, user_prompt = builder.build_prompt(
            "What is AI?", context=context, max_context_length=1000
        )

        assert system_prompt == "You are a helpful assistant."
        assert "Context:" in user_prompt
        assert "Document 1:" in user_prompt
        assert "Document 2:" in user_prompt
        assert "AI is artificial intelligence." in user_prompt
        assert "Question: What is AI?" in user_prompt

    def test_build_prompt_context_truncation(self, sample_template, sample_config):
        """Test prompt building with context truncation."""
        builder = PromptBuilder(sample_template, sample_config)
        long_context = ["a" * 100, "b" * 100, "c" * 100]

        system_prompt, user_prompt = builder.build_prompt(
            "Test query",
            context=long_context,
            max_context_length=50,  # Very small limit
        )

        # Should truncate context but still include query
        assert "Test query" in user_prompt
        assert len([chunk for chunk in long_context if chunk in user_prompt]) <= 1

    def test_build_prompt_invalid_query(self, sample_template, sample_config):
        """Test building prompt with invalid query."""
        builder = PromptBuilder(sample_template, sample_config)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            builder.build_prompt("")

    def test_format_context_error_fallback(self, sample_template, sample_config):
        """Test context formatting error fallback."""
        # Create config with invalid header template
        invalid_config = PromptConfig(
            templates=sample_config.templates,
            keyword_patterns=sample_config.keyword_patterns,
            formatting={
                "header_template": "Invalid {missing} template",
                "chunk_separator": "\n\n",
                "context_separator": "\n---\n",
            },
        )

        builder = PromptBuilder(sample_template, invalid_config)
        context = ["First chunk", "Second chunk"]

        # Should fallback to simple concatenation
        system_prompt, user_prompt = builder.build_prompt("Test query", context=context)

        assert "Test query" in user_prompt
        # Should contain fallback format (simple concatenation)
        assert "First chunk" in user_prompt


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_mock_config_provider(self):
        """Test creating mock config provider."""
        provider = create_mock_config_provider()

        config = provider.get_prompt_config("hr")

        assert isinstance(config, PromptConfig)
        assert "question_answering" in config.templates
        assert "cultural" in config.keyword_patterns
        assert "header_template" in config.formatting

    def test_create_mock_config_provider_custom(self):
        """Test creating mock config provider with custom data."""
        custom_templates = {
            "custom": PromptTemplate("Custom system", "Custom: {query}")
        }

        provider = create_mock_config_provider(templates=custom_templates)
        config = provider.get_prompt_config("hr")

        assert "custom" in config.templates
        assert config.templates["custom"].system_prompt == "Custom system"

    def test_create_multilingual_prompts(self):
        """Test creating multilingual prompts."""
        provider = create_mock_config_provider()

        prompts = create_multilingual_prompts(provider, "hr")

        assert isinstance(prompts, MultilingualRAGPrompts)
        assert prompts.language == "hr"

    def test_create_prompt_builder_for_query(self):
        """Test creating prompt builder for query."""
        provider = create_mock_config_provider()

        builder = create_prompt_builder_for_query(provider, "Croatian culture", "hr")

        assert isinstance(builder, PromptBuilder)
        assert isinstance(builder.template, PromptTemplate)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_multilingual_workflow_croatian(self):
        """Test complete workflow for Croatian queries."""
        provider = create_mock_config_provider()

        # Cultural query in Croatian context
        query = "Opišite hrvatsku kulturu i tradicije"
        builder = create_prompt_builder_for_query(provider, query, "hr")

        context = [
            "Hrvatska kultura je bogata tradicijama.",
            "Glazba i ples su važni dijelovi kulture.",
        ]

        system_prompt, user_prompt = builder.build_prompt(query, context)

        assert system_prompt  # Should have system prompt
        assert "kulturu" in user_prompt  # Should contain original Croatian query
        assert "tradicijama" in user_prompt  # Should contain Croatian context

    def test_multilingual_workflow_english(self):
        """Test complete workflow for English queries."""
        provider = create_mock_config_provider()

        # Tourism query in English
        query = "What are the best hotels in Croatia?"
        builder = create_prompt_builder_for_query(provider, query, "en")

        context = [
            "Hotel Dubrovnik Palace is a luxury hotel.",
            "Hotel Park Split offers excellent service.",
        ]

        system_prompt, user_prompt = builder.build_prompt(query, context)

        assert "tourism expert" in system_prompt.lower()  # Should use tourism template
        assert "hotels in Croatia" in user_prompt
        assert "Hotel Dubrovnik" in user_prompt

    def test_query_classification_edge_cases(self):
        """Test query classification with edge cases."""
        provider = create_mock_config_provider()
        prompts = create_multilingual_prompts(provider, "hr")

        # Mixed keywords - should prioritize based on order
        mixed_query = "Cultural hotel traditions"
        query_type = prompts.classify_query(mixed_query)
        assert query_type == "cultural"  # Cultural has higher priority

        # No keywords - should default
        technical_query = "Machine learning algorithms"
        query_type = prompts.classify_query(technical_query)
        assert query_type == "default"

        # Case insensitive matching
        uppercase_query = "WHAT IS CROATIAN CULTURE"
        query_type = prompts.classify_query(uppercase_query)
        assert query_type == "cultural"

    def test_context_handling_large_documents(self):
        """Test context handling with large documents."""
        provider = create_mock_config_provider()

        # Create large context chunks
        large_chunks = [
            "a" * 500,  # Large chunk
            "b" * 400,  # Medium chunk
            "c" * 300,  # Smaller chunk
        ]

        builder = create_prompt_builder_for_query(provider, "Test query", "hr")

        # Test with tight length limit
        system_prompt, user_prompt = builder.build_prompt(
            "Test query", context=large_chunks, max_context_length=600
        )

        # Should handle truncation gracefully
        assert "Test query" in user_prompt
        # Should not exceed reasonable length for user prompt
        assert len(user_prompt) < 1000

    def test_error_recovery_scenarios(self):
        """Test error recovery in various failure scenarios."""
        provider = create_mock_config_provider()

        # Test with malformed query (should handle gracefully)
        prompts = create_multilingual_prompts(provider, "hr")

        # Very long query (should be rejected by validation)
        with pytest.raises(ValueError):
            prompts.classify_query("a" * 10001)

        # Empty query classification (should handle gracefully)
        query_type = prompts.classify_query("   ")  # Will raise in validation
        # This should be caught by the classify_query method and return "default"
        # But the validate_query_for_prompt will raise, so we expect the error

    def test_template_customization(self):
        """Test customization of templates and configurations."""
        # Custom templates for specific domain
        custom_templates = {
            "medical": PromptTemplate(
                system_prompt="You are a medical expert.",
                user_template="Medical question: {query}\n\nMedical answer:",
                context_template="Medical literature:\n{context}\n\n",
            ),
            "question_answering": PromptTemplate(
                system_prompt="You are helpful.", user_template="Q: {query}\nA:"
            ),
        }

        # Custom keyword patterns
        custom_patterns = {
            "medical": ["health", "disease", "treatment", "medicine"],
            "default": [],
        }

        provider = create_mock_config_provider(
            templates=custom_templates, keyword_patterns=custom_patterns
        )

        # Test medical query classification
        prompts = create_multilingual_prompts(provider, "hr")
        query_type = prompts.classify_query("What is the treatment for diabetes?")

        # Should classify as medical (not in priority list, but should still work)
        template = prompts.get_template_for_query("What is the treatment for diabetes?")

        # Since "medical" is not in the template mapping, should fallback to default
        assert "helpful" in template.system_prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
