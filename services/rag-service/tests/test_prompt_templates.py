"""
Comprehensive tests for generation/prompt_templates.py

Tests all prompt template functionality including:
- Pure functions for query validation and context formatting
- Data structures (PromptTemplate, PromptConfig)
- Protocol implementations
- MultilingualRAGPrompts class
- PromptBuilder class
- Factory functions
"""

import unittest
from unittest.mock import MagicMock, patch

from src.generation.prompt_templates import (
    ConfigProvider,
    MultilingualRAGPrompts,
    PromptBuilder,
    PromptConfig,
    PromptTemplate,
    PromptTemplateProvider,
    build_complete_prompt,
    classify_query_type,
    create_multilingual_prompts,
    create_prompt_builder_for_query,
    format_context_with_headers,
    truncate_context_chunks,
    validate_query_for_prompt,
)
from tests.conftest import (
    create_mock_config_provider,
)


class TestPureFunctions(unittest.TestCase):
    """Test pure functions for query validation and context formatting."""

    def test_validate_query_for_prompt_valid(self):
        """Test validating valid queries."""
        self.assertEqual(validate_query_for_prompt("What is AI?"), "What is AI?")
        self.assertEqual(validate_query_for_prompt("  How   does   it work?  "), "How does it work?")
        self.assertEqual(validate_query_for_prompt("Simple query"), "Simple query")

    def test_validate_query_for_prompt_empty(self):
        """Test validating empty queries."""
        with self.assertRaises(ValueError) as cm:
            validate_query_for_prompt("")
        self.assertIn("Query cannot be empty", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            validate_query_for_prompt("   ")
        self.assertIn("Query cannot be only whitespace", str(cm.exception))

    def test_validate_query_for_prompt_invalid_type(self):
        """Test validating non-string queries."""
        with self.assertRaises(ValueError) as cm:
            validate_query_for_prompt(123)
        self.assertIn("Query must be string", str(cm.exception))

        # None is caught by empty check first
        with self.assertRaises(ValueError) as cm:
            validate_query_for_prompt(None)
        self.assertIn("Query cannot be empty", str(cm.exception))

    def test_validate_query_for_prompt_too_long(self):
        """Test validating overly long queries."""
        long_query = "a" * 10001
        with self.assertRaises(ValueError) as cm:
            validate_query_for_prompt(long_query)
        self.assertIn("Query too long", str(cm.exception))

    def test_truncate_context_chunks_valid(self):
        """Test truncating context chunks within limit."""
        chunks = ["Short chunk", "Another short chunk", "Third chunk"]
        result = truncate_context_chunks(chunks, 100, "\n---\n")
        self.assertEqual(result, chunks)

    def test_truncate_context_chunks_exceeds_limit(self):
        """Test truncating when chunks exceed limit."""
        chunks = ["This is a long chunk", "Another long chunk", "Yet another long chunk"]
        result = truncate_context_chunks(chunks, 30, "\n---\n")
        # Should fit first chunk, maybe second
        self.assertTrue(len(result) >= 1)
        self.assertTrue(all(isinstance(chunk, str) for chunk in result))

    def test_truncate_context_chunks_empty(self):
        """Test truncating empty chunks list."""
        result = truncate_context_chunks([], 100)
        self.assertEqual(result, [])

    def test_truncate_context_chunks_invalid_length(self):
        """Test truncating with invalid max length."""
        with self.assertRaises(ValueError) as cm:
            truncate_context_chunks(["chunk"], 0)
        self.assertIn("max_total_length must be positive", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            truncate_context_chunks(["chunk"], -10)
        self.assertIn("max_total_length must be positive", str(cm.exception))

    def test_truncate_context_chunks_invalid_chunks(self):
        """Test truncating with invalid chunk types."""
        with self.assertRaises(ValueError) as cm:
            truncate_context_chunks(["valid", 123, "also valid"], 100)
        self.assertIn("All chunks must be strings", str(cm.exception))

    def test_truncate_context_chunks_first_chunk_too_long(self):
        """Test truncating when first chunk is too long."""
        chunks = ["This is a very long first chunk that exceeds the limit"]
        result = truncate_context_chunks(chunks, 20, "\n---\n")
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].endswith("..."))
        self.assertLessEqual(len(result[0]), 20)

    def test_format_context_with_headers_valid(self):
        """Test formatting context with headers."""
        chunks = ["First content", "Second content"]
        result = format_context_with_headers(chunks, "Doc {index}:", "\n\n")
        expected = "Doc 1:\nFirst content\n\nDoc 2:\nSecond content"
        self.assertEqual(result, expected)

    def test_format_context_with_headers_empty(self):
        """Test formatting empty context."""
        result = format_context_with_headers([], "Doc {index}:")
        self.assertEqual(result, "")

    def test_format_context_with_headers_invalid_template(self):
        """Test formatting with invalid header template."""
        with self.assertRaises(ValueError) as cm:
            format_context_with_headers(["content"], "Invalid template")
        self.assertIn("header_template must contain {index} placeholder", str(cm.exception))

    def test_format_context_with_headers_template_error(self):
        """Test formatting with template that has invalid placeholders."""
        with self.assertRaises(ValueError) as cm:
            format_context_with_headers(["content"], "Doc {index} {invalid}:")
        self.assertIn("Invalid header_template", str(cm.exception))

    def test_format_context_with_headers_skip_empty(self):
        """Test formatting skips empty chunks but maintains index."""
        chunks = ["First content", "", "   ", "Second content"]
        result = format_context_with_headers(chunks, "Doc {index}:", "\n\n")
        # Index continues counting even for skipped chunks
        expected = "Doc 1:\nFirst content\n\nDoc 4:\nSecond content"
        self.assertEqual(result, expected)

    def test_classify_query_type_valid(self):
        """Test classifying valid query types."""
        patterns = {
            "tourism": ["hotel", "restaurant"],
            "cultural": ["history", "tradition"],
            "factual": ["when", "where"],
        }

        self.assertEqual(classify_query_type("Where is the hotel?", patterns), "tourism")
        self.assertEqual(classify_query_type("Tell me about history", patterns), "cultural")
        self.assertEqual(classify_query_type("When did it happen?", patterns), "factual")
        self.assertEqual(classify_query_type("Random question", patterns), "default")

    def test_classify_query_type_priority_order(self):
        """Test query classification respects priority order."""
        patterns = {
            "tourism": ["travel"],
            "cultural": ["travel"],  # Same keyword but lower priority
            "factual": ["travel"],   # Same keyword but even lower priority
        }

        # Should return tourism due to priority order
        self.assertEqual(classify_query_type("travel plans", patterns), "tourism")

    def test_classify_query_type_empty_query(self):
        """Test classifying empty query."""
        patterns = {"tourism": ["hotel"]}
        with self.assertRaises(ValueError) as cm:
            classify_query_type("", patterns)
        self.assertIn("Query cannot be empty", str(cm.exception))

    def test_classify_query_type_invalid_patterns(self):
        """Test classifying with invalid patterns."""
        with self.assertRaises(ValueError) as cm:
            classify_query_type("query", "not a dict")
        self.assertIn("keyword_patterns must be dict", str(cm.exception))

    def test_classify_query_type_case_insensitive(self):
        """Test query classification is case insensitive."""
        patterns = {"tourism": ["Hotel", "Restaurant"]}
        self.assertEqual(classify_query_type("where is the HOTEL?", patterns), "tourism")
        self.assertEqual(classify_query_type("RESTAURANT info", patterns), "tourism")

    def test_build_complete_prompt_valid(self):
        """Test building complete prompt with valid inputs."""
        system = "You are helpful"
        user_template = "Question: {query}"
        context_template = "Context: {context}"
        query = "What is AI?"
        context = "AI is artificial intelligence"

        sys_result, user_result = build_complete_prompt(system, user_template, context_template, query, context)

        self.assertEqual(sys_result, "You are helpful")
        self.assertEqual(user_result, "Context: AI is artificial intelligenceQuestion: What is AI?")

    def test_build_complete_prompt_no_context(self):
        """Test building prompt without context."""
        system = "You are helpful"
        user_template = "Question: {query}"
        context_template = "Context: {context}"
        query = "What is AI?"

        sys_result, user_result = build_complete_prompt(system, user_template, context_template, query, "")

        self.assertEqual(sys_result, "You are helpful")
        self.assertEqual(user_result, "Question: What is AI?")

    def test_build_complete_prompt_invalid_inputs(self):
        """Test building prompt with invalid inputs."""
        # Invalid system prompt type
        with self.assertRaises(ValueError) as cm:
            build_complete_prompt(123, "Question: {query}", "Context: {context}", "query")
        self.assertIn("system_prompt must be string", str(cm.exception))

        # Invalid user template type
        with self.assertRaises(ValueError) as cm:
            build_complete_prompt("system", 123, "Context: {context}", "query")
        self.assertIn("user_template must be string", str(cm.exception))

        # Missing query placeholder
        with self.assertRaises(ValueError) as cm:
            build_complete_prompt("system", "No placeholder", "Context: {context}", "query")
        self.assertIn("user_template must contain {query} placeholder", str(cm.exception))

        # Invalid context template when context provided
        with self.assertRaises(ValueError) as cm:
            build_complete_prompt("system", "Question: {query}", 123, "query", "context")
        self.assertIn("context_template must be string", str(cm.exception))

        # Missing context placeholder
        with self.assertRaises(ValueError) as cm:
            build_complete_prompt("system", "Question: {query}", "No placeholder", "query", "context")
        self.assertIn("context_template must contain {context} placeholder", str(cm.exception))


class TestDataStructures(unittest.TestCase):
    """Test data structure classes."""

    def test_prompt_template_valid(self):
        """Test creating valid PromptTemplate."""
        template = PromptTemplate(
            system_prompt="You are helpful",
            user_template="Question: {query}",
            context_template="Context: {context}"
        )

        self.assertEqual(template.system_prompt, "You are helpful")
        self.assertEqual(template.user_template, "Question: {query}")
        self.assertEqual(template.context_template, "Context: {context}")

    def test_prompt_template_default_context(self):
        """Test PromptTemplate with default context template."""
        template = PromptTemplate(
            system_prompt="You are helpful",
            user_template="Question: {query}"
        )

        self.assertEqual(template.context_template, "Context:\n{context}\n\n")

    def test_prompt_template_invalid(self):
        """Test creating invalid PromptTemplate."""
        # Empty system prompt
        with self.assertRaises(ValueError) as cm:
            PromptTemplate("", "Question: {query}")
        self.assertIn("system_prompt cannot be empty", str(cm.exception))

        # Empty user template
        with self.assertRaises(ValueError) as cm:
            PromptTemplate("System", "")
        self.assertIn("user_template cannot be empty", str(cm.exception))

        # Missing query placeholder
        with self.assertRaises(ValueError) as cm:
            PromptTemplate("System", "No placeholder")
        self.assertIn("user_template must contain {query} placeholder", str(cm.exception))

        # Invalid context template
        with self.assertRaises(ValueError) as cm:
            PromptTemplate("System", "Question: {query}", "No placeholder")
        self.assertIn("context_template must contain {context} placeholder", str(cm.exception))

    def test_prompt_config_valid(self):
        """Test creating valid PromptConfig."""
        templates = {
            "test": PromptTemplate("System", "Question: {query}")
        }
        patterns = {"factual": ["when", "where"]}
        formatting = {
            "header_template": "Doc {index}:",
            "chunk_separator": "\n\n",
            "context_separator": "\n---\n"
        }

        config = PromptConfig(templates, patterns, formatting)
        self.assertEqual(config.templates, templates)
        self.assertEqual(config.keyword_patterns, patterns)
        self.assertEqual(config.formatting, formatting)

    def test_prompt_config_invalid_types(self):
        """Test creating PromptConfig with invalid types."""
        valid_templates = {"test": PromptTemplate("System", "Question: {query}")}
        valid_patterns = {"factual": ["when"]}
        valid_formatting = {"header_template": "Doc {index}:", "chunk_separator": "\n\n", "context_separator": "\n---\n"}

        # Invalid templates type
        with self.assertRaises(ValueError) as cm:
            PromptConfig("not dict", valid_patterns, valid_formatting)
        self.assertIn("templates must be dict", str(cm.exception))

        # Invalid patterns type
        with self.assertRaises(ValueError) as cm:
            PromptConfig(valid_templates, "not dict", valid_formatting)
        self.assertIn("keyword_patterns must be dict", str(cm.exception))

        # Invalid formatting type
        with self.assertRaises(ValueError) as cm:
            PromptConfig(valid_templates, valid_patterns, "not dict")
        self.assertIn("formatting must be dict", str(cm.exception))

    def test_prompt_config_missing_formatting_keys(self):
        """Test PromptConfig with missing required formatting keys."""
        templates = {"test": PromptTemplate("System", "Question: {query}")}
        patterns = {"factual": ["when"]}

        # Missing header_template
        formatting = {"chunk_separator": "\n\n", "context_separator": "\n---\n"}
        with self.assertRaises(ValueError) as cm:
            PromptConfig(templates, patterns, formatting)
        self.assertIn("formatting must contain header_template", str(cm.exception))

        # Missing chunk_separator
        formatting = {"header_template": "Doc {index}:", "context_separator": "\n---\n"}
        with self.assertRaises(ValueError) as cm:
            PromptConfig(templates, patterns, formatting)
        self.assertIn("formatting must contain chunk_separator", str(cm.exception))

        # Missing context_separator
        formatting = {"header_template": "Doc {index}:", "chunk_separator": "\n\n"}
        with self.assertRaises(ValueError) as cm:
            PromptConfig(templates, patterns, formatting)
        self.assertIn("formatting must contain context_separator", str(cm.exception))


class TestMultilingualRAGPrompts(unittest.TestCase):
    """Test MultilingualRAGPrompts class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config_provider = create_mock_config_provider()
        self.prompts = MultilingualRAGPrompts(self.mock_config_provider, "hr")

    def test_initialization_valid(self):
        """Test valid initialization."""
        self.assertEqual(self.prompts.language, "hr")
        self.assertIsNotNone(self.prompts.templates)
        self.assertIn("question_answering", self.prompts.templates)

    def test_initialization_config_error(self):
        """Test initialization with config provider error."""
        mock_provider = MagicMock()
        mock_provider.get_prompt_config.side_effect = Exception("Config error")

        with self.assertRaises(Exception) as cm:
            MultilingualRAGPrompts(mock_provider, "hr")
        self.assertIn("Config error", str(cm.exception))

    def test_get_template_valid(self):
        """Test getting valid template."""
        template = self.prompts.get_template("question_answering")
        self.assertIsInstance(template, PromptTemplate)
        self.assertEqual(template.system_prompt, "You are a helpful assistant.")

    def test_get_template_invalid(self):
        """Test getting invalid template."""
        with self.assertRaises(KeyError) as cm:
            self.prompts.get_template("nonexistent")
        self.assertIn("Template 'nonexistent' not found", str(cm.exception))
        self.assertIn("Available:", str(cm.exception))

    def test_classify_query_valid(self):
        """Test classifying valid queries."""
        self.assertEqual(self.prompts.classify_query("Where is the hotel?"), "tourism")
        self.assertEqual(self.prompts.classify_query("Tell me about culture"), "cultural")
        self.assertEqual(self.prompts.classify_query("Random question"), "default")

    def test_classify_query_invalid(self):
        """Test classifying invalid query."""
        with self.assertRaises(ValueError):
            self.prompts.classify_query("")

    def test_get_template_for_query_known_types(self):
        """Test getting templates for known query types."""
        # Tourism query
        template = self.prompts.get_template_for_query("Where is the hotel?")
        self.assertIsInstance(template, PromptTemplate)

        # Cultural query
        template = self.prompts.get_template_for_query("Tell me about tradition")
        self.assertIsInstance(template, PromptTemplate)

        # Default query
        template = self.prompts.get_template_for_query("Random question")
        self.assertIsInstance(template, PromptTemplate)

    def test_get_template_for_query_fallback(self):
        """Test template selection fallback to default."""
        # Mock a scenario where specific template is missing
        with patch.object(self.prompts, 'get_template') as mock_get:
            mock_get.side_effect = [KeyError("Template not found"), PromptTemplate("System", "Question: {query}")]

            template = self.prompts.get_template_for_query("Where is the hotel?")
            self.assertIsInstance(template, PromptTemplate)

            # Should call get_template twice: once for tourism, once for question_answering fallback
            self.assertEqual(mock_get.call_count, 2)

    def test_get_template_for_query_unknown_type(self):
        """Test getting template for unknown query type."""
        # Patch classify_query to return unknown type
        with patch.object(self.prompts, 'classify_query', return_value="unknown_type"):
            with self.assertRaises(ValueError) as cm:
                self.prompts.get_template_for_query("test query")
            self.assertIn("Unknown query type 'unknown_type'", str(cm.exception))


class TestPromptBuilder(unittest.TestCase):
    """Test PromptBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.template = PromptTemplate(
            system_prompt="You are helpful",
            user_template="Question: {query}",
            context_template="Context:\n{context}\n\n"
        )

        self.config = PromptConfig(
            templates={"test": self.template},
            keyword_patterns={"factual": ["when"]},
            formatting={
                "header_template": "Document {index}:",
                "chunk_separator": "\n\n",
                "context_separator": "\n---\n"
            }
        )

        self.builder = PromptBuilder(self.template, self.config)

    def test_initialization(self):
        """Test PromptBuilder initialization."""
        self.assertEqual(self.builder.template, self.template)
        self.assertEqual(self.builder.config, self.config)

    def test_build_prompt_without_context(self):
        """Test building prompt without context."""
        system, user = self.builder.build_prompt("What is AI?")

        self.assertEqual(system, "You are helpful")
        self.assertEqual(user, "Question: What is AI?")

    def test_build_prompt_with_context(self):
        """Test building prompt with context."""
        context = ["AI is artificial intelligence", "It involves machine learning"]
        system, user = self.builder.build_prompt("What is AI?", context)

        self.assertEqual(system, "You are helpful")
        self.assertIn("Document 1:", user)
        self.assertIn("AI is artificial intelligence", user)
        self.assertIn("Question: What is AI?", user)

    def test_build_prompt_with_long_context(self):
        """Test building prompt with context that needs truncation."""
        long_context = ["Very long content " * 100 for _ in range(5)]
        system, user = self.builder.build_prompt("What is AI?", long_context, max_context_length=100)

        self.assertEqual(system, "You are helpful")
        # Should truncate context to fit length limit
        self.assertLess(len(user), 1000)  # Much shorter than full context

    def test_build_prompt_invalid_query(self):
        """Test building prompt with invalid query."""
        with self.assertRaises(ValueError):
            self.builder.build_prompt("")

    def test_format_context_empty(self):
        """Test formatting empty context."""
        result = self.builder._format_context([], 1000)
        self.assertEqual(result, "")

    def test_format_context_with_chunks(self):
        """Test formatting context with chunks."""
        context = ["First chunk", "Second chunk"]
        result = self.builder._format_context(context, 1000)

        self.assertIn("Document 1:", result)
        self.assertIn("First chunk", result)
        self.assertIn("Document 2:", result)
        self.assertIn("Second chunk", result)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_create_multilingual_prompts(self):
        """Test creating multilingual prompts."""
        config_provider = create_mock_config_provider()
        prompts = create_multilingual_prompts(config_provider, "en")

        self.assertIsInstance(prompts, MultilingualRAGPrompts)
        self.assertEqual(prompts.language, "en")

    def test_create_prompt_builder_for_query(self):
        """Test creating prompt builder for specific query."""
        config_provider = create_mock_config_provider()
        builder = create_prompt_builder_for_query(config_provider, "Where is the hotel?", "hr")

        self.assertIsInstance(builder, PromptBuilder)
        self.assertIsInstance(builder.template, PromptTemplate)

    def test_create_mock_config_provider_default(self):
        """Test creating mock config provider with defaults."""
        provider = create_mock_config_provider()
        config = provider.get_prompt_config("hr")

        self.assertIsInstance(config, PromptConfig)
        self.assertIn("question_answering", config.templates)
        self.assertIn("tourism", config.keyword_patterns)
        self.assertIn("header_template", config.formatting)

    def test_create_mock_config_provider_custom(self):
        """Test creating mock config provider with custom values."""
        custom_templates = {
            "custom": PromptTemplate("Custom system", "Custom: {query}")
        }
        custom_patterns = {"custom": ["custom"]}
        custom_formatting = {
            "header_template": "Custom {index}:",
            "chunk_separator": "|||",
            "context_separator": "---"
        }

        provider = create_mock_config_provider(
            templates=custom_templates,
            keyword_patterns=custom_patterns,
            formatting=custom_formatting
        )
        config = provider.get_prompt_config("test")

        self.assertEqual(config.templates, custom_templates)
        self.assertEqual(config.keyword_patterns, custom_patterns)
        self.assertEqual(config.formatting, custom_formatting)


class TestProtocols(unittest.TestCase):
    """Test protocol implementations."""

    def test_config_provider_protocol(self):
        """Test ConfigProvider protocol implementation."""
        provider = create_mock_config_provider()

        # Should implement the protocol
        self.assertTrue(hasattr(provider, 'get_prompt_config'))
        self.assertTrue(callable(getattr(provider, 'get_prompt_config')))

        # Should return proper type
        config = provider.get_prompt_config("hr")
        self.assertIsInstance(config, PromptConfig)

    def test_prompt_template_provider_protocol(self):
        """Test PromptTemplateProvider protocol with MultilingualRAGPrompts."""
        config_provider = create_mock_config_provider()
        prompts = MultilingualRAGPrompts(config_provider, "hr")

        # Should implement the protocol methods
        self.assertTrue(hasattr(prompts, 'get_template'))
        self.assertTrue(hasattr(prompts, 'templates'))  # Property, not method
        self.assertTrue(callable(getattr(prompts, 'get_template')))

        # Should return proper types
        template = prompts.get_template("question_answering")
        self.assertIsInstance(template, PromptTemplate)

        templates = prompts.templates  # Use property
        self.assertIsInstance(templates, dict)
        self.assertTrue(all(isinstance(t, PromptTemplate) for t in templates.values()))


class TestIntegration(unittest.TestCase):
    """Test integration between components."""

    def test_end_to_end_prompt_generation(self):
        """Test complete prompt generation workflow."""
        # Create components
        config_provider = create_mock_config_provider()
        prompts = create_multilingual_prompts(config_provider, "hr")

        # Get template for query
        query = "Where is the best hotel in Zagreb?"
        template = prompts.get_template_for_query(query)

        # Build prompt
        builder = PromptBuilder(template, prompts._config)
        context = ["Zagreb has many hotels", "The best hotels are in the city center"]
        system, user = builder.build_prompt(query, context)

        # Verify complete prompt
        self.assertIsInstance(system, str)
        self.assertIsInstance(user, str)
        self.assertIn("hotel", user.lower())
        self.assertIn("zagreb", user.lower())
        self.assertTrue(len(system) > 0)
        self.assertTrue(len(user) > 0)

    def test_multilingual_support(self):
        """Test multilingual configuration support."""
        config_provider = create_mock_config_provider()

        # Test different languages
        for language in ["hr", "en", "de"]:
            prompts = create_multilingual_prompts(config_provider, language)
            self.assertEqual(prompts.language, language)

            # Should work with same queries
            template = prompts.get_template_for_query("What is tourism?")
            self.assertIsInstance(template, PromptTemplate)

    def test_error_handling_chain(self):
        """Test error handling propagates correctly through the chain."""
        config_provider = create_mock_config_provider()
        prompts = create_multilingual_prompts(config_provider, "hr")
        template = prompts.get_template("question_answering")
        builder = PromptBuilder(template, prompts._config)

        # Invalid query should propagate through chain
        with self.assertRaises(ValueError):
            builder.build_prompt("")  # Empty query should fail validation

    def test_dependency_injection_isolation(self):
        """Test that different instances are properly isolated."""
        provider1 = create_mock_config_provider()
        provider2 = create_mock_config_provider()

        prompts1 = create_multilingual_prompts(provider1, "hr")
        prompts2 = create_multilingual_prompts(provider2, "en")

        # Should be independent instances
        self.assertNotEqual(prompts1, prompts2)
        self.assertEqual(prompts1.language, "hr")
        self.assertEqual(prompts2.language, "en")


if __name__ == "__main__":
    unittest.main()