"""
Comprehensive test suite for enhanced_prompt_templates_v2.py dependency injection implementation.
Tests pure functions, DI orchestration, mock providers, and integration scenarios.
"""

from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
from src.generation.enhanced_prompt_templates import (  # Data classes; Pure functions; DI classes; Legacy compatibility
    BuildPromptResult, ContextFormattingOptions, EnhancedPromptBuilder,
    LegacyEnhancedPromptBuilder, PromptConfig, PromptTemplate, PromptType,
    TemplateStats, ValidationResult, _EnhancedPromptBuilder,
    build_category_prompt, build_followup_prompt, calculate_template_stats,
    create_enhanced_prompt_builder, find_template_by_content,
    format_context_chunks, get_missing_templates, parse_config_templates,
    suggest_template_improvements, validate_templates)
from src.generation.enhanced_prompt_templates_providers import (
    MockConfigProvider, MockLoggerProvider, create_invalid_config,
    create_minimal_config, create_mock_setup, create_test_config,
    create_test_prompt_builder)
from src.retrieval.categorization import DocumentCategory

# ================================
# PURE FUNCTION TESTS
# ================================


class TestPureFunctions:
    """Test all pure business logic functions."""

    def test_parse_config_templates(self):
        """Test parsing configuration templates into PromptTemplate objects."""
        config_data = {
            "general": {
                "system": "You are helpful.",
                "user": "Q: {query}\nA:",
                "invalid_type": "This should be skipped",
            },
            "technical": {
                "system": "You are technical expert.",
                "user": "Tech Q: {query}\nTech A:",
            },
            "invalid_category": {"system": "This should be skipped"},
        }

        templates = parse_config_templates(config_data, "hr")

        # Check valid categories were parsed
        assert DocumentCategory.GENERAL in templates
        assert DocumentCategory.TECHNICAL in templates

        # Check valid prompt types were parsed
        general_templates = templates[DocumentCategory.GENERAL]
        assert PromptType.SYSTEM in general_templates
        assert PromptType.USER in general_templates
        assert len(general_templates) == 2  # invalid_type should be skipped

        # Check template properties
        system_template = general_templates[PromptType.SYSTEM]
        assert system_template.template == "You are helpful."
        assert system_template.category == DocumentCategory.GENERAL
        assert system_template.language == "hr"

    def test_format_context_chunks(self):
        """Test context chunk formatting with various options."""
        chunks = ["First chunk content", "Second chunk content", "Third chunk content"]

        # Test basic formatting with attribution
        options = ContextFormattingOptions(
            max_length=1000,
            include_attribution=True,
            source_label="Source",
            truncation_indicator="...",
        )

        formatted, truncated = format_context_chunks(chunks, options)

        assert not truncated
        assert "[Source 1]:" in formatted
        assert "[Source 2]:" in formatted
        assert "[Source 3]:" in formatted
        assert "First chunk content" in formatted

        # Test without attribution
        options.include_attribution = False
        formatted, truncated = format_context_chunks(chunks, options)

        assert not truncated
        assert "[Source" not in formatted
        assert "First chunk content" in formatted

        # Test truncation
        options.max_length = 50  # Very small limit
        formatted, truncated = format_context_chunks(chunks, options)

        assert truncated
        assert "..." in formatted

        # Test empty chunks
        formatted, truncated = format_context_chunks([], options, "No context")
        assert formatted == "No context"
        assert not truncated

    def test_build_category_prompt(self):
        """Test building category-specific prompts."""
        # Create test templates
        templates = {
            DocumentCategory.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="You are helpful.",
                    category=DocumentCategory.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="hr",
                ),
                PromptType.USER: PromptTemplate(
                    template="Q: {query}\nContext: {context}\nA:",
                    category=DocumentCategory.GENERAL,
                    prompt_type=PromptType.USER,
                    language="hr",
                ),
            }
        }

        options = ContextFormattingOptions(max_length=1000, include_attribution=True)

        result = build_category_prompt(
            query="What is this?",
            context_chunks=["Some relevant information"],
            category=DocumentCategory.GENERAL,
            templates=templates,
            formatting_options=options,
            no_context_message="No context",
        )

        assert isinstance(result, BuildPromptResult)
        assert result.system_prompt == "You are helpful."
        assert "What is this?" in result.user_prompt
        assert "Some relevant information" in result.user_prompt
        assert result.chunks_included == 1
        assert not result.truncated

        # Test missing category
        with pytest.raises(KeyError, match="No templates found for category"):
            build_category_prompt(
                query="test",
                context_chunks=[],
                category=DocumentCategory.TECHNICAL,  # Not in templates
                templates=templates,
                formatting_options=options,
            )

    def test_build_followup_prompt(self):
        """Test building follow-up prompts."""
        templates = {
            DocumentCategory.GENERAL: {
                PromptType.FOLLOWUP: PromptTemplate(
                    template="Previous: {original_query} -> {original_answer}\nNew: {followup_query}",
                    category=DocumentCategory.GENERAL,
                    prompt_type=PromptType.FOLLOWUP,
                    language="hr",
                )
            }
        }

        result = build_followup_prompt(
            original_query="What is AI?",
            original_answer="AI is artificial intelligence.",
            followup_query="How does it work?",
            category=DocumentCategory.GENERAL,
            templates=templates,
        )

        assert "What is AI?" in result
        assert "AI is artificial intelligence." in result
        assert "How does it work?" in result

        # Test missing followup template
        templates[DocumentCategory.GENERAL] = {}  # Remove followup template

        with pytest.raises(KeyError, match="No followup template found"):
            build_followup_prompt("q", "a", "f", DocumentCategory.GENERAL, templates)

    def test_calculate_template_stats(self):
        """Test template statistics calculation."""
        templates = {
            DocumentCategory.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    "Short", DocumentCategory.GENERAL, PromptType.SYSTEM, "hr"
                ),
                PromptType.USER: PromptTemplate(
                    "Medium length template",
                    DocumentCategory.GENERAL,
                    PromptType.USER,
                    "hr",
                ),
            },
            DocumentCategory.TECHNICAL: {
                PromptType.SYSTEM: PromptTemplate(
                    "Technical system prompt",
                    DocumentCategory.TECHNICAL,
                    PromptType.SYSTEM,
                    "hr",
                )
            },
        }

        stats = calculate_template_stats(templates, "hr")

        assert stats.total_categories == 2
        assert stats.language == "hr"
        assert "general" in stats.categories
        assert "technical" in stats.categories

        general_stats = stats.categories["general"]
        assert general_stats["template_count"] == 2
        assert "system" in general_stats["template_types"]
        assert "user" in general_stats["template_types"]
        assert general_stats["avg_template_length"] > 0

    def test_validate_templates(self):
        """Test template validation."""
        # Valid templates
        valid_templates = {
            DocumentCategory.GENERAL: {
                PromptType.USER: PromptTemplate(
                    "Q: {query}\nContext: {context}\nA:",
                    DocumentCategory.GENERAL,
                    PromptType.USER,
                    "hr",
                ),
                PromptType.FOLLOWUP: PromptTemplate(
                    "Previous: {original_query} -> {original_answer}\nNew: {followup_query}",
                    DocumentCategory.GENERAL,
                    PromptType.FOLLOWUP,
                    "hr",
                ),
            }
        }

        result = validate_templates(valid_templates)
        assert result.valid
        assert len(result.issues) == 0

        # Invalid templates (missing placeholders)
        invalid_templates = {
            DocumentCategory.GENERAL: {
                PromptType.USER: PromptTemplate(
                    "Missing placeholders",  # Missing {query} and {context}
                    DocumentCategory.GENERAL,
                    PromptType.USER,
                    "hr",
                ),
                PromptType.FOLLOWUP: PromptTemplate(
                    "Missing: {original_query}",  # Missing {original_answer} and {followup_query}
                    DocumentCategory.GENERAL,
                    PromptType.FOLLOWUP,
                    "hr",
                ),
            }
        }

        result = validate_templates(invalid_templates)
        assert not result.valid
        assert (
            len(result.issues) >= 3
        )  # Missing query, context, original_answer, followup_query

        # Check specific issues
        issue_text = " ".join(result.issues)
        assert "Missing {query} placeholder" in issue_text
        assert "Missing {context} placeholder" in issue_text

    def test_suggest_template_improvements(self):
        """Test template improvement suggestions."""
        templates = {
            DocumentCategory.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    "System", DocumentCategory.GENERAL, PromptType.SYSTEM, "hr"
                )
                # Missing USER and FOLLOWUP templates
            }
        }

        # Test missing category
        suggestions = suggest_template_improvements(
            templates, DocumentCategory.TECHNICAL, {}
        )
        assert any("Add specialized templates for technical" in s for s in suggestions)

        # Test missing template types
        suggestions = suggest_template_improvements(
            templates, DocumentCategory.GENERAL, {}
        )
        assert any("Add missing template types" in s for s in suggestions)

        # Test performance-based suggestions
        usage_stats = {"avg_confidence": 0.5, "avg_response_length": 50}
        suggestions = suggest_template_improvements(
            templates, DocumentCategory.GENERAL, usage_stats
        )
        assert any("improve response quality" in s for s in suggestions)
        assert any("too brief responses" in s for s in suggestions)

    def test_get_missing_templates(self):
        """Test missing template identification."""
        templates = {
            DocumentCategory.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    "System", DocumentCategory.GENERAL, PromptType.SYSTEM, "hr"
                )
                # Missing USER
            }
            # Missing TECHNICAL category
        }

        required_categories = [DocumentCategory.GENERAL, DocumentCategory.TECHNICAL]
        required_types = [PromptType.SYSTEM, PromptType.USER]

        missing = get_missing_templates(templates, required_categories, required_types)

        assert "technical" in missing["categories"]
        assert "general/user" in missing["templates"]

    def test_find_template_by_content(self):
        """Test finding templates by content."""
        templates = {
            DocumentCategory.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    "You are a helpful assistant",
                    DocumentCategory.GENERAL,
                    PromptType.SYSTEM,
                    "hr",
                ),
                PromptType.USER: PromptTemplate(
                    "Question: {query}", DocumentCategory.GENERAL, PromptType.USER, "hr"
                ),
            },
            DocumentCategory.TECHNICAL: {
                PromptType.SYSTEM: PromptTemplate(
                    "You are a technical expert",
                    DocumentCategory.TECHNICAL,
                    PromptType.SYSTEM,
                    "hr",
                )
            },
        }

        # Find templates containing "helpful"
        matches = find_template_by_content(templates, "helpful")
        assert len(matches) == 1
        assert (DocumentCategory.GENERAL, PromptType.SYSTEM) in matches

        # Find templates containing "you are" (should match multiple)
        matches = find_template_by_content(templates, "you are")
        assert len(matches) == 2

        # Find non-existent content
        matches = find_template_by_content(templates, "nonexistent")
        assert len(matches) == 0


# ================================
# MOCK PROVIDER TESTS
# ================================


class TestMockProviders:
    """Test mock providers for complete test isolation."""

    def test_mock_config_provider(self):
        """Test mock configuration provider."""
        # Test default configuration
        provider = MockConfigProvider()
        config = provider.get_prompt_config("hr")

        assert isinstance(config, PromptConfig)
        assert config.language == "hr"
        assert DocumentCategory.GENERAL in config.category_templates
        assert len(provider.call_history) == 1

        # Test custom configuration
        custom_config = create_test_config(language="en", include_technical=False)
        provider.set_config(custom_config)
        config = provider.get_prompt_config("en")

        assert config.language == "en"
        assert DocumentCategory.TECHNICAL not in config.category_templates

    def test_mock_config_provider_modifications(self):
        """Test mock provider template modifications."""
        provider = MockConfigProvider()

        # Add new template
        provider.add_category_template(
            DocumentCategory.CULTURAL, PromptType.SYSTEM, "Cultural system prompt"
        )

        config = provider.get_prompt_config("hr")
        assert DocumentCategory.CULTURAL in config.category_templates
        assert (
            config.category_templates[DocumentCategory.CULTURAL][PromptType.SYSTEM]
            == "Cultural system prompt"
        )

        # Remove template
        provider.remove_template(DocumentCategory.GENERAL, PromptType.USER)
        config = provider.get_prompt_config("hr")
        assert (
            PromptType.USER not in config.category_templates[DocumentCategory.GENERAL]
        )

    def test_mock_logger_provider(self):
        """Test mock logger provider message capture."""
        logger = MockLoggerProvider()

        # Test message capture
        logger.info("Test info")
        logger.debug("Test debug")
        logger.warning("Test warning")
        logger.error("Test error")

        assert len(logger.get_messages("info")) == 1
        assert len(logger.get_messages("debug")) == 1
        assert len(logger.get_messages("warning")) == 1
        assert len(logger.get_messages("error")) == 1

        assert logger.get_messages("info")[0] == "Test info"
        assert logger.get_messages("error")[0] == "Test error"

        # Test clear messages
        logger.clear_messages()
        assert len(logger.get_messages("info")) == 0


# ================================
# DEPENDENCY INJECTION TESTS
# ================================


class TestDependencyInjection:
    """Test dependency injection orchestration."""

    def test_enhanced_prompt_builder_initialization(self):
        """Test enhanced prompt builder initialization with DI."""
        config_provider, logger_provider = create_mock_setup()

        builder = create_enhanced_prompt_builder(
            language="hr",
            config_provider=config_provider,
            logger_provider=logger_provider,
        )

        assert isinstance(builder, _EnhancedPromptBuilder)
        assert builder.language == "hr"

        # Verify configuration was loaded
        assert len(config_provider.call_history) >= 1

        # Verify logger captured initialization
        init_messages = logger_provider.get_messages("info")
        assert len(init_messages) >= 1
        assert "template categories" in init_messages[0]

    def test_prompt_building_with_di(self):
        """Test prompt building with dependency injection."""
        custom_templates = {
            DocumentCategory.GENERAL: {
                PromptType.SYSTEM: "Custom system prompt.",
                PromptType.USER: "Custom Q: {query}\nCustom Context: {context}\nCustom A:",
            }
        }

        builder, (config_provider, logger_provider) = create_test_prompt_builder(
            language="hr", custom_templates=custom_templates
        )

        system_prompt, user_prompt = builder.build_prompt(
            query="Test question",
            context_chunks=["Test context chunk"],
            category=DocumentCategory.GENERAL,
            max_context_length=1000,
        )

        assert system_prompt == "Custom system prompt."
        assert "Test question" in user_prompt
        assert "Test context chunk" in user_prompt
        assert "Custom Q:" in user_prompt
        assert "Custom Context:" in user_prompt

    def test_followup_prompt_with_di(self):
        """Test follow-up prompt generation with DI."""
        custom_templates = {
            DocumentCategory.GENERAL: {
                PromptType.FOLLOWUP: "Prev Q: {original_query}\nPrev A: {original_answer}\nNew Q: {followup_query}"
            }
        }

        builder, providers = create_test_prompt_builder(
            custom_templates=custom_templates
        )

        followup = builder.get_followup_prompt(
            original_query="What is AI?",
            original_answer="AI is artificial intelligence.",
            followup_query="How does it work?",
            category=DocumentCategory.GENERAL,
        )

        assert "What is AI?" in followup
        assert "AI is artificial intelligence." in followup
        assert "How does it work?" in followup

    def test_template_validation_with_di(self):
        """Test template validation with dependency injection."""
        # Create invalid configuration (missing USER template)
        invalid_config = create_invalid_config("hr")

        builder, providers = create_test_prompt_builder(config=invalid_config)

        validation_result = builder.validate_templates()

        assert not validation_result["valid"]
        assert len(validation_result["issues"]) > 0

        # Verify logger captured validation warnings
        logger_provider = providers[1]
        warning_messages = logger_provider.get_messages("warning")
        assert len(warning_messages) >= 1
        assert "validation failed" in warning_messages[0]

    def test_template_stats_with_di(self):
        """Test template statistics with dependency injection."""
        builder, providers = create_test_prompt_builder()

        stats = builder.get_template_stats()

        assert "total_categories" in stats
        assert "language" in stats
        assert "categories" in stats
        assert stats["language"] == "hr"
        assert stats["total_categories"] > 0

    def test_error_handling_with_di(self):
        """Test error handling with dependency injection."""
        # Create minimal config without required templates
        minimal_config = create_minimal_config("hr")

        builder, (config_provider, logger_provider) = create_test_prompt_builder(
            config=minimal_config
        )

        # Remove USER template to cause error
        config_provider.remove_template(DocumentCategory.GENERAL, PromptType.USER)

        with pytest.raises(KeyError):
            builder.build_prompt(
                query="test",
                context_chunks=["context"],
                category=DocumentCategory.GENERAL,
            )

        # Verify error was logged
        error_messages = logger_provider.get_messages("error")
        assert len(error_messages) >= 1
        assert "Template not found" in error_messages[0]


# ================================
# INTEGRATION TESTS
# ================================


class TestIntegration:
    """Test complete integration scenarios."""

    def test_factory_functions(self):
        """Test factory function integration."""
        builder, providers = create_test_prompt_builder()
        config_provider, logger_provider = providers

        assert isinstance(builder, _EnhancedPromptBuilder)
        assert isinstance(config_provider, MockConfigProvider)
        assert isinstance(logger_provider, MockLoggerProvider)

        # Test functionality works
        system_prompt, user_prompt = builder.build_prompt(
            query="Integration test", context_chunks=["Integration context"]
        )

        assert "Integration test" in user_prompt
        assert "Integration context" in user_prompt

    def test_multilingual_integration(self):
        """Test integration with multiple languages."""
        # Create builders for different languages
        hr_builder, _ = create_test_prompt_builder(language="hr")
        en_builder, _ = create_test_prompt_builder(language="en")

        # Test Croatian formatting
        hr_config = create_test_config(language="hr")
        hr_system, hr_user = hr_builder.build_prompt(
            query="Što je ovo?", context_chunks=["Hrvatski kontekst"]
        )

        # Test English formatting
        en_config = create_test_config(language="en")
        en_system, en_user = en_builder.build_prompt(
            query="What is this?", context_chunks=["English context"]
        )

        assert "Što je ovo?" in hr_user
        assert "What is this?" in en_user
        assert "Hrvatski kontekst" in hr_user
        assert "English context" in en_user

    def test_category_specific_integration(self):
        """Test integration with different document categories."""
        builder, providers = create_test_prompt_builder()

        # Test different categories
        categories_to_test = [
            DocumentCategory.GENERAL,
            DocumentCategory.TECHNICAL,
            DocumentCategory.CULTURAL,
        ]

        for category in categories_to_test:
            system_prompt, user_prompt = builder.build_prompt(
                query=f"Test {category.value} question",
                context_chunks=[f"Test {category.value} context"],
                category=category,
            )

            assert f"Test {category.value} question" in user_prompt
            assert f"Test {category.value} context" in user_prompt

    def test_context_length_management(self):
        """Test context length management integration."""
        builder, providers = create_test_prompt_builder()

        # Create very long context chunks
        long_chunks = [
            "This is a very long context chunk that contains a lot of information "
            * 10,
            "This is another very long context chunk with even more information " * 10,
            "And yet another extremely long context chunk with detailed content " * 10,
        ]

        system_prompt, user_prompt = builder.build_prompt(
            query="Test question with long context",
            context_chunks=long_chunks,
            max_context_length=200,  # Small limit to force truncation
        )

        assert "Test question with long context" in user_prompt
        assert len(user_prompt) > 0  # Should still generate a prompt

        # Check if truncation occurred (logger should capture debug message)
        logger_provider = providers[1]
        debug_messages = logger_provider.get_messages("debug")
        # Might have truncation debug message, but not required for this test
