"""
Tests for enhanced prompt templates system.

Tests all data classes, pure functions, and the _EnhancedPromptBuilder class
with proper dependency injection patterns.
"""

import pytest
from unittest.mock import Mock

from src.generation.enhanced_prompt_templates import (
    # Data Classes
    PromptTemplate,
    PromptConfig,
    ContextFormattingOptions,
    BuildPromptResult,
    ValidationResult,
    TemplateStats,
    PromptType,

    # Protocols
    ConfigProvider,
    LoggerProvider,

    # Pure Functions
    parse_config_templates,
    format_context_chunks,
    build_category_prompt,
    build_followup_prompt,
    calculate_template_stats,
    validate_templates,
    suggest_template_improvements,
    get_missing_templates,
    find_template_by_content,

    # Main Class
    _EnhancedPromptBuilder,

    # Factory Functions
    create_enhanced_prompt_builder,
    EnhancedPromptBuilder,
)
from src.retrieval.categorization import CategoryType


class TestPromptTemplate:
    """Test PromptTemplate data class."""

    def test_prompt_template_creation(self):
        """Test basic prompt template creation."""
        template = PromptTemplate(
            template="Hello {name}",
            category=CategoryType.GENERAL,
            prompt_type=PromptType.SYSTEM,
            language="en"
        )

        assert template.template == "Hello {name}"
        assert template.category == CategoryType.GENERAL
        assert template.prompt_type == PromptType.SYSTEM
        assert template.language == "en"
        assert template.priority == 0
        assert template.metadata is None

    def test_prompt_template_with_metadata(self):
        """Test prompt template with metadata."""
        metadata = {"author": "test", "version": "1.0"}
        template = PromptTemplate(
            template="Test template",
            category=CategoryType.TECHNICAL,
            prompt_type=PromptType.USER,
            language="hr",
            priority=5,
            metadata=metadata
        )

        assert template.priority == 5
        assert template.metadata == metadata

    def test_format_success(self):
        """Test successful template formatting."""
        template = PromptTemplate(
            template="Hello {name}, you are {age} years old",
            category=CategoryType.GENERAL,
            prompt_type=PromptType.USER,
            language="en"
        )

        result = template.format(name="John", age=25)
        assert result == "Hello John, you are 25 years old"

    def test_format_missing_variable(self):
        """Test template formatting with missing variable."""
        template = PromptTemplate(
            template="Hello {name}, you are {age} years old",
            category=CategoryType.GENERAL,
            prompt_type=PromptType.USER,
            language="en"
        )

        result = template.format(name="John")
        # The format method returns the original template with missing variable info
        assert "{age} years old [MISSING: 'age']" in result


class TestPromptConfig:
    """Test PromptConfig data class."""

    def test_prompt_config_creation(self):
        """Test prompt configuration creation."""
        category_templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: "System prompt",
                PromptType.USER: "User prompt"
            }
        }
        messages = {"no_context": "No context available"}
        formatting = {"source_label": "Source"}

        config = PromptConfig(
            category_templates=category_templates,
            messages=messages,
            formatting=formatting,
            language="en"
        )

        assert config.category_templates == category_templates
        assert config.messages == messages
        assert config.formatting == formatting
        assert config.language == "en"


class TestContextFormattingOptions:
    """Test ContextFormattingOptions data class."""

    def test_default_options(self):
        """Test default formatting options."""
        options = ContextFormattingOptions()

        assert options.max_length == 2000
        assert options.include_attribution is True
        assert options.source_label == "Source"
        assert options.truncation_indicator == "..."
        assert options.min_chunk_size == 100

    def test_custom_options(self):
        """Test custom formatting options."""
        options = ContextFormattingOptions(
            max_length=1500,
            include_attribution=False,
            source_label="Izvor",
            truncation_indicator="...[skraćeno]",
            min_chunk_size=150
        )

        assert options.max_length == 1500
        assert options.include_attribution is False
        assert options.source_label == "Izvor"
        assert options.truncation_indicator == "...[skraćeno]"
        assert options.min_chunk_size == 150


class TestBuildPromptResult:
    """Test BuildPromptResult data class."""

    def test_build_prompt_result_creation(self):
        """Test build prompt result creation."""
        result = BuildPromptResult(
            system_prompt="System: Answer questions",
            user_prompt="User: What is RAG?",
            context_used="Context about RAG",
            chunks_included=3,
            truncated=False
        )

        assert result.system_prompt == "System: Answer questions"
        assert result.user_prompt == "User: What is RAG?"
        assert result.context_used == "Context about RAG"
        assert result.chunks_included == 3
        assert result.truncated is False


class TestValidationResult:
    """Test ValidationResult data class."""

    def test_validation_result_creation(self):
        """Test validation result creation."""
        issues = ["Missing {query} placeholder"]
        warnings = ["Template is very short"]

        result = ValidationResult(
            valid=False,
            issues=issues,
            warnings=warnings
        )

        assert result.valid is False
        assert result.issues == issues
        assert result.warnings == warnings


class TestTemplateStats:
    """Test TemplateStats data class."""

    def test_template_stats_creation(self):
        """Test template statistics creation."""
        categories = {
            "general": {"template_count": 2, "template_types": ["system", "user"]}
        }

        stats = TemplateStats(
            total_categories=1,
            language="en",
            categories=categories
        )

        assert stats.total_categories == 1
        assert stats.language == "en"
        assert stats.categories == categories


class TestPureFunctions:
    """Test pure business logic functions."""

    def test_parse_config_templates_basic(self):
        """Test parsing configuration templates."""
        config_data = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: "System prompt for general queries",
                PromptType.USER: "User: {query} Context: {context}"
            }
        }

        templates = parse_config_templates(config_data, "en")

        assert CategoryType.GENERAL in templates
        assert PromptType.SYSTEM in templates[CategoryType.GENERAL]
        assert PromptType.USER in templates[CategoryType.GENERAL]

        system_template = templates[CategoryType.GENERAL][PromptType.SYSTEM]
        assert system_template.template == "System prompt for general queries"
        assert system_template.category == CategoryType.GENERAL
        assert system_template.language == "en"

    def test_parse_config_templates_string_keys(self):
        """Test parsing templates with string keys from TOML."""
        config_data = {
            "general": {
                "system": "System prompt",
                "user": "User prompt"
            }
        }

        templates = parse_config_templates(config_data, "hr")

        assert CategoryType.GENERAL in templates
        assert PromptType.SYSTEM in templates[CategoryType.GENERAL]
        assert templates[CategoryType.GENERAL][PromptType.SYSTEM].language == "hr"

    def test_parse_config_templates_unknown_category(self):
        """Test parsing with unknown category."""
        config_data = {
            "unknown_category": {
                "system": "System prompt"
            }
        }

        templates = parse_config_templates(config_data, "en")

        # Unknown categories should be skipped
        assert len(templates) == 0

    def test_parse_config_templates_unknown_type(self):
        """Test parsing with unknown prompt type."""
        config_data = {
            CategoryType.GENERAL: {
                "unknown_type": "Some prompt"
            }
        }

        templates = parse_config_templates(config_data, "en")

        # Unknown types should be skipped
        assert CategoryType.GENERAL in templates
        assert len(templates[CategoryType.GENERAL]) == 0

    def test_format_context_chunks_empty(self):
        """Test formatting empty context chunks."""
        options = ContextFormattingOptions()

        result, truncated = format_context_chunks([], options)

        assert result == "No relevant context available."
        assert truncated is False

    def test_format_context_chunks_basic(self):
        """Test formatting basic context chunks."""
        chunks = ["First chunk content", "Second chunk content"]
        options = ContextFormattingOptions(include_attribution=True)

        result, truncated = format_context_chunks(chunks, options)

        assert "[Source 1]:" in result
        assert "[Source 2]:" in result
        assert "First chunk content" in result
        assert "Second chunk content" in result
        assert truncated is False

    def test_format_context_chunks_no_attribution(self):
        """Test formatting without attribution."""
        chunks = ["First chunk", "Second chunk"]
        options = ContextFormattingOptions(include_attribution=False)

        result, truncated = format_context_chunks(chunks, options)

        assert "[Source" not in result
        assert "First chunk" in result
        assert "Second chunk" in result

    def test_format_context_chunks_truncation(self):
        """Test context chunk truncation."""
        # Create chunks that exceed max length
        long_chunk = "a" * 1000
        chunks = [long_chunk, long_chunk, long_chunk]
        options = ContextFormattingOptions(max_length=1500, include_attribution=True)

        result, truncated = format_context_chunks(chunks, options)

        assert truncated is True
        assert len(result) <= options.max_length
        assert options.truncation_indicator in result

    def test_format_context_chunks_custom_options(self):
        """Test formatting with custom options."""
        chunks = ["Content 1", "Content 2"]
        options = ContextFormattingOptions(
            include_attribution=True,
            source_label="Izvor",
            truncation_indicator="...[skraćeno]"
        )

        result, truncated = format_context_chunks(chunks, options)

        assert "[Izvor 1]:" in result
        assert "[Izvor 2]:" in result

    def test_build_category_prompt_success(self):
        """Test successful category prompt building."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="You are a helpful assistant.",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                ),
                PromptType.USER: PromptTemplate(
                    template="Question: {query}\nContext: {context}",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.USER,
                    language="en"
                )
            }
        }

        chunks = ["Relevant information here"]
        options = ContextFormattingOptions(include_attribution=False)

        result = build_category_prompt(
            query="What is RAG?",
            context_chunks=chunks,
            category=CategoryType.GENERAL,
            templates=templates,
            formatting_options=options
        )

        assert result.system_prompt == "You are a helpful assistant."
        assert "What is RAG?" in result.user_prompt
        assert "Relevant information here" in result.user_prompt
        assert result.chunks_included == 1
        assert result.truncated is False

    def test_build_category_prompt_missing_category(self):
        """Test building prompt with missing category."""
        templates = {}

        with pytest.raises(KeyError, match="No templates found for category"):
            build_category_prompt(
                query="Test",
                context_chunks=[],
                category=CategoryType.GENERAL,
                templates=templates,
                formatting_options=ContextFormattingOptions()
            )

    def test_build_category_prompt_missing_system_template(self):
        """Test building prompt with missing system template."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.USER: PromptTemplate(
                    template="User prompt",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.USER,
                    language="en"
                )
            }
        }

        with pytest.raises(KeyError, match="No system template found"):
            build_category_prompt(
                query="Test",
                context_chunks=[],
                category=CategoryType.GENERAL,
                templates=templates,
                formatting_options=ContextFormattingOptions()
            )

    def test_build_category_prompt_missing_user_template(self):
        """Test building prompt with missing user template."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="System prompt",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                )
            }
        }

        with pytest.raises(KeyError, match="No user template found"):
            build_category_prompt(
                query="Test",
                context_chunks=[],
                category=CategoryType.GENERAL,
                templates=templates,
                formatting_options=ContextFormattingOptions()
            )

    def test_build_followup_prompt_success(self):
        """Test successful followup prompt building."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.FOLLOWUP: PromptTemplate(
                    template="Previous: {original_query}\nAnswer: {original_answer}\nNew: {followup_query}",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.FOLLOWUP,
                    language="en"
                )
            }
        }

        result = build_followup_prompt(
            original_query="What is AI?",
            original_answer="AI is artificial intelligence",
            followup_query="How does it work?",
            category=CategoryType.GENERAL,
            templates=templates
        )

        assert "What is AI?" in result
        assert "AI is artificial intelligence" in result
        assert "How does it work?" in result

    def test_build_followup_prompt_missing_category(self):
        """Test followup prompt with missing category."""
        templates = {}

        with pytest.raises(KeyError, match="No templates found for category"):
            build_followup_prompt("q1", "a1", "q2", CategoryType.GENERAL, templates)

    def test_build_followup_prompt_missing_template(self):
        """Test followup prompt with missing followup template."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="System",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                )
            }
        }

        with pytest.raises(KeyError, match="No followup template found"):
            build_followup_prompt("q1", "a1", "q2", CategoryType.GENERAL, templates)

    def test_calculate_template_stats(self):
        """Test template statistics calculation."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="Short",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                ),
                PromptType.USER: PromptTemplate(
                    template="Longer template content",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.USER,
                    language="en"
                )
            },
            CategoryType.TECHNICAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="Technical system prompt",
                    category=CategoryType.TECHNICAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                )
            }
        }

        stats = calculate_template_stats(templates, "en")

        assert stats.total_categories == 2
        assert stats.language == "en"
        assert "general" in stats.categories
        assert "technical" in stats.categories
        assert stats.categories["general"]["template_count"] == 2
        assert stats.categories["technical"]["template_count"] == 1
        assert "system" in stats.categories["general"]["template_types"]
        assert "user" in stats.categories["general"]["template_types"]

    def test_validate_templates_success(self):
        """Test successful template validation."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="You are a helpful assistant specialized in answering questions.",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                ),
                PromptType.USER: PromptTemplate(
                    template="Question: {query}\nContext: {context}\nAnswer:",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.USER,
                    language="en"
                ),
                PromptType.FOLLOWUP: PromptTemplate(
                    template="Original: {original_query}\nPrevious: {original_answer}\nNew: {followup_query}",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.FOLLOWUP,
                    language="en"
                )
            }
        }

        result = validate_templates(templates)

        assert result.valid is True
        assert len(result.issues) == 0
        assert len(result.warnings) == 0

    def test_validate_templates_missing_placeholders(self):
        """Test validation with missing placeholders."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.USER: PromptTemplate(
                    template="Question without placeholders",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.USER,
                    language="en"
                ),
                PromptType.FOLLOWUP: PromptTemplate(
                    template="Followup without required placeholders",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.FOLLOWUP,
                    language="en"
                )
            }
        }

        result = validate_templates(templates)

        assert result.valid is False
        assert any("Missing {query} placeholder" in issue for issue in result.issues)
        assert any("Missing {context} placeholder" in issue for issue in result.issues)
        assert any("Missing {original_query} placeholder" in issue for issue in result.issues)
        assert any("Missing {original_answer} placeholder" in issue for issue in result.issues)
        assert any("Missing {followup_query} placeholder" in issue for issue in result.issues)

    def test_validate_templates_length_warnings(self):
        """Test validation with template length warnings."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="Short",  # Too short
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                ),
                PromptType.USER: PromptTemplate(
                    template="Question: {query} Context: {context} " + "a" * 500,  # Long but with required placeholders
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.USER,
                    language="en"
                )
            }
        }

        result = validate_templates(templates)

        assert result.valid is True  # No issues, just warnings
        assert any("Very short template" in warning for warning in result.warnings)
        assert any("Very long template" in warning for warning in result.warnings)

    def test_suggest_template_improvements_missing_category(self):
        """Test suggestions for missing category."""
        templates = {}

        suggestions = suggest_template_improvements(templates, CategoryType.GENERAL, {})

        assert any("Add specialized templates for general category" in s for s in suggestions)

    def test_suggest_template_improvements_missing_types(self):
        """Test suggestions for missing template types."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="System only",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                )
            }
        }

        suggestions = suggest_template_improvements(templates, CategoryType.GENERAL, {})

        assert any("Add missing template types" in s for s in suggestions)

    def test_suggest_template_improvements_performance(self):
        """Test performance-based suggestions."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="System",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                ),
                PromptType.USER: PromptTemplate(
                    template="User: {query} Context: {context}",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.USER,
                    language="en"
                )
            }
        }

        usage_stats = {"avg_confidence": 0.5, "avg_response_length": 50}
        suggestions = suggest_template_improvements(templates, CategoryType.GENERAL, usage_stats)

        assert any("improve response quality" in s for s in suggestions)
        assert any("too brief responses" in s for s in suggestions)

    def test_get_missing_templates_complete(self):
        """Test identifying missing templates."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="System",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                )
            }
        }

        required_categories = [CategoryType.GENERAL, CategoryType.TECHNICAL]
        required_types = [PromptType.SYSTEM, PromptType.USER]

        missing = get_missing_templates(templates, required_categories, required_types)

        assert "technical" in missing["categories"]
        assert "general/user" in missing["templates"]

    def test_find_template_by_content(self):
        """Test finding templates by content."""
        templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="You are a helpful assistant",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                ),
                PromptType.USER: PromptTemplate(
                    template="Question: {query}",
                    category=CategoryType.GENERAL,
                    prompt_type=PromptType.USER,
                    language="en"
                )
            },
            CategoryType.TECHNICAL: {
                PromptType.SYSTEM: PromptTemplate(
                    template="Technical assistance provided",
                    category=CategoryType.TECHNICAL,
                    prompt_type=PromptType.SYSTEM,
                    language="en"
                )
            }
        }

        matches = find_template_by_content(templates, "helpful")

        assert len(matches) == 1
        assert matches[0] == (CategoryType.GENERAL, PromptType.SYSTEM)

        # Test case insensitive search
        matches = find_template_by_content(templates, "QUESTION")
        assert len(matches) == 1
        assert matches[0] == (CategoryType.GENERAL, PromptType.USER)


class TestEnhancedPromptBuilder:
    """Test _EnhancedPromptBuilder class."""

    def create_test_providers(self):
        """Create mock providers for testing."""
        config_provider = Mock(spec=ConfigProvider)
        logger_provider = Mock(spec=LoggerProvider)

        # Mock prompt configuration
        category_templates = {
            CategoryType.GENERAL: {
                PromptType.SYSTEM: "You are a helpful assistant.",
                PromptType.USER: "Question: {query}\nContext: {context}",
                PromptType.FOLLOWUP: "Previous: {original_query}\nAnswer: {original_answer}\nNew: {followup_query}"
            },
            CategoryType.TECHNICAL: {
                PromptType.SYSTEM: "You are a technical expert.",
                PromptType.USER: "Technical question: {query}\nReference: {context}"
            }
        }

        messages = {
            "no_context": "No relevant context available."
        }

        formatting = {
            "source_label": "Source",
            "truncation_indicator": "...",
            "min_chunk_size": "100"
        }

        prompt_config = PromptConfig(
            category_templates=category_templates,
            messages=messages,
            formatting=formatting,
            language="en"
        )

        config_provider.get_prompt_config.return_value = prompt_config

        return config_provider, logger_provider

    def test_enhanced_prompt_builder_initialization(self):
        """Test enhanced prompt builder initialization."""
        config_provider, logger_provider = self.create_test_providers()

        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        assert builder.language == "en"
        assert builder._config_provider == config_provider
        assert builder._logger == logger_provider
        config_provider.get_prompt_config.assert_called_once_with("en")
        logger_provider.info.assert_called_once()

    def test_enhanced_prompt_builder_without_logger(self):
        """Test builder initialization without logger."""
        config_provider, _ = self.create_test_providers()

        builder = _EnhancedPromptBuilder("hr", config_provider, None)

        assert builder.language == "hr"
        assert builder._logger is None

    def test_build_prompt_success(self):
        """Test successful prompt building."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        system_prompt, user_prompt = builder.build_prompt(
            query="What is RAG?",
            context_chunks=["RAG stands for Retrieval-Augmented Generation"],
            category=CategoryType.GENERAL
        )

        assert system_prompt == "You are a helpful assistant."
        assert "What is RAG?" in user_prompt
        assert "RAG stands for Retrieval-Augmented Generation" in user_prompt

    def test_build_prompt_with_options(self):
        """Test prompt building with custom options."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        system_prompt, user_prompt = builder.build_prompt(
            query="Technical question",
            context_chunks=["Technical context here"],
            category=CategoryType.TECHNICAL,
            max_context_length=1000,
            include_source_attribution=False
        )

        assert system_prompt == "You are a technical expert."
        assert "Technical question" in user_prompt
        assert "Technical context here" in user_prompt
        assert "[Source" not in user_prompt  # No attribution

    def test_build_prompt_missing_template(self):
        """Test prompt building with missing template."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        with pytest.raises(KeyError):
            builder.build_prompt(
                query="Test",
                context_chunks=[],
                category=CategoryType.ACADEMIC  # Not in mock config
            )

        logger_provider.error.assert_called_once()

    def test_get_followup_prompt_success(self):
        """Test successful followup prompt generation."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        followup = builder.get_followup_prompt(
            original_query="What is AI?",
            original_answer="AI is artificial intelligence",
            followup_query="How does it work?",
            category=CategoryType.GENERAL
        )

        assert "What is AI?" in followup
        assert "AI is artificial intelligence" in followup
        assert "How does it work?" in followup

    def test_get_followup_prompt_missing_template(self):
        """Test followup prompt with missing template."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        with pytest.raises(KeyError):
            builder.get_followup_prompt(
                original_query="Test",
                original_answer="Answer",
                followup_query="Followup",
                category=CategoryType.TECHNICAL  # No followup template
            )

        logger_provider.error.assert_called_once()

    def test_suggest_template_improvements(self):
        """Test template improvement suggestions."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        usage_stats = {"avg_confidence": 0.5}
        suggestions = builder.suggest_template_improvements(CategoryType.GENERAL, usage_stats)

        assert isinstance(suggestions, list)
        logger_provider.info.assert_called()  # Once for init, once for suggestions

    def test_get_template_stats(self):
        """Test getting template statistics."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        stats = builder.get_template_stats()

        assert "total_categories" in stats
        assert "language" in stats
        assert "categories" in stats
        assert stats["language"] == "en"
        assert stats["total_categories"] == 2  # GENERAL and TECHNICAL

    def test_validate_templates_success(self):
        """Test successful template validation."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        validation = builder.validate_templates()

        assert "valid" in validation
        assert "issues" in validation
        assert "warnings" in validation
        assert validation["valid"] is True

    def test_validate_templates_with_issues(self):
        """Test template validation with issues."""
        config_provider, logger_provider = self.create_test_providers()

        # Modify config to have invalid templates
        invalid_config = config_provider.get_prompt_config.return_value
        invalid_config.category_templates[CategoryType.GENERAL][PromptType.USER] = "Invalid template"

        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)
        validation = builder.validate_templates()

        assert validation["valid"] is False
        assert len(validation["issues"]) > 0
        logger_provider.warning.assert_called()

    def test_get_missing_templates_default_params(self):
        """Test getting missing templates with default parameters."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        missing = builder.get_missing_templates()

        assert "categories" in missing
        assert "templates" in missing
        # Should check all categories and system/user types by default

    def test_get_missing_templates_custom_params(self):
        """Test getting missing templates with custom parameters."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        missing = builder.get_missing_templates(
            required_categories=[CategoryType.ACADEMIC],
            required_types=[PromptType.SYSTEM]
        )

        assert "academic" in missing["categories"]

    def test_find_templates_by_content(self):
        """Test finding templates by content."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        matches = builder.find_templates_by_content("helpful")

        assert isinstance(matches, list)
        assert len(matches) > 0
        assert all(isinstance(match, tuple) and len(match) == 2 for match in matches)

    def test_logging_methods(self):
        """Test logging methods."""
        config_provider, logger_provider = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, logger_provider)

        builder._log_info("Test info")
        builder._log_debug("Test debug")
        builder._log_warning("Test warning")
        builder._log_error("Test error")

        logger_provider.info.assert_called()
        logger_provider.debug.assert_called_with("Test debug")
        logger_provider.warning.assert_called_with("Test warning")
        logger_provider.error.assert_called_with("Test error")

    def test_logging_methods_without_logger(self):
        """Test logging methods without logger provider."""
        config_provider, _ = self.create_test_providers()
        builder = _EnhancedPromptBuilder("en", config_provider, None)

        # Should not raise exceptions
        builder._log_info("Test")
        builder._log_debug("Test")
        builder._log_warning("Test")
        builder._log_error("Test")


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_enhanced_prompt_builder(self):
        """Test create_enhanced_prompt_builder factory."""
        config_provider = Mock(spec=ConfigProvider)
        logger_provider = Mock(spec=LoggerProvider)

        # Mock configuration
        prompt_config = PromptConfig(
            category_templates={},
            messages={},
            formatting={},
            language="en"
        )
        config_provider.get_prompt_config.return_value = prompt_config

        builder = create_enhanced_prompt_builder("en", config_provider, logger_provider)

        assert isinstance(builder, _EnhancedPromptBuilder)
        assert builder.language == "en"

    def test_enhanced_prompt_builder_with_providers(self):
        """Test EnhancedPromptBuilder with explicit providers."""
        config_provider = Mock(spec=ConfigProvider)
        logger_provider = Mock(spec=LoggerProvider)

        # Mock configuration
        prompt_config = PromptConfig(
            category_templates={},
            messages={},
            formatting={},
            language="hr"
        )
        config_provider.get_prompt_config.return_value = prompt_config

        builder = EnhancedPromptBuilder("hr", config_provider, logger_provider)

        assert isinstance(builder, _EnhancedPromptBuilder)
        assert builder.language == "hr"