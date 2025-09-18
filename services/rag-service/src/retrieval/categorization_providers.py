"""
Provider implementations for categorization dependency injection.
Production and mock providers for testable categorization system.
"""

from typing import Any

from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start, log_error_context
from .categorization import ConfigProvider


class ProductionConfigProvider:
    """Production configuration provider using real TOML configuration."""

    def __init__(self):
        logger = get_system_logger()
        log_component_start("production_config_provider", "init")

        # Import at runtime to avoid circular dependencies
        from ..utils import config_loader

        self._config_loader = config_loader

        logger.debug("production_config_provider", "init", "Production config provider initialized")
        log_component_end("production_config_provider", "init", "Production config provider ready")

    def get_categorization_config(self, language: str) -> dict[str, Any]:
        """Get categorization configuration for specified language."""
        logger = get_system_logger()
        log_component_start("production_config_provider", "get_categorization_config", language=language)

        try:
            # Use the main config provider to get properly merged categorization config
            from ..utils.config_protocol import get_config_provider

            logger.debug(
                "production_config_provider",
                "get_categorization_config",
                f"Loading categorization config for {language}",
            )
            main_provider = get_config_provider()
            config = main_provider.get_categorization_config(language)

            logger.debug(
                "production_config_provider", "get_categorization_config", f"Loaded config with {len(config)} keys"
            )
            logger.trace(
                "production_config_provider", "get_categorization_config", f"Config keys: {list(config.keys())}"
            )

            log_component_end(
                "production_config_provider",
                "get_categorization_config",
                f"Categorization config loaded for {language}",
            )
            return config

        except Exception as e:
            log_error_context("production_config_provider", "get_categorization_config", e, {"language": language})
            raise


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self, mock_configs: dict[str, dict[str, Any]] | None = None):
        """Initialize with mock configuration data."""
        self.mock_configs = mock_configs or {}
        self._default_config = self._create_default_test_config()

    def set_categorization_config(self, language: str, config_data: dict[str, Any]) -> None:
        """Set mock categorization configuration for specified language."""
        self.mock_configs[f"categorization_{language}"] = config_data

    def get_categorization_config(self, language: str) -> dict[str, Any]:
        """Get mock categorization configuration."""
        config_key = f"categorization_{language}"
        if config_key in self.mock_configs:
            return self.mock_configs[config_key]

        # Return language-specific default if no mock set
        return self._get_language_default_config(language)

    def _create_default_test_config(self) -> dict[str, Any]:
        """Create default test configuration."""
        return {
            "categories": {
                "general": {"priority": 1},
                "technical": {"priority": 2},
                "cultural": {"priority": 3},
                "academic": {"priority": 4},
            },
            "patterns": {
                "general": ["test", "example"],
                "technical": ["API", "database", "server", "kod", "programming"],
                "cultural": ["kultura", "tradicija", "culture", "tradition"],
                "academic": ["research", "study", "istraživanje", "studij"],
            },
            "cultural_keywords": {
                "test_croatian": ["test_hrvatski", "test_zagreb"],
                "test_english": ["test_english", "test_london"],
            },
            "complexity_thresholds": {"simple": 1.0, "moderate": 3.0, "complex": 6.0, "analytical": 10.0},
            "retrieval_strategies": {
                "default": "test_hybrid",
                "category_technical": "test_dense",
                "category_cultural": "test_cultural",
                "complexity_simple": "test_sparse",
                "cultural_context": "test_cultural_aware",
            },
        }

    def _get_language_default_config(self, language: str) -> dict[str, Any]:
        """Get language-specific default configuration for testing."""
        base_config = self._default_config.copy()

        if language == "hr":
            # Croatian-specific test patterns
            base_config["patterns"]["cultural"].extend(["hrvatska", "dubrovnik", "split", "zagreb"])
            base_config["cultural_keywords"]["croatian_test"] = ["test_hr", "test_croatia", "test_jadran"]
        elif language == "en":
            # English-specific test patterns
            base_config["patterns"]["cultural"].extend(["england", "london", "british", "american"])
            base_config["cultural_keywords"]["english_test"] = ["test_en", "test_uk", "test_usa"]

        return base_config


class NoOpLoggerProvider:
    """No-operation logger for testing (silent)."""

    def info(self, message: str) -> None:
        """Silent info logging."""
        pass

    def debug(self, message: str) -> None:
        """Silent debug logging."""
        pass

    def warning(self, message: str) -> None:
        """Silent warning logging."""
        pass


class MockLoggerProvider:
    """Test logger that captures messages for verification."""

    def __init__(self):
        """Initialize message capture."""
        self.messages: dict[str, list[str]] = {"info": [], "debug": [], "warning": []}

    def info(self, message: str) -> None:
        """Capture info message."""
        self.messages["info"].append(message)

    def debug(self, message: str) -> None:
        """Capture debug message."""
        self.messages["debug"].append(message)

    def warning(self, message: str) -> None:
        """Capture warning message."""
        self.messages["warning"].append(message)

    def clear_messages(self) -> None:
        """Clear all captured messages."""
        for level in self.messages:
            self.messages[level].clear()

    def get_messages(self, level: str | None = None) -> dict[str, list[str]] | list[str]:
        """Get captured messages by level or all messages."""
        if level:
            return self.messages.get(level, [])
        return self.messages


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_config_provider(use_mock: bool = False) -> ConfigProvider:
    """
    Create configuration provider based on environment.

    Args:
        use_mock: Whether to use mock provider for testing

    Returns:
        ConfigProvider instance
    """
    if use_mock:
        return MockConfigProvider()
    else:
        return ProductionConfigProvider()


def create_test_categorization_setup(
    language: str = "hr", custom_config: dict[str, Any] | None = None
) -> tuple[MockConfigProvider, MockLoggerProvider]:
    """
    Create complete test setup for categorization testing.

    Args:
        language: Language for test setup
        custom_config: Custom configuration to use (optional)

    Returns:
        Tuple of (mock_config_provider, test_logger_provider)
    """
    mock_config = MockConfigProvider()

    if custom_config:
        mock_config.set_categorization_config(language, custom_config)

    test_logger = MockLoggerProvider()

    return mock_config, test_logger


def create_minimal_test_config() -> dict[str, Any]:
    """Create minimal test configuration for basic testing."""
    return {
        "categories": {"general": {"priority": 1}, "technical": {"priority": 2}},
        "patterns": {"general": ["test"], "technical": ["API", "kod"]},
        "cultural_keywords": {"test": ["test_keyword"]},
        "complexity_thresholds": {"simple": 2.0, "moderate": 5.0, "complex": 8.0, "analytical": 12.0},
        "retrieval_strategies": {"default": "hybrid", "category_technical": "dense"},
    }


def create_complex_test_config() -> dict[str, Any]:
    """Create complex test configuration for comprehensive testing."""
    return {
        "categories": {
            "general": {"priority": 1},
            "technical": {"priority": 2},
            "cultural": {"priority": 3},
            "academic": {"priority": 4},
            "legal": {"priority": 5},
            "medical": {"priority": 6},
        },
        "patterns": {
            "general": ["what", "how", "why", "što", "kako", "zašto"],
            "technical": ["API", "database", "server", "kod", "programming", "software", "system"],
            "cultural": ["kultura", "tradicija", "povijest", "culture", "tradition", "history"],
            "academic": ["research", "study", "analysis", "istraživanje", "studij", "analiza"],
            "legal": ["law", "legal", "court", "zakon", "pravni", "sud"],
            "medical": ["health", "medicine", "treatment", "zdravlje", "medicina", "liječenje"],
        },
        "cultural_keywords": {
            "croatian_culture": ["hrvatski", "hrvatska", "zagreb", "split", "dubrovnik", "jadran"],
            "croatian_history": ["povijest", "domovinski rat", "jugoslavija", "ndh"],
            "croatian_language": ["ije", "je", "kajkavski", "čakavski", "štokavski"],
            "english_culture": ["english", "british", "american", "london", "new york"],
            "general_culture": ["culture", "tradition", "heritage", "kultura", "tradicija"],
        },
        "complexity_thresholds": {"simple": 1.5, "moderate": 4.0, "complex": 7.5, "analytical": 11.0},
        "retrieval_strategies": {
            "default": "hybrid",
            "category_general": "sparse",
            "category_technical": "dense",
            "category_cultural": "cultural_context",
            "category_academic": "hierarchical",
            "category_legal": "precise",
            "category_medical": "specialized",
            "complexity_simple": "sparse",
            "complexity_moderate": "hybrid",
            "complexity_complex": "dense",
            "complexity_analytical": "hierarchical",
            "cultural_context": "cultural_aware",
        },
    }
