"""
Provider implementations for categorization dependency injection.
Production and mock providers for testable categorization system.
"""

from typing import Any

from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start, log_error_context
from .categorization import ConfigProvider


class DefaultConfigProvider:
    """Default configuration provider using TOML configuration files."""

    def __init__(self):
        logger = get_system_logger()
        log_component_start("config_provider", "init")

        # Import at runtime to avoid circular dependencies
        from ..utils import config_loader

        self._config_loader = config_loader

        logger.debug("config_provider", "init", "Config provider initialized")
        log_component_end("config_provider", "init", "Config provider ready")

    def get_categorization_config(self, language: str) -> dict[str, Any]:
        """Get categorization configuration for specified language."""
        logger = get_system_logger()
        log_component_start("config_provider", "get_categorization_config", language=language)

        try:
            # Use the main config provider to get properly merged categorization config
            from ..utils.config_protocol import get_config_provider

            logger.debug(
                "config_provider", "get_categorization_config", f"Loading categorization config for {language}"
            )
            main_provider = get_config_provider()
            config = main_provider.get_categorization_config(language)

            logger.debug("config_provider", "get_categorization_config", f"Loaded config with {len(config)} keys")
            logger.trace("config_provider", "get_categorization_config", f"Config keys: {list(config.keys())}")

            log_component_end(
                "config_provider", "get_categorization_config", f"Categorization config loaded for {language}"
            )
            return config

        except Exception as e:
            log_error_context("config_provider", "get_categorization_config", e, {"language": language})
            raise


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


# ================================
# CONVENIENCE FACTORY FUNCTIONS
# ================================


def create_config_provider(use_mock: bool = False) -> ConfigProvider:
    """
    Create configuration provider.

    Args:
        use_mock: If True, return MockConfigProvider; otherwise DefaultConfigProvider

    Returns:
        ConfigProvider instance
    """
    if use_mock:
        from tests.conftest import MockConfigProvider

        return MockConfigProvider()
    return DefaultConfigProvider()


def create_minimal_config() -> dict[str, Any]:
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
