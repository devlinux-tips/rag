"""
Language configuration providers for Ollama client dependency injection.
Provides testable language configuration abstraction layer.
"""

from typing import Any, cast

from ..utils.config_loader import ConfigError as ConfigurationError
from ..utils.logging_factory import get_system_logger, log_component_end, log_component_start, log_error_context


class DefaultLanguageProvider:
    """
    Production language config provider using actual configuration files.
    """

    def get_formal_prompts(self, language: str) -> dict[str, str]:
        """Get formal prompt templates for language.

        Raises:
            ConfigurationError: If language configuration is missing or invalid
        """
        logger = get_system_logger()
        log_component_start("language_provider", "get_formal_prompts", language=language)

        try:
            from ..utils.config_loader import get_language_specific_config

            logger.debug("language_provider", "get_formal_prompts", f"Loading prompts config for {language}")
            language_config = get_language_specific_config("prompts", language)

            if "formal" not in language_config:
                error_msg = f"Missing 'formal' prompts section in {language}.toml configuration"
                logger.error("language_provider", "get_formal_prompts", error_msg)
                raise ConfigurationError(error_msg)

            formal_prompts = cast(dict[str, str], language_config["formal"])
            logger.debug("language_provider", "get_formal_prompts", f"Loaded {len(formal_prompts)} formal prompts")
            logger.trace("language_provider", "get_formal_prompts", f"Prompt keys: {list(formal_prompts.keys())}")

            log_component_end("language_provider", "get_formal_prompts", f"Formal prompts loaded for {language}")
            return formal_prompts

        except Exception as e:
            log_error_context("language_provider", "get_formal_prompts", e, {"language": language})
            raise ConfigurationError(f"Failed to load formal prompts for {language}: {e}") from e

    def get_error_template(self, language: str) -> str:
        """Get error message template for language.

        Raises:
            ConfigurationError: If language configuration is missing or invalid
        """
        logger = get_system_logger()
        log_component_start("language_provider", "get_error_template", language=language)

        try:
            from ..utils.config_loader import get_language_specific_config

            logger.debug("language_provider", "get_error_template", f"Loading error template for {language}")
            language_config = get_language_specific_config("prompts", language)

            if "error_message_template" not in language_config:
                error_msg = f"Missing 'error_message_template' in {language}.toml prompts configuration"
                logger.error("language_provider", "get_error_template", error_msg)
                raise ConfigurationError(error_msg)

            error_template = cast(str, language_config["error_message_template"])
            logger.trace(
                "language_provider", "get_error_template", f"Error template length: {len(error_template)} chars"
            )

            log_component_end("language_provider", "get_error_template", f"Error template loaded for {language}")
            return error_template

        except Exception as e:
            log_error_context("language_provider", "get_error_template", e, {"language": language})
            raise ConfigurationError(f"Failed to load error template for {language}: {e}") from e

    def get_confidence_settings(self, language: str) -> dict[str, Any]:
        """Get confidence calculation settings for language.

        Raises:
            ConfigurationError: If language configuration is missing or invalid
        """
        logger = get_system_logger()
        log_component_start("language_provider", "get_confidence_settings", language=language)

        try:
            from ..utils.config_loader import get_language_specific_config

            logger.debug("language_provider", "get_confidence_settings", f"Loading confidence settings for {language}")
            confidence_settings = cast(dict[str, Any], get_language_specific_config("confidence", language))

            logger.debug(
                "language_provider",
                "get_confidence_settings",
                f"Loaded confidence settings with {len(confidence_settings)} keys",
            )
            logger.trace(
                "language_provider",
                "get_confidence_settings",
                f"Confidence setting keys: {list(confidence_settings.keys())}",
            )

            log_component_end(
                "language_provider", "get_confidence_settings", f"Confidence settings loaded for {language}"
            )
            return confidence_settings

        except Exception as e:
            log_error_context("language_provider", "get_confidence_settings", e, {"language": language})
            raise ConfigurationError(f"Failed to load confidence settings for {language}: {e}") from e
