"""
Configuration Protocol for Dependency Injection.

This module defines the ConfigProvider protocol to enable dependency injection
for configuration loading, making the system testable while maintaining clean APIs.
"""

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration providers to enable dependency injection."""

    def load_config(self, config_name: str, use_cache: bool = True) -> dict[str, Any]:
        """Load a configuration file."""
        ...

    def get_config_section(self, config_name: str, section: str) -> dict[str, Any]:
        """Get a specific section from a config file."""
        ...

    def get_shared_config(self) -> dict[str, Any]:
        """Get shared configuration."""
        ...

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get configuration for specified language."""
        ...

    def get_language_specific_config(
        self, section: str, language: str
    ) -> dict[str, Any]:
        """Get specific configuration section for specified language."""
        ...


class ProductionConfigProvider:
    """Production configuration provider using real TOML files."""

    def __init__(self):
        # Import at runtime to avoid circular dependencies
        from . import config_loader

        self._config_loader = config_loader

    def load_config(self, config_name: str, use_cache: bool = True) -> dict[str, Any]:
        """Load a configuration file."""
        return self._config_loader.load_config(config_name, use_cache)

    def get_config_section(self, config_name: str, section: str) -> dict[str, Any]:
        """Get a specific section from a config file."""
        return self._config_loader.get_config_section(config_name, section)

    def get_shared_config(self) -> dict[str, Any]:
        """Get shared configuration."""
        return self._config_loader.get_shared_config()

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get configuration for specified language."""
        return self._config_loader.get_language_config(language)

    def get_language_specific_config(
        self, section: str, language: str
    ) -> dict[str, Any]:
        """Get specific configuration section for specified language."""
        return self._config_loader.get_language_specific_config(section, language)


class MockConfigProvider:
    """Mock configuration provider for testing."""

    def __init__(self, mock_configs: dict[str, dict[str, Any]] = None):
        """Initialize with mock configuration data."""
        self.mock_configs = mock_configs or {}
        self.mock_language_configs = {}
        self.mock_shared_config = {}

    def set_config(self, config_name: str, config_data: dict[str, Any]) -> None:
        """Set mock configuration data."""
        self.mock_configs[config_name] = config_data

    def set_language_config(self, language: str, config_data: dict[str, Any]) -> None:
        """Set mock language configuration data."""
        self.mock_language_configs[language] = config_data

    def set_shared_config(self, config_data: dict[str, Any]) -> None:
        """Set mock shared configuration data."""
        self.mock_shared_config = config_data

    def load_config(self, config_name: str, use_cache: bool = True) -> dict[str, Any]:
        """Load mock configuration."""
        if config_name not in self.mock_configs:
            raise KeyError(f"Mock config '{config_name}' not found")
        return self.mock_configs[config_name]

    def get_config_section(self, config_name: str, section: str) -> dict[str, Any]:
        """Get mock configuration section."""
        config = self.load_config(config_name)
        if section not in config:
            raise KeyError(f"Mock section '{section}' not found in '{config_name}'")
        return config[section]

    def get_shared_config(self) -> dict[str, Any]:
        """Get mock shared configuration."""
        return self.mock_shared_config

    def get_language_config(self, language: str) -> dict[str, Any]:
        """Get mock language configuration."""
        if language not in self.mock_language_configs:
            raise KeyError(f"Mock language config '{language}' not found")
        return self.mock_language_configs[language]

    def get_language_specific_config(
        self, section: str, language: str
    ) -> dict[str, Any]:
        """Get mock language-specific configuration section."""
        language_config = self.get_language_config(language)
        if section not in language_config:
            raise KeyError(
                f"Mock section '{section}' not found in language '{language}'"
            )
        return language_config[section]


# Global default provider - can be overridden for testing
_default_provider: ConfigProvider = ProductionConfigProvider()


def set_config_provider(provider: ConfigProvider) -> None:
    """Set the global configuration provider (mainly for testing)."""
    global _default_provider
    _default_provider = provider


def get_config_provider() -> ConfigProvider:
    """Get the current configuration provider."""
    return _default_provider


def reset_config_provider() -> None:
    """Reset to production configuration provider."""
    global _default_provider
    _default_provider = ProductionConfigProvider()
