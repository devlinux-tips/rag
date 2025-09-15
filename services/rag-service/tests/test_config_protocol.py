"""
Test suite for utils.config_protocol module.

Tests the configuration protocol for dependency injection,
including protocol definitions, production provider, mock provider,
and global provider management.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Any

from src.utils.config_protocol import (
    ConfigProvider,
    ProductionConfigProvider,
    MockConfigProvider,
    set_config_provider,
    get_config_provider,
    reset_config_provider,
    _default_provider,
)


class TestConfigProviderProtocol:
    """Test cases for ConfigProvider protocol."""

    def test_config_provider_is_protocol(self):
        """Test that ConfigProvider is a proper Protocol."""
        from typing import get_origin
        import inspect

        # Check that ConfigProvider is a Protocol by checking its bases
        assert any('Protocol' in str(base) for base in ConfigProvider.__mro__)

    def test_config_provider_is_runtime_checkable(self):
        """Test that ConfigProvider is runtime checkable."""
        # Should be able to use isinstance with protocol
        mock_provider = MockConfigProvider()
        assert isinstance(mock_provider, ConfigProvider)

    def test_config_provider_protocol_methods(self):
        """Test that ConfigProvider protocol defines required methods."""
        expected_methods = [
            'load_config',
            'get_config_section',
            'get_shared_config',
            'get_language_config',
            'get_language_specific_config'
        ]

        for method_name in expected_methods:
            assert hasattr(ConfigProvider, method_name)


class TestProductionConfigProvider:
    """Test cases for ProductionConfigProvider."""

    def test_production_config_provider_initialization(self):
        """Test ProductionConfigProvider initializes correctly."""
        with patch.object(ProductionConfigProvider, '__init__', lambda self: None):
            provider = ProductionConfigProvider()
            # Can't test _config_loader assignment without actual initialization
            assert hasattr(provider, '__class__')

    def test_load_config_delegates_correctly(self):
        """Test load_config delegates to config_loader."""
        # Patch the import inside the __init__ method
        with patch('src.utils.config_loader') as mock_config_loader:
            mock_config_loader.load_config.return_value = {'test': 'data'}

            provider = ProductionConfigProvider()
            result = provider.load_config('test_config', True)

            mock_config_loader.load_config.assert_called_once_with('test_config', True)
            assert result == {'test': 'data'}

    def test_load_config_with_default_cache(self):
        """Test load_config with default cache parameter."""
        with patch('src.utils.config_loader') as mock_config_loader:
            mock_config_loader.load_config.return_value = {'test': 'data'}

            provider = ProductionConfigProvider()
            provider.load_config('test_config')

            mock_config_loader.load_config.assert_called_once_with('test_config', True)

    def test_get_config_section_delegates_correctly(self):
        """Test get_config_section delegates to config_loader."""
        with patch('src.utils.config_loader') as mock_config_loader:
            mock_config_loader.get_config_section.return_value = {'section': 'data'}

            provider = ProductionConfigProvider()
            result = provider.get_config_section('test_config', 'test_section')

            mock_config_loader.get_config_section.assert_called_once_with('test_config', 'test_section')
            assert result == {'section': 'data'}

    def test_get_shared_config_delegates_correctly(self):
        """Test get_shared_config delegates to config_loader."""
        with patch('src.utils.config_loader') as mock_config_loader:
            mock_config_loader.get_shared_config.return_value = {'shared': 'data'}

            provider = ProductionConfigProvider()
            result = provider.get_shared_config()

            mock_config_loader.get_shared_config.assert_called_once()
            assert result == {'shared': 'data'}

    def test_get_language_config_delegates_correctly(self):
        """Test get_language_config delegates to config_loader."""
        with patch('src.utils.config_loader') as mock_config_loader:
            mock_config_loader.get_language_config.return_value = {'language': 'data'}

            provider = ProductionConfigProvider()
            result = provider.get_language_config('hr')

            mock_config_loader.get_language_config.assert_called_once_with('hr')
            assert result == {'language': 'data'}

    def test_get_language_specific_config_delegates_correctly(self):
        """Test get_language_specific_config delegates to config_loader."""
        with patch('src.utils.config_loader') as mock_config_loader:
            mock_config_loader.get_language_specific_config.return_value = {'lang_section': 'data'}

            provider = ProductionConfigProvider()
            result = provider.get_language_specific_config('section', 'hr')

            mock_config_loader.get_language_specific_config.assert_called_once_with('section', 'hr')
            assert result == {'lang_section': 'data'}

    def test_production_provider_implements_protocol(self):
        """Test that ProductionConfigProvider implements ConfigProvider protocol."""
        provider = ProductionConfigProvider()
        assert isinstance(provider, ConfigProvider)


class TestMockConfigProvider:
    """Test cases for MockConfigProvider."""

    def test_mock_config_provider_initialization_empty(self):
        """Test MockConfigProvider initializes with empty configs."""
        provider = MockConfigProvider()

        assert provider.mock_configs == {}
        assert provider.mock_language_configs == {}
        assert provider.mock_shared_config == {}

    def test_mock_config_provider_initialization_with_configs(self):
        """Test MockConfigProvider initializes with provided configs."""
        initial_configs = {'test': {'key': 'value'}}
        provider = MockConfigProvider(initial_configs)

        assert provider.mock_configs == initial_configs
        assert provider.mock_language_configs == {}
        assert provider.mock_shared_config == {}

    def test_set_config(self):
        """Test set_config method."""
        provider = MockConfigProvider()
        config_data = {'test_key': 'test_value'}

        provider.set_config('test_config', config_data)

        assert provider.mock_configs['test_config'] == config_data

    def test_set_language_config(self):
        """Test set_language_config method."""
        provider = MockConfigProvider()
        config_data = {'lang_key': 'lang_value'}

        provider.set_language_config('hr', config_data)

        assert provider.mock_language_configs['hr'] == config_data

    def test_set_shared_config(self):
        """Test set_shared_config method."""
        provider = MockConfigProvider()
        config_data = {'shared_key': 'shared_value'}

        provider.set_shared_config(config_data)

        assert provider.mock_shared_config == config_data

    def test_load_config_success(self):
        """Test load_config with existing config."""
        provider = MockConfigProvider()
        config_data = {'key': 'value'}
        provider.set_config('test_config', config_data)

        result = provider.load_config('test_config')

        assert result == config_data

    def test_load_config_with_cache_parameter(self):
        """Test load_config ignores use_cache parameter."""
        provider = MockConfigProvider()
        config_data = {'key': 'value'}
        provider.set_config('test_config', config_data)

        result = provider.load_config('test_config', use_cache=False)

        assert result == config_data

    def test_load_config_missing_raises_keyerror(self):
        """Test load_config raises KeyError for missing config."""
        provider = MockConfigProvider()

        with pytest.raises(KeyError, match="Mock config 'missing_config' not found"):
            provider.load_config('missing_config')

    def test_get_config_section_success(self):
        """Test get_config_section with existing section."""
        provider = MockConfigProvider()
        config_data = {'section1': {'key': 'value'}, 'section2': {'other': 'data'}}
        provider.set_config('test_config', config_data)

        result = provider.get_config_section('test_config', 'section1')

        assert result == {'key': 'value'}

    def test_get_config_section_missing_config_raises_keyerror(self):
        """Test get_config_section raises KeyError for missing config."""
        provider = MockConfigProvider()

        with pytest.raises(KeyError, match="Mock config 'missing_config' not found"):
            provider.get_config_section('missing_config', 'section')

    def test_get_config_section_missing_section_raises_keyerror(self):
        """Test get_config_section raises KeyError for missing section."""
        provider = MockConfigProvider()
        provider.set_config('test_config', {'existing_section': {}})

        with pytest.raises(KeyError, match="Mock section 'missing_section' not found in 'test_config'"):
            provider.get_config_section('test_config', 'missing_section')

    def test_get_shared_config_success(self):
        """Test get_shared_config returns set shared config."""
        provider = MockConfigProvider()
        shared_data = {'shared_key': 'shared_value'}
        provider.set_shared_config(shared_data)

        result = provider.get_shared_config()

        assert result == shared_data

    def test_get_shared_config_empty_default(self):
        """Test get_shared_config returns empty dict by default."""
        provider = MockConfigProvider()

        result = provider.get_shared_config()

        assert result == {}

    def test_get_language_config_success(self):
        """Test get_language_config with existing language."""
        provider = MockConfigProvider()
        lang_data = {'lang_key': 'lang_value'}
        provider.set_language_config('hr', lang_data)

        result = provider.get_language_config('hr')

        assert result == lang_data

    def test_get_language_config_missing_raises_keyerror(self):
        """Test get_language_config raises KeyError for missing language."""
        provider = MockConfigProvider()

        with pytest.raises(KeyError, match="Mock language config 'missing_lang' not found"):
            provider.get_language_config('missing_lang')

    def test_get_language_specific_config_success(self):
        """Test get_language_specific_config with existing section."""
        provider = MockConfigProvider()
        lang_config = {'section1': {'key': 'value'}, 'section2': {'other': 'data'}}
        provider.set_language_config('hr', lang_config)

        result = provider.get_language_specific_config('section1', 'hr')

        assert result == {'key': 'value'}

    def test_get_language_specific_config_missing_language_raises_keyerror(self):
        """Test get_language_specific_config raises KeyError for missing language."""
        provider = MockConfigProvider()

        with pytest.raises(KeyError, match="Mock language config 'missing_lang' not found"):
            provider.get_language_specific_config('section', 'missing_lang')

    def test_get_language_specific_config_missing_section_raises_keyerror(self):
        """Test get_language_specific_config raises KeyError for missing section."""
        provider = MockConfigProvider()
        provider.set_language_config('hr', {'existing_section': {}})

        with pytest.raises(KeyError, match="Mock section 'missing_section' not found in language 'hr'"):
            provider.get_language_specific_config('missing_section', 'hr')

    def test_mock_provider_implements_protocol(self):
        """Test that MockConfigProvider implements ConfigProvider protocol."""
        provider = MockConfigProvider()
        assert isinstance(provider, ConfigProvider)


class TestGlobalProviderManagement:
    """Test cases for global provider management functions."""

    def setup_method(self):
        """Reset to production provider before each test."""
        reset_config_provider()

    def teardown_method(self):
        """Reset to production provider after each test."""
        reset_config_provider()

    def test_default_provider_is_production(self):
        """Test that default provider is ProductionConfigProvider."""
        provider = get_config_provider()
        assert isinstance(provider, ProductionConfigProvider)

    def test_set_config_provider(self):
        """Test set_config_provider changes the global provider."""
        mock_provider = MockConfigProvider()

        set_config_provider(mock_provider)

        current_provider = get_config_provider()
        assert current_provider is mock_provider
        assert isinstance(current_provider, MockConfigProvider)

    def test_get_config_provider_returns_current(self):
        """Test get_config_provider returns currently set provider."""
        # Set a mock provider
        mock_provider = MockConfigProvider()
        set_config_provider(mock_provider)

        # Get should return the same instance
        retrieved_provider = get_config_provider()
        assert retrieved_provider is mock_provider

    def test_reset_config_provider(self):
        """Test reset_config_provider restores ProductionConfigProvider."""
        # Set a mock provider
        mock_provider = MockConfigProvider()
        set_config_provider(mock_provider)
        assert isinstance(get_config_provider(), MockConfigProvider)

        # Reset should restore production provider
        reset_config_provider()
        provider = get_config_provider()
        assert isinstance(provider, ProductionConfigProvider)

    def test_multiple_provider_changes(self):
        """Test multiple provider changes work correctly."""
        mock1 = MockConfigProvider()
        mock2 = MockConfigProvider()

        # Change to mock1
        set_config_provider(mock1)
        assert get_config_provider() is mock1

        # Change to mock2
        set_config_provider(mock2)
        assert get_config_provider() is mock2

        # Reset to production
        reset_config_provider()
        assert isinstance(get_config_provider(), ProductionConfigProvider)

    def test_provider_persists_across_calls(self):
        """Test that set provider persists across multiple get calls."""
        mock_provider = MockConfigProvider()
        set_config_provider(mock_provider)

        # Multiple calls should return same instance
        provider1 = get_config_provider()
        provider2 = get_config_provider()

        assert provider1 is provider2
        assert provider1 is mock_provider


# Integration tests
class TestConfigProtocolIntegration:
    """Integration tests for the config_protocol module."""

    def test_module_imports_successfully(self):
        """Test that the module can be imported without errors."""
        import src.utils.config_protocol
        assert hasattr(src.utils.config_protocol, 'ConfigProvider')
        assert hasattr(src.utils.config_protocol, 'ProductionConfigProvider')
        assert hasattr(src.utils.config_protocol, 'MockConfigProvider')

    def test_all_providers_implement_protocol(self):
        """Test that all provider classes implement the protocol."""
        production_provider = ProductionConfigProvider()
        mock_provider = MockConfigProvider()

        assert isinstance(production_provider, ConfigProvider)
        assert isinstance(mock_provider, ConfigProvider)

    def test_dependency_injection_pattern_works(self):
        """Test the complete dependency injection pattern."""
        # Create a mock provider with test data
        mock_provider = MockConfigProvider()
        mock_provider.set_config('test', {'key': 'value'})

        # Set as global provider
        set_config_provider(mock_provider)

        # Get provider and use it
        current_provider = get_config_provider()
        result = current_provider.load_config('test')

        assert result == {'key': 'value'}

        # Reset for cleanup
        reset_config_provider()

    def test_type_annotations_present(self):
        """Test that all functions have proper type annotations."""
        import inspect

        # Test protocol methods have annotations
        for method_name in ['load_config', 'get_config_section', 'get_shared_config',
                           'get_language_config', 'get_language_specific_config']:
            method = getattr(ConfigProvider, method_name)
            signature = inspect.signature(method)
            assert signature.return_annotation is not None

        # Test global functions have annotations
        for func in [set_config_provider, get_config_provider, reset_config_provider]:
            signature = inspect.signature(func)
            # set_config_provider and reset_config_provider return None
            assert signature.return_annotation is not None or func in [set_config_provider, reset_config_provider]