"""
Tests for Weaviate configuration module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.vectordb.weaviate_config import (
    WeaviateConfiguration,
    WeaviateConnectionConfig,
    WeaviateHNSWConfig,
    WeaviateCompressionConfig,
    WeaviateGeneralConfig,
    WeaviateBackupConfig,
    create_weaviate_configuration,
)


class TestWeaviateConfig:
    """Test Weaviate configuration creation and validation."""

    def test_connection_config_creation(self):
        """Test creating a WeaviateConnectionConfig instance."""
        config = WeaviateConnectionConfig(
            host="localhost",
            port=8080,
            grpc_port=50051,
            scheme="http",
            timeout=30.0,
            startup_period=5,
            additional_headers={"Authorization": "Bearer test-key"}
        )

        assert config.host == "localhost"
        assert config.port == 8080
        assert config.grpc_port == 50051
        assert config.scheme == "http"
        assert config.timeout == 30.0
        assert config.startup_period == 5
        assert config.additional_headers == {"Authorization": "Bearer test-key"}

    def test_connection_config_defaults(self):
        """Test WeaviateConnectionConfig with default values."""
        config = WeaviateConnectionConfig()

        assert config.host == "localhost"
        assert config.port == 8080
        assert config.grpc_port == 50051
        assert config.scheme == "http"
        assert config.timeout == 30.0
        assert config.startup_period == 5
        assert config.additional_headers == {}

    def test_hnsw_config_creation(self):
        """Test creating an WeaviateHNSWConfig instance."""
        config = WeaviateHNSWConfig(
            type="hnsw",
            ef_construction=200,
            ef=100,
            max_connections=16,
            ef_dynamic=150,
            cleanup_interval_seconds=300,
            vector_cache_max_objects=1000000
        )

        assert config.type == "hnsw"
        assert config.ef_construction == 200
        assert config.ef == 100
        assert config.max_connections == 16
        assert config.ef_dynamic == 150
        assert config.cleanup_interval_seconds == 300
        assert config.vector_cache_max_objects == 1000000

    def test_hnsw_config_defaults(self):
        """Test WeaviateHNSWConfig with default values."""
        config = WeaviateHNSWConfig()

        assert config.type == "hnsw"
        assert config.ef_construction == 128
        assert config.ef == -1
        assert config.max_connections == 32
        assert config.ef_dynamic == 100
        assert config.cleanup_interval_seconds == 300
        assert config.vector_cache_max_objects == 1000000

    def test_compression_config_creation(self):
        """Test creating a WeaviateCompressionConfig instance."""
        config = WeaviateCompressionConfig(
            enabled=True,
            type="sq",
            rescore_limit=200,
            cache=True
        )

        assert config.enabled is True
        assert config.type == "sq"
        assert config.rescore_limit == 200
        assert config.cache is True

    def test_compression_config_defaults(self):
        """Test WeaviateCompressionConfig with default values."""
        config = WeaviateCompressionConfig()

        assert config.enabled is False
        assert config.type == "pq"
        assert config.rescore_limit == 100
        assert config.training_limit == 100000
        assert config.cache is False

    def test_compression_config_to_weaviate(self):
        """Test WeaviateCompressionConfig.to_weaviate_config method."""
        # Test with compression disabled
        config = WeaviateCompressionConfig(enabled=False)
        assert config.to_weaviate_config() is None

        # Test with SQ compression
        config = WeaviateCompressionConfig(enabled=True, type="sq")
        weaviate_config = config.to_weaviate_config()
        assert weaviate_config is not None
        assert weaviate_config["vectorCompressionType"] == "SQ"
        assert "rescoreLimit" in weaviate_config
        assert "trainingLimit" in weaviate_config

    def test_general_config_creation(self):
        """Test creating a WeaviateGeneralConfig instance."""
        config = WeaviateGeneralConfig(
            vectorizer="text2vec-transformers",
            collection_name_template="{tenant}_{user}_{language}_collection",
            distance_metric="l2",
            batch_size=200,
            timeout=60.0,
            max_retries=5,
            retry_delay=2.0
        )

        assert config.vectorizer == "text2vec-transformers"
        assert config.collection_name_template == "{tenant}_{user}_{language}_collection"
        assert config.distance_metric == "l2"
        assert config.batch_size == 200
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0

    def test_general_config_defaults(self):
        """Test WeaviateGeneralConfig with default values."""
        config = WeaviateGeneralConfig()

        assert config.vectorizer == "none"
        assert config.collection_name_template == "{tenant}_{language}_collection"
        assert config.distance_metric == "cosine"
        assert config.batch_size == 100
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_weaviate_configuration_creation(self):
        """Test creating a WeaviateConfiguration instance."""
        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(
                host="localhost",
                port=8080,
                grpc_port=50051,
                scheme="http",
                timeout=30.0,
                startup_period=5
            ),
            hnsw=WeaviateHNSWConfig(
                ef_construction=200,
                ef=100,
                max_connections=16
            ),
            compression=WeaviateCompressionConfig(
                enabled=False,
                type="none"
            ),
            general=WeaviateGeneralConfig(
                batch_size=100,
                timeout=30.0,
                max_retries=3
            ),
            backup=WeaviateBackupConfig(
                enabled=False
            )
        )

        assert config.connection.url == "http://localhost:8080"
        assert config.connection.port == 8080
        assert config.hnsw.ef == 100
        assert config.general.batch_size == 100

    def test_create_weaviate_configuration_from_dict(self):
        """Test creating WeaviateConfiguration from dictionary."""
        config_dict = {
            "vectordb": {
                "provider": "weaviate",
                "collection_name_template": "{tenant}_{language}_collection",
                "distance_metric": "cosine",
                "batch_size": 200,
                "timeout": 30.0,
                "max_retries": 3,
                "retry_delay": 1.0,
                "weaviate": {
                    "host": "localhost",
                    "port": 8080,
                    "grpc_port": 50051,
                    "scheme": "http",
                    "timeout": 30.0,
                    "startup_period": 5,
                    "vectorizer": "none",
                    "index": {
                        "type": "hnsw",
                        "ef_construction": 250,
                        "ef": 150,
                        "max_connections": 32,
                        "ef_dynamic": 100,
                        "cleanup_interval_seconds": 300,
                        "vector_cache_max_objects": 1000000
                    },
                    "compression": {
                        "enabled": True,
                        "type": "sq",
                        "rescore_limit": 100,
                        "training_limit": 100000,
                        "cache": False
                    },
                    "backup": {
                        "enabled": False,
                        "backend": "filesystem",
                        "backup_id": "default",
                        "include_meta": True
                    }
                }
            }
        }

        config = create_weaviate_configuration(config_dict)

        assert config.connection.url == "http://localhost:8080"
        assert config.hnsw.ef_construction == 250
        assert config.hnsw.ef == 150
        assert config.compression.enabled is True
        assert config.compression.type == "sq"
        assert config.general.batch_size == 200

    def test_create_weaviate_configuration_with_defaults(self):
        """Test creating WeaviateConfiguration with minimal config."""
        # This test can't work with the fail-fast pattern
        # since from_config methods require all keys to be present
        # We'll need to provide a complete config
        config_dict = {
            "vectordb": {
                "provider": "weaviate",
                "collection_name_template": "{tenant}_{language}_collection",
                "distance_metric": "cosine",
                "batch_size": 100,
                "timeout": 30.0,
                "max_retries": 3,
                "retry_delay": 1.0,
                "weaviate": {
                    "host": "custom",
                    "port": 9999,
                    "grpc_port": 50051,
                    "scheme": "http",
                    "timeout": 30.0,
                    "startup_period": 5,
                    "vectorizer": "none",
                    "index": {
                        "type": "hnsw",
                        "ef_construction": 200,
                        "ef": -1,
                        "max_connections": 32,
                        "ef_dynamic": 100,
                        "cleanup_interval_seconds": 300,
                        "vector_cache_max_objects": 1000000
                    },
                    "compression": {
                        "enabled": False,
                        "type": "pq",
                        "rescore_limit": 100,
                        "training_limit": 100000,
                        "cache": False
                    },
                    "backup": {
                        "enabled": False,
                        "backend": "filesystem",
                        "backup_id": "default",
                        "include_meta": True
                    }
                }
            }
        }

        config = create_weaviate_configuration(config_dict)

        # Should use provided URL
        assert config.connection.url == "http://custom:9999"

        # Check other values
        assert config.hnsw.ef_construction == 200
        assert config.compression.enabled is False
        assert config.general.batch_size == 100

    def test_create_weaviate_configuration_empty_dict(self):
        """Test creating WeaviateConfiguration with empty dict."""
        # With fail-fast pattern, this should raise an error
        config_dict = {}

        with pytest.raises(KeyError):
            config = create_weaviate_configuration(config_dict)


class TestWeaviateConfigIntegration:
    """Integration tests for Weaviate configuration."""

    def test_full_config_creation_flow(self):
        """Test complete configuration creation flow."""
        # Create config from dict
        config_dict = {
            "vectordb": {
                "provider": "weaviate",
                "collection_name_template": "{tenant}_{language}_collection",
                "distance_metric": "cosine",
                "batch_size": 500,
                "timeout": 30.0,
                "max_retries": 3,
                "retry_delay": 1.0,
                "weaviate": {
                    "host": "weaviate",
                    "port": 8080,
                    "grpc_port": 50051,
                    "scheme": "http",
                    "timeout": 30.0,
                    "startup_period": 5,
                    "vectorizer": "none",
                    "index": {
                        "type": "hnsw",
                        "ef_construction": 300,
                        "ef": 200,
                        "max_connections": 32,
                        "ef_dynamic": 100,
                        "cleanup_interval_seconds": 300,
                        "vector_cache_max_objects": 1000000
                    },
                    "compression": {
                        "enabled": True,
                        "type": "sq",
                        "rescore_limit": 100,
                        "training_limit": 100000,
                        "cache": False
                    },
                    "backup": {
                        "enabled": False,
                        "backend": "filesystem",
                        "backup_id": "default",
                        "include_meta": True
                    }
                }
            }
        }

        config = create_weaviate_configuration(config_dict)

        # Verify all settings were applied
        assert config.connection.url == "http://weaviate:8080"
        assert config.connection.host == "weaviate"
        assert config.hnsw.ef_construction == 300
        assert config.hnsw.ef == 200
        assert config.compression.enabled is True
        assert config.compression.type == "sq"
        assert config.general.batch_size == 500

    def test_config_with_auth(self):
        """Test configuration with authentication."""
        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(
                host="weaviate.example.com",
                port=8080,
                grpc_port=50051,
                scheme="http"
            ),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(
                batch_size=50,
                max_retries=5
            ),
            backup=WeaviateBackupConfig()
        )

        assert config.connection.host == "weaviate.example.com"
        assert config.general.batch_size == 50

    def test_config_serialization(self):
        """Test config can be used with expected attributes."""
        config = WeaviateConfiguration(
            connection=WeaviateConnectionConfig(
                host="localhost",
                port=8080,
                scheme="http"
            ),
            hnsw=WeaviateHNSWConfig(),
            compression=WeaviateCompressionConfig(),
            general=WeaviateGeneralConfig(),
            backup=WeaviateBackupConfig()
        )

        # Test that config has expected attributes
        assert config.connection.url == "http://localhost:8080"
        assert isinstance(config.hnsw, WeaviateHNSWConfig)
        assert isinstance(config.compression, WeaviateCompressionConfig)
        assert isinstance(config.general, WeaviateGeneralConfig)

    def test_compression_variants(self):
        """Test different compression configurations."""
        # No compression
        config1 = WeaviateCompressionConfig(enabled=False)
        assert config1.to_weaviate_config() is None

        # SQ compression
        config2 = WeaviateCompressionConfig(enabled=True, type="sq", rescore_limit=100)
        result = config2.to_weaviate_config()
        assert result is not None
        assert result["vectorCompressionType"] == "SQ"
        assert result["rescoreLimit"] == 100

        # PQ compression
        config3 = WeaviateCompressionConfig(enabled=True, type="pq")
        result = config3.to_weaviate_config()
        assert result is not None
        assert result["vectorCompressionType"] == "PQ"