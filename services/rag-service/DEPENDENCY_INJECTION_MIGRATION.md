# Dependency Injection Migration Strategy

## Current Status: PROOF OF CONCEPT IMPLEMENTED

✅ **RankingConfig.from_config()** - Successfully refactored with dependency injection
✅ **ConfigProvider Protocol** - Production and Mock providers implemented
✅ **Unit Tests** - 4 tests pass, demonstrating isolated testing capability

## Problem Statement

**Before**: 25+ components with hard-coded config dependencies
```python
@classmethod
def from_config(cls) -> "SomeConfig":
    from ..utils.config_loader import get_some_config  # HARD-CODED!
    config = get_some_config()  # FILE SYSTEM DEPENDENCY!
    return cls(...)
```

**Issues**:
- ❌ **No test isolation** - requires actual TOML files
- ❌ **Hard-coded dependencies** - can't mock configuration
- ❌ **File system coupling** - tests depend on disk I/O
- ❌ **No error simulation** - can't test configuration failures

## Solution: Dependency Injection Pattern

**After**: Injectable configuration provider
```python
@classmethod
def from_config(
    cls,
    config_dict: Optional[Dict[str, Any]] = None,
    config_provider: Optional["ConfigProvider"] = None
) -> "SomeConfig":
    if config_dict:
        config = config_dict  # EXISTING API: Direct dictionary
    else:
        provider = config_provider or get_config_provider()  # INJECTED!
        config = provider.get_some_config()  # MOCKABLE!
    return cls(...)
```

**Benefits**:
- ✅ **100% Backward Compatible** - All existing APIs unchanged
- ✅ **Test Isolation** - Mock providers for unit testing
- ✅ **Error Simulation** - Test configuration failure scenarios
- ✅ **Fast Tests** - No file system dependencies

## Migration Priority Matrix

### 🔴 **HIGH PRIORITY** - Core Components (Week 1)
1. **src/generation/ollama_client.py** - `OllamaConfig.from_config()`
2. **src/vectordb/search.py** - `SearchConfig.from_config()`
3. **src/pipeline/config.py** - All 6+ config classes
4. **src/retrieval/reranker.py** - `RerankerConfig.from_config()`

### 🟡 **MEDIUM PRIORITY** - Feature Components (Week 2)
5. **src/vectordb/embeddings.py** - Embedding configuration
6. **src/vectordb/storage.py** - Storage configuration
7. **src/generation/enhanced_prompt_templates.py** - Template loading
8. **src/retrieval/query_processor.py** - Query processing config
9. **src/retrieval/hybrid_retriever.py** - Hybrid retrieval config

### 🟢 **LOW PRIORITY** - Utilities (Week 3)
10. **src/generation/response_parser.py** - Response parsing config
11. **src/preprocessing/** - All preprocessing configs
12. **src/utils/** - Utility configurations

## Migration Steps Per Component

### Step 1: Update Method Signature
```python
# Before
@classmethod
def from_config(cls) -> "ConfigClass":

# After
@classmethod
def from_config(
    cls,
    config_dict: Optional[Dict[str, Any]] = None,
    config_provider: Optional["ConfigProvider"] = None
) -> "ConfigClass":
```

### Step 2: Add Type Import
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.config_protocol import ConfigProvider
```

### Step 3: Replace Hard-coded Import
```python
# Before
from ..utils.config_loader import get_some_config
config = get_some_config()

# After
if config_dict:
    config = config_dict
else:
    provider = config_provider or get_config_provider()
    config = provider.get_some_config()  # or appropriate method
```

### Step 4: Create Unit Tests
```python
def test_from_config_with_mock_provider(self):
    mock_provider = MockConfigProvider()
    mock_provider.set_config("config_name", {...})
    config = ConfigClass.from_config(config_provider=mock_provider)
    # Assert config values
```

## Testing Strategy

### Unit Test Template
```python
import unittest
from src.utils.config_protocol import MockConfigProvider

class TestConfigClassDI(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MockConfigProvider()
        self.mock_config = {...}  # Mock configuration data
        self.mock_provider.set_config("config_name", self.mock_config)

    def test_production_api_unchanged(self):
        """Existing production API works unchanged."""
        config = ConfigClass.from_config()
        # Assert production config loads

    def test_dependency_injection(self):
        """Mock provider enables isolated testing."""
        config = ConfigClass.from_config(config_provider=self.mock_provider)
        # Assert mock config values

    def test_direct_dictionary(self):
        """Direct dictionary API works unchanged."""
        config = ConfigClass.from_config(config_dict={...})
        # Assert direct config values

    def test_error_handling(self):
        """Test configuration error scenarios."""
        empty_provider = MockConfigProvider()
        with self.assertRaises(KeyError):
            ConfigClass.from_config(config_provider=empty_provider)
```

## Implementation Plan

### Phase 1: Core Infrastructure (✅ DONE)
- [x] ConfigProvider Protocol
- [x] ProductionConfigProvider & MockConfigProvider
- [x] RankingConfig proof of concept
- [x] Unit test validation

### Phase 2: High Priority Components (1-2 weeks)
- [ ] OllamaConfig - Generation system
- [ ] SearchConfig - Vector database
- [ ] Pipeline configs - Core orchestration
- [ ] RerankerConfig - Advanced retrieval

### Phase 3: Medium Priority Components (1-2 weeks)
- [ ] Embedding configs
- [ ] Storage configs
- [ ] Template configs
- [ ] Query processor configs

### Phase 4: Testing & Validation (1 week)
- [ ] Comprehensive unit test suite
- [ ] Integration test updates
- [ ] Performance benchmarking
- [ ] Documentation updates

## Expected Benefits

### **Testing Improvements**
- **10x faster unit tests** - No file system dependencies
- **100% test isolation** - Each test independent
- **Error scenario testing** - Simulate configuration failures
- **Parallel test execution** - No shared file state

### **Development Experience**
- **Easier debugging** - Inject specific configs for reproduction
- **Better test coverage** - Test edge cases and error conditions
- **Faster development cycles** - Immediate test feedback
- **Cleaner test code** - No temporary file management

### **System Reliability**
- **Configuration validation** - Test invalid configurations
- **Error handling** - Validate graceful failure modes
- **Backward compatibility** - Existing code unchanged
- **Future flexibility** - Easy to add new config sources

## Next Steps

1. **Choose next component** from high priority list
2. **Apply migration steps** systematically
3. **Create unit tests** for each component
4. **Validate production behavior** unchanged
5. **Repeat** until all components migrated

The dependency injection pattern is now **proven and ready for systematic rollout** across the entire RAG system.
