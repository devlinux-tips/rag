# Missing Production Implementations - Test-Driven Plan

**Current Status:** 2,182/2,209 tests passing (98.8%)
**Recent Changes:**
- ✅ Removed `src/models/` folder (not needed)
- ✅ Removed `src/utils/folder_manager.py` (never used in production)
- ✅ Created simplified `src/utils/multitenant_models.py` (3 dataclasses only)
- ✅ All production code updated to use new imports
- ✅ Removed FolderConfig/FolderStats stubs from tests/conftest.py
- ✅ Deleted obsolete tests/conftest_consolidated*.py files
**Remaining:** 27 test failures - legitimate API mismatches that need fixing

---

## Category 1: Factory Functions (13 failures)

### 1.1 folder_manager_providers.py (5 failures)
**Missing:** `create_file_system_provider()`, `create_test_folder_manager()`

**Error:**
```
ImportError: cannot import name 'create_file_system_provider' from 'src.utils.folder_manager_providers'
```

**Implementation Needed:**
```python
# File: src/utils/folder_manager_providers.py

def create_file_system_provider(mock: bool = False) -> FileSystemProvider:
    """Factory: Create file system provider (real or mock)."""
    if mock:
        from tests.conftest import MockFileSystemProvider
        return MockFileSystemProvider()
    return RealFileSystemProvider()  # Need to implement

def create_test_folder_manager(
    config: FolderConfig | None = None,
    filesystem_provider: FileSystemProvider | None = None,
    logger_provider: LoggerProvider | None = None,
    existing_folders: list[str] | None = None,
    filesystem_failures: list[str] | None = None,
    folder_stats: dict[str, FolderStats] | None = None,
) -> FolderManager:
    """Factory: Create folder manager for testing."""
    # Implementation needed
```

**Dependencies:**
- `RealFileSystemProvider` class (needs implementation)
- `FolderManager` class (exists)
- Test integration

---

### 1.2 language_manager_providers.py (2 failures)
**Missing:** `create_test_language_manager()`

**Error:**
```
AssertionError: <MockConfigProvider> is not an instance of <MockPatternProvider>
```

**Implementation Needed:**
```python
# File: src.utils.language_manager_providers.py

def create_test_language_manager(
    settings: LanguageSettings | None = None,
    pattern_provider: PatternProvider | None = None,
    config_provider: ConfigProvider | None = None,
    custom_patterns: dict | None = None,
) -> LanguageManager:
    """Factory: Create language manager for testing."""
    # Implementation needed
```

**Dependencies:**
- Proper MockPatternProvider usage from conftest
- LanguageManager class (exists)

---

### 1.3 query_processor_providers.py (6 failures)
**Missing:** `create_mock_language_provider()`, `create_test_providers()` improvements

**Errors:**
```
AssertionError: <Mock> is not an instance of <MockLanguageDataProvider>
AttributeError: 'MockConfigProvider' object has no attribute 'language'
```

**Implementation Needed:**
```python
# File: src.retrieval.query_processor_providers.py

def create_mock_language_provider(
    language: str = "hr",
    stop_words: set[str] | None = None,
    **kwargs
) -> MockLanguageDataProvider:
    """Factory: Create mock language data provider."""
    from tests.conftest import MockLanguageDataProvider
    provider = MockLanguageDataProvider(language=language)
    if stop_words:
        provider.set_stop_words(stop_words)
    return provider

def create_test_providers(
    language: str = "hr",
    custom_config: dict | None = None,
    custom_language_data: dict | None = None,
) -> tuple[QueryProcessingConfig, LanguageDataProvider, ConfigProvider]:
    """Factory: Create providers for testing."""
    # Return QueryProcessingConfig (not MockConfigProvider)
    config = QueryProcessingConfig(language=language, **(custom_config or {}))
    language_provider = create_mock_language_provider(language, custom_language_data)
    config_provider = create_mock_config_provider(config)
    return config, language_provider, config_provider
```

**Dependencies:**
- `QueryProcessingConfig` dataclass (exists in src/utils/config_models.py)
- `MockLanguageDataProvider` needs `language` attribute

---

## Category 2: Mock Configuration Issues (10 failures)

### 2.1 MockLoggerProvider.get_messages() / get_all_messages() (4 failures)

**Error:**
```
KeyError: 'error'
AssertionError: {'info': [...], 'debug': [...], 'warning': []} != {'info': [...], 'debug': [...], 'warning': [], 'error': []}
```

**Fix in conftest.py:**
```python
class MockLoggerProvider:
    def get_messages(self, level: str | None = None) -> list[str] | dict[str, list[str]]:
        """Get logged messages."""
        if level is None:
            # Return ALL levels including error
            return {
                "info": self.messages.get("info", []),
                "debug": self.messages.get("debug", []),
                "warning": self.messages.get("warning", []),
                "error": self.messages.get("error", []),
            }
        # Return specific level or raise ValueError for unknown
        if level not in ["info", "debug", "warning", "error"]:
            raise ValueError(f"Unknown log level: {level}")
        return self.messages.get(level, [])

    def get_all_messages(self) -> dict[str, list[str]]:
        """Get all messages by level."""
        return {
            "info": self.messages.get("info", []),
            "debug": self.messages.get("debug", []),
            "warning": self.messages.get("warning", []),
            "error": self.messages.get("error", []),
        }
```

---

### 2.2 Cleaners Config Issues (3 failures)

**Error:**
```
KeyError: 'Mock cleaning config not set'
```

**Fix in conftest.py:**
```python
class MockConfigProvider:
    def get_cleaning_config(self, language: str) -> dict:
        """Get cleaning configuration for language."""
        # Return mock cleaning config with defaults
        return {
            "word_char_pattern": r"[\w\u0400-\u04FF]+",
            "diacritic_map": {"č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "d"},
            "cleaning_prefixes": ["Answer:", "Response:"],
            "locale": "hr_HR.UTF-8",
        }
```

---

### 2.3 Extractor Test Provider Return Order (3 failures)

**Error:**
```
AssertionError: <MockLoggerProvider> is not an instance of <FileSystemProvider>
```

**Issue:** `create_test_providers()` for extractors returns wrong tuple order

**Fix in conftest.py:**
```python
def create_test_providers(...) -> tuple:
    # Extractor mode detection
    if is_extractor_mode:
        # Return: (config_provider, fs_provider, logger_provider)
        return config_provider, fs_provider, logger_provider
    else:
        # Query processing mode
        # Return: (config, language_provider, config_provider)
        return config, language_provider, config_provider
```

---

## Category 3: Production Factory Functions Using unittest.mock.Mock (44 failures)

**Root Cause:** Production factory functions in `*_providers.py` files are using `unittest.mock.Mock` instead of real mocks from conftest.

### 3.1 prompt_templates.py (19 failures)

**Error:**
```
TypeError: object of type 'Mock' has no len()
AssertionError: <Mock name='mock.get_prompt_config()'> is not an instance of <PromptConfig>
```

**File:** `src/generation/prompt_templates.py`

**Bad Pattern:**
```python
from unittest.mock import Mock

def create_mock_config_provider(templates=None, **kwargs):
    mock = Mock()
    mock.get_prompt_config.return_value = Mock()  # ❌ Wrong!
    return mock
```

**Fix:**
```python
def create_mock_config_provider(
    templates: dict | None = None,
    keyword_patterns: dict | None = None,
    **kwargs
) -> ConfigProvider:
    """Factory: Create mock config provider for testing."""
    from tests.conftest import create_mock_config_provider as conftest_factory
    return conftest_factory(templates=templates, keyword_patterns=keyword_patterns, **kwargs)
```

---

### 3.2 response_parser.py (15 failures)

**Error:**
```
ValueError: Prefixes must be list, got <class 'unittest.mock.Mock'>
TypeError: argument of type 'Mock' is not iterable
```

**File:** `src/generation/response_parser.py`

**Fix:**
```python
def create_mock_config_provider(
    no_answer_patterns: list[str] | None = None,
    **kwargs
) -> ConfigProvider:
    """Factory: Create mock config provider."""
    from tests.conftest import create_mock_config_provider as conftest_factory
    return conftest_factory(
        no_answer_patterns=no_answer_patterns or ["I don't know"],
        **kwargs
    )
```

---

### 3.3 ranker_providers.py (3 failures)

**Error:**
```
AssertionError: <Mock> is not <MagicMock name='MockConfigProvider()'>
```

**File:** `src.retrieval.ranker_providers.py`

**Fix:**
```python
def create_mock_config_provider(config: dict | None = None) -> ConfigProvider:
    """Factory: Create mock config provider."""
    from tests.conftest import MockConfigProvider
    return MockConfigProvider(config_dict=config or {})

def create_mock_language_provider(language: str = "hr") -> LanguageProvider:
    """Factory: Create mock language provider."""
    from tests.conftest import MockLanguageProvider
    return MockLanguageProvider(language=language)
```

---

### 3.4 ranker.py (2 failures)

**Error:**
```
TypeError: 'Mock' object is not subscriptable
```

**File:** `src/retrieval/ranker.py`

**Fix:**
```python
def create_mock_ranker(config: dict | None = None) -> Ranker:
    """Factory: Create mock ranker for testing."""
    from tests.conftest import MockConfigProvider, MockLanguageProvider
    config_provider = MockConfigProvider(config_dict=config or {})
    language_provider = MockLanguageProvider()
    return Ranker(config_provider=config_provider, language_provider=language_provider)
```

---

### 3.5 search_providers.py (2 failures)

**Error:**
```
AssertionError: <Mock> is not an instance of <MockConfigProvider>
```

**File:** `src/retrieval/search_providers.py`

**Fix:**
```python
def create_mock_config_provider(config: dict | None = None) -> ConfigProvider:
    """Factory: Create mock config provider."""
    from tests.conftest import MockConfigProvider
    return MockConfigProvider(config_dict=config or {})
```

---

### 3.6 extractors_providers.py (3 failures)

**Already fixed in conftest.py - just need production module to exist**

**File:** `src/preprocessing/extractors_providers.py` (CREATE FILE)

```python
"""Dependency injection providers for document extractors."""

from typing import Protocol


class ConfigProvider(Protocol):
    """Protocol for configuration providers."""
    def get_extraction_config(self) -> dict: ...


class FileSystemProvider(Protocol):
    """Protocol for file system operations."""
    def file_exists(self, path: str) -> bool: ...
    def get_file_size_mb(self, path: str) -> float: ...


class LoggerProvider(Protocol):
    """Protocol for logging."""
    def info(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...


# Real implementations
class RealConfigProvider:
    """Production config provider."""
    def get_extraction_config(self) -> dict:
        from src.utils.config_loader import get_unified_config
        return get_unified_config().get("extraction", {})


class RealFileSystemProvider:
    """Production file system provider."""
    def file_exists(self, path: str) -> bool:
        from pathlib import Path
        return Path(path).exists()

    def get_file_size_mb(self, path: str) -> float:
        from pathlib import Path
        return Path(path).stat().st_size / (1024 * 1024)


class RealLoggerProvider:
    """Production logger provider."""
    def __init__(self, logger_name: str = "extractors"):
        from src.utils.logging_factory import get_system_logger
        self.logger = get_system_logger()

    def info(self, message: str) -> None:
        self.logger.info("extraction", "operation", message)

    def error(self, message: str) -> None:
        self.logger.error("extraction", "operation", message)


# Factory functions
def create_config_provider(mock: bool = False) -> ConfigProvider:
    """Factory: Create config provider."""
    if mock:
        from tests.conftest import MockConfigProvider
        return MockConfigProvider()
    return RealConfigProvider()


def create_file_system_provider(
    mock: bool = False,
    files: dict[str, bytes] | None = None
) -> FileSystemProvider:
    """Factory: Create file system provider."""
    if mock:
        from tests.conftest import create_file_system_provider as conftest_factory
        return conftest_factory(files=files)
    return RealFileSystemProvider()


def create_logger_provider(
    logger_name: str | None = None,
    mock: bool = False
) -> LoggerProvider:
    """Factory: Create logger provider."""
    if mock:
        from tests.conftest import MockLoggerProvider
        return MockLoggerProvider()
    return RealLoggerProvider(logger_name or "extractors")


def create_providers(mock: bool = False) -> tuple[ConfigProvider, FileSystemProvider, LoggerProvider]:
    """Factory: Create all providers."""
    return (
        create_config_provider(mock=mock),
        create_file_system_provider(mock=mock),
        create_logger_provider(mock=mock),
    )
```

---

## Implementation Order (Test-Driven)

### Phase 1: Quick Wins (Fix conftest.py - 10 failures)
1. ✅ Fix `MockLoggerProvider.get_messages()` and `get_all_messages()` - **4 failures**
2. ✅ Fix `MockConfigProvider.get_cleaning_config()` - **3 failures**
3. ✅ Fix `create_test_providers()` tuple order for extractors - **3 failures**

**Test after Phase 1:** Should be ~2,181/2,194 (99.4%)

---

### Phase 2: Production Factory Functions (Replace unittest.mock.Mock - 44 failures)
4. ✅ Create `src/preprocessing/extractors_providers.py` - **3 failures**
5. ✅ Fix `src/generation/response_parser.py` factories - **15 failures**
6. ✅ Fix `src/generation/prompt_templates.py` factories - **19 failures**
7. ✅ Fix `src/retrieval/search_providers.py` factories - **2 failures**
8. ✅ Fix `src/retrieval/ranker_providers.py` factories - **3 failures**
9. ✅ Fix `src/retrieval/ranker.py` factories - **2 failures**

**Test after Phase 2:** Should be ~2,191/2,194 (99.9%)

---

### Phase 3: Missing Factory Implementations (13 failures)
10. ✅ Implement `folder_manager_providers.py` factories - **5 failures**
11. ✅ Implement `language_manager_providers.py` factories - **2 failures**
12. ✅ Implement `query_processor_providers.py` factories - **6 failures**

**Test after Phase 3:** Should be **2,194/2,194 (100%)**

---

## Success Criteria

- ✅ All 2,194 tests passing
- ✅ Zero `unittest.mock.Mock` usage in production code
- ✅ All factory functions delegate to conftest mocks when `mock=True`
- ✅ All production providers implement proper protocols
- ✅ Clean separation: Tests use conftest, Production uses real implementations

---

## Next Steps

1. Start with **Phase 1** (conftest.py fixes)
2. Run tests after each fix
3. Validate improvement before moving to next item
4. Document any issues encountered
