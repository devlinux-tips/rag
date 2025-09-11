# CONFIG ARCHITECTURE SPECIFICATION
**Version**: 1.0
**Status**: IMMUTABLE STANDARD
**Purpose**: Prevent refactoring cycles, ensure consistency

---

## 🚫 **ANTI-REFACTORING COVENANT**

**CRITICAL**: This document defines the **FINAL** configuration architecture. Any future refactoring that deviates from these principles must be **explicitly justified** and approved. No more vicious cycles.

---

## 🎯 **CORE PRINCIPLES**

### 1. **TWO-PHASE CONFIG SYSTEM**
```
Phase 1: ConfigValidator → Startup validation (fail-fast)
Phase 2: Clean DI → Direct config access (no .get() fallbacks)
```

### 2. **FAIL-FAST PHILOSOPHY**
- ✅ **Explicit validation** at startup
- ❌ **No silent fallbacks** with `.get()`
- ❌ **No magic defaults** scattered in code
- ✅ **Clear error messages** with exact config file + key path

### 3. **BASE INTERFACE + LANGUAGE EXTENSIONS**
- **Base Interface**: Common structure across all languages
- **Language Extensions**: Language-specific additions when needed
- **Current Reality**: hr.toml and en.toml have identical structure

---

## 📁 **CONFIG FILE STRUCTURE**

### **Validated Structure** (as of 2025-09-10)
```
config/
├── config.toml          # Main shared configuration
├── hr.toml              # Croatian language config
└── en.toml              # English language config
```

### **Base Interface Schema**
All language config files MUST have these sections:
```toml
[language]                    # Language metadata
[shared]                      # Shared constants
[shared.question_patterns]    # Question type patterns
[shared.stopwords]           # Language stopwords
[categorization]             # Category indicators
[patterns]                   # Regex patterns
[text_processing]            # Text preprocessing
[chunking]                   # Document chunking
[embeddings]                 # Embedding settings
[vectordb]                   # Vector database
[retrieval]                  # Retrieval configuration
[generation]                 # Text generation
[prompts]                    # Prompt templates
[confidence]                 # Confidence calculation
[response_parsing]           # Response processing
[pipeline]                   # Pipeline settings
```

**RULE**: New sections can be added, but existing sections cannot be removed or renamed without system-wide impact analysis.

---

## 🏗️ **IMPLEMENTATION ARCHITECTURE**

### **Phase 1: ConfigValidator**

```python
@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    missing_keys: List[str]
    invalid_types: List[str]
    config_file: str

class ConfigValidator:
    """
    Two-phase configuration validator
    Phase 1: Validates ALL required keys exist
    Phase 2: Enables fail-fast DI component creation
    """

    # Base interface schema - MUST be kept in sync with actual config files
    BASE_INTERFACE_SCHEMA = {
        # Main config.toml required keys
        "shared.cache_dir": str,
        "shared.default_timeout": (int, float),
        "languages.supported": list,
        "languages.default": str,
        "embeddings.model_name": str,
        "query_processing.expand_synonyms": bool,
        "query_processing.normalize_case": bool,
        "query_processing.remove_stopwords": bool,
        "query_processing.min_query_length": int,
        "query_processing.max_expanded_terms": int,
        "query_processing.enable_spell_check": bool,
        "query_processing.min_word_length": int,

        # Language config required keys (applies to hr.toml, en.toml, etc.)
        "language.code": str,
        "language.name": str,
        "shared.stopwords.words": list,
        "shared.question_patterns.factual": list,
        "shared.question_patterns.explanatory": list,
        "categorization.cultural_indicators": list,
        "patterns.cultural": list,
        "text_processing.normalize_case": bool,
        "chunking.sentence_endings": list,
        "prompts.system_base": str,
        "confidence.error_phrases": list,
        "pipeline.enable_morphological_expansion": bool,
    }

    @classmethod
    def validate_startup_config(cls,
                              main_config: dict,
                              language_configs: Dict[str, dict]) -> None:
        """
        PHASE 1: Validate ALL configuration at system startup
        Fails fast with exact error location if ANY required key missing
        """
        # Validate main config
        main_result = cls._validate_config_section(
            config=main_config,
            schema=cls._get_main_config_schema(),
            config_file="config/config.toml"
        )

        if not main_result.is_valid:
            raise ConfigurationError(
                f"Invalid main configuration in {main_result.config_file}:\n"
                f"Missing keys: {main_result.missing_keys}\n"
                f"Invalid types: {main_result.invalid_types}"
            )

        # Validate each language config
        for lang_code, lang_config in language_configs.items():
            lang_result = cls._validate_config_section(
                config=lang_config,
                schema=cls._get_language_config_schema(),
                config_file=f"config/{lang_code}.toml"
            )

            if not lang_result.is_valid:
                raise ConfigurationError(
                    f"Invalid language configuration in {lang_result.config_file}:\n"
                    f"Missing keys: {lang_result.missing_keys}\n"
                    f"Invalid types: {lang_result.invalid_types}"
                )

    @classmethod
    def _validate_config_section(cls,
                                config: dict,
                                schema: Dict[str, type],
                                config_file: str) -> ConfigValidationResult:
        """Validate individual config section against schema"""
        missing_keys = []
        invalid_types = []

        for key_path, expected_type in schema.items():
            try:
                # Navigate nested dictionary structure
                current = config
                keys = key_path.split('.')
                for key in keys:
                    current = current[key]  # Direct access - no .get()

                # Type validation
                if not isinstance(current, expected_type):
                    invalid_types.append(f"{key_path}: expected {expected_type}, got {type(current)}")

            except KeyError:
                missing_keys.append(f"{key_path} in {config_file}")

        return ConfigValidationResult(
            is_valid=(len(missing_keys) == 0 and len(invalid_types) == 0),
            missing_keys=missing_keys,
            invalid_types=invalid_types,
            config_file=config_file
        )
```

### **Phase 2: Clean DI Components**

```python
@dataclass
class QueryProcessingConfig:
    """
    Clean configuration dataclass
    ConfigValidator guarantees all keys exist - no .get() fallbacks needed
    """
    language: str
    expand_synonyms: bool
    normalize_case: bool
    remove_stopwords: bool
    min_query_length: int
    max_expanded_terms: int
    enable_spell_check: bool
    min_word_length: int

    @classmethod
    def from_validated_config(cls,
                            main_config: dict,
                            language: str) -> "QueryProcessingConfig":
        """
        PHASE 2: Create config from validated configuration
        Uses direct dictionary access - ConfigValidator guarantees existence
        """
        query_config = main_config["query_processing"]  # Direct access - guaranteed to exist

        return cls(
            language=language,
            expand_synonyms=query_config["expand_synonyms"],      # Direct - no .get()
            normalize_case=query_config["normalize_case"],        # Direct - no .get()
            remove_stopwords=query_config["remove_stopwords"],   # Direct - no .get()
            min_query_length=query_config["min_query_length"],   # Direct - no .get()
            max_expanded_terms=query_config["max_expanded_terms"], # Direct - no .get()
            enable_spell_check=query_config["enable_spell_check"], # Direct - no .get()
            min_word_length=query_config["min_word_length"],     # Direct - no .get()
        )

class MultilingualQueryProcessor:
    """
    Clean DI implementation - no fallbacks, explicit dependencies
    """
    def __init__(self,
                 config: QueryProcessingConfig,
                 language_data_provider: LanguageDataProvider):
        # All dependencies guaranteed valid by ConfigValidator
        self.config = config
        self.language_data_provider = language_data_provider

    def _get_stop_words(self) -> Set[str]:
        """
        NO fallbacks - provider guaranteed to have data by ConfigValidator
        """
        return self.language_data_provider.get_stop_words(self.config.language)
```

---

## 🚀 **SYSTEM STARTUP PATTERN**

```python
def initialize_rag_system(language: str) -> RAGSystem:
    """
    OFFICIAL system initialization pattern
    This pattern MUST be followed by all system entry points
    """

    # Step 1: Load ALL config files (no validation yet)
    main_config = load_toml("config/config.toml")
    language_configs = {
        "hr": load_toml("config/hr.toml"),
        "en": load_toml("config/en.toml"),
        # Add new languages here as needed
    }

    # Step 2: VALIDATE EVERYTHING FIRST - fail fast if anything missing
    ConfigValidator.validate_startup_config(main_config, language_configs)

    # Step 3: If we reach here, ALL required keys exist
    # Create components with guaranteed-valid config
    query_config = QueryProcessingConfig.from_validated_config(main_config, language)
    embedding_config = EmbeddingConfig.from_validated_config(main_config)
    generation_config = GenerationConfig.from_validated_config(main_config, language_configs[language])

    # Step 4: Create DI components with explicit dependencies
    language_provider = ProductionLanguageDataProvider(language_configs[language])
    query_processor = MultilingualQueryProcessor(query_config, language_provider)

    # Step 5: Return fully initialized system
    return RAGSystem(
        query_processor=query_processor,
        embedding_service=embedding_service,
        generation_service=generation_service
    )
```

---

## ❌ **FORBIDDEN PATTERNS**

### **NEVER USE - Silent Fallbacks**
```python
# ❌ FORBIDDEN: Silent fallback with .get()
expand_synonyms = config.get("expand_synonyms", True)  # Hides missing config

# ❌ FORBIDDEN: Magic defaults in code
min_length = query_config.get("min_query_length", 3)  # Should be in config file

# ❌ FORBIDDEN: Scattered error handling
try:
    model = config["embeddings"]["model_name"]
except KeyError:
    model = "default-model"  # Silent failure
```

### **ALWAYS USE - Explicit Patterns**
```python
# ✅ CORRECT: ConfigValidator ensures existence
expand_synonyms = config["query_processing"]["expand_synonyms"]  # Guaranteed to exist

# ✅ CORRECT: Explicit validation with clear errors
if "expand_synonyms" not in config["query_processing"]:
    raise ConfigurationError("Missing required config: query_processing.expand_synonyms in config/config.toml")

# ✅ CORRECT: Fail-fast at startup, not runtime
ConfigValidator.validate_startup_config(main_config, language_configs)
```

---

## 📋 **REFACTORING PROTOCOL**

### **When Adding New Config Keys**

1. **Update Base Schema**: Add to `ConfigValidator.BASE_INTERFACE_SCHEMA`
2. **Update All Language Files**: Add key to hr.toml, en.toml, etc.
3. **Update DI Classes**: Modify dataclasses to include new fields
4. **Test Validation**: Ensure missing key fails at startup

### **When Adding New Languages**

1. **Create Language File**: `config/{language_code}.toml`
2. **Follow Base Interface**: Copy structure from hr.toml or en.toml
3. **Update System Initialization**: Add to language_configs loading
4. **Validate Structure**: Ensure ConfigValidator passes

### **NEVER DO**
- Add `.get()` fallbacks to "fix" missing config
- Hardcode defaults in business logic
- Skip ConfigValidator for "quick fixes"
- Mix old/new patterns in same codebase

---

## 🔄 **MIGRATION STRATEGY**

### **Phase 1: Implement ConfigValidator** ⏳
- Create ConfigValidator class
- Define complete BASE_INTERFACE_SCHEMA
- Add to system startup (fail-fast)
- Test with existing config files

### **Phase 2: Remove Silent Fallbacks** 🚫
Priority order (200+ instances found):
1. `services/rag-service/src/retrieval/query_processor.py` (11 fallbacks)
2. `services/rag-service/src/retrieval/ranker.py` (15 fallbacks)
3. `services/rag-service/src/pipeline/config.py` (20+ fallbacks)
4. `services/rag-service/src/generation/response_parser.py` (10+ fallbacks)
5. Continue through all modules systematically

### **Phase 3: Update DI Components** 🏗️
- Convert all config classes to dataclasses
- Use `from_validated_config()` pattern
- Remove all `.get()` with defaults
- Direct dictionary access throughout

### **Phase 4: Testing & Validation** ✅
- Verify missing configs fail at startup
- Ensure no runtime config errors
- Test all language combinations
- Document new patterns

---

## 📚 **REFERENCES & COMPLIANCE**

### **AI Instructions Compliance**
- ✅ **Explicit Communication**: No assumptions, clear error messages
- ✅ **Consistency Over Compatibility**: Complete systematic changes
- ✅ **Fail-Fast Philosophy**: Startup validation, no silent fallbacks
- ✅ **Debate and Critical Thinking**: Architectural alternatives considered
- ✅ **Clean Code & DRY**: Single responsibility, no duplication

### **Related Documentation**
- `AI_INSTRUCTIONS.md` - Core development principles
- `CLAUDE.md` - Project context and architecture
- `config/config.toml` - Main configuration file
- `config/hr.toml` - Croatian language configuration
- `config/en.toml` - English language configuration

---

## ⚠️ **VIOLATION DETECTION**

**RED FLAGS** that indicate deviation from this architecture:
- New `.get()` patterns in config access
- Magic defaults hardcoded in business logic
- Silent error handling for missing config
- Mixed old/new patterns in same module
- Skip ConfigValidator "just this once"

**ENFORCEMENT**: All PRs must include config architecture compliance check.

---

**END OF SPECIFICATION**
**Status**: ✅ APPROVED FOR IMPLEMENTATION
**Next Action**: Begin Phase 1 - ConfigValidator implementation
