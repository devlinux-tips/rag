# Multilingual RAG Platform

## AI Development Principles

### **CRITICAL: These are your operating commandments**

#### **1. FAIL-FAST Philosophy**
- **SYSTEM MUST FAIL** if config values are missing - they MUST be in config files
- **SYSTEM MUST FAIL** if required packages not installed - they MUST be in requirements.txt
- **NO fallback defaults** in code - use direct dictionary access after validation
- **NO silent `.get()` patterns** - use explicit validation at startup
- **NO import fallbacks** - all imports must succeed or system fails
- **NO OR fallbacks** - no `value or "default"` patterns
- **NO exception swallowing** - no `except: pass` patterns
- **Validate everything at initialization** - fail loud, fail clear, fail fast

#### **2. FORBIDDEN Patterns**
- ❌ **ANY fallback patterns** - no `.get(key, default)`, no `value or default`, no `except: pass`
- ❌ **Import fallbacks** - no `try: import X except ImportError: X = None`
- ❌ **Silent failures** - no swallowing exceptions or using defaults
- ❌ **Hard-coded values** without explicit permission
- ❌ **New config values** without explicit permission
- ❌ **Noise comments** describing obvious code
- ❌ **Test logic** left in production code
- ❌ **Language-specific comments** in multilingual system

#### **3. Consistency Above All**
- **Complete systematic changes** - no half-refactored code
- **Remove ALL instances** of deprecated patterns
- **One consistent approach** applied everywhere
- **No backward compatibility** if it compromises code quality

#### **4. Think Before Code**
- **Debate and challenge first** - question requirements and architectural implications
- **Analyze the problem** - describe it, debate alternatives, then plan
- **No "AI cargo culting"** - don't fix for the sake of fixing

#### **5. Code Quality Standards (MANDATORY)**
- **MYPY COMPLIANCE BY DEFAULT**: ALL new code MUST pass MyPy type checking
  - **Write proper type annotations** from the start - never add them later
  - **Use correct Protocol implementations** - match method signatures exactly
  - **Handle None types explicitly** - no silent `.get()` fallbacks
  - **Import types correctly** - use proper `from typing import` statements
  - **Test MyPy compliance** before claiming code is complete
- **NEVER exclude files from flake8** unless they contain intentional alignment patterns (E221 only)
- **Fix all quality issues properly**:
  - **F811** (redefinition): Rename variables or remove duplicates
  - **F402** (shadowing): Rename loop variables that shadow imports
  - **F821** (undefined): Add proper imports or define variables
- **Acceptable E221 exceptions** (alignment patterns only):
  - `dependency_analyzer.py` - Complex dependency tables
  - `validate_language_configs.py` - Configuration matrices
- **Quality workflow**: `format_code.py` → `mypy src/` → `git add .` → `git commit` (all pre-commit hooks must pass)
- **Main entry point**: Always use `python rag.py` - never use module imports or internal paths
- **Push critical thinking** - force explicit discussion of pros/cons

#### **5. Architecture Rules**
- **Dependency Injection**: All dependencies explicitly provided
- **Pure Functions**: Separate business logic from I/O
- **Protocol-Based Design**: Use typing.Protocol for interfaces
- **Language Equality**: All languages get same features, only content differs

## Current System Status

### **Active Models (Config-Driven)**
- **Croatian**: `classla/bcms-bertic` (768-dim ELECTRA)
- **English**: `BAAI/bge-large-en-v1.5` (1024-dim BGE)
- **Fallback**: `BAAI/bge-m3` (multilingual, config-defined)

### **Architecture**
- **Configuration**: TOML-based with language-specific overrides (`config/hr.toml`, `config/en.toml`)
- **Storage**: Multi-tenant Weaviate collections with scope-based naming:
  - User scope: `{tenant}_{user}_{language}_documents` (e.g., `Development_dev_user_hr`)
  - Feature scope: `Features_{feature_name}_{language}` (e.g., `Features_narodne_novine_hr`)
- **Pipeline**: Document extraction → chunking → embedding → retrieval → generation
- **LLM**: qwen2.5:7b-instruct via Ollama (multilingual)

### **Key Constraints**
- **Language Parameter Required**: All components need explicit language: `RAGSystem(language="hr")`
- **Tenant Context**: Multi-tenant data isolation in `data/{tenant_slug}/`
- **Dimension Compatibility**: Different models = different dimensions = collection recreation needed

## Virtual Environment Management

### **CRITICAL: Use ONLY ONE venv**
- **Local Development**: `/home/x/src/rag/learn-rag/venv`
- **Server Deployment**: `/home/rag/src/rag/venv`
- **DO NOT CREATE**: `services/rag-service/venv` (NEVER use this)
- **All commands MUST use**: `cd <repo-root> && source venv/bin/activate`
  - Local: `cd /home/x/src/rag/learn-rag && source venv/bin/activate`
  - Server: `cd /home/rag/src/rag && source venv/bin/activate`
- **Never use relative paths for venv**: Always use absolute path from repo root

## Dependency Management

### **CRITICAL: Use ONLY ONE requirements.txt**
- **Repository Root**: `<repo-root>/requirements.txt` (MASTER FILE)
  - Local: `/home/x/src/rag/learn-rag/requirements.txt`
  - Server: `/home/rag/src/rag/requirements.txt`
- **DO NOT CREATE**: Multiple requirements.txt files in services/ subdirectories
- **Contains**: Complete RAG stack + FastAPI + development tools + psutil>=5.9.0
- **Install command**: `cd <repo-root> && source venv/bin/activate && pip install -r requirements.txt`
- **Never split dependencies** - all Python packages go in the single root requirements.txt
- **Docker containers**: Update Dockerfile COPY commands to use `COPY requirements.txt ./` from repo root

## Essential Commands

### **Development**
```bash
# ALWAYS activate from repository root
# Local: cd /home/x/src/rag/learn-rag
# Server: cd /home/rag/src/rag
cd <repo-root>
source venv/bin/activate

# CLI Usage (Multi-tenant - User scope)
python rag.py --tenant development --user dev_user --language hr query "Što je RAG?"
python rag.py --tenant development --user dev_user --language en query "What is RAG?"
python rag.py --tenant development --user dev_user --language hr process-docs data/development/users/dev_user/documents/hr/

# CLI Usage (Feature scope)
python rag.py --language hr --scope feature --feature narodne-novine query "Kolika je najviša cijena goriva?"
python rag.py --language hr --scope feature --feature narodne-novine process-docs data/features/narodne_novine/documents/hr/

# System Status
python rag.py --language hr status
python rag.py --language hr list-collections

# Configuration Testing
python -c "from src.utils.config_loader import get_unified_config; print(get_unified_config())"

# Running services - ALWAYS from repo root with venv
cd <repo-root> && source venv/bin/activate && python services/rag-api/main.py
cd <repo-root> && source venv/bin/activate && python services/rag-service/scripts/any_script.py
```

### **Quality Checks**
```bash
# ALWAYS from repo root with venv activated
cd <repo-root> && source venv/bin/activate

# Testing - Use python_test_runner.py (repository root)
python python_test_runner.py                     # Run all tests
python python_test_runner.py --coverage          # With coverage report
python python_test_runner.py --category rag-service  # Run specific category
python python_test_runner.py -vv --trace         # Verbose with AI trace logging

# Code Quality
black src/ --line-length 88
ruff check src/
mypy src/
```

## Working Directory Context

**Repository Root** (Environment-specific):
- **Local Development**: `/home/x/src/rag/learn-rag/` (ALWAYS start here)
- **Server Deployment**: `/home/rag/src/rag/` (ALWAYS start here)

**Primary Work Location**: `services/rag-service/` (relative to repo root)
**Configuration Files**: `services/rag-service/config/`
**Source Code**: `services/rag-service/src/`
**Entry Point**: `rag.py` (repo root - delegates to services/rag-service/src/cli/rag_cli.py)

## Documentation

**Detailed documentation in `/docs/`:**
- Technical architecture
- API specifications
- Deployment guides
- Model validation system design

## Development Protocol

### **Before Any Implementation:**
1. **Challenge the approach** - is this the right solution?
2. **Define explicit requirements** - what exactly should this do?
3. **Plan complete changes** - no partial implementations
4. **Identify all affected components** - systematic consistency

### **Implementation Rules:**
- **Ask before hard-coding anything**
- **Rebuild vs repair** - if easier to create new, build from ground up
- **Complete refactoring** - no mixed old/new patterns
- **Test each change** - prove it works before claiming done

### **Configuration Pattern:**
```python
# ✅ CORRECT: Direct access after validation
model_name = config["embeddings"]["model_name"]
fallback_model = config["embeddings"]["fallback_model"]

# ❌ FORBIDDEN: Silent fallbacks in production code
model_name = config.get("model_name", "default-model")

# ✅ ACCEPTABLE: Mock providers for testing only
class MockProvider:
    def get_config(self, key: str) -> dict:
        return self.test_configs.get(key, {})  # OK in tests
```

### **Error Handling Pattern:**
```python
# ✅ CORRECT: Fail-fast validation
if "model_name" not in config["embeddings"]:
    raise ConfigurationError("Missing required config: embeddings.model_name")

# ❌ FORBIDDEN: Silent error handling
try:
    model = config["embeddings"]["model_name"]
except KeyError:
    model = "default-model"  # Hides configuration problems
```

---

## AI-Specific Debugging Infrastructure

### **AI-Friendly Repository Notice**
This codebase is designed for AI implementation and debugging. Logging and error messages prioritize AI pattern recognition over human readability.

### **AI Debugging Logging Patterns**

#### **Mock Detection (Critical for Test Debugging)**
```python
def log_mock_detection(component: str, operation: str, obj: Any, expected_type: str = None):
    """Log when dealing with mock vs real objects - critical for AI test debugging."""
    is_mock = isinstance(obj, Mock) or 'Mock' in str(type(obj)) or hasattr(obj, '_mock_name')
    logger = get_system_logger()
    logger.trace("mock_detection", operation,
        f"obj={type(obj).__name__} | is_mock={is_mock} | expected={expected_type} | obj_id={id(obj)}")
```

#### **Test Assertion Context (For Systematic Fixing)**
```python
def log_assertion_context(test_name: str, component: str, expected: Any, actual: Any, assertion_type: str = "equality"):
    """Detailed assertion context for AI pattern recognition."""
    logger = get_system_logger()
    logger.error("test_assertion", test_name,
        f"ASSERTION_FAIL | component={component} | type={assertion_type} | "
        f"expected_type={type(expected).__name__} | actual_type={type(actual).__name__} | "
        f"expected_value={repr(expected)[:100]} | actual_value={repr(actual)[:100]}")
```

#### **API Contract Validation (For Provider Issues)**
```python
def log_api_mismatch(component: str, method: str, expected_sig: str, actual_call: dict):
    """Track API signature mismatches for systematic provider fixing."""
    logger = get_system_logger()
    logger.error("api_contract", f"{component}.{method}",
        f"SIGNATURE_MISMATCH | expected={expected_sig} | actual_params={list(actual_call.keys())} | "
        f"missing_params={set(expected_sig.split(',')) - set(actual_call.keys())}")
```

#### **Config Schema Evolution (For Backward Compatibility)**
```python
def log_config_evolution(key_path: str, old_format: Any, new_format: Any, compat_action: str):
    """Track configuration schema changes and compatibility decisions."""
    logger = get_system_logger()
    logger.info("config_evolution", "schema_change",
        f"path={key_path} | old_type={type(old_format).__name__} | new_type={type(new_format).__name__} | "
        f"compat_action={compat_action} | migration_needed={old_format != new_format}")
```

### **AI Debugging Commandments**
1. **Every error must include object type information**
2. **Every mock interaction must be logged with is_mock flag**
3. **Every config access must log format detection**
4. **Every test failure must include expected vs actual context**
5. **Every provider call must validate API contract**

---

**Remember: You are a Senior Architect Developer. Challenge assumptions, ensure consistency, and maintain these principles as your operational bible.**
