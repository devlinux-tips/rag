# Model Validation and Dimension Management System

## Problem Statement

When embedding models change in configuration, the system faces critical compatibility issues:

- **Dimension Mismatches**: Different models produce different embedding dimensions (classla/bcms-bertic: 768-dim vs BAAI/bge-large-en-v1.5: 1024-dim)
- **Collection Corruption**: Existing ChromaDB collections become unusable with new model dimensions
- **Silent Failures**: Runtime errors occur during query processing instead of clear startup validation
- **Manual Recovery**: Requires manual collection cleanup and document reprocessing

## System Architecture

### Core Components

#### 1. Model Registry
**Purpose**: Track model configurations and their embedding dimensions
**Location**: `src/utils/model_registry.py`

```python
@dataclass
class ModelInfo:
    name: str
    dimensions: int
    language_optimized: str
    supports_multilingual: bool
    last_validated: Optional[datetime]
    validation_status: str  # "valid", "invalid", "unknown"

class ModelRegistry:
    def get_model_info(self, model_name: str) -> ModelInfo
    def validate_model_dimensions(self, model_name: str) -> int
    def register_model_change(self, language: str, old_model: str, new_model: str)
```

#### 2. Collection Validator
**Purpose**: Validate existing collections against current model configurations
**Location**: `src/utils/collection_validator.py`

```python
@dataclass
class ValidationResult:
    collection_name: str
    expected_dimensions: int
    actual_dimensions: int
    is_compatible: bool
    requires_migration: bool
    error_details: Optional[str]

class CollectionValidator:
    def validate_collection_compatibility(self, collection_name: str, model_name: str) -> ValidationResult
    def get_collection_metadata(self, collection_name: str) -> Dict[str, Any]
    def check_all_collections(self) -> List[ValidationResult]
```

#### 3. Migration Orchestrator
**Purpose**: Coordinate collection recreation when model changes are detected
**Location**: `src/utils/migration_orchestrator.py`

```python
@dataclass
class MigrationPlan:
    collections_to_recreate: List[str]
    documents_to_reprocess: List[str]
    backup_strategy: str
    estimated_processing_time: float

class MigrationOrchestrator:
    def create_migration_plan(self, validation_results: List[ValidationResult]) -> MigrationPlan
    def execute_migration(self, plan: MigrationPlan) -> MigrationResult
    def backup_collection(self, collection_name: str) -> str
    def restore_collection(self, backup_id: str) -> bool
```

### Integration Points

#### Startup Validation
**Trigger**: System initialization
**Process**:
1. Load current model configuration
2. Validate all existing collections
3. FAIL FAST if incompatibilities detected
4. Log validation results

#### Configuration Change Detection
**Trigger**: Model configuration changes in TOML files
**Process**:
1. Detect configuration changes via file monitoring
2. Compare old vs new model configurations
3. Generate migration plan if models changed
4. Queue migration job for execution

#### Scheduled validation
**Trigger**: Orchestration system (future job scheduler)
**Frequency**: Daily or on-demand
**Process**:
1. Comprehensive system health check
2. Model availability verification
3. Collection integrity validation
4. Performance metrics collection

## Validation Rules

### Model Dimension Validation
```python
def validate_model_dimensions(model_name: str, expected_dimensions: Optional[int] = None) -> int:
    """
    Load model and verify its embedding dimensions.
    FAIL FAST if model cannot be loaded or dimensions don't match expectations.
    """
    actual_dimensions = load_and_inspect_model(model_name)

    if expected_dimensions and actual_dimensions != expected_dimensions:
        raise ConfigurationError(
            f"Model {model_name} produces {actual_dimensions}D embeddings, "
            f"expected {expected_dimensions}D"
        )

    return actual_dimensions
```

### Collection Compatibility Check
```python
def validate_collection_compatibility(collection: ChromaCollection, model_dimensions: int) -> ValidationResult:
    """
    Check if existing collection is compatible with current model.
    Collections are incompatible if embedding dimensions don't match.
    """
    try:
        # Get first document to check dimensions
        sample = collection.peek(limit=1)
        if sample and sample['embeddings']:
            actual_dimensions = len(sample['embeddings'][0])
            is_compatible = actual_dimensions == model_dimensions

            return ValidationResult(
                collection_name=collection.name,
                expected_dimensions=model_dimensions,
                actual_dimensions=actual_dimensions,
                is_compatible=is_compatible,
                requires_migration=not is_compatible
            )
    except Exception as e:
        return ValidationResult(
            collection_name=collection.name,
            expected_dimensions=model_dimensions,
            actual_dimensions=0,
            is_compatible=False,
            requires_migration=True,
            error_details=str(e)
        )
```

## Configuration Integration

### Model Metadata Extension
Add dimension tracking to language configuration files:

```toml
# hr.toml
[embeddings]
model_name = "classla/bcms-bertic"
fallback_model = "BAAI/bge-m3"
expected_dimensions = 768  # Add explicit dimension tracking
supports_multilingual = false
language_optimized = true

[embeddings.validation]
require_dimension_check = true
allow_automatic_migration = false  # FAIL FAST - no automatic fixes
backup_before_migration = true
```

### Collection Metadata Enhancement
Store model information in ChromaDB collection metadata:

```python
collection_metadata = {
    "description": f"Documents for tenant:{tenant}, user:{user}, language:{language}",
    "model_name": "classla/bcms-bertic",
    "model_dimensions": 768,
    "created_timestamp": "2024-01-15T10:30:00Z",
    "last_validated": "2024-01-15T10:30:00Z",
    "validation_status": "valid"
}
```

## Error Handling Strategy

### FAIL-FAST Approach
System must fail clearly and immediately when:
- Model cannot be loaded
- Dimension mismatch detected
- Collection corruption found
- Configuration validation fails

### Error Messages
```python
# Clear, actionable error messages
raise ConfigurationError(
    f"Model dimension mismatch detected:\n"
    f"  Language: {language}\n"
    f"  Collection: {collection_name}\n"
    f"  Current model: {current_model} ({current_dims}D)\n"
    f"  Collection expects: {expected_dims}D embeddings\n"
    f"  Action required: Run migration to recreate collection"
)
```

## Future Job Orchestration Integration

### Validation Jobs
```python
@job_task
def daily_model_validation():
    """Daily comprehensive model and collection validation"""
    validator = CollectionValidator()
    results = validator.check_all_collections()

    if any(not r.is_compatible for r in results):
        # Queue migration jobs
        orchestrator = MigrationOrchestrator()
        plan = orchestrator.create_migration_plan(results)
        queue_migration_job(plan)

@job_task
def config_change_migration(old_config: dict, new_config: dict):
    """Triggered when model configuration changes"""
    changed_models = detect_model_changes(old_config, new_config)

    for language, model_change in changed_models.items():
        # Create language-specific migration plan
        plan = create_language_migration_plan(language, model_change)
        queue_migration_job(plan)
```

### Migration Jobs
```python
@job_task
def execute_collection_migration(migration_plan: MigrationPlan):
    """Execute collection recreation with new model dimensions"""
    orchestrator = MigrationOrchestrator()

    # Backup existing collections
    backup_ids = []
    for collection in migration_plan.collections_to_recreate:
        backup_id = orchestrator.backup_collection(collection)
        backup_ids.append(backup_id)

    try:
        # Execute migration
        result = orchestrator.execute_migration(migration_plan)

        if result.success:
            # Clean up backups after successful migration
            cleanup_backups(backup_ids)
        else:
            # Restore from backups on failure
            restore_from_backups(backup_ids)
            raise MigrationError(f"Migration failed: {result.error}")

    except Exception as e:
        # Ensure backups are restored on any failure
        restore_from_backups(backup_ids)
        raise
```

## Implementation Plan

### Phase 1: Core Validation Infrastructure
1. **Model Registry** - Track model configurations and dimensions
2. **Collection Validator** - Validate existing collections against models
3. **Startup Validation** - FAIL FAST on system initialization if incompatibilities exist

### Phase 2: Migration System
1. **Migration Orchestrator** - Plan and execute collection recreation
2. **Backup/Restore** - Safe migration with rollback capability
3. **Configuration Integration** - Enhanced metadata tracking

### Phase 3: Job Integration
1. **Scheduled Validation** - Daily health checks
2. **Config Change Detection** - Automatic migration triggers
3. **Performance Monitoring** - Migration metrics and optimization

## Usage Examples

### Manual Validation
```bash
# Check all collections for compatibility
python -m src.cli.model_validator --check-all

# Validate specific language configuration
python -m src.cli.model_validator --language hr --validate-config

# Create migration plan without executing
python -m src.cli.model_validator --language en --plan-migration
```

### Programmatic Integration
```python
# In RAG system initialization
validator = CollectionValidator()
results = validator.check_all_collections()

incompatible = [r for r in results if not r.is_compatible]
if incompatible:
    error_msg = format_validation_errors(incompatible)
    raise ConfigurationError(f"Collection validation failed:\n{error_msg}")
```

This system ensures model changes are handled systematically with proper validation, migration planning, and FAIL-FAST error handling, while remaining fully configurable and integration-ready for future job orchestration.
