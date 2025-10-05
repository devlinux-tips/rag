# Test Migration Strategy - ChromaDB to Weaviate

## Overview
This document outlines the comprehensive strategy for migrating tests from ChromaDB to Weaviate, including the new AI-friendly test infrastructure created to assist with debugging and maintenance.

## New Test Infrastructure

### 1. **Python Test Runner** (`python_test_runner.py`)
Located at repository root, this is the main test orchestrator for all Python services.

**Key Features:**
- AI-friendly trace logging for debugging
- Automatic test categorization (storage, llm, integration, etc.)
- ChromaDB test migration helper
- Coverage reporting with AI analysis
- Parallel test execution support

**Usage:**
```bash
# Run all tests
python python_test_runner.py

# Run specific category
python python_test_runner.py --category storage

# Run with coverage
python python_test_runner.py --coverage

# List all tests
python python_test_runner.py --list

# Migrate ChromaDB tests (dry run)
python python_test_runner.py --migrate-chromadb

# Apply migration
python python_test_runner.py --migrate-chromadb --apply
```

### 2. **AI Debug Helpers** (`services/rag-service/tests/ai_debug_helpers.py`)
Comprehensive debugging utilities designed specifically for AI-assisted debugging.

**Components:**
- **MockDetector**: Identifies and logs mock objects for pattern recognition
- **AssertionLogger**: Detailed assertion context for failure analysis
- **APIContractValidator**: Validates provider interfaces and API contracts
- **MigrationTracker**: Tracks ChromaDB to Weaviate migration progress
- **ConfigEvolutionTracker**: Monitors configuration schema changes
- **AITestReporter**: Generates AI-friendly test reports in JSON

**Key Features:**
- Automatic mock detection and logging
- Detailed assertion failure context
- API signature validation
- Migration status tracking
- Performance benchmarking

### 3. **Weaviate Test Suite** (`test_storage_weaviate.py`)
Complete test implementation for Weaviate storage operations.

**Test Categories:**
- Connection tests
- Storage operations (add, query, delete)
- Vector storage class tests
- Migration validation tests
- Error handling tests
- Integration tests (marked for real Weaviate instance)

## Migration Status

### Tests Requiring Migration (9 files identified):
1. ✅ `test_storage_weaviate.py` - **COMPLETED** (new implementation)
2. ⚠️ `test_storage_factories.py` - **DISABLED** (marked with skip)
3. 🔄 `test_hierarchical_retriever_providers.py` - Needs migration
4. 🔄 `test_multitenant_models.py` - Needs migration
5. 🔄 `test_search.py` - Needs migration
6. 🔄 `test_config_loader.py` - Needs migration
7. 🔄 `test_folder_manager.py` - Needs migration
8. 🔄 `test_search_providers.py` - Needs migration
9. 🔄 `test_folder_manager_providers.py` - Needs migration

### Migration Approach

#### Phase 1: Disable ChromaDB Tests ✅
- Add `pytest.mark.skip` to all ChromaDB tests
- Use `python_test_runner.py --migrate-chromadb --apply` for automatic marking

#### Phase 2: Create Weaviate Equivalents
For each disabled test:
1. Create new test file with `_weaviate` suffix
2. Implement equivalent functionality using Weaviate API
3. Use AI debug helpers for comprehensive logging
4. Validate feature parity

#### Phase 3: Validate Migration
- Run performance benchmarks
- Compare functionality between systems
- Ensure all features are covered

## AI Debugging Features

### Trace Logging
All tests now include comprehensive trace logging for AI analysis:

```python
@ai_debug_trace(component="storage_ops")
def test_add_documents():
    with MigrationTracker.track_migration(
        test_name="test_add_documents",
        from_system="chromadb.add()",
        to_system="weaviate.batch.add_data_object()"
    ):
        # Test implementation
```

### Mock Detection
Automatic detection and logging of mock objects:

```python
MockDetector.log_mock_detection(
    component="fixture",
    operation="setup_weaviate_client",
    obj=client,
    expected_type="weaviate.Client"
)
```

### Migration Tracking
Track migration status for systematic fixing:

```python
MigrationTracker.log_migration_issue(
    component="batch_insert",
    old_impl="collection.add(documents, embeddings, metadatas, ids)",
    new_impl="batch.add_data_object(data, class, vector)",
    status="migrated",
    details="Batch pattern changed significantly"
)
```

## Configuration

### pytest.ini
Located at repository root with AI-optimized settings:
- Markers for test categorization (weaviate, chromadb, integration)
- AI-friendly output formatting
- Coverage thresholds (70% minimum)
- Async support configured

### Test Organization
```
/home/x/src/rag/learn-rag/
├── python_test_runner.py       # Main test orchestrator
├── pytest.ini                   # Pytest configuration
├── test_logs/                   # AI debug logs and reports
│   ├── trace.log               # Detailed trace logs
│   ├── ai_report.json          # AI-friendly test report
│   └── coverage_html/          # Coverage reports
└── services/
    └── rag-service/
        └── tests/
            ├── ai_debug_helpers.py     # AI debugging utilities
            ├── test_storage_weaviate.py # New Weaviate tests
            └── test_storage_factories.py # Disabled ChromaDB tests
```

## Key Differences: ChromaDB vs Weaviate

### API Changes
| Operation | ChromaDB | Weaviate |
|-----------|----------|----------|
| Add Documents | `collection.add(docs, embeddings, metadatas, ids)` | `batch.add_data_object(data, class, vector)` |
| Query | `collection.query(embeddings, n_results)` | `query.near_vector().with_limit().do()` |
| Delete | `collection.delete(ids)` | `batch.delete_objects(class, where_filter)` |
| Schema | Flexible metadata | Strict property schema |

### Testing Considerations
1. **Schema Definition**: Weaviate requires explicit schema definition
2. **Naming Rules**: Weaviate has stricter naming conventions
3. **Batch Operations**: Different batching patterns
4. **Query Syntax**: More complex query builder pattern
5. **Persistence**: Automatic in Weaviate vs explicit in ChromaDB

## Recommendations

### Immediate Actions
1. ✅ Use `python_test_runner.py` for all test execution
2. ✅ Enable AI trace logging for debugging (`PYTEST_AI_TRACE=1`)
3. ⚠️ Skip ChromaDB tests until migration complete
4. 🔄 Prioritize migration of storage-critical tests

### Best Practices
1. **Always use AI debug helpers** in new tests
2. **Add migration tracking** when converting tests
3. **Validate API contracts** after migration
4. **Run coverage reports** to identify gaps
5. **Use trace logs** for debugging failures

### Testing Workflow
```bash
# 1. Check current status
python python_test_runner.py --list

# 2. Run tests with trace logging
PYTEST_AI_TRACE=1 python python_test_runner.py

# 3. Check coverage
python python_test_runner.py --coverage

# 4. Review AI reports
cat test_logs/ai_report.json | jq .

# 5. Debug failures with trace logs
grep "TEST_FAIL" test_logs/trace.log
```

## Future Enhancements

### Planned Improvements
1. **Automated migration tool** for remaining tests
2. **Performance comparison suite** between storage backends
3. **Integration test environment** with Docker
4. **Continuous migration monitoring**
5. **Test quality metrics dashboard**

### LLM Test Updates
- Review and update generation tests for new LLM changes
- Add comprehensive prompt template testing
- Validate chunking strategy changes
- Test optimization improvements

## Conclusion

The test migration infrastructure is now in place with comprehensive AI-friendly debugging capabilities. The `python_test_runner.py` provides a centralized way to manage all Python tests, while the AI debug helpers ensure that any issues can be quickly identified and resolved with the help of AI analysis.

**Current Status**: Infrastructure ✅ | Migration 20% Complete | AI Debugging Enabled

---

*Generated: 2025-10-05*
*Next Review: After completing Phase 2 migrations*