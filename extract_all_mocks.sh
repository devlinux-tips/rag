#!/bin/bash
# Systematic mock extraction script - extracts all mocks to conftest.py

set -e

REPO_ROOT="/home/x/src/rag/learn-rag"
cd "$REPO_ROOT"

echo "===================================="
echo "Mock Extraction - Systematic Process"
echo "===================================="
echo ""

# Files to process (in order)
FILES=(
    "services/rag-service/src/utils/language_manager_providers.py"
    "services/rag-service/src/utils/folder_manager_providers.py"
    "services/rag-service/src/generation/prompt_templates.py"
    "services/rag-service/src/generation/response_parser.py"
    "services/rag-service/src/generation/enhanced_prompt_templates_providers.py"
    "services/rag-service/src/generation/http_clients.py"
    "services/rag-service/src/generation/language_providers.py"
    "services/rag-service/src/preprocessing/extractors_providers.py"
    "services/rag-service/src/preprocessing/cleaners_providers.py"
    "services/rag-service/src/retrieval/hybrid_retriever.py"
    "services/rag-service/src/retrieval/ranker.py"
    "services/rag-service/src/retrieval/query_processor_providers.py"
    "services/rag-service/src/retrieval/hierarchical_retriever_providers.py"
    "services/rag-service/src/retrieval/categorization_providers.py"
    "services/rag-service/src/retrieval/ranker_providers.py"
    "services/rag-service/src/retrieval/reranker.py"
    "services/rag-service/src/vectordb/search_providers.py"
    "services/rag-service/src/vectordb/storage.py"
    "services/rag-service/src/vectordb/chromadb_factories.py"
    "services/rag-service/src/vectordb/embedding_loaders.py"
    "services/rag-service/src/vectordb/embedding_devices.py"
    "services/rag-service/src/cli/rag_cli.py"
)

echo "Found ${#FILES[@]} files to process"
echo ""

# Process each file
for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    FILE_NUM=$((i + 1))

    echo "[$FILE_NUM/${#FILES[@]}] Processing: $FILE"

    # Check if file exists
    if [ ! -f "$FILE" ]; then
        echo "  ⚠️  File not found, skipping"
        continue
    fi

    # Search for mock patterns
    MOCK_COUNT=$(grep -c "class Mock\|def create_mock" "$FILE" || true)

    if [ "$MOCK_COUNT" -eq 0 ]; then
        echo "  ✓ No mocks found"
    else
        echo "  ⚠️  Found $MOCK_COUNT mock definitions"
        grep -n "class Mock\|def create_mock" "$FILE" || true
    fi

    echo ""
done

echo "===================================="
echo "Summary Complete"
echo "===================================="
echo ""
echo "Next steps:"
echo "1. Review the output above"
echo "2. Extract mocks manually to conftest.py"
echo "3. Remove mocks from production files"
echo "4. Update test imports"
echo "5. Run tests"
