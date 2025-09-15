#!/bin/bash
# SurrealDB Query Examples and Testing Script
# Demonstrates various queries and tests the multitenant schema

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default connection settings
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="8000"
DEFAULT_USER="root"
DEFAULT_PASS="root"
DEFAULT_NS="rag"
DEFAULT_DB="multitenant"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}======================================"
    echo -e "$1"
    echo -e "======================================${NC}"
}

print_step() {
    echo -e "${BLUE}[QUERY]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [QUERY_TYPE]

Test SurrealDB multitenant schema with example queries

Options:
    --host HOST         Database host (default: $DEFAULT_HOST)
    --port PORT         Database port (default: $DEFAULT_PORT)
    --user USER         Database user (default: $DEFAULT_USER)
    --pass PASS         Database password (default: $DEFAULT_PASS)
    --namespace NS      Database namespace (default: $DEFAULT_NS)
    --database DB       Database name (default: $DEFAULT_DB)
    --docker            Use Docker container connection
    --interactive       Interactive query mode
    --help, -h          Show this help

Query Types:
    all                 Run all test queries (default)
    tenants            Query tenant information
    users              Query user information
    documents          Query document information
    categorization     Query categorization templates
    analytics          Query search analytics
    functions          Test database functions
    custom             Run custom query (interactive)

Examples:
    $0                          # Run all tests
    $0 tenants                  # Test tenant queries only
    $0 --docker users           # Test user queries via Docker
    $0 --interactive            # Interactive query mode
EOF
}

# Parse command line arguments
parse_args() {
    HOST="$DEFAULT_HOST"
    PORT="$DEFAULT_PORT"
    USER="$DEFAULT_USER"
    PASS="$DEFAULT_PASS"
    NAMESPACE="$DEFAULT_NS"
    DATABASE="$DEFAULT_DB"
    DOCKER_MODE=false
    INTERACTIVE=false
    QUERY_TYPE="all"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --user)
                USER="$2"
                shift 2
                ;;
            --pass)
                PASS="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --database)
                DATABASE="$2"
                shift 2
                ;;
            --docker)
                DOCKER_MODE=true
                shift
                ;;
            --interactive)
                INTERACTIVE=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            all|tenants|users|documents|categorization|analytics|functions|custom)
                QUERY_TYPE="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Execute SQL query with formatting
execute_query() {
    local description="$1"
    local query="$2"
    local show_output="${3:-true}"

    print_step "$description"

    local conn_string
    if [[ "$DOCKER_MODE" == true ]]; then
        conn_string="http://$HOST:$PORT"
    else
        conn_string="ws://$HOST:$PORT"
    fi

    echo -e "${YELLOW}SQL:${NC} $query"
    echo

    if [[ "$show_output" == "true" ]]; then
        if surreal sql \
            --conn "$conn_string" \
            --user "$USER" \
            --pass "$PASS" \
            --ns "$NAMESPACE" \
            --db "$DATABASE" \
            --pretty \
            "$query" 2>/dev/null; then
            print_success "Query executed successfully"
        else
            print_error "Query failed"
            return 1
        fi
    else
        surreal sql \
            --conn "$conn_string" \
            --user "$USER" \
            --pass "$PASS" \
            --ns "$NAMESPACE" \
            --db "$DATABASE" \
            "$query" >/dev/null 2>&1
        if [[ $? -eq 0 ]]; then
            print_success "Query executed successfully"
        else
            print_error "Query failed"
            return 1
        fi
    fi

    echo
}

# Test tenant queries
test_tenant_queries() {
    print_header "TENANT MANAGEMENT QUERIES"

    # List all tenants
    execute_query "List all tenants" "SELECT * FROM tenant;"

    # Get specific tenant with settings
    execute_query "Get development tenant details" "SELECT * FROM tenant:development;"

    # Count tenants by status
    execute_query "Count tenants by status" "
        SELECT status, count() AS tenant_count
        FROM tenant
        GROUP BY status;"

    # Get tenant with most users
    execute_query "Get tenant with user counts" "
        SELECT
            tenant.*,
            count(<-user.tenant_id) AS user_count
        FROM tenant
        ORDER BY user_count DESC;"
}

# Test user queries
test_user_queries() {
    print_header "USER MANAGEMENT QUERIES"

    # List all users with tenant info
    execute_query "List all users with tenant information" "
        SELECT
            user.*,
            tenant_id.name AS tenant_name,
            tenant_id.slug AS tenant_slug
        FROM user;"

    # Get users by role
    execute_query "Count users by role" "
        SELECT role, count() AS user_count
        FROM user
        GROUP BY role;"

    # Get users by tenant
    execute_query "Get users for development tenant" "
        SELECT * FROM user
        WHERE tenant_id = tenant:development;"

    # Find admin users
    execute_query "Find all admin users" "
        SELECT
            username,
            full_name,
            tenant_id.name AS tenant_name
        FROM user
        WHERE role = 'admin';"
}

# Test document queries
test_document_queries() {
    print_header "DOCUMENT MANAGEMENT QUERIES"

    # Count documents by status
    execute_query "Count documents by status" "
        SELECT status, count() AS doc_count
        FROM document
        GROUP BY status;"

    # Count documents by language
    execute_query "Count documents by language" "
        SELECT language, count() AS doc_count
        FROM document
        GROUP BY language;"

    # Count documents by scope
    execute_query "Count documents by scope" "
        SELECT scope, count() AS doc_count
        FROM document
        GROUP BY scope;"

    # Get documents with user and tenant info
    execute_query "Get documents with owner information" "
        SELECT
            document.title,
            document.filename,
            document.status,
            document.language,
            user_id.username AS owner,
            tenant_id.name AS tenant_name
        FROM document
        LIMIT 5;"

    # Get document categories
    execute_query "Get unique document categories" "
        SELECT array::flatten(categories) AS all_categories
        FROM document
        WHERE categories != [];"
}

# Test categorization queries
test_categorization_queries() {
    print_header "CATEGORIZATION TEMPLATE QUERIES"

    # List all categorization templates
    execute_query "List all categorization templates" "
        SELECT * FROM categorization_template;"

    # Get templates by category
    execute_query "Count templates by category" "
        SELECT category, count() AS template_count
        FROM categorization_template
        GROUP BY category;"

    # Get system default templates
    execute_query "Get system default templates" "
        SELECT name, category, language, is_system_default
        FROM categorization_template
        WHERE is_system_default = true;"

    # Get active templates for Croatian
    execute_query "Get active Croatian templates" "
        SELECT name, category, priority
        FROM categorization_template
        WHERE language = 'hr' AND is_active = true
        ORDER BY priority DESC;"
}

# Test analytics queries
test_analytics_queries() {
    print_header "SEARCH ANALYTICS QUERIES"

    # Note: This will be empty in a fresh database
    execute_query "Count search queries by language" "
        SELECT query_language, count() AS query_count
        FROM search_query
        GROUP BY query_language;"

    execute_query "Get average response times" "
        SELECT
            query_language,
            math::round(math::mean(response_time_ms)) AS avg_response_ms,
            count() AS total_queries
        FROM search_query
        GROUP BY query_language;"

    execute_query "Get popular categories" "
        SELECT primary_category, count() AS query_count
        FROM search_query
        WHERE primary_category IS NOT NONE
        GROUP BY primary_category
        ORDER BY query_count DESC;"
}

# Test database functions
test_functions() {
    print_header "DATABASE FUNCTIONS"

    # Test collection name function
    execute_query "Test collection name function" "
        SELECT fn::get_collection_name('development', 'user', 'hr') AS collection_name;"

    # Test user access function
    execute_query "Test user access function" "
        SELECT fn::user_can_access_document(user:dev_user, {
            tenant_id: tenant:development,
            scope: 'user',
            user_id: user:dev_user
        }) AS can_access;"

    # Test system configuration
    execute_query "Get system configuration" "
        SELECT * FROM system_config
        WHERE is_system_config = true;"
}

# Interactive query mode
interactive_mode() {
    print_header "INTERACTIVE QUERY MODE"
    echo "Enter SurrealQL queries. Type 'exit' to quit, 'help' for examples."
    echo

    local conn_string
    if [[ "$DOCKER_MODE" == true ]]; then
        conn_string="http://$HOST:$PORT"
    else
        conn_string="ws://$HOST:$PORT"
    fi

    while true; do
        echo -n "surreal> "
        read -r query

        case "$query" in
            "exit"|"quit"|"q")
                break
                ;;
            "help"|"h")
                echo "Example queries:"
                echo "  SELECT * FROM tenant;"
                echo "  SELECT * FROM user WHERE role = 'admin';"
                echo "  SELECT count() FROM document;"
                echo "  INFO FOR DATABASE;"
                echo "  exit"
                echo
                ;;
            "")
                continue
                ;;
            *)
                echo
                surreal sql \
                    --conn "$conn_string" \
                    --user "$USER" \
                    --pass "$PASS" \
                    --ns "$NAMESPACE" \
                    --db "$DATABASE" \
                    --pretty \
                    "$query" 2>/dev/null || print_error "Query failed"
                echo
                ;;
        esac
    done
}

# Insert sample data for testing
insert_sample_data() {
    print_header "INSERTING SAMPLE DATA FOR TESTING"

    # Insert a sample search query
    execute_query "Insert sample search query" "
        INSERT INTO search_query {
            tenant_id: tenant:development,
            user_id: user:dev_user,
            query_text: 'Što je hrvatski identitet?',
            query_language: 'hr',
            detected_language: 'hr',
            primary_category: 'cultural',
            secondary_categories: ['identity', 'tradition'],
            retrieval_strategy: 'semantic',
            scope_searched: ['user', 'tenant'],
            results_count: 5,
            response_time_ms: 450,
            satisfaction_rating: 4
        };" false

    # Insert a sample document
    execute_query "Insert sample document" "
        INSERT INTO document {
            tenant_id: tenant:development,
            user_id: user:dev_user,
            title: 'Hrvatska kultura i tradicija',
            filename: 'kultura.pdf',
            file_path: '/data/development/users/dev_user/documents/hr/kultura.pdf',
            file_size: 2048576,
            file_type: 'pdf',
            language: 'hr',
            scope: 'user',
            status: 'processed',
            content_hash: 'sha256:abc123',
            categories: ['cultural', 'traditional'],
            tags: ['hrvatska', 'kultura', 'identitet'],
            chunk_count: 15
        };" false

    print_success "Sample data inserted"
}

# Connection test
test_connection() {
    local conn_string
    if [[ "$DOCKER_MODE" == true ]]; then
        conn_string="http://$HOST:$PORT"
    else
        conn_string="ws://$HOST:$PORT"
    fi

    print_step "Testing database connection..."

    if surreal sql \
        --conn "$conn_string" \
        --user "$USER" \
        --pass "$PASS" \
        --ns "$NAMESPACE" \
        --db "$DATABASE" \
        "INFO KV;" >/dev/null 2>&1; then
        print_success "Database connection successful"
        return 0
    else
        print_error "Database connection failed"
        echo "Connection details:"
        echo "  Host: $HOST"
        echo "  Port: $PORT"
        echo "  User: $USER"
        echo "  Namespace: $NAMESPACE"
        echo "  Database: $DATABASE"
        return 1
    fi
}

# Main function
main() {
    echo "======================================"
    echo "SurrealDB Query Examples & Testing"
    echo "======================================"
    echo

    parse_args "$@"

    # Test connection first
    if ! test_connection; then
        print_error "Cannot connect to database. Make sure SurrealDB is running:"
        echo "  ./scripts/setup_surrealdb.sh"
        exit 1
    fi

    # Handle interactive mode
    if [[ "$INTERACTIVE" == true ]]; then
        interactive_mode
        exit 0
    fi

    # Insert sample data for better testing
    if [[ "$QUERY_TYPE" == "all" ]]; then
        insert_sample_data
    fi

    # Run specific query types
    case "$QUERY_TYPE" in
        "all")
            test_tenant_queries
            test_user_queries
            test_document_queries
            test_categorization_queries
            test_analytics_queries
            test_functions
            ;;
        "tenants")
            test_tenant_queries
            ;;
        "users")
            test_user_queries
            ;;
        "documents")
            test_document_queries
            ;;
        "categorization")
            test_categorization_queries
            ;;
        "analytics")
            test_analytics_queries
            ;;
        "functions")
            test_functions
            ;;
        "custom")
            interactive_mode
            ;;
    esac

    echo
    print_success "Query testing completed!"
    echo
    echo "To run interactive queries:"
    echo "  $0 --interactive"
    echo
}

# Run main function
main "$@"