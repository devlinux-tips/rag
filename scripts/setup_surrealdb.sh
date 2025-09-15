#!/bin/bash
# SurrealDB Database Setup Script
# Sets up database, loads schema, and creates initial data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCHEMA_FILE="$PROJECT_ROOT/services/rag-service/schema/multitenant_schema.surql"
DATA_DIR="$PROJECT_ROOT/data/surrealdb"

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
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup SurrealDB database and schema for Multi-tenant RAG System

Options:
    --host HOST         Database host (default: $DEFAULT_HOST)
    --port PORT         Database port (default: $DEFAULT_PORT)
    --user USER         Database user (default: $DEFAULT_USER)
    --pass PASS         Database password (default: $DEFAULT_PASS)
    --namespace NS      Database namespace (default: $DEFAULT_NS)
    --database DB       Database name (default: $DEFAULT_DB)
    --file-mode         Use file-based database instead of server
    --docker            Use Docker container setup
    --reset             Reset database (drop all data)
    --schema-only       Only load schema, skip data
    --help, -h          Show this help

Examples:
    $0                                  # Setup with defaults
    $0 --docker                        # Setup with Docker
    $0 --file-mode                     # Use file-based database
    $0 --reset --schema-only           # Reset and load schema only
    $0 --host db.example.com --port 8080 --user admin --pass secret

Database will be created at:
    Server mode: ws://$DEFAULT_HOST:$DEFAULT_PORT
    File mode:   $DATA_DIR/rag.db
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
    FILE_MODE=false
    DOCKER_MODE=false
    RESET_DB=false
    SCHEMA_ONLY=false

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
            --file-mode)
                FILE_MODE=true
                shift
                ;;
            --docker)
                DOCKER_MODE=true
                shift
                ;;
            --reset)
                RESET_DB=true
                shift
                ;;
            --schema-only)
                SCHEMA_ONLY=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Validate prerequisites
validate_prerequisites() {
    print_step "Validating prerequisites..."

    # Check if SurrealDB is installed (unless Docker mode)
    if [[ "$DOCKER_MODE" == false ]] && ! command_exists surreal; then
        print_error "SurrealDB not found. Please install it first:"
        echo "  Run: ./scripts/install_surrealdb.sh"
        exit 1
    fi

    # Check if schema file exists
    if [[ ! -f "$SCHEMA_FILE" ]]; then
        print_error "Schema file not found: $SCHEMA_FILE"
        exit 1
    fi

    # Create data directory if needed
    if [[ "$FILE_MODE" == true ]] || [[ "$DOCKER_MODE" == true ]]; then
        mkdir -p "$DATA_DIR"
        print_step "Created data directory: $DATA_DIR"
    fi

    print_success "Prerequisites validated"
}

# Start SurrealDB server in background for file mode
start_file_server() {
    local db_file="$DATA_DIR/rag.db"
    local pid_file="$DATA_DIR/surreal.pid"

    print_step "Starting SurrealDB in file mode..."

    # Kill existing process if running
    if [[ -f "$pid_file" ]]; then
        local old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            print_step "Stopping existing SurrealDB process ($old_pid)..."
            kill "$old_pid"
            sleep 2
        fi
        rm -f "$pid_file"
    fi

    # Start SurrealDB
    surreal start \
        --log info \
        --user "$USER" \
        --pass "$PASS" \
        --bind "0.0.0.0:$PORT" \
        "file://$db_file" &

    local pid=$!
    echo "$pid" > "$pid_file"

    # Wait for server to start
    print_step "Waiting for SurrealDB to start..."
    for i in {1..30}; do
        if surreal sql \
            --conn "ws://$HOST:$PORT" \
            --user "$USER" \
            --pass "$PASS" \
            --ns "$NAMESPACE" \
            --db "$DATABASE" \
            "INFO KV;" >/dev/null 2>&1; then
            print_success "SurrealDB started successfully (PID: $pid)"
            return 0
        fi
        sleep 1
    done

    print_error "Failed to start SurrealDB"
    kill "$pid" 2>/dev/null || true
    rm -f "$pid_file"
    return 1
}

# Start Docker container
start_docker_container() {
    print_step "Starting SurrealDB Docker container..."

    local compose_file="$PROJECT_ROOT/docker-compose.surrealdb.yml"

    if [[ ! -f "$compose_file" ]]; then
        print_error "Docker Compose file not found: $compose_file"
        print_step "Run the installation script first: ./scripts/install_surrealdb.sh --docker"
        exit 1
    fi

    # Start container
    cd "$PROJECT_ROOT"
    if docker-compose -f docker-compose.surrealdb.yml up -d; then
        print_success "Docker container started"

        # Wait for container to be ready
        print_step "Waiting for Docker container to be ready..."
        for i in {1..60}; do
            if docker-compose -f docker-compose.surrealdb.yml exec -T surrealdb \
                surreal sql \
                --conn "http://localhost:8000" \
                --user "$USER" \
                --pass "$PASS" \
                --ns "$NAMESPACE" \
                --db "$DATABASE" \
                "INFO KV;" >/dev/null 2>&1; then
                print_success "Docker container is ready"
                return 0
            fi
            sleep 2
        done

        print_error "Docker container failed to become ready"
        return 1
    else
        print_error "Failed to start Docker container"
        return 1
    fi
}

# Test database connection
test_connection() {
    print_step "Testing database connection..."

    local conn_string
    if [[ "$DOCKER_MODE" == true ]]; then
        conn_string="http://$HOST:$PORT"
    else
        conn_string="ws://$HOST:$PORT"
    fi

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
        print_step "Connection details:"
        echo "  Host: $HOST"
        echo "  Port: $PORT"
        echo "  User: $USER"
        echo "  Namespace: $NAMESPACE"
        echo "  Database: $DATABASE"
        return 1
    fi
}

# Reset database
reset_database() {
    if [[ "$RESET_DB" == true ]]; then
        print_warning "Resetting database - ALL DATA WILL BE LOST!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Reset cancelled"
            return 0
        fi

        print_step "Dropping database..."
        local conn_string
        if [[ "$DOCKER_MODE" == true ]]; then
            conn_string="http://$HOST:$PORT"
        else
            conn_string="ws://$HOST:$PORT"
        fi

        surreal sql \
            --conn "$conn_string" \
            --user "$USER" \
            --pass "$PASS" \
            --ns "$NAMESPACE" \
            --db "$DATABASE" \
            "REMOVE DATABASE $DATABASE;" || true

        print_success "Database reset completed"
    fi
}

# Load schema
load_schema() {
    print_step "Loading database schema..."

    local conn_string
    if [[ "$DOCKER_MODE" == true ]]; then
        conn_string="http://$HOST:$PORT"
    else
        conn_string="ws://$HOST:$PORT"
    fi

    if surreal import \
        --conn "$conn_string" \
        --user "$USER" \
        --pass "$PASS" \
        --ns "$NAMESPACE" \
        --db "$DATABASE" \
        "$SCHEMA_FILE"; then
        print_success "Schema loaded successfully"
    else
        print_error "Failed to load schema"
        return 1
    fi
}

# Verify schema installation
verify_schema() {
    print_step "Verifying schema installation..."

    local conn_string
    if [[ "$DOCKER_MODE" == true ]]; then
        conn_string="http://$HOST:$PORT"
    else
        conn_string="ws://$HOST:$PORT"
    fi

    # Check if tables exist
    local tables=$(surreal sql \
        --conn "$conn_string" \
        --user "$USER" \
        --pass "$PASS" \
        --ns "$NAMESPACE" \
        --db "$DATABASE" \
        --pretty \
        "INFO FOR DATABASE;" 2>/dev/null | grep -o 'TABLE [a-zA-Z_]*' | wc -l || echo "0")

    if [[ "$tables" -ge 6 ]]; then
        print_success "Schema verification successful ($tables tables found)"

        # Check if initial data exists
        local tenant_count=$(surreal sql \
            --conn "$conn_string" \
            --user "$USER" \
            --pass "$PASS" \
            --ns "$NAMESPACE" \
            --db "$DATABASE" \
            "SELECT count() FROM tenant;" 2>/dev/null | grep -o '[0-9]\+' | head -1 || echo "0")

        if [[ "$tenant_count" -gt 0 ]]; then
            print_success "Initial data found ($tenant_count tenants)"
        else
            print_warning "No initial data found"
        fi
    else
        print_error "Schema verification failed (expected at least 6 tables, found $tables)"
        return 1
    fi
}

# Show connection information
show_connection_info() {
    echo
    echo "======================================"
    echo "Database Setup Complete!"
    echo "======================================"
    echo
    echo "Connection Details:"
    echo "  Host: $HOST"
    echo "  Port: $PORT"
    echo "  User: $USER"
    echo "  Namespace: $NAMESPACE"
    echo "  Database: $DATABASE"
    echo

    if [[ "$FILE_MODE" == true ]]; then
        echo "Database File: $DATA_DIR/rag.db"
        echo "PID File: $DATA_DIR/surreal.pid"
        echo
        echo "To stop the server:"
        echo "  kill \$(cat $DATA_DIR/surreal.pid)"
        echo
    elif [[ "$DOCKER_MODE" == true ]]; then
        echo "Docker Container: rag-surrealdb"
        echo
        echo "Container Management:"
        echo "  docker-compose -f docker-compose.surrealdb.yml stop"
        echo "  docker-compose -f docker-compose.surrealdb.yml start"
        echo "  docker-compose -f docker-compose.surrealdb.yml down"
        echo
    fi

    echo "Test the setup:"
    echo "  ./scripts/test_surrealdb_queries.sh"
    echo
    echo "Query examples:"
    echo "  surreal sql --conn ws://$HOST:$PORT --user $USER --pass $PASS --ns $NAMESPACE --db $DATABASE"
    echo
}

# Main setup function
main() {
    echo "======================================"
    echo "SurrealDB Database Setup"
    echo "======================================"
    echo

    parse_args "$@"
    validate_prerequisites

    # Setup database server
    if [[ "$DOCKER_MODE" == true ]]; then
        start_docker_container
    elif [[ "$FILE_MODE" == true ]]; then
        start_file_server
    fi

    # Test connection
    test_connection

    # Reset if requested
    reset_database

    # Load schema
    load_schema

    # Verify installation
    verify_schema

    # Show connection info
    show_connection_info
}

# Handle interrupt
cleanup() {
    if [[ "$FILE_MODE" == true ]] && [[ -f "$DATA_DIR/surreal.pid" ]]; then
        local pid=$(cat "$DATA_DIR/surreal.pid")
        print_step "Cleaning up server process..."
        kill "$pid" 2>/dev/null || true
        rm -f "$DATA_DIR/surreal.pid"
    fi
}

trap cleanup EXIT INT TERM

# Run main function
main "$@"