#!/bin/bash
# SurrealDB Installation Script for Multi-tenant RAG System
# Supports Linux, macOS, and provides Docker alternative

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SURREALDB_VERSION="2.1.0"
INSTALL_DIR="/usr/local/bin"
DATA_DIR="$PROJECT_ROOT/data/surrealdb"

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

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        CYGWIN*)    echo "windows";;
        MINGW*)     echo "windows";;
        *)          echo "unknown";;
    esac
}

# Detect architecture
detect_arch() {
    case "$(uname -m)" in
        x86_64)     echo "amd64";;
        aarch64)    echo "arm64";;
        arm64)      echo "arm64";;
        *)          echo "amd64";;  # Default fallback
    esac
}

# Install SurrealDB using official installer
install_surrealdb_official() {
    print_step "Installing SurrealDB using official installer..."

    if curl --proto '=https' --tlsv1.2 -sSf https://install.surrealdb.com | sh; then
        print_success "SurrealDB installed successfully via official installer"
        return 0
    else
        print_error "Official installer failed"
        return 1
    fi
}

# Manual binary installation
install_surrealdb_binary() {
    local os="$1"
    local arch="$2"

    print_step "Installing SurrealDB manually for $os-$arch..."

    # Construct download URL
    local binary_name="surreal"
    local archive_ext="tar.gz"

    if [[ "$os" == "windows" ]]; then
        binary_name="surreal.exe"
        archive_ext="zip"
    fi

    local download_url="https://github.com/surrealdb/surrealdb/releases/download/v${SURREALDB_VERSION}/surreal-v${SURREALDB_VERSION}.${os}-${arch}.${archive_ext}"
    local temp_dir=$(mktemp -d)
    local archive_file="$temp_dir/surrealdb.${archive_ext}"

    print_step "Downloading from: $download_url"

    if ! curl -L -o "$archive_file" "$download_url"; then
        print_error "Failed to download SurrealDB"
        rm -rf "$temp_dir"
        return 1
    fi

    print_step "Extracting archive..."
    cd "$temp_dir"

    if [[ "$archive_ext" == "tar.gz" ]]; then
        tar -xzf "$archive_file"
    else
        unzip "$archive_file"
    fi

    # Find and install binary
    local binary_path=$(find . -name "$binary_name" -type f | head -1)

    if [[ -z "$binary_path" ]]; then
        print_error "Binary not found in archive"
        rm -rf "$temp_dir"
        return 1
    fi

    print_step "Installing to $INSTALL_DIR..."
    sudo install "$binary_path" "$INSTALL_DIR/surreal"

    rm -rf "$temp_dir"
    print_success "SurrealDB installed successfully"
    return 0
}

# Install using package manager
install_surrealdb_package_manager() {
    local os="$1"

    case "$os" in
        "linux")
            if command_exists apt-get; then
                print_step "Installing via apt-get..."
                # Note: Official apt repository might not be available
                print_warning "SurrealDB apt repository not available, using binary installation"
                return 1
            elif command_exists yum; then
                print_step "Installing via yum..."
                print_warning "SurrealDB yum repository not available, using binary installation"
                return 1
            fi
            ;;
        "macos")
            if command_exists brew; then
                print_step "Installing via Homebrew..."
                if brew install surrealdb/tap/surreal; then
                    print_success "SurrealDB installed via Homebrew"
                    return 0
                else
                    print_warning "Homebrew installation failed, trying binary installation"
                    return 1
                fi
            fi
            ;;
    esac

    return 1
}

# Docker installation
install_surrealdb_docker() {
    print_step "Setting up SurrealDB with Docker..."

    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        return 1
    fi

    # Create docker-compose file for development
    cat > "$PROJECT_ROOT/docker-compose.surrealdb.yml" << 'EOF'
version: '3.8'

services:
  surrealdb:
    image: surrealdb/surrealdb:v2.1.0
    container_name: rag-surrealdb
    command:
      - start
      - --log=info
      - --user=root
      - --pass=root
      - file:///data/database.db
    ports:
      - "8000:8000"
    volumes:
      - ./data/surrealdb:/data
    environment:
      - SURREAL_LOG=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "surreal", "sql", "--conn", "http://localhost:8000", "--user", "root", "--pass", "root", "--ns", "test", "--db", "test", "--pretty", "INFO DB;"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

networks:
  default:
    name: rag-network
    external: false
EOF

    print_success "Docker Compose file created at $PROJECT_ROOT/docker-compose.surrealdb.yml"
    print_step "To start SurrealDB with Docker, run:"
    echo "  cd $PROJECT_ROOT && docker-compose -f docker-compose.surrealdb.yml up -d"

    return 0
}

# Create data directory
create_data_directory() {
    print_step "Creating data directory at $DATA_DIR..."
    mkdir -p "$DATA_DIR"
    print_success "Data directory created"
}

# Verify installation
verify_installation() {
    print_step "Verifying SurrealDB installation..."

    if command_exists surreal; then
        local version=$(surreal version 2>/dev/null | head -1 || echo "Unknown")
        print_success "SurrealDB is installed: $version"
        return 0
    else
        print_error "SurrealDB not found in PATH"
        return 1
    fi
}

# Main installation function
main() {
    echo "======================================"
    echo "SurrealDB Installation Script"
    echo "======================================"
    echo

    local os=$(detect_os)
    local arch=$(detect_arch)

    print_step "Detected OS: $os"
    print_step "Detected Architecture: $arch"

    # Check if already installed
    if command_exists surreal; then
        local current_version=$(surreal version 2>/dev/null | head -1 || echo "Unknown version")
        print_warning "SurrealDB is already installed: $current_version"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_step "Skipping installation"
            create_data_directory
            verify_installation
            exit 0
        fi
    fi

    # Create data directory
    create_data_directory

    # Offer installation methods
    echo
    echo "Choose installation method:"
    echo "1) Official installer (recommended)"
    echo "2) Package manager (Homebrew on macOS)"
    echo "3) Manual binary download"
    echo "4) Docker setup"
    echo
    read -p "Select option (1-4): " -n 1 -r choice
    echo

    case "$choice" in
        1)
            if install_surrealdb_official; then
                verify_installation
                exit 0
            else
                print_warning "Falling back to binary installation..."
                install_surrealdb_binary "$os" "$arch"
            fi
            ;;
        2)
            if ! install_surrealdb_package_manager "$os"; then
                print_warning "Package manager installation failed, trying binary..."
                install_surrealdb_binary "$os" "$arch"
            fi
            ;;
        3)
            install_surrealdb_binary "$os" "$arch"
            ;;
        4)
            install_surrealdb_docker
            exit 0
            ;;
        *)
            print_error "Invalid option selected"
            exit 1
            ;;
    esac

    # Verify final installation
    if verify_installation; then
        echo
        print_success "Installation completed successfully!"
        echo
        echo "Next steps:"
        echo "1. Run the database setup script: ./scripts/setup_surrealdb.sh"
        echo "2. Follow the setup guide: ./docs/surrealdb-setup.md"
        echo
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --docker)
        install_surrealdb_docker
        exit 0
        ;;
    --binary)
        os=$(detect_os)
        arch=$(detect_arch)
        install_surrealdb_binary "$os" "$arch"
        verify_installation
        exit 0
        ;;
    --official)
        install_surrealdb_official
        verify_installation
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 [--docker|--binary|--official|--help]"
        echo "  --docker    Use Docker installation"
        echo "  --binary    Manual binary installation"
        echo "  --official  Use official installer"
        echo "  --help      Show this help"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac