#!/bin/bash
#
# RAG Platform - Local Server Setup Script
# Ubuntu 24.04 LTS
# Installs all services natively without Docker
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RAG_HOME="/home/rag/src/rag"
WEAVIATE_VERSION="v1.33.0"
WEAVIATE_URL="https://github.com/weaviate/weaviate/releases/download/${WEAVIATE_VERSION}/weaviate-${WEAVIATE_VERSION}-linux-amd64.tar.gz"
WEAVIATE_DATA_DIR="${RAG_HOME}/weaviate_data"
WEAVIATE_INSTALL_DIR="/opt/weaviate"

POSTGRES_VERSION="16"
POSTGRES_DB="ragdb"
POSTGRES_USER="raguser"
POSTGRES_PASSWORD="x|B&h@p4F@o|k6t;~X]1A((Z.,RG"
POSTGRES_PORT="5434"

# Service ports
WEAVIATE_PORT="8080"
WEAVIATE_GRPC_PORT="50051"
REDIS_PORT="6379"
RAG_API_PORT="8082"
WEB_API_PORT="3000"
WEB_UI_PORT="5173"
NGINX_PORT="80"

# Generate secure JWT secret (will be generated on first run)
JWT_SECRET=""

# Log function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Generate secure random JWT secret
generate_jwt_secret() {
    openssl rand -base64 64 | tr -d '\n'
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check system requirements
check_system() {
    log "Checking system requirements..."

    # Check OS
    if [ ! -f /etc/os-release ]; then
        log_error "Cannot determine OS version"
        exit 1
    fi

    source /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        log_warning "This script is designed for Ubuntu. Your OS: $ID"
    fi

    # Check architecture
    ARCH=$(uname -m)
    if [ "$ARCH" != "x86_64" ]; then
        log_error "This script requires x86_64 architecture. Found: $ARCH"
        exit 1
    fi

    # Check available memory (need at least 8GB, recommend 16GB+)
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 8 ]; then
        log_error "Insufficient memory. Need at least 8GB, found: ${TOTAL_MEM}GB"
        exit 1
    fi

    log "System check passed: Ubuntu on x86_64 with ${TOTAL_MEM}GB RAM"
}

# Update system packages
update_system() {
    log "Updating system packages..."
    apt-get update -qq
    apt-get upgrade -y -qq
    log "System packages updated"
}

# Install PostgreSQL 16
install_postgresql() {
    log "Installing PostgreSQL ${POSTGRES_VERSION}..."

    # Check if already installed
    if command -v psql &> /dev/null; then
        INSTALLED_VERSION=$(psql --version | awk '{print $3}' | cut -d. -f1)
        if [ "$INSTALLED_VERSION" == "$POSTGRES_VERSION" ]; then
            log_info "PostgreSQL ${POSTGRES_VERSION} already installed"
            return
        fi
    fi

    # Install PostgreSQL
    apt-get install -y -qq postgresql-${POSTGRES_VERSION} postgresql-contrib-${POSTGRES_VERSION}

    # Start PostgreSQL
    systemctl start postgresql
    systemctl enable postgresql

    log "PostgreSQL ${POSTGRES_VERSION} installed and started"
}

# Configure PostgreSQL
configure_postgresql() {
    log "Configuring PostgreSQL..."

    # Create database and user
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS ${POSTGRES_DB};" 2>/dev/null || true
    sudo -u postgres psql -c "DROP USER IF EXISTS ${POSTGRES_USER};" 2>/dev/null || true
    sudo -u postgres psql -c "CREATE USER ${POSTGRES_USER} WITH PASSWORD '${POSTGRES_PASSWORD}';"
    sudo -u postgres psql -c "CREATE DATABASE ${POSTGRES_DB} OWNER ${POSTGRES_USER};"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO ${POSTGRES_USER};"

    # Configure to listen on custom port
    PG_CONFIG="/etc/postgresql/${POSTGRES_VERSION}/main/postgresql.conf"
    if [ -f "$PG_CONFIG" ]; then
        sed -i "s/#port = 5432/port = ${POSTGRES_PORT}/" "$PG_CONFIG"
        sed -i "s/port = 5432/port = ${POSTGRES_PORT}/" "$PG_CONFIG"
    fi

    # Allow connections from localhost
    PG_HBA="/etc/postgresql/${POSTGRES_VERSION}/main/pg_hba.conf"
    if [ -f "$PG_HBA" ]; then
        echo "host    ${POSTGRES_DB}    ${POSTGRES_USER}    127.0.0.1/32    scram-sha-256" >> "$PG_HBA"
        echo "host    ${POSTGRES_DB}    ${POSTGRES_USER}    ::1/128         scram-sha-256" >> "$PG_HBA"
    fi

    # Restart PostgreSQL
    systemctl restart postgresql

    log "PostgreSQL configured: database=${POSTGRES_DB}, port=${POSTGRES_PORT}"
}

# Install Redis
install_redis() {
    log "Installing Redis..."

    if command -v redis-server &> /dev/null; then
        log_info "Redis already installed"
        return
    fi

    apt-get install -y -qq redis-server

    # Configure Redis
    sed -i 's/^supervised no/supervised systemd/' /etc/redis/redis.conf
    sed -i 's/^# maxmemory <bytes>/maxmemory 8gb/' /etc/redis/redis.conf
    sed -i 's/^# maxmemory-policy noeviction/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf

    # Start Redis
    systemctl start redis-server
    systemctl enable redis-server

    log "Redis installed and started"
}

# Install Nginx
install_nginx() {
    log "Installing Nginx..."

    if command -v nginx &> /dev/null; then
        log_info "Nginx already installed"
        return
    fi

    apt-get install -y -qq nginx

    # Stop Nginx for now (we'll configure it later)
    systemctl stop nginx

    log "Nginx installed"
}

# Install Python dependencies
install_python_deps() {
    log "Installing Python dependencies..."

    # Install Python build dependencies
    apt-get install -y -qq python3-pip python3-venv python3-dev build-essential

    # Create virtualenv if it doesn't exist
    if [ ! -d "${RAG_HOME}/venv" ]; then
        log_info "Creating Python virtual environment at ${RAG_HOME}/venv..."
        sudo -u rag python3 -m venv "${RAG_HOME}/venv"
        log "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi

    # Install requirements
    log_info "Installing Python packages from requirements.txt..."
    cd "${RAG_HOME}"
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    deactivate

    log "Python dependencies installed"
}

# Install Node.js dependencies
install_node_deps() {
    log "Installing Node.js dependencies..."

    # Check Node.js version
    if ! command -v node &> /dev/null; then
        log_error "Node.js not found. Please install Node.js 18+ first"
        exit 1
    fi

    NODE_VERSION=$(node --version | cut -d. -f1 | sed 's/v//')
    if [ "$NODE_VERSION" -lt 18 ]; then
        log_error "Node.js version 18+ required. Found: $(node --version)"
        exit 1
    fi

    # Install web-api dependencies
    if [ -d "${RAG_HOME}/services/web-api" ]; then
        log_info "Installing web-api dependencies..."
        cd "${RAG_HOME}/services/web-api"
        npm install --silent
    fi

    # Install web-ui dependencies
    if [ -d "${RAG_HOME}/services/web-ui" ]; then
        log_info "Installing web-ui dependencies..."
        cd "${RAG_HOME}/services/web-ui"
        npm install --silent
    fi

    log "Node.js dependencies installed"
}

# Install Weaviate
install_weaviate() {
    log "Installing Weaviate ${WEAVIATE_VERSION}..."

    # Create installation directory
    mkdir -p "${WEAVIATE_INSTALL_DIR}"
    mkdir -p "${WEAVIATE_DATA_DIR}"

    # Download and extract Weaviate
    log_info "Downloading Weaviate from ${WEAVIATE_URL}..."
    TEMP_DIR=$(mktemp -d)
    cd "${TEMP_DIR}"

    wget -q --show-progress "${WEAVIATE_URL}" -O weaviate.tar.gz
    tar -xzf weaviate.tar.gz

    # Move binary to installation directory
    mv weaviate "${WEAVIATE_INSTALL_DIR}/"
    chmod +x "${WEAVIATE_INSTALL_DIR}/weaviate"

    # Cleanup
    rm -rf "${TEMP_DIR}"

    # Set ownership
    chown -R rag:rag "${WEAVIATE_DATA_DIR}"

    log "Weaviate installed to ${WEAVIATE_INSTALL_DIR}"
}

# Create systemd service for Weaviate
create_weaviate_service() {
    log "Creating Weaviate systemd service..."

    cat > /etc/systemd/system/weaviate.service << EOF
[Unit]
Description=Weaviate Vector Database
After=network.target
Documentation=https://weaviate.io/developers/weaviate

[Service]
Type=simple
User=rag
Group=rag
WorkingDirectory=${WEAVIATE_INSTALL_DIR}
ExecStart=${WEAVIATE_INSTALL_DIR}/weaviate --host 0.0.0.0 --port ${WEAVIATE_PORT} --scheme http

# Environment variables for optimal performance
Environment="QUERY_DEFAULTS_LIMIT=25"
Environment="AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true"
Environment="PERSISTENCE_DATA_PATH=${WEAVIATE_DATA_DIR}"
Environment="DEFAULT_VECTORIZER_MODULE=none"
Environment="ENABLE_MODULES="
Environment="CLUSTER_HOSTNAME=node1"

# Performance settings - USE ALL RESOURCES!
Environment="GOGC=100"
Environment="LIMIT_RESOURCES=false"
Environment="PERSISTENCE_LSM_ACCESS_STRATEGY=mmap"
Environment="GOMEMLIMIT=200GiB"
Environment="PERSISTENCE_LSM_MAX_SEGMENT_SIZE=10GB"
Environment="GOMAXPROCS=140"

# Restart behavior
Restart=always
RestartSec=10s
StartLimitInterval=5min
StartLimitBurst=5

# Resource limits (optional - commented out to allow full usage)
# LimitNOFILE=65536
# LimitNPROC=4096

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=weaviate

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log "Weaviate service created"
}

# Create systemd service for RAG API (Python FastAPI)
create_rag_api_service() {
    log "Creating RAG API systemd service..."

    cat > /etc/systemd/system/rag-api.service << EOF
[Unit]
Description=RAG Python API (FastAPI)
After=network.target weaviate.service postgresql.service redis-server.service
Requires=weaviate.service postgresql.service redis-server.service

[Service]
Type=simple
User=rag
Group=rag
WorkingDirectory=${RAG_HOME}/services/rag-service
ExecStart=${RAG_HOME}/venv/bin/python ${RAG_HOME}/services/rag-api/main.py

# Environment variables
Environment="WEAVIATE_HOST=localhost"
Environment="WEAVIATE_PORT=${WEAVIATE_PORT}"
Environment="POSTGRES_HOST=localhost"
Environment="POSTGRES_PORT=${POSTGRES_PORT}"
Environment="POSTGRES_DB=${POSTGRES_DB}"
Environment="POSTGRES_USER=${POSTGRES_USER}"
Environment="POSTGRES_PASSWORD=${POSTGRES_PASSWORD}"
Environment="PYTHONPATH=${RAG_HOME}/services/rag-service"

# Restart behavior
Restart=always
RestartSec=5s
StartLimitInterval=5min
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rag-api

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log "RAG API service created"
}

# Create systemd service for Web API (Node.js Express)
create_web_api_service() {
    log "Creating Web API systemd service..."

    # Build TypeScript to JavaScript
    log_info "Building web-api TypeScript project..."
    cd "${RAG_HOME}/services/web-api"
    sudo -u rag npm run build
    cd "${RAG_HOME}"

    cat > /etc/systemd/system/web-api.service << EOF
[Unit]
Description=RAG Web API (Node.js Express)
After=network.target rag-api.service postgresql.service redis-server.service
Requires=rag-api.service postgresql.service redis-server.service

[Service]
Type=simple
User=rag
Group=rag
WorkingDirectory=${RAG_HOME}/services/web-api
ExecStart=/usr/bin/node dist/index.js

# Environment variables
Environment="NODE_ENV=production"
Environment="PORT=${WEB_API_PORT}"
Environment="AUTH_MODE=jwt"
Environment="JWT_SECRET=${JWT_SECRET}"
Environment="PYTHON_RAG_URL=http://localhost:${RAG_API_PORT}"
Environment="PYTHON_RAG_TIMEOUT=60000"
Environment="REDIS_URL=redis://localhost:${REDIS_PORT}"
Environment="DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}?sslmode=disable"
Environment="WEAVIATE_HOST=localhost"
Environment="WEAVIATE_PORT=${WEAVIATE_PORT}"
Environment="WEAVIATE_GRPC_PORT=${WEAVIATE_GRPC_PORT}"
Environment="WEAVIATE_SCHEME=http"
Environment="CORS_ORIGIN=http://localhost:5173,http://localhost:3001"
Environment="RATE_LIMIT_WINDOW_MS=60000"
Environment="RATE_LIMIT_MAX_REQUESTS=60"

# Restart behavior
Restart=always
RestartSec=5s
StartLimitInterval=5min
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=web-api

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log "Web API service created"
}

# Create systemd service for Web UI (React)
create_web_ui_service() {
    log "Creating Web UI systemd service..."

    # First, build the web UI
    if [ -d "${RAG_HOME}/services/web-ui" ]; then
        log_info "Building Web UI..."
        cd "${RAG_HOME}/services/web-ui"
        sudo -u rag npm run build
    fi

    cat > /etc/systemd/system/web-ui.service << EOF
[Unit]
Description=RAG Web UI (React/Vite)
After=network.target web-api.service

[Service]
Type=simple
User=rag
Group=rag
WorkingDirectory=${RAG_HOME}/services/web-ui
ExecStart=/usr/bin/npm run preview -- --host 0.0.0.0 --port ${WEB_UI_PORT}

# Environment variables
Environment="NODE_ENV=production"

# Restart behavior
Restart=always
RestartSec=5s
StartLimitInterval=5min
StartLimitBurst=5

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=web-ui

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    log "Web UI service created"
}

# Configure Nginx reverse proxy
configure_nginx() {
    log "Configuring Nginx reverse proxy..."

    cat > /etc/nginx/sites-available/rag-platform << 'EOF'
upstream rag_api {
    server localhost:8082;
}

upstream web_api {
    server localhost:3000;
}

upstream web_ui {
    server localhost:5173;
}

server {
    listen 80 default_server;
    listen [::]:80 default_server;

    server_name _;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Logging
    access_log /var/log/nginx/rag-access.log;
    error_log /var/log/nginx/rag-error.log;

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }

    # Web UI (React) - root
    location / {
        proxy_pass http://web_ui;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Web API (Express) - /api/v1
    location /api/v1/ {
        proxy_pass http://web_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for RAG operations
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # RAG API (FastAPI) - /rag/
    location /rag/ {
        proxy_pass http://rag_api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for RAG operations
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    # Weaviate API (optional direct access)
    location /weaviate/ {
        proxy_pass http://localhost:8080/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

    # Enable site
    ln -sf /etc/nginx/sites-available/rag-platform /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default

    # Test configuration
    nginx -t

    log "Nginx configured"
}

# Create environment file
create_env_file() {
    log "Creating environment configuration..."

    cat > "${RAG_HOME}/.env.local" << EOF
# RAG Platform - Local Server Configuration
# Generated: $(date)

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=${POSTGRES_PORT}
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

# Weaviate Configuration
WEAVIATE_HOST=localhost
WEAVIATE_PORT=${WEAVIATE_PORT}
WEAVIATE_GRPC_PORT=${WEAVIATE_GRPC_PORT}
WEAVIATE_SCHEME=http

# Redis Configuration
REDIS_URL=redis://localhost:${REDIS_PORT}

# Service Ports
RAG_API_PORT=${RAG_API_PORT}
WEB_API_PORT=${WEB_API_PORT}
WEB_UI_PORT=${WEB_UI_PORT}

# Application Configuration
RAG_ENV=production
RAG_LOG_LEVEL=INFO
RAG_DEFAULT_TENANT=development
RAG_DEFAULT_USER=admin
RAG_DEFAULT_LANGUAGE=hr

# Authentication
AUTH_MODE=jwt
JWT_SECRET=${JWT_SECRET}

# OpenRouter API (optional)
# OPENROUTER_API_KEY=your-key-here
EOF

    chown rag:rag "${RAG_HOME}/.env.local"
    chmod 600 "${RAG_HOME}/.env.local"

    log "Environment file created: ${RAG_HOME}/.env.local"
    log_info "JWT Secret: ${JWT_SECRET:0:20}... (truncated for security)"
}

# Start all services
start_services() {
    log "Starting all services..."

    # Start infrastructure services first
    systemctl start redis-server
    systemctl start postgresql
    systemctl start weaviate

    # Wait for services to be ready
    log_info "Waiting for services to initialize..."
    sleep 5

    # Start application services
    systemctl start rag-api
    sleep 3
    systemctl start web-api
    sleep 2
    systemctl start web-ui
    sleep 2

    # Start Nginx
    systemctl start nginx

    # Enable all services for auto-start
    systemctl enable redis-server
    systemctl enable postgresql
    systemctl enable weaviate
    systemctl enable rag-api
    systemctl enable web-api
    systemctl enable web-ui
    systemctl enable nginx

    log "All services started and enabled"
}

# Check service status
check_services() {
    log "Checking service status..."

    echo ""
    echo "=== Service Status ==="

    services=("postgresql" "redis-server" "weaviate" "rag-api" "web-api" "web-ui" "nginx")

    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            echo -e "  ${GREEN}✓${NC} $service: running"
        else
            echo -e "  ${RED}✗${NC} $service: stopped"
        fi
    done

    echo ""
    echo "=== Port Status ==="
    netstat -tlnp | grep -E "(${POSTGRES_PORT}|${REDIS_PORT}|${WEAVIATE_PORT}|${RAG_API_PORT}|${WEB_API_PORT}|${WEB_UI_PORT}|${NGINX_PORT})" || true

    echo ""
    echo "=== Access URLs ==="
    echo "  Web UI:        http://localhost (or http://$(hostname -I | awk '{print $1}'))"
    echo "  Web API:       http://localhost:${WEB_API_PORT}"
    echo "  RAG API:       http://localhost:${RAG_API_PORT}"
    echo "  Weaviate:      http://localhost:${WEAVIATE_PORT}"
    echo "  PostgreSQL:    localhost:${POSTGRES_PORT}"
    echo "  Redis:         localhost:${REDIS_PORT}"
}

# Main installation flow
main() {
    log "=== RAG Platform - Local Server Setup ==="
    log "Installation started at: $(date)"
    echo ""

    check_root
    check_system
    echo ""

    # Generate secure JWT secret
    log "Generating secure JWT secret..."
    JWT_SECRET=$(generate_jwt_secret)
    log "JWT secret generated (will be saved in .env.local)"
    echo ""

    # System updates
    update_system
    echo ""

    # Install infrastructure
    install_postgresql
    configure_postgresql
    echo ""

    install_redis
    echo ""

    install_nginx
    echo ""

    # Install Weaviate
    install_weaviate
    create_weaviate_service
    echo ""

    # Install application dependencies
    install_python_deps
    echo ""

    install_node_deps
    echo ""

    # Create systemd services
    create_rag_api_service
    create_web_api_service
    create_web_ui_service
    echo ""

    # Configure Nginx
    configure_nginx
    echo ""

    # Create environment file
    create_env_file
    echo ""

    # Start services
    start_services
    echo ""

    # Check status
    check_services
    echo ""

    log "=== Installation Complete ==="
    log "Finished at: $(date)"
    echo ""
    log_info "Next steps:"
    echo "  1. Review configuration: ${RAG_HOME}/.env.local"
    echo "  2. Check service logs: journalctl -u <service-name> -f"
    echo "  3. Access web UI: http://localhost or http://$(hostname -I | awk '{print $1}')"
    echo "  4. Run CLI commands: cd ${RAG_HOME} && source venv/bin/activate && python rag.py --help"
    echo ""
    log_warning "IMPORTANT: Change JWT_SECRET and other secrets in production!"
}

# Run main installation
main
