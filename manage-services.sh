#!/bin/bash
#
# RAG Platform - Service Management Script
# Quick commands to manage all services
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SERVICES=(
    "postgresql"
    "redis-server"
    "weaviate"
    "rag-api"
    "web-api"
    "web-ui"
    "nginx"
)

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1"
}

# Check if running as root
require_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This command requires root privileges (use sudo)"
        exit 1
    fi
}

# Start all services
start_all() {
    require_root
    log "Starting all services..."

    for service in "${SERVICES[@]}"; do
        echo -n "  Starting $service... "
        if systemctl start "$service" 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
        fi
    done

    log "All services started"
}

# Stop all services
stop_all() {
    require_root
    log "Stopping all services..."

    # Stop in reverse order
    for ((i=${#SERVICES[@]}-1; i>=0; i--)); do
        service="${SERVICES[$i]}"
        echo -n "  Stopping $service... "
        if systemctl stop "$service" 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${YELLOW}⚠${NC}"
        fi
    done

    log "All services stopped"
}

# Restart all services
restart_all() {
    require_root
    log "Restarting all services..."

    stop_all
    sleep 2
    start_all

    log "All services restarted"
}

# Show status of all services
status_all() {
    echo ""
    echo "=== Service Status ==="
    echo ""

    for service in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} $service: ${GREEN}running${NC}"
        else
            echo -e "  ${RED}●${NC} $service: ${RED}stopped${NC}"
        fi
    done

    echo ""
    echo "=== Port Status ==="
    echo ""
    netstat -tlnp 2>/dev/null | grep -E "(5434|6379|8080|8082|3000|5173|:80)" | awk '{print "  " $4 " -> " $7}' || echo "  netstat not available"

    echo ""
    echo "=== Recent Errors (last 10 lines per service) ==="
    echo ""
    for service in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$service" 2>/dev/null; then
            errors=$(journalctl -u "$service" -p err -n 10 --no-pager --since "1 hour ago" 2>/dev/null | grep -v "^--" | tail -5)
            if [ -n "$errors" ]; then
                echo -e "${YELLOW}  $service:${NC}"
                echo "$errors" | sed 's/^/    /'
                echo ""
            fi
        fi
    done
}

# Show logs for a specific service
logs() {
    local service=$1
    local lines=${2:-50}

    if [ -z "$service" ]; then
        log_error "Usage: $0 logs <service-name> [lines]"
        echo ""
        echo "Available services:"
        for s in "${SERVICES[@]}"; do
            echo "  - $s"
        done
        exit 1
    fi

    log "Showing last $lines lines for $service..."
    journalctl -u "$service" -n "$lines" -f
}

# Follow logs for all services
logs_all() {
    log "Following logs for all services (Ctrl+C to stop)..."
    journalctl -u postgresql -u redis-server -u weaviate -u rag-api -u web-api -u web-ui -u nginx -f
}

# Enable auto-start for all services
enable_all() {
    require_root
    log "Enabling auto-start for all services..."

    for service in "${SERVICES[@]}"; do
        echo -n "  Enabling $service... "
        if systemctl enable "$service" 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
        fi
    done

    log "Auto-start enabled"
}

# Disable auto-start for all services
disable_all() {
    require_root
    log "Disabling auto-start for all services..."

    for service in "${SERVICES[@]}"; do
        echo -n "  Disabling $service... "
        if systemctl disable "$service" 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${YELLOW}⚠${NC}"
        fi
    done

    log "Auto-start disabled"
}

# Health check
health_check() {
    log "Performing health checks..."
    echo ""

    # PostgreSQL
    echo -n "  PostgreSQL... "
    if pg_isready -h localhost -p 5434 -U raguser -d ragdb >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Redis
    echo -n "  Redis... "
    if redis-cli -p 6379 ping >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Weaviate
    echo -n "  Weaviate... "
    if curl -sf http://localhost:8080/v1/.well-known/ready >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # RAG API
    echo -n "  RAG API... "
    if curl -sf http://localhost:8082/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Web API
    echo -n "  Web API... "
    if curl -sf http://localhost:3000/api/v1/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Web UI
    echo -n "  Web UI... "
    if curl -sf http://localhost:5173 >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    # Nginx
    echo -n "  Nginx... "
    if curl -sf http://localhost/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi

    echo ""
}

# Restart a specific service
restart_service() {
    require_root
    local service=$1

    if [ -z "$service" ]; then
        log_error "Usage: $0 restart <service-name>"
        exit 1
    fi

    log "Restarting $service..."
    systemctl restart "$service"
    sleep 1
    systemctl status "$service" --no-pager -l
}

# Show help
show_help() {
    cat << EOF
RAG Platform - Service Management

Usage: $(basename "$0") <command> [options]

Commands:
  start           Start all services
  stop            Stop all services
  restart         Restart all services
  status          Show status of all services
  logs <service>  Show logs for a specific service
  logs-all        Follow logs for all services
  health          Perform health checks on all services
  enable          Enable auto-start for all services
  disable         Disable auto-start for all services
  restart-one <service>  Restart a specific service

Available Services:
  postgresql      PostgreSQL database (port 5434)
  redis-server    Redis cache (port 6379)
  weaviate        Weaviate vector database (port 8080)
  rag-api         RAG Python API (port 8082)
  web-api         Web Node.js API (port 3000)
  web-ui          React Web UI (port 5173)
  nginx           Nginx reverse proxy (port 80)

Examples:
  sudo ./manage-services.sh start
  sudo ./manage-services.sh status
  sudo ./manage-services.sh logs rag-api
  sudo ./manage-services.sh logs-all
  sudo ./manage-services.sh health
  sudo ./manage-services.sh restart-one weaviate

Note: Most commands require root privileges (sudo)
EOF
}

# Main command dispatcher
case "${1:-}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        restart_all
        ;;
    status)
        status_all
        ;;
    logs)
        logs "$2" "$3"
        ;;
    logs-all)
        logs_all
        ;;
    health)
        health_check
        ;;
    enable)
        enable_all
        ;;
    disable)
        disable_all
        ;;
    restart-one)
        restart_service "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: ${1:-}"
        echo ""
        show_help
        exit 1
        ;;
esac
