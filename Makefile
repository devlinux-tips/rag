# Multilingual RAG Platform - Local Server Makefile
# Updated for native deployment (no Docker)

.PHONY: help setup build start stop restart status logs health clean install-deps prisma-generate dev

.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

#==============================================================================
# Help
#==============================================================================

help: ## Show this help message
	@echo "$(BLUE)RAG Platform - Local Server Management$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Installation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## .*$$/ {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Database (Prisma):$(NC)"
	@echo "  $(YELLOW)prisma-generate$(NC)          Generate Prisma client"
	@echo "  $(YELLOW)prisma-migrate$(NC)           Apply migrations (production)"
	@echo "  $(YELLOW)prisma-push$(NC)              Push schema without migration"
	@echo "  $(YELLOW)prisma-studio$(NC)            Open database GUI"
	@echo "  $(YELLOW)prisma-validate$(NC)          Validate schema"
	@echo "  $(YELLOW)prisma-status$(NC)            Show migration status"
	@echo ""
	@echo "$(GREEN)Service Management:$(NC)"
	@echo "  $(YELLOW)start$(NC)                    Start all services"
	@echo "  $(YELLOW)stop$(NC)                     Stop all services"
	@echo "  $(YELLOW)restart$(NC)                  Restart all services"
	@echo "  $(YELLOW)status$(NC)                   Check service status"
	@echo "  $(YELLOW)logs SERVICE$(NC)             View logs for a service"
	@echo "  $(YELLOW)health$(NC)                   Run health checks"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  $(YELLOW)dev$(NC)                      Start all services"
	@echo "  $(YELLOW)build$(NC)                    Build all services"
	@echo "  $(YELLOW)clean$(NC)                    Clean build artifacts"
	@echo ""

#==============================================================================
# Installation & Setup
#==============================================================================

install: ## Run full server installation
	@echo "$(GREEN)Running full server installation...$(NC)"
	@sudo ./setup-local-server.sh

install-deps: ## Install Python and Node dependencies only
	@echo "$(GREEN)Installing Python dependencies...$(NC)"
	@test -d venv || python3 -m venv venv
	@. venv/bin/activate && pip install -q -r requirements.txt
	@echo "$(GREEN)Installing web-api dependencies...$(NC)"
	@cd services/web-api && npm install
	@echo "$(GREEN)Installing web-ui dependencies...$(NC)"
	@cd services/web-ui && npm install
	@echo "$(GREEN)Dependencies installed!$(NC)"

setup: install-deps sync-env prisma-migrate prisma-generate ## Setup environment (deps + sync env + Prisma migrate + generate)
	@echo "$(GREEN)Setup complete!$(NC)"

sync-env: ## Sync environment files to all services
	@echo "$(GREEN)Syncing environment files...$(NC)"
	@./scripts/sync-env.sh

#==============================================================================
# Build Commands
#==============================================================================

build: build-web-api build-web-ui ## Build all services

build-web-api: prisma-generate ## Build web-api (TypeScript)
	@echo "$(GREEN)Building web-api...$(NC)"
	@cd services/web-api && npm run build
	@echo "$(GREEN)web-api built successfully!$(NC)"

build-web-ui: ## Build web-ui (React/Vite)
	@echo "$(GREEN)Building web-ui...$(NC)"
	@cd services/web-ui && npm run build
	@echo "$(GREEN)web-ui built successfully!$(NC)"

prisma-generate: ## Generate Prisma client
	@echo "$(GREEN)Generating Prisma client...$(NC)"
	@cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma generate

prisma-migrate: ## Run Prisma migrations (production)
	@echo "$(GREEN)Running Prisma migrations...$(NC)"
	@cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma migrate deploy

prisma-migrate-dev: ## Create and apply new migration (development)
	@echo "$(GREEN)Creating new Prisma migration...$(NC)"
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME parameter required$(NC)"; \
		echo "Usage: make prisma-migrate-dev NAME=migration_name"; \
		exit 1; \
	fi
	@cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma migrate dev --name $(NAME)

prisma-push: ## Push schema changes to database (without migration)
	@echo "$(GREEN)Pushing Prisma schema to database...$(NC)"
	@cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma db push

prisma-reset: ## Reset database (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will delete all data in the database!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma migrate reset --force; \
	else \
		echo "$(YELLOW)Operation cancelled$(NC)"; \
	fi

prisma-studio: ## Open Prisma Studio (database GUI)
	@echo "$(GREEN)Opening Prisma Studio...$(NC)"
	@cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma studio

prisma-format: ## Format Prisma schema file
	@echo "$(GREEN)Formatting Prisma schema...$(NC)"
	@cd services/web-api && npx prisma format

prisma-validate: ## Validate Prisma schema
	@echo "$(GREEN)Validating Prisma schema...$(NC)"
	@cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma validate

#==============================================================================
# Service Management (requires sudo)
#==============================================================================

start: ## Start all services
	@sudo ./manage-services.sh start

stop: ## Stop all services
	@sudo ./manage-services.sh stop

restart: ## Restart all services
	@sudo ./manage-services.sh restart

restart-one: ## Restart a specific service (usage: make restart-one SERVICE=rag-api)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		echo "Usage: make restart-one SERVICE=<service-name>"; \
		echo "Available services: postgresql, redis-server, weaviate, rag-api, web-api, web-ui, nginx"; \
		exit 1; \
	fi
	@sudo ./manage-services.sh restart-one $(SERVICE)

status: ## Check status of all services
	@sudo ./manage-services.sh status

logs: ## View logs for a service (usage: make logs SERVICE=rag-api)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		echo "Usage: make logs SERVICE=<service-name>"; \
		echo "Available services: postgresql, redis-server, weaviate, rag-api, web-api, web-ui, nginx"; \
		exit 1; \
	fi
	@sudo ./manage-services.sh logs $(SERVICE)

logs-all: ## Follow logs for all services
	@sudo ./manage-services.sh logs-all

health: ## Run health checks on all services
	@sudo ./manage-services.sh health

enable: ## Enable auto-start on boot for all services
	@sudo ./manage-services.sh enable

disable: ## Disable auto-start on boot for all services
	@sudo ./manage-services.sh disable

#==============================================================================
# Development Commands
#==============================================================================

dev: ## Start all services (alias for start)
	@$(MAKE) start

dev-web-api: ## Run web-api in development mode (hot reload)
	@echo "$(GREEN)Starting web-api in development mode...$(NC)"
	@cd services/web-api && npm run dev

dev-web-ui: ## Run web-ui in development mode (hot reload)
	@echo "$(GREEN)Starting web-ui in development mode...$(NC)"
	@cd services/web-ui && npm run dev

#==============================================================================
# Testing
#==============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running Python tests...$(NC)"
	@. venv/bin/activate && python python_test_runner.py

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@. venv/bin/activate && python python_test_runner.py --coverage

test-category: ## Run specific test category (usage: make test-category CATEGORY=rag-service)
	@if [ -z "$(CATEGORY)" ]; then \
		echo "$(RED)Error: CATEGORY parameter required$(NC)"; \
		echo "Usage: make test-category CATEGORY=<category-name>"; \
		exit 1; \
	fi
	@. venv/bin/activate && python python_test_runner.py --category $(CATEGORY)

#==============================================================================
# Cleanup
#==============================================================================

clean: clean-python clean-node clean-logs ## Clean all build artifacts

clean-python: ## Clean Python build artifacts
	@echo "$(GREEN)Cleaning Python artifacts...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-node: ## Clean Node.js build artifacts
	@echo "$(GREEN)Cleaning Node.js artifacts...$(NC)"
	@cd services/web-api && rm -rf dist/ node_modules/.cache/ 2>/dev/null || true
	@cd services/web-ui && rm -rf dist/ node_modules/.cache/ 2>/dev/null || true

clean-logs: ## Clean log files
	@echo "$(GREEN)Cleaning log files...$(NC)"
	@rm -rf logs/*.log 2>/dev/null || true

clean-all: clean ## Clean everything including dependencies (nuclear option)
	@echo "$(YELLOW)WARNING: This will remove node_modules and venv!$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		rm -rf venv services/web-api/node_modules services/web-ui/node_modules; \
		echo "$(GREEN)Deep clean complete!$(NC)"; \
	else \
		echo ""; \
		echo "Cancelled."; \
	fi

#==============================================================================
# Database Management
#==============================================================================

db-shell: ## Connect to PostgreSQL database
	@psql -h localhost -p 5434 -U raguser -d ragdb

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy ALL database data!$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		sudo -u postgres psql -c "DROP DATABASE IF EXISTS ragdb;"; \
		sudo -u postgres psql -c "DROP USER IF EXISTS raguser;"; \
		sudo -u postgres psql -c "CREATE USER raguser WITH PASSWORD 'x|B&h@p4F@o|k6t;~X]1A((Z.,RG';"; \
		sudo -u postgres psql -c "CREATE DATABASE ragdb OWNER raguser;"; \
		cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma migrate deploy; \
		echo "$(GREEN)Database reset complete!$(NC)"; \
	else \
		echo ""; \
		echo "Cancelled."; \
	fi

db-test: ## Test database connection
	@echo "$(GREEN)Testing database connection...$(NC)"
	@cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma db pull --force 2>/dev/null && echo "$(GREEN)✓ Database connection successful$(NC)" || echo "$(RED)✗ Database connection failed$(NC)"

prisma-status: ## Show Prisma migration status
	@echo "$(BLUE)Prisma Migration Status:$(NC)"
	@cd services/web-api && eval "$$(grep '^DATABASE_URL=' ../../.env | sed 's/^/export /')" && npx prisma migrate status

#==============================================================================
# Information
#==============================================================================

info: ## Show system information
	@echo "$(BLUE)RAG Platform - System Information$(NC)"
	@echo ""
	@echo "$(GREEN)Paths:$(NC)"
	@echo "  RAG Home:     $(shell pwd)"
	@echo "  Python venv:  $(shell pwd)/venv"
	@echo "  Weaviate:     /opt/weaviate"
	@echo ""
	@echo "$(GREEN)Services:$(NC)"
	@echo "  Web UI:       http://localhost:5173"
	@echo "  Web API:      http://localhost:3000"
	@echo "  RAG API:      http://localhost:8082"
	@echo "  Weaviate:     http://localhost:8080"
	@echo "  PostgreSQL:   localhost:5434"
	@echo "  Redis:        localhost:6379"
	@echo "  Nginx:        http://localhost (reverse proxy)"
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@echo "  Environment:  .env.local"
	@echo "  JWT Secret:   $(shell grep '^JWT_SECRET=' .env.local 2>/dev/null | cut -c1-35)..."
	@echo ""

version: ## Show version information
	@echo "RAG Platform - Local Server"
	@echo "Python: $(shell python3 --version)"
	@echo "Node.js: $(shell node --version)"
	@echo "PostgreSQL: $(shell psql --version | head -1)"
	@echo "Redis: $(shell redis-server --version | head -1)"
	@echo "Nginx: $(shell nginx -v 2>&1)"

#==============================================================================
# SystemD Management
#==============================================================================

systemd-reload: ## Reload systemd daemon
	@echo "$(GREEN)Reloading systemd daemon...$(NC)"
	@sudo systemctl daemon-reload

restart-api: ## Restart both API services (rag-api + web-api)
	@echo "$(GREEN)Restarting API services...$(NC)"
	@sudo systemctl restart rag-api web-api
	@$(MAKE) status

#==============================================================================
# Convenience Aliases
#==============================================================================

up: start ## Alias for start
down: stop ## Alias for stop
ps: status ## Alias for status
reload: systemd-reload ## Alias for systemd-reload
