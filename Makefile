# Makefile for Automated Trading System
# Provides commands for building, testing, and managing the trading system

.PHONY: help build build-base build-services test test-unit test-integration \
		test-real-integration test-integration-quick test-integration-clean \
		start stop restart status logs clean clean-volumes clean-images \
		setup init-db migrate lint format check-env health backup restore \
		dev prod debug shell deploy deploy-zero-downtime maintenance security \
		export-data audit monitor-security compliance

# Default target
help: ## Show this help message
	@echo "Automated Trading System - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup: ## Initial project setup
	@echo "Setting up automated trading system..."
	@cp .env.example .env
	@echo "✓ Created .env file from template"
	@echo "⚠️  Please edit .env file with your API keys and configuration"
	@mkdir -p data/logs data/parquet
	@echo "✓ Created data directories"
	@chmod +x scripts/setup.sh
	@echo "✓ Made scripts executable"
	@echo "Run 'make check-env' to validate your configuration"

check-env: ## Check environment configuration
	@echo "Checking environment configuration..."
	@python3 -c "from shared.config import Config; config = Config(); print('✓ Configuration is valid')" || \
		(echo "✗ Configuration validation failed. Please check your .env file"; exit 1)

init-db: ## Initialize database schema
	@echo "Initializing database..."
	@docker-compose up -d postgres redis
	@sleep 10
	@docker-compose exec postgres psql -U $$DB_USER -d $$DB_NAME -f /docker-entrypoint-initdb.d/init_db.sql
	@echo "✓ Database initialized"

# =============================================================================
# BUILD COMMANDS
# =============================================================================

build: build-base build-services ## Build all Docker images

build-base: ## Build base Docker image
	@echo "Building base Docker image..."
	@docker build -t trading-system-base:latest -f Dockerfile.base .
	@echo "✓ Base image built successfully"

build-services: ## Build all service Docker images
	@echo "Building service Docker images..."
	@docker-compose build --parallel
	@echo "✓ All service images built successfully"

rebuild: clean-images build ## Clean and rebuild all images

# =============================================================================
# SERVICE MANAGEMENT
# =============================================================================

start: ## Start all services
	@echo "Starting trading system services..."
	@docker-compose up -d
	@echo "✓ All services started"
	@make status

stop: ## Stop all services
	@echo "Stopping trading system services..."
	@docker-compose down
	@echo "✓ All services stopped"

restart: stop start ## Restart all services

status: ## Show service status
	@echo "Trading System Service Status:"
	@echo "=============================="
	@docker-compose ps

logs: ## Show logs for all services
	@docker-compose logs -f

logs-service: ## Show logs for specific service (usage: make logs-service SERVICE=data_collector)
	@docker-compose logs -f $(SERVICE)

health: ## Check health of all services
	@echo "Checking service health..."
	@echo "=========================="
	@services="data_collector strategy_engine risk_manager trade_executor scheduler"; \
	for service in $$services; do \
		port=$$(docker-compose port $$service 800* 2>/dev/null | cut -d: -f2); \
		if [ -n "$$port" ]; then \
			echo -n "$$service: "; \
			curl -s -f http://localhost:$$port/health >/dev/null && echo "✓ Healthy" || echo "✗ Unhealthy"; \
		else \
			echo "$$service: ✗ Not running"; \
		fi; \
	done

# =============================================================================
# DEVELOPMENT COMMANDS
# =============================================================================

dev: ## Start services in development mode with hot reload
	@echo "Starting development environment..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@make status

debug: ## Start services with debug logging
	@echo "Starting services with debug logging..."
	@LOG_LEVEL=DEBUG docker-compose up -d
	@make logs

shell: ## Get shell access to a service (usage: make shell SERVICE=data_collector)
	@docker-compose exec $(SERVICE) /bin/bash

shell-db: ## Get PostgreSQL shell
	@docker-compose exec postgres psql -U $$DB_USER -d $$DB_NAME

shell-redis: ## Get Redis CLI shell
	@docker-compose exec redis redis-cli

# =============================================================================
# TESTING
# =============================================================================

test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests
	@echo "Running unit tests..."
	@docker-compose exec data_collector python -m pytest tests/ -v
	@docker-compose exec strategy_engine python -m pytest tests/ -v
	@docker-compose exec risk_manager python -m pytest tests/ -v
	@docker-compose exec trade_executor python -m pytest tests/ -v
	@docker-compose exec scheduler python -m pytest tests/ -v
	@echo "✓ Unit tests completed"

test-integration: ## Run integration tests (mocked)
	@echo "Running integration tests..."
	@python -m pytest tests/integration/ -v
	@echo "✓ Integration tests completed"

test-real-integration: ## Run real system integration tests with actual services
	@echo "Running real system integration tests..."
	@COMPOSE_PROJECT_NAME=trading_system_integration venv/bin/python scripts/run_integration_tests.py
	@echo "✓ Real integration tests completed"

test-integration-quick: ## Run quick real integration tests (faster, limited scope)
	@echo "Running quick real integration tests..."
	@COMPOSE_PROJECT_NAME=trading_system_integration venv/bin/python scripts/run_integration_tests.py --quick
	@echo "✓ Quick integration tests completed"

test-integration-setup: ## Setup environment for integration tests
	@echo "Setting up integration test environment..."
	@cp .env.integration.example .env.integration || cp .env .env.integration
	@echo "⚠️  Please edit .env.integration with test credentials"
	@mkdir -p integration_test_data/logs integration_test_data/data integration_test_data/backups
	@echo "✓ Integration test environment setup completed"

test-integration-infrastructure: ## Start only integration test infrastructure
	@echo "Starting integration test infrastructure..."
	@COMPOSE_PROJECT_NAME=trading_system_integration docker compose -f docker-compose.integration.yml up -d postgres_integration redis_integration
	@echo "✓ Integration test infrastructure started"

test-integration-stop: ## Stop integration test infrastructure
	@echo "Stopping integration test infrastructure..."
	@COMPOSE_PROJECT_NAME=trading_system_integration docker compose -f docker-compose.integration.yml down
	@echo "✓ Integration test infrastructure stopped"

test-integration-clean: ## Clean integration test environment
	@echo "Cleaning integration test environment..."
	@COMPOSE_PROJECT_NAME=trading_system_integration docker compose -f docker-compose.integration.yml down --volumes --remove-orphans
	@rm -rf integration_test_data/
	@echo "✓ Integration test environment cleaned"

test-integration-logs: ## Show integration test logs
	@echo "Integration test logs:"
	@echo "======================"
	@docker-compose -f docker-compose.integration.yml logs -f

test-integration-monitor: ## Monitor integration test Redis messages
	@echo "Monitoring integration test Redis messages..."
	@docker-compose -f docker-compose.integration.yml up integration_monitor

test-integration-validate: ## Validate integration test credentials
	@echo "Validating integration test credentials..."
	@venv/bin/python scripts/validate_integration_credentials.py

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@docker-compose exec data_collector python -m pytest tests/ --cov=src --cov-report=html
	@echo "✓ Coverage report generated"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run linting checks
	@echo "Running linting checks..."
	@flake8 . --exclude=.venv --ignore=E501,W503
	@mypy .
	@echo "✓ Linting checks completed"

format: ## Format code
	@echo "Formatting code..."
	@black .
	@isort .
	@echo "✓ Code formatted"

format-check:
	@echo "Running format checks..."
	@black --check .
	@isort --check-only .
	@echo "✓ Format checking completed"

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

migrate: ## Run database migrations
	@echo "Running database migrations..."
	@docker-compose exec scheduler python -m src.database.migrate
	@echo "✓ Database migrations completed"

backup: ## Backup database
	@echo "Creating database backup..."
	@mkdir -p backups
	@docker-compose exec postgres pg_dump -U $$DB_USER $$DB_NAME > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "✓ Database backup created"

restore: ## Restore database from backup (usage: make restore BACKUP=backup_20231201_120000.sql)
	@echo "Restoring database from backup..."
	@docker-compose exec -T postgres psql -U $$DB_USER -d $$DB_NAME < backups/$(BACKUP)
	@echo "✓ Database restored"

reset-db: ## Reset database (WARNING: This will delete all data)
	@echo "⚠️  This will delete all database data. Are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	@docker-compose down
	@docker volume rm full-ai-trader_postgres_data || true
	@docker-compose up -d postgres redis
	@sleep 10
	@make init-db
	@echo "✓ Database reset completed"

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

clean-data: ## Clean old data files
	@echo "Cleaning old data files..."
	@find data/parquet -name "*.parquet" -mtime +30 -delete 2>/dev/null || true
	@find data/logs -name "*.log.*" -mtime +7 -delete 2>/dev/null || true
	@echo "✓ Old data files cleaned"

backup-data: ## Backup data files
	@echo "Creating data backup..."
	@mkdir -p backups
	@tar -czf backups/data_backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/
	@echo "✓ Data backup created"

# =============================================================================
# MONITORING AND DIAGNOSTICS
# =============================================================================

monitor: ## Show real-time service metrics
	@echo "Monitoring services (Press Ctrl+C to stop)..."
	@watch -n 5 'make status && echo "" && make health'

stats: ## Show system statistics
	@echo "System Statistics:"
	@echo "=================="
	@echo "Docker containers:"
	@docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
	@echo ""
	@echo "Disk usage:"
	@df -h data/
	@echo ""
	@echo "Recent logs (last 10 lines):"
	@tail -n 10 data/logs/*.log 2>/dev/null || echo "No log files found"

# =============================================================================
# PRODUCTION COMMANDS
# =============================================================================

prod: ## Start services in production mode
	@echo "Starting production environment..."
	@ENVIRONMENT=production docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@make status

deploy: build prod ## Build and deploy to production

# =============================================================================
# CLEANUP COMMANDS
# =============================================================================

clean: ## Clean up temporary files and containers
	@echo "Cleaning up..."
	@docker-compose down
	@docker system prune -f
	@echo "✓ Cleanup completed"

clean-volumes: ## Remove Docker volumes (WARNING: This will delete all data)
	@echo "⚠️  This will delete all database and cache data. Are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	@docker-compose down -v
	@docker volume prune -f
	@echo "✓ Volumes cleaned"

clean-images: ## Remove Docker images
	@echo "Removing Docker images..."
	@docker-compose down --rmi all
	@docker image prune -f
	@echo "✓ Images cleaned"

clean-all: clean-volumes clean-images clean ## Complete cleanup (WARNING: Deletes everything)

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

ps: ## Show running containers
	@docker-compose ps

top: ## Show container resource usage
	@docker-compose top

exec: ## Execute command in service container (usage: make exec SERVICE=data_collector COMMAND="ls -la")
	@docker-compose exec $(SERVICE) $(COMMAND)

update: ## Update Docker images
	@echo "Updating Docker images..."
	@docker-compose pull
	@echo "✓ Images updated. Run 'make restart' to use updated images"

# =============================================================================
# BACKTEST AND ANALYSIS
# =============================================================================

backtest: ## Run backtesting (usage: make backtest STRATEGY=moving_average SYMBOL=AAPL)
	@echo "Running backtest..."
	@python scripts/backtest.py --strategy=$(STRATEGY) --symbol=$(SYMBOL)

analyze: ## Run analysis on trading performance
	@echo "Running performance analysis..."
	@docker-compose exec scheduler python -m src.analysis.performance
	@echo "✓ Analysis completed"

# =============================================================================
# INSTALLATION AND DEPENDENCIES
# =============================================================================

install-deps: ## Install Python dependencies locally (for development)
	@echo "Creating virtual environment..."
	@python -m venv venv
	@echo "Activating virtual environment..."
	@venv/bin/activate
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt
	@pip install -r requirements.ci.txt
	@pip install -r services/data_collector/requirements.txt
	@pip install -r services/export_service/requirements.txt
	@pip install -r services/maintenance_service/requirements.txt
	@pip install -r services/notification_service/requirements.txt
	@pip install -r services/risk_manager/requirements.txt
	@pip install -r services/scheduler/requirements.txt
	@pip install -r services/strategy_engine/requirements.txt
	@pip install -r services/trade_executor/requirements.txt
	@echo "✓ Dependencies installed"


install-dev-deps: ## Install development dependencies
	@echo "Installing development dependencies..."
	@pip install pytest pytest-cov black isort flake8 mypy pre-commit
	@echo "✓ Development dependencies installed"

# =============================================================================
# HELP AND DOCUMENTATION
# =============================================================================

docs: ## Generate documentation
	@echo "Generating documentation..."
	@docker-compose exec data_collector python -m pdoc --html --output-dir /app/docs src/
	@echo "✓ Documentation generated in docs/"

validate: ## Validate configuration and setup
	@echo "Validating system setup..."
	@make check-env
	@make lint
	@echo "✓ System validation completed"

version: ## Show version information
	@echo "Automated Trading System"
	@echo "Version: 1.0.0"
	@echo "Docker Compose: $$(docker-compose --version)"
	@echo "Docker: $$(docker --version)"
	@echo "Python: $$(python3 --version)"

# =============================================================================
# QUICK COMMANDS
# =============================================================================

up: start ## Alias for start
down: stop ## Alias for stop
build-up: build start ## Build and start services
fresh: clean build start ## Clean build and start
quick-test: build test ## Quick build and test
