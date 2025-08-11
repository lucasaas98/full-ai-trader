#!/bin/bash

# Automated Trading System - Setup Script
# This script initializes the trading system environment and dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo -e "${BLUE}"
echo "================================================================"
echo "    Automated Trading System - Setup Script"
echo "================================================================"
echo -e "${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root for security reasons"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check minimum version
check_version() {
    local tool=$1
    local required_version=$2
    local current_version=$3

    if [ "$(printf '%s\n' "$required_version" "$current_version" | sort -V | head -n1)" = "$required_version" ]; then
        return 0
    else
        return 1
    fi
}

# Check system requirements
log_info "Checking system requirements..."

# Check Docker
if ! command_exists docker; then
    log_error "Docker is not installed. Please install Docker first:"
    echo "  - Linux: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
    echo "  - macOS: Download Docker Desktop from https://docker.com"
    echo "  - Windows: Download Docker Desktop from https://docker.com"
    exit 1
fi

DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -n1)
if ! check_version "20.10.0" "$DOCKER_VERSION"; then
    log_warning "Docker version $DOCKER_VERSION detected. Recommended: 20.10.0 or higher"
else
    log_success "Docker $DOCKER_VERSION detected"
fi

# Check Docker Compose
if ! command_exists docker-compose; then
    log_error "Docker Compose is not installed. Please install it:"
    echo "  sudo curl -L \"https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose"
    echo "  sudo chmod +x /usr/local/bin/docker-compose"
    exit 1
fi

COMPOSE_VERSION=$(docker-compose --version | grep -oP '\d+\.\d+\.\d+' | head -n1)
if ! check_version "2.0.0" "$COMPOSE_VERSION"; then
    log_warning "Docker Compose version $COMPOSE_VERSION detected. Recommended: 2.0.0 or higher"
else
    log_success "Docker Compose $COMPOSE_VERSION detected"
fi

# Check Python
if ! command_exists python3; then
    log_error "Python 3 is not installed. Please install Python 3.9 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+\.\d+')
if ! check_version "3.9.0" "$PYTHON_VERSION"; then
    log_error "Python $PYTHON_VERSION detected. Python 3.9 or higher is required"
    exit 1
else
    log_success "Python $PYTHON_VERSION detected"
fi

# Check pip
if ! command_exists pip3; then
    log_error "pip3 is not installed. Please install pip3"
    exit 1
fi

# Check make
if ! command_exists make; then
    log_error "make is not installed. Please install make"
    echo "  - Ubuntu/Debian: sudo apt-get install build-essential"
    echo "  - CentOS/RHEL: sudo yum groupinstall 'Development Tools'"
    echo "  - macOS: xcode-select --install"
    exit 1
fi

# Check curl
if ! command_exists curl; then
    log_error "curl is not installed. Please install curl"
    exit 1
fi

# Check git
if ! command_exists git; then
    log_warning "git is not installed. Some features may not work properly"
fi

log_success "All system requirements met"

# Create directory structure
log_info "Creating directory structure..."

# Ensure all necessary directories exist
mkdir -p data/logs
mkdir -p data/parquet
mkdir -p data/backups
mkdir -p data/exports
mkdir -p backups
mkdir -p logs
mkdir -p tests/integration
mkdir -p docs

log_success "Directory structure created"

# Setup environment file
log_info "Setting up environment configuration..."

if [ ! -f .env ]; then
    cp .env.example .env
    log_success "Created .env file from template"
    log_warning "Please edit .env file with your API keys and configuration before starting services"
else
    log_info ".env file already exists, skipping creation"
fi

# Create .gitignore if it doesn't exist
log_info "Setting up .gitignore..."

if [ ! -f .gitignore ]; then
    cat > .gitignore << 'EOF'
# Environment files
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/
data/logs/*.log*

# Data files
data/parquet/*.parquet
data/exports/*
backups/*.sql
backups/*.tar.gz

# Docker
.docker/

# Jupyter
.ipynb_checkpoints/

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Testing
coverage.xml
*.cover
.hypothesis/

# Documentation
docs/_build/

# Temporary files
*.tmp
*.temp
.cache/
EOF
    log_success "Created .gitignore file"
else
    log_info ".gitignore file already exists"
fi

# Set up pre-commit hooks (optional)
log_info "Setting up development tools..."

if command_exists pip3; then
    log_info "Installing pre-commit hooks..."
    pip3 install pre-commit 2>/dev/null || log_warning "Failed to install pre-commit (optional)"

    # Create pre-commit config
    if [ ! -f .pre-commit-config.yaml ]; then
        cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
EOF

        if command_exists pre-commit; then
            pre-commit install 2>/dev/null || log_warning "Failed to install pre-commit hooks"
            log_success "Pre-commit hooks configured"
        fi
    fi
fi

# Create database initialization script
log_info "Creating database initialization script..."

cat > scripts/init_db.sql << 'EOF'
-- Database initialization script for trading system

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Market data tables
CREATE TABLE IF NOT EXISTS market_data.ohlcv (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(15,8) NOT NULL,
    high DECIMAL(15,8) NOT NULL,
    low DECIMAL(15,8) NOT NULL,
    close DECIMAL(15,8) NOT NULL,
    volume BIGINT NOT NULL,
    asset_type VARCHAR(20) DEFAULT 'stock',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, timestamp, timeframe)
);

CREATE TABLE IF NOT EXISTS market_data.finviz_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    company VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    country VARCHAR(50),
    market_cap VARCHAR(20),
    pe_ratio DECIMAL(10,2),
    price DECIMAL(15,8),
    change_percent DECIMAL(8,4),
    volume BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, DATE(timestamp))
);

-- Trading tables
CREATE TABLE IF NOT EXISTS trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    broker_order_id VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    price DECIMAL(15,8),
    filled_price DECIMAL(15,8),
    stop_price DECIMAL(15,8),
    time_in_force VARCHAR(10) DEFAULT 'day',
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    commission DECIMAL(10,4),
    strategy_name VARCHAR(100),
    client_order_id VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(15,8) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    order_id UUID REFERENCES trading.orders(id),
    strategy_name VARCHAR(100),
    pnl DECIMAL(15,8)
);

CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL UNIQUE,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    current_price DECIMAL(15,8) NOT NULL,
    unrealized_pnl DECIMAL(15,8),
    market_value DECIMAL(15,8),
    cost_basis DECIMAL(15,8),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk management tables
CREATE TABLE IF NOT EXISTS risk.portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cash DECIMAL(15,8) NOT NULL,
    buying_power DECIMAL(15,8) NOT NULL,
    total_equity DECIMAL(15,8) NOT NULL,
    day_trades_count INTEGER DEFAULT 0,
    pattern_day_trader BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS risk.risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    description TEXT,
    severity VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Analytics tables
CREATE TABLE IF NOT EXISTS analytics.strategy_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(15,8) DEFAULT 0,
    win_rate DECIMAL(5,4),
    avg_trade_duration INTERVAL,
    max_consecutive_wins INTEGER DEFAULT 0,
    max_consecutive_losses INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON market_data.ohlcv(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe ON market_data.ohlcv(timeframe);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON trading.orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON trading.orders(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trading.trades(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON risk.portfolio_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk.risk_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_name ON analytics.strategy_performance(strategy_name);

-- Create views for common queries
CREATE OR REPLACE VIEW trading.daily_pnl AS
SELECT
    DATE(timestamp) as trade_date,
    symbol,
    strategy_name,
    SUM(CASE WHEN side = 'buy' THEN -1 * (price * quantity + commission)
             WHEN side = 'sell' THEN (price * quantity - commission)
             END) as daily_pnl,
    COUNT(*) as trade_count
FROM trading.trades
GROUP BY DATE(timestamp), symbol, strategy_name
ORDER BY trade_date DESC;

CREATE OR REPLACE VIEW analytics.portfolio_summary AS
SELECT
    p.symbol,
    p.quantity,
    p.entry_price,
    p.current_price,
    p.unrealized_pnl,
    p.market_value,
    (p.market_value / NULLIF(ps.total_equity, 0)) * 100 as position_weight_percent,
    p.last_updated
FROM trading.positions p
CROSS JOIN LATERAL (
    SELECT total_equity
    FROM risk.portfolio_snapshots
    ORDER BY timestamp DESC
    LIMIT 1
) ps
WHERE p.quantity != 0;

-- Grant permissions
GRANT USAGE ON SCHEMA market_data TO trader;
GRANT USAGE ON SCHEMA trading TO trader;
GRANT USAGE ON SCHEMA risk TO trader;
GRANT USAGE ON SCHEMA analytics TO trader;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA risk TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO trader;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA market_data TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA risk TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO trader;

-- Insert initial data
INSERT INTO risk.portfolio_snapshots (account_id, cash, buying_power, total_equity)
VALUES ('main', 100000.00, 100000.00, 100000.00)
ON CONFLICT DO NOTHING;

COMMIT;
EOF

log_success "Database initialization script created"

# Create Redis configuration
log_info "Creating Redis configuration..."

cat > scripts/redis.conf << 'EOF'
# Redis configuration for trading system

# Basic settings
port 6379
bind 0.0.0.0
protected-mode yes

# Memory settings
maxmemory 512mb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile ""

# Security
# requirepass will be set by environment variable

# Performance
tcp-keepalive 300
timeout 0

# Append only file
appendonly yes
appendfsync everysec

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128
EOF

log_success "Redis configuration created"

# Check Docker daemon
log_info "Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon is not running. Please start Docker first"
    exit 1
fi
log_success "Docker daemon is running"

# Pull base images
log_info "Pulling base Docker images..."
docker pull python:3.11-slim || log_warning "Failed to pull Python base image"
docker pull postgres:15-alpine || log_warning "Failed to pull PostgreSQL image"
docker pull redis:7-alpine || log_warning "Failed to pull Redis image"
log_success "Base images pulled"

# Build base image
log_info "Building base Docker image..."
if docker build -t trading-system-base:latest -f Dockerfile.base . >/dev/null 2>&1; then
    log_success "Base Docker image built successfully"
else
    log_error "Failed to build base Docker image"
    exit 1
fi

# Install Python development dependencies locally (optional)
if [ "${INSTALL_LOCAL_DEPS:-false}" = "true" ]; then
    log_info "Installing local Python dependencies..."

    if command_exists pip3; then
        pip3 install --user \
            pydantic \
            fastapi \
            uvicorn \
            pandas \
            numpy \
            pytest \
            black \
            isort \
            flake8 \
            mypy \
            pre-commit 2>/dev/null || log_warning "Some local dependencies failed to install"
        log_success "Local dependencies installed"
    fi
fi

# Create development compose override
log_info "Creating development configuration..."

cat > docker-compose.dev.yml << 'EOF'
version: '3.8'

# Development overrides for docker-compose.yml
services:
  data_collector:
    volumes:
      - ./services/data_collector/src:/app/src
      - ./shared:/app/shared
    environment:
      - LOG_LEVEL=DEBUG
      - DEBUG=true

  strategy_engine:
    volumes:
      - ./services/strategy_engine/src:/app/src
      - ./shared:/app/shared
    environment:
      - LOG_LEVEL=DEBUG
      - DEBUG=true

  risk_manager:
    volumes:
      - ./services/risk_manager/src:/app/src
      - ./shared:/app/shared
    environment:
      - LOG_LEVEL=DEBUG
      - DEBUG=true

  trade_executor:
    volumes:
      - ./services/trade_executor/src:/app/src
      - ./shared:/app/shared
    environment:
      - LOG_LEVEL=DEBUG
      - DEBUG=true

  scheduler:
    volumes:
      - ./services/scheduler/src:/app/src
      - ./shared:/app/shared
    environment:
      - LOG_LEVEL=DEBUG
      - DEBUG=true
EOF

log_success "Development configuration created"

# Create production compose override
log_info "Creating production configuration..."

cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

# Production overrides for docker-compose.yml
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  redis:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'

  data_collector:
    environment:
      - LOG_LEVEL=INFO
      - DEBUG=false
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        max_attempts: 3

  strategy_engine:
    environment:
      - LOG_LEVEL=INFO
      - DEBUG=false
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
      restart_policy:
        condition: on-failure
        max_attempts: 3

  risk_manager:
    environment:
      - LOG_LEVEL=INFO
      - DEBUG=false
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        max_attempts: 3

  trade_executor:
    environment:
      - LOG_LEVEL=INFO
      - DEBUG=false
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        max_attempts: 3

  scheduler:
    environment:
      - LOG_LEVEL=INFO
      - DEBUG=false
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        max_attempts: 3
EOF

log_success "Production configuration created"

# Set file permissions
log_info "Setting file permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.py 2>/dev/null || true
log_success "File permissions set"

# Final instructions
echo -e "\n${GREEN}================================================================"
echo "                    Setup Complete!"
echo "================================================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Edit .env file with your API keys and configuration:"
echo "   nano .env"
echo ""
echo "2. Validate your configuration:"
echo "   make check-env"
echo ""
echo "3. Build the Docker images:"
echo "   make build"
echo ""
echo "4. Initialize the database:"
echo "   make init-db"
echo ""
echo "5. Start the services:"
echo "   make start"
echo ""
echo "6. Check service health:"
echo "   make health"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "- Make sure to set your API keys in the .env file"
echo "- For production use, update security settings"
echo "- Review risk parameters before live trading"
echo "- Test with paper trading first"
echo ""
echo -e "${GREEN}For help, run: make help${NC}"
echo ""

log_success "Automated Trading System setup completed successfully!"

exit 0
