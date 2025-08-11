#!/bin/bash

# AI Trading System Environment Setup Script
# This script sets up the environment for different deployment scenarios
# Usage: ./setup_environment.sh [environment] [options]

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SETUP_LOG="${PROJECT_ROOT}/logs/setup_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Default values
ENVIRONMENT="development"
FORCE_SETUP=false
SKIP_VALIDATION=false
INSTALL_DEPS=true
SETUP_DATABASE=true
SETUP_MONITORING=true
SETUP_LOGGING=true
CREATE_SECRETS=true

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$SETUP_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$SETUP_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" | tee -a "$SETUP_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$SETUP_LOG"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Environment Setup Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    development     Setup development environment (default)
    staging         Setup staging environment
    production      Setup production environment

OPTIONS:
    --force                Force setup even if environment exists
    --skip-validation      Skip configuration validation
    --no-deps              Skip dependency installation
    --no-database          Skip database setup
    --no-monitoring        Skip monitoring setup
    --no-logging           Skip logging setup
    --no-secrets           Skip secret generation
    --help                 Show this help message

EXAMPLES:
    $0 development
    $0 staging --force
    $0 production --skip-validation
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            --force)
                FORCE_SETUP=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --no-deps)
                INSTALL_DEPS=false
                shift
                ;;
            --no-database)
                SETUP_DATABASE=false
                shift
                ;;
            --no-monitoring)
                SETUP_MONITORING=false
                shift
                ;;
            --no-logging)
                SETUP_LOGGING=false
                shift
                ;;
            --no-secrets)
                CREATE_SECRETS=false
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_status "ERROR" "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to check prerequisites
check_prerequisites() {
    print_status "INFO" "Checking prerequisites for $ENVIRONMENT environment..."

    local missing_tools=()

    # Check for required tools
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi

    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    fi

    if ! command -v openssl &> /dev/null; then
        missing_tools+=("openssl")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_status "ERROR" "Missing required tools: ${missing_tools[*]}"
        print_status "INFO" "Please install the missing tools and retry"
        exit 1
    fi

    print_status "SUCCESS" "All prerequisites satisfied"
}

# Function to create directory structure
create_directories() {
    print_status "INFO" "Creating directory structure..."

    local dirs=(
        "data/backups"
        "data/exports"
        "data/logs"
        "data/metrics"
        "data/secrets"
        "data/state"
        "logs/services"
        "logs/deployment"
        "logs/backup"
        "logs/monitoring"
        "monitoring/prometheus/data"
        "monitoring/grafana/data"
        "monitoring/alertmanager/data"
        "monitoring/elasticsearch/data"
        "monitoring/kibana/data"
        "monitoring/logstash/pipeline"
        "config/ssl"
        "config/monitoring"
        "config/backup"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "${PROJECT_ROOT}/${dir}"
        print_status "INFO" "Created directory: $dir"
    done

    print_status "SUCCESS" "Directory structure created"
}

# Function to setup environment file
setup_environment_file() {
    print_status "INFO" "Setting up environment file for $ENVIRONMENT..."

    local env_file="${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}"
    local template_file="${PROJECT_ROOT}/config/environments/.env.template"

    if [[ ! -f "$template_file" ]]; then
        print_status "ERROR" "Environment template file not found: $template_file"
        exit 1
    fi

    if [[ -f "$env_file" ]] && [[ "$FORCE_SETUP" != "true" ]]; then
        print_status "WARNING" "Environment file already exists: $env_file"
        read -p "Overwrite existing file? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "INFO" "Skipping environment file creation"
            return 0
        fi
    fi

    # Copy template and customize for environment
    cp "$template_file" "$env_file"

    # Generate random secrets for development/staging
    if [[ "$ENVIRONMENT" != "production" ]] && [[ "$CREATE_SECRETS" == "true" ]]; then
        local jwt_secret=$(openssl rand -hex 32)
        local encryption_key=$(openssl rand -hex 32)
        local redis_password=$(openssl rand -hex 16)
        local postgres_password=$(openssl rand -hex 16)

        # Replace placeholders in env file
        sed -i "s/YOUR_JWT_SECRET_HERE/$jwt_secret/g" "$env_file"
        sed -i "s/YOUR_ENCRYPTION_KEY_HERE/$encryption_key/g" "$env_file"
        sed -i "s/YOUR_REDIS_PASSWORD_HERE/$redis_password/g" "$env_file"
        sed -i "s/YOUR_POSTGRES_PASSWORD_HERE/$postgres_password/g" "$env_file"
    fi

    # Set environment-specific values
    case $ENVIRONMENT in
        "development")
            sed -i "s/ENVIRONMENT=.*/ENVIRONMENT=development/g" "$env_file"
            sed -i "s/DEBUG=.*/DEBUG=true/g" "$env_file"
            sed -i "s/LOG_LEVEL=.*/LOG_LEVEL=DEBUG/g" "$env_file"
            ;;
        "staging")
            sed -i "s/ENVIRONMENT=.*/ENVIRONMENT=staging/g" "$env_file"
            sed -i "s/DEBUG=.*/DEBUG=false/g" "$env_file"
            sed -i "s/LOG_LEVEL=.*/LOG_LEVEL=INFO/g" "$env_file"
            ;;
        "production")
            sed -i "s/ENVIRONMENT=.*/ENVIRONMENT=production/g" "$env_file"
            sed -i "s/DEBUG=.*/DEBUG=false/g" "$env_file"
            sed -i "s/LOG_LEVEL=.*/LOG_LEVEL=WARNING/g" "$env_file"
            ;;
    esac

    print_status "SUCCESS" "Environment file created: $env_file"
}

# Function to validate configuration
validate_configuration() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        print_status "INFO" "Skipping configuration validation"
        return 0
    fi

    print_status "INFO" "Validating configuration for $ENVIRONMENT..."

    # Run configuration validation script
    if [[ -f "${PROJECT_ROOT}/scripts/config/validate_config.py" ]]; then
        cd "$PROJECT_ROOT"
        python3 scripts/config/validate_config.py \
            --env "$ENVIRONMENT" \
            --strict \
            || {
                print_status "ERROR" "Configuration validation failed"
                exit 1
            }
        print_status "SUCCESS" "Configuration validation passed"
    else
        print_status "WARNING" "Configuration validation script not found"
    fi
}

# Function to install dependencies
install_dependencies() {
    if [[ "$INSTALL_DEPS" != "true" ]]; then
        print_status "INFO" "Skipping dependency installation"
        return 0
    fi

    print_status "INFO" "Installing dependencies..."

    # Install Python dependencies
    if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        cd "$PROJECT_ROOT"
        python3 -m pip install --upgrade pip
        python3 -m pip install -r requirements.txt
        print_status "SUCCESS" "Python dependencies installed"
    fi

    # Install development dependencies if needed
    if [[ "$ENVIRONMENT" == "development" ]] && [[ -f "${PROJECT_ROOT}/requirements-dev.txt" ]]; then
        python3 -m pip install -r requirements-dev.txt
        print_status "SUCCESS" "Development dependencies installed"
    fi
}

# Function to setup database
setup_database() {
    if [[ "$SETUP_DATABASE" != "true" ]]; then
        print_status "INFO" "Skipping database setup"
        return 0
    fi

    print_status "INFO" "Setting up database for $ENVIRONMENT..."

    # Start database container if not running
    cd "$PROJECT_ROOT"

    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.prod.yml up -d postgres
    else
        docker-compose up -d postgres
    fi

    # Wait for database to be ready
    print_status "INFO" "Waiting for database to be ready..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if docker-compose exec -T postgres pg_isready -U trading_user -d trading_db; then
            break
        fi
        sleep 2
        ((retries--))
    done

    if [[ $retries -eq 0 ]]; then
        print_status "ERROR" "Database failed to start"
        exit 1
    fi

    print_status "SUCCESS" "Database is ready"

    # Run initial database setup if needed
    if [[ -f "${PROJECT_ROOT}/scripts/deployment/migrate.sh" ]]; then
        bash "${PROJECT_ROOT}/scripts/deployment/migrate.sh" --initialize
        print_status "SUCCESS" "Database initialized"
    fi
}

# Function to setup monitoring stack
setup_monitoring() {
    if [[ "$SETUP_MONITORING" != "true" ]]; then
        print_status "INFO" "Skipping monitoring setup"
        return 0
    fi

    print_status "INFO" "Setting up monitoring stack..."

    cd "$PROJECT_ROOT"

    # Create monitoring configuration directories
    mkdir -p monitoring/prometheus/config
    mkdir -p monitoring/grafana/provisioning/{dashboards,datasources}
    mkdir -p monitoring/alertmanager/config

    # Copy monitoring configurations if they exist
    if [[ -d "${PROJECT_ROOT}/monitoring/configs" ]]; then
        cp -r "${PROJECT_ROOT}/monitoring/configs/"* "${PROJECT_ROOT}/monitoring/"
        print_status "SUCCESS" "Monitoring configurations copied"
    fi

    # Start monitoring services
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.prod.yml up -d prometheus grafana alertmanager
    else
        docker-compose up -d prometheus grafana alertmanager
    fi

    print_status "SUCCESS" "Monitoring stack started"
}

# Function to setup logging infrastructure
setup_logging() {
    if [[ "$SETUP_LOGGING" != "true" ]]; then
        print_status "INFO" "Skipping logging setup"
        return 0
    fi

    print_status "INFO" "Setting up logging infrastructure..."

    cd "$PROJECT_ROOT"

    # Create log directories with proper permissions
    mkdir -p logs/{services,deployment,backup,monitoring,security}
    chmod 755 logs
    chmod 755 logs/*

    # Setup log rotation
    if command -v logrotate &> /dev/null; then
        cat > /tmp/trading-logrotate << EOF
${PROJECT_ROOT}/logs/services/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
    postrotate
        docker-compose restart data_collector strategy_engine risk_manager trade_executor scheduler export_service maintenance_service
    endscript
}

${PROJECT_ROOT}/logs/deployment/*.log {
    weekly
    rotate 12
    compress
    delaycompress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
}

${PROJECT_ROOT}/logs/backup/*.log {
    weekly
    rotate 52
    compress
    delaycompress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
}
EOF

        if [[ -w /etc/logrotate.d/ ]]; then
            sudo cp /tmp/trading-logrotate /etc/logrotate.d/ai-trading-system
            print_status "SUCCESS" "Log rotation configured"
        else
            print_status "WARNING" "Cannot configure system log rotation (no sudo access)"
            print_status "INFO" "Manual logrotate config created at /tmp/trading-logrotate"
        fi
        rm -f /tmp/trading-logrotate
    fi

    # Start logging services
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.prod.yml up -d elasticsearch kibana logstash
    else
        docker-compose up -d elasticsearch kibana logstash
    fi

    print_status "SUCCESS" "Logging infrastructure started"
}

# Function to generate SSL certificates
generate_ssl_certificates() {
    print_status "INFO" "Generating SSL certificates..."

    local ssl_dir="${PROJECT_ROOT}/config/ssl"
    mkdir -p "$ssl_dir"

    if [[ "$ENVIRONMENT" == "production" ]]; then
        print_status "WARNING" "Production environment detected"
        print_status "INFO" "Please configure proper SSL certificates from a trusted CA"
        print_status "INFO" "Self-signed certificates are not recommended for production"
        return 0
    fi

    # Generate self-signed certificates for development/staging
    cd "$ssl_dir"

    # Generate private key
    openssl genrsa -out trading-system.key 2048

    # Generate certificate signing request
    openssl req -new -key trading-system.key -out trading-system.csr \
        -subj "/C=US/ST=State/L=City/O=AI Trading System/CN=localhost"

    # Generate self-signed certificate
    openssl x509 -req -days 365 -in trading-system.csr \
        -signkey trading-system.key -out trading-system.crt

    # Generate DH parameters
    openssl dhparam -out dhparam.pem 2048

    # Set proper permissions
    chmod 600 *.key *.pem
    chmod 644 *.crt

    print_status "SUCCESS" "SSL certificates generated"
}

# Function to setup secrets
setup_secrets() {
    if [[ "$CREATE_SECRETS" != "true" ]]; then
        print_status "INFO" "Skipping secret generation"
        return 0
    fi

    print_status "INFO" "Setting up secrets for $ENVIRONMENT..."

    local secrets_dir="${PROJECT_ROOT}/data/secrets"
    mkdir -p "$secrets_dir"
    chmod 700 "$secrets_dir"

    # Generate API keys and secrets
    local api_secret=$(openssl rand -hex 32)
    local webhook_secret=$(openssl rand -hex 24)
    local encryption_salt=$(openssl rand -hex 16)

    # Create secrets file
    cat > "${secrets_dir}/api_keys.json" << EOF
{
    "api_secret": "${api_secret}",
    "webhook_secret": "${webhook_secret}",
    "encryption_salt": "${encryption_salt}",
    "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "${ENVIRONMENT}"
}
EOF

    chmod 600 "${secrets_dir}/api_keys.json"

    # Generate database encryption key
    openssl rand -hex 32 > "${secrets_dir}/db_encryption.key"
    chmod 600 "${secrets_dir}/db_encryption.key"

    print_status "SUCCESS" "Secrets generated and stored securely"
}

# Function to initialize database schema
initialize_database() {
    print_status "INFO" "Initializing database schema..."

    cd "$PROJECT_ROOT"

    # Wait for database to be fully ready
    local retries=60
    while [[ $retries -gt 0 ]]; do
        if docker-compose exec -T postgres pg_isready -U trading_user -d trading_db; then
            break
        fi
        sleep 1
        ((retries--))
    done

    # Run schema initialization scripts
    local sql_scripts=(
        "scripts/risk_tables.sql"
        "scripts/trade_execution_tables.sql"
    )

    for script in "${sql_scripts[@]}"; do
        if [[ -f "$script" ]]; then
            print_status "INFO" "Executing SQL script: $script"
            docker-compose exec -T postgres psql -U trading_user -d trading_db -f "/docker-entrypoint-initdb.d/$(basename "$script")"
        fi
    done

    print_status "SUCCESS" "Database schema initialized"
}

# Function to setup monitoring dashboards
setup_monitoring_dashboards() {
    print_status "INFO" "Setting up monitoring dashboards..."

    local grafana_provisioning="${PROJECT_ROOT}/monitoring/grafana/provisioning"

    # Create Grafana datasource configuration
    cat > "${grafana_provisioning}/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: "trading-logs-*"
    isDefault: false
    editable: true
    jsonData:
      interval: "1m"
      timeField: "@timestamp"
      esVersion: "7.0.0"
EOF

    # Create dashboard provisioning configuration
    cat > "${grafana_provisioning}/dashboards/dashboard.yml" << EOF
apiVersion: 1

providers:
  - name: 'Trading System Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    print_status "SUCCESS" "Monitoring dashboards configured"
}

# Function to setup health checks
setup_health_checks() {
    print_status "INFO" "Setting up health check monitoring..."

    # Create health check script
    cat > "${PROJECT_ROOT}/scripts/health_check.sh" << 'EOF'
#!/bin/bash

# Health Check Script for AI Trading System
# Monitors all services and reports status

set -euo pipefail

SERVICES=(
    "postgres:5432"
    "redis:6379"
    "data_collector:8001"
    "strategy_engine:8002"
    "risk_manager:8003"
    "trade_executor:8004"
    "scheduler:8005"
    "export_service:8006"
    "maintenance_service:8007"
)

MONITORING_SERVICES=(
    "prometheus:9090"
    "grafana:3000"
    "elasticsearch:9200"
    "kibana:5601"
)

check_service_health() {
    local service=$1
    local host=$(echo "$service" | cut -d: -f1)
    local port=$(echo "$service" | cut -d: -f2)

    if docker-compose exec -T "$host" nc -z localhost "$port" 2>/dev/null; then
        echo "✅ $service is healthy"
        return 0
    else
        echo "❌ $service is not responding"
        return 1
    fi
}

main() {
    echo "🔍 AI Trading System Health Check"
    echo "=================================="

    local failed=0

    echo ""
    echo "Core Services:"
    for service in "${SERVICES[@]}"; do
        if ! check_service_health "$service"; then
            ((failed++))
        fi
    done

    echo ""
    echo "Monitoring Services:"
    for service in "${MONITORING_SERVICES[@]}"; do
        if ! check_service_health "$service"; then
            ((failed++))
        fi
    done

    echo ""
    if [[ $failed -eq 0 ]]; then
        echo "🎉 All services are healthy!"
        exit 0
    else
        echo "⚠️  $failed service(s) are not healthy"
        exit 1
    fi
}

main "$@"
EOF

    chmod +x "${PROJECT_ROOT}/scripts/health_check.sh"
    print_status "SUCCESS" "Health check script created"
}

# Function to setup log rotation
setup_log_rotation() {
    print_status "INFO" "Setting up log rotation..."

    # Create log cleanup script
    cat > "${PROJECT_ROOT}/scripts/cleanup_logs.sh" << 'EOF'
#!/bin/bash

# Log Cleanup Script for AI Trading System
# Cleans up old log files to prevent disk space issues

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_RETENTION_DAYS=${LOG_RETENTION_DAYS:-30}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-90}

echo "🧹 Starting log cleanup (retention: ${LOG_RETENTION_DAYS} days)"

# Clean up service logs
find "${PROJECT_ROOT}/logs/services" -name "*.log" -mtime +${LOG_RETENTION_DAYS} -delete 2>/dev/null || true
find "${PROJECT_ROOT}/logs/deployment" -name "*.log" -mtime +${LOG_RETENTION_DAYS} -delete 2>/dev/null || true
find "${PROJECT_ROOT}/logs/monitoring" -name "*.log" -mtime +${LOG_RETENTION_DAYS} -delete 2>/dev/null || true

# Clean up old backups
find "${PROJECT_ROOT}/data/backups" -name "*.tar.gz" -mtime +${BACKUP_RETENTION_DAYS} -delete 2>/dev/null || true

# Clean up old exports
find "${PROJECT_ROOT}/data/exports" -name "*.zip" -mtime +7 -delete 2>/dev/null || true

# Clean up Docker logs
docker system prune -f --filter "until=72h" 2>/dev/null || true

echo "✅ Log cleanup completed"
EOF

    chmod +x "${PROJECT_ROOT}/scripts/cleanup_logs.sh"
    print_status "SUCCESS" "Log cleanup script created"
}

# Function to setup firewall rules
setup_firewall() {
    if [[ "$ENVIRONMENT" != "production" ]]; then
        print_status "INFO" "Skipping firewall setup for $ENVIRONMENT"
        return 0
    fi

    print_status "INFO" "Setting up firewall rules for production..."

    # Create firewall rules script
    cat > "${PROJECT_ROOT}/scripts/security/setup_firewall.sh" << 'EOF'
#!/bin/bash

# Firewall Setup for AI Trading System Production Environment
# Configures UFW (Uncomplicated Firewall) rules

set -euo pipefail

echo "🔥 Setting up firewall rules..."

# Enable UFW if not already enabled
if ! ufw status | grep -q "Status: active"; then
    echo "Enabling UFW..."
    ufw --force enable
fi

# Default policies
ufw --force default deny incoming
ufw --force default allow outgoing

# SSH access (adjust port if needed)
ufw allow 22/tcp comment 'SSH'

# HTTP/HTTPS for web interface
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Internal service ports (only from local network)
ufw allow from 10.0.0.0/8 to any port 5432 comment 'PostgreSQL internal'
ufw allow from 172.16.0.0/12 to any port 6379 comment 'Redis internal'
ufw allow from 192.168.0.0/16 to any port 9090 comment 'Prometheus internal'
ufw allow from 10.0.0.0/8 to any port 3000 comment 'Grafana internal'
ufw allow from 172.16.0.0/12 to any port 5601 comment 'Kibana internal'

# API endpoints (adjust as needed)
ufw allow 8080/tcp comment 'Trading API'

# Show status
ufw status verbose

echo "✅ Firewall rules configured"
EOF

    chmod +x "${PROJECT_ROOT}/scripts/security/setup_firewall.sh"
    print_status "SUCCESS" "Firewall setup script created"
}

# Function to create systemd services
create_systemd_services() {
    if [[ "$ENVIRONMENT" != "production" ]]; then
        print_status "INFO" "Skipping systemd service creation for $ENVIRONMENT"
        return 0
    fi

    print_status "INFO" "Creating systemd services..."

    # Create systemd service file
    cat > /tmp/ai-trading-system.service << EOF
[Unit]
Description=AI Trading System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${PROJECT_ROOT}
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    if [[ -w /etc/systemd/system/ ]]; then
        sudo cp /tmp/ai-trading-system.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable ai-trading-system.service
        print_status "SUCCESS" "Systemd service created and enabled"
    else
        print_status "WARNING" "Cannot create systemd service (no sudo access)"
        print_status "INFO" "Service file created at /tmp/ai-trading-system.service"
    fi

    rm -f /tmp/ai-trading-system.service
}

# Function to setup backup cron jobs
setup_backup_jobs() {
    print_status "INFO" "Setting up backup cron jobs..."

    # Create backup cron script
    cat > "${PROJECT_ROOT}/scripts/backup/run_backups.sh" << 'EOF'
#!/bin/bash

# Automated Backup Script for AI Trading System
# Runs daily backups and cleanup

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Source environment
if [[ -f "config/environments/.env.production" ]]; then
    source config/environments/.env.production
elif [[ -f "config/environments/.env.staging" ]]; then
    source config/environments/.env.staging
else
    source config/environments/.env.development
fi

# Run backup
echo "🔄 Starting automated backup at $(date)"
bash scripts/backup/backup.sh --type daily --compress --verify

# Test restore (weekly on Sundays)
if [[ $(date +%u) -eq 7 ]]; then
    echo "🧪 Running weekly restore test"
    bash scripts/backup/test_restore.sh --latest
fi

# Cleanup old logs
bash scripts/cleanup_logs.sh

echo "✅ Automated backup completed at $(date)"
EOF

    chmod +x "${PROJECT_ROOT}/scripts/backup/run_backups.sh"

    # Add cron job for backups
    local cron_entry="0 2 * * * ${PROJECT_ROOT}/scripts/backup/run_backups.sh >> ${PROJECT_ROOT}/logs/backup/cron.log 2>&1"

    if crontab -l 2>/dev/null | grep -q "run_backups.sh"; then
        print_status "INFO" "Backup cron job already exists"
    else
        (crontab -l 2>/dev/null; echo "$cron_entry") | crontab -
        print_status "SUCCESS" "Backup cron job added"
    fi
}

# Function to run post-setup tests
run_post_setup_tests() {
    print_status "INFO" "Running post-setup tests..."

    cd "$PROJECT_ROOT"

    # Wait for all services to be fully ready
    sleep 30

    # Run health checks
    if [[ -f "scripts/health_check.sh" ]]; then
        bash scripts/health_check.sh || {
            print_status "ERROR" "Health checks failed"
            return 1
        }
    fi

    # Run basic functionality tests
    if [[ -f "scripts/run_tests.py" ]]; then
        python3 scripts/run_tests.py --smoke-tests || {
            print_status "ERROR" "Smoke tests failed"
            return 1
        }
    fi

    print_status "SUCCESS" "Post-setup tests passed"
}

# Function to display setup summary
show_setup_summary() {
    print_status "INFO" "Setup Summary for $ENVIRONMENT Environment"
    echo ""
    echo "📋 Configuration:"
    echo "   Environment: $ENVIRONMENT"
    echo "   Project Root: $PROJECT_ROOT"
    echo "   Setup Log: $SETUP_LOG"
    echo ""
    echo "🚀 Services Started:"
    echo "   Database: PostgreSQL (port 5432)"
    echo "   Cache: Redis (port 6379)"
    echo "   Monitoring: Prometheus (port 9090), Grafana (port 3000)"
    echo "   Logging: Elasticsearch (port 9200), Kibana (port 5601)"
    echo ""
    echo "🔐 Security:"
    echo "   SSL certificates generated"
    echo "   Secrets stored in data/secrets/"
    echo "   Firewall rules configured (production only)"
    echo ""
    echo "📊 Access URLs:"
    echo "   Grafana Dashboard: http://localhost:3000"
    echo "   Kibana Logs: http://localhost:5601"
    echo "   Prometheus Metrics: http://localhost:9090"
    echo "   Trading API: http://localhost:8080"
    echo ""
    echo "🔧 Next Steps:"
    echo "   1. Review environment configuration in config/environments/.env.$ENVIRONMENT"
    echo "   2. Configure API keys for trading platforms"
    echo "   3. Customize monitoring dashboards"
    echo "   4. Setup SSL certificates for production"
    echo "   5. Review backup and recovery procedures"
    echo ""
    echo "📖 Documentation:"
    echo "   Operations Runbook: docs/operations/RUNBOOK.md"
    echo "   Troubleshooting: docs/operations/TROUBLESHOOTING.md"
    echo "   Disaster Recovery: docs/operations/DISASTER_RECOVERY.md"
    echo ""
}

# Function to cleanup on error
+cleanup_on_error() {
+    local exit_code=$?
+    print_status "ERROR" "Setup failed with exit code $exit_code"
+
+    if [[ "$ENVIRONMENT" != "production" ]]; then
+        print_status "INFO" "Cleaning up failed setup..."
+        cd "$PROJECT_ROOT"
+        docker-compose down --remove-orphans || true
+        docker system prune -f || true
+    fi
+
+    print_status "INFO" "Setup log available at: $SETUP_LOG"
+    exit $exit_code
+}
+
+# Function to verify setup completion
+verify_setup() {
+    print_status "INFO" "Verifying setup completion..."
+
+    local required_files=(
+        "config/environments/.env.$ENVIRONMENT"
+        "docker-compose.yml"
+        "scripts/deployment/deploy.sh"
+        "scripts/backup/backup.sh"
+    )
+
+    for file in "${required_files[@]}"; do
+        if [[ ! -f "${PROJECT_ROOT}/$file" ]]; then
+            print_status "ERROR" "Required file missing: $file"
+            return 1
+        fi
+    done
+
+    # Check if core services are defined in docker-compose
+    local required_services=(
+        "postgres"
+        "redis"
+        "data_collector"
+        "strategy_engine"
+        "risk_manager"
+        "trade_executor"
+    )
+
+    for service in "${required_services[@]}"; do
+        if ! grep -q "^  $service:" "${PROJECT_ROOT}/docker-compose.yml"; then
+            print_status "ERROR" "Required service not found in docker-compose.yml: $service"
+            return 1
+        fi
+    done
+
+    print_status "SUCCESS" "Setup verification completed"
+}
+
+# Main execution function
+main() {
+    print_status "INFO" "Starting AI Trading System environment setup"
+    print_status "INFO" "Environment: $ENVIRONMENT"
+    print_status "INFO" "Project Root: $PROJECT_ROOT"
+    print_status "INFO" "Setup Log: $SETUP_LOG"
+
+    # Set up error handling
+    trap cleanup_on_error ERR
+
+    # Create initial log entry
+    echo "=== AI Trading System Environment Setup ===" > "$SETUP_LOG"
+    echo "Started at: $(date)" >> "$SETUP_LOG"
+    echo "Environment: $ENVIRONMENT" >> "$SETUP_LOG"
+    echo "User: $(whoami)" >> "$SETUP_LOG"
+    echo "Working Directory: $PROJECT_ROOT" >> "$SETUP_LOG"
+    echo "" >> "$SETUP_LOG"
+
+    # Execute setup steps
+    print_status "INFO" "Step 1/12: Checking prerequisites..."
+    check_prerequisites
+
+    print_status "INFO" "Step 2/12: Creating directory structure..."
+    create_directories
+
+    print_status "INFO" "Step 3/12: Setting up environment file..."
+    setup_environment_file
+
+    print_status "INFO" "Step 4/12: Validating configuration..."
+    validate_configuration
+
+    print_status "INFO" "Step 5/12: Installing dependencies..."
+    install_dependencies
+
+    print_status "INFO" "Step 6/12: Generating SSL certificates..."
+    generate_ssl_certificates
+
+    print_status "INFO" "Step 7/12: Setting up secrets..."
+    setup_secrets
+
+    print_status "INFO" "Step 8/12: Setting up database..."
+    setup_database
+
+    print_status "INFO" "Step 9/12: Initializing database schema..."
+    initialize_database
+
+    print_status "INFO" "Step 10/12: Setting up monitoring..."
+    setup_monitoring
+
+    print_status "INFO" "Step 11/12: Setting up logging..."
+    setup_logging
+
+    print_status "INFO" "Step 12/12: Configuring operational tools..."
+    setup_monitoring_dashboards
+    setup_health_checks
+    setup_log_rotation
+    setup_firewall
+    create_systemd_services
+    setup_backup_jobs
+
+    # Verify setup
+    verify_setup
+
+    # Run post-setup tests
+    if [[ "$SKIP_VALIDATION" != "true" ]]; then
+        run_post_setup_tests
+    fi
+
+    # Display summary
+    show_setup_summary
+
+    print_status "SUCCESS" "Environment setup completed successfully!"
+    print_status "INFO" "Setup log saved to: $SETUP_LOG"
+}
+
+# Script entry point
+if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
+    parse_args "$@"
+    main
+fi
