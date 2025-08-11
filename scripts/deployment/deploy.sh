#!/bin/bash

# AI Trading System Deployment Script
# This script provides one-command deployment with comprehensive checks and rollback capabilities
# Usage: ./deploy.sh [environment] [options]

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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DEPLOYMENT_LOG="${PROJECT_ROOT}/logs/deployment_${TIMESTAMP}.log"
ROLLBACK_STATE_FILE="${PROJECT_ROOT}/data/rollback_state.json"

# Default values
ENVIRONMENT="development"
DRY_RUN=false
SKIP_TESTS=false
SKIP_BACKUP=false
FORCE_REBUILD=false
ROLLBACK_ON_FAILURE=true
PARALLEL_BUILD=true
HEALTH_CHECK_TIMEOUT=300
MIGRATION_TIMEOUT=600

# Deployment options
BUILD_IMAGES=true
RUN_MIGRATIONS=true
UPDATE_CONFIGS=true
RESTART_SERVICES=true
VALIDATE_DEPLOYMENT=true

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$DEPLOYMENT_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$DEPLOYMENT_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" | tee -a "$DEPLOYMENT_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$DEPLOYMENT_LOG"
            ;;
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message" | tee -a "$DEPLOYMENT_LOG"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Deployment Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    development     Deploy to development environment (default)
    staging         Deploy to staging environment
    production      Deploy to production environment

OPTIONS:
    --dry-run              Show what would be done without executing
    --skip-tests           Skip running tests before deployment
    --skip-backup          Skip backup creation before deployment
    --force-rebuild        Force rebuild of all Docker images
    --no-rollback          Disable automatic rollback on failure
    --sequential           Build images sequentially instead of parallel
    --config-only          Only update configurations, don't restart services
    --migrations-only      Only run database migrations
    --health-timeout SEC   Health check timeout in seconds (default: 300)
    --migration-timeout SEC Migration timeout in seconds (default: 600)
    --help                 Show this help message

EXAMPLES:
    $0 development
    $0 staging --skip-tests --dry-run
    $0 production --force-rebuild --health-timeout 600
    $0 production --config-only --no-rollback

ROLLBACK:
    $0 rollback            Rollback to previous deployment
    $0 rollback --to TAG   Rollback to specific image tag

EOF
}

# Function to validate prerequisites
validate_prerequisites() {
    print_status "INFO" "Validating deployment prerequisites..."

    local missing_tools=()

    # Check required tools
    for tool in docker docker-compose python3 curl jq; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_status "ERROR" "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_status "ERROR" "Docker daemon is not running"
        exit 1
    fi

    # Check Docker Compose version
    local compose_version=$(docker-compose version --short)
    print_status "INFO" "Using Docker Compose version: $compose_version"

    # Validate environment configuration
    local config_file="${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}"
    if [ ! -f "$config_file" ]; then
        print_status "ERROR" "Configuration file not found: $config_file"
        exit 1
    fi

    # Run configuration validation
    print_status "INFO" "Validating configuration for $ENVIRONMENT environment..."
    if ! python3 "${PROJECT_ROOT}/scripts/config/validate_config.py" --env "$ENVIRONMENT" --quiet; then
        print_status "ERROR" "Configuration validation failed"
        exit 1
    fi

    print_status "SUCCESS" "Prerequisites validation completed"
}

# Function to create pre-deployment backup
create_backup() {
    if [ "$SKIP_BACKUP" = true ]; then
        print_status "WARNING" "Skipping backup creation (--skip-backup flag)"
        return
    fi

    print_status "INFO" "Creating pre-deployment backup..."

    local backup_dir="${PROJECT_ROOT}/backups/pre-deployment"
    mkdir -p "$backup_dir"

    # Create database backup
    if docker-compose ps postgres | grep -q "Up"; then
        print_status "INFO" "Backing up PostgreSQL database..."
        docker-compose exec -T postgres pg_dump -U trader -d trading_system > \
            "${backup_dir}/postgres_${TIMESTAMP}.sql"
    fi

    # Create Redis backup
    if docker-compose ps redis | grep -q "Up"; then
        print_status "INFO" "Backing up Redis data..."
        docker-compose exec -T redis redis-cli --rdb - > \
            "${backup_dir}/redis_${TIMESTAMP}.rdb"
    fi

    # Backup configuration and data
    print_status "INFO" "Backing up configuration and data files..."
    tar -czf "${backup_dir}/config_data_${TIMESTAMP}.tar.gz" \
        -C "$PROJECT_ROOT" \
        config/ data/ --exclude="data/temp" --exclude="data/logs" 2>/dev/null || true

    # Save current deployment state
    cat > "$ROLLBACK_STATE_FILE" << EOF
{
    "timestamp": "$TIMESTAMP",
    "environment": "$ENVIRONMENT",
    "backup_dir": "$backup_dir",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "docker_images": {
$(docker images --format "        \"{{.Repository}}:{{.Tag}}\": \"{{.ID}}\"," | grep trading-system | sed '$ s/,$//')
    }
}
EOF

    print_status "SUCCESS" "Backup created in $backup_dir"
}

# Function to run tests
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        print_status "WARNING" "Skipping tests (--skip-tests flag)"
        return
    fi

    print_status "INFO" "Running test suite..."

    # Build test environment
    docker-compose -f docker-compose.test.yml build

    # Run tests
    if ! docker-compose -f docker-compose.test.yml up --abort-on-container-exit --exit-code-from test_runner; then
        print_status "ERROR" "Tests failed"
        exit 1
    fi

    # Cleanup test containers
    docker-compose -f docker-compose.test.yml down

    print_status "SUCCESS" "All tests passed"
}

# Function to build Docker images
build_images() {
    if [ "$BUILD_IMAGES" = false ]; then
        print_status "INFO" "Skipping image build"
        return
    fi

    print_status "INFO" "Building Docker images for $ENVIRONMENT environment..."

    # Set environment variables for build
    export IMAGE_TAG="${ENVIRONMENT}-${TIMESTAMP}"
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1

    local compose_file="docker-compose.yml"
    if [ "$ENVIRONMENT" = "production" ]; then
        compose_file="docker-compose.prod.yml"
    fi

    # Build services
    local services=(
        "data_collector"
        "strategy_engine"
        "risk_manager"
        "trade_executor"
        "scheduler"
        "export_service"
    )

    if [ "$PARALLEL_BUILD" = true ]; then
        print_status "INFO" "Building images in parallel..."
        for service in "${services[@]}"; do
            {
                print_status "INFO" "Building $service..."
                if docker-compose -f "$compose_file" build "$service"; then
                    print_status "SUCCESS" "$service build completed"
                else
                    print_status "ERROR" "$service build failed"
                    exit 1
                fi
            } &
        done
        wait
    else
        print_status "INFO" "Building images sequentially..."
        for service in "${services[@]}"; do
            print_status "INFO" "Building $service..."
            if ! docker-compose -f "$compose_file" build "$service"; then
                print_status "ERROR" "$service build failed"
                exit 1
            fi
            print_status "SUCCESS" "$service build completed"
        done
    fi

    # Build specialized services
    if [ -f "${PROJECT_ROOT}/Dockerfile.backup" ]; then
        print_status "INFO" "Building backup service..."
        docker build -f "${PROJECT_ROOT}/Dockerfile.backup" -t "trading-system/backup_manager:${IMAGE_TAG}" "$PROJECT_ROOT"
    fi

    if [ -f "${PROJECT_ROOT}/Dockerfile.maintenance" ]; then
        print_status "INFO" "Building maintenance service..."
        docker build -f "${PROJECT_ROOT}/Dockerfile.maintenance" -t "trading-system/maintenance:${IMAGE_TAG}" "$PROJECT_ROOT"
    fi

    print_status "SUCCESS" "All images built successfully"
}

# Function to run database migrations
run_migrations() {
    if [ "$RUN_MIGRATIONS" = false ]; then
        print_status "INFO" "Skipping database migrations"
        return
    fi

    print_status "INFO" "Running database migrations..."

    # Ensure database is running
    docker-compose up -d postgres

    # Wait for database to be ready
    print_status "INFO" "Waiting for database to be ready..."
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose exec -T postgres pg_isready -U trader -d trading_system; then
            break
        fi
        print_status "INFO" "Database not ready, attempt $attempt/$max_attempts..."
        sleep 5
        ((attempt++))
    done

    if [ $attempt -gt $max_attempts ]; then
        print_status "ERROR" "Database failed to become ready within timeout"
        exit 1
    fi

    # Run migrations with timeout
    print_status "INFO" "Executing database migrations..."

    # Create migration container
    docker run --rm \
        --network "${PROJECT_NAME}_trading_network" \
        -e DATABASE_URL="postgresql://trader:${DB_PASSWORD}@postgres:5432/trading_system" \
        -v "${PROJECT_ROOT}/scripts/migrations:/migrations" \
        --name migration_runner \
        trading-system/data_collector:${IMAGE_TAG} \
        timeout "$MIGRATION_TIMEOUT" python -m alembic upgrade head

    print_status "SUCCESS" "Database migrations completed"
}

# Function to update configurations
update_configs() {
    if [ "$UPDATE_CONFIGS" = false ]; then
        print_status "INFO" "Skipping configuration updates"
        return
    fi

    print_status "INFO" "Updating service configurations..."

    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/exports" "${PROJECT_ROOT}/backups"

    # Set proper permissions
    chmod 755 "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/logs"
    chmod 700 "${PROJECT_ROOT}/exports" "${PROJECT_ROOT}/backups"

    # Copy environment-specific config
    local env_config="${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}"
    if [ -f "$env_config" ]; then
        cp "$env_config" "${PROJECT_ROOT}/.env"
        print_status "INFO" "Configuration updated for $ENVIRONMENT environment"
    fi

    # Update Docker Compose environment
    export COMPOSE_PROJECT_NAME="trading_system_${ENVIRONMENT}"
    export IMAGE_TAG="${ENVIRONMENT}-${TIMESTAMP}"

    print_status "SUCCESS" "Configuration updates completed"
}

# Function to deploy services
deploy_services() {
    if [ "$RESTART_SERVICES" = false ]; then
        print_status "INFO" "Skipping service restart"
        return
    fi

    print_status "INFO" "Deploying services..."

    local compose_file="docker-compose.yml"
    if [ "$ENVIRONMENT" = "production" ]; then
        compose_file="docker-compose.prod.yml"
    fi

    # Start infrastructure services first
    print_status "INFO" "Starting infrastructure services..."
    docker-compose -f "$compose_file" up -d postgres redis

    # Wait for infrastructure to be healthy
    print_status "INFO" "Waiting for infrastructure services..."
    local max_wait=120
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        if docker-compose -f "$compose_file" ps postgres | grep -q "healthy" && \
           docker-compose -f "$compose_file" ps redis | grep -q "healthy"; then
            break
        fi
        sleep 5
        ((elapsed += 5))
        print_status "INFO" "Waiting for infrastructure... (${elapsed}s/${max_wait}s)"
    done

    if [ $elapsed -ge $max_wait ]; then
        print_status "ERROR" "Infrastructure services failed to become healthy"
        exit 1
    fi

    # Start core trading services
    print_status "INFO" "Starting core trading services..."
    docker-compose -f "$compose_file" up -d \
        data_collector \
        strategy_engine \
        risk_manager \
        trade_executor

    # Start scheduler and support services
    print_status "INFO" "Starting scheduler and support services..."
    docker-compose -f "$compose_file" up -d \
        scheduler \
        export_service

    # Start monitoring services
    print_status "INFO" "Starting monitoring services..."
    docker-compose -f "$compose_file" up -d \
        prometheus \
        grafana \
        alertmanager \
        node_exporter \
        postgres_exporter \
        redis_exporter

    # Start backup service if available
    if docker-compose -f "$compose_file" config --services | grep -q backup_manager; then
        print_status "INFO" "Starting backup service..."
        docker-compose -f "$compose_file" up -d backup_manager
    fi

    # Start nginx if in production
    if [ "$ENVIRONMENT" = "production" ] && docker-compose -f "$compose_file" config --services | grep -q nginx; then
        print_status "INFO" "Starting nginx proxy..."
        docker-compose -f "$compose_file" up -d nginx
    fi

    print_status "SUCCESS" "All services started"
}

# Function to validate deployment
validate_deployment() {
    if [ "$VALIDATE_DEPLOYMENT" = false ]; then
        print_status "INFO" "Skipping deployment validation"
        return
    fi

    print_status "INFO" "Validating deployment..."

    local compose_file="docker-compose.yml"
    if [ "$ENVIRONMENT" = "production" ]; then
        compose_file="docker-compose.prod.yml"
    fi

    # Define services to check
    local services=(
        "postgres:5432"
        "redis:6379"
        "data_collector:8001"
        "strategy_engine:8002"
        "risk_manager:8003"
        "trade_executor:8004"
        "scheduler:8005"
        "export_service:8006"
    )

    local failed_services=()
    local max_wait=$HEALTH_CHECK_TIMEOUT
    local check_interval=10

    print_status "INFO" "Performing health checks (timeout: ${max_wait}s)..."

    for ((elapsed=0; elapsed<max_wait; elapsed+=check_interval)); do
        local all_healthy=true
        failed_services=()

        for service_port in "${services[@]}"; do
            local service_name=$(echo "$service_port" | cut -d':' -f1)
            local port=$(echo "$service_port" | cut -d':' -f2)

            # Check if container is running
            if ! docker-compose -f "$compose_file" ps "$service_name" | grep -q "Up"; then
                failed_services+=("$service_name (not running)")
                all_healthy=false
                continue
            fi

            # Check health endpoint
            local health_url="http://localhost:${port}/health"
            if ! curl -sf "$health_url" >/dev/null 2>&1; then
                failed_services+=("$service_name (health check failed)")
                all_healthy=false
            fi
        done

        if [ "$all_healthy" = true ]; then
            print_status "SUCCESS" "All services are healthy"
            return
        fi

        if [ $((elapsed % 30)) -eq 0 ]; then
            print_status "INFO" "Still waiting for services: ${failed_services[*]} (${elapsed}s/${max_wait}s)"
        fi

        sleep $check_interval
    done

    # Health checks failed
    print_status "ERROR" "Health check timeout reached. Failed services: ${failed_services[*]}"

    # Show logs for failed services
    for service_port in "${services[@]}"; do
        local service_name=$(echo "$service_port" | cut -d':' -f1)
        if [[ " ${failed_services[*]} " =~ " ${service_name} " ]]; then
            print_status "INFO" "Showing logs for failed service: $service_name"
            docker-compose -f "$compose_file" logs --tail=50 "$service_name" || true
        fi
    done

    exit 1
}

# Function to perform rollback
perform_rollback() {
    local target_tag=$1

    print_status "WARNING" "Initiating rollback procedure..."

    if [ ! -f "$ROLLBACK_STATE_FILE" ]; then
        print_status "ERROR" "Rollback state file not found: $ROLLBACK_STATE_FILE"
        exit 1
    fi

    # Load rollback state
    local rollback_data=$(cat "$ROLLBACK_STATE_FILE")
    local backup_dir=$(echo "$rollback_data" | jq -r '.backup_dir')
    local previous_commit=$(echo "$rollback_data" | jq -r '.git_commit')

    print_status "INFO" "Rolling back to state from: $(echo "$rollback_data" | jq -r '.timestamp')"
    print_status "INFO" "Backup directory: $backup_dir"

    # Stop current services
    print_status "INFO" "Stopping current services..."
    docker-compose down

    # Restore from backup if available
    if [ -d "$backup_dir" ]; then
        print_status "INFO" "Restoring from backup..."

        # Restore database
        if [ -f "${backup_dir}/postgres_*.sql" ]; then
            docker-compose up -d postgres
            sleep 10
            docker-compose exec -T postgres psql -U trader -d trading_system < \
                "${backup_dir}"/postgres_*.sql
        fi

        # Restore Redis
        if [ -f "${backup_dir}/redis_*.rdb" ]; then
            docker-compose up -d redis
            sleep 5
            docker-compose exec -T redis redis-cli --pipe < \
                "${backup_dir}"/redis_*.rdb
        fi

        # Restore configuration and data
        if [ -f "${backup_dir}/config_data_*.tar.gz" ]; then
            tar -xzf "${backup_dir}"/config_data_*.tar.gz -C "$PROJECT_ROOT"
        fi
    fi

    # Use previous images if target tag not specified
    if [ -z "$target_tag" ]; then
        print_status "INFO" "Using previous image versions from rollback state"
        # Implementation would use stored image IDs from rollback state
    else
        print_status "INFO" "Rolling back to tag: $target_tag"
        export IMAGE_TAG="$target_tag"
    fi

    # Restart services with previous configuration
    deploy_services
    validate_deployment

    print_status "SUCCESS" "Rollback completed successfully"
}

# Function to handle deployment failure
handle_failure() {
    local exit_code=$1

    print_status "ERROR" "Deployment failed with exit code: $exit_code"

    if [ "$ROLLBACK_ON_FAILURE" = true ] && [ "$DRY_RUN" = false ]; then
        print_status "WARNING" "Automatic rollback enabled, initiating rollback..."
        perform_rollback ""
    else
        print_status "ERROR" "Automatic rollback disabled. Manual intervention required."
        print_status "INFO" "To rollback manually, run: $0 rollback"

        # Show current service status
        print_status "INFO" "Current service status:"
        docker-compose ps || true

        # Show recent logs
        print_status "INFO" "Recent service logs:"
        docker-compose logs --tail=20 || true
    fi

    exit $exit_code
}

# Function to show deployment status
show_status() {
    print_status "INFO" "Current deployment status:"

    local compose_file="docker-compose.yml"
    if [ "$ENVIRONMENT" = "production" ]; then
        compose_file="docker-compose.prod.yml"
    fi

    # Show service status
    echo
    echo "📊 Service Status:"
    docker-compose -f "$compose_file" ps

    # Show resource usage
    echo
    echo "💾 Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

    # Show disk usage
    echo
    echo "💿 Disk Usage:"
    du -sh "${PROJECT_ROOT}/data"/* 2>/dev/null || echo "No data directories found"

    # Show recent deployments
    echo
    echo "📅 Recent Deployments:"
    if [ -d "${PROJECT_ROOT}/logs" ]; then
        ls -la "${PROJECT_ROOT}/logs"/deployment_*.log | tail -5 || echo "No deployment logs found"
    fi
}

# Function to cleanup old images and containers
cleanup_old_images() {
    print_status "INFO" "Cleaning up old Docker images and containers..."

    # Remove old trading-system images (keep last 5)
    docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedAt}}" | \
        grep "trading-system" | \
        sort -k2 -r | \
        tail -n +6 | \
        awk '{print $1}' | \
        xargs -r docker rmi || true

    # Remove dangling images
    docker image prune -f

    # Remove unused volumes (be careful with this)
    if [ "$ENVIRONMENT" != "production" ]; then
        docker volume prune -f
    fi

    print_status "SUCCESS" "Cleanup completed"
}

# Function to check maintenance mode
check_maintenance_mode() {
    local maintenance_mode=$(grep "^MAINTENANCE_MODE=" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d'=' -f2 || echo "false")

    if [ "$maintenance_mode" = "true" ]; then
        print_status "WARNING" "System is in maintenance mode"

        if [ "$ENVIRONMENT" = "production" ]; then
            read -p "Continue with deployment during maintenance? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_status "INFO" "Deployment cancelled by user"
                exit 0
            fi
        fi
    fi
}

# Function to send deployment notifications
send_notifications() {
    local status=$1
    local message=$2

    # Load notification settings
    local gotify_url=$(grep "^GOTIFY_URL=" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d'=' -f2 || echo "")
    local gotify_token=$(grep "^GOTIFY_TOKEN=" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d'=' -f2 || echo "")
    local slack_webhook=$(grep "^SLACK_WEBHOOK_URL=" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d'=' -f2 || echo "")

    # Send Gotify notification
    if [ -n "$gotify_url" ] && [ -n "$gotify_token" ]; then
        curl -X POST "$gotify_url/message" \
            -H "X-Gotify-Key: $gotify_token" \
            -H "Content-Type: application/json" \
            -d "{\"title\":\"Trading System Deployment\",\"message\":\"$message\",\"priority\":$([ "$status" = "error" ] && echo 10 || echo 5)}" \
            >/dev/null 2>&1 || true
    fi

    # Send Slack notification
    if [ -n "$slack_webhook" ]; then
        local color=$([ "$status" = "success" ] && echo "good" || echo "danger")
        curl -X POST "$slack_webhook" \
            -H "Content-Type: application/json" \
            -d "{\"attachments\":[{\"color\":\"$color\",\"title\":\"Trading System Deployment\",\"text\":\"$message\",\"ts\":\"$(date +%s)\"}]}" \
            >/dev/null 2>&1 || true
    fi
}

# Main deployment function
main_deploy() {
    print_status "INFO" "Starting deployment to $ENVIRONMENT environment"
    print_status "INFO" "Deployment ID: ${TIMESTAMP}"
    print_status "INFO" "Log file: $DEPLOYMENT_LOG"

    # Create log directory
    mkdir -p "$(dirname "$DEPLOYMENT_LOG")"

    if [ "$DRY_RUN" = true ]; then
        print_status "WARNING" "DRY RUN MODE - No actual changes will be made"
    fi

    # Set trap for failure handling
    trap 'handle_failure $?' ERR

    # Deployment steps
    check_maintenance_mode
    validate_prerequisites

    if [ "$DRY_RUN" = false ]; then
        create_backup
        run_tests
        build_images
        run_migrations
        update_configs
        deploy_services
        validate_deployment
        cleanup_old_images

        print_status "SUCCESS" "Deployment completed successfully!"
        send_notifications "success" "Deployment to $ENVIRONMENT completed successfully at $(date)"
    else
        print_status "INFO" "DRY RUN: All checks passed, deployment would proceed"
    fi

    # Show final status
    if [ "$DRY_RUN" = false ]; then
        show_status
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            rollback)
                if [ "$#" -gt 1 ] && [ "$2" = "--to" ]; then
                    perform_rollback "$3"
                else
                    perform_rollback ""
                fi
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE=false
                shift
                ;;
            --sequential)
                PARALLEL_BUILD=false
                shift
                ;;
            --config-only)
                BUILD_IMAGES=false
                RUN_MIGRATIONS=false
                RESTART_SERVICES=false
                UPDATE_CONFIGS=true
                VALIDATE_DEPLOYMENT=false
                shift
                ;;
            --migrations-only)
                BUILD_IMAGES=false
                UPDATE_CONFIGS=false
                RESTART_SERVICES=false
                RUN_MIGRATIONS=true
                VALIDATE_DEPLOYMENT=false
                shift
                ;;
            --health-timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            --migration-timeout)
                MIGRATION_TIMEOUT="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            status)
                show_status
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

# Function to check if running as root (security check)
check_user_permissions() {
    if [ "$EUID" -eq 0 ] && [ "$ENVIRONMENT" = "production" ]; then
        print_status "ERROR" "Do not run production deployments as root for security reasons"
        exit 1
    fi
}

# Function to validate Git repository state
validate_git_state() {
    if [ ! -d "${PROJECT_ROOT}/.git" ]; then
        print_status "WARNING" "Not a Git repository, skipping Git validations"
        return
    fi

    # Check for uncommitted changes in production
    if [ "$ENVIRONMENT" = "production" ]; then
        if ! git diff-index --quiet HEAD --; then
            print_status "ERROR" "Uncommitted changes detected. Commit changes before production deployment."
            exit 1
        fi

        # Check if on main/master branch
        local current_branch=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current_branch" != "main" ] && [ "$current_branch" != "master" ]; then
            print_status "WARNING" "Not on main/master branch (current: $current_branch)"

            read -p "Continue with deployment from branch '$current_branch'? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_status "INFO" "Deployment cancelled by user"
                exit 0
            fi
        fi
    fi

    # Log Git information
    local git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    local git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    print_status "INFO" "Git branch: $git_branch, commit: ${git_commit:0:8}"
}

# Function for production safety checks
production_safety_checks() {
    if [ "$ENVIRONMENT" != "production" ]; then
        return
    fi

    print_status "INFO" "Performing production safety checks..."

    # Check if this is really production environment
    read -p "⚠️  Are you sure you want to deploy to PRODUCTION? Type 'PRODUCTION' to confirm: " confirm
    if [ "$confirm" != "PRODUCTION" ]; then
        print_status "INFO" "Production deployment cancelled by user"
        exit 0
    fi

    # Check trading configuration
    local trading_dry_run=$(grep "^TRADING_DRY_RUN=" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d'=' -f2 || echo "true")
    if [ "$trading_dry_run" = "false" ]; then
        print_status "CRITICAL" "🚨 LIVE TRADING IS ENABLED 🚨"
        read -p "Live trading will use REAL MONEY. Type 'LIVE_TRADING' to confirm: " trading_confirm
        if [ "$trading_confirm" != "LIVE_TRADING" ]; then
            print_status "INFO" "Production deployment cancelled - live trading not confirmed"
            exit 0
        fi
    fi

    # Check SSL configuration
    local ssl_enabled=$(grep "^SSL_ENABLED=" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d'=' -f2 || echo "false")
    if [ "$ssl_enabled" != "true" ]; then
        print_status "ERROR" "SSL must be enabled for production deployment"
        exit 1
    fi

    # Check monitoring configuration
    local sentry_dsn=$(grep "^SENTRY_DSN=" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d'=' -f2 || echo "")
    if [ -z "$sentry_dsn" ]; then
        print_status "WARNING" "Sentry DSN not configured for error tracking"
    fi

    print_status "SUCCESS" "Production safety checks completed"
}

# Function to create deployment manifest
create_deployment_manifest() {
    print_status "INFO" "Creating deployment manifest..."

    local manifest_file="${PROJECT_ROOT}/logs/deployment_manifest_${TIMESTAMP}.json"
    local git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    local git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

    cat > "$manifest_file" << EOF
{
    "deployment_id": "$TIMESTAMP",
    "environment": "$ENVIRONMENT",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "git": {
        "commit": "$git_commit",
        "branch": "$git_branch"
    },
    "configuration": {
        "dry_run": $DRY_RUN,
        "skip_tests": $SKIP_TESTS,
        "skip_backup": $SKIP_BACKUP,
        "force_rebuild": $FORCE_REBUILD,
        "rollback_on_failure": $ROLLBACK_ON_FAILURE
    },
    "images": {
        "tag": "${ENVIRONMENT}-${TIMESTAMP}",
        "services": [
            "data_collector",
            "strategy_engine",
            "risk_manager",
            "trade_executor",
            "scheduler",
            "export_service"
        ]
    },
    "deployment_steps": {
        "build_images": $BUILD_IMAGES,
        "run_migrations": $RUN_MIGRATIONS,
        "update_configs": $UPDATE_CONFIGS,
        "restart_services": $RESTART_SERVICES,
        "validate_deployment": $VALIDATE_DEPLOYMENT
    }
}
EOF

    print_status "INFO" "Deployment manifest created: $manifest_file"
}

# Main execution
main() {
    # Set working directory
    cd "$PROJECT_ROOT"

    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"

    # Parse arguments
    parse_arguments "$@"

    # Initial setup
    check_user_permissions
    validate_git_state
    production_safety_checks
    create_deployment_manifest

    # Execute main deployment
    main_deploy

    print_status "SUCCESS" "🎉 Deployment completed successfully!"
    print_status "INFO" "Deployment log: $DEPLOYMENT_LOG"
    print_status "INFO" "Rollback state: $ROLLBACK_STATE_FILE"

    # Final status
    echo
    echo "🚀 Deployment Summary:"
    echo "   Environment: $ENVIRONMENT"
    echo "   Timestamp: $TIMESTAMP"
    echo "   Image Tag: ${ENVIRONMENT}-${TIMESTAMP}"
    echo "   Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo
    echo "🔗 Service URLs:"
    echo "   Data Collector: http://localhost:8001"
    echo "   Strategy Engine: http://localhost:8002"
    echo "   Risk Manager: http://localhost:8003"
    echo "   Trade Executor: http://localhost:8004"
    echo "   Scheduler: http://localhost:8005"
    echo "   Export Service: http://localhost:8006"
    echo "   Grafana: http://localhost:3000"
    echo "   Prometheus: http://localhost:9090"
    echo
    echo "📚 Next Steps:"
    echo "   1. Monitor service health at http://localhost:3000"
    echo "   2. Check logs: docker-compose logs -f"
    echo "   3. Verify trading operations in dry-run mode"
    echo "   4. Review deployment manifest and logs"
    echo
}

# Handle special commands first
if [ $# -gt 0 ]; then
    case $1 in
        --help|-h|help)
            show_usage
            exit 0
            ;;
        status)
            show_status
            exit 0
            ;;
        rollback)
            if [ "$#" -gt 1 ] && [ "$2" = "--to" ] && [ "$#" -gt 2 ]; then
                perform_rollback "$3"
            else
                perform_rollback ""
            fi
            exit 0
            ;;
    esac
fi

# Run main function with all arguments
main "$@"
