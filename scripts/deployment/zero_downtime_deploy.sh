#!/bin/bash

# AI Trading System Zero-Downtime Deployment Script
# This script performs rolling deployments with health checks and automatic rollback
# Usage: ./zero_downtime_deploy.sh [environment] [options]

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
DEPLOYMENT_LOG="${PROJECT_ROOT}/logs/zero_downtime_${TIMESTAMP}.log"

# Default values
ENVIRONMENT="production"
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL=10
DEPLOYMENT_TIMEOUT=1800
ROLLBACK_ON_FAILURE=true
BACKUP_BEFORE_DEPLOY=true
PARALLEL_DEPLOYMENT=false
CANARY_PERCENTAGE=20

# Service deployment order (dependencies first)
DEPLOYMENT_ORDER=(
    "postgres"
    "redis"
    "data_collector"
    "risk_manager"
    "strategy_engine"
    "trade_executor"
    "scheduler"
    "export_service"
    "maintenance_service"
)

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
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
Zero-Downtime Deployment Script for AI Trading System

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    staging         Deploy to staging environment
    production      Deploy to production environment (default)

OPTIONS:
    --no-backup             Skip backup creation before deployment
    --no-rollback           Disable automatic rollback on failure
    --parallel              Deploy services in parallel (faster but riskier)
    --canary PERCENT        Deploy to percentage of instances first (default: 20)
    --timeout SECONDS       Deployment timeout in seconds (default: 1800)
    --health-retries NUM    Health check retry count (default: 10)
    --health-interval SEC   Health check interval in seconds (default: 10)
    --dry-run              Show what would be done without executing
    --force                 Force deployment even with warnings
    --help                 Show this help message

EXAMPLES:
    $0 production
    $0 staging --parallel --no-backup
    $0 production --canary 50 --timeout 3600
    $0 production --dry-run

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            production|staging)
                ENVIRONMENT="$1"
                shift
                ;;
            --no-backup)
                BACKUP_BEFORE_DEPLOY=false
                shift
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE=false
                shift
                ;;
            --parallel)
                PARALLEL_DEPLOYMENT=true
                shift
                ;;
            --canary)
                CANARY_PERCENTAGE="$2"
                shift 2
                ;;
            --timeout)
                DEPLOYMENT_TIMEOUT="$2"
                shift 2
                ;;
            --health-retries)
                HEALTH_CHECK_RETRIES="$2"
                shift 2
                ;;
            --health-interval)
                HEALTH_CHECK_INTERVAL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE_DEPLOYMENT=true
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
    print_status "INFO" "Checking deployment prerequisites..."

    # Check required tools
    local required_tools=("docker" "docker-compose" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_status "ERROR" "Required tool not found: $tool"
            exit 1
        fi
    done

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_status "ERROR" "Docker daemon is not running"
        exit 1
    fi

    # Validate environment configuration
    local config_file="${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}"
    if [ ! -f "$config_file" ]; then
        print_status "ERROR" "Environment configuration not found: $config_file"
        exit 1
    fi

    # Load environment variables
    export $(grep -v '^#' "$config_file" | xargs)

    # Validate critical environment variables
    local required_vars=("DB_PASSWORD" "REDIS_PASSWORD" "API_SECRET_KEY")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            print_status "ERROR" "Required environment variable not set: $var"
            exit 1
        fi
    done

    print_status "SUCCESS" "Prerequisites check completed"
}

# Function to create pre-deployment backup
create_backup() {
    if [ "$BACKUP_BEFORE_DEPLOY" = true ]; then
        print_status "INFO" "Creating pre-deployment backup..."

        local backup_tag="pre_deploy_${TIMESTAMP}"
        if ./scripts/backup/backup.sh --tag "$backup_tag" --compress; then
            print_status "SUCCESS" "Pre-deployment backup created: $backup_tag"
            echo "$backup_tag" > "${PROJECT_ROOT}/data/last_backup_tag.txt"
        else
            print_status "ERROR" "Failed to create pre-deployment backup"
            exit 1
        fi
    else
        print_status "WARNING" "Skipping pre-deployment backup"
    fi
}

# Function to check service health
check_service_health() {
    local service_name=$1
    local port=$2
    local retries=${3:-$HEALTH_CHECK_RETRIES}
    local interval=${4:-$HEALTH_CHECK_INTERVAL}

    print_status "INFO" "Checking health for $service_name on port $port"

    for ((i=1; i<=retries; i++)); do
        if curl -f -s "http://localhost:$port/health" > /dev/null 2>&1; then
            print_status "SUCCESS" "$service_name is healthy (attempt $i/$retries)"
            return 0
        fi

        if [ $i -lt $retries ]; then
            print_status "WARNING" "$service_name health check failed (attempt $i/$retries), retrying in ${interval}s..."
            sleep "$interval"
        fi
    done

    print_status "ERROR" "$service_name failed health check after $retries attempts"
    return 1
}

# Function to get service port
get_service_port() {
    local service=$1
    case $service in
        "data_collector") echo "9101" ;;
        "strategy_engine") echo "9102" ;;
        "risk_manager") echo "9103" ;;
        "trade_executor") echo "9104" ;;
        "scheduler") echo "9105" ;;
        "export_service") echo "9106" ;;
        "maintenance_service") echo "9107" ;;
        *) echo "" ;;
    esac
}

# Function to get current image tag
get_current_image_tag() {
    local service=$1
    docker-compose images -q "$service" 2>/dev/null || echo ""
}

# Function to perform canary deployment
deploy_canary() {
    local service=$1
    local new_image=$2

    print_status "INFO" "Starting canary deployment for $service ($CANARY_PERCENTAGE%)"

    # For simplicity, we'll deploy to a test instance first
    # In a real production setup, this would deploy to a subset of load-balanced instances

    # Create canary service name
    local canary_service="${service}_canary"
    local service_port=$(get_service_port "$service")
    local canary_port=$((service_port + 1000))

    # Start canary instance
    docker run -d \
        --name "$canary_service" \
        --network "full-ai-trader_trading_network" \
        -p "$canary_port:$service_port" \
        -e "SERVICE_PORT=$service_port" \
        --env-file "config/environments/.env.${ENVIRONMENT}" \
        "$new_image" || {
        print_status "ERROR" "Failed to start canary instance for $service"
        return 1
    }

    # Wait for canary to be healthy
    if check_service_health "$canary_service" "$canary_port" 5 5; then
        print_status "SUCCESS" "Canary deployment successful for $service"

        # Run canary tests
        if run_canary_tests "$service" "$canary_port"; then
            print_status "SUCCESS" "Canary tests passed for $service"
            # Clean up canary
            docker stop "$canary_service" && docker rm "$canary_service"
            return 0
        else
            print_status "ERROR" "Canary tests failed for $service"
            docker stop "$canary_service" && docker rm "$canary_service"
            return 1
        fi
    else
        print_status "ERROR" "Canary health check failed for $service"
        docker stop "$canary_service" && docker rm "$canary_service" 2>/dev/null || true
        return 1
    fi
}

# Function to run canary tests
run_canary_tests() {
    local service=$1
    local port=$2

    print_status "INFO" "Running canary tests for $service on port $port"

    # Basic health check
    if ! curl -f -s "http://localhost:$port/health" > /dev/null; then
        return 1
    fi

    # Service-specific tests
    case $service in
        "data_collector")
            curl -f -s "http://localhost:$port/data/status" > /dev/null
            ;;
        "strategy_engine")
            curl -f -s "http://localhost:$port/strategies/status" > /dev/null
            ;;
        "risk_manager")
            curl -f -s "http://localhost:$port/risk/status" > /dev/null
            ;;
        "trade_executor")
            curl -f -s "http://localhost:$port/broker/status" > /dev/null
            ;;
    esac

    return $?
}

# Function to deploy a single service with zero downtime
deploy_service_zero_downtime() {
    local service=$1
    local service_port=$(get_service_port "$service")

    print_status "INFO" "Starting zero-downtime deployment for $service"

    # Skip infrastructure services (they need special handling)
    if [[ "$service" == "postgres" || "$service" == "redis" ]]; then
        print_status "INFO" "Skipping zero-downtime deployment for infrastructure service: $service"
        return 0
    fi

    # Get current image tag for rollback
    local current_image=$(get_current_image_tag "$service")
    if [ -n "$current_image" ]; then
        echo "$current_image" > "${PROJECT_ROOT}/data/rollback_${service}_${TIMESTAMP}.txt"
    fi

    # Build new image
    print_status "INFO" "Building new image for $service"
    if ! docker-compose build "$service"; then
        print_status "ERROR" "Failed to build new image for $service"
        return 1
    fi

    # Perform canary deployment if enabled
    if [ "$CANARY_PERCENTAGE" -gt 0 ] && [ "$CANARY_PERCENTAGE" -lt 100 ]; then
        local new_image=$(docker-compose images -q "$service")
        if ! deploy_canary "$service" "$new_image"; then
            print_status "ERROR" "Canary deployment failed for $service"
            return 1
        fi
    fi

    # Create new service instance with temporary name
    local temp_service="${service}_new"
    local temp_port=$((service_port + 2000))

    print_status "INFO" "Starting new instance of $service on port $temp_port"

    # Start new instance
    docker-compose run -d \
        --name "$temp_service" \
        -p "$temp_port:$service_port" \
        "$service" || {
        print_status "ERROR" "Failed to start new instance of $service"
        return 1
    }

    # Wait for new instance to be healthy
    if check_service_health "$temp_service" "$temp_port"; then
        print_status "SUCCESS" "New instance of $service is healthy"

        # Switch traffic to new instance
        print_status "INFO" "Switching traffic to new $service instance"

        # Stop old instance
        docker-compose stop "$service"

        # Remove port mapping from new instance and start with correct name
        docker stop "$temp_service"
        docker rm "$temp_service"

        # Start service with production configuration
        docker-compose up -d "$service"

        # Final health check
        if check_service_health "$service" "$service_port"; then
            print_status "SUCCESS" "Zero-downtime deployment completed for $service"
            return 0
        else
            print_status "ERROR" "Final health check failed for $service"
            # Attempt rollback
            if [ "$ROLLBACK_ON_FAILURE" = true ] && [ -n "$current_image" ]; then
                rollback_service "$service" "$current_image"
            fi
            return 1
        fi
    else
        print_status "ERROR" "New instance of $service failed health check"
        docker stop "$temp_service" && docker rm "$temp_service" 2>/dev/null || true
        return 1
    fi
}

# Function to rollback a service
rollback_service() {
    local service=$1
    local previous_image=$2

    print_status "WARNING" "Rolling back $service to previous image"

    # Stop current service
    docker-compose stop "$service"

    # Tag previous image as latest
    docker tag "$previous_image" "${service}:latest"

    # Start service with previous image
    docker-compose up -d "$service"

    # Check health
    local service_port=$(get_service_port "$service")
    if check_service_health "$service" "$service_port" 5 5; then
        print_status "SUCCESS" "Rollback completed for $service"
        return 0
    else
        print_status "ERROR" "Rollback failed for $service"
        return 1
    fi
}

# Function to check system trading status
check_trading_status() {
    print_status "INFO" "Checking trading system status"

    # Check if trading is enabled
    local trading_status=$(curl -s http://localhost:9104/trading/status | jq -r '.trading_enabled' 2>/dev/null || echo "unknown")

    if [ "$trading_status" = "true" ]; then
        print_status "SUCCESS" "Trading is enabled and active"
        return 0
    elif [ "$trading_status" = "false" ]; then
        print_status "WARNING" "Trading is disabled"
        return 1
    else
        print_status "ERROR" "Unable to determine trading status"
        return 1
    fi
}

# Function to pause trading during deployment
pause_trading() {
    print_status "INFO" "Pausing trading operations for deployment"

    # Set trading to read-only mode
    curl -X POST http://localhost:9107/maintenance/enter \
        -H "Content-Type: application/json" \
        -d "{\"mode\": \"read_only\", \"reason\": \"Zero-downtime deployment in progress\", \"estimated_minutes\": 30}" \
        > /dev/null 2>&1 || {
        print_status "WARNING" "Failed to set maintenance mode, continuing with deployment"
    }

    # Wait for current trades to complete
    print_status "INFO" "Waiting for pending trades to complete..."
    local max_wait=300  # 5 minutes
    local waited=0

    while [ $waited -lt $max_wait ]; do
        local pending_trades=$(curl -s http://localhost:9104/trades/pending/count 2>/dev/null || echo "0")

        if [ "$pending_trades" = "0" ]; then
            print_status "SUCCESS" "No pending trades, safe to proceed"
            break
        fi

        print_status "INFO" "Waiting for $pending_trades pending trades to complete..."
        sleep 10
        waited=$((waited + 10))
    done

    if [ $waited -ge $max_wait ]; then
        print_status "WARNING" "Timeout waiting for trades to complete, proceeding with deployment"
    fi
}

# Function to resume trading after deployment
resume_trading() {
    print_status "INFO" "Resuming normal trading operations"

    # Exit maintenance mode
    curl -X POST http://localhost:9107/maintenance/exit \
        > /dev/null 2>&1 || {
        print_status "WARNING" "Failed to exit maintenance mode automatically"
    }

    # Verify trading is resumed
    sleep 5
    if check_trading_status; then
        print_status "SUCCESS" "Trading operations resumed successfully"
    else
        print_status "WARNING" "Trading may not have resumed properly, manual check required"
    fi
}

# Function to validate deployment
validate_deployment() {
    print_status "INFO" "Validating deployment..."

    local validation_failed=false

    # Check all services are healthy
    for service in "${DEPLOYMENT_ORDER[@]}"; do
        local port=$(get_service_port "$service")
        if [ -n "$port" ]; then
            if ! check_service_health "$service" "$port" 3 5; then
                print_status "ERROR" "Service validation failed: $service"
                validation_failed=true
            fi
        fi
    done

    # Check database connectivity
    if ! docker-compose exec postgres pg_isready -U trader > /dev/null 2>&1; then
        print_status "ERROR" "Database connectivity validation failed"
        validation_failed=true
    fi

    # Check Redis connectivity
    if ! docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
        print_status "ERROR" "Redis connectivity validation failed"
        validation_failed=true
    fi

    # Check inter-service communication
    if ! curl -f -s http://localhost:9102/health > /dev/null; then
        print_status "ERROR" "Inter-service communication validation failed"
        validation_failed=true
    fi

    # Run integration tests
    print_status "INFO" "Running post-deployment integration tests"
    if ! python "${PROJECT_ROOT}/tests/integration/test_deployment.py" --quick; then
        print_status "ERROR" "Integration tests failed"
        validation_failed=true
    fi

    if [ "$validation_failed" = true ]; then
        print_status "ERROR" "Deployment validation failed"
        return 1
    else
        print_status "SUCCESS" "Deployment validation passed"
        return 0
    fi
}

# Function to perform rolling deployment
perform_rolling_deployment() {
    print_status "INFO" "Starting rolling deployment"

    local failed_services=()

    for service in "${DEPLOYMENT_ORDER[@]}"; do
        # Skip infrastructure services for rolling deployment
        if [[ "$service" == "postgres" || "$service" == "redis" ]]; then
            continue
        fi

        print_status "INFO" "Deploying service: $service"

        if [ "$PARALLEL_DEPLOYMENT" = true ]; then
            # Deploy in background for parallel deployment
            deploy_service_zero_downtime "$service" &
        else
            # Deploy sequentially
            if ! deploy_service_zero_downtime "$service"; then
                failed_services+=("$service")

                if [ "$ROLLBACK_ON_FAILURE" = true ]; then
                    print_status "ERROR" "Service deployment failed: $service, initiating rollback"
                    rollback_deployment
                    return 1
                fi
            fi
        fi
    done

    # Wait for parallel deployments to complete
    if [ "$PARALLEL_DEPLOYMENT" = true ]; then
        wait

        # Check which services failed
        for service in "${DEPLOYMENT_ORDER[@]}"; do
            if [[ "$service" != "postgres" && "$service" != "redis" ]]; then
                local port=$(get_service_port "$service")
                if [ -n "$port" ] && ! check_service_health "$service" "$port" 1 1; then
                    failed_services+=("$service")
                fi
            fi
        done
    fi

    if [ ${#failed_services[@]} -gt 0 ]; then
        print_status "ERROR" "Failed services: ${failed_services[*]}"
        return 1
    else
        print_status "SUCCESS" "All services deployed successfully"
        return 0
    fi
}

# Function to rollback entire deployment
rollback_deployment() {
    print_status "WARNING" "Rolling back entire deployment"

    # Read rollback information
    local rollback_files=("${PROJECT_ROOT}"/data/rollback_*_"${TIMESTAMP}".txt)

    for rollback_file in "${rollback_files[@]}"; do
        if [ -f "$rollback_file" ]; then
            local service=$(basename "$rollback_file" | sed "s/rollback_\(.*\)_${TIMESTAMP}.txt/\1/")
            local previous_image=$(cat "$rollback_file")

            if [ -n "$previous_image" ]; then
                rollback_service "$service" "$previous_image"
            fi
        fi
    done

    # Resume trading if it was paused
    resume_trading
}

# Function to update infrastructure services safely
update_infrastructure() {
    print_status "INFO" "Updating infrastructure services"

    # Update Redis (with data preservation)
    if docker-compose ps redis | grep -q "Up"; then
        print_status "INFO" "Updating Redis with data preservation"

        # Create Redis backup
        docker-compose exec redis redis-cli BGSAVE

        # Wait for backup to complete
        while [ "$(docker-compose exec redis redis-cli LASTSAVE)" = "$(docker-compose exec redis redis-cli LASTSAVE)" ]; do
            sleep 1
        done

        # Update Redis
        docker-compose up -d redis

        # Verify Redis is healthy
        if ! check_service_health "redis" "6379" 5 5; then
            print_status "ERROR" "Redis update failed"
            return 1
        fi
    fi

    # Update PostgreSQL (requires careful handling)
    if docker-compose ps postgres | grep -q "Up"; then
        print_status "INFO" "PostgreSQL updates require maintenance window"
        print_status "WARNING" "Skipping PostgreSQL update - use maintenance window"
    fi

    return 0
}

# Function to send deployment notifications
send_deployment_notification() {
    local status=$1
    local message=$2

    if [ -n "${GOTIFY_URL:-}" ] && [ -n "${GOTIFY_TOKEN:-}" ]; then
        curl -X POST "${GOTIFY_URL}/message" \
            -H "X-Gotify-Key: ${GOTIFY_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"title\": \"Deployment ${status}\",
                \"message\": \"${message}\",
                \"priority\": $([ "$status" = "SUCCESS" ] && echo "5" || echo "8")
            }" > /dev/null 2>&1 || {
            print_status "WARNING" "Failed to send deployment notification"
        }
    fi
}

# Function to monitor deployment progress
monitor_deployment() {
    local deployment_start=$(date +%s)

    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - deployment_start))

        if [ $elapsed -gt $DEPLOYMENT_TIMEOUT ]; then
            print_status "ERROR" "Deployment timeout exceeded ($DEPLOYMENT_TIMEOUT seconds)"
            return 1
        fi

        # Check if all services are healthy
        local all_healthy=true
        for service in "${DEPLOYMENT_ORDER[@]}"; do
            local port=$(get_service_port "$service")
            if [ -n "$port" ] && ! curl -f -s "http://localhost:$port/health" > /dev/null 2>&1; then
                all_healthy=false
                break
            fi
        done

        if [ "$all_healthy" = true ]; then
            print_status "SUCCESS" "All services are healthy"
            break
        fi

        sleep 10
    done

    return 0
}

# Main deployment function
main() {
    local start_time=$(date +%s)

    print_status "INFO" "Starting zero-downtime deployment for $ENVIRONMENT environment"

    # Parse arguments
    parse_arguments "$@"

    # Check prerequisites
    check_prerequisites

    # Create backup
    create_backup

    # Pause trading operations
    pause_trading

    # Update infrastructure services first
    if ! update_infrastructure; then
        print_status "ERROR" "Infrastructure update failed"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback_deployment
        fi
        exit 1
    fi

    # Perform rolling deployment
    if ! perform_rolling_deployment; then
        print_status "ERROR" "Rolling deployment failed"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback_deployment
        fi
        exit 1
    fi

    # Validate deployment
    if ! validate_deployment; then
        print_status "ERROR" "Deployment validation failed"
        if [ "$ROLLBACK_ON_FAILURE" = true ]; then
            rollback_deployment
        fi
        exit 1
    fi

    # Resume trading operations
    resume_trading

    # Monitor deployment for stability
    if ! monitor_deployment; then
        print_status "ERROR" "Deployment monitoring failed"
        exit 1
    fi

    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))

    print_status "SUCCESS" "Zero-downtime deployment completed successfully in ${total_time} seconds"

    # Send success notification
    send_deployment_notification "SUCCESS" "Zero-downtime deployment to $ENVIRONMENT completed in ${total_time}s"

    # Cleanup rollback files
    rm -f "${PROJECT_ROOT}"/data/rollback_*_"${TIMESTAMP}".txt
}

# Cleanup function
cleanup() {
    print_status "INFO" "Cleaning up deployment artifacts"

    # Remove any temporary containers
    for service in "${DEPLOYMENT_ORDER[@]}"; do
        docker rm -f "${service}_new" 2>/dev/null || true
        docker rm -f "${service}_canary" 2>/dev/null || true
    done

    # Resume trading if it was paused
    resume_trading 2>/dev/null || true
}

# Set up trap for cleanup
trap cleanup EXIT

# Handle signals gracefully
trap 'print_status "WARNING" "Deployment interrupted by signal"; exit 130' INT TERM

# Main execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
