#!/bin/bash

# AI Trading System Operations Manager
# Centralized script for managing system operations
# Usage: ./ops_manager.sh [command] [options]

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
OPS_LOG="${PROJECT_ROOT}/logs/operations/ops_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs/operations"

# Default configuration
ENVIRONMENT="development"
COMPOSE_FILE="docker-compose.yml"
WAIT_TIMEOUT=300
FORCE_ACTION=false
VERBOSE=false
DRY_RUN=false
PARALLEL_OPERATIONS=true

# Service groups for ordered operations
INFRASTRUCTURE_SERVICES=("postgres" "redis")
CORE_SERVICES=("data_collector" "strategy_engine" "risk_manager" "trade_executor" "scheduler")
SUPPORT_SERVICES=("export_service" "maintenance_service")
MONITORING_SERVICES=("prometheus" "grafana" "alertmanager" "elasticsearch" "kibana" "logstash")
ALL_SERVICES=("${INFRASTRUCTURE_SERVICES[@]}" "${CORE_SERVICES[@]}" "${SUPPORT_SERVICES[@]}" "${MONITORING_SERVICES[@]}")

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message"
            ;;
    esac

    # Log to file
    echo "[$level] $timestamp - $message" >> "$OPS_LOG"
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Operations Manager

Usage: $0 COMMAND [OPTIONS]

COMMANDS:
    start                   Start all services
    stop                    Stop all services
    restart                 Restart all services
    status                  Show status of all services
    logs                    Show logs from services
    health                  Run comprehensive health check
    scale                   Scale services up/down
    backup                  Create system backup
    restore                 Restore from backup
    maintenance             Enter/exit maintenance mode
    emergency-stop          Emergency shutdown of all services
    cleanup                 Clean up unused resources
    update                  Update service configurations

SERVICE COMMANDS:
    start-core              Start only core trading services
    stop-core               Stop only core trading services
    start-monitoring        Start only monitoring services
    stop-monitoring         Stop only monitoring services
    start-infrastructure    Start only infrastructure services
    stop-infrastructure     Stop only infrastructure services

OPTIONS:
    --env ENV               Environment (development/staging/production)
    --compose-file FILE     Docker compose file to use
    --timeout SEC           Wait timeout in seconds (default: 300)
    --force                 Force action without confirmation
    --verbose               Enable verbose output
    --dry-run               Show what would be done without executing
    --sequential            Run operations sequentially instead of parallel
    --service NAME          Target specific service
    --help                  Show this help message

EXAMPLES:
    $0 start --env production
    $0 restart --service data_collector --verbose
    $0 stop --force --env staging
    $0 health --timeout 60
    $0 scale --service strategy_engine --replicas 3
    $0 maintenance --enter --message "System update"
EOF
}

# Parse command line arguments
parse_args() {
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi

    COMMAND="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENVIRONMENT="$2"
                case $ENVIRONMENT in
                    "production")
                        COMPOSE_FILE="docker-compose.prod.yml"
                        ;;
                    "staging")
                        COMPOSE_FILE="docker-compose.staging.yml"
                        ;;
                    "development")
                        COMPOSE_FILE="docker-compose.yml"
                        ;;
                esac
                shift 2
                ;;
            --compose-file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            --timeout)
                WAIT_TIMEOUT="$2"
                shift 2
                ;;
            --force)
                FORCE_ACTION=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --sequential)
                PARALLEL_OPERATIONS=false
                shift
                ;;
            --service)
                TARGET_SERVICE="$2"
                shift 2
                ;;
            --replicas)
                REPLICAS="$2"
                shift 2
                ;;
            --enter)
                MAINTENANCE_ACTION="enter"
                shift
                ;;
            --exit)
                MAINTENANCE_ACTION="exit"
                shift
                ;;
            --message)
                MAINTENANCE_MESSAGE="$2"
                shift 2
                ;;
            --backup-id)
                BACKUP_ID="$2"
                shift 2
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

# Function to check if compose file exists
check_compose_file() {
    if [[ ! -f "${PROJECT_ROOT}/${COMPOSE_FILE}" ]]; then
        print_status "ERROR" "Docker compose file not found: ${COMPOSE_FILE}"
        exit 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local service=$1
    local timeout=${2:-$WAIT_TIMEOUT}
    local check_type=${3:-"health"}

    print_status "INFO" "Waiting for $service to be ready (timeout: ${timeout}s)..."

    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        case $check_type in
            "health")
                if docker-compose -f "$COMPOSE_FILE" exec -T "$service" sh -c 'exit 0' &>/dev/null; then
                    if [[ "$service" == "postgres" ]]; then
                        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U trading_user -d trading_db &>/dev/null; then
                            break
                        fi
                    elif [[ "$service" == "redis" ]]; then
                        if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping &>/dev/null; then
                            break
                        fi
                    else
                        # For other services, try to connect to their health endpoint
                        local port=$(docker-compose -f "$COMPOSE_FILE" port "$service" 8000 2>/dev/null | cut -d: -f2 || echo "")
                        if [[ -n "$port" ]] && curl -s -f "http://localhost:$port/health" &>/dev/null; then
                            break
                        elif docker-compose -f "$COMPOSE_FILE" exec -T "$service" sh -c 'exit 0' &>/dev/null; then
                            break
                        fi
                    fi
                fi
                ;;
            "running")
                if docker-compose -f "$COMPOSE_FILE" ps -q "$service" | xargs docker inspect -f '{{.State.Running}}' | grep -q true; then
                    break
                fi
                ;;
        esac

        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [[ $elapsed -ge $timeout ]]; then
        print_status "ERROR" "Service $service failed to become ready within ${timeout}s"
        return 1
    fi

    print_status "SUCCESS" "Service $service is ready"
}

# Function to start services
start_services() {
    local services=("$@")

    if [[ ${#services[@]} -eq 0 ]]; then
        services=("${ALL_SERVICES[@]}")
    fi

    print_status "INFO" "Starting services: ${services[*]}"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would start: ${services[*]}"
        return 0
    fi

    cd "$PROJECT_ROOT"

    # Start infrastructure services first
    local infra_to_start=()
    for service in "${services[@]}"; do
        if [[ " ${INFRASTRUCTURE_SERVICES[*]} " =~ " ${service} " ]]; then
            infra_to_start+=("$service")
        fi
    done

    if [[ ${#infra_to_start[@]} -gt 0 ]]; then
        print_status "INFO" "Starting infrastructure services: ${infra_to_start[*]}"
        docker-compose -f "$COMPOSE_FILE" up -d "${infra_to_start[@]}"

        for service in "${infra_to_start[@]}"; do
            wait_for_service "$service"
        done
    fi

    # Start core services
    local core_to_start=()
    for service in "${services[@]}"; do
        if [[ " ${CORE_SERVICES[*]} " =~ " ${service} " ]]; then
            core_to_start+=("$service")
        fi
    done

    if [[ ${#core_to_start[@]} -gt 0 ]]; then
        print_status "INFO" "Starting core services: ${core_to_start[*]}"
        if [[ "$PARALLEL_OPERATIONS" == "true" ]]; then
            docker-compose -f "$COMPOSE_FILE" up -d "${core_to_start[@]}"
            for service in "${core_to_start[@]}"; do
                wait_for_service "$service" &
            done
            wait
        else
            for service in "${core_to_start[@]}"; do
                docker-compose -f "$COMPOSE_FILE" up -d "$service"
                wait_for_service "$service"
            done
        fi
    fi

    # Start remaining services
    local remaining_to_start=()
    for service in "${services[@]}"; do
        if [[ ! " ${INFRASTRUCTURE_SERVICES[*]} ${CORE_SERVICES[*]} " =~ " ${service} " ]]; then
            remaining_to_start+=("$service")
        fi
    done

    if [[ ${#remaining_to_start[@]} -gt 0 ]]; then
        print_status "INFO" "Starting remaining services: ${remaining_to_start[*]}"
        docker-compose -f "$COMPOSE_FILE" up -d "${remaining_to_start[@]}"
    fi

    print_status "SUCCESS" "All requested services started"
}

# Function to stop services
stop_services() {
    local services=("$@")

    if [[ ${#services[@]} -eq 0 ]]; then
        services=("${ALL_SERVICES[@]}")
    fi

    print_status "INFO" "Stopping services: ${services[*]}"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would stop: ${services[*]}"
        return 0
    fi

    if [[ "$FORCE_ACTION" != "true" ]] && [[ ${#services[@]} -gt 5 ]]; then
        echo -n "Are you sure you want to stop ${#services[@]} services? (y/N): "
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "INFO" "Operation cancelled"
            return 0
        fi
    fi

    cd "$PROJECT_ROOT"

    # Stop in reverse order: support -> core -> infrastructure
    local services_to_stop=()

    # Add support and monitoring services first
    for service in "${services[@]}"; do
        if [[ " ${SUPPORT_SERVICES[*]} ${MONITORING_SERVICES[*]} " =~ " ${service} " ]]; then
            services_to_stop+=("$service")
        fi
    done

    # Add core services
    for service in "${services[@]}"; do
        if [[ " ${CORE_SERVICES[*]} " =~ " ${service} " ]]; then
            services_to_stop+=("$service")
        fi
    done

    # Add infrastructure services last
    for service in "${services[@]}"; do
        if [[ " ${INFRASTRUCTURE_SERVICES[*]} " =~ " ${service} " ]]; then
            services_to_stop+=("$service")
        fi
    done

    # Stop services
    if [[ ${#services_to_stop[@]} -gt 0 ]]; then
        print_status "INFO" "Stopping services in order: ${services_to_stop[*]}"
        docker-compose -f "$COMPOSE_FILE" stop "${services_to_stop[@]}"
    fi

    print_status "SUCCESS" "All requested services stopped"
}

# Function to restart services
restart_services() {
    local services=("$@")

    print_status "INFO" "Restarting services: ${services[*]}"

    stop_services "${services[@]}"
    sleep 5
    start_services "${services[@]}"
}

# Function to show service status
show_service_status() {
    print_status "INFO" "Checking service status..."

    cd "$PROJECT_ROOT"

    echo ""
    echo "🔍 Service Status Report"
    echo "========================"
    echo ""

    # Check Docker Compose services
    print_status "INFO" "Docker Compose Services:"
    docker-compose -f "$COMPOSE_FILE" ps

    echo ""
    print_status "INFO" "Detailed Service Health:"

    for service_group in "INFRASTRUCTURE_SERVICES" "CORE_SERVICES" "SUPPORT_SERVICES" "MONITORING_SERVICES"; do
        eval "services=(\"\${${service_group}[@]}\")"
        echo ""
        echo "  $(echo $service_group | sed 's/_/ /g' | tr '[:upper:]' '[:lower:]' | sed 's/\b\w/\U&/g'):"

        for service in "${services[@]}"; do
            local status="❌ DOWN"
            local health_status=""

            # Check if container is running
            if docker-compose -f "$COMPOSE_FILE" ps -q "$service" | xargs docker inspect -f '{{.State.Running}}' 2>/dev/null | grep -q true; then
                status="✅ UP"

                # Additional health checks
                case $service in
                    "postgres")
                        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U trading_user -d trading_db &>/dev/null; then
                            health_status=" (healthy)"
                        else
                            health_status=" (not ready)"
                        fi
                        ;;
                    "redis")
                        if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping &>/dev/null; then
                            health_status=" (healthy)"
                        else
                            health_status=" (not ready)"
                        fi
                        ;;
                    *)
                        # Try to check API health endpoint
                        local port=$(docker-compose -f "$COMPOSE_FILE" port "$service" 8000 2>/dev/null | cut -d: -f2 || echo "")
                        if [[ -n "$port" ]]; then
                            if curl -s -f "http://localhost:$port/health" &>/dev/null; then
                                health_status=" (healthy)"
                            else
                                health_status=" (api not responding)"
                            fi
                        fi
                        ;;
                esac
            fi

            printf "    %-20s %s%s\n" "$service" "$status" "$health_status"
        done
    done

    echo ""
    print_status "INFO" "System Resources:"
    echo "  CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')"
    echo "  Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", ($3/$2) * 100.0}')"
    echo "  Disk Usage: $(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}')"
    echo "  Docker Images: $(docker images -q | wc -l)"
    echo "  Docker Volumes: $(docker volume ls -q | wc -l)"
}

# Function to show service logs
show_service_logs() {
    local service=${TARGET_SERVICE:-""}
    local follow_logs=false
    local lines=100

    if [[ -z "$service" ]]; then
        print_status "INFO" "Available services for logs:"
        for svc in "${ALL_SERVICES[@]}"; do
            echo "  - $svc"
        done
        echo ""
        echo -n "Enter service name (or 'all' for all services): "
        read -r service
    fi

    cd "$PROJECT_ROOT"

    if [[ "$service" == "all" ]]; then
        print_status "INFO" "Showing logs from all services (last $lines lines)..."
        docker-compose -f "$COMPOSE_FILE" logs --tail="$lines"
    else
        print_status "INFO" "Showing logs from $service (last $lines lines)..."
        docker-compose -f "$COMPOSE_FILE" logs --tail="$lines" "$service"
    fi
}

# Function to run health checks
run_health_checks() {
    print_status "INFO" "Running comprehensive health checks..."

    cd "$PROJECT_ROOT"

    local failed_checks=0
    local total_checks=0

    echo ""
    echo "🏥 Health Check Report"
    echo "====================="
    echo ""

    # Check each service group
    for service_group in "INFRASTRUCTURE_SERVICES" "CORE_SERVICES" "SUPPORT_SERVICES" "MONITORING_SERVICES"; do
        eval "services=(\"\${${service_group}[@]}\")"
        echo "  $(echo $service_group | sed 's/_/ /g' | tr '[:upper:]' '[:lower:]' | sed 's/\b\w/\U&/g'):"

        for service in "${services[@]}"; do
            ((total_checks++))
            local status="❌ FAILED"

            case $service in
                "postgres")
                    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U trading_user -d trading_db &>/dev/null; then
                        status="✅ HEALTHY"
                    else
                        ((failed_checks++))
                    fi
                    ;;
                "redis")
                    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping &>/dev/null; then
                        status="✅ HEALTHY"
                    else
                        ((failed_checks++))
                    fi
                    ;;
                *)
                    local port=$(docker-compose -f "$COMPOSE_FILE" port "$service" 8000 2>/dev/null | cut -d: -f2 || echo "")
                    if [[ -n "$port" ]] && timeout "$HEALTH_CHECK_TIMEOUT" curl -s -f "http://localhost:$port/health" &>/dev/null; then
                        status="✅ HEALTHY"
                    else
                        ((failed_checks++))
                    fi
                    ;;
            esac

            printf "    %-20s %s\n" "$service" "$status"
        done
        echo ""
    done

    # Overall health summary
    local health_percentage=$(( (total_checks - failed_checks) * 100 / total_checks ))

    echo "📊 Health Summary:"
    echo "   Total Checks: $total_checks"
    echo "   Passed: $((total_checks - failed_checks))"
    echo "   Failed: $failed_checks"
    echo "   Health Score: ${health_percentage}%"

    if [[ $failed_checks -eq 0 ]]; then
        print_status "SUCCESS" "All health checks passed (${health_percentage}%)"
        return 0
    else
        print_status "WARNING" "$failed_checks health check(s) failed (${health_percentage}%)"
        return 1
    fi
}

# Function to scale services
scale_services() {
    local service=${TARGET_SERVICE:-""}
    local replicas=${REPLICAS:-1}

    if [[ -z "$service" ]]; then
        print_status "ERROR" "Service name required for scaling"
        exit 1
    fi

    print_status "INFO" "Scaling $service to $replicas replicas..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would scale $service to $replicas replicas"
        return 0
    fi

    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" up -d --scale "$service=$replicas" "$service"

    print_status "SUCCESS" "Service $service scaled to $replicas replicas"
}

# Function to enter/exit maintenance mode
maintenance_mode() {
    local action=${MAINTENANCE_ACTION:-"status"}
    local message=${MAINTENANCE_MESSAGE:-"Scheduled maintenance"}

    case $action in
        "enter")
            print_status "INFO" "Entering maintenance mode: $message"

            if [[ "$DRY_RUN" == "true" ]]; then
                print_status "INFO" "[DRY RUN] Would enter maintenance mode"
                return 0
            fi

            # Call maintenance service API if available
            if curl -s -X POST "http://localhost:8007/maintenance/enter" \
                -H "Content-Type: application/json" \
                -d "{\"message\": \"$message\", \"initiated_by\": \"$(whoami)\"}" &>/dev/null; then
                print_status "SUCCESS" "Maintenance mode activated"
            else
                print_status "WARNING" "Maintenance service not available, stopping core services manually"
                stop_services "${CORE_SERVICES[@]}"
            fi
            ;;
        "exit")
            print_status "INFO" "Exiting maintenance mode"

            if [[ "$DRY_RUN" == "true" ]]; then
                print_status "INFO" "[DRY RUN] Would exit maintenance mode"
                return 0
            fi

            # Call maintenance service API if available
            if curl -s -X POST "http://localhost:8007/maintenance/exit" &>/dev/null; then
                print_status "SUCCESS" "Maintenance mode deactivated"
            else
                print_status "WARNING" "Maintenance service not available, starting core services manually"
                start_services "${CORE_SERVICES[@]}"
            fi
            ;;
        "status")
            if curl -s "http://localhost:8007/status" | jq -r '.maintenance_mode' 2>/dev/null | grep -q true; then
                print_status "INFO" "System is currently in maintenance mode"
            else
                print_status "INFO" "System is not in maintenance mode"
            fi
            ;;
    esac
}

# Function to perform emergency shutdown
emergency_stop() {
    print_status "CRITICAL" "Performing emergency shutdown..."

    if [[ "$FORCE_ACTION" != "true" ]]; then
        echo -n "⚠️  This will immediately stop ALL services. Continue? (y/N): "
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "INFO" "Emergency stop cancelled"
            return 0
        fi
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would perform emergency shutdown"
        return 0
    fi

    cd "$PROJECT_ROOT"

    # Try graceful shutdown first
    print_status "INFO" "Attempting graceful shutdown..."
    if timeout 30 docker-compose -f "$COMPOSE_FILE" down; then
        print_status "SUCCESS" "Graceful shutdown completed"
    else
        print_status "WARNING" "Graceful shutdown timed out, forcing stop..."
        docker-compose -f "$COMPOSE_FILE" kill
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    fi

    print_status "SUCCESS" "Emergency stop completed"
}

# Function to cleanup resources
cleanup_resources() {
    print_status "INFO" "Cleaning up unused resources..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would cleanup unused Docker resources"
        return 0
    fi

    cd "$PROJECT_ROOT"

    # Cleanup Docker resources
    print_status "INFO" "Removing unused containers..."
    docker container prune -f

    print_status "INFO" "Removing unused images..."
    docker image prune -f

    print_status "INFO" "Removing unused volumes..."
    docker volume prune -f

    print_status "INFO" "Removing unused networks..."
    docker network prune -f

    # Cleanup logs if they exist
    if [[ -f "${PROJECT_ROOT}/scripts/cleanup_logs.sh" ]]; then
        bash "${PROJECT_ROOT}/scripts/cleanup_logs.sh"
    fi

    print_status "SUCCESS" "Resource cleanup completed"
}

# Function to create backup
create_backup() {
    print_status "INFO" "Creating system backup..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would create system backup"
        return 0
    fi

    if [[ -f "${PROJECT_ROOT}/scripts/backup/backup.sh" ]]; then
        bash "${PROJECT_ROOT}/scripts/backup/backup.sh" --type manual --compress
        print_status "SUCCESS" "Backup created successfully"
    else
        print_status "ERROR" "Backup script not found"
        return 1
    fi
}

# Function to restore from backup
restore_from_backup() {
    local backup_id=${BACKUP_ID:-""}

    if [[ -z "$backup_id" ]]; then
        print_status "INFO" "Available backups:"
        ls -la "${PROJECT_ROOT}/data/backups/" | grep ".tar.gz" || {
            print_status "ERROR" "No backups found"
            return 1
        }
        echo ""
        echo -n "Enter backup ID or filename: "
        read -r backup_id
    fi

    print_status "WARNING" "Restoring from backup: $backup_id"

    if [[ "$FORCE_ACTION" != "true" ]]; then
        echo -n "⚠️  This will overwrite current data. Continue? (y/N): "
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "INFO" "Restore cancelled"
            return 0
        fi
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would restore from backup: $backup_id"
        return 0
    fi

    # Stop services before restore
    stop_services "${CORE_SERVICES[@]}"

    # Run restore
    if [[ -f "${PROJECT_ROOT}/scripts/backup/restore.sh" ]]; then
        bash "${PROJECT_ROOT}/scripts/backup/restore.sh" --backup-id "$backup_id"
        print_status "SUCCESS" "Restore completed successfully"

        # Restart services
        start_services "${CORE_SERVICES[@]}"
    else
        print_status "ERROR" "Restore script not found"
        return 1
    fi
}

# Function to update configurations
update_configurations() {
    print_status "INFO" "Updating service configurations..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would update configurations"
        return 0
    fi

    cd "$PROJECT_ROOT"

    # Validate configurations
    if [[ -f "scripts/config/validate_config.py" ]]; then
        python3 scripts/config/validate_config.py --env "$ENVIRONMENT" || {
            print_status "ERROR" "Configuration validation failed"
            return 1
        }
    fi

    # Reload configurations without restarting services
    print_status "INFO" "Reloading configurations..."

    # Send SIGHUP to services that support config reload
    local config_reload_services=("data_collector" "strategy_engine" "risk_manager")

    for service in "${config_reload_services[@]}"; do
        local container_id=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null || echo "")
        if [[ -n "$container_id" ]]; then
            docker kill --signal=HUP "$container_id" 2>/dev/null || true
            print_status "INFO" "Sent config reload signal to $service"
        fi
    done

    print_status "SUCCESS" "Configuration update completed"
}

# Function to check prerequisites
check_prerequisites() {
    local missing_tools=()

    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi

    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_status "ERROR" "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    check_compose_file
}

# Main execution function
main() {
    # Create initial log entry
    {
        echo "=== AI Trading System Operations Session ==="
        echo "Started at: $(date)"
        echo "Command: $COMMAND"
        echo "Environment: $ENVIRONMENT"
        echo "Compose File: $COMPOSE_FILE"
        echo "User: $(whoami)"
        echo ""
    } >> "$OPS_LOG"

    print_status "INFO" "AI Trading System Operations Manager"
    print_status "INFO" "Command: $COMMAND"
    print_status "INFO" "Environment: $ENVIRONMENT"
    print_status "INFO" "Compose File: $COMPOSE_FILE"

    # Check prerequisites
    check_prerequisites

    # Execute command
    case $COMMAND in
        "start")
            if [[ -n "${TARGET_SERVICE:-}" ]]; then
                start_services "$TARGET_SERVICE"
            else
                start_services
            fi
            ;;
        "stop")
            if [[ -n "${TARGET_SERVICE:-}" ]]; then
                stop_services "$TARGET_SERVICE"
            else
                stop_services
            fi
            ;;
        "restart")
            if [[ -n "${TARGET_SERVICE:-}" ]]; then
                restart_services "$TARGET_SERVICE"
            else
                restart_services
            fi
            ;;
        "start-core")
            start_services "${CORE_SERVICES[@]}"
            ;;
        "stop-core")
            stop_services "${CORE_SERVICES[@]}"
            ;;
        "start-monitoring")
            start_services "${MONITORING_SERVICES[@]}"
            ;;
        "stop-monitoring")
            stop_services "${MONITORING_SERVICES[@]}"
            ;;
        "start-infrastructure")
            start_services "${INFRASTRUCTURE_SERVICES[@]}"
            ;;
        "stop-infrastructure")
            stop_services "${INFRASTRUCTURE_SERVICES[@]}"
            ;;
        "status")
            show_service_status
            ;;
        "logs")
            show_service_logs
            ;;
        "health")
            run_health_checks
            ;;
        "scale")
            scale_services
            ;;
        "backup")
            create_backup
            ;;
        "restore")
            restore_from_backup
            ;;
        "maintenance")
            maintenance_mode
            ;;
        "emergency-stop")
            emergency_stop
            ;;
        "cleanup")
            cleanup_resources
            ;;
        "update")
            update_configurations
            ;;
        *)
            print_status "ERROR" "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac

    print_status "SUCCESS" "Operation completed successfully"
    print_status "INFO" "Operations log saved to: $OPS_LOG"
}

# Function to check if running as root
check_permissions() {
    if [[ $EUID -eq 0 ]] && [[ "$ENVIRONMENT" == "production" ]]; then
        print_status "WARNING" "Running as root in production environment"
        print_status "INFO" "Consider using a dedicated service user"
    fi
}

# Function to create system snapshot
+create_system_snapshot() {
+    print_status "INFO" "Creating system snapshot..."
+
+    local snapshot_dir="${PROJECT_ROOT}/data/snapshots/$(date +%Y%m%d_%H%M%S)"
+    mkdir -p "$snapshot_dir"
+
+    # Save Docker state
+    docker-compose -f "$COMPOSE_FILE" config > "${snapshot_dir}/docker-compose-resolved.yml"
+    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" > "${snapshot_dir}/images.txt"
+    docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" > "${snapshot_dir}/containers.txt"
+
+    # Save system state
+    df -h > "${snapshot_dir}/disk_usage.txt"
+    free -h > "${snapshot_dir}/memory_usage.txt"
+    ps aux > "${snapshot_dir}/processes.txt"
+
+    print_status "SUCCESS" "System snapshot saved to: $snapshot_dir"
+}
+
+# Function to show system dashboard
+show_dashboard() {
+    clear
+    echo "🚀 AI Trading System Operations Dashboard"
+    echo "========================================"
+    echo ""
+    echo "📅 $(date)"
+    echo "🖥️  Host: $(hostname)"
+    echo "👤 User: $(whoami)"
+    echo "🌍 Environment: $ENVIRONMENT"
+    echo ""
+
+    # Quick status
+    echo "📊 Quick Status:"
+    local running_services=$(docker-compose -f "$COMPOSE_FILE" ps --services --filter "status=running" | wc -l)
+    local total_services=$(docker-compose -f "$COMPOSE_FILE" ps --services | wc -l)
+    echo "   Services: $running_services/$total_services running"
+    echo "   Uptime: $(uptime -p)"
+    echo "   Load: $(uptime | awk -F'load average:' '{print $2}' | xargs)"
+    echo ""
+
+    # Resource usage
+    echo "💾 Resources:"
+    echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')"
+    echo "   Memory: $(free | grep Mem | awk '{printf "%.1f%%", ($3/$2) * 100.0}')"
+    echo "   Disk: $(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}')"
+    echo ""
+
+    # Recent alerts
+    echo "🚨 Recent Alerts (last 24h):"
+    if [[ -f "$ALERT_LOG" ]]; then
+        local alert_count=$(grep "$(date +%Y-%m-%d)" "$ALERT_LOG" 2>/dev/null | wc -l || echo "0")
+        echo "   Total: $alert_count alerts"
+        if [[ $alert_count -gt 0 ]]; then
+            echo "   Latest:"
+            tail -n 3 "$ALERT_LOG" 2>/dev/null | sed 's/^/     /' || echo "     No recent alerts"
+        fi
+    else
+        echo "   No alert log found"
+    fi
+    echo ""
+}
+
+# Script entry point
+if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
+    # Parse arguments
+    parse_args "$@"
+
+    # Check permissions
+    check_permissions
+
+    # Special handling for dashboard command
+    if [[ "$COMMAND" == "dashboard" ]]; then
+        while true; do
+            show_dashboard
+            echo "Press 'q' to quit, 'r' to refresh, or wait 30s for auto-refresh..."
+            if read -t 30 -n 1 key; then
+                case $key in
+                    'q'|'Q')
+                        echo ""
+                        print_status "INFO" "Dashboard closed"
+                        exit 0
+                        ;;
+                    'r'|'R')
+                        continue
+                        ;;
+                esac
+            fi
+        done
+    else
+        # Run main function for all other commands
+        main
+    fi
+fi
