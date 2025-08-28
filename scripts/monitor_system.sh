#!/bin/bash

# AI Trading System Operational Monitoring Script
# Real-time monitoring with alerts and automated responses
# Usage: ./monitor_system.sh [options]

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
MONITOR_LOG="${PROJECT_ROOT}/logs/monitoring/monitor_${TIMESTAMP}.log"
ALERT_LOG="${PROJECT_ROOT}/logs/monitoring/alerts.log"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs/monitoring"

# Default configuration
MONITOR_INTERVAL=30
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
ALERT_THRESHOLD_DISK=90
ALERT_THRESHOLD_NETWORK=1000  # MB/s
HEALTH_CHECK_TIMEOUT=10
MAX_CONSECUTIVE_FAILURES=3
ENABLE_AUTO_RECOVERY=true
ENABLE_SLACK_ALERTS=false
ENABLE_EMAIL_ALERTS=false
CONTINUOUS_MODE=false
DAEMON_MODE=false

# Service definitions
CORE_SERVICES=(
    "postgres:5432:database"
    "redis:6379:cache"
    "data_collector:9101:api"
    "strategy_engine:9102:api"
    "risk_manager:9103:api"
    "trade_executor:9104:api"
    "scheduler:9105:api"
)

MONITORING_SERVICES=(
    "prometheus:9090:http"
    "grafana:3000:http"
    "elasticsearch:9200:http"
    "kibana:5601:http"
)

OPTIONAL_SERVICES=(
    "export_service:9106:api"
    "maintenance_service:9107:api"
    "alertmanager:9093:http"
)

# Failure counters
declare -A failure_counts
declare -A last_alert_time
declare -A service_status

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
    echo "[$level] $timestamp - $message" >> "$MONITOR_LOG"
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Monitoring Script

Usage: $0 [OPTIONS]

OPTIONS:
    --interval SEC         Monitoring interval in seconds (default: 30)
    --cpu-threshold PCT    CPU usage alert threshold (default: 80)
    --memory-threshold PCT Memory usage alert threshold (default: 85)
    --disk-threshold PCT   Disk usage alert threshold (default: 90)
    --network-threshold MB Network usage alert threshold MB/s (default: 1000)
    --timeout SEC          Health check timeout (default: 10)
    --max-failures NUM     Max consecutive failures before alert (default: 3)
    --no-auto-recovery     Disable automatic service recovery
    --enable-slack         Enable Slack notifications
    --enable-email         Enable email notifications
    --continuous           Run continuously (monitoring daemon)
    --daemon               Run as background daemon
    --help                 Show this help message

EXAMPLES:
    $0 --interval 60 --continuous
    $0 --cpu-threshold 90 --enable-slack
    $0 --daemon --enable-email --enable-slack

SIGNALS:
    SIGUSR1               Force health check
    SIGUSR2               Rotate logs
    SIGTERM/SIGINT        Graceful shutdown
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --interval)
                MONITOR_INTERVAL="$2"
                shift 2
                ;;
            --cpu-threshold)
                ALERT_THRESHOLD_CPU="$2"
                shift 2
                ;;
            --memory-threshold)
                ALERT_THRESHOLD_MEMORY="$2"
                shift 2
                ;;
            --disk-threshold)
                ALERT_THRESHOLD_DISK="$2"
                shift 2
                ;;
            --network-threshold)
                ALERT_THRESHOLD_NETWORK="$2"
                shift 2
                ;;
            --timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            --max-failures)
                MAX_CONSECUTIVE_FAILURES="$2"
                shift 2
                ;;
            --no-auto-recovery)
                ENABLE_AUTO_RECOVERY=false
                shift
                ;;
            --enable-slack)
                ENABLE_SLACK_ALERTS=true
                shift
                ;;
            --enable-email)
                ENABLE_EMAIL_ALERTS=true
                shift
                ;;
            --continuous)
                CONTINUOUS_MODE=true
                shift
                ;;
            --daemon)
                DAEMON_MODE=true
                CONTINUOUS_MODE=true
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

# Function to check service health
check_service_health() {
    local service_info=$1
    local service_name=$(echo "$service_info" | cut -d: -f1)
    local port=$(echo "$service_info" | cut -d: -f2)
    local type=$(echo "$service_info" | cut -d: -f3)

    case $type in
        "database")
            if docker-compose exec -T "$service_name" pg_isready -U trading_user -d trading_db &>/dev/null; then
                return 0
            fi
            ;;
        "cache")
            if docker-compose exec -T "$service_name" redis-cli ping &>/dev/null; then
                return 0
            fi
            ;;
        "api"|"http")
            if timeout "$HEALTH_CHECK_TIMEOUT" curl -s -f "http://localhost:$port/health" &>/dev/null; then
                return 0
            fi
            ;;
    esac

    return 1
}

# Function to get system metrics
get_system_metrics() {
    local cpu_usage disk_usage memory_usage network_rx network_tx

    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')

    # Memory usage
    memory_usage=$(free | grep Mem | awk '{printf "%.1f", ($3/$2) * 100.0}')

    # Disk usage
    disk_usage=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')

    # Network usage (simplified)
    local network_stats=$(cat /proc/net/dev | grep eth0 || cat /proc/net/dev | grep enp || echo "0 0 0 0 0 0 0 0 0 0")
    network_rx=$(echo "$network_stats" | awk '{print $2}')
    network_tx=$(echo "$network_stats" | awk '{print $10}')

    echo "CPU:${cpu_usage:-0}|MEM:${memory_usage:-0}|DISK:${disk_usage:-0}|NET_RX:${network_rx:-0}|NET_TX:${network_tx:-0}"
}

# Function to check Docker resources
check_docker_resources() {
    local containers_running containers_total images_count volumes_count

    containers_running=$(docker ps --format "table {{.Names}}" | tail -n +2 | wc -l)
    containers_total=$(docker ps -a --format "table {{.Names}}" | tail -n +2 | wc -l)
    images_count=$(docker images -q | wc -l)
    volumes_count=$(docker volume ls -q | wc -l)

    echo "CONTAINERS_RUNNING:$containers_running|CONTAINERS_TOTAL:$containers_total|IMAGES:$images_count|VOLUMES:$volumes_count"
}

# Function to send alert
send_alert() {
    local level=$1
    local service=$2
    local message=$3
    local alert_key="${service}_${level}"
    local current_time=$(date +%s)

    # Rate limiting: don't send same alert more than once per hour
    if [[ -n "${last_alert_time[$alert_key]:-}" ]]; then
        local time_diff=$((current_time - ${last_alert_time[$alert_key]}))
        if [[ $time_diff -lt 3600 ]]; then
            return 0
        fi
    fi

    last_alert_time[$alert_key]=$current_time

    # Log alert
    local alert_msg="[$level] $service: $message"
    print_status "$level" "$alert_msg"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $alert_msg" >> "$ALERT_LOG"

    # Send notifications if enabled
    if [[ "$ENABLE_SLACK_ALERTS" == "true" ]] && [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        send_slack_alert "$level" "$service" "$message"
    fi

    if [[ "$ENABLE_EMAIL_ALERTS" == "true" ]] && [[ -n "${ALERT_EMAIL:-}" ]]; then
        send_email_alert "$level" "$service" "$message"
    fi
}

# Function to send Slack alert
send_slack_alert() {
    local level=$1
    local service=$2
    local message=$3

    local emoji color
    case $level in
        "WARNING") emoji="⚠️"; color="warning" ;;
        "ERROR") emoji="❌"; color="danger" ;;
        "CRITICAL") emoji="🚨"; color="danger" ;;
        *) emoji="ℹ️"; color="good" ;;
    esac

    local payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "fields": [
                {
                    "title": "$emoji AI Trading System Alert",
                    "value": "**Service:** $service\\n**Level:** $level\\n**Message:** $message\\n**Time:** $(date)",
                    "short": false
                }
            ]
        }
    ]
}
EOF
)

    curl -s -X POST -H 'Content-type: application/json' \
        --data "$payload" \
        "${SLACK_WEBHOOK_URL}" &>/dev/null || true
}

# Function to send email alert
send_email_alert() {
    local level=$1
    local service=$2
    local message=$3

    local subject="[$level] AI Trading System Alert - $service"
    local body="Service: $service
Level: $level
Message: $message
Time: $(date)
Host: $(hostname)
Environment: ${ENVIRONMENT}

Please check the system status and take appropriate action.

Monitoring Log: $MONITOR_LOG
Alert Log: $ALERT_LOG"

    if command -v mail &>/dev/null; then
        echo "$body" | mail -s "$subject" "${ALERT_EMAIL}" || true
    elif command -v sendmail &>/dev/null; then
        {
            echo "To: ${ALERT_EMAIL}"
            echo "Subject: $subject"
            echo ""
            echo "$body"
        } | sendmail "${ALERT_EMAIL}" || true
    fi
}

# Function to attempt service recovery
attempt_service_recovery() {
    local service=$1

    if [[ "$ENABLE_AUTO_RECOVERY" != "true" ]]; then
        return 1
    fi

    print_status "INFO" "Attempting to recover service: $service"

    cd "$PROJECT_ROOT"

    # Try to restart the service
    if docker-compose restart "$service" &>/dev/null; then
        sleep 10
        if check_service_health "$service:0:api"; then
            print_status "SUCCESS" "Service $service recovered successfully"
            failure_counts[$service]=0
            return 0
        fi
    fi

    # If restart failed, try full recreation
    print_status "WARNING" "Restart failed, attempting to recreate service: $service"
    if docker-compose up -d --force-recreate "$service" &>/dev/null; then
        sleep 15
        if check_service_health "$service:0:api"; then
            print_status "SUCCESS" "Service $service recreated successfully"
            failure_counts[$service]=0
            return 0
        fi
    fi

    print_status "ERROR" "Failed to recover service: $service"
    return 1
}

# Function to monitor services
monitor_services() {
    local services=("${CORE_SERVICES[@]}" "${MONITORING_SERVICES[@]}" "${OPTIONAL_SERVICES[@]}")
    local failed_services=()

    for service_info in "${services[@]}"; do
        local service_name=$(echo "$service_info" | cut -d: -f1)

        if check_service_health "$service_info"; then
            if [[ "${service_status[$service_name]:-}" == "DOWN" ]]; then
                print_status "SUCCESS" "Service $service_name is back online"
                send_alert "INFO" "$service_name" "Service recovered and is now healthy"
            fi
            service_status[$service_name]="UP"
            failure_counts[$service_name]=0
        else
            service_status[$service_name]="DOWN"
            failure_counts[$service_name]=$((${failure_counts[$service_name]:-0} + 1))
            failed_services+=("$service_name")

            local failure_count=${failure_counts[$service_name]}

            if [[ $failure_count -eq 1 ]]; then
                send_alert "WARNING" "$service_name" "Service health check failed (attempt 1/$MAX_CONSECUTIVE_FAILURES)"
            elif [[ $failure_count -eq $MAX_CONSECUTIVE_FAILURES ]]; then
                send_alert "ERROR" "$service_name" "Service is down (failed $failure_count consecutive checks)"

                # Attempt recovery for core services
                if [[ " ${CORE_SERVICES[*]} " =~ " ${service_info} " ]]; then
                    attempt_service_recovery "$service_name"
                fi
            elif [[ $failure_count -gt $MAX_CONSECUTIVE_FAILURES ]]; then
                if [[ $((failure_count % 10)) -eq 0 ]]; then  # Alert every 10 failures after threshold
                    send_alert "CRITICAL" "$service_name" "Service has been down for $failure_count consecutive checks"
                fi
            fi
        fi
    done

    return ${#failed_services[@]}
}

# Function to monitor system resources
monitor_system_resources() {
    local metrics=$(get_system_metrics)
    local docker_metrics=$(check_docker_resources)

    # Parse metrics
    local cpu=$(echo "$metrics" | grep -o 'CPU:[^|]*' | cut -d: -f2)
    local memory=$(echo "$metrics" | grep -o 'MEM:[^|]*' | cut -d: -f2)
    local disk=$(echo "$metrics" | grep -o 'DISK:[^|]*' | cut -d: -f2)

    # Check thresholds and send alerts
    if [[ $(echo "$cpu > $ALERT_THRESHOLD_CPU" | bc -l) -eq 1 ]]; then
        send_alert "WARNING" "SYSTEM" "High CPU usage: ${cpu}%"
    fi

    if [[ $(echo "$memory > $ALERT_THRESHOLD_MEMORY" | bc -l) -eq 1 ]]; then
        send_alert "WARNING" "SYSTEM" "High memory usage: ${memory}%"
    fi

    if [[ $(echo "$disk > $ALERT_THRESHOLD_DISK" | bc -l) -eq 1 ]]; then
        send_alert "ERROR" "SYSTEM" "High disk usage: ${disk}%"
    fi

    # Log metrics
    echo "$(date '+%Y-%m-%d %H:%M:%S') - METRICS: $metrics | DOCKER: $docker_metrics" >> "${PROJECT_ROOT}/logs/monitoring/metrics.log"
}

# Function to monitor trading performance
monitor_trading_performance() {
    # Check if trading is active
    local active_positions=0
    local recent_trades=0
    local system_pnl=0

    # Query database for metrics (if available)
    if docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c "SELECT 1;" &>/dev/null; then
        # Get active positions
        active_positions=$(docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c \
            "SELECT COUNT(*) FROM positions WHERE status = 'OPEN';" 2>/dev/null | tr -d ' ' || echo "0")

        # Get recent trades (last hour)
        recent_trades=$(docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c \
            "SELECT COUNT(*) FROM trades WHERE created_at > NOW() - INTERVAL '1 hour';" 2>/dev/null | tr -d ' ' || echo "0")

        # Get system P&L for today
        system_pnl=$(docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c \
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE DATE(created_at) = CURRENT_DATE;" 2>/dev/null | tr -d ' ' || echo "0")
    fi

    # Log trading metrics
    echo "$(date '+%Y-%m-%d %H:%M:%S') - TRADING: Positions:$active_positions|Trades:$recent_trades|PnL:$system_pnl" >> "${PROJECT_ROOT}/logs/monitoring/trading_metrics.log"

    # Check for concerning patterns
    if [[ $recent_trades -eq 0 ]] && [[ $active_positions -gt 0 ]]; then
        send_alert "WARNING" "TRADING" "No trades executed in last hour despite open positions"
    fi

    if [[ $(echo "$system_pnl < -1000" | bc -l) -eq 1 ]]; then
        send_alert "ERROR" "TRADING" "Significant daily loss detected: \$${system_pnl}"
    fi
}

# Function to check disk space
check_disk_space() {
    local critical_paths=(
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/logs"
        "/var/lib/docker"
    )

    for path in "${critical_paths[@]}"; do
        if [[ -d "$path" ]]; then
            local usage=$(df -h "$path" | awk 'NR==2 {print $5}' | sed 's/%//')
            if [[ $usage -gt $ALERT_THRESHOLD_DISK ]]; then
                send_alert "ERROR" "DISK" "High disk usage in $path: ${usage}%"

                # Auto-cleanup if enabled
                if [[ "$ENABLE_AUTO_RECOVERY" == "true" ]]; then
                    print_status "INFO" "Attempting automatic cleanup of $path"
                    if [[ "$path" == *"/logs" ]]; then
                        bash "${PROJECT_ROOT}/scripts/cleanup_logs.sh" || true
                    elif [[ "$path" == "/var/lib/docker" ]]; then
                        docker system prune -f --volumes || true
                    fi
                fi
            fi
        fi
    done
}

# Function to check for security issues
check_security_status() {
    # Check for failed login attempts
    local failed_logins=$(grep "Failed" /var/log/auth.log 2>/dev/null | grep "$(date +%b\ %d)" | wc -l || echo "0")

    if [[ $failed_logins -gt 10 ]]; then
        send_alert "WARNING" "SECURITY" "High number of failed login attempts: $failed_logins"
    fi

    # Check container security
    local privileged_containers=$(docker ps --format "table {{.Names}}" --filter "status=running" | tail -n +2 | xargs -I {} docker inspect {} --format '{{.Name}}: {{.HostConfig.Privileged}}' | grep "true" | wc -l || echo "0")

    if [[ $privileged_containers -gt 1 ]]; then  # Allow one for cadvisor
        send_alert "WARNING" "SECURITY" "Multiple containers running in privileged mode: $privileged_containers"
    fi
}

# Function to generate status report
generate_status_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local report_file="${PROJECT_ROOT}/logs/monitoring/status_report_$(date +%Y%m%d_%H%M%S).json"

    # Collect all metrics
    local system_metrics=$(get_system_metrics)
    local docker_metrics=$(check_docker_resources)

    # Create JSON report
    cat > "$report_file" << EOF
{
    "timestamp": "$timestamp",
    "environment": "${ENVIRONMENT:-unknown}",
    "system_metrics": {
        "cpu_usage": "$(echo "$system_metrics" | grep -o 'CPU:[^|]*' | cut -d: -f2)",
        "memory_usage": "$(echo "$system_metrics" | grep -o 'MEM:[^|]*' | cut -d: -f2)",
        "disk_usage": "$(echo "$system_metrics" | grep -o 'DISK:[^|]*' | cut -d: -f2)"
    },
    "docker_metrics": {
        "containers_running": "$(echo "$docker_metrics" | grep -o 'CONTAINERS_RUNNING:[^|]*' | cut -d: -f2)",
        "containers_total": "$(echo "$docker_metrics" | grep -o 'CONTAINERS_TOTAL:[^|]*' | cut -d: -f2)",
        "images": "$(echo "$docker_metrics" | grep -o 'IMAGES:[^|]*' | cut -d: -f2)",
        "volumes": "$(echo "$docker_metrics" | grep -o 'VOLUMES:[^|]*' | cut -d: -f2)"
    },
    "service_status": {
EOF

    # Add service status
    local first=true
    for service_info in "${CORE_SERVICES[@]}" "${MONITORING_SERVICES[@]}"; do
        local service_name=$(echo "$service_info" | cut -d: -f1)
        local status="${service_status[$service_name]:-UNKNOWN}"

        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo "," >> "$report_file"
        fi

        echo "        \"$service_name\": \"$status\"" >> "$report_file"
    done

    cat >> "$report_file" << EOF
    },
    "alerts_last_24h": $(grep "$(date +%Y-%m-%d)" "$ALERT_LOG" 2>/dev/null | wc -l || echo "0"),
    "uptime": "$(uptime -p)",
    "load_average": "$(uptime | awk -F'load average:' '{print $2}' | xargs)"
}
EOF

    print_status "INFO" "Status report generated: $report_file"
}

# Function to setup signal handlers
setup_signal_handlers() {
    # Force health check on SIGUSR1
    trap 'print_status "INFO" "Forcing health check..."; monitor_services; monitor_system_resources' SIGUSR1

    # Rotate logs on SIGUSR2
    trap 'print_status "INFO" "Rotating logs..."; MONITOR_LOG="${PROJECT_ROOT}/logs/monitoring/monitor_$(date +%Y%m%d_%H%M%S).log"' SIGUSR2

    # Graceful shutdown on SIGTERM/SIGINT
    trap 'print_status "INFO" "Received shutdown signal, exiting..."; exit 0' SIGTERM SIGINT
}

# Function to run continuous monitoring
run_continuous_monitoring() {
    print_status "INFO" "Starting continuous monitoring (interval: ${MONITOR_INTERVAL}s)"

    while true; do
        local start_time=$(date +%s)

        # Monitor all components
        monitor_services
        monitor_system_resources
        monitor_trading_performance
        check_disk_space
        check_security_status

        # Generate periodic status report (every hour)
        local current_minute=$(date +%M)
        if [[ "$current_minute" == "00" ]]; then
            generate_status_report
        fi

        # Calculate sleep time to maintain interval
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        local sleep_time=$((MONITOR_INTERVAL - elapsed))

        if [[ $sleep_time -gt 0 ]]; then
            sleep $sleep_time
        fi
    done
}

# Function to run as daemon
run_as_daemon() {
    print_status "INFO" "Starting monitoring daemon..."

    # Create PID file
    local pid_file="${PROJECT_ROOT}/data/monitor.pid"
    echo $$ > "$pid_file"

    # Redirect output to log file
    exec 1>> "$MONITOR_LOG"
    exec 2>> "$MONITOR_LOG"

    # Run continuous monitoring
    run_continuous_monitoring
}

# Function to run single check
run_single_check() {
    print_status "INFO" "Running single monitoring check..."

    local failed_services=0

    # Check services
    failed_services=$(monitor_services)

    # Check system resources
    monitor_system_resources

    # Check trading performance
    monitor_trading_performance

    # Check disk space
    check_disk_space

    # Check security
    check_security_status

    # Generate status report
    generate_status_report

    if [[ $failed_services -eq 0 ]]; then
        print_status "SUCCESS" "All services are healthy"
        return 0
    else
        print_status "WARNING" "$failed_services service(s) are unhealthy"
        return 1
    fi
}

# Main execution function
main() {
    # Load environment variables if available
    if [[ -f "${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT:-development}" ]]; then
        source "${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT:-development}"
    fi

    # Setup signal handlers
    setup_signal_handlers

    # Create initial log entry
    {
        echo "=== AI Trading System Monitoring Session ==="
        echo "Started at: $(date)"
        echo "Mode: $(if [[ "$CONTINUOUS_MODE" == "true" ]]; then echo "Continuous"; else echo "Single Check"; fi)"
        echo "Interval: ${MONITOR_INTERVAL}s"
        echo "Auto Recovery: $ENABLE_AUTO_RECOVERY"
        echo "Alerts: Slack=$ENABLE_SLACK_ALERTS, Email=$ENABLE_EMAIL_ALERTS"
        echo ""
    } >> "$MONITOR_LOG"

    print_status "INFO" "AI Trading System Monitoring Started"
    print_status "INFO" "Monitor Log: $MONITOR_LOG"
    print_status "INFO" "Alert Log: $ALERT_LOG"

    # Run monitoring based on mode
    if [[ "$DAEMON_MODE" == "true" ]]; then
        run_as_daemon
    elif [[ "$CONTINUOUS_MODE" == "true" ]]; then
        run_continuous_monitoring
    else
        run_single_check
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    main
fi
