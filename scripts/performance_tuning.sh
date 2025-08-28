#!/bin/bash

# AI Trading System Performance Tuning Script
# Optimizes system performance for different workloads and environments
# Usage: ./performance_tuning.sh [profile] [options]

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
TUNING_LOG="${PROJECT_ROOT}/logs/performance/tuning_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs/performance"

# Default configuration
PERFORMANCE_PROFILE="balanced"
ENVIRONMENT="development"
APPLY_CHANGES=false
DRY_RUN=false
VERBOSE=false
BACKUP_CONFIGS=true

# Performance profiles
declare -A CPU_PROFILES=(
    ["conservative"]="50"
    ["balanced"]="75"
    ["aggressive"]="90"
    ["maximum"]="95"
)

declare -A MEMORY_PROFILES=(
    ["conservative"]="1g"
    ["balanced"]="2g"
    ["aggressive"]="4g"
    ["maximum"]="6g"
)

declare -A WORKER_PROFILES=(
    ["conservative"]="2"
    ["balanced"]="4"
    ["aggressive"]="8"
    ["maximum"]="12"
)

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
    esac

    # Log to file
    echo "[$level] $timestamp - $message" >> "$TUNING_LOG"
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Performance Tuning Script

Usage: $0 [PROFILE] [OPTIONS]

PERFORMANCE PROFILES:
    conservative            Low resource usage, stable performance
    balanced                Balanced resource usage and performance (default)
    aggressive              High performance, higher resource usage
    maximum                 Maximum performance, all available resources

OPTIONS:
    --env ENV               Environment (development/staging/production)
    --apply                 Apply changes (without this, only shows recommendations)
    --dry-run               Show what would be changed without applying
    --verbose               Enable verbose output
    --no-backup             Skip configuration backup
    --cpu-limit PCT         Override CPU limit percentage
    --memory-limit SIZE     Override memory limit (e.g., 2g, 1024m)
    --workers NUM           Override number of worker processes
    --help                  Show this help message

EXAMPLES:
    $0 balanced --env production --apply
    $0 aggressive --cpu-limit 85 --memory-limit 3g --apply
    $0 conservative --dry-run --verbose
    $0 maximum --env production --apply --no-backup

TUNING AREAS:
    - Docker container resource limits
    - Database connection pools
    - Redis configuration
    - Application worker processes
    - System kernel parameters
    - Network buffer sizes
    - File system optimizations
EOF
}

# Parse command line arguments
parse_args() {
    if [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; then
        PERFORMANCE_PROFILE="$1"
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --apply)
                APPLY_CHANGES=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --no-backup)
                BACKUP_CONFIGS=false
                shift
                ;;
            --cpu-limit)
                CUSTOM_CPU_LIMIT="$2"
                shift 2
                ;;
            --memory-limit)
                CUSTOM_MEMORY_LIMIT="$2"
                shift 2
                ;;
            --workers)
                CUSTOM_WORKERS="$2"
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

# Function to analyze current system
analyze_system() {
    print_status "INFO" "Analyzing current system performance..."

    # Get system specs
    local cpu_cores=$(nproc)
    local total_memory=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    local disk_space=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')

    echo ""
    echo "🖥️  System Specifications:"
    echo "   CPU Cores: $cpu_cores"
    echo "   Total Memory: ${total_memory}GB"
    echo "   Available Disk: $disk_space"
    echo ""

    # Analyze current Docker resource usage
    echo "🐳 Current Docker Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -10

    echo ""

    # Check current performance metrics
    echo "📈 Current Performance Metrics:"
    echo "   Load Average: $(uptime | awk -F'load average:' '{print $2}' | xargs)"
    echo "   CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')"
    echo "   Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", ($3/$2) * 100.0}')"
    echo "   Disk I/O: $(iostat -x 1 1 2>/dev/null | tail -n +4 | awk 'NR==1{print $10"%"}' || echo "N/A")"

    # Database performance
    if docker-compose ps postgres | grep -q "Up"; then
        echo ""
        echo "🗄️  Database Performance:"
        local active_connections=$(docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d ' ' || echo "N/A")
        local cache_hit_ratio=$(docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c "SELECT round(sum(blks_hit)*100.0/sum(blks_hit+blks_read), 2) FROM pg_stat_database;" 2>/dev/null | tr -d ' ' || echo "N/A")
        echo "   Active Connections: $active_connections"
        echo "   Cache Hit Ratio: ${cache_hit_ratio}%"
    fi

    echo ""
}

# Function to backup configurations
backup_configurations() {
    if [[ "$BACKUP_CONFIGS" != "true" ]]; then
        return 0
    fi

    print_status "INFO" "Backing up current configurations..."

    local backup_dir="${PROJECT_ROOT}/data/config_backups/${TIMESTAMP}"
    mkdir -p "$backup_dir"

    # Backup Docker Compose files
    cp "${PROJECT_ROOT}/docker-compose"*.yml "$backup_dir/" 2>/dev/null || true

    # Backup environment files
    cp -r "${PROJECT_ROOT}/config/environments" "$backup_dir/" 2>/dev/null || true

    # Backup monitoring configurations
    cp -r "${PROJECT_ROOT}/monitoring" "$backup_dir/" 2>/dev/null || true

    print_status "SUCCESS" "Configurations backed up to: $backup_dir"
}

# Function to optimize Docker containers
optimize_docker_containers() {
    local cpu_limit=${CUSTOM_CPU_LIMIT:-${CPU_PROFILES[$PERFORMANCE_PROFILE]}}
    local memory_limit=${CUSTOM_MEMORY_LIMIT:-${MEMORY_PROFILES[$PERFORMANCE_PROFILE]}}
    local workers=${CUSTOM_WORKERS:-${WORKER_PROFILES[$PERFORMANCE_PROFILE]}}

    print_status "INFO" "Optimizing Docker container resources..."
    print_status "INFO" "Profile: $PERFORMANCE_PROFILE (CPU: ${cpu_limit}%, Memory: $memory_limit, Workers: $workers)"

    # Create optimized docker-compose override
    local override_file="${PROJECT_ROOT}/docker-compose.override.yml"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would create optimized docker-compose.override.yml"
        return 0
    fi

    cat > "$override_file" << EOF
version: '3.8'

services:
  data_collector:
    deploy:
      resources:
        limits:
          cpus: '0.${cpu_limit}'
          memory: $memory_limit
        reservations:
          cpus: '0.25'
          memory: 256m
    environment:
      - WORKER_PROCESSES=$workers
      - WORKER_CONNECTIONS=1000
      - MAX_REQUESTS=10000
      - MAX_REQUESTS_JITTER=1000

  strategy_engine:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: $(echo "$memory_limit" | sed 's/g/*2g/' | bc)
        reservations:
          cpus: '0.5'
          memory: 512m
    environment:
      - WORKER_PROCESSES=$((workers * 2))
      - STRATEGY_WORKERS=$workers
      - BACKTEST_WORKERS=$((workers / 2))
      - CACHE_SIZE=1000000

  risk_manager:
    deploy:
      resources:
        limits:
          cpus: '0.75'
          memory: $memory_limit
        reservations:
          cpus: '0.25'
          memory: 256m
    environment:
      - RISK_WORKERS=$workers
      - POSITION_CACHE_SIZE=100000
      - RISK_CALCULATION_THREADS=$workers

  trade_executor:
    deploy:
      resources:
        limits:
          cpus: '0.${cpu_limit}'
          memory: $memory_limit
        reservations:
          cpus: '0.25'
          memory: 256m
    environment:
      - EXECUTOR_WORKERS=$workers
      - ORDER_QUEUE_SIZE=10000
      - EXECUTION_TIMEOUT=30

  postgres:
    environment:
      - POSTGRES_SHARED_BUFFERS=$(($(echo "$memory_limit" | sed 's/g//' | bc) * 256))MB
      - POSTGRES_EFFECTIVE_CACHE_SIZE=$(($(echo "$memory_limit" | sed 's/g//' | bc) * 512))MB
      - POSTGRES_WORK_MEM=$(($(echo "$memory_limit" | sed 's/g//' | bc) * 4))MB
      - POSTGRES_MAINTENANCE_WORK_MEM=$(($(echo "$memory_limit" | sed 's/g//' | bc) * 64))MB
      - POSTGRES_MAX_CONNECTIONS=$((workers * 10))
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: $(echo "$memory_limit" | sed 's/g/*3g/' | bc)
        reservations:
          cpus: '0.5'
          memory: 512m

  redis:
    environment:
      - REDIS_MAXMEMORY=$(echo "$memory_limit" | sed 's/g/gb/')
      - REDIS_MAXMEMORY_POLICY=allkeys-lru
      - REDIS_SAVE=""  # Disable persistence for performance
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: $memory_limit
        reservations:
          cpus: '0.25'
          memory: 128m
EOF

    print_status "SUCCESS" "Docker container optimization applied"
}

# Function to optimize PostgreSQL
optimize_postgresql() {
    print_status "INFO" "Optimizing PostgreSQL configuration..."

    local cpu_limit=${CUSTOM_CPU_LIMIT:-${CPU_PROFILES[$PERFORMANCE_PROFILE]}}
    local memory_limit=${CUSTOM_MEMORY_LIMIT:-${MEMORY_PROFILES[$PERFORMANCE_PROFILE]}}
    local memory_mb=$(echo "$memory_limit" | sed 's/g//' | awk '{print $1 * 1024}')

    # Calculate PostgreSQL settings based on available resources
    local shared_buffers=$((memory_mb / 4))
    local effective_cache_size=$((memory_mb * 3 / 4))
    local work_mem=$((memory_mb / 64))
    local maintenance_work_mem=$((memory_mb / 16))

    # Create PostgreSQL configuration
    local pg_config="${PROJECT_ROOT}/config/postgresql/performance.conf"
    mkdir -p "${PROJECT_ROOT}/config/postgresql"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would optimize PostgreSQL with shared_buffers=${shared_buffers}MB"
        return 0
    fi

    cat > "$pg_config" << EOF
# PostgreSQL Performance Configuration - Generated $(date)
# Profile: $PERFORMANCE_PROFILE

# Memory Settings
shared_buffers = ${shared_buffers}MB
effective_cache_size = ${effective_cache_size}MB
work_mem = ${work_mem}MB
maintenance_work_mem = ${maintenance_work_mem}MB

# Connection Settings
max_connections = 200
superuser_reserved_connections = 3

# Checkpoint Settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Query Planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

# Performance Monitoring
track_activities = on
track_counts = on
track_io_timing = on
track_functions = all

# Autovacuum Tuning
autovacuum_max_workers = 3
autovacuum_naptime = 20s
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05

# WAL Settings
wal_level = replica
max_wal_size = 4GB
min_wal_size = 80MB
EOF

    print_status "SUCCESS" "PostgreSQL configuration optimized: $pg_config"
}

# Function to optimize Redis
optimize_redis() {
    print_status "INFO" "Optimizing Redis configuration..."

    local memory_limit=${CUSTOM_MEMORY_LIMIT:-${MEMORY_PROFILES[$PERFORMANCE_PROFILE]}}

    # Create Redis configuration
    local redis_config="${PROJECT_ROOT}/config/redis/performance.conf"
    mkdir -p "${PROJECT_ROOT}/config/redis"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would optimize Redis with maxmemory=$memory_limit"
        return 0
    fi

    cat > "$redis_config" << EOF
# Redis Performance Configuration - Generated $(date)
# Profile: $PERFORMANCE_PROFILE

# Memory Management
maxmemory ${memory_limit}b
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Persistence (optimized for performance)
EOF

    case $PERFORMANCE_PROFILE in
        "conservative")
            cat >> "$redis_config" << EOF
save 900 1
save 300 10
save 60 10000
EOF
            ;;
        "balanced")
            cat >> "$redis_config" << EOF
save 300 10
save 60 10000
EOF
            ;;
        "aggressive"|"maximum")
            cat >> "$redis_config" << EOF
save ""
appendonly yes
appendfsync everysec
EOF
            ;;
    esac

    cat >> "$redis_config" << EOF

# Network
tcp-keepalive 300
timeout 0
tcp-backlog 511

# Performance
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Threading
io-threads 4
io-threads-do-reads yes

# Latency Monitoring
latency-monitor-threshold 100
EOF

    print_status "SUCCESS" "Redis configuration optimized: $redis_config"
}

# Function to optimize application settings
optimize_application_settings() {
    print_status "INFO" "Optimizing application settings..."

    local workers=${CUSTOM_WORKERS:-${WORKER_PROFILES[$PERFORMANCE_PROFILE]}}

    # Create application performance configuration
    local app_config="${PROJECT_ROOT}/config/performance/application.yml"
    mkdir -p "${PROJECT_ROOT}/config/performance"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would optimize application with $workers workers"
        return 0
    fi

    cat > "$app_config" << EOF
# Application Performance Configuration - Generated $(date)
# Profile: $PERFORMANCE_PROFILE

# Worker Configuration
worker_processes: $workers
worker_connections: 1000
worker_rlimit_nofile: 65535

# Threading
max_threads: $((workers * 4))
thread_pool_size: $workers
async_workers: $workers

# Queue Settings
queue_size: 10000
batch_size: 100
prefetch_count: $((workers * 2))

# Caching
cache_size: 1000000
cache_ttl: 3600
memory_cache_size: "512MB"

# Database Connection Pool
db_pool_size: $((workers * 5))
db_pool_max_overflow: $((workers * 2))
db_pool_timeout: 30
db_pool_recycle: 3600

# Redis Connection Pool
redis_pool_size: $((workers * 3))
redis_pool_max_connections: $((workers * 5))

# API Rate Limiting
rate_limit_requests: 1000
rate_limit_window: 60
rate_limit_burst: 100

# Performance Monitoring
enable_metrics: true
metrics_interval: 30
slow_query_threshold: 1000
memory_profiling: $(if [[ "$ENVIRONMENT" == "development" ]]; then echo "true"; else echo "false"; fi)

# Trading Specific
strategy_execution_threads: $workers
market_data_buffer_size: 100000
order_processing_threads: $((workers / 2))
position_monitoring_interval: 1000

# Backtesting Performance
backtest_parallel_workers: $workers
backtest_chunk_size: 10000
backtest_memory_limit: "2GB"
EOF

    print_status "SUCCESS" "Application configuration optimized: $app_config"
}

# Function to optimize system kernel parameters
optimize_kernel_parameters() {
    if [[ "$ENVIRONMENT" != "production" ]]; then
        print_status "INFO" "Skipping kernel optimization for $ENVIRONMENT environment"
        return 0
    fi

    print_status "INFO" "Optimizing kernel parameters..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would optimize kernel parameters"
        return 0
    fi

    # Create sysctl configuration
    local sysctl_config="${PROJECT_ROOT}/config/system/99-trading-performance.conf"
    mkdir -p "${PROJECT_ROOT}/config/system"

    cat > "$sysctl_config" << EOF
# AI Trading System Kernel Performance Tuning - Generated $(date)
# Profile: $PERFORMANCE_PROFILE

# Network Performance
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 65536
net.core.wmem_default = 65536
net.core.netdev_max_backlog = 5000
net.core.somaxconn = 65535

# TCP Performance
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_mtu_probing = 1
net.ipv4.tcp_slow_start_after_idle = 0

# File System Performance
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288

# Memory Management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.dirty_expire_centisecs = 12000

# Process Limits
kernel.pid_max = 4194304
EOF

    if [[ -w /etc/sysctl.d/ ]]; then
        sudo cp "$sysctl_config" /etc/sysctl.d/99-trading-performance.conf
        sudo sysctl -p /etc/sysctl.d/99-trading-performance.conf
        print_status "SUCCESS" "Kernel parameters optimized and applied"
    else
        print_status "WARNING" "Cannot apply kernel parameters (no sudo access)"
        print_status "INFO" "Configuration saved to: $sysctl_config"
        print_status "INFO" "Manually copy to /etc/sysctl.d/ and run 'sysctl -p'"
    fi
}

# Function to optimize monitoring stack
optimize_monitoring() {
    print_status "INFO" "Optimizing monitoring stack..."

    local memory_limit=${CUSTOM_MEMORY_LIMIT:-${MEMORY_PROFILES[$PERFORMANCE_PROFILE]}}

    # Optimize Prometheus configuration
    local prometheus_config="${PROJECT_ROOT}/monitoring/prometheus/prometheus-optimized.yml"
    mkdir -p "${PROJECT_ROOT}/monitoring/prometheus"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would optimize monitoring configurations"
        return 0
    fi

    cat > "$prometheus_config" << EOF
# Prometheus Optimized Configuration - Generated $(date)
# Profile: $PERFORMANCE_PROFILE

global:
  scrape_interval: $(if [[ "$PERFORMANCE_PROFILE" == "conservative" ]]; then echo "30s"; else echo "15s"; fi)
  evaluation_interval: $(if [[ "$PERFORMANCE_PROFILE" == "conservative" ]]; then echo "30s"; else echo "15s"; fi)
  external_labels:
    environment: '$ENVIRONMENT'
    instance: '$(hostname)'

rule_files:
  - "alerts/*.yml"

scrape_configs:
  - job_name: 'trading-services'
    static_configs:
      - targets:
        - 'data_collector:9101'
        - 'strategy_engine:9102'
        - 'risk_manager:9103'
        - 'trade_executor:9104'
        - 'scheduler:9105'
    scrape_interval: 10s
    metrics_path: /metrics

  - job_name: 'infrastructure'
    static_configs:
      - targets:
        - 'postgres_exporter:9187'
        - 'redis_exporter:9121'
        - 'node_exporter:9100'
    scrape_interval: 15s

  - job_name: 'monitoring'
    static_configs:
      - targets:
        - 'cadvisor:8080'
    scrape_interval: 30s

storage:
  tsdb:
    retention.time: $(if [[ "$PERFORMANCE_PROFILE" == "conservative" ]]; then echo "15d"; else echo "30d"; fi)
    retention.size: $(if [[ "$PERFORMANCE_PROFILE" == "conservative" ]]; then echo "10GB"; else echo "50GB"; fi)
EOF

    # Optimize Grafana settings
    local grafana_config="${PROJECT_ROOT}/monitoring/grafana/grafana-optimized.ini"

    cat > "$grafana_config" << EOF
# Grafana Optimized Configuration - Generated $(date)
# Profile: $PERFORMANCE_PROFILE

[server]
protocol = http
http_port = 3000
enforce_domain = false
enable_gzip = true

[database]
type = sqlite3
path = grafana.db
cache_mode = private
wal = true

[session]
provider = memory
cookie_secure = false
session_life_time = 86400

[analytics]
reporting_enabled = false
check_for_updates = false

[security]
admin_user = admin
admin_password = admin
secret_key = SW2YcwTIb9zpOOhoPsMm
disable_gravatar = true

[snapshots]
external_enabled = false

[users]
allow_sign_up = false
auto_assign_org_role = Viewer

[auth.anonymous]
enabled = false

[log]
mode = console
level = warn
EOF

    print_status "SUCCESS" "Monitoring stack optimized"
}

# Function to create performance monitoring
create_performance_monitoring() {
    print_status "INFO" "Setting up performance monitoring..."

    # Create performance monitoring script
    cat > "${PROJECT_ROOT}/scripts/monitor_performance.sh" << 'EOF'
#!/bin/bash

# Performance Monitoring Script
# Continuously monitors and reports system performance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

monitor_performance() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local perf_log="${PROJECT_ROOT}/logs/performance/performance_$(date +%Y%m%d).log"

    # System metrics
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", ($3/$2) * 100.0}')
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')

    # Trading specific metrics
    local active_positions=0
    local trades_per_minute=0
    local avg_execution_time=0

    if docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c "SELECT 1;" &>/dev/null; then
        active_positions=$(docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c \
            "SELECT COUNT(*) FROM positions WHERE status = 'OPEN';" 2>/dev/null | tr -d ' ' || echo "0")

        trades_per_minute=$(docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c \
            "SELECT COUNT(*) FROM trades WHERE created_at > NOW() - INTERVAL '1 minute';" 2>/dev/null | tr -d ' ' || echo "0")

        avg_execution_time=$(docker-compose exec -T postgres psql -U trading_user -d trading_db -t -c \
            "SELECT COALESCE(AVG(EXTRACT(EPOCH FROM (executed_at - created_at)) * 1000), 0) FROM trades WHERE created_at > NOW() - INTERVAL '5 minutes';" 2>/dev/null | tr -d ' ' | cut -d. -f1 || echo "0")
    fi

    # Log performance data
    echo "$timestamp,CPU:$cpu_usage,MEM:$memory_usage,LOAD:$load_avg,POSITIONS:$active_positions,TPM:$trades_per_minute,EXEC_TIME:$avg_execution_time" >> "$perf_log"

    # Check for performance issues
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        echo "$timestamp - HIGH CPU USAGE: $cpu_usage%" >> "${PROJECT_ROOT}/logs/performance/alerts.log"
    fi

    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        echo "$timestamp - HIGH MEMORY USAGE: $memory_usage%" >> "${PROJECT_ROOT}/logs/performance/alerts.log"
    fi

    if (( avg_execution_time > 5000 )); then
        echo "$timestamp - SLOW TRADE EXECUTION: ${avg_execution_time}ms" >> "${PROJECT_ROOT}/logs/performance/alerts.log"
    fi
}

# Run monitoring
if [[ "${1:-}" == "--once" ]]; then
    monitor_performance
else
    while true; do
        monitor_performance
        sleep 60
    done
fi
EOF

    chmod +x "${PROJECT_ROOT}/scripts/monitor_performance.sh"
    print_status "SUCCESS" "Performance monitoring script created"
}

# Function to show performance recommendations
show_recommendations() {
    print_status "INFO" "Performance Recommendations for $PERFORMANCE_PROFILE profile:"

    echo ""
    echo "🚀 Performance Profile: $PERFORMANCE_PROFILE"
    echo "=================================="
    echo ""

    case $PERFORMANCE_PROFILE in
        "conservative")
            echo "📋 Conservative Profile Recommendations:"
            echo "   • Lower resource usage for cost optimization"
            echo "   • Suitable for development and light trading"
            echo "   • CPU Limit: ${CPU_PROFILES[$PERFORMANCE_PROFILE]}%"
            echo "   • Memory Limit: ${MEMORY_PROFILES[$PERFORMANCE_PROFILE]}"
            echo "   • Workers: ${WORKER_PROFILES[$PERFORMANCE_PROFILE]}"
            echo ""
            echo "⚠️  Trade-offs:"
            echo "   • Lower throughput"
            echo "   • Increased latency"
            echo "   • Limited concurrent operations"
            ;;
        "balanced")
            echo "📋 Balanced Profile Recommendations:"
            echo "   • Good balance of performance and resource usage"
            echo "   • Suitable for staging and moderate production loads"
            echo "   • CPU Limit: ${CPU_PROFILES[$PERFORMANCE_PROFILE]}%"
            echo "   • Memory Limit: ${MEMORY_PROFILES[$PERFORMANCE_PROFILE]}"
            echo "   • Workers: ${WORKER_PROFILES[$PERFORMANCE_PROFILE]}"
            echo ""
            echo "✅ Benefits:"
            echo "   • Reasonable performance"
            echo "   • Efficient resource utilization"
            echo "   • Good stability"
            ;;
        "aggressive")
            echo "📋 Aggressive Profile Recommendations:"
            echo "   • High performance with increased resource usage"
            echo "   • Suitable for high-frequency trading"
            echo "   • CPU Limit: ${CPU_PROFILES[$PERFORMANCE_PROFILE]}%"
            echo "   • Memory Limit: ${MEMORY_PROFILES[$PERFORMANCE_PROFILE]}"
            echo "   • Workers: ${WORKER_PROFILES[$PERFORMANCE_PROFILE]}"
            echo ""
            echo "⚡ Benefits:"
            echo "   • High throughput"
            echo "   • Low latency"
            echo "   • Better concurrent handling"
            echo ""
            echo "⚠️  Considerations:"
            echo "   • Higher resource costs"
            echo "   • Increased heat generation"
            echo "   • Potential stability issues under load"
            ;;
        "maximum")
            echo "📋 Maximum Profile Recommendations:"
            echo "   • Absolute maximum performance"
            echo "   • Suitable for critical high-frequency trading"
            echo "   • CPU Limit: ${CPU_PROFILES[$PERFORMANCE_PROFILE]}%"
            echo "   • Memory Limit: ${MEMORY_PROFILES[$PERFORMANCE_PROFILE]}"
            echo "   • Workers: ${WORKER_PROFILES[$PERFORMANCE_PROFILE]}"
            echo ""
            echo "🔥 Benefits:"
            echo "   • Maximum throughput"
            echo "   • Minimum latency"
            echo "   • Optimal resource utilization"
            echo ""
            echo "⚠️  Risks:"
            echo "   • High resource consumption"
            echo "   • Potential system instability"
            echo "   • Requires careful monitoring"
            ;;
    esac

    echo ""
    echo "💡 Additional Recommendations:"
    echo "   • Monitor system metrics after applying changes"
    echo "   • Gradually increase load to test stability"
    echo "   • Keep backups of working configurations"
    echo "   • Use '--dry-run' to preview changes first"
    echo ""
}

# Function to apply optimizations
apply_optimizations() {
    print_status "INFO" "Applying performance optimizations..."

    if [[ "$BACKUP_CONFIGS" == "true" ]]; then
        backup_configurations
    fi

    # Apply optimizations
    optimize_docker_containers
    optimize_postgresql
    optimize_redis
    optimize_application_settings
    optimize_kernel_parameters
    optimize_monitoring
    create_performance_monitoring

    if [[ "$APPLY_CHANGES" == "true" ]]; then
        print_status "INFO" "Restarting services to apply changes..."
        cd "$PROJECT_ROOT"

        # Restart services to apply new configurations
        docker-compose down
        sleep 5
        docker-compose up -d

        print_status "SUCCESS" "All optimizations applied and services restarted"
    else
        print_status "INFO" "Optimizations prepared but not applied"
        print_status "INFO" "Use --apply to apply changes"
    fi
}

# Function to benchmark system
+benchmark_system() {
+    print_status "INFO" "Running system benchmark..."
+
+    local benchmark_dir="${PROJECT_ROOT}/logs/performance/benchmark_${TIMESTAMP}"
+    mkdir -p "$benchmark_dir"
+
+    # CPU benchmark
+    print_status "INFO" "Running CPU benchmark..."
+    {
+        echo "=== CPU Benchmark ==="
+        time python3 -c "
+import time
+start = time.time()
+for i in range(1000000):
+    _ = i ** 2
+print(f'CPU benchmark completed in {time.time() - start:.2f}s')
+"
+    } > "${benchmark_dir}/cpu_benchmark.txt" 2>&1
+
+    # Memory benchmark
+    print_status "INFO" "Running memory benchmark..."
+    {
+        echo "=== Memory Benchmark ==="
+        python3 -c "
+import time
+import gc
+
+start = time.time()
+data = []
+for i in range(100000):
+    data.append(list(range(100)))
+
+gc.collect()
+print(f'Memory benchmark completed in {time.time() - start:.2f}s')
+print(f'Memory allocated: {len(data) * 100 * 8 / 1024 / 1024:.2f}MB')
+"
+    } > "${benchmark_dir}/memory_benchmark.txt" 2>&1
+
+    # Database benchmark
+    if docker-compose ps postgres | grep -q "Up"; then
+        print_status "INFO" "Running database benchmark..."
+        {
+            echo "=== Database Benchmark ==="
+            docker-compose exec -T postgres psql -U trading_user -d trading_db -c "
+            CREATE TABLE IF NOT EXISTS benchmark_test (
+                id SERIAL PRIMARY KEY,
+                data TEXT,
+                created_at TIMESTAMP DEFAULT NOW()
+            );
+
+            -- Insert benchmark
+            \timing on
+            INSERT INTO benchmark_test (data)
+            SELECT 'benchmark_data_' || generate_series(1, 10000);
+
+            -- Select benchmark
+            SELECT COUNT(*) FROM benchmark_test;
+
+            -- Cleanup
+            DROP TABLE benchmark_test;
+            "
+        } > "${benchmark_dir}/database_benchmark.txt" 2>&1
+    fi
+
+    # Trading system benchmark
+    print_status "INFO" "Running trading system benchmark..."
+    {
+        echo "=== Trading System Benchmark ==="
+        # Test API response times
+        for service in data_collector strategy_engine risk_manager trade_executor; do
+            local port=$(docker-compose port "$service" 8000 2>/dev/null | cut -d: -f2 || echo "")
+            if [[ -n "$port" ]]; then
+                echo "Testing $service (port $port):"
+                for i in {1..10}; do
+                    curl -w "@-" -s -o /dev/null "http://localhost:$port/health" << 'CURL_FORMAT'
+     time_namelookup:  %{time_namelookup}s\n
+        time_connect:  %{time_connect}s\n
+     time_appconnect:  %{time_appconnect}s\n
+    time_pretransfer:  %{time_pretransfer}s\n
+       time_redirect:  %{time_redirect}s\n
+  time_starttransfer:  %{time_starttransfer}s\n
+                     ----------\n
+          time_total:  %{time_total}s\n
+CURL_FORMAT
+                done
+                echo ""
+            fi
+        done
+    } > "${benchmark_dir}/trading_benchmark.txt" 2>&1
+
+    print_status "SUCCESS" "Benchmark completed, results in: $benchmark_dir"
+}
+
+# Function to show current settings
+show_current_settings() {
+    print_status "INFO" "Current Performance Settings:"
+
+    echo ""
+    echo "🔧 Docker Resource Limits:"
+    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -10
+
+    echo ""
+    echo "🗄️  Database Settings:"
+    if docker-compose ps postgres | grep -q "Up"; then
+        docker-compose exec -T postgres psql -U trading_user -d trading_db -c "
+        SELECT name, setting, unit FROM pg_settings
+        WHERE name IN ('shared_buffers', 'effective_cache_size', 'work_mem', 'max_connections')
+        ORDER BY name;
+        " 2>/dev/null || echo "Cannot retrieve database settings"
+    else
+        echo "   Database not running"
+    fi
+
+    echo ""
+    echo "📊 Redis Settings:"
+    if docker-compose ps redis | grep -q "Up"; then
+        echo "   Max Memory: $(docker-compose exec -T redis redis-cli config get maxmemory 2>/dev/null | tail -n1 || echo "N/A")"
+        echo "   Memory Policy: $(docker-compose exec -T redis redis-cli config get maxmemory-policy 2>/dev/null | tail -n1 || echo "N/A")"
+    else
+        echo "   Redis not running"
+    fi
+}
+
+# Main execution function
+main() {
+    # Create initial log entry
+    {
+        echo "=== AI Trading System Performance Tuning Session ==="
+        echo "Started at: $(date)"
+        echo "Profile: $PERFORMANCE_PROFILE"
+        echo "Environment: $ENVIRONMENT"
+        echo "Apply Changes: $APPLY_CHANGES"
+        echo "User: $(whoami)"
+        echo ""
+    } >> "$TUNING_LOG"
+
+    print_status "INFO" "AI Trading System Performance Tuning"
+    print_status "INFO" "Profile: $PERFORMANCE_PROFILE"
+    print_status "INFO" "Environment: $ENVIRONMENT"
+    print_status "INFO" "Tuning Log: $TUNING_LOG"
+
+    # Analyze current system
+    analyze_system
+
+    # Show current settings
+    show_current_settings
+
+    # Show recommendations
+    show_recommendations
+
+    # Apply optimizations if requested
+    if [[ "$APPLY_CHANGES" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
+        apply_optimizations
+    else
+        echo ""
+        print_status "INFO" "To apply these optimizations, run with --apply flag"
+        print_status "INFO" "To see what would change, run with --dry-run flag"
+    fi
+
+    # Offer to run benchmark
+    if [[ "$APPLY_CHANGES" == "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
+        echo ""
+        echo -n "Run performance benchmark? (y/N): "
+        read -n 1 -r
+        echo
+        if [[ $REPLY =~ ^[Yy]$ ]]; then
+            benchmark_system
+        fi
+    fi
+
+    print_status "SUCCESS" "Performance tuning session completed"
+}
+
+# Script entry point
+if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
+    parse_args "$@"
+    main
+fi
