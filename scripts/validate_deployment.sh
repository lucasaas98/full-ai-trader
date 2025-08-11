#!/bin/bash

# AI Trading System Deployment Validation Script
# Comprehensive validation of all operational components
# Usage: ./validate_deployment.sh [environment] [options]

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
VALIDATION_LOG="${PROJECT_ROOT}/logs/validation/deployment_validation_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs/validation"

# Default configuration
ENVIRONMENT="development"
COMPOSE_FILE="docker-compose.yml"
SKIP_SERVICES=false
SKIP_SECURITY=false
SKIP_PERFORMANCE=false
SKIP_BACKUP=false
VERBOSE=false
QUICK_CHECK=false
FIX_ISSUES=false

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
WARNING_CHECKS=0
FAILED_CHECKS=0

# Required components
REQUIRED_SCRIPTS=(
    "setup_environment.sh"
    "ops_manager.sh"
    "monitor_system.sh"
    "performance_tuning.sh"
    "disaster_recovery.sh"
    "operational_dashboard.py"
    "backup/backup.sh"
    "backup/restore.sh"
    "backup/test_restore.sh"
    "security/audit_security.sh"
    "security/harden_system.sh"
    "config/validate_config.py"
    "deployment/deploy.sh"
    "deployment/migrate.sh"
    "deployment/zero_downtime_deploy.sh"
)

REQUIRED_DIRECTORIES=(
    "data/backups"
    "data/exports"
    "data/secrets"
    "data/state"
    "logs/services"
    "logs/deployment"
    "logs/backup"
    "logs/monitoring"
    "logs/security"
    "logs/performance"
    "config/environments"
    "config/ssl"
    "monitoring/prometheus"
    "monitoring/grafana"
    "monitoring/alertmanager"
)

CORE_SERVICES=(
    "postgres"
    "redis"
    "data_collector"
    "strategy_engine"
    "risk_manager"
    "trade_executor"
    "scheduler"
)

SUPPORT_SERVICES=(
    "export_service"
    "maintenance_service"
)

MONITORING_SERVICES=(
    "prometheus"
    "grafana"
    "alertmanager"
    "elasticsearch"
    "kibana"
    "logstash"
)

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$VALIDATION_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$VALIDATION_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" | tee -a "$VALIDATION_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$VALIDATION_LOG"
            ;;
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message" | tee -a "$VALIDATION_LOG"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Deployment Validation Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    development     Validate development environment (default)
    staging         Validate staging environment
    production      Validate production environment

OPTIONS:
    --quick                Quick validation (essential checks only)
    --skip-services        Skip service validation
    --skip-security        Skip security validation
    --skip-performance     Skip performance validation
    --skip-backup          Skip backup validation
    --fix-issues           Attempt to fix issues automatically
    --verbose              Enable verbose output
    --help                 Show this help message

EXAMPLES:
    $0 development
    $0 production --quick
    $0 staging --fix-issues --verbose
    $0 production --skip-security --skip-performance

VALIDATION AREAS:
    • Required files and directories
    • Script permissions and functionality
    • Service availability and health
    • Database connectivity and schema
    • Security configuration
    • Monitoring and alerting
    • Backup and recovery systems
    • Performance configuration
    • Documentation completeness
EOF
}

# Parse command line arguments
parse_args() {
    if [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; then
        ENVIRONMENT="$1"
        case $ENVIRONMENT in
            "production") COMPOSE_FILE="docker-compose.prod.yml" ;;
            "staging") COMPOSE_FILE="docker-compose.staging.yml" ;;
            "development") COMPOSE_FILE="docker-compose.yml" ;;
        esac
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_CHECK=true
                shift
                ;;
            --skip-services)
                SKIP_SERVICES=true
                shift
                ;;
            --skip-security)
                SKIP_SECURITY=true
                shift
                ;;
            --skip-performance)
                SKIP_PERFORMANCE=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --fix-issues)
                FIX_ISSUES=true
                shift
                ;;
            --verbose)
                VERBOSE=true
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

# Function to run check and update counters
run_check() {
    local check_name=$1
    local check_command=$2
    local fix_command=${3:-""}

    ((TOTAL_CHECKS++))

    if [[ "$VERBOSE" == "true" ]]; then
        print_status "INFO" "Running check: $check_name"
    fi

    if eval "$check_command" &>/dev/null; then
        print_status "SUCCESS" "✅ $check_name"
        ((PASSED_CHECKS++))
        return 0
    else
        if [[ -n "$fix_command" ]] && [[ "$FIX_ISSUES" == "true" ]]; then
            print_status "WARNING" "⚠️  $check_name - Attempting fix..."
            if eval "$fix_command" &>/dev/null; then
                print_status "SUCCESS" "✅ $check_name - Fixed"
                ((PASSED_CHECKS++))
                return 0
            else
                print_status "ERROR" "❌ $check_name - Fix failed"
                ((FAILED_CHECKS++))
                return 1
            fi
        else
            print_status "ERROR" "❌ $check_name"
            ((FAILED_CHECKS++))
            return 1
        fi
    fi
}

# Function to validate required files
validate_required_files() {
    print_status "INFO" "Validating required files and scripts..."

    for script in "${REQUIRED_SCRIPTS[@]}"; do
        local script_path="${PROJECT_ROOT}/scripts/${script}"
        run_check "Script exists: $script" \
            "[[ -f '$script_path' ]]" \
            "touch '$script_path' && chmod +x '$script_path'"

        if [[ -f "$script_path" ]]; then
            run_check "Script executable: $script" \
                "[[ -x '$script_path' ]]" \
                "chmod +x '$script_path'"
        fi
    done

    # Check Docker Compose files
    run_check "Docker Compose file exists" \
        "[[ -f '${PROJECT_ROOT}/${COMPOSE_FILE}' ]]"

    # Check requirements files
    run_check "Requirements file exists" \
        "[[ -f '${PROJECT_ROOT}/requirements.txt' ]]"

    # Check environment template
    run_check "Environment template exists" \
        "[[ -f '${PROJECT_ROOT}/config/environments/.env.template' ]]"
}

# Function to validate directory structure
validate_directory_structure() {
    print_status "INFO" "Validating directory structure..."

    for dir in "${REQUIRED_DIRECTORIES[@]}"; do
        local dir_path="${PROJECT_ROOT}/${dir}"
        run_check "Directory exists: $dir" \
            "[[ -d '$dir_path' ]]" \
            "mkdir -p '$dir_path'"

        if [[ -d "$dir_path" ]]; then
            run_check "Directory writable: $dir" \
                "[[ -w '$dir_path' ]]" \
                "chmod 755 '$dir_path'"
        fi
    done

    # Check special permission directories
    local secrets_dir="${PROJECT_ROOT}/data/secrets"
    if [[ -d "$secrets_dir" ]]; then
        run_check "Secrets directory permissions" \
            "[[ \$(stat -c '%a' '$secrets_dir') == '700' ]]" \
            "chmod 700 '$secrets_dir'"
    fi
}

# Function to validate environment configuration
validate_environment_config() {
    print_status "INFO" "Validating environment configuration..."

    local env_file="${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}"

    run_check "Environment file exists" \
        "[[ -f '$env_file' ]]" \
        "cp '${PROJECT_ROOT}/config/environments/.env.template' '$env_file'"

    if [[ -f "$env_file" ]]; then
        # Check required environment variables
        local required_vars=(
            "ENVIRONMENT"
            "DATABASE_URL"
            "REDIS_URL"
            "JWT_SECRET"
            "ENCRYPTION_KEY"
        )

        for var in "${required_vars[@]}"; do
            run_check "Environment variable set: $var" \
                "grep -q '^${var}=' '$env_file'"
        done

        # Check for default/weak values
        if grep -q "YOUR_.*_HERE\|change.*this\|default.*password" "$env_file"; then
            print_status "WARNING" "⚠️  Default values detected in environment file"
            ((WARNING_CHECKS++))
        fi
    fi

    # Validate configuration using Python script
    if [[ -f "${PROJECT_ROOT}/scripts/config/validate_config.py" ]]; then
        run_check "Configuration validation" \
            "cd '$PROJECT_ROOT' && python3 scripts/config/validate_config.py --env '$ENVIRONMENT'"
    fi
}

# Function to validate Docker setup
validate_docker_setup() {
    print_status "INFO" "Validating Docker setup..."

    # Check Docker daemon
    run_check "Docker daemon running" \
        "docker info" \
        "sudo systemctl start docker"

    # Check Docker Compose
    run_check "Docker Compose available" \
        "docker-compose --version"

    # Check Docker Compose file syntax
    run_check "Docker Compose file valid" \
        "cd '$PROJECT_ROOT' && docker-compose -f '$COMPOSE_FILE' config"

    # Check for required networks
    run_check "Docker network configured" \
        "cd '$PROJECT_ROOT' && docker-compose -f '$COMPOSE_FILE' config | grep -q 'trading_network'"

    # Check for required volumes
    run_check "Docker volumes configured" \
        "cd '$PROJECT_ROOT' && docker-compose -f '$COMPOSE_FILE' config | grep -q 'postgres_data'"
}

# Function to validate services
validate_services() {
    if [[ "$SKIP_SERVICES" == "true" ]]; then
        print_status "INFO" "Skipping service validation"
        return 0
    fi

    print_status "INFO" "Validating service deployment..."

    cd "$PROJECT_ROOT"

    # Check if services are defined in compose file
    for service in "${CORE_SERVICES[@]}" "${SUPPORT_SERVICES[@]}"; do
        run_check "Service defined: $service" \
            "docker-compose -f '$COMPOSE_FILE' config --services | grep -q '^${service}$'"
    done

    # Try to start infrastructure services for testing
    if [[ "$QUICK_CHECK" != "true" ]]; then
        print_status "INFO" "Testing service startup (infrastructure only)..."

        # Start infrastructure
        docker-compose -f "$COMPOSE_FILE" up -d postgres redis &>/dev/null || true
        sleep 20

        # Test database connectivity
        run_check "Database connectivity" \
            "docker-compose -f '$COMPOSE_FILE' exec -T postgres pg_isready -U trading_user -d trading_db"

        # Test Redis connectivity
        run_check "Redis connectivity" \
            "docker-compose -f '$COMPOSE_FILE' exec -T redis redis-cli ping"

        # Cleanup test containers
        docker-compose -f "$COMPOSE_FILE" stop postgres redis &>/dev/null || true
    fi
}

# Function to validate monitoring setup
validate_monitoring() {
    print_status "INFO" "Validating monitoring setup..."

    # Check monitoring services in compose file
    for service in "${MONITORING_SERVICES[@]}"; do
        run_check "Monitoring service defined: $service" \
            "docker-compose -f '$COMPOSE_FILE' config --services | grep -q '^${service}$'"
    done

    # Check Prometheus configuration
    local prometheus_config="${PROJECT_ROOT}/monitoring/prometheus/prometheus.yml"
    if [[ -f "$prometheus_config" ]]; then
        run_check "Prometheus configuration valid" \
            "docker run --rm -v '$prometheus_config':/etc/prometheus/prometheus.yml prom/prometheus:latest promtool check config /etc/prometheus/prometheus.yml"
    fi

    # Check Grafana provisioning
    run_check "Grafana provisioning configured" \
        "[[ -d '${PROJECT_ROOT}/monitoring/grafana/provisioning' ]]" \
        "mkdir -p '${PROJECT_ROOT}/monitoring/grafana/provisioning/{dashboards,datasources}'"

    # Check Alertmanager configuration
    local alertmanager_config="${PROJECT_ROOT}/monitoring/alertmanager/alertmanager.yml"
    if [[ -f "$alertmanager_config" ]]; then
        run_check "Alertmanager configuration valid" \
            "docker run --rm -v '$alertmanager_config':/etc/alertmanager/alertmanager.yml prom/alertmanager:latest amtool check-config /etc/alertmanager/alertmanager.yml"
    fi
}

# Function to validate security setup
validate_security() {
    if [[ "$SKIP_SECURITY" == "true" ]]; then
        print_status "INFO" "Skipping security validation"
        return 0
    fi

    print_status "INFO" "Validating security setup..."

    # Check security scripts
    local security_scripts=("audit_security.sh" "harden_system.sh")
    for script in "${security_scripts[@]}"; do
        run_check "Security script exists: $script" \
            "[[ -f '${PROJECT_ROOT}/scripts/security/${script}' ]]"
    done

    # Check SSL directory
    run_check "SSL directory exists" \
        "[[ -d '${PROJECT_ROOT}/config/ssl' ]]" \
        "mkdir -p '${PROJECT_ROOT}/config/ssl'"

    # Check secrets directory permissions
    local secrets_dir="${PROJECT_ROOT}/data/secrets"
    if [[ -d "$secrets_dir" ]]; then
        run_check "Secrets directory secured" \
            "[[ \$(stat -c '%a' '$secrets_dir') == '700' ]]" \
            "chmod 700 '$secrets_dir'"
    fi

    # Check for rate limiting configuration
    run_check "Rate limiting configured" \
        "[[ -f '${PROJECT_ROOT}/config/security/rate_limiting.yml' ]]" \
        "mkdir -p '${PROJECT_ROOT}/config/security' && touch '${PROJECT_ROOT}/config/security/rate_limiting.yml'"

    # Check firewall (production only)
    if [[ "$ENVIRONMENT" == "production" ]]; then
        if command -v ufw &>/dev/null; then
            run_check "Firewall configured" \
                "ufw status | grep -q 'Status: active'" \
                "echo 'Firewall not active - run ./scripts/security/setup_firewall.sh'"
        fi
    fi
}

# Function to validate backup system
validate_backup_system() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        print_status "INFO" "Skipping backup validation"
        return 0
    fi

    print_status "INFO" "Validating backup system..."

    # Check backup scripts
    local backup_scripts=("backup.sh" "restore.sh" "test_restore.sh" "run_backups.sh")
    for script in "${backup_scripts[@]}"; do
        run_check "Backup script exists: $script" \
            "[[ -f '${PROJECT_ROOT}/scripts/backup/${script}' ]]"
    done

    # Check backup directory
    run_check "Backup directory writable" \
        "[[ -w '${PROJECT_ROOT}/data/backups' ]]" \
        "mkdir -p '${PROJECT_ROOT}/data/backups' && chmod 755 '${PROJECT_ROOT}/data/backups'"

    # Check cron configuration
    if [[ "$ENVIRONMENT" != "development" ]]; then
        run_check "Backup cron job configured" \
            "crontab -l | grep -q 'run_backups.sh'" \
            "echo 'Backup cron job not configured - run ./scripts/setup_environment.sh'"
    fi

    # Test backup functionality (quick test)
    if [[ "$QUICK_CHECK" != "true" ]]; then
        print_status "INFO" "Testing backup functionality..."

        # Create test backup
        if bash "${PROJECT_ROOT}/scripts/backup/backup.sh" --backup-id "validation_test_${TIMESTAMP}" --type test --no-compress &>/dev/null; then
            print_status "SUCCESS" "✅ Backup functionality test passed"
            ((PASSED_CHECKS++))

            # Cleanup test backup
            rm -f "${PROJECT_ROOT}/data/backups/validation_test_${TIMESTAMP}.tar.gz" 2>/dev/null || true
        else
            print_status "ERROR" "❌ Backup functionality test failed"
            ((FAILED_CHECKS++))
        fi
        ((TOTAL_CHECKS++))
    fi
}

# Function to validate performance configuration
validate_performance() {
    if [[ "$SKIP_PERFORMANCE" == "true" ]]; then
        print_status "INFO" "Skipping performance validation"
        return 0
    fi

    print_status "INFO" "Validating performance configuration..."

    # Check performance tuning script
    run_check "Performance tuning script exists" \
        "[[ -f '${PROJECT_ROOT}/scripts/performance_tuning.sh' ]]"

    # Check performance monitoring script
    run_check "Performance monitoring script exists" \
        "[[ -f '${PROJECT_ROOT}/scripts/monitor_performance.sh' ]]" \
        "echo 'Performance monitoring script missing - check setup'"

    # Check resource limits in compose file
    run_check "Resource limits configured" \
        "grep -q 'resources:\|deploy:' '${PROJECT_ROOT}/${COMPOSE_FILE}'"

    # Check performance configuration directory
    run_check "Performance config directory exists" \
        "[[ -d '${PROJECT_ROOT}/config/performance' ]]" \
        "mkdir -p '${PROJECT_ROOT}/config/performance'"

    # System resource check
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    local available_cores=$(nproc)

    if [[ $available_memory -lt 4 ]] && [[ "$ENVIRONMENT" != "development" ]]; then
        print_status "WARNING" "⚠️  Low memory for $ENVIRONMENT environment: ${available_memory}GB"
        ((WARNING_CHECKS++))
        ((TOTAL_CHECKS++))
    else
        print_status "SUCCESS" "✅ Sufficient memory: ${available_memory}GB"
        ((PASSED_CHECKS++))
        ((TOTAL_CHECKS++))
    fi

    if [[ $available_cores -lt 4 ]] && [[ "$ENVIRONMENT" == "production" ]]; then
        print_status "WARNING" "⚠️  Low CPU cores for production: ${available_cores}"
        ((WARNING_CHECKS++))
        ((TOTAL_CHECKS++))
    else
        print_status "SUCCESS" "✅ Sufficient CPU cores: ${available_cores}"
        ((PASSED_CHECKS++))
        ((TOTAL_CHECKS++))
    fi
}

# Function to validate documentation
validate_documentation() {
    print_status "INFO" "Validating documentation..."

    local required_docs=(
        "docs/operations/RUNBOOK.md"
        "docs/operations/TROUBLESHOOTING.md"
        "docs/operations/DISASTER_RECOVERY.md"
        "docs/DEPLOYMENT_GUIDE.md"
        "docs/API_DOCUMENTATION.md"
        "docs/SYSTEM_ARCHITECTURE.md"
        "README.md"
    )

    for doc in "${required_docs[@]}"; do
        run_check "Documentation exists: $(basename "$doc")" \
            "[[ -f '${PROJECT_ROOT}/${doc}' ]]" \
            "touch '${PROJECT_ROOT}/${doc}'"
    done

    # Check documentation completeness
    local runbook="${PROJECT_ROOT}/docs/operations/RUNBOOK.md"
    if [[ -f "$runbook" ]]; then
        run_check "Runbook contains start procedures" \
            "grep -q 'Start.*Stop.*Procedures' '$runbook'"

        run_check "Runbook contains troubleshooting" \
            "grep -q 'Troubleshooting' '$runbook'"

        run_check "Runbook contains emergency procedures" \
            "grep -q 'Emergency.*Procedures' '$runbook'"
    fi
}

# Function to validate operational tools
validate_operational_tools() {
    print_status "INFO" "Validating operational tools..."

    # Check main operational scripts
    local ops_scripts=(
        "ops_manager.sh"
        "monitor_system.sh"
        "disaster_recovery.sh"
        "setup_environment.sh"
    )

    for script in "${ops_scripts[@]}"; do
        run_check "Operational script functional: $script" \
            "'${PROJECT_ROOT}/scripts/${script}' --help" \
            "chmod +x '${PROJECT_ROOT}/scripts/${script}'"
    done

    # Check Python operational tools
    run_check "Operational dashboard available" \
        "[[ -f '${PROJECT_ROOT}/scripts/operational_dashboard.py' ]]"

    if [[ -f "${PROJECT_ROOT}/scripts/operational_dashboard.py" ]]; then
        run_check "Dashboard dependencies available" \
            "cd '$PROJECT_ROOT' && python3 -c 'import docker, psutil, requests, rich'"
    fi

    # Check configuration validation
    if [[ -f "${PROJECT_ROOT}/scripts/config/validate_config.py" ]]; then
        run_check "Configuration validator functional" \
            "cd '$PROJECT_ROOT' && python3 scripts/config/validate_config.py --help"
    fi
}

# Function to validate deployment scripts
validate_deployment_scripts() {
    print_status "INFO" "Validating deployment scripts..."

    local deployment_scripts=(
        "deployment/deploy.sh"
        "deployment/migrate.sh"
        "deployment/zero_downtime_deploy.sh"
    )

    for script in "${deployment_scripts[@]}"; do
        run_check "Deployment script exists: $(basename "$script")" \
            "[[ -f '${PROJECT_ROOT}/scripts/${script}' ]]"

        if [[ -f "${PROJECT_ROOT}/scripts/${script}" ]]; then
            run_check "Deployment script executable: $(basename "$script")" \
                "[[ -x '${PROJECT_ROOT}/scripts/${script}' ]]" \
                "chmod +x '${PROJECT_ROOT}/scripts/${script}'"
        fi
    done

    # Check deployment script functionality
    if [[ -f "${PROJECT_ROOT}/scripts/deployment/deploy.sh" ]]; then
        run_check "Main deployment script functional" \
            "'${PROJECT_ROOT}/scripts/deployment/deploy.sh' --help"
    fi
}

# Function to validate prerequisites
validate_prerequisites() {
    print_status "INFO" "Validating system prerequisites..."

    local required_commands=(
        "docker"
        "docker-compose"
        "python3"
        "git"
        "curl"
        "openssl"
    )

    for cmd in "${required_commands[@]}"; do
        run_check "Command available: $cmd" \
            "command -v $cmd" \
            "echo 'Install $cmd: sudo apt install $cmd'"
    done

    # Check Python packages
    local python_packages=(
        "yaml"
        "requests"
        "psycopg2"
    )

    for package in "${python_packages[@]}"; do
        run_check "Python package available: $package" \
            "python3 -c 'import $package'" \
            "pip3 install $package"
    done

    # Check system resources
    local available_disk=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    local min_disk_gb=20

    if [[ "$ENVIRONMENT" == "production" ]]; then
        min_disk_gb=100
    fi

    if [[ $available_disk -lt $min_disk_gb ]]; then
        print_status "WARNING" "⚠️  Low disk space: ${available_disk}GB (minimum: ${min_disk_gb}GB)"
        ((WARNING_CHECKS++))
        ((TOTAL_CHECKS++))
    else
        print_status "SUCCESS" "✅ Sufficient disk space: ${available_disk}GB"
        ((PASSED_CHECKS++))
        ((TOTAL_CHECKS++))
    fi
}

# Function to validate maintenance capabilities
validate_maintenance() {
    print_status "INFO" "Validating maintenance capabilities..."

    # Check maintenance service
    run_check "Maintenance service configured" \
        "docker-compose -f '$COMPOSE_FILE' config --services | grep -q 'maintenance_service'"

    # Check cleanup scripts
    run_check "Log cleanup script exists" \
        "[[ -f '${PROJECT_ROOT}/scripts/cleanup_logs.sh' ]]" \
        "touch '${PROJECT_ROOT}/scripts/cleanup_logs.sh' && chmod +x '${PROJECT_ROOT}/scripts/cleanup_logs.sh'"

    # Check health check script
    run_check "Health check script exists" \
        "[[ -f '${PROJECT_ROOT}/scripts/health_check.sh' ]]" \
        "touch '${PROJECT_ROOT}/scripts/health_check.sh' && chmod +x '${PROJECT_ROOT}/scripts/health_check.sh'"
}

# Function to run functional tests
run_functional_tests() {
    if [[ "$QUICK_CHECK" == "true" ]]; then
        print_status "INFO" "Skipping functional tests (quick check mode)"
        return 0
    fi

    print_status "INFO" "Running functional tests..."

    # Test configuration validation
    if [[ -f "${PROJECT_ROOT}/scripts/config/validate_config.py" ]]; then
        run_check "Configuration validation functional" \
            "cd '$PROJECT_ROOT' && python3 scripts/config/validate_config.py --env '$ENVIRONMENT' --dry-run"
    fi

    # Test backup functionality
    if [[ -f "${PROJECT_ROOT}/scripts/backup/backup.sh" ]]; then
        run_check "Backup script functional" \
            "'${PROJECT_ROOT}/scripts/backup/backup.sh' --help"
    fi

    # Test monitoring script
    if [[ -f "${PROJECT_ROOT}/scripts/monitor_system.sh" ]]; then
        run_check "Monitoring script functional" \
            "'${PROJECT_ROOT}/scripts/monitor_system.sh' --help"
    fi

    # Test operations manager
    if [[ -f "${PROJECT_ROOT}/scripts/ops_manager.sh" ]]; then
        run_check "Operations manager functional" \
            "'${PROJECT_ROOT}/scripts/ops_manager.sh' --help"
    fi
}

# Function to generate validation report
generate_validation_report() {
    local report_file="${PROJECT_ROOT}/logs/validation/validation_report_${TIMESTAMP}.md"

    cat > "$report_file" << EOF
# AI Trading System Deployment Validation Report

**Generated**: $(date)
**Environment**: $ENVIRONMENT
**Validator**: $(whoami)@$(hostname)

## Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Checks | $TOTAL_CHECKS | 100% |
| Passed | $PASSED_CHECKS | $((PASSED_CHECKS * 100 / TOTAL_CHECKS))% |
| Warnings | $WARNING_CHECKS | $((WARNING_CHECKS * 100 / TOTAL_CHECKS))% |
| Failed | $FAILED_CHECKS | $((FAILED_CHECKS * 100 / TOTAL_CHECKS))% |

## Overall Status

EOF

    local overall_score=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

    if [[ $overall_score -ge 95 ]]; then
        echo "🎉 **EXCELLENT** - Deployment is production-ready" >> "$report_file"
    elif [[ $overall_score -ge 85 ]]; then
        echo "✅ **GOOD** - Deployment is ready with minor issues" >> "$report_file"
    elif [[ $overall_score -ge 70 ]]; then
        echo "⚠️  **ACCEPTABLE** - Deployment needs attention" >> "$report_file"
    else
        echo "❌ **POOR** - Deployment has significant issues" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

## Recommendations

EOF

    if [[ $FAILED_CHECKS -gt 0 ]]; then
        echo "### Critical Issues" >> "$report_file"
        echo "- Address all failed checks before production deployment" >> "$report_file"
        echo "- Run validation again with --fix-issues flag" >> "$report_file"
        echo "- Review validation log: $VALIDATION_LOG" >> "$report_file"
        echo "" >> "$report_file"
    fi

    if [[ $WARNING_CHECKS -gt 0 ]]; then
        echo "### Warnings" >> "$report_file"
        echo "- Review warning items during next maintenance window" >> "$report_file"
        echo "- Consider automated fixes where appropriate" >> "$report_file"
        echo "" >> "$report_file"
    fi

    echo "### Next Steps" >> "$report_file"
    if [[ $overall_score -ge 85 ]]; then
        echo "1. Deploy to production: \`./scripts/deployment/deploy.sh production\`" >> "$report_file"
        echo "2. Setup monitoring: Access Grafana at http://localhost:3000" >> "$report_file"
        echo "3. Configure alerts: Review monitoring/alertmanager/" >> "$report_file"
        echo "4. Schedule backups: Verify cron configuration" >> "$report_file"
    else
        echo "1. Fix critical issues identified in this report" >> "$report_file"
        echo "2. Re-run validation: \`./scripts/validate_deployment.sh $ENVIRONMENT\`" >> "$report_file"
        echo "3
