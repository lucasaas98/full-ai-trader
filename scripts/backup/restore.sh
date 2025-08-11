#!/bin/bash

# AI Trading System Backup Restoration Script
# Comprehensive restoration with validation and rollback capabilities
# Usage: ./restore.sh [options]

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
RESTORE_LOG="${PROJECT_ROOT}/logs/backup/restore_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs/backup"

# Default configuration
BACKUP_ID=""
BACKUP_TYPE="full"
DRY_RUN=false
FORCE_RESTORE=false
SKIP_VALIDATION=false
CREATE_SNAPSHOT=true
RESTORE_DATABASE=true
RESTORE_CONFIGS=true
RESTORE_DATA=true
VERIFY_INTEGRITY=true
AUTO_START_SERVICES=true

# Service management
COMPOSE_FILE="docker-compose.yml"
ENVIRONMENT="development"

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$RESTORE_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$RESTORE_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" | tee -a "$RESTORE_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$RESTORE_LOG"
            ;;
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message" | tee -a "$RESTORE_LOG"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Backup Restoration Script

Usage: $0 [OPTIONS]

OPTIONS:
    --backup-id ID          Backup ID or filename to restore
    --type TYPE             Backup type (full, database, configs, data)
    --env ENV               Environment (development/staging/production)
    --compose-file FILE     Docker compose file to use
    --dry-run               Show what would be restored without executing
    --force                 Force restore without confirmation
    --skip-validation       Skip backup validation
    --no-snapshot           Skip creating pre-restore snapshot
    --database-only         Restore only database
    --configs-only          Restore only configurations
    --data-only             Restore only data files
    --no-verify             Skip integrity verification
    --no-start              Don't auto-start services after restore
    --help                  Show this help message

EXAMPLES:
    $0 --backup-id backup_20240101_120000
    $0 --backup-id latest --env production --force
    $0 --type database --backup-id backup_20240101_120000
    $0 --dry-run --backup-id backup_20240101_120000
    $0 --configs-only --backup-id backup_20240101_120000

BACKUP TYPES:
    full                    Complete system restore (default)
    database                Database only
    configs                 Configuration files only
    data                    Data files and exports only

RESTORE PROCESS:
    1. Validate backup integrity
    2. Create pre-restore snapshot
    3. Stop affected services
    4. Restore data
    5. Verify restoration
    6. Start services
    7. Run post-restore tests
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup-id)
                BACKUP_ID="$2"
                shift 2
                ;;
            --type)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            --env)
                ENVIRONMENT="$2"
                case $ENVIRONMENT in
                    "production") COMPOSE_FILE="docker-compose.prod.yml" ;;
                    "staging") COMPOSE_FILE="docker-compose.staging.yml" ;;
                    "development") COMPOSE_FILE="docker-compose.yml" ;;
                esac
                shift 2
                ;;
            --compose-file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE_RESTORE=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --no-snapshot)
                CREATE_SNAPSHOT=false
                shift
                ;;
            --database-only)
                BACKUP_TYPE="database"
                RESTORE_CONFIGS=false
                RESTORE_DATA=false
                shift
                ;;
            --configs-only)
                BACKUP_TYPE="configs"
                RESTORE_DATABASE=false
                RESTORE_DATA=false
                shift
                ;;
            --data-only)
                BACKUP_TYPE="data"
                RESTORE_DATABASE=false
                RESTORE_CONFIGS=false
                shift
                ;;
            --no-verify)
                VERIFY_INTEGRITY=false
                shift
                ;;
            --no-start)
                AUTO_START_SERVICES=false
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

# Function to find and validate backup
find_and_validate_backup() {
    print_status "INFO" "Finding and validating backup..."

    local backup_dir="${PROJECT_ROOT}/data/backups"

    if [[ -z "$BACKUP_ID" ]]; then
        print_status "INFO" "Available backups:"
        if [[ -d "$backup_dir" ]]; then
            ls -la "$backup_dir"/*.tar.gz 2>/dev/null | head -10 || {
                print_status "ERROR" "No backups found in $backup_dir"
                exit 1
            }
        else
            print_status "ERROR" "Backup directory not found: $backup_dir"
            exit 1
        fi

        echo ""
        echo -n "Enter backup ID or filename: "
        read -r BACKUP_ID
    fi

    # Handle special cases
    if [[ "$BACKUP_ID" == "latest" ]]; then
        BACKUP_ID=$(ls -t "$backup_dir"/*.tar.gz 2>/dev/null | head -1 | xargs basename | sed 's/\.tar\.gz$//' || "")
        if [[ -z "$BACKUP_ID" ]]; then
            print_status "ERROR" "No backups found"
            exit 1
        fi
        print_status "INFO" "Using latest backup: $BACKUP_ID"
    fi

    # Find backup file
    local backup_file=""
    if [[ -f "${backup_dir}/${BACKUP_ID}.tar.gz" ]]; then
        backup_file="${backup_dir}/${BACKUP_ID}.tar.gz"
    elif [[ -f "${backup_dir}/${BACKUP_ID}" ]]; then
        backup_file="${backup_dir}/${BACKUP_ID}"
    elif [[ -f "$BACKUP_ID" ]]; then
        backup_file="$BACKUP_ID"
    else
        print_status "ERROR" "Backup file not found: $BACKUP_ID"
        exit 1
    fi

    BACKUP_FILE="$backup_file"
    print_status "SUCCESS" "Found backup file: $BACKUP_FILE"

    # Validate backup integrity
    if [[ "$SKIP_VALIDATION" != "true" ]]; then
        print_status "INFO" "Validating backup integrity..."

        # Check if file is readable
        if [[ ! -r "$BACKUP_FILE" ]]; then
            print_status "ERROR" "Cannot read backup file: $BACKUP_FILE"
            exit 1
        fi

        # Check file size
        local file_size=$(stat -c%s "$BACKUP_FILE")
        if [[ $file_size -lt 1024 ]]; then
            print_status "ERROR" "Backup file too small (${file_size} bytes), possibly corrupted"
            exit 1
        fi

        # Test archive integrity
        if ! tar -tzf "$BACKUP_FILE" >/dev/null 2>&1; then
            print_status "ERROR" "Backup archive is corrupted or invalid"
            exit 1
        fi

        # Check for required components
        local required_components=()
        case $BACKUP_TYPE in
            "full")
                required_components=("database/" "configs/" "data/")
                ;;
            "database")
                required_components=("database/")
                ;;
            "configs")
                required_components=("configs/")
                ;;
            "data")
                required_components=("data/")
                ;;
        esac

        for component in "${required_components[@]}"; do
            if ! tar -tzf "$BACKUP_FILE" | grep -q "^$component"; then
                print_status "ERROR" "Backup missing required component: $component"
                exit 1
            fi
        done

        print_status "SUCCESS" "Backup validation passed"
    fi
}

# Function to create pre-restore snapshot
create_pre_restore_snapshot() {
    if [[ "$CREATE_SNAPSHOT" != "true" ]]; then
        print_status "INFO" "Skipping pre-restore snapshot"
        return 0
    fi

    print_status "INFO" "Creating pre-restore snapshot..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would create pre-restore snapshot"
        return 0
    fi

    local snapshot_id="pre_restore_${TIMESTAMP}"
    bash "${PROJECT_ROOT}/scripts/backup/backup.sh" \
        --backup-id "$snapshot_id" \
        --type snapshot \
        --compress \
        --no-verify || {
            print_status "ERROR" "Failed to create pre-restore snapshot"
            exit 1
        }

    SNAPSHOT_ID="$snapshot_id"
    print_status "SUCCESS" "Pre-restore snapshot created: $snapshot_id"
}

# Function to stop services safely
stop_services_safely() {
    print_status "INFO" "Stopping services for restoration..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would stop services"
        return 0
    fi

    cd "$PROJECT_ROOT"

    # Determine which services to stop based on restore type
    local services_to_stop=()

    case $BACKUP_TYPE in
        "full")
            services_to_stop=("scheduler" "trade_executor" "risk_manager" "strategy_engine" "data_collector" "export_service" "maintenance_service")
            ;;
        "database")
            services_to_stop=("scheduler" "trade_executor" "risk_manager" "strategy_engine" "data_collector")
            ;;
        "configs")
            services_to_stop=("scheduler" "trade_executor" "risk_manager" "strategy_engine" "data_collector" "export_service" "maintenance_service")
            ;;
        "data")
            services_to_stop=("export_service")
            ;;
    esac

    if [[ ${#services_to_stop[@]} -gt 0 ]]; then
        print_status "INFO" "Stopping services: ${services_to_stop[*]}"

        # Graceful shutdown with timeout
        timeout 60 docker-compose -f "$COMPOSE_FILE" stop "${services_to_stop[@]}" || {
            print_status "WARNING" "Graceful stop timed out, forcing stop..."
            docker-compose -f "$COMPOSE_FILE" kill "${services_to_stop[@]}"
        }

        print_status "SUCCESS" "Services stopped successfully"
    fi

    # Stop database last if needed
    if [[ "$RESTORE_DATABASE" == "true" ]]; then
        print_status "INFO" "Stopping database..."
        timeout 30 docker-compose -f "$COMPOSE_FILE" stop postgres || {
            print_status "WARNING" "Database graceful stop timed out, forcing stop..."
            docker-compose -f "$COMPOSE_FILE" kill postgres
        }
    fi
}

# Function to extract backup
extract_backup() {
    print_status "INFO" "Extracting backup archive..."

    local extract_dir="${PROJECT_ROOT}/tmp/restore_${TIMESTAMP}"
    mkdir -p "$extract_dir"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would extract backup to: $extract_dir"
        EXTRACT_DIR="$extract_dir"
        return 0
    fi

    # Extract backup
    tar -xzf "$BACKUP_FILE" -C "$extract_dir" || {
        print_status "ERROR" "Failed to extract backup archive"
        exit 1
    }

    EXTRACT_DIR="$extract_dir"
    print_status "SUCCESS" "Backup extracted to: $extract_dir"

    # List extracted contents
    if [[ "$VERBOSE" == "true" ]]; then
        print_status "INFO" "Extracted contents:"
        find "$extract_dir" -type f | sort | sed 's/^/  /'
    fi
}

# Function to restore database
restore_database() {
    if [[ "$RESTORE_DATABASE" != "true" ]]; then
        print_status "INFO" "Skipping database restore"
        return 0
    fi

    print_status "INFO" "Restoring database..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would restore database from backup"
        return 0
    fi

    local db_backup_dir="${EXTRACT_DIR}/database"
    if [[ ! -d "$db_backup_dir" ]]; then
        print_status "ERROR" "Database backup not found in archive"
        exit 1
    fi

    # Start database in recovery mode
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" up -d postgres

    # Wait for database to be ready
    print_status "INFO" "Waiting for database to be ready..."
    local retries=60
    while [[ $retries -gt 0 ]]; do
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U trading_user -d trading_db; then
            break
        fi
        sleep 2
        ((retries--))
    done

    if [[ $retries -eq 0 ]]; then
        print_status "ERROR" "Database failed to start for restoration"
        exit 1
    fi

    # Drop existing database and recreate
    print_status "INFO" "Dropping existing database..."
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U postgres -c "DROP DATABASE IF EXISTS trading_db;"
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U postgres -c "CREATE DATABASE trading_db OWNER trading_user;"

    # Restore database dump
    if [[ -f "${db_backup_dir}/trading_db.sql" ]]; then
        print_status "INFO" "Restoring database from SQL dump..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U trading_user -d trading_db < "${db_backup_dir}/trading_db.sql" || {
            print_status "ERROR" "Failed to restore database from SQL dump"
            exit 1
        }
    elif [[ -f "${db_backup_dir}/trading_db.dump" ]]; then
        print_status "INFO" "Restoring database from binary dump..."
        docker cp "${db_backup_dir}/trading_db.dump" "$(docker-compose -f "$COMPOSE_FILE" ps -q postgres):/tmp/"
        docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_restore -U trading_user -d trading_db /tmp/trading_db.dump || {
            print_status "ERROR" "Failed to restore database from binary dump"
            exit 1
        }
    else
        print_status "ERROR" "No database dump found in backup"
        exit 1
    fi

    print_status "SUCCESS" "Database restored successfully"
}

# Function to restore configurations
restore_configurations() {
    if [[ "$RESTORE_CONFIGS" != "true" ]]; then
        print_status "INFO" "Skipping configuration restore"
        return 0
    fi

    print_status "INFO" "Restoring configurations..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would restore configurations from backup"
        return 0
    fi

    local config_backup_dir="${EXTRACT_DIR}/configs"
    if [[ ! -d "$config_backup_dir" ]]; then
        print_status "ERROR" "Configuration backup not found in archive"
        exit 1
    fi

    # Backup current configs
    local current_config_backup="${PROJECT_ROOT}/data/config_backups/pre_restore_${TIMESTAMP}"
    mkdir -p "$current_config_backup"
    cp -r "${PROJECT_ROOT}/config" "$current_config_backup/" 2>/dev/null || true

    # Restore configurations
    print_status "INFO" "Restoring configuration files..."
    cp -r "${config_backup_dir}/"* "${PROJECT_ROOT}/config/" || {
        print_status "ERROR" "Failed to restore configuration files"
        exit 1
    }

    # Restore monitoring configurations
    if [[ -d "${EXTRACT_DIR}/monitoring" ]]; then
        print_status "INFO" "Restoring monitoring configurations..."
        cp -r "${EXTRACT_DIR}/monitoring/"* "${PROJECT_ROOT}/monitoring/" || true
    fi

    # Restore Docker compose files
    if [[ -f "${EXTRACT_DIR}/docker-compose.yml" ]]; then
        print_status "INFO" "Restoring Docker compose configuration..."
        cp "${EXTRACT_DIR}/docker-compose"*.yml "${PROJECT_ROOT}/" || true
    fi

    print_status "SUCCESS" "Configurations restored successfully"
}

# Function to restore data files
restore_data_files() {
    if [[ "$RESTORE_DATA" != "true" ]]; then
        print_status "INFO" "Skipping data file restore"
        return 0
    fi

    print_status "INFO" "Restoring data files..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would restore data files from backup"
        return 0
    fi

    local data_backup_dir="${EXTRACT_DIR}/data"
    if [[ ! -d "$data_backup_dir" ]]; then
        print_status "WARNING" "Data backup not found in archive"
        return 0
    fi

    # Create backup of current data
    local current_data_backup="${PROJECT_ROOT}/data/data_backups/pre_restore_${TIMESTAMP}"
    mkdir -p "$current_data_backup"

    # Backup critical data directories
    local data_dirs=("exports" "logs" "metrics" "state")
    for dir in "${data_dirs[@]}"; do
        if [[ -d "${PROJECT_ROOT}/data/$dir" ]]; then
            cp -r "${PROJECT_ROOT}/data/$dir" "$current_data_backup/" || true
        fi
    done

    # Restore data files
    print_status "INFO" "Restoring data directories..."
    for dir in "${data_dirs[@]}"; do
        if [[ -d "${data_backup_dir}/$dir" ]]; then
            rm -rf "${PROJECT_ROOT}/data/$dir" 2>/dev/null || true
            cp -r "${data_backup_dir}/$dir" "${PROJECT_ROOT}/data/" || {
                print_status "WARNING" "Failed to restore data directory: $dir"
            }
        fi
    done

    # Restore Parquet files
    if [[ -d "${data_backup_dir}/parquet" ]]; then
        print_status "INFO" "Restoring Parquet files..."
        rm -rf "${PROJECT_ROOT}/data/parquet" 2>/dev/null || true
        cp -r "${data_backup_dir}/parquet" "${PROJECT_ROOT}/data/" || {
            print_status "WARNING" "Failed to restore Parquet files"
        }
    fi

    print_status "SUCCESS" "Data files restored successfully"
}

# Function to verify restoration
verify_restoration() {
    if [[ "$VERIFY_INTEGRITY" != "true" ]]; then
        print_status "INFO" "Skipping restoration verification"
        return 0
    fi

    print_status "INFO" "Verifying restoration integrity..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would verify restoration integrity"
        return 0
    fi

    local verification_failed=false

    # Verify database
    if [[ "$RESTORE_DATABASE" == "true" ]]; then
        print_status "INFO" "Verifying database restoration..."

        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" up -d postgres

        # Wait for database
        local retries=30
        while [[ $retries -gt 0 ]]; do
            if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U trading_user -d trading_db; then
                break
            fi
            sleep 2
            ((retries--))
        done

        # Check database integrity
        local table_count=$(docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U trading_user -d trading_db -t -c \
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ' || echo "0")

        if [[ $table_count -lt 5 ]]; then
            print_status "ERROR" "Database verification failed: insufficient tables ($table_count)"
            verification_failed=true
        else
            print_status "SUCCESS" "Database verification passed ($table_count tables)"
        fi
    fi

    # Verify configurations
    if [[ "$RESTORE_CONFIGS" == "true" ]]; then
        print_status "INFO" "Verifying configuration restoration..."

        local required_configs=(
            "config/environments/.env.${ENVIRONMENT}"
            "docker-compose.yml"
        )

        for config in "${required_configs[@]}"; do
            if [[ ! -f "${PROJECT_ROOT}/$config" ]]; then
                print_status "ERROR" "Configuration verification failed: missing $config"
                verification_failed=true
            fi
        done

        if [[ "$verification_failed" != "true" ]]; then
            print_status "SUCCESS" "Configuration verification passed"
        fi
    fi

    # Verify data files
    if [[ "$RESTORE_DATA" == "true" ]]; then
        print_status "INFO" "Verifying data file restoration..."

        # Check if data directories exist and contain files
        local data_dirs=("exports" "logs" "metrics")
        for dir in "${data_dirs[@]}"; do
            if [[ -d "${PROJECT_ROOT}/data/$dir" ]]; then
                local file_count=$(find "${PROJECT_ROOT}/data/$dir" -type f | wc -l)
                print_status "INFO" "Data directory $dir contains $file_count files"
            fi
        done

        print_status "SUCCESS" "Data file verification completed"
    fi

    if [[ "$verification_failed" == "true" ]]; then
        print_status "ERROR" "Restoration verification failed"
        return 1
    fi

    print_status "SUCCESS" "All verification checks passed"
}

# Function to start services
start_services() {
    if [[ "$AUTO_START_SERVICES" != "true" ]]; then
        print_status "INFO" "Skipping service startup"
        return 0
    fi

    print_status "INFO" "Starting services after restoration..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would start services"
        return 0
    fi

    cd "$PROJECT_ROOT"

    # Start services in correct order
    local startup_order=()

    case $BACKUP_TYPE in
        "full")
            startup_order=("postgres" "redis" "data_collector" "strategy_engine" "risk_manager" "trade_executor" "scheduler" "export_service" "maintenance_service")
            ;;
        "database")
            startup_order=("postgres" "data_collector" "strategy_engine" "risk_manager" "trade_executor" "scheduler")
            ;;
        "configs")
            startup_order=("data_collector" "strategy_engine" "risk_manager" "trade_executor" "scheduler" "export_service" "maintenance_service")
            ;;
        "data")
            startup_order=("export_service")
            ;;
    esac

    for service in "${startup_order[@]}"; do
        print_status "INFO" "Starting $service..."
        docker-compose -f "$COMPOSE_FILE" up -d "$service"

        # Wait for service to be ready
        case $service in
            "postgres")
                local retries=30
                while [[ $retries -gt 0 ]]; do
                    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U trading_user -d trading_db; then
                        break
                    fi
                    sleep 2
                    ((retries--))
                done
                ;;
            "redis")
                local retries=15
                while [[ $retries -gt 0 ]]; do
                    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping; then
                        break
                    fi
                    sleep 2
                    ((retries--))
                done
                ;;
            *)
                # Wait for API services
                sleep 10
                ;;
        esac

        print_status "SUCCESS" "Service $service started"
    done

    print_status "SUCCESS" "All services started successfully"
}

# Function to run post-restore tests
run_post_restore_tests() {
    print_status "INFO" "Running post-restore tests..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would run post-restore tests"
        return 0
    fi

    cd "$PROJECT_ROOT"

    # Wait for services to stabilize
    sleep 30

    # Run health checks
    if [[ -f "scripts/health_check.sh" ]]; then
        bash scripts/health_check.sh || {
            print_status "ERROR" "Post-restore health checks failed"
            return 1
        }
    fi

    # Run basic functionality tests
    if [[ -f "scripts/run_tests.py" ]]; then
        python3 scripts/run_tests.py --integration-tests || {
            print_status "ERROR" "Post-restore integration tests failed"
            return 1
        }
    fi

    # Test database connectivity and basic queries
    if [[ "$RESTORE_DATABASE" == "true" ]]; then
        print_status "INFO" "Testing database functionality..."

        local test_query="SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
        local table_count=$(docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U trading_user -d trading_db -t -c "$test_query" 2>/dev/null | tr -d ' ' || echo "0")

        if [[ $table_count -gt 0 ]]; then
            print_status "SUCCESS" "Database functionality test passed ($table_count tables)"
        else
            print_status "ERROR" "Database functionality test failed"
            return 1
        fi
    fi

    print_status "SUCCESS" "All post-restore tests passed"
}

# Function to cleanup temporary files
cleanup_temporary_files() {
    print_status "INFO" "Cleaning up temporary files..."

    if [[ -n "${EXTRACT_DIR:-}" ]] && [[ -d "$EXTRACT_DIR" ]]; then
        rm -rf "$EXTRACT_DIR"
        print_status "SUCCESS" "Temporary files cleaned up"
    fi
}

# Function to show restoration summary
show_restoration_summary() {
    print_status "INFO" "Restoration Summary"
    echo ""
    echo "📋 Restoration Details:"
    echo "   Backup ID: ${BACKUP_ID}"
    echo "   Backup File: ${BACKUP_FILE}"
    echo "   Backup Type: ${BACKUP_TYPE}"
    echo "   Environment: ${ENVIRONMENT}"
    echo "   Started: $(head -2 "$RESTORE_LOG" | tail -1 | awk '{print $2, $3}')"
    echo "   Completed: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "✅ Restored Components:"
    [[ "$RESTORE_DATABASE" == "true" ]] && echo "   • Database"
    [[ "$RESTORE_CONFIGS" == "true" ]] && echo "   • Configurations"
    [[ "$RESTORE_DATA" == "true" ]] && echo "   • Data Files"
    echo ""
    echo "📊 Post-Restore Status:"
    docker-compose -f "$COMPOSE_FILE" ps 2>/dev/null || echo "   Services status unavailable"
    echo ""
    echo "📖 Logs:"
    echo "   Restoration Log: $RESTORE_LOG"
    if [[ -n "${SNAPSHOT_ID:-}" ]]; then
        echo "   Pre-restore Snapshot: ${SNAPSHOT_ID}"
    fi
    echo ""
    echo "🔧 Next Steps:"
    echo "   1. Verify system functionality"
    echo "   2. Run comprehensive tests"
    echo "   3. Monitor performance"
    echo "   4. Update any environment-specific settings"
    echo ""
}

# Function to handle restoration failure
handle_restoration_failure() {
    local exit_code=$?
    print_status "CRITICAL" "Restoration failed with exit code $exit_code"

    if [[ -n "${SNAPSHOT_ID:-}" ]] && [[ "$FORCE_RESTORE" != "true" ]]; then
        echo ""
        echo -n "Attempt to rollback to pre-restore snapshot? (Y/n): "
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            print_status "INFO" "Rolling back to pre-restore snapshot..."
            bash "${SCRIPT_DIR}/restore.sh" --backup-id "$SNAPSHOT_ID" --force --no-snapshot
        fi
    fi

    cleanup_temporary_files
    print_status "INFO" "Restoration log available at: $RESTORE_LOG"
    exit $exit_code
}

# Function to confirm restoration
confirm_restoration() {
    if [[ "$FORCE_RESTORE" == "true" ]]; then
        return 0
    fi

    echo ""
    echo "⚠️  RESTORATION WARNING"
    echo "======================="
    echo ""
    echo "You are about to restore from backup:"
    echo "   Backup ID: ${BACKUP_ID}"
    echo "   Backup Type: ${BACKUP_TYPE}"
    echo "   Environment: ${ENVIRONMENT}"
    echo ""
    echo "This will:"
    [[ "$RESTORE_DATABASE" == "true" ]] && echo "   • REPLACE the current database"
    [[ "$RESTORE_CONFIGS" == "true" ]] && echo "   • REPLACE current configurations"
    [[ "$RESTORE_DATA" == "true" ]] && echo "   • REPLACE current data files"
    echo ""
    [[ "$CREATE_SNAPSHOT" == "true" ]] && echo "   • A pre-restore snapshot will be created for rollback"
    echo ""
    echo -n "Do you want to continue? (y/N): "
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "INFO" "Restoration cancelled by user"
        exit 0
    fi
}

# Main execution function
main() {
    # Create initial log entry
    {
        echo "=== AI Trading System Backup Restoration ==="
        echo "Started at: $(date)"
        echo "Backup ID: $BACKUP_ID"
        echo "Backup Type: $BACKUP_TYPE"
        echo "Environment: $ENVIRONMENT"
        echo "User: $(whoami)"
        echo ""
    } >> "$RESTORE_LOG"

    print_status "INFO" "AI Trading System Backup Restoration"
    print_status "INFO" "Backup ID: ${BACKUP_ID:-'(to be selected)'}"
    print_status "INFO" "Backup Type: $BACKUP_TYPE"
    print_status "INFO" "Environment: $ENVIRONMENT"
    print_status "INFO" "Restore Log: $RESTORE_LOG"

    # Set up error handling
    trap handle_restoration_failure ERR
    trap cleanup_temporary_files EXIT

    # Find and validate backup
    find_and_validate_backup

    # Confirm restoration
    confirm_restoration

    # Create pre-restore snapshot
    create_pre_restore_snapshot

    # Extract backup
    extract_backup

    # Stop services
    stop_services_safely

    # Perform restoration
    restore_database
    restore_configurations
    restore_data_files

    # Verify restoration
    verify_restoration

    # Start services
    start_services

    # Run post-restore tests
    run_post_restore_tests

    # Show summary
    show_restoration_summary

    print_status "SUCCESS" "Backup restoration completed successfully!"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    main
fi
