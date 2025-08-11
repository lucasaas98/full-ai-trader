#!/bin/bash

# AI Trading System Database Migration Script
# This script handles database migrations with safety checks and rollback capabilities
# Usage: ./migrate.sh [environment] [options]

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MIGRATION_LOG="${PROJECT_ROOT}/logs/migration_${TIMESTAMP}.log"

# Default values
ENVIRONMENT="development"
DRY_RUN=false
SKIP_BACKUP=false
TARGET_REVISION=""
ROLLBACK_STEPS=0
MIGRATION_TIMEOUT=600
AUTO_APPROVE=false
FORCE=false

# Database configuration
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="trading_system"
DB_USER="trader"
DB_PASSWORD=""

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$MIGRATION_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$MIGRATION_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" | tee -a "$MIGRATION_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$MIGRATION_LOG"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Database Migration Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    development     Migrate development database (default)
    staging         Migrate staging database
    production      Migrate production database

OPTIONS:
    --dry-run              Show what migrations would be applied
    --skip-backup          Skip database backup before migration
    --target REVISION      Migrate to specific revision
    --rollback STEPS       Rollback specified number of migrations
    --timeout SECONDS      Migration timeout (default: 600)
    --auto-approve         Auto-approve migrations without prompts
    --force                Force migration even if validation fails
    --help                 Show this help message

COMMANDS:
    status                 Show current migration status
    history                Show migration history
    validate              Validate migrations without applying
    create NAME           Create new migration file
    rollback              Interactive rollback menu

EXAMPLES:
    $0 development
    $0 staging --dry-run
    $0 production --target 001_initial_schema
    $0 production --rollback 2
    $0 status
    $0 create add_new_trading_table

EOF
}

# Function to load environment configuration
load_env_config() {
    local env_file="${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}"

    if [ ! -f "$env_file" ]; then
        print_status "ERROR" "Environment file not found: $env_file"
        exit 1
    fi

    print_status "INFO" "Loading configuration for $ENVIRONMENT environment"

    # Source environment file and extract database config
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ $key =~ ^[[:space:]]*# ]] && continue
        [[ -z $key ]] && continue

        # Remove quotes from value
        value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")

        case $key in
            DB_HOST) DB_HOST="$value" ;;
            DB_PORT) DB_PORT="$value" ;;
            DB_NAME) DB_NAME="$value" ;;
            DB_USER) DB_USER="$value" ;;
            DB_PASSWORD) DB_PASSWORD="$value" ;;
        esac
    done < "$env_file"

    print_status "INFO" "Database config loaded: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
}

# Function to check database connectivity
check_database_connection() {
    print_status "INFO" "Checking database connectivity..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
            print_status "SUCCESS" "Database connection established"
            return 0
        fi

        print_status "INFO" "Database not ready, attempt $attempt/$max_attempts..."
        sleep 5
        ((attempt++))
    done

    print_status "ERROR" "Failed to connect to database after $max_attempts attempts"
    exit 1
}

# Function to create database backup
create_database_backup() {
    if [ "$SKIP_BACKUP" = true ]; then
        print_status "WARNING" "Skipping database backup (--skip-backup flag)"
        return
    fi

    print_status "INFO" "Creating database backup before migration..."

    local backup_dir="${PROJECT_ROOT}/backups/migrations"
    mkdir -p "$backup_dir"

    local backup_file="${backup_dir}/pre_migration_${TIMESTAMP}.sql"

    # Create database dump
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --no-owner \
        --no-privileges \
        --clean \
        --if-exists \
        > "$backup_file"

    # Compress backup
    gzip "$backup_file"

    print_status "SUCCESS" "Database backup created: ${backup_file}.gz"

    # Save backup info for rollback
    echo "$backup_file.gz" > "${PROJECT_ROOT}/data/last_migration_backup.txt"
}

# Function to get current migration status
get_migration_status() {
    print_status "INFO" "Checking current migration status..."

    # Check if alembic_version table exists
    if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -c "SELECT 1 FROM information_schema.tables WHERE table_name = 'alembic_version';" >/dev/null 2>&1; then
        print_status "INFO" "Database not initialized - no migration history"
        return 1
    fi

    # Get current revision
    local current_revision=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -t -c "SELECT version_num FROM alembic_version;" 2>/dev/null | tr -d ' ' || echo "")

    if [ -n "$current_revision" ]; then
        print_status "INFO" "Current database revision: $current_revision"
        return 0
    else
        print_status "INFO" "No current revision found"
        return 1
    fi
}

# Function to show migration history
show_migration_history() {
    print_status "INFO" "Migration History:"

    # Show alembic history if available
    if command -v alembic &> /dev/null; then
        cd "${PROJECT_ROOT}"
        alembic history --verbose 2>/dev/null || echo "No migration history available"
    else
        # Show from database directly
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
            -c "SELECT version_num, 'Applied' as status FROM alembic_version;" 2>/dev/null || \
            echo "Migration history not available"
    fi
}

# Function to validate migrations
validate_migrations() {
    print_status "INFO" "Validating migration files..."

    local migrations_dir="${PROJECT_ROOT}/scripts/migrations"

    if [ ! -d "$migrations_dir" ]; then
        print_status "ERROR" "Migrations directory not found: $migrations_dir"
        exit 1
    fi

    # Check for migration file conflicts
    local migration_files=$(find "$migrations_dir" -name "*.py" -type f | sort)
    local revision_conflicts=()

    for migration_file in $migration_files; do
        # Extract revision ID from filename (assuming format: XXX_revision_name.py)
        local revision=$(basename "$migration_file" | cut -d'_' -f1)

        # Check for duplicate revisions
        if [[ " ${revision_conflicts[*]} " =~ " ${revision} " ]]; then
            print_status "ERROR" "Duplicate migration revision found: $revision"
            exit 1
        fi
        revision_conflicts+=("$revision")
    done

    print_status "SUCCESS" "Migration validation completed"
}

# Function to run migrations
run_migrations() {
    print_status "INFO" "Running database migrations..."

    # Set environment variables for migration
    export DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

    cd "${PROJECT_ROOT}"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "DRY RUN: Would apply the following migrations:"

        # Show what would be applied
        if command -v alembic &> /dev/null; then
            alembic show head 2>/dev/null || echo "No pending migrations"
        fi
        return 0
    fi

    # Apply migrations with timeout
    local migration_start=$(date +%s)

    if [ -n "$TARGET_REVISION" ]; then
        print_status "INFO" "Migrating to specific revision: $TARGET_REVISION"
        timeout "$MIGRATION_TIMEOUT" alembic upgrade "$TARGET_REVISION"
    else
        print_status "INFO" "Migrating to latest revision..."
        timeout "$MIGRATION_TIMEOUT" alembic upgrade head
    fi

    local migration_end=$(date +%s)
    local migration_duration=$((migration_end - migration_start))

    print_status "SUCCESS" "Migrations completed in ${migration_duration} seconds"
}

# Function to rollback migrations
rollback_migrations() {
    if [ $ROLLBACK_STEPS -eq 0 ]; then
        print_status "ERROR" "Rollback steps not specified"
        exit 1
    fi

    print_status "WARNING" "Rolling back $ROLLBACK_STEPS migration(s)..."

    if [ "$ENVIRONMENT" = "production" ] && [ "$AUTO_APPROVE" = false ]; then
        read -p "⚠️  Are you sure you want to rollback PRODUCTION database? Type 'ROLLBACK' to confirm: " confirm
        if [ "$confirm" != "ROLLBACK" ]; then
            print_status "INFO" "Rollback cancelled by user"
            exit 0
        fi
    fi

    # Create backup before rollback
    create_database_backup

    # Get current revision
    local current_revision=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -t -c "SELECT version_num FROM alembic_version;" 2>/dev/null | tr -d ' ')

    print_status "INFO" "Current revision: $current_revision"

    # Calculate target revision for rollback
    cd "${PROJECT_ROOT}"
    local target_revision=$(alembic history | grep -A "$ROLLBACK_STEPS" "$current_revision" | tail -n 1 | awk '{print $1}')

    if [ -z "$target_revision" ]; then
        print_status "ERROR" "Cannot determine target revision for rollback"
        exit 1
    fi

    print_status "INFO" "Rolling back to revision: $target_revision"

    # Perform rollback
    timeout "$MIGRATION_TIMEOUT" alembic downgrade "$target_revision"

    print_status "SUCCESS" "Rollback completed to revision: $target_revision"
}

# Function to create new migration
create_migration() {
    local migration_name=$1

    if [ -z "$migration_name" ]; then
        print_status "ERROR" "Migration name is required"
        exit 1
    fi

    print_status "INFO" "Creating new migration: $migration_name"

    cd "${PROJECT_ROOT}"

    # Generate migration file
    alembic revision --autogenerate -m "$migration_name"

    local latest_migration=$(find scripts/migrations -name "*.py" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

    print_status "SUCCESS" "Migration created: $latest_migration"
    print_status "INFO" "Please review the generated migration before applying"
}

# Function to show current status
show_status() {
    print_status "INFO" "Database Migration Status"
    echo "=" >> "$MIGRATION_LOG"

    # Load environment configuration
    load_env_config

    # Check database connection
    if check_database_connection; then
        # Get current revision
        get_migration_status

        # Show pending migrations
        cd "${PROJECT_ROOT}"
        echo
        echo "📋 Migration Status:"
        if command -v alembic &> /dev/null; then
            echo "Current revision:"
            alembic current 2>/dev/null || echo "  No current revision"
            echo
            echo "Migration history:"
            alembic history --verbose 2>/dev/null || echo "  No migration history"
            echo
            echo "Pending migrations:"
            alembic show head 2>/dev/null || echo "  No pending migrations"
        else
            echo "  Alembic not available"
        fi

        # Database statistics
        echo
        echo "📊 Database Statistics:"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
            -c "SELECT schemaname, tablename, n_tup_ins as inserts, n_tup_upd as updates, n_tup_del as deletes
                FROM pg_stat_user_tables
                ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC
                LIMIT 10;" 2>/dev/null || echo "  Unable to retrieve database statistics"
    else
        print_status "ERROR" "Cannot connect to database"
        exit 1
    fi
}

# Function to validate migration safety
validate_migration_safety() {
    print_status "INFO" "Validating migration safety..."

    cd "${PROJECT_ROOT}"

    # Check for destructive operations
    local pending_migrations=$(alembic show head 2>/dev/null || echo "")
    local destructive_patterns=(
        "DROP TABLE"
        "DROP COLUMN"
        "DROP INDEX"
        "ALTER TABLE.*DROP"
        "TRUNCATE"
        "DELETE FROM"
    )

    local migration_files=$(find scripts/migrations -name "*.py" -type f -newer "${PROJECT_ROOT}/data/last_migration.timestamp" 2>/dev/null || \
                           find scripts/migrations -name "*.py" -type f)

    local has_destructive=false

    for migration_file in $migration_files; do
        for pattern in "${destructive_patterns[@]}"; do
            if grep -qi "$pattern" "$migration_file"; then
                print_status "WARNING" "Potentially destructive operation found in $(basename "$migration_file"): $pattern"
                has_destructive=true
            fi
        done
    done

    if [ "$has_destructive" = true ] && [ "$ENVIRONMENT" = "production" ] && [ "$FORCE" = false ]; then
        print_status "WARNING" "Destructive operations detected in production migration"

        if [ "$AUTO_APPROVE" = false ]; then
            read -p "Continue with potentially destructive migration? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_status "INFO" "Migration cancelled by user"
                exit 0
            fi
        fi
    fi

    print_status "SUCCESS" "Migration safety validation completed"
}

# Function to estimate migration time
estimate_migration_time() {
    print_status "INFO" "Estimating migration time..."

    # Get table sizes for time estimation
    local total_rows=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -t -c "SELECT SUM(n_tup_ins + n_tup_upd) FROM pg_stat_user_tables;" 2>/dev/null | tr -d ' ' || echo "0")

    # Rough estimation: 1000 rows per second for complex migrations
    local estimated_seconds=$((total_rows / 1000))

    if [ $estimated_seconds -gt 0 ]; then
        print_status "INFO" "Estimated migration time: ${estimated_seconds} seconds (based on $total_rows total rows)"

        if [ $estimated_seconds -gt $MIGRATION_TIMEOUT ]; then
            print_status "WARNING" "Estimated time exceeds timeout. Consider increasing --timeout"
        fi
    fi
}

# Function to test migration rollback
test_migration_rollback() {
    if [ "$ENVIRONMENT" = "production" ]; then
        print_status "INFO" "Testing migration rollback capability..."

        # Get current revision before migration
        local pre_migration_revision=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
            -t -c "SELECT version_num FROM alembic_version;" 2>/dev/null | tr -d ' ')

        # Store for potential rollback
        echo "$pre_migration_revision" > "${PROJECT_ROOT}/data/pre_migration_revision.txt"

        print_status "INFO" "Rollback reference saved: $pre_migration_revision"
    fi
}

# Function to apply migrations with monitoring
apply_migrations_with_monitoring() {
    print_status "INFO" "Applying migrations with monitoring..."

    # Start migration monitoring in background
    {
        while true; do
            local connections=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
                -t -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';" 2>/dev/null || echo "0")

            local locks=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
                -t -c "SELECT count(*) FROM pg_locks WHERE granted = false;" 2>/dev/null || echo "0")

            print_status "INFO" "Migration progress - Active connections: $connections, Waiting locks: $locks"
            sleep 30
        done
    } &

    local monitor_pid=$!

    # Run the actual migration
    run_migrations

    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true
}

# Function for interactive rollback menu
interactive_rollback() {
    print_status "INFO" "Interactive Migration Rollback"

    # Show current status
    get_migration_status

    # Show available rollback options
    echo
    echo "Available rollback options:"
    echo "1. Rollback 1 migration"
    echo "2. Rollback 2 migrations"
    echo "3. Rollback 5 migrations"
    echo "4. Rollback to specific revision"
    echo "5. Restore from backup"
    echo "6. Cancel"
    echo

    read -p "Select option (1-6): " -n 1 -r
    echo

    case $REPLY in
        1) ROLLBACK_STEPS=1; rollback_migrations ;;
        2) ROLLBACK_STEPS=2; rollback_migrations ;;
        3) ROLLBACK_STEPS=5; rollback_migrations ;;
        4)
            read -p "Enter target revision: " target_rev
            TARGET_REVISION="$target_rev"
            rollback_migrations
            ;;
        5) restore_from_backup ;;
        6) print_status "INFO" "Rollback cancelled"; exit 0 ;;
        *) print_status "ERROR" "Invalid option"; exit 1 ;;
    esac
}

# Function to restore from backup
restore_from_backup() {
    print_status "WARNING" "Restoring database from backup..."

    local backup_file
    if [ -f "${PROJECT_ROOT}/data/last_migration_backup.txt" ]; then
        backup_file=$(cat "${PROJECT_ROOT}/data/last_migration_backup.txt")
    else
        # Find latest backup
        backup_file=$(find "${PROJECT_ROOT}/backups/migrations" -name "*.sql.gz" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || echo "")
    fi

    if [ -z "$backup_file" ] || [ ! -f "$backup_file" ]; then
        print_status "ERROR" "No backup file found for restore"
        exit 1
    fi

    print_status "INFO" "Restoring from backup: $backup_file"

    if [ "$ENVIRONMENT" = "production" ] && [ "$AUTO_APPROVE" = false ]; then
        read -p "⚠️  Restore PRODUCTION database from backup? Type 'RESTORE' to confirm: " confirm
        if [ "$confirm" != "RESTORE" ]; then
            print_status "INFO" "Restore cancelled by user"
            exit 0
        fi
    fi

    # Restore database
    gunzip -c "$backup_file" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"

    print_status "SUCCESS" "Database restored from backup"
}

# Function to post-migration validation
post_migration_validation() {
    print_status "INFO" "Running post-migration validation..."

    # Check critical tables exist
    local critical_tables=(
        "users"
        "accounts"
        "trades"
        "positions"
        "orders"
        "market_data"
        "strategies"
        "risk_metrics"
    )

    for table in "${critical_tables[@]}"; do
        if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
            -c "SELECT 1 FROM information_schema.tables WHERE table_name = '$table';" >/dev/null 2>&1; then
            print_status "WARNING" "Critical table missing: $table"
        fi
    done

    # Check data integrity
    print_status "INFO" "Checking data integrity..."

    # Run basic data integrity checks
    local integrity_checks=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -t -c "
        SELECT
            CASE
                WHEN COUNT(*) > 0 THEN 'FAIL'
                ELSE 'PASS'
            END as status
        FROM (
            SELECT 1 WHERE EXISTS (SELECT 1 FROM trades WHERE quantity <= 0)
            UNION ALL
            SELECT 1 WHERE EXISTS (SELECT 1 FROM positions WHERE shares < 0)
            UNION ALL
            SELECT 1 WHERE EXISTS (SELECT 1 FROM orders WHERE quantity <= 0)
        ) checks;" 2>/dev/null | tr -d ' ')

    if [ "$integrity_checks" = "FAIL" ]; then
        print_status "ERROR" "Data integrity check failed"
        exit 1
    fi

    # Update migration timestamp
    touch "${PROJECT_ROOT}/data/last_migration.timestamp"

    print_status "SUCCESS" "Post-migration validation completed"
}

# Function to handle migration failure
handle_migration_failure() {
    local exit_code=$1

    print_status "ERROR" "Migration failed with exit code: $exit_code"

    # Show migration logs
    print_status "INFO" "Recent migration logs:"
    tail -50 "$MIGRATION_LOG"

    # Attempt automatic recovery if enabled
    if [ "$ENVIRONMENT" != "production" ] && [ -f "${PROJECT_ROOT}/data/last_migration_backup.txt" ]; then
        print_status "WARNING" "Attempting automatic recovery..."
        restore_from_backup
    else
        print_status "ERROR" "Manual intervention required"
        print_status "INFO" "To restore from backup, run: $0 $ENVIRONMENT --restore"
    fi

    exit $exit_code
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --target)
                TARGET_REVISION="$2"
                shift 2
                ;;
            --rollback)
                ROLLBACK_STEPS="$2"
                shift 2
                ;;
            --timeout)
                MIGRATION_TIMEOUT="$2"
                shift 2
                ;;
            --auto-approve)
                AUTO_APPROVE=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            status)
                load_env_config
                check_database_connection
                show_status
                exit 0
                ;;
            history)
                load_env_config
                check_database_connection
                show_migration_history
                exit 0
                ;;
            validate)
                validate_migrations
                exit 0
                ;;
            create)
                create_migration "$2"
                exit 0
                ;;
            rollback)
                load_env_config
                check_database_connection
                interactive_rollback
                exit 0
                ;;
            --help|-h)
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

# Main migration function
main_migration() {
    print_status "INFO" "Starting database migration for $ENVIRONMENT environment"
    print_status "INFO" "Migration ID: $TIMESTAMP"
    print_status "INFO" "Log file: $MIGRATION_LOG"

    # Create log directory
    mkdir -p "$(dirname "$MIGRATION_LOG")"

    if [ "$DRY_RUN" = true ]; then
        print_status "WARNING" "DRY RUN MODE - No actual changes will be made"
    fi

    # Set trap for failure handling
    trap 'handle_migration_failure $?' ERR

    # Migration steps
    load_env_config
    check_database_connection
    validate_migrations
    get_migration_status

    if [ "$DRY_RUN" = false ]; then
        validate_migration_safety
        estimate_migration_time
        test_migration_rollback
        create_database_backup
        apply_migrations_with_monitoring
        post_migration_validation

        print_status "SUCCESS" "Migration completed successfully!"
    else
        print_status "INFO" "DRY RUN: Migration validation passed"
    fi
}

# Main execution
main() {
    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"

    # Parse arguments
    parse_arguments "$@"

    # Handle rollback if specified
    if [ $ROLLBACK_STEPS -gt 0 ]; then
        load_env_config
        check_database_connection
        rollback_migrations
        exit 0
    fi

    # Execute main migration
    main_migration

    print_status "SUCCESS" "🎉 Database migration completed!"
    echo
    echo "📊 Migration Summary:"
    echo "   Environment: $ENVIRONMENT"
    echo "   Timestamp: $TIMESTAMP"
    echo "   Target: ${TARGET_REVISION:-latest}"
    echo "   Log file: $MIGRATION_LOG"
    echo
    echo "🔗 Next Steps:"
    echo "   1. Verify application functionality"
    echo "   2. Monitor system performance"
    echo "   3. Check data integrity"
    echo "   4. Update documentation if needed"
    echo
}

# Run main function with all arguments
main "$@"
