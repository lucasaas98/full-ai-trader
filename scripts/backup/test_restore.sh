#!/bin/bash

# AI Trading System Automated Restore Testing Script
# This script automatically tests backup restore procedures to ensure backups are viable
# Usage: ./test_restore.sh [environment] [backup_file]

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
TEST_LOG="${PROJECT_ROOT}/logs/restore_test_${TIMESTAMP}.log"

# Default values
ENVIRONMENT="development"
BACKUP_FILE=""
CLEANUP_AFTER_TEST=true
PARALLEL_TESTING=true
TEST_TIMEOUT=1800
CREATE_TEST_BACKUP=true
VERIFY_DATA_INTEGRITY=true

# Test environment configuration
TEST_DB_NAME="trading_system_restore_test"
TEST_REDIS_DB=15
TEST_DATA_DIR="${PROJECT_ROOT}/data/restore_test"
TEST_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.test-restore.yml"

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$TEST_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$TEST_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" | tee -a "$TEST_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$TEST_LOG"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Automated Restore Testing Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    development     Test with development backups (default)
    staging         Test with staging backups
    production      Test with production backups (read-only)

OPTIONS:
    --backup-file FILE     Test specific backup file
    --no-cleanup          Don't cleanup test environment after testing
    --sequential          Run tests sequentially instead of parallel
    --timeout SECONDS     Test timeout in seconds (default: 1800)
    --no-create-backup    Don't create fresh backup for testing
    --skip-integrity      Skip data integrity verification
    --help               Show this help message

COMMANDS:
    all                  Test all available backups
    latest               Test latest backup only
    daily                Run daily restore test (automated)
    weekly               Run weekly comprehensive test
    report               Generate test report

EXAMPLES:
    $0 development
    $0 staging --backup-file /backups/full_20240101_120000.tar.gz.enc
    $0 production --no-create-backup --timeout 3600
    $0 all --sequential
    $0 daily

EOF
}

# Function to setup test environment
setup_test_environment() {
    print_status "INFO" "Setting up isolated test environment..."

    # Create test directories
    mkdir -p "$TEST_DATA_DIR"
    mkdir -p "${PROJECT_ROOT}/logs"

    # Create isolated test Docker Compose file
    cat > "$TEST_COMPOSE_FILE" << EOF
version: "3.8"

services:
  test_postgres:
    image: postgres:15-alpine
    container_name: restore_test_postgres
    environment:
      POSTGRES_DB: $TEST_DB_NAME
      POSTGRES_USER: test_trader
      POSTGRES_PASSWORD: test_password
    ports:
      - "15432:5432"
    volumes:
      - test_postgres_data:/var/lib/postgresql/data
    networks:
      - test_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test_trader -d $TEST_DB_NAME"]
      interval: 5s
      timeout: 3s
      retries: 5

  test_redis:
    image: redis:7-alpine
    container_name: restore_test_redis
    ports:
      - "16379:6379"
    volumes:
      - test_redis_data:/data
    networks:
      - test_network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  test_postgres_data:
  test_redis_data:

networks:
  test_network:
    driver: bridge
EOF

    print_status "SUCCESS" "Test environment configuration created"
}

# Function to start test containers
start_test_containers() {
    print_status "INFO" "Starting test containers..."

    # Start test database and Redis
    docker-compose -f "$TEST_COMPOSE_FILE" up -d

    # Wait for services to be healthy
    local max_wait=60
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        if docker-compose -f "$TEST_COMPOSE_FILE" ps test_postgres | grep -q "healthy" && \
           docker-compose -f "$TEST_COMPOSE_FILE" ps test_redis | grep -q "healthy"; then
            print_status "SUCCESS" "Test containers are healthy"
            return 0
        fi
        sleep 5
        ((elapsed += 5))
        print_status "INFO" "Waiting for test containers... (${elapsed}s/${max_wait}s)"
    done

    print_status "ERROR" "Test containers failed to become healthy"
    exit 1
}

# Function to create test backup
create_test_backup() {
    if [ "$CREATE_TEST_BACKUP" = false ]; then
        print_status "INFO" "Skipping test backup creation"
        return
    fi

    print_status "INFO" "Creating fresh test backup..."

    # Load current environment config
    local env_file="${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}"
    if [ -f "$env_file" ]; then
        source "$env_file"
    fi

    # Create a fresh backup for testing
    "${SCRIPT_DIR}/backup.sh" full --env "$ENVIRONMENT" --quiet

    # Find the latest backup
    BACKUP_FILE=$(find "${PROJECT_ROOT}/backups" -name "full_*.tar*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "$BACKUP_FILE" ]; then
        print_status "SUCCESS" "Test backup created: $(basename "$BACKUP_FILE")"
    else
        print_status "ERROR" "Failed to create test backup"
        exit 1
    fi
}

# Function to test database restore
test_database_restore() {
    local backup_file=$1
    print_status "INFO" "Testing database restore from $(basename "$backup_file")..."

    # Extract database backup from full backup if needed
    local db_backup_file="$backup_file"

    if [[ "$backup_file" =~ full_ ]]; then
        # Extract database backup from full backup
        local extract_dir="${TEST_DATA_DIR}/extract_${TIMESTAMP}"
        mkdir -p "$extract_dir"

        # Decrypt if necessary
        if [[ "$backup_file" =~ \.enc$ ]]; then
            local decrypted_file="${extract_dir}/decrypted.tar"
            echo "$ENCRYPTION_KEY" | gpg --batch --yes --passphrase-fd 0 --decrypt "$backup_file" > "$decrypted_file"
            backup_file="$decrypted_file"
        fi

        # Extract
        tar -xzf "$backup_file" -C "$extract_dir"

        # Find database backup
        db_backup_file=$(find "$extract_dir" -name "postgres_*.sql*" | head -1)

        if [ -z "$db_backup_file" ]; then
            print_status "ERROR" "No database backup found in archive"
            return 1
        fi
    fi

    # Decompress if necessary
    if [[ "$db_backup_file" =~ \.gz$ ]]; then
        local decompressed="${TEST_DATA_DIR}/postgres_test_${TIMESTAMP}.sql"
        gunzip -c "$db_backup_file" > "$decompressed"
        db_backup_file="$decompressed"
    fi

    # Restore to test database
    print_status "INFO" "Restoring database backup to test instance..."

    PGPASSWORD="test_password" psql \
        -h localhost \
        -p 15432 \
        -U test_trader \
        -d "$TEST_DB_NAME" \
        < "$db_backup_file" \
        2>>"$TEST_LOG"

    # Verify restore
    local table_count=$(PGPASSWORD="test_password" psql \
        -h localhost \
        -p 15432 \
        -U test_trader \
        -d "$TEST_DB_NAME" \
        -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" \
        2>/dev/null | tr -d ' ')

    if [ "$table_count" -gt 0 ]; then
        print_status "SUCCESS" "Database restore test passed ($table_count tables restored)"
    else
        print_status "ERROR" "Database restore test failed (no tables found)"
        return 1
    fi

    # Test critical table data
    test_critical_table_data

    # Cleanup extracted files
    if [[ "$backup_file" =~ extract_ ]]; then
        rm -rf "$(dirname "$db_backup_file")"
    fi
}

# Function to test critical table data
test_critical_table_data() {
    print_status "INFO" "Testing critical table data integrity..."

    local critical_tables=(
        "alembic_version"
        "users"
        "accounts"
        "trades"
        "positions"
        "market_data"
    )

    local failed_tables=()

    for table in "${critical_tables[@]}"; do
        # Check if table exists and has data
        local row_count=$(PGPASSWORD="test_password" psql \
            -h localhost \
            -p 15432 \
            -U test_trader \
            -d "$TEST_DB_NAME" \
            -t -c "SELECT count(*) FROM $table;" \
            2>/dev/null | tr -d ' ' || echo "0")

        if [ "$row_count" = "0" ]; then
            # Check if table exists but is empty
            local table_exists=$(PGPASSWORD="test_password" psql \
                -h localhost \
                -p 15432 \
                -U test_trader \
                -d "$TEST_DB_NAME" \
                -t -c "SELECT count(*) FROM information_schema.tables WHERE table_name = '$table';" \
                2>/dev/null | tr -d ' ')

            if [ "$table_exists" = "0" ]; then
                failed_tables+=("$table (missing)")
            else
                print_status "WARNING" "Table $table exists but is empty"
            fi
        else
            print_status "INFO" "Table $table: $row_count rows"
        fi
    done

    if [ ${#failed_tables[@]} -gt 0 ]; then
        print_status "ERROR" "Critical tables missing: ${failed_tables[*]}"
        return 1
    fi

    print_status "SUCCESS" "Critical table data integrity test passed"
}

# Function to test Redis restore
test_redis_restore() {
    local backup_file=$1
    print_status "INFO" "Testing Redis restore from $(basename "$backup_file")..."

    # Extract Redis backup if from full backup
    local redis_backup_file="$backup_file"

    if [[ "$backup_file" =~ full_ ]]; then
        local extract_dir="${TEST_DATA_DIR}/extract_redis_${TIMESTAMP}"
        mkdir -p "$extract_dir"

        # Decrypt and extract if necessary
        if [[ "$backup_file" =~ \.enc$ ]]; then
            local decrypted_file="${extract_dir}/decrypted.tar"
            echo "$ENCRYPTION_KEY" | gpg --batch --yes --passphrase-fd 0 --decrypt "$backup_file" > "$decrypted_file"
            backup_file="$decrypted_file"
        fi

        tar -xzf "$backup_file" -C "$extract_dir"
        redis_backup_file=$(find "$extract_dir" -name "redis_*.rdb*" | head -1)

        if [ -z "$redis_backup_file" ]; then
            print_status "ERROR" "No Redis backup found in archive"
            return 1
        fi
    fi

    # Restore Redis data
    print_status "INFO" "Restoring Redis backup to test instance..."

    # Stop Redis to restore data
    docker-compose -f "$TEST_COMPOSE_FILE" exec test_redis redis-cli FLUSHALL

    # Copy backup file into container and restore
    docker cp "$redis_backup_file" restore_test_redis:/data/restore.rdb
    docker-compose -f "$TEST_COMPOSE_FILE" restart test_redis

    # Wait for Redis to restart
    sleep 10

    # Verify restore
    local key_count=$(docker-compose -f "$TEST_COMPOSE_FILE" exec -T test_redis redis-cli DBSIZE 2>/dev/null || echo "0")

    if [ "$key_count" -gt 0 ]; then
        print_status "SUCCESS" "Redis restore test passed ($key_count keys restored)"
    else
        print_status "WARNING" "Redis restore test: no keys found (may be normal if Redis was empty)"
    fi

    # Cleanup
    if [[ "$redis_backup_file" =~ extract_ ]]; then
        rm -rf "$(dirname "$redis_backup_file")"
    fi
}

# Function to test file restore
test_file_restore() {
    local backup_file=$1
    print_status "INFO" "Testing file restore from $(basename "$backup_file")..."

    local test_restore_dir="${TEST_DATA_DIR}/file_restore_${TIMESTAMP}"
    mkdir -p "$test_restore_dir"

    # Decrypt if necessary
    local restore_file="$backup_file"
    if [[ "$backup_file" =~ \.enc$ ]]; then
        restore_file="${test_restore_dir}/decrypted.tar"
        echo "$ENCRYPTION_KEY" | gpg --batch --yes --passphrase-fd 0 --decrypt "$backup_file" > "$restore_file"
    fi

    # Extract files
    if [[ "$restore_file" =~ \.gz$ ]]; then
        tar -xzf "$restore_file" -C "$test_restore_dir" 2>>"$TEST_LOG"
    else
        tar -xf "$restore_file" -C "$test_restore_dir" 2>>"$TEST_LOG"
    fi

    # Verify extracted files
    local file_count=$(find "$test_restore_dir" -type f | wc -l)
    local total_size=$(du -sh "$test_restore_dir" 2>/dev/null | cut -f1 || echo "0")

    if [ "$file_count" -gt 0 ]; then
        print_status "SUCCESS" "File restore test passed ($file_count files, $total_size total)"
    else
        print_status "ERROR" "File restore test failed (no files extracted)"
        return 1
    fi

    # Test specific file types
    test_parquet_files "$test_restore_dir"
    test_config_files "$test_restore_dir"

    # Cleanup
    if [ "$CLEANUP_AFTER_TEST" = true ]; then
        rm -rf "$test_restore_dir"
    fi
}

# Function to test parquet files
test_parquet_files() {
    local restore_dir=$1
    print_status "INFO" "Testing parquet file integrity..."

    local parquet_files=$(find "$restore_dir" -name "*.parquet" -type f)

    if [ -z "$parquet_files" ]; then
        print_status "INFO" "No parquet files found in backup"
        return
    fi

    local valid_files=0
    local invalid_files=0

    for parquet_file in $parquet_files; do
        # Test parquet file integrity using Python
        if python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_parquet('$parquet_file')
    print(f'✓ {len(df)} rows in $(basename $parquet_file)')
    sys.exit(0)
except Exception as e:
    print(f'✗ Error reading $(basename $parquet_file): {e}')
    sys.exit(1)
" 2>>"$TEST_LOG"; then
            ((valid_files++))
        else
            ((invalid_files++))
            print_status "WARNING" "Invalid parquet file: $(basename "$parquet_file")"
        fi
    done

    if [ $invalid_files -eq 0 ]; then
        print_status "SUCCESS" "Parquet file integrity test passed ($valid_files files)"
    else
        print_status "WARNING" "Parquet integrity issues: $invalid_files invalid, $valid_files valid"
    fi
}

# Function to test config files
test_config_files() {
    local restore_dir=$1
    print_status "INFO" "Testing configuration file integrity..."

    # Test YAML files
    local yaml_files=$(find "$restore_dir" -name "*.yml" -o -name "*.yaml" -type f)
    for yaml_file in $yaml_files; do
        if python3 -c "
import yaml
import sys
try:
    with open('$yaml_file', 'r') as f:
        yaml.safe_load(f)
    sys.exit(0)
except Exception as e:
    print(f'Invalid YAML: $(basename $yaml_file)')
    sys.exit(1)
" 2>>"$TEST_LOG"; then
            print_status "INFO" "Valid YAML: $(basename "$yaml_file")"
        else
            print_status "WARNING" "Invalid YAML file: $(basename "$yaml_file")"
        fi
    done

    # Test JSON files
    local json_files=$(find "$restore_dir" -name "*.json" -type f)
    for json_file in $json_files; do
        if jq . "$json_file" >/dev/null 2>&1; then
            print_status "INFO" "Valid JSON: $(basename "$json_file")"
        else
            print_status "WARNING" "Invalid JSON file: $(basename "$json_file")"
        fi
    done

    print_status "SUCCESS" "Configuration file integrity test completed"
}

# Function to test data integrity after restore
test_data_integrity() {
    if [ "$VERIFY_DATA_INTEGRITY" = false ]; then
        print_status "INFO" "Skipping data integrity verification"
        return
    fi

    print_status "INFO" "Testing data integrity after restore..."

    # Test foreign key constraints
    local constraint_violations=$(PGPASSWORD="test_password" psql \
        -h localhost \
        -p 15432 \
        -U test_trader \
        -d "$TEST_DB_NAME" \
        -t -c "
        SELECT count(*)
        FROM information_schema.table_constraints tc
        JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND NOT EXISTS (
            SELECT 1 FROM information_schema.referential_constraints rc
            WHERE rc.constraint_name = tc.constraint_name
        );" 2>/dev/null | tr -d ' ')

    if [ "$constraint_violations" -eq 0 ]; then
        print_status "SUCCESS" "Foreign key integrity check passed"
    else
        print_status "ERROR" "Foreign key integrity violations found: $constraint_violations"
        return 1
    fi

    # Test data consistency
    test_trading_data_consistency

    print_status "SUCCESS" "Data integrity verification completed"
}

# Function to test trading data consistency
test_trading_data_consistency() {
    print_status "INFO" "Testing trading data consistency..."

    # Test 1: Verify no negative quantities
    local negative_qty=$(PGPASSWORD="test_password" psql \
        -h localhost \
        -p 15432 \
        -U test_trader \
        -d "$TEST_DB_NAME" \
        -t -c "SELECT count(*) FROM trades WHERE quantity <= 0;" 2>/dev/null | tr -d ' ')

    if [ "$negative_qty" -gt 0 ]; then
        print_status "ERROR" "Found $negative_qty trades with invalid quantities"
        return 1
    fi

    # Test 2: Verify position calculations
    local position_errors=$(PGPASSWORD="test_password" psql \
        -h localhost \
        -p 15432 \
        -U test_trader \
        -d "$TEST_DB_NAME" \
        -t -c "
        SELECT count(*)
        FROM positions p
        WHERE p.shares != (
            SELECT COALESCE(SUM(CASE WHEN t.side = 'buy' THEN t.quantity ELSE -t.quantity END), 0)
            FROM trades t
            WHERE t.symbol = p.symbol AND t.account_id = p.account_id
        );" 2>/dev/null | tr -d ' ')

    if [ "$position_errors" -gt 0 ]; then
        print_status "ERROR" "Found $position_errors positions with calculation errors"
        return 1
    fi

    # Test 3: Verify account balances
    local balance_errors=$(PGPASSWORD="test_password" psql \
        -h localhost \
        -p 15432 \
        -U test_trader \
        -d "$TEST_DB_NAME" \
        -t -c "SELECT count(*) FROM accounts WHERE cash_balance < 0;" 2>/dev/null | tr -d ' ')

    if [ "$balance_errors" -gt 0 ]; then
        print_status "WARNING" "Found $balance_errors accounts with negative cash balance"
    fi

    print_status "SUCCESS" "Trading data consistency tests passed"
}

# Function to test performance after restore
test_restore_performance() {
    print_status "INFO" "Testing database performance after restore..."

    # Test query performance on restored database
    local test_queries=(
        "SELECT count(*) FROM trades WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'"
        "SELECT symbol, sum(quantity) FROM trades GROUP BY symbol ORDER BY sum(quantity) DESC LIMIT 10"
        "SELECT account_id, sum(market_value) FROM positions GROUP BY account_id"
    )

    local total_time=0

    for query in "${test_queries[@]}"; do
        local start_time=$(date +%s.%N)

        PGPASSWORD="test_password" psql \
            -h localhost \
            -p 15432 \
            -U test_trader \
            -d "$TEST_DB_NAME" \
            -c "$query" >/dev/null 2>&1

        local end_time=$(date +%s.%N)
        local query_time=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $query_time" | bc)

        print_status "INFO" "Query completed in ${query_time}s"
    done

    print_status "INFO" "Total query time: ${total_time}s"

    # Performance threshold check (adjust as needed)
    if (( $(echo "$total_time < 10.0" | bc -l) )); then
        print_status "SUCCESS" "Database performance test passed"
    else
        print_status "WARNING" "Database performance slower than expected: ${total_time}s"
    fi
}

# Function to run comprehensive test suite
run_comprehensive_test() {
    local backup_file=$1
    print_status "INFO" "Running comprehensive restore test for $(basename "$backup_file")..."

    local test_start=$(date +%s)
    local test_results=()

    # Test backup verification
    if verify_backup "$backup_file"; then
        test_results+=("backup_verification:PASS")
    else
        test_results+=("backup_verification:FAIL")
        print_status "ERROR" "Backup verification failed, skipping further tests"
        return 1
    fi

    # Determine backup type
    local backup_type="unknown"
    if [[ "$backup_file" =~ full_ ]]; then
        backup_type="full"
    elif [[ "$backup_file" =~ postgres_ ]]; then
        backup_type="database"
    elif [[ "$backup_file" =~ files_ ]]; then
        backup_type="files"
    fi

    # Test based on backup type
    case $backup_type in
        "full")
            if test_database_restore "$backup_file"; then
                test_results+=("database_restore:PASS")
            else
                test_results+=("database_restore:FAIL")
            fi

            if test_redis_restore "$backup_file"; then
                test_results+=("redis_restore:PASS")
            else
                test_results+=("redis_restore:FAIL")
            fi

            if test_file_restore "$backup_file"; then
                test_results+=("file_restore:PASS")
            else
                test_results+=("file_restore:FAIL")
            fi
            ;;
        "database")
            if test_database_restore "$backup_file"; then
                test_results+=("database_restore:PASS")
            else
                test_results+=("database_restore:FAIL")
            fi
            ;;
        "files")
            if test_file_restore "$backup_file"; then
                test_results+=("file_restore:PASS")
            else
                test_results+=("file_restore:FAIL")
            fi
            ;;
    esac

    # Test data integrity
    if test_data_integrity; then
        test_results+=("data_integrity:PASS")
    else
        test_results+=("data_integrity:FAIL")
    fi

    # Test performance
    if test_restore_performance; then
        test_results+=("performance:PASS")
    else
        test_results+=("performance:FAIL")
    fi

    local test_end=$(date +%s)
    local test_duration=$((test_end - test_start))

    # Generate test report
    generate_test_report "$backup_file" "${test_results[@]}" "$test_duration"

    # Check overall results
    local failed_tests=$(printf '%s\n' "${test_results[@]}" | grep -c ":FAIL" || echo "0")

    if [ "$failed_tests" -eq 0 ]; then
        print_status "SUCCESS" "All restore tests passed for $(basename "$backup_file")"
        return 0
    else
        print_status "ERROR" "$failed_tests restore tests failed for $(basename "$backup_file")"
        return 1
    fi
}

# Function to generate test report
generate_test_report() {
    local backup_file=$1
    shift
    local test_results=("$@")
    local duration=${test_results[-1]}
    unset 'test_results[-1]'

    local report_file="${PROJECT_ROOT}/logs/restore_test_report_${TIMESTAMP}.json"

    cat > "$report_file" << EOF
{
    "test_id": "$TIMESTAMP",
    "environment": "$ENVIRONMENT",
    "backup_file": "$backup_file",
    "test_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "test_duration_seconds": $duration,
    "results": {
EOF

    local first=true
    for result in "${test_results[@]}"; do
        local test_name=$(echo "$result" | cut -d':' -f1)
        local test_status=$(echo "$result" | cut -d':' -f2)

        if [ "$first" = false ]; then
            echo "," >> "$report_file"
        fi
        echo "        \"$test_name\": \"$test_status\"" >> "$report_file"
        first=false
    done

    cat >> "$report_file" << EOF
    },
    "summary": {
        "total_tests": ${#test_results[@]},
        "passed": $(printf '%s\n' "${test_results[@]}" | grep -c ":PASS" || echo "0"),
        "failed": $(printf '%s\n' "${test_results[@]}" | grep -c ":FAIL" || echo "0")
    },
    "metadata": {
        "backup_size": "$(du -h "$backup_file" 2>/dev/null | cut -f1 || echo "unknown")",
        "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    }
}
EOF

    print_status "SUCCESS" "Test report generated: $report_file"
}

# Function to test all available backups
test_all_backups() {
    print_status "INFO" "Testing all available backups..."

    local backup_files=$(find "${PROJECT_ROOT}/backups" -name "*.tar*" -o -name "*.sql*" | sort -r)

    if [ -z "$backup_files" ]; then
        print_status "WARNING" "No backup files found to test"
        return
    fi

    local total_tests=0
    local passed_tests=0
    local failed_tests=0

    for backup_file in $backup_files; do
        print_status "INFO" "Testing backup: $(basename "$backup_file")"
        ((total_tests++))

        if run_comprehensive_test "$backup_file"; then
            ((passed_tests++))
        else
            ((failed_tests++))
        fi

        # Reset test environment between tests
        cleanup_test_environment
        start_test_containers
    done

    print_status "INFO" "Backup testing summary: $passed_tests/$total_tests passed"

    if [ $failed_tests -eq 0 ]; then
        print_status "SUCCESS" "All backup tests passed"
    else
        print_status "WARNING" "$failed_tests backup tests failed"
    fi
}

# Function to cleanup test environment
cleanup_test_environment() {
    if [ "$CLEANUP_AFTER_TEST" = false ]; then
        print_status "INFO" "Skipping test environment cleanup"
        return
    fi

    print_status "INFO" "Cleaning up test environment..."

    # Stop and remove test containers
    if [ -f "$TEST_COMPOSE_FILE" ]; then
        docker-compose -f "$TEST_COMPOSE_FILE" down -v 2>/dev/null || true
        rm -f "$TEST_COMPOSE_FILE"
    fi

    # Remove test data directory
    rm -rf "$TEST_DATA_DIR"

    # Remove test images if any
    docker images --format "{{.Repository}}:{{.Tag}}" | grep "restore_test" | xargs -r docker rmi || true

    print_status "SUCCESS" "Test environment cleanup completed"
}

# Function for daily automated testing
daily_restore_test() {
    print_status "INFO" "Running daily automated restore test..."

    # Find latest backup
    local latest_backup=$(find "${PROJECT_ROOT}/backups" -name "full_*.tar*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$latest_backup" ]; then
        print_status "ERROR" "No backups found for daily test"
        exit 1
    fi

    # Test latest backup
    run_comprehensive_test "$latest_backup"

    # Generate daily report
    generate_daily_test_report
}

# Function to generate daily test report
generate_daily_test_report() {
    local report_file="${PROJECT_ROOT}/logs/daily_restore_test_$(date +%Y_%m_%d).log"

    cat > "$report_file" << EOF
Daily Restore Test Report
========================
Date: $(date)
Environment: $ENVIRONMENT
Test ID: $TIMESTAMP

Test Results Summary:
$(grep -E "\[(SUCCESS|ERROR|WARNING)\]" "$TEST_LOG" | tail -20)

Backup Files Tested:
$(find "${PROJECT_ROOT}/backups" -name "*${TIMESTAMP}*" -type f | while read -r file; do
    echo "  - $(basename "$file") ($(du -h "$file" | cut -f1))"
done)

System Status:
- Database Connection: $(PGPASSWORD="test_password" psql -h localhost -p 15432 -U test_trader -d "$TEST_DB_NAME" -c "SELECT 'OK'" 2>/dev/null || echo "FAILED")
- Redis Connection: $(redis-cli -p 16379 ping 2>/dev/null || echo "FAILED")

Recommendations:
- Review any failed tests above
- Verify backup retention policy
- Check storage capacity
- Update backup schedules if needed
EOF

    print_status "SUCCESS" "Daily test report generated: $report_file"
}

# Function for weekly comprehensive testing
+weekly_restore_test() {
+    print_status "INFO" "Running weekly comprehensive restore test..."
+
+    # Test multiple backup types and ages
+    local backup_patterns=(
+        "full_*"
+        "database_*"
+        "files_*"
+        "incremental_*"
+    )
+
+    local weekly_results=()
+
+    for pattern in "${backup_patterns[@]}"; do
+        local backups=$(find "${PROJECT_ROOT}/backups" -name "$pattern" -type f | sort -r | head -3)
+
+        for backup_file in $backups; do
+            print_status "INFO" "Testing weekly backup: $(basename "$backup_file")"
+
+            if run_comprehensive_test "$backup_file"; then
+                weekly_results+=("$(basename "$backup_file"):PASS")
+            else
+                weekly_results+=("$(basename "$backup_file"):FAIL")
+            fi
+
+            # Reset environment
+            cleanup_test_environment
+            start_test_containers
+        done
+    done
+
+    # Generate weekly report
+    generate_weekly_test_report "${weekly_results[@]}"
+}
+
+# Function to generate weekly test report
++generate_weekly_test_report() {
+    local results=("$@")
+    local report_file="${PROJECT_ROOT}/logs/weekly_restore_test_$(date +%Y_week_%U).json"
+
+    cat > "$report_file" << EOF
+{
+    "test_type": "weekly_comprehensive",
+    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
+    "environment": "$ENVIRONMENT",
+    "results": [
+EOF
+
+    local first=true
+    for result in "${results[@]}"; do
+        local backup_name=$(echo "$result" | cut -d':' -f1)
+        local status=$(echo "$result" | cut -d':' -f2)
+
+        if [ "$first" = false ]; then
+            echo "," >> "$report_file"
+        fi
+
+        cat >> "$report_file" << EOF
+        {
+            "backup_file": "$backup_name",
+            "status": "$status",
+            "test_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
+        }
+EOF
+        first=false
+    done
+
+    cat >> "$report_file" << EOF
+    ],
+    "summary": {
+        "total_tests": ${#results[@]},
+        "passed": $(printf '%s\n' "${results[@]}" | grep -c ":PASS" || echo "0"),
+        "failed": $(printf '%s\n' "${results[@]}" | grep -c ":FAIL" || echo "0")
+    }
+}
+EOF
+
+    print_status "SUCCESS" "Weekly test report generated: $report_file"
+}
+
+# Parse command line arguments
++parse_arguments() {
+    while [[ $# -gt 0 ]]; do
+        case $1 in
+            development|staging|production)
+                ENVIRONMENT="$1"
+                shift
+                ;;
+            --backup-file)
+                BACKUP_FILE="$2"
+                shift 2
+                ;;
+            --no-cleanup)
+                CLEANUP_AFTER_TEST=false
+                shift
+                ;;
+            --sequential)
+                PARALLEL_TESTING=false
+                shift
+                ;;
+            --timeout)
+                TEST_TIMEOUT="$2"
+                shift 2
+                ;;
+            --no-create-backup)
+                CREATE_TEST_BACKUP=false
+                shift
+                ;;
+            --skip-integrity)
+                VERIFY_DATA_INTEGRITY=false
+                shift
+                ;;
+            all)
+                test_all_backups
+                exit 0
+                ;;
+            latest)
+                BACKUP_FILE=$(find "${PROJECT_ROOT}/backups" -name "full_*.tar*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
+                shift
+                ;;
+            daily)
+                daily_restore_test
+                exit 0
+                ;;
+            weekly)
+                weekly_restore_test
+                exit 0
+                ;;
+            report)
+                generate_daily_test_report
+                exit 0
+                ;;
+            --help|-h)
+                show_usage
+                exit 0
+                ;;
+            *)
+                print_status "ERROR" "Unknown option: $1"
+                show_usage
+                exit 1
+                ;;
+        esac
+    done
+}
+
+# Main testing function
++main_test() {
+    print_status "INFO" "Starting automated restore testing"
+    print_status "INFO" "Test ID: $TIMESTAMP"
+    print_status "INFO" "Environment: $ENVIRONMENT"
+    print_status "INFO" "Log file: $TEST_LOG"
+
+    # Setup test environment
+    setup_test_environment
+    start_test_containers
+
+    # Set timeout for entire test
+    {
+        sleep $TEST_TIMEOUT
+        print_status "ERROR" "Test timeout reached ($TEST_TIMEOUT seconds)"
+        cleanup_test_environment
+        exit 1
+    } &
+    local timeout_pid=$!
+
+    # Create test backup if requested
+    if [ "$CREATE_TEST_BACKUP" = true ] && [ -z "$BACKUP_FILE" ]; then
+        create_test_backup
+    fi
+
+    # Determine backup file to test
+    if [ -z "$BACKUP_FILE" ]; then
+        BACKUP_FILE=$(find "${PROJECT_ROOT}/backups" -name "full_*.tar*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
+
+        if [ -z "$BACKUP_FILE" ]; then
+            print_status "ERROR" "No backup file found to test"
+            cleanup_test_environment
+            exit 1
+        fi
+    fi
+
+    # Run comprehensive test
+    local test_success=false
+    if run_comprehensive_test "$BACKUP_FILE"; then
+        test_success=true
+    fi
+
+    # Kill timeout process
+    kill $timeout_pid 2>/dev/null || true
+
+    # Cleanup
+    cleanup_test_environment
+
+    if [ "$test_success" = true ]; then
+        print_status "SUCCESS" "🎉 Restore testing completed successfully!"
+    else
+        print_status "ERROR" "❌ Restore testing failed!"
+        exit 1
+    fi
+
+    echo
+    echo "📊 Test Summary:"
+    echo "   Environment: $ENVIRONMENT"
+    echo "   Backup File: $(basename "$BACKUP_FILE")"
+    echo "   Test ID: $TIMESTAMP"
+    echo "   Log: $TEST_LOG"
+    echo
+    echo "🔍 Next Steps:"
+    echo "   1. Review test logs for any warnings"
+    echo "   2. Verify backup schedules are working"
+    echo "   3. Check backup retention policies"
+    echo "   4. Update disaster recovery procedures"
+    echo
+}
+
+# Main execution
++main() {
+    # Create logs directory
+    mkdir -p "${PROJECT_ROOT}/logs"
+
+    # Parse arguments
+    parse_arguments "$@"
+
+    # Handle errors
+    trap 'print_status "ERROR" "Restore test failed"; cleanup_test_environment; exit 1' ERR
+
+    # Execute main test
+    main_test
+}
+
+# Run main function with all arguments
++main "$@"
