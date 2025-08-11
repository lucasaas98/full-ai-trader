#!/bin/bash

# AI Trading System Security Audit Script
# Comprehensive security auditing with automated remediation
# Usage: ./audit_security.sh [options]

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
AUDIT_LOG="${PROJECT_ROOT}/logs/security/audit_${TIMESTAMP}.log"
VULNERABILITY_REPORT="${PROJECT_ROOT}/logs/security/vulnerability_report_${TIMESTAMP}.json"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs/security"

# Default configuration
AUDIT_TYPE="full"
ENVIRONMENT="production"
COMPOSE_FILE="docker-compose.yml"
DRY_RUN=false
AUTO_REMEDIATE=false
GENERATE_REPORT=true
CHECK_VULNERABILITIES=true
CHECK_COMPLIANCE=true
CHECK_CONFIGURATIONS=true
CHECK_NETWORK_SECURITY=true
CHECK_ACCESS_CONTROLS=true
SEVERITY_THRESHOLD="medium"

# Security check categories
SECURITY_CATEGORIES=(
    "authentication"
    "authorization"
    "encryption"
    "network"
    "containers"
    "secrets"
    "logging"
    "compliance"
)

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$AUDIT_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$AUDIT_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" | tee -a "$AUDIT_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$AUDIT_LOG"
            ;;
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message" | tee -a "$AUDIT_LOG"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Security Audit Script

Usage: $0 [OPTIONS]

AUDIT TYPES:
    full                    Complete security audit (default)
    quick                   Quick security scan
    compliance              Compliance-focused audit
    vulnerability           Vulnerability assessment only
    configuration           Configuration security review

OPTIONS:
    --type TYPE             Audit type (full/quick/compliance/vulnerability/configuration)
    --env ENV               Environment (development/staging/production)
    --compose-file FILE     Docker compose file to use
    --dry-run               Show what would be checked without executing
    --auto-remediate        Automatically fix issues where possible
    --no-report             Skip generating detailed report
    --skip-vulns            Skip vulnerability scanning
    --skip-compliance       Skip compliance checks
    --skip-configs          Skip configuration checks
    --skip-network          Skip network security checks
    --skip-access           Skip access control checks
    --severity LEVEL        Minimum severity to report (low/medium/high/critical)
    --output FORMAT         Output format (text/json/html)
    --help                  Show this help message

EXAMPLES:
    $0 --type full --env production --auto-remediate
    $0 --type quick --severity high
    $0 --type vulnerability --output json
    $0 --type compliance --env production --dry-run

SECURITY AREAS CHECKED:
    • Authentication mechanisms
    • Authorization controls
    • Encryption implementation
    • Network security
    • Container security
    • Secret management
    • Audit logging
    • Compliance requirements
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                AUDIT_TYPE="$2"
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
            --auto-remediate)
                AUTO_REMEDIATE=true
                shift
                ;;
            --no-report)
                GENERATE_REPORT=false
                shift
                ;;
            --skip-vulns)
                CHECK_VULNERABILITIES=false
                shift
                ;;
            --skip-compliance)
                CHECK_COMPLIANCE=false
                shift
                ;;
            --skip-configs)
                CHECK_CONFIGURATIONS=false
                shift
                ;;
            --skip-network)
                CHECK_NETWORK_SECURITY=false
                shift
                ;;
            --skip-access)
                CHECK_ACCESS_CONTROLS=false
                shift
                ;;
            --severity)
                SEVERITY_THRESHOLD="$2"
                shift 2
                ;;
            --output)
                OUTPUT_FORMAT="$2"
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

# Function to initialize audit report
initialize_audit_report() {
    cat > "$VULNERABILITY_REPORT" << EOF
{
    "audit_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "audit_type": "$AUDIT_TYPE",
    "environment": "$ENVIRONMENT",
    "host": "$(hostname)",
    "auditor": "$(whoami)",
    "findings": [],
    "summary": {
        "total_checks": 0,
        "passed": 0,
        "warnings": 0,
        "errors": 0,
        "critical": 0
    },
    "remediation_actions": []
}
EOF
}

# Function to add finding to report
add_finding() {
    local category=$1
    local severity=$2
    local title=$3
    local description=$4
    local remediation=${5:-"Manual review required"}

    # Create finding JSON
    local finding=$(cat << EOF
{
    "category": "$category",
    "severity": "$severity",
    "title": "$title",
    "description": "$description",
    "remediation": "$remediation",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
)

    # Add to report (simplified - in production would use jq)
    echo "Security Finding: [$severity] $category - $title" >> "$AUDIT_LOG"
    echo "  Description: $description" >> "$AUDIT_LOG"
    echo "  Remediation: $remediation" >> "$AUDIT_LOG"
    echo "" >> "$AUDIT_LOG"
}

# Function to check authentication security
check_authentication_security() {
    print_status "INFO" "Checking authentication security..."

    local findings=0

    # Check for default passwords
    if [[ -f "${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}" ]]; then
        if grep -q "password.*admin\|password.*123\|password.*password" "${PROJECT_ROOT}/config/environments/.env.${ENVIRONMENT}" 2>/dev/null; then
            add_finding "authentication" "high" "Default passwords detected" "Default or weak passwords found in environment configuration" "Change all default passwords immediately"
            ((findings++))
        fi
    fi

    # Check JWT secret strength
    if [[ -f "${PROJECT_ROOT}/data/secrets/api_keys.json" ]]; then
        local jwt_secret_length=$(jq -r '.api_secret' "${PROJECT_ROOT}/data/secrets/api_keys.json" 2>/dev/null | wc -c || echo "0")
        if [[ $jwt_secret_length -lt 32 ]]; then
            add_finding "authentication" "high" "Weak JWT secret" "JWT secret is too short (${jwt_secret_length} chars)" "Generate stronger JWT secret (minimum 32 characters)"
            ((findings++))
        fi
    fi

    # Check for unencrypted API endpoints
    cd "$PROJECT_ROOT"
    if docker-compose -f "$COMPOSE_FILE" ps | grep -E ":80[0-9][0-9]->80[0-9][0-9]/tcp" | grep -v "127.0.0.1"; then
        add_finding "authentication" "medium" "Unencrypted API endpoints" "API endpoints exposed without HTTPS" "Configure SSL/TLS for all API endpoints"
        ((findings++))
    fi

    # Check session configuration
    if grep -r "session_timeout\|SESSION_TIMEOUT" "${PROJECT_ROOT}/config" 2>/dev/null | grep -q "86400\|3600[0-9]"; then
        add_finding "authentication" "medium" "Long session timeout" "Session timeout may be too long" "Review and reduce session timeout values"
        ((findings++))
    fi

    print_status "SUCCESS" "Authentication security check completed ($findings findings)"
    return $findings
}

# Function to check authorization controls
check_authorization_controls() {
    print_status "INFO" "Checking authorization controls..."

    local findings=0

    # Check for role-based access controls
    if ! grep -r "RBAC\|role.*permission\|user.*role" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/services" 2>/dev/null; then
        add_finding "authorization" "medium" "Missing RBAC implementation" "Role-based access controls not detected" "Implement proper RBAC system"
        ((findings++))
    fi

    # Check for admin endpoints without authentication
    if grep -r "admin\|/api/admin" "${PROJECT_ROOT}" 2>/dev/null | grep -v "auth\|login\|token"; then
        add_finding "authorization" "high" "Unprotected admin endpoints" "Admin endpoints may lack proper authentication" "Secure all administrative endpoints"
        ((findings++))
    fi

    # Check Docker container privileges
    local privileged_containers=$(docker ps --format "table {{.Names}}" --filter "status=running" | tail -n +2 | xargs -I {} docker inspect {} --format '{{.Name}}: {{.HostConfig.Privileged}}' 2>/dev/null | grep "true" | wc -l || echo "0")

    if [[ $privileged_containers -gt 1 ]]; then  # Allow one for monitoring
        add_finding "authorization" "high" "Excessive privileged containers" "$privileged_containers containers running in privileged mode" "Review and remove unnecessary privileged access"
        ((findings++))
    fi

    # Check file permissions
    local world_writable=$(find "${PROJECT_ROOT}" -type f -perm -002 2>/dev/null | wc -l || echo "0")
    if [[ $world_writable -gt 0 ]]; then
        add_finding "authorization" "medium" "World-writable files detected" "$world_writable files are world-writable" "Fix file permissions (chmod 644 or 600)"
        ((findings++))
    fi

    print_status "SUCCESS" "Authorization controls check completed ($findings findings)"
    return $findings
}

# Function to check encryption implementation
check_encryption_security() {
    print_status "INFO" "Checking encryption implementation..."

    local findings=0

    # Check SSL/TLS certificates
    if [[ ! -d "${PROJECT_ROOT}/config/ssl" ]] || [[ ! -f "${PROJECT_ROOT}/config/ssl/trading-system.crt" ]]; then
        add_finding "encryption" "high" "Missing SSL certificates" "SSL/TLS certificates not found" "Generate and configure SSL certificates"
        ((findings++))
    else
        # Check certificate expiration
        local cert_file="${PROJECT_ROOT}/config/ssl/trading-system.crt"
        local expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" 2>/dev/null | cut -d= -f2 || echo "")
        if [[ -n "$expiry_date" ]]; then
            local expiry_timestamp=$(date -d "$expiry_date" +%s 2>/dev/null || echo "0")
            local current_timestamp=$(date +%s)
            local days_until_expiry=$(( (expiry_timestamp - current_timestamp) / 86400 ))

            if [[ $days_until_expiry -lt 30 ]]; then
                add_finding "encryption" "high" "SSL certificate expiring soon" "Certificate expires in $days_until_expiry days" "Renew SSL certificate"
                ((findings++))
            fi
        fi
    fi

    # Check for unencrypted database connections
    if grep -r "sslmode.*disable\|ssl.*false" "${PROJECT_ROOT}/config" 2>/dev/null; then
        add_finding "encryption" "high" "Unencrypted database connections" "Database connections not using SSL" "Enable SSL for database connections"
        ((findings++))
    fi

    # Check for hardcoded secrets
    local hardcoded_secrets=$(grep -r -E "password.*=.*['\"][^'\"]{3,}['\"]|secret.*=.*['\"][^'\"]{3,}['\"]|key.*=.*['\"][^'\"]{10,}['\"]" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/services" 2>/dev/null | wc -l || echo "0")
    if [[ $hardcoded_secrets -gt 0 ]]; then
        add_finding "encryption" "critical" "Hardcoded secrets detected" "$hardcoded_secrets instances of hardcoded secrets found" "Move all secrets to secure configuration"
        ((findings++))
    fi

    # Check encryption key strength
    if [[ -f "${PROJECT_ROOT}/data/secrets/db_encryption.key" ]]; then
        local key_length=$(wc -c < "${PROJECT_ROOT}/data/secrets/db_encryption.key")
        if [[ $key_length -lt 32 ]]; then
            add_finding "encryption" "high" "Weak encryption key" "Database encryption key too short (${key_length} chars)" "Generate stronger encryption key"
            ((findings++))
        fi
    else
        add_finding "encryption" "high" "Missing encryption key" "Database encryption key not found" "Generate database encryption key"
        ((findings++))
    fi

    print_status "SUCCESS" "Encryption security check completed ($findings findings)"
    return $findings
}

# Function to check network security
check_network_security() {
    print_status "INFO" "Checking network security..."

    local findings=0

    # Check for exposed ports
    local exposed_ports=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E "0\.0\.0\.0:" | wc -l || echo "0")
    if [[ $exposed_ports -gt 2 ]]; then  # Allow for web UI and API
        add_finding "network" "medium" "Multiple exposed ports" "$exposed_ports services exposed to all interfaces" "Review and restrict port exposure"
        ((findings++))
    fi

    # Check firewall status
    if command -v ufw &>/dev/null; then
        if ! ufw status | grep -q "Status: active"; then
            add_finding "network" "high" "Firewall not active" "UFW firewall is not enabled" "Enable and configure firewall"
            ((findings++))
        fi
    elif command -v iptables &>/dev/null; then
        local iptables_rules=$(iptables -L | wc -l)
        if [[ $iptables_rules -lt 10 ]]; then
            add_finding "network" "medium" "Minimal firewall rules" "Very few iptables rules configured" "Configure comprehensive firewall rules"
            ((findings++))
        fi
    else
        add_finding "network" "high" "No firewall detected" "No firewall system found" "Install and configure firewall"
        ((findings++))
    fi

    # Check for default Docker bridge network usage
    local bridge_usage=$(docker network ls | grep bridge | wc -l || echo "0")
    if [[ $bridge_usage -gt 1 ]]; then
        add_finding "network" "low" "Default bridge network usage" "Services may be using default Docker bridge" "Use custom networks for better isolation"
        ((findings++))
    fi

    # Check network encryption
    if ! grep -r "tls\|ssl" "${PROJECT_ROOT}/monitoring/prometheus" "${PROJECT_ROOT}/monitoring/grafana" 2>/dev/null; then
        add_finding "network" "medium" "Monitoring traffic not encrypted" "Monitoring services may not use encryption" "Enable TLS for monitoring services"
        ((findings++))
    fi

    print_status "SUCCESS" "Network security check completed ($findings findings)"
    return $findings
}

# Function to check container security
check_container_security() {
    print_status "INFO" "Checking container security..."

    local findings=0
    cd "$PROJECT_ROOT"

    # Check for containers running as root
    local root_containers=()
    for container in $(docker-compose -f "$COMPOSE_FILE" ps -q 2>/dev/null); do
        local user=$(docker inspect "$container" --format '{{.Config.User}}' 2>/dev/null || echo "")
        local name=$(docker inspect "$container" --format '{{.Name}}' 2>/dev/null | sed 's/^[/]//')

        if [[ -z "$user" ]] || [[ "$user" == "root" ]] || [[ "$user" == "0" ]]; then
            # Skip containers that legitimately need root (like monitoring)
            if [[ ! "$name" =~ (cadvisor|node_exporter|postgres) ]]; then
                root_containers+=("$name")
            fi
        fi
    done

    if [[ ${#root_containers[@]} -gt 0 ]]; then
        add_finding "containers" "medium" "Containers running as root" "${#root_containers[@]} containers running as root: ${root_containers[*]}" "Configure non-root users for containers"
        ((findings++))
    fi

    # Check for containers with host network mode
    local host_network_containers=$(docker ps --format "table {{.Names}}" --filter "status=running" | tail -n +2 | xargs -I {} docker inspect {} --format '{{.Name}}: {{.HostConfig.NetworkMode}}' 2>/dev/null | grep "host" | wc -l || echo "0")

    if [[ $host_network_containers -gt 0 ]]; then
        add_finding "containers" "high" "Host network mode usage" "$host_network_containers containers using host network mode" "Use bridge networks instead of host mode"
        ((findings++))
    fi

    # Check for bind mounts to sensitive directories
    local sensitive_mounts=$(docker ps --format "table {{.Names}}" --filter "status=running" | tail -n +2 | xargs -I {} docker inspect {} --format '{{.Name}}: {{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}' 2>/dev/null | grep -E "/etc|/var|/usr|/root" | wc -l || echo "0")

    if [[ $sensitive_mounts -gt 2 ]]; then  # Allow some for legitimate monitoring
        add_finding "containers" "medium" "Sensitive directory mounts" "$sensitive_mounts containers mounting sensitive host directories" "Review and minimize host directory mounts"
        ((findings++))
    fi

    # Check container image vulnerabilities
    if command -v trivy &>/dev/null && [[ "$CHECK_VULNERABILITIES" == "true" ]]; then
        print_status "INFO" "Scanning container images for vulnerabilities..."

        local image_vulns=0
        for image in $(docker-compose -f "$COMPOSE_FILE" config --services 2>/dev/null); do
            local image_name=$(docker-compose -f "$COMPOSE_FILE" images "$image" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null || echo "")
            if [[ -n "$image_name" ]] && [[ "$image_name" != "<none>:<none>" ]]; then
                local vuln_count=$(trivy image --severity HIGH,CRITICAL --format json "$image_name" 2>/dev/null | jq '.Results[]?.Vulnerabilities? // [] | length' | awk '{sum+=$1} END {print sum+0}' || echo "0")
                if [[ $vuln_count -gt 0 ]]; then
                    add_finding "containers" "high" "Container vulnerabilities in $image" "$vuln_count high/critical vulnerabilities found" "Update container image or apply security patches"
                    ((image_vulns++))
                fi
            fi
        done
        findings=$((findings + image_vulns))
    fi

    # Check for latest base images
    local outdated_images=$(docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}" | grep -E "(python|node|alpine|ubuntu)" | awk '{print $3}' | grep -c "months\|year" || echo "0")
    if [[ $outdated_images -gt 0 ]]; then
        add_finding "containers" "low" "Outdated base images" "$outdated_images base images may be outdated" "Update to latest base images"
        ((findings++))
    fi

    print_status "SUCCESS" "Container security check completed ($findings findings)"
    return $findings
}

# Function to check secret management
check_secret_management() {
    print_status "INFO" "Checking secret management..."

    local findings=0

    # Check secret file permissions
    if [[ -d "${PROJECT_ROOT}/data/secrets" ]]; then
        local secret_files=$(find "${PROJECT_ROOT}/data/secrets" -type f -not -perm 600 2>/dev/null | wc -l || echo "0")
        if [[ $secret_files -gt 0 ]]; then
            add_finding "secrets" "high" "Insecure secret file permissions" "$secret_files secret files with incorrect permissions" "Set permissions to 600 (owner read/write only)"
            ((findings++))
        fi

        # Check secret directory permissions
        local secret_dir_perms=$(stat -c "%a" "${PROJECT_ROOT}/data/secrets" 2>/dev/null || echo "755")
        if [[ "$secret_dir_perms" != "700" ]]; then
            add_finding "secrets" "medium" "Insecure secret directory permissions" "Secret directory permissions: $secret_dir_perms" "Set directory permissions to 700"
            ((findings++))
        fi
    else
        add_finding "secrets" "high" "Missing secrets directory" "Secrets directory not found" "Create secure secrets directory"
        ((findings++))
    fi

    # Check for secrets in environment files
    if find "${PROJECT_ROOT}/config" -name "*.env*" -exec grep -l "password\|secret\|key" {} \; 2>/dev/null | grep -v ".example\|.template"; then
        add_finding "secrets" "high" "Secrets in environment files" "Secrets found in environment configuration files" "Move secrets to secure secret management system"
        ((findings++))
    fi

    # Check for secrets in Docker Compose files
    if grep -E "password.*:|secret.*:|key.*:" "${PROJECT_ROOT}"/docker-compose*.yml 2>/dev/null; then
        add_finding "secrets" "high" "Secrets in Docker Compose" "Secrets hardcoded in Docker Compose files" "Use external secret management or environment variables"
        ((findings++))
    fi

    # Check secret rotation
    if [[ -f "${PROJECT_ROOT}/data/secrets/api_keys.json" ]]; then
        local secret_age=$(stat -c %Y "${PROJECT_ROOT}/data/secrets/api_keys.json" 2>/dev/null || echo "0")
        local current_time=$(date +%s)
        local age_days=$(( (current_time - secret_age) / 86400 ))

        if [[ $age_days -gt 90 ]]; then
            add_finding "secrets" "medium" "Secrets not rotated" "Secrets are $age_days days old" "Implement regular secret rotation"
            ((findings++))
        fi
    fi

    print_status "SUCCESS" "Secret management check completed ($findings findings)"
    return $findings
}

# Function to check audit logging
check_audit_logging() {
    print_status "INFO" "Checking audit logging..."

    local findings=0

    # Check if audit logging is enabled
    if [[ ! -d "${PROJECT_ROOT}/logs/security" ]]; then
        add_finding "logging" "medium" "Security logging directory missing" "No security-specific logging directory found" "Create security logging directory"
        ((findings++))
    fi

    # Check log file permissions
    local insecure_logs=$(find "${PROJECT_ROOT}/logs" -type f -perm /044 2>/dev/null | wc -l || echo "0")
    if [[ $insecure_logs -gt 0 ]]; then
        add_finding "logging" "medium" "Insecure log file permissions" "$insecure_logs log files have world-readable permissions" "Restrict log file permissions"
        ((findings++))
    fi

    # Check for log tampering protection
    if ! grep -r "log.*integrity\|immutable.*log" "${PROJECT_ROOT}/config" 2>/dev/null; then
        add_finding "logging" "low" "No log integrity protection" "Log files lack tampering protection" "Implement log integrity verification"
        ((findings++))
    fi

    # Check log retention policy
    local old_logs=$(find "${PROJECT_ROOT}/logs" -name "*.log" -mtime +30 2>/dev/null | wc -l || echo "0")
    if [[ $old_logs -gt 100 ]]; then
        add_finding "logging" "low" "Excessive log retention" "$old_logs old log files found" "Implement automated log cleanup"
        ((findings++))
    fi

    # Check for sensitive data in logs
    local sensitive_in_logs=$(grep -r -i "password\|secret\|key\|token" "${PROJECT_ROOT}/logs" 2>/dev/null | head -5 | wc -l || echo "0")
    if [[ $sensitive_in_logs -gt 0 ]]; then
        add_finding "logging" "critical" "Sensitive data in logs" "Sensitive information found in log files" "Implement log sanitization immediately"
        ((findings++))
    fi

    print_status "SUCCESS" "Audit logging check completed ($findings findings)"
    return $findings
}

# Function to check compliance requirements
check_compliance() {
    print_status "INFO" "Checking compliance requirements..."

    local findings=0

    # Check data retention policies
    if [[ ! -f "${PROJECT_ROOT}/docs/DATA_RETENTION_POLICY.md" ]]; then
        add_finding "compliance" "medium" "Missing data retention policy" "No documented data retention policy found" "Create and document data retention policy"
        ((findings++))
    fi

    # Check audit trail completeness
    if ! grep -r "audit.*trail\|audit.*log" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/services" 2>/dev/null; then
        add_finding "compliance" "medium" "Incomplete audit trails" "Audit trail implementation not detected" "Implement comprehensive audit logging"
        ((findings++))
    fi

    # Check for data classification
    if ! grep -r "data.*classification\|data.*sensitivity" "${PROJECT_ROOT}/docs" 2>/dev/null; then
        add_finding "compliance" "low" "Missing data classification" "No data classification documentation found" "Implement data classification system"
        ((findings++))
    fi

    # Check backup encryption
    local unencrypted_backups=$(find "${PROJECT_ROOT}/data/backups" -name "*.tar.gz" -exec file {} \; 2>/dev/null | grep -v "encrypted\|gpg" | wc -l || echo "0")
    if [[ $unencrypted_backups -gt 0 ]]; then
        add_finding "compliance" "high" "Unencrypted backups" "$unencrypted_backups backup files are not encrypted" "Encrypt all backup files"
        ((findings++))
    fi

    # Check for incident response plan
    if [[ ! -f "${PROJECT_ROOT}/docs/INCIDENT_RESPONSE.md" ]]; then
        add_finding "compliance" "medium" "Missing incident response plan" "No incident response documentation found" "Create incident response plan"
        ((findings++))
    fi

    print_status "SUCCESS" "Compliance check completed ($findings findings)"
    return $findings
}

# Function to implement rate limiting
implement_rate_limiting() {
    print_status "INFO" "Implementing rate limiting..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "INFO" "[DRY RUN] Would implement rate limiting"
        return 0
    fi

    # Create rate limiting configuration
    local rate_limit_config="${PROJECT_ROOT}/config/security/rate_limiting.yml"
    mkdir -p "${PROJECT_ROOT}/config/security"

    cat > "$rate_limit_config" << EOF
# Rate Limiting Configuration - Generated $(date)
# Environment: $ENVIRONMENT

# Global rate limits
global:
  requests_per_minute: 1000
  requests_per_hour: 10000
  burst_limit: 100

# Service-specific rate limits
services:
  data_collector:
    requests_per_minute: 500
    requests_per_hour: 5000
    burst_limit: 50
    whitelist_ips:
      - "127.0.0.1"
      - "::1"

  strategy_engine:
    requests_per_minute: 200
    requests_per_hour: 2000
    burst_limit: 20

  risk_manager:
    requests_per_minute: 300
    requests_per_hour: 3000
    burst_limit: 30

  trade_executor:
    requests_per_minute: 100
    requests_per_hour: 1000
    burst_limit: 10

  export_service:
    requests_per_minute: 50
    requests_per_hour: 200
    burst_limit: 5

# Rate limiting by endpoint
endpoints:
  "/api/auth/login":
    requests_per_minute: 10
    requests_per_hour: 60
    burst_limit: 3

  "/api/trading/execute":
    requests_per_minute: 50
    requests_per_hour: 500
    burst_limit: 5

  "/api/data/historical":
    requests_per_minute: 100
    requests_per_hour: 1000
    burst_limit: 20

# IP-based restrictions
ip_restrictions:
  blacklist: []
  whitelist: []
  auto_ban:
    enabled: true
    threshold: 1000  # requests per minute
    ban_duration: 3600  # seconds

# Geo-blocking (if required)
geo_restrictions:
  enabled: false
  allowed_countries: []
  blocked_countries: []
EOF

    # Create rate limiting middleware implementation
    cat > "${PROJECT_ROOT}/shared/middleware/rate_limiter.py" << 'EOF'
"""
Rate Limiting Middleware for AI Trading System
Implements configurable rate limiting with Redis backend
"""

import time
import json
import redis
import yaml
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RateLimitConfig:
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    whitelist_ips: list = None


class RateLimiter:
    def __init__(self
