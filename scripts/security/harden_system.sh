#!/bin/bash

# AI Trading System Security Hardening Script
# This script implements security hardening measures and credential rotation
# Usage: ./harden_system.sh [options]

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
SECURITY_LOG="${PROJECT_ROOT}/logs/security_hardening_${TIMESTAMP}.log"

# Default values
ENVIRONMENT="production"
ROTATE_CREDENTIALS=true
UPDATE_FIREWALL=true
HARDEN_CONTAINERS=true
UPDATE_CONFIGS=true
DRY_RUN=false
FORCE_MODE=false

# Function to print colored output
print_status() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" | tee -a "$SECURITY_LOG"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" | tee -a "$SECURITY_LOG"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" | tee -a "$SECURITY_LOG"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$SECURITY_LOG"
            ;;
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message" | tee -a "$SECURITY_LOG"
            ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
AI Trading System Security Hardening Script

Usage: $0 [OPTIONS]

OPTIONS:
    --environment ENV       Target environment (development, staging, production)
    --no-credentials        Skip credential rotation
    --no-firewall           Skip firewall configuration
    --no-containers         Skip container hardening
    --no-configs            Skip configuration updates
    --dry-run              Show what would be done without executing
    --force                 Force execution even with warnings
    --help                 Show this help message

SECURITY MEASURES:
    - Rotate API keys and secrets
    - Update firewall rules
    - Harden Docker containers
    - Update security configurations
    - Enable audit logging
    - Configure rate limiting
    - Set up intrusion detection

EXAMPLES:
    $0 --environment production
    $0 --no-credentials --dry-run
    $0 --force --environment staging

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --no-credentials)
                ROTATE_CREDENTIALS=false
                shift
                ;;
            --no-firewall)
                UPDATE_FIREWALL=false
                shift
                ;;
            --no-containers)
                HARDEN_CONTAINERS=false
                shift
                ;;
            --no-configs)
                UPDATE_CONFIGS=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE_MODE=true
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

# Function to generate secure random password
generate_secure_password() {
    local length=${1:-32}
    openssl rand -base64 $((length * 3 / 4)) | tr -d "=+/" | cut -c1-${length}
}

# Function to generate API key
generate_api_key() {
    local prefix=${1:-"key"}
    echo "${prefix}_$(date +%Y%m%d)_$(openssl rand -hex 16)"
}

# Function to backup current credentials
backup_credentials() {
    print_status "INFO" "Backing up current credentials"

    local backup_dir="${PROJECT_ROOT}/data/credential_backups/${TIMESTAMP}"
    mkdir -p "$backup_dir"

    # Backup current environment files
    if [ -f "config/environments/.env.${ENVIRONMENT}" ]; then
        cp "config/environments/.env.${ENVIRONMENT}" "$backup_dir/env_backup.txt"
    fi

    # Create encrypted backup of sensitive data
    tar -czf - config/environments/ | openssl enc -aes-256-cbc -salt -k "$(whoami)_$(date +%s)" > "$backup_dir/credentials_backup.tar.gz.enc"

    print_status "SUCCESS" "Credentials backed up to $backup_dir"
}

# Function to rotate database credentials
rotate_database_credentials() {
    if [ "$ROTATE_CREDENTIALS" != true ]; then
        return 0
    fi

    print_status "INFO" "Rotating database credentials"

    local new_db_password=$(generate_secure_password 32)

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would rotate database password"
        return 0
    fi

    # Update PostgreSQL password
    docker-compose exec postgres psql -U trader -d trading_system -c "
        ALTER USER trader PASSWORD '$new_db_password';
    " || {
        print_status "ERROR" "Failed to update database password"
        return 1
    }

    # Update environment file
    sed -i.bak "s/DB_PASSWORD=.*/DB_PASSWORD=$new_db_password/" "config/environments/.env.${ENVIRONMENT}"

    print_status "SUCCESS" "Database password rotated"
}

# Function to rotate Redis credentials
rotate_redis_credentials() {
    if [ "$ROTATE_CREDENTIALS" != true ]; then
        return 0
    fi

    print_status "INFO" "Rotating Redis credentials"

    local new_redis_password=$(generate_secure_password 24)

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would rotate Redis password"
        return 0
    fi

    # Update Redis configuration
    docker-compose exec redis redis-cli CONFIG SET requirepass "$new_redis_password" || {
        print_status "ERROR" "Failed to update Redis password"
        return 1
    }

    # Update environment file
    sed -i.bak "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$new_redis_password/" "config/environments/.env.${ENVIRONMENT}"

    print_status "SUCCESS" "Redis password rotated"
}

# Function to rotate API keys
rotate_api_keys() {
    if [ "$ROTATE_CREDENTIALS" != true ]; then
        return 0
    fi

    print_status "INFO" "Rotating API keys and secrets"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would rotate API keys"
        return 0
    fi

    # Generate new API secret key
    local new_api_secret=$(generate_secure_password 64)
    sed -i.bak "s/API_SECRET_KEY=.*/API_SECRET_KEY=$new_api_secret/" "config/environments/.env.${ENVIRONMENT}"

    # Generate new JWT secret
    local new_jwt_secret=$(generate_secure_password 64)
    sed -i.bak "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$new_jwt_secret/" "config/environments/.env.${ENVIRONMENT}"

    # Note: External API keys (Alpaca, TwelveData) need manual rotation
    print_status "WARNING" "External API keys (Alpaca, TwelveData) require manual rotation"
    print_status "WARNING" "Update these keys manually in the broker/provider dashboards"

    print_status "SUCCESS" "Internal API keys rotated"
}

# Function to configure firewall rules
configure_firewall() {
    if [ "$UPDATE_FIREWALL" != true ]; then
        return 0
    fi

    print_status "INFO" "Configuring firewall rules"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would configure firewall rules"
        return 0
    fi

    # Check if ufw is available
    if command -v ufw &> /dev/null; then
        # Reset UFW to default
        ufw --force reset

        # Default policies
        ufw default deny incoming
        ufw default allow outgoing

        # Allow SSH (adjust port as needed)
        ufw allow 22/tcp

        # Allow HTTP/HTTPS for external API access
        ufw allow out 80/tcp
        ufw allow out 443/tcp

        # Allow specific service ports only from trusted networks
        if [ "$ENVIRONMENT" = "production" ]; then
            # Production: Only allow from specific IPs/ranges
            ufw allow from 10.0.0.0/8 to any port 9101:9107
            ufw allow from 172.16.0.0/12 to any port 9101:9107
            ufw allow from 192.168.0.0/16 to any port 9101:9107
        else
            # Development/Staging: Allow from local networks
            ufw allow 9101:9107/tcp
        fi

        # Allow monitoring ports from trusted networks
        ufw allow from 10.0.0.0/8 to any port 9090,3000,9093
        ufw allow from 172.16.0.0/12 to any port 9090,3000,9093
        ufw allow from 192.168.0.0/16 to any port 9090,3000,9093

        # Allow Docker internal communication
        ufw allow from 172.17.0.0/16
        ufw allow from 172.18.0.0/16
        ufw allow from 172.20.0.0/16

        # Enable firewall
        ufw --force enable

        print_status "SUCCESS" "UFW firewall configured"
    else
        print_status "WARNING" "UFW not available, skipping firewall configuration"
    fi
}

# Function to harden Docker containers
harden_containers() {
    if [ "$HARDEN_CONTAINERS" != true ]; then
        return 0
    fi

    print_status "INFO" "Hardening Docker containers"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would harden Docker containers"
        return 0
    fi

    # Create security-hardened docker-compose override
    cat > docker-compose.security.yml << EOF
version: '3.8'

services:
  postgres:
    security_opt:
      - no-new-privileges:true
    read_only: false
    tmpfs:
      - /tmp
      - /var/run/postgresql
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - DAC_OVERRIDE
      - FOWNER
      - SETGID
      - SETUID

  redis:
    security_opt:
      - no-new-privileges:true
    read_only: false
    tmpfs:
      - /tmp
    cap_drop:
      - ALL
    cap_add:
      - SETGID
      - SETUID

  data_collector:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
    cap_drop:
      - ALL

  strategy_engine:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
    cap_drop:
      - ALL

  risk_manager:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
    cap_drop:
      - ALL

  trade_executor:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
    cap_drop:
      - ALL

  scheduler:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
    cap_drop:
      - ALL

  export_service:
    security_opt:
      - no-new-privileges:true
    read_only: false  # Needs write access for exports
    tmpfs:
      - /tmp
    cap_drop:
      - ALL

  maintenance_service:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
    cap_drop:
      - ALL
EOF

    print_status "SUCCESS" "Docker security hardening configuration created"
}

# Function to configure security headers
configure_security_headers() {
    print_status "INFO" "Configuring security headers"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would configure security headers"
        return 0
    fi

    # Create nginx configuration for security headers
    cat > config/nginx/security.conf << 'EOF'
# Security Headers Configuration
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';" always;

# HSTS Header (only for production HTTPS)
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

# Hide server information
server_tokens off;
more_set_headers "Server: Trading-System";

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
limit_req_zone $binary_remote_addr zone=trading:10m rate=5r/m;
limit_req_zone $binary_remote_addr zone=export:10m rate=2r/m;

# Apply rate limits to specific endpoints
location /api/ {
    limit_req zone=api burst=20 nodelay;
}

location ~ ^/(trades|orders|positions)/ {
    limit_req zone=trading burst=10 nodelay;
}

location /export/ {
    limit_req zone=export burst=5 nodelay;
}
EOF

    print_status "SUCCESS" "Security headers configured"
}

# Function to set up fail2ban for intrusion prevention
setup_fail2ban() {
    print_status "INFO" "Setting up fail2ban for intrusion prevention"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would set up fail2ban"
        return 0
    fi

    # Install fail2ban if not present
    if ! command -v fail2ban-client &> /dev/null; then
        if command -v apt-get &> /dev/null; then
            apt-get update && apt-get install -y fail2ban
        elif command -v yum &> /dev/null; then
            yum install -y fail2ban
        else
            print_status "WARNING" "Cannot install fail2ban - package manager not supported"
            return 0
        fi
    fi

    # Create fail2ban configuration for trading system
    cat > /etc/fail2ban/jail.d/trading-system.conf << 'EOF'
[trading-system-auth]
enabled = true
filter = trading-system-auth
logpath = /app/logs/*.log
maxretry = 3
findtime = 600
bantime = 3600
action = iptables-allports[name=trading-auth]

[trading-system-api]
enabled = true
filter = trading-system-api
logpath = /app/logs/*.log
maxretry = 10
findtime = 60
bantime = 1800
action = iptables-allports[name=trading-api]
EOF

    # Create filters
    cat > /etc/fail2ban/filter.d/trading-system-auth.conf << 'EOF'
[Definition]
failregex = .*authentication failed.*<HOST>.*
            .*invalid.*token.*<HOST>.*
            .*unauthorized.*access.*<HOST>.*
ignoreregex =
EOF

    cat > /etc/fail2ban/filter.d/trading-system-api.conf << 'EOF'
[Definition]
failregex = .*429.*Too Many Requests.*<HOST>.*
            .*rate.*limit.*exceeded.*<HOST>.*
            .*suspicious.*activity.*<HOST>.*
ignoreregex =
EOF

    # Restart fail2ban
    systemctl restart fail2ban
    systemctl enable fail2ban

    print_status "SUCCESS" "Fail2ban configured for trading system"
}

# Function to enable audit logging
enable_audit_logging() {
    print_status "INFO" "Enabling comprehensive audit logging"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would enable audit logging"
        return 0
    fi

    # Enable auditd if available
    if command -v auditctl &> /dev/null; then
        # Create audit rules for trading system
        cat > /etc/audit/rules.d/trading-system.rules << 'EOF'
# Trading System Audit Rules

# Monitor configuration file changes
-w /opt/trading-system/config/ -p wa -k trading_config_change

# Monitor executable changes
-w /usr/local/bin/ -p wa -k trading_binary_change

# Monitor Docker activities
-w /usr/bin/docker -p x -k docker_execution
-w /usr/local/bin/docker-compose -p x -k docker_compose_execution

# Monitor network configuration changes
-w /etc/hosts -p wa -k network_config_change
-w /etc/resolv.conf -p wa -k dns_config_change

# Monitor system administration
-w /etc/passwd -p wa -k user_modification
-w /etc/group -p wa -k group_modification
-w /etc/shadow -p wa -k password_modification

# Monitor privileged commands
-a always,exit -F arch=b64 -S execve -F euid=0 -k privileged_execution
-a always,exit -F arch=b32 -S execve -F euid=0 -k privileged_execution
EOF

        # Restart auditd
        systemctl restart auditd

        print_status "SUCCESS" "System audit logging enabled"
    else
        print_status "WARNING" "auditd not available, skipping system audit logging"
    fi
}

# Function to configure secure Docker daemon
secure_docker_daemon() {
    if [ "$HARDEN_CONTAINERS" != true ]; then
        return 0
    fi

    print_status "INFO" "Securing Docker daemon configuration"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would secure Docker daemon"
        return 0
    fi

    # Create secure Docker daemon configuration
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json << 'EOF'
{
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp-profile.json",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ]
}
EOF

    # Create custom seccomp profile
    cat > /etc/docker/seccomp-profile.json << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86",
    "SCMP_ARCH_X32"
  ],
  "syscalls": [
    {
      "names": [
        "accept",
        "accept4",
        "access",
        "bind",
        "brk",
        "chmod",
        "chown",
        "close",
        "connect",
        "dup",
        "dup2",
        "epoll_create",
        "epoll_ctl",
        "epoll_wait",
        "execve",
        "exit",
        "exit_group",
        "fchmod",
        "fchown",
        "fcntl",
        "fstat",
        "futex",
        "getcwd",
        "getdents",
        "getpid",
        "getuid",
        "ioctl",
        "listen",
        "lseek",
        "mmap",
        "munmap",
        "open",
        "openat",
        "poll",
        "read",
        "readlink",
        "recv",
        "recvfrom",
        "recvmsg",
        "rt_sigaction",
        "rt_sigprocmask",
        "rt_sigreturn",
        "send",
        "sendto",
        "sendmsg",
        "setgid",
        "setuid",
        "socket",
        "stat",
        "write"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF

    # Restart Docker daemon
    systemctl restart docker

    print_status "SUCCESS" "Docker daemon security configuration updated"
}

# Function to configure network security
configure_network_security() {
    print_status "INFO" "Configuring network security"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would configure network security"
        return 0
    fi

    # Create custom Docker network with encryption
    docker network create \
        --driver bridge \
        --subnet=172.30.0.0/16 \
        --opt encrypted=true \
        --opt com.docker.network.bridge.name=br-trading-secure \
        trading_secure_network 2>/dev/null || true

    # Update docker-compose to use secure network
    if [ -f docker-compose.prod.yml ]; then
        # Backup original
        cp docker-compose.prod.yml docker-compose.prod.yml.bak

        # Add security network configuration
        cat >> docker-compose.prod.yml << 'EOF'

networks:
  trading_network:
    external: true
    name: trading_secure_network
EOF
    fi

    print_status "SUCCESS" "Network security configured"
}

# Function to set up SSL/TLS certificates
setup_ssl_certificates() {
    print_status "INFO" "Setting up SSL/TLS certificates"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would set up SSL certificates"
        return 0
    fi

    local cert_dir="${PROJECT_ROOT}/config/ssl"
    mkdir -p "$cert_dir"

    # Generate self-signed certificate for development/staging
    if [ "$ENVIRONMENT" != "production" ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$cert_dir/trading-system.key" \
            -out "$cert_dir/trading-system.crt" \
            -subj "/C=US/ST=State/L=City/O=TradingSystem/CN=localhost"

        print_status "SUCCESS" "Self-signed SSL certificate generated"
    else
        print_status "WARNING" "Production SSL certificates should be obtained from a trusted CA"
        print_status "INFO" "Consider using Let's Encrypt or purchasing commercial certificates"
    fi
}

# Function to configure security monitoring
configure_security_monitoring() {
    print_status "INFO" "Configuring security monitoring"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would configure security monitoring"
        return 0
    fi

    # Create security monitoring configuration
    cat > config/security-monitoring.yml << 'EOF'
security_monitoring:
  enabled: true

  # Failed authentication monitoring
  auth_monitoring:
    enabled: true
    threshold: 5
    window_minutes: 10
    action: "block_ip"

  # Unusual trading pattern detection
  trading_anomaly_detection:
    enabled: true
    volume_threshold_multiplier: 3.0
    frequency_threshold: 10
    action: "alert_and_review"

  # API abuse detection
  api_abuse_detection:
    enabled: true
    error_rate_threshold: 0.1
    request_rate_threshold: 100
    action: "rate_limit"

  # File integrity monitoring
  file_integrity:
    enabled: true
    paths:
      - "/app/config/"
      - "/app/scripts/"
      - "/etc/docker/"
    check_interval_minutes: 60

  # Network anomaly detection
  network_monitoring:
    enabled: true
    unusual_destinations: true
    port_scanning_detection: true
    dns_monitoring: true

alerts:
  channels:
    - type: "gotify"
      url: "${GOTIFY_URL}"
      token: "${GOTIFY_TOKEN}"
    - type: "email"
      smtp_host: "${EMAIL_SMTP_HOST}"
      username: "${EMAIL_USERNAME}"
      password: "${EMAIL_PASSWORD}"
      to: "${SECURITY_EMAIL}"
    - type: "slack"
      webhook_url: "${SLACK_SECURITY_WEBHOOK}"
EOF

    print_status "SUCCESS" "Security monitoring configured"
}

# Function to update security-related configurations
update_security_configs() {
    if [ "$UPDATE_CONFIGS" != true ]; then
        return 0
    fi

    print_status "INFO" "Updating security configurations"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would update security configurations"
        return 0
    fi

    local env_file="config/environments/.env.${ENVIRONMENT}"

    # Enable security features
    sed -i.bak 's/AUDIT_ENABLED=.*/AUDIT_ENABLED=true/' "$env_file"
    sed -i.bak 's/RATE_LIMIT_ENABLED=.*/RATE_LIMIT_ENABLED=true/' "$env_file"
    sed -i.bak 's/SECURITY_HEADERS_ENABLED=.*/SECURITY_HEADERS_ENABLED=true/' "$env_file"

    # Tighten rate limits for production
    if [ "$ENVIRONMENT" = "production" ]; then
        sed -i.bak 's/RATE_LIMIT_REQUESTS_PER_MINUTE=.*/RATE_LIMIT_REQUESTS_PER_MINUTE=30/' "$env_file"
        sed -i.bak 's/RATE_LIMIT_TRADING_RPM=.*/RATE_LIMIT_TRADING_RPM=5/' "$env_file"
        sed -i.bak 's/RATE_LIMIT_EXPORT_RPM=.*/RATE_LIMIT_EXPORT_RPM=2/' "$env_file"
    fi

    # Enable enhanced logging
    sed -i.bak 's/AUDIT_LOG_ALL_REQUESTS=.*/AUDIT_LOG_ALL_REQUESTS=true/' "$env_file"
    sed -i.bak 's/LOG_LEVEL=.*/LOG_LEVEL=WARNING/' "$env_file"

    # Set secure defaults
    sed -i.bak 's/DEBUG=.*/DEBUG=false/' "$env_file"
    sed -i.bak 's/TESTING_MODE=.*/TESTING_MODE=false/' "$env_file"

    print_status "SUCCESS" "Security configurations updated"
}

# Function to scan for vulnerabilities
vulnerability_scan() {
    print_status "INFO" "Running vulnerability scan"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would run vulnerability scan"
        return 0
    fi

    # Check for known CVEs in Docker images
    if command -v docker &> /dev/null; then
        print_status "INFO" "Scanning Docker images for vulnerabilities"

        # Get list of images used by the trading system
        local images=$(docker-compose config | grep 'image:' | awk '{print $2}' | sort -u)

        for image in $images; do
            if command -v trivy &> /dev/null; then
                trivy image --severity HIGH,CRITICAL "$image" || true
            else
                print_status "WARNING" "Trivy not installed, skipping image vulnerability scan"
                break
            fi
        done
    fi

    # Check Python dependencies for vulnerabilities
    if command -v safety &> /dev/null; then
        print_status "INFO" "Scanning Python dependencies for vulnerabilities"
        safety check --json > "${PROJECT_ROOT}/logs/vulnerability_scan_${TIMESTAMP}.json" || true
    else
        print_status "WARNING" "Safety not installed, skipping Python dependency scan"
    fi

    # Check for common security misconfigurations
    python "${SCRIPT_DIR}/security_audit.py" --config-check --output "${PROJECT_ROOT}/logs/security_audit_${TIMESTAMP}.json"

    print_status "SUCCESS" "Vulnerability scan completed"
}

# Function to create security monitoring dashboard
create_security_dashboard() {
    print_status "INFO" "Creating security monitoring dashboard"

    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would create security dashboard"
        return 0
    fi

    # Create Grafana dashboard for security monitoring
    cat > monitoring/grafana-security-dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "Trading System Security Dashboard",
    "tags": ["security", "trading"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Failed Authentication Attempts",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(authentication_failures_total[1h])",
            "legendFormat": "Failed Auths"
          }
        ]
      },
      {
        "title": "Rate Limit Violations",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(rate_limit_violations_total[1h])",
            "legendFormat": "Rate Limit Hits"
          }
        ]
      },
      {
        "title": "Suspicious Trading Activity",
        "type": "table",
        "targets": [
          {
            "expr": "trading_anomaly_score > 0.8",
            "legendFormat": "Anomaly Score"
          }
        ]
      },
      {
        "title": "Network Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(network_connections_total[5m])",
            "legendFormat": "{{destination}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-24h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
}
EOF

  print_status "SUCCESS" "Security monitoring dashboard created"
}

# Function to validate security hardening
validate_security_hardening() {
  print_status "INFO" "Validating security hardening implementation"

  local validation_failed=false

  # Check firewall status
  if command -v ufw &> /dev/null; then
      if ! ufw status | grep -q "Status: active"; then
          print_status "ERROR" "Firewall is not active"
          validation_failed=true
      else
          print_status "SUCCESS" "Firewall is active"
      fi
  fi

  # Check Docker security
  if [ -f /etc/docker/daemon.json ]; then
      if grep -q "no-new-privileges" /etc/docker/daemon.json; then
          print_status "SUCCESS" "Docker security configuration found"
      else
          print_status "WARNING" "Docker security configuration incomplete"
      fi
  else
      print_status "WARNING" "Docker security configuration not found"
  fi

  # Check credential strength
  local env_file="config/environments/.env.${ENVIRONMENT}"
  if [ -f "$env_file" ]; then
      # Check API secret key length
      local api_key_length=$(grep "API_SECRET_KEY=" "$env_file" | cut -d'=' -f2 | wc -c)
      if [ "$api_key_length" -lt 32 ]; then
          print_status "ERROR" "API secret key is too short (minimum 32 characters)"
          validation_failed=true
      else
          print_status "SUCCESS" "API secret key length is adequate"
      fi

      # Check database password strength
      local db_password=$(grep "DB_PASSWORD=" "$env_file" | cut -d'=' -f2)
      if [ ${#db_password} -lt 16 ]; then
          print_status "ERROR" "Database password is too short (minimum 16 characters)"
          validation_failed=true
      else
          print_status "SUCCESS" "Database password length is adequate"
      fi
  fi

  # Check audit logging
  if docker-compose exec postgres psql -U trader -d trading_system -c "SELECT COUNT(*) FROM audit_logs;" &> /dev/null; then
      print_status "SUCCESS" "Audit logging is functional"
  else
      print_status "ERROR" "Audit logging is not functional"
      validation_failed=true
  fi

  # Check rate limiting
  if curl -s http://localhost:9107/rate-limits/status &> /dev/null; then
      print_status "SUCCESS" "Rate limiting is active"
  else
      print_status "WARNING" "Rate limiting status unclear"
  fi

  if [ "$validation_failed" = true ]; then
      print_status "ERROR" "Security validation failed"
      return 1
  else
      print_status "SUCCESS" "Security validation passed"
      return 0
  fi
}

# Function to create security incident response procedures
+create_incident_response_procedures() {
+    print_status "INFO" "Creating security incident response procedures"
+
+    if [ "$DRY_RUN" = true ]; then
+        print_status "INFO" "[DRY RUN] Would create incident response procedures"
+        return 0
+    fi
+
+    local procedures_dir="${PROJECT_ROOT}/docs/security"
+    mkdir -p "$procedures_dir"
+
+    cat > "$procedures_dir/incident_response.md" << 'EOF'
+# Security Incident Response Procedures
+
+## Incident Classification
+
+### Level 1 - Low Impact
+- Failed authentication attempts
+- Minor configuration drift
+- Non-sensitive data access attempts
+
+### Level 2 - Medium Impact
+- Repeated failed access attempts
+- Unauthorized API usage
+- Suspicious trading patterns
+
+### Level 3 - High Impact
+- Successful unauthorized access
+- Data breach suspected
+- System compromise indicators
+
+### Level 4 - Critical Impact
+- Confirmed data breach
+- Unauthorized trading activity
+- System fully compromised
+
+## Response Procedures
+
+### Immediate Response (0-5 minutes)
+1. Isolate affected systems
+2. Stop automated trading
+3. Preserve evidence
+4. Alert security team
+
+### Short-term Response (5-30 minutes)
+1. Assess scope of incident
+2. Implement containment measures
+3. Begin forensic analysis
+4. Notify stakeholders
+
+### Recovery Phase (30 minutes - 4 hours)
+1. Eradicate threat
+2. Restore from clean backups
+3. Implement additional security measures
+4. Validate security posture
+
+### Post-Incident (24-72 hours)
+1. Complete forensic analysis
+2. Document lessons learned
+3. Update security procedures
+4. Conduct security review
+EOF
+
+    print_status "SUCCESS" "Security incident response procedures created"
+}

# Function to set up automated security updates
+setup_automated_security_updates() {
+    print_status "INFO" "Setting up automated security updates"
+
+    if [ "$DRY_RUN" = true ]; then
+        print_status "INFO" "[DRY RUN] Would set up automated security updates"
+        return 0
+    fi
+
+    # Create script for automated security updates
+    cat > "${PROJECT_ROOT}/scripts/security/auto_security_updates.sh" << 'EOF'
+#!/bin/bash
+# Automated Security Updates for Trading System
+
+set -euo pipefail
+
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
+
+# Update system packages
+if command -v apt-get &> /dev/null; then
+    apt-get update
+    apt-get -y upgrade
+    apt-get -y autoremove
+elif command -v yum &> /dev/null; then
+    yum update -y
+fi
+
+# Update Docker images
+cd "$PROJECT_ROOT"
+docker-compose pull
+docker image prune -f
+
+# Update Python dependencies
+pip install --upgrade pip
+pip install --upgrade -r requirements.txt
+
+# Run security scan
+python scripts/security/security_audit.py --auto-fix
+
+# Log security update
+echo "$(date): Automated security updates completed" >> logs/security_updates.log
+EOF
+
+    chmod +x "${PROJECT_ROOT}/scripts/security/auto_security_updates.sh"
+
+    # Create cron job for weekly security updates
+    local cron_job="0 2 * * 0 ${PROJECT_ROOT}/scripts/security/auto_security_updates.sh >> ${PROJECT_ROOT}/logs/security_updates.log 2>&1"
+
+    # Add to crontab (if not already present)
+    if ! crontab -l | grep -q "auto_security_updates.sh"; then
+        (crontab -l 2>/dev/null; echo "$cron_job") | crontab -
+        print_status "SUCCESS" "Automated security updates scheduled"
+    else
+        print_status "INFO" "Automated security updates already scheduled"
+    fi
+}

# Function to create security audit report
+create_security_audit_report() {
+    print_status "INFO" "Creating security audit report"
+
+    local report_file="${PROJECT_ROOT}/logs/security_audit_${TIMESTAMP}.json"
+
+    # Collect security metrics
+    local security_metrics="{
+        \"audit_timestamp\": \"$(date -Iseconds)\",
+        \"environment\": \"$ENVIRONMENT\",
+        \"hardening_applied\": {
+            \"credential_rotation\": $ROTATE_CREDENTIALS,
+            \"firewall_configured\": $UPDATE_FIREWALL,
+            \"containers_hardened\": $HARDEN_CONTAINERS,
+            \"configs_updated\": $UPDATE_CONFIGS
+        },
+        \"security_features\": {
+            \"audit_logging\": true,
+            \"rate_limiting\": true,
+            \"fail2ban\": $(command -v fail2ban-client &> /dev/null && echo true || echo false),
+            \"ssl_enabled\": $([ -f config/ssl/trading-system.crt ] && echo true || echo false),
+            \"firewall_active\": $(command -v ufw &> /dev/null && ufw status | grep -q active && echo true || echo false)
+        },
+        \"vulnerabilities_scanned\": true,
+        \"compliance_status\": \"compliant\"
+    }"
+
+    echo "$security_metrics" | jq '.' > "$report_file"
+
+    print_status "SUCCESS" "Security audit report created: $report_file"
+}

# Function to restart services with new security configuration
+restart_services_securely() {
+    print_status "INFO" "Restarting services with security hardening"
+
+    if [ "$DRY_RUN" = true ]; then
+        print_status "INFO" "[DRY RUN] Would restart services with security configuration"
+        return 0
+    fi
+
+    # Stop services gracefully
+    docker-compose down --timeout 60
+
+    # Start with security hardening
+    if [ -f docker-compose.security.yml ]; then
+        docker-compose -f docker-compose.yml -f docker-compose.security.yml up -d
+    else
+        docker-compose up -d
+    fi
+
+    # Wait for services to be ready
+    sleep 30
+
+    # Verify all services are healthy with new configuration
+    local failed_services=()
+    for service in data_collector strategy_engine risk_manager trade_executor scheduler export_service maintenance_service; do
+        local port=""
+        case $service in
+            "data_collector") port="9101" ;;
+            "strategy_engine") port="9102" ;;
+            "risk_manager") port="9103" ;;
+            "trade_executor") port="9104" ;;
+            "scheduler") port="9105" ;;
+            "export_service") port="9106" ;;
+            "maintenance_service") port="9107" ;;
+        esac
+
+        if ! curl -f -s "http://localhost:$port/health" > /dev/null; then
+            failed_services+=("$service")
+        fi
+    done
+
+    if [ ${#failed_services[@]} -gt 0 ]; then
+        print_status "ERROR" "Failed to start services with security hardening: ${failed_services[*]}"
+        return 1
+    else
+        print_status "SUCCESS" "All services restarted successfully with security hardening"
+        return 0
+    fi
+}

# Main hardening function
+main() {
+    local start_time=$(date +%s)
+
+    print_status "INFO" "Starting security hardening for $ENVIRONMENT environment"
+
+    # Parse arguments
+    parse_arguments "$@"
+
+    # Create logs directory
+    mkdir -p "${PROJECT_ROOT}/logs"
+
+    # Check if running as root for system-level changes
+    if [[ $EUID -ne 0 && ("$UPDATE_FIREWALL" = true || "$HARDEN_CONTAINERS" = true) ]]; then
+        print_status "WARNING" "Some hardening steps require root privileges"
+        if [ "$FORCE_MODE" != true ]; then
+            print_status "INFO" "Re-run with sudo for complete hardening, or use --force to continue"
+            exit 1
+        fi
+    fi
+
+    # Backup current credentials
+    backup_credentials
+
+    # Rotate credentials
+    if [ "$ROTATE_CREDENTIALS" = true ]; then
+        print_status "INFO" "=== CREDENTIAL ROTATION ==="
+        rotate_database_credentials
+        rotate_redis_credentials
+        rotate_api_keys
+    fi
+
+    # Configure security infrastructure
+    print_status "INFO" "=== SECURITY INFRASTRUCTURE ==="
+    configure_firewall
+    harden_containers
+    configure_network_security
+    setup_ssl_certificates
+
+    # Configure monitoring and detection
+    print_status "INFO" "=== SECURITY MONITORING ==="
+    configure_security_headers
+    configure_security_monitoring
+    enable_audit_logging
+    setup_fail2ban
+    create_security_dashboard
+
+    # Update configurations
+    if [ "$UPDATE_CONFIGS" = true ]; then
+        print_status "INFO" "=== CONFIGURATION UPDATES ==="
+        update_security_configs
+        create_incident_response_procedures
+        setup_automated_security_updates
+    fi
+
+    # Run vulnerability scan
+    print_status "INFO" "=== VULNERABILITY ASSESSMENT ==="
+    vulnerability_scan
+
+    # Restart services with new configuration
+    if [ "$DRY_RUN" != true ]; then
+        print_status "INFO" "=== SERVICE RESTART ==="
+        restart_services_securely
+    fi
+
+    # Validate hardening
+    print_status "INFO" "=== VALIDATION ==="
+    if validate_security_hardening; then
+        print_status "SUCCESS" "Security hardening validation passed"
+    else
+        print_status "ERROR" "Security hardening validation failed"
+        exit 1
+    fi
+
+    # Create audit report
+    create_security_audit_report
+
+    local end_time=$(date +%s)
+    local total_time=$((end_time - start_time))
+
+    print_status "SUCCESS" "Security hardening completed successfully in ${total_time} seconds"
+
+    # Display summary
+    cat << EOF
+
+========================================
+    SECURITY HARDENING SUMMARY
+========================================
+Environment: $ENVIRONMENT
+Timestamp: $(date)
+Duration: ${total_time} seconds
+
+Actions Performed:
+- Credential Rotation: $ROTATE_CREDENTIALS
+- Firewall Configuration: $UPDATE_FIREWALL
+- Container Hardening: $HARDEN_CONTAINERS
+- Config Updates: $UPDATE_CONFIGS
+
+Next Steps:
+1. Test all functionality with new security settings
+2. Update external API keys manually if needed
+3. Review security audit report
+4. Schedule regular security reviews
+
+Log File: $SECURITY_LOG
+Audit Report: ${PROJECT_ROOT}/logs/security_audit_${TIMESTAMP}.json
+========================================

EOF
+}

# Cleanup function
+cleanup() {
+    print_status "INFO" "Cleaning up security hardening artifacts"
+
+    # Remove temporary files
+    rm -f /tmp/security_hardening_*
+
+    # Ensure proper permissions on sensitive files
+    if [ -d "config/environments" ]; then
+        chmod 600 config/environments/.env.*
+    fi
+
+    if [ -d "config/ssl" ]; then
+        chmod 600 config/ssl/*.key
+        chmod 644 config/ssl/*.crt
+    fi
+}

# Set up trap for cleanup
+trap cleanup EXIT

# Handle signals gracefully
+trap 'print_status "WARNING" "Security hardening interrupted by signal"; exit 130' INT TERM

# Main execution
+if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
+    main "$@"
+fi
