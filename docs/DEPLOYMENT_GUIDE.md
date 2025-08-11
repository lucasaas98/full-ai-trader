# AI Trading System Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Initial Deployment](#initial-deployment)
5. [Production Deployment](#production-deployment)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Monitoring Setup](#monitoring-setup)
8. [Security Configuration](#security-configuration)
9. [Backup Configuration](#backup-configuration)
10. [Operational Procedures](#operational-procedures)
11. [Troubleshooting](#troubleshooting)
12. [Appendices](#appendices)

---

## Overview

This guide provides comprehensive instructions for deploying the AI Trading System across different environments (development, staging, production). The system uses Docker containers orchestrated with Docker Compose, providing scalable and maintainable deployment.

### Deployment Architecture
- **Multi-service architecture** with independent scaling
- **Infrastructure services**: PostgreSQL, Redis
- **Core trading services**: Data Collector, Strategy Engine, Risk Manager, Trade Executor, Scheduler
- **Support services**: Export Service, Maintenance Service
- **Monitoring stack**: Prometheus, Grafana, Elasticsearch, Kibana, Alertmanager
- **Security layer**: Rate limiting, audit logging, encryption

### Supported Environments
- **Development**: Local development with minimal resources
- **Staging**: Pre-production testing environment
- **Production**: High-availability production deployment

---

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk**: 50 GB SSD
- **Network**: 100 Mbps

#### Recommended Requirements (Production)
- **CPU**: 16 cores
- **RAM**: 32 GB
- **Disk**: 500 GB NVMe SSD
- **Network**: 1 Gbps

### Software Dependencies
```bash
# Required
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- Git
- curl
- openssl

# Optional (for enhanced operations)
- ufw (firewall)
- logrotate
- cron
- htop
- jq
```

### Installation Commands
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose python3 python3-pip git curl openssl ufw

# CentOS/RHEL
sudo yum install -y docker docker-compose python3 python3-pip git curl openssl

# Enable Docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```

### Network Configuration
```bash
# Open required ports (production)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8080/tcp    # Trading API

# Internal ports (restrict to local network)
sudo ufw allow from 10.0.0.0/8 to any port 5432    # PostgreSQL
sudo ufw allow from 172.16.0.0/12 to any port 6379  # Redis
sudo ufw allow from 192.168.0.0/16 to any port 9090 # Prometheus
```

---

## Environment Setup

### 1. Download and Setup Project
```bash
# Clone repository
git clone <repository-url> ai-trading-system
cd ai-trading-system

# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*/*.sh
chmod +x scripts/*/*/*.sh

# Create required directories
mkdir -p {data,logs,config}/{backups,exports,secrets,monitoring}
```

### 2. Environment Configuration
```bash
# Run automated environment setup
./scripts/setup_environment.sh development

# Or for production
./scripts/setup_environment.sh production --force
```

### 3. Manual Environment Setup (if needed)
```bash
# Copy environment template
cp config/environments/.env.template config/environments/.env.development

# Edit environment variables
nano config/environments/.env.development
```

Required environment variables:
```bash
# Core Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://trading_user:password@postgres:5432/trading_db
POSTGRES_DB=trading_db
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_URL=redis://:password@redis:6379/0
REDIS_PASSWORD=secure_password

# Security
JWT_SECRET=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here
API_SECRET_KEY=your-api-secret-here

# Trading Configuration
TRADING_MODE=paper  # or live for production
EXCHANGE_API_KEY=your-exchange-api-key
EXCHANGE_SECRET=your-exchange-secret
RISK_LIMIT_DAILY=10000
MAX_POSITION_SIZE=1000

# Monitoring
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=admin
SLACK_WEBHOOK_URL=https://hooks.slack.com/your-webhook
ALERT_EMAIL=alerts@yourcompany.com
```

---

## Initial Deployment

### Development Environment

#### Quick Start
```bash
# 1. Setup environment
./scripts/setup_environment.sh development

# 2. Start all services
./scripts/ops_manager.sh start

# 3. Verify deployment
./scripts/ops_manager.sh health

# 4. Access interfaces
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Trading API: http://localhost:8080"
```

#### Step-by-Step Development Deployment
```bash
# 1. Start infrastructure
./scripts/ops_manager.sh start-infrastructure
sleep 30

# 2. Run database migrations
./scripts/deployment/migrate.sh --initialize

# 3. Start core services
./scripts/ops_manager.sh start-core
sleep 20

# 4. Start monitoring
./scripts/ops_manager.sh start-monitoring

# 5. Start support services
docker-compose up -d export_service maintenance_service

# 6. Verify all services
./scripts/ops_manager.sh status
```

### Staging Environment

#### Staging Deployment
```bash
# 1. Setup staging environment
./scripts/setup_environment.sh staging --force

# 2. Deploy with testing
./scripts/deployment/deploy.sh staging --skip-backup

# 3. Run integration tests
python3 scripts/run_tests.py --integration-tests

# 4. Performance testing
./scripts/performance_tuning.sh balanced --dry-run

# 5. Security validation
./scripts/security/audit_security.sh --type quick
```

---

## Production Deployment

### Pre-Deployment Checklist
- [ ] Infrastructure requirements verified
- [ ] Network security configured
- [ ] SSL certificates obtained
- [ ] Secrets and API keys configured
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Team notifications setup
- [ ] Staging environment tested
- [ ] Rollback plan prepared

### Production Deployment Steps

#### 1. Infrastructure Preparation
```bash
# System hardening
./scripts/security/harden_system.sh

# Firewall configuration
./scripts/security/setup_firewall.sh

# Performance optimization
./scripts/performance_tuning.sh aggressive --apply
```

#### 2. Security Setup
```bash
# Generate production secrets
./scripts/setup_environment.sh production --no-deps --no-database

# Configure SSL certificates
# Copy your SSL certificates to config/ssl/
# Or generate self-signed for testing:
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout config/ssl/trading-system.key \
  -out config/ssl/trading-system.crt

# Set proper permissions
chmod 600 config/ssl/*.key
chmod 644 config/ssl/*.crt
```

#### 3. Database Setup
```bash
# Start database
docker-compose -f docker-compose.prod.yml up -d postgres

# Wait for database
sleep 30

# Initialize schema
./scripts/deployment/migrate.sh --initialize --env production

# Verify database
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "\dt"
```

#### 4. Full Production Deployment
```bash
# One-command production deployment
./scripts/deployment/deploy.sh production \
  --force-rebuild \
  --health-timeout 600 \
  --migration-timeout 900

# Or step-by-step deployment
./scripts/ops_manager.sh start-infrastructure --env production
sleep 60
./scripts/ops_manager.sh start-core --env production  
sleep 30
./scripts/ops_manager.sh start-monitoring --env production
```

#### 5. Post-Deployment Configuration
```bash
# Configure monitoring dashboards
# Import dashboards in Grafana at http://localhost:3000

# Setup alerting
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d @monitoring/alertmanager/test-alert.json

# Configure backup schedule
./scripts/backup/run_backups.sh
```

---

## Post-Deployment Verification

### Health Checks
```bash
# Comprehensive health check
./scripts/ops_manager.sh health

# Service-specific checks
curl http://localhost:8001/health  # Data Collector
curl http://localhost:8002/health  # Strategy Engine
curl http://localhost:8003/health  # Risk Manager
curl http://localhost:8004/health  # Trade Executor
curl http://localhost:8005/health  # Scheduler

# Database connectivity
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U trading_user -d trading_db

# Redis connectivity  
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
```

### Integration Testing
```bash
# Run integration tests
python3 scripts/run_tests.py --integration-tests --env production

# Test data collection
curl -X POST http://localhost:8001/api/test/collect

# Test strategy execution
curl -X POST http://localhost:8002/api/test/strategy

# Test trade execution (paper mode)
curl -X POST http://localhost:8004/api/test/execute
```

### Performance Validation
```bash
# Benchmark system
./scripts/performance_tuning.sh balanced --apply
# Follow prompts to run benchmark

# Monitor initial performance
./scripts/monitor_performance.sh --once

# Check resource usage
./scripts/monitor_system.sh
```

### Security Validation
```bash
# Security audit
./scripts/security/audit_security.sh --type full --env production

# Test rate limiting
for i in {1..20}; do curl http://localhost:8001/health; done

# Verify SSL
curl -I https://localhost:8080  # If HTTPS configured
```

---

## Monitoring Setup

### Grafana Configuration
```bash
# Access Grafana
echo "URL: http://localhost:3000"
echo "User: admin"
echo "Pass: admin"

# Import dashboards
# 1. Go to + > Import
# 2. Upload dashboard JSON files from monitoring/grafana/dashboards/
# 3. Configure data sources if needed
```

### Prometheus Configuration
```bash
# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Check metrics collection
curl "http://localhost:9090/api/v1/query?query=up"
```

### Alerting Setup
```bash
# Configure Slack notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/your-webhook"

# Configure email notifications  
export ALERT_EMAIL="alerts@yourcompany.com"

# Test notifications
./scripts/monitor_system.sh --enable-slack --enable-email --test-alerts
```

---

## Security Configuration

### SSL/TLS Setup
```bash
# For production, use certificates from trusted CA
# For development/staging:
./scripts/security/generate_certificates.sh

# Configure HTTPS
# Edit docker-compose.prod.yml to enable HTTPS ports
# Update nginx configuration if using reverse proxy
```

### Secret Management
```bash
# Generate production secrets
./scripts/security/generate_secrets.sh --env production

# Rotate secrets periodically
./scripts/security/rotate_secrets.sh --all

# Audit secret usage
./scripts/security/audit_security.sh --type secrets
```

### Rate Limiting
```bash
# Configure rate limits
nano config/security/rate_limiting.yml

# Apply rate limiting configuration
./scripts/ops_manager.sh update

# Monitor rate limiting
grep "rate_limit" logs/services/*/api.log
```

---

## Backup Configuration

### Automated Backups
```bash
# Setup automated daily backups
./scripts/backup/run_backups.sh

# Verify cron job
crontab -l | grep backup

# Test backup creation
./scripts/backup/backup.sh --type test
```

### Backup Verification
```bash
# Test backup restoration
./scripts/backup/test_restore.sh --latest

# Verify backup integrity
./scripts/backup/backup.sh --verify-only --backup-id latest
```

### Backup Storage
```bash
# Local backups
ls -la data/backups/

# Configure remote backup storage (recommended for production)
# Edit scripts/backup/backup.sh to add S3/cloud storage
```

---

## Operational Procedures

### Daily Operations
```bash
# Morning checklist
./scripts/ops_manager.sh status
python3 scripts/operational_dashboard.py --mode status
./scripts/monitor_system.sh

# Health monitoring  
./scripts/ops_manager.sh health

# Performance check
./scripts/monitor_performance.sh --once

# Evening checklist
./scripts/backup/backup.sh --type daily
./scripts/cleanup_logs.sh
./scripts/security/audit_security.sh --type quick
```

### Weekly Maintenance
```bash
# Enter maintenance mode
./scripts/ops_manager.sh maintenance --enter --message "Weekly maintenance"

# Full system backup
./scripts/backup/backup.sh --type weekly --compress --verify

# Security audit
./scripts/security/audit_security.sh --type full --auto-remediate

# Performance optimization
./scripts/performance_tuning.sh balanced --apply

# System cleanup
./scripts/ops_manager.sh cleanup

# Exit maintenance mode
./scripts/ops_manager.sh maintenance --exit
```

### Configuration Updates
```bash
# 1. Validate configuration
python3 scripts/config/validate_config.py --env production --strict

# 2. Create backup
./scripts/backup/backup.sh --type config

# 3. Deploy configuration
./scripts/deployment/deploy.sh production --config-only

# 4. Verify deployment
./scripts/ops_manager.sh health
```

### Zero-Downtime Updates
```bash
# Rolling update deployment
./scripts/deployment/zero_downtime_deploy.sh --env production

# With health checks
./scripts/deployment/deploy.sh production --health-timeout 300
```

---

## Troubleshooting

### Common Deployment Issues

#### Services Won't Start
```bash
# Check prerequisites
./scripts/setup_environment.sh production --skip-validation

# Check Docker status
sudo systemctl status docker
docker info

# Check resource availability
df -h
free -h
docker system df

# Force rebuild
./scripts/deployment/deploy.sh production --force-rebuild
```

#### Database Connection Failures
```bash
# Check database status
./scripts/ops_manager.sh logs --service postgres

# Verify credentials
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "SELECT 1;"

# Reset database
./scripts/disaster_recovery.sh recover --scenario database_failure
```

#### Port Conflicts
```bash
# Check port usage
sudo netstat -tulpn | grep :8001

# Kill conflicting processes
sudo lsof -ti:8001 | xargs sudo kill -9

# Use alternative ports
# Edit docker-compose.prod.yml port mappings
```

#### Memory Issues
```bash
# Check memory usage
./scripts/monitor_system.sh

# Free memory
./scripts/disaster_recovery.sh recover --scenario memory_exhaustion

# Optimize memory usage
./scripts/performance_tuning.sh conservative --apply
```

### Service-Specific Issues

#### Data Collector Issues
```bash
# Check API connectivity
curl http://localhost:8001/health

# Verify exchange connectivity
./scripts/ops_manager.sh logs --service data_collector | grep -i error

# Restart data collector
./scripts/ops_manager.sh restart --service data_collector
```

#### Strategy Engine Problems
```bash
# Check strategy logs
./scripts/ops_manager.sh logs --service strategy_engine

# Verify strategy configuration
curl http://localhost:8002/api/strategies/status

# Reset strategy state
curl -X POST http://localhost:8002/api/strategies/reset
```

#### Trade Execution Issues
```bash
# Check executor status
curl http://localhost:8004/health

# View pending orders
curl http://localhost:8004/api/orders/pending

# Emergency position closure
curl -X POST http://localhost:8004/api/emergency/close-all
```

---

## Appendices

### A. Environment Templates

#### Development Environment (.env.development)
```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
TRADING_MODE=paper
DATABASE_URL=postgresql://trading_user:dev_password@postgres:5432/trading_db
REDIS_URL=redis://:dev_password@redis:6379/0
JWT_SECRET=dev-jwt-secret-change-in-production
ENCRYPTION_KEY=dev-encryption-key-change-in-production
```

#### Production Environment (.env.production)
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
TRADING_MODE=live
DATABASE_URL=postgresql://trading_user:${POSTGRES_PASSWORD}@postgres:5432/trading_db
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}
```

### B. Service Configuration

#### Docker Compose Override (production)
```yaml
# docker-compose.override.yml
version: '3.8'

services:
  data_collector:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2g
    restart: always
    
  strategy_engine:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4g
    restart: always
    
  postgres:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 8g
    restart: always
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql/performance.conf:/etc/postgresql/postgresql.conf
```

### C. Monitoring Queries

#### Prometheus Queries
```promql
# Service availability
up{job="trading-services"}

# CPU usage
rate(cpu_usage_total[5m])

# Memory usage
memory_usage_percent

# Trading metrics
trading_positions_total
trading_pnl_daily
trading_execution_time_seconds
```

#### Database Queries
```sql
-- Active positions
SELECT COUNT(*) FROM positions WHERE status = 'OPEN';

-- Daily P&L
SELECT SUM(realized_pnl) FROM trades WHERE DATE(created_at) = CURRENT_DATE;

-- System performance
SELECT * FROM pg_stat_database WHERE datname = 'trading_db';
```

### D. Backup Strategy

#### Backup Types and Schedule
```bash
# Continuous (every 15 minutes) - Transaction logs
*/15 * * * * /path/to/scripts/backup_transactions.sh

# Hourly - Database incremental
0 * * * * /path/to/scripts/backup/backup.sh --type incremental

# Daily (2 AM) - Full system backup
0 2 * * * /path/to/scripts/backup/run_backups.sh

# Weekly (Sunday 1 AM) - Compressed archive
0 1 * * 0 /path/to/scripts/backup/backup.sh --type weekly --compress

# Monthly - Offsite backup
0 3 1 * * /path/to/scripts/backup/backup.sh --type monthly --offsite
```

### E. Performance Benchmarks

#### Expected Performance Metrics
```
Development Environment:
- Trade execution: < 100ms
- Strategy calculation: < 50ms
- Market data processing: < 10ms
- API response time: < 20ms

Production Environment:
- Trade execution: < 50ms
- Strategy calculation: < 20ms
- Market data processing: < 5ms
- API response time: < 10ms
```

### F. Security Checklist

#### Production Security Hardening
- [ ] Change all default passwords
- [ ] Configure SSL/TLS certificates
- [ ] Enable firewall with minimal rules
- [ ] Set up rate limiting
- [ ] Configure audit logging
- [ ] Encrypt sensitive data
- [ ] Regular security scans
- [ ] Access control implementation
- [ ] Network segmentation
- [ ] Container security hardening

### G. Emergency Contact Information

#### Escalation Matrix
```
Level 1: Service Issues
- Operations Team: ops@yourcompany.com
- Response Time: 15 minutes

Level 2: System Outage  
- Engineering Team: engineering@yourcompany.com
- Response Time: 5 minutes

Level 3: Critical Financial Issues
- Risk Management: risk@yourcompany.com
- Trading Team: trading@yourcompany.com
- Response Time: Immediate

Level 4: Security Incidents
- Security Team: security@yourcompany.com
- Legal/Compliance: legal@yourcompany.com
- Response Time: Immediate
```

### H. Useful Commands Reference

#### Docker Management
```bash
# View container resources
docker stats --no-stream

# Clean up resources
docker system prune -f --volumes

# View container logs
docker-compose logs -f <service>

# Execute commands in container
docker-compose exec <service> <command>

# Scale services
docker-compose up -d --scale strategy_engine=3
```

#### Database Management
```bash
# Database backup
docker-compose exec postgres pg_dump -U trading_user trading_db > backup.sql

# Database restore
docker-compose exec -T postgres psql -U trading_user -d trading_db < backup.sql

# Database console
docker-compose exec postgres psql -U trading_user -d trading_db

# Check database size
docker-compose exec postgres psql -U trading_user -d trading_db -c "SELECT pg_size_pretty(pg_database_size('trading_db'));"
```

#### Log Management
```bash
# View all logs
./scripts/ops_manager.sh logs

# Search logs for errors
grep -r "ERROR\|CRITICAL" logs/

# Monitor logs in real-time
tail -f logs/services/*/app.log

# Rotate logs
./scripts/cleanup_logs.sh
```

---

## Support and Documentation

### Additional Resources
- **System Architecture**: [docs/SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
- **API Documentation**: [docs/API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Troubleshooting Guide**: [docs/operations/TROUBLESHOOTING.md](operations/TROUBLESHOOTING.md)
- **Disaster Recovery**: [docs/operations/DISASTER_RECOVERY.md](operations/DISASTER_RECOVERY.md)

### Getting Help
1. Check this runbook first
2. Review troubleshooting guide
3. Check system logs
4. Run diagnostic scripts
5. Contact operations team
6. Escalate to engineering team

### Contributing to Documentation
- Update procedures after changes
- Document new issues and solutions
- Keep contact information current
- Test procedures in staging first

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0     | 2024-01-01 | Initial deployment guide | AI Assistant |
| 1.1     | 2024-01-02 | Added operational scripts | AI Assistant |
| 1.2     | 2024-01-03 | Enhanced security procedures | AI Assistant |

---

*Last updated: $(date)*
*Environment: Production Ready*
*Status: Complete Operational Setup*