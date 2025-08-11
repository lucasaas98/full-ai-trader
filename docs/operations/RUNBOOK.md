# AI Trading System Operations Runbook

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [System Architecture](#system-architecture)
3. [Start/Stop Procedures](#startstop-procedures)
4. [Maintenance Operations](#maintenance-operations)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Performance Tuning](#performance-tuning)
7. [Disaster Recovery](#disaster-recovery)
8. [Monitoring and Alerting](#monitoring-and-alerting)
9. [Security Operations](#security-operations)
10. [Backup and Recovery](#backup-and-recovery)
11. [Emergency Procedures](#emergency-procedures)
12. [Common Issues](#common-issues)

---

## Quick Reference

### Emergency Contacts
- **Primary On-Call**: [Your contact info]
- **Secondary On-Call**: [Backup contact]
- **Escalation**: [Manager/Team lead]

### Critical Commands
```bash
# Emergency shutdown
./scripts/ops_manager.sh emergency-stop --force

# Check system status
./scripts/ops_manager.sh status

# Operational dashboard
python3 scripts/operational_dashboard.py --mode interactive

# View logs (last 100 lines)
./scripts/ops_manager.sh logs --service all

# Enter maintenance mode
./scripts/ops_manager.sh maintenance --enter --message "Emergency maintenance"

# Monitor system in real-time
./scripts/monitor_system.sh --continuous

# Run health checks
./scripts/ops_manager.sh health

# Performance monitoring
./scripts/performance_tuning.sh balanced --dry-run

# Disaster recovery assessment
./scripts/disaster_recovery.sh assess
```

### Service Ports
- **Data Collector**: 8001
- **Strategy Engine**: 8002
- **Risk Manager**: 8003
- **Trade Executor**: 8004
- **Scheduler**: 8005
- **Export Service**: 8006
- **Maintenance Service**: 8007
- **PostgreSQL**: 5432
- **Redis**: 6379
- **Prometheus**: 9090
- **Grafana**: 3000
- **Elasticsearch**: 9200
- **Kibana**: 5601

### Operational Scripts
- **Setup Environment**: `./scripts/setup_environment.sh`
- **Operations Manager**: `./scripts/ops_manager.sh`
- **System Monitor**: `./scripts/monitor_system.sh`
- **Performance Tuning**: `./scripts/performance_tuning.sh`
- **Disaster Recovery**: `./scripts/disaster_recovery.sh`
- **Security Audit**: `./scripts/security/audit_security.sh`
- **Operational Dashboard**: `python3 scripts/operational_dashboard.py`

### Environment Files
- **Development**: `config/environments/.env.development`
- **Staging**: `config/environments/.env.staging`  
- **Production**: `config/environments/.env.production`
- **PostgreSQL**: 5432
- **Redis**: 6379
- **Prometheus**: 9090
- **Grafana**: 3000

---

## Quick Start Guide

### Initial Setup
```bash
# 1. Setup environment
./scripts/setup_environment.sh production

# 2. Start all services
./scripts/ops_manager.sh start --env production

# 3. Verify system health
./scripts/ops_manager.sh health

# 4. Launch monitoring dashboard
python3 scripts/operational_dashboard.py --env production
```

### Daily Operations Checklist
```bash
# Morning checklist
./scripts/ops_manager.sh status                    # Check service status
./scripts/monitor_system.sh                       # Run system checks
python3 scripts/operational_dashboard.py --mode status  # Review metrics
./scripts/backup/backup.sh --type daily           # Create daily backup

# Evening checklist
./scripts/ops_manager.sh health                    # Run health checks
./scripts/security/audit_security.sh --type quick # Security audit
./scripts/cleanup_logs.sh                         # Clean old logs
```

---

## System Architecture

### Service Dependencies
```
Scheduler → Strategy Engine → Risk Manager → Trade Executor
    ↓              ↓              ↓              ↓
Data Collector ← PostgreSQL ← Redis ← Monitoring Stack
    ↓
Export Service
```

### Data Flow
1. **Data Collector** gathers market data → PostgreSQL/Redis
2. **Scheduler** triggers strategy execution
3. **Strategy Engine** analyzes data → generates signals
4. **Risk Manager** validates signals → applies risk controls
5. **Trade Executor** executes approved trades → records results
6. **Export Service** provides reporting and audit trails

---

### Core Components
- **Infrastructure Layer**: PostgreSQL, Redis
- **Trading Core**: Data Collector, Strategy Engine, Risk Manager, Trade Executor, Scheduler
- **Support Services**: Export Service, Maintenance Service
- **Monitoring Stack**: Prometheus, Grafana, Elasticsearch, Kibana, Alertmanager
- **Security Layer**: Rate limiting, audit logging, encryption

---

## Start/Stop Procedures

### Automated Operations (Recommended)

#### Start System
```bash
# Start all services
./scripts/ops_manager.sh start --env production

# Start only core trading services
./scripts/ops_manager.sh start-core

# Start monitoring services
./scripts/ops_manager.sh start-monitoring

# Start specific service
./scripts/ops_manager.sh start --service data_collector
```

#### Stop System
```bash
# Stop all services (with confirmation)
./scripts/ops_manager.sh stop --env production

# Force stop without confirmation
./scripts/ops_manager.sh stop --force

# Stop only core services
./scripts/ops_manager.sh stop-core

# Emergency shutdown (immediate)
./scripts/ops_manager.sh emergency-stop --force
```

#### Restart Services
```bash
# Restart all services
./scripts/ops_manager.sh restart

# Restart specific service
./scripts/ops_manager.sh restart --service strategy_engine

# Rolling restart (zero downtime)
./scripts/deployment/zero_downtime_deploy.sh --restart-only
```

### Manual Procedures (Emergency Only)

### Normal Startup (Development)
```bash
# 1. Ensure all environment variables are set
cp config/environments/.env.template .env
# Edit .env with your configuration

# 2. Start all services
docker-compose up -d

# 3. Verify all services are healthy
make health-check

# 4. Check logs for any errors
docker-compose logs --tail=50
```

### Normal Startup (Production)
```bash
# 1. Load production environment
export $(grep -v '^#' config/environments/.env.production | xargs)

# 2. Validate configuration
python scripts/config/validate_config.py --env production

# 3. Start with production compose file
docker-compose -f docker-compose.prod.yml up -d

# 4. Run deployment health checks
./scripts/deployment/deploy.sh production --health-check-only

# 5. Verify trading is enabled
curl -H "Authorization: Bearer $API_TOKEN" \
  http://localhost:8004/status
```

### Graceful Shutdown
```bash
# 1. Enter maintenance mode
curl -X POST http://localhost:8007/maintenance/enter \
  -H "Authorization: Bearer $API_TOKEN" \
  -d "mode=maintenance&reason=Planned shutdown"

# 2. Wait for current trades to complete (max 5 minutes)
sleep 300

# 3. Stop services in order
docker-compose stop scheduler
docker-compose stop trade_executor
docker-compose stop strategy_engine
docker-compose stop risk_manager
docker-compose stop data_collector

# 4. Stop supporting services
docker-compose stop export_service maintenance_service

# 5. Stop infrastructure
docker-compose stop prometheus grafana redis postgres
```

### Emergency Shutdown
```bash
# Immediate stop all trading
curl -X POST http://localhost:8007/maintenance/emergency-shutdown \
  -H "Authorization: Bearer $API_TOKEN" \
  -d "reason=Emergency situation"

# Force stop if needed
docker-compose down --timeout 30
```

---

#### Start Services Manually
```bash
cd /path/to/ai-trading-system

# 1. Start infrastructure
docker-compose -f docker-compose.prod.yml up -d postgres redis

# 2. Wait for infrastructure
sleep 30

# 3. Start core services
docker-compose -f docker-compose.prod.yml up -d data_collector strategy_engine risk_manager

# 4. Wait for core services
sleep 20

# 5. Start remaining services
docker-compose -f docker-compose.prod.yml up -d trade_executor scheduler export_service maintenance_service

# 6. Start monitoring
docker-compose -f docker-compose.prod.yml up -d prometheus grafana alertmanager
```

#### Stop Services Manually
```bash
# 1. Stop non-essential services
docker-compose -f docker-compose.prod.yml stop export_service maintenance_service

# 2. Stop trading services in reverse order
docker-compose -f docker-compose.prod.yml stop scheduler trade_executor risk_manager strategy_engine data_collector

# 3. Stop infrastructure
docker-compose -f docker-compose.prod.yml stop postgres redis

# 4. Stop monitoring
docker-compose -f docker-compose.prod.yml stop prometheus grafana alertmanager elasticsearch kibana
```

### Service Dependencies
```
postgres, redis (infrastructure)
    ↓
data_collector (market data)
    ↓
strategy_engine (signal generation)
    ↓
risk_manager (risk validation)
    ↓
trade_executor (order execution)
    ↓
scheduler (coordination)
```

---

## Maintenance Operations

### Entering Maintenance Mode
```bash
# Automated maintenance mode
./scripts/ops_manager.sh maintenance --enter --message "Scheduled maintenance"

# Manual maintenance steps
./scripts/ops_manager.sh stop-core                # Stop trading
curl -X POST http://localhost:8007/maintenance/enter  # Enable maintenance mode
```

### Exiting Maintenance Mode
```bash
# Automated exit
./scripts/ops_manager.sh maintenance --exit

# Manual exit
curl -X POST http://localhost:8007/maintenance/exit   # Disable maintenance mode
./scripts/ops_manager.sh start-core                   # Start trading
./scripts/ops_manager.sh health                       # Verify health
```

### Maintenance Tasks

#### Weekly Maintenance
```bash
# 1. Create full backup
./scripts/backup/backup.sh --type weekly --compress --verify

# 2. Security audit
./scripts/security/audit_security.sh --type full --auto-remediate

# 3. Performance review
./scripts/performance_tuning.sh balanced --dry-run

# 4. Log cleanup
./scripts/cleanup_logs.sh

# 5. Update system packages (if needed)
sudo apt update && sudo apt upgrade -y

# 6. Restart services for fresh state
./scripts/ops_manager.sh restart
```

#### Monthly Maintenance
```bash
# 1. Comprehensive security audit
./scripts/security/audit_security.sh --type compliance --auto-remediate

# 2. Disaster recovery test
./scripts/disaster_recovery.sh test --scenario service_crash

# 3. Performance optimization
./scripts/performance_tuning.sh aggressive --apply

# 4. Backup verification
./scripts/backup/test_restore.sh --latest

# 5. Documentation review
# Review and update operational procedures
```

### Configuration Updates
```bash
# 1. Validate new configuration
python3 scripts/config/validate_config.py --env production --strict

# 2. Update configurations without restart
./scripts/ops_manager.sh update

# 3. If restart required
./scripts/deployment/zero_downtime_deploy.sh --config-only
```

### Scheduled Maintenance Window
**Recommended Schedule**: Sunday 2:00 AM EST (market closed)

#### Pre-maintenance Checklist
- [ ] Notify stakeholders 24 hours in advance
- [ ] Verify backup completion
- [ ] Review upcoming economic events
- [ ] Prepare rollback plan
- [ ] Test maintenance procedures in staging

#### Maintenance Procedure
1. **Enter Maintenance Mode**
   ```bash
   ./scripts/deployment/deploy.sh maintenance-mode --duration 60
   ```

2. **Perform Updates**
   ```bash
   # Update system packages
   ./scripts/deployment/deploy.sh production --update-system
   
   # Update application
   ./scripts/deployment/deploy.sh production --update-app
   
   # Run database migrations
   ./scripts/deployment/migrate.sh production
   ```

3. **Validate Changes**
   ```bash
   # Run health checks
   ./scripts/deployment/deploy.sh production --health-check
   
   # Run integration tests
   docker-compose -f docker-compose.test.yml run tests
   ```

4. **Exit Maintenance Mode**
   ```bash
   curl -X POST http://localhost:8007/maintenance/exit \
     -H "Authorization: Bearer $API_TOKEN"
   ```

#### Post-maintenance Checklist
- [ ] Verify all services are healthy
- [ ] Check trading functionality
- [ ] Monitor for 30 minutes
- [ ] Update maintenance log
- [ ] Notify stakeholders of completion

### Database Maintenance
```bash
# Analyze database performance
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
  FROM pg_stat_user_tables
  ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC;
"

# Vacuum and analyze
docker-compose exec postgres psql -U trader -d trading_system -c "
  VACUUM ANALYZE;
"

# Check database size
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT pg_size_pretty(pg_database_size('trading_system'));
"
```

---

---

## Troubleshooting Guide

### Service Issues

#### Service Won't Start
```bash
# 1. Check service logs
./scripts/ops_manager.sh logs --service <service_name>

# 2. Check container status
docker-compose -f docker-compose.prod.yml ps <service_name>

# 3. Check resource usage
./scripts/monitor_system.sh

# 4. Force recreate service
docker-compose -f docker-compose.prod.yml up -d --force-recreate <service_name>

# 5. Check configuration
python3 scripts/config/validate_config.py --env production
```

#### Database Connection Issues
```bash
# 1. Check database status
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U trading_user -d trading_db

# 2. Check database logs
./scripts/ops_manager.sh logs --service postgres

# 3. Check connections
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "SELECT count(*) FROM pg_stat_activity;"

# 4. Restart database (if safe)
./scripts/ops_manager.sh restart --service postgres

# 5. Emergency database recovery
./scripts/disaster_recovery.sh recover --scenario database_failure
```

#### High Resource Usage
```bash
# 1. Identify resource hogs
./scripts/monitor_system.sh
docker stats --no-stream

# 2. Check for memory leaks
./scripts/performance_tuning.sh conservative --apply

# 3. Clean up resources
./scripts/ops_manager.sh cleanup

# 4. Emergency resource recovery
./scripts/disaster_recovery.sh recover --scenario memory_exhaustion
```

#### Network Connectivity Issues
```bash
# 1. Check service interconnectivity
./scripts/ops_manager.sh health

# 2. Test external connectivity
curl -I https://api.exchange.com  # Replace with actual exchange

# 3. Check Docker networks
docker network ls
docker network inspect <network_name>

# 4. Recreate network
./scripts/disaster_recovery.sh recover --scenario network_partition
```

### Performance Issues

#### Slow Trade Execution
```bash
# 1. Check trade executor logs
./scripts/ops_manager.sh logs --service trade_executor

# 2. Monitor database performance
./scripts/monitor_system.sh

# 3. Check risk manager latency
curl http://localhost:8003/health

# 4. Optimize performance
./scripts/performance_tuning.sh aggressive --apply
```

#### High Latency
```bash
# 1. Check network latency
ping api.exchange.com

# 2. Monitor API response times
./scripts/monitor_performance.sh --once

# 3. Check database query performance
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# 4. Tune performance
./scripts/performance_tuning.sh maximum --apply
```

### Service Won't Start

#### Symptoms
- Service container exits immediately
- Health check fails
- Database connection errors

#### Diagnosis Steps
1. **Check logs**
   ```bash
   docker-compose logs [service_name]
   ```

2. **Verify configuration**
   ```bash
   python scripts/config/validate_config.py --env $ENVIRONMENT
   ```

3. **Check dependencies**
   ```bash
   docker-compose ps
   ```

4. **Test database connection**
   ```bash
   docker-compose exec postgres pg_isready -U trader
   ```

#### Resolution
- **Bad configuration**: Fix environment variables
- **Database issues**: Check PostgreSQL logs and restart if needed
- **Redis issues**: Check Redis logs and restart if needed
- **Port conflicts**: Change ports in docker-compose.yml

### High Memory Usage

#### Symptoms
- OOMKilled containers
- Slow response times
- System alerts

#### Diagnosis
```bash
# Check container memory usage
docker stats --no-stream

# Check service-specific metrics
curl http://localhost:9090/api/v1/query?query=container_memory_usage_bytes

# Review memory-intensive queries
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT query, calls, total_time, mean_time, rows
  FROM pg_stat_statements
  ORDER BY total_time DESC
  LIMIT 10;
"
```

#### Resolution
1. **Immediate**: Restart high-memory services
2. **Short-term**: Increase memory limits in docker-compose.yml
3. **Long-term**: Optimize queries and implement caching

### Trading Execution Issues

#### Symptoms
- Trades not executing
- Order rejections
- API errors from broker

#### Diagnosis
1. **Check trade executor logs**
   ```bash
   docker-compose logs trade_executor --tail=100
   ```

2. **Verify broker API status**
   ```bash
   curl -H "Authorization: Bearer $ALPACA_API_KEY" \
     https://api.alpaca.markets/v2/account
   ```

3. **Check risk manager decisions**
   ```bash
   curl http://localhost:8003/risk/recent-decisions
   ```

#### Resolution
- **API issues**: Check API keys and rate limits
- **Risk blocks**: Review and adjust risk parameters
- **Market hours**: Verify trading session times
- **Account issues**: Check broker account status

---

### Data Issues

#### Missing Market Data
```bash
# 1. Check data collector
./scripts/ops_manager.sh logs --service data_collector

# 2. Verify exchange connectivity
curl http://localhost:8001/health

# 3. Check data storage
ls -la data/parquet/

# 4. Restart data collection
./scripts/ops_manager.sh restart --service data_collector
```

#### Export Failures
```bash
# 1. Check export service
./scripts/ops_manager.sh logs --service export_service

# 2. Check disk space
df -h

# 3. Check export directory
ls -la data/exports/

# 4. Manual export
curl -X POST http://localhost:8006/export/tradenote
```

---

## Performance Tuning

### Performance Profiles
```bash
# Conservative (low resource usage)
./scripts/performance_tuning.sh conservative --apply

# Balanced (recommended for most scenarios)
./scripts/performance_tuning.sh balanced --apply

# Aggressive (high performance trading)
./scripts/performance_tuning.sh aggressive --apply

# Maximum (absolute performance)
./scripts/performance_tuning.sh maximum --apply
```

### Custom Performance Tuning
```bash
# Custom resource limits
./scripts/performance_tuning.sh balanced \
  --cpu-limit 85 \
  --memory-limit 4g \
  --workers 8 \
  --apply

# Performance analysis
./scripts/performance_tuning.sh balanced --dry-run --verbose

# Benchmark system
./scripts/performance_tuning.sh balanced --apply
# Then run benchmark when prompted
```

### Database Performance
```bash
# Optimize PostgreSQL
# Edit config/postgresql/performance.conf
# Restart database: ./scripts/ops_manager.sh restart --service postgres

# Monitor database performance
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
FROM pg_stat_user_tables 
ORDER BY n_tup_ins DESC LIMIT 10;"

# Check slow queries
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;"
```

### Redis Performance
```bash
# Monitor Redis performance
docker-compose -f docker-compose.prod.yml exec redis redis-cli info memory
docker-compose -f docker-compose.prod.yml exec redis redis-cli info stats

# Check Redis slow log
docker-compose -f docker-compose.prod.yml exec redis redis-cli slowlog get 10
```

### Database Optimization
```sql
-- Check slow queries
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
WHERE calls > 100
ORDER BY total_time DESC
LIMIT 20;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan < 100
ORDER BY idx_scan;

-- Identify missing indexes
SELECT schemaname, tablename, seq_scan, seq_tup_read, 
       seq_tup_read / seq_scan as avg_tup_read
FROM pg_stat_user_tables
WHERE seq_scan > 1000
ORDER BY seq_tup_read DESC;
```

### Redis Optimization
```bash
# Check Redis memory usage
docker-compose exec redis redis-cli info memory

# Check key statistics
docker-compose exec redis redis-cli info keyspace

# Monitor slow operations
docker-compose exec redis redis-cli slowlog get 10
```

### Application Performance
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null http://localhost:8001/health

# Monitor container resources
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])
```

### Performance Tuning Checklist
- [ ] Database indexes optimized
- [ ] Redis memory usage < 80%
- [ ] Application response times < 2s
- [ ] Container CPU usage < 70%
- [ ] Container memory usage < 80%
- [ ] Network latency < 100ms

---

---

## Disaster Recovery

### Disaster Recovery Assessment
```bash
# Assess current system status
./scripts/disaster_recovery.sh assess

# Check disaster recovery readiness
./scripts/disaster_recovery.sh prepare --env production
```

### Disaster Scenarios and Recovery

#### Database Failure
```bash
# Automated recovery
./scripts/disaster_recovery.sh recover --scenario database_failure --auto-failover

# Manual recovery steps
./scripts/ops_manager.sh stop-core
./scripts/backup/restore.sh --backup-id latest --database-only --force
./scripts/ops_manager.sh start-core
./scripts/ops_manager.sh health
```

#### Service Crashes
```bash
# Automated recovery
./scripts/disaster_recovery.sh recover --scenario service_crash

# Manual service restart
./scripts/ops_manager.sh restart --service <crashed_service>
./scripts/ops_manager.sh health
```

#### Complete System Failure
```bash
# Emergency recovery
./scripts/disaster_recovery.sh recover --scenario complete_system_failure --force

# Manual complete recovery
./scripts/ops_manager.sh emergency-stop --force
./scripts/backup/restore.sh --backup-id latest --type full --force
./scripts/ops_manager.sh start
./scripts/ops_manager.sh health
```

#### Security Breach
```bash
# Immediate response
./scripts/disaster_recovery.sh recover --scenario security_breach --force

# Manual security response
./scripts/ops_manager.sh stop                     # Stop all services
./scripts/security/rotate_secrets.sh --emergency  # Rotate all secrets
./scripts/security/audit_security.sh --type full  # Full security audit
# Review logs and investigate breach
# Apply security patches
./scripts/ops_manager.sh start                     # Restart when safe
```

### Failover Procedures
```bash
# Automated failover
./scripts/disaster_recovery.sh failover --auto-failover

# Manual failover to backup system
./scripts/ops_manager.sh stop
# Switch to backup infrastructure
./scripts/ops_manager.sh start --env backup
```

### Testing Disaster Recovery
```bash
# Test specific scenario
./scripts/disaster_recovery.sh test --scenario database_failure --dry-run

# Test complete DR procedures
./scripts/disaster_recovery.sh test --scenario complete_system_failure
```

### Recovery Time Objectives (RTO)
- **Critical Services**: 15 minutes
- **Full System**: 30 minutes
- **Historical Data**: 2 hours

### Recovery Point Objectives (RPO)
- **Trade Data**: 5 minutes
- **Configuration**: 1 hour
- **Historical Data**: 24 hours

### Disaster Scenarios

#### Complete System Failure
1. **Assess Damage**
   ```bash
   # Check what's available
   docker-compose ps
   systemctl status docker
   df -h
   ```

2. **Emergency Restore**
   ```bash
   # Restore from latest backup
   ./scripts/backup/restore.sh --emergency --latest
   
   # Start minimal services
   docker-compose up -d postgres redis
   
   # Restore database
   ./scripts/backup/restore_db.sh --latest
   
   # Start trading services
   docker-compose up -d data_collector risk_manager trade_executor
   ```

3. **Verify Recovery**
   ```bash
   # Check data integrity
   ./scripts/deployment/deploy.sh production --validate-data
   
   # Test critical functions
   python tests/integration/test_critical_path.py
   ```

#### Database Corruption
1. **Stop all services**
   ```bash
   docker-compose stop
   ```

2. **Assess corruption**
   ```bash
   docker-compose up -d postgres
   docker-compose exec postgres pg_dump trading_system > /tmp/db_check.sql
   ```

3. **Restore from backup**
   ```bash
   ./scripts/backup/restore_db.sh --date $(date -d "yesterday" +%Y-%m-%d)
   ```

#### Network Partition
1. **Identify affected services**
2. **Enable degraded mode**
   ```bash
   curl -X POST http://localhost:8007/maintenance/enter \
     -d "mode=read_only&reason=Network issues"
   ```
3. **Monitor and wait for recovery**
4. **Resume normal operations when resolved**

---

---

## Monitoring and Alerting

### Real-time Monitoring
```bash
# Interactive dashboard
python3 scripts/operational_dashboard.py --env production

# Continuous system monitoring
./scripts/monitor_system.sh --continuous --enable-slack --enable-email

# Performance monitoring
./scripts/monitor_performance.sh
```

### Monitoring Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **Alertmanager**: http://localhost:9093

### Key Metrics to Monitor
```bash
# System health score (aim for >95%)
curl http://localhost:8007/health | jq '.health_score'

# Trading performance
curl http://localhost:8002/metrics | grep trading_

# Resource utilization
curl http://localhost:9090/api/v1/query?query=rate(cpu_usage_total[5m])

# Error rates
curl http://localhost:9090/api/v1/query?query=rate(errors_total[5m])
```

### Alert Configuration
```bash
# Configure alert thresholds
./scripts/monitor_system.sh \
  --cpu-threshold 85 \
  --memory-threshold 90 \
  --disk-threshold 95 \
  --continuous

# Test alert notifications
./scripts/monitor_system.sh \
  --enable-slack \
  --enable-email \
  --test-alerts
```

### Log Analysis
```bash
# View service logs
./scripts/ops_manager.sh logs --service <service_name>

# Search logs for errors
grep -r "ERROR\|CRITICAL" logs/services/

# Analyze performance logs
tail -f logs/performance/performance_$(date +%Y%m%d).log

# Security log analysis
grep -r "authentication\|authorization" logs/security/
```

### Key Metrics to Monitor

#### System Health
- **CPU Usage**: < 70% normal, > 90% critical
- **Memory Usage**: < 80% normal, > 95% critical
- **Disk Usage**: < 80% normal, > 90% critical
- **Network Latency**: < 100ms normal, > 500ms critical

#### Trading Metrics
- **Trade Execution Time**: < 1s normal, > 5s critical
- **Order Fill Rate**: > 95% normal, < 90% critical
- **Daily P&L**: Monitor for unusual losses
- **Position Sizes**: Monitor for limit breaches

#### Application Metrics
- **API Response Time**: < 200ms normal, > 2s critical
- **Error Rate**: < 1% normal, > 5% critical
- **Database Connections**: < 80% pool normal
- **Queue Depth**: < 100 normal, > 1000 critical

### Alert Escalation

#### Level 1 - Information
- Daily performance reports
- Backup completion notifications
- Scheduled maintenance reminders

#### Level 2 - Warning
- High resource usage (80%+ for 5 minutes)
- Slow response times (> 1s)
- Non-critical service degradation

#### Level 3 - Critical
- Service failures
- Database connection issues
- Trading execution failures
- Security incidents

#### Level 4 - Emergency
- Complete system failure
- Data corruption detected
- Large financial losses
- Security breaches

### Grafana Dashboards
1. **System Overview**: http://localhost:3000/d/trading-overview
2. **Trading Performance**: http://localhost:3000/d/trading-performance
3. **Risk Metrics**: http://localhost:3000/d/risk-dashboard
4. **Infrastructure**: http://localhost:3000/d/infrastructure

---

---

## Security Operations

### Security Auditing
```bash
# Full security audit
./scripts/security/audit_security.sh --type full --auto-remediate

# Quick security scan
./scripts/security/audit_security.sh --type quick

# Compliance audit
./scripts/security/audit_security.sh --type compliance --env production

# Vulnerability assessment
./scripts/security/audit_security.sh --type vulnerability --severity high
```

### Security Hardening
```bash
# System hardening
./scripts/security/harden_system.sh

# Firewall setup
./scripts/security/setup_firewall.sh

# SSL certificate management
./scripts/security/manage_certificates.sh --renew
```

### Incident Response
```bash
# Security incident response
./scripts/disaster_recovery.sh recover --scenario security_breach

# Audit trail analysis
./scripts/security/audit_security.sh --type full --output json

# Secret rotation
./scripts/security/rotate_secrets.sh --all
```

### Rate Limiting Management
```bash
# Check current rate limits
curl http://localhost:8001/rate-limit/status

# Update rate limits
# Edit config/security/rate_limiting.yml
./scripts/ops_manager.sh update

# Monitor rate limiting
grep "rate_limit" logs/services/*/api.log
```

### Daily Security Checks
```bash
# Check for failed login attempts
grep "authentication failed" logs/*.log

# Review audit logs
curl -H "Authorization: Bearer $API_TOKEN" \
  http://localhost:8006/export/audit?start_date=$(date -d "yesterday" +%Y-%m-%d)

# Check for unusual trading patterns
python scripts/security/detect_anomalies.py --days 1
```

### Security Incident Response

#### Suspected Breach
1. **Immediate Actions**
   ```bash
   # Enter emergency shutdown
   curl -X POST http://localhost:8007/maintenance/emergency-shutdown \
     -d "reason=Security incident"
   
   # Preserve evidence
   ./scripts/backup/emergency_backup.sh --preserve-logs
   
   # Change API keys
   ./scripts/security/rotate_keys.sh --emergency
   ```

2. **Investigation**
   - Review audit logs
   - Check access patterns
   - Analyze trade history
   - Contact broker to verify trades

3. **Recovery**
   - Implement additional security measures
   - Update access controls
   - Resume operations with enhanced monitoring

### Access Control Management
```bash
# Generate new API tokens
python scripts/security/generate_tokens.py --user admin --expiry 30d

# Revoke compromised tokens
python scripts/security/revoke_token.py --token-id 12345

# Audit user access
python scripts/security/audit_access.py --days 7
```

---

---

## Backup and Recovery

### Backup Operations
```bash
# Daily automated backup
./scripts/backup/backup.sh --type daily --compress --verify

# Manual backup
./scripts/backup/backup.sh --type manual --compress

# Emergency backup
./scripts/backup/backup.sh --type emergency --no-verify

# Database-only backup
./scripts/backup/backup.sh --type database --compress

# Configuration backup
./scripts/backup/backup.sh --type config
```

### Backup Verification
```bash
# Test latest backup
./scripts/backup/test_restore.sh --latest

# Test specific backup
./scripts/backup/test_restore.sh --backup-id backup_20240101_120000

# Verify backup integrity
./scripts/backup/backup.sh --verify-only --backup-id backup_20240101_120000
```

### Restoration Procedures
```bash
# Full system restore
./scripts/backup/restore.sh --backup-id latest --type full --force

# Database-only restore
./scripts/backup/restore.sh --backup-id latest --database-only

# Configuration restore
./scripts/backup/restore.sh --backup-id latest --configs-only

# Selective data restore
./scripts/backup/restore.sh --backup-id latest --data-only
```

### Backup Scheduling
```bash
# Setup automated backups (already configured in setup)
crontab -l | grep backup

# Manual cron setup
echo "0 2 * * * /path/to/ai-trading-system/scripts/backup/run_backups.sh" | crontab -
```

### Backup Types

#### Daily Automated Backups
- **Time**: 2:00 AM EST
- **Includes**: Database, configuration, parquet files
- **Retention**: 30 days local, 90 days S3
- **Validation**: Automatic restore test

#### On-Demand Backups
```bash
# Full system backup
./scripts/backup/backup.sh --full --compress

# Database only
./scripts/backup/backup.sh --database-only

# Configuration only
./scripts/backup/backup.sh --config-only
```

#### Pre-deployment Backups
```bash
# Automatic with deployment script
./scripts/deployment/deploy.sh production  # Includes backup

# Manual pre-deployment backup
./scripts/backup/backup.sh --pre-deployment --tag $(git rev-parse HEAD)
```

### Restore Procedures

#### Point-in-Time Recovery
```bash
# List available backups
./scripts/backup/list_backups.sh

# Restore to specific timestamp
./scripts/backup/restore.sh --timestamp "2024-01-15 14:30:00"

# Restore specific backup
./scripts/backup/restore.sh --backup-id backup_20240115_143000
```

#### Selective Restore
```bash
# Restore only database
./scripts/backup/restore_db.sh --backup-id backup_20240115_143000

# Restore only configuration
./scripts/backup/restore_config.sh --backup-id backup_20240115_143000

# Restore only parquet files
./scripts/backup/restore_data.sh --backup-id backup_20240115_143000
```

---

---

## Emergency Procedures

### Critical System Failure
```bash
# 1. Immediate assessment
./scripts/disaster_recovery.sh assess

# 2. Emergency shutdown
./scripts/ops_manager.sh emergency-stop --force

# 3. Notify stakeholders
./scripts/disaster_recovery.sh recover --scenario complete_system_failure --notify all

# 4. Attempt recovery
./scripts/disaster_recovery.sh recover --scenario complete_system_failure --auto-failover
```

### Trading Halt Procedures
```bash
# 1. Immediate halt
curl -X POST http://localhost:8007/emergency/halt

# 2. Close all positions (if safe)
curl -X POST http://localhost:8004/emergency/close-all-positions

# 3. Enter maintenance mode
./scripts/ops_manager.sh maintenance --enter --message "Emergency halt"

# 4. Investigate and resolve issues
./scripts/ops_manager.sh logs --service trade_executor
./scripts/ops_manager.sh logs --service risk_manager
```

### Data Corruption Response
```bash
# 1. Stop data writing services
./scripts/ops_manager.sh stop --service "data_collector trade_executor"

# 2. Assess corruption
./scripts/disaster_recovery.sh assess

# 3. Recovery procedures
./scripts/disaster_recovery.sh recover --scenario data_corruption

# 4. Verify data integrity
./scripts/backup/test_restore.sh --latest
```

### Security Incident Response
```bash
# 1. Immediate lockdown
./scripts/ops_manager.sh stop
./scripts/security/setup_firewall.sh --lockdown

# 2. Forensic backup
./scripts/backup/backup.sh --type forensic

# 3. Security audit
./scripts/security/audit_security.sh --type full

# 4. Incident response
./scripts/disaster_recovery.sh recover --scenario security_breach
```

### Recovery Validation
```bash
# 1. Health checks
./scripts/ops_manager.sh health

# 2. Integration tests
python3 scripts/run_tests.py --integration-tests

# 3. Performance validation
./scripts/performance_tuning.sh balanced --dry-run

# 4. Security validation
./scripts/security/audit_security.sh --type quick
```

### Financial Emergency (Large Losses)
1. **Immediate Actions**
   ```bash
   # Stop all trading immediately
   curl -X POST http://localhost:8004/trading/emergency-stop
   
   # Close all positions (if safe to do so)
   curl -X POST http://localhost:8004/positions/close-all
   
   # Enter emergency maintenance
   curl -X POST http://localhost:8007/maintenance/emergency-shutdown \
     -d "reason=Financial emergency - large losses detected"
   ```

2. **Assessment**
   - Review recent trades and positions
   - Check for system errors or bugs
   - Analyze market conditions
   - Verify trade execution accuracy

3. **Communication**
   - Notify stakeholders immediately
   - Document incident timeline
   - Prepare incident report

### Technical Emergency (System Compromise)
1. **Isolate System**
   ```bash
   # Disconnect from external networks
   docker network disconnect bridge trading_network
   
   # Stop all external API calls
   ./scripts/security/block_external_access.sh
   ```

2. **Preserve Evidence**
   ```bash
   # Create forensic backup
   ./scripts/backup/forensic_backup.sh
   
   # Capture system state
   docker-compose ps > incident_$(date +%Y%m%d_%H%M%S).log
   docker-compose logs >> incident_$(date +%Y%m%d_%H%M%S).log
   ```

3. **Recovery**
   - Restore from clean backup
   - Implement additional security measures
   - Resume operations with enhanced monitoring

### Data Corruption Emergency
1. **Stop Data Writes**
   ```bash
   # Set system to read-only
   curl -X POST http://localhost:8007/maintenance/enter \
     -d "mode=read_only&reason=Data corruption detected"
   ```

2. **Assess Corruption**
   ```bash
   # Check database integrity
   docker-compose exec postgres pg_dump --schema-only trading_system
   
   # Validate parquet files
   python scripts/data/validate_parquet.py --all
   ```

3. **Recovery Strategy**
   - Restore from latest clean backup
   - Replay transactions if possible
   - Validate data integrity
   - Resume normal operations

---

### Escalation Procedures
1. **Level 1**: Service restart, basic troubleshooting
2. **Level 2**: System recovery, backup restoration
3. **Level 3**: Disaster recovery, complete rebuild
4. **Level 4**: External support, vendor escalation

---

## Common Issues and Solutions

### Docker Issues

#### "Container already exists" Error
```bash
./scripts/ops_manager.sh stop --force
docker-compose -f docker-compose.prod.yml rm -f
./scripts/ops_manager.sh start
```

#### "Port already in use" Error
```bash
# Find process using port
sudo netstat -tulpn | grep :8001

# Kill process if safe
sudo kill -9 <PID>

# Or use different ports
# Edit docker-compose.yml port mappings
```

#### "No space left on device" Error
```bash
./scripts/disaster_recovery.sh recover --scenario disk_full
```

### Database Issues

#### Connection Pool Exhausted
```bash
# Check active connections
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "SELECT count(*) FROM pg_stat_activity;"

# Kill long-running queries
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';"

# Restart services to reset connections
./scripts/ops_manager.sh restart-core
```

#### Database Lock Issues
```bash
# Check for locks
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "SELECT * FROM pg_locks WHERE NOT granted;"

# Kill blocking queries
docker-compose -f docker-compose.prod.yml exec postgres psql -U trading_user -d trading_db -c "SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND waiting;"
```

### Application Issues

#### Memory Leaks
```bash
# Monitor memory usage
./scripts/monitor_system.sh --memory-threshold 80 --continuous

# Restart affected service
./scripts/ops_manager.sh restart --service <service_name>

# Apply memory optimization
./scripts/performance_tuning.sh conservative --apply
```

#### API Timeouts
```bash
# Check API health
./scripts/ops_manager.sh health

# Increase timeout settings
# Edit config/performance/application.yml
./scripts/ops_manager.sh update

# Scale service if needed
./scripts/ops_manager.sh scale --service <service_name> --replicas 2
```

### Monitoring Issues

#### Metrics Not Collecting
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Restart monitoring stack
./scripts/ops_manager.sh restart-monitoring

# Check service metrics endpoints
curl http://localhost:8001/metrics
curl http://localhost:8002/metrics
```

#### Alerts Not Firing
```bash
# Check Alertmanager configuration
curl http://localhost:9093/api/v1/alerts

# Test alert rules
curl http://localhost:9090/api/v1/rules

# Send test alert
curl -X POST http://localhost:9093/api/v1/alerts
```

---

## Appendices

### A. Service URLs and Endpoints
```
Core Services:
- Data Collector:    http://localhost:8001
- Strategy Engine:   http://localhost:8002  
- Risk Manager:      http://localhost:8003
- Trade Executor:    http://localhost:8004
- Scheduler:         http://localhost:8005
- Export Service:    http://localhost:8006
- Maintenance:       http://localhost:8007

Monitoring:
- Grafana:           http://localhost:3000
- Prometheus:        http://localhost:9090
- Kibana:            http://localhost:5601
- Alertmanager:      http://localhost:9093

Health Endpoints:
- /health            Basic health check
- /health/detailed   Detailed health info
- /metrics           Prometheus metrics
- /status            Service status
```

### B. Configuration Files
```
Environment:
- config/environments/.env.development
- config/environments/.env.staging
- config/environments/.env.production

Docker:
- docker-compose.yml (development)
- docker-compose.prod.yml (production)
- docker-compose.test.yml (testing)

Performance:
- config/performance/application.yml
- config/postgresql/performance.conf
- config/redis/performance.conf

Security:
- config/security/rate_limiting.yml
- config/ssl/trading-system.crt
- data/secrets/api_keys.json

Monitoring:
- monitoring/prometheus/prometheus.yml
- monitoring/grafana/provisioning/
- monitoring/alertmanager/alertmanager.yml
```

### C. Log Locations
```
Service Logs:
- logs/services/<service_name>/
- Docker logs: docker-compose logs <service>

Operational Logs:
- logs/operations/
- logs/deployment/
- logs/backup/
- logs/monitoring/
- logs/security/

Performance Logs:
- logs/performance/
- logs/performance/trading_metrics.log
- logs/performance/alerts.log
```

### D. Recovery Time Objectives (RTO)
- **Service Restart**: < 5 minutes
- **Database Recovery**: < 30 minutes  
- **Full System Recovery**: < 60 minutes
- **Disaster Recovery**: < 4 hours

### E. Recovery Point Objectives (RPO)
- **Database**: < 15 minutes (automatic backups)
- **Configuration**: < 24 hours (daily backups)
- **Trade Data**: < 5 minutes (real-time replication)
- **Market Data**: < 1 minute (continuous collection)

### Issue: Service Fails to Start

**Symptoms**: Container exits with code 1
**Cause**: Usually configuration or dependency issues

**Solution**:
```bash
# Check configuration
python scripts/config/validate_config.py --env $ENVIRONMENT

# Check dependencies
docker-compose ps

# Review logs
docker-compose logs [service_name] --tail=50

# Restart dependencies
docker-compose restart postgres redis
```

### Issue: High Database Load

**Symptoms**: Slow queries, high CPU on PostgreSQL
**Cause**: Inefficient queries or missing indexes

**Solution**:
```bash
# Identify slow queries
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT query, calls, total_time, mean_time
  FROM pg_stat_statements
  ORDER BY total_time DESC
  LIMIT 10;
"

# Add missing indexes
python scripts/database/analyze_queries.py --suggest-indexes

# Optimize configuration
vim config/postgres/postgresql.conf
docker-compose restart postgres
```

### Issue: Redis Memory Issues

**Symptoms**: Redis evictions, memory alerts
**Cause**: Too much cached data

**Solution**:
```bash
# Check memory usage
docker-compose exec redis redis-cli info memory

# Clear non-essential caches
docker-compose exec redis redis-cli FLUSHDB 1

# Adjust cache TTL
vim config/environments/.env.production
docker-compose restart
```

### Issue: Trading Stops Working

**Symptoms**: No new trades, signals not executing
**Cause**: Risk limits, API issues, or market conditions

**Solution**:
```bash
# Check risk manager status
curl http://localhost:8003/risk/status

# Verify broker connectivity
curl http://localhost:8004/broker/status

# Review recent risk decisions
curl http://localhost:8003/risk/recent-decisions

# Check market data freshness
curl http://localhost:8001/data/status
```

---

## Performance Monitoring

### Daily Performance Review
```bash
# Generate performance report
curl -X POST http://localhost:8006/export/performance \
  -H "Authorization: Bearer $API_TOKEN" \
  -d "start_date=$(date -d "yesterday" +%Y-%m-%d)"

# Check system metrics
curl http://localhost:9090/api/v1/query?query=up

# Review error rates
grep -c "ERROR" logs/$(date +%Y-%m-%d)*.log
```

### Weekly Performance Analysis
1. Review trading performance metrics
2. Analyze resource utilization trends
3. Check backup and restore times
4. Review security audit logs
5. Update performance baselines

### Monthly Performance Optimization
1. Database maintenance and optimization
2. Review and update alerting thresholds
3. Capacity planning review
4. Security assessment
5. Disaster recovery testing

---

## Maintenance Schedule

### Daily (Automated)
- Health checks every 5 minutes
- Backup verification
- Log rotation
- Performance metric collection

### Weekly (Automated)
- Database statistics update
- Security audit log review
- Capacity trend analysis
- Backup restore testing

### Monthly (Manual)
- Full system health review
- Security assessment
- Performance optimization
- Disaster recovery testing
- Documentation updates

### Quarterly (Manual)
- Full disaster recovery drill
- Security penetration testing
- Performance benchmark update
- Configuration audit
- Vendor relationship review

---

## Contacts and Escalation

### On-Call Rotation
- **Week 1**: Primary engineer
- **Week 2**: Secondary engineer
- **Holidays**: Escalation team

### Escalation Path
1. **Level 1**: On-call engineer (0-15 minutes)
2. **Level 2**: Senior engineer (15-30 minutes)
3. **Level 3**: Engineering manager (30-60 minutes)
4. **Level 4**: VP Engineering (1+ hours)

### External Contacts
- **Broker Support**: [Contact information]
- **Data Provider Support**: [Contact information]
- **Cloud Provider Support**: [Contact information]
- **Legal/Compliance**: [Contact information]

---

## Documentation Updates

This runbook should be updated:
- After any significant system changes
- Following incident resolution
- During quarterly reviews
- When procedures change

**Last Updated**: [Current Date]
**Version**: 1.0.0
**Next Review**: [Date + 3 months]

---

## Appendix

### Useful Commands Reference
```bash
# Service management
docker-compose up -d [service]
docker-compose restart [service]
docker-compose logs -f [service]

# System monitoring
docker stats
docker system df
docker system prune -f

# Database operations
docker-compose exec postgres psql -U trader -d trading_system
docker-compose exec postgres pg_dump trading_system > backup.sql

# Redis operations
docker-compose exec redis redis-cli
docker-compose exec redis redis-cli monitor

# Export operations
curl -X POST http://localhost:8006/export/tradenote
curl -X POST http://localhost:8006/export/performance
curl -X POST http://localhost:8006/export/full
```

### Log Locations
- **Application Logs**: `/app/logs/`
- **Container Logs**: `docker-compose logs`
- **System Logs**: `/var/log/`
- **Audit Logs**: Database `audit_logs` table
- **Access Logs**: Reverse proxy logs

### Configuration Files
- **Environment**: `config/environments/.env.*`
- **Docker**: `docker-compose*.yml`
- **Database**: `config/postgres/`
- **Monitoring**: `monitoring/`
- **Backups**: `scripts/backup/`

### Recovery Contacts
Keep this information readily available:
- Emergency contact numbers
- Broker emergency contacts
- Cloud provider support
- Database vendor support
- Security incident response team