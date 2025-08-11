# AI Trading System Troubleshooting Guide

## Table of Contents
1. [Quick Diagnosis](#quick-diagnosis)
2. [Service-Specific Issues](#service-specific-issues)
3. [Database Issues](#database-issues)
4. [Redis Issues](#redis-issues)
5. [Network and Connectivity](#network-and-connectivity)
6. [Performance Issues](#performance-issues)
7. [Trading Execution Problems](#trading-execution-problems)
8. [Data Quality Issues](#data-quality-issues)
9. [Security and Access Issues](#security-and-access-issues)
10. [Deployment Issues](#deployment-issues)
11. [Emergency Scenarios](#emergency-scenarios)

---

## Quick Diagnosis

### Health Check Commands
```bash
# Overall system health
curl http://localhost:8007/status

# Individual service health
curl http://localhost:8001/health  # Data Collector
curl http://localhost:8002/health  # Strategy Engine
curl http://localhost:8003/health  # Risk Manager
curl http://localhost:8004/health  # Trade Executor
curl http://localhost:8005/health  # Scheduler
curl http://localhost:8006/health  # Export Service
curl http://localhost:8007/health  # Maintenance Service

# Check all services at once
for port in 8001 8002 8003 8004 8005 8006 8007; do
  echo -n "Port $port: "
  curl -s -w "%{http_code}" http://localhost:$port/health -o /dev/null || echo "FAILED"
done
```

### Container Status
```bash
# Check all containers
docker-compose ps

# Check container resource usage
docker stats --no-stream

# Check container logs
docker-compose logs --tail=50 -f

# Check specific service logs
docker-compose logs [service_name] --tail=100
```

### Database Quick Check
```bash
# Test database connection
docker-compose exec postgres pg_isready -U trader

# Check database size and connections
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT 
    pg_size_pretty(pg_database_size('trading_system')) as db_size,
    count(*) as active_connections
  FROM pg_stat_activity
  WHERE datname = 'trading_system';
"
```

### Redis Quick Check
```bash
# Test Redis connection
docker-compose exec redis redis-cli ping

# Check Redis memory usage
docker-compose exec redis redis-cli info memory | grep used_memory_human

# Check Redis key count
docker-compose exec redis redis-cli info keyspace
```

---

## Service-Specific Issues

### Data Collector Service (Port 8001)

#### Symptoms: Service won't start
**Common Causes:**
- API key issues
- Database connection failure
- Redis connection failure
- Port already in use

**Diagnosis:**
```bash
# Check logs
docker-compose logs data_collector --tail=50

# Test API keys
curl -H "Authorization: Bearer $TWELVE_DATA_API_KEY" \
  "https://api.twelvedata.com/time_series?symbol=AAPL&interval=1min&outputsize=5"

# Test database connection from container
docker-compose exec data_collector python -c "
import asyncio
from shared.database import DatabaseManager
from shared.config import Config
async def test():
    config = Config()
    db = DatabaseManager(config)
    await db.initialize()
    print('Database connection successful')
asyncio.run(test())
"
```

**Solutions:**
1. **API Key Issues**: Verify and update API keys in environment file
2. **Database Issues**: Check PostgreSQL logs and restart if needed
3. **Port Conflicts**: Change port in docker-compose.yml
4. **Dependencies**: Restart postgres and redis services first

#### Symptoms: No market data being collected
**Diagnosis:**
```bash
# Check recent data in database
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT symbol, timestamp, close_price 
  FROM market_data 
  ORDER BY timestamp DESC 
  LIMIT 10;
"

# Check API rate limits
curl http://localhost:8001/api/status

# Check scheduler triggers
docker-compose logs scheduler | grep "market_data"
```

**Solutions:**
1. **API Rate Limits**: Check API quota and upgrade if needed
2. **Network Issues**: Verify external connectivity
3. **Database Full**: Check disk space and clean old data
4. **Scheduler Issues**: Restart scheduler service

### Strategy Engine Service (Port 8002)

#### Symptoms: No trading signals generated
**Diagnosis:**
```bash
# Check strategy execution logs
docker-compose logs strategy_engine | grep -E "(signal|strategy)"

# Check available market data
curl http://localhost:8002/strategies/data-status

# Check strategy configurations
curl http://localhost:8002/strategies/list
```

**Solutions:**
1. **Insufficient Data**: Ensure market data is current
2. **Strategy Errors**: Check strategy algorithm logs
3. **Configuration Issues**: Verify strategy parameters
4. **Market Conditions**: Some strategies may not generate signals in certain markets

### Risk Manager Service (Port 8003)

#### Symptoms: All trades being rejected
**Diagnosis:**
```bash
# Check recent risk decisions
curl http://localhost:8003/risk/recent-decisions

# Check current risk metrics
curl http://localhost:8003/risk/current-metrics

# Check risk configuration
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT * FROM risk_parameters ORDER BY updated_at DESC LIMIT 5;
"
```

**Solutions:**
1. **Risk Limits Too Strict**: Review and adjust risk parameters
2. **Portfolio Limits Reached**: Check current position sizes
3. **Market Volatility**: Risk manager may be protective during volatile periods
4. **Configuration Error**: Verify risk management settings

### Trade Executor Service (Port 8004)

#### Symptoms: Orders not executing
**Diagnosis:**
```bash
# Check broker connectivity
curl http://localhost:8004/broker/status

# Check recent orders
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT id, symbol, side, quantity, status, error_message, timestamp 
  FROM trades 
  ORDER BY timestamp DESC 
  LIMIT 10;
"

# Test broker API directly
curl -H "Authorization: Bearer $ALPACA_API_KEY" \
  https://paper-api.alpaca.markets/v2/account
```

**Solutions:**
1. **API Key Issues**: Verify Alpaca API credentials
2. **Account Issues**: Check broker account status and buying power
3. **Market Hours**: Verify trading session times
4. **Order Validation**: Check order parameters (size, price, etc.)

---

## Database Issues

### Connection Problems

#### Symptoms: "Connection refused" or "Database unavailable"
**Diagnosis:**
```bash
# Check PostgreSQL container
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs postgres --tail=50

# Test connection from host
docker-compose exec postgres psql -U trader -d trading_system -c "SELECT NOW();"

# Check database process
docker-compose exec postgres ps aux | grep postgres
```

**Solutions:**
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Check disk space
df -h

# Verify PostgreSQL configuration
docker-compose exec postgres cat /var/lib/postgresql/data/postgresql.conf | grep -E "(max_connections|shared_buffers)"

# Reset PostgreSQL if corrupted
docker-compose down
docker volume rm full-ai-trader_postgres_data
docker-compose up -d postgres
./scripts/deployment/migrate.sh development
```

### Performance Issues

#### Symptoms: Slow database queries
**Diagnosis:**
```bash
# Check slow queries
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT query, calls, total_time, mean_time, rows
  FROM pg_stat_statements
  ORDER BY total_time DESC
  LIMIT 10;
"

# Check database locks
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT pid, usename, application_name, state, query_start, query
  FROM pg_stat_activity
  WHERE state != 'idle';
"

# Check database size
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT 
    schemaname, 
    tablename, 
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
  FROM pg_tables 
  WHERE schemaname = 'public'
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

**Solutions:**
```bash
# Add missing indexes
docker-compose exec postgres psql -U trader -d trading_system -c "
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_symbol ON positions(symbol);
"

# Update table statistics
docker-compose exec postgres psql -U trader -d trading_system -c "ANALYZE;"

# Vacuum tables
docker-compose exec postgres psql -U trader -d trading_system -c "VACUUM ANALYZE;"
```

### Data Corruption

#### Symptoms: Inconsistent data or constraint violations
**Diagnosis:**
```bash
# Check for constraint violations
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT conname, conrelid::regclass, confrelid::regclass
  FROM pg_constraint
  WHERE NOT convalidated;
"

# Check data integrity
python scripts/data/validate_integrity.py --full-check

# Check for duplicate records
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT symbol, timestamp, COUNT(*)
  FROM market_data
  GROUP BY symbol, timestamp
  HAVING COUNT(*) > 1;
"
```

**Solutions:**
```bash
# Restore from backup
./scripts/backup/restore.sh --latest

# Fix constraint violations
docker-compose exec postgres psql -U trader -d trading_system -c "
  -- Remove duplicate records
  DELETE FROM market_data a USING market_data b
  WHERE a.id > b.id AND a.symbol = b.symbol AND a.timestamp = b.timestamp;
"

# Rebuild indexes
docker-compose exec postgres psql -U trader -d trading_system -c "REINDEX DATABASE trading_system;"
```

---

## Redis Issues

### Connection Problems

#### Symptoms: Redis connection timeouts
**Diagnosis:**
```bash
# Check Redis container
docker-compose ps redis

# Check Redis logs
docker-compose logs redis --tail=50

# Test connection
docker-compose exec redis redis-cli ping

# Check Redis configuration
docker-compose exec redis redis-cli config get "*"
```

**Solutions:**
```bash
# Restart Redis
docker-compose restart redis

# Clear Redis data if corrupted
docker-compose exec redis redis-cli FLUSHALL

# Check Redis memory limits
docker-compose exec redis redis-cli info memory
```

### Memory Issues

#### Symptoms: Redis evictions or out of memory
**Diagnosis:**
```bash
# Check memory usage
docker-compose exec redis redis-cli info memory

# Check eviction policy
docker-compose exec redis redis-cli config get maxmemory-policy

# Find large keys
docker-compose exec redis redis-cli --bigkeys
```

**Solutions:**
```bash
# Increase memory limit in docker-compose.yml
# Or clear non-essential data
docker-compose exec redis redis-cli FLUSHDB 1  # Clear cache database

# Optimize cache TTL
# Update environment variables and restart services
```

---

## Network and Connectivity

### External API Issues

#### Symptoms: Market data or broker API failures
**Diagnosis:**
```bash
# Test external connectivity
curl -I https://api.twelvedata.com
curl -I https://paper-api.alpaca.markets

# Check DNS resolution
nslookup api.twelvedata.com
nslookup paper-api.alpaca.markets

# Test from within containers
docker-compose exec data_collector curl -I https://api.twelvedata.com
```

**Solutions:**
1. **DNS Issues**: Configure proper DNS servers
2. **Firewall**: Check firewall rules for outbound connections
3. **API Downtime**: Check provider status pages
4. **Rate Limits**: Implement exponential backoff

### Inter-Service Communication

#### Symptoms: Services can't communicate with each other
**Diagnosis:**
```bash
# Check Docker network
docker network ls
docker network inspect full-ai-trader_trading_network

# Test service-to-service communication
docker-compose exec strategy_engine curl http://data_collector:8001/health
docker-compose exec trade_executor curl http://risk_manager:8003/health

# Check service discovery
docker-compose exec postgres nslookup data_collector
```

**Solutions:**
```bash
# Recreate Docker network
docker-compose down
docker network prune
docker-compose up -d

# Check service names in docker-compose.yml
# Ensure consistent naming across all compose files
```

---

## Performance Issues

### High CPU Usage

#### Symptoms: Container CPU usage > 80%
**Diagnosis:**
```bash
# Check container CPU usage
docker stats --no-stream

# Check process details within container
docker-compose exec [service_name] top

# Check for CPU-intensive queries
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT pid, usename, application_name, state, query_start, query
  FROM pg_stat_activity
  WHERE state = 'active'
  ORDER BY query_start;
"
```

**Solutions:**
1. **Database Optimization**: Add indexes, optimize queries
2. **Strategy Optimization**: Profile and optimize strategy algorithms
3. **Resource Limits**: Increase CPU limits in docker-compose.yml
4. **Load Distribution**: Consider horizontal scaling

### High Memory Usage

#### Symptoms: Container memory usage > 80% or OOMKilled
**Diagnosis:**
```bash
# Check memory usage
docker stats --no-stream

# Check for memory leaks
docker-compose exec [service_name] cat /proc/meminfo

# Check large objects in Redis
docker-compose exec redis redis-cli --bigkeys

# Check database memory usage
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT 
    schemaname, 
    tablename, 
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
  FROM pg_tables 
  WHERE schemaname = 'public'
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
  LIMIT 10;
"
```

**Solutions:**
```bash
# Increase memory limits
# Edit docker-compose.yml and add:
# deploy:
#   resources:
#     limits:
#       memory: 2G

# Clear Redis cache
docker-compose exec redis redis-cli FLUSHDB

# Restart memory-intensive services
docker-compose restart strategy_engine risk_manager

# Optimize data retention
python scripts/data/cleanup_old_data.py --days 30
```

### Slow Response Times

#### Symptoms: API responses > 2 seconds
**Diagnosis:**
```bash
# Test response times
curl -w "@curl-format.txt" -o /dev/null http://localhost:8001/health

# Check database performance
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT 
    schemaname, 
    tablename, 
    seq_scan, 
    seq_tup_read, 
    idx_scan, 
    idx_tup_fetch
  FROM pg_stat_user_tables
  ORDER BY seq_tup_read DESC;
"

# Check Redis latency
docker-compose exec redis redis-cli --latency -h redis
```

**Solutions:**
1. **Database**: Add indexes, optimize queries, increase connection pool
2. **Redis**: Optimize data structures, increase memory
3. **Application**: Profile code, optimize algorithms
4. **Network**: Check for network latency issues

---

## Trading Execution Problems

### Orders Not Executing

#### Symptoms: Orders stuck in pending state
**Diagnosis:**
```bash
# Check pending orders
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT id, symbol, side, quantity, status, error_message, timestamp
  FROM trades
  WHERE status IN ('pending', 'submitted')
  ORDER BY timestamp DESC;
"

# Check broker API status
curl -H "Authorization: Bearer $ALPACA_API_KEY" \
  https://paper-api.alpaca.markets/v2/account

# Check market hours
python -c "
from datetime import datetime
import pytz
now = datetime.now(pytz.timezone('America/New_York'))
print(f'Current market time: {now}')
print(f'Market open: {9 <= now.hour < 16 and now.weekday() < 5}')
"

# Check risk manager decisions
curl http://localhost:8003/risk/recent-decisions
```

**Solutions:**
1. **Market Closed**: Wait for market hours or use extended hours
2. **Insufficient Funds**: Check account buying power
3. **Risk Limits**: Review risk management settings
4. **API Issues**: Check broker API status and credentials

### Order Rejections

#### Symptoms: High rate of order rejections
**Diagnosis:**
```bash
# Check rejection reasons
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT error_message, COUNT(*) as count
  FROM trades
  WHERE status = 'rejected'
  AND timestamp > NOW() - INTERVAL '24 hours'
  GROUP BY error_message
  ORDER BY count DESC;
"

# Check account status
curl -H "Authorization: Bearer $ALPACA_API_KEY" \
  https://paper-api.alpaca.markets/v2/account

# Check position limits
curl http://localhost:8003/risk/position-limits
```

**Solutions:**
1. **Account Restrictions**: Contact broker support
2. **Position Limits**: Adjust risk management parameters
3. **Order Size**: Check minimum/maximum order sizes
4. **Symbol Restrictions**: Verify symbol is tradeable

---

## Data Quality Issues

### Missing Market Data

#### Symptoms: Gaps in market data or stale prices
**Diagnosis:**
```bash
# Check data freshness
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT 
    symbol, 
    MAX(timestamp) as latest_data,
    NOW() - MAX(timestamp) as data_age
  FROM market_data
  GROUP BY symbol
  ORDER BY data_age DESC;
"

# Check data collector status
curl http://localhost:8001/data/status

# Check API response times
curl -w "%{time_total}" http://localhost:8001/data/AAPL
```

**Solutions:**
```bash
# Trigger manual data collection
curl -X POST http://localhost:8001/data/collect \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "SPY"], "force": true}'

# Restart data collector
docker-compose restart data_collector

# Check API quotas and upgrade if needed
```

### Inconsistent Data

#### Symptoms: Data validation errors or impossible values
**Diagnosis:**
```bash
# Run data validation
python scripts/data/validate_integrity.py --detailed

# Check for outliers
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT symbol, timestamp, open_price, high_price, low_price, close_price
  FROM market_data
  WHERE high_price < low_price OR close_price <= 0
  ORDER BY timestamp DESC;
"

# Check data sources
curl http://localhost:8001/data/sources/status
```

**Solutions:**
1. **Data Source Issues**: Switch to backup data provider
2. **Processing Errors**: Review data transformation logic
3. **Corruption**: Restore from backup
4. **Validation Rules**: Update data validation parameters

---

## Security and Access Issues

### Authentication Failures

#### Symptoms: 401 Unauthorized errors
**Diagnosis:**
```bash
# Check API tokens
echo $API_SECRET_KEY | wc -c  # Should be at least 32 characters

# Check recent authentication attempts
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT timestamp, user_id, ip_address, success, error_message
  FROM audit_logs
  WHERE action = 'authenticate'
  ORDER BY timestamp DESC
  LIMIT 10;
"

# Test token validation
curl -H "Authorization: Bearer $API_TOKEN" http://localhost:8001/health
```

**Solutions:**
1. **Expired Tokens**: Generate new API tokens
2. **Invalid Keys**: Update API keys in environment
3. **Clock Skew**: Synchronize system clocks
4. **Service Restart**: Restart authentication-related services

### Rate Limiting Issues

#### Symptoms: 429 Too Many Requests errors
**Diagnosis:**
```bash
# Check rate limiting status
curl http://localhost:8007/rate-limits/status

# Check Redis rate limit keys
docker-compose exec redis redis-cli keys "rate_limit:*" | head -10

# Check current usage
docker-compose exec redis redis-cli zcard "rate_limit:api_general:192.168.1.100"
```

**Solutions:**
```bash
# Temporarily increase limits
# Edit environment file and restart services

# Clear rate limit data
docker-compose exec redis redis-cli DEL "rate_limit:*"

# Whitelist specific IPs
# Add to bypass_ips in rate limiting configuration
```

---

## Emergency Scenarios

### System Unresponsive

#### Immediate Actions
```bash
# Check system resources
top
df -h
free -h

# Check Docker daemon
systemctl status docker

# Emergency restart
docker-compose down --timeout 30
docker-compose up -d
```

### Data Loss Incident

#### Immediate Actions
```bash
# Stop all services immediately
docker-compose down

# Assess damage
ls -la data/
docker volume ls

# Restore from latest backup
./scripts/backup/restore.sh --emergency --latest

# Verify data integrity
python scripts/data/validate_integrity.py --full
```

### Security Incident

#### Immediate Actions
```bash
# Enter emergency shutdown
curl -X POST http://localhost:8007/maintenance/emergency-shutdown \
  -H "Authorization: Bearer $API_TOKEN" \
  -d "reason=Security incident detected"

# Preserve logs
cp -r logs/ incident_logs_$(date +%Y%m%d_%H%M%S)/

# Change all API keys
./scripts/security/rotate_all_keys.sh

# Review access logs
grep "401\|403\|429" logs/*.log | tail -100
```

---

## Preventive Maintenance

### Daily Checks
```bash
# System health
./scripts/monitoring/daily_health_check.sh

# Backup verification
./scripts/backup/verify_backups.sh

# Log analysis
grep -c "ERROR" logs/$(date +%Y-%m-%d)*.log

# Disk space
df -h | grep -E "(9[0-9]%|100%)"
```

### Weekly Maintenance
```bash
# Database maintenance
docker-compose exec postgres psql -U trader -d trading_system -c "
  VACUUM ANALYZE;
  REINDEX DATABASE trading_system;
"

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete

# Update system packages
apt update && apt upgrade -y

# Security audit
python scripts/security/security_audit.py
```

### Monthly Tasks
```bash
# Full backup verification
./scripts/backup/test_restore.sh --full-test

# Performance analysis
python scripts/monitoring/performance_report.py --period 30

# Security review
python scripts/security/access_review.py --days 30

# Capacity planning
python scripts/monitoring/capacity_analysis.py
```

---

## Diagnostic Scripts

### System Information
```bash
#!/bin/bash
# Save as scripts/diagnostics/system_info.sh

echo "=== System Information ==="
date
uname -a
df -h
free -h
docker --version
docker-compose --version

echo "=== Container Status ==="
docker-compose ps

echo "=== Container Resources ==="
docker stats --no-stream

echo "=== Service Health ==="
for port in 8001 8002 8003 8004 8005 8006 8007; do
  echo -n "Port $port: "
  curl -s -w "%{http_code}" http://localhost:$port/health -o /dev/null || echo "FAILED"
done

echo "=== Database Status ==="
docker-compose exec postgres pg_isready -U trader

echo "=== Redis Status ==="
docker-compose exec redis redis-cli ping
```

### Log Analysis
```bash
#!/bin/bash
# Save as scripts/diagnostics/analyze_logs.sh

echo "=== Error Summary (Last 24 Hours) ==="
find logs/ -name "*.log" -mtime -1 -exec grep -h "ERROR" {} \; | \
  cut -d' ' -f4- | sort | uniq -c | sort -nr | head -20

echo "=== Warning Summary (Last 24 Hours) ==="
find logs/ -name "*.log" -mtime -1 -exec grep -h "WARNING" {} \; | \
  cut -d' ' -f4- | sort | uniq -c | sort -nr | head -10

echo "=== Recent Critical Events ==="
docker-compose exec postgres psql -U trader -d trading_system -c "
  SELECT timestamp, service_name, action, error_message
  FROM audit_logs
  WHERE severity = 'critical'
  AND timestamp > NOW() - INTERVAL '24 hours'
  ORDER BY timestamp DESC
  LIMIT 10;
"
```

---

## Recovery Procedures

### Service Recovery Priority
1. **PostgreSQL** (Foundation)
2. **Redis** (Caching layer)
3. **Data Collector** (Market data)
4. **Risk Manager** (Safety)
5. **Strategy Engine** (Decision making)
6. **Trade Executor** (Execution)
7. **Scheduler** (Orchestration)
8. **Export/Maintenance** (Support services)

### Minimal Recovery Mode
```bash
# Start only essential services for emergency trading
docker-compose up -d postgres redis
sleep 30
docker-compose up -d data_collector risk_manager trade_executor

# Verify minimal functionality
curl http://localhost:8001/health
curl http://localhost:8003/health
curl http://localhost:8004/health

# Test emergency trade execution
curl -X POST http://localhost:8004/emergency/close-all-positions
```

---

## Contact Information

### Internal Escalation
- **Level 1**: Operations team
- **Level 2**: Development team
- **Level 3**: System architects
- **Level 4**: CTO/VP Engineering

### External Support
- **Alpaca Markets**: support@alpaca.markets
- **Twelve Data**: support@twelvedata.com
- **AWS Support**: [Your support plan]
- **Database Vendor**: [Support contact]

---

## Documentation and Resources

### Log Locations
- **Application Logs**: `logs/`
- **Container Logs**: `docker-compose logs [service]`
- **System Logs**: `/var/log/`
- **Audit Logs**: Database `audit_logs` table

### Configuration Files
- **Environment**: `config/environments/`
- **Docker**: `docker-compose*.yml`
- **Database**: `scripts/*.sql`
- **Monitoring**: `monitoring/`

### Useful Queries
```sql
-- Check recent trade activity
SELECT COUNT(*), status
FROM trades
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY status;

-- Check system performance
SELECT 
  service_name,
  COUNT(*) as requests,
  AVG(execution_time_ms) as avg_response_time
FROM audit_logs
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY service_name;

-- Check error rates
SELECT 
  service_name,
  COUNT(CASE WHEN success THEN 1 END) as successful,
  COUNT(CASE WHEN NOT success THEN 1 END) as failed
FROM audit_logs
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY service_name;
```

---

## Update History

**Last Updated**: 2024-01-15
**Version**: 1.0.0
**Next Review**: 2024-04-15

This document should be updated after:
- Major system changes
- New issue discoveries
- Procedure improvements
- Infrastructure changes