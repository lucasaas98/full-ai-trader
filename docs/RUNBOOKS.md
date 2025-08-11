# AI Trading System - Operational Runbooks

## Overview

This document provides step-by-step procedures for handling common operational issues in the AI Trading System. These runbooks are designed for on-call engineers and system administrators.

## Emergency Contacts

- **Trading Team Lead**: +1-555-0101
- **DevOps Lead**: +1-555-0102
- **Risk Manager**: +1-555-0103
- **Security Team**: +1-555-0104
- **Compliance Officer**: +1-555-0105

## Severity Levels

- **P0 (Critical)**: Trading halted, potential data loss, revenue impact
- **P1 (High)**: Degraded performance, partial service unavailability
- **P2 (Medium)**: Non-critical issues, monitoring alerts
- **P3 (Low)**: Minor issues, maintenance items

---

## P0 - Critical Issues

### 1. Complete System Outage

**Symptoms:**
- All services returning 500/503 errors
- No market data updates
- Trading completely halted
- Dashboard shows all services down

**Immediate Actions (5 minutes):**

1. **Verify the outage:**
   ```bash
   # Check all service health
   curl -f http://localhost:8001/health || echo "Data Collector DOWN"
   curl -f http://localhost:8002/health || echo "Strategy Engine DOWN"
   curl -f http://localhost:8003/health || echo "Risk Manager DOWN"
   curl -f http://localhost:8004/health || echo "Trade Executor DOWN"
   curl -f http://localhost:8006/health || echo "Scheduler DOWN"
   ```

2. **Check infrastructure:**
   ```bash
   # Check Docker containers
   docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
   
   # Check system resources
   df -h
   free -h
   top -bn1 | head -20
   ```

3. **Emergency notification:**
   ```bash
   # Send critical alert
   curl -X POST http://gotify:8080/message \
     -H "X-Gotify-Key: $GOTIFY_TOKEN" \
     -d '{
       "title": "CRITICAL: Trading System Outage",
       "message": "Complete system outage detected. All trading halted.",
       "priority": 10
     }'
   ```

**Recovery Steps:**

1. **Restart infrastructure services:**
   ```bash
   # Restart database if needed
   docker-compose restart postgres
   docker-compose restart redis
   
   # Wait for database to be ready
   until docker-compose exec postgres pg_isready -U trader; do
     echo "Waiting for database..."
     sleep 2
   done
   ```

2. **Restart core services in order:**
   ```bash
   # Start in dependency order
   docker-compose up -d data_collector
   sleep 10
   docker-compose up -d strategy_engine
   sleep 10
   docker-compose up -d risk_manager
   sleep 10
   docker-compose up -d trade_executor
   sleep 10
   docker-compose up -d scheduler
   ```

3. **Verify recovery:**
   ```bash
   # Check all services
   make health-check
   
   # Verify trading functionality
   python scripts/smoke_test.py
   ```

4. **Resume trading:**
   ```bash
   # Re-enable automatic trading
   curl -X POST http://localhost:8006/api/v1/scheduler/resume-trading
   ```

**Escalation:** If not resolved in 15 minutes, escalate to Trading Team Lead.

### 2. Database Connection Failure

**Symptoms:**
- Services reporting database connection errors
- 500 errors on portfolio/trade endpoints
- "Connection refused" or "Connection timeout" in logs

**Immediate Actions (3 minutes):**

1. **Check database status:**
   ```bash
   docker-compose ps postgres
   docker-compose logs postgres | tail -20
   ```

2. **Test connectivity:**
   ```bash
   # Test from host
   pg_isready -h localhost -p 5432 -U trader
   
   # Test from container
   docker-compose exec postgres psql -U trader -d trading_system -c "SELECT 1;"
   ```

3. **Check connection limits:**
   ```sql
   -- Check current connections
   SELECT count(*) as connections, state 
   FROM pg_stat_activity 
   WHERE datname = 'trading_system' 
   GROUP BY state;
   
   -- Check max connections setting
   SHOW max_connections;
   ```

**Recovery Steps:**

1. **If database is down:**
   ```bash
   # Restart database container
   docker-compose restart postgres
   
   # Monitor startup logs
   docker-compose logs -f postgres
   ```

2. **If connection limit exceeded:**
   ```sql
   -- Kill idle connections
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE datname = 'trading_system' 
   AND state = 'idle' 
   AND state_change < now() - interval '5 minutes';
   ```

3. **Restart affected services:**
   ```bash
   # Restart services that lost DB connection
   docker-compose restart trade_executor
   docker-compose restart risk_manager
   ```

### 3. Market Data Feed Failure

**Symptoms:**
- No new market data updates
- Stale prices in portfolio
- Strategy engine not generating signals
- "Market data provider unreachable" alerts

**Immediate Actions (2 minutes):**

1. **Check data collector status:**
   ```bash
   docker-compose logs data_collector | tail -10
   curl http://localhost:8001/api/v1/status
   ```

2. **Test external APIs:**
   ```bash
   # Test Alpha Vantage
   curl "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=$ALPHA_VANTAGE_KEY"
   
   # Test Yahoo Finance backup
   curl "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
   ```

3. **Check Redis pub/sub:**
   ```bash
   # Monitor market data channel
   docker-compose exec redis redis-cli MONITOR | grep "market_data"
   ```

**Recovery Steps:**

1. **Switch to backup data provider:**
   ```bash
   # Update data collector config
   curl -X PUT http://localhost:8001/api/v1/config \
     -H "Content-Type: application/json" \
     -d '{"primary_provider": "yahoo_finance", "fallback_enabled": true}'
   ```

2. **Restart data collection:**
   ```bash
   docker-compose restart data_collector
   
   # Force immediate collection
   curl -X POST http://localhost:8001/api/v1/collect-now
   ```

3. **Verify data flow:**
   ```bash
   # Check recent data
   curl http://localhost:8001/api/v1/market-data/AAPL?limit=5
   ```

---

## P1 - High Priority Issues

### 4. High Error Rate in Trading Operations

**Symptoms:**
- Error rate > 5% in trade execution
- Multiple failed orders
- Risk manager rejecting many signals

**Investigation Steps:**

1. **Check error patterns:**
   ```bash
   # Recent errors in trade executor
   docker-compose logs trade_executor | grep -i error | tail -20
   
   # Error rate by type
   curl http://localhost:9090/api/v1/query?query='rate(trading_errors_total[5m])'
   ```

2. **Analyze rejection reasons:**
   ```bash
   # Risk manager rejection logs
   docker-compose logs risk_manager | grep -i "rejected\|denied" | tail -10
   ```

3. **Check market conditions:**
   ```bash
   # Recent market volatility
   python scripts/check_market_volatility.py --symbols AAPL,GOOGL,SPY
   ```

**Resolution Steps:**

1. **Adjust risk parameters temporarily:**
   ```bash
   curl -X PUT http://localhost:8003/api/v1/risk/limits \
     -H "Content-Type: application/json" \
     -d '{
       "max_position_concentration": 0.12,
       "volatility_multiplier": 1.2,
       "temporary_adjustment": true
     }'
   ```

2. **Review and pause problematic strategies:**
   ```bash
   # Get strategy performance
   curl http://localhost:8002/api/v1/strategies/momentum_strategy/performance
   
   # Pause if underperforming
   curl -X PUT http://localhost:8002/api/v1/strategies/momentum_strategy \
     -d '{"enabled": false}'
   ```

### 5. Database Performance Degradation

**Symptoms:**
- Slow API responses
- Database query timeouts
- High CPU usage on database server

**Investigation Steps:**

1. **Check active queries:**
   ```sql
   -- Long-running queries
   SELECT pid, now() - pg_stat_activity.query_start AS duration, query
   FROM pg_stat_activity
   WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
   AND state = 'active';
   ```

2. **Check database locks:**
   ```sql
   -- Blocking queries
   SELECT blocked_locks.pid AS blocked_pid,
          blocked_activity.usename AS blocked_user,
          blocking_locks.pid AS blocking_pid,
          blocking_activity.usename AS blocking_user,
          blocked_activity.query AS blocked_statement,
          blocking_activity.query AS current_statement_in_blocking_process
   FROM pg_catalog.pg_locks blocked_locks
   JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
   JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
   JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
   WHERE NOT blocked_locks.granted;
   ```

3. **Check system resources:**
   ```bash
   # Database container resources
   docker stats postgres --no-stream
   
   # Disk I/O
   iostat -x 1 3
   ```

**Resolution Steps:**

1. **Kill problematic queries:**
   ```sql
   -- Kill long-running queries
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE (now() - query_start) > interval '10 minutes'
   AND state = 'active'
   AND query NOT LIKE '%pg_stat_activity%';
   ```

2. **Optimize queries:**
   ```sql
   -- Update table statistics
   ANALYZE;
   
   -- Rebuild indexes if needed
   REINDEX INDEX CONCURRENTLY idx_trades_symbol_executed;
   ```

3. **Scale resources if needed:**
   ```bash
   # Increase database memory (requires restart)
   # Edit docker-compose.yml and restart
   docker-compose up -d postgres
   ```

### 6. Redis Memory Issues

**Symptoms:**
- Redis running out of memory
- Pub/sub message delays
- Cache miss rate increasing

**Investigation Steps:**

1. **Check Redis memory usage:**
   ```bash
   docker-compose exec redis redis-cli INFO memory
   docker-compose exec redis redis-cli --bigkeys
   ```

2. **Check key expiration:**
   ```bash
   # Keys without expiration
   docker-compose exec redis redis-cli EVAL "
   local keys = redis.call('keys', '*')
   local no_expiry = {}
   for i=1,#keys do
     if redis.call('ttl', keys[i]) == -1 then
       table.insert(no_expiry, keys[i])
     end
   end
   return no_expiry
   " 0
   ```

**Resolution Steps:**

1. **Clean up old keys:**
   ```bash
   # Remove old market data (older than 1 day)
   docker-compose exec redis redis-cli EVAL "
   local keys = redis.call('keys', 'market_data:*')
   local deleted = 0
   for i=1,#keys do
     local age = redis.call('time')[1] - redis.call('hget', keys[i], 'timestamp')
     if age > 86400 then
       redis.call('del', keys[i])
       deleted = deleted + 1
     end
   end
   return deleted
   " 0
   ```

2. **Adjust memory policy:**
   ```bash
   # Set eviction policy
   docker-compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
   ```

---

## P2 - Medium Priority Issues

### 7. High API Response Times

**Symptoms:**
- API endpoints responding slowly (> 1 second)
- Timeout errors from clients
- Poor user experience

**Investigation Steps:**

1. **Check service performance:**
   ```bash
   # Recent response times
   curl http://localhost:9090/api/v1/query?query='histogram_quantile(0.95, rate(trading_api_request_duration_seconds_bucket[5m]))'
   ```

2. **Identify slow endpoints:**
   ```bash
   # Slow endpoint analysis
   docker-compose logs nginx | grep "upstream_response_time" | sort -k12 -nr | head -10
   ```

3. **Check resource usage:**
   ```bash
   # Service resource usage
   docker stats --no-stream | grep trading
   ```

**Resolution Steps:**

1. **Scale problematic services:**
   ```bash
   # Scale strategy engine if needed
   docker-compose up -d --scale strategy_engine=3
   ```

2. **Optimize database queries:**
   ```sql
   -- Find slow queries
   SELECT query, mean_time, calls
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

3. **Clear cache if needed:**
   ```bash
   # Clear Redis cache
   docker-compose exec redis redis-cli FLUSHDB
   ```

### 8. Strategy Underperformance

**Symptoms:**
- Strategy returns below benchmark
- High drawdown periods
- Low signal accuracy

**Investigation Steps:**

1. **Check strategy metrics:**
   ```bash
   curl http://localhost:8002/api/v1/strategies/momentum_strategy/performance?days=30
   ```

2. **Analyze recent signals:**
   ```bash
   # Recent signal accuracy
   python scripts/analyze_signals.py --strategy momentum_strategy --days 7
   ```

3. **Check market conditions:**
   ```bash
   # Market regime analysis
   python scripts/market_regime_detector.py --symbols SPY,VIX
   ```

**Resolution Steps:**

1. **Adjust strategy parameters:**
   ```bash
   curl -X PUT http://localhost:8002/api/v1/strategies/momentum_strategy/parameters \
     -H "Content-Type: application/json" \
     -d '{
       "rsi_period": 21,
       "rsi_oversold": 25,
       "position_size_multiplier": 0.8
     }'
   ```

2. **Reduce position sizes:**
   ```bash
   curl -X PUT http://localhost:8003/api/v1/risk/limits \
     -d '{"max_position_size": 0.08, "reason": "strategy_underperformance"}'
   ```

3. **Consider strategy pause:**
   ```bash
   # Pause strategy if severe underperformance
   curl -X PUT http://localhost:8002/api/v1/strategies/momentum_strategy \
     -d '{"enabled": false, "reason": "performance_review"}'
   ```

---

## P3 - Routine Maintenance

### 9. Log Rotation and Cleanup

**Schedule:** Daily at 2 AM

**Steps:**

1. **Rotate application logs:**
   ```bash
   # Compress old logs
   find /app/logs -name "*.log" -mtime +1 -exec gzip {} \;
   
   # Remove very old compressed logs
   find /app/logs -name "*.log.gz" -mtime +30 -delete
   ```

2. **Clean up database logs:**
   ```sql
   -- Remove old audit logs
   DELETE FROM audit_logs WHERE created_at < NOW() - INTERVAL '90 days';
   
   -- Remove old performance metrics
   DELETE FROM performance_metrics WHERE timestamp < NOW() - INTERVAL '30 days';
   ```

3. **Clean up Redis:**
   ```bash
   # Remove expired keys
   docker-compose exec redis redis-cli --eval scripts/cleanup_redis.lua
   ```

### 10. Performance Optimization

**Schedule:** Weekly

**Steps:**

1. **Database maintenance:**
   ```sql
   -- Update table statistics
   ANALYZE;
   
   -- Vacuum tables
   VACUUM ANALYZE trades;
   VACUUM ANALYZE market_data;
   VACUUM ANALYZE portfolio_positions;
   ```

2. **Check and rebuild indexes:**
   ```sql
   -- Check index usage
   SELECT schemaname, tablename, attname, n_distinct, correlation
   FROM pg_stats
   WHERE tablename IN ('trades', 'market_data')
   ORDER BY n_distinct DESC;
   
   -- Rebuild unused indexes
   REINDEX INDEX CONCURRENTLY idx_trades_executed_at;
   ```

3. **Optimize Redis:**
   ```bash
   # Defragment Redis memory
   docker-compose exec redis redis-cli MEMORY PURGE
   
   # Check fragmentation
   docker-compose exec redis redis-cli INFO memory | grep fragmentation
   ```

---

## Monitoring and Alerting Procedures

### Alert Response Matrix

| Alert | Severity | Response Time | Actions |
|-------|----------|---------------|---------|
| Service Down | P0 | 2 minutes | Restart service, check dependencies |
| High Error Rate | P1 | 5 minutes | Investigate logs, check resources |
| Memory Usage High | P2 | 15 minutes | Scale resources, optimize queries |
| Disk Space Low | P2 | 30 minutes | Clean up logs, expand storage |
| Strategy Underperform | P3 | 1 hour | Review parameters, market analysis |

### Common Alert Queries

```bash
# Check for active alerts
curl http://localhost:9093/api/v1/alerts

# Service health summary
curl http://localhost:9090/api/v1/query?query='up{job="trading-system"}'

# Error rate by service
curl http://localhost:9090/api/v1/query?query='rate(trading_errors_total[5m])'

# Memory usage
curl http://localhost:9090/api/v1/query?query='trading_system_memory_usage_percent'
```

---

## Security Incident Response

### 1. Suspected Breach

**Immediate Actions (1 minute):**

1. **Isolate the system:**
   ```bash
   # Block external access
   docker-compose exec nginx nginx -s reload -c /etc/nginx/nginx-emergency.conf
   
   # Revoke all API keys
   curl -X POST http://localhost:8009/api/v1/security/revoke-all-keys
   ```

2. **Preserve evidence:**
   ```bash
   # Capture current logs
   docker-compose logs > /tmp/security-incident-$(date +%Y%m%d-%H%M%S).log
   
   # Snapshot system state
   docker-compose exec postgres pg_dump trading_system > /tmp/db-snapshot-$(date +%Y%m%d-%H%M%S).sql
   ```

3. **Notify security team:**
   ```bash
   curl -X POST http://gotify:8080/message \
     -H "X-Gotify-Key: $GOTIFY_TOKEN" \
     -d '{
       "title": "SECURITY INCIDENT",
       "message": "Suspected security breach. System isolated.",
       "priority": 10
     }'
   ```

### 2. API Key Compromise

**Steps:**

1. **Revoke compromised key:**
   ```bash
   curl -X DELETE http://localhost:8009/api/v1/security/api-keys/{key_id}
   ```

2. **Check for unauthorized access:**
   ```bash
   # Search for suspicious activity
   grep -i "unauthorized\|forbidden\|401" /app/logs/security.log
   ```

3. **Generate new keys:**
   ```bash
   curl -X POST http://localhost:8009/api/v1/security/api-keys \
     -d '{"name": "replacement_key", "scopes": ["trading", "read"]}'
   ```

---

## Data Recovery Procedures

### 1. Market Data Recovery

**When:** After extended data feed outage

**Steps:**

1. **Identify data gaps:**
   ```sql
   -- Find missing data periods
   SELECT symbol, 
          MIN(timestamp) as start_gap,
          MAX(timestamp) as end_gap,
          COUNT(*) as missing_points
   FROM generate_series(
     '2024-01-15 09:30:00'::timestamp,
     '2024-01-15 16:00:00'::timestamp,
     '1 minute'::interval
   ) AS expected_time
   LEFT JOIN market_quotes ON market_quotes.timestamp = expected_time
   WHERE market_quotes.timestamp IS NULL
   GROUP BY symbol;
   ```

2. **Backfill missing data:**
   ```bash
   # Run backfill script
   python scripts/backfill_market_data.py \
     --start-date "2024-01-15 09:30:00" \
     --end-date "2024-01-15 16:00:00" \
     --symbols AAPL,GOOGL,MSFT
   ```

3. **Verify data integrity:**
   ```bash
   python scripts/validate_market_data.py --date 2024-01-15
   ```

### 2. Trade Data Recovery

**When:** After trade executor failure

**Steps:**

1. **Reconcile with broker:**
   ```bash
   # Get broker trade confirmations
   python scripts/broker_reconciliation.py --date $(date +%Y-%m-%d)
   ```

2. **Update missing trades:**
   ```bash
   # Import confirmed trades
   python scripts/import_broker_trades.py --file broker_trades.csv
   ```

3. **Recalculate portfolio:**
   ```bash
   # Rebuild portfolio from trades
   python scripts/rebuild_portfolio.py --account test_account
   ```

---

## Configuration Management

### 1. Strategy Parameter Updates

**Process:**

1. **Backup current configuration:**
   ```bash
   curl http://localhost:8002/api/v1/strategies/momentum_strategy > /tmp/strategy-backup-$(date +%Y%m%d).json
   ```

2. **Update parameters:**
   ```bash
   curl -X PUT http://localhost:8002/api/v1/strategies/momentum_strategy/parameters \
     -H "Content-Type: application/json" \
     -d '{
       "rsi_period": 21,
       "sma_short": 15,
       "sma_long": 45
     }'
   ```

3. **Monitor impact:**
   ```bash
   # Watch for 30 minutes
   watch -n 30 'curl -s http://localhost:8002/api/v1/strategies/momentum_strategy/performance | jq .performance.total_return'
   ```

### 2. Risk Limit Adjustments

**Process:**

1. **Document current limits:**
   ```bash
   curl http://localhost:8003/api/v1/risk/limits > /tmp/risk-limits-backup-$(date +%Y%m%d).json
   ```

2. **Calculate new limits:**
   ```bash
   python scripts/calculate_risk_limits.py --volatility-adjustment 1.2
   ```

3. **Apply new limits:**
   ```bash
   curl -X PUT http://localhost:8003/api/v1/risk/limits \
     -H "Content-Type: application/json" \
     -d @new_risk_limits.json
   ```

---

## Backup and Recovery

### 1. Daily Backup Procedure

**Schedule:** Daily at 1 AM

```bash
#!/bin/bash
# Daily backup script

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/$BACKUP_DATE"

mkdir -p $BACKUP_DIR

# Database backup
docker-compose exec postgres pg_dump -U trader trading_system | gzip > $BACKUP_DIR/database.sql.gz

# Redis backup
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb $BACKUP_DIR/redis.rdb

# Configuration backup
cp -r config/ $BACKUP_DIR/config/

# Log backup
tar czf $BACKUP_DIR/logs.tar.gz logs/

# Upload to cloud storage (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    aws s3 sync $BACKUP_DIR s3://$AWS_S3_BUCKET/backups/$BACKUP_DATE/
fi

echo "Backup completed: $BACKUP_DIR"
```

### 2. Point-in-Time Recovery

**When:** Data corruption or accidental deletion

**Steps:**

1. **Stop trading operations:**
   ```bash
   curl -X POST http://localhost:8006/api/v1/scheduler/emergency-stop
   ```

2. **Restore database:**
   ```bash
   # Stop services
   docker-compose stop
   
   # Restore database
   docker-compose up -d postgres
   zcat /backups/20240115/database.sql.gz | docker-compose exec -T postgres psql -U trader trading_system
   ```

3. **Verify data integrity:**
   ```bash
   python scripts/verify_data_integrity.py --date 2024-01-15
   ```

4. **Resume operations:**
   ```bash
   docker-compose up -d
   curl -X POST http://localhost:8006/api/v1/scheduler/resume-trading
   ```

---

## Performance Tuning

### 1. Database Optimization

**Monthly Tasks:**

1. **Analyze query performance:**
   ```sql
   -- Slowest queries
   SELECT query, mean_time, calls, total_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 20;
   ```

2. **Index optimization:**
   ```sql
   -- Unused indexes
   SELECT s.schemaname, s.tablename, s.indexname, s.idx_scan
   FROM pg_stat_user_indexes s
   JOIN pg_index i ON s.indexrelid = i.indexrelid
   WHERE s.idx_scan = 0 AND NOT i.indisunique;
   ```

3. **Partition management:**
   ```sql
   -- Create monthly partitions for trades
   CREATE TABLE trades_2024_02 PARTITION OF trades
   FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
   ```

### 2. Application Optimization

**Weekly Tasks:**

1. **Memory profiling:**
   ```bash
   # Python memory profiling
   python -m memory_profiler scripts/profile_strategy_engine.py
   ```

2. **Code profiling:**
   ```bash
   # Performance profiling
   python -m cProfile -o strategy_profile.prof services/strategy_engine/src/main.py
   ```

3. **Dependency updates:**
   ```bash
   # Check for updates
   pip list --outdated
   
   # Update non-breaking changes
   pip install -U --upgrade-strategy only-if-needed -r requirements.txt
   ```

---

## Troubleshooting Common Issues

### Issue: "Out of Memory" Errors

**Symptoms:**
- Container restarts frequently
- OOMKilled in docker logs
- Slow response times

**Solution:**
1. **Increase container memory:**
   ```yaml
   # In docker-compose.yml
   strategy_engine:
     deploy:
       resources:
         limits:
           memory: 2G
         reservations:
           memory: 1G
   ```

2. **Optimize memory usage:**
   ```python
   # In Python services, use memory-efficient data structures
   import gc
   gc.collect()  # Force garbage collection
   ```

### Issue: "Connection Pool Exhausted"

**Symptoms:**
- Database connection errors
- "Too many clients" errors

**Solution:**
1. **Increase connection pool:**
   ```python
   # In database config
   DATABASE_CONFIG = {
       'max_connections': 50,
       'min_connections': 5,
       'connection_timeout': 30
   }
   ```

2. **Fix connection leaks:**
   ```python
   # Always use context managers
   async with db_pool.acquire() as connection:
       # Use connection
       pass  # Connection automatically released
   ```

### Issue: "Redis Connection Timeout"

**Symptoms:**
- Pub/sub message delays
- Cache operation failures

**Solution:**
1. **Check Redis configuration:**
   ```bash
   docker-compose exec redis redis-cli CONFIG GET timeout
   docker-compose exec redis redis-cli CONFIG GET tcp-keepalive
   ```

2. **Optimize Redis settings:**
   ```bash
   # Increase timeout
   docker-compose exec redis redis-cli CONFIG SET timeout 300
   
   # Enable keepalive
   docker-compose exec redis redis-cli CONFIG SET tcp-keepalive 60
   ```

---

## Emergency Procedures

### 1. Emergency Trading Halt

**When to use:**
- Major system issues affecting trading accuracy
- Suspected security breach
- Regulatory requirements

**Steps:**

1. **Immediate halt:**
   ```bash
   # Stop all trading activities
   curl -X POST http://localhost:8006/api/v1/emergency/halt-trading
   ```

2. **Cancel pending orders:**
   ```bash
   # Cancel all open orders
   curl -X POST http://localhost:8004/api/v1/orders/cancel-all
   ```

3. **Notify stakeholders:**
   ```bash
   python scripts/emergency_notification.py --type trading_halt --reason "system_issue"
   ```

### 2. Position Liquidation

**When to use:**
- Risk limits severely breached
- Market crash scenario
- System failure requiring position protection

**Steps:**

1. **Calculate liquidation orders:**
   ```bash
   python scripts/calculate_liquidation_orders.py --max-risk 0.05
   ```

2. **Execute liquidation:**
   ```bash
   # Submit market orders to close positions
   curl -X POST http://localhost:8004/api/v1/emergency/liquidate-positions \
     -d '{"reason": "risk_management", "max_loss_percent": 0.10}'
   ```

3. **Monitor execution:**
   ```bash
   watch -n 5 'curl -s http://localhost:8005/api/v1/portfolio | jq .total_value'
   ```

---

## Health Check Scripts

### 1. System Health Check

```bash
#!/bin/bash
# system_health_check.sh

echo "=== Trading System Health Check ==="
echo "Timestamp: $(date)"
echo

# Service health
echo "Service Health:"
for service in data_collector strategy_engine risk_manager trade_executor scheduler; do
    status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:800${service:0:1}/health)
    if [ "$status" = "200" ]; then
        echo "✅ $service: Healthy"
    else
        echo "❌ $service: Unhealthy (HTTP $status)"
    fi
done

# Database health
echo -e "\nDatabase Health:"
if docker-compose exec postgres pg_isready -U trader > /dev/null 2>&1; then
    echo "✅ PostgreSQL: Connected"
else
    echo "❌ PostgreSQL: Connection failed"
fi

# Redis health
echo -e "\nRedis Health:"
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Connected"
else
    echo "❌ Redis: Connection failed"
fi

# Resource usage
echo -e "\nResource Usage:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')%"
echo "Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"

# Trading status
echo -e "\nTrading Status:"
trading_enabled=$(curl -s http://localhost:8006/api/v1/status | jq -r .trading_enabled)
echo "Trading Enabled: $trading_enabled"

# Recent performance
echo -e "\nRecent Performance:"
portfolio_value=$(curl -s http://localhost:8005/api/v1/portfolio | jq -r .total_value)