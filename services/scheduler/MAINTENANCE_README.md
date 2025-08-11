# Trading Scheduler Maintenance System

A comprehensive maintenance system for the Full AI Trader that orchestrates system health, data cleanup, performance optimization, and automated reporting.

## Overview

The maintenance system provides:

- **Automated Task Scheduling**: Daily, weekly, and monthly maintenance cycles
- **Intelligent System Analysis**: AI-driven performance optimization recommendations
- **Data Pipeline Management**: Coordinated data cleanup and optimization
- **System Health Monitoring**: Real-time resource monitoring and alerting
- **Backup & Recovery**: Automated backups with compression and remote storage
- **TradeNote Integration**: Automated export of trading data for analysis
- **Portfolio Reconciliation**: Cross-system data validation and consistency checks
- **Performance Optimization**: Resource usage optimization and cleanup
- **Comprehensive Reporting**: Detailed analytics and trend analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Maintenance System                           │
├─────────────────────────────────────────────────────────────────┤
│  MaintenanceManager                                             │
│  ├── Task Registry & Execution                                  │
│  ├── Result Storage & History                                   │
│  └── Performance Monitoring                                     │
│                                                                 │
│  MaintenanceScheduler                                           │
│  ├── Daily/Weekly/Monthly Schedules                            │
│  ├── Emergency Task Triggers                                   │
│  └── Smart Maintenance Orchestration                           │
│                                                                 │
│  Individual Maintenance Tasks                                   │
│  ├── DataCleanupTask           ├── BackupTask                  │
│  ├── LogRotationTask           ├── SecurityAuditTask           │
│  ├── DatabaseMaintenanceTask   ├── SystemHealthCheckTask       │
│  ├── CacheCleanupTask          ├── TradingDataMaintenanceTask  │
│  ├── PerformanceOptimizationTask └── IntelligentMaintenanceTask │
│                                                                 │
│  Reporting & Analytics                                          │
│  ├── MaintenanceReportGenerator                                │
│  ├── Performance Trend Analysis                                │
│  └── Optimization Recommendations                              │
└─────────────────────────────────────────────────────────────────┘
```

## Maintenance Tasks

### Core System Tasks

#### 1. System Health Check
- **Frequency**: Every 5 minutes during market hours, hourly after hours
- **Purpose**: Monitor CPU, memory, disk usage, service connectivity
- **Alerts**: Triggers alerts for resource thresholds (CPU >80%, Memory >85%, Disk >90%)
- **Output**: Health score (0-100) and detailed diagnostics

#### 2. Data Cleanup
- **Frequency**: Daily at 2:00 AM
- **Purpose**: Clean old parquet files, temporary files, exports
- **Retention**: 90 days for market data, 30 days for logs, 7 days for temp files
- **Features**: Automatic compression of old files, duplicate removal

#### 3. Log Rotation
- **Frequency**: Daily at 2:30 AM  
- **Purpose**: Rotate large log files, compress old logs
- **Configuration**: Max file size 100MB, keep 5 backup files
- **Compression**: GZIP compression for files >7 days old

#### 4. Cache Cleanup
- **Frequency**: Daily at 3:00 AM, emergency as needed
- **Purpose**: Clean expired Redis keys, optimize memory usage
- **Features**: TTL validation, orphaned key removal, memory optimization

### Trading-Specific Tasks

#### 5. Trading Data Maintenance
- **Frequency**: Weekly on Saturday at 11:00 PM
- **Purpose**: Consolidate fragmented price data, validate consistency
- **Features**: 
  - Consolidate daily files into monthly archives
  - Remove duplicate market data entries
  - Validate data integrity and repair inconsistencies
  - Optimize storage efficiency

#### 6. Portfolio Reconciliation
- **Frequency**: Weekly on Friday at 6:00 PM
- **Purpose**: Validate portfolio data consistency across systems
- **Checks**:
  - Compare Redis positions with broker positions
  - Validate cash balance accuracy
  - Identify orphaned or stuck orders
  - Reconcile trade history discrepancies

#### 7. TradeNote Export
- **Frequency**: Daily at 6:00 PM (after market close)
- **Purpose**: Export trading data for TradeNote analysis
- **Exports**:
  - Completed trades in TradeNote CSV format
  - Portfolio performance metrics
  - Risk management statistics
  - Compressed export packages

### Database & Infrastructure

#### 8. Database Maintenance
- **Frequency**: Weekly on Sunday at 1:00 AM
- **Purpose**: Optimize database performance and cleanup
- **Operations**:
  - VACUUM and ANALYZE tables
  - Clean up old trade records (>1 year)
  - Reindex fragmented indexes
  - Update table statistics

#### 9. Backup Operations
- **Frequency**: Weekly on Sunday at 1:30 AM
- **Purpose**: Backup critical system data
- **Features**:
  - Configuration backup
  - Redis data export (positions, orders, metrics)
  - Trading data backup (recent trades, portfolio state)
  - Database schema and critical table backup
  - Compressed archives with TAR.GZ
  - Remote storage upload (if configured)
  - Automatic cleanup (7 days local, 30 days remote)

#### 10. Performance Optimization
- **Frequency**: Weekly on Sunday at 2:00 AM
- **Purpose**: System-wide performance optimization
- **Operations**:
  - Memory optimization and garbage collection
  - Redis memory optimization
  - File system cleanup and optimization
  - Network connection optimization
  - Resource usage analysis

### Security & Monitoring

#### 11. Security Audit
- **Frequency**: Weekly on Sunday at 4:00 AM
- **Purpose**: Security validation and cleanup
- **Checks**:
  - Scan logs for exposed secrets
  - Validate file permissions
  - Check API key validity and rotation
  - Network security assessment

#### 12. API Rate Limit Management
- **Frequency**: Daily at 12:01 AM
- **Purpose**: Reset and optimize API rate limits
- **Features**:
  - Reset daily counters
  - Analyze utilization patterns
  - Optimize API call distribution
  - Update rate limit configurations

## Usage

### CLI Interface

#### Basic Commands

```bash
# Check system status
python -m scheduler.cli status

# Run specific maintenance task
python -m scheduler.cli maintenance run --task data_cleanup

# Run all maintenance tasks
python -m scheduler.cli maintenance run-all

# View maintenance history
python -m scheduler.cli maintenance history --limit 20

# Generate maintenance report
python -m scheduler.cli maintenance report --type daily --format html

# View maintenance schedule
python -m scheduler.cli maintenance schedule

# Real-time maintenance dashboard
python -m scheduler.cli dashboard --refresh 5
```

#### Advanced Operations

```bash
# Emergency maintenance
python -m scheduler.cli maintenance run --task system_health_check
python -m scheduler.cli maintenance run --task cache_cleanup

# TradeNote export
python -m scheduler.cli export --type tradenote --date today

# Portfolio operations
python -m scheduler.cli positions
python -m scheduler.cli portfolio

# Service management
python -m scheduler.cli services
python -m scheduler.cli restart --service data-collector
```

### REST API

#### Maintenance Endpoints

```bash
# Run specific maintenance task
POST /maintenance/tasks/{task_name}/run

# Run all maintenance tasks  
POST /maintenance/run-all

# Get maintenance status
GET /maintenance/status

# Get maintenance history
GET /maintenance/history?limit=50

# Generate reports
POST /maintenance/reports/generate
{
  "report_type": "daily",
  "format_type": "html",
  "include_details": true
}

# Get maintenance schedule
GET /maintenance/schedule

# Pause/resume scheduled tasks
POST /maintenance/schedule/{schedule_id}/pause
POST /maintenance/schedule/{schedule_id}/resume

# Emergency maintenance
POST /maintenance/emergency?task_name=system_health_check

# Maintenance dashboard data
GET /maintenance/dashboard

# Maintenance metrics
GET /maintenance/metrics?task_name=data_cleanup
```

### Python API

#### Direct Usage

```python
import asyncio
from scheduler.maintenance import MaintenanceManager, MaintenanceScheduler
from shared.config import get_config
import redis.asyncio as redis

async def run_maintenance():
    config = get_config()
    redis_client = redis.from_url(config.redis.url)
    
    # Initialize maintenance system
    manager = MaintenanceManager(config, redis_client)
    await manager.register_tasks()
    
    # Run specific task
    result = await manager.run_task("system_health_check")
    print(f"Health check: {result.message}")
    
    # Run all tasks
    results = await manager.run_all_tasks()
    
    # Generate report
    from scheduler.maintenance import MaintenanceReportGenerator
    report_gen = MaintenanceReportGenerator(manager)
    report = await report_gen.generate_daily_report()
    
    await redis_client.close()

# Run maintenance
asyncio.run(run_maintenance())
```

#### Scheduling Integration

```python
from scheduler.maintenance import MaintenanceScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Setup with APScheduler
scheduler = AsyncIOScheduler()
maintenance_scheduler = MaintenanceScheduler(maintenance_manager)

# Schedule daily maintenance
scheduler.add_job(
    maintenance_scheduler.run_scheduled_maintenance,
    'cron',
    args=['daily_data_cleanup'],
    hour=2, minute=0,
    id='daily_maintenance'
)

scheduler.start()
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20

# Database Configuration  
DATABASE_URL=postgresql://user:pass@localhost:5432/trading
DATABASE_POOL_SIZE=10

# Backup Configuration
BACKUP_REMOTE_ENABLED=true
BACKUP_S3_BUCKET=trading-backups
BACKUP_RETENTION_DAYS=30

# Alert Configuration
ALERT_WEBHOOK_URL=https://hooks.slack.com/...
ALERT_EMAIL_ENABLED=true
ALERT_COOLDOWN_MINUTES=15

# Performance Configuration
MAX_CONCURRENT_TASKS=3
TASK_TIMEOUT_MINUTES=30
ENABLE_COMPRESSION=true
```

### Configuration File

```yaml
# config/maintenance.yml
maintenance:
  data_retention:
    parquet_files: 90  # days
    log_files: 30      # days
    temp_files: 1      # days
    backup_files: 7    # days
  
  performance:
    max_file_size_mb: 100
    compression_enabled: true
    parallel_processing: true
    max_workers: 4
  
  scheduling:
    daily_maintenance_hour: 2
    weekly_maintenance_day: "sunday"
    emergency_health_threshold: 30
  
  alerts:
    cpu_threshold: 80.0
    memory_threshold: 85.0
    disk_threshold: 90.0
    enable_notifications: true
```

## Monitoring & Alerting

### Health Metrics

The system tracks:
- **System Resources**: CPU, memory, disk usage, load average
- **Redis Metrics**: Memory usage, connection count, hit rate
- **Task Performance**: Execution time, success rate, bytes freed
- **Service Connectivity**: Health checks for all trading services
- **Data Integrity**: Parquet file validation, consistency checks

### Alert Types

- **Critical**: System resources >90%, service failures, data corruption
- **Warning**: Performance degradation, high resource usage, task failures  
- **Info**: Maintenance completion, optimization recommendations

### Alert Channels

- **Slack**: Real-time notifications with formatted messages
- **Webhook**: HTTP POST to custom endpoints
- **Email**: Critical alerts and daily summaries
- **Dashboard**: Real-time visual monitoring

## Reporting

### Daily Reports

Generated automatically at 6:00 PM:
- Task execution summary
- Resource usage trends
- Performance metrics
- Recommendations for optimization
- Export to HTML, JSON, or CSV

### Weekly Reports

Generated on Sunday at 6:00 AM:
- Weekly performance trends
- Efficiency analysis
- System health trends
- Strategic recommendations
- Task effectiveness ranking

### Custom Reports

Available on-demand:
- Performance benchmarks
- Resource usage analysis
- Error pattern analysis
- Optimization impact assessment

## Performance Optimization

### Intelligent Maintenance

The system includes AI-driven maintenance that:
- Analyzes system performance patterns
- Correlates maintenance impact with trading performance
- Recommends optimal maintenance schedules
- Automatically adjusts task priorities based on system state
- Provides predictive maintenance insights

### Resource Optimization

Automatic optimization includes:
- **Memory**: Garbage collection, cache optimization, Redis memory management
- **CPU**: Task scheduling optimization, parallel processing tuning
- **Disk**: File consolidation, compression, cleanup scheduling
- **Network**: Connection pooling, timeout optimization

## Troubleshooting

### Common Issues

#### High Resource Usage
```bash
# Emergency cleanup
python -m scheduler.cli maintenance run --task cache_cleanup
python -m scheduler.cli maintenance run --task resource_optimization

# Check system health
python -m scheduler.cli maintenance run --task system_health_check
```

#### Failed Maintenance Tasks
```bash
# Check maintenance history
python -m scheduler.cli maintenance history --limit 10

# Run specific task with verbose logging
python -m scheduler.cli maintenance run --task data_cleanup --verbose

# Generate diagnostic report
python -m scheduler.cli maintenance report --type daily
```

#### Performance Issues
```bash
# Run performance analysis
python -m scheduler.cli maintenance run --task intelligent_maintenance

# Benchmark maintenance tasks
python demo_maintenance.py --mode benchmark

# Check resource usage
python -m scheduler.cli metrics
```

### Log Analysis

Maintenance logs are stored in:
- `data/logs/maintenance_runner.log` - Main maintenance operations
- `data/logs/scheduler.log` - Overall scheduler activity
- `data/reports/maintenance/` - Detailed maintenance reports

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m scheduler.cli maintenance run --task system_health_check --verbose
```

## Integration

### Docker Compose

```yaml
services:
  scheduler:
    build: ./services/scheduler
    environment:
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:password@db:5432/trading
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
      - db
```

### Kubernetes

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-maintenance
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: maintenance
            image: trading-scheduler:latest
            command: ["python", "-m", "scheduler.maintenance_runner", "run", "--cycle-type", "daily"]
            env:
            - name: REDIS_URL
              value: "redis://redis-service:6379/0"
```

### Systemd Service

```ini
[Unit]
Description=Trading Maintenance Monitor
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/trading
ExecStart=/opt/trading/venv/bin/python -m scheduler.maintenance_runner monitor
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Testing

### Unit Tests

```bash
# Run maintenance system tests
python -m pytest services/scheduler/tests/test_maintenance.py -v

# Run specific test
python -m pytest services/scheduler/tests/test_maintenance.py::test_data_cleanup -v
```

### Integration Tests

```bash
# Run comprehensive test suite
python services/scheduler/src/maintenance_test.py

# Test specific components
python services/scheduler/src/maintenance_test.py --component scheduling
```

### Demo & Validation

```bash
# Interactive demo
python services/scheduler/src/demo_maintenance.py --mode interactive

# Automated validation
python services/scheduler/src/demo_maintenance.py --mode automated

# Performance benchmarks
python services/scheduler/src/demo_maintenance.py --mode benchmark
```

## Best Practices

### Task Development

When creating new maintenance tasks:

```python
from scheduler.maintenance import BaseMaintenanceTask, MaintenanceResult

class CustomMaintenanceTask(BaseMaintenanceTask):
    """Custom maintenance task description."""
    
    async def execute(self) -> MaintenanceResult:
        """Execute the custom maintenance task."""
        try:
            # Task implementation
            files_processed = 0
            bytes_freed = 0
            
            # Your maintenance logic here
            
            return MaintenanceResult(
                task_name="custom_task",
                success=True,
                duration=0.0,  # Set by caller
                message="Custom task completed successfully",
                details={'custom_metric': 'value'},
                files_processed=files_processed,
                bytes_freed=bytes_freed
            )
            
        except Exception as e:
            return MaintenanceResult(
                task_name="custom_task",
                success=False,
                duration=0.0,
                message=f"Custom task failed: {str(e)}"
            )
```

### Error Handling

- Always use try-catch blocks in maintenance tasks
- Return meaningful error messages
- Log errors at appropriate levels
- Don't fail the entire system for single task failures
- Implement retry logic for transient failures

### Performance Guidelines

- Use async/await for I/O operations
- Implement timeout mechanisms
- Process large datasets in chunks
- Use compression for data archival
- Monitor resource usage during execution

### Security Considerations

- Never log sensitive data (API keys, passwords)
- Validate file paths to prevent directory traversal
- Use secure file permissions (600 for sensitive files)
- Regularly rotate API keys and credentials
- Encrypt backups containing sensitive data

## Monitoring

### Metrics Collection

The system automatically collects:
- Task execution times and success rates
- Resource usage patterns
- Data processing volumes
- Error frequencies and patterns
- System health trends

### Dashboard

Real-time dashboard shows:
- Current system health
- Running maintenance tasks
- Recent task results
- Active alerts
- Performance metrics

Access via CLI:
```bash
python -m scheduler.cli dashboard
```

Or web interface:
```
http://localhost:8000/maintenance/dashboard
```

### Alerting Rules

Default alert thresholds:
- CPU usage >80% for 5 minutes
- Memory usage >85% for 5 minutes  
- Disk usage >90%
- Maintenance task failures
- Performance degradation >50%
- Service connectivity issues

## Backup & Recovery

### Backup Strategy

- **Local Backups**: 7 days retention
- **Remote Backups**: 30 days retention (if configured)
- **Incremental**: Only changed data
- **Compression**: GZIP/TAR.GZ compression
- **Encryption**: Optional AES-256 encryption

### Recovery Procedures

#### Configuration Recovery
```bash
# Restore from latest backup
python -m scheduler.cli restore --type config --backup latest

# Manual configuration restore
cp data/backups/config_YYYYMMDD_HHMMSS.json config/restored_config.json
```

#### Data Recovery
```bash
# Restore trading data
python -m scheduler.cli restore --type trading_data --date YYYY-MM-DD

# Restore Redis data
python -m scheduler.cli restore --type redis --backup latest
```

## Development

### Adding New Tasks

1. Create task class inheriting from `BaseMaintenanceTask`
2. Implement `execute()` method
3. Register in `MaintenanceManager.register_tasks()`
4. Add to appropriate schedule in `MaintenanceScheduler`
5. Add tests in `tests/test_maintenance.py`

### Testing New Features

```bash
# Run specific task tests
python -m pytest tests/test_maintenance.py::test_new_task -v

# Integration testing
python src/maintenance_test.py

# Demo validation
python src/demo_maintenance.py --mode automated
```

### Contributing

1. Follow existing code patterns
2. Add comprehensive error handling
3. Include logging and metrics
4. Write unit tests
5. Update documentation
6. Test with demo script

## Troubleshooting Guide

### Task Failures

#### Data Cleanup Issues
- Check file permissions
- Verify disk space availability
- Validate retention configuration
- Review parquet file integrity

#### Database Maintenance Problems
- Verify database connectivity
- Check for long-running transactions
- Ensure sufficient database privileges
- Monitor disk space for temp files

#### Backup Failures
- Check backup directory permissions
- Verify remote storage credentials
- Monitor available disk space
- Review compression settings

### Performance Issues

#### Slow Task Execution
- Check system resource usage
- Review parallel processing settings
- Optimize file I/O patterns
- Consider task scheduling adjustments

#### High Resource Usage
- Run resource optimization task
- Check for memory leaks
- Review cache settings
- Monitor concurrent task limits

### Configuration Issues

#### Redis Connection Problems
- Verify Redis service status
- Check connection parameters
- Review Redis memory settings
- Validate authentication

#### Scheduling Problems
- Verify timezone settings
- Check cron expression syntax
- Review task dependencies
- Monitor scheduler service health

## Support

### Log Files

- `data/logs/maintenance_runner.log` - Maintenance operations
- `data/logs/scheduler.log` - Scheduler activity
- `data/logs/error.log` - Error details
- `data/reports/maintenance/` - Detailed reports

### Metrics Storage

- Redis key patterns:
  - `maintenance:result:*` - Task results
  - `maintenance:metrics:*` - Performance metrics
  - `maintenance:alerts` - Active alerts
  - `maintenance:history` - Task history

### Contact

For issues or questions:
- Check logs in `data/logs/`
- Review maintenance reports in `data/reports/`
- Run diagnostic: `python -m scheduler.cli maintenance run --task system_health_check`
- Generate report: `python -m scheduler.cli maintenance report --type daily`

---

## License

This maintenance system is part of the Full AI Trader project.
See the main project LICENSE file for details.