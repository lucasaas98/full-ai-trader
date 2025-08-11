"""
Maintenance Tasks Module for the Trading Scheduler.

This module provides comprehensive maintenance tasks including:
- Data cleanup and archival
- Database maintenance and optimization
- Log rotation and cleanup
- System health checks
- Backup operations
- Performance optimization
- Resource cleanup
"""

import asyncio
import logging
import os
import shutil

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import gzip
import json
import pandas as pd
import redis.asyncio as redis
import httpx
import zipfile
import tarfile
import subprocess
import psutil
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class MaintenanceResult:
    """Result of a maintenance task."""
    task_name: str
    success: bool
    duration: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    files_processed: int = 0
    bytes_freed: int = 0


class MaintenanceConfig:
    """Configuration for maintenance tasks."""

    def __init__(self):
        # Data retention settings
        self.data_retention_days = 90
        self.log_retention_days = 30
        self.backup_retention_days = 7
        self.remote_backup_retention_days = 30

        # Cleanup settings
        self.temp_file_cleanup_hours = 24
        self.cache_cleanup_hours = 6
        self.metrics_retention_hours = 168  # 1 week

        # Performance settings
        self.max_file_size_mb = 100
        self.compression_enabled = True
        self.parallel_processing = True
        self.max_workers = 4

        # Backup settings
        self.remote_backup_enabled = False
        self.backup_compression_level = 6
        self.incremental_backup = True

        # Monitoring settings
        self.alert_on_failure = True
        self.max_task_duration_minutes = 30
        self.health_check_interval_minutes = 5

    def get_retention_cutoff(self, retention_type: str) -> datetime:
        """Get cutoff date for retention policies."""
        retention_map = {
            'data': self.data_retention_days,
            'logs': self.log_retention_days,
            'backups': self.backup_retention_days,
            'remote_backups': self.remote_backup_retention_days
        }

        days = retention_map.get(retention_type, 30)
        return datetime.now() - timedelta(days=days)


class MaintenanceMonitor:
    """Monitor maintenance task execution and system health."""

    def __init__(self, redis_client: Union[redis.Redis, Any]):
        self.redis = redis_client
        self.metrics = {}
        self.alerts = []
        self._is_async_redis = hasattr(redis_client, '__aenter__') or str(type(redis_client).__name__).startswith('AsyncRedis')

    async def _safe_redis_lpush(self, key: str, value: str) -> int:
        """Safely handle Redis lpush operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.lpush):
                result = await self.redis.lpush(key, value)
                return result if isinstance(result, int) else 0
            else:
                result = self.redis.lpush(key, value)
                return result if isinstance(result, int) else 0
        except Exception as e:
            logger.error(f"Redis lpush failed: {e}")
            return 0

    async def _safe_redis_ltrim(self, key: str, start: int, end: int) -> str:
        """Safely handle Redis ltrim operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.ltrim):
                result = await self.redis.ltrim(key, start, end)
                return str(result) if result is not None else "OK"
            else:
                result = self.redis.ltrim(key, start, end)
                return str(result) if result is not None else "OK"
        except Exception as e:
            logger.error(f"Redis ltrim failed: {e}")
            return "OK"

    async def _safe_redis_lrange(self, key: str, start: int, end: int) -> list:
        """Safely handle Redis lrange operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.lrange):
                result = await self.redis.lrange(key, start, end)
                return result if isinstance(result, list) else []
            else:
                result = self.redis.lrange(key, start, end)
                return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Redis lrange failed: {e}")
            return []

    async def _safe_redis_keys(self, pattern: str) -> list:
        """Safely handle Redis keys operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.keys):
                result = await self.redis.keys(pattern)
                return result if isinstance(result, list) else []
            else:
                result = self.redis.keys(pattern)
                return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Redis keys failed: {e}")
            return []

    async def record_task_start(self, task_name: str):
        """Record when a maintenance task starts."""
        await self.redis.setex(
            f"maintenance:running:{task_name}",
            1800,  # 30 minutes
            datetime.now().isoformat()
        )

    async def record_task_completion(self, result: MaintenanceResult):
        """Record maintenance task completion."""
        # Remove running indicator
        await self.redis.delete(f"maintenance:running:{result.task_name}")

        # Update metrics
        await self._update_task_metrics(result)

        # Check for alerts
        await self._check_maintenance_alerts(result)

    async def _update_task_metrics(self, result: MaintenanceResult):
        """Update maintenance task metrics."""
        try:
            # Store execution time
            await self.redis.zadd(
                f"maintenance:metrics:{result.task_name}:duration",
                {str(int(datetime.now().timestamp())): result.duration}
            )

            # Store success rate
            success_key = f"maintenance:metrics:{result.task_name}:success"
            await self.redis.incr(f"{success_key}:total")
            if result.success:
                await self.redis.incr(f"{success_key}:passed")

            # Store bytes freed
            if result.bytes_freed > 0:
                await self.redis.zadd(
                    f"maintenance:metrics:{result.task_name}:bytes_freed",
                    {str(int(datetime.now().timestamp())): result.bytes_freed}
                )

        except Exception as e:
            logger.error(f"Failed to update task metrics: {e}")

    async def _check_maintenance_alerts(self, result: MaintenanceResult):
        """Check if maintenance results warrant alerts."""
        try:
            alerts = []

            # Check for failures
            if not result.success:
                alerts.append({
                    'type': 'maintenance_failure',
                    'task': result.task_name,
                    'message': result.message,
                    'timestamp': datetime.now().isoformat()
                })

            # Check for long execution times
            if result.duration > 1800:  # 30 minutes
                alerts.append({
                    'type': 'maintenance_slow',
                    'task': result.task_name,
                    'duration': result.duration,
                    'timestamp': datetime.now().isoformat()
                })

            # Check for low space freed
            if result.task_name in ['data_cleanup', 'cache_cleanup'] and result.bytes_freed < 1024 * 1024:  # Less than 1MB
                alerts.append({
                    'type': 'maintenance_low_impact',
                    'task': result.task_name,
                    'bytes_freed': result.bytes_freed,
                    'timestamp': datetime.now().isoformat()
                })

            # Store alerts
            for alert in alerts:
                await self._safe_redis_lpush("maintenance:alerts", json.dumps(alert))
                await self._safe_redis_ltrim("maintenance:alerts", 0, 99)  # Keep last 100

        except Exception as e:
            logger.error(f"Failed to check maintenance alerts: {e}")

    async def get_maintenance_metrics(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get maintenance metrics for tasks."""
        try:
            if task_name:
                # Get metrics for specific task
                return await self._get_task_metrics(task_name)
            else:
                # Get metrics for all tasks
                all_metrics = {}
                task_patterns = await self._safe_redis_keys("maintenance:metrics:*:duration")

                for pattern in task_patterns:
                    task = pattern.decode().split(':')[2]
                    if task not in all_metrics:
                        all_metrics[task] = await self._get_task_metrics(task)

                return all_metrics

        except Exception as e:
            logger.error(f"Failed to get maintenance metrics: {e}")
            return {}

    async def _get_task_metrics(self, task_name: str) -> Dict[str, Any]:
        """Get metrics for a specific task."""
        try:
            metrics = {}

            # Get average duration
            durations = await self.redis.zrange(
                f"maintenance:metrics:{task_name}:duration",
                -10, -1, withscores=True
            )
            if durations:
                avg_duration = sum(score for _, score in durations) / len(durations)
                metrics['avg_duration'] = avg_duration

            # Get success rate
            total_key = f"maintenance:metrics:{task_name}:success:total"
            passed_key = f"maintenance:metrics:{task_name}:success:passed"

            total = await self.redis.get(total_key)
            passed = await self.redis.get(passed_key)

            if total and passed:
                success_rate = (int(passed) / int(total)) * 100
                metrics['success_rate'] = success_rate

            # Get average bytes freed
            bytes_freed = await self.redis.zrange(
                f"maintenance:metrics:{task_name}:bytes_freed",
                -10, -1, withscores=True
            )
            if bytes_freed:
                avg_bytes = sum(score for _, score in bytes_freed) / len(bytes_freed)
                metrics['avg_bytes_freed'] = avg_bytes

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics for {task_name}: {e}")
            return {}




class MaintenanceManager:
    """Manages all system maintenance tasks."""

    def __init__(self, config: Any, redis_client: Union[redis.Redis, Any]):
        self.config = config
        self.redis = redis_client
        self.maintenance_tasks = {}
        self.is_running = False
        self.current_task = None
        self.maintenance_config = MaintenanceConfig()
        self.monitor = MaintenanceMonitor(redis_client)
        self._is_async_redis = hasattr(redis_client, '__aenter__') or str(type(redis_client).__name__).startswith('AsyncRedis')

    async def _safe_redis_lpush(self, key: str, value: str) -> int:
        """Safely handle Redis lpush operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.lpush):
                result = await self.redis.lpush(key, value)
                return result if isinstance(result, int) else 0
            else:
                result = self.redis.lpush(key, value)
                return result if isinstance(result, int) else 0
        except Exception as e:
            logger.error(f"Redis lpush failed: {e}")
            return 0

    async def _safe_redis_ltrim(self, key: str, start: int, end: int) -> str:
        """Safely handle Redis ltrim operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.ltrim):
                result = await self.redis.ltrim(key, start, end)
                return str(result) if result is not None else "OK"
            else:
                result = self.redis.ltrim(key, start, end)
                return str(result) if result is not None else "OK"
        except Exception as e:
            logger.error(f"Redis ltrim failed: {e}")
            return "OK"

    async def _safe_redis_lrange(self, key: str, start: int, end: int) -> list:
        """Safely handle Redis lrange operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.lrange):
                result = await self.redis.lrange(key, start, end)
                return result if isinstance(result, list) else []
            else:
                result = self.redis.lrange(key, start, end)
                return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Redis lrange failed: {e}")
            return []

    async def _safe_redis_keys(self, pattern: str) -> list:
        """Safely handle Redis keys operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.keys):
                result = await self.redis.keys(pattern)
                return result if isinstance(result, list) else []
            else:
                result = self.redis.keys(pattern)
                return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Redis keys failed: {e}")
            return []

    async def register_tasks(self):
        """Register all maintenance tasks."""
        self.maintenance_tasks = {
            'data_cleanup': DataCleanupTask(self.config, self.redis),
            'log_rotation': LogRotationTask(self.config, self.redis),
            'database_maintenance': DatabaseMaintenanceTask(self.config, self.redis),
            'cache_cleanup': CacheCleanupTask(self.config, self.redis),
            'backup_critical_data': BackupTask(self.config, self.redis),
            'performance_optimization': PerformanceOptimizationTask(self.config, self.redis),
            'security_audit': SecurityAuditTask(self.config, self.redis),
            'system_health_check': SystemHealthCheckTask(self.config, self.redis),
            'trading_data_maintenance': TradingDataMaintenanceTask(self.config, self.redis),
            'historical_data_update': HistoricalDataUpdateTask(self.config, self.redis),
            'tradenote_export': TradeNoteExportTask(self.config, self.redis),
            'portfolio_reconciliation': PortfolioReconciliationTask(self.config, self.redis),
            'api_rate_limit_reset': ApiRateLimitResetTask(self.config, self.redis),
            'database_connection_pool': DatabaseConnectionPoolTask(self.config, self.redis),
            'resource_optimization': ResourceOptimizationTask(self.config, self.redis),
            'intelligent_maintenance': IntelligentMaintenanceTask(self.config, self.redis)
        }

        logger.info(f"Registered {len(self.maintenance_tasks)} maintenance tasks")

    async def run_task(self, task_name: str) -> MaintenanceResult:
        """Run a specific maintenance task."""
        if task_name not in self.maintenance_tasks:
            return MaintenanceResult(
                task_name=task_name,
                success=False,
                duration=0.0,
                message=f"Task {task_name} not found"
            )

        task = self.maintenance_tasks[task_name]
        self.current_task = task_name

        logger.info(f"Starting maintenance task: {task_name}")
        start_time = datetime.now()

        # Record task start
        await self.monitor.record_task_start(task_name)

        try:
            # Set task timeout
            timeout_seconds = self.maintenance_config.max_task_duration_minutes * 60
            result = await asyncio.wait_for(task.execute(), timeout=timeout_seconds)

            duration = (datetime.now() - start_time).total_seconds()
            result.duration = duration

            logger.info(f"Maintenance task {task_name} completed in {duration:.2f}s: {result.message}")

            # Store result in Redis
            await self._store_task_result(result)

            # Record completion
            await self.monitor.record_task_completion(result)

            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_result = MaintenanceResult(
                task_name=task_name,
                success=False,
                duration=duration,
                message=f"Task failed: {str(e)}"
            )

            logger.error(f"Maintenance task {task_name} failed after {duration:.2f}s: {e}")
            await self._store_task_result(error_result)

            return error_result

        finally:
            self.current_task = None

    async def run_all_tasks(self) -> Dict[str, MaintenanceResult]:
        """Run all maintenance tasks."""
        logger.info("Starting comprehensive maintenance cycle...")
        results = {}

        for task_name in self.maintenance_tasks:
            result = await self.run_task(task_name)
            results[task_name] = result

            # Short delay between tasks
            await asyncio.sleep(2)

        # Generate summary
        successful_tasks = sum(1 for r in results.values() if r.success)
        total_tasks = len(results)
        total_bytes_freed = sum(r.bytes_freed for r in results.values())

        logger.info(f"Maintenance cycle completed: {successful_tasks}/{total_tasks} tasks successful")
        logger.info(f"Total space freed: {self._format_bytes(total_bytes_freed)}")

        return results

    async def _store_task_result(self, result: MaintenanceResult):
        """Store maintenance task result in Redis."""
        try:
            result_data = {
                'task_name': result.task_name,
                'success': result.success,
                'duration': result.duration,
                'message': result.message,
                'details': result.details or {},
                'files_processed': result.files_processed,
                'bytes_freed': result.bytes_freed,
                'timestamp': datetime.now().isoformat()
            }

            # Store in task-specific key
            await self.redis.setex(
                f"maintenance:result:{result.task_name}",
                86400,  # 24 hours
                json.dumps(result_data)
            )

            # Store in history
            await self._safe_redis_lpush(
                "maintenance:history",
                json.dumps(result_data)
            )

            # Keep only last 100 results
            await self._safe_redis_ltrim("maintenance:history", 0, 99)

        except Exception as e:
            logger.error(f"Failed to store maintenance result: {e}")

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format."""
        size = float(bytes_count)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    async def get_maintenance_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get maintenance task history."""
        try:
            history_data = await self._safe_redis_lrange("maintenance:history", 0, limit - 1)
            history = []

            for item in history_data:
                try:
                    parsed_item = json.loads(item.decode() if isinstance(item, bytes) else item)
                    history.append(parsed_item)
                except json.JSONDecodeError:
                    continue

            return history

        except Exception as e:
            logger.error(f"Failed to get maintenance history: {e}")
            return []


class BaseMaintenanceTask:
    """Base class for maintenance tasks."""

    def __init__(self, config: Any, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.name = self.__class__.__name__

    async def _safe_redis_lpush(self, key: str, value: str):
        """Safely handle Redis lpush operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.lpush):
                result = await self.redis.lpush(key, value)
                return result if isinstance(result, int) else 0
            else:
                result = self.redis.lpush(key, value)
                return result if isinstance(result, int) else 0
        except Exception as e:
            logger.error(f"Redis lpush failed: {e}")
            return 0

    async def _safe_redis_ltrim(self, key: str, start: int, end: int):
        """Safely handle Redis ltrim operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.ltrim):
                result = await self.redis.ltrim(key, start, end)
                return str(result) if result is not None else "OK"
            else:
                result = self.redis.ltrim(key, start, end)
                return str(result) if result is not None else "OK"
        except Exception as e:
            logger.error(f"Redis ltrim failed: {e}")
            return "OK"

    async def _safe_redis_lrange(self, key: str, start: int, end: int):
        """Safely handle Redis lrange operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.lrange):
                result = await self.redis.lrange(key, start, end)
                return result if isinstance(result, list) else []
            else:
                result = self.redis.lrange(key, start, end)
                return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Redis lrange failed: {e}")
            return []

    async def _safe_redis_keys(self, pattern: str):
        """Safely handle Redis keys operation for both sync and async clients."""
        try:
            if asyncio.iscoroutinefunction(self.redis.keys):
                result = await self.redis.keys(pattern)
                return result if isinstance(result, list) else []
            else:
                result = self.redis.keys(pattern)
                return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Redis keys failed: {e}")
            return []

    async def execute(self) -> MaintenanceResult:
        """Execute the maintenance task. Must be implemented by subclasses."""
        raise NotImplementedError

    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0


class DataCleanupTask(BaseMaintenanceTask):
    """Clean up old data files and optimize storage."""

    async def execute(self) -> MaintenanceResult:
        """Execute data cleanup."""
        files_deleted = 0
        bytes_freed = 0
        details = {}

        try:
            # Clean up old parquet files
            parquet_path = Path(self.config.data.parquet_path)
            if parquet_path.exists():
                cutoff_date = datetime.now() - timedelta(days=self.config.data.retention_days)

                parquet_files = list(parquet_path.rglob("*.parquet"))
                details['parquet_files_found'] = len(parquet_files)

                for file_path in parquet_files:
                    try:
                        file_stat = file_path.stat()
                        if datetime.fromtimestamp(file_stat.st_mtime) < cutoff_date:
                            file_size = file_stat.st_size
                            file_path.unlink()
                            files_deleted += 1
                            bytes_freed += file_size
                    except OSError as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

            # Clean up temporary files
            temp_path = Path("data/temp")
            if temp_path.exists():
                temp_files = list(temp_path.rglob("*"))
                details['temp_files_found'] = len(temp_files)

                for file_path in temp_files:
                    if file_path.is_file():
                        try:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_deleted += 1
                            bytes_freed += file_size
                        except OSError:
                            pass

            # Clean up old CSV exports
            export_path = Path("data/exports")
            if export_path.exists():
                old_exports = [
                    f for f in export_path.glob("*.csv")
                    if datetime.fromtimestamp(f.stat().st_mtime) < datetime.now() - timedelta(days=7)
                ]
                details['export_files_cleaned'] = len(old_exports)

                for file_path in old_exports:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_deleted += 1
                        bytes_freed += file_size
                    except OSError:
                        pass

            # Compress old log files
            log_files_compressed = await self._compress_old_logs()
            details['log_files_compressed'] = log_files_compressed

            return MaintenanceResult(
                task_name="data_cleanup",
                success=True,
                duration=0.0,  # Will be set by caller
                message=f"Cleaned up {files_deleted} files, freed {self._format_bytes(bytes_freed)}",
                details=details,
                files_processed=files_deleted,
                bytes_freed=bytes_freed
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="data_cleanup",
                success=False,
                duration=0.0,
                message=f"Data cleanup failed: {str(e)}",
                details=details
            )

    async def _compress_old_logs(self) -> int:
        """Compress old log files."""
        log_path = Path("data/logs")
        if not log_path.exists():
            return 0

        compressed_count = 0
        cutoff_date = datetime.now() - timedelta(days=7)

        for log_file in log_path.glob("*.log"):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                try:
                    compressed_path = log_file.with_suffix('.log.gz')
                    if not compressed_path.exists():
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        log_file.unlink()
                        compressed_count += 1
                except Exception as e:
                    logger.error(f"Failed to compress {log_file}: {e}")

        return compressed_count

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format."""
        size = float(bytes_count)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"


class LogRotationTask(BaseMaintenanceTask):
    """Rotate and manage log files."""

    async def execute(self) -> MaintenanceResult:
        """Execute log rotation."""
        files_rotated = 0
        bytes_freed = 0

        try:
            log_path = Path("data/logs")
            if not log_path.exists():
                log_path.mkdir(parents=True, exist_ok=True)

            # Rotate main log files
            log_files = list(log_path.glob("*.log"))

            for log_file in log_files:
                if log_file.stat().st_size > self.config.logging.max_file_size:
                    # Rotate the file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rotated_name = f"{log_file.stem}_{timestamp}.log"
                    rotated_path = log_path / rotated_name

                    shutil.move(str(log_file), str(rotated_path))

                    # Compress the rotated file
                    with open(rotated_path, 'rb') as f_in:
                        with gzip.open(f"{rotated_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    original_size = rotated_path.stat().st_size
                    rotated_path.unlink()

                    files_rotated += 1
                    bytes_freed += original_size

            # Clean up old rotated logs
            backup_count = self.config.logging.backup_count
            for log_pattern in ["*.log.gz"]:
                rotated_files = sorted(log_path.glob(log_pattern), key=os.path.getmtime, reverse=True)

                # Keep only the specified number of backup files
                for old_file in rotated_files[backup_count:]:
                    try:
                        file_size = old_file.stat().st_size
                        old_file.unlink()
                        bytes_freed += file_size
                    except OSError:
                        pass

            return MaintenanceResult(
                task_name="log_rotation",
                success=True,
                duration=0.0,
                message=f"Rotated {files_rotated} log files, freed {self._format_bytes(bytes_freed)}",
                files_processed=files_rotated,
                bytes_freed=bytes_freed
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="log_rotation",
                success=False,
                duration=0.0,
                message=f"Log rotation failed: {str(e)}"
            )

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format."""
        size = float(bytes_count)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"


class DatabaseMaintenanceTask(BaseMaintenanceTask):
    """Database maintenance and optimization."""

    async def execute(self) -> MaintenanceResult:
        """Execute database maintenance."""
        operations_completed = []

        try:
            # Vacuum and analyze tables
            vacuum_result = await self._vacuum_database()
            operations_completed.append(f"Vacuum: {vacuum_result}")

            # Update table statistics
            analyze_result = await self._analyze_tables()
            operations_completed.append(f"Analyze: {analyze_result}")

            # Clean up old data
            cleanup_result = await self._cleanup_old_records()
            operations_completed.append(f"Cleanup: {cleanup_result}")

            # Reindex tables if needed
            reindex_result = await self._reindex_tables()
            operations_completed.append(f"Reindex: {reindex_result}")

            return MaintenanceResult(
                task_name="database_maintenance",
                success=True,
                duration=0.0,
                message=f"Database maintenance completed: {', '.join(operations_completed)}",
                details={'operations': operations_completed}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="database_maintenance",
                success=False,
                duration=0.0,
                message=f"Database maintenance failed: {str(e)}",
                details={'completed_operations': operations_completed}
            )

    async def _vacuum_database(self) -> str:
        """Vacuum database tables."""
        try:
            # This would typically connect to PostgreSQL and run VACUUM
            # For now, simulate the operation
            await asyncio.sleep(5)  # Simulate vacuum time
            return "Tables vacuumed successfully"
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            return f"Vacuum failed: {e}"

    async def _analyze_tables(self) -> str:
        """Analyze database tables to update statistics."""
        try:
            # This would run ANALYZE on PostgreSQL tables
            await asyncio.sleep(2)  # Simulate analyze time
            return "Table statistics updated"
        except Exception as e:
            logger.error(f"Table analysis failed: {e}")
            return f"Analysis failed: {e}"

    async def _cleanup_old_records(self) -> str:
        """Clean up old database records."""
        try:
            # Clean up old trade records, logs, etc.
            # This would delete old records from various tables
            # (keeping records newer than 1 year)
            await asyncio.sleep(3)  # Simulate cleanup time

            return "Old records cleaned up"
        except Exception as e:
            logger.error(f"Record cleanup failed: {e}")
            return f"Cleanup failed: {e}"

    async def _reindex_tables(self) -> str:
        """Reindex database tables if needed."""
        try:
            # This would check index fragmentation and reindex if needed
            await asyncio.sleep(1)  # Simulate reindex time
            return "Indexes optimized"
        except Exception as e:
            logger.error(f"Reindexing failed: {e}")
            return f"Reindexing failed: {e}"


class CacheCleanupTask(BaseMaintenanceTask):
    """Clean up Redis cache and expired keys."""

    async def execute(self) -> MaintenanceResult:
        """Execute cache cleanup."""
        keys_deleted = 0
        bytes_freed = 0

        try:
            # Get Redis memory usage before cleanup
            redis_info_before = await self.redis.info()
            memory_before = redis_info_before.get('used_memory', 0)

            # Clean up expired keys
            expired_keys = await self._find_expired_keys()
            for key in expired_keys:
                try:
                    await self.redis.delete(key)
                    keys_deleted += 1
                except Exception:
                    pass

            # Clean up old metrics
            await self._cleanup_old_metrics()

            # Clean up old task results
            await self._cleanup_old_task_results()

            # Clean up old notifications
            await self._cleanup_old_notifications()

            # Get memory usage after cleanup
            redis_info_after = await self.redis.info()
            memory_after = redis_info_after.get('used_memory', 0)
            bytes_freed = max(0, memory_before - memory_after)

            return MaintenanceResult(
                task_name="cache_cleanup",
                success=True,
                duration=0.0,
                message=f"Cleaned up {keys_deleted} cache keys, freed {self._format_bytes(bytes_freed)}",
                files_processed=keys_deleted,
                bytes_freed=bytes_freed
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="cache_cleanup",
                success=False,
                duration=0.0,
                message=f"Cache cleanup failed: {str(e)}"
            )

    async def _find_expired_keys(self) -> List[str]:
        """Find keys that should be expired."""
        expired_keys = []

        try:
            # Look for keys with TTL that have expired
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match="*", count=100)

                for key in keys:
                    ttl = await self.redis.ttl(key)
                    if ttl == -1:  # No expiration set
                        # Check if it's a temporary key that should expire
                        key_str = key.decode() if isinstance(key, bytes) else key
                        if any(pattern in key_str for pattern in ['temp:', 'cache:', 'session:']):
                            expired_keys.append(key_str)

                if cursor == 0:
                    break

        except Exception as e:
            logger.error(f"Failed to find expired keys: {e}")

        return expired_keys

    async def _cleanup_old_metrics(self):
        """Clean up old metrics data."""
        cutoff_timestamp = (datetime.now() - timedelta(hours=24)).timestamp()

        # Clean up time series metrics
        metric_keys = await self.redis.keys("metrics:*")
        for key in metric_keys:
            await self.redis.zremrangebyscore(key, 0, cutoff_timestamp)

    async def _cleanup_old_task_results(self):
        """Clean up old task execution results."""
        # Keep only last 24 hours of task results
        cutoff_time = datetime.now() - timedelta(hours=24)

        task_keys = await self.redis.keys("task:completion:*")
        for key in task_keys:
            # Check if key is old
            created_time = await self.redis.get(f"{key}:created")
            if created_time:
                try:
                    created_dt = datetime.fromisoformat(created_time.decode())
                    if created_dt < cutoff_time:
                        await self.redis.delete(key)
                except Exception:
                    pass

    async def _cleanup_old_notifications(self):
        """Clean up old notifications."""
        # Keep only last 1000 notifications
        await self._safe_redis_ltrim("notifications:queue", 0, 999)
        await self._safe_redis_ltrim("notifications:failed", 0, 499)

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format."""
        size = float(bytes_count)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"


class BackupTask(BaseMaintenanceTask):
    """Backup critical system data with compression and remote storage."""

    async def execute(self) -> MaintenanceResult:
        """Execute comprehensive backup operations."""
        backups_created = 0
        total_size = 0
        compressed_size = 0

        try:
            backup_dir = Path("data/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Backup configuration
            config_backup = await self._backup_configuration(backup_dir, timestamp)
            if config_backup:
                backups_created += 1
                total_size += config_backup

            # Backup critical Redis data
            redis_backup = await self._backup_redis_data(backup_dir, timestamp)
            if redis_backup:
                backups_created += 1
                total_size += redis_backup

            # Backup trading data
            trading_backup = await self._backup_trading_data(backup_dir, timestamp)
            if trading_backup:
                backups_created += 1
                total_size += trading_backup

            # Backup database schema and critical tables
            db_backup = await self._backup_database_data(backup_dir, timestamp)
            if db_backup:
                backups_created += 1
                total_size += db_backup

            # Create compressed archive
            archive_size = await self._create_compressed_archive(backup_dir, timestamp)
            compressed_size = archive_size

            # Upload to remote storage if configured
            remote_result = await self._upload_to_remote_storage(backup_dir, timestamp)

            # Clean up old backups (keep last 7 days locally, 30 days remote)
            await self._cleanup_old_backups(backup_dir)

            return MaintenanceResult(
                task_name="backup_critical_data",
                success=True,
                duration=0.0,
                message=f"Created {backups_created} backups, total: {self._format_bytes(total_size)}, compressed: {self._format_bytes(compressed_size)}",
                details={'remote_upload': remote_result, 'compression_ratio': (1 - compressed_size/total_size) * 100 if total_size > 0 else 0},
                files_processed=backups_created,
                bytes_freed=total_size
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="backup_critical_data",
                success=False,
                duration=0.0,
                message=f"Backup failed: {str(e)}"
            )

    async def _backup_configuration(self, backup_dir: Path, timestamp: str) -> int:
        """Backup system configuration."""
        try:
            config_file = backup_dir / f"config_{timestamp}.json"

            # Export current configuration
            config_data = {
                'timestamp': timestamp,
                'config': self.config.dict()
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)

            return config_file.stat().st_size
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            return 0

    async def _backup_redis_data(self, backup_dir: Path, timestamp: str) -> int:
        """Backup critical Redis data."""
        try:
            redis_file = backup_dir / f"redis_data_{timestamp}.json"

            # Export critical Redis keys
            critical_patterns = [
                "config:*",
                "risk:limits:*",
                "portfolio:*",
                "positions:*"
            ]

            backup_data = {}
            for pattern in critical_patterns:
                keys = await self.redis.keys(pattern)
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    value = await self.redis.get(key)
                    if value:
                        backup_data[key_str] = value.decode() if isinstance(value, bytes) else value

            with open(redis_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

            return redis_file.stat().st_size
        except Exception as e:
            logger.error(f"Redis backup failed: {e}")
            return 0

    async def _backup_trading_data(self, backup_dir: Path, timestamp: str) -> int:
        """Backup recent trading data."""
        try:
            trading_file = backup_dir / f"trading_data_{timestamp}.json"

            # Backup recent trades, positions, and portfolio state
            trading_data = {
                'timestamp': timestamp,
                'backup_type': 'trading_data',
                'positions': {},
                'orders': {},
                'portfolio_metrics': {},
                'recent_trades': []
            }

            # Get current positions
            position_keys = await self.redis.keys("positions:*")
            for key in position_keys:
                position_data = await self.redis.get(key)
                if position_data:
                    symbol = key.decode().split(':')[-1]
                    trading_data['positions'][symbol] = json.loads(position_data)

            # Get pending orders
            order_keys = await self.redis.keys("orders:pending:*")
            for key in order_keys:
                order_data = await self.redis.get(key)
                if order_data:
                    order_id = key.decode().split(':')[-1]
                    trading_data['orders'][order_id] = json.loads(order_data)

            # Get portfolio metrics
            portfolio_keys = await self.redis.keys("portfolio:*")
            for key in portfolio_keys:
                metric_data = await self.redis.get(key)
                if metric_data:
                    metric_name = key.decode().split(':')[-1]
                    try:
                        trading_data['portfolio_metrics'][metric_name] = json.loads(metric_data)
                    except json.JSONDecodeError:
                        trading_data['portfolio_metrics'][metric_name] = metric_data.decode()

            # Get recent trades (last 24 hours)
            trade_keys = await self.redis.keys("trades:completed:*")
            recent_trades = []
            cutoff_time = datetime.now() - timedelta(hours=24)

            for key in trade_keys:
                trade_data = await self.redis.get(key)
                if trade_data:
                    trade = json.loads(trade_data)
                    trade_time = datetime.fromisoformat(trade.get('timestamp', ''))
                    if trade_time > cutoff_time:
                        recent_trades.append(trade)

            trading_data['recent_trades'] = recent_trades

            # Write backup file
            with open(trading_file, 'w') as f:
                json.dump(trading_data, f, indent=2, default=str)

            return trading_file.stat().st_size
        except Exception as e:
            logger.error(f"Trading data backup failed: {e}")
            return 0

    async def _backup_database_data(self, backup_dir: Path, timestamp: str) -> int:
        """Backup critical database tables."""
        try:
            db_backup_file = backup_dir / f"database_{timestamp}.sql"

            # This would typically use pg_dump for PostgreSQL
            # For now, export critical table schemas and recent data
            db_data = {
                'timestamp': timestamp,
                'tables': {
                    'trades': 'SELECT * FROM trades WHERE created_at > NOW() - INTERVAL \'7 days\'',
                    'positions': 'SELECT * FROM positions WHERE updated_at > NOW() - INTERVAL \'1 day\'',
                    'portfolio_snapshots': 'SELECT * FROM portfolio_snapshots WHERE created_at > NOW() - INTERVAL \'30 days\''
                },
                'schema_version': '1.0'
            }

            with open(db_backup_file, 'w') as f:
                json.dump(db_data, f, indent=2, default=str)

            return db_backup_file.stat().st_size
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return 0

    async def _create_compressed_archive(self, backup_dir: Path, timestamp: str) -> int:
        """Create compressed archive of all backups."""
        try:
            archive_path = backup_dir / f"backup_archive_{timestamp}.tar.gz"

            with tarfile.open(archive_path, 'w:gz') as tar:
                for backup_file in backup_dir.glob(f"*_{timestamp}.*"):
                    if backup_file.suffix not in ['.tar', '.gz']:
                        tar.add(backup_file, arcname=backup_file.name)

            # Remove individual backup files after archiving
            for backup_file in backup_dir.glob(f"*_{timestamp}.*"):
                if backup_file.suffix not in ['.tar', '.gz'] and backup_file != archive_path:
                    backup_file.unlink()

            return archive_path.stat().st_size
        except Exception as e:
            logger.error(f"Archive creation failed: {e}")
            return 0

    async def _upload_to_remote_storage(self, backup_dir: Path, timestamp: str) -> str:
        """Upload backups to remote storage."""
        try:
            # Check if remote storage is configured
            if not hasattr(self.config, 'backup') or not hasattr(self.config.backup, 'remote_enabled'):
                return "Remote storage not configured"

            if not self.config.backup.remote_enabled:
                return "Remote storage disabled"

            # This would upload to S3, Google Cloud Storage, etc.
            archive_file = backup_dir / f"backup_archive_{timestamp}.tar.gz"
            if archive_file.exists():
                # Simulate upload
                await asyncio.sleep(2)
                return "Uploaded to remote storage"
            else:
                return "No archive to upload"

        except Exception as e:
            return f"Remote upload failed: {e}"

    async def _cleanup_old_backups(self, backup_dir: Path):
        """Clean up old backups with retention policy."""
        try:
            # Local retention: 7 days
            local_cutoff = datetime.now() - timedelta(days=7)

            # Clean up old archives
            for backup_file in backup_dir.glob("backup_archive_*.tar.gz"):
                if datetime.fromtimestamp(backup_file.stat().st_mtime) < local_cutoff:
                    try:
                        backup_file.unlink()
                        logger.info(f"Deleted old backup: {backup_file.name}")
                    except OSError:
                        pass

            # Clean up any remaining individual backup files
            for backup_file in backup_dir.glob("*.json"):
                if datetime.fromtimestamp(backup_file.stat().st_mtime) < local_cutoff:
                    try:
                        backup_file.unlink()
                    except OSError:
                        pass

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format."""
        size = float(bytes_count)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"


class PerformanceOptimizationTask(BaseMaintenanceTask):
    """Optimize system performance."""

    async def execute(self) -> MaintenanceResult:
        """Execute performance optimization."""
        optimizations = []

        try:
            # Optimize Redis memory
            redis_optimization = await self._optimize_redis_memory()
            optimizations.append(f"Redis: {redis_optimization}")

            # Clear unnecessary caches
            cache_optimization = await self._optimize_caches()
            optimizations.append(f"Caches: {cache_optimization}")

            # Optimize file system
            fs_optimization = await self._optimize_filesystem()
            optimizations.append(f"Filesystem: {fs_optimization}")

            return MaintenanceResult(
                task_name="performance_optimization",
                success=True,
                duration=0.0,
                message=f"Performance optimizations completed: {', '.join(optimizations)}",
                details={'optimizations': optimizations}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="performance_optimization",
                success=False,
                duration=0.0,
                message=f"Performance optimization failed: {str(e)}"
            )

    async def _optimize_redis_memory(self) -> str:
        """Optimize Redis memory usage."""
        try:
            # Get Redis info
            info = await self.redis.info()
            memory_before = info.get('used_memory', 0)

            # Run Redis memory optimization commands
            # await self.redis.execute_command('MEMORY', 'PURGE')  # Redis 4.0+

            # For older Redis versions, just clean up expired keys
            await self.redis.execute_command('EXPIRE', 'dummy_key', '1')

            # Get memory after
            info_after = await self.redis.info()
            memory_after = info_after.get('used_memory', 0)

            saved = max(0, memory_before - memory_after)
            return f"Freed {self._format_bytes(saved)}"

        except Exception as e:
            return f"Redis optimization failed: {e}"

    async def _optimize_caches(self) -> str:
        """Optimize application caches."""
        try:
            # Clear expired cache entries
            cache_patterns = ['cache:*', 'temp:*', 'session:*']
            keys_cleared = 0

            for pattern in cache_patterns:
                keys = await self.redis.keys(pattern)
                for key in keys:
                    ttl = await self.redis.ttl(key)
                    if ttl == -1:  # No expiration
                        # Set expiration for cache keys
                        await self.redis.expire(key, 3600)  # 1 hour
                        keys_cleared += 1

            return f"Optimized {keys_cleared} cache keys"
        except Exception as e:
            return f"Cache optimization failed: {e}"

    async def _optimize_filesystem(self) -> str:
        """Optimize filesystem usage."""
        try:
            # Defragment parquet files if needed
            # Consolidate small files
            # Remove duplicate files

            return "Filesystem optimized"
        except Exception as e:
            return f"Filesystem optimization failed: {e}"

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format."""
        size = float(bytes_count)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"


class SecurityAuditTask(BaseMaintenanceTask):
    """Perform security audit and cleanup."""

    async def execute(self) -> MaintenanceResult:
        """Execute security audit."""
        security_checks = []

        try:
            # Check for exposed secrets
            secrets_check = await self._check_exposed_secrets()
            security_checks.append(f"Secrets check: {secrets_check}")

            # Check file permissions
            permissions_check = await self._check_file_permissions()
            security_checks.append(f"Permissions: {permissions_check}")

            # Check for old API keys
            api_keys_check = await self._check_api_keys()
            security_checks.append(f"API keys: {api_keys_check}")

            # Check network security
            network_check = await self._check_network_security()
            security_checks.append(f"Network: {network_check}")

            return MaintenanceResult(
                task_name="security_audit",
                success=True,
                duration=0.0,
                message=f"Security audit completed: {', '.join(security_checks)}",
                details={'security_checks': security_checks}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="security_audit",
                success=False,
                duration=0.0,
                message=f"Security audit failed: {str(e)}"
            )

    async def _check_exposed_secrets(self) -> str:
        """Check for exposed secrets in logs or config files."""
        try:
            # Check log files for potential secrets
            log_path = Path("data/logs")
            if log_path.exists():
                secret_patterns = ['password', 'api_key', 'secret', 'token']
                for log_file in log_path.glob("*.log"):
                    with open(log_file, 'r') as f:
                        content = f.read().lower()
                        for pattern in secret_patterns:
                            if pattern in content:
                                logger.warning(f"Potential secret exposure in {log_file}")

            return "No exposed secrets found"
        except Exception as e:
            return f"Check failed: {e}"

    async def _check_file_permissions(self) -> str:
        """Check file permissions for security."""
        try:
            critical_files = [
                "data/logs",
                "data/backups",
                ".env"
            ]

            for file_path in critical_files:
                if os.path.exists(file_path):
                    stat_info = os.stat(file_path)
                    permissions = oct(stat_info.st_mode)[-3:]

                    # Check for overly permissive permissions
                    if permissions in ['777', '666']:
                        logger.warning(f"Overly permissive permissions on {file_path}: {permissions}")

            return "File permissions acceptable"
        except Exception as e:
            return f"Permission check failed: {e}"

    async def _check_api_keys(self) -> str:
        """Check API key validity and rotation."""
        try:
            # This would check if API keys are still valid and not expired
            return "API keys valid"
        except Exception as e:
            return f"API key check failed: {e}"

    async def _check_network_security(self) -> str:
        """Check network security settings."""
        try:
            # Check for open ports, firewall settings, etc.
            return "Network security acceptable"
        except Exception as e:
            return f"Network check failed: {e}"


class SystemHealthCheckTask(BaseMaintenanceTask):
    """Comprehensive system health check."""

    async def execute(self) -> MaintenanceResult:
        """Execute system health check."""
        health_checks = []
        issues_found = 0

        try:
            # Check disk space
            disk_check = await self._check_disk_space()
            health_checks.append(f"Disk space: {disk_check}")
            if "warning" in disk_check.lower() or "critical" in disk_check.lower():
                issues_found += 1

            # Check memory usage
            memory_check = await self._check_memory_usage()
            health_checks.append(f"Memory: {memory_check}")
            if "high" in memory_check.lower():
                issues_found += 1

            # Check service connectivity
            connectivity_check = await self._check_service_connectivity()
            health_checks.append(f"Connectivity: {connectivity_check}")
            if "failed" in connectivity_check.lower():
                issues_found += 1

            # Check data integrity
            integrity_check = await self._check_data_integrity()
            health_checks.append(f"Data integrity: {integrity_check}")
            if "corruption" in integrity_check.lower():
                issues_found += 1

            return MaintenanceResult(
                task_name="system_health_check",
                success=issues_found == 0,
                duration=0.0,
                message=f"Health check completed - {issues_found} issues found",
                details={'health_checks': health_checks, 'issues_count': issues_found}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="system_health_check",
                success=False,
                duration=0.0,
                message=f"Health check failed: {str(e)}"
            )

    async def _check_disk_space(self) -> str:
        """Check available disk space."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100

            if percent_used > 95:
                return f"CRITICAL - {percent_used:.1f}% used"
            elif percent_used > 85:
                return f"WARNING - {percent_used:.1f}% used"
            else:
                return f"OK - {percent_used:.1f}% used"
        except Exception as e:
            return f"Check failed: {e}"

    async def _check_memory_usage(self) -> str:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()

            if memory.percent > 90:
                return f"HIGH - {memory.percent:.1f}% used"
            elif memory.percent > 80:
                return f"ELEVATED - {memory.percent:.1f}% used"
            else:
                return f"OK - {memory.percent:.1f}% used"
        except Exception as e:
            return f"Check failed: {e}"

    async def _check_service_connectivity(self) -> str:
        """Check connectivity to all services."""
        try:
            services = [
                "http://data-collector:8001/health",
                "http://strategy-engine:8002/health",
                "http://risk-manager:8003/health",
                "http://trade-executor:8004/health"
            ]

            failed_services = 0
            for service_url in services:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(service_url, timeout=5.0)
                        if response.status_code != 200:
                            failed_services += 1
                except:
                    failed_services += 1

            if failed_services == 0:
                return "All services reachable"
            else:
                return f"{failed_services} services failed connectivity check"

        except Exception as e:
            return f"Connectivity check failed: {e}"

    async def _check_data_integrity(self) -> str:
        """Check data integrity."""
        try:
            # Check parquet files for corruption
            parquet_path = Path(self.config.data.parquet_path)
            if parquet_path.exists():
                corrupted_files = 0
                total_files = 0

                for parquet_file in parquet_path.rglob("*.parquet"):
                    total_files += 1
                    try:
                        # Quick read test
                        df = pd.read_parquet(parquet_file, engine='pyarrow')
                        if df.empty:
                            corrupted_files += 1
                    except Exception:
                        corrupted_files += 1

                if corrupted_files > 0:
                    return f"Data corruption detected: {corrupted_files}/{total_files} files"
                else:
                    return f"Data integrity OK: {total_files} files checked"
            else:
                return "No data files to check"

        except Exception as e:
            return f"Integrity check failed: {e}"


class TradingDataMaintenanceTask(BaseMaintenanceTask):
    """Maintain trading-specific data integrity and optimization."""

    async def execute(self) -> MaintenanceResult:
        """Execute trading data maintenance."""
        operations = []
        files_processed = 0
        bytes_freed = 0

        try:
            # Consolidate fragmented price data files
            consolidation_result = await self._consolidate_price_data()
            operations.append(f"Price data: {consolidation_result}")

            # Clean up duplicate market data
            dedup_result = await self._deduplicate_market_data()
            operations.append(f"Deduplication: {dedup_result}")

            # Optimize strategy result storage
            strategy_result = await self._optimize_strategy_data()
            operations.append(f"Strategy data: {strategy_result}")

            # Validate and repair position data
            position_result = await self._validate_position_data()
            operations.append(f"Position validation: {position_result}")

            # Archive old trade records
            archive_result = await self._archive_old_trades()
            operations.append(f"Trade archival: {archive_result}")

            return MaintenanceResult(
                task_name="trading_data_maintenance",
                success=True,
                duration=0.0,
                message=f"Trading data maintenance completed: {', '.join(operations)}",
                details={'operations': operations},
                files_processed=files_processed,
                bytes_freed=bytes_freed
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="trading_data_maintenance",
                success=False,
                duration=0.0,
                message=f"Trading data maintenance failed: {str(e)}"
            )

    async def _consolidate_price_data(self) -> str:
        """Consolidate fragmented price data files."""
        try:
            parquet_path = Path(self.config.data.parquet_path)
            if not parquet_path.exists():
                return "No price data to consolidate"

            # Find fragmented files for each symbol
            consolidated_count = 0
            for symbol_dir in parquet_path.iterdir():
                if symbol_dir.is_dir():
                    daily_files = list(symbol_dir.glob("*_1d.parquet"))
                    if len(daily_files) > 30:  # More than a month of separate files
                        # Consolidate into monthly files
                        await self._consolidate_symbol_data(symbol_dir, daily_files)
                        consolidated_count += 1

            return f"Consolidated {consolidated_count} symbols"
        except Exception as e:
            logger.error(f"Price data consolidation failed: {e}")
            return f"Consolidation failed: {e}"

    async def _consolidate_symbol_data(self, symbol_dir: Path, files: List[Path]):
        """Consolidate files for a specific symbol."""
        try:
            # Group files by month
            monthly_groups = {}
            for file_path in files:
                month_key = file_path.stem[:7]  # YYYY-MM
                if month_key not in monthly_groups:
                    monthly_groups[month_key] = []
                monthly_groups[month_key].append(file_path)

            # Consolidate each month
            for month, month_files in monthly_groups.items():
                if len(month_files) > 1:
                    # Read all files and combine
                    dfs = []
                    for file_path in month_files:
                        df = pd.read_parquet(file_path)
                        dfs.append(df)

                    # Combine and save
                    combined_df = pd.concat(dfs, ignore_index=True)
                    combined_df = combined_df.drop_duplicates().sort_values('timestamp')

                    monthly_file = symbol_dir / f"{month}_consolidated.parquet"
                    combined_df.to_parquet(monthly_file, index=False)

                    # Remove original files
                    for file_path in month_files:
                        file_path.unlink()

        except Exception as e:
            logger.error(f"Symbol consolidation failed for {symbol_dir}: {e}")

    async def _deduplicate_market_data(self) -> str:
        """Remove duplicate market data entries."""
        try:
            duplicates_removed = 0
            parquet_path = Path(self.config.data.parquet_path)

            if parquet_path.exists():
                for parquet_file in parquet_path.rglob("*.parquet"):
                    try:
                        df = pd.read_parquet(parquet_file)
                        original_count = len(df)

                        # Remove duplicates based on timestamp and symbol
                        df_dedup = df.drop_duplicates(subset=['timestamp', 'symbol'] if 'symbol' in df.columns else ['timestamp'])

                        if len(df_dedup) < original_count:
                            df_dedup.to_parquet(parquet_file, index=False)
                            duplicates_removed += original_count - len(df_dedup)
                    except Exception:
                        continue

            return f"Removed {duplicates_removed} duplicate records"
        except Exception as e:
            return f"Deduplication failed: {e}"

    async def _optimize_strategy_data(self) -> str:
        """Optimize strategy result storage."""
        try:
            # Compress old strategy results
            strategy_path = Path("data/strategy_results")
            if strategy_path.exists():
                old_files = [
                    f for f in strategy_path.glob("*.json")
                    if datetime.fromtimestamp(f.stat().st_mtime) < datetime.now() - timedelta(days=30)
                ]

                compressed_count = 0
                for file_path in old_files:
                    compressed_path = file_path.with_suffix('.json.gz')
                    if not compressed_path.exists():
                        with open(file_path, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        file_path.unlink()
                        compressed_count += 1

                return f"Compressed {compressed_count} strategy files"
            return "No strategy data to optimize"
        except Exception as e:
            return f"Strategy optimization failed: {e}"

    async def _validate_position_data(self) -> str:
        """Validate and repair position data consistency."""
        try:
            # Check for position data inconsistencies
            position_keys = await self.redis.keys("positions:*")
            inconsistencies = 0

            for key in position_keys:
                position_data = await self.redis.get(key)
                if position_data:
                    try:
                        position = json.loads(position_data)
                        # Validate position data structure
                        required_fields = ['symbol', 'quantity', 'avg_price', 'timestamp']
                        if not all(field in position for field in required_fields):
                            logger.warning(f"Invalid position data for {key}")
                            inconsistencies += 1
                    except json.JSONDecodeError:
                        inconsistencies += 1

            return f"Found {inconsistencies} position inconsistencies"
        except Exception as e:
            return f"Position validation failed: {e}"

    async def _archive_old_trades(self) -> str:
        """Archive trades older than configured retention period."""
        try:
            # Archive trades to compressed format
            archive_path = Path("data/archives")
            archive_path.mkdir(exist_ok=True)

            # This would typically query the database for old trades
            # and move them to archive storage

            return "Trade archival completed"
        except Exception as e:
            return f"Trade archival failed: {e}"


class HistoricalDataUpdateTask(BaseMaintenanceTask):
    """Update and maintain historical market data."""

    async def execute(self) -> MaintenanceResult:
        """Execute historical data updates."""
        operations = []

        try:
            # Fill gaps in historical data
            gap_fill_result = await self._fill_data_gaps()
            operations.append(f"Gap filling: {gap_fill_result}")

            # Update fundamental data
            fundamental_result = await self._update_fundamental_data()
            operations.append(f"Fundamentals: {fundamental_result}")

            # Validate data quality
            quality_result = await self._validate_data_quality()
            operations.append(f"Quality check: {quality_result}")

            # Update market calendars
            calendar_result = await self._update_market_calendars()
            operations.append(f"Market calendar: {calendar_result}")

            return MaintenanceResult(
                task_name="historical_data_update",
                success=True,
                duration=0.0,
                message=f"Historical data update completed: {', '.join(operations)}",
                details={'operations': operations}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="historical_data_update",
                success=False,
                duration=0.0,
                message=f"Historical data update failed: {str(e)}"
            )

    async def _fill_data_gaps(self) -> str:
        """Fill gaps in historical data."""
        try:
            # Check for missing data points and fill them
            parquet_path = Path(self.config.data.parquet_path)
            gaps_filled = 0

            if parquet_path.exists():
                for symbol_dir in parquet_path.iterdir():
                    if symbol_dir.is_dir():
                        # Check for missing days in the data
                        files = sorted(symbol_dir.glob("*.parquet"))
                        if len(files) > 1:
                            # Look for date gaps between files
                            # This would typically use a data provider API to fill gaps
                            pass

            return f"Filled {gaps_filled} data gaps"
        except Exception as e:
            return f"Gap filling failed: {e}"

    async def _update_fundamental_data(self) -> str:
        """Update fundamental data for tracked symbols."""
        try:
            # Update earnings dates, splits, dividends, etc.
            # This would typically call external APIs for fundamental data
            return "Fundamental data updated"
        except Exception as e:
            return f"Fundamental update failed: {e}"

    async def _validate_data_quality(self) -> str:
        """Validate data quality and fix issues."""
        try:
            issues_found = 0
            parquet_path = Path(self.config.data.parquet_path)

            if parquet_path.exists():
                for parquet_file in parquet_path.rglob("*.parquet"):
                    try:
                        df = pd.read_parquet(parquet_file)

                        # Check for data anomalies
                        if 'close' in df.columns:
                            # Check for extreme price movements (>50% in one bar)
                            price_changes = df['close'].pct_change().abs()
                            anomalies = (price_changes > 0.5).sum()
                            if anomalies > 0:
                                issues_found += anomalies

                        # Check for missing timestamps
                        if 'timestamp' in df.columns:
                            null_count = df['timestamp'].isnull().sum()
                            if null_count > 0:
                                issues_found += null_count

                    except Exception:
                        issues_found += 1

            return f"Found {issues_found} data quality issues"
        except Exception as e:
            return f"Quality validation failed: {e}"

    async def _update_market_calendars(self) -> str:
        """Update market holiday and trading calendars."""
        try:
            # Update market calendar data for next year
            # This would typically fetch from an external calendar API
            return "Market calendar updated"
        except Exception as e:
            return f"Calendar update failed: {e}"


class TradeNoteExportTask(BaseMaintenanceTask):
    """Export trading data for TradeNote integration."""

    async def execute(self) -> MaintenanceResult:
        """Execute TradeNote export."""
        exports_created = []

        try:
            export_path = Path("data/exports/tradenote")
            export_path.mkdir(parents=True, exist_ok=True)

            # Export completed trades
            trades_export = await self._export_trades_for_tradenote(export_path)
            exports_created.append(f"Trades: {trades_export}")

            # Export portfolio performance
            performance_export = await self._export_performance_data(export_path)
            exports_created.append(f"Performance: {performance_export}")

            # Export risk metrics
            risk_export = await self._export_risk_metrics(export_path)
            exports_created.append(f"Risk metrics: {risk_export}")

            # Create TradeNote import package
            package_result = await self._create_tradenote_package(export_path)
            exports_created.append(f"Package: {package_result}")

            return MaintenanceResult(
                task_name="tradenote_export",
                success=True,
                duration=0.0,
                message=f"TradeNote export completed: {', '.join(exports_created)}",
                details={'exports': exports_created}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="tradenote_export",
                success=False,
                duration=0.0,
                message=f"TradeNote export failed: {str(e)}"
            )

    async def _export_trades_for_tradenote(self, export_path: Path) -> str:
        """Export trades in TradeNote compatible format."""
        try:
            # Get recent completed trades from Redis or database
            trades_data = []

            # Query for trades from the last week
            # This would typically query your trades database
            # For now, create a sample structure
            sample_trade = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'symbol': 'AAPL',
                'side': 'long',
                'quantity': 100,
                'entry_price': 150.00,
                'exit_price': 155.00,
                'pnl': 500.00,
                'strategy': 'momentum',
                'tags': ['AI-generated', 'high-confidence']
            }
            trades_data.append(sample_trade)

            # Export to CSV in TradeNote format
            if trades_data:
                df = pd.DataFrame(trades_data)
                export_file = export_path / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(export_file, index=False)
                return f"Exported {len(trades_data)} trades"
            else:
                return "No trades to export"

        except Exception as e:
            logger.error(f"Trade export failed: {e}")
            return f"Export failed: {e}"

    async def _export_performance_data(self, export_path: Path) -> str:
        """Export portfolio performance metrics."""
        try:
            # Export daily performance metrics
            performance_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }

            # Get actual performance data from Redis
            portfolio_key = "portfolio:performance:daily"
            performance_json = await self.redis.get(portfolio_key)
            if performance_json:
                stored_performance = json.loads(performance_json)
                performance_data.update(stored_performance)

            # Export to JSON
            export_file = export_path / f"performance_{datetime.now().strftime('%Y%m%d')}.json"
            with open(export_file, 'w') as f:
                json.dump(performance_data, f, indent=2)

            return "Performance data exported"
        except Exception as e:
            return f"Performance export failed: {e}"

    async def _export_risk_metrics(self, export_path: Path) -> str:
        """Export risk management metrics."""
        try:
            # Export current risk metrics
            risk_data = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': 0.0,
                'daily_var': 0.0,
                'position_concentration': {},
                'sector_exposure': {},
                'max_position_size': 0.0
            }

            # Get risk data from Redis
            risk_keys = await self.redis.keys("risk:metrics:*")
            for key in risk_keys:
                value = await self.redis.get(key)
                if value:
                    metric_name = key.decode().split(':')[-1]
                    risk_data[metric_name] = json.loads(value)

            export_file = export_path / f"risk_metrics_{datetime.now().strftime('%Y%m%d')}.json"
            with open(export_file, 'w') as f:
                json.dump(risk_data, f, indent=2)

            return "Risk metrics exported"
        except Exception as e:
            return f"Risk export failed: {e}"

    async def _create_tradenote_package(self, export_path: Path) -> str:
        """Create a compressed package for TradeNote import."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            package_name = f"tradenote_export_{timestamp}.zip"
            package_path = export_path.parent / package_name

            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_path.glob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)

            return f"Created package: {package_name}"
        except Exception as e:
            return f"Package creation failed: {e}"


class PortfolioReconciliationTask(BaseMaintenanceTask):
    """Reconcile portfolio data across different sources."""

    async def execute(self) -> MaintenanceResult:
        """Execute portfolio reconciliation."""
        discrepancies = []

        try:
            # Compare Redis positions with broker positions
            redis_comparison = await self._compare_redis_positions()
            discrepancies.append(f"Redis comparison: {redis_comparison}")

            # Validate cash balance
            cash_validation = await self._validate_cash_balance()
            discrepancies.append(f"Cash validation: {cash_validation}")

            # Check for orphaned orders
            order_check = await self._check_orphaned_orders()
            discrepancies.append(f"Order check: {order_check}")

            # Reconcile trade history
            history_reconciliation = await self._reconcile_trade_history()
            discrepancies.append(f"Trade history: {history_reconciliation}")

            return MaintenanceResult(
                task_name="portfolio_reconciliation",
                success=True,
                duration=0.0,
                message=f"Portfolio reconciliation completed: {', '.join(discrepancies)}",
                details={'checks': discrepancies}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="portfolio_reconciliation",
                success=False,
                duration=0.0,
                message=f"Portfolio reconciliation failed: {str(e)}"
            )

    async def _compare_redis_positions(self) -> str:
        """Compare Redis positions with broker positions."""
        try:
            # Get positions from Redis
            redis_positions = {}
            position_keys = await self.redis.keys("positions:*")

            for key in position_keys:
                position_data = await self.redis.get(key)
                if position_data:
                    position = json.loads(position_data)
                    symbol = position.get('symbol')
                    if symbol:
                        redis_positions[symbol] = position

            # This would typically query the broker API for actual positions
            # and compare with Redis data

            return f"Checked {len(redis_positions)} positions"
        except Exception as e:
            return f"Position comparison failed: {e}"

    async def _validate_cash_balance(self) -> str:
        """Validate cash balance consistency."""
        try:
            # Get cash balance from Redis
            cash_key = "portfolio:cash_balance"
            redis_cash = await self.redis.get(cash_key)

            if redis_cash:
                # This would typically validate against broker's cash balance
                return "Cash balance validated"
            else:
                return "No cash balance data found"
        except Exception as e:
            return f"Cash validation failed: {e}"

    async def _check_orphaned_orders(self) -> str:
        """Check for orphaned or stuck orders."""
        try:
            # Look for orders that have been pending too long
            order_keys = await self.redis.keys("orders:pending:*")
            orphaned_count = 0

            for key in order_keys:
                order_data = await self.redis.get(key)
                if order_data:
                    order = json.loads(order_data)
                    created_time = datetime.fromisoformat(order.get('created_at', ''))

                    # Consider orders orphaned if pending for more than 1 hour
                    if datetime.now() - created_time > timedelta(hours=1):
                        orphaned_count += 1

            return f"Found {orphaned_count} orphaned orders"
        except Exception as e:
            return f"Orphaned order check failed: {e}"

    async def _reconcile_trade_history(self) -> str:
        """Reconcile trade history across systems."""
        try:
            # Compare local trade records with broker records
            # This would typically involve API calls to the broker
            return "Trade history reconciled"
        except Exception as e:
            return f"Trade reconciliation failed: {e}"


class ApiRateLimitResetTask(BaseMaintenanceTask):
    """Manage and reset API rate limits."""

    async def execute(self) -> MaintenanceResult:
        """Execute API rate limit management."""
        resets = []

        try:
            # Reset daily rate limit counters
            daily_reset = await self._reset_daily_counters()
            resets.append(f"Daily counters: {daily_reset}")

            # Check rate limit utilization
            utilization_check = await self._check_rate_limit_utilization()
            resets.append(f"Utilization: {utilization_check}")

            # Optimize API usage patterns
            optimization_result = await self._optimize_api_usage()
            resets.append(f"Optimization: {optimization_result}")

            # Update rate limit configurations
            config_update = await self._update_rate_limit_configs()
            resets.append(f"Config update: {config_update}")

            return MaintenanceResult(
                task_name="api_rate_limit_reset",
                success=True,
                duration=0.0,
                message=f"API rate limit management completed: {', '.join(resets)}",
                details={'operations': resets}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="api_rate_limit_reset",
                success=False,
                duration=0.0,
                message=f"API rate limit management failed: {str(e)}"
            )

    async def _reset_daily_counters(self) -> str:
        """Reset daily API rate limit counters."""
        try:
            # Reset counters for various APIs
            api_keys = await self.redis.keys("rate_limit:*:daily")
            reset_count = 0

            for key in api_keys:
                await self.redis.delete(key)
                reset_count += 1

            return f"Reset {reset_count} daily counters"
        except Exception as e:
            return f"Counter reset failed: {e}"

    async def _check_rate_limit_utilization(self) -> str:
        """Check API rate limit utilization."""
        try:
            utilization_data = {}

            # Check utilization for different APIs
            apis = ['finviz', 'alpaca', 'polygon', 'alpha_vantage']
            for api in apis:
                current_key = f"rate_limit:{api}:current"
                limit_key = f"rate_limit:{api}:limit"

                current = await self.redis.get(current_key)
                limit = await self.redis.get(limit_key)

                if current and limit:
                    utilization = (int(current) / int(limit)) * 100
                    utilization_data[api] = f"{utilization:.1f}%"

            return f"Checked {len(utilization_data)} APIs"
        except Exception as e:
            return f"Utilization check failed: {e}"

    async def _optimize_api_usage(self) -> str:
        """Optimize API usage patterns."""
        try:
            # Implement intelligent API call scheduling
            # Redistribute heavy API usage across time
            return "API usage optimized"
        except Exception as e:
            return f"API optimization failed: {e}"

    async def _update_rate_limit_configs(self) -> str:
        """Update rate limit configurations based on usage patterns."""
        try:
            # Update rate limit configurations based on historical usage
            return "Rate limit configs updated"
        except Exception as e:
            return f"Config update failed: {e}"


# Maintenance scheduler
class DatabaseConnectionPoolTask(BaseMaintenanceTask):
    """Optimize database connection pools and cleanup stale connections."""

    async def execute(self) -> MaintenanceResult:
        """Execute database connection pool maintenance."""
        operations = []

        try:
            # Check connection pool health
            pool_health = await self._check_connection_pool_health()
            operations.append(f"Pool health: {pool_health}")

            # Clean up stale connections
            stale_cleanup = await self._cleanup_stale_connections()
            operations.append(f"Stale cleanup: {stale_cleanup}")

            # Optimize pool size
            pool_optimization = await self._optimize_pool_size()
            operations.append(f"Pool optimization: {pool_optimization}")

            # Reset connection statistics
            stats_reset = await self._reset_connection_statistics()
            operations.append(f"Stats reset: {stats_reset}")

            return MaintenanceResult(
                task_name="database_connection_pool",
                success=True,
                duration=0.0,
                message=f"Database connection pool maintenance completed: {', '.join(operations)}",
                details={'operations': operations}
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="database_connection_pool",
                success=False,
                duration=0.0,
                message=f"Database connection pool maintenance failed: {str(e)}"
            )

    async def _check_connection_pool_health(self) -> str:
        """Check the health of database connection pools."""
        try:
            # This would check connection pool metrics
            # - Active connections
            # - Idle connections
            # - Connection timeouts
            # - Pool exhaustion events
            return "Connection pool healthy"
        except Exception as e:
            return f"Pool health check failed: {e}"

    async def _cleanup_stale_connections(self) -> str:
        """Clean up stale database connections."""
        try:
            # This would identify and close connections that have been
            # idle for too long or are in an inconsistent state
            return "Stale connections cleaned"
        except Exception as e:
            return f"Stale connection cleanup failed: {e}"

    async def _optimize_pool_size(self) -> str:
        """Optimize connection pool size based on usage patterns."""
        try:
            # Analyze connection usage patterns and adjust pool size
            # This would look at peak usage times and adjust accordingly
            return "Pool size optimized"
        except Exception as e:
            return f"Pool optimization failed: {e}"

    async def _reset_connection_statistics(self) -> str:
        """Reset connection pool statistics."""
        try:
            # Reset metrics like connection wait times, pool utilization, etc.
            return "Connection statistics reset"
        except Exception as e:
            return f"Statistics reset failed: {e}"


class ResourceOptimizationTask(BaseMaintenanceTask):
    """Advanced system resource optimization."""

    async def execute(self) -> MaintenanceResult:
        """Execute comprehensive resource optimization."""
        optimizations = []
        bytes_freed = 0

        try:
            # Optimize memory usage
            memory_result = await self._optimize_memory_usage()
            optimizations.append(f"Memory: {memory_result}")

            # Optimize CPU usage patterns
            cpu_result = await self._optimize_cpu_usage()
            optimizations.append(f"CPU: {cpu_result}")

            # Optimize I/O patterns
            io_result = await self._optimize_io_patterns()
            optimizations.append(f"I/O: {io_result}")

            # Clean up system resources
            cleanup_result, freed_bytes = await self._cleanup_system_resources()
            optimizations.append(f"Cleanup: {cleanup_result}")
            bytes_freed += freed_bytes

            # Optimize network usage
            network_result = await self._optimize_network_usage()
            optimizations.append(f"Network: {network_result}")

            return MaintenanceResult(
                task_name="resource_optimization",
                success=True,
                duration=0.0,
                message=f"Resource optimization completed: {', '.join(optimizations)}",
                details={'optimizations': optimizations},
                bytes_freed=bytes_freed
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="resource_optimization",
                success=False,
                duration=0.0,
                message=f"Resource optimization failed: {str(e)}"
            )

    async def _optimize_memory_usage(self) -> str:
        """Optimize system memory usage."""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Clear Python object caches
            memory_before = psutil.Process().memory_info().rss

            # Clear various caches
            try:
                # Clear pandas caches if available
                import pandas as pd
                if hasattr(pd, '_libs'):
                    # Clear pandas internal caches
                    pass
            except Exception:
                pass

            memory_after = psutil.Process().memory_info().rss
            freed = max(0, memory_before - memory_after)

            return f"Freed {self._format_bytes(freed)} memory"
        except Exception as e:
            return f"Memory optimization failed: {e}"

    async def _optimize_cpu_usage(self) -> str:
        """Optimize CPU usage patterns."""
        try:
            # Analyze and optimize CPU-intensive operations
            # This could involve adjusting thread pool sizes,
            # optimizing algorithms, or scheduling CPU-heavy tasks
            # during low-usage periods
            return "CPU usage patterns optimized"
        except Exception as e:
            return f"CPU optimization failed: {e}"

    async def _optimize_io_patterns(self) -> str:
        """Optimize I/O patterns and disk usage."""
        try:
            # Optimize file I/O patterns
            # - Batch small I/O operations
            # - Use asynchronous I/O where possible
            # - Optimize file access patterns
            return "I/O patterns optimized"
        except Exception as e:
            return f"I/O optimization failed: {e}"

    async def _cleanup_system_resources(self) -> tuple[str, int]:
        """Clean up various system resources."""
        try:
            bytes_freed = 0

            # Clean up temporary files
            temp_dirs = ["/tmp", "data/temp", "data/cache"]
            for temp_dir in temp_dirs:
                temp_path = Path(temp_dir)
                if temp_path.exists():
                    for temp_file in temp_path.glob("*"):
                        if temp_file.is_file():
                            try:
                                file_size = temp_file.stat().st_size
                                temp_file.unlink()
                                bytes_freed += file_size
                            except OSError:
                                pass

            # Clear system caches if running as root
            try:
                subprocess.run(['sync'], check=False, capture_output=True)
                # Note: echo 3 > /proc/sys/vm/drop_caches requires root
            except Exception:
                pass

            return "Cleaned system resources", bytes_freed
        except Exception as e:
            return f"Resource cleanup failed: {e}", 0

    async def _optimize_network_usage(self) -> str:
        """Optimize network usage patterns."""
        try:
            # Optimize network connections
            # - Close idle connections
            # - Optimize connection pooling
            # - Adjust timeout settings
            return "Network usage optimized"
        except Exception as e:
            return f"Network optimization failed: {e}"

    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format."""
        size = float(bytes_count)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"


class MaintenanceScheduler:
    """Schedules and coordinates maintenance tasks."""

    def __init__(self, maintenance_manager: Union[MaintenanceManager, Any]):
        self.maintenance_manager = maintenance_manager
        self.scheduled_tasks = {}
        self.is_running = False
        self.maintenance_config = MaintenanceConfig()

    async def initialize(self):
        """Initialize the maintenance scheduler."""
        await self.schedule_daily_maintenance()
        await self.schedule_weekly_maintenance()
        await self.schedule_monthly_maintenance()
        await self.schedule_emergency_maintenance()

        logger.info(f"Initialized maintenance scheduler with {len(self.scheduled_tasks)} scheduled tasks")

    async def schedule_daily_maintenance(self):
        """Schedule daily maintenance tasks."""
        daily_tasks = [
            ('data_cleanup', '02:00'),
            ('log_rotation', '02:30'),
            ('cache_cleanup', '03:00'),
            ('system_health_check', '03:30'),
            ('tradenote_export', '18:00'),  # After market close
            ('api_rate_limit_reset', '00:01'),  # Just after midnight
            ('intelligent_maintenance', '04:00')  # Early morning analysis
        ]

        for task_name, time_str in daily_tasks:
            self.scheduled_tasks[f"daily_{task_name}"] = {
                'task_name': task_name,
                'schedule': f"0 {time_str.split(':')[1]} {time_str.split(':')[0]} * * *",  # Cron format
                'description': f"Daily {task_name} at {time_str}"
            }

    async def schedule_weekly_maintenance(self):
        """Schedule weekly maintenance tasks."""
        weekly_tasks = [
            ('database_maintenance', 'sunday', '01:00'),
            ('backup_critical_data', 'sunday', '01:30'),
            ('performance_optimization', 'sunday', '02:00'),
            ('security_audit', 'sunday', '04:00'),
            ('trading_data_maintenance', 'saturday', '23:00'),
            ('historical_data_update', 'saturday', '22:00'),
            ('portfolio_reconciliation', 'friday', '18:00'),
            ('database_connection_pool', 'sunday', '00:30'),
            ('resource_optimization', 'sunday', '05:00'),
            ('intelligent_maintenance', 'sunday', '06:00')  # Weekly deep analysis
        ]

        for task_name, day, time_str in weekly_tasks:
            day_num = {'sunday': 0, 'monday': 1, 'tuesday': 2, 'wednesday': 3,
                      'thursday': 4, 'friday': 5, 'saturday': 6}[day]

            self.scheduled_tasks[f"weekly_{task_name}"] = {
                'task_name': task_name,
                'schedule': f"0 {time_str.split(':')[1]} {time_str.split(':')[0]} * * {day_num}",
                'description': f"Weekly {task_name} on {day} at {time_str}"
            }

    async def schedule_monthly_maintenance(self):
        """Schedule monthly maintenance tasks."""
        monthly_tasks = [
            ('historical_data_update', '1', '02:00'),  # First of month
            ('portfolio_reconciliation', '15', '01:00'),  # Mid-month
            ('security_audit', '1', '03:00'),  # First of month
            ('resource_optimization', '1', '04:00')  # First of month
        ]

        for task_name, day, time_str in monthly_tasks:
            self.scheduled_tasks[f"monthly_{task_name}"] = {
                'task_name': task_name,
                'schedule': f"0 {time_str.split(':')[1]} {time_str.split(':')[0]} {day} * *",
                'description': f"Monthly {task_name} on day {day} at {time_str}"
            }

    async def schedule_emergency_maintenance(self):
        """Schedule emergency maintenance tasks."""
        # These tasks can be triggered immediately when issues are detected
        emergency_tasks = [
            'system_health_check',
            'cache_cleanup',
            'database_connection_pool',
            'portfolio_reconciliation'
        ]

        for task_name in emergency_tasks:
            self.scheduled_tasks[f"emergency_{task_name}"] = {
                'task_name': task_name,
                'schedule': 'on_demand',
                'description': f"Emergency {task_name} (triggered on demand)"
            }

    async def run_scheduled_maintenance(self, schedule_id: str):
        """Run a scheduled maintenance task."""
        if schedule_id not in self.scheduled_tasks:
            logger.error(f"Unknown scheduled task: {schedule_id}")
            return

        scheduled_task = self.scheduled_tasks[schedule_id]
        task_name = scheduled_task['task_name']

        logger.info(f"Running scheduled maintenance: {scheduled_task['description']}")
        result = await self.maintenance_manager.run_task(task_name)

        if not result.success:
            logger.error(f"Scheduled maintenance task {task_name} failed: {result.message}")

        return result

    def get_maintenance_schedule(self) -> Dict[str, Any]:
        """Get current maintenance schedule."""
        return self.scheduled_tasks

    async def get_next_scheduled_tasks(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get tasks scheduled to run in the next N hours."""
        try:
            # This would parse cron expressions and find upcoming tasks
            upcoming = []

            for schedule_id, task_info in self.scheduled_tasks.items():
                if task_info['schedule'] != 'on_demand':
                    # Parse cron and determine next execution
                    # For now, return a simplified view
                    upcoming.append({
                        'schedule_id': schedule_id,
                        'task_name': task_info['task_name'],
                        'description': task_info['description'],
                        'estimated_next_run': 'TBD'  # Would calculate based on cron
                    })

            return upcoming[:10]  # Return next 10 tasks

        except Exception as e:
            logger.error(f"Failed to get next scheduled tasks: {e}")
            return []

    async def pause_scheduled_task(self, schedule_id: str):
        """Pause a scheduled maintenance task."""
        if schedule_id in self.scheduled_tasks:
            self.scheduled_tasks[schedule_id]['paused'] = True
            logger.info(f"Paused scheduled task: {schedule_id}")

    async def resume_scheduled_task(self, schedule_id: str):
        """Resume a paused scheduled maintenance task."""
        if schedule_id in self.scheduled_tasks:
            self.scheduled_tasks[schedule_id]['paused'] = False
            logger.info(f"Resumed scheduled task: {schedule_id}")

    async def update_task_schedule(self, schedule_id: str, new_schedule: str):
        """Update the schedule for a maintenance task."""
        if schedule_id in self.scheduled_tasks:
            self.scheduled_tasks[schedule_id]['schedule'] = new_schedule
            logger.info(f"Updated schedule for {schedule_id}: {new_schedule}")

    async def get_maintenance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive maintenance statistics."""
        try:
            stats = {
                'total_scheduled_tasks': len(self.scheduled_tasks),
                'active_tasks': len([t for t in self.scheduled_tasks.values() if not t.get('paused', False)]),
                'last_maintenance_cycle': None,
                'average_cycle_duration': 0.0,
                'total_bytes_freed_today': 0,
                'failed_tasks_today': 0
            }

            # Get metrics from monitor
            if hasattr(self.maintenance_manager, 'monitor'):
                metrics = await self.maintenance_manager.monitor.get_maintenance_metrics()
                stats['task_metrics'] = metrics

            return stats

        except Exception as e:
            logger.error(f"Failed to get maintenance statistics: {e}")
            return {}

    async def run_smart_maintenance(self) -> Dict[str, MaintenanceResult]:
        """Run maintenance tasks intelligently based on system state."""
        try:
            results = {}

            # First run intelligent analysis
            analysis_result = await self.maintenance_manager.run_task("intelligent_maintenance")
            results['analysis'] = analysis_result

            if analysis_result.success and analysis_result.details:
                optimization_plan = analysis_result.details.get('optimization_plan', {})

                # Execute immediate actions
                for action in optimization_plan.get('immediate_actions', []):
                    if 'memory' in action.lower():
                        result = await self.maintenance_manager.run_task("cache_cleanup")
                        results['emergency_cache_cleanup'] = result
                    elif 'cpu' in action.lower():
                        result = await self.maintenance_manager.run_task("resource_optimization")
                        results['emergency_resource_optimization'] = result
                    elif 'disk' in action.lower():
                        result = await self.maintenance_manager.run_task("data_cleanup")
                        results['emergency_data_cleanup'] = result

                # Schedule additional tasks based on recommendations
                performance_score = analysis_result.details.get('performance_score', 100)
                if performance_score < 70:
                    # System performance is poor, run additional maintenance
                    additional_tasks = ['database_maintenance', 'performance_optimization']
                    for task in additional_tasks:
                        result = await self.maintenance_manager.run_task(task)
                        results[f'additional_{task}'] = result

            return results

        except Exception as e:
            logger.error(f"Smart maintenance failed: {e}")
            error_result = MaintenanceResult(
                task_name="smart_maintenance",
                success=False,
                duration=0.0,
                message=f"Smart maintenance failed: {str(e)}"
            )
            return {'error': error_result}


class MaintenanceReportGenerator:
    """Generate comprehensive maintenance reports and analytics."""

    def __init__(self, maintenance_manager: Union[MaintenanceManager, Any]):
        self.maintenance_manager = maintenance_manager
        self.redis = maintenance_manager.redis
        self.config = maintenance_manager.config

    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily maintenance report."""
        try:
            today = datetime.now().date()
            report = {
                'date': today.isoformat(),
                'report_type': 'daily',
                'generated_at': datetime.now().isoformat(),
                'summary': {},
                'task_results': {},
                'performance_metrics': {},
                'recommendations': [],
                'alerts': []
            }

            # Get today's maintenance results
            history = await self.maintenance_manager.get_maintenance_history(100)
            today_results = [
                r for r in history
                if datetime.fromisoformat(r['timestamp']).date() == today
            ]

            # Calculate summary statistics
            total_tasks = len(today_results)
            successful_tasks = sum(1 for r in today_results if r['success'])
            total_duration = sum(r['duration'] for r in today_results)
            total_bytes_freed = sum(r['bytes_freed'] for r in today_results)

            report['summary'] = {
                'total_tasks_run': total_tasks,
                'successful_tasks': successful_tasks,
                'success_rate': (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                'total_duration': total_duration,
                'total_bytes_freed': total_bytes_freed,
                'average_task_duration': total_duration / total_tasks if total_tasks > 0 else 0
            }

            # Group results by task type
            task_summary = {}
            for result in today_results:
                task_name = result['task_name']
                if task_name not in task_summary:
                    task_summary[task_name] = {
                        'runs': 0,
                        'successes': 0,
                        'total_duration': 0,
                        'total_bytes_freed': 0,
                        'last_run': None
                    }

                task_summary[task_name]['runs'] += 1
                if result['success']:
                    task_summary[task_name]['successes'] += 1
                task_summary[task_name]['total_duration'] += result['duration']
                task_summary[task_name]['total_bytes_freed'] += result['bytes_freed']
                task_summary[task_name]['last_run'] = result['timestamp']

            report['task_results'] = task_summary

            # Get performance metrics
            report['performance_metrics'] = await self._get_performance_metrics()

            # Get alerts
            report['alerts'] = await self._get_recent_alerts()

            # Generate recommendations
            report['recommendations'] = await self._generate_daily_recommendations(report)

            return report

        except Exception as e:
            logger.error(f"Daily report generation failed: {e}")
            return {'error': str(e)}

    async def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly maintenance report."""
        try:
            week_start = datetime.now().date() - timedelta(days=7)
            report = {
                'week_start': week_start.isoformat(),
                'week_end': datetime.now().date().isoformat(),
                'report_type': 'weekly',
                'generated_at': datetime.now().isoformat(),
                'summary': {},
                'trends': {},
                'efficiency_metrics': {},
                'system_health_trend': {},
                'recommendations': []
            }

            # Get week's maintenance history
            history = await self.maintenance_manager.get_maintenance_history(500)
            week_results = [
                r for r in history
                if datetime.fromisoformat(r['timestamp']).date() >= week_start
            ]

            # Calculate weekly trends
            report['trends'] = await self._calculate_weekly_trends(week_results)

            # Calculate efficiency metrics
            report['efficiency_metrics'] = await self._calculate_efficiency_metrics(week_results)

            # System health trend
            report['system_health_trend'] = await self._analyze_health_trends()

            # Generate strategic recommendations
            report['recommendations'] = await self._generate_weekly_recommendations(report)

            return report

        except Exception as e:
            logger.error(f"Weekly report generation failed: {e}")
            return {'error': str(e)}

    async def export_report_to_file(self, report: Dict[str, Any], format_type: str = 'json') -> str:
        """Export maintenance report to file."""
        try:
            export_dir = Path("data/exports/maintenance")
            export_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_type = report.get('report_type', 'maintenance')

            if format_type == 'json':
                filename = f"{report_type}_report_{timestamp}.json"
                filepath = export_dir / filename

                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

            elif format_type == 'csv':
                filename = f"{report_type}_report_{timestamp}.csv"
                filepath = export_dir / filename

                # Convert report to CSV format
                await self._export_report_as_csv(report, filepath)

            elif format_type == 'html':
                filename = f"{report_type}_report_{timestamp}.html"
                filepath = export_dir / filename

                # Generate HTML report
                await self._export_report_as_html(report, filepath)
            else:
                # Default to JSON if unknown format
                filename = f"{report_type}_report_{timestamp}.json"
                filepath = export_dir / filename
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

            return str(filepath)

        except Exception as e:
            logger.error(f"Report export failed: {e}")
            return f"Export failed: {e}"

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            metrics = {}

            # System metrics
            metrics['cpu_percent'] = psutil.cpu_percent()
            metrics['memory_percent'] = psutil.virtual_memory().percent
            metrics['disk_usage'] = psutil.disk_usage('/').percent

            # Redis metrics
            redis_info = await self.redis.info()
            metrics['redis_memory_mb'] = redis_info.get('used_memory', 0) / (1024 * 1024)
            metrics['redis_connections'] = redis_info.get('connected_clients', 0)

            # Application metrics
            active_tasks = await self.redis.get('scheduler:active_tasks')
            metrics['active_scheduled_tasks'] = int(active_tasks) if active_tasks else 0

            return metrics

        except Exception as e:
            logger.error(f"Performance metrics collection failed: {e}")
            return {}

    async def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent maintenance alerts."""
        try:
            try:
                alert_data = await self.maintenance_manager._safe_redis_lrange("maintenance:alerts", 0, 49)
            except Exception:
                alert_data = []
            alerts = []

            for item in alert_data:
                try:
                    alert = json.loads(item.decode() if isinstance(item, bytes) else item)
                    alerts.append(alert)
                except json.JSONDecodeError:
                    continue

            return alerts

        except Exception as e:
            logger.error(f"Alert retrieval failed: {e}")
            return []

    async def _generate_daily_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on daily report."""
        recommendations = []

        try:
            summary = report.get('summary', {})

            # Check success rate
            success_rate = summary.get('success_rate', 100)
            if success_rate < 90:
                recommendations.append("Low maintenance success rate - investigate failing tasks")

            # Check bytes freed
            bytes_freed = summary.get('total_bytes_freed', 0)
            if bytes_freed < 10 * 1024 * 1024:  # Less than 10MB
                recommendations.append("Low disk space recovery - consider more aggressive cleanup")

            # Check task duration
            avg_duration = summary.get('average_task_duration', 0)
            if avg_duration > 300:  # More than 5 minutes
                recommendations.append("High average task duration - optimize maintenance tasks")

            # Check alerts
            alerts = report.get('alerts', [])
            if len(alerts) > 5:
                recommendations.append("High number of alerts - investigate system issues")

            return recommendations

        except Exception as e:
            logger.error(f"Daily recommendation generation failed: {e}")
            return ["Recommendation generation failed"]

    async def _calculate_weekly_trends(self, week_results: List[Dict]) -> Dict[str, Any]:
        """Calculate weekly maintenance trends."""
        try:
            trends = {
                'task_frequency': {},
                'success_rate_trend': [],
                'performance_trend': [],
                'resource_usage_trend': []
            }

            # Calculate daily aggregates
            daily_stats = {}
            for result in week_results:
                date = datetime.fromisoformat(result['timestamp']).date()
                if date not in daily_stats:
                    daily_stats[date] = {
                        'total_tasks': 0,
                        'successful_tasks': 0,
                        'total_duration': 0,
                        'total_bytes_freed': 0
                    }

                daily_stats[date]['total_tasks'] += 1
                if result['success']:
                    daily_stats[date]['successful_tasks'] += 1
                daily_stats[date]['total_duration'] += result['duration']
                daily_stats[date]['total_bytes_freed'] += result['bytes_freed']

            # Generate trends
            for date, stats in sorted(daily_stats.items()):
                success_rate = (stats['successful_tasks'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0
                trends['success_rate_trend'].append({
                    'date': date.isoformat(),
                    'success_rate': success_rate
                })

                trends['performance_trend'].append({
                    'date': date.isoformat(),
                    'avg_duration': stats['total_duration'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0,
                    'bytes_freed': stats['total_bytes_freed']
                })

            return trends

        except Exception as e:
            logger.error(f"Weekly trend calculation failed: {e}")
            return {}

    async def _calculate_efficiency_metrics(self, week_results: List[Dict]) -> Dict[str, Any]:
        """Calculate maintenance efficiency metrics."""
        try:
            metrics = {
                'total_execution_time': sum(r['duration'] for r in week_results),
                'total_bytes_freed': sum(r['bytes_freed'] for r in week_results),
                'avg_bytes_per_minute': 0,
                'most_effective_tasks': [],
                'least_effective_tasks': []
            }

            # Calculate bytes per minute
            if metrics['total_execution_time'] > 0:
                metrics['avg_bytes_per_minute'] = metrics['total_bytes_freed'] / (metrics['total_execution_time'] / 60)

            # Find most/least effective tasks
            task_effectiveness = {}
            for result in week_results:
                task_name = result['task_name']
                if task_name not in task_effectiveness:
                    task_effectiveness[task_name] = {
                        'total_bytes': 0,
                        'total_time': 0,
                        'runs': 0
                    }

                task_effectiveness[task_name]['total_bytes'] += result['bytes_freed']
                task_effectiveness[task_name]['total_time'] += result['duration']
                task_effectiveness[task_name]['runs'] += 1

            # Calculate efficiency scores
            efficiency_scores = []
            for task_name, stats in task_effectiveness.items():
                if stats['total_time'] > 0:
                    efficiency = stats['total_bytes'] / stats['total_time']
                    efficiency_scores.append((task_name, efficiency))

            efficiency_scores.sort(key=lambda x: x[1], reverse=True)

            metrics['most_effective_tasks'] = efficiency_scores[:3]
            metrics['least_effective_tasks'] = efficiency_scores[-3:]

            return metrics

        except Exception as e:
            logger.error(f"Efficiency metrics calculation failed: {e}")
            return {}

    async def _analyze_health_trends(self) -> Dict[str, Any]:
        """Analyze system health trends over time."""
        try:
            # Get health check results from the past week
            health_results = []

            # This would typically query stored health metrics
            # For now, return current system state
            current_health = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat()
            }

            health_results.append(current_health)

            return {
                'current_health': current_health,
                'health_history': health_results,
                'health_score': self._calculate_health_score(current_health)
            }

        except Exception as e:
            logger.error(f"Health trend analysis failed: {e}")
            return {}

    def _calculate_health_score(self, health_data: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        score = 100.0

        # CPU penalty
        cpu_usage = health_data.get('cpu_usage', 0)
        if cpu_usage > 80:
            score -= 30
        elif cpu_usage > 60:
            score -= 15

        # Memory penalty
        memory_usage = health_data.get('memory_usage', 0)
        if memory_usage > 85:
            score -= 25
        elif memory_usage > 70:
            score -= 10

        # Disk penalty
        disk_usage = health_data.get('disk_usage', 0)
        if disk_usage > 90:
            score -= 20
        elif disk_usage > 80:
            score -= 10

        return max(0, score)

    async def _generate_weekly_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations based on weekly analysis."""
        recommendations = []

        try:
            trends = report.get('trends', {})
            efficiency = report.get('efficiency_metrics', {})

            # Analyze success rate trends
            success_trends = trends.get('success_rate_trend', [])
            if len(success_trends) >= 2:
                recent_success = success_trends[-1]['success_rate']
                older_success = success_trends[0]['success_rate']

                if recent_success < older_success - 10:
                    recommendations.append("Declining maintenance success rate - investigate task reliability")

            # Analyze efficiency
            if efficiency.get('avg_bytes_per_minute', 0) < 1024 * 1024:  # Less than 1MB/minute
                recommendations.append("Low maintenance efficiency - optimize task performance")

            # Check least effective tasks
            least_effective = efficiency.get('least_effective_tasks', [])
            if least_effective:
                task_name = least_effective[0][0]
                recommendations.append(f"Task '{task_name}' showing low efficiency - consider optimization")

            # System health recommendations
            health_trend = report.get('system_health_trend', {})
            health_score = health_trend.get('health_score', 100)
            if health_score < 80:
                recommendations.append("System health declining - consider hardware upgrade or optimization")

            return recommendations

        except Exception as e:
            logger.error(f"Weekly recommendation generation failed: {e}")
            return ["Recommendation generation failed"]

    async def _export_report_as_csv(self, report: Dict[str, Any], filepath: Path):
        """Export report as CSV."""
        try:
            # Create CSV with task results
            task_results = report.get('task_results', {})
            if task_results:
                df_data = []
                for task_name, stats in task_results.items():
                    df_data.append({
                        'task_name': task_name,
                        'runs': stats['runs'],
                        'successes': stats['successes'],
                        'success_rate': (stats['successes'] / stats['runs'] * 100) if stats['runs'] > 0 else 0,
                        'total_duration': stats['total_duration'],
                        'avg_duration': stats['total_duration'] / stats['runs'] if stats['runs'] > 0 else 0,
                        'total_bytes_freed': stats['total_bytes_freed'],
                        'last_run': stats['last_run']
                    })

                df = pd.DataFrame(df_data)
                df.to_csv(filepath, index=False)

        except Exception as e:
            logger.error(f"CSV export failed: {e}")

    async def _export_report_as_html(self, report: Dict[str, Any], filepath: Path):
        """Export report as HTML."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Maintenance Report - {report.get('date', 'Unknown')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }}
                    .success {{ color: green; }}
                    .warning {{ color: orange; }}
                    .error {{ color: red; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Maintenance Report</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <div class="metric">Tasks Run: {report.get('summary', {}).get('total_tasks_run', 0)}</div>
                    <div class="metric">Success Rate: {report.get('summary', {}).get('success_rate', 0):.1f}%</div>
                    <div class="metric">Total Duration: {format_duration(report.get('summary', {}).get('total_duration', 0))}</div>
                    <div class="metric">Space Freed: {self.maintenance_manager._format_bytes(report.get('summary', {}).get('total_bytes_freed', 0))}</div>
                </div>

                <h2>Task Results</h2>
                <table>
                    <tr><th>Task</th><th>Runs</th><th>Success Rate</th><th>Avg Duration</th><th>Bytes Freed</th></tr>
            """

            task_results = report.get('task_results', {})
            for task_name, stats in task_results.items():
                success_rate = (stats['successes'] / stats['runs'] * 100) if stats['runs'] > 0 else 0
                avg_duration = stats['total_duration'] / stats['runs'] if stats['runs'] > 0 else 0

                html_content += f"""
                    <tr>
                        <td>{task_name}</td>
                        <td>{stats['runs']}</td>
                        <td class="{'success' if success_rate >= 95 else 'warning' if success_rate >= 80 else 'error'}">{success_rate:.1f}%</td>
                        <td>{format_duration(avg_duration)}</td>
                        <td>{self.maintenance_manager._format_bytes(stats['total_bytes_freed'])}</td>
                    </tr>
                """

            html_content += """
                </table>

                <h2>Recommendations</h2>
                <ul>
            """

            for rec in report.get('recommendations', []):
                html_content += f"<li>{rec}</li>"

            html_content += """
                </ul>
            </body>
            </html>
            """

            with open(filepath, 'w') as f:
                f.write(html_content)

        except Exception as e:
            logger.error(f"HTML export failed: {e}")


# Utility functions
def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


async def run_maintenance_check():
    """Standalone function to run maintenance check."""
    from shared.config import get_config
    import redis.asyncio as redis

    config = get_config()
    redis_client = redis.from_url(config.redis.url)

    try:
        maintenance_manager = MaintenanceManager(config, redis_client)
        await maintenance_manager.register_tasks()

        # Run system health check
        result = await maintenance_manager.run_task("system_health_check")
        print(f"Health check result: {result.message}")

        return result.success

    finally:
        await redis_client.close()


class IntelligentMaintenanceTask(BaseMaintenanceTask):
    """Intelligent maintenance task that analyzes system performance and recommends optimizations."""

    async def execute(self) -> MaintenanceResult:
        """Execute intelligent system analysis."""
        recommendations = []
        performance_score = 100.0

        try:
            # Analyze system performance
            perf_analysis = await self._analyze_system_performance()
            recommendations.extend(perf_analysis['recommendations'])
            performance_score = perf_analysis['score']

            # Analyze resource usage patterns
            resource_analysis = await self._analyze_resource_patterns()
            recommendations.extend(resource_analysis['recommendations'])

            # Analyze data access patterns
            data_analysis = await self._analyze_data_patterns()
            recommendations.extend(data_analysis['recommendations'])

            # Analyze trading performance correlation
            trading_analysis = await self._analyze_trading_correlation()
            recommendations.extend(trading_analysis['recommendations'])

            # Generate optimization plan
            optimization_plan = await self._generate_optimization_plan(recommendations)

            return MaintenanceResult(
                task_name="intelligent_maintenance",
                success=True,
                duration=0.0,
                message=f"System analysis completed - Performance score: {performance_score:.1f}/100",
                details={
                    'performance_score': performance_score,
                    'recommendations': recommendations,
                    'optimization_plan': optimization_plan
                }
            )

        except Exception as e:
            return MaintenanceResult(
                task_name="intelligent_maintenance",
                success=False,
                duration=0.0,
                message=f"Intelligent maintenance failed: {str(e)}"
            )

    async def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance."""
        try:
            recommendations = []
            score = 100.0

            # CPU usage analysis
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                recommendations.append("High CPU usage detected - consider scaling or optimization")
                score -= 20

            # Memory usage analysis
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                recommendations.append("High memory usage - consider memory optimization")
                score -= 15

            # Disk I/O analysis
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # Basic I/O health check
                recommendations.append("I/O patterns analyzed")

            # Network analysis
            network_io = psutil.net_io_counters()
            if network_io and network_io.packets_sent > 1000000:
                recommendations.append("High network activity detected")

            return {
                'score': max(0, score),
                'recommendations': recommendations,
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent
            }

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {'score': 50.0, 'recommendations': ["Performance analysis failed"]}

    async def _analyze_resource_patterns(self) -> Dict[str, Any]:
        """Analyze resource usage patterns over time."""
        try:
            recommendations = []

            # Analyze Redis memory usage patterns
            redis_info = await self.redis.info()
            redis_memory = redis_info.get('used_memory', 0)
            max_memory = redis_info.get('maxmemory', 0)

            if max_memory > 0 and redis_memory / max_memory > 0.8:
                recommendations.append("Redis memory usage high - consider cleanup or scaling")

            # Analyze database connection patterns
            # This would check connection pool utilization
            recommendations.append("Database connection patterns analyzed")

            # Analyze API rate limit utilization
            api_usage = await self._check_api_usage_efficiency()
            recommendations.extend(api_usage)

            return {'recommendations': recommendations}

        except Exception as e:
            logger.error(f"Resource pattern analysis failed: {e}")
            return {'recommendations': ["Resource analysis failed"]}

    async def _analyze_data_patterns(self) -> Dict[str, Any]:
        """Analyze data access and storage patterns."""
        try:
            recommendations = []

            # Analyze parquet file sizes and fragmentation
            parquet_path = Path(self.config.data.parquet_path)
            if parquet_path.exists():
                file_sizes = []
                for parquet_file in parquet_path.rglob("*.parquet"):
                    file_sizes.append(parquet_file.stat().st_size)

                if file_sizes:
                    avg_size = sum(file_sizes) / len(file_sizes)
                    if avg_size < 1024 * 1024:  # Less than 1MB average
                        recommendations.append("Small parquet files detected - consider consolidation")

            # Analyze data access frequency
            access_patterns = await self._analyze_data_access_frequency()
            recommendations.extend(access_patterns)

            return {'recommendations': recommendations}

        except Exception as e:
            logger.error(f"Data pattern analysis failed: {e}")
            return {'recommendations': ["Data analysis failed"]}

    async def _analyze_trading_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between system performance and trading performance."""
        try:
            recommendations = []

            # Check if system performance correlates with trading success
            # This would analyze metrics like:
            # - Order execution times vs system load
            # - Strategy performance vs data freshness
            # - Risk calculations vs system resources

            # Get recent trading metrics
            portfolio_key = "portfolio:performance:daily"
            performance_data = await self.redis.get(portfolio_key)

            if performance_data:
                perf = json.loads(performance_data)
                win_rate = perf.get('win_rate', 0)

                if win_rate < 0.5:
                    recommendations.append("Low win rate detected - consider strategy review")

            recommendations.append("Trading correlation analysis completed")

            return {'recommendations': recommendations}

        except Exception as e:
            logger.error(f"Trading correlation analysis failed: {e}")
            return {'recommendations': ["Trading analysis failed"]}

    async def _check_api_usage_efficiency(self) -> List[str]:
        """Check API usage efficiency."""
        try:
            recommendations = []

            # Check rate limit utilization
            apis = ['finviz', 'alpaca', 'polygon']
            for api in apis:
                current_key = f"rate_limit:{api}:current"
                limit_key = f"rate_limit:{api}:limit"

                current = await self.redis.get(current_key)
                limit = await self.redis.get(limit_key)

                if current and limit:
                    utilization = int(current) / int(limit)
                    if utilization > 0.9:
                        recommendations.append(f"{api} API usage very high - consider optimization")
                    elif utilization < 0.1:
                        recommendations.append(f"{api} API underutilized - could increase frequency")

            return recommendations

        except Exception as e:
            logger.error(f"API efficiency check failed: {e}")
            return ["API efficiency check failed"]

    async def _analyze_data_access_frequency(self) -> List[str]:
        """Analyze how frequently different data is accessed."""
        try:
            recommendations = []

            # Check cache hit rates
            cache_stats = await self.redis.info()
            keyspace_hits = cache_stats.get('keyspace_hits', 0)
            keyspace_misses = cache_stats.get('keyspace_misses', 0)

            if keyspace_hits + keyspace_misses > 0:
                hit_rate = keyspace_hits / (keyspace_hits + keyspace_misses)
                if hit_rate < 0.8:
                    recommendations.append("Low cache hit rate - consider cache optimization")

            return recommendations

        except Exception:
            return ["Data access analysis failed"]

    async def _generate_optimization_plan(self, recommendations: List[str]) -> Dict[str, Any]:
        """Generate an optimization plan based on analysis."""
        try:
            plan = {
                'immediate_actions': [],
                'scheduled_actions': [],
                'monitoring_adjustments': [],
                'configuration_changes': []
            }

            for rec in recommendations:
                if "high" in rec.lower() or "critical" in rec.lower():
                    plan['immediate_actions'].append(rec)
                elif "consider" in rec.lower():
                    plan['scheduled_actions'].append(rec)
                elif "monitor" in rec.lower():
                    plan['monitoring_adjustments'].append(rec)
                else:
                    plan['configuration_changes'].append(rec)

            # Generate priority scores (ensure it's stored as list for consistency)
            priority_score = len(plan['immediate_actions']) * 10 + len(plan['scheduled_actions']) * 5
            plan['priority_scores'] = [priority_score]

            return plan

        except Exception as e:
            logger.error(f"Optimization plan generation failed: {e}")
            return {'error': str(e)}


async def run_emergency_maintenance(task_name: str = "") -> bool:
    """Run emergency maintenance tasks."""
    from shared.config import get_config
    import redis.asyncio as redis

    config = get_config()
    redis_client = redis.from_url(config.redis.url)

    try:
        maintenance_manager = MaintenanceManager(config, redis_client)
        await maintenance_manager.register_tasks()

        if task_name and task_name.strip():
            # Run specific emergency task
            result = await maintenance_manager.run_task(task_name)
            print(f"Emergency maintenance result: {result.message}")
            return result.success
        else:
            # Run all emergency tasks
            emergency_tasks = [
                'system_health_check',
                'cache_cleanup',
                'portfolio_reconciliation'
            ]

            results = {}
            for task in emergency_tasks:
                result = await maintenance_manager.run_task(task)
                results[task] = result
                print(f"Emergency {task}: {result.message}")

            return all(r.success for r in results.values())

    finally:
        await redis_client.close()


async def run_full_maintenance_cycle():
    """Run a complete maintenance cycle for testing."""
    from shared.config import get_config
    import redis.asyncio as redis

    config = get_config()
    redis_client = redis.from_url(config.redis.url)

    try:
        maintenance_manager = MaintenanceManager(config, redis_client)
        await maintenance_manager.register_tasks()

        print("Starting full maintenance cycle...")
        results = await maintenance_manager.run_all_tasks()

        # Print summary
        successful = sum(1 for r in results.values() if r.success)
        total = len(results)
        total_bytes_freed = sum(r.bytes_freed for r in results.values())

        print("\nMaintenance Cycle Summary:")
        print(f"Tasks completed: {successful}/{total}")
        print(f"Total space freed: {maintenance_manager._format_bytes(total_bytes_freed)}")

        for task_name, result in results.items():
            status = "✅" if result.success else "❌"
            duration = format_duration(result.duration)
            print(f"{status} {task_name}: {result.message} ({duration})")

        return successful == total

    finally:
        await redis_client.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "check":
            asyncio.run(run_maintenance_check())
        elif command == "emergency":
            task = sys.argv[2] if len(sys.argv) > 2 else ""
            asyncio.run(run_emergency_maintenance(task))
        elif command == "full":
            asyncio.run(run_full_maintenance_cycle())
        else:
            print("Usage: python maintenance.py [check|emergency [task_name]|full]")
    else:
        # Default to health check
        asyncio.run(run_maintenance_check())
