"""
System Monitor for the Trading Scheduler.

This module provides comprehensive system monitoring including:
- Resource usage tracking (CPU, memory, disk, network)
- Service performance monitoring
- API rate limit tracking
- Database connection pool monitoring
- Redis queue monitoring
- Alert generation and management
- Performance metrics collection
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import httpx
import psutil
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """System alert representation."""

    id: str
    timestamp: datetime
    severity: AlertSeverity
    source: str
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Threshold:
    """Metric threshold configuration."""

    warning: float
    critical: float
    duration: int = 60  # seconds - how long threshold must be exceeded
    enabled: bool = True


class SystemMonitor:
    """Comprehensive system monitoring service."""

    def __init__(self, redis_client: redis.Redis, config: Any):
        self.redis = redis_client
        self.config = config

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: Set[asyncio.Task] = set()

        # Metrics storage
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_history_minutes = 60

        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable] = []

        # Service tracking
        self.service_metrics: Dict[str, Dict[str, Any]] = {}
        self.api_rate_limits: Dict[str, Dict[str, Any]] = {}

        # Thresholds configuration
        self.thresholds = {
            "cpu_percent": Threshold(warning=70.0, critical=85.0),
            "memory_percent": Threshold(warning=80.0, critical=90.0),
            "disk_percent": Threshold(warning=85.0, critical=95.0),
            "response_time": Threshold(warning=1000.0, critical=5000.0),  # milliseconds
            "error_rate": Threshold(warning=5.0, critical=10.0),  # percentage
            "queue_length": Threshold(warning=100.0, critical=500.0),
        }

        # Performance tracking
        self.performance_history = defaultdict(list)
        self.last_performance_check = datetime.now()

    async def start_monitoring(self):
        """Start all monitoring tasks."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        logger.info("Starting system monitoring...")
        self.is_monitoring = True

        # Start monitoring tasks
        tasks = [
            self._system_resource_monitor,
            self._service_health_monitor,
            self._api_rate_limit_monitor,
            self._database_monitor,
            self._redis_monitor,
            self._alert_processor,
            self._metrics_cleanup,
        ]

        for task_func in tasks:
            task = asyncio.create_task(task_func())
            self.monitoring_tasks.add(task)

        logger.info("System monitoring started")

    async def stop_monitoring(self):
        """Stop all monitoring tasks."""
        logger.info("Stopping system monitoring...")
        self.is_monitoring = False

        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        self.monitoring_tasks.clear()
        logger.info("System monitoring stopped")

    async def _system_resource_monitor(self):
        """Monitor system resources (CPU, memory, disk, network)."""
        while self.is_monitoring:
            try:
                timestamp = datetime.now()

                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=1)
                await self._record_metric("cpu_percent", cpu_percent, timestamp)

                # Memory monitoring
                memory = psutil.virtual_memory()
                await self._record_metric("memory_percent", memory.percent, timestamp)
                await self._record_metric(
                    "memory_used_gb", memory.used / (1024**3), timestamp
                )
                await self._record_metric(
                    "memory_available_gb", memory.available / (1024**3), timestamp
                )

                # Disk monitoring
                disk = psutil.disk_usage("/")
                disk_percent = (disk.used / disk.total) * 100
                await self._record_metric("disk_percent", disk_percent, timestamp)
                await self._record_metric(
                    "disk_used_gb", disk.used / (1024**3), timestamp
                )
                await self._record_metric(
                    "disk_free_gb", disk.free / (1024**3), timestamp
                )

                # Network monitoring
                network = psutil.net_io_counters()
                await self._record_metric(
                    "network_bytes_sent", network.bytes_sent, timestamp
                )
                await self._record_metric(
                    "network_bytes_recv", network.bytes_recv, timestamp
                )

                # Process monitoring
                process_count = len(psutil.pids())
                await self._record_metric("process_count", process_count, timestamp)

                # Load average (Unix systems)
                try:
                    load_avg = psutil.getloadavg()
                    await self._record_metric("load_avg_1m", load_avg[0], timestamp)
                    await self._record_metric("load_avg_5m", load_avg[1], timestamp)
                    await self._record_metric("load_avg_15m", load_avg[2], timestamp)
                except (AttributeError, OSError):
                    # getloadavg not available on Windows
                    pass

                # Check thresholds
                await self._check_resource_thresholds(timestamp)

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System resource monitoring error: {e}")
                await asyncio.sleep(60)

    async def _service_health_monitor(self):
        """Monitor health and performance of trading services."""
        while self.is_monitoring:
            try:
                timestamp = datetime.now()

                # Monitor each service
                services = [
                    ("data_collector", "http://data_collector:9101"),
                    ("strategy_engine", "http://strategy_engine:9102"),
                    ("risk_manager", "http://risk_manager:9103"),
                    ("trade_executor", "http://trade_executor:9104"),
                ]

                for service_name, base_url in services:
                    await self._monitor_service_performance(
                        service_name, base_url, timestamp
                    )

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Service health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _monitor_service_performance(
        self, service_name: str, base_url: str, timestamp: datetime
    ):
        """Monitor performance metrics for a specific service."""
        try:
            # Health check with timing
            start_time = time.time()

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(f"{base_url}/health", timeout=5.0)
                    response_time = (time.time() - start_time) * 1000  # milliseconds

                    is_healthy = response.status_code == 200
                    await self._record_metric(
                        f"service_{service_name}_response_time",
                        response_time,
                        timestamp,
                    )
                    await self._record_metric(
                        f"service_{service_name}_healthy",
                        1 if is_healthy else 0,
                        timestamp,
                    )

                    # Get detailed metrics if available
                    if is_healthy:
                        try:
                            metrics_response = await client.get(
                                f"{base_url}/metrics", timeout=5.0
                            )
                            if metrics_response.status_code == 200:
                                service_metrics = metrics_response.json()
                                await self._process_service_metrics(
                                    service_name, service_metrics, timestamp
                                )
                        except Exception:
                            pass  # Metrics endpoint might not exist

                except asyncio.TimeoutError:
                    await self._record_metric(
                        f"service_{service_name}_healthy", 0, timestamp
                    )
                    await self._record_metric(
                        f"service_{service_name}_response_time", 5000, timestamp
                    )
                    logger.warning(f"Health check timeout for {service_name}")

        except Exception as e:
            logger.error(f"Failed to monitor service {service_name}: {e}")
            await self._record_metric(f"service_{service_name}_healthy", 0, timestamp)

    async def _process_service_metrics(
        self, service_name: str, metrics: Dict[str, Any], timestamp: datetime
    ):
        """Process and store service-specific metrics."""
        try:
            # Store service metrics
            self.service_metrics[service_name] = metrics

            # Extract common metrics
            if "requests_total" in metrics:
                await self._record_metric(
                    f"service_{service_name}_requests_total",
                    metrics["requests_total"],
                    timestamp,
                )

            if "requests_failed" in metrics:
                await self._record_metric(
                    f"service_{service_name}_requests_failed",
                    metrics["requests_failed"],
                    timestamp,
                )

            if "active_connections" in metrics:
                await self._record_metric(
                    f"service_{service_name}_connections",
                    metrics["active_connections"],
                    timestamp,
                )

            if "memory_usage_mb" in metrics:
                await self._record_metric(
                    f"service_{service_name}_memory_mb",
                    metrics["memory_usage_mb"],
                    timestamp,
                )

            # Calculate error rate
            if "requests_total" in metrics and "requests_failed" in metrics:
                total = metrics["requests_total"]
                failed = metrics["requests_failed"]
                error_rate = (failed / total * 100) if total > 0 else 0
                await self._record_metric(
                    f"service_{service_name}_error_rate", error_rate, timestamp
                )

        except Exception as e:
            logger.error(f"Failed to process service metrics for {service_name}: {e}")

    async def _api_rate_limit_monitor(self):
        """Monitor API rate limits for external services."""
        while self.is_monitoring:
            try:
                timestamp = datetime.now()

                # Monitor known API endpoints
                api_services = {
                    "alpaca": self.config.alpaca.base_url,
                    "twelvedata": self.config.twelvedata.base_url,
                    "finviz": self.config.finviz.base_url,
                }

                for api_name, base_url in api_services.items():
                    await self._check_api_rate_limits(api_name, base_url, timestamp)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"API rate limit monitoring error: {e}")
                await asyncio.sleep(60)

    async def _check_api_rate_limits(
        self, api_name: str, base_url: str, timestamp: datetime
    ):
        """Check rate limits for a specific API."""
        try:
            # Get rate limit info from Redis
            rate_limit_key = f"api:rate_limit:{api_name}"
            rate_limit_data = await self.redis.get(rate_limit_key)

            if rate_limit_data:
                rate_limit_info = json.loads(rate_limit_data)

                requests_made = rate_limit_info.get("requests_made", 0)
                limit = rate_limit_info.get("limit", 1000)
                window_start = datetime.fromisoformat(
                    rate_limit_info.get("window_start", timestamp.isoformat())
                )

                # Calculate usage percentage
                usage_percent = (requests_made / limit * 100) if limit > 0 else 0
                await self._record_metric(
                    f"api_{api_name}_rate_limit_usage", usage_percent, timestamp
                )

                # Store API rate limit info
                self.api_rate_limits[api_name] = {
                    "requests_made": requests_made,
                    "limit": limit,
                    "usage_percent": usage_percent,
                    "window_start": window_start,
                    "last_updated": timestamp,
                }

                # Check thresholds
                if usage_percent > 80:
                    await self._create_alert(
                        f"api_rate_limit_{api_name}",
                        (
                            AlertSeverity.HIGH
                            if usage_percent > 90
                            else AlertSeverity.MEDIUM
                        ),
                        f"API rate limit usage high for {api_name}: {usage_percent:.1f}%",
                        source="monitor",
                        metadata={"api": api_name, "usage": usage_percent},
                    )

        except Exception as e:
            logger.error(f"Failed to check rate limits for {api_name}: {e}")

    async def _database_monitor(self):
        """Monitor database connection pools and performance."""
        while self.is_monitoring:
            try:
                timestamp = datetime.now()

                # Check database connectivity
                try:
                    # This would check actual database connections
                    # For now, simulate database metrics

                    # Connection pool metrics
                    await self._record_metric("db_connections_active", 5, timestamp)
                    await self._record_metric("db_connections_idle", 3, timestamp)
                    await self._record_metric("db_connections_total", 8, timestamp)

                    # Query performance
                    await self._record_metric(
                        "db_avg_query_time", 50.0, timestamp
                    )  # milliseconds
                    await self._record_metric("db_slow_queries", 0, timestamp)

                    # Database size
                    await self._record_metric("db_size_mb", 1024.0, timestamp)

                except Exception as e:
                    logger.error(f"Database monitoring error: {e}")
                    await self._record_metric("db_healthy", 0, timestamp)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Database monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _redis_monitor(self):
        """Monitor Redis performance and queue lengths."""
        while self.is_monitoring:
            try:
                timestamp = datetime.now()

                # Get Redis info
                redis_info = await self.redis.info()

                # Basic Redis metrics
                await self._record_metric(
                    "redis_connected_clients",
                    redis_info.get("connected_clients", 0),
                    timestamp,
                )
                await self._record_metric(
                    "redis_used_memory_mb",
                    redis_info.get("used_memory", 0) / (1024 * 1024),
                    timestamp,
                )
                await self._record_metric(
                    "redis_keyspace_hits", redis_info.get("keyspace_hits", 0), timestamp
                )
                await self._record_metric(
                    "redis_keyspace_misses",
                    redis_info.get("keyspace_misses", 0),
                    timestamp,
                )

                # Queue monitoring
                await self._monitor_redis_queues(timestamp)

                # Check Redis health
                try:
                    await self.redis.ping()
                    await self._record_metric("redis_healthy", 1, timestamp)
                except Exception:
                    await self._record_metric("redis_healthy", 0, timestamp)
                    await self._create_alert(
                        "redis_connection",
                        AlertSeverity.CRITICAL,
                        "Redis connection failed",
                        source="monitor",
                    )

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Redis monitoring error: {e}")
                await asyncio.sleep(60)

    async def _monitor_redis_queues(self, timestamp: datetime):
        """Monitor Redis queue lengths and performance."""
        try:
            # Task queues
            queue_names = [
                "scheduler:queue:critical",
                "scheduler:queue:high",
                "scheduler:queue:normal",
                "scheduler:queue:low",
            ]

            total_queue_length = 0
            for queue_name in queue_names:
                queue_length = await self.redis.llen(queue_name)  # type: ignore
                total_queue_length += queue_length

                queue_type = queue_name.split(":")[-1]
                await self._record_metric(
                    f"queue_{queue_type}_length", queue_length, timestamp
                )

            await self._record_metric(
                "queue_total_length", total_queue_length, timestamp
            )

            # Check queue thresholds
            if total_queue_length > self.thresholds["queue_length"].critical:
                await self._create_alert(
                    "queue_length_critical",
                    AlertSeverity.CRITICAL,
                    f"Total queue length critical: {total_queue_length}",
                    source="monitor",
                    metadata={"queue_length": total_queue_length},
                )
            elif total_queue_length > self.thresholds["queue_length"].warning:
                await self._create_alert(
                    "queue_length_warning",
                    AlertSeverity.MEDIUM,
                    f"Total queue length high: {total_queue_length}",
                    source="monitor",
                    metadata={"queue_length": total_queue_length},
                )

            # Notification queues
            notification_queues = ["notifications:queue", "notifications:failed"]

            for queue_name in notification_queues:
                queue_length = await self.redis.llen(queue_name)  # type: ignore
                queue_type = queue_name.split(":")[-1]
                await self._record_metric(
                    f"notifications_{queue_type}_length", queue_length, timestamp
                )

        except Exception as e:
            logger.error(f"Queue monitoring error: {e}")

    async def _alert_processor(self):
        """Process and manage alerts."""
        while self.is_monitoring:
            try:
                # Check for alert resolution
                current_time = datetime.now()
                alerts_to_resolve = []

                for alert_id, alert in self.active_alerts.items():
                    if not alert.resolved:
                        # Check if alert condition still exists
                        if await self._should_resolve_alert(alert):
                            alert.resolved = True
                            alert.resolved_at = current_time
                            alerts_to_resolve.append(alert_id)

                # Notify about resolved alerts
                for alert_id in alerts_to_resolve:
                    alert = self.active_alerts[alert_id]
                    logger.info(f"Alert resolved: {alert.message}")
                    await self._notify_alert_resolved(alert)

                # Clean up old resolved alerts
                await self._cleanup_old_alerts()

                await asyncio.sleep(30)  # Process every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)

    async def _metrics_cleanup(self):
        """Clean up old metrics data."""
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=24)

                # Clean up in-memory metrics
                for metric_name, points in self.metrics_buffer.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff_time:
                        points.popleft()

                # Clean up Redis metrics
                cutoff_timestamp = cutoff_time.timestamp()
                metric_keys = await self.redis.keys("metrics:*")

                for key in metric_keys:
                    await self.redis.zremrangebyscore(key, 0, cutoff_timestamp)

                logger.debug("Metrics cleanup completed")
                await asyncio.sleep(3600)  # Clean up every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(3600)

    async def _record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Record a metric point."""
        try:
            if tags is None:
                tags = {}

            metric_point = MetricPoint(timestamp=timestamp, value=value, tags=tags)

            # Store in memory buffer
            self.metrics_buffer[metric_name].append(metric_point)

            # Store in Redis for persistence
            redis_key = f"metrics:{metric_name}"
            await self.redis.zadd(
                redis_key,
                {
                    json.dumps(
                        {
                            "value": value,
                            "tags": tags,
                            "timestamp": timestamp.isoformat(),
                        }
                    ): timestamp.timestamp()
                },
            )

        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")

    async def _check_resource_thresholds(self, timestamp: datetime):
        """Check if any resource thresholds are exceeded."""
        for metric_name, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue

            # Get recent metric values
            recent_values = []
            cutoff_time = timestamp - timedelta(seconds=threshold.duration)

            if metric_name in self.metrics_buffer:
                for point in self.metrics_buffer[metric_name]:
                    if point.timestamp >= cutoff_time:
                        recent_values.append(point.value)

            if not recent_values:
                continue

            # Check if threshold exceeded for duration
            avg_value = sum(recent_values) / len(recent_values)

            if avg_value >= threshold.critical:
                await self._create_alert(
                    f"threshold_{metric_name}_critical",
                    AlertSeverity.CRITICAL,
                    f"{metric_name} critical threshold exceeded: {avg_value:.2f} >= {threshold.critical}",
                    source="monitor",
                    metadata={
                        "metric": metric_name,
                        "value": avg_value,
                        "threshold": threshold.critical,
                        "duration": threshold.duration,
                    },
                )
            elif avg_value >= threshold.warning:
                await self._create_alert(
                    f"threshold_{metric_name}_warning",
                    AlertSeverity.MEDIUM,
                    f"{metric_name} warning threshold exceeded: {avg_value:.2f} >= {threshold.warning}",
                    source="monitor",
                    metadata={
                        "metric": metric_name,
                        "value": avg_value,
                        "threshold": threshold.warning,
                        "duration": threshold.duration,
                    },
                )

    async def _create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a new system alert."""
        if metadata is None:
            metadata = {}

        # Check if alert already exists and is not resolved
        if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
            return  # Don't duplicate active alerts

        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            source=source,
            message=message,
            metadata=metadata,
        )

        self.active_alerts[alert_id] = alert

        # Store alert in Redis
        self.redis.lpush(
            f"alerts:{severity.value}",
            json.dumps(
                {
                    "id": alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": severity.value,
                    "source": source,
                    "message": message,
                    "metadata": metadata,
                }
            ),
        )

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Alert created [{severity.value}]: {message}")

    async def _should_resolve_alert(self, alert: Alert) -> bool:
        """Check if an alert should be resolved."""
        try:
            # Basic resolution logic based on alert type
            if alert.id.startswith("threshold_"):
                metric_name = (
                    alert.id.replace("threshold_", "")
                    .replace("_critical", "")
                    .replace("_warning", "")
                )

                if (
                    metric_name in self.metrics_buffer
                    and self.metrics_buffer[metric_name]
                ):
                    # Check if recent values are below threshold
                    recent_point = self.metrics_buffer[metric_name][-1]
                    threshold = self.thresholds.get(metric_name)

                    if threshold:
                        # Resolve if value is below warning threshold
                        return recent_point.value < threshold.warning

            elif alert.id.startswith("api_rate_limit_"):
                api_name = alert.id.replace("api_rate_limit_", "")
                if api_name in self.api_rate_limits:
                    usage = self.api_rate_limits[api_name].get("usage_percent", 0)
                    return usage < 70  # Resolve when usage drops below 70%

            elif alert.id == "redis_connection":
                # Check if Redis is responsive
                try:
                    await self.redis.ping()
                    return True
                except Exception:
                    return False

        except Exception as e:
            logger.error(f"Error checking alert resolution for {alert.id}: {e}")

        return False

    async def _notify_alert_resolved(self, alert: Alert):
        """Notify that an alert has been resolved."""
        logger.info(f"Alert resolved: {alert.message}")

        # Ensure alert is actually resolved
        assert alert.resolved_at is not None, "Alert must have resolved_at timestamp"

        # Store resolution in Redis
        self.redis.lpush(
            "alerts:resolved",
            json.dumps(
                {
                    "id": alert.id,
                    "original_timestamp": alert.timestamp.isoformat(),
                    "resolved_timestamp": alert.resolved_at.isoformat(),
                    "duration": (alert.resolved_at - alert.timestamp).total_seconds(),
                    "message": alert.message,
                }
            ),
        )

    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(hours=24)

        alerts_to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time:
                alerts_to_remove.append(alert_id)

        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]

    def register_alert_callback(self, callback: Callable):
        """Register callback for new alerts."""
        self.alert_callbacks.append(callback)

    async def get_metrics(
        self, metric_name: str, duration: Optional[timedelta] = None
    ) -> List[MetricPoint]:
        """Get metric data points for a specific duration."""
        if duration is None:
            duration = timedelta(hours=1)

        cutoff_time = datetime.now() - duration

        if metric_name in self.metrics_buffer:
            return [
                point
                for point in self.metrics_buffer[metric_name]
                if point.timestamp >= cutoff_time
            ]

        return []

    async def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        summary = {}
        current_time = datetime.now()

        for metric_name, points in self.metrics_buffer.items():
            if not points:
                continue

            recent_points = [
                p for p in points if (current_time - p.timestamp).total_seconds() < 300
            ]  # Last 5 minutes

            if recent_points:
                values = [p.value for p in recent_points]
                summary[metric_name] = {
                    "current": recent_points[-1].value,
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "last_update": recent_points[-1].timestamp.isoformat(),
                }

        return summary

    async def get_alerts(
        self, severity: Optional[AlertSeverity] = None, resolved: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        alerts = []

        for alert in self.active_alerts.values():
            # Apply filters
            if severity and alert.severity != severity:
                continue
            if resolved is not None and alert.resolved != resolved:
                continue

            alerts.append(
                {
                    "id": alert.id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "message": alert.message,
                    "resolved": alert.resolved,
                    "resolved_at": (
                        alert.resolved_at.isoformat() if alert.resolved_at else None
                    ),
                    "metadata": alert.metadata,
                }
            )

        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        return alerts

    async def clear_alerts(self, severity: Optional[AlertSeverity] = None):
        """Clear alerts with optional severity filtering."""
        alerts_to_remove = []

        for alert_id, alert in self.active_alerts.items():
            if severity is None or alert.severity == severity:
                alerts_to_remove.append(alert_id)

        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]

        # Clear from Redis as well
        if severity:
            await self.redis.delete(f"alerts:{severity.value}")
        else:
            for sev in AlertSeverity:
                await self.redis.delete(f"alerts:{sev.value}")

        logger.info(f"Cleared {len(alerts_to_remove)} alerts")

    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_time = datetime.now()

        # System resources
        system_summary = await self._get_system_resource_summary()

        # Service performance
        service_summary = await self._get_service_performance_summary()

        # Alert summary
        alert_summary = await self._get_alert_summary()

        # API rate limits
        api_summary = dict(self.api_rate_limits)

        return {
            "timestamp": current_time.isoformat(),
            "system_resources": system_summary,
            "services": service_summary,
            "alerts": alert_summary,
            "api_rate_limits": api_summary,
            "monitoring_uptime": (
                current_time - self.last_performance_check
            ).total_seconds(),
        }

    async def _get_system_resource_summary(self) -> Dict[str, Any]:
        """Get system resource usage summary."""
        resource_metrics = ["cpu_percent", "memory_percent", "disk_percent"]
        summary = {}

        for metric in resource_metrics:
            if metric in self.metrics_buffer and self.metrics_buffer[metric]:
                recent_points = [
                    p
                    for p in self.metrics_buffer[metric]
                    if (datetime.now() - p.timestamp).total_seconds() < 300
                ]
                if recent_points:
                    values = [p.value for p in recent_points]
                    summary[metric] = {
                        "current": recent_points[-1].value,
                        "average": sum(values) / len(values),
                        "max": max(values),
                        "min": min(values),
                    }

        return summary

    async def _get_service_performance_summary(self) -> Dict[str, Any]:
        """Get service performance summary."""
        summary = {}

        for service_name, metrics in self.service_metrics.items():
            if "response_time" in metrics:
                summary[service_name] = {
                    "response_time": metrics["response_time"],
                    "requests_total": metrics.get("requests_total", 0),
                    "error_rate": metrics.get("error_rate", 0),
                    "last_updated": metrics.get(
                        "last_updated", datetime.now().isoformat()
                    ),
                }

        return summary

    async def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        total_alerts = len(self.active_alerts)
        resolved_alerts = sum(
            1 for alert in self.active_alerts.values() if alert.resolved
        )
        active_alerts = total_alerts - resolved_alerts

        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = sum(
                1
                for alert in self.active_alerts.values()
                if alert.severity == severity and not alert.resolved
            )

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_alerts,
            "by_severity": severity_counts,
        }

    async def get_health_check_summary(self) -> Dict[str, Any]:
        """Get health check summary for all monitored endpoints."""
        return {
            "total_endpoints": len(self.service_metrics),
            "healthy_endpoints": sum(
                1
                for metrics in self.service_metrics.values()
                if metrics.get("healthy", False)
            ),
            "response_times": {
                service: metrics.get("response_time", 0)
                for service, metrics in self.service_metrics.items()
            },
        }

    def set_threshold(
        self, metric_name: str, warning: float, critical: float, duration: int = 60
    ):
        """Set or update threshold for a metric."""
        self.thresholds[metric_name] = Threshold(
            warning=warning, critical=critical, duration=duration
        )
        logger.info(
            f"Updated threshold for {metric_name}: warning={warning}, critical={critical}"
        )

    def disable_threshold(self, metric_name: str):
        """Disable threshold checking for a metric."""
        if metric_name in self.thresholds:
            self.thresholds[metric_name].enabled = False
            logger.info(f"Disabled threshold for {metric_name}")

    def enable_threshold(self, metric_name: str):
        """Enable threshold checking for a metric."""
        if metric_name in self.thresholds:
            self.thresholds[metric_name].enabled = True
            logger.info(f"Enabled threshold for {metric_name}")

    async def export_metrics(
        self, metric_names: List[str], duration: timedelta
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Export metrics data for analysis."""
        exported_data = {}

        for metric_name in metric_names:
            metric_points = await self.get_metrics(metric_name, duration)
            exported_data[metric_name] = [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "tags": point.tags,
                }
                for point in metric_points
            ]

        return exported_data

    async def get_metric_statistics(
        self, metric_name: str, duration: timedelta
    ) -> Dict[str, float]:
        """Get statistical summary of a metric over a duration."""
        metric_points = await self.get_metrics(metric_name, duration)

        if not metric_points:
            return {}

        values = [point.value for point in metric_points]

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "std_dev": (
                sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)
            )
            ** 0.5,
        }

    async def force_metric_collection(self):
        """Force immediate collection of all metrics."""
        logger.info("Forcing immediate metric collection...")

        timestamp = datetime.now()

        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        await self._record_metric("cpu_percent", cpu_percent, timestamp)
        await self._record_metric("memory_percent", memory.percent, timestamp)
        await self._record_metric(
            "disk_percent", (disk.used / disk.total) * 100, timestamp
        )

        # Redis metrics
        try:
            redis_info = await self.redis.info()
            await self._record_metric(
                "redis_connected_clients",
                redis_info.get("connected_clients", 0),
                timestamp,
            )
            await self._record_metric(
                "redis_used_memory_mb",
                redis_info.get("used_memory", 0) / (1024 * 1024),
                timestamp,
            )
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")

        logger.info("Forced metric collection completed")

    async def shutdown(self):
        """Graceful shutdown of the monitor."""
        logger.info("Shutting down system monitor...")
        await self.stop_monitoring()
        logger.info("System monitor shutdown completed")
