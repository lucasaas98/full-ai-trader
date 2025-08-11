#!/usr/bin/env python3
"""
Production Maintenance Runner for the Trading Scheduler.

This script provides a production-ready maintenance runner with:
- Comprehensive error handling and recovery
- Real-time monitoring and alerting
- Graceful degradation on failures
- Detailed logging and metrics collection
- Integration with external monitoring systems
- Automatic retry mechanisms
- Health status reporting
"""

import asyncio
import json
import logging
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import argparse
from dataclasses import dataclass, asdict

import redis.asyncio as redis
import httpx
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/maintenance_runner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MaintenanceRunnerConfig:
    """Configuration for the maintenance runner."""

    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_timeout: int = 30

    # Scheduler service configuration
    scheduler_url: str = "http://localhost:8000"
    scheduler_timeout: int = 60

    # Maintenance configuration
    max_concurrent_tasks: int = 3
    task_timeout_minutes: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 60

    # Monitoring configuration
    health_check_interval: int = 300  # 5 minutes
    metrics_collection_interval: int = 60  # 1 minute
    alert_cooldown_minutes: int = 15

    # Emergency thresholds
    cpu_threshold: float = 90.0
    memory_threshold: float = 90.0
    disk_threshold: float = 95.0

    # Notification settings
    enable_alerts: bool = True
    alert_webhook_url: Optional[str] = None
    slack_webhook_url: Optional[str] = None

    # Task priorities
    critical_tasks: Optional[Set[str]] = None

    def __post_init__(self):
        if self.critical_tasks is None:
            self.critical_tasks = {
                'system_health_check',
                'portfolio_reconciliation',
                'database_maintenance',
                'backup_critical_data'
            }


@dataclass
class MaintenanceRunStatus:
    """Status of a maintenance run."""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_bytes_freed: int = 0
    total_duration: float = 0.0
    error_message: Optional[str] = None
    system_health_score: float = 100.0


class AlertManager:
    """Manages alerting and notifications for maintenance operations."""

    def __init__(self, config: MaintenanceRunnerConfig):
        self.config = config
        self.last_alert_times = {}
        self.alert_counts = {}

    async def send_alert(self, alert_type: str, message: str, severity: str = "warning", details: Optional[Dict[str, Any]] = None):
        """Send alert with cooldown and deduplication."""
        try:
            if not self.config.enable_alerts:
                return

            # Check cooldown
            now = datetime.now()
            alert_key = f"{alert_type}:{severity}"

            if alert_key in self.last_alert_times:
                time_since_last = (now - self.last_alert_times[alert_key]).total_seconds()
                if time_since_last < self.config.alert_cooldown_minutes * 60:
                    logger.debug(f"Alert suppressed due to cooldown: {alert_key}")
                    return

            self.last_alert_times[alert_key] = now
            self.alert_counts[alert_key] = self.alert_counts.get(alert_key, 0) + 1

            alert_data = {
                "type": alert_type,
                "severity": severity,
                "message": message,
                "timestamp": now.isoformat(),
                "count": self.alert_counts[alert_key],
                "details": details or {}
            }

            # Send to configured endpoints
            await self._send_to_webhook(alert_data)
            await self._send_to_slack(alert_data)

            logger.warning(f"Alert sent: {alert_type} - {message}")

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def _send_to_webhook(self, alert_data: Dict[str, Any]):
        """Send alert to webhook endpoint."""
        if not self.config.alert_webhook_url:
            return

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.alert_webhook_url,
                    json=alert_data,
                    timeout=10.0
                )
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")

    async def _send_to_slack(self, alert_data: Dict[str, Any]):
        """Send alert to Slack webhook."""
        if not self.config.slack_webhook_url:
            return

        try:
            severity_emoji = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "🔴",
                "critical": "🚨"
            }.get(alert_data["severity"], "📢")

            slack_message = {
                "text": f"{severity_emoji} Maintenance Alert",
                "attachments": [{
                    "color": {
                        "info": "good",
                        "warning": "warning",
                        "error": "danger",
                        "critical": "danger"
                    }.get(alert_data["severity"], "warning"),
                    "fields": [
                        {"title": "Type", "value": alert_data["type"], "short": True},
                        {"title": "Severity", "value": alert_data["severity"], "short": True},
                        {"title": "Message", "value": alert_data["message"], "short": False},
                        {"title": "Timestamp", "value": alert_data["timestamp"], "short": True}
                    ]
                }]
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.slack_webhook_url,
                    json=slack_message,
                    timeout=10.0
                )
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")


class MaintenanceOrchestrator:
    """Orchestrates maintenance operations with monitoring and recovery."""

    def __init__(self, config: MaintenanceRunnerConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.alert_manager = AlertManager(config)
        self.running_tasks = {}
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.current_run_status = None

        # Performance tracking
        self.task_history = []
        self.system_metrics_history = []

        # Emergency state
        self.emergency_mode = False
        self.emergency_tasks_only = False

    async def initialize(self):
        """Initialize the maintenance orchestrator."""
        try:
            logger.info("Initializing maintenance orchestrator...")

            # Initialize Redis connection
            self.redis_client = redis.from_url(
                self.config.redis_url,
                socket_timeout=self.config.redis_timeout,
                retry_on_timeout=True
            )

            # Test Redis connection
            await self.redis_client.ping()
            logger.info("Redis connection established")

            # Register shutdown handlers
            self._register_signal_handlers()

            logger.info("Maintenance orchestrator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize maintenance orchestrator: {e}")
            await self.alert_manager.send_alert(
                "orchestrator_init_failed",
                f"Maintenance orchestrator initialization failed: {str(e)}",
                "critical"
            )
            return False

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def run_maintenance_cycle(self, cycle_type: str = "auto") -> MaintenanceRunStatus:
        """Run a complete maintenance cycle."""
        run_id = f"maintenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_run_status = MaintenanceRunStatus(
            run_id=run_id,
            start_time=datetime.now()
        )

        logger.info(f"Starting maintenance cycle: {run_id} (type: {cycle_type})")

        try:
            # Pre-maintenance health check
            pre_health = await self._check_system_health()
            self.current_run_status.system_health_score = pre_health['health_score']

            if pre_health['health_score'] < 50 and cycle_type != "emergency":
                logger.warning("Low system health detected, switching to emergency mode")
                self.emergency_mode = True
                cycle_type = "emergency"

            # Determine tasks to run
            tasks_to_run = await self._determine_maintenance_tasks(cycle_type)

            # Execute maintenance tasks
            results = await self._execute_maintenance_tasks(tasks_to_run)

            # Process results
            await self._process_maintenance_results(results)

            # Post-maintenance health check
            post_health = await self._check_system_health()

            # Update run status
            self.current_run_status.end_time = datetime.now()
            self.current_run_status.status = "completed"
            self.current_run_status.total_duration = (
                self.current_run_status.end_time - self.current_run_status.start_time
            ).total_seconds()

            # Generate and store report
            await self._generate_maintenance_report(results, pre_health, post_health)

            logger.info(f"Maintenance cycle {run_id} completed successfully")

            return self.current_run_status

        except Exception as e:
            logger.error(f"Maintenance cycle {run_id} failed: {e}")

            self.current_run_status.end_time = datetime.now()
            self.current_run_status.status = "failed"
            self.current_run_status.error_message = str(e)

            await self.alert_manager.send_alert(
                "maintenance_cycle_failed",
                f"Maintenance cycle {run_id} failed: {str(e)}",
                "error",
                {"run_id": run_id, "error": str(e), "traceback": traceback.format_exc()}
            )

            return self.current_run_status

    async def _check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        try:
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
                "health_score": 100.0,
                "issues": []
            }

            # Calculate health score
            if health_data["cpu_percent"] > self.config.cpu_threshold:
                health_data["health_score"] -= 30
                health_data["issues"].append(f"High CPU usage: {health_data['cpu_percent']:.1f}%")

            if health_data["memory_percent"] > self.config.memory_threshold:
                health_data["health_score"] -= 25
                health_data["issues"].append(f"High memory usage: {health_data['memory_percent']:.1f}%")

            if health_data["disk_percent"] > self.config.disk_threshold:
                health_data["health_score"] -= 20
                health_data["issues"].append(f"High disk usage: {health_data['disk_percent']:.1f}%")

            # Check Redis connectivity
            try:
                if self.redis_client:
                    await self.redis_client.ping()
                    redis_info = await self.redis_client.info()
                    health_data["redis_memory_mb"] = redis_info.get('used_memory', 0) / (1024 * 1024)
                    health_data["redis_connections"] = redis_info.get('connected_clients', 0)
                else:
                    health_data["health_score"] -= 15
                    health_data["issues"].append("Redis client not initialized")
            except Exception as e:
                health_data["health_score"] -= 15
                health_data["issues"].append(f"Redis connectivity issue: {str(e)}")

            # Check scheduler service
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.config.scheduler_url}/health", timeout=10.0)
                    if response.status_code != 200:
                        health_data["health_score"] -= 10
                        health_data["issues"].append("Scheduler service unhealthy")
            except Exception as e:
                health_data["health_score"] -= 15
                health_data["issues"].append(f"Scheduler service unreachable: {str(e)}")

            health_data["health_score"] = max(0, health_data["health_score"])

            # Store health metrics
            await self._store_health_metrics(health_data)

            return health_data

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "health_score": 0.0,
                "issues": [f"Health check failed: {str(e)}"],
                "error": str(e)
            }

    async def _determine_maintenance_tasks(self, cycle_type: str) -> List[Dict[str, Any]]:
        """Determine which maintenance tasks to run based on cycle type and system state."""
        try:
            if cycle_type == "emergency":
                return [
                    {"name": "system_health_check", "priority": 1, "timeout": 5},
                    {"name": "cache_cleanup", "priority": 1, "timeout": 10},
                    {"name": "portfolio_reconciliation", "priority": 1, "timeout": 15},
                    {"name": "resource_optimization", "priority": 2, "timeout": 20}
                ]

            elif cycle_type == "daily":
                return [
                    {"name": "system_health_check", "priority": 1, "timeout": 10},
                    {"name": "data_cleanup", "priority": 2, "timeout": 30},
                    {"name": "log_rotation", "priority": 2, "timeout": 15},
                    {"name": "cache_cleanup", "priority": 2, "timeout": 15},
                    {"name": "tradenote_export", "priority": 3, "timeout": 20},
                    {"name": "api_rate_limit_reset", "priority": 3, "timeout": 5},
                    {"name": "intelligent_maintenance", "priority": 3, "timeout": 25}
                ]

            elif cycle_type == "weekly":
                return [
                    {"name": "system_health_check", "priority": 1, "timeout": 10},
                    {"name": "database_maintenance", "priority": 1, "timeout": 60},
                    {"name": "backup_critical_data", "priority": 1, "timeout": 45},
                    {"name": "performance_optimization", "priority": 2, "timeout": 40},
                    {"name": "security_audit", "priority": 2, "timeout": 30},
                    {"name": "trading_data_maintenance", "priority": 2, "timeout": 50},
                    {"name": "historical_data_update", "priority": 3, "timeout": 60},
                    {"name": "database_connection_pool", "priority": 3, "timeout": 20},
                    {"name": "resource_optimization", "priority": 3, "timeout": 30}
                ]

            elif cycle_type == "auto":
                # Determine based on system state and time
                current_hour = datetime.now().hour
                day_of_week = datetime.now().weekday()

                if day_of_week == 6:  # Sunday
                    return await self._determine_maintenance_tasks("weekly")
                elif current_hour < 6:  # Early morning
                    return await self._determine_maintenance_tasks("daily")
                else:
                    return await self._determine_maintenance_tasks("emergency")

            else:
                # Custom task list
                return [{"name": cycle_type, "priority": 1, "timeout": 30}]

        except Exception as e:
            logger.error(f"Failed to determine maintenance tasks: {e}")
            # Fallback to essential tasks
            return [
                {"name": "system_health_check", "priority": 1, "timeout": 10},
                {"name": "cache_cleanup", "priority": 1, "timeout": 15}
            ]

    async def _execute_maintenance_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute maintenance tasks with proper orchestration and error handling."""
        results = {}
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        # Sort tasks by priority
        tasks.sort(key=lambda x: x.get('priority', 999))

        async def execute_single_task(task_config: Dict[str, Any]):
            async with semaphore:
                task_name = task_config['name']
                timeout = task_config.get('timeout', self.config.task_timeout_minutes) * 60

                logger.info(f"Executing maintenance task: {task_name}")
                start_time = time.time()

                for attempt in range(1, self.config.retry_attempts + 1):
                    try:
                        # Record task start
                        self.running_tasks[task_name] = {
                            "start_time": datetime.now(),
                            "attempt": attempt,
                            "timeout": timeout
                        }

                        # Execute via scheduler API
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                f"{self.config.scheduler_url}/maintenance/tasks/{task_name}/run",
                                json={},
                                timeout=timeout
                            )
                            response.raise_for_status()
                            result_data = response.json()

                        # Process result
                        duration = time.time() - start_time
                        task_result = {
                            'success': result_data.get('success', False),
                            'duration': duration,
                            'message': result_data.get('message', ''),
                            'bytes_freed': result_data.get('bytes_freed', 0),
                            'files_processed': result_data.get('files_processed', 0),
                            'attempts': attempt,
                            'details': result_data.get('details', {})
                        }

                        # Update run status
                        if self.current_run_status:
                            if task_result['success']:
                                self.current_run_status.tasks_completed += 1
                                self.current_run_status.total_bytes_freed += task_result['bytes_freed']
                            else:
                                self.current_run_status.tasks_failed += 1

                            self.current_run_status.total_duration += duration

                        # Remove from running tasks
                        self.running_tasks.pop(task_name, None)

                        # Log result
                        if task_result['success']:
                            logger.info(f"✅ Task {task_name} completed in {duration:.2f}s")
                        else:
                            logger.error(f"❌ Task {task_name} failed: {task_result['message']}")

                            # Send alert for critical task failures
                            if task_name in self.config.critical_tasks:
                                await self.alert_manager.send_alert(
                                    "critical_maintenance_failure",
                                    f"Critical maintenance task {task_name} failed: {task_result['message']}",
                                    "error",
                                    task_result
                                )

                        return task_result

                    except asyncio.TimeoutError:
                        logger.error(f"Task {task_name} timed out (attempt {attempt})")
                        if attempt < self.config.retry_attempts:
                            await asyncio.sleep(self.config.retry_delay_seconds)
                            continue
                        else:
                            return {
                                'success': False,
                                'duration': time.time() - start_time,
                                'message': f'Task timed out after {self.config.retry_attempts} attempts',
                                'attempts': attempt
                            }

                    except Exception as e:
                        logger.error(f"Task {task_name} failed (attempt {attempt}): {e}")
                        if attempt < self.config.retry_attempts:
                            await asyncio.sleep(self.config.retry_delay_seconds)
                            continue
                        else:
                            return {
                                'success': False,
                                'duration': time.time() - start_time,
                                'message': f'Task failed after {self.config.retry_attempts} attempts: {str(e)}',
                                'attempts': attempt,
                                'error': str(e)
                            }

                # Remove from running tasks if we get here
                self.running_tasks.pop(task_name, None)

        # Execute all tasks
        task_coroutines = [execute_single_task(task) for task in tasks]
        task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Compile results
        for i, result in enumerate(task_results):
            task_name = tasks[i]['name']
            if isinstance(result, Exception):
                results[task_name] = {
                    'success': False,
                    'duration': 0.0,
                    'message': f'Task execution failed: {str(result)}',
                    'error': str(result)
                }
            else:
                results[task_name] = result

        return results

    async def _process_maintenance_results(self, results: Dict[str, Any]):
        """Process and analyze maintenance results."""
        try:
            # Store results in Redis
            await self._store_maintenance_results(results)

            # Analyze results for patterns
            await self._analyze_maintenance_patterns(results)

            # Check for concerning trends
            await self._check_maintenance_trends(results)

            # Update system metrics
            await self._update_system_metrics(results)

        except Exception as e:
            logger.error(f"Failed to process maintenance results: {e}")

    async def _store_maintenance_results(self, results: Dict[str, Any]):
        """Store maintenance results in Redis for analysis."""
        try:
            timestamp = datetime.now().isoformat()

            # Store individual results
            for task_name, result in results.items():
                result_data = {
                    **result,
                    'timestamp': timestamp,
                    "run_id": self.current_run_status.run_id if self.current_run_status else "unknown",
                }

                if self.redis_client:
                    await self.redis_client.setex(
                    f"maintenance:latest:{task_name}",
                    86400,  # 24 hours
                    json.dumps(result_data, default=str)
                )

            # Store run summary
            if self.current_run_status is not None:
                run_summary = asdict(self.current_run_status)
                if self.redis_client is not None:
                    await self.redis_client.setex(
                        f"maintenance:run:{self.current_run_status.run_id}",
                        604800,  # 1 week
                        json.dumps(run_summary, default=str)
                    )

        except Exception as e:
            logger.error(f"Failed to store maintenance results: {e}")

    async def _analyze_maintenance_patterns(self, results: Dict[str, Any]):
        """Analyze maintenance results for patterns and anomalies."""
        try:
            # Track task performance over time
            for task_name, result in results.items():
                if result['success']:
                    # Store performance metrics
                    if self.redis_client:
                        await self.redis_client.zadd(
                        f"maintenance:performance:{task_name}",
                        {str(int(time.time())): result['duration']}
                    )

                    # Clean old performance data (keep 30 days)
                    cutoff = time.time() - (30 * 24 * 3600)
                    if self.redis_client:
                        if self.redis_client:
                            await self.redis_client.zremrangebyscore(
                        f"maintenance:performance:{task_name}",
                        0, cutoff
                    )

            # Detect performance degradation
            await self._detect_performance_degradation(results)

        except Exception as e:
            logger.error(f"Failed to analyze maintenance patterns: {e}")

    async def _detect_performance_degradation(self, current_results: Dict[str, Any]):
        """Detect if maintenance tasks are showing performance degradation."""
        try:
            for task_name, result in current_results.items():
                if not result['success']:
                    continue

                # Get historical performance
                if self.redis_client is not None:
                    historical_runs = await self.redis_client.zrange(
                        f"maintenance:performance:{task_name}",
                        -10, -1, withscores=True
                    )
                else:
                    historical_runs = []

                if len(historical_runs) >= 5:
                    # Calculate average historical duration
                    avg_historical = sum(score for _, score in historical_runs) / len(historical_runs)
                    current_duration = result['duration']

                    # Check for significant degradation (>50% slower)
                    if current_duration > avg_historical * 1.5:
                        await self.alert_manager.send_alert(
                            "maintenance_performance_degradation",
                            f"Task {task_name} performance degraded: {current_duration:.2f}s vs {avg_historical:.2f}s avg",
                            "warning",
                            {
                                "task_name": task_name,
                                "current_duration": current_duration,
                                "historical_average": avg_historical,
                                "degradation_percent": ((current_duration / avg_historical - 1) * 100)
                            }
                        )

        except Exception as e:
            logger.error(f"Performance degradation detection failed: {e}")

    async def _check_maintenance_trends(self, results: Dict[str, Any]):
        """Check for concerning maintenance trends."""
        try:
            # Check failure rate
            failed_tasks = [name for name, result in results.items() if not result['success']]
            if len(failed_tasks) > len(results) * 0.3:  # More than 30% failed
                await self.alert_manager.send_alert(
                    "high_maintenance_failure_rate",
                    f"High maintenance failure rate: {len(failed_tasks)}/{len(results)} tasks failed",
                    "error",
                    {"failed_tasks": failed_tasks}
                )

            # Check total execution time
            total_duration = sum(result['duration'] for result in results.values())
            if total_duration > 3600:  # More than 1 hour
                await self.alert_manager.send_alert(
                    "maintenance_cycle_slow",
                    f"Maintenance cycle took {total_duration:.0f} seconds",
                    "warning",
                    {"total_duration": total_duration}
                )

        except Exception as e:
            logger.error(f"Maintenance trend checking failed: {e}")

    async def _store_health_metrics(self, health_data: Dict[str, Any]):
        """Store system health metrics for trending."""
        try:
            timestamp = int(time.time())

            # Store in time series
            if self.redis_client:
                await self.redis_client.zadd(
                "maintenance:health_metrics",
                {json.dumps(health_data, default=str): timestamp}
            )

            # Keep only last 7 days
            cutoff = timestamp - (7 * 24 * 3600)
            if self.redis_client is not None:
                await self.redis_client.zremrangebyscore("maintenance:health_metrics", 0, cutoff)

        except Exception as e:
            logger.error(f"Failed to store health metrics: {e}")

    async def _update_system_metrics(self, results: Dict[str, Any]):
        """Update system-wide metrics based on maintenance results."""
        try:
            metrics = {
                "last_maintenance_run": datetime.now().isoformat(),
                "tasks_completed": self.current_run_status.tasks_completed if self.current_run_status else 0,
                "tasks_failed": self.current_run_status.tasks_failed if self.current_run_status else 0,
                "total_bytes_freed": self.current_run_status.total_bytes_freed if self.current_run_status else 0,
                "total_duration": self.current_run_status.total_duration if self.current_run_status else 0,
                "system_health_score": self.current_run_status.system_health_score if self.current_run_status else 0,
            }

            if self.redis_client:
                await self.redis_client.setex(
                "maintenance:system_metrics",
                86400,  # 24 hours
                json.dumps(metrics, default=str)
            )

        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    async def _generate_maintenance_report(self, results: Dict[str, Any], pre_health: Dict[str, Any], post_health: Dict[str, Any]):
        """Generate comprehensive maintenance report."""
        try:
            report = {
                "run_id": self.current_run_status.run_id if self.current_run_status else "unknown",
                "timestamp": datetime.now().isoformat(),
                "cycle_type": "maintenance_cycle",
                "duration": self.current_run_status.total_duration if self.current_run_status else 0,
                "summary": {
                    "completed": self.current_run_status.tasks_completed if self.current_run_status else 0,
                    "failed": self.current_run_status.tasks_failed if self.current_run_status else 0,
                    "total_freed_mb": (self.current_run_status.total_bytes_freed / (1024 * 1024)) if self.current_run_status else 0,
                    "success_rate": (self.current_run_status.tasks_completed /
                                   (self.current_run_status.tasks_completed + self.current_run_status.tasks_failed) * 100
                                   if self.current_run_status and (self.current_run_status.tasks_completed + self.current_run_status.tasks_failed) > 0 else 0)
                },
                "health_comparison": {
                    "pre_maintenance": pre_health,
                    "post_maintenance": post_health,
                    "health_improvement": post_health.get('health_score', 0) - pre_health.get('health_score', 0)
                },
                "task_results": results,
                "recommendations": []
            }

            # Generate recommendations
            if post_health.get('health_score', 0) < pre_health.get('health_score', 0):
                report['recommendations'].append("System health declined during maintenance - investigate causes")

            failed_tasks = [name for name, result in results.items() if not result['success']]
            if failed_tasks:
                report['recommendations'].append(f"Failed tasks require attention: {', '.join(failed_tasks)}")

            if self.current_run_status and self.current_run_status.total_bytes_freed < 10 * 1024 * 1024:  # Less than 10MB
                report['recommendations'].append("Low disk space recovery - consider more aggressive cleanup")

            # Store report
            await self._store_maintenance_report(report)

            if self.current_run_status:
                logger.info(f"Maintenance report generated for run {self.current_run_status.run_id}")
            else:
                logger.info("Maintenance report generated")

        except Exception as e:
            logger.error(f"Failed to generate maintenance report: {e}")

    async def _store_maintenance_report(self, report: Dict[str, Any]):
        """Store maintenance report in Redis and file system."""
        try:
            # Store in Redis
            if self.redis_client:
                await self.redis_client.setex(
                f"maintenance:report:{report['run_id']}",
                604800,  # 1 week
                json.dumps(report, default=str)
            )

            # Store in file system
            reports_dir = Path("data/reports/maintenance")
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = reports_dir / f"maintenance_report_{timestamp}.json"

            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to store maintenance report: {e}")

    async def run_continuous_monitoring(self):
        """Run continuous monitoring loop."""
        logger.info("Starting continuous maintenance monitoring...")
        self.is_running = True

        try:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # System health check
                    health_data = await self._check_system_health()

                    # Check for emergency conditions
                    if health_data['health_score'] < 30:
                        logger.warning("Critical system health detected, running emergency maintenance")
                        await self.run_maintenance_cycle("emergency")

                    # Regular monitoring tasks
                    await self._monitor_running_tasks()
                    await self._check_resource_alerts()
                    await self._collect_metrics()

                    # Wait for next cycle
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(),
                            timeout=self.config.health_check_interval
                        )
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        continue  # Normal timeout, continue monitoring

                except Exception as e:
                    logger.error(f"Monitoring cycle error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying

        except Exception as e:
            logger.error(f"Continuous monitoring failed: {e}")
        finally:
            self.is_running = False
            logger.info("Continuous monitoring stopped")

    async def _monitor_running_tasks(self):
        """Monitor currently running tasks for timeouts and issues."""
        try:
            current_time = datetime.now()

            for task_name, task_info in list(self.running_tasks.items()):
                start_time = task_info['start_time']
                timeout = task_info['timeout']

                # Check for timeout
                if (current_time - start_time).total_seconds() > timeout:
                    logger.warning(f"Task {task_name} exceeded timeout, may need intervention")

                    await self.alert_manager.send_alert(
                        "maintenance_task_timeout",
                        f"Maintenance task {task_name} running for {(current_time - start_time).total_seconds():.0f}s",
                        "warning",
                        {
                            "task_name": task_name,
                            "start_time": start_time.isoformat(),
                            "timeout": timeout,
                            "current_duration": (current_time - start_time).total_seconds()
                        }
                    )

        except Exception as e:
            logger.error(f"Task monitoring failed: {e}")

    async def _check_resource_alerts(self):
        """Check for resource-based alerts."""
        try:
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.cpu_threshold:
                await self.alert_manager.send_alert(
                    "high_cpu_usage",
                    f"CPU usage critical: {cpu_percent:.1f}%",
                    "critical",
                    {"cpu_percent": cpu_percent}
                )

            # Check memory
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.config.memory_threshold:
                await self.alert_manager.send_alert(
                    "high_memory_usage",
                    f"Memory usage critical: {memory_percent:.1f}%",
                    "critical",
                    {"memory_percent": memory_percent}
                )

            # Check disk space
            disk_percent = psutil.disk_usage('/').percent
            if disk_percent > self.config.disk_threshold:
                await self.alert_manager.send_alert(
                    "high_disk_usage",
                    f"Disk usage critical: {disk_percent:.1f}%",
                    "critical",
                    {"disk_percent": disk_percent}
                )

        except Exception as e:
            logger.error(f"Resource alert checking failed: {e}")

    async def _collect_metrics(self):
        """Collect and store system metrics."""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "running_tasks": len(self.running_tasks),
                "is_emergency_mode": self.emergency_mode
            }

            # Store metrics
            if self.redis_client:
                await self.redis_client.zadd(
                "maintenance:system_metrics_timeseries",
                {json.dumps(metrics, default=str): time.time()}
            )

            # Clean old metrics (keep 24 hours)
            cutoff = time.time() - (24 * 3600)
            if self.redis_client:
                await self.redis_client.zremrangebyscore(
                "maintenance:system_metrics_timeseries",
                0, cutoff
            )

        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")

    async def shutdown(self):
        """Gracefully shutdown the maintenance orchestrator."""
        logger.info("Initiating maintenance orchestrator shutdown...")

        self.is_running = False
        self.shutdown_event.set()

        # Wait for running tasks to complete (with timeout)
        if self.running_tasks:
            logger.info(f"Waiting for {len(self.running_tasks)} running tasks to complete...")

            max_wait = 300  # 5 minutes
            wait_start = time.time()

            while self.running_tasks and (time.time() - wait_start) < max_wait:
                await asyncio.sleep(5)

            if self.running_tasks:
                logger.warning(f"Forcefully terminating {len(self.running_tasks)} tasks")

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        logger.info("Maintenance orchestrator shutdown completed")

    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the maintenance orchestrator."""
        return {
            "is_running": self.is_running,
            "emergency_mode": self.emergency_mode,
            "running_tasks": len(self.running_tasks),
            "current_run": asdict(self.current_run_status) if self.current_run_status else None,
            "task_details": {
                name: {
                    "start_time": info["start_time"].isoformat(),
                    "attempt": info["attempt"],
                    "timeout": info["timeout"]
                }
                for name, info in self.running_tasks.items()
            }
        }


async def run_maintenance_cycle(
    cycle_type: str = "auto",
    config_path: Optional[str] = None,
    redis_url: Optional[str] = None,
    scheduler_url: Optional[str] = None
) -> bool:
    """Run a single maintenance cycle."""
    try:
        # Load configuration
        config = MaintenanceRunnerConfig()
        if redis_url:
            config.redis_url = redis_url
        if scheduler_url:
            config.scheduler_url = scheduler_url

        # Initialize orchestrator
        orchestrator = MaintenanceOrchestrator(config)
        if not await orchestrator.initialize():
            return False

        # Run maintenance cycle
        status = await orchestrator.run_maintenance_cycle(cycle_type)

        # Cleanup
        await orchestrator.shutdown()

        return status.status == "completed"

    except Exception as e:
        logger.error(f"Maintenance cycle execution failed: {e}")
        return False


async def run_continuous_maintenance(
    config_path: Optional[str] = None,
    redis_url: Optional[str] = None,
    scheduler_url: Optional[str] = None
):
    """Run continuous maintenance monitoring."""
    orchestrator = None
    try:
        # Load configuration
        config = MaintenanceRunnerConfig()
        if redis_url:
            config.redis_url = redis_url
        if scheduler_url:
            config.scheduler_url = scheduler_url

        # Initialize orchestrator
        orchestrator = MaintenanceOrchestrator(config)
        if not await orchestrator.initialize():
            logger.error("Failed to initialize maintenance orchestrator")
            return

        # Run continuous monitoring
        await orchestrator.run_continuous_monitoring()

    except KeyboardInterrupt:
        logger.info("Continuous maintenance interrupted by user")
    except Exception as e:
        logger.error(f"Continuous maintenance failed: {e}")
    finally:
        try:
            if orchestrator is not None:
                await orchestrator.shutdown()
        except Exception as e:
            logger.error(f"Failed to shutdown orchestrator: {e}")


async def get_maintenance_status(
    redis_url: Optional[str] = None
) -> Dict[str, Any]:
    """Get current maintenance system status."""
    try:
        config = MaintenanceRunnerConfig()
        if redis_url:
            config.redis_url = redis_url

        redis_client = redis.from_url(config.redis_url)

        try:
            # Get system metrics
            metrics_data = await redis_client.get("maintenance:system_metrics")
            metrics = json.loads(metrics_data) if metrics_data else {}

            # Get recent health data
            health_data = await redis_client.zrange(
                "maintenance:health_metrics",
                -1, -1, withscores=True
            )

            health = {}
            if health_data:
                health = json.loads(health_data[0][0])

            # Get running tasks
            running_tasks = await redis_client.keys("maintenance:running:*")

            status = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": metrics,
                "system_health": health,
                "running_tasks": len(running_tasks),
                "last_maintenance": metrics.get("last_maintenance_run"),
                "health_score": health.get("health_score", 0)
            }

            return status

        finally:
            await redis_client.close()

    except Exception as e:
        logger.error(f"Failed to get maintenance status: {e}")
        return {"error": str(e)}


def main():
    """Main entry point for the maintenance runner."""
    parser = argparse.ArgumentParser(description="Trading System Maintenance Runner")
    parser.add_argument(
        "command",
        choices=["run", "monitor", "status", "emergency"],
        help="Command to execute"
    )
    parser.add_argument(
        "--cycle-type",
        default="auto",
        choices=["auto", "daily", "weekly", "emergency"],
        help="Type of maintenance cycle to run"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis connection URL"
    )
    parser.add_argument(
        "--scheduler-url",
        default="http://localhost:8000",
        help="Scheduler service URL"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Execute command
    try:
        if args.command == "run":
            success = asyncio.run(run_maintenance_cycle(
                cycle_type=args.cycle_type,
                config_path=args.config,
                redis_url=args.redis_url,
                scheduler_url=args.scheduler_url
            ))
            sys.exit(0 if success else 1)

        elif args.command == "monitor":
            asyncio.run(run_continuous_maintenance(
                config_path=args.config,
                redis_url=args.redis_url,
                scheduler_url=args.scheduler_url
            ))

        elif args.command == "status":
            status = asyncio.run(get_maintenance_status(args.redis_url))
            print(json.dumps(status, indent=2, default=str))

        elif args.command == "emergency":
            success = asyncio.run(run_maintenance_cycle(
                cycle_type="emergency",
                config_path=args.config,
                redis_url=args.redis_url,
                scheduler_url=args.scheduler_url
            ))
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
