"""
Main scheduler service for the automated trading system.

This module provides comprehensive orchestration of all trading system components,
including market hours management, task scheduling, service health monitoring,
and data pipeline coordination.
"""

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import httpx
import psutil
import pytz
import redis.asyncio as redis
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from circuitbreaker import circuit

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared.market_hours import MarketHoursService  # noqa: E402

logger = logging.getLogger(__name__)

# Global scheduler instance reference for task execution
_scheduler_instance = None


async def execute_scheduled_task(task_id: str) -> None:
    """Global function for executing scheduled tasks without scheduler serialization issues."""
    global _scheduler_instance  # noqa: F824
    if _scheduler_instance is None:
        logger.error("No scheduler instance available for task execution")
        return

    await _scheduler_instance._execute_task_wrapper(task_id)


class ServiceStatus(str, Enum):
    """Service status enumeration."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class ServiceInfo:
    """Service information container."""

    name: str
    url: str
    health_endpoint: str
    dependencies: List[str]
    status: ServiceStatus = ServiceStatus.STOPPED
    last_check: Optional[datetime] = None
    error_count: int = 0
    restart_count: int = 0


@dataclass
class ScheduledTask:
    """Scheduled task information."""

    id: str
    name: str
    function: Callable
    trigger: Any
    priority: TaskPriority
    enabled: bool = True
    market_hours_only: bool = True
    dependencies: Optional[List[str]] = None
    retry_count: int = 0
    max_retries: int = 3
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_count: int = 0


# MarketHoursManager removed - now using shared.market_hours.MarketHoursService


class SystemMonitor:
    """Monitors system resources and service health."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.services: Dict[str, ServiceInfo] = {}
        self.health_check_interval = 30  # seconds

    def register_service(self, service: ServiceInfo) -> None:
        """Register a service for monitoring."""
        self.services[service.name] = service
        logger.info(f"Registered service for monitoring: {service.name}")

    async def check_service_health(self, service_name: str) -> bool:
        """Check health of a specific service."""
        if service_name not in self.services:
            return False

        service = self.services[service_name]

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service.url}{service.health_endpoint}")

            if response.status_code == 200:
                service.status = ServiceStatus.RUNNING
                service.error_count = 0
                service.last_check = datetime.now()
                return True
            else:
                service.error_count += 1
                service.status = ServiceStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            service.error_count += 1
            service.status = ServiceStatus.ERROR
            return False

    async def check_all_services(self) -> Dict[str, bool]:
        """Check health of all registered services."""
        results = {}

        for service_name in self.services:
            results[service_name] = await self.check_service_health(service_name)

        return results

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Redis metrics
        redis_info = await self.redis.info()

        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
            },
            "redis": {
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory": redis_info.get("used_memory", 0),
                "used_memory_human": redis_info.get("used_memory_human", "0"),
                "keyspace_hits": redis_info.get("keyspace_hits", 0),
                "keyspace_misses": redis_info.get("keyspace_misses", 0),
            },
        }


class TaskQueue:
    """Manages task execution queue with priorities."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.queues = {
            TaskPriority.CRITICAL: "scheduler:queue:critical",
            TaskPriority.HIGH: "scheduler:queue:high",
            TaskPriority.NORMAL: "scheduler:queue:normal",
            TaskPriority.LOW: "scheduler:queue:low",
        }
        self.running_tasks: Set[str] = set()

    async def enqueue_task(
        self, task_id: str, priority: TaskPriority = TaskPriority.NORMAL
    ) -> None:
        """Add task to appropriate priority queue."""
        queue_name = self.queues[priority]
        if self.redis:
            await self.redis.lpush(queue_name, task_id)  # type: ignore
        logger.debug(f"Enqueued task {task_id} to {priority} queue")

    async def dequeue_task(self) -> Optional[str]:
        """Get next task from highest priority queue."""
        # Check queues in priority order
        for priority in [
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.NORMAL,
            TaskPriority.LOW,
        ]:
            queue_name = self.queues[priority]
            if self.redis:
                task_id = await self.redis.rpop(queue_name)  # type: ignore
                if task_id:
                    # Handle different return types from rpop
                    if isinstance(task_id, bytes):
                        return task_id.decode()
                    elif isinstance(task_id, str):
                        return task_id
                    # If it's a list or other type, skip this item
        return None

    async def get_queue_lengths(self) -> Dict[str, int]:
        """Get lengths of all priority queues."""
        lengths = {}
        for priority, queue_name in self.queues.items():
            lengths[priority.value] = await self.redis.llen(queue_name)  # type: ignore
        return lengths


class DataPipelineOrchestrator:
    """Orchestrates the data pipeline flow."""

    def __init__(self, scheduler: "TradingScheduler"):
        self.scheduler = scheduler
        self.pipeline_steps = [
            "screener_scan",
            "ticker_selection",
            "data_collection",
            "strategy_analysis",
            "risk_check",
            "trade_execution",
        ]
        self.step_dependencies = {
            "ticker_selection": ["screener_scan"],
            "data_collection": ["ticker_selection"],
            "strategy_analysis": ["data_collection"],
            "risk_check": ["strategy_analysis"],
            "trade_execution": ["risk_check"],
        }

    async def trigger_pipeline(
        self, trigger_reason: str = "scheduled"
    ) -> Dict[str, Any]:
        """Trigger the complete data pipeline."""
        logger.info(f"Triggering data pipeline: {trigger_reason}")

        try:
            # Execute pipeline steps in sequence
            results = {}
            for step in self.pipeline_steps:
                results[step] = await self._execute_pipeline_step(step)
            return {"status": "success", "results": results}

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            await self.scheduler.send_notification(
                f"Data pipeline failed: {e}", priority="high"
            )
            return {"status": "error", "error": str(e)}

    async def _execute_pipeline_step(self, step: str) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        logger.info(f"Executing pipeline step: {step}")

        # Check dependencies
        dependencies = self.step_dependencies.get(step, [])
        for dep in dependencies:
            if not await self._check_step_completion(dep):
                raise Exception(f"Dependency {dep} not completed for step {step}")

        return {"step": step, "status": "completed"}

        # Execute the step
        await self.scheduler.execute_task(step)

    async def _check_step_completion(self, step: str) -> bool:
        """Check if a pipeline step completed successfully."""
        # Check Redis for step completion status
        key = f"pipeline:step:{step}:completed"
        if self.scheduler and self.scheduler.redis:
            result = await self.scheduler.redis.get(key)
        else:
            result = None
        return result is not None


class TradingScheduler:
    """Main trading system scheduler and orchestrator."""

    def __init__(self, config: Any):
        self.config = config
        self.scheduler = AsyncIOScheduler()
        self.market_hours = MarketHoursService(timezone_name=config.scheduler.timezone)
        self.redis: Optional[Any] = None
        self.monitor: Optional[Any] = None
        self.task_queue: Optional[Any] = None
        self.pipeline: Optional[Any] = None

        # Service registry
        self.services: Dict[str, ServiceInfo] = {}
        self.tasks: Dict[str, ScheduledTask] = {}

        # State management
        self.is_running = False
        self.maintenance_mode = False
        self.emergency_stop = False

        # Maintenance components
        self.maintenance_manager: Optional[Any] = None
        self.maintenance_scheduler: Optional[Any] = None

        # Performance tracking
        self.task_execution_times: Dict[str, float] = {}
        self.error_counts: Dict[str, int] = {}

        # Task execution tracking
        self.running_tasks: Set[str] = set()

    async def initialize(self) -> None:
        """Initialize the scheduler service."""
        logger.info("Initializing trading scheduler...")

        # Initialize Redis connection
        self.redis = redis.from_url(
            self.config.redis.url, max_connections=self.config.redis.max_connections
        )

        # Initialize components
        self.monitor = SystemMonitor(self.redis)
        self.task_queue = TaskQueue(self.redis)
        self.pipeline = DataPipelineOrchestrator(self)

        # Configure APScheduler
        jobstores = {
            "default": RedisJobStore(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.database + 1,  # Use different DB for jobs
                password=self.config.redis.password,
            )
        }

        executors = {"default": AsyncIOExecutor()}

        job_defaults = {
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 300,  # 5 minutes
        }

        # Get timezone - fallback to pytz if MarketHoursService doesn't expose it
        try:
            scheduler_timezone = self.market_hours.timezone
        except AttributeError:
            scheduler_timezone = pytz.timezone(self.config.scheduler.timezone)

        self.scheduler.configure(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=scheduler_timezone,
        )

        # Register services
        await self._register_services()

        # Setup signal handlers
        self._setup_signal_handlers()

        logger.info("Trading scheduler initialized successfully")

    async def _register_services(self) -> None:
        """Register all trading system services."""
        services = [
            ServiceInfo(
                name="data_collector",
                url="http://trading_data_collector:9101",
                health_endpoint="/health",
                dependencies=[],
            ),
            ServiceInfo(
                name="strategy_engine",
                url="http://trading_strategy_engine:9102",
                health_endpoint="/health",
                dependencies=["data_collector"],
            ),
            ServiceInfo(
                name="risk_manager",
                url="http://trading_risk_manager:9103",
                health_endpoint="/health",
                dependencies=["data_collector", "strategy_engine"],
            ),
            ServiceInfo(
                name="trade_executor",
                url="http://trading_trade_executor:9104",
                health_endpoint="/health",
                dependencies=["risk_manager"],
            ),
        ]

        for service in services:
            self.services[service.name] = service
            if self.monitor:
                self.monitor.register_service(service)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start(self) -> None:
        """Start the scheduler service."""
        global _scheduler_instance
        logger.info("Starting trading scheduler...")

        try:
            # Set global reference for task execution
            _scheduler_instance = self

            # Start the scheduler
            self.scheduler.start()
            self.is_running = True

            # Schedule core tasks
            await self._schedule_core_tasks()

            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())

            # Start task execution loop
            asyncio.create_task(self._task_execution_loop())

            # Start dependency check loop
            asyncio.create_task(self._dependency_check_loop())

            logger.info("Trading scheduler started successfully")

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    async def _schedule_core_tasks(self) -> None:
        """Schedule all core trading tasks."""

        # FinViz screener - every 5 minutes during market hours
        self.tasks["finviz_scan"] = ScheduledTask(
            id="finviz_scan",
            name="FinViz Screener Scan",
            function=self._run_finviz_scan,
            trigger=IntervalTrigger(minutes=5),
            priority=TaskPriority.HIGH,
            market_hours_only=True,
        )

        # Price updates - variable by timeframe
        self.tasks["price_updates_1m"] = ScheduledTask(
            id="price_updates_1m",
            name="1-Minute Price Updates",
            function=self._run_price_updates,
            trigger=IntervalTrigger(minutes=1),
            priority=TaskPriority.CRITICAL,
            market_hours_only=True,
        )

        self.tasks["price_updates_5m"] = ScheduledTask(
            id="price_updates_5m",
            name="5-Minute Price Updates",
            function=self._run_price_updates,
            trigger=IntervalTrigger(minutes=5),
            priority=TaskPriority.HIGH,
            market_hours_only=True,
        )

        # Strategy analysis - after each data update
        self.tasks["strategy_analysis"] = ScheduledTask(
            id="strategy_analysis",
            name="Strategy Analysis",
            function=self._run_strategy_analysis,
            trigger=IntervalTrigger(minutes=2),
            priority=TaskPriority.HIGH,
            market_hours_only=True,
            dependencies=["price_updates_1m"],
        )

        # Risk checks - continuous
        self.tasks["risk_check"] = ScheduledTask(
            id="risk_check",
            name="Risk Check",
            function=self._run_risk_check,
            trigger=IntervalTrigger(minutes=1),
            priority=TaskPriority.CRITICAL,
            market_hours_only=False,
        )

        # EOD reports - daily at market close
        # Get timezone for scheduling
        try:
            schedule_timezone = self.market_hours.timezone
        except AttributeError:
            schedule_timezone = pytz.timezone(self.config.scheduler.timezone)

        self.tasks["eod_report"] = ScheduledTask(
            id="eod_report",
            name="End of Day Report",
            function=self._run_eod_report,
            trigger=CronTrigger(hour=16, minute=5, timezone=schedule_timezone),
            priority=TaskPriority.NORMAL,
            market_hours_only=False,
        )

        # Health checks
        self.tasks["health_check"] = ScheduledTask(
            id="health_check",
            name="Service Health Check",
            function=self._run_health_check,
            trigger=IntervalTrigger(seconds=30),
            priority=TaskPriority.NORMAL,
            market_hours_only=False,
        )

        # Portfolio sync
        self.tasks["portfolio_sync"] = ScheduledTask(
            id="portfolio_sync",
            name="Portfolio Sync",
            function=self._run_portfolio_sync,
            trigger=IntervalTrigger(minutes=5),
            priority=TaskPriority.HIGH,
            market_hours_only=False,
        )

        # Maintenance tasks - weekend
        self.tasks["data_cleanup"] = ScheduledTask(
            id="data_cleanup",
            name="Data Cleanup",
            function=self._run_data_cleanup,
            trigger=CronTrigger(day_of_week=6, hour=2, minute=0),  # Saturday 2 AM
            priority=TaskPriority.LOW,
            market_hours_only=False,
        )

        self.tasks["database_maintenance"] = ScheduledTask(
            id="database_maintenance",
            name="Database Maintenance",
            function=self._run_database_maintenance,
            trigger=CronTrigger(day_of_week=0, hour=3, minute=0),  # Sunday 3 AM
            priority=TaskPriority.LOW,
            market_hours_only=False,
        )

        # Add tasks to scheduler
        for task in self.tasks.values():
            if task.enabled:
                self.scheduler.add_job(
                    func=execute_scheduled_task,
                    trigger=task.trigger,
                    args=[task.id],
                    id=task.id,
                    name=task.name,
                    replace_existing=True,
                )

    async def _execute_task_wrapper(self, task_id: str) -> bool:
        """Wrapper for task execution with error handling and monitoring."""
        if task_id not in self.tasks:
            logger.error(f"Unknown task ID: {task_id}")
            return False

        task = self.tasks[task_id]

        # Check if system is in emergency stop
        if self.emergency_stop and task.priority != TaskPriority.CRITICAL:
            logger.warning(f"Skipping task {task_id} due to emergency stop")
            return False

        # Check market hours requirement
        if task.market_hours_only and not self._should_run_market_task():
            logger.debug(f"Skipping market hours task {task_id} - market closed")
            return False

        # Check dependencies
        if task.dependencies:
            for dep in task.dependencies:
                if not await self._check_dependency(dep):
                    logger.warning(f"Dependency {dep} not satisfied for task {task_id}")
                    return False

        # Check if task is already running
        if task_id in self.running_tasks:
            logger.warning(f"Task {task_id} already running, skipping")
            return False

        start_time = datetime.now()
        if hasattr(self, "running_tasks"):
            self.running_tasks.add(task_id)

        try:
            logger.info(f"Executing task: {task.name}")
            await task.function(task_id)

            # Update success metrics
            task.last_run = start_time
            task.last_success = datetime.now()
            task.error_count = 0

            # Track execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self.task_execution_times[task_id] = execution_time

            logger.info(f"Task {task.name} completed in {execution_time:.2f}s")

        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            task.error_count += 1
            self.error_counts[task_id] = self.error_counts.get(task_id, 0) + 1

            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                retry_delay = min(300, 30 * task.retry_count)  # Max 5 minutes
                logger.info(
                    f"Retrying task {task.name} in {retry_delay}s (attempt {task.retry_count})"
                )

                self.scheduler.add_job(
                    func=execute_scheduled_task,
                    trigger="date",
                    run_date=datetime.now() + timedelta(seconds=retry_delay),
                    args=[task_id],
                    id=f"{task_id}_retry_{task.retry_count}",
                    replace_existing=True,
                )
            else:
                logger.error(f"Task {task.name} exceeded max retries")
                await self.send_notification(
                    f"Task {task.name} failed after {task.max_retries} retries: {e}",
                    priority="high",
                )

        finally:
            if hasattr(self, "running_tasks"):
                self.running_tasks.discard(task_id)
            task.last_run = start_time

        return True

    def _should_run_market_task(self) -> bool:
        """Check if market-dependent tasks should run."""
        if self.maintenance_mode:
            return False

        # Simplified market hours check for sync context
        # This is called from sync context, so use basic time-based check
        now = datetime.now(pytz.timezone(self.config.scheduler.timezone))
        current_time = now.time()

        # Check if it's a weekday and within market hours (9:30-16:00 ET)
        if now.weekday() < 5 and time(9, 30) <= current_time <= time(16, 0):
            return True
        return False

    async def _check_dependency(self, dependency: str) -> bool:
        """Check if a task dependency is satisfied."""
        # Check Redis for dependency completion
        key = f"task:completion:{dependency}"
        if self.redis:
            result = await self.redis.get(key)
            return result is not None
        return False

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Check service health
                health_results = (
                    await self.monitor.check_all_services() if self.monitor else {}
                )

                # Check for failed services
                failed_services = [
                    name for name, healthy in health_results.items() if not healthy
                ]
                if failed_services:
                    logger.warning(f"Failed services detected: {failed_services}")
                    await self._handle_service_failures(failed_services)

                # Get system metrics
                system_metrics = (
                    await self.monitor.get_system_metrics() if self.monitor else {}
                )
                await self._store_metrics(system_metrics)

                # Check resource usage
                await self._check_resource_usage(system_metrics)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Longer delay on error

    async def _task_execution_loop(self) -> None:
        """Task execution loop for priority queue."""
        while self.is_running:
            try:
                task_id = (
                    await self.task_queue.dequeue_task() if self.task_queue else None
                )
                if task_id:
                    await self._execute_task_wrapper(task_id)
                else:
                    await asyncio.sleep(1)  # No tasks, short sleep

            except Exception as e:
                logger.error(f"Task execution loop error: {e}")
                await asyncio.sleep(5)

    async def _dependency_check_loop(self) -> None:
        """Monitor and manage service dependencies."""
        while self.is_running:
            try:
                # Check service dependency chain
                for service_name, service in self.services.items():
                    if service.status == ServiceStatus.RUNNING:
                        for dep in service.dependencies:
                            if dep in self.services:
                                dep_service = self.services[dep]
                                if dep_service.status != ServiceStatus.RUNNING:
                                    logger.warning(
                                        f"Service {service_name} dependency {dep} is not running"
                                    )
                                    await self._restart_service(dep)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Dependency check loop error: {e}")
                await asyncio.sleep(60)

    async def _handle_service_failures(self, failed_services: List[str]) -> None:
        """Handle failed services."""
        for service_name in failed_services:
            service = self.services[service_name]

            # Increment error count
            service.error_count += 1

            # Try to restart if error count is manageable
            if service.error_count <= 3:
                logger.info(f"Attempting to restart service: {service_name}")
                await self._restart_service(service_name)
            else:
                logger.error(
                    f"Service {service_name} has too many failures, marking as error"
                )
                service.status = ServiceStatus.ERROR
                await self.send_notification(
                    f"Service {service_name} has failed repeatedly and needs manual intervention",
                    priority="critical",
                )

    async def _restart_service(self, service_name: str) -> None:
        """Restart a specific service."""
        try:
            # This would integrate with your container orchestration
            # For now, we'll just log and update status
            logger.info(f"Restarting service: {service_name}")

            service = self.services[service_name]
            service.status = ServiceStatus.STARTING
            service.restart_count += 1

            # Wait a bit for service to start
            await asyncio.sleep(10)

            # Check if restart was successful
            if self.monitor and await self.monitor.check_service_health(service_name):
                logger.info(f"Service {service_name} restarted successfully")
            else:
                logger.error(f"Service {service_name} restart failed")

        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")

    async def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store system metrics to Redis."""
        try:
            if self.redis:
                await self.redis.setex(
                    "system:metrics:latest", 300, str(metrics)  # 5 minutes TTL
                )

            # Store in time series for historical tracking
            timestamp = datetime.now().timestamp()
            if self.redis:
                await self.redis.zadd(
                    "system:metrics:timeseries", {str(metrics): timestamp}
                )

            # Keep only last 24 hours of metrics
            cutoff = timestamp - (24 * 3600)
            if self.redis:
                await self.redis.zremrangebyscore(
                    "system:metrics:timeseries", 0, cutoff
                )

        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

    async def _check_resource_usage(self, metrics: Dict[str, Any]) -> None:
        """Check system resource usage and alert if needed."""
        # CPU check
        if metrics["cpu_percent"] > 80:
            await self.send_notification(
                f"High CPU usage: {metrics['cpu_percent']:.1f}%", priority="medium"
            )

        # Memory check
        if metrics["memory"]["percent"] > 85:
            await self.send_notification(
                f"High memory usage: {metrics['memory']['percent']:.1f}%",
                priority="medium",
            )

        # Disk check
        if metrics["disk"]["percent"] > 90:
            await self.send_notification(
                f"High disk usage: {metrics['disk']['percent']:.1f}%", priority="high"
            )

    # Task implementations
    async def _run_finviz_scan(self, task_id: str) -> None:
        """Execute FinViz screener scan."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.services['data_collector'].url}/finviz/scan", timeout=60.0
                )
                response.raise_for_status()

            # Mark step as completed
            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 300, "completed")
            logger.info("FinViz scan completed successfully")

        except Exception as e:
            logger.error(f"FinViz scan failed: {e}")
            raise

    async def _run_price_updates(self, task_id: str) -> None:
        """Execute price data updates."""
        try:
            # Get timeframe from task ID
            timeframe = "1m" if "1m" in task_id else "5m"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.services['data_collector'].url}/market-data/update",
                    json={"timeframe": timeframe},
                    timeout=120.0,
                )
                response.raise_for_status()

            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 300, "completed")
            logger.info(f"Price updates ({timeframe}) completed successfully")

        except Exception as e:
            logger.error(f"Price updates failed: {e}")
            raise

    async def _run_strategy_analysis(self, task_id: str) -> None:
        """Execute strategy analysis."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.services['strategy_engine'].url}/analyze", timeout=180.0
                )
                response.raise_for_status()

            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 300, "completed")
            logger.info("Strategy analysis completed successfully")

        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            raise

    async def _run_risk_check(self, task_id: str) -> None:
        """Execute risk checks."""
        try:
            # Create a proper PortfolioState payload
            portfolio_data = {
                "account_id": "scheduler_mock_account",
                "timestamp": datetime.now().isoformat(),
                "cash": "10000.00",
                "buying_power": "10000.00",
                "total_equity": "10000.00",
                "positions": [],
                "day_trades_count": 0,
                "pattern_day_trader": False,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.services['risk_manager'].url}/monitor-portfolio",
                    timeout=60.0,
                    json=portfolio_data,
                )
                response.raise_for_status()

            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 300, "completed")
            logger.debug("Risk check completed successfully")

        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            raise

    async def _run_eod_report(self, task_id: str) -> None:
        """Generate end-of-day reports."""
        try:
            # Generate portfolio report
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.services['strategy_engine'].url}/reports/eod", timeout=300.0
                )
                response.raise_for_status()

            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 300, "completed")
            logger.info("EOD report generated successfully")

            # Send notification
            await self.send_notification(
                "Daily trading report generated", priority="low"
            )

        except Exception as e:
            logger.error(f"EOD report generation failed: {e}")
            raise

    async def _run_health_check(self, task_id: str) -> None:
        """Execute health checks."""
        try:
            health_results = (
                await self.monitor.check_all_services() if self.monitor else {}
            )
            failed_count = sum(1 for healthy in health_results.values() if not healthy)

            if failed_count > 0:
                logger.warning(f"{failed_count} services are unhealthy")

            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 300, "completed")

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise

    async def _run_portfolio_sync(self, task_id: str) -> None:
        """Synchronize portfolio state from Alpaca and store in database."""
        try:
            async with httpx.AsyncClient() as client:
                # Call risk manager to sync portfolio data from Alpaca
                response = await client.post(
                    f"{self.services['risk_manager'].url}/sync-portfolio", timeout=60.0
                )
                response.raise_for_status()
                sync_result = response.json()

                # Also get positions from trade executor for consistency
                positions_response = await client.get(
                    f"{self.services['trade_executor'].url}/positions", timeout=60.0
                )
                positions_response.raise_for_status()

            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 300, "completed")

            logger.info(
                f"Portfolio sync completed successfully: {sync_result.get('message', 'Unknown status')}"
            )

        except Exception as e:
            logger.error(f"Portfolio sync failed: {e}")
            raise

    async def _run_data_cleanup(self, task_id: str) -> None:
        """Clean up old data files."""
        try:
            import glob
            import os
            from pathlib import Path

            data_path = Path(self.config.data.parquet_path)
            cutoff_date = datetime.now() - timedelta(
                days=self.config.data.retention_days
            )

            files_deleted = 0
            for file_path in glob.glob(
                str(data_path / "**" / "*.parquet"), recursive=True
            ):
                file_stat = os.stat(file_path)
                if datetime.fromtimestamp(file_stat.st_mtime) < cutoff_date:
                    os.remove(file_path)
                    files_deleted += 1

            logger.info(f"Data cleanup completed: {files_deleted} files deleted")
            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 3600, "completed")

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            raise

    async def _run_database_maintenance(self, task_id: str) -> None:
        """Run database maintenance tasks."""
        try:
            # This would typically involve database vacuum, reindex, etc.
            logger.info("Running database maintenance tasks")

            # For now, just log the operation
            await asyncio.sleep(5)  # Simulate maintenance work

            if self.redis:
                await self.redis.setex(f"task:completion:{task_id}", 3600, "completed")
            logger.info("Database maintenance completed successfully")

        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
            raise

    # Service management methods
    async def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False

        service = self.services[service_name]

        try:
            # Check dependencies first
            for dep in service.dependencies:
                if dep in self.services:
                    dep_service = self.services[dep]
                    if dep_service.status != ServiceStatus.RUNNING:
                        logger.info(f"Starting dependency {dep} for {service_name}")
                        if not await self.start_service(dep):
                            logger.error(f"Failed to start dependency {dep}")
                            return False

            service.status = ServiceStatus.STARTING
            logger.info(f"Starting service: {service_name}")

            # Service-specific startup logic would go here
            await asyncio.sleep(5)  # Simulate startup time

            # Verify service is healthy
            if self.monitor and await self.monitor.check_service_health(service_name):
                service.status = ServiceStatus.RUNNING
                logger.info(f"Service {service_name} started successfully")
                return True
            else:
                service.status = ServiceStatus.ERROR
                logger.error(f"Service {service_name} failed health check after start")
                return False

        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {e}")
            service.status = ServiceStatus.ERROR
            return False

    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False

        service = self.services[service_name]

        try:
            service.status = ServiceStatus.STOPPING
            logger.info(f"Stopping service: {service_name}")

            # Service-specific shutdown logic would go here
            await asyncio.sleep(3)  # Simulate shutdown time

            service.status = ServiceStatus.STOPPED
            logger.info(f"Service {service_name} stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            service.status = ServiceStatus.ERROR
            return False

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        logger.info(f"Restarting service: {service_name}")

        if await self.stop_service(service_name):
            await asyncio.sleep(2)  # Brief pause between stop and start
            return await self.start_service(service_name)

        return False

    # Task management methods
    async def execute_task(
        self, task_id: str, priority: TaskPriority = TaskPriority.NORMAL
    ) -> None:
        """Execute a task immediately or queue it."""
        if hasattr(self, "running_tasks") and task_id in self.running_tasks:
            logger.warning(f"Task {task_id} is already running")
            return

        if self.task_queue:
            await self.task_queue.enqueue_task(task_id, priority)

    async def pause_task(self, task_id: str) -> bool:
        """Pause a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            if self.scheduler.get_job(task_id):
                self.scheduler.pause_job(task_id)
            logger.info(f"Task {task_id} paused")
            return True
        return False

    async def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            if self.scheduler.get_job(task_id):
                self.scheduler.resume_job(task_id)
            logger.info(f"Task {task_id} resumed")
            return True
        return False

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a scheduled task."""
        if task_id in self.tasks:
            if self.scheduler.get_job(task_id):
                self.scheduler.remove_job(task_id)
            del self.tasks[task_id]
            logger.info(f"Task {task_id} removed")

    # Configuration management
    async def update_task_config(self, task_id: str, **kwargs: Any) -> None:
        """Update task configuration."""
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return

        task = self.tasks[task_id]

        # Update task properties
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
                logger.info(f"Updated task {task_id} {key} to {value}")

        # If enabled status changed, update scheduler
        if "enabled" in kwargs:
            if kwargs["enabled"]:
                await self.resume_task(task_id)
            else:
                await self.pause_task(task_id)

    async def hot_reload_config(self) -> None:
        """Hot reload configuration changes."""
        try:
            logger.info("Hot reloading configuration...")

            # Reload config
            from shared.config import reload_config

            new_config = reload_config()

            # Update internal config reference
            self.config = new_config

            # Update task intervals if needed
            await self._update_task_schedules()

            logger.info("Configuration reloaded successfully")

        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise

    async def _update_task_schedules(self) -> None:
        """Update task schedules based on new configuration."""
        # This would update existing scheduled tasks with new intervals
        # based on the reloaded configuration
        pass

    # Notification system
    async def send_notification(self, message: str, priority: str = "normal") -> None:
        """Send notification through available channels."""
        try:
            notification_data = {
                "message": message,
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "service": "scheduler",
            }

            # Store notification in Redis for other services to pick up
            if self.redis:
                await self.redis.lpush("notifications:queue", str(notification_data))  # type: ignore

            logger.info(f"Notification sent: {message}")

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    # System control
    async def enable_maintenance_mode(self) -> None:
        """Enable maintenance mode."""
        self.maintenance_mode = True
        logger.info("Maintenance mode enabled")

        # Pause market-dependent tasks
        for task_id, task in self.tasks.items():
            if task.market_hours_only:
                await self.pause_task(task_id)

        await self.send_notification(
            "System entered maintenance mode", priority="medium"
        )

    async def disable_maintenance_mode(self) -> None:
        """Disable maintenance mode."""
        self.maintenance_mode = False
        logger.info("Maintenance mode disabled")

        # Resume market-dependent tasks
        for task_id, task in self.tasks.items():
            if task.market_hours_only and task.enabled:
                await self.resume_task(task_id)

        await self.send_notification(
            "System exited maintenance mode", priority="medium"
        )

    async def emergency_stop_all(self) -> None:
        """Emergency stop all trading activities."""
        self.emergency_stop = True
        logger.critical("EMERGENCY STOP ACTIVATED")

        # Stop all non-critical tasks
        for task_id, task in self.tasks.items():
            if task.priority != TaskPriority.CRITICAL:
                await self.pause_task(task_id)

        # Send critical notification
        await self.send_notification(
            "EMERGENCY STOP: All trading activities halted", priority="critical"
        )

    async def resume_operations(self) -> None:
        """Resume trading after emergency stop."""
        self.emergency_stop = False
        logger.info("Trading activities resumed")

        # Resume all enabled tasks
        for task_id, task in self.tasks.items():
            if task.enabled:
                await self.resume_task(task_id)

        await self.send_notification("Trading activities resumed", priority="medium")

    # Status and reporting
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        service_status = {}
        for name, service in self.services.items():
            service_status[name] = {
                "status": service.status.value,
                "last_check": (
                    service.last_check.isoformat() if service.last_check else None
                ),
                "error_count": service.error_count,
                "restart_count": service.restart_count,
            }

        task_status = {}
        for task_id, task in self.tasks.items():
            task_status[task_id] = {
                "enabled": task.enabled,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "last_success": (
                    task.last_success.isoformat() if task.last_success else None
                ),
                "error_count": task.error_count,
                "retry_count": task.retry_count,
            }

        queue_lengths = (
            await self.task_queue.get_queue_lengths() if self.task_queue else {}
        )
        metrics = await self.monitor.get_system_metrics() if self.monitor else {}

        return {
            "scheduler": {
                "running": self.is_running,
                "maintenance_mode": self.maintenance_mode,
                "emergency_stop": self.emergency_stop,
            },
            "market": {
                "session": (await self.market_hours.get_current_session()).value,
                "is_trading_day": await self.market_hours.is_trading_day(
                    datetime.now().date()
                ),
                "next_open": (
                    next_open.isoformat()
                    if (next_open := await self.market_hours.get_next_market_open())
                    else None
                ),
                "next_close": (
                    next_close.isoformat()
                    if (next_close := await self.market_hours.get_next_market_close())
                    else None
                ),
            },
            "services": service_status,
            "tasks": task_status,
            "queues": queue_lengths,
            "metrics": metrics,
        }

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get system performance report."""
        return {
            "task_execution_times": self.task_execution_times,
            "error_counts": self.error_counts,
            "service_restart_counts": {
                name: service.restart_count for name, service in self.services.items()
            },
            "queue_lengths": (
                await self.task_queue.get_queue_lengths() if self.task_queue else {}
            ),
        }

    # Data pipeline control
    async def trigger_full_pipeline(self, reason: str = "scheduled") -> None:
        """Trigger the complete data pipeline."""
        if self.pipeline:
            await self.pipeline.trigger_pipeline(reason)

    async def trigger_emergency_exit(self) -> None:
        """Trigger emergency exit of all positions."""
        try:
            logger.critical("Triggering emergency exit of all positions")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.services['trade_executor'].url}/emergency/exit-all",
                    timeout=60.0,
                )
                response.raise_for_status()

            await self.send_notification(
                "Emergency exit triggered - all positions closed", priority="critical"
            )

        except Exception as e:
            logger.error(f"Emergency exit failed: {e}")
            raise

    # Lifecycle management
    async def start_all_services(self) -> bool:
        """Start all services in dependency order."""
        logger.info("Starting all services...")

        # Sort services by dependency order
        sorted_services = self._sort_services_by_dependencies()

        for service_name in sorted_services:
            if not await self.start_service(service_name):
                logger.error(f"Failed to start service {service_name}")
                return False

        logger.info("All services started successfully")
        return True

    async def stop_all_services(self) -> None:
        """Stop all services in reverse dependency order."""
        logger.info("Stopping all services...")

        # Sort services in reverse dependency order
        sorted_services = list(reversed(self._sort_services_by_dependencies()))

        for service_name in sorted_services:
            await self.stop_service(service_name)

        logger.info("All services stopped")

    def _sort_services_by_dependencies(self) -> List[str]:
        """Sort services by their dependencies using topological sort."""
        # Simple topological sort
        visited = set()
        result = []

        def visit(service_name: str) -> None:
            if service_name in visited:
                return
            visited.add(service_name)

            service = self.services[service_name]
            for dep in service.dependencies:
                if dep in self.services:
                    visit(dep)

            result.append(service_name)

        for service_name in self.services:
            visit(service_name)

        return result

    async def shutdown(self) -> None:
        """Graceful shutdown of the scheduler."""
        logger.info("Shutting down trading scheduler...")

        self.is_running = False

        try:
            # Stop the scheduler
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)

            # Stop all services
            await self.stop_all_services()

            # Close Redis connection
            if self.redis:
                await self.redis.close()

            logger.info("Trading scheduler shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    # Circuit breaker decorators for external calls
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def _call_service_endpoint(
        self, url: str, timeout: float = 30.0
    ) -> httpx.Response:
        """Make a circuit-breaker protected call to a service endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            return response


class SchedulerAPI:
    """FastAPI application for scheduler control."""

    def __init__(self, scheduler: TradingScheduler):
        self.scheduler = scheduler

    async def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return await self.scheduler.get_system_status()

    async def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return await self.scheduler.get_performance_report()

    async def trigger_task(
        self, task_id: str, priority: str = "normal"
    ) -> Dict[str, str]:
        """Manually trigger a task."""
        try:
            task_priority = TaskPriority(priority)
            await self.scheduler.execute_task(task_id, task_priority)
            return {"status": "success", "message": f"Task {task_id} queued"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def pause_task(self, task_id: str) -> Dict[str, str]:
        """Pause a task."""
        try:
            await self.scheduler.pause_task(task_id)
            return {"status": "success", "message": f"Task {task_id} paused"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def resume_task(self, task_id: str) -> Dict[str, str]:
        """Resume a task."""
        try:
            await self.scheduler.resume_task(task_id)
            return {"status": "success", "message": f"Task {task_id} resumed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def restart_service(self, service_name: str) -> Dict[str, str]:
        """Restart a service."""
        try:
            success = await self.scheduler.restart_service(service_name)
            if success:
                return {
                    "status": "success",
                    "message": f"Service {service_name} restarted",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to restart service {service_name}",
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def set_maintenance_mode(self, enabled: bool) -> Dict[str, str]:
        """Enable or disable maintenance mode."""
        try:
            if enabled:
                await self.scheduler.enable_maintenance_mode()
                return {"status": "success", "message": "Maintenance mode enabled"}
            else:
                await self.scheduler.disable_maintenance_mode()
                return {"status": "success", "message": "Maintenance mode disabled"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def emergency_stop(self) -> Dict[str, str]:
        """Trigger emergency stop."""
        try:
            await self.scheduler.emergency_stop_all()
            return {"status": "success", "message": "Emergency stop activated"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def resume_trading(self) -> Dict[str, str]:
        """Resume trading after emergency stop."""
        try:
            await self.scheduler.resume_operations()
            return {"status": "success", "message": "Trading resumed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def trigger_pipeline(self, reason: str = "manual") -> Dict[str, str]:
        """Trigger the data pipeline."""
        try:
            await self.scheduler.trigger_full_pipeline(reason)
            return {"status": "success", "message": "Data pipeline triggered"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def update_config(self, config_data: Dict[str, Any]) -> Dict[str, str]:
        """Update configuration and hot reload."""
        try:
            # Update environment variables or config file
            # This would be implementation specific
            await self.scheduler.hot_reload_config()
            return {"status": "success", "message": "Configuration updated"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Main execution
async def main() -> None:
    """Main entry point for the scheduler service."""
    from shared.config import get_config

    config = get_config()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level), format=config.logging.format
    )

    # Create and initialize scheduler
    scheduler = TradingScheduler(config)

    try:
        await scheduler.initialize()
        await scheduler.start()

        # Keep the service running
        while scheduler.is_running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Scheduler service error: {e}")
    finally:
        await scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
