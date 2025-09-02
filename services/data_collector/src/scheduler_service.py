"""
Scheduler service for data collection with market hours awareness.

This module provides intelligent scheduling for data collection tasks,
taking into account market hours, holidays, weekends, and rate limiting
requirements for optimal data collection timing.

Enhanced with Alpaca API integration for accurate market hours detection.
"""

import asyncio
import logging
from datetime import date, datetime
from datetime import time as dt_time
from datetime import timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import pytz
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel, Field

from shared.models import TimeFrame

# Import Alpaca market hours functionality with fallback
try:
    from shared.market_hours import (
        get_market_status,
    )
    from shared.market_hours import is_market_open as alpaca_is_market_open

    ALPACA_MARKET_HOURS_AVAILABLE = True
except ImportError:
    logger.warning(
        "Alpaca market hours functionality not available, using fallback logic"
    )
    ALPACA_MARKET_HOURS_AVAILABLE = False
    alpaca_is_market_open = None
    get_market_status = None


logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """Market session types."""

    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_MARKET = "after_market"
    CLOSED = "closed"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class SchedulerConfig(BaseModel):
    """Configuration for scheduler service."""

    # Market hours
    market_open_time: str = Field(
        default="09:30", description="Market open time (HH:MM)"
    )
    market_close_time: str = Field(
        default="16:00", description="Market close time (HH:MM)"
    )
    pre_market_start: str = Field(
        default="04:00", description="Pre-market start time (HH:MM)"
    )
    after_market_end: str = Field(
        default="20:00", description="After-market end time (HH:MM)"
    )
    timezone: str = Field(default="America/New_York", description="Market timezone")

    # Weekend and holiday trading
    trade_weekends: bool = Field(
        default=False, description="Enable weekend data collection"
    )
    trade_holidays: bool = Field(default=False, description="Enable holiday trading")

    # Enhanced market hours detection
    use_alpaca_market_hours: bool = Field(
        default=True, description="Use Alpaca API for accurate market hours"
    )

    # Scheduler settings
    max_workers: int = Field(default=10, description="Maximum concurrent tasks")
    job_defaults: Dict[str, Any] = Field(
        default={"coalesce": True, "max_instances": 1, "misfire_grace_time": 30},
        description="Default job settings",
    )

    # Rate limiting
    enable_smart_scheduling: bool = Field(
        default=True, description="Enable smart task spacing"
    )
    min_task_interval: float = Field(
        default=1.0, description="Minimum interval between tasks"
    )

    # Known US market holidays (will be expanded)
    holidays: List[str] = Field(
        default=[
            "2024-01-01",  # New Year's Day
            "2024-01-15",  # Martin Luther King Jr. Day
            "2024-02-19",  # Presidents' Day
            "2024-03-29",  # Good Friday
            "2024-05-27",  # Memorial Day
            "2024-06-19",  # Juneteenth
            "2024-07-04",  # Independence Day
            "2024-09-02",  # Labor Day
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas
        ],
        description="Market holiday dates (YYYY-MM-DD)",
    )


class ScheduledTask(BaseModel):
    """Scheduled task definition."""

    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    callback: str = Field(..., description="Callback function name")
    priority: TaskPriority = Field(
        default=TaskPriority.NORMAL, description="Task priority"
    )
    enabled: bool = Field(default=True, description="Whether task is enabled")
    market_hours_only: bool = Field(
        default=False, description="Run only during market hours"
    )
    timeframes: Optional[List[TimeFrame]] = Field(
        None, description="Associated timeframes"
    )

    # Scheduling
    interval_seconds: Optional[int] = Field(None, description="Interval in seconds")
    cron_expression: Optional[str] = Field(None, description="Cron expression")

    # Execution settings
    max_runtime_seconds: int = Field(default=300, description="Maximum runtime")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    retry_delay: float = Field(default=5.0, description="Delay between retries")

    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Task dependencies")


class TaskExecutionStats(BaseModel):
    """Statistics for task execution."""

    task_id: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    average_runtime: float = 0.0
    total_runtime: float = 0.0


class MarketHoursManager:
    """Manages market hours and session detection."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.timezone = pytz.timezone(config.timezone)
        self._holiday_dates = set(
            datetime.strptime(h, "%Y-%m-%d").date() for h in config.holidays
        )

        # Enable Alpaca-enhanced mode if available
        self.use_alpaca_api = ALPACA_MARKET_HOURS_AVAILABLE and getattr(
            config, "use_alpaca_market_hours", True
        )
        if self.use_alpaca_api:
            logger.info(
                "MarketHoursManager: Using Alpaca API for enhanced market hours detection"
            )
        else:
            logger.info("MarketHoursManager: Using fallback market hours logic")

    def get_current_session(self) -> MarketSession:
        """Get current market session."""
        now = datetime.now(self.timezone)
        current_date = now.date()
        current_time = now.time()

        # Check if it's a holiday
        if current_date in self._holiday_dates and not self.config.trade_holidays:
            return MarketSession.CLOSED

        # Check if it's a weekend
        if (
            now.weekday() >= 5 and not self.config.trade_weekends
        ):  # Saturday = 5, Sunday = 6
            return MarketSession.CLOSED

        # Parse time strings
        pre_market_start = dt_time.fromisoformat(self.config.pre_market_start)
        market_open = dt_time.fromisoformat(self.config.market_open_time)
        market_close = dt_time.fromisoformat(self.config.market_close_time)
        after_market_end = dt_time.fromisoformat(self.config.after_market_end)

        # Determine session
        if pre_market_start <= current_time < market_open:
            return MarketSession.PRE_MARKET
        elif market_open <= current_time < market_close:
            return MarketSession.REGULAR
        elif market_close <= current_time < after_market_end:
            return MarketSession.AFTER_MARKET
        else:
            return MarketSession.CLOSED

    def is_market_open(self) -> bool:
        """Check if market is currently open (regular hours)."""
        if self.use_alpaca_api:
            try:
                import asyncio

                # Try to get current event loop, create new one if needed
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, can't use run_until_complete
                        # Fall back to traditional method
                        return self.get_current_session() == MarketSession.REGULAR
                    else:
                        return loop.run_until_complete(alpaca_is_market_open())
                except RuntimeError:
                    # No event loop, create one
                    return asyncio.run(alpaca_is_market_open())
            except Exception as e:
                logger.warning(
                    f"Alpaca API market hours check failed, using fallback: {e}"
                )
                # Fall back to original logic
                return self.get_current_session() == MarketSession.REGULAR
        else:
            return self.get_current_session() == MarketSession.REGULAR

    def is_trading_session(self) -> bool:
        """Check if any trading session is active (including pre/after market)."""
        if self.use_alpaca_api:
            try:
                import asyncio

                # Get detailed market status from Alpaca
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Fall back to traditional method in async context
                        session = self.get_current_session()
                        return session in [
                            MarketSession.PRE_MARKET,
                            MarketSession.REGULAR,
                            MarketSession.AFTER_MARKET,
                        ]
                    else:
                        status = loop.run_until_complete(get_market_status())
                        return status.is_trading_session
                except RuntimeError:
                    # No event loop, create one
                    status = asyncio.run(get_market_status())
                    return status.is_trading_session
            except Exception as e:
                logger.warning(
                    f"Alpaca API trading session check failed, using fallback: {e}"
                )
                # Fall back to original logic
                session = self.get_current_session()
                return session in [
                    MarketSession.PRE_MARKET,
                    MarketSession.REGULAR,
                    MarketSession.AFTER_MARKET,
                ]
        else:
            session = self.get_current_session()
            return session in [
                MarketSession.PRE_MARKET,
                MarketSession.REGULAR,
                MarketSession.AFTER_MARKET,
            ]

    def time_until_market_open(self) -> Optional[timedelta]:
        """Get time until next market open."""
        now = datetime.now(self.timezone)

        # If market is open, return None
        if self.is_market_open():
            return None

        # Calculate next market open
        next_open = now.replace(
            hour=int(self.config.market_open_time.split(":")[0]),
            minute=int(self.config.market_open_time.split(":")[1]),
            second=0,
            microsecond=0,
        )

        # If today's market open has passed, move to next trading day
        if now.time() >= dt_time.fromisoformat(self.config.market_open_time):
            next_open += timedelta(days=1)

        # Skip weekends and holidays
        while next_open.weekday() >= 5 or next_open.date() in self._holiday_dates:
            next_open += timedelta(days=1)

        return next_open - now

    def time_until_market_close(self) -> Optional[timedelta]:
        """Get time until market close."""
        if not self.is_market_open():
            return None

        now = datetime.now(self.timezone)
        market_close = now.replace(
            hour=int(self.config.market_close_time.split(":")[0]),
            minute=int(self.config.market_close_time.split(":")[1]),
            second=0,
            microsecond=0,
        )

        return market_close - now


class SchedulerService:
    """
    Intelligent scheduler service for data collection tasks.

    Provides market-aware scheduling, task dependency management,
    priority-based execution, and comprehensive error handling.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.market_hours = MarketHoursManager(config)

        # Initialize APScheduler
        jobstores = {"default": MemoryJobStore()}
        executors = {"default": AsyncIOExecutor()}

        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=config.job_defaults,
            timezone=config.timezone,
        )

        # Task management
        self._tasks: Dict[str, ScheduledTask] = {}
        self._task_stats: Dict[str, TaskExecutionStats] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._task_dependencies: Dict[str, Set[str]] = {}

        # State management
        self.is_running = False
        self._shutdown_event = asyncio.Event()

        # Initialize holiday dates for market hours calculation
        self._holiday_dates = set(
            datetime.strptime(d, "%Y-%m-%d").date() for d in config.holidays
        )

    async def start(self):
        """Start the scheduler service."""
        if self.is_running:
            logger.warning("Scheduler service is already running")
            return

        logger.info("Starting scheduler service...")

        try:
            self.scheduler.start()
            self.is_running = True

            # Add health check job
            await self._add_health_check_job()

            logger.info("Scheduler service started successfully")

        except Exception as e:
            logger.error(f"Failed to start scheduler service: {e}")
            raise

    async def stop(self):
        """Stop the scheduler service."""
        if not self.is_running:
            return

        logger.info("Stopping scheduler service...")

        try:
            self._shutdown_event.set()

            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)

            self.is_running = False
            logger.info("Scheduler service stopped")

        except Exception as e:
            logger.error(f"Error stopping scheduler service: {e}")

    async def register_task(self, task: ScheduledTask, callback: Callable):
        """
        Register a new scheduled task.

        Args:
            task: Task definition
            callback: Async callback function to execute
        """
        logger.info(f"Registering task: {task.name} (ID: {task.id})")

        self._tasks[task.id] = task
        self._callbacks[task.id] = callback
        self._task_stats[task.id] = TaskExecutionStats(task_id=task.id)

        # Set up dependencies
        if task.depends_on:
            self._task_dependencies[task.id] = set(task.depends_on)

        # Schedule the task
        await self._schedule_task(task)

    async def _schedule_task(self, task: ScheduledTask):
        """Schedule a task with APScheduler."""
        if not task.enabled:
            logger.info(f"Task {task.id} is disabled, skipping scheduling")
            return

        # Create wrapped callback with error handling and stats
        wrapped_callback = self._create_wrapped_callback(task)

        try:
            if task.interval_seconds:
                # Interval-based scheduling
                trigger = IntervalTrigger(
                    seconds=task.interval_seconds, timezone=self.config.timezone
                )
            elif task.cron_expression:
                # Cron-based scheduling
                trigger = CronTrigger.from_crontab(
                    task.cron_expression, timezone=self.config.timezone
                )
            else:
                logger.error(f"Task {task.id} has no scheduling configuration")
                return

            # Add job to scheduler
            self.scheduler.add_job(
                wrapped_callback,
                trigger=trigger,
                id=task.id,
                name=task.name,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=60,
            )

            logger.info(f"Scheduled task {task.id} with trigger: {trigger}")

        except Exception as e:
            logger.error(f"Failed to schedule task {task.id}: {e}")

    def _create_wrapped_callback(self, task: ScheduledTask) -> Callable:
        """Create wrapped callback with error handling and statistics."""

        async def wrapped_callback():
            task_id = task.id
            start_time = datetime.now(timezone.utc)

            # Update stats
            stats = self._task_stats[task_id]
            stats.total_runs += 1
            stats.last_run = start_time

            try:
                # Check market hours if required
                if task.market_hours_only and not self._should_run_market_hours_task():
                    logger.debug(f"Skipping {task_id} - outside market hours")
                    return

                # Check dependencies
                if not await self._check_dependencies(task_id):
                    logger.warning(f"Task {task_id} dependencies not met, skipping")
                    return

                logger.info(f"Executing task: {task.name}")

                # Execute callback with timeout
                callback = self._callbacks[task_id]

                try:
                    await asyncio.wait_for(callback(), timeout=task.max_runtime_seconds)

                    # Update success stats
                    stats.successful_runs += 1
                    stats.last_success = datetime.now(timezone.utc)

                    execution_time = (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds()
                    stats.total_runtime += execution_time
                    stats.average_runtime = stats.total_runtime / stats.successful_runs

                    logger.info(
                        f"Task {task.name} completed successfully in {execution_time:.2f}s"
                    )

                except asyncio.TimeoutError:
                    logger.error(
                        f"Task {task_id} timed out after {task.max_runtime_seconds}s"
                    )
                    stats.failed_runs += 1
                    stats.last_failure = datetime.now(timezone.utc)

            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                stats.failed_runs += 1
                stats.last_failure = datetime.now(timezone.utc)

                # Implement retry logic if configured
                if task.retry_count > 0:
                    await self._schedule_retry(task, e)

        return wrapped_callback

    async def _check_dependencies(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied."""
        if task_id not in self._task_dependencies:
            return True

        dependencies = self._task_dependencies[task_id]

        for dep_id in dependencies:
            if dep_id not in self._task_stats:
                logger.warning(f"Dependency {dep_id} not found for task {task_id}")
                return False

            dep_stats = self._task_stats[dep_id]

            # Check if dependency has run successfully recently
            if not dep_stats.last_success:
                logger.debug(f"Dependency {dep_id} has never run successfully")
                return False

            # Check if dependency ran within reasonable time
            time_since_success = datetime.now(timezone.utc) - dep_stats.last_success
            if time_since_success > timedelta(hours=1):  # Configurable threshold
                logger.debug(
                    f"Dependency {dep_id} last success too old: {time_since_success}"
                )
                return False

        return True

    def _should_run_market_hours_task(self) -> bool:
        """Check if market hours tasks should run."""
        session = self.market_hours.get_current_session()

        # Run during regular market hours
        if session == MarketSession.REGULAR:
            return True

        # Run during extended hours if configured
        if session in [MarketSession.PRE_MARKET, MarketSession.AFTER_MARKET]:
            return True  # Allow extended hours for data collection

        return False

    async def _schedule_retry(self, task: ScheduledTask, error: Exception):
        """Schedule a retry for a failed task."""
        retry_time = datetime.now(timezone.utc) + timedelta(seconds=task.retry_delay)

        logger.info(f"Scheduling retry for task {task.id} at {retry_time}")

        self.scheduler.add_job(
            self._callbacks[task.id],
            trigger=DateTrigger(run_date=retry_time),
            id=f"{task.id}_retry_{int(retry_time.timestamp())}",
            max_instances=1,
        )

    async def _add_health_check_job(self):
        """Add health check job for scheduler monitoring."""

        async def health_check():
            try:
                stats = self.get_scheduler_stats()
                logger.debug(
                    f"Scheduler health check: {stats['running_jobs']} jobs running"
                )

                # Check for stuck jobs
                for job in self.scheduler.get_jobs():
                    if hasattr(job, "next_run_time") and job.next_run_time:
                        if job.next_run_time < datetime.now(
                            pytz.timezone(self.config.timezone)
                        ) - timedelta(minutes=10):
                            logger.warning(f"Job {job.id} appears to be stuck")

            except Exception as e:
                logger.error(f"Health check failed: {e}")

        self.scheduler.add_job(
            health_check,
            IntervalTrigger(minutes=5),
            id="scheduler_health_check",
            max_instances=1,
        )

    def create_data_collection_tasks(self) -> List[ScheduledTask]:
        """Create standard data collection tasks."""
        tasks = []

        # FinViz screener task
        tasks.append(
            ScheduledTask(
                id="finviz_momentum_scan",
                name="FinViz Momentum Screener",
                callback="run_finviz_scan",
                priority=TaskPriority.HIGH,
                interval_seconds=300,  # 5 minutes
                cron_expression=None,
                market_hours_only=False,  # Can run outside market hours
                timeframes=[],  # Not applicable for screener
                max_runtime_seconds=120,
                retry_count=2,
            )
        )

        # TwelveData price updates
        tasks.append(
            ScheduledTask(
                id="price_update_5min",
                name="5-Minute Price Update",
                callback="update_5min_data",
                priority=TaskPriority.HIGH,
                interval_seconds=300,  # 5 minutes
                cron_expression=None,
                market_hours_only=True,
                timeframes=[TimeFrame.FIVE_MINUTES],
                max_runtime_seconds=180,
                depends_on=["finviz_momentum_scan"],
            )
        )

        tasks.append(
            ScheduledTask(
                id="price_update_15min",
                name="15-Minute Price Update",
                callback="update_15min_data",
                priority=TaskPriority.NORMAL,
                interval_seconds=900,  # 15 minutes
                cron_expression=None,
                market_hours_only=True,
                timeframes=[TimeFrame.FIFTEEN_MINUTES],
                max_runtime_seconds=240,
            )
        )

        tasks.append(
            ScheduledTask(
                id="price_update_hourly",
                name="Hourly Price Update",
                callback="update_hourly_data",
                priority=TaskPriority.NORMAL,
                interval_seconds=3600,  # 1 hour
                cron_expression=None,
                market_hours_only=True,
                timeframes=[TimeFrame.ONE_HOUR],
                max_runtime_seconds=300,
            )
        )

        # Daily data update at market close
        tasks.append(
            ScheduledTask(
                id="price_update_daily",
                name="Daily Price Update",
                callback="update_daily_data",
                priority=TaskPriority.HIGH,
                interval_seconds=None,
                cron_expression="30 16 * * MON-FRI",  # 4:30 PM weekdays
                market_hours_only=False,
                timeframes=[TimeFrame.ONE_DAY],
                max_runtime_seconds=600,
            )
        )

        # Maintenance tasks
        tasks.append(
            ScheduledTask(
                id="data_cleanup",
                name="Data Cleanup",
                callback="cleanup_old_data",
                priority=TaskPriority.LOW,
                interval_seconds=None,
                cron_expression="0 2 * * *",  # 2 AM daily
                market_hours_only=False,
                timeframes=[],  # Not applicable for maintenance
                max_runtime_seconds=1800,
            )
        )

        tasks.append(
            ScheduledTask(
                id="data_validation",
                name="Data Integrity Validation",
                callback="validate_data_integrity",
                priority=TaskPriority.LOW,
                interval_seconds=None,
                cron_expression="0 3 * * *",  # 3 AM daily
                market_hours_only=False,
                timeframes=[],  # Not applicable for validation
                max_runtime_seconds=3600,
            )
        )

        tasks.append(
            ScheduledTask(
                id="storage_optimization",
                name="Storage Optimization",
                callback="optimize_storage",
                priority=TaskPriority.LOW,
                interval_seconds=None,
                cron_expression="0 1 * * SUN",  # 1 AM every Sunday
                market_hours_only=False,
                timeframes=[],  # Not applicable for optimization
                max_runtime_seconds=7200,
            )
        )

        return tasks

    async def enable_task(self, task_id: str):
        """Enable a scheduled task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True

            # Reschedule if needed
            if self.scheduler.get_job(task_id):
                self.scheduler.resume_job(task_id)
            else:
                await self._schedule_task(self._tasks[task_id])

            logger.info(f"Enabled task: {task_id}")

    async def disable_task(self, task_id: str):
        """Disable a scheduled task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False

            # Pause the job
            if self.scheduler.get_job(task_id):
                self.scheduler.pause_job(task_id)

            logger.info(f"Disabled task: {task_id}")

    async def run_task_now(self, task_id: str):
        """Execute a task immediately."""
        if task_id not in self._tasks or task_id not in self._callbacks:
            logger.error(f"Task {task_id} not found")
            return False

        task = self._tasks[task_id]
        callback = self._callbacks[task_id]

        logger.info(f"Running task {task.name} immediately")

        try:
            # Check dependencies
            if not await self._check_dependencies(task_id):
                logger.warning(f"Cannot run {task_id} - dependencies not met")
                return False

            # Execute callback
            await callback()
            return True

        except Exception as e:
            logger.error(f"Immediate execution of {task_id} failed: {e}")
            return False

    def get_task_stats(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get task execution statistics.

        Args:
            task_id: Specific task ID (all tasks if None)

        Returns:
            Task statistics
        """
        if task_id:
            if task_id in self._task_stats:
                return self._task_stats[task_id].dict()
            else:
                return {}

        return {task_id: stats.dict() for task_id, stats in self._task_stats.items()}

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get overall scheduler statistics."""
        jobs = self.scheduler.get_jobs()

        return {
            "is_running": self.is_running,
            "total_jobs": len(jobs),
            "running_jobs": len([j for j in jobs if j.next_run_time]),
            "paused_jobs": len([j for j in jobs if not j.next_run_time]),
            "market_session": self.market_hours.get_current_session().value,
            "is_market_open": self.market_hours.is_market_open(),
            "time_until_market_open": str(self.market_hours.time_until_market_open()),
            "time_until_market_close": str(self.market_hours.time_until_market_close()),
            "next_job_run": min(
                (j.next_run_time for j in jobs if j.next_run_time), default=None
            ),
        }

    def get_next_job_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get next scheduled job runs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of upcoming job runs
        """
        jobs = self.scheduler.get_jobs()

        # Filter and sort by next run time
        upcoming_jobs = [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time,
                "trigger": str(job.trigger),
            }
            for job in jobs
            if job.next_run_time
        ]

        upcoming_jobs.sort(key=lambda x: x["next_run"])
        return upcoming_jobs[:limit]

    async def reschedule_task(
        self,
        task_id: str,
        new_interval: Optional[int] = None,
        new_cron: Optional[str] = None,
    ):
        """
        Reschedule an existing task.

        Args:
            task_id: Task to reschedule
            new_interval: New interval in seconds
            new_cron: New cron expression
        """
        if task_id not in self._tasks:
            logger.error(f"Task {task_id} not found")
            return

        task = self._tasks[task_id]

        # Update task configuration
        if new_interval:
            task.interval_seconds = new_interval
            task.cron_expression = None
        elif new_cron:
            task.cron_expression = new_cron
            task.interval_seconds = None

        # Remove existing job
        if self.scheduler.get_job(task_id):
            self.scheduler.remove_job(task_id)

        # Reschedule
        await self._schedule_task(task)

        logger.info(f"Rescheduled task {task_id}")

    def pause_all_market_tasks(self):
        """Pause all market-hours-only tasks."""
        for task_id, task in self._tasks.items():
            if task.market_hours_only and self.scheduler.get_job(task_id):
                self.scheduler.pause_job(task_id)
                logger.info(f"Paused market task: {task_id}")

    def resume_all_market_tasks(self):
        """Resume all market-hours-only tasks."""
        for task_id, task in self._tasks.items():
            if task.market_hours_only and self.scheduler.get_job(task_id):
                self.scheduler.resume_job(task_id)
                logger.info(f"Resumed market task: {task_id}")

    async def schedule_one_time_task(
        self,
        name: str,
        callback: Callable,
        run_time: datetime,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """
        Schedule a one-time task.

        Args:
            name: Task name
            callback: Callback function
            run_time: When to run the task
            priority: Task priority

        Returns:
            Task ID
        """
        task_id = f"onetime_{name}_{int(run_time.timestamp())}"

        try:
            self.scheduler.add_job(
                callback,
                trigger=DateTrigger(run_date=run_time),
                id=task_id,
                name=name,
                max_instances=1,
            )

            logger.info(f"Scheduled one-time task {name} for {run_time}")
            return task_id

        except Exception as e:
            logger.error(f"Failed to schedule one-time task {name}: {e}")
            raise

    def adjust_schedule_for_market_events(self, event_type: str, event_time: datetime):
        """
        Adjust scheduling around market events.

        Args:
            event_type: Type of event (earnings, economic_data, etc.)
            event_time: Event timestamp
        """
        logger.info(f"Adjusting schedule for {event_type} at {event_time}")

        # Increase data collection frequency around events
        if event_type in ["earnings", "economic_data"]:
            # Schedule extra data collection 15 minutes before and after
            before_time = event_time - timedelta(minutes=15)
            after_time = event_time + timedelta(minutes=15)

            for timing, desc in [(before_time, "before"), (after_time, "after")]:
                if timing > datetime.now(timezone.utc):
                    self.scheduler.add_job(
                        lambda: self._callbacks.get(
                            "price_update_5min", lambda: None
                        )(),
                        trigger=DateTrigger(run_date=timing),
                        id=f"event_update_{event_type}_{desc}_{int(timing.timestamp())}",
                        max_instances=1,
                    )

    def get_market_schedule_for_week(
        self, start_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get market schedule for the week.

        Args:
            start_date: Start date (current week if None)

        Returns:
            Weekly market schedule
        """
        if start_date is None:
            start_date = date.today()

        # Get start of week (Monday)
        days_since_monday = start_date.weekday()
        week_start = start_date - timedelta(days=days_since_monday)

        schedule = {}

        for i in range(7):
            day = week_start + timedelta(days=i)
            day_name = day.strftime("%A")

            # Check if it's a trading day
            is_holiday = day in self._holiday_dates
            is_weekend = day.weekday() >= 5

            session_info = {
                "date": day.isoformat(),
                "day_name": day_name,
                "is_trading_day": not (is_holiday or is_weekend),
                "is_holiday": is_holiday,
                "is_weekend": is_weekend,
            }

            if session_info["is_trading_day"]:
                session_info.update(
                    {
                        "pre_market_start": self.config.pre_market_start,
                        "market_open": self.config.market_open_time,
                        "market_close": self.config.market_close_time,
                        "after_market_end": self.config.after_market_end,
                    }
                )

            schedule[day.isoformat()] = session_info

        return {
            "week_start": week_start.isoformat(),
            "timezone": self.config.timezone,
            "schedule": schedule,
        }

    async def optimize_task_scheduling(self):
        """Optimize task scheduling based on historical performance."""
        logger.info("Optimizing task scheduling based on performance data...")

        try:
            # Analyze task performance
            for task_id, stats in self._task_stats.items():
                if stats.total_runs < 5:  # Need sufficient data
                    continue

                task = self._tasks[task_id]
                success_rate = stats.successful_runs / stats.total_runs

                # Adjust retry settings based on success rate
                if success_rate < 0.8:  # Less than 80% success
                    task.retry_count = min(task.retry_count + 1, 5)
                    task.retry_delay = min(task.retry_delay * 1.5, 60.0)
                    logger.info(
                        f"Increased retry settings for {task_id} (success rate: {success_rate:.2f})"
                    )

                elif success_rate > 0.95:  # More than 95% success
                    task.retry_count = max(task.retry_count - 1, 1)
                    task.retry_delay = max(task.retry_delay * 0.8, 1.0)

                # Adjust timeout based on average runtime
                if stats.average_runtime > 0:
                    # Set timeout to 3x average runtime with min/max bounds
                    new_timeout = max(60, min(1800, int(stats.average_runtime * 3)))
                    if abs(new_timeout - task.max_runtime_seconds) > 30:
                        task.max_runtime_seconds = new_timeout
                        logger.info(f"Adjusted timeout for {task_id} to {new_timeout}s")

        except Exception as e:
            logger.error(f"Task optimization failed: {e}")

    async def get_task_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive task performance report.

        Returns:
            Performance report with recommendations
        """
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scheduler_status": self.get_scheduler_stats(),
            "task_performance": {},
            "recommendations": [],
            "alerts": [],
        }

        try:
            total_tasks = len(self._task_stats)
            total_runs = sum(stats.total_runs for stats in self._task_stats.values())
            total_failures = sum(
                stats.failed_runs for stats in self._task_stats.values()
            )

            report["summary"] = {
                "total_tasks": total_tasks,
                "total_runs": total_runs,
                "total_failures": total_failures,
                "overall_success_rate": (
                    (total_runs - total_failures) / total_runs if total_runs > 0 else 0
                ),
            }

            # Analyze each task
            for task_id, stats in self._task_stats.items():
                task = self._tasks[task_id]

                if stats.total_runs == 0:
                    continue

                success_rate = stats.successful_runs / stats.total_runs
                avg_runtime = stats.average_runtime

                task_report = {
                    "task_name": task.name,
                    "total_runs": stats.total_runs,
                    "success_rate": success_rate,
                    "average_runtime": avg_runtime,
                    "last_run": stats.last_run.isoformat() if stats.last_run else None,
                    "last_success": (
                        stats.last_success.isoformat() if stats.last_success else None
                    ),
                    "last_failure": (
                        stats.last_failure.isoformat() if stats.last_failure else None
                    ),
                    "priority": task.priority.value,
                    "enabled": task.enabled,
                }

                # Generate recommendations
                if success_rate < 0.8:
                    report["alerts"].append(
                        f"Task {task.name} has low success rate: {success_rate:.2f}"
                    )

                if stats.last_success and datetime.now(
                    timezone.utc
                ) - stats.last_success > timedelta(hours=24):
                    report["alerts"].append(
                        f"Task {task.name} hasn't succeeded in over 24 hours"
                    )

                if avg_runtime > task.max_runtime_seconds * 0.8:
                    report["recommendations"].append(
                        f"Consider increasing timeout for {task.name}"
                    )

                report["task_performance"][task_id] = task_report

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            report["error"] = str(e)

        return report

    async def handle_market_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Handle market events by adjusting scheduling.

        Args:
            event_type: Type of market event
            event_data: Event details
        """
        logger.info(f"Handling market event: {event_type}")

        try:
            if event_type == "market_open":
                # Resume market-hours tasks
                self.resume_all_market_tasks()

                # Schedule increased data collection frequency for first hour
                for minutes in [5, 15, 30, 45]:
                    run_time = datetime.now(
                        pytz.timezone(self.config.timezone)
                    ) + timedelta(minutes=minutes)
                    await self.schedule_one_time_task(
                        f"market_open_boost_{minutes}min",
                        self._callbacks.get("update_5min_data", lambda: None),
                        run_time,
                        TaskPriority.HIGH,
                    )

            elif event_type == "market_close":
                # Pause market-hours tasks
                self.pause_all_market_tasks()

                # Schedule end-of-day data collection
                eod_time = datetime.now(
                    pytz.timezone(self.config.timezone)
                ) + timedelta(minutes=30)
                await self.schedule_one_time_task(
                    "end_of_day_collection",
                    self._callbacks.get("update_daily_data", lambda: None),
                    eod_time,
                    TaskPriority.HIGH,
                )

            elif event_type == "high_volatility":
                # Increase data collection frequency
                ticker = event_data.get("ticker")
                if ticker:
                    # Schedule extra updates for the next hour
                    for minutes in [5, 10, 20, 30, 45, 60]:
                        run_time = datetime.now(
                            pytz.timezone(self.config.timezone)
                        ) + timedelta(minutes=minutes)
                        await self.schedule_one_time_task(
                            f"volatility_boost_{ticker}_{minutes}min",
                            lambda t=ticker: self._callbacks.get(
                                "force_ticker_update", lambda x: None
                            )(t),
                            run_time,
                            TaskPriority.HIGH,
                        )

            elif event_type == "api_error":
                # Reduce frequency temporarily
                service = event_data.get("service")
                if service == "twelvedata":
                    # Pause TwelveData tasks for 10 minutes
                    for task_id in self._tasks:
                        if "price_update" in task_id:
                            job = self.scheduler.get_job(task_id)
                            if job:
                                self.scheduler.pause_job(task_id)

                    # Schedule resumption
                    resume_time = datetime.now(
                        pytz.timezone(self.config.timezone)
                    ) + timedelta(minutes=10)
                    await self.schedule_one_time_task(
                        "resume_twelvedata_tasks",
                        self._resume_twelvedata_tasks,
                        resume_time,
                        TaskPriority.NORMAL,
                    )

        except Exception as e:
            logger.error(f"Failed to handle market event {event_type}: {e}")

    async def _resume_twelvedata_tasks(self):
        """Resume TwelveData-related tasks."""
        for task_id in self._tasks:
            if "price_update" in task_id:
                job = self.scheduler.get_job(task_id)
                if job:
                    self.scheduler.resume_job(task_id)
                    logger.info(f"Resumed task: {task_id}")

    def get_task_queue_status(self) -> Dict[str, Any]:
        """
        Get current task queue status.

        Returns:
            Task queue information
        """
        jobs = self.scheduler.get_jobs()

        # Categorize jobs by status
        running_jobs = []
        pending_jobs = []
        paused_jobs = []

        for job in jobs:
            job_info = {
                "id": job.id,
                "name": job.name,
                "next_run": (
                    job.next_run_time.isoformat() if job.next_run_time else None
                ),
                "trigger": str(job.trigger),
            }

            if not job.next_run_time:
                paused_jobs.append(job_info)
            elif job.next_run_time <= datetime.now(pytz.timezone(self.config.timezone)):
                running_jobs.append(job_info)
            else:
                pending_jobs.append(job_info)

        return {
            "total_jobs": len(jobs),
            "running_jobs": len(running_jobs),
            "pending_jobs": len(pending_jobs),
            "paused_jobs": len(paused_jobs),
            "running": running_jobs,
            "pending": pending_jobs[:10],  # Show next 10 pending
            "paused": paused_jobs,
        }

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, "scheduler") and self.scheduler.running:
            self.scheduler.shutdown(wait=False)


# Utility functions
def create_market_aware_trigger(
    base_interval: int,
    market_hours_only: bool = True,
    timezone: str = "America/New_York",
) -> IntervalTrigger:
    """
    Create market-aware interval trigger.

    Args:
        base_interval: Base interval in seconds
        market_hours_only: Whether to run only during market hours
        timezone: Market timezone

    Returns:
        Configured IntervalTrigger
    """
    # For market hours only, we use cron triggers during trading days
    if market_hours_only:
        # This is a simplified approach - in practice, you'd want more sophisticated logic
        return IntervalTrigger(seconds=base_interval, timezone=timezone)
    else:
        return IntervalTrigger(seconds=base_interval, timezone=timezone)


def calculate_optimal_intervals(
    api_rate_limits: Dict[str, int],
    active_tickers: int,
    timeframes: List[TimeFrame],
    market_volatility: float = 1.0,
    priority_weights: Optional[Dict[TimeFrame, float]] = None,
) -> Dict[TimeFrame, int]:
    """
    Calculate optimal update intervals based on API limits, ticker count, and market conditions.

    This sophisticated algorithm considers:
    - Multiple API rate limits from different services
    - Market volatility for dynamic adjustment
    - Priority weights for different timeframes
    - Dynamic batch sizing with efficiency curves
    - Safety margins and burst allowances

    Args:
        api_rate_limits: API rate limits per service (requests per minute)
        active_tickers: Number of active tickers to track
        timeframes: List of timeframes to schedule
        market_volatility: Market volatility multiplier (1.0 = normal, >1.0 = high volatility)
        priority_weights: Optional priority weights for timeframes (higher = more frequent)

    Returns:
        Dictionary mapping timeframes to optimal intervals in seconds
    """
    # Base intervals in seconds (minimum update frequencies)
    base_intervals = {
        TimeFrame.FIVE_MINUTES: 300,
        TimeFrame.FIFTEEN_MINUTES: 900,
        TimeFrame.ONE_HOUR: 3600,
        TimeFrame.ONE_DAY: 86400,
    }

    # Default priority weights (higher values = more frequent updates)
    if priority_weights is None:
        priority_weights = {
            TimeFrame.FIVE_MINUTES: 4.0,  # Highest priority for short-term trading
            TimeFrame.FIFTEEN_MINUTES: 3.0,  # High priority for intraday
            TimeFrame.ONE_HOUR: 2.0,  # Medium priority for swing trading
            TimeFrame.ONE_DAY: 1.0,  # Lower priority for long-term analysis
        }

    # Calculate effective API rate limit (use the most restrictive)
    effective_rate_limit = min(api_rate_limits.values()) if api_rate_limits else 500

    # Apply safety margin (use 80% of rate limit to avoid hitting limits)
    safe_rate_limit = int(effective_rate_limit * 0.8)

    # Dynamic batch sizing with efficiency curves
    def calculate_optimal_batch_size(ticker_count: int) -> int:
        """Calculate optimal batch size based on ticker count and API efficiency."""
        if ticker_count <= 0:
            return 1  # Minimum batch size for edge cases
        elif ticker_count <= 10:
            return min(5, ticker_count)  # Small batches for few tickers
        elif ticker_count <= 50:
            return min(10, ticker_count)  # Medium batches
        elif ticker_count <= 200:
            return min(25, ticker_count)  # Large batches for efficiency
        else:
            return min(50, ticker_count)  # Max batch size for very large sets

    # Calculate sophisticated intervals
    optimal_intervals = {}

    for timeframe in timeframes:
        base_interval = base_intervals.get(timeframe, 300)
        priority_weight = priority_weights.get(timeframe, 1.0)

        # Calculate requests needed per update cycle
        batch_size = calculate_optimal_batch_size(active_tickers)
        if active_tickers <= 0:
            batches_needed = 1  # Handle zero tickers case
        else:
            batches_needed = max(
                1, (active_tickers + batch_size - 1) // batch_size
            )  # Ceiling division

        # Apply volatility adjustment (higher volatility = more frequent updates)
        volatility_multiplier = 1.0 / max(
            0.5, min(2.0, market_volatility)
        )  # Clamp between 0.5x and 2x

        # Apply priority weighting (higher priority = more frequent updates)
        priority_multiplier = 1.0 / max(
            0.5, min(3.0, priority_weight)
        )  # Clamp between 0.33x and 2x

        # Calculate minimum interval based on rate limits
        # Allow for burst capacity by spreading requests over a longer window
        requests_per_minute = batches_needed
        if safe_rate_limit <= 0:
            # Handle zero or negative rate limit edge case
            min_interval_from_rate_limit = base_interval * 10  # Conservative fallback
        elif requests_per_minute > safe_rate_limit:
            # Need to slow down to stay within rate limits
            minutes_needed = requests_per_minute / safe_rate_limit
            min_interval_from_rate_limit = int(minutes_needed * 60)
        else:
            # Can update more frequently, but respect minimum API call spacing
            min_interval_from_rate_limit = max(
                60, int(60 / max(1, safe_rate_limit) * requests_per_minute)
            )

        # Combine all factors
        calculated_interval = max(
            base_interval,  # Never go below the base interval for the timeframe
            min_interval_from_rate_limit,  # Respect rate limits
        )

        # Apply market condition adjustments
        calculated_interval = int(
            calculated_interval * volatility_multiplier * priority_multiplier
        )

        # Ensure we don't update too frequently (minimum 30 seconds)
        optimal_interval = max(30, calculated_interval)

        # Ensure we don't update too infrequently (maximum 24 hours)
        optimal_interval = min(86400, optimal_interval)

        # Final validation: ensure we never go below base interval
        optimal_interval = max(optimal_interval, base_interval)

        optimal_intervals[timeframe] = optimal_interval

    return optimal_intervals


# Example usage
if __name__ == "__main__":

    async def main():
        config = SchedulerConfig(
            market_open_time="09:30",
            market_close_time="16:00",
            timezone="America/New_York",
            max_workers=5,
        )

        scheduler = SchedulerService(config)

        # Example callback functions
        async def dummy_finviz_scan():
            logger.info("Running FinViz scan...")
            await asyncio.sleep(2)

        async def dummy_price_update():
            logger.info("Updating prices...")
            await asyncio.sleep(1)

        try:
            await scheduler.start()

            # Register tasks
            tasks = scheduler.create_data_collection_tasks()

            # Register callbacks
            callbacks = {
                "run_finviz_scan": dummy_finviz_scan,
                "update_5min_data": dummy_price_update,
                "update_15min_data": dummy_price_update,
                "update_hourly_data": dummy_price_update,
                "update_daily_data": dummy_price_update,
                "cleanup_old_data": lambda: logger.info("Cleaning up..."),
                "validate_data_integrity": lambda: logger.info("Validating..."),
                "optimize_storage": lambda: logger.info("Optimizing storage..."),
            }

            for task in tasks:
                if task.callback in callbacks:
                    await scheduler.register_task(task, callbacks[task.callback])

            # Run for a short time
            await asyncio.sleep(10)

            # Get stats
            stats = scheduler.get_scheduler_stats()
            print(f"Scheduler stats: {stats}")

            performance_report = await scheduler.get_task_performance_report()
            print(f"Performance report: {performance_report}")

        finally:
            await scheduler.stop()

    asyncio.run(main())
