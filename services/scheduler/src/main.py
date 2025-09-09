"""
Main entry point for the Trading Scheduler service.

This module provides the main application entry point, configuration setup,
logging initialization, and service coordination for the trading system scheduler.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import redis.asyncio as redis
import uvicorn
from pydantic import ValidationError

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from shared.config import Config, get_config  # noqa: E402
from shared.market_hours import MarketHoursService  # noqa: E402
from shared.utils import setup_logging  # noqa: E402

from .api import create_app  # noqa: E402
from .maintenance import MaintenanceManager, MaintenanceScheduler  # noqa: E402
from .monitor import SystemMonitor  # noqa: E402
from .orchestrator import (  # noqa: E402
    HealthCheck,
    ServiceConfiguration,
    ServiceDependency,
    ServiceOrchestrator,
)
from .scheduler import TradingScheduler  # noqa: E402

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SchedulerService:
    """Main scheduler service class."""

    def __init__(self, config: Config) -> None:
        logger.debug("Initializing SchedulerService instance")
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.scheduler: Optional[TradingScheduler] = None
        self.orchestrator: Optional[ServiceOrchestrator] = None
        self.market_hours: Optional[MarketHoursService] = None
        self.monitor: Optional[SystemMonitor] = None
        self.maintenance_manager: Optional[MaintenanceManager] = None
        self.maintenance_scheduler: Optional[MaintenanceScheduler] = None
        self.app: Optional[Any] = None

        # Service state
        self.is_running = False
        self.startup_complete = False

        # Signal handling
        self.shutdown_event = asyncio.Event()
        logger.debug("SchedulerService instance initialized")

    async def initialize(self) -> None:
        """Initialize all service components."""
        logger.info("Initializing scheduler service...")
        logger.debug(f"Configuration environment: {self.config.environment}")
        logger.debug(f"Redis URL: {self.config.redis.url}")
        logger.debug(f"Max Redis connections: {self.config.redis.max_connections}")

        try:
            # Initialize Redis connection
            logger.debug("Creating Redis connection")
            self.redis_client = redis.from_url(
                self.config.redis.url,
                max_connections=self.config.redis.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )

            # Test Redis connection
            logger.debug("Testing Redis connection with ping")
            await self.redis_client.ping()
            logger.info("Redis connection established")
            logger.debug("Redis connection test successful")

            # Initialize components
            logger.debug(
                f"Initializing MarketHoursService with timezone: {self.config.scheduler.timezone}"
            )
            self.market_hours = MarketHoursService(
                timezone_name=self.config.scheduler.timezone
            )
            logger.debug("MarketHoursService initialized")

            logger.debug("Initializing SystemMonitor")
            self.monitor = SystemMonitor(self.redis_client, self.config)
            logger.debug("SystemMonitor initialized")

            logger.debug("Initializing ServiceOrchestrator")
            self.orchestrator = ServiceOrchestrator(self.redis_client)
            logger.debug("ServiceOrchestrator initialized")

            logger.debug("Initializing TradingScheduler")
            self.scheduler = TradingScheduler(self.config)
            logger.debug("TradingScheduler initialized")

            # Initialize maintenance system
            logger.debug("Initializing MaintenanceManager")
            self.maintenance_manager = MaintenanceManager(
                self.config, self.redis_client
            )
            logger.debug("Registering maintenance tasks")
            await self.maintenance_manager.register_tasks()
            logger.debug("Maintenance tasks registered")

            logger.debug("Initializing MaintenanceScheduler")
            self.maintenance_scheduler = MaintenanceScheduler(self.maintenance_manager)
            await self.maintenance_scheduler.initialize()
            logger.debug("MaintenanceScheduler initialized")

            # Register services with orchestrator
            logger.debug("Registering services with orchestrator")
            await self._register_services()
            logger.debug("Services registered with orchestrator")

            # Initialize scheduler
            logger.debug("Initializing scheduler")
            await self.scheduler.initialize()
            logger.debug("Scheduler initialized")

            # Attach maintenance components to scheduler for API access
            logger.debug("Attaching maintenance components to scheduler")
            self.scheduler.maintenance_manager = self.maintenance_manager
            self.scheduler.maintenance_scheduler = self.maintenance_scheduler
            logger.debug("Maintenance components attached to scheduler")

            # Create FastAPI app
            logger.debug("Creating FastAPI app")
            self.app = create_app()
            logger.debug("FastAPI app created")

            logger.info("Scheduler service initialization completed")

        except Exception as e:
            logger.error(f"Failed to run scheduler service: {e}")
            raise

    async def _run_startup_maintenance_check(self) -> None:
        """Run maintenance check during startup to ensure system health."""
        try:
            logger.info("Running startup maintenance check...")
            logger.debug("Starting maintenance health checks")

            if self.maintenance_manager:
                # Run system health check first
                logger.debug("Running system health check task")
                health_result = await self.maintenance_manager.run_task(
                    "system_health_check"
                )
                logger.debug(
                    f"System health check result: success={health_result.success}"
                )

                if not health_result.success:
                    logger.warning("System health check failed during startup")
                    logger.debug(
                        f"Health check failure details: {health_result.details}"
                    )

                # Run intelligent maintenance analysis
                logger.debug("Running intelligent maintenance analysis")
                analysis_result = await self.maintenance_manager.run_task(
                    "intelligent_maintenance"
                )
                logger.debug(
                    f"Intelligent maintenance result: success={analysis_result.success}"
                )

                if analysis_result.success and analysis_result.details:
                    performance_score = analysis_result.details.get(
                        "performance_score", 100
                    )
                    logger.debug(f"System performance score: {performance_score}")

                    if performance_score < 70:
                        logger.warning(
                            f"Low performance score detected: {performance_score}"
                        )
                        logger.debug(
                            "Performance score below threshold, running smart maintenance"
                        )
                        # Run smart maintenance if system performance is poor
                        if self.maintenance_scheduler:
                            await self.maintenance_scheduler.run_smart_maintenance()
                            logger.debug("Smart maintenance completed")

                # Quick cache cleanup to start fresh
                logger.debug("Running cache cleanup task")
                await self.maintenance_manager.run_task("cache_cleanup")
                logger.debug("Cache cleanup completed")

                logger.info("Startup maintenance check completed")

        except Exception as e:
            logger.error(f"Startup maintenance check failed: {e}")
            logger.debug(
                f"Maintenance check exception details: {type(e).__name__}: {e}"
            )
            # Don't fail startup for maintenance issues

    async def _register_services(self) -> None:
        """Register all trading system services with the orchestrator."""
        logger.debug("Beginning service registration process")

        # Data Collector Service
        logger.debug("Creating data collector service configuration")
        data_collector = ServiceConfiguration(
            name="data_collector",
            url="http://trading_data_collector:9101",
            port=9101,
            health_check=HealthCheck(
                endpoint="/health", timeout=5.0, interval=30.0, failure_threshold=3
            ),
            dependencies=[],  # No dependencies
            startup_timeout=120.0,
            shutdown_timeout=30.0,
            restart_policy="on-failure",
            max_restarts=5,
        )
        assert self.orchestrator is not None
        self.orchestrator.register_service(data_collector)
        logger.debug("Data collector service registered")

        # Strategy Engine Service
        logger.debug("Creating strategy engine service configuration")
        strategy_engine = ServiceConfiguration(
            name="strategy_engine",
            url="http://trading_strategy_engine:9102",
            port=9102,
            health_check=HealthCheck(
                endpoint="/health", timeout=10.0, interval=30.0, failure_threshold=3
            ),
            dependencies=[
                ServiceDependency("data_collector", required=True, startup_delay=5.0)
            ],
            startup_timeout=180.0,
            shutdown_timeout=60.0,
            restart_policy="on-failure",
            max_restarts=3,
        )
        assert self.orchestrator is not None
        self.orchestrator.register_service(strategy_engine)
        logger.debug("Strategy engine service registered")

        # Risk Manager Service
        logger.debug("Creating risk manager service configuration")
        risk_manager = ServiceConfiguration(
            name="risk_manager",
            url="http://trading_risk_manager:9103",
            port=9103,
            health_check=HealthCheck(
                endpoint="/health", timeout=5.0, interval=20.0, failure_threshold=2
            ),
            dependencies=[
                ServiceDependency("data_collector", required=True, startup_delay=2.0),
                ServiceDependency("strategy_engine", required=True, startup_delay=3.0),
            ],
            startup_timeout=90.0,
            shutdown_timeout=30.0,
            restart_policy="on-failure",
            max_restarts=5,
        )
        assert self.orchestrator is not None
        self.orchestrator.register_service(risk_manager)
        logger.debug("Risk manager service registered")

        # Trade Executor Service
        logger.debug("Creating trade executor service configuration")
        trade_executor = ServiceConfiguration(
            name="trade_executor",
            url="http://trading_trade_executor:9104",
            port=9104,
            health_check=HealthCheck(
                endpoint="/health", timeout=5.0, interval=30.0, failure_threshold=2
            ),
            dependencies=[
                ServiceDependency("data_collector", required=True, startup_delay=2.0),
                ServiceDependency("strategy_engine", required=True, startup_delay=2.0),
                ServiceDependency("risk_manager", required=True, startup_delay=2.0),
            ],
            startup_timeout=120.0,
            shutdown_timeout=45.0,
            restart_policy="on-failure",
            max_restarts=3,
        )
        assert self.orchestrator is not None
        self.orchestrator.register_service(trade_executor)
        logger.debug("Trade executor service registered")

        logger.info("All services registered with orchestrator")
        logger.debug("Service registration process completed")

    async def start(self) -> None:
        """Start the scheduler service."""
        logger.info("Starting scheduler service...")
        logger.debug("Beginning service startup sequence")

        try:
            # Start market hours monitoring
            logger.debug("Starting market hours monitoring")
            assert self.market_hours is not None
            await self.market_hours.start_monitoring()
            logger.debug("Market hours monitoring started")

            # Start system monitoring
            logger.debug("Starting system monitoring")
            assert self.monitor is not None
            await self.monitor.start_monitoring()
            logger.debug("System monitoring started")

            # Start service orchestrator
            logger.debug("Starting service orchestrator")
            assert self.orchestrator is not None
            if not await self.orchestrator.start_all_services():
                logger.error("Service orchestrator failed to start all services")
                raise Exception("Failed to start all services")
            logger.debug("Service orchestrator started all services successfully")

            # Start the scheduler
            logger.debug("Starting the trading scheduler")
            assert self.scheduler is not None
            await self.scheduler.start()
            logger.debug("Trading scheduler started")

            # Run startup maintenance check
            logger.debug("Running startup maintenance check")
            await self._run_startup_maintenance_check()
            logger.debug("Startup maintenance check completed")

            # Mark as running
            self.is_running = True
            self.startup_complete = True
            logger.debug("Service marked as running and startup complete")

            logger.info("Scheduler service started successfully")

            # Register for market session changes
            logger.debug("Registering market session change callback")
            assert self.market_hours is not None
            self.market_hours.register_session_change_callback(
                self._on_market_session_change
            )
            logger.debug("Market session change callback registered")

            # Register for service status changes
            logger.debug("Registering service status change callback")
            assert self.orchestrator is not None
            self.orchestrator.register_status_change_callback(
                self._on_service_status_change
            )
            logger.debug("Service status change callback registered")

            # Register for system alerts
            logger.debug("Registering system alert callback")
            assert self.monitor is not None
            self.monitor.register_alert_callback(self._on_system_alert)
            logger.debug("System alert callback registered")

        except Exception as e:
            logger.error(f"Failed to start scheduler service: {e}")
            logger.debug(f"Startup failure exception details: {type(e).__name__}: {e}")
            await self.shutdown()
            raise

    async def _on_market_session_change(
        self, old_session: str, new_session: str
    ) -> None:
        """Handle market session changes."""
        logger.info(f"Market session changed: {old_session} -> {new_session}")
        logger.debug(
            f"Processing market session change from {old_session} to {new_session}"
        )

        # Trigger appropriate actions based on session change
        if new_session == "regular":
            # Market opened
            logger.debug("Market opened - processing regular session start")
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                "Market opened - regular trading session started", priority="medium"
            )
            logger.debug("Market open notification sent")

            # Resume market-hours tasks
            market_tasks = [
                "finviz_scan",
                "price_updates_1m",
                "price_updates_5m",
                "strategy_analysis",
            ]
            logger.debug(f"Resuming market hours tasks: {market_tasks}")
            for task_id in market_tasks:
                assert self.scheduler is not None
                await self.scheduler.resume_task(task_id)
                logger.debug(f"Resumed task: {task_id}")

        elif new_session == "closed" and old_session == "regular":
            # Market closed
            logger.debug("Market closed - processing regular session end")
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                "Market closed - regular trading session ended", priority="medium"
            )
            logger.debug("Market close notification sent")

            # Trigger EOD tasks
            logger.debug("Executing end-of-day report task")
            assert self.scheduler is not None
            await self.scheduler.execute_task("eod_report")
            logger.debug("EOD report task executed")

            # Run daily maintenance after market close
            if self.maintenance_scheduler:
                logger.debug("Running daily maintenance after market close")
                await self.maintenance_scheduler.run_scheduled_maintenance(
                    "daily_tradenote_export"
                )
                logger.debug("Daily maintenance completed")

    async def _on_service_status_change(
        self, service_name: str, old_status: str, new_status: str
    ) -> None:
        """Handle service status changes."""
        logger.info(
            f"Service {service_name} status changed: {old_status} -> {new_status}"
        )
        logger.debug(
            f"Processing service status change for {service_name}: {old_status} -> {new_status}"
        )

        if new_status == "error":
            logger.debug(
                f"Service {service_name} entered error state, sending high priority notification"
            )
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                f"Service {service_name} encountered an error", priority="high"
            )
            logger.debug(f"Error notification sent for service {service_name}")
        elif new_status == "running":
            logger.debug(
                f"Service {service_name} is now running, sending low priority notification"
            )
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                f"Service {service_name} is now running", priority="low"
            )
            logger.debug(f"Running notification sent for service {service_name}")

    async def _on_system_alert(self, alert: Any) -> None:
        """Handle system alerts."""
        logger.warning(f"System alert [{alert.severity.value}]: {alert.message}")
        logger.debug(
            f"Processing system alert with severity {alert.severity.value}: {alert.message}"
        )

        # Send notification for high and critical alerts
        if alert.severity.value in ["high", "critical"]:
            logger.debug(f"Alert severity {alert.severity.value} requires notification")
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                f"System Alert [{alert.severity.value.upper()}]: {alert.message}",
                priority=alert.severity.value,
            )
            logger.debug(
                f"System alert notification sent for severity {alert.severity.value}"
            )
        else:
            logger.debug(
                f"Alert severity {alert.severity.value} does not require notification"
            )

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        logger.debug("Setting up signal handlers")

        def signal_handler(signum: int, frame: any) -> None:
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            logger.debug(f"Signal handler triggered for signal {signum}")
            self.shutdown_event.set()

        # Register signal handlers
        logger.debug("Registering SIGINT and SIGTERM handlers")
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Handle SIGUSR1 for configuration reload
        def reload_handler(signum: int, frame: any) -> None:
            logger.info("Received SIGUSR1, reloading configuration...")
            logger.debug("SIGUSR1 signal handler triggered for configuration reload")
            asyncio.create_task(self._reload_configuration())

        logger.debug("Registering SIGUSR1 handler for configuration reload")
        signal.signal(signal.SIGUSR1, reload_handler)
        logger.debug("All signal handlers registered")

    async def _reload_configuration(self) -> None:
        """Reload configuration without restarting."""
        try:
            logger.info("Reloading configuration...")
            logger.debug("Starting hot reload of configuration")
            assert self.scheduler is not None
            await self.scheduler.hot_reload_config()
            logger.info("Configuration reloaded successfully")
            logger.debug("Configuration hot reload completed")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            logger.debug(
                f"Configuration reload exception details: {type(e).__name__}: {e}"
            )

    async def run_server(self) -> None:
        """Run the FastAPI server."""
        logger.debug("Starting FastAPI server")
        if not self.app:
            logger.error("Cannot start server - application not initialized")
            raise RuntimeError("Application not initialized")

        port = int(os.getenv("SERVICE_PORT", 8000))
        logger.debug(f"Server will run on port {port}")
        config_uvicorn = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=port,
            log_level=self.config.logging.level.lower(),
            access_log=True,
            loop="asyncio",
        )
        logger.debug(
            f"Uvicorn config: host=0.0.0.0, port={port}, log_level={self.config.logging.level.lower()}"
        )

        server = uvicorn.Server(config_uvicorn)

        # Start server in background
        logger.debug("Creating server task")
        server_task = asyncio.create_task(server.serve())
        logger.debug("Server task created, waiting for shutdown signal")

        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            logger.debug("Shutdown signal received")
        finally:
            # Graceful server shutdown
            logger.debug("Initiating graceful server shutdown")
            server.should_exit = True
            await server_task
            logger.debug("Server shutdown completed")

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        logger.debug("Starting comprehensive health check")
        try:
            health_status: Dict[str, Any] = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {},
            }
            logger.debug("Initialized health status structure")

            # Check Redis
            logger.debug("Checking Redis health")
            try:
                assert self.redis_client is not None
                await self.redis_client.ping()
                health_status["components"]["redis"] = "healthy"
                logger.debug("Redis health check: healthy")
            except Exception as e:
                health_status["components"]["redis"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
                logger.debug(f"Redis health check failed: {e}")

            # Check scheduler
            logger.debug("Checking scheduler health")
            if self.scheduler and self.scheduler.is_running:
                health_status["components"]["scheduler"] = "healthy"
                logger.debug("Scheduler health check: healthy")
            else:
                health_status["components"]["scheduler"] = "unhealthy"
                health_status["status"] = "degraded"
                logger.debug("Scheduler health check: unhealthy")

            # Check orchestrator
            logger.debug("Checking orchestrator health")
            if self.orchestrator and self.orchestrator.is_running:
                health_status["components"]["orchestrator"] = "healthy"
                logger.debug("Orchestrator health check: healthy")
            else:
                health_status["components"]["orchestrator"] = "unhealthy"
                health_status["status"] = "degraded"
                logger.debug("Orchestrator health check: unhealthy")

            # Check monitor
            logger.debug("Checking monitor health")
            if self.monitor and self.monitor.is_monitoring:
                health_status["components"]["monitor"] = "healthy"
                logger.debug("Monitor health check: healthy")
            else:
                health_status["components"]["monitor"] = "unhealthy"
                health_status["status"] = "degraded"
                logger.debug("Monitor health check: unhealthy")

            # Check market hours service
            logger.debug("Checking market hours service health")
            try:
                assert self.market_hours is not None
                session = await self.market_hours.get_current_session()
                health_status["components"]["market_hours"] = "healthy"
                health_status["market_session"] = session.value
                logger.debug(
                    f"Market hours health check: healthy, session={session.value}"
                )
            except Exception as e:
                health_status["components"]["market_hours"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
                logger.debug(f"Market hours health check failed: {e}")

            logger.debug(
                f"Health check completed with status: {health_status['status']}"
            )
            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            logger.debug(f"Health check exception details: {type(e).__name__}: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    async def shutdown(self) -> None:
        """Graceful shutdown of the scheduler service."""
        logger.info("Shutting down scheduler service...")
        logger.debug("Beginning graceful shutdown sequence")

        self.is_running = False
        logger.debug("Service marked as not running")

        try:
            # Stop scheduler
            if self.scheduler:
                logger.debug("Shutting down trading scheduler")
                await self.scheduler.shutdown()
                logger.debug("Trading scheduler shutdown completed")

            # Stop orchestrator
            if self.orchestrator:
                logger.debug("Shutting down service orchestrator")
                await self.orchestrator.shutdown()
                logger.debug("Service orchestrator shutdown completed")

            # Stop monitor
            if self.monitor:
                logger.debug("Shutting down system monitor")
                await self.monitor.shutdown()
                logger.debug("System monitor shutdown completed")

            # Stop market hours monitoring
            if self.market_hours:
                logger.debug("Shutting down market hours monitoring")
                await self.market_hours.shutdown()
                logger.debug("Market hours monitoring shutdown completed")

            # Shutdown maintenance system
            if self.maintenance_manager:
                logger.info("Shutting down maintenance manager...")
                logger.debug("Stopping maintenance manager operations")
                # Cancel any running maintenance tasks
                if self.maintenance_manager.current_task:
                    logger.info(
                        f"Cancelling running maintenance task: {self.maintenance_manager.current_task}"
                    )
                    logger.debug(
                        f"Cancelling current maintenance task: {self.maintenance_manager.current_task}"
                    )
                self.maintenance_manager.is_running = False
                logger.debug("Maintenance manager stopped")

            if self.maintenance_scheduler:
                logger.info("Shutting down maintenance scheduler...")
                logger.debug("Stopping maintenance scheduler")
                self.maintenance_scheduler.is_running = False
                logger.debug("Maintenance scheduler stopped")

            # Close Redis connection
            if self.redis_client:
                logger.debug("Closing Redis connection")
                await self.redis_client.close()
                logger.debug("Redis connection closed")

            logger.info("Scheduler service shutdown completed")
            logger.debug("Graceful shutdown sequence completed successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.debug(f"Shutdown exception details: {type(e).__name__}: {e}")

    async def run(self) -> None:
        """Main run method."""
        logger.debug("Starting main run method")
        try:
            # Setup signal handlers
            logger.debug("Setting up signal handlers")
            self.setup_signal_handlers()
            logger.debug("Signal handlers set up")

            # Initialize service
            logger.debug("Initializing service")
            await self.initialize()
            logger.debug("Service initialization completed")

            # Start the HTTP server first so health checks work
            logger.debug("Creating HTTP server task")
            server_task = asyncio.create_task(self.run_server())
            logger.info("HTTP server started")
            logger.debug("HTTP server task created")

            # Start all components in background
            logger.debug("Starting all components in background")
            asyncio.create_task(self.start())
            logger.debug("All components started")

            # Wait for the server
            logger.debug("Waiting for server task to complete")
            await server_task
            logger.debug("Server task completed")

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            logger.debug("KeyboardInterrupt caught in main run method")
        except Exception as e:
            logger.error(f"Service error: {e}")
            logger.debug(f"Service error exception details: {type(e).__name__}: {e}")
            raise
        finally:
            logger.debug("Entering finally block - shutting down")
            await self.shutdown()
            logger.debug("Main run method shutdown completed")


async def main() -> None:
    """Main entry point."""
    logger.debug("Entering main entry point")
    try:
        # Load configuration
        logger.debug("Loading configuration")
        config = get_config()
        logger.debug(
            f"Configuration loaded: environment={config.environment}, debug={config.debug}"
        )

        # Setup logging
        logger.debug("Setting up logging")
        setup_logging(config, "scheduler")
        logger.debug("Logging setup completed")

        logger.info("Starting Trading Scheduler Service...")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {config.debug}")

        # Validate critical configuration
        logger.debug("Validating critical configuration")
        if not config.redis.host:
            logger.error("Redis host configuration is missing")
            raise ValueError("Redis configuration is required")

        if not config.database.password:
            logger.error("Database password configuration is missing")
            raise ValueError("Database password is required")

        logger.debug("Configuration validation passed")

        # Create and run service
        logger.debug("Creating scheduler service instance")
        service = SchedulerService(config)
        logger.debug("Running scheduler service")
        await service.run()
        logger.debug("Scheduler service run completed")

    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        logger.debug(f"ValidationError details: {type(e).__name__}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        logger.debug(f"Service startup exception details: {type(e).__name__}: {e}")
        sys.exit(1)


def run_cli() -> None:
    """Entry point for CLI commands."""
    logger.debug("Entering CLI entry point")
    from .cli import app as cli_app

    logger.debug("Starting CLI app")
    cli_app()
    logger.debug("CLI app completed")


def run_dev_server() -> None:
    """Entry point for development server."""
    logger.debug("Entering development server entry point")
    config = get_config()
    logger.debug(f"Development server config loaded: level={config.logging.level}")

    # Setup basic logging for development
    logging.basicConfig(
        level=getattr(logging, config.logging.level), format=config.logging.format
    )
    logger.debug("Basic logging configured for development")

    # Run with uvicorn directly for development
    logger.debug("Starting uvicorn development server on port 8000")
    uvicorn.run(
        "main:create_dev_app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=config.logging.level.lower(),
    )
    logger.debug("Development server completed")


def create_dev_app() -> Any:
    """Create FastAPI app for development."""
    logger.debug("Creating FastAPI app for development")
    app = create_app()
    logger.debug("FastAPI app created for development")
    return app


class SchedulerApp:
    """Application wrapper for Scheduler service for integration testing."""

    def __init__(self) -> None:
        """Initialize the Scheduler application."""
        logger.debug("Initializing SchedulerApp wrapper")
        self.service: Optional[SchedulerService] = None
        self._initialized = False
        logger.debug("SchedulerApp wrapper initialized")

    async def initialize(self) -> None:
        """Initialize the application."""
        logger.debug("SchedulerApp initialize called")
        if not self._initialized:
            logger.debug("SchedulerApp not yet initialized, starting initialization")
            config = get_config()
            logger.debug("Configuration loaded for SchedulerApp")
            self.service = SchedulerService(config)
            logger.debug("SchedulerService instance created")
            await self.service.initialize()
            logger.debug("SchedulerService initialized")
            self._initialized = True
            logger.debug("SchedulerApp initialization completed")
        else:
            logger.debug("SchedulerApp already initialized, skipping")

    async def start(self) -> None:
        """Start the Scheduler service."""
        logger.debug("SchedulerApp start called")
        await self.initialize()
        logger.debug("SchedulerApp initialization completed")
        if self.service:
            logger.debug("Creating task to run scheduler service")
            asyncio.create_task(self.service.run())
            logger.debug("Scheduler service task created")
        else:
            logger.debug("No scheduler service available to start")

    async def stop(self) -> None:
        """Stop the Scheduler service."""
        logger.debug("SchedulerApp stop called")
        if self.service:
            logger.debug("Shutting down scheduler service")
            await self.service.shutdown()
            logger.debug("Scheduler service shutdown completed")
        else:
            logger.debug("No scheduler service to stop")
        self._initialized = False
        logger.debug("SchedulerApp marked as not initialized")

    def get_service(self) -> Optional[SchedulerService]:
        """Get the underlying service instance."""
        logger.debug(
            f"SchedulerApp get_service called, returning: {self.service is not None}"
        )
        return self.service


if __name__ == "__main__":
    # Check if running as CLI
    if len(sys.argv) > 1 and sys.argv[1] in [
        "status",
        "tasks",
        "trigger",
        "pause",
        "resume",
        "services",
        "maintenance",
        "emergency",
        "pipeline",
        "metrics",
        "logs",
        "positions",
        "portfolio",
        "config",
        "monitor",
        "strategy",
        "risk",
        "backtest",
        "alerts",
        "health",
        "queue",
        "trade",
        "export",
        "version",
    ]:
        logger.debug(f"Running as CLI with command: {sys.argv[1]}")
        run_cli()
    else:
        # Run main scheduler service
        logger.debug("Running as main scheduler service")
        try:
            logger.debug("Starting asyncio event loop")
            asyncio.run(main())
            logger.debug("Asyncio event loop completed")
        except KeyboardInterrupt:
            logger.info("Scheduler service stopped by user")
            logger.debug("KeyboardInterrupt caught in main block")
        except Exception as e:
            logger.error(f"Scheduler service failed: {e}")
            logger.debug(f"Main block exception details: {type(e).__name__}: {e}")
            sys.exit(1)
