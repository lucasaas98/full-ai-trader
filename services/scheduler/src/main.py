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

logger = logging.getLogger(__name__)


class SchedulerService:
    """Main scheduler service class."""

    def __init__(self, config: Config):
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

    async def initialize(self):
        """Initialize all service components."""
        logger.info("Initializing scheduler service...")

        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(
                self.config.redis.url,
                max_connections=self.config.redis.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )

            # Test Redis connection
            await self.redis_client.ping()
            logger.info("Redis connection established")

            # Initialize components
            self.market_hours = MarketHoursService(
                timezone_name=self.config.scheduler.timezone
            )
            self.monitor = SystemMonitor(self.redis_client, self.config)
            self.orchestrator = ServiceOrchestrator(self.redis_client)
            self.scheduler = TradingScheduler(self.config)

            # Initialize maintenance system
            self.maintenance_manager = MaintenanceManager(
                self.config, self.redis_client
            )
            await self.maintenance_manager.register_tasks()

            self.maintenance_scheduler = MaintenanceScheduler(self.maintenance_manager)
            await self.maintenance_scheduler.initialize()

            # Register services with orchestrator
            await self._register_services()

            # Initialize scheduler
            await self.scheduler.initialize()

            # Attach maintenance components to scheduler for API access
            self.scheduler.maintenance_manager = self.maintenance_manager
            self.scheduler.maintenance_scheduler = self.maintenance_scheduler

            # Create FastAPI app
            self.app = create_app()

            logger.info("Scheduler service initialization completed")

        except Exception as e:
            logger.error(f"Failed to run scheduler service: {e}")
            raise

    async def _run_startup_maintenance_check(self):
        """Run maintenance check during startup to ensure system health."""
        try:
            logger.info("Running startup maintenance check...")

            if self.maintenance_manager:
                # Run system health check first
                health_result = await self.maintenance_manager.run_task(
                    "system_health_check"
                )

                if not health_result.success:
                    logger.warning("System health check failed during startup")

                # Run intelligent maintenance analysis
                analysis_result = await self.maintenance_manager.run_task(
                    "intelligent_maintenance"
                )

                if analysis_result.success and analysis_result.details:
                    performance_score = analysis_result.details.get(
                        "performance_score", 100
                    )

                    if performance_score < 70:
                        logger.warning(
                            f"Low performance score detected: {performance_score}"
                        )
                        # Run smart maintenance if system performance is poor
                        if self.maintenance_scheduler:
                            await self.maintenance_scheduler.run_smart_maintenance()

                # Quick cache cleanup to start fresh
                await self.maintenance_manager.run_task("cache_cleanup")

                logger.info("Startup maintenance check completed")

        except Exception as e:
            logger.error(f"Startup maintenance check failed: {e}")
            # Don't fail startup for maintenance issues

    async def _register_services(self):
        """Register all trading system services with the orchestrator."""

        # Data Collector Service
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

        # Strategy Engine Service
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

        # Risk Manager Service
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

        # Trade Executor Service
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

        logger.info("All services registered with orchestrator")

    async def start(self):
        """Start the scheduler service."""
        logger.info("Starting scheduler service...")

        try:
            # Start market hours monitoring
            assert self.market_hours is not None
            await self.market_hours.start_monitoring()

            # Start system monitoring
            assert self.monitor is not None
            await self.monitor.start_monitoring()

            # Start service orchestrator
            assert self.orchestrator is not None
            if not await self.orchestrator.start_all_services():
                raise Exception("Failed to start all services")

            # Start the scheduler
            assert self.scheduler is not None
            await self.scheduler.start()

            # Run startup maintenance check
            await self._run_startup_maintenance_check()

            # Mark as running
            self.is_running = True
            self.startup_complete = True

            logger.info("Scheduler service started successfully")

            # Register for market session changes
            assert self.market_hours is not None
            self.market_hours.register_session_change_callback(
                self._on_market_session_change
            )

            # Register for service status changes
            assert self.orchestrator is not None
            self.orchestrator.register_status_change_callback(
                self._on_service_status_change
            )

            # Register for system alerts
            assert self.monitor is not None
            self.monitor.register_alert_callback(self._on_system_alert)

        except Exception as e:
            logger.error(f"Failed to start scheduler service: {e}")
            await self.shutdown()
            raise

    async def _on_market_session_change(self, old_session, new_session):
        """Handle market session changes."""
        logger.info(f"Market session changed: {old_session} -> {new_session}")

        # Trigger appropriate actions based on session change
        if new_session == "regular":
            # Market opened
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                "Market opened - regular trading session started", priority="medium"
            )
            # Resume market-hours tasks
            market_tasks = [
                "finviz_scan",
                "price_updates_1m",
                "price_updates_5m",
                "strategy_analysis",
            ]
            for task_id in market_tasks:
                assert self.scheduler is not None
                await self.scheduler.resume_task(task_id)

        elif new_session == "closed" and old_session == "regular":
            # Market closed
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                "Market closed - regular trading session ended", priority="medium"
            )
            # Trigger EOD tasks
            assert self.scheduler is not None
            await self.scheduler.execute_task("eod_report")
            # Run daily maintenance after market close
            if self.maintenance_scheduler:
                await self.maintenance_scheduler.run_scheduled_maintenance(
                    "daily_tradenote_export"
                )

    async def _on_service_status_change(
        self, service_name: str, old_status, new_status
    ):
        """Handle service status changes."""
        logger.info(
            f"Service {service_name} status changed: {old_status} -> {new_status}"
        )

        if new_status == "error":
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                f"Service {service_name} encountered an error", priority="high"
            )
        elif new_status == "running":
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                f"Service {service_name} is now running", priority="low"
            )

    async def _on_system_alert(self, alert):
        """Handle system alerts."""
        logger.warning(f"System alert [{alert.severity.value}]: {alert.message}")

        # Send notification for high and critical alerts
        if alert.severity.value in ["high", "critical"]:
            assert self.scheduler is not None
            await self.scheduler.send_notification(
                f"System Alert [{alert.severity.value.upper()}]: {alert.message}",
                priority=alert.severity.value,
            )

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_event.set()

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Handle SIGUSR1 for configuration reload
        def reload_handler(signum, frame):
            logger.info("Received SIGUSR1, reloading configuration...")
            asyncio.create_task(self._reload_configuration())

        signal.signal(signal.SIGUSR1, reload_handler)

    async def _reload_configuration(self):
        """Reload configuration without restarting."""
        try:
            logger.info("Reloading configuration...")
            assert self.scheduler is not None
            await self.scheduler.hot_reload_config()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")

    async def run_server(self):
        """Run the FastAPI server."""
        if not self.app:
            raise RuntimeError("Application not initialized")

        port = int(os.getenv("SERVICE_PORT", 8000))
        config_uvicorn = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=port,
            log_level=self.config.logging.level.lower(),
            access_log=True,
            loop="asyncio",
        )

        server = uvicorn.Server(config_uvicorn)

        # Start server in background
        server_task = asyncio.create_task(server.serve())

        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
        finally:
            # Graceful server shutdown
            server.should_exit = True
            await server_task

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {},
            }

            # Check Redis
            try:
                assert self.redis_client is not None
                await self.redis_client.ping()
                health_status["components"]["redis"] = "healthy"
            except Exception as e:
                health_status["components"]["redis"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"

            # Check scheduler
            if self.scheduler and self.scheduler.is_running:
                health_status["components"]["scheduler"] = "healthy"
            else:
                health_status["components"]["scheduler"] = "unhealthy"
                health_status["status"] = "degraded"

            # Check orchestrator
            if self.orchestrator and self.orchestrator.is_running:
                health_status["components"]["orchestrator"] = "healthy"
            else:
                health_status["components"]["orchestrator"] = "unhealthy"
                health_status["status"] = "degraded"

            # Check monitor
            if self.monitor and self.monitor.is_monitoring:
                health_status["components"]["monitor"] = "healthy"
            else:
                health_status["components"]["monitor"] = "unhealthy"
                health_status["status"] = "degraded"

            # Check market hours service
            try:
                assert self.market_hours is not None
                session = await self.market_hours.get_current_session()
                health_status["components"]["market_hours"] = "healthy"
                health_status["market_session"] = session.value
            except Exception as e:
                health_status["components"]["market_hours"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    async def shutdown(self):
        """Graceful shutdown of the scheduler service."""
        logger.info("Shutting down scheduler service...")

        self.is_running = False

        try:
            # Stop scheduler
            if self.scheduler:
                await self.scheduler.shutdown()

            # Stop orchestrator
            if self.orchestrator:
                await self.orchestrator.shutdown()

            # Stop monitor
            if self.monitor:
                await self.monitor.shutdown()

            # Stop market hours monitoring
            if self.market_hours:
                await self.market_hours.shutdown()

            # Shutdown maintenance system
            if self.maintenance_manager:
                logger.info("Shutting down maintenance manager...")
                # Cancel any running maintenance tasks
                if self.maintenance_manager.current_task:
                    logger.info(
                        f"Cancelling running maintenance task: {self.maintenance_manager.current_task}"
                    )
                self.maintenance_manager.is_running = False

            if self.maintenance_scheduler:
                logger.info("Shutting down maintenance scheduler...")
                self.maintenance_scheduler.is_running = False

            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()

            logger.info("Scheduler service shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def run(self):
        """Main run method."""
        try:
            # Setup signal handlers
            self.setup_signal_handlers()

            # Initialize service
            await self.initialize()

            # Start the HTTP server first so health checks work
            server_task = asyncio.create_task(self.run_server())
            logger.info("HTTP server started")

            # Start all components in background
            asyncio.create_task(self.start())

            # Wait for the server
            await server_task

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Service error: {e}")
            raise
        finally:
            await self.shutdown()


async def main():
    """Main entry point."""
    try:
        # Load configuration
        config = get_config()

        # Setup logging
        setup_logging(config, "scheduler")

        logger.info("Starting Trading Scheduler Service...")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Debug mode: {config.debug}")

        # Validate critical configuration
        if not config.redis.host:
            raise ValueError("Redis configuration is required")

        if not config.database.password:
            raise ValueError("Database password is required")

        # Create and run service
        service = SchedulerService(config)
        await service.run()

    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        sys.exit(1)


def run_cli():
    """Entry point for CLI commands."""
    from .cli import app as cli_app

    cli_app()


def run_dev_server():
    """Entry point for development server."""
    config = get_config()

    # Setup basic logging for development
    logging.basicConfig(
        level=getattr(logging, config.logging.level), format=config.logging.format
    )

    # Run with uvicorn directly for development
    uvicorn.run(
        "main:create_dev_app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=config.logging.level.lower(),
    )


def create_dev_app():
    """Create FastAPI app for development."""
    return create_app()


class SchedulerApp:
    """Application wrapper for Scheduler service for integration testing."""

    def __init__(self):
        """Initialize the Scheduler application."""
        self.service = None
        self._initialized = False

    async def initialize(self):
        """Initialize the application."""
        if not self._initialized:
            config = get_config()
            self.service = SchedulerService(config)
            await self.service.initialize()
            self._initialized = True

    async def start(self):
        """Start the Scheduler service."""
        await self.initialize()
        asyncio.create_task(self.service.run())

    async def stop(self):
        """Stop the Scheduler service."""
        if self.service:
            await self.service.shutdown()
        self._initialized = False

    def get_service(self):
        """Get the underlying service instance."""
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
        run_cli()
    else:
        # Run main scheduler service
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Scheduler service stopped by user")
        except Exception as e:
            logger.error(f"Scheduler service failed: {e}")
            sys.exit(1)
