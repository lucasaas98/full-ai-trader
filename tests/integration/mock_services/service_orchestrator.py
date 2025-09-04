"""
Service Orchestrator for Integration Tests

This module orchestrates the execution of real trading system services
in separate threads/tasks for integration testing. It manages the lifecycle
of services and provides a unified interface for testing.
"""

import asyncio
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from services.risk_manager.src.main import RiskManagerApp  # noqa: E402
from services.scheduler.src.main import SchedulerApp  # noqa: E402

# Import service modules
from services.strategy_engine.src.main import StrategyEngineApp  # noqa: E402
from services.trade_executor.src.main import TradeExecutorApp  # noqa: E402
from shared.config import Config, get_config  # noqa: E402

from .mock_data_collector import (  # noqa: E402
    MockDataCollector,
    MockDataCollectorConfig,
)

logger = logging.getLogger(__name__)


class ServiceStatus:
    """Represents the status of a service."""

    def __init__(self, name: str):
        self.name = name
        self.status = "stopped"
        self.started_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.health_data: Dict[str, Any] = {}

    def start(self):
        self.status = "running"
        self.started_at = datetime.now(timezone.utc)
        self.error = None

    def stop(self):
        self.status = "stopped"
        self.error = None

    def set_error(self, error: str):
        self.status = "error"
        self.error = error

    def update_health(self, health_data: Dict[str, Any]):
        self.health_data = health_data


class ServiceOrchestrator:
    """
    Orchestrates multiple trading system services for integration testing.

    This class:
    1. Manages service lifecycle (start, stop, health checks)
    2. Runs services in separate threads/processes
    3. Provides unified logging and monitoring
    4. Handles graceful shutdown
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.services: Dict[str, ServiceStatus] = {}
        self.service_tasks: Dict[str, asyncio.Task] = {}
        self.service_threads: Dict[str, threading.Thread] = {}
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=6)

        # Service instances
        self.mock_data_collector: Optional[MockDataCollector] = None
        self.strategy_engine: Optional[StrategyEngineApp] = None
        self.risk_manager: Optional[RiskManagerApp] = None
        self.trade_executor: Optional[TradeExecutorApp] = None
        self.scheduler: Optional[SchedulerApp] = None

        # Initialize service status tracking
        self._init_service_status()

    def _init_service_status(self):
        """Initialize service status tracking."""
        service_names = [
            "mock_data_collector",
            "strategy_engine",
            "risk_manager",
            "trade_executor",
            "scheduler",
        ]

        for name in service_names:
            self.services[name] = ServiceStatus(name)

    async def start_all_services(self):
        """Start all services in the correct order."""
        logger.info("🚀 Starting all services for integration testing...")

        self.is_running = True

        try:
            # Start services in dependency order
            await self._start_mock_data_collector()
            await asyncio.sleep(2)  # Let data collector initialize

            await self._start_strategy_engine()
            await asyncio.sleep(1)

            await self._start_risk_manager()
            await asyncio.sleep(1)

            await self._start_trade_executor()
            await asyncio.sleep(1)

            await self._start_scheduler()

            # Start health monitoring
            self.service_tasks["health_monitor"] = asyncio.create_task(
                self._health_monitor_loop()
            )

            logger.info("✅ All services started successfully")

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            await self.stop_all_services()
            raise

    async def stop_all_services(self):
        """Stop all services gracefully."""
        logger.info("🛑 Stopping all services...")

        self.is_running = False
        self._shutdown_event.set()

        # Cancel all service tasks
        for name, task in self.service_tasks.items():
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop services in reverse order
        services_to_stop = [
            ("scheduler", self.scheduler),
            ("trade_executor", self.trade_executor),
            ("risk_manager", self.risk_manager),
            ("strategy_engine", self.strategy_engine),
            ("mock_data_collector", self.mock_data_collector),
        ]

        for service_name, service_instance in services_to_stop:
            if service_instance:
                try:
                    await self._stop_service(service_name, service_instance)
                except Exception as e:
                    logger.error(f"Error stopping {service_name}: {e}")

        # Close executor
        self._executor.shutdown(wait=False)

        logger.info("✅ All services stopped")

    async def _start_mock_data_collector(self):
        """Start the mock data collector."""
        logger.info("Starting Mock Data Collector...")

        try:
            config = MockDataCollectorConfig(
                historical_data_path=self.config.data.base_path or "data/parquet",
                available_symbols=["AAPL", "SPY", "QQQ", "MSFT", "TSLA"],
                redis_publish_interval=30,
                simulate_screener=True,
                screener_interval=300,
            )

            self.mock_data_collector = MockDataCollector(config)
            await self.mock_data_collector.start()

            self.services["mock_data_collector"].start()
            logger.info("✅ Mock Data Collector started")

        except Exception as e:
            self.services["mock_data_collector"].set_error(str(e))
            logger.error(f"Failed to start Mock Data Collector: {e}")
            raise

    async def _start_strategy_engine(self):
        """Start the strategy engine."""
        logger.info("Starting Strategy Engine...")

        try:
            self.strategy_engine = StrategyEngineApp()

            # Run in thread to avoid blocking
            def run_strategy_engine():
                asyncio.run(self.strategy_engine.run())

            thread = threading.Thread(target=run_strategy_engine, daemon=True)
            thread.start()
            self.service_threads["strategy_engine"] = thread

            # Give it time to start
            await asyncio.sleep(2)

            self.services["strategy_engine"].start()
            logger.info("✅ Strategy Engine started")

        except Exception as e:
            self.services["strategy_engine"].set_error(str(e))
            logger.error(f"Failed to start Strategy Engine: {e}")
            raise

    async def _start_risk_manager(self):
        """Start the risk manager."""
        logger.info("Starting Risk Manager...")

        try:
            self.risk_manager = RiskManagerApp()

            def run_risk_manager():
                asyncio.run(self.risk_manager.run())

            thread = threading.Thread(target=run_risk_manager, daemon=True)
            thread.start()
            self.service_threads["risk_manager"] = thread

            await asyncio.sleep(2)

            self.services["risk_manager"].start()
            logger.info("✅ Risk Manager started")

        except Exception as e:
            self.services["risk_manager"].set_error(str(e))
            logger.error(f"Failed to start Risk Manager: {e}")
            raise

    async def _start_trade_executor(self):
        """Start the trade executor."""
        logger.info("Starting Trade Executor...")

        try:
            self.trade_executor = TradeExecutorApp()

            def run_trade_executor():
                asyncio.run(self.trade_executor.run())

            thread = threading.Thread(target=run_trade_executor, daemon=True)
            thread.start()
            self.service_threads["trade_executor"] = thread

            await asyncio.sleep(2)

            self.services["trade_executor"].start()
            logger.info("✅ Trade Executor started")

        except Exception as e:
            self.services["trade_executor"].set_error(str(e))
            logger.error(f"Failed to start Trade Executor: {e}")
            raise

    async def _start_scheduler(self):
        """Start the scheduler."""
        logger.info("Starting Scheduler...")

        try:
            self.scheduler = SchedulerApp()

            def run_scheduler():
                asyncio.run(self.scheduler.run())

            thread = threading.Thread(target=run_scheduler, daemon=True)
            thread.start()
            self.service_threads["scheduler"] = thread

            await asyncio.sleep(2)

            self.services["scheduler"].start()
            logger.info("✅ Scheduler started")

        except Exception as e:
            self.services["scheduler"].set_error(str(e))
            logger.error(f"Failed to start Scheduler: {e}")
            raise

    async def _stop_service(self, service_name: str, service_instance: Any):
        """Stop a specific service."""
        logger.info(f"Stopping {service_name}...")

        try:
            if hasattr(service_instance, "stop") and callable(service_instance.stop):
                await service_instance.stop()
            elif hasattr(service_instance, "shutdown") and callable(
                service_instance.shutdown
            ):
                await service_instance.shutdown()

            # Stop associated thread
            if service_name in self.service_threads:
                thread = self.service_threads[service_name]
                if thread.is_alive():
                    # Give thread time to stop gracefully
                    thread.join(timeout=5)

            self.services[service_name].stop()
            logger.info(f"✅ {service_name} stopped")

        except Exception as e:
            logger.error(f"Error stopping {service_name}: {e}")
            self.services[service_name].set_error(str(e))

    async def _health_monitor_loop(self):
        """Monitor service health continuously."""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                await self._check_all_service_health()
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)

    async def _check_all_service_health(self):
        """Check health of all services."""
        for service_name in self.services:
            try:
                health_data = await self._get_service_health(service_name)
                self.services[service_name].update_health(health_data)

            except Exception as e:
                logger.warning(f"Health check failed for {service_name}: {e}")

    async def _get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health data for a specific service."""
        service_map = {
            "mock_data_collector": self.mock_data_collector,
            "strategy_engine": self.strategy_engine,
            "risk_manager": self.risk_manager,
            "trade_executor": self.trade_executor,
            "scheduler": self.scheduler,
        }

        service_instance = service_map.get(service_name)

        if not service_instance:
            return {"status": "not_started", "error": "Service instance not found"}

        try:
            if hasattr(service_instance, "health_check") and callable(
                service_instance.health_check
            ):
                return await service_instance.health_check()
            else:
                # Basic health check based on thread status
                thread = self.service_threads.get(service_name)
                if thread and thread.is_alive():
                    return {"status": "running", "thread_alive": True}
                else:
                    return {"status": "unknown", "thread_alive": False}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        status = {
            "orchestrator_running": self.is_running,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {},
        }

        for service_name, service_status in self.services.items():
            status["services"][service_name] = {
                "status": service_status.status,
                "started_at": (
                    service_status.started_at.isoformat()
                    if service_status.started_at
                    else None
                ),
                "error": service_status.error,
                "health": service_status.health_data,
            }

        return status

    async def wait_for_services_ready(self, timeout: int = 60) -> bool:
        """Wait for all services to be ready."""
        logger.info("⏳ Waiting for services to be ready...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await self.get_service_status()

            ready_count = 0
            total_count = len(self.services)

            for service_name, service_info in status["services"].items():
                if service_info["status"] == "running":
                    ready_count += 1
                elif service_info["status"] == "error":
                    logger.error(
                        f"Service {service_name} has error: {service_info['error']}"
                    )
                    return False

            logger.info(f"Services ready: {ready_count}/{total_count}")

            if ready_count == total_count:
                logger.info("✅ All services are ready!")
                return True

            await asyncio.sleep(2)

        logger.error(f"⏰ Timeout waiting for services to be ready after {timeout}s")
        return False

    async def simulate_trading_day(self, duration_minutes: int = 30):
        """Simulate a trading day by advancing time and triggering events."""
        logger.info(
            f"🎭 Starting trading day simulation for {duration_minutes} minutes..."
        )

        if not self.mock_data_collector:
            raise ValueError("Mock data collector not started")

        try:
            # Reset simulation to market open

            today = date.today()
            await self.mock_data_collector.reset_simulation_date(today)

            # Run simulation
            end_time = time.time() + (duration_minutes * 60)

            while time.time() < end_time and self.is_running:
                # Trigger screener update
                await self.mock_data_collector.force_screener_update()

                # Wait before next cycle
                await asyncio.sleep(30)

            logger.info("✅ Trading day simulation completed")

        except Exception as e:
            logger.error(f"Error in trading day simulation: {e}")
            raise


async def create_service_orchestrator(
    config: Optional[Config] = None,
) -> ServiceOrchestrator:
    """Factory function to create a service orchestrator."""
    orchestrator = ServiceOrchestrator(config)
    return orchestrator


if __name__ == "__main__":

    async def main():
        orchestrator = await create_service_orchestrator()

        try:
            await orchestrator.start_all_services()

            if await orchestrator.wait_for_services_ready():
                logger.info("Running integration test simulation...")
                await orchestrator.simulate_trading_day(5)

                status = await orchestrator.get_service_status()
                print("\n" + "=" * 50)
                print("INTEGRATION TEST COMPLETED")
                print("=" * 50)
                for service_name, service_info in status["services"].items():
                    print(f"{service_name}: {service_info['status']}")
                print("=" * 50)

            else:
                logger.error("Services failed to start properly")

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
        finally:
            await orchestrator.stop_all_services()

    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
