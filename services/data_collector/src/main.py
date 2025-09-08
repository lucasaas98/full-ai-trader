"""
Main application entry point for the data collection service.

This module provides the main application that orchestrates all components
of the data collection system including FinViz screener, TwelveData API,
data storage, Redis integration, and intelligent scheduling.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
# Add project root to path for shared imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from datetime import date, datetime, timedelta, timezone  # noqa: E402
from typing import Any, Dict, Optional  # noqa: E402

from pydantic import ValidationError  # noqa: E402

from shared.config import get_config  # noqa: E402
from shared.models import TimeFrame  # noqa: E402

from .data_collection_service import (  # noqa: E402
    DataCollectionConfig,
    DataCollectionService,
)
from .data_store import DataStore, DataStoreConfig  # noqa: E402
from .http_server import DataCollectorHTTPServer  # noqa: E402
from .scheduler_service import SchedulerService  # noqa: E402


# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_config = get_config().logging

    # Create logs directory if it doesn't exist
    log_path = Path(log_config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_config.format,
        handlers=[
            h
            for h in [
                (
                    logging.StreamHandler(sys.stdout)
                    if log_config.enable_console
                    else None
                ),
                (
                    logging.FileHandler(log_config.file_path)
                    if log_config.enable_file
                    else None
                ),
            ]
            if h is not None
        ],
    )

    return logging.getLogger(__name__)


class DataCollectorApp:
    """Main application class for data collection service."""

    def __init__(self):
        self.logger = setup_logging()
        self.logger.debug("DataCollectorApp.__init__: Initializing DataCollectorApp")
        self.config = self._load_configuration()
        self.logger.debug(
            f"DataCollectorApp.__init__: Configuration loaded with {len(self.config.model_dump())} parameters"
        )
        self.data_service: Optional[DataCollectionService] = None
        self.scheduler_service: Optional[SchedulerService] = None
        self.http_server: Optional[DataCollectorHTTPServer] = None
        self._shutdown_event = asyncio.Event()
        self.logger.debug("DataCollectorApp.__init__: Initialization complete")

    def _load_configuration(self) -> DataCollectionConfig:
        """Load and validate configuration."""
        self.logger.debug("_load_configuration: Starting configuration loading")
        try:
            # Get base config
            self.logger.debug("_load_configuration: Getting base configuration")
            base_config = get_config()
            self.logger.debug(
                f"_load_configuration: Base config loaded - Redis enabled: {hasattr(base_config, 'redis')}, TwelveData key present: {bool(getattr(base_config.twelvedata, 'api_key', None))}"
            )

            # Create data collection config
            config = DataCollectionConfig(
                service_name="data_collector",
                enable_finviz=True,
                enable_twelvedata=bool(base_config.twelvedata.api_key),
                enable_redis=True,
                # Scheduling intervals from base config
                finviz_scan_interval=base_config.scheduler.finviz_scan_interval,
                price_update_interval_5m=300,  # 5 minutes
                price_update_interval_15m=900,  # 15 minutes
                price_update_interval_1h=3600,  # 1 hour
                price_update_interval_1d=86400,  # 1 day
                # Data settings
                max_active_tickers=50,
                historical_data_years=2,
                screener_result_limit=20,
                # Market hours
                market_open_time=base_config.scheduler.trading_start_time,
                market_close_time=base_config.scheduler.trading_end_time,
                timezone=base_config.scheduler.timezone,
                # Performance
                max_retries=3,
                retry_delay=5.0,
                concurrent_downloads=10,
                batch_size=20,
            )

            self.logger.debug(
                f"_load_configuration: DataCollectionConfig created successfully with {config.max_active_tickers} max tickers"
            )
            return config

        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            self.logger.debug(
                f"_load_configuration: Validation error details: {str(e)}"
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.logger.debug(f"_load_configuration: Unexpected error: {str(e)}")
            raise

    async def start(self):
        """Start the data collection application."""
        self.logger.info("Starting Data Collection Service...")
        self.logger.debug("start: Beginning service startup sequence")

        try:
            # Initialize and start data collection service
            self.logger.debug("start: Creating DataCollectionService instance")
            self.data_service = DataCollectionService(self.config)
            self.logger.debug("start: Starting data collection service")
            await self.data_service.start()
            self.logger.debug("start: Data collection service started successfully")

            # Initialize and start HTTP server
            import os

            port = int(os.environ.get("SERVICE_PORT", 9101))
            self.logger.debug(f"start: Initializing HTTP server on port {port}")
            self.http_server = DataCollectorHTTPServer(
                data_service=self.data_service, port=port, host="0.0.0.0"
            )
            self.logger.debug("start: Starting HTTP server")
            await self.http_server.start()
            self.logger.debug("start: HTTP server started successfully")

            # Set up signal handlers for graceful shutdown
            self.logger.debug("start: Setting up signal handlers")
            self._setup_signal_handlers()

            self.logger.info("Data Collection Service started successfully")
            self.logger.info(f"HTTP server running on port {port}")
            self.logger.info(f"Service configuration: {self.config.model_dump()}")

            # Log service status
            self.logger.debug("start: Getting service status for logging")
            status = await self.data_service.get_service_status()
            self.logger.info(f"Active tickers: {status['active_tickers_count']}")
            self.logger.info(f"Scheduled jobs: {len(status['scheduler_jobs'])}")
            self.logger.debug(f"start: Service status details: {status}")

            self.logger.debug("start: Application startup completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            self.logger.debug(f"start: Startup failed with exception: {str(e)}")
            raise

    async def stop(self):
        """Stop the data collection application."""
        self.logger.info("Stopping Data Collection Service...")
        self.logger.debug("stop: Beginning shutdown sequence")

        try:
            # Signal shutdown
            self.logger.debug("stop: Setting shutdown event")
            self._shutdown_event.set()

            # Stop HTTP server
            if self.http_server:
                self.logger.debug("stop: Stopping HTTP server")
                await self.http_server.stop()
                self.logger.debug("stop: HTTP server stopped")

            # Stop services
            if self.data_service:
                self.logger.debug("stop: Stopping data collection service")
                await self.data_service.stop()
                self.logger.debug("stop: Data collection service stopped")

            if self.scheduler_service:
                self.logger.debug("stop: Stopping scheduler service")
                await self.scheduler_service.stop()
                self.logger.debug("stop: Scheduler service stopped")

            self.logger.info("Data Collection Service stopped successfully")
            self.logger.debug("stop: Shutdown sequence completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.debug(f"stop: Shutdown failed with exception: {str(e)}")

    async def run(self):
        """Run the application until shutdown."""
        self.logger.debug("run: Starting application run loop")
        try:
            self.logger.debug("run: Calling start() method")
            await self.start()

            # Run until shutdown signal
            self.logger.info("Application is running. Press Ctrl+C to stop.")
            self.logger.debug("run: Waiting for shutdown event")
            await self._shutdown_event.wait()
            self.logger.debug("run: Shutdown event received")

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.logger.debug("run: KeyboardInterrupt caught")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            self.logger.debug(f"run: Exception in run loop: {str(e)}")
            raise
        finally:
            self.logger.debug("run: Executing finally block - calling stop()")
            await self.stop()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        self.logger.debug(
            "_setup_signal_handlers: Setting up signal handlers for SIGINT and SIGTERM"
        )

        def signal_handler(signum, frame):
            self.logger.info(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            self.logger.debug(
                f"_setup_signal_handlers: Signal handler called with signum={signum}"
            )
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self.logger.debug(
            "_setup_signal_handlers: Signal handlers registered successfully"
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        self.logger.debug("health_check: Starting comprehensive health check")
        health_info: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "data_collector",
            "status": "unknown",
            "components": {},
            "errors": [],
        }

        try:
            if not self.data_service:
                self.logger.debug("health_check: Data service not initialized")
                health_info["status"] = "not_started"
                return health_info

            # Get service status
            self.logger.debug("health_check: Getting data service status")
            service_status = await self.data_service.get_service_status()
            self.logger.debug(
                f"health_check: Service status retrieved - running: {service_status['is_running']}"
            )
            health_info["components"]["data_service"] = {
                "status": "healthy" if service_status["is_running"] else "unhealthy",
                "active_tickers": service_status["active_tickers_count"],
                "statistics": service_status["statistics"],
            }

            # Check individual components
            if self.data_service.finviz_screener:
                self.logger.debug("health_check: Checking FinViz screener health")
                finviz_healthy = (
                    await self.data_service.finviz_screener.validate_connection()
                )
                self.logger.debug(
                    f"health_check: FinViz screener health: {finviz_healthy}"
                )
                health_info["components"]["finviz"] = {
                    "status": "healthy" if finviz_healthy else "unhealthy"
                }

                if not finviz_healthy:
                    health_info["errors"].append("FinViz connection failed")

            if self.data_service.twelvedata_client:
                self.logger.debug("health_check: Checking TwelveData client health")
                twelvedata_healthy = (
                    await self.data_service.twelvedata_client.test_connection()
                )
                self.logger.debug(
                    f"health_check: TwelveData client health: {twelvedata_healthy}"
                )
                health_info["components"]["twelvedata"] = {
                    "status": "healthy" if twelvedata_healthy else "unhealthy"
                }

                if not twelvedata_healthy:
                    health_info["errors"].append("TwelveData connection failed")

            if self.data_service.redis_client:
                self.logger.debug("health_check: Checking Redis client health")
                redis_health = await self.data_service.redis_client.health_check()
                self.logger.debug(f"health_check: Redis health: {redis_health}")
                health_info["components"]["redis"] = redis_health

                if not redis_health.get("connected", False):
                    health_info["errors"].append("Redis connection failed")

            if self.data_service.data_store:
                self.logger.debug("health_check: Checking data store health")
                try:
                    summary = await self.data_service.data_store.get_data_summary()
                    self.logger.debug(
                        f"health_check: Data store summary retrieved - files: {summary.get('total_files', 0)}"
                    )
                    health_info["components"]["data_store"] = {
                        "status": "healthy",
                        "total_files": summary.get("total_files", 0),
                        "total_size_mb": summary.get("total_size_mb", 0),
                    }
                except Exception as e:
                    self.logger.debug(
                        f"health_check: Data store check failed: {str(e)}"
                    )
                    health_info["components"]["data_store"] = {"status": "unhealthy"}
                    health_info["errors"].append(f"Data store error: {str(e)}")

            # Determine overall status
            component_statuses = [
                comp.get("status")
                for comp in health_info["components"].values()
                if isinstance(comp, dict) and "status" in comp
            ]

            if all(status == "healthy" for status in component_statuses):
                health_info["status"] = "healthy"
            elif any(status == "healthy" for status in component_statuses):
                health_info["status"] = "degraded"
            else:
                health_info["status"] = "unhealthy"

        except Exception as e:
            health_info["status"] = "error"
            health_info["errors"].append(f"Health check failed: {str(e)}")
            self.logger.error(f"Health check failed: {e}")
            self.logger.debug(f"health_check: Exception during health check: {str(e)}")

        self.logger.debug(
            f"health_check: Health check completed with status: {health_info['status']}"
        )
        return health_info


async def run_health_check():
    """Run a standalone health check."""
    logger = setup_logging()
    logger.debug("run_health_check: Starting standalone health check")

    app = DataCollectorApp()

    try:
        # Quick health check without starting full service
        logger.debug("run_health_check: Executing health check")
        health = await app.health_check()
        logger.debug(f"run_health_check: Health check result: {health['status']}")

        print("Health Check Results:")
        print(f"Overall Status: {health['status']}")
        print(f"Timestamp: {health['timestamp']}")

        if health["components"]:
            print("\nComponent Status:")
            for component, status in health["components"].items():
                if isinstance(status, dict):
                    print(f"  {component}: {status.get('status', 'unknown')}")
                else:
                    print(f"  {component}: {status}")

        if health["errors"]:
            print("\nErrors:")
            for error in health["errors"]:
                print(f"  - {error}")

        return health["status"] == "healthy"

    except Exception as e:
        logger.debug(f"run_health_check: Exception occurred: {str(e)}")
        print(f"Health check failed: {e}")
        return False


async def run_data_export(
    ticker: str, timeframe: str, days: int = 30, format: str = "csv"
):
    """Run data export utility."""
    logger = setup_logging()
    logger.debug(
        f"run_data_export: Starting export for {ticker} {timeframe} {days} days in {format} format"
    )

    try:
        # Convert timeframe string to enum
        tf_map = {
            "5m": TimeFrame.FIVE_MINUTES,
            "15m": TimeFrame.FIFTEEN_MINUTES,
            "1h": TimeFrame.ONE_HOUR,
            "1d": TimeFrame.ONE_DAY,
        }

        timeframe_enum = tf_map.get(timeframe.lower())
        if not timeframe_enum:
            logger.debug(f"run_data_export: Invalid timeframe provided: {timeframe}")
            print(f"Invalid timeframe: {timeframe}. Use: 5m, 15m, 1h, 1d")
            return False

        # Initialize data store
        logger.debug("run_data_export: Initializing data store")
        config = DataStoreConfig()
        data_store = DataStore(config)

        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        logger.debug(f"run_data_export: Date range: {start_date} to {end_date}")

        # Export data
        logger.debug(f"run_data_export: Starting export for {ticker.upper()}")
        export_path = await data_store.export_data(
            ticker=ticker.upper(),
            timeframe=timeframe_enum,
            start_date=start_date,
            end_date=end_date,
            format=format,
        )

        if export_path:
            logger.debug(f"run_data_export: Export successful to {export_path}")
            print(f"Data exported successfully to: {export_path}")
            return True
        else:
            logger.debug(f"run_data_export: No data found for {ticker} {timeframe}")
            print(f"No data found for {ticker} {timeframe}")
            return False

    except Exception as e:
        logger.debug(f"run_data_export: Export failed with exception: {str(e)}")
        print(f"Export failed: {e}")
        return False


async def run_data_summary():
    """Run data summary utility."""
    logger = setup_logging()
    logger.debug("run_data_summary: Starting data summary utility")

    try:
        # Initialize data store
        logger.debug("run_data_summary: Initializing data store")
        config = DataStoreConfig()
        data_store = DataStore(config)

        # Get summary
        logger.debug("run_data_summary: Getting data summary")
        summary = await data_store.get_data_summary()
        logger.debug(
            f"run_data_summary: Summary retrieved with {summary.get('total_files', 0)} files"
        )

        print("Data Storage Summary:")
        print(f"Total files: {summary.get('total_files', 0)}")
        print(f"Total size: {summary.get('total_size_mb', 0):.2f} MB")
        print(f"Tracked tickers: {len(summary.get('tickers', []))}")
        print(f"Available timeframes: {summary.get('timeframes', [])}")

        if summary.get("date_range", {}).get("earliest"):
            print(
                f"Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}"
            )

        if summary.get("tickers"):
            print(f"Tickers: {', '.join(summary['tickers'][:10])}")
            if len(summary["tickers"]) > 10:
                print(f"... and {len(summary['tickers']) - 10} more")

        logger.debug("run_data_summary: Summary completed successfully")
        return True

    except Exception as e:
        logger.debug(f"run_data_summary: Summary failed with exception: {str(e)}")
        print(f"Summary failed: {e}")
        return False


async def main():
    """Main application entry point."""
    logger = setup_logging()
    logger.debug(f"main: Starting main function with args: {sys.argv}")

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        logger.debug(f"main: Command line command: {command}")

        if command == "health":
            # Run health check
            logger.debug("main: Executing health check command")
            success = await run_health_check()
            logger.debug(f"main: Health check completed with success: {success}")
            sys.exit(0 if success else 1)

        elif command == "export" and len(sys.argv) >= 4:
            # Run data export: python main.py export AAPL 5m [days] [format]
            ticker = sys.argv[2]
            timeframe = sys.argv[3]
            days = int(sys.argv[4]) if len(sys.argv) > 4 else 30
            format = sys.argv[5] if len(sys.argv) > 5 else "csv"
            logger.debug(f"main: Executing export command for {ticker} {timeframe}")

            success = await run_data_export(ticker, timeframe, days, format)
            logger.debug(f"main: Export completed with success: {success}")
            sys.exit(0 if success else 1)

        elif command == "summary":
            # Run data summary
            logger.debug("main: Executing summary command")
            success = await run_data_summary()
            logger.debug(f"main: Summary completed with success: {success}")
            sys.exit(0 if success else 1)

        elif command == "help":
            print("Data Collection Service Commands:")
            print("  python main.py                    - Start the service")
            print("  python main.py health             - Run health check")
            print("  python main.py summary            - Show data summary")
            print(
                "  python main.py export TICKER TF   - Export data (TF: 5m,15m,1h,1d)"
            )
            print("  python main.py help               - Show this help")
            sys.exit(0)

        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py help' for available commands")
            sys.exit(1)

    # Start the main service
    logger.info("=" * 50)
    logger.info("STARTING DATA COLLECTION SERVICE")
    logger.info("=" * 50)
    logger.debug("main: Starting main data collection service")

    app = DataCollectorApp()

    try:
        logger.debug("main: Calling app.run()")
        await app.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        logger.debug("main: KeyboardInterrupt received in main")
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.debug(f"main: Exception in main: {str(e)}")
        sys.exit(1)

    logger.info("Data Collection Service shut down")
    logger.debug("main: Main function completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
