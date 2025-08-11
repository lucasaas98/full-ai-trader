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
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

from typing import Dict, Any
from pydantic import ValidationError

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from shared.config import get_config
from shared.models import TimeFrame

from .data_collection_service import DataCollectionService, DataCollectionConfig
from .scheduler_service import SchedulerService
from .data_store import DataStore, DataStoreConfig


# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_config = get_config().logging

    # Create logs directory if it doesn't exist
    log_path = Path(log_config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.level),
        format=log_config.format,
        handlers=[
            h for h in [
                logging.StreamHandler(sys.stdout) if log_config.enable_console else None,
                logging.FileHandler(log_config.file_path) if log_config.enable_file else None
            ] if h is not None
        ]
    )

    return logging.getLogger(__name__)


class DataCollectorApp:
    """Main application class for data collection service."""

    def __init__(self):
        self.logger = setup_logging()
        self.config = self._load_configuration()
        self.data_service: Optional[DataCollectionService] = None
        self.scheduler_service: Optional[SchedulerService] = None
        self._shutdown_event = asyncio.Event()

    def _load_configuration(self) -> DataCollectionConfig:
        """Load and validate configuration."""
        try:
            # Get base config
            base_config = get_config()

            # Create data collection config
            config = DataCollectionConfig(
                service_name="data_collector",
                enable_finviz=True,
                enable_twelvedata=bool(base_config.twelvedata.api_key),
                enable_redis=True,

                # Scheduling intervals from base config
                finviz_scan_interval=base_config.scheduler.finviz_scan_interval,
                price_update_interval_5m=300,   # 5 minutes
                price_update_interval_15m=900,  # 15 minutes
                price_update_interval_1h=3600,  # 1 hour
                price_update_interval_1d=86400, # 1 day

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
                batch_size=20
            )

            return config

        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    async def start(self):
        """Start the data collection application."""
        self.logger.info("Starting Data Collection Service...")

        try:
            # Initialize and start data collection service
            self.data_service = DataCollectionService(self.config)
            await self.data_service.start()

            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()

            self.logger.info("Data Collection Service started successfully")
            self.logger.info(f"Service configuration: {self.config.dict()}")

            # Log service status
            status = await self.data_service.get_service_status()
            self.logger.info(f"Active tickers: {status['active_tickers_count']}")
            self.logger.info(f"Scheduled jobs: {len(status['scheduler_jobs'])}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            raise

    async def stop(self):
        """Stop the data collection application."""
        self.logger.info("Stopping Data Collection Service...")

        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Stop services
            if self.data_service:
                await self.data_service.stop()

            if self.scheduler_service:
                await self.scheduler_service.stop()

            self.logger.info("Data Collection Service stopped successfully")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def run(self):
        """Run the application until shutdown."""
        try:
            await self.start()

            # Run until shutdown signal
            self.logger.info("Application is running. Press Ctrl+C to stop.")
            await self._shutdown_event.wait()

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise
        finally:
            await self.stop()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "data_collector",
            "status": "unknown",
            "components": {},
            "errors": []
        }

        try:
            if not self.data_service:
                health_info["status"] = "not_started"
                return health_info

            # Get service status
            service_status = await self.data_service.get_service_status()
            health_info["components"]["data_service"] = {
                "status": "healthy" if service_status["is_running"] else "unhealthy",
                "active_tickers": service_status["active_tickers_count"],
                "statistics": service_status["statistics"]
            }

            # Check individual components
            if self.data_service.finviz_screener:
                finviz_healthy = await self.data_service.finviz_screener.validate_connection()
                health_info["components"]["finviz"] = {
                    "status": "healthy" if finviz_healthy else "unhealthy"
                }

                if not finviz_healthy:
                    health_info["errors"].append("FinViz connection failed")

            if self.data_service.twelvedata_client:
                twelvedata_healthy = await self.data_service.twelvedata_client.test_connection()
                health_info["components"]["twelvedata"] = {
                    "status": "healthy" if twelvedata_healthy else "unhealthy"
                }

                if not twelvedata_healthy:
                    health_info["errors"].append("TwelveData connection failed")

            if self.data_service.redis_client:
                redis_health = await self.data_service.redis_client.health_check()
                health_info["components"]["redis"] = redis_health

                if not redis_health.get("connected", False):
                    health_info["errors"].append("Redis connection failed")

            if self.data_service.data_store:
                try:
                    summary = await self.data_service.data_store.get_data_summary()
                    health_info["components"]["data_store"] = {
                        "status": "healthy",
                        "total_files": summary.get("total_files", 0),
                        "total_size_mb": summary.get("total_size_mb", 0)
                    }
                except Exception as e:
                    health_info["components"]["data_store"] = {"status": "unhealthy"}
                    health_info["errors"].append(f"Data store error: {str(e)}")

            # Determine overall status
            component_statuses = [
                comp.get("status") for comp in health_info["components"].values()
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

        return health_info


async def run_health_check():
    """Run a standalone health check."""
    setup_logging()

    app = DataCollectorApp()

    try:
        # Quick health check without starting full service
        health = await app.health_check()

        print("Health Check Results:")
        print(f"Overall Status: {health['status']}")
        print(f"Timestamp: {health['timestamp']}")

        if health['components']:
            print("\nComponent Status:")
            for component, status in health['components'].items():
                if isinstance(status, dict):
                    print(f"  {component}: {status.get('status', 'unknown')}")
                else:
                    print(f"  {component}: {status}")

        if health['errors']:
            print("\nErrors:")
            for error in health['errors']:
                print(f"  - {error}")

        return health['status'] == "healthy"

    except Exception as e:
        print(f"Health check failed: {e}")
        return False


async def run_data_export(ticker: str, timeframe: str, days: int = 30, format: str = "csv"):
    """Run data export utility."""
    setup_logging()

    try:
        # Convert timeframe string to enum
        tf_map = {
            "5m": TimeFrame.FIVE_MINUTES,
            "15m": TimeFrame.FIFTEEN_MINUTES,
            "1h": TimeFrame.ONE_HOUR,
            "1d": TimeFrame.ONE_DAY
        }

        timeframe_enum = tf_map.get(timeframe.lower())
        if not timeframe_enum:
            print(f"Invalid timeframe: {timeframe}. Use: 5m, 15m, 1h, 1d")
            return False

        # Initialize data store
        config = DataStoreConfig()
        data_store = DataStore(config)

        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Export data
        export_path = await data_store.export_data(
            ticker=ticker.upper(),
            timeframe=timeframe_enum,
            start_date=start_date,
            end_date=end_date,
            format=format
        )

        if export_path:
            print(f"Data exported successfully to: {export_path}")
            return True
        else:
            print(f"No data found for {ticker} {timeframe}")
            return False

    except Exception as e:
        print(f"Export failed: {e}")
        return False


async def run_data_summary():
    """Run data summary utility."""
    setup_logging()

    try:
        # Initialize data store
        config = DataStoreConfig()
        data_store = DataStore(config)

        # Get summary
        summary = await data_store.get_data_summary()

        print("Data Storage Summary:")
        print(f"Total files: {summary.get('total_files', 0)}")
        print(f"Total size: {summary.get('total_size_mb', 0):.2f} MB")
        print(f"Tracked tickers: {len(summary.get('tickers', []))}")
        print(f"Available timeframes: {summary.get('timeframes', [])}")

        if summary.get('date_range', {}).get('earliest'):
            print(f"Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")

        if summary.get('tickers'):
            print(f"Tickers: {', '.join(summary['tickers'][:10])}")
            if len(summary['tickers']) > 10:
                print(f"... and {len(summary['tickers']) - 10} more")

        return True

    except Exception as e:
        print(f"Summary failed: {e}")
        return False


async def main():
    """Main application entry point."""
    logger = setup_logging()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "health":
            # Run health check
            success = await run_health_check()
            sys.exit(0 if success else 1)

        elif command == "export" and len(sys.argv) >= 4:
            # Run data export: python main.py export AAPL 5m [days] [format]
            ticker = sys.argv[2]
            timeframe = sys.argv[3]
            days = int(sys.argv[4]) if len(sys.argv) > 4 else 30
            format = sys.argv[5] if len(sys.argv) > 5 else "csv"

            success = await run_data_export(ticker, timeframe, days, format)
            sys.exit(0 if success else 1)

        elif command == "summary":
            # Run data summary
            success = await run_data_summary()
            sys.exit(0 if success else 1)

        elif command == "help":
            print("Data Collection Service Commands:")
            print("  python main.py                    - Start the service")
            print("  python main.py health             - Run health check")
            print("  python main.py summary            - Show data summary")
            print("  python main.py export TICKER TF   - Export data (TF: 5m,15m,1h,1d)")
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

    app = DataCollectorApp()

    try:
        await app.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

    logger.info("Data Collection Service shut down")


if __name__ == "__main__":
    # Ensure we're using the right event loop policy on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
