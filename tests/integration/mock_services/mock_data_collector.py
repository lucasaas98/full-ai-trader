"""
Mock Data Collector for Integration Tests

This module provides a mock data collector that serves historical data from
parquet files instead of fetching from external APIs. It maintains the same
Redis pub/sub communication patterns as the real data collector.
"""

import asyncio
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import polars as pl
import redis.asyncio as redis
from pydantic import BaseModel, Field

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from shared.config import get_config
from shared.models import AssetType, FinVizData, MarketData, TimeFrame

logger = logging.getLogger(__name__)


class MockDataCollectorConfig(BaseModel):
    """Configuration for mock data collector."""

    historical_data_path: str = Field(
        default="data/parquet", description="Path to historical parquet data"
    )
    available_symbols: List[str] = Field(
        default=["AAPL", "SPY", "QQQ", "MSFT", "TSLA"], description="Available symbols"
    )
    redis_publish_interval: int = Field(
        default=30, description="Interval for publishing data updates (seconds)"
    )
    simulate_screener: bool = Field(
        default=True, description="Simulate screener updates"
    )
    screener_interval: int = Field(
        default=300, description="Screener simulation interval (seconds)"
    )
    enable_redis: bool = Field(default=True, description="Enable Redis pub/sub")


class MockDataCollector:
    """
    Mock data collector that serves historical data and simulates real-time updates.

    This class:
    1. Loads historical market data from parquet files
    2. Publishes data updates via Redis pub/sub
    3. Simulates screener results
    4. Maintains compatibility with real data collector interface
    """

    def __init__(self, config: MockDataCollectorConfig):
        self.config = config
        self.base_path = Path(config.historical_data_path)
        self.redis_client: Optional[redis.Redis] = None
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        self._cached_data: Dict[str, Dict[str, pl.DataFrame]] = {}
        self._simulation_date = date.today() - timedelta(
            days=30
        )  # Start from 30 days ago

        # Redis channels (same as real data collector)
        self.CHANNELS = {
            "TICKERS_NEW": "tickers:new",
            "PRICES": "prices:",
            "SCREENER_UPDATES": "screener:updates",
            "MARKET_DATA_UPDATES": "market_data:updates",
            "SYSTEM_ALERTS": "system:alerts",
            "DATA_VALIDATION": "data:validation",
        }

    async def start(self):
        """Start the mock data collector."""
        logger.info("Starting Mock Data Collector...")

        # Initialize Redis connection
        if self.config.enable_redis:
            await self._setup_redis()

        # Load historical data
        await self._load_historical_data()

        # Start background tasks
        self.is_running = True
        if self.config.enable_redis:
            self._tasks.append(asyncio.create_task(self._publish_data_updates()))

        if self.config.simulate_screener:
            self._tasks.append(asyncio.create_task(self._simulate_screener()))

        logger.info(
            f"Mock Data Collector started with {len(self.config.available_symbols)} symbols"
        )

    async def stop(self):
        """Stop the mock data collector."""
        logger.info("Stopping Mock Data Collector...")

        self.is_running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        logger.info("Mock Data Collector stopped")

    async def _setup_redis(self):
        """Setup Redis connection."""
        try:
            config = get_config()
            self.redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                password=config.redis.password,
                db=config.redis.db,
                decode_responses=True,
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")

        except Exception as e:
            logger.error(f"Failed to setup Redis: {e}")
            self.redis_client = None

    async def _load_historical_data(self):
        """Load historical data from parquet files."""
        logger.info("Loading historical data...")

        for symbol in self.config.available_symbols:
            symbol_path = self.base_path / "market_data" / symbol

            if not symbol_path.exists():
                logger.warning(f"No data found for symbol {symbol}")
                continue

            self._cached_data[symbol] = {}

            # Load different timeframes
            timeframe_mapping = {
                "5m": TimeFrame.FIVE_MINUTES,
                "15m": TimeFrame.FIFTEEN_MINUTES,
                "1h": TimeFrame.ONE_HOUR,
                "1d": TimeFrame.ONE_DAY,
            }

            for tf_str, timeframe in timeframe_mapping.items():
                tf_data = []

                # Load all parquet files for this timeframe
                for parquet_file in symbol_path.glob(f"*{tf_str}*.parquet"):
                    try:
                        df = pl.read_parquet(parquet_file)
                        tf_data.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to load {parquet_file}: {e}")

                if tf_data:
                    combined_df = pl.concat(tf_data).sort("timestamp")
                    self._cached_data[symbol][tf_str] = combined_df
                    logger.debug(
                        f"Loaded {len(combined_df)} records for {symbol} {tf_str}"
                    )

        total_symbols = len([s for s in self._cached_data if self._cached_data[s]])
        logger.info(f"Historical data loaded for {total_symbols} symbols")

    async def _publish_data_updates(self):
        """Simulate real-time data updates."""
        while self.is_running:
            try:
                # Advance simulation date
                self._simulation_date += timedelta(minutes=5)
                current_time = datetime.combine(
                    self._simulation_date, datetime.min.time()
                )

                # Publish market data updates for each symbol
                for symbol in self.config.available_symbols:
                    if symbol in self._cached_data and self._cached_data[symbol]:
                        await self._publish_symbol_update(symbol, current_time)

                await asyncio.sleep(self.config.redis_publish_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data updates: {e}")
                await asyncio.sleep(5)

    async def _publish_symbol_update(self, symbol: str, current_time: datetime):
        """Publish market data update for a symbol."""
        if not self.redis_client:
            return

        try:
            # Get data for the current simulation time
            symbol_data = self._cached_data.get(symbol, {})

            for timeframe_str, df in symbol_data.items():
                # Filter data up to current simulation time
                filtered_df = df.filter(pl.col("timestamp") <= current_time)

                if len(filtered_df) > 0:
                    # Get the latest record
                    latest_record = filtered_df.tail(1).to_dicts()[0]

                    # Create MarketData object
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=latest_record["timestamp"],
                        timeframe=TimeFrame(
                            timeframe_str.replace("m", "_minutes")
                            .replace("h", "_hour")
                            .replace("d", "_day")
                        ),
                        open=float(latest_record["open"]),
                        high=float(latest_record["high"]),
                        low=float(latest_record["low"]),
                        close=float(latest_record["close"]),
                        volume=int(latest_record["volume"]),
                        asset_type=AssetType.STOCK,
                    )

                    # Publish to Redis
                    channel = f"{self.CHANNELS['PRICES']}{symbol}:{timeframe_str}"
                    message = {
                        "type": "market_data_update",
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "timestamp": current_time.isoformat(),
                        "data": market_data.dict(),
                        "source": "mock_data_collector",
                    }

                    await self.redis_client.publish(
                        channel, json.dumps(message, default=str)
                    )

                    # Also publish to general market data updates channel
                    await self.redis_client.publish(
                        self.CHANNELS["MARKET_DATA_UPDATES"],
                        json.dumps(message, default=str),
                    )

        except Exception as e:
            logger.error(f"Error publishing update for {symbol}: {e}")

    async def _simulate_screener(self):
        """Simulate screener updates."""
        while self.is_running:
            try:
                # Create mock screener data
                screener_data = []

                for symbol in self.config.available_symbols:
                    if symbol in self._cached_data and self._cached_data[symbol]:
                        # Get latest price data
                        daily_data = self._cached_data[symbol].get("1d")
                        if daily_data is not None and len(daily_data) > 0:
                            latest = daily_data.tail(1).to_dicts()[0]

                            finviz_data = FinVizData(
                                symbol=symbol,
                                company=f"{symbol} Corporation",
                                sector=(
                                    "Technology"
                                    if symbol in ["AAPL", "MSFT", "TSLA"]
                                    else "Financial"
                                ),
                                industry=(
                                    "Software"
                                    if symbol in ["AAPL", "MSFT"]
                                    else "ETF" if symbol in ["SPY", "QQQ"] else "Auto"
                                ),
                                country="USA",
                                market_cap="1B-10B",
                                pe_ratio=25.5,
                                price=float(latest["close"]),
                                change=float(latest["close"] - latest["open"]),
                                volume=int(latest["volume"]),
                            )

                            screener_data.append(finviz_data)

                # Publish screener update
                if screener_data and self.redis_client:
                    message = {
                        "type": "screener_update",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "screener_type": "mock_screener",
                        "results_count": len(screener_data),
                        "data": [stock.dict() for stock in screener_data],
                        "source": "mock_data_collector",
                    }

                    await self.redis_client.publish(
                        self.CHANNELS["SCREENER_UPDATES"],
                        json.dumps(message, default=str),
                    )

                    logger.debug(
                        f"Published mock screener update with {len(screener_data)} stocks"
                    )

                await asyncio.sleep(self.config.screener_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in screener simulation: {e}")
                await asyncio.sleep(10)

    async def get_historical_data(
        self, symbol: str, timeframe: str, start_date: date, end_date: date
    ) -> List[MarketData]:
        """Get historical data for a symbol and timeframe."""
        if (
            symbol not in self._cached_data
            or timeframe not in self._cached_data[symbol]
        ):
            return []

        df = self._cached_data[symbol][timeframe]

        # Filter by date range
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        filtered_df = df.filter(
            (pl.col("timestamp") >= start_dt) & (pl.col("timestamp") <= end_dt)
        )

        # Convert to MarketData objects
        market_data = []
        for record in filtered_df.to_dicts():
            market_data.append(
                MarketData(
                    symbol=symbol,
                    timestamp=record["timestamp"],
                    timeframe=TimeFrame(
                        timeframe.replace("m", "_minutes")
                        .replace("h", "_hour")
                        .replace("d", "_day")
                    ),
                    open=float(record["open"]),
                    high=float(record["high"]),
                    low=float(record["low"]),
                    close=float(record["close"]),
                    volume=int(record["volume"]),
                    asset_type=AssetType.STOCK,
                )
            )

        return market_data

    async def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return [
            symbol
            for symbol in self.config.available_symbols
            if symbol in self._cached_data
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "available_symbols": len(self._cached_data),
            "redis_connected": self.redis_client is not None,
            "simulation_date": self._simulation_date.isoformat(),
            "running_tasks": len([t for t in self._tasks if not t.done()]),
            "service": "mock_data_collector",
        }

    async def force_screener_update(self):
        """Force an immediate screener update."""
        if self.config.simulate_screener and self.is_running:
            await self._simulate_screener()

    async def reset_simulation_date(self, new_date: Optional[date] = None):
        """Reset the simulation date."""
        if new_date:
            self._simulation_date = new_date
        else:
            self._simulation_date = date.today() - timedelta(days=30)

        logger.info(f"Simulation date reset to {self._simulation_date}")


async def create_mock_data_collector(
    config: Optional[MockDataCollectorConfig] = None,
) -> MockDataCollector:
    """Factory function to create and start a mock data collector."""
    if config is None:
        config = MockDataCollectorConfig()

    collector = MockDataCollector(config)
    await collector.start()
    return collector


if __name__ == "__main__":
    import sys

    async def main():
        # Simple test runner
        config = MockDataCollectorConfig(
            available_symbols=["AAPL", "SPY"],
            redis_publish_interval=10,
            simulate_screener=True,
        )

        collector = await create_mock_data_collector(config)

        try:
            print("Mock Data Collector started. Press Ctrl+C to stop...")
            await asyncio.sleep(3600)  # Run for 1 hour
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            await collector.stop()

    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
