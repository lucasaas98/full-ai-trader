"""
Main data collection service that orchestrates all components.

This module provides the primary data collection service that coordinates
FinViz screener, TwelveData API, data storage, Redis integration, and
scheduling for a comprehensive market data collection system.
"""

import asyncio
import logging
from datetime import datetime, timedelta, time as dt_time, timezone
from typing import Dict, List, Optional, Any, Set


from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import polars as pl
from pydantic import BaseModel, Field

from shared.models import TimeFrame, FinVizData
from shared.config import get_config
from shared.market_hours import is_market_open

from .finviz_screener import FinVizScreener, FinVizScreenerParams
from .twelvedata_client import TwelveDataClient, TwelveDataConfig
from .data_store import DataStore, DataStoreConfig
from .redis_client import RedisClient, RedisConfig


logger = logging.getLogger(__name__)


class DataCollectionConfig(BaseModel):
    """Configuration for data collection service."""

    # Service settings
    service_name: str = Field(default="data_collector", description="Service name")
    enable_finviz: bool = Field(default=True, description="Enable FinViz screener")
    enable_twelvedata: bool = Field(default=True, description="Enable TwelveData API")
    enable_redis: bool = Field(default=True, description="Enable Redis integration")

    # Scheduling intervals (in seconds)
    finviz_scan_interval: int = Field(default=300, description="FinViz scan interval")
    price_update_interval_5m: int = Field(default=300, description="5-minute data update interval")
    price_update_interval_15m: int = Field(default=900, description="15-minute data update interval")
    price_update_interval_1h: int = Field(default=3600, description="1-hour data update interval")
    price_update_interval_1d: int = Field(default=86400, description="Daily data update interval")

    # Data collection settings
    max_active_tickers: int = Field(default=50, description="Maximum active tickers to track")
    historical_data_years: int = Field(default=2, description="Years of historical data to collect")
    screener_result_limit: int = Field(default=20, description="Max results from screener")

    # Market hours
    market_open_time: str = Field(default="09:30", description="Market open time (HH:MM)")
    market_close_time: str = Field(default="16:00", description="Market close time (HH:MM)")
    timezone: str = Field(default="America/New_York", description="Market timezone")

    # Error handling
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=5.0, description="Delay between retries")

    # Performance settings
    concurrent_downloads: int = Field(default=10, description="Concurrent data downloads")
    batch_size: int = Field(default=20, description="Batch size for API requests")


class DataCollectionService:
    """
    Main data collection service.

    Orchestrates FinViz screening, TwelveData API calls, data storage,
    Redis integration, and scheduling for comprehensive market data collection.
    """

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.is_running = False
        self._shutdown_event = asyncio.Event()

        # Initialize components
        self.scheduler = AsyncIOScheduler()
        self.finviz_screener: Optional[FinVizScreener] = None
        self.twelvedata_client: Optional[TwelveDataClient] = None
        self.data_store: Optional[DataStore] = None
        self.redis_client: Optional[RedisClient] = None

        # Active tickers management
        self._active_tickers: Set[str] = set()
        self._ticker_lock = asyncio.Lock()

        # Performance tracking
        self._stats = {
            "screener_runs": 0,
            "data_updates": 0,
            "total_records_saved": 0,
            "errors": 0,
            "last_finviz_scan": None,
            "last_price_update": None
        }

    async def initialize(self):
        """Initialize all service components."""
        logger.info("Initializing data collection service...")

        try:
            # Initialize data store
            data_store_config = DataStoreConfig(
                base_path=get_config().data.parquet_path,
                compression=get_config().data.compression,
                retention_days=get_config().data.retention_days,
                batch_size=get_config().data.batch_size
            )
            self.data_store = DataStore(data_store_config)

            # Initialize Redis client if enabled
            if self.config.enable_redis:
                redis_config = RedisConfig(
                    host=get_config().redis.host,
                    port=get_config().redis.port,
                    db=get_config().redis.database,
                    password=get_config().redis.password,
                    max_connections=get_config().redis.max_connections
                )
                self.redis_client = RedisClient(redis_config)
                await self.redis_client.connect()

            # Initialize FinViz screener if enabled
            if self.config.enable_finviz:
                self.finviz_screener = FinVizScreener(
                    base_url=get_config().finviz.base_url,
                    rate_limit_interval=30.0,  # 30 seconds
                    timeout=get_config().finviz.timeout
                )

            # Initialize TwelveData client if enabled
            if self.config.enable_twelvedata:
                twelvedata_config = TwelveDataConfig(
                    api_key=get_config().twelvedata.api_key,
                    base_url=get_config().twelvedata.base_url,
                    rate_limit_requests=get_config().twelvedata.rate_limit_requests,
                    rate_limit_period=get_config().twelvedata.rate_limit_period,
                    timeout=get_config().twelvedata.timeout
                )
                self.twelvedata_client = TwelveDataClient(twelvedata_config)

            # Load existing active tickers
            await self._load_active_tickers()

            logger.info("Data collection service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize data collection service: {e}")
            raise

    async def start(self):
        """Start the data collection service."""
        if self.is_running:
            logger.warning("Service is already running")
            return

        logger.info("Starting data collection service...")

        try:
            await self.initialize()

            # Set up scheduled jobs
            await self._setup_scheduler()

            # Start scheduler
            self.scheduler.start()
            self.is_running = True

            # Update service status in Redis
            if self.redis_client:
                await self.redis_client.set_system_status(
                    "data_collector",
                    "online",
                    {"started_at": datetime.now(timezone.utc).isoformat()}
                )

            logger.info("Data collection service started successfully")

            # Run initial data collection in background to avoid blocking startup
            asyncio.create_task(self._run_initial_collection())

        except Exception as e:
            logger.error(f"Failed to start data collection service: {e}")
            self.is_running = False
            raise

    async def stop(self):
        """Stop the data collection service."""
        if not self.is_running:
            return

        logger.info("Stopping data collection service...")

        try:
            # Set shutdown event
            self._shutdown_event.set()

            # Stop scheduler
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)

            # Update service status
            if self.redis_client:
                await self.redis_client.set_system_status(
                    "data_collector",
                    "offline",
                    {"stopped_at": datetime.now(timezone.utc).isoformat()}
                )

            # Close connections
            if self.finviz_screener:
                await self.finviz_screener.__aexit__(None, None, None)

            if self.twelvedata_client:
                await self.twelvedata_client.__aexit__(None, None, None)

            if self.redis_client:
                await self.redis_client.disconnect()

            self.is_running = False
            logger.info("Data collection service stopped")

        except Exception as e:
            logger.error(f"Error stopping data collection service: {e}")

    async def _setup_scheduler(self):
        """Set up scheduled jobs."""
        logger.info("Setting up scheduled jobs...")

        # FinViz screener job (every 5 minutes)
        if self.config.enable_finviz:
            self.scheduler.add_job(
                self._run_finviz_scan,
                IntervalTrigger(seconds=self.config.finviz_scan_interval),
                id="finviz_scan",
                max_instances=1,
                coalesce=True
            )

        # Price update jobs for different timeframes
        if self.config.enable_twelvedata:
            # 5-minute data updates
            self.scheduler.add_job(
                self._update_price_data,
                IntervalTrigger(seconds=self.config.price_update_interval_5m),
                args=[TimeFrame.FIVE_MINUTES],
                id="price_update_5m",
                max_instances=1
            )

            # 15-minute data updates
            self.scheduler.add_job(
                self._update_price_data,
                IntervalTrigger(seconds=self.config.price_update_interval_15m),
                args=[TimeFrame.FIFTEEN_MINUTES],
                id="price_update_15m",
                max_instances=1
            )

            # Hourly data updates
            self.scheduler.add_job(
                self._update_price_data,
                IntervalTrigger(seconds=self.config.price_update_interval_1h),
                args=[TimeFrame.ONE_HOUR],
                id="price_update_1h",
                max_instances=1
            )

            # Daily data updates (at market close)
            self.scheduler.add_job(
                self._update_price_data,
                CronTrigger(hour=16, minute=30, timezone=self.config.timezone),
                args=[TimeFrame.ONE_DAY],
                id="price_update_daily",
                max_instances=1
            )

        # Maintenance jobs
        self.scheduler.add_job(
            self._cleanup_old_data,
            CronTrigger(hour=2, minute=0),  # Daily at 2 AM
            id="cleanup_old_data",
            max_instances=1
        )

        self.scheduler.add_job(
            self._validate_data_integrity,
            CronTrigger(hour=3, minute=0),  # Daily at 3 AM
            id="validate_data",
            max_instances=1
        )

        # Health check job
        self.scheduler.add_job(
            self._health_check,
            IntervalTrigger(minutes=5),
            id="health_check",
            max_instances=1
        )

    async def _run_initial_collection(self):
        """Run initial data collection on startup."""
        logger.info("Running initial data collection...")

        try:
            # Run FinViz scan first to get initial tickers
            if self.config.enable_finviz:
                await self._run_finviz_scan()

            # Download historical data for active tickers
            if self.config.enable_twelvedata and self._active_tickers:
                await self._download_historical_data()

        except Exception as e:
            logger.error(f"Initial data collection failed: {e}")

    async def _load_active_tickers(self):
        """Load active tickers from Redis or data store."""
        if self.redis_client:
            try:
                tickers = await self.redis_client.get_active_tickers()
                async with self._ticker_lock:
                    self._active_tickers.update(tickers)
                logger.info(f"Loaded {len(tickers)} active tickers from Redis")
            except Exception as e:
                logger.error(f"Failed to load tickers from Redis: {e}")

        # Fallback: scan data store for existing tickers
        if not self._active_tickers and self.data_store:
            try:
                summary = await self.data_store.get_data_summary()
                async with self._ticker_lock:
                    self._active_tickers.update(summary.get("tickers", []))
                logger.info(f"Loaded {len(self._active_tickers)} tickers from data store")
            except Exception as e:
                logger.error(f"Failed to load tickers from data store: {e}")

    async def _run_finviz_scan(self):
        """Run multiple FinViz screeners to discover new tickers."""
        if not self.finviz_screener:
            return

        try:
            logger.info("Running multiple FinViz screeners...")

            # Define screening strategies - conservative approaches only
            screening_strategies = [
                ("breakouts", lambda: self.finviz_screener.get_high_volume_breakouts(
                    limit=max(8, self.config.screener_result_limit // 6)
                )),
                ("stable_growth", lambda: self.finviz_screener.get_stable_growth_stocks(
                    limit=max(10, self.config.screener_result_limit // 5)
                )),
                ("value_stocks", lambda: self.finviz_screener.get_value_stocks(
                    limit=max(8, self.config.screener_result_limit // 6)
                )),
                ("dividend_stocks", lambda: self.finviz_screener.get_dividend_stocks(
                    limit=max(6, self.config.screener_result_limit // 8)
                ))
            ]

            all_stocks = []
            all_new_tickers = set()
            successful_screeners = 0

            # Run each screening strategy
            for strategy_name, screener_func in screening_strategies:
                try:
                    logger.info(f"Running {strategy_name} screener...")
                    result = await screener_func()
                    logger.info(f"DEBUG: {strategy_name} screener returned result: {result is not None}")

                    if result:
                        logger.info(f"DEBUG: Result has data: {hasattr(result, 'data')}, data count: {len(result.data) if hasattr(result, 'data') and result.data else 0}")

                    if result and result.data:
                        data_count = len(result.data)
                        logger.info(f"DEBUG: About to extend all_stocks with {data_count} items")
                        all_stocks.extend(result.data)
                        logger.info(f"DEBUG: Successfully extended all_stocks, now has {len(all_stocks)} items")
                        logger.info(f"DEBUG: About to create strategy_tickers set")
                        strategy_tickers = {stock.symbol for stock in result.data}
                        logger.info(f"DEBUG: Created strategy_tickers with {len(strategy_tickers)} items")
                        all_new_tickers.update(strategy_tickers)
                        logger.info(f"DEBUG: Updated all_new_tickers")
                        successful_screeners += 1
                        logger.info(f"DEBUG: Incremented successful_screeners to {successful_screeners}")

                        logger.info(f"{strategy_name} screener found {data_count} stocks")
                        logger.info(f"DEBUG: About to continue loop iteration after {strategy_name}")
                        # Save individual screener data with enhanced error handling
                        if self.data_store:
                            try:
                                logger.info(f"DEBUG: Attempting to save screener data for {strategy_name}")
                                logger.info(f"DEBUG: First few records: {result.data[:2] if result.data else 'No data'}")
                                saved_count = await self.data_store.save_screener_data(result.data, strategy_name)
                                logger.info(f"DEBUG: Successfully saved {saved_count} records for {strategy_name}")
                            except Exception as e:
                                logger.error(f"DEBUG: Failed to save screener data for {strategy_name}: {e}")
                                logger.error(f"DEBUG: Exception type: {type(e).__name__}")
                                import traceback
                                logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
                                # Continue processing even if save fails
                                pass
                        logger.info(f"DEBUG: Finished processing screener data for {strategy_name}")
                    else:
                        logger.warning(f"{strategy_name} screener returned no results")

                except Exception as e:
                    logger.error(f"{strategy_name} screener failed: {e}")
                    continue

            logger.info(f"DEBUG: Finished all screener strategies, total stocks: {len(all_stocks)}")

            # If no stocks found from screeners, use fallback ticker lists
            if not all_stocks:
                logger.warning("All FinViz screeners returned no results, using fallback tickers")
                await self._use_fallback_tickers()
                return

            logger.info("DEBUG: Starting to remove duplicates from stocks")
            # Remove duplicates while preserving the best stocks
            unique_stocks = {}
            for i, stock in enumerate(all_stocks):
                logger.info(f"DEBUG: Processing stock {i}: {stock.symbol if hasattr(stock, 'symbol') else 'NO SYMBOL'}")
                symbol = stock.symbol
                if symbol not in unique_stocks:
                    unique_stocks[symbol] = stock
                else:
                    # Keep the stock with higher volume
                    if (stock.volume or 0) > (unique_stocks[symbol].volume or 0):
                        unique_stocks[symbol] = stock

            final_stocks = list(unique_stocks.values())
            new_tickers = set(unique_stocks.keys())

            async with self._ticker_lock:
                previously_active = self._active_tickers.copy()

                # Update active tickers (keep existing + add new, but limit total)
                combined_tickers = self._active_tickers.union(new_tickers)

                # If we exceed max tickers, prioritize by volume
                if len(combined_tickers) > self.config.max_active_tickers:
                    # Sort by volume and take top N
                    sorted_stocks = sorted(final_stocks, key=lambda x: x.volume or 0, reverse=True)
                    top_tickers = {stock.symbol for stock in sorted_stocks[:self.config.max_active_tickers]}

                    # Keep some existing tickers to maintain continuity
                    existing_to_keep = min(10, len(previously_active))
                    if existing_to_keep > 0:
                        existing_list = list(previously_active)[:existing_to_keep]
                        combined_tickers = top_tickers.union(set(existing_list))

                        # Still might exceed limit, so trim again
                        if len(combined_tickers) > self.config.max_active_tickers:
                            combined_tickers = set(list(combined_tickers)[:self.config.max_active_tickers])

                truly_new_tickers = combined_tickers - previously_active
                self._active_tickers = combined_tickers

            # Publish new tickers to Redis
            if truly_new_tickers and self.redis_client:
                await self.redis_client.publish_new_tickers(
                    list(truly_new_tickers),
                    {
                        "source": "finviz_multi_screener",
                        "strategies_used": successful_screeners,
                        "total_screened": len(final_stocks),
                        "unique_tickers": len(unique_stocks)
                    }
                )

            # Publish screener update with combined results
            if self.redis_client:
                logger.info(f"DEBUG: About to publish screener update with {len(final_stocks)} stocks")
                await self.redis_client.publish_screener_update(final_stocks, "multi_strategy")
                logger.info("DEBUG: Successfully published screener update")

            self._stats["screener_runs"] += 1
            self._stats["last_finviz_scan"] = datetime.now(timezone.utc)

            logger.info(f"Multi-strategy FinViz scan completed: {len(final_stocks)} total stocks, "
                       f"{len(truly_new_tickers)} new tickers from {successful_screeners} successful screeners")

        except Exception as e:
            logger.error(f"FinViz multi-screener scan failed: {e}")
            self._stats["errors"] += 1

            # Publish error alert
            if self.redis_client:
                await self.redis_client.publish_data_validation_alert(
                    "FINVIZ_SCREENER",
                    [f"Multi-screener scan failed: {str(e)}"],
                    "error"
                )

    async def _use_fallback_tickers(self):
        """Use fallback ticker lists when screeners return no results."""
        try:
            # Define popular, liquid stocks as fallbacks
            fallback_tickers = [
                # Large cap tech
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                # Financial
                "JPM", "BAC", "WFC", "GS", "MS",
                # Healthcare
                "JNJ", "UNH", "PFE", "ABBV", "TMO",
                # Consumer
                "PG", "KO", "PEP", "WMT", "HD",
                # ETFs for stability
                "SPY", "QQQ", "IWM", "XLF", "XLK"
            ]

            # Limit to max active tickers
            selected_tickers = fallback_tickers[:min(len(fallback_tickers), self.config.max_active_tickers)]

            async with self._ticker_lock:
                previously_active = self._active_tickers.copy()
                self._active_tickers = set(selected_tickers)
                truly_new_tickers = self._active_tickers - previously_active

            # Publish new tickers to Redis
            if truly_new_tickers and self.redis_client:
                await self.redis_client.publish_new_tickers(
                    list(truly_new_tickers),
                    {
                        "source": "fallback_tickers",
                        "reason": "screeners_returned_no_results",
                        "total_tickers": len(selected_tickers)
                    }
                )

            self._stats["screener_runs"] += 1
            self._stats["last_finviz_scan"] = datetime.now(timezone.utc)

            logger.info(f"Fallback tickers activated: {len(selected_tickers)} tickers, {len(truly_new_tickers)} new")

        except Exception as e:
            logger.error(f"Fallback ticker selection failed: {e}")
            self._stats["errors"] += 1

    async def _update_price_data(self, timeframe: TimeFrame):
        """
        Update price data for active tickers.

        Args:
            timeframe: Data timeframe to update
        """
        if not self.twelvedata_client or not self._active_tickers:
            return

        # Check if we're in market hours for intraday data
        if timeframe in [TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES] and not await self._is_market_hours():
            logger.debug(f"Skipping {timeframe} update outside market hours")
            return

        try:
            logger.info(f"Updating {timeframe} price data for {len(self._active_tickers)} tickers...")

            tickers_list = list(self._active_tickers)

            # Determine how much data to fetch
            if timeframe == TimeFrame.ONE_DAY:
                # For daily data, get last 5 days
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=5)
            else:
                # For intraday data, ensure we span at least 2 days for TwelveData API
                end_date = datetime.now(timezone.utc)

                # Calculate days back based on timeframe, minimum 2 days
                days_map = {
                    TimeFrame.FIVE_MINUTES: 2,    # Last 2 days
                    TimeFrame.FIFTEEN_MINUTES: 3,  # Last 3 days
                    TimeFrame.ONE_HOUR: 5         # Last 5 days
                }
                days_back = days_map.get(timeframe, 2)
                start_date = end_date - timedelta(days=days_back)

            # Fetch data in batches to respect API limits
            batch_size = min(self.config.batch_size, len(tickers_list))
            batches = [tickers_list[i:i + batch_size] for i in range(0, len(tickers_list), batch_size)]

            all_market_data = []
            successful_updates = 0

            for batch_num, batch_tickers in enumerate(batches, 1):
                logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch_tickers)} tickers")

                try:
                    # Fetch data for this batch
                    batch_data = await self.twelvedata_client.get_batch_time_series(
                        symbols=batch_tickers,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )

                    # Process results
                    for ticker, data_list in batch_data.items():
                        if data_list:
                            all_market_data.extend(data_list)
                            successful_updates += 1

                            # Update last update time
                            if self.redis_client:
                                await self.redis_client.update_last_update_time(ticker, timeframe)

                    # Small delay between batches
                    if batch_num < len(batches):
                        await asyncio.sleep(2.0)

                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    self._stats["errors"] += 1

            # Save all collected data
            if all_market_data and self.data_store:
                save_stats = await self.data_store.save_market_data(all_market_data, append=True)
                self._stats["total_records_saved"] += save_stats["total_saved"]

                logger.info(f"Saved {save_stats['total_saved']} {timeframe} records for {successful_updates} tickers")

                # Publish update notification
                if self.redis_client:
                    for ticker in set(md.symbol for md in all_market_data):
                        ticker_data = [md for md in all_market_data if md.symbol == ticker]
                        await self.redis_client.publish_market_data_update(
                            ticker, timeframe, len(ticker_data), "scheduled_update"
                        )

            # Cache latest prices if this is real-time data
            if timeframe == TimeFrame.FIVE_MINUTES and self.redis_client:
                latest_prices = {}
                for data in all_market_data:
                    ticker = data.symbol
                    if ticker not in latest_prices or data.timestamp > latest_prices[ticker].timestamp:
                        latest_prices[ticker] = data

                if latest_prices:
                    await self.redis_client.batch_cache_prices(latest_prices)

            self._stats["data_updates"] += 1
            self._stats["last_price_update"] = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Price data update failed for {timeframe}: {e}")
            self._stats["errors"] += 1

    async def _download_historical_data(self):
        """Download historical data for new tickers."""
        if not self.twelvedata_client or not self.data_store:
            return

        logger.info("Downloading historical data for active tickers...")

        timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES, TimeFrame.ONE_HOUR, TimeFrame.ONE_DAY]

        for ticker in list(self._active_tickers):
            try:
                # Check if we already have recent data
                has_recent_data = False
                for tf in timeframes:
                    date_range = await self.data_store.get_available_data_range(ticker, tf)
                    if date_range:
                        _, latest_date = date_range
                        if latest_date >= datetime.now().date() - timedelta(days=7):
                            has_recent_data = True
                            break

                if has_recent_data:
                    logger.debug(f"Skipping historical download for {ticker} - recent data exists")
                    continue

                logger.info(f"Downloading historical data for {ticker}")

                # Download data for each timeframe
                for timeframe in timeframes:
                    try:
                        historical_data = await self.twelvedata_client.get_historical_data(
                            ticker, timeframe, self.config.historical_data_years
                        )

                        if historical_data:
                            await self.data_store.save_market_data(historical_data, append=True)
                            logger.info(f"Downloaded {len(historical_data)} {timeframe} records for {ticker}")

                        # Small delay between timeframes
                        await asyncio.sleep(1.0)

                    except Exception as e:
                        logger.error(f"Failed to download {timeframe} data for {ticker}: {e}")

                # Delay between tickers to respect rate limits
                await asyncio.sleep(5.0)

            except Exception as e:
                logger.error(f"Historical data download failed for {ticker}: {e}")

    async def _is_market_hours(self) -> bool:
        """Check if current time is within market hours using Alpaca API."""
        try:
            return await is_market_open()
        except Exception as e:
            logger.error(f"Failed to check market hours via Alpaca API: {e}")
            # Fallback to simple time-based check
            return self._is_market_hours_fallback()

    def _is_market_hours_fallback(self) -> bool:
        """Fallback market hours check when Alpaca API is unavailable."""
        try:
            now = datetime.now()

            # Skip weekends
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False

            # Parse market hours
            market_open = dt_time.fromisoformat(self.config.market_open_time)
            market_close = dt_time.fromisoformat(self.config.market_close_time)

            current_time = now.time()
            return market_open <= current_time <= market_close

        except Exception as e:
            logger.error(f"Failed fallback market hours check: {e}")
            return True  # Default to True to avoid missing updates

    async def _cleanup_old_data(self):
        """Clean up old data files."""
        if not self.data_store:
            return

        try:
            logger.info("Running data cleanup...")
            cleanup_stats = await self.data_store.cleanup_old_data()
            logger.info(f"Cleanup completed: {cleanup_stats}")

            # Update stats in Redis
            if self.redis_client:
                await self.redis_client.cache_data_statistics(
                    "SYSTEM",
                    TimeFrame.ONE_DAY,
                    {"last_cleanup": cleanup_stats},
                    ttl=86400  # 24 hours
                )

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

    async def _validate_data_integrity(self):
        """Validate data integrity for active tickers."""
        if not self.data_store:
            return

        try:
            logger.info("Running data integrity validation...")

            validation_issues = []

            for ticker in list(self._active_tickers):
                for timeframe in [TimeFrame.FIVE_MINUTES, TimeFrame.ONE_DAY]:
                    try:
                        validation = await self.data_store.validate_data_integrity(
                            ticker, timeframe
                        )

                        if not validation.get("valid", True):
                            issues = validation.get("issues", [])
                            if issues:
                                validation_issues.extend([f"{ticker} {timeframe}: {issue}" for issue in issues])

                    except Exception as e:
                        validation_issues.append(f"{ticker} {timeframe}: Validation failed - {str(e)}")

            # Report validation issues
            if validation_issues:
                logger.warning(f"Found {len(validation_issues)} data validation issues")

                if self.redis_client:
                    await self.redis_client.publish_data_validation_alert(
                        "DATA_VALIDATION",
                        validation_issues,
                        "warning"
                    )
            else:
                logger.info("Data integrity validation passed")

        except Exception as e:
            logger.error(f"Data validation failed: {e}")

    async def _health_check(self):
        """Perform health check on all components."""
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy",
            "components": {}
        }

        try:
            # Check FinViz screener
            if self.finviz_screener:
                finviz_healthy = await self.finviz_screener.validate_connection()
                health_status["components"]["finviz"] = {
                    "status": "healthy" if finviz_healthy else "unhealthy",
                    "last_scan": self._stats["last_finviz_scan"]
                }
                if not finviz_healthy:
                    health_status["overall_status"] = "degraded"

            # Check TwelveData client
            if self.twelvedata_client:
                twelvedata_healthy = await self.twelvedata_client.test_connection()
                health_status["components"]["twelvedata"] = {
                    "status": "healthy" if twelvedata_healthy else "unhealthy",
                    "last_update": self._stats["last_price_update"]
                }
                if not twelvedata_healthy:
                    health_status["overall_status"] = "degraded"

            # Check Redis
            if self.redis_client:
                redis_health = await self.redis_client.health_check()
                health_status["components"]["redis"] = {
                    "status": "healthy" if redis_health["connected"] else "unhealthy",
                    "latency_ms": redis_health.get("latency_ms"),
                    "memory_usage": redis_health.get("memory_usage")
                }
                if not redis_health["connected"]:
                    health_status["overall_status"] = "degraded"

            # Check data store
            if self.data_store:
                try:
                    summary = await self.data_store.get_data_summary()
                    health_status["components"]["data_store"] = {
                        "status": "healthy",
                        "total_files": summary.get("total_files", 0),
                        "total_size_mb": summary.get("total_size_mb", 0),
                        "tickers_count": len(summary.get("tickers", []))
                    }
                except Exception:
                    health_status["components"]["data_store"] = {"status": "unhealthy"}
                    health_status["overall_status"] = "degraded"

            # Add service statistics
            health_status["statistics"] = self._stats.copy()
            health_status["active_tickers_count"] = len(self._active_tickers)

            # Update health status in Redis
            if self.redis_client:
                await self.redis_client.set_system_status(
                    "data_collector_health",
                    health_status["overall_status"],
                    health_status
                )

            logger.debug(f"Health check completed: {health_status['overall_status']}")

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)

    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get current service status and statistics.

        Returns:
            Service status information
        """
        return {
            "service_name": self.config.service_name,
            "is_running": self.is_running,
            "active_tickers_count": len(self._active_tickers),
            "active_tickers": list(self._active_tickers),
            "statistics": self._stats.copy(),
            "configuration": self.config.dict(),
            "scheduler_jobs": [
                {
                    "id": job.id,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                }
                for job in self.scheduler.get_jobs()
            ] if self.scheduler else []
        }

    async def add_ticker(self, ticker: str, fetch_historical: bool = True):
        """
        Manually add a ticker to tracking.

        Args:
            ticker: Ticker symbol to add
            fetch_historical: Whether to fetch historical data
        """
        ticker = ticker.upper()

        async with self._ticker_lock:
            if ticker not in self._active_tickers:
                logger.info(f"Ticker {ticker} not currently being tracked")
                return

            self._active_tickers.remove(ticker)

        # Remove from Redis
        if self.redis_client:
            await self.redis_client.remove_ticker(ticker)

        logger.info(f"Removed ticker {ticker} from tracking")

    async def get_ticker_data(
        self,
        ticker: str,
        timeframe: TimeFrame,
        days_back: int = 30
    ) -> Optional[pl.DataFrame]:
        """
        Get recent data for a specific ticker.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            days_back: Number of days of data to retrieve

        Returns:
            DataFrame with ticker data or None if not found
        """
        if not self.data_store:
            return None

        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)

            df = await self.data_store.load_market_data(
                ticker, timeframe, start_date, end_date
            )

            return df if not df.is_empty() else None

        except Exception as e:
            logger.error(f"Failed to get data for {ticker}: {e}")
            return None

    async def force_ticker_update(self, ticker: str, timeframes: Optional[List[TimeFrame]] = None):
        """
        Force immediate update for a specific ticker.

        Args:
            ticker: Ticker symbol to update
            timeframes: List of timeframes to update (all if None)
        """
        if not self.twelvedata_client:
            logger.error("TwelveData client not available")
            return

        if timeframes is None:
            timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES, TimeFrame.ONE_HOUR, TimeFrame.ONE_DAY]

        logger.info(f"Force updating {ticker} for timeframes: {[tf.value for tf in timeframes]}")

        try:
            for timeframe in timeframes:
                # Get recent data
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(hours=24)  # Last 24 hours

                data = await self.twelvedata_client.get_time_series(
                    ticker, timeframe, start_date, end_date
                )

                if data and self.data_store:
                    await self.data_store.save_market_data(data, append=True)
                    logger.info(f"Force updated {len(data)} {timeframe} records for {ticker}")

                    # Publish update
                    if self.redis_client:
                        await self.redis_client.publish_market_data_update(
                            ticker, timeframe, len(data), "force_update"
                        )

        except Exception as e:
            logger.error(f"Force update failed for {ticker}: {e}")

    async def get_active_tickers(self) -> List[str]:
        """Get list of currently active tickers."""
        async with self._ticker_lock:
            return list(self._active_tickers)

    async def run_custom_screener(
        self,
        screener_params: FinVizScreenerParams,
        limit: int = 15,
        add_to_tracking: bool = False
    ) -> List[FinVizData]:
        """
        Run custom screener with specific parameters.

        Args:
            screener_params: Custom screener parameters
            limit: Maximum results to return
            add_to_tracking: Whether to add results to active tracking

        Returns:
            List of screener results
        """
        if not self.finviz_screener:
            logger.error("FinViz screener not available")
            return []

        try:
            result = await self.finviz_screener.fetch_screener_data(screener_params, limit)

            if result.data:
                # Save screener data
                if self.data_store:
                    await self.data_store.save_screener_data(result.data, "custom")

                # Add to tracking if requested
                if add_to_tracking:
                    new_tickers = [stock.symbol for stock in result.data]
                    async with self._ticker_lock:
                        for ticker in new_tickers:
                            if len(self._active_tickers) < self.config.max_active_tickers:
                                self._active_tickers.add(ticker)

                    if self.redis_client:
                        await self.redis_client.publish_new_tickers(
                            new_tickers,
                            {"source": "custom_screener"}
                        )

                logger.info(f"Custom screener found {len(result.data)} results")

            return result.data

        except Exception as e:
            logger.error(f"Custom screener failed: {e}")
            return []

    async def get_real_time_prices(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get real-time prices for tickers.

        Args:
            tickers: List of tickers (uses active tickers if None)

        Returns:
            Dictionary with real-time price data
        """
        if not self.twelvedata_client:
            logger.error("TwelveData client not available")
            return {}

        if tickers is None:
            tickers = list(self._active_tickers)

        if not tickers:
            return {}

        try:
            # First try to get from cache
            cached_prices = {}
            if self.redis_client:
                cached_prices = await self.redis_client.get_batch_cached_prices(tickers)

            # Get fresh data for tickers not in cache or with stale cache
            fresh_data_needed = []
            for ticker in tickers:
                cached = cached_prices.get(ticker)
                if not cached:
                    fresh_data_needed.append(ticker)
                else:
                    # Check if cache is recent (less than 5 minutes old)
                    try:
                        cache_time = datetime.fromisoformat(cached["timestamp"])
                        if datetime.now(timezone.utc) - cache_time > timedelta(minutes=5):
                            fresh_data_needed.append(ticker)
                    except (KeyError, ValueError):
                        fresh_data_needed.append(ticker)

            # Fetch fresh data if needed
            if fresh_data_needed:
                fresh_prices = await self.twelvedata_client.get_batch_real_time_prices(fresh_data_needed)

                # Update cache (filter out None values)
                if self.redis_client:
                    valid_prices = {k: v for k, v in fresh_prices.items() if v is not None}
                    if valid_prices:
                        await self.redis_client.batch_cache_prices(valid_prices, ttl=300)  # type: ignore

                # Merge fresh data with cached data
                for ticker, price_data in fresh_prices.items():
                    if price_data:
                        cached_prices[ticker] = {
                            "symbol": price_data.symbol,
                            "timestamp": price_data.timestamp.isoformat(),
                            "price": float(price_data.close),
                            "open": float(price_data.open),
                            "high": float(price_data.high),
                            "low": float(price_data.low),
                            "volume": price_data.volume
                        }

            return cached_prices

        except Exception as e:
            logger.error(f"Failed to get real-time prices: {e}")
            return {}

    async def export_ticker_data(
        self,
        ticker: str,
        timeframe: TimeFrame,
        days_back: int = 30,
        format: str = "csv"
    ) -> Optional[str]:
        """
        Export data for a specific ticker.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            days_back: Number of days to export
            format: Export format

        Returns:
            Path to exported file or None if failed
        """
        if not self.data_store:
            return None

        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)

            export_path = await self.data_store.export_data(
                ticker, timeframe, start_date, end_date, format
            )

            logger.info(f"Exported {ticker} data to {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"Failed to export data for {ticker}: {e}")
            return None

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'scheduler') and self.scheduler.running:
            self.scheduler.shutdown(wait=False)


# Utility functions for service management
async def create_and_start_service(config: Optional[DataCollectionConfig] = None) -> DataCollectionService:
    """
    Create and start data collection service.

    Args:
        config: Service configuration (uses defaults if None)

    Returns:
        Started DataCollectionService instance
    """
    if config is None:
        config = DataCollectionConfig()

    service = DataCollectionService(config)
    await service.start()
    return service


async def run_service_with_graceful_shutdown(config: Optional[DataCollectionConfig] = None):
    """
    Run data collection service with graceful shutdown handling.

    Args:
        config: Service configuration
    """
    service = None
    try:
        service = await create_and_start_service(config)

        logger.info("Data collection service is running. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        await service._shutdown_event.wait()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        if service:
            await service.stop()


# Example usage
if __name__ == "__main__":
    async def main():
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create service configuration
        config = DataCollectionConfig(
            enable_finviz=True,
            enable_twelvedata=True,
            enable_redis=True,
            finviz_scan_interval=300,  # 5 minutes
            max_active_tickers=30,
            screener_result_limit=20
        )

        # Run service
        await run_service_with_graceful_shutdown(config)

    asyncio.run(main())
