"""
Redis integration for data collection service.

This module provides Redis integration for publishing ticker updates,
caching frequently accessed data, and managing real-time data streams.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import redis.asyncio as redis
from pydantic import BaseModel, Field

from shared.models import FinVizData, MarketData, TimeFrame

logger = logging.getLogger(__name__)


class RedisConfig(BaseModel):
    """Redis configuration."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(None, description="Redis password")
    max_connections: int = Field(default=20, description="Maximum connections in pool")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    decode_responses: bool = Field(
        default=True, description="Decode responses to strings"
    )

    @property
    def url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class RedisChannels:
    """Redis channel names."""

    TICKERS_NEW = "tickers:new"
    PRICES_PREFIX = "prices:"
    SCREENER_UPDATES = "screener:updates"
    MARKET_DATA_UPDATES = "market_data:updates"
    SYSTEM_ALERTS = "system:alerts"
    DATA_VALIDATION = "data:validation"


class RedisKeys:
    """Redis key patterns."""

    TICKER_LIST = "active_tickers"
    PRICE_CACHE_PREFIX = "price_cache:"
    SCREENER_CACHE_PREFIX = "screener_cache:"
    LAST_UPDATE_PREFIX = "last_update:"
    DATA_STATS_PREFIX = "data_stats:"


class RedisClient:
    """
    Redis client for data collection service.

    Provides high-level interface for publishing ticker updates,
    caching data, and managing real-time streams.
    """

    def __init__(self, config: RedisConfig):
        self.config = config
        self._pool: Optional[redis.ConnectionPool] = None
        self.redis: Optional[redis.Redis] = None

    async def connect(self):
        """Establish connection to Redis."""
        try:
            self.redis = redis.Redis.from_url(
                self.config.url,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=self.config.decode_responses,
            )

            # Test connection
            if self.redis:
                await self.redis.ping()
                logger.info("Successfully connected to Redis")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def publish_new_tickers(
        self, tickers: List[str], metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Publish new ticker selections to Redis channel.

        Args:
            tickers: List of new ticker symbols
            metadata: Optional metadata about the tickers
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        message = {
            "tickers": tickers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(tickers),
            "metadata": metadata or {},
        }

        try:
            await self.redis.publish(
                RedisChannels.TICKERS_NEW, json.dumps(message, default=str)
            )

            # Also update the active tickers list
            sadd_result = self.redis.sadd(RedisKeys.TICKER_LIST, *tickers)
            if hasattr(sadd_result, "__await__"):
                await sadd_result

            logger.info(f"Published {len(tickers)} new tickers to Redis")

        except Exception as e:
            logger.error(f"Failed to publish new tickers: {e}")
            raise

    async def publish_price_update(
        self,
        ticker: str,
        price_data: Union[MarketData, Dict[str, Any]],
        ttl: int = 300,  # 5 minutes default
    ):
        """
        Publish price update for a specific ticker.

        Args:
            ticker: Ticker symbol
            price_data: Price data (MarketData object or dict)
            ttl: Time to live in seconds
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        # Convert MarketData to dict if needed
        if isinstance(price_data, MarketData):
            data_dict = {
                "symbol": price_data.symbol,
                "timestamp": price_data.timestamp.isoformat(),
                "timeframe": price_data.timeframe.value,
                "open": float(price_data.open),
                "high": float(price_data.high),
                "low": float(price_data.low),
                "close": float(price_data.close),
                "volume": price_data.volume,
                "asset_type": price_data.asset_type.value,
            }
        else:
            data_dict = price_data

        channel = f"{RedisChannels.PRICES_PREFIX}{ticker.upper()}"

        try:
            # Publish to channel
            await self.redis.publish(channel, json.dumps(data_dict, default=str))

            # Cache the latest price with TTL
            cache_key = f"{RedisKeys.PRICE_CACHE_PREFIX}{ticker.upper()}"
            await self.redis.setex(cache_key, ttl, json.dumps(data_dict, default=str))

            logger.debug(f"Published price update for {ticker}")

        except Exception as e:
            logger.error(f"Failed to publish price update for {ticker}: {e}")
            raise

    async def publish_screener_update(
        self, screener_data: List[FinVizData], screener_type: str = "momentum"
    ):
        """
        Publish screener update to Redis.

        Args:
            screener_data: List of FinViz data
            screener_type: Type of screener
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        # Convert to serializable format
        data_list = []
        for item in screener_data:
            data_list.append(
                {
                    "symbol": item.symbol,
                    "company": item.company or "",
                    "sector": item.sector or "",
                    "industry": item.industry or "",
                    "country": item.country or "",
                    "market_cap": (
                        str(item.market_cap) if item.market_cap is not None else None
                    ),
                    "pe_ratio": (
                        float(item.pe_ratio) if item.pe_ratio is not None else None
                    ),
                    "price": float(item.price) if item.price is not None else None,
                    "change": float(item.change) if item.change is not None else None,
                    "volume": int(item.volume) if item.volume is not None else None,
                    "timestamp": (
                        item.timestamp.isoformat()
                        if item.timestamp
                        else datetime.now(timezone.utc).isoformat()
                    ),
                }
            )

        message = {
            "screener_type": screener_type,
            "data": data_list,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(data_list),
        }

        try:
            await self.redis.publish(
                RedisChannels.SCREENER_UPDATES, json.dumps(message, default=str)
            )

            # Cache screener results
            cache_key = f"{RedisKeys.SCREENER_CACHE_PREFIX}{screener_type}"
            await self.redis.setex(
                cache_key, 1800, json.dumps(message, default=str)  # 30 minutes TTL
            )

            logger.info(f"Published screener update with {len(data_list)} items")

        except Exception as e:
            logger.error(f"Failed to publish screener update: {e}")
            raise

    async def cache_market_data(
        self,
        ticker: str,
        timeframe: TimeFrame,
        data: List[MarketData],
        ttl: int = 3600,  # 1 hour default
    ):
        """
        Cache market data in Redis.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            data: List of MarketData objects
            ttl: Time to live in seconds
        """
        if not self.redis or not data:
            return

        cache_key = f"market_data:{ticker.upper()}:{timeframe.value}"

        # Convert to serializable format
        data_list = []
        for item in data:
            data_list.append(
                {
                    "timestamp": item.timestamp.isoformat(),
                    "open": float(item.open),
                    "high": float(item.high),
                    "low": float(item.low),
                    "close": float(item.close),
                    "volume": item.volume,
                }
            )

        try:
            await self.redis.setex(cache_key, ttl, json.dumps(data_list, default=str))

            logger.debug(f"Cached {len(data)} records for {ticker} {timeframe}")

        except Exception as e:
            logger.error(f"Failed to cache market data for {ticker}: {e}")

    async def get_cached_market_data(
        self, ticker: str, timeframe: TimeFrame
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached market data from Redis.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe

        Returns:
            List of cached data or None if not found
        """
        if not self.redis:
            return None

        cache_key = f"market_data:{ticker.upper()}:{timeframe.value}"

        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)

        except Exception as e:
            logger.error(f"Failed to get cached market data for {ticker}: {e}")

        return None

    async def get_cached_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get cached price data for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Cached price data or None if not found
        """
        if not self.redis:
            return None

        cache_key = f"{RedisKeys.PRICE_CACHE_PREFIX}{ticker.upper()}"

        try:
            cached_price = await self.redis.get(cache_key)
            if cached_price:
                return json.loads(cached_price)

        except Exception as e:
            logger.error(f"Failed to get cached price for {ticker}: {e}")

        return None

    async def get_active_tickers(self) -> List[str]:
        """
        Get list of active tickers from Redis.

        Returns:
            List of active ticker symbols
        """
        if not self.redis:
            return []

        try:
            tickers = self.redis.smembers(RedisKeys.TICKER_LIST)
            if hasattr(tickers, "__await__"):
                tickers = await tickers
            return list(tickers) if tickers else []

        except Exception as e:
            logger.error(f"Failed to get active tickers: {e}")
            return []

    async def remove_ticker(self, ticker: str):
        """
        Remove ticker from active list and clear its cached data.

        Args:
            ticker: Ticker symbol to remove
        """
        if not self.redis:
            return

        try:
            # Remove from active tickers set
            srem_result = self.redis.srem(RedisKeys.TICKER_LIST, ticker.upper())
            if hasattr(srem_result, "__await__"):
                await srem_result

            # Clear cached price data
            price_key = f"{RedisKeys.PRICE_CACHE_PREFIX}{ticker.upper()}"
            await self.redis.delete(price_key)

            # Clear cached market data
            pattern = f"market_data:{ticker.upper()}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)

            logger.info(f"Removed ticker {ticker} from Redis")

        except Exception as e:
            logger.error(f"Failed to remove ticker {ticker}: {e}")

    async def update_last_update_time(
        self, ticker: str, timeframe: TimeFrame, timestamp: Optional[datetime] = None
    ):
        """
        Update last update time for a ticker/timeframe combination.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            timestamp: Update timestamp (current time if None)
        """
        if not self.redis:
            return

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        key = f"{RedisKeys.LAST_UPDATE_PREFIX}{ticker.upper()}:{timeframe.value}"

        try:
            await self.redis.set(key, timestamp.isoformat())
            logger.debug(f"Updated last update time for {ticker} {timeframe}")

        except Exception as e:
            logger.error(f"Failed to update last update time for {ticker}: {e}")

    async def get_last_update_time(
        self, ticker: str, timeframe: TimeFrame
    ) -> Optional[datetime]:
        """
        Get last update time for a ticker/timeframe combination.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe

        Returns:
            Last update timestamp or None if not found
        """
        if not self.redis:
            return None

        key = f"{RedisKeys.LAST_UPDATE_PREFIX}{ticker.upper()}:{timeframe.value}"

        try:
            timestamp_str = await self.redis.get(key)
            if timestamp_str:
                return datetime.fromisoformat(timestamp_str)

        except Exception as e:
            logger.error(f"Failed to get last update time for {ticker}: {e}")

        return None

    async def publish_data_validation_alert(
        self, ticker: str, issues: List[str], severity: str = "warning"
    ):
        """
        Publish data validation alert.

        Args:
            ticker: Ticker symbol
            issues: List of validation issues
            severity: Alert severity (info, warning, error)
        """
        if not self.redis:
            return

        alert = {
            "ticker": ticker,
            "issues": issues,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_type": "data_validation",
        }

        try:
            await self.redis.publish(
                RedisChannels.DATA_VALIDATION, json.dumps(alert, default=str)
            )

            logger.info(
                f"Published data validation alert for {ticker}: {len(issues)} issues"
            )

        except Exception as e:
            logger.error(f"Failed to publish validation alert: {e}")

    async def cache_data_statistics(
        self,
        ticker: str,
        timeframe: TimeFrame,
        stats: Dict[str, Any],
        ttl: int = 7200,  # 2 hours default
    ):
        """
        Cache data statistics in Redis.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            stats: Statistics to cache
            ttl: Time to live in seconds
        """
        if not self.redis:
            return

        key = f"{RedisKeys.DATA_STATS_PREFIX}{ticker.upper()}:{timeframe.value}"

        try:
            await self.redis.setex(key, ttl, json.dumps(stats, default=str))

            logger.debug(f"Cached statistics for {ticker} {timeframe}")

        except Exception as e:
            logger.error(f"Failed to cache statistics for {ticker}: {e}")

    async def get_cached_statistics(
        self, ticker: str, timeframe: TimeFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached statistics from Redis.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe

        Returns:
            Cached statistics or None if not found
        """
        if not self.redis:
            return None

        key = f"{RedisKeys.DATA_STATS_PREFIX}{ticker.upper()}:{timeframe.value}"

        try:
            cached_stats = await self.redis.get(key)
            if cached_stats:
                return json.loads(cached_stats)

        except Exception as e:
            logger.error(f"Failed to get cached statistics for {ticker}: {e}")

        return None

    async def set_system_status(
        self, component: str, status: str, details: Optional[Dict[str, Any]] = None
    ):
        """
        Set system component status in Redis.

        Args:
            component: Component name (e.g., 'finviz_screener', 'twelvedata_client')
            status: Status (online, offline, error, maintenance)
            details: Optional status details
        """
        if not self.redis:
            return

        status_data = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }

        key = f"system_status:{component}"

        try:
            await self.redis.setex(
                key, 300, json.dumps(status_data, default=str)  # 5 minutes TTL
            )

            logger.debug(f"Updated system status for {component}: {status}")

        except Exception as e:
            logger.error(f"Failed to set system status for {component}: {e}")

    async def get_system_status(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Get system component status from Redis.

        Args:
            component: Component name

        Returns:
            Status data or None if not found
        """
        if not self.redis:
            return None

        key = f"system_status:{component}"

        try:
            status_data = await self.redis.get(key)
            if status_data:
                return json.loads(status_data)

        except Exception as e:
            logger.error(f"Failed to get system status for {component}: {e}")

        return None

    async def batch_cache_prices(
        self, price_data: Dict[str, MarketData], ttl: int = 300
    ):
        """
        Cache multiple price updates in batch.

        Args:
            price_data: Dictionary mapping tickers to price data
            ttl: Time to live in seconds
        """
        if not self.redis or not price_data:
            return

        pipeline = self.redis.pipeline()

        try:
            for ticker, data in price_data.items():
                # Convert MarketData to dict if needed
                if isinstance(data, MarketData):
                    data_dict = {
                        "symbol": data.symbol,
                        "timestamp": data.timestamp.isoformat(),
                        "price": float(data.close),
                        "open": float(data.open),
                        "high": float(data.high),
                        "low": float(data.low),
                        "volume": data.volume,
                    }
                else:
                    data_dict = data

                # Add to pipeline
                cache_key = f"{RedisKeys.PRICE_CACHE_PREFIX}{ticker.upper()}"
                pipeline.setex(cache_key, ttl, json.dumps(data_dict, default=str))

                # Also publish to individual ticker channel
                channel = f"{RedisChannels.PRICES_PREFIX}{ticker.upper()}"
                pipeline.publish(channel, json.dumps(data_dict, default=str))

            # Execute all commands
            await pipeline.execute()

            logger.info(f"Batch cached {len(price_data)} price updates")

        except Exception as e:
            logger.error(f"Failed to batch cache prices: {e}")
            raise

    async def get_batch_cached_prices(
        self, tickers: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get cached prices for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping tickers to cached price data
        """
        if not self.redis or not tickers:
            return {}

        pipeline = self.redis.pipeline()

        # Add all get commands to pipeline
        for ticker in tickers:
            cache_key = f"{RedisKeys.PRICE_CACHE_PREFIX}{ticker.upper()}"
            pipeline.get(cache_key)

        try:
            results = await pipeline.execute()

            # Parse results
            price_data = {}
            for ticker, result in zip(tickers, results):
                if result:
                    try:
                        price_data[ticker] = json.loads(result)
                    except json.JSONDecodeError:
                        price_data[ticker] = None
                else:
                    price_data[ticker] = None

            return price_data

        except Exception as e:
            logger.error(f"Failed to get batch cached prices: {e}")
            return {ticker: None for ticker in tickers}

    async def publish_market_data_update(
        self,
        ticker: str,
        timeframe: TimeFrame,
        record_count: int,
        update_type: str = "new_data",
    ):
        """
        Publish market data update notification.

        Args:
            ticker: Ticker symbol
            timeframe: Data timeframe
            record_count: Number of records updated
            update_type: Type of update (new_data, backfill, correction)
        """
        if not self.redis:
            return

        message = {
            "ticker": ticker,
            "timeframe": timeframe.value,
            "record_count": record_count,
            "update_type": update_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            await self.redis.publish(
                RedisChannels.MARKET_DATA_UPDATES, json.dumps(message, default=str)
            )

            logger.debug(
                f"Published market data update for {ticker} {timeframe}: {record_count} records"
            )

        except Exception as e:
            logger.error(f"Failed to publish market data update: {e}")

    async def subscribe_to_channel(
        self, channel: str, callback: Callable, pattern: bool = False
    ):
        """
        Subscribe to Redis channel with callback.

        Args:
            channel: Channel name or pattern
            callback: Callback function for messages
            pattern: Whether channel is a pattern
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")

        try:
            if pattern:
                pubsub = self.redis.pubsub()
                await pubsub.psubscribe(channel)
            else:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(channel)

            logger.info(
                f"Subscribed to {'pattern' if pattern else 'channel'}: {channel}"
            )

            async for message in pubsub.listen():
                if message["type"] == "message" or (
                    pattern and message["type"] == "pmessage"
                ):
                    try:
                        data = json.loads(message["data"])
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Callback error for channel {channel}: {e}")

        except Exception as e:
            logger.error(f"Subscription error for {channel}: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform Redis health check.

        Returns:
            Health check results
        """
        health_info = {
            "connected": False,
            "latency_ms": 0.0,
            "memory_usage": {},
            "error": "",
        }

        if not self.redis:
            health_info["error"] = "Redis not connected"
            return health_info

        try:
            # Measure latency
            start_time = datetime.now(timezone.utc)
            await self.redis.ping()
            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            health_info["connected"] = True
            health_info["latency_ms"] = round(latency, 2)

            # Get memory info
            info = await self.redis.info("memory")
            health_info["memory_usage"] = {
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "used_memory_peak": info.get("used_memory_peak"),
                "used_memory_peak_human": info.get("used_memory_peak_human"),
            }

        except Exception as e:
            health_info["error"] = str(e)
            logger.error(f"Redis health check failed: {e}")

        return health_info

    async def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cached data.

        Args:
            pattern: Optional pattern to match keys (clears all cache if None)
        """
        if not self.redis:
            return

        try:
            if pattern:
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
                    logger.info(f"Cleared {len(keys)} keys matching pattern: {pattern}")
            else:
                # Clear common cache keys
                patterns = [
                    f"{RedisKeys.PRICE_CACHE_PREFIX}*",
                    f"{RedisKeys.SCREENER_CACHE_PREFIX}*",
                    f"{RedisKeys.DATA_STATS_PREFIX}*",
                    "market_data:*",
                ]

                total_cleared = 0
                for pattern in patterns:
                    keys = await self.redis.keys(pattern)
                    if keys:
                        await self.redis.delete(*keys)
                        total_cleared += len(keys)

                logger.info(f"Cleared {total_cleared} cache keys")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache usage statistics.

        Returns:
            Cache statistics
        """
        if not self.redis:
            return {}

        try:
            info = await self.redis.info()

            # Count keys by type
            price_keys = await self.redis.keys(f"{RedisKeys.PRICE_CACHE_PREFIX}*")
            screener_keys = await self.redis.keys(f"{RedisKeys.SCREENER_CACHE_PREFIX}*")
            market_data_keys = await self.redis.keys("market_data:*")
            stats_keys = await self.redis.keys(f"{RedisKeys.DATA_STATS_PREFIX}*")

            return {
                "total_keys": info.get("db0", {}).get("keys", 0),
                "cache_keys": {
                    "prices": len(price_keys),
                    "screener": len(screener_keys),
                    "market_data": len(market_data_keys),
                    "statistics": len(stats_keys),
                },
                "memory_usage": {
                    "used": info.get("used_memory_human"),
                    "peak": info.get("used_memory_peak_human"),
                    "percentage": info.get("used_memory_percentage"),
                },
                "connections": info.get("connected_clients", 0),
                "commands_processed": info.get("total_commands_processed", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"error": str(e)}

    async def set_rate_limit_counter(
        self, service: str, count: int, window_seconds: int = 60
    ):
        """
        Set rate limit counter for a service.

        Args:
            service: Service name
            count: Current count
            window_seconds: Time window in seconds
        """
        if not self.redis:
            return

        key = f"rate_limit:{service}"

        try:
            await self.redis.setex(key, window_seconds, count)

        except Exception as e:
            logger.error(f"Failed to set rate limit counter for {service}: {e}")

    async def get_rate_limit_counter(self, service: str) -> int:
        """
        Get current rate limit counter for a service.

        Args:
            service: Service name

        Returns:
            Current count
        """
        if not self.redis:
            return 0

        key = f"rate_limit:{service}"

        try:
            count = await self.redis.get(key)
            return int(count) if count else 0

        except Exception as e:
            logger.error(f"Failed to get rate limit counter for {service}: {e}")
            return 0

    async def increment_rate_limit_counter(self, service: str) -> int:
        """
        Increment rate limit counter for a service.

        Args:
            service: Service name

        Returns:
            New count value
        """
        if not self.redis:
            return 0

        key = f"rate_limit:{service}"

        try:
            return await self.redis.incr(key)

        except Exception as e:
            logger.error(f"Failed to increment rate limit counter for {service}: {e}")
            return 0


# Utility functions
async def publish_ticker_discovery(
    redis_client: RedisClient,
    discovered_tickers: List[str],
    source: str = "finviz_screener",
):
    """
    Convenience function to publish ticker discovery.

    Args:
        redis_client: RedisClient instance
        discovered_tickers: List of discovered tickers
        source: Source of discovery
    """
    metadata = {"source": source, "discovery_method": "automated_screening"}

    await redis_client.publish_new_tickers(discovered_tickers, metadata)


async def cache_latest_prices(
    redis_client: RedisClient, market_data: List[MarketData], ttl: int = 300
):
    """
    Cache latest prices from market data.

    Args:
        redis_client: RedisClient instance
        market_data: List of MarketData objects
        ttl: Time to live in seconds
    """
    if not market_data:
        return

    # Group by ticker and get latest price for each
    latest_prices: Dict[str, MarketData] = {}
    for data in market_data:
        ticker = data.symbol
        if (
            ticker not in latest_prices
            or data.timestamp > latest_prices[ticker].timestamp
        ):
            latest_prices[ticker] = data

    await redis_client.batch_cache_prices(latest_prices, ttl)


# Example usage
if __name__ == "__main__":

    async def main():
        config = RedisConfig(host="localhost", port=6379, db=0, password=None)

        async with RedisClient(config) as redis_client:
            # Health check
            health = await redis_client.health_check()
            print(f"Redis health: {health}")

            # Test publishing new tickers
            test_tickers = ["AAPL", "MSFT", "GOOGL"]
            await redis_client.publish_new_tickers(test_tickers)

            # Test price update
            test_price_data = {
                "symbol": "AAPL",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": 150.25,
                "volume": 1000000,
            }
            await redis_client.publish_price_update("AAPL", test_price_data)

            # Get active tickers
            active_tickers = await redis_client.get_active_tickers()
            print(f"Active tickers: {active_tickers}")

            # Get cached prices
            cached_prices = await redis_client.get_batch_cached_prices(test_tickers)
            print(f"Cached prices: {cached_prices}")

            # Cache statistics
            cache_stats = await redis_client.get_cache_statistics()
            print(f"Cache statistics: {cache_stats}")

    asyncio.run(main())
