"""
Multi-Timeframe Data Fetcher

This module provides utilities to fetch and organize market data across
multiple timeframes for multi-timeframe analysis. It integrates with the
existing data collection infrastructure to provide synchronized data
across different time horizons.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import polars as pl

from .base_strategy import StrategyMode

logger = logging.getLogger(__name__)


@dataclass
class TimeFrameDataRequest:
    """Request specification for multi-timeframe data."""

    symbol: str
    timeframes: List[str]
    periods: int = 100  # Number of periods to fetch
    end_time: Optional[datetime] = None
    include_volume: bool = True
    include_indicators: bool = False


@dataclass
class TimeFrameDataResult:
    """Result containing multi-timeframe data."""

    symbol: str
    data: Dict[str, pl.DataFrame]
    request_time: datetime
    available_timeframes: List[str]
    missing_timeframes: List[str]
    data_quality_score: float
    metadata: Dict[str, Any]


class MultiTimeFrameDataFetcher:
    """
    Fetches and synchronizes market data across multiple timeframes.

    This class handles the complexity of fetching data for different
    timeframes, ensuring data alignment and quality for analysis.
    """

    def __init__(self, redis_client=None, data_store=None):
        """
        Initialize multi-timeframe data fetcher.

        Args:
            redis_client: Redis client for caching
            data_store: Data store for historical data
        """
        self.redis_client = redis_client
        self.data_store = data_store
        self.logger = logging.getLogger(f"{__name__}.DataFetcher")

        # Timeframe hierarchies for different strategies
        self.strategy_timeframes = {
            StrategyMode.DAY_TRADING: ["1m", "5m", "15m", "30m", "1h"],
            StrategyMode.SWING_TRADING: ["15m", "30m", "1h", "4h", "1d"],
            StrategyMode.POSITION_TRADING: ["4h", "12h", "1d", "1w", "1M"],
        }

        # Cache settings
        self.cache_ttl = {
            "1m": 60,  # 1 minute
            "5m": 300,  # 5 minutes
            "15m": 900,  # 15 minutes
            "30m": 1800,  # 30 minutes
            "1h": 3600,  # 1 hour
            "4h": 14400,  # 4 hours
            "1d": 86400,  # 1 day
            "1w": 604800,  # 1 week
            "1M": 2592000,  # 1 month
        }

    async def fetch_multi_timeframe_data(
        self,
        symbol: str,
        strategy_mode: StrategyMode,
        periods: int = 100,
        custom_timeframes: Optional[List[str]] = None,
    ) -> TimeFrameDataResult:
        """
        Fetch data for multiple timeframes based on strategy mode.

        Args:
            symbol: Trading symbol
            strategy_mode: Strategy mode to determine timeframes
            periods: Number of periods to fetch for each timeframe
            custom_timeframes: Optional custom timeframe list

        Returns:
            Multi-timeframe data result
        """
        try:
            # Determine timeframes to fetch
            timeframes = custom_timeframes or self.strategy_timeframes.get(
                strategy_mode, self.strategy_timeframes[StrategyMode.SWING_TRADING]
            )

            request = TimeFrameDataRequest(
                symbol=symbol,
                timeframes=timeframes,
                periods=periods,
                end_time=datetime.now(timezone.utc),
            )

            return await self._fetch_data_for_request(request)

        except Exception as e:
            self.logger.error(f"Error fetching multi-timeframe data for {symbol}: {e}")
            return self._create_error_result(symbol, timeframes, str(e))

    async def _fetch_data_for_request(
        self, request: TimeFrameDataRequest
    ) -> TimeFrameDataResult:
        """Fetch data for a specific request."""
        try:
            data_dict = {}
            available_timeframes = []
            missing_timeframes = []

            # Fetch data for each timeframe concurrently
            fetch_tasks = [
                self._fetch_timeframe_data(
                    request.symbol, tf, request.periods, request.end_time
                )
                for tf in request.timeframes
            ]

            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                timeframe = request.timeframes[i]

                if isinstance(result, Exception):
                    self.logger.warning(
                        f"Failed to fetch data for {request.symbol} {timeframe}: {result}"
                    )
                    missing_timeframes.append(timeframe)
                elif (
                    result is not None
                    and isinstance(result, pl.DataFrame)
                    and not result.is_empty()
                ):
                    data_dict[timeframe] = result
                    available_timeframes.append(timeframe)
                else:
                    missing_timeframes.append(timeframe)

            # Calculate data quality score - filter out BaseExceptions
            filtered_data = {
                k: v for k, v in data_dict.items() if isinstance(v, pl.DataFrame)
            }
            quality_score = self._calculate_data_quality_score(
                filtered_data, request.timeframes
            )

            return TimeFrameDataResult(
                symbol=request.symbol,
                data=filtered_data,
                request_time=datetime.now(timezone.utc),
                available_timeframes=available_timeframes,
                missing_timeframes=missing_timeframes,
                data_quality_score=quality_score,
                metadata={
                    "requested_periods": request.periods,
                    "total_timeframes_requested": len(request.timeframes),
                    "successful_fetches": len(available_timeframes),
                },
            )

        except Exception as e:
            self.logger.error(f"Error in _fetch_data_for_request: {e}")
            return self._create_error_result(request.symbol, request.timeframes, str(e))

    async def _fetch_timeframe_data(
        self, symbol: str, timeframe: str, periods: int, end_time: Optional[datetime]
    ) -> Optional[pl.DataFrame]:
        """Fetch data for a single timeframe."""
        try:
            # Try cache first
            if self.redis_client:
                cached_data = await self._get_cached_data(symbol, timeframe)
                if cached_data is not None:
                    return cached_data

            # Fetch from data store
            data = await self._fetch_from_data_store(
                symbol, timeframe, periods, end_time
            )

            if data is not None and not data.is_empty():
                # Cache the result
                if self.redis_client:
                    await self._cache_data(symbol, timeframe, data)

                return data

            # If no data available, try to fetch from external source
            return await self._fetch_from_external_source(
                symbol, timeframe, periods, end_time
            )

        except Exception as e:
            self.logger.error(f"Error fetching {symbol} {timeframe} data: {e}")
            return None

    async def _get_cached_data(
        self, symbol: str, timeframe: str
    ) -> Optional[pl.DataFrame]:
        """Get cached data from Redis."""
        try:
            if not self.redis_client:
                return None

            cache_key = f"market_data:{symbol}:{timeframe}"
            cached_json = await self.redis_client.get(cache_key)

            if cached_json:
                # Convert JSON back to Polars DataFrame
                import json

                data_dict = json.loads(cached_json)
                return pl.DataFrame(data_dict)

            return None

        except Exception as e:
            self.logger.error(
                f"Error getting cached data for {symbol} {timeframe}: {e}"
            )
            return None

    async def _cache_data(self, symbol: str, timeframe: str, data: pl.DataFrame):
        """Cache data to Redis."""
        try:
            if not self.redis_client or data.is_empty():
                return

            cache_key = f"market_data:{symbol}:{timeframe}"
            ttl = self.cache_ttl.get(timeframe, 3600)

            # Convert to JSON for caching
            data_dict = data.to_dict(as_series=False)
            cached_json = json.dumps(data_dict, default=str)

            await self.redis_client.setex(cache_key, ttl, cached_json)

        except Exception as e:
            self.logger.error(f"Error caching data for {symbol} {timeframe}: {e}")

    async def _fetch_from_data_store(
        self, symbol: str, timeframe: str, periods: int, end_time: Optional[datetime]
    ) -> Optional[pl.DataFrame]:
        """Fetch data from the data store."""
        try:
            if not self.data_store:
                return None

            # Calculate start time based on periods and timeframe
            start_time = self._calculate_start_time(timeframe, periods, end_time)

            # Fetch data from data store
            data = await self.data_store.get_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time or datetime.now(timezone.utc),
            )

            return data

        except Exception as e:
            self.logger.error(
                f"Error fetching from data store for {symbol} {timeframe}: {e}"
            )
            return None

    async def _fetch_from_external_source(
        self, symbol: str, timeframe: str, periods: int, end_time: Optional[datetime]
    ) -> Optional[pl.DataFrame]:
        """Fetch data from external data source as fallback."""
        try:
            # This would typically integrate with your external data provider
            # For now, we'll create a placeholder implementation

            self.logger.info(f"Fetching {symbol} {timeframe} from external source")

            # Placeholder: In real implementation, this would call your data provider
            # For example, TwelveData, Alpha Vantage, Yahoo Finance, etc.

            # Return None to indicate no data available
            return None

        except Exception as e:
            self.logger.error(
                f"Error fetching from external source for {symbol} {timeframe}: {e}"
            )
            return None

    def _calculate_start_time(
        self, timeframe: str, periods: int, end_time: Optional[datetime]
    ) -> datetime:
        """Calculate start time based on timeframe and periods."""
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        # Timeframe to timedelta mapping
        timeframe_deltas = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "12h": timedelta(hours=12),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1M": timedelta(days=30),  # Approximate
        }

        delta = timeframe_deltas.get(timeframe, timedelta(hours=1))
        return end_time - (delta * periods)

    def _calculate_data_quality_score(
        self, data_dict: Dict[str, pl.DataFrame], requested_timeframes: List[str]
    ) -> float:
        """Calculate data quality score based on available data."""
        try:
            if not data_dict:
                return 0.0

            total_score = 0.0
            total_weight = 0.0

            for timeframe in requested_timeframes:
                weight = 1.0  # Equal weight for now, can be adjusted
                total_weight += weight

                if timeframe in data_dict:
                    df = data_dict[timeframe]

                    # Check data completeness
                    completeness_score = 1.0 if not df.is_empty() else 0.0

                    # Check for required columns
                    required_columns = ["open", "high", "low", "close", "timestamp"]
                    has_all_columns = all(col in df.columns for col in required_columns)
                    column_score = 1.0 if has_all_columns else 0.5

                    # Check data freshness (for shorter timeframes)
                    freshness_score = self._calculate_freshness_score(df, timeframe)

                    # Combine scores
                    timeframe_score = (
                        completeness_score + column_score + freshness_score
                    ) / 3
                    total_score += timeframe_score * weight

            return (total_score / total_weight) * 100 if total_weight > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating data quality score: {e}")
            return 0.0

    def _calculate_freshness_score(self, df: pl.DataFrame, timeframe: str) -> float:
        """Calculate freshness score based on how recent the data is."""
        try:
            if df.is_empty() or "timestamp" not in df.columns:
                return 0.0

            # Get the latest timestamp
            latest_timestamp = df.select("timestamp").max().item()

            if isinstance(latest_timestamp, str):
                latest_timestamp = datetime.fromisoformat(
                    latest_timestamp.replace("Z", "+00:00")
                )

            current_time = datetime.now(timezone.utc)
            time_diff = current_time - latest_timestamp

            # Define acceptable delays for different timeframes
            acceptable_delays = {
                "1m": timedelta(minutes=5),
                "5m": timedelta(minutes=15),
                "15m": timedelta(minutes=30),
                "30m": timedelta(hours=1),
                "1h": timedelta(hours=2),
                "4h": timedelta(hours=6),
                "12h": timedelta(hours=24),
                "1d": timedelta(days=2),
                "1w": timedelta(weeks=1),
                "1M": timedelta(days=7),
            }

            acceptable_delay = acceptable_delays.get(timeframe, timedelta(hours=2))

            if time_diff <= acceptable_delay:
                return 1.0
            elif time_diff <= acceptable_delay * 2:
                return 0.7
            elif time_diff <= acceptable_delay * 5:
                return 0.4
            else:
                return 0.1

        except Exception as e:
            self.logger.error(f"Error calculating freshness score: {e}")
            return 0.5

    def _create_error_result(
        self, symbol: str, timeframes: List[str], error_message: str
    ) -> TimeFrameDataResult:
        """Create error result for failed data fetch."""
        return TimeFrameDataResult(
            symbol=symbol,
            data={},
            request_time=datetime.now(timezone.utc),
            available_timeframes=[],
            missing_timeframes=timeframes,
            data_quality_score=0.0,
            metadata={
                "error": error_message,
                "total_timeframes_requested": len(timeframes),
                "successful_fetches": 0,
            },
        )

    async def validate_data_alignment(
        self, data_dict: Dict[str, pl.DataFrame]
    ) -> Dict[str, Any]:
        """
        Validate that data across timeframes is properly aligned.

        Args:
            data_dict: Dictionary of timeframe data

        Returns:
            Validation results with alignment information
        """
        try:
            validation_results: Dict[str, Any] = {
                "is_aligned": True,
                "alignment_issues": [],
                "timestamp_ranges": {},
                "data_gaps": {},
            }

            # Check timestamp alignment across timeframes
            for timeframe, df in data_dict.items():
                if df.is_empty() or "timestamp" not in df.columns:
                    validation_results["alignment_issues"].append(
                        f"No timestamp data for {timeframe}"
                    )
                    validation_results["is_aligned"] = False
                    continue

                # Get timestamp range
                timestamps = df.select("timestamp")
                min_ts = timestamps.min().item()
                max_ts = timestamps.max().item()
                validation_results["timestamp_ranges"][timeframe] = {
                    "start": min_ts,
                    "end": max_ts,
                    "count": df.height,
                }

                # Check for gaps in data
                if df.height > 1:
                    # Simple gap detection - can be enhanced based on timeframe
                    expected_intervals = self._get_expected_intervals(timeframe)
                    if expected_intervals:
                        gaps = self._detect_gaps(df, expected_intervals)
                        if gaps:
                            validation_results["data_gaps"][timeframe] = gaps

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating data alignment: {e}")
            return {
                "is_aligned": False,
                "alignment_issues": [f"Validation error: {str(e)}"],
                "timestamp_ranges": {},
                "data_gaps": {},
            }

    def _get_expected_intervals(self, timeframe: str) -> Optional[timedelta]:
        """Get expected time interval for a timeframe."""
        intervals = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "12h": timedelta(hours=12),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1M": timedelta(days=30),
        }
        return intervals.get(timeframe)

    def _detect_gaps(
        self, df: pl.DataFrame, expected_interval: timedelta
    ) -> List[Dict[str, Any]]:
        """Detect gaps in time series data."""
        try:
            gaps = []

            # Sort by timestamp to ensure proper order
            df_sorted = df.sort("timestamp")
            timestamps = df_sorted.select("timestamp").to_series().to_list()

            for i in range(1, len(timestamps)):
                current = timestamps[i]
                previous = timestamps[i - 1]

                # Convert to datetime if needed
                if isinstance(current, str):
                    current = datetime.fromisoformat(current.replace("Z", "+00:00"))
                if isinstance(previous, str):
                    previous = datetime.fromisoformat(previous.replace("Z", "+00:00"))

                actual_interval = current - previous

                # Allow some tolerance (50% extra)
                max_allowed = expected_interval * 1.5

                if actual_interval > max_allowed:
                    gaps.append(
                        {
                            "start": previous.isoformat(),
                            "end": current.isoformat(),
                            "duration": str(actual_interval),
                            "expected": str(expected_interval),
                        }
                    )

            return gaps

        except Exception as e:
            self.logger.error(f"Error detecting gaps: {e}")
            return []


# Convenience function for easy integration
def create_multi_timeframe_fetcher(
    redis_client=None, data_store=None
) -> MultiTimeFrameDataFetcher:
    """
    Create multi-timeframe data fetcher with optional dependencies.

    Args:
        redis_client: Optional Redis client for caching
        data_store: Optional data store for historical data

    Returns:
        MultiTimeFrameDataFetcher instance
    """
    return MultiTimeFrameDataFetcher(redis_client=redis_client, data_store=data_store)
