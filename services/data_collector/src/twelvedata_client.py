"""
TwelveData API client for fetching OHLCV data with multiple timeframes.

This module provides a comprehensive interface to TwelveData API with support
for historical data, real-time data, batch requests, and proper rate limiting.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any

import aiohttp
import polars as pl
from pydantic import BaseModel, Field

from shared.models import MarketData, TimeFrame, AssetType


logger = logging.getLogger(__name__)


class TwelveDataConfig(BaseModel):
    """TwelveData API configuration."""

    api_key: str = Field(..., description="TwelveData API key")
    base_url: str = Field(default="https://api.twelvedata.com", description="Base API URL")
    rate_limit_requests: int = Field(default=800, description="Requests per minute")
    rate_limit_period: int = Field(default=60, description="Rate limit period in seconds")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    batch_size: int = Field(default=120, description="Maximum symbols per batch request")


class TwelveDataRateLimiter:
    """Rate limiter for TwelveData API calls."""

    def __init__(self, requests_per_period: int = 800, period: int = 60):
        self.requests_per_period = requests_per_period
        self.period = period
        self.requests = []
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if necessary to respect rate limiting."""
        async with self._lock:
            current_time = time.time()

            # Remove requests older than the period
            self.requests = [req_time for req_time in self.requests
                           if current_time - req_time < self.period]

            # Check if we need to wait
            if len(self.requests) >= self.requests_per_period:
                oldest_request = min(self.requests)
                wait_time = self.period - (current_time - oldest_request)
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)

            # Record this request
            self.requests.append(current_time)


class TwelveDataResponse(BaseModel):
    """TwelveData API response model."""

    data: List[Dict[str, Any]]
    symbol: str
    timeframe: str
    status: str = "ok"
    error_message: Optional[str] = None
    request_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TwelveDataClient:
    """
    TwelveData API client for fetching OHLCV data.

    Supports multiple timeframes, batch requests, rate limiting,
    and both historical and real-time data.
    """

    def __init__(self, config: TwelveDataConfig, session: Optional[aiohttp.ClientSession] = None):
        self.config = config
        self._session = session
        self._rate_limiter = TwelveDataRateLimiter(
            config.rate_limit_requests,
            config.rate_limit_period
        )

        # Supported timeframes mapping
        self.timeframe_map = {
            TimeFrame.FIVE_MINUTES: "5min",
            TimeFrame.FIFTEEN_MINUTES: "15min",
            TimeFrame.ONE_HOUR: "1h",
            TimeFrame.ONE_DAY: "1day",
            TimeFrame.ONE_WEEK: "1week",
            TimeFrame.ONE_MONTH: "1month"
        }

    async def __aenter__(self):
        """Async context manager entry."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        retries: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request to TwelveData API with error handling.

        Args:
            endpoint: API endpoint
            params: Request parameters
            retries: Current retry attempt

        Returns:
            JSON response data

        Raises:
            aiohttp.ClientError: On HTTP errors
            ValueError: On API errors
        """
        await self._rate_limiter.wait_if_needed()

        session = await self._get_session()
        url = f"{self.config.base_url}/{endpoint}"

        # Add API key to parameters
        params["apikey"] = self.config.api_key

        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                # Check for API errors
                if isinstance(data, dict):
                    if "status" in data and data["status"] == "error":
                        error_msg = data.get("message", "Unknown API error")
                        raise ValueError(f"TwelveData API error: {error_msg}")

                return data

        except aiohttp.ClientError as e:
            if retries < self.config.max_retries:
                wait_time = 2 ** retries
                logger.warning(f"Request failed (attempt {retries + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                return await self._make_request(endpoint, params, retries + 1)

            logger.error(f"Request failed after {self.config.max_retries} retries: {e}")
            raise

    async def get_time_series(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        outputsize: int = 5000
    ) -> List[MarketData]:
        """
        Get time series data for a single symbol.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for historical data
            end_date: End date for historical data
            outputsize: Maximum number of data points

        Returns:
            List of MarketData objects
        """
        interval = self.timeframe_map.get(timeframe, "5min")

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "outputsize": min(outputsize, 5000),  # TwelveData limit
            "format": "JSON"
        }

        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching {interval} data for {symbol}")

        try:
            response = await self._make_request("time_series", params)
            return self._parse_time_series_response(response, symbol, timeframe)

        except Exception as e:
            logger.error(f"Failed to fetch time series for {symbol}: {e}")
            raise

    async def get_batch_time_series(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, List[MarketData]]:
        """
        Get time series data for multiple symbols in batch.

        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            Dictionary mapping symbols to MarketData lists
        """
        # Split symbols into batches to respect API limits
        batches = [symbols[i:i + self.config.batch_size]
                  for i in range(0, len(symbols), self.config.batch_size)]

        all_results = {}

        for batch_num, batch_symbols in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch_symbols)} symbols")

            # Create concurrent tasks for this batch
            tasks = [
                self.get_time_series(symbol, timeframe, start_date, end_date)
                for symbol in batch_symbols
            ]

            # Execute batch with some delay between batches
            if batch_num > 1:
                await asyncio.sleep(1.0)  # Small delay between batches

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for symbol, result in zip(batch_symbols, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch data for {symbol}: {result}")
                    all_results[symbol] = []
                else:
                    all_results[symbol] = result

        return all_results

    async def get_real_time_price(self, symbol: str) -> Optional[MarketData]:
        """
        Get real-time price data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            MarketData with current price information
        """
        params = {
            "symbol": symbol.upper(),
            "format": "JSON"
        }

        try:
            response = await self._make_request("price", params)
            return self._parse_real_time_response(response, symbol)

        except Exception as e:
            logger.error(f"Failed to fetch real-time price for {symbol}: {e}")
            return None

    async def get_batch_real_time_prices(self, symbols: List[str]) -> Dict[str, Optional[MarketData]]:
        """
        Get real-time prices for multiple symbols.

        Args:
            symbols: List of trading symbols

        Returns:
            Dictionary mapping symbols to MarketData
        """
        # TwelveData supports batch real-time requests
        symbol_string = ",".join(symbol.upper() for symbol in symbols)

        params = {
            "symbol": symbol_string,
            "format": "JSON"
        }

        try:
            response = await self._make_request("price", params)
            return self._parse_batch_real_time_response(response, symbols)

        except Exception as e:
            logger.error(f"Failed to fetch batch real-time prices: {e}")
            return {symbol: None for symbol in symbols}  # type: ignore

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        years: int = 2
    ) -> List[MarketData]:
        """
        Get historical data for backtesting.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            years: Number of years of historical data

        Returns:
            List of MarketData objects
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=years * 365)

        return await self.get_time_series(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            outputsize=5000
        )

    async def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[TimeFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[TimeFrame, List[MarketData]]:
        """
        Get data for multiple timeframes for a single symbol.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to fetch
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            Dictionary mapping timeframes to MarketData lists
        """
        tasks = [
            self.get_time_series(symbol, tf, start_date, end_date)
            for tf in timeframes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for timeframe, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {timeframe} data for {symbol}: {result}")
                output[timeframe] = []
            else:
                output[timeframe] = result

        return output

    def _parse_time_series_response(
        self,
        response: Dict[str, Any],
        symbol: str,
        timeframe: TimeFrame
    ) -> List[MarketData]:
        """
        Parse time series response into MarketData objects.

        Args:
            response: Raw API response
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            List of MarketData objects
        """
        market_data = []

        try:
            # Handle different response formats
            if "values" in response:
                data_points = response["values"]
            elif isinstance(response, list):
                data_points = response
            else:
                data_points = [response]

            for point in data_points:
                try:
                    # Ensure point is a dictionary
                    if not isinstance(point, dict):
                        continue

                    # Parse timestamp
                    timestamp_str = point.get("datetime") or point.get("timestamp")
                    if not timestamp_str:
                        continue

                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                    # Parse OHLCV data
                    open_price = Decimal(str(point.get("open", 0)))
                    high_price = Decimal(str(point.get("high", 0)))
                    low_price = Decimal(str(point.get("low", 0)))
                    close_price = Decimal(str(point.get("close", 0)))
                    volume = int(float(point.get("volume", 0)))

                    market_data.append(MarketData(
                        symbol=symbol.upper(),
                        timestamp=timestamp,
                        timeframe=timeframe,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        close=close_price,
                        volume=volume,
                        adjusted_close=close_price,
                        asset_type=AssetType.STOCK
                    ))

                except Exception as e:
                    logger.warning(f"Failed to parse data point for {symbol}: {e}")
                    continue

            # Sort by timestamp (oldest first)
            market_data.sort(key=lambda x: x.timestamp)

        except Exception as e:
            logger.error(f"Failed to parse time series response for {symbol}: {e}")

        return market_data

    def _parse_real_time_response(
        self,
        response: Dict[str, Any],
        symbol: str
    ) -> Optional[MarketData]:
        """
        Parse real-time price response into MarketData object.

        Args:
            response: Raw API response
            symbol: Trading symbol

        Returns:
            MarketData object or None if parsing fails
        """
        try:
            price = Decimal(str(response.get("price", 0)))

            # For real-time data, we often only get price
            # Use price for all OHLC values as approximation
            current_time = datetime.now(timezone.utc)

            return MarketData(
                symbol=symbol.upper(),
                timestamp=current_time,
                timeframe=TimeFrame.ONE_MINUTE,  # Real-time approximation
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0,  # Volume not available in real-time price endpoint
                adjusted_close=price,
                asset_type=AssetType.STOCK
            )

        except Exception as e:
            logger.error(f"Failed to parse real-time response for {symbol}: {e}")
            return None

    def _parse_batch_real_time_response(
        self,
        response: Dict[str, Any],
        symbols: List[str]
    ) -> Dict[str, Optional[MarketData]]:
        """
        Parse batch real-time price response.

        Args:
            response: Raw API response
            symbols: List of symbols requested

        Returns:
            Dictionary mapping symbols to MarketData
        """
        results = {}

        try:
            # Handle different response formats
            if isinstance(response, dict) and "price" in response:
                # Single symbol response
                symbol = symbols[0] if symbols else "UNKNOWN"
                results[symbol] = self._parse_real_time_response(response, symbol)

            if isinstance(response, list):
                # Multiple symbols in list
                for i, data in enumerate(response):
                    if i < len(symbols):
                        symbol = symbols[i]
                        if isinstance(data, dict):
                            results[symbol] = self._parse_real_time_response(data, symbol)
                        else:
                            results[symbol] = None

            elif isinstance(response, dict):
                # Response might be keyed by symbol
                for symbol in symbols:
                    if symbol.upper() in response:
                        symbol_data = response[symbol.upper()]
                        if isinstance(symbol_data, dict):
                            results[symbol] = self._parse_real_time_response(symbol_data, symbol)
                        else:
                            results[symbol] = None
                    else:
                        results[symbol] = None

        except Exception as e:
            logger.error(f"Failed to parse batch real-time response: {e}")
            # Return None for all symbols on parsing error
            results: Dict[str, Optional[MarketData]] = {symbol: None for symbol in symbols}

        # Ensure all requested symbols are in results
        for symbol in symbols:
            if symbol not in results:
                results[symbol] = None

        return results

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed quote data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote data dictionary or None if failed
        """
        params = {
            "symbol": symbol.upper(),
            "format": "JSON"
        }

        try:
            response = await self._make_request("quote", params)
            return response

        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return None

    async def get_market_state(self) -> Dict[str, Any]:
        """
        Get current market state information.

        Returns:
            Market state data
        """
        try:
            response = await self._make_request("market_state", {"format": "JSON"})
            return response

        except Exception as e:
            logger.error(f"Failed to fetch market state: {e}")
            return {}

    async def search_instruments(
        self,
        query: str,
        exchange: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for trading instruments.

        Args:
            query: Search query
            exchange: Optional exchange filter

        Returns:
            List of matching instruments
        """
        params = {
            "symbol": query,
            "format": "JSON"
        }

        if exchange:
            params["exchange"] = exchange

        try:
            response = await self._make_request("symbol_search", params)
            return response.get("data", []) if isinstance(response, dict) else []

        except Exception as e:
            logger.error(f"Failed to search instruments for '{query}': {e}")
            return []

    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate if symbols are supported by TwelveData.

        Args:
            symbols: List of symbols to validate

        Returns:
            Dictionary mapping symbols to validation status
        """
        results = {}

        # Use batch quote requests for validation
        batch_size = 10  # Smaller batch for validation
        batches = [symbols[i:i + batch_size]
                  for i in range(0, len(symbols), batch_size)]

        for batch in batches:
            for symbol in batch:
                try:
                    quote = await self.get_quote(symbol)
                    results[symbol] = quote is not None and "price" in (quote or {})

                except Exception:
                    results[symbol] = False

                # Small delay between symbol validations
                await asyncio.sleep(0.1)

        return results

    async def get_technical_indicators(
        self,
        symbol: str,
        indicator: str,
        timeframe: TimeFrame,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get technical indicator data.

        Args:
            symbol: Trading symbol
            indicator: Technical indicator name (e.g., 'sma', 'rsi', 'macd')
            timeframe: Data timeframe
            **kwargs: Additional indicator parameters

        Returns:
            Technical indicator data or None if failed
        """
        interval = self.timeframe_map.get(timeframe, "5min")

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "format": "JSON",
            **kwargs
        }

        try:
            response = await self._make_request(indicator, params)
            return response

        except Exception as e:
            logger.error(f"Failed to fetch {indicator} for {symbol}: {e}")
            return None

    def to_polars_dataframe(
        self,
        market_data: List[MarketData]
    ) -> pl.DataFrame:
        """
        Convert MarketData list to Polars DataFrame.

        Args:
            market_data: List of MarketData objects

        Returns:
            Polars DataFrame
        """
        if not market_data:
            return pl.DataFrame()

        # Convert to list of dictionaries
        data_dicts = []
        for md in market_data:
            data_dicts.append({
                "symbol": md.symbol,
                "timestamp": md.timestamp,
                "timeframe": md.timeframe.value,
                "open": float(md.open),
                "high": float(md.high),
                "low": float(md.low),
                "close": float(md.close),
                "volume": md.volume,
                "asset_type": md.asset_type.value
            })

        return pl.DataFrame(data_dicts)

    async def get_earnings_calendar(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get earnings calendar data.

        Args:
            start_date: Start date for earnings
            end_date: End date for earnings

        Returns:
            List of earnings events
        """
        params = {"format": "JSON"}

        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")

        try:
            response = await self._make_request("earnings_calendar", params)
            return response.get("earnings", []) if isinstance(response, dict) else []

        except Exception as e:
            logger.error(f"Failed to fetch earnings calendar: {e}")
            return []

    async def get_economic_calendar(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get economic calendar data.

        Args:
            start_date: Start date for events
            end_date: End date for events

        Returns:
            List of economic events
        """
        params = {"format": "JSON"}

        if start_date:
            params["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["end_date"] = end_date.strftime("%Y-%m-%d")

        try:
            response = await self._make_request("economic_calendar", params)
            return response.get("events", []) if isinstance(response, dict) else []

        except Exception as e:
            logger.error(f"Failed to fetch economic calendar: {e}")
            return []

    def get_supported_timeframes(self) -> List[TimeFrame]:
        """Get list of supported timeframes."""
        return list(self.timeframe_map.keys())

    def estimate_api_usage(
        self,
        symbols: List[str],
        timeframes: List[TimeFrame],
        historical_years: int = 2
    ) -> Dict[str, int]:
        """
        Estimate API usage for a data collection plan.

        Args:
            symbols: List of symbols
            timeframes: List of timeframes
            historical_years: Years of historical data

        Returns:
            Dictionary with usage estimates
        """
        # Rough estimates based on TwelveData limits
        historical_requests = len(symbols) * len(timeframes)

        # Estimate daily update requests (trading days per year ≈ 252)
        daily_requests_per_year = len(symbols) * len(timeframes) * 252

        # Real-time requests (assuming every 5 minutes during market hours)
        realtime_requests_per_day = len(symbols) * 12 * 6.5  # 78 requests per symbol per day
        realtime_requests_per_year = realtime_requests_per_day * 252

        return {
            "initial_historical_requests": historical_requests,
            "daily_update_requests": len(symbols) * len(timeframes),
            "yearly_update_requests": daily_requests_per_year,
            "daily_realtime_requests": int(realtime_requests_per_day),
            "yearly_realtime_requests": int(realtime_requests_per_year),
            "total_yearly_estimate": historical_requests + daily_requests_per_year + int(realtime_requests_per_year)
        }

    async def test_connection(self) -> bool:
        """
        Test connection to TwelveData API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try a simple quote request for a major stock
            quote = await self.get_quote("AAPL")
            return quote is not None and "price" in quote

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def get_api_usage(self) -> Dict[str, Any]:
        """
        Get current API usage statistics.

        Returns:
            API usage information
        """
        try:
            response = await self._make_request("api_usage", {"format": "JSON"})
            return response

        except Exception as e:
            logger.error(f"Failed to fetch API usage: {e}")
            return {}


# Utility functions for common operations
async def fetch_latest_data_for_symbols(
    client: TwelveDataClient,
    symbols: List[str],
    timeframe: TimeFrame = TimeFrame.FIVE_MINUTES,
    days_back: int = 5
) -> pl.DataFrame:
    """
    Fetch latest data for multiple symbols and return as Polars DataFrame.

    Args:
        client: TwelveDataClient instance
        symbols: List of symbols to fetch
        timeframe: Data timeframe
        days_back: Number of days of data to fetch

    Returns:
        Combined Polars DataFrame with all symbols
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    # Fetch data for all symbols
    batch_data = await client.get_batch_time_series(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    # Combine all data into single list
    all_data = []
    for symbol, data_list in batch_data.items():
        all_data.extend(data_list)

    # Convert to Polars DataFrame
    return client.to_polars_dataframe(all_data)


async def update_symbol_data(
    client: TwelveDataClient,
    symbol: str,
    timeframes: List[TimeFrame],
    last_update: Optional[datetime] = None
) -> Dict[TimeFrame, List[MarketData]]:
    """
    Update data for a single symbol across multiple timeframes.

    Args:
        client: TwelveDataClient instance
        symbol: Symbol to update
        timeframes: List of timeframes to update
        last_update: Last update timestamp (fetch from this point)

    Returns:
        Dictionary mapping timeframes to new data
    """
    if last_update is None:
        # Default to last 24 hours
        last_update = datetime.now(timezone.utc) - timedelta(hours=24)

    return await client.get_multi_timeframe_data(
        symbol=symbol,
        timeframes=timeframes,
        start_date=last_update,
        end_date=datetime.now(timezone.utc)
    )


# Example usage
if __name__ == "__main__":
    async def main():
        config = TwelveDataConfig(
            api_key="your_api_key_here",
            rate_limit_requests=800
        )

        async with TwelveDataClient(config) as client:
            # Test connection
            if not await client.test_connection():
                print("Failed to connect to TwelveData API")
                return

            # Get some sample data
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

            # Fetch 5-minute data for the last week
            df = await fetch_latest_data_for_symbols(
                client, symbols, TimeFrame.FIVE_MINUTES, days_back=7
            )

            print(f"Fetched data shape: {df.shape}")
            print(f"Symbols: {df['symbol'].unique().to_list()}")

            # Get real-time prices
            real_time_prices = await client.get_batch_real_time_prices(symbols)

            print("\nReal-time prices:")
            for symbol, data in real_time_prices.items():
                if data:
                    print(f"  {symbol}: ${data.close}")

    asyncio.run(main())
