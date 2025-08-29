"""
HTTP client for Data Collector Service API.

This client provides methods to communicate with the data collector service
through HTTP API calls instead of direct imports, maintaining proper
microservices architecture.
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import polars as pl
from decimal import Decimal

from shared.models import TimeFrame


logger = logging.getLogger(__name__)


class DataCollectorClient:
    """HTTP client for Data Collector Service API."""

    def __init__(self, base_url: str = "http://localhost:9101", timeout: float = 30.0):
        """
        Initialize the data collector client.

        Args:
            base_url: Base URL of the data collector service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the data collector service.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON body data

        Returns:
            JSON response data

        Raises:
            aiohttp.ClientError: On HTTP errors
            ValueError: On API errors
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        try:
            async with session.request(method, url, params=params, json=json_data) as response:
                response.raise_for_status()
                data = await response.json()

                # Check for API errors
                if isinstance(data, dict) and data.get("status") == "error":
                    error_msg = data.get("message", "Unknown API error")
                    raise ValueError(f"Data Collector API error: {error_msg}")

                return data

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {method} {url} - {e}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            raise

    async def get_historical_data(
        self,
        symbol: str,
        days: int = 30,
        timeframe: str = "1d"
    ) -> Optional[pl.DataFrame]:
        """
        Get historical market data for a symbol.

        Args:
            symbol: Trading symbol
            days: Number of days of data
            timeframe: Data timeframe ('1d', '1h', '5m', etc.)

        Returns:
            Polars DataFrame with historical data or None if no data
        """
        try:
            endpoint = f"/market-data/historical/{symbol}"
            params = {
                "days": days,
                "timeframe": timeframe
            }

            response = await self._make_request("GET", endpoint, params=params)

            if response.get("status") == "success" and response.get("data"):
                data = response["data"]

                # Convert to DataFrame
                if data:
                    # Parse timestamps
                    for record in data:
                        if record.get("timestamp"):
                            record["timestamp"] = datetime.fromisoformat(record["timestamp"].replace('Z', '+00:00'))

                    return pl.DataFrame(data)

            return None

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    async def get_latest_data(
        self,
        symbol: str,
        limit: int = 1,
        timeframe: str = "1d"
    ) -> Optional[pl.DataFrame]:
        """
        Get latest market data for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of latest records
            timeframe: Data timeframe

        Returns:
            Polars DataFrame with latest data or None if no data
        """
        try:
            endpoint = f"/market-data/latest/{symbol}"
            params = {
                "limit": limit,
                "timeframe": timeframe
            }

            response = await self._make_request("GET", endpoint, params=params)

            if response.get("status") == "success" and response.get("data"):
                data = response["data"]

                if data:
                    # Parse timestamps
                    for record in data:
                        if record.get("timestamp"):
                            record["timestamp"] = datetime.fromisoformat(record["timestamp"].replace('Z', '+00:00'))

                    return pl.DataFrame(data)

            return None

        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return None

    async def get_symbol_volatility(
        self,
        symbol: str,
        days: int = 252
    ) -> float:
        """
        Get volatility for a symbol.

        Args:
            symbol: Trading symbol
            days: Number of days for calculation

        Returns:
            Annualized volatility as a float
        """
        try:
            endpoint = f"/market-data/volatility/{symbol}"
            params = {"days": days}

            response = await self._make_request("GET", endpoint, params=params)

            if response.get("status") == "success":
                return float(response.get("volatility", 0.25))

            return 0.25  # Default volatility

        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.25

    async def get_symbol_correlation(
        self,
        symbol1: str,
        symbol2: str,
        days: int = 252
    ) -> float:
        """
        Get correlation between two symbols.

        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            days: Number of days for calculation

        Returns:
            Correlation coefficient (-1 to 1)
        """
        try:
            endpoint = f"/market-data/correlation/{symbol1}/{symbol2}"
            params = {"days": days}

            response = await self._make_request("GET", endpoint, params=params)

            if response.get("status") == "success":
                return float(response.get("correlation", 0.0))

            return 0.0  # Default correlation

        except Exception as e:
            logger.error(f"Error getting correlation between {symbol1} and {symbol2}: {e}")
            return 0.0

    async def get_atr(
        self,
        symbol: str,
        period: int = 14,
        days: int = 30
    ) -> float:
        """
        Get Average True Range for a symbol.

        Args:
            symbol: Trading symbol
            period: ATR period
            days: Number of days of data to use

        Returns:
            ATR as a decimal percentage (e.g., 0.02 for 2%)
        """
        try:
            endpoint = f"/market-data/atr/{symbol}"
            params = {
                "period": period,
                "days": days
            }

            response = await self._make_request("GET", endpoint, params=params)

            if response.get("status") == "success":
                return float(response.get("atr", 0.02))

            return 0.02  # Default ATR

        except Exception as e:
            logger.error(f"Error getting ATR for {symbol}: {e}")
            return 0.02

    async def load_market_data(
        self,
        ticker: str,
        timeframe: TimeFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Load market data (compatible with DataStore interface).

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            start_date: Start date for data range
            end_date: End date for data range
            limit: Maximum number of records to return

        Returns:
            Polars DataFrame with market data
        """
        try:
            # Calculate days from date range
            if start_date and end_date:
                days = (end_date - start_date).days
            elif start_date:
                days = (date.today() - start_date).days
            elif end_date:
                days = 30  # Default if only end_date provided
            else:
                days = 30  # Default

            # Map TimeFrame to string
            timeframe_map = {
                TimeFrame.ONE_MINUTE: "1m",
                TimeFrame.FIVE_MINUTES: "5m",
                TimeFrame.FIFTEEN_MINUTES: "15m",
                TimeFrame.THIRTY_MINUTES: "30m",
                TimeFrame.ONE_HOUR: "1h",
                TimeFrame.ONE_DAY: "1d",
                TimeFrame.ONE_WEEK: "1w",
                TimeFrame.ONE_MONTH: "1M"
            }

            timeframe_str = timeframe_map.get(timeframe, "1d")

            # Get data
            if limit and limit <= 100:
                # Use latest data endpoint for small limits
                df = await self.get_latest_data(ticker, limit=limit, timeframe=timeframe_str)
            else:
                # Use historical data endpoint
                df = await self.get_historical_data(ticker, days=days, timeframe=timeframe_str)

            if df is None:
                # Return empty DataFrame with expected schema
                return pl.DataFrame({
                    "symbol": [],
                    "timestamp": [],
                    "timeframe": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                    "asset_type": []
                })

            # Add missing columns to match DataStore interface
            df = df.with_columns([
                pl.lit(ticker.upper()).alias("symbol"),
                pl.lit(timeframe.value).alias("timeframe"),
                pl.lit("STOCK").alias("asset_type")
            ])

            # Apply date filters if specified
            if start_date or end_date:
                if "timestamp" in df.columns:
                    if start_date:
                        start_datetime = datetime.combine(start_date, datetime.min.time())
                        df = df.filter(pl.col("timestamp") >= start_datetime)
                    if end_date:
                        end_datetime = datetime.combine(end_date, datetime.max.time())
                        df = df.filter(pl.col("timestamp") <= end_datetime)

            # Apply limit if specified
            if limit:
                df = df.head(limit)

            return df

        except Exception as e:
            logger.error(f"Error loading market data for {ticker}: {e}")
            # Return empty DataFrame with expected schema
            return pl.DataFrame({
                "symbol": [],
                "timestamp": [],
                "timeframe": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "asset_type": []
            })

    async def get_latest_data_compatible(
        self,
        ticker: str,
        timeframe: TimeFrame,
        limit: int = 100
    ) -> pl.DataFrame:
        """
        Get latest data (compatible with DataStore interface).

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            limit: Maximum number of records to return

        Returns:
            Polars DataFrame with latest data
        """
        return await self.load_market_data(ticker, timeframe, limit=limit)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the data collector service is healthy.

        Returns:
            Health check response
        """
        try:
            response = await self._make_request("GET", "/health")
            return response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get service status information.

        Returns:
            Service status response
        """
        try:
            response = await self._make_request("GET", "/status")
            return response
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def trigger_market_data_update(self, timeframe: str = "5m") -> Dict[str, Any]:
        """
        Trigger market data update.

        Args:
            timeframe: Timeframe to update

        Returns:
            Update response
        """
        try:
            json_data = {"timeframe": timeframe}
            response = await self._make_request("POST", "/market-data/update", json_data=json_data)
            return response
        except Exception as e:
            logger.error(f"Market data update failed: {e}")
            return {"status": "error", "message": str(e)}

    async def trigger_finviz_scan(self) -> Dict[str, Any]:
        """
        Trigger FinViz screener scan.

        Returns:
            Scan response
        """
        try:
            response = await self._make_request("POST", "/finviz/scan")
            return response
        except Exception as e:
            logger.error(f"FinViz scan failed: {e}")
            return {"status": "error", "message": str(e)}

    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            # Schedule cleanup for next event loop iteration
            asyncio.create_task(self.close())


    async def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote via TwelveData relay.

        Args:
            symbol: Trading symbol

        Returns:
            Quote data or None if failed
        """
        try:
            endpoint = f"/api/twelvedata/quote/{symbol}"
            response = await self._make_request("GET", endpoint)

            if response.get("status") == "success":
                return response.get("data")

            return None

        except Exception as e:
            logger.error(f"Error getting real-time quote for {symbol}: {e}")
            return None

    async def get_twelvedata_time_series(
        self,
        symbol: str,
        interval: str = "5min",
        outputsize: int = 30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get time series data via TwelveData relay.

        Args:
            symbol: Trading symbol
            interval: Time interval
            outputsize: Number of data points
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Time series data or None if failed
        """
        try:
            endpoint = f"/api/twelvedata/time-series/{symbol}"
            params = {
                "interval": interval,
                "outputsize": outputsize
            }

            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            response = await self._make_request("GET", endpoint, params=params)

            if response.get("status") == "success":
                return response

            return None

        except Exception as e:
            logger.error(f"Error getting time series for {symbol}: {e}")
            return None

    async def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get batch quotes via TwelveData relay.

        Args:
            symbols: List of trading symbols

        Returns:
            Dictionary mapping symbols to quote data
        """
        try:
            endpoint = "/api/twelvedata/batch-quotes"
            json_data = {"symbols": symbols}

            response = await self._make_request("POST", endpoint, json_data=json_data)

            if response.get("status") == "success":
                return response.get("data", {})

            return {symbol: None for symbol in symbols}

        except Exception as e:
            logger.error(f"Error getting batch quotes: {e}")
            return {symbol: None for symbol in symbols}

    async def search_symbols(
        self,
        query: str,
        exchange: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search symbols via TwelveData relay.

        Args:
            query: Search query
            exchange: Optional exchange filter

        Returns:
            List of matching symbols
        """
        try:
            endpoint = f"/api/twelvedata/search/{query}"
            params = {}
            if exchange:
                params["exchange"] = exchange

            response = await self._make_request("GET", endpoint, params=params)

            if response.get("status") == "success":
                return response.get("data", [])

            return []

        except Exception as e:
            logger.error(f"Error searching symbols for '{query}': {e}")
            return []

    async def get_technical_indicator(
        self,
        indicator: str,
        symbol: str,
        interval: str = "1day",
        time_period: int = 9,
        series_type: str = "close"
    ) -> Optional[Dict[str, Any]]:
        """
        Get technical indicator data via TwelveData relay.

        Args:
            indicator: Technical indicator name
            symbol: Trading symbol
            interval: Time interval
            time_period: Period for calculation
            series_type: Price series type

        Returns:
            Technical indicator data or None if failed
        """
        try:
            endpoint = f"/api/twelvedata/technical/{indicator}/{symbol}"
            params = {
                "interval": interval,
                "time_period": time_period,
                "series_type": series_type
            }

            response = await self._make_request("GET", endpoint, params=params)

            if response.get("status") == "success":
                return response.get("data")

            return None

        except Exception as e:
            logger.error(f"Error getting {indicator} for {symbol}: {e}")
            return None

    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate symbols using quote checks via relay.

        Args:
            symbols: List of symbols to validate

        Returns:
            Dictionary mapping symbols to validation status
        """
        try:
            batch_quotes = await self.get_batch_quotes(symbols)

            results = {}
            for symbol in symbols:
                quote_data = batch_quotes.get(symbol)
                results[symbol] = (
                    quote_data is not None and
                    "error" not in quote_data and
                    "price" in quote_data
                )

            return results

        except Exception as e:
            logger.error(f"Error validating symbols: {e}")
            return {symbol: False for symbol in symbols}


# Utility functions for backward compatibility
async def get_historical_data_for_symbol(
    client: DataCollectorClient,
    symbol: str,
    timeframe: TimeFrame = TimeFrame.ONE_DAY,
    days: int = 252
) -> Optional[pl.DataFrame]:
    """
    Utility function to get historical data for a symbol.

    Args:
        client: DataCollectorClient instance
        symbol: Trading symbol
        timeframe: Data timeframe
        days: Number of days of data

    Returns:
        DataFrame with historical data or None
    """
    timeframe_map = {
        TimeFrame.ONE_MINUTE: "1m",
        TimeFrame.FIVE_MINUTES: "5m",
        TimeFrame.FIFTEEN_MINUTES: "15m",
        TimeFrame.THIRTY_MINUTES: "30m",
        TimeFrame.ONE_HOUR: "1h",
        TimeFrame.ONE_DAY: "1d",
        TimeFrame.ONE_WEEK: "1w",
        TimeFrame.ONE_MONTH: "1M"
    }

    timeframe_str = timeframe_map.get(timeframe, "1d")
    return await client.get_historical_data(symbol, days=days, timeframe=timeframe_str)


async def calculate_returns_from_data(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    Calculate returns from price data.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with returns column added
    """
    try:
        if df is None or len(df) < 2:
            return None

        # Calculate daily returns
        returns_df = df.with_columns([
            pl.col("close").pct_change().alias("returns")
        ]).drop_nulls()

        return returns_df

    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        return None
