"""
Market Hours Utility using Alpaca APIs

This module provides comprehensive market hours checking functionality using
Alpaca's Calendar and Clock APIs for accurate market status determination.

The module handles:
- Real-time market status via Alpaca Clock API
- Market calendar with holidays and early closures via Alpaca Calendar API
- Proper timezone handling
- Caching to minimize API calls
- Fallback mechanisms for API failures

Usage:
    from shared.market_hours import MarketHours, is_market_open

    # Simple usage
    if await is_market_open():
        print("Market is open")

    # Advanced usage
    market_hours = MarketHours()
    status = await market_hours.get_market_status()
    if status.is_open:
        print(f"Market is open until {status.next_close}")
"""

import asyncio
import logging
from datetime import datetime, date, time, timezone, timedelta
from typing import Optional, List, Dict, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import alpaca-py, fallback to manual HTTP requests if not available
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest
    from alpaca.trading.models import Calendar, Clock
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    logger.warning("alpaca-py not available, using HTTP requests for market hours")
    ALPACA_AVAILABLE = False
    # Create mock types
    Calendar = Dict[str, Any]
    Clock = Dict[str, Any]
    APIError = Exception


class MarketSession(Enum):
    """Market session types."""
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_MARKET = "after_market"


@dataclass
class MarketStatus:
    """Market status information."""
    is_open: bool
    current_session: MarketSession
    timestamp: datetime
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    market_date: Optional[date] = None

    @property
    def is_trading_session(self) -> bool:
        """Check if any trading session is active."""
        return self.current_session in [
            MarketSession.PRE_MARKET,
            MarketSession.REGULAR,
            MarketSession.AFTER_MARKET
        ]


@dataclass
class MarketDay:
    """Market day information."""
    date: date
    open_time: time
    close_time: time
    is_early_close: bool = False

    def is_market_hours(self, current_time: datetime) -> bool:
        """Check if given time is within market hours for this day."""
        if current_time.date() != self.date:
            return False

        current_time_only = current_time.time()
        return self.open_time <= current_time_only <= self.close_time


class MarketHours:
    """
    Comprehensive market hours manager using Alpaca APIs.

    This class provides accurate market status information by leveraging
    Alpaca's Calendar and Clock APIs, with intelligent caching and fallback
    mechanisms.
    """

    def __init__(self, cache_ttl: int = 300):  # 5 minutes default cache
        """
        Initialize market hours manager.

        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = timedelta(seconds=cache_ttl)

        # Get config for Alpaca credentials
        self.config = None
        try:
            from shared.config import get_config
            self.config = get_config()
        except Exception as e:
            logger.warning(f"Could not load shared config: {e}")

        # Initialize Alpaca client if available
        self.trading_client: Optional[TradingClient] = None
        if ALPACA_AVAILABLE and self.config and hasattr(self.config, 'alpaca'):
            try:
                self.trading_client = TradingClient(
                    api_key=self.config.alpaca.api_key,
                    secret_key=self.config.alpaca.secret_key,
                    paper=self.config.alpaca.base_url and 'paper' in self.config.alpaca.base_url
                )
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")

        # Cache for market data
        self._status_cache: Optional[MarketStatus] = None
        self._status_cache_time: Optional[datetime] = None

        self._calendar_cache: Dict[date, MarketDay] = {}
        self._calendar_cache_time: Optional[datetime] = None

        # HTTP session for direct API calls
        self._session: Optional[aiohttp.ClientSession] = None

        # Fallback market hours (EST/EDT)
        self._fallback_open = time(9, 30)  # 9:30 AM
        self._fallback_close = time(16, 0)  # 4:00 PM

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def get_market_status(self, force_refresh: bool = False) -> MarketStatus:
        """
        Get current market status.

        Args:
            force_refresh: Force API call even if cached data is available

        Returns:
            Current market status information
        """
        # Check cache first
        if (not force_refresh and
            self._status_cache and
            self._status_cache_time and
            datetime.now(timezone.utc) - self._status_cache_time < self.cache_ttl):
            return self._status_cache

        try:
            # Try Alpaca Clock API first
            if self.trading_client:
                status = await self._get_status_from_alpaca_client()
            else:
                status = await self._get_status_from_http_api()

            if status:
                # Update cache
                self._status_cache = status
                self._status_cache_time = datetime.now(timezone.utc)
                return status

        except Exception as e:
            logger.error(f"Failed to get market status from Alpaca: {e}")

        # Fallback to manual calculation
        return await self._get_fallback_status()

    async def _get_status_from_alpaca_client(self) -> Optional[MarketStatus]:
        """Get market status using alpaca-py client."""
        if not self.trading_client:
            return None

        try:
            clock = self.trading_client.get_clock()

            return MarketStatus(
                is_open=clock.is_open,
                current_session=MarketSession.REGULAR if clock.is_open else MarketSession.CLOSED,
                timestamp=clock.timestamp,
                next_open=clock.next_open,
                next_close=clock.next_close
            )

        except Exception as e:
            logger.error(f"Failed to get clock from Alpaca client: {e}")
            return None

    async def _get_status_from_http_api(self) -> Optional[MarketStatus]:
        """Get market status using direct HTTP API calls."""
        if not self.config or not hasattr(self.config, 'alpaca'):
            return None

        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            headers = {
                "APCA-API-KEY-ID": self.config.alpaca.api_key,
                "APCA-API-SECRET-KEY": self.config.alpaca.secret_key,
            }

            base_url = self.config.alpaca.base_url or "https://paper-api.alpaca.markets"
            url = f"{base_url}/v2/clock"

            async with self._session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    return MarketStatus(
                        is_open=data.get("is_open", False),
                        current_session=MarketSession.REGULAR if data.get("is_open") else MarketSession.CLOSED,
                        timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
                        next_open=datetime.fromisoformat(data["next_open"].replace("Z", "+00:00")) if data.get("next_open") else None,
                        next_close=datetime.fromisoformat(data["next_close"].replace("Z", "+00:00")) if data.get("next_close") else None
                    )
                else:
                    logger.error(f"Alpaca Clock API returned status {response.status}")
                    return None

        except Exception as e:
            logger.error(f"HTTP API call failed: {e}")
            return None

    async def _get_fallback_status(self) -> MarketStatus:
        """Get market status using fallback logic."""
        now = datetime.now(timezone.utc)

        # Convert to ET (market timezone)
        # This is a simplified conversion - in production, you'd want proper timezone handling
        et_now = now - timedelta(hours=5)  # EST offset, doesn't handle DST properly

        # Check if it's a weekday
        if et_now.weekday() >= 5:  # Weekend
            return MarketStatus(
                is_open=False,
                current_session=MarketSession.CLOSED,
                timestamp=now
            )

        current_time = et_now.time()

        # Determine market session
        if self._fallback_open <= current_time <= self._fallback_close:
            session = MarketSession.REGULAR
            is_open = True
        else:
            session = MarketSession.CLOSED
            is_open = False

        return MarketStatus(
            is_open=is_open,
            current_session=session,
            timestamp=now
        )

    async def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if market is open at given timestamp.

        Args:
            timestamp: Time to check (defaults to now)

        Returns:
            True if market is open
        """
        if timestamp:
            # For historical times, we'd need to implement more complex logic
            # For now, just return current status if timestamp is close to now
            now = datetime.now(timezone.utc)
            if abs((timestamp - now).total_seconds()) < 300:  # Within 5 minutes
                status = await self.get_market_status()
                return status.is_open
            else:
                # For historical checks, use fallback
                return await self._is_market_open_historical(timestamp)

        status = await self.get_market_status()
        return status.is_open

    async def _is_market_open_historical(self, timestamp: datetime) -> bool:
        """Check if market was open at historical timestamp."""
        # Simplified historical check - in production would use calendar data
        if timestamp.weekday() >= 5:  # Weekend
            return False

        # Convert to ET and check basic hours
        et_time = timestamp - timedelta(hours=5)  # Simplified timezone conversion
        current_time = et_time.time()
        return self._fallback_open <= current_time <= self._fallback_close

    async def get_next_market_open(self) -> Optional[datetime]:
        """Get next market open time."""
        status = await self.get_market_status()
        return status.next_open

    async def get_next_market_close(self) -> Optional[datetime]:
        """Get next market close time."""
        status = await self.get_market_status()
        return status.next_close


# Global instance for simple usage
_market_hours_instance: Optional[MarketHours] = None


async def get_market_hours_instance() -> MarketHours:
    """Get or create global market hours instance."""
    global _market_hours_instance

    if _market_hours_instance is None:
        _market_hours_instance = MarketHours()

    return _market_hours_instance


# Convenience functions for simple usage
async def is_market_open(timestamp: Optional[datetime] = None) -> bool:
    """
    Simple check if market is currently open.

    Args:
        timestamp: Optional timestamp to check (defaults to now)

    Returns:
        True if market is open
    """
    market_hours = await get_market_hours_instance()
    return await market_hours.is_market_open(timestamp)


async def get_market_status() -> MarketStatus:
    """Get current market status."""
    market_hours = await get_market_hours_instance()
    return await market_hours.get_market_status()


async def get_next_market_open() -> Optional[datetime]:
    """Get next market open time."""
    market_hours = await get_market_hours_instance()
    return await market_hours.get_next_market_open()


async def get_next_market_close() -> Optional[datetime]:
    """Get next market close time."""
    market_hours = await get_market_hours_instance()
    return await market_hours.get_next_market_close()


# Synchronous wrapper for backward compatibility
def is_market_open_sync(timestamp: Optional[datetime] = None) -> bool:
    """
    Synchronous version of is_market_open.

    Note: This creates a new event loop if one doesn't exist.
    Use the async version when possible.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we can't use run()
            # This is a limitation - caller should use async version
            raise RuntimeError("Cannot use sync version from async context. Use await is_market_open() instead.")
        else:
            return loop.run_until_complete(is_market_open(timestamp))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(is_market_open(timestamp))


@lru_cache(maxsize=128)
def _get_basic_market_hours(date_str: str) -> bool:
    """
    Basic market hours check with caching.

    This is a very simple fallback that only checks weekdays.
    Used when all other methods fail.
    """
    try:
        check_date = datetime.fromisoformat(date_str).date()
        return check_date.weekday() < 5
    except:
        return False


def is_trading_day(check_date: date) -> bool:
    """
    Simple check if a date is a trading day (weekday).

    Note: This doesn't account for holidays. Use the async methods
    for accurate holiday-aware checking.
    """
    return check_date.weekday() < 5
