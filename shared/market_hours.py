"""
Market Hours Service using Alpaca APIs

This module provides comprehensive market hours management using Alpaca's APIs as the
primary source of truth for accurate market status, holidays, and trading schedules.

The module handles:
- Real-time market status via Alpaca Clock and Calendar APIs
- Market session detection (pre-market, regular, after-hours, closed)
- Event listening and callbacks for market state changes
- Async monitoring and event scheduling
- Comprehensive market status reporting
- Fallback mechanisms for API failures

Alpaca APIs provide accurate data including:
- Bank holidays and market closures
- Early close days (Thanksgiving, Christmas Eve, etc.)
- Accurate timezone handling
- Real-time market status

Usage:
    from shared.market_hours import MarketHoursService, is_market_open

    # Simple usage
    if await is_market_open():
        print("Market is open")

    # Advanced usage with monitoring
    market_service = MarketHoursService()
    await market_service.start_monitoring()

    # Register for market events
    market_service.register_event_listener(MarketEvent.MARKET_OPEN, my_callback)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Try to import alpaca-py, fallback to manual HTTP requests if not available
try:
    from alpaca.common.exceptions import APIError
    from alpaca.trading.client import TradingClient
    from alpaca.trading.models import Calendar, Clock
    from alpaca.trading.requests import GetCalendarRequest

    ALPACA_AVAILABLE = True
except ImportError:
    logger.warning("alpaca-py not available, using HTTP requests for market hours")
    ALPACA_AVAILABLE = False
    # Create mock types
    Calendar = None  # type: ignore
    Clock = None  # type: ignore
    APIError = None  # type: ignore

try:
    import pytz

    PYTZ_AVAILABLE = True
except ImportError:
    pytz = None  # type: ignore
    PYTZ_AVAILABLE = False
    logger.warning("pytz not available, using basic timezone handling")


class MarketSession(str, Enum):
    """Market session types."""

    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class MarketEvent(str, Enum):
    """Market events that trigger actions."""

    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    PRE_MARKET_START = "pre_market_start"
    AFTER_HOURS_END = "after_hours_end"
    SESSION_CHANGE = "session_change"
    TRADING_DAY_START = "trading_day_start"
    TRADING_DAY_END = "trading_day_end"


@dataclass
class MarketStatus:
    """Comprehensive market status information."""

    is_open: bool
    current_session: MarketSession
    timestamp: datetime
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    market_date: Optional[date] = None
    is_trading_day: bool = False
    early_close: bool = False

    @property
    def is_trading_session(self) -> bool:
        """Check if any trading session is active."""
        return self.current_session in [
            MarketSession.PRE_MARKET,
            MarketSession.REGULAR,
            MarketSession.AFTER_HOURS,
        ]


@dataclass
class MarketDay:
    """Market day information from Alpaca Calendar API."""

    date: date
    open_time: datetime
    close_time: datetime
    early_close: bool = False
    session_open: Optional[datetime] = None
    session_close: Optional[datetime] = None

    def is_market_hours(self, current_time: datetime) -> bool:
        """Check if given time is within market hours for this day."""
        return self.open_time <= current_time <= self.close_time


@dataclass
class MarketEventInfo:
    """Market event with timing information."""

    event_type: MarketEvent
    timestamp: datetime
    description: str
    session_from: Optional[MarketSession] = None
    session_to: Optional[MarketSession] = None


class MarketHoursService:
    """
    Comprehensive market hours service using Alpaca APIs as the primary data source.

    This service provides accurate market hours information by leveraging Alpaca's
    Clock and Calendar APIs, with intelligent caching, event monitoring, and
    comprehensive market status reporting.
    """

    def __init__(self, cache_ttl: int = 300, timezone_name: str = "America/New_York"):
        """
        Initialize market hours service.

        Args:
            cache_ttl: Cache time-to-live in seconds
            timezone_name: Market timezone (defaults to America/New_York)
        """
        self.cache_ttl = timedelta(seconds=cache_ttl)

        # Set up timezone
        if PYTZ_AVAILABLE:
            self.timezone = pytz.timezone(timezone_name)
        else:
            from datetime import timezone

            self.timezone = timezone.utc  # type: ignore
            logger.warning(f"pytz not available, using UTC instead of {timezone_name}")

        # Get config for Alpaca credentials
        self.config = None
        try:
            from shared.config import get_config

            self.config = get_config()
        except Exception as e:
            logger.warning(f"Could not load shared config: {e}")

        # Initialize Alpaca client if available
        self.trading_client: Optional[TradingClient] = None
        if ALPACA_AVAILABLE and self.config and hasattr(self.config, "alpaca"):
            try:
                paper_trading = bool(
                    self.config.alpaca.base_url
                    and "paper" in self.config.alpaca.base_url
                )
                self.trading_client = TradingClient(
                    api_key=self.config.alpaca.api_key,
                    secret_key=self.config.alpaca.secret_key,
                    paper=paper_trading,
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

        # Event listeners
        self._event_listeners: Dict[MarketEvent, List[Callable]] = {}

        # Session tracking for event generation
        self._current_session: Optional[MarketSession] = None
        self._session_change_callbacks: List[Callable] = []

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_tasks: List[asyncio.Task] = []

        # Fallback market hours (EST/EDT)
        self._fallback_times = {
            "pre_market_start": time(4, 0),  # 4:00 AM
            "market_open": time(9, 30),  # 9:30 AM
            "market_close": time(16, 0),  # 4:00 PM
            "after_hours_end": time(20, 0),  # 8:00 PM
        }

    async def __aenter__(self):
        """Async context manager entry."""
        if not self._session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    async def get_market_status(self, force_refresh: bool = False) -> MarketStatus:
        """
        Get comprehensive market status using Alpaca APIs.

        Args:
            force_refresh: Force API call even if cached data is available

        Returns:
            Complete market status information
        """
        # Check cache first
        if (
            not force_refresh
            and self._status_cache
            and self._status_cache_time
            and datetime.now(timezone.utc) - self._status_cache_time < self.cache_ttl
        ):
            return self._status_cache

        try:
            # Try Alpaca Clock API first
            if self.trading_client:
                status = await self._get_status_from_alpaca_client()
            else:
                status = await self._get_status_from_http_api()

            if status:
                # Enhance with session information
                status = await self._enhance_status_with_sessions(status)

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

            # Get today's calendar info for additional details
            calendar_info = None
            try:
                today = datetime.now(timezone.utc).date()
                calendar_request = GetCalendarRequest(start=today, end=today)
                calendar_data = self.trading_client.get_calendar(calendar_request)
                if (
                    calendar_data
                    and isinstance(calendar_data, list)
                    and len(calendar_data) > 0
                ):
                    calendar_info = calendar_data[0]
            except Exception as e:
                logger.debug(f"Could not get calendar info: {e}")

            if isinstance(clock, dict):
                is_open = clock.get("is_open", False)
                timestamp = clock.get("timestamp")
                next_open = clock.get("next_open")
                next_close = clock.get("next_close")
            else:
                is_open = getattr(clock, "is_open", False)
                timestamp = getattr(clock, "timestamp", None)
                next_open = getattr(clock, "next_open", None)
                next_close = getattr(clock, "next_close", None)

            return MarketStatus(
                is_open=is_open,
                current_session=(
                    MarketSession.REGULAR if is_open else MarketSession.CLOSED
                ),
                timestamp=timestamp or datetime.now(),
                next_open=next_open,
                next_close=next_close,
                market_date=timestamp.date() if timestamp else None,
                is_trading_day=bool(calendar_info),
                early_close=False,  # Would need additional logic to determine this
            )

        except Exception as e:
            logger.error(f"Failed to get clock from Alpaca client: {e}")
            return None

    async def _get_status_from_http_api(self) -> Optional[MarketStatus]:
        """Get market status using direct HTTP API calls."""
        if not self.config or not hasattr(self.config, "alpaca"):
            logger.warning("Alpaca configuration not available")
            return None

        # Validate credentials
        if not self.config.alpaca.api_key or not self.config.alpaca.secret_key:
            logger.error(
                "Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables"
            )
            return None

        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            headers = {
                "APCA-API-KEY-ID": self.config.alpaca.api_key,
                "APCA-API-SECRET-KEY": self.config.alpaca.secret_key,
            }

            base_url = self.config.alpaca.base_url or "https://paper-api.alpaca.markets"

            # Get clock data
            clock_url = f"{base_url}/v2/clock"
            async with self._session.get(clock_url, headers=headers) as response:
                if response.status == 403:
                    logger.error(
                        "Alpaca Clock API returned status 403 - Invalid API credentials. Please check ALPACA_API_KEY and ALPACA_SECRET_KEY"
                    )
                    return None
                elif response.status != 200:
                    logger.error(
                        f"Alpaca Clock API returned status {response.status}: {await response.text()}"
                    )
                    return None

                clock_data = await response.json()

            # Get calendar data for today
            today = datetime.now(timezone.utc).date()
            calendar_url = f"{base_url}/v2/calendar"
            calendar_params = {"start": today.isoformat(), "end": today.isoformat()}

            calendar_info = None
            try:
                async with self._session.get(
                    calendar_url, headers=headers, params=calendar_params
                ) as response:
                    if response.status == 200:
                        calendar_data = await response.json()
                        if calendar_data:
                            calendar_info = calendar_data[0]
            except Exception as e:
                logger.debug(f"Could not get calendar info: {e}")

            return MarketStatus(
                is_open=clock_data.get("is_open", False),
                current_session=(
                    MarketSession.REGULAR
                    if clock_data.get("is_open")
                    else MarketSession.CLOSED
                ),
                timestamp=datetime.fromisoformat(
                    clock_data["timestamp"].replace("Z", "+00:00")
                ),
                next_open=(
                    datetime.fromisoformat(
                        clock_data["next_open"].replace("Z", "+00:00")
                    )
                    if clock_data.get("next_open")
                    else None
                ),
                next_close=(
                    datetime.fromisoformat(
                        clock_data["next_close"].replace("Z", "+00:00")
                    )
                    if clock_data.get("next_close")
                    else None
                ),
                market_date=today,
                is_trading_day=bool(calendar_info),
                early_close=False,  # Would need additional calendar logic
            )

        except Exception as e:
            logger.error(f"HTTP API call failed: {e}")
            return None

    async def _enhance_status_with_sessions(self, status: MarketStatus) -> MarketStatus:
        """Enhance basic status with detailed session information."""
        if not status.is_trading_day:
            status.current_session = MarketSession.CLOSED
            return status

        now = status.timestamp
        current_time = now.time()

        # Use fallback times to determine sessions if market is a trading day
        if current_time < self._fallback_times["market_open"]:
            if current_time >= self._fallback_times["pre_market_start"]:
                status.current_session = MarketSession.PRE_MARKET
            else:
                status.current_session = MarketSession.CLOSED
        elif current_time < self._fallback_times["market_close"]:
            status.current_session = MarketSession.REGULAR
            status.is_open = True
        elif current_time < self._fallback_times["after_hours_end"]:
            status.current_session = MarketSession.AFTER_HOURS
        else:
            status.current_session = MarketSession.CLOSED

        return status

    async def _get_fallback_status(self) -> MarketStatus:
        """Get market status using fallback logic when APIs are unavailable."""
        now = datetime.now(timezone.utc)

        # Simple weekday check
        is_trading_day = now.weekday() < 5  # Monday=0, Friday=4

        if not is_trading_day:
            return MarketStatus(
                is_open=False,
                current_session=MarketSession.CLOSED,
                timestamp=now,
                is_trading_day=False,
            )

        # Convert to ET (simplified, doesn't handle DST properly)
        et_now = now - timedelta(hours=5)
        current_time = et_now.time()

        # Determine session
        if current_time < self._fallback_times["market_open"]:
            if current_time >= self._fallback_times["pre_market_start"]:
                session = MarketSession.PRE_MARKET
                is_open = False
            else:
                session = MarketSession.CLOSED
                is_open = False
        elif current_time < self._fallback_times["market_close"]:
            session = MarketSession.REGULAR
            is_open = True
        elif current_time < self._fallback_times["after_hours_end"]:
            session = MarketSession.AFTER_HOURS
            is_open = False
        else:
            session = MarketSession.CLOSED
            is_open = False

        return MarketStatus(
            is_open=is_open,
            current_session=session,
            timestamp=now,
            is_trading_day=is_trading_day,
        )

    async def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if market is open at given timestamp.

        Args:
            timestamp: Time to check (defaults to now)

        Returns:
            True if market is open for regular trading
        """
        if timestamp:
            # For historical times, we'd need calendar data
            now = datetime.now(timezone.utc)
            if abs((timestamp - now).total_seconds()) < 300:  # Within 5 minutes
                status = await self.get_market_status()
                return status.is_open
            else:
                return await self._is_market_open_historical(timestamp)

        status = await self.get_market_status()
        return status.is_open

    async def _is_market_open_historical(self, timestamp: datetime) -> bool:
        """Check if market was open at historical timestamp."""
        # Would need calendar data for accurate historical checks
        # For now, use simple weekday + time check
        if timestamp.weekday() >= 5:  # Weekend
            return False

        # Simple time check (doesn't account for holidays)
        time_check = timestamp.time()
        return (
            self._fallback_times["market_open"]
            <= time_check
            <= self._fallback_times["market_close"]
        )

    async def get_current_session(self) -> MarketSession:
        """Get the current market session."""
        status = await self.get_market_status()
        return status.current_session

    async def get_next_market_open(self) -> Optional[datetime]:
        """Get next market open time using Alpaca data."""
        status = await self.get_market_status()
        return status.next_open

    async def get_next_market_close(self) -> Optional[datetime]:
        """Get next market close time using Alpaca data."""
        status = await self.get_market_status()
        return status.next_close

    async def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        """Check if given date is a trading day using Alpaca calendar."""
        if check_date is None:
            check_date = datetime.now(timezone.utc).date()

        # Check cache first
        if check_date in self._calendar_cache:
            return True

        try:
            # Try to get calendar data from Alpaca
            if self.trading_client:
                calendar_request = GetCalendarRequest(start=check_date, end=check_date)
                calendar_data = self.trading_client.get_calendar(calendar_request)
                return len(calendar_data) > 0
            elif self.config and hasattr(self.config, "alpaca"):
                return await self._is_trading_day_http(check_date)
        except Exception as e:
            logger.error(f"Failed to check trading day via Alpaca: {e}")

        # Fallback: simple weekday check
        return check_date.weekday() < 5

    async def _is_trading_day_http(self, check_date: date) -> bool:
        """Check trading day via HTTP API."""
        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            if not self.config:
                return False

            headers = {
                "APCA-API-KEY-ID": self.config.alpaca.api_key,
                "APCA-API-SECRET-KEY": self.config.alpaca.secret_key,
            }

            base_url = self.config.alpaca.base_url or "https://paper-api.alpaca.markets"
            url = f"{base_url}/v2/calendar"
            params = {"start": check_date.isoformat(), "end": check_date.isoformat()}

            async with self._session.get(
                url, headers=headers, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return len(data) > 0
                else:
                    return False
        except Exception:
            return False

    def register_event_listener(self, event_type: MarketEvent, callback: Callable):
        """Register a callback for market events."""
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []

        self._event_listeners[event_type].append(callback)
        logger.info(f"Registered event listener for {event_type.value}")

    def register_session_change_callback(self, callback: Callable):
        """Register a callback for session changes."""
        self._session_change_callbacks.append(callback)
        logger.info("Registered session change callback")

    async def start_monitoring(self):
        """Start market hours monitoring and event generation."""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return

        self._monitoring_active = True
        logger.info("Starting market hours monitoring...")

        # Start monitoring tasks
        self._monitoring_tasks = [
            asyncio.create_task(self._session_monitoring_loop()),
            asyncio.create_task(self._event_monitoring_loop()),
        ]

    async def stop_monitoring(self):
        """Stop market hours monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        logger.info("Stopping market hours monitoring...")

        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._monitoring_tasks.clear()

    async def _session_monitoring_loop(self):
        """Monitor for market session changes."""
        while self._monitoring_active:
            try:
                current_session = await self.get_current_session()

                # Check for session change
                if current_session != self._current_session:
                    logger.info(
                        f"Market session changed: {self._current_session} -> {current_session}"
                    )

                    # Create event info
                    event_info = MarketEventInfo(
                        event_type=MarketEvent.SESSION_CHANGE,
                        timestamp=datetime.now(timezone.utc),
                        description=f"Session changed from {self._current_session} to {current_session}",
                        session_from=self._current_session,
                        session_to=current_session,
                    )

                    # Notify session change callbacks
                    for callback in self._session_change_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(self._current_session, current_session)
                            else:
                                callback(self._current_session, current_session)
                        except Exception as e:
                            logger.error(f"Session change callback failed: {e}")

                    # Trigger session change event
                    await self._trigger_event(event_info)

                    # Generate specific session events
                    await self._generate_session_events(
                        self._current_session, current_session
                    )

                    self._current_session = current_session

                # Sleep for 1 minute before next check
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Session monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _event_monitoring_loop(self):
        """Monitor for specific market events."""
        while self._monitoring_active:
            try:
                # Check for market open/close events based on next transitions
                status = await self.get_market_status()
                now = datetime.now(timezone.utc)

                # Check for upcoming events within next 2 minutes
                if status.next_open and (status.next_open - now).total_seconds() <= 120:
                    event_info = MarketEventInfo(
                        event_type=MarketEvent.MARKET_OPEN,
                        timestamp=status.next_open,
                        description="Market opening",
                    )
                    await self._trigger_event(event_info)

                if (
                    status.next_close
                    and (status.next_close - now).total_seconds() <= 120
                ):
                    event_info = MarketEventInfo(
                        event_type=MarketEvent.MARKET_CLOSE,
                        timestamp=status.next_close,
                        description="Market closing",
                    )
                    await self._trigger_event(event_info)

                # Sleep for 1 minute
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Event monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _generate_session_events(
        self, old_session: Optional[MarketSession], new_session: MarketSession
    ):
        """Generate specific events based on session transitions."""
        now = datetime.now(timezone.utc)

        if (
            new_session == MarketSession.PRE_MARKET
            and old_session == MarketSession.CLOSED
        ):
            event = MarketEventInfo(
                event_type=MarketEvent.PRE_MARKET_START,
                timestamp=now,
                description="Pre-market trading started",
            )
            await self._trigger_event(event)

        elif (
            new_session == MarketSession.REGULAR
            and old_session == MarketSession.PRE_MARKET
        ):
            event = MarketEventInfo(
                event_type=MarketEvent.MARKET_OPEN,
                timestamp=now,
                description="Regular market opened",
            )
            await self._trigger_event(event)

        elif (
            new_session == MarketSession.AFTER_HOURS
            and old_session == MarketSession.REGULAR
        ):
            event = MarketEventInfo(
                event_type=MarketEvent.MARKET_CLOSE,
                timestamp=now,
                description="Regular market closed",
            )
            await self._trigger_event(event)

        elif (
            new_session == MarketSession.CLOSED
            and old_session == MarketSession.AFTER_HOURS
        ):
            event = MarketEventInfo(
                event_type=MarketEvent.AFTER_HOURS_END,
                timestamp=now,
                description="After-hours trading ended",
            )
            await self._trigger_event(event)

    async def _trigger_event(self, event: MarketEventInfo):
        """Trigger a market event and notify listeners."""
        event_type = event.event_type
        logger.debug(f"Triggering market event: {event.description}")

        if event_type in self._event_listeners:
            for listener in self._event_listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(event)
                    else:
                        listener(event)
                except Exception as e:
                    logger.error(f"Event listener failed for {event_type.value}: {e}")

    async def get_market_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive market status summary."""
        status = await self.get_market_status()

        return {
            "timestamp": status.timestamp.isoformat() if status.timestamp else None,
            "is_open": status.is_open,
            "current_session": status.current_session.value,
            "is_trading_day": status.is_trading_day,
            "is_trading_session": status.is_trading_session,
            "market_date": (
                status.market_date.isoformat() if status.market_date else None
            ),
            "early_close": status.early_close,
            "next_open": status.next_open.isoformat() if status.next_open else None,
            "next_close": status.next_close.isoformat() if status.next_close else None,
            "monitoring_active": self._monitoring_active,
        }

    async def wait_for_market_open(self):
        """Wait until market opens."""
        while True:
            status = await self.get_market_status()
            if status.is_open:
                break

            if status.next_open:
                wait_time = (
                    status.next_open - datetime.now(timezone.utc)
                ).total_seconds()
                if wait_time > 0:
                    # Wait for market open, but check every minute
                    sleep_time = min(60, wait_time)
                    logger.info(f"Waiting for market open in {wait_time:.0f} seconds")
                    await asyncio.sleep(sleep_time)
                else:
                    break
            else:
                # No next open time available, wait a minute and check again
                await asyncio.sleep(60)

    async def wait_for_session(
        self, target_session: MarketSession, timeout: Optional[int] = None
    ):
        """Wait for a specific market session."""
        start_time = datetime.now()

        while True:
            current_session = await self.get_current_session()
            if current_session == target_session:
                break

            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Timeout waiting for session {target_session.value}"
                    )

            await asyncio.sleep(30)  # Check every 30 seconds

    async def shutdown(self):
        """Shutdown the market hours service."""
        await self.stop_monitoring()

        if self._session:
            await self._session.close()
            self._session = None


# Global instance for simple usage
_market_hours_instance: Optional[MarketHoursService] = None


async def get_market_hours_service() -> MarketHoursService:
    """Get or create global market hours service instance."""
    global _market_hours_instance

    if _market_hours_instance is None:
        _market_hours_instance = MarketHoursService()

    return _market_hours_instance


# Convenience functions for simple usage
async def is_market_open(timestamp: Optional[datetime] = None) -> bool:
    """
    Simple check if market is currently open.

    Args:
        timestamp: Optional timestamp to check (defaults to now)

    Returns:
        True if market is open for regular trading
    """
    service = await get_market_hours_service()
    return await service.is_market_open(timestamp)


async def get_market_status() -> MarketStatus:
    """Get current market status."""
    service = await get_market_hours_service()
    return await service.get_market_status()


async def get_current_session() -> MarketSession:
    """Get current market session."""
    service = await get_market_hours_service()
    return await service.get_current_session()


async def get_next_market_open() -> Optional[datetime]:
    """Get next market open time."""
    service = await get_market_hours_service()
    return await service.get_next_market_open()


async def get_next_market_close() -> Optional[datetime]:
    """Get next market close time."""
    service = await get_market_hours_service()
    return await service.get_next_market_close()


async def is_trading_day(check_date: Optional[date] = None) -> bool:
    """Check if given date is a trading day."""
    service = await get_market_hours_service()
    return await service.is_trading_day(check_date)


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
            raise RuntimeError(
                "Cannot use sync version from async context. Use await is_market_open() instead."
            )
        else:
            return loop.run_until_complete(is_market_open(timestamp))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(is_market_open(timestamp))


def is_trading_day_simple(check_date: date) -> bool:
    """
    Simple synchronous check if a date is a trading day (weekday).

    Note: This doesn't account for holidays. Use the async methods
    for accurate holiday-aware checking via Alpaca APIs.
    """
    return check_date.weekday() < 5


@lru_cache(maxsize=128)
def _cached_weekday_check(date_str: str) -> bool:
    """Cached basic weekday check for performance."""
    try:
        check_date = datetime.fromisoformat(date_str).date()
        return check_date.weekday() < 5
    except Exception:
        return False
