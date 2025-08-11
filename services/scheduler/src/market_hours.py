"""
Market Hours Service for the Trading Scheduler.

This module provides comprehensive market hours management including:
- Market session detection (pre-market, regular, after-hours, closed)
- Holiday calendar management
- Trading day validation
- Market status monitoring
- Time-based triggers for market events
"""

import logging
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import pytz
try:
    import pandas as pd
    import pandas_market_calendars as mcal
except ImportError:
    pd = None
    mcal = None
from dataclasses import dataclass
import asyncio


logger = logging.getLogger(__name__)


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
    HOLIDAY_START = "holiday_start"
    HOLIDAY_END = "holiday_end"
    WEEKEND_START = "weekend_start"
    WEEKEND_END = "weekend_end"


@dataclass
class MarketHours:
    """Market hours for a specific date."""
    date: date
    is_trading_day: bool
    pre_market_start: Optional[datetime] = None
    market_open: Optional[datetime] = None
    market_close: Optional[datetime] = None
    after_hours_end: Optional[datetime] = None
    early_close: bool = False
    holiday_name: Optional[str] = None


@dataclass
class MarketEventInfo:
    """Market event with timing information."""
    event_type: MarketEvent
    timestamp: datetime
    description: str
    next_occurrence: Optional[datetime] = None


class MarketHoursService:
    """Service for managing market hours and trading sessions."""

    def __init__(self, timezone: str = "America/New_York"):
        self.timezone = pytz.timezone(timezone)

        # Market calendars
        if mcal:
            self.nyse = mcal.get_calendar('NYSE')
            self.nasdaq = mcal.get_calendar('NASDAQ')
        else:
            self.nyse = None
            self.nasdaq = None

        # Default trading hours (ET)
        self.default_hours = {
            'pre_market_start': time(4, 0),    # 4:00 AM
            'market_open': time(9, 30),        # 9:30 AM
            'market_close': time(16, 0),       # 4:00 PM
            'after_hours_end': time(20, 0)     # 8:00 PM
        }

        # Early close times (holidays like Thanksgiving, Christmas Eve)
        self.early_close_times = {
            'market_close': time(13, 0),       # 1:00 PM
            'after_hours_end': time(17, 0)     # 5:00 PM
        }

        # Cache for market hours
        self._market_hours_cache: Dict[date, MarketHours] = {}
        self._cache_expiry = timedelta(days=1)

        # Event listeners
        self._event_listeners: Dict[MarketEvent, List[Callable]] = {}

        # Current session tracking
        self._current_session: Optional[MarketSession] = None
        self._session_change_callbacks: List[Callable] = []

    def get_market_hours(self, target_date: date) -> Optional[MarketHours]:
        """Get market hours for a specific date."""
        # Check cache first
        if target_date in self._market_hours_cache:
            cached_hours = self._market_hours_cache[target_date]
            # Simple cache validation (you might want more sophisticated logic)
            return cached_hours

        # Calculate market hours
        hours = self._calculate_market_hours(target_date)

        # Cache the result
        if hours:
            self._market_hours_cache[target_date] = hours

        return hours

    def _calculate_market_hours(self, target_date: date) -> Optional[MarketHours]:
        """Generate market hours for a specific date."""
        # Check if it's a weekend
        if target_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return MarketHours(
                date=target_date,
                is_trading_day=False
            )

        # Get NYSE schedule for the date
        try:
            if not self.nyse:
                # Fallback to default hours when mcal is not available
                return self._get_default_market_hours(target_date)

            schedule = self.nyse.schedule(start_date=target_date, end_date=target_date)

            if schedule.empty:
                # Market holiday
                holiday_name = self._get_holiday_name(target_date)
                return MarketHours(
                    date=target_date,
                    is_trading_day=False,
                    holiday_name=holiday_name
                )

            # Regular trading day
            market_open_utc = schedule.iloc[0]['market_open']
            market_close_utc = schedule.iloc[0]['market_close']

            # Convert to local timezone
            market_open = market_open_utc.tz_convert(self.timezone)
            market_close = market_close_utc.tz_convert(self.timezone)

            # Check for early close
            is_early_close = self._is_early_close_day(target_date)

            if is_early_close:
                # Adjust close time for early close
                market_close = self.timezone.localize(
                    datetime.combine(target_date, self.early_close_times['market_close'])
                )
                after_hours_end = self.timezone.localize(
                    datetime.combine(target_date, self.early_close_times['after_hours_end'])
                )
            else:
                after_hours_end = self.timezone.localize(
                    datetime.combine(target_date, self.default_hours['after_hours_end'])
                )

            # Pre-market start
            pre_market_start = self.timezone.localize(
                datetime.combine(target_date, self.default_hours['pre_market_start'])
            )

            return MarketHours(
                date=target_date,
                is_trading_day=True,
                pre_market_start=pre_market_start,
                market_open=market_open,
                market_close=market_close,
                after_hours_end=after_hours_end,
                early_close=is_early_close
            )

        except Exception as e:
            logger.error(f"Failed to generate market hours for {target_date}: {e}")
            # Return closed market as fallback
            return MarketHours(
                date=target_date,
                is_trading_day=False
            )

    def _get_default_market_hours(self, target_date: date) -> Optional[MarketHours]:
        """Get default market hours when mcal is not available."""
        # Check if it's a weekend
        if target_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return MarketHours(
                date=target_date,
                is_trading_day=False
            )

        # Check for basic holidays
        holiday_name = self._get_holiday_name(target_date)
        if holiday_name:
            return MarketHours(
                date=target_date,
                is_trading_day=False,
                holiday_name=holiday_name
            )

        # Default trading hours (9:30 AM - 4:00 PM ET)
        market_open = datetime.combine(target_date, time(9, 30)).replace(tzinfo=self.timezone)
        market_close = datetime.combine(target_date, time(16, 0)).replace(tzinfo=self.timezone)
        pre_market_start = datetime.combine(target_date, time(4, 0)).replace(tzinfo=self.timezone)
        after_hours_end = datetime.combine(target_date, time(20, 0)).replace(tzinfo=self.timezone)

        return MarketHours(
            date=target_date,
            is_trading_day=True,
            pre_market_start=pre_market_start,
            market_open=market_open,
            market_close=market_close,
            after_hours_end=after_hours_end,
            early_close=False
        )

    def _get_holiday_name(self, target_date: date) -> Optional[str]:
        """Get holiday name for a specific date."""
        # This would typically query a holiday database or API
        # For now, return None or implement basic holiday detection

        # Common US market holidays
        holidays = {
            (1, 1): "New Year's Day",
            (1, 15): "Martin Luther King Jr. Day",  # 3rd Monday
            (2, 19): "Presidents Day",              # 3rd Monday
            (5, 27): "Memorial Day",                # Last Monday
            (7, 4): "Independence Day",
            (9, 2): "Labor Day",                    # 1st Monday
            (11, 28): "Thanksgiving",               # 4th Thursday
            (12, 25): "Christmas Day"
        }

        month_day = (target_date.month, target_date.day)
        return holidays.get(month_day)

    def _is_early_close_day(self, target_date: date) -> bool:
        """Check if the market closes early on this date."""
        # Common early close days
        # Day after Thanksgiving (Black Friday)
        # Christmas Eve (if not weekend)
        # July 3rd (if July 4th is weekend)

        # Simple check - this could be more sophisticated
        return (
            # Day after Thanksgiving
            (target_date.month == 11 and target_date.day == 29 and target_date.weekday() == 4) or
            # Christmas Eve
            (target_date.month == 12 and target_date.day == 24 and target_date.weekday() < 5) or
            # July 3rd when July 4th is weekend
            (target_date.month == 7 and target_date.day == 3 and
             datetime(target_date.year, 7, 4).weekday() >= 5)
        )

    def get_current_session(self) -> MarketSession:
        """Get the current market session."""
        now = datetime.now(self.timezone)
        return self.get_session_at_time(now)

    def get_session_at_time(self, dt: datetime) -> MarketSession:
        """Get market session for a specific datetime."""
        market_hours = self.get_market_hours(dt.date())

        if not market_hours or not market_hours.is_trading_day:
            return MarketSession.CLOSED

        current_time = dt.time()

        # Check each session
        if (market_hours.pre_market_start and market_hours.market_open and
            market_hours.pre_market_start.time() <= current_time < market_hours.market_open.time()):
            return MarketSession.PRE_MARKET
        elif (market_hours.market_open and market_hours.market_close and
              market_hours.market_open.time() <= current_time < market_hours.market_close.time()):
            return MarketSession.REGULAR
        elif (market_hours.market_close and market_hours.after_hours_end and
              market_hours.market_close.time() <= current_time < market_hours.after_hours_end.time()):
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED

    def is_trading_day(self, target_date: Optional[date] = None) -> bool:
        """Check if the given date (or today) is a trading day."""
        if target_date is None:
            target_date = datetime.now(self.timezone).date()

        market_hours = self.get_market_hours(target_date)
        return market_hours.is_trading_day if market_hours else False

    def is_market_open(self) -> bool:
        """Check if the market is currently open for regular trading."""
        return self.get_current_session() == MarketSession.REGULAR

    def is_extended_hours(self) -> bool:
        """Check if currently in extended hours (pre-market or after-hours)."""
        session = self.get_current_session()
        return session in [MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS]

    def get_next_market_event(self) -> Optional[MarketEventInfo]:
        """Get the next market event (open, close, etc.)."""
        now = datetime.now(self.timezone)
        today_hours = self.get_market_hours(now.date())

        if today_hours and today_hours.is_trading_day:
            current_time = now.time()

            # Check what's next today
            if today_hours.pre_market_start and today_hours.pre_market_start.time() and current_time < today_hours.pre_market_start.time():
                return MarketEventInfo(
                    event_type=MarketEvent.PRE_MARKET_START,
                    timestamp=today_hours.pre_market_start,
                    description="Pre-market trading begins"
                )
            elif today_hours.market_open and today_hours.market_open.time() and current_time < today_hours.market_open.time():
                return MarketEventInfo(
                    event_type=MarketEvent.MARKET_OPEN,
                    timestamp=today_hours.market_open,
                    description="Regular market opens"
                )
            elif today_hours.market_close and today_hours.market_close.time() and current_time < today_hours.market_close.time():
                return MarketEventInfo(
                    event_type=MarketEvent.MARKET_CLOSE,
                    timestamp=today_hours.market_close,
                    description="Regular market closes"
                )
            elif today_hours.after_hours_end and today_hours.after_hours_end.time() and current_time < today_hours.after_hours_end.time():
                return MarketEventInfo(
                    event_type=MarketEvent.AFTER_HOURS_END,
                    timestamp=today_hours.after_hours_end,
                    description="After-hours trading ends"
                )

        # Look for next trading day
        next_trading_day = self.get_next_trading_day(now.date())
        if next_trading_day:
            next_hours = self.get_market_hours(next_trading_day)
            if next_hours and next_hours.pre_market_start:
                return MarketEventInfo(
                    event_type=MarketEvent.PRE_MARKET_START,
                    timestamp=next_hours.pre_market_start,
                    description="Next pre-market trading begins"
                )

        return None

    def get_next_trading_day(self, start_date: Optional[date] = None) -> Optional[date]:
        """Get the next trading day after the given date."""
        if start_date is None:
            start_date = datetime.now(self.timezone).date()

        # Look ahead up to 10 days
        for i in range(1, 11):
            candidate_date = start_date + timedelta(days=i)
            if self.is_trading_day(candidate_date):
                return candidate_date

        return None

    def get_previous_trading_day(self, start_date: Optional[date] = None) -> Optional[date]:
        """Get the previous trading day before the given date."""
        if start_date is None:
            start_date = datetime.now(self.timezone).date()

        # Look back up to 10 days
        for i in range(1, 11):
            candidate_date = start_date - timedelta(days=i)
            if self.is_trading_day(candidate_date):
                return candidate_date

        return None

    def time_until_market_open(self) -> timedelta:
        """Get time until next market open."""
        now = datetime.now(self.timezone)

        # If market is open today and we haven't passed opening
        today_hours = self.get_market_hours(now.date())
        if today_hours and today_hours.is_trading_day and today_hours.market_open and now.time() < today_hours.market_open.time():
            return today_hours.market_open - now

        # Find next trading day
        next_trading_day = self.get_next_trading_day(now.date())
        if next_trading_day:
            next_hours = self.get_market_hours(next_trading_day)
            if next_hours and next_hours.market_open:
                return next_hours.market_open - now

        return timedelta(0)

    def time_until_market_close(self) -> timedelta:
        """Get time until market close (only if market is open today)."""
        now = datetime.now(self.timezone)
        today_hours = self.get_market_hours(now.date())

        if (today_hours and today_hours.is_trading_day and
            today_hours.market_open and today_hours.market_close and
            today_hours.market_open.time() <= now.time() < today_hours.market_close.time()):
            return today_hours.market_close - now

        return timedelta(0)

    def time_in_session(self, session: MarketSession) -> timedelta:
        """Get time remaining in the specified session."""
        now = datetime.now(self.timezone)
        current_session = self.get_current_session()

        if current_session != session:
            return timedelta(0)

        today_hours = self.get_market_hours(now.date())

        if not today_hours:
            return timedelta(0)

        if session == MarketSession.PRE_MARKET and today_hours.pre_market_start and today_hours.market_open:
            return today_hours.market_open - now
        elif session == MarketSession.REGULAR and today_hours.market_close:
            return today_hours.market_close - now
        elif session == MarketSession.AFTER_HOURS and today_hours.after_hours_end:
            return today_hours.after_hours_end - now

        return timedelta(0)

    def get_trading_days_between(self, start_date: date, end_date: date) -> List[date]:
        """Get all trading days between two dates."""
        trading_days = []
        current_date = start_date

        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

    def get_market_calendar(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get market calendar with trading days and holidays."""
        calendar_data = []
        current_date = start_date

        while current_date <= end_date:
            market_hours = self.get_market_hours(current_date)

            if market_hours:
                calendar_data.append({
                    'date': current_date.isoformat(),
                    'day_of_week': current_date.strftime('%A'),
                    'is_trading_day': market_hours.is_trading_day,
                    'is_early_close': market_hours.early_close,
                    'holiday_name': market_hours.holiday_name,
                    'market_open': market_hours.market_open.isoformat() if market_hours.market_open else None,
                    'market_close': market_hours.market_close.isoformat() if market_hours.market_close else None
                })
            else:
                calendar_data.append({
                    'date': current_date.isoformat(),
                    'day_of_week': current_date.strftime('%A'),
                    'is_trading_day': False,
                    'is_early_close': False,
                    'holiday_name': None,
                    'market_open': None,
                    'market_close': None
                })

            current_date += timedelta(days=1)

        return calendar_data

    def get_session_duration(self, session: MarketSession, target_date: Optional[date] = None) -> timedelta:
        """Get the duration of a specific market session."""
        if target_date is None:
            target_date = datetime.now(self.timezone).date()

        market_hours = self.get_market_hours(target_date)

        if not market_hours or not market_hours.is_trading_day:
            return timedelta(0)

        if session == MarketSession.PRE_MARKET:
            if market_hours.pre_market_start and market_hours.market_open:
                return market_hours.market_open - market_hours.pre_market_start
        elif session == MarketSession.REGULAR:
            if market_hours.market_open and market_hours.market_close:
                return market_hours.market_close - market_hours.market_open
        elif session == MarketSession.AFTER_HOURS:
            if market_hours.market_close and market_hours.after_hours_end:
                return market_hours.after_hours_end - market_hours.market_close

        return timedelta(0)

    def is_session_active(self, session: MarketSession) -> bool:
        """Check if a specific session is currently active."""
        return self.get_current_session() == session

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
        """Start market hours monitoring loop."""
        logger.info("Starting market hours monitoring...")

        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._event_scheduler_loop())

    async def _monitoring_loop(self):
        """Monitor market session changes."""
        while True:
            try:
                current_session = self.get_current_session()

                # Check for session change
                if current_session != self._current_session:
                    logger.info(f"Market session changed: {self._current_session} -> {current_session}")

                    # Notify callbacks
                    for callback in self._session_change_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(self._current_session, current_session)
                            else:
                                callback(self._current_session, current_session)
                        except Exception as e:
                            logger.error(f"Session change callback failed: {e}")

                    self._current_session = current_session

                # Sleep for a minute before next check
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Market monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _event_scheduler_loop(self):
        """Schedule and trigger market events."""
        while True:
            try:
                # Check for upcoming market events
                next_event = self.get_next_market_event()

                if next_event:
                    now = datetime.now(self.timezone)
                    time_until_event = next_event.timestamp - now

                    # If event is within the next minute, trigger it
                    if timedelta(0) <= time_until_event <= timedelta(minutes=1):
                        logger.info(f"Triggering market event: {next_event.description}")
                        await self._trigger_event(next_event)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Event scheduler loop error: {e}")
                await asyncio.sleep(60)

    async def _trigger_event(self, event: MarketEventInfo):
        """Trigger a market event and notify listeners."""
        event_type = event.event_type

        if event_type in self._event_listeners:
            for listener in self._event_listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(event)
                    else:
                        listener(event)
                except Exception as e:
                    logger.error(f"Event listener failed for {event_type.value}: {e}")

    def get_market_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive market status summary."""
        now = datetime.now(self.timezone)
        today = now.date()

        market_hours = self.get_market_hours(today)
        current_session = self.get_current_session()
        next_event = self.get_next_market_event()

        if market_hours:
            market_hours_data = {
                'pre_market_start': market_hours.pre_market_start.isoformat() if market_hours.pre_market_start else None,
                'market_open': market_hours.market_open.isoformat() if market_hours.market_open else None,
                'market_close': market_hours.market_close.isoformat() if market_hours.market_close else None,
                'after_hours_end': market_hours.after_hours_end.isoformat() if market_hours.after_hours_end else None,
                'early_close': market_hours.early_close,
                'holiday_name': market_hours.holiday_name
            }
            is_trading_day = market_hours.is_trading_day
        else:
            market_hours_data = {
                'pre_market_start': None,
                'market_open': None,
                'market_close': None,
                'after_hours_end': None,
                'early_close': False,
                'holiday_name': None
            }
            is_trading_day = False

        return {
            'timestamp': now.isoformat(),
            'date': today.isoformat(),
            'current_session': current_session.value,
            'is_trading_day': is_trading_day,
            'is_market_open': self.is_market_open(),
            'is_extended_hours': self.is_extended_hours(),
            'market_hours': market_hours_data,
            'time_remaining': {
                'until_market_open': str(self.time_until_market_open()),
                'until_market_close': str(self.time_until_market_close()),
                'in_current_session': str(self.time_in_session(current_session))
            },
            'next_event': {
                'type': next_event.event_type.value if next_event else None,
                'timestamp': next_event.timestamp.isoformat() if next_event and next_event.timestamp else None,
                'description': next_event.description if next_event and next_event.description else None
            },
            'next_trading_day': (lambda d: d.isoformat() if d else None)(self.get_next_trading_day(today)),
            'previous_trading_day': (lambda d: d.isoformat() if d else None)(self.get_previous_trading_day(today))
        }

    def should_run_task(self, market_hours_only: bool = True,
                       allowed_sessions: Optional[List[MarketSession]] = None) -> bool:
        """Determine if a task should run based on market hours."""
        if not market_hours_only:
            return True

        current_session = self.get_current_session()

        if allowed_sessions:
            return current_session in allowed_sessions

        # Default: run during regular hours and pre-market
        return current_session in [MarketSession.REGULAR, MarketSession.PRE_MARKET]

    def get_market_phase_info(self) -> Dict[str, Any]:
        """Get detailed information about current market phase."""
        now = datetime.now(self.timezone)
        today = now.date()
        current_session = self.get_current_session()

        # Calculate phase progress
        phase_progress = 0.0
        phase_remaining = timedelta(0)

        if current_session != MarketSession.CLOSED:
            session_duration = self.get_session_duration(current_session, today)
            time_remaining = self.time_in_session(current_session)

            if session_duration.total_seconds() > 0:
                elapsed = session_duration - time_remaining
                phase_progress = elapsed.total_seconds() / session_duration.total_seconds()
                phase_remaining = time_remaining

        return {
            'current_session': current_session.value,
            'phase_progress': min(1.0, max(0.0, phase_progress)),
            'time_remaining': str(phase_remaining),
            'session_duration': str(self.get_session_duration(current_session, today)),
            'is_trading_active': current_session in [MarketSession.REGULAR, MarketSession.PRE_MARKET]
        }

    def get_weekly_schedule(self, start_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """Get week's trading schedule."""
        if start_date is None:
            start_date = datetime.now(self.timezone).date()

        # Get start of week (Monday)
        days_since_monday = start_date.weekday()
        week_start = start_date - timedelta(days=days_since_monday)

        weekly_schedule = []

        for i in range(7):
            day = week_start + timedelta(days=i)
            market_hours = self.get_market_hours(day)

            if market_hours:
                weekly_schedule.append({
                    'date': day.isoformat(),
                    'day_name': day.strftime('%A'),
                    'is_trading_day': market_hours.is_trading_day,
                    'market_open': market_hours.market_open.strftime('%H:%M') if market_hours.market_open else None,
                    'market_close': market_hours.market_close.strftime('%H:%M') if market_hours.market_close else None,
                    'early_close': market_hours.early_close,
                    'holiday_name': market_hours.holiday_name
                })
            else:
                weekly_schedule.append({
                    'date': day.isoformat(),
                    'day_name': day.strftime('%A'),
                    'is_trading_day': False,
                    'market_open': None,
                    'market_close': None,
                    'early_close': False,
                    'holiday_name': None
                })

        return weekly_schedule

    def calculate_trading_minutes_remaining(self) -> int:
        """Calculate trading minutes remaining in the current session."""
        if not self.is_market_open():
            return 0

        time_remaining = self.time_until_market_close()
        return int(time_remaining.total_seconds() / 60)

    def get_market_holidays(self, year: int) -> List[Dict[str, Any]]:
        """Get market holidays for a specific year."""
        try:
            # Get NYSE calendar for the year
            start_date = date(year, 1, 1)
            end_date = date(year, 12, 31)

            if not self.nyse:
                return []

            schedule = self.nyse.schedule(start_date=start_date, end_date=end_date)

            # Find gaps in the schedule to identify holidays
            holidays = []
            all_dates = set()

            # Generate all weekdays in the year
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                    all_dates.add(current_date)
                current_date += timedelta(days=1)

            # Get trading days from schedule
            # Convert index to dates properly
            try:
                # Convert schedule index to date objects
                trading_days = set()
                for idx in schedule.index:
                    if hasattr(idx, 'date'):
                        trading_days.add(idx.date())
                    else:
                        # Assume it's already a date-like object
                        trading_days.add(idx)
            except (AttributeError, TypeError):
                # Fallback: assume index contains date-like objects
                trading_days = set(schedule.index)

            # Find holidays (weekdays that are not trading days)
            holiday_dates = all_dates - trading_days

            for holiday_date in sorted(holiday_dates):
                holiday_name = self._get_holiday_name(holiday_date)
                holidays.append({
                    'date': holiday_date.isoformat(),
                    'name': holiday_name or 'Market Holiday',
                    'day_of_week': holiday_date.strftime('%A')
                })

            return holidays

        except Exception as e:
            logger.error(f"Failed to get market holidays for {year}: {e}")
            return []

    def is_extended_hours_enabled(self) -> bool:
        """Check if extended hours trading is enabled."""
        # This would check configuration or broker capabilities
        return True  # Assuming extended hours are available

    def get_extended_hours_info(self) -> Dict[str, Any]:
        """Get extended hours trading information."""
        now = datetime.now(self.timezone)
        today_hours = self.get_market_hours(now.date())
        current_session = self.get_current_session()

        if today_hours:
            return {
                'pre_market_available': today_hours.pre_market_start is not None,
                'after_hours_available': today_hours.after_hours_end is not None,
                'currently_in_extended_hours': self.is_extended_hours(),
                'current_session': current_session.value,
                'extended_hours_enabled': self.is_extended_hours_enabled()
            }
        else:
            return {
                'pre_market_available': False,
                'after_hours_available': False,
                'currently_in_extended_hours': False,
                'current_session': current_session.value,
                'extended_hours_enabled': False
            }

    async def wait_for_market_open(self):
        """Async function that waits until market opens."""
        while not self.is_market_open():
            time_until_open = self.time_until_market_open()

            if time_until_open.total_seconds() <= 0:
                break

            # Wait for market open, but check every minute
            wait_time = min(60, time_until_open.total_seconds())
            logger.info(f"Waiting for market open: {time_until_open}")
            await asyncio.sleep(wait_time)

    async def wait_for_session(self, target_session: MarketSession, timeout: Optional[int] = None):
        """Wait for a specific market session."""
        start_time = datetime.now()

        while self.get_current_session() != target_session:
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Timeout waiting for session {target_session.value}")

            await asyncio.sleep(30)  # Check every 30 seconds

    def get_session_boundaries(self, target_date: Optional[date] = None) -> Dict[str, Optional[datetime]]:
        """Get all session boundary times for a date."""
        if target_date is None:
            target_date = datetime.now(self.timezone).date()

        hours = self.get_market_hours(target_date)

        if hours:
            return {
                'pre_market_start': hours.pre_market_start,
                'market_open': hours.market_open,
                'market_close': hours.market_close,
                'after_hours_end': hours.after_hours_end
            }
        else:
            return {
                'pre_market_start': None,
                'market_open': None,
                'market_close': None,
                'after_hours_end': None
            }

    def format_market_status(self) -> str:
        """Get formatted market status string."""
        current_session = self.get_current_session()

        status_messages = {
            MarketSession.PRE_MARKET: "Pre-market trading active",
            MarketSession.REGULAR: "Market is open",
            MarketSession.AFTER_HOURS: "After-hours trading active",
            MarketSession.CLOSED: "Market is closed"
        }

        base_message = status_messages.get(current_session, "Unknown session")

        if current_session == MarketSession.CLOSED:
            time_until_open = self.time_until_market_open()
            if time_until_open.days > 0:
                base_message += f" - Opens in {time_until_open.days} days"
            else:
                hours = int(time_until_open.total_seconds() / 3600)
                minutes = int((time_until_open.total_seconds() % 3600) / 60)
                base_message += f" - Opens in {hours}h {minutes}m"
        elif current_session == MarketSession.REGULAR:
            time_until_close = self.time_until_market_close()
            hours = int(time_until_close.total_seconds() / 3600)
            minutes = int((time_until_close.total_seconds() % 3600) / 60)
            base_message += f" - Closes in {hours}h {minutes}m"

        return base_message

    def get_trading_week_info(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """Get trading information for the current week."""
        if target_date is None:
            target_date = datetime.now(self.timezone).date()

        # Get start of week (Monday)
        days_since_monday = target_date.weekday()
        week_start = target_date - timedelta(days=days_since_monday)
        week_end = week_start + timedelta(days=6)

        trading_days = self.get_trading_days_between(week_start, week_end)

        return {
            'week_start': week_start.isoformat(),
            'week_end': week_end.isoformat(),
            'trading_days': [day.isoformat() for day in trading_days],
            'trading_days_count': len(trading_days),
            'total_trading_hours': sum(
                self.get_session_duration(MarketSession.REGULAR, day).total_seconds() / 3600
                for day in trading_days
            )
        }

    def clear_cache(self):
        """Clear the market hours cache."""
        self._market_hours_cache.clear()
        logger.info("Market hours cache cleared")

    def preload_cache(self, days_ahead: int = 30):
        """Preload market hours cache for upcoming days."""
        today = datetime.now(self.timezone).date()

        for i in range(days_ahead):
            future_date = today + timedelta(days=i)
            self.get_market_hours(future_date)

        logger.info(f"Preloaded market hours cache for {days_ahead} days")

    def validate_trading_time(self, target_time: datetime,
                            session_required: Optional[MarketSession] = None) -> bool:
        """Validate if a given time is appropriate for trading."""
        session_at_time = self.get_session_at_time(target_time)

        if session_required:
            return session_at_time == session_required

        # Default: valid during regular hours or pre-market
        return session_at_time in [MarketSession.REGULAR, MarketSession.PRE_MARKET]
