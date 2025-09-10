"""
Utilities for Trade Execution Service.

This module provides error handling, audit logging, retry mechanisms,
and various helper functions for the trade execution service.
"""

import asyncio
import functools
import json
import logging
import math
import time
import traceback
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID

import asyncpg
import pytz
import redis.asyncio as redis

from shared.config import get_config

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Custom exception for execution errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)


class AuditLogger:
    """
    Comprehensive audit logging system for trade execution.

    Logs all trading activities, decisions, and system events
    with detailed context for compliance and debugging.
    """

    def __init__(
        self, db_pool: Optional[asyncpg.Pool], redis_client: Optional[redis.Redis]
    ):
        """Initialize audit logger."""
        self.db_pool = db_pool
        self.redis = redis_client
        self.config = get_config()

    async def log_trade_decision(
        self,
        signal_id: UUID,
        symbol: str,
        decision: str,
        reasoning: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a trade decision."""
        try:
            audit_entry = {
                "event_type": "TRADE_DECISION",
                "signal_id": str(signal_id),
                "symbol": symbol,
                "decision": decision,
                "reasoning": reasoning,
                "context": context or {},
                "timestamp": datetime.now(timezone.utc),
                "service": "trade_executor",
            }

            await self._store_audit_entry(audit_entry)

        except Exception as e:
            logger.error(f"Failed to log trade decision: {e}")

    async def log_order_event(
        self,
        order_id: UUID,
        broker_order_id: str,
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        """Log order-related events."""
        try:
            audit_entry = {
                "event_type": f"ORDER_{event_type.upper()}",
                "order_id": str(order_id),
                "broker_order_id": broker_order_id,
                "details": details,
                "timestamp": datetime.now(timezone.utc),
                "service": "trade_executor",
            }

            await self._store_audit_entry(audit_entry)

        except Exception as e:
            logger.error(f"Failed to log order event: {e}")

    async def log_position_event(
        self, position_id: UUID, symbol: str, event_type: str, details: Dict[str, Any]
    ) -> None:
        """Log position-related events."""
        try:
            audit_entry = {
                "event_type": f"POSITION_{event_type.upper()}",
                "position_id": str(position_id),
                "symbol": symbol,
                "details": details,
                "timestamp": datetime.now(timezone.utc),
                "service": "trade_executor",
            }

            await self._store_audit_entry(audit_entry)

        except Exception as e:
            logger.error(f"Failed to log position event: {e}")

    async def log_risk_event(
        self,
        event_type: str,
        symbol: str,
        severity: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log risk management events."""
        try:
            audit_entry = {
                "event_type": f"RISK_{event_type.upper()}",
                "severity": severity,
                "symbol": symbol,
                "description": description,
                "context": context or {},
                "timestamp": datetime.now(timezone.utc),
                "service": "trade_executor",
            }

            await self._store_audit_entry(audit_entry)

        except Exception as e:
            logger.error(f"Failed to log risk event: {e}")

    async def log_system_event(
        self, event_type: str, message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log system events."""
        try:
            audit_entry = {
                "event_type": f"SYSTEM_{event_type.upper()}",
                "message": message,
                "context": context or {},
                "timestamp": datetime.now(timezone.utc),
                "service": "trade_executor",
            }

            await self._store_audit_entry(audit_entry)

        except Exception as e:
            logger.error(f"Failed to log system event: {e}")

    async def _store_audit_entry(self, entry: Dict[str, Any]) -> None:
        """Store audit entry in database and Redis."""
        try:
            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO audit.execution_log (
                            event_type, symbol, details, timestamp, service
                        ) VALUES ($1, $2, $3, $4, $5)
                    """,
                        entry["event_type"],
                        entry.get("symbol"),
                        json.dumps(entry["details"], default=str),
                        entry["timestamp"],
                        entry["service"],
                    )

            # Also publish to Redis for real-time monitoring
            if self.redis:
                await self.redis.publish("audit_log", json.dumps(entry, default=str))

        except Exception as e:
            logger.error(f"Failed to store audit entry: {e}")


class RetryManager:
    """
    Advanced retry management with exponential backoff,
    circuit breaker patterns, and dead letter queue.
    """

    def __init__(self, redis_client: redis.Redis):
        """Initialize retry manager."""
        self.redis = redis_client
        self.config = get_config()
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}

    async def with_retry(
        self,
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: tuple = (Exception,),
        circuit_breaker_key: Optional[str] = None,
    ) -> Any:
        """
        Execute function with advanced retry logic.

        Args:
            func: Function to execute
            max_attempts: Maximum retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Exponential backoff base
            jitter: Add random jitter to delays
            retry_on: Tuple of exceptions to retry on
            circuit_breaker_key: Key for circuit breaker

        Returns:
            Function result
        """
        attempt = 0
        last_exception = None

        # Check circuit breaker
        if circuit_breaker_key and await self._is_circuit_open(circuit_breaker_key):
            raise ExecutionError(f"Circuit breaker open for {circuit_breaker_key}")

        while attempt < max_attempts:
            try:
                result = await func()

                # Reset circuit breaker on success
                if circuit_breaker_key:
                    await self._reset_circuit_breaker(circuit_breaker_key)

                return result

            except retry_on as e:
                last_exception = e
                attempt += 1

                if attempt >= max_attempts:
                    # Record failure in circuit breaker
                    if circuit_breaker_key:
                        await self._record_circuit_failure(circuit_breaker_key)

                    # Send to dead letter queue
                    await self._send_to_dlq(
                        func.__name__,
                        str(e),
                        {"attempts": attempt, "last_error": str(e)},
                    )

                    break

                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)

                if jitter:
                    import random

                    delay *= 0.5 + random.random()

                logger.warning(
                    f"Retry {attempt}/{max_attempts} for {func.__name__} after {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)

        if "last_exception" in locals() and last_exception:
            raise last_exception
        else:
            raise ExecutionError("Maximum retry attempts exceeded")

    async def _is_circuit_open(self, key: str) -> bool:
        """Check if circuit breaker is open."""
        try:
            if self.redis:
                circuit_data = self.redis.hgetall(f"circuit:{key}")
                if hasattr(circuit_data, "__await__"):
                    circuit_data = await circuit_data
            else:
                return False
            if not circuit_data:
                return False

            failure_count = int(circuit_data.get("failures", 0))
            last_failure = circuit_data.get("last_failure")

            if failure_count >= 5:  # Failure threshold
                if last_failure:
                    last_failure_time = datetime.fromisoformat(last_failure)
                    if datetime.now(timezone.utc) - last_failure_time < timedelta(
                        minutes=5
                    ):  # Cool-down period
                        return True

            return False

        except Exception as e:
            logger.error(f"Error checking circuit breaker {key}: {e}")
            return False

    async def _record_circuit_failure(self, key: str) -> None:
        """Record failure in circuit breaker."""
        try:
            result = self.redis.hincrby(f"circuit:{key}", "failures", 1)
            if hasattr(result, "__await__"):
                await result
            hset_result = self.redis.hset(
                f"circuit:{key}", "last_failure", datetime.now(timezone.utc).isoformat()
            )
            if hasattr(hset_result, "__await__"):
                await hset_result
            expire_result = self.redis.expire(
                f"circuit:{key}", 3600
            )  # Expire after 1 hour
            if hasattr(expire_result, "__await__"):
                await expire_result

        except Exception as e:
            logger.error(f"Error recording circuit failure {key}: {e}")

    async def _reset_circuit_breaker(self, key: str) -> None:
        """Reset circuit breaker on success."""
        try:
            delete_result = self.redis.delete(f"circuit:{key}")
            if hasattr(delete_result, "__await__"):
                await delete_result

        except Exception as e:
            logger.error(f"Error resetting circuit breaker {key}: {e}")

    async def _send_to_dlq(
        self, operation: str, error: str, context: Dict[str, Any]
    ) -> None:
        """Send failed operation to dead letter queue."""
        try:
            dlq_entry = {
                "operation": operation,
                "error": error,
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retry_count": context.get("attempts", 0),
            }

            lpush_result = self.redis.lpush(
                "execution_dlq", json.dumps(dlq_entry, default=str)
            )
            if hasattr(lpush_result, "__await__"):
                await lpush_result

            # Keep only last 1000 entries
            ltrim_result = self.redis.ltrim("execution_dlq", 0, 999)
            if hasattr(ltrim_result, "__await__"):
                await ltrim_result

            logger.error(f"Sent to DLQ: {operation} - {error}")

        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")


class DecimalUtils:
    """Utilities for precise decimal calculations in trading."""

    @staticmethod
    def round_to_tick(price: Decimal, tick_size: Decimal = Decimal("0.01")) -> Decimal:
        """
        Round price to nearest tick size.

        Args:
            price: Price to round
            tick_size: Minimum tick size

        Returns:
            Rounded price
        """
        return (price / tick_size).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        ) * tick_size

    @staticmethod
    def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> Decimal:
        """Calculate percentage change between two values."""
        if old_value == 0:
            return Decimal("0")
        return (new_value - old_value) / old_value * 100

    @staticmethod
    def calculate_position_value(quantity: int, price: Decimal) -> Decimal:
        """Calculate position value with proper decimal precision."""
        return Decimal(str(abs(quantity))) * price

    @staticmethod
    def calculate_pnl(
        entry_price: Decimal, exit_price: Decimal, quantity: int, side: str
    ) -> Decimal:
        """
        Calculate P&L for a trade.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity
            side: Position side ('long' or 'short')

        Returns:
            Realized P&L
        """
        if side.lower() == "long":
            return (exit_price - entry_price) * abs(quantity)
        else:  # short
            return (entry_price - exit_price) * abs(quantity)

    @staticmethod
    def format_currency(amount: Union[Decimal, float], currency: str = "USD") -> str:
        """Format amount as currency string."""
        return f"{currency} {amount:,.2f}"

    @staticmethod
    def safe_divide(
        numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
    ) -> Decimal:
        """Safe division that handles zero denominator."""
        if denominator == 0:
            return default
        return numerator / denominator


class MarketDataUtils:
    """Utilities for market data processing and analysis."""

    @staticmethod
    def calculate_volatility(prices: List[Decimal], periods: int = 20) -> Decimal:
        """
        Calculate price volatility using standard deviation of returns.

        Args:
            prices: List of prices
            periods: Number of periods for calculation

        Returns:
            Volatility as standard deviation
        """
        if len(prices) < 2:
            return Decimal("0")

        # Calculate returns
        returns: List[Decimal] = []
        for i in range(1, min(len(prices), periods + 1)):
            ret = (Decimal(str(prices[i])) - Decimal(str(prices[i - 1]))) / Decimal(
                str(prices[i - 1])
            )
            returns.append(ret)

        if not returns:
            return Decimal("0")

        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum(
            float((float(ret) - float(mean_return)) ** 2) for ret in returns
        ) / len(returns)

        return (
            Decimal(str(math.sqrt(float(variance)))) if variance > 0 else Decimal("0")
        )

    @staticmethod
    def calculate_rsi(prices: List[Decimal], periods: int = 14) -> Optional[Decimal]:
        """
        Calculate Relative Strength Index.

        Args:
            prices: List of prices
            periods: RSI period

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(prices) < periods + 1:
            return None

        gains = []
        losses = []

        for i in range(1, periods + 1):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        avg_gain = sum(gains) / Decimal(str(periods))
        avg_loss = sum(losses) / Decimal(str(periods))

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return Decimal(str(rsi))

    @staticmethod
    def is_market_hours() -> bool:
        """Check if current time is within market hours."""
        from datetime import datetime

        import pytz

        ny_tz = pytz.timezone("America/New_York")
        ny_time = datetime.now(ny_tz)

        # Basic market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
        if ny_time.weekday() >= 5:  # Weekend
            return False

        market_open = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= ny_time <= market_close


class PerformanceUtils:
    """Utilities for performance calculations and analysis."""

    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[Decimal], risk_free_rate: Decimal = Decimal("0.02")
    ) -> Optional[Decimal]:
        """
        Calculate Sharpe ratio.

        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return None

        # Convert to daily risk-free rate
        daily_rf_rate = risk_free_rate / 252

        # Calculate excess returns
        excess_returns = [
            Decimal(str(float(ret))) - Decimal(str(float(daily_rf_rate)))
            for ret in returns
        ]

        # Calculate mean and std
        mean_excess = sum(excess_returns) / len(excess_returns)

        if len(excess_returns) < 2:
            return None

        variance = sum(
            (Decimal(str(float(ret))) - Decimal(str(float(mean_excess)))) ** 2
            for ret in excess_returns
        ) / (len(excess_returns) - 1)
        std_excess = (
            Decimal(str(math.sqrt(float(variance)))) if variance > 0 else Decimal("0")
        )

        if std_excess == 0:
            return None

        # Annualize
        sharpe = (Decimal(str(float(mean_excess))) / std_excess) * Decimal(
            str(math.sqrt(252))
        )
        # Clamp the value to database field limits (DECIMAL(8,6) = ±99.999999)
        return max(Decimal("-99.999999"), min(Decimal("99.999999"), sharpe))

    @staticmethod
    def calculate_sortino_ratio(
        returns: List[Decimal], risk_free_rate: Decimal = Decimal("0.02")
    ) -> Optional[Decimal]:
        """
        Calculate Sortino ratio (using downside deviation).

        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return None

        daily_rf_rate = risk_free_rate / 252
        excess_returns = [ret - daily_rf_rate for ret in returns]

        # Calculate downside returns only
        downside_returns = [ret for ret in excess_returns if ret < 0]

        if not downside_returns:
            return None

        mean_excess = sum(excess_returns) / len(excess_returns)
        downside_variance = sum(
            Decimal(str(float(ret))) ** 2 for ret in downside_returns
        ) / len(downside_returns)
        std_downside = (
            Decimal(str(math.sqrt(float(downside_variance))))
            if downside_variance > 0
            else Decimal("0")
        )

        if std_downside == 0:
            return None

        # Annualize
        sortino = (Decimal(str(float(mean_excess))) / std_downside) * Decimal(
            str(math.sqrt(252))
        )
        return sortino

    @staticmethod
    def calculate_max_drawdown(values: List[Decimal]) -> Dict[str, Any]:
        """
        Calculate maximum drawdown from a series of values.

        Args:
            values: List of portfolio values

        Returns:
            Dictionary with drawdown metrics
        """
        if len(values) < 2:
            return {"max_drawdown": 0, "drawdown_duration": 0}

        peak = values[0]
        max_drawdown = Decimal("0")
        max_duration = 0
        current_duration = 0
        drawdown_start = None

        for i, value in enumerate(values):
            if value > peak:
                peak = value
                if drawdown_start is not None:
                    max_duration = max(max_duration, current_duration)
                    drawdown_start = None
                    current_duration = 0
            else:
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                if drawdown_start is None:
                    drawdown_start = i
                    current_duration = 1
                else:
                    current_duration = i - drawdown_start + 1

        # Check if we're still in drawdown
        if drawdown_start is not None:
            max_duration = max(max_duration, current_duration)

        return {
            "max_drawdown": float(max_drawdown),
            "max_drawdown_duration": max_duration,
            "current_drawdown": float((peak - values[-1]) / peak) if peak > 0 else 0,
        }


class ValidationUtils:
    """Utilities for trade and order validation."""

    @staticmethod
    def validate_price_levels(
        entry_price: Decimal,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal],
        side: str,
    ) -> List[str]:
        """
        Validate price levels for logical consistency.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            side: Position side ('long' or 'short')

        Returns:
            List of validation errors
        """
        errors = []

        if side.lower() == "long":
            if stop_loss and stop_loss >= entry_price:
                errors.append("Stop loss must be below entry price for long positions")

            if take_profit and take_profit <= entry_price:
                errors.append(
                    "Take profit must be above entry price for long positions"
                )

        elif side.lower() == "short":
            if stop_loss and stop_loss <= entry_price:
                errors.append("Stop loss must be above entry price for short positions")

            if take_profit and take_profit >= entry_price:
                errors.append(
                    "Take profit must be below entry price for short positions"
                )

        # Check that stop loss and take profit make sense relative to each other
        if stop_loss and take_profit:
            if side.lower() == "long" and stop_loss >= take_profit:
                errors.append("Stop loss must be below take profit for long positions")
            elif side.lower() == "short" and stop_loss <= take_profit:
                errors.append("Stop loss must be above take profit for short positions")

        return errors

    @staticmethod
    def validate_quantity(quantity: int, max_quantity: int = 100000) -> List[str]:
        """Validate order quantity."""
        errors = []

        if quantity <= 0:
            errors.append("Quantity must be positive")

        if quantity > max_quantity:
            errors.append(f"Quantity exceeds maximum allowed: {max_quantity}")

        return errors

    @staticmethod
    def validate_symbol(symbol: str) -> List[str]:
        """Validate trading symbol format."""
        errors = []

        if not symbol:
            errors.append("Symbol cannot be empty")
            return errors

        if len(symbol) > 10:
            errors.append("Symbol too long (max 10 characters)")

        if not symbol.isalpha():
            errors.append("Symbol must contain only letters")

        return errors


class TimeUtils:
    """Utilities for time and date handling."""

    @staticmethod
    def get_market_timezone() -> pytz.BaseTzInfo:
        """Get market timezone."""
        return pytz.timezone("America/New_York")

    @staticmethod
    def to_market_time(dt: datetime) -> datetime:
        """Convert datetime to market timezone."""
        import pytz

        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        return dt.astimezone(TimeUtils.get_market_timezone())

    @staticmethod
    def get_next_market_open() -> datetime:
        """Get next market open time."""
        # import pytz  # Commented out unused import
        from datetime import time

        ny_tz = TimeUtils.get_market_timezone()
        now = datetime.now(ny_tz)

        # Market opens at 9:30 AM ET
        market_open_time = time(9, 30)

        # If before market open today, return today's open
        if now.time() < market_open_time and now.weekday() < 5:
            return ny_tz.localize(datetime.combine(now.date(), market_open_time))

        # Otherwise, find next business day
        days_ahead = 1
        while True:
            next_day = now + timedelta(days=days_ahead)
            if next_day.weekday() < 5:  # Monday = 0, Friday = 4
                return ny_tz.localize(
                    datetime.combine(next_day.date(), market_open_time)
                )
            days_ahead += 1

    @staticmethod
    def get_trading_days_between(start_date: datetime, end_date: datetime) -> int:
        """Count trading days between two dates."""
        current = start_date.date()
        end = end_date.date()
        trading_days = 0

        while current <= end:
            if current.weekday() < 5:  # Monday-Friday
                trading_days += 1
            current += timedelta(days=1)

        return trading_days


class NotificationUtils:
    """Utilities for sending notifications and alerts."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize notification utilities."""
        self.redis = redis_client
        self.config = get_config()

    async def send_execution_alert(
        self,
        symbol: str,
        message: str,
        severity: str = "info",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send execution-related alert."""
        try:
            alert = {
                "type": "execution_alert",
                "symbol": symbol,
                "message": message,
                "severity": severity,
                "context": context or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "trade_executor",
            }

            # Publish to alerts channel
            publish_result = self.redis.publish(
                "alerts:execution", json.dumps(alert, default=str)
            )
            if hasattr(publish_result, "__await__"):
                await publish_result

            # Store in alerts list for persistence
            lpush_result = self.redis.lpush(
                "alerts:history", json.dumps(alert, default=str)
            )
            if hasattr(lpush_result, "__await__"):
                await lpush_result
            ltrim_result = self.redis.ltrim(
                "alerts:history", 0, 999
            )  # Keep last 1000 alerts
            if hasattr(ltrim_result, "__await__"):
                await ltrim_result

            logger.info(f"Alert sent: {symbol} - {message}")

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def send_risk_alert(
        self,
        position_id: UUID,
        symbol: str,
        risk_type: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send risk management alert."""
        try:
            alert = {
                "type": "risk_alert",
                "position_id": str(position_id),
                "symbol": symbol,
                "risk_type": risk_type,
                "message": message,
                "metrics": metrics or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "trade_executor",
            }

            publish_result = self.redis.publish(
                "alerts:risk", json.dumps(alert, default=str)
            )
            if hasattr(publish_result, "__await__"):
                await publish_result
            lpush_result = self.redis.lpush(
                "alerts:risk_history", json.dumps(alert, default=str)
            )
            if hasattr(lpush_result, "__await__"):
                await lpush_result
            ltrim_result = self.redis.ltrim("alerts:risk_history", 0, 999)
            if hasattr(ltrim_result, "__await__"):
                await ltrim_result

            logger.warning(f"Risk alert: {symbol} - {message}")

        except Exception as e:
            logger.error(f"Failed to send risk alert: {e}")

    async def send_performance_notification(
        self, metrics: Dict[str, Any], notification_type: str = "daily_summary"
    ) -> None:
        """Send performance notification."""
        try:
            notification = {
                "type": "performance_notification",
                "notification_type": notification_type,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "trade_executor",
            }

            await self.redis.publish(
                "notifications:performance", json.dumps(notification, default=str)
            )

            logger.info(f"Performance notification sent: {notification_type}")

        except Exception as e:
            logger.error(f"Failed to send performance notification: {e}")


class CacheManager:
    """Redis-based cache manager for execution data."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize cache manager."""
        self.redis = redis_client

    async def cache_position_data(
        self, symbol: str, position_data: Dict[str, Any], ttl: int = 300
    ) -> None:
        """Cache position data."""
        try:
            cache_key = f"position:{symbol}"
            await self.redis.setex(
                cache_key, ttl, json.dumps(position_data, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to cache position data for {symbol}: {e}")

    async def get_cached_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached position data."""
        try:
            cache_key = f"position:{symbol}"
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get cached position for {symbol}: {e}")
            return None

    async def cache_market_data(
        self, symbol: str, data: Dict[str, Any], ttl: int = 60
    ) -> None:
        """Cache market data."""
        try:
            cache_key = f"market_data:{symbol}"
            await self.redis.setex(cache_key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Failed to cache market data for {symbol}: {e}")

    async def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data."""
        try:
            cache_key = f"market_data:{symbol}"
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get cached market data for {symbol}: {e}")
            return None

    async def invalidate_cache(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries matching {pattern}")
        except Exception as e:
            logger.error(f"Failed to invalidate cache {pattern}: {e}")


def rate_limit(max_calls: int, period: int) -> Callable:
    """
    Rate limiting decorator.

    Args:
        max_calls: Maximum calls allowed
        period: Time period in seconds
    """

    def decorator(func: Callable) -> Callable:
        calls: list = []

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.time()

            # Remove old calls
            calls[:] = [call_time for call_time in calls if now - call_time < period]

            # Check rate limit
            if len(calls) >= max_calls:
                raise ExecutionError(
                    f"Rate limit exceeded: {max_calls} calls per {period} seconds"
                )

            # Record this call
            calls.append(now)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper


class ErrorHandler:
    """Centralized error handling for the execution service."""

    def __init__(
        self, audit_logger: AuditLogger, notification_utils: NotificationUtils
    ):
        """Initialize error handler."""
        self.audit_logger = audit_logger
        self.notification_utils = notification_utils

    async def handle_execution_error(
        self, error: Exception, context: Dict[str, Any], severity: str = "error"
    ) -> None:
        """Handle execution errors with proper logging and notifications."""
        try:
            error_details = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "stack_trace": traceback.format_exc(),
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Log to audit system
            await self.audit_logger.log_system_event(
                "ERROR", f"Execution error: {error}", error_details
            )

            # Send notification for critical errors
            if severity in ["critical", "error"]:
                await self.notification_utils.send_execution_alert(
                    context.get("symbol", "SYSTEM"),
                    f"Execution error: {error}",
                    severity,
                    error_details,
                )

            logger.error(f"Handled execution error: {error}")

        except Exception as e:
            logger.error(f"Error handler failed: {e}")

    async def handle_api_error(
        self, error: Exception, api_name: str, operation: str, context: Dict[str, Any]
    ) -> None:
        """Handle API-specific errors."""
        try:
            error_details = {
                "api_name": api_name,
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await self.audit_logger.log_system_event(
                "API_ERROR", f"{api_name} {operation} failed: {error}", error_details
            )

            # Specific handling for different APIs
            if api_name.lower() == "alpaca":
                await self._handle_alpaca_error(error, operation, context)

        except Exception as e:
            logger.error(f"API error handler failed: {e}")

    async def _handle_alpaca_error(
        self, error: Exception, operation: str, context: Dict[str, Any]
    ) -> None:
        """Handle Alpaca-specific errors."""
        error_str = str(error).lower()

        # Insufficient funds
        if "insufficient" in error_str and "buying power" in error_str:
            await self.notification_utils.send_execution_alert(
                context.get("symbol", "SYSTEM"),
                f"Insufficient buying power for {operation}",
                "warning",
                context,
            )

        # Market closed
        elif "market" in error_str and "closed" in error_str:
            logger.info(
                f"Market closed error for {operation} - this is expected outside market hours"
            )

        # Rate limiting
        elif "rate limit" in error_str or "429" in error_str:
            await self.notification_utils.send_execution_alert(
                context.get("symbol", "SYSTEM"),
                f"Alpaca rate limit hit during {operation}",
                "warning",
                context,
            )

        # Symbol not found
        elif "symbol" in error_str and (
            "not found" in error_str or "invalid" in error_str
        ):
            await self.notification_utils.send_execution_alert(
                context.get("symbol", "UNKNOWN"),
                f"Invalid symbol in {operation}",
                "error",
                context,
            )


class HealthChecker:
    """System health monitoring utilities."""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        """Initialize health checker."""
        self.db_pool = db_pool
        self.redis = redis_client

    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()

            async with self.db_pool.acquire() as conn:
                # Simple connectivity test
                await conn.fetchval("SELECT 1")

                # Check table accessibility
                tables_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'trading'
                """
                )

                # Check recent activity
                recent_orders = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM trading.orders
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                """
                )

            latency = time.time() - start_time

            return {
                "status": "healthy",
                "latency_seconds": latency,
                "trading_tables_count": tables_count,
                "recent_orders_1h": recent_orders,
                "pool_size": self.db_pool.get_size(),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

    async def check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            start_time = time.time()

            # Ping test
            await self.redis.ping()

            # Set/get test
            test_key = f"health_check:{int(time.time())}"
            await self.redis.setex(test_key, 60, "health_check")
            await self.redis.get(test_key)
            await self.redis.delete(test_key)

            latency = time.time() - start_time

            # Get Redis info
            info = await self.redis.info()

            return {
                "status": "healthy",
                "latency_seconds": latency,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

    async def check_overall_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            db_health = await self.check_database_health()
            redis_health = await self.check_redis_health()

            overall_status = "healthy"
            if db_health["status"] != "healthy" or redis_health["status"] != "healthy":
                overall_status = "degraded"

            return {
                "overall_status": overall_status,
                "database": db_health,
                "redis": redis_health,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }


class ConfigValidator:
    """Validate configuration for trade execution service."""

    @staticmethod
    def validate_alpaca_config(config: Any) -> List[str]:
        """Validate Alpaca configuration."""
        errors = []

        if not config.alpaca.api_key:
            errors.append("Alpaca API key is required")

        if not config.alpaca.secret_key:
            errors.append("Alpaca secret key is required")

        if config.alpaca.paper_trading and "paper" not in config.alpaca.base_url:
            errors.append("Paper trading enabled but live trading URL configured")

        if not config.alpaca.paper_trading and "paper" in config.alpaca.base_url:
            errors.append("Live trading mode but paper trading URL configured")

        return errors

    @staticmethod
    def validate_risk_config(config: Any) -> List[str]:
        """Validate risk configuration."""
        errors = []

        if config.risk.max_position_size <= 0 or config.risk.max_position_size > 1:
            errors.append("Max position size must be between 0 and 1")

        if config.risk.max_portfolio_risk <= 0 or config.risk.max_portfolio_risk > 0.1:
            errors.append("Max portfolio risk should be between 0 and 0.1 (10%)")

        if config.risk.stop_loss_percentage <= 0:
            errors.append("Stop loss percentage must be positive")

        if config.risk.take_profit_percentage <= config.risk.stop_loss_percentage:
            errors.append("Take profit should be larger than stop loss percentage")

        return errors

    @staticmethod
    def validate_database_config(config: Any) -> List[str]:
        """Validate database configuration."""
        errors = []

        if not config.database.host:
            errors.append("Database host is required")

        if not config.database.password:
            errors.append("Database password is required")

        if config.database.pool_size <= 0:
            errors.append("Database pool size must be positive")

        return errors

    @staticmethod
    def validate_all_config(config: Any) -> Dict[str, List[str]]:
        """Validate all configuration sections."""
        validation_results = {
            "alpaca": ConfigValidator.validate_alpaca_config(config),
            "risk": ConfigValidator.validate_risk_config(config),
            "database": ConfigValidator.validate_database_config(config),
        }

        return {k: v for k, v in validation_results.items() if v}


class MetricsCollector:
    """Collect and publish execution metrics."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize metrics collector."""
        self.redis = redis_client
        self._metrics_buffer: dict[str, Any] = {}

    async def record_execution_time(
        self, operation: str, execution_time: float
    ) -> None:
        """Record execution time for an operation."""
        try:
            metric_key = f"metrics:execution_time:{operation}"
            lpush_result = self.redis.lpush(metric_key, execution_time)
            if hasattr(lpush_result, "__await__"):
                await lpush_result
            ltrim_result = self.redis.ltrim(
                metric_key, 0, 999
            )  # Keep last 1000 measurements
            if hasattr(ltrim_result, "__await__"):
                await ltrim_result
            expire_result = self.redis.expire(
                metric_key, 3600 * 24
            )  # Expire after 24 hours
            if hasattr(expire_result, "__await__"):
                await expire_result

        except Exception as e:
            logger.error(f"Failed to record execution time: {e}")

    async def record_counter(self, metric_name: str, value: int = 1) -> None:
        """Record counter metric."""
        try:
            incrby_result = self.redis.incrby(f"metrics:counter:{metric_name}", value)
            if hasattr(incrby_result, "__await__"):
                await incrby_result
            expire_result = self.redis.expire(
                f"metrics:counter:{metric_name}", 3600 * 24
            )
            if hasattr(expire_result, "__await__"):
                await expire_result

        except Exception as e:
            logger.error(f"Failed to record counter: {e}")

    async def record_gauge(self, metric_name: str, value: float) -> None:
        """Record gauge metric."""
        try:
            set_result = self.redis.set(f"metrics:gauge:{metric_name}", value)
            if hasattr(set_result, "__await__"):
                await set_result
            expire_result = self.redis.expire(f"metrics:gauge:{metric_name}", 3600 * 24)
            if hasattr(expire_result, "__await__"):
                await expire_result

        except Exception as e:
            logger.error(f"Failed to record gauge: {e}")

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        try:
            # Get all metric keys (redis.asyncio always returns awaitables)
            counter_keys = await self.redis.keys("metrics:counter:*")
            gauge_keys = await self.redis.keys("metrics:gauge:*")
            execution_time_keys = await self.redis.keys("metrics:execution_time:*")

            metrics: dict[str, Any] = {
                "counters": {},
                "gauges": {},
                "execution_times": {},
            }

            # Get counter values
            if counter_keys:
                for key in counter_keys:
                    metric_name = key.decode().replace("metrics:counter:", "")
                    value = await self.redis.get(key)
                    metrics["counters"][metric_name] = int(value) if value else 0

            # Get gauge values
            if gauge_keys:
                for key in gauge_keys:
                    metric_name = key.decode().replace("metrics:gauge:", "")
                    value = await self.redis.get(key)
                    metrics["gauges"][metric_name] = float(value) if value else 0.0

            # Get execution time averages
            if execution_time_keys:
                for key in execution_time_keys:
                    metric_name = key.decode().replace("metrics:execution_time:", "")
                    values_result = self.redis.lrange(key, 0, -1)
                    if hasattr(values_result, "__await__"):
                        values = await values_result
                    else:
                        values = values_result
                    if values:
                        avg_time = sum(float(v) for v in values) / len(values)
                        metrics["execution_times"][metric_name] = {
                            "avg": avg_time,
                            "count": len(values),
                            "latest": float(values[0]) if values else 0,
                        }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}


def safe_json_dumps(obj: Any, default: Callable = str) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, default=default)
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        return json.dumps({"error": "serialization_failed", "type": str(type(obj))})


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely deserialize JSON string."""
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON deserialization failed: {e}")
        return default


def format_decimal(value: Decimal, precision: int = 2) -> str:
    """Format decimal for display."""
    format_str = f"{{:.{precision}f}}"
    return format_str.format(float(value))


def truncate_string(text: str, max_length: int = 100) -> str:
    """Truncate string to maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


async def ensure_database_schema(db_pool: asyncpg.Pool) -> bool:
    """Ensure all required database tables exist."""
    try:
        async with db_pool.acquire() as conn:
            # Check if trading schema exists
            schema_exists = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.schemata
                    WHERE schema_name = 'trading'
                )
            """
            )

            if not schema_exists:
                logger.warning(
                    "Trading schema does not exist - run database migration scripts"
                )
                return False

            # Check required tables
            required_tables = [
                "positions",
                "orders",
                "fills",
                "trade_performance",
                "daily_performance",
                "execution_errors",
                "bracket_orders",
            ]

            for table in required_tables:
                table_exists = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'trading' AND table_name = $1
                    )
                """,
                    table,
                )

                if not table_exists:
                    logger.error(f"Required table trading.{table} does not exist")
                    return False

            logger.info("Database schema validation passed")
            return True

    except Exception as e:
        logger.error(f"Database schema validation failed: {e}")
        return False


async def create_audit_schema(db_pool: asyncpg.Pool) -> bool:
    """Create audit schema if it doesn't exist."""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                CREATE SCHEMA IF NOT EXISTS audit;

                CREATE TABLE IF NOT EXISTS audit.execution_log (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    event_type VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20),
                    details JSONB,
                    timestamp TIMESTAMPTZ NOT NULL,
                    service VARCHAR(50) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_execution_log_event_type
                ON audit.execution_log(event_type);

                CREATE INDEX IF NOT EXISTS idx_execution_log_symbol
                ON audit.execution_log(symbol);

                CREATE INDEX IF NOT EXISTS idx_execution_log_timestamp
                ON audit.execution_log(timestamp);

                CREATE INDEX IF NOT EXISTS idx_execution_log_service
                ON audit.execution_log(service);
            """
            )

        logger.info("Audit schema created/verified")
        return True

    except Exception as e:
        logger.error(f"Failed to create audit schema: {e}")
        return False


def calculate_position_sizing(
    account_value: Decimal,
    risk_per_trade: Decimal,
    entry_price: Decimal,
    stop_loss: Decimal,
    max_position_size: Decimal,
) -> Dict[str, Any]:
    """
    Calculate optimal position sizing based on risk parameters.

    Args:
        account_value: Total account value
        risk_per_trade: Amount to risk per trade
        entry_price: Entry price
        stop_loss: Stop loss price
        max_position_size: Maximum position size as % of account

    Returns:
        Position sizing information
    """
    try:
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            return {
                "shares": 0,
                "position_value": Decimal("0"),
                "risk_amount": Decimal("0"),
                "error": "No risk per share (entry price equals stop loss)",
            }

        # Calculate shares based on risk
        shares_by_risk = int(risk_per_trade / risk_per_share)

        # Calculate shares based on max position size
        max_position_value = account_value * max_position_size
        shares_by_position_limit = int(max_position_value / entry_price)

        # Take the minimum (most conservative)
        optimal_shares = min(shares_by_risk, shares_by_position_limit)
        optimal_shares = max(1, optimal_shares)  # Minimum 1 share

        position_value = optimal_shares * entry_price
        actual_risk = optimal_shares * risk_per_share

        return {
            "shares": optimal_shares,
            "position_value": position_value,
            "risk_amount": actual_risk,
            "risk_percentage": (actual_risk / account_value) * 100,
            "position_percentage": (position_value / account_value) * 100,
            "shares_by_risk": shares_by_risk,
            "shares_by_position_limit": shares_by_position_limit,
        }

    except Exception as e:
        logger.error(f"Position sizing calculation failed: {e}")
        return {
            "shares": 0,
            "position_value": Decimal("0"),
            "risk_amount": Decimal("0"),
            "error": str(e),
        }


def is_tradeable_symbol(symbol: str) -> bool:
    """Check if symbol is tradeable (basic validation)."""
    if not symbol or len(symbol) > 10:
        return False

    # Basic symbol format validation
    if not symbol.isalpha():
        return False

    # Check against common non-tradeable patterns
    non_tradeable_patterns = ["TEST", "DEMO", "FAKE"]
    if any(pattern in symbol.upper() for pattern in non_tradeable_patterns):
        return False

    return True


def calculate_slippage(expected_price: Decimal, actual_price: Decimal) -> Decimal:
    """Calculate slippage between expected and actual execution price."""
    if expected_price == 0:
        return Decimal("0")
    return abs(actual_price - expected_price) / expected_price


def get_market_session(dt: Optional[datetime] = None) -> str:
    """
    Determine market session for given datetime.

    Args:
        dt: Datetime to check (default: now)

    Returns:
        Market session: 'pre_market', 'regular', 'after_hours', 'closed'
    """
    import pytz

    dt = dt or datetime.now(timezone.utc)
    ny_tz = pytz.timezone("America/New_York")

    # Convert to NY time
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    ny_time = dt.astimezone(ny_tz)

    # Check if weekday
    if ny_time.weekday() >= 5:  # Weekend
        return "closed"

    # Define session times
    pre_market_start = ny_time.replace(hour=4, minute=0, second=0, microsecond=0)
    regular_start = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
    regular_end = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
    after_hours_end = ny_time.replace(hour=20, minute=0, second=0, microsecond=0)

    if pre_market_start <= ny_time < regular_start:
        return "pre_market"
    elif regular_start <= ny_time < regular_end:
        return "regular"
    elif regular_end <= ny_time < after_hours_end:
        return "after_hours"
    else:
        return "closed"


async def wait_for_market_open(max_wait_hours: int = 24) -> None:
    """
    Wait for market to open.

    Args:
        max_wait_hours: Maximum hours to wait
    """
    start_wait = datetime.now(timezone.utc)
    max_wait = timedelta(hours=max_wait_hours)

    while datetime.now(timezone.utc) - start_wait < max_wait:
        session = get_market_session()
        if session in ["regular", "pre_market"]:
            logger.info("Market is open, proceeding")
            return

        # Calculate time to next market open
        next_open = TimeUtils.get_next_market_open()
        wait_time = min(
            (next_open - datetime.now(timezone.utc)).total_seconds(), 3600
        )  # Max 1 hour wait

        logger.info(
            f"Market closed, waiting {wait_time / 60:.1f} minutes until next check"
        )
        await asyncio.sleep(wait_time)

    logger.warning(f"Stopped waiting for market open after {max_wait_hours} hours")


class ExecutionQueue:
    """Queue manager for execution tasks."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize execution queue."""
        self.redis = redis_client
        self._processors: dict[str, Any] = {}

    async def enqueue_signal(self, signal: Dict[str, Any], priority: int = 0) -> None:
        """Enqueue a signal for processing."""
        try:
            queue_item = {
                "signal": signal,
                "priority": priority,
                "enqueued_at": datetime.now(timezone.utc).isoformat(),
                "attempts": 0,
            }

            # Use priority queue (sorted set)
            await self.redis.zadd(
                "execution_queue", {json.dumps(queue_item, default=str): priority}
            )

            logger.info(
                f"Signal queued for {signal.get('symbol', 'unknown')} with priority {priority}"
            )

        except Exception as e:
            logger.error(f"Failed to enqueue signal: {e}")

    async def dequeue_signal(self, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Dequeue next signal for processing."""
        try:
            # Get highest priority item
            items = await self.redis.zrevrange("execution_queue", 0, 0, withscores=True)

            if not items:
                return None

            item_data, priority = items[0]
            queue_item = json.loads(item_data)

            # Remove from queue
            await self.redis.zrem("execution_queue", item_data)

            logger.debug(
                f"Dequeued signal for {queue_item['signal'].get('symbol', 'unknown')}"
            )
            return queue_item

        except Exception as e:
            logger.error(f"Failed to dequeue signal: {e}")
            return None

    async def get_queue_size(self) -> int:
        """Get current queue size."""
        try:
            return await self.redis.zcard("execution_queue")
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0


# Utility functions for common operations
async def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
    """Count business days between two dates."""
    current = start_date.date()
    end = end_date.date()
    business_days = 0

    while current <= end:
        if current.weekday() < 5:  # Monday-Friday
            business_days += 1
        current += timedelta(days=1)

    return business_days


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


async def batch_process(
    items: List[Any],
    processor: Callable,
    batch_size: int = 10,
    delay_between_batches: float = 1.0,
) -> List[Any]:
    """Process items in batches with delay."""
    results = []
    batches = chunk_list(items, batch_size)

    for i, batch in enumerate(batches):
        try:
            batch_results = await asyncio.gather(*[processor(item) for item in batch])
            results.extend(batch_results)

            # Delay between batches (except last batch)
            if i < len(batches) - 1 and delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)

        except Exception as e:
            logger.error(f"Batch processing error in batch {i}: {e}")
            # Continue with next batch

    return results


def create_correlation_matrix(
    symbols: List[str], correlations: Dict[Tuple[str, str], float]
) -> Dict[str, Dict[str, float]]:
    """Create correlation matrix from pairwise correlations."""
    matrix: dict[str, Any] = {}

    for symbol1 in symbols:
        matrix[symbol1] = {}
        for symbol2 in symbols:
            if symbol1 == symbol2:
                matrix[symbol1][symbol2] = 1.0
            else:
                # Try both directions
                key1 = (symbol1, symbol2)
                key2 = (symbol2, symbol1)
                correlation = correlations.get(key1, correlations.get(key2, 0.0))
                matrix[symbol1][symbol2] = correlation

    return matrix


# Export key utilities for easy import
__all__ = [
    "ExecutionError",
    "AuditLogger",
    "RetryManager",
    "DecimalUtils",
    "MarketDataUtils",
    "PerformanceUtils",
    "ValidationUtils",
    "TimeUtils",
    "NotificationUtils",
    "CacheManager",
    "ErrorHandler",
    "HealthChecker",
    "ConfigValidator",
    "MetricsCollector",
    "ExecutionQueue",
    "rate_limit",
    "timing_decorator",
    "safe_json_dumps",
    "safe_json_loads",
    "format_decimal",
    "truncate_string",
    "ensure_database_schema",
    "create_audit_schema",
    "calculate_position_sizing",
    "is_tradeable_symbol",
    "calculate_slippage",
    "get_market_session",
    "wait_for_market_open",
    "get_business_days_between",
    "chunk_list",
    "batch_process",
    "create_correlation_matrix",
]
