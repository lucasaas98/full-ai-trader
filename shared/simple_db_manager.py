"""
Simple Database Manager using asyncpg

This module provides a lightweight database manager using only asyncpg
for services that need database access without SQLAlchemy dependencies.
"""

import asyncio
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import asyncpg

from .config import get_config

logger = logging.getLogger(__name__)


class SimpleDatabaseManager:
    """Simple database manager using asyncpg for lightweight database operations."""

    def __init__(self, config=None):
        """Initialize database manager."""
        self.config = config or get_config()
        self.pool = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection pool."""
        if self._initialized:
            return

        try:
            # Try environment variables first (more reliable in containers)
            host = os.getenv("DB_HOST", "localhost")
            port = int(os.getenv("DB_PORT", "5432"))
            database = os.getenv("DB_NAME", "trading_system")
            user = os.getenv("DB_USER", "trader")
            password = os.getenv("DB_PASSWORD", "")

            # Fall back to config if env vars not available
            if host == "localhost" and hasattr(self.config, "database"):
                db_config = self.config.database
                host = getattr(db_config, "host", host)
                port = getattr(db_config, "port", port)
                database = getattr(db_config, "name", database)
                user = getattr(db_config, "user", user)
                password = getattr(db_config, "password", password)

            logger.info(f"Connecting to database: {user}@{host}:{port}/{database}")

            # Create connection pool using individual parameters
            self.pool = await asyncpg.create_pool(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                min_size=1,
                max_size=5,
                command_timeout=60,
            )

            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            logger.info("Database connection pool established successfully")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if not self.pool:
                return False

            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def get_portfolio_snapshots(
        self, start_date: datetime = None, end_date: datetime = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """Get portfolio snapshots between dates or latest snapshots."""
        try:
            if not self.pool:
                return []

            async with self.pool.acquire() as conn:
                if start_date and end_date:
                    query = """
                        SELECT
                            id, account_id, timestamp, cash, buying_power, total_equity,
                            day_trades_count, pattern_day_trader, data
                        FROM portfolio_snapshots
                        WHERE timestamp >= $1 AND timestamp <= $2
                        ORDER BY timestamp DESC
                    """
                    if limit:
                        query += f" LIMIT {limit}"
                    rows = await conn.fetch(query, start_date, end_date)
                else:
                    query = """
                        SELECT
                            id, account_id, timestamp, cash, buying_power, total_equity,
                            day_trades_count, pattern_day_trader, data
                        FROM portfolio_snapshots
                        ORDER BY timestamp DESC
                    """
                    if limit:
                        query += f" LIMIT {limit}"
                    rows = await conn.fetch(query)

                snapshots = []
                for row in rows:
                    snapshot = {
                        "id": row["id"],
                        "account_id": row["account_id"],
                        "timestamp": row["timestamp"],
                        "cash": float(row["cash"]) if row["cash"] else 0,
                        "buying_power": (
                            float(row["buying_power"]) if row["buying_power"] else 0
                        ),
                        "total_equity": (
                            float(row["total_equity"]) if row["total_equity"] else 0
                        ),
                        "day_trades_count": row["day_trades_count"] or 0,
                        "pattern_day_trader": row["pattern_day_trader"] or False,
                        "data": row["data"] or {},
                    }
                    snapshots.append(snapshot)

                return snapshots

        except Exception as e:
            logger.error(f"Failed to get portfolio snapshots: {e}")
            return []

    async def get_latest_portfolio_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest portfolio metrics."""
        try:
            if not self.pool:
                return None

            async with self.pool.acquire() as conn:
                query = """
                    SELECT
                        id, timestamp, total_exposure, cash_percentage, position_count,
                        concentration_risk, portfolio_beta, portfolio_correlation,
                        value_at_risk_1d, value_at_risk_5d, expected_shortfall_1d,
                        expected_shortfall_5d, maximum_drawdown, current_drawdown,
                        expected_shortfall, sharpe_ratio, max_drawdown, volatility, data
                    FROM portfolio_metrics
                    ORDER BY timestamp DESC
                    LIMIT 1
                """

                row = await conn.fetchrow(query)

                if not row:
                    return None

                return {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "total_exposure": (
                        float(row["total_exposure"]) if row["total_exposure"] else 0
                    ),
                    "cash_percentage": (
                        float(row["cash_percentage"]) if row["cash_percentage"] else 0
                    ),
                    "position_count": row["position_count"] or 0,
                    "concentration_risk": (
                        float(row["concentration_risk"])
                        if row["concentration_risk"]
                        else 0
                    ),
                    "portfolio_beta": (
                        float(row["portfolio_beta"]) if row["portfolio_beta"] else 0
                    ),
                    "portfolio_correlation": (
                        float(row["portfolio_correlation"])
                        if row["portfolio_correlation"]
                        else 0
                    ),
                    "value_at_risk_1d": (
                        float(row["value_at_risk_1d"]) if row["value_at_risk_1d"] else 0
                    ),
                    "value_at_risk_5d": (
                        float(row["value_at_risk_5d"]) if row["value_at_risk_5d"] else 0
                    ),
                    "expected_shortfall_1d": (
                        float(row["expected_shortfall_1d"])
                        if row["expected_shortfall_1d"]
                        else 0
                    ),
                    "expected_shortfall_5d": (
                        float(row["expected_shortfall_5d"])
                        if row["expected_shortfall_5d"]
                        else 0
                    ),
                    "maximum_drawdown": (
                        float(row["maximum_drawdown"]) if row["maximum_drawdown"] else 0
                    ),
                    "current_drawdown": (
                        float(row["current_drawdown"]) if row["current_drawdown"] else 0
                    ),
                    "expected_shortfall": (
                        float(row["expected_shortfall"])
                        if row["expected_shortfall"]
                        else 0
                    ),
                    "sharpe_ratio": (
                        float(row["sharpe_ratio"]) if row["sharpe_ratio"] else 0
                    ),
                    "max_drawdown": (
                        float(row["max_drawdown"]) if row["max_drawdown"] else 0
                    ),
                    "volatility": float(row["volatility"]) if row["volatility"] else 0,
                    "data": row["data"] or {},
                }

        except Exception as e:
            logger.error(f"Failed to get latest portfolio metrics: {e}")
            return None

    async def get_risk_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get risk statistics for the specified number of days."""
        try:
            if not self.pool:
                return {}

            start_date = datetime.now(timezone.utc) - timedelta(days=days)

            async with self.pool.acquire() as conn:
                # Get trade statistics
                trade_query = """
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                        AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                        SUM(pnl) as total_pnl,
                        MAX(pnl) as max_win,
                        MIN(pnl) as min_loss,
                        SUM(commission) as total_commission,
                        SUM(fees) as total_fees
                    FROM trades
                    WHERE timestamp >= $1 AND pnl IS NOT NULL
                """

                trade_row = await conn.fetchrow(trade_query, start_date)

                # Calculate metrics
                total_trades = trade_row["total_trades"] or 0
                winning_trades = trade_row["winning_trades"] or 0
                losing_trades = trade_row["losing_trades"] or 0

                stats = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": (
                        (winning_trades / max(total_trades, 1)) if total_trades else 0
                    ),
                    "avg_win": (
                        float(trade_row["avg_win"]) if trade_row["avg_win"] else 0
                    ),
                    "avg_loss": (
                        float(trade_row["avg_loss"]) if trade_row["avg_loss"] else 0
                    ),
                    "total_pnl": (
                        float(trade_row["total_pnl"]) if trade_row["total_pnl"] else 0
                    ),
                    "max_win": (
                        float(trade_row["max_win"]) if trade_row["max_win"] else 0
                    ),
                    "min_loss": (
                        float(trade_row["min_loss"]) if trade_row["min_loss"] else 0
                    ),
                    "total_commission": (
                        float(trade_row["total_commission"])
                        if trade_row["total_commission"]
                        else 0
                    ),
                    "total_fees": (
                        float(trade_row["total_fees"]) if trade_row["total_fees"] else 0
                    ),
                    "period_days": days,
                }

                # Calculate profit factor
                if trade_row["avg_loss"] and trade_row["avg_loss"] < 0:
                    stats["profit_factor"] = abs(trade_row["avg_win"] or 0) / abs(
                        trade_row["avg_loss"]
                    )
                else:
                    stats["profit_factor"] = None

                return stats

        except Exception as e:
            logger.error(f"Failed to get risk statistics: {e}")
            return {}

    async def get_daily_trades(self, target_date: date) -> List[Dict[str, Any]]:
        """Get trades for a specific date."""
        try:
            if not self.pool:
                return []

            start_datetime = datetime.combine(target_date, datetime.min.time())
            end_datetime = datetime.combine(target_date, datetime.max.time())

            async with self.pool.acquire() as conn:
                query = """
                    SELECT
                        id, symbol, side, quantity, price, commission,
                        timestamp, order_id, strategy_name, pnl, fees, data
                    FROM trades
                    WHERE timestamp >= $1 AND timestamp <= $2
                    ORDER BY timestamp
                """

                rows = await conn.fetch(query, start_datetime, end_datetime)

                trades = []
                for row in rows:
                    trade = {
                        "id": str(row["id"]),
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "quantity": int(row["quantity"]),
                        "price": float(row["price"]),
                        "commission": (
                            float(row["commission"]) if row["commission"] else 0
                        ),
                        "timestamp": row["timestamp"],
                        "order_id": str(row["order_id"]),
                        "strategy_name": row["strategy_name"],
                        "pnl": float(row["pnl"]) if row["pnl"] else None,
                        "fees": float(row["fees"]) if row["fees"] else None,
                        "data": row["data"] or {},
                    }
                    trades.append(trade)

                return trades

        except Exception as e:
            logger.error(f"Failed to get daily trades: {e}")
            return []

    async def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            if not self.pool:
                return []

            async with self.pool.acquire() as conn:
                query = """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """

                rows = await conn.fetch(query)
                return [row["table_name"] for row in rows]

        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []

    async def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a specific table."""
        try:
            if not self.pool:
                return 0

            async with self.pool.acquire() as conn:
                # Use parameterized query for table name (though asyncpg doesn't support table name parameters)
                # We'll use string formatting but validate the table name first
                if not table_name.replace("_", "").isalnum():
                    logger.warning(f"Invalid table name: {table_name}")
                    return 0

                query = f"SELECT COUNT(*) FROM {table_name}"
                result = await conn.fetchval(query)
                return result or 0

        except Exception as e:
            logger.error(f"Failed to get row count for {table_name}: {e}")
            return 0

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
            self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def get_risk_events(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        days: int = 1,
        severity: str = None,
    ) -> List[Dict[str, Any]]:
        """Get risk events for the specified date range or number of days."""
        try:
            if not self.pool:
                return []

            if not start_date:
                start_date = datetime.now(timezone.utc) - timedelta(days=days)
            if not end_date:
                end_date = datetime.now(timezone.utc)

            async with self.pool.acquire() as conn:
                if severity:
                    query = """
                        SELECT id, event_type, severity, symbol, description,
                               timestamp, resolved_at, action_taken, metadata
                        FROM risk_events
                        WHERE timestamp >= $1 AND timestamp <= $2 AND severity = $3
                        ORDER BY timestamp DESC
                    """
                    rows = await conn.fetch(query, start_date, end_date, severity)
                else:
                    query = """
                        SELECT id, event_type, severity, symbol, description,
                               timestamp, resolved_at, action_taken, metadata
                        FROM risk_events
                        WHERE timestamp >= $1 AND timestamp <= $2
                        ORDER BY timestamp DESC
                    """
                    rows = await conn.fetch(query, start_date, end_date)

                events = []
                for row in rows:
                    event = {
                        "id": str(row["id"]) if row["id"] else None,
                        "event_type": row["event_type"],
                        "severity": row["severity"],
                        "symbol": row["symbol"],
                        "description": row["description"],
                        "timestamp": row["timestamp"],
                        "resolved_at": row["resolved_at"],
                        "action_taken": row["action_taken"],
                        "metadata": row["metadata"] or {},
                    }
                    events.append(event)

                return events

        except Exception as e:
            logger.error(f"Failed to get risk events: {e}")
            return []

    async def get_active_positions_count(self) -> int:
        """Get count of active positions."""
        try:
            if not self.pool:
                return 0

            async with self.pool.acquire() as conn:
                query = """
                    SELECT COUNT(DISTINCT symbol) as position_count
                    FROM positions
                    WHERE quantity != 0
                    AND last_updated >= NOW() - INTERVAL '1 day'
                """

                result = await conn.fetchval(query)
                return result or 0

        except Exception as e:
            logger.error(f"Failed to get active positions count: {e}")
            return 0
