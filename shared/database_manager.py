"""
Shared Database Manager

This module provides a centralized database manager that can be used across
all services in the trading system. It handles connections to PostgreSQL
and provides methods for common database operations.
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from .config import get_config

logger = logging.getLogger(__name__)

Base = declarative_base()


class SharedDatabaseManager:
    """Shared database manager for all trading system services."""

    def __init__(self, config=None):
        """Initialize database manager."""
        self.config = config or get_config()
        self.engine = None
        self.session_factory = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection and create tables if needed."""
        if self._initialized:
            return

        try:
            # Get database URL from config
            db_url = getattr(self.config.database, "url", None)
            if not db_url:
                # Fallback to constructing URL from individual components
                host = getattr(self.config.database, "host", "localhost")
                port = getattr(self.config.database, "port", 5432)
                name = getattr(self.config.database, "name", "trading_system")
                user = getattr(self.config.database, "user", "trader")
                password = getattr(self.config.database, "password", "")
                db_url = f"postgresql://{user}:{password}@{host}:{port}/{name}"

            # Ensure we use asyncpg driver
            if not db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

            logger.info(f"Connecting to database: {db_url.split('@')[0]}@***")

            # Create async engine
            self.engine = create_async_engine(
                db_url,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine, class_=AsyncSession, expire_on_commit=False
            )

            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            logger.info("Database connection established successfully")
            self._initialized = True

            # Create tables if they don't exist
            await self._ensure_tables()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def _ensure_tables(self):
        """Ensure required tables exist in the database."""
        try:
            if self.engine is None:
                raise RuntimeError("Database not initialized. Call initialize() first.")
            async with self.engine.begin() as conn:
                # Create tables if they don't exist
                await conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id SERIAL PRIMARY KEY,
                        account_id VARCHAR(100) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        cash NUMERIC(15,2) NOT NULL,
                        buying_power NUMERIC(15,2) NOT NULL,
                        total_equity NUMERIC(15,2) NOT NULL,
                        day_trades_count INTEGER DEFAULT 0,
                        pattern_day_trader BOOLEAN DEFAULT FALSE,
                        data JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """
                    )
                )

                await conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS portfolio_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        total_exposure NUMERIC(15,2) DEFAULT 0,
                        cash_percentage NUMERIC(8,4) DEFAULT 0,
                        position_count INTEGER DEFAULT 0,
                        concentration_risk NUMERIC(8,4) DEFAULT 0,
                        portfolio_beta NUMERIC(8,4) DEFAULT 0,
                        portfolio_correlation NUMERIC(8,4) DEFAULT 0,
                        value_at_risk_1d NUMERIC(15,2) DEFAULT 0,
                        value_at_risk_5d NUMERIC(15,2) DEFAULT 0,
                        expected_shortfall_1d NUMERIC(15,2) DEFAULT 0,
                        expected_shortfall_5d NUMERIC(15,2) DEFAULT 0,
                        maximum_drawdown NUMERIC(15,2) DEFAULT 0,
                        current_drawdown NUMERIC(15,2) DEFAULT 0,
                        expected_shortfall NUMERIC(15,2) DEFAULT 0,
                        sharpe_ratio NUMERIC(8,4) DEFAULT 0,
                        max_drawdown NUMERIC(15,2) DEFAULT 0,
                        volatility NUMERIC(8,4) DEFAULT 0,
                        data JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """
                    )
                )

                await conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS trades (
                        id UUID PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        quantity INTEGER NOT NULL,
                        price NUMERIC(15,6) NOT NULL,
                        commission NUMERIC(10,2) DEFAULT 0,
                        timestamp TIMESTAMPTZ NOT NULL,
                        order_id UUID NOT NULL,
                        strategy_name VARCHAR(100) NOT NULL,
                        pnl NUMERIC(15,2),
                        fees NUMERIC(10,2),
                        data JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """
                    )
                )

                await conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS positions (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        quantity INTEGER NOT NULL,
                        entry_price NUMERIC(15,6) NOT NULL,
                        current_price NUMERIC(15,6) NOT NULL,
                        unrealized_pnl NUMERIC(15,2) NOT NULL,
                        market_value NUMERIC(15,2) NOT NULL,
                        cost_basis NUMERIC(15,2) NOT NULL,
                        last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        data JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(symbol, last_updated)
                    );
                """
                    )
                )

                await conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS risk_events (
                        id UUID PRIMARY KEY,
                        event_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        symbol VARCHAR(20),
                        description TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        resolved_at TIMESTAMPTZ,
                        action_taken TEXT,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """
                    )
                )

                # Create indexes for better performance
                await conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp
                    ON portfolio_snapshots(timestamp);
                """
                    )
                )

                await conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                    ON trades(timestamp);
                """
                    )
                )

                await conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol
                    ON trades(symbol);
                """
                    )
                )

            logger.info("Database tables ensured successfully")

        except Exception as e:
            logger.error(f"Failed to ensure database tables: {e}")
            # Don't raise here - we can still work with existing tables

    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if self.session_factory is None:
                return False
            async with self.session_factory() as session:
                result = await session.execute(text("SELECT 1 as test"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def get_portfolio_snapshots(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get portfolio snapshots between dates."""
        try:
            if self.session_factory is None:
                return []
            async with self.session_factory() as session:
                query = text(
                    """
                    SELECT
                        id, account_id, timestamp, cash, buying_power, total_equity,
                        day_trades_count, pattern_day_trader, data
                    FROM portfolio_snapshots
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                    ORDER BY timestamp
                """
                )

                result = await session.execute(
                    query, {"start_date": start_date, "end_date": end_date}
                )

                snapshots = []
                for row in result:
                    snapshot = {
                        "id": row.id,
                        "account_id": row.account_id,
                        "timestamp": row.timestamp,
                        "cash": float(row.cash) if row.cash else 0,
                        "buying_power": (
                            float(row.buying_power) if row.buying_power else 0
                        ),
                        "total_equity": (
                            float(row.total_equity) if row.total_equity else 0
                        ),
                        "day_trades_count": row.day_trades_count or 0,
                        "pattern_day_trader": row.pattern_day_trader or False,
                        "data": row.data or {},
                    }
                    snapshots.append(snapshot)

                return snapshots

        except Exception as e:
            logger.error(f"Failed to get portfolio snapshots: {e}")
            return []

    async def get_latest_portfolio_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest portfolio metrics."""
        try:
            if self.session_factory is None:
                return None
            async with self.session_factory() as session:
                query = text(
                    """
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
                )

                result = await session.execute(query)
                row = result.first()

                if not row:
                    return None

                return {
                    "id": row.id,
                    "timestamp": row.timestamp,
                    "total_exposure": (
                        float(row.total_exposure) if row.total_exposure else 0
                    ),
                    "cash_percentage": (
                        float(row.cash_percentage) if row.cash_percentage else 0
                    ),
                    "position_count": row.position_count or 0,
                    "concentration_risk": (
                        float(row.concentration_risk) if row.concentration_risk else 0
                    ),
                    "portfolio_beta": (
                        float(row.portfolio_beta) if row.portfolio_beta else 0
                    ),
                    "portfolio_correlation": (
                        float(row.portfolio_correlation)
                        if row.portfolio_correlation
                        else 0
                    ),
                    "value_at_risk_1d": (
                        float(row.value_at_risk_1d) if row.value_at_risk_1d else 0
                    ),
                    "value_at_risk_5d": (
                        float(row.value_at_risk_5d) if row.value_at_risk_5d else 0
                    ),
                    "expected_shortfall_1d": (
                        float(row.expected_shortfall_1d)
                        if row.expected_shortfall_1d
                        else 0
                    ),
                    "expected_shortfall_5d": (
                        float(row.expected_shortfall_5d)
                        if row.expected_shortfall_5d
                        else 0
                    ),
                    "maximum_drawdown": (
                        float(row.maximum_drawdown) if row.maximum_drawdown else 0
                    ),
                    "current_drawdown": (
                        float(row.current_drawdown) if row.current_drawdown else 0
                    ),
                    "expected_shortfall": (
                        float(row.expected_shortfall) if row.expected_shortfall else 0
                    ),
                    "sharpe_ratio": float(row.sharpe_ratio) if row.sharpe_ratio else 0,
                    "max_drawdown": float(row.max_drawdown) if row.max_drawdown else 0,
                    "volatility": float(row.volatility) if row.volatility else 0,
                    "data": row.data or {},
                }

        except Exception as e:
            logger.error(f"Failed to get latest portfolio metrics: {e}")
            return None

    async def get_risk_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get risk statistics for the specified number of days."""
        try:
            if self.session_factory is None:
                return {}
            start_date = datetime.now(timezone.utc) - timedelta(days=days)

            async with self.session_factory() as session:
                # Get trade statistics
                trade_query = text(
                    """
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                        AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                        SUM(pnl) as total_pnl,
                        MAX(pnl) as max_win,
                        MIN(pnl) as min_loss
                    FROM trades
                    WHERE timestamp >= :start_date AND pnl IS NOT NULL
                """
                )

                trade_result = await session.execute(
                    trade_query, {"start_date": start_date}
                )
                trade_row = trade_result.first()

                # Get risk events
                risk_query = text(
                    """
                    SELECT
                        event_type,
                        severity,
                        COUNT(*) as count
                    FROM risk_events
                    WHERE timestamp >= :start_date
                    GROUP BY event_type, severity
                """
                )

                risk_result = await session.execute(
                    risk_query, {"start_date": start_date}
                )
                risk_events = {}
                for row in risk_result:
                    key = f"{row.event_type}_{row.severity}"
                    risk_events[key] = row.count

                # Calculate metrics
                stats = {
                    "total_trades": trade_row.total_trades or 0,
                    "winning_trades": trade_row.winning_trades or 0,
                    "losing_trades": trade_row.losing_trades or 0,
                    "win_rate": (
                        (trade_row.winning_trades / max(trade_row.total_trades, 1))
                        if trade_row.total_trades
                        else 0
                    ),
                    "avg_win": float(trade_row.avg_win) if trade_row.avg_win else 0,
                    "avg_loss": float(trade_row.avg_loss) if trade_row.avg_loss else 0,
                    "total_pnl": (
                        float(trade_row.total_pnl) if trade_row.total_pnl else 0
                    ),
                    "max_win": float(trade_row.max_win) if trade_row.max_win else 0,
                    "min_loss": float(trade_row.min_loss) if trade_row.min_loss else 0,
                    "risk_events": risk_events,
                    "period_days": days,
                }

                # Calculate profit factor
                if trade_row.avg_loss and trade_row.avg_loss < 0:
                    stats["profit_factor"] = abs(trade_row.avg_win or 0) / abs(
                        trade_row.avg_loss
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
            if self.session_factory is None:
                return []
            start_datetime = datetime.combine(target_date, datetime.min.time())
            end_datetime = datetime.combine(target_date, datetime.max.time())

            async with self.session_factory() as session:
                query = text(
                    """
                    SELECT
                        id, symbol, side, quantity, price, commission,
                        timestamp, order_id, strategy_name, pnl, fees, data
                    FROM trades
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                    ORDER BY timestamp
                """
                )

                result = await session.execute(
                    query, {"start_date": start_datetime, "end_date": end_datetime}
                )

                trades = []
                for row in result:
                    trade = {
                        "id": str(row.id),
                        "symbol": row.symbol,
                        "side": row.side,
                        "quantity": int(row.quantity),
                        "price": float(row.price),
                        "commission": float(row.commission) if row.commission else 0,
                        "timestamp": row.timestamp,
                        "order_id": str(row.order_id),
                        "strategy_name": row.strategy_name,
                        "pnl": float(row.pnl) if row.pnl else None,
                        "fees": float(row.fees) if row.fees else None,
                        "data": row.data or {},
                    }
                    trades.append(trade)

                return trades

        except Exception as e:
            logger.error(f"Failed to get daily trades: {e}")
            return []

    async def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            if self.session_factory is None:
                return []
            if self.session_factory is None:
                return []
            async with self.session_factory() as session:
                query = text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """
                )

                result = await session.execute(query)
                return [row.table_name for row in result]

        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []

    async def insert_portfolio_snapshot(self, snapshot_data: Dict[str, Any]) -> bool:
        """Insert a new portfolio snapshot."""
        try:
            if self.session_factory is None:
                return False
            async with self.session_factory() as session:
                query = text(
                    """
                    INSERT INTO portfolio_snapshots
                    (account_id, timestamp, cash, buying_power, total_equity,
                     day_trades_count, pattern_day_trader, data)
                    VALUES
                    (:account_id, :timestamp, :cash, :buying_power, :total_equity,
                     :day_trades_count, :pattern_day_trader, :data)
                """
                )

                await session.execute(
                    query,
                    {
                        "account_id": snapshot_data.get("account_id", "default"),
                        "timestamp": snapshot_data.get(
                            "timestamp", datetime.now(timezone.utc)
                        ),
                        "cash": snapshot_data.get("cash", 0),
                        "buying_power": snapshot_data.get("buying_power", 0),
                        "total_equity": snapshot_data.get("total_equity", 0),
                        "day_trades_count": snapshot_data.get("day_trades_count", 0),
                        "pattern_day_trader": snapshot_data.get(
                            "pattern_day_trader", False
                        ),
                        "data": json.dumps(snapshot_data.get("data", {})),
                    },
                )

                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to insert portfolio snapshot: {e}")
            return False

    async def get_risk_events(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get risk events between dates."""
        try:
            if self.session_factory is None:
                return []
            async with self.session_factory() as session:
                query = text(
                    """
                    SELECT
                        id, event_type, severity, symbol, description,
                        timestamp, resolved_at, action_taken, metadata
                    FROM risk_events
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                    ORDER BY timestamp
                """
                )

                result = await session.execute(
                    query, {"start_date": start_date, "end_date": end_date}
                )

                events = []
                for row in result:
                    event = {
                        "id": str(row.id),
                        "event_type": row.event_type,
                        "severity": row.severity,
                        "symbol": row.symbol,
                        "description": row.description,
                        "timestamp": row.timestamp,
                        "resolved_at": row.resolved_at,
                        "action_taken": row.action_taken,
                        "metadata": row.metadata or {},
                    }
                    events.append(event)

                return events

        except Exception as e:
            logger.error(f"Failed to get risk events: {e}")
            return []

    async def close(self):
        """Close database connection."""
        if self.engine is not None:
            await self.engine.dispose()
            logger.info("Database connection closed")
        self._initialized = False
