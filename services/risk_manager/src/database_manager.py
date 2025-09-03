"""
Database Manager for Risk Management

This module handles all database operations for the risk management service including:
- Risk events storage and retrieval
- Portfolio snapshots management
- Risk metrics persistence
- Performance tracking
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from shared.config import get_config
from shared.models import (
    DailyRiskReport,
    PortfolioMetrics,
    PortfolioState,
    PositionRisk,
    RiskAlert,
    RiskEvent,
)

logger = logging.getLogger(__name__)


class RiskDatabaseManager:
    """Manages database operations for risk management."""

    def __init__(self):
        """Initialize database manager."""
        self.config = get_config()
        self.db_config = self.config.database
        self.engine = None
        self.session_factory = None

    async def initialize(self):
        """Initialize database connection and create tables."""
        try:
            # Create async engine
            db_url = getattr(self.db_config, "url", "postgresql://localhost/trading")
            if not db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

            self.engine = create_async_engine(
                db_url,
                pool_size=getattr(self.db_config, "pool_size", 10),
                max_overflow=getattr(self.db_config, "max_overflow", 20),
                echo=getattr(self.config, "debug", False),
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine, class_=AsyncSession, expire_on_commit=False
            )

            # Create additional tables if needed
            await self._create_risk_tables()

            logger.info("Risk database manager initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database manager: {e}")
            raise

    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")

    async def _create_risk_tables(self):
        """Create additional risk management tables."""
        # Split SQL into individual statements for asyncpg compatibility
        sql_statements = [
            'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"',
            "CREATE SCHEMA IF NOT EXISTS risk",
            """CREATE TABLE IF NOT EXISTS risk.risk_events (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                event_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                symbol VARCHAR(20),
                description TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE,
                action_taken VARCHAR(200),
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS risk.portfolio_snapshots (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                account_id VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                cash DECIMAL(15,8) NOT NULL,
                buying_power DECIMAL(15,8) NOT NULL,
                total_equity DECIMAL(15,8) NOT NULL,
                total_market_value DECIMAL(15,8) NOT NULL,
                total_unrealized_pnl DECIMAL(15,8) NOT NULL,
                day_trades_count INTEGER DEFAULT 0,
                pattern_day_trader BOOLEAN DEFAULT FALSE,
                positions JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS risk.portfolio_metrics (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                total_exposure DECIMAL(15,8) NOT NULL,
                cash_percentage DECIMAL(8,6) NOT NULL,
                position_count INTEGER NOT NULL,
                concentration_risk DECIMAL(8,6) NOT NULL,
                portfolio_beta DECIMAL(8,6) NOT NULL,
                portfolio_correlation DECIMAL(8,6) NOT NULL,
                value_at_risk_1d DECIMAL(15,8) NOT NULL,
                value_at_risk_5d DECIMAL(15,8) NOT NULL,
                expected_shortfall DECIMAL(15,8) NOT NULL,
                sharpe_ratio DECIMAL(8,6) NOT NULL,
                max_drawdown DECIMAL(8,6) NOT NULL,
                current_drawdown DECIMAL(8,6) NOT NULL,
                volatility DECIMAL(8,6) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS risk.position_risks (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                symbol VARCHAR(20) NOT NULL,
                position_size DECIMAL(15,8) NOT NULL,
                portfolio_percentage DECIMAL(8,6) NOT NULL,
                volatility DECIMAL(8,6) NOT NULL,
                beta DECIMAL(8,6) NOT NULL,
                var_1d DECIMAL(15,8) NOT NULL,
                expected_return DECIMAL(8,6) NOT NULL,
                sharpe_ratio DECIMAL(8,6) NOT NULL,
                correlation_with_portfolio DECIMAL(8,6) NOT NULL,
                sector VARCHAR(50),
                risk_score DECIMAL(4,2) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS risk.risk_alerts (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                alert_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                symbol VARCHAR(20),
                title VARCHAR(200) NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_at TIMESTAMP WITH TIME ZONE,
                acknowledged_by VARCHAR(100),
                action_required BOOLEAN DEFAULT FALSE,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS risk.daily_reports (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                report_date DATE NOT NULL UNIQUE,
                portfolio_value DECIMAL(15,8) NOT NULL,
                daily_pnl DECIMAL(15,8) NOT NULL,
                daily_return DECIMAL(8,6) NOT NULL,
                max_drawdown DECIMAL(8,6) NOT NULL,
                current_drawdown DECIMAL(8,6) NOT NULL,
                volatility DECIMAL(8,6) NOT NULL,
                sharpe_ratio DECIMAL(8,6) NOT NULL,
                var_1d DECIMAL(15,8) NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                risk_events_count INTEGER NOT NULL,
                compliance_violations JSONB,
                report_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            """CREATE TABLE IF NOT EXISTS risk.trailing_stops (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                symbol VARCHAR(20) NOT NULL UNIQUE,
                enabled BOOLEAN DEFAULT TRUE,
                trail_percentage DECIMAL(8,6) NOT NULL,
                current_stop_price DECIMAL(15,8) NOT NULL,
                highest_price DECIMAL(15,8) NOT NULL,
                entry_price DECIMAL(15,8) NOT NULL,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )""",
            "CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk.risk_events(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_risk_events_type_severity ON risk.risk_events(event_type, severity)",
            "CREATE INDEX IF NOT EXISTS idx_risk_events_symbol ON risk.risk_events(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON risk.portfolio_snapshots(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_account ON risk.portfolio_snapshots(account_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_metrics_timestamp ON risk.portfolio_metrics(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_position_risks_timestamp ON risk.position_risks(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_position_risks_symbol ON risk.position_risks(symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_risk_alerts_timestamp ON risk.risk_alerts(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_risk_alerts_severity ON risk.risk_alerts(severity, acknowledged)",
            "CREATE INDEX IF NOT EXISTS idx_daily_reports_date ON risk.daily_reports(report_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_trailing_stops_symbol ON risk.trailing_stops(symbol)",
        ]

        try:
            if self.engine:
                async with self.engine.begin() as conn:
                    for sql in sql_statements:
                        await conn.execute(text(sql))
                logger.info("Risk management tables created/verified successfully")
            else:
                logger.error("Engine not initialized")
                raise RuntimeError("Database engine not initialized")
        except Exception as e:
            logger.error(f"Error creating risk tables: {e}")
            raise

    async def store_risk_event(self, event: RiskEvent) -> bool:
        """Store a risk event in the database."""
        try:
            insert_sql = """
            INSERT INTO risk_events (
                id, event_type, severity, symbol, description,
                timestamp, resolved_at, action_taken, metadata
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8, :p9
            )
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "p1": str(event.id or uuid4()),
                        "p2": (
                            event.event_type.value
                            if hasattr(event.event_type, "value")
                            else str(event.event_type)
                        ),
                        "p3": (
                            event.severity.value
                            if hasattr(event.severity, "value")
                            else str(event.severity)
                        ),
                        "p4": event.symbol,
                        "p5": event.description,
                        "p6": event.timestamp,
                        "p7": event.resolved_at,
                        "p8": event.action_taken,
                        "p9": json.dumps(event.metadata) if event.metadata else None,
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing risk event: {e}")
            return False

    async def store_portfolio_snapshot(self, portfolio: PortfolioState) -> bool:
        """Store a portfolio snapshot in the database."""
        try:
            insert_sql = """
            INSERT INTO portfolio_snapshots (
                account_id, timestamp, cash, buying_power, total_equity,
                day_trades_count, pattern_day_trader, data
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8
            )
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            # Convert positions to JSON and calculate additional metrics
            positions_data = [
                {
                    "symbol": pos.symbol,
                    "quantity": float(pos.quantity),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pnl or 0),
                    "cost_basis": float(pos.cost_basis or 0),
                    "side": "long" if pos.quantity > 0 else "short",
                }
                for pos in portfolio.positions
            ]

            # Calculate total market value and unrealized P&L
            total_market_value = sum(
                float(pos.market_value) for pos in portfolio.positions
            )
            total_unrealized_pnl = sum(
                float(pos.unrealized_pnl or 0) for pos in portfolio.positions
            )

            # Store additional data in the JSON field
            portfolio_data = {
                "positions": positions_data,
                "total_market_value": total_market_value,
                "total_unrealized_pnl": total_unrealized_pnl,
                "position_count": len(portfolio.positions),
            }

            async with self.session_factory() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "p1": portfolio.account_id,
                        "p2": portfolio.timestamp,
                        "p3": float(portfolio.cash),
                        "p4": float(portfolio.buying_power),
                        "p5": float(portfolio.total_equity),
                        "p6": portfolio.day_trades_count,
                        "p7": portfolio.pattern_day_trader,
                        "p8": json.dumps(portfolio_data),
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing portfolio snapshot: {e}")
            return False

    async def store_portfolio_metrics(self, metrics: PortfolioMetrics) -> bool:
        """Store portfolio metrics in the database."""
        try:
            insert_sql = """
            INSERT INTO portfolio_metrics (
                timestamp, total_exposure, cash_percentage, position_count,
                concentration_risk, portfolio_beta, portfolio_correlation,
                value_at_risk_1d, value_at_risk_5d, expected_shortfall,
                sharpe_ratio, max_drawdown, current_drawdown, volatility
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8, :p9, :p10, :p11, :p12, :p13, :p14
            )
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "p1": metrics.timestamp,
                        "p2": float(metrics.total_exposure),
                        "p3": float(metrics.cash_percentage),
                        "p4": metrics.position_count,
                        "p5": float(metrics.concentration_risk),
                        "p6": float(metrics.portfolio_beta),
                        "p7": float(metrics.portfolio_correlation),
                        "p8": float(metrics.value_at_risk_1d),
                        "p9": float(metrics.value_at_risk_5d),
                        "p10": float(metrics.expected_shortfall),
                        "p11": float(metrics.sharpe_ratio),
                        "p12": float(metrics.max_drawdown),
                        "p13": float(metrics.current_drawdown),
                        "p14": float(metrics.volatility),
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing portfolio metrics: {e}")
            return False

    async def store_position_risk(self, position_risk: PositionRisk) -> bool:
        """Store position risk data in the database."""
        try:
            insert_sql = """
            INSERT INTO position_risks (
                symbol, position_size, portfolio_percentage,
                volatility, beta, var_1d, expected_return, sharpe_ratio,
                correlation_with_portfolio, sector, risk_score
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8, :p9, :p10, :p11
            )
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "p1": position_risk.symbol,
                        "p2": (
                            float(position_risk.position_size)
                            if position_risk.position_size is not None
                            else 0.0
                        ),
                        "p3": (
                            float(position_risk.portfolio_percentage)
                            if position_risk.portfolio_percentage is not None
                            else 0.0
                        ),
                        "p4": (
                            float(position_risk.volatility)
                            if position_risk.volatility is not None
                            else 0.0
                        ),
                        "p5": (
                            float(position_risk.beta)
                            if position_risk.beta is not None
                            else 0.0
                        ),
                        "p6": (
                            float(position_risk.var_1d)
                            if position_risk.var_1d is not None
                            else 0.0
                        ),
                        "p7": (
                            float(position_risk.expected_return)
                            if position_risk.expected_return is not None
                            else 0.0
                        ),
                        "p8": (
                            float(position_risk.sharpe_ratio)
                            if position_risk.sharpe_ratio is not None
                            else 0.0
                        ),
                        "p9": (
                            float(position_risk.correlation_with_portfolio)
                            if position_risk.correlation_with_portfolio is not None
                            else 0.0
                        ),
                        "p10": position_risk.sector,
                        "p11": (
                            float(position_risk.risk_score)
                            if position_risk.risk_score is not None
                            else 0.0
                        ),
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing position risk: {e}")
            return False

    async def store_risk_alert(self, alert: RiskAlert) -> bool:
        """Store a risk alert in the database."""
        try:
            insert_sql = """
            INSERT INTO risk_alerts (
                alert_type, severity, symbol, title, message, timestamp,
                acknowledged, action_required, metadata
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8, :p9
            )
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "p1": (
                            alert.alert_type.value
                            if hasattr(alert.alert_type, "value")
                            else str(alert.alert_type)
                        ),
                        "p2": (
                            alert.severity.value
                            if hasattr(alert.severity, "value")
                            else str(alert.severity)
                        ),
                        "p3": alert.symbol,
                        "p4": alert.title,
                        "p5": alert.message,
                        "p6": alert.timestamp,
                        "p7": alert.acknowledged,
                        "p8": alert.action_required,
                        "p9": None,  # acknowledged_at not in model
                        "p10": None,  # acknowledged_by not in model
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing risk alert: {e}")
            return False

    async def store_daily_report(self, report: DailyRiskReport) -> bool:
        """Store daily risk report in the database."""
        try:
            insert_sql = """
            INSERT INTO daily_risk_reports (
                date, portfolio_value, daily_pnl, daily_return,
                max_drawdown, current_drawdown, volatility, sharpe_ratio,
                var_1d, total_trades, winning_trades, risk_events_count,
                report_data
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8, :p9, :p10, :p11, :p12, :p13
            )
            ON CONFLICT (date) DO UPDATE SET
                portfolio_value = EXCLUDED.portfolio_value,
                daily_pnl = EXCLUDED.daily_pnl,
                daily_return = EXCLUDED.daily_return,
                max_drawdown = EXCLUDED.max_drawdown,
                current_drawdown = EXCLUDED.current_drawdown,
                volatility = EXCLUDED.volatility,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                var_1d = EXCLUDED.var_1d,
                total_trades = EXCLUDED.total_trades,
                winning_trades = EXCLUDED.winning_trades,
                risk_events_count = EXCLUDED.risk_events_count,
                report_data = EXCLUDED.report_data
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "p1": report.date,
                        "p2": float(report.portfolio_value),
                        "p3": float(report.daily_pnl),
                        "p4": float(report.daily_return),
                        "p5": float(report.max_drawdown),
                        "p6": float(report.current_drawdown),
                        "p7": float(report.volatility),
                        "p8": float(report.sharpe_ratio),
                        "p9": float(report.var_1d),
                        "p10": report.total_trades,
                        "p11": report.winning_trades,
                        "p12": len(
                            report.risk_events
                        ),  # Use length of risk_events list
                        "p13": json.dumps(
                            {
                                "risk_events": [e.dict() for e in report.risk_events],
                                "position_risks": [
                                    p.dict() for p in report.position_risks
                                ],
                                "compliance_violations": report.compliance_violations,
                            }
                        ),
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing daily report: {e}")
            return False

    async def get_risk_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve risk events from the database."""
        try:
            conditions = []
            params = []
            param_count = 0

            base_query = "SELECT * FROM risk.risk_events WHERE 1=1"

            if start_date:
                param_count += 1
                conditions.append(f" AND timestamp >= ${param_count}")
                params.append(start_date)

            if end_date:
                param_count += 1
                conditions.append(f" AND timestamp <= ${param_count}")
                params.append(end_date)

            if event_type:
                param_count += 1
                conditions.append(f" AND event_type = ${param_count}")
                params.append(event_type)

            if severity:
                param_count += 1
                conditions.append(f" AND severity = ${param_count}")
                params.append(severity)

            if symbol:
                param_count += 1
                conditions.append(f" AND symbol = ${param_count}")
                params.append(symbol)

            query = (
                base_query
                + "".join(conditions)
                + f" ORDER BY timestamp DESC LIMIT ${param_count + 1}"
            )
            params.append(limit)

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                result = await session.execute(text(query), params)
                rows = result.fetchall()

                events = []
                for row in rows:
                    events.append(
                        {
                            "id": str(row.id),
                            "event_type": row.event_type,
                            "severity": row.severity,
                            "symbol": row.symbol,
                            "description": row.description,
                            "timestamp": row.timestamp,
                            "resolved_at": row.resolved_at,
                            "action_taken": row.action_taken,
                            "metadata": (
                                json.loads(row.metadata) if row.metadata else None
                            ),
                            "created_at": row.created_at,
                        }
                    )

                return events

        except Exception as e:
            logger.error(f"Error retrieving risk events: {e}")
            return []

    async def get_portfolio_snapshots(
        self,
        account_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve portfolio snapshots from the database."""
        try:
            conditions = []
            params = []
            param_count = 0

            base_query = "SELECT * FROM risk.portfolio_snapshots WHERE 1=1"

            if account_id:
                param_count += 1
                conditions.append(f" AND account_id = ${param_count}")
                params.append(account_id)

            if start_date:
                param_count += 1
                conditions.append(f" AND timestamp >= ${param_count}")
                params.append(start_date)

            if end_date:
                param_count += 1
                conditions.append(f" AND timestamp <= ${param_count}")
                params.append(end_date)

            query = (
                base_query
                + "".join(conditions)
                + f" ORDER BY timestamp DESC LIMIT ${param_count + 1}"
            )
            params.append(limit)

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                result = await session.execute(text(query), params)
                rows = result.fetchall()

                snapshots = []
                for row in rows:
                    snapshots.append(
                        {
                            "id": str(row.id),
                            "account_id": row.account_id,
                            "timestamp": row.timestamp,
                            "cash": float(row.cash),
                            "buying_power": float(row.buying_power),
                            "total_equity": float(row.total_equity),
                            "total_market_value": float(row.total_market_value),
                            "total_unrealized_pnl": float(row.total_unrealized_pnl),
                            "day_trades_count": row.day_trades_count,
                            "pattern_day_trader": row.pattern_day_trader,
                            "positions": (
                                json.loads(row.positions) if row.positions else []
                            ),
                            "created_at": row.created_at,
                        }
                    )

                return snapshots

        except Exception as e:
            logger.error(f"Error retrieving portfolio snapshots: {e}")
            return []

    async def get_latest_portfolio_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest portfolio metrics."""
        try:
            query = """
            SELECT * FROM risk.portfolio_metrics
            ORDER BY timestamp DESC
            LIMIT 1
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                result = await session.execute(text(query))
                row = result.fetchone()

                if row:
                    return {
                        "id": str(row.id),
                        "timestamp": row.timestamp,
                        "total_exposure": float(row.total_exposure),
                        "cash_percentage": float(row.cash_percentage),
                        "position_count": row.position_count,
                        "concentration_risk": float(row.concentration_risk),
                        "portfolio_beta": float(row.portfolio_beta),
                        "portfolio_correlation": float(row.portfolio_correlation),
                        "value_at_risk_1d": float(row.value_at_risk_1d),
                        "value_at_risk_5d": float(row.value_at_risk_5d),
                        "expected_shortfall": float(row.expected_shortfall),
                        "sharpe_ratio": float(row.sharpe_ratio),
                        "max_drawdown": float(row.max_drawdown),
                        "current_drawdown": float(row.current_drawdown),
                        "volatility": float(row.volatility),
                        "created_at": row.created_at,
                    }

                return None

        except Exception as e:
            logger.error(f"Error retrieving latest portfolio metrics: {e}")
            return None

    async def get_unacknowledged_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get unacknowledged risk alerts."""
        try:
            query = """
            SELECT * FROM risk.risk_alerts
            WHERE acknowledged = FALSE
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 4
                    WHEN 'high' THEN 3
                    WHEN 'medium' THEN 2
                    ELSE 1
                END DESC,
                timestamp DESC
            LIMIT :p1
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                result = await session.execute(text(query), {"p1": limit})
                rows = result.fetchall()

                alerts = []
                for row in rows:
                    alerts.append(
                        {
                            "id": str(row.id),
                            "alert_type": row.alert_type,
                            "severity": row.severity,
                            "symbol": row.symbol,
                            "title": row.title,
                            "message": row.message,
                            "timestamp": row.timestamp,
                            "acknowledged": row.acknowledged,
                            "acknowledged_at": row.acknowledged_at,
                            "acknowledged_by": row.acknowledged_by,
                            "action_required": row.action_required,
                            "metadata": (
                                json.loads(row.metadata) if row.metadata else None
                            ),
                            "created_at": row.created_at,
                        }
                    )

                return alerts

        except Exception as e:
            logger.error(f"Error retrieving unacknowledged alerts: {e}")
            return []

    async def acknowledge_alert(self, alert_id: UUID, acknowledged_by: str) -> bool:
        """Acknowledge a risk alert."""
        try:
            update_sql = """
            UPDATE risk_alerts
            SET acknowledged = TRUE
            WHERE id = :p1
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(text(update_sql), {"p1": str(alert_id)})
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False

    async def get_daily_report(self, report_date: date) -> Optional[Dict[str, Any]]:
        """Get daily risk report for a specific date."""
        try:
            query = """
            SELECT * FROM daily_risk_reports
            WHERE date = :p1
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                result = await session.execute(text(query), {"p1": report_date})
                row = result.fetchone()

                if row:
                    return {
                        "id": str(row.id),
                        "report_date": row.report_date,
                        "portfolio_value": float(row.portfolio_value),
                        "daily_pnl": float(row.daily_pnl),
                        "daily_return": float(row.daily_return),
                        "max_drawdown": float(row.max_drawdown),
                        "current_drawdown": float(row.current_drawdown),
                        "volatility": float(row.volatility),
                        "sharpe_ratio": float(row.sharpe_ratio),
                        "var_1d": float(row.var_1d),
                        "total_trades": row.total_trades,
                        "winning_trades": row.winning_trades,
                        "risk_events_count": row.risk_events_count,
                        "compliance_violations": (
                            json.loads(row.compliance_violations)
                            if row.compliance_violations
                            else None
                        ),
                        "report_data": (
                            json.loads(row.report_data) if row.report_data else None
                        ),
                        "created_at": row.created_at,
                    }

                return None

        except Exception as e:
            logger.error(f"Error retrieving daily report: {e}")
            return None

    async def get_trailing_stops(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trailing stop configurations."""
        try:
            if symbol:
                query = "SELECT * FROM trailing_stops WHERE symbol = :p1"
                params = {"p1": symbol}
            else:
                query = "SELECT * FROM trailing_stops WHERE enabled = TRUE"
                params = {}

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                result = await session.execute(text(query), params)
                rows = result.fetchall()

                stops = []
                for row in rows:
                    stops.append(
                        {
                            "id": str(row.id),
                            "symbol": row.symbol,
                            "enabled": row.enabled,
                            "trail_percentage": float(row.trail_percentage),
                            "current_stop_price": float(row.current_stop_price),
                            "highest_price": float(row.highest_price),
                            "entry_price": float(row.entry_price),
                            "last_updated": row.last_updated,
                            "created_at": row.created_at,
                        }
                    )

                return stops

        except Exception as e:
            logger.error(f"Error retrieving trailing stops: {e}")
            return []

    async def update_trailing_stop(
        self, symbol: str, stop_price: Decimal, highest_price: Decimal
    ) -> bool:
        """Update trailing stop for a symbol."""
        try:
            update_sql = """
            UPDATE trailing_stops
            SET current_stop_price = :p1,
                highest_price = :p2,
                last_updated = NOW()
            WHERE symbol = :p3 AND enabled = TRUE
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(update_sql),
                    {"p1": float(stop_price), "p2": float(highest_price), "p3": symbol},
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return False

    async def store_trailing_stop(
        self,
        symbol: str,
        enabled: bool,
        trail_percentage: Decimal,
        current_stop_price: Decimal,
        highest_price: Decimal,
        entry_price: Decimal,
    ) -> bool:
        """Store or update trailing stop configuration."""
        try:
            upsert_sql = """
            INSERT INTO trailing_stops (
                symbol, enabled, trail_percentage, current_stop_price,
                highest_price, entry_price, last_updated
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5, :p6, NOW()
            )
            ON CONFLICT (symbol) DO UPDATE SET
                enabled = EXCLUDED.enabled,
                trail_percentage = EXCLUDED.trail_percentage,
                current_stop_price = EXCLUDED.current_stop_price,
                highest_price = EXCLUDED.highest_price,
                entry_price = EXCLUDED.entry_price,
                last_updated = NOW()
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(upsert_sql),
                    {
                        "p1": symbol,
                        "p2": enabled,
                        "p3": float(trail_percentage),
                        "p4": float(current_stop_price),
                        "p5": float(highest_price),
                        "p6": float(entry_price),
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing trailing stop: {e}")
            return False

    async def store_position_sizing_history(
        self,
        symbol: str,
        signal_timestamp: datetime,
        recommended_shares: int,
        recommended_value: Decimal,
        position_percentage: Decimal,
        confidence_score: Optional[Decimal],
        volatility_adjustment: Decimal,
        sizing_method: str,
        max_loss_amount: Decimal,
        risk_reward_ratio: Decimal,
        portfolio_value: Decimal,
    ) -> bool:
        """Store position sizing history."""
        try:
            insert_sql = """
            INSERT INTO position_sizing_history (
                symbol, signal_timestamp, recommended_shares, recommended_value,
                position_percentage, confidence_score, volatility_adjustment,
                sizing_method, max_loss_amount, risk_reward_ratio, portfolio_value
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8, :p9, :p10, :p11
            )
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "p1": symbol,
                        "p2": signal_timestamp,
                        "p3": recommended_shares,
                        "p4": float(recommended_value),
                        "p5": float(position_percentage),
                        "p6": float(confidence_score) if confidence_score else None,
                        "p7": float(volatility_adjustment),
                        "p8": sizing_method,
                        "p9": float(max_loss_amount),
                        "p10": float(risk_reward_ratio),
                        "p11": float(portfolio_value),
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing position sizing history: {e}")
            return False

    async def store_circuit_breaker_event(
        self,
        trigger_type: str,
        trigger_value: Optional[Decimal],
        threshold_value: Optional[Decimal],
        duration_minutes: int = 15,
        portfolio_impact: Optional[Dict] = None,
    ) -> bool:
        """Store circuit breaker activation event."""
        try:
            insert_sql = """
            INSERT INTO circuit_breaker_events (
                trigger_type, trigger_value, threshold_value, duration_minutes, portfolio_impact
            ) VALUES (
                :p1, :p2, :p3, :p4, :p5
            )
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(
                    text(insert_sql),
                    {
                        "p1": trigger_type,
                        "p2": float(trigger_value) if trigger_value else None,
                        "p3": float(threshold_value) if threshold_value else None,
                        "p4": duration_minutes,
                        "p5": (
                            json.dumps(portfolio_impact) if portfolio_impact else None
                        ),
                    },
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing circuit breaker event: {e}")
            return False

    async def get_active_circuit_breakers(self) -> List[Dict[str, Any]]:
        """Get currently active circuit breakers."""
        try:
            query = """
            SELECT * FROM risk.circuit_breaker_events
            WHERE deactivated_at IS NULL
            AND trigger_timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY trigger_timestamp DESC
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                result = await session.execute(text(query))
                rows = result.fetchall()

                breakers = []
                for row in rows:
                    breakers.append(
                        {
                            "id": str(row.id),
                            "trigger_timestamp": row.trigger_timestamp,
                            "trigger_type": row.trigger_type,
                            "trigger_value": (
                                float(row.trigger_value) if row.trigger_value else None
                            ),
                            "threshold_value": (
                                float(row.threshold_value)
                                if row.threshold_value
                                else None
                            ),
                            "duration_minutes": row.duration_minutes,
                            "portfolio_impact": (
                                json.loads(row.portfolio_impact)
                                if row.portfolio_impact
                                else None
                            ),
                            "created_at": row.created_at,
                        }
                    )

                return breakers

        except Exception as e:
            logger.error(f"Error retrieving active circuit breakers: {e}")
            return []

    async def deactivate_circuit_breaker(
        self, breaker_id: UUID, deactivated_by: str
    ) -> bool:
        """Deactivate a circuit breaker."""
        try:
            update_sql = """
            UPDATE circuit_breaker_events
            SET deactivated_at = NOW()
            WHERE id = :p1
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                await session.execute(text(update_sql), {"p1": breaker_id})
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error deactivating circuit breaker: {e}")
            return False

    async def get_risk_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get risk statistics for the specified period."""
        try:
            start_date = datetime.now(timezone.utc) - timedelta(days=days)

            # Risk events statistics
            events_query = """
            SELECT
                event_type,
                severity,
                COUNT(*) as count
            FROM risk_events
            WHERE timestamp >= :p1
            GROUP BY event_type, severity
            """

            # Portfolio metrics statistics
            metrics_query = """
            SELECT
                AVG(volatility) as avg_volatility,
                AVG(sharpe_ratio) as avg_sharpe_ratio,
                AVG(concentration_risk) as avg_concentration_risk,
                MAX(max_drawdown) as max_drawdown_period,
                AVG(portfolio_beta) as avg_beta
            FROM portfolio_metrics
            WHERE timestamp >= :p1
            """

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:
                # Get events statistics
                events_result = await session.execute(
                    text(events_query), {"p1": start_date}
                )
                events_rows = events_result.fetchall()

                # Get metrics statistics
                metrics_result = await session.execute(
                    text(metrics_query), {"p1": start_date}
                )
                metrics_row = metrics_result.fetchone()

                # Process events data
                events_by_type = {}
                events_by_severity = {}

                for row in events_rows:
                    events_by_type[row.event_type] = (
                        events_by_type.get(row.event_type, 0) + row.count
                    )
                    events_by_severity[row.severity] = (
                        events_by_severity.get(row.severity, 0) + row.count
                    )

                # Compile statistics
                statistics = {
                    "period_days": days,
                    "events_by_type": events_by_type,
                    "events_by_severity": events_by_severity,
                    "total_events": sum(events_by_type.values()),
                    "metrics": {
                        "avg_volatility": (
                            float(metrics_row.avg_volatility)
                            if metrics_row and metrics_row.avg_volatility
                            else 0.0
                        ),
                        "avg_sharpe_ratio": (
                            float(metrics_row.avg_sharpe_ratio)
                            if metrics_row and metrics_row.avg_sharpe_ratio
                            else 0.0
                        ),
                        "avg_concentration_risk": (
                            float(metrics_row.avg_concentration_risk)
                            if metrics_row and metrics_row.avg_concentration_risk
                            else 0.0
                        ),
                        "max_drawdown_period": (
                            float(metrics_row.max_drawdown_period)
                            if metrics_row and metrics_row.max_drawdown_period
                            else 0.0
                        ),
                        "avg_beta": (
                            float(metrics_row.avg_beta)
                            if metrics_row and metrics_row.avg_beta
                            else 1.0
                        ),
                    },
                }

                return statistics

        except Exception as e:
            logger.error(f"Error retrieving risk statistics: {e}")
            return {
                "period_days": days,
                "events_by_type": {},
                "events_by_severity": {},
                "total_events": 0,
                "metrics": {},
            }

    async def cleanup_old_data(self, retention_days: int = 365) -> int:
        """Clean up old risk management data."""
        try:
            _ = datetime.now(timezone.utc) - timedelta(days=retention_days)

            _ = [
                "DELETE FROM risk_events WHERE created_at < :p1",
                "DELETE FROM portfolio_snapshots WHERE created_at < :p1",
                "DELETE FROM portfolio_metrics WHERE created_at < :p1",
                "DELETE FROM position_risks WHERE created_at < :p1",
                "DELETE FROM risk_alerts WHERE created_at < :p1 AND acknowledged = TRUE",
            ]

            total_deleted = 0

            if not self.session_factory:
                raise RuntimeError("Database session factory not initialized")

            async with self.session_factory() as session:

                await session.commit()

            logger.info(f"Cleanup completed: {total_deleted} total records deleted")
            return total_deleted

        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
            return 0

    async def health_check(self) -> bool:
        """Perform database health check."""
        try:
            if not self.session_factory:
                return False

            async with self.session_factory() as session:
                result = await session.execute(text("SELECT 1"))
                return result.fetchone() is not None

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
