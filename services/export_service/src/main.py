"""
Export Service for AI Trading System

This service provides comprehensive export functionality including:
- TradeNote compatible exports
- Performance reports
- Tax reporting data
- Audit trails
- Portfolio analysis
"""

import json
import logging
import os
import sys
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import aiofiles
import asyncpg
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Add parent directories to Python path for shared imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from shared.config import Config  # noqa: E402


# Simplified logging setup
def setup_logging(name):
    """Simple logging setup without complex dependencies"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


# Models imported as needed in functions

# Configure logging
logger = setup_logging(__name__)

# Security
security = HTTPBearer()


# Database connection pool
class DatabaseManager:
    def __init__(self):
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool"""
        config = Config()
        try:
            self.pool = await asyncpg.create_pool(
                host=config.database.host,
                port=config.database.port,
                database=config.database.database,
                user=config.database.username,
                password=config.database.password,
                min_size=1,
                max_size=5,
            )
            logger.info("Database pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            await self.initialize()
        if not self.pool:
            raise HTTPException(status_code=500, detail="Database pool not initialized")
        try:
            return await self.pool.acquire()
        except Exception as e:
            logger.error(f"Failed to acquire database connection: {e}")
            raise HTTPException(status_code=500, detail="Database connection failed")


# Global database manager
db_manager = DatabaseManager()


@dataclass
class ExportRequest:
    """Export request configuration"""

    format: str  # 'tradenote', 'csv', 'json', 'excel', 'pdf'
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols: Optional[List[str]] = None
    strategies: Optional[List[str]] = None
    include_closed_only: bool = True
    include_metadata: bool = True
    compress: bool = False


class ExportService:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.exports_dir = Path("/tmp/exports")
        self.exports_dir.mkdir(exist_ok=True)

    async def export_tradenote_format(self, request: ExportRequest) -> str:
        """Export trades in TradeNote compatible format"""
        logger.info(f"Starting TradeNote export: {request}")

        conn = await self.db.get_connection()
        try:
            # Build query based on request parameters
            query = """
                SELECT
                    t.id,
                    t.symbol,
                    t.side,
                    t.quantity,
                    t.price,
                    t.executed_at,
                    t.strategy_id,
                    t.status,
                    t.commission,
                    t.pnl,
                    p.name as position_name,
                    s.name as strategy_name
                FROM trades t
                LEFT JOIN positions p ON t.position_id = p.id
                LEFT JOIN strategies s ON t.strategy_id = s.id
                WHERE t.status = 'FILLED'
            """

            params: List[Any] = []
            param_count = 1

            if request.start_date:
                query += f" AND t.executed_at >= ${param_count}"
                params.append(request.start_date)
                param_count += 1

            if request.end_date:
                query += f" AND t.executed_at <= ${param_count}"
                params.append(request.end_date)
                param_count += 1

            if request.symbols:
                query += f" AND t.symbol = ANY(${param_count})"
                params.append(list(request.symbols))
                param_count += 1

            if request.strategies:
                query += f" AND s.name = ANY(${param_count})"
                params.append(list(request.strategies))
                param_count += 1

            query += " ORDER BY t.executed_at DESC"

            rows = await conn.fetch(query, *params)

            # Convert to TradeNote format
            tradenote_data = []
            for row in rows:
                tradenote_data.append(
                    {
                        "Date": row["executed_at"].strftime("%Y-%m-%d"),
                        "Time": row["executed_at"].strftime("%H:%M:%S"),
                        "Symbol": row["symbol"],
                        "Side": row["side"],
                        "Quantity": float(row["quantity"]),
                        "Price": float(row["price"]),
                        "Commission": float(row["commission"] or 0),
                        "PnL": float(row["pnl"] or 0),
                        "Strategy": row["strategy_name"] or "",
                        "Notes": "",
                    }
                )

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tradenote_export_{timestamp}.csv"
            filepath = self.exports_dir / filename

            # Write CSV file
            if tradenote_data:
                df = pd.DataFrame(tradenote_data)
                df.to_csv(filepath, index=False)
            else:
                # Create empty file with headers
                headers = [
                    "Date",
                    "Time",
                    "Symbol",
                    "Side",
                    "Quantity",
                    "Price",
                    "Commission",
                    "PnL",
                    "Strategy",
                    "Notes",
                ]
                pd.DataFrame(columns=headers).to_csv(filepath, index=False)

            logger.info(f"TradeNote export completed: {filepath}")
            return str(filepath)

        finally:
            if conn and self.db.pool:
                await self.db.pool.release(conn)

    async def export_performance_report(self, request: ExportRequest) -> str:
        """Export comprehensive performance report"""
        logger.info(f"Starting performance report export: {request}")

        conn = await self.db.get_connection()
        try:
            # Portfolio summary
            portfolio_query = """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_win,
                    MIN(pnl) as max_loss,
                    SUM(commission) as total_commission
                FROM trades
                WHERE status = 'FILLED'
            """

            params = []
            param_count = 1

            if request.start_date:
                portfolio_query += f" AND executed_at >= ${param_count}"
                params.append(request.start_date)
                param_count += 1

            if request.end_date:
                portfolio_query += f" AND executed_at <= ${param_count}"
                params.append(request.end_date)
                param_count += 1

            portfolio_row = await conn.fetchrow(portfolio_query, *params)

            # Calculate performance metrics
            total_trades = portfolio_row["total_trades"] or 0
            winning_trades = portfolio_row["winning_trades"] or 0
            losing_trades = portfolio_row["losing_trades"] or 0
            total_pnl = float(portfolio_row["total_pnl"] or 0)
            avg_pnl = float(portfolio_row["avg_pnl"] or 0)
            max_win = float(portfolio_row["max_win"] or 0)
            max_loss = float(portfolio_row["max_loss"] or 0)
            total_commission = float(portfolio_row["total_commission"] or 0)

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            profit_factor = abs(max_win / max_loss) if max_loss != 0 else 0

            # Strategy breakdown
            strategy_query = """
                SELECT
                    s.name as strategy_name,
                    COUNT(*) as trades,
                    SUM(t.pnl) as total_pnl,
                    AVG(t.pnl) as avg_pnl,
                    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN t.pnl < 0 THEN 1 ELSE 0 END) as losses
                FROM trades t
                LEFT JOIN strategies s ON t.strategy_id = s.id
                WHERE t.status = 'FILLED'
            """

            if request.start_date or request.end_date:
                strategy_params = []
                strategy_param_count = 1

                if request.start_date:
                    strategy_query += f" AND t.executed_at >= ${strategy_param_count}"
                    strategy_params.append(request.start_date)
                    strategy_param_count += 1

                if request.end_date:
                    strategy_query += f" AND t.executed_at <= ${strategy_param_count}"
                    strategy_params.append(request.end_date)
                    strategy_param_count += 1

                strategy_query += " GROUP BY s.name ORDER BY total_pnl DESC"
                strategy_rows = await conn.fetch(strategy_query, *strategy_params)
            else:
                strategy_query += " GROUP BY s.name ORDER BY total_pnl DESC"
                strategy_rows = await conn.fetch(strategy_query)

            strategy_breakdown = []
            for row in strategy_rows:
                strategy_trades = row["trades"] or 0
                strategy_win_rate = (
                    (row["wins"] / strategy_trades * 100) if strategy_trades > 0 else 0
                )

                strategy_breakdown.append(
                    {
                        "strategy": row["strategy_name"] or "Unknown",
                        "trades": strategy_trades,
                        "total_pnl": float(row["total_pnl"] or 0),
                        "avg_pnl": float(row["avg_pnl"] or 0),
                        "win_rate": round(strategy_win_rate, 2),
                        "wins": row["wins"] or 0,
                        "losses": row["losses"] or 0,
                    }
                )

            # Portfolio timeline (daily P&L)
            timeline_query = """
                SELECT
                    DATE(executed_at) as trade_date,
                    SUM(pnl) as daily_pnl,
                    COUNT(*) as daily_trades
                FROM trades
                WHERE status = 'FILLED'
            """

            if request.start_date or request.end_date:
                timeline_params = []
                timeline_param_count = 1

                if request.start_date:
                    timeline_query += f" AND executed_at >= ${timeline_param_count}"
                    timeline_params.append(request.start_date)
                    timeline_param_count += 1

                if request.end_date:
                    timeline_query += f" AND executed_at <= ${timeline_param_count}"
                    timeline_params.append(request.end_date)
                    timeline_param_count += 1

                timeline_query += " GROUP BY DATE(executed_at) ORDER BY trade_date"
                timeline_rows = await conn.fetch(timeline_query, *timeline_params)
            else:
                timeline_query += " GROUP BY DATE(executed_at) ORDER BY trade_date"
                timeline_rows = await conn.fetch(timeline_query)

            portfolio_timeline = []
            for row in timeline_rows:
                portfolio_timeline.append(
                    {
                        "date": row["trade_date"].strftime("%Y-%m-%d"),
                        "daily_pnl": float(row["daily_pnl"] or 0),
                        "daily_trades": row["daily_trades"] or 0,
                    }
                )

            # Compile report
            report = {
                "portfolio_summary": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": round(win_rate, 2),
                    "total_pnl": total_pnl,
                    "avg_pnl": avg_pnl,
                    "max_win": max_win,
                    "max_loss": max_loss,
                    "profit_factor": round(profit_factor, 2),
                    "total_commission": total_commission,
                    "net_pnl": total_pnl - total_commission,
                },
                "performance_metrics": {
                    "sharpe_ratio": 0,  # Would need daily returns to calculate
                    "max_drawdown": 0,  # Would need equity curve to calculate
                    "avg_win": max_win / winning_trades if winning_trades > 0 else 0,
                    "avg_loss": max_loss / losing_trades if losing_trades > 0 else 0,
                    "largest_win": max_win,
                    "largest_loss": max_loss,
                },
                "strategy_breakdown": strategy_breakdown,
                "portfolio_timeline": portfolio_timeline,
                "generated_at": datetime.now().isoformat(),
            }

            # Generate filename and save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
            filepath = self.exports_dir / filename

            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(report, indent=2))

            logger.info(f"Performance report export completed: {filepath}")
            return str(filepath)

        finally:
            if conn and self.db.pool:
                await self.db.pool.release(conn)

    async def export_tax_report(self, request: ExportRequest) -> str:
        """Export tax reporting data"""
        logger.info(f"Starting tax report export: {request}")

        conn = await self.db.get_connection()
        try:
            # Get all closed positions for tax year
            query = """
                SELECT
                    t.symbol,
                    t.side,
                    t.quantity,
                    t.price,
                    t.executed_at,
                    t.pnl,
                    t.commission,
                    p.opened_at,
                    p.closed_at,
                    p.opening_price,
                    p.closing_price
                FROM trades t
                LEFT JOIN positions p ON t.position_id = p.id
                WHERE t.status = 'FILLED'
                AND p.status = 'CLOSED'
            """

            params = []
            param_count = 1

            if request.start_date:
                query += f" AND p.closed_at >= ${param_count}"
                params.append(request.start_date)
                param_count += 1

            if request.end_date:
                query += f" AND p.closed_at <= ${param_count}"
                params.append(request.end_date)
                param_count += 1

            query += " ORDER BY p.closed_at DESC"

            rows = await conn.fetch(query, *params)

            # Calculate tax lots
            tax_lots = []
            for row in rows:
                holding_period = (row["closed_at"] - row["opened_at"]).days

                tax_lots.append(
                    {
                        "symbol": row["symbol"],
                        "description": f"{row['symbol']} Stock",
                        "acquisition_date": row["opened_at"].strftime("%Y-%m-%d"),
                        "sale_date": row["closed_at"].strftime("%Y-%m-%d"),
                        "quantity": float(row["quantity"]),
                        "cost_basis": float(row["opening_price"])
                        * float(row["quantity"]),
                        "sale_proceeds": float(row["closing_price"])
                        * float(row["quantity"]),
                        "gain_loss": float(row["pnl"]),
                        "holding_period_days": holding_period,
                        "term": "Long" if holding_period > 365 else "Short",
                        "commission": float(row["commission"] or 0),
                    }
                )

            # Calculate summary
            total_short_term_gain = sum(
                lot["gain_loss"] for lot in tax_lots if lot["term"] == "Short"
            )
            total_long_term_gain = sum(
                lot["gain_loss"] for lot in tax_lots if lot["term"] == "Long"
            )
            total_commission = sum(lot["commission"] for lot in tax_lots)

            tax_report = {
                "tax_year": (
                    request.start_date.year
                    if request.start_date
                    else datetime.now().year
                ),
                "summary": {
                    "total_transactions": len(tax_lots),
                    "short_term_gain_loss": total_short_term_gain,
                    "long_term_gain_loss": total_long_term_gain,
                    "total_gain_loss": total_short_term_gain + total_long_term_gain,
                    "total_commission": total_commission,
                },
                "tax_lots": tax_lots,
                "generated_at": datetime.now().isoformat(),
            }

            # Generate filename and save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tax_report_{timestamp}.json"
            filepath = self.exports_dir / filename

            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(tax_report, indent=2))

            logger.info(f"Tax report export completed: {filepath}")
            return str(filepath)

        finally:
            if conn and self.db.pool:
                await self.db.pool.release(conn)

    async def export_audit_trail(self, request: ExportRequest) -> str:
        """Export comprehensive audit trail"""
        logger.info(f"Starting audit trail export: {request}")

        conn = await self.db.get_connection()
        try:
            # Get all trading activities
            query = """
                SELECT
                    'TRADE' as event_type,
                    t.id::text as event_id,
                    t.executed_at as timestamp,
                    t.symbol,
                    t.side,
                    t.quantity,
                    t.price,
                    t.status,
                    t.commission,
                    t.pnl,
                    s.name as strategy_name,
                    t.metadata
                FROM trades t
                LEFT JOIN strategies s ON t.strategy_id = s.id

                UNION ALL

                SELECT
                    'POSITION' as event_type,
                    p.id::text as event_id,
                    p.created_at as timestamp,
                    p.symbol,
                    CASE WHEN p.quantity > 0 THEN 'BUY' ELSE 'SELL' END as side,
                    p.quantity,
                    p.avg_price as price,
                    p.status,
                    0 as commission,
                    p.unrealized_pnl as pnl,
                    s.name as strategy_name,
                    p.metadata
                FROM positions p
                LEFT JOIN strategies s ON p.strategy_id = s.id

                UNION ALL

                SELECT
                    'ORDER' as event_type,
                    o.id::text as event_id,
                    o.created_at as timestamp,
                    o.symbol,
                    o.side,
                    o.quantity,
                    o.price,
                    o.status,
                    0 as commission,
                    0 as pnl,
                    s.name as strategy_name,
                    o.metadata
                FROM orders o
                LEFT JOIN strategies s ON o.strategy_id = s.id
            """

            params = []
            param_count = 1

            if request.start_date:
                query = query.replace("WHERE", f"WHERE timestamp >= ${param_count} AND")
                query = query.replace(
                    "UNION ALL", f"AND timestamp >= ${param_count} UNION ALL"
                )
                params.append(request.start_date)
                param_count += 1

            if request.end_date:
                end_param = f"${param_count}"
                query = query.replace("WHERE", f"WHERE timestamp <= {end_param} AND")
                query = query.replace(
                    "UNION ALL", f"AND timestamp <= {end_param} UNION ALL"
                )
                params.append(request.end_date)
                param_count += 1

            query += " ORDER BY timestamp DESC"

            rows = await conn.fetch(query, *params)

            # Format audit trail
            audit_events = []
            for row in rows:
                audit_events.append(
                    {
                        "timestamp": row["timestamp"].isoformat(),
                        "event_type": row["event_type"],
                        "event_id": row["event_id"],
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "quantity": float(row["quantity"]),
                        "price": float(row["price"]),
                        "status": row["status"],
                        "commission": float(row["commission"]),
                        "pnl": float(row["pnl"] or 0),
                        "strategy": row["strategy_name"],
                        "metadata": row["metadata"],
                    }
                )

            audit_report = {
                "period": {
                    "start_date": (
                        request.start_date.isoformat() if request.start_date else None
                    ),
                    "end_date": (
                        request.end_date.isoformat() if request.end_date else None
                    ),
                },
                "total_events": len(audit_events),
                "events": audit_events,
                "generated_at": datetime.now().isoformat(),
            }

            # Generate filename and save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_trail_{timestamp}.json"
            filepath = self.exports_dir / filename

            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(audit_report, indent=2))

            logger.info(f"Audit trail export completed: {filepath}")
            return str(filepath)

        finally:
            if conn and self.db.pool:
                await self.db.pool.release(conn)

    async def export_portfolio_analysis(self, request: ExportRequest) -> str:
        """Export detailed portfolio analysis"""
        logger.info(f"Starting portfolio analysis export: {request}")

        conn = await self.db.get_connection()
        try:
            # Current positions
            positions_query = """
                SELECT
                    p.symbol,
                    p.quantity,
                    p.avg_price,
                    p.current_price,
                    p.unrealized_pnl,
                    p.realized_pnl,
                    p.created_at,
                    s.name as strategy_name,
                    p.metadata
                FROM positions p
                LEFT JOIN strategies s ON p.strategy_id = s.id
                WHERE p.status = 'OPEN'
                ORDER BY p.unrealized_pnl DESC
            """

            position_rows = await conn.fetch(positions_query)

            # Risk metrics by symbol
            risk_query = """
                SELECT
                    symbol,
                    COUNT(*) as trade_count,
                    SUM(ABS(quantity * price)) as total_volume,
                    STDDEV(pnl) as pnl_volatility,
                    MAX(pnl) as max_gain,
                    MIN(pnl) as max_loss,
                    AVG(pnl) as avg_pnl
                FROM trades
                WHERE status = 'FILLED'
                GROUP BY symbol
                ORDER BY total_volume DESC
            """

            risk_rows = await conn.fetch(risk_query)

            # Compile analysis
            current_positions = []
            for row in position_rows:
                market_value = float(row["quantity"]) * float(row["current_price"])
                cost_basis = float(row["quantity"]) * float(row["avg_price"])

                current_positions.append(
                    {
                        "symbol": row["symbol"],
                        "quantity": float(row["quantity"]),
                        "avg_price": float(row["avg_price"]),
                        "current_price": float(row["current_price"]),
                        "market_value": market_value,
                        "cost_basis": cost_basis,
                        "unrealized_pnl": float(row["unrealized_pnl"]),
                        "realized_pnl": float(row["realized_pnl"]),
                        "position_age_days": (datetime.now() - row["created_at"]).days,
                        "strategy": row["strategy_name"],
                        "metadata": row["metadata"],
                    }
                )

            risk_metrics = []
            for row in risk_rows:
                risk_metrics.append(
                    {
                        "symbol": row["symbol"],
                        "trade_count": row["trade_count"],
                        "total_volume": float(row["total_volume"]),
                        "pnl_volatility": float(row["pnl_volatility"] or 0),
                        "max_gain": float(row["max_gain"]),
                        "max_loss": float(row["max_loss"]),
                        "avg_pnl": float(row["avg_pnl"]),
                        "risk_score": (
                            float(row["pnl_volatility"] or 0)
                            / abs(float(row["avg_pnl"]))
                            if row["avg_pnl"]
                            else 0
                        ),
                    }
                )

            # Portfolio summary
            total_market_value = sum(pos["market_value"] for pos in current_positions)
            total_cost_basis = sum(pos["cost_basis"] for pos in current_positions)
            total_unrealized_pnl = sum(
                pos["unrealized_pnl"] for pos in current_positions
            )

            analysis_report = {
                "portfolio_summary": {
                    "total_positions": len(current_positions),
                    "total_market_value": total_market_value,
                    "total_cost_basis": total_cost_basis,
                    "total_unrealized_pnl": total_unrealized_pnl,
                    "portfolio_return_pct": (
                        (total_unrealized_pnl / total_cost_basis * 100)
                        if total_cost_basis > 0
                        else 0
                    ),
                },
                "current_positions": current_positions,
                "risk_metrics": risk_metrics,
                "analysis_date": datetime.now().isoformat(),
            }

            # Generate filename and save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_analysis_{timestamp}.json"
            filepath = self.exports_dir / filename

            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(analysis_report, indent=2))

            logger.info(f"Portfolio analysis export completed: {filepath}")
            return str(filepath)

        finally:
            if conn and self.db.pool:
                await self.db.pool.release(conn)

    async def create_export_archive(
        self, filepaths: List[str], archive_name: str
    ) -> str:
        """Create a compressed archive of multiple export files"""
        archive_path = self.exports_dir / f"{archive_name}.zip"

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for filepath in filepaths:
                if os.path.exists(filepath):
                    zipf.write(filepath, os.path.basename(filepath))

        logger.info(f"Export archive created: {archive_path}")
        return str(archive_path)


# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


app = FastAPI(
    title="Export Service",
    description="AI Trading System Export Service",
    version="1.0.0",
    lifespan=lifespan,
)


async def get_export_service() -> ExportService:
    """Dependency to get export service instance"""
    return ExportService(db_manager)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Verify JWT token"""
    # Simplified token verification
    token = credentials.credentials
    # In production, verify JWT signature and claims
    return token


async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        await db_manager.initialize()

        # Initialize metrics collection
        try:
            from monitoring.metrics import MetricsCollector
            from shared.config import get_config

            config = get_config()
            metrics = MetricsCollector(config)
            await metrics.startup()
            logger.info("Metrics collector initialized")
        except Exception as e:
            logger.warning(f"Metrics initialization failed: {e}")

        # Create exports directory
        exports_dir = Path("/tmp/exports")
        exports_dir.mkdir(exist_ok=True)

        logger.info("Export service startup completed")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await db_manager.close()
        logger.info("Export service shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "export_service",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/exports/tradenote")
async def export_tradenote(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols"),
    strategies: Optional[str] = Query(None, description="Comma-separated strategies"),
    service: ExportService = Depends(get_export_service),
    token: str = Depends(verify_token),
):
    """Export trades in TradeNote compatible format"""
    try:
        request = ExportRequest(
            format="tradenote",
            start_date=datetime.fromisoformat(start_date) if start_date else None,
            end_date=datetime.fromisoformat(end_date) if end_date else None,
            symbols=symbols.split(",") if symbols else None,
            strategies=strategies.split(",") if strategies else None,
        )

        file_path = await service.export_tradenote_format(request)
        return FileResponse(
            path=file_path,
            filename=f"tradenote_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            media_type="text/csv",
        )
    except Exception as e:
        logger.error(f"TradeNote export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SERVICE_PORT", "9106"))
    host = "0.0.0.0"

    logger.info(f"Starting Export Service on {host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")
