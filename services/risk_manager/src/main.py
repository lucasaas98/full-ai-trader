"""
Main application entry point for the Risk Management Service.

This module provides the main FastAPI application that orchestrates all components
of the risk management system including position sizing, risk monitoring,
alert management, and portfolio protection.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, generate_latest
from pydantic import BaseModel

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from decimal import Decimal  # noqa: E402

from shared.config import get_config  # noqa: E402
from shared.models import (  # noqa: E402
    DailyRiskReport,
    OrderRequest,
    PortfolioMetrics,
    PortfolioState,
    PositionSizing,
    RiskAlert,
    RiskEvent,
    RiskEventType,
    RiskLimits,
    RiskSeverity,
    TradeSignal,
)

from .alert_manager import AlertManager  # noqa: E402
from .alpaca_client import AlpacaRiskClient  # noqa: E402
from .database_manager import RiskDatabaseManager  # noqa: E402
from .portfolio_monitor import PortfolioMonitor  # noqa: E402
from .position_sizer import PositionSizer  # noqa: E402
from .risk_manager import RiskManager  # noqa: E402


# Configure logging
def setup_logging():
    """Set up logging configuration."""
    config = get_config()
    log_config = config.logging

    # Create logs directory if it doesn't exist
    log_path = Path(log_config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    handlers: List[logging.Handler] = []
    if log_config.enable_console:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_config.enable_file:
        handlers.append(logging.FileHandler(log_config.file_path))

    logging.basicConfig(
        level=logging.DEBUG,
        format=log_config.format,
        handlers=handlers,
    )


setup_logging()
logger = logging.getLogger(__name__)

# Global service instances
risk_manager: Optional[RiskManager] = None
position_sizer: Optional[PositionSizer] = None
portfolio_monitor: Optional[PortfolioMonitor] = None
alert_manager: Optional[AlertManager] = None
database_manager: Optional[RiskDatabaseManager] = None
alpaca_client: Optional[AlpacaRiskClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global risk_manager, position_sizer, portfolio_monitor, alert_manager, database_manager

    # Initialize Prometheus metrics
    init_prometheus_metrics()

    # Startup
    logger.info("Starting Risk Management Service...")
    logger.debug("Initializing application components...")

    try:
        # Initialize configuration
        config = get_config()
        logger.debug(
            f"Configuration loaded: risk.max_position_size={config.risk.max_position_size}"
        )

        # Initialize database manager
        global database_manager
        database_manager = RiskDatabaseManager()
        logger.debug("Initializing database manager...")
        await database_manager.initialize()
        logger.debug("Database manager initialized successfully")

        # Initialize risk limits
        risk_limits = RiskLimits(
            max_position_percentage=config.risk.max_position_size,
            stop_loss_percentage=config.risk.stop_loss_percentage,
            take_profit_percentage=config.risk.take_profit_percentage,
            max_daily_loss_percentage=Decimal("0.05"),  # 5% daily loss limit
            max_positions=5,
            max_correlation_threshold=config.risk.max_correlation,
            emergency_stop_percentage=Decimal("0.10"),  # 10% emergency stop
        )

        # Initialize service components
        global risk_manager, position_sizer, portfolio_monitor, alert_manager, alpaca_client
        logger.debug("Initializing core service components...")
        risk_manager = RiskManager()
        logger.debug("Risk manager initialized")
        position_sizer = PositionSizer(risk_limits)
        logger.debug("Position sizer initialized")
        portfolio_monitor = PortfolioMonitor(risk_limits)
        logger.debug("Portfolio monitor initialized")
        alert_manager = AlertManager()
        logger.debug("Alert manager initialized")

        # Initialize Alpaca client
        logger.debug("Initializing Alpaca client...")
        alpaca_client = AlpacaRiskClient()
        logger.debug("Alpaca client initialized")

        # Start background tasks
        logger.debug("Starting background tasks...")
        asyncio.create_task(alert_manager.process_alert_queue())
        logger.debug("Alert processing task started")
        asyncio.create_task(periodic_portfolio_monitoring())
        logger.debug("Portfolio monitoring task started")
        asyncio.create_task(daily_cleanup_task())
        logger.debug("Daily cleanup task started")

        logger.info("Risk Management Service started successfully")

        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    # Shutdown
    logger.info("Shutting down Risk Management Service...")

    try:
        if database_manager:
            await database_manager.close()
        logger.info("Risk Management Service shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# FastAPI app
app = FastAPI(
    title="Risk Management Service",
    description="Comprehensive risk management for automated trading",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
risk_events_counter = None
portfolio_value_gauge = None
positions_count_gauge = None
alerts_counter = None
service_health_gauge = None


# Request/Response Models
class TradeValidationRequest(BaseModel):
    order_request: OrderRequest
    portfolio: PortfolioState
    signal: Optional[TradeSignal] = None


class TradeValidationResponse(BaseModel):
    is_valid: bool
    risk_filters: List[Dict]
    recommended_adjustments: Optional[Dict] = None


class PositionSizingRequest(BaseModel):
    symbol: str
    current_price: Decimal
    portfolio: PortfolioState
    confidence_score: Optional[float] = None
    signal: Optional[TradeSignal] = None


class RiskOverrideRequest(BaseModel):
    override_type: str
    reason: str
    duration_minutes: Optional[int] = 60
    authorized_by: str


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    try:
        # Check database connection
        db_healthy = (
            await database_manager.health_check() if database_manager else False
        )

        # Check service components
        services_healthy = all(
            [
                risk_manager is not None,
                position_sizer is not None,
                portfolio_monitor is not None,
                alert_manager is not None,
            ]
        )

        overall_healthy = db_healthy and services_healthy

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": "healthy" if db_healthy else "unhealthy",
            "services": "healthy" if services_healthy else "unhealthy",
            "emergency_stop_active": (
                risk_manager.emergency_stop_active if risk_manager else False
            ),
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}


# Risk validation endpoints
@app.post("/validate-trade", response_model=TradeValidationResponse)
async def validate_trade(request: TradeValidationRequest):
    """Validate a trade request against risk parameters."""
    logger.debug(
        f"Trade validation request for {request.order_request.symbol}: {request.order_request.quantity} shares"
    )
    try:
        if not risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        is_valid, filters = await risk_manager.validate_trade_request(
            request.order_request, request.portfolio, request.signal
        )
        logger.debug(
            f"Trade validation result for {request.order_request.symbol}: valid={is_valid}, filters={len(filters)}"
        )

        # Convert filters to dict format
        filter_dicts = []
        for f in filters:
            filter_dict = {
                "passed": f.passed,
                "filter_name": f.filter_name,
                "reason": f.reason,
                "value": f.value,
                "limit": f.limit,
                "severity": f.severity,
            }
            filter_dicts.append(filter_dict)

        # Generate recommended adjustments if trade is invalid
        recommendations = None
        if not is_valid:
            recommendations = await _generate_trade_recommendations(request, filters)

        return TradeValidationResponse(
            is_valid=is_valid,
            risk_filters=filter_dicts,
            recommended_adjustments=recommendations,
        )

    except Exception as e:
        logger.error(f"Error validating trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate-position-size", response_model=PositionSizing)
async def calculate_position_size(request: PositionSizingRequest):
    """Calculate optimal position size for a trade."""
    logger.debug(
        f"Position sizing request for {request.symbol} at ${request.current_price}"
    )
    try:
        if not position_sizer:
            raise HTTPException(
                status_code=503, detail="Position sizer not initialized"
            )

        sizing = await position_sizer.calculate_position_size(
            symbol=request.symbol,
            current_price=request.current_price,
            portfolio=request.portfolio,
            signal=request.signal,
        )
        logger.debug(
            f"Position sizing result for {request.symbol}: {sizing.recommended_shares} shares (${sizing.recommended_value})"
        )

        return sizing

    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio monitoring endpoints
@app.get("/portfolio-metrics", response_model=PortfolioMetrics)
async def get_portfolio_metrics(portfolio: PortfolioState):
    """Get current portfolio risk metrics."""
    logger.debug(
        f"Portfolio metrics request for portfolio with {len(portfolio.positions)} positions"
    )
    try:
        if not portfolio_monitor:
            raise HTTPException(
                status_code=503, detail="Portfolio monitor not initialized"
            )

        metrics = await portfolio_monitor.calculate_detailed_metrics(portfolio)
        logger.debug(
            f"Portfolio metrics calculated: exposure=${metrics.total_exposure}, volatility={metrics.volatility}"
        )

        # Store metrics in database
        if database_manager:
            await database_manager.store_portfolio_metrics(metrics)
            logger.debug("Portfolio metrics stored in database")

        return metrics

    except Exception as e:
        logger.error(f"Error getting portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor-portfolio")
async def monitor_portfolio(
    portfolio: PortfolioState, background_tasks: BackgroundTasks
):
    """Monitor portfolio and generate alerts."""
    logger.debug(
        f"Portfolio monitoring request for portfolio value ${portfolio.total_equity}"
    )
    try:
        if not portfolio_monitor or not alert_manager:
            raise HTTPException(status_code=503, detail="Services not initialized")

        metrics, alerts = await portfolio_monitor.monitor_portfolio(portfolio)
        logger.debug(f"Portfolio monitoring completed: {len(alerts)} alerts generated")

        # Store portfolio snapshot
        if database_manager:
            background_tasks.add_task(
                database_manager.store_portfolio_snapshot, portfolio
            )
            background_tasks.add_task(database_manager.store_portfolio_metrics, metrics)

        # Send alerts
        if alerts:
            background_tasks.add_task(alert_manager.send_bulk_alerts, alerts)
            logger.debug(f"Queued {len(alerts)} alerts for sending")

        # Update risk manager state
        if risk_manager:
            risk_manager.set_portfolio_snapshot(portfolio)
            logger.debug("Risk manager portfolio snapshot updated")

        return {
            "status": "monitoring_complete",
            "metrics": metrics,
            "alerts_generated": len(alerts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error monitoring portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Stop loss and position management endpoints
@app.post("/update-trailing-stops")
async def update_trailing_stops(
    portfolio: PortfolioState, market_prices: Dict[str, Decimal]
):
    """Update trailing stops for all positions."""
    logger.debug(
        f"Updating trailing stops for {len(portfolio.positions)} positions with {len(market_prices)} price updates"
    )
    try:
        if not risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        events = await risk_manager.update_trailing_stops(portfolio, market_prices)
        logger.debug(f"Trailing stops update generated {len(events)} events")

        # Store events in database
        if database_manager and events:
            for event in events:
                await database_manager.store_risk_event(event)
            logger.debug(f"Stored {len(events)} risk events in database")

        # Send alerts for triggered stops
        if events and alert_manager:
            alerts = []
            for event in events:
                alert = RiskAlert(
                    alert_type=event.event_type,
                    severity=event.severity,
                    symbol=event.symbol,
                    title=f"Stop Loss Triggered: {event.symbol}",
                    message=event.description,
                    action_required=True,
                    metadata=event.metadata,
                )
                alerts.append(alert)

            await alert_manager.send_bulk_alerts(alerts)

        return {
            "events_generated": len(events),
            "trailing_stops_updated": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error updating trailing stops: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stop-loss-levels/{symbol}")
async def get_stop_loss_levels(symbol: str, entry_price: Decimal, side: str):
    """Calculate stop loss and take profit levels for a position."""
    logger.debug(
        f"Stop loss calculation request for {symbol} at ${entry_price} ({side})"
    )
    try:
        if not risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        from shared.models import OrderSide

        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        stop_loss, take_profit = await risk_manager.calculate_stop_loss_take_profit(
            symbol, entry_price, order_side
        )

        return {
            "symbol": symbol,
            "entry_price": str(entry_price),
            "stop_loss": str(stop_loss),
            "take_profit": str(take_profit),
            "risk_reward_ratio": (
                float((take_profit - entry_price) / (entry_price - stop_loss))
                if order_side == OrderSide.BUY
                else float((entry_price - take_profit) / (stop_loss - entry_price))
            ),
            "side": side,
        }

    except Exception as e:
        logger.error(f"Error calculating stop levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Emergency controls
@app.post("/emergency-stop")
async def activate_emergency_stop(reason: str):
    """Activate emergency stop for all trading."""
    logger.debug(f"Emergency stop activation requested: {reason}")
    try:
        if not risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        risk_manager.activate_emergency_stop(reason)
        logger.debug("Emergency stop activated in risk manager")

        # Create emergency event
        event = RiskEvent(
            event_type=RiskEventType.EMERGENCY_STOP,
            severity=RiskSeverity.CRITICAL,
            symbol=None,
            description=f"Emergency stop activated: {reason}",
            action_taken="halt_all_trading",
            resolved_at=None,
        )

        # Send emergency alert
        if alert_manager:
            await alert_manager.send_emergency_alert(event)

        # Store event in database
        if database_manager:
            await database_manager.store_risk_event(event)

        logger.critical(f"Emergency stop activated: {reason}")

        return {
            "status": "emergency_stop_activated",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error activating emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deactivate-emergency-stop")
async def deactivate_emergency_stop(authorized_by: str):
    """Deactivate emergency stop (requires authorization)."""
    logger.debug(f"Emergency stop deactivation requested by {authorized_by}")
    try:
        if not risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        risk_manager.deactivate_emergency_stop()
        logger.debug("Emergency stop deactivated in risk manager")

        logger.warning(f"Emergency stop deactivated by {authorized_by}")

        return {
            "status": "emergency_stop_deactivated",
            "authorized_by": authorized_by,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error deactivating emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Risk reporting endpoints
@app.get("/risk-events")
async def get_risk_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
):
    """Get risk events with filtering."""
    logger.debug(
        f"Risk events query: start={start_date}, end={end_date}, type={event_type}, severity={severity}, symbol={symbol}"
    )
    try:
        if not database_manager:
            raise HTTPException(
                status_code=503, detail="Database manager not initialized"
            )

        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        events = await database_manager.get_risk_events(
            start_date=start_dt,
            end_date=end_dt,
            event_type=event_type,
            severity=severity,
            symbol=symbol,
            limit=limit,
        )

        return {"events": events, "total": len(events)}

    except Exception as e:
        logger.error(f"Error retrieving risk events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-statistics")
async def get_risk_statistics(days: int = 30):
    """Get risk statistics for the specified period."""
    logger.debug(f"Risk statistics request for {days} days")
    try:
        if not database_manager:
            raise HTTPException(
                status_code=503, detail="Database manager not initialized"
            )

        statistics = await database_manager.get_risk_statistics(days)

        return statistics

    except Exception as e:
        logger.error(f"Error getting risk statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/daily-report")
async def get_daily_report(date: Optional[str] = None):
    """Generate or retrieve daily risk report."""
    logger.debug(f"Daily report request for date: {date}")
    try:
        if not portfolio_monitor or not database_manager:
            raise HTTPException(status_code=503, detail="Services not initialized")

        # Use today if no date specified
        if date:
            report_date = datetime.fromisoformat(date)
        else:
            report_date = datetime.now(timezone.utc)

        # Get risk events for the day
        start_of_day = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        events_dict = await database_manager.get_risk_events(
            start_date=start_of_day, end_date=end_of_day
        )

        # Convert dictionary events to RiskEvent objects
        events = []
        for event_dict in events_dict:
            try:
                event = RiskEvent(
                    id=event_dict.get("id"),
                    event_type=RiskEventType(
                        event_dict.get("event_type", "emergency_stop")
                    ),
                    severity=RiskSeverity(event_dict.get("severity", "medium")),
                    symbol=event_dict.get("symbol"),
                    description=event_dict.get("description", ""),
                    timestamp=event_dict.get("timestamp", datetime.now(timezone.utc)),
                    resolved_at=event_dict.get("resolved_at"),
                    action_taken=event_dict.get("action_taken"),
                    metadata=event_dict.get("metadata", {}),
                )
                events.append(event)
            except Exception as e:
                logger.warning(f"Error converting risk event dict to object: {e}")
                continue

        # Get latest portfolio metrics
        metrics = await database_manager.get_latest_portfolio_metrics()

        if metrics:
            # Create daily report
            report = DailyRiskReport(
                date=report_date,
                portfolio_value=Decimal(str(metrics["total_exposure"])),
                daily_pnl=Decimal("0"),  # Would need to calculate from historical data
                daily_return=0.0,
                max_drawdown=Decimal(str(metrics["max_drawdown"])),
                current_drawdown=Decimal(str(metrics["current_drawdown"])),
                volatility=float(metrics["volatility"]),
                sharpe_ratio=float(metrics["sharpe_ratio"]),
                var_1d=Decimal(str(metrics["value_at_risk_1d"])),
                total_trades=0,  # Would need to get from trades data
                winning_trades=0,
                risk_events=events,
                position_risks=[],  # Would get from position_risks table
                compliance_violations=[],
            )

            # Store report
            await database_manager.store_daily_report(report)

            return {
                "date": report_date.date().isoformat(),
                "portfolio_value": str(report.portfolio_value),
                "daily_pnl": str(report.daily_pnl),
                "daily_return": report.daily_return,
                "max_drawdown": str(report.max_drawdown),
                "current_drawdown": str(report.current_drawdown),
                "volatility": report.volatility,
                "sharpe_ratio": report.sharpe_ratio,
                "var_1d": str(report.var_1d),
                "total_trades": report.total_trades,
                "winning_trades": report.winning_trades,
                "risk_events_count": len(report.risk_events),
                "compliance_violations": report.compliance_violations,
            }
        else:
            raise HTTPException(
                status_code=404, detail="No portfolio metrics available"
            )

    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Alert management endpoints
@app.get("/alerts")
async def get_alerts(acknowledged: bool = False, limit: int = 50):
    """Get risk alerts."""
    logger.debug(f"Alerts request: acknowledged={acknowledged}, limit={limit}")
    try:
        if not database_manager:
            raise HTTPException(
                status_code=503, detail="Database manager not initialized"
            )

        if not acknowledged:
            alerts = await database_manager.get_unacknowledged_alerts(limit=limit)
        else:
            # Would need additional method to get acknowledged alerts
            alerts = []

        return {"alerts": alerts, "total": len(alerts)}

    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str):
    """Acknowledge a risk alert."""
    logger.debug(f"Alert acknowledgment request: {alert_id} by {acknowledged_by}")
    try:
        if not database_manager:
            raise HTTPException(
                status_code=503, detail="Database manager not initialized"
            )

        from uuid import UUID

        alert_uuid = UUID(alert_id)

        success = await database_manager.acknowledge_alert(alert_uuid, acknowledged_by)

        if success:
            return {
                "status": "acknowledged",
                "alert_id": alert_id,
                "acknowledged_by": acknowledged_by,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alert ID format")
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@app.get("/risk-limits")
async def get_risk_limits():
    """Get current risk limits configuration."""
    logger.debug("Risk limits configuration request")
    try:
        if not risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        return {
            "max_position_percentage": str(
                risk_manager.risk_limits.max_position_percentage
            ),
            "max_positions": risk_manager.risk_limits.max_positions,
            "stop_loss_percentage": str(risk_manager.risk_limits.stop_loss_percentage),
            "take_profit_percentage": str(
                risk_manager.risk_limits.take_profit_percentage
            ),
            "max_daily_loss_percentage": str(
                risk_manager.risk_limits.max_daily_loss_percentage
            ),
            "max_correlation_threshold": risk_manager.risk_limits.max_correlation_threshold,
            "emergency_stop_percentage": str(
                risk_manager.risk_limits.emergency_stop_percentage
            ),
            "max_position_volatility": risk_manager.risk_limits.max_position_volatility,
        }

    except Exception as e:
        logger.error(f"Error getting risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def init_prometheus_metrics():
    """Initialize Prometheus metrics."""
    global risk_events_counter, portfolio_value_gauge, positions_count_gauge, alerts_counter, service_health_gauge

    risk_events_counter = Counter(
        "risk_manager_events_total",
        "Total number of risk events",
        ["event_type", "severity"],
    )

    portfolio_value_gauge = Gauge(
        "risk_manager_portfolio_value", "Current portfolio value"
    )

    positions_count_gauge = Gauge(
        "risk_manager_positions_total", "Number of active positions"
    )

    alerts_counter = Counter(
        "risk_manager_alerts_total", "Total number of alerts generated", ["severity"]
    )

    service_health_gauge = Gauge(
        "risk_manager_service_health",
        "Health status of risk manager components",
        ["component"],
    )


async def update_prometheus_metrics():
    """Update Prometheus metrics with current values."""
    global risk_events_counter, portfolio_value_gauge, positions_count_gauge, alerts_counter, service_health_gauge  # noqa: F824

    try:
        # Update service health metrics
        if service_health_gauge:
            service_health_gauge.labels(component="database").set(
                1 if database_manager else 0
            )
            service_health_gauge.labels(component="risk_manager").set(
                1 if risk_manager else 0
            )
            service_health_gauge.labels(component="portfolio_monitor").set(
                1 if portfolio_monitor else 0
            )
            service_health_gauge.labels(component="alert_manager").set(
                1 if alert_manager else 0
            )

        # Get latest portfolio metrics if available
        if database_manager:
            try:
                latest_metrics = await database_manager.get_latest_portfolio_metrics()
                if latest_metrics and portfolio_value_gauge and positions_count_gauge:
                    portfolio_value_gauge.set(latest_metrics.get("total_exposure", 0))
                    positions_count_gauge.set(latest_metrics.get("position_count", 0))
            except Exception as e:
                logger.debug(f"Could not update portfolio metrics: {e}")

    except Exception as e:
        logger.error(f"Failed to update Prometheus metrics: {e}")


@app.put("/risk-limits")
async def update_risk_limits(limits: Dict):
    """Update risk limits configuration."""
    logger.debug(f"Risk limits update request: {limits}")
    try:
        if not risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not initialized")

        # Update risk limits
        for key, value in limits.items():
            if hasattr(risk_manager.risk_limits, key):
                if isinstance(value, str) and key.endswith("_percentage"):
                    setattr(risk_manager.risk_limits, key, Decimal(value))
                else:
                    setattr(risk_manager.risk_limits, key, value)

        logger.debug(f"Risk limits updated in memory: {limits}")
        logger.info(f"Risk limits updated: {limits}")

        return {
            "status": "updated",
            "updated_limits": limits,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    try:
        # Update metrics before returning them
        await update_prometheus_metrics()

        # Generate Prometheus format
        metrics_output = generate_latest()

        return Response(content=metrics_output, media_type="text/plain; version=0.0.4")

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain; version=0.0.4",
            status_code=500,
        )


# Testing and diagnostics endpoints
@app.post("/test-notifications")
async def test_notifications():
    """Test all notification channels."""
    logger.debug("Notification test request")
    try:
        if not alert_manager:
            raise HTTPException(status_code=503, detail="Alert manager not initialized")

        results = await alert_manager.test_notifications()

        return {
            "test_results": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error testing notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system-status")
async def get_system_status():
    """Get comprehensive system status."""
    logger.debug("System status request")
    try:
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {
                "risk_manager": risk_manager is not None,
                "position_sizer": position_sizer is not None,
                "portfolio_monitor": portfolio_monitor is not None,
                "alert_manager": alert_manager is not None,
                "database_manager": database_manager is not None,
            },
            "emergency_stop_active": (
                risk_manager.emergency_stop_active if risk_manager else False
            ),
            "daily_trade_count": risk_manager.daily_trade_count if risk_manager else 0,
            "daily_pnl": str(risk_manager.daily_pnl) if risk_manager else "0",
        }

        # Add database health check
        if database_manager:
            status["database_healthy"] = await database_manager.health_check()

        # Add alert statistics
        if alert_manager:
            status["alert_statistics"] = alert_manager.get_alert_statistics()

        # Add monitoring statistics
        if portfolio_monitor:
            status["monitoring_statistics"] = (
                portfolio_monitor.get_monitoring_statistics()
            )

        return status

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def periodic_portfolio_monitoring():
    """Periodic portfolio monitoring task."""
    logger.debug("Starting periodic portfolio monitoring task")
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            logger.debug("Executing periodic portfolio monitoring cycle")

            # Fetch real portfolio data from Alpaca and store it
            await sync_portfolio_data()

        except Exception as e:
            logger.error(f"Error in periodic monitoring: {e}")
            logger.debug("Retrying periodic monitoring in 5 minutes")
            await asyncio.sleep(300)


async def sync_portfolio_data():
    """Fetch portfolio data from Alpaca and store in database."""
    logger.debug("Starting portfolio data sync from Alpaca")
    try:
        if not alpaca_client:
            logger.warning("Alpaca client not available for portfolio sync")
            return False

        # Get current portfolio state from Alpaca
        logger.debug("Fetching portfolio state from Alpaca")
        portfolio = await alpaca_client.get_portfolio_state(force_refresh=True)
        if not portfolio:
            logger.error("Failed to fetch portfolio data from Alpaca")
            return False
        logger.debug(
            f"Portfolio data received: ${portfolio.total_equity} equity, {len(portfolio.positions)} positions"
        )

        # Store portfolio snapshot in database
        if database_manager:
            logger.debug("Storing portfolio snapshot in database")
            await database_manager.store_portfolio_snapshot(portfolio)
            logger.info(
                f"Portfolio snapshot stored: ${portfolio.total_equity} equity, {len(portfolio.positions)} positions"
            )

        # Calculate and store metrics if we have a portfolio monitor
        if portfolio_monitor:
            logger.debug("Calculating detailed portfolio metrics")
            metrics = await portfolio_monitor.calculate_detailed_metrics(portfolio)
            if database_manager:
                await database_manager.store_portfolio_metrics(metrics)
                logger.debug("Portfolio metrics calculated and stored")

        return True

    except Exception as e:
        logger.error(f"Error syncing portfolio data: {e}")
        return False


@app.post("/sync-portfolio")
async def sync_portfolio_endpoint():
    """Endpoint to manually trigger portfolio data sync from Alpaca."""
    logger.debug("Manual portfolio sync requested")
    try:
        success = await sync_portfolio_data()
        if success:
            return {
                "status": "success",
                "message": "Portfolio data synced successfully",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to sync portfolio data")

    except Exception as e:
        logger.error(f"Error in sync portfolio endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def daily_cleanup_task():
    """Daily cleanup task."""
    logger.debug("Starting daily cleanup task")
    while True:
        try:
            # Wait until next day
            now = datetime.now(timezone.utc)
            next_day = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            wait_seconds = (next_day - now).total_seconds()

            await asyncio.sleep(wait_seconds)
            logger.debug("Executing daily cleanup cycle")

            # Reset daily counters
            if risk_manager:
                risk_manager.reset_daily_counters()
                logger.debug("Risk manager daily counters reset")

            if portfolio_monitor:
                portfolio_monitor.reset_monitoring_state()
                logger.debug("Portfolio monitor state reset")

            # Cleanup old data
            if database_manager:
                logger.debug("Starting database cleanup")
                deleted_count = await database_manager.cleanup_old_data()
                logger.info(f"Daily cleanup completed: {deleted_count} records deleted")

        except Exception as e:
            logger.error(f"Error in daily cleanup: {e}")
            logger.debug("Retrying daily cleanup in 1 hour")
            await asyncio.sleep(3600)  # Retry in 1 hour


async def _generate_trade_recommendations(
    request: TradeValidationRequest, filters
) -> Dict:
    """Generate recommendations for fixing trade validation issues."""
    logger.debug(
        f"Generating trade recommendations for {len([f for f in filters if not f.passed])} failed filters"
    )
    recommendations = {}

    failed_filters = [f for f in filters if not f.passed]

    for filter_result in failed_filters:
        if filter_result.filter_name == "position_size":
            # Recommend smaller position size
            current_size = request.order_request.quantity
            recommended_size = int(current_size * 0.5)  # Reduce by 50%
            recommendations["position_size"] = {
                "current_shares": current_size,
                "recommended_shares": recommended_size,
                "reason": "Reduce position size to meet risk limits",
            }

        elif filter_result.filter_name == "buying_power":
            # Recommend position size based on available buying power
            available = request.portfolio.buying_power
            price = request.order_request.price or Decimal("100")
            max_shares = int(
                available / price * Decimal("0.95")
            )  # Use 95% of available
            recommendations["buying_power"] = {
                "current_shares": request.order_request.quantity,
                "max_affordable_shares": max_shares,
                "reason": "Adjust position size based on available buying power",
            }

        elif filter_result.filter_name == "correlation_limit":
            recommendations["correlation"] = {
                "reason": "Consider reducing existing correlated positions before opening new position",
                "action": "Review portfolio correlation and reduce similar positions",
            }

    return recommendations


# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    logger.debug(
        "Signal handler triggered, cleanup will be managed by lifespan context"
    )
    # The lifespan context manager will handle cleanup


class RiskManagerApp:
    """Application wrapper for Risk Manager service for integration testing."""

    def __init__(self):
        """Initialize the Risk Manager application."""
        self.app = app
        self._initialized = False
        logger.debug("RiskManagerApp instance created")

    async def initialize(self):
        """Initialize the application."""
        logger.debug("Initializing RiskManagerApp...")
        if not self._initialized:
            # Initialize database if needed
            global database_manager
            if database_manager is None:
                logger.debug("Creating database manager for app wrapper")
                database_manager = RiskDatabaseManager()
                await database_manager.initialize()
            self._initialized = True
            logger.debug("RiskManagerApp initialized successfully")

    async def start(self):
        """Start the Risk Manager service."""
        logger.debug("Starting RiskManagerApp...")
        await self.initialize()

    async def stop(self):
        """Stop the Risk Manager service."""
        logger.debug("Stopping RiskManagerApp...")
        if database_manager:
            await database_manager.close()
        self._initialized = False
        logger.debug("RiskManagerApp stopped")

    def get_app(self):
        """Get the FastAPI application instance."""
        return self.app


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get configuration
    config = get_config()
    logger.info("Starting Risk Manager service from main...")

    # Run the application
    import os

    port = int(os.environ.get("SERVICE_PORT", 9103))
    logger.info(f"Starting Risk Manager service on port {port}")
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=config.logging.level.lower(),
        access_log=True,
    )
