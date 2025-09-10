"""
Risk Management Service Integration

This module provides the main risk service that integrates all risk management
components into a cohesive system for portfolio protection and risk monitoring.
"""

import asyncio
import logging
from asyncio import Task
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.config import get_config
from shared.models import (
    DailyRiskReport,
    OrderRequest,
    OrderSide,
    PortfolioMetrics,
    PortfolioState,
    PositionSizing,
    PositionSizingMethod,
    RiskAlert,
    RiskEvent,
    RiskEventType,
    RiskLimits,
    RiskSeverity,
    TradeSignal,
)

from .alert_manager import AlertManager
from .alpaca_client import AlpacaRiskClient
from .database_manager import RiskDatabaseManager
from .portfolio_monitor import PortfolioMonitor
from .position_sizer import PositionSizer
from .risk_calculator import RiskCalculator
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


def _convert_risk_event_dict_to_object(event_dict: Dict[str, Any]) -> RiskEvent:
    """Convert risk event dictionary from database to RiskEvent object."""
    return RiskEvent(
        id=event_dict.get("id"),
        event_type=RiskEventType(event_dict.get("event_type", "emergency_stop")),
        severity=RiskSeverity(event_dict.get("severity", "medium")),
        symbol=event_dict.get("symbol"),
        description=event_dict.get("description", ""),
        timestamp=event_dict.get("timestamp", datetime.now(timezone.utc)),
        resolved_at=event_dict.get("resolved_at"),
        action_taken=event_dict.get("action_taken"),
        metadata=event_dict.get("metadata", {}),
    )


class RiskService:
    """Comprehensive risk management service."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize risk service with all components."""
        self.config = get_config()

        # Initialize risk limits
        custom_config = config or {}
        self.risk_limits = RiskLimits(
            max_position_percentage=custom_config.get(
                "max_position_percentage", Decimal("0.20")
            ),
            max_positions=custom_config.get("max_positions", 5),
            stop_loss_percentage=custom_config.get(
                "stop_loss_percentage", Decimal("0.02")
            ),
            take_profit_percentage=custom_config.get(
                "take_profit_percentage", Decimal("0.03")
            ),
            max_daily_loss_percentage=custom_config.get(
                "max_daily_loss_percentage", Decimal("0.05")
            ),
            max_correlation_threshold=custom_config.get(
                "max_correlation_threshold", 0.7
            ),
            emergency_stop_percentage=custom_config.get(
                "emergency_stop_percentage", Decimal("0.10")
            ),
            max_position_volatility=custom_config.get("max_position_volatility", 0.50),
        )

        # Initialize service components
        self.risk_manager = RiskManager(custom_config)
        self.position_sizer = PositionSizer(self.risk_limits)
        self.portfolio_monitor = PortfolioMonitor(self.risk_limits)
        self.alert_manager = AlertManager()
        self.database_manager = RiskDatabaseManager()
        self.alpaca_client = AlpacaRiskClient()
        self.risk_calculator = RiskCalculator(self.alpaca_client)

        # Service state
        self.is_initialized = False
        self.last_portfolio_update: Optional[datetime] = None
        self.monitoring_active = False

        # Background tasks
        self.monitoring_task: Optional[Task[Any]] = None
        self.alert_processing_task: Optional[Task[Any]] = None

    async def initialize(self) -> bool:
        """Initialize all service components."""
        try:
            logger.info("Initializing Risk Management Service...")

            # Initialize database
            await self.database_manager.initialize()

            # Validate Alpaca connection
            alpaca_connected = await self.alpaca_client.validate_connection()
            if not alpaca_connected:
                logger.warning("Alpaca connection failed - using fallback mode")

            # Start background tasks
            self.alert_processing_task = asyncio.create_task(
                self.alert_manager.process_alert_queue()
            )

            self.is_initialized = True
            logger.info("Risk Management Service initialized successfully")

            return True

        except Exception as e:
            logger.error(f"Error initializing risk service: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown risk service and cleanup resources."""
        try:
            logger.info("Shutting down Risk Management Service...")

            # Stop monitoring
            self.monitoring_active = False

            # Cancel background tasks
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()

            if self.alert_processing_task and not self.alert_processing_task.done():
                self.alert_processing_task.cancel()

            # Close database connections
            await self.database_manager.close()

            logger.info("Risk Management Service shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def validate_trade(
        self, order_request: OrderRequest, signal: Optional[TradeSignal] = None
    ) -> Tuple[bool, List[Dict], Optional[PositionSizing]]:
        """
        Comprehensive trade validation with position sizing recommendation.

        Args:
            order_request: The proposed trade order
            signal: Optional trade signal with confidence

        Returns:
            Tuple of (is_valid, risk_filters, recommended_sizing)
        """
        try:
            # Get current portfolio state
            portfolio = await self.alpaca_client.get_portfolio_state()
            if not portfolio:
                logger.error("Unable to retrieve portfolio state for trade validation")
                return False, [{"error": "Portfolio state unavailable"}], None

            # Validate against risk filters
            is_valid, filters = await self.risk_manager.validate_trade_request(
                order_request, portfolio, signal
            )

            # Calculate recommended position sizing
            sizing = None
            if order_request.side == OrderSide.BUY:
                try:
                    sizing = await self.position_sizer.calculate_position_size(
                        symbol=order_request.symbol,
                        current_price=order_request.price or Decimal("0"),
                        portfolio=portfolio,
                        signal=signal,
                    )
                except Exception as e:
                    logger.error(f"Error calculating position sizing: {e}")

            # Convert filters to dict format
            filter_dicts = []
            for f in filters:
                filter_dict = {
                    "passed": f.passed,
                    "filter_name": f.filter_name,
                    "reason": f.reason,
                    "value": f.value,
                    "limit": f.limit,
                    "severity": f.severity if f.severity else None,
                }
                filter_dicts.append(filter_dict)

            # Store validation result
            if not is_valid:
                event = RiskEvent(
                    event_type=RiskEventType.POSITION_SIZE_VIOLATION,
                    severity=RiskSeverity.MEDIUM,
                    symbol=order_request.symbol,
                    description=f"Trade validation failed: {[f.filter_name for f in filters if not f.passed]}",
                    resolved_at=None,
                    action_taken=None,
                    metadata={
                        "order_request": {
                            "symbol": order_request.symbol,
                            "side": order_request.side.value,
                            "quantity": order_request.quantity,
                            "price": (
                                str(order_request.price)
                                if order_request.price
                                else None
                            ),
                        },
                        "failed_filters": [
                            f.filter_name for f in filters if not f.passed
                        ],
                    },
                )
                await self.database_manager.store_risk_event(event)

            return is_valid, filter_dicts, sizing

        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, [{"error": str(e)}], None

    async def monitor_portfolio_risk(self) -> Tuple[PortfolioMetrics, List[RiskAlert]]:
        """Perform comprehensive portfolio risk monitoring."""
        try:
            # Get current portfolio state
            portfolio = await self.alpaca_client.get_portfolio_state(force_refresh=True)
            if not portfolio:
                logger.error("Unable to retrieve portfolio state for monitoring")
                # Return empty metrics on error
                empty_metrics = PortfolioMetrics(
                    timestamp=datetime.now(timezone.utc),
                    total_exposure=Decimal("0"),
                    cash_percentage=Decimal("0"),
                    position_count=0,
                    concentration_risk=0.0,
                    portfolio_beta=0.0,
                    portfolio_correlation=0.0,
                    value_at_risk_1d=Decimal("0"),
                    value_at_risk_5d=Decimal("0"),
                    expected_shortfall=Decimal("0"),
                    sharpe_ratio=0.0,
                    max_drawdown=Decimal("0"),
                    current_drawdown=Decimal("0"),
                    volatility=0.0,
                )
                return empty_metrics, []

            # Update risk manager state
            self.risk_manager.set_portfolio_snapshot(portfolio)

            # Calculate comprehensive metrics using real market data
            metrics = await self._calculate_enhanced_metrics(portfolio)

            # Monitor for risk violations
            violations = await self.risk_manager.check_risk_violations(portfolio)

            # Generate alerts from portfolio monitor
            _, monitor_alerts = await self.portfolio_monitor.monitor_portfolio(
                portfolio
            )

            # Create alerts from violations
            violation_alerts = []
            for violation in violations:
                alert = RiskAlert(
                    alert_type=violation.event_type,
                    severity=violation.severity,
                    symbol=violation.symbol,
                    title=f"Risk Violation: {violation.event_type.replace('_', ' ').title()}",
                    message=violation.description,
                    action_required=(
                        violation.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]
                    ),
                    metadata=violation.metadata,
                )
                violation_alerts.append(alert)

            all_alerts = monitor_alerts + violation_alerts

            # Store data in database
            await self.database_manager.store_portfolio_snapshot(portfolio)
            await self.database_manager.store_portfolio_metrics(metrics)

            # Store risk events and alerts
            for violation in violations:
                await self.database_manager.store_risk_event(violation)

            for alert in all_alerts:
                await self.database_manager.store_risk_alert(alert)

            # Send alerts
            if all_alerts:
                await self.alert_manager.send_bulk_alerts(all_alerts)

            self.last_portfolio_update = datetime.now(timezone.utc)

            return metrics, all_alerts

        except Exception as e:
            logger.error(f"Error monitoring portfolio risk: {e}")
            # Return empty metrics on error
            empty_metrics = PortfolioMetrics(
                timestamp=datetime.now(timezone.utc),
                total_exposure=Decimal("0"),
                cash_percentage=Decimal("0"),
                position_count=0,
                concentration_risk=0.0,
                portfolio_beta=0.0,
                portfolio_correlation=0.0,
                value_at_risk_1d=Decimal("0"),
                value_at_risk_5d=Decimal("0"),
                expected_shortfall=Decimal("0"),
                sharpe_ratio=0.0,
                max_drawdown=Decimal("0"),
                current_drawdown=Decimal("0"),
                volatility=0.0,
            )
            return empty_metrics, []

    async def update_position_stops(self) -> List[RiskEvent]:
        """Update trailing stops and check for stop loss triggers."""
        try:
            # Get current portfolio
            portfolio = await self.alpaca_client.get_portfolio_state()
            if not portfolio:
                return []

            # Get current market prices
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]
            current_prices = await self.alpaca_client.get_current_prices(symbols)

            # Update trailing stops
            stop_events = await self.risk_manager.update_trailing_stops(
                portfolio, current_prices
            )

            # Store events
            for event in stop_events:
                await self.database_manager.store_risk_event(event)

            # Check for scale-out opportunities
            scale_out_events = await self._check_scale_out_opportunities(
                portfolio, current_prices
            )
            stop_events.extend(scale_out_events)

            return stop_events

        except Exception as e:
            logger.error(f"Error updating position stops: {e}")
            return []

    async def calculate_optimal_position_size(
        self, symbol: str, signal: TradeSignal, current_price: Optional[Decimal] = None
    ) -> PositionSizing:
        """Calculate optimal position size with all risk considerations."""
        try:
            # Get current price if not provided
            if not current_price:
                prices = await self.alpaca_client.get_current_prices([symbol])
                current_price = prices.get(symbol, Decimal("0"))

            if current_price <= 0:
                raise ValueError(f"Invalid price for {symbol}: {current_price}")

            # Get portfolio state
            portfolio = await self.alpaca_client.get_portfolio_state()
            if not portfolio:
                raise ValueError("Unable to retrieve portfolio state")

            # Calculate position sizing
            sizing = await self.position_sizer.calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                portfolio=portfolio,
                signal=signal,
            )

            # Validate sizing against current portfolio
            violations = await self.position_sizer.validate_position_sizing(
                sizing, portfolio
            )
            if violations:
                logger.warning(f"Position sizing violations for {symbol}: {violations}")

            return sizing

        except Exception as e:
            logger.error(f"Error calculating optimal position size for {symbol}: {e}")
            # Return minimal safe sizing
            return PositionSizing(
                symbol=symbol,
                recommended_shares=1,
                recommended_value=current_price or Decimal("100"),
                position_percentage=Decimal("0.01"),
                confidence_adjustment=0.5,
                volatility_adjustment=0.5,
                sizing_method=PositionSizingMethod.FIXED_PERCENTAGE,
                max_loss_amount=Decimal("100"),
                risk_reward_ratio=1.5,
            )

    async def generate_daily_risk_report(
        self, date: Optional[datetime] = None
    ) -> DailyRiskReport:
        """Generate comprehensive daily risk report."""
        try:
            report_date = date or datetime.now(timezone.utc)
            start_of_day = report_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end_of_day = start_of_day + timedelta(days=1)

            # Get portfolio state
            portfolio = await self.alpaca_client.get_portfolio_state()
            if not portfolio:
                raise ValueError("Unable to retrieve portfolio state for report")

            # Calculate daily P&L
            daily_pnl = await self.alpaca_client.get_daily_pnl()

            # Get risk events for the day
            risk_events_dict = await self.database_manager.get_risk_events(
                start_date=start_of_day, end_date=end_of_day
            )

            # Convert dictionary events to RiskEvent objects
            risk_events = [
                _convert_risk_event_dict_to_object(event) for event in risk_events_dict
            ]

            # Calculate portfolio metrics
            metrics = await self._calculate_enhanced_metrics(portfolio)

            # Get position risks
            position_risks = await self.portfolio_monitor.calculate_position_risks(
                portfolio
            )

            # Calculate performance metrics
            daily_return = (
                float(daily_pnl / portfolio.total_equity)
                if portfolio.total_equity > 0
                else 0.0
            )

            # Get trade statistics
            trades = await self.alpaca_client.get_trade_history(days=1)
            total_trades = len(trades)
            winning_trades = len([t for t in trades if self._is_winning_trade(t)])

            # Check compliance violations
            compliance_violations = await self._check_compliance_violations(
                portfolio, metrics
            )

            # Create daily report
            report = DailyRiskReport(
                date=report_date,
                portfolio_value=portfolio.total_equity,
                daily_pnl=daily_pnl,
                daily_return=daily_return,
                max_drawdown=metrics.max_drawdown,
                current_drawdown=metrics.current_drawdown,
                volatility=metrics.volatility,
                sharpe_ratio=metrics.sharpe_ratio,
                var_1d=metrics.value_at_risk_1d,
                total_trades=total_trades,
                winning_trades=winning_trades,
                risk_events=risk_events,
                position_risks=position_risks,
                compliance_violations=compliance_violations,
            )

            # Store report
            await self.database_manager.store_daily_report(report)

            # Send report via notifications
            await self.alert_manager.send_daily_risk_report(
                portfolio_metrics=metrics.__dict__,
                risk_events=risk_events,  # Use RiskEvent objects
            )

            logger.info(f"Daily risk report generated for {report_date.date()}")
            return report

        except Exception as e:
            logger.error(f"Error generating daily risk report: {e}")
            raise

    async def activate_emergency_stop(
        self, reason: str, triggered_by: str = "system"
    ) -> Dict[str, Any]:
        """Activate emergency stop with full notification cascade."""
        try:
            # Activate in risk manager
            self.risk_manager.activate_emergency_stop(reason)

            # Create emergency event
            event = RiskEvent(
                event_type=RiskEventType.EMERGENCY_STOP,
                severity=RiskSeverity.CRITICAL,
                symbol=None,
                description=f"Emergency stop activated: {reason}",
                resolved_at=None,
                action_taken="halt_all_trading",
                metadata={
                    "triggered_by": triggered_by,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Store event
            await self.database_manager.store_risk_event(event)

            # Send emergency alert
            await self.alert_manager.send_emergency_alert(event)

            logger.critical(f"Emergency stop activated: {reason}")

            return {
                "status": "activated",
                "reason": reason,
                "triggered_by": triggered_by,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error activating emergency stop: {e}")
            raise

    async def check_position_management_rules(
        self, portfolio: PortfolioState
    ) -> List[Dict]:
        """Check all position management rules and return recommended actions."""
        try:
            recommendations = []

            # Get current market prices
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]
            current_prices = await self.alpaca_client.get_current_prices(symbols)

            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                symbol = position.symbol
                current_price = current_prices.get(symbol, position.current_price)

                # Check for scale-out opportunities
                should_scale, scale_percentage = (
                    await self.risk_manager.should_scale_out_position(
                        position, current_price
                    )
                )

                if should_scale:
                    recommendations.append(
                        {
                            "action": "scale_out",
                            "symbol": symbol,
                            "percentage": scale_percentage,
                            "reason": f"Position up {((current_price - position.entry_price) / position.entry_price * 100):.1f}%",
                            "current_price": str(current_price),
                            "entry_price": str(position.entry_price),
                        }
                    )

                # Check position size relative to limits
                position_pct = (
                    abs(position.market_value) / portfolio.total_equity
                    if portfolio.total_equity > 0
                    else Decimal("0")
                )
                if position_pct > self.risk_limits.max_position_percentage:
                    recommendations.append(
                        {
                            "action": "reduce_position",
                            "symbol": symbol,
                            "current_percentage": float(position_pct),
                            "max_percentage": float(
                                self.risk_limits.max_position_percentage
                            ),
                            "reason": "Position size exceeds maximum allocation",
                        }
                    )

                # Check for stop loss violations
                loss_pct = (
                    position.current_price - position.entry_price
                ) / position.entry_price
                if position.quantity > 0 and loss_pct <= -float(
                    self.risk_limits.stop_loss_percentage
                ):
                    recommendations.append(
                        {
                            "action": "stop_loss",
                            "symbol": symbol,
                            "loss_percentage": float(loss_pct),
                            "reason": "Stop loss threshold reached",
                            "urgency": "high",
                        }
                    )

            return recommendations

        except Exception as e:
            logger.error(f"Error checking position management rules: {e}")
            return []

    async def calculate_portfolio_risk_budget(self, portfolio: PortfolioState) -> Dict:
        """Calculate risk budget allocation across positions."""
        try:
            # Calculate component VaR for each position
            component_vars = await self.risk_calculator.calculate_component_var(
                portfolio
            )

            # Calculate total VaR
            total_var, _ = await self.risk_calculator.calculate_portfolio_var(portfolio)

            # Calculate risk contributions
            risk_budget = {}
            for symbol, component_var in component_vars.items():
                if total_var > 0:
                    risk_contribution = float(component_var / total_var)
                    risk_budget[symbol] = {
                        "component_var": str(component_var),
                        "risk_contribution": risk_contribution,
                        "risk_contribution_pct": f"{risk_contribution * 100:.1f}%",
                    }

            # Add portfolio summary
            risk_budget["portfolio_summary"] = {
                "total_var": str(total_var),
                "number_of_positions": len(
                    [p for p in portfolio.positions if p.quantity != 0]
                ),
                "risk_budget_used": sum(
                    self._safe_float_convert(rb.get("risk_contribution", 0))
                    for rb in risk_budget.values()
                    if isinstance(rb, dict) and rb.get("risk_contribution") is not None
                ),
            }

            return risk_budget
        except Exception as e:
            logger.error(f"Error calculating risk budget: {e}")
            return {}

    def _safe_float_convert(self, value: Any) -> float:
        """Safely convert a value to float."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0

    async def perform_stress_test(self, scenarios: Optional[List[Dict]] = None) -> Dict:
        """Perform comprehensive stress testing."""
        try:
            # Get current portfolio
            portfolio = await self.alpaca_client.get_portfolio_state()
            if not portfolio:
                return {"error": "Portfolio state unavailable"}

            # Default stress scenarios if none provided
            if not scenarios:
                scenarios = self._get_default_stress_scenarios()

            # Perform stress testing
            stress_results = await self.risk_calculator.stress_test_portfolio(
                portfolio, scenarios
            )

            # Add current portfolio context
            stress_results["portfolio_context"] = {
                "total_equity": str(portfolio.total_equity),
                "position_count": len(
                    [p for p in portfolio.positions if p.quantity != 0]
                ),
                "cash_percentage": (
                    float(portfolio.cash / portfolio.total_equity)
                    if portfolio.total_equity > 0
                    else 0.0
                ),
                "test_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Store stress test results
            stress_event = RiskEvent(
                event_type=RiskEventType.VOLATILITY_SPIKE,  # Using as general risk event
                severity=RiskSeverity.LOW,
                symbol=None,
                description="Stress test performed",
                resolved_at=None,
                action_taken=None,
                metadata={
                    "stress_test_results": stress_results,
                    "scenarios_tested": len(scenarios),
                },
            )
            await self.database_manager.store_risk_event(stress_event)

            return stress_results

        except Exception as e:
            logger.error(f"Error performing stress test: {e}")
            return {"error": str(e)}

    async def get_risk_dashboard_data(self) -> Dict:
        """Get comprehensive risk dashboard data."""
        try:
            # Get current portfolio
            portfolio = await self.alpaca_client.get_portfolio_state()
            if not portfolio:
                return {"error": "Portfolio state unavailable"}

            # Calculate metrics
            metrics = await self._calculate_enhanced_metrics(portfolio)

            # Get recent alerts
            recent_alerts = await self.database_manager.get_unacknowledged_alerts(
                limit=20
            )

            # Get risk statistics
            risk_stats = await self.database_manager.get_risk_statistics(days=7)

            # Get position risks
            position_risks = await self.portfolio_monitor.calculate_position_risks(
                portfolio
            )

            # Get market conditions
            market_conditions = await self.alpaca_client.check_market_conditions()

            # Portfolio health check
            health_status = await self.portfolio_monitor.check_portfolio_health(
                portfolio
            )

            # Risk warnings
            risk_warnings = await self.portfolio_monitor.generate_risk_warnings(
                portfolio
            )

            dashboard_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio_summary": {
                    "total_equity": str(portfolio.total_equity),
                    "cash": str(portfolio.cash),
                    "buying_power": str(portfolio.buying_power),
                    "position_count": len(
                        [p for p in portfolio.positions if p.quantity != 0]
                    ),
                    "day_trades_used": portfolio.day_trades_count,
                },
                "risk_metrics": {
                    "total_exposure": str(metrics.total_exposure),
                    "concentration_risk": metrics.concentration_risk,
                    "portfolio_beta": metrics.portfolio_beta,
                    "volatility": metrics.volatility,
                    "var_1d": str(metrics.value_at_risk_1d),
                    "current_drawdown": str(metrics.current_drawdown),
                    "sharpe_ratio": metrics.sharpe_ratio,
                },
                "health_status": health_status,
                "risk_warnings": risk_warnings,
                "recent_alerts": [
                    {
                        "id": str(alert.get("id", "")),
                        "alert_type": alert.get("alert_type", ""),
                        "severity": alert.get("severity", ""),
                        "title": alert.get("title", ""),
                        "timestamp": alert.get("timestamp", ""),
                        "action_required": alert.get("action_required", False),
                    }
                    for alert in recent_alerts[:5]  # Last 5 alerts
                ],
                "position_risks": [
                    {
                        "symbol": risk.symbol,
                        "portfolio_percentage": str(risk.portfolio_percentage),
                        "risk_score": risk.risk_score,
                        "volatility": risk.volatility,
                        "sector": risk.sector,
                    }
                    for risk in position_risks
                ],
                "market_conditions": market_conditions,
                "risk_limits": {
                    "max_position_percentage": str(
                        self.risk_limits.max_position_percentage
                    ),
                    "max_positions": self.risk_limits.max_positions,
                    "stop_loss_percentage": str(self.risk_limits.stop_loss_percentage),
                    "max_daily_loss_percentage": str(
                        self.risk_limits.max_daily_loss_percentage
                    ),
                    "emergency_stop_active": self.risk_manager.emergency_stop_active,
                },
                "risk_statistics": risk_stats,
                "system_status": {
                    "monitoring_active": self.monitoring_active,
                    "last_update": (
                        self.last_portfolio_update.isoformat()
                        if self.last_portfolio_update
                        else None
                    ),
                    "alert_queue_size": self.alert_manager.alert_queue.qsize(),
                },
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}

    async def start_monitoring(self, interval_seconds: int = 60) -> Dict[str, Any]:
        """Start continuous portfolio monitoring."""
        try:
            if self.monitoring_active:
                logger.warning("Monitoring already active")
                return {
                    "status": "already_active",
                    "message": "Monitoring already active",
                }

            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(
                self._monitoring_loop(interval_seconds)
            )

            logger.info(f"Portfolio monitoring started (interval: {interval_seconds}s)")

            return {
                "status": "started",
                "interval_seconds": interval_seconds,
                "message": f"Started portfolio monitoring (interval: {interval_seconds}s)",
            }

        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            raise

    async def stop_monitoring(self) -> None:
        """Stop continuous portfolio monitoring."""
        try:
            self.monitoring_active = False

            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()

            logger.info("Portfolio monitoring stopped")

        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")

    async def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Perform risk monitoring
                await self.monitor_portfolio_risk()

                # Update trailing stops
                await self.update_position_stops()

                # Sleep until next monitoring cycle
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(min(interval_seconds, 60))  # Fallback sleep

    async def _calculate_enhanced_metrics(
        self, portfolio: PortfolioState
    ) -> PortfolioMetrics:
        """Calculate enhanced portfolio metrics using real market data."""
        try:
            # Use risk calculator for advanced metrics
            portfolio_vol = await self.risk_calculator.calculate_portfolio_volatility(
                portfolio
            )
            portfolio_beta = await self.risk_calculator.calculate_portfolio_beta(
                portfolio
            )

            # Calculate VaR using multiple methods
            var_1d, expected_shortfall = (
                await self.risk_calculator.calculate_portfolio_var(
                    portfolio, confidence_level=0.95, method="historical"
                )
            )
            var_5d, _ = await self.risk_calculator.calculate_portfolio_var(
                portfolio, confidence_level=0.95, method="historical", holding_period=5
            )

            # Get concentration metrics
            concentration_metrics = (
                await self.risk_calculator.calculate_concentration_metrics(portfolio)
            )

            # Calculate correlation
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]
            if len(symbols) > 1:
                corr_matrix = await self.risk_calculator.calculate_correlation_matrix(
                    symbols
                )
                avg_correlation = float(
                    corr_matrix.values[
                        np.triu_indices_from(corr_matrix.values, k=1)
                    ].mean()
                )
            else:
                avg_correlation = 0.0

            # Use portfolio monitor for remaining metrics
            basic_metrics = await self.portfolio_monitor.calculate_detailed_metrics(
                portfolio
            )

            # Combine enhanced metrics
            enhanced_metrics = PortfolioMetrics(
                timestamp=datetime.now(timezone.utc),
                total_exposure=portfolio.total_market_value,
                cash_percentage=(
                    portfolio.cash / portfolio.total_equity
                    if portfolio.total_equity > 0
                    else Decimal("1")
                ),
                position_count=len([p for p in portfolio.positions if p.quantity != 0]),
                concentration_risk=concentration_metrics.get("herfindahl_index", 0.0),
                portfolio_beta=portfolio_beta,
                portfolio_correlation=avg_correlation,
                value_at_risk_1d=var_1d,
                value_at_risk_5d=var_5d,
                expected_shortfall=expected_shortfall,
                sharpe_ratio=basic_metrics.sharpe_ratio,  # From portfolio monitor
                max_drawdown=basic_metrics.max_drawdown,
                current_drawdown=basic_metrics.current_drawdown,
                volatility=portfolio_vol,
            )

            return enhanced_metrics

        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {e}")
            # Return basic metrics as fallback
            return await self.portfolio_monitor.calculate_detailed_metrics(portfolio)

    async def _check_scale_out_opportunities(
        self, portfolio: PortfolioState, current_prices: Dict[str, Decimal]
    ) -> List[RiskEvent]:
        """Check for scale-out opportunities in winning positions."""
        events = []

        for position in portfolio.positions:
            if position.quantity <= 0:
                continue

            symbol = position.symbol
            current_price = current_prices.get(symbol, position.current_price)

            should_scale, scale_pct = await self.risk_manager.should_scale_out_position(
                position, current_price
            )

            if should_scale:
                profit_pct = (
                    current_price - position.entry_price
                ) / position.entry_price
                event = RiskEvent(
                    event_type=RiskEventType.TAKE_PROFIT_TRIGGERED,
                    severity=RiskSeverity.MEDIUM,
                    symbol=symbol,
                    description=f"Scale out opportunity: {symbol} up {profit_pct:.1%}, recommend {scale_pct:.0%} reduction",
                    resolved_at=None,
                    action_taken=f"scale_out_{scale_pct:.0%}",
                    metadata={
                        "current_price": str(current_price),
                        "entry_price": str(position.entry_price),
                        "profit_percentage": float(profit_pct),
                        "scale_out_percentage": scale_pct,
                        "position_size": position.quantity,
                    },
                )
                events.append(event)

        return events

    def _is_winning_trade(self, trade: Dict) -> bool:
        """Determine if a trade was winning based on current market conditions."""
        try:
            symbol = trade.get("symbol")
            side = trade.get("side", "").upper()
            trade_price = Decimal(str(trade.get("price", 0)))

            if not symbol or trade_price <= 0:
                return False

            # Get current market price
            import asyncio

            try:
                # Try to get current price from data collector
                current_prices = asyncio.run(
                    self.alpaca_client.get_current_prices([symbol])
                )
                current_price = current_prices.get(symbol)

                if not current_price or current_price <= 0:
                    return False

                # For BUY trades: winning if current price > trade price
                if side == "BUY":
                    return current_price > trade_price

                # For SELL trades: winning if we sold above our entry price
                # Try to get position info to compare against entry price
                elif side == "SELL":
                    try:
                        portfolio = asyncio.run(
                            self.alpaca_client.get_portfolio_state()
                        )
                        if portfolio and portfolio.positions:
                            # Look for matching position to get entry price
                            for position in portfolio.positions:
                                if position.symbol == symbol and position.quantity > 0:
                                    return trade_price > position.entry_price

                        # If no position found, assume SELL was profitable
                        # (conservative assumption since sells are usually for profit/stop)
                        return True
                    except Exception:
                        # Fallback: assume profitable sell
                        return True

            except Exception as e:
                # Fallback to basic heuristic if data access fails
                logger.warning(f"Unable to get current price for {symbol}: {e}")
                # Simple heuristic: trades above $10 are more likely to be winning
                return trade_price > Decimal("10")

            return False

        except Exception as e:
            logger.warning(f"Error determining if trade is winning: {e}")
            return False

    async def _check_compliance_violations(
        self, portfolio: PortfolioState, metrics: PortfolioMetrics
    ) -> List[str]:
        """Check for compliance violations."""
        violations = []

        try:
            # Position limits
            if metrics.position_count > self.risk_limits.max_positions:
                violations.append(
                    f"Position count {metrics.position_count} exceeds limit {self.risk_limits.max_positions}"
                )

            # Concentration limits
            if metrics.concentration_risk > 0.8:
                violations.append(
                    f"Portfolio concentration risk {metrics.concentration_risk:.2f} exceeds safe threshold 0.8"
                )

            # Drawdown limits
            if metrics.current_drawdown > self.risk_limits.max_drawdown_percentage:
                violations.append(
                    f"Current drawdown {metrics.current_drawdown:.2%} exceeds limit {self.risk_limits.max_drawdown_percentage:.2%}"
                )

            # Correlation limits
            if (
                metrics.portfolio_correlation
                > self.risk_limits.max_correlation_threshold
            ):
                violations.append(
                    f"Portfolio correlation {metrics.portfolio_correlation:.2f} exceeds limit {self.risk_limits.max_correlation_threshold}"
                )

            # Individual position limits
            for position in portfolio.positions:
                if position.quantity != 0:
                    position_pct = (
                        abs(position.market_value) / portfolio.total_equity
                        if portfolio.total_equity > 0
                        else Decimal("0")
                    )
                    if position_pct > self.risk_limits.max_position_percentage:
                        violations.append(
                            f"Position {position.symbol} size {position_pct:.2%} exceeds limit {self.risk_limits.max_position_percentage:.2%}"
                        )

            return violations

        except Exception as e:
            logger.error(f"Error checking compliance violations: {e}")
            return ["Error checking compliance violations"]

    def _get_default_stress_scenarios(self) -> List[Dict]:
        """Get default stress testing scenarios."""
        return [
            {
                "name": "market_crash_10pct",
                "description": "10% market-wide decline",
                "shocks": {"*": -0.10},  # Apply to all positions
            },
            {
                "name": "market_crash_20pct",
                "description": "20% market-wide decline",
                "shocks": {"*": -0.20},
            },
            {
                "name": "tech_sector_crash",
                "description": "30% decline in tech stocks",
                "shocks": {
                    "AAPL": -0.30,
                    "MSFT": -0.30,
                    "GOOGL": -0.30,
                    "AMZN": -0.25,
                    "TSLA": -0.35,
                    "NVDA": -0.30,
                },
            },
            {
                "name": "financial_crisis",
                "description": "Financial sector stress",
                "shocks": {
                    "JPM": -0.25,
                    "BAC": -0.30,
                    "WFC": -0.28,
                    "C": -0.32,
                    "GS": -0.35,
                },
            },
            {
                "name": "volatility_spike",
                "description": "High volatility environment",
                "shocks": {"*": 0.0},  # No price shock, but increased volatility
                "volatility_multiplier": 2.0,
            },
            {
                "name": "interest_rate_shock",
                "description": "200bps interest rate increase",
                "shocks": {
                    "SPY": -0.08,
                    "QQQ": -0.12,
                    "IWM": -0.10,
                    # Growth stocks more sensitive
                    "AAPL": -0.15,
                    "MSFT": -0.12,
                    "GOOGL": -0.18,
                },
            },
            {
                "name": "black_swan",
                "description": "Extreme tail risk event",
                "shocks": {"*": -0.35},  # 35% decline across all positions
            },
        ]

    async def reset_daily_risk_state(self) -> None:
        """Reset daily risk state (called at market open)."""
        try:
            # Reset risk manager counters
            self.risk_manager.reset_daily_counters()

            # Reset portfolio monitor
            self.portfolio_monitor.reset_monitoring_state()

            # Clear emergency stop if it was activated
            if self.risk_manager.emergency_stop_active:
                logger.warning(
                    "Emergency stop was active - manual review required before deactivation"
                )

            logger.info("Daily risk state reset completed")

        except Exception as e:
            logger.error(f"Error resetting daily risk state: {e}")

    def get_service_status(self) -> Dict:
        """Get comprehensive service status."""
        return {
            "initialized": self.is_initialized,
            "monitoring_active": self.monitoring_active,
            "emergency_stop_active": (
                self.risk_manager.emergency_stop_active if self.risk_manager else False
            ),
            "last_portfolio_update": (
                self.last_portfolio_update.isoformat()
                if self.last_portfolio_update
                else None
            ),
            "components": {
                "risk_manager": self.risk_manager is not None,
                "position_sizer": self.position_sizer is not None,
                "portfolio_monitor": self.portfolio_monitor is not None,
                "alert_manager": self.alert_manager is not None,
                "database_manager": self.database_manager is not None,
                "alpaca_client": self.alpaca_client is not None,
                "risk_calculator": self.risk_calculator is not None,
            },
            "risk_limits": {
                "max_position_percentage": str(
                    self.risk_limits.max_position_percentage
                ),
                "max_positions": self.risk_limits.max_positions,
                "stop_loss_percentage": str(self.risk_limits.stop_loss_percentage),
                "take_profit_percentage": str(self.risk_limits.take_profit_percentage),
                "max_daily_loss_percentage": str(
                    self.risk_limits.max_daily_loss_percentage
                ),
                "emergency_stop_percentage": str(
                    self.risk_limits.emergency_stop_percentage
                ),
            },
        }
