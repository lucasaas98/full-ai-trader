"""
Portfolio Monitor

This module provides real-time portfolio monitoring capabilities including:
- Continuous portfolio state tracking
- Risk metrics calculation and monitoring
- Position-level risk assessment
- Real-time alerts and notifications
- Performance tracking and analytics
"""

# TODO: Comprehensive testing is required for all enhanced features:
#
# 1. Unit Tests Needed:
#    - test_get_position_volatility() - Verify sector-based volatility calculations
#    - test_get_position_beta() - Verify sector-based beta estimates
#    - test_calculate_expected_return() - Verify CAPM and sector premium calculations
#    - test_estimate_correlation() - Verify sector correlation matrix and special cases
#    - test_get_position_sector() - Verify comprehensive sector mapping
#    - test_is_tech_stock() - Verify enhanced tech stock detection
#    - test_similar_market_cap() - Verify market cap similarity logic
#
# 2. Integration Tests Needed:
#    - test_portfolio_risk_calculations() - End-to-end risk metric calculations
#    - test_correlation_caching() - Verify caching behavior and TTL
#    - test_volatility_caching() - Verify caching behavior and TTL
#    - test_sector_risk_exposure() - Verify sector-based risk aggregation
#
# 3. Performance Tests Needed:
#    - test_large_portfolio_performance() - Performance with 100+ positions
#    - test_cache_effectiveness() - Memory usage and cache hit rates
#    - test_correlation_matrix_performance() - N^2 correlation calculations
#
# 4. Market Data Integration Tests (Future):
#    - test_real_volatility_calculation() - Historical price data integration
#    - test_real_beta_calculation() - Market index correlation calculation
#    - test_real_correlation_calculation() - Rolling correlation windows
#    - test_risk_free_rate_integration() - Fed data integration
#    - test_sector_data_integration() - Financial data provider integration
#
# 5. Edge Case Tests Needed:
#    - test_unknown_symbols() - Behavior with unrecognized symbols
#    - test_market_regime_changes() - Correlation adjustments during crises
#    - test_extreme_volatility() - Bounds checking for volatility calculations
#    - test_negative_correlations() - Handling of negative correlation scenarios
#
# 6. Enhanced Integration Tests Needed:
#    - test_enhanced_portfolio_monitor.py - Run comprehensive test suite for new features
#    - test_screener_data_integration() - Verify FinViz data loading and sector mapping
#    - test_historical_price_correlation() - Verify real price correlation calculations
#    - test_fallback_mechanisms() - Verify graceful degradation when data unavailable
#    - test_cache_invalidation() - Verify TTL and cache refresh behavior
#    - test_data_validation() - Verify handling of corrupted or invalid data files
#

import asyncio
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import redis.asyncio as redis

from shared.config import get_config
from shared.models import (
    PortfolioMetrics,
    PortfolioState,
    Position,
    PositionRisk,
    RiskAlert,
    RiskEventType,
    RiskLimits,
    RiskSeverity,
)

logger = logging.getLogger(__name__)


class PortfolioMonitor:
    """Real-time portfolio monitoring and risk assessment."""

    def __init__(self, risk_limits: RiskLimits):
        """Initialize portfolio monitor."""
        self.risk_limits = risk_limits
        self.config = get_config()

        # Historical data for calculations
        self.portfolio_history: List[Tuple[datetime, PortfolioState]] = []
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self.max_history_days = 252  # 1 year of trading days

        # Risk tracking
        self.risk_alerts: List[RiskAlert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=5)  # Prevent spam

        # Performance tracking
        self.daily_returns: List[float] = []
        self.peak_portfolio_value = Decimal("0")
        self.portfolio_start_value = Decimal("0")

        # Cache attributes
        self._correlation_cache: Dict[Tuple[str, str], Tuple[float, datetime]] = {}
        self._volatility_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)

        # Redis integration for screener updates
        self._redis = None
        self._pubsub = None
        self._screener_subscriber_task = None
        self._screener_callbacks: List[Callable[[Dict], Awaitable[None]]] = []
        self._previous_portfolio = None

        # Initialize Redis connection in background
        asyncio.create_task(self._initialize_redis())

        # Data paths for accessing stored market data
        self._data_path = Path(self.config.data.parquet_path)
        self._sector_cache: Dict[str, Tuple[str, datetime]] = {}

    async def monitor_portfolio(
        self, portfolio: PortfolioState
    ) -> Tuple[PortfolioMetrics, List[RiskAlert]]:
        """
        Monitor portfolio and generate metrics and alerts.

        Args:
            portfolio: Current portfolio state

        Returns:
            Tuple of (portfolio_metrics, risk_alerts)
        """
        try:
            # Update historical data
            await self._update_portfolio_history(portfolio)

            # Calculate comprehensive metrics
            metrics = await self.calculate_detailed_metrics(portfolio)

            # Check for risk alerts
            alerts = await self._check_risk_alerts(portfolio, metrics)

            # Update peak values for drawdown calculation
            self._update_peak_values(portfolio)

            logger.debug(
                f"Portfolio monitoring complete. Metrics: {metrics.position_count} positions, "
                f"exposure: {metrics.total_exposure}, alerts: {len(alerts)}"
            )

            return metrics, alerts

        except Exception as e:
            logger.error(f"Error monitoring portfolio: {e}")
            # Return basic metrics on error
            basic_metrics = PortfolioMetrics(
                total_exposure=portfolio.total_market_value,
                cash_percentage=(
                    portfolio.cash / portfolio.total_equity
                    if portfolio.total_equity > 0
                    else Decimal("1")
                ),
                position_count=len([p for p in portfolio.positions if p.quantity != 0]),
                concentration_risk=0.0,
                portfolio_beta=1.0,
                portfolio_correlation=0.0,
                value_at_risk_1d=Decimal("0"),
                value_at_risk_5d=Decimal("0"),
                expected_shortfall=Decimal("0"),
                sharpe_ratio=0.0,
                max_drawdown=Decimal("0"),
                current_drawdown=Decimal("0"),
                volatility=0.15,
            )
            return basic_metrics, []

    async def calculate_detailed_metrics(
        self, portfolio: PortfolioState
    ) -> PortfolioMetrics:
        """Calculate detailed portfolio risk metrics."""

        # Basic metrics
        total_exposure = portfolio.total_market_value
        cash_percentage = (
            portfolio.cash / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("1")
        )
        position_count = len([p for p in portfolio.positions if p.quantity != 0])

        # Advanced risk metrics
        concentration_risk = await self._calculate_concentration_risk(portfolio)
        portfolio_beta = await self._calculate_portfolio_beta(portfolio)
        avg_correlation = await self._calculate_average_correlation(portfolio)

        # VaR calculations
        var_1d, var_5d, expected_shortfall = await self._calculate_var_metrics(
            portfolio
        )

        # Performance metrics
        sharpe_ratio = await self._calculate_sharpe_ratio()
        max_drawdown, current_drawdown = await self._calculate_drawdown_metrics(
            portfolio
        )
        volatility = await self._calculate_portfolio_volatility(portfolio)

        return PortfolioMetrics(
            total_exposure=total_exposure,
            cash_percentage=cash_percentage,
            position_count=position_count,
            concentration_risk=concentration_risk,
            portfolio_beta=portfolio_beta,
            portfolio_correlation=avg_correlation,
            value_at_risk_1d=var_1d,
            value_at_risk_5d=var_5d,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
        )

    async def calculate_position_risks(
        self, portfolio: PortfolioState
    ) -> List[PositionRisk]:
        """Calculate risk metrics for individual positions."""
        position_risks = []

        for position in portfolio.positions:
            if position.quantity == 0:
                continue

            try:
                risk = await self._calculate_individual_position_risk(
                    position, portfolio
                )
                position_risks.append(risk)
            except Exception as e:
                logger.error(
                    f"Error calculating risk for position {position.symbol}: {e}"
                )

        return position_risks

    async def _calculate_individual_position_risk(
        self, position: Position, portfolio: PortfolioState
    ) -> PositionRisk:
        """Calculate risk metrics for a single position."""

        # Basic metrics
        position_size = abs(position.market_value)
        portfolio_percentage = (
            position_size / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("0")
        )

        # Get volatility and beta
        volatility = await self._get_position_volatility(position.symbol)
        beta = await self._get_position_beta(position.symbol)

        # Calculate VaR for position
        var_1d = (
            position_size
            * Decimal(str(volatility))
            * Decimal("1.645")
            / Decimal(str(np.sqrt(252)))
        )

        # Expected return calculation based on sector and market conditions
        expected_return = await self._calculate_expected_return(
            position.symbol, beta, volatility
        )

        # Sharpe ratio calculation
        risk_free_rate = (
            0.0001  # 0.01% daily risk-free rate (TODO: fetch from Fed data)
        )
        sharpe_ratio = (
            (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        )

        # Correlation with portfolio
        correlation_with_portfolio = await self._get_portfolio_correlation(
            position.symbol, portfolio
        )

        # Sector classification from FinViz data
        sector = await self._get_position_sector(position.symbol)

        # Overall risk score (0-10 scale)
        risk_score = await self._calculate_position_risk_score(
            volatility, float(portfolio_percentage), correlation_with_portfolio, beta
        )

        return PositionRisk(
            symbol=position.symbol,
            position_size=position_size,
            portfolio_percentage=float(portfolio_percentage),
            volatility=volatility,
            beta=beta,
            var_1d=var_1d,
            expected_return=expected_return,
            sharpe_ratio=sharpe_ratio,
            correlation_with_portfolio=correlation_with_portfolio,
            sector=sector,
            risk_score=risk_score,
        )

    async def _check_risk_alerts(
        self, portfolio: PortfolioState, metrics: PortfolioMetrics
    ) -> List[RiskAlert]:
        """Check for conditions that should trigger risk alerts."""
        alerts = []

        # Drawdown alerts
        if (
            metrics.current_drawdown
            > self.risk_limits.max_drawdown_percentage * Decimal("0.8")
        ):  # 80% of limit
            alert = await self._create_alert(
                RiskEventType.PORTFOLIO_DRAWDOWN,
                RiskSeverity.HIGH,
                "Portfolio Drawdown Warning",
                f"Current drawdown {metrics.current_drawdown:.2%} approaching limit {self.risk_limits.max_drawdown_percentage:.2%}",
                metadata={"current_drawdown": str(metrics.current_drawdown)},
            )
            if alert:
                alerts.append(alert)

        # Concentration risk alerts
        if metrics.concentration_risk > 0.7:  # High concentration
            alert = await self._create_alert(
                RiskEventType.POSITION_SIZE_VIOLATION,
                RiskSeverity.MEDIUM,
                "High Portfolio Concentration",
                f"Portfolio concentration risk is high: {metrics.concentration_risk:.2f}",
                metadata={"concentration_risk": metrics.concentration_risk},
            )
            if alert:
                alerts.append(alert)

        # VaR alerts
        var_percentage = (
            metrics.value_at_risk_1d / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("0")
        )
        if var_percentage > Decimal("0.05"):  # 5% VaR threshold
            alert = await self._create_alert(
                RiskEventType.VOLATILITY_SPIKE,
                RiskSeverity.MEDIUM,
                "High Value at Risk",
                f"1-day VaR is {var_percentage:.2%} of portfolio",
                metadata={"var_1d": str(metrics.value_at_risk_1d)},
            )
            if alert:
                alerts.append(alert)

        # Position count alerts
        if metrics.position_count >= self.risk_limits.max_positions:
            alert = await self._create_alert(
                RiskEventType.POSITION_LIMIT_REACHED,
                RiskSeverity.MEDIUM,
                "Position Limit Reached",
                f"Portfolio has {metrics.position_count} positions at maximum limit",
                metadata={"position_count": metrics.position_count},
            )
            if alert:
                alerts.append(alert)

        # Check individual position alerts
        position_alerts = await self._check_position_alerts(portfolio)
        alerts.extend(position_alerts)

        return alerts

    async def _check_position_alerts(
        self, portfolio: PortfolioState
    ) -> List[RiskAlert]:
        """Check for position-specific alerts."""
        alerts = []

        for position in portfolio.positions:
            if position.quantity == 0:
                continue

            # Large position alert
            position_pct = (
                abs(position.market_value) / portfolio.total_equity
                if portfolio.total_equity > 0
                else Decimal("0")
            )
            if position_pct > self.risk_limits.max_position_percentage * Decimal(
                "0.9"
            ):  # 90% of limit
                alert = await self._create_alert(
                    RiskEventType.POSITION_SIZE_VIOLATION,
                    RiskSeverity.MEDIUM,
                    "Large Position Warning",
                    f"Position {position.symbol} is {position_pct:.2%} of portfolio",
                    symbol=position.symbol,
                    metadata={"position_percentage": str(position_pct)},
                )
                if alert:
                    alerts.append(alert)

            # Unrealized loss alert
            loss_pct = (
                position.unrealized_pnl / abs(position.cost_basis)
                if position.cost_basis != 0
                else Decimal("0")
            )
            if loss_pct <= -self.risk_limits.stop_loss_percentage * Decimal(
                "0.8"
            ):  # 80% of stop loss
                alert = await self._create_alert(
                    RiskEventType.STOP_LOSS_TRIGGERED,
                    RiskSeverity.HIGH,
                    "Position Loss Warning",
                    f"Position {position.symbol} down {loss_pct:.2%}",
                    symbol=position.symbol,
                    metadata={"unrealized_loss": str(loss_pct)},
                )
                if alert:
                    alerts.append(alert)

        return alerts

    async def _create_alert(
        self,
        event_type: RiskEventType,
        severity: RiskSeverity,
        title: str,
        message: str,
        symbol: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[RiskAlert]:
        """Create a risk alert if not in cooldown period."""

        alert_key = f"{event_type}_{symbol or 'portfolio'}"

        # Check cooldown
        if alert_key in self.last_alert_times:
            time_since_last = (
                datetime.now(timezone.utc) - self.last_alert_times[alert_key]
            )
            if time_since_last < self.alert_cooldown:
                return None

        # Create alert
        alert = RiskAlert(
            alert_type=event_type,
            severity=severity,
            symbol=symbol,
            title=title,
            message=message,
            action_required=(severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]),
            metadata=metadata or {},
        )

        # Update cooldown tracker
        self.last_alert_times[alert_key] = datetime.now(timezone.utc)

        return alert

    async def _update_portfolio_history(self, portfolio: PortfolioState) -> None:
        """Update portfolio historical data."""

        # Add current snapshot
        self.portfolio_history.append((datetime.now(timezone.utc), portfolio))

        # Trim history to max days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_history_days)
        self.portfolio_history = [
            (timestamp, state)
            for timestamp, state in self.portfolio_history
            if timestamp > cutoff_date
        ]

        # Update daily returns if we have previous day data
        await self._update_daily_returns(portfolio)

    async def _update_daily_returns(self, portfolio: PortfolioState) -> None:
        """Update daily returns calculation."""

        if len(self.portfolio_history) < 2:
            return

        # Get yesterday's portfolio value
        # Get yesterday's date for comparison
        previous_value = None

        for timestamp, state in reversed(self.portfolio_history[:-1]):
            if timestamp.date() < datetime.now(timezone.utc).date():
                previous_value = state.total_equity
                break

        if previous_value and previous_value > 0:
            daily_return = float(
                (portfolio.total_equity - previous_value) / previous_value
            )
            self.daily_returns.append(daily_return)

            # Keep only recent returns
            if len(self.daily_returns) > 252:  # 1 year
                self.daily_returns = self.daily_returns[-252:]

    async def _calculate_concentration_risk(self, portfolio: PortfolioState) -> float:
        """Calculate portfolio concentration risk using Herfindahl-Hirschman Index."""

        if not portfolio.positions or portfolio.total_equity <= 0:
            return 0.0

        # Calculate position weights
        weights = []
        for position in portfolio.positions:
            if position.quantity != 0:
                weight = abs(position.market_value) / portfolio.total_equity
                weights.append(float(weight))

        if not weights:
            return 0.0

        # Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in weights)

        # Normalize to 0-1 scale
        return min(1.0, hhi)

    async def _calculate_portfolio_beta(self, portfolio: PortfolioState) -> float:
        """Calculate portfolio beta relative to market."""

        try:
            if not portfolio.positions:
                return 1.0

            # Weight-averaged beta calculation
            total_value = portfolio.total_market_value
            if total_value <= 0:
                return 1.0

            weighted_beta = 0.0
            for position in portfolio.positions:
                if position.quantity != 0:
                    weight = float(abs(position.market_value) / total_value)
                    position_beta = await self._get_position_beta(position.symbol)
                    weighted_beta += weight * position_beta

            return float(weighted_beta)

        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0

    async def _calculate_average_correlation(self, portfolio: PortfolioState) -> float:
        """Calculate average correlation between portfolio positions."""

        try:
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]

            if len(symbols) < 2:
                return 0.0

            correlations = []
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i + 1 :]:
                    correlation = await self._get_symbol_correlation(symbol1, symbol2)
                    correlations.append(correlation)

            return float(np.mean(correlations)) if correlations else 0.0

        except Exception as e:
            logger.error(f"Error calculating average correlation: {e}")
            return 0.0

    async def _calculate_var_metrics(
        self, portfolio: PortfolioState
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate Value at Risk and Expected Shortfall using historical simulation."""

        try:
            if len(self.daily_returns) < 30:  # Need sufficient data
                # Use parametric VaR as fallback
                return await self._parametric_var(portfolio)

            # Historical simulation VaR
            returns_array = np.array(self.daily_returns)
            portfolio_value = portfolio.total_equity

            # 95% confidence level
            var_95_return = np.percentile(returns_array, 5)  # 5th percentile
            var_1d = portfolio_value * Decimal(str(abs(var_95_return)))

            # 5-day VaR (square root of time scaling)
            var_5d = var_1d * Decimal(str(np.sqrt(5)))

            # Expected Shortfall (average of returns below VaR)
            tail_returns = returns_array[returns_array <= var_95_return]
            if len(tail_returns) > 0:
                expected_shortfall = portfolio_value * Decimal(
                    str(abs(np.mean(tail_returns)))
                )
            else:
                expected_shortfall = var_1d * Decimal("1.3")

            return var_1d, var_5d, expected_shortfall

        except Exception as e:
            logger.error(f"Error calculating VaR metrics: {e}")
            return await self._parametric_var(portfolio)

    async def _parametric_var(
        self, portfolio: PortfolioState
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate parametric VaR using normal distribution assumption."""

        try:
            portfolio_value = portfolio.total_equity
            volatility = await self._calculate_portfolio_volatility(portfolio)

            # 95% confidence level (1.645 standard deviations)
            daily_vol = volatility / np.sqrt(252)  # Annualized to daily
            var_1d = portfolio_value * Decimal(str(daily_vol * 1.645))
            var_5d = var_1d * Decimal(str(np.sqrt(5)))

            # Expected Shortfall for normal distribution
            expected_shortfall = var_1d * Decimal(
                "1.282"
            )  # E[X|X<-1.645σ] for normal dist

            return var_1d, var_5d, expected_shortfall

        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            return Decimal("0"), Decimal("0"), Decimal("0")

    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from historical returns."""

        try:
            if len(self.daily_returns) < 30:
                return 0.0

            returns_array = np.array(self.daily_returns)
            risk_free_rate = 0.0001  # 0.01% daily risk-free rate

            excess_returns = returns_array - risk_free_rate

            std_dev = np.std(excess_returns)
            if std_dev > 1e-9:
                sharpe_ratio = np.mean(excess_returns) / std_dev
                # Annualize
                annualized_sharpe = sharpe_ratio * np.sqrt(252)
                # Clamp the value to database field limits (DECIMAL(8,6) = ±99.999999)
                return max(-99.999999, min(99.999999, annualized_sharpe))
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    async def _calculate_drawdown_metrics(
        self, portfolio: PortfolioState
    ) -> Tuple[Decimal, Decimal]:
        """Calculate maximum and current drawdown."""

        try:
            current_value = portfolio.total_equity

            # Update peak value
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value

            # Calculate current drawdown
            if self.peak_portfolio_value > 0:
                current_drawdown = (
                    self.peak_portfolio_value - current_value
                ) / self.peak_portfolio_value
            else:
                current_drawdown = Decimal("0")

            # Calculate maximum drawdown from history
            max_drawdown = await self._calculate_max_historical_drawdown()

            return max_drawdown, current_drawdown

        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return Decimal("0"), Decimal("0")

    async def _calculate_max_historical_drawdown(self) -> Decimal:
        """Calculate maximum historical drawdown."""

        try:
            if len(self.portfolio_history) < 2:
                return Decimal("0")

            # Extract portfolio values
            values = [state.total_equity for _, state in self.portfolio_history]

            max_dd = Decimal("0")
            peak = values[0]

            for value in values[1:]:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak if peak > 0 else Decimal("0")
                    if drawdown > max_dd:
                        max_dd = drawdown

            return max_dd

        except Exception as e:
            logger.error(f"Error calculating max historical drawdown: {e}")
            return Decimal("0")

    async def _calculate_portfolio_volatility(self, portfolio: PortfolioState) -> float:
        """Calculate portfolio volatility using position weights and correlations."""

        try:
            if not portfolio.positions or portfolio.total_equity <= 0:
                return 0.15  # Default volatility

            # Simple approach: weighted average of individual volatilities
            # In production, you'd use the full covariance matrix

            total_value = portfolio.total_market_value
            if total_value <= 0:
                return 0.15

            weighted_volatility = 0.0
            for position in portfolio.positions:
                if position.quantity != 0:
                    weight = float(abs(position.market_value) / total_value)
                    position_vol = await self._get_position_volatility(position.symbol)
                    weighted_volatility += weight * position_vol

            return float(weighted_volatility)

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.15

    def _update_peak_values(self, portfolio: PortfolioState) -> None:
        """Update peak portfolio values for drawdown tracking."""

        if portfolio.total_equity > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio.total_equity
            logger.debug(f"New portfolio peak: ${self.peak_portfolio_value}")

        # Set initial portfolio value if not set
        if self.portfolio_start_value == Decimal("0"):
            self.portfolio_start_value = portfolio.total_equity

    async def _get_position_volatility(self, symbol: str) -> float:
        """
        Get volatility for a specific position with sector-based and market cap estimates.

        TODO: Integrate with market data provider for real-time volatility calculations
        based on historical price data (e.g., 30-day realized volatility).
        """
        try:
            # Check cache first
            if symbol in self._volatility_cache:
                volatility, cached_time = self._volatility_cache[symbol]
                if datetime.now(timezone.utc) - cached_time < self._cache_ttl:
                    return volatility

            # Get sector for sector-based volatility estimates
            sector = await self._get_position_sector(symbol)

            # Sector-based volatility estimates (annualized)
            sector_volatility = {
                "ETF-Broad Market": 0.12,  # Broad market ETFs
                "ETF-Large Cap": 0.13,  # Large cap ETFs
                "ETF-Small Cap": 0.18,  # Small cap ETFs
                "ETF-Technology": 0.22,  # Tech sector ETFs
                "ETF-Sector": 0.20,  # Sector-specific ETFs
                "ETF-International": 0.16,  # International ETFs
                "ETF-Emerging Markets": 0.25,  # Emerging market ETFs
                "ETF-Commodities": 0.28,  # Commodity ETFs
                "ETF-Innovation": 0.35,  # Innovation/growth ETFs
                "ETF-Vanguard": 0.14,  # Vanguard ETFs (typically broad)
                "ETF-iShares": 0.15,  # iShares ETFs
                "ETF-SPDR": 0.15,  # SPDR ETFs
                "ETF-Other": 0.18,  # Other ETFs
                "Technology": 0.25,  # Technology stocks
                "Healthcare": 0.18,  # Healthcare stocks
                "Financials": 0.22,  # Financial stocks
                "Consumer Discretionary": 0.20,  # Consumer discretionary
                "Consumer Staples": 0.14,  # Consumer staples
                "Energy": 0.30,  # Energy stocks
                "Industrials": 0.18,  # Industrial stocks
                "Materials": 0.22,  # Materials stocks
                "Real Estate": 0.20,  # REITs
                "Utilities": 0.14,  # Utility stocks
                "Communication Services": 0.20,  # Communications
                "Cryptocurrency": 0.45,  # Crypto-related stocks
            }

            # Get base volatility from sector
            base_volatility = sector_volatility.get(sector or "Unknown", 0.20)

            # Market cap adjustments based on symbol characteristics
            if sector and sector.startswith("ETF"):
                # ETFs already have appropriate volatility
                volatility = base_volatility
            else:
                # Apply market cap adjustments for individual stocks
                market_cap_multiplier = self._get_market_cap_volatility_multiplier(
                    symbol
                )
                volatility = base_volatility * market_cap_multiplier

                # Special adjustments for known high-volatility stocks
                high_vol_stocks = ["TSLA", "GME", "AMC", "PLTR", "ARKK", "RIOT", "MARA"]
                if symbol in high_vol_stocks:
                    volatility *= 1.5

                # Special adjustments for known low-volatility stocks
                low_vol_stocks = ["BRK.A", "BRK.B", "JNJ", "PG", "KO", "WMT"]
                if symbol in low_vol_stocks:
                    volatility *= 0.8

            # Ensure reasonable bounds
            volatility = max(0.08, min(volatility, 1.0))  # 8% to 100% annual volatility

            # Cache the result
            self._volatility_cache[symbol] = (volatility, datetime.now(timezone.utc))

            return volatility

        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.20

    def _get_market_cap_volatility_multiplier(self, symbol: str) -> float:
        """Get market cap-based volatility multiplier."""
        # Mega cap stocks (typically 3 letters or less, well-known)
        mega_cap_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "BRK.A",
            "BRK.B",
            "UNH",
            "JNJ",
            "JPM",
            "V",
            "PG",
            "MA",
            "HD",
            "CVX",
        ]

        if symbol in mega_cap_symbols:
            return 0.85  # Lower volatility for mega caps
        elif len(symbol) <= 3 and symbol.isupper():
            return 0.95  # Slightly lower for large caps
        elif len(symbol) == 4 and symbol.isupper():
            return 1.1  # Higher for mid caps
        elif len(symbol) >= 5 or any(char.isdigit() for char in symbol):
            return 1.3  # Higher for small caps and complex symbols
        else:
            return 1.0  # Default multiplier

    async def _get_position_beta(self, symbol: str) -> float:
        """
        Get beta for a specific position with sector-based estimates.

        TODO: Integrate with market data provider for real-time beta calculations
        based on historical correlation with market index (e.g., SPY).
        """
        try:
            # Get sector for sector-based beta estimates
            sector = await self._get_position_sector(symbol)

            # Sector-based beta estimates relative to S&P 500
            sector_betas = {
                "ETF-Broad Market": 1.0,  # Market beta
                "ETF-Large Cap": 0.98,  # Slightly lower than market
                "ETF-Small Cap": 1.15,  # Higher beta for small caps
                "ETF-Technology": 1.25,  # Tech sector higher beta
                "ETF-Sector": 1.1,  # Sector ETFs slightly higher
                "ETF-International": 0.85,  # International lower correlation
                "ETF-Emerging Markets": 1.2,  # Higher beta for emerging markets
                "ETF-Commodities": 0.7,  # Lower correlation with stocks
                "ETF-Innovation": 1.4,  # High beta for growth/innovation
                "ETF-Vanguard": 0.95,  # Broad market focus
                "ETF-iShares": 1.0,  # Market beta
                "ETF-SPDR": 1.0,  # Market beta
                "ETF-Other": 1.05,  # Slightly above market
                "Technology": 1.3,  # High beta sector
                "Healthcare": 0.9,  # Defensive sector
                "Financials": 1.2,  # Cyclical, higher beta
                "Consumer Discretionary": 1.15,  # Cyclical
                "Consumer Staples": 0.7,  # Defensive, low beta
                "Energy": 1.25,  # Cyclical, volatile
                "Industrials": 1.1,  # Cyclical
                "Materials": 1.15,  # Cyclical, commodity exposure
                "Real Estate": 0.85,  # Different risk factors
                "Utilities": 0.6,  # Defensive, low beta
                "Communication Services": 1.0,  # Market beta
                "Cryptocurrency": 1.8,  # Very high beta
            }

            # Get base beta from sector
            base_beta = sector_betas.get(sector or "Unknown", 1.0)

            # Individual stock adjustments
            high_beta_stocks = {
                "TSLA": 2.0,
                "NVDA": 1.8,
                "AMD": 1.7,
                "ARKK": 1.6,
                "RIOT": 2.5,
                "MARA": 2.3,
                "GME": 2.8,
                "AMC": 2.5,
                "PLTR": 2.0,
                "ZOOM": 1.5,
                "ROKU": 1.8,
                "UBER": 1.4,
            }

            low_beta_stocks = {
                "BRK.A": 0.8,
                "BRK.B": 0.8,
                "JNJ": 0.7,
                "PG": 0.6,
                "KO": 0.6,
                "WMT": 0.5,
                "UNH": 0.8,
                "NEE": 0.6,
                "D": 0.5,
                "SO": 0.5,
                "DUK": 0.5,
            }

            if symbol in high_beta_stocks:
                return high_beta_stocks[symbol]
            elif symbol in low_beta_stocks:
                return low_beta_stocks[symbol]
            else:
                # Apply market cap adjustments
                if len(symbol) <= 3 and symbol.isupper():
                    # Large cap - slightly lower beta
                    return base_beta * 0.95
                elif len(symbol) >= 5:
                    # Small cap - higher beta
                    return base_beta * 1.15
                else:
                    return base_beta

        except Exception as e:
            logger.error(f"Error getting beta for {symbol}: {e}")
            return 1.0

    async def _calculate_expected_return(
        self, symbol: str, beta: float, volatility: float
    ) -> float:
        """
        Calculate expected return for a position based on sector, beta, and market conditions.

        TODO: Integrate with market data for real-time risk-free rates and market risk premiums.
        Consider using CAPM model with updated market data.
        """
        try:
            # Get sector for sector-based return estimates
            sector = await self._get_position_sector(symbol)

            # Market risk premium (historical average ~6-8% annually)
            market_risk_premium = (
                0.07 / 252
            )  # Convert to daily (7% annual / 252 trading days)

            # Risk-free rate (approximation - should fetch from Fed data)
            risk_free_rate = 0.02 / 252  # 2% annual risk-free rate

            # Sector-based expected return adjustments (annual basis, converted to daily)
            sector_premiums = {
                "ETF-Broad Market": 0.0,  # Market return
                "ETF-Large Cap": -0.005,  # Slightly below market
                "ETF-Small Cap": 0.015,  # Small cap premium
                "ETF-Technology": 0.01,  # Tech growth premium
                "ETF-Sector": 0.005,  # Sector concentration premium
                "ETF-International": -0.01,  # Lower expected returns
                "ETF-Emerging Markets": 0.02,  # Emerging market premium
                "ETF-Commodities": 0.005,  # Commodity premium
                "ETF-Innovation": 0.025,  # High growth premium
                "ETF-Vanguard": -0.002,  # Low cost advantage
                "ETF-iShares": 0.0,  # Market return
                "ETF-SPDR": 0.0,  # Market return
                "ETF-Other": 0.002,  # Slight premium
                "Technology": 0.015,  # Growth sector premium
                "Healthcare": 0.005,  # Stable growth
                "Financials": 0.008,  # Economic sensitivity premium
                "Consumer Discretionary": 0.01,  # Economic growth exposure
                "Consumer Staples": -0.005,  # Defensive discount
                "Energy": 0.012,  # Volatility premium
                "Industrials": 0.007,  # Economic cycle exposure
                "Materials": 0.008,  # Commodity exposure
                "Real Estate": 0.002,  # Different risk factors
                "Utilities": -0.008,  # Low growth, defensive
                "Communication Services": 0.005,  # Mixed growth prospects
                "Cryptocurrency": 0.05,  # Very high risk premium
            }

            # Get sector premium (convert annual to daily)
            sector_premium = sector_premiums.get(sector or "Unknown", 0.0) / 252

            # CAPM-based expected return: Risk-free rate + Beta * Market risk premium
            capm_return = risk_free_rate + (beta * market_risk_premium)

            # Add sector-specific premium
            expected_return = capm_return + sector_premium

            # Quality/momentum adjustments for individual stocks
            quality_adjustments = {
                # High quality stocks (lower required return)
                "AAPL": -0.001,
                "MSFT": -0.001,
                "JNJ": -0.0005,
                "PG": -0.0005,
                "BRK.A": -0.0008,
                "BRK.B": -0.0008,
                "UNH": -0.0005,
                # High growth/momentum stocks (higher expected return)
                "TSLA": 0.002,
                "NVDA": 0.0015,
                "AMD": 0.001,
                "ARKK": 0.002,
                # High risk stocks (higher required return)
                "GME": 0.003,
                "AMC": 0.003,
                "RIOT": 0.004,
                "MARA": 0.004,
                "PLTR": 0.002,
                # Defensive stocks (lower expected return)
                "WMT": -0.0003,
                "KO": -0.0003,
                "NEE": -0.0003,
                "D": -0.0003,
            }

            if symbol in quality_adjustments:
                expected_return += quality_adjustments[symbol]

            # Volatility adjustment (higher volatility should demand higher return)
            volatility_adjustment = max(
                0, (volatility - 0.15) * 0.1 / 252
            )  # Extra return for vol > 15%
            expected_return += volatility_adjustment

            # Ensure reasonable bounds (daily returns between -0.5% to 2%)
            expected_return = max(-0.005, min(expected_return, 0.02))

            return expected_return

        except Exception as e:
            logger.error(f"Error calculating expected return for {symbol}: {e}")
            return 0.0003  # Default 0.03% daily return

    async def _get_portfolio_correlation(
        self, symbol: str, portfolio: PortfolioState
    ) -> float:
        """Calculate correlation between symbol and rest of portfolio."""

        try:
            if not portfolio.positions:
                return 0.0

            # Calculate weighted correlation with existing positions
            total_value = portfolio.total_market_value
            other_positions_value = total_value

            # Find this symbol's position to exclude from calculation
            for position in portfolio.positions:
                if position.symbol == symbol and position.quantity != 0:
                    other_positions_value -= abs(position.market_value)
                    break

            if other_positions_value <= 0:
                return 0.0

            weighted_correlation = 0.0
            for position in portfolio.positions:
                if position.symbol != symbol and position.quantity != 0:
                    weight = float(abs(position.market_value) / other_positions_value)
                    correlation = await self._get_symbol_correlation(
                        symbol, position.symbol
                    )
                    weighted_correlation += weight * correlation

            return weighted_correlation

        except Exception as e:
            logger.error(f"Error calculating portfolio correlation for {symbol}: {e}")
            return 0.0

    async def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""

        if symbol1 == symbol2:
            return 1.0

        # Create cache key (sorted to avoid duplicates)
        sorted_symbols = sorted([symbol1, symbol2])
        cache_key = (sorted_symbols[0], sorted_symbols[1])

        # Check cache
        if cache_key in self._correlation_cache:
            correlation, timestamp = self._correlation_cache[cache_key]
            if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                return correlation

        try:
            # Calculate correlation from historical price data
            correlation_result = await self._calculate_price_correlation(
                symbol1, symbol2
            )
            correlation = correlation_result if correlation_result is not None else 0.0

            # Fallback to estimation if no historical data available
            if correlation == 0.0:
                estimated_correlation = await self._estimate_correlation(
                    symbol1, symbol2
                )
                correlation = (
                    estimated_correlation if estimated_correlation is not None else 0.0
                )

            # Cache result
            self._correlation_cache[cache_key] = (
                correlation,
                datetime.now(timezone.utc),
            )

            return correlation

        except Exception as e:
            logger.error(
                f"Error getting correlation between {symbol1} and {symbol2}: {e}"
            )
            return 0.0

    async def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Estimate correlation between symbols using sector correlation matrices and market regime analysis.

        TODO: Integrate with market data provider for real-time correlation calculations
        based on rolling historical price data (e.g., 60-day rolling correlation).
        """

        # Same symbol always has correlation of 1
        if symbol1 == symbol2:
            return 1.0

        # Get sectors for both symbols
        sector1 = await self._get_position_sector(symbol1)
        sector2 = await self._get_position_sector(symbol2)

        # Sector correlation matrix based on historical relationships
        sector_correlations = {
            # Technology sector correlations
            ("Technology", "Technology"): 0.75,
            ("Technology", "Consumer Discretionary"): 0.55,
            ("Technology", "Communication Services"): 0.65,
            ("Technology", "Healthcare"): 0.35,
            ("Technology", "Financials"): 0.45,
            ("Technology", "Industrials"): 0.40,
            ("Technology", "Materials"): 0.35,
            ("Technology", "Energy"): 0.25,
            ("Technology", "Utilities"): 0.15,
            ("Technology", "Consumer Staples"): 0.20,
            ("Technology", "Real Estate"): 0.25,
            # Financial sector correlations
            ("Financials", "Financials"): 0.80,
            ("Financials", "Real Estate"): 0.45,
            ("Financials", "Industrials"): 0.55,
            ("Financials", "Materials"): 0.50,
            ("Financials", "Energy"): 0.45,
            ("Financials", "Consumer Discretionary"): 0.60,
            ("Financials", "Healthcare"): 0.35,
            ("Financials", "Utilities"): 0.30,
            ("Financials", "Consumer Staples"): 0.35,
            ("Financials", "Communication Services"): 0.40,
            # Healthcare sector correlations
            ("Healthcare", "Healthcare"): 0.70,
            ("Healthcare", "Consumer Staples"): 0.45,
            ("Healthcare", "Utilities"): 0.35,
            ("Healthcare", "Real Estate"): 0.30,
            ("Healthcare", "Consumer Discretionary"): 0.40,
            ("Healthcare", "Industrials"): 0.35,
            ("Healthcare", "Materials"): 0.30,
            ("Healthcare", "Energy"): 0.25,
            ("Healthcare", "Communication Services"): 0.30,
            # Energy sector correlations
            ("Energy", "Energy"): 0.85,
            ("Energy", "Materials"): 0.65,
            ("Energy", "Industrials"): 0.50,
            ("Energy", "Utilities"): 0.40,
            ("Energy", "Consumer Discretionary"): 0.40,
            ("Energy", "Consumer Staples"): 0.30,
            ("Energy", "Real Estate"): 0.35,
            ("Energy", "Communication Services"): 0.25,
            # Consumer sectors
            ("Consumer Discretionary", "Consumer Discretionary"): 0.75,
            ("Consumer Discretionary", "Consumer Staples"): 0.40,
            ("Consumer Discretionary", "Industrials"): 0.55,
            ("Consumer Discretionary", "Materials"): 0.45,
            ("Consumer Discretionary", "Utilities"): 0.25,
            ("Consumer Discretionary", "Real Estate"): 0.35,
            ("Consumer Discretionary", "Communication Services"): 0.45,
            ("Consumer Staples", "Consumer Staples"): 0.65,
            ("Consumer Staples", "Utilities"): 0.50,
            ("Consumer Staples", "Real Estate"): 0.35,
            ("Consumer Staples", "Industrials"): 0.35,
            ("Consumer Staples", "Materials"): 0.30,
            ("Consumer Staples", "Communication Services"): 0.25,
            # Industrial and Materials
            ("Industrials", "Industrials"): 0.70,
            ("Industrials", "Materials"): 0.60,
            ("Industrials", "Utilities"): 0.40,
            ("Industrials", "Real Estate"): 0.35,
            ("Industrials", "Communication Services"): 0.35,
            ("Materials", "Materials"): 0.75,
            ("Materials", "Utilities"): 0.35,
            ("Materials", "Real Estate"): 0.30,
            ("Materials", "Communication Services"): 0.25,
            # Utilities and Real Estate
            ("Utilities", "Utilities"): 0.60,
            ("Utilities", "Real Estate"): 0.45,
            ("Utilities", "Communication Services"): 0.25,
            ("Real Estate", "Real Estate"): 0.65,
            ("Real Estate", "Communication Services"): 0.25,
            # Communication Services
            ("Communication Services", "Communication Services"): 0.70,
            # Special cases for ETFs
            ("ETF-Broad Market", "ETF-Broad Market"): 0.95,
            ("ETF-Broad Market", "ETF-Large Cap"): 0.98,
            ("ETF-Broad Market", "ETF-Small Cap"): 0.85,
            ("ETF-Large Cap", "ETF-Large Cap"): 0.95,
            ("ETF-Large Cap", "ETF-Small Cap"): 0.80,
            ("ETF-Small Cap", "ETF-Small Cap"): 0.90,
            # Sector ETFs with their underlying sectors
            ("ETF-Technology", "Technology"): 0.95,
            ("ETF-Sector", "Financials"): 0.90,
            ("ETF-Sector", "Healthcare"): 0.90,
            ("ETF-Sector", "Energy"): 0.90,
            # Cryptocurrency correlations
            ("Cryptocurrency", "Cryptocurrency"): 0.80,
            ("Cryptocurrency", "Technology"): 0.35,
        }

        # Look up correlation using both orders, handling None sectors
        if sector1 is None or sector2 is None:
            base_correlation = 0.25
        else:
            correlation_key1 = (sector1, sector2)
            correlation_key2 = (sector2, sector1)

            if correlation_key1 in sector_correlations:
                base_correlation = sector_correlations[correlation_key1]
            elif correlation_key2 in sector_correlations:
                base_correlation = sector_correlations[correlation_key2]
            else:
                # Default inter-sector correlation
                base_correlation = 0.25

        # Individual stock adjustments
        correlation = base_correlation

        # High correlation pairs (known to move together)
        high_correlation_pairs = {
            ("AAPL", "MSFT"): 0.70,
            ("GOOGL", "GOOG"): 0.99,  # Same company
            ("BRK.A", "BRK.B"): 0.99,  # Same company
            ("JPM", "BAC"): 0.75,  # Major banks
            ("XOM", "CVX"): 0.80,  # Oil companies
            ("KO", "PEP"): 0.65,  # Beverage companies
            ("WMT", "COST"): 0.60,  # Retail
            ("UNH", "ANTM"): 0.70,  # Health insurers
        }

        # Check for specific high correlation pairs
        pair_key1 = (symbol1, symbol2)
        pair_key2 = (symbol2, symbol1)

        if pair_key1 in high_correlation_pairs:
            correlation = high_correlation_pairs[pair_key1]
        elif pair_key2 in high_correlation_pairs:
            correlation = high_correlation_pairs[pair_key2]

        # Market regime adjustments
        # TODO: Implement market regime detection (bull/bear/crisis)
        # During crisis periods, correlations tend to increase
        # market_regime = await self._detect_market_regime()
        # if market_regime == 'crisis':
        #     correlation = min(0.95, correlation * 1.3)

        # Market cap similarity adjustment
        if self._similar_market_cap(symbol1, symbol2):
            correlation *= 1.1  # Similar sized companies tend to be more correlated

        # ETF special handling
        if (
            sector1
            and sector1.startswith("ETF")
            and sector2
            and sector2.startswith("ETF")
        ):
            # ETFs generally have higher correlations
            correlation = max(correlation, 0.6)
        elif (sector1 and sector1.startswith("ETF")) or (
            sector2 and sector2.startswith("ETF")
        ):
            # ETF with individual stock
            if sector1 and not sector1.startswith("ETF"):
                # Check if ETF contains this sector
                if sector2 == "ETF-Technology" and sector1 == "Technology":
                    correlation = 0.85
                elif sector2 == "ETF-Broad Market":
                    correlation = 0.70  # Broad market exposure
            elif sector2 and not sector2.startswith("ETF"):
                if sector1 == "ETF-Technology" and sector2 == "Technology":
                    correlation = 0.85
                elif sector1 == "ETF-Broad Market":
                    correlation = 0.70

        # Ensure correlation is within valid bounds
        correlation = max(-0.5, min(correlation, 0.99))

        return correlation

    def _similar_market_cap(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols have similar market capitalizations."""
        # Mega cap symbols (>$500B)
        mega_cap = {
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "BRK.A",
            "BRK.B",
            "UNH",
            "JNJ",
        }

        # Large cap symbols ($10B-$500B)
        large_cap = {
            "JPM",
            "V",
            "PG",
            "MA",
            "HD",
            "CVX",
            "XOM",
            "PFE",
            "ABBV",
            "KO",
            "MRK",
            "TMO",
            "ABT",
            "COST",
            "ADBE",
            "NFLX",
            "CRM",
            "DHR",
            "NKE",
            "LIN",
            "T",
            "VZ",
            "CMCSA",
            "WFC",
            "BAC",
            "INTC",
            "AMD",
            "QCOM",
        }

        # Mid cap symbols ($2B-$10B) - typically 4 character symbols
        def mid_cap_patterns(s: str) -> bool:
            return len(s) == 4 and s.isupper() and not any(c.isdigit() for c in s)

        # Small cap symbols (<$2B) - typically longer symbols or with numbers
        def small_cap_patterns(s: str) -> bool:
            return len(s) >= 5 or any(c.isdigit() for c in s)

        # Check if both are in same category
        if symbol1 in mega_cap and symbol2 in mega_cap:
            return True
        elif symbol1 in large_cap and symbol2 in large_cap:
            return True
        elif mid_cap_patterns(symbol1) and mid_cap_patterns(symbol2):
            return True
        elif small_cap_patterns(symbol1) and small_cap_patterns(symbol2):
            return True

        return False

    async def _same_sector(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are in the same sector."""
        sector1 = await self._get_position_sector(symbol1)
        sector2 = await self._get_position_sector(symbol2)
        return sector1 == sector2 and sector1 is not None

    async def _is_tech_stock(self, symbol: str) -> bool:
        """
        Check if symbol is a technology stock with comprehensive detection.

        TODO: Integrate with sector classification API for real-time sector data.
        """
        # Core technology companies
        core_tech = {
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "META",
            "NVDA",
            "AMD",
            "INTC",
            "ORCL",
            "IBM",
            "CRM",
            "ADBE",
            "NFLX",
            "PYPL",
            "UBER",
            "LYFT",
            "ZOOM",
            "ROKU",
            "SQ",
            "SHOP",
            "SNOW",
            "PLTR",
            "CRWD",
            "ZS",
            "OKTA",
            "TWLO",
            "DOCU",
            "WORK",
            "TEAM",
            "ATLX",
            "NOW",
            "SPLK",
        }

        # Cloud and SaaS companies
        cloud_saas = {
            "AMZN",
            "CRM",
            "MSFT",
            "GOOGL",
            "ADBE",
            "NOW",
            "WDAY",
            "VEEV",
            "DDOG",
            "NET",
            "FSLY",
            "MDB",
            "ESTC",
            "ZM",
            "DOCN",
            "GTLB",
        }

        # Semiconductor companies
        semiconductors = {
            "NVDA",
            "AMD",
            "INTC",
            "TSM",
            "ASML",
            "QCOM",
            "AVGO",
            "TXN",
            "ADI",
            "MRVL",
            "XLNX",
            "LRCX",
            "AMAT",
            "KLAC",
            "MCHP",
            "SWKS",
        }

        # E-commerce and digital platforms
        ecommerce = {"AMZN", "SHOP", "EBAY", "ETSY", "BABA", "JD", "MELI", "SE"}

        # Fintech companies
        fintech = {"PYPL", "SQ", "V", "MA", "COIN", "HOOD", "AFRM", "SOFI", "LC"}

        # Cybersecurity companies
        cybersecurity = {"CRWD", "ZS", "OKTA", "PANW", "FTNT", "CHKP", "CYBR", "TENB"}

        # Check if symbol is in any tech category
        if symbol in core_tech or symbol in cloud_saas or symbol in semiconductors:
            return True
        elif symbol in ecommerce or symbol in fintech or symbol in cybersecurity:
            return True

        # Pattern-based detection for newer tech companies
        tech_patterns = ["TECH", "SOFT", "DATA", "CLOUD", "CYBER", "AI", "ROBO", "AUTO"]

        if any(pattern in symbol.upper() for pattern in tech_patterns):
            return True

        # Check sector classification
        sector = await self._get_position_sector(symbol)
        return sector == "Technology"

    async def _get_position_sector(self, symbol: str) -> str | None:
        """Get sector for a position with comprehensive sector mapping."""
        # TODO: Add test_get_position_sector() - Verify sector lookup from screener data and fallback
        # Check cache first
        cache_key = symbol.upper()
        if cache_key in self._sector_cache:
            sector, timestamp = self._sector_cache[cache_key]
            if datetime.now(timezone.utc) - timestamp < timedelta(hours=24):
                return sector

        # Try to get sector from screener data
        try:
            screener_df = await self._load_screener_data()
            if screener_df is not None:
                symbol_data = screener_df.filter(pl.col("symbol") == symbol.upper())
                if len(symbol_data) > 0:
                    sector = symbol_data.select("sector").item()
                    if sector and sector != "Unknown" and sector.strip():
                        # Cache the result
                        self._sector_cache[cache_key] = (
                            sector,
                            datetime.now(timezone.utc),
                        )
                        logger.debug(
                            f"Found sector '{sector}' for {symbol} from screener data"
                        )
                        return sector
        except Exception as e:
            logger.warning(f"Failed to get sector from screener data for {symbol}: {e}")

        # Fallback to comprehensive sector mapping
        sector_map = {
            # Technology
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "GOOG": "Technology",
            "AMZN": "Technology",
            "META": "Technology",
            "NVDA": "Technology",
            "TSLA": "Technology",
            "NFLX": "Technology",
            "ADBE": "Technology",
            "CRM": "Technology",
            "ORCL": "Technology",
            "INTC": "Technology",
            "IBM": "Technology",
            "AMD": "Technology",
            "PYPL": "Technology",
            "UBER": "Technology",
            "LYFT": "Technology",
            "ZOOM": "Technology",
            "ROKU": "Technology",
            "SQ": "Technology",
            "SHOP": "Technology",
            "SNOW": "Technology",
            "PLTR": "Technology",
            # Healthcare & Pharmaceuticals
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "UNH": "Healthcare",
            "ABBV": "Healthcare",
            "MRK": "Healthcare",
            "TMO": "Healthcare",
            "ABT": "Healthcare",
            "DHR": "Healthcare",
            "BMY": "Healthcare",
            "AMGN": "Healthcare",
            "GILD": "Healthcare",
            "BIIB": "Healthcare",
            "CVS": "Healthcare",
            "ANTM": "Healthcare",
            "CI": "Healthcare",
            "HUM": "Healthcare",
            # Financials
            "JPM": "Financials",
            "BAC": "Financials",
            "WFC": "Financials",
            "C": "Financials",
            "GS": "Financials",
            "MS": "Financials",
            "AXP": "Financials",
            "BLK": "Financials",
            "SCHW": "Financials",
            "USB": "Financials",
            "PNC": "Financials",
            "TFC": "Financials",
            "BRK.A": "Financials",
            "BRK.B": "Financials",
            "V": "Financials",
            "MA": "Financials",
            # Consumer Discretionary
            "HD": "Consumer Discretionary",
            "NKE": "Consumer Discretionary",
            "MCD": "Consumer Discretionary",
            "SBUX": "Consumer Discretionary",
            "LOW": "Consumer Discretionary",
            "TJX": "Consumer Discretionary",
            "BKNG": "Consumer Discretionary",
            "CMG": "Consumer Discretionary",
            "F": "Consumer Discretionary",
            "GM": "Consumer Discretionary",
            "DIS": "Consumer Discretionary",
            # Consumer Staples
            "PG": "Consumer Staples",
            "KO": "Consumer Staples",
            "PEP": "Consumer Staples",
            "WMT": "Consumer Staples",
            "COST": "Consumer Staples",
            "CL": "Consumer Staples",
            "KMB": "Consumer Staples",
            "GIS": "Consumer Staples",
            "K": "Consumer Staples",
            # Energy
            "XOM": "Energy",
            "CVX": "Energy",
            "COP": "Energy",
            "SLB": "Energy",
            "EOG": "Energy",
            "PXD": "Energy",
            "KMI": "Energy",
            "OKE": "Energy",
            # Industrials
            "BA": "Industrials",
            "CAT": "Industrials",
            "HON": "Industrials",
            "UPS": "Industrials",
            "RTX": "Industrials",
            "LMT": "Industrials",
            "GE": "Industrials",
            "MMM": "Industrials",
            "DE": "Industrials",
            "FDX": "Industrials",
            "UNP": "Industrials",
            "CSX": "Industrials",
            # Materials
            "LIN": "Materials",
            "APD": "Materials",
            "SHW": "Materials",
            "FCX": "Materials",
            "NEM": "Materials",
            "DOW": "Materials",
            "DD": "Materials",
            "PPG": "Materials",
            # Real Estate
            "AMT": "Real Estate",
            "PLD": "Real Estate",
            "CCI": "Real Estate",
            "EQIX": "Real Estate",
            "SPG": "Real Estate",
            "O": "Real Estate",
            "WELL": "Real Estate",
            "PSA": "Real Estate",
            # Utilities
            "NEE": "Utilities",
            "DUK": "Utilities",
            "SO": "Utilities",
            "D": "Utilities",
            "AEP": "Utilities",
            "EXC": "Utilities",
            "XEL": "Utilities",
            "SRE": "Utilities",
            # Communications
            "T": "Communication Services",
            "VZ": "Communication Services",
            "CMCSA": "Communication Services",
            "CHTR": "Communication Services",
            "TMUS": "Communication Services",
            "ATVI": "Communication Services",
            # ETFs and Index Funds
            "SPY": "ETF-Broad Market",
            "QQQ": "ETF-Technology",
            "IWM": "ETF-Small Cap",
            "VTI": "ETF-Broad Market",
            "VOO": "ETF-Large Cap",
            "VEA": "ETF-International",
            "VWO": "ETF-Emerging Markets",
            "GLD": "ETF-Commodities",
            "SLV": "ETF-Commodities",
            "XLF": "ETF-Sector",
            "XLK": "ETF-Sector",
            "XLE": "ETF-Sector",
            "XLV": "ETF-Sector",
            "ARKK": "ETF-Innovation",
            "ARKQ": "ETF-Innovation",
            "ARKG": "ETF-Innovation",
            # Cryptocurrency-related
            "COIN": "Cryptocurrency",
            "MSTR": "Cryptocurrency",
            "RIOT": "Cryptocurrency",
            "MARA": "Cryptocurrency",
            "HUT": "Cryptocurrency",
            "BITF": "Cryptocurrency",
        }

        # Handle specific cases
        sector_lookup: Optional[str] = sector_map.get(symbol)

        if sector_lookup is not None:
            # Cache and return
            self._sector_cache[cache_key] = (sector_lookup, datetime.now(timezone.utc))
            return sector_lookup

        # ETF detection by common patterns
        if any(
            symbol.startswith(prefix) for prefix in ["VT", "VO", "VE", "VB", "VU", "VI"]
        ):
            sector = "ETF-Vanguard"
        elif any(
            symbol.startswith(prefix) for prefix in ["IVV", "IWF", "IWD", "IJH", "IJR"]
        ):
            sector = "ETF-iShares"
        elif symbol.startswith("SPD"):
            sector = "ETF-SPDR"
        elif symbol.endswith("ETF") or len(symbol) == 3 and symbol.isupper():
            sector = "ETF-Other"
        # Cryptocurrency detection
        elif any(
            crypto in symbol.upper()
            for crypto in ["BTC", "ETH", "CRYPTO", "COIN", "BLOCKCHAIN"]
        ):
            sector = "Cryptocurrency"
        # Default classification based on symbol characteristics
        elif (
            len(symbol) <= 4
            and symbol.isupper()
            and not any(char.isdigit() for char in symbol)
        ):
            sector = "Equity-Unknown"
        else:
            sector = "Unknown"

        # Cache the result
        self._sector_cache[cache_key] = (sector, datetime.now(timezone.utc))
        return sector

    async def _calculate_position_risk_score(
        self,
        volatility: float,
        portfolio_percentage: float,
        correlation: float,
        beta: float,
    ) -> float:
        """Calculate overall risk score for a position (0-10 scale)."""

        try:
            # Risk components (0-10 scale each)

            # Volatility risk (0-10)
            vol_risk = min(10, (volatility / 0.5) * 10)  # Scale where 50% vol = 10

            # Size risk (0-10)
            size_risk = min(
                10, (portfolio_percentage / 0.3) * 10
            )  # Scale where 30% = 10

            # Correlation risk (0-10)
            corr_risk = correlation * 10  # Direct scaling

            # Beta risk (0-10)
            beta_risk = min(10, abs(beta - 1.0) * 5)  # Distance from market beta

            # Weighted average (volatility and size are most important)
            risk_score = (
                vol_risk * 0.4  # 40% weight
                + size_risk * 0.3  # 30% weight
                + corr_risk * 0.2  # 20% weight
                + beta_risk * 0.1  # 10% weight
            )

            return min(10.0, max(0.0, risk_score))

        except Exception as e:
            logger.error(f"Error calculating position risk score: {e}")
            return 5.0  # Medium risk score on error

    def get_portfolio_summary(self, portfolio: PortfolioState) -> Dict:
        """Get a summary of current portfolio status."""

        try:
            active_positions = [p for p in portfolio.positions if p.quantity != 0]

            summary = {
                "total_equity": str(portfolio.total_equity),
                "cash": str(portfolio.cash),
                "buying_power": str(portfolio.buying_power),
                "total_market_value": str(portfolio.total_market_value),
                "total_unrealized_pnl": str(portfolio.total_unrealized_pnl),
                "position_count": len(active_positions),
                "day_trades_count": portfolio.day_trades_count,
                "pattern_day_trader": portfolio.pattern_day_trader,
            }

            # Add position details
            positions_summary = []
            for position in active_positions:
                pos_summary = {
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "market_value": str(position.market_value),
                    "unrealized_pnl": str(position.unrealized_pnl),
                    "percentage": (
                        str(abs(position.market_value) / portfolio.total_equity * 100)
                        if portfolio.total_equity > 0
                        else "0"
                    ),
                }
                positions_summary.append(pos_summary)

            summary["positions"] = positions_summary

            return summary

        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {
                "error": "Failed to generate portfolio summary",
                "total_equity": str(portfolio.total_equity),
                "position_count": len(portfolio.positions),
            }

    async def check_portfolio_health(
        self, portfolio: PortfolioState
    ) -> Dict[str, bool]:
        """Perform comprehensive portfolio health checks."""

        health_status = {
            "overall_healthy": True,
            "within_risk_limits": True,
            "positions_under_limit": True,
            "drawdown_acceptable": True,
            "correlation_acceptable": True,
            "volatility_acceptable": True,
        }

        try:
            # Check position count
            active_positions = len([p for p in portfolio.positions if p.quantity != 0])
            if active_positions > self.risk_limits.max_positions:
                health_status["positions_under_limit"] = False
                health_status["overall_healthy"] = False

            # Check drawdown
            _, current_drawdown = await self._calculate_drawdown_metrics(portfolio)
            if current_drawdown > self.risk_limits.max_drawdown_percentage:
                health_status["drawdown_acceptable"] = False
                health_status["overall_healthy"] = False

            # Check individual position sizes
            for position in portfolio.positions:
                if position.quantity != 0:
                    position_pct = (
                        abs(position.market_value) / portfolio.total_equity
                        if portfolio.total_equity > 0
                        else Decimal("0")
                    )
                    if position_pct > self.risk_limits.max_position_percentage:
                        health_status["within_risk_limits"] = False
                        health_status["overall_healthy"] = False
                        break

            # Check average correlation
            avg_correlation = await self._calculate_average_correlation(portfolio)
            if avg_correlation > self.risk_limits.max_correlation_threshold:
                health_status["correlation_acceptable"] = False
                health_status["overall_healthy"] = False

            # Check portfolio volatility
            portfolio_vol = await self._calculate_portfolio_volatility(portfolio)
            if portfolio_vol > self.risk_limits.max_position_volatility:
                health_status["volatility_acceptable"] = False
                health_status["overall_healthy"] = False

            return health_status

        except Exception as e:
            logger.error(f"Error checking portfolio health: {e}")
            return {key: False for key in health_status.keys()}

    def get_risk_exposure_by_sector(
        self, portfolio: PortfolioState
    ) -> Dict[str, Decimal]:
        """Calculate risk exposure by sector."""

        sector_exposure: Dict[str, Any] = {}

        try:
            if portfolio.total_equity <= 0:
                return sector_exposure

            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                # Get sector from cached data
                sector = self._get_position_sector_sync(position.symbol)
                exposure = abs(position.market_value) / portfolio.total_equity

                if sector is not None:
                    if sector in sector_exposure:
                        sector_exposure[sector] += exposure
                    else:
                        sector_exposure[sector] = exposure

            return sector_exposure

        except Exception as e:
            logger.error(f"Error calculating sector exposure: {e}")
            return {}

    def _get_position_sector_sync(self, symbol: str) -> str | None:
        """Synchronous version of sector lookup."""
        # TODO: Add test_get_position_sector_sync() - Verify sync sector lookup and caching
        # Check cache first
        cache_key = symbol.upper()
        if cache_key in self._sector_cache:
            sector, timestamp = self._sector_cache[cache_key]
            if datetime.now(timezone.utc) - timestamp < timedelta(hours=24):
                return sector

        # Try to load screener data synchronously (limited recent data)
        try:
            screener_dir = self._data_path / "screener_data" / "momentum"
            if screener_dir.exists():
                # Check only today's file for sync access (to avoid blocking)
                today_file = (
                    screener_dir / f"{date.today().strftime('%Y-%m-%d')}.parquet"
                )
                if today_file.exists():
                    df = pl.read_parquet(today_file)
                    symbol_data = df.filter(pl.col("symbol") == symbol.upper())
                    if len(symbol_data) > 0:
                        sector = symbol_data.select("sector").item()
                        if sector and sector != "Unknown" and sector.strip():
                            # Cache the result
                            self._sector_cache[cache_key] = (
                                sector,
                                datetime.now(timezone.utc),
                            )
                            logger.debug(
                                f"Found sector '{sector}' for {symbol} from today's screener data (sync)"
                            )
                            return sector
        except Exception:
            pass  # Fall through to hardcoded mapping

        # Fallback to basic sector mapping
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "GOOG": "Technology",
            "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary",
            "NVDA": "Technology",
            "META": "Technology",
            "JPM": "Financials",
            "BAC": "Financials",
            "WFC": "Financials",
            "GS": "Financials",
            "SPY": "ETF",
            "QQQ": "ETF",
            "IWM": "ETF",
            "VTI": "ETF",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "UNH": "Healthcare",
            "XOM": "Energy",
            "CVX": "Energy",
        }

        sector = sector_map.get(symbol, "Unknown")

        # Cache the result
        self._sector_cache[cache_key] = (sector, datetime.now(timezone.utc))

        return sector

    async def generate_risk_warnings(self, portfolio: PortfolioState) -> List[str]:
        """Generate list of current risk warnings."""

        warnings = []

        try:
            # Check concentration
            concentration = await self._calculate_concentration_risk(portfolio)
            if concentration > 0.6:
                warnings.append(f"High portfolio concentration: {concentration:.1%}")

            # Check large positions
            for position in portfolio.positions:
                if position.quantity != 0:
                    position_pct = (
                        abs(position.market_value) / portfolio.total_equity
                        if portfolio.total_equity > 0
                        else Decimal("0")
                    )
                    if (
                        position_pct
                        > self.risk_limits.max_position_percentage * Decimal("0.8")
                    ):
                        warnings.append(
                            f"Large position {position.symbol}: {position_pct:.1%}"
                        )

            # Check drawdown
            _, current_drawdown = await self._calculate_drawdown_metrics(portfolio)
            if current_drawdown > self.risk_limits.max_drawdown_percentage * Decimal(
                "0.7"
            ):
                warnings.append(f"High drawdown: {current_drawdown:.1%}")

            # Check correlation
            avg_correlation = await self._calculate_average_correlation(portfolio)
            if avg_correlation > self.risk_limits.max_correlation_threshold * 0.8:
                warnings.append(f"High average correlation: {avg_correlation:.2f}")

            return warnings

        except Exception as e:
            logger.error(f"Error generating risk warnings: {e}")
            return ["Error generating risk warnings"]

    def reset_monitoring_state(self) -> None:
        """Reset monitoring state (for new trading session)."""

        self.daily_returns.clear()
        self.risk_alerts.clear()
        self.last_alert_times.clear()

        logger.info("Portfolio monitoring state reset")

    async def _load_screener_data(self, days_back: int = 30) -> Optional[pl.DataFrame]:
        """
        Load recent screener data from stored Parquet files.

        Args:
            days_back: Number of days to look back for data

        Returns:
            DataFrame with screener data or None if not found
        """
        # TODO: Add test_load_screener_data() - Verify screener data loading and caching
        try:
            screener_dir = self._data_path / "screener_data" / "momentum"
            if not screener_dir.exists():
                logger.warning(f"Screener data directory not found: {screener_dir}")
                return None

            # Try to load recent files
            current_date = date.today()
            for i in range(days_back):
                check_date = current_date - timedelta(days=i)
                file_path = screener_dir / f"{check_date.strftime('%Y-%m-%d')}.parquet"

                if file_path.exists():
                    try:
                        df = pl.read_parquet(file_path)
                        logger.debug(
                            f"Loaded screener data from {file_path} with {len(df)} records"
                        )
                        return df
                    except Exception as e:
                        logger.warning(f"Failed to read screener file {file_path}: {e}")
                        continue

            logger.warning(f"No screener data found in last {days_back} days")
            return None

        except Exception as e:
            logger.error(f"Error loading screener data: {e}")
            return None

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for screener updates."""
        try:
            # Build Redis URL with password from environment
            redis_password = os.getenv("REDIS_PASSWORD", "")
            if redis_password:
                redis_url = f"redis://:{redis_password}@redis:6379"
            else:
                redis_url = "redis://redis:6379"

            self._redis = redis.from_url(redis_url, decode_responses=True)
            if self._redis is not None:
                await self._redis.ping()

            # Start screener update subscription
            await self._start_screener_subscription()
            logger.info("Redis connection initialized for portfolio monitor")

        except Exception as e:
            logger.warning(f"Failed to initialize Redis connection: {e}")
            self._redis = None

    async def _start_screener_subscription(self) -> None:
        """Start subscribing to screener updates."""
        if not self._redis:
            return

        try:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe("screener:updates")

            # Start background task to listen for messages
            self._screener_subscriber_task = asyncio.create_task(
                self._screener_listener()
            )
            logger.info("Started screener update subscription")

        except Exception as e:
            logger.error(f"Failed to start screener subscription: {e}")

    async def _screener_listener(self) -> None:
        """Listen for screener update messages."""
        if not self._pubsub:
            return

        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        await self._handle_screener_update(message)
                    except Exception as e:
                        logger.error(f"Error handling screener update: {e}")
        except Exception as e:
            logger.error(f"Error in screener listener: {e}")

    async def _handle_screener_update(self, message: Dict) -> None:
        """
        Handle incoming screener update messages.

        Args:
            message: Redis message with screener data
        """
        try:
            # Parse screener data
            data = json.loads(message["data"])
            screener_type = data.get("screener_type", "unknown")
            stocks_data = data.get("data", [])

            logger.info(
                f"Received screener update: {screener_type} with {len(stocks_data)} stocks"
            )

            # Update sector cache with new screener data
            updated_symbols = set()
            for stock in stocks_data:
                symbol = stock.get("symbol")
                sector = stock.get("sector")
                if symbol and sector and sector != "Unknown":
                    self._sector_cache[symbol.upper()] = (
                        sector,
                        datetime.now(timezone.utc),
                    )
                    updated_symbols.add(symbol.upper())

            if updated_symbols:
                logger.info(
                    f"Updated sector cache for {len(updated_symbols)} symbols from screener"
                )

            # Trigger portfolio re-analysis if we have cached portfolio
            if self._previous_portfolio and updated_symbols:
                await self._trigger_portfolio_reanalysis(updated_symbols)

            # Call registered callbacks
            for callback in self._screener_callbacks:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in screener callback: {e}")

        except Exception as e:
            logger.error(f"Error processing screener update: {e}")

    async def _trigger_portfolio_reanalysis(self, updated_symbols: Set[str]) -> None:
        """
        Trigger portfolio re-analysis when screener data updates affect current positions.

        Args:
            updated_symbols: Set of symbols that were updated
        """
        if not self._previous_portfolio:
            return

        try:
            # Check if any updated symbols are in current portfolio
            portfolio_symbols = {
                pos.symbol.upper() for pos in self._previous_portfolio.positions
            }
            affected_symbols = updated_symbols.intersection(portfolio_symbols)

            if affected_symbols:
                logger.info(
                    f"Re-analyzing portfolio due to screener updates affecting: {affected_symbols}"
                )

                # Log that we would recalculate diversification metrics
                logger.info("Portfolio re-analysis triggered by screener update")

        except Exception as e:
            logger.error(f"Error in portfolio re-analysis: {e}")

    def register_screener_callback(
        self, callback: Callable[[Dict], Awaitable[None]]
    ) -> None:
        """
        Register a callback for screener updates.

        Args:
            callback: Async callback function
        """
        self._screener_callbacks.append(callback)

    async def shutdown_redis(self) -> None:
        """Shutdown Redis connection and background tasks."""
        try:
            if self._screener_subscriber_task:
                self._screener_subscriber_task.cancel()

            if self._pubsub:
                await self._pubsub.close()

            if self._redis:
                await self._redis.close()

            logger.info("Redis connection closed for portfolio monitor")

        except Exception as e:
            logger.error(f"Error shutting down Redis: {e}")

    async def _calculate_price_correlation(
        self, symbol1: str, symbol2: str
    ) -> Optional[float]:
        """
        Calculate correlation between two symbols using historical price data.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient or None if insufficient data
        """
        # TODO: Add test_calculate_price_correlation() - Verify price correlation calculation
        # TODO: Add test_correlation_edge_cases() - Test insufficient data, identical symbols, invalid data
        try:
            # Load historical data for both symbols
            data1 = await self._load_historical_prices(symbol1, days=60)
            data2 = await self._load_historical_prices(symbol2, days=60)

            if data1 is None or data2 is None:
                return None

            if len(data1) < 20 or len(data2) < 20:  # Need at least 20 data points
                return None

            # Align the data by timestamp and calculate returns
            merged = data1.join(data2, on="timestamp", suffix="_2")
            if len(merged) < 20:
                return None

            # Calculate daily returns
            merged = merged.with_columns(
                [
                    (
                        (pl.col("close") - pl.col("close").shift(1))
                        / pl.col("close").shift(1)
                    ).alias("return1"),
                    (
                        (pl.col("close_2") - pl.col("close_2").shift(1))
                        / pl.col("close_2").shift(1)
                    ).alias("return2"),
                ]
            ).drop_nulls()

            if len(merged) < 15:  # Need minimum data for correlation
                return None

            # Calculate correlation
            correlation = merged.select(pl.corr("return1", "return2")).item()

            # Ensure correlation is within valid range
            if correlation is None or np.isnan(correlation):
                logger.debug(
                    f"Correlation calculation returned NaN for {symbol1}/{symbol2}"
                )
                return 0.0

            final_correlation = max(-1.0, min(1.0, float(correlation)))
            logger.debug(
                f"Calculated price correlation for {symbol1}/{symbol2}: {final_correlation:.3f} (from {len(merged)} data points)"
            )
            return final_correlation

        except Exception as e:
            logger.warning(
                f"Failed to calculate price correlation for {symbol1}/{symbol2}: {e}"
            )
            return None

    async def _load_historical_prices(
        self, symbol: str, days: int = 60
    ) -> Optional[pl.DataFrame]:
        """
        Load historical price data for a symbol.

        Args:
            symbol: Stock symbol
            days: Number of days of data to load

        Returns:
            DataFrame with historical prices or None if not found
        """
        # TODO: Add test_load_historical_prices() - Verify historical data loading
        try:
            market_data_dir = self._data_path / "market_data" / symbol.upper()
            if not market_data_dir.exists():
                return None

            # Try different timeframes, preferring daily data
            timeframes = ["1day", "1hour", "15min"]

            for tf in timeframes:
                tf_dir = market_data_dir / tf
                if not tf_dir.exists():
                    continue

                # Load recent files
                end_date = date.today()
                start_date = end_date - timedelta(days=days)

                dfs = []
                current_date = start_date
                while current_date <= end_date:
                    file_path = tf_dir / f"{current_date.strftime('%Y-%m-%d')}.parquet"
                    if file_path.exists():
                        try:
                            df = pl.read_parquet(file_path)
                            dfs.append(df)
                        except Exception:
                            pass
                    current_date += timedelta(days=1)

                if dfs:
                    combined_df = pl.concat(dfs).sort("timestamp")
                    # For intraday data, sample to daily closes
                    if tf != "1day":
                        combined_df = combined_df.group_by_dynamic(
                            "timestamp", every="1d"
                        ).agg(pl.col("close").last())

                    logger.debug(
                        f"Loaded {len(combined_df)} price records for {symbol} using {tf} timeframe"
                    )
                    return combined_df.select(["timestamp", "close"])

            return None

        except Exception as e:
            logger.warning(f"Failed to load historical prices for {symbol}: {e}")
            return None

    def get_monitoring_statistics(self) -> Dict:
        """Get monitoring statistics."""

        return {
            "portfolio_history_length": len(self.portfolio_history),
            "daily_returns_count": len(self.daily_returns),
            "active_alerts": len(self.risk_alerts),
            "peak_portfolio_value": str(self.peak_portfolio_value),
            "start_portfolio_value": str(self.portfolio_start_value),
            "volatility_cache_size": len(self._volatility_cache),
            "correlation_cache_size": len(self._correlation_cache),
        }
