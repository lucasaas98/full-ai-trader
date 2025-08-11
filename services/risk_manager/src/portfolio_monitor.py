"""
Portfolio Monitor

This module provides real-time portfolio monitoring capabilities including:
- Continuous portfolio state tracking
- Risk metrics calculation and monitoring
- Position-level risk assessment
- Real-time alerts and notifications
- Performance tracking and analytics
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import numpy as np

from shared.config import get_config
from shared.models import (
    Position, PortfolioState, PortfolioMetrics, PositionRisk,
    RiskEventType, RiskSeverity, RiskAlert, RiskLimits
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

    async def monitor_portfolio(self, portfolio: PortfolioState) -> Tuple[PortfolioMetrics, List[RiskAlert]]:
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

            logger.debug(f"Portfolio monitoring complete. Metrics: {metrics.position_count} positions, "
                        f"exposure: {metrics.total_exposure}, alerts: {len(alerts)}")

            return metrics, alerts

        except Exception as e:
            logger.error(f"Error monitoring portfolio: {e}")
            # Return basic metrics on error
            basic_metrics = PortfolioMetrics(
                total_exposure=portfolio.total_market_value,
                cash_percentage=portfolio.cash / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("1"),
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
                volatility=0.15
            )
            return basic_metrics, []

    async def calculate_detailed_metrics(self, portfolio: PortfolioState) -> PortfolioMetrics:
        """Calculate detailed portfolio risk metrics."""

        # Basic metrics
        total_exposure = portfolio.total_market_value
        cash_percentage = portfolio.cash / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("1")
        position_count = len([p for p in portfolio.positions if p.quantity != 0])

        # Advanced risk metrics
        concentration_risk = await self._calculate_concentration_risk(portfolio)
        portfolio_beta = await self._calculate_portfolio_beta(portfolio)
        avg_correlation = await self._calculate_average_correlation(portfolio)

        # VaR calculations
        var_1d, var_5d, expected_shortfall = await self._calculate_var_metrics(portfolio)

        # Performance metrics
        sharpe_ratio = await self._calculate_sharpe_ratio()
        max_drawdown, current_drawdown = await self._calculate_drawdown_metrics(portfolio)
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
            volatility=volatility
        )

    async def calculate_position_risks(self, portfolio: PortfolioState) -> List[PositionRisk]:
        """Calculate risk metrics for individual positions."""
        position_risks = []

        for position in portfolio.positions:
            if position.quantity == 0:
                continue

            try:
                risk = await self._calculate_individual_position_risk(position, portfolio)
                position_risks.append(risk)
            except Exception as e:
                logger.error(f"Error calculating risk for position {position.symbol}: {e}")

        return position_risks

    async def _calculate_individual_position_risk(self, position: Position, portfolio: PortfolioState) -> PositionRisk:
        """Calculate risk metrics for a single position."""

        # Basic metrics
        position_size = abs(position.market_value)
        portfolio_percentage = position_size / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

        # Get volatility and beta
        volatility = await self._get_position_volatility(position.symbol)
        beta = await self._get_position_beta(position.symbol)

        # Calculate VaR for position
        var_1d = position_size * Decimal(str(volatility)) * Decimal("1.645") / Decimal(str(np.sqrt(252)))

        # Expected return (placeholder)
        expected_return = 0.0001  # 0.01% daily

        # Sharpe ratio (placeholder)
        risk_free_rate = 0.0001  # 0.01% daily risk-free rate
        sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0

        # Correlation with portfolio
        correlation_with_portfolio = await self._get_portfolio_correlation(position.symbol, portfolio)

        # Sector classification (placeholder)
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
            risk_score=risk_score
        )

    async def _check_risk_alerts(self, portfolio: PortfolioState, metrics: PortfolioMetrics) -> List[RiskAlert]:
        """Check for conditions that should trigger risk alerts."""
        alerts = []

        # Drawdown alerts
        if metrics.current_drawdown > self.risk_limits.max_drawdown_percentage * Decimal("0.8"):  # 80% of limit
            alert = await self._create_alert(
                RiskEventType.PORTFOLIO_DRAWDOWN,
                RiskSeverity.HIGH,
                "Portfolio Drawdown Warning",
                f"Current drawdown {metrics.current_drawdown:.2%} approaching limit {self.risk_limits.max_drawdown_percentage:.2%}",
                metadata={"current_drawdown": str(metrics.current_drawdown)}
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
                metadata={"concentration_risk": metrics.concentration_risk}
            )
            if alert:
                alerts.append(alert)

        # VaR alerts
        var_percentage = metrics.value_at_risk_1d / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")
        if var_percentage > Decimal("0.05"):  # 5% VaR threshold
            alert = await self._create_alert(
                RiskEventType.VOLATILITY_SPIKE,
                RiskSeverity.MEDIUM,
                "High Value at Risk",
                f"1-day VaR is {var_percentage:.2%} of portfolio",
                metadata={"var_1d": str(metrics.value_at_risk_1d)}
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
                metadata={"position_count": metrics.position_count}
            )
            if alert:
                alerts.append(alert)

        # Check individual position alerts
        position_alerts = await self._check_position_alerts(portfolio)
        alerts.extend(position_alerts)

        return alerts

    async def _check_position_alerts(self, portfolio: PortfolioState) -> List[RiskAlert]:
        """Check for position-specific alerts."""
        alerts = []

        for position in portfolio.positions:
            if position.quantity == 0:
                continue

            # Large position alert
            position_pct = abs(position.market_value) / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")
            if position_pct > self.risk_limits.max_position_percentage * Decimal("0.9"):  # 90% of limit
                alert = await self._create_alert(
                    RiskEventType.POSITION_SIZE_VIOLATION,
                    RiskSeverity.MEDIUM,
                    "Large Position Warning",
                    f"Position {position.symbol} is {position_pct:.2%} of portfolio",
                    symbol=position.symbol,
                    metadata={"position_percentage": str(position_pct)}
                )
                if alert:
                    alerts.append(alert)

            # Unrealized loss alert
            loss_pct = position.unrealized_pnl / abs(position.cost_basis) if position.cost_basis != 0 else Decimal("0")
            if loss_pct <= -self.risk_limits.stop_loss_percentage * Decimal("0.8"):  # 80% of stop loss
                alert = await self._create_alert(
                    RiskEventType.STOP_LOSS_TRIGGERED,
                    RiskSeverity.HIGH,
                    "Position Loss Warning",
                    f"Position {position.symbol} down {loss_pct:.2%}",
                    symbol=position.symbol,
                    metadata={"unrealized_loss": str(loss_pct)}
                )
                if alert:
                    alerts.append(alert)

        return alerts

    async def _create_alert(self,
                          event_type: RiskEventType,
                          severity: RiskSeverity,
                          title: str,
                          message: str,
                          symbol: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> Optional[RiskAlert]:
        """Create a risk alert if not in cooldown period."""

        alert_key = f"{event_type}_{symbol or 'portfolio'}"

        # Check cooldown
        if alert_key in self.last_alert_times:
            time_since_last = datetime.now(timezone.utc) - self.last_alert_times[alert_key]
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
            metadata=metadata or {}
        )

        # Update cooldown tracker
        self.last_alert_times[alert_key] = datetime.now(timezone.utc)

        return alert

    async def _update_portfolio_history(self, portfolio: PortfolioState):
        """Update portfolio historical data."""

        # Add current snapshot
        self.portfolio_history.append((datetime.now(timezone.utc), portfolio))

        # Trim history to max days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.max_history_days)
        self.portfolio_history = [
            (timestamp, state) for timestamp, state in self.portfolio_history
            if timestamp > cutoff_date
        ]

        # Update daily returns if we have previous day data
        await self._update_daily_returns(portfolio)

    async def _update_daily_returns(self, portfolio: PortfolioState):
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
            daily_return = float((portfolio.total_equity - previous_value) / previous_value)
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
                for symbol2 in symbols[i+1:]:
                    correlation = await self._get_symbol_correlation(symbol1, symbol2)
                    correlations.append(correlation)

            return float(np.mean(correlations)) if correlations else 0.0

        except Exception as e:
            logger.error(f"Error calculating average correlation: {e}")
            return 0.0

    async def _calculate_var_metrics(self, portfolio: PortfolioState) -> Tuple[Decimal, Decimal, Decimal]:
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
                expected_shortfall = portfolio_value * Decimal(str(abs(np.mean(tail_returns))))
            else:
                expected_shortfall = var_1d * Decimal("1.3")

            return var_1d, var_5d, expected_shortfall

        except Exception as e:
            logger.error(f"Error calculating VaR metrics: {e}")
            return await self._parametric_var(portfolio)

    async def _parametric_var(self, portfolio: PortfolioState) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate parametric VaR using normal distribution assumption."""

        try:
            portfolio_value = portfolio.total_equity
            volatility = await self._calculate_portfolio_volatility(portfolio)

            # 95% confidence level (1.645 standard deviations)
            daily_vol = volatility / np.sqrt(252)  # Annualized to daily
            var_1d = portfolio_value * Decimal(str(daily_vol * 1.645))
            var_5d = var_1d * Decimal(str(np.sqrt(5)))

            # Expected Shortfall for normal distribution
            expected_shortfall = var_1d * Decimal("1.282")  # E[X|X<-1.645σ] for normal dist

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

            if np.std(excess_returns) > 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
                # Annualize
                return sharpe_ratio * np.sqrt(252)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    async def _calculate_drawdown_metrics(self, portfolio: PortfolioState) -> Tuple[Decimal, Decimal]:
        """Calculate maximum and current drawdown."""

        try:
            current_value = portfolio.total_equity

            # Update peak value
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value

            # Calculate current drawdown
            if self.peak_portfolio_value > 0:
                current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
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

    def _update_peak_values(self, portfolio: PortfolioState):
        """Update peak portfolio values for drawdown tracking."""

        if portfolio.total_equity > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio.total_equity
            logger.debug(f"New portfolio peak: ${self.peak_portfolio_value}")

        # Set initial portfolio value if not set
        if self.portfolio_start_value == Decimal("0"):
            self.portfolio_start_value = portfolio.total_equity

    async def _get_position_volatility(self, symbol: str) -> float:
        """Get volatility for a specific position."""
        try:
            # In production, calculate from historical price data
            # Placeholder implementation based on symbol characteristics

            if symbol.startswith(('SPY', 'QQQ', 'IWM')):
                return 0.12  # ETFs - lower volatility
            elif len(symbol) <= 3:
                return 0.18  # Large cap stocks
            elif len(symbol) == 4:
                return 0.25  # Mid cap stocks
            else:
                return 0.35  # Small cap or specialized stocks

        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.20

    async def _get_position_beta(self, symbol: str) -> float:
        """Get beta for a specific position."""
        try:
            # In production, calculate from historical data vs market index
            # Placeholder implementation

            if symbol.startswith('SPY'):
                return 1.0  # Market ETF
            elif symbol.startswith('QQQ'):
                return 1.2  # Tech ETF - higher beta
            elif symbol.startswith('IWM'):
                return 1.1  # Small cap ETF
            else:
                return 1.0  # Default beta

        except Exception as e:
            logger.error(f"Error getting beta for {symbol}: {e}")
            return 1.0

    async def _get_portfolio_correlation(self, symbol: str, portfolio: PortfolioState) -> float:
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
                    correlation = await self._get_symbol_correlation(symbol, position.symbol)
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
            # In production, calculate from historical price data
            # Placeholder implementation based on symbol similarity
            correlation = await self._estimate_correlation(symbol1, symbol2)

            # Cache result
            self._correlation_cache[cache_key] = (correlation, datetime.now(timezone.utc))

            return correlation

        except Exception as e:
            logger.error(f"Error getting correlation between {symbol1} and {symbol2}: {e}")
            return 0.0

    async def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Estimate correlation between symbols (placeholder implementation)."""

        # This is a simplified estimation - in production you'd use historical data

        # Same sector/industry correlation
        if await self._same_sector(symbol1, symbol2):
            return 0.6

        # ETF correlations
        if symbol1.startswith(('SPY', 'QQQ', 'IWM')) and symbol2.startswith(('SPY', 'QQQ', 'IWM')):
            return 0.8

        # Tech stocks correlation
        if await self._is_tech_stock(symbol1) and await self._is_tech_stock(symbol2):
            return 0.5

        # Default low correlation
        return 0.2

    async def _same_sector(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are in the same sector."""
        # Placeholder implementation
        sector1 = await self._get_position_sector(symbol1)
        sector2 = await self._get_position_sector(symbol2)
        return sector1 == sector2 and sector1 is not None

    async def _is_tech_stock(self, symbol: str) -> bool:
        """Check if symbol is a tech stock."""
        # Placeholder implementation
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        return symbol in tech_symbols

    async def _get_position_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a position."""
        # Placeholder implementation - in production, fetch from financial data provider
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'JPM': 'Financials',
            'BAC': 'Financials',
            'SPY': 'ETF',
            'QQQ': 'ETF',
            'IWM': 'ETF'
        }
        return sector_map.get(symbol, 'Unknown')

    async def _calculate_position_risk_score(self,
                                           volatility: float,
                                           portfolio_percentage: float,
                                           correlation: float,
                                           beta: float) -> float:
        """Calculate overall risk score for a position (0-10 scale)."""

        try:
            # Risk components (0-10 scale each)

            # Volatility risk (0-10)
            vol_risk = min(10, (volatility / 0.5) * 10)  # Scale where 50% vol = 10

            # Size risk (0-10)
            size_risk = min(10, (portfolio_percentage / 0.3) * 10)  # Scale where 30% = 10

            # Correlation risk (0-10)
            corr_risk = correlation * 10  # Direct scaling

            # Beta risk (0-10)
            beta_risk = min(10, abs(beta - 1.0) * 5)  # Distance from market beta

            # Weighted average (volatility and size are most important)
            risk_score = (
                vol_risk * 0.4 +      # 40% weight
                size_risk * 0.3 +     # 30% weight
                corr_risk * 0.2 +     # 20% weight
                beta_risk * 0.1       # 10% weight
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
                "pattern_day_trader": portfolio.pattern_day_trader
            }

            # Add position details
            positions_summary = []
            for position in active_positions:
                pos_summary = {
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "market_value": str(position.market_value),
                    "unrealized_pnl": str(position.unrealized_pnl),
                    "percentage": str(abs(position.market_value) / portfolio.total_equity * 100) if portfolio.total_equity > 0 else "0"
                }
                positions_summary.append(pos_summary)

            summary["positions"] = positions_summary

            return summary

        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {
                "error": "Failed to generate portfolio summary",
                "total_equity": str(portfolio.total_equity),
                "position_count": len(portfolio.positions)
            }

    async def check_portfolio_health(self, portfolio: PortfolioState) -> Dict[str, bool]:
        """Perform comprehensive portfolio health checks."""

        health_status = {
            "overall_healthy": True,
            "within_risk_limits": True,
            "positions_under_limit": True,
            "drawdown_acceptable": True,
            "correlation_acceptable": True,
            "volatility_acceptable": True
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
                    position_pct = abs(position.market_value) / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")
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

    def get_risk_exposure_by_sector(self, portfolio: PortfolioState) -> Dict[str, Decimal]:
        """Calculate risk exposure by sector."""

        sector_exposure = {}

        try:
            if portfolio.total_equity <= 0:
                return sector_exposure

            for position in portfolio.positions:
                if position.quantity == 0:
                    continue

                # Get sector (placeholder implementation)
                sector = self._get_position_sector_sync(position.symbol)
                exposure = abs(position.market_value) / portfolio.total_equity

                if sector in sector_exposure:
                    sector_exposure[sector] += exposure
                else:
                    sector_exposure[sector] = exposure

            return sector_exposure

        except Exception as e:
            logger.error(f"Error calculating sector exposure: {e}")
            return {}

    def _get_position_sector_sync(self, symbol: str) -> str:
        """Synchronous version of sector lookup."""
        sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'JPM': 'Financials',
            'BAC': 'Financials',
            'SPY': 'ETF',
            'QQQ': 'ETF',
            'IWM': 'ETF'
        }
        return sector_map.get(symbol, 'Unknown')

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
                    position_pct = abs(position.market_value) / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")
                    if position_pct > self.risk_limits.max_position_percentage * Decimal("0.8"):
                        warnings.append(f"Large position {position.symbol}: {position_pct:.1%}")

            # Check drawdown
            _, current_drawdown = await self._calculate_drawdown_metrics(portfolio)
            if current_drawdown > self.risk_limits.max_drawdown_percentage * Decimal("0.7"):
                warnings.append(f"High drawdown: {current_drawdown:.1%}")

            # Check correlation
            avg_correlation = await self._calculate_average_correlation(portfolio)
            if avg_correlation > self.risk_limits.max_correlation_threshold * 0.8:
                warnings.append(f"High average correlation: {avg_correlation:.2f}")

            return warnings

        except Exception as e:
            logger.error(f"Error generating risk warnings: {e}")
            return ["Error generating risk warnings"]

    def reset_monitoring_state(self):
        """Reset monitoring state (for new trading session)."""

        self.daily_returns.clear()
        self.risk_alerts.clear()
        self.last_alert_times.clear()

        logger.info("Portfolio monitoring state reset")

    def get_monitoring_statistics(self) -> Dict:
        """Get monitoring statistics."""

        return {
            "portfolio_history_length": len(self.portfolio_history),
            "daily_returns_count": len(self.daily_returns),
            "active_alerts": len(self.risk_alerts),
            "peak_portfolio_value": str(self.peak_portfolio_value),
            "start_portfolio_value": str(self.portfolio_start_value),
            "volatility_cache_size": len(self._volatility_cache),
            "correlation_cache_size": len(self._correlation_cache)
        }
