"""
Risk Management Service

This module provides comprehensive risk management capabilities including:
- Position sizing with confidence-based adjustments
- Real-time risk monitoring and filtering
- Portfolio metrics calculation
- Circuit breakers and emergency stops
- Position management with stop losses and trailing stops
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import numpy as np
import asyncio
import polars as pl

from shared.config import get_config
from shared.models import (
    Position, PortfolioState, PortfolioMetrics, TradeSignal, OrderRequest, OrderSide,
    RiskEvent, RiskEventType, RiskSeverity, PositionSizing, PositionSizingMethod,
    RiskFilter, TrailingStop, RiskLimits, TimeFrame
)

logger = logging.getLogger(__name__)


class RiskManager:
    """Main risk management service."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize risk manager."""
        self.config = get_config()
        self.risk_limits = RiskLimits(**(config or {}))
        self.trailing_stops: Dict[str, TrailingStop] = {}
        self.daily_pnl = Decimal("0")
        self.daily_trade_count = 0
        self.emergency_stop_active = False
        self.last_portfolio_snapshot: Optional[PortfolioState] = None
        self.position_correlations: Dict[str, float] = {}

        # Risk metrics cache
        self._portfolio_metrics_cache: Optional[PortfolioMetrics] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=1)

        # Data access
        self._data_store = None
        self._data_client = None
        self._price_cache: Dict[str, Tuple[datetime, float]] = {}
        self._correlation_cache: Dict[str, Tuple[datetime, float]] = {}
        self._volatility_cache: Dict[str, Tuple[datetime, float]] = {}
        self._atr_cache: Dict[str, Tuple[datetime, float]] = {}

        # Initialize data services
        asyncio.create_task(self._initialize_data_services())

    async def validate_trade_request(self,
                                   order_request: OrderRequest,
                                   portfolio: PortfolioState,
                                   signal: Optional[TradeSignal] = None) -> Tuple[bool, List[RiskFilter]]:
        """
        Validate a trade request against all risk filters.

        Args:
            order_request: The proposed trade order
            portfolio: Current portfolio state
            signal: Optional trade signal with confidence

        Returns:
            Tuple of (is_valid, list_of_failed_filters)
        """
        filters = []

        try:
            # Emergency stop check
            emergency_filter = await self._check_emergency_stop(portfolio)
            filters.append(emergency_filter)
            if not emergency_filter.passed:
                return False, filters

            # Daily loss limit check
            daily_loss_filter = await self._check_daily_loss_limit(portfolio)
            filters.append(daily_loss_filter)

            # Position limit check
            position_limit_filter = await self._check_position_limits(portfolio, order_request)
            filters.append(position_limit_filter)

            # Buying power check
            buying_power_filter = await self._check_buying_power(portfolio, order_request)
            filters.append(buying_power_filter)

            # Position size check
            position_size_filter = await self._check_position_size(portfolio, order_request)
            filters.append(position_size_filter)

            # Correlation check (for new positions)
            if order_request.side == OrderSide.BUY:
                correlation_filter = await self._check_correlation(portfolio, order_request.symbol)
                filters.append(correlation_filter)

            # Volatility check
            volatility_filter = await self._check_volatility(order_request.symbol)
            filters.append(volatility_filter)

            # Check if all filters passed
            all_passed = all(f.passed for f in filters)

            # Log failed filters
            failed_filters = [f for f in filters if not f.passed]
            if failed_filters:
                logger.warning(f"Trade validation failed for {order_request.symbol}: "
                             f"{[f.filter_name for f in failed_filters]}")

            return all_passed, filters

        except Exception as e:
            logger.error(f"Error validating trade request: {e}")
            # Fail safe - reject trade on error
            error_filter = RiskFilter(
                name="validation_error",
                max_position_size=None,
                max_sector_exposure=None,
                min_liquidity=None,
                max_volatility=None,
                passed=False,
                filter_name="validation_error",
                reason=f"Validation error: {str(e)}",
                value=None,
                limit=None,
                severity=RiskSeverity.HIGH
            )
            return False, [error_filter]

    async def calculate_position_size(self,
                                    symbol: str,
                                    current_price: Decimal,
                                    portfolio: PortfolioState,
                                    confidence_score: Optional[float] = None,
                                    signal: Optional[TradeSignal] = None) -> PositionSizing:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            symbol: Trading symbol
            current_price: Current stock price
            portfolio: Current portfolio state
            confidence_score: AI confidence score (0-1)
            signal: Optional trade signal

        Returns:
            PositionSizing calculation result
        """
        try:
            # Base position size (default 20% of portfolio)
            base_percentage = self.risk_limits.max_position_percentage

            # Adjust based on confidence score
            confidence_adjustment = 1.0
            if confidence_score is not None:
                # Scale from 0.5x to 1.5x based on confidence
                confidence_adjustment = 0.5 + (confidence_score * 1.0)

            # Get volatility adjustment
            volatility_adjustment = await self._calculate_volatility_adjustment(symbol)

            # Calculate adjusted position percentage
            adjusted_percentage = base_percentage * Decimal(str(confidence_adjustment)) * Decimal(str(volatility_adjustment))

            # Ensure we don't exceed limits
            adjusted_percentage = min(adjusted_percentage, self.risk_limits.max_position_percentage)

            # Calculate position value
            available_capital = portfolio.total_equity
            position_value = available_capital * adjusted_percentage

            # Calculate number of shares
            shares = int(position_value / current_price)
            actual_position_value = Decimal(shares) * current_price
            actual_percentage = actual_position_value / available_capital if available_capital > 0 else Decimal("0")

            # Calculate risk metrics
            stop_loss_price = current_price * (Decimal("1") - self.risk_limits.stop_loss_percentage)
            max_loss = (current_price - stop_loss_price) * Decimal(shares)

            take_profit_price = current_price * (Decimal("1") + self.risk_limits.take_profit_percentage)
            max_gain = (take_profit_price - current_price) * Decimal(shares)

            risk_reward_ratio = float(max_gain / max_loss) if max_loss > 0 else 0.0

            return PositionSizing(
                symbol=symbol,
                recommended_shares=shares,
                recommended_value=actual_position_value,
                position_percentage=actual_percentage,
                confidence_adjustment=confidence_adjustment,
                volatility_adjustment=volatility_adjustment,
                sizing_method=PositionSizingMethod.CONFIDENCE_BASED,
                max_loss_amount=max_loss,
                risk_reward_ratio=risk_reward_ratio
            )

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            # Return minimal safe position as fallback
            safe_shares = max(1, int((portfolio.total_equity * Decimal("0.01")) / current_price))
            return PositionSizing(
                symbol=symbol,
                recommended_shares=safe_shares,
                recommended_value=Decimal(safe_shares) * current_price,
                position_percentage=Decimal("0.01"),
                confidence_adjustment=0.5,
                volatility_adjustment=0.5,
                sizing_method=PositionSizingMethod.FIXED_PERCENTAGE,
                max_loss_amount=Decimal(safe_shares) * current_price * Decimal("0.02"),
                risk_reward_ratio=1.5
            )

    async def calculate_portfolio_metrics(self, portfolio: PortfolioState) -> PortfolioMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            # Check cache
            if (self._portfolio_metrics_cache and
                self._cache_timestamp and
                datetime.now(timezone.utc) - self._cache_timestamp < self._cache_ttl):
                return self._portfolio_metrics_cache

            # Calculate basic metrics
            total_exposure = portfolio.total_market_value
            cash_percentage = portfolio.cash / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("1")
            position_count = len([p for p in portfolio.positions if p.quantity != 0])

            # Calculate concentration risk
            concentration_risk = await self._calculate_concentration_risk(portfolio)

            # Calculate portfolio beta and correlation
            portfolio_beta, avg_correlation = await self._calculate_portfolio_beta_correlation(portfolio)

            # Calculate VaR and Expected Shortfall
            var_1d, var_5d, expected_shortfall = await self._calculate_var_metrics(portfolio)

            # Calculate performance metrics
            sharpe_ratio = await self._calculate_sharpe_ratio(portfolio)
            max_drawdown, current_drawdown = await self._calculate_drawdown_metrics(portfolio)
            volatility = await self._calculate_portfolio_volatility(portfolio)

            metrics = PortfolioMetrics(
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

            # Cache the result
            self._portfolio_metrics_cache = metrics
            self._cache_timestamp = datetime.now(timezone.utc)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            # Return default metrics on error
            return PortfolioMetrics(
                total_exposure=portfolio.total_market_value,
                cash_percentage=Decimal("1"),
                position_count=len(portfolio.positions),
                concentration_risk=0.0,
                portfolio_beta=1.0,
                portfolio_correlation=0.0,
                value_at_risk_1d=Decimal("0"),
                value_at_risk_5d=Decimal("0"),
                expected_shortfall=Decimal("0"),
                sharpe_ratio=0.0,
                max_drawdown=Decimal("0"),
                current_drawdown=Decimal("0"),
                volatility=0.0
            )

    async def update_trailing_stops(self, portfolio: PortfolioState, market_prices: Dict[str, Decimal]) -> List[RiskEvent]:
        """Update trailing stops for all positions."""
        events = []

        for position in portfolio.positions:
            if position.quantity <= 0:  # Only for long positions
                continue

            symbol = position.symbol
            current_price = market_prices.get(symbol, position.current_price)

            # Initialize trailing stop if not exists
            if symbol not in self.trailing_stops:
                self.trailing_stops[symbol] = TrailingStop(
                    symbol=symbol,
                    enabled=True,
                    trail_percentage=self.risk_limits.stop_loss_percentage,
                    current_stop_price=current_price * (Decimal("1") - self.risk_limits.stop_loss_percentage),
                    highest_price=current_price,
                    entry_price=position.entry_price
                )

            trailing_stop = self.trailing_stops[symbol]

            # Update highest price
            if current_price > trailing_stop.highest_price:
                trailing_stop.highest_price = current_price
                new_stop_price = current_price * (Decimal("1") - trailing_stop.trail_percentage)

                # Only move stop up, never down
                if new_stop_price > trailing_stop.current_stop_price:
                    old_stop = trailing_stop.current_stop_price
                    trailing_stop.current_stop_price = new_stop_price
                    trailing_stop.last_updated = datetime.now(timezone.utc)

                    logger.info(f"Updated trailing stop for {symbol}: {old_stop} -> {new_stop_price}")

            # Check if stop loss should be triggered
            if current_price <= trailing_stop.current_stop_price:
                event = RiskEvent(
                    event_type=RiskEventType.STOP_LOSS_TRIGGERED,
                    severity=RiskSeverity.HIGH,
                    symbol=symbol,
                    description=f"Trailing stop triggered for {symbol} at {current_price}",
                    resolved_at=None,
                    action_taken="sell_position",
                    metadata={
                        "stop_price": str(trailing_stop.current_stop_price),
                        "current_price": str(current_price),
                        "position_size": str(position.quantity),
                        "unrealized_pnl": str(position.unrealized_pnl)
                    }
                )
                events.append(event)

        return events

    async def check_risk_violations(self, portfolio: PortfolioState) -> List[RiskEvent]:
        """Check for any risk violations in the current portfolio."""
        violations = []

        try:
            # Calculate portfolio metrics
            metrics = await self.calculate_portfolio_metrics(portfolio)

            # Check drawdown limits
            if metrics.current_drawdown > self.risk_limits.max_drawdown_percentage:
                violations.append(RiskEvent(
                    event_type=RiskEventType.PORTFOLIO_DRAWDOWN,
                    severity=RiskSeverity.CRITICAL,
                    symbol=None,
                    description=f"Portfolio drawdown {metrics.current_drawdown:.2%} exceeds limit {self.risk_limits.max_drawdown_percentage:.2%}",
                    resolved_at=None,
                    action_taken=None,
                    metadata={"current_drawdown": str(metrics.current_drawdown)}
                ))

            # Check daily loss limit
            daily_loss_check = await self._check_daily_loss_limit(portfolio)
            if not daily_loss_check.passed:
                violations.append(RiskEvent(
                    event_type=RiskEventType.DAILY_LOSS_LIMIT,
                    severity=RiskSeverity.HIGH,
                    symbol=None,
                    description=daily_loss_check.reason or "Daily loss limit exceeded",
                    resolved_at=None,
                    action_taken=None
                ))

            # Check position concentration
            if metrics.concentration_risk > 0.8:  # High concentration threshold
                violations.append(RiskEvent(
                    event_type=RiskEventType.POSITION_SIZE_VIOLATION,
                    severity=RiskSeverity.MEDIUM,
                    symbol=None,
                    description=f"High portfolio concentration risk: {metrics.concentration_risk:.2f}",
                    resolved_at=None,
                    action_taken=None,
                    metadata={"concentration_risk": metrics.concentration_risk}
                ))

            # Check correlation violations
            high_correlation_pairs = await self._find_high_correlation_pairs(portfolio)
            for pair, correlation in high_correlation_pairs:
                violations.append(RiskEvent(
                    event_type=RiskEventType.CORRELATION_BREACH,
                    severity=RiskSeverity.MEDIUM,
                    symbol=None,
                    description=f"High correlation between {pair[0]} and {pair[1]}: {correlation:.2f}",
                    resolved_at=None,
                    action_taken=None,
                    metadata={"pair": pair, "correlation": correlation}
                ))

            return violations

        except Exception as e:
            logger.error(f"Error checking risk violations: {e}")
            return [RiskEvent(
                event_type=RiskEventType.EMERGENCY_STOP,
                severity=RiskSeverity.CRITICAL,
                symbol=None,
                description=f"Risk check error: {str(e)}",
                resolved_at=None,
                action_taken=None,
                metadata={"error": str(e)}
            )]

    async def calculate_stop_loss_take_profit(self,
                                            symbol: str,
                                            entry_price: Decimal,
                                            side: OrderSide,
                                            atr_multiplier: Optional[float] = None) -> Tuple[Decimal, Decimal]:
        """
        Calculate stop loss and take profit levels.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: Order side (buy/sell)
            atr_multiplier: Optional ATR multiplier for volatility-based stops

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        try:
            if atr_multiplier:
                # Volatility-based stops using ATR
                atr = await self._get_atr(symbol)
                stop_distance = Decimal(str(atr * atr_multiplier))
                profit_distance = stop_distance * Decimal("1.5")  # 1.5:1 risk/reward
            else:
                # Fixed percentage stops
                stop_distance = entry_price * self.risk_limits.stop_loss_percentage
                profit_distance = entry_price * self.risk_limits.take_profit_percentage

            if side == OrderSide.BUY:
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + profit_distance
            else:  # SHORT
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - profit_distance

            # Ensure prices are positive
            stop_loss = max(stop_loss, Decimal("0.01"))
            take_profit = max(take_profit, Decimal("0.01"))

            logger.info(f"Calculated stops for {symbol}: SL={stop_loss}, TP={take_profit}")

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error calculating stops for {symbol}: {e}")
            # Return conservative default stops
            if side == OrderSide.BUY:
                return (entry_price * Decimal("0.98"), entry_price * Decimal("1.03"))
            else:
                return (entry_price * Decimal("1.02"), entry_price * Decimal("0.97"))

    async def should_scale_out_position(self, position: Position, current_price: Decimal) -> Tuple[bool, float]:
        """
        Determine if a position should be partially closed (scaled out).

        Args:
            position: Current position
            current_price: Current market price

        Returns:
            Tuple of (should_scale_out, scale_out_percentage)
        """
        try:
            if position.quantity <= 0:
                return False, 0.0

            # Calculate current profit percentage
            profit_pct = (current_price - position.entry_price) / position.entry_price

            # Scale out rules
            if profit_pct >= Decimal("0.10"):  # 10% profit
                return True, 0.5  # Take 50% off
            elif profit_pct >= Decimal("0.06"):  # 6% profit
                return True, 0.25  # Take 25% off
            elif profit_pct >= Decimal("0.03"):  # 3% profit (first take profit)
                return True, 0.33  # Take 33% off

            return False, 0.0

        except Exception as e:
            logger.error(f"Error checking scale out for {position.symbol}: {e}")
            return False, 0.0

    # Private helper methods

    async def _check_emergency_stop(self, portfolio: PortfolioState) -> RiskFilter:
        """Check if emergency stop should be triggered."""
        if self.emergency_stop_active:
            return RiskFilter(
                name="emergency_stop",
                max_position_size=None,
                max_sector_exposure=None,
                min_liquidity=None,
                max_volatility=None,
                passed=False,
                filter_name="emergency_stop",
                reason="Trading halted due to emergency stop",
                value=None,
                limit=None,
                severity=RiskSeverity.CRITICAL
            )

        # Check if portfolio loss exceeds emergency threshold
        if self.last_portfolio_snapshot:
            loss_pct = (portfolio.total_equity - self.last_portfolio_snapshot.total_equity) / self.last_portfolio_snapshot.total_equity
            if loss_pct <= -self.risk_limits.emergency_stop_percentage:
                self.emergency_stop_active = True
                return RiskFilter(
                    name="emergency_stop",
                    max_position_size=None,
                    max_sector_exposure=None,
                    min_liquidity=None,
                    max_volatility=None,
                    passed=False,
                    filter_name="emergency_stop",
                    reason=f"Portfolio loss {loss_pct:.2%} exceeds emergency threshold {self.risk_limits.emergency_stop_percentage:.2%}",
                    value=float(loss_pct),
                    limit=float(-self.risk_limits.emergency_stop_percentage),
                    severity=RiskSeverity.CRITICAL
                )

        return RiskFilter(
            name="emergency_stop",
            max_position_size=None,
            max_sector_exposure=None,
            min_liquidity=None,
            max_volatility=None,
            passed=True,
            filter_name="emergency_stop",
            reason=None,
            value=None,
            limit=None
        )

    async def _check_daily_loss_limit(self, portfolio: PortfolioState) -> RiskFilter:
        """Check daily loss limit."""
        if self.last_portfolio_snapshot:
            daily_loss = portfolio.total_equity - self.last_portfolio_snapshot.total_equity
            daily_loss_pct = daily_loss / self.last_portfolio_snapshot.total_equity if self.last_portfolio_snapshot.total_equity > 0 else Decimal("0")

            if daily_loss_pct <= -self.risk_limits.max_daily_loss_percentage:
                return RiskFilter(
                    name="daily_loss_limit",
                    max_position_size=None,
                    max_sector_exposure=None,
                    min_liquidity=None,
                    max_volatility=None,
                    passed=False,
                    filter_name="daily_loss_limit",
                    reason=f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.risk_limits.max_daily_loss_percentage:.2%}",
                    value=float(daily_loss_pct),
                    limit=float(-self.risk_limits.max_daily_loss_percentage),
                    severity=RiskSeverity.HIGH
                )

        return RiskFilter(
            name="daily_loss_limit",
            max_position_size=None,
            max_sector_exposure=None,
            min_liquidity=None,
            max_volatility=None,
            passed=True,
            filter_name="daily_loss_limit",
            reason=None,
            value=None,
            limit=None
        )

    async def _check_position_limits(self, portfolio: PortfolioState, order_request: OrderRequest) -> RiskFilter:
        """Check position count limits."""
        current_positions = len([p for p in portfolio.positions if p.quantity != 0])

        # If buying and would create new position
        if order_request.side == OrderSide.BUY:
            existing_position = next((p for p in portfolio.positions if p.symbol == order_request.symbol), None)
            if not existing_position or existing_position.quantity == 0:
                if current_positions >= self.risk_limits.max_positions:
                    return RiskFilter(
                        name="position_count",
                        max_position_size=None,
                        max_sector_exposure=None,
                        min_liquidity=None,
                        max_volatility=None,
                        passed=False,
                        filter_name="position_count",
                        reason=f"Position count would exceed limit: {current_positions + 1} > {self.risk_limits.max_positions}",
                        value=current_positions + 1,
                        limit=self.risk_limits.max_positions,
                        severity=RiskSeverity.MEDIUM
                    )

        return RiskFilter(
            name="position_count",
            max_position_size=None,
            max_sector_exposure=None,
            min_liquidity=None,
            max_volatility=None,
            passed=True,
            filter_name="position_count",
            reason=None,
            value=None,
            limit=None
        )

    async def _check_buying_power(self, portfolio: PortfolioState, order_request: OrderRequest) -> RiskFilter:
        """Check sufficient buying power."""
        required_capital = order_request.quantity * order_request.price if order_request.price else Decimal("0")

        if order_request.side == OrderSide.BUY and required_capital > portfolio.buying_power:
            return RiskFilter(
                name="buying_power",
                max_position_size=None,
                max_sector_exposure=None,
                min_liquidity=None,
                max_volatility=None,
                passed=False,
                filter_name="buying_power",
                reason=f"Insufficient buying power: ${required_capital} > ${portfolio.buying_power}",
                value=float(required_capital),
                limit=float(portfolio.buying_power),
                severity=RiskSeverity.MEDIUM
            )

        return RiskFilter(
            name="buying_power",
            max_position_size=None,
            max_sector_exposure=None,
            min_liquidity=None,
            max_volatility=None,
            passed=True,
            filter_name="buying_power",
            reason=None,
            value=None,
            limit=None
        )

    async def _check_position_size(self, portfolio: PortfolioState, order_request: OrderRequest) -> RiskFilter:
        """Check position size limits."""
        position_value = order_request.quantity * order_request.price if order_request.price else Decimal("0")
        position_pct = position_value / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

        if position_pct > self.risk_limits.max_position_percentage:
            return RiskFilter(
                name="position_size",
                max_position_size=None,
                max_sector_exposure=None,
                min_liquidity=None,
                max_volatility=None,
                passed=False,
                filter_name="position_size",
                reason=f"Position size {position_pct:.2%} exceeds limit {self.risk_limits.max_position_percentage:.2%}",
                value=float(position_pct),
                limit=float(self.risk_limits.max_position_percentage),
                severity=RiskSeverity.MEDIUM
            )

        return RiskFilter(
            name="position_size",
            max_position_size=None,
            max_sector_exposure=None,
            min_liquidity=None,
            max_volatility=None,
            passed=True,
            filter_name="position_size",
            reason=None,
            value=None,
            limit=None
        )

    async def _check_correlation(self, portfolio: PortfolioState, new_symbol: str) -> RiskFilter:
        """Check correlation with existing positions."""
        try:
            correlations = await self._calculate_symbol_correlations(portfolio, new_symbol)

            high_correlations = [corr for corr in correlations.values() if corr > self.risk_limits.max_correlation_threshold]

            if high_correlations:

                return RiskFilter(
                    name="correlation",
                    max_position_size=None,
                    max_sector_exposure=None,
                    min_liquidity=None,
                    max_volatility=None,
                    passed=False,
                    filter_name="correlation",
                    reason="High correlation with existing positions",
                    value=max(high_correlations),
                    limit=self.risk_limits.max_correlation_threshold,
                    severity=RiskSeverity.MEDIUM
                )

            return RiskFilter(
                name="correlation",
                max_position_size=None,
                max_sector_exposure=None,
                min_liquidity=None,
                max_volatility=None,
                passed=True,
                filter_name="correlation",
                reason=None,
                value=None,
                limit=None
            )

        except Exception as e:
            logger.warning(f"Error checking correlation for {new_symbol}: {e}")
            # Pass on error to avoid blocking trades
            return RiskFilter(
                name="correlation",
                max_position_size=None,
                max_sector_exposure=None,
                min_liquidity=None,
                max_volatility=None,
                passed=True,
                filter_name="correlation",
                reason=None,
                value=None,
                limit=None
            )

    async def _check_volatility(self, symbol: str) -> RiskFilter:
        """Check if symbol volatility is within acceptable limits."""
        try:
            volatility = await self._get_symbol_volatility(symbol)

            if volatility > self.risk_limits.max_position_volatility:
                return RiskFilter(
                    name="volatility_limit",
                    max_position_size=None,
                    max_sector_exposure=None,
                    min_liquidity=None,
                    max_volatility=None,
                    passed=False,
                    filter_name="volatility_limit",
                    reason=f"Symbol volatility {volatility:.2%} exceeds limit {self.risk_limits.max_position_volatility:.2%}",
                    value=volatility,
                    limit=self.risk_limits.max_position_volatility,
                    severity=RiskSeverity.MEDIUM
                )

            return RiskFilter(
                name="volatility_limit",
                max_position_size=None,
                max_sector_exposure=None,
                min_liquidity=None,
                max_volatility=None,
                passed=True,
                filter_name="volatility_limit",
                reason=None,
                value=None,
                limit=None
            )

        except Exception as e:
            logger.warning(f"Error checking volatility for {symbol}: {e}")
            # Pass on error to avoid blocking trades
            return RiskFilter(
                name="volatility_limit",
                max_position_size=None,
                max_sector_exposure=None,
                min_liquidity=None,
                max_volatility=None,
                passed=True,
                filter_name="volatility_limit",
                reason=None,
                value=None,
                limit=None
            )

    async def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility-based position size adjustment."""
        try:
            volatility = await self._get_symbol_volatility(symbol)

            # Inverse relationship: higher volatility = smaller position
            # Normalize volatility (assume 0.15 = normal volatility)
            normal_vol = 0.15
            vol_ratio = volatility / normal_vol

            # Adjustment factor: 0.5x to 1.5x based on volatility
            adjustment = max(0.5, min(1.5, 1.0 / vol_ratio))

            return adjustment

        except Exception as e:
            logger.warning(f"Error calculating volatility adjustment for {symbol}: {e}")
            return 1.0  # Default adjustment

    async def _calculate_concentration_risk(self, portfolio: PortfolioState) -> float:
        """Calculate portfolio concentration risk using Herfindahl index."""
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

        # Normalize to 0-1 scale (1 = maximum concentration)
        return min(1.0, hhi)

    async def _calculate_portfolio_beta_correlation(self, portfolio: PortfolioState) -> Tuple[float, float]:
        """Calculate portfolio beta and average correlation."""
        try:
            # Calculate portfolio beta and correlation using historical data
            beta_sum = 0.0
            correlation_sum = 0.0
            count = 0

            # Get SPY data as market benchmark
            spy_data = await self._get_historical_data("SPY", days=252)
            spy_returns = None
            if spy_data is not None:
                spy_returns = await self._calculate_returns(spy_data)

            for position in portfolio.positions:
                if position.quantity != 0:
                    # Calculate actual beta and correlation if we have market data
                    if spy_returns is not None:
                        # Get stock data and calculate beta
                        stock_data = await self._get_historical_data(position.symbol, days=252)
                        if stock_data is not None:
                            stock_returns = await self._calculate_returns(stock_data)
                            if stock_returns is not None and len(stock_returns) > 20:
                                # Merge returns data on timestamp
                                merged = stock_returns.select(["timestamp", "returns"]).join(
                                    spy_returns.select(["timestamp", "returns"]),
                                    on="timestamp",
                                    how="inner",
                                    suffix="_market"
                                )

                                if len(merged) > 20:
                                    # Calculate beta = cov(stock, market) / var(market)
                                    stock_rets = merged["returns"].to_numpy()
                                    market_rets = merged["returns_market"].to_numpy()

                                    if len(stock_rets) > 0 and len(market_rets) > 0:
                                        covariance = np.cov(stock_rets, market_rets)[0, 1]
                                        market_variance = np.var(market_rets)

                                        if market_variance > 0:
                                            beta = covariance / market_variance
                                            beta_sum += max(0.1, min(3.0, beta))  # Clamp beta
                                        else:
                                            beta_sum += 1.0

                                        # Correlation is already calculated
                                        correlation = np.corrcoef(stock_rets, market_rets)[0, 1]
                                        correlation_sum += abs(correlation) if not np.isnan(correlation) else 0.3
                                    else:
                                        beta_sum += 1.0
                                        correlation_sum += 0.3
                                else:
                                    beta_sum += 1.0
                                    correlation_sum += 0.3
                            else:
                                beta_sum += 1.0
                                correlation_sum += 0.3
                        else:
                            beta_sum += 1.0
                            correlation_sum += 0.3
                    else:
                        # Fallback to default values
                        beta_sum += 1.0
                        correlation_sum += 0.3

                    count += 1

            portfolio_beta = beta_sum / count if count > 0 else 1.0
            avg_correlation = correlation_sum / count if count > 0 else 0.0

            return portfolio_beta, avg_correlation

        except Exception as e:
            logger.error(f"Error calculating portfolio beta/correlation: {e}")
            return 1.0, 0.0

    async def _calculate_var_metrics(self, portfolio: PortfolioState) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate Value at Risk and Expected Shortfall."""
        try:
            # Simplified VaR calculation
            # In production, this would use historical simulation or Monte Carlo

            portfolio_value = portfolio.total_equity
            volatility = await self._calculate_portfolio_volatility(portfolio)

            # 95% confidence level (1.645 standard deviations)
            var_1d = portfolio_value * Decimal(str(volatility)) * Decimal("1.645") / Decimal(str(np.sqrt(252)))
            var_5d = var_1d * Decimal(str(np.sqrt(5)))

            # Expected Shortfall (typically 1.3x VaR for normal distribution)
            expected_shortfall = var_1d * Decimal("1.3")

            return var_1d, var_5d, expected_shortfall

        except Exception as e:
            logger.error(f"Error calculating VaR metrics: {e}")
            return Decimal("0"), Decimal("0"), Decimal("0")

    async def _calculate_sharpe_ratio(self, portfolio: PortfolioState) -> float:
        """Calculate portfolio Sharpe ratio."""
        try:
            # Get portfolio historical performance
            if not portfolio.positions:
                return 0.0

            # Calculate weighted returns for portfolio
            total_returns = []
            total_weights = 0.0

            for position in portfolio.positions:
                stock_data = await self._get_historical_data(position.symbol, days=252)
                if stock_data is not None:
                    returns_df = await self._calculate_returns(stock_data)
                    if returns_df is not None and len(returns_df) > 20:
                        weight = float(position.market_value / portfolio.total_market_value)
                        returns_list = returns_df["returns"].to_list()

                        if not total_returns:
                            total_returns = [r * weight for r in returns_list]
                        else:
                            # Add weighted returns
                            min_len = min(len(total_returns), len(returns_list))
                            total_returns = [total_returns[i] + returns_list[i] * weight
                                           for i in range(min_len)]

                        total_weights += weight

            if not total_returns or len(total_returns) < 20:
                return 0.0

            # Calculate Sharpe ratio
            mean_return = np.mean(total_returns)
            std_return = np.std(total_returns)

            # Assume risk-free rate of 2% annually (0.02/252 daily)
            risk_free_rate = 0.02 / 252

            if std_return > 0:
                sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
                return float(sharpe_ratio)

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    async def _calculate_drawdown_metrics(self, portfolio: PortfolioState) -> Tuple[Decimal, Decimal]:
        """Calculate max drawdown and current drawdown."""
        try:
            # Calculate portfolio value series from historical data
            portfolio_values = []

            if not portfolio.positions:
                return Decimal("0"), Decimal("0")

            # Get the shortest data series among all positions
            min_data_length = float('inf')
            position_data = {}

            for position in portfolio.positions:
                stock_data = await self._get_historical_data(position.symbol, days=252)
                if stock_data is not None and len(stock_data) > 20:
                    position_data[position.symbol] = stock_data
                    min_data_length = min(min_data_length, len(stock_data))

            if not position_data or min_data_length == float('inf'):
                return Decimal("0"), Decimal("0")

            # Calculate portfolio values over time
            for i in range(int(min_data_length)):
                portfolio_value = 0.0
                for position in portfolio.positions:
                    if position.symbol in position_data:
                        stock_df = position_data[position.symbol]
                        if i < len(stock_df):
                            price = float(stock_df[i]["close"])
                            value = price * float(position.quantity)
                            portfolio_value += value

                portfolio_values.append(portfolio_value)

            if len(portfolio_values) < 2:
                return Decimal("0"), Decimal("0")

            # Calculate drawdowns
            peak = portfolio_values[0]
            max_drawdown = 0.0
            current_drawdown = 0.0

            for value in portfolio_values:
                if value > peak:
                    peak = value

                drawdown = (peak - value) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

            # Current drawdown from current peak
            current_value = portfolio_values[-1]
            current_drawdown = (peak - current_value) / peak if peak > 0 else 0.0

            return Decimal(str(max_drawdown)), Decimal(str(current_drawdown))

        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return Decimal("0"), Decimal("0")

    async def _calculate_portfolio_volatility(self, portfolio: PortfolioState) -> float:
        """Calculate portfolio volatility using correlation matrix."""
        try:
            if not portfolio.positions or len(portfolio.positions) == 1:
                # Single asset or empty portfolio
                if portfolio.positions:
                    return await self._get_symbol_volatility(portfolio.positions[0].symbol)
                return 0.15

            # Get volatilities and weights for all positions
            symbols = [pos.symbol for pos in portfolio.positions]
            weights = []
            volatilities = []

            for position in portfolio.positions:
                weight = float(position.market_value / portfolio.total_market_value)
                volatility = await self._get_symbol_volatility(position.symbol)
                weights.append(weight)
                volatilities.append(volatility)

            # Calculate portfolio variance using correlation matrix
            portfolio_variance = 0.0

            for i in range(len(symbols)):
                for j in range(len(symbols)):
                    if i == j:
                        # Variance term
                        portfolio_variance += (weights[i] ** 2) * (volatilities[i] ** 2)
                    else:
                        # Covariance term
                        correlation = await self._get_symbol_correlation(symbols[i], symbols[j])
                        covariance = correlation * volatilities[i] * volatilities[j]
                        portfolio_variance += 2 * weights[i] * weights[j] * covariance

            # Portfolio volatility is square root of variance
            portfolio_volatility = np.sqrt(max(0, portfolio_variance))

            # Clamp to reasonable bounds
            return max(0.05, min(1.0, portfolio_volatility))

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.15

    async def _find_high_correlation_pairs(self, portfolio: PortfolioState) -> List[Tuple[Tuple[str, str], float]]:
        """Find pairs of positions with high correlation."""
        high_corr_pairs = []

        try:
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]

            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = await self._get_symbol_correlation(symbol1, symbol2)
                    if correlation > self.risk_limits.max_correlation_threshold:
                        high_corr_pairs.append(((symbol1, symbol2), correlation))

            return high_corr_pairs

        except Exception as e:
            logger.error(f"Error finding high correlation pairs: {e}")
            return []

    async def _calculate_symbol_correlations(self, portfolio: PortfolioState, new_symbol: str) -> Dict[str, float]:
        """Calculate correlations between new symbol and existing positions."""
        correlations = {}

        try:
            for position in portfolio.positions:
                if position.quantity != 0:
                    correlation = await self._get_symbol_correlation(new_symbol, position.symbol)
                    correlations[position.symbol] = correlation

            return correlations

        except Exception as e:
            logger.error(f"Error calculating symbol correlations: {e}")
            return {}

    async def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        try:
            if symbol1 == symbol2:
                return 1.0

            # Check cache first
            cache_key = f"{symbol1}_{symbol2}" if symbol1 < symbol2 else f"{symbol2}_{symbol1}"
            if cache_key in self._correlation_cache:
                cached_time, cached_value = self._correlation_cache[cache_key]
                cache_age = datetime.now() - cached_time
                if cache_age < timedelta(hours=1):
                    return cached_value

            # Get historical data for both symbols
            df1 = await self._get_historical_data(symbol1, days=252)
            df2 = await self._get_historical_data(symbol2, days=252)

            if df1 is None or df2 is None:
                # Fallback to placeholder logic
                correlation = 0.3 if len(set(symbol1) & set(symbol2)) > 1 else 0.1
            else:
                # Calculate returns for both symbols
                returns1 = await self._calculate_returns(df1)
                returns2 = await self._calculate_returns(df2)

                if returns1 is None or returns2 is None or len(returns1) < 10 or len(returns2) < 10:
                    correlation = 0.1
                else:
                    # Merge on timestamp and calculate correlation
                    merged = returns1.select(["timestamp", "returns"]).join(
                        returns2.select(["timestamp", "returns"]),
                        on="timestamp",
                        how="inner",
                        suffix="_2"
                    )

                    if len(merged) < 10:
                        correlation = 0.1
                    else:
                        # Calculate Pearson correlation
                        corr_matrix = merged.select(["returns", "returns_2"]).corr()
                        correlation = float(corr_matrix[0, 1]) if corr_matrix[0, 1] is not None else 0.1

                        # Clamp correlation to reasonable bounds
                        correlation = max(-1.0, min(1.0, correlation))

            # Cache the result
            self._correlation_cache[cache_key] = (datetime.now(), correlation)
            return correlation

        except Exception as e:
            logger.error(f"Error getting correlation between {symbol1} and {symbol2}: {e}")
            return 0.0

    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for a symbol."""
        try:
            # Check cache first
            if symbol in self._volatility_cache:
                cached_time, cached_value = self._volatility_cache[symbol]
                cache_age = datetime.now() - cached_time
                if cache_age < timedelta(hours=1):
                    return cached_value

            # Get historical data
            df = await self._get_historical_data(symbol, days=252)

            if df is None or len(df) < 20:
                # Fallback to symbol-based estimation
                if any(char.isdigit() for char in symbol):
                    volatility = 0.35  # Higher volatility for symbols with numbers
                elif len(symbol) <= 3:
                    volatility = 0.25  # Medium volatility for short symbols
                else:
                    volatility = 0.20  # Lower volatility for longer symbols
            else:
                # Calculate returns
                returns_df = await self._calculate_returns(df)

                if returns_df is None or len(returns_df) < 20:
                    volatility = 0.25
                else:
                    # Calculate annualized volatility
                    returns_std_val = returns_df["returns"].std()
                    returns_std = float(returns_std_val) if returns_std_val is not None else 0.0
                    volatility = returns_std * np.sqrt(252)  # Annualized

                    # Clamp to reasonable bounds
                    volatility = max(0.05, min(2.0, volatility))

            # Cache the result
            self._volatility_cache[symbol] = (datetime.now(), volatility)
            return volatility

        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.25  # Default volatility

    async def _get_atr(self, symbol: str, period: int = 14) -> float:
        """Get Average True Range for a symbol."""
        try:
            # Check cache first
            if symbol in self._atr_cache:
                cached_time, cached_value = self._atr_cache[symbol]
                cache_age = datetime.now() - cached_time
                if cache_age < timedelta(hours=1):
                    return cached_value

            # Get historical OHLC data
            df = await self._get_historical_data(symbol, days=30)

            if df is None or len(df) < period:
                # Fallback to placeholder
                atr = 0.02
            else:
                # Calculate True Range
                df_with_tr = df.with_columns([
                    # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
                    pl.max_horizontal([
                        pl.col("high") - pl.col("low"),
                        (pl.col("high") - pl.col("close").shift(1)).abs(),
                        (pl.col("low") - pl.col("close").shift(1)).abs()
                    ]).alias("true_range")
                ]).drop_nulls()

                if len(df_with_tr) < period:
                    atr = 0.02
                else:
                    # Calculate period-ATR
                    atr_mean_val = df_with_tr["true_range"].tail(period).mean()
                    atr_value = float(atr_mean_val) if atr_mean_val is not None else 0.02
                    price_val = df_with_tr["close"].tail(1).item()
                    current_price = float(price_val) if price_val is not None else 1.0

                    # Convert to percentage
                    atr = atr_value / current_price if current_price > 0 else 0.02

                    # Clamp to reasonable bounds
                    atr = max(0.005, min(0.10, atr))

            # Cache the result
            self._atr_cache[symbol] = (datetime.now(), atr)
            return atr

        except Exception as e:
            logger.error(f"Error getting ATR for {symbol}: {e}")
            return 0.02  # Default ATR

    async def _initialize_data_services(self):
        """Initialize data store and client connections."""
        try:
            # Import data services
            from services.data_collector.src.data_store import DataStore, DataStoreConfig
            from services.data_collector.src.twelvedata_client import TwelveDataClient, TwelveDataConfig

            # Create data store config from main config
            data_config = self.config.data
            data_store_config = DataStoreConfig(
                base_path=data_config.parquet_path,
                max_workers=4,
                retention_days=data_config.retention_days,
                compression=data_config.compression,
                batch_size=data_config.batch_size
            )

            # Create TwelveData config using values from main config
            shared_twelvedata = self.config.twelvedata
            twelvedata_config = TwelveDataConfig(
                api_key=shared_twelvedata.api_key,
                base_url=shared_twelvedata.base_url,
                rate_limit_requests=shared_twelvedata.rate_limit_requests,
                rate_limit_period=shared_twelvedata.rate_limit_period,
                timeout=float(shared_twelvedata.timeout)
            )

            # Initialize data store
            self._data_store = DataStore(data_store_config)

            # Initialize TwelveData client
            self._data_client = TwelveDataClient(twelvedata_config)

            logger.info("Data services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data services: {e}")
            # Continue without data services - will use fallback methods

    async def _get_historical_data(self, symbol: str, days: int = 30) -> Optional[pl.DataFrame]:
        """Get historical market data for a symbol."""
        try:
            if self._data_store is None:
                return None

            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            # Try to load from data store first
            df = await self._data_store.load_market_data(
                ticker=symbol,
                timeframe=TimeFrame.ONE_DAY,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                return df

            # If no local data and we have a client, try to fetch
            if self._data_client is not None:
                try:
                    market_data_list = await self._data_client.get_historical_data(
                        symbol=symbol,
                        timeframe=TimeFrame.ONE_DAY,
                        years=max(1, days // 365)
                    )

                    if market_data_list:
                        # Convert to DataFrame
                        data_dicts = []
                        for md in market_data_list[-days:]:  # Get last N days
                            data_dicts.append({
                                'timestamp': md.timestamp,
                                'open': float(md.open),
                                'high': float(md.high),
                                'low': float(md.low),
                                'close': float(md.close),
                                'volume': int(md.volume)
                            })

                        if data_dicts:
                            return pl.DataFrame(data_dicts)

                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")

            return None

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    async def _calculate_returns(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Calculate returns from price data."""
        try:
            if df is None or len(df) < 2:
                return None

            # Calculate daily returns
            returns_df = df.with_columns([
                pl.col("close").pct_change().alias("returns")
            ]).drop_nulls()

            return returns_df

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return None

    def reset_daily_counters(self):
        """Reset daily trading counters (call at market open)."""
        self.daily_pnl = Decimal("0")
        self.daily_trade_count = 0
        logger.info("Daily risk counters reset")

    def activate_emergency_stop(self, reason: str):
        """Activate emergency stop."""
        self.emergency_stop_active = True
        logger.critical(f"Emergency stop activated: {reason}")

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop (manual intervention required)."""
        self.emergency_stop_active = False
        logger.warning("Emergency stop deactivated")

    def update_daily_pnl(self, trade_pnl: Decimal):
        """Update daily P&L tracking."""
        self.daily_pnl += trade_pnl
        self.daily_trade_count += 1

    def set_portfolio_snapshot(self, portfolio: PortfolioState):
        """Set portfolio snapshot for comparison."""
        self.last_portfolio_snapshot = portfolio
