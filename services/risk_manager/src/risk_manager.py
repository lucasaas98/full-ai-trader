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

from shared.config import get_config
from shared.models import (
    Position, PortfolioState, PortfolioMetrics, TradeSignal, OrderRequest, OrderSide,
    RiskEvent, RiskEventType, RiskSeverity, PositionSizing, PositionSizingMethod,
    RiskFilter, TrailingStop, RiskLimits
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
            # This would typically require historical price data
            # For now, return reasonable defaults
            # In a real implementation, you'd calculate these from historical data

            # Placeholder implementation
            beta_sum = 0.0
            correlation_sum = 0.0
            count = 0

            for position in portfolio.positions:
                if position.quantity != 0:
                    # In real implementation, fetch beta and correlation data
                    beta_sum += 1.0  # Assume beta of 1.0
                    correlation_sum += 0.3  # Assume moderate correlation
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
        """Calculate current Sharpe ratio."""
        try:
            # This would require historical returns data
            # Placeholder implementation
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    async def _calculate_drawdown_metrics(self, portfolio: PortfolioState) -> Tuple[Decimal, Decimal]:
        """Calculate maximum and current drawdown."""
        try:
            # This would require historical portfolio values
            # Placeholder implementation
            return Decimal("0"), Decimal("0")
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return Decimal("0"), Decimal("0")

    async def _calculate_portfolio_volatility(self, portfolio: PortfolioState) -> float:
        """Calculate portfolio volatility."""
        try:
            # This would require historical price data and correlation matrix
            # Placeholder implementation
            return 0.15  # 15% annualized volatility
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
            # In production, this would fetch historical data and calculate correlation
            # For now, return a placeholder value
            if symbol1 == symbol2:
                return 1.0

            # Simulate some correlation based on symbol similarity
            # This is just a placeholder - real implementation would use price data
            return 0.3 if len(set(symbol1) & set(symbol2)) > 1 else 0.1

        except Exception as e:
            logger.error(f"Error getting correlation between {symbol1} and {symbol2}: {e}")
            return 0.0

    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get symbol volatility (annualized)."""
        try:
            # In production, this would calculate from historical price data
            # For now, return a reasonable default based on symbol characteristics

            # Simulate volatility based on symbol (placeholder)
            if any(char.isdigit() for char in symbol):
                return 0.35  # Higher volatility for symbols with numbers
            elif len(symbol) <= 3:
                return 0.25  # Medium volatility for short symbols
            else:
                return 0.20  # Lower volatility for longer symbols

        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.20  # Default volatility

    async def _get_atr(self, symbol: str, period: int = 14) -> float:
        """Get Average True Range for volatility-based stops."""
        try:
            # In production, this would calculate ATR from OHLC data
            # For now, return a placeholder based on current price estimation

            # Simulate ATR as percentage of price (placeholder)
            return 0.02  # 2% ATR

        except Exception as e:
            logger.error(f"Error getting ATR for {symbol}: {e}")
            return 0.02  # Default ATR

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
