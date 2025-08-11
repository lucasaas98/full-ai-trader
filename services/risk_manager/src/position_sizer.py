"""
Position Sizing Calculator

This module provides advanced position sizing algorithms including:
- Fixed percentage sizing with confidence adjustments
- Volatility-adjusted sizing using ATR
- Kelly Criterion for optimal sizing
- Equal weight distribution
- Dynamic confidence-based sizing
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple

from shared.config import get_config
from shared.models import (
    Position, PortfolioState, TradeSignal, PositionSizing,
    PositionSizingMethod, RiskLimits
)

logger = logging.getLogger(__name__)


class PositionSizer:
    """Advanced position sizing calculator."""

    def __init__(self, risk_limits: RiskLimits):
        """Initialize position sizer with risk limits."""
        self.risk_limits = risk_limits
        self.config = get_config()

        # Cache for volatility and correlation data
        self._volatility_cache: Dict[str, Tuple[float, datetime]] = {}
        self._correlation_cache: Dict[Tuple[str, str], Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)

    async def calculate_position_size(self,
                                    symbol: str,
                                    current_price: Decimal,
                                    portfolio: PortfolioState,
                                    signal: Optional[TradeSignal] = None,
                                    method: PositionSizingMethod = PositionSizingMethod.CONFIDENCE_BASED) -> PositionSizing:
        """
        Calculate optimal position size using specified method.

        Args:
            symbol: Trading symbol
            current_price: Current stock price
            portfolio: Current portfolio state
            signal: Optional trade signal with confidence
            method: Position sizing method to use

        Returns:
            PositionSizing calculation result
        """
        try:
            # Get base sizing parameters
            confidence_score = getattr(signal, 'confidence', 0.7) if signal else 0.7

            # Calculate size based on method
            if method == PositionSizingMethod.FIXED_PERCENTAGE:
                sizing_result = await self._fixed_percentage_sizing(
                    symbol, current_price, portfolio, confidence_score
                )
            elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
                sizing_result = await self._volatility_adjusted_sizing(
                    symbol, current_price, portfolio, confidence_score
                )
            elif method == PositionSizingMethod.KELLY_CRITERION:
                sizing_result = await self._kelly_criterion_sizing(
                    symbol, current_price, portfolio, signal
                )
            elif method == PositionSizingMethod.EQUAL_WEIGHT:
                sizing_result = await self._equal_weight_sizing(
                    symbol, current_price, portfolio
                )
            else:  # CONFIDENCE_BASED (default)
                sizing_result = await self._confidence_based_sizing(
                    symbol, current_price, portfolio, confidence_score
                )

            # Apply final safety checks
            sizing_result = await self._apply_safety_limits(sizing_result, portfolio)

            logger.info(f"Position sizing for {symbol}: {sizing_result.recommended_shares} shares "
                       f"({sizing_result.position_percentage:.2%} of portfolio)")

            return sizing_result

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return await self._fallback_sizing(symbol, current_price, portfolio)

    async def _fixed_percentage_sizing(self,
                                     symbol: str,
                                     current_price: Decimal,
                                     portfolio: PortfolioState,
                                     confidence_score: float) -> PositionSizing:
        """Fixed percentage sizing with confidence adjustment."""

        # Base allocation (20% default)
        base_percentage = self.risk_limits.max_position_percentage

        # Confidence adjustment (0.5x to 1.5x)
        confidence_adjustment = 0.5 + (confidence_score * 1.0)

        # Calculate position size
        adjusted_percentage = base_percentage * Decimal(str(confidence_adjustment))
        position_value = portfolio.total_equity * adjusted_percentage
        shares = int(position_value / current_price)

        # Calculate actual values
        actual_value = Decimal(shares) * current_price
        actual_percentage = actual_value / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

        # Risk calculations
        max_loss = actual_value * self.risk_limits.stop_loss_percentage
        max_gain = actual_value * self.risk_limits.take_profit_percentage
        risk_reward_ratio = float(max_gain / max_loss) if max_loss > 0 else 0.0

        return PositionSizing(
            symbol=symbol,
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_percentage,
            confidence_adjustment=confidence_adjustment,
            volatility_adjustment=1.0,
            sizing_method=PositionSizingMethod.FIXED_PERCENTAGE,
            max_loss_amount=max_loss,
            risk_reward_ratio=risk_reward_ratio
        )

    async def _volatility_adjusted_sizing(self,
                                        symbol: str,
                                        current_price: Decimal,
                                        portfolio: PortfolioState,
                                        confidence_score: float) -> PositionSizing:
        """Volatility-adjusted position sizing."""

        # Get symbol volatility
        volatility = await self._get_symbol_volatility(symbol)

        # Base percentage
        base_percentage = self.risk_limits.max_position_percentage

        # Volatility adjustment (inverse relationship)
        # Target volatility of 20% - scale inversely
        target_vol = 0.20
        vol_adjustment = min(2.0, max(0.5, target_vol / volatility))

        # Confidence adjustment
        confidence_adjustment = 0.5 + (confidence_score * 1.0)

        # Combined adjustment
        total_adjustment = vol_adjustment * confidence_adjustment
        adjusted_percentage = base_percentage * Decimal(str(total_adjustment))

        # Calculate position
        position_value = portfolio.total_equity * adjusted_percentage
        shares = int(position_value / current_price)

        actual_value = Decimal(shares) * current_price
        actual_percentage = actual_value / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

        # Risk calculations
        max_loss = actual_value * self.risk_limits.stop_loss_percentage
        max_gain = actual_value * self.risk_limits.take_profit_percentage
        risk_reward_ratio = float(max_gain / max_loss) if max_loss > 0 else 0.0

        return PositionSizing(
            symbol=symbol,
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_percentage,
            confidence_adjustment=confidence_adjustment,
            volatility_adjustment=vol_adjustment,
            sizing_method=PositionSizingMethod.VOLATILITY_ADJUSTED,
            max_loss_amount=max_loss,
            risk_reward_ratio=risk_reward_ratio
        )

    async def _kelly_criterion_sizing(self,
                                    symbol: str,
                                    current_price: Decimal,
                                    portfolio: PortfolioState,
                                    signal: Optional[TradeSignal]) -> PositionSizing:
        """Kelly Criterion optimal position sizing."""

        try:
            # Get historical performance data for Kelly calculation
            win_rate = await self._get_strategy_win_rate(symbol)
            avg_win = await self._get_average_win(symbol)
            avg_loss = await self._get_average_loss(symbol)

            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly_fraction = (b * win_rate - (1 - win_rate)) / b
            else:
                kelly_fraction = 0.1  # Conservative fallback

            # Cap Kelly fraction at 25% for safety
            kelly_fraction = max(0.0, min(0.25, kelly_fraction))

            # Apply confidence adjustment
            confidence_score = getattr(signal, 'confidence', 0.7) if signal else 0.7
            confidence_adjustment = 0.5 + (confidence_score * 1.0)

            adjusted_fraction = kelly_fraction * confidence_adjustment

            # Calculate position
            position_value = portfolio.total_equity * Decimal(str(adjusted_fraction))
            shares = int(position_value / current_price)

            actual_value = Decimal(shares) * current_price
            actual_percentage = actual_value / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

            # Risk calculations
            max_loss = actual_value * self.risk_limits.stop_loss_percentage
            max_gain = actual_value * self.risk_limits.take_profit_percentage
            risk_reward_ratio = float(max_gain / max_loss) if max_loss > 0 else 0.0

            return PositionSizing(
                symbol=symbol,
                recommended_shares=shares,
                recommended_value=actual_value,
                position_percentage=actual_percentage,
                confidence_adjustment=confidence_adjustment,
                volatility_adjustment=1.0,
                sizing_method=PositionSizingMethod.KELLY_CRITERION,
                max_loss_amount=max_loss,
                risk_reward_ratio=risk_reward_ratio
            )

        except Exception as e:
            logger.error(f"Error in Kelly criterion sizing for {symbol}: {e}")
            return await self._fixed_percentage_sizing(symbol, current_price, portfolio, 0.7)

    async def _equal_weight_sizing(self,
                                 symbol: str,
                                 current_price: Decimal,
                                 portfolio: PortfolioState) -> PositionSizing:
        """Equal weight position sizing."""

        # Calculate target positions (max positions)
        target_positions = self.risk_limits.max_positions
        position_percentage = Decimal("1") / Decimal(target_positions)

        # Calculate position
        position_value = portfolio.total_equity * position_percentage
        shares = int(position_value / current_price)

        actual_value = Decimal(shares) * current_price
        actual_percentage = actual_value / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

        # Risk calculations
        max_loss = actual_value * self.risk_limits.stop_loss_percentage
        max_gain = actual_value * self.risk_limits.take_profit_percentage
        risk_reward_ratio = float(max_gain / max_loss) if max_loss > 0 else 0.0

        return PositionSizing(
            symbol=symbol,
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_percentage,
            confidence_adjustment=1.0,
            volatility_adjustment=1.0,
            sizing_method=PositionSizingMethod.EQUAL_WEIGHT,
            max_loss_amount=max_loss,
            risk_reward_ratio=risk_reward_ratio
        )

    async def _confidence_based_sizing(self,
                                     symbol: str,
                                     current_price: Decimal,
                                     portfolio: PortfolioState,
                                     confidence_score: float) -> PositionSizing:
        """Confidence-based position sizing (combines multiple factors)."""

        # Base percentage
        base_percentage = self.risk_limits.max_position_percentage

        # Confidence adjustment (exponential curve for better scaling)
        confidence_adjustment = 0.3 + (confidence_score ** 1.5) * 1.2
        confidence_adjustment = min(1.5, max(0.3, confidence_adjustment))

        # Volatility adjustment
        volatility = await self._get_symbol_volatility(symbol)
        vol_adjustment = min(1.5, max(0.5, 0.20 / volatility))

        # Portfolio concentration adjustment
        concentration_adjustment = await self._calculate_concentration_adjustment(portfolio)

        # Combined adjustment
        total_adjustment = confidence_adjustment * vol_adjustment * concentration_adjustment
        adjusted_percentage = base_percentage * Decimal(str(total_adjustment))

        # Calculate position
        position_value = portfolio.total_equity * adjusted_percentage
        shares = int(position_value / current_price)

        actual_value = Decimal(shares) * current_price
        actual_percentage = actual_value / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

        # Risk calculations
        max_loss = actual_value * self.risk_limits.stop_loss_percentage
        max_gain = actual_value * self.risk_limits.take_profit_percentage
        risk_reward_ratio = float(max_gain / max_loss) if max_loss > 0 else 0.0

        return PositionSizing(
            symbol=symbol,
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_percentage,
            confidence_adjustment=confidence_adjustment,
            volatility_adjustment=vol_adjustment,
            sizing_method=PositionSizingMethod.CONFIDENCE_BASED,
            max_loss_amount=max_loss,
            risk_reward_ratio=risk_reward_ratio
        )

    async def _apply_safety_limits(self, sizing: PositionSizing, portfolio: PortfolioState) -> PositionSizing:
        """Apply final safety limits to position sizing."""

        # Ensure position doesn't exceed maximum percentage
        if sizing.position_percentage > self.risk_limits.max_position_percentage:
            max_value = portfolio.total_equity * self.risk_limits.max_position_percentage
            max_shares = int(max_value / (sizing.recommended_value / Decimal(sizing.recommended_shares)))

            sizing.recommended_shares = max_shares
            sizing.recommended_value = max_value
            sizing.position_percentage = self.risk_limits.max_position_percentage

            logger.warning(f"Position size capped at maximum limit for {sizing.symbol}")

        # Ensure minimum position size
        min_shares = 1
        if sizing.recommended_shares < min_shares:
            sizing.recommended_shares = min_shares
            sizing.recommended_value = Decimal(min_shares) * (sizing.recommended_value / Decimal(max(1, sizing.recommended_shares)))
            sizing.position_percentage = sizing.recommended_value / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

        # Recalculate risk metrics
        sizing.max_loss_amount = sizing.recommended_value * self.risk_limits.stop_loss_percentage
        max_gain = sizing.recommended_value * self.risk_limits.take_profit_percentage
        sizing.risk_reward_ratio = float(max_gain / sizing.max_loss_amount) if sizing.max_loss_amount > 0 else 0.0

        return sizing

    async def _fallback_sizing(self, symbol: str, current_price: Decimal, portfolio: PortfolioState) -> PositionSizing:
        """Fallback sizing method for error cases."""

        # Very conservative sizing - 1% of portfolio
        conservative_percentage = Decimal("0.01")
        position_value = portfolio.total_equity * conservative_percentage
        shares = max(1, int(position_value / current_price))

        actual_value = Decimal(shares) * current_price
        actual_percentage = actual_value / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")

        return PositionSizing(
            symbol=symbol,
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_percentage,
            confidence_adjustment=0.5,
            volatility_adjustment=0.5,
            sizing_method=PositionSizingMethod.FIXED_PERCENTAGE,
            max_loss_amount=actual_value * self.risk_limits.stop_loss_percentage,
            risk_reward_ratio=1.5
        )

    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get cached or calculate symbol volatility."""

        # Check cache
        if symbol in self._volatility_cache:
            volatility, timestamp = self._volatility_cache[symbol]
            if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                return volatility

        try:
            # In production, this would fetch historical data and calculate volatility
            # For now, simulate based on symbol characteristics
            volatility = await self._calculate_historical_volatility(symbol)

            # Cache the result
            self._volatility_cache[symbol] = (volatility, datetime.now(timezone.utc))

            return volatility

        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.20  # Default 20% volatility

    async def _calculate_historical_volatility(self, symbol: str, lookback_days: int = 30) -> float:
        """Calculate historical volatility from price data."""
        try:
            # This is a placeholder implementation
            # In production, you would:
            # 1. Fetch historical price data from your data store
            # 2. Calculate daily returns
            # 3. Calculate standard deviation
            # 4. Annualize the volatility

            # Simulate volatility based on symbol characteristics
            if symbol.startswith(('QQQ', 'SPY', 'IWM')):
                return 0.15  # ETFs tend to be less volatile
            elif len(symbol) <= 3:
                return 0.25  # Large caps
            elif any(keyword in symbol.upper() for keyword in ['TECH', 'GROWTH', 'BIO']):
                return 0.35  # High volatility sectors
            else:
                return 0.22  # Default mid-cap volatility

        except Exception as e:
            logger.error(f"Error calculating historical volatility for {symbol}: {e}")
            return 0.20

    async def _calculate_concentration_adjustment(self, portfolio: PortfolioState) -> float:
        """Calculate adjustment factor based on portfolio concentration."""
        try:
            if not portfolio.positions:
                return 1.0

            # Calculate current concentration (Herfindahl index)
            total_value = portfolio.total_market_value
            if total_value <= 0:
                return 1.0

            # Calculate position weights
            weights = []
            for position in portfolio.positions:
                if position.quantity != 0:
                    weight = abs(position.market_value) / total_value
                    weights.append(float(weight))

            if not weights:
                return 1.0

            # Herfindahl-Hirschman Index
            hhi = sum(w**2 for w in weights)

            # Adjustment: reduce size if already concentrated
            # HHI ranges from 1/n (perfectly diversified) to 1 (concentrated)
            # Apply reduction if HHI > 0.4 (moderate concentration)
            if hhi > 0.4:
                adjustment = max(0.5, 1.0 - (hhi - 0.4))
            else:
                adjustment = 1.0

            return adjustment

        except Exception as e:
            logger.error(f"Error calculating concentration adjustment: {e}")
            return 1.0

    async def _get_strategy_win_rate(self, symbol: str) -> float:
        """Get historical win rate for the strategy on this symbol."""
        try:
            # This would query the database for historical performance
            # Placeholder implementation
            return 0.6  # 60% win rate
        except Exception as e:
            logger.error(f"Error getting win rate for {symbol}: {e}")
            return 0.5

    async def _get_average_win(self, symbol: str) -> float:
        """Get average winning trade percentage."""
        try:
            # This would query historical trade data
            # Placeholder implementation
            return 0.04  # 4% average win
        except Exception as e:
            logger.error(f"Error getting average win for {symbol}: {e}")
            return 0.03

    async def _get_average_loss(self, symbol: str) -> float:
        """Get average losing trade percentage."""
        try:
            # This would query historical trade data
            # Placeholder implementation
            return 0.02  # 2% average loss
        except Exception as e:
            logger.error(f"Error getting average loss for {symbol}: {e}")
            return 0.02

    def calculate_shares_from_dollar_amount(self, dollar_amount: Decimal, price_per_share: Decimal) -> int:
        """Calculate number of shares from dollar amount."""
        if price_per_share <= 0:
            return 0

        shares = dollar_amount / price_per_share
        return int(shares.quantize(Decimal('1'), rounding=ROUND_HALF_UP))

    def calculate_position_value(self, shares: int, price_per_share: Decimal) -> Decimal:
        """Calculate total position value."""
        return Decimal(shares) * price_per_share

    def calculate_position_percentage(self, position_value: Decimal, portfolio_value: Decimal) -> Decimal:
        """Calculate position as percentage of portfolio."""
        if portfolio_value <= 0:
            return Decimal("0")
        return position_value / portfolio_value

    async def adjust_size_for_existing_position(self,
                                              sizing: PositionSizing,
                                              existing_position: Optional[Position]) -> PositionSizing:
        """Adjust position size if there's an existing position."""

        if not existing_position or existing_position.quantity == 0:
            return sizing

        # If we already have a position, reduce the new position size
        # to avoid over-concentration
        reduction_factor = 0.5  # Reduce by 50% if position exists

        sizing.recommended_shares = int(sizing.recommended_shares * reduction_factor)
        sizing.recommended_value = sizing.recommended_value * Decimal(str(reduction_factor))
        sizing.position_percentage = sizing.position_percentage * Decimal(str(reduction_factor))

        logger.info(f"Reduced position size for {sizing.symbol} due to existing position")

        return sizing

    def get_risk_adjusted_size(self,
                             base_size: int,
                             volatility: float,
                             max_risk_per_trade: Decimal) -> int:
        """Calculate risk-adjusted position size based on volatility."""

        try:
            # Calculate position size based on maximum risk per trade
            # Risk per trade = position_size * volatility * stop_loss_distance

            stop_loss_distance = float(self.risk_limits.stop_loss_percentage)
            max_position_risk = volatility * stop_loss_distance

            if max_position_risk <= 0:
                return base_size

            # Calculate maximum position size to stay within risk limit
            max_risk_dollars = float(max_risk_per_trade)
            max_position_dollars = max_risk_dollars / max_position_risk

            # Convert back to shares (approximate)
            avg_share_price = 100  # Placeholder - would use actual price
            max_shares = int(max_position_dollars / avg_share_price)

            # Return the smaller of base size or risk-adjusted size
            return min(base_size, max_shares)

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted size: {e}")
            return base_size

    async def validate_position_sizing(self, sizing: PositionSizing, portfolio: PortfolioState) -> List[str]:
        """Validate position sizing against all constraints."""

        violations = []

        # Check maximum position percentage
        if sizing.position_percentage > self.risk_limits.max_position_percentage:
            violations.append(f"Position percentage {sizing.position_percentage:.2%} exceeds maximum {self.risk_limits.max_position_percentage:.2%}")

        # Check minimum position size
        if sizing.recommended_shares < 1:
            violations.append("Position size is less than 1 share")

        # Check if position value exceeds buying power
        if sizing.recommended_value > portfolio.buying_power:
            violations.append(f"Position value ${sizing.recommended_value} exceeds buying power ${portfolio.buying_power}")

        # Check maximum loss amount
        max_loss_percentage = sizing.max_loss_amount / portfolio.total_equity if portfolio.total_equity > 0 else Decimal("0")
        if max_loss_percentage > self.risk_limits.stop_loss_percentage * Decimal("2"):  # 2x normal stop loss
            violations.append(f"Maximum loss amount {max_loss_percentage:.2%} is too high")

        return violations
