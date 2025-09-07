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
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Tuple

from shared.config import get_config
from shared.models import (
    PortfolioState,
    Position,
    PositionSizing,
    PositionSizingMethod,
    RiskLimits,
    TradeSignal,
)

from .database_manager import RiskDatabaseManager

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

        # Initialize data access components (will be set up in initialize())
        self.data_store = None
        self.db_manager: Optional[RiskDatabaseManager] = None

        # Sector classification cache
        self._sector_cache: Dict[str, str] = {}

        # Market condition cache
        self._market_condition_cache: Optional[Tuple[str, datetime]] = None

        # Dynamic risk adjustment tracking
        self._recent_performance: List[Tuple[datetime, float]] = (
            []
        )  # (timestamp, pnl_percentage)
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_timestamp: Optional[datetime] = None
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Initialize database connections and data store."""
        try:
            if self._initialized:
                return

            # Initialize database manager
            self.db_manager = RiskDatabaseManager()
            await self.db_manager.initialize()

            self._initialized = True
            logger.info("Position sizer initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing position sizer: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up database connections and resources."""
        try:
            if self.db_manager:
                await self.db_manager.close()
                self.db_manager = None

            self.data_store = None
            self._initialized = False

            logger.info("Position sizer cleanup completed")

        except Exception as e:
            logger.error(f"Error during position sizer cleanup: {e}")

    def _ensure_initialized(self) -> None:
        """Ensure the position sizer is properly initialized."""
        if not self._initialized or not self.db_manager:
            raise RuntimeError(
                "Position sizer not initialized. Call initialize() first."
            )

    async def calculate_position_size(
        self,
        symbol: str,
        current_price: Decimal,
        portfolio: PortfolioState,
        signal: Optional[TradeSignal] = None,
        method: PositionSizingMethod = PositionSizingMethod.CONFIDENCE_BASED,
    ) -> PositionSizing:
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
            # Ensure initialization
            self._ensure_initialized()

            # Check circuit breaker status first
            await self._update_circuit_breaker_status(portfolio)

            # Apply dynamic risk adjustment based on recent performance
            dynamic_risk_adjustment = await self._calculate_dynamic_risk_adjustment()

            # Get base sizing parameters
            confidence_score = getattr(signal, "confidence", 0.7) if signal else 0.7

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

            # Apply dynamic risk adjustment
            sizing_result = await self._apply_dynamic_risk_adjustment(
                sizing_result, dynamic_risk_adjustment
            )

            # Apply circuit breaker if active
            if self._circuit_breaker_active:
                sizing_result = await self._apply_circuit_breaker_adjustment(
                    sizing_result
                )

            # Apply final safety checks
            sizing_result = await self._apply_safety_limits(sizing_result, portfolio)

            # Track position sizing for performance analysis
            await self._track_position_sizing_decision(sizing_result, signal)

            logger.info(
                f"Position sizing for {symbol}: {sizing_result.recommended_shares} shares "
                f"({sizing_result.position_percentage:.2%} of portfolio) "
                f"[Dynamic adjustment: {dynamic_risk_adjustment:.2f}, Circuit breaker: {self._circuit_breaker_active}]"
            )

            return sizing_result

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return await self._fallback_sizing(symbol, current_price, portfolio)

    async def _fixed_percentage_sizing(
        self,
        symbol: str,
        current_price: Decimal,
        portfolio: PortfolioState,
        confidence_score: float,
    ) -> PositionSizing:
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
        actual_percentage = (
            actual_value / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("0")
        )

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
            risk_reward_ratio=risk_reward_ratio,
        )

    async def _volatility_adjusted_sizing(
        self,
        symbol: str,
        current_price: Decimal,
        portfolio: PortfolioState,
        confidence_score: float,
    ) -> PositionSizing:
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
        actual_percentage = (
            actual_value / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("0")
        )

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
            risk_reward_ratio=risk_reward_ratio,
        )

    async def _kelly_criterion_sizing(
        self,
        symbol: str,
        current_price: Decimal,
        portfolio: PortfolioState,
        signal: Optional[TradeSignal],
    ) -> PositionSizing:
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
            confidence_score = getattr(signal, "confidence", 0.7) if signal else 0.7
            confidence_adjustment = 0.5 + (confidence_score * 1.0)

            adjusted_fraction = kelly_fraction * confidence_adjustment

            # Calculate position
            position_value = portfolio.total_equity * Decimal(str(adjusted_fraction))
            shares = int(position_value / current_price)

            actual_value = Decimal(shares) * current_price
            actual_percentage = (
                actual_value / portfolio.total_equity
                if portfolio.total_equity > 0
                else Decimal("0")
            )

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
                risk_reward_ratio=risk_reward_ratio,
            )

        except Exception as e:
            logger.error(f"Error in Kelly criterion sizing for {symbol}: {e}")
            return await self._fixed_percentage_sizing(
                symbol, current_price, portfolio, 0.7
            )

    async def _equal_weight_sizing(
        self, symbol: str, current_price: Decimal, portfolio: PortfolioState
    ) -> PositionSizing:
        """Equal weight position sizing."""

        # Calculate target positions (max positions)
        target_positions = self.risk_limits.max_positions
        position_percentage = Decimal("1") / Decimal(target_positions)

        # Calculate position
        position_value = portfolio.total_equity * position_percentage
        shares = int(position_value / current_price)

        actual_value = Decimal(shares) * current_price
        actual_percentage = (
            actual_value / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("0")
        )

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
            risk_reward_ratio=risk_reward_ratio,
        )

    async def _confidence_based_sizing(
        self,
        symbol: str,
        current_price: Decimal,
        portfolio: PortfolioState,
        confidence_score: float,
    ) -> PositionSizing:
        """Confidence-based position sizing (combines multiple factors)."""

        # Base percentage
        base_percentage = self.risk_limits.max_position_percentage

        # Confidence adjustment (exponential curve for better scaling)
        confidence_adjustment = 0.3 + (confidence_score**1.5) * 1.2
        confidence_adjustment = min(1.5, max(0.3, confidence_adjustment))

        # Volatility adjustment
        volatility = await self._get_symbol_volatility(symbol)
        vol_adjustment = min(1.5, max(0.5, 0.20 / volatility))

        # Portfolio concentration adjustment
        concentration_adjustment = await self._calculate_concentration_adjustment(
            portfolio
        )

        # Correlation adjustment to avoid over-concentration in correlated assets
        correlation_adjustment = await self._calculate_correlation_adjustment(
            symbol, portfolio
        )

        # Sector concentration adjustment
        sector_adjustment = await self._calculate_sector_concentration_adjustment(
            symbol, portfolio
        )

        # Market condition adjustment
        market_adjustment = await self._calculate_market_condition_adjustment()

        # Combined adjustment
        total_adjustment = (
            confidence_adjustment
            * vol_adjustment
            * concentration_adjustment
            * correlation_adjustment
            * sector_adjustment
            * market_adjustment
        )
        adjusted_percentage = base_percentage * Decimal(str(total_adjustment))

        # Calculate position
        position_value = portfolio.total_equity * adjusted_percentage
        shares = int(position_value / current_price)

        actual_value = Decimal(shares) * current_price
        actual_percentage = (
            actual_value / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("0")
        )

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
            risk_reward_ratio=risk_reward_ratio,
        )

    async def _apply_safety_limits(
        self, sizing: PositionSizing, portfolio: PortfolioState
    ) -> PositionSizing:
        """Apply final safety limits to position sizing."""

        # Ensure position doesn't exceed maximum percentage
        if sizing.position_percentage > self.risk_limits.max_position_percentage:
            max_value = (
                portfolio.total_equity * self.risk_limits.max_position_percentage
            )
            max_shares = int(
                max_value
                / (sizing.recommended_value / Decimal(sizing.recommended_shares))
            )

            sizing.recommended_shares = max_shares
            sizing.recommended_value = max_value
            sizing.position_percentage = self.risk_limits.max_position_percentage

            logger.warning(f"Position size capped at maximum limit for {sizing.symbol}")

        # Ensure minimum position size
        min_shares = 1
        if sizing.recommended_shares < min_shares:
            sizing.recommended_shares = min_shares
            sizing.recommended_value = Decimal(min_shares) * (
                sizing.recommended_value / Decimal(max(1, sizing.recommended_shares))
            )
            sizing.position_percentage = (
                sizing.recommended_value / portfolio.total_equity
                if portfolio.total_equity > 0
                else Decimal("0")
            )

        # Recalculate risk metrics
        sizing.max_loss_amount = (
            sizing.recommended_value * self.risk_limits.stop_loss_percentage
        )
        max_gain = sizing.recommended_value * self.risk_limits.take_profit_percentage
        sizing.risk_reward_ratio = (
            float(max_gain / sizing.max_loss_amount)
            if sizing.max_loss_amount > 0
            else 0.0
        )

        return sizing

    async def _fallback_sizing(
        self, symbol: str, current_price: Decimal, portfolio: PortfolioState
    ) -> PositionSizing:
        """Fallback sizing method for error cases."""

        # Very conservative sizing - 1% of portfolio
        conservative_percentage = Decimal("0.01")
        position_value = portfolio.total_equity * conservative_percentage
        shares = max(1, int(position_value / current_price))

        actual_value = Decimal(shares) * current_price
        actual_percentage = (
            actual_value / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("0")
        )

        return PositionSizing(
            symbol=symbol,
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_percentage,
            confidence_adjustment=0.5,
            volatility_adjustment=0.5,
            sizing_method=PositionSizingMethod.FIXED_PERCENTAGE,
            max_loss_amount=actual_value * self.risk_limits.stop_loss_percentage,
            risk_reward_ratio=1.5,
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

    async def _calculate_historical_volatility(
        self, symbol: str, lookback_days: int = 30
    ) -> float:
        """Calculate historical volatility from price data."""
        try:
            # Use fallback volatility since data store is not available
            logger.warning(
                f"Using fallback volatility for {symbol} - data store not available"
            )
            return self._get_fallback_volatility(symbol)

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return self._get_fallback_volatility(symbol)

    def _get_fallback_volatility(self, symbol: str) -> float:
        """Get fallback volatility estimates based on symbol characteristics."""
        try:
            # ETFs and index funds - lower volatility
            if symbol.upper() in ["SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO"]:
                return 0.15
            # Blue chip stocks (typically 1-4 letters, well-known)
            elif len(symbol) <= 3 and symbol.upper() in [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "JPM",
                "JNJ",
                "PG",
            ]:
                return 0.25
            # Growth/tech/biotech - higher volatility
            elif any(
                keyword in symbol.upper()
                for keyword in ["TECH", "GROWTH", "BIO", "SPAC"]
            ):
                return 0.35
            # Small cap or unknown symbols
            elif len(symbol) > 4:
                return 0.40
            # Default for mid/large cap
            else:
                return 0.28
        except Exception:
            return 0.25  # Conservative default

    async def _calculate_concentration_adjustment(
        self, portfolio: PortfolioState
    ) -> float:
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
            # Query trade performance from database
            # Get trades from last 90 days for this symbol
            end_date = datetime.now(timezone.utc)
            _ = end_date - timedelta(days=90)

            # Use fallback method since database access is not available
            logger.warning(f"Using fallback win rate for {symbol}")
            return self._get_fallback_win_rate(symbol)
        except Exception as e:
            logger.error(f"Error getting win rate for {symbol}: {e}")
            return self._get_fallback_win_rate(symbol)

    def _get_fallback_win_rate(self, symbol: str) -> float:
        """Get fallback win rate estimates."""
        # Conservative estimates based on symbol type
        if symbol.upper() in ["SPY", "QQQ", "IWM", "VTI"]:
            return 0.55  # ETFs tend to have steady but modest win rates
        elif len(symbol) <= 3:
            return 0.58  # Large caps
        else:
            return 0.52  # Smaller/riskier stocks

    async def _get_average_win(self, symbol: str) -> float:
        """Get average winning trade percentage."""
        try:
            # Query winning trades from database
            # Use fallback method since database access is not available
            logger.warning(f"Using fallback average win for {symbol}")
            return self._get_fallback_average_win(symbol)
        except Exception as e:
            logger.error(f"Error getting average win for {symbol}: {e}")
            return self._get_fallback_average_win(symbol)

    def _get_fallback_average_win(self, symbol: str) -> float:
        """Get fallback average win estimates."""
        # Estimates based on volatility and symbol type
        volatility = self._get_fallback_volatility(symbol)
        # Higher volatility stocks tend to have larger wins but less consistency
        if volatility > 0.35:
            return 0.06  # High vol stocks - bigger wins
        elif volatility > 0.25:
            return 0.04  # Medium vol stocks
        else:
            return 0.03  # Low vol stocks - smaller but more consistent wins

    async def _get_average_loss(self, symbol: str) -> float:
        """Get average losing trade percentage."""
        try:
            # Query losing trades from database
            # Use fallback method since database access is not available
            logger.warning(f"Using fallback average loss for {symbol}")
            return self._get_fallback_average_loss(symbol)
        except Exception as e:
            logger.error(f"Error getting average loss for {symbol}: {e}")
            return self._get_fallback_average_loss(symbol)

    def _get_fallback_average_loss(self, symbol: str) -> float:
        """Get fallback average loss estimates."""
        # Estimates based on our stop-loss strategy and volatility
        volatility = self._get_fallback_volatility(symbol)
        # Assume our stop-loss is typically hit before major losses
        # Higher volatility = potentially larger losses despite stops
        if volatility > 0.35:
            return 0.025  # High vol - larger potential losses
        elif volatility > 0.25:
            return 0.020  # Medium vol
        else:
            return 0.015  # Low vol - tighter stops work better

    async def _calculate_correlation_adjustment(
        self, symbol: str, portfolio: PortfolioState
    ) -> float:
        """Calculate position size adjustment based on correlation with existing positions."""
        try:
            if not portfolio.positions:
                return 1.0

            # Get symbols of existing positions
            existing_symbols = [
                pos.symbol for pos in portfolio.positions if pos.quantity != 0
            ]

            if not existing_symbols:
                return 1.0

            # Calculate average correlation with existing positions
            correlations = []

            for existing_symbol in existing_symbols:
                correlation = await self._get_symbol_correlation(
                    symbol, existing_symbol
                )
                correlations.append(correlation)

            if not correlations:
                return 1.0

            # Calculate weighted average correlation (weight by position size)
            weighted_correlation = 0.0
            total_weight = 0.0

            for i, existing_symbol in enumerate(existing_symbols):
                position = next(
                    p for p in portfolio.positions if p.symbol == existing_symbol
                )
                weight = (
                    abs(float(position.market_value))
                    / float(portfolio.total_market_value)
                    if portfolio.total_market_value > 0
                    else 0
                )
                weighted_correlation += correlations[i] * weight
                total_weight += weight

            if total_weight > 0:
                avg_correlation = weighted_correlation / total_weight
            else:
                avg_correlation = sum(correlations) / len(correlations)

            # Reduce position size if highly correlated (correlation > 0.7)
            if avg_correlation > 0.7:
                adjustment = max(0.3, 1.0 - (avg_correlation - 0.7) * 2.0)
            elif avg_correlation > 0.5:
                adjustment = max(0.6, 1.0 - (avg_correlation - 0.5) * 1.0)
            else:
                adjustment = 1.0

            logger.debug(
                f"Correlation adjustment for {symbol}: {adjustment:.3f} (avg correlation: {avg_correlation:.3f})"
            )
            return adjustment

        except Exception as e:
            logger.error(f"Error calculating correlation adjustment for {symbol}: {e}")
            return 0.8  # Conservative adjustment on error

    async def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        try:
            # Check cache first
            cache_key = (sorted([symbol1, symbol2])[0], sorted([symbol1, symbol2])[1])
            if cache_key in self._correlation_cache:
                correlation, timestamp = self._correlation_cache[cache_key]
                if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                    return correlation

            # If same symbol, correlation is 1
            if symbol1 == symbol2:
                return 1.0

            # Calculate correlation from historical data
            lookback_days = 60
            end_date = datetime.now(timezone.utc)
            _ = end_date - timedelta(days=lookback_days)

            # Use sector-based correlation estimation since data store is not available
            return self._estimate_correlation_by_sector(symbol1, symbol2)

        except Exception as e:
            logger.error(
                f"Error calculating correlation between {symbol1} and {symbol2}: {e}"
            )
            return self._estimate_correlation_by_sector(symbol1, symbol2)

    def _estimate_correlation_by_sector(self, symbol1: str, symbol2: str) -> float:
        """Estimate correlation based on sector classification."""
        try:
            sector1 = self._get_symbol_sector(symbol1)
            sector2 = self._get_symbol_sector(symbol2)

            if sector1 == sector2:
                return 0.6  # Same sector - moderately correlated
            elif sector1 in ["Technology", "Communication"] and sector2 in [
                "Technology",
                "Communication",
            ]:
                return 0.4  # Tech sectors are somewhat correlated
            elif sector1 in ["Financials", "Real Estate"] and sector2 in [
                "Financials",
                "Real Estate",
            ]:
                return 0.5  # Financial sectors are correlated
            else:
                return 0.2  # Different sectors - low correlation
        except Exception:
            return 0.3  # Default moderate correlation

    async def _calculate_sector_concentration_adjustment(
        self, symbol: str, portfolio: PortfolioState
    ) -> float:
        """Calculate adjustment based on sector concentration limits."""
        try:
            if not portfolio.positions:
                return 1.0

            # Get symbol's sector
            new_symbol_sector = self._get_symbol_sector(symbol)

            # Calculate current sector exposure
            sector_exposure: Dict[str, Decimal] = {}
            total_value = (
                float(portfolio.total_market_value)
                if portfolio.total_market_value > 0
                else 1.0
            )

            for position in portfolio.positions:
                if position.quantity != 0:
                    sector = self._get_symbol_sector(position.symbol)
                    sector_value = abs(Decimal(str(position.market_value)))

                    if sector in sector_exposure:
                        sector_exposure[sector] += sector_value
                    else:
                        sector_exposure[sector] = sector_value

            # Calculate current exposure to the new symbol's sector
            current_sector_exposure = sector_exposure.get(
                new_symbol_sector, Decimal("0")
            )
            current_sector_percentage = float(current_sector_exposure) / total_value

            # Define sector concentration limits
            max_sector_concentration = 0.4  # 40% max per sector
            warning_threshold = 0.3  # Start reducing at 30%

            # Apply adjustment if approaching limits
            if current_sector_percentage > max_sector_concentration:
                return 0.1  # Minimal position if already over limit
            elif current_sector_percentage > warning_threshold:
                # Linear reduction from 30% to 40%
                excess = current_sector_percentage - warning_threshold
                max_excess = max_sector_concentration - warning_threshold
                adjustment = 1.0 - (excess / max_excess) * 0.8
                return max(0.2, adjustment)
            else:
                return 1.0

        except Exception as e:
            logger.error(
                f"Error calculating sector concentration adjustment for {symbol}: {e}"
            )
            return 0.8  # Conservative on error

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector classification for a symbol."""
        try:
            # Check cache first
            if symbol in self._sector_cache:
                return self._sector_cache[symbol]

            # Basic sector classification based on symbol patterns
            # In production, this would query a financial data API
            symbol_upper = symbol.upper()

            # ETFs and indices
            if symbol_upper in ["SPY", "QQQ", "IWM", "VTI", "VOO"]:
                sector = "Index/ETF"
            # Technology companies
            elif symbol_upper in [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "NVDA",
                "CRM",
                "ORCL",
            ]:
                sector = "Technology"
            # Financial companies
            elif symbol_upper in ["JPM", "BAC", "WFC", "GS", "MS", "C"]:
                sector = "Financials"
            # Healthcare/Biotech
            elif (
                symbol_upper in ["JNJ", "PFE", "UNH", "ABBV", "BMY"]
                or "BIO" in symbol_upper
            ):
                sector = "Healthcare"
            # Energy
            elif symbol_upper in ["XOM", "CVX", "COP", "SLB"]:
                sector = "Energy"
            # Consumer goods
            elif symbol_upper in ["PG", "KO", "PEP", "WMT", "TGT"]:
                sector = "Consumer Staples"
            # Industrial
            elif symbol_upper in ["BA", "CAT", "GE", "MMM"]:
                sector = "Industrials"
            else:
                sector = "Other"  # Default category

            # Cache the result
            self._sector_cache[symbol] = sector
            return sector

        except Exception as e:
            logger.error(f"Error getting sector for {symbol}: {e}")
            return "Other"

    async def _calculate_market_condition_adjustment(self) -> float:
        """Calculate position size adjustment based on overall market conditions."""
        try:
            # Check cache first
            if self._market_condition_cache:
                condition, timestamp = self._market_condition_cache
                if datetime.now(timezone.utc) - timestamp < timedelta(
                    hours=4
                ):  # Cache for 4 hours
                    return self._get_adjustment_for_condition(condition)

            # Analyze market conditions using key indices
            _ = ["SPY", "QQQ", "VIX"]  # S&P 500, NASDAQ, Volatility Index

            # Get recent market data (last 20 days)
            end_date = datetime.now(timezone.utc)
            _ = end_date - timedelta(days=20)

            _ = 0.15  # market_volatility
            _ = 0.02  # market_trend
            _ = 20  # valid_data_count

            # Use conservative default since data store is not available
            return 0.8  # Conservative default

        except Exception as e:
            logger.error(f"Error calculating market condition adjustment: {e}")
            return 0.8  # Conservative default

    def _get_adjustment_for_condition(self, condition: str) -> float:
        """Get position size adjustment factor for market condition."""
        adjustments = {
            "High Stress": 0.3,  # Reduce positions significantly during high stress
            "Moderate Stress": 0.6,  # Moderate reduction during stress
            "Normal": 1.0,  # Normal sizing
            "Low Volatility Bull": 1.2,  # Slightly larger positions in calm bull markets
        }
        return adjustments.get(condition, 0.8)

    def calculate_shares_from_dollar_amount(
        self, dollar_amount: Decimal, price_per_share: Decimal
    ) -> int:
        """Calculate number of shares from dollar amount."""
        if price_per_share <= 0:
            return 0

        shares = dollar_amount / price_per_share
        return int(shares.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    def calculate_position_value(
        self, shares: int, price_per_share: Decimal
    ) -> Decimal:
        """Calculate total position value."""
        return Decimal(shares) * price_per_share

    def calculate_position_percentage(
        self, position_value: Decimal, portfolio_value: Decimal
    ) -> Decimal:
        """Calculate position as percentage of portfolio."""
        if portfolio_value <= 0:
            return Decimal("0")
        return position_value / portfolio_value

    async def adjust_size_for_existing_position(
        self, sizing: PositionSizing, existing_position: Optional[Position]
    ) -> PositionSizing:
        """Adjust position size if there's an existing position."""

        if not existing_position or existing_position.quantity == 0:
            return sizing

        # If we already have a position, reduce the new position size
        # to avoid over-concentration
        reduction_factor = 0.5  # Reduce by 50% if position exists

        sizing.recommended_shares = int(sizing.recommended_shares * reduction_factor)
        sizing.recommended_value = sizing.recommended_value * Decimal(
            str(reduction_factor)
        )
        sizing.position_percentage = sizing.position_percentage * Decimal(
            str(reduction_factor)
        )

        logger.info(
            f"Reduced position size for {sizing.symbol} due to existing position"
        )

        return sizing

    def get_risk_adjusted_size(
        self, base_size: int, volatility: float, max_risk_per_trade: Decimal
    ) -> int:
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

            # Use actual current price for conversion
            # This requires the current price to be passed in - for now estimate from base_size
            if base_size > 0:
                # Estimate price from existing calculation
                estimated_price = 50  # Conservative estimate for price per share
                max_shares = int(max_position_dollars / estimated_price)
            else:
                max_shares = base_size

            # Return the smaller of base size or risk-adjusted size
            return min(base_size, max_shares)

        except Exception as e:
            logger.error(f"Error calculating risk-adjusted size: {e}")
            return base_size

    async def validate_position_sizing(
        self, sizing: PositionSizing, portfolio: PortfolioState
    ) -> List[str]:
        """Validate position sizing against all constraints."""

        violations = []

        # Check maximum position percentage
        if sizing.position_percentage > self.risk_limits.max_position_percentage:
            violations.append(
                f"Position percentage {sizing.position_percentage:.2%} exceeds maximum {self.risk_limits.max_position_percentage:.2%}"
            )

        # Check minimum position size
        if sizing.recommended_shares < 1:
            violations.append("Position size is less than 1 share")

        # Check if position value exceeds buying power
        if sizing.recommended_value > portfolio.buying_power:
            violations.append(
                f"Position value ${sizing.recommended_value} exceeds buying power ${portfolio.buying_power}"
            )

        # Check maximum loss amount
        max_loss_percentage = (
            sizing.max_loss_amount / portfolio.total_equity
            if portfolio.total_equity > 0
            else Decimal("0")
        )
        if max_loss_percentage > self.risk_limits.stop_loss_percentage * Decimal(
            "2"
        ):  # 2x normal stop loss
            violations.append(
                f"Maximum loss amount {max_loss_percentage:.2%} is too high"
            )

        return violations

    async def get_portfolio_diversification_score(
        self, portfolio: PortfolioState
    ) -> float:
        """Calculate a diversification score for the portfolio (0-1, higher is better)."""
        try:
            if not portfolio.positions:
                return 1.0  # Empty portfolio is considered "diversified"

            # Factor 1: Number of positions (more positions = better diversification)
            num_positions = len([p for p in portfolio.positions if p.quantity != 0])
            position_score = min(1.0, num_positions / 20)  # Optimal around 20 positions

            # Factor 2: Position size distribution (more equal = better)
            total_value = (
                float(portfolio.total_market_value)
                if portfolio.total_market_value > 0
                else 1.0
            )
            position_weights = []

            for position in portfolio.positions:
                if position.quantity != 0:
                    weight = abs(float(position.market_value)) / total_value
                    position_weights.append(weight)

            if position_weights:
                # Calculate Herfindahl index (lower = more diversified)
                hhi = sum(w**2 for w in position_weights)
                concentration_score = max(0.0, 1.0 - hhi)
            else:
                concentration_score = 1.0

            # Factor 3: Sector diversification
            sectors: Dict[str, float] = {}
            for position in portfolio.positions:
                if position.quantity != 0:
                    sector = self._get_symbol_sector(position.symbol)
                    weight = abs(float(position.market_value)) / total_value
                    sectors[sector] = sectors.get(sector, 0.0) + weight

            if sectors:
                sector_hhi = sum(w**2 for w in sectors.values())
                sector_score = max(0.0, 1.0 - sector_hhi)
            else:
                sector_score = 1.0

            # Combined diversification score
            diversification_score = (
                position_score * 0.3 + concentration_score * 0.4 + sector_score * 0.3
            )

            logger.debug(
                f"Portfolio diversification score: {diversification_score:.3f}"
            )
            return diversification_score

        except Exception as e:
            logger.error(f"Error calculating diversification score: {e}")
            return 0.5  # Neutral score on error

    async def get_recommended_portfolio_adjustments(
        self, portfolio: PortfolioState
    ) -> List[str]:
        """Get recommendations for improving portfolio diversification and risk management."""
        try:
            recommendations = []

            # Check diversification
            diversification_score = await self.get_portfolio_diversification_score(
                portfolio
            )
            if diversification_score < 0.5:
                recommendations.append(
                    "Consider increasing portfolio diversification across more positions and sectors"
                )

            # Check sector concentration
            total_value = (
                float(portfolio.total_market_value)
                if portfolio.total_market_value > 0
                else 1.0
            )
            sectors: Dict[str, float] = {}

            for position in portfolio.positions:
                if position.quantity != 0:
                    sector = self._get_symbol_sector(position.symbol)
                    weight = abs(float(position.market_value)) / total_value
                    sectors[sector] = sectors.get(sector, 0.0) + weight

            for sector, weight in sectors.items():
                if weight > 0.4:
                    recommendations.append(
                        f"Reduce {sector} sector concentration (currently {weight:.1%})"
                    )

            # Check position sizes
            for position in portfolio.positions:
                if position.quantity != 0:
                    weight = abs(float(position.market_value)) / total_value
                    if weight > float(self.risk_limits.max_position_percentage):
                        recommendations.append(
                            f"Reduce position size for {position.symbol} (currently {weight:.1%})"
                        )

            # Check correlation risks
            symbols = [p.symbol for p in portfolio.positions if p.quantity != 0]
            high_correlation_pairs = []

            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i + 1 :]:
                    correlation = await self._get_symbol_correlation(symbol1, symbol2)
                    if correlation > 0.8:
                        high_correlation_pairs.append((symbol1, symbol2, correlation))

            if high_correlation_pairs:
                recommendations.append(
                    f"Consider reducing correlation risk between highly correlated positions: {high_correlation_pairs[0][0]} and {high_correlation_pairs[0][1]}"
                )

            # Market condition recommendations
            if self._market_condition_cache:
                condition, _ = self._market_condition_cache
                if condition == "High Stress":
                    recommendations.append(
                        "Consider reducing overall position sizes due to high market stress"
                    )
                elif condition == "Low Volatility Bull":
                    recommendations.append(
                        "Market conditions favorable for slightly larger position sizes"
                    )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            return ["Unable to generate recommendations due to data access issues"]

    async def _calculate_dynamic_risk_adjustment(self) -> float:
        """Calculate dynamic risk adjustment based on recent portfolio performance."""
        try:
            # Clean old performance data (keep last 30 days)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            self._recent_performance = [
                (ts, pnl) for ts, pnl in self._recent_performance if ts > cutoff_date
            ]

            if len(self._recent_performance) < 5:
                return 1.0  # Not enough data for adjustment

            # Calculate recent performance metrics
            recent_pnls = [pnl for _, pnl in self._recent_performance]

            # Calculate win rate and average returns
            wins = len([pnl for pnl in recent_pnls if pnl > 0])
            win_rate = wins / len(recent_pnls)
            avg_return = sum(recent_pnls) / len(recent_pnls)

            # Calculate volatility of returns
            if len(recent_pnls) > 1:
                variance = sum((pnl - avg_return) ** 2 for pnl in recent_pnls) / len(
                    recent_pnls
                )
                volatility = variance**0.5
            else:
                volatility = 0.1

            # Calculate Sharpe-like ratio
            risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate, daily
            sharpe_ratio = (
                (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
            )

            # Dynamic adjustment based on performance
            base_adjustment = 1.0

            # Adjust based on win rate
            if win_rate > 0.6:
                base_adjustment *= 1.1  # Increase if performing well
            elif win_rate < 0.4:
                base_adjustment *= 0.8  # Decrease if struggling

            # Adjust based on Sharpe ratio
            if sharpe_ratio > 1.0:
                base_adjustment *= 1.15  # Good risk-adjusted returns
            elif sharpe_ratio < -0.5:
                base_adjustment *= 0.7  # Poor risk-adjusted returns

            # Adjust based on recent trend (last 5 trades)
            if len(recent_pnls) >= 5:
                recent_trend = sum(recent_pnls[-5:]) / 5
                if recent_trend > 0.01:  # Recent positive trend
                    base_adjustment *= 1.05
                elif recent_trend < -0.01:  # Recent negative trend
                    base_adjustment *= 0.85

            # Ensure reasonable bounds (0.3x to 1.5x)
            adjustment = max(0.3, min(1.5, base_adjustment))

            logger.debug(
                f"Dynamic risk adjustment: {adjustment:.3f} "
                f"(win_rate: {win_rate:.2f}, sharpe: {sharpe_ratio:.2f}, "
                f"avg_return: {avg_return:.4f})"
            )

            return adjustment

        except Exception as e:
            logger.error(f"Error calculating dynamic risk adjustment: {e}")
            return 0.8  # Conservative on error

    async def _update_circuit_breaker_status(self, portfolio: PortfolioState) -> None:
        """Update circuit breaker status based on portfolio drawdown."""
        try:
            # Get portfolio performance over last 5 days
            recent_performance = [
                pnl
                for ts, pnl in self._recent_performance
                if ts > datetime.now(timezone.utc) - timedelta(days=5)
            ]

            if len(recent_performance) < 3:
                return  # Not enough data

            # Calculate recent cumulative return
            cumulative_return = sum(recent_performance)

            # Circuit breaker thresholds
            major_drawdown_threshold = -0.05  # -5% over 5 days
            severe_drawdown_threshold = -0.08  # -8% over 5 days

            current_time = datetime.now(timezone.utc)

            # Activate circuit breaker on severe drawdown
            if cumulative_return <= severe_drawdown_threshold:
                if not self._circuit_breaker_active:
                    self._circuit_breaker_active = True
                    self._circuit_breaker_timestamp = current_time
                    logger.warning(
                        f"CIRCUIT BREAKER ACTIVATED: Severe drawdown detected ({cumulative_return:.2%})"
                    )

                    # Log circuit breaker event to database
                    await self._log_circuit_breaker_event(
                        "ACTIVATED", cumulative_return
                    )

            # Deactivate circuit breaker if recovering
            elif (
                self._circuit_breaker_active
                and cumulative_return > major_drawdown_threshold
            ):
                # Wait at least 24 hours before deactivating
                if (
                    self._circuit_breaker_timestamp
                    and current_time - self._circuit_breaker_timestamp
                    > timedelta(hours=24)
                ):
                    self._circuit_breaker_active = False
                    logger.info(
                        f"Circuit breaker deactivated: Portfolio recovering ({cumulative_return:.2%})"
                    )

                    # Log deactivation
                    await self._log_circuit_breaker_event(
                        "DEACTIVATED", cumulative_return
                    )

        except Exception as e:
            logger.error(f"Error updating circuit breaker status: {e}")

    async def _apply_dynamic_risk_adjustment(
        self, sizing: PositionSizing, adjustment: float
    ) -> PositionSizing:
        """Apply dynamic risk adjustment to position sizing."""
        try:
            if adjustment == 1.0:
                return sizing  # No adjustment needed

            # Apply adjustment to position size
            original_shares = sizing.recommended_shares
            _ = sizing.recommended_value

            sizing.recommended_shares = int(sizing.recommended_shares * adjustment)
            sizing.recommended_value = sizing.recommended_value * Decimal(
                str(adjustment)
            )
            sizing.position_percentage = sizing.position_percentage * Decimal(
                str(adjustment)
            )

            # Update risk metrics
            sizing.max_loss_amount = (
                sizing.recommended_value * self.risk_limits.stop_loss_percentage
            )
            max_gain = (
                sizing.recommended_value * self.risk_limits.take_profit_percentage
            )
            sizing.risk_reward_ratio = (
                float(max_gain / sizing.max_loss_amount)
                if sizing.max_loss_amount > 0
                else 0.0
            )

            logger.debug(
                f"Applied dynamic risk adjustment {adjustment:.2f} to {sizing.symbol}: "
                f"{original_shares} -> {sizing.recommended_shares} shares"
            )

            return sizing

        except Exception as e:
            logger.error(f"Error applying dynamic risk adjustment: {e}")
            return sizing

    async def _apply_circuit_breaker_adjustment(
        self, sizing: PositionSizing
    ) -> PositionSizing:
        """Apply circuit breaker position size reduction."""
        try:
            # Reduce position size by 70% during circuit breaker
            circuit_breaker_factor = 0.3

            original_shares = sizing.recommended_shares
            sizing.recommended_shares = int(
                sizing.recommended_shares * circuit_breaker_factor
            )
            sizing.recommended_value = sizing.recommended_value * Decimal(
                str(circuit_breaker_factor)
            )
            sizing.position_percentage = sizing.position_percentage * Decimal(
                str(circuit_breaker_factor)
            )

            # Update risk metrics
            sizing.max_loss_amount = (
                sizing.recommended_value * self.risk_limits.stop_loss_percentage
            )
            max_gain = (
                sizing.recommended_value * self.risk_limits.take_profit_percentage
            )
            sizing.risk_reward_ratio = (
                float(max_gain / sizing.max_loss_amount)
                if sizing.max_loss_amount > 0
                else 0.0
            )

            logger.warning(
                f"Circuit breaker applied to {sizing.symbol}: "
                f"{original_shares} -> {sizing.recommended_shares} shares (70% reduction)"
            )

            return sizing

        except Exception as e:
            logger.error(f"Error applying circuit breaker adjustment: {e}")
            return sizing

    async def _track_position_sizing_decision(
        self, sizing: PositionSizing, signal: Optional[TradeSignal]
    ) -> None:
        """Track position sizing decision for performance analysis."""
        try:
            # Ensure database manager is available
            if not self.db_manager:
                logger.warning(
                    "Database not available for tracking position sizing decision"
                )
                return

            # Store sizing decision in database for later analysis
            await self.db_manager.store_position_sizing_history(
                symbol=sizing.symbol,
                signal_timestamp=datetime.now(timezone.utc),
                recommended_shares=sizing.recommended_shares,
                recommended_value=sizing.recommended_value,
                position_percentage=sizing.position_percentage,
                confidence_score=Decimal(
                    str(getattr(signal, "confidence", 0.7) if signal else 0.7)
                ),
                volatility_adjustment=Decimal(str(sizing.volatility_adjustment)),
                sizing_method=sizing.sizing_method.value,
                max_loss_amount=Decimal("0"),
                risk_reward_ratio=Decimal("2.0"),
                portfolio_value=sizing.recommended_value or Decimal("100000"),
            )

        except Exception as e:
            logger.error(f"Error tracking position sizing decision: {e}")

    async def _log_circuit_breaker_event(
        self, event_type: str, drawdown: float
    ) -> None:
        """Log circuit breaker activation/deactivation."""
        try:
            # Ensure database manager is available
            if not self.db_manager:
                logger.warning(
                    "Database not available for logging circuit breaker event"
                )
                return

            await self.db_manager.store_circuit_breaker_event(
                trigger_type=event_type,
                trigger_value=Decimal(str(drawdown)) if drawdown is not None else None,
                threshold_value=None,
                duration_minutes=15,
                portfolio_impact=None,
            )
        except Exception as e:
            logger.error(f"Error logging circuit breaker event: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False

    async def update_performance_tracking(
        self, symbol: str, realized_pnl_percentage: float
    ) -> None:
        """Update performance tracking with realized P&L."""
        try:
            timestamp = datetime.now(timezone.utc)
            self._recent_performance.append((timestamp, realized_pnl_percentage))

            # Keep only last 100 trades to prevent memory issues
            if len(self._recent_performance) > 100:
                self._recent_performance = self._recent_performance[-100:]

            logger.debug(
                f"Updated performance tracking: {symbol} realized {realized_pnl_percentage:.4f}"
            )

        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")

    async def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of recent position sizing performance."""
        try:
            if not self._recent_performance:
                return {}

            recent_pnls = [pnl for _, pnl in self._recent_performance]

            wins = len([pnl for pnl in recent_pnls if pnl > 0])
            _ = len([pnl for pnl in recent_pnls if pnl < 0])

            summary = {
                "total_trades": len(recent_pnls),
                "win_rate": wins / len(recent_pnls) if recent_pnls else 0,
                "avg_return": sum(recent_pnls) / len(recent_pnls) if recent_pnls else 0,
                "total_return": sum(recent_pnls),
                "best_trade": max(recent_pnls) if recent_pnls else 0,
                "worst_trade": min(recent_pnls) if recent_pnls else 0,
                "circuit_breaker_active": self._circuit_breaker_active,
            }

            if len(recent_pnls) > 1:
                avg_return = summary["avg_return"]
                variance = sum((pnl - avg_return) ** 2 for pnl in recent_pnls) / len(
                    recent_pnls
                )
                summary["volatility"] = variance**0.5
                summary["sharpe_ratio"] = (
                    avg_return / summary["volatility"]
                    if summary["volatility"] > 0
                    else 0
                )

            return summary

        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {}


# Usage Examples:
#
# Basic usage with context manager:
# ```python
# risk_limits = RiskLimits(
#     max_position_percentage=Decimal("0.20"),
#     stop_loss_percentage=Decimal("0.02"),
#     take_profit_percentage=Decimal("0.04"),
#     max_positions=20
# )
#
# async with PositionSizer(risk_limits) as sizer:
#     sizing = await sizer.calculate_position_size(
#         symbol="AAPL",
#         current_price=Decimal("150.00"),
#         portfolio=portfolio_state,
#         method=PositionSizingMethod.CONFIDENCE_BASED
#     )
#     print(f"Recommended shares: {sizing.recommended_shares}")
# ```
#
# Manual initialization:
# ```python
# sizer = PositionSizer(risk_limits)
# await sizer.initialize()
# try:
#     sizing = await sizer.calculate_position_size(...)
#     # Update performance when trade is closed
#     await sizer.update_performance_tracking("AAPL", 0.03)  # 3% gain
# finally:
#     await sizer.cleanup()
# ```
#
# Get portfolio recommendations:
# ```python
# async with PositionSizer(risk_limits) as sizer:
#     recommendations = await sizer.get_recommended_portfolio_adjustments(portfolio)
#     for rec in recommendations:
#         print(f"Recommendation: {rec}")
#
#     diversification_score = await sizer.get_portfolio_diversification_score(portfolio)
#     print(f"Diversification score: {diversification_score:.2f}")
# ```

# TODO: Add comprehensive unit tests for all position sizing methods
# TODO: Add integration tests with real market data
# TODO: Add performance benchmarking for position sizing calculations
# TODO: Add validation for edge cases (zero prices, negative portfolio values, etc.)
# TODO: Add machine learning models for dynamic position sizing
# TODO: Add options and derivatives position sizing
# TODO: Add currency hedging considerations for international positions
# TODO: Add ESG (Environmental, Social, Governance) factor adjustments
# TODO: Add real-time market sentiment analysis integration
# TODO: Add stress testing capabilities for extreme market scenarios
