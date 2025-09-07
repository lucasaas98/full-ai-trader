"""
Production Strategy Adapter

This adapter provides a simplified interface to production-like trading strategies
that works around import issues with the full production strategy modules.
It implements the core logic and parameters from the actual production strategies.
"""

import logging
import math
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from backtest_models import MarketData, SignalType

logger = logging.getLogger(__name__)


class StrategyMode(Enum):
    """Strategy execution modes."""

    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"


class ProductionSignal:
    """Production-like trading signal."""

    def __init__(
        self,
        symbol: str,
        action: SignalType,
        confidence: float,
        entry_price: Decimal,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        position_size: float = 0.1,
        reasoning: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.symbol = symbol
        self.action = action
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.reasoning = reasoning
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)


class ProductionStrategyAdapter:
    """
    Adapter that simulates production strategy behavior based on actual parameters
    from the production system configuration.
    """

    def __init__(self, strategy_mode: StrategyMode = StrategyMode.DAY_TRADING):
        self.strategy_mode = strategy_mode
        self.name = (
            f"Production_{strategy_mode.value.title().replace('_', '')}_Strategy"
        )
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

        # Configure strategy parameters based on actual production settings
        self._configure_strategy_parameters()

        # Performance tracking
        self.signals_generated = 0
        self.analysis_calls = 0

    def _configure_strategy_parameters(self):
        """Configure parameters based on production strategy settings."""

        if self.strategy_mode == StrategyMode.DAY_TRADING:
            # Day trading parameters from actual production config
            self.config = {
                "min_confidence": 70.0,
                "max_position_size": 0.15,  # 15% max per position
                "stop_loss_pct": 0.015,  # 1.5% stop loss
                "take_profit_pct": 0.02,  # 2% take profit
                "lookback_period": 30,
                "ta_weight": 0.70,  # 70% technical analysis weight
                "fa_weight": 0.30,  # 30% fundamental analysis weight
                "min_technical_score": 60.0,
                "min_fundamental_score": 30.0,
                "signal_decay_hours": 2,
                "volume_threshold": 2.0,
                "risk_reward_ratio": 1.5,
                "max_hold_days": 1,
            }

        elif self.strategy_mode == StrategyMode.SWING_TRADING:
            # Swing trading parameters from actual production config
            self.config = {
                "min_confidence": 65.0,
                "max_position_size": 0.20,  # 20% max per position
                "stop_loss_pct": 0.03,  # 3% stop loss
                "take_profit_pct": 0.06,  # 6% take profit
                "lookback_period": 50,
                "ta_weight": 0.50,  # 50% technical analysis weight
                "fa_weight": 0.50,  # 50% fundamental analysis weight
                "min_technical_score": 55.0,
                "min_fundamental_score": 40.0,
                "signal_decay_hours": 24,
                "volume_threshold": 1.5,
                "risk_reward_ratio": 2.0,
                "max_hold_days": 7,
            }

        else:  # POSITION_TRADING
            # Position trading parameters from actual production config
            self.config = {
                "min_confidence": 60.0,
                "max_position_size": 0.25,  # 25% max per position
                "stop_loss_pct": 0.05,  # 5% stop loss
                "take_profit_pct": 0.10,  # 10% take profit
                "lookback_period": 100,
                "ta_weight": 0.40,  # 40% technical analysis weight
                "fa_weight": 0.60,  # 60% fundamental analysis weight
                "min_technical_score": 50.0,
                "min_fundamental_score": 50.0,
                "signal_decay_hours": 168,  # 1 week
                "volume_threshold": 1.2,
                "risk_reward_ratio": 2.5,
                "max_hold_days": 30,
            }

    async def analyze_symbol(
        self,
        symbol: str,
        current_data: MarketData,
        historical_data: List[MarketData],
        market_context: Dict[str, Any],
    ) -> Optional[ProductionSignal]:
        """
        Analyze symbol using production strategy logic.

        This simulates the hybrid strategy approach with technical and fundamental analysis.
        """
        try:
            self.analysis_calls += 1

            if len(historical_data) < 20:
                return None

            # Calculate technical indicators (simplified versions of production indicators)
            technical_score = await self._calculate_technical_score(
                current_data, historical_data
            )
            fundamental_score = await self._calculate_fundamental_score(
                current_data, historical_data, market_context
            )

            # Check minimum score thresholds
            if (
                technical_score < self.config["min_technical_score"]
                or fundamental_score < self.config["min_fundamental_score"]
            ):
                return None

            # Calculate hybrid score using production weighting
            hybrid_score = (
                technical_score * self.config["ta_weight"]
                + fundamental_score * self.config["fa_weight"]
            )

            # Apply confidence threshold
            if hybrid_score < self.config["min_confidence"]:
                return None

            # Determine signal action based on technical momentum and trend
            signal_action = await self._determine_signal_action(
                current_data, historical_data, technical_score
            )

            if signal_action == SignalType.HOLD:
                return None

            # Calculate risk management levels
            entry_price = current_data.close
            stop_loss = self._calculate_stop_loss(entry_price, signal_action)
            take_profit = self._calculate_take_profit(entry_price, signal_action)

            # Determine position size based on confidence and volatility
            position_size = self._calculate_position_size(hybrid_score, historical_data)

            # Create reasoning string
            reasoning = self._create_reasoning_string(
                technical_score, fundamental_score, hybrid_score
            )

            # Create signal
            signal = ProductionSignal(
                symbol=symbol,
                action=signal_action,
                confidence=hybrid_score,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                metadata={
                    "technical_score": technical_score,
                    "fundamental_score": fundamental_score,
                    "strategy_mode": self.strategy_mode.value,
                    "ta_weight": self.config["ta_weight"],
                    "fa_weight": self.config["fa_weight"],
                },
            )

            self.signals_generated += 1
            self.logger.debug(
                f"Generated {signal_action.value} signal for {symbol}: {hybrid_score:.1f}% confidence"
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    async def _calculate_technical_score(
        self, current_data: MarketData, historical_data: List[MarketData]
    ) -> float:
        """Calculate technical analysis score (0-100)."""
        try:
            scores: List[float] = []

            # Price trend analysis
            recent_closes = [float(d.close) for d in historical_data[-20:]]
            if len(recent_closes) >= 20:
                sma_5 = sum(recent_closes[-5:]) / 5
                sma_20 = sum(recent_closes) / 20
                current_price = float(current_data.close)

                # Trend strength
                if current_price > sma_5 > sma_20:
                    trend_score = 85
                elif current_price > sma_20:
                    trend_score = 65
                elif current_price > sma_5:
                    trend_score = 45
                else:
                    trend_score = 25

                scores.append(trend_score)

            # Volume analysis
            recent_volumes = [d.volume for d in historical_data[-10:]]
            if recent_volumes:
                avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1])
                current_volume = current_data.volume
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                if volume_ratio >= self.config["volume_threshold"]:
                    volume_score = min(90, 60 + (volume_ratio - 1) * 20)
                else:
                    volume_score = 40

                scores.append(volume_score)

            # RSI-like momentum indicator
            if len(recent_closes) >= 14:
                gains: List[float] = []
                losses: List[float] = []

                for i in range(1, len(recent_closes)):
                    change = recent_closes[i] - recent_closes[i - 1]
                    if change > 0:
                        gains.append(float(change))
                        losses.append(0.0)
                    else:
                        gains.append(0.0)
                        losses.append(float(abs(change)))

                avg_gain = sum(gains[-14:]) / 14 if gains else 0.01
                avg_loss = sum(losses[-14:]) / 14 if losses else 0.01

                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

                # RSI scoring based on strategy mode
                if self.strategy_mode == StrategyMode.DAY_TRADING:
                    if 30 <= rsi <= 70:  # Sweet spot for day trading
                        rsi_score = 80
                    elif rsi < 30 or rsi > 70:
                        rsi_score = 60  # Potential reversal
                    else:
                        rsi_score = 40
                else:  # Swing/Position trading
                    if 40 <= rsi <= 60:
                        rsi_score = 75
                    elif 30 <= rsi <= 70:
                        rsi_score = 60
                    else:
                        rsi_score = 35

                scores.append(rsi_score)

            # Volatility analysis
            if len(recent_closes) >= 10:
                volatility = self._calculate_volatility(recent_closes[-10:])

                # Optimal volatility ranges by strategy
                if self.strategy_mode == StrategyMode.DAY_TRADING:
                    optimal_vol = (0.015, 0.04)  # 1.5% - 4% daily volatility
                elif self.strategy_mode == StrategyMode.SWING_TRADING:
                    optimal_vol = (0.02, 0.06)  # 2% - 6% daily volatility
                else:
                    optimal_vol = (0.015, 0.05)  # 1.5% - 5% daily volatility

                if optimal_vol[0] <= volatility <= optimal_vol[1]:
                    vol_score = 80
                elif (
                    volatility < optimal_vol[0] * 0.5 or volatility > optimal_vol[1] * 2
                ):
                    vol_score = 30
                else:
                    vol_score = 55

                scores.append(vol_score)

            # Return weighted average of all technical scores
            return sum(scores) / len(scores) if scores else 50.0

        except Exception as e:
            self.logger.debug(f"Error in technical analysis: {e}")
            return 50.0

    async def _calculate_fundamental_score(
        self,
        current_data: MarketData,
        historical_data: List[MarketData],
        market_context: Dict[str, Any],
    ) -> float:
        """Calculate fundamental analysis score (0-100) using price-based proxies."""
        try:
            scores: List[float] = []

            # Value analysis (price relative to historical levels)
            if len(historical_data) >= 252:  # ~1 year of data
                year_high = max(float(d.high) for d in historical_data[-252:])
                year_low = min(float(d.low) for d in historical_data[-252:])
                current_price = float(current_data.close)

                # Position within 52-week range
                range_position = (
                    (current_price - year_low) / (year_high - year_low)
                    if year_high > year_low
                    else 0.5
                )

                # Value scoring - different preferences by strategy
                if self.strategy_mode == StrategyMode.POSITION_TRADING:
                    # Position trading prefers value opportunities
                    if range_position < 0.3:  # Near lows
                        value_score = 85
                    elif range_position < 0.6:
                        value_score = 70
                    else:
                        value_score = 40
                else:
                    # Day/swing trading prefer momentum
                    if range_position > 0.7:  # Near highs
                        value_score = 75
                    elif range_position > 0.4:
                        value_score = 60
                    else:
                        value_score = 45

                scores.append(value_score)

            # Growth proxy (price momentum over different periods)
            if len(historical_data) >= 60:
                current_price = float(current_data.close)

                # 1-month growth
                month_ago_price = (
                    float(historical_data[-21].close)
                    if len(historical_data) >= 21
                    else current_price
                )
                monthly_growth = (
                    (current_price - month_ago_price) / month_ago_price
                    if month_ago_price > 0
                    else 0
                )

                # 3-month growth
                quarter_ago_price = (
                    float(historical_data[-63].close)
                    if len(historical_data) >= 63
                    else current_price
                )
                quarterly_growth = (
                    (current_price - quarter_ago_price) / quarter_ago_price
                    if quarter_ago_price > 0
                    else 0
                )

                # Growth scoring
                if self.strategy_mode == StrategyMode.DAY_TRADING:
                    # Day trading focuses on recent momentum
                    if monthly_growth > 0.1:  # 10%+ monthly growth
                        growth_score = 85
                    elif monthly_growth > 0.05:
                        growth_score = 70
                    elif monthly_growth > 0:
                        growth_score = 55
                    else:
                        growth_score = 30
                else:
                    # Swing/position trading considers longer trends
                    if (
                        quarterly_growth > 0.2 and monthly_growth > 0
                    ):  # Sustained growth
                        growth_score = 90
                    elif quarterly_growth > 0.1:
                        growth_score = 75
                    elif monthly_growth > 0.05:
                        growth_score = 60
                    else:
                        growth_score = 40

                scores.append(growth_score)

            # Quality proxy (price stability and trend consistency)
            if len(historical_data) >= 30:
                recent_prices = [float(d.close) for d in historical_data[-30:]]

                # Price stability (inverse of volatility)
                volatility = self._calculate_volatility(recent_prices)
                stability_score = max(
                    30, min(90, 90 - (volatility * 1000))
                )  # Scale volatility to score

                scores.append(stability_score)

                # Trend consistency
                positive_days = sum(
                    1
                    for i in range(1, len(recent_prices))
                    if recent_prices[i] > recent_prices[i - 1]
                )
                consistency = positive_days / (len(recent_prices) - 1)

                consistency_score = 30 + (consistency * 40)  # 30-70 range
                scores.append(consistency_score)

            # Market context scoring
            # portfolio_value = market_context.get("portfolio_value", 100000)  # Unused for now
            positions_count = market_context.get("positions_count", 0)

            # Prefer diversification
            if positions_count < 3:
                diversification_score = 80
            elif positions_count < 6:
                diversification_score = 70
            else:
                diversification_score = 50

            scores.append(diversification_score)

            # Return weighted average
            return sum(scores) / len(scores) if scores else 50.0

        except Exception as e:
            self.logger.debug(f"Error in fundamental analysis: {e}")
            return 50.0

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility."""
        if len(prices) < 2:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            ret = (
                (prices[i] - prices[i - 1]) / prices[i - 1] if prices[i - 1] > 0 else 0
            )
            returns.append(ret)

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

        return math.sqrt(variance)

    async def _determine_signal_action(
        self,
        current_data: MarketData,
        historical_data: List[MarketData],
        technical_score: float,
    ) -> SignalType:
        """Determine the signal action based on analysis."""
        try:
            # Default to no action
            if technical_score < self.config["min_technical_score"]:
                return SignalType.HOLD

            # Analyze recent price action
            recent_closes = [float(d.close) for d in historical_data[-5:]]
            current_price = float(current_data.close)

            if len(recent_closes) >= 2:
                recent_trend = (current_price - recent_closes[0]) / recent_closes[0]

                # Volume confirmation
                current_volume = current_data.volume
                avg_volume = sum(d.volume for d in historical_data[-10:]) / 10
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # Signal logic based on strategy mode
                if self.strategy_mode == StrategyMode.DAY_TRADING:
                    # Day trading: look for momentum with volume
                    if (
                        recent_trend > 0.01 and volume_ratio >= 1.5
                    ):  # 1%+ move with volume
                        return SignalType.BUY
                    elif (
                        recent_trend < -0.01 and volume_ratio >= 1.5
                    ):  # Short opportunity
                        return SignalType.SELL

                elif self.strategy_mode == StrategyMode.SWING_TRADING:
                    # Swing trading: look for sustained moves
                    if (
                        recent_trend > 0.02 and technical_score > 70
                    ):  # 2%+ move, strong technicals
                        return SignalType.BUY
                    elif recent_trend < -0.02 and technical_score < 40:
                        return SignalType.SELL

                else:  # Position trading
                    # Position trading: look for longer-term setups
                    if (
                        recent_trend > 0.03 and technical_score > 65
                    ):  # 3%+ move, good technicals
                        return SignalType.BUY
                    elif recent_trend < -0.03 and technical_score < 35:
                        return SignalType.SELL

            return SignalType.HOLD

        except Exception as e:
            self.logger.debug(f"Error determining signal action: {e}")
            return SignalType.HOLD

    def _calculate_stop_loss(
        self, entry_price: Decimal, action: SignalType
    ) -> Optional[Decimal]:
        """Calculate stop loss level."""
        if action == SignalType.BUY:
            return entry_price * (
                Decimal("1") - Decimal(str(self.config["stop_loss_pct"]))
            )
        elif action == SignalType.SELL:
            return entry_price * (
                Decimal("1") + Decimal(str(self.config["stop_loss_pct"]))
            )
        return None

    def _calculate_take_profit(
        self, entry_price: Decimal, action: SignalType
    ) -> Optional[Decimal]:
        """Calculate take profit level."""
        if action == SignalType.BUY:
            return entry_price * (
                Decimal("1") + Decimal(str(self.config["take_profit_pct"]))
            )
        elif action == SignalType.SELL:
            return entry_price * (
                Decimal("1") - Decimal(str(self.config["take_profit_pct"]))
            )
        return None

    def _calculate_position_size(
        self, confidence: float, historical_data: List[MarketData]
    ) -> float:
        """Calculate position size based on confidence and volatility."""
        try:
            # Base position size
            base_size = self.config["max_position_size"]

            # Confidence adjustment
            confidence_multiplier = min(1.0, confidence / 100.0)

            # Volatility adjustment
            if len(historical_data) >= 10:
                recent_closes = [float(d.close) for d in historical_data[-10:]]
                volatility = self._calculate_volatility(recent_closes)

                # Reduce position size for high volatility
                vol_multiplier = max(0.5, min(1.0, 0.03 / max(volatility, 0.01)))
            else:
                vol_multiplier = 1.0

            # Strategy-specific adjustments
            if self.strategy_mode == StrategyMode.DAY_TRADING:
                strategy_multiplier = 0.8  # Smaller positions for day trading
            elif self.strategy_mode == StrategyMode.POSITION_TRADING:
                strategy_multiplier = 1.2  # Larger positions for position trading
            else:
                strategy_multiplier = 1.0

            final_size = (
                base_size * confidence_multiplier * vol_multiplier * strategy_multiplier
            )
            return max(0.05, min(self.config["max_position_size"], final_size))

        except Exception as e:
            self.logger.debug(f"Error calculating position size: {e}")
            return self.config["max_position_size"] * 0.5

    def _create_reasoning_string(
        self, technical_score: float, fundamental_score: float, hybrid_score: float
    ) -> str:
        """Create human-readable reasoning for the signal."""
        return (
            f"{self.strategy_mode.value.replace('_', ' ').title()}: "
            f"Tech={technical_score:.1f}, Fund={fundamental_score:.1f}, "
            f"Hybrid={hybrid_score:.1f}"
        )


class ProductionStrategyFactory:
    """Factory for creating production strategy adapters."""

    @staticmethod
    def create_day_trading_strategy() -> ProductionStrategyAdapter:
        """Create day trading strategy."""
        return ProductionStrategyAdapter(StrategyMode.DAY_TRADING)

    @staticmethod
    def create_swing_trading_strategy() -> ProductionStrategyAdapter:
        """Create swing trading strategy."""
        return ProductionStrategyAdapter(StrategyMode.SWING_TRADING)

    @staticmethod
    def create_position_trading_strategy() -> ProductionStrategyAdapter:
        """Create position trading strategy."""
        return ProductionStrategyAdapter(StrategyMode.POSITION_TRADING)

    @staticmethod
    def create_all_strategies() -> List[ProductionStrategyAdapter]:
        """Create all three strategy types."""
        return [
            ProductionStrategyFactory.create_day_trading_strategy(),
            ProductionStrategyFactory.create_swing_trading_strategy(),
            ProductionStrategyFactory.create_position_trading_strategy(),
        ]


# Indicate that production strategies are available through the adapter
PRODUCTION_STRATEGIES_AVAILABLE = True

logger.info("Production strategy adapter initialized successfully")
