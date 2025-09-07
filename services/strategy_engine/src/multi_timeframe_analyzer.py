"""
Multi-Timeframe Confirmation Module

This module provides sophisticated multi-timeframe analysis and confirmation
for trading signals. It ensures signal alignment across different time horizons
before confirming entries, significantly improving signal quality and reducing
false signals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyMode

logger = logging.getLogger(__name__)


class TimeFrameAlignment(Enum):
    """Different types of timeframe alignment patterns."""

    BULLISH_ALIGNMENT = "bullish_alignment"  # All timeframes bullish
    BEARISH_ALIGNMENT = "bearish_alignment"  # All timeframes bearish
    MIXED_BULLISH = "mixed_bullish"  # Majority bullish
    MIXED_BEARISH = "mixed_bearish"  # Majority bearish
    CONFLICTED = "conflicted"  # No clear direction
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough timeframes


class ConfirmationStrength(Enum):
    """Strength levels for multi-timeframe confirmation."""

    VERY_STRONG = "very_strong"  # 100% alignment
    STRONG = "strong"  # 80%+ alignment
    MODERATE = "moderate"  # 60%+ alignment
    WEAK = "weak"  # 40%+ alignment
    VERY_WEAK = "very_weak"  # <40% alignment


@dataclass
class TimeFrameSignal:
    """Individual timeframe signal with metadata."""

    timeframe: str
    signal_type: SignalType
    confidence: float
    strength: float
    trend_direction: str  # "up", "down", "sideways"
    momentum: float
    volume_confirmation: bool
    support_resistance_level: Optional[float] = None
    key_indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MultiTimeFrameConfirmation:
    """Results of multi-timeframe confirmation analysis."""

    primary_signal: SignalType
    confirmation_strength: ConfirmationStrength
    alignment_type: TimeFrameAlignment
    overall_confidence: float
    timeframe_signals: List[TimeFrameSignal]
    confirmation_score: float  # 0-100
    conflicting_timeframes: List[str]
    supporting_timeframes: List[str]
    key_confluence_factors: List[str]
    risk_factors: List[str]
    optimal_entry_timeframe: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiTimeFrameAnalyzer:
    """
    Advanced multi-timeframe confirmation analyzer.

    Analyzes signals across multiple timeframes to provide robust
    confirmation before allowing trade entries.
    """

    def __init__(self, strategy_mode: StrategyMode):
        """
        Initialize multi-timeframe analyzer.

        Args:
            strategy_mode: Trading strategy mode to determine timeframe combinations
        """
        self.strategy_mode = strategy_mode
        self.logger = logging.getLogger(f"{__name__}.{strategy_mode.value}")

        # Configure timeframe hierarchies based on strategy mode
        self.timeframe_config = self._get_timeframe_config()

        # Confirmation thresholds
        self.confirmation_thresholds = {
            ConfirmationStrength.VERY_STRONG: 0.9,
            ConfirmationStrength.STRONG: 0.8,
            ConfirmationStrength.MODERATE: 0.6,
            ConfirmationStrength.WEAK: 0.4,
            ConfirmationStrength.VERY_WEAK: 0.0,
        }

        # Signal quality weights for different timeframes
        self.timeframe_weights = self._get_timeframe_weights()

    def _get_timeframe_config(self) -> Dict[str, Any]:
        """Get timeframe configuration based on strategy mode."""
        configs = {
            StrategyMode.DAY_TRADING: {
                "primary_timeframes": ["5m", "15m"],
                "confirmation_timeframes": ["1m", "30m", "1h"],
                "trend_timeframes": ["1h", "4h"],
                "min_confirmation_count": 2,
                "max_conflicting_allowed": 1,
            },
            StrategyMode.SWING_TRADING: {
                "primary_timeframes": ["1h", "4h"],
                "confirmation_timeframes": ["15m", "30m", "1d"],
                "trend_timeframes": ["1d", "1w"],
                "min_confirmation_count": 3,
                "max_conflicting_allowed": 1,
            },
            StrategyMode.POSITION_TRADING: {
                "primary_timeframes": ["1d", "1w"],
                "confirmation_timeframes": ["4h", "12h", "1M"],
                "trend_timeframes": ["1M", "3M"],
                "min_confirmation_count": 3,
                "max_conflicting_allowed": 2,
            },
        }

        return configs.get(self.strategy_mode, configs[StrategyMode.SWING_TRADING])

    def _get_timeframe_weights(self) -> Dict[str, float]:
        """Get importance weights for different timeframes."""
        weight_configs = {
            StrategyMode.DAY_TRADING: {
                "1m": 0.15,
                "5m": 0.25,
                "15m": 0.30,
                "30m": 0.20,
                "1h": 0.10,
            },
            StrategyMode.SWING_TRADING: {
                "15m": 0.10,
                "30m": 0.15,
                "1h": 0.25,
                "4h": 0.30,
                "1d": 0.20,
            },
            StrategyMode.POSITION_TRADING: {
                "4h": 0.10,
                "12h": 0.15,
                "1d": 0.25,
                "1w": 0.30,
                "1M": 0.20,
            },
        }

        return weight_configs.get(
            self.strategy_mode, weight_configs[StrategyMode.SWING_TRADING]
        )

    async def confirm_signal(
        self,
        symbol: str,
        primary_signal: Signal,
        multi_timeframe_data: Dict[str, pl.DataFrame],
        strategy_instance: Optional[BaseStrategy] = None,
    ) -> MultiTimeFrameConfirmation:
        """
        Perform multi-timeframe confirmation analysis.

        Args:
            symbol: Trading symbol
            primary_signal: Primary signal to confirm
            multi_timeframe_data: Data for different timeframes
            strategy_instance: Strategy instance for analysis

        Returns:
            Multi-timeframe confirmation results
        """
        try:
            # Analyze each timeframe
            timeframe_signals = []
            available_timeframes = list(multi_timeframe_data.keys())

            if (
                len(available_timeframes)
                < self.timeframe_config["min_confirmation_count"]
            ):
                return self._create_insufficient_data_result(
                    primary_signal, available_timeframes
                )

            # Generate signals for each timeframe
            for timeframe in available_timeframes:
                if timeframe in multi_timeframe_data:
                    tf_signal = await self._analyze_timeframe(
                        symbol,
                        timeframe,
                        multi_timeframe_data[timeframe],
                        strategy_instance,
                    )
                    if tf_signal:
                        timeframe_signals.append(tf_signal)

            # Perform confirmation analysis
            confirmation = await self._perform_confirmation_analysis(
                primary_signal, timeframe_signals
            )

            # Add risk factors and confluence analysis
            confirmation = self._enhance_confirmation_with_analysis(
                confirmation, timeframe_signals, multi_timeframe_data
            )

            return confirmation

        except Exception as e:
            self.logger.error(
                f"Error in multi-timeframe confirmation for {symbol}: {e}"
            )
            return self._create_error_result(primary_signal, str(e))

    async def _analyze_timeframe(
        self,
        symbol: str,
        timeframe: str,
        data: pl.DataFrame,
        strategy_instance: Optional[BaseStrategy],
    ) -> Optional[TimeFrameSignal]:
        """Analyze individual timeframe for signal generation."""
        try:
            if data.height < 20:  # Need minimum data points
                return None

            # Basic technical analysis for this timeframe
            close_prices = data.select("close").to_numpy().flatten()
            high_prices = data.select("high").to_numpy().flatten()
            low_prices = data.select("low").to_numpy().flatten()
            volumes = (
                data.select("volume").to_numpy().flatten()
                if "volume" in data.columns
                else None
            )

            # Calculate key indicators
            indicators = self._calculate_timeframe_indicators(
                close_prices, high_prices, low_prices, volumes
            )

            # Determine trend direction
            trend_direction = self._determine_trend_direction(close_prices, indicators)

            # Calculate signal strength
            strength = self._calculate_signal_strength(indicators, trend_direction)

            # Determine signal type
            signal_type = self._determine_signal_type(
                indicators, trend_direction, strength
            )

            # Calculate confidence
            confidence = self._calculate_timeframe_confidence(
                indicators, trend_direction, strength
            )

            # Volume confirmation
            volume_confirmation = (
                self._check_volume_confirmation(volumes, trend_direction)
                if volumes is not None
                else False
            )

            return TimeFrameSignal(
                timeframe=timeframe,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                trend_direction=trend_direction,
                momentum=indicators.get("momentum", 0.0),
                volume_confirmation=volume_confirmation,
                key_indicators=indicators,
            )

        except Exception as e:
            self.logger.error(
                f"Error analyzing timeframe {timeframe} for {symbol}: {e}"
            )
            return None

    def _calculate_timeframe_indicators(
        self,
        close_prices: Any,
        high_prices: Any,
        low_prices: Any,
        volumes: Optional[Any],
    ) -> Dict[str, float]:
        """Calculate key technical indicators for timeframe analysis."""
        try:
            import numpy as np

            # Convert to numpy arrays if needed
            close = np.array(close_prices)
            high = np.array(high_prices)
            low = np.array(low_prices)

            indicators = {}

            # Moving averages
            if len(close) >= 20:
                indicators["sma_20"] = np.mean(close[-20:])
                indicators["sma_50"] = (
                    np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)
                )

            # Price position relative to moving averages
            current_price = close[-1]
            if "sma_20" in indicators:
                indicators["price_vs_sma20"] = (
                    current_price - indicators["sma_20"]
                ) / indicators["sma_20"]

            # Simple momentum (rate of change)
            if len(close) >= 10:
                indicators["momentum"] = (current_price - close[-10]) / close[-10]

            # Recent high/low analysis
            if len(close) >= 14:
                recent_high = np.max(high[-14:])
                recent_low = np.min(low[-14:])
                indicators["high_low_position"] = (current_price - recent_low) / (
                    recent_high - recent_low
                )

            # Volatility (simple standard deviation)
            if len(close) >= 20:
                returns = np.diff(close[-20:]) / close[-20:-1]
                indicators["volatility"] = np.std(returns)

            # Volume trend if available
            if volumes is not None and len(volumes) >= 10:
                vol = np.array(volumes)
                recent_vol_avg = np.mean(vol[-10:])
                longer_vol_avg = (
                    np.mean(vol[-20:]) if len(vol) >= 20 else recent_vol_avg
                )
                indicators["volume_trend"] = (
                    (recent_vol_avg - longer_vol_avg) / longer_vol_avg
                    if longer_vol_avg > 0
                    else 0
                )

            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating timeframe indicators: {e}")
            return {}

    def _determine_trend_direction(
        self, close_prices: Any, indicators: Dict[str, float]
    ) -> str:
        """Determine trend direction for the timeframe."""
        try:
            import numpy as np

            close = np.array(close_prices)

            # Use multiple factors to determine trend
            trend_signals = []

            # Price vs SMA
            if "price_vs_sma20" in indicators:
                if indicators["price_vs_sma20"] > 0.02:  # 2% above SMA
                    trend_signals.append("up")
                elif indicators["price_vs_sma20"] < -0.02:  # 2% below SMA
                    trend_signals.append("down")
                else:
                    trend_signals.append("sideways")

            # Momentum
            if "momentum" in indicators:
                if indicators["momentum"] > 0.01:  # 1% positive momentum
                    trend_signals.append("up")
                elif indicators["momentum"] < -0.01:  # 1% negative momentum
                    trend_signals.append("down")
                else:
                    trend_signals.append("sideways")

            # High/Low position
            if "high_low_position" in indicators:
                if indicators["high_low_position"] > 0.7:  # Near recent highs
                    trend_signals.append("up")
                elif indicators["high_low_position"] < 0.3:  # Near recent lows
                    trend_signals.append("down")
                else:
                    trend_signals.append("sideways")

            # Simple moving average slope
            if len(close) >= 10:
                recent_slope = (close[-1] - close[-10]) / 10
                price_change_pct = recent_slope / close[-10]
                if price_change_pct > 0.005:  # 0.5% positive slope
                    trend_signals.append("up")
                elif price_change_pct < -0.005:  # 0.5% negative slope
                    trend_signals.append("down")
                else:
                    trend_signals.append("sideways")

            # Determine consensus
            if not trend_signals:
                return "sideways"

            up_count = trend_signals.count("up")
            down_count = trend_signals.count("down")
            sideways_count = trend_signals.count("sideways")

            if up_count > down_count and up_count > sideways_count:
                return "up"
            elif down_count > up_count and down_count > sideways_count:
                return "down"
            else:
                return "sideways"

        except Exception as e:
            self.logger.error(f"Error determining trend direction: {e}")
            return "sideways"

    def _calculate_signal_strength(
        self, indicators: Dict[str, float], trend_direction: str
    ) -> float:
        """Calculate signal strength based on indicators alignment."""
        try:
            strength_factors = []

            # Momentum strength
            if "momentum" in indicators:
                momentum_strength = min(
                    abs(indicators["momentum"]) * 10, 1.0
                )  # Scale to 0-1
                strength_factors.append(momentum_strength)

            # Price position strength
            if "high_low_position" in indicators:
                pos = indicators["high_low_position"]
                if trend_direction == "up":
                    position_strength = pos  # Higher is stronger for uptrend
                elif trend_direction == "down":
                    position_strength = 1.0 - pos  # Lower is stronger for downtrend
                else:
                    position_strength = 0.5  # Neutral for sideways
                strength_factors.append(position_strength)

            # SMA relationship strength
            if "price_vs_sma20" in indicators:
                sma_strength = min(
                    abs(indicators["price_vs_sma20"]) * 5, 1.0
                )  # Scale to 0-1
                strength_factors.append(sma_strength)

            # Volume confirmation strength
            if "volume_trend" in indicators and trend_direction != "sideways":
                vol_trend = indicators["volume_trend"]
                if (trend_direction == "up" and vol_trend > 0) or (
                    trend_direction == "down" and vol_trend > 0
                ):
                    volume_strength = min(abs(vol_trend) * 2, 1.0)
                    strength_factors.append(volume_strength)

            return (
                sum(strength_factors) / len(strength_factors)
                if strength_factors
                else 0.5
            )

        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.5

    def _determine_signal_type(
        self, indicators: Dict[str, float], trend_direction: str, strength: float
    ) -> SignalType:
        """Determine signal type based on analysis."""
        try:
            # Require minimum strength for directional signals
            min_strength_threshold = 0.6

            if trend_direction == "up" and strength >= min_strength_threshold:
                return SignalType.BUY
            elif trend_direction == "down" and strength >= min_strength_threshold:
                return SignalType.SELL
            else:
                return SignalType.HOLD

        except Exception as e:
            self.logger.error(f"Error determining signal type: {e}")
            return SignalType.HOLD

    def _calculate_timeframe_confidence(
        self, indicators: Dict[str, float], trend_direction: str, strength: float
    ) -> float:
        """Calculate confidence level for timeframe signal."""
        try:
            base_confidence = strength * 100  # Convert to percentage

            # Boost confidence for strong trend signals
            if trend_direction != "sideways":
                # Momentum boost
                if "momentum" in indicators:
                    momentum_boost = min(
                        abs(indicators["momentum"]) * 20, 10
                    )  # Max 10 point boost
                    base_confidence += momentum_boost

                # Volume confirmation boost
                if "volume_trend" in indicators:
                    vol_trend = indicators["volume_trend"]
                    if (trend_direction == "up" and vol_trend > 0.1) or (
                        trend_direction == "down" and vol_trend > 0.1
                    ):
                        base_confidence += 5

            return min(base_confidence, 100.0)

        except Exception as e:
            self.logger.error(f"Error calculating timeframe confidence: {e}")
            return 50.0

    def _check_volume_confirmation(self, volumes: Any, trend_direction: str) -> bool:
        """Check if volume confirms the trend direction."""
        try:
            import numpy as np

            if volumes is None or len(volumes) < 10:
                return False

            vol = np.array(volumes)
            recent_vol_avg = np.mean(vol[-5:])  # Last 5 periods
            longer_vol_avg = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)

            # Volume should be increasing for strong directional moves
            volume_increase = recent_vol_avg > longer_vol_avg * 1.2  # 20% increase

            return volume_increase if trend_direction != "sideways" else True

        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False

    async def _perform_confirmation_analysis(
        self, primary_signal: Signal, timeframe_signals: List[TimeFrameSignal]
    ) -> MultiTimeFrameConfirmation:
        """Perform comprehensive confirmation analysis."""
        try:
            if not timeframe_signals:
                return self._create_insufficient_data_result(primary_signal, [])

            # Separate signals by type
            buy_signals = [
                s for s in timeframe_signals if s.signal_type == SignalType.BUY
            ]
            sell_signals = [
                s for s in timeframe_signals if s.signal_type == SignalType.SELL
            ]
            hold_signals = [
                s for s in timeframe_signals if s.signal_type == SignalType.HOLD
            ]

            total_signals = len(timeframe_signals)
            buy_count = len(buy_signals)
            sell_count = len(sell_signals)
            hold_count = len(hold_signals)

            # Calculate weighted scores
            weighted_buy_score = sum(
                s.confidence * self.timeframe_weights.get(s.timeframe, 0.2)
                for s in buy_signals
            )
            weighted_sell_score = sum(
                s.confidence * self.timeframe_weights.get(s.timeframe, 0.2)
                for s in sell_signals
            )

            # Determine primary signal and alignment
            if weighted_buy_score > weighted_sell_score and buy_count >= sell_count:
                confirmed_signal = SignalType.BUY
                supporting_timeframes = [s.timeframe for s in buy_signals]
                conflicting_timeframes = [s.timeframe for s in sell_signals]
                dominant_score = weighted_buy_score
            elif weighted_sell_score > weighted_buy_score and sell_count >= buy_count:
                confirmed_signal = SignalType.SELL
                supporting_timeframes = [s.timeframe for s in sell_signals]
                conflicting_timeframes = [s.timeframe for s in buy_signals]
                dominant_score = weighted_sell_score
            else:
                confirmed_signal = SignalType.HOLD
                supporting_timeframes = [s.timeframe for s in hold_signals]
                conflicting_timeframes = [
                    s.timeframe for s in buy_signals + sell_signals
                ]
                dominant_score = 50.0

            # Calculate confirmation metrics
            support_ratio = len(supporting_timeframes) / total_signals
            conflict_ratio = len(conflicting_timeframes) / total_signals

            # Determine confirmation strength
            confirmation_strength = self._determine_confirmation_strength(
                support_ratio, conflict_ratio
            )

            # Determine alignment type
            alignment_type = self._determine_alignment_type(
                buy_count, sell_count, hold_count, total_signals
            )

            # Calculate overall confidence
            base_confidence = (
                dominant_score / len(supporting_timeframes)
                if supporting_timeframes
                else 50.0
            )
            confidence_penalty = conflict_ratio * 30  # Reduce confidence for conflicts
            overall_confidence = max(base_confidence - confidence_penalty, 0.0)

            # Calculate confirmation score (0-100)
            confirmation_score = support_ratio * 100

            # Find optimal entry timeframe
            optimal_entry_tf = self._find_optimal_entry_timeframe(
                supporting_timeframes, timeframe_signals
            )

            return MultiTimeFrameConfirmation(
                primary_signal=confirmed_signal,
                confirmation_strength=confirmation_strength,
                alignment_type=alignment_type,
                overall_confidence=overall_confidence,
                timeframe_signals=timeframe_signals,
                confirmation_score=confirmation_score,
                conflicting_timeframes=conflicting_timeframes,
                supporting_timeframes=supporting_timeframes,
                key_confluence_factors=[],  # Will be populated later
                risk_factors=[],  # Will be populated later
                optimal_entry_timeframe=optimal_entry_tf,
            )

        except Exception as e:
            self.logger.error(f"Error in confirmation analysis: {e}")
            return self._create_error_result(primary_signal, str(e))

    def _determine_confirmation_strength(
        self, support_ratio: float, conflict_ratio: float
    ) -> ConfirmationStrength:
        """Determine confirmation strength based on support/conflict ratios."""
        if support_ratio >= 0.9:
            return ConfirmationStrength.VERY_STRONG
        elif support_ratio >= 0.8:
            return ConfirmationStrength.STRONG
        elif support_ratio >= 0.6:
            return ConfirmationStrength.MODERATE
        elif support_ratio >= 0.4:
            return ConfirmationStrength.WEAK
        else:
            return ConfirmationStrength.VERY_WEAK

    def _determine_alignment_type(
        self, buy_count: int, sell_count: int, hold_count: int, total: int
    ) -> TimeFrameAlignment:
        """Determine the type of timeframe alignment."""
        buy_ratio = buy_count / total
        sell_ratio = sell_count / total

        if buy_ratio >= 0.8:
            return TimeFrameAlignment.BULLISH_ALIGNMENT
        elif sell_ratio >= 0.8:
            return TimeFrameAlignment.BEARISH_ALIGNMENT
        elif buy_ratio >= 0.6:
            return TimeFrameAlignment.MIXED_BULLISH
        elif sell_ratio >= 0.6:
            return TimeFrameAlignment.MIXED_BEARISH
        else:
            return TimeFrameAlignment.CONFLICTED

    def _find_optimal_entry_timeframe(
        self, supporting_timeframes: List[str], all_signals: List[TimeFrameSignal]
    ) -> Optional[str]:
        """Find the optimal timeframe for entry based on signal quality."""
        if not supporting_timeframes:
            return None

        # Filter to supporting signals only
        supporting_signals = [
            s for s in all_signals if s.timeframe in supporting_timeframes
        ]

        if not supporting_signals:
            return None

        # Score each timeframe based on confidence, strength, and weight
        scored_timeframes = []
        for signal in supporting_signals:
            weight = self.timeframe_weights.get(signal.timeframe, 0.2)
            score = (
                (signal.confidence * 0.4)
                + (signal.strength * 100 * 0.4)
                + (weight * 100 * 0.2)
            )
            scored_timeframes.append((signal.timeframe, score))

        # Return timeframe with highest score
        best_timeframe = max(scored_timeframes, key=lambda x: x[1])
        return best_timeframe[0]

    def _enhance_confirmation_with_analysis(
        self,
        confirmation: MultiTimeFrameConfirmation,
        timeframe_signals: List[TimeFrameSignal],
        multi_timeframe_data: Dict[str, pl.DataFrame],
    ) -> MultiTimeFrameConfirmation:
        """Enhance confirmation with additional analysis factors."""
        try:
            confluence_factors = []
            risk_factors = []

            # Analyze confluence factors
            volume_confirmed_count = sum(
                1 for s in timeframe_signals if s.volume_confirmation
            )
            if volume_confirmed_count >= len(timeframe_signals) * 0.6:
                confluence_factors.append(
                    "Strong volume confirmation across timeframes"
                )

            # Check for momentum alignment
            momentum_values = [
                s.momentum for s in timeframe_signals if hasattr(s, "momentum")
            ]
            if momentum_values:
                avg_momentum = sum(momentum_values) / len(momentum_values)
                if abs(avg_momentum) > 0.02:  # 2% average momentum
                    direction = "positive" if avg_momentum > 0 else "negative"
                    confluence_factors.append(f"Strong {direction} momentum alignment")

            # Check for trend consistency
            trend_directions = [s.trend_direction for s in timeframe_signals]
            dominant_trend = max(set(trend_directions), key=trend_directions.count)
            trend_consistency = trend_directions.count(dominant_trend) / len(
                trend_directions
            )

            if trend_consistency >= 0.75:
                confluence_factors.append(f"High trend consistency ({dominant_trend})")
            elif trend_consistency < 0.5:
                risk_factors.append("Conflicting trend directions across timeframes")

            # Analyze risk factors
            if (
                len(confirmation.conflicting_timeframes)
                > self.timeframe_config["max_conflicting_allowed"]
            ):
                risk_factors.append("Too many conflicting timeframes")

            # Check for low confidence signals
            low_confidence_count = sum(
                1 for s in timeframe_signals if s.confidence < 60
            )
            if low_confidence_count > len(timeframe_signals) * 0.4:
                risk_factors.append("Multiple low-confidence timeframe signals")

            # Update confirmation object
            confirmation.key_confluence_factors = confluence_factors
            confirmation.risk_factors = risk_factors

            # Adjust overall confidence based on confluence and risk factors
            confluence_boost = (
                len(confluence_factors) * 5
            )  # 5 points per confluence factor
            risk_penalty = len(risk_factors) * 10  # 10 points per risk factor

            confirmation.overall_confidence = min(
                max(
                    confirmation.overall_confidence + confluence_boost - risk_penalty,
                    0.0,
                ),
                100.0,
            )

            return confirmation

        except Exception as e:
            self.logger.error(f"Error enhancing confirmation analysis: {e}")
            return confirmation

    def _create_insufficient_data_result(
        self, primary_signal: Signal, available_timeframes: List[str]
    ) -> MultiTimeFrameConfirmation:
        """Create result for insufficient data scenario."""
        return MultiTimeFrameConfirmation(
            primary_signal=SignalType.HOLD,
            confirmation_strength=ConfirmationStrength.VERY_WEAK,
            alignment_type=TimeFrameAlignment.INSUFFICIENT_DATA,
            overall_confidence=0.0,
            timeframe_signals=[],
            confirmation_score=0.0,
            conflicting_timeframes=[],
            supporting_timeframes=[],
            key_confluence_factors=[],
            risk_factors=[
                f"Insufficient timeframe data: only {len(available_timeframes)} available"
            ],
            metadata={"available_timeframes": available_timeframes},
        )

    def _create_error_result(
        self, primary_signal: Signal, error_message: str
    ) -> MultiTimeFrameConfirmation:
        """Create result for error scenario."""
        return MultiTimeFrameConfirmation(
            primary_signal=SignalType.HOLD,
            confirmation_strength=ConfirmationStrength.VERY_WEAK,
            alignment_type=TimeFrameAlignment.INSUFFICIENT_DATA,
            overall_confidence=0.0,
            timeframe_signals=[],
            confirmation_score=0.0,
            conflicting_timeframes=[],
            supporting_timeframes=[],
            key_confluence_factors=[],
            risk_factors=[f"Analysis error: {error_message}"],
            metadata={"error": error_message},
        )

    def should_allow_entry(
        self, confirmation: MultiTimeFrameConfirmation
    ) -> Tuple[bool, str]:
        """
        Determine if entry should be allowed based on confirmation results.

        Args:
            confirmation: Multi-timeframe confirmation results

        Returns:
            Tuple of (should_allow, reason)
        """
        try:
            # Check minimum confirmation strength
            min_strength_required = {
                StrategyMode.DAY_TRADING: ConfirmationStrength.MODERATE,
                StrategyMode.SWING_TRADING: ConfirmationStrength.STRONG,
                StrategyMode.POSITION_TRADING: ConfirmationStrength.STRONG,
            }

            required_strength = min_strength_required.get(
                self.strategy_mode, ConfirmationStrength.MODERATE
            )

            # Check if confirmation strength meets requirements
            strength_levels = {
                ConfirmationStrength.VERY_STRONG: 5,
                ConfirmationStrength.STRONG: 4,
                ConfirmationStrength.MODERATE: 3,
                ConfirmationStrength.WEAK: 2,
                ConfirmationStrength.VERY_WEAK: 1,
            }

            current_strength_level = strength_levels.get(
                confirmation.confirmation_strength, 0
            )
            required_strength_level = strength_levels.get(required_strength, 3)

            if current_strength_level < required_strength_level:
                return (
                    False,
                    f"Insufficient confirmation strength: {confirmation.confirmation_strength.value} (required: {required_strength.value})",
                )

            # Check for critical risk factors
            critical_risks = [
                "Too many conflicting timeframes",
                "Multiple low-confidence timeframe signals",
            ]

            for risk in confirmation.risk_factors:
                if any(critical_risk in risk for critical_risk in critical_risks):
                    return False, f"Critical risk factor: {risk}"

            # Check minimum confidence threshold
            min_confidence_thresholds = {
                StrategyMode.DAY_TRADING: 65.0,
                StrategyMode.SWING_TRADING: 70.0,
                StrategyMode.POSITION_TRADING: 60.0,
            }

            min_confidence = min_confidence_thresholds.get(self.strategy_mode, 65.0)
            if confirmation.overall_confidence < min_confidence:
                return (
                    False,
                    f"Confidence too low: {confirmation.overall_confidence:.1f}% (required: {min_confidence}%)",
                )

            # All checks passed
            return (
                True,
                f"Multi-timeframe confirmation passed: {confirmation.confirmation_strength.value} with {confirmation.overall_confidence:.1f}% confidence",
            )

        except Exception as e:
            self.logger.error(f"Error in should_allow_entry: {e}")
            return False, f"Entry validation error: {str(e)}"


class MultiTimeFrameSignalEnhancer:
    """
    Enhances primary signals with multi-timeframe confirmation.

    This class integrates with existing strategies to add multi-timeframe
    confirmation without breaking existing functionality.
    """

    def __init__(self, analyzer: MultiTimeFrameAnalyzer):
        """
        Initialize signal enhancer.

        Args:
            analyzer: Multi-timeframe analyzer instance
        """
        self.analyzer = analyzer
        self.logger = logging.getLogger(f"{__name__}.SignalEnhancer")

    async def enhance_signal_with_confirmation(
        self,
        symbol: str,
        primary_signal: Signal,
        multi_timeframe_data: Dict[str, pl.DataFrame],
        strategy_instance: Optional[BaseStrategy] = None,
    ) -> Tuple[Signal, MultiTimeFrameConfirmation]:
        """
        Enhance a primary signal with multi-timeframe confirmation.

        Args:
            symbol: Trading symbol
            primary_signal: Original signal to enhance
            multi_timeframe_data: Data for multiple timeframes
            strategy_instance: Strategy instance for analysis

        Returns:
            Tuple of (enhanced_signal, confirmation_results)
        """
        try:
            # Get multi-timeframe confirmation
            confirmation = await self.analyzer.confirm_signal(
                symbol, primary_signal, multi_timeframe_data, strategy_instance
            )

            # Check if entry should be allowed
            allow_entry, reason = self.analyzer.should_allow_entry(confirmation)

            # Create enhanced signal
            enhanced_signal = self._create_enhanced_signal(
                primary_signal, confirmation, allow_entry, reason
            )

            return enhanced_signal, confirmation

        except Exception as e:
            self.logger.error(f"Error enhancing signal for {symbol}: {e}")
            # Return original signal with error information
            error_signal = Signal(
                action=SignalType.HOLD,
                confidence=0.0,
                position_size=0.0,
                reasoning=f"Multi-timeframe analysis error: {str(e)}",
                metadata=primary_signal.metadata.copy(),
            )

            error_confirmation = self.analyzer._create_error_result(
                primary_signal, str(e)
            )
            return error_signal, error_confirmation

    def _create_enhanced_signal(
        self,
        primary_signal: Signal,
        confirmation: MultiTimeFrameConfirmation,
        allow_entry: bool,
        reason: str,
    ) -> Signal:
        """Create enhanced signal based on confirmation results."""
        try:
            # Determine final action
            if allow_entry and confirmation.primary_signal != SignalType.HOLD:
                final_action = confirmation.primary_signal
                final_confidence = confirmation.overall_confidence

                # Adjust position size based on confirmation strength
                confirmation_multiplier = self._get_confirmation_multiplier(
                    confirmation.confirmation_strength
                )
                adjusted_position_size = (
                    primary_signal.position_size * confirmation_multiplier
                )

            else:
                final_action = SignalType.HOLD  # type: ignore
                final_confidence = 0.0
                adjusted_position_size = 0.0

            # Build enhanced reasoning
            enhanced_reasoning = self._build_enhanced_reasoning(
                primary_signal, confirmation, allow_entry, reason
            )

            # Create enhanced metadata
            enhanced_metadata = primary_signal.metadata.copy()
            enhanced_metadata.update(
                {
                    "multi_timeframe_confirmation": {
                        "confirmation_strength": confirmation.confirmation_strength.value,
                        "alignment_type": confirmation.alignment_type.value,
                        "confirmation_score": confirmation.confirmation_score,
                        "supporting_timeframes": confirmation.supporting_timeframes,
                        "conflicting_timeframes": confirmation.conflicting_timeframes,
                        "confluence_factors": confirmation.key_confluence_factors,
                        "risk_factors": confirmation.risk_factors,
                        "optimal_entry_timeframe": confirmation.optimal_entry_timeframe,
                        "allow_entry": allow_entry,
                        "reason": reason,
                    },
                    "original_signal": {
                        "action": (
                            primary_signal.action.value
                            if hasattr(primary_signal.action, "value")
                            else str(primary_signal.action)
                        ),
                        "confidence": primary_signal.confidence,
                        "position_size": primary_signal.position_size,
                    },
                }
            )

            return Signal(
                action=final_action,
                confidence=final_confidence,
                entry_price=primary_signal.entry_price,
                stop_loss=primary_signal.stop_loss,
                take_profit=primary_signal.take_profit,
                position_size=adjusted_position_size,
                reasoning=enhanced_reasoning,
                metadata=enhanced_metadata,
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            self.logger.error(f"Error creating enhanced signal: {e}")
            return primary_signal  # Return original signal on error

    def _get_confirmation_multiplier(self, strength: ConfirmationStrength) -> float:
        """Get position size multiplier based on confirmation strength."""
        multipliers = {
            ConfirmationStrength.VERY_STRONG: 1.0,
            ConfirmationStrength.STRONG: 0.9,
            ConfirmationStrength.MODERATE: 0.7,
            ConfirmationStrength.WEAK: 0.5,
            ConfirmationStrength.VERY_WEAK: 0.2,
        }
        return multipliers.get(strength, 0.5)

    def _build_enhanced_reasoning(
        self,
        primary_signal: Signal,
        confirmation: MultiTimeFrameConfirmation,
        allow_entry: bool,
        reason: str,
    ) -> str:
        """Build comprehensive reasoning for enhanced signal."""
        try:
            reasoning_parts = []

            # Original signal reasoning
            if primary_signal.reasoning:
                reasoning_parts.append(f"Primary Analysis: {primary_signal.reasoning}")

            # Multi-timeframe confirmation summary
            reasoning_parts.append(
                f"Multi-Timeframe Analysis: {confirmation.confirmation_strength.value}"
            )
            reasoning_parts.append(f"Alignment: {confirmation.alignment_type.value}")

            # Supporting/conflicting timeframes
            if confirmation.supporting_timeframes:
                reasoning_parts.append(
                    f"Supporting timeframes: {', '.join(confirmation.supporting_timeframes)}"
                )

            if confirmation.conflicting_timeframes:
                reasoning_parts.append(
                    f"Conflicting timeframes: {', '.join(confirmation.conflicting_timeframes)}"
                )

            # Confluence factors
            if confirmation.key_confluence_factors:
                reasoning_parts.append(
                    f"Confluence factors: {'; '.join(confirmation.key_confluence_factors)}"
                )

            # Risk factors
            if confirmation.risk_factors:
                reasoning_parts.append(
                    f"Risk factors: {'; '.join(confirmation.risk_factors)}"
                )

            # Final decision reason
            reasoning_parts.append(f"Decision: {reason}")

            return " | ".join(reasoning_parts)

        except Exception as e:
            self.logger.error(f"Error building enhanced reasoning: {e}")
            return f"Enhanced signal analysis (error in reasoning generation: {str(e)})"


# Convenience function for easy integration
def create_multi_timeframe_analyzer(
    strategy_mode: StrategyMode,
) -> Tuple[MultiTimeFrameAnalyzer, MultiTimeFrameSignalEnhancer]:
    """
    Create multi-timeframe analyzer and enhancer for a strategy mode.

    Args:
        strategy_mode: Strategy mode to configure analyzer for

    Returns:
        Tuple of (analyzer, enhancer)
    """
    analyzer = MultiTimeFrameAnalyzer(strategy_mode)
    enhancer = MultiTimeFrameSignalEnhancer(analyzer)
    return analyzer, enhancer
