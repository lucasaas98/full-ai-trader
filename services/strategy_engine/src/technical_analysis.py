"""
Technical Analysis Module

This module implements high-performance technical indicators using Polars
for maximum speed and efficiency. Includes moving averages, oscillators,
volume indicators, and pattern recognition.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from shared.models import SignalType

from .base_strategy import BaseStrategy, Signal, StrategyConfig, StrategyMode


class TrendDirection(Enum):
    """Trend direction enumeration."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


class PatternType(Enum):
    """Chart pattern types."""

    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE = "triangle"
    WEDGE = "wedge"
    FLAG = "flag"
    PENNANT = "pennant"


@dataclass
class PatternSignal:
    """Chart pattern signal."""

    pattern: PatternType
    confidence: float
    target_price: Optional[float] = None
    time_horizon: Optional[int] = None  # Days


class TechnicalIndicators:
    """High-performance technical indicators using Polars."""

    @staticmethod
    def sma(data: pl.DataFrame, period: int, column: str = "close") -> pl.DataFrame:
        """Simple Moving Average."""
        return data.with_columns(
            [pl.col(column).rolling_mean(window_size=period).alias(f"sma_{period}")]
        )

    @staticmethod
    def ema(data: pl.DataFrame, period: int, column: str = "close") -> pl.DataFrame:
        """Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        return data.with_columns(
            [pl.col(column).ewm_mean(alpha=alpha, adjust=False).alias(f"ema_{period}")]
        )

    @staticmethod
    def wma(data: pl.DataFrame, period: int, column: str = "close") -> pl.DataFrame:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        weights = weights / weights.sum()

        def weighted_mean(values):
            if len(values) < period:
                return None
            return np.dot(values[-period:], weights)

        return data.with_columns(
            [
                pl.col(column)
                .map_batches(lambda s: s.rolling_map(weighted_mean, window_size=period))
                .alias(f"wma_{period}")
            ]
        )

    @staticmethod
    def rsi(
        data: pl.DataFrame, period: int = 14, column: str = "close"
    ) -> pl.DataFrame:
        """Relative Strength Index."""
        return (
            data.with_columns(
                [
                    # Calculate price changes
                    pl.col(column)
                    .diff()
                    .alias("price_change")
                ]
            )
            .with_columns(
                [
                    # Separate gains and losses
                    pl.when(pl.col("price_change") > 0)
                    .then(pl.col("price_change"))
                    .otherwise(0.0)
                    .alias("gains"),
                    pl.when(pl.col("price_change") < 0)
                    .then(-pl.col("price_change"))
                    .otherwise(0.0)
                    .alias("losses"),
                ]
            )
            .with_columns(
                [
                    # Calculate average gains and losses
                    pl.col("gains")
                    .ewm_mean(alpha=1.0 / period, adjust=False)
                    .alias("avg_gains"),
                    pl.col("losses")
                    .ewm_mean(alpha=1.0 / period, adjust=False)
                    .alias("avg_losses"),
                ]
            )
            .with_columns(
                [
                    # Calculate RSI
                    (
                        100.0
                        - (100.0 / (1.0 + (pl.col("avg_gains") / pl.col("avg_losses"))))
                    ).alias(f"rsi_{period}")
                ]
            )
        )

    @staticmethod
    def macd(
        data: pl.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> pl.DataFrame:
        """MACD (Moving Average Convergence Divergence)."""
        return (
            data.with_columns(
                [
                    pl.col(column)
                    .ewm_mean(alpha=2.0 / (fast + 1), adjust=False)
                    .alias("ema_fast"),
                    pl.col(column)
                    .ewm_mean(alpha=2.0 / (slow + 1), adjust=False)
                    .alias("ema_slow"),
                ]
            )
            .with_columns(
                [(pl.col("ema_fast") - pl.col("ema_slow")).alias("macd_line")]
            )
            .with_columns(
                [
                    pl.col("macd_line")
                    .ewm_mean(alpha=2.0 / (signal + 1), adjust=False)
                    .alias("macd_signal")
                ]
            )
            .with_columns(
                [(pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram")]
            )
        )

    @staticmethod
    def bollinger_bands(
        data: pl.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> pl.DataFrame:
        """Bollinger Bands."""
        return (
            data.with_columns(
                [
                    pl.col(column).rolling_mean(window_size=period).alias("bb_middle"),
                    pl.col(column).rolling_std(window_size=period).alias("bb_std"),
                ]
            )
            .with_columns(
                [
                    (pl.col("bb_middle") + (pl.col("bb_std") * std_dev)).alias(
                        "bb_upper"
                    ),
                    (pl.col("bb_middle") - (pl.col("bb_std") * std_dev)).alias(
                        "bb_lower"
                    ),
                ]
            )
            .with_columns(
                [
                    # Bollinger Band Position (0-1, where 0.5 is middle)
                    (
                        (pl.col(column) - pl.col("bb_lower"))
                        / (pl.col("bb_upper") - pl.col("bb_lower"))
                    ).alias("bb_position"),
                    # Bollinger Band Width (volatility measure)
                    (
                        (pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")
                    ).alias("bb_width"),
                ]
            )
        )

    @staticmethod
    def atr(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Average True Range."""
        return (
            data.with_columns(
                [
                    # True Range components
                    (pl.col("high") - pl.col("low")).alias("hl"),
                    (pl.col("high") - pl.col("close").shift(1)).abs().alias("hc"),
                    (pl.col("low") - pl.col("close").shift(1)).abs().alias("lc"),
                ]
            )
            .with_columns(
                [
                    # True Range (maximum of the three components)
                    pl.max_horizontal(["hl", "hc", "lc"]).alias("true_range")
                ]
            )
            .with_columns(
                [
                    # Average True Range
                    pl.col("true_range")
                    .ewm_mean(alpha=1.0 / period, adjust=False)
                    .alias(f"atr_{period}")
                ]
            )
        )

    @staticmethod
    def obv(data: pl.DataFrame) -> pl.DataFrame:
        """On Balance Volume."""
        return (
            data.with_columns([pl.col("close").diff().alias("price_change")])
            .with_columns(
                [
                    pl.when(pl.col("price_change") > 0)
                    .then(pl.col("volume"))
                    .when(pl.col("price_change") < 0)
                    .then(-pl.col("volume"))
                    .otherwise(0)
                    .alias("obv_change")
                ]
            )
            .with_columns([pl.col("obv_change").cum_sum().alias("obv")])
        )

    @staticmethod
    def vwap(data: pl.DataFrame) -> pl.DataFrame:
        """Volume Weighted Average Price."""
        return (
            data.with_columns(
                [
                    # Typical price
                    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias(
                        "typical_price"
                    )
                ]
            )
            .with_columns(
                [
                    # Price * Volume
                    (pl.col("typical_price") * pl.col("volume")).alias("pv")
                ]
            )
            .with_columns(
                [
                    # VWAP calculation
                    (pl.col("pv").cum_sum() / pl.col("volume").cum_sum()).alias("vwap")
                ]
            )
        )

    @staticmethod
    def stochastic(
        data: pl.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pl.DataFrame:
        """Stochastic Oscillator."""
        return (
            data.with_columns(
                [
                    pl.col("low").rolling_min(window_size=k_period).alias("lowest_low"),
                    pl.col("high")
                    .rolling_max(window_size=k_period)
                    .alias("highest_high"),
                ]
            )
            .with_columns(
                [
                    # %K calculation
                    (
                        (
                            (pl.col("close") - pl.col("lowest_low"))
                            / (pl.col("highest_high") - pl.col("lowest_low"))
                        )
                        * 100.0
                    ).alias("stoch_k")
                ]
            )
            .with_columns(
                [
                    # %D calculation (SMA of %K)
                    pl.col("stoch_k")
                    .rolling_mean(window_size=d_period)
                    .alias("stoch_d")
                ]
            )
        )

    @staticmethod
    def williams_r(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Williams %R."""
        return data.with_columns(
            [
                pl.col("high").rolling_max(window_size=period).alias("highest_high"),
                pl.col("low").rolling_min(window_size=period).alias("lowest_low"),
            ]
        ).with_columns(
            [
                # Williams %R calculation
                (
                    (
                        (pl.col("highest_high") - pl.col("close"))
                        / (pl.col("highest_high") - pl.col("lowest_low"))
                    )
                    * -100.0
                ).alias(f"williams_r_{period}")
            ]
        )

    @staticmethod
    def cci(data: pl.DataFrame, period: int = 20) -> pl.DataFrame:
        """Commodity Channel Index."""
        return (
            data.with_columns(
                [
                    # Typical price
                    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias(
                        "typical_price"
                    )
                ]
            )
            .with_columns(
                [
                    # SMA of typical price
                    pl.col("typical_price")
                    .rolling_mean(window_size=period)
                    .alias("sma_tp")
                ]
            )
            .with_columns(
                [
                    # Mean deviation
                    (pl.col("typical_price") - pl.col("sma_tp"))
                    .abs()
                    .rolling_mean(window_size=period)
                    .alias("mean_deviation")
                ]
            )
            .with_columns(
                [
                    # CCI calculation
                    (
                        (pl.col("typical_price") - pl.col("sma_tp"))
                        / (0.015 * pl.col("mean_deviation"))
                    ).alias(f"cci_{period}")
                ]
            )
        )

    @staticmethod
    def momentum(
        data: pl.DataFrame, period: int = 10, column: str = "close"
    ) -> pl.DataFrame:
        """Price Momentum."""
        return data.with_columns(
            [
                (pl.col(column) - pl.col(column).shift(period)).alias(
                    f"momentum_{period}"
                )
            ]
        )

    @staticmethod
    def rate_of_change(
        data: pl.DataFrame, period: int = 10, column: str = "close"
    ) -> pl.DataFrame:
        """Rate of Change."""
        return data.with_columns(
            [
                (
                    (pl.col(column) - pl.col(column).shift(period))
                    / pl.col(column).shift(period)
                    * 100.0
                ).alias(f"roc_{period}")
            ]
        )


class SupportResistanceLevels:
    """Support and resistance level detection."""

    @staticmethod
    def find_pivot_points(data: pl.DataFrame, window: int = 5) -> pl.DataFrame:
        """Find pivot highs and lows."""
        return data.with_columns(
            [
                # Pivot highs
                pl.when(
                    (
                        pl.col("high")
                        == pl.col("high").rolling_max(window_size=window * 2 + 1)
                    )
                    & (pl.col("high") > pl.col("high").shift(1))
                    & (pl.col("high") > pl.col("high").shift(-1))
                )
                .then(pl.col("high"))
                .otherwise(None)
                .alias("pivot_high"),
                # Pivot lows
                pl.when(
                    (
                        pl.col("low")
                        == pl.col("low").rolling_min(window_size=window * 2 + 1)
                    )
                    & (pl.col("low") < pl.col("low").shift(1))
                    & (pl.col("low") < pl.col("low").shift(-1))
                )
                .then(pl.col("low"))
                .otherwise(None)
                .alias("pivot_low"),
            ]
        )

    @staticmethod
    def calculate_support_resistance(
        data: pl.DataFrame, lookback: int = 100
    ) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels."""
        # Get pivot points
        pivots = SupportResistanceLevels.find_pivot_points(data.tail(lookback))

        # Extract non-null pivot highs and lows
        resistance_levels = (
            pivots.select("pivot_high")
            .filter(pl.col("pivot_high").is_not_null())
            .to_series()
            .to_list()
        )

        support_levels = (
            pivots.select("pivot_low")
            .filter(pl.col("pivot_low").is_not_null())
            .to_series()
            .to_list()
        )

        # Cluster nearby levels (within 1% of each other)
        resistance_levels = SupportResistanceLevels._cluster_levels(resistance_levels)
        support_levels = SupportResistanceLevels._cluster_levels(support_levels)

        return support_levels, resistance_levels

    @staticmethod
    def _cluster_levels(levels: List[float], tolerance: float = 0.01) -> List[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []

        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                # Take average of cluster
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        # Add the last cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))

        return clustered


class PatternRecognition:
    """Chart pattern recognition using Polars."""

    @staticmethod
    def detect_breakout(
        data: pl.DataFrame, window: int = 20
    ) -> Optional[PatternSignal]:
        """Detect price breakouts above resistance."""
        # Get recent high and current price
        recent_data = data.tail(window)
        current_price = float(recent_data.select("close").tail(1).item())
        recent_high = float(recent_data.select("high").max().item())

        # Volume confirmation
        recent_volume = recent_data.select("volume").tail(5).mean().item()
        avg_volume = recent_data.select("volume").mean().item()

        # Check for breakout conditions
        if (
            current_price > recent_high * 1.001  # 0.1% above recent high
            and recent_volume > avg_volume * 1.2
        ):  # 20% above average volume

            confidence = min(90.0, 60.0 + (recent_volume / avg_volume - 1) * 30)
            target = current_price * 1.05  # 5% target

            return PatternSignal(
                pattern=PatternType.BREAKOUT,
                confidence=confidence,
                target_price=target,
                time_horizon=5,
            )

        return None

    @staticmethod
    def detect_breakdown(
        data: pl.DataFrame, window: int = 20
    ) -> Optional[PatternSignal]:
        """Detect price breakdowns below support."""
        recent_data = data.tail(window)
        current_price = float(recent_data.select("close").tail(1).item())
        recent_low = float(recent_data.select("low").min().item())

        # Volume confirmation
        recent_volume = recent_data.select("volume").tail(5).mean().item()
        avg_volume = recent_data.select("volume").mean().item()

        if (
            current_price < recent_low * 0.999  # 0.1% below recent low
            and recent_volume > avg_volume * 1.2
        ):

            confidence = min(90.0, 60.0 + (recent_volume / avg_volume - 1) * 30)
            target = current_price * 0.95  # 5% target down

            return PatternSignal(
                pattern=PatternType.BREAKDOWN,
                confidence=confidence,
                target_price=target,
                time_horizon=5,
            )

        return None

    @staticmethod
    def detect_double_top(
        data: pl.DataFrame, window: int = 50
    ) -> Optional[PatternSignal]:
        """Detect double top pattern."""
        # Find pivot points
        pivots = SupportResistanceLevels.find_pivot_points(data.tail(window))
        highs = (
            pivots.select(["timestamp", "pivot_high"])
            .filter(pl.col("pivot_high").is_not_null())
            .sort("timestamp")
        )

        if highs.height < 2:
            return None

        # Check last two highs
        last_two = highs.tail(2)
        high1 = float(last_two.select("pivot_high").slice(0, 1).item())
        high2 = float(last_two.select("pivot_high").slice(1, 1).item())

        # Double top criteria: highs within 2% of each other
        if abs(high1 - high2) / max(high1, high2) <= 0.02:
            current_price = float(data.select("close").tail(1).item())

            # Confirmation: price should be below both highs
            if current_price < min(high1, high2) * 0.98:
                confidence = 75.0
                target = current_price * 0.90  # 10% target down

                return PatternSignal(
                    pattern=PatternType.DOUBLE_TOP,
                    confidence=confidence,
                    target_price=target,
                    time_horizon=10,
                )

        return None

    @staticmethod
    def detect_double_bottom(
        data: pl.DataFrame, window: int = 50
    ) -> Optional[PatternSignal]:
        """Detect double bottom pattern."""
        pivots = SupportResistanceLevels.find_pivot_points(data.tail(window))
        lows = (
            pivots.select(["timestamp", "pivot_low"])
            .filter(pl.col("pivot_low").is_not_null())
            .sort("timestamp")
        )

        if lows.height < 2:
            return None

        # Check last two lows
        last_two = lows.tail(2)
        low1 = float(last_two.select("pivot_low").slice(0, 1).item())
        low2 = float(last_two.select("pivot_low").slice(1, 1).item())

        # Double bottom criteria
        if abs(low1 - low2) / max(low1, low2) <= 0.02:
            current_price = float(data.select("close").tail(1).item())

            # Confirmation: price should be above both lows
            if current_price > max(low1, low2) * 1.02:
                confidence = 75.0
                target = current_price * 1.10  # 10% target up

                return PatternSignal(
                    pattern=PatternType.DOUBLE_BOTTOM,
                    confidence=confidence,
                    target_price=target,
                    time_horizon=10,
                )

        return None


class TechnicalStrategy(BaseStrategy):
    """
    Technical analysis strategy that combines multiple indicators.

    This strategy uses various technical indicators to generate trading signals
    with confidence scores based on indicator convergence and pattern recognition.
    """

    def __init__(self, config: StrategyConfig):
        """Initialize technical strategy."""
        super().__init__(config)

        # Default technical parameters
        self.default_params = {
            "sma_short": 20,
            "sma_long": 50,
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "volume_threshold": 1.5,  # Volume surge threshold
            "pattern_weight": 0.3,  # Weight for pattern signals
            "momentum_weight": 0.4,  # Weight for momentum signals
            "mean_reversion_weight": 0.3,  # Weight for mean reversion signals
        }

        # Merge with user parameters
        self.params = {**self.default_params, **(self.config.parameters or {})}

    def _setup_indicators(self) -> None:
        """Setup technical indicators."""
        self.logger.info(f"Setting up technical indicators for {self.name}")

        # Indicator setup is dynamic - indicators are calculated on-demand
        # This allows for efficient memory usage with large datasets

        self.indicators = TechnicalIndicators()
        self.support_resistance = SupportResistanceLevels()
        self.pattern_recognition = PatternRecognition()

    async def analyze(self, symbol: str, data: pl.DataFrame) -> Signal:
        """
        Perform comprehensive technical analysis.

        Args:
            symbol: Trading symbol
            data: Historical market data

        Returns:
            Trading signal with confidence score
        """
        if not self.validate_data(data):
            return Signal(
                action=SignalType.HOLD,
                confidence=0.0,
                position_size=0.0,
                reasoning="Invalid or insufficient data for technical analysis",
            )

        # Calculate all indicators
        analyzed_data = self._calculate_all_indicators(data)

        # Get latest values
        latest = analyzed_data.tail(1)

        # Generate signals from different components
        momentum_signal = self._analyze_momentum(analyzed_data, latest)
        mean_reversion_signal = self._analyze_mean_reversion(analyzed_data, latest)
        pattern_signal = self._analyze_patterns(analyzed_data)
        volume_signal = self._analyze_volume(analyzed_data, latest)

        # Combine signals
        final_signal = self._combine_signals(
            symbol,
            latest,
            momentum_signal,
            mean_reversion_signal,
            pattern_signal,
            volume_signal,
        )

        return final_signal

    def _calculate_all_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate all technical indicators efficiently."""
        # Start with base data
        result = data.clone()

        # Moving averages
        result = self.indicators.sma(result, self.params["sma_short"])
        result = self.indicators.sma(result, self.params["sma_long"])
        result = self.indicators.ema(result, self.params["ema_fast"])
        result = self.indicators.ema(result, self.params["ema_slow"])

        # Oscillators
        result = self.indicators.rsi(result, self.params["rsi_period"])
        result = self.indicators.macd(
            result,
            self.params["macd_fast"],
            self.params["macd_slow"],
            self.params["macd_signal"],
        )
        result = self.indicators.stochastic(result)
        result = self.indicators.williams_r(result)
        result = self.indicators.cci(result)

        # Volatility and volume
        result = self.indicators.bollinger_bands(
            result, self.params["bb_period"], self.params["bb_std"]
        )
        result = self.indicators.atr(result, self.params["atr_period"])
        result = self.indicators.obv(result)
        result = self.indicators.vwap(result)

        # Momentum indicators
        result = self.indicators.momentum(result)
        result = self.indicators.rate_of_change(result)

        return result

    def _analyze_momentum(
        self, data: pl.DataFrame, latest: pl.DataFrame
    ) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        try:
            # Get latest values
            sma_short = latest.select(f"sma_{self.params['sma_short']}").item()
            sma_long = latest.select(f"sma_{self.params['sma_long']}").item()
            ema_fast = latest.select(f"ema_{self.params['ema_fast']}").item()
            ema_slow = latest.select(f"ema_{self.params['ema_slow']}").item()
            macd_line = latest.select("macd_line").item()
            macd_signal = latest.select("macd_signal").item()
            macd_histogram = latest.select("macd_histogram").item()
            roc = latest.select("roc_10").item()

            signals = []
            confidence = 0.0

            # Moving average signals
            if sma_short and sma_long:
                if sma_short > sma_long:
                    signals.append("MA_BULLISH")
                    confidence += 15
                elif sma_short < sma_long:
                    signals.append("MA_BEARISH")
                    confidence -= 15

            # EMA crossover
            if ema_fast and ema_slow:
                if ema_fast > ema_slow:
                    signals.append("EMA_BULLISH")
                    confidence += 15
                elif ema_fast < ema_slow:
                    signals.append("EMA_BEARISH")
                    confidence -= 15

            # MACD signals
            if macd_line and macd_signal:
                if macd_line > macd_signal and macd_histogram > 0:
                    signals.append("MACD_BULLISH")
                    confidence += 20
                elif macd_line < macd_signal and macd_histogram < 0:
                    signals.append("MACD_BEARISH")
                    confidence -= 20

            # Rate of change momentum
            if roc:
                if roc > 5:  # Strong positive momentum
                    signals.append("ROC_STRONG_BULLISH")
                    confidence += 10
                elif roc > 0:
                    signals.append("ROC_BULLISH")
                    confidence += 5
                elif roc < -5:  # Strong negative momentum
                    signals.append("ROC_STRONG_BEARISH")
                    confidence -= 10
                elif roc < 0:
                    signals.append("ROC_BEARISH")
                    confidence -= 5

            # Determine action
            action = SignalType.HOLD
            if confidence >= 30:
                action = SignalType.BUY
            elif confidence <= -30:
                action = SignalType.SELL

            return {
                "action": action,
                "confidence": abs(confidence),
                "signals": signals,
                "component": "momentum",
            }

        except Exception as e:
            self.logger.error(f"Error in momentum analysis: {e}")
            return {
                "action": SignalType.HOLD,
                "confidence": 0.0,
                "signals": [],
                "component": "momentum",
            }

    def _analyze_mean_reversion(
        self, data: pl.DataFrame, latest: pl.DataFrame
    ) -> Dict[str, Any]:
        """Analyze mean reversion indicators."""
        try:
            # Get latest values
            rsi = latest.select(f"rsi_{self.params['rsi_period']}").item()
            bb_position = latest.select("bb_position").item()
            stoch_k = latest.select("stoch_k").item()
            stoch_d = latest.select("stoch_d").item()
            williams_r = latest.select("williams_r_14").item()
            cci = latest.select("cci_20").item()

            signals = []
            confidence = 0.0

            # RSI signals
            if rsi:
                if rsi <= self.params["rsi_oversold"]:
                    signals.append("RSI_OVERSOLD")
                    confidence += 25
                elif rsi >= self.params["rsi_overbought"]:
                    signals.append("RSI_OVERBOUGHT")
                    confidence -= 25
                elif 40 <= rsi <= 60:
                    signals.append("RSI_NEUTRAL")

            # Bollinger Bands
            if bb_position is not None:
                if bb_position <= 0.1:  # Near lower band
                    signals.append("BB_OVERSOLD")
                    confidence += 20
                elif bb_position >= 0.9:  # Near upper band
                    signals.append("BB_OVERBOUGHT")
                    confidence -= 20

            # Stochastic
            if stoch_k and stoch_d:
                if stoch_k <= 20 and stoch_d <= 20:
                    signals.append("STOCH_OVERSOLD")
                    confidence += 15
                elif stoch_k >= 80 and stoch_d >= 80:
                    signals.append("STOCH_OVERBOUGHT")
                    confidence -= 15

            # Williams %R
            if williams_r:
                if williams_r <= -80:
                    signals.append("WR_OVERSOLD")
                    confidence += 10
                elif williams_r >= -20:
                    signals.append("WR_OVERBOUGHT")
                    confidence -= 10

            # CCI
            if cci:
                if cci <= -100:
                    signals.append("CCI_OVERSOLD")
                    confidence += 15
                elif cci >= 100:
                    signals.append("CCI_OVERBOUGHT")
                    confidence -= 15

            # Determine action
            action = SignalType.HOLD
            if confidence >= 40:  # Multiple oversold signals
                action = SignalType.BUY
            elif confidence <= -40:  # Multiple overbought signals
                action = SignalType.SELL

            return {
                "action": action,
                "confidence": abs(confidence),
                "signals": signals,
                "component": "mean_reversion",
            }

        except Exception as e:
            self.logger.error(f"Error in mean reversion analysis: {e}")
            return {
                "action": SignalType.HOLD,
                "confidence": 0.0,
                "signals": [],
                "component": "mean_reversion",
            }

    def _analyze_patterns(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Analyze chart patterns."""
        try:
            signals = []
            confidence = 0.0
            detected_patterns = []

            # Detect various patterns
            breakout = self.pattern_recognition.detect_breakout(data)
            if breakout:
                detected_patterns.append(breakout)
                if breakout.pattern == PatternType.BREAKOUT:
                    signals.append("BREAKOUT")
                    confidence += breakout.confidence * 0.3
                elif breakout.pattern == PatternType.BREAKDOWN:
                    signals.append("BREAKDOWN")
                    confidence -= breakout.confidence * 0.3

            breakdown = self.pattern_recognition.detect_breakdown(data)
            if breakdown:
                detected_patterns.append(breakdown)
                signals.append("BREAKDOWN")
                confidence -= breakdown.confidence * 0.3

            double_top = self.pattern_recognition.detect_double_top(data)
            if double_top:
                detected_patterns.append(double_top)
                signals.append("DOUBLE_TOP")
                confidence -= double_top.confidence * 0.4

            double_bottom = self.pattern_recognition.detect_double_bottom(data)
            if double_bottom:
                detected_patterns.append(double_bottom)
                signals.append("DOUBLE_BOTTOM")
                confidence += double_bottom.confidence * 0.4

            # Determine action based on patterns
            action = SignalType.HOLD
            if confidence >= 20:
                action = SignalType.BUY
            elif confidence <= -20:
                action = SignalType.SELL

            return {
                "action": action,
                "confidence": abs(confidence),
                "signals": signals,
                "patterns": detected_patterns,
                "component": "patterns",
            }

        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
            return {
                "action": SignalType.HOLD,
                "confidence": 0.0,
                "signals": [],
                "patterns": [],
                "component": "patterns",
            }

    def _analyze_volume(
        self, data: pl.DataFrame, latest: pl.DataFrame
    ) -> Dict[str, Any]:
        """Analyze volume indicators."""
        try:
            # Get recent volume data
            recent_volume = data.select("volume").tail(5).mean().item()
            avg_volume = data.select("volume").tail(20).mean().item()
            obv_current = latest.select("obv").item()
            vwap_current = latest.select("vwap").item()
            current_price = latest.select("close").item()

            # Get previous OBV for trend
            if data.height >= 2:
                obv_prev = data.select("obv").slice(-2, 1).item()
                obv_trend = "rising" if obv_current > obv_prev else "falling"
            else:
                obv_trend = "neutral"

            signals = []
            confidence = 0.0

            # Volume surge detection
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            if volume_ratio >= self.params["volume_threshold"]:
                signals.append(f"VOLUME_SURGE_{volume_ratio:.1f}x")
                confidence += min(30, (volume_ratio - 1) * 20)

            # OBV analysis
            if obv_trend == "rising":
                signals.append("OBV_RISING")
                confidence += 15
            elif obv_trend == "falling":
                signals.append("OBV_FALLING")
                confidence -= 15

            # VWAP analysis
            if current_price and vwap_current:
                vwap_diff = (current_price - vwap_current) / vwap_current
                if vwap_diff > 0.002:  # Above VWAP
                    signals.append("ABOVE_VWAP")
                    confidence += 10
                elif vwap_diff < -0.002:  # Below VWAP
                    signals.append("BELOW_VWAP")
                    confidence -= 10

            # Determine action
            action = SignalType.HOLD
            if confidence >= 25:
                action = SignalType.BUY
            elif confidence <= -25:
                action = SignalType.SELL

            return {
                "action": action,
                "confidence": abs(confidence),
                "signals": signals,
                "volume_ratio": volume_ratio,
                "obv_trend": obv_trend,
                "component": "volume",
            }

        except Exception as e:
            self.logger.error(f"Error in volume analysis: {e}")
            return {
                "action": SignalType.HOLD,
                "confidence": 0.0,
                "signals": [],
                "component": "volume",
            }

    def _combine_signals(
        self,
        symbol: str,
        latest: pl.DataFrame,
        momentum_signal: Dict,
        mean_reversion_signal: Dict,
        pattern_signal: Dict,
        volume_signal: Dict,
    ) -> Signal:
        """Combine all technical signals into final signal."""
        try:
            # Extract components
            momentum_action = momentum_signal["action"]
            momentum_conf = momentum_signal["confidence"]

            mean_reversion_action = mean_reversion_signal["action"]
            mean_reversion_conf = mean_reversion_signal["confidence"]

            pattern_action = pattern_signal["action"]
            pattern_conf = pattern_signal["confidence"]

            volume_action = volume_signal["action"]
            volume_conf = volume_signal["confidence"]

            # Weight the signals based on strategy mode
            if self.config.mode == StrategyMode.DAY_TRADING:
                # Day trading favors momentum and volume
                momentum_weight = 0.4
                volume_weight = 0.3
                pattern_weight = 0.2
                mean_reversion_weight = 0.1
            elif self.config.mode == StrategyMode.SWING_TRADING:
                # Swing trading balances all factors
                momentum_weight = 0.3
                volume_weight = 0.2
                pattern_weight = 0.3
                mean_reversion_weight = 0.2
            else:  # Position trading
                # Position trading favors patterns and momentum
                momentum_weight = 0.4
                volume_weight = 0.1
                pattern_weight = 0.4
                mean_reversion_weight = 0.1

            # Calculate weighted confidence
            total_confidence = 0.0
            buy_signals = []
            sell_signals = []
            all_signals = []

            # Process each component
            components = [
                (momentum_action, momentum_conf, momentum_weight, momentum_signal),
                (
                    mean_reversion_action,
                    mean_reversion_conf,
                    mean_reversion_weight,
                    mean_reversion_signal,
                ),
                (pattern_action, pattern_conf, pattern_weight, pattern_signal),
                (volume_action, volume_conf, volume_weight, volume_signal),
            ]

            for action, conf, weight, signal_data in components:
                weighted_conf = conf * weight
                all_signals.extend(signal_data.get("signals", []))

                if action == SignalType.BUY:
                    total_confidence += weighted_conf
                    buy_signals.extend(signal_data.get("signals", []))
                elif action == SignalType.SELL:
                    total_confidence -= weighted_conf
                    sell_signals.extend(signal_data.get("signals", []))

            # Determine final action
            final_action = SignalType.HOLD
            final_confidence = abs(total_confidence)

            if total_confidence >= 30:
                final_action = SignalType.BUY
            elif total_confidence <= -30:
                final_action = SignalType.SELL

            # Build reasoning
            reasoning_parts = []
            if buy_signals:
                reasoning_parts.append(f"Bullish: {', '.join(buy_signals)}")
            if sell_signals:
                reasoning_parts.append(f"Bearish: {', '.join(sell_signals)}")

            reasoning = (
                " | ".join(reasoning_parts) if reasoning_parts else "Neutral signals"
            )

            # Get current price for entry price
            current_price = float(latest.select("close").item())

            return Signal(
                action=final_action,
                confidence=min(100.0, final_confidence),
                entry_price=Decimal(str(current_price)),
                position_size=self._calculate_position_size(final_confidence),
                reasoning=reasoning,
                metadata={
                    "strategy_type": "technical",
                    "momentum": momentum_signal,
                    "mean_reversion": mean_reversion_signal,
                    "patterns": pattern_signal,
                    "volume": volume_signal,
                    "all_signals": all_signals,
                    "weighted_confidence": total_confidence,
                },
            )

        except Exception as e:
            self.logger.error(f"Error combining technical signals: {e}")
            return Signal(
                action=SignalType.HOLD,
                confidence=50.0,
                position_size=0.0,
                reasoning=f"Technical analysis error: {str(e)}",
            )


class MarketRegimeDetector:
    """Detect market regime (trending, ranging, volatile)."""

    @staticmethod
    def detect_regime(data: pl.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """
        Detect current market regime.

        Args:
            data: Market data
            lookback: Period to analyze

        Returns:
            Dictionary with regime information
        """
        recent_data = data.tail(lookback)

        # Calculate trend strength
        trend_strength = MarketRegimeDetector._calculate_trend_strength(recent_data)

        # Calculate volatility
        volatility = MarketRegimeDetector._calculate_volatility(recent_data)

        # Calculate range-bound characteristics
        range_bound_score = MarketRegimeDetector._calculate_range_bound_score(
            recent_data
        )

        # Determine regime
        if trend_strength > 0.6:
            regime = (
                TrendDirection.BULLISH if trend_strength > 0 else TrendDirection.BEARISH
            )
        elif range_bound_score > 0.7:
            regime = TrendDirection.SIDEWAYS
        else:
            regime = TrendDirection.SIDEWAYS  # Default to sideways if unclear

        # Volatility classification
        volatility_regime = (
            "high" if volatility > 0.02 else "normal" if volatility > 0.01 else "low"
        )

        return {
            "regime": regime,
            "trend_strength": abs(trend_strength),
            "volatility": volatility,
            "volatility_regime": volatility_regime,
            "range_bound_score": range_bound_score,
            "confidence": max(abs(trend_strength), range_bound_score),
        }

    @staticmethod
    def _calculate_trend_strength(data: pl.DataFrame) -> float:
        """Calculate trend strength (-1 to 1)."""
        # Linear regression slope of closing prices
        closes = data.select("close").to_series().to_numpy()
        x = np.arange(len(closes))

        if len(closes) < 2:
            return 0.0

        # Calculate slope using least squares
        slope = np.polyfit(x, closes, 1)[0]

        # Normalize by average price to get percentage slope
        avg_price = np.mean(closes)
        normalized_slope = slope / avg_price * len(closes)

        # Scale to -1 to 1 range
        return max(-1.0, min(1.0, normalized_slope * 2))

    @staticmethod
    def _calculate_volatility(data: pl.DataFrame) -> float:
        """Calculate realized volatility."""
        returns = (
            data.with_columns(pl.col("close").pct_change().alias("returns"))
            .select("returns")
            .drop_nulls()
            .to_series()
            .to_numpy()
        )

        if len(returns) < 2:
            return 0.0

        # Annualized volatility
        return np.std(returns) * np.sqrt(252)

    @staticmethod
    def _calculate_range_bound_score(data: pl.DataFrame) -> float:
        """Calculate how range-bound the market is (0-1)."""
        # Look at price oscillations around moving average
        data_with_ma = TechnicalIndicators.sma(data, 20)

        if data_with_ma.height < 20:
            return 0.0

        # Calculate how often price crosses the moving average
        recent_data = data_with_ma.tail(20)

        crossings = (
            recent_data.with_columns(
                [(pl.col("close") > pl.col("sma_20")).cast(pl.Int32).alias("above_ma")]
            )
            .with_columns([pl.col("above_ma").diff().abs().alias("crossing")])
            .select("crossing")
            .sum()
            .item()
            or 0
        )

        # More crossings indicate range-bound behavior
        max_crossings = 10  # Theoretical maximum for 20 periods
        range_score = min(1.0, crossings / max_crossings)

        return range_score


class VolatilityAnalyzer:
    """Advanced volatility analysis."""

    @staticmethod
    def calculate_volatility_metrics(data: pl.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics."""
        # Historical volatility (multiple periods)
        vol_5d = VolatilityAnalyzer._calculate_realized_vol(data, 5)
        vol_20d = VolatilityAnalyzer._calculate_realized_vol(data, 20)
        vol_60d = VolatilityAnalyzer._calculate_realized_vol(data, 60)

        # Volatility of volatility
        vol_of_vol = VolatilityAnalyzer._calculate_vol_of_vol(data)

        # ATR-based volatility
        atr_data = TechnicalIndicators.atr(data, 14)
        atr_vol = VolatilityAnalyzer._atr_to_volatility(atr_data)

        return {
            "realized_vol_5d": vol_5d,
            "realized_vol_20d": vol_20d,
            "realized_vol_60d": vol_60d,
            "vol_of_vol": vol_of_vol,
            "atr_vol": atr_vol,
            "vol_regime": (
                "high" if vol_20d > 0.25 else "normal" if vol_20d > 0.15 else "low"
            ),
        }

    @staticmethod
    def _calculate_realized_vol(data: pl.DataFrame, period: int) -> float:
        """Calculate realized volatility for given period."""
        if data.height < period + 1:
            return 0.0

        returns = (
            data.tail(period + 1)
            .with_columns(pl.col("close").pct_change().alias("returns"))
            .select("returns")
            .drop_nulls()
            .to_series()
            .to_numpy()
        )

        if len(returns) < 2:
            return 0.0

        return float(np.std(returns) * np.sqrt(252))

    @staticmethod
    def _calculate_vol_of_vol(data: pl.DataFrame, window: int = 20) -> float:
        """Calculate volatility of volatility."""
        if data.height < window * 2:
            return 0.0

        # Calculate rolling volatility
        vol_data = (
            data.with_columns(pl.col("close").pct_change().alias("returns"))
            .with_columns(
                pl.col("returns").rolling_std(window_size=window).alias("rolling_vol")
            )
            .select("rolling_vol")
            .drop_nulls()
        )

        if vol_data.height < 5:
            return 0.0

        vol_series = vol_data.to_series().to_numpy()
        return float(np.std(vol_series))

    @staticmethod
    def _atr_to_volatility(data: pl.DataFrame) -> float:
        """Convert ATR to annualized volatility."""
        if "atr_14" not in data.columns or data.height == 0:
            return 0.0

        latest_atr = data.select("atr_14").tail(1).item()
        latest_price = data.select("close").tail(1).item()

        if not latest_atr or not latest_price or latest_price == 0:
            return 0.0

        # ATR as percentage of price, annualized
        return float((latest_atr / latest_price) * np.sqrt(252))


class TechnicalAnalysisEngine:
    """Main technical analysis engine that orchestrates all components."""

    def __init__(self):
        """Initialize the technical analysis engine."""
        self.logger = logging.getLogger("technical_analysis")
        self.indicators = TechnicalIndicators()
        self.support_resistance = SupportResistanceLevels()
        self.pattern_recognition = PatternRecognition()
        self.regime_detector = MarketRegimeDetector()
        self.volatility_analyzer = VolatilityAnalyzer()

    def full_analysis(self, symbol: str, data: pl.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.

        Args:
            symbol: Trading symbol
            data: Historical market data

        Returns:
            Complete technical analysis results
        """
        try:
            # Calculate all indicators
            analyzed_data = self._calculate_comprehensive_indicators(data)

            # Market regime detection
            regime_info = self.regime_detector.detect_regime(analyzed_data)

            # Support/resistance levels
            support_levels, resistance_levels = (
                self.support_resistance.calculate_support_resistance(analyzed_data)
            )

            # Volatility analysis
            volatility_metrics = self.volatility_analyzer.calculate_volatility_metrics(
                analyzed_data
            )

            # Pattern detection
            patterns = self._detect_all_patterns(analyzed_data)

            # Current indicator values
            latest = analyzed_data.tail(1)
            current_indicators = self._extract_current_indicators(latest)

            # Generate technical score
            technical_score = self._calculate_technical_score(
                analyzed_data, latest, regime_info, patterns
            )

            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc),
                "regime": regime_info,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "volatility": volatility_metrics,
                "patterns": patterns,
                "indicators": current_indicators,
                "technical_score": technical_score,
                "data_points": analyzed_data.height,
            }

        except Exception as e:
            self.logger.error(f"Error in full technical analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc),
                "error": str(e),
                "technical_score": 50.0,  # Neutral score on error
            }

    def _calculate_comprehensive_indicators(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate all technical indicators efficiently."""
        result = data.clone()

        # Moving averages
        for period in [10, 20, 50, 100, 200]:
            result = self.indicators.sma(result, period)
            result = self.indicators.ema(result, period)

        # Oscillators
        result = self.indicators.rsi(result, 14)
        result = self.indicators.rsi(result, 21)  # Additional RSI period
        result = self.indicators.macd(result)
        result = self.indicators.stochastic(result)
        result = self.indicators.williams_r(result)
        result = self.indicators.cci(result)

        # Volatility indicators
        result = self.indicators.atr(result, 14)
        result = self.indicators.bollinger_bands(result, 20)

        # Volume indicators
        result = self.indicators.obv(result)
        result = self.indicators.vwap(result)

        # Momentum indicators
        result = self.indicators.momentum(result, 10)
        result = self.indicators.rate_of_change(result, 10)

        return result

    def _detect_all_patterns(self, data: pl.DataFrame) -> List[PatternSignal]:
        """Detect all chart patterns."""
        patterns = []

        # Detect various patterns
        pattern_methods = [
            self.pattern_recognition.detect_breakout,
            self.pattern_recognition.detect_breakdown,
            self.pattern_recognition.detect_double_top,
            self.pattern_recognition.detect_double_bottom,
        ]

        for method in pattern_methods:
            try:
                pattern = method(data)
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                self.logger.warning(f"Pattern detection error: {e}")

        return patterns

    def _extract_current_indicators(self, latest: pl.DataFrame) -> Dict[str, Any]:
        """Extract current indicator values."""
        indicators = {}

        # List of indicators to extract
        indicator_columns = [
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_10",
            "ema_20",
            "ema_50",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_position",
            "bb_width",
            "atr_14",
            "obv",
            "vwap",
            "stoch_k",
            "stoch_d",
            "williams_r_14",
            "cci_20",
            "momentum_10",
            "roc_10",
        ]

        for col in indicator_columns:
            if col in latest.columns:
                try:
                    value = latest.select(col).item()
                    if value is not None:
                        indicators[col] = float(value)
                except Exception:
                    continue

        return indicators

    def _calculate_technical_score(
        self,
        data: pl.DataFrame,
        latest: pl.DataFrame,
        regime_info: Dict,
        patterns: List[PatternSignal],
    ) -> float:
        """Calculate overall technical score (0-100)."""
        try:
            score = 50.0  # Start with neutral

            # Trend component (20 points)
            trend_score = self._score_trend(latest)
            score += (trend_score - 50) * 0.4

            # Momentum component (25 points)
            momentum_score = self._score_momentum(latest)
            score += (momentum_score - 50) * 0.5

            # Mean reversion component (20 points)
            mean_reversion_score = self._score_mean_reversion(latest)
            score += (mean_reversion_score - 50) * 0.4

            # Volume component (15 points)
            volume_score = self._score_volume(data, latest)
            score += (volume_score - 50) * 0.3

            # Pattern component (20 points)
            pattern_score = self._score_patterns(patterns)
            score += (pattern_score - 50) * 0.4

            # Regime adjustment
            if regime_info["regime"] == TrendDirection.SIDEWAYS:
                # Reduce confidence in trending signals during sideways markets
                if score > 60:
                    score = 50 + (score - 50) * 0.7
                elif score < 40:
                    score = 50 - (50 - score) * 0.7

            return max(0.0, min(100.0, score))

        except Exception as e:
            self.logger.error(f"Error calculating technical score: {e}")
            return 50.0

    def _score_trend(self, latest: pl.DataFrame) -> float:
        """Score trend indicators (0-100)."""
        score = 50.0

        try:
            # Moving average alignment
            sma20 = latest.select("sma_20").item()
            sma50 = latest.select("sma_50").item()
            ema20 = latest.select("ema_20").item()
            current_price = latest.select("close").item()

            if all(v is not None for v in [sma20, sma50, ema20, current_price]):
                # Price above moving averages
                if current_price > sma20 > sma50:
                    score += 15
                elif current_price > ema20:
                    score += 8
                elif current_price < sma20 < sma50:
                    score -= 15
                elif current_price < ema20:
                    score -= 8

                # Moving average slope
                if sma20 > sma50:
                    score += 10
                elif sma20 < sma50:
                    score -= 10

        except Exception:
            pass

        return max(0.0, min(100.0, score))

    def _score_momentum(self, latest: pl.DataFrame) -> float:
        """Score momentum indicators (0-100)."""
        score = 50.0

        try:
            # MACD
            macd_line = latest.select("macd_line").item()
            macd_signal = latest.select("macd_signal").item()
            macd_hist = latest.select("macd_histogram").item()

            if all(v is not None for v in [macd_line, macd_signal, macd_hist]):
                if macd_line > macd_signal and macd_hist > 0:
                    score += 15
                elif macd_line < macd_signal and macd_hist < 0:
                    score -= 15

            # Rate of Change
            roc = latest.select("roc_10").item()
            if roc is not None:
                if roc > 5:
                    score += 10
                elif roc > 0:
                    score += 5
                elif roc < -5:
                    score -= 10
                elif roc < 0:
                    score -= 5

            # Momentum
            momentum = latest.select("momentum_10").item()
            if momentum is not None:
                if momentum > 0:
                    score += 5
                else:
                    score -= 5

        except Exception:
            pass

        return max(0.0, min(100.0, score))

    def _score_mean_reversion(self, latest: pl.DataFrame) -> float:
        """Score mean reversion indicators (0-100)."""
        score = 50.0

        try:
            # RSI
            rsi = latest.select("rsi_14").item()
            if rsi is not None:
                if rsi <= 30:
                    score += 20  # Oversold - bullish for mean reversion
                elif rsi >= 70:
                    score -= 20  # Overbought - bearish for mean reversion
                elif 45 <= rsi <= 55:
                    score += 5  # Neutral zone

            # Bollinger Bands position
            bb_position = latest.select("bb_position").item()
            if bb_position is not None:
                if bb_position <= 0.1:
                    score += 15  # Near lower band
                elif bb_position >= 0.9:
                    score -= 15  # Near upper band

            # Stochastic
            stoch_k = latest.select("stoch_k").item()
            if stoch_k is not None:
                if stoch_k <= 20:
                    score += 10
                elif stoch_k >= 80:
                    score -= 10

        except Exception:
            pass

        return max(0.0, min(100.0, score))

    def _score_volume(self, data: pl.DataFrame, latest: pl.DataFrame) -> float:
        """Score volume indicators (0-100)."""
        score = 50.0

        try:
            # Volume trend
            current_volume = latest.select("volume").item()
            avg_volume = data.select("volume").tail(20).mean().item()

            if current_volume and avg_volume and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                if volume_ratio >= 2.0:
                    score += 20  # High volume
                elif volume_ratio >= 1.5:
                    score += 10  # Above average volume
                elif volume_ratio <= 0.5:
                    score -= 10  # Low volume

            # OBV trend
            if data.height >= 2:
                obv_current = latest.select("obv").item()
                obv_prev = data.select("obv").slice(-2, 1).item()

                if obv_current and obv_prev:
                    if obv_current > obv_prev:
                        score += 10
                    elif obv_current < obv_prev:
                        score -= 10

            # VWAP position
            current_price = latest.select("close").item()
            vwap = latest.select("vwap").item()

            if current_price and vwap:
                if current_price > vwap * 1.005:  # Above VWAP
                    score += 5
                elif current_price < vwap * 0.995:  # Below VWAP
                    score -= 5

        except Exception:
            pass

        return max(0.0, min(100.0, score))

    def _score_patterns(self, patterns: List[PatternSignal]) -> float:
        """Score chart patterns (0-100)."""
        if not patterns:
            return 50.0

        score = 50.0

        for pattern in patterns:
            if pattern.pattern in [PatternType.BREAKOUT, PatternType.DOUBLE_BOTTOM]:
                score += pattern.confidence * 0.3
            elif pattern.pattern in [PatternType.BREAKDOWN, PatternType.DOUBLE_TOP]:
                score -= pattern.confidence * 0.3

        return max(0.0, min(100.0, score))
