"""
Market Regime Detection Module

This module identifies market conditions (trending, ranging, volatile) and adjusts
strategy parameters accordingly. It implements regime filters to avoid trading
in unfavorable conditions and provides market state analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats

from shared.models import SignalType


class MarketRegime(Enum):
    """Market regime types."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class VolatilityRegime(Enum):
    """Volatility regimes."""

    EXTREME_LOW = "extreme_low"  # < 10th percentile
    LOW = "low"  # 10th-30th percentile
    NORMAL = "normal"  # 30th-70th percentile
    HIGH = "high"  # 70th-90th percentile
    EXTREME_HIGH = "extreme_high"  # > 90th percentile


@dataclass
class RegimeState:
    """Current market regime state."""

    primary_regime: MarketRegime
    volatility_regime: VolatilityRegime
    confidence: float
    trend_strength: float
    volatility_percentile: float
    regime_duration: int  # Days in current regime
    previous_regime: Optional[MarketRegime] = None
    regime_change_probability: float = 0.0
    favorable_for_trading: bool = True
    recommended_position_size: float = 1.0  # Multiplier for position sizing


class MarketRegimeDetector:
    """Advanced market regime detection using multiple methodologies."""

    def __init__(self, lookback_period: int = 100):
        """
        Initialize market regime detector.

        Args:
            lookback_period: Period for regime analysis
        """
        self.lookback_period = lookback_period
        self.logger = logging.getLogger("market_regime")
        self._regime_history = []
        self._volatility_history = []

    def detect_regime(self, data: pl.DataFrame) -> RegimeState:
        """
        Detect current market regime using multiple methodologies.

        Args:
            data: Historical market data

        Returns:
            Current regime state with confidence scores
        """
        try:
            if data.height < self.lookback_period:
                self.logger.warning(
                    f"Insufficient data for regime detection: "
                    f"{data.height} < {self.lookback_period}"
                )
                return self._default_regime_state()

            # Prepare data for analysis
            recent_data = data.tail(self.lookback_period)

            # Multiple regime detection methods
            trend_analysis = self._analyze_trend_regime(recent_data)
            volatility_analysis = self._analyze_volatility_regime(recent_data)
            momentum_analysis = self._analyze_momentum_regime(recent_data)
            volume_analysis = self._analyze_volume_regime(recent_data)
            pattern_analysis = self._analyze_pattern_regime(recent_data)

            # Combine analyses to determine primary regime
            regime_scores = self._combine_regime_analyses(
                trend_analysis,
                volatility_analysis,
                momentum_analysis,
                volume_analysis,
                pattern_analysis,
            )

            # Determine primary regime
            regime_scores_dict = dict(regime_scores)
            primary_regime: MarketRegime = max(
                regime_scores_dict.keys(), key=lambda k: regime_scores_dict[k]
            )
            confidence = regime_scores[primary_regime]

            # Volatility regime
            volatility_regime = self._classify_volatility_regime(
                float(volatility_analysis["current_volatility"]),
                float(volatility_analysis["volatility_percentile"]),
            )

            # Regime duration and stability
            regime_duration = self._calculate_regime_duration(primary_regime)
            regime_change_prob = self._calculate_regime_change_probability(recent_data)

            # Trading favorability
            favorable_for_trading = self._assess_trading_favorability(
                primary_regime, volatility_regime, confidence
            )

            # Position size adjustment
            position_size_multiplier = self._calculate_position_size_multiplier(
                primary_regime, volatility_regime, confidence
            )

            regime_state = RegimeState(
                primary_regime=primary_regime,
                volatility_regime=volatility_regime,
                confidence=confidence,
                trend_strength=trend_analysis["trend_strength"],
                volatility_percentile=volatility_analysis["volatility_percentile"],
                regime_duration=regime_duration,
                previous_regime=self._get_previous_regime(),
                regime_change_probability=regime_change_prob,
                favorable_for_trading=favorable_for_trading,
                recommended_position_size=position_size_multiplier,
            )

            # Update history
            self._update_regime_history(regime_state)

            return regime_state

        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            return self._default_regime_state()

    def _analyze_trend_regime(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics of the market."""
        try:
            # Linear regression trend
            closes = data.select("close").to_series().to_numpy()
            x = np.arange(len(closes))
            linregress_result = tuple(stats.linregress(x, closes))
            slope = float(linregress_result[0])
            r_value = float(linregress_result[2])

            # Normalize slope by average price
            avg_price = float(np.mean(closes))
            trend_strength = (slope / avg_price) * len(closes)

            # R-squared for trend reliability
            trend_reliability = r_value**2

            # Moving average analysis
            data_with_ma = data.with_columns(
                [
                    pl.col("close").rolling_mean(window_size=20).alias("ma20"),
                    pl.col("close").rolling_mean(window_size=50).alias("ma50"),
                ]
            )

            latest = data_with_ma.tail(1)
            current_price = latest.select("close").item()
            ma20 = latest.select("ma20").item()
            ma50 = latest.select("ma50").item()

            # Trend classification
            if trend_strength > 0.02 and trend_reliability > 0.7:
                trend_regime = MarketRegime.TRENDING_UP
                trend_score = min(100.0, 50 + trend_strength * 1000)
            elif trend_strength < -0.02 and trend_reliability > 0.7:
                trend_regime = MarketRegime.TRENDING_DOWN
                trend_score = min(100.0, 50 + abs(trend_strength) * 1000)
            else:
                trend_regime = MarketRegime.RANGING
                trend_score = max(0.0, 50 - abs(trend_strength) * 1000)

            # Moving average confirmation
            ma_alignment_score = 0
            if ma20 and ma50 and current_price:
                if current_price > ma20 > ma50:
                    ma_alignment_score = 20
                elif current_price < ma20 < ma50:
                    ma_alignment_score = 20
                elif abs(ma20 - ma50) / ma50 < 0.02:  # MAs close together
                    ma_alignment_score = -10

            return {
                "trend_regime": trend_regime,
                "trend_strength": trend_strength,
                "trend_reliability": trend_reliability,
                "trend_score": min(100.0, max(0.0, trend_score + ma_alignment_score)),
                "slope": slope,
                "r_squared": trend_reliability,
                "ma_alignment": ma_alignment_score,
            }

        except Exception as e:
            self.logger.error(f"Error in trend regime analysis: {e}")
            return {
                "trend_regime": MarketRegime.RANGING,
                "trend_strength": 0.0,
                "trend_score": 50.0,
            }

    def _analyze_volatility_regime(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Analyze volatility characteristics."""
        try:
            # Calculate returns
            returns_data = data.with_columns(
                [pl.col("close").pct_change().alias("returns")]
            ).drop_nulls()

            if returns_data.height < 10:
                return {
                    "volatility_regime": VolatilityRegime.NORMAL,
                    "current_volatility": 0.0,
                }

            returns = returns_data.select("returns").to_series().to_numpy()

            # Current volatility (annualized)
            current_vol = (
                np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.0
            )

            # Historical volatility percentiles
            vol_history = []
            window = 20
            for i in range(window, len(returns)):
                vol = np.std(returns[i - window : i]) * np.sqrt(252)
                vol_history.append(vol)

            if vol_history:
                vol_percentile = float(
                    (np.sum(np.array(vol_history) <= current_vol) / len(vol_history))
                    * 100
                )
            else:
                vol_percentile = 50.0

            # GARCH-like volatility clustering
            volatility_clustering = self._detect_volatility_clustering(returns)

            # ATR-based volatility
            atr_vol = self._calculate_atr_volatility(data)

            return {
                "volatility_regime": self._classify_volatility_regime(
                    current_vol, vol_percentile
                ),
                "current_volatility": current_vol,
                "volatility_percentile": vol_percentile,
                "atr_volatility": atr_vol,
                "volatility_clustering": volatility_clustering,
                "vol_score": self._score_volatility_for_trading(float(vol_percentile)),
            }

        except Exception as e:
            self.logger.error(f"Error in volatility regime analysis: {e}")
            return {
                "volatility_regime": VolatilityRegime.NORMAL,
                "current_volatility": 0.0,
                "vol_score": 50.0,
            }

    def _analyze_momentum_regime(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Analyze momentum characteristics."""
        try:
            # Calculate momentum indicators
            momentum_data = data.with_columns(
                [
                    # Price momentum
                    (pl.col("close") / pl.col("close").shift(10) - 1).alias(
                        "momentum_10d"
                    ),
                    (pl.col("close") / pl.col("close").shift(20) - 1).alias(
                        "momentum_20d"
                    ),
                    # Rate of change
                    (
                        (pl.col("close") - pl.col("close").shift(10))
                        / pl.col("close").shift(10)
                        * 100
                    ).alias("roc_10d"),
                ]
            ).drop_nulls()

            if momentum_data.height < 5:
                return {"momentum_regime": MarketRegime.RANGING, "momentum_score": 50.0}

            latest = momentum_data.tail(1)
            momentum_10d = latest.select("momentum_10d").item() or 0.0
            momentum_20d = latest.select("momentum_20d").item() or 0.0
            roc_10d = latest.select("roc_10d").item() or 0.0

            # Momentum regime classification
            if momentum_10d > 0.05 and momentum_20d > 0.03:
                momentum_regime = MarketRegime.TRENDING_UP
                momentum_score = min(90.0, 50 + momentum_10d * 500)
            elif momentum_10d < -0.05 and momentum_20d < -0.03:
                momentum_regime = MarketRegime.TRENDING_DOWN
                momentum_score = min(90.0, 50 + abs(momentum_10d) * 500)
            else:
                momentum_regime = MarketRegime.RANGING
                momentum_score = max(10.0, 50 - abs(momentum_10d) * 200)

            # Momentum acceleration
            momentum_acceleration = momentum_10d - momentum_20d
            acceleration_signal = (
                "accelerating"
                if momentum_acceleration > 0.01
                else "decelerating" if momentum_acceleration < -0.01 else "stable"
            )

            return {
                "momentum_regime": momentum_regime,
                "momentum_score": momentum_score,
                "momentum_10d": momentum_10d,
                "momentum_20d": momentum_20d,
                "roc_10d": roc_10d,
                "acceleration": momentum_acceleration,
                "acceleration_signal": acceleration_signal,
            }

        except Exception as e:
            self.logger.error(f"Error in momentum regime analysis: {e}")
            return {"momentum_regime": MarketRegime.RANGING, "momentum_score": 50.0}

    def _analyze_volume_regime(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns and regimes."""
        try:
            # Volume analysis
            recent_volume = data.select("volume").tail(5).mean().item()
            avg_volume = data.select("volume").tail(50).mean().item()
            volume_trend = data.select("volume").tail(20).to_series().to_numpy()

            # Volume trend analysis
            if len(volume_trend) >= 10:
                vol_linregress_result = tuple(
                    stats.linregress(np.arange(len(volume_trend)), volume_trend)
                )
                vol_slope = float(vol_linregress_result[0])
                vol_r_value = float(vol_linregress_result[2])
                volume_trend_strength = vol_slope / float(np.mean(volume_trend))
            else:
                volume_trend_strength = 0.0
                vol_r_value = 0.0

            # Volume surge detection
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Price-volume relationship
            price_volume_correlation = self._calculate_price_volume_correlation(data)

            # Volume regime classification
            if volume_ratio >= 2.0 and price_volume_correlation > 0.3:
                volume_regime = MarketRegime.ACCUMULATION
                volume_score = 80.0
            elif volume_ratio >= 2.0 and price_volume_correlation < -0.3:
                volume_regime = MarketRegime.DISTRIBUTION
                volume_score = 20.0
            elif volume_trend_strength > 0.02:
                volume_regime = MarketRegime.ACCUMULATION
                volume_score = 70.0
            elif volume_trend_strength < -0.02:
                volume_regime = MarketRegime.DISTRIBUTION
                volume_score = 30.0
            else:
                volume_regime = MarketRegime.RANGING
                volume_score = 50.0

            return {
                "volume_regime": volume_regime,
                "volume_score": volume_score,
                "volume_ratio": volume_ratio,
                "volume_trend_strength": volume_trend_strength,
                "price_volume_correlation": price_volume_correlation,
                "volume_trend_reliability": float(vol_r_value) ** 2,
            }

        except Exception as e:
            self.logger.error(f"Error in volume regime analysis: {e}")
            return {"volume_regime": MarketRegime.RANGING, "volume_score": 50.0}

    def _analyze_pattern_regime(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Analyze chart pattern regimes."""
        try:
            # Price range analysis
            recent_data = data.tail(50)
            price_range = self._calculate_price_range_metrics(recent_data)

            # Breakout/breakdown detection
            breakout_analysis = self._detect_breakout_regime(recent_data)

            # Support/resistance strength
            sr_analysis = self._analyze_support_resistance_strength(recent_data)

            # Combine pattern signals
            pattern_signals = []
            pattern_score = 50.0

            if breakout_analysis["is_breakout"]:
                pattern_signals.append("breakout")
                pattern_score += 20
            elif breakout_analysis["is_breakdown"]:
                pattern_signals.append("breakdown")
                pattern_score -= 20

            if sr_analysis["strong_support"]:
                pattern_signals.append("strong_support")
                pattern_score += 10
            elif sr_analysis["strong_resistance"]:
                pattern_signals.append("strong_resistance")
                pattern_score -= 10

            # Determine pattern regime
            if "breakout" in pattern_signals:
                pattern_regime = MarketRegime.BREAKOUT
            elif "breakdown" in pattern_signals:
                pattern_regime = MarketRegime.BREAKDOWN
            else:
                pattern_regime = MarketRegime.RANGING

            return {
                "pattern_regime": pattern_regime,
                "pattern_score": max(0.0, min(100.0, pattern_score)),
                "pattern_signals": pattern_signals,
                "price_range": price_range,
                "breakout_analysis": breakout_analysis,
                "support_resistance": sr_analysis,
            }

        except Exception as e:
            self.logger.error(f"Error in pattern regime analysis: {e}")
            return {"pattern_regime": MarketRegime.RANGING, "pattern_score": 50.0}

    def _combine_regime_analyses(
        self, trend: Dict, volatility: Dict, momentum: Dict, volume: Dict, pattern: Dict
    ) -> Dict[MarketRegime, float]:
        """Combine multiple regime analyses into final scores."""
        try:
            # Initialize regime scores
            regime_scores = {regime: 0.0 for regime in MarketRegime}

            # Trend analysis contribution (40% weight)
            trend_regime = trend.get("trend_regime", MarketRegime.RANGING)
            trend_score = trend.get("trend_score", 50.0)
            regime_scores[trend_regime] += trend_score * 0.4

            # Momentum analysis contribution (25% weight)
            momentum_regime = momentum.get("momentum_regime", MarketRegime.RANGING)
            momentum_score = momentum.get("momentum_score", 50.0)
            regime_scores[momentum_regime] += momentum_score * 0.25

            # Volume analysis contribution (20% weight)
            volume_regime = volume.get("volume_regime", MarketRegime.RANGING)
            volume_score = volume.get("volume_score", 50.0)
            regime_scores[volume_regime] += volume_score * 0.2

            # Pattern analysis contribution (15% weight)
            pattern_regime = pattern.get("pattern_regime", MarketRegime.RANGING)
            pattern_score = pattern.get("pattern_score", 50.0)
            regime_scores[pattern_regime] += pattern_score * 0.15

            # Normalize scores
            for regime in regime_scores:
                regime_scores[regime] = min(100.0, regime_scores[regime])

            return regime_scores

        except Exception as e:
            self.logger.error(f"Error combining regime analyses: {e}")
            return {MarketRegime.RANGING: 50.0}

    def _classify_volatility_regime(
        self, current_vol: float, vol_percentile: float
    ) -> VolatilityRegime:
        """Classify volatility regime based on current volatility and percentile."""
        if vol_percentile >= 90:
            return VolatilityRegime.EXTREME_HIGH
        elif vol_percentile >= 70:
            return VolatilityRegime.HIGH
        elif vol_percentile >= 30:
            return VolatilityRegime.NORMAL
        elif vol_percentile >= 10:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.EXTREME_LOW

    def _calculate_price_range_metrics(self, data: pl.DataFrame) -> Dict[str, float]:
        """Calculate price range and consolidation metrics."""
        try:
            high = data.select("high").max().item()
            low = data.select("low").min().item()
            current = data.select("close").tail(1).item()

            range_pct = (high - low) / low if low > 0 else 0.0
            position_in_range = (current - low) / (high - low) if high != low else 0.5

            # Consolidation detection
            recent_range = data.tail(20)
            recent_high = recent_range.select("high").max().item()
            recent_low = recent_range.select("low").min().item()
            recent_range_pct = (
                (recent_high - recent_low) / recent_low if recent_low > 0 else 0.0
            )

            consolidation_score = max(0.0, 100.0 - recent_range_pct * 500)

            return {
                "total_range_pct": range_pct,
                "position_in_range": position_in_range,
                "recent_range_pct": recent_range_pct,
                "consolidation_score": consolidation_score,
                "is_consolidating": recent_range_pct < 0.05,  # Less than 5% range
            }

        except Exception as e:
            self.logger.error(f"Error calculating price range metrics: {e}")
            return {"total_range_pct": 0.0, "consolidation_score": 50.0}

    def _detect_breakout_regime(self, data: pl.DataFrame) -> Dict[str, Any]:
        """Detect breakout/breakdown regimes."""
        try:
            # Recent price action
            recent = data.tail(20)
            current_price = recent.select("close").tail(1).item()

            # Resistance and support levels
            resistance = recent.select("high").quantile(0.9).item()
            support = recent.select("low").quantile(0.1).item()

            # Volume confirmation
            recent_volume = recent.select("volume").tail(5).mean().item()
            avg_volume = recent.select("volume").mean().item()
            volume_confirmation = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Breakout detection
            is_breakout = (
                current_price > resistance * 1.002 and volume_confirmation > 1.5
            )

            is_breakdown = current_price < support * 0.998 and volume_confirmation > 1.5

            # Breakout strength
            if is_breakout:
                breakout_strength = ((current_price - resistance) / resistance) * 100
            elif is_breakdown:
                breakout_strength = ((support - current_price) / support) * 100
            else:
                breakout_strength = 0.0

            return {
                "is_breakout": is_breakout,
                "is_breakdown": is_breakdown,
                "breakout_strength": breakout_strength,
                "resistance_level": resistance,
                "support_level": support,
                "volume_confirmation": volume_confirmation,
                "breakout_score": min(100.0, max(0.0, 50 + breakout_strength * 10)),
            }

        except Exception as e:
            self.logger.error(f"Error detecting breakout regime: {e}")
            return {"is_breakout": False, "is_breakdown": False, "breakout_score": 50.0}

    def _analyze_support_resistance_strength(
        self, data: pl.DataFrame
    ) -> Dict[str, Any]:
        """Analyze strength of support and resistance levels."""
        try:
            # Find pivot points
            pivots = data.with_columns(
                [
                    # Pivot highs
                    pl.when(
                        (pl.col("high") == pl.col("high").rolling_max(window_size=5))
                        & (pl.col("high") > pl.col("high").shift(1))
                        & (pl.col("high") > pl.col("high").shift(-1))
                    )
                    .then(pl.col("high"))
                    .otherwise(None)
                    .alias("pivot_high"),
                    # Pivot lows
                    pl.when(
                        (pl.col("low") == pl.col("low").rolling_min(window_size=5))
                        & (pl.col("low") < pl.col("low").shift(1))
                        & (pl.col("low") < pl.col("low").shift(-1))
                    )
                    .then(pl.col("low"))
                    .otherwise(None)
                    .alias("pivot_low"),
                ]
            )

            # Count touches of key levels
            current_price = data.select("close").tail(1).item()

            # Resistance touches
            resistance_touches = 0
            support_touches = 0

            # Simplified approach - count how many times price approached key levels
            highs = data.select("high").tail(20).to_series().to_numpy()
            lows = data.select("low").tail(20).to_series().to_numpy()

            if len(highs) > 0 and len(lows) > 0:
                key_resistance = np.percentile(highs, 95)
                key_support = np.percentile(lows, 5)

                # Count approaches to these levels
                for high in highs:
                    if abs(high - key_resistance) / key_resistance < 0.01:
                        resistance_touches += 1

                for low in lows:
                    if abs(low - key_support) / key_support < 0.01:
                        support_touches += 1

            strong_resistance = resistance_touches >= 3
            strong_support = support_touches >= 3

            return {
                "strong_resistance": strong_resistance,
                "strong_support": strong_support,
                "resistance_touches": resistance_touches,
                "support_touches": support_touches,
                "key_resistance": locals().get("key_resistance", current_price),
                "key_support": locals().get("key_support", current_price),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing support/resistance strength: {e}")
            return {"strong_resistance": False, "strong_support": False}

    def _detect_volatility_clustering(self, returns: np.ndarray) -> Dict[str, Any]:
        """Detect volatility clustering patterns."""
        try:
            if len(returns) < 30:
                return {"clustering_detected": False, "clustering_strength": 0.0}

            # Calculate rolling volatility
            window = 10
            rolling_vols = []
            for i in range(window, len(returns)):
                vol = np.std(returns[i - window : i])
                rolling_vols.append(vol)

            if len(rolling_vols) < 10:
                return {"clustering_detected": False, "clustering_strength": 0.0}

            # Detect clustering using autocorrelation
            vol_series = np.array(rolling_vols)
            vol_autocorr = np.corrcoef(vol_series[:-1], vol_series[1:])[0, 1]

            clustering_detected = vol_autocorr > 0.3
            clustering_strength = max(0.0, vol_autocorr)

            return {
                "clustering_detected": clustering_detected,
                "clustering_strength": clustering_strength,
                "vol_autocorr": vol_autocorr,
            }

        except Exception as e:
            self.logger.error(f"Error detecting volatility clustering: {e}")
            return {"clustering_detected": False, "clustering_strength": 0.0}

    def _calculate_atr_volatility(self, data: pl.DataFrame) -> float:
        """Calculate ATR-based volatility measure."""
        try:
            # Calculate True Range
            tr_data = data.with_columns(
                [
                    (pl.col("high") - pl.col("low")).alias("hl"),
                    (pl.col("high") - pl.col("close").shift(1)).abs().alias("hc"),
                    (pl.col("low") - pl.col("close").shift(1)).abs().alias("lc"),
                ]
            ).with_columns([pl.max_horizontal(["hl", "hc", "lc"]).alias("true_range")])

            # Average True Range
            atr = tr_data.select("true_range").tail(14).mean().item()
            current_price = data.select("close").tail(1).item()

            if atr and current_price and current_price > 0:
                return (atr / current_price) * np.sqrt(252)  # Annualized
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating ATR volatility: {e}")
            return 0.0

    def _calculate_price_volume_correlation(self, data: pl.DataFrame) -> float:
        """Calculate price-volume correlation."""
        try:
            # Calculate price changes and volume
            corr_data = data.with_columns(
                [
                    pl.col("close").pct_change().alias("price_change"),
                    pl.col("volume").alias("volume"),
                ]
            ).drop_nulls()

            if corr_data.height < 10:
                return 0.0

            price_changes = corr_data.select("price_change").to_series().to_numpy()
            volumes = corr_data.select("volume").to_series().to_numpy()

            correlation = np.corrcoef(price_changes, volumes)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating price-volume correlation: {e}")
            return 0.0

    def _score_volatility_for_trading(self, vol_percentile: float) -> float:
        """Score volatility favorability for trading."""
        # Moderate volatility is best for most strategies
        if 30 <= vol_percentile <= 70:
            return 80.0  # Good for trading
        elif 20 <= vol_percentile <= 80:
            return 60.0  # Acceptable
        elif vol_percentile >= 90:
            return 20.0  # Too volatile
        elif vol_percentile <= 10:
            return 30.0  # Too quiet
        else:
            return 40.0  # Suboptimal

    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long we've been in the current regime."""
        if not self._regime_history:
            return 1

        duration = 1
        for i in range(len(self._regime_history) - 1, -1, -1):
            if self._regime_history[i] == current_regime:
                duration += 1
            else:
                break

        return duration

    def _calculate_regime_change_probability(self, data: pl.DataFrame) -> float:
        """Calculate probability of regime change."""
        try:
            # Use volatility and trend instability as proxy
            if data.height < 20:
                return 0.1  # Low probability with insufficient data

            # Recent volatility vs historical
            recent_vol = self._calculate_atr_volatility(data.tail(10))
            historical_vol = self._calculate_atr_volatility(data)

            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0

            # Trend instability
            recent_closes = data.select("close").tail(10).to_series().to_numpy()
            if len(recent_closes) >= 10:
                recent_linregress_result = tuple(
                    stats.linregress(np.arange(len(recent_closes)), recent_closes)
                )
                recent_r = float(recent_linregress_result[2])
                trend_instability = 1.0 - recent_r**2
            else:
                trend_instability = 0.5

            # Combine factors
            change_prob = min(0.8, (vol_ratio - 1.0) * 0.5 + trend_instability * 0.3)
            return max(0.0, change_prob)

        except Exception as e:
            self.logger.error(f"Error calculating regime change probability: {e}")
            return 0.1

    def _assess_trading_favorability(
        self, regime: MarketRegime, vol_regime: VolatilityRegime, confidence: float
    ) -> bool:
        """Assess if current regime is favorable for trading."""
        try:
            # High confidence regimes are generally favorable
            if confidence < 60:
                return False

            # Regime-specific favorability
            favorable_regimes = {
                MarketRegime.TRENDING_UP: True,
                MarketRegime.TRENDING_DOWN: True,
                MarketRegime.BREAKOUT: True,
                MarketRegime.BREAKDOWN: True,
                MarketRegime.ACCUMULATION: True,
                MarketRegime.RANGING: False,  # Generally unfavorable
                MarketRegime.DISTRIBUTION: False,
                MarketRegime.HIGH_VOLATILITY: False,
            }

            regime_favorable = favorable_regimes.get(regime, True)

            # Volatility favorability
            vol_favorable = vol_regime not in [
                VolatilityRegime.EXTREME_HIGH,
                VolatilityRegime.EXTREME_LOW,
            ]

            return regime_favorable and vol_favorable

        except Exception:
            return True  # Default to favorable if analysis fails

    def _calculate_position_size_multiplier(
        self, regime: MarketRegime, vol_regime: VolatilityRegime, confidence: float
    ) -> float:
        """Calculate position size multiplier based on regime."""
        try:
            multiplier = 1.0

            # Regime-based adjustments
            regime_multipliers = {
                MarketRegime.TRENDING_UP: 1.2,
                MarketRegime.TRENDING_DOWN: 1.1,
                MarketRegime.BREAKOUT: 1.3,
                MarketRegime.BREAKDOWN: 1.1,
                MarketRegime.RANGING: 0.7,
                MarketRegime.HIGH_VOLATILITY: 0.6,
                MarketRegime.ACCUMULATION: 1.1,
                MarketRegime.DISTRIBUTION: 0.8,
            }

            multiplier *= regime_multipliers.get(regime, 1.0)

            # Volatility-based adjustments
            vol_multipliers = {
                VolatilityRegime.EXTREME_LOW: 1.2,
                VolatilityRegime.LOW: 1.1,
                VolatilityRegime.NORMAL: 1.0,
                VolatilityRegime.HIGH: 0.8,
                VolatilityRegime.EXTREME_HIGH: 0.5,
            }

            multiplier *= vol_multipliers.get(vol_regime, 1.0)

            # Confidence adjustment
            confidence_factor = confidence / 100.0
            multiplier *= 0.5 + confidence_factor * 0.5

            return max(0.1, min(2.0, multiplier))

        except Exception:
            return 1.0

    def _get_previous_regime(self) -> Optional[MarketRegime]:
        """Get the previous regime from history."""
        return self._regime_history[-2] if len(self._regime_history) >= 2 else None

    def _update_regime_history(self, regime_state: RegimeState) -> None:
        """Update regime history."""
        self._regime_history.append(regime_state.primary_regime)
        # Keep only last 100 regime states
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]

    def _default_regime_state(self) -> RegimeState:
        """Return default regime state when detection fails."""
        return RegimeState(
            primary_regime=MarketRegime.RANGING,
            volatility_regime=VolatilityRegime.NORMAL,
            confidence=50.0,
            trend_strength=0.0,
            volatility_percentile=50.0,
            regime_duration=1,
            favorable_for_trading=True,
            recommended_position_size=1.0,
        )


class RegimeFilter:
    """Filter trading signals based on market regime."""

    def __init__(self):
        """Initialize regime filter."""
        self.logger = logging.getLogger("regime_filter")

        # Define regime-strategy compatibility
        self.regime_strategy_compatibility = {
            # Momentum strategies work well in trending markets
            "momentum": {
                MarketRegime.TRENDING_UP: 1.0,
                MarketRegime.TRENDING_DOWN: 1.0,
                MarketRegime.BREAKOUT: 1.2,
                MarketRegime.BREAKDOWN: 1.1,
                MarketRegime.RANGING: 0.3,
                MarketRegime.HIGH_VOLATILITY: 0.6,
                MarketRegime.ACCUMULATION: 0.8,
                MarketRegime.DISTRIBUTION: 0.7,
            },
            # Mean reversion strategies work well in ranging markets
            "mean_reversion": {
                MarketRegime.TRENDING_UP: 0.4,
                MarketRegime.TRENDING_DOWN: 0.4,
                MarketRegime.BREAKOUT: 0.2,
                MarketRegime.BREAKDOWN: 0.2,
                MarketRegime.RANGING: 1.2,
                MarketRegime.HIGH_VOLATILITY: 1.0,
                MarketRegime.ACCUMULATION: 1.0,
                MarketRegime.DISTRIBUTION: 0.8,
            },
            # Trend following strategies
            "trend_following": {
                MarketRegime.TRENDING_UP: 1.3,
                MarketRegime.TRENDING_DOWN: 1.3,
                MarketRegime.BREAKOUT: 1.1,
                MarketRegime.BREAKDOWN: 1.1,
                MarketRegime.RANGING: 0.2,
                MarketRegime.HIGH_VOLATILITY: 0.7,
                MarketRegime.ACCUMULATION: 1.0,
                MarketRegime.DISTRIBUTION: 0.9,
            },
        }

    def should_trade(
        self, regime_state: RegimeState, strategy_type: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Determine if trading should be allowed in current regime.

        Args:
            regime_state: Current market regime state
            strategy_type: Type of strategy (momentum, mean_reversion, trend_following, balanced)

        Returns:
            Trading decision with reasoning
        """
        try:
            # Base favorability from regime state
            base_favorable = regime_state.favorable_for_trading

            # Strategy-specific compatibility
            compatibility = self.regime_strategy_compatibility.get(strategy_type, {})
            regime_multiplier = compatibility.get(regime_state.primary_regime, 0.8)

            # Confidence threshold
            confidence_ok = regime_state.confidence >= 60.0

            # Volatility check
            vol_ok = regime_state.volatility_regime not in [
                VolatilityRegime.EXTREME_HIGH,
                VolatilityRegime.EXTREME_LOW,
            ]

            # Regime stability (avoid trading during regime transitions)
            regime_stable = regime_state.regime_change_probability < 0.6

            # Combined decision
            should_trade = (
                base_favorable
                and confidence_ok
                and vol_ok
                and regime_stable
                and regime_multiplier >= 0.6
            )

            # Reasoning
            reasons = []
            if not base_favorable:
                reasons.append("unfavorable regime")
            if not confidence_ok:
                reasons.append(f"low confidence ({regime_state.confidence:.1f})")
            if not vol_ok:
                reasons.append(
                    f"extreme volatility ({regime_state.volatility_regime.value})"
                )
            if not regime_stable:
                reasons.append(
                    f"regime transition likely ({regime_state.regime_change_probability:.1f})"
                )
            if regime_multiplier < 0.6:
                reasons.append("strategy incompatible with regime")

            reasoning = "; ".join(reasons) if reasons else "favorable conditions"

            return {
                "should_trade": should_trade,
                "confidence_multiplier": regime_multiplier,
                "position_size_multiplier": regime_state.recommended_position_size
                * regime_multiplier,
                "reasoning": reasoning,
                "regime_details": {
                    "primary_regime": regime_state.primary_regime.value,
                    "volatility_regime": regime_state.volatility_regime.value,
                    "confidence": regime_state.confidence,
                    "regime_duration": regime_state.regime_duration,
                },
            }

        except Exception as e:
            self.logger.error(f"Error assessing trading favorability: {e}")
            return {
                "should_trade": False,
                "confidence_multiplier": 0.5,
                "position_size_multiplier": 0.5,
                "reasoning": f"regime filter error: {str(e)}",
            }

    def adjust_strategy_parameters(
        self, base_params: Dict[str, Any], regime_state: RegimeState
    ) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on current market regime.

        Args:
            base_params: Base strategy parameters
            regime_state: Current regime state

        Returns:
            Adjusted parameters
        """
        try:
            adjusted_params = base_params.copy()

            # Regime-specific parameter adjustments
            if regime_state.primary_regime == MarketRegime.TRENDING_UP:
                # Favor trend-following parameters
                adjusted_params["stop_loss_pct"] = min(
                    base_params.get("stop_loss_pct", 0.02) * 1.5, 0.05
                )
                adjusted_params["take_profit_pct"] = (
                    base_params.get("take_profit_pct", 0.04) * 1.2
                )
                adjusted_params["min_confidence"] = max(
                    base_params.get("min_confidence", 60) - 10, 50
                )

            elif regime_state.primary_regime == MarketRegime.TRENDING_DOWN:
                # Tighter stops in downtrends
                adjusted_params["stop_loss_pct"] = (
                    base_params.get("stop_loss_pct", 0.02) * 0.8
                )
                adjusted_params["take_profit_pct"] = (
                    base_params.get("take_profit_pct", 0.04) * 0.8
                )
                adjusted_params["min_confidence"] = (
                    base_params.get("min_confidence", 60) + 10
                )

            elif regime_state.primary_regime == MarketRegime.RANGING:
                # Mean reversion parameters
                adjusted_params["stop_loss_pct"] = (
                    base_params.get("stop_loss_pct", 0.02) * 1.2
                )
                adjusted_params["take_profit_pct"] = (
                    base_params.get("take_profit_pct", 0.04) * 0.8
                )
                adjusted_params["min_confidence"] = (
                    base_params.get("min_confidence", 60) + 5
                )

            elif regime_state.primary_regime in [MarketRegime.HIGH_VOLATILITY]:
                # Conservative parameters for high volatility
                adjusted_params["stop_loss_pct"] = (
                    base_params.get("stop_loss_pct", 0.02) * 0.7
                )
                adjusted_params["take_profit_pct"] = (
                    base_params.get("take_profit_pct", 0.04) * 0.7
                )
                adjusted_params["max_position_size"] = (
                    base_params.get("max_position_size", 0.2) * 0.7
                )
                adjusted_params["min_confidence"] = (
                    base_params.get("min_confidence", 60) + 15
                )

            # Volatility regime adjustments
            if regime_state.volatility_regime == VolatilityRegime.EXTREME_HIGH:
                adjusted_params["max_position_size"] = (
                    base_params.get("max_position_size", 0.2) * 0.5
                )
                adjusted_params["min_confidence"] = (
                    base_params.get("min_confidence", 60) + 20
                )

            elif regime_state.volatility_regime == VolatilityRegime.EXTREME_LOW:
                adjusted_params["min_confidence"] = (
                    base_params.get("min_confidence", 60) + 10
                )

            return adjusted_params

        except Exception as e:
            self.logger.error(f"Error adjusting strategy parameters: {e}")
            return base_params


class MarketStateAnalyzer:
    """Analyze overall market state and conditions."""

    def __init__(self):
        """Initialize market state analyzer."""
        self.logger = logging.getLogger("market_state")

    def analyze_market_breadth(self, sector_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze market breadth using sector performance.

        Args:
            sector_data: Dictionary of sector performances

        Returns:
            Market breadth analysis
        """
        try:
            if not sector_data:
                return {"breadth_score": 50.0, "breadth_signal": "neutral"}

            performances = list(sector_data.values())
            positive_sectors = sum(1 for perf in performances if perf > 0)
            total_sectors = len(performances)

            breadth_ratio = positive_sectors / total_sectors
            avg_performance = np.mean(performances)
            performance_std = np.std(performances)

            # Breadth score
            if breadth_ratio >= 0.8:
                breadth_score = 90.0
                breadth_signal = "strong_breadth"
            elif breadth_ratio >= 0.6:
                breadth_score = 70.0
                breadth_signal = "good_breadth"
            elif breadth_ratio >= 0.4:
                breadth_score = 50.0
                breadth_signal = "neutral_breadth"
            elif breadth_ratio >= 0.2:
                breadth_score = 30.0
                breadth_signal = "weak_breadth"
            else:
                breadth_score = 10.0
                breadth_signal = "very_weak_breadth"

            return {
                "breadth_score": breadth_score,
                "breadth_signal": breadth_signal,
                "breadth_ratio": breadth_ratio,
                "avg_sector_performance": avg_performance,
                "sector_dispersion": performance_std,
                "positive_sectors": positive_sectors,
                "total_sectors": total_sectors,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing market breadth: {e}")
            return {"breadth_score": 50.0, "breadth_signal": "error"}

    def detect_market_stress(
        self, vix_level: Optional[float] = None, correlation_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Detect market stress conditions.

        Args:
            vix_level: VIX volatility index level
            correlation_data: Asset correlation data

        Returns:
            Market stress analysis
        """
        try:
            stress_signals = []
            stress_score = 0.0

            # VIX-based stress detection
            if vix_level is not None:
                if vix_level >= 40:
                    stress_signals.append("extreme_fear")
                    stress_score += 40
                elif vix_level >= 30:
                    stress_signals.append("high_fear")
                    stress_score += 25
                elif vix_level >= 20:
                    stress_signals.append("moderate_fear")
                    stress_score += 10
                elif vix_level <= 12:
                    stress_signals.append("complacency")
                    stress_score += 15  # Complacency is also a stress signal

            # Correlation-based stress
            if correlation_data:
                avg_correlation = correlation_data.get("average_correlation", 0.5)
                if avg_correlation >= 0.8:
                    stress_signals.append("high_correlation")
                    stress_score += 20

            # Classify stress level
            if stress_score >= 50:
                stress_level = "extreme"
            elif stress_score >= 30:
                stress_level = "high"
            elif stress_score >= 15:
                stress_level = "moderate"
            else:
                stress_level = "low"

            return {
                "stress_level": stress_level,
                "stress_score": stress_score,
                "stress_signals": stress_signals,
                "vix_level": vix_level,
                "avoid_trading": stress_score >= 40,
            }

        except Exception as e:
            self.logger.error(f"Error detecting market stress: {e}")
            return {"stress_level": "unknown", "stress_score": 25.0}


class RegimeBasedParameterOptimizer:
    """Optimize strategy parameters based on detected market regime."""

    def __init__(self):
        """Initialize parameter optimizer."""
        self.logger = logging.getLogger("regime_optimizer")

        # Optimal parameters for different regimes
        self.regime_optimal_params = {
            MarketRegime.TRENDING_UP: {
                "momentum_lookback": 10,
                "ma_fast": 8,
                "ma_slow": 21,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 1.3,
            },
            MarketRegime.TRENDING_DOWN: {
                "momentum_lookback": 8,
                "ma_fast": 5,
                "ma_slow": 15,
                "rsi_oversold": 35,
                "rsi_overbought": 70,
                "stop_loss_multiplier": 0.8,
                "take_profit_multiplier": 0.9,
            },
            MarketRegime.RANGING: {
                "momentum_lookback": 20,
                "ma_fast": 15,
                "ma_slow": 30,
                "rsi_oversold": 20,
                "rsi_overbought": 80,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 0.8,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "momentum_lookback": 5,
                "ma_fast": 3,
                "ma_slow": 10,
                "rsi_oversold": 40,
                "rsi_overbought": 60,
                "stop_loss_multiplier": 0.6,
                "take_profit_multiplier": 0.6,
            },
        }

    def optimize_for_regime(
        self, base_params: Dict[str, Any], regime_state: RegimeState
    ) -> Dict[str, Any]:
        """
        Optimize parameters for current market regime.

        Args:
            base_params: Base strategy parameters
            regime_state: Current market regime

        Returns:
            Optimized parameters
        """
        try:
            optimized = base_params.copy()
            regime = regime_state.primary_regime

            # Get optimal parameters for regime
            optimal_params = self.regime_optimal_params.get(regime, {})

            # Apply regime-specific optimizations
            for param, value in optimal_params.items():
                if param.endswith("_multiplier"):
                    # Apply multipliers to existing parameters
                    base_param = param.replace("_multiplier", "")
                    if base_param in optimized:
                        optimized[base_param] = optimized[base_param] * value
                else:
                    # Direct parameter replacement
                    optimized[param] = value

            # Volatility adjustments
            vol_regime = regime_state.volatility_regime
            if vol_regime == VolatilityRegime.EXTREME_HIGH:
                # Reduce all risk parameters in extreme volatility
                risk_params = ["stop_loss_pct", "take_profit_pct", "max_position_size"]
                for param in risk_params:
                    if param in optimized:
                        optimized[param] = optimized[param] * 0.7

            elif vol_regime == VolatilityRegime.EXTREME_LOW:
                # Can be more aggressive in low volatility
                if "max_position_size" in optimized:
                    optimized["max_position_size"] = min(
                        0.3, optimized["max_position_size"] * 1.2
                    )

            # Confidence-based adjustments
            confidence_factor = regime_state.confidence / 100.0
            if "min_confidence" in optimized:
                # Higher regime confidence allows lower signal confidence threshold
                adjustment = (1.0 - confidence_factor) * 10
                optimized["min_confidence"] = max(
                    40.0, optimized["min_confidence"] - adjustment
                )

            self.logger.info(f"Optimized parameters for {regime.value} regime")

            return optimized

        except Exception as e:
            self.logger.error(f"Error optimizing parameters for regime: {e}")
            return base_params


class RegimeAwareStrategyManager:
    """Manage strategy behavior based on market regime."""

    def __init__(self):
        """Initialize regime-aware strategy manager."""
        self.logger = logging.getLogger("regime_strategy_manager")
        self.detector = MarketRegimeDetector()
        self.filter = RegimeFilter()
        self.optimizer = RegimeBasedParameterOptimizer()
        self.state_analyzer = MarketStateAnalyzer()

        # Current regime state
        self._current_regime = None
        self._regime_history = []

    async def get_regime_adjusted_signal(
        self,
        base_signal: Any,
        market_data: pl.DataFrame,
        strategy_type: str = "balanced",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Adjust trading signal based on current market regime.

        Args:
            base_signal: Original trading signal
            market_data: Market data for regime detection
            strategy_type: Type of strategy

        Returns:
            Tuple of (adjusted_signal, regime_info)
        """
        try:
            # Detect current regime
            regime_state = self.detector.detect_regime(market_data)
            self._current_regime = regime_state

            # Check if trading should be allowed
            trading_decision = self.filter.should_trade(regime_state, strategy_type)

            # Adjust signal based on regime
            if not trading_decision["should_trade"]:
                # Override signal to HOLD
                adjusted_signal = (
                    base_signal.copy() if hasattr(base_signal, "copy") else base_signal
                )
                if hasattr(adjusted_signal, "action"):
                    adjusted_signal.action = SignalType.HOLD
                    adjusted_signal.confidence = 0.0
                    adjusted_signal.reasoning = (
                        f"Regime filter: {trading_decision['reasoning']}"
                    )

            else:
                # Adjust signal confidence and position size
                adjusted_signal = (
                    base_signal.copy() if hasattr(base_signal, "copy") else base_signal
                )

                if hasattr(adjusted_signal, "confidence"):
                    # Apply confidence multiplier
                    conf_multiplier = trading_decision.get("confidence_multiplier", 1.0)
                    adjusted_signal.confidence = min(
                        100.0, adjusted_signal.confidence * conf_multiplier
                    )

                if hasattr(adjusted_signal, "position_size"):
                    # Apply position size multiplier
                    pos_multiplier = trading_decision.get(
                        "position_size_multiplier", 1.0
                    )
                    adjusted_signal.position_size = min(
                        0.5, adjusted_signal.position_size * pos_multiplier
                    )

                # Update reasoning
                if hasattr(adjusted_signal, "reasoning"):
                    regime_note = f" | Regime: {regime_state.primary_regime.value} (conf: {regime_state.confidence:.0f})"
                    adjusted_signal.reasoning += regime_note

                # Add regime metadata
                if hasattr(adjusted_signal, "metadata"):
                    adjusted_signal.metadata.update(
                        {
                            "regime_state": regime_state.__dict__,
                            "trading_decision": trading_decision,
                            "regime_adjustments_applied": True,
                        }
                    )

            # Prepare regime info
            regime_info = {
                "regime_state": regime_state,
                "trading_decision": trading_decision,
                "adjustments_applied": True,
                "regime_duration": regime_state.regime_duration,
                "regime_confidence": regime_state.confidence,
            }

            return adjusted_signal, regime_info

        except Exception as e:
            self.logger.error(f"Error adjusting signal for regime: {e}")
            # Return original signal with error info
            regime_info = {
                "error": str(e),
                "adjustments_applied": False,
                "regime_state": self._current_regime,
            }
            return base_signal, regime_info

    def get_current_regime(self) -> Optional[RegimeState]:
        """Get current market regime state."""
        return self._current_regime

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime state."""
        if not self._current_regime:
            return {"status": "no_regime_detected"}

        regime = self._current_regime
        return {
            "primary_regime": regime.primary_regime.value,
            "volatility_regime": regime.volatility_regime.value,
            "confidence": regime.confidence,
            "trend_strength": regime.trend_strength,
            "regime_duration": regime.regime_duration,
            "favorable_for_trading": regime.favorable_for_trading,
            "position_size_multiplier": regime.recommended_position_size,
            "regime_change_probability": regime.regime_change_probability,
        }

    def should_halt_trading(self, stress_threshold: float = 40.0) -> Dict[str, Any]:
        """
        Determine if trading should be halted due to market conditions.

        Args:
            stress_threshold: Stress score threshold for halting

        Returns:
            Trading halt decision
        """
        try:
            if not self._current_regime:
                return {"should_halt": False, "reason": "no_regime_data"}

            regime = self._current_regime

            # Halt conditions
            halt_reasons = []

            # Extreme volatility
            if regime.volatility_regime == VolatilityRegime.EXTREME_HIGH:
                halt_reasons.append("extreme_volatility")

            # Low confidence in regime detection
            if regime.confidence < 30.0:
                halt_reasons.append("low_regime_confidence")

            # High probability of regime change
            if regime.regime_change_probability > 0.7:
                halt_reasons.append("regime_transition")

            # Unfavorable trading conditions
            if not regime.favorable_for_trading:
                halt_reasons.append("unfavorable_conditions")

            should_halt = len(halt_reasons) >= 2  # Multiple warning signals

            return {
                "should_halt": should_halt,
                "halt_reasons": halt_reasons,
                "regime_summary": self.get_regime_summary(),
                "recommendation": "halt_trading" if should_halt else "continue_trading",
            }

        except Exception as e:
            self.logger.error(f"Error determining trading halt: {e}")
            return {"should_halt": True, "reason": f"error: {str(e)}"}
