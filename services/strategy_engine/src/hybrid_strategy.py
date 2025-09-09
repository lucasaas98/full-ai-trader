"""
Hybrid Strategy Module

This module implements a comprehensive hybrid strategy that combines technical
and fundamental analysis with intelligent weighting based on market conditions
and trading modes (day trading vs swing trading).
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import polars as pl

from shared.models import FinVizData, SignalType

from .base_strategy import BaseStrategy, Signal, StrategyConfig, StrategyMode
from .fundamental_analysis import FundamentalAnalysisEngine, FundamentalStrategy
from .market_regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeAwareStrategyManager,
    RegimeState,
    VolatilityRegime,
)
from .multi_timeframe_analyzer import (
    MultiTimeFrameConfirmation,
    create_multi_timeframe_analyzer,
)
from .multi_timeframe_data import create_multi_timeframe_fetcher
from .technical_analysis import TechnicalAnalysisEngine, TechnicalStrategy


class HybridMode(Enum):
    """Hybrid strategy modes with different TA/FA weightings."""

    DAY_TRADING = "day_trading"  # 70% TA, 30% FA
    SWING_TRADING = "swing_trading"  # 50% TA, 50% FA
    POSITION_TRADING = "position_trading"  # 40% TA, 60% FA
    ADAPTIVE = "adaptive"  # Dynamic weighting based on market conditions


class SignalStrength(Enum):
    """Signal strength classifications."""

    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class HybridSignal(Signal):
    """Enhanced signal with hybrid analysis details."""

    technical_score: float = 0.0
    fundamental_score: float = 0.0
    combined_score: float = 0.0
    ta_weight: float = 0.5
    fa_weight: float = 0.5
    signal_strength: SignalStrength = SignalStrength.MODERATE
    regime_adjusted: bool = False
    strategy_components: Dict[str, Any] = {}


class HybridStrategy(BaseStrategy):
    """
    Hybrid strategy combining technical and fundamental analysis.

    This strategy intelligently weighs technical and fundamental signals
    based on market conditions, trading timeframe, and signal reliability.
    """

    def __init__(
        self, config: StrategyConfig, hybrid_mode: HybridMode = HybridMode.SWING_TRADING
    ):
        """
        Initialize hybrid strategy.

        Args:
            config: Strategy configuration
            hybrid_mode: Hybrid strategy mode determining TA/FA weighting
        """
        super().__init__(config)
        self.hybrid_mode = hybrid_mode

        # Initialize component strategies
        self.technical_strategy: Optional[TechnicalStrategy] = None
        self.fundamental_strategy: Optional[FundamentalStrategy] = None

        # Analysis engines
        self.technical_engine = TechnicalAnalysisEngine()
        self.fundamental_engine = FundamentalAnalysisEngine()
        self.regime_manager = RegimeAwareStrategyManager()

        # Multi-timeframe confirmation
        self.mtf_analyzer, self.mtf_enhancer = create_multi_timeframe_analyzer(
            config.mode
        )
        self.mtf_data_fetcher = create_multi_timeframe_fetcher()
        self.enable_mtf_confirmation = True

        # Default hybrid parameters
        self.default_params = {
            # Weighting parameters
            "ta_base_weight": self._get_default_ta_weight(),
            "fa_base_weight": self._get_default_fa_weight(),
            "adaptive_weighting": True,
            # Signal combination parameters
            "min_technical_score": 40.0,
            "min_fundamental_score": 40.0,
            "min_combined_confidence": 65.0,
            "signal_divergence_threshold": 30.0,  # Max difference between TA and FA
            # Market condition adjustments
            "trending_market_ta_boost": 0.2,
            "ranging_market_fa_boost": 0.15,
            "high_vol_confidence_penalty": 0.3,
            # Risk management
            "max_signal_strength_position": 0.25,  # Max position for very strong signals
            # Multi-timeframe parameters
            "enable_mtf_confirmation": True,
            "mtf_min_timeframes": 3,
            "mtf_confidence_boost": 10.0,  # Boost for strong MTF confirmation
            "mtf_confidence_penalty": 20.0,  # Penalty for weak MTF confirmation
            "min_signal_strength_position": 0.05,  # Min position for weak signals
            "regime_filter_enabled": True,
            # Performance optimization
            "enable_walk_forward": True,
            "rebalance_frequency_days": 30,
            "signal_decay_hours": 4,  # How long signals remain valid
        }

        # Merge with user parameters
        self.params = {**self.default_params, **(self.config.parameters or {})}

    def _get_default_ta_weight(self) -> float:
        """Get default technical analysis weight based on mode."""
        weight_map = {
            HybridMode.DAY_TRADING: 0.70,
            HybridMode.SWING_TRADING: 0.50,
            HybridMode.POSITION_TRADING: 0.40,
            HybridMode.ADAPTIVE: 0.50,
        }
        return weight_map.get(self.hybrid_mode, 0.50)

    def _get_default_fa_weight(self) -> float:
        """Get default fundamental analysis weight based on mode."""
        return 1.0 - self._get_default_ta_weight()

    def _setup_indicators(self) -> None:
        """Setup component strategies and analysis engines."""
        try:
            self.logger.info(f"Setting up hybrid strategy components for {self.name}")

            # Create technical strategy configuration
            tech_config = StrategyConfig(
                name=f"{self.name}_technical",
                mode=self.config.mode,
                lookback_period=self.config.lookback_period,
                min_confidence=self.params.get("min_technical_score", 40.0),
                max_position_size=self.config.max_position_size,
                parameters=self._extract_technical_params(),
            )

            # Create fundamental strategy configuration
            fund_config = StrategyConfig(
                name=f"{self.name}_fundamental",
                mode=self.config.mode,
                lookback_period=max(
                    60, self.config.lookback_period
                ),  # FA needs more data
                min_confidence=self.params.get("min_fundamental_score", 40.0),
                max_position_size=self.config.max_position_size,
                parameters=self._extract_fundamental_params(),
            )

            # Initialize component strategies
            self.technical_strategy = TechnicalStrategy(tech_config)
            self.fundamental_strategy = FundamentalStrategy(fund_config)

            # Initialize components
            self.technical_strategy.initialize()
            self.fundamental_strategy.initialize()

            self.logger.info(f"Hybrid strategy {self.name} setup complete")

        except Exception as e:
            self.logger.error(f"Error setting up hybrid strategy: {e}")
            raise

    def _extract_technical_params(self) -> Dict[str, Any]:
        """Extract technical analysis parameters from hybrid config."""
        tech_params = {}

        # Filter parameters that belong to technical analysis
        tech_prefixes = [
            "sma_",
            "ema_",
            "rsi_",
            "macd_",
            "bb_",
            "atr_",
            "volume_",
            "momentum_",
        ]

        for key, value in self.params.items():
            if any(key.startswith(prefix) for prefix in tech_prefixes):
                tech_params[key] = value

        return tech_params

    def _extract_fundamental_params(self) -> Dict[str, Any]:
        """Extract fundamental analysis parameters from hybrid config."""
        fund_params = {}

        # Filter parameters that belong to fundamental analysis
        fund_prefixes = [
            "min_market_cap",
            "max_pe_",
            "min_roe",
            "debt_",
            "growth_",
            "valuation_",
            "health_",
        ]

        for key, value in self.params.items():
            if any(key.startswith(prefix) for prefix in fund_prefixes):
                fund_params[key] = value

        return fund_params

    async def analyze(
        self, symbol: str, data: pl.DataFrame, finviz_data: Optional[FinVizData] = None
    ) -> HybridSignal:
        """
        Perform hybrid analysis combining technical and fundamental analysis.

        Args:
            symbol: Trading symbol
            data: Historical market data
            finviz_data: Fundamental data from FinViz

        Returns:
            Hybrid trading signal with detailed analysis
        """
        try:
            if not self.validate_data(data):
                return HybridSignal(
                    action=SignalType.HOLD,
                    confidence=0.0,
                    position_size=0.0,
                    reasoning="Invalid or insufficient data",
                )

            # Perform parallel analysis
            technical_task = self._perform_technical_analysis(symbol, data)
            fundamental_task = self._perform_fundamental_analysis(
                symbol, data, finviz_data
            )

            # Execute analyses
            results = await asyncio.gather(
                technical_task, fundamental_task, return_exceptions=True
            )

            technical_result = results[0]
            fundamental_result = results[1]

            # Handle analysis errors
            if isinstance(technical_result, Exception):
                self.logger.error(f"Technical analysis error: {technical_result}")
                technical_result = self._default_technical_result()

            # Ensure technical_result is a dict
            if not isinstance(technical_result, dict):
                technical_result = self._default_technical_result()

            if isinstance(fundamental_result, Exception):
                self.logger.error(f"Fundamental analysis error: {fundamental_result}")
                fundamental_result = self._default_fundamental_result()

            # Ensure fundamental_result is a dict
            if not isinstance(fundamental_result, dict):
                fundamental_result = self._default_fundamental_result()

            # Ensure results are dictionaries
            if not isinstance(technical_result, dict):
                technical_result = self._default_technical_result()

            if not isinstance(fundamental_result, dict):
                fundamental_result = self._default_fundamental_result()

            # Determine optimal weights based on market conditions
            ta_weight, fa_weight = await self._calculate_adaptive_weights(
                symbol, data, technical_result, fundamental_result
            )

            # Combine signals
            hybrid_signal = await self._combine_signals(
                symbol, data, technical_result, fundamental_result, ta_weight, fa_weight
            )

            # Apply regime-based adjustments
            if self.params.get("regime_filter_enabled", True):
                hybrid_signal, regime_info = (
                    await self.regime_manager.get_regime_adjusted_signal(
                        hybrid_signal, data, "balanced"
                    )
                )
                hybrid_signal.regime_adjusted = True
                hybrid_signal.metadata.update({"regime_info": regime_info})

            # Apply multi-timeframe confirmation if enabled
            if (
                self.params.get("enable_mtf_confirmation", True)
                and self.enable_mtf_confirmation
            ):
                hybrid_signal = await self._apply_multi_timeframe_confirmation(
                    symbol, hybrid_signal, data
                )

            return hybrid_signal

        except Exception as e:
            self.logger.error(f"Error in hybrid analysis for {symbol}: {e}")
            return HybridSignal(
                action=SignalType.HOLD,
                confidence=50.0,
                position_size=0.0,
                reasoning=f"Analysis error: {str(e)}",
            )

    async def _apply_multi_timeframe_confirmation(
        self, symbol: str, primary_signal: HybridSignal, primary_data: pl.DataFrame
    ) -> HybridSignal:
        """
        Apply multi-timeframe confirmation to enhance signal reliability.

        Args:
            symbol: Trading symbol
            primary_signal: Primary hybrid signal to confirm
            primary_data: Primary timeframe data

        Returns:
            Enhanced signal with multi-timeframe confirmation
        """
        try:
            # Only apply MTF confirmation for entry signals
            if primary_signal.action == SignalType.HOLD:
                return primary_signal

            self.logger.info(f"Applying multi-timeframe confirmation for {symbol}")

            # Fetch multi-timeframe data
            mtf_data_result = await self.mtf_data_fetcher.fetch_multi_timeframe_data(
                symbol=symbol,
                strategy_mode=self.config.mode,
                periods=min(self.config.lookback_period, 100),
            )

            # Check if we have sufficient timeframes
            min_timeframes = self.params.get("mtf_min_timeframes", 3)
            if len(mtf_data_result.available_timeframes) < min_timeframes:
                self.logger.warning(
                    f"Insufficient timeframes for MTF confirmation: "
                    f"{len(mtf_data_result.available_timeframes)} < {min_timeframes}"
                )
                # Return signal with reduced confidence
                primary_signal.confidence *= 0.8
                primary_signal.reasoning += " | MTF: Insufficient timeframe data"
                return primary_signal

            # Apply multi-timeframe confirmation
            enhanced_signal, mtf_confirmation = (
                await self.mtf_enhancer.enhance_signal_with_confirmation(
                    symbol=symbol,
                    primary_signal=primary_signal,
                    multi_timeframe_data=mtf_data_result.data,
                    strategy_instance=self,
                )
            )

            # Apply confidence adjustments based on MTF results
            enhanced_signal = self._apply_mtf_confidence_adjustments(
                enhanced_signal, mtf_confirmation
            )

            # Add MTF metadata to hybrid signal
            if isinstance(enhanced_signal, HybridSignal):
                enhanced_signal.metadata.update(
                    {
                        "mtf_confirmation": {
                            "applied": True,
                            "data_quality_score": mtf_data_result.data_quality_score,
                            "available_timeframes": mtf_data_result.available_timeframes,
                            "missing_timeframes": mtf_data_result.missing_timeframes,
                        }
                    }
                )
            else:
                # Convert to HybridSignal if needed
                enhanced_signal = self._convert_to_hybrid_signal(
                    enhanced_signal, primary_signal
                )

            self.logger.info(
                f"MTF confirmation for {symbol}: "
                f"{mtf_confirmation.confirmation_strength.value} "
                f"({enhanced_signal.confidence:.1f}% confidence)"
            )

            return enhanced_signal

        except Exception as e:
            self.logger.error(
                f"Error applying multi-timeframe confirmation for {symbol}: {e}"
            )
            # Return original signal with error note
            primary_signal.reasoning += f" | MTF Error: {str(e)}"
            return primary_signal

    def _apply_mtf_confidence_adjustments(
        self, signal: Signal, mtf_confirmation: MultiTimeFrameConfirmation
    ) -> Signal:
        """Apply confidence adjustments based on MTF confirmation results."""
        try:
            original_confidence = signal.confidence

            # Apply confidence boost for strong confirmations
            confidence_boost = self.params.get("mtf_confidence_boost", 10.0)
            confidence_penalty = self.params.get("mtf_confidence_penalty", 20.0)

            if mtf_confirmation.confirmation_strength.value in [
                "very_strong",
                "strong",
            ]:
                signal.confidence = min(signal.confidence + confidence_boost, 100.0)
            elif mtf_confirmation.confirmation_strength.value in ["weak", "very_weak"]:
                signal.confidence = max(signal.confidence - confidence_penalty, 0.0)

            # Log confidence adjustment
            if signal.confidence != original_confidence:
                self.logger.debug(
                    f"MTF confidence adjustment: {original_confidence:.1f}% -> {signal.confidence:.1f}% "
                    f"(strength: {mtf_confirmation.confirmation_strength.value})"
                )

            return signal

        except Exception as e:
            self.logger.error(f"Error applying MTF confidence adjustments: {e}")
            return signal

    def _convert_to_hybrid_signal(
        self, signal: Signal, original_hybrid: HybridSignal
    ) -> HybridSignal:
        """Convert a regular Signal to HybridSignal preserving hybrid-specific attributes."""
        try:
            return HybridSignal(
                action=signal.action,
                confidence=signal.confidence,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size=signal.position_size,
                reasoning=signal.reasoning,
                metadata=signal.metadata,
                timestamp=signal.timestamp,
                # Preserve hybrid-specific attributes
                technical_score=original_hybrid.technical_score,
                fundamental_score=original_hybrid.fundamental_score,
                combined_score=original_hybrid.combined_score,
                ta_weight=original_hybrid.ta_weight,
                fa_weight=original_hybrid.fa_weight,
                signal_strength=original_hybrid.signal_strength,
                regime_adjusted=original_hybrid.regime_adjusted,
                strategy_components=original_hybrid.strategy_components,
            )
        except Exception as e:
            self.logger.error(f"Error converting to HybridSignal: {e}")
            return original_hybrid

    async def _perform_technical_analysis(
        self, symbol: str, data: pl.DataFrame
    ) -> Dict[str, Any]:
        """Perform technical analysis component."""
        try:
            # Use technical strategy if available, otherwise use engine directly
            if self.technical_strategy:
                tech_signal = await self.technical_strategy.analyze(symbol, data)
                tech_score = tech_signal.confidence
            else:
                # Fallback to direct engine analysis
                tech_analysis = self.technical_engine.full_analysis(symbol, data)
                tech_score = tech_analysis.get("technical_score", 50.0)
                tech_signal = Signal(
                    action=(
                        SignalType.BUY
                        if tech_score > 60
                        else SignalType.SELL if tech_score < 40 else SignalType.HOLD
                    ),
                    confidence=tech_score,
                    position_size=0.1,
                    reasoning="Technical analysis",
                )

            return {
                "signal": tech_signal,
                "score": tech_score,
                "analysis": (
                    self.technical_engine.full_analysis(symbol, data)
                    if hasattr(self, "technical_engine")
                    else {}
                ),
                "component": "technical",
                "timestamp": datetime.now(timezone.utc),
            }

        except Exception as e:
            self.logger.error(f"Technical analysis error for {symbol}: {e}")
            return self._default_technical_result()

    async def _perform_fundamental_analysis(
        self, symbol: str, data: pl.DataFrame, finviz_data: Optional[FinVizData]
    ) -> Dict[str, Any]:
        """Perform fundamental analysis component."""
        try:
            if not finviz_data:
                return self._default_fundamental_result()

            # Use fundamental strategy if available
            if self.fundamental_strategy:
                fund_signal = await self.fundamental_strategy.analyze(
                    symbol, data, finviz_data
                )
                fund_score = fund_signal.confidence
            else:
                # Fallback to direct engine analysis
                fund_analysis = self.fundamental_engine.full_analysis(
                    symbol, finviz_data, data
                )
                fund_score = fund_analysis.get("composite_score", 50.0)
                fund_signal = Signal(
                    action=(
                        SignalType.BUY
                        if fund_score > 60
                        else SignalType.SELL if fund_score < 40 else SignalType.HOLD
                    ),
                    confidence=fund_score,
                    position_size=0.1,
                    reasoning="Fundamental analysis",
                )

            return {
                "signal": fund_signal,
                "score": fund_score,
                "analysis": self.fundamental_engine.full_analysis(
                    symbol, finviz_data, data
                ),
                "component": "fundamental",
                "timestamp": datetime.now(timezone.utc),
            }

        except Exception as e:
            self.logger.error(f"Fundamental analysis error for {symbol}: {e}")
            return self._default_fundamental_result()

    async def _calculate_adaptive_weights(
        self,
        symbol: str,
        data: pl.DataFrame,
        technical_result: Dict,
        fundamental_result: Dict,
    ) -> Tuple[float, float]:
        """
        Calculate adaptive weights for TA and FA based on market conditions.

        Args:
            symbol: Trading symbol
            data: Market data
            technical_result: Technical analysis results
            fundamental_result: Fundamental analysis results

        Returns:
            Tuple of (ta_weight, fa_weight)
        """
        try:
            # Start with base weights
            ta_weight = self.params["ta_base_weight"]
            fa_weight = self.params["fa_base_weight"]

            if not self.params.get("adaptive_weighting", True):
                return ta_weight, fa_weight

            # Market regime detection
            regime_detector = MarketRegimeDetector()
            regime_state: RegimeState = regime_detector.detect_regime(data)

            # Adjust weights based on market regime
            if regime_state.primary_regime in [
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
            ]:
                # Trending markets favor technical analysis
                ta_boost = self.params.get("trending_market_ta_boost", 0.2)
                ta_weight = min(0.9, ta_weight + ta_boost)
                fa_weight = 1.0 - ta_weight

            elif regime_state.primary_regime == MarketRegime.RANGING:
                # Ranging markets may favor fundamental analysis for stock picking
                fa_boost = self.params.get("ranging_market_fa_boost", 0.15)
                fa_weight = min(0.8, fa_weight + fa_boost)
                ta_weight = 1.0 - fa_weight

            # Volatility adjustments
            if regime_state.volatility_regime in [
                VolatilityRegime.EXTREME_HIGH,
                VolatilityRegime.HIGH,
            ]:
                # High volatility reduces fundamental analysis reliability
                ta_weight = min(0.85, ta_weight + 0.15)
                fa_weight = 1.0 - ta_weight

            # Signal quality adjustments
            tech_quality = self._assess_signal_quality(technical_result)
            fund_quality = self._assess_signal_quality(fundamental_result)

            # Boost weight of higher quality signals
            quality_diff = tech_quality - fund_quality
            if abs(quality_diff) > 0.2:
                if quality_diff > 0:  # Technical is better
                    ta_weight = min(0.85, ta_weight + 0.1)
                else:  # Fundamental is better
                    fa_weight = min(0.85, fa_weight + 0.1)

                # Renormalize
                total = ta_weight + fa_weight
                ta_weight /= total
                fa_weight /= total

            # Trading mode specific adjustments
            if self.config.mode == StrategyMode.DAY_TRADING:
                # Day trading heavily favors technical analysis
                ta_weight = max(0.65, ta_weight)
                fa_weight = 1.0 - ta_weight
            elif self.config.mode == StrategyMode.POSITION_TRADING:
                # Position trading favors fundamental analysis
                fa_weight = max(0.55, fa_weight)
                ta_weight = 1.0 - fa_weight

            self.logger.debug(
                f"Adaptive weights for {symbol}: TA={ta_weight:.2f}, FA={fa_weight:.2f}"
            )

            return ta_weight, fa_weight

        except Exception as e:
            self.logger.error(f"Error calculating adaptive weights: {e}")
            return self.params["ta_base_weight"], self.params["fa_base_weight"]

    def _assess_signal_quality(self, analysis_result: Dict) -> float:
        """Assess the quality of an analysis result."""
        try:
            signal = analysis_result.get("signal")
            if not signal:
                return 0.0

            # Base quality from confidence
            quality = signal.confidence / 100.0

            # Adjust based on available data
            analysis = analysis_result.get("analysis", {})
            if "error" in analysis:
                quality *= 0.3
            elif analysis.get("data_points", 0) < 50:
                quality *= 0.7
            elif analysis.get("metrics_available", 0) < 3:
                quality *= 0.8

            return max(0.0, min(1.0, quality))

        except Exception:
            return 0.5

    async def _combine_signals(
        self,
        symbol: str,
        data: pl.DataFrame,
        technical_result: Dict,
        fundamental_result: Dict,
        ta_weight: float,
        fa_weight: float,
    ) -> HybridSignal:
        """
        Combine technical and fundamental signals into hybrid signal.

        Args:
            symbol: Trading symbol
            data: Market data
            technical_result: Technical analysis results
            fundamental_result: Fundamental analysis results
            ta_weight: Technical analysis weight
            fa_weight: Fundamental analysis weight

        Returns:
            Combined hybrid signal
        """
        try:
            tech_signal = technical_result.get("signal")
            fund_signal = fundamental_result.get("signal")

            tech_score = technical_result.get("score", 50.0)
            fund_score = fundamental_result.get("score", 50.0)

            # Check minimum score thresholds
            tech_meets_min = tech_score >= self.params["min_technical_score"]
            fund_meets_min = fund_score >= self.params["min_fundamental_score"]

            if not tech_meets_min and not fund_meets_min:
                return HybridSignal(
                    action=SignalType.HOLD,
                    confidence=0.0,
                    position_size=0.0,
                    technical_score=tech_score,
                    fundamental_score=fund_score,
                    ta_weight=ta_weight,
                    fa_weight=fa_weight,
                    reasoning="Neither technical nor fundamental signals meet minimum thresholds",
                )

            # Calculate weighted scores
            if tech_meets_min and fund_meets_min:
                # Both signals valid - use weighted combination
                combined_score = (tech_score * ta_weight) + (fund_score * fa_weight)

                # Check for signal divergence
                score_divergence = abs(tech_score - fund_score)
                if score_divergence > self.params["signal_divergence_threshold"]:
                    # Reduce confidence when signals diverge significantly
                    divergence_penalty = min(0.3, score_divergence / 100.0)
                    combined_score *= 1.0 - divergence_penalty

            elif tech_meets_min:
                # Only technical signal valid
                combined_score = tech_score * 0.8  # Reduced confidence
                ta_weight = 1.0
                fa_weight = 0.0

            else:  # fund_meets_min
                # Only fundamental signal valid
                combined_score = fund_score * 0.8  # Reduced confidence
                ta_weight = 0.0
                fa_weight = 1.0

            # Determine action based on combined score and individual signals
            final_action, action_confidence = self._determine_final_action(
                tech_signal, fund_signal, combined_score, ta_weight, fa_weight
            )

            # Calculate position size based on signal strength
            signal_strength = self._classify_signal_strength(combined_score)
            position_size = self._calculate_signal_based_position_size(
                combined_score, signal_strength
            )

            # Build reasoning
            reasoning = self._build_hybrid_reasoning(
                technical_result,
                fundamental_result,
                ta_weight,
                fa_weight,
                signal_strength,
            )

            # Get current price
            current_price = float(data.select("close").tail(1).item())

            # Create hybrid signal
            hybrid_signal = HybridSignal(
                action=final_action,
                confidence=min(100.0, action_confidence),
                entry_price=Decimal(str(current_price)),
                position_size=position_size,
                reasoning=reasoning,
                technical_score=tech_score,
                fundamental_score=fund_score,
                combined_score=combined_score,
                ta_weight=ta_weight,
                fa_weight=fa_weight,
                signal_strength=signal_strength,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "strategy_type": f"hybrid_{self.hybrid_mode.value}",
                    "technical_analysis": technical_result,
                    "fundamental_analysis": fundamental_result,
                    "weight_adjustments": {
                        "original_ta_weight": self.params["ta_base_weight"],
                        "original_fa_weight": self.params["fa_base_weight"],
                        "final_ta_weight": ta_weight,
                        "final_fa_weight": fa_weight,
                    },
                    "signal_combination": {
                        "tech_meets_min": tech_meets_min,
                        "fund_meets_min": fund_meets_min,
                        "score_divergence": (
                            abs(tech_score - fund_score)
                            if tech_meets_min and fund_meets_min
                            else 0
                        ),
                    },
                },
            )

            return hybrid_signal

        except Exception as e:
            self.logger.error(f"Error combining signals for {symbol}: {e}")
            return HybridSignal(
                action=SignalType.HOLD,
                confidence=0.0,
                position_size=0.0,
                reasoning=f"Signal combination error: {str(e)}",
            )

    def _determine_final_action(
        self,
        tech_signal: Optional[Signal],
        fund_signal: Optional[Signal],
        combined_score: float,
        ta_weight: float,
        fa_weight: float,
    ) -> Tuple[SignalType, float]:
        """Determine final action and confidence from component signals."""
        try:
            # Extract actions if signals exist
            tech_action = tech_signal.action if tech_signal else SignalType.HOLD
            fund_action = fund_signal.action if fund_signal else SignalType.HOLD

            # Confidence threshold
            min_confidence = self.params["min_combined_confidence"]

            # Action determination logic
            if combined_score >= 70:
                # Strong combined signal
                if tech_action == fund_action and tech_action != SignalType.HOLD:
                    # Both agree on direction
                    final_action = tech_action
                    confidence = combined_score
                elif ta_weight > 0.6 and tech_action != SignalType.HOLD:
                    # Technical analysis dominates
                    final_action = tech_action
                    confidence = combined_score * 0.9
                elif fa_weight > 0.6 and fund_action != SignalType.HOLD:
                    # Fundamental analysis dominates
                    final_action = fund_action
                    confidence = combined_score * 0.9
                else:
                    # Conflicting signals
                    final_action = SignalType.HOLD  # type: ignore
                    confidence = 40.0

            elif combined_score >= min_confidence:
                # Moderate signal - require agreement or strong weight
                if tech_action == fund_action and tech_action != SignalType.HOLD:
                    final_action = tech_action
                    confidence = combined_score
                elif (
                    ta_weight >= 0.7
                    and tech_action != SignalType.HOLD
                    and tech_signal
                    and tech_signal.confidence >= 70
                ):
                    final_action = tech_action
                    confidence = combined_score * 0.85
                elif (
                    fa_weight >= 0.7
                    and fund_action != SignalType.HOLD
                    and fund_signal
                    and fund_signal.confidence >= 70
                ):
                    final_action = fund_action
                    confidence = combined_score * 0.85
                else:
                    final_action = SignalType.HOLD  # type: ignore
                    confidence = 30.0
            else:
                # Weak signal
                final_action = SignalType.HOLD  # type: ignore
                confidence = combined_score * 0.5

            return final_action, confidence

        except Exception as e:
            self.logger.error(f"Error determining final action: {e}")
            return SignalType.HOLD, 0.0

    def _classify_signal_strength(self, combined_score: float) -> SignalStrength:
        """Classify signal strength based on combined score."""
        if combined_score >= 85:
            return SignalStrength.VERY_STRONG
        elif combined_score >= 75:
            return SignalStrength.STRONG
        elif combined_score >= 60:
            return SignalStrength.MODERATE
        elif combined_score >= 45:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def _calculate_signal_based_position_size(
        self, combined_score: float, signal_strength: SignalStrength
    ) -> float:
        """Calculate position size based on signal strength."""
        try:
            base_size = self.config.max_position_size

            # Strength-based multipliers
            strength_multipliers = {
                SignalStrength.VERY_STRONG: 1.0,
                SignalStrength.STRONG: 0.8,
                SignalStrength.MODERATE: 0.6,
                SignalStrength.WEAK: 0.4,
                SignalStrength.VERY_WEAK: 0.2,
            }

            multiplier = strength_multipliers.get(signal_strength, 0.5)

            # Score-based fine-tuning
            score_factor = combined_score / 100.0
            final_multiplier = multiplier * score_factor

            # Apply limits
            max_size = self.params.get("max_signal_strength_position", 0.25)
            min_size = self.params.get("min_signal_strength_position", 0.05)

            position_size = base_size * final_multiplier
            position_size = max(min_size, min(max_size, position_size))

            return round(position_size, 4)

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.config.max_position_size * 0.5

    def _build_hybrid_reasoning(
        self,
        technical_result: Dict,
        fundamental_result: Dict,
        ta_weight: float,
        fa_weight: float,
        signal_strength: SignalStrength,
    ) -> str:
        """Build comprehensive reasoning for hybrid signal."""
        try:
            reasoning_parts = []

            # Signal strength
            reasoning_parts.append(
                f"{signal_strength.value.replace('_', ' ').title()} signal"
            )

            # Technical component
            if ta_weight > 0.1:
                tech_reasoning = technical_result.get("signal", {}).get("reasoning", "")
                if tech_reasoning:
                    reasoning_parts.append(
                        f"Technical ({ta_weight:.0%}): {tech_reasoning}"
                    )

            # Fundamental component
            if fa_weight > 0.1:
                fund_reasoning = fundamental_result.get("signal", {}).get(
                    "reasoning", ""
                )
                if fund_reasoning:
                    reasoning_parts.append(
                        f"Fundamental ({fa_weight:.0%}): {fund_reasoning}"
                    )

            # Market regime context
            current_regime = self.regime_manager.get_current_regime()
            if current_regime:
                reasoning_parts.append(f"Market: {current_regime.primary_regime.value}")

            return " | ".join(reasoning_parts) if reasoning_parts else "Hybrid analysis"

        except Exception as e:
            self.logger.error(f"Error building reasoning: {e}")
            return "Hybrid analysis completed"

    def _default_technical_result(self) -> Dict[str, Any]:
        """Return default technical result when analysis fails."""
        return {
            "signal": Signal(
                action=SignalType.HOLD,
                confidence=50.0,
                position_size=0.0,
                reasoning="Technical analysis unavailable",
            ),
            "score": 50.0,
            "analysis": {},
            "component": "technical",
            "timestamp": datetime.now(timezone.utc),
        }

    def _default_fundamental_result(self) -> Dict[str, Any]:
        """Return default fundamental result when analysis fails."""
        return {
            "signal": Signal(
                action=SignalType.HOLD,
                confidence=50.0,
                position_size=0.0,
                reasoning="Fundamental analysis unavailable",
            ),
            "score": 50.0,
            "analysis": {},
            "component": "fundamental",
            "timestamp": datetime.now(timezone.utc),
        }

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy component weights."""
        return {
            "technical_weight": self.params["ta_base_weight"],
            "fundamental_weight": self.params["fa_base_weight"],
            "adaptive_enabled": self.params.get("adaptive_weighting", True),
        }

    def update_hybrid_mode(self, new_mode: HybridMode) -> None:
        """
        Update hybrid strategy mode and adjust weights accordingly.

        Args:
            new_mode: New hybrid mode
        """
        try:
            old_mode = self.hybrid_mode
            self.hybrid_mode = new_mode

            # Update base weights
            self.params["ta_base_weight"] = self._get_default_ta_weight()
            self.params["fa_base_weight"] = self._get_default_fa_weight()

            self.logger.info(
                f"Updated hybrid mode from {old_mode.value} to {new_mode.value}"
            )

        except Exception as e:
            self.logger.error(f"Error updating hybrid mode: {e}")

    def get_signal_breakdown(self, signal: HybridSignal) -> Dict[str, Any]:
        """
        Get detailed breakdown of how hybrid signal was constructed.

        Args:
            signal: Hybrid signal to analyze

        Returns:
            Signal construction breakdown
        """
        try:
            return {
                "final_signal": {
                    "action": signal.action.value if signal.action else "HOLD",
                    "confidence": signal.confidence,
                    "position_size": signal.position_size,
                    "signal_strength": signal.signal_strength.value,
                },
                "component_scores": {
                    "technical_score": signal.technical_score,
                    "fundamental_score": signal.fundamental_score,
                    "combined_score": signal.combined_score,
                },
                "weights_applied": {
                    "technical_weight": signal.ta_weight,
                    "fundamental_weight": signal.fa_weight,
                },
                "regime_context": {
                    "regime_adjusted": signal.regime_adjusted,
                    "current_regime": self.regime_manager.get_regime_summary(),
                },
                "analysis_details": {
                    "technical_signals": signal.metadata.get("technical_analysis", {})
                    .get("signal", {})
                    .get("metadata", {}),
                    "fundamental_signals": signal.metadata.get(
                        "fundamental_analysis", {}
                    )
                    .get("signal", {})
                    .get("metadata", {}),
                    "weight_adjustments": signal.metadata.get("weight_adjustments", {}),
                    "signal_combination": signal.metadata.get("signal_combination", {}),
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting signal breakdown: {e}")
            return {"error": str(e)}


class HybridSignalGenerator:
    """Generate formatted signals for the trading system."""

    def __init__(self) -> None:
        """Initialize signal generator."""
        self.logger = logging.getLogger("hybrid_signal_generator")

    def generate_formatted_signal(
        self,
        symbol: str,
        hybrid_signal: HybridSignal,
        strategy_type: str = "swing_trade",
    ) -> Dict[str, Any]:
        """
        Generate formatted signal output as specified.

        Args:
            symbol: Trading symbol
            hybrid_signal: Hybrid signal from analysis
            strategy_type: Strategy type for output

        Returns:
            Formatted signal dictionary
        """
        try:
            # Calculate stop loss and take profit if not set
            entry_price = (
                float(hybrid_signal.entry_price) if hybrid_signal.entry_price else 0.0
            )

            if not hybrid_signal.stop_loss and entry_price > 0:
                if hybrid_signal.action == SignalType.BUY:
                    stop_loss = entry_price * (1 - 0.02)  # 2% default
                else:
                    stop_loss = entry_price * (1 + 0.02)
            else:
                stop_loss = (
                    float(hybrid_signal.stop_loss)
                    if hybrid_signal.stop_loss
                    else entry_price * 0.98
                )

            if not hybrid_signal.take_profit and entry_price > 0:
                if hybrid_signal.action == SignalType.BUY:
                    take_profit = entry_price * (1 + 0.02)  # 2% default
                else:
                    take_profit = entry_price * (1 - 0.02)
            else:
                take_profit = (
                    float(hybrid_signal.take_profit)
                    if hybrid_signal.take_profit
                    else entry_price * 1.02
                )

            formatted_signal = {
                "ticker": symbol,
                "action": hybrid_signal.action.value.upper(),
                "confidence": int(hybrid_signal.confidence),
                "strategy_type": strategy_type,
                "entry_price": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "position_size": round(hybrid_signal.position_size, 3),
                "reasoning": hybrid_signal.reasoning,
                "timestamp": hybrid_signal.timestamp.isoformat() + "Z",
            }

            return formatted_signal

        except Exception as e:
            self.logger.error(f"Error generating formatted signal for {symbol}: {e}")
            return {
                "ticker": symbol,
                "action": "HOLD",
                "confidence": 0,
                "strategy_type": strategy_type,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "position_size": 0.0,
                "reasoning": f"Signal generation error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }


class HybridStrategyFactory:
    """Factory for creating different types of hybrid strategies."""

    @staticmethod
    def create_day_trading_strategy(name: str = "hybrid_day_trader") -> HybridStrategy:
        """Create a day trading focused hybrid strategy."""
        config = StrategyConfig(
            name=name,
            mode=StrategyMode.DAY_TRADING,
            lookback_period=30,
            min_confidence=70.0,
            max_position_size=0.15,
            default_stop_loss_pct=0.015,  # 1.5% stop loss
            default_take_profit_pct=0.02,  # 2% take profit
            parameters={
                "ta_base_weight": 0.70,
                "fa_base_weight": 0.30,
                "min_technical_score": 60.0,
                "min_fundamental_score": 30.0,
                "signal_decay_hours": 2,
                "volume_threshold": 2.0,
            },
        )
        return HybridStrategy(config, HybridMode.DAY_TRADING)

    @staticmethod
    def create_swing_trading_strategy(
        name: str = "hybrid_swing_trader",
    ) -> HybridStrategy:
        """Create a swing trading focused hybrid strategy."""
        config = StrategyConfig(
            name=name,
            mode=StrategyMode.SWING_TRADING,
            lookback_period=50,
            min_confidence=65.0,
            max_position_size=0.20,
            default_stop_loss_pct=0.03,  # 3% stop loss
            default_take_profit_pct=0.06,  # 6% take profit
            parameters={
                "ta_base_weight": 0.50,
                "fa_base_weight": 0.50,
                "min_technical_score": 50.0,
                "min_fundamental_score": 50.0,
                "signal_decay_hours": 24,
                "volume_threshold": 1.5,
            },
        )
        return HybridStrategy(config, HybridMode.SWING_TRADING)

    @staticmethod
    def create_position_trading_strategy(
        name: str = "hybrid_position_trader",
    ) -> HybridStrategy:
        """Create a position trading focused hybrid strategy."""
        config = StrategyConfig(
            name=name,
            mode=StrategyMode.POSITION_TRADING,
            lookback_period=100,
            min_confidence=60.0,
            max_position_size=0.25,
            default_stop_loss_pct=0.05,  # 5% stop loss
            default_take_profit_pct=0.15,  # 15% take profit
            parameters={
                "ta_base_weight": 0.40,
                "fa_base_weight": 0.60,
                "min_technical_score": 40.0,
                "min_fundamental_score": 60.0,
                "signal_decay_hours": 168,  # 7 days
                "volume_threshold": 1.2,
            },
        )
        return HybridStrategy(config, HybridMode.POSITION_TRADING)

    @staticmethod
    def create_adaptive_strategy(name: str = "hybrid_adaptive") -> HybridStrategy:
        """Create an adaptive hybrid strategy that adjusts to market conditions."""
        config = StrategyConfig(
            name=name,
            mode=StrategyMode.SWING_TRADING,  # Default mode
            lookback_period=75,
            min_confidence=60.0,
            max_position_size=0.20,
            default_stop_loss_pct=0.025,
            default_take_profit_pct=0.05,
            parameters={
                "ta_base_weight": 0.50,
                "fa_base_weight": 0.50,
                "adaptive_weighting": True,
                "regime_filter_enabled": True,
                "min_technical_score": 45.0,
                "min_fundamental_score": 45.0,
                "signal_decay_hours": 12,
                "volume_threshold": 1.8,
                "trending_market_ta_boost": 0.25,
                "ranging_market_fa_boost": 0.20,
            },
        )
        return HybridStrategy(config, HybridMode.ADAPTIVE)


class HybridStrategyValidator:
    """Validate hybrid strategy signals and configurations."""

    def __init__(self) -> None:
        """Initialize strategy validator."""
        self.logger = logging.getLogger("hybrid_validator")

    def validate_signal(self, signal: HybridSignal) -> Dict[str, Any]:
        """
        Validate hybrid signal for consistency and safety.

        Args:
            signal: Hybrid signal to validate

        Returns:
            Validation results
        """
        try:
            validation_errors = []
            validation_warnings = []

            # Basic signal validation
            if not signal.action:
                validation_errors.append("Missing signal action")

            if signal.confidence < 0 or signal.confidence > 100:
                validation_errors.append(f"Invalid confidence: {signal.confidence}")

            if signal.position_size < 0 or signal.position_size > 1:
                validation_errors.append(
                    f"Invalid position size: {signal.position_size}"
                )

            # Technical vs Fundamental score validation
            score_diff = abs(signal.technical_score - signal.fundamental_score)
            if score_diff > 40:
                validation_warnings.append(
                    f"Large divergence between technical ({signal.technical_score:.1f}) "
                    f"and fundamental ({signal.fundamental_score:.1f}) scores"
                )

            # Weight validation
            weight_sum = signal.ta_weight + signal.fa_weight
            if abs(weight_sum - 1.0) > 0.01:
                validation_errors.append(f"Weights don't sum to 1.0: {weight_sum:.3f}")

            # Price validation
            if signal.entry_price and signal.entry_price <= 0:
                validation_errors.append("Invalid entry price")

            if (
                signal.stop_loss
                and signal.take_profit
                and signal.entry_price
                and signal.action == SignalType.BUY
            ):
                if signal.stop_loss >= signal.entry_price:
                    validation_errors.append(
                        "Stop loss above entry price for BUY signal"
                    )
                if signal.take_profit <= signal.entry_price:
                    validation_errors.append(
                        "Take profit below entry price for BUY signal"
                    )

            # Signal strength consistency
            expected_strength = self._expected_signal_strength(signal.combined_score)
            if signal.signal_strength != expected_strength:
                validation_warnings.append(
                    f"Signal strength mismatch: expected {expected_strength.value}, "
                    f"got {signal.signal_strength.value}"
                )

            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "warnings": validation_warnings,
                "error_count": len(validation_errors),
                "warning_count": len(validation_warnings),
            }

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
            }

    def _expected_signal_strength(self, combined_score: float) -> SignalStrength:
        """Get expected signal strength for a given score."""
        if combined_score >= 85:
            return SignalStrength.VERY_STRONG
        elif combined_score >= 75:
            return SignalStrength.STRONG
        elif combined_score >= 60:
            return SignalStrength.MODERATE
        elif combined_score >= 45:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def validate_strategy_config(self, strategy: HybridStrategy) -> Dict[str, Any]:
        """
        Validate hybrid strategy configuration.

        Args:
            strategy: Hybrid strategy to validate

        Returns:
            Configuration validation results
        """
        try:
            config_errors = []
            config_warnings = []

            # Weight validation
            ta_weight = strategy.params.get("ta_base_weight", 0.5)
            fa_weight = strategy.params.get("fa_base_weight", 0.5)

            if abs(ta_weight + fa_weight - 1.0) > 0.01:
                config_errors.append(
                    f"Base weights don't sum to 1.0: TA={ta_weight}, FA={fa_weight}"
                )

            # Threshold validation
            min_tech = strategy.params.get("min_technical_score", 0)
            min_fund = strategy.params.get("min_fundamental_score", 0)

            if min_tech > 80 and min_fund > 80:
                config_warnings.append(
                    "Both minimum scores are very high - may miss opportunities"
                )

            # Risk parameter validation
            max_pos = strategy.config.max_position_size
            if max_pos > 0.5:
                config_warnings.append(f"High maximum position size: {max_pos:.1%}")

            stop_loss = strategy.config.default_stop_loss_pct
            take_profit = strategy.config.default_take_profit_pct

            if stop_loss > take_profit:
                config_errors.append(
                    "Stop loss percentage exceeds take profit percentage"
                )

            risk_reward = take_profit / stop_loss if stop_loss > 0 else 0
            if risk_reward < 1.0:
                config_warnings.append(f"Poor risk/reward ratio: {risk_reward:.2f}")

            return {
                "valid": len(config_errors) == 0,
                "errors": config_errors,
                "warnings": config_warnings,
                "config_summary": {
                    "ta_weight": ta_weight,
                    "fa_weight": fa_weight,
                    "min_technical_score": min_tech,
                    "min_fundamental_score": min_fund,
                    "max_position_size": max_pos,
                    "risk_reward_ratio": risk_reward,
                },
            }

        except Exception as e:
            self.logger.error(f"Error validating strategy config: {e}")
            return {
                "valid": False,
                "errors": [f"Config validation error: {str(e)}"],
                "warnings": [],
            }
