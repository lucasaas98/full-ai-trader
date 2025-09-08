"""
Multi-Timeframe Strategy Example

This example demonstrates how to create a strategy that uses multiple timeframes
for analysis, leveraging the new timeframe mapping and multi-timeframe data support.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

# Add the strategy engine to path
sys.path.append(
    str(Path(__file__).parent.parent.parent / "services/strategy_engine/src")
)

# Import from added path (flake8: noqa)
from base_strategy import (  # noqa: E402
    BaseStrategy,
    Signal,
    StrategyConfig,
    StrategyMode,
    TimeFrameMapper,
)

from shared.models import SignalType  # noqa: E402


class MultiTimeframeMovingAverageStrategy(BaseStrategy):
    """
    Example strategy that uses multiple timeframes for trend confirmation.

    - Uses 1h timeframe for primary trend
    - Uses 15min timeframe for entry timing
    - Uses 1d timeframe for overall market direction
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # Strategy uses multiple timeframes
        self.required_timeframes = ["15m", "1h", "1d"]

        # Parameters for moving averages
        self.fast_ma_period = self.config.parameters.get("fast_ma_period", 20)
        self.slow_ma_period = self.config.parameters.get("slow_ma_period", 50)
        self.trend_ma_period = self.config.parameters.get("trend_ma_period", 200)

    def get_required_timeframes(self) -> list[str]:
        """Override to return our custom timeframes."""
        return self.required_timeframes

    async def analyze(self, symbol: str, data: pl.DataFrame) -> Signal:
        """
        Single timeframe analysis (fallback).

        This method is called when only single timeframe data is available.
        """
        try:
            if data.height < self.slow_ma_period:
                return Signal(
                    action=SignalType.HOLD,
                    confidence=0.0,
                    symbol=symbol,
                    strategy=self.name,
                    metadata={"reason": "Insufficient data for analysis"},
                )

            # Simple moving average crossover on single timeframe
            fast_ma = data.select(
                [pl.col("close").rolling_mean(self.fast_ma_period).alias("fast_ma")]
            )["fast_ma"].to_list()[-1]

            slow_ma = data.select(
                [pl.col("close").rolling_mean(self.slow_ma_period).alias("slow_ma")]
            )["slow_ma"].to_list()[-1]

            current_price = data["close"].to_list()[-1]

            # Generate signal
            if fast_ma > slow_ma:
                action = SignalType.BUY
                confidence = min(80.0, abs(fast_ma - slow_ma) / slow_ma * 1000)
            elif fast_ma < slow_ma:
                action = SignalType.SELL
                confidence = min(80.0, abs(fast_ma - slow_ma) / slow_ma * 1000)
            else:
                action = SignalType.HOLD
                confidence = 0.0

            return Signal(
                action=action,
                confidence=confidence,
                symbol=symbol,
                strategy=self.name,
                price=current_price,
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "method": "single_timeframe",
                },
            )

        except Exception as e:
            self.logger.error(f"Error in single timeframe analysis: {e}")
            return Signal(
                action=SignalType.HOLD,
                confidence=0.0,
                symbol=symbol,
                strategy=self.name,
                metadata={"error": str(e)},
            )

    async def analyze_multi_timeframe(
        self,
        symbol: str,
        multi_tf_data: Dict[str, pl.DataFrame],
        finviz_data: Optional[Any] = None,
    ) -> Signal:
        """
        Multi-timeframe analysis for enhanced signal generation.

        This method is called when multi-timeframe data is available.
        """
        try:
            self.logger.info(f"Running multi-timeframe analysis for {symbol}")
            self.logger.info(f"Available timeframes: {list(multi_tf_data.keys())}")

            # Map our strategy timeframes to data timeframes
            tf_mapping = {"15min": "15m", "1h": "1h", "1day": "1d"}

            signals = {}
            confidence_weights = {
                "1day": 0.4,  # Daily trend gets highest weight
                "1h": 0.4,  # Hourly trend for medium-term
                "15min": 0.2,  # 15min for entry timing
            }

            total_confidence = 0.0
            final_action = SignalType.HOLD
            metadata: Dict[str, Any] = {
                "timeframe_analysis": {},
                "method": "multi_timeframe",
            }

            # Analyze each timeframe
            for data_tf, strategy_tf in tf_mapping.items():
                if data_tf not in multi_tf_data:
                    self.logger.debug(f"No data for timeframe {data_tf}")
                    continue

                tf_data = multi_tf_data[data_tf]
                if tf_data.height < self.slow_ma_period:
                    self.logger.debug(f"Insufficient data for timeframe {data_tf}")
                    continue

                # Calculate moving averages for this timeframe
                fast_ma_list: List[Any] = tf_data.select(
                    [pl.col("close").rolling_mean(self.fast_ma_period).alias("fast_ma")]
                )["fast_ma"].to_list()
                fast_ma = fast_ma_list[-1]

                slow_ma_list: List[Any] = tf_data.select(
                    [pl.col("close").rolling_mean(self.slow_ma_period).alias("slow_ma")]
                )["slow_ma"].to_list()
                slow_ma = slow_ma_list[-1]

                # For daily timeframe, also check long-term trend
                trend_ma: Optional[Any] = None
                if data_tf == "1day" and tf_data.height >= self.trend_ma_period:
                    trend_ma_list: List[Any] = tf_data.select(
                        [
                            pl.col("close")
                            .rolling_mean(self.trend_ma_period)
                            .alias("trend_ma")
                        ]
                    )["trend_ma"].to_list()
                    trend_ma = trend_ma_list[-1]

                current_price_list: List[Any] = tf_data["close"].to_list()
                current_price = current_price_list[-1]

                # Determine signal for this timeframe
                if fast_ma > slow_ma:
                    tf_signal = SignalType.BUY
                    tf_strength = abs(fast_ma - slow_ma) / slow_ma * 100
                elif fast_ma < slow_ma:
                    tf_signal = SignalType.SELL
                    tf_strength = abs(fast_ma - slow_ma) / slow_ma * 100
                else:
                    tf_signal = SignalType.HOLD
                    tf_strength = 0.0

                # Apply trend filter for daily timeframe
                if data_tf == "1day" and trend_ma:
                    if current_price < trend_ma and tf_signal == SignalType.BUY:
                        tf_strength *= 0.5  # Reduce buy signal strength in downtrend
                    elif current_price > trend_ma and tf_signal == SignalType.SELL:
                        tf_strength *= 0.5  # Reduce sell signal strength in uptrend

                # Weight the signal by timeframe importance
                weighted_strength = tf_strength * confidence_weights[data_tf]
                total_confidence += weighted_strength

                # Store analysis for this timeframe
                tf_analysis: Dict[str, Any] = {
                    "signal": tf_signal.value,
                    "strength": tf_strength,
                    "weighted_strength": weighted_strength,
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "trend_ma": trend_ma,
                    "current_price": current_price,
                }
                timeframe_analysis = metadata.setdefault("timeframe_analysis", {})
                timeframe_analysis[strategy_tf] = tf_analysis

                signals[data_tf] = (tf_signal, weighted_strength)

            # Combine signals across timeframes
            if not signals:
                return Signal(
                    action=SignalType.HOLD,
                    confidence=0.0,
                    symbol=symbol,
                    strategy=self.name,
                    metadata={"reason": "No valid timeframe data for analysis"},
                )

            # Calculate net signal strength
            buy_strength = sum(
                strength
                for signal, strength in signals.values()
                if signal == SignalType.BUY
            )
            sell_strength = sum(
                strength
                for signal, strength in signals.values()
                if signal == SignalType.SELL
            )

            # Determine final action
            net_strength = buy_strength - sell_strength
            if abs(net_strength) < 1.0:  # Very weak signal
                final_action = SignalType.HOLD
                final_confidence = 0.0
            elif net_strength > 0:
                final_action = SignalType.BUY
                final_confidence = min(95.0, net_strength)
            else:
                final_action = SignalType.SELL
                final_confidence = min(95.0, abs(net_strength))

            # Add confidence threshold check
            if final_confidence < self.config.min_confidence:
                final_action = SignalType.HOLD
                metadata["reason"] = (
                    f"Confidence {final_confidence:.1f}% below threshold {self.config.min_confidence}%"
                )

            # Get current price from most recent timeframe
            final_price: Optional[Any] = None
            for tf_data in multi_tf_data.values():
                if not tf_data.is_empty():
                    price_list: List[Any] = tf_data["close"].to_list()
                    final_price = price_list[-1]
                    break

            additional_metadata: Dict[str, Any] = {
                "buy_strength": buy_strength,
                "sell_strength": sell_strength,
                "net_strength": net_strength,
                "timeframes_analyzed": len(signals),
                "total_confidence": total_confidence,
            }
            metadata.update(additional_metadata)

            return Signal(
                action=final_action,
                confidence=final_confidence,
                symbol=symbol,
                strategy=self.name,
                price=final_price,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {e}")
            return Signal(
                action=SignalType.HOLD,
                confidence=0.0,
                symbol=symbol,
                strategy=self.name,
                metadata={"error": str(e), "method": "multi_timeframe"},
            )


async def main():
    """
    Example usage of the multi-timeframe strategy.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create strategy configuration with custom timeframes
    config = StrategyConfig(
        name="multi_tf_ma_strategy",
        mode=StrategyMode.SWING_TRADING,
        lookback_period=100,
        min_confidence=65.0,
        custom_timeframes=["15m", "1h", "1d"],  # Use custom timeframes
        parameters={"fast_ma_period": 20, "slow_ma_period": 50, "trend_ma_period": 200},
    )

    # Create the strategy
    strategy = MultiTimeframeMovingAverageStrategy(config)
    strategy.initialize()

    print("=== Multi-Timeframe Strategy Example ===")
    print(f"Strategy: {strategy.name}")
    print(f"Required timeframes: {strategy.get_required_timeframes()}")
    print(f"Data timeframes: {strategy.get_required_data_timeframes()}")

    # Validate timeframes
    availability = strategy.validate_timeframe_availability()
    print(f"Timeframe availability: {availability}")

    # Show available timeframes in the system
    print(
        f"Available strategy timeframes: {TimeFrameMapper.get_available_strategy_timeframes()}"
    )
    print(
        f"Available data timeframes: {TimeFrameMapper.get_available_data_timeframes()}"
    )

    # Demonstrate timeframe mapping
    strategy_tfs = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    data_tfs = TimeFrameMapper.strategy_to_data(strategy_tfs)
    print("\nTimeframe mapping demo:")
    print(f"Strategy TFs: {strategy_tfs}")
    print(f"Data TFs: {data_tfs}")

    available, unavailable = TimeFrameMapper.validate_timeframes(strategy_tfs)
    print(f"Available: {available}")
    print(f"Unavailable: {unavailable}")


if __name__ == "__main__":
    asyncio.run(main())
