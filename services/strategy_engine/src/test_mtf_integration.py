"""
Integration Test for Multi-Timeframe Confirmation

This script tests the multi-timeframe confirmation system with the hybrid strategy
to ensure everything integrates properly and works as expected.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List

# Add shared path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import polars as pl
from base_strategy import StrategyConfig, StrategyMode
from hybrid_strategy import HybridMode, HybridStrategy
from models import SignalType
from multi_timeframe_analyzer import create_multi_timeframe_analyzer
from multi_timeframe_data import create_multi_timeframe_fetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(
    timeframe: str, periods: int = 100, symbol: str = "AAPL"
) -> pl.DataFrame:
    """Create sample market data for testing."""

    # Base price and parameters
    base_price = 150.0
    volatility = 0.02
    trend = 0.001  # Small upward trend

    # Generate timestamps based on timeframe
    timeframe_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    interval_minutes = timeframe_minutes.get(timeframe, 60)
    end_time = datetime.now(timezone.utc)
    timestamps = [
        end_time - timedelta(minutes=interval_minutes * i)
        for i in range(periods - 1, -1, -1)
    ]

    # Generate OHLCV data with some realistic patterns
    np.random.seed(42)  # For reproducible results

    data = []
    current_price = base_price

    for i, timestamp in enumerate(timestamps):
        # Add trend and random walk
        price_change = np.random.normal(trend, volatility)
        current_price = current_price * (1 + price_change)

        # Generate OHLC from current price
        daily_volatility = volatility * 2
        high = current_price * (1 + np.random.uniform(0, daily_volatility))
        low = current_price * (1 - np.random.uniform(0, daily_volatility))

        # Ensure OHLC relationships are valid
        open_price = current_price * (1 + np.random.normal(0, volatility * 0.5))
        open_price = max(min(open_price, high), low)
        close_price = current_price
        close_price = max(min(close_price, high), low)

        volume = np.random.randint(100000, 1000000)

        data.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": volume,
            }
        )

    return pl.DataFrame(data)


def create_bullish_data(timeframe: str, periods: int = 100) -> pl.DataFrame:
    """Create sample data with bullish pattern."""
    base_data = create_sample_data(timeframe, periods)

    # Add bullish trend by modifying close prices
    closes = base_data.select("close").to_numpy().flatten()

    # Apply stronger upward trend
    trend_factor = np.linspace(1.0, 1.15, len(closes))  # 15% increase over period
    bullish_closes = closes * trend_factor

    # Update the DataFrame with bullish closes
    return base_data.with_columns([pl.lit(bullish_closes).alias("close")])


def create_bearish_data(timeframe: str, periods: int = 100) -> pl.DataFrame:
    """Create sample data with bearish pattern."""
    base_data = create_sample_data(timeframe, periods)

    # Add bearish trend
    closes = base_data.select("close").to_numpy().flatten()
    trend_factor = np.linspace(1.0, 0.90, len(closes))  # 10% decrease over period
    bearish_closes = closes * trend_factor

    return base_data.with_columns([pl.lit(bearish_closes).alias("close")])


class MockDataFetcher:
    """Mock data fetcher for testing."""

    def __init__(self, data_scenario: str = "bullish"):
        self.data_scenario = data_scenario

    async def fetch_multi_timeframe_data(
        self, symbol: str, strategy_mode: StrategyMode, periods: int = 100
    ):
        """Mock fetch multi-timeframe data."""
        from multi_timeframe_data import TimeFrameDataResult

        # Define timeframes based on strategy mode
        timeframe_configs = {
            StrategyMode.DAY_TRADING: ["1m", "5m", "15m", "30m"],
            StrategyMode.SWING_TRADING: ["15m", "30m", "1h", "4h"],
            StrategyMode.POSITION_TRADING: ["1h", "4h", "1d"],
        }

        timeframes = timeframe_configs.get(strategy_mode, ["15m", "1h", "4h"])

        # Generate data based on scenario
        data_dict = {}
        for tf in timeframes:
            if self.data_scenario == "bullish":
                data_dict[tf] = create_bullish_data(tf, periods)
            elif self.data_scenario == "bearish":
                data_dict[tf] = create_bearish_data(tf, periods)
            else:
                data_dict[tf] = create_sample_data(tf, periods)

        return TimeFrameDataResult(
            symbol=symbol,
            data=data_dict,
            request_time=datetime.now(timezone.utc),
            available_timeframes=timeframes,
            missing_timeframes=[],
            data_quality_score=95.0,
            metadata={"test_scenario": self.data_scenario},
        )


async def test_multi_timeframe_confirmation():
    """Test multi-timeframe confirmation integration."""
    logger.info("Starting multi-timeframe confirmation test")

    try:
        # Create strategy configuration
        config = StrategyConfig(
            name="test_hybrid_mtf",
            mode=StrategyMode.SWING_TRADING,
            lookback_period=50,
            min_confidence=65.0,
            max_position_size=0.15,
            default_stop_loss_pct=0.03,
            default_take_profit_pct=0.06,
            parameters={
                "enable_mtf_confirmation": True,
                "mtf_min_timeframes": 3,
                "mtf_confidence_boost": 10.0,
                "mtf_confidence_penalty": 15.0,
            },
        )

        # Create hybrid strategy
        strategy = HybridStrategy(config, HybridMode.SWING_TRADING)

        # Replace data fetcher with mock for testing
        strategy.mtf_data_fetcher = MockDataFetcher("bullish")

        # Initialize strategy
        strategy.initialize()

        logger.info("Testing with bullish multi-timeframe data...")

        # Create primary timeframe data (bullish)
        primary_data = create_bullish_data("1h", 60)

        # Run analysis
        signal = await strategy.analyze(symbol="AAPL", data=primary_data)

        logger.info(f"Bullish test result:")
        logger.info(f"  Action: {signal.action}")
        logger.info(f"  Confidence: {signal.confidence:.1f}%")
        logger.info(f"  Position Size: {signal.position_size:.3f}")
        logger.info(f"  Reasoning: {signal.reasoning}")

        # Check if MTF confirmation was applied
        mtf_data = signal.metadata.get("multi_timeframe_confirmation", {})
        if mtf_data.get("applied"):
            logger.info("  Multi-timeframe confirmation: APPLIED")
            logger.info(
                f"  Data quality score: {mtf_data.get('data_quality_score', 'N/A')}"
            )
            logger.info(
                f"  Available timeframes: {mtf_data.get('available_timeframes', [])}"
            )
        else:
            logger.info("  Multi-timeframe confirmation: NOT APPLIED")

        # Test with bearish scenario
        logger.info("\nTesting with bearish multi-timeframe data...")

        strategy.mtf_data_fetcher = MockDataFetcher("bearish")
        bearish_data = create_bearish_data("1h", 60)

        bearish_signal = await strategy.analyze(symbol="AAPL", data=bearish_data)

        logger.info(f"Bearish test result:")
        logger.info(f"  Action: {bearish_signal.action}")
        logger.info(f"  Confidence: {bearish_signal.confidence:.1f}%")
        logger.info(f"  Position Size: {bearish_signal.position_size:.3f}")
        logger.info(f"  Reasoning: {bearish_signal.reasoning}")

        # Test with mixed/conflicting data
        logger.info("\nTesting with mixed multi-timeframe data...")

        strategy.mtf_data_fetcher = MockDataFetcher("mixed")
        mixed_data = create_sample_data("1h", 60)

        mixed_signal = await strategy.analyze(symbol="AAPL", data=mixed_data)

        logger.info(f"Mixed test result:")
        logger.info(f"  Action: {mixed_signal.action}")
        logger.info(f"  Confidence: {mixed_signal.confidence:.1f}%")
        logger.info(f"  Position Size: {mixed_signal.position_size:.3f}")
        logger.info(f"  Reasoning: {mixed_signal.reasoning}")

        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


async def test_analyzer_directly():
    """Test the multi-timeframe analyzer directly."""
    logger.info("\nTesting multi-timeframe analyzer directly...")

    try:
        # Create analyzer and enhancer
        analyzer, enhancer = create_multi_timeframe_analyzer(StrategyMode.SWING_TRADING)

        # Create mock multi-timeframe data
        multi_tf_data = {
            "15m": create_bullish_data("15m", 50),
            "1h": create_bullish_data("1h", 50),
            "4h": create_bullish_data("4h", 50),
        }

        # Create a dummy primary signal
        from base_strategy import Signal

        primary_signal = Signal(
            action=SignalType.BUY,
            confidence=70.0,
            position_size=0.15,
            reasoning="Test primary signal",
        )

        # Get multi-timeframe confirmation
        confirmation = await analyzer.confirm_signal(
            symbol="AAPL",
            primary_signal=primary_signal,
            multi_timeframe_data=multi_tf_data,
        )

        logger.info(f"Direct analyzer test results:")
        logger.info(f"  Primary signal: {confirmation.primary_signal}")
        logger.info(
            f"  Confirmation strength: {confirmation.confirmation_strength.value}"
        )
        logger.info(f"  Alignment type: {confirmation.alignment_type.value}")
        logger.info(f"  Overall confidence: {confirmation.overall_confidence:.1f}%")
        logger.info(f"  Supporting timeframes: {confirmation.supporting_timeframes}")
        logger.info(f"  Conflicting timeframes: {confirmation.conflicting_timeframes}")

        if confirmation.key_confluence_factors:
            logger.info(f"  Confluence factors: {confirmation.key_confluence_factors}")

        if confirmation.risk_factors:
            logger.info(f"  Risk factors: {confirmation.risk_factors}")

        # Test entry decision
        allow_entry, reason = analyzer.should_allow_entry(confirmation)
        logger.info(f"  Allow entry: {allow_entry}")
        logger.info(f"  Reason: {reason}")

        return True

    except Exception as e:
        logger.error(f"Direct analyzer test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


async def run_comprehensive_test():
    """Run comprehensive test of multi-timeframe confirmation system."""
    logger.info("=" * 60)
    logger.info("Multi-Timeframe Confirmation Integration Test")
    logger.info("=" * 60)

    results = []

    # Test 1: Direct analyzer test
    logger.info("\n1. Testing Multi-Timeframe Analyzer Directly")
    logger.info("-" * 45)
    result1 = await test_analyzer_directly()
    results.append(("Direct Analyzer Test", result1))

    # Test 2: Hybrid strategy integration test
    logger.info("\n2. Testing Hybrid Strategy Integration")
    logger.info("-" * 40)
    result2 = await test_multi_timeframe_confirmation()
    results.append(("Hybrid Strategy Integration", result2))

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {test_name}: {status}")

    overall_success = all(result[1] for result in results)
    logger.info(f"\nOverall Result: {'SUCCESS' if overall_success else 'FAILURE'}")

    return overall_success


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)
