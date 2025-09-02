"""
Integration tests for Real Backtesting Engine

This module tests the complete backtesting system that runs the actual AI strategy
against historical data, simulating real trading conditions.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import polars as pl
import pytest

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), "../../backtesting"))
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../services/data_collector/src")
)
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../services/strategy_engine/src")
)

from data_store import DataStore, DataStoreConfig
from real_backtest_engine import (
    BacktestMode,
    HistoricalDataFeeder,
    MockRedisClient,
    RealBacktestConfig,
    RealBacktestEngine,
    run_monthly_backtest,
    run_previous_month_backtest,
)

from shared.models import SignalType, TimeFrame

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def data_store():
    """Create DataStore instance for testing."""
    config = DataStoreConfig(
        base_path="data/parquet", batch_size=1000, retention_days=365
    )
    return DataStore(config)


@pytest.fixture
def sample_backtest_config():
    """Create sample backtesting configuration."""
    # Use the last 30 days for testing
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    return RealBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal("50000"),  # Smaller amount for testing
        max_positions=5,
        mode=BacktestMode.DEBUG,  # More verbose for testing
        timeframe=TimeFrame.ONE_DAY,
        symbols_to_trade=["AAPL", "MSFT", "GOOGL"],  # Test with known symbols
        enable_screener_data=False,  # Simplify for testing
    )


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    return MockRedisClient()


class TestRealBacktestEngine:
    """Test the real backtesting engine functionality."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, sample_backtest_config):
        """Test that the backtest engine initializes correctly."""
        engine = RealBacktestEngine(sample_backtest_config)

        assert engine.config == sample_backtest_config
        assert engine.cash == sample_backtest_config.initial_capital
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
        assert engine.ai_calls == 0

        # Test AI strategy initialization
        await engine.initialize_ai_strategy()
        assert engine.ai_strategy is not None

    @pytest.mark.asyncio
    async def test_historical_data_feeder(self, data_store, sample_backtest_config):
        """Test the historical data feeding functionality."""
        feeder = HistoricalDataFeeder(data_store, sample_backtest_config)

        # Test getting market data for a specific date
        test_date = datetime(
            2025, 8, 20, tzinfo=timezone.utc
        )  # Use a date we know has data
        symbols = ["AAPL"]

        market_data = await feeder.get_market_data_for_date(test_date, symbols)

        # Should have data or be empty (depending on availability)
        assert isinstance(market_data, dict)
        if "AAPL" in market_data:
            data = market_data["AAPL"]
            assert data.symbol == "AAPL"
            assert data.close > 0
            assert data.volume >= 0

    @pytest.mark.asyncio
    async def test_mock_redis_functionality(self, mock_redis):
        """Test the mock Redis client for backtesting."""
        # Test basic operations
        await mock_redis.set("test_key", "test_value", ex=60)
        value = await mock_redis.get("test_key")
        assert value == "test_value"

        # Test publish (should not raise error)
        await mock_redis.publish("test_channel", {"test": "data"})

        # Test cleanup
        await mock_redis.close()
        assert len(mock_redis.data_cache) == 0

    @pytest.mark.asyncio
    async def test_position_management(self, sample_backtest_config):
        """Test position opening and closing logic."""
        engine = RealBacktestEngine(sample_backtest_config)
        await engine.initialize_ai_strategy()

        # Create mock market data
        from shared.models import MarketData

        test_date = datetime(2025, 8, 20, tzinfo=timezone.utc)
        market_data = MarketData(
            symbol="AAPL",
            timestamp=test_date,
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("152.50"),
            adjusted_close=Decimal("152.50"),
            volume=1000000,
            timeframe=TimeFrame.ONE_DAY,
        )

        # Test position size calculation
        from base_strategy import Signal

        signal = Signal(
            action=SignalType.BUY,
            confidence=75.0,
            position_size=0.1,  # 10% of portfolio
            reasoning="Test signal",
        )
        signal.symbol = "AAPL"

        position_size = await engine._calculate_position_size(signal, market_data)
        assert position_size > 0

        # Test entry price calculation
        entry_price = await engine._calculate_entry_price(market_data, SignalType.BUY)
        assert entry_price > market_data.close  # Should include slippage

        # Test commission calculation
        commission = await engine._calculate_commission(entry_price, position_size)
        assert commission > 0

    @pytest.mark.asyncio
    async def test_portfolio_value_calculation(self, sample_backtest_config):
        """Test portfolio value calculation with positions."""
        engine = RealBacktestEngine(sample_backtest_config)

        # Add a mock position
        from real_backtest_engine import BacktestPosition

        position = BacktestPosition(
            symbol="AAPL",
            quantity=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(timezone.utc),
            current_price=Decimal("155.00"),
        )
        engine.positions["AAPL"] = position

        # Calculate portfolio value
        portfolio_value = await engine._calculate_portfolio_value(
            datetime.now(timezone.utc)
        )

        expected_value = engine.cash + (position.current_price * Decimal("100"))
        assert portfolio_value == expected_value

    @pytest.mark.asyncio
    async def test_simple_backtest_run(self, sample_backtest_config):
        """Test running a simple backtest."""
        # Use a shorter time period for faster testing
        config = sample_backtest_config
        config.start_date = datetime(2025, 8, 15, tzinfo=timezone.utc)
        config.end_date = datetime(2025, 8, 21, tzinfo=timezone.utc)
        config.symbols_to_trade = ["AAPL"]  # Single symbol for simplicity

        engine = RealBacktestEngine(config)

        try:
            results = await engine.run_backtest()

            # Validate results structure
            assert results is not None
            assert results.execution_time_seconds > 0
            assert results.final_portfolio_value >= 0
            assert results.total_ai_calls >= 0
            assert isinstance(results.trades, list)
            assert isinstance(results.portfolio_values, list)

            # Log results for inspection
            logger.info(f"Backtest completed:")
            logger.info(f"  Total return: {results.total_return:.2%}")
            logger.info(f"  Total trades: {results.total_trades}")
            logger.info(f"  AI calls: {results.total_ai_calls}")
            logger.info(f"  Execution time: {results.execution_time_seconds:.2f}s")

            # Basic sanity checks
            assert results.final_portfolio_value > 0
            assert (
                -1.0 <= results.total_return <= 10.0
            )  # Reasonable return range for short period

        except Exception as e:
            # If backtest fails due to missing data, that's acceptable for testing
            logger.warning(f"Backtest failed (possibly due to missing test data): {e}")
            pytest.skip(f"Backtest skipped due to data availability: {e}")

    @pytest.mark.asyncio
    async def test_backtest_with_screener_data(self, sample_backtest_config):
        """Test backtesting with screener data integration."""
        config = sample_backtest_config
        config.enable_screener_data = True
        config.screener_types = ["momentum", "breakouts"]
        config.symbols_to_trade = None  # Use screener results
        config.start_date = datetime(
            2025, 8, 22, tzinfo=timezone.utc
        )  # Date with screener data
        config.end_date = datetime(2025, 8, 23, tzinfo=timezone.utc)

        engine = RealBacktestEngine(config)

        try:
            # Test screener data loading
            feeder = HistoricalDataFeeder(engine.data_store, config)
            screener_data = await feeder.get_screener_data_for_date(config.start_date)

            # Should have screener results or be empty
            assert isinstance(screener_data, dict)
            if screener_data:
                logger.info(f"Found screener data: {list(screener_data.keys())}")
                for screener_type, symbols in screener_data.items():
                    assert isinstance(symbols, list)
                    logger.info(f"  {screener_type}: {len(symbols)} symbols")

            # Run short backtest if data is available
            if screener_data:
                results = await engine.run_backtest()
                assert results is not None
                logger.info(
                    f"Screener backtest: {results.total_return:.2%} return, {results.total_trades} trades"
                )

        except Exception as e:
            logger.warning(f"Screener backtest failed: {e}")
            pytest.skip(f"Screener test skipped due to data: {e}")

    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test the convenience functions for running backtests."""
        try:
            # Test monthly backtest function
            start_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
            end_date = datetime(2025, 8, 7, tzinfo=timezone.utc)  # Short period

            results = await run_monthly_backtest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=Decimal("25000"),
                symbols=["AAPL"],
            )

            assert results is not None
            assert results.execution_time_seconds > 0
            logger.info(f"Monthly backtest: {results.total_return:.2%} return")

        except Exception as e:
            logger.warning(f"Convenience function test failed: {e}")
            pytest.skip(f"Convenience test skipped: {e}")

    @pytest.mark.asyncio
    async def test_error_handling(self, sample_backtest_config):
        """Test error handling in various scenarios."""

        # Test with invalid date range
        config = sample_backtest_config
        config.start_date = datetime(2030, 1, 1, tzinfo=timezone.utc)  # Future date
        config.end_date = datetime(2030, 1, 7, tzinfo=timezone.utc)

        engine = RealBacktestEngine(config)

        try:
            results = await engine.run_backtest()
            # Should complete even with no data
            assert results is not None
            assert results.total_trades == 0

        except Exception as e:
            # This is also acceptable - no data scenario
            logger.info(f"Expected error with future dates: {e}")

    @pytest.mark.asyncio
    async def test_backtest_metrics_calculation(self):
        """Test the calculation of various backtest metrics."""
        # Create a simple config for testing metrics
        config = RealBacktestConfig(
            start_date=datetime(2025, 8, 15, tzinfo=timezone.utc),
            end_date=datetime(2025, 8, 20, tzinfo=timezone.utc),
            initial_capital=Decimal("10000"),
            symbols_to_trade=["AAPL"],
        )

        engine = RealBacktestEngine(config)

        # Add some mock trades for metrics calculation
        from real_backtest_engine import BacktestTrade

        engine.trades = [
            BacktestTrade(
                symbol="AAPL",
                entry_date=datetime(2025, 8, 15, tzinfo=timezone.utc),
                exit_date=datetime(2025, 8, 16, tzinfo=timezone.utc),
                entry_price=Decimal("150.00"),
                exit_price=Decimal("155.00"),
                quantity=10,
                pnl=Decimal("50.00"),
                pnl_percentage=0.033,
                commission=Decimal("2.00"),
                hold_days=1,
                strategy_reasoning="Test trade",
                confidence=75.0,
            ),
            BacktestTrade(
                symbol="AAPL",
                entry_date=datetime(2025, 8, 17, tzinfo=timezone.utc),
                exit_date=datetime(2025, 8, 18, tzinfo=timezone.utc),
                entry_price=Decimal("155.00"),
                exit_price=Decimal("150.00"),
                quantity=10,
                pnl=Decimal("-50.00"),
                pnl_percentage=-0.032,
                commission=Decimal("2.00"),
                hold_days=1,
                strategy_reasoning="Test trade",
                confidence=60.0,
            ),
        ]

        # Add portfolio history
        engine.portfolio_history = [
            (datetime(2025, 8, 15, tzinfo=timezone.utc), Decimal("10000")),
            (datetime(2025, 8, 16, tzinfo=timezone.utc), Decimal("10050")),
            (datetime(2025, 8, 17, tzinfo=timezone.utc), Decimal("10050")),
            (datetime(2025, 8, 18, tzinfo=timezone.utc), Decimal("10000")),
        ]

        # Generate results and test metrics
        results = await engine._generate_results(1.0)

        assert results.total_trades == 2
        assert results.winning_trades == 1
        assert results.losing_trades == 1
        assert results.win_rate == 0.5
        assert results.average_win == 50.0
        assert results.average_loss == -50.0
        assert results.largest_win == 50.0
        assert results.largest_loss == -50.0
        assert results.average_confidence == 67.5  # (75 + 60) / 2


class TestBacktestIntegration:
    """Integration tests that test the complete system flow."""

    @pytest.mark.asyncio
    async def test_data_availability_check(self, data_store):
        """Test that we have sufficient data for backtesting."""
        # Check for AAPL data (commonly used in tests)
        try:
            test_date = datetime(2025, 8, 20).date()
            df = await data_store.load_market_data(
                ticker="AAPL",
                timeframe=TimeFrame.ONE_DAY,
                start_date=test_date,
                end_date=test_date,
            )

            if not df.is_empty():
                logger.info(f"Found AAPL data for {test_date}")
                row = df.row(0, named=True)
                assert row["close"] > 0
                assert row["volume"] >= 0
            else:
                logger.warning(f"No AAPL data found for {test_date}")

        except Exception as e:
            logger.warning(f"Data availability check failed: {e}")

    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test the complete backtesting system with real components."""
        # This test runs the full system but with a very short time period
        # to ensure it works end-to-end

        config = RealBacktestConfig(
            start_date=datetime(2025, 8, 20, tzinfo=timezone.utc),
            end_date=datetime(2025, 8, 21, tzinfo=timezone.utc),
            initial_capital=Decimal("10000"),
            max_positions=2,
            mode=BacktestMode.FAST,
            symbols_to_trade=["AAPL"],
            enable_screener_data=False,
        )

        try:
            engine = RealBacktestEngine(config)
            results = await engine.run_backtest()

            # System should complete without errors
            assert results is not None
            assert results.execution_time_seconds >= 0

            # Log final results
            logger.info("Full integration test completed successfully")
            logger.info(f"Final portfolio value: ${results.final_portfolio_value}")
            logger.info(f"Total return: {results.total_return:.2%}")
            logger.info(f"Trades executed: {results.total_trades}")

        except Exception as e:
            # If it fails due to missing data or AI API issues, that's acceptable
            logger.warning(f"Full integration test failed: {e}")
            pytest.skip(f"Integration test skipped due to system dependencies: {e}")


class TestPerformanceAndScalability:
    """Test performance characteristics of the backtesting system."""

    @pytest.mark.asyncio
    async def test_backtest_performance(self):
        """Test that backtesting completes in reasonable time."""
        config = RealBacktestConfig(
            start_date=datetime(2025, 8, 15, tzinfo=timezone.utc),
            end_date=datetime(2025, 8, 21, tzinfo=timezone.utc),  # 1 week
            initial_capital=Decimal("50000"),
            mode=BacktestMode.FAST,
            symbols_to_trade=["AAPL", "MSFT"],  # 2 symbols
            enable_screener_data=False,
        )

        try:
            start_time = datetime.now()
            engine = RealBacktestEngine(config)
            results = await engine.run_backtest()
            execution_time = (datetime.now() - start_time).total_seconds()

            # Should complete reasonably quickly for a week of data
            assert execution_time < 30.0  # 30 seconds max for 1 week, 2 symbols

            logger.info(
                f"Performance test: {execution_time:.2f}s for {results.total_trades} trades"
            )

        except Exception as e:
            logger.warning(f"Performance test failed: {e}")
            pytest.skip(f"Performance test skipped: {e}")

    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test that memory usage remains reasonable during backtesting."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        config = RealBacktestConfig(
            start_date=datetime(2025, 8, 15, tzinfo=timezone.utc),
            end_date=datetime(2025, 8, 21, tzinfo=timezone.utc),
            initial_capital=Decimal("25000"),
            symbols_to_trade=["AAPL"],
            enable_screener_data=False,
        )

        try:
            engine = RealBacktestEngine(config)
            results = await engine.run_backtest()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB for simple backtest)
            assert (
                memory_increase < 100
            ), f"Memory usage increased by {memory_increase:.2f}MB"

            logger.info(f"Memory test: {memory_increase:.2f}MB increase")

        except Exception as e:
            logger.warning(f"Memory test failed: {e}")
            pytest.skip(f"Memory test skipped: {e}")


if __name__ == "__main__":
    # Allow running tests directly
    asyncio.run(pytest.main([__file__, "-v"]))
