"""
Unit tests for technical indicators.

This module tests the technical analysis indicators used by trading strategies,
ensuring accuracy and proper handling of edge cases.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from services.strategy_engine.src.technical_analysis import TechnicalIndicators


class TestTechnicalIndicators:
    """Test technical analysis indicators."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        n_points = 100
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, n_points)  # Daily returns
        prices = [base_price]

        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        # Create OHLCV data
        data = []
        for i, close in enumerate(prices[1:]):
            open_price = prices[i]
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.normal(1000000, 200000))

            data.append(
                {
                    "timestamp": datetime.now(timezone.utc)
                    - timedelta(days=n_points - i - 1),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": max(volume, 100000),  # Ensure positive volume
                }
            )

        return pl.DataFrame(data)

    @pytest.fixture
    def trending_data(self):
        """Create data with clear uptrend for testing trend indicators."""
        data = []
        base_price = 100.0

        for i in range(50):
            # Strong uptrend with some noise
            price = base_price + (i * 0.5) + np.random.normal(0, 0.2)
            data.append(
                {
                    "timestamp": datetime.now(timezone.utc) - timedelta(days=50 - i),
                    "open": price * 0.995,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000000,
                }
            )

        return pl.DataFrame(data)

    @pytest.mark.unit
    def test_sma_calculation(self, sample_price_data):
        """Test Simple Moving Average calculation."""
        result = TechnicalIndicators.sma(sample_price_data, period=20)

        # Should have SMA column
        assert "sma_20" in result.columns

        # First 19 values should be null
        sma_values = result.select("sma_20").to_series()
        assert sma_values[:19].is_null().all()

        # Remaining values should be valid numbers
        valid_sma = sma_values[19:].drop_nulls()
        assert len(valid_sma) > 0
        assert all(val > 0 for val in valid_sma)

        # Manual verification of first valid SMA
        first_20_closes = sample_price_data.select("close").to_series()[:20]
        expected_sma = first_20_closes.mean()
        actual_sma = sma_values[19]
        assert abs(actual_sma - expected_sma) < 0.001

    @pytest.mark.unit
    def test_ema_calculation(self, sample_price_data):
        """Test Exponential Moving Average calculation."""
        result = TechnicalIndicators.ema(sample_price_data, period=12)

        assert "ema_12" in result.columns

        ema_values = result.select("ema_12").to_series()

        # EMA should have values (may have some nulls at beginning)
        valid_ema = ema_values.drop_nulls()
        assert len(valid_ema) > 0
        assert all(val > 0 for val in valid_ema)

    @pytest.mark.unit
    def test_rsi_calculation(self, sample_price_data):
        """Test Relative Strength Index calculation."""
        result = TechnicalIndicators.rsi(sample_price_data, period=14)

        assert "rsi_14" in result.columns

        rsi_values = result.select("rsi_14").to_series()

        # Valid RSI values should be between 0 and 100
        # First drop nulls and NaN values
        valid_rsi = rsi_values.drop_nulls()
        # Filter out any remaining NaN values that might not be caught by drop_nulls
        valid_rsi = [
            val for val in valid_rsi if not (isinstance(val, float) and val != val)
        ]

        assert len(valid_rsi) > 0
        # RSI should be between 0 and 100
        assert all(0 <= val <= 100 for val in valid_rsi)

    @pytest.mark.unit
    def test_macd_calculation(self, sample_price_data):
        """Test MACD calculation."""
        result = TechnicalIndicators.macd(sample_price_data, fast=12, slow=26, signal=9)

        # Should have MACD line, signal line, and histogram
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

        macd_line = result.select("macd_line").to_series()
        macd_signal = result.select("macd_signal").to_series()
        macd_histogram = result.select("macd_histogram").to_series()

        # Histogram should be difference between MACD and signal
        valid_data = result.filter(
            pl.col("macd_line").is_not_null()
            & pl.col("macd_signal").is_not_null()
            & pl.col("macd_histogram").is_not_null()
        )

        if len(valid_data) > 0:
            for row in valid_data.rows():
                macd_val, signal_val, hist_val = row[-3:]  # Last 3 columns
                assert abs(hist_val - (macd_val - signal_val)) < 0.001

    @pytest.mark.unit
    def test_bollinger_bands_calculation(self, sample_price_data):
        """Test Bollinger Bands calculation."""
        result = TechnicalIndicators.bollinger_bands(
            sample_price_data, period=20, std_dev=2
        )

        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns

        # Filter valid data (after period)
        valid_data = result.filter(
            pl.col("bb_upper").is_not_null()
            & pl.col("bb_middle").is_not_null()
            & pl.col("bb_lower").is_not_null()
        )

        if len(valid_data) > 0:
            # Upper band should be above middle, middle above lower
            for row in valid_data.select(
                ["close", "bb_lower", "bb_middle", "bb_upper"]
            ).rows():
                close, lower, middle, upper = row
                assert lower < middle < upper

            # Middle band should be the SMA
            middle_values = valid_data.select("bb_middle").to_series()
            sma_result = TechnicalIndicators.sma(sample_price_data, period=20)
            sma_values = sma_result.select("sma_20").to_series().drop_nulls()

            # Compare valid portions
            min_len = min(len(middle_values), len(sma_values))
            if min_len > 0:
                for i in range(min_len):
                    assert abs(middle_values[i] - sma_values[i]) < 0.001

    @pytest.mark.unit
    def test_stochastic_calculation(self, sample_price_data):
        """Test Stochastic Oscillator calculation."""
        result = TechnicalIndicators.stochastic(
            sample_price_data, k_period=14, d_period=3
        )

        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

        stoch_k = result.select("stoch_k").to_series()
        stoch_d = result.select("stoch_d").to_series()

        # Valid stochastic values should be between 0 and 100
        valid_k = stoch_k.drop_nulls()
        valid_d = stoch_d.drop_nulls()

        if len(valid_k) > 0:
            assert all(0 <= val <= 100 for val in valid_k)
        if len(valid_d) > 0:
            assert all(0 <= val <= 100 for val in valid_d)

    @pytest.mark.unit
    def test_atr_calculation(self, sample_price_data):
        """Test Average True Range calculation."""
        result = TechnicalIndicators.atr(sample_price_data, period=14)

        assert "atr_14" in result.columns

        atr_values = result.select("atr_14").to_series()
        valid_atr = atr_values.drop_nulls()

        # ATR should be positive
        if len(valid_atr) > 0:
            assert all(val >= 0 for val in valid_atr)

    @pytest.mark.unit
    def test_volume_indicators(self, sample_price_data):
        """Test volume-based indicators."""
        # Test OBV
        result_obv = TechnicalIndicators.obv(sample_price_data)
        assert "obv" in result_obv.columns

        # Test VWAP
        result_vwap = TechnicalIndicators.vwap(sample_price_data)
        assert "vwap" in result_vwap.columns

        # VWAP should be positive
        vwap_values = result_vwap.select("vwap").to_series().drop_nulls()
        if len(vwap_values) > 0:
            assert all(val > 0 for val in vwap_values)

    @pytest.mark.unit
    def test_trend_detection(self, trending_data):
        """Test trend detection with clear trending data."""
        # Add moving averages for trend detection
        result = TechnicalIndicators.sma(trending_data, period=10)
        result = TechnicalIndicators.sma(result, period=20)

        # In uptrending data, shorter MA should generally be above longer MA
        valid_data = result.filter(
            pl.col("sma_10").is_not_null() & pl.col("sma_20").is_not_null()
        )

        if len(valid_data) > 10:  # Check last 10 points
            last_points = valid_data.tail(10)
            bullish_count = 0

            for row in last_points.select(["sma_10", "sma_20"]).rows():
                sma_10, sma_20 = row
                if sma_10 > sma_20:
                    bullish_count += 1

            # Should be bullish most of the time in trending data
            assert bullish_count >= 7  # At least 70% bullish signals

    @pytest.mark.unit
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty data
        empty_data = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )

        # Should handle empty data gracefully
        result = TechnicalIndicators.sma(empty_data, period=20)
        assert len(result) == 0

        # Single data point
        single_point = pl.DataFrame(
            {
                "timestamp": [datetime.now(timezone.utc)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000],
            }
        )

        result = TechnicalIndicators.sma(single_point, period=20)
        assert len(result) == 1
        assert result.select("sma_20").to_series()[0] is None  # Should be null

    @pytest.mark.unit
    def test_invalid_parameters(self, sample_price_data):
        """Test handling of invalid parameters."""
        # Period larger than data length
        result = TechnicalIndicators.sma(sample_price_data, period=200)
        sma_values = result.select("sma_200").to_series()
        assert sma_values.is_null().all()  # All should be null

        # Zero or negative period should return empty results or handle gracefully
        try:
            result_zero = TechnicalIndicators.sma(sample_price_data, period=0)
            # If it doesn't raise an exception, it should handle gracefully
        except (ValueError, Exception):
            # It's also acceptable to raise an exception
            pass

        try:
            result_negative = TechnicalIndicators.sma(sample_price_data, period=-5)
            # If it doesn't raise an exception, it should handle gracefully
        except (ValueError, Exception):
            # It's also acceptable to raise an exception
            pass

    @pytest.mark.unit
    def test_data_types_consistency(self, sample_price_data):
        """Test that indicators return consistent data types."""
        result = TechnicalIndicators.sma(sample_price_data, period=20)

        # Should maintain original columns plus new indicator
        original_cols = set(sample_price_data.columns)
        result_cols = set(result.columns)

        # All original columns should be present
        assert original_cols.issubset(result_cols)

        # New column should be added
        assert "sma_20" in result_cols

        # Data types should be preserved
        for col in original_cols:
            assert sample_price_data[col].dtype == result[col].dtype
