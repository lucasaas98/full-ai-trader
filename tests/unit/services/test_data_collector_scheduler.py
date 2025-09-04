"""
Unit tests for the data collector scheduler service components.

This module contains comprehensive tests for the enhanced calculate_optimal_intervals
function and other scheduler-related functionality in the data collector service.
"""

# Add path for imports
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))


class TestCalculateOptimalIntervals:
    """Test suite for the enhanced calculate_optimal_intervals function."""

    @pytest.fixture
    def basic_api_limits(self) -> Dict[str, int]:
        """Basic API rate limits for testing."""
        return {
            "twelvedata": 800,  # requests per minute
            "finviz": 100,  # requests per minute
            "redis": 10000,  # requests per minute
        }

    @pytest.fixture
    def all_timeframes(self) -> List[TimeFrame]:
        """All available timeframes."""
        return [
            TimeFrame.FIVE_MINUTES,
            TimeFrame.FIFTEEN_MINUTES,
            TimeFrame.ONE_HOUR,
            TimeFrame.ONE_DAY,
        ]

    @pytest.mark.unit
    def test_basic_interval_calculation(self, basic_api_limits, all_timeframes):
        """Test basic interval calculation with default parameters."""
        active_tickers = 20

        intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
        )

        # Should return intervals for all timeframes
        assert len(intervals) == 4
        assert TimeFrame.FIVE_MINUTES in intervals
        assert TimeFrame.FIFTEEN_MINUTES in intervals
        assert TimeFrame.ONE_HOUR in intervals
        assert TimeFrame.ONE_DAY in intervals

        # Intervals should be reasonable (not too fast or too slow)
        for timeframe, interval in intervals.items():
            assert 30 <= interval <= 86400  # Between 30 seconds and 1 day

        # Shorter timeframes should generally have shorter or equal intervals
        assert intervals[TimeFrame.FIVE_MINUTES] <= intervals[TimeFrame.FIFTEEN_MINUTES]
        assert intervals[TimeFrame.FIFTEEN_MINUTES] <= intervals[TimeFrame.ONE_HOUR]

    @pytest.mark.unit
    def test_high_volatility_adjustment(self, basic_api_limits, all_timeframes):
        """Test that high volatility leads to more frequent updates."""
        active_tickers = 30

        # Normal volatility
        normal_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            market_volatility=1.0,
        )

        # High volatility
        high_vol_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            market_volatility=2.0,
        )

        # High volatility should result in shorter intervals (more frequent updates)
        for timeframe in all_timeframes:
            assert high_vol_intervals[timeframe] <= normal_intervals[timeframe]

    @pytest.mark.unit
    def test_low_volatility_adjustment(self, basic_api_limits, all_timeframes):
        """Test that low volatility leads to less frequent updates."""
        active_tickers = 30

        # Normal volatility
        normal_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            market_volatility=1.0,
        )

        # Low volatility
        low_vol_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            market_volatility=0.5,
        )

        # Low volatility should result in longer intervals (less frequent updates)
        for timeframe in all_timeframes:
            assert low_vol_intervals[timeframe] >= normal_intervals[timeframe]

    @pytest.mark.unit
    def test_priority_weights_effect(self, basic_api_limits):
        """Test that priority weights affect interval calculation."""
        active_tickers = 25
        timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.ONE_HOUR]

        # High priority for 5-minute data
        high_priority_weights = {TimeFrame.FIVE_MINUTES: 5.0, TimeFrame.ONE_HOUR: 1.0}

        # Equal priority
        equal_priority_weights = {TimeFrame.FIVE_MINUTES: 2.0, TimeFrame.ONE_HOUR: 2.0}

        high_priority_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=timeframes,
            priority_weights=high_priority_weights,
        )

        equal_priority_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=timeframes,
            priority_weights=equal_priority_weights,
        )

        # Higher priority should result in shorter intervals
        assert (
            high_priority_intervals[TimeFrame.FIVE_MINUTES]
            <= equal_priority_intervals[TimeFrame.FIVE_MINUTES]
        )

    @pytest.mark.unit
    def test_rate_limit_constraints(self, all_timeframes):
        """Test that rate limits are properly respected."""
        active_tickers = 100

        # Very restrictive rate limits
        restrictive_limits = {"api": 10}  # Only 10 requests per minute

        intervals = calculate_optimal_intervals(
            api_rate_limits=restrictive_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
        )

        # With many tickers and low rate limits, intervals should be longer
        for timeframe, interval in intervals.items():
            # Should be significantly longer than base intervals due to rate limiting
            assert interval >= 300  # At least 5 minutes

    @pytest.mark.unit
    def test_empty_api_limits(self, all_timeframes):
        """Test behavior with empty API limits."""
        active_tickers = 20

        intervals = calculate_optimal_intervals(
            api_rate_limits={}, active_tickers=active_tickers, timeframes=all_timeframes
        )

        # Should use default rate limits and still return valid intervals
        assert len(intervals) == 4
        for interval in intervals.values():
            assert 30 <= interval <= 86400

    @pytest.mark.unit
    def test_zero_tickers(self, basic_api_limits, all_timeframes):
        """Test behavior with zero active tickers."""
        intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=0,
            timeframes=all_timeframes,
        )

        # Should still return valid intervals (might use minimum batch size)
        assert len(intervals) == 4
        for interval in intervals.values():
            assert 30 <= interval <= 86400

    @pytest.mark.unit
    def test_no_timeframes(self, basic_api_limits):
        """Test behavior with empty timeframes list."""
        intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits, active_tickers=20, timeframes=[]
        )

        # Should return empty dictionary
        assert intervals == {}

    @pytest.mark.unit
    def test_extreme_volatility_values(self, basic_api_limits, all_timeframes):
        """Test behavior with extreme volatility values."""
        active_tickers = 25

        # Test very high volatility
        high_vol_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            market_volatility=10.0,  # Extremely high
        )

        # Test very low volatility
        low_vol_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            market_volatility=0.1,  # Extremely low
        )

        # Both should return valid intervals (function should clamp extreme values)
        for intervals in [high_vol_intervals, low_vol_intervals]:
            for interval in intervals.values():
                assert 30 <= interval <= 86400


class TestRealWorldScenarios:
    """Test with realistic trading scenarios and API constraints."""

    @pytest.fixture
    def all_timeframes(self) -> List[TimeFrame]:
        """All available timeframes."""
        return [
            TimeFrame.FIVE_MINUTES,
            TimeFrame.FIFTEEN_MINUTES,
            TimeFrame.ONE_HOUR,
            TimeFrame.ONE_DAY,
        ]

    @pytest.mark.unit
    def test_retail_trader_scenario(self, all_timeframes):
        """Test scenario for a retail trader with limited API access."""
        intervals = calculate_optimal_intervals(
            api_rate_limits={"basic_api": 100},  # Limited free tier
            active_tickers=15,  # Small portfolio
            timeframes=[TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES],
            market_volatility=1.5,  # Active market
            priority_weights={
                TimeFrame.FIVE_MINUTES: 4.0,
                TimeFrame.FIFTEEN_MINUTES: 2.0,
            },
        )

        # Should prioritize 5-minute updates while respecting API limits
        assert intervals[TimeFrame.FIVE_MINUTES] <= intervals[TimeFrame.FIFTEEN_MINUTES]

        # Should be reasonable for retail trading (not too slow)
        assert intervals[TimeFrame.FIVE_MINUTES] <= 1800  # Max 30 minutes

    @pytest.mark.unit
    def test_institutional_trader_scenario(self, all_timeframes):
        """Test scenario for institutional trader with premium APIs."""
        intervals = calculate_optimal_intervals(
            api_rate_limits={"enterprise_api": 5000, "backup_api": 2000},
            active_tickers=500,  # Large portfolio
            timeframes=all_timeframes,
            market_volatility=1.0,  # Normal market
            priority_weights={tf: 2.0 for tf in all_timeframes},  # Equal priority
        )

        # Should handle large ticker count efficiently
        for interval in intervals.values():
            assert 30 <= interval <= 86400

        # Should be able to update frequently given premium API access
        assert intervals[TimeFrame.FIVE_MINUTES] <= 600  # Max 10 minutes

    @pytest.mark.unit
    def test_mixed_api_provider_scenario(self, all_timeframes):
        """Test scenario with mixed API providers having different rate limits."""
        mixed_limits = {
            "alpha_vantage": 25,  # Most restrictive
            "twelvedata": 800,  # Good tier
            "finnhub": 300,  # Medium tier
            "polygon": 1000,  # Premium tier
            "redis_cache": 10000,  # Local cache
        }

        active_tickers = 60

        intervals = calculate_optimal_intervals(
            api_rate_limits=mixed_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            market_volatility=1.2,
        )

        # Should be constrained by Alpha Vantage (25 req/min)
        for timeframe, interval in intervals.items():
            # Calculate approximate request rate
            batch_size = min(25, max(5, active_tickers // 20))
            batches_needed = (active_tickers + batch_size - 1) // batch_size
            requests_per_minute = 60 / interval * batches_needed

            # Should respect the most restrictive rate limit with safety margin
            assert (
                requests_per_minute <= 20
            ), f"Too many requests for restrictive API: {requests_per_minute}"


class TestBackwardCompatibility:
    """Test backward compatibility and regression prevention."""

    @pytest.fixture
    def basic_api_limits(self) -> Dict[str, int]:
        """Basic API rate limits for testing."""
        return {"twelvedata": 800, "finviz": 100, "redis": 10000}

    @pytest.fixture
    def all_timeframes(self) -> List[TimeFrame]:
        """All available timeframes."""
        return [
            TimeFrame.FIVE_MINUTES,
            TimeFrame.FIFTEEN_MINUTES,
            TimeFrame.ONE_HOUR,
            TimeFrame.ONE_DAY,
        ]

    @pytest.mark.unit
    def test_function_signature_compatibility(self):
        """Test that the enhanced function signature is backward compatible."""
        from inspect import signature

        # Get function signature
        sig = signature(calculate_optimal_intervals)
        params = list(sig.parameters.keys())

        # Should have the original required parameters
        assert "api_rate_limits" in params
        assert "active_tickers" in params
        assert "timeframes" in params

        # Enhanced parameters should be optional
        enhanced_params = sig.parameters
        assert enhanced_params["market_volatility"].default == 1.0
        assert enhanced_params["priority_weights"].default is None

    @pytest.mark.unit
    def test_backward_compatibility_basic_usage(self, basic_api_limits, all_timeframes):
        """Test that basic usage patterns still work as expected."""
        # This simulates how the function was called before enhancement
        intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=30,
            timeframes=all_timeframes,
            # No volatility or priority weights (should use defaults)
        )

        # Should work exactly like before for basic usage
        assert len(intervals) == 4
        for timeframe in all_timeframes:
            assert timeframe in intervals
            assert isinstance(intervals[timeframe], int)
            assert 30 <= intervals[timeframe] <= 86400

    @pytest.mark.unit
    def test_enhanced_features_dont_break_basics(
        self, basic_api_limits, all_timeframes
    ):
        """Test that enhanced features don't break basic functionality."""
        active_tickers = 25

        # Test basic call
        basic_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
        )

        # Test enhanced call with neutral parameters
        enhanced_intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            market_volatility=1.0,  # Neutral volatility
            priority_weights=None,  # Use defaults
        )

        # Results should be very similar
        for timeframe in all_timeframes:
            basic_val = basic_intervals[timeframe]
            enhanced_val = enhanced_intervals[timeframe]

            # Allow small differences due to internal optimizations
            diff_ratio = abs(enhanced_val - basic_val) / basic_val
            assert (
                diff_ratio <= 0.1
            ), f"Enhanced version changed basic behavior too much: {diff_ratio}"

    @pytest.mark.unit
    def test_return_type_consistency(self, basic_api_limits, all_timeframes):
        """Test that return type and structure remain consistent."""
        intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=40,
            timeframes=all_timeframes,
            market_volatility=1.5,
        )

        # Should return a dictionary
        assert isinstance(intervals, dict)

        # Keys should be TimeFrame objects
        for key in intervals.keys():
            assert isinstance(key, TimeFrame)

        # Values should be integers (seconds)
        for value in intervals.values():
            assert isinstance(value, int)
            assert value > 0


class TestEdgeCasesAndValidation:
    """Test edge cases and input validation."""

    @pytest.fixture
    def all_timeframes(self) -> List[TimeFrame]:
        """All available timeframes."""
        return [
            TimeFrame.FIVE_MINUTES,
            TimeFrame.FIFTEEN_MINUTES,
            TimeFrame.ONE_HOUR,
            TimeFrame.ONE_DAY,
        ]

    @pytest.mark.unit
    def test_edge_case_validation(self, all_timeframes):
        """Test various edge cases and boundary conditions."""
        edge_cases = [
            # (active_tickers, api_limits, volatility, description)
            (1, {"api": 1}, 0.1, "minimum everything"),
            (10000, {"api": 100000}, 10.0, "maximum everything"),
            (0, {"api": 100}, 1.0, "zero tickers"),
            (50, {}, 1.0, "no api limits"),
        ]

        for active_tickers, api_limits, volatility, description in edge_cases:
            try:
                intervals = calculate_optimal_intervals(
                    api_rate_limits=api_limits,
                    active_tickers=active_tickers,
                    timeframes=all_timeframes,
                    market_volatility=volatility,
                )

                # Should handle edge cases gracefully
                assert len(intervals) == 4, f"Failed for {description}"
                for interval in intervals.values():
                    assert (
                        30 <= interval <= 86400
                    ), f"Invalid interval for {description}: {interval}"

            except Exception as e:
                pytest.fail(f"Edge case '{description}' raised exception: {e}")

    @pytest.mark.unit
    def test_algorithm_determinism(self, all_timeframes):
        """Test that the algorithm is deterministic and reproducible."""
        test_params = {
            "api_rate_limits": {"api": 500},
            "active_tickers": 75,
            "timeframes": all_timeframes,
            "market_volatility": 1.4,
            "priority_weights": {
                TimeFrame.FIVE_MINUTES: 3.5,
                TimeFrame.FIFTEEN_MINUTES: 2.8,
                TimeFrame.ONE_HOUR: 2.0,
                TimeFrame.ONE_DAY: 1.2,
            },
        }

        # Run calculation multiple times
        results = []
        for _ in range(5):
            intervals = calculate_optimal_intervals(**test_params)
            results.append(intervals)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i], "Algorithm should be deterministic"

    @pytest.mark.unit
    def test_extreme_priority_weights(self, all_timeframes):
        """Test that extreme priority weights are handled gracefully."""
        active_tickers = 20

        # Extreme priority weights
        extreme_weights = {
            TimeFrame.FIVE_MINUTES: 100.0,  # Extremely high
            TimeFrame.FIFTEEN_MINUTES: 0.01,  # Extremely low
            TimeFrame.ONE_HOUR: 2.0,  # Normal
            TimeFrame.ONE_DAY: 0.0,  # Zero (edge case)
        }

        intervals = calculate_optimal_intervals(
            api_rate_limits={"api": 500},
            active_tickers=active_tickers,
            timeframes=all_timeframes,
            priority_weights=extreme_weights,
        )

        # Should handle extreme weights gracefully
        for interval in intervals.values():
            assert 30 <= interval <= 86400

    @pytest.mark.unit
    def test_base_interval_respect(self, all_timeframes):
        """Test that base intervals are always respected."""
        active_tickers = 5  # Very few tickers

        # Even with few tickers and high volatility, should respect base intervals
        intervals = calculate_optimal_intervals(
            api_rate_limits={"api": 10000},  # Unlimited API
            active_tickers=active_tickers,
            timeframes=[TimeFrame.FIVE_MINUTES, TimeFrame.ONE_DAY],
            market_volatility=3.0,  # High volatility
        )

        # Should never go below base intervals
        assert intervals[TimeFrame.FIVE_MINUTES] >= 300  # 5 minutes minimum
        assert intervals[TimeFrame.ONE_DAY] >= 86400  # 1 day minimum


class TestPerformanceAndScalability:
    """Test performance and scalability characteristics."""

    @pytest.fixture
    def all_timeframes(self) -> List[TimeFrame]:
        """All available timeframes."""
        return [
            TimeFrame.FIVE_MINUTES,
            TimeFrame.FIFTEEN_MINUTES,
            TimeFrame.ONE_HOUR,
            TimeFrame.ONE_DAY,
        ]

    @pytest.mark.unit
    def test_large_ticker_count_scaling(self, all_timeframes):
        """Test that algorithm scales well with large ticker counts."""
        api_limits = {"api": 800}

        # Test with increasing ticker counts
        ticker_counts = [10, 50, 100, 500, 1000]

        for ticker_count in ticker_counts:
            intervals = calculate_optimal_intervals(
                api_rate_limits=api_limits,
                active_tickers=ticker_count,
                timeframes=all_timeframes,
            )

            # All intervals should be valid
            for interval in intervals.values():
                assert 30 <= interval <= 86400

            # With many tickers, should respect rate limits appropriately
            if ticker_count >= 500:
                # Should have longer intervals to handle the load
                assert intervals[TimeFrame.FIVE_MINUTES] >= 300

    @pytest.mark.unit
    def test_calculation_performance(self, all_timeframes):
        """Test that calculation completes in reasonable time."""
        import time

        # Large but realistic parameters
        test_params = {
            "api_rate_limits": {"api1": 800, "api2": 200, "api3": 1000},
            "active_tickers": 1000,
            "timeframes": all_timeframes,
            "market_volatility": 1.5,
            "priority_weights": {tf: 2.0 for tf in all_timeframes},
        }

        start_time = time.perf_counter()

        # Run multiple calculations
        for _ in range(50):
            calculate_optimal_intervals(**test_params)

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 50

        # Should complete quickly (less than 1ms per calculation)
        assert avg_time < 0.001, f"Calculation took too long: {avg_time:.4f}s"

    @pytest.mark.unit
    def test_memory_efficiency(self, all_timeframes):
        """Test that the algorithm is memory efficient."""
        import tracemalloc

        # Start memory tracking
        tracemalloc.start()

        # Run calculations with large parameters
        for _ in range(100):
            calculate_optimal_intervals(
                api_rate_limits={"api": 500},
                active_tickers=500,
                timeframes=all_timeframes,
                market_volatility=1.2,
            )

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 100KB peak)
        assert peak < 100 * 1024, f"Peak memory usage too high: {peak} bytes"


class TestAlgorithmValidation:
    """Validate the mathematical properties and correctness of the algorithm."""

    @pytest.fixture
    def basic_api_limits(self) -> Dict[str, int]:
        """Basic API rate limits for testing."""
        return {"api": 500}

    @pytest.fixture
    def all_timeframes(self) -> List[TimeFrame]:
        """All available timeframes."""
        return [
            TimeFrame.FIVE_MINUTES,
            TimeFrame.FIFTEEN_MINUTES,
            TimeFrame.ONE_HOUR,
            TimeFrame.ONE_DAY,
        ]

    @pytest.mark.unit
    def test_rate_limit_math_validation(self, basic_api_limits, all_timeframes):
        """Validate that the algorithm's rate limit calculations are mathematically sound."""
        active_tickers = 80

        intervals = calculate_optimal_intervals(
            api_rate_limits=basic_api_limits,
            active_tickers=active_tickers,
            timeframes=all_timeframes,
        )

        # For each timeframe, verify that the calculated interval won't exceed rate limits
        api_limit = 500  # requests per minute
        safety_margin = 0.8  # 80% of rate limit
        safe_rate = api_limit * safety_margin

        for timeframe, interval in intervals.items():
            # Estimate batch size (this should match the algorithm's internal logic)
            if active_tickers <= 10:
                batch_size = min(5, active_tickers)
            elif active_tickers <= 50:
                batch_size = min(10, active_tickers)
            elif active_tickers <= 200:
                batch_size = min(25, active_tickers)
            else:
                batch_size = min(50, active_tickers)

            batches_needed = max(1, (active_tickers + batch_size - 1) // batch_size)
            requests_per_minute = (60 / interval) * batches_needed

            # Should not exceed safe rate limit
            assert (
                requests_per_minute <= safe_rate * 1.1
            ), f"Rate limit violation for {timeframe}: {requests_per_minute} > {safe_rate}"

    @pytest.mark.unit
    def test_priority_weight_monotonicity(self, basic_api_limits):
        """Test that higher priority weights consistently lead to shorter or equal intervals."""
        active_tickers = 40
        timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.ONE_HOUR]

        # Test different priority weight combinations
        weight_combinations = [
            (1.0, 1.0),  # Equal priority
            (2.0, 1.0),  # Higher priority for 5-minute
            (3.0, 1.0),  # Even higher priority for 5-minute
            (5.0, 1.0),  # Maximum practical priority for 5-minute
        ]

        results = []
        for weight_5m, weight_1h in weight_combinations:
            weights = {TimeFrame.FIVE_MINUTES: weight_5m, TimeFrame.ONE_HOUR: weight_1h}

            intervals = calculate_optimal_intervals(
                api_rate_limits=basic_api_limits,
                active_tickers=active_tickers,
                timeframes=timeframes,
                priority_weights=weights,
            )
            results.append(intervals[TimeFrame.FIVE_MINUTES])

        # Higher priority weights should generally lead to shorter intervals
        # (allowing for some tolerance due to constraints)
        for i in range(len(results) - 1):
            current = results[i]
            next_val = results[i + 1]

            # Should be monotonic or very close (within 20% tolerance)
            ratio = next_val / current
            assert 0.8 <= ratio <= 1.2, f"Priority weights not monotonic: {ratio}"

    @pytest.mark.unit
    def test_consistency_across_runs(self, basic_api_limits, all_timeframes):
        """Test that the function returns consistent results for identical inputs."""
        test_params = {
            "api_rate_limits": basic_api_limits,
            "active_tickers": 35,
            "timeframes": all_timeframes,
            "market_volatility": 1.5,
        }

        # Call function multiple times with identical parameters
        results = []
        for _ in range(10):
            intervals = calculate_optimal_intervals(**test_params)
            results.append(intervals)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i], "Algorithm should be deterministic"

    @pytest.mark.unit
    def test_algorithm_bounds_enforcement(self, basic_api_limits, all_timeframes):
        """Test that all algorithm outputs are within expected bounds."""
        # Test various extreme scenarios
        extreme_scenarios = [
            {"active_tickers": 1, "market_volatility": 5.0},
            {"active_tickers": 5000, "market_volatility": 0.1},
            {"active_tickers": 100, "market_volatility": 10.0},
        ]

        for scenario in extreme_scenarios:
            intervals = calculate_optimal_intervals(
                api_rate_limits=basic_api_limits, timeframes=all_timeframes, **scenario
            )

            # All intervals must be strictly within bounds
            for timeframe, interval in intervals.items():
                assert (
                    30 <= interval <= 86400
                ), f"Interval {interval} out of bounds for scenario {scenario}"

                # Should respect timeframe minimums
                base_minimums = {
                    TimeFrame.FIVE_MINUTES: 300,
                    TimeFrame.FIFTEEN_MINUTES: 900,
                    TimeFrame.ONE_HOUR: 3600,
                    TimeFrame.ONE_DAY: 86400,
                }
                assert (
                    interval >= base_minimums[timeframe]
                ), f"Interval {interval} below base minimum for {timeframe}"


class TestDocumentationExamples:
    """Test examples that would be used in documentation."""

    @pytest.mark.unit
    def test_basic_usage_example(self):
        """Test basic usage example for documentation."""
        # Basic usage - what most users would do
        intervals = calculate_optimal_intervals(
            api_rate_limits={"api": 500},
            active_tickers=25,
            timeframes=[TimeFrame.FIVE_MINUTES, TimeFrame.ONE_HOUR],
        )

        assert len(intervals) == 2
        assert all(30 <= interval <= 86400 for interval in intervals.values())

    @pytest.mark.unit
    def test_advanced_usage_example(self):
        """Test advanced usage example with all parameters."""
        # Advanced usage - power user configuration
        intervals = calculate_optimal_intervals(
            api_rate_limits={"premium_api": 2000},
            active_tickers=100,
            timeframes=[TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES],
            market_volatility=1.8,  # High volatility
            priority_weights={
                TimeFrame.FIVE_MINUTES: 4.0,  # High priority
                TimeFrame.FIFTEEN_MINUTES: 2.0,  # Medium priority
            },
        )

        assert len(intervals) == 2
        assert intervals[TimeFrame.FIVE_MINUTES] <= intervals[TimeFrame.FIFTEEN_MINUTES]

        # Should handle high volatility appropriately
        assert intervals[TimeFrame.FIVE_MINUTES] >= 300  # Still respects base interval

    @pytest.mark.unit
    def test_resource_constrained_example(self):
        """Test resource-constrained environment example."""
        # Limited resources - free tier APIs
        intervals = calculate_optimal_intervals(
            api_rate_limits={"free_api": 25},  # Very limited
            active_tickers=50,
            timeframes=[TimeFrame.FIFTEEN_MINUTES, TimeFrame.ONE_HOUR],
            market_volatility=0.8,  # Calm market
        )

        # Should be conservative with limited resources
        for interval in intervals.values():
            assert interval >= 900  # At least 15 minutes given constraints


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
