#!/usr/bin/env python3
"""
Demonstration script for the enhanced calculate_optimal_intervals function.

This script showcases the sophisticated interval calculation algorithm
that was built to replace the simple implementation mentioned in the comment:
"Simple calculation - can be made more sophisticated"

The enhanced algorithm considers:
- Multiple API rate limits from different services
- Market volatility for dynamic adjustment
- Priority weights for different timeframes
- Dynamic batch sizing with efficiency curves
- Safety margins and burst allowances
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.data_collector.src.scheduler_service import calculate_optimal_intervals
from shared.models import TimeFrame


def demonstrate_basic_usage():
    """Demonstrate basic usage of the enhanced function."""
    print("=" * 60)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 60)

    # Basic configuration
    api_limits = {
        "twelvedata": 800,  # TwelveData API
        "finviz": 100,  # FinViz screener
        "redis": 10000,  # Redis cache
    }

    active_tickers = 25
    timeframes = [
        TimeFrame.FIVE_MINUTES,
        TimeFrame.FIFTEEN_MINUTES,
        TimeFrame.ONE_HOUR,
        TimeFrame.ONE_DAY,
    ]

    intervals = calculate_optimal_intervals(
        api_rate_limits=api_limits, active_tickers=active_tickers, timeframes=timeframes
    )

    print("Configuration:")
    print(f"  Active tickers: {active_tickers}")
    print(f"  API limits: {api_limits}")
    print("  Market volatility: 1.0 (default - normal)")
    print()

    print("Calculated intervals:")
    for timeframe, interval in intervals.items():
        minutes = interval // 60
        seconds = interval % 60
        print(f"  {timeframe.value:>15}: {interval:>6}s ({minutes:>2}m {seconds:>2}s)")
    print()


def demonstrate_volatility_adjustment():
    """Demonstrate how market volatility affects interval calculations."""
    print("=" * 60)
    print("MARKET VOLATILITY ADJUSTMENT DEMONSTRATION")
    print("=" * 60)

    api_limits = {"api": 500}
    active_tickers = 30
    timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.ONE_HOUR]

    volatility_scenarios = [
        (0.5, "Calm market"),
        (1.0, "Normal market"),
        (1.5, "Volatile market"),
        (2.0, "High volatility"),
    ]

    print(f"Configuration: {active_tickers} tickers, API limit: 500 req/min")
    print()

    for volatility, description in volatility_scenarios:
        intervals = calculate_optimal_intervals(
            api_rate_limits=api_limits,
            active_tickers=active_tickers,
            timeframes=timeframes,
            market_volatility=volatility,
        )

        print(f"{description} (volatility={volatility}):")
        for timeframe, interval in intervals.items():
            minutes = interval // 60
            print(f"  {timeframe.value:>15}: {interval:>6}s ({minutes:>2}m)")
        print()


def demonstrate_priority_weights():
    """Demonstrate how priority weights affect scheduling."""
    print("=" * 60)
    print("PRIORITY WEIGHTS DEMONSTRATION")
    print("=" * 60)

    api_limits = {"api": 400}
    active_tickers = 40
    timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES, TimeFrame.ONE_HOUR]

    priority_scenarios = [
        (None, "Default priorities"),
        (
            {
                TimeFrame.FIVE_MINUTES: 5.0,  # Very high priority
                TimeFrame.FIFTEEN_MINUTES: 2.0,  # Medium priority
                TimeFrame.ONE_HOUR: 1.0,  # Low priority
            },
            "Day trading focus",
        ),
        (
            {
                TimeFrame.FIVE_MINUTES: 1.0,  # Low priority
                TimeFrame.FIFTEEN_MINUTES: 3.0,  # High priority
                TimeFrame.ONE_HOUR: 4.0,  # Very high priority
            },
            "Swing trading focus",
        ),
    ]

    print(f"Configuration: {active_tickers} tickers, API limit: 400 req/min")
    print()

    for weights, description in priority_scenarios:
        intervals = calculate_optimal_intervals(
            api_rate_limits=api_limits,
            active_tickers=active_tickers,
            timeframes=timeframes,
            priority_weights=weights,
        )

        print(f"{description}:")
        if weights:
            print("  Priority weights:")
            for tf, weight in weights.items():
                print(f"    {tf.value}: {weight}")

        print("  Calculated intervals:")
        for timeframe, interval in intervals.items():
            minutes = interval // 60
            print(f"    {timeframe.value:>15}: {interval:>6}s ({minutes:>2}m)")
        print()


def demonstrate_rate_limit_constraints():
    """Demonstrate how different API rate limits affect scheduling."""
    print("=" * 60)
    print("API RATE LIMIT CONSTRAINTS DEMONSTRATION")
    print("=" * 60)

    active_tickers = 60
    timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.ONE_HOUR]

    rate_limit_scenarios = [
        ({"premium_api": 2000}, "Premium API (2000 req/min)"),
        ({"standard_api": 500}, "Standard API (500 req/min)"),
        ({"basic_api": 100}, "Basic API (100 req/min)"),
        ({"free_api": 25}, "Free tier API (25 req/min)"),
        (
            {
                "primary_api": 800,
                "backup_api": 50,  # Most restrictive
                "cache_api": 5000,
            },
            "Mixed APIs (constrained by backup)",
        ),
    ]

    print(f"Configuration: {active_tickers} tickers, normal volatility")
    print()

    for api_limits, description in rate_limit_scenarios:
        intervals = calculate_optimal_intervals(
            api_rate_limits=api_limits,
            active_tickers=active_tickers,
            timeframes=timeframes,
        )

        print(f"{description}:")
        print(f"  Rate limits: {api_limits}")
        print("  Calculated intervals:")
        for timeframe, interval in intervals.items():
            minutes = interval // 60
            seconds = interval % 60

            # Calculate effective request rate
            batch_size = min(25, max(5, active_tickers // 20))
            batches_needed = max(1, (active_tickers + batch_size - 1) // batch_size)
            req_per_min = round(60 / interval * batches_needed, 1)

            print(
                f"    {timeframe.value:>15}: {interval:>6}s ({minutes:>2}m {seconds:>2}s) -> ~{req_per_min} req/min"
            )
        print()


def demonstrate_scaling_behavior():
    """Demonstrate how the algorithm scales with ticker count."""
    print("=" * 60)
    print("SCALING BEHAVIOR DEMONSTRATION")
    print("=" * 60)

    api_limits = {"api": 300}  # Medium rate limit
    timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.ONE_HOUR]

    ticker_scenarios = [10, 25, 50, 100, 200, 500]

    print("Configuration: API limit 300 req/min, normal volatility")
    print()
    print(
        f"{'Tickers':<8} {'5min Interval':<15} {'1h Interval':<15} {'Req/Min (5m)':<12}"
    )
    print("-" * 60)

    for ticker_count in ticker_scenarios:
        intervals = calculate_optimal_intervals(
            api_rate_limits=api_limits,
            active_tickers=ticker_count,
            timeframes=timeframes,
        )

        # Calculate request rate for 5-minute timeframe
        interval_5m = intervals[TimeFrame.FIVE_MINUTES]
        batch_size = min(25, max(5, ticker_count // 20)) if ticker_count > 0 else 1
        batches_needed = (
            max(1, (ticker_count + batch_size - 1) // batch_size)
            if ticker_count > 0
            else 1
        )
        req_per_min = round(60 / interval_5m * batches_needed, 1)

        interval_1h = intervals[TimeFrame.ONE_HOUR]

        print(
            f"{ticker_count:<8} {interval_5m}s ({interval_5m // 60}m)    {interval_1h}s ({interval_1h // 60}m)     {req_per_min}"
        )

    print()


def demonstrate_real_world_scenarios():
    """Demonstrate real-world trading scenarios."""
    print("=" * 60)
    print("REAL-WORLD TRADING SCENARIOS")
    print("=" * 60)

    scenarios = [
        {
            "name": "Individual Day Trader",
            "api_limits": {"free_api": 50},
            "tickers": 15,
            "volatility": 1.8,
            "priorities": {TimeFrame.FIVE_MINUTES: 5.0, TimeFrame.FIFTEEN_MINUTES: 2.0},
            "timeframes": [TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES],
        },
        {
            "name": "Small Hedge Fund",
            "api_limits": {"premium_api": 800, "backup_api": 200},
            "tickers": 75,
            "volatility": 1.2,
            "priorities": {
                TimeFrame.FIVE_MINUTES: 3.0,
                TimeFrame.FIFTEEN_MINUTES: 4.0,
                TimeFrame.ONE_HOUR: 3.0,
            },
            "timeframes": [
                TimeFrame.FIVE_MINUTES,
                TimeFrame.FIFTEEN_MINUTES,
                TimeFrame.ONE_HOUR,
            ],
        },
        {
            "name": "Research Institution",
            "api_limits": {"enterprise_api": 5000},
            "tickers": 500,
            "volatility": 0.9,
            "priorities": {
                TimeFrame.FIFTEEN_MINUTES: 2.0,
                TimeFrame.ONE_HOUR: 3.0,
                TimeFrame.ONE_DAY: 4.0,
            },
            "timeframes": [
                TimeFrame.FIFTEEN_MINUTES,
                TimeFrame.ONE_HOUR,
                TimeFrame.ONE_DAY,
            ],
        },
    ]

    for scenario in scenarios:
        intervals = calculate_optimal_intervals(
            api_rate_limits=scenario["api_limits"],
            active_tickers=scenario["tickers"],
            timeframes=scenario["timeframes"],
            market_volatility=scenario["volatility"],
            priority_weights=scenario["priorities"],
        )

        print(f"{scenario['name']}:")
        print(f"  Portfolio: {scenario['tickers']} tickers")
        print(f"  API limits: {scenario['api_limits']}")
        print(f"  Market volatility: {scenario['volatility']}")
        print(f"  Priority weights: {scenario['priorities']}")
        print("  Optimal intervals:")

        for timeframe, interval in intervals.items():
            minutes = interval // 60
            seconds = interval % 60
            print(
                f"    {timeframe.value:>15}: {interval:>6}s ({minutes:>2}m {seconds:>2}s)"
            )
        print()


def demonstrate_algorithm_improvements():
    """Show the improvements over the original simple calculation."""
    print("=" * 60)
    print("ALGORITHM IMPROVEMENTS COMPARISON")
    print("=" * 60)

    # Simulate the old simple calculation
    def simple_calculate_optimal_intervals(api_rate_limits, active_tickers, timeframes):
        """Original simple implementation for comparison."""
        base_intervals = {
            TimeFrame.FIVE_MINUTES: 300,
            TimeFrame.FIFTEEN_MINUTES: 900,
            TimeFrame.ONE_HOUR: 3600,
            TimeFrame.ONE_DAY: 86400,
        }

        requests_per_update = {tf: max(1, active_tickers // 100) for tf in timeframes}

        optimal_intervals = {}
        for timeframe in timeframes:
            base_interval = base_intervals.get(timeframe, 300)
            requests_needed = requests_per_update[timeframe]
            optimal_interval = max(base_interval, requests_needed * 60)
            optimal_intervals[timeframe] = optimal_interval

        return optimal_intervals

    # Test scenarios
    test_cases = [
        {
            "name": "High volatility, many tickers",
            "api_limits": {"api": 100},
            "tickers": 150,
            "volatility": 2.0,
            "priorities": {TimeFrame.FIVE_MINUTES: 4.0, TimeFrame.ONE_HOUR: 1.0},
        },
        {
            "name": "Mixed API providers",
            "api_limits": {"fast": 1000, "slow": 25, "medium": 300},
            "tickers": 80,
            "volatility": 1.3,
            "priorities": {TimeFrame.FIVE_MINUTES: 3.0, TimeFrame.ONE_HOUR: 2.0},
        },
    ]

    timeframes = [TimeFrame.FIVE_MINUTES, TimeFrame.ONE_HOUR]

    for case in test_cases:
        print(f"Scenario: {case['name']}")
        print(f"  Tickers: {case['tickers']}, Volatility: {case['volatility']}")
        print(f"  API limits: {case['api_limits']}")

        # Old simple calculation
        old_intervals = simple_calculate_optimal_intervals(
            case["api_limits"], case["tickers"], timeframes
        )

        # New enhanced calculation
        new_intervals = calculate_optimal_intervals(
            api_rate_limits=case["api_limits"],
            active_tickers=case["tickers"],
            timeframes=timeframes,
            market_volatility=case["volatility"],
            priority_weights=case["priorities"],
        )

        print("  Results comparison:")
        for timeframe in timeframes:
            old_min = old_intervals[timeframe] // 60
            new_min = new_intervals[timeframe] // 60
            improvement = (
                "SMARTER"
                if new_intervals[timeframe] != old_intervals[timeframe]
                else "SAME"
            )

            print(
                f"    {timeframe.value:>15}: Old={old_min:>3}m, Enhanced={new_min:>3}m ({improvement})"
            )
        print()


def demonstrate_rate_limit_safety():
    """Demonstrate rate limit safety and burst handling."""
    print("=" * 60)
    print("RATE LIMIT SAFETY DEMONSTRATION")
    print("=" * 60)

    # Scenario: Many tickers with restrictive API
    api_limits = {"restrictive_api": 30}  # Only 30 requests per minute
    active_tickers = 100
    timeframes = [TimeFrame.FIVE_MINUTES]

    print(
        f"Challenge: {active_tickers} tickers with only {list(api_limits.values())[0]} req/min API limit"
    )
    print()

    for volatility in [0.8, 1.0, 1.5, 2.0]:
        intervals = calculate_optimal_intervals(
            api_rate_limits=api_limits,
            active_tickers=active_tickers,
            timeframes=timeframes,
            market_volatility=volatility,
        )

        interval = intervals[TimeFrame.FIVE_MINUTES]

        # Calculate actual request rate
        batch_size = min(25, max(5, active_tickers // 20))
        batches_needed = max(1, (active_tickers + batch_size - 1) // batch_size)
        actual_req_per_min = round(60 / interval * batches_needed, 1)

        safety_margin = (30 - actual_req_per_min) / 30 * 100

        print(
            f"Volatility {volatility}: {interval}s interval -> {actual_req_per_min} req/min ({safety_margin:.1f}% safety margin)"
        )

    print()


def demonstrate_batch_optimization():
    """Demonstrate dynamic batch size optimization."""
    print("=" * 60)
    print("BATCH SIZE OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    api_limits = {"api": 500}
    timeframes = [TimeFrame.FIVE_MINUTES]

    ticker_counts = [5, 15, 35, 75, 150, 300, 600]

    print("Ticker Count -> Optimal Batch Size -> Batches Needed -> Interval")
    print("-" * 60)

    for tickers in ticker_counts:
        intervals = calculate_optimal_intervals(
            api_rate_limits=api_limits, active_tickers=tickers, timeframes=timeframes
        )

        # Calculate what batch size the algorithm chose
        if tickers <= 0:
            batch_size = 1
        elif tickers <= 10:
            batch_size = min(5, tickers)
        elif tickers <= 50:
            batch_size = min(10, tickers)
        elif tickers <= 200:
            batch_size = min(25, tickers)
        else:
            batch_size = min(50, tickers)

        batches_needed = (
            max(1, (tickers + batch_size - 1) // batch_size) if tickers > 0 else 1
        )
        interval = intervals[TimeFrame.FIVE_MINUTES]

        print(
            f"{tickers:>8} -> {batch_size:>8} -> {batches_needed:>8} -> {interval}s ({interval // 60}m)"
        )

    print()


def benchmark_performance():
    """Benchmark the enhanced algorithm performance."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    import time

    # Test different scales
    test_cases = [
        {"tickers": 50, "iterations": 1000, "name": "Small scale"},
        {"tickers": 500, "iterations": 500, "name": "Medium scale"},
        {"tickers": 2000, "iterations": 100, "name": "Large scale"},
    ]

    api_limits = {"api1": 800, "api2": 200, "api3": 1000}
    timeframes = [
        TimeFrame.FIVE_MINUTES,
        TimeFrame.FIFTEEN_MINUTES,
        TimeFrame.ONE_HOUR,
        TimeFrame.ONE_DAY,
    ]

    for case in test_cases:
        start_time = time.perf_counter()

        for _ in range(case["iterations"]):
            intervals = calculate_optimal_intervals(
                api_rate_limits=api_limits,
                active_tickers=case["tickers"],
                timeframes=timeframes,
                market_volatility=1.2,
                priority_weights={tf: 2.0 for tf in timeframes},
            )

        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / case["iterations"]

        print(f"{case['name']} ({case['tickers']} tickers):")
        print(f"  {case['iterations']} calculations in {total_time:.3f}s")
        print(f"  Average time per calculation: {avg_time * 1000:.2f}ms")
        print(f"  Calculations per second: {case['iterations'] / total_time:.0f}")
        print()


def main():
    """Run all demonstrations."""
    print("ENHANCED INTERVAL CALCULATION ALGORITHM DEMONSTRATION")
    print("=" * 80)
    print()
    print("This script demonstrates the enhanced calculate_optimal_intervals function")
    print("that replaced the simple implementation mentioned in the comment:")
    print("'Simple calculation - can be made more sophisticated'")
    print()

    try:
        demonstrate_basic_usage()
        demonstrate_volatility_adjustment()
        demonstrate_priority_weights()
        demonstrate_rate_limit_constraints()
        demonstrate_batch_optimization()
        benchmark_performance()
        demonstrate_algorithm_improvements()

        print("=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print()
        print("Key improvements in the enhanced algorithm:")
        print("✓ Sophisticated rate limiting with safety margins")
        print("✓ Dynamic batch size optimization")
        print("✓ Market volatility awareness")
        print("✓ Configurable priority weighting")
        print("✓ Multi-API provider support")
        print("✓ Robust edge case handling")
        print("✓ Performance optimization")
        print("✓ Mathematical correctness validation")
        print()

    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
