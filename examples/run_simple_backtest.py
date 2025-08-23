#!/usr/bin/env python3
"""
Simple Backtesting Example

This script demonstrates how to run a backtest using the real AI trading strategy
against historical data. It serves as a basic example of the backtesting capabilities.

Usage:
    python examples/run_simple_backtest.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root / "backtesting"))
sys.path.append(str(project_root / "services" / "data_collector" / "src"))
sys.path.append(str(project_root / "services" / "strategy_engine" / "src"))
sys.path.append(str(project_root / "shared"))

try:
    from real_backtest_engine import (
        RealBacktestEngine,
        RealBacktestConfig,
        BacktestMode,
        run_monthly_backtest,
        run_previous_month_backtest
    )
    from backtest_models import TimeFrame, SignalType, MarketData
    from simple_data_store import SimpleDataStore
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def format_currency(amount):
    """Format currency for display."""
    return f"${amount:,.2f}"


def format_percentage(value):
    """Format percentage for display."""
    return f"{value:.2%}"


def display_simple_results(results, config):
    """Display backtest results in a simple format."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)

    print(f"\nPeriod: {config.start_date.date()} to {config.end_date.date()}")
    print(f"Initial Capital: {format_currency(float(config.initial_capital))}")
    print(f"Final Portfolio Value: {format_currency(float(results.final_portfolio_value))}")
    print(f"Total Return: {format_percentage(results.total_return)}")

    if results.total_trades > 0:
        print(f"\nTrading Activity:")
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Win Rate: {format_percentage(results.win_rate)}")
        print(f"  AI Calls: {results.total_ai_calls}")
        print(f"  Signals Generated: {results.signals_generated}")
        print(f"  Signals Executed: {results.signals_executed}")
    else:
        print(f"\nNo trades were executed during this period.")

    print(f"\nExecution Time: {results.execution_time_seconds:.2f} seconds")
    print("="*60)


async def run_simple_example():
    """Run a simple backtesting example."""
    print("Starting Simple Backtesting Example")
    print("=====================================")

    try:
        # Example 1: Simple backtest with recent data
        print("\n1. Running simple backtest for recent days...")

        try:
            # Use recent dates that likely have data
            end_date = datetime(2025, 8, 21, tzinfo=timezone.utc)
            start_date = datetime(2025, 8, 18, tzinfo=timezone.utc)

            config = RealBacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=Decimal('10000'),
                max_positions=2,
                mode=BacktestMode.FAST,
                timeframe=TimeFrame.ONE_DAY,
                symbols_to_trade=["AAPL"],
                enable_screener_data=False
            )

            engine = RealBacktestEngine(config)
            results = await engine.run_backtest()

            print("✓ Simple backtest completed successfully!")
            display_simple_results(results, config)

        except Exception as e:
            print(f"✗ Simple backtest failed: {e}")
            print("This might be due to insufficient historical data.")

        # Example 2: Multi-symbol backtest
        print("\n2. Running multi-symbol backtest...")

        try:
            config = RealBacktestConfig(
                start_date=datetime(2025, 8, 18, tzinfo=timezone.utc),
                end_date=datetime(2025, 8, 21, tzinfo=timezone.utc),
                initial_capital=Decimal('20000'),
                max_positions=3,
                mode=BacktestMode.FAST,
                timeframe=TimeFrame.ONE_DAY,
                symbols_to_trade=["AAPL", "MSFT"],  # Try multiple symbols
                enable_screener_data=False
            )

            engine = RealBacktestEngine(config)
            results = await engine.run_backtest()

            print("✓ Multi-symbol backtest completed successfully!")
            display_simple_results(results, config)

        except Exception as e:
            print(f"✗ Multi-symbol backtest failed: {e}")
            print("This might be due to insufficient historical data for some symbols.")

        # Example 3: Screener-based backtest
        print("\n3. Running screener-based backtest...")

        try:
            config = RealBacktestConfig(
                start_date=datetime(2025, 8, 22, tzinfo=timezone.utc),
                end_date=datetime(2025, 8, 23, tzinfo=timezone.utc),
                initial_capital=Decimal('15000'),
                max_positions=5,
                mode=BacktestMode.FAST,
                timeframe=TimeFrame.ONE_DAY,
                symbols_to_trade=None,  # Use screener
                enable_screener_data=True,
                screener_types=["momentum", "breakouts"]
            )

            engine = RealBacktestEngine(config)
            results = await engine.run_backtest()

            print("✓ Screener-based backtest completed successfully!")
            display_simple_results(results, config)

        except Exception as e:
            print(f"✗ Screener-based backtest failed: {e}")
            print("This might be due to insufficient screener data.")

    except Exception as e:
        print(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()


async def run_data_availability_check():
    """Check what data is available for backtesting."""
    print("\nChecking Data Availability")
    print("==========================")

    try:
        # Initialize simple data store
        data_store = SimpleDataStore(base_path="data/parquet")

        # Check for some common symbols
        available_symbols = data_store.get_available_symbols()
        print(f"Available symbols in data store: {len(available_symbols)} total")
        print(f"First 10 symbols: {available_symbols[:10]}")

        # Check specific symbols
        test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
        test_date = datetime(2025, 8, 20).date()

        print(f"\nChecking for data on {test_date}:")

        available_test_symbols = []
        for symbol in test_symbols:
            if symbol in available_symbols:
                try:
                    df = await data_store.load_market_data(
                        ticker=symbol,
                        timeframe=TimeFrame.ONE_DAY,
                        start_date=test_date,
                        end_date=test_date
                    )

                    if not df.is_empty():
                        available_test_symbols.append(symbol)
                        row = df.row(0, named=True)
                        print(f"  ✓ {symbol}: Close=${row['close']:.2f}, Volume={row['volume']:,}")
                    else:
                        print(f"  ✗ {symbol}: No data for {test_date}")

                except Exception as e:
                    print(f"  ✗ {symbol}: Error - {e}")
            else:
                print(f"  ✗ {symbol}: Not in data store")

        if available_test_symbols:
            print(f"\nRecommended symbols for backtesting: {', '.join(available_test_symbols)}")
        else:
            print(f"\n⚠️  None of the test symbols have data for {test_date}")
            print("You can try other symbols or dates")

        # Check date ranges for available symbols
        print(f"\nDate ranges for first few symbols:")
        for symbol in available_symbols[:3]:
            timeframes = data_store.get_available_timeframes(symbol)
            if timeframes:
                date_range = data_store.get_date_range_for_symbol(symbol, timeframes[0])
                print(f"  {symbol} ({timeframes[0].value}): {date_range[0]} to {date_range[1]}")

        # Check screener data availability
        print(f"\nChecking screener data for {test_date}:")
        screener_types = ["momentum", "breakouts", "value_stocks"]

        for screener_type in screener_types:
            try:
                df = await data_store.load_screener_data(
                    screener_type=screener_type,
                    start_date=test_date,
                    end_date=test_date
                )

                if not df.is_empty():
                    symbols = df.select("symbol").to_series().to_list()
                    print(f"  ✓ {screener_type}: {len(symbols)} symbols")
                else:
                    print(f"  ✗ {screener_type}: No data")

            except Exception as e:
                print(f"  ✗ {screener_type}: Error - {e}")

    except Exception as e:
        print(f"Data availability check failed: {e}")


async def main():
    """Main execution function."""
    setup_logging()

    print("Real AI Trading Strategy Backtest Example")
    print("=========================================")
    print("This example demonstrates how to backtest the actual AI trading strategy")
    print("using historical market data stored in parquet files.")
    print()

    # Check data availability first
    await run_data_availability_check()

    # Run the backtest examples
    await run_simple_example()

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
    print("Next Steps:")
    print("1. Check the 'scripts/run_monthly_backtest.py' for more advanced options")
    print("2. Review the 'tests/integration/test_real_backtesting.py' for comprehensive tests")
    print("3. Modify the configuration parameters to test different scenarios")
    print("4. Ensure your data collection service is running to get fresh data")
    print("5. Try different date ranges and symbols based on data availability")
    print("6. Experiment with different AI strategy parameters")
    print()
    print("Quick test command:")
    print("./venv/bin/python examples/run_simple_backtest.py")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()
