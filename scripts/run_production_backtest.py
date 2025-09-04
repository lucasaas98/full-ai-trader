#!/usr/bin/env python3
"""
Production Backtesting Script

This script runs real production trading strategies against historical market data.
It uses the actual HybridStrategy implementations (day trading, swing trading, position trading)
and simulates screener alerts based on FinViz criteria.

Usage:
    python scripts/run_production_backtest.py [options]

Examples:
    # Run day trading strategy for last month
    python scripts/run_production_backtest.py --strategy day_trading --months 1

    # Run all strategies for 3 months with comparison
    python scripts/run_production_backtest.py --compare-strategies --months 3

    # Run swing trading with custom parameters
    python scripts/run_production_backtest.py --strategy swing_trading --start-date 2025-06-01 --end-date 2025-08-21 --capital 50000

    # Run with specific symbols (no screener simulation)
    python scripts/run_production_backtest.py --strategy day_trading --symbols AAPL,MSFT,GOOGL --weeks 4
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root / "backtesting"))

from datetime import datetime, timedelta, timezone  # noqa: E402
from decimal import Decimal  # noqa: E402
from typing import Any, Dict  # noqa: E402

from backtest_models import TimeFrame  # noqa: E402
from production_backtest_engine import (  # noqa: E402
    PRODUCTION_STRATEGIES_AVAILABLE,
    BacktestMode,
    ProductionBacktestConfig,
    ProductionBacktestEngine,
    ScreenerCriteria,
    run_multi_strategy_comparison,
)


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("production_backtest.log"),
        ],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run production trading strategy backtests using historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    # Strategy selection
    strategy_group = parser.add_argument_group("Strategy Selection")
    strategy_group.add_argument(
        "--strategy",
        choices=["day_trading", "swing_trading", "position_trading"],
        default="day_trading",
        help="Trading strategy to use (default: day_trading)",
    )
    strategy_group.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Run all three strategies and compare results",
    )

    # Date range options
    date_group = parser.add_argument_group("Date Range")
    date_group.add_argument(
        "--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)"
    )
    date_group.add_argument(
        "--end-date", type=str, help="End date for backtest (YYYY-MM-DD)"
    )
    date_group.add_argument(
        "--months",
        type=int,
        help="Number of months to backtest from end date (e.g., 3 for last 3 months)",
    )
    date_group.add_argument(
        "--weeks", type=int, help="Number of weeks to backtest from end date"
    )
    date_group.add_argument(
        "--days", type=int, help="Number of days to backtest from end date"
    )

    # Portfolio settings
    portfolio_group = parser.add_argument_group("Portfolio Settings")
    portfolio_group.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)",
    )
    portfolio_group.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum concurrent positions (default: 10)",
    )
    portfolio_group.add_argument(
        "--max-position-size",
        type=float,
        default=0.15,
        help="Maximum position size as fraction of portfolio (default: 0.15)",
    )

    # Symbol selection
    symbol_group = parser.add_argument_group("Symbol Selection")
    symbol_group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of specific symbols to trade (disables screener simulation)",
    )
    symbol_group.add_argument(
        "--screener-types",
        type=str,
        default="breakouts,momentum,value_stocks",
        help="Comma-separated screener types (default: breakouts,momentum,value_stocks)",
    )
    symbol_group.add_argument(
        "--max-screener-symbols",
        type=int,
        default=50,
        help="Maximum symbols per day from screener (default: 50)",
    )

    # Execution settings
    exec_group = parser.add_argument_group("Execution Settings")
    exec_group.add_argument(
        "--timeframe",
        choices=["1day", "1h", "15min", "5min"],
        default="1day",
        help="Timeframe for analysis (default: 1day)",
    )
    exec_group.add_argument(
        "--mode",
        choices=["fast", "detailed", "debug"],
        default="fast",
        help="Execution mode (default: fast)",
    )
    exec_group.add_argument(
        "--commission",
        type=float,
        default=1.0,
        help="Fixed commission per trade (default: 1.0)",
    )
    exec_group.add_argument(
        "--commission-pct",
        type=float,
        default=0.0005,
        help="Commission percentage (default: 0.0005 = 0.05%)",
    )

    # Screener criteria customization
    screener_group = parser.add_argument_group("Screener Criteria")
    screener_group.add_argument(
        "--breakout-volume-ratio",
        type=float,
        default=2.0,
        help="Minimum volume ratio for breakouts (default: 2.0)",
    )
    screener_group.add_argument(
        "--momentum-change-min",
        type=float,
        default=5.0,
        help="Minimum price change for momentum (default: 5.0%%)",
    )
    screener_group.add_argument(
        "--disable-screener",
        action="store_true",
        help="Disable screener simulation (requires --symbols)",
    )

    # Output settings
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results",
        help="Directory to save results (default: backtest_results)",
    )
    output_group.add_argument(
        "--save-trades", action="store_true", help="Save detailed trade records to CSV"
    )
    output_group.add_argument(
        "--save-daily-values",
        action="store_true",
        help="Save daily portfolio values to CSV",
    )
    output_group.add_argument(
        "--no-save", action="store_true", help="Do not save results to files"
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--benchmark", type=str, help="Benchmark symbol for comparison (e.g., SPY)"
    )
    analysis_group.add_argument(
        "--detailed-analysis",
        action="store_true",
        help="Generate detailed performance analysis",
    )

    # Logging options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    return parser.parse_args()


def determine_date_range(args: argparse.Namespace) -> tuple[datetime, datetime]:
    """Determine the date range for backtesting."""
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    else:
        # Calculate end date (today or specified end date)
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        else:
            end_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        # Calculate start date based on period
        if args.months:
            start_date = end_date - timedelta(days=args.months * 30)
        elif args.weeks:
            start_date = end_date - timedelta(weeks=args.weeks)
        elif args.days:
            start_date = end_date - timedelta(days=args.days)
        else:
            # Default to 1 month
            start_date = end_date - timedelta(days=30)

    return start_date, end_date


def create_backtest_config(
    args: argparse.Namespace, start_date: datetime, end_date: datetime
) -> ProductionBacktestConfig:
    """Create backtesting configuration from arguments."""

    # Parse symbols
    specific_symbols = None
    if args.symbols:
        specific_symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Parse screener types
    screener_types = [s.strip() for s in args.screener_types.split(",")]

    # Parse timeframe
    timeframe_map = {
        "1day": TimeFrame.ONE_DAY,
        "1h": TimeFrame.ONE_HOUR,
        "15min": TimeFrame.FIFTEEN_MIN,
        "5min": TimeFrame.FIVE_MIN,
    }
    timeframe = timeframe_map[args.timeframe]

    # Parse mode
    mode_map = {
        "fast": BacktestMode.FAST,
        "detailed": BacktestMode.DETAILED,
        "debug": BacktestMode.DEBUG,
    }
    mode = mode_map[args.mode]

    # Create screener criteria
    screener_criteria = ScreenerCriteria(
        breakout_volume_ratio=args.breakout_volume_ratio,
        momentum_change_min=args.momentum_change_min,
    )

    return ProductionBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        strategy_type=args.strategy,
        initial_capital=Decimal(str(args.capital)),
        max_positions=args.max_positions,
        max_position_size=Decimal(str(args.max_position_size)),
        mode=mode,
        timeframe=timeframe,
        commission_per_trade=Decimal(str(args.commission)),
        commission_percentage=Decimal(str(args.commission_pct)),
        specific_symbols=specific_symbols,
        enable_screener_simulation=not args.disable_screener
        and specific_symbols is None,
        screener_criteria=screener_criteria,
        screener_types=screener_types,
        max_screener_symbols_per_day=args.max_screener_symbols,
    )


def format_currency(amount: float) -> str:
    """Format currency amount."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage."""
    return f"{value:.2%}"


def display_results(results, quiet: bool = False) -> None:
    """Display comprehensive backtest results."""
    if quiet:
        return

    print("\n" + "=" * 80)
    print("PRODUCTION STRATEGY BACKTEST RESULTS")
    print("=" * 80)

    # Configuration summary
    config = results.config
    print("\nConfiguration:")
    print(f"  Strategy: {config.strategy_type}")
    print(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"  Initial Capital: {format_currency(float(config.initial_capital))}")
    print(f"  Timeframe: {config.timeframe.value}")
    print(f"  Max Positions: {config.max_positions}")
    print(f"  Screener Enabled: {config.enable_screener_simulation}")

    # Performance Summary
    print("\nPerformance Summary:")
    print(f"  Total Return: {format_percentage(results.total_return)}")
    print(f"  Annualized Return: {format_percentage(results.annualized_return)}")
    print(f"  Final Capital: {format_currency(float(results.final_capital))}")
    print(f"  Max Drawdown: {format_percentage(results.max_drawdown)}")
    print(f"  Current Drawdown: {format_percentage(results.current_drawdown)}")

    # Risk Metrics
    print("\nRisk Metrics:")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {results.sortino_ratio:.3f}")
    print(f"  Calmar Ratio: {results.calmar_ratio:.3f}")
    print(f"  Profit Factor: {results.profit_factor:.2f}")

    # Trading Statistics
    print("\nTrading Statistics:")
    print(f"  Total Trades: {results.total_trades}")
    print(f"  Winning Trades: {results.winning_trades}")
    print(f"  Losing Trades: {results.losing_trades}")
    print(f"  Win Rate: {format_percentage(results.win_rate)}")

    if results.total_trades > 0:
        print(f"  Average Win: {format_currency(float(results.avg_win_amount))}")
        print(f"  Average Loss: {format_currency(float(results.avg_loss_amount))}")
        print(f"  Largest Win: {format_currency(float(results.largest_win))}")
        print(f"  Largest Loss: {format_currency(float(results.largest_loss))}")
        print(f"  Avg Hold Time: {results.avg_hold_time_hours:.1f} hours")

    # Strategy Performance
    print("\nStrategy Performance:")
    print(f"  Signals Generated: {results.total_signals_generated}")
    print(f"  Signals Executed: {results.signals_executed}")
    print(
        f"  Signal Execution Rate: {format_percentage(results.signal_execution_rate)}"
    )
    print(f"  Avg Signal Confidence: {results.avg_signal_confidence:.1f}%")

    # Screener Statistics
    print("\nScreener Statistics:")
    print(f"  Screener Alerts Simulated: {results.screener_alerts_simulated}")
    print(f"  Unique Symbols Traded: {results.unique_symbols_traded}")

    # Execution Info
    print("\nExecution Info:")
    print(f"  Execution Time: {results.execution_time_seconds:.2f} seconds")
    print(f"  Strategy: {results.strategy_name}")

    print("\n" + "=" * 80)


def display_strategy_comparison(
    results_dict: Dict[str, Any], quiet: bool = False
) -> None:
    """Display comparison of multiple strategies."""
    if quiet:
        return

    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)

    # Create comparison table
    strategies = list(results_dict.keys())

    print(f"\n{'Metric':<25} ", end="")
    for strategy in strategies:
        print(f"{strategy.replace('_', ' ').title():<20}", end="")
    print()

    print("-" * (25 + 20 * len(strategies)))

    # Performance metrics
    metrics = [
        ("Total Return", "total_return", format_percentage),
        ("Annualized Return", "annualized_return", format_percentage),
        ("Max Drawdown", "max_drawdown", format_percentage),
        ("Sharpe Ratio", "sharpe_ratio", lambda x: f"{x:.3f}"),
        ("Win Rate", "win_rate", format_percentage),
        ("Total Trades", "total_trades", str),
        ("Profit Factor", "profit_factor", lambda x: f"{x:.2f}"),
        ("Avg Hold Time (hrs)", "avg_hold_time_hours", lambda x: f"{x:.1f}"),
    ]

    for metric_name, metric_key, formatter in metrics:
        print(f"{metric_name:<25} ", end="")
        for strategy in strategies:
            if strategy in results_dict:
                value = getattr(results_dict[strategy], metric_key, 0)
                formatted_value = formatter(value)
                print(f"{formatted_value:<20}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

    print("\n" + "=" * 100)

    # Winner analysis
    best_return = max(results_dict.items(), key=lambda x: x[1].total_return)
    best_sharpe = max(results_dict.items(), key=lambda x: x[1].sharpe_ratio)
    best_win_rate = max(results_dict.items(), key=lambda x: x[1].win_rate)

    print("\nStrategy Winners:")
    print(
        f"  Best Total Return: {best_return[0].replace('_', ' ').title()} ({format_percentage(best_return[1].total_return)})"
    )
    print(
        f"  Best Risk-Adjusted Return: {best_sharpe[0].replace('_', ' ').title()} (Sharpe: {best_sharpe[1].sharpe_ratio:.3f})"
    )
    print(
        f"  Best Win Rate: {best_win_rate[0].replace('_', ' ').title()} ({format_percentage(best_win_rate[1].win_rate)})"
    )


def save_results(results, args: argparse.Namespace) -> None:
    """Save backtest results to files."""
    if args.no_save:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"production_backtest_{results.config.strategy_type}_{timestamp}"

    # Save summary results
    summary_file = output_dir / f"{base_filename}_summary.json"
    summary_data = {
        "config": {
            "strategy_type": results.config.strategy_type,
            "start_date": results.start_time.isoformat(),
            "end_date": results.end_time.isoformat(),
            "initial_capital": str(results.initial_capital),
            "timeframe": results.config.timeframe.value,
            "max_positions": results.config.max_positions,
            "screener_enabled": results.config.enable_screener_simulation,
        },
        "performance": {
            "total_return": results.total_return,
            "annualized_return": results.annualized_return,
            "max_drawdown": results.max_drawdown,
            "sharpe_ratio": results.sharpe_ratio,
            "sortino_ratio": results.sortino_ratio,
            "calmar_ratio": results.calmar_ratio,
            "profit_factor": results.profit_factor,
            "win_rate": results.win_rate,
        },
        "trading": {
            "total_trades": results.total_trades,
            "winning_trades": results.winning_trades,
            "losing_trades": results.losing_trades,
            "avg_win_amount": str(results.avg_win_amount),
            "avg_loss_amount": str(results.avg_loss_amount),
            "largest_win": str(results.largest_win),
            "largest_loss": str(results.largest_loss),
        },
        "strategy": {
            "signals_generated": results.total_signals_generated,
            "signals_executed": results.signals_executed,
            "signal_execution_rate": results.signal_execution_rate,
            "avg_signal_confidence": results.avg_signal_confidence,
            "strategy_name": results.strategy_name,
        },
        "screener": {
            "alerts_simulated": results.screener_alerts_simulated,
            "unique_symbols_traded": results.unique_symbols_traded,
        },
        "execution": {"execution_time_seconds": results.execution_time_seconds},
    }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nResults saved to: {summary_file}")

    # Save detailed trades if requested
    if args.save_trades and results.trades:
        trades_file = output_dir / f"{base_filename}_trades.csv"
        with open(trades_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Symbol",
                    "Strategy",
                    "Entry Date",
                    "Exit Date",
                    "Entry Price",
                    "Exit Price",
                    "Quantity",
                    "Gross P&L",
                    "Net P&L",
                    "P&L %",
                    "Commission",
                    "Hold Hours",
                    "Entry Reason",
                    "Exit Reason",
                ]
            )

            for trade in results.trades:
                writer.writerow(
                    [
                        trade.symbol,
                        trade.strategy_name,
                        trade.entry_date.isoformat(),
                        trade.exit_date.isoformat(),
                        str(trade.entry_price),
                        str(trade.exit_price),
                        trade.quantity,
                        str(trade.gross_pnl),
                        str(trade.net_pnl),
                        f"{trade.pnl_percentage:.4f}",
                        str(trade.commission_total),
                        f"{trade.hold_duration_hours:.2f}",
                        trade.entry_reason,
                        trade.exit_reason,
                    ]
                )

        print(f"Trade details saved to: {trades_file}")

    # Save daily portfolio values if requested
    if args.save_daily_values and results.daily_portfolio_values:
        daily_file = output_dir / f"{base_filename}_daily_values.csv"
        with open(daily_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Portfolio Value"])

            for date, value in results.daily_portfolio_values:
                writer.writerow([date.isoformat(), str(value)])

        print(f"Daily portfolio values saved to: {daily_file}")


async def main():
    """Main execution function."""
    args = parse_arguments()

    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # Check if production strategies are available
    if not PRODUCTION_STRATEGIES_AVAILABLE:
        print("ERROR: Production strategies are not available.")
        print("This usually means there are import issues with the strategy modules.")
        print(
            "Please check that all required dependencies are installed and modules are accessible."
        )
        return 1

    try:
        # Determine date range
        start_date, end_date = determine_date_range(args)

        if not args.quiet:
            print("Production Strategy Backtesting")
            print("===============================")
            print(f"Date Range: {start_date.date()} to {end_date.date()}")
            print(f"Period: {(end_date - start_date).days} days")
            print(f"Initial Capital: {format_currency(args.capital)}")

        # Run comparison or single strategy
        if args.compare_strategies:
            print("\nRunning multi-strategy comparison...")

            results_dict = await run_multi_strategy_comparison(
                start_date=start_date,
                end_date=end_date,
                initial_capital=Decimal(str(args.capital)),
            )

            if not results_dict:
                print("ERROR: No strategies completed successfully.")
                return 1

            # Display comparison results
            display_strategy_comparison(results_dict, args.quiet)

            # Save comparison results
            if not args.no_save:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                comparison_file = output_dir / f"strategy_comparison_{timestamp}.json"
                comparison_data = {}

                for strategy_name, results in results_dict.items():
                    comparison_data[strategy_name] = {
                        "total_return": results.total_return,
                        "annualized_return": results.annualized_return,
                        "max_drawdown": results.max_drawdown,
                        "sharpe_ratio": results.sharpe_ratio,
                        "win_rate": results.win_rate,
                        "total_trades": results.total_trades,
                        "profit_factor": results.profit_factor,
                        "execution_time_seconds": results.execution_time_seconds,
                    }

                with open(comparison_file, "w") as f:
                    json.dump(comparison_data, f, indent=2)

                print(f"\nComparison results saved to: {comparison_file}")

        else:
            # Single strategy backtest
            config = create_backtest_config(args, start_date, end_date)

            if not args.quiet:
                print(f"Strategy: {config.strategy_type}")
                print(
                    f"Screener Simulation: {'Enabled' if config.enable_screener_simulation else 'Disabled'}"
                )
                if config.specific_symbols:
                    print(f"Trading Symbols: {', '.join(config.specific_symbols)}")

            print(f"\nRunning {config.strategy_type} backtest...")

            engine = ProductionBacktestEngine(config)
            results = await engine.run_backtest()

            # Display results
            display_results(results, args.quiet)

            # Save results
            save_results(results, args)

        return 0

    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
