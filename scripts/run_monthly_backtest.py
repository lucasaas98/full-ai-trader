#!/usr/bin/env python3
"""
Monthly Backtesting Script

This script runs the real AI trading strategy against historical data for backtesting.
It simulates the complete trading system using past market data to evaluate performance.

Usage:
    python scripts/run_monthly_backtest.py [options]

Examples:
    # Run backtest for previous month
    python scripts/run_monthly_backtest.py

    # Run backtest for specific period
    python scripts/run_monthly_backtest.py --start-date 2025-07-01 --end-date 2025-07-31

    # Run with custom capital and symbols
    python scripts/run_monthly_backtest.py --capital 50000 --symbols AAPL,MSFT,GOOGL

    # Run in debug mode with detailed logging
    python scripts/run_monthly_backtest.py --debug --save-trades
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root / "backtesting"))
sys.path.append(str(project_root / "services" / "data_collector" / "src"))
sys.path.append(str(project_root / "services" / "strategy_engine" / "src"))
sys.path.append(str(project_root / "shared"))

from real_backtest_engine import (
    BacktestMode,
    RealBacktestConfig,
    RealBacktestEngine,
    run_monthly_backtest,
    run_previous_month_backtest,
)

from shared.models import TimeFrame


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("backtest.log"),
        ],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AI trading strategy backtest using historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )

    # Date range options
    date_group = parser.add_argument_group("Date Range")
    date_group.add_argument(
        "--start-date",
        type=str,
        help="Start date for backtest (YYYY-MM-DD). If not specified, uses previous month.",
    )
    date_group.add_argument(
        "--end-date",
        type=str,
        help="End date for backtest (YYYY-MM-DD). If not specified, uses previous month.",
    )
    date_group.add_argument(
        "--days",
        type=int,
        help="Number of days to backtest from end date (alternative to start-date)",
    )

    # Portfolio options
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
        help="Maximum number of concurrent positions (default: 10)",
    )
    portfolio_group.add_argument(
        "--position-size",
        type=float,
        default=0.2,
        help="Maximum position size as fraction of portfolio (default: 0.2)",
    )

    # Symbol selection
    symbol_group = parser.add_argument_group("Symbol Selection")
    symbol_group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to trade (e.g., AAPL,MSFT,GOOGL)",
    )
    symbol_group.add_argument(
        "--use-screener",
        action="store_true",
        help="Use screener data to find symbols (default if no symbols specified)",
    )
    symbol_group.add_argument(
        "--screener-types",
        type=str,
        default="momentum,breakouts,value_stocks",
        help="Comma-separated screener types to use (default: momentum,breakouts,value_stocks)",
    )

    # Execution options
    exec_group = parser.add_argument_group("Execution Settings")
    exec_group.add_argument(
        "--timeframe",
        choices=["1day", "1h", "15min", "5min"],
        default="1day",
        help="Timeframe for analysis (default: 1day)",
    )
    exec_group.add_argument(
        "--mode",
        choices=["fast", "realtime", "debug"],
        default="fast",
        help="Execution mode (default: fast)",
    )
    exec_group.add_argument(
        "--commission",
        type=float,
        default=1.0,
        help="Commission per trade (default: 1.0)",
    )
    exec_group.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5.0)",
    )

    # Output options
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
        "--save-portfolio",
        action="store_true",
        help="Save daily portfolio values to CSV",
    )
    output_group.add_argument(
        "--no-save", action="store_true", help="Do not save any results to files"
    )

    # Logging options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    return parser.parse_args()


def create_backtest_config(args: argparse.Namespace) -> RealBacktestConfig:
    """Create backtest configuration from command line arguments."""

    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    elif args.days:
        end_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_date = end_date - timedelta(days=args.days)
    else:
        # Default to previous month
        today = datetime.now(timezone.utc)
        start_of_current_month = today.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        start_date = (start_of_current_month - timedelta(days=1)).replace(day=1)
        end_date = start_of_current_month - timedelta(days=1)

    # Parse symbols
    symbols_to_trade = None
    if args.symbols:
        symbols_to_trade = [s.strip().upper() for s in args.symbols.split(",")]

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
        "realtime": BacktestMode.REALTIME,
        "debug": BacktestMode.DEBUG,
    }
    mode = mode_map[args.mode]

    return RealBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(args.capital)),
        max_positions=args.max_positions,
        position_sizing_method="ai_determined",
        mode=mode,
        timeframe=timeframe,
        commission_per_trade=Decimal(str(args.commission)),
        slippage_bps=Decimal(str(args.slippage_bps)),
        symbols_to_trade=symbols_to_trade,
        enable_screener_data=args.use_screener or symbols_to_trade is None,
        screener_types=screener_types,
        max_position_size=Decimal(str(args.position_size)),
    )


def format_currency(amount: float) -> str:
    """Format currency amount for display."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage for display."""
    return f"{value:.2%}"


def display_results(results, config: RealBacktestConfig, quiet: bool = False) -> None:
    """Display backtest results in a formatted way."""
    if quiet:
        return

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    # Configuration summary
    print(f"\nConfiguration:")
    print(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"  Initial Capital: {format_currency(float(config.initial_capital))}")
    print(f"  Timeframe: {config.timeframe.value}")
    print(f"  Mode: {config.mode.value}")
    print(
        f"  Symbols: {config.symbols_to_trade if config.symbols_to_trade else 'Screener-based'}"
    )

    print(f"\nPerformance Summary:")
    print(f"  Total Return: {format_percentage(results.total_return)}")
    print(f"  Annualized Return: {format_percentage(results.annualized_return)}")
    print(
        f"  Final Portfolio Value: {format_currency(float(results.final_portfolio_value))}"
    )
    print(f"  Max Drawdown: {format_percentage(results.max_drawdown)}")

    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {results.sortino_ratio:.3f}")
    print(f"  Calmar Ratio: {results.calmar_ratio:.3f}")

    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {results.total_trades}")
    print(f"  Winning Trades: {results.winning_trades}")
    print(f"  Losing Trades: {results.losing_trades}")
    print(f"  Win Rate: {format_percentage(results.win_rate)}")
    print(f"  Profit Factor: {results.profit_factor:.2f}")

    if results.total_trades > 0:
        print(f"  Average Win: {format_currency(results.average_win)}")
        print(f"  Average Loss: {format_currency(results.average_loss)}")
        print(f"  Largest Win: {format_currency(results.largest_win)}")
        print(f"  Largest Loss: {format_currency(results.largest_loss)}")

    print(f"\nAI Strategy Metrics:")
    print(f"  Total AI Calls: {results.total_ai_calls}")
    print(f"  Signals Generated: {results.signals_generated}")
    print(f"  Signals Executed: {results.signals_executed}")
    print(
        f"  Signal Execution Rate: {format_percentage(results.signals_executed / max(results.signals_generated, 1))}"
    )
    print(f"  Average Confidence: {results.average_confidence:.1f}%")

    print(f"\nExecution Metrics:")
    print(f"  Total Commissions: {format_currency(float(results.total_commissions))}")
    print(f"  Total Slippage: {format_currency(float(results.total_slippage))}")
    print(f"  Execution Time: {results.execution_time_seconds:.2f} seconds")

    print("\n" + "=" * 80)


def save_results(results, config: RealBacktestConfig, args: argparse.Namespace) -> None:
    """Save backtest results to files."""
    if args.no_save:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"backtest_{config.start_date.strftime('%Y%m%d')}_{config.end_date.strftime('%Y%m%d')}_{timestamp}"

    # Save summary results
    summary_file = output_dir / f"{base_filename}_summary.json"
    summary_data = {
        "config": {
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "initial_capital": str(config.initial_capital),
            "timeframe": config.timeframe.value,
            "mode": config.mode.value,
            "symbols": config.symbols_to_trade,
            "max_positions": config.max_positions,
            "max_position_size": str(config.max_position_size),
        },
        "results": {
            "total_return": results.total_return,
            "annualized_return": results.annualized_return,
            "final_portfolio_value": str(results.final_portfolio_value),
            "max_drawdown": results.max_drawdown,
            "sharpe_ratio": results.sharpe_ratio,
            "sortino_ratio": results.sortino_ratio,
            "calmar_ratio": results.calmar_ratio,
            "total_trades": results.total_trades,
            "winning_trades": results.winning_trades,
            "losing_trades": results.losing_trades,
            "win_rate": results.win_rate,
            "profit_factor": results.profit_factor,
            "average_win": results.average_win,
            "average_loss": results.average_loss,
            "largest_win": results.largest_win,
            "largest_loss": results.largest_loss,
            "total_ai_calls": results.total_ai_calls,
            "signals_generated": results.signals_generated,
            "signals_executed": results.signals_executed,
            "average_confidence": results.average_confidence,
            "total_commissions": str(results.total_commissions),
            "total_slippage": str(results.total_slippage),
            "execution_time_seconds": results.execution_time_seconds,
        },
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
                    "Entry Date",
                    "Exit Date",
                    "Entry Price",
                    "Exit Price",
                    "Quantity",
                    "PnL",
                    "PnL %",
                    "Commission",
                    "Hold Days",
                    "Reasoning",
                    "Confidence",
                ]
            )

            for trade in results.trades:
                writer.writerow(
                    [
                        trade.symbol,
                        trade.entry_date.isoformat(),
                        trade.exit_date.isoformat(),
                        str(trade.entry_price),
                        str(trade.exit_price),
                        trade.quantity,
                        str(trade.pnl),
                        f"{trade.pnl_percentage:.4f}",
                        str(trade.commission),
                        trade.hold_days,
                        trade.strategy_reasoning,
                        trade.confidence,
                    ]
                )

        print(f"Trade details saved to: {trades_file}")

    # Save portfolio values if requested
    if args.save_portfolio and results.portfolio_values:
        portfolio_file = output_dir / f"{base_filename}_portfolio.csv"
        with open(portfolio_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Portfolio Value"])

            for date, value in results.portfolio_values:
                writer.writerow([date.isoformat(), str(value)])

        print(f"Portfolio history saved to: {portfolio_file}")


async def main():
    """Main execution function."""
    args = parse_arguments()

    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    try:
        # Create configuration
        config = create_backtest_config(args)

        if not args.quiet:
            print(
                f"Starting backtest from {config.start_date.date()} to {config.end_date.date()}"
            )
            print(f"Initial capital: {format_currency(float(config.initial_capital))}")
            if config.symbols_to_trade:
                print(f"Trading symbols: {', '.join(config.symbols_to_trade)}")
            else:
                print(f"Using screener data: {', '.join(config.screener_types)}")

        # Run backtest
        logger.info("Starting backtest execution")
        engine = RealBacktestEngine(config)
        results = await engine.run_backtest()
        logger.info("Backtest completed successfully")

        # Display results
        display_results(results, config, args.quiet)

        # Save results
        save_results(results, config, args)

        # Return success
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
