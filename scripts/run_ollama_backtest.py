#!/usr/bin/env python3
"""
Ollama AI Strategy Backtesting Script

This script runs comprehensive backtests using Ollama-powered AI strategy
against real production data with full technical indicators. It replaces
the expensive cloud AI with free local models while maintaining full
compatibility with the production prompt system.

Usage:
    python scripts/run_ollama_backtest.py [options]

Examples:
    # Basic backtest for last week
    python scripts/run_ollama_backtest.py --days 7

    # Multi-stock backtest with custom capital
    python scripts/run_ollama_backtest.py --symbols AAPL,MSFT,GOOGL --capital 50000

    # Technical analysis focused backtest
    python scripts/run_ollama_backtest.py --days 14 --confidence-threshold 70 --save-trades

    # Full month analysis with detailed reporting
    python scripts/run_ollama_backtest.py --start-date 2025-07-01 --end-date 2025-07-31 --detailed-report
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

# pandas import removed as unused

# Add project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.extend(
    [
        str(project_root),
        str(project_root / "backtesting"),
        str(project_root / "services" / "data_collector" / "src"),
        str(project_root / "services" / "strategy_engine" / "src"),
        str(project_root / "shared"),
    ]
)

try:
    from backtest_models import BacktestResults, TimeFrame
    from ollama_ai_strategy_adapter import OllamaAIStrategyAdapter
    from real_backtest_engine import (
        BacktestMode,
        RealBacktestConfig,
        RealBacktestEngine,
    )
    from simple_data_store import SimpleDataStore
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    print("Example: ./venv/bin/python scripts/run_ollama_backtest.py")
    sys.exit(1)


class OllamaBacktestRunner:
    """Orchestrates Ollama-powered backtesting with comprehensive reporting."""

    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_store = SimpleDataStore(base_path="data/parquet")
        self.ollama_strategy = None
        self.results = []

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

        # Reduce noise from some modules
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    async def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """Run comprehensive Ollama-powered backtest."""

        print("🚀 Ollama AI Strategy Backtesting")
        print("=" * 50)

        # Health check Ollama
        await self._check_ollama_health()

        # Determine date range
        start_date, end_date = self._parse_date_range()
        print(f"📅 Period: {start_date.date()} to {end_date.date()}")

        # Get symbols to test
        symbols = await self._get_test_symbols()
        print(f"📊 Symbols: {', '.join(symbols)}")

        # Initialize Ollama AI Strategy
        await self._initialize_ollama_strategy()

        # Execute backtest
        results, config = await self._execute_backtest(start_date, end_date, symbols)

        print("📈 Generating comprehensive report...")
        report = await self._generate_comprehensive_report(
            results, config, start_date, end_date, symbols
        )

        # Add trade details to report for display
        if hasattr(results, "trades") and results.trades:
            report["trade_details"] = []
            for trade in results.trades:
                trade_dict = {
                    "symbol": trade.symbol,
                    "entry_date": trade.entry_date,
                    "exit_date": trade.exit_date,
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price),
                    "quantity": trade.quantity,
                    "pnl": float(trade.pnl),
                    "pnl_percentage": float(trade.pnl_percentage),
                    "hold_days": trade.hold_days,
                    "commission": float(trade.commission),
                }
                report["trade_details"].append(trade_dict)

        # Save results if requested
        if self.args.save_results or self.args.save_trades:
            await self._save_results(results, report)

        return report

    async def _check_ollama_health(self):
        """Check Ollama server health."""
        print("🔍 Checking Ollama server health...")

        from services.strategy_engine.src.ollama_client import OllamaClient

        ollama_url = self.args.ollama_url
        ollama_model = self.args.ollama_model

        client = OllamaClient(ollama_url, ollama_model)

        try:
            health = await client.health_check()
            if not health:
                print(f"❌ Ollama server not healthy at {ollama_url}")
                print("   Make sure Ollama is running and the model is available")
                sys.exit(1)

            models = await client.list_models()
            if ollama_model not in models:
                print(f"❌ Model {ollama_model} not found")
                print(f"   Available models: {', '.join(models)}")
                sys.exit(1)

            print(f"✅ Ollama healthy: {ollama_url} using {ollama_model}")

        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            sys.exit(1)
        finally:
            await client.close()

    def _parse_date_range(self) -> Tuple[datetime, datetime]:
        """Parse and validate date range."""

        if self.args.start_date and self.args.end_date:
            start_date = datetime.strptime(self.args.start_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            end_date = datetime.strptime(self.args.end_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        elif self.args.days:
            end_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start_date = end_date - timedelta(days=self.args.days)
        else:
            # Default to last 7 days
            end_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start_date = end_date - timedelta(days=7)

        return start_date, end_date

    async def _get_test_symbols(self) -> List[str]:
        """Get symbols to test."""

        if self.args.symbols:
            symbols = [s.strip().upper() for s in self.args.symbols.split(",")]
        else:
            # Auto-select symbols with good data availability
            print("🔍 Auto-selecting symbols with good data coverage...")

            all_symbols = self.data_store.get_available_symbols()

            # Priority symbols (large caps with good liquidity)
            priority_symbols = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "NVDA",
                "META",
                "SPY",
                "QQQ",
            ]

            symbols = []
            for symbol in priority_symbols:
                if symbol in all_symbols:
                    symbols.append(symbol)
                    if len(symbols) >= 5:  # Limit for reasonable test time
                        break

            if not symbols:
                # Fallback to first available symbols
                symbols = all_symbols[:3]

        if not symbols:
            print("❌ No symbols available for testing")
            sys.exit(1)

        return symbols

    async def _initialize_ollama_strategy(self):
        """Initialize Ollama AI Strategy."""
        print("🤖 Initializing Ollama AI Strategy...")

        config = {
            "confidence_threshold": self.args.confidence_threshold,
            "max_position_size": self.args.max_position_size,
            "risk_tolerance": "medium",
        }

        self.ollama_strategy = OllamaAIStrategyAdapter(
            ollama_url=self.args.ollama_url,
            ollama_model=self.args.ollama_model,
            config=config,
        )

        print(
            f"✅ AI Strategy initialized with confidence threshold {self.args.confidence_threshold}%"
        )

    async def _execute_backtest(
        self, start_date: datetime, end_date: datetime, symbols: List[str]
    ) -> tuple:
        """Execute the backtest."""
        print("⚡ Running backtest...")

        config = RealBacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal(str(self.args.capital)),
            max_positions=min(
                len(symbols), 10
            ),  # Don't exceed reasonable position count
            mode=BacktestMode.FAST,
            timeframe=TimeFrame.ONE_DAY,
            symbols_to_trade=symbols,
            enable_screener_data=False,  # Use specified symbols only
            ai_strategy_config={
                "confidence_threshold": self.args.confidence_threshold,
                "max_position_size": self.args.max_position_size,
            },
        )

        # Create engine and replace AI strategy with Ollama
        engine = RealBacktestEngine(config)
        engine.ai_strategy = self.ollama_strategy  # Replace mock with Ollama

        start_time = time.time()
        results = await engine.run_backtest()
        execution_time = time.time() - start_time

        print(f"✅ Backtest completed in {execution_time:.2f} seconds")

        return results, config

    async def _generate_comprehensive_report(
        self,
        results: BacktestResults,
        config: RealBacktestConfig,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
    ) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""

        print("📈 Generating comprehensive report...")

        # Get AI strategy performance
        ai_performance = (
            self.ollama_strategy.get_performance_summary()
            if self.ollama_strategy
            else {}
        )

        # Calculate additional metrics
        days_tested = (end_date - start_date).days
        trades_per_day = results.total_trades / max(1, days_tested)

        report = {
            # Basic Info
            "backtest_info": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days_tested": days_tested,
                "symbols_tested": symbols,
                "ai_model": self.args.ollama_model,
                "ai_backend": "ollama",
                "confidence_threshold": self.args.confidence_threshold,
                "execution_time": results.execution_time_seconds,
            },
            # Portfolio Performance
            "portfolio_performance": {
                "initial_capital": float(config.initial_capital),
                "final_value": float(results.final_portfolio_value),
                "total_return": results.total_return,
                "total_return_pct": results.total_return * 100,
                "annualized_return": results.annualized_return,
                "max_drawdown": results.max_drawdown,
                "max_drawdown_pct": results.max_drawdown * 100,
                "sharpe_ratio": results.sharpe_ratio,
                "profit_factor": results.profit_factor,
            },
            # Trading Activity
            "trading_activity": {
                "total_trades": results.total_trades,
                "winning_trades": results.winning_trades,
                "losing_trades": results.losing_trades,
                "win_rate": results.win_rate,
                "win_rate_pct": results.win_rate * 100,
                "average_win": results.average_win,
                "average_loss": results.average_loss,
                "largest_win": results.largest_win,
                "largest_loss": results.largest_loss,
                "trades_per_day": trades_per_day,
            },
            # AI Strategy Performance
            "ai_performance": {
                "total_ai_calls": ai_performance.get("total_calls", 0),
                "successful_calls": ai_performance.get("successful_calls", 0),
                "success_rate": ai_performance.get("success_rate", 0),
                "average_response_time": ai_performance.get("average_response_time", 0),
                "total_cost": ai_performance.get(
                    "total_cost", 0
                ),  # Always 0 for Ollama
                "signals_generated": results.signals_generated,
                "signals_executed": results.signals_executed,
                "signal_execution_rate": (
                    results.signals_executed / max(1, results.signals_generated)
                )
                * 100,
            },
            # Cost Analysis
            "cost_analysis": {
                "total_ai_cost": 0.0,  # Free with Ollama
                "commission_costs": float(results.total_commissions),
                "cost_per_trade": float(results.total_commissions)
                / max(1, results.total_trades),
                "cost_savings_vs_cloud_ai": self._estimate_cloud_ai_cost_savings(
                    ai_performance.get("total_calls", 0)
                ),
            },
        }

        # Add trade details if available
        if hasattr(results, "trades") and results.trades:
            report["trade_details"] = self._analyze_trade_details(results.trades)

        return report

    def _estimate_cloud_ai_cost_savings(self, total_calls: int) -> float:
        """Estimate cost savings vs cloud AI."""
        # Rough estimate: Claude-3-Sonnet costs ~$3 per million input tokens
        # Assume average 1000 tokens per call
        estimated_tokens = total_calls * 1000
        estimated_cost = (estimated_tokens / 1_000_000) * 3.0
        return round(estimated_cost, 2)

    def _analyze_trade_details(self, trades: List[Any]) -> Dict[str, Any]:
        """Analyze trade details for insights."""
        if not trades:
            return {}

        # Group trades by symbol
        trades_by_symbol = {}
        hold_periods = []
        returns_by_action = {"BUY": [], "SELL": []}

        for trade in trades:
            symbol = trade.symbol
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)

            # Collect metrics
            if hasattr(trade, "hold_period_days"):
                hold_periods.append(trade.hold_period_days)

            if hasattr(trade, "return_percentage") and hasattr(trade, "action"):
                action = (
                    trade.action.value
                    if hasattr(trade.action, "value")
                    else str(trade.action)
                )
                if action in returns_by_action:
                    returns_by_action[action].append(trade.return_percentage)

        # Calculate symbol performance
        symbol_performance = {}
        for symbol, symbol_trades in trades_by_symbol.items():
            total_return = sum(
                t.profit_loss for t in symbol_trades if hasattr(t, "profit_loss")
            )
            trade_count = len(symbol_trades)
            win_rate = len(
                [
                    t
                    for t in symbol_trades
                    if hasattr(t, "profit_loss") and t.profit_loss > 0
                ]
            ) / max(1, trade_count)

            symbol_performance[symbol] = {
                "trades": trade_count,
                "total_return": float(total_return),
                "win_rate": win_rate,
                "avg_return_per_trade": float(total_return) / max(1, trade_count),
            }

        return {
            "symbol_performance": symbol_performance,
            "average_hold_period": sum(hold_periods) / max(1, len(hold_periods)),
            "buy_vs_sell_performance": {
                "buy_avg_return": sum(returns_by_action["BUY"])
                / max(1, len(returns_by_action["BUY"])),
                "sell_avg_return": sum(returns_by_action["SELL"])
                / max(1, len(returns_by_action["SELL"])),
                "buy_trade_count": len(returns_by_action["BUY"]),
                "sell_trade_count": len(returns_by_action["SELL"]),
            },
        }

    def _print_summary_report(self, report: Dict[str, Any]):
        """Print formatted summary report."""
        print("\n" + "=" * 80)
        print("🎯 OLLAMA AI STRATEGY BACKTEST RESULTS")
        print("=" * 80)

        # Portfolio Performance
        perf = report["portfolio_performance"]
        print("\n📊 PORTFOLIO PERFORMANCE")
        print(f"   Initial Capital: ${perf['initial_capital']:,.2f}")
        print(f"   Final Value:     ${perf['final_value']:,.2f}")
        print(f"   Total Return:    {perf['total_return_pct']:+.2f}%")
        print(f"   Annualized:      {perf['annualized_return'] * 100:+.2f}%")
        print(f"   Max Drawdown:    {perf['max_drawdown_pct']:-.2f}%")
        print(f"   Sharpe Ratio:    {perf['sharpe_ratio']:.2f}")

        # Trading Activity
        trading = report["trading_activity"]
        print("\n📈 TRADING ACTIVITY")
        print(f"   Total Trades:    {trading['total_trades']}")
        print(
            f"   Win Rate:        {trading['win_rate_pct']:.1f}% ({trading['winning_trades']}/{trading['total_trades']})"
        )
        print(f"   Avg Win:         ${trading['average_win']:.2f}")
        print(f"   Avg Loss:        ${trading['average_loss']:.2f}")
        print(f"   Profit Factor:   {perf['profit_factor']:.2f}")
        print(f"   Trades/Day:      {trading['trades_per_day']:.1f}")

        # AI Performance
        ai_perf = report["ai_performance"]
        print("\n🤖 AI STRATEGY PERFORMANCE")
        print(f"   AI Calls Made:   {ai_perf['total_ai_calls']}")
        print(f"   Success Rate:    {ai_perf['success_rate']:.1f}%")
        print(f"   Avg Response:    {ai_perf['average_response_time']:.2f}s")
        print(f"   Signals Generated: {ai_perf['signals_generated']}")
        print(f"   Signals Executed:  {ai_perf['signals_executed']}")
        print(f"   Execution Rate:    {ai_perf['signal_execution_rate']:.1f}%")

        # Cost Analysis
        costs = report["cost_analysis"]
        print("\n💰 COST ANALYSIS")
        print("   AI Costs:        $0.00 (Free with Ollama! 🎉)")
        print(f"   Commission:      ${costs['commission_costs']:.2f}")
        print(f"   Cost per Trade:  ${costs['cost_per_trade']:.2f}")
        print(f"   Savings vs Cloud: ~${costs['cost_savings_vs_cloud_ai']:.2f}")

        # Configuration
        info = report["backtest_info"]
        print("\n⚙️  CONFIGURATION")
        print(f"   Period:          {info['days_tested']} days")
        print(f"   Symbols:         {', '.join(info['symbols_tested'])}")
        print(f"   AI Model:        {info['ai_model']}")
        print(f"   Confidence:      {info['confidence_threshold']}%")
        print(f"   Execution Time:  {info['execution_time']:.2f}s")

        # Symbol Performance (if available)
        if (
            "trade_details" in report
            and "symbol_performance" in report["trade_details"]
        ):
            print("\n🎯 SYMBOL PERFORMANCE")
            symbol_perf = report["trade_details"]["symbol_performance"]
            for symbol, perf in symbol_perf.items():
                print(
                    f"   {symbol}: {perf['trades']} trades, {perf['total_return']:+.2f} return, {perf['win_rate'] * 100:.0f}% win rate"
                )

        print("=" * 80)

        # Performance Assessment
        total_return_pct = perf["total_return_pct"]
        win_rate_pct = trading["win_rate_pct"]

        if total_return_pct > 10 and win_rate_pct > 60:
            print("🏆 EXCELLENT PERFORMANCE! Strategy shows strong potential.")
        elif total_return_pct > 5 and win_rate_pct > 50:
            print(
                "👍 GOOD PERFORMANCE. Strategy shows promise with room for optimization."
            )
        elif total_return_pct > 0:
            print("📊 MODEST PERFORMANCE. Consider adjusting parameters or strategy.")
        else:
            print("⚠️  UNDERPERFORMANCE. Strategy may need significant adjustments.")

        print("\n💡 Next Steps:")
        if ai_perf["success_rate"] < 80:
            print("   - Consider adjusting confidence threshold or prompt templates")
        if trading["trades_per_day"] < 0.1:
            print(
                "   - Strategy may be too conservative, consider lower confidence threshold"
            )
        if perf["max_drawdown_pct"] > 10:
            print("   - Implement better risk management and position sizing")
        if costs["cost_per_trade"] > 5:
            print(
                "   - Consider optimizing trade frequency to reduce commission impact"
            )

        print("   - Try different time periods and market conditions")
        print("   - Test with different AI models (try deepseek-r1:8b for comparison)")
        print("   - Experiment with technical indicator parameters")

        # Show individual trade results if available
        if "trade_details" in report and report["trade_details"]:
            self._print_individual_trades(report["trade_details"])

        print("=" * 80)

    def _print_individual_trades(self, trades):
        """Print detailed individual trade results."""
        if not trades:
            return

        print(f"\n📊 INDIVIDUAL TRADE RESULTS ({len(trades)} trades)")
        print("-" * 80)

        for i, trade in enumerate(trades, 1):
            symbol = trade.get("symbol", "N/A")
            pnl = trade.get("pnl", 0)
            pnl_pct = trade.get("pnl_percentage", 0)
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            hold_days = trade.get("hold_days", 0)
            quantity = trade.get("quantity", 0)
            direction = "LONG" if quantity > 0 else "SHORT"

            result_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚫"

            print(
                f"{i:2d}. {result_emoji} {direction} {symbol:6s} | "
                f"${pnl:+8.2f} ({pnl_pct:+6.2f}%) | "
                f"{hold_days:2d}d | "
                f"${entry_price:7.2f} → ${exit_price:7.2f}"
            )

        print("-" * 80)

    async def _save_results(self, results: BacktestResults, report: Dict[str, Any]):
        """Save results to files."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        if self.args.save_results:
            results_file = f"backtest_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📁 Results saved to {results_file}")

        # Save trade details
        if self.args.save_trades and hasattr(results, "trades") and results.trades:
            trades_data = []
            for trade in results.trades:
                trade_dict = {
                    "symbol": trade.symbol,
                    "action": (
                        trade.action.value
                        if hasattr(trade.action, "value")
                        else str(trade.action)
                    ),
                    "entry_date": (
                        trade.entry_date.isoformat()
                        if hasattr(trade, "entry_date")
                        else None
                    ),
                    "exit_date": (
                        trade.exit_date.isoformat()
                        if hasattr(trade, "exit_date")
                        else None
                    ),
                    "entry_price": (
                        float(trade.entry_price)
                        if hasattr(trade, "entry_price")
                        else None
                    ),
                    "exit_price": (
                        float(trade.exit_price)
                        if hasattr(trade, "exit_price")
                        else None
                    ),
                    "quantity": (
                        float(trade.quantity) if hasattr(trade, "quantity") else None
                    ),
                    "profit_loss": (
                        float(trade.profit_loss)
                        if hasattr(trade, "profit_loss")
                        else None
                    ),
                    "return_percentage": (
                        trade.return_percentage
                        if hasattr(trade, "return_percentage")
                        else None
                    ),
                    "hold_period_days": (
                        trade.hold_period_days
                        if hasattr(trade, "hold_period_days")
                        else None
                    ),
                }
                trades_data.append(trade_dict)

            trades_file = f"backtest_trades_{timestamp}.json"
            with open(trades_file, "w") as f:
                json.dump(trades_data, f, indent=2)
            print(f"📁 Trades saved to {trades_file}")

    async def cleanup(self):
        """Cleanup resources."""
        if self.ollama_strategy:
            await self.ollama_strategy.cleanup()


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Ollama AI Strategy Backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 7-day backtest
  python scripts/run_ollama_backtest.py --days 7

  # Custom symbols and capital
  python scripts/run_ollama_backtest.py --symbols AAPL,MSFT,GOOGL --capital 50000

  # High confidence threshold
  python scripts/run_ollama_backtest.py --confidence-threshold 75 --days 14

  # Full analysis with trade details
  python scripts/run_ollama_backtest.py --start-date 2025-07-01 --end-date 2025-07-31 --save-trades --detailed-report
""",
    )

    # Date range options
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--days", type=int, default=7, help="Number of days to backtest (default: 7)"
    )
    date_group.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end-date", type=str, help="End date (YYYY-MM-DD, required with --start-date)"
    )

    # Symbols and capital
    parser.add_argument(
        "--symbols", type=str, help="Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )

    # AI configuration
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://192.168.1.133:11434",
        help="Ollama server URL (default: http://192.168.1.133:11434)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:latest",
        help="Ollama model to use (default: llama3.1:latest)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=60.0,
        help="AI confidence threshold %% (default: 60.0)",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=0.10,
        help="Maximum position size as decimal (default: 0.10)",
    )

    # Output options
    parser.add_argument(
        "--save-results", action="store_true", help="Save results to JSON file"
    )
    parser.add_argument(
        "--save-trades", action="store_true", help="Save individual trade details"
    )
    parser.add_argument(
        "--detailed-report", action="store_true", help="Show detailed analysis report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


async def main():
    """Main execution function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.start_date and not args.end_date:
        print("❌ --end-date is required when using --start-date")
        sys.exit(1)

    if args.confidence_threshold < 0 or args.confidence_threshold > 100:
        print("❌ Confidence threshold must be between 0 and 100")
        sys.exit(1)

    if args.max_position_size <= 0 or args.max_position_size > 1:
        print("❌ Max position size must be between 0 and 1")
        sys.exit(1)

    # Initialize runner
    runner = OllamaBacktestRunner(args)

    try:
        # Run backtest
        report = await runner.run_comprehensive_backtest()

        # Display results
        runner._print_summary_report(report)

        if args.detailed_report:
            print("\n📋 Detailed Report:")
            print(json.dumps(report, indent=2, default=str))

    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
    finally:
        await runner.cleanup()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
