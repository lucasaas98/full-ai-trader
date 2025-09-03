#!/usr/bin/env python3
"""
Full Parameter Optimization Example

This example demonstrates the complete workflow for optimizing trading strategy parameters:
1. Run parameter optimization for multiple strategies
2. Analyze results for statistical significance
3. Generate recommendations and reports
4. Compare strategies and identify best configurations

This is a comprehensive example showing how to systematically improve your trading strategies
using historical data and statistical analysis.

Usage:
    python examples/run_full_optimization_example.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "scripts"))
sys.path.append(str(project_root / "backtesting"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("optimization_example.log")],
)
logger = logging.getLogger(__name__)


async def run_optimization_example():
    """Run the complete optimization example workflow."""

    print("🚀 COMPLETE TRADING STRATEGY OPTIMIZATION EXAMPLE")
    print("=" * 70)
    print("This example will:")
    print("1. Optimize parameters for day trading strategy")
    print("2. Optimize parameters for swing trading strategy")
    print("3. Analyze results and generate recommendations")
    print("4. Create comparison reports and visualizations")
    print("5. Provide actionable insights for strategy improvement")
    print()

    # Import the optimization modules
    try:
        from analyze_optimization_results import OptimizationAnalyzer
        from optimize_strategy_parameters import (
            ParameterOptimizer,
            generate_test_periods,
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        return

    # Configuration
    strategies_to_optimize = ["day_trading", "swing_trading"]
    optimization_level = "standard"  # standard, quick, or detailed
    num_periods = 6  # Number of 30-day test periods
    initial_capital = 100000  # Starting capital for each test

    print("Configuration:")
    print(f"  Strategies: {', '.join(strategies_to_optimize)}")
    print(f"  Optimization Level: {optimization_level}")
    print(f"  Test Periods: {num_periods} x 30 days")
    print(f"  Initial Capital: ${initial_capital:,}")
    print()

    # Step 1: Generate test periods
    print("📅 Step 1: Generating test periods...")
    test_periods = generate_test_periods(num_periods, 30)

    print(f"Generated {len(test_periods)} test periods:")
    for i, (start, end) in enumerate(test_periods, 1):
        period_days = (end - start).days
        print(f"  Period {i}: {start.date()} to {end.date()} ({period_days} days)")
    print()

    # Step 2: Run optimization for each strategy
    optimization_results = {}

    for strategy_type in strategies_to_optimize:
        print(
            f"🎯 Step 2.{len(optimization_results) + 1}: Optimizing {strategy_type.replace('_', ' ').title()} Strategy"
        )
        print("-" * 60)

        try:
            # Create optimizer
            optimizer = ParameterOptimizer(
                strategy_type=strategy_type,
                test_periods=test_periods,
                initial_capital=initial_capital,
            )

            # Run optimization
            print(f"Starting optimization with {optimization_level} settings...")
            start_time = datetime.now()

            summaries = await optimizer.run_optimization(optimization_level)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"✅ Optimization completed in {duration:.1f} seconds")
            print(f"📊 Tested {len(summaries)} parameter combinations")

            if summaries:
                best = summaries[0]
                print(f"🏆 Best configuration: {best.parameter_set.name}")
                print(f"   Average Return: {best.avg_return:.2%}")
                print(f"   Risk-Adjusted Score: {best.risk_adjusted_score:.3f}")
                print(f"   Win Rate: {best.avg_win_rate:.1%}")
                print(f"   Consistency Score: {best.consistency_score:.3f}")

                # Store results
                optimization_results[strategy_type] = {
                    "optimizer": optimizer,
                    "summaries": summaries,
                    "best_config": best,
                }

                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                optimizer.save_results("optimization_results", timestamp)
                print(f"💾 Results saved with timestamp: {timestamp}")

            else:
                print("⚠️  No valid optimization results generated")

        except Exception as e:
            print(f"❌ Optimization failed for {strategy_type}: {e}")
            logger.error(f"Optimization error: {e}", exc_info=True)
            continue

        print()

    # Step 3: Analyze and compare results
    if optimization_results:
        print("📈 Step 3: Analyzing Results and Generating Insights")
        print("-" * 50)

        # Display strategy comparison
        if len(optimization_results) > 1:
            print("🏁 STRATEGY PERFORMANCE COMPARISON")
            print("=" * 40)

            comparison_data = []
            for strategy_type, results in optimization_results.items():
                best = results["best_config"]
                comparison_data.append(
                    {
                        "Strategy": strategy_type.replace("_", " ").title(),
                        "Best Config": best.parameter_set.name,
                        "Avg Return": f"{best.avg_return:.2%}",
                        "Max Drawdown": f"{best.avg_max_drawdown:.2%}",
                        "Win Rate": f"{best.avg_win_rate:.1%}",
                        "Risk-Adj Score": f"{best.risk_adjusted_score:.3f}",
                        "Total Trades": best.total_trades,
                        "Profitable Periods": f"{best.periods_profitable}/{best.periods_tested}",
                    }
                )

            # Display comparison table
            print(
                f"{'Strategy':<15} {'Config':<15} {'Return':<10} {'Drawdown':<10} {'Win Rate':<10} {'Score':<8}"
            )
            print("-" * 80)
            for data in comparison_data:
                print(
                    f"{data['Strategy']:<15} {data['Best Config']:<15} {data['Avg Return']:<10} "
                    f"{data['Max Drawdown']:<10} {data['Win Rate']:<10} {data['Risk-Adj Score']:<8}"
                )
            print()

        # Generate detailed insights for each strategy
        for strategy_type, results in optimization_results.items():
            print(f"🔍 DETAILED ANALYSIS: {strategy_type.replace('_', ' ').title()}")
            print("=" * 50)

            summaries = results["summaries"]
            best = results["best_config"]

            # Parameter recommendations
            print("📋 RECOMMENDED PARAMETER CONFIGURATION:")
            print("-" * 40)

            param_config = best.parameter_set
            print(f"Configuration Name: {param_config.name}")
            print(f"Stop Loss: {param_config.stop_loss_pct:.1%}")
            print(f"Take Profit: {param_config.take_profit_pct:.1%}")
            print(f"Minimum Confidence: {param_config.min_confidence:.1f}%")
            print(f"Maximum Position Size: {param_config.max_position_size:.1%}")
            print(f"Technical Analysis Weight: {param_config.ta_weight:.1%}")
            print(f"Fundamental Analysis Weight: {param_config.fa_weight:.1%}")
            print(f"Volume Threshold: {param_config.volume_threshold:.1f}x")
            print()

            # Performance expectations
            print("📊 EXPECTED PERFORMANCE:")
            print("-" * 25)
            print(f"Average Return: {best.avg_return:.2%}")
            print(f"Return Volatility: {best.std_return:.2%}")
            print(f"Maximum Drawdown: {best.avg_max_drawdown:.2%}")
            print(f"Win Rate: {best.avg_win_rate:.1%}")
            print(f"Average Profit Factor: {best.avg_profit_factor:.2f}")
            print(f"Consistency Score: {best.consistency_score:.3f}")
            print(f"Risk-Adjusted Score: {best.risk_adjusted_score:.3f}")
            print()

            # Alternative configurations
            if len(summaries) > 3:
                print("🎛️  ALTERNATIVE CONFIGURATIONS:")
                print("-" * 30)

                # Most consistent (if profitable)
                profitable_configs = [s for s in summaries if s.avg_return > 0]
                if profitable_configs:
                    most_consistent = max(
                        profitable_configs, key=lambda x: x.consistency_score
                    )
                    if most_consistent != best:
                        print(f"Most Consistent: {most_consistent.parameter_set.name}")
                        print(f"  Return: {most_consistent.avg_return:.2%}")
                        print(f"  Consistency: {most_consistent.consistency_score:.3f}")
                        print()

                # Highest return (regardless of risk)
                highest_return = max(summaries, key=lambda x: x.avg_return)
                if highest_return != best:
                    print(f"Highest Return: {highest_return.parameter_set.name}")
                    print(f"  Return: {highest_return.avg_return:.2%}")
                    print(f"  Max Drawdown: {highest_return.avg_max_drawdown:.2%}")
                    print()

                # Lowest risk
                lowest_risk = min(summaries, key=lambda x: x.avg_max_drawdown)
                if lowest_risk != best:
                    print(f"Lowest Risk: {lowest_risk.parameter_set.name}")
                    print(f"  Return: {lowest_risk.avg_return:.2%}")
                    print(f"  Max Drawdown: {lowest_risk.avg_max_drawdown:.2%}")
                    print()

            print()

    # Step 4: Generate actionable recommendations
    print("💡 Step 4: Implementation Recommendations")
    print("-" * 45)

    if optimization_results:
        print("🚀 IMPLEMENTATION GUIDE:")
        print("=" * 25)

        for strategy_type, results in optimization_results.items():
            best = results["best_config"]

            print(
                f"\n{strategy_type.replace('_', ' ').title()} Strategy Implementation:"
            )
            print(
                f"1. Update your strategy configuration to use: {best.parameter_set.name}"
            )
            print(f"2. Set stop loss to {best.parameter_set.stop_loss_pct:.1%}")
            print(f"3. Set take profit to {best.parameter_set.take_profit_pct:.1%}")
            print(
                f"4. Use minimum confidence threshold of {best.parameter_set.min_confidence:.1f}%"
            )
            print(
                f"5. Limit position sizes to {best.parameter_set.max_position_size:.1%} of portfolio"
            )

            # Risk assessment
            if best.avg_return > 0:
                if best.periods_profitable >= best.periods_tested * 0.6:
                    risk_level = "LOW"
                    recommendation = "RECOMMENDED for live trading"
                elif best.periods_profitable >= best.periods_tested * 0.4:
                    risk_level = "MODERATE"
                    recommendation = "Consider paper trading first"
                else:
                    risk_level = "HIGH"
                    recommendation = "Further optimization needed"
            else:
                risk_level = "VERY HIGH"
                recommendation = "NOT RECOMMENDED - needs major improvements"

            print(f"6. Risk Level: {risk_level}")
            print(f"7. Recommendation: {recommendation}")

        print("\n🔄 MONITORING & ADJUSTMENT:")
        print("=" * 30)
        print("1. Monitor live performance against backtest expectations")
        print("2. Re-run optimization monthly with fresh data")
        print("3. Consider market regime changes that might affect parameters")
        print("4. Track parameter drift and performance degradation")
        print("5. Maintain position sizing discipline")

        print("\n⚠️  IMPORTANT CONSIDERATIONS:")
        print("=" * 30)
        print("1. Past performance does not guarantee future results")
        print("2. Market conditions change - parameters may need adjustment")
        print("3. Always start with paper trading to validate results")
        print("4. Consider transaction costs in real trading")
        print("5. Maintain proper risk management regardless of optimization results")

    else:
        print("❌ No successful optimizations completed")
        print("Consider:")
        print("1. Checking data availability for the test periods")
        print("2. Reducing the optimization complexity")
        print("3. Adjusting the parameter ranges")
        print("4. Using fewer test periods initially")

    # Step 5: Save comprehensive analysis
    print("\n💾 Step 5: Saving Analysis Results")
    print("-" * 35)

    try:
        # Try to run the analysis script
        analyzer = OptimizationAnalyzer()
        analyzer.load_results(directory="optimization_results")

        if analyzer.results_data:
            # Generate comprehensive report
            analyzer.generate_report("comprehensive_optimization_report.txt")
            print(
                "✅ Comprehensive analysis report saved: comprehensive_optimization_report.txt"
            )

            # Generate plots if possible
            try:
                analyzer.generate_plots("optimization_plots")
                print("✅ Analysis plots saved to: optimization_plots/")
            except Exception as e:
                print(f"⚠️  Plot generation skipped: {e}")
        else:
            print("⚠️  No optimization results found for detailed analysis")

    except Exception as e:
        print(f"⚠️  Advanced analysis skipped: {e}")

    # Final summary
    print("\n✨ OPTIMIZATION EXAMPLE COMPLETED")
    print("=" * 40)
    print("Results and recommendations have been generated!")
    print()
    print("📁 Files created:")
    print("   - optimization_results/ (detailed CSV and JSON files)")
    print("   - comprehensive_optimization_report.txt")
    print("   - optimization_plots/ (visualization charts)")
    print("   - optimization_example.log (execution log)")
    print()
    print("🎯 Next steps:")
    print("   1. Review the comprehensive report")
    print("   2. Examine the parameter recommendations")
    print("   3. Consider paper trading with optimized settings")
    print("   4. Implement gradual parameter changes in live trading")
    print("   5. Set up regular optimization schedule (monthly)")


async def main():
    """Main entry point."""
    try:
        await run_optimization_example()
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Example interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        logger.error(f"Example execution error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    print("Starting full parameter optimization example...")
    print("This may take several minutes to complete.")
    print()

    # Check if running from correct directory
    if not Path("backtesting").exists():
        print("❌ Please run this script from the project root directory")
        print(
            "   Expected structure: full-ai-trader/examples/run_full_optimization_example.py"
        )
        sys.exit(1)

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
