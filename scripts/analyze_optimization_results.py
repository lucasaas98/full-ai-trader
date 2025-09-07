#!/usr/bin/env python3
"""
Optimization Results Analysis Script

This script analyzes the results from parameter optimization runs to provide
insights, recommendations, and statistical analysis of trading strategy performance
across different parameter combinations and market conditions.

Features:
- Load and analyze optimization results from multiple runs
- Statistical analysis of parameter sensitivity
- Market condition correlation analysis
- Performance consistency evaluation
- Parameter recommendation generation
- Comparative analysis across strategies
- Export detailed reports and visualizations

Usage:
    python scripts/analyze_optimization_results.py [options]

Examples:
    # Analyze single strategy results
    python scripts/analyze_optimization_results.py --file optimization_results_day_trading_*.json

    # Compare multiple strategies
    python scripts/analyze_optimization_results.py --dir optimization_results/ --compare

    # Generate detailed report with visualizations
    python scripts/analyze_optimization_results.py --dir optimization_results/ --detailed --plots
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class OptimizationAnalyzer:
    """Main class for analyzing optimization results."""

    def __init__(self):
        self.results_data = {}  # strategy -> results
        self.detailed_data = {}  # strategy -> detailed results DataFrame
        self.summary_data = {}  # strategy -> summary DataFrame

    def load_results(
        self, file_path: Optional[str] = None, directory: Optional[str] = None
    ) -> None:
        """Load optimization results from files."""

        files_to_load = []

        if file_path:
            files_to_load.append(Path(file_path))
        elif directory:
            results_dir = Path(directory)
            if results_dir.exists():
                # Find all JSON result files
                json_files = list(results_dir.glob("optimization_results_*.json"))
                files_to_load.extend(json_files)

        if not files_to_load:
            raise ValueError("No result files found")

        print(f"Loading {len(files_to_load)} result files...")

        for file_path_obj in files_to_load:
            try:
                with open(file_path_obj, "r") as f:
                    data = json.load(f)

                strategy_type = data.get("strategy_type")
                if strategy_type:
                    self.results_data[strategy_type] = data
                    print(f"  ✓ Loaded {strategy_type} results")

                    # Also try to load corresponding detailed CSV
                    csv_pattern = f"optimization_detailed_{strategy_type}_*.csv"
                    csv_files = list(file_path_obj.parent.glob(csv_pattern))
                    if csv_files:
                        detailed_df = pd.read_csv(csv_files[0])
                        self.detailed_data[strategy_type] = detailed_df

                    # Load summary CSV
                    summary_pattern = f"optimization_summary_{strategy_type}_*.csv"
                    summary_files = list(file_path_obj.parent.glob(summary_pattern))
                    if summary_files:
                        summary_df = pd.read_csv(summary_files[0])
                        self.summary_data[strategy_type] = summary_df

            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {e}")

    def analyze_parameter_sensitivity(self, strategy: str) -> Dict[str, Any]:
        """Analyze how sensitive performance is to each parameter."""

        if strategy not in self.summary_data:
            return {}

        df = self.summary_data[strategy]

        # Parameters to analyze
        param_columns = [
            "stop_loss_pct",
            "take_profit_pct",
            "min_confidence",
            "max_position_size",
            "ta_weight",
            "volume_threshold",
        ]

        sensitivity_analysis = {}

        for param in param_columns:
            if param in df.columns:
                # Calculate correlation with performance metrics
                correlations = {}

                if "avg_return" in df.columns:
                    corr_return = df[param].corr(df["avg_return"])
                    correlations["avg_return"] = corr_return

                if "risk_adjusted_score" in df.columns:
                    corr_risk_adj = df[param].corr(df["risk_adjusted_score"])
                    correlations["risk_adjusted_score"] = corr_risk_adj

                if "avg_win_rate" in df.columns:
                    corr_win_rate = df[param].corr(df["avg_win_rate"])
                    correlations["win_rate"] = corr_win_rate

                # Calculate parameter impact (range of performance across parameter values)
                if "avg_return" in df.columns:
                    param_groups = df.groupby(pd.cut(df[param], bins=3))["avg_return"]
                    param_impact = param_groups.max().max() - param_groups.min().min()
                else:
                    param_impact = 0

                sensitivity_analysis[param] = {
                    "correlations": correlations,
                    "impact_range": param_impact,
                    "optimal_value": self._find_optimal_parameter_value(df, param),
                }

        return sensitivity_analysis

    def _find_optimal_parameter_value(self, df: pd.DataFrame, param: str) -> float:
        """Find the parameter value that gives the best average performance."""

        if "risk_adjusted_score" in df.columns:
            best_idx = df["risk_adjusted_score"].idxmax()
            param_value = df.loc[best_idx, param]
            if pd.notna(param_value):
                try:
                    return float(str(param_value))
                except (ValueError, TypeError):
                    return 0.0
            else:
                return 0.0
        elif "avg_return" in df.columns:
            best_idx = df["avg_return"].idxmax()
            param_value = df.loc[best_idx, param]
            if pd.notna(param_value):
                try:
                    return float(str(param_value))
                except (ValueError, TypeError):
                    return 0.0
            else:
                return 0.0
        else:
            mean_value = df[param].mean()
            return float(mean_value) if pd.notna(mean_value) else 0.0

    def analyze_market_conditions(self, strategy: str) -> Dict[str, Any]:
        """Analyze performance across different market conditions."""

        if strategy not in self.detailed_data:
            return {}

        df = self.detailed_data[strategy]

        # Parse period information
        df["period_start"] = pd.to_datetime(df["period_start"])
        df["period_end"] = pd.to_datetime(df["period_end"])
        df["period_month"] = df["period_start"].dt.month
        df["period_year"] = df["period_start"].dt.year

        analysis = {}

        # Performance by month
        monthly_perf = (
            df.groupby("period_month")
            .agg(
                {
                    "total_return": ["mean", "std", "count"],
                    "max_drawdown": "mean",
                    "win_rate": "mean",
                }
            )
            .round(4)
        )

        analysis["monthly_performance"] = monthly_perf

        # Performance by year
        yearly_perf = (
            df.groupby("period_year")
            .agg(
                {
                    "total_return": ["mean", "std", "count"],
                    "max_drawdown": "mean",
                    "win_rate": "mean",
                }
            )
            .round(4)
        )

        analysis["yearly_performance"] = yearly_perf

        # Identify best/worst performing periods
        best_periods = df.nlargest(3, "total_return")[
            ["period_start", "period_end", "total_return", "parameter_set"]
        ]
        worst_periods = df.nsmallest(3, "total_return")[
            ["period_start", "period_end", "total_return", "parameter_set"]
        ]

        analysis["best_periods"] = best_periods
        analysis["worst_periods"] = worst_periods

        return analysis

    def generate_recommendations(self, strategy: str) -> Dict[str, Any]:
        """Generate parameter recommendations based on analysis."""

        recommendations: Dict[str, Any] = {}

        if strategy not in self.summary_data:
            return recommendations

        df = self.summary_data[strategy]

        # Find best overall parameter set
        if "risk_adjusted_score" in df.columns:
            best_config = df.loc[df["risk_adjusted_score"].idxmax()]
        elif "avg_return" in df.columns:
            best_config = df.loc[df["avg_return"].idxmax()]
        else:
            best_config = df.iloc[0]

        recommendations["best_overall"] = best_config.to_dict()

        # Find most consistent parameter set
        if "consistency_score" in df.columns and "avg_return" in df.columns:
            # Filter to positive average returns first, then find most consistent
            profitable_configs = df[df["avg_return"] > 0]
            if not profitable_configs.empty:
                most_consistent = profitable_configs.loc[
                    profitable_configs["consistency_score"].idxmax()
                ]
                recommendations["most_consistent"] = most_consistent.to_dict()

        # Find highest return parameter set (regardless of risk)
        if "avg_return" in df.columns:
            highest_return = df.loc[df["avg_return"].idxmax()]
            recommendations["highest_return"] = highest_return.to_dict()

        # Find lowest risk parameter set
        if "avg_max_drawdown" in df.columns:
            lowest_risk = df.loc[df["avg_max_drawdown"].idxmin()]
            recommendations["lowest_risk"] = lowest_risk.to_dict()

        # Parameter sensitivity recommendations
        sensitivity = self.analyze_parameter_sensitivity(strategy)
        param_recommendations = {}

        for param, analysis in sensitivity.items():
            param_recommendations[param] = {
                "optimal_value": analysis["optimal_value"],
                "sensitivity": analysis["impact_range"],
                "correlations": analysis["correlations"],
            }

        recommendations["parameter_sensitivity"] = param_recommendations

        return recommendations

    def compare_strategies(self) -> Dict[str, Any]:
        """Compare performance across different strategies."""

        if len(self.results_data) < 2:
            return {}

        comparison: Dict[str, Any] = {}
        strategy_summaries = {}

        # Collect best performance for each strategy
        for strategy, data in self.results_data.items():
            if "all_summaries" in data and data["all_summaries"]:
                best_summary = data["all_summaries"][
                    0
                ]  # Assuming sorted by performance
                strategy_summaries[strategy] = best_summary

        if not strategy_summaries:
            return comparison

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(strategy_summaries).T

        # Key metrics comparison
        key_metrics = [
            "avg_return",
            "avg_max_drawdown",
            "avg_win_rate",
            "consistency_score",
            "risk_adjusted_score",
        ]

        comparison["metrics_comparison"] = comparison_df[key_metrics].round(4)

        # Rank strategies by different criteria
        rankings = {}
        for metric in key_metrics:
            if metric in comparison_df.columns:
                if metric == "avg_max_drawdown":  # Lower is better
                    rankings[metric] = comparison_df[metric].rank(ascending=True)
                else:  # Higher is better
                    rankings[metric] = comparison_df[metric].rank(ascending=False)

        comparison["rankings"] = pd.DataFrame(rankings)

        # Overall winner by category
        winners = {}
        if "avg_return" in comparison_df.columns:
            winners["highest_return"] = comparison_df["avg_return"].idxmax()
        if "consistency_score" in comparison_df.columns:
            winners["most_consistent"] = comparison_df["consistency_score"].idxmax()
        if "avg_max_drawdown" in comparison_df.columns:
            winners["lowest_risk"] = comparison_df["avg_max_drawdown"].idxmin()
        if "risk_adjusted_score" in comparison_df.columns:
            winners["best_risk_adjusted"] = comparison_df[
                "risk_adjusted_score"
            ].idxmax()

        comparison["category_winners"] = winners

        return comparison

    def generate_plots(self, output_dir: str = "optimization_plots") -> None:
        """Generate visualization plots."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for strategy in self.results_data.keys():
            self._plot_strategy_analysis(strategy, output_path)

        # Strategy comparison plots
        if len(self.results_data) > 1:
            self._plot_strategy_comparison(output_path)

        print(f"Plots saved to: {output_path}/")

    def _plot_strategy_analysis(self, strategy: str, output_path: Path) -> None:
        """Generate plots for a single strategy."""

        if strategy not in self.summary_data:
            return

        df = self.summary_data[strategy]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f'{strategy.replace("_", " ").title()} Strategy Analysis', fontsize=16
        )

        # Plot 1: Return vs Risk
        if "avg_return" in df.columns and "avg_max_drawdown" in df.columns:
            axes[0, 0].scatter(df["avg_max_drawdown"], df["avg_return"], alpha=0.6)
            axes[0, 0].set_xlabel("Average Max Drawdown")
            axes[0, 0].set_ylabel("Average Return")
            axes[0, 0].set_title("Risk vs Return")
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Parameter sensitivity (example with stop_loss_pct)
        if "stop_loss_pct" in df.columns and "avg_return" in df.columns:
            axes[0, 1].scatter(df["stop_loss_pct"], df["avg_return"], alpha=0.6)
            axes[0, 1].set_xlabel("Stop Loss %")
            axes[0, 1].set_ylabel("Average Return")
            axes[0, 1].set_title("Stop Loss Sensitivity")
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Win Rate distribution
        if "avg_win_rate" in df.columns:
            axes[1, 0].hist(df["avg_win_rate"], bins=15, alpha=0.7, edgecolor="black")
            axes[1, 0].set_xlabel("Average Win Rate")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Win Rate Distribution")
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Risk-Adjusted Score vs Confidence
        if "min_confidence" in df.columns and "risk_adjusted_score" in df.columns:
            axes[1, 1].scatter(
                df["min_confidence"], df["risk_adjusted_score"], alpha=0.6
            )
            axes[1, 1].set_xlabel("Minimum Confidence")
            axes[1, 1].set_ylabel("Risk-Adjusted Score")
            axes[1, 1].set_title("Confidence vs Performance")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_path / f"{strategy}_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_strategy_comparison(self, output_path: Path) -> None:
        """Generate strategy comparison plots."""

        comparison = self.compare_strategies()
        if not comparison:
            return

        metrics_df = comparison["metrics_comparison"]

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Strategy Comparison", fontsize=16)

        # Plot 1: Performance metrics radar chart (simplified as bar chart)
        key_metrics = ["avg_return", "avg_win_rate", "consistency_score"]
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]

        if available_metrics:
            metrics_df[available_metrics].plot(kind="bar", ax=axes[0])
            axes[0].set_title("Performance Metrics Comparison")
            axes[0].set_ylabel("Score")
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Risk vs Return
        if (
            "avg_return" in metrics_df.columns
            and "avg_max_drawdown" in metrics_df.columns
        ):
            for strategy in metrics_df.index:
                axes[1].scatter(
                    metrics_df.loc[strategy, "avg_max_drawdown"],
                    metrics_df.loc[strategy, "avg_return"],
                    label=strategy.replace("_", " ").title(),
                    s=100,
                )

            axes[1].set_xlabel("Average Max Drawdown")
            axes[1].set_ylabel("Average Return")
            axes[1].set_title("Risk vs Return by Strategy")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_path / "strategy_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_report(
        self, output_file: str = "optimization_analysis_report.txt"
    ) -> None:
        """Generate a comprehensive text report."""

        with open(output_file, "w") as f:
            f.write("TRADING STRATEGY PARAMETER OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary of loaded data
            f.write("LOADED DATA SUMMARY\n")
            f.write("-" * 30 + "\n")
            for strategy in self.results_data.keys():
                data = self.results_data[strategy]
                f.write(f"Strategy: {strategy.replace('_', ' ').title()}\n")
                f.write(
                    f"  Optimization Date: {data.get('optimization_timestamp', 'Unknown')}\n"
                )
                f.write(f"  Periods Tested: {data.get('periods_tested', 'Unknown')}\n")
                f.write(
                    f"  Parameter Combinations: {data.get('parameter_combinations_tested', 'Unknown')}\n"
                )
            f.write("\n")

            # Analysis for each strategy
            for strategy in self.results_data.keys():
                f.write(f"{strategy.replace('_', ' ').title()} STRATEGY ANALYSIS\n")
                f.write("=" * 50 + "\n\n")

                # Best parameters
                recommendations = self.generate_recommendations(strategy)
                if "best_overall" in recommendations:
                    best = recommendations["best_overall"]
                    f.write("RECOMMENDED PARAMETERS (Best Overall Performance):\n")
                    f.write("-" * 45 + "\n")

                    param_mapping = {
                        "stop_loss_pct": "Stop Loss",
                        "take_profit_pct": "Take Profit",
                        "min_confidence": "Min Confidence",
                        "max_position_size": "Max Position Size",
                        "ta_weight": "Technical Analysis Weight",
                        "fa_weight": "Fundamental Analysis Weight",
                        "volume_threshold": "Volume Threshold",
                    }

                    for param, display_name in param_mapping.items():
                        if param in best:
                            value = best[param]
                            if "pct" in param or "weight" in param or "size" in param:
                                f.write(f"  {display_name}: {value:.1%}\n")
                            else:
                                f.write(f"  {display_name}: {value:.2f}\n")

                    f.write("\nExpected Performance:\n")
                    f.write(f"  Average Return: {best.get('avg_return', 0):.2%}\n")
                    f.write(
                        f"  Average Max Drawdown: {best.get('avg_max_drawdown', 0):.2%}\n"
                    )
                    f.write(f"  Win Rate: {best.get('avg_win_rate', 0):.1%}\n")
                    f.write(
                        f"  Risk-Adjusted Score: {best.get('risk_adjusted_score', 0):.3f}\n"
                    )

                # Parameter sensitivity analysis
                sensitivity = self.analyze_parameter_sensitivity(strategy)
                if sensitivity:
                    f.write("\nPARAMETER SENSITIVITY ANALYSIS:\n")
                    f.write("-" * 35 + "\n")

                    for param, analysis in sensitivity.items():
                        impact = analysis.get("impact_range", 0)
                        optimal = analysis.get("optimal_value", 0)
                        correlations = analysis.get("correlations", {})

                        f.write(f"  {param.replace('_', ' ').title()}:\n")
                        f.write(f"    Optimal Value: {optimal:.3f}\n")
                        f.write(f"    Performance Impact Range: {impact:.4f}\n")

                        if correlations:
                            f.write("    Correlations:\n")
                            for metric, corr in correlations.items():
                                f.write(f"      {metric}: {corr:.3f}\n")
                        f.write("\n")

                f.write("\n")

            # Strategy comparison
            if len(self.results_data) > 1:
                f.write("STRATEGY COMPARISON\n")
                f.write("=" * 30 + "\n\n")

                comparison = self.compare_strategies()
                if "category_winners" in comparison:
                    winners = comparison["category_winners"]
                    f.write("Category Winners:\n")
                    f.write("-" * 20 + "\n")
                    for category, strategy in winners.items():
                        f.write(
                            f"  {category.replace('_', ' ').title()}: {strategy.replace('_', ' ').title()}\n"
                        )
                    f.write("\n")

                if "metrics_comparison" in comparison:
                    metrics_df = comparison["metrics_comparison"]
                    f.write("Performance Comparison:\n")
                    f.write("-" * 25 + "\n")
                    f.write(metrics_df.to_string())
                    f.write("\n\n")

            # Recommendations
            f.write("FINAL RECOMMENDATIONS\n")
            f.write("=" * 25 + "\n\n")

            for strategy in self.results_data.keys():
                recommendations = self.generate_recommendations(strategy)
                if recommendations:
                    f.write(f"{strategy.replace('_', ' ').title()} Strategy:\n")

                    if "best_overall" in recommendations:
                        best = recommendations["best_overall"]
                        f.write(
                            f"  Use parameter set: {best.get('param_name', 'Unknown')}\n"
                        )
                        f.write(f"  Expected return: {best.get('avg_return', 0):.2%}\n")
                        f.write(
                            f"  Expected max drawdown: {best.get('avg_max_drawdown', 0):.2%}\n"
                        )

                    f.write("\n")

        print(f"Analysis report saved to: {output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze trading strategy parameter optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--file", type=str, help="Single optimization results JSON file to analyze"
    )
    input_group.add_argument(
        "--dir",
        type=str,
        default="optimization_results",
        help="Directory containing optimization result files (default: optimization_results)",
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed analysis including sensitivity analysis",
    )
    analysis_group.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple strategies (requires multiple result files)",
    )
    analysis_group.add_argument(
        "--plots", action="store_true", help="Generate visualization plots"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-report",
        type=str,
        default="optimization_analysis_report.txt",
        help="Output report filename (default: optimization_analysis_report.txt)",
    )
    output_group.add_argument(
        "--plot-dir",
        type=str,
        default="optimization_plots",
        help="Directory for plot outputs (default: optimization_plots)",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    try:
        # Create analyzer
        analyzer = OptimizationAnalyzer()

        # Load results
        print("🔍 OPTIMIZATION RESULTS ANALYSIS")
        print("=" * 50)

        if args.file:
            analyzer.load_results(file_path=args.file)
        else:
            analyzer.load_results(directory=args.dir)

        if not analyzer.results_data:
            print("❌ No optimization results found!")
            return 1

        print(f"✅ Loaded results for {len(analyzer.results_data)} strategies")

        # Generate analysis
        print("\n📊 Analyzing results...")

        # Strategy-specific analysis
        for strategy in analyzer.results_data.keys():
            print(f"\n🎯 Analyzing {strategy.replace('_', ' ').title()} Strategy:")

            if args.detailed:
                # Parameter sensitivity
                sensitivity = analyzer.analyze_parameter_sensitivity(strategy)
                if sensitivity:
                    print("  ✓ Parameter sensitivity analysis completed")

                # Market conditions analysis
                market_analysis = analyzer.analyze_market_conditions(strategy)
                if market_analysis:
                    print("  ✓ Market conditions analysis completed")

            # Generate recommendations
            recommendations = analyzer.generate_recommendations(strategy)
            if recommendations and "best_overall" in recommendations:
                best = recommendations["best_overall"]
                print(f"  📈 Best configuration: {best.get('param_name', 'Unknown')}")
                print(f"     Expected return: {best.get('avg_return', 0):.2%}")
                print(
                    f"     Risk-adjusted score: {best.get('risk_adjusted_score', 0):.3f}"
                )

        # Strategy comparison
        if args.compare and len(analyzer.results_data) > 1:
            print("\n⚖️  Comparing strategies...")
            comparison = analyzer.compare_strategies()
            if comparison and "category_winners" in comparison:
                winners = comparison["category_winners"]
                print("  🏆 Category winners:")
                for category, strategy in winners.items():
                    print(
                        f"     {category.replace('_', ' ').title()}: {strategy.replace('_', ' ').title()}"
                    )

        # Generate plots
        if args.plots:
            print("\n📈 Generating plots...")
            try:
                analyzer.generate_plots(args.plot_dir)
                print("  ✅ Plots generated successfully")
            except Exception as e:
                print(f"  ⚠️  Plot generation failed: {e}")

        # Generate report
        print("\n📝 Generating analysis report...")
        analyzer.generate_report(args.output_report)

        print("\n✅ Analysis completed!")
        print(f"📄 Report: {args.output_report}")
        if args.plots:
            print(f"📊 Plots: {args.plot_dir}/")

        return 0

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
