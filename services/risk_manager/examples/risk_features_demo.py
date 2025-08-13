#!/usr/bin/env python3
"""
Risk Calculator Features Demonstration

This script demonstrates the advanced risk management features added to the RiskCalculator,
including liquidity risk, VaR backtesting, risk-adjusted returns, options Greeks,
enhanced stress testing, and comprehensive risk reporting.

TODO: Add comprehensive tests for all demonstrated features
"""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List

# Import the risk calculator and related models
from src.risk_calculator import RiskCalculator
from shared.models import Position, PortfolioState


def create_sample_portfolio() -> PortfolioState:
    """Create a sample portfolio for demonstration purposes."""
    positions = [
        Position(
            symbol="AAPL",
            quantity=100,
            current_price=Decimal("175.50"),
            market_value=Decimal("17550.00")
        ),
        Position(
            symbol="MSFT",
            quantity=75,
            current_price=Decimal("380.00"),
            market_value=Decimal("28500.00")
        ),
        Position(
            symbol="GOOGL",
            quantity=25,
            current_price=Decimal("140.00"),
            market_value=Decimal("3500.00")
        ),
        Position(
            symbol="TSLA",
            quantity=50,
            current_price=Decimal("220.00"),
            market_value=Decimal("11000.00")
        ),
        Position(
            symbol="JPM",
            quantity=150,
            current_price=Decimal("165.00"),
            market_value=Decimal("24750.00")
        ),
        Position(
            symbol="SPY",
            quantity=100,
            current_price=Decimal("450.00"),
            market_value=Decimal("45000.00")
        )
    ]

    return PortfolioState(
        timestamp=datetime.now(timezone.utc),
        cash_balance=Decimal("30000.00"),
        positions=positions,
        total_equity=Decimal("160300.00")
    )


def create_sample_options_positions() -> List[Dict]:
    """Create sample options positions for Greeks demonstration."""
    return [
        {
            "symbol": "AAPL_CALL_180_30D",
            "quantity": 10,
            "option_type": "call",
            "strike": 180.0,
            "underlying_price": 175.5,
            "days_to_expiry": 30,
            "implied_volatility": 0.25,
            "risk_free_rate": 0.05
        },
        {
            "symbol": "TSLA_PUT_200_45D",
            "quantity": 5,
            "option_type": "put",
            "strike": 200.0,
            "underlying_price": 220.0,
            "days_to_expiry": 45,
            "implied_volatility": 0.40,
            "risk_free_rate": 0.05
        }
    ]


def create_sample_portfolio_history() -> List[Dict]:
    """Create sample portfolio history for VaR backtesting."""
    # Simulate 30 days of portfolio values with some volatility
    base_value = 160000
    history = []

    import random
    random.seed(42)  # For reproducible results

    for i in range(30):
        # Add some random daily returns
        daily_return = random.gauss(0.001, 0.02)  # 0.1% mean, 2% daily vol
        current_value = base_value * (1 + daily_return)
        base_value = current_value

        history.append({
            "date": f"2024-01-{i+1:02d}",
            "total_equity": current_value
        })

    return history


async def demonstrate_liquidity_risk(calculator: RiskCalculator, portfolio: PortfolioState):
    """Demonstrate liquidity risk assessment."""
    print("\n" + "="*60)
    print("LIQUIDITY RISK ASSESSMENT")
    print("="*60)

    liquidity_metrics = await calculator.calculate_liquidity_risk(portfolio)

    print(f"Portfolio Liquidity Score: {liquidity_metrics.get('portfolio_liquidity_score', 0):.3f}")
    print(f"Concentration Risk (largest position): {liquidity_metrics.get('concentration_risk', 0):.1%}")

    print("\nPosition Liquidity Breakdown:")
    for symbol, data in liquidity_metrics.get("position_liquidity", {}).items():
        print(f"  {symbol}: Score {data['score']:.2f}, Weight {data['weight']:.1%}")

    illiquid_positions = liquidity_metrics.get("illiquid_positions", [])
    if illiquid_positions:
        print(f"\nIlliquid Positions ({len(illiquid_positions)}):")
        for pos in illiquid_positions:
            print(f"  {pos['symbol']}: {pos['weight']:.1%} weight, {pos['liquidity_score']:.2f} score")
    else:
        print("\nNo significantly illiquid positions detected.")


async def demonstrate_var_backtesting(calculator: RiskCalculator):
    """Demonstrate VaR model backtesting."""
    print("\n" + "="*60)
    print("VAR MODEL BACKTESTING")
    print("="*60)

    # Create sample data for backtesting
    portfolio_history = create_sample_portfolio_history()

    # Simulate VaR predictions (normally these would come from your model)
    var_predictions = [3200.0] * (len(portfolio_history) - 1)  # 2% of portfolio value

    backtest_results = await calculator.backtest_var_model(
        portfolio_history, var_predictions, confidence_level=0.95
    )

    print(f"Total Observations: {backtest_results.get('total_observations', 0)}")
    print(f"VaR Violations: {backtest_results.get('violations', 0)}")
    print(f"Violation Rate: {backtest_results.get('violation_rate', 0):.1%}")
    print(f"Expected Rate: {backtest_results.get('expected_violation_rate', 0):.1%}")
    print(f"Model Adequate: {backtest_results.get('model_adequate', False)}")
    print(f"Kupiec Statistic: {backtest_results.get('kupiec_statistic', 0):.4f}")


async def demonstrate_risk_adjusted_returns(calculator: RiskCalculator, portfolio: PortfolioState):
    """Demonstrate risk-adjusted return calculations."""
    print("\n" + "="*60)
    print("RISK-ADJUSTED RETURNS")
    print("="*60)

    risk_adjusted = await calculator.calculate_risk_adjusted_returns(portfolio)

    if "error" not in risk_adjusted:
        print(f"Annualized Return: {risk_adjusted.get('mean_return', 0):.2%}")
        print(f"Annualized Volatility: {risk_adjusted.get('volatility', 0):.2%}")
        print(f"Sharpe Ratio: {risk_adjusted.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio: {risk_adjusted.get('sortino_ratio', 0):.3f}")
        print(f"Calmar Ratio: {risk_adjusted.get('calmar_ratio', 0):.3f}")
        print(f"Information Ratio: {risk_adjusted.get('information_ratio', 0):.3f}")
        print(f"Maximum Drawdown: {risk_adjusted.get('max_drawdown', 0):.2%}")
        print(f"Downside Volatility: {risk_adjusted.get('downside_volatility', 0):.2%}")
    else:
        print(f"Error: {risk_adjusted['error']}")


async def demonstrate_options_greeks(calculator: RiskCalculator):
    """Demonstrate options Greeks calculations."""
    print("\n" + "="*60)
    print("OPTIONS GREEKS ANALYSIS")
    print("="*60)

    options_positions = create_sample_options_positions()
    greeks = await calculator.calculate_options_greeks(options_positions)

    print("Portfolio Greeks:")
    print(f"  Total Delta: {greeks.get('portfolio_delta', 0):.2f}")
    print(f"  Total Gamma: {greeks.get('portfolio_gamma', 0):.4f}")
    print(f"  Total Theta: {greeks.get('portfolio_theta', 0):.2f} (per day)")
    print(f"  Total Vega: {greeks.get('portfolio_vega', 0):.2f} (per 1% vol)")

    print("\nPosition-Level Greeks:")
    for symbol, position_greeks in greeks.get("position_greeks", {}).items():
        print(f"  {symbol}:")
        print(f"    Delta: {position_greeks['delta']:.2f}")
        print(f"    Gamma: {position_greeks['gamma']:.4f}")
        print(f"    Theta: {position_greeks['theta']:.2f}")
        print(f"    Vega: {position_greeks['vega']:.2f}")


async def demonstrate_enhanced_stress_testing(calculator: RiskCalculator, portfolio: PortfolioState):
    """Demonstrate enhanced stress testing."""
    print("\n" + "="*60)
    print("ENHANCED STRESS TESTING")
    print("="*60)

    stress_results = await calculator.enhanced_stress_test(portfolio)

    # Show summary
    summary = stress_results.get("summary", {})
    print(f"Worst Case Scenario Return: {summary.get('worst_case_return', 0):.1f}%")
    print(f"Average Stress Return: {summary.get('average_stress_return', 0):.1f}%")
    print(f"Number of Scenarios: {summary.get('stress_scenarios_count', 0)}")

    print("\nScenario Results:")
    for scenario_name, result in stress_results.items():
        if scenario_name == "summary":
            continue

        print(f"\n  {scenario_name.replace('_', ' ').title()}:")
        print(f"    Description: {result.get('description', 'N/A')}")
        print(f"    Portfolio Return: {result.get('portfolio_return', 0):.1f}%")
        print(f"    Total P&L: ${result.get('total_pnl', 0):,.0f}")
        print(f"    Worst Position: {result.get('worst_position', 'N/A')}")
        print(f"    Positions at Risk: {result.get('positions_at_risk', 0)}")


async def demonstrate_risk_attribution(calculator: RiskCalculator, portfolio: PortfolioState):
    """Demonstrate risk attribution analysis."""
    print("\n" + "="*60)
    print("RISK ATTRIBUTION ANALYSIS")
    print("="*60)

    attribution = await calculator.calculate_risk_attribution(portfolio)

    print(f"Total Portfolio Risk: {attribution.get('total_portfolio_risk', 0):.2%}")

    print("\nPosition Risk Contributions:")
    for symbol, data in attribution.get("position_contributions", {}).items():
        print(f"  {symbol}: {data['risk_percentage']:.1f}% of total risk")
        print(f"    Weight: {data['weight']:.1%}, Volatility: {data['volatility']:.1%}")

    print("\nSector Risk Breakdown:")
    for sector, data in attribution.get("sector_risk", {}).items():
        print(f"  {sector}: {data['total_weight']:.1%} weight")
        print(f"    Positions: {', '.join(data['positions'])}")

    print("\nFactor Contributions:")
    factors = attribution.get("factor_contributions", {})
    print(f"  Market Beta: {factors.get('market_beta', 0):.2f}")
    print(f"  Concentration Risk: {factors.get('concentration_risk', 0):.1%}")
    print(f"  Sector Concentration: {factors.get('sector_concentration', 0):.1%}")


async def demonstrate_comprehensive_report(calculator: RiskCalculator, portfolio: PortfolioState):
    """Demonstrate comprehensive risk report generation."""
    print("\n" + "="*60)
    print("COMPREHENSIVE RISK REPORT")
    print("="*60)

    report = await calculator.generate_comprehensive_risk_report(portfolio)

    # Show key highlights
    portfolio_summary = report.get("portfolio_summary", {})
    print(f"Portfolio Value: ${portfolio_summary.get('total_value', 0):,.2f}")
    print(f"Number of Positions: {portfolio_summary.get('num_positions', 0)}")

    # VaR Analysis
    var_analysis = report.get("risk_metrics", {}).get("var_analysis", {})
    print(f"\n95% VaR: ${var_analysis.get('var_95_percent', 0):,.2f} ({var_analysis.get('var_as_percent_of_portfolio', {}).get('95_percent', 0):.1f}% of portfolio)")
    print(f"99% VaR: ${var_analysis.get('var_99_percent', 0):,.2f} ({var_analysis.get('var_as_percent_of_portfolio', {}).get('99_percent', 0):.1f}% of portfolio)")

    # Risk Warnings
    warnings = report.get("warnings", [])
    if warnings:
        print(f"\nRisk Warnings ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    # Performance metrics
    performance = report.get("performance_metrics", {})
    if performance and "error" not in performance:
        print(f"\nRisk-Adjusted Performance:")
        print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio: {performance.get('sortino_ratio', 0):.3f}")

    # Save full report to file for detailed analysis
    try:
        with open("risk_report_sample.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: risk_report_sample.json")
    except Exception as e:
        print(f"\nCould not save report: {e}")


async def main():
    """Main demonstration function."""
    print("="*60)
    print("RISK CALCULATOR ADVANCED FEATURES DEMONSTRATION")
    print("="*60)
    print("\nThis demo showcases the enhanced risk management capabilities")
    print("including liquidity risk, VaR backtesting, options Greeks,")
    print("stress testing, and comprehensive risk reporting.")
    print("\nNOTE: This uses simulated data for demonstration purposes.")

    # Initialize risk calculator
    calculator = RiskCalculator()

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    print(f"\nSample Portfolio Overview:")
    print(f"Total Value: ${portfolio.total_equity:,.2f}")
    print(f"Cash: ${portfolio.cash_balance:,.2f}")
    print(f"Positions: {len([p for p in portfolio.positions if p.quantity != 0])}")

    # Demonstrate all features
    await demonstrate_liquidity_risk(calculator, portfolio)
    await demonstrate_var_backtesting(calculator)
    await demonstrate_risk_adjusted_returns(calculator, portfolio)
    await demonstrate_options_greeks(calculator)
    await demonstrate_enhanced_stress_testing(calculator, portfolio)
    await demonstrate_risk_attribution(calculator, portfolio)
    await demonstrate_comprehensive_report(calculator, portfolio)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nAll new risk management features have been demonstrated.")
    print("Check the generated 'risk_report_sample.json' for detailed analysis.")
    print("\nTODO: Implement comprehensive test coverage for all features")


if __name__ == "__main__":
    asyncio.run(main())
