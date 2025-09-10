#!/usr/bin/env python3
"""
Script to create Plane tasks for TODO comments found in the AI trader codebase.
Uses requests library for synchronous HTTP calls.
"""

import sys
import time
from typing import Dict

import requests

# List of tasks identified from TODO comments and improvement needs
TASKS = [
    {
        "title": "[Zed] Implement comprehensive testing for portfolio monitor enhanced features",
        "description": "Add comprehensive test suite for portfolio monitor new features including:\n- test_enhanced_portfolio_monitor.py\n- test_screener_data_integration()\n- test_historical_price_correlation()\n\nLocation: services/risk_manager/src/portfolio_monitor.py:46-50",
        "priority": "high",
    },
    {
        "title": "[Zed] Implement market regime detection for correlation adjustments",
        "description": "Implement market regime detection (bull/bear/crisis) for correlation adjustments. During crisis periods, correlations tend to increase.\n\nLocation: services/risk_manager/src/portfolio_monitor.py:1144-1148",
        "priority": "medium",
    },
    {
        "title": "[Zed] Add unit tests for position sector lookup functionality",
        "description": "Add test_get_position_sector() to verify sector lookup from screener data and fallback mechanisms.\n\nLocation: services/risk_manager/src/portfolio_monitor.py:1278-1288",
        "priority": "high",
    },
    {
        "title": "[Zed] Add unit tests for sync sector lookup and caching",
        "description": "Add test_get_position_sector_sync() to verify synchronous sector lookup and caching functionality.\n\nLocation: services/risk_manager/src/portfolio_monitor.py:1560-1570",
        "priority": "high",
    },
    {
        "title": "[Zed] Add tests for screener data loading and caching",
        "description": "Add test_load_screener_data() to verify screener data loading and caching from Parquet files.\n\nLocation: services/risk_manager/src/portfolio_monitor.py:1664-1674",
        "priority": "medium",
    },
    {
        "title": "[Zed] Add price correlation calculation tests",
        "description": "Add comprehensive tests for price correlation calculation:\n- test_calculate_price_correlation()\n- test_correlation_edge_cases() for insufficient data, identical symbols, invalid data\n\nLocation: services/risk_manager/src/portfolio_monitor.py:1703-1716",
        "priority": "high",
    },
    {
        "title": "[Zed] Add historical price data loading tests",
        "description": "Add test_load_historical_prices() to verify historical data loading functionality.\n\nLocation: services/risk_manager/src/portfolio_monitor.py:1757-1767",
        "priority": "medium",
    },
    {
        "title": "[Zed] Comprehensive unit tests for position sizing methods",
        "description": "Add comprehensive unit tests for all position sizing methods including edge cases validation.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "high",
    },
    {
        "title": "[Zed] Add integration tests with real market data for position sizing",
        "description": "Implement integration tests with real market data for position sizing calculations.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "medium",
    },
    {
        "title": "[Zed] Add performance benchmarking for position sizing calculations",
        "description": "Implement performance benchmarking to measure and optimize position sizing calculation speed.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "low",
    },
    {
        "title": "[Zed] Add machine learning models for dynamic position sizing",
        "description": "Integrate machine learning models to enable dynamic, adaptive position sizing based on market conditions.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "medium",
    },
    {
        "title": "[Zed] Add options and derivatives position sizing",
        "description": "Implement position sizing algorithms specifically designed for options and derivatives trading.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "low",
    },
    {
        "title": "[Zed] Add currency hedging considerations for international positions",
        "description": "Implement currency hedging logic for international positions to manage FX risk.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "medium",
    },
    {
        "title": "[Zed] Add ESG factor adjustments to position sizing",
        "description": "Integrate Environmental, Social, and Governance (ESG) factors into position sizing decisions.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "low",
    },
    {
        "title": "[Zed] Add real-time market sentiment analysis integration",
        "description": "Integrate real-time market sentiment analysis to influence position sizing decisions.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "medium",
    },
    {
        "title": "[Zed] Add stress testing capabilities for extreme market scenarios",
        "description": "Implement stress testing to validate position sizing under extreme market scenarios.\n\nLocation: services/risk_manager/src/position_sizer.py:1613-1625",
        "priority": "high",
    },
    {
        "title": "[Zed] Add comprehensive tests for liquidity risk calculations",
        "description": "Implement comprehensive tests for liquidity risk calculation functionality.\n\nLocation: services/risk_manager/src/risk_calculator.py:282-292",
        "priority": "high",
    },
    {
        "title": "[Zed] Integrate real market data for liquidity scoring",
        "description": "Replace simple liquidity scoring with real market data including bid-ask spreads, volume, and other liquidity indicators.\n\nLocation: services/risk_manager/src/risk_calculator.py:304-314",
        "priority": "high",
    },
    {
        "title": "[Zed] Add comprehensive tests for position liquidity scoring",
        "description": "Add comprehensive tests for position liquidity scoring functionality including edge cases.\n\nLocation: services/risk_manager/src/risk_calculator.py:346-356",
        "priority": "high",
    },
    {
        "title": "[Zed] Add comprehensive tests for VaR backtesting",
        "description": "Implement comprehensive tests for Value at Risk backtesting model validation.\n\nLocation: services/risk_manager/src/risk_calculator.py:369-373",
        "priority": "high",
    },
    {
        "title": "[Zed] Add comprehensive tests for risk-adjusted returns",
        "description": "Implement comprehensive tests for risk-adjusted returns calculation.\n\nLocation: services/risk_manager/src/risk_calculator.py:432-436",
        "priority": "medium",
    },
    {
        "title": "[Zed] Add comprehensive tests for options Greeks calculations",
        "description": "Implement comprehensive tests for options Greeks calculations including Black-Scholes model validation.\n\nLocation: services/risk_manager/src/risk_calculator.py:489-499",
        "priority": "medium",
    },
    {
        "title": "[Zed] Implement proper JWT token verification for production",
        "description": "Replace simplified token verification with proper JWT signature and claims verification for production use.\n\nLocation: services/export_service/src/main.py:744-748",
        "priority": "high",
    },
    {
        "title": "[Zed] Implement proper API token verification for production",
        "description": "Replace simplified token verification with proper API token verification mechanism for production use.\n\nLocation: services/maintenance_service/src/main.py:549-553",
        "priority": "high",
    },
    {
        "title": "[Zed] Implement proper timezone handling for trading sessions",
        "description": "Replace simplified timezone conversion with proper timezone handling and account for daylight saving time changes.\n\nLocation: services/risk_manager/src/alpaca_client.py:848-858",
        "priority": "medium",
    },
    {
        "title": "[Zed] Implement FIFO/LIFO accounting for P&L calculations",
        "description": "Replace simple P&L calculation with proper FIFO/LIFO accounting methods for accurate profit/loss tracking.\n\nLocation: services/risk_manager/src/alpaca_client.py:986-996",
        "priority": "high",
    },
    {
        "title": "[Zed] Implement full covariance matrix for portfolio volatility",
        "description": "Replace simple weighted average approach with full covariance matrix calculation for accurate portfolio volatility.\n\nLocation: services/risk_manager/src/portfolio_monitor.py:622-632",
        "priority": "medium",
    },
    {
        "title": "[Zed] Integrate historical simulation or Monte Carlo for VaR",
        "description": "Replace simplified VaR calculation with proper historical simulation or Monte Carlo methods.\n\nLocation: services/risk_manager/src/risk_manager.py:908-918",
        "priority": "high",
    },
    {
        "title": "[Zed] Implement proper market holidays handling",
        "description": "Add proper market holidays checking to trading day calculations instead of simple weekday logic.\n\nLocation: tests/unit/test_data_collector.py:791-794",
        "priority": "medium",
    },
    {
        "title": "[Zed] Implement full covariance matrix for volatility contribution",
        "description": "Replace simplified volatility contribution with full covariance matrix and marginal contribution formula.\n\nLocation: services/risk_manager/src/risk_calculator.py:1397-1407",
        "priority": "medium",
    },
]


def create_plane_task(
    api_key: str, workspace: str, project_id: str, plane_instance: str, task: Dict
) -> bool:
    """Create a single task in Plane using requests."""
    url = (
        f"{plane_instance}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/"
    )

    headers = {"Content-Type": "application/json", "x-api-key": api_key}

    payload = {
        "name": task["title"],
        "description": task["description"],
        "state": "c39189ef-7e1a-4272-a566-c54fe5680fe7",
        "priority": task["priority"],
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 201:
            print(f"✅ Created task: {task['title']}")
            return True
        else:
            print(
                f"❌ Failed to create task '{task['title']}': {response.status_code} - {response.text}"
            )
            return False
    except Exception as e:
        print(f"❌ Error creating task '{task['title']}': {str(e)}")
        return False


def main() -> bool:
    """Main function to create all Plane tasks."""
    api_token = ""
    workspace = ""
    project_id = ""
    plane_instance = ""

    print(f"Creating {len(TASKS)} tasks in Plane...")
    print(f"Workspace: {workspace}")
    print(f"Project ID: {project_id}")
    print(f"API Token: {api_token[:10]}...")
    print()

    successful_tasks = 0
    failed_tasks = 0

    for i, task in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] Creating task: {task['title'][:60]}...")

        success = create_plane_task(
            api_token, workspace, project_id, plane_instance, task
        )
        if success:
            successful_tasks += 1
        else:
            failed_tasks += 1

        # Small delay to be respectful to the API
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("Task creation completed!")
    print(f"✅ Successful: {successful_tasks}")
    print(f"❌ Failed: {failed_tasks}")
    print(f"📊 Total: {len(TASKS)}")

    return successful_tasks > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
