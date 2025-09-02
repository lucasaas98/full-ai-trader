#!/usr/bin/env python3
"""
Simple Trading CLI Tool
A straightforward command-line interface for interacting with the trading system.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import requests


class TradingCLI:
    """Simple CLI for trading system interaction."""

    def __init__(self):
        self.data_collector_url = "http://localhost:9101"
        self.strategy_engine_url = "http://localhost:9102"
        self.risk_manager_url = "http://localhost:9103"
        self.trade_executor_url = "http://localhost:9104"
        self.scheduler_url = "http://localhost:9105"

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services."""
        services = {
            "data_collector": self.data_collector_url,
            "strategy_engine": self.strategy_engine_url,
            "risk_manager": self.risk_manager_url,
            "trade_executor": self.trade_executor_url,
            "scheduler": self.scheduler_url,
        }

        status = {}
        for service_name, url in services.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    status[service_name] = {
                        "status": "healthy",
                        "data": response.json(),
                    }
                else:
                    status[service_name] = {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                    }
            except Exception as e:
                status[service_name] = {"status": "error", "error": str(e)}

        return status

    def get_account_info(self) -> Dict[str, Any]:
        """Get trading account information."""
        try:
            response = requests.get(f"{self.trade_executor_url}/account", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        try:
            response = requests.get(f"{self.trade_executor_url}/positions", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_active_orders(self) -> Dict[str, Any]:
        """Get active orders."""
        try:
            response = requests.get(
                f"{self.trade_executor_url}/orders/active", timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_strategies(self) -> Dict[str, Any]:
        """Get available trading strategies."""
        try:
            response = requests.get(
                f"{self.strategy_engine_url}/strategies", timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def create_trade_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_name: str = "manual",
    ) -> Dict[str, Any]:
        """Create and execute a trade signal."""
        signal_data = {
            "id": str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "signal_type": signal_type.lower(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "confidence": confidence,
            "strategy_name": strategy_name,
            "metadata": {"created_by": "simple_cli", "manual_trade": True},
        }

        if price is not None:
            signal_data["price"] = price
        if quantity is not None:
            signal_data["quantity"] = quantity
        if stop_loss is not None:
            signal_data["stop_loss"] = stop_loss
        if take_profit is not None:
            signal_data["take_profit"] = take_profit

        try:
            response = requests.post(
                f"{self.trade_executor_url}/signals/execute",
                json=signal_data,
                timeout=30,
            )
            response.raise_for_status()
            return {
                "status": "success",
                "signal_id": signal_data["id"],
                "response": response.json(),
            }
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "signal_id": signal_data["id"],
                "message": "Signal submitted but response timed out",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an active order."""
        try:
            response = requests.post(
                f"{self.trade_executor_url}/orders/{order_id}/cancel", timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_performance(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            response = requests.get(
                f"{self.trade_executor_url}/performance/summary", timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def format_health_status(status: Dict[str, Any]) -> None:
    """Format and print health status."""
    print("=== SYSTEM HEALTH STATUS ===")
    for service, info in status.items():
        status_emoji = "✅" if info["status"] == "healthy" else "❌"
        print(f"{status_emoji} {service.replace('_', ' ').title()}: {info['status']}")
        if info["status"] != "healthy" and "error" in info:
            print(f"   Error: {info['error']}")
    print()


def format_account_info(account: Dict[str, Any]) -> None:
    """Format and print account information."""
    if "error" in account:
        print(f"❌ Error getting account info: {account['error']}")
        return

    print("=== ACCOUNT INFORMATION ===")
    acc_data = account.get("account", {})
    print(f"💰 Equity: ${acc_data.get('equity', 0):,.2f}")
    print(f"💵 Cash: ${acc_data.get('cash', 0):,.2f}")
    print(f"🔋 Buying Power: ${acc_data.get('buying_power', 0):,.2f}")
    print(f"📊 Day Trades: {acc_data.get('day_trades_count', 0)}")
    print(
        f"🏷️  PDT Status: {'Yes' if acc_data.get('pattern_day_trader', False) else 'No'}"
    )

    positions = account.get("positions", {})
    print(f"📈 Active Positions: {positions.get('count', 0)}")
    print(f"💹 Total Market Value: ${positions.get('total_market_value', 0):,.2f}")
    print(f"📊 Unrealized P&L: ${positions.get('total_unrealized_pnl', 0):,.2f}")
    print()


def format_positions(positions: Dict[str, Any]) -> None:
    """Format and print positions."""
    if "error" in positions:
        print(f"❌ Error getting positions: {positions['error']}")
        return

    print("=== CURRENT POSITIONS ===")
    if positions.get("count", 0) == 0:
        print("📭 No active positions")
    else:
        for pos in positions.get("positions", []):
            print(
                f"📈 {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}"
            )
            print(
                f"   Current: ${pos['current_price']:.2f} | P&L: ${pos['unrealized_pnl']:+.2f}"
            )
    print()


def format_orders(orders: Dict[str, Any]) -> None:
    """Format and print active orders."""
    if "error" in orders:
        print(f"❌ Error getting orders: {orders['error']}")
        return

    print("=== ACTIVE ORDERS ===")
    if orders.get("count", 0) == 0:
        print("📭 No active orders")
    else:
        for order in orders.get("orders", []):
            side_emoji = "🟢" if order["side"] == "buy" else "🔴"
            print(
                f"{side_emoji} {order['symbol']} {order['side'].upper()} {order['quantity']} @ {order['order_type']}"
            )
            print(
                f"   Status: {order['status']} | Submitted: {order['submitted_at'][:19]}"
            )
    print()


def format_strategies(strategies: Dict[str, Any]) -> None:
    """Format and print available strategies."""
    if "error" in strategies:
        print(f"❌ Error getting strategies: {strategies['error']}")
        return

    print("=== AVAILABLE STRATEGIES ===")
    for strategy in strategies.get("strategies", []):
        print(f"🎯 {strategy['name']} ({strategy['mode']})")
        info = strategy.get("info", {})
        print(
            f"   Confidence: {info.get('min_confidence', 0)}% | Position Size: {info.get('max_position_size', 0)*100:.0f}%"
        )
        print(
            f"   Stop Loss: {info.get('stop_loss_pct', 0)*100:.1f}% | Take Profit: {info.get('take_profit_pct', 0)*100:.1f}%"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Simple Trading System CLI")
    parser.add_argument(
        "command",
        choices=[
            "status",
            "account",
            "positions",
            "orders",
            "strategies",
            "trade",
            "cancel",
            "performance",
        ],
        help="Command to execute",
    )

    # Trade command arguments
    parser.add_argument("--symbol", help="Trading symbol (e.g., AAPL)")
    parser.add_argument(
        "--action", choices=["buy", "sell", "close"], help="Trade action"
    )
    parser.add_argument("--quantity", type=int, help="Number of shares")
    parser.add_argument("--price", type=float, help="Limit price (optional)")
    parser.add_argument(
        "--confidence", type=float, default=0.8, help="Signal confidence (0.0-1.0)"
    )
    parser.add_argument("--stop-loss", type=float, help="Stop loss price")
    parser.add_argument("--take-profit", type=float, help="Take profit price")
    parser.add_argument("--strategy", default="manual", help="Strategy name")

    # Cancel command arguments
    parser.add_argument("--order-id", help="Order ID to cancel")

    args = parser.parse_args()

    cli = TradingCLI()

    if args.command == "status":
        status = cli.get_health_status()
        format_health_status(status)

    elif args.command == "account":
        account = cli.get_account_info()
        format_account_info(account)

    elif args.command == "positions":
        positions = cli.get_positions()
        format_positions(positions)

    elif args.command == "orders":
        orders = cli.get_active_orders()
        format_orders(orders)

    elif args.command == "strategies":
        strategies = cli.get_strategies()
        format_strategies(strategies)

    elif args.command == "performance":
        performance = cli.get_performance()
        print("=== PERFORMANCE SUMMARY ===")
        print(json.dumps(performance, indent=2))
        print()

    elif args.command == "trade":
        if not args.symbol or not args.action:
            print("❌ Error: --symbol and --action are required for trade command")
            sys.exit(1)

        result = cli.create_trade_signal(
            symbol=args.symbol,
            signal_type=args.action,
            confidence=args.confidence,
            price=args.price,
            quantity=args.quantity,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            strategy_name=args.strategy,
        )

        if result["status"] == "success":
            print(f"✅ Trade signal submitted successfully!")
            print(f"   Signal ID: {result['signal_id']}")
        elif result["status"] == "timeout":
            print(f"⏳ Trade signal submitted but response timed out")
            print(f"   Signal ID: {result['signal_id']}")
            print("   Check orders to see if it was processed")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

    elif args.command == "cancel":
        if not args.order_id:
            print("❌ Error: --order-id is required for cancel command")
            sys.exit(1)

        result = cli.cancel_order(args.order_id)
        if "error" in result:
            print(f"❌ Error canceling order: {result['error']}")
        else:
            print("✅ Order cancellation requested")


if __name__ == "__main__":
    main()
