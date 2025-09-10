#!/usr/bin/env python3
"""
Backtesting Script for Automated Trading System

This script provides backtesting capabilities for trading strategies,
allowing users to test strategy performance on historical data.

Usage:
    python scripts/backtest.py --strategy moving_average --symbol AAPL --start 2023-01-01 --end 2023-12-31
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import uuid4

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.models import (  # noqa: E402
    BacktestResult,
    MarketData,
    OrderSide,
    Position,
    SignalType,
    TimeFrame,
    Trade,
    TradeSignal,
)
from shared.utils import format_currency, format_percentage  # noqa: E402


class SimpleBacktester:
    """Simple backtesting engine for trading strategies."""

    def __init__(self, initial_capital: Decimal = Decimal("100000")) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

    def process_signal(self, signal: TradeSignal, current_price: Decimal) -> bool:
        """Process a trade signal and execute if valid."""
        symbol = signal.symbol

        if signal.signal_type == SignalType.BUY:
            return self._execute_buy(symbol, current_price, signal.quantity or 100)
        elif signal.signal_type == SignalType.SELL:
            return self._execute_sell(symbol, current_price, signal.quantity or None)

        return False

    def _execute_buy(self, symbol: str, price: Decimal, quantity: int) -> bool:
        """Execute buy order."""
        cost = price * quantity

        if cost > self.cash:
            return False  # Insufficient funds

        self.cash -= cost

        if symbol in self.positions:
            # Average into existing position
            pos = self.positions[symbol]
            total_cost = (pos.entry_price * pos.quantity) + cost
            total_quantity = pos.quantity + quantity
            pos.entry_price = total_cost / total_quantity
            pos.quantity = total_quantity
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=Decimal("0"),
                market_value=cost,
                cost_basis=cost,
            )

        # Record trade
        trade = Trade(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            strategy_name="backtest",
            order_id=uuid4(),
            fees=Decimal("0"),
            pnl=None,
        )
        self.trades.append(trade)

        return True

    def _execute_sell(
        self, symbol: str, price: Decimal, quantity: Optional[int] = None
    ) -> bool:
        """Execute sell order."""
        if symbol not in self.positions:
            return False  # No position to sell

        pos = self.positions[symbol]
        sell_quantity = quantity or pos.quantity

        if sell_quantity > pos.quantity:
            sell_quantity = pos.quantity

        proceeds = price * sell_quantity
        self.cash += proceeds

        # Calculate PnL
        cost_basis = pos.entry_price * sell_quantity
        pnl = proceeds - cost_basis

        # Update position
        pos.quantity -= sell_quantity
        if pos.quantity == 0:
            del self.positions[symbol]

        # Record trade
        trade = Trade(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=sell_quantity,
            price=price,
            strategy_name="backtest",
            order_id=uuid4(),
            pnl=pnl,
            fees=Decimal("0"),
        )
        self.trades.append(trade)

        return True

    def update_positions(self, market_data: Dict[str, Decimal]) -> None:
        """Update position values with current market prices."""
        for symbol, pos in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                pos.current_price = current_price
                pos.market_value = current_price * pos.quantity
                pos.unrealized_pnl = pos.market_value - (pos.entry_price * pos.quantity)

    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value."""
        market_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + market_value

    def record_equity_point(self, timestamp: datetime) -> None:
        """Record equity curve point."""
        total_value = self.get_portfolio_value()
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "total_equity": total_value,
                "cash": self.cash,
                "positions_value": total_value - self.cash,
                "positions_count": len(self.positions),
            }
        )


class MovingAverageStrategy:
    """Simple moving average crossover strategy."""

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.price_history: Dict[str, List[Decimal]] = {}

    def generate_signal(self, symbol: str, price: Decimal) -> Optional[TradeSignal]:
        """Generate trading signal based on moving average crossover."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(price)

        # Keep only necessary history
        max_window = max(self.short_window, self.long_window)
        if len(self.price_history[symbol]) > max_window + 10:
            self.price_history[symbol] = self.price_history[symbol][-max_window:]

        prices = self.price_history[symbol]

        if len(prices) < self.long_window:
            return None

        # Calculate moving averages
        short_ma = sum(prices[-self.short_window :]) / self.short_window
        long_ma = sum(prices[-self.long_window :]) / self.long_window

        # Previous moving averages for crossover detection
        if len(prices) >= self.long_window + 1:
            prev_short_ma = sum(prices[-self.short_window - 1 : -1]) / self.short_window
            prev_long_ma = sum(prices[-self.long_window - 1 : -1]) / self.long_window

            # Bullish crossover
            if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                return TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.7,
                    price=price,
                    strategy_name="moving_average",
                    quantity=100,
                    stop_loss=None,
                    take_profit=None,
                )

            # Bearish crossover
            if short_ma < long_ma and prev_short_ma >= prev_long_ma:
                return TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=0.7,
                    price=price,
                    strategy_name="moving_average",
                    quantity=100,
                    stop_loss=None,
                    take_profit=None,
                )

        return None


def load_sample_data(
    symbol: str, start_date: datetime, end_date: datetime
) -> List[MarketData]:
    """Load sample market data for backtesting."""
    # This is a placeholder - in reality, you'd load from your database or files
    # For now, generate some sample data

    data = []
    current_date = start_date
    current_price = Decimal("150.00")  # Starting price

    while current_date <= end_date:
        # Simple random walk for demonstration
        import random

        price_change = Decimal(str(random.uniform(-0.05, 0.05)))  # ±5% daily change
        current_price = current_price * (1 + price_change)

        # Ensure reasonable OHLCV data
        daily_volatility = current_price * Decimal("0.02")  # 2% intraday volatility

        high = current_price + (daily_volatility * Decimal("0.5"))
        low = current_price - (daily_volatility * Decimal("0.5"))
        open_price = current_price + (
            daily_volatility * Decimal(str(random.uniform(-0.3, 0.3)))
        )
        volume = random.randint(1000000, 10000000)

        market_data = MarketData(
            symbol=symbol,
            timestamp=current_date,
            timeframe=TimeFrame.ONE_DAY,
            open=open_price,
            high=high,
            low=low,
            close=current_price,
            volume=volume,
            adjusted_close=current_price,
        )
        data.append(market_data)

        current_date += timedelta(days=1)

        # Skip weekends (simplified)
        if current_date.weekday() >= 5:
            current_date += timedelta(days=2)

    return data


def calculate_backtest_metrics(
    backtester: SimpleBacktester, start_date: datetime, end_date: datetime
) -> BacktestResult:
    """Calculate backtest performance metrics."""

    # Basic calculations
    final_equity = backtester.get_portfolio_value()
    total_return = (
        (final_equity - backtester.initial_capital) / backtester.initial_capital
    ) * 100

    # Calculate annualized return
    days = (end_date - start_date).days
    years = days / 365.25
    annual_return = (
        (float(final_equity / backtester.initial_capital) ** (1 / years) - 1) * 100
        if years > 0
        else 0
    )

    # Trade statistics
    winning_trades = [t for t in backtester.trades if t.pnl and t.pnl > 0]
    losing_trades = [t for t in backtester.trades if t.pnl and t.pnl < 0]

    win_rate = len(winning_trades) / len(backtester.trades) if backtester.trades else 0

    avg_win = (
        Decimal(
            str(
                sum(t.pnl for t in winning_trades if t.pnl is not None)
                / len(winning_trades)
            )
        )
        if winning_trades
        else Decimal("0")
    )
    avg_loss = (
        Decimal(
            str(
                sum(t.pnl for t in losing_trades if t.pnl is not None)
                / len(losing_trades)
            )
        )
        if losing_trades
        else Decimal("0")
    )

    # largest_win = max(
    #     (t.pnl for t in winning_trades if t.pnl is not None), default=Decimal("0")
    # )  # Unused variable
    # largest_loss = min(
    #     (t.pnl for t in losing_trades if t.pnl is not None), default=Decimal("0")
    # )  # Unused variable

    # Calculate maximum drawdown
    max_equity = backtester.initial_capital
    max_drawdown = Decimal("0")

    for point in backtester.equity_curve:
        equity = point["total_equity"]
        if equity > max_equity:
            max_equity = equity

        drawdown = (
            (equity - max_equity) / max_equity if max_equity > 0 else Decimal("0")
        )
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    return BacktestResult(
        strategy_name="backtest_strategy",
        start_date=start_date,
        end_date=end_date,
        initial_capital=backtester.initial_capital,
        final_capital=final_equity,
        total_return=total_return,
        total_return_pct=float(total_return / backtester.initial_capital * 100),
        annualized_return=annual_return,
        max_drawdown=max_drawdown,
        max_drawdown_pct=float(max_drawdown * 100),
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        win_rate=win_rate,
        total_trades=len(backtester.trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=(
            float(avg_win * len(winning_trades) / abs(avg_loss * len(losing_trades)))
            if losing_trades and avg_loss != 0
            else 0.0
        ),
    )


def run_backtest(
    strategy_name: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: Decimal = Decimal("100000"),
) -> BacktestResult:
    """Run backtest for specified strategy."""

    print(f"Running backtest for {strategy_name} strategy on {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: {format_currency(initial_capital)}")
    print("-" * 50)

    # Initialize backtester
    backtester = SimpleBacktester(initial_capital)

    # Initialize strategy
    if strategy_name == "moving_average":
        strategy = MovingAverageStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Load market data
    print("Loading market data...")
    market_data = load_sample_data(symbol, start_date, end_date)
    print(f"Loaded {len(market_data)} data points")

    # Run backtest
    print("Running backtest...")
    for i, data in enumerate(market_data):
        # Update position values
        backtester.update_positions({symbol: data.close})

        # Record equity point
        backtester.record_equity_point(data.timestamp)

        # Generate signal
        signal = strategy.generate_signal(symbol, data.close)

        if signal:
            executed = backtester.process_signal(signal, data.close)
            if executed:
                print(
                    f"  {data.timestamp.date()}: {signal.signal_type.value.upper()} {symbol} @ {format_currency(data.close)}"
                )

        # Progress indicator
        if (i + 1) % 50 == 0:
            progress = (i + 1) / len(market_data) * 100
            print(f"  Progress: {progress:.1f}%")

    # Calculate final metrics
    print("\nCalculating performance metrics...")
    results = calculate_backtest_metrics(backtester, start_date, end_date)

    return results


def print_results(results: BacktestResult) -> None:
    """Print backtest results in a formatted way."""

    print("\n" + "=" * 60)
    print("                  BACKTEST RESULTS")
    print("=" * 60)

    print(f"Strategy:           {results.strategy_name}")
    print(
        f"Period:             {results.start_date.date()} to {results.end_date.date()}"
    )
    print(f"Duration:           {(results.end_date - results.start_date).days} days")

    print("\nPortfolio Performance:")
    print(f"Initial Capital:    {format_currency(results.initial_capital)}")
    print(f"Final Equity:       {format_currency(results.final_capital)}")
    print(f"Total Return:       {format_percentage(results.total_return_pct / 100)}")
    print(f"Annual Return:      {format_percentage(results.annualized_return / 100)}")
    print(f"Max Drawdown:       {format_percentage(results.max_drawdown_pct / 100)}")

    print("\nTrade Statistics:")
    print(f"Total Trades:       {results.total_trades}")
    print(f"Winning Trades:     {results.winning_trades}")
    print(f"Losing Trades:      {results.losing_trades}")
    print(f"Win Rate:           {format_percentage(results.win_rate)}")

    if results.total_trades > 0:
        print(f"Average Win:        {format_currency(results.avg_win)}")
        print(f"Average Loss:       {format_currency(results.avg_loss)}")
        print(f"Profit Factor:      {results.profit_factor:.2f}")

        if results.avg_loss != 0:
            profit_factor = abs(results.avg_win * results.winning_trades) / abs(
                results.avg_loss * results.losing_trades
            )
            print(f"Profit Factor:      {profit_factor:.2f}")

    # Risk metrics
    if results.sharpe_ratio:
        print("\nRisk Metrics:")
        print(f"Sharpe Ratio:       {results.sharpe_ratio:.2f}")

    if results.sortino_ratio:
        print(f"Sortino Ratio:      {results.sortino_ratio:.2f}")

    print("=" * 60)


def save_results(results: BacktestResult, output_file: Optional[str] = None) -> None:
    """Save backtest results to file."""

    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"backtest_results_{results.strategy_name}_{timestamp}.json"

    # Ensure output directory exists
    os.makedirs("data/exports", exist_ok=True)
    output_path = os.path.join("data/exports", output_file)

    # Convert to JSON-serializable format
    results_dict = results.dict()

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Main backtesting function."""

    parser = argparse.ArgumentParser(description="Run backtest for trading strategies")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["moving_average"],
        help="Strategy to backtest",
    )
    parser.add_argument("--symbol", required=True, help="Symbol to backtest")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--save", action="store_true", help="Save results to file")

    args = parser.parse_args()

    try:
        # Parse dates
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")

        if start_date >= end_date:
            raise ValueError("Start date must be before end date")

        # Validate symbol
        symbol = args.symbol.upper()

        # Run backtest
        results = run_backtest(
            strategy_name=args.strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal(str(args.capital)),
        )

        # Print results
        print_results(results)

        # Save results if requested
        if args.save:
            save_results(results, args.output)

        # Exit with success code based on performance
        if results.total_return > 0:
            print("\n✅ Backtest completed successfully - Strategy was profitable!")
            sys.exit(0)
        else:
            print("\n⚠️  Backtest completed - Strategy was not profitable")
            sys.exit(0)

    except ValueError as e:
        print(f"❌ Input validation error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
