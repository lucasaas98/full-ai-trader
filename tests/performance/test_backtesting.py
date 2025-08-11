import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List
from dataclasses import dataclass
import warnings
from uuid import uuid4
warnings.filterwarnings('ignore')

# Import shared modules
import sys
sys.path.append('/app/shared')
sys.path.append('/app/backtesting')

from shared.models import TradeSignal, Trade, SignalType, OrderSide
from decimal import Decimal


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    num_simulations: int
    mean_final_value: float
    std_final_value: float
    percentile_5: float
    percentile_95: float
    probability_of_loss: float


class SimpleBacktestEngine:
    """Simplified backtest engine for performance testing"""

    def __init__(self, initial_capital: float = 100000, commission: float = 1.0):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = 0.001  # 0.1% slippage

    def run_backtest(self, signals: List[TradeSignal], price_data: pd.DataFrame) -> PerformanceMetrics:
        """Run a simple backtest with given signals and price data"""
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}
        trades = []

        for signal in signals:
            if signal.signal_type == SignalType.BUY and cash > 1000:
                # Buy logic
                signal_price = signal.price or Decimal("0")
                quantity = int(cash * 0.1 / float(signal_price))  # 10% of cash
                cost = quantity * float(signal_price) + self.commission

                if cost <= cash:
                    cash -= cost
                    positions[signal.symbol] = {
                        'quantity': quantity,
                        'avg_price': float(signal_price),
                        'entry_time': signal.timestamp
                    }

                    trades.append(Trade(
                        id=uuid4(),
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        price=signal_price,
                        commission=Decimal(str(self.commission)),
                        timestamp=signal.timestamp,
                        order_id=uuid4(),
                        strategy_name=signal.strategy_name,
                        pnl=None,
                        fees=Decimal("0.0")
                    ))

            elif signal.signal_type == SignalType.SELL and signal.symbol in positions:
                # Sell logic
                position = positions[signal.symbol]
                quantity = position['quantity']
                signal_price = signal.price or Decimal("0")
                proceeds = quantity * float(signal_price) - self.commission
                cash += proceeds

                # Calculate PnL
                pnl = proceeds - (quantity * position['avg_price'] + self.commission)

                trades.append(Trade(
                    id=uuid4(),
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    price=signal_price,
                    commission=Decimal(str(self.commission)),
                    timestamp=signal.timestamp,
                    order_id=uuid4(),
                    strategy_name=signal.strategy_name,
                    pnl=Decimal(str(pnl)),
                    fees=Decimal("0.0")
                ))

                del positions[signal.symbol]

        # Calculate final portfolio value
        final_portfolio_value = cash
        for symbol, position in positions.items():
            if symbol in price_data.columns:
                current_price = price_data[symbol].iloc[-1]
                final_portfolio_value += position['quantity'] * current_price

        # Calculate metrics
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital

        winning_trades = [t for t in trades if t.pnl and float(t.pnl) > 0]
        losing_trades = [t for t in trades if t.pnl and float(t.pnl) < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        avg_win = np.mean([float(t.pnl) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(float(t.pnl)) for t in losing_trades]) if losing_trades else 0

        profit_factor = float(abs(avg_win) / abs(avg_loss)) if avg_loss != 0 else 0.0

        # Simple Sharpe ratio calculation
        returns = pd.Series([float(t.pnl) if t.pnl else 0 for t in trades])
        sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=total_return,  # Simplified
            sharpe_ratio=sharpe_ratio,
            max_drawdown=0.0,  # Simplified
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades)
        )


class MonteCarloSimulator:
    """Simple Monte Carlo simulator for performance testing"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def run_simulation(self, num_simulations: int = 1000, time_horizon: int = 252) -> MonteCarloResult:
        """Run Monte Carlo simulation"""
        np.random.seed(42)  # For reproducible results

        # Generate random returns
        daily_returns = np.random.normal(0.001, 0.02, (num_simulations, time_horizon))

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1)
        final_values = self.initial_capital * cumulative_returns[:, -1]

        return MonteCarloResult(
            num_simulations=num_simulations,
            mean_final_value=float(np.mean(final_values)),
            std_final_value=float(np.std(final_values)),
            percentile_5=float(np.percentile(final_values, 5)),
            percentile_95=float(np.percentile(final_values, 95)),
            probability_of_loss=float(np.sum(final_values < self.initial_capital) / num_simulations)
        )


class TestBacktestingPerformance:
    """Performance tests for backtesting functionality"""

    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data for testing"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)

        # Generate realistic price movements
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)

        return pd.DataFrame({
            'AAPL': prices,
            'GOOGL': prices * 1.5 + np.random.normal(0, 5, len(dates)),
            'MSFT': prices * 0.8 + np.random.normal(0, 3, len(dates))
        }, index=dates)

    @pytest.fixture
    def sample_signals(self):
        """Generate sample trading signals for testing"""
        signals = []
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

        for i in range(50):
            signal = TradeSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                confidence=0.7,
                price=Decimal(str(100 + i)),
                quantity=100,
                timestamp=base_time + timedelta(days=i * 7),
                strategy_name="TestStrategy",
                stop_loss=Decimal(str(95 + i)),
                take_profit=Decimal(str(105 + i))
            )
            signals.append(signal)

        return signals

    @pytest.mark.performance
    def test_simple_backtest_performance(self, sample_signals, sample_price_data):
        """Test basic backtest performance"""
        engine = SimpleBacktestEngine()

        # Run backtest
        result = engine.run_backtest(sample_signals, sample_price_data)

        # Validate results
        assert isinstance(result, PerformanceMetrics)
        assert result.total_trades > 0
        assert -1.0 <= result.total_return <= 10.0  # Reasonable bounds
        assert 0.0 <= result.win_rate <= 1.0
        assert result.profit_factor >= 0

    @pytest.mark.performance
    def test_monte_carlo_simulation_performance(self):
        """Test Monte Carlo simulation performance"""
        simulator = MonteCarloSimulator()

        # Run simulation
        result = simulator.run_simulation(num_simulations=100, time_horizon=252)

        # Validate results
        assert isinstance(result, MonteCarloResult)
        assert result.num_simulations == 100
        assert result.mean_final_value > 0
        assert result.std_final_value > 0
        assert 0.0 <= result.probability_of_loss <= 1.0
        assert result.percentile_5 < result.percentile_95

    @pytest.mark.performance
    def test_large_dataset_backtesting(self):
        """Test backtesting with large datasets"""
        # Generate large dataset
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='H')  # Hourly data
        dates_list = [datetime.fromtimestamp(ts.timestamp(), tz=timezone.utc) for ts in dates]
        np.random.seed(42)

        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.001, len(dates)))
        large_data = pd.DataFrame({'AAPL': prices}, index=dates)

        # Generate many signals
        signals = []
        for i in range(0, len(dates), 100):  # Every 100 hours
            signal = TradeSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY if i % 200 == 0 else SignalType.SELL,
                confidence=0.6,
                price=Decimal(str(prices[i])),
                quantity=100,
                timestamp=dates_list[i],
                strategy_name="LargeDataStrategy",
                stop_loss=Decimal(str(prices[i] * 0.95)),
                take_profit=Decimal(str(prices[i] * 1.05))
            )
            signals.append(signal)

        engine = SimpleBacktestEngine()

        # Measure performance
        start_time = datetime.now()
        result = engine.run_backtest(signals, large_data)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        # Should complete in reasonable time (less than 10 seconds)
        assert execution_time < 10.0
        assert result.total_trades > 0

    @pytest.mark.performance
    def test_multiple_symbol_backtesting(self, sample_price_data):
        """Test backtesting with multiple symbols"""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        signals = []

        # Generate signals for multiple symbols
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        for i, symbol in enumerate(symbols):
            for j in range(20):
                signal = TradeSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY if j % 2 == 0 else SignalType.SELL,
                    confidence=0.7,
                    price=Decimal(str(sample_price_data[symbol].iloc[j * 10])),
                    quantity=100,
                    timestamp=base_time + timedelta(days=j * 10),
                    strategy_name=f"Strategy_{symbol}",
                    stop_loss=Decimal(str(sample_price_data[symbol].iloc[j * 10] * 0.95)),
                    take_profit=Decimal(str(sample_price_data[symbol].iloc[j * 10] * 1.05))
                )
                signals.append(signal)

        engine = SimpleBacktestEngine()
        result = engine.run_backtest(signals, sample_price_data)

        assert result.total_trades > 0
        assert len(symbols) <= result.total_trades  # At least one trade per symbol potentially

    @pytest.mark.performance
    def test_monte_carlo_stress_test(self):
        """Stress test Monte Carlo simulation with many simulations"""
        simulator = MonteCarloSimulator()

        # Run large simulation
        start_time = datetime.now()
        result = simulator.run_simulation(num_simulations=10000, time_horizon=1000)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        # Should complete in reasonable time
        assert execution_time < 30.0  # 30 seconds max
        assert result.num_simulations == 10000
        assert result.mean_final_value > 0

    @pytest.mark.performance
    def test_backtest_memory_usage(self, sample_price_data):
        """Test memory usage during backtesting"""
        import tracemalloc

        # Start memory tracking
        tracemalloc.start()

        # Generate large number of signals
        signals = []
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

        for i in range(1000):  # 1000 signals
            signal = TradeSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                confidence=0.7,
                price=Decimal(str(100 + i * 0.1)),
                quantity=100,
                timestamp=base_time + timedelta(hours=i),
                strategy_name="MemoryTestStrategy",
                stop_loss=Decimal(str((100 + i * 0.1) * 0.95)),
                take_profit=Decimal(str((100 + i * 0.1) * 1.05))
            )
            signals.append(signal)

        engine = SimpleBacktestEngine()
        result = engine.run_backtest(signals, sample_price_data)

        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 50MB)
        assert peak < 50 * 1024 * 1024  # 50MB
        assert result.total_trades > 0

    @pytest.mark.performance
    def test_parallel_backtest_simulation(self):
        """Test parallel execution of multiple backtests"""
        from concurrent.futures import ThreadPoolExecutor

        def run_single_backtest(seed: int) -> PerformanceMetrics:
            """Run a single backtest with given random seed"""
            np.random.seed(seed)

            # Generate random price data
            dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
            dates_list = [datetime.fromtimestamp(ts.timestamp(), tz=timezone.utc) for ts in dates]
            prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates)))
            price_data = pd.DataFrame({'AAPL': prices}, index=dates)

            # Generate random signals
            signals = []
            for i in range(50):
                signal = TradeSignal(
                    symbol="AAPL",
                    signal_type=SignalType.BUY if np.random.random() > 0.5 else SignalType.SELL,
                    confidence=0.7,
                    price=Decimal(str(prices[i])),
                    quantity=100,
                    timestamp=dates_list[i],
                    strategy_name=f"ParallelStrategy_{seed}",
                    stop_loss=Decimal(str(prices[i] * 0.95)),
                    take_profit=Decimal(str(prices[i] * 1.05))
                )
                signals.append(signal)

            engine = SimpleBacktestEngine()
            return engine.run_backtest(signals, price_data)

        # Run multiple backtests in parallel
        start_time = datetime.now()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_single_backtest, i) for i in range(10)]
            results = [future.result() for future in futures]

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Should complete all backtests
        assert len(results) == 10
        assert all(isinstance(r, PerformanceMetrics) for r in results)

        # Parallel execution should be faster than sequential
        assert execution_time < 15.0  # Should complete in reasonable time

    @pytest.mark.performance
    def test_strategy_optimization_performance(self, sample_price_data):
        """Test strategy parameter optimization performance"""

        def test_strategy_with_params(sma_period: int, rsi_period: int) -> float:
            """Test a simple strategy with given parameters"""
            # Calculate indicators
            sma = sample_price_data['AAPL'].rolling(sma_period).mean()

            # Generate signals based on SMA crossover
            signals = []
            base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

            for i in range(len(sample_price_data) - sma_period):
                current_price = sample_price_data['AAPL'].iloc[i + sma_period]
                current_sma = sma.iloc[i + sma_period]

                if pd.notna(current_sma) and current_price > current_sma * 1.02:
                    signal = TradeSignal(
                        symbol="AAPL",
                        signal_type=SignalType.BUY,
                        confidence=0.7,
                        price=Decimal(str(current_price)),
                        quantity=100,
                        timestamp=base_time + timedelta(days=i),
                        strategy_name="OptimizedStrategy",
                        stop_loss=Decimal(str(current_price * 0.95)),
                        take_profit=Decimal(str(current_price * 1.05))
                    )
                    signals.append(signal)
                elif pd.notna(current_sma) and current_price < current_sma * 0.98:
                    signal = TradeSignal(
                        symbol="AAPL",
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        price=Decimal(str(current_price)),
                        quantity=100,
                        timestamp=base_time + timedelta(days=i),
                        strategy_name="OptimizedStrategy",
                        stop_loss=Decimal(str(current_price * 1.05)),
                        take_profit=Decimal(str(current_price * 0.95))
                    )
                    signals.append(signal)

            engine = SimpleBacktestEngine()
            result = engine.run_backtest(signals, sample_price_data)
            return result.total_return

        # Test parameter optimization
        best_return = -999
        best_params = None

        start_time = datetime.now()

        # Grid search over parameter space
        for sma_period in range(10, 31, 5):
            for rsi_period in range(10, 21, 5):
                try:
                    return_value = test_strategy_with_params(sma_period, rsi_period)
                    if return_value > best_return:
                        best_return = return_value
                        best_params = (sma_period, rsi_period)
                except Exception:
                    continue  # Skip invalid parameter combinations

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Should find some parameters and complete in reasonable time
        assert best_params is not None
        assert execution_time < 60.0  # 1 minute max
        assert isinstance(best_return, float)

    @pytest.mark.performance
    def test_backtest_scalability(self):
        """Test backtesting scalability with increasing data sizes"""
        results = {}

        for data_size in [100, 500, 1000, 2000]:
            # Generate data of varying sizes
            dates = pd.date_range('2023-01-01', periods=data_size, freq='H')
            dates_list = [datetime.fromtimestamp(ts.timestamp(), tz=timezone.utc) for ts in dates]
            np.random.seed(42)
            prices = 100 * np.cumprod(1 + np.random.normal(0, 0.001, data_size))
            price_data = pd.DataFrame({'AAPL': prices}, index=dates)

            # Generate proportional number of signals
            num_signals = min(data_size // 10, 200)
            signals = []

            for i in range(num_signals):
                signal = TradeSignal(
                    symbol="AAPL",
                    signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                    confidence=0.7,
                    price=Decimal(str(prices[i * (data_size // num_signals)])),
                    quantity=100,
                    timestamp=dates_list[i * (data_size // num_signals)],
                    strategy_name="ScalabilityStrategy",
                    stop_loss=Decimal(str(prices[i * (data_size // num_signals)] * 0.95)),
                    take_profit=Decimal(str(prices[i * (data_size // num_signals)] * 1.05))
                )
                signals.append(signal)

            # Measure execution time
            start_time = datetime.now()
            engine = SimpleBacktestEngine()
            result = engine.run_backtest(signals, price_data)
            end_time = datetime.now()

            execution_time = (end_time - start_time).total_seconds()
            results[data_size] = execution_time

            # Validate results
            assert result.total_trades >= 0
            assert execution_time < 5.0  # Should complete quickly

        # Execution time should scale reasonably
        assert all(time < 5.0 for time in results.values())

    @pytest.mark.performance
    def test_high_frequency_signal_processing(self):
        """Test processing high-frequency trading signals"""
        # Generate minute-by-minute signals for a trading day
        trading_start = datetime(2023, 6, 1, 9, 30, tzinfo=timezone.utc)
        trading_end = datetime(2023, 6, 1, 16, 0, tzinfo=timezone.utc)

        signals = []
        current_time = trading_start
        price = 100.0

        while current_time <= trading_end:
            # Random price movement
            price += np.random.normal(0, 0.1)

            # Generate signal every 5 minutes
            if current_time.minute % 5 == 0:
                signal = TradeSignal(
                    symbol="AAPL",
                    signal_type=SignalType.BUY if np.random.random() > 0.5 else SignalType.SELL,
                    confidence=0.6,
                    price=Decimal(str(round(price, 2))),
                    quantity=100,
                    timestamp=current_time,
                    strategy_name="HighFrequencyStrategy",
                    stop_loss=Decimal(str(round(price * 0.95, 2))),
                    take_profit=Decimal(str(round(price * 1.05, 2)))
                )
                signals.append(signal)

            current_time += timedelta(minutes=1)

        # Create price data
        timestamps = pd.date_range(trading_start, trading_end, freq='min')
        prices_series = 100 * np.cumprod(1 + np.random.normal(0, 0.0005, len(timestamps)))
        price_data = pd.DataFrame({'AAPL': prices_series}, index=timestamps)

        # Run backtest
        start_time = datetime.now()
        engine = SimpleBacktestEngine()
        result = engine.run_backtest(signals, price_data)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        # Should handle high-frequency data efficiently
        assert execution_time < 2.0  # 2 seconds max
        assert result.total_trades > 0
        assert len(signals) > 50  # Should have generated many signals

    @pytest.mark.performance
    def test_multi_strategy_backtesting(self):
        """Test running multiple strategies simultaneously"""
        strategies = ['MomentumStrategy', 'MeanReversionStrategy', 'BreakoutStrategy']

        all_signals = []
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

        # Generate signals for each strategy
        for strategy_name in strategies:
            for i in range(30):
                signal = TradeSignal(
                    symbol="AAPL",
                    signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                    confidence=0.7,
                    price=Decimal(str(100 + i + len(strategy_name))),
                    quantity=100,
                    timestamp=base_time + timedelta(days=i * 3),
                    strategy_name=strategy_name,
                    stop_loss=Decimal(str((100 + i + len(strategy_name)) * 0.95)),
                    take_profit=Decimal(str((100 + i + len(strategy_name)) * 1.05))
                )
                all_signals.append(signal)

        # Generate price data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, len(dates)))
        price_data = pd.DataFrame({'AAPL': prices}, index=dates)

        # Run combined backtest
        start_time = datetime.now()
        engine = SimpleBacktestEngine()
        result = engine.run_backtest(all_signals, price_data)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        # Should handle multiple strategies efficiently
        assert execution_time < 3.0
        assert result.total_trades > 0
        assert len(all_signals) == len(strategies) * 30
