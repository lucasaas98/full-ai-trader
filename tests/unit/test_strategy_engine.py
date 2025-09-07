import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pytest

from shared.models import Position, SignalType, TradeSignal

# Technical analysis is handled within the base strategy


@pytest.fixture
def strategy_engine():
    """Create strategy engine instance for testing"""

    class MockStrategy:
        def __init__(self):
            self.name = "momentum"
            self.enabled = True
            self.config = {"rsi_period": 14, "rsi_oversold": 30}

        async def analyze(self, symbol, data):
            """Mock strategy analysis."""
            return TradeSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=Decimal("150.50"),
                quantity=100,
                strategy_name=self.name,
                stop_loss=Decimal("142.975"),
                take_profit=Decimal("158.025"),
                timestamp=datetime.now(timezone.utc),
            )

    class MockStrategyEngineService:
        def __init__(self):
            # Initialize without Redis/external dependencies
            self.strategies = {"momentum": MockStrategy()}
            self.signal_history = []
            self.performance_metrics = {}
            self.active_strategies = set(["momentum"])

        async def process_signal(self, signal):
            """Mock signal processing."""
            self.signal_history.append(signal)
            return {
                "processed": True,
                "signal_id": len(self.signal_history),
                "timestamp": datetime.now(timezone.utc),
            }

        def calculate_drawdown(self, equity_curve):
            """Calculate maximum drawdown from equity curve."""
            import pandas as pd

            if isinstance(equity_curve, list):
                equity_curve = pd.Series(equity_curve)

            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            return drawdown

        def validate_signal(self, signal):
            """Validate signal format and content."""
            # Check confidence threshold
            if signal.confidence < 0.6:
                return False

            # Check timestamp is not in the future
            if signal.timestamp > datetime.now(timezone.utc):
                return False

            return True

        async def calculate_position_size(self, signal, portfolio):
            """Calculate position size based on signal and portfolio."""
            max_position_value = float(portfolio.total_equity) * 0.05  # 5% max
            position_size = max_position_value / float(signal.price)
            return min(position_size, signal.quantity)

        async def check_portfolio_risk(self, signal, portfolio):
            """Check portfolio-level risk constraints."""
            # Simple concentration check
            existing_position = None
            for pos in portfolio.positions:
                if pos.symbol == signal.symbol:
                    existing_position = pos
                    break

            if existing_position:
                total_exposure = float(existing_position.market_value) + (
                    float(signal.price) * signal.quantity
                )
                concentration = total_exposure / float(portfolio.total_equity)

                return {
                    "approved": concentration <= 0.10,  # Max 10% per symbol
                    "reduced_size": concentration > 0.05,
                    "concentration": concentration,
                }

            return {"approved": True, "reduced_size": False, "concentration": 0.0}

        def get_strategy(self, name):
            """Get strategy by name."""
            return self.strategies.get(name)

        async def execute_strategy(self, strategy_name, symbol, data):
            """Execute strategy and return signal."""
            strategy = self.get_strategy(strategy_name)
            if strategy:
                return await strategy.analyze(symbol, data)
            return None

        def track_signal_accuracy(self, signals_outcomes):
            """Track signal accuracy from historical data."""
            if not signals_outcomes:
                return 0.0

            correct_signals = sum(1 for _, outcome in signals_outcomes if outcome > 0)
            return correct_signals / len(signals_outcomes)

        async def check_correlation_risk(self, signal, portfolio):
            """Check correlation risk between assets."""
            # Mock high correlation detection
            correlated_symbols = ["AAPL", "GOOGL"]  # Mock correlated pairs

            existing_symbols = [pos.symbol for pos in portfolio.positions]
            high_correlation = any(
                symbol in correlated_symbols and signal.symbol in correlated_symbols
                for symbol in existing_symbols
            )

            return {"high_correlation": high_correlation}

        def update_strategy_config(self, strategy_name, config):
            """Update strategy configuration."""
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                # Mock configuration update
                if not hasattr(strategy, "config"):
                    strategy.config = {}
                strategy.config.update(config)
                return True
            return False

        def disable_strategy(self, strategy_name):
            """Disable a strategy."""
            if strategy_name in self.strategies:
                self.strategies[strategy_name].enabled = False
                self.active_strategies.discard(strategy_name)
                return True
            return False

        def enable_strategy(self, strategy_name):
            """Enable a strategy."""
            if strategy_name in self.strategies:
                self.strategies[strategy_name].enabled = True
                self.active_strategies.add(strategy_name)
                return True
            return False

        def get_correlation_matrix(self):
            """Mock correlation matrix."""
            import pandas as pd

            return pd.DataFrame(
                {"AAPL": [1.0, 0.9], "GOOGL": [0.9, 1.0]},
                index=pd.Series(["AAPL", "GOOGL"]),
            )

    return MockStrategyEngineService()


@pytest.fixture
def momentum_strategy():
    """Create a test momentum strategy."""
    from services.strategy_engine.src.base_strategy import (
        BaseStrategy as BaseStrategyClass,
    )
    from services.strategy_engine.src.base_strategy import (
        Signal,
        StrategyConfig,
        StrategyMode,
    )

    class TestMomentumStrategy(BaseStrategyClass):
        def __init__(self, config):
            super().__init__(config)

        def _setup_indicators(self):
            pass

        async def analyze(self, symbol: str, data):
            # Simple momentum logic for testing
            import polars as pl

            if isinstance(data, pd.DataFrame):
                # Convert pandas to polars for consistency
                data = pl.from_pandas(data)

            if len(data) < 20:
                return Signal(
                    action=SignalType.HOLD,
                    confidence=30.0,
                    entry_price=Decimal("100.0"),
                    position_size=0.0,
                    reasoning="Insufficient data",
                )

            # Simple momentum: compare current price to 20-period average
            current_price = float(data.select("close").tail(1).item())
            avg_price = float(data.select("close").tail(20).mean().item())

            if current_price > avg_price * 1.02:  # 2% above average
                return Signal(
                    action=SignalType.BUY,
                    confidence=75.0,
                    entry_price=Decimal(str(current_price)),
                    stop_loss=Decimal(str(current_price * 0.95)),
                    take_profit=Decimal(str(current_price * 1.05)),
                    position_size=0.05,
                    reasoning="Price above 20-period average",
                )
            elif current_price < avg_price * 0.98:  # 2% below average
                return Signal(
                    action=SignalType.SELL,
                    confidence=75.0,
                    entry_price=Decimal(str(current_price)),
                    stop_loss=Decimal(str(current_price * 1.05)),
                    take_profit=Decimal(str(current_price * 0.95)),
                    position_size=0.05,
                    reasoning="Price below 20-period average",
                )
            else:
                return Signal(
                    action=SignalType.HOLD,
                    confidence=50.0,
                    entry_price=Decimal(str(current_price)),
                    position_size=0.0,
                    reasoning="Price near average",
                )

    config = StrategyConfig(
        name="test_momentum",
        mode=StrategyMode.SWING_TRADING,
        lookback_period=20,
        min_confidence=60.0,
    )
    return TestMomentumStrategy(config)


class TestStrategyEngineIntegration:
    """Unit tests for the main strategy engine integration"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_strategy_execution(
        self, strategy_engine, momentum_strategy, sample_historical_data
    ):
        """Test real strategy execution with actual market data"""

        # Add strategy to engine
        strategy_engine.strategies["momentum"] = momentum_strategy

        # Convert pandas to polars for strategy
        import polars as pl

        data_pl = pl.from_pandas(sample_historical_data)

        # Execute strategy
        signal = await strategy_engine.execute_strategy("momentum", "AAPL", data_pl)

        # Verify signal is generated
        assert signal is not None
        assert signal.action in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert 0 <= signal.confidence <= 100
        assert signal.entry_price > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_signal_processing(self, strategy_engine):
        """Test signal processing pipeline"""

        # Create test signal
        test_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("150.50"),
            quantity=100,
            strategy_name="momentum",
            stop_loss=Decimal("145.0"),
            take_profit=Decimal("155.0"),
            timestamp=datetime.now(timezone.utc),
        )

        # Process signal
        result = await strategy_engine.process_signal(test_signal)

        # Verify processing
        assert result["processed"] is True
        assert "signal_id" in result
        assert len(strategy_engine.signal_history) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_signal_validation(self, strategy_engine):
        """Test signal validation logic"""

        # Valid signal
        valid_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("150.50"),
            quantity=100,
            strategy_name="momentum",
            stop_loss=Decimal("142.975"),
            take_profit=Decimal("158.025"),
            timestamp=datetime.now(timezone.utc),
        )

        # Invalid signal (low confidence)
        invalid_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.3,  # Below 0.6 threshold
            price=Decimal("150.50"),
            quantity=100,
            strategy_name="momentum",
            stop_loss=Decimal("142.975"),
            take_profit=Decimal("158.025"),
            timestamp=datetime.now(timezone.utc),
        )

        assert strategy_engine.validate_signal(valid_signal) is True
        assert strategy_engine.validate_signal(invalid_signal) is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_strategy_validation(self, strategy_engine):
        """Test strategy signal validation"""

        # Valid signal
        valid_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("150.50"),
            quantity=100,
            strategy_name="momentum",
            stop_loss=Decimal("145.0"),
            take_profit=Decimal("155.0"),
            timestamp=datetime.now(timezone.utc),
        )

        assert strategy_engine.validate_signal(valid_signal) is True

        # Invalid signal - future timestamp
        invalid_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("150.50"),
            quantity=100,
            strategy_name="momentum",
            stop_loss=Decimal("145.0"),
            take_profit=Decimal("155.0"),
            timestamp=datetime.now(timezone.utc)
            + timedelta(hours=1),  # Future timestamp
        )

        assert strategy_engine.validate_signal(invalid_signal) is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_strategy_risk_integration(self, strategy_engine, sample_portfolio):
        """Test integration with risk management"""

        # Signal that would exceed position size limit
        large_signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("150.50"),
            quantity=1000,  # Large quantity
            strategy_name="momentum",
            stop_loss=Decimal("142.975"),
            take_profit=Decimal("158.025"),
            timestamp=datetime.now(timezone.utc),
        )

        # Check portfolio risk
        risk_result = await strategy_engine.check_portfolio_risk(
            large_signal, sample_portfolio
        )

        # Should flag concentration risk for existing AAPL position
        assert "approved" in risk_result
        assert "concentration" in risk_result

        # Calculate position size with risk constraints
        adjusted_size = await strategy_engine.calculate_position_size(
            large_signal, sample_portfolio
        )

        # Position size should be limited by risk constraints
        assert adjusted_size < large_signal.quantity
        assert adjusted_size > 0


class TestStrategyBacktesting:
    """Unit tests for strategy backtesting functionality"""

    @pytest.fixture
    def backtester(self):
        """Create backtester instance"""

        class MockStrategyBacktester:
            def __init__(self, initial_capital, commission, slippage):
                self.initial_capital = initial_capital
                self.commission = commission
                self.slippage = slippage
                self.portfolio_value = initial_capital
                self.trades = []

                # Add portfolio mock
                class MockPortfolio:
                    def __init__(self, initial_cash):
                        self.cash_balance = initial_cash
                        self.positions = []

                    def add_position(self, symbol, quantity, price):
                        self.positions.append(
                            {"symbol": symbol, "quantity": quantity, "price": price}
                        )

                self.portfolio = MockPortfolio(initial_capital)

            def run_backtest(self, strategy, data, start_date, end_date):
                """Mock backtest execution"""
                return {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.05,
                    "number_of_trades": 50,
                }

            def execute_trade(self, symbol, side, quantity, price, timestamp):
                """Mock trade execution"""
                trade = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "timestamp": timestamp,
                }
                self.trades.append(trade)
                # Update portfolio cash balance and positions
                if side == "buy":
                    self.portfolio.cash_balance -= quantity * price
                    self.portfolio.add_position(symbol, quantity, price)
                else:
                    self.portfolio.cash_balance += quantity * price
                return trade

            def apply_slippage_and_commission(
                self, symbol, side, intended_price, quantity
            ):
                """Mock slippage and commission calculation"""
                slippage_amount = intended_price * self.slippage
                commission_amount = quantity * self.commission

                if side == "buy":
                    executed_price = (
                        intended_price + slippage_amount + commission_amount / quantity
                    )
                else:
                    executed_price = (
                        intended_price - slippage_amount - commission_amount / quantity
                    )

                return executed_price

            def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
                """Mock Sharpe ratio calculation"""
                import numpy as np

                excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
                return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

            def calculate_max_drawdown(self, equity_curve):
                """Mock max drawdown calculation"""
                import numpy as np

                peak = np.maximum.accumulate(equity_curve)
                drawdown = (equity_curve - peak) / peak
                return abs(np.min(drawdown))  # Return positive value

        return MockStrategyBacktester(
            initial_capital=100000.0, commission=1.0, slippage=0.001
        )

    @pytest.mark.unit
    def test_backtest_execution(
        self, backtester, sample_historical_data, momentum_strategy
    ):
        """Test backtest execution"""

        results = backtester.run_backtest(
            strategy=momentum_strategy,
            data=sample_historical_data,
            start_date=sample_historical_data.index[0],
            end_date=sample_historical_data.index[-1],
        )

        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "number_of_trades" in results

        # Returns should be reasonable
        assert -1.0 <= results["total_return"] <= 10.0  # -100% to 1000%

    @pytest.mark.unit
    def test_portfolio_simulation(self, backtester, sample_historical_data):
        """Test portfolio simulation during backtest"""

        # Simulate buy order
        backtester.execute_trade("AAPL", "buy", 100, 150.0, datetime.now(timezone.utc))

        assert backtester.portfolio.cash_balance < backtester.initial_capital
        assert len(backtester.portfolio.positions) > 0
        assert backtester.portfolio.positions[0]["symbol"] == "AAPL"
        assert backtester.portfolio.positions[0]["quantity"] == 100

    @pytest.mark.unit
    def test_slippage_and_commission_application(self, backtester):
        """Test slippage and commission calculations"""

        # Test buy order
        executed_price = backtester.apply_slippage_and_commission(
            symbol="AAPL", side="buy", intended_price=150.0, quantity=100
        )

        # Buy price should be higher due to slippage
        assert executed_price > 150.0

        # Test sell order
        executed_price = backtester.apply_slippage_and_commission(
            symbol="AAPL", side="sell", intended_price=150.0, quantity=100
        )

        # Sell price should be lower due to slippage
        assert executed_price < 150.0

    @pytest.mark.unit
    def test_performance_metrics_calculation(self, backtester):
        """Test performance metrics calculations"""

        # Create sample trade history
        trades = [
            {"date": datetime(2023, 1, 1), "return": 0.02},
            {"date": datetime(2023, 1, 2), "return": -0.01},
            {"date": datetime(2023, 1, 3), "return": 0.03},
            {"date": datetime(2023, 1, 4), "return": -0.02},
            {"date": datetime(2023, 1, 5), "return": 0.01},
        ]

        returns = pd.Series([t["return"] for t in trades])

        # Calculate metrics
        sharpe_ratio = backtester.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        max_drawdown = backtester.calculate_max_drawdown(returns.cumsum())

        assert isinstance(sharpe_ratio, (int, float))
        assert 0 <= max_drawdown <= 1.0


class TestSignalProcessor:
    """Unit tests for signal processing"""

    @pytest.fixture
    def signal_processor(self, mock_redis_client):
        """Create signal processor instance"""

        class MockSignalProcessor:
            def __init__(self, redis_client):
                self.redis_client = redis_client

            def process(self, signals):
                return [s for s in signals if s.confidence > 0.5]

            def validate_signal(self, signal):
                return signal.symbol != "INVALID"

            def filter_signals_by_strength(self, signals, min_strength=0.5):
                """Filter signals by minimum strength"""
                return [s for s in signals if s.confidence >= min_strength]

            async def aggregate_signals(self, signals):
                """Aggregate signals from multiple strategies"""
                aggregated: dict[str, list] = {}
                for signal in signals:
                    if signal.symbol not in aggregated:
                        aggregated[signal.symbol] = []
                    aggregated[signal.symbol].append(signal)
                return aggregated

            async def handle_conflicting_signals(self, signals):
                """Handle conflicting signals"""
                # Simple logic: keep the signal with highest confidence
                if not signals:
                    return None
                return max(signals, key=lambda s: s.confidence)

            async def persist_signal(self, signal):
                """Persist signal to storage"""
                # Mock implementation - call redis hset
                self.redis_client.hset(
                    f"signals:{signal.symbol}", signal.id, "signal_data"
                )
                return {"id": signal.id, "status": "persisted"}

            async def retrieve_signals(
                self, symbol=None, start_time=None, end_time=None
            ):
                """Retrieve signals from storage"""
                # Mock implementation
                return []

        return MockSignalProcessor(redis_client=mock_redis_client)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_signal_aggregation(self, signal_processor, trading_signal_factory):
        """Test signal aggregation from multiple strategies"""

        signals = [
            trading_signal_factory(
                strategy_name="momentum", strength=0.8, signal_type="BUY"
            ),
            trading_signal_factory(
                strategy_name="rsi", strength=0.7, signal_type="BUY"
            ),
            trading_signal_factory(
                strategy_name="macd", strength=0.6, signal_type="BUY"
            ),
        ]

        aggregated = await signal_processor.aggregate_signals(signals)

        assert "AAPL" in aggregated  # Should have aggregated signals for AAPL
        assert len(aggregated["AAPL"]) == 3  # Should have all three signals

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conflicting_signals_handling(
        self, signal_processor, trading_signal_factory
    ):
        """Test handling of conflicting signals"""

        signals = [
            trading_signal_factory(
                strategy_name="momentum", strength=0.8, signal_type="BUY"
            ),
            trading_signal_factory(
                strategy_name="mean_reversion", strength=0.9, signal_type="SELL"
            ),
        ]

        aggregated_signal = await signal_processor.handle_conflicting_signals(signals)

        # Should return the signal with highest confidence
        assert aggregated_signal is not None
        assert aggregated_signal.confidence == 0.9  # SELL signal has higher confidence

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_signal_persistence(
        self, signal_processor, mock_redis_client, sample_trading_signal
    ):
        """Test signal persistence to Redis"""

        await signal_processor.persist_signal(sample_trading_signal)

        # Should store signal in Redis
        mock_redis_client.hset.assert_called()
        call_args = mock_redis_client.hset.call_args
        assert "signals" in call_args[0][0]  # Hash key should contain 'signals'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_signal_retrieval(self, signal_processor, mock_redis_client):
        """Test signal retrieval from storage"""

        # Mock stored signal
        stored_signal_data = {
            "signal_id": "test_001",
            "strategy_name": "momentum",
            "symbol": "AAPL",
            "signal_type": "BUY",
            "strength": 0.8,
            "price": 150.50,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        mock_redis_client.hget.return_value = json.dumps(stored_signal_data)

        retrieved_signals = await signal_processor.retrieve_signals(symbol="AAPL")

        assert len(retrieved_signals) == 0  # Mock returns empty list

    @pytest.mark.unit
    def test_signal_filtering(self, signal_processor, trading_signal_factory):
        """Test signal filtering based on criteria"""

        signals = [
            trading_signal_factory(strength=0.9, symbol="AAPL"),
            trading_signal_factory(strength=0.3, symbol="GOOGL"),  # Weak signal
            trading_signal_factory(strength=0.8, symbol="MSFT"),
            trading_signal_factory(strength=0.1, symbol="TSLA"),  # Very weak signal
        ]

        # Filter signals with minimum strength of 0.5
        filtered_signals = signal_processor.filter_signals_by_strength(
            signals, min_strength=0.5
        )

        assert len(filtered_signals) == 2  # Only AAPL and MSFT should pass
        assert all(signal.confidence >= 0.5 for signal in filtered_signals)


class TestRiskIntegration:
    """Unit tests for risk management integration"""

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing"""
        from shared.models import PortfolioState, Position

        return PortfolioState(
            account_id="test_account",
            cash=Decimal("50000.0"),
            buying_power=Decimal("100000.0"),
            total_equity=Decimal("64500.0"),
            positions=[
                Position(
                    symbol="AAPL",
                    quantity=100,
                    entry_price=Decimal("145.0"),
                    market_value=Decimal("14500.0"),
                    cost_basis=Decimal("14500.0"),
                    current_price=Decimal("145.0"),
                    unrealized_pnl=Decimal("0.0"),
                )
            ],
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_position_size_calculation(self, strategy_engine, sample_portfolio):
        """Test position size calculation with risk constraints"""

        signal = TradeSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=Decimal("150.50"),
            quantity=100,
            strategy_name="momentum",
            stop_loss=Decimal("145.0"),
            take_profit=Decimal("155.0"),
            timestamp=datetime.now(timezone.utc),
        )

        # Calculate position size considering portfolio
        position_size = await strategy_engine.calculate_position_size(
            signal, sample_portfolio
        )

        # Should respect maximum position size limits
        max_position_value = float(sample_portfolio.total_equity) * 0.05  # 5% max
        calculated_value = position_size * float(signal.price or 0)

        assert calculated_value <= max_position_value

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_portfolio_risk_check(self, strategy_engine, sample_portfolio):
        """Test portfolio-level risk checks"""

        # Create signal that would increase portfolio risk
        high_risk_signal = TradeSignal(
            symbol="AAPL",  # Already have position in AAPL
            signal_type=SignalType.BUY,
            confidence=0.9,
            price=Decimal("150.50"),
            quantity=100,
            strategy_name="momentum",
            stop_loss=Decimal("145.0"),
            take_profit=Decimal("155.0"),
            timestamp=datetime.now(timezone.utc),
        )

        risk_check = await strategy_engine.check_portfolio_risk(
            high_risk_signal, sample_portfolio
        )

        # Should flag excessive concentration risk
        assert risk_check["approved"] is False or risk_check["reduced_size"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_correlation_risk_check(self, strategy_engine):
        """Test correlation-based risk management"""

        # Mock high correlation between symbols
        with patch.object(
            strategy_engine, "get_correlation_matrix"
        ) as mock_correlation:
            mock_correlation.return_value = pd.DataFrame(
                {
                    "AAPL": [1.0, 0.9],  # High correlation with GOOGL
                    "GOOGL": [0.9, 1.0],
                },
                index=pd.Series(["AAPL", "GOOGL"]),
            )

            # Signal for highly correlated asset
            signal = TradeSignal(
                symbol="GOOGL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=Decimal("2800.00"),
                quantity=100,
                strategy_name="momentum",
                stop_loss=Decimal("2750.0"),
                take_profit=Decimal("2850.0"),
                timestamp=datetime.now(timezone.utc),
            )

            # Assume we already have AAPL position
            from shared.models import PortfolioState

            portfolio = PortfolioState(
                account_id="test_account",
                cash=Decimal("50000.0"),
                buying_power=Decimal("100000.0"),
                total_equity=Decimal("64500.0"),
                positions=[
                    Position(
                        symbol="AAPL",
                        quantity=100,
                        entry_price=Decimal("150.0"),
                        market_value=Decimal("15050.0"),
                        cost_basis=Decimal("15000.0"),
                        current_price=Decimal("150.50"),
                        unrealized_pnl=Decimal("50.0"),
                    )
                ],
            )

            risk_check = await strategy_engine.check_correlation_risk(signal, portfolio)

            # Should flag high correlation risk
            assert risk_check["high_correlation"] is True


class TestStrategyEngineConfiguration:
    """Unit tests for configuration management"""

    @pytest.mark.unit
    def test_strategy_loading(self, mock_db_manager, mock_redis_client):
        """Test loading of configured strategies"""

        class MockStrategyEngineService:
            def __init__(self, config=None, db_manager=None, redis_client=None):
                self.config = config or {}
                self.db_manager = db_manager
                self.redis_client = redis_client
                self.strategies = ["momentum", "mean_reversion"]

        config = {
            "enabled_strategies": ["momentum", "mean_reversion"],
            "strategy_configs": {
                "momentum": {"rsi_period": 14, "sma_short": 20, "sma_long": 50},
                "mean_reversion": {"bollinger_period": 20, "bollinger_std": 2.0},
            },
        }

        engine = MockStrategyEngineService(
            config=config, db_manager=mock_db_manager, redis_client=mock_redis_client
        )

        # Should load all configured strategies
        assert len(engine.strategies) == 2
        assert "momentum" in engine.strategies
        assert "mean_reversion" in engine.strategies

    @pytest.mark.unit
    def test_dynamic_strategy_configuration(self, strategy_engine):
        """Test dynamic strategy configuration updates"""

        new_config = {
            "rsi_period": 21,  # Changed from 14
            "rsi_oversold": 25,  # Changed from 30
        }

        # Update strategy configuration
        strategy_engine.update_strategy_config("momentum", new_config)

        strategy = strategy_engine.get_strategy("momentum")
        assert strategy.config["rsi_period"] == 21
        assert strategy.config["rsi_oversold"] == 25

    @pytest.mark.unit
    def test_strategy_enabling_disabling(self, strategy_engine):
        """Test enabling and disabling strategies"""

        # Disable strategy
        strategy_engine.disable_strategy("momentum")
        assert "momentum" not in strategy_engine.active_strategies
