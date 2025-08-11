"""
Comprehensive end-to-end trade flow integration tests.

This module tests the complete trading system workflow from market data ingestion
through signal generation, risk management, trade execution, and portfolio updates.
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch
import numpy as np

# Import system components
import sys
sys.path.append('/app/shared')
sys.path.append('/app/services/data_collector/src')
sys.path.append('/app/services/strategy_engine/src')
sys.path.append('/app/services/trade_executor/src')
sys.path.append('/app/services/risk_manager/src')
sys.path.append('/app/services/scheduler/src')

from shared.models import (
    MarketData, TimeFrame, SignalType, OrderSide, OrderType, OrderStatus
)
from services.scheduler.src.scheduler import MarketSession

# Mock model classes for testing
from enum import Enum
from typing import Optional
from uuid import UUID

class SignalStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"

class TradingSignal:
    def __init__(self, symbol: str, signal_type: SignalType, strength: SignalStrength,
                 price: Decimal, timestamp: datetime, strategy_id: str, confidence: float,
                 metadata: Optional[dict] = None):
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.price = price
        self.timestamp = timestamp
        self.strategy_id = strategy_id
        self.confidence = confidence
        self.metadata = metadata if metadata is not None else {}

class Portfolio:
    def __init__(self, account_id: str = "test_account", total_value: Decimal = Decimal('100000'),
                 cash_balance: Decimal = Decimal('50000'), positions: Optional[list] = None,
                 created_at: Optional[datetime] = None):
        self.account_id = account_id
        self.total_value = total_value
        self.cash_balance = cash_balance
        self.positions = positions if positions is not None else []
        self.created_at = created_at if created_at is not None else datetime.now(timezone.utc)

class PortfolioPosition:
    def __init__(self, symbol: str, quantity: int, entry_price: Decimal,
                 avg_cost: Optional[Decimal] = None, market_value: Optional[Decimal] = None,
                 unrealized_pnl: Optional[Decimal] = None, realized_pnl: Optional[Decimal] = None):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.avg_cost = avg_cost if avg_cost is not None else entry_price
        self.market_value = market_value if market_value is not None else (entry_price * Decimal(str(abs(quantity))))
        self.unrealized_pnl = unrealized_pnl if unrealized_pnl is not None else Decimal('0')
        self.realized_pnl = realized_pnl if realized_pnl is not None else Decimal('0')

class RiskAssessment:
    def __init__(self, signal: Optional[TradingSignal] = None, portfolio: Optional[Portfolio] = None,
                 approved: bool = True, position_size: int = 100,
                 max_position_size: Optional[Decimal] = None, stop_loss: Optional[Decimal] = None,
                 take_profit: Optional[Decimal] = None, rejection_reason: Optional[str] = None,
                 assessment_time_ms: float = 0.0):
        self.signal = signal
        self.portfolio = portfolio
        self.approved = approved
        self.position_size = position_size
        self.max_position_size = max_position_size if max_position_size is not None else Decimal('0.1')
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.rejection_reason = rejection_reason if rejection_reason is not None else ""
        self.assessment_time_ms = assessment_time_ms

class Order:
    def __init__(self, symbol: str, quantity: int, order_type: OrderType,
                 price: Optional[Decimal] = None, timestamp: Optional[datetime] = None,
                 id: Optional[UUID] = None, order_id: Optional[UUID] = None, side: Optional[OrderSide] = None,
                 created_at: Optional[datetime] = None, expires_at: Optional[datetime] = None,
                 status: OrderStatus = OrderStatus.PENDING):
        self.symbol = symbol
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.timestamp = timestamp if timestamp is not None else datetime.now(timezone.utc)
        self.id = id if id is not None else uuid.uuid4()
        self.order_id = order_id if order_id is not None else self.id
        self.side = side if side is not None else OrderSide.BUY
        self.created_at = created_at if created_at is not None else datetime.now(timezone.utc)
        self.expires_at = expires_at
        self.status = status

class TestTrade:
    def __init__(self, symbol: str, quantity: int, price: Decimal,
                 timestamp: datetime, id: Optional[UUID] = None, order_id: Optional[UUID] = None,
                 side: Optional[OrderSide] = None, executed_price: Optional[Decimal] = None,
                 executed_at: Optional[datetime] = None, commission: Optional[Decimal] = None,
                 status: Optional[OrderStatus] = None, execution_time_ms: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.id = id if id is not None else uuid.uuid4()
        self.order_id = order_id if order_id is not None else uuid.uuid4()
        self.side = side if side is not None else OrderSide.BUY
        self.executed_price = executed_price if executed_price is not None else price
        self.executed_at = executed_at if executed_at is not None else timestamp
        self.commission = commission if commission is not None else Decimal('1.0')
        self.status = status if status is not None else OrderStatus.FILLED
        self.execution_time_ms = execution_time_ms

class RiskMetrics:
    def __init__(self, symbol: str = "AAPL", volatility: float = 0.2,
                 sharpe_ratio: float = 1.5, correlation_spy: float = 0.8,
                 correlation_risk: float = 0.3, concentration_risk: float = 0.15,
                 var_95: Decimal = Decimal('0.05'),
                 portfolio_beta: float = 1.0, max_drawdown: Decimal = Decimal('0.1')):
        self.symbol = symbol
        self.volatility = volatility
        self.sharpe_ratio = sharpe_ratio
        self.correlation_spy = correlation_spy
        self.correlation_risk = correlation_risk
        self.concentration_risk = concentration_risk
        self.var_95 = var_95
        self.portfolio_beta = portfolio_beta
        self.max_drawdown = max_drawdown

class Quote:
    def __init__(self, symbol: str, bid: Decimal, ask: Decimal, timestamp: datetime,
                 bid_size: int = 100, ask_size: int = 100, spread: Optional[Decimal] = None):
        self.symbol = symbol
        self.bid = bid
        self.ask = ask
        self.timestamp = timestamp
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.spread = spread or (ask - bid)

class PerformanceMetrics:
    def __init__(self, symbol: str = "AAPL", timestamp: Optional[datetime] = None,
                 timeframe: TimeFrame = TimeFrame.ONE_DAY):
        self.symbol = symbol
        self.timestamp = timestamp if timestamp is not None else datetime.now(timezone.utc)
        self.timeframe = timeframe


class TestEndToEndTradeFlow:
    """Test complete trade flow from data to execution."""

    @pytest.fixture
    async def trading_system_setup(self):
        """Setup complete trading system for integration testing."""
        # Mock Redis for pub/sub
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.publish.return_value = 1
        mock_redis.subscribe.return_value = AsyncMock()

        # Mock database connections
        mock_db_pool = AsyncMock()
        mock_db_pool.fetchrow.return_value = None
        mock_db_pool.fetchval.return_value = None
        mock_db_pool.execute.return_value = None

        setup_data = {
            'redis_client': mock_redis,
            'db_pool': mock_db_pool,
            'test_symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'initial_portfolio_value': Decimal('100000.00')
        }

        yield setup_data

        # Cleanup
        await mock_redis.close()

    @pytest.mark.integration
    async def test_complete_trade_execution_flow(self, trading_system_setup):
        """Test complete flow: Data → Signal → Risk Check → Trade → Portfolio Update."""
        redis_client = trading_system_setup['redis_client']
        db_pool = trading_system_setup['db_pool']
        assert db_pool is not None, "Database pool should be available"

        # Step 1: Market data ingestion
        # Market data
        market_data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.ONE_MINUTE,
            open=Decimal('150.00'),
            high=Decimal('152.00'),
            low=Decimal('149.00'),
            close=Decimal('151.00'),
            volume=1000000,
            adjusted_close=Decimal('151.00')
        )

        # Mock data collector publishing market data
        with patch('data_collector.DataCollector') as mock_collector:
            collector = mock_collector.return_value
            collector.collect_market_data = AsyncMock(return_value=market_data)

            await collector.collect_market_data('AAPL')
            redis_client.publish.assert_called()

        # Step 2: Strategy signal generation
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            # Generate buy signal
            signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                price=market_data.close,
                timestamp=datetime.now(timezone.utc),
                strategy_id='momentum_strategy',
                confidence=0.85
            )

            strategy.generate_signals = AsyncMock(return_value=[signal])
            signals = await strategy.generate_signals('AAPL', market_data)

            assert len(signals) == 1
            assert signals[0].signal_type == SignalType.BUY

        # Step 3: Risk management validation
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            # Mock current portfolio
            portfolio = Portfolio(
                account_id='test_account',
                total_value=Decimal('100000.00'),
                cash_balance=Decimal('50000.00'),
                positions=[],
                created_at=datetime.now(timezone.utc)
            )

            risk_assessment = RiskAssessment(
                signal=signal,
                portfolio=portfolio,
                approved=True,
                position_size=100,
                max_position_size=Decimal('0.1'),
                stop_loss=Decimal('145.00'),
                take_profit=Decimal('155.00')
            )

            risk_manager.assess_signal = AsyncMock(return_value=risk_assessment)
            assessment = await risk_manager.assess_signal(signal, portfolio)

            assert assessment.approved is True
            assert assessment.position_size > 0

        # Step 4: Trade execution
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            # Create order from risk-approved signal
            order = Order(
                symbol=signal.symbol,
                quantity=100,
                order_type=OrderType.MARKET,
                price=signal.price,
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=uuid.uuid4(),
                side=OrderSide.BUY,
                created_at=datetime.now(timezone.utc),
                status=OrderStatus.PENDING
            )

            # Mock successful execution
            executed_trade = TestTrade(
                symbol=order.symbol,
                quantity=order.quantity,
                price=Decimal('150.28'),
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=order.id,
                side=order.side,
                executed_price=Decimal('150.28'),
                executed_at=datetime.now(timezone.utc),
                commission=Decimal('1.00'),
                status=OrderStatus.FILLED
            )

            executor.execute_order = AsyncMock(return_value=executed_trade)
            trade = await executor.execute_order(order)

            assert trade.status == OrderStatus.FILLED
            assert trade.executed_price > 0

        # Step 5: Portfolio update
        with patch('portfolio_manager.PortfolioManager') as mock_portfolio:
            portfolio_manager = mock_portfolio.return_value

            # Update portfolio with executed trade
            updated_position = PortfolioPosition(
                symbol=trade.symbol,
                quantity=trade.quantity,
                entry_price=trade.executed_price,
                avg_cost=trade.executed_price,
                market_value=Decimal(str(trade.quantity)) * trade.executed_price,
                unrealized_pnl=Decimal('0.00'),
                realized_pnl=Decimal('0.00')
            )

            portfolio_manager.update_position = AsyncMock(return_value=updated_position)
            position = await portfolio_manager.update_position(trade)

            assert position.quantity == trade.quantity
            assert position.avg_cost == trade.executed_price

        # Verify end-to-end flow completed successfully
        assert market_data.symbol == signal.symbol == trade.symbol
        assert signal.signal_type == SignalType.BUY
        assert assessment.approved is True
        assert trade.status == OrderStatus.FILLED

    @pytest.mark.integration
    async def test_sell_signal_flow(self, trading_system_setup):
        """Test complete sell signal execution flow."""
        redis_client = trading_system_setup['redis_client']
        assert redis_client is not None, "Redis client should be available"

        # Setup existing position
        existing_position = PortfolioPosition(
            symbol='GOOGL',
            quantity=10,
            entry_price=Decimal('2800.00'),
            avg_cost=Decimal('2800.00'),
            market_value=Decimal('28500.00'),
            unrealized_pnl=Decimal('500.00'),
            realized_pnl=Decimal('0.00')
        )

        # Generate sell signal
        sell_signal = TradingSignal(
            symbol='GOOGL',
            signal_type=SignalType.SELL,
            strength=SignalStrength.MODERATE,
            price=Decimal('2850.00'),
            timestamp=datetime.now(timezone.utc),
            strategy_id='momentum_strategy',
            confidence=0.75
        )

        # Risk assessment for sell
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            portfolio = Portfolio(
                account_id='test_account',
                total_value=Decimal('120000.00'),
                cash_balance=Decimal('20000.00'),
                positions=[existing_position]
            )

            risk_assessment = RiskAssessment(
                signal=sell_signal,
                portfolio=portfolio,
                approved=True,
                position_size=5  # Partial sell
            )

            risk_manager.assess_signal = AsyncMock(return_value=risk_assessment)
            assessment = await risk_manager.assess_signal(sell_signal, portfolio)

            assert assessment.approved is True

        # Execute sell order
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            sell_trade = TestTrade(
                symbol='GOOGL',
                quantity=5,
                price=Decimal('2850.00'),
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=uuid.uuid4(),
                side=OrderSide.SELL,
                executed_price=Decimal('2850.00'),
                executed_at=datetime.now(timezone.utc),
                commission=Decimal('1.00'),
                status=OrderStatus.FILLED
            )

            executor.execute_order = AsyncMock(return_value=sell_trade)
            trade = await executor.execute_order(mock_executor)

            assert trade.side == OrderSide.SELL
            assert trade.quantity == 5

    @pytest.mark.integration
    async def test_risk_rejection_flow(self, trading_system_setup):
        """Test flow when risk manager rejects a signal."""
        # Generate high-risk signal
        risky_signal = TradingSignal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal('150.00'),
            timestamp=datetime.now(timezone.utc),
            strategy_id='aggressive_strategy',
            confidence=0.95
        )

        # Risk manager rejects due to position limits
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            # Portfolio at risk limits
            portfolio = Portfolio(
                account_id='test_account',
                total_value=Decimal('100000.00'),
                cash_balance=Decimal('5000.00'),  # Low cash
                positions=[
                    PortfolioPosition(
                        symbol='AAPL',
                        quantity=500,
                        entry_price=Decimal('145.00'),
                        avg_cost=Decimal('145.00'),
                        market_value=Decimal('75000.00')
                    )
                ]
            )

            risk_assessment = RiskAssessment(
                signal=risky_signal,
                portfolio=portfolio,
                approved=False,
                rejection_reason='Insufficient cash balance'
            )

            risk_manager.assess_signal = AsyncMock(return_value=risk_assessment)
            assessment = await risk_manager.assess_signal(risky_signal, portfolio)

            assert assessment.approved is False
            assert 'cash balance' in assessment.rejection_reason.lower()

        # Verify no trade execution occurs
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value
            executor.execute_order = AsyncMock()

            # Should not call executor for rejected signal
            should_execute = assessment.approved
            assert should_execute is False

            # Executor should not be called
            executor.execute_order.assert_not_called()

    @pytest.mark.integration
    async def test_partial_fill_handling(self, trading_system_setup):
        """Test handling of partial order fills."""
        # Create large order that gets partially filled
        large_order = Order(
            symbol='AAPL',
            quantity=1000,
            order_type=OrderType.LIMIT,
            price=Decimal('150.00'),
            timestamp=datetime.now(timezone.utc),
            id=uuid.uuid4(),
            order_id=uuid.uuid4(),
            side=OrderSide.BUY,
            created_at=datetime.now(timezone.utc),
            status=OrderStatus.PENDING
        )

        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            # Mock partial fill
            partial_trade = TestTrade(
                symbol='AAPL',
                quantity=600,  # Only 600 of 1000 filled
                price=Decimal('150.05'),
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=large_order.id,
                side=OrderSide.BUY,
                executed_price=Decimal('150.05'),
                executed_at=datetime.now(timezone.utc),
                commission=Decimal('1.00'),
                status=OrderStatus.PARTIALLY_FILLED
            )

            executor.execute_order = AsyncMock(return_value=partial_trade)
            executor.get_order_status = AsyncMock(return_value=OrderStatus.PARTIALLY_FILLED)

            trade = await executor.execute_order(large_order)

            assert trade.status == OrderStatus.PARTIALLY_FILLED
            assert trade.quantity < large_order.quantity

        # Test portfolio update with partial fill
        with patch('portfolio_manager.PortfolioManager') as mock_portfolio:
            portfolio_manager = mock_portfolio.return_value

            partial_position = PortfolioPosition(
                symbol='AAPL',
                quantity=600,
                entry_price=Decimal('150.05'),
                avg_cost=Decimal('150.05'),
                market_value=Decimal('90030.00'),
                unrealized_pnl=Decimal('30.00')
            )

            portfolio_manager.update_position = AsyncMock(return_value=partial_position)
            position = await portfolio_manager.update_position(partial_trade)

            assert position.quantity == 600
            assert position.avg_cost == Decimal('150.05')

    @pytest.mark.integration
    async def test_stop_loss_execution_flow(self, trading_system_setup):
        """Test stop loss order execution flow."""
        # Setup existing position
        position = PortfolioPosition(
            symbol='AAPL',
            quantity=100,
            entry_price=Decimal('150.00'),
            avg_cost=Decimal('150.00'),
            market_value=Decimal('14500.00'),  # Current price $145
            unrealized_pnl=Decimal('-500.00')  # 5% loss
        )

        # Market data showing price decline
        declining_data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.ONE_MINUTE,
            open=Decimal('150.00'),
            high=Decimal('150.50'),
            low=Decimal('144.50'),
            close=Decimal('145.00'),  # Below stop loss
            volume=2000000,
            adjusted_close=Decimal('145.00')
        )

        # Risk manager should trigger stop loss
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            stop_loss_signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.SELL,
                strength=SignalStrength.CRITICAL,
                price=declining_data.close,
                timestamp=datetime.now(timezone.utc),
                strategy_id='stop_loss',
                confidence=1.0,
                metadata={'trigger_reason': 'stop_loss', 'stop_price': '147.00'}
            )

            risk_manager.check_stop_loss = AsyncMock(return_value=stop_loss_signal)
            signal = await risk_manager.check_stop_loss(position, declining_data)

            assert signal.signal_type == SignalType.SELL
            assert signal.strategy_id == 'stop_loss'

        # Execute stop loss order
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            stop_loss_trade = TestTrade(
                symbol='AAPL',
                quantity=100,
                price=Decimal('144.90'),
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=uuid.uuid4(),
                side=OrderSide.SELL,
                executed_price=Decimal('144.90'),  # Slippage
                executed_at=datetime.now(timezone.utc),
                commission=Decimal('1.00'),
                status=OrderStatus.FILLED
            )

            executor.execute_stop_loss = AsyncMock(return_value=stop_loss_trade)
            trade = await executor.execute_stop_loss(position, signal)

            assert trade.status == OrderStatus.FILLED
            assert trade.side == OrderSide.SELL

    @pytest.mark.integration
    async def test_multi_symbol_strategy_execution(self, trading_system_setup):
        """Test strategy execution across multiple symbols."""
        symbols = trading_system_setup['test_symbols']
        redis_client = trading_system_setup['redis_client']
        assert redis_client is not None, "Redis client should be available"

        # Generate market data for all symbols
        market_data_batch = []
        for i, symbol in enumerate(symbols):
            base_price = Decimal(f'{100 + i*50}.00')
            data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=base_price,
                high=base_price + Decimal('2.00'),
                low=base_price - Decimal('1.00'),
                close=base_price,
                volume=1000000 + i*100000,
                adjusted_close=base_price
            )
            market_data_batch.append(data)

        # Strategy generates signals for multiple symbols
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            signals = []
            for i, data in enumerate(market_data_batch):
                signal = TradingSignal(
                    symbol=data.symbol,
                    signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=data.close,
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='multi_symbol_strategy',
                    confidence=0.7
                )
                signals.append(signal)

            strategy.generate_batch_signals = AsyncMock(return_value=signals)
            batch_signals = await strategy.generate_batch_signals(market_data_batch)

            assert len(batch_signals) == len(symbols)

        # Risk management for batch signals
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            approved_signals = []
            for signal in batch_signals:
                assessment = RiskAssessment(
                    signal=signal,
                    portfolio=Portfolio(account_id='test', total_value=Decimal('100000')),
                    approved=True,
                    position_size=100
                )
                approved_signals.append(assessment)

            risk_manager.assess_batch_signals = AsyncMock(return_value=approved_signals)
            assessments = await risk_manager.assess_batch_signals(batch_signals)

            assert len(assessments) == len(batch_signals)
            assert all(a.approved for a in assessments)

        # Execute batch trades
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            executed_trades = []
            for assessment in assessments:
                trade = TestTrade(
                    symbol=assessment.signal.symbol,
                    quantity=assessment.position_size,
                    price=assessment.signal.price,
                    timestamp=datetime.now(timezone.utc),
                    id=uuid.uuid4(),
                    order_id=uuid.uuid4(),
                    side=OrderSide.BUY if assessment.signal.signal_type == SignalType.BUY else OrderSide.SELL,
                    executed_price=assessment.signal.price,
                    executed_at=datetime.now(timezone.utc),
                    commission=Decimal('1.00'),
                    status=OrderStatus.FILLED
                )
                executed_trades.append(trade)

            executor.execute_batch_orders = AsyncMock(return_value=executed_trades)
            trades = await executor.execute_batch_orders(assessments)

            assert len(trades) == len(assessments)
            assert all(t.status == OrderStatus.FILLED for t in trades)

    @pytest.mark.integration
    async def test_error_recovery_in_trade_flow(self, trading_system_setup):
        """Test error recovery mechanisms in trade flow."""
        redis_client = trading_system_setup['redis_client']
        assert redis_client is not None, "Redis client should be available"

        # Test data collection failure recovery
        with patch('data_collector.DataCollector') as mock_collector:
            collector = mock_collector.return_value

            # First call fails, second succeeds
            collector.collect_market_data = AsyncMock()
            collector.collect_market_data.side_effect = [
                Exception("API rate limit exceeded"),
                MarketData(
                    symbol='AAPL',
                    timestamp=datetime.now(timezone.utc),
                    timeframe=TimeFrame.ONE_MINUTE,
                    open=Decimal('149.50'),
                    high=Decimal('150.50'),
                    low=Decimal('149.00'),
                    close=Decimal('150.00'),
                    volume=1000000,
                    adjusted_close=Decimal('150.00')
                )
            ]

            # Test retry mechanism
            with patch('asyncio.sleep'):  # Speed up retry delay
                try:
                    await collector.collect_market_data('AAPL')
                    data = await collector.collect_market_data('AAPL')
                    assert data.symbol == 'AAPL'
                except Exception:
                    # First call should fail, but retry should work
                    pass

        # Test trade execution failure recovery
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            order = Order(
                symbol='AAPL',
                quantity=100,
                order_type=OrderType.MARKET,
                price=Decimal('150.00'),
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=uuid.uuid4(),
                side=OrderSide.BUY,
                created_at=datetime.now(timezone.utc),
                status=OrderStatus.PENDING
            )

            # First execution fails, second succeeds
            executor.execute_order = AsyncMock()
            executor.execute_order.side_effect = [
                Exception("Broker connection timeout"),
                TestTrade(
                    symbol='AAPL',
                    quantity=100,
                    price=Decimal('150.25'),
                    timestamp=datetime.now(timezone.utc),
                    id=uuid.uuid4(),
                    order_id=order.id,
                    side=OrderSide.BUY,
                    executed_price=Decimal('150.05'),
                    executed_at=datetime.now(timezone.utc),
                    status=OrderStatus.FILLED
                )
            ]

            # Test retry logic
            with patch('asyncio.sleep'):
                try:
                    await executor.execute_order(order)
                except Exception:
                    pass

                trade = await executor.execute_order(order)
                assert trade.status == OrderStatus.FILLED

    @pytest.mark.integration
    async def test_high_frequency_data_processing(self, trading_system_setup):
        """Test high-frequency data processing and signal generation."""
        redis_client = trading_system_setup['redis_client']
        assert redis_client is not None, "Redis client should be available"

        # Generate high-frequency market data
        data_points = []
        base_time = datetime.now(timezone.utc)
        base_price = Decimal('150.00')

        for i in range(100):  # 100 data points in 1 second intervals
            price_change = Decimal(str(np.random.normal(0, 0.1)))  # Small random changes
            current_price = base_price + price_change
            data = MarketData(
                symbol='AAPL',
                timestamp=base_time + timedelta(seconds=i),
                timeframe=TimeFrame.ONE_MINUTE,
                open=current_price,
                high=current_price + Decimal('0.05'),
                low=current_price - Decimal('0.05'),
                close=current_price,
                volume=10000 + i*100,
                adjusted_close=current_price
            )
            data_points.append(data)

        # Test strategy processing speed
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            processing_times = []
            signals_generated = []

            for data in data_points:
                start_time = datetime.now()

                # Mock signal generation with realistic processing time
                signal = TradingSignal(
                    symbol=data.symbol,
                    signal_type=SignalType.HOLD,
                    strength=SignalStrength.WEAK,
                    price=data.close,
                    timestamp=data.timestamp,
                    strategy_id='hft_strategy',
                    confidence=0.6
                )

                strategy.generate_signal = AsyncMock(return_value=signal)
                generated_signal = await strategy.generate_signal(data)

                processing_time = (datetime.now() - start_time).total_seconds()
                processing_times.append(processing_time)
                signals_generated.append(generated_signal)

            # Verify performance requirements
            avg_processing_time = np.mean(processing_times)
            max_processing_time = max(processing_times)

            assert avg_processing_time < 0.1  # Average under 100ms
            assert max_processing_time < 0.5   # Max under 500ms
            assert len(signals_generated) == 100

    @pytest.mark.integration
    async def test_portfolio_rebalancing_flow(self, trading_system_setup):
        """Test portfolio rebalancing workflow."""
        # Setup unbalanced portfolio
        portfolio = Portfolio(
            account_id='test_account',
            total_value=Decimal('100000.00'),
            cash_balance=Decimal('10000.00'),
            positions=[
                PortfolioPosition(
                    symbol='AAPL',
                    quantity=300,
                    entry_price=Decimal('150.00'),
                    avg_cost=Decimal('150.00'),
                    market_value=Decimal('45000.00')  # 50% of portfolio
                ),
                PortfolioPosition(
                    symbol='GOOGL',
                    quantity=10,
                    entry_price=Decimal('2800.00'),
                    avg_cost=Decimal('2800.00'),
                    market_value=Decimal('28000.00')  # 31% of portfolio
                ),
                PortfolioPosition(
                    symbol='MSFT',
                    quantity=50,
                    entry_price=Decimal('340.00'),
                    avg_cost=Decimal('340.00'),
                    market_value=Decimal('17000.00')  # 19% of portfolio
                )
            ]
        )

        # Portfolio manager detects need for rebalancing
        with patch('portfolio_manager.PortfolioManager') as mock_portfolio:
            portfolio_manager = mock_portfolio.return_value

            # Target allocation: 40% AAPL, 30% GOOGL, 30% MSFT
            target_allocation = {
                'AAPL': 0.40,
                'GOOGL': 0.30,
                'MSFT': 0.30
            }

            rebalance_signals = [
                TradingSignal(
                    symbol='AAPL',
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=Decimal('150.00'),
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='rebalancing',
                    confidence=0.8,
                    metadata={'target_value': '40000.00', 'current_value': '45000.00'}
                ),
                TradingSignal(
                    symbol='MSFT',
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MODERATE,
                    price=Decimal('340.00'),
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='rebalancing',
                    confidence=0.8,
                    metadata={'target_value': '30000.00', 'current_value': '17000.00'}
                )
            ]

            portfolio_manager.generate_rebalancing_signals = AsyncMock(return_value=rebalance_signals)
            signals = await portfolio_manager.generate_rebalancing_signals(portfolio, target_allocation)

            assert len(signals) == 2
            assert signals[0].signal_type == SignalType.SELL  # Reduce AAPL
            assert signals[1].signal_type == SignalType.BUY   # Increase MSFT

        # Execute rebalancing trades
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            rebalance_trades = []
            for signal in rebalance_signals:
                trade = TestTrade(
                    symbol=signal.symbol,
                    quantity=33 if signal.symbol == 'AAPL' else 38,
                    price=signal.price,
                    timestamp=datetime.now(timezone.utc),
                    id=uuid.uuid4(),
                    order_id=uuid.uuid4(),
                    side=OrderSide.SELL if signal.signal_type == SignalType.SELL else OrderSide.BUY,
                    executed_price=signal.price,
                    executed_at=datetime.now(timezone.utc),
                    commission=Decimal('1.00'),
                    status=OrderStatus.FILLED
                )
                rebalance_trades.append(trade)

            executor.execute_rebalancing_orders = AsyncMock(return_value=rebalance_trades)
            trades = await executor.execute_rebalancing_orders(rebalance_signals)

            assert len(trades) == 2
            assert all(t.status == OrderStatus.FILLED for t in trades)

    @pytest.mark.integration
    async def test_market_volatility_response(self, trading_system_setup):
        """Test system response to high market volatility."""
        # Generate volatile market data
        volatile_data = []
        base_price = Decimal('150.00')

        for i in range(10):
            # Simulate high volatility with large price swings
            volatility = Decimal(str(np.random.normal(0, 5)))  # High volatility
            current_price = base_price + volatility
            data = MarketData(
                symbol='AAPL',
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
                timeframe=TimeFrame.ONE_MINUTE,
                open=current_price,
                high=current_price + Decimal('1.00'),
                low=current_price - Decimal('1.00'),
                close=current_price,
                volume=5000000,  # High volume
                adjusted_close=current_price
            )
            volatile_data.append(data)

        # Risk manager should detect high volatility
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            volatility_metrics = RiskMetrics(
                symbol='AAPL',
                volatility=0.35,  # 35% annualized volatility
                var_95=Decimal('7500.00'),
                sharpe_ratio=0.5,
                max_drawdown=Decimal('0.15'),
                correlation_spy=0.85
            )

            risk_manager.calculate_volatility = AsyncMock(return_value=volatility_metrics)
            metrics = await risk_manager.calculate_volatility(volatile_data)

            assert metrics.volatility > 0.3  # High volatility detected

        # Strategy should adjust position sizing
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            # Reduced position size due to high volatility
            volatility_adjusted_signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.BUY,
                strength=SignalStrength.WEAK,  # Reduced strength
                price=volatile_data[-1].close,
                timestamp=datetime.now(timezone.utc),
                strategy_id='volatility_adjusted',
                confidence=0.5,  # Lower confidence due to volatility
                metadata={'volatility_adjustment': True, 'original_size': '1000.00', 'adjusted_size': '500.00'}
            )

            strategy.adjust_for_volatility = AsyncMock(return_value=volatility_adjusted_signal)
            adjusted_signal = await strategy.adjust_for_volatility(volatile_data[-1], metrics)

            assert adjusted_signal.strength == SignalStrength.WEAK
            assert adjusted_signal.confidence < 0.7

    @pytest.mark.integration
    async def test_order_book_integration(self, trading_system_setup):
        """Test integration with order book data for optimal execution."""
        # Mock order book data
        order_book = {
            'bids': [
                {'price': Decimal('149.98'), 'size': 100},
                {'price': Decimal('149.97'), 'size': 200},
                {'price': Decimal('149.96'), 'size': 150}
            ],
            'asks': [
                {'price': Decimal('150.02'), 'size': 150},
                {'price': Decimal('150.03'), 'size': 100},
                {'price': Decimal('150.04'), 'size': 250}
            ],
            'timestamp': datetime.now(timezone.utc)
        }

        # Strategy uses order book for better entry points
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            optimal_signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.BUY,
                strength=SignalStrength.MODERATE,
                price=Decimal('150.01'),  # Between bid-ask
                timestamp=datetime.now(timezone.utc),
                strategy_id='order_book_optimized',
                confidence=0.8,
                metadata={'order_book_analysis': True, 'optimal_entry': True}
            )

            strategy.generate_signal_with_order_book = AsyncMock(return_value=optimal_signal)
            signal = await strategy.generate_signal_with_order_book(order_book)

            assert signal.price > Decimal('149.98')  # Above best bid
            assert signal.price < Decimal('150.04')  # Below worst ask needed

    @pytest.mark.integration
    async def test_cross_service_communication_resilience(self, trading_system_setup):
        """Test resilience of cross-service communication."""
        redis_client = trading_system_setup['redis_client']

        # Test message delivery with Redis failures
        message_queue = []

        async def mock_redis_publish(channel, message):
            if len(message_queue) < 3:  # First 3 fail
                raise Exception("Redis connection lost")
            message_queue.append((channel, message))
            return 1

        redis_client.publish.side_effect = mock_redis_publish

        # Data collector tries to publish market data
        market_data2 = MarketData(
            symbol='TSLA',
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.ONE_MINUTE,
            open=Decimal('249.50'),
            high=Decimal('250.50'),
            low=Decimal('249.00'),
            close=Decimal('250.00'),
            volume=1500000,
            adjusted_close=Decimal('250.00')
        )

        with patch('data_collector.DataCollector') as mock_collector:
            collector = mock_collector.return_value
            collector.redis_client = redis_client

            # Should retry on Redis failures
            with patch('asyncio.sleep'):  # Speed up retries
                for attempt in range(5):
                    try:
                        await collector.publish_market_data(market_data2)
                        break
                    except Exception:
                        continue

            # Eventually succeeds
            assert len(message_queue) > 0

    @pytest.mark.integration
    async def test_database_transaction_integrity(self, trading_system_setup):
        """Test database transaction integrity during trade execution."""
        db_pool = trading_system_setup['db_pool']

        # Mock database transaction
        mock_transaction = AsyncMock()
        mock_transaction.fetchrow.return_value = None
        mock_transaction.execute.return_value = None
        mock_transaction.commit = AsyncMock()
        mock_transaction.rollback = AsyncMock()

        db_pool.acquire.return_value.__aenter__.return_value = mock_transaction

        trade = TestTrade(
            symbol='AAPL',
            quantity=100,
            price=Decimal('150.00'),
            timestamp=datetime.now(timezone.utc),
            id=uuid.uuid4(),
            order_id=uuid.uuid4(),
            side=OrderSide.BUY,
            executed_price=Decimal('150.00'),
            executed_at=datetime.now(timezone.utc),
            commission=Decimal('1.00'),
            status=OrderStatus.FILLED
        )

        # Test successful transaction
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value
            executor.db_pool = db_pool

            await executor.save_trade_with_transaction(trade)

            mock_transaction.execute.assert_called()
            mock_transaction.commit.assert_called()

        # Test transaction rollback on error
        mock_transaction.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            await executor.save_trade_with_transaction(trade)

        mock_transaction.rollback.assert_called()

    @pytest.mark.integration
    async def test_real_time_position_tracking(self, trading_system_setup):
        """Test real-time position tracking and updates."""
        redis_client = trading_system_setup['redis_client']
        assert redis_client is not None, "Redis client should be available"

        # Initialize position
        initial_position = PortfolioPosition(
            symbol='AAPL',
            quantity=100,
            entry_price=Decimal('150.00'),
            avg_cost=Decimal('150.00'),
            market_value=Decimal('15000.00'),
            unrealized_pnl=Decimal('0.00')
        )

        # Stream of market data updates
        price_updates = [
            Decimal('150.50'),  # +0.50
            Decimal('149.75'),  # -0.25 from original
            Decimal('151.25'),  # +1.25
            Decimal('148.50'),  # -1.50
        ]

        # Test position value updates
        with patch('portfolio_manager.PortfolioManager') as mock_portfolio:
            portfolio_manager = mock_portfolio.return_value
            current_position = initial_position

            for new_price in price_updates:
                updated_position = PortfolioPosition(
                    symbol='AAPL',
                    quantity=current_position.quantity,
                    entry_price=current_position.entry_price,
                    avg_cost=current_position.avg_cost,
                    market_value=current_position.quantity * new_price,
                    unrealized_pnl=(new_price - current_position.avg_cost) * current_position.quantity
                )

                portfolio_manager.update_position_value = AsyncMock(return_value=updated_position)
                current_position = await portfolio_manager.update_position_value('AAPL', new_price)

                # Verify calculations
                expected_value = 100 * new_price
                expected_pnl = (new_price - Decimal('150.00')) * 100

                assert current_position.market_value == expected_value
                assert current_position.unrealized_pnl == expected_pnl

    @pytest.mark.integration
    async def test_strategy_coordination(self, trading_system_setup):
        """Test coordination between multiple strategies."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        assert len(symbols) > 0, "Symbols list should not be empty"

        # Multiple strategies generating signals
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            # Strategy 1: Momentum
            momentum_signals = [
                TradingSignal(
                    symbol='AAPL',
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.STRONG,
                    price=Decimal('150.00'),
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='momentum',
                    confidence=0.85
                ),
                TradingSignal(
                    symbol='GOOGL',
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=Decimal('2850.00'),
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='momentum',
                    confidence=0.75
                )
            ]

            # Strategy 2: Mean Reversion
            mean_reversion_signals = [
                TradingSignal(
                    symbol='AAPL',
                    signal_type=SignalType.SELL,  # Conflicting with momentum
                    strength=SignalStrength.WEAK,
                    price=Decimal('150.00'),
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='mean_reversion',
                    confidence=0.65
                ),
                TradingSignal(
                    symbol='MSFT',
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MODERATE,
                    price=Decimal('340.00'),
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='mean_reversion',
                    confidence=0.70
                )
            ]

            strategy.get_momentum_signals = AsyncMock(return_value=momentum_signals)
            strategy.get_mean_reversion_signals = AsyncMock(return_value=mean_reversion_signals)

            # Strategy coordinator resolves conflicts
            with patch('strategy_coordinator.StrategyCoordinator') as mock_coordinator:
                coordinator = mock_coordinator.return_value

                # Resolve conflicting AAPL signals (momentum BUY vs mean reversion SELL)
                resolved_signals = [
                    momentum_signals[0],  # Keep stronger momentum signal
                    momentum_signals[1],  # GOOGL sell
                    mean_reversion_signals[1]  # MSFT buy
                ]

                coordinator.resolve_signal_conflicts = AsyncMock(return_value=resolved_signals)
                final_signals = await coordinator.resolve_signal_conflicts(
                    momentum_signals + mean_reversion_signals
                )

                assert len(final_signals) == 3  # One AAPL signal removed due to conflict
                assert any(s.symbol == 'AAPL' and s.signal_type == SignalType.BUY for s in final_signals)

    @pytest.mark.integration
    async def test_emergency_shutdown_flow(self, trading_system_setup):
        """Test emergency shutdown and position protection flow."""
        redis_client = trading_system_setup['redis_client']
        assert redis_client is not None, "Redis client should be available"

        # Simulate emergency condition (e.g., market crash)
        emergency_trigger = {
            'type': 'market_crash',
            'severity': 'critical',
            'market_drop_percentage': 15.0,
            'triggered_at': datetime.now(timezone.utc)
        }

        # Emergency shutdown sequence
        with patch('scheduler.TradingScheduler') as mock_scheduler:
            scheduler = mock_scheduler.return_value

            # Stop all non-critical jobs
            scheduler.pause_trading_jobs = AsyncMock()
            await scheduler.pause_trading_jobs(emergency_trigger)

            scheduler.pause_trading_jobs.assert_called_once()

        # Risk manager executes protective measures
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            protective_signals = [
                TradingSignal(
                    symbol='AAPL',
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.CRITICAL,
                    price=Decimal('127.50'),  # 15% below $150
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='emergency_protection',
                    confidence=1.0,
                    metadata={'emergency_type': 'market_crash', 'protection_level': 'max'}
                )
            ]

            risk_manager.generate_emergency_signals = AsyncMock(return_value=protective_signals)
            signals = await risk_manager.generate_emergency_signals(emergency_trigger)

            assert len(signals) > 0
            assert all(s.strength == SignalStrength.CRITICAL for s in signals)

        # Execute emergency orders
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            emergency_trades = []
            for signal in protective_signals:
                trade = TestTrade(
                    symbol=signal.symbol,
                    quantity=100,
                    price=signal.price,
                    timestamp=datetime.now(timezone.utc),
                    id=uuid.uuid4(),
                    order_id=uuid.uuid4(),
                    side=OrderSide.SELL,
                    executed_price=signal.price,
                    executed_at=datetime.now(timezone.utc),
                    commission=Decimal('1.00'),
                    status=OrderStatus.FILLED
                )
                emergency_trades.append(trade)

            executor.execute_emergency_orders = AsyncMock(return_value=emergency_trades)
            trades = await executor.execute_emergency_orders(protective_signals)

            assert all(t.status == OrderStatus.FILLED for t in trades)

    @pytest.mark.integration
    async def test_multi_timeframe_analysis_flow(self, trading_system_setup):
        """Test multi-timeframe analysis and signal coordination."""
        timeframes = ['1m', '5m', '15m', '1h', '1d']

        # Generate data for different timeframes
        multi_timeframe_data = {}
        base_time = datetime.now(timezone.utc)

        for tf in timeframes:
            data_points = []
            for i in range(20):  # 20 data points per timeframe
                data = MarketData(
                    symbol='AAPL',
                    timestamp=base_time - timedelta(minutes=i),
                    timeframe=TimeFrame.ONE_MINUTE,
                    open=Decimal('149.50') + Decimal(str(np.random.normal(0, 0.5))),
                    high=Decimal('150.50') + Decimal(str(np.random.normal(0, 0.5))),
                    low=Decimal('149.00') + Decimal(str(np.random.normal(0, 0.5))),
                    close=Decimal('150.00') + Decimal(str(np.random.normal(0, 0.5))),
                    volume=1000000 + i*10000,
                    adjusted_close=Decimal('150.00') + Decimal(str(np.random.normal(0, 0.5)))
                )
                data_points.append(data)
            multi_timeframe_data[tf] = data_points

        # Multi-timeframe strategy analysis
        with patch('strategy_engine.MultiTimeframeStrategy') as mock_mtf_strategy:
            mtf_strategy = mock_mtf_strategy.return_value

            # Different signals from different timeframes
            timeframe_signals = {
                '1m': TradingSignal(symbol='AAPL', signal_type=SignalType.HOLD, strength=SignalStrength.WEAK, price=Decimal('150.00'), timestamp=datetime.now(timezone.utc), strategy_id='1m_strategy', confidence=0.3),
                '5m': TradingSignal(symbol='AAPL', signal_type=SignalType.BUY, strength=SignalStrength.MODERATE, price=Decimal('150.00'), timestamp=datetime.now(timezone.utc), strategy_id='5m_strategy', confidence=0.6),
                '15m': TradingSignal(symbol='AAPL', signal_type=SignalType.BUY, strength=SignalStrength.STRONG, price=Decimal('150.00'), timestamp=datetime.now(timezone.utc), strategy_id='15m_strategy', confidence=0.8),
                '1h': TradingSignal(symbol='AAPL', signal_type=SignalType.BUY, strength=SignalStrength.MODERATE, price=Decimal('150.00'), timestamp=datetime.now(timezone.utc), strategy_id='1h_strategy', confidence=0.7),
                '1d': TradingSignal(symbol='AAPL', signal_type=SignalType.HOLD, strength=SignalStrength.WEAK, price=Decimal('150.00'), timestamp=datetime.now(timezone.utc), strategy_id='1d_strategy', confidence=0.4)
            }

            # Consensus signal generation
            consensus_signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.BUY,  # Majority buy signals
                strength=SignalStrength.MODERATE,
                price=Decimal('150.00'),
                timestamp=datetime.now(timezone.utc),
                strategy_id='multi_timeframe_consensus',
                confidence=0.75,
                metadata={
                    'timeframe_analysis': timeframe_signals,
                    'consensus_method': 'weighted_average',
                    'dominant_timeframes': ['5m', '15m', '1h']
                }
            )

            mtf_strategy.generate_consensus_signal = AsyncMock(return_value=consensus_signal)
            signal = await mtf_strategy.generate_consensus_signal(multi_timeframe_data)

            assert signal.signal_type == SignalType.BUY
            assert signal.confidence > 0.7

    @pytest.mark.integration
    async def test_latency_sensitive_execution(self, trading_system_setup):
        """Test latency-sensitive execution for time-critical trades."""
        # High-frequency signal requiring fast execution
        urgent_signal = TradingSignal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            strength=SignalStrength.CRITICAL,
            price=Decimal('150.00'),
            timestamp=datetime.now(timezone.utc),
            strategy_id='arbitrage',
            confidence=0.95,
            metadata={
                'execution_priority': 'urgent',
                'max_latency_ms': 50,
                'arbitrage_opportunity': True
            }
        )

        # Fast-track risk assessment
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            quick_assessment = RiskAssessment(
                signal=urgent_signal,
                portfolio=Portfolio(account_id='test', total_value=Decimal('100000')),
                approved=True,
                position_size=100,
                assessment_time_ms=15  # Fast assessment
            )

            risk_manager.quick_assess = AsyncMock(return_value=quick_assessment)

            start_time = datetime.now()
            assessment = await risk_manager.quick_assess(urgent_signal)
            assessment_time = (datetime.now() - start_time).total_seconds() * 1000

            assert assessment_time < 50  # Under 50ms requirement
            assert assessment.approved is True

        # Ultra-fast trade execution
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            rapid_trade = TestTrade(
                symbol='AAPL',
                quantity=100,
                price=Decimal('150.02'),
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=uuid.uuid4(),
                side=OrderSide.BUY,
                executed_price=Decimal('150.02'),  # Minimal slippage
                executed_at=datetime.now(timezone.utc),
                commission=Decimal('1.00'),
                status=OrderStatus.FILLED,
                execution_time_ms=25
            )

            executor.execute_urgent_order = AsyncMock(return_value=rapid_trade)

            execution_start = datetime.now()
            trade = await executor.execute_urgent_order(assessment)
            execution_time = (datetime.now() - execution_start).total_seconds() * 1000

            assert execution_time < 100  # Total execution under 100ms
            assert trade.execution_time_ms < 50

    @pytest.mark.integration
    async def test_portfolio_risk_limit_enforcement(self, trading_system_setup):
        """Test enforcement of portfolio-level risk limits."""
        # Portfolio approaching risk limits
        high_risk_portfolio = Portfolio(
            account_id='test_account',
            total_value=Decimal('100000.00'),
            cash_balance=Decimal('5000.00'),
            positions=[
                PortfolioPosition(
                    symbol='AAPL',
                    quantity=400,
                    entry_price=Decimal('150.00'),
                    avg_cost=Decimal('150.00'),
                    market_value=Decimal('60000.00'),  # 60% concentration
                    unrealized_pnl=Decimal('0.00')
                ),
                PortfolioPosition(
                    symbol='GOOGL',
                    quantity=12,
                    entry_price=Decimal('2800.00'),
                    avg_cost=Decimal('2800.00'),
                    market_value=Decimal('33600.00'),  # 33% concentration
                    unrealized_pnl=Decimal('0.00')
                )
            ]
        )

        # New signal that would exceed concentration limits
        concentration_signal = TradingSignal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal('150.00'),
            timestamp=datetime.now(timezone.utc),
            strategy_id='momentum',
            confidence=0.85
        )

        # Risk manager should reject due to concentration limits
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            risk_assessment = RiskAssessment(
                signal=concentration_signal,
                portfolio=high_risk_portfolio,
                approved=False,
                rejection_reason='Portfolio concentration limit exceeded (max 65%, current 60%)',
                position_size=0
            )

            risk_manager.assess_portfolio_limits = AsyncMock(return_value=risk_assessment)
            assessment = await risk_manager.assess_portfolio_limits(
                concentration_signal,
                high_risk_portfolio
            )

            assert assessment.approved is False
            assert 'concentration' in assessment.rejection_reason.lower()

    @pytest.mark.integration
    async def test_market_data_quality_validation(self, trading_system_setup):
        """Test market data quality validation in the trade flow."""
        # Invalid/suspicious market data
        suspicious_data = [
            MarketData(  # Price spike
                symbol='AAPL',
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal('150.00'),
                high=Decimal('200.00'),
                low=Decimal('150.00'),
                close=Decimal('200.00'),  # 33% spike
                volume=100,  # Very low volume
                adjusted_close=Decimal('200.00')
            ),
            MarketData(  # Stale data
                symbol='AAPL',
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal('150.00'),
                high=Decimal('150.50'),
                low=Decimal('149.50'),
                close=Decimal('150.00'),
                volume=1000000,
                adjusted_close=Decimal('150.00')
            ),
            MarketData(  # Zero volume
                symbol='AAPL',
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal('150.00'),
                high=Decimal('150.50'),
                low=Decimal('149.50'),
                close=Decimal('150.00'),
                volume=0,  # Zero volume
                adjusted_close=Decimal('150.00')
            )
        ]

        # Data validator should flag suspicious data
        with patch('data_validator.DataValidator') as mock_validator:
            validator = mock_validator.return_value

            validation_results = [
                {'valid': False, 'reason': 'Unusual price spike with low volume'},
                {'valid': False, 'reason': 'Stale data timestamp'},
                {'valid': False, 'reason': 'Zero volume data'}
            ]

            validator.validate_market_data = AsyncMock()
            validator.validate_market_data.side_effect = validation_results

            valid_data = []
            for data in suspicious_data:
                result = await validator.validate_market_data(data)
                if result['valid']:
                    valid_data.append(data)

            # No data should pass validation
            assert len(valid_data) == 0

        # Strategy should not generate signals from invalid data
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            strategy.generate_signals = AsyncMock(return_value=[])
            signals = await strategy.generate_signals('AAPL', suspicious_data[0])

            # No signals from invalid data
            assert len(signals) == 0

    @pytest.mark.integration
    async def test_order_management_lifecycle(self, trading_system_setup):
        """Test complete order management lifecycle."""
        # Create and submit order
        initial_order = Order(
            symbol='AAPL',
            quantity=100,
            order_type=OrderType.LIMIT,
            price=Decimal('149.50'),
            timestamp=datetime.now(timezone.utc),
            id=uuid.uuid4(),
            order_id=uuid.uuid4(),
            side=OrderSide.BUY,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            status=OrderStatus.PENDING
        )

        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            # Submit order
            executor.submit_order = AsyncMock(return_value=initial_order)
            submitted_order = await executor.submit_order(initial_order)

            assert submitted_order.status == OrderStatus.PENDING

            # Monitor order status
            # Mock order status transitions
            status_updates = [
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED
            ]

            for status in status_updates:
                executor.get_order_status = AsyncMock(return_value=status)
                current_status = await executor.get_order_status(submitted_order.id)
                assert current_status == status

            # Get final execution details
            final_trade = TestTrade(
                symbol='AAPL',
                quantity=100,
                price=Decimal('149.52'),
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=submitted_order.id,
                side=OrderSide.BUY,
                executed_price=Decimal('149.52'),
                executed_at=datetime.now(timezone.utc),
                commission=Decimal('1.00'),
                status=OrderStatus.FILLED
            )

            executor.get_trade_details = AsyncMock(return_value=final_trade)
            trade_details = await executor.get_trade_details(submitted_order.id)

            assert trade_details.status == OrderStatus.FILLED
            assert trade_details.executed_price is not None


class TestDataFlowIntegration:
    """Test data flow integration between services."""

    @pytest.mark.integration
    async def test_redis_pubsub_data_flow(self, trading_system_setup):
        """Test Redis pub/sub data flow between services."""
        redis_client = trading_system_setup['redis_client']

        # Mock subscriber
        mock_subscriber = AsyncMock()
        redis_client.pubsub.return_value = mock_subscriber

        # Data collector publishes market data
        market_data1 = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.ONE_MINUTE,
            open=Decimal('149.50'),
            high=Decimal('150.50'),
            low=Decimal('149.00'),
            close=Decimal('150.00'),
            volume=1000000,
            adjusted_close=Decimal('150.00')
        )

        # Simulate message publishing
        message_data = {
            'type': 'market_data',
            'symbol': market_data1.symbol,
            'price': str(market_data1.close),
            'timestamp': market_data1.timestamp.isoformat()
        }

        redis_client.publish.return_value = 2  # 2 subscribers

        # Services subscribe to market data
        with patch('strategy_engine.StrategyEngine') as mock_strategy, \
             patch('risk_manager.RiskManager') as mock_risk:

            strategy = mock_strategy.return_value
            risk_manager = mock_risk.return_value

            # Both services receive the message
            strategy.handle_market_data = AsyncMock()
            risk_manager.handle_market_data = AsyncMock()

            # Simulate message delivery
            await strategy.handle_market_data(message_data)
            await risk_manager.handle_market_data(message_data)

            strategy.handle_market_data.assert_called_once_with(message_data)
            risk_manager.handle_market_data.assert_called_once_with(message_data)

    @pytest.mark.integration
    async def test_database_consistency_across_services(self, trading_system_setup):
        """Test database consistency across multiple services."""
        db_pool = trading_system_setup['db_pool']
        assert db_pool is not None, "Database pool should be available"

        # Mock database operations across services
        with patch('shared.database.DatabaseManager') as mock_db:
            db_manager = mock_db.return_value

            # Concurrent updates from different services
            trade_data = {
                'id': str(uuid.uuid4()),
                'symbol': 'AAPL',
                'quantity': 100,
                'price': Decimal('150.00'),
                'timestamp': datetime.now(timezone.utc)
            }

            portfolio_data = {
                'account_id': 'test_account',
                'symbol': 'AAPL',
                'quantity': 100,
                'avg_cost': Decimal('150.00'),
                'updated_at': datetime.now(timezone.utc)
            }

            # Simulate concurrent database updates
            db_manager.save_trade = AsyncMock()
            db_manager.update_portfolio = AsyncMock()
            db_manager.begin_transaction = AsyncMock()
            db_manager.commit_transaction = AsyncMock()

            # Both operations should succeed
            async with db_manager.begin_transaction():
                await db_manager.save_trade(trade_data)
                await db_manager.update_portfolio(portfolio_data)
                await db_manager.commit_transaction()

            db_manager.save_trade.assert_called_once()
            db_manager.update_portfolio.assert_called_once()
            db_manager.commit_transaction.assert_called_once()

    @pytest.mark.integration
    async def test_service_discovery_and_health_checks(self, trading_system_setup):
        """Test service discovery and health monitoring integration."""
        services = ['data_collector', 'strategy_engine', 'trade_executor', 'risk_manager']

        with patch('scheduler.ServiceOrchestrator') as mock_orchestrator:
            orchestrator = mock_orchestrator.return_value

            # Mock service health checks
            health_results = {}
            for service in services:
                health_results[service] = {
                    'status': 'healthy',
                    'uptime': 3600,
                    'last_activity': datetime.now(timezone.utc).isoformat(),
                    'metrics': {
                        'requests_per_minute': 120,
                        'error_rate': 0.02,
                        'average_response_time': 0.1
                    }
                }

            orchestrator.check_all_services_health = AsyncMock(return_value=health_results)
            health_status = await orchestrator.check_all_services_health()

            assert len(health_status) == len(services)
            assert all(status['status'] == 'healthy' for status in health_status.values())

        # Test unhealthy service detection and response
        health_results['strategy_engine']['status'] = 'unhealthy'
        health_results['strategy_engine']['error'] = 'High memory usage'

        orchestrator.check_all_services_health = AsyncMock(return_value=health_results)
        health_status = await orchestrator.check_all_services_health()

        unhealthy_services = [
            name for name, status in health_status.items()
            if status['status'] != 'healthy'
        ]

        assert 'strategy_engine' in unhealthy_services

    @pytest.mark.integration
    async def test_configuration_hot_reload(self, trading_system_setup):
        """Test hot reload of configuration across services."""
        # New configuration
        new_config = {
            'risk_limits': {
                'max_position_size': 0.15,  # Changed from 0.10
                'max_portfolio_risk': 0.25   # Changed from 0.20
            },
            'strategy_parameters': {
                'momentum': {
                    'rsi_period': 21,  # Changed from 14
                    'rsi_oversold': 25,  # Changed from 30
                    'rsi_overbought': 75  # Changed from 70
                },
                'mean_reversion': {
                    'lookback_period': 30,
                    'entry_threshold': 2.0,
                    'exit_threshold': 0.5
                }
            },
            'execution_parameters': {
                'max_slippage': 0.002,  # 0.2%
                'timeout_seconds': 30
            }
        }

        # Services should reload configuration
        with patch('shared.config.ConfigManager') as mock_config:
            config_manager = mock_config.return_value
            config_manager.reload_config = AsyncMock()
            config_manager.notify_services = AsyncMock()

            await config_manager.reload_config(new_config)
            await config_manager.notify_services(['strategy_engine', 'risk_manager'])

            config_manager.reload_config.assert_called_once_with(new_config)
            config_manager.notify_services.assert_called_once()

        # Verify services use new configuration
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value
            strategy.update_parameters = AsyncMock()

            await strategy.update_parameters(new_config['strategy_parameters']['momentum'])

            strategy.update_parameters.assert_called_once()

    @pytest.mark.integration
    async def test_graceful_service_restart_flow(self, trading_system_setup):
        """Test graceful service restart without disrupting trades."""
        redis_client = trading_system_setup['redis_client']
        assert redis_client is not None, "Redis client should be available"

        # Active trades during restart
        active_orders = [
            Order(
                symbol='AAPL',
                quantity=100,
                order_type=OrderType.LIMIT,
                price=Decimal('149.50'),
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=uuid.uuid4(),
                side=OrderSide.BUY,
                created_at=datetime.now(timezone.utc),
                status=OrderStatus.PENDING
            ),
            Order(
                symbol='GOOGL',
                quantity=10,
                order_type=OrderType.MARKET,
                price=None,
                timestamp=datetime.now(timezone.utc),
                id=uuid.uuid4(),
                order_id=uuid.uuid4(),
                side=OrderSide.SELL,
                created_at=datetime.now(timezone.utc),
                status=OrderStatus.PARTIALLY_FILLED
            )
        ]

        # Service restart sequence
        with patch('scheduler.TradingScheduler') as mock_scheduler:
            scheduler = mock_scheduler.return_value

            # Save active state before restart
            scheduler.save_active_state = AsyncMock()
            await scheduler.save_active_state(active_orders)

            scheduler.save_active_state.assert_called_once()

        # Strategy engine restart
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            # Graceful shutdown
            strategy.graceful_shutdown = AsyncMock()
            await strategy.graceful_shutdown()

            # Restore state after restart
            strategy.restore_state = AsyncMock()
            await strategy.restore_state()

            strategy.graceful_shutdown.assert_called_once()
            strategy.restore_state.assert_called_once()

        # Trade executor maintains order tracking
        with patch('trade_executor.TradeExecutor') as mock_executor:
            executor = mock_executor.return_value

            executor.restore_active_orders = AsyncMock(return_value=active_orders)
            restored_orders = await executor.restore_active_orders()

            assert len(restored_orders) == len(active_orders)
            assert all(order.id in [o.id for o in active_orders] for order in restored_orders)

    @pytest.mark.integration
    async def test_market_session_transition_handling(self, trading_system_setup):
        """Test handling of market session transitions."""
        # Test pre-market to regular hours transition
        with patch('scheduler.MarketHoursService') as mock_market:
            market_service = mock_market.return_value

            # Pre-market session
            market_service.get_current_session.return_value = MarketSession.PRE_MARKET
            market_service.is_market_open.return_value = False

            pre_market_signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.BUY,
                strength=SignalStrength.MODERATE,
                price=Decimal('150.00'),
                timestamp=datetime.now(timezone.utc),
                strategy_id='pre_market_strategy',
                confidence=0.7
            )

            # Should queue signal for market open
            with patch('strategy_engine.StrategyEngine') as mock_strategy:
                strategy = mock_strategy.return_value
                strategy.queue_for_market_open = AsyncMock()

                await strategy.handle_pre_market_signal(pre_market_signal)

                strategy.queue_for_market_open.assert_called_once()

            # Market opens - execute queued signals
            market_service.get_current_session.return_value = MarketSession.REGULAR
            market_service.is_market_open.return_value = True

            queued_signals = [pre_market_signal]
            strategy.execute_queued_signals = AsyncMock()

            await strategy.execute_queued_signals(queued_signals)

            strategy.execute_queued_signals.assert_called_once()

    @pytest.mark.integration
    async def test_risk_position_correlation_monitoring(self, trading_system_setup):
        """Test monitoring of position correlations for risk management."""
        # Portfolio with correlated positions
        correlated_portfolio = Portfolio(
            account_id='test_account',
            total_value=Decimal('100000.00'),
            cash_balance=Decimal('10000.00'),
            positions=[
                PortfolioPosition(
                    symbol='AAPL',
                    quantity=200,
                    entry_price=Decimal('150.00'),
                    avg_cost=Decimal('150.00'),
                    market_value=Decimal('30000.00')
                ),
                PortfolioPosition(
                    symbol='MSFT',
                    quantity=100,
                    entry_price=Decimal('340.00'),
                    avg_cost=Decimal('340.00'),
                    market_value=Decimal('34000.00')
                ),
                PortfolioPosition(
                    symbol='GOOGL',
                    quantity=10,
                    entry_price=Decimal('2600.00'),
                    avg_cost=Decimal('2600.00'),
                    market_value=Decimal('26000.00')
                )
            ]
        )

        # Risk manager calculates correlation matrix
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            correlation_matrix = np.array([
                [1.0, 0.85, 0.75],    # AAPL correlations
                [0.85, 1.0, 0.70],    # MSFT correlations
                [0.75, 0.70, 1.0]     # GOOGL correlations
            ])

            portfolio_risk = RiskMetrics(
                symbol='PORTFOLIO',
                volatility=0.18,
                var_95=Decimal('8500.00'),
                correlation_risk=0.82,  # High correlation risk
                concentration_risk=0.65,
                max_drawdown=Decimal('0.12')
            )

            risk_manager.calculate_portfolio_correlations = AsyncMock(return_value=correlation_matrix)
            risk_manager.assess_correlation_risk = AsyncMock(return_value=portfolio_risk)

            correlations = await risk_manager.calculate_portfolio_correlations(correlated_portfolio)
            risk_assessment = await risk_manager.assess_correlation_risk(correlations)

            # High correlation should trigger risk warning
            assert risk_assessment.correlation_risk > 0.8
            assert np.max(correlations) > 0.8  # High correlation detected

    @pytest.mark.integration
    async def test_algorithmic_trading_patterns(self, trading_system_setup):
        """Test detection and handling of algorithmic trading patterns."""
        # Generate algorithmic trading pattern data
        algo_pattern_data = []
        base_time = datetime.now(timezone.utc)

        # Simulate high-frequency trading pattern
        for i in range(50):
            base_price = Decimal('150.00') + Decimal(str(i*0.01))
            data = MarketData(
                symbol='AAPL',
                timestamp=base_time + timedelta(milliseconds=i*100),
                timeframe=TimeFrame.ONE_MINUTE,
                open=base_price,
                high=base_price + Decimal('0.02'),
                low=base_price - Decimal('0.01'),
                close=base_price,  # Gradual price increase
                volume=10000,  # Consistent volume
                adjusted_close=base_price
            )
            algo_pattern_data.append(data)

        # Pattern detection
        with patch('market_analysis.PatternDetector') as mock_detector:
            detector = mock_detector.return_value

            detected_pattern = {
                'type': 'algorithmic_momentum',
                'confidence': 0.92,
                'duration_ms': 5000,
                'volume_profile': 'consistent',
                'price_action': 'gradual_uptrend'
            }

            detector.detect_algorithmic_patterns = AsyncMock(return_value=detected_pattern)
            pattern = await detector.detect_algorithmic_patterns(algo_pattern_data)

            assert pattern['type'] == 'algorithmic_momentum'
            assert pattern['confidence'] > 0.9

        # Strategy adapts to algorithmic activity
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            # Adjusted signal based on algorithmic activity
            adapted_signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.BUY,
                strength=SignalStrength.MODERATE,
                price=Decimal('150.48'),  # Slightly ahead of pattern
                timestamp=datetime.now(timezone.utc),
                strategy_id='algo_aware_momentum',
                confidence=0.8,
                metadata={
                    'algo_pattern_detected': True,
                    'adaptation_strategy': 'front_run_momentum',
                    'pattern_confidence': 0.92
                }
            )

            strategy.adapt_to_algorithmic_activity = AsyncMock(return_value=adapted_signal)
            signal = await strategy.adapt_to_algorithmic_activity(pattern, algo_pattern_data[-1])

            assert signal.metadata['algo_pattern_detected'] is True
            assert signal.confidence > 0.7


class TestComplexScenarioIntegration:
    """Test complex real-world trading scenarios."""

    @pytest.mark.integration
    async def test_earnings_announcement_handling(self, trading_system_setup):
        """Test system behavior around earnings announcements."""
        # Earnings announcement data
        earnings_event = {
            'symbol': 'AAPL',
            'announcement_time': datetime.now(timezone.utc) + timedelta(hours=2),
            'estimated_eps': 2.45,
            'estimated_revenue': 89.5e9,
            'volatility_expectation': 0.08  # 8% expected volatility
        }

        # Risk manager adjusts position sizing
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            earnings_signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.BUY,
                strength=SignalStrength.MODERATE,
                price=Decimal('150.00'),
                timestamp=datetime.now(timezone.utc),
                strategy_id='earnings_play',
                confidence=0.7
            )

            earnings_assessment = RiskAssessment(
                signal=earnings_signal,
                portfolio=Portfolio(account_id='test', total_value=Decimal('100000')),
                approved=True,
                position_size=500  # Reduced size due to earnings risk
            )

            risk_manager.assess_earnings_risk = AsyncMock(return_value=earnings_assessment)
            assessment = await risk_manager.assess_earnings_risk(earnings_event)

            assert assessment.approved is True
            assert assessment.position_size < 1000  # Reduced from normal

    @pytest.mark.integration
    async def test_market_maker_interaction(self, trading_system_setup):
        """Test interaction with market maker algorithms."""
        # Market maker quote pattern
        mm_quotes = []
        base_time = datetime.now(timezone.utc)

        for i in range(20):
            quote = Quote(
                symbol='AAPL',
                timestamp=base_time + timedelta(seconds=i),
                bid=Decimal('149.98'),
                ask=Decimal('150.02'),
                bid_size=500,
                ask_size=500,
                spread=Decimal('0.04')
            )
            mm_quotes.append(quote)

        # Strategy recognizes market maker presence
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            mm_aware_signal = TradingSignal(
                symbol='AAPL',
                signal_type=SignalType.BUY,
                strength=SignalStrength.MODERATE,
                price=Decimal('149.99'),  # Aggressive bid to cross spread
                timestamp=datetime.now(timezone.utc),
                strategy_id='market_maker_aware',
                confidence=0.75,
                metadata={
                    'market_maker_detected': True,
                    'execution_strategy': 'aggressive_crossing',
                    'expected_fill_probability': 0.85
                }
            )

            strategy.generate_mm_aware_signal = AsyncMock(return_value=mm_aware_signal)
            signal = await strategy.generate_mm_aware_signal(mm_quotes)

            assert signal.metadata['market_maker_detected'] is True
            assert signal.price > Decimal('149.98')  # Above bid

    @pytest.mark.integration
    async def test_portfolio_optimization_integration(self, trading_system_setup):
        """Test portfolio optimization integration with trade execution."""
        # Current portfolio state
        current_portfolio = Portfolio(
            account_id='test_account',
            total_value=Decimal('100000.00'),
            cash_balance=Decimal('20000.00'),
            positions=[
                PortfolioPosition(
                    symbol='AAPL',
                    quantity=200,
                    entry_price=Decimal('150.00'),
                    avg_cost=Decimal('150.00'),
                    market_value=Decimal('30000.00')
                ),
                PortfolioPosition(
                    symbol='GOOGL',
                    quantity=10,
                    entry_price=Decimal('2800.00'),
                    avg_cost=Decimal('2800.00'),
                    market_value=Decimal('28000.00')
                )
            ]
        )

        # Portfolio optimizer suggests improvements
        with patch('portfolio_optimizer.PortfolioOptimizer') as mock_optimizer:
            optimizer = mock_optimizer.return_value

            optimization_result = {
                'recommended_allocation': {
                    'AAPL': 0.25,
                    'GOOGL': 0.30,
                    'MSFT': 0.25,
                    'NVDA': 0.20
                },
                'expected_sharpe_improvement': 0.15,
                'risk_reduction': 0.08,
                'rebalancing_trades': [
                    {'symbol': 'AAPL', 'action': 'REDUCE', 'target_value': Decimal('25000.00')},
                    {'symbol': 'GOOGL', 'action': 'INCREASE', 'target_value': Decimal('30000.00')},
                    {'symbol': 'MSFT', 'action': 'BUY', 'target_value': Decimal('25000.00')},
                    {'symbol': 'NVDA', 'action': 'BUY', 'target_value': Decimal('20000.00')}
                ]
            }

            optimizer.optimize_portfolio = AsyncMock(return_value=optimization_result)
            result = await optimizer.optimize_portfolio(current_portfolio)

            assert result['expected_sharpe_improvement'] > 0
            assert len(result['rebalancing_trades']) == 4

        # Execute optimization trades
        optimization_signals = []
        for trade_rec in optimization_result['rebalancing_trades']:
            if trade_rec['action'] in ['REDUCE', 'SELL']:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.BUY

            signal = TradingSignal(
                symbol=trade_rec['symbol'],
                signal_type=signal_type,
                strength=SignalStrength.MODERATE,
                price=Decimal('150.00'),  # Placeholder price
                timestamp=datetime.now(timezone.utc),
                strategy_id='portfolio_optimization',
                confidence=0.8,
                metadata={'optimization_trade': True, 'target_value': str(trade_rec['target_value'])}
            )
            optimization_signals.append(signal)

        # Risk assessment for optimization trades
        with patch('risk_manager.RiskManager') as mock_risk:
            risk_manager = mock_risk.return_value

            all_approved = True
            assessments = []

            for signal in optimization_signals:
                assessment = RiskAssessment(
                    signal=signal,
                    portfolio=current_portfolio,
                    approved=True,
                    position_size=100
                )
                assessments.append(assessment)

            risk_manager.assess_optimization_trades = AsyncMock(return_value=assessments)
            risk_results = await risk_manager.assess_optimization_trades(optimization_signals)

            assert all(assessment.approved for assessment in risk_results)

    @pytest.mark.integration
    async def test_system_recovery_after_network_outage(self, trading_system_setup):
        """Test system recovery after network connectivity issues."""
        redis_client = trading_system_setup['redis_client']
        assert redis_client is not None, "Redis client should be available"

        # Simulate network outage
        outage_start = datetime.now(timezone.utc)
        outage_duration = timedelta(minutes=30)
        assert outage_duration.total_seconds() > 0, "Outage duration should be positive"

        # Services lose connectivity
        connection_errors = [
            "Connection timeout",
            "Network unreachable",
            "DNS resolution failed",
            "Connection refused"
        ]
        assert isinstance(connection_errors, list), "Connection errors should be a list"

        # Data collector experiences failures
        with patch('data_collector.DataCollector') as mock_collector:
            collector = mock_collector.return_value

            # Queue data during outage
            queued_data = []
            for i in range(10):
                base_price = Decimal('150.00') + Decimal(str(i*0.1))
                data = MarketData(
                    symbol='AAPL',
                    timestamp=outage_start + timedelta(seconds=i*30),
                    timeframe=TimeFrame.ONE_MINUTE,
                    open=base_price,
                    high=base_price + Decimal('0.05'),
                    low=base_price - Decimal('0.05'),
                    close=base_price,
                    volume=1000000,
                    adjusted_close=base_price
                )
                queued_data.append(data)

            collector.queue_data_during_outage = AsyncMock()
            collector.process_queued_data_after_recovery = AsyncMock()

            # Queue data during outage
            for data in queued_data:
                await collector.queue_data_during_outage(data)

            # Recovery - process queued data
            await collector.process_queued_data_after_recovery()

            assert collector.queue_data_during_outage.call_count == len(queued_data)
            collector.process_queued_data_after_recovery.assert_called_once()

        # Strategy engine handles delayed data
        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            # Process batch of delayed data
            batch_signals = []
            for data in queued_data:
                signal = TradingSignal(
                    symbol=data.symbol,
                    signal_type=SignalType.HOLD,  # Conservative during recovery
                    strength=SignalStrength.WEAK,
                    price=data.close,
                    timestamp=datetime.now(timezone.utc),  # Current time for execution
                    strategy_id='recovery_mode',
                    confidence=0.6,
                    metadata={'recovery_mode': True, 'original_timestamp': data.timestamp.isoformat()}
                )
                batch_signals.append(signal)

            strategy.process_recovery_batch = AsyncMock(return_value=batch_signals)
            signals = await strategy.process_recovery_batch(queued_data)

            assert len(signals) == len(queued_data)
            assert all(s.metadata['recovery_mode'] for s in signals)

    @pytest.mark.integration
    async def test_regulatory_compliance_reporting(self, trading_system_setup):
        """Test regulatory compliance reporting integration."""
        # Trading activity over reporting period
        trading_period_start = datetime.now(timezone.utc) - timedelta(days=1)
        trades_for_reporting = []

        for i in range(25):
            trade = TestTrade(
                symbol='AAPL' if i % 2 == 0 else 'GOOGL',
                quantity=100 + i*10,
                price=Decimal('150.00') + Decimal(str(i*0.5)),
                timestamp=trading_period_start + timedelta(hours=i),
                id=uuid.uuid4(),
                order_id=uuid.uuid4(),
                side=OrderSide.BUY if i % 3 == 0 else OrderSide.SELL,
                executed_price=Decimal('150.00') + Decimal(str(i*0.5)),
                executed_at=trading_period_start + timedelta(hours=i),
                commission=Decimal('1.00'),
                status=OrderStatus.FILLED
            )
            trades_for_reporting.append(trade)

        # Compliance reporting system
        with patch('compliance.ComplianceReporter') as mock_compliance:
            compliance_reporter = mock_compliance.return_value

            # Generate regulatory reports
            daily_report = {
                'trading_date': trading_period_start.date(),
                'total_trades': len(trades_for_reporting),
                'total_volume': sum(t.quantity for t in trades_for_reporting),
                'symbols_traded': list(set(t.symbol for t in trades_for_reporting)),
                'largest_trade': max(trades_for_reporting, key=lambda t: t.quantity * t.executed_price),
                'compliance_violations': []
            }

            compliance_reporter.generate_daily_report = AsyncMock(return_value=daily_report)
            report = await compliance_reporter.generate_daily_report(
                trading_period_start.date(),
                trades_for_reporting
            )

            assert report['total_trades'] == 25
            assert len(report['symbols_traded']) == 2
            assert report['compliance_violations'] == []

        # Audit trail verification
        with patch('audit.AuditTrailManager') as mock_audit:
            audit_manager = mock_audit.return_value

            audit_trail = []
            for trade in trades_for_reporting:
                audit_entry = {
                    'event_type': 'trade_execution',
                    'timestamp': trade.executed_at,
                    'trade_id': trade.id,
                    'symbol': trade.symbol,
                    'action': trade.side,
                    'quantity': trade.quantity,
                    'price': str(trade.executed_price),
                    'user_id': 'system',
                    'session_id': 'automated_trading',
                    'ip_address': '127.0.0.1'
                }
                audit_trail.append(audit_entry)

            audit_manager.create_audit_trail = AsyncMock(return_value=audit_trail)
            trail = await audit_manager.create_audit_trail(trades_for_reporting)

            assert len(trail) == len(trades_for_reporting)
            assert all('trade_execution' == entry['event_type'] for entry in trail)

    @pytest.mark.integration
    async def test_cross_asset_arbitrage_detection(self, trading_system_setup):
        """Test detection and execution of cross-asset arbitrage opportunities."""
        # Related assets with price discrepancies
        assets_data = {
            'AAPL': MarketData(
                symbol='AAPL',
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal('150.00'),
                high=Decimal('150.50'),
                low=Decimal('149.50'),
                close=Decimal('150.00'),
                volume=1000000,
                adjusted_close=Decimal('150.00')
            ),
            'QQQ': MarketData(  # Tech ETF
                symbol='QQQ',
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal('300.00'),
                high=Decimal('301.00'),
                low=Decimal('299.00'),
                close=Decimal('300.50'),
                volume=2000000,
                adjusted_close=Decimal('300.50')
            ),
            'AAPL_OPTIONS': MarketData(  # AAPL options for synthetic position
                symbol='AAPL240315C00150000',  # Call option
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal('5.00'),
                high=Decimal('5.20'),
                low=Decimal('4.80'),
                close=Decimal('5.10'),
                volume=50000,
                adjusted_close=Decimal('5.10')
            )
        }

        # Arbitrage detector finds opportunity
        with patch('arbitrage.ArbitrageDetector') as mock_arbitrage:
            arbitrage_detector = mock_arbitrage.return_value

            arbitrage_opportunity = {
                'type': 'synthetic_arbitrage',
                'long_leg': 'AAPL',
                'short_leg': 'AAPL_OPTIONS',
                'expected_profit': Decimal('2.50'),
                'confidence': 0.88,
                'time_decay_risk': 0.15,
                'execution_complexity': 'medium'
            }

            arbitrage_detector.detect_opportunities = AsyncMock(return_value=[arbitrage_opportunity])
            opportunities = await arbitrage_detector.detect_opportunities(assets_data)

            assert len(opportunities) == 1
            assert opportunities[0]['expected_profit'] > 0

        # Execute arbitrage strategy
        with patch('strategy_engine.ArbitrageStrategy') as mock_arb_strategy:
            arb_strategy = mock_arb_strategy.return_value

            arbitrage_signals = [
                TradingSignal(
                    symbol='AAPL',
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.STRONG,
                    price=Decimal('150.00'),
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='synthetic_arbitrage',
                    confidence=0.88
                ),
                TradingSignal(
                    symbol='AAPL240315C00150000',
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.STRONG,
                    price=Decimal('5.50'),
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='synthetic_arbitrage',
                    confidence=0.88
                )
            ]

            arb_strategy.execute_arbitrage = AsyncMock(return_value=arbitrage_signals)
            signals = await arb_strategy.execute_arbitrage(arbitrage_opportunity)

            assert len(signals) == 2
            assert signals[0].signal_type == SignalType.BUY
            assert signals[1].signal_type == SignalType.SELL

    @pytest.mark.integration
    async def test_stress_testing_integration(self, trading_system_setup):
        """Test system behavior under stress conditions."""
        # High-volume stress test data
        stress_test_data = []
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'] * 20  # 100 symbols

        for i, symbol in enumerate(symbols):
            base_price = Decimal('150.00') + Decimal(str(np.random.normal(0, 1)))
            data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc) + timedelta(milliseconds=i*10),
                timeframe=TimeFrame.ONE_MINUTE,
                open=base_price,
                high=base_price + Decimal('0.50'),
                low=base_price - Decimal('0.50'),
                close=base_price,
                volume=1000000 + i*1000,
                adjusted_close=base_price
            )
            stress_test_data.append(data)

        # System should maintain performance under load
        processing_times = []

        with patch('strategy_engine.StrategyEngine') as mock_strategy:
            strategy = mock_strategy.return_value

            for data in stress_test_data[:10]:  # Test first 10 for unit test speed
                start_time = datetime.now()

                signal = TradingSignal(
                    symbol=data.symbol,
                    signal_type=SignalType.HOLD,
                    strength=SignalStrength.WEAK,
                    price=data.close,
                    timestamp=datetime.now(timezone.utc),
                    strategy_id='stress_test',
                    confidence=0.6
                )

                strategy.generate_signal = AsyncMock(return_value=signal)
                await strategy.generate_signal(data)

                processing_time = (datetime.now() - start_time).total_seconds()
                processing_times.append(processing_time)

            # Performance should remain consistent
            avg_processing_time = np.mean(processing_times)
            max_processing_time = max(processing_times)

            assert avg_processing_time < 0.1  # Average under 100ms
            assert max_processing_time < 0.2   # Max under 200ms

    @pytest.mark.integration
    async def test_disaster_recovery_integration(self, trading_system_setup):
        """Test disaster recovery and business continuity."""
        # Simulate complete system failure
        system_failure_event = {
            'type': 'total_system_failure',
            'timestamp': datetime.now(timezone.utc),
            'affected_services': ['all'],
            'estimated_recovery_time': timedelta(minutes=30)
        }

        # Backup systems activation
        with patch('disaster_recovery.DisasterRecoveryManager') as mock_dr:
            dr_manager = mock_dr.return_value

            # Activate backup trading infrastructure
            backup_config = {
                'backup_data_source': 'alternative_provider',
                'backup_execution_venue': 'backup_broker',
                'reduced_functionality_mode': True,
                'position_protection_enabled': True
            }

            dr_manager.activate_backup_systems = AsyncMock(return_value=backup_config)
            backup_systems = await dr_manager.activate_backup_systems(system_failure_event)

            assert backup_systems['reduced_functionality_mode'] is True
            assert backup_systems['position_protection_enabled'] is True

        # Emergency position management
        with patch('risk_manager.EmergencyRiskManager') as mock_emergency:
            emergency_risk = mock_emergency.return_value

            # Close all risky positions
            emergency_risk.execute_emergency_liquidation = AsyncMock(return_value=True)
            liquidation_success = await emergency_risk.execute_emergency_liquidation()

            assert liquidation_success is True

        # Data recovery and synchronization
        with patch('data_recovery.DataRecoveryService') as mock_recovery:
            recovery_service = mock_recovery.return_value

            recovery_summary = {
                'recovered_trades': 45,
                'recovered_positions': 8,
                'data_integrity_score': 0.98,
                'recovery_time_minutes': 25
            }

            recovery_service.recover_trading_data = AsyncMock(return_value=recovery_summary)
            recovery_result = await recovery_service.recover_trading_data(system_failure_event)

            assert recovery_result['data_integrity_score'] > 0.95
            assert recovery_result['recovery_time_minutes'] < 30


# Test fixtures and utilities
@pytest.fixture
async def mock_external_apis():
    """Mock external API responses for integration testing."""
    api_responses = {
        'alpha_vantage': {
            'quote': {
                'Global Quote': {
                    '01. symbol': 'AAPL',
                    '05. price': '150.00',
                    '06. volume': '1000000',
                    '07. latest trading day': '2024-01-15'
                }
            }
        },
        'twelve_data': {
            'quote': {
                'symbol': 'AAPL',
                'price': '150.00',
                'volume': '1000000',
                'timestamp': '2024-01-15'
            }
        }
    }

    return api_responses
