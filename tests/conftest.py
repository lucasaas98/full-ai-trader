import asyncio
import json
import shutil

# Import shared modules
import sys
import tempfile

sys.path.append("/app/shared")

from datetime import datetime, timedelta, timezone  # noqa: E402
from decimal import Decimal  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import (  # noqa: E402
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
)
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import redis  # noqa: E402

from shared.models import (  # noqa: E402
    OrderRequest,
    OrderSide,
    OrderType,
    PortfolioState,
    Position,
    SignalType,
    Trade,
    TradeSignal,
)


class MockDatabaseManager:
    """Mock database manager for testing"""

    def __init__(self) -> None:
        pass

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


class MockMarketData:
    """Mock market data for testing"""

    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> None:
        self.symbol = symbol
        self.timestamp = timestamp
        self.price = price
        self.volume = volume
        self.bid = bid or price - 0.01
        self.ask = ask or price + 0.01


class MockOHLCV:
    """Mock OHLCV data for testing"""

    def __init__(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis_client() -> MagicMock:
    """Mock Redis client for testing"""
    mock_redis = MagicMock(spec=redis.Redis)
    mock_redis.ping.return_value = True
    mock_redis.set.return_value = True
    mock_redis.get.return_value = None
    mock_redis.exists.return_value = False
    mock_redis.delete.return_value = 1
    mock_redis.publish.return_value = 1
    mock_redis.hset.return_value = 1
    mock_redis.hget.return_value = None
    mock_redis.hgetall.return_value = {}
    mock_redis.zadd.return_value = 1
    mock_redis.zrange.return_value = []
    return mock_redis


@pytest.fixture
async def mock_db_manager() -> AsyncMock:
    """Mock database manager for testing"""
    mock_db = AsyncMock(spec=MockDatabaseManager)
    mock_db.connect.return_value = None
    mock_db.disconnect.return_value = None
    mock_db.execute_query.return_value = []
    mock_db.fetch_one.return_value = None
    mock_db.fetch_all.return_value = []
    mock_db.insert.return_value = 1
    mock_db.update.return_value = 1
    mock_db.delete.return_value = 1
    return mock_db


@pytest.fixture
def sample_market_data() -> MockMarketData:
    """Create sample market data"""
    return MockMarketData(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=150.25,
        volume=1000000,
        bid=150.20,
        ask=150.30,
    )


@pytest.fixture
def sample_ohlcv_data() -> MockOHLCV:
    """Sample OHLCV data for testing"""
    return MockOHLCV(
        timestamp=datetime.now(timezone.utc),
        open_price=150.00,
        high=151.00,
        low=149.50,
        close=150.50,
        volume=1000000,
    )


@pytest.fixture
def sample_trade_signal() -> TradeSignal:
    """Sample trading signal for testing"""
    return TradeSignal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence=0.8,
        price=Decimal("150.50"),
        quantity=100,
        strategy_name="test_strategy",
        stop_loss=Decimal("145.0"),
        take_profit=Decimal("155.0"),
        timestamp=datetime.now(timezone.utc),
        metadata={"rsi": 30.5, "sma_crossover": True},
    )


@pytest.fixture
def sample_trade() -> OrderRequest:
    """Sample trade for testing"""
    return OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        price=None,
        stop_price=None,
        time_in_force="day",
        client_order_id=None,
    )


@pytest.fixture
def sample_order() -> Trade:
    """Sample executed trade for testing"""
    from uuid import uuid4

    return Trade(
        order_id=uuid4(),
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=Decimal("150.50"),
        commission=Decimal("1.00"),
        timestamp=datetime.now(timezone.utc),
        strategy_name="test_strategy",
        pnl=None,
        fees=Decimal("0.50"),
    )


@pytest.fixture
def sample_position() -> Position:
    """Sample position for testing"""
    return Position(
        symbol="AAPL",
        quantity=100,
        entry_price=Decimal("150.25"),
        market_value=Decimal("15050.00"),
        cost_basis=Decimal("15025.00"),
        current_price=Decimal("150.50"),
        unrealized_pnl=Decimal("25.00"),
    )


@pytest.fixture
def sample_portfolio() -> PortfolioState:
    """Sample portfolio for testing"""
    return PortfolioState(
        account_id="test_account",
        cash=Decimal("10000.00"),
        buying_power=Decimal("20000.00"),
        total_equity=Decimal("15000.00"),
        positions=[
            Position(
                symbol="AAPL",
                quantity=100,
                entry_price=Decimal("150.25"),
                market_value=Decimal("15050.00"),
                cost_basis=Decimal("15025.00"),
                current_price=Decimal("150.50"),
                unrealized_pnl=Decimal("25.00"),
            )
        ],
    )


@pytest.fixture
def sample_historical_data() -> pd.DataFrame:
    """Sample historical market data for backtesting"""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 100.0
    prices = []

    for i in range(len(dates)):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std dev
        base_price *= 1 + change

        # Generate OHLC from base price
        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = base_price + np.random.normal(0, base_price * 0.005)
        close_price = base_price
        volume = int(np.random.normal(1000000, 200000))

        prices.append(
            {
                "timestamp": dates[i],
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": max(volume, 100000),  # Ensure minimum volume
            }
        )

    return pd.DataFrame(prices)


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_external_api() -> Dict[str, Any]:
    """Mock external API responses"""
    return {
        "twelve_data": {
            "time_series": {
                "AAPL": {
                    "meta": {
                        "symbol": "AAPL",
                        "interval": "1min",
                        "currency": "USD",
                        "exchange_timezone": "America/New_York",
                    },
                    "values": [
                        {
                            "datetime": "2023-12-01 15:59:00",
                            "open": "150.00",
                            "high": "150.50",
                            "low": "149.75",
                            "close": "150.25",
                            "volume": "1500000",
                        }
                    ],
                }
            }
        },
        "alpaca": {
            "account": {
                "id": "test_account",
                "account_number": "123456789",
                "status": "ACTIVE",
                "currency": "USD",
                "buying_power": "50000.00",
                "cash": "25000.00",
                "portfolio_value": "75000.00",
            },
            "positions": [
                {
                    "symbol": "AAPL",
                    "qty": "100",
                    "side": "long",
                    "market_value": "15050.00",
                    "cost_basis": "15000.00",
                    "unrealized_pl": "50.00",
                    "unrealized_plpc": "0.0033",
                }
            ],
            "orders": {
                "id": "test_order_001",
                "client_order_id": "client_001",
                "created_at": "2023-12-01T15:59:00Z",
                "updated_at": "2023-12-01T15:59:00Z",
                "submitted_at": "2023-12-01T15:59:00Z",
                "filled_at": None,
                "expired_at": None,
                "canceled_at": None,
                "failed_at": None,
                "replaced_at": None,
                "replaced_by": None,
                "replaces": None,
                "asset_id": "asset_001",
                "symbol": "AAPL",
                "asset_class": "us_equity",
                "notional": None,
                "qty": "100",
                "filled_qty": "0",
                "filled_avg_price": None,
                "order_class": "",
                "order_type": "market",
                "type": "market",
                "side": "buy",
                "time_in_force": "day",
                "limit_price": None,
                "stop_price": None,
                "status": "accepted",
                "extended_hours": False,
                "legs": None,
                "trail_percent": None,
                "trail_price": None,
                "hwm": None,
            },
        },
    }


@pytest.fixture
def mock_gotify_client() -> MagicMock:
    """Mock Gotify client for notifications"""
    mock_client = MagicMock()
    mock_client.send_message.return_value = {
        "id": 1,
        "appid": 1,
        "message": "Test message",
        "title": "Test title",
        "priority": 5,
        "date": datetime.now(timezone.utc).isoformat(),
    }
    return mock_client


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration dictionary"""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_trading_system",
            "user": "test_trader",
            "password": "test_password",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": None,
            "db": 1,  # Use different DB for tests
        },
        "external_apis": {
            "twelve_data": {
                "api_key": "test_twelve_data_key",
                "base_url": "https://api.twelvedata.com",
            },
            "alpaca": {
                "api_key": "test_alpaca_key",
                "secret_key": "test_alpaca_secret",
                "base_url": "https://paper-api.alpaca.markets",
                "data_url": "https://data.alpaca.markets",
            },
        },
        "risk_management": {
            "max_position_size": 0.05,
            "max_portfolio_risk": 0.02,
            "max_drawdown": 0.15,
            "max_correlation": 0.7,
        },
        "strategies": {
            "momentum": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "sma_short": 20,
                "sma_long": 50,
            }
        },
        "gotify": {"url": "http://localhost:8080", "token": "test_gotify_token"},
    }


@pytest.fixture
def trading_signal_factory() -> Callable:
    """Factory for creating trading signals with different parameters"""

    def create_signal(
        symbol: str = "AAPL",
        signal_type: SignalType = SignalType.BUY,
        confidence: float = 0.8,
        price: float = 150.50,
        strategy_name: str = "test_strategy",
        timestamp: Optional[datetime] = None,
        **kwargs: Any,
    ) -> TradeSignal:
        return TradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=Decimal(str(price)),
            quantity=kwargs.get("quantity", 100),
            strategy_name=strategy_name,
            stop_loss=Decimal(str(price * 0.95)),
            take_profit=Decimal(str(price * 1.05)),
            timestamp=datetime.now(timezone.utc),
            metadata=kwargs,
        )

    return create_signal


@pytest.fixture
def order_factory() -> Callable:
    """Factory for creating orders with different parameters"""

    def create_order(
        symbol: str = "AAPL",
        side: OrderSide = OrderSide.BUY,
        order_type: OrderType = OrderType.MARKET,
        quantity: int = 100,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> OrderRequest:
        return OrderRequest(
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
            quantity=quantity,
            price=Decimal(str(price)) if price else None,
            stop_price=None,
            time_in_force="day",
            client_order_id=None,
        )

    return create_order


@pytest.fixture
def trade_factory() -> Callable:
    """Factory for creating executed trades"""

    def create_trade(
        symbol: str = "AAPL",
        side: OrderSide = OrderSide.BUY,
        quantity: int = 100,
        price: float = 150.0,
        **kwargs: Any,
    ) -> Trade:
        from uuid import uuid4

        return Trade(
            order_id=uuid4(),
            symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            quantity=quantity,
            price=Decimal(str(price)),
            commission=Decimal(str(kwargs.get("commission", 1.00))),
            timestamp=datetime.now(timezone.utc),
            strategy_name=kwargs.get("strategy_name", "test_strategy"),
            pnl=None,
            fees=Decimal(str(kwargs.get("fees", 0.50))),
        )

    return create_trade


@pytest.fixture
def market_data_factory() -> Callable:
    """Factory for creating market data"""

    def create_market_data(
        symbol: str = "AAPL",
        price: float = 150.0,
        volume: float = 1000.0,
        timestamp: Optional[datetime] = None,
        **kwargs: Any,
    ) -> MockMarketData:
        return MockMarketData(
            symbol=symbol,
            timestamp=timestamp or datetime.now(timezone.utc),
            price=price,
            volume=volume,
            bid=kwargs.get("bid", price * 0.999),
            ask=kwargs.get("ask", price * 1.001),
        )

    return create_market_data


@pytest.fixture
def ohlcv_factory() -> Callable:
    """Factory for creating OHLCV data"""

    def create_ohlcv(
        timestamp: Optional[datetime] = None,
        open_price: float = 150.0,
        high: float = 151.0,
        low: float = 149.0,
        close: float = 150.5,
        volume: float = 1000.0,
        **kwargs: Any,
    ) -> MockOHLCV:
        return MockOHLCV(
            timestamp=timestamp or datetime.now(timezone.utc),
            open_price=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )

    return create_ohlcv


@pytest.fixture
def mock_prometheus_metrics() -> MagicMock:
    """Mock Prometheus metrics for testing"""
    from unittest.mock import MagicMock

    metrics = MagicMock()

    # Mock counters
    metrics.api_requests_total = MagicMock()
    metrics.trades_total = MagicMock()
    metrics.errors_total = MagicMock()

    # Mock gauges
    metrics.portfolio_value = MagicMock()
    metrics.cash_balance = MagicMock()
    metrics.active_positions = MagicMock()

    # Mock histograms
    metrics.api_request_duration = MagicMock()
    metrics.trade_execution_duration = MagicMock()
    metrics.strategy_execution_duration = MagicMock()

    return metrics


@pytest.fixture
def mock_twelve_data_api() -> Callable[[str, str], Dict[str, Any]]:
    """Mock TwelveData API responses"""

    def mock_response(symbol: str = "AAPL", interval: str = "1min") -> Dict[str, Any]:
        return {
            "meta": {
                "symbol": symbol,
                "interval": interval,
                "currency": "USD",
                "exchange_timezone": "America/New_York",
                "exchange": "NASDAQ",
                "mic_code": "XNGS",
                "type": "Common Stock",
            },
            "values": [
                {
                    "datetime": "2023-12-01 15:59:00",
                    "open": "150.00",
                    "high": "150.50",
                    "low": "149.75",
                    "close": "150.25",
                    "volume": "1500000",
                },
                {
                    "datetime": "2023-12-01 15:58:00",
                    "open": "149.80",
                    "high": "150.05",
                    "low": "149.60",
                    "close": "150.00",
                    "volume": "1200000",
                },
            ],
            "status": "ok",
        }

    return mock_response


@pytest.fixture
def mock_alpaca_api() -> Dict[str, Any]:
    """Mock Alpaca API responses"""
    return {
        "account": {
            "id": "test_account_id",
            "account_number": "123456789",
            "status": "ACTIVE",
            "crypto_status": "ACTIVE",
            "currency": "USD",
            "buying_power": "50000.00",
            "regt_buying_power": "50000.00",
            "daytrading_buying_power": "200000.00",
            "non_marginable_buying_power": "25000.00",
            "cash": "25000.00",
            "accrued_fees": "0.00",
            "pending_transfer_out": "0.00",
            "pending_transfer_in": "0.00",
            "portfolio_value": "75000.00",
            "pattern_day_trader": False,
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "created_at": "2023-01-01T00:00:00Z",
            "trade_suspended_by_user": False,
            "multiplier": "4",
            "shorting_enabled": True,
            "equity": "75000.00",
            "last_equity": "74500.00",
            "long_market_value": "50000.00",
            "short_market_value": "0.00",
            "initial_margin": "25000.00",
            "maintenance_margin": "15000.00",
            "last_maintenance_margin": "15000.00",
            "sma": "75000.00",
            "daytrade_count": 0,
        }
    }


@pytest.fixture
async def clean_test_data() -> AsyncGenerator[None, None]:
    """Clean up test data before and after tests"""
    # Setup: Clean any existing test data
    yield
    # Teardown: Clean up test data created during tests
    pass


@pytest.fixture
def performance_test_data() -> Callable:
    """Generate performance test data"""

    def generate_data(num_symbols: int = 10, num_days: int = 30) -> Dict[str, Any]:
        symbols = [f"TEST{i:03d}" for i in range(num_symbols)]
        data = {}

        for symbol in symbols:
            start_date = datetime.now() - timedelta(days=num_days)
            end_date = datetime.now()
            dates = pd.date_range(start=start_date, end=end_date, freq="1min")

            # Generate random but realistic price data
            np.random.seed(hash(symbol) % 2**32)  # Deterministic per symbol
            base_price = np.random.uniform(50, 500)

            prices = []
            for date in dates:
                change = np.random.normal(0, 0.001)  # Small random changes
                base_price *= 1 + change

                prices.append(
                    {
                        "symbol": symbol,
                        "timestamp": date,
                        "price": round(base_price, 2),
                        "volume": int(np.random.normal(100000, 20000)),
                        "bid": round(base_price - 0.01, 2),
                        "ask": round(base_price + 0.01, 2),
                    }
                )

            data[symbol] = pd.DataFrame(prices)

        return data

    return generate_data


@pytest.fixture
def load_test_config() -> Dict[str, Any]:
    """Configuration for load testing"""
    return {
        "concurrent_users": 10,
        "requests_per_user": 100,
        "ramp_up_time": 30,  # seconds
        "test_duration": 300,  # seconds
        "target_endpoints": [
            "http://localhost:9101/health",
            "http://localhost:9102/health",
            "http://localhost:9103/health",
            "http://localhost:9104/health",
            "http://localhost:9105/health",
        ],
        "acceptable_response_time": 1.0,  # seconds
        "acceptable_error_rate": 0.01,  # 1%
    }


class TestHelper:
    """Helper class for common test operations"""

    @staticmethod
    def assert_trading_signal_valid(signal: TradeSignal) -> None:
        """Assert that a trading signal is valid"""
        assert signal.id is not None
        assert signal.symbol is not None
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
        assert signal.confidence <= 1
        assert signal.price is not None and signal.price > 0
        assert signal.timestamp is not None
        assert signal.strategy_name is not None

    @staticmethod
    def assert_order_valid(order: OrderRequest) -> None:
        """Assert that an order is valid"""
        assert order.id is not None
        assert order.symbol is not None
        assert order.side in [OrderSide.BUY, OrderSide.SELL]
        assert order.quantity > 0
        assert order.order_type in [
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.STOP_LIMIT,
        ]
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            assert order.price and order.price > 0

    @staticmethod
    def assert_trade_valid(trade: Trade) -> None:
        """Assert that a trade is valid"""
        assert trade.id is not None
        assert trade.symbol is not None
        assert trade.side in [OrderSide.BUY, OrderSide.SELL]
        assert trade.quantity > 0
        assert trade.price > 0
        assert trade.timestamp is not None

    @staticmethod
    def assert_market_data_valid(market_data: MockMarketData) -> None:
        """Assert that market data is valid"""
        assert market_data.symbol is not None
        assert market_data.price > 0
        assert market_data.volume >= 0
        assert market_data.bid > 0
        assert market_data.ask > 0
        assert market_data.ask > market_data.bid
        assert market_data.timestamp is not None

    @staticmethod
    def create_test_portfolio(
        symbols: List[str], cash: float = 10000.0
    ) -> PortfolioState:
        """Create a test portfolio with given symbols"""
        positions = []
        for symbol in symbols:
            position = Position(
                symbol=symbol,
                quantity=100,
                entry_price=Decimal("100.0"),
                market_value=Decimal("10500.0"),
                cost_basis=Decimal("10000.0"),
                current_price=Decimal("105.0"),
                unrealized_pnl=Decimal("500.0"),
            )
            positions.append(position)

        return PortfolioState(
            account_id="test_account",
            cash=Decimal(str(cash)),
            buying_power=Decimal(str(cash * 2)),
            total_equity=Decimal(str(cash + len(symbols) * 500)),
            positions=positions,
        )


@pytest.fixture
def test_helper() -> TestHelper:
    """Provide TestHelper instance"""
    return TestHelper()


# Performance testing markers
def pytest_configure(config: Any) -> None:
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "load: Load tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_environment() -> Generator[None, None, None]:
    """Automatically clean up test environment"""
    # Setup
    yield
    # Teardown - clean up any test artifacts
    test_log_dir = Path("/tmp/test_logs")
    if test_log_dir.exists():
        shutil.rmtree(test_log_dir)


@pytest.fixture
def mock_environment_variables() -> Generator[Dict[str, str], None, None]:
    """Mock environment variables for testing"""
    test_env = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "test_trading_system",
        "DB_USER": "test_trader",
        "DB_PASSWORD": "test_password",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_PASSWORD": "",
        "TWELVE_DATA_API_KEY": "test_twelve_data_key",
        "ALPACA_API_KEY": "test_alpaca_key",
        "ALPACA_SECRET_KEY": "test_alpaca_secret",
        "GOTIFY_URL": "http://localhost:8080",
        "GOTIFY_TOKEN": "test_gotify_token",
    }

    with patch.dict("os.environ", test_env, clear=True):
        yield test_env


# Database fixtures for integration tests
@pytest.fixture
async def test_db_connection() -> None:
    """Create test database connection"""
    # This would create a test database connection
    # Implementation depends on your database setup
    pass


@pytest.fixture
def mock_websocket() -> MagicMock:
    """Mock WebSocket connection for testing real-time data"""
    mock_ws = AsyncMock()
    mock_ws.send.return_value = None
    mock_ws.recv.return_value = json.dumps(
        {
            "symbol": "AAPL",
            "price": 150.50,
            "volume": 1000,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    mock_ws.close.return_value = None
    return mock_ws


@pytest.fixture
def benchmark_data() -> Dict[str, Any]:
    """Benchmark data for performance comparisons"""
    return {
        "sp500_returns": pd.Series([0.001, 0.002, -0.001, 0.003, -0.002] * 100),
        "nasdaq_returns": pd.Series([0.002, 0.003, -0.002, 0.004, -0.001] * 100),
        "risk_free_rate": 0.02,  # 2% annual
    }
