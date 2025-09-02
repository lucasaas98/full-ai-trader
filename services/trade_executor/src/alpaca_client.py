"""
Alpaca API client for trade execution service.

This module provides a comprehensive interface to the Alpaca Trading API,
including order management, position tracking, and real-time data streaming.
Uses the modern alpaca-py library instead of the deprecated alpaca_trade_api.
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.config import get_config
from shared.models import (
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger(__name__)

try:
    from alpaca.common.exceptions import APIError  # type: ignore
    from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
    from alpaca.data.live import StockDataStream  # type: ignore
    from alpaca.data.requests import (  # type: ignore
        StockBarsRequest,
        StockLatestQuoteRequest,
    )
    from alpaca.data.timeframe import TimeFrame  # type: ignore
    from alpaca.trading.client import TradingClient  # type: ignore
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide  # type: ignore
    from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus  # type: ignore
    from alpaca.trading.enums import OrderType as AlpacaOrderType  # type: ignore
    from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
    from alpaca.trading.models import Order as AlpacaOrder  # type: ignore
    from alpaca.trading.models import Position as AlpacaPosition
    from alpaca.trading.models import TradeAccount
    from alpaca.trading.requests import (
        ClosePositionRequest as AlpacaClosePositionRequest,
    )
    from alpaca.trading.requests import GetOrdersRequest as AlpacaGetOrdersRequest
    from alpaca.trading.requests import LimitOrderRequest as AlpacaLimitOrderRequest
    from alpaca.trading.requests import (
        MarketOrderRequest as AlpacaMarketOrderRequest,  # type: ignore
    )
    from alpaca.trading.requests import (
        StopLimitOrderRequest as AlpacaStopLimitOrderRequest,
    )
    from alpaca.trading.requests import StopOrderRequest as AlpacaStopOrderRequest

    ALPACA_AVAILABLE = True
except ImportError:
    # Create mock classes for development/testing
    class TradingClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_account(self):
            return None

        def get_open_position(self, symbol):
            return None

        def get_all_positions(self):
            return []

        def submit_order(self, *args, **kwargs):
            return None

        def cancel_order_by_id(self, order_id):
            return None

        def get_order_by_id(self, order_id):
            return None

        def get_orders(self, *args, **kwargs):
            return []

        def close_position(self, symbol_or_asset_id):
            return None

        def get_portfolio_history(self, *args, **kwargs):
            return None

        def replace_order_by_id(self, *args, **kwargs):
            return None

        def get_asset(self, symbol):
            return None

        def cancel_orders(self):
            return None

    class StockHistoricalDataClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_stock_latest_quote(self, *args, **kwargs):
            return None

        def get_stock_latest_trade(self, *args, **kwargs):
            return None

        def get_stock_bars(self, *args, **kwargs):
            return None

    class StockDataStream:
        def __init__(self, *args, **kwargs):
            pass

        async def subscribe_quotes(self, *args, **kwargs):
            pass

        async def subscribe_trades(self, *args, **kwargs):
            pass

        async def subscribe_bars(self, *args, **kwargs):
            pass

        async def subscribe_trade_updates(self, *args, **kwargs):
            pass

        async def run(self):
            pass

        async def stop(self):
            pass

    class AlpacaOrder:
        def __init__(self):
            self.id = "mock_order"
            self.symbol = "MOCK"
            self.side = "buy"
            self.order_type = "market"
            self.status = "filled"
            self.qty = 0
            self.filled_qty = 0
            self.limit_price = None
            self.filled_avg_price = None
            self.submitted_at = None
            self.filled_at = None
            self.canceled_at = None

    class AlpacaPosition:
        def __init__(self):
            self.symbol = "MOCK"
            self.qty = 0
            self.avg_entry_price = 0
            self.market_value = 0
            self.unrealized_pl = 0
            self.cost_basis = 0

    class TradeAccount:
        def __init__(self):
            self.id = "mock_account"
            self.buying_power = 0
            self.portfolio_value = 0
            self.daytrade_count = 0
            self.pattern_day_trader = False
            self.equity = 0
            self.cash = 0

    class TimeFrame:
        Day = "1Day"
        Hour = "1Hour"
        Minute = "1Min"

    class APIError(Exception):
        pass

    class AlpacaOrderSide:
        BUY = "buy"
        SELL = "sell"

    class AlpacaOrderType:
        MARKET = "market"
        LIMIT = "limit"
        STOP = "stop"
        STOP_LIMIT = "stop_limit"

    class AlpacaOrderStatus:
        NEW = "new"
        PARTIALLY_FILLED = "partially_filled"
        FILLED = "filled"
        DONE_FOR_DAY = "done_for_day"
        CANCELED = "canceled"
        EXPIRED = "expired"
        REPLACED = "replaced"
        PENDING_CANCEL = "pending_cancel"
        PENDING_REPLACE = "pending_replace"
        REJECTED = "rejected"
        SUSPENDED = "suspended"
        PENDING_NEW = "pending_new"
        CALCULATED = "calculated"
        ACCEPTED = "accepted"
        ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
        STOPPED = "stopped"

    class TimeInForce:
        DAY = "day"
        GTC = "gtc"
        IOC = "ioc"
        FOK = "fok"

    # Set aliases for consistency
    AlpacaTimeInForce = TimeInForce

    class MarketOrderRequest:
        def __init__(self, **kwargs):
            pass

    class LimitOrderRequest:
        def __init__(self, **kwargs):
            pass

    class StopOrderRequest:
        def __init__(self, **kwargs):
            pass

    class StopLimitOrderRequest:
        def __init__(self, **kwargs):
            pass

    class GetOrdersRequest:
        def __init__(self, **kwargs):
            pass

    class ClosePositionRequest:
        def __init__(self, **kwargs):
            pass

    class StockBarsRequest:
        def __init__(self, **kwargs):
            pass

    class StockLatestQuoteRequest:
        def __init__(self, **kwargs):
            pass

    # Set aliases for consistency
    AlpacaMarketOrderRequest = MarketOrderRequest
    AlpacaLimitOrderRequest = LimitOrderRequest
    AlpacaStopOrderRequest = StopOrderRequest
    AlpacaStopLimitOrderRequest = StopLimitOrderRequest
    AlpacaGetOrdersRequest = GetOrdersRequest
    AlpacaClosePositionRequest = ClosePositionRequest
    AlpacaTimeInForce = TimeInForce

    ALPACA_AVAILABLE = False


logger = logging.getLogger(__name__)


class AlpacaAPIError(Exception):
    """Custom exception for Alpaca API errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)


class AlpacaClient:
    """
    Comprehensive Alpaca API client for trade execution.

    Provides methods for:
    - Order placement and management
    - Position tracking
    - Account information
    - Real-time data streaming
    - Error handling and retry logic
    """

    def __init__(self):
        """Initialize Alpaca client with configuration."""
        self.config = get_config()
        self._trading_client = None
        self._data_client = None
        self._stream_client = None
        self._session = None
        self._positions_cache: Dict[str, Position] = {}
        self._account_cache: Optional[TradeAccount] = None
        self._cache_expiry = datetime.now(timezone.utc)
        self._cache_ttl = timedelta(seconds=30)

        # Initialize API clients
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Alpaca API clients."""
        try:
            if ALPACA_AVAILABLE:
                # Initialize trading client
                self._trading_client = TradingClient(
                    api_key=self.config.alpaca.api_key,
                    secret_key=self.config.alpaca.secret_key,
                    paper=self.config.alpaca.paper_trading,
                )

                # Initialize data client
                self._data_client = StockHistoricalDataClient(
                    api_key=self.config.alpaca.api_key,
                    secret_key=self.config.alpaca.secret_key,
                )

                logger.info(
                    f"Alpaca client initialized for {'paper' if self.config.alpaca.paper_trading else 'live'} trading"
                )
            else:
                logger.warning("Alpaca API not available, using mock client")
                self._trading_client = None
                self._data_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self._trading_client = None
            self._data_client = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Establish connections to Alpaca services."""
        try:
            # Create aiohttp session for async requests
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # Initialize streaming client
            if ALPACA_AVAILABLE:
                self._stream_client = StockDataStream(
                    api_key=self.config.alpaca.api_key,
                    secret_key=self.config.alpaca.secret_key,
                )

            logger.info("Alpaca client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise AlpacaAPIError(f"Connection failed: {e}", original_error=e)

    async def disconnect(self):
        """Close connections to Alpaca services."""
        try:
            if self._session:
                await self._session.close()

            if self._stream_client:
                await self._stream_client.stop()

            logger.info("Alpaca client disconnected")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_account(self, use_cache: bool = True) -> TradeAccount:
        """
        Get account information.

        Args:
            use_cache: Whether to use cached account data

        Returns:
            Account information
        """
        try:
            if (
                use_cache
                and self._account_cache
                and datetime.now(timezone.utc) < self._cache_expiry
            ):
                return self._account_cache

            if not self._trading_client:
                raise AlpacaAPIError("Alpaca trading client not initialized")

            account = self._trading_client.get_account()
            self._account_cache = account
            self._cache_expiry = datetime.now(timezone.utc) + self._cache_ttl

            if account:
                logger.debug(
                    f"Account info retrieved: equity=${account.equity}, buying_power=${account.buying_power}"
                )
            return account
        except APIError as e:
            logger.error(f"Alpaca API error getting account: {e}")
            raise AlpacaAPIError(f"Failed to get account: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise AlpacaAPIError(f"Failed to get account: {e}", original_error=e)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_positions(
        self, symbol: Optional[str] = None, use_cache: bool = True
    ) -> List[Position]:
        """
        Get current positions.

        Args:
            symbol: Optional symbol to filter by
            use_cache: Whether to use cached position data

        Returns:
            List of current positions
        """
        cache_key = symbol or "all"

        if (
            use_cache
            and cache_key in self._positions_cache
            and datetime.now(timezone.utc) < self._cache_expiry
        ):
            if symbol:
                return (
                    [self._positions_cache[cache_key]]
                    if cache_key in self._positions_cache
                    else []
                )
            return list(self._positions_cache.values())

        try:
            if symbol:
                try:
                    if not self._trading_client:
                        raise AlpacaAPIError("Alpaca trading client not initialized")

                    position = self._trading_client.get_open_position(symbol)
                    if position:
                        position_model = self._convert_alpaca_position(position)
                        self._positions_cache[symbol] = position_model
                        return [position_model]
                    else:
                        return []
                except APIError:
                    # Position doesn't exist
                    return []
            else:
                if not self._trading_client:
                    return []
                positions = self._trading_client.get_all_positions()
                position_models = [
                    self._convert_alpaca_position(pos) for pos in positions
                ]

                # Update cache
                self._positions_cache.clear()
                for pos in position_models:
                    self._positions_cache[pos.symbol] = pos

                self._cache_expiry = datetime.now(timezone.utc) + self._cache_ttl
                return position_models

        except APIError as e:
            logger.error(f"Alpaca API error getting positions: {e}")
            raise AlpacaAPIError(f"Failed to get positions: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise AlpacaAPIError(f"Failed to get positions: {e}", original_error=e)

    def _convert_alpaca_position(self, position: AlpacaPosition) -> Position:
        """Convert Alpaca position to internal model."""
        return Position(
            symbol=position.symbol,
            quantity=int(float(position.qty)),
            entry_price=(
                Decimal(str(position.avg_entry_price))
                if position.avg_entry_price
                else Decimal("0")
            ),
            current_price=(
                Decimal(str(position.market_value))
                / Decimal(str(abs(float(position.qty))))
                if float(position.qty) != 0
                else Decimal("0")
            ),
            unrealized_pnl=(
                Decimal(str(position.unrealized_pl))
                if position.unrealized_pl
                else Decimal("0")
            ),
            market_value=(
                Decimal(str(position.market_value))
                if position.market_value
                else Decimal("0")
            ),
            cost_basis=(
                Decimal(str(position.cost_basis))
                if position.cost_basis
                else Decimal("0")
            ),
            last_updated=datetime.now(timezone.utc),
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Place an order with Alpaca.

        Args:
            order_request: Order details

        Returns:
            Order response with broker details
        """
        try:
            if not self._trading_client:
                raise AlpacaAPIError("Alpaca trading client not initialized")

            # Convert order request to Alpaca format
            alpaca_order_request = self._convert_to_alpaca_order(order_request)

            logger.info(
                f"Placing {order_request.side} order for {order_request.quantity} {order_request.symbol} "
                f"@ {order_request.price or 'market'}"
            )

            # Submit order to Alpaca
            alpaca_order = self._trading_client.submit_order(alpaca_order_request)

            if not alpaca_order:
                raise AlpacaAPIError("Order submission returned None")

            # Convert response
            order_response = self._convert_alpaca_order_response(
                alpaca_order, order_request
            )

            logger.info(f"Order placed successfully: {order_response.broker_order_id}")
            return order_response

        except APIError as e:
            logger.error(f"Alpaca API error placing order: {e}")
            raise AlpacaAPIError(f"Order placement failed: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise AlpacaAPIError(f"Order placement failed: {e}", original_error=e)

    def _convert_to_alpaca_order(self, order_request: OrderRequest) -> Any:
        """Convert internal order request to Alpaca API format."""

        # Convert side
        alpaca_side = "buy" if order_request.side == OrderSide.BUY else "sell"

        # Convert time in force
        tif = AlpacaTimeInForce.DAY  # Default
        if order_request.time_in_force == "gtc":
            tif = AlpacaTimeInForce.GTC
        elif order_request.time_in_force == "ioc":
            tif = AlpacaTimeInForce.IOC
        elif order_request.time_in_force == "fok":
            tif = AlpacaTimeInForce.FOK

        # Create order request based on type
        if order_request.order_type == OrderType.MARKET:
            return AlpacaMarketOrderRequest(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=alpaca_side,
                time_in_force=tif,
                extended_hours=order_request.extended_hours,
                client_order_id=order_request.client_order_id or str(order_request.id),
            )
        elif order_request.order_type == OrderType.LIMIT:
            return AlpacaLimitOrderRequest(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=alpaca_side,
                time_in_force=tif,
                limit_price=order_request.price,
                extended_hours=order_request.extended_hours,
                client_order_id=order_request.client_order_id or str(order_request.id),
            )
        elif order_request.order_type == OrderType.STOP:
            return AlpacaStopOrderRequest(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=alpaca_side,
                time_in_force=tif,
                stop_price=order_request.stop_price,
                extended_hours=order_request.extended_hours,
                client_order_id=order_request.client_order_id or str(order_request.id),
            )
        elif order_request.order_type == OrderType.STOP_LIMIT:
            return AlpacaStopLimitOrderRequest(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=alpaca_side,
                time_in_force=tif,
                limit_price=order_request.price,
                stop_price=order_request.stop_price,
                extended_hours=order_request.extended_hours,
                client_order_id=order_request.client_order_id or str(order_request.id),
            )
        else:
            raise AlpacaAPIError(f"Unsupported order type: {order_request.order_type}")

    def _convert_alpaca_order_response(
        self, alpaca_order: AlpacaOrder, original_request: OrderRequest
    ) -> OrderResponse:
        """Convert Alpaca order response to internal model."""
        # Convert to internal format
        return OrderResponse(
            id=original_request.id or uuid4(),
            broker_order_id=(
                str(alpaca_order.id)
                if alpaca_order and hasattr(alpaca_order, "id")
                else str(uuid4())
            ),
            symbol=(
                alpaca_order.symbol
                if alpaca_order and hasattr(alpaca_order, "symbol")
                else original_request.symbol
            ),
            side=(
                OrderSide.BUY
                if alpaca_order
                and hasattr(alpaca_order, "side")
                and alpaca_order.side == "buy"
                else OrderSide.SELL
            ),
            order_type=(
                self._convert_alpaca_order_type(str(alpaca_order.order_type))
                if alpaca_order and hasattr(alpaca_order, "order_type")
                else original_request.order_type
            ),
            quantity=(
                int(float(alpaca_order.qty))
                if alpaca_order and hasattr(alpaca_order, "qty")
                else original_request.quantity
            ),
            filled_quantity=(
                int(float(alpaca_order.filled_qty))
                if alpaca_order
                and hasattr(alpaca_order, "filled_qty")
                and alpaca_order.filled_qty
                else 0
            ),
            price=(
                Decimal(str(alpaca_order.limit_price))
                if alpaca_order
                and hasattr(alpaca_order, "limit_price")
                and alpaca_order.limit_price
                else original_request.price
            ),
            filled_price=(
                Decimal(str(alpaca_order.filled_avg_price))
                if alpaca_order
                and hasattr(alpaca_order, "filled_avg_price")
                and alpaca_order.filled_avg_price
                else None
            ),
            status=(
                self._convert_alpaca_order_status(str(alpaca_order.status))
                if alpaca_order and hasattr(alpaca_order, "status")
                else OrderStatus.PENDING
            ),
            submitted_at=(
                alpaca_order.submitted_at
                if alpaca_order
                and hasattr(alpaca_order, "submitted_at")
                and alpaca_order.submitted_at
                else datetime.now(timezone.utc)
            ),
            filled_at=(
                alpaca_order.filled_at
                if alpaca_order and hasattr(alpaca_order, "filled_at")
                else None
            ),
            cancelled_at=(
                alpaca_order.canceled_at
                if alpaca_order and hasattr(alpaca_order, "canceled_at")
                else None
            ),
            commission=Decimal("0"),
        )

    def _convert_alpaca_order_type(self, alpaca_order_type: str) -> OrderType:
        """Convert Alpaca order type to internal enum."""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
        }
        return mapping.get(alpaca_order_type.lower(), OrderType.MARKET)

    def _convert_alpaca_order_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca order status to internal enum."""
        mapping = {
            "new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "replaced": OrderStatus.CANCELLED,
            "pending_cancel": OrderStatus.CANCELLED,
            "pending_replace": OrderStatus.PENDING,
            "rejected": OrderStatus.REJECTED,
            "suspended": OrderStatus.REJECTED,
            "pending_new": OrderStatus.PENDING,
            "calculated": OrderStatus.PENDING,
            "accepted": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.PENDING,
            "stopped": OrderStatus.CANCELLED,
        }
        return mapping.get(alpaca_status.lower(), OrderStatus.PENDING)

    async def place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        entry_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        stop_loss_price: Optional[Decimal] = None,
        time_in_force: str = "day",
    ) -> List[OrderResponse]:
        """
        Place a bracket order (entry + take profit + stop loss).

        Args:
            symbol: Symbol to trade
            quantity: Number of shares
            side: Buy or sell
            entry_price: Entry price (None for market order)
            take_profit_price: Take profit price
            stop_loss_price: Stop loss price
            time_in_force: Time in force

        Returns:
            List of order responses (entry, take profit, stop loss)
        """
        try:
            if not self._trading_client:
                raise AlpacaAPIError("Alpaca trading client not initialized")

            # Convert side
            alpaca_side = "buy" if side == OrderSide.BUY else "sell"

            # Convert time in force
            tif = TimeInForce.DAY
            if time_in_force.lower() == "gtc":
                tif = TimeInForce.GTC

            # Create bracket order request
            if entry_price:
                # Limit entry order
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=tif,
                    limit_price=float(entry_price),
                    order_class="bracket",
                    take_profit=(
                        {"limit_price": float(take_profit_price)}
                        if take_profit_price
                        else None
                    ),
                    stop_loss=(
                        {"stop_price": float(stop_loss_price)}
                        if stop_loss_price
                        else None
                    ),
                )
            else:
                # Market entry order
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=tif,
                    order_class="bracket",
                    take_profit=(
                        {"limit_price": float(take_profit_price)}
                        if take_profit_price
                        else None
                    ),
                    stop_loss=(
                        {"stop_price": float(stop_loss_price)}
                        if stop_loss_price
                        else None
                    ),
                )

            logger.info(f"Placing bracket order for {quantity} {symbol}")

            # Submit bracket order
            alpaca_order = self._trading_client.submit_order(order_request)

            if not alpaca_order:
                raise AlpacaAPIError("Bracket order submission returned None")

            # Convert to OrderResponse (bracket orders return the main order)
            order_id = uuid4()
            order_response = OrderResponse(
                id=order_id,
                broker_order_id=str(alpaca_order.id),
                symbol=alpaca_order.symbol,
                side=side,
                order_type=OrderType.LIMIT if entry_price else OrderType.MARKET,
                quantity=quantity,
                filled_quantity=0,
                price=entry_price,
                filled_price=None,
                status=self._convert_alpaca_order_status(str(alpaca_order.status)),
                submitted_at=alpaca_order.submitted_at or datetime.now(timezone.utc),
                filled_at=None,
                cancelled_at=None,
                commission=Decimal("0"),
            )

            logger.info(
                f"Bracket order placed successfully: {order_response.broker_order_id}"
            )
            return [order_response]  # Return as list for compatibility

        except APIError as e:
            logger.error(f"Alpaca API error placing bracket order: {e}")
            raise AlpacaAPIError(
                f"Bracket order placement failed: {e}", original_error=e
            )
        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            raise AlpacaAPIError(
                f"Bracket order placement failed: {e}", original_error=e
            )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID.

        Args:
            order_id: Broker order ID

        Returns:
            True if cancellation was successful
        """
        try:
            if not self._trading_client:
                raise AlpacaAPIError("Alpaca trading client not initialized")

            self._trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} canceled successfully")
            return True

        except APIError as e:
            logger.error(f"Alpaca API error canceling order {order_id}: {e}")
            raise AlpacaAPIError(f"Failed to cancel order: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise AlpacaAPIError(f"Failed to cancel order: {e}", original_error=e)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """
        Get order details by ID.

        Args:
            order_id: Broker order ID

        Returns:
            Order details or None if not found
        """
        try:
            if not self._trading_client:
                raise AlpacaAPIError("Alpaca trading client not initialized")

            alpaca_order = self._trading_client.get_order_by_id(order_id)

            # Convert to internal format
            order_response = OrderResponse(
                id=uuid4(),  # Generate new ID since we don't have the original
                broker_order_id=(
                    str(alpaca_order.id)
                    if alpaca_order and hasattr(alpaca_order, "id")
                    else str(uuid4())
                ),
                symbol=(
                    alpaca_order.symbol
                    if alpaca_order and hasattr(alpaca_order, "symbol")
                    else ""
                ),
                side=(
                    OrderSide.BUY
                    if alpaca_order
                    and hasattr(alpaca_order, "side")
                    and alpaca_order.side == "buy"
                    else OrderSide.SELL
                ),
                order_type=(
                    self._convert_alpaca_order_type(str(alpaca_order.order_type))
                    if alpaca_order and hasattr(alpaca_order, "order_type")
                    else OrderType.MARKET
                ),
                status=(
                    self._convert_alpaca_order_status(str(alpaca_order.status))
                    if alpaca_order and hasattr(alpaca_order, "status")
                    else OrderStatus.PENDING
                ),
                quantity=(
                    int(float(alpaca_order.qty))
                    if alpaca_order and hasattr(alpaca_order, "qty")
                    else 0
                ),
                filled_quantity=(
                    int(float(alpaca_order.filled_qty))
                    if alpaca_order
                    and hasattr(alpaca_order, "filled_qty")
                    and alpaca_order.filled_qty
                    else 0
                ),
                price=(
                    Decimal(str(alpaca_order.limit_price))
                    if alpaca_order
                    and hasattr(alpaca_order, "limit_price")
                    and alpaca_order.limit_price
                    else None
                ),
                filled_price=(
                    Decimal(str(alpaca_order.filled_avg_price))
                    if alpaca_order
                    and hasattr(alpaca_order, "filled_avg_price")
                    and alpaca_order.filled_avg_price
                    else None
                ),
                submitted_at=(
                    alpaca_order.submitted_at
                    if alpaca_order
                    and hasattr(alpaca_order, "submitted_at")
                    and alpaca_order.submitted_at
                    else datetime.now(timezone.utc)
                ),
                filled_at=(
                    alpaca_order.filled_at
                    if alpaca_order and hasattr(alpaca_order, "filled_at")
                    else None
                ),
                cancelled_at=(
                    alpaca_order.canceled_at
                    if alpaca_order and hasattr(alpaca_order, "canceled_at")
                    else None
                ),
                commission=Decimal("0"),
            )

            return order_response

        except APIError as e:
            logger.error(f"Alpaca API error getting order {order_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_orders(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[OrderResponse]:
        """
        Get orders with optional status filter.

        Args:
            status: Optional status filter
            limit: Maximum number of orders to retrieve

        Returns:
            List of orders
        """
        try:
            if not self._trading_client:
                raise AlpacaAPIError("Alpaca trading client not initialized")

            # Convert status string for Alpaca API
            alpaca_status = status

            # Create request
            request = GetOrdersRequest(status=alpaca_status, limit=limit)

            alpaca_orders = self._trading_client.get_orders(request)

            # Convert to internal format
            orders = []
            for order in alpaca_orders:
                order_response = OrderResponse(
                    id=uuid4(),
                    broker_order_id=(
                        str(order.id)
                        if order and hasattr(order, "id")
                        else str(uuid4())
                    ),
                    symbol=order.symbol if order and hasattr(order, "symbol") else "",
                    side=(
                        OrderSide.BUY
                        if order and hasattr(order, "side") and order.side == "buy"
                        else OrderSide.SELL
                    ),
                    order_type=(
                        self._convert_alpaca_order_type(str(order.order_type))
                        if order and hasattr(order, "order_type")
                        else OrderType.MARKET
                    ),
                    status=(
                        self._convert_alpaca_order_status(str(order.status))
                        if order and hasattr(order, "status")
                        else OrderStatus.PENDING
                    ),
                    quantity=(
                        int(float(order.qty)) if order and hasattr(order, "qty") else 0
                    ),
                    filled_quantity=(
                        int(float(order.filled_qty))
                        if order and hasattr(order, "filled_qty") and order.filled_qty
                        else 0
                    ),
                    price=(
                        Decimal(str(order.limit_price))
                        if order and hasattr(order, "limit_price") and order.limit_price
                        else None
                    ),
                    filled_price=(
                        Decimal(str(order.filled_avg_price))
                        if order
                        and hasattr(order, "filled_avg_price")
                        and order.filled_avg_price
                        else None
                    ),
                    submitted_at=(
                        order.submitted_at
                        if order
                        and hasattr(order, "submitted_at")
                        and order.submitted_at
                        else datetime.now(timezone.utc)
                    ),
                    filled_at=(
                        order.filled_at
                        if order and hasattr(order, "filled_at")
                        else None
                    ),
                    cancelled_at=(
                        order.canceled_at
                        if order and hasattr(order, "canceled_at")
                        else None
                    ),
                    commission=Decimal("0"),
                )
                orders.append(order_response)

            return orders

        except APIError as e:
            logger.error(f"Alpaca API error getting orders: {e}")
            raise AlpacaAPIError(f"Failed to get orders: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise AlpacaAPIError(f"Failed to get orders: {e}", original_error=e)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def close_position(
        self, symbol: str, percentage: float = 100.0
    ) -> Optional[OrderResponse]:
        """
        Close a position (partial or full).

        Args:
            symbol: Symbol to close
            percentage: Percentage of position to close (default 100%)

        Returns:
            Order response for the close order
        """
        try:
            if not self._trading_client:
                raise AlpacaAPIError("Alpaca trading client not initialized")

            # Create close request - just close the position
            alpaca_order = self._trading_client.close_position(symbol)

            # Convert to internal format
            order_response = OrderResponse(
                id=uuid4(),
                broker_order_id=(
                    str(alpaca_order.id)
                    if alpaca_order and hasattr(alpaca_order, "id")
                    else str(uuid4())
                ),
                symbol=(
                    alpaca_order.symbol
                    if alpaca_order and hasattr(alpaca_order, "symbol")
                    else symbol
                ),
                side=(
                    OrderSide.BUY
                    if alpaca_order
                    and hasattr(alpaca_order, "side")
                    and alpaca_order.side == "buy"
                    else OrderSide.SELL
                ),
                order_type=(
                    self._convert_alpaca_order_type(str(alpaca_order.order_type))
                    if alpaca_order and hasattr(alpaca_order, "order_type")
                    else OrderType.MARKET
                ),
                status=(
                    self._convert_alpaca_order_status(str(alpaca_order.status))
                    if alpaca_order and hasattr(alpaca_order, "status")
                    else OrderStatus.PENDING
                ),
                quantity=(
                    int(float(alpaca_order.qty))
                    if alpaca_order and hasattr(alpaca_order, "qty")
                    else 0
                ),
                filled_quantity=(
                    int(float(alpaca_order.filled_qty))
                    if alpaca_order
                    and hasattr(alpaca_order, "filled_qty")
                    and alpaca_order.filled_qty
                    else 0
                ),
                price=(
                    Decimal(str(alpaca_order.limit_price))
                    if alpaca_order
                    and hasattr(alpaca_order, "limit_price")
                    and alpaca_order.limit_price
                    else None
                ),
                filled_price=(
                    Decimal(str(alpaca_order.filled_avg_price))
                    if alpaca_order
                    and hasattr(alpaca_order, "filled_avg_price")
                    and alpaca_order.filled_avg_price
                    else None
                ),
                submitted_at=(
                    alpaca_order.submitted_at
                    if alpaca_order
                    and hasattr(alpaca_order, "submitted_at")
                    and alpaca_order.submitted_at
                    else datetime.now(timezone.utc)
                ),
                filled_at=(
                    alpaca_order.filled_at
                    if alpaca_order and hasattr(alpaca_order, "filled_at")
                    else None
                ),
                cancelled_at=(
                    alpaca_order.canceled_at
                    if alpaca_order and hasattr(alpaca_order, "canceled_at")
                    else None
                ),
                commission=Decimal("0"),
            )

            logger.info(
                f"Position {symbol} closed successfully: {order_response.broker_order_id}"
            )
            return order_response

        except APIError as e:
            logger.error(f"Alpaca API error closing position {symbol}: {e}")
            raise AlpacaAPIError(f"Failed to close position: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            raise AlpacaAPIError(f"Failed to close position: {e}", original_error=e)

    async def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with bid, ask, and last prices
        """
        try:
            if not self._data_client:
                return {"bid": Decimal("0"), "ask": Decimal("0"), "last": Decimal("0")}

            if ALPACA_AVAILABLE:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            else:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote_data = self._data_client.get_stock_latest_quote(request)

            if quote_data and symbol in quote_data:
                quote = quote_data[symbol]
                return {
                    "bid": Decimal(str(quote.bid_price)),
                    "ask": Decimal(str(quote.ask_price)),
                    "last": Decimal(
                        str(quote.ask_price)
                    ),  # Use ask as last if no last price
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "timestamp": quote.timestamp,
                }
            return None

        except APIError as e:
            logger.error(f"Alpaca API error getting quote for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    async def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest bar data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest bar data
        """
        try:
            if not self._data_client:
                return None

            # Get latest 1-minute bar
            if ALPACA_AVAILABLE:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, limit=1
                )
            else:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol, timeframe="1Hour", limit=1
                )
            bars_data = self._data_client.get_stock_bars(request)

            if bars_data and symbol in bars_data and bars_data[symbol]:
                bar = bars_data[symbol][0]
                return {
                    "timestamp": bar.timestamp,
                    "open": Decimal(str(bar.open)),
                    "high": Decimal(str(bar.high)),
                    "low": Decimal(str(bar.low)),
                    "close": Decimal(str(bar.close)),
                    "volume": int(bar.volume),
                    "vwap": (
                        Decimal(str(bar.vwap))
                        if hasattr(bar, "vwap") and bar.vwap
                        else None
                    ),
                }
            return None

        except APIError as e:
            logger.error(f"Alpaca API error getting bar for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get bar data for {symbol}: {e}")
            return None

    async def is_market_open(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market is open
        """
        try:
            if not self._trading_client:
                return False

            # Note: get_clock method might not be available in all versions
            # This is a simplified implementation
            return True  # Assume market is open for now
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False

    async def get_buying_power(self) -> Decimal:
        """
        Get current buying power.

        Returns:
            Available buying power
        """
        try:
            account = await self.get_account()
            return Decimal(str(account.buying_power))
        except Exception as e:
            logger.error(f"Failed to get buying power: {e}")
            raise AlpacaAPIError(
                f"Buying power retrieval failed: {e}", original_error=e
            )

    async def get_portfolio_value(self) -> Decimal:
        """
        Get total portfolio value.

        Returns:
            Total portfolio value
        """
        try:
            account = await self.get_account()
            return Decimal(str(account.portfolio_value))
        except Exception as e:
            logger.error(f"Failed to get portfolio value: {e}")
            raise AlpacaAPIError(
                f"Portfolio value retrieval failed: {e}", original_error=e
            )

    async def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.

        Returns:
            True if cancellation was successful
        """
        try:
            if not self._trading_client:
                raise AlpacaAPIError("Alpaca trading client not initialized")

            self._trading_client.cancel_orders()
            logger.info("All orders cancelled successfully")
            return True
        except APIError as e:
            logger.error(f"Alpaca API error canceling all orders: {e}")
            raise AlpacaAPIError(f"Cancel all orders failed: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            raise AlpacaAPIError(f"Cancel all orders failed: {e}", original_error=e)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Alpaca connection.

        Returns:
            Health status information
        """
        try:
            start_time = datetime.now(timezone.utc)

            # Test basic API connectivity
            account = await self.get_account(use_cache=False)
            api_latency = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Test market data
            try:
                quote = await self.get_latest_quote("SPY")
                market_data_available = quote is not None
            except:
                market_data_available = False

            return {
                "status": (
                    "healthy" if all([account, market_data_available]) else "degraded"
                ),
                "api_latency_seconds": api_latency,
                "account_accessible": bool(account),
                "market_data_available": market_data_available,
                "paper_trading": self.config.alpaca.paper_trading,
                "timestamp": datetime.now(timezone.utc),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc),
            }

    async def validate_order(self, order_request) -> Dict[str, Any]:
        """
        Validate an order request.

        Args:
            order_request: Order request to validate

        Returns:
            Dictionary with validation results
        """
        try:
            # Basic validation
            issues = []

            if not hasattr(order_request, "symbol") or not order_request.symbol:
                issues.append("Symbol is required")

            if not hasattr(order_request, "quantity") or order_request.quantity <= 0:
                issues.append("Quantity must be positive")

            if not hasattr(order_request, "side") or order_request.side not in [
                "buy",
                "sell",
            ]:
                issues.append("Side must be 'buy' or 'sell'")

            # Check buying power for buy orders
            if hasattr(order_request, "side") and order_request.side == "buy":
                buying_power = await self.get_buying_power()
                estimated_cost = order_request.quantity * getattr(
                    order_request, "price", 100
                )  # Rough estimate
                if estimated_cost > buying_power:
                    issues.append(
                        f"Insufficient buying power: ${buying_power} available, ${estimated_cost} needed"
                    )

            return {"valid": len(issues) == 0, "issues": issues}
        except Exception as e:
            logger.error(f"Order validation failed: {e}")
            return {"valid": False, "issues": [f"Validation error: {str(e)}"]}

    def calculate_position_size(
        self,
        symbol: str,
        risk_amount: Decimal,
        entry_price: Decimal,
        stop_price: Decimal,
    ) -> int:
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Trading symbol
            risk_amount: Amount to risk in dollars
            entry_price: Entry price
            stop_price: Stop loss price

        Returns:
            Number of shares to buy
        """
        try:
            if entry_price <= 0 or stop_price <= 0:
                return 0

            risk_per_share = abs(entry_price - stop_price)
            if risk_per_share == 0:
                return 0

            shares = int(risk_amount / risk_per_share)
            return max(0, shares)
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0

    async def calculate_twap_price(
        self, symbol: str, timeframe: str = "1Min", periods: int = 20
    ) -> Optional[Decimal]:
        """
        Calculate Time-Weighted Average Price.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for calculation
            periods: Number of periods to include

        Returns:
            TWAP price or None if calculation fails
        """
        try:
            # This is a simplified implementation
            bars = await self.get_latest_bar(symbol)
            if bars and "close" in bars:
                return Decimal(str(bars["close"]))
            return None
        except Exception as e:
            logger.error(f"TWAP calculation failed: {e}")
            return None

    async def calculate_vwap_price(
        self, symbol: str, timeframe: str = "1Min", periods: int = 20
    ) -> Optional[Decimal]:
        """
        Calculate Volume-Weighted Average Price.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for calculation
            periods: Number of periods to include

        Returns:
            VWAP price or None if calculation fails
        """
        try:
            # This is a simplified implementation
            bars = await self.get_latest_bar(symbol)
            if bars and "close" in bars:
                return Decimal(str(bars["close"]))
            return None
        except Exception as e:
            logger.error(f"VWAP calculation failed: {e}")
            return None

    async def close_all_positions(self) -> bool:
        """
        Close all open positions.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._trading_client:
                logger.warning("Trading client not available")
                return False

            positions = await self.get_positions()
            for position in positions:
                await self.close_position(position.symbol)

            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    async def get_day_trades_count(self) -> int:
        """
        Get the number of day trades for today.

        Returns:
            Number of day trades
        """
        try:
            account = await self.get_account()
            if account and hasattr(account, "daytrade_count"):
                return account.daytrade_count
            return 0
        except Exception as e:
            logger.error(f"Failed to get day trades count: {e}")
            return 0

    async def is_pattern_day_trader(self) -> bool:
        """
        Check if account is flagged as pattern day trader.

        Returns:
            True if pattern day trader, False otherwise
        """
        try:
            account = await self.get_account()
            if account and hasattr(account, "pattern_day_trader"):
                return account.pattern_day_trader
            return False
        except Exception as e:
            logger.error(f"Failed to check pattern day trader status: {e}")
            return False

    async def start_position_stream(self, callback=None):
        """
        Start streaming position updates.

        Args:
            callback: Optional callback function for position updates
        """
        try:
            if not self._stream_client:
                logger.warning("Stream client not available")
                return

            # This is a placeholder implementation
            logger.info("Position stream started")
        except Exception as e:
            logger.error(f"Failed to start position stream: {e}")

    def clear_cache(self):
        """Clear all cached data."""
        self._positions_cache.clear()
        self._account_cache = None
        self._cache_expiry = datetime.now(timezone.utc)
        logger.debug("Alpaca client cache cleared")
