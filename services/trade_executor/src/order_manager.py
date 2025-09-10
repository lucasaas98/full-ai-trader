"""
Order Management System for Trade Execution Service.

This module handles the complete order lifecycle including:
- Order validation and placement
- Bracket order management
- Partial fill handling
- Order cancellations and replacements
- Retry logic for failed orders
- Integration with risk management
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import asyncpg
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.config import get_config
from shared.models import (
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeSignal,
)

from .alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class OrderExecutionStrategy(str, Enum):
    """Order execution strategies."""

    IMMEDIATE = "immediate"
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    ADAPTIVE = "adaptive"


class BracketOrderStatus(str, Enum):
    """Bracket order status."""

    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderManager:
    """
    Comprehensive order management system.

    Handles order placement, tracking, and lifecycle management
    with sophisticated execution strategies and error handling.
    """

    def __init__(self, alpaca_client: AlpacaClient):
        """
        Initialize order manager.

        Args:
            alpaca_client: Alpaca API client instance
        """
        self.config = get_config()
        self.alpaca = alpaca_client
        self._db_pool = None
        self._redis = None
        self._active_orders: Dict[UUID, OrderResponse] = {}
        self._bracket_orders: Dict[UUID, Dict[str, OrderResponse]] = {}
        self._retry_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._running = False

    async def initialize(self) -> None:
        """Initialize database and Redis connections."""
        try:
            # Initialize database connection pool
            self._db_pool = await asyncpg.create_pool(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                user=self.config.database.username,
                password=self.config.database.password,
                min_size=5,
                max_size=20,
                command_timeout=30,
            )

            # Initialize Redis connection
            self._redis = redis.from_url(
                self.config.redis.url, max_connections=20, retry_on_timeout=True
            )

            logger.info("OrderManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OrderManager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._running = False
            if self._db_pool:
                await self._db_pool.close()
            if self._redis:
                await self._redis.close()
            logger.info("OrderManager cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def process_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """
        Process a trade signal and execute orders.

        Args:
            signal: Trade signal to process

        Returns:
            Execution result with order details
        """
        try:
            logger.info(
                f"Processing signal: {signal.id} for {signal.symbol} - {signal.signal_type}"
            )

            # Validate signal
            validation_result = await self._validate_signal(signal)
            if not validation_result["valid"]:
                logger.warning(
                    f"Signal validation failed: {validation_result['issues']}"
                )
                return {
                    "success": False,
                    "signal_id": signal.id,
                    "error": "Signal validation failed",
                    "issues": validation_result["issues"],
                }

            # Determine execution strategy
            execution_strategy = await self._determine_execution_strategy(signal)

            # Calculate position size
            position_size = await self._calculate_position_size(signal)
            if position_size <= 0:
                logger.warning(f"Invalid position size calculated: {position_size}")
                return {
                    "success": False,
                    "signal_id": signal.id,
                    "error": "Invalid position size",
                }

            # Execute based on strategy
            if execution_strategy == OrderExecutionStrategy.IMMEDIATE:
                result = await self._execute_immediate(signal, position_size)
            elif execution_strategy == OrderExecutionStrategy.TWAP:
                result = await self._execute_twap(signal, position_size)
            elif execution_strategy == OrderExecutionStrategy.VWAP:
                result = await self._execute_vwap(signal, position_size)
            elif execution_strategy == OrderExecutionStrategy.ICEBERG:
                result = await self._execute_iceberg(signal, position_size)
            else:  # ADAPTIVE
                result = await self._execute_adaptive(signal, position_size)

            # Store execution record
            await self._store_signal_execution(signal, result)

            return result

        except Exception as e:
            logger.error(f"Failed to process signal {signal.id}: {e}")
            await self._handle_execution_error(signal, e)
            return {"success": False, "signal_id": signal.id, "error": str(e)}

    async def _validate_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """Validate a trade signal before execution."""
        issues = []
        warnings = []

        try:
            # Check if market is open (unless extended hours allowed)
            if not await self.alpaca.is_market_open():
                issues.append("Market is closed")

            # Check for duplicate signals
            recent_signals = await self._get_recent_signals(signal.symbol, minutes=5)
            if any(
                s["signal_type"] == signal.signal_type.value for s in recent_signals
            ):
                warnings.append("Similar signal received recently")

            # Validate price levels
            if signal.stop_loss and signal.price:
                if (
                    signal.signal_type.value == "buy"
                    and signal.stop_loss >= signal.price
                ):
                    issues.append("Stop loss must be below entry price for buy signals")
                elif (
                    signal.signal_type.value == "sell"
                    and signal.stop_loss <= signal.price
                ):
                    issues.append(
                        "Stop loss must be above entry price for sell signals"
                    )

            if signal.take_profit and signal.price:
                if (
                    signal.signal_type.value == "buy"
                    and signal.take_profit <= signal.price
                ):
                    issues.append(
                        "Take profit must be above entry price for buy signals"
                    )
                elif (
                    signal.signal_type.value == "sell"
                    and signal.take_profit >= signal.price
                ):
                    issues.append(
                        "Take profit must be below entry price for sell signals"
                    )

            # Check confidence threshold
            if signal.confidence < 0.6:  # Configurable threshold
                warnings.append(f"Low confidence signal: {signal.confidence}")

            return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}

        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {e}"],
                "warnings": [],
            }

    async def _determine_execution_strategy(
        self, signal: TradeSignal
    ) -> OrderExecutionStrategy:
        """
        Determine the best execution strategy for a signal.

        Args:
            signal: Trade signal

        Returns:
            Chosen execution strategy
        """
        try:
            # Get market conditions
            quote = await self.alpaca.get_latest_quote(signal.symbol)

            # Calculate volatility and spread
            if not quote or "ask" not in quote or "bid" not in quote:
                return OrderExecutionStrategy.MARKET  # Default if no quote data

            spread_pct = (quote["ask"] - quote["bid"]) / (
                (quote["ask"] + quote["bid"]) / 2
            )

            # High confidence + low volatility = immediate execution
            if signal.confidence > 0.8 and spread_pct < 0.001:
                return OrderExecutionStrategy.IMMEDIATE

            # Large orders use TWAP/VWAP
            estimated_cost = (signal.quantity or 100) * quote["ask"]
            if estimated_cost > 50000:  # $50k threshold
                return OrderExecutionStrategy.VWAP

            # High volatility uses adaptive strategy
            if spread_pct > 0.005:
                return OrderExecutionStrategy.ADAPTIVE

            # Default to TWAP for medium-sized orders
            return OrderExecutionStrategy.TWAP

        except Exception as e:
            logger.warning(
                f"Failed to determine execution strategy, using default: {e}"
            )
            return OrderExecutionStrategy.IMMEDIATE

    async def _calculate_position_size(self, signal: TradeSignal) -> int:
        """
        Calculate position size based on risk parameters.

        Args:
            signal: Trade signal

        Returns:
            Position size in shares
        """
        try:
            # Use signal quantity if provided
            if signal.quantity and signal.quantity > 0:
                return signal.quantity

            # Calculate based on risk parameters
            risk_config = self.config.risk
            account = await self.alpaca.get_account()
            portfolio_value = Decimal(str(account.portfolio_value))

            # Risk per trade
            risk_per_trade = portfolio_value * risk_config.max_portfolio_risk

            # Use Alpaca's position sizing
            # Calculate stop price for position sizing
            entry_price = signal.price or Decimal("100")  # Default fallback
            stop_price = entry_price * (Decimal("1") - risk_config.stop_loss_percentage)

            shares = self.alpaca.calculate_position_size(
                signal.symbol, risk_per_trade, entry_price, stop_price
            )

            return shares

        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0

    async def _execute_immediate(
        self, signal: TradeSignal, quantity: int
    ) -> Dict[str, Any]:
        """Execute order immediately using market orders."""
        try:
            side = (
                OrderSide.BUY
                if signal.signal_type.value in ["buy", "long"]
                else OrderSide.SELL
            )

            # Place bracket order if stop loss and take profit are provided
            if signal.stop_loss and signal.take_profit:
                orders = await self.alpaca.place_bracket_order(
                    symbol=signal.symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=None,  # Market order
                    stop_loss_price=signal.stop_loss,
                    take_profit_price=signal.take_profit,
                )

                # Store bracket order
                bracket_id = uuid4()
                await self._store_bracket_order(bracket_id, orders, signal)

                return {
                    "success": True,
                    "execution_strategy": OrderExecutionStrategy.IMMEDIATE,
                    "orders": orders,
                    "bracket_id": bracket_id,
                    "quantity": quantity,
                }
            else:
                # Simple market order
                order_request = OrderRequest(
                    symbol=signal.symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    price=None,
                    stop_price=None,
                    client_order_id=str(uuid4()),
                )

                order_response = await self.alpaca.place_order(order_request)
                await self._store_order(order_request, order_response, signal.id)

                return {
                    "success": True,
                    "execution_strategy": OrderExecutionStrategy.IMMEDIATE,
                    "order": order_response,
                    "quantity": quantity,
                }

        except Exception as e:
            logger.error(f"Immediate execution failed for {signal.symbol}: {e}")
            raise

    async def _execute_twap(
        self, signal: TradeSignal, quantity: int, duration_minutes: int = 30
    ) -> Dict[str, Any]:
        """Execute order using Time-Weighted Average Price strategy."""
        try:
            logger.info(
                f"Executing TWAP order for {quantity} {signal.symbol} over {duration_minutes} minutes"
            )

            # Calculate slice parameters
            num_slices = min(
                10, duration_minutes // 3
            )  # Max 10 slices, min 3 minutes per slice
            slice_size = quantity // num_slices
            remaining_qty = quantity % num_slices
            slice_interval = timedelta(minutes=duration_minutes // num_slices)

            side = (
                OrderSide.BUY
                if signal.signal_type.value in ["buy", "long"]
                else OrderSide.SELL
            )
            orders = []

            # Execute slices
            for i in range(num_slices):
                try:
                    # Add remainder to last slice
                    current_slice_size = slice_size + (
                        remaining_qty if i == num_slices - 1 else 0
                    )

                    # Calculate TWAP target price
                    twap_price = await self.alpaca.calculate_twap_price(signal.symbol)
                    limit_price = None  # Initialize to avoid unbound variable error

                    # Check if TWAP price is available
                    if twap_price is None:
                        logger.warning(
                            f"TWAP price not available for {signal.symbol}, using market order"
                        )
                        order_request = OrderRequest(
                            symbol=signal.symbol,
                            side=side,
                            order_type=OrderType.MARKET,
                            quantity=current_slice_size,
                            price=None,
                            stop_price=None,
                            client_order_id=None,
                        )
                    else:
                        # Adjust price slightly for better execution probability
                        if side == OrderSide.BUY:
                            limit_price = twap_price * Decimal(
                                "1.001"
                            )  # 0.1% above TWAP
                        else:
                            limit_price = twap_price * Decimal(
                                "0.999"
                            )  # 0.1% below TWAP

                        order_request = OrderRequest(
                            symbol=signal.symbol,
                            side=side,
                            order_type=OrderType.LIMIT,
                            quantity=current_slice_size,
                            price=limit_price,
                            stop_price=None,
                            client_order_id=None,
                        )

                    order_response = await self.alpaca.place_order(order_request)
                    orders.append(order_response)
                    await self._store_order(order_request, order_response, signal.id)

                    logger.info(
                        f"VWAP slice {i + 1}/{num_slices} placed: {current_slice_size} @ {limit_price}"
                    )

                    # Wait before next slice (except for last one)
                    if i < num_slices - 1:
                        await asyncio.sleep(slice_interval.total_seconds())

                except Exception as e:
                    logger.error(f"TWAP slice {i + 1} failed: {e}")
                    # Continue with remaining slices
                    continue

            return {
                "success": True,
                "execution_strategy": OrderExecutionStrategy.TWAP,
                "orders": orders,
                "total_quantity": quantity,
                "executed_slices": len(orders),
            }

        except Exception as e:
            logger.error(f"TWAP execution failed for {signal.symbol}: {e}")
            raise

    async def _execute_vwap(
        self, signal: TradeSignal, quantity: int, duration_minutes: int = 30
    ) -> Dict[str, Any]:
        """Execute order using Volume-Weighted Average Price strategy."""
        try:
            logger.info(
                f"Executing VWAP order for {quantity} {signal.symbol} over {duration_minutes} minutes"
            )

            # Get historical volume pattern to determine slice sizes
            volume_profile = await self._get_volume_profile(
                signal.symbol, duration_minutes
            )

            side = (
                OrderSide.BUY
                if signal.signal_type.value in ["buy", "long"]
                else OrderSide.SELL
            )
            orders = []
            total_executed = 0

            for i, volume_weight in enumerate(volume_profile):
                try:
                    # Calculate slice size based on volume profile
                    slice_size = int(quantity * volume_weight)
                    if slice_size == 0:
                        continue

                    # Calculate VWAP target price
                    vwap_price = await self.alpaca.calculate_vwap_price(signal.symbol)
                    limit_price = None  # Initialize to avoid unbound variable error

                    # Check if VWAP price is available
                    if vwap_price is None:
                        logger.warning(
                            f"VWAP price not available for {signal.symbol}, using market order"
                        )
                        order_request = OrderRequest(
                            symbol=signal.symbol,
                            side=side,
                            order_type=OrderType.MARKET,
                            quantity=slice_size,
                            price=None,
                            stop_price=None,
                            client_order_id=None,
                        )
                    else:
                        # Adjust price for execution
                        if side == OrderSide.BUY:
                            limit_price = vwap_price * Decimal(
                                "1.0015"
                            )  # 0.15% above VWAP
                        else:
                            limit_price = vwap_price * Decimal(
                                "0.9985"
                            )  # 0.15% below VWAP

                        order_request = OrderRequest(
                            symbol=signal.symbol,
                            side=side,
                            order_type=OrderType.LIMIT,
                            quantity=slice_size,
                            price=limit_price,
                            stop_price=None,
                            client_order_id=None,
                        )

                    order_response = await self.alpaca.place_order(order_request)
                    orders.append(order_response)
                    await self._store_order(order_request, order_response, signal.id)

                    total_executed += slice_size
                    logger.info(
                        f"VWAP slice {i + 1} placed: {slice_size} @ {limit_price}"
                    )

                    # Wait between slices
                    await asyncio.sleep(duration_minutes * 60 / len(volume_profile))

                except Exception as e:
                    logger.error(f"VWAP slice {i + 1} failed: {e}")
                    continue

            return {
                "success": True,
                "execution_strategy": OrderExecutionStrategy.VWAP,
                "orders": orders,
                "total_quantity": quantity,
                "executed_quantity": total_executed,
            }

        except Exception as e:
            logger.error(f"VWAP execution failed for {signal.symbol}: {e}")
            raise

    async def _execute_iceberg(
        self, signal: TradeSignal, quantity: int, slice_size: int = 100
    ) -> Dict[str, Any]:
        """Execute large order using iceberg strategy."""
        try:
            logger.info(
                f"Executing iceberg order for {quantity} {signal.symbol} with slice size {slice_size}"
            )

            side = (
                OrderSide.BUY
                if signal.signal_type.value in ["buy", "long"]
                else OrderSide.SELL
            )
            orders = []
            remaining_qty = quantity

            while remaining_qty > 0:
                try:
                    current_slice = min(slice_size, remaining_qty)

                    # Get current best price
                    quote = await self.alpaca.get_latest_quote(signal.symbol)

                    # Check if quote data is available
                    if not quote or "ask" not in quote or "bid" not in quote:
                        logger.warning(
                            f"No quote data available for {signal.symbol}, using market order"
                        )
                        # Fall back to market order
                        order_request = OrderRequest(
                            symbol=signal.symbol,
                            side=side,
                            order_type=OrderType.MARKET,
                            quantity=current_slice,
                            price=None,
                            stop_price=None,
                            client_order_id=None,
                        )
                    else:
                        # Use slightly aggressive pricing to ensure fills
                        if side == OrderSide.BUY:
                            limit_price = quote["ask"] * Decimal(
                                "1.0005"
                            )  # 0.05% above ask
                        else:
                            limit_price = quote["bid"] * Decimal(
                                "0.9995"
                            )  # 0.05% below bid

                        order_request = OrderRequest(
                            symbol=signal.symbol,
                            side=side,
                            order_type=OrderType.LIMIT,
                            quantity=current_slice,
                            price=limit_price,
                            stop_price=None,
                            client_order_id=str(uuid4()),
                        )

                    order_response = await self.alpaca.place_order(order_request)
                    orders.append(order_response)
                    await self._store_order(order_request, order_response, signal.id)

                    # Wait for fill confirmation
                    if order_response.broker_order_id:
                        filled_qty = await self._wait_for_fill(
                            order_response.broker_order_id, timeout_seconds=60
                        )
                    else:
                        logger.warning(
                            "No broker order ID available, skipping fill wait"
                        )
                        filled_qty = current_slice
                    remaining_qty -= filled_qty

                    logger.info(
                        f"Iceberg slice executed: {filled_qty}/{current_slice}, remaining: {remaining_qty}"
                    )

                    # Small delay between slices
                    await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"Iceberg slice failed: {e}")
                    break

            return {
                "success": True,
                "execution_strategy": OrderExecutionStrategy.ICEBERG,
                "orders": orders,
                "total_quantity": quantity,
                "remaining_quantity": remaining_qty,
            }

        except Exception as e:
            logger.error(f"Iceberg execution failed for {signal.symbol}: {e}")
            raise

    async def _execute_adaptive(
        self, signal: TradeSignal, quantity: int
    ) -> Dict[str, Any]:
        """Execute order using adaptive strategy based on market conditions."""
        try:
            # Analyze current market conditions
            quote = await self.alpaca.get_latest_quote(signal.symbol)

            # Check if quote data is available
            if not quote or "ask" not in quote or "bid" not in quote:
                logger.warning(
                    f"No quote data available for {signal.symbol}, falling back to market order"
                )
                return await self._execute_immediate(signal, quantity)

            spread_pct = (quote["ask"] - quote["bid"]) / (
                (quote["ask"] + quote["bid"]) / 2
            )

            # Choose strategy based on conditions
            if spread_pct < 0.001 and signal.confidence > 0.8:
                # Low spread, high confidence - use market order
                return await self._execute_immediate(signal, quantity)
            elif quantity > 1000 or (quantity * quote["ask"]) > 25000:
                # Large order - use VWAP
                return await self._execute_vwap(signal, quantity)
            else:
                # Medium order - use TWAP
                return await self._execute_twap(signal, quantity, duration_minutes=15)

        except Exception as e:
            logger.error(f"Adaptive execution failed for {signal.symbol}: {e}")
            # Fallback to immediate execution
            return await self._execute_immediate(signal, quantity)

    async def place_bracket_order(
        self,
        signal: TradeSignal,
        quantity: int,
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Place a comprehensive bracket order.

        Args:
            signal: Trade signal
            quantity: Order quantity
            entry_price: Entry price (None for market)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Bracket order result
        """
        try:
            side = (
                OrderSide.BUY
                if signal.signal_type.value in ["buy", "long"]
                else OrderSide.SELL
            )

            # Use signal prices if not provided
            entry_price = entry_price or signal.price
            stop_loss = stop_loss or signal.stop_loss
            take_profit = take_profit or signal.take_profit

            logger.info(
                f"Placing bracket order: {quantity} {signal.symbol} @ {entry_price}"
            )

            # Place bracket order with Alpaca
            orders = await self.alpaca.place_bracket_order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )

            # Store bracket order relationship
            bracket_id = uuid4()
            await self._store_bracket_order(bracket_id, orders, signal)

            # Store individual orders
            for i, order in enumerate(orders):
                order_request = OrderRequest(
                    symbol=signal.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    price=getattr(order, "price", None),
                    stop_price=getattr(order, "stop_price", None),
                    client_order_id=str(uuid4()),
                )
                await self._store_order(order_request, order, signal.id)

            return {
                "success": True,
                "bracket_id": bracket_id,
                "orders": orders,
                "quantity": quantity,
            }

        except Exception as e:
            logger.error(f"Bracket order failed for {signal.symbol}: {e}")
            raise

    async def cancel_order(self, order_id: UUID, reason: str = "User request") -> bool:
        """
        Cancel an order.

        Args:
            order_id: Internal order ID
            reason: Cancellation reason

        Returns:
            True if cancellation successful
        """
        try:
            # Get order from database
            order = await self._get_order_by_id(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found")
                return False

            # Cancel with Alpaca
            success = await self.alpaca.cancel_order(order["broker_order_id"])

            if success:
                # Update order status in database
                await self._update_order_status(order_id, OrderStatus.CANCELLED, reason)
                logger.info(f"Order {order_id} cancelled: {reason}")

            return success

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_bracket_order(
        self, bracket_id: UUID, reason: str = "User request"
    ) -> bool:
        """
        Cancel all orders in a bracket.

        Args:
            bracket_id: Bracket order ID
            reason: Cancellation reason

        Returns:
            True if all cancellations successful
        """
        try:
            # Get bracket order details
            bracket = await self._get_bracket_order(bracket_id)
            if not bracket:
                logger.warning(f"Bracket order {bracket_id} not found")
                return False

            cancellation_results = []

            # Cancel all orders in bracket
            for order_type in [
                "entry_order_id",
                "stop_loss_order_id",
                "take_profit_order_id",
            ]:
                order_id = bracket.get(order_type)
                if order_id:
                    try:
                        result = await self.cancel_order(UUID(order_id), reason)
                        cancellation_results.append(result)
                    except Exception as e:
                        logger.error(
                            f"Failed to cancel {order_type} in bracket {bracket_id}: {e}"
                        )
                        cancellation_results.append(False)

            # Update bracket status
            if any(cancellation_results):
                await self._update_bracket_status(
                    bracket_id, BracketOrderStatus.CANCELLED
                )

            success = all(cancellation_results)
            logger.info(f"Bracket order {bracket_id} cancellation: {success}")
            return success

        except Exception as e:
            logger.error(f"Failed to cancel bracket order {bracket_id}: {e}")
            return False

    async def handle_partial_fill(
        self, order_id: UUID, filled_quantity: int
    ) -> Dict[str, Any]:
        """
        Handle partial order fills.

        Args:
            order_id: Order ID that was partially filled
            filled_quantity: Quantity that was filled

        Returns:
            Handling result
        """
        try:
            # Get order details
            order = await self._get_order_by_id(order_id)
            if not order:
                logger.error(f"Order {order_id} not found for partial fill handling")
                return {"success": False, "error": "Order not found"}

            remaining_qty = order["quantity"] - filled_quantity

            logger.info(
                f"Handling partial fill: {filled_quantity}/{order['quantity']} for {order['symbol']}"
            )

            # Update order status
            if remaining_qty > 0:
                await self._update_order_status(order_id, OrderStatus.PARTIALLY_FILLED)

                # Check if we should place order for remaining quantity
                if remaining_qty >= 10:  # Minimum remaining size threshold
                    # Create new order for remaining quantity
                    remaining_order = OrderRequest(
                        symbol=order["symbol"],
                        side=OrderSide(order["side"]),
                        order_type=OrderType(order["order_type"]),
                        quantity=remaining_qty,
                        price=order.get("price"),
                        stop_price=order.get("stop_price"),
                        client_order_id=str(uuid4()),
                        time_in_force=order.get("time_in_force", "day"),
                    )

                    try:
                        remaining_response = await self.alpaca.place_order(
                            remaining_order
                        )
                        await self._store_order(
                            remaining_order, remaining_response, order.get("signal_id")
                        )

                        logger.info(
                            f"Placed order for remaining quantity: {remaining_qty}"
                        )
                        return {
                            "success": True,
                            "filled_quantity": filled_quantity,
                            "remaining_quantity": remaining_qty,
                            "remaining_order": remaining_response,
                        }
                    except Exception as e:
                        logger.error(f"Failed to place remaining order: {e}")

            else:
                await self._update_order_status(order_id, OrderStatus.FILLED)

            return {
                "success": True,
                "filled_quantity": filled_quantity,
                "remaining_quantity": remaining_qty,
            }

        except Exception as e:
            logger.error(f"Failed to handle partial fill for order {order_id}: {e}")
            return {"success": False, "error": str(e)}

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def retry_failed_order(self, order_id: UUID) -> Dict[str, Any]:
        """
        Retry a failed order with updated parameters.

        Args:
            order_id: Failed order ID

        Returns:
            Retry result
        """
        try:
            # Get original order
            original_order = await self._get_order_by_id(order_id)
            if not original_order:
                return {"success": False, "error": "Original order not found"}

            # Check if original_order exists and has valid data
            if (
                not original_order
                or "symbol" not in original_order
                or "quantity" not in original_order
            ):
                return {"success": False, "error": "Invalid original order data"}

            # Update retry count
            retry_count = original_order.get("retry_count", 0) + 1
            if retry_count > 3:
                logger.warning(f"Order {order_id} exceeded max retry attempts")
                await self._update_order_status(
                    order_id, OrderStatus.REJECTED, "Max retries exceeded"
                )
                return {"success": False, "error": "Max retries exceeded"}

            # Get fresh market data
            quote = await self.alpaca.get_latest_quote(original_order["symbol"])

            # Check if quote data is available
            if not quote or "ask" not in quote or "bid" not in quote:
                logger.warning(f"No quote data available for retry of order {order_id}")
                return {"success": False, "error": "No quote data available"}

            # Adjust price for better execution probability
            side = OrderSide(original_order["side"])
            if side == OrderSide.BUY:
                new_price = quote["ask"] * Decimal("1.002")  # 0.2% above ask
            else:
                new_price = quote["bid"] * Decimal("0.998")  # 0.2% below bid

            # Create retry order
            retry_order = OrderRequest(
                symbol=original_order["symbol"],
                side=side,
                order_type=OrderType.LIMIT,
                quantity=original_order["quantity"],
                price=new_price,
                stop_price=None,
                client_order_id=str(uuid4()),
                time_in_force="ioc",
            )

            # Place retry order
            retry_response = await self.alpaca.place_order(retry_order)
            await self._store_order(
                retry_order, retry_response, original_order.get("signal_id")
            )

            # Update retry tracking
            await self._update_retry_count(order_id, retry_count)

            logger.info(
                f"Order {order_id} retry {retry_count} placed: {retry_response.broker_order_id}"
            )

            return {
                "success": True,
                "retry_count": retry_count,
                "retry_order": retry_response,
                "adjusted_price": new_price,
            }

        except Exception as e:
            logger.error(f"Order retry failed for {order_id}: {e}")
            await self._log_execution_error(order_id, None, "RETRY_FAILED", str(e))
            return {"success": False, "error": str(e)}

    async def _wait_for_fill(
        self, broker_order_id: str, timeout_seconds: int = 300
    ) -> int:
        """
        Wait for order to fill and return filled quantity.

        Args:
            broker_order_id: Broker order ID
            timeout_seconds: Maximum time to wait

        Returns:
            Filled quantity
        """
        try:
            start_time = datetime.now(timezone.utc)
            timeout = timedelta(seconds=timeout_seconds)

            while (datetime.now(timezone.utc) - start_time) < timeout:
                order = await self.alpaca.get_order(broker_order_id)

                if (
                    order
                    and hasattr(order, "status")
                    and order.status == OrderStatus.FILLED
                ):
                    return (
                        order.filled_quantity
                        if hasattr(order, "filled_quantity")
                        else 0
                    )
                elif (
                    order
                    and hasattr(order, "status")
                    and order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]
                ):
                    return (
                        order.filled_quantity
                        if hasattr(order, "filled_quantity")
                        else 0
                    )  # Return partial fill if any

                await asyncio.sleep(5)  # Check every 5 seconds

            # Timeout reached
            order = await self.alpaca.get_order(broker_order_id)
            return (
                order.filled_quantity
                if order and hasattr(order, "filled_quantity")
                else 0
            )

        except Exception as e:
            logger.error(f"Error waiting for fill of order {broker_order_id}: {e}")
            return 0

    async def _store_order(
        self,
        order_request: OrderRequest,
        order_response: OrderResponse,
        signal_id: Optional[UUID],
    ) -> Optional[bool]:
        """Store order in database."""
        try:
            if not self._db_pool:
                return False
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO trading.orders (
                        id, client_order_id, broker_order_id, symbol, side, order_type,
                        quantity, price, stop_price, filled_quantity, filled_price,
                        status, time_in_force, extended_hours, submitted_at,
                        signal_id, retry_count
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                    order_response.id,
                    order_request.client_order_id,
                    order_response.broker_order_id,
                    order_response.symbol,
                    order_response.side.value,
                    order_response.order_type.value,
                    order_response.quantity,
                    order_request.price,
                    order_request.stop_price,
                    order_response.filled_quantity,
                    order_response.filled_price,
                    order_response.status.value,
                    order_request.time_in_force,
                    order_request.extended_hours,
                    order_response.submitted_at,
                    signal_id,
                    0,
                )

            # Cache active order
            if order_response.status not in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            ]:
                self._active_orders[order_response.id] = order_response

        except Exception as e:
            logger.error(f"Failed to store order {order_response.id}: {e}")
            raise

    async def _store_bracket_order(
        self, bracket_id: UUID, orders: List[OrderResponse], signal: TradeSignal
    ) -> None:
        """Store bracket order relationship."""
        try:
            if not self._db_pool:
                return
            async with self._db_pool.acquire() as conn:
                # Assume first order is entry, subsequent are stop/profit
                entry_order = orders[0] if len(orders) > 0 else None
                stop_order = orders[1] if len(orders) > 1 else None
                profit_order = orders[2] if len(orders) > 2 else None

                entry_id = entry_order.id if entry_order else None
                stop_id = stop_order.id if stop_order else None
                profit_id = profit_order.id if profit_order else None

                await conn.execute(
                    """
                    INSERT INTO trading.bracket_orders (
                        id, parent_order_id, entry_order_id, stop_loss_order_id,
                        take_profit_order_id, status
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    bracket_id,
                    entry_id,
                    entry_id,
                    stop_id,
                    profit_id,
                    BracketOrderStatus.ACTIVE.value,
                )

            # Cache bracket order - convert list to dict format
            orders_dict = {}
            if len(orders) > 0:
                orders_dict["entry"] = orders[0]
            if len(orders) > 1:
                orders_dict["stop_loss"] = orders[1]
            if len(orders) > 2:
                orders_dict["take_profit"] = orders[2]
            self._bracket_orders[bracket_id] = orders_dict

        except Exception as e:
            logger.error(f"Failed to store bracket order {bracket_id}: {e}")
            raise

    async def _get_order_by_id(self, order_id: UUID) -> Optional[Dict[str, Any]]:
        """Get order details from database."""
        try:
            if not self._db_pool:
                return None
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM trading.orders WHERE id = $1
                """,
                    order_id,
                )

                if row:
                    return dict(row)
                return None

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def _get_bracket_order(self, bracket_id: UUID) -> Optional[Dict[str, Any]]:
        """Get bracket order details from database."""
        try:
            if not self._db_pool:
                return None
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM trading.bracket_orders WHERE id = $1
                """,
                    bracket_id,
                )

                if row:
                    return dict(row)
                return None

        except Exception as e:
            logger.error(f"Failed to get bracket order {bracket_id}: {e}")
            return None

    async def _update_order_status(
        self, order_id: UUID, status: OrderStatus, error_message: Optional[str] = None
    ) -> Optional[bool]:
        """Update order status in database."""
        try:
            if not self._db_pool:
                return False
            async with self._db_pool.acquire() as conn:
                if error_message:
                    await conn.execute(
                        """
                        UPDATE trading.orders
                        SET status = $2, error_message = $3, updated_at = NOW()
                        WHERE id = $1
                    """,
                        order_id,
                        status.value,
                        error_message,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE trading.orders
                        SET status = $2, updated_at = NOW()
                        WHERE id = $1
                    """,
                        order_id,
                        status.value,
                    )

            # Update cache
            if order_id in self._active_orders:
                self._active_orders[order_id].status = status
                if status in [
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                ]:
                    del self._active_orders[order_id]

        except Exception as e:
            logger.error(f"Failed to update order status {order_id}: {e}")
            raise

    async def _update_bracket_status(
        self, bracket_id: UUID, status: BracketOrderStatus
    ) -> None:
        """Update bracket order status."""
        try:
            if not self._db_pool:
                return
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE trading.bracket_orders
                    SET status = $2, updated_at = NOW()
                    WHERE id = $1
                """,
                    bracket_id,
                    status.value,
                )

        except Exception as e:
            logger.error(f"Failed to update bracket status {bracket_id}: {e}")
            raise

    async def _update_retry_count(self, order_id: UUID, retry_count: int) -> None:
        """Update order retry count."""
        try:
            if not self._db_pool:
                return
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE trading.orders
                    SET retry_count = $2, last_retry_at = NOW(), updated_at = NOW()
                    WHERE id = $1
                """,
                    order_id,
                    retry_count,
                )

        except Exception as e:
            logger.error(f"Failed to update retry count for {order_id}: {e}")

    async def _get_recent_signals(
        self, symbol: str, minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent signals for duplicate detection."""
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT signal_id, symbol, side, created_at
                    FROM trading.orders
                    WHERE symbol = $1
                    AND created_at > NOW() - INTERVAL '1 minute' * $2
                    ORDER BY created_at DESC
                """,
                    symbol,
                    minutes,
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get recent signals for {symbol}: {e}")
            return []

    async def _store_signal_execution(
        self, signal: TradeSignal, execution_result: Dict[str, Any]
    ) -> None:
        """Store signal execution record."""
        try:
            if not self._db_pool:
                return
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO trading.signal_executions (
                        signal_id, signal_timestamp, signal_price, signal_confidence,
                        execution_timestamp, execution_status
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    signal.id,
                    signal.timestamp,
                    signal.price,
                    signal.confidence,
                    datetime.now(timezone.utc),
                    "success" if execution_result["success"] else "failed",
                )

        except Exception as e:
            logger.error(f"Failed to store signal execution: {e}")

    async def _handle_execution_error(
        self, signal: TradeSignal, error: Exception
    ) -> None:
        """Handle execution errors."""
        try:
            await self._log_execution_error(
                None, signal.id, "EXECUTION_FAILED", str(error)
            )

            # Publish error to Redis
            if self._redis:
                import json

                await self._redis.publish(
                    f"execution_errors:{signal.symbol}",
                    json.dumps(
                        {
                            "signal_id": str(signal.id),
                            "symbol": signal.symbol,
                            "error": str(error),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        default=str,
                    ),
                )

        except Exception as e:
            logger.error(f"Failed to handle execution error: {e}")

    async def _log_execution_error(
        self,
        order_id: Optional[UUID],
        signal_id: Optional[UUID],
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log execution error to database."""
        try:
            if not self._db_pool:
                return
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO trading.execution_errors (
                        order_id, signal_id, error_type, error_message, context
                    ) VALUES ($1, $2, $3, $4, $5)
                """,
                    order_id,
                    signal_id,
                    error_type,
                    error_message,
                    context,
                )

        except Exception as e:
            logger.error(f"Failed to log execution error: {e}")

    async def get_active_orders(
        self, symbol: Optional[str] = None
    ) -> List[OrderResponse]:
        """
        Get currently active orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of active orders
        """
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                if symbol:
                    query = """
                        SELECT * FROM trading.orders
                        WHERE symbol = $1 AND status IN ('pending', 'submitted', 'partially_filled')
                        ORDER BY created_at DESC
                    """
                    rows = await conn.fetch(query, symbol)
                else:
                    query = """
                        SELECT * FROM trading.orders
                        WHERE status IN ('pending', 'submitted', 'partially_filled')
                        ORDER BY created_at DESC
                    """
                    rows = await conn.fetch(query)

                orders = []
                for row in rows:
                    order = OrderResponse(
                        id=row["id"],
                        broker_order_id=row["broker_order_id"],
                        symbol=row["symbol"],
                        side=OrderSide(row["side"]),
                        order_type=OrderType(row["order_type"]),
                        status=OrderStatus(row["status"]),
                        quantity=row["quantity"],
                        filled_quantity=row["filled_quantity"],
                        price=row["price"],
                        filled_price=row["filled_price"],
                        submitted_at=row["submitted_at"],
                        filled_at=row["filled_at"],
                        cancelled_at=row["cancelled_at"],
                        commission=row["commission"],
                    )
                    orders.append(order)

                return orders

        except Exception as e:
            logger.error(f"Failed to get active orders: {e}")
            return []

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[OrderResponse]:
        """
        Get order history from database.

        Args:
            symbol: Optional symbol filter
            start_date: Optional start date
            limit: Maximum orders to return

        Returns:
            List of historical orders
        """
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                conditions = []
                params = []
                param_count = 0

                if symbol:
                    param_count += 1
                    conditions.append(f"symbol = ${param_count}")
                    params.append(symbol)

                if start_date:
                    param_count += 1
                    conditions.append(f"created_at >= ${param_count}")
                    params.append(start_date)

                where_clause = (
                    " WHERE " + " AND ".join(conditions) if conditions else ""
                )
                param_count += 1

                query = f"""
                    SELECT * FROM trading.orders
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ${param_count}
                """
                params.append(limit)

                rows = await conn.fetch(query, *params)

                orders = []
                for row in rows:
                    order = OrderResponse(
                        id=row["id"],
                        broker_order_id=row["broker_order_id"],
                        symbol=row["symbol"],
                        side=OrderSide(row["side"]),
                        order_type=OrderType(row["order_type"]),
                        status=OrderStatus(row["status"]),
                        quantity=row["quantity"],
                        filled_quantity=row["filled_quantity"],
                        price=row["price"],
                        filled_price=row["filled_price"],
                        submitted_at=row["submitted_at"],
                        filled_at=row["filled_at"],
                        cancelled_at=row["cancelled_at"],
                        commission=row["commission"],
                    )
                    orders.append(order)

                return orders

        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            return []

    async def sync_order_status(self, order_id: UUID) -> bool:
        """
        Sync order status with Alpaca.

        Args:
            order_id: Order ID to sync

        Returns:
            True if sync successful
        """
        try:
            # Get order from database
            order = await self._get_order_by_id(order_id)
            if not order:
                return False

            # Get current status from Alpaca
            alpaca_order = await self.alpaca.get_order(order["broker_order_id"])

            # Check if alpaca order exists and has status
            if not alpaca_order or not hasattr(alpaca_order, "status"):
                return False

            # Update if status changed
            if alpaca_order.status != OrderStatus(order["status"]):
                if self._db_pool:
                    async with self._db_pool.acquire() as conn:
                        await conn.execute(
                            """
                            UPDATE trading.orders
                            SET status = $2, filled_quantity = $3, filled_price = $4,
                                filled_at = $5, cancelled_at = $6, updated_at = NOW()
                            WHERE id = $1
                        """,
                            order_id,
                            getattr(alpaca_order, "status", "unknown"),
                            getattr(alpaca_order, "filled_qty", 0),
                            getattr(alpaca_order, "filled_avg_price", None),
                            getattr(alpaca_order, "filled_at", None),
                            getattr(alpaca_order, "canceled_at", None),
                        )

                logger.info(f"Order {order_id} status synced: {alpaca_order.status}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to sync order status {order_id}: {e}")
            return False

    async def start_order_monitoring(self) -> None:
        """Start background task for order monitoring."""
        self._running = True

        async def monitor_orders() -> None:
            """Monitor active orders and handle updates."""
            while self._running:
                try:
                    # Get all active orders
                    active_orders = await self.get_active_orders()

                    for order in active_orders:
                        try:
                            # Sync status with Alpaca
                            await self.sync_order_status(order.id)

                            # Handle timeouts
                            if order.submitted_at:
                                age = datetime.now(timezone.utc) - order.submitted_at
                                if (
                                    age > timedelta(hours=1)
                                    and order.status == OrderStatus.SUBMITTED
                                ):
                                    logger.warning(
                                        f"Order {order.id} has been pending for {age}"
                                    )
                                    # Consider cancelling stale orders

                        except Exception as e:
                            logger.error(f"Error monitoring order {order.id}: {e}")

                    await asyncio.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    logger.error(f"Error in order monitoring loop: {e}")
                    await asyncio.sleep(60)  # Longer wait on error

        # Start monitoring task
        asyncio.create_task(monitor_orders())
        logger.info("Order monitoring started")

    async def get_execution_metrics(
        self, symbol: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get execution performance metrics.

        Args:
            symbol: Optional symbol filter
            days: Number of days to analyze

        Returns:
            Execution metrics
        """
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                # Base query conditions
                conditions = ["created_at >= NOW() - INTERVAL '1 day' * $1"]
                params: List[Any] = [days]

                if symbol:
                    conditions.append("symbol = $2")
                    params.append(symbol)

                where_clause = " WHERE " + " AND ".join(conditions)

                # Get basic metrics
                metrics_row = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*) as total_orders,
                        COUNT(CASE WHEN status = 'filled' THEN 1 END) as filled_orders,
                        COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_orders,
                        COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected_orders,
                        COUNT(CASE WHEN status = 'partially_filled' THEN 1 END) as partial_fills,
                        AVG(CASE WHEN filled_at IS NOT NULL AND submitted_at IS NOT NULL
                            THEN EXTRACT(EPOCH FROM (filled_at - submitted_at)) END) as avg_fill_time_seconds,
                        AVG(retry_count) as avg_retries
                    FROM trading.orders
                    {where_clause}
                """,
                    *params,
                )

                # Get slippage metrics
                slippage_row = await conn.fetchrow(
                    f"""
                    SELECT
                        AVG(ABS(filled_price - price) / NULLIF(price, 0)) as avg_slippage,
                        MAX(ABS(filled_price - price) / NULLIF(price, 0)) as max_slippage
                    FROM trading.orders
                    {where_clause}
                    AND filled_price IS NOT NULL AND price IS NOT NULL
                """,
                    *params,
                )

                return {
                    "period_days": days,
                    "symbol": symbol,
                    "total_orders": metrics_row["total_orders"],
                    "fill_rate": metrics_row["filled_orders"]
                    / max(metrics_row["total_orders"], 1),
                    "cancellation_rate": metrics_row["cancelled_orders"]
                    / max(metrics_row["total_orders"], 1),
                    "rejection_rate": metrics_row["rejected_orders"]
                    / max(metrics_row["total_orders"], 1),
                    "partial_fill_rate": metrics_row["partial_fills"]
                    / max(metrics_row["total_orders"], 1),
                    "avg_fill_time_seconds": float(
                        metrics_row["avg_fill_time_seconds"] or 0
                    ),
                    "avg_retries": float(metrics_row["avg_retries"] or 0),
                    "avg_slippage": float(slippage_row["avg_slippage"] or 0),
                    "max_slippage": float(slippage_row["max_slippage"] or 0),
                }

        except Exception as e:
            logger.error(f"Failed to get execution metrics: {e}")
            return {}

    async def _get_volume_profile(
        self, symbol: str, duration_minutes: int
    ) -> List[float]:
        """
        Get volume profile for VWAP execution.

        Args:
            symbol: Trading symbol
            duration_minutes: Duration in minutes

        Returns:
            List of volume weights for each time slice
        """
        try:
            # Simple implementation - return equal weights for now
            # In a real implementation, this would analyze historical volume patterns
            num_slices = max(1, duration_minutes // 5)  # 5-minute slices
            weight_per_slice = 1.0 / num_slices
            return [weight_per_slice] * num_slices
        except Exception as e:
            logger.error(f"Failed to get volume profile for {symbol}: {e}")
            # Return equal weights as fallback
            return [0.2, 0.2, 0.2, 0.2, 0.2]  # 5 equal slices
