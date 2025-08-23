"""
Trade Execution Engine - Main Coordinator.

This module provides the main execution engine that coordinates all components
of the trade execution service including order management, position tracking,
performance monitoring, and Redis integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Optional, Any, List
from uuid import UUID
import json

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware


from shared.models import TradeSignal, OrderRequest, OrderSide, SignalType
from shared.config import get_config
from .alpaca_client import AlpacaClient, AlpacaAPIError
from .order_manager import OrderManager
from .position_tracker import PositionTracker
from .performance_tracker import PerformanceTracker


logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Main trade execution engine.

    Coordinates all execution components and provides the main API interface
    for trade execution, position management, and performance tracking.
    """

    def __init__(self):
        """Initialize execution engine."""
        self.config = get_config()
        self.app = FastAPI(
            title="Trade Execution Service",
            description="Automated trade execution with Alpaca API",
            version="1.0.0"
        )

        # Initialize components
        self.alpaca_client = AlpacaClient()
        self.order_manager = OrderManager(self.alpaca_client)
        self.position_tracker = PositionTracker(self.alpaca_client)
        self.performance_tracker = PerformanceTracker()

        # Redis and database
        self._redis = None
        self._db_pool = None
        self._running = False

        # Setup FastAPI routes
        self._setup_routes()

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def initialize(self):
        """Initialize all components and connections."""
        try:
            logger.info("Initializing Trade Execution Engine...")

            # Initialize database pool
            self._db_pool = await asyncpg.create_pool(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                user=self.config.database.username,
                password=self.config.database.password,
                min_size=10,
                max_size=30,
                command_timeout=60
            )

            # Initialize Redis
            self._redis = redis.from_url(
                self.config.redis.url,
                max_connections=20,
                retry_on_timeout=True
            )

            # Initialize components
            await self.alpaca_client.connect()
            await self.order_manager.initialize()
            await self.position_tracker.initialize()
            await self.performance_tracker.initialize()

            # Start background tasks
            await self._start_background_tasks()

            logger.info("Trade Execution Engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize execution engine: {e}")
            raise

    async def cleanup(self):
        """Clean up all resources."""
        try:
            self._running = False

            # Cleanup components
            await self.alpaca_client.disconnect()
            await self.order_manager.cleanup()
            await self.position_tracker.cleanup()
            await self.performance_tracker.cleanup()

            # Close connections
            if self._db_pool:
                await self._db_pool.close()
            if self._redis:
                await self._redis.close()

            logger.info("Execution engine cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                alpaca_health = await self.alpaca_client.health_check()
                return {
                    "status": "healthy",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alpaca_status": alpaca_health,
                    "components": {
                        "order_manager": True,
                        "position_tracker": True,
                        "performance_tracker": True
                    }
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

        @self.app.post("/execute/signal")
        async def execute_signal(signal: TradeSignal, background_tasks: BackgroundTasks):
            """Execute a trade signal."""
            try:
                logger.info(f"Received signal for execution: {signal.id}")

                # Add to background execution
                background_tasks.add_task(self._execute_signal_async, signal)

                return {
                    "success": True,
                    "signal_id": signal.id,
                    "message": "Signal queued for execution",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            except Exception as e:
                logger.error(f"Failed to queue signal {signal.id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/orders/place")
        async def place_order(order_request: OrderRequest):
            """Place a single order."""
            try:
                # Validate order
                validation = await self.alpaca_client.validate_order(order_request)
                if not validation['valid']:
                    raise HTTPException(status_code=400, detail=validation['issues'])

                # Place order
                order_response = await self.alpaca_client.place_order(order_request)

                return {
                    "success": True,
                    "order": order_response.dict(),
                    "warnings": validation.get('warnings', [])
                }

            except AlpacaAPIError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Failed to place order: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/orders/{order_id}/cancel")
        async def cancel_order(order_id: UUID):
            """Cancel an order."""
            try:
                success = await self.order_manager.cancel_order(order_id)
                return {
                    "success": success,
                    "order_id": order_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/positions")
        async def get_positions():
            """Get all current positions."""
            try:
                positions = await self.position_tracker.get_all_positions()
                portfolio_metrics = await self.position_tracker.calculate_portfolio_metrics()

                return {
                    "positions": positions,
                    "portfolio_metrics": portfolio_metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to get positions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/positions/{symbol}")
        async def get_position(symbol: str):
            """Get position for specific symbol."""
            try:
                position = await self.position_tracker.get_position(symbol)
                if not position:
                    raise HTTPException(status_code=404, detail="Position not found")

                return position
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get position for {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/positions/{symbol}/close")
        async def close_position(symbol: str, percentage: Optional[float] = None):
            """Close a position."""
            try:
                position = await self.position_tracker.get_position(symbol)
                if not position:
                    raise HTTPException(status_code=404, detail="Position not found")

                # Close position
                close_order = await self.alpaca_client.close_position(
                    symbol=symbol,
                    percentage=percentage or 100.0
                )

                return {
                    "success": True,
                    "close_order": close_order.dict() if close_order and hasattr(close_order, 'dict') else {},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            except Exception as e:
                logger.error(f"Failed to close position {symbol}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/orders/active")
        async def get_active_orders():
            """Get all active orders."""
            try:
                orders = await self.order_manager.get_active_orders()
                return {
                    "orders": [order.dict() for order in orders],
                    "count": len(orders),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to get active orders: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/performance/summary")
        async def get_performance_summary(days: int = 30):
            """Get performance summary."""
            try:
                summary = await self.performance_tracker.get_performance_summary(days)
                return summary
            except Exception as e:
                logger.error(f"Failed to get performance summary: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/performance/daily")
        async def get_daily_performance():
            """Get today's performance."""
            try:
                performance = await self.performance_tracker.calculate_daily_performance()
                return performance
            except Exception as e:
                logger.error(f"Failed to get daily performance: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/performance/strategy/{strategy_name}")
        async def get_strategy_performance(strategy_name: str, days: int = 30):
            """Get performance for specific strategy."""
            try:
                performance = await self.performance_tracker.calculate_strategy_performance(strategy_name, days)
                return performance
            except Exception as e:
                logger.error(f"Failed to get strategy performance: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/export/tradenote")
        async def export_tradenote(
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            strategy: Optional[str] = None
        ):
            """Export data for TradeNote."""
            try:
                from datetime import date, timezone

                start_dt = date.fromisoformat(start_date) if start_date else None
                end_dt = date.fromisoformat(end_date) if end_date else None

                trades = await self.performance_tracker.export_for_tradenote(
                    start_date=start_dt,
                    end_date=end_dt,
                    strategy_name=strategy
                )

                return {
                    "trades": trades,
                    "count": len(trades),
                    "export_timestamp": datetime.now(timezone.utc).isoformat()
                }

            except Exception as e:
                logger.error(f"Failed to export TradeNote data: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sync/positions")
        async def sync_positions():
            """Sync positions with Alpaca."""
            try:
                result = await self.position_tracker.sync_with_alpaca()
                return result
            except Exception as e:
                logger.error(f"Failed to sync positions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/account")
        async def get_account_info():
            """Get account information."""
            try:
                account = await self.alpaca_client.get_account()
                if not account:
                    return {
                        "account_id": "unknown",
                        "equity": 0.0,
                        "cash": 0.0,
                        "buying_power": 0.0,
                        "portfolio_value": 0.0,
                        "day_trades_count": 0,
                        "pattern_day_trader": False,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                return {
                    "account_id": getattr(account, 'id', 'unknown'),
                    "equity": float(getattr(account, 'equity', 0)),
                    "cash": float(getattr(account, 'cash', 0)),
                    "buying_power": float(getattr(account, 'buying_power', 0)),
                    "portfolio_value": float(getattr(account, 'portfolio_value', 0)),
                    "day_trades_count": int(getattr(account, 'daytrade_count', 0) or 0),
                    "pattern_day_trader": bool(getattr(account, 'pattern_day_trader', False)),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to get account info: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _execute_signal_async(self, signal: TradeSignal):
        """Execute signal asynchronously."""
        try:
            logger.info(f"Executing signal: {signal.id} for {signal.symbol}")

            # Process signal through order manager
            result = await self.order_manager.process_signal(signal)

            # Publish execution result
            await self._publish_execution_result(signal, result)

            logger.info(f"Signal {signal.id} execution completed: {result['success']}")

        except Exception as e:
            logger.error(f"Failed to execute signal {signal.id}: {e}")
            await self._publish_execution_error(signal, str(e))

    async def _publish_execution_result(self, signal: TradeSignal, result: Dict[str, Any]):
        """Publish execution result to Redis."""
        try:
            message = {
                "signal_id": str(signal.id),
                "symbol": signal.symbol,
                "strategy": signal.strategy_name,
                "success": result["success"],
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Publish to multiple channels
            if self._redis:
                await self._redis.publish(f"executions:{signal.symbol}", json.dumps(message, default=str))
                await self._redis.publish("executions:all", json.dumps(message, default=str))

        except Exception as e:
            logger.error(f"Failed to publish execution result: {e}")

    async def _publish_execution_error(self, signal: TradeSignal, error: str):
        """Publish execution error to Redis."""
        try:
            message = {
                "signal_id": str(signal.id),
                "symbol": signal.symbol,
                "strategy": signal.strategy_name,
                "success": False,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            if self._redis:
                await self._redis.publish(f"execution_errors:{signal.symbol}", json.dumps(message, default=str))
                await self._redis.publish("execution_errors:all", json.dumps(message, default=str))

        except Exception as e:
            logger.error(f"Failed to publish execution error: {e}")

    async def _start_background_tasks(self):
        """Start all background monitoring tasks."""
        try:
            self._running = True

            # Start signal listener
            asyncio.create_task(self._signal_listener())

            # Start order monitoring
            await self.order_manager.start_order_monitoring()

            # Start performance monitoring
            await self.performance_tracker.start_performance_monitoring()

            # Start position risk monitoring
            asyncio.create_task(self.position_tracker.monitor_position_risks())

            # Start periodic sync
            asyncio.create_task(self._periodic_sync())

            # Start daily cleanup
            asyncio.create_task(self._daily_cleanup())

            logger.info("Background tasks started")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise

    async def _signal_listener(self):
        """Listen for trade signals on Redis."""
        try:
            if self._redis:
                pubsub = self._redis.pubsub()
            else:
                logger.error("Redis connection not available")
                return
            await pubsub.subscribe("signals:*")

            logger.info("Signal listener started")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Parse signal
                        signal_data = json.loads(message["data"])
                        signal = TradeSignal(**signal_data)

                        # Process signal
                        asyncio.create_task(self._execute_signal_async(signal))

                    except Exception as e:
                        logger.error(f"Error processing signal message: {e}")

        except Exception as e:
            logger.error(f"Signal listener error: {e}")
            if self._running:
                # Restart listener after delay
                await asyncio.sleep(30)
                asyncio.create_task(self._signal_listener())

    async def _periodic_sync(self):
        """Periodic synchronization with Alpaca."""
        while self._running:
            try:
                # Sync positions every 5 minutes
                await self.position_tracker.sync_with_alpaca()

                # Update account snapshot
                await self._update_account_snapshot()

                # Sync order statuses
                await self._sync_all_order_statuses()

                logger.debug("Periodic sync completed")

            except Exception as e:
                logger.error(f"Error in periodic sync: {e}")

            await asyncio.sleep(300)  # 5 minutes

    async def _update_account_snapshot(self):
        """Update account snapshot in database."""
        try:
            account = await self.alpaca_client.get_account()

            if self._db_pool:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                    INSERT INTO trading.account_snapshots (
                        account_id, timestamp, cash, buying_power, total_equity,
                        total_market_value, total_unrealized_pnl, day_trades_count,
                        pattern_day_trader
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    str(getattr(account, 'id', 'unknown')), datetime.now(timezone.utc), Decimal(str(getattr(account, 'cash', 0))),
                    Decimal(str(getattr(account, 'buying_power', 0))), Decimal(str(getattr(account, 'equity', 0))),
                    Decimal(str(getattr(account, 'portfolio_value', 0))), Decimal(str(getattr(account, 'portfolio_value', 0))) - Decimal(str(getattr(account, 'cash', 0))),
                    int(getattr(account, 'daytrade_count', 0) or 0), bool(getattr(account, 'pattern_day_trader', False))
                )

        except Exception as e:
            logger.error(f"Failed to update account snapshot: {e}")

    async def _sync_all_order_statuses(self):
        """Sync status of all active orders."""
        try:
            active_orders = await self.order_manager.get_active_orders()

            for order in active_orders:
                try:
                    await self.order_manager.sync_order_status(order.id)
                except Exception as e:
                    logger.error(f"Failed to sync order {order.id}: {e}")

        except Exception as e:
            logger.error(f"Failed to sync order statuses: {e}")

    async def _daily_cleanup(self):
        """Daily cleanup tasks."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Run cleanup at 2 AM UTC
                if now.hour == 2 and now.minute < 5:
                    logger.info("Starting daily cleanup")

                    # Clean up old snapshots
                    await self.position_tracker.cleanup_old_snapshots(days_to_keep=7)

                    # Clean up old performance data
                    await self.performance_tracker._cleanup_old_performance_data(days_to_keep=365)

                    # Update yesterday's performance
                    yesterday = now.date() - timedelta(days=1)
                    await self.performance_tracker.update_daily_metrics(yesterday)

                    logger.info("Daily cleanup completed")

                    # Sleep for an hour to avoid running multiple times
                    await asyncio.sleep(3600)
                else:
                    # Check every hour
                    await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Error in daily cleanup: {e}")
                await asyncio.sleep(3600)

    async def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal data

        Returns:
            Execution result
        """
        try:
            return await self.order_manager.process_signal(signal)
        except Exception as e:
            logger.error(f"Signal execution failed: {e}")
            raise

    async def update_watchlist(self, symbols: List[str]) -> None:
        """
        Update the watchlist with new symbols from screener updates.

        Args:
            symbols: List of symbols to add to watchlist
        """
        try:
            if not symbols:
                return

            # Add symbols to internal tracking
            if not hasattr(self, '_watchlist'):
                self._watchlist = set()

            new_symbols = set(symbols) - self._watchlist
            if new_symbols:
                self._watchlist.update(new_symbols)
                logger.info(f"Added {len(new_symbols)} new symbols to watchlist: {new_symbols}")

                # Optionally trigger position tracking initialization for new symbols
                if hasattr(self, 'position_tracker') and self.position_tracker:
                    for symbol in new_symbols:
                        await self.position_tracker.initialize_symbol_tracking(symbol)

        except Exception as e:
            logger.error(f"Error updating watchlist: {e}")

    async def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        strategy_name: str = "manual"
    ) -> Dict[str, Any]:
        """
        Place a bracket order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            entry_price: Entry price (None for market)
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy_name: Strategy name for tracking

        Returns:
            Bracket order result
        """
        try:
            # Create trade signal for bracket order
            signal = TradeSignal(
                symbol=symbol,
                signal_type=SignalType.BUY if side == OrderSide.BUY else SignalType.SELL,
                confidence=1.0,
                price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name=strategy_name
            )

            # Execute through order manager
            return await self.order_manager.place_bracket_order(
                signal, quantity, entry_price, stop_loss, take_profit
            )

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            raise

    async def get_account_summary(self) -> Dict[str, Any]:
        """Get comprehensive account summary."""
        try:
            # Get account info
            account = await self.alpaca_client.get_account()

            # Get positions
            positions = await self.position_tracker.get_all_positions()

            # Get active orders
            active_orders = await self.order_manager.get_active_orders()

            # Get performance metrics
            performance = await self.performance_tracker.get_performance_summary(days=30)

            # Calculate portfolio metrics
            portfolio_metrics = await self.position_tracker.calculate_portfolio_metrics()

            return {
                "account": {
                    "equity": float(getattr(account, 'equity', 0)),
                    "cash": float(getattr(account, 'cash', 0)),
                    "buying_power": float(getattr(account, 'buying_power', 0)),
                    "day_trades_count": int(getattr(account, 'daytrade_count', 0) or 0),
                    "pattern_day_trader": bool(getattr(account, 'pattern_day_trader', False))
                },
                "positions": {
                    "count": len(positions),
                    "total_market_value": portfolio_metrics.get('total_market_value', 0),
                    "total_unrealized_pnl": portfolio_metrics.get('total_unrealized_pnl', 0)
                },
                "orders": {
                    "active_count": len(active_orders)
                },
                "performance_30d": performance,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            raise

    async def emergency_stop(self) -> Dict[str, Any]:
        """
        Emergency stop - cancel all orders and close all positions.

        Returns:
            Emergency stop result
        """
        try:
            logger.warning("EMERGENCY STOP INITIATED")

            # Cancel all orders
            cancel_success = await self.alpaca_client.cancel_all_orders()

            # Close all positions
            close_orders = await self.alpaca_client.close_all_positions()

            # Log emergency stop
            if self._db_pool:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                    INSERT INTO trading.execution_errors (
                        error_type, error_message, context
                    ) VALUES ($1, $2, $3)
                """,
                    "EMERGENCY_STOP", "Emergency stop executed",
                    {"timestamp": datetime.now(timezone.utc).isoformat()}
                )

            # Publish emergency stop notification
            if self._redis:
                await self._redis.publish("emergency_stop", json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "orders_cancelled": cancel_success,
                "positions_closed": 1 if close_orders else 0
                }, default=str))

            return {
                "success": True,
                "orders_cancelled": cancel_success,
                "positions_closed": 1 if close_orders else 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            raise

    async def get_execution_metrics(self, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Get execution quality metrics.

        Args:
            symbol: Optional symbol filter
            days: Number of days to analyze

        Returns:
            Execution metrics
        """
        try:
            order_metrics = await self.order_manager.get_execution_metrics(symbol, days)

            # Get additional execution data
            execution_data = None
            if self._db_pool:
                async with self._db_pool.acquire() as conn:
                    execution_data = await conn.fetchrow("""
                    SELECT
                        AVG(slippage) as avg_slippage,
                        AVG(price_improvement) as avg_price_improvement,
                        AVG(execution_duration_seconds) as avg_execution_time,
                        COUNT(*) as total_executions
                    FROM trading.execution_metrics
                    WHERE created_at >= NOW() - INTERVAL '1 day' * $2
                    AND ($1 IS NULL OR symbol = $1)
                """, days, symbol)

            if execution_data:
                order_metrics.update({
                    "avg_slippage": float(execution_data['avg_slippage'] or 0),
                    "avg_price_improvement": float(execution_data['avg_price_improvement'] or 0),
                    "avg_execution_time": float(execution_data['avg_execution_time'] or 0),
                    "total_executions": execution_data['total_executions']
                })

            return order_metrics

        except Exception as e:
            logger.error(f"Failed to get execution metrics: {e}")
            return {}

    async def force_position_close(self, symbol: str, reason: str = "forced") -> Dict[str, Any]:
        """
        Force close a position (emergency).

        Args:
            symbol: Symbol to close
            reason: Reason for forced close

        Returns:
            Close result
        """
        try:
            logger.warning(f"Force closing position for {symbol}: {reason}")

            # Get current position
            position = await self.position_tracker.get_position(symbol)
            if not position:
                return {"success": False, "error": "Position not found"}

            # Get current price
            quote = await self.alpaca_client.get_latest_quote(symbol)
            if not quote or 'bid' not in quote or 'ask' not in quote:
                logger.warning(f"No quote data available for {symbol}, using fallback price")
                current_price = Decimal("0")
            else:
                current_price = Decimal(str((quote['bid'] + quote['ask']) / 2))

            # Close position via Alpaca
            close_order = await self.alpaca_client.close_position(symbol)

            # Update position status
            await self.position_tracker.close_position(
                position['id'],
                current_price,
                datetime.now(timezone.utc),
                reason
            )

            # Log forced close
            if self._db_pool:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                    INSERT INTO trading.execution_errors (
                        error_type, error_message, context
                    ) VALUES ($1, $2, $3)
                """,
                    "FORCED_CLOSE", f"Position {symbol} force closed: {reason}",
                    {"symbol": symbol, "reason": reason, "price": str(current_price)}
                )

            return {
                "success": True,
                "symbol": symbol,
                "close_order": close_order.dict() if close_order and hasattr(close_order, 'dict') else {},
                "close_price": float(current_price),
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to force close position {symbol}: {e}")
            return {"success": False, "error": str(e)}

    async def adjust_position_stops(
        self,
        symbol: str,
        new_stop_loss: Optional[Decimal] = None,
        new_take_profit: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Adjust stop loss and take profit for a position.

        Args:
            symbol: Trading symbol
            new_stop_loss: New stop loss price
            new_take_profit: New take profit price

        Returns:
            Adjustment result
        """
        try:
            position = await self.position_tracker.get_position(symbol)
            if not position:
                return {"success": False, "error": "Position not found"}

            results = {}

            # Update stop loss
            if new_stop_loss is not None:
                success = await self.position_tracker.update_stop_loss(position['id'], new_stop_loss)
                results['stop_loss_updated'] = success

            # Update take profit
            if new_take_profit is not None:
                success = await self.position_tracker.update_take_profit(position['id'], new_take_profit)
                results['take_profit_updated'] = success

            # Publish adjustment
            if self._redis:
                await self._redis.publish(f"position_adjustments:{symbol}", json.dumps({
                    "position_id": str(position['id']),
                    "new_stop_loss": str(new_stop_loss) if new_stop_loss else None,
                    "new_take_profit": str(new_take_profit) if new_take_profit else None,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, default=str))

            return {
                "success": True,
                "symbol": symbol,
                "adjustments": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to adjust stops for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    async def get_risk_report(self) -> Dict[str, Any]:
        """
        Get comprehensive risk report.

        Returns:
            Risk analysis report
        """
        try:
            # Get positions at risk
            at_risk_positions = await self.position_tracker.get_positions_at_risk()

            # Get portfolio metrics
            portfolio_metrics = await self.position_tracker.calculate_portfolio_metrics()

            # Get risk metrics
            risk_metrics = await self.performance_tracker.calculate_risk_metrics(days=30)

            # Get account info
            account = await self.alpaca_client.get_account()

            # Calculate risk utilization
            total_equity = Decimal(str(getattr(account, 'equity', 0)))
            position_value = portfolio_metrics.get('total_market_value', Decimal("0"))
            risk_utilization = (position_value / total_equity) if total_equity > 0 else Decimal("0")

            # Get day trading info
            day_trades_count = await self.alpaca_client.get_day_trades_count()
            is_pdt = await self.alpaca_client.is_pattern_day_trader()

            return {
                "at_risk_positions": at_risk_positions,
                "portfolio_metrics": portfolio_metrics,
                "risk_metrics": risk_metrics,
                "risk_utilization": float(risk_utilization),
                "day_trades_count": day_trades_count,
                "is_pattern_day_trader": is_pdt,
                "max_day_trades": 3 if not is_pdt else None,
                "account_equity": float(total_equity),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get risk report: {e}")
            return {"error": str(e)}

    async def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the execution engine as a web service.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        try:
            # Initialize before starting server
            await self.initialize()

            # Configure logging for uvicorn
            import uvicorn.config
            log_config = uvicorn.config.LOGGING_CONFIG.copy()
            log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelprefix)s %(message)s"

            # Start server
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_config=log_config,
                access_log=True
            )

            server = uvicorn.Server(config)
            logger.info(f"Starting Trade Execution Engine on {host}:{port}")

            # Setup graceful shutdown
            async def shutdown_handler():
                logger.info("Shutting down execution engine...")
                await self.cleanup()

            # Add shutdown handler
            import signal

            def signal_handler(signum, frame):
                asyncio.create_task(shutdown_handler())

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Run server
            await server.serve()

        except Exception as e:
            logger.error(f"Failed to run execution engine: {e}")
            raise

    def get_app(self) -> FastAPI:
        """Get FastAPI application instance."""
        return self.app


# Global execution engine instance
execution_engine = ExecutionEngine()


async def main():
    """Main entry point for the execution engine."""
    try:
        await execution_engine.run_server(
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000))
        )
    except KeyboardInterrupt:
        logger.info("Execution engine stopped by user")
    except Exception as e:
        logger.error(f"Execution engine failed: {e}")
        raise


if __name__ == "__main__":
    import os
    asyncio.run(main())
