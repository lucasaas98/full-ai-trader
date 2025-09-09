"""
Trade Execution Service - Main Entry Point.

This module provides the main entry point for the trade execution service,
including Redis signal processing, WebSocket connections, and service lifecycle management.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis.asyncio as redis
import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, generate_latest

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pathlib import Path  # noqa: E402

from shared.config import get_config  # noqa: E402
from shared.models import TradeSignal  # noqa: E402

from .execution_engine import ExecutionEngine  # noqa: E402

# from .alpaca_client import AlpacaClient  # Removed unused import


log_file_path = os.getenv("LOG_FILE_PATH", "data/logs/trade_executor.log")
log_path = Path(log_file_path).parent
log_path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path, mode="a")],
)

logger = logging.getLogger(__name__)


class TradeExecutorService:
    """
    Main trade execution service with Redis integration and WebSocket support.
    """

    def __init__(self) -> None:
        """Initialize the trade executor service."""
        logger.debug("Initializing TradeExecutorService instance")
        self.config = get_config()
        logger.debug(
            f"Config loaded: environment={getattr(self.config, 'environment', 'unknown')}"
        )
        self.execution_engine = ExecutionEngine()
        logger.debug("ExecutionEngine instance created")
        self._redis = None
        self._running = False
        self._websocket_connections: set = set()
        self._signal_processing_stats = {
            "total_processed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "last_signal_time": None,
        }
        self._screener_watchlist: set = set()
        logger.debug("TradeExecutorService initialization completed")

    async def initialize(self) -> None:
        """Initialize all service components."""
        try:
            logger.info("Initializing Trade Executor Service...")
            logger.debug("Starting component initialization process")

            # Initialize Redis connection
            redis_url = (
                self.config.redis.url
                if hasattr(self.config, "redis")
                else "redis://localhost:6379"
            )
            logger.debug(f"Redis URL: {redis_url}")
            self._redis = redis.from_url(
                redis_url, max_connections=30, retry_on_timeout=True
            )
            logger.debug("Redis connection object created")

            # Initialize execution engine
            logger.debug("Initializing execution engine...")
            await self.execution_engine.initialize()
            logger.debug("Execution engine initialized successfully")

            # Test connections
            logger.debug("Testing all connections...")
            await self._test_connections()
            logger.debug("Connection tests completed")

            logger.info("Trade Executor Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service: {e}", exc_info=True)
            raise

    async def _test_connections(self) -> None:
        """Test all external connections."""
        try:
            logger.debug("Starting connection tests")

            # Test Redis
            if self._redis:
                logger.debug("Testing Redis connection...")
                ping_result = await self._redis.ping()
                logger.debug(f"Redis ping result: {ping_result}")
                logger.info("Redis connection successful")
            else:
                logger.warning("Redis connection not initialized")

            # Test Alpaca
            logger.debug("Testing Alpaca connection...")
            alpaca_health = await self.execution_engine.alpaca_client.health_check()
            logger.debug(f"Alpaca health check result: {alpaca_health}")
            if alpaca_health["status"] != "healthy":
                logger.warning(f"Alpaca connection issues: {alpaca_health}")
            else:
                logger.info("Alpaca connection successful")

            # Test database
            logger.debug("Testing database connection...")
            if (
                hasattr(self.execution_engine, "_db_pool")
                and self.execution_engine._db_pool
            ):
                async with self.execution_engine._db_pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1")
                    logger.debug(f"Database test query result: {result}")
                logger.info("Database connection successful")
            else:
                logger.warning("Database connection pool not available")

            logger.debug("All connection tests completed")

        except Exception as e:
            logger.error(f"Connection test failed: {e}", exc_info=True)
            raise

    async def _setup_signal_processing(self) -> None:
        """Start processing trade signals from Redis."""
        try:
            self._running = True
            logger.info("Starting signal processing...")
            logger.debug(f"Signal processing running flag set to: {self._running}")

            # Subscribe to signal channels and screener updates
            if self._redis:
                logger.debug("Creating Redis pubsub connection")
                pubsub = self._redis.pubsub()
                logger.debug("Subscribing to channels...")
                await pubsub.subscribe("signals:*")
                await pubsub.subscribe("screener:updates")
                logger.debug("Channel subscriptions completed")
            else:
                logger.error("Redis connection not available for signal processing")
                return

            logger.info("Subscribed to signals:* and screener:updates channels")

            # Process messages
            logger.debug("Starting message processing loop")
            async for message in pubsub.listen():
                logger.debug(
                    f"Received message: type={message.get('type')}, channel={message.get('channel')}"
                )

                if not self._running:
                    logger.debug(
                        "Signal processing stopped, breaking from message loop"
                    )
                    break

                if message["type"] == "message":
                    try:
                        channel = (
                            message["channel"].decode()
                            if isinstance(message["channel"], bytes)
                            else message["channel"]
                        )
                        logger.debug(f"Processing message on channel: {channel}")

                        if channel.startswith("signals:"):
                            logger.debug("Routing to signal message processor")
                            await self._process_signal_message(message)
                        elif channel == "screener:updates":
                            logger.debug("Routing to screener message processor")
                            await self._process_screener_message(message)
                        else:
                            logger.debug(f"Unknown channel type: {channel}")
                    except Exception as e:
                        logger.error(f"Error processing signal: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Signal processing error: {e}", exc_info=True)
            if self._running:
                logger.debug("Restarting signal processing after error delay")
                # Restart after delay
                await asyncio.sleep(10)
                asyncio.create_task(self.start_signal_processing())

    async def _process_screener_message(self, message: Dict) -> None:
        """
        Process screener update message.

        Args:
            message: Redis message with screener data
        """
        try:
            logger.debug(f"Processing screener message: {message}")

            # Parse screener data
            data = json.loads(message["data"])
            screener_type = data.get("screener_type", "unknown")
            stocks_data = data.get("data", [])

            logger.debug(
                f"Parsed screener data: type={screener_type}, stocks_count={len(stocks_data)}"
            )
            logger.info(
                f"Processing screener update: {screener_type} with {len(stocks_data)} stocks"
            )

            # Update watchlist with new symbols
            new_symbols = set()
            for stock in stocks_data:
                symbol = stock.get("symbol")
                if symbol:
                    new_symbols.add(symbol)
                    logger.debug(f"Added symbol to watchlist: {symbol}")

            logger.debug(f"New symbols extracted: {new_symbols}")

            # Update internal watchlist
            previous_count = len(self._screener_watchlist)
            self._screener_watchlist.update(new_symbols)

            logger.debug(
                f"Internal watchlist size: {previous_count} -> {len(self._screener_watchlist)}"
            )
            logger.info(
                f"Watchlist updated: {previous_count} -> {len(self._screener_watchlist)} symbols"
            )

            # Notify execution engine about new symbols
            if self.execution_engine and hasattr(
                self.execution_engine, "update_watchlist"
            ):
                logger.debug("Notifying execution engine about watchlist update")
                await self.execution_engine.update_watchlist(list(new_symbols))
                logger.debug("Execution engine watchlist update completed")
            else:
                logger.debug("Execution engine watchlist update not available")

        except Exception as e:
            logger.error(f"Error processing screener message: {e}", exc_info=True)

    async def _process_signal_message(self, message) -> None:
        """Process individual signal message."""
        try:
            logger.debug(f"Processing signal message: {message}")

            # Parse message
            channel = message["channel"].decode()
            data = json.loads(message["data"])

            logger.debug(f"Parsed signal data: channel={channel}, data={data}")
            logger.info(
                f"Received signal on channel {channel}: {data.get('symbol', 'unknown')}"
            )

            # Extract symbol from channel if not in data
            if "symbol" not in data and ":" in channel:
                extracted_symbol = channel.split(":")[-1]
                data["symbol"] = extracted_symbol
                logger.debug(f"Extracted symbol from channel: {extracted_symbol}")

            # Create trade signal
            logger.debug("Creating TradeSignal object")
            signal = TradeSignal(**data)
            logger.debug(
                f"TradeSignal created: id={signal.id}, symbol={signal.symbol}, signal_type={signal.signal_type}"
            )

            # Update stats
            self._signal_processing_stats["total_processed"] = (
                self._signal_processing_stats.get("total_processed") or 0
            ) + 1
            self._signal_processing_stats["last_signal_time"] = int(
                datetime.now(timezone.utc).timestamp()
            )
            logger.debug(
                f"Updated stats: total_processed={self._signal_processing_stats['total_processed']}"
            )

            # Execute signal
            logger.debug(f"Executing signal {signal.id}")
            # Convert TradeSignal to dict for execute_signal
            signal_dict = {
                "symbol": signal.symbol,
                "action": signal.signal_type.value,
                "confidence": signal.confidence,
                "quantity": signal.quantity,
                "price": float(signal.price) if signal.price else None,
                "strategy": signal.strategy_name,
                "metadata": signal.metadata,
            }
            result = await self.execution_engine.execute_signal(signal_dict)
            logger.debug(f"Signal execution result: {result}")

            # Update execution stats
            if result.get("success"):
                self._signal_processing_stats["successful_executions"] = (
                    self._signal_processing_stats.get("successful_executions") or 0
                ) + 1

                logger.info(f"Signal {signal.id} executed successfully")
                logger.debug(
                    f"Successful executions: {self._signal_processing_stats['successful_executions']}"
                )
            else:
                self._signal_processing_stats["failed_executions"] = (
                    self._signal_processing_stats.get("failed_executions") or 0
                ) + 1
                logger.warning(
                    f"Signal {signal.id} execution failed: {result.get('error')}"
                )
                logger.debug(
                    f"Failed executions: {self._signal_processing_stats['failed_executions']}"
                )

            # Broadcast result to WebSocket clients
            broadcast_message = {
                "type": "execution_result",
                "signal_id": str(signal.id),
                "symbol": signal.symbol,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.debug(f"Broadcasting to WebSocket clients: {broadcast_message}")
            await self._broadcast_to_websockets(broadcast_message)

        except Exception as e:
            logger.error(f"Failed to process signal message: {e}", exc_info=True)
            self._signal_processing_stats["failed_executions"] = (
                self._signal_processing_stats.get("failed_executions") or 0
            ) + 1

    async def start_status_broadcaster(self) -> None:
        """Start broadcasting system status updates."""
        try:
            logger.debug("Starting status broadcaster")
            while self._running:
                try:
                    logger.debug("Getting system status for broadcast")
                    # Get current status
                    status = await self._get_system_status()
                    logger.debug(f"System status retrieved: {status}")

                    # Publish to Redis
                    if self._redis:
                        logger.debug("Publishing status to Redis")
                        await self._redis.publish(
                            "system_status", json.dumps(status, default=str)
                        )
                        logger.debug("Status published to Redis successfully")

                    # Broadcast to WebSocket clients
                    broadcast_data = {"type": "system_status", "data": status}
                    logger.debug(
                        f"Broadcasting status to {len(self._websocket_connections)} WebSocket clients"
                    )
                    await self._broadcast_to_websockets(broadcast_data)

                    # Wait 30 seconds
                    logger.debug("Status broadcast complete, waiting 30 seconds")
                    await asyncio.sleep(30)

                except Exception as e:
                    logger.error(f"Error broadcasting status: {e}", exc_info=True)
                    logger.debug(
                        "Status broadcast error, waiting 60 seconds before retry"
                    )
                    await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Status broadcaster error: {e}", exc_info=True)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            logger.debug("Starting system status collection")

            # Get account summary
            logger.debug("Getting account summary")
            account_summary = await self.execution_engine.get_account_summary()
            logger.debug(f"Account summary retrieved: {account_summary}")

            # Get active orders count
            logger.debug("Getting active orders")
            active_orders = (
                await self.execution_engine.order_manager.get_active_orders()
            )
            logger.debug(f"Active orders count: {len(active_orders)}")

            # Get positions count
            logger.debug("Getting positions")
            positions = await self.execution_engine.position_tracker.get_all_positions()
            logger.debug(f"Positions count: {len(positions)}")

            # Get performance summary
            logger.debug("Getting performance summary")
            performance = (
                await self.execution_engine.performance_tracker.get_performance_summary(
                    days=1
                )
            )
            logger.debug(f"Performance summary: {performance}")

            status = {
                "service": "trade_executor",
                "status": "running",
                "timestamp": datetime.now(timezone.utc),
                "signal_processing": self._signal_processing_stats,
                "account": {
                    "equity": account_summary.get("account", {}).get("equity", 0),
                    "buying_power": account_summary.get("account", {}).get(
                        "buying_power", 0
                    ),
                    "day_trades": account_summary.get("account", {}).get(
                        "day_trades_count", 0
                    ),
                },
                "positions": {
                    "count": len(positions),
                    "total_value": account_summary.get("positions", {}).get(
                        "total_market_value", 0
                    ),
                    "unrealized_pnl": account_summary.get("positions", {}).get(
                        "total_unrealized_pnl", 0
                    ),
                },
                "orders": {"active_count": len(active_orders)},
                "performance_today": {
                    "trades": performance.get("total_trades", 0),
                    "pnl": performance.get("total_pnl", 0),
                    "win_rate": performance.get("win_rate", 0),
                },
                "websocket_connections": len(self._websocket_connections),
            }

            logger.debug("System status collection completed successfully")
            return status

        except Exception as e:
            logger.error(f"Failed to get system status: {e}", exc_info=True)
            return {
                "service": "trade_executor",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc),
            }

    async def _broadcast_to_websockets(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all WebSocket connections."""
        if not self._websocket_connections:
            logger.debug("No WebSocket connections to broadcast to")
            return

        logger.debug(
            f"Broadcasting to {len(self._websocket_connections)} WebSocket connections"
        )

        # Create a copy of connections to avoid modification during iteration
        connections = self._websocket_connections.copy()

        for websocket in connections:
            try:
                logger.debug("Sending message to WebSocket connection")
                await websocket.send_json(message)
                logger.debug("WebSocket message sent successfully")
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                logger.debug("Removing failed WebSocket connection")
                # Remove failed connection
                self._websocket_connections.discard(websocket)

    async def cleanup(self) -> None:
        """Clean up all resources."""
        try:
            logger.info("Cleaning up Trade Executor Service...")
            logger.debug("Setting running flag to False")
            self._running = False

            # Close WebSocket connections
            logger.debug(
                f"Closing {len(self._websocket_connections)} WebSocket connections"
            )
            for ws in self._websocket_connections.copy():
                try:
                    logger.debug("Closing WebSocket connection")
                    await ws.close()
                except Exception as ws_error:
                    logger.debug(f"Error closing WebSocket: {ws_error}")
                    pass
            self._websocket_connections.clear()
            logger.debug("All WebSocket connections closed")

            # Cleanup execution engine
            logger.debug("Cleaning up execution engine")
            await self.execution_engine.cleanup()
            logger.debug("Execution engine cleanup completed")

            # Close Redis connection
            if self._redis:
                logger.debug("Closing Redis connection")
                await self._redis.close()
                logger.debug("Redis connection closed")

            logger.info("Trade Executor Service cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)


# Create FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.debug("Starting application lifespan management")
    service = TradeExecutorService()
    app.state.service = service

    try:
        logger.debug("Initializing service in lifespan")
        await service.initialize()

        # Start background tasks
        logger.debug("Starting background tasks")
        task1 = asyncio.create_task(service.start_signal_processing())
        task2 = asyncio.create_task(service.start_status_broadcaster())
        logger.debug(
            f"Background tasks created: signal_processing={task1}, status_broadcaster={task2}"
        )

        logger.info("Trade Executor Service started")
        yield

    finally:
        # Shutdown
        logger.debug("Shutting down application lifespan")
        await service.cleanup()
        logger.debug("Application lifespan shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Trade Execution Service",
    description="Automated trade execution service with Alpaca API integration",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    logger.debug("WebSocket connection attempt received")
    await websocket.accept()
    logger.debug("WebSocket connection accepted")
    app.state.service._websocket_connections.add(websocket)
    logger.debug(
        f"WebSocket connection added to pool. Total connections: {len(app.state.service._websocket_connections)}"
    )

    try:
        logger.info("WebSocket client connected")

        # Send initial status
        logger.debug("Getting initial system status for WebSocket client")
        status = await app.state.service._get_system_status()
        logger.debug(f"Initial status retrieved: {status}")

        initial_message = {"type": "connection_established", "status": status}
        logger.debug(f"Sending initial message to WebSocket client: {initial_message}")
        await websocket.send_json(initial_message)
        logger.debug("Initial status sent to WebSocket client")

        # Keep connection alive
        logger.debug("Starting WebSocket message loop")
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                logger.debug("Waiting for WebSocket message from client")
                message = await websocket.receive_text()
                logger.debug(f"Received WebSocket message from client: {message}")

                # Handle client messages if needed
                if message == "ping":
                    logger.debug("Received ping from WebSocket client, sending pong")
                    pong_message = {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await websocket.send_json(pong_message)
                    logger.debug(f"Sent pong to WebSocket client: {pong_message}")
                else:
                    logger.debug(f"Unhandled WebSocket message: {message}")

            except WebSocketDisconnect:
                logger.debug("WebSocket client disconnected normally")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                logger.debug("Breaking from WebSocket message loop due to error")
                break

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
    finally:
        logger.debug("Removing WebSocket connection from pool")
        app.state.service._websocket_connections.discard(websocket)
        logger.debug(
            f"WebSocket connection removed. Remaining connections: {len(app.state.service._websocket_connections)}"
        )
        logger.info("WebSocket client disconnected")


@app.get("/")
async def root():
    """Root endpoint."""
    logger.debug("Root endpoint accessed")
    response = {
        "service": "trade_executor",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }
    logger.debug(f"Root endpoint response: {response}")
    return response


@app.get("/health")
async def healthcheck() -> Dict[str, Any]:
    """Health check endpoint."""
    logger.debug("Health check endpoint accessed")
    try:
        if not hasattr(app.state, "service"):
            logger.debug("Service not initialized, returning initializing status")
            return {"status": "initializing"}

        # Get detailed health from execution engine
        logger.debug("Getting health check from execution engine")
        health = await app.state.service.execution_engine.alpaca_client.health_check()
        logger.debug(f"Alpaca health check result: {health}")

        response = {
            "status": "healthy" if health.get("status") == "healthy" else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alpaca": health,
            "signal_processing": app.state.service._signal_processing_stats,
            "websocket_connections": len(app.state.service._websocket_connections),
        }
        logger.debug(f"Health check response: {response}")
        return response

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        error_response = {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(f"Health check error response: {error_response}")
        return error_response


@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    logger.debug("Stats endpoint accessed")
    try:
        service = app.state.service
        logger.debug("Getting service statistics")

        stats = {
            "signal_processing": service._signal_processing_stats,
            "websocket_connections": len(service._websocket_connections),
            "uptime_seconds": (
                datetime.now(timezone.utc)
                - service._signal_processing_stats.get(
                    "start_time", datetime.now(timezone.utc)
                )
            ).total_seconds(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(f"Service statistics: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/signals/execute")
async def execute_signal_direct(signal: TradeSignal):
    """Direct signal execution endpoint."""
    logger.debug(f"Direct signal execution requested: {signal}")
    try:
        logger.debug(
            f"Executing signal directly: id={signal.id}, symbol={signal.symbol}"
        )
        result = await app.state.service.execution_engine.execute_signal(signal)
        logger.debug(f"Direct signal execution result: {result}")

        response = {
            "success": True,
            "signal_id": signal.id,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(f"Direct execution response: {response}")
        return response
    except Exception as e:
        logger.error(f"Direct signal execution failed: {e}", exc_info=True)
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(f"Direct execution error response: {error_response}")
        return error_response


@app.post("/emergency/stop")
async def emergency_stop() -> Dict[str, str]:
    """Emergency stop endpoint."""
    logger.debug("Emergency stop endpoint accessed")
    logger.warning("Emergency stop initiated via API")
    try:
        logger.debug("Calling execution engine emergency stop")
        result = await app.state.service.execution_engine.emergency_stop()
        logger.debug(f"Emergency stop result: {result}")
        return result
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}", exc_info=True)
        error_result = {"success": False, "error": str(e)}
        logger.debug(f"Emergency stop error result: {error_result}")
        return error_result


@app.get("/positions")
async def get_positions() -> Dict[str, Any]:
    """Get all positions."""
    logger.debug("Get positions endpoint accessed")
    try:
        logger.debug("Fetching all positions from position tracker")
        positions = (
            await app.state.service.execution_engine.position_tracker.get_all_positions()
        )
        logger.debug(f"Retrieved {len(positions)} positions")

        logger.debug("Calculating portfolio metrics")
        metrics = (
            await app.state.service.execution_engine.position_tracker.calculate_portfolio_metrics()
        )
        logger.debug(f"Portfolio metrics calculated: {metrics}")

        response = {
            "positions": positions,
            "metrics": metrics,
            "count": len(positions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(f"Positions endpoint response: positions_count={len(positions)}")
        return response
    except Exception as e:
        logger.error(f"Failed to get positions: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/positions/{symbol}")
async def get_position(symbol: str):
    """Get position for specific symbol."""
    logger.debug(f"Get position endpoint accessed for symbol: {symbol}")
    try:
        logger.debug(f"Fetching position for symbol: {symbol}")
        position = (
            await app.state.service.execution_engine.position_tracker.get_position(
                symbol
            )
        )
        logger.debug(f"Position data for {symbol}: {position}")
        if not position:
            logger.debug(f"No position found for symbol: {symbol}")
            return {"error": "Position not found"}
        logger.debug(f"Returning position data for {symbol}")
        return position
    except Exception as e:
        logger.error(f"Failed to get position for {symbol}: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/positions/{symbol}/close")
async def close_position(symbol: str, percentage: float = 1.0):
    """Close position for symbol."""
    logger.debug(
        f"Close position endpoint accessed for symbol: {symbol}, percentage: {percentage}"
    )
    try:
        logger.debug(f"Initiating position close for {symbol} at {percentage * 100}%")
        result = await app.state.service.execution_engine.force_position_close(
            symbol, f"API close request - {percentage * 100}%"
        )
        logger.debug(f"Position close result for {symbol}: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to close position {symbol}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.get("/orders/active")
async def get_active_orders() -> Dict[str, Any]:
    """Get active orders."""
    logger.debug("Get active orders endpoint accessed")
    try:
        logger.debug("Fetching active orders from order manager")
        orders = (
            await app.state.service.execution_engine.order_manager.get_active_orders()
        )
        logger.debug(f"Retrieved {len(orders)} active orders")

        response = {
            "orders": [order.dict() for order in orders],
            "count": len(orders),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(f"Active orders response: count={len(orders)}")
        return response
    except Exception as e:
        logger.error(f"Failed to get active orders: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel an order."""
    logger.debug(f"Cancel order endpoint accessed for order_id: {order_id}")
    try:
        from uuid import UUID

        logger.debug(f"Converting order_id to UUID and cancelling: {order_id}")
        success = await app.state.service.execution_engine.order_manager.cancel_order(
            UUID(order_id), "API cancellation request"
        )
        logger.debug(f"Order cancellation result for {order_id}: success={success}")

        response = {
            "success": success,
            "order_id": order_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return response
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.get("/performance/summary")
async def get_performance_summary(days: int = 30) -> Dict[str, Any]:
    """Get performance summary."""
    logger.debug(f"Get performance summary endpoint accessed for {days} days")
    try:
        logger.debug(f"Fetching performance summary for last {days} days")
        summary = await app.state.service.execution_engine.performance_tracker.get_performance_summary(
            days
        )
        logger.debug(f"Performance summary retrieved: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/performance/daily")
async def get_daily_performance() -> Dict[str, Any]:
    """Get daily performance."""
    logger.debug("Get daily performance endpoint accessed")
    try:
        logger.debug("Calculating daily performance")
        performance = (
            await app.state.service.execution_engine.performance_tracker.calculate_daily_performance()
        )
        logger.debug(f"Daily performance calculated: {performance}")
        return performance
    except Exception as e:
        logger.error(f"Failed to get daily performance: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/performance/risk")
async def get_risk_report():
    """Get risk analysis report."""
    logger.debug("Get risk report endpoint accessed")
    try:
        logger.debug("Generating risk analysis report")
        risk_report = await app.state.service.execution_engine.get_risk_report()
        logger.debug(f"Risk report generated: {risk_report}")
        return risk_report
    except Exception as e:
        logger.error(f"Failed to get risk report: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/export/tradenote")
async def export_tradenote(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Export trades for TradeNote."""
    logger.debug(
        f"TradeNote export endpoint accessed: start_date={start_date}, end_date={end_date}, strategy={strategy}"
    )
    try:
        from datetime import date, timezone

        logger.debug("Converting date parameters")
        start_dt = (
            date.fromisoformat(start_date) if start_date else datetime.now().date()
        )
        end_dt = date.fromisoformat(end_date) if end_date else datetime.now().date()
        logger.debug(f"Date range: {start_dt} to {end_dt}")

        logger.debug(
            f"Exporting trades for TradeNote with strategy: {strategy or 'default'}"
        )
        trades = await app.state.service.execution_engine.performance_tracker.export_for_tradenote(
            start_date=start_dt, end_date=end_dt, strategy_name=strategy or "default"
        )
        logger.debug(f"Exported {len(trades)} trades for TradeNote")

        response = {
            "trades": trades,
            "count": len(trades),
            "export_params": {
                "start_date": start_date,
                "end_date": end_date,
                "strategy": strategy,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return response
    except Exception as e:
        logger.error(f"Failed to export TradeNote data: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/account")
async def get_account() -> Dict[str, Any]:
    """Get account information."""
    logger.debug("Get account endpoint accessed")
    try:
        logger.debug("Fetching account information from execution engine")
        account_info = await app.state.service.execution_engine.get_account_summary()
        logger.debug(f"Account info retrieved: {account_info}")
        return account_info
    except Exception as e:
        logger.error(f"Failed to get account info: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/sync/positions")
async def sync_positions() -> Dict[str, Any]:
    """Sync positions with Alpaca."""
    logger.debug("Sync positions endpoint accessed")
    try:
        logger.debug("Initiating position sync with Alpaca")
        result = (
            await app.state.service.execution_engine.position_tracker.sync_with_alpaca()
        )
        logger.debug(f"Position sync result: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to sync positions: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.get("/metrics/execution")
async def get_execution_metrics(
    symbol: Optional[str] = None, days: int = 30
) -> Dict[str, Any]:
    """Get execution quality metrics."""
    logger.debug(
        f"Get execution metrics endpoint accessed: symbol={symbol}, days={days}"
    )
    try:
        target_symbol = symbol or "SPY"
        logger.debug(f"Fetching execution metrics for {target_symbol} over {days} days")
        metrics = await app.state.service.execution_engine.get_execution_metrics(
            target_symbol, days
        )
        logger.debug(f"Execution metrics retrieved: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to get execution metrics: {e}", exc_info=True)
        return {"error": str(e)}


# Prometheus metrics (initialized lazily to avoid registration conflicts)
orders_executed_counter = None
orders_failed_counter = None
execution_latency_gauge = None
service_health_gauge = None


def _initialize_metrics() -> None:
    """Initialize Prometheus metrics if not already done."""
    global orders_executed_counter, orders_failed_counter, execution_latency_gauge, service_health_gauge

    if orders_executed_counter is None:
        orders_executed_counter = Counter(
            "trade_executor_orders_executed_total",
            "Total orders executed",
            ["symbol", "side"],
        )
        orders_failed_counter = Counter(
            "trade_executor_orders_failed_total",
            "Total orders failed",
            ["symbol", "reason"],
        )
        execution_latency_gauge = Gauge(
            "trade_executor_execution_latency_seconds", "Order execution latency"
        )
        service_health_gauge = Gauge(
            "trade_executor_service_health",
            "Health status of components",
            ["component"],
        )


@app.get("/metrics")
async def get_prometheus_metrics() -> Response:
    """Prometheus metrics endpoint."""
    logger.debug("Prometheus metrics endpoint accessed")
    try:
        # Initialize metrics if not already done
        logger.debug("Initializing Prometheus metrics")
        _initialize_metrics()

        # Update metrics before returning them
        logger.debug("Updating service health metrics")
        if service_health_gauge:
            service_health_gauge.labels(component="alpaca").set(
                1 if hasattr(app.state, "service") else 0
            )
            service_health_gauge.labels(component="service").set(1)

        # Generate Prometheus format
        logger.debug("Generating Prometheus metrics output")
        metrics_output = generate_latest()
        logger.debug(f"Prometheus metrics generated, size: {len(metrics_output)} bytes")
        return Response(
            content=metrics_output,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    except Exception as e:
        logger.error(f"Error generating metrics: {e}", exc_info=True)
        return Response(content="", media_type="text/plain")


def setup_signal_handlers(service: TradeExecutorService) -> None:
    """Setup graceful shutdown signal handlers."""
    logger.debug("Setting up signal handlers for graceful shutdown")

    def signal_handler(signum, frame) -> None:
        logger.info(f"Received signal {signum}, initiating shutdown...")
        logger.debug(f"Signal handler triggered: signum={signum}, frame={frame}")
        asyncio.create_task(service.cleanup())
        sys.exit(0)

    logger.debug("Registering SIGINT and SIGTERM handlers")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.debug("Signal handlers registered successfully")


async def main() -> None:
    """Main entry point."""
    logger.debug("Starting main function")
    service = None
    try:
        # Create service instance
        logger.debug("Creating TradeExecutorService instance")
        service = TradeExecutorService()
        logger.debug("Service instance created")

        # Setup signal handlers
        logger.debug("Setting up signal handlers")
        setup_signal_handlers(service)
        logger.debug("Signal handlers configured")

        # Initialize service
        logger.debug("Initializing service")
        await service.initialize()
        logger.debug("Service initialization completed")

        # Start background tasks with error handling
        try:
            logger.debug("Starting background tasks")
            task1 = asyncio.create_task(service.start_signal_processing())
            task2 = asyncio.create_task(service.start_status_broadcaster())
            logger.info("Background tasks started")
            logger.debug(
                f"Background task references: signal_processing={task1}, status_broadcaster={task2}"
            )

            # Give background tasks a moment to initialize
            logger.debug("Waiting 1 second for background tasks to initialize")
            await asyncio.sleep(1)
            logger.debug("Background task initialization wait completed")
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}", exc_info=True)

        # Get port from environment or config
        port = int(os.getenv("SERVICE_PORT", service.config.service_port))
        host = os.getenv("HOST", "0.0.0.0")
        logger.debug(f"Server configuration: host={host}, port={port}")

        # Configure uvicorn
        logger.debug("Configuring uvicorn server")
        uvicorn_config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            loop="asyncio",
        )

        server = uvicorn.Server(uvicorn_config)
        logger.debug("Uvicorn server configured")

        logger.info(f"Starting Trade Executor Service on {host}:{port}")
        logger.info(f"Paper trading: {service.config.alpaca.paper_trading}")
        logger.info(f"Environment: {service.config.environment}")
        logger.debug("Starting uvicorn server")

        # Run server
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        logger.debug("KeyboardInterrupt received")
    except Exception as e:
        logger.error(f"Service failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        logger.debug("Starting cleanup process")
        try:
            if service:
                logger.debug("Cleaning up service")
                await service.cleanup()
                logger.debug("Service cleanup completed")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}", exc_info=True)


class TradeExecutorApp:
    """Application wrapper for Trade Executor service for integration testing."""

    def __init__(self):
        """Initialize the Trade Executor application."""
        self.service = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the application."""
        if not self._initialized:
            self.service = TradeExecutorService()
            await self.service.initialize()
            self._initialized = True

    async def start(self) -> None:
        """Start the Trade Executor service."""
        await self.initialize()
        # Start background tasks
        if self.service:
            asyncio.create_task(self.service.start_signal_processing())
            asyncio.create_task(self.service.start_status_broadcaster())

    async def stop(self) -> None:
        """Stop the Trade Executor service."""
        if self.service:
            await self.service.cleanup()
        self._initialized = False

    def get_service(self) -> TradeExecutorService:
        """Get the underlying service instance."""
        return self.service


if __name__ == "__main__":
    # Ensure required directories exist
    logger.debug("Creating required directories")
    os.makedirs("data/logs", exist_ok=True)
    logger.debug("Required directories created")

    # Set service name for logging
    logger.debug("Setting service name environment variable")
    os.environ.setdefault("SERVICE_NAME", "trade_executor")
    logger.debug(f"Service name set to: {os.environ.get('SERVICE_NAME')}")

    # Run the service
    logger.debug("Starting asyncio main loop")
    asyncio.run(main())
