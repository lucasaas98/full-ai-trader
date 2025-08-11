"""
Trade Execution Service - Main Entry Point.

This module provides the main entry point for the trade execution service,
including Redis signal processing, WebSocket connections, and service lifecycle management.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import json

import aioredis
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.models import TradeSignal
from shared.config import get_config
from execution_engine import ExecutionEngine
# from .alpaca_client import AlpacaClient  # Removed unused import


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/trade_executor.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class TradeExecutorService:
    """
    Main trade execution service with Redis integration and WebSocket support.
    """

    def __init__(self):
        """Initialize the trade executor service."""
        self.config = get_config()
        self.execution_engine = ExecutionEngine()
        self._redis = None
        self._running = False
        self._websocket_connections = set()
        self._signal_processing_stats = {
            'total_processed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'last_signal_time': None
        }

    async def initialize(self):
        """Initialize all service components."""
        try:
            logger.info("Initializing Trade Executor Service...")

            # Initialize Redis connection
            redis_url = getattr(self.config, 'redis', {}).get('url') or 'redis://localhost:6379'
            self._redis = aioredis.from_url(
                redis_url,
                max_connections=30,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )

            # Initialize execution engine
            await self.execution_engine.initialize()

            # Test connections
            await self._test_connections()

            logger.info("Trade Executor Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise

    async def _test_connections(self):
        """Test all external connections."""
        try:
            # Test Redis
            if self._redis:
                await self._redis.ping()
                logger.info("Redis connection successful")
            else:
                logger.warning("Redis connection not initialized")

            # Test Alpaca
            alpaca_health = await self.execution_engine.alpaca_client.health_check()
            if alpaca_health['status'] != 'healthy':
                logger.warning(f"Alpaca connection issues: {alpaca_health}")
            else:
                logger.info("Alpaca connection successful")

            # Test database
            if hasattr(self.execution_engine, '_db_pool') and self.execution_engine._db_pool:
                async with self.execution_engine._db_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                logger.info("Database connection successful")
            else:
                logger.warning("Database connection pool not available")

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise

    async def start_signal_processing(self):
        """Start processing trade signals from Redis."""
        try:
            self._running = True
            logger.info("Starting signal processing...")

            # Subscribe to signal channels
            if self._redis:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe("signals:*")
            else:
                logger.error("Redis connection not available for signal processing")
                return

            logger.info("Subscribed to signals:* channels")

            # Process messages
            async for message in pubsub.listen():
                if not self._running:
                    break

                if message["type"] == "message":
                    try:
                        await self._process_signal_message(message)
                    except Exception as e:
                        logger.error(f"Error processing signal message: {e}")

        except Exception as e:
            logger.error(f"Signal processing error: {e}")
            if self._running:
                # Restart after delay
                await asyncio.sleep(10)
                asyncio.create_task(self.start_signal_processing())

    async def _process_signal_message(self, message):
        """Process individual signal message."""
        try:
            # Parse message
            channel = message["channel"].decode()
            data = json.loads(message["data"])

            logger.info(f"Received signal on channel {channel}: {data.get('symbol', 'unknown')}")

            # Extract symbol from channel if not in data
            if 'symbol' not in data and ':' in channel:
                data['symbol'] = channel.split(':')[-1]

            # Create trade signal
            signal = TradeSignal(**data)

            # Update stats
            self._signal_processing_stats['total_processed'] += 1
            self._signal_processing_stats['last_signal_time'] = datetime.now(timezone.utc)

            # Execute signal
            result = await self.execution_engine.execute_signal(signal)

            # Update execution stats
            if result.get('success'):
                self._signal_processing_stats['successful_executions'] += 1
                logger.info(f"Signal {signal.id} executed successfully")
            else:
                self._signal_processing_stats['failed_executions'] += 1
                logger.warning(f"Signal {signal.id} execution failed: {result.get('error')}")

            # Broadcast result to WebSocket clients
            await self._broadcast_to_websockets({
                'type': 'execution_result',
                'signal_id': str(signal.id),
                'symbol': signal.symbol,
                'result': result,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to process signal message: {e}")
            self._signal_processing_stats['failed_executions'] += 1

    async def start_status_broadcaster(self):
        """Start broadcasting system status updates."""
        try:
            while self._running:
                try:
                    # Get current status
                    status = await self._get_system_status()

                    # Publish to Redis
                    if self._redis:
                        await self._redis.publish("system_status", json.dumps(status, default=str))

                    # Broadcast to WebSocket clients
                    await self._broadcast_to_websockets({
                        'type': 'system_status',
                        'data': status
                    })

                    # Wait 30 seconds
                    await asyncio.sleep(30)

                except Exception as e:
                    logger.error(f"Error broadcasting status: {e}")
                    await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Status broadcaster error: {e}")

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get account summary
            account_summary = await self.execution_engine.get_account_summary()

            # Get active orders count
            active_orders = await self.execution_engine.order_manager.get_active_orders()

            # Get positions count
            positions = await self.execution_engine.position_tracker.get_all_positions()

            # Get performance summary
            performance = await self.execution_engine.performance_tracker.get_performance_summary(days=1)

            return {
                'service': 'trade_executor',
                'status': 'running',
                'timestamp': datetime.now(timezone.utc),
                'signal_processing': self._signal_processing_stats,
                'account': {
                    'equity': account_summary.get('account', {}).get('equity', 0),
                    'buying_power': account_summary.get('account', {}).get('buying_power', 0),
                    'day_trades': account_summary.get('account', {}).get('day_trades_count', 0)
                },
                'positions': {
                    'count': len(positions),
                    'total_value': account_summary.get('positions', {}).get('total_market_value', 0),
                    'unrealized_pnl': account_summary.get('positions', {}).get('total_unrealized_pnl', 0)
                },
                'orders': {
                    'active_count': len(active_orders)
                },
                'performance_today': {
                    'trades': performance.get('total_trades', 0),
                    'pnl': performance.get('total_pnl', 0),
                    'win_rate': performance.get('win_rate', 0)
                },
                'websocket_connections': len(self._websocket_connections)
            }

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'service': 'trade_executor',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc)
            }

    async def _broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections."""
        if not self._websocket_connections:
            return

        # Create a copy of connections to avoid modification during iteration
        connections = self._websocket_connections.copy()

        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                # Remove failed connection
                self._websocket_connections.discard(websocket)

    async def cleanup(self):
        """Clean up all resources."""
        try:
            logger.info("Cleaning up Trade Executor Service...")
            self._running = False

            # Close WebSocket connections
            for ws in self._websocket_connections.copy():
                try:
                    await ws.close()
                except:
                    pass
            self._websocket_connections.clear()

            # Cleanup execution engine
            await self.execution_engine.cleanup()

            # Close Redis connection
            if self._redis:
                await self._redis.close()

            logger.info("Trade Executor Service cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Create FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    service = TradeExecutorService()
    app.state.service = service

    try:
        await service.initialize()

        # Start background tasks
        asyncio.create_task(service.start_signal_processing())
        asyncio.create_task(service.start_status_broadcaster())

        logger.info("Trade Executor Service started")
        yield

    finally:
        # Shutdown
        await service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Trade Execution Service",
    description="Automated trade execution service with Alpaca API integration",
    version="1.0.0",
    lifespan=lifespan
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
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    app.state.service._websocket_connections.add(websocket)

    try:
        logger.info("WebSocket client connected")

        # Send initial status
        status = await app.state.service._get_system_status()
        await websocket.send_json({
            'type': 'connection_established',
            'status': status
        })

        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                message = await websocket.receive_text()

                # Handle client messages if needed
                if message == "ping":
                    await websocket.send_json({'type': 'pong', 'timestamp': datetime.now(timezone.utc).isoformat()})

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        app.state.service._websocket_connections.discard(websocket)
        logger.info("WebSocket client disconnected")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "trade_executor",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if not hasattr(app.state, 'service'):
            return {"status": "initializing"}

        # Get detailed health from execution engine
        health = await app.state.service.execution_engine.alpaca_client.health_check()

        return {
            "status": "healthy" if health.get('status') == 'healthy' else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alpaca": health,
            "signal_processing": app.state.service._signal_processing_stats,
            "websocket_connections": len(app.state.service._websocket_connections)
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    try:
        service = app.state.service

        return {
            "signal_processing": service._signal_processing_stats,
            "websocket_connections": len(service._websocket_connections),
            "uptime_seconds": (datetime.now(timezone.utc) - service._signal_processing_stats.get('start_time', datetime.now(timezone.utc))).total_seconds(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/signals/execute")
async def execute_signal_direct(signal: TradeSignal):
    """Direct signal execution endpoint."""
    try:
        result = await app.state.service.execution_engine.execute_signal(signal)
        return {
            "success": True,
            "signal_id": signal.id,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Direct signal execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@app.post("/emergency/stop")
async def emergency_stop():
    """Emergency stop endpoint."""
    try:
        result = await app.state.service.execution_engine.emergency_stop()
        return result
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/positions")
async def get_positions():
    """Get all positions."""
    try:
        positions = await app.state.service.execution_engine.position_tracker.get_all_positions()
        metrics = await app.state.service.execution_engine.position_tracker.calculate_portfolio_metrics()

        return {
            "positions": positions,
            "metrics": metrics,
            "count": len(positions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return {"error": str(e)}


@app.get("/positions/{symbol}")
async def get_position(symbol: str):
    """Get position for specific symbol."""
    try:
        position = await app.state.service.execution_engine.position_tracker.get_position(symbol)
        if not position:
            return {"error": "Position not found"}
        return position
    except Exception as e:
        logger.error(f"Failed to get position for {symbol}: {e}")
        return {"error": str(e)}


@app.post("/positions/{symbol}/close")
async def close_position(symbol: str, percentage: float = 1.0):
    """Close position for symbol."""
    try:
        result = await app.state.service.execution_engine.force_position_close(
            symbol, f"API close request - {percentage*100}%"
        )
        return result
    except Exception as e:
        logger.error(f"Failed to close position {symbol}: {e}")
        return {"success": False, "error": str(e)}


@app.get("/orders/active")
async def get_active_orders():
    """Get active orders."""
    try:
        orders = await app.state.service.execution_engine.order_manager.get_active_orders()
        return {
            "orders": [order.dict() for order in orders],
            "count": len(orders),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get active orders: {e}")
        return {"error": str(e)}


@app.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel an order."""
    try:
        from uuid import UUID
        success = await app.state.service.execution_engine.order_manager.cancel_order(
            UUID(order_id), "API cancellation request"
        )
        return {
            "success": success,
            "order_id": order_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}")
        return {"success": False, "error": str(e)}


@app.get("/performance/summary")
async def get_performance_summary(days: int = 30):
    """Get performance summary."""
    try:
        summary = await app.state.service.execution_engine.performance_tracker.get_performance_summary(days)
        return summary
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        return {"error": str(e)}


@app.get("/performance/daily")
async def get_daily_performance():
    """Get daily performance."""
    try:
        performance = await app.state.service.execution_engine.performance_tracker.calculate_daily_performance()
        return performance
    except Exception as e:
        logger.error(f"Failed to get daily performance: {e}")
        return {"error": str(e)}


@app.get("/performance/risk")
async def get_risk_report():
    """Get risk analysis report."""
    try:
        risk_report = await app.state.service.execution_engine.get_risk_report()
        return risk_report
    except Exception as e:
        logger.error(f"Failed to get risk report: {e}")
        return {"error": str(e)}


@app.get("/export/tradenote")
async def export_tradenote(start_date: Optional[str] = None, end_date: Optional[str] = None, strategy: Optional[str] = None):
    """Export trades for TradeNote."""
    try:
        from datetime import date, timezone

        start_dt = date.fromisoformat(start_date) if start_date else datetime.now().date()
        end_dt = date.fromisoformat(end_date) if end_date else datetime.now().date()

        trades = await app.state.service.execution_engine.performance_tracker.export_for_tradenote(
            start_date=start_dt,
            end_date=end_dt,
            strategy_name=strategy or "default"
        )

        return {
            "trades": trades,
            "count": len(trades),
            "export_params": {
                "start_date": start_date,
                "end_date": end_date,
                "strategy": strategy
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to export TradeNote data: {e}")
        return {"error": str(e)}


@app.get("/account")
async def get_account():
    """Get account information."""
    try:
        account_info = await app.state.service.execution_engine.get_account_summary()
        return account_info
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        return {"error": str(e)}


@app.post("/sync/positions")
async def sync_positions():
    """Sync positions with Alpaca."""
    try:
        result = await app.state.service.execution_engine.position_tracker.sync_with_alpaca()
        return result
    except Exception as e:
        logger.error(f"Failed to sync positions: {e}")
        return {"success": False, "error": str(e)}


@app.get("/metrics/execution")
async def get_execution_metrics(symbol: Optional[str] = None, days: int = 30):
    """Get execution quality metrics."""
    try:
        metrics = await app.state.service.execution_engine.get_execution_metrics(symbol or "SPY", days)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get execution metrics: {e}")
        return {"error": str(e)}


def setup_signal_handlers(service: TradeExecutorService):
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(service.cleanup())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    service = None
    try:
        # Create service instance
        service = TradeExecutorService()

        # Setup signal handlers
        setup_signal_handlers(service)

        # Initialize service
        await service.initialize()

        # Start background tasks
        await asyncio.gather(
            service.start_signal_processing(),
            service.start_status_broadcaster(),
            return_exceptions=True
        )

        # Get port from environment or config
        port = int(os.getenv("PORT", service.config.service_port))
        host = os.getenv("HOST", "0.0.0.0")

        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            loop="asyncio"
        )

        server = uvicorn.Server(uvicorn_config)

        logger.info(f"Starting Trade Executor Service on {host}:{port}")
        logger.info(f"Paper trading: {service.config.alpaca.paper_trading}")
        logger.info(f"Environment: {service.config.environment}")

        # Run server
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            if service:
                await service.cleanup()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("data/logs", exist_ok=True)

    # Set service name for logging
    os.environ.setdefault("SERVICE_NAME", "trade_executor")

    # Run the service
    asyncio.run(main())
