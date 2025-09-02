"""
Notification Service for Trading System.

This service subscribes to Redis events from various trading components
and sends notifications through Gotify and other channels.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

import redis.asyncio as redis
from redis.asyncio.client import PubSub

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from monitoring.gotify_client import NotificationManager
from shared.config import Config, get_config
from shared.models import Notification

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NotificationService:
    """
    Service that monitors Redis events and sends notifications.

    Subscribes to various Redis channels for trade executions, errors,
    and other events, then sends appropriate notifications through Gotify.
    """

    def __init__(self):
        """Initialize the notification service."""
        self.config: Config = get_config()
        self.notification_manager: Optional[NotificationManager] = None
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[PubSub] = None
        self._running = False
        self._tasks: Set[asyncio.Task] = set()

        # Channels to subscribe to
        self.channels = [
            "executions:all",  # All trade executions
            "execution_errors:all",  # All execution errors
            "signals:all",  # Trading signals (optional)
            "alerts:*",  # All system alerts
            "risk:alerts",  # Risk management alerts
            "portfolio:updates",  # Portfolio updates
            "system:status",  # System status updates
            "market:alerts",  # Market condition alerts
            "screener:updates",  # Screener data updates
        ]

        # Track notification rate limiting
        self._last_notifications: Dict[str, datetime] = {}
        self._notification_cooldown = 60  # seconds between similar notifications

    async def initialize(self):
        """Initialize all service components."""
        try:
            logger.info("Initializing Notification Service...")

            # Initialize Redis connection using environment variable
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
            logger.info(f"Connecting to Redis at: {redis_url}")

            self._redis = await redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
            )

            # Test Redis connection
            await self._redis.ping()
            logger.info("Redis connection established")

            # Initialize notification manager
            self.notification_manager = NotificationManager(self.config)
            await self.notification_manager.startup()
            logger.info("Notification manager initialized")

            # Create pubsub instance
            self._pubsub = self._redis.pubsub()

            # Subscribe to channels
            await self._subscribe_to_channels()

            logger.info("Notification Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise

    async def _subscribe_to_channels(self):
        """Subscribe to Redis channels."""
        try:
            if not self._pubsub:
                raise ValueError("PubSub not initialized")

            for channel in self.channels:
                if "*" in channel:
                    # Pattern subscription
                    await self._pubsub.psubscribe(channel)
                    logger.info(f"Pattern subscribed to: {channel}")
                else:
                    # Regular subscription
                    await self._pubsub.subscribe(channel)
                    logger.info(f"Subscribed to channel: {channel}")

            logger.info(f"Subscribed to {len(self.channels)} channels")

        except Exception as e:
            logger.error(f"Failed to subscribe to channels: {e}")
            raise

    async def start(self):
        """Start the notification service."""
        try:
            self._running = True
            logger.info("Starting Notification Service...")

            # Start message processing
            message_task = asyncio.create_task(self._process_messages())
            self._tasks.add(message_task)

            # Start health check task
            health_task = asyncio.create_task(self._health_check_loop())
            self._tasks.add(health_task)

            # Start daily summary task
            summary_task = asyncio.create_task(self._daily_summary_loop())
            self._tasks.add(summary_task)

            logger.info("Notification Service started")

            # Wait for tasks to complete
            await asyncio.gather(*self._tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in notification service: {e}")
            raise

    async def _process_messages(self):
        """Process messages from Redis channels."""
        try:
            logger.info("Starting message processing...")

            if not self._pubsub:
                logger.error("PubSub not initialized")
                return

            async for message in self._pubsub.listen():
                if not self._running:
                    break

                if message["type"] in ["message", "pmessage"]:
                    try:
                        await self._handle_message(message)
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")

        except Exception as e:
            logger.error(f"Message processing error: {e}")
            if self._running:
                # Restart after delay
                await asyncio.sleep(5)
                asyncio.create_task(self._process_messages())

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle individual Redis message."""
        try:
            channel = message.get("channel", "")
            data_str = message.get("data", "{}")

            # Parse message data
            try:
                data = json.loads(data_str) if isinstance(data_str, str) else data_str
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse message data: {data_str}")
                return

            logger.debug(f"Received message on {channel}: {data.get('symbol', 'N/A')}")

            # Route message to appropriate handler
            if channel.startswith("executions:"):
                await self._handle_execution(data)
            elif channel.startswith("execution_errors:"):
                await self._handle_execution_error(data)
            elif channel.startswith("alerts:"):
                await self._handle_alert(data)
            elif channel == "risk:alerts":
                await self._handle_risk_alert(data)
            elif channel == "portfolio:updates":
                await self._handle_portfolio_update(data)
            elif channel == "system:status":
                await self._handle_system_status(data)
            elif channel == "market:alerts":
                await self._handle_market_alert(data)
            elif channel.startswith("signals:"):
                await self._handle_signal(data)
            elif channel == "screener:updates":
                await self._handle_screener_update(data)
            else:
                logger.debug(f"Unhandled channel: {channel}")

        except Exception as e:
            logger.error(f"Failed to handle message: {e}")

    async def _handle_execution(self, data: Dict[str, Any]):
        """Handle trade execution notification."""
        try:
            # Extract execution details
            symbol = data.get("symbol", "UNKNOWN")
            success = data.get("success", False)
            result = data.get("result", {})

            if not success:
                return  # Failed executions are handled separately

            # Check rate limiting
            if self._is_rate_limited(f"execution_{symbol}"):
                return

            # Extract trade details from result
            trade_data = {
                "symbol": symbol,
                "side": result.get("side", "UNKNOWN"),
                "quantity": result.get("quantity", 0),
                "price": result.get("price", 0.0),
                "strategy": data.get("strategy", "Unknown"),
                "execution_strategy": result.get("execution_strategy", "IMMEDIATE"),
                "order_id": (
                    result.get("order", {}).get("broker_order_id")
                    if isinstance(result.get("order"), dict)
                    else None
                ),
                "timestamp": data.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
            }

            # Calculate value
            trade_data["value"] = trade_data["quantity"] * trade_data["price"]

            # Send notification
            if self.notification_manager:
                success = await self.notification_manager.send_trading_notification(
                    trade_data
                )
                if success:
                    logger.info(f"Sent trade notification for {symbol}")
                else:
                    logger.warning(f"Failed to send trade notification for {symbol}")

        except Exception as e:
            logger.error(f"Failed to handle execution: {e}")

    async def _handle_execution_error(self, data: Dict[str, Any]):
        """Handle trade execution error notification."""
        try:
            symbol = data.get("symbol", "UNKNOWN")
            error = data.get("error", "Unknown error")
            strategy = data.get("strategy", "Unknown")

            # Check rate limiting
            if self._is_rate_limited(f"exec_error_{symbol}"):
                return

            # Send critical alert
            title = f"❌ Trade Execution Failed: {symbol}"
            message = f"""
Failed to execute trade for {symbol}
Strategy: {strategy}
Error: {error}
Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

            if self.notification_manager and self.notification_manager.gotify_client:
                await self.notification_manager.gotify_client.send_critical_alert(
                    title, message.strip(), data
                )
                logger.warning(f"Sent execution error notification for {symbol}")

        except Exception as e:
            logger.error(f"Failed to handle execution error: {e}")

    async def _handle_alert(self, data: Dict[str, Any]):
        """Handle general system alert."""
        try:
            alert_type = data.get("type", "unknown")
            severity = data.get("severity", "info")
            message = data.get("message", "System alert")

            # Check rate limiting
            if self._is_rate_limited(f"alert_{alert_type}"):
                return

            if (
                not self.notification_manager
                or not self.notification_manager.gotify_client
            ):
                return

            # Send appropriate alert level
            if severity == "critical":
                await self.notification_manager.gotify_client.send_critical_alert(
                    f"🚨 {alert_type.upper()}", message, data
                )
            elif severity == "warning":
                await self.notification_manager.gotify_client.send_warning_alert(
                    f"⚠️ {alert_type.upper()}", message, data
                )
            else:
                await self.notification_manager.gotify_client.send_info_notification(
                    f"ℹ️ {alert_type.upper()}", message, data
                )

        except Exception as e:
            logger.error(f"Failed to handle alert: {e}")

    async def _handle_risk_alert(self, data: Dict[str, Any]):
        """Handle risk management alert."""
        try:
            # Check rate limiting
            if self._is_rate_limited("risk_alert"):
                return

            if self.notification_manager:
                # The NotificationManager expects specific risk event format
                # We'll send it as a general alert instead
                title = f"⚠️ Risk Alert: {data.get('alert_type', 'UNKNOWN')}"
                message = data.get("message", "Risk threshold exceeded")

                if self.notification_manager.gotify_client:
                    await self.notification_manager.gotify_client.send_warning_alert(
                        title, message, data
                    )
                    logger.info("Sent risk alert notification")

        except Exception as e:
            logger.error(f"Failed to handle risk alert: {e}")

    async def _handle_portfolio_update(self, data: Dict[str, Any]):
        """Handle portfolio update notification."""
        try:
            # Only send portfolio summaries periodically (not for every update)
            update_type = data.get("type", "update")

            if update_type == "daily_summary":
                if self.notification_manager:
                    await self.notification_manager.send_portfolio_summary(data)
                    logger.info("Sent portfolio summary notification")
            elif update_type == "significant_change":
                # Send notification for significant changes
                change_pct = data.get("change_percentage", 0)
                if abs(change_pct) > 5:  # More than 5% change
                    if self.notification_manager:
                        await self.notification_manager.send_portfolio_summary(data)
                        logger.info(
                            f"Sent portfolio notification for {change_pct:.2f}% change"
                        )

        except Exception as e:
            logger.error(f"Failed to handle portfolio update: {e}")

    async def _handle_system_status(self, data: Dict[str, Any]):
        """Handle system status update."""
        try:
            status = data.get("status", "unknown")
            service = data.get("service", "unknown")

            # Only notify on status changes or errors
            if status in ["error", "degraded", "maintenance"]:
                # Check rate limiting
                if self._is_rate_limited(f"system_{service}"):
                    return

                if (
                    self.notification_manager
                    and self.notification_manager.gotify_client
                ):
                    alert_type = f"{status.upper()}"
                    message = (
                        f"Service {service} is {status}: {data.get('message', '')}"
                    )
                    severity = "high" if status == "error" else "medium"

                    await self.notification_manager.gotify_client.send_system_alert(
                        service, alert_type, message, severity
                    )
                    logger.warning(f"Sent system alert for {service}: {status}")

        except Exception as e:
            logger.error(f"Failed to handle system status: {e}")

    async def _handle_market_alert(self, data: Dict[str, Any]):
        """Handle market condition alert."""
        try:
            # Check rate limiting
            if self._is_rate_limited("market_alert"):
                return

            if self.notification_manager and self.notification_manager.gotify_client:
                title = f"Market Alert: {data.get('alert_type', 'UNKNOWN')}"
                message = data.get("message", "Market condition detected")

                await self.notification_manager.gotify_client.send_warning_alert(
                    title, message, data
                )
                logger.info("Sent market alert notification")

        except Exception as e:
            logger.error(f"Failed to handle market alert: {e}")

    async def _handle_signal(self, data: Dict[str, Any]):
        """Handle trading signal notification."""
        try:
            # Only notify for high-confidence signals
            confidence = data.get("confidence", 0)
            if confidence < 0.8:  # Skip low confidence signals
                return

            symbol = data.get("symbol", "UNKNOWN")
            signal_type = data.get("signal_type", "unknown")

            # Check rate limiting
            if self._is_rate_limited(f"signal_{symbol}"):
                return

            title = f"📊 Trading Signal: {symbol}"
            message = f"""
Signal: {signal_type.upper()}
Symbol: {symbol}
Confidence: {confidence:.2%}
Strategy: {data.get('strategy_name', 'Unknown')}
"""

            if self.notification_manager and self.notification_manager.gotify_client:
                await self.notification_manager.gotify_client.send_info_notification(
                    title, message.strip(), data
                )
                logger.info(f"Sent signal notification for {symbol}")

        except Exception as e:
            logger.error(f"Failed to handle signal: {e}")

    async def _handle_screener_update(self, data: Dict[str, Any]):
        """Handle screener update notifications."""
        try:
            screener_type = data.get("screener_type", "unknown")
            stocks_data = data.get("data", [])
            timestamp = data.get("timestamp")
            count = data.get("count", len(stocks_data))

            # Rate limiting check
            rate_key = f"screener_update_{screener_type}"
            if self._is_rate_limited(rate_key):
                return

            # Create notification message
            title = f"🔍 Screener Update: {screener_type.replace('_', ' ').title()}"

            # Build message with top symbols
            message_parts = [f"Found {count} stocks in {screener_type} screener"]

            if stocks_data:
                # Show top 5 symbols as examples
                top_symbols = []
                for i, stock in enumerate(stocks_data):
                    symbol = stock.get("symbol", "N/A")
                    price = stock.get("price")
                    change = stock.get("change")

                    if price and change:
                        change_str = (
                            f"+{change:.2f}%" if change > 0 else f"{change:.2f}%"
                        )
                        top_symbols.append(f"{symbol}: ${price:.2f} ({change_str})")
                    else:
                        top_symbols.append(symbol)

                if top_symbols:
                    separator = ",\n\t "
                    symbols_text = separator.join(top_symbols)
                    message_parts.append(f"\nTop symbols:\n\t {symbols_text}")

            message = "\n".join(message_parts)

            # Send notification
            if self.notification_manager:
                notification = Notification(
                    title=title,
                    message=message.strip(),
                    service="screener",
                    priority=5,  # Normal priority
                    tags=["screener", screener_type],
                    metadata={
                        "screener_type": screener_type,
                        "stock_count": count,
                        "timestamp": (
                            str(timestamp)
                            if timestamp
                            else str(datetime.now(timezone.utc))
                        ),
                    },
                )
                await self.notification_manager.send_notification(notification)
                logger.info(
                    f"Sent screener update notification: {screener_type} with {count} stocks"
                )

        except Exception as e:
            logger.error(f"Failed to handle screener update: {e}")

    def _is_rate_limited(self, key: str) -> bool:
        """Check if notification should be rate limited."""
        now = datetime.now(timezone.utc)
        last_sent = self._last_notifications.get(key)

        if last_sent:
            elapsed = (now - last_sent).total_seconds()
            if elapsed < self._notification_cooldown:
                return True

        self._last_notifications[key] = now
        return False

    async def _health_check_loop(self):
        """Periodic health check."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Check Redis connection
                if self._redis:
                    await self._redis.ping()

                # Check notification manager
                if (
                    self.notification_manager
                    and self.notification_manager.gotify_client
                ):
                    await self.notification_manager.gotify_client.test_connection()

                logger.debug("Health check passed")

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                # Try to reconnect
                try:
                    await self.initialize()
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect: {reconnect_error}")

    async def _daily_summary_loop(self):
        """Send daily summary notifications."""
        while self._running:
            try:
                # Calculate time until next summary (e.g., 9 PM UTC)
                now = datetime.now(timezone.utc)
                target_hour = 21  # 9 PM UTC

                if now.hour >= target_hour:
                    # Already past today's summary time, wait for tomorrow
                    next_summary = now.replace(
                        hour=target_hour, minute=0, second=0, microsecond=0
                    ) + timedelta(days=1)
                else:
                    # Today's summary time hasn't passed yet
                    next_summary = now.replace(
                        hour=target_hour, minute=0, second=0, microsecond=0
                    )

                wait_seconds = (next_summary - now).total_seconds()
                await asyncio.sleep(wait_seconds)

                # Send daily summary
                if self.notification_manager:
                    await self.notification_manager.send_daily_summary()
                    logger.info("Sent daily summary notification")

            except Exception as e:
                logger.error(f"Failed to send daily summary: {e}")
                await asyncio.sleep(3600)  # Wait an hour before retrying

    async def shutdown(self):
        """Shutdown the notification service."""
        try:
            logger.info("Shutting down Notification Service...")
            self._running = False

            # Cancel all tasks
            for task in self._tasks:
                task.cancel()

            # Wait for tasks to complete
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

            # Close pubsub
            if self._pubsub:
                await self._pubsub.unsubscribe()
                await self._pubsub.close()

            # Close Redis connection
            if self._redis:
                await self._redis.close()

            # Shutdown notification manager
            if self.notification_manager:
                await self.notification_manager.shutdown()

            logger.info("Notification Service shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point for the notification service."""
    service = NotificationService()

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(service.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize service
        await service.initialize()

        # Start service
        await service.start()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        await service.shutdown()


if __name__ == "__main__":
    # Run the service
    asyncio.run(main())
