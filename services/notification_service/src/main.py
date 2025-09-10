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

from monitoring.gotify_client import NotificationManager  # noqa: E402
from shared.config import Config, get_config  # noqa: E402
from shared.models import Notification  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NotificationService:
    """
    Service that monitors Redis events and sends notifications.

    Subscribes to various Redis channels for trade executions, errors,
    and other events, then sends appropriate notifications through Gotify.
    """

    def __init__(self) -> None:
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

    async def initialize(self) -> None:
        """Initialize all service components."""
        try:
            logger.info("Initializing Notification Service...")
            logger.debug(f"Config loaded: {self.config}")

            # Initialize Redis connection using environment variable
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
            logger.info(f"Connecting to Redis at: {redis_url}")
            logger.debug(
                "Redis connection parameters: encoding=utf-8, decode_responses=True, retry_on_timeout=True"
            )

            self._redis = await redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
            )

            # Test Redis connection
            logger.debug("Testing Redis connection...")
            await self._redis.ping()
            logger.info("Redis connection established")
            logger.debug("Redis ping successful")

            # Initialize notification manager
            logger.debug("Initializing notification manager...")
            self.notification_manager = NotificationManager(self.config)
            await self.notification_manager.startup()
            logger.info("Notification manager initialized")
            logger.debug("Notification manager startup completed")

            # Create pubsub instance
            logger.debug("Creating Redis PubSub instance...")
            self._pubsub = self._redis.pubsub()
            logger.debug("PubSub instance created")

            # Subscribe to channels
            logger.debug("Starting channel subscription process...")
            await self._subscribe_to_channels()

            logger.info("Notification Service initialized successfully")
            logger.debug("All initialization steps completed")

        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            logger.debug(f"Initialization error details: {type(e).__name__}: {str(e)}")
            raise

    async def _subscribe_to_channels(self) -> None:
        """Subscribe to Redis channels."""
        try:
            logger.debug("Starting channel subscription process...")
            if not self._pubsub:
                logger.error("PubSub not initialized")
                raise ValueError("PubSub not initialized")

            logger.debug(f"Total channels to subscribe: {len(self.channels)}")
            for i, channel in enumerate(self.channels):
                logger.debug(
                    f"Processing channel {i + 1}/{len(self.channels)}: {channel}"
                )
                if "*" in channel:
                    # Pattern subscription
                    logger.debug(f"Using pattern subscription for: {channel}")
                    await self._pubsub.psubscribe(channel)
                    logger.info(f"Pattern subscribed to: {channel}")
                else:
                    # Regular subscription
                    logger.debug(f"Using regular subscription for: {channel}")
                    await self._pubsub.subscribe(channel)
                    logger.info(f"Subscribed to channel: {channel}")

            logger.info(f"Subscribed to {len(self.channels)} channels")
            logger.debug("Channel subscription process completed")

        except Exception as e:
            logger.error(f"Failed to subscribe to channels: {e}")
            logger.debug(f"Subscription error details: {type(e).__name__}: {str(e)}")
            raise

    async def start(self) -> None:
        """Start the notification service."""
        try:
            self._running = True
            logger.info("Starting Notification Service...")
            logger.debug(f"Service running state set to: {self._running}")

            # Start message processing
            logger.debug("Creating message processing task...")
            message_task = asyncio.create_task(self._process_messages())
            self._tasks.add(message_task)
            logger.debug("Message processing task created and added")

            # Start health check task
            logger.debug("Creating health check task...")
            health_task = asyncio.create_task(self._health_check_loop())
            self._tasks.add(health_task)
            logger.debug("Health check task created and added")

            # Start daily summary task
            logger.debug("Creating daily summary task...")
            summary_task = asyncio.create_task(self._daily_summary_loop())
            self._tasks.add(summary_task)
            logger.debug("Daily summary task created and added")

            logger.info("Notification Service started")
            logger.debug(f"Total tasks running: {len(self._tasks)}")

            # Wait for tasks to complete
            logger.debug("Waiting for tasks to complete...")
            await asyncio.gather(*self._tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in notification service: {e}")
            logger.debug(f"Service error details: {type(e).__name__}: {str(e)}")
            raise

    async def _process_messages(self) -> None:
        """Process messages from Redis channels."""
        try:
            logger.info("Starting message processing...")
            logger.debug("Initializing message processing loop...")

            if not self._pubsub:
                logger.error("PubSub not initialized")
                return

            logger.debug("Beginning to listen for Redis messages...")
            message_count = 0
            async for message in self._pubsub.listen():
                logger.debug(
                    f"Received raw message (#{message_count + 1}): type={message.get('type')}"
                )

                if not self._running:
                    logger.debug("Service not running, breaking from message loop")
                    break

                if message["type"] in ["message", "pmessage"]:
                    message_count += 1
                    logger.debug(f"Processing message #{message_count}")
                    try:
                        await self._handle_message(message)
                        logger.debug(f"Successfully processed message #{message_count}")
                    except Exception as e:
                        logger.error(f"Error handling message #{message_count}: {e}")
                        logger.debug(
                            f"Message handling error details: {type(e).__name__}: {str(e)}"
                        )
                else:
                    logger.debug(f"Skipping message type: {message['type']}")

        except Exception as e:
            logger.error(f"Message processing error: {e}")
            logger.debug(
                f"Message processing error details: {type(e).__name__}: {str(e)}"
            )
            if self._running:
                logger.debug(
                    "Service still running, restarting message processing after 5 second delay"
                )
                # Restart after delay
                await asyncio.sleep(5)
                asyncio.create_task(self._process_messages())

    async def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle individual Redis message."""
        try:
            channel = message.get("channel", "")
            data_str = message.get("data", "{}")
            logger.debug(f"Handling message from channel: {channel}")

            # Parse message data
            try:
                data = json.loads(data_str) if isinstance(data_str, str) else data_str
                logger.debug(
                    f"Parsed message data type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse message data: {data_str}")
                logger.debug(f"Raw data that failed parsing: {repr(data_str)}")
                return {
                    "status": "parse_error",
                    "message": "Failed to parse message data",
                }

            logger.debug(f"Received message on {channel}: {data.get('symbol', 'N/A')}")

            # Route message to appropriate handler
            if channel.startswith("executions:"):
                logger.debug(f"Routing to execution handler for channel: {channel}")
                await self._handle_execution(data)
            elif channel.startswith("execution_errors:"):
                logger.debug(
                    f"Routing to execution error handler for channel: {channel}"
                )
                await self._handle_execution_error(data)
            elif channel.startswith("alerts:"):
                logger.debug(f"Routing to alert handler for channel: {channel}")
                await self._handle_alert(data)
            elif channel == "risk:alerts":
                logger.debug("Routing to risk alert handler")
                await self._handle_risk_alert(data)
            elif channel == "portfolio:updates":
                logger.debug("Routing to portfolio update handler")
                await self._handle_portfolio_update(data)
            elif channel == "system:status":
                logger.debug("Routing to system status handler")
                await self._handle_system_status(data)
            elif channel == "market:alerts":
                logger.debug("Routing to market alert handler")
                await self._handle_market_alert(data)
            elif channel.startswith("signals:"):
                logger.debug(f"Routing to signal handler for channel: {channel}")
                await self._handle_signal(data)
            elif channel == "screener:updates":
                logger.debug("Routing to screener update handler")
                await self._handle_screener_update(data)
            else:
                logger.debug(f"Unhandled channel: {channel}")
                logger.debug(
                    "Available handlers: executions, execution_errors, alerts, risk, portfolio, system, market, signals, screener"
                )

        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
            logger.debug(
                f"Message handling error details: {type(e).__name__}: {str(e)}"
            )
            logger.debug(
                f"Failed message channel: {message.get('channel')}, data preview: {str(message.get('data', ''))[:100]}"
            )
            return {"status": "error", "message": str(e)}

        return {"status": "success", "channel": channel}

    async def _handle_execution(self, data: Dict[str, Any]) -> None:
        """Handle trade execution notification."""
        try:
            logger.debug("Processing execution notification")
            # Extract execution details
            symbol = data.get("symbol", "UNKNOWN")
            success = data.get("success", False)
            result = data.get("result", {})

            logger.debug(
                f"Execution details - Symbol: {symbol}, Success: {success}, Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
            )

            if not success:
                logger.debug(
                    "Execution not successful, skipping (handled by error handler)"
                )
                return  # Failed executions are handled separately

            # Check rate limiting
            rate_key = f"execution_{symbol}"
            if self._is_rate_limited(rate_key):
                logger.debug(f"Rate limited execution notification for {symbol}")
                return
            logger.debug(f"Rate limiting passed for execution {symbol}")

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

    async def _handle_execution_error(self, data: Dict[str, Any]) -> None:
        """Handle trade execution error notification."""
        try:
            logger.debug("Processing execution error notification")
            symbol = data.get("symbol", "UNKNOWN")
            error = data.get("error", "Unknown error")
            strategy = data.get("strategy", "Unknown")

            logger.debug(
                f"Execution error details - Symbol: {symbol}, Error: {error}, Strategy: {strategy}"
            )

            # Check rate limiting
            rate_key = f"exec_error_{symbol}"
            if self._is_rate_limited(rate_key):
                logger.debug(f"Rate limited execution error notification for {symbol}")
                return
            logger.debug(f"Rate limiting passed for execution error {symbol}")

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

    async def _handle_alert(self, data: Dict[str, Any]) -> None:
        """Handle general system alert."""
        try:
            logger.debug("Processing system alert")
            alert_type = data.get("type", "unknown")
            severity = data.get("severity", "info")
            message = data.get("message", "System alert")

            logger.debug(
                f"Alert details - Type: {alert_type}, Severity: {severity}, Message preview: {message[:50]}..."
            )

            # Check rate limiting
            rate_key = f"alert_{alert_type}"
            if self._is_rate_limited(rate_key):
                logger.debug(f"Rate limited alert notification for type: {alert_type}")
                return
            logger.debug(f"Rate limiting passed for alert type: {alert_type}")

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

    async def _handle_risk_alert(self, data: Dict[str, Any]) -> None:
        """Handle risk management alert."""
        try:
            logger.debug("Processing risk alert")
            logger.debug(
                f"Risk alert data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )
            # Check rate limiting
            if self._is_rate_limited("risk_alert"):
                logger.debug("Rate limited risk alert notification")
                return
            logger.debug("Rate limiting passed for risk alert")

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

    async def _handle_portfolio_update(self, data: Dict[str, Any]) -> None:
        """Handle portfolio update notification."""
        try:
            logger.debug("Processing portfolio update")
            # Only send portfolio summaries periodically (not for every update)
            update_type = data.get("type", "update")
            logger.debug(f"Portfolio update type: {update_type}")

            if update_type == "daily_summary":
                logger.debug("Processing daily portfolio summary")
                if self.notification_manager:
                    await self.notification_manager.send_portfolio_summary(data)
                    logger.info("Sent portfolio summary notification")
                else:
                    logger.debug(
                        "Notification manager not available for portfolio summary"
                    )
            elif update_type == "significant_change":
                # Send notification for significant changes
                change_pct = data.get("change_percentage", 0)
                logger.debug(f"Portfolio change percentage: {change_pct}")
                if abs(change_pct) > 5:  # More than 5% change
                    logger.debug(
                        f"Significant portfolio change detected: {change_pct:.2f}%"
                    )
                    if self.notification_manager:
                        await self.notification_manager.send_portfolio_summary(data)
                        logger.info(
                            f"Sent portfolio notification for {change_pct:.2f}% change"
                        )
                    else:
                        logger.debug(
                            "Notification manager not available for significant change notification"
                        )
                else:
                    logger.debug(
                        f"Portfolio change not significant enough: {change_pct:.2f}%"
                    )
            else:
                logger.debug(f"Portfolio update type '{update_type}' not handled")

        except Exception as e:
            logger.error(f"Failed to handle portfolio update: {e}")

    async def _handle_system_status(self, data: Dict[str, Any]) -> None:
        """Handle system status update."""
        try:
            logger.debug("Processing system status update")
            status = data.get("status", "unknown")
            service = data.get("service", "unknown")
            logger.debug(f"System status - Service: {service}, Status: {status}")

            # Only notify on status changes or errors
            if status in ["error", "degraded", "maintenance"]:
                logger.debug(f"Status requires notification: {status}")
                # Check rate limiting
                rate_key = f"system_{service}"
                if self._is_rate_limited(rate_key):
                    logger.debug(
                        f"Rate limited system status notification for {service}"
                    )
                    return
                logger.debug(f"Rate limiting passed for system status {service}")
            else:
                logger.debug(f"Status '{status}' does not require notification")
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

    async def _handle_market_alert(self, data: Dict[str, Any]) -> None:
        """Handle market condition alert."""
        try:
            logger.debug("Processing market alert")
            logger.debug(
                f"Market alert data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )
            # Check rate limiting
            if self._is_rate_limited("market_alert"):
                logger.debug("Rate limited market alert notification")
                return
            logger.debug("Rate limiting passed for market alert")

            if self.notification_manager and self.notification_manager.gotify_client:
                title = f"Market Alert: {data.get('alert_type', 'UNKNOWN')}"
                message = data.get("message", "Market condition detected")

                await self.notification_manager.gotify_client.send_warning_alert(
                    title, message, data
                )
                logger.info("Sent market alert notification")

        except Exception as e:
            logger.error(f"Failed to handle market alert: {e}")

    async def _handle_signal(self, data: Dict[str, Any]) -> None:
        """Handle trading signal notification."""
        try:
            logger.debug("Processing trading signal")
            # Only notify for high-confidence signals
            confidence = data.get("confidence", 0)
            symbol = data.get("symbol", "UNKNOWN")
            signal_type = data.get("signal_type", "unknown")

            logger.debug(
                f"Signal details - Symbol: {symbol}, Type: {signal_type}, Confidence: {confidence}"
            )

            if confidence < 0.8:  # Skip low confidence signals
                logger.debug(f"Signal confidence too low ({confidence}), skipping")
                return
            logger.debug("Signal confidence sufficient for notification")

            # Check rate limiting
            rate_key = f"signal_{symbol}"
            if self._is_rate_limited(rate_key):
                logger.debug(f"Rate limited signal notification for {symbol}")
                return
            logger.debug(f"Rate limiting passed for signal {symbol}")

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

    async def _handle_screener_update(self, data: Dict[str, Any]) -> None:
        """Handle screener update notifications."""
        try:
            logger.debug("Processing screener update")
            screener_type = data.get("screener_type", "unknown")
            stocks_data = data.get("data", [])
            timestamp = data.get("timestamp")
            count = data.get("count", len(stocks_data))

            logger.debug(
                f"Screener update - Type: {screener_type}, Count: {count}, Data length: {len(stocks_data)}"
            )

            # Rate limiting check
            rate_key = f"screener_update_{screener_type}"
            if self._is_rate_limited(rate_key):
                logger.debug(
                    f"Rate limited screener update notification for {screener_type}"
                )
                return
            logger.debug(f"Rate limiting passed for screener update {screener_type}")

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
            logger.debug(
                f"Rate limit check for '{key}': elapsed={elapsed:.1f}s, cooldown={self._notification_cooldown}s"
            )
            if elapsed < self._notification_cooldown:
                logger.debug(
                    f"Rate limiting '{key}' - {elapsed:.1f}s < {self._notification_cooldown}s"
                )
                return True

        self._last_notifications[key] = now
        logger.debug(f"Rate limit passed for '{key}' - updating timestamp")
        return False

    async def _health_check_loop(self) -> None:
        """Periodic health check."""
        logger.debug("Starting health check loop")
        while self._running:
            try:
                logger.debug("Health check sleeping for 5 minutes...")
                await asyncio.sleep(300)  # Check every 5 minutes

                if not self._running:
                    logger.debug("Service not running, exiting health check loop")
                    break

                logger.debug("Performing health check...")

                # Check Redis connection
                if self._redis:
                    logger.debug("Testing Redis connection...")
                    await self._redis.ping()
                    logger.debug("Redis health check passed")
                else:
                    logger.warning("Redis connection not available for health check")

                # Check notification manager
                if (
                    self.notification_manager
                    and self.notification_manager.gotify_client
                ):
                    logger.debug("Testing notification manager connection...")
                    await self.notification_manager.gotify_client.test_connection()
                    logger.debug("Notification manager health check passed")
                else:
                    logger.warning(
                        "Notification manager not available for health check"
                    )

                logger.debug("Health check passed")

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                logger.debug(
                    f"Health check error details: {type(e).__name__}: {str(e)}"
                )
                # Try to reconnect
                try:
                    logger.debug(
                        "Attempting to reinitialize after health check failure..."
                    )
                    await self.initialize()
                    logger.debug(
                        "Reinitialization after health check failure successful"
                    )
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect: {reconnect_error}")
                    logger.debug(
                        f"Reconnection error details: {type(reconnect_error).__name__}: {str(reconnect_error)}"
                    )

    async def _daily_summary_loop(self) -> None:
        """Send daily summary notifications."""
        logger.debug("Starting daily summary loop")
        while self._running:
            try:
                # Calculate time until next summary (e.g., 9 PM UTC)
                now = datetime.now(timezone.utc)
                target_hour = 21  # 9 PM UTC
                logger.debug(f"Current time: {now}, Target hour: {target_hour}")

                if now.hour >= target_hour:
                    # Already past today's summary time, wait for tomorrow
                    next_summary = now.replace(
                        hour=target_hour, minute=0, second=0, microsecond=0
                    ) + timedelta(days=1)
                    logger.debug("Past today's summary time, scheduling for tomorrow")
                else:
                    # Today's summary time hasn't passed yet
                    next_summary = now.replace(
                        hour=target_hour, minute=0, second=0, microsecond=0
                    )
                    logger.debug("Today's summary time not reached yet")

                wait_seconds = (next_summary - now).total_seconds()
                logger.debug(
                    f"Next summary at: {next_summary}, waiting {wait_seconds:.1f} seconds"
                )

                await asyncio.sleep(wait_seconds)

                if not self._running:
                    logger.debug("Service not running, exiting daily summary loop")
                    break

                logger.debug("Time for daily summary, attempting to send...")
                # Send daily summary
                if self.notification_manager:
                    logger.debug("Sending daily summary via notification manager")
                    await self.notification_manager.send_daily_summary()
                    logger.info("Sent daily summary notification")
                else:
                    logger.warning(
                        "Notification manager not available for daily summary"
                    )

            except Exception as e:
                logger.error(f"Failed to send daily summary: {e}")
                logger.debug(
                    f"Daily summary error details: {type(e).__name__}: {str(e)}"
                )
                logger.debug("Waiting 1 hour before retrying daily summary")
                await asyncio.sleep(3600)  # Wait an hour before retrying

    async def shutdown(self) -> None:
        """Shutdown the notification service."""
        try:
            logger.info("Shutting down Notification Service...")
            logger.debug("Starting shutdown process...")
            self._running = False
            logger.debug(f"Service running state set to: {self._running}")

            # Cancel all tasks
            logger.debug(f"Cancelling {len(self._tasks)} tasks...")
            for i, task in enumerate(self._tasks):
                logger.debug(
                    f"Cancelling task {i + 1}/{len(self._tasks)}: {task.get_name() if hasattr(task, 'get_name') else 'unnamed'}"
                )
                task.cancel()
            logger.debug("All tasks cancelled")

            # Wait for tasks to complete
            if self._tasks:
                logger.debug("Waiting for tasks to complete...")
                await asyncio.gather(*self._tasks, return_exceptions=True)
                logger.debug("All tasks completed")
            else:
                logger.debug("No tasks to wait for")

            # Close pubsub
            if self._pubsub:
                logger.debug("Closing Redis PubSub...")
                await self._pubsub.unsubscribe()
                await self._pubsub.close()
                logger.debug("PubSub closed")
            else:
                logger.debug("No PubSub to close")

            # Close Redis connection
            if self._redis:
                logger.debug("Closing Redis connection...")
                await self._redis.close()
                logger.debug("Redis connection closed")
            else:
                logger.debug("No Redis connection to close")

            # Shutdown notification manager
            if self.notification_manager:
                logger.debug("Shutting down notification manager...")
                await self.notification_manager.shutdown()
                logger.debug("Notification manager shutdown complete")
            else:
                logger.debug("No notification manager to shutdown")

            logger.info("Notification Service shutdown complete")
            logger.debug("All shutdown steps completed successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.debug(f"Shutdown error details: {type(e).__name__}: {str(e)}")


async def main() -> None:
    """Main entry point for the notification service."""
    logger.debug("Starting main function")
    service = NotificationService()
    logger.debug("NotificationService instance created")

    # Setup signal handlers
    def signal_handler(sig: int, frame: Any) -> None:
        logger.info(f"Received signal {sig}")
        logger.debug(f"Signal handler called with signal: {sig}, frame: {frame}")
        asyncio.create_task(service.shutdown())

    logger.debug("Setting up signal handlers...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.debug("Signal handlers configured for SIGINT and SIGTERM")

    try:
        logger.debug("Attempting to initialize service...")
        # Initialize service
        await service.initialize()
        logger.debug("Service initialization completed")

        logger.debug("Attempting to start service...")
        # Start service
        await service.start()
        logger.debug("Service start completed")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        logger.debug("KeyboardInterrupt caught in main")
    except Exception as e:
        logger.error(f"Service error: {e}")
        logger.debug(f"Main function error details: {type(e).__name__}: {str(e)}")
    finally:
        logger.debug("Entering finally block - ensuring service shutdown")
        await service.shutdown()
        logger.debug("Main function completed")


if __name__ == "__main__":
    logger.debug("Script started - running main function")
    # Run the service
    asyncio.run(main())
    logger.debug("Script completed")
