import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import httpx
from enum import Enum
import ssl
from dataclasses import dataclass

# Import shared models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.models import RiskEvent, RiskSeverity, Notification
from shared.config import Config


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class GotifyMessage:
    """Gotify message structure"""
    title: str
    message: str
    priority: int = NotificationPriority.NORMAL.value
    extras: Optional[Dict[str, Any]] = None


class GotifyClient:
    """Client for sending notifications via Gotify"""

    def __init__(self, gotify_url: str, gotify_token: str):
        self.gotify_url = gotify_url.rstrip('/')
        self.gotify_token = gotify_token
        self.logger = logging.getLogger(__name__)
        self.session: Optional[httpx.AsyncClient] = None
        self.retry_count = 3
        self.retry_delay = 1.0

    async def startup(self):
        """Initialize Gotify client"""
        # Create SSL context for secure connections
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            verify=False  # For self-signed certificates
        )

        # Test connection
        try:
            await self.test_connection()
            self.logger.info("Gotify client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gotify client: {e}")

    async def shutdown(self):
        """Cleanup Gotify client"""
        if self.session:
            await self.session.aclose()

    async def test_connection(self) -> bool:
        """Test connection to Gotify server"""
        if self.session is None:
            self.logger.error("Gotify session not initialized")
            return False

        try:
            response = await self.session.get(
                f"{self.gotify_url}/health",
                headers={"X-Gotify-Key": self.gotify_token}
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Gotify connection test failed: {e}")
            return False

    async def send_notification(self, message: GotifyMessage) -> bool:
        """Send notification to Gotify"""
        if self.session is None:
            self.logger.warning("Gotify session not available, logging notification instead")
            self.logger.info(f"NOTIFICATION: {message.title} - {message.message}")
            return True

        payload = {
            "title": message.title,
            "message": message.message,
            "priority": message.priority
        }

        if message.extras:
            payload["extras"] = message.extras

        for attempt in range(self.retry_count):
            try:
                response = await self.session.post(
                    f"{self.gotify_url}/message",
                    params={"token": self.gotify_token},
                    json=payload
                )

                if response.status_code == 200:
                    self.logger.debug(f"Notification sent successfully: {message.title}")
                    return True
                else:
                    self.logger.warning(
                        f"Failed to send notification (attempt {attempt + 1}): "
                        f"HTTP {response.status_code} - {response.text}"
                    )

            except Exception as e:
                self.logger.error(f"Error sending notification (attempt {attempt + 1}): {e}")

            if attempt < self.retry_count - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        return False

    async def send_critical_alert(self, title: str, message: str, metadata: Optional[Dict] = None) -> bool:
        """Send critical alert notification"""
        gotify_message = GotifyMessage(
            title=f"🚨 CRITICAL: {title}",
            message=message,
            priority=NotificationPriority.CRITICAL.value,
            extras=metadata
        )
        return await self.send_notification(gotify_message)

    async def send_warning_alert(self, title: str, message: str, metadata: Optional[Dict] = None) -> bool:
        """Send warning alert notification"""
        gotify_message = GotifyMessage(
            title=f"⚠️ WARNING: {title}",
            message=message,
            priority=NotificationPriority.HIGH.value,
            extras=metadata
        )
        return await self.send_notification(gotify_message)

    async def send_info_notification(self, title: str, message: str, metadata: Optional[Dict] = None) -> bool:
        """Send info notification"""
        gotify_message = GotifyMessage(
            title=f"ℹ️ INFO: {title}",
            message=message,
            priority=NotificationPriority.NORMAL.value,
            extras=metadata
        )
        return await self.send_notification(gotify_message)

    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade execution notification"""
        symbol = trade_data.get('symbol', 'UNKNOWN')
        side = trade_data.get('side', 'UNKNOWN')
        quantity = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0.0)
        pnl = trade_data.get('pnl')

        title = f"Trade Executed: {symbol}"

        message = f"""
        📈 Trade Details:
        Symbol: {symbol}
        Side: {side.upper()}
        Quantity: {quantity:,}
        Price: ${price:.2f}
        Value: ${quantity * price:,.2f}
        """

        if pnl is not None:
            message += f"\nP&L: ${pnl:,.2f}"
            if pnl > 0:
                title = f"✅ Profitable Trade: {symbol}"
            else:
                title = f"❌ Loss Trade: {symbol}"

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=NotificationPriority.NORMAL.value,
            extras=trade_data
        )

        return await self.send_notification(gotify_message)

    async def send_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> bool:
        """Send daily portfolio summary"""
        total_value = portfolio_data.get('total_value', 0)
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        total_pnl = portfolio_data.get('total_pnl', 0)
        drawdown = portfolio_data.get('drawdown', 0)
        positions_count = portfolio_data.get('positions_count', 0)

        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

        title = f"{pnl_emoji} Daily Portfolio Summary"

        message = f"""
        💰 Portfolio Overview:
        Total Value: ${total_value:,.2f}
        Daily P&L: ${daily_pnl:,.2f} ({daily_pnl/total_value*100:.2f}%)
        Total P&L: ${total_pnl:,.2f} ({total_pnl/(total_value-total_pnl)*100:.2f}%)
        Current Drawdown: {drawdown:.2%}
        Open Positions: {positions_count}

        📊 Risk Metrics:
        Max Drawdown: {portfolio_data.get('max_drawdown', 0):.2%}
        Sharpe Ratio: {portfolio_data.get('sharpe_ratio', 0):.2f}
        """

        if 'top_performers' in portfolio_data:
            message += "\n🏆 Top Performers:\n"
            for symbol, perf in portfolio_data['top_performers'].items():
                message += f"  {symbol}: {perf:.2%}\n"

        if 'worst_performers' in portfolio_data:
            message += "\n📉 Worst Performers:\n"
            for symbol, perf in portfolio_data['worst_performers'].items():
                message += f"  {symbol}: {perf:.2%}\n"

        priority = NotificationPriority.HIGH.value if abs(daily_pnl) > 5000 else NotificationPriority.NORMAL.value

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=priority,
            extras=portfolio_data
        )

        return await self.send_notification(gotify_message)

    async def send_risk_alert(self, risk_event: RiskEvent) -> bool:
        """Send risk management alert"""
        severity_emojis = {
            RiskSeverity.LOW: "🟡",
            RiskSeverity.MEDIUM: "🟠",
            RiskSeverity.HIGH: "🔴",
            RiskSeverity.CRITICAL: "🚨"
        }

        emoji = severity_emojis.get(risk_event.severity, "⚠️")
        title = f"{emoji} Risk Alert: {risk_event.event_type.value}"

        message = f"""
        🛡️ Risk Event Details:
        Type: {risk_event.event_type.value}
        Severity: {risk_event.severity.value}

        📝 Description:
        {risk_event.description}

        🎯 Symbol:
        {risk_event.symbol or 'N/A'}

        💡 Action Taken:
        {risk_event.action_taken or 'Pending'}

        ⏰ Time: {risk_event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        priority_map = {
            RiskSeverity.LOW: NotificationPriority.LOW,
            RiskSeverity.MEDIUM: NotificationPriority.NORMAL,
            RiskSeverity.HIGH: NotificationPriority.HIGH,
            RiskSeverity.CRITICAL: NotificationPriority.CRITICAL
        }

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=priority_map[risk_event.severity].value,
            extras={
                "event_type": risk_event.event_type.value,
                "severity": risk_event.severity.value,
                "symbol": risk_event.symbol,
                "timestamp": risk_event.timestamp.isoformat()
            }
        )

        return await self.send_notification(gotify_message)

    async def send_system_alert(self, service_name: str, alert_type: str,
                              message: str, severity: str = "warning") -> bool:
        """Send system-level alert"""
        severity_emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨"
        }

        emoji = severity_emojis.get(severity, "⚠️")
        title = f"{emoji} System Alert: {service_name}"

        notification_message = f"""
        🖥️ System Event:
        Service: {service_name}
        Type: {alert_type}

        📝 Details:
        {message}

        ⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        priority_map = {
            "info": NotificationPriority.LOW,
            "warning": NotificationPriority.HIGH,
            "critical": NotificationPriority.CRITICAL
        }

        gotify_message = GotifyMessage(
            title=title,
            message=notification_message.strip(),
            priority=priority_map.get(severity, NotificationPriority.NORMAL).value,
            extras={
                "service": service_name,
                "alert_type": alert_type,
                "severity": severity
            }
        )

        return await self.send_notification(gotify_message)

    async def send_market_alert(self, symbol: str, alert_type: str,
                              current_price: float, change_percent: float,
                              volume_ratio: Optional[float] = None) -> bool:
        """Send market condition alert"""
        title = f"📊 Market Alert: {symbol}"

        message = f"""
        📈 Market Event:
        Symbol: {symbol}
        Alert Type: {alert_type}
        Current Price: ${current_price:.2f}
        Price Change: {change_percent:.2%}
        """

        if volume_ratio:
            message += f"Volume Ratio: {volume_ratio:.1f}x average\n"

        message += f"\n⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"

        # Determine priority based on change magnitude
        if abs(change_percent) > 0.1:  # 10% change
            priority = NotificationPriority.CRITICAL
        elif abs(change_percent) > 0.05:  # 5% change
            priority = NotificationPriority.HIGH
        else:
            priority = NotificationPriority.NORMAL

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=priority.value,
            extras={
                "symbol": symbol,
                "alert_type": alert_type,
                "price": current_price,
                "change_percent": change_percent,
                "volume_ratio": volume_ratio
            }
        )

        return await self.send_notification(gotify_message)

    async def send_strategy_performance_alert(self, strategy_name: str,
                                            performance_data: Dict[str, Any]) -> bool:
        """Send strategy performance alert"""
        win_rate = performance_data.get('win_rate', 0)
        sharpe_ratio = performance_data.get('sharpe_ratio', 0)
        total_return = performance_data.get('total_return', 0)
        max_drawdown = performance_data.get('max_drawdown', 0)

        # Determine alert type based on performance
        if sharpe_ratio < 0.5 or max_drawdown > 0.2:
            alert_emoji = "📉"
            alert_type = "Poor Performance"
            priority = NotificationPriority.HIGH
        elif sharpe_ratio > 2.0 and win_rate > 0.7:
            alert_emoji = "🚀"
            alert_type = "Excellent Performance"
            priority = NotificationPriority.NORMAL
        else:
            alert_emoji = "📊"
            alert_type = "Performance Update"
            priority = NotificationPriority.LOW

        title = f"{alert_emoji} Strategy Alert: {strategy_name}"

        message = f"""
        🎯 Strategy Performance:
        Name: {strategy_name}
        Status: {alert_type}

        📈 Metrics:
        Win Rate: {win_rate:.1%}
        Sharpe Ratio: {sharpe_ratio:.2f}
        Total Return: {total_return:.2%}
        Max Drawdown: {max_drawdown:.2%}

        📊 Additional Info:
        Total Trades: {performance_data.get('total_trades', 0)}
        Avg Trade Return: {performance_data.get('avg_trade_return', 0):.2%}
        Profit Factor: {performance_data.get('profit_factor', 0):.2f}

        ⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=priority.value,
            extras=performance_data
        )

        return await self.send_notification(gotify_message)

    async def send_api_failure_alert(self, api_name: str, endpoint: str,
                                   error_message: str, error_count: int) -> bool:
        """Send API failure alert"""
        title = f"🔌 API Failure: {api_name}"

        message = f"""
        🔧 API Connection Issue:
        API: {api_name}
        Endpoint: {endpoint}
        Error Count: {error_count}

        🔍 Error Details:
        {error_message}

        💡 Impact:
        This may affect data collection and trade execution.
        Please check API credentials and connectivity.

        ⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        priority = NotificationPriority.CRITICAL if error_count > 5 else NotificationPriority.HIGH

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=priority.value,
            extras={
                "api": api_name,
                "endpoint": endpoint,
                "error_count": error_count,
                "error_message": error_message
            }
        )

        return await self.send_notification(gotify_message)

    async def send_large_loss_alert(self, loss_amount: float, portfolio_value: float,
                                  loss_percentage: float, affected_positions: List[str]) -> bool:
        """Send large loss alert"""
        title = f"📉 Large Loss Alert: ${abs(loss_amount):,.2f}"

        message = f"""
        💸 Significant Loss Detected:
        Loss Amount: ${abs(loss_amount):,.2f}
        Portfolio Value: ${portfolio_value:,.2f}
        Loss Percentage: {loss_percentage:.2%}

        📊 Affected Positions:
        {', '.join(affected_positions)}

        🛡️ Risk Management:
        Current drawdown limits and stop losses are being enforced.
        Consider reviewing position sizes and risk parameters.

        ⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        priority = NotificationPriority.CRITICAL if abs(loss_percentage) > 0.05 else NotificationPriority.HIGH

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=priority.value,
            extras={
                "loss_amount": loss_amount,
                "portfolio_value": portfolio_value,
                "loss_percentage": loss_percentage,
                "affected_positions": affected_positions
            }
        )

        return await self.send_notification(gotify_message)

    async def send_unusual_market_conditions_alert(self, conditions: Dict[str, Any]) -> bool:
        """Send unusual market conditions alert"""
        title = "🌪️ Unusual Market Conditions"

        message = """
        🌊 Market Anomaly Detected:

        📊 Conditions:
        """

        for condition, value in conditions.items():
            if isinstance(value, float):
                if 'percentage' in condition or 'ratio' in condition:
                    message += f"{condition}: {value:.2%}\n"
                else:
                    message += f"{condition}: {value:.2f}\n"
            else:
                message += f"{condition}: {value}\n"

        message += f"""

        ⚡ Recommendations:
        - Review open positions for exposure
        - Consider reducing position sizes
        - Monitor for further developments
        - Check stop-loss orders

        ⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=NotificationPriority.HIGH.value,
            extras=conditions
        )

        return await self.send_notification(gotify_message)

    async def send_system_startup_notification(self, services_status: Dict[str, bool]) -> bool:
        """Send system startup notification"""
        healthy_services = [name for name, status in services_status.items() if status]
        unhealthy_services = [name for name, status in services_status.items() if not status]

        if unhealthy_services:
            title = "⚠️ System Startup - Some Issues"
            emoji = "⚠️"
            priority = NotificationPriority.HIGH
        else:
            title = "✅ System Startup - All Services Healthy"
            emoji = "✅"
            priority = NotificationPriority.NORMAL

        message = f"""
        {emoji} Trading System Status:

        ✅ Healthy Services ({len(healthy_services)}):
        {', '.join(healthy_services) if healthy_services else 'None'}

        ❌ Unhealthy Services ({len(unhealthy_services)}):
        {', '.join(unhealthy_services) if unhealthy_services else 'None'}

        🕐 Startup Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        """

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=priority.value,
            extras=services_status
        )

        return await self.send_notification(gotify_message)

    async def send_scheduled_report(self, report_type: str, report_data: Dict[str, Any]) -> bool:
        """Send scheduled reports (daily, weekly, monthly)"""
        report_emojis = {
            "daily": "📅",
            "weekly": "📊",
            "monthly": "📈"
        }

        emoji = report_emojis.get(report_type, "📋")
        title = f"{emoji} {report_type.title()} Trading Report"

        message = f"""
        📊 {report_type.title()} Performance Summary:

        💰 Portfolio Metrics:
        Starting Value: ${report_data.get('starting_value', 0):,.2f}
        Ending Value: ${report_data.get('ending_value', 0):,.2f}
        Period Return: {report_data.get('period_return', 0):.2%}

        📈 Trading Activity:
        Total Trades: {report_data.get('total_trades', 0)}
        Winning Trades: {report_data.get('winning_trades', 0)}
        Win Rate: {report_data.get('win_rate', 0):.1%}

        🎯 Performance:
        Sharpe Ratio: {report_data.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {report_data.get('max_drawdown', 0):.2%}
        Volatility: {report_data.get('volatility', 0):.2%}

        💸 Costs:
        Total Commission: ${report_data.get('total_commission', 0):.2f}
        Total Slippage: ${report_data.get('total_slippage', 0):.2f}
        """

        if 'best_strategy' in report_data:
            message += f"\n🏆 Best Strategy: {report_data['best_strategy']}"

        if 'worst_strategy' in report_data:
            message += f"\n📉 Worst Strategy: {report_data['worst_strategy']}"

        gotify_message = GotifyMessage(
            title=title,
            message=message.strip(),
            priority=NotificationPriority.NORMAL.value,
            extras=report_data
        )

        return await self.send_notification(gotify_message)


class NotificationManager:
    """Manages all notifications for the trading system"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gotify_client: Optional[GotifyClient] = None
        self.notification_history: List[Dict] = []
        self.rate_limits: Dict[str, float] = {}
        self.last_notifications: Dict[str, datetime] = {}

    async def startup(self):
        """Initialize notification manager"""
        if (hasattr(self.config, 'notifications') and
            self.config.notifications.gotify_url and
            self.config.notifications.gotify_token):
            self.gotify_client = GotifyClient(self.config.notifications.gotify_url, self.config.notifications.gotify_token)
            await self.gotify_client.startup()
            self.logger.info("Notification manager initialized with Gotify")
        else:
            self.logger.info("Notification manager initialized without Gotify (logging only)")

    async def shutdown(self):
        """Cleanup notification manager"""
        if self.gotify_client:
            await self.gotify_client.shutdown()

    async def send_notification(self, notification: Notification) -> bool:
        """Send notification with rate limiting and deduplication"""
        notification_key = f"{notification.service}_{notification.title}"

        # Check rate limiting
        if self._is_rate_limited(notification_key, notification.priority):
            self.logger.debug(f"Notification rate limited: {notification.title}")
            return False

        # Log notification
        self.logger.info(f"Sending notification: {notification.title}")

        success = False

        # Send via Gotify if available
        if self.gotify_client:
            gotify_message = GotifyMessage(
                title=notification.title,
                message=notification.message,
                priority=self._map_priority(str(notification.priority)),
                extras=notification.metadata
            )
            success = await self.gotify_client.send_notification(gotify_message)

        # Always log the notification
        self._log_notification(notification)

        # Record in history
        self.notification_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": notification.title,
            "message": notification.message,
            "service": notification.service,
            "priority": notification.priority,
            "sent_successfully": success
        })

        # Trim history to last 1000 notifications
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]

        # Update rate limiting
        self.last_notifications[notification_key] = datetime.now(timezone.utc)

        return success

    def _is_rate_limited(self, notification_key: str, priority: int) -> bool:
        """Check if notification is rate limited"""
        now = datetime.now(timezone.utc)
        last_sent = self.last_notifications.get(notification_key)

        if not last_sent:
            return False

        # Rate limits based on priority
        rate_limits = {
            10: timedelta(minutes=1),   # Critical: max once per minute
            8: timedelta(minutes=5),    # High: max once per 5 minutes
            5: timedelta(minutes=15),   # Medium: max once per 15 minutes
            1: timedelta(hours=1)       # Low: max once per hour
        }

        rate_limit = rate_limits.get(priority, timedelta(minutes=15))
        return (now - last_sent) < rate_limit

    def _map_priority(self, priority: str) -> int:
        """Map priority string to Gotify priority number"""
        priority_map = {
            "low": NotificationPriority.LOW.value,
            "medium": NotificationPriority.NORMAL.value,
            "high": NotificationPriority.HIGH.value,
            "critical": NotificationPriority.CRITICAL.value
        }
        return priority_map.get(priority.lower(), NotificationPriority.NORMAL.value)

    def _log_notification(self, notification: Notification):
        """Log notification to file"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": notification.title,
            "message": notification.message,
            "service": notification.service,
            "priority": notification.priority,
            "metadata": notification.metadata
        }

        # Log to appropriate level based on priority
        if notification.priority >= 10:
            self.logger.critical(f"NOTIFICATION: {json.dumps(log_entry)}")
        elif notification.priority >= 8:
            self.logger.error(f"NOTIFICATION: {json.dumps(log_entry)}")
        elif notification.priority >= 5:
            self.logger.warning(f"NOTIFICATION: {json.dumps(log_entry)}")
        else:
            self.logger.info(f"NOTIFICATION: {json.dumps(log_entry)}")

    async def send_trading_notification(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade execution notification"""
        if self.gotify_client:
            return await self.gotify_client.send_trade_notification(trade_data)
        return True

    async def send_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> bool:
        """Send portfolio summary notification"""
        if self.gotify_client:
            return await self.gotify_client.send_portfolio_summary(portfolio_data)
        return True

    async def send_risk_alert(self, risk_event: RiskEvent) -> bool:
        """Send risk alert notification"""
        if self.gotify_client:
            return await self.gotify_client.send_risk_alert(risk_event)
        return True

    async def send_system_alert(self, service_name: str, alert_type: str,
                              message: str, severity: str = "warning") -> bool:
        """Send system alert notification"""
        if self.gotify_client:
            return await self.gotify_client.send_system_alert(service_name, alert_type, message, severity)
        return True

    async def send_daily_summary(self) -> bool:
        """Send daily trading summary"""
        try:
            # Collect daily summary data
            summary_data = await self._collect_daily_summary()

            if self.gotify_client:
                return await self.gotify_client.send_scheduled_report("daily", summary_data)

            return True

        except Exception as e:
            self.logger.error(f"Failed to send daily summary: {e}")
            return False

    async def _collect_daily_summary(self) -> Dict[str, Any]:
        """Collect data for daily summary"""
        # This would integrate with the database to collect actual metrics
        # For now, return placeholder data
        return {
            "starting_value": 100000.0,
            "ending_value": 101500.0,
            "period_return": 0.015,
            "total_trades": 12,
            "winning_trades": 8,
            "win_rate": 0.667,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.03,
            "volatility": 0.15,
            "total_commission": 24.0,
            "total_slippage": 45.0,
            "best_strategy": "moving_average_20_50",
            "worst_strategy": "rsi_14_30_70"
        }

    def get_notification_history(self, limit: int = 100) -> List[Dict]:
        """Get recent notification history"""
        return self.notification_history[-limit:]

    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        now = datetime.now(timezone.utc)
        today = now.date()

        today_notifications = [
            n for n in self.notification_history
            if datetime.fromisoformat(n['timestamp']).date() == today
        ]

        return {
            "total_notifications": len(self.notification_history),
            "today_notifications": len(today_notifications)
        }
