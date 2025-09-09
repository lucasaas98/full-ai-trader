"""
Risk Alert Manager

This module handles risk alerts, notifications, and communication with external
alert systems including Gotify, Slack, and email notifications.
"""

import asyncio
import logging
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Set

import aiohttp

from shared.config import get_config
from shared.models import RiskAlert, RiskEvent, RiskEventType, RiskSeverity

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages risk alerts and notifications."""

    def __init__(self) -> None:
        """Initialize alert manager."""
        self.config = get_config()
        self.notification_config = self.config.notifications

        # Alert tracking
        self.sent_alerts: Set[str] = set()
        self.alert_history: List[RiskAlert] = []
        self.last_notification_times: Dict[str, datetime] = {}

        # Rate limiting
        self.notification_cooldown = timedelta(minutes=2)
        self.critical_cooldown = timedelta(seconds=30)

        # Queue for batch processing
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        self.batch_size = 5
        self.batch_timeout = timedelta(seconds=30)

    async def send_risk_alert(self, alert: RiskAlert) -> bool:
        """
        Send a risk alert through configured notification channels.

        Args:
            alert: Risk alert to send

        Returns:
            bool: True if alert was sent successfully
        """
        try:
            # Check if we should send this alert (rate limiting)
            if not await self._should_send_alert(alert):
                logger.debug(f"Skipping alert due to rate limiting: {alert.title}")
                return False

            # Add to queue for batch processing
            await self.alert_queue.put(alert)

            # Process immediately for critical alerts
            if alert.severity == RiskSeverity.CRITICAL:
                await self._process_alert_immediately(alert)

            # Track sent alert
            self._track_sent_alert(alert)

            logger.info(f"Risk alert queued: {alert.title} ({alert.severity})")
            return True

        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")
            return False

    async def send_bulk_alerts(self, alerts: List[RiskAlert]) -> int:
        """
        Send multiple alerts efficiently.

        Args:
            alerts: List of alerts to send

        Returns:
            int: Number of alerts sent successfully
        """
        sent_count = 0

        # Group alerts by severity
        critical_alerts = [a for a in alerts if a.severity == RiskSeverity.CRITICAL]
        other_alerts = [a for a in alerts if a.severity != RiskSeverity.CRITICAL]

        # Send critical alerts immediately
        for alert in critical_alerts:
            if await self.send_risk_alert(alert):
                sent_count += 1

        # Batch non-critical alerts
        if other_alerts:
            batch_sent = await self._send_alert_batch(other_alerts)
            sent_count += batch_sent

        return sent_count

    async def _process_alert_immediately(self, alert: RiskAlert) -> bool:
        """Process critical alert immediately."""
        try:
            # Send to all configured channels
            tasks = []

            if (
                self.notification_config.gotify_url
                and self.notification_config.gotify_token
            ):
                tasks.append(self._send_gotify_alert(alert))

            if self.notification_config.slack_webhook_url:
                tasks.append(self._send_slack_alert(alert))

            if (
                self.notification_config.email_smtp_host
                and self.notification_config.email_to
            ):
                tasks.append(self._send_email_alert(alert))

            # Execute all notifications concurrently
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                success_count = sum(1 for result in results if result is True)
                logger.info(
                    f"Critical alert sent to {success_count}/{len(tasks)} channels"
                )

        except Exception as e:
            logger.error(f"Error processing immediate alert: {e}")
            return False
        return True

    async def _send_alert_batch(self, alerts: List[RiskAlert]) -> int:
        """Send a batch of alerts."""
        if not alerts:
            return 0

        try:
            # Create batch notification
            batch_notification = self._create_batch_notification(alerts)

            # Send batch to all channels
            success_count = 0

            if (
                self.notification_config.gotify_url
                and self.notification_config.gotify_token
            ):
                if await self._send_gotify_notification(batch_notification):
                    success_count += 1

            if self.notification_config.slack_webhook_url:
                if await self._send_slack_notification(batch_notification):
                    success_count += 1

            return len(alerts) if success_count > 0 else 0

        except Exception as e:
            logger.error(f"Error sending alert batch: {e}")
            return 0

    async def _send_gotify_alert(self, alert: RiskAlert) -> bool:
        """Send alert via Gotify."""
        try:
            url = f"{self.notification_config.gotify_url}/message"

            # Map severity to Gotify priority
            priority_map = {
                RiskSeverity.LOW: 2,
                RiskSeverity.MEDIUM: 5,
                RiskSeverity.HIGH: 8,
                RiskSeverity.CRITICAL: 10,
            }

            payload = {
                "title": f"[{alert.severity.upper()}] {alert.title}",
                "message": alert.message,
                "priority": priority_map.get(alert.severity, 5),
                "extras": {
                    "client::display": {"contentType": "text/markdown"},
                    "alert_type": alert.alert_type,
                    "symbol": alert.symbol,
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata,
                },
            }

            headers = {
                "Content-Type": "application/json",
            }
            if self.notification_config.gotify_token:
                headers["X-Gotify-Key"] = self.notification_config.gotify_token

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Gotify alert sent successfully: {alert.title}")
                        return True
                    else:
                        logger.warning(
                            f"Gotify alert failed with status {response.status}"
                        )
                        return False

        except Exception as e:
            logger.error(f"Error sending Gotify alert: {e}")
            return False

    async def _send_slack_alert(self, alert: RiskAlert) -> bool:
        """Send alert via Slack webhook."""
        try:
            # Color coding by severity
            color_map = {
                RiskSeverity.LOW: "#36a64f",  # Green
                RiskSeverity.MEDIUM: "#ff9800",  # Orange
                RiskSeverity.HIGH: "#f44336",  # Red
                RiskSeverity.CRITICAL: "#9c27b0",  # Purple
            }

            # Create Slack payload
            payload: Dict[str, Any] = {
                "username": "Risk Manager",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#ff9800"),
                        "title": f"{alert.severity.upper()}: {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Alert Type",
                                "value": alert.alert_type,
                                "short": True,
                            },
                            {
                                "title": "Symbol",
                                "value": alert.symbol or "Portfolio",
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime(
                                    "%Y-%m-%d %H:%M:%S UTC"
                                ),
                                "short": True,
                            },
                            {
                                "title": "Action Required",
                                "value": "Yes" if alert.action_required else "No",
                                "short": True,
                            },
                        ],
                        "footer": "Risk Management System",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }

            # Add metadata fields if present
            if alert.metadata and isinstance(alert.metadata, dict):
                for key, value in alert.metadata.items():
                    payload["attachments"][0]["fields"].append(
                        {
                            "title": str(key).replace("_", " ").title(),
                            "value": str(value),
                            "short": True,
                        }
                    )

            if not self.notification_config.slack_webhook_url:
                logger.warning("Slack webhook URL not configured")
                return False

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.notification_config.slack_webhook_url,
                    json=payload,
                    timeout=timeout,
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Slack alert sent successfully: {alert.title}")
                        return True
                    else:
                        logger.warning(
                            f"Slack alert failed with status {response.status}"
                        )
                        return False

        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False

    async def _send_email_alert(self, alert: RiskAlert) -> bool:
        """Send alert via email."""
        try:
            if not all(
                [
                    self.notification_config.email_smtp_host,
                    self.notification_config.email_username,
                    self.notification_config.email_password,
                    self.notification_config.email_from,
                    self.notification_config.email_to,
                ]
            ):
                logger.warning("Email configuration incomplete, skipping email alert")
                return False

            # Validate email configuration
            if not self.notification_config.email_from:
                logger.warning("Email from address not configured")
                return False

            if not self.notification_config.email_to:
                logger.warning("Email to addresses not configured")
                return False

            # Create email message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.upper()}] Risk Alert: {alert.title}"
            msg["From"] = self.notification_config.email_from
            msg["To"] = ", ".join(self.notification_config.email_to)

            # Create HTML and text versions
            text_content = self._create_text_email_content(alert)
            html_content = self._create_html_email_content(alert)

            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")

            msg.attach(text_part)
            msg.attach(html_part)

            # Validate SMTP configuration
            if not self.notification_config.email_smtp_host:
                logger.warning("SMTP host not configured")
                return False

            if (
                not self.notification_config.email_username
                or not self.notification_config.email_password
            ):
                logger.warning("SMTP credentials not configured")
                return False

            # Send email
            with smtplib.SMTP(
                self.notification_config.email_smtp_host,
                self.notification_config.email_smtp_port,
            ) as server:
                server.starttls()
                server.login(
                    self.notification_config.email_username,
                    self.notification_config.email_password,
                )
                text = msg.as_string()
                server.sendmail(
                    self.notification_config.email_from,
                    self.notification_config.email_to,
                    text,
                )

            logger.debug(f"Email alert sent successfully: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False

    def _create_text_email_content(self, alert: RiskAlert) -> str:
        """Create plain text email content."""
        content = f"""
Risk Management Alert

Alert: {alert.title}
Severity: {alert.severity.upper()}
Type: {alert.alert_type}
Symbol: {alert.symbol or 'Portfolio'}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Action Required: {'Yes' if alert.action_required else 'No'}

Message:
{alert.message}
"""

        if alert.metadata:
            content += "\n\nAdditional Information:\n"
            for key, value in alert.metadata.items():
                content += f"- {key.replace('_', ' ').title()}: {value}\n"

        content += "\n\nThis alert was generated by the Risk Management System."

        return content

    def _create_html_email_content(self, alert: RiskAlert) -> str:
        """Create HTML email content."""

        # Color coding by severity
        color_map = {
            RiskSeverity.LOW: "#4caf50",  # Green
            RiskSeverity.MEDIUM: "#ff9800",  # Orange
            RiskSeverity.HIGH: "#f44336",  # Red
            RiskSeverity.CRITICAL: "#9c27b0",  # Purple
        }

        color = color_map.get(alert.severity, "#ff9800")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .alert-header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
        .alert-body {{ padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
        .metadata {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; margin-top: 10px; }}
        .footer {{ font-size: 12px; color: #666; margin-top: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .label {{ font-weight: bold; width: 150px; }}
    </style>
</head>
<body>
    <div class="alert-header">
        <h2>Risk Management Alert</h2>
        <h3>{alert.title}</h3>
    </div>

    <div class="alert-body">
        <table>
            <tr>
                <td class="label">Severity:</td>
                <td>{alert.severity.upper()}</td>
            </tr>
            <tr>
                <td class="label">Alert Type:</td>
                <td>{alert.alert_type}</td>
            </tr>
            <tr>
                <td class="label">Symbol:</td>
                <td>{alert.symbol or 'Portfolio'}</td>
            </tr>
            <tr>
                <td class="label">Time:</td>
                <td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
            </tr>
            <tr>
                <td class="label">Action Required:</td>
                <td>{'Yes' if alert.action_required else 'No'}</td>
            </tr>
        </table>

        <h4>Message:</h4>
        <p>{alert.message}</p>
"""

        if alert.metadata:
            html += """
        <div class="metadata">
            <h4>Additional Information:</h4>
            <table>
"""
            for key, value in alert.metadata.items():
                html += f"""
                <tr>
                    <td class="label">{key.replace('_', ' ').title()}:</td>
                    <td>{value}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        html += """
    </div>

    <div class="footer">
        <p>This alert was generated by the Risk Management System.</p>
    </div>
</body>
</html>
"""

        return html

    async def _send_gotify_notification(self, notification: Dict) -> bool:
        """Send notification via Gotify."""
        try:
            if (
                not self.notification_config.gotify_url
                or not self.notification_config.gotify_token
            ):
                return False

            url = f"{self.notification_config.gotify_url}/message"
            headers = {
                "Content-Type": "application/json",
            }
            if self.notification_config.gotify_token:
                headers["X-Gotify-Key"] = self.notification_config.gotify_token

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=notification, headers=headers, timeout=timeout
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Error sending Gotify notification: {e}")
            return False

    async def _send_slack_notification(self, notification: Dict) -> bool:
        """Send notification via Slack."""
        try:
            if not self.notification_config.slack_webhook_url:
                return False

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.notification_config.slack_webhook_url,
                    json=notification,
                    timeout=timeout,
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

    def _create_batch_notification(self, alerts: List[RiskAlert]) -> Dict:
        """Create a batch notification from multiple alerts."""

        # Group by severity
        severity_counts: Dict[str, int] = {}
        for alert in alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        # Create summary
        summary_parts = []
        for severity, count in severity_counts.items():
            summary_parts.append(f"{count} {severity}")

        summary = f"Risk Alert Summary: {', '.join(summary_parts)}"

        # Create detailed message
        message_parts = [f"**{summary}**\n"]

        for alert in alerts:
            message_parts.append(f"• **{alert.title}** ({alert.severity})")
            if alert.symbol:
                message_parts.append(f"  Symbol: {alert.symbol}")
            message_parts.append(f"  {alert.message}")
            message_parts.append("")  # Empty line

        return {
            "title": summary,
            "message": "\n".join(message_parts),
            "priority": max(
                2 if RiskSeverity.LOW in severity_counts else 0,
                5 if RiskSeverity.MEDIUM in severity_counts else 0,
                8 if RiskSeverity.HIGH in severity_counts else 0,
                10 if RiskSeverity.CRITICAL in severity_counts else 0,
            ),
        }

    async def _should_send_alert(self, alert: RiskAlert) -> bool:
        """Check if alert should be sent based on rate limiting."""

        # Create unique key for this alert type
        alert_key = f"{alert.alert_type}_{alert.symbol or 'portfolio'}"

        # Check if we've sent this alert recently
        if alert_key in self.last_notification_times:
            time_since_last = (
                datetime.now(timezone.utc) - self.last_notification_times[alert_key]
            )

            # Different cooldowns based on severity
            if alert.severity == RiskSeverity.CRITICAL:
                required_cooldown = self.critical_cooldown
            else:
                required_cooldown = self.notification_cooldown

            if time_since_last < required_cooldown:
                return False

        return True

    def _track_sent_alert(self, alert: RiskAlert) -> None:
        """Track that an alert was sent."""
        alert_key = f"{alert.alert_type}_{alert.symbol or 'portfolio'}"
        self.last_notification_times[alert_key] = datetime.now(timezone.utc)

        # Add to history
        self.alert_history.append(alert)

        # Trim history to last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    async def send_daily_risk_report(
        self, portfolio_metrics: Dict, risk_events: List[RiskEvent]
    ) -> bool:
        """Send daily risk management report."""
        try:
            # Create comprehensive daily report
            report = self._create_daily_report(portfolio_metrics, risk_events)

            # Send via email (primary channel for reports)
            if (
                self.notification_config.email_smtp_host
                and self.notification_config.email_to
            ):
                return await self._send_daily_report_email(report)

            # Fallback to Gotify if email not configured
            if (
                self.notification_config.gotify_url
                and self.notification_config.gotify_token
            ):
                return await self._send_gotify_notification(
                    {"title": "Daily Risk Report", "message": report, "priority": 3}
                )

            return False

        except Exception as e:
            logger.error(f"Error sending daily risk report: {e}")
            return False

    def _create_daily_report(
        self, portfolio_metrics: Dict, risk_events: List[RiskEvent]
    ) -> str:
        """Create daily risk report content."""

        report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        report = f"""
# Daily Risk Management Report - {report_date}

## Portfolio Summary
- Total Equity: ${portfolio_metrics.get('total_equity', 'N/A')}
- Total Exposure: ${portfolio_metrics.get('total_exposure', 'N/A')}
- Cash Percentage: {portfolio_metrics.get('cash_percentage', 'N/A')}
- Active Positions: {portfolio_metrics.get('position_count', 'N/A')}

## Risk Metrics
- Portfolio Beta: {portfolio_metrics.get('portfolio_beta', 'N/A')}
- Portfolio Volatility: {portfolio_metrics.get('volatility', 'N/A'):.1%}
- Value at Risk (1d): ${portfolio_metrics.get('value_at_risk_1d', 'N/A')}
- Max Drawdown: {portfolio_metrics.get('max_drawdown', 'N/A'):.2%}
- Current Drawdown: {portfolio_metrics.get('current_drawdown', 'N/A'):.2%}
- Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 'N/A'):.2f}

## Risk Events Today
"""

        if risk_events:
            for event in risk_events:
                report += f"- **{event.event_type}** ({event.severity}): {event.description}\n"
        else:
            report += "- No risk events recorded today\n"

        # Add compliance status
        report += f"""

## Compliance Status
- Within Position Limits: {'✓' if portfolio_metrics.get('position_count', 0) <= 5 else '✗'}
- Within Risk Limits: {'✓' if portfolio_metrics.get('current_drawdown', 0) < 0.15 else '✗'}
- Correlation Acceptable: {'✓' if portfolio_metrics.get('portfolio_correlation', 0) < 0.7 else '✗'}

---
*Generated by Risk Management System at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""

        return report

    async def _send_daily_report_email(self, report_content: str) -> bool:
        """Send daily report via email."""
        try:
            # Validate email configuration
            if not self.notification_config.email_from:
                logger.warning("Email from address not configured")
                return False

            if not self.notification_config.email_to:
                logger.warning("Email to addresses not configured")
                return False

            msg = MIMEMultipart("alternative")
            msg["Subject"] = (
                f"Daily Risk Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            )
            msg["From"] = self.notification_config.email_from
            msg["To"] = ", ".join(self.notification_config.email_to)

            # Convert markdown to basic HTML
            html_content = self._markdown_to_html(report_content)

            text_part = MIMEText(report_content, "plain")
            html_part = MIMEText(html_content, "html")

            msg.attach(text_part)
            msg.attach(html_part)

            # Validate SMTP configuration
            if not self.notification_config.email_smtp_host:
                logger.warning("SMTP host not configured")
                return False

            if (
                not self.notification_config.email_username
                or not self.notification_config.email_password
            ):
                logger.warning("SMTP credentials not configured")
                return False

            # Send email
            with smtplib.SMTP(
                self.notification_config.email_smtp_host,
                self.notification_config.email_smtp_port,
            ) as server:
                server.starttls()
                server.login(
                    self.notification_config.email_username,
                    self.notification_config.email_password,
                )
                text = msg.as_string()
                server.sendmail(
                    self.notification_config.email_from,
                    self.notification_config.email_to,
                    text,
                )

            logger.info("Daily risk report sent successfully")
            return True

        except Exception as e:
            logger.error(f"Error sending daily report email: {e}")
            return False

    def _markdown_to_html(self, markdown_content: str) -> str:
        """Convert basic markdown to HTML."""
        html = markdown_content

        # Replace headers
        html = html.replace("# ", "<h1>").replace("\n", "</h1>\n", 1)
        html = html.replace("## ", "<h2>").replace("\n", "</h2>\n")
        html = html.replace("### ", "<h3>").replace("\n", "</h3>\n")

        # Replace bold
        import re

        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)

        # Replace line breaks
        html = html.replace("\n", "<br>\n")

        # Wrap in basic HTML structure
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .footer {{ font-size: 12px; color: #999; margin-top: 20px; }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""

    async def send_emergency_alert(self, event: RiskEvent) -> bool:
        """Send emergency alert for critical events."""

        alert = RiskAlert(
            alert_type=event.event_type,
            severity=RiskSeverity.CRITICAL,
            symbol=event.symbol,
            title=f"EMERGENCY: {event.event_type.replace('_', ' ').title()}",
            message=f"CRITICAL RISK EVENT: {event.description}",
            action_required=True,
            metadata=event.metadata,
        )

        # Send immediately to all channels
        await self._process_alert_immediately(alert)

        return True

    def get_alert_statistics(self) -> Dict:
        """Get alert statistics."""

        # Count alerts by severity
        severity_counts: Dict[str, int] = {}
        for alert in self.alert_history:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        # Count alerts by type
        type_counts: Dict[str, int] = {}
        for alert in self.alert_history:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1

        return {
            "total_alerts": len(self.alert_history),
            "alerts_by_severity": severity_counts,
            "alerts_by_type": type_counts,
            "active_cooldowns": len(self.last_notification_times),
            "queue_size": self.alert_queue.qsize(),
        }

    async def process_alert_queue(self) -> None:
        """Process queued alerts in batches."""
        batch = []
        last_batch_time = datetime.now(timezone.utc)

        while True:
            try:
                # Wait for alert or timeout
                try:
                    alert = await asyncio.wait_for(
                        self.alert_queue.get(),
                        timeout=self.batch_timeout.total_seconds(),
                    )
                    batch.append(alert)
                except asyncio.TimeoutError:
                    # Process current batch on timeout
                    if batch:
                        await self._send_alert_batch(batch)
                        batch.clear()
                    continue

                # Process batch if full or timeout reached
                time_since_batch = datetime.now(timezone.utc) - last_batch_time
                if (
                    len(batch) >= self.batch_size
                    or time_since_batch >= self.batch_timeout
                ):
                    await self._send_alert_batch(batch)
                    batch.clear()
                    last_batch_time = datetime.now(timezone.utc)

            except Exception as e:
                logger.error(f"Error processing alert queue: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    def clear_alert_history(self) -> None:
        """Clear alert history (for testing or maintenance)."""
        self.alert_history.clear()
        self.last_notification_times.clear()
        logger.info("Alert history cleared")

    async def test_notifications(self) -> Dict[str, bool]:
        """Test all notification channels."""
        results = {}

        test_alert = RiskAlert(
            alert_type=RiskEventType.EMERGENCY_STOP,
            severity=RiskSeverity.LOW,
            symbol="TEST",
            title="Test Alert",
            message="This is a test alert from the Risk Management System",
            action_required=False,
            metadata={"test": True},
        )

        # Test Gotify
        if (
            self.notification_config.gotify_url
            and self.notification_config.gotify_token
        ):
            results["gotify"] = await self._send_gotify_alert(test_alert)

        # Test Slack
        if self.notification_config.slack_webhook_url:
            results["slack"] = await self._send_slack_alert(test_alert)

        # Test Email
        if (
            self.notification_config.email_smtp_host
            and self.notification_config.email_to
        ):
            results["email"] = await self._send_email_alert(test_alert)

        return results
