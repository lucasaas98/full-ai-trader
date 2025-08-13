# Notification Service

## Overview

The Notification Service is a critical component of the Full AI Trader system that monitors Redis event streams and sends notifications through various channels (primarily Gotify). It subscribes to trade executions, errors, alerts, and other system events to keep users informed about important trading activities.

## Architecture

The service follows an event-driven architecture:

```
Trading Services → Redis Pub/Sub → Notification Service → Gotify/Other Channels
```

### Key Features

- **Event-Driven**: Subscribes to Redis channels for real-time event processing
- **Multi-Channel Support**: Currently supports Gotify, extensible for Slack, Email, etc.
- **Rate Limiting**: Prevents notification spam with configurable cooldown periods
- **Resilient**: Automatic reconnection and error recovery
- **Async Processing**: Non-blocking event handling for high throughput

## Events Monitored

### Trade Executions (`executions:all`)
- Successful trade executions
- Order fills and partial fills
- Trade details (symbol, side, quantity, price)

### Execution Errors (`execution_errors:all`)
- Failed trade attempts
- Order rejections
- API errors

### Risk Alerts (`risk:alerts`)
- Position size warnings
- Drawdown alerts
- Exposure limit violations

### System Status (`system:status`)
- Service health changes
- Component failures
- Maintenance notifications

### Market Alerts (`market:alerts`)
- Unusual market conditions
- Volatility spikes
- Trading halts

### Portfolio Updates (`portfolio:updates`)
- Daily summaries
- Significant P&L changes
- Position changes

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379

# Gotify Configuration
GOTIFY_URL=http://your-gotify-server:80
GOTIFY_TOKEN=your-gotify-app-token

# Notification Settings
NOTIFICATION_COOLDOWN=60  # Seconds between similar notifications
LOG_LEVEL=INFO
TZ=UTC

# Daily Summary Time (24-hour format)
DAILY_SUMMARY_HOUR=21  # 9 PM UTC
```

### Rate Limiting

The service implements intelligent rate limiting to prevent notification fatigue:

- **Per-Symbol Cooldown**: Prevents multiple notifications for the same symbol within the cooldown period
- **Error Deduplication**: Groups similar errors to avoid spam
- **Priority-Based**: Critical alerts bypass rate limiting

## Installation

### Docker (Recommended)

1. Build the image:
```bash
docker-compose build notification-service
```

2. Start the service:
```bash
docker-compose up -d notification-service
```

### Manual Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export REDIS_URL=redis://localhost:6379
export GOTIFY_URL=http://your-gotify-server:80
export GOTIFY_TOKEN=your-token
```

3. Run the service:
```bash
python src/main.py
```

## Notification Examples

### Trade Execution
```
📈 Trade Executed: AAPL
Symbol: AAPL
Side: BUY
Quantity: 100
Price: $150.50
Value: $15,050.00
Strategy: AI_MOMENTUM
```

### Execution Error
```
❌ Trade Execution Failed: TSLA
Failed to execute trade for TSLA
Strategy: MEAN_REVERSION
Error: Insufficient buying power
Time: 2024-01-15 14:30:00 UTC
```

### Risk Alert
```
⚠️ RISK ALERT
Portfolio drawdown exceeds threshold
Current: -8.5%
Threshold: -7.0%
Action: Reducing position sizes
```

### Daily Summary
```
📊 Daily Trading Summary
Total Trades: 15
Winners: 10 (66.7%)
Losers: 5 (33.3%)
Net P&L: +$2,450.00
Best Trade: NVDA +$850
Worst Trade: META -$320
```

## Monitoring

### Health Check

The service exposes health status through:
- Redis connection status
- Gotify connection status
- Message processing metrics

### Logs

Logs are written to:
- Console (stdout)
- File: `/app/logs/notification_service.log` (in Docker)

### Metrics

Key metrics tracked:
- Events received
- Notifications sent
- Notification failures
- Rate limit hits
- Processing latency

## Troubleshooting

### No Notifications Received

1. Check Gotify configuration:
```bash
curl -X GET "$GOTIFY_URL/health" -H "X-Gotify-Key: $GOTIFY_TOKEN"
```

2. Verify Redis connection:
```bash
redis-cli ping
```

3. Check service logs:
```bash
docker logs trading-notification-service
```

### Rate Limiting Issues

If legitimate notifications are being rate-limited:
1. Adjust `NOTIFICATION_COOLDOWN` environment variable
2. Implement notification batching for high-frequency events

### Connection Failures

The service implements automatic reconnection with exponential backoff. If connections repeatedly fail:
1. Verify network connectivity
2. Check Redis/Gotify service status
3. Review firewall rules

## Development

### Adding New Event Handlers

1. Subscribe to new channel in `__init__`:
```python
self.channels.append("new:channel")
```

2. Add handler method:
```python
async def _handle_new_event(self, data: Dict[str, Any]):
    # Process event
    await self.notification_manager.send_notification(...)
```

3. Route events in `_handle_message`:
```python
elif channel == "new:channel":
    await self._handle_new_event(data)
```

### Testing

Run tests:
```bash
pytest tests/test_notification_service.py
```

Test with mock events:
```bash
redis-cli PUBLISH "executions:all" '{"symbol":"TEST","success":true,"result":{"side":"BUY","quantity":100,"price":50.0}}'
```

## Integration with Trading System

The notification service integrates seamlessly with:
- **Trade Executor**: Monitors execution results
- **Risk Manager**: Receives risk alerts
- **Strategy Engine**: Tracks signal generation
- **Portfolio Manager**: Reports position changes

## Performance Considerations

- **Async Processing**: All event handling is non-blocking
- **Connection Pooling**: Redis connections are pooled and reused
- **Batch Processing**: High-frequency events can be batched
- **Memory Management**: Event history is limited to prevent memory leaks

## Security

- **Token Security**: Gotify tokens should be stored securely (use secrets management)
- **Network Security**: Use TLS for Redis connections in production
- **Input Validation**: All event data is validated before processing
- **Rate Limiting**: Prevents notification-based DoS attacks

## Future Enhancements

- [ ] Support for additional notification channels (Slack, Discord, Telegram)
- [ ] Customizable notification templates
- [ ] User-specific notification preferences
- [ ] Notification scheduling and quiet hours
- [ ] Rich media notifications (charts, graphs)
- [ ] Two-way communication (acknowledge alerts)
- [ ] Notification history and analytics

## Support

For issues or questions:
1. Check the logs for error messages
2. Review this README and configuration
3. Check the main project documentation
4. Open an issue in the project repository