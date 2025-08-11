# AI Trading System - API Documentation

## Overview

This document provides comprehensive API documentation for the AI Trading System. The system consists of multiple microservices that communicate via REST APIs and Redis pub/sub messaging.

## Base URL

- **Development**: `http://localhost:8000`
- **Staging**: `https://staging-api.trading-system.com`
- **Production**: `https://api.trading-system.com`

## Authentication

All API endpoints require authentication using API keys or JWT tokens.

### API Key Authentication

Include the API key in the request header:

```http
Authorization: Bearer <your-api-key>
```

### JWT Token Authentication

For user-specific operations:

```http
Authorization: JWT <your-jwt-token>
```

## Services and Endpoints

### 1. Data Collector Service (Port 8001)

#### Get Market Data

```http
GET /api/v1/market-data/{symbol}
```

**Parameters:**
- `symbol` (string): Stock symbol (e.g., "AAPL", "GOOGL")

**Query Parameters:**
- `timeframe` (string, optional): Timeframe for data ("1m", "5m", "1h", "1d")
- `limit` (integer, optional): Number of records to return (default: 100, max: 1000)
- `start_date` (string, optional): Start date in ISO format
- `end_date` (string, optional): End date in ISO format

**Response:**
```json
{
  "symbol": "AAPL",
  "data": [
    {
      "timestamp": "2024-01-15T14:30:00Z",
      "open": 150.25,
      "high": 150.75,
      "low": 149.80,
      "close": 150.50,
      "volume": 1000000,
      "adjusted_close": 150.50
    }
  ],
  "metadata": {
    "count": 100,
    "timeframe": "5m",
    "last_updated": "2024-01-15T14:35:00Z"
  }
}
```

#### Get Real-time Quote

```http
GET /api/v1/quote/{symbol}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 150.50,
  "bid": 150.45,
  "ask": 150.55,
  "volume": 1000000,
  "timestamp": "2024-01-15T14:30:00Z",
  "change": 0.25,
  "change_percent": 0.17
}
```

#### Subscribe to Real-time Data

```http
POST /api/v1/subscriptions
```

**Request Body:**
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "data_types": ["quotes", "trades", "level2"],
  "callback_url": "https://your-system.com/webhook"
}
```

**Response:**
```json
{
  "subscription_id": "sub_123456789",
  "status": "active",
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "created_at": "2024-01-15T14:30:00Z"
}
```

### 2. Strategy Engine Service (Port 8002)

#### Get Available Strategies

```http
GET /api/v1/strategies
```

**Response:**
```json
{
  "strategies": [
    {
      "id": "momentum_strategy",
      "name": "Momentum Strategy",
      "description": "RSI and moving average crossover strategy",
      "parameters": {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "sma_short": 20,
        "sma_long": 50
      },
      "status": "active",
      "performance": {
        "total_return": 0.156,
        "sharpe_ratio": 1.23,
        "max_drawdown": -0.08,
        "win_rate": 0.67
      }
    }
  ]
}
```

#### Generate Trading Signals

```http
POST /api/v1/strategies/{strategy_id}/signals
```

**Request Body:**
```json
{
  "symbols": ["AAPL", "GOOGL"],
  "market_data": {
    "AAPL": {
      "price": 150.50,
      "volume": 1000000,
      "timestamp": "2024-01-15T14:30:00Z"
    }
  },
  "parameters": {
    "rsi_period": 14,
    "confidence_threshold": 0.7
  }
}
```

**Response:**
```json
{
  "signals": [
    {
      "signal_id": "sig_123456789",
      "symbol": "AAPL",
      "signal_type": "BUY",
      "strength": "STRONG",
      "confidence": 0.85,
      "price": 150.50,
      "timestamp": "2024-01-15T14:30:00Z",
      "strategy_id": "momentum_strategy",
      "metadata": {
        "rsi": 25.5,
        "sma_crossover": true,
        "volume_confirmation": true
      },
      "expiry": "2024-01-15T15:00:00Z"
    }
  ]
}
```

#### Update Strategy Parameters

```http
PUT /api/v1/strategies/{strategy_id}/parameters
```

**Request Body:**
```json
{
  "parameters": {
    "rsi_period": 21,
    "rsi_oversold": 25,
    "rsi_overbought": 75
  }
}
```

#### Get Strategy Performance

```http
GET /api/v1/strategies/{strategy_id}/performance
```

**Query Parameters:**
- `start_date` (string): Start date for performance calculation
- `end_date` (string): End date for performance calculation
- `benchmark` (string, optional): Benchmark symbol (default: "SPY")

**Response:**
```json
{
  "strategy_id": "momentum_strategy",
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  },
  "performance": {
    "total_return": 0.156,
    "annualized_return": 0.234,
    "volatility": 0.18,
    "sharpe_ratio": 1.23,
    "sortino_ratio": 1.45,
    "max_drawdown": -0.08,
    "calmar_ratio": 2.92,
    "win_rate": 0.67,
    "profit_factor": 1.85,
    "average_win": 245.50,
    "average_loss": -132.25,
    "total_trades": 156,
    "winning_trades": 104,
    "losing_trades": 52
  },
  "benchmark_comparison": {
    "benchmark_return": 0.089,
    "alpha": 0.067,
    "beta": 0.85,
    "correlation": 0.72,
    "tracking_error": 0.12
  }
}
```

### 3. Risk Manager Service (Port 8003)

#### Assess Trading Signal Risk

```http
POST /api/v1/risk/assess-signal
```

**Request Body:**
```json
{
  "signal": {
    "symbol": "AAPL",
    "signal_type": "BUY",
    "strength": "STRONG",
    "price": 150.50,
    "quantity": 100,
    "strategy_id": "momentum_strategy"
  },
  "portfolio": {
    "total_value": 100000.00,
    "cash_balance": 25000.00,
    "positions": {
      "AAPL": {
        "quantity": 200,
        "avg_cost": 145.00,
        "market_value": 30100.00
      }
    }
  }
}
```

**Response:**
```json
{
  "assessment_id": "risk_123456789",
  "approved": true,
  "risk_score": 0.35,
  "position_size": 1500.00,
  "max_position_size": 5000.00,
  "risk_metrics": {
    "portfolio_var_95": 8500.00,
    "concentration_risk": 0.45,
    "correlation_risk": 0.62,
    "volatility_adjusted_size": 1200.00
  },
  "risk_limits": {
    "max_position_concentration": 0.15,
    "max_sector_concentration": 0.40,
    "max_daily_loss": 5000.00,
    "max_portfolio_leverage": 1.5
  },
  "stop_loss": 145.00,
  "take_profit": 165.00,
  "metadata": {
    "assessment_time_ms": 45,
    "rules_evaluated": 12,
    "warnings": []
  }
}
```

#### Get Portfolio Risk Metrics

```http
GET /api/v1/risk/portfolio-metrics
```

**Response:**
```json
{
  "portfolio_id": "portfolio_123",
  "risk_metrics": {
    "value_at_risk_95": 8500.00,
    "value_at_risk_99": 12750.00,
    "expected_shortfall": 15200.00,
    "portfolio_volatility": 0.18,
    "sharpe_ratio": 1.23,
    "max_drawdown": 0.08,
    "beta": 0.85,
    "correlation_spy": 0.72
  },
  "position_risks": {
    "AAPL": {
      "position_var": 2100.00,
      "concentration_risk": 0.30,
      "correlation_risk": 0.45
    }
  },
  "risk_limits_status": {
    "within_limits": true,
    "limit_utilization": {
      "max_position_size": 0.60,
      "max_sector_exposure": 0.75,
      "max_daily_loss": 0.25
    }
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

#### Set Risk Limits

```http
PUT /api/v1/risk/limits
```

**Request Body:**
```json
{
  "limits": {
    "max_position_concentration": 0.15,
    "max_sector_concentration": 0.40,
    "max_daily_loss": 5000.00,
    "max_portfolio_leverage": 1.5,
    "stop_loss_threshold": 0.05,
    "correlation_limit": 0.80
  }
}
```

### 4. Trade Executor Service (Port 8004)

#### Execute Trade Order

```http
POST /api/v1/orders
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "order_type": "MARKET",
  "side": "BUY",
  "quantity": 100,
  "price": 150.50,
  "time_in_force": "DAY",
  "strategy_id": "momentum_strategy",
  "signal_id": "sig_123456789",
  "risk_assessment_id": "risk_123456789",
  "metadata": {
    "stop_loss": 145.00,
    "take_profit": 165.00,
    "max_slippage": 0.002
  }
}
```

**Response:**
```json
{
  "order_id": "ord_123456789",
  "status": "PENDING",
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100,
  "order_type": "MARKET",
  "submitted_at": "2024-01-15T14:30:00Z",
  "estimated_fill_price": 150.52,
  "estimated_commission": 1.00,
  "time_in_force": "DAY",
  "expires_at": "2024-01-15T20:00:00Z"
}
```

#### Get Order Status

```http
GET /api/v1/orders/{order_id}
```

**Response:**
```json
{
  "order_id": "ord_123456789",
  "status": "FILLED",
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100,
  "filled_quantity": 100,
  "remaining_quantity": 0,
  "average_fill_price": 150.52,
  "total_commission": 1.00,
  "submitted_at": "2024-01-15T14:30:00Z",
  "filled_at": "2024-01-15T14:30:15Z",
  "fills": [
    {
      "fill_id": "fill_123",
      "quantity": 100,
      "price": 150.52,
      "timestamp": "2024-01-15T14:30:15Z",
      "venue": "NASDAQ"
    }
  ]
}
```

#### Get Trade History

```http
GET /api/v1/trades
```

**Query Parameters:**
- `symbol` (string, optional): Filter by symbol
- `start_date` (string, optional): Start date filter
- `end_date` (string, optional): End date filter
- `strategy` (string, optional): Filter by strategy ID
- `limit` (integer, optional): Number of records (default: 100, max: 1000)
- `offset` (integer, optional): Pagination offset

**Response:**
```json
{
  "trades": [
    {
      "trade_id": "trade_123456789",
      "order_id": "ord_123456789",
      "symbol": "AAPL",
      "side": "BUY",
      "quantity": 100,
      "executed_price": 150.52,
      "commission": 1.00,
      "executed_at": "2024-01-15T14:30:15Z",
      "strategy_id": "momentum_strategy",
      "pnl": 0.00,
      "status": "FILLED"
    }
  ],
  "pagination": {
    "total": 1250,
    "limit": 100,
    "offset": 0,
    "has_next": true
  }
}
```

#### Cancel Order

```http
DELETE /api/v1/orders/{order_id}
```

**Response:**
```json
{
  "order_id": "ord_123456789",
  "status": "CANCELLED",
  "cancelled_at": "2024-01-15T14:35:00Z",
  "reason": "User requested cancellation"
}
```

### 5. Portfolio Manager Service (Port 8005)

#### Get Portfolio Summary

```http
GET /api/v1/portfolio
```

**Response:**
```json
{
  "account_id": "account_123",
  "total_value": 125750.00,
  "cash_balance": 15250.00,
  "invested_value": 110500.00,
  "day_pnl": 1250.00,
  "day_pnl_percent": 1.01,
  "total_pnl": 25750.00,
  "total_pnl_percent": 25.75,
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 200,
      "avg_cost": 145.00,
      "current_price": 150.50,
      "market_value": 30100.00,
      "unrealized_pnl": 1100.00,
      "unrealized_pnl_percent": 3.79,
      "day_pnl": 50.00,
      "allocation_percent": 23.95
    }
  ],
  "performance": {
    "sharpe_ratio": 1.23,
    "max_drawdown": -0.08,
    "volatility": 0.18,
    "beta": 0.85
  },
  "last_updated": "2024-01-15T14:30:00Z"
}
```

#### Get Position Details

```http
GET /api/v1/portfolio/positions/{symbol}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "quantity": 200,
  "avg_cost": 145.00,
  "current_price": 150.50,
  "market_value": 30100.00,
  "unrealized_pnl": 1100.00,
  "unrealized_pnl_percent": 3.79,
  "day_pnl": 50.00,
  "allocation_percent": 23.95,
  "cost_basis": 29000.00,
  "trades": [
    {
      "trade_id": "trade_123",
      "quantity": 100,
      "price": 145.00,
      "executed_at": "2024-01-10T10:30:00Z"
    }
  ],
  "risk_metrics": {
    "var_95": 2100.00,
    "volatility": 0.22,
    "beta": 1.15
  }
}
```

#### Rebalance Portfolio

```http
POST /api/v1/portfolio/rebalance
```

**Request Body:**
```json
{
  "target_allocation": {
    "AAPL": 0.25,
    "GOOGL": 0.25,
    "MSFT": 0.25,
    "NVDA": 0.25
  },
  "rebalance_threshold": 0.05,
  "execution_strategy": "gradual"
}
```

**Response:**
```json
{
  "rebalance_id": "rebal_123456789",
  "status": "pending",
  "target_allocation": {
    "AAPL": 0.25,
    "GOOGL": 0.25,
    "MSFT": 0.25,
    "NVDA": 0.25
  },
  "required_trades": [
    {
      "symbol": "AAPL",
      "action": "SELL",
      "quantity": 50,
      "estimated_value": 7525.00
    }
  ],
  "estimated_cost": 4.00,
  "estimated_completion": "2024-01-15T15:00:00Z"
}
```

### 6. Scheduler Service (Port 8006)

#### Get Scheduled Jobs

```http
GET /api/v1/scheduler/jobs
```

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job_data_collection",
      "name": "Market Data Collection",
      "schedule": "*/1 * * * *",
      "next_run": "2024-01-15T14:31:00Z",
      "last_run": "2024-01-15T14:30:00Z",
      "status": "active",
      "success_rate": 0.99,
      "average_duration_ms": 250
    }
  ]
}
```

#### Create Scheduled Job

```http
POST /api/v1/scheduler/jobs
```

**Request Body:**
```json
{
  "name": "Custom Strategy Execution",
  "schedule": "0 9-16 * * 1-5",
  "task": "execute_strategy",
  "parameters": {
    "strategy_id": "momentum_strategy",
    "symbols": ["AAPL", "GOOGL"]
  },
  "timezone": "US/Eastern",
  "enabled": true
}
```

#### Update Job Status

```http
PUT /api/v1/scheduler/jobs/{job_id}
```

**Request Body:**
```json
{
  "enabled": false,
  "schedule": "0 10-15 * * 1-5"
}
```

### 7. Backtesting Service (Port 8007)

#### Run Backtest

```http
POST /api/v1/backtesting/run
```

**Request Body:**
```json
{
  "strategy_config": {
    "strategy_id": "momentum_strategy",
    "parameters": {
      "rsi_period": 14,
      "rsi_oversold": 30,
      "rsi_overbought": 70
    }
  },
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 100000.00,
  "commission": 1.00,
  "benchmark": "SPY"
}
```

**Response:**
```json
{
  "backtest_id": "bt_123456789",
  "status": "running",
  "estimated_completion": "2024-01-15T14:40:00Z",
  "progress": 0.15
}
```

#### Get Backtest Results

```http
GET /api/v1/backtesting/results/{backtest_id}
```

**Response:**
```json
{
  "backtest_id": "bt_123456789",
  "status": "completed",
  "strategy": {
    "strategy_id": "momentum_strategy",
    "parameters": {
      "rsi_period": 14,
      "rsi_oversold": 30,
      "rsi_overbought": 70
    }
  },
  "period": {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  },
  "performance": {
    "total_return": 0.234,
    "annualized_return": 0.234,
    "volatility": 0.16,
    "sharpe_ratio": 1.46,
    "max_drawdown": -0.12,
    "calmar_ratio": 1.95,
    "win_rate": 0.65,
    "profit_factor": 1.73,
    "total_trades": 287,
    "final_portfolio_value": 123400.00
  },
  "benchmark_comparison": {
    "benchmark_return": 0.196,
    "alpha": 0.038,
    "beta": 0.92,
    "information_ratio": 0.31
  },
  "risk_analysis": {
    "var_95": 7200.00,
    "var_99": 11500.00,
    "expected_shortfall": 13800.00,
    "worst_month": -0.08,
    "best_month": 0.12
  },
  "trade_analysis": {
    "average_holding_period": 5.2,
    "largest_win": 1250.00,
    "largest_loss": -890.00,
    "consecutive_wins": 8,
    "consecutive_losses": 3
  }
}
```

#### Run Monte Carlo Simulation

```http
POST /api/v1/backtesting/monte-carlo
```

**Request Body:**
```json
{
  "strategy_config": {
    "strategy_id": "momentum_strategy"
  },
  "simulation_config": {
    "num_simulations": 1000,
    "time_horizon_days": 252,
    "confidence_levels": [0.95, 0.99],
    "correlation_model": "empirical"
  },
  "initial_portfolio_value": 100000.00
}
```

## WebSocket APIs

### Real-time Market Data Stream

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/market-data');

ws.onopen = function() {
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['AAPL', 'GOOGL'],
    data_types: ['quotes', 'trades']
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  // Handle real-time market data
};
```

### Real-time Trading Signals

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/signals');

ws.onmessage = function(event) {
  const signal = JSON.parse(event.data);
  /*
  {
    "signal_id": "sig_123456789",
    "symbol": "AAPL",
    "signal_type": "BUY",
    "strength": "STRONG",
    "confidence": 0.85,
    "timestamp": "2024-01-15T14:30:00Z"
  }
  */
};
```

### Portfolio Updates Stream

```javascript
const ws = new WebSocket('ws://localhost:8005/ws/portfolio');

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  /*
  {
    "type": "position_update",
    "symbol": "AAPL",
    "new_quantity": 200,
    "new_market_value": 30100.00,
    "pnl_change": 50.00,
    "timestamp": "2024-01-15T14:30:00Z"
  }
  */
};
```

## Error Responses

All API endpoints return standardized error responses:

```json
{
  "error": {
    "code": "INSUFFICIENT_FUNDS",
    "message": "Insufficient cash balance for trade execution",
    "details": {
      "required_amount": 15050.00,
      "available_balance": 12000.00,
      "shortfall": 3050.00
    },
    "timestamp": "2024-01-15T14:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_SYMBOL` | Invalid or unsupported trading symbol |
| `INSUFFICIENT_FUNDS` | Insufficient cash balance |
| `RISK_LIMIT_EXCEEDED` | Trade would exceed risk limits |
| `MARKET_CLOSED` | Market is currently closed |
| `INVALID_PARAMETERS` | Invalid request parameters |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `STRATEGY_NOT_FOUND` | Specified strategy does not exist |
| `UNAUTHORIZED` | Invalid or expired authentication |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable |
| `VALIDATION_ERROR` | Request validation failed |

## Rate Limiting

API endpoints are rate limited to prevent abuse:

- **Market Data**: 1000 requests per minute
- **Trading Operations**: 100 requests per minute
- **Portfolio Queries**: 500 requests per minute
- **WebSocket Connections**: 10 concurrent connections per API key

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 856
X-RateLimit-Reset: 1642234567
```

## Webhooks

### Trade Execution Webhook

Configure webhook URL to receive trade execution notifications:

```json
{
  "webhook_id": "webhook_123",
  "event_type": "trade.executed",
  "payload": {
    "trade_id": "trade_123456789",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 100,
    "executed_price": 150.52,
    "executed_at": "2024-01-15T14:30:15Z",
    "strategy_id": "momentum_strategy"
  },
  "timestamp": "2024-01-15T14:30:15Z",
  "signature": "sha256=abc123..."
}
```

### Risk Alert Webhook

```json
{
  "webhook_id": "webhook_124",
  "event_type": "risk.alert",
  "payload": {
    "alert_type": "position_limit_approached",
    "symbol": "AAPL",
    "current_allocation": 0.28,
    "limit": 0.30,
    "risk_score": 0.85,
    "recommended_action": "reduce_position"
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

## Data Models

### Market Data Model

```json
{
  "symbol": "string",
  "timestamp": "datetime",
  "price": "decimal",
  "volume": "integer",
  "bid": "decimal",
  "ask": "decimal",
  "spread": "decimal",
  "high_52w": "decimal",
  "low_52w": "decimal"
}
```

### Trading Signal Model

```json
{
  "signal_id": "string",
  "symbol": "string",
  "signal_type": "BUY|SELL|HOLD",
  "strength": "WEAK|MODERATE|STRONG|CRITICAL",
  "confidence": "float (0-1)",
  "price": "decimal",
  "timestamp": "datetime",
  "strategy_id": "string",
  "expiry": "datetime",
  "metadata": "object"
}
```

### Order Model

```json
{
  "order_id": "string",
  "symbol": "string",
  "order_type": "MARKET|LIMIT|STOP|STOP_LIMIT",
  "side": "BUY|SELL",
  "quantity": "integer",
  "price": "decimal",
  "status": "PENDING|ACCEPTED|FILLED|CANCELLED|REJECTED",
  "time_in_force": "DAY|GTC|IOC|FOK",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### Trade Model

```json
{
  "trade_id": "string",
  "order_id": "string",
  "symbol": "string",
  "side": "BUY|SELL",
  "quantity": "integer",
  "executed_price": "decimal",
  "commission": "decimal",
  "executed_at": "datetime",
  "strategy_id": "string",
  "pnl": "decimal"
}
```

## SDK Examples

### Python SDK Usage

```python
from trading_system_sdk import TradingSystemClient

# Initialize client
client = TradingSystemClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Get market data
market_data = await client.get_market_data("AAPL", timeframe="5m")

# Generate trading signal
signal = await client.generate_signal(
    strategy_id="momentum_strategy",
    symbol="AAPL",
    market_data=market_data
)

# Execute trade if signal is strong
if signal.strength == "STRONG" and signal.confidence > 0.8:
    risk_assessment = await client.assess_risk(signal)
    
    if risk_assessment.approved:
        order = await client.execute_trade(
            symbol=signal.symbol,
            side=signal.signal_type,
            quantity=100,
            order_type="MARKET"
        )
        
        print(f"Order executed: {order.order_id}")
```

### JavaScript SDK Usage

```javascript
import { TradingSystemClient } from 'trading-system-sdk';

const client = new TradingSystemClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Get portfolio summary
const portfolio = await client.getPortfolio();
console.log(`Total value: $${portfolio.total_value}`);

// Subscribe to real-time signals
client.subscribeToSignals(['AAPL', 'GOOGL'], (signal) => {
  console.log(`New signal: ${signal.signal_type} ${signal.symbol}`);
});
```

## Testing APIs

### Test Data Endpoints

```http
GET /api/v1/test/market-data/{symbol}
```

Returns mock market data for testing purposes.

### Simulation Mode

Add header to enable simulation mode:

```http
X-Trading-Mode: simulation
```

### Health Check Endpoints

All services provide health check endpoints:

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "data_collector",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "external_apis": "healthy"
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

## Monitoring and Metrics

### Prometheus Metrics Endpoint

```http
GET /metrics
```

Returns Prometheus-formatted metrics for monitoring.

### Custom Metrics

#### Trading Metrics
- `trading_trades_executed_total{symbol, strategy, side}`
- `trading_trade_execution_latency_ms{symbol, venue}`
- `trading_portfolio_total_value`
- `trading_portfolio_daily_pnl`
- `trading_position_market_value{symbol}`
- `trading_signal_generation_duration_ms{strategy}`
- `trading_strategy_success_rate{strategy}`

#### System Metrics
- `trading_api_request_duration_seconds{endpoint, method}`
- `trading_api_requests_total{endpoint, method, status}`
- `trading_db_query_duration_seconds{query_type}`
- `trading_redis_operations_duration_seconds{operation}`
- `trading_system_memory_usage_bytes{service}`
- `trading_system_cpu_usage_percent{service}`

#### Risk Metrics
- `trading_portfolio_var_95`
- `trading_portfolio_volatility`
- `trading_risk_limit_utilization{limit_type}`
- `trading_position_concentration_risk{symbol}`

## Security

### API Key Management

API keys should be:
- Stored securely using environment variables
- Rotated regularly (recommended: every 90 days)
- Scoped to specific operations
- Monitored for unusual usage patterns

### Request Signing

For high-security operations, requests must be signed:

```python
import hmac
import hashlib
import time

def sign_request(api_secret, method, path, body=""):
    timestamp = str(int(time.time()))
    message = f"{timestamp}{method}{path}{body}"
    signature = hmac.new(
        api_secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return {
        'X-Timestamp': timestamp,
        'X-Signature': signature
    }
```

### Rate Limiting

Rate limits are enforced per API key:

| Tier | Requests/Minute | WebSocket Connections |
|------|-----------------|----------------------|
| Basic | 1000 | 5 |
| Professional | 5000 | 25 |
| Enterprise | 20000 | 100 |

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
try:
    response = await client.execute_trade(order)
except TradingSystemError as e:
    if e.code == "INSUFFICIENT_FUNDS":
        # Handle insufficient funds
        pass
    elif e.code == "RISK_LIMIT_EXCEEDED":
        # Handle risk limit
        pass
    else:
        # Handle other errors
        pass
```

### 2. Idempotency

Use idempotency keys for critical operations:

```http
POST /api/v1/orders
Idempotency-Key: unique-key-123
```

### 3. Pagination

Use pagination for large result sets:

```http
GET /api/v1/trades?limit=100&offset=200
```

### 4. Caching

Leverage caching headers:

```http
Cache-Control: max-age=60
ETag: "abc123"
```

### 5. Compression

Enable compression for large responses:

```http
Accept-Encoding: gzip, deflate
```

## Development Environment

### Local Setup

1. Start the development environment:
```bash
docker-compose up -d
```

2. Run database migrations:
```bash
docker-compose exec api python manage.py migrate
```

3. Create test data:
```bash
docker-compose exec api python manage.py create_test_data
```

### API Testing

Use the provided Postman collection or test scripts:

```bash
# Run API tests
python scripts/test_api.py

# Load test
python scripts/load_test_api.py --concurrent=10 --duration=60
```

## Troubleshooting

### Common Issues

#### 1. 500 Internal Server Error
- Check service logs: `docker-compose logs [service-name]`
- Verify database connectivity
- Check Redis connectivity

#### 2. 429 Rate Limit Exceeded
- Reduce request frequency
- Implement exponential backoff
- Consider upgrading API tier

#### 3. 401 Unauthorized
- Verify API key is correct
- Check if API key has expired
- Ensure proper authentication headers

#### 4. WebSocket Connection Drops
- Implement reconnection logic
- Check network stability
- Verify WebSocket URL and authentication

### Debug Mode

Enable debug mode for detailed error information:

```http
X-Debug-Mode: true
```

### Support

For API support:
- Email: api-support@trading-system.com
- Slack: #api-support
- Documentation: https://docs.trading-system.com

## Changelog

### Version 1.2.0 (2024-01-15)
- Added Monte Carlo simulation endpoints
- Enhanced risk assessment APIs
- Improved WebSocket performance
- Added portfolio optimization endpoints

### Version 1.1.0 (2024-01-01)
- Added backtesting service APIs
- Enhanced market data streaming
- Improved error handling
- Added webhook support

### Version 1.0.0 (2023-12-01)
- Initial API release
- Core trading functionality
- Basic monitoring endpoints