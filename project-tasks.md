
#### Created by Claude Opus 4.1

# Automated Trading System - Task Breakdown & AI Agent Prompts

## System Architecture Overview
- **Microservices Architecture** using Docker Compose
- **Message Queue**: Redis (lightweight, you're familiar with it)
- **Database**: PostgreSQL for trade history, Parquet files for market data
- **Data Processing**: Polars for fast parallel processing
- **Monitoring**: Logging + Gotify for alerts

---

## Task 1: Project Foundation & Structure

### AI Agent Prompt:
```
Create a Python project structure for an automated trading system with the following requirements:

1. Create a monorepo structure with these services:
   - data_collector: Fetches data from TwelveData and FinViz
   - strategy_engine: Implements trading strategies
   - trade_executor: Handles Alpaca API operations
   - risk_manager: Manages position sizing and risk
   - scheduler: Orchestrates all services
   - database: PostgreSQL service configuration

2. Project structure should be:
```
trading-system/
├── docker-compose.yml
├── .env.example
├── README.md
├── shared/
│   ├── __init__.py
│   ├── models.py (Pydantic models for data validation)
│   ├── config.py (centralized configuration)
│   └── utils.py
├── services/
│   ├── data_collector/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   └── tests/
│   ├── strategy_engine/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   └── tests/
│   ├── trade_executor/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   └── tests/
│   ├── risk_manager/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   └── tests/
│   └── scheduler/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── src/
│       └── tests/
├── data/
│   ├── parquet/
│   └── logs/
└── scripts/
    ├── setup.sh
    └── backtest.py
```

3. Create docker-compose.yml with:
   - All services defined above
   - Redis for message queue
   - PostgreSQL for trade history
   - Proper networking and volume mounts
   - Environment variable configuration

4. Create base Dockerfile template that each service can extend

5. Setup .env.example with:
   - ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
   - TWELVE_DATA_API_KEY
   - FINVIZ_API_KEY
   - Database credentials
   - Redis configuration
   - Gotify URL and token

6. Create a Makefile with commands for:
   - Building all services
   - Running tests
   - Starting/stopping services
   - Viewing logs
   - Database migrations

7. Setup logging configuration that:
   - Uses Python's logging module
   - Outputs to both console and files
   - Includes log rotation
   - Has different levels for different modules

8. Create shared Pydantic models for:
   - Market data (OHLCV)
   - Trade signals
   - Portfolio state
   - Risk parameters

Include proper .gitignore and development setup instructions.
```

---

## Task 2: Data Collection Service

### AI Agent Prompt:
```
Create a data collection service for the trading system with these specifications:

1. **FinViz Elite Integration**:
   - Create a FinvizScreener class that:
     - Fetches data from: https://elite.finviz.com/export.ashx
     - Uses these screener parameters as default (but configurable):
       * Market cap: 2M to 2B (cap_2to)
       * Average volume: Over 1000K (sh_avgvol_o1000)
       * Current volume: Over 400K (sh_curvol_o400)
       * Price: $5 to $35 (sh_price_5to35)
       * Above SMA20 (ta_sma20_pa)
       * Weekly volatility over 6% (ta_volatility_wo6)
     - Implements rate limiting (can be called every 5 minutes)
     - Parses CSV response and returns structured data
     - Includes real-time price data from the screener
     - Handles top 10-20 tickers by volume

2. **TwelveData Integration**:
   - Create TwelveDataClient class that:
     - Fetches OHLCV data for multiple timeframes (5min, 15min, 1hour, 1day)
     - Downloads 2 years of historical data for backtesting
     - Implements batch requests to optimize API usage
     - Handles rate limiting properly
     - Supports both historical and real-time data modes

3. **Data Storage with Polars & Parquet**:
   - Create DataStore class that:
     - Saves all data as Parquet files using Polars
     - Organizes files by: data/parquet/{ticker}/{timeframe}/{date}.parquet
     - Implements efficient data updates (append new, don't rewrite)
     - Provides fast data retrieval methods
     - Handles data deduplication
     - Implements data validation and cleaning

4. **Redis Integration**:
   - Publish new ticker selections to Redis channel "tickers:new"
   - Publish price updates to "prices:{ticker}"
   - Cache frequently accessed data with TTL

5. **Scheduling**:
   - Run FinViz screener every 5 minutes
   - Update price data based on timeframe:
     - 5-min data: every 5 minutes during market hours
     - 15-min data: every 15 minutes
     - Hourly: every hour
     - Daily: at market close
   - Use APScheduler for task scheduling

6. **Data Quality**:
   - Implement data validation (check for missing values, outliers)
   - Handle market holidays and weekends
   - Detect and handle stock splits/dividends
   - Log all data anomalies

7. **Performance Requirements**:
   - Use asyncio for concurrent API calls
   - Leverage Polars for all data operations
   - Implement connection pooling for API requests
   - Cache recent data in memory

Create comprehensive error handling, retry logic, and logging throughout.
```

---

## Task 3: Strategy Engine

### AI Agent Prompt:
```
Build a flexible strategy engine that combines technical and fundamental analysis:

1. **Strategy Framework**:
   - Create abstract BaseStrategy class with:
     - analyze() method for generating signals
     - get_entry_signal() and get_exit_signal() methods
     - backtest() method for historical testing
     - Configuration through strategy parameters

2. **Technical Analysis Module**:
   - Implement indicators using Polars for speed:
     - Moving averages (SMA, EMA, WMA)
     - RSI, MACD, Bollinger Bands
     - Volume indicators (OBV, VWAP)
     - ATR for volatility
     - Support/Resistance levels
   - Create a TechnicalStrategy class that:
     - Combines multiple indicators
     - Generates confidence scores (0-100)
     - Identifies chart patterns

3. **Fundamental Analysis Module**:
   - Create FundamentalStrategy class that uses FinViz data:
     - P/E ratio analysis
     - Volume surge detection
     - Market cap considerations
     - Sector/industry performance
   - Score stocks based on fundamental health

4. **Combined Strategy**:
   - Create HybridStrategy that:
     - Weighs both TA and FA signals
     - Adjusts weights based on market conditions
     - Implements different modes:
       * Day trading: 70% TA, 30% FA
       * Swing trading: 50% TA, 50% FA
     - Generates final BUY/SELL/HOLD signals with confidence

5. **Signal Generation**:
   - Output signal format:
     ```python
     {
         "ticker": "AAPL",
         "action": "BUY",
         "confidence": 85,
         "strategy_type": "day_trade",
         "entry_price": 150.00,
         "stop_loss": 147.00,  # 2% default
         "take_profit": 153.00,  # 2% default, adjustable
         "position_size": 0.20,  # 20% of portfolio
         "reasoning": "Strong momentum + volume surge",
         "timestamp": "2024-01-01T10:30:00Z"
     }
     ```

6. **Market Regime Detection**:
   - Identify market conditions (trending, ranging, volatile)
   - Adjust strategy parameters accordingly
   - Implement regime filters to avoid trading in unfavorable conditions

7. **Backtesting Engine**:
   - Test strategies on historical data
   - Calculate metrics: Sharpe ratio, win rate, max drawdown
   - Optimize parameters using walk-forward analysis
   - Generate detailed reports

8. **Redis Integration**:
   - Subscribe to price updates
   - Publish signals to "signals:{ticker}" channel
   - Cache strategy states

Ensure all calculations use Polars for maximum performance. Include comprehensive logging of all decision points.
```

---

## Task 4: Risk Manager

### AI Agent Prompt:
```
Create a risk management service that protects the portfolio:

1. **Position Sizing**:
   - Implement fixed percentage sizing:
     - Default 20% per position (100% / 5 positions)
     - Adjust based on confidence score
     - Never exceed maximum allocation per trade
   - Track current portfolio value from Alpaca
   - Calculate exact share quantities

2. **Risk Parameters**:
   - Stop loss: Default 2% but configurable
   - Take profit: Default 3% (1.5:1 ratio) but adjustable
   - Implement trailing stops for winning positions
   - Maximum daily loss limit (e.g., 5% of portfolio)
   - Maximum position limit: 5 concurrent positions

3. **Portfolio Monitoring**:
   - Track all open positions
   - Calculate portfolio-wide metrics:
     - Total exposure
     - Beta-weighted delta
     - Correlation between positions
     - Value at Risk (VaR)
   - Monitor drawdown in real-time

4. **Risk Filters**:
   - Block trades if:
     - Daily loss limit reached
     - Position limit reached
     - Insufficient buying power
     - Stock correlation too high with existing positions
   - Emergency stop: halt all trading if portfolio drops X%

5. **Position Management**:
   - Adjust stop losses based on volatility (ATR)
   - Scale out of winning positions
   - Rebalance if position grows too large
   - Close positions before market close for day trading

6. **Database Integration**:
   - Store all risk events in PostgreSQL:
     ```sql
     CREATE TABLE risk_events (
         id SERIAL PRIMARY KEY,
         timestamp TIMESTAMPTZ,
         event_type VARCHAR(50),
         ticker VARCHAR(10),
         details JSONB,
         action_taken VARCHAR(100)
     );
     ```

7. **Alert System**:
   - Send Gotify notifications for:
     - Stop loss triggered
     - Daily loss limit approaching
     - Unusual market conditions
     - System errors

8. **Risk Reports**:
   - Generate daily risk reports
   - Track risk-adjusted returns
   - Monitor strategy performance by type

Implement circuit breakers and failsafes throughout. All risk checks must be performed before trade execution.
```

---

## Task 5: Trade Executor

### AI Agent Prompt:
```
Build a trade execution service using Alpaca API:

1. **Alpaca Integration**:
   - Create AlpacaClient class with:
     - Paper trading configuration initially
     - Order placement (market, limit, stop-loss)
     - Position management
     - Account info retrieval
     - Real-time position tracking

2. **Order Management**:
   - Implement OrderManager that:
     - Receives signals from strategy engine
     - Validates with risk manager
     - Places bracket orders (entry + stop loss + take profit)
     - Handles partial fills
     - Manages order cancellations
     - Implements retry logic for failed orders

3. **Execution Logic**:
   - Smart order routing:
     - Use limit orders during low volatility
     - Market orders for high confidence signals
     - Implement TWAP/VWAP execution for large orders
   - Slippage management
   - Monitor bid-ask spreads

4. **Position Tracking**:
   - Maintain position state in PostgreSQL:
     ```sql
     CREATE TABLE positions (
         id SERIAL PRIMARY KEY,
         ticker VARCHAR(10),
         entry_time TIMESTAMPTZ,
         entry_price DECIMAL(10,2),
         quantity INTEGER,
         stop_loss DECIMAL(10,2),
         take_profit DECIMAL(10,2),
         status VARCHAR(20),
         strategy_type VARCHAR(20),
         exit_time TIMESTAMPTZ,
         exit_price DECIMAL(10,2),
         pnl DECIMAL(10,2),
         INDEX idx_ticker (ticker),
         INDEX idx_status (status)
     );
     ```

5. **Trade Lifecycle**:
   - Entry: Validate → Place Order → Confirm Fill → Store Position
   - Management: Monitor Price → Adjust Stops → Track P&L
   - Exit: Trigger Exit → Place Order → Confirm Fill → Update Records

6. **Performance Tracking**:
   - Calculate and store:
     - Win/loss ratio
     - Average win/loss size
     - Profit factor
     - Expectancy
   - Export data in format compatible with TradeNote

7. **Error Handling**:
   - Handle API errors gracefully
   - Implement exponential backoff
   - Dead letter queue for failed trades
   - Manual intervention alerts

8. **Redis Integration**:
   - Subscribe to "signals:*" channels
   - Publish execution status to "executions:{ticker}"
   - Maintain execution queue

Ensure all monetary calculations use Decimal for precision. Include comprehensive audit logging.
```

---

## Task 6: Scheduler & Orchestration

### AI Agent Prompt:
```
Create a scheduler service that orchestrates all components:

1. **Market Hours Management**:
   - Track market hours and holidays
   - Start/stop services based on market status
   - Pre-market and after-hours handling
   - Weekend maintenance tasks

2. **Task Scheduling**:
   - Implement with APScheduler:
     - FinViz screener: every 5 minutes during market hours
     - Price updates: variable by timeframe
     - Strategy analysis: after each data update
     - Risk checks: continuous
     - EOD reports: daily at market close

3. **Service Orchestration**:
   - Health checks for all services
   - Dependency management
   - Graceful startup/shutdown sequences
   - Service recovery on failure

4. **Data Pipeline**:
   - Coordinate data flow:
     1. Screener → Ticker Selection
     2. Ticker Selection → Data Collection
     3. Data Collection → Strategy Analysis
     4. Strategy Analysis → Risk Check
     5. Risk Check → Trade Execution
   - Ensure proper sequencing

5. **System Monitoring**:
   - Monitor service health
   - Track API rate limits
   - System resource usage
   - Database connection pools
   - Redis queue lengths

6. **Maintenance Tasks**:
   - Data cleanup (old parquet files)
   - Database vacuum
   - Log rotation
   - Backup critical data
   - Update historical data

7. **CLI Interface**:
   - Commands for:
     - System status
     - Manual trade triggers
     - Strategy parameter updates
     - Export reports for TradeNote
     - Pause/resume trading
     - View current positions

8. **Configuration Management**:
   - Hot-reload configuration changes
   - A/B testing different strategies
   - Enable/disable specific strategies
   - Adjust risk parameters on the fly

Create robust error recovery and ensure system resilience.
```

---

## Task 7: Testing & Monitoring

### AI Agent Prompt:
```
Implement comprehensive testing and monitoring:

1. **Unit Tests**:
   - Test each component in isolation
   - Mock external APIs
   - Test edge cases and error conditions
   - Achieve 80%+ code coverage

2. **Integration Tests**:
   - Test service communication
   - Redis pub/sub functionality
   - Database operations
   - End-to-end trade flow

3. **Backtesting Framework**:
   - Historical simulation engine
   - Walk-forward analysis
   - Monte Carlo simulations
   - Strategy optimization
   - Generate performance reports

4. **Monitoring Setup**:
   - Prometheus metrics for:
     - API response times
     - Trade execution latency
     - Strategy performance
     - System resources
   - Grafana dashboards
   - Alert rules for anomalies

5. **Logging Strategy**:
   - Structured logging (JSON format)
   - Log aggregation setup
   - Search and analysis capabilities
   - Audit trail for all trades

6. **Performance Testing**:
   - Load testing for high-volume scenarios
   - Latency optimization
   - Database query optimization
   - Memory leak detection

7. **Gotify Integration**:
   - Critical alerts for:
     - System failures
     - Large losses
     - Unusual market conditions
     - API errors
   - Daily summary notifications

8. **Documentation**:
   - API documentation
   - System architecture diagrams
   - Runbooks for common issues
   - Strategy documentation

Ensure all tests can run in Docker containers.
```

---

## Task 8: Deployment & Operations

### AI Agent Prompt:
```
Create deployment and operational setup:

1. **Docker Configuration**:
   - Multi-stage builds for smaller images
   - Docker Compose with:
     - Service dependencies
     - Health checks
     - Restart policies
     - Resource limits
   - Development vs production configs

2. **Environment Management**:
   - .env files for different environments
   - Secret management
   - Configuration validation
   - Environment-specific settings

3. **Deployment Scripts**:
   - One-command deployment
   - Database migration handling
   - Rollback procedures
   - Zero-downtime updates

4. **Backup Strategy**:
   - Database backups (daily)
   - Parquet file backups
   - Configuration backups
   - Automated restore testing

5. **Security**:
   - Audit logging
   - Rate limiting

6. **Operations Runbook**:
   - Start/stop procedures
   - Troubleshooting guide
   - Performance tuning
   - Disaster recovery

7. **Export Functionality**:
   - TradeNote compatible exports
   - Performance reports
   - Tax reporting data
   - Audit trails

8. **Maintenance Mode**:
   - Graceful shutdown
   - Read-only mode
   - Maintenance notifications
   - Service degradation handling

Include clear documentation for all operational procedures.
```

---

## Development Order

1. **Phase 1**: Project Foundation (Task 1)
2. **Phase 2**: Data Collection (Task 2)
3. **Phase 3**: Database & Storage Setup
4. **Phase 4**: Strategy Engine (Task 3)
5. **Phase 5**: Risk Manager (Task 4)
6. **Phase 6**: Trade Executor (Task 5)
7. **Phase 7**: Scheduler (Task 6)
8. **Phase 8**: Testing & Monitoring (Task 7)
9. **Phase 9**: Deployment (Task 8)

## Key Success Factors

- **Modularity**: Each service is independent and communicates via Redis
- **Performance**: Polars for all data operations, async where possible
- **Reliability**: Comprehensive error handling and recovery
- **Observability**: Extensive logging and monitoring
- **Flexibility**: Easy to modify strategies and parameters
- **Safety**: Multiple layers of risk management

## Additional Recommendations

1. Start with paper trading for at least 1 month
2. Implement gradual position sizing when going live
3. Keep detailed logs of all strategy changes
4. Regular backups of all data and configurations
5. Monitor system 24/7 during initial deployment
6. Have manual override capabilities for emergencies







alpaca-trade-api is deprecated. use alpaca-py
