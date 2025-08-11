# Automated Trading System

A comprehensive automated trading system built with Python, Docker, and PostgreSQL. This system provides data collection, strategy execution, risk management, and trade execution capabilities for algorithmic trading.

## 🏗️ Architecture

The system follows a microservices architecture with the following components:

- **Data Collector**: Fetches market data from TwelveData and FinViz
- **Strategy Engine**: Implements and executes trading strategies
- **Trade Executor**: Handles order execution via Alpaca API
- **Risk Manager**: Manages position sizing and portfolio risk
- **Scheduler**: Orchestrates all services and manages timing
- **Database**: PostgreSQL for persistent data storage
- **Cache**: Redis for real-time data and message queuing

## 📁 Project Structure

```
trading-system/
├── docker-compose.yml          # Main service orchestration
├── docker-compose.dev.yml      # Development overrides
├── docker-compose.prod.yml     # Production overrides
├── Dockerfile.base             # Base Docker image
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── Makefile                   # Build and management commands
├── README.md                  # This file
├── shared/                    # Shared code across services
│   ├── __init__.py
│   ├── models.py              # Pydantic data models
│   ├── config.py              # Configuration management
│   └── utils.py               # Shared utilities
├── services/                  # Microservices
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
├── data/                      # Data storage
│   ├── parquet/               # Market data files
│   └── logs/                  # Application logs
└── scripts/                   # Utility scripts
    ├── setup.sh               # System setup script
    └── backtest.py            # Backtesting script
```

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Python 3.9+
- Make
- Git

### 1. Clone and Setup

```bash
git clone <repository-url>
cd full-ai-trader
make setup
```

### 2. Configure Environment

Edit the `.env` file with your API keys and configuration:

```bash
nano .env
```

**Required API Keys:**
- Alpaca API Key & Secret (for trading)
- TwelveData API Key (for market data)
- FinViz API Key (for screening - optional)
- Gotify URL & Token (for notifications - optional)

### 3. Build and Start Services

```bash
# Build all Docker images
make build

# Initialize database
make init-db

# Start all services
make start

# Check service health
make health
```

### 4. Verify Installation

```bash
# Check service status
make status

# View logs
make logs

# Run tests
make test
```

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Trading API (Required)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Market Data (Required)
TWELVE_DATA_API_KEY=your_twelvedata_api_key

# Database
DB_PASSWORD=your_secure_password

# Risk Management
RISK_MAX_POSITION_SIZE=0.05        # 5% max position size
RISK_MAX_PORTFOLIO_RISK=0.02       # 2% max portfolio risk
RISK_DRAWDOWN_LIMIT=0.15          # 15% max drawdown

# Trading Hours
TRADING_START_TIME=09:30
TRADING_END_TIME=16:00
TRADING_TIMEZONE=America/New_York
```

### Risk Parameters

The system includes comprehensive risk management:

- **Position Sizing**: Maximum position size as percentage of portfolio
- **Portfolio Risk**: Maximum risk exposure across all positions
- **Drawdown Limits**: Automatic shutdown on excessive losses
- **Daily Trade Limits**: Maximum number of trades per day
- **Correlation Limits**: Maximum correlation between positions

## 🎯 Usage

### Starting the System

```bash
# Development mode (with hot reload)
make dev

# Production mode
make prod

# Debug mode (verbose logging)
make debug
```

### Managing Services

```bash
# Start/stop individual services
docker-compose up -d data_collector
docker-compose stop strategy_engine

# View service logs
make logs-service SERVICE=data_collector

# Get shell access
make shell SERVICE=strategy_engine

# Database shell
make shell-db

# Redis shell
make shell-redis
```

### Running Backtests

```bash
# Run backtest for moving average strategy
make backtest STRATEGY=moving_average SYMBOL=AAPL

# Or use the script directly
python scripts/backtest.py --strategy moving_average --symbol AAPL --start 2023-01-01 --end 2023-12-31
```

### Monitoring

```bash
# Check system health
make health

# View real-time metrics
make monitor

# System statistics
make stats
```

## 📊 API Endpoints

Each service exposes REST API endpoints:

### Data Collector (Port 8001)
- `GET /health` - Health check
- `GET /status` - Service status
- `POST /collect/market-data` - Trigger data collection
- `GET /data/symbols` - Get tracked symbols
- `GET /data/latest/{symbol}` - Get latest data for symbol

### Strategy Engine (Port 8002)
- `GET /health` - Health check
- `GET /strategies` - List available strategies
- `POST /signals/generate` - Generate trading signals

### Risk Manager (Port 8003)
- `GET /health` - Health check
- `GET /risk/portfolio` - Portfolio risk assessment
- `POST /risk/validate` - Validate trade request

### Trade Executor (Port 8004)
- `GET /health` - Health check
- `POST /orders` - Submit order
- `GET /orders/{id}` - Get order status
- `GET /positions` - Get current positions

### Scheduler (Port 8005)
- `GET /health` - Health check
- `GET /jobs` - List scheduled jobs
- `POST /jobs/{job_id}/trigger` - Manually trigger job

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run with coverage
make test-coverage
```

### Test Structure

Each service has its own test suite:

```
services/{service}/tests/
├── test_main.py              # Main application tests
├── test_models.py            # Model validation tests
├── test_api.py               # API endpoint tests
└── integration/              # Integration tests
```

## 🔒 Security

### API Keys and Secrets

- Store all API keys in `.env` file
- Never commit `.env` to version control
- Use paper trading for testing (`ALPACA_PAPER_TRADING=true`)
- Regularly rotate API keys

### Network Security

- Services communicate via internal Docker network
- Only necessary ports are exposed
- Database and Redis are not directly accessible from outside

### Data Protection

- Database connections use encrypted passwords
- All financial data is stored with proper decimal precision
- Logs do not contain sensitive information

## 📈 Strategies

The system supports multiple trading strategies:

### Moving Average Crossover
- **Description**: Generates signals based on moving average crossovers
- **Parameters**: Short window (default: 20), Long window (default: 50)
- **Signals**: BUY on bullish crossover, SELL on bearish crossover

### Adding Custom Strategies

1. Create strategy class in `services/strategy_engine/src/strategies/`
2. Implement `generate_signal()` method
3. Register strategy in strategy engine
4. Add tests in `tests/` directory

## 🛠️ Development

### Local Development Setup

```bash
# Install development dependencies
make install-dev-deps

# Format code
make format

# Run linting
make lint

# Type checking
make type-check
```

### Adding New Services

1. Create service directory structure:
   ```bash
   mkdir -p services/new_service/{src,tests}
   ```

2. Create Dockerfile, requirements.txt, and source code

3. Add service to `docker-compose.yml`

4. Update Makefile with service-specific commands

### Database Migrations

```bash
# Run migrations
make migrate

# Create backup before migrations
make backup
```

## 📊 Monitoring and Alerts

### Gotify Notifications

Configure Gotify for real-time alerts:

```bash
GOTIFY_URL=http://your-gotify-server:8080
GOTIFY_TOKEN=your_gotify_token
```

### Log Monitoring

Logs are stored in `data/logs/` with automatic rotation:

- `trading_system.log` - Main application log
- `{service_name}.log` - Service-specific logs
- `error.log` - Error-only log

### Health Monitoring

The system includes comprehensive health monitoring:

- Service health checks every 30 seconds
- Database connection monitoring
- Redis connectivity checks
- API rate limit monitoring
- Memory and CPU usage tracking

## 🎛️ Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup` | Initial project setup |
| `make build` | Build all Docker images |
| `make start` | Start all services |
| `make stop` | Stop all services |
| `make restart` | Restart all services |
| `make status` | Show service status |
| `make logs` | Show all service logs |
| `make health` | Check service health |
| `make test` | Run all tests |
| `make clean` | Clean up containers |
| `make backup` | Backup database |
| `make migrate` | Run database migrations |

## ⚠️ Important Notes

### Paper Trading First

Always test with paper trading before using real money:

```bash
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER_TRADING=true
```

### Risk Management

- Review risk parameters before live trading
- Start with small position sizes
- Monitor system performance continuously
- Have stop-loss mechanisms in place

### API Rate Limits

Be aware of API rate limits:

- TwelveData: 800 requests/minute (free tier)
- Alpaca: 200 requests/minute
- FinViz: Rate limited to prevent blocking

## 🐛 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check Docker daemon
   docker info
   
   # Check logs
   make logs
   
   # Verify configuration
   make check-env
   ```

2. **Database connection errors**
   ```bash
   # Check database status
   make shell-db
   
   # Reset database
   make reset-db
   ```

3. **API authentication errors**
   - Verify API keys in `.env`
   - Check API key permissions
   - Ensure correct base URLs

4. **Memory issues**
   ```bash
   # Check container resources
   docker stats
   
   # View system statistics
   make stats
   ```

### Getting Help

1. Check logs: `make logs`
2. Verify configuration: `make check-env`
3. Run health checks: `make health`
4. Check system resources: `make stats`

## 📜 License

This project is licensed under the MIT License. See LICENSE file for details.

## ⚡ Performance Optimization

### Production Recommendations

1. **Resource Allocation**
   - Allocate adequate CPU and memory in production
   - Use SSD storage for database and Redis
   - Monitor resource usage regularly

2. **Database Optimization**
   - Regular database maintenance and optimization
   - Proper indexing for query performance
   - Connection pooling for efficiency

3. **Caching Strategy**
   - Use Redis for frequently accessed data
   - Implement proper cache invalidation
   - Monitor cache hit rates

## 🔄 Updates and Maintenance

### Regular Maintenance

```bash
# Update Docker images
make update

# Clean old data
make clean-data

# Backup data
make backup-data

# System cleanup
make clean
```

### Version Updates

1. Backup system before updates
2. Test in development environment
3. Update dependencies gradually
4. Monitor system after updates

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Always test thoroughly before using with real money. Trading carries risk of financial loss.