# Real AI Trading Strategy Backtesting System

## Overview

This backtesting system runs the **actual production AI trading strategy** against historical market data to evaluate performance. Unlike simplified backtesting frameworks, this system simulates the complete trading pipeline using real components but fed with historical data instead of live market feeds.

## Key Features

- **Real Strategy Testing**: Uses the actual AI strategy that runs in production
- **Historical Data Integration**: Reads from parquet files containing real market data
- **Accelerated Execution**: Processes historical data quickly without real-time delays
- **Comprehensive Metrics**: Provides detailed performance analysis including Sharpe ratio, drawdown, win rate, etc.
- **Flexible Configuration**: Supports different timeframes, symbols, and strategy parameters
- **Integration Test Style**: Tests the complete system flow from data ingestion to trade execution

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Historical    │    │  Real AI        │    │   Backtest      │
│   Data Store    ├────┤  Strategy       ├────┤   Results       │
│  (Parquet)      │    │  (Production)   │    │   Analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## File Structure

```
backtesting/
├── README.md                    # This file
├── real_backtest_engine.py      # Main backtesting engine
├── backtest_models.py           # Standalone models (no config deps)
├── simple_data_store.py         # Simplified data access layer
├── backtest_engine.py           # Original backtesting framework
└── monte_carlo.py               # Monte Carlo simulation tools

examples/
└── run_simple_backtest.py       # Simple usage examples

scripts/
└── run_monthly_backtest.py      # Full-featured CLI script

tests/integration/
└── test_real_backtesting.py     # Comprehensive integration tests
```

## Quick Start

### 1. Basic Example

```python
import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from real_backtest_engine import RealBacktestEngine, RealBacktestConfig, BacktestMode
from backtest_models import TimeFrame

async def simple_backtest():
    config = RealBacktestConfig(
        start_date=datetime(2025, 8, 18, tzinfo=timezone.utc),
        end_date=datetime(2025, 8, 21, tzinfo=timezone.utc),
        initial_capital=Decimal('10000'),
        symbols_to_trade=["AAPL"],
        timeframe=TimeFrame.ONE_DAY
    )
    
    engine = RealBacktestEngine(config)
    results = await engine.run_backtest()
    
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Total Trades: {results.total_trades}")
    print(f"Win Rate: {results.win_rate:.1%}")

# Run the backtest
asyncio.run(simple_backtest())
```

### 2. Command Line Usage

```bash
# Run simple example
./venv/bin/python examples/run_simple_backtest.py

# Run backtest for previous month
./venv/bin/python scripts/run_monthly_backtest.py

# Run with custom parameters
./venv/bin/python scripts/run_monthly_backtest.py \
  --start-date 2025-07-01 \
  --end-date 2025-07-31 \
  --capital 50000 \
  --symbols AAPL,MSFT,GOOGL \
  --save-trades
```

## Configuration Options

### RealBacktestConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | datetime | Required | Backtest start date |
| `end_date` | datetime | Required | Backtest end date |
| `initial_capital` | Decimal | 100000 | Starting capital amount |
| `max_positions` | int | 10 | Maximum concurrent positions |
| `timeframe` | TimeFrame | ONE_DAY | Data timeframe to use |
| `symbols_to_trade` | List[str] | None | Specific symbols (None = use screener) |
| `enable_screener_data` | bool | True | Use screener results for symbol selection |
| `screener_types` | List[str] | ["momentum", "breakouts", "value_stocks"] | Screener types to use |
| `mode` | BacktestMode | FAST | Execution mode (FAST/REALTIME/DEBUG) |
| `commission_per_trade` | Decimal | 1.00 | Fixed commission per trade |
| `commission_percentage` | Decimal | 0.0005 | Percentage commission (0.05%) |
| `slippage_bps` | Decimal | 5.0 | Slippage in basis points |
| `max_position_size` | Decimal | 0.20 | Maximum position as % of portfolio |
| `data_path` | str | "data/parquet" | Path to parquet data files |

### Execution Modes

- **FAST**: Skip delays, process all data quickly (recommended)
- **REALTIME**: Simulate actual timing (slower, for testing)
- **DEBUG**: Add extra logging and validation

## Data Requirements

### Data Structure

The system expects data to be organized in parquet files:

```
data/parquet/
├── market_data/
│   ├── {SYMBOL}/
│   │   ├── 1day/
│   │   │   ├── 2025-08-01.parquet
│   │   │   ├── 2025-08-02.parquet
│   │   │   └── ...
│   │   ├── 1h/
│   │   └── 5min/
│   └── ...
└── screener_data/
    ├── momentum/
    ├── breakouts/
    └── value_stocks/
```

### Required Columns

**Market Data:**
- `symbol`: Stock ticker
- `timestamp`: Date/time
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `timeframe`: Data timeframe

**Screener Data:**
- `symbol`: Stock ticker
- `timestamp`: Date/time
- Additional screener-specific fields

## Results Analysis

### Performance Metrics

The system provides comprehensive performance analysis:

```python
class BacktestResults:
    # Performance metrics
    total_return: float              # Overall return percentage
    annualized_return: float         # Annualized return
    max_drawdown: float             # Maximum drawdown
    sharpe_ratio: float             # Risk-adjusted return
    sortino_ratio: float            # Downside risk-adjusted return
    calmar_ratio: float             # Return vs max drawdown
    
    # Trading metrics
    total_trades: int               # Number of trades executed
    win_rate: float                # Percentage of winning trades
    profit_factor: float           # Gross profit / gross loss
    average_win: float             # Average winning trade
    average_loss: float            # Average losing trade
    
    # AI Strategy metrics
    total_ai_calls: int            # Number of AI strategy calls
    signals_generated: int         # Signals generated by AI
    signals_executed: int          # Signals actually executed
    average_confidence: float      # Average AI confidence
```

### Detailed Trade Records

Each trade includes:
- Entry/exit dates and prices
- Profit/loss and percentage return
- Hold period in days
- AI reasoning and confidence
- Commission costs

## Advanced Usage

### Custom AI Strategy Parameters

```python
config = RealBacktestConfig(
    # ... other parameters ...
    ai_strategy_config={
        "confidence_threshold": 70.0,
        "risk_tolerance": 0.02,
        "max_correlation": 0.7
    }
)
```

### Multiple Timeframe Analysis

```python
timeframes = [TimeFrame.ONE_DAY, TimeFrame.ONE_HOUR, TimeFrame.FIFTEEN_MIN]

for tf in timeframes:
    config = RealBacktestConfig(
        timeframe=tf,
        # ... other parameters ...
    )
    results = await engine.run_backtest()
    print(f"{tf.value}: {results.total_return:.2%}")
```

### Screener-Based Symbol Selection

```python
config = RealBacktestConfig(
    symbols_to_trade=None,          # Don't specify symbols
    enable_screener_data=True,      # Enable screener
    screener_types=["momentum", "breakouts", "value_stocks"]
)
```

## Testing

### Run Integration Tests

```bash
# Run all backtesting tests
./venv/bin/python -m pytest tests/integration/test_real_backtesting.py -v

# Run specific test
./venv/bin/python -m pytest tests/integration/test_real_backtesting.py::TestRealBacktestEngine::test_simple_backtest_run -v
```

### Data Availability Check

```python
from simple_data_store import SimpleDataStore

store = SimpleDataStore("data/parquet")
symbols = store.get_available_symbols()
print(f"Available symbols: {len(symbols)}")

# Check date range for a symbol
date_range = store.get_date_range_for_symbol("AAPL", TimeFrame.ONE_DAY)
print(f"AAPL date range: {date_range}")
```

## Performance Considerations

### Memory Usage

- Loads data incrementally by date
- Processes one trading day at a time
- Memory usage scales with number of concurrent positions, not data size

### Execution Speed

- **FAST mode**: Processes months of data in seconds
- **1 month daily data**: ~1-5 seconds
- **1 week hourly data**: ~2-10 seconds

### Optimization Tips

1. **Use FAST mode** for routine backtesting
2. **Limit symbols** for faster execution
3. **Choose appropriate timeframe** (daily is fastest)
4. **Filter screener results** to reduce noise

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're using the virtual environment
./venv/bin/python your_script.py

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backtesting"
```

**No Data Found:**
```python
# Check data availability
from simple_data_store import SimpleDataStore
store = SimpleDataStore("data/parquet")
symbols = store.get_available_symbols()
print(f"Available: {symbols[:10]}")
```

**Configuration Errors:**
- Ensure dates are timezone-aware (`timezone.utc`)
- Use `Decimal` for monetary amounts
- Check that timeframe matches available data

**AI Strategy Errors:**
- System falls back to mock strategy if real AI strategy fails
- Check logs for AI strategy initialization messages

### Debug Mode

Enable detailed logging:

```python
config = RealBacktestConfig(
    mode=BacktestMode.DEBUG,
    # ... other parameters
)
```

## Extending the System

### Custom Strategies

To use a custom strategy instead of the AI strategy:

```python
class CustomStrategy:
    async def generate_signal(self, symbol, current_data, historical_data, market_context):
        # Your strategy logic here
        return Signal(
            symbol=symbol,
            action=SignalType.BUY,
            confidence=75.0,
            position_size=0.1
        )

# Replace in RealBacktestEngine
engine.ai_strategy = CustomStrategy()
```

### Custom Metrics

Add custom performance metrics in the results generation:

```python
# In _generate_results method
custom_metrics = {
    "max_consecutive_wins": self._calculate_consecutive_wins(),
    "volatility_adjusted_return": self._calculate_vol_adjusted_return()
}
```

## Support and Development

### Logging

Enable comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Contributing

1. Run tests before submitting changes
2. Update documentation for new features
3. Follow existing code style and patterns
4. Test with different data sets and time periods

### Known Limitations

- Requires historical data in specific parquet format
- AI strategy fallback uses mock signals
- Limited to equity markets (stocks/ETFs)
- Assumes no overnight gaps or halts

## License and Disclaimer

This backtesting system is for research and educational purposes. Past performance does not guarantee future results. Always validate strategies with paper trading before deploying with real capital.