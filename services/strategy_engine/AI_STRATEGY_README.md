# AI Strategy Engine - LLM Market Intelligence Engine (LMIE)

## Overview

The AI Strategy Engine is a sophisticated trading strategy module that leverages Anthropic's Claude models (Opus, Sonnet, Haiku) to make intelligent trading decisions. It implements a multi-model consensus mechanism, comprehensive risk assessment, and adaptive market regime analysis.

## Architecture

### Core Components

```
ai_strategy/
├── ai_strategy.py          # Main AI strategy engine
├── ai_models.py           # Data models and database schema
├── ai_integration.py      # System integration layer
├── prompts.yaml          # AI prompt templates
└── tests/
    └── test_ai_strategy.py  # Comprehensive test suite
```

### Key Features

1. **Multi-Model Consensus**: Queries multiple Claude models and builds consensus
2. **Cost Optimization**: Intelligent model selection based on decision importance
3. **Response Caching**: Reduces API costs by caching similar queries
4. **Market Regime Analysis**: Adapts strategy based on market conditions
5. **Performance Tracking**: Monitors AI decision accuracy and profitability
6. **Risk Management**: Comprehensive risk assessment for each trade

## Installation

### Prerequisites

```bash
# Required environment variables
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export DATABASE_URL="postgresql://user:password@localhost/trading"
export REDIS_URL="redis://localhost:6379"
```

### Dependencies

```bash
pip install -r services/strategy_engine/requirements.txt
```

Key dependencies:
- `anthropic>=0.19.1` - Claude API client
- `polars>=0.20.0` - High-performance dataframes
- `sqlalchemy>=2.0.23` - Database ORM
- `redis>=5.0.1` - Pub/sub and caching
- `tenacity>=8.2.3` - Retry logic

## Configuration

### Prompt Configuration (`config/ai_strategy/prompts.yaml`)

```yaml
prompts:
  master_analyst:
    version: "1.0"
    model_preference: "claude-3-opus"
    max_tokens: 2000
    temperature: 0.3
    template: |
      # Your prompt template here

models:
  claude-3-opus:
    cost_per_million_input_tokens: 15.0
    cost_per_million_output_tokens: 75.0
    
cost_management:
  daily_limit_usd: 5.0
  monthly_limit_usd: 100.0
  cache_ttl_seconds: 300
```

### Strategy Parameters

```python
config = {
    'min_confidence': 60,           # Minimum confidence for signals
    'max_positions': 10,            # Maximum concurrent positions
    'max_position_size': 0.1,       # Maximum position size (10% of portfolio)
    'daily_loss_limit': -1000,      # Daily loss limit in USD
    'consensus_min_models': 3,      # Minimum models for consensus
}
```

## Usage

### Basic Integration

```python
from ai_integration import AIStrategyIntegration
import redis.asyncio as redis

# Initialize Redis
redis_client = await redis.from_url('redis://localhost:6379')

# Configuration
config = {
    'min_confidence': 60,
    'max_positions': 10,
    'anthropic_api_key': 'your-key'
}

# Create integration
integration = AIStrategyIntegration(
    redis_client=redis_client,
    db_connection_string='postgresql://localhost/trading',
    config=config
)

# Initialize and start
await integration.initialize()

# Get status
status = await integration.get_strategy_status()
print(f"Active positions: {status['active_positions']}")
```

### Direct Strategy Usage

```python
from ai_strategy import AIStrategyEngine
from base_strategy import StrategyConfig, StrategyMode

# Configure strategy
config = StrategyConfig(
    name="ai_strategy",
    mode=StrategyMode.LONG_SHORT,
    parameters={'anthropic_api_key': 'your-key'}
)

# Create strategy
strategy = AIStrategyEngine(config)
strategy.initialize()

# Analyze ticker
signal = await strategy.analyze("AAPL", price_data)
print(f"Decision: {signal.action} @ {signal.confidence}% confidence")
```

## AI Models and Prompts

### Model Selection Strategy

| Model | Cost | Speed | Use Case |
|-------|------|-------|----------|
| **Haiku** | $0.25/1M tokens | Fast | Initial screening, momentum detection |
| **Sonnet** | $3/1M tokens | Medium | Risk assessment, exit strategies |
| **Opus** | $15/1M tokens | Slower | High-conviction trades, complex analysis |

### Prompt Types

1. **Master Analyst** - Primary trading decision
2. **Market Regime** - Market condition assessment
3. **Risk Assessment** - Position risk evaluation
4. **Momentum Catalyst** - Momentum opportunity detection
5. **Contrarian** - Devil's advocate perspective
6. **Exit Optimizer** - Exit strategy optimization

### Example AI Response

```json
{
    "decision": "BUY",
    "confidence": 75,
    "entry_price": 150.50,
    "stop_loss": 147.00,
    "take_profit": 156.00,
    "risk_reward_ratio": 2.5,
    "position_size_suggestion": 0.05,
    "reasoning": "Strong momentum with volume confirmation, breaking resistance",
    "key_risks": ["Market volatility", "Sector rotation risk"],
    "timeframe": "day_trade"
}
```

## Cost Management

### Estimated Daily Costs

| Activity | Queries/Day | Tokens/Query | Model | Cost |
|----------|-------------|--------------|-------|------|
| Screening | 1000 | 2K | Haiku | $0.50 |
| Analysis | 50 | 5K | Sonnet | $0.75 |
| Deep Analysis | 5 | 8K | Opus | $0.60 |
| **Total** | | | | **~$2-3/day** |

### Cost Optimization Strategies

1. **Caching**: 5-minute cache for similar queries
2. **Tiered Analysis**: Use cheaper models for initial screening
3. **Batch Processing**: Group related queries
4. **Token Optimization**: Compress data, use structured formats

## Performance Monitoring

### Database Schema

The system tracks all AI decisions and performance metrics:

```sql
-- AI Decisions
CREATE TABLE ai_decisions (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    ticker VARCHAR(10),
    decision VARCHAR(10),
    confidence FLOAT,
    models_used JSON,
    total_cost FLOAT,
    actual_outcome FLOAT
);

-- Performance Metrics
CREATE TABLE ai_performance_metrics (
    date TIMESTAMP,
    total_decisions INTEGER,
    accuracy_rate FLOAT,
    win_rate FLOAT,
    total_pnl FLOAT,
    total_api_cost FLOAT
);
```

### Performance Tracking

```python
# Get performance summary
async with integration.async_session() as session:
    metrics = await session.execute(
        "SELECT * FROM ai_performance_metrics WHERE date > NOW() - INTERVAL '7 days'"
    )
    
    for metric in metrics:
        print(f"Date: {metric.date}")
        print(f"Accuracy: {metric.accuracy_rate:.2%}")
        print(f"Win Rate: {metric.win_rate:.2%}")
        print(f"ROI on API Cost: ${metric.total_pnl / metric.total_api_cost:.2f}")
```

## Testing

### Run Tests

```bash
# Run all AI strategy tests
pytest tests/test_ai_strategy.py -v

# Run specific test categories
pytest tests/test_ai_strategy.py::TestAnthropicClient -v
pytest tests/test_ai_strategy.py::TestConsensusEngine -v
```

### Test Coverage

- Unit tests for all components
- Integration tests for Redis/DB interaction
- Performance tests for cost tracking
- Mock API responses for offline testing

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   ```python
   # Adjust rate limits in RateLimiter
   self.min_delay = {
       AIModel.OPUS: 2.0,  # Increase delay
   }
   ```

2. **High Costs**
   - Increase cache TTL
   - Use more Haiku, less Opus
   - Implement more aggressive filtering

3. **Low Confidence Signals**
   - Adjust prompt templates
   - Increase consensus threshold
   - Add more context data

### Monitoring

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor API costs
tracker = integration.ai_strategy.anthropic_client.cost_tracker
print(f"Daily cost: ${tracker.daily_costs:.2f}")
print(f"Monthly cost: ${tracker.monthly_costs:.2f}")
```

## Best Practices

### 1. Prompt Engineering

- Be specific and structured in prompts
- Request JSON responses for parsing
- Include relevant context without bloat
- Version and A/B test prompts

### 2. Risk Management

- Never bypass position limits
- Always set stop losses
- Monitor correlation between positions
- Track AI accuracy by market regime

### 3. Cost Control

- Cache aggressively but intelligently
- Use tiered model approach
- Monitor cost per profitable trade
- Set hard daily/monthly limits

### 4. Performance Optimization

- Batch similar analyses
- Pre-filter obvious non-opportunities
- Use parallel processing for multiple tickers
- Implement circuit breakers for poor performance

## Future Enhancements

### Planned Features

1. **Multi-Strategy Ensemble**: Combine AI with traditional strategies
2. **Sentiment Analysis**: Integrate news and social media
3. **Options Strategy**: AI-driven options trading
4. **Portfolio Optimization**: AI-based portfolio rebalancing
5. **Custom Training**: Fine-tune models on historical performance

### Research Areas

- Prompt optimization using genetic algorithms
- Multi-agent debate systems for decisions
- Reinforcement learning for prompt selection
- Real-time learning from trade outcomes

## Support

For issues or questions:
1. Check logs in `/var/log/ai_strategy/`
2. Review performance metrics in database
3. Verify API key and limits
4. Check Redis connectivity

## License

Proprietary - All rights reserved

---

*Last Updated: 2024*
*Version: 1.0.0*